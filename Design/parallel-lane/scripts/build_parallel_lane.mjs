// Build the parallel-structure lane: how one program's functions get split, typed and bound to cores.
//
//   node Design/parallel-lane/scripts/build_parallel_lane.mjs
//
// Reads the pass dump directly (valid PyPTO DSL text) and writes data/parallel-lane.json.

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "../../..");
const RUN_ID = "_jit_l3_decode_csa_20260903_010617";
const RUN_DIR = join(ROOT, "Data/DeepseekV4", RUN_ID, "passes_dump");
const OUT = join(HERE, "../data/parallel-lane.json");

// The six passes the lane model names as emitters, plus the frontend baseline and the inline
// pass that collapses the hand-written helpers. Every other snapshot in this run carries an
// identical function set, so it would add a column and no information.
const STAGES = [
  [0, "frontend", "00_frontend.py", "基线"],
  [1, "InlineFunctions", "01_after_InlineFunctions.py", "规范化"],
  [8, "OutlineHierarchyScopes", "08_after_OutlineHierarchyScopes.py", "并行"],
  [9, "OutlineIncoreScopes", "09_after_OutlineIncoreScopes.py", "并行"],
  [10, "OutlineClusterScopes", "10_after_OutlineClusterScopes.py", "并行"],
  [22, "LowerAutoVectorSplit", "22_after_LowerAutoVectorSplit.py", "并行"],
  [23, "ExpandMixedKernel", "23_after_ExpandMixedKernel.py", "并行"],
  [25, "SplitVectorKernel", "25_after_SplitVectorKernel.py", "并行"],
];

// The die this run targets. The cube:vector ratio is what the kernel mix gets judged against.
const SOC = {
  name: "Ascend910B", aicCores: 24, aivCores: 48,
  source: "repo/pto/src/backend/common/soc.cpp:144",
};

// Level ordering is the on-chip hierarchy: host orchestrator down to a single core.
const LEVEL_ORDER = ["HOST", "CHIP", "CHIP_DIE", "CORE_GROUP", "AIC", "AIV"];
const TYPE_ORDER = ["Host", "Orchestration", "Spmd", "Group", "InCore", "AIC", "AIV", "Inline"];

// The host entry point carries level/role but no `type=`; name it rather than leave it untyped.
const typeOf = (args) => {
  const m = /FunctionType\.(\w+)/.exec(args);
  if (m) return m[1];
  return /Role\.Orchestrator/.test(args) ? "Host" : null;
};

const splitTop = (s) => {
  const out = [];
  let depth = 0;
  let cur = "";
  for (const ch of s) {
    if ("([{".includes(ch)) depth += 1;
    else if (")]}".includes(ch)) depth -= 1;
    if (ch === "," && depth === 0) {
      out.push(cur.trim());
      cur = "";
    } else cur += ch;
  }
  if (cur.trim()) out.push(cur.trim());
  return out;
};

// Read a keyword argument out of a call's argument text, respecting nesting.
function kwarg(argText, key) {
  for (const part of splitTop(argText)) {
    const m = new RegExp(`^${key}\\s*=\\s*([\\s\\S]+)$`).exec(part);
    if (m) return m[1].trim();
  }
  return null;
}

function matchParen(s, open) {
  let depth = 0;
  for (let i = open; i < s.length; i += 1) {
    if (s[i] === "(") depth += 1;
    else if (s[i] === ")" && --depth === 0) return i;
  }
  return -1;
}

const DECO = /^ {4}@pl\.function\((.*)\)\s*$/;
const DEF = /^ {4}def (\w+)\(/;

function parseSnapshot(file) {
  const lines = readFileSync(join(RUN_DIR, file), "utf8").split(/\r?\n/);
  const fns = new Map();
  let pending = null;
  let current = null;

  const close = (endLine) => {
    if (current) {
      current.bodyLines = endLine - current.defLine;
      current = null;
    }
  };

  lines.forEach((line, i) => {
    const d = DECO.exec(line);
    if (d) {
      close(i);
      pending = d[1];
      return;
    }
    const f = DEF.exec(line);
    if (!f) {
      if (current) current.body.push(line);
      return;
    }
    close(i);
    const args = pending || "";
    current = {
      name: f[1],
      type: typeOf(args),
      level: (/Level\.(\w+)/.exec(args) || [, null])[1],
      role: (/Role\.(\w+)/.exec(args) || [, null])[1],
      defLine: i + 1,
      body: [],
      hasSelf: /^ {4}def \w+\(self\b/.test(line),
    };
    fns.set(f[1], current);
    pending = null;
  });
  close(lines.length);

  // Call edges. `pl.submit(self.X, …)` and `pl.spmd_submit(self.X, …, core_num=…)` come from the
  // orchestrator; `self.X(…)` is the plain in-scope call an SPMD wrapper or Group body makes.
  const edges = [];
  fns.forEach((fn) => {
    fn.body.forEach((line, bi) => {
      const at = fn.defLine + bi + 1;
      for (const m of line.matchAll(/pl\.(spmd_submit|submit)\(/g)) {
        const open = m.index + m[0].length - 1;
        const end = matchParen(line, open);
        if (end < 0) continue;
        const argText = line.slice(open + 1, end);
        const target = (/^self\.(\w+)/.exec(splitTop(argText)[0] || "") || [, null])[1];
        if (!target) continue;
        const core = m[1] === "spmd_submit" ? kwarg(argText, "core_num") : null;
        edges.push({
          from: fn.name, to: target, kind: m[1], line: at, coreNum: core,
          coreNumConst: core !== null && /^\d+$/.test(core) ? Number(core) : null,
        });
      }
      for (const m of line.matchAll(/self\.(\w+)\(/g)) {
        edges.push({ from: fn.name, to: m[1], kind: "call", line: at, coreNum: null, coreNumConst: null });
      }
    });
  });
  fns.forEach((fn) => delete fn.body);
  return { fns, edges };
}

const snaps = STAGES.map(([index, pass, file, group]) => ({ index, pass, file, group, ...parseSnapshot(file) }));
const last = snaps[snaps.length - 1];

// Identity: the function name, inside one unbroken run of stages. A name that dies and comes back
// is NOT one function — two of the hand-written `Inline` helpers here get inlined away at pass 01
// and their names are later handed to freshly outlined `InCore` functions. Splitting on the gap
// keeps each subject a single real function instead of a splice of two.
const names = [...new Set(snaps.flatMap((s) => [...s.fns.keys()]))];

const functions = [];
names.forEach((name) => {
  const raw = snaps.map((s) => {
    const f = s.fns.get(name);
    return f ? { type: f.type, level: f.level, role: f.role, bodyLines: f.bodyLines, defLine: f.defLine } : null;
  });
  const runs = [];
  raw.forEach((cur, i) => {
    if (!cur) return;
    const open = runs[runs.length - 1];
    if (open && open.end === i - 1) open.end = i;
    else runs.push({ start: i, end: i });
  });
  runs.forEach((run, ri) => {
    const perStage = raw.map((cur, i) => (i >= run.start && i <= run.end ? cur : null));
    const retypes = perStage.reduce((a, cur, i) => {
      const prev = i ? perStage[i - 1] : null;
      return a + (prev && cur && prev.type !== cur.type ? 1 : 0);
    }, 0);
    functions.push({
      id: ri ? `${name}#${ri + 1}` : name,
      name,
      nameReused: runs.length > 1,
      incarnation: ri + 1,
      incarnations: runs.length,
      perStage,
      bornStage: run.start,
      lastStage: run.end,
      retypes,
      absorbed: run.end < snaps.length - 1,
      finalType: perStage[run.end].type,
      finalLevel: perStage[run.end].level,
    });
  });
});

// Call tree of the finished program, rooted at the orchestrator.
const nodes = [...last.fns.values()].map((f) => ({
  name: f.name, type: f.type, level: f.level, role: f.role, bodyLines: f.bodyLines,
}));
const outgoing = new Map();
const treeEdges = last.edges.filter((e) => last.fns.has(e.to));
treeEdges.forEach((e) => {
  if (!outgoing.has(e.from)) outgoing.set(e.from, []);
  outgoing.get(e.from).push(e);
});
// The root is whoever nobody submits to or calls — the host entry, above the chip orchestrator.
const incoming = new Set(treeEdges.map((e) => e.to));
const root = nodes.find((n) => !incoming.has(n.name));
const depthOf = new Map([[root.name, 0]]);
(function walk(n, d) {
  (outgoing.get(n) || []).forEach((e) => {
    if (depthOf.has(e.to)) return;
    depthOf.set(e.to, d + 1);
    walk(e.to, d + 1);
  });
})(root.name, 0);

const stageOut = snaps.map((s, i) => {
  const prev = i ? snaps[i - 1] : null;
  const typeMix = {};
  const levelMix = {};
  s.fns.forEach((f) => {
    const t = f.type || "untyped";
    typeMix[t] = (typeMix[t] || 0) + 1;
    const l = f.level || "—";
    levelMix[l] = (levelMix[l] || 0) + 1;
  });
  let born = null;
  let absorbed = null;
  let retyped = null;
  if (prev) {
    born = [...s.fns.keys()].filter((n) => !prev.fns.has(n)).length;
    absorbed = [...prev.fns.keys()].filter((n) => !s.fns.has(n)).length;
    retyped = [...s.fns.keys()].filter((n) => prev.fns.has(n) && prev.fns.get(n).type !== s.fns.get(n).type).length;
  }
  return {
    index: s.index, pass: s.pass, file: s.file, group: s.group,
    functionCount: s.fns.size, typeMix, levelMix, born, absorbed, retyped,
    submitCount: s.edges.filter((e) => e.kind !== "call").length,
    callCount: s.edges.filter((e) => e.kind === "call").length,
  };
});

// ---- KPIs -----------------------------------------------------------------
const finalMix = stageOut[stageOut.length - 1].typeMix;
const submits = treeEdges.filter((e) => e.kind !== "call");
const spmd = submits.filter((e) => e.kind === "spmd_submit");
const constWidths = spmd.filter((e) => e.coreNumConst !== null);
const leaves = nodes.filter((n) => !(outgoing.get(n.name) || []).length);

const kpi = {
  functionCount: last.fns.size,
  functionTrail: stageOut.map((s) => s.functionCount),
  aic: finalMix.AIC || 0,
  aiv: finalMix.AIV || 0,
  vecPerCube: (finalMix.AIV || 0) / (finalMix.AIC || 1),
  socVecPerCube: SOC.aivCores / SOC.aicCores,
  vecSkew: (finalMix.AIV || 0) / (finalMix.AIC || 1) / (SOC.aivCores / SOC.aicCores),
  spmdTasks: finalMix.Spmd || 0,
  submitSites: submits.length,
  spmdSites: spmd.length,
  staticWidthSites: constWidths.length,
  dynamicWidthSites: spmd.length - constWidths.length,
  maxStaticWidth: constWidths.length ? Math.max(...constWidths.map((e) => e.coreNumConst)) : null,
  treeDepth: Math.max(...depthOf.values()) + 1,
  leafCount: leaves.length,
  unreachable: nodes.filter((n) => !depthOf.has(n.name)).map((n) => n.name),
  levels: LEVEL_ORDER.filter((l) => nodes.some((n) => n.level === l)),
  subjectCount: functions.length,
  reusedNames: [...new Set(functions.filter((f) => f.nameReused).map((f) => f.name))],
};

const doc = {
  schemaVersion: "parallel-lane.v1",
  run: RUN_ID,
  program: "_jit_l3_decode_csa",
  sourceDirectory: `Data/DeepseekV4/${RUN_ID}/passes_dump`,
  typeOrder: TYPE_ORDER,
  levelOrder: LEVEL_ORDER,
  soc: SOC,
  kpi,
  stages: stageOut,
  functions,
  tree: { root: root.name, nodes: nodes.map((n) => ({ ...n, depth: depthOf.has(n.name) ? depthOf.get(n.name) : null })), edges: treeEdges },
  caveats: [
    "主体身份用函数名匹配。本 run 每个快照内函数名互不重复，pass 09 诞生的 64 个名字到最后一个快照一字未改，因此跨列连续性可靠；换一个 run 需重新核对。",
    "并行宽度 core_num 多数是动态符号表达式，编译期取不到具体核数，页面只统计常量宽度。",
    "ExpandMixedKernel 之后的核内拆分（PTOOutlineSIMTSections、VPTOSplitCVModule）在 PTOAS，不产出 pass dump，这里取不到。",
    "函数体行数是 dump 文本行数，不是指令条数，只能横比不能当代价。",
    "Cube:Vector 比的是函数个数，不是执行时间。配比偏离硬件只说明 vector 侧被切得更碎，是否真的失衡要看每个 kernel 的耗时，dump 里没有。",
  ],
};

mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, `${JSON.stringify(doc)}\n`, "utf8");

console.log(`stages=${stageOut.length} names=${names.length} subjects=${functions.length} treeNodes=${nodes.length} edges=${treeEdges.length}`);
stageOut.forEach((s) => console.log(
  `  ${String(s.index).padStart(2)} ${s.pass.padEnd(24)} fns=${String(s.functionCount).padStart(3)}`
  + ` born=${String(s.born === null ? "—" : s.born).padStart(3)}`
  + ` absorbed=${String(s.absorbed === null ? "—" : s.absorbed).padStart(3)}`
  + ` retyped=${String(s.retyped === null ? "—" : s.retyped).padStart(3)}`
  + `  ${Object.entries(s.typeMix).map(([k, v]) => `${k}=${v}`).join(" ")}`));
console.log("KPI:", JSON.stringify({
  fns: kpi.functionCount,
  trail: kpi.functionTrail.join("→"),
  aic: kpi.aic, aiv: kpi.aiv, ratio: `1:${kpi.vecPerCube.toFixed(2)} vs soc 1:${kpi.socVecPerCube}`, spmd: kpi.spmdTasks,
  submits: kpi.submitSites, depth: kpi.treeDepth, leaves: kpi.leafCount,
  width: `${kpi.staticWidthSites} const / ${kpi.dynamicWidthSites} dyn, max=${kpi.maxStaticWidth}`,
  unreachable: kpi.unreachable.length, reusedNames: kpi.reusedNames,
}));
console.log(`written: ${OUT}`);

// Inline the payload into the page as well. A relative fetch fails under file:// and under the
// data: URL the desktop file preview uses, so the page carries its own copy and opens from disk.
// The .json file stays the source of truth; both are rewritten together by this script.
const PAGE = join(HERE, "../index.html");
const TAG = '<script type="application/json" id="lane-data">';
const TAG_END = "</scr" + "ipt>";
function embed(json) {
  const page = readFileSync(PAGE, "utf8");
  const a = page.indexOf(TAG);
  if (a < 0) throw new Error(`${TAG} not found in ${PAGE}`);
  const b = page.indexOf(TAG_END, a);
  if (b < 0) throw new Error(`unterminated data tag in ${PAGE}`);
  // Minified: the page only needs it parseable, and a one-line payload keeps the HTML diff
  // readable. '<' only occurs inside JSON strings, so escaping it keeps the JSON valid and stops
  // a '</script' in the data from ending the tag early.
  const inline = JSON.stringify(JSON.parse(json)).replace(/</g, "\\u003c");
  writeFileSync(PAGE, page.slice(0, a + TAG.length) + inline + page.slice(b), "utf8");
}
embed(readFileSync(OUT, "utf8"));
console.log(`inlined : ${PAGE}`);
