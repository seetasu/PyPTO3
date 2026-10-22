// Build the device performance lane: one decode run's on-chip execution, joined back to the
// compile-time parallel structure.
//
//   node Design/device-perf-lane/scripts/build_perf_lane.mjs
//
// Reads dfx_outputs/<rank>/d0/{merged_swimlane_*,deps,name_map,chip_swimlane_records}.json and
// writes data/perf-lane.json, then inlines that payload into index.html.

import { readFileSync, writeFileSync, mkdirSync, readdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "../../..");
const RUN_ID = "_jit_l3_decode_csa_20260903_010617";
const DFX = join(ROOT, "Data/DeepseekV4", RUN_ID, "dfx_outputs");
const COMPILE = join(ROOT, "Design/parallel-lane/data/parallel-lane.json");
const OUT = join(HERE, "../data/perf-lane.json");
const RANKS = ["rank0", "rank1"];

// Bytes per element. Only the dtypes this run actually uses need to be here; an unknown one
// falls back to 4 and is counted so the page can say the byte totals are approximate.
const DT = { FLOAT32: 4, FLOAT16: 2, BFLOAT16: 2, INT64: 8, INT32: 4, INT16: 2, INT8: 1, UINT8: 1, BOOL: 1 };
let unknownDtypes = new Set();

// FuncId and CoreId are packed into the event-hint string rather than carried as fields.
const HINT = /Task:(\S+?),\s*FuncId:\s*(\d+),\s*CoreId:\s*(\d+)/;

const sum = (xs, f) => xs.reduce((a, x) => a + f(x), 0);
const r1 = (x) => Math.round(x * 10) / 10;
const pct = (a, b) => (b ? a / b : 0);

function quantile(sorted, q) {
  if (!sorted.length) return 0;
  const i = (sorted.length - 1) * q;
  const lo = Math.floor(i);
  const hi = Math.ceil(i);
  return lo === hi ? sorted[lo] : sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
}

function loadRank(rank) {
  const dir = join(DFX, rank, "d0");
  const mergedName = readdirSync(dir).find((f) => f.startsWith("merged_swimlane"));
  const trace = JSON.parse(readFileSync(join(dir, mergedName), "utf8")).traceEvents;
  const deps = JSON.parse(readFileSync(join(dir, "deps.json"), "utf8"));
  const nameMap = JSON.parse(readFileSync(join(dir, "name_map.json"), "utf8")).callable_id_to_name;
  const chip = JSON.parse(readFileSync(join(dir, "chip_swimlane_records.json"), "utf8"));
  const program = JSON.parse(readFileSync(join(dir, "dispatch_program.json"), "utf8")).program;
  return { rank, dir, mergedName, trace, deps, nameMap, chip, program };
}

function buildRank(src, compile) {
  const { trace, deps, nameMap, chip } = src;
  const coreTypes = chip.metadata.core_types;
  const aicCores = coreTypes.filter((t) => t === "aic").length;
  const aivCores = coreTypes.filter((t) => t === "aiv").length;

  // ---- per-block execution records ---------------------------------------
  // The merged trace carries every block TWICE: a detailed record with the timing breakdown, and
  // a shadow record with only ts/dur. Keeping both doubles the block count and the slot time, and
  // (because the shadow has no kernel-duration-us) injects a fake 0 µs into every duration list.
  // Only the detailed records are real; their count matches chip_swimlane_records.aicore_tasks.
  const blocks = [];
  for (const e of trace) {
    if (e.cat !== "event" || e.ph !== "X") continue;
    if (e.args === undefined || e.args["kernel-duration-us"] === undefined) continue;
    const m = HINT.exec((e.args && e.args["event-hint"]) || "");
    if (!m) continue;
    blocks.push({
      tag: m[1], leaf: nameMap[m[2]], core: Number(m[3]), ct: coreTypes[Number(m[3])],
      taskId: String(e.args.taskId), ts: e.ts, dur: e.dur,
      compute: e.args["kernel-duration-us"] || 0, setup: e.args["local_setup_us"] || 0,
    });
  }
  const t0 = Math.min(...blocks.map((b) => b.ts));
  const t1 = Math.max(...blocks.map((b) => b.ts + b.dur));
  const windowUs = t1 - t0;

  // ---- deps: block_num, tensor bytes, names -------------------------------
  const depTask = new Map(deps.tasks.map((t) => [String(t.task_id), t]));
  const bytesOf = (t) => sum(t.args || [], (a) => {
    if (!DT[a.dtype]) unknownDtypes.add(a.dtype);
    return (a.shape || []).reduce((p, q) => p * q, 1) * (DT[a.dtype] || 4);
  });

  // ---- group blocks into one record per launch ---------------------------
  // A taskId is one launch. Two of them (the Group functions) carry an AIC and an AIV kernel,
  // so the grouping key is taskId+leaf and those launches produce two rows that share a span.
  const byKey = new Map();
  for (const b of blocks) {
    const key = `${b.taskId}|${b.leaf}`;
    let g = byKey.get(key);
    if (!g) { g = { taskId: b.taskId, tag: b.tag, leaf: b.leaf, ct: b.ct, items: [], cores: new Set() }; byKey.set(key, g); }
    g.items.push(b);
    g.cores.add(b.core);
  }

  const compileNode = new Map(compile.tree.nodes.map((n) => [n.name, n]));
  // A leaf kernel inherits the core_num of the spmd_submit that launched it. The chain is
  // wrapper -> callee, but a Group callee adds another hop (qk_pv_spmd -> qk_pv -> qk_pv_aic),
  // so walk all descendants rather than just the direct children.
  const widthOf = new Map();
  const childrenOf = new Map();
  compile.tree.edges.forEach((e) => {
    if (!childrenOf.has(e.from)) childrenOf.set(e.from, []);
    childrenOf.get(e.from).push(e.to);
  });
  compile.tree.edges.filter((e) => e.kind === "spmd_submit").forEach((e) => {
    const seen = new Set([e.to]);
    const stack = [...(childrenOf.get(e.to) || [])];
    while (stack.length) {
      const n = stack.pop();
      if (seen.has(n)) continue;
      seen.add(n);
      if (!widthOf.has(n)) widthOf.set(n, e);
      stack.push(...(childrenOf.get(n) || []));
    }
  });
  const bornOf = new Map(compile.functions.map((f) => [f.name, f]));

  const launches = [...byKey.values()].map((g) => {
    const durs = g.items.map((b) => b.compute).sort((a, b) => a - b);
    const mean = sum(durs, (d) => d) / durs.length;
    const max = durs[durs.length - 1];
    const dt = depTask.get(g.taskId);
    const compute = sum(g.items, (b) => b.compute);
    const setup = sum(g.items, (b) => b.setup);
    const slot = sum(g.items, (b) => b.dur);
    const start = Math.min(...g.items.map((b) => b.ts));
    const end = Math.max(...g.items.map((b) => b.ts + b.dur));
    const bytes = dt ? bytesOf(dt) : 0;
    const n = compileNode.get(g.leaf);
    const w = widthOf.get(g.leaf);
    const f = bornOf.get(g.leaf);
    return {
      id: `${g.taskId}|${g.leaf}`, taskId: g.taskId, tag: g.tag, leaf: g.leaf,
      ct: g.ct.toUpperCase(), coresUsed: g.cores.size,
      blockNum: dt ? dt.block_num : durs.length,
      startUs: r1(start - t0), endUs: r1(end - t0), spanUs: r1(end - start),
      computeUs: r1(compute), setupUs: r1(setup), slotUs: r1(slot),
      waitUs: r1(Math.max(0, slot - compute - setup)),
      blocks: { n: durs.length, min: r1(durs[0]), p50: r1(quantile(durs, 0.5)),
        p90: r1(quantile(durs, 0.9)), max: r1(max), mean: r1(mean),
        spread: mean ? Math.round((max / mean) * 100) / 100 : 0,
        zero: durs.filter((d) => d === 0).length,
        durs: durs.map(r1),
        // Per-block physical-core intervals power the device Task swimlane. Keep the measured
        // CoreId and normalized trace start/end; never infer lane placement from block order.
        items: g.items.map((b) => ({ core: b.core, startUs: r1(b.ts - t0), endUs: r1(b.ts + b.dur - t0),
          slotUs: r1(b.dur), computeUs: r1(b.compute), setupUs: r1(b.setup) })) },
      tailWasteUs: r1((max - mean) * durs.length),
      bytes, gbPerS: compute ? Math.round(bytes / 1e9 / (compute / 1e6)) : null,
      compile: {
        type: n ? n.type : null, level: n ? n.level : null,
        launchKind: w ? w.kind : "submit",
        coreNum: w ? w.coreNum : null, coreNumConst: w ? w.coreNumConst : null,
        via: w ? w.to : null, bornPass: f ? compile.stages[f.bornStage].index : null,
        bornPassName: f ? compile.stages[f.bornStage].pass : null,
        retypedAt: f && f.retypes ? compile.stages.find((s, i) => i > 0 && f.perStage[i] && f.perStage[i - 1] && f.perStage[i].type !== f.perStage[i - 1].type).index : null,
      },
    };
  });

  // Guard the dedup above: a single-kernel launch runs exactly block_num blocks, and a Group
  // launch runs block_num AIC blocks plus twice that many AIV blocks (one cube + two vector
  // cores per group). A mismatch means the trace was read wrong, not that the run was odd.
  const expected = (l) => {
    const twins = launches.filter((x) => x.taskId === l.taskId);
    if (twins.length === 1) return l.blockNum;
    return l.ct === "AIC" ? l.blockNum : l.blockNum * 2;
  };
  const offenders = launches.filter((l) => l.blocks.n !== expected(l));
  if (offenders.length) {
    throw new Error(`block count mismatch in ${src.rank}: `
      + offenders.slice(0, 3).map((l) => `${l.leaf} got ${l.blocks.n} want ${expected(l)}`).join("; "));
  }
  if (blocks.length !== chip.aicore_tasks.length) {
    throw new Error(`${src.rank}: ${blocks.length} detailed events but chip_swimlane_records has ${chip.aicore_tasks.length}`);
  }

  // ---- per-core occupancy -------------------------------------------------
  const cores = coreTypes.map((t, i) => ({ id: i, ct: t.toUpperCase(), slotUs: 0, computeUs: 0, setupUs: 0 }));
  blocks.forEach((b) => {
    const c = cores[b.core];
    c.slotUs += b.dur; c.computeUs += b.compute; c.setupUs += b.setup;
  });
  cores.forEach((c) => {
    c.slotUs = r1(c.slotUs); c.computeUs = r1(c.computeUs); c.setupUs = r1(c.setupUs);
    c.waitUs = r1(Math.max(0, c.slotUs - c.computeUs - c.setupUs));
  });

  // ---- critical path over the task DAG ------------------------------------
  // Longest chain by measured span. Tasks with no measured blocks (host-side waits that never
  // reach a core) contribute 0, so they stay on the path without inflating it.
  const spanOfTask = new Map();
  launches.forEach((l) => spanOfTask.set(l.taskId, Math.max(spanOfTask.get(l.taskId) || 0, l.spanUs)));
  const preds = new Map(deps.tasks.map((t) => [String(t.task_id), []]));
  deps.edges.forEach((e) => {
    const s = String(e.succ);
    if (preds.has(s)) preds.get(s).push(String(e.pred));
  });
  const memo = new Map();
  const earliestFinish = (id) => {
    if (memo.has(id)) return memo.get(id);
    memo.set(id, 0); // guards a cycle; none observed in this run
    const v = Math.max(0, ...(preds.get(id) || []).map(earliestFinish)) + (spanOfTask.get(id) || 0);
    memo.set(id, v);
    return v;
  };
  const allTaskIds = [...preds.keys()];
  allTaskIds.forEach(earliestFinish);
  let last = allTaskIds.reduce((a, b) => (earliestFinish(a) >= earliestFinish(b) ? a : b));
  const path = [];
  for (let cur = last; cur; ) {
    path.unshift(cur);
    const ps = preds.get(cur) || [];
    cur = ps.length ? ps.reduce((a, b) => (earliestFinish(a) >= earliestFinish(b) ? a : b)) : null;
  }
  const pathUs = earliestFinish(last);

  // Name every task on the path, including the ones that never reached a core.
  const leafNamesOf = (id) => {
    const t = depTask.get(id);
    if (!t) return "(未知)";
    const names = (t.kernel_ids || []).filter((i) => i >= 0).map((i) => nameMap[String(i)]).filter(Boolean);
    return names.length ? [...new Set(names)].join(" + ") : "(编排)";
  };
  const criticalPath = path.map((id) => {
    const rows = launches.filter((l) => l.taskId === id);
    return {
      taskId: id, name: leafNamesOf(id),
      spanUs: r1(spanOfTask.get(id) || 0),
      computeUs: r1(sum(rows, (l) => l.computeUs)),
      blockNum: depTask.get(id) ? depTask.get(id).block_num : null,
      onCore: rows.length > 0,
      ids: rows.map((l) => l.id),
    };
  });

  // ---- KPIs ---------------------------------------------------------------
  const compute = sum(launches, (l) => l.computeUs);
  const setup = sum(launches, (l) => l.setupUs);
  const slot = sum(launches, (l) => l.slotUs);
  const tailWaste = sum(launches, (l) => l.tailWasteUs);
  const capacity = windowUs * coreTypes.length;
  const waitOnPath = criticalPath.filter((p) => !p.onCore || p.blockNum === 1).reduce((a, p) => a + p.spanUs, 0);
  const byType = {};
  for (const t of ["AIC", "AIV"]) {
    const cs = cores.filter((c) => c.ct === t);
    byType[t] = {
      cores: cs.length,
      slotUs: r1(sum(cs, (c) => c.slotUs)), computeUs: r1(sum(cs, (c) => c.computeUs)),
      setupUs: r1(sum(cs, (c) => c.setupUs)), waitUs: r1(sum(cs, (c) => c.waitUs)),
      slotShare: pct(sum(cs, (c) => c.slotUs) / cs.length, windowUs),
      computeShare: pct(sum(cs, (c) => c.computeUs) / cs.length, windowUs),
    };
  }

  const kpi = {
    windowUs: r1(windowUs),
    criticalPathUs: r1(pathUs), criticalPathTasks: criticalPath.length, totalTasks: allTaskIds.length,
    waitOnPathUs: r1(waitOnPath), waitOnPathShare: pct(waitOnPath, pathUs),
    computeUs: r1(compute), setupUs: r1(setup), slotUs: r1(slot),
    tailWasteUs: r1(tailWaste), tailWasteRatio: pct(tailWaste, compute),
    capacityUs: r1(capacity), occupancy: pct(compute, capacity),
    setupShare: pct(setup, compute),
    aicCores, aivCores, totalCores: coreTypes.length,
    byType,
    bytesTotal: sum(launches, (l) => l.bytes),
  };

  return {
    rank: src.rank, program: src.program, source: `${src.rank}/d0/${src.mergedName}`,
    windowUs: r1(windowUs), kpi, launches, cores, criticalPath,
    edges: deps.edges.map((e) => ({ pred: String(e.pred), succ: String(e.succ), flags: e.flags || [] })),
    taskNames: Object.fromEntries(allTaskIds.map((id) => [id, leafNamesOf(id)])),
  };
}

const compile = JSON.parse(readFileSync(COMPILE, "utf8"));
const ranks = RANKS.map((r) => buildRank(loadRank(r), compile));

const doc = {
  schemaVersion: "perf-lane.v1",
  run: RUN_ID,
  sourceDirectory: `Data/DeepseekV4/${RUN_ID}/dfx_outputs`,
  compileSource: "Design/parallel-lane/data/parallel-lane.json",
  soc: { name: "Ascend910B", aicCores: 24, aivCores: 48, source: "repo/pto/src/backend/common/soc.cpp:144" },
  ranks,
  caveats: [
    "时间口径有三层：槽位占用（duration-us，核被该任务占住的时长）、真正计算（kernel-duration-us）、启动（local_setup_us）。三者之差记为「槽内等待」——数据只能给出等了多久，给不出等的是搬运、跨核同步还是上游依赖。",
    "关键路径按任务实测跨度做最长链，长度 5.55 ms 略大于窗口 4.83 ms：相邻任务在设备上有重叠，链式相加会略微高估。用它排序，不要当作绝对耗时。",
    "尾块浪费 =（本次提交最慢块 − 平均块）× 块数，是假设所有块同时启动的核时损失上界。块之间若有流水，实际损失小于这个数。",
    "搬运量按 deps.json 里每个 task 的 tensor shape × dtype 求和，是声明的张量大小，不是实际 GM 读写量（不计重复读、不计片上命中）。等效带宽 = 字节 ÷ 实测算时，只能用于排序。",
    "编译期一侧来自 Design/parallel-lane 的 pass dump 解析；64 个叶子 kernel 与设备 name_map 完全对应，86 个 task 的 block_num 与编译期 core_num 全部一致。",
    "本 run 只有一次 decode 的 trace，没有基线可比；所有「多/少」都是同一次运行内部的横比。",
  ],
};

mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, `${JSON.stringify(doc)}\n`, "utf8");

// Inline the payload into the page so it also opens straight off disk: a relative fetch fails
// under file:// and under the data: URL the desktop file preview uses.
const PAGE = join(HERE, "../index.html");
const TAG = '<script type="application/json" id="lane-data">';
const TAG_END = "</scr" + "ipt>";
function embed(json) {
  const page = readFileSync(PAGE, "utf8");
  const a = page.indexOf(TAG);
  if (a < 0) throw new Error(`${TAG} not found in ${PAGE}`);
  const b = page.indexOf(TAG_END, a);
  if (b < 0) throw new Error(`unterminated data tag in ${PAGE}`);
  writeFileSync(PAGE, page.slice(0, a + TAG.length) + json.replace(/</g, "\\u003c") + page.slice(b), "utf8");
}
embed(JSON.stringify(doc));

if (unknownDtypes.size) console.log(`WARN 未知 dtype（按 4 字节计）: ${[...unknownDtypes].join(", ")}`);
ranks.forEach((r) => {
  const k = r.kpi;
  console.log(`\n${r.rank}  窗口 ${(k.windowUs / 1000).toFixed(2)} ms  launches=${r.launches.length}  tasks=${k.totalTasks}`);
  console.log(`  关键路径 ${(k.criticalPathUs / 1000).toFixed(2)} ms / ${k.criticalPathTasks} 任务   其中单块等待 ${(k.waitOnPathUs / 1000).toFixed(2)} ms (${(k.waitOnPathShare * 100).toFixed(0)}%)`);
  console.log(`  算 ${(k.computeUs / 1000).toFixed(1)} ms   setup ${(k.setupUs / 1000).toFixed(1)} ms (${(k.setupShare * 100).toFixed(1)}%)   尾块浪费 ${(k.tailWasteUs / 1000).toFixed(1)} ms (${(k.tailWasteRatio * 100).toFixed(0)}%)`);
  ["AIC", "AIV"].forEach((t) => {
    const b = k.byType[t];
    console.log(`  ${t} ${b.cores} 核  槽位 ${(b.slotUs / b.cores / 1000).toFixed(2)} ms/核 (${(b.slotShare * 100).toFixed(0)}%)  算 ${(b.computeUs / b.cores / 1000).toFixed(2)} ms (${(b.computeShare * 100).toFixed(0)}%)  等 ${(b.waitUs / b.cores / 1000).toFixed(2)} ms`);
  });
});
console.log(`\nwritten : ${OUT}`);
console.log(`inlined : ${PAGE}`);
