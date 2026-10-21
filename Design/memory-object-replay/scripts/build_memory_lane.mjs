// Build the memory lane for one callable across all eight memory-related passes.
//
//   node Design/memory-object-replay/scripts/build_memory_lane.mjs
//
// Reads the pass dump directly (it is valid PyPTO DSL text) and writes data/memory-lane.json.

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "../../..");
const RUN_ID = "_jit_l3_decode_csa_20260903_010617";
const RUN_DIR = join(ROOT, "Data/DeepseekV4", RUN_ID, "passes_dump");
const OUT = join(HERE, "../data/memory-lane.json");
const CALLABLE = "csa_merge_pack_publish";

// Every pass on the memory lane, in pipeline order. The four that the earlier demo replayed
// are the only ones that move slots, but the other four are what sets the scene.
const STAGES = [
  [12, "OptimizeOrchTensors", "12_after_OptimizeOrchTensors.py"],
  [19, "InferTileMemorySpace", "19_after_InferTileMemorySpace.py"],
  [24, "InjectGMPipeBuffer", "24_after_InjectGMPipeBuffer.py"],
  [33, "InitMemRef", "33_after_InitMemRef.py"],
  [34, "MaterializeSemanticAliases", "34_after_MaterializeSemanticAliases.py"],
  [35, "MemoryReuse", "35_after_MemoryReuse.py"],
  [36, "AllocateMemoryAddr", "36_after_AllocateMemoryAddr.py"],
  [38, "FuseCreateAssembleToSlice", "38_after_FuseCreateAssembleToSlice.py"],
];

// Vec budget: repo/pto/src/backend/common/soc.cpp — 184KB safe, 192KB physical (pto-isa#170
// reserves ~8KB at the top of the UB).
const BUDGET = { space: "Vec", safeBytes: 184 * 1024, physicalBytes: 192 * 1024,
  source: "repo/pto/src/backend/common/soc.cpp:138" };

const DT = { FP32: 4, FP16: 2, BF16: 2, INT64: 8, INT32: 4, INT16: 2, INT8: 1, UINT8: 1, BOOL: 1, FP8: 1, HF8: 1 };

const matchBracket = (s, open) => {
  const close = { "[": "]", "(": ")" }[s[open]];
  let depth = 0;
  for (let i = open; i < s.length; i += 1) {
    if (s[i] === s[open]) depth += 1;
    else if (s[i] === close && --depth === 0) return i;
  }
  return -1;
};

const splitTop = (s) => {
  const out = []; let depth = 0, cur = "";
  for (const ch of s) {
    if ("([".includes(ch)) depth += 1;
    else if (")]".includes(ch)) depth -= 1;
    if (ch === "," && depth === 0) { out.push(cur.trim()); cur = ""; } else cur += ch;
  }
  if (cur.trim()) out.push(cur.trim());
  return out;
};

function callableBody(source) {
  const lines = source.split(/\r?\n/);
  const start = lines.findIndex((l) => l.trimStart().startsWith(`def ${CALLABLE}(`));
  if (start < 0) return null;
  const indent = lines[start].length - lines[start].trimStart().length;
  let end = lines.length;
  for (let i = start + 1; i < lines.length; i += 1) {
    const l = lines[i];
    if (!l.trim()) continue;
    const k = l.length - l.trimStart().length;
    if (k <= indent && !l.trimStart().startsWith(")")) { end = i; break; }
  }
  return { lines: lines.slice(start, end), offset: start };
}

function parseTile(line) {
  const m = /^\s*([A-Za-z_]\w*)\s*:\s*pl\.Tile\[/.exec(line);
  if (!m) return null;
  const open = line.indexOf("[", m[0].length - 1);
  const close = matchBracket(line, open);
  if (close < 0) return null;
  const args = splitTop(line.slice(open + 1, close));
  const shape = args[0].startsWith("[") ? args[0] : `[${args[0]}]`;
  const dims = splitTop(shape.slice(1, -1)).map((d) => Number(d)).filter((n) => Number.isFinite(n));
  const rest = args.slice(1);
  const dtype = (rest.find((a) => /^pl\.[A-Z]/.test(a) && !a.startsWith("pl.Mem") && !a.startsWith("pl.MemRef")) || "")
    .replace(/^pl\./, "");
  const spaceArg = rest.find((a) => a.startsWith("pl.Mem."));
  const refArg = rest.find((a) => a.startsWith("pl.MemRef("));
  let base = null, offset = null, refBytes = null;
  if (refArg) {
    const parts = splitTop(refArg.slice("pl.MemRef(".length, -1));
    base = parts[0];
    const off = /pl\.const\(\s*(-?\d+)/.exec(parts[1] || "");
    offset = off ? Number(off[1]) : null;
    refBytes = Number(parts[2]);
  }
  const eq = line.indexOf("=", close);
  const rhs = eq < 0 ? "" : line.slice(eq + 1).trim();
  const op = (/^([\w.]+)\s*\(/.exec(rhs) || [, rhs.slice(0, 32)])[1];
  const bytes = Number.isFinite(refBytes) ? refBytes
    : dims.length ? dims.reduce((a, b) => a * b, 1) * (DT[dtype] || 4) : null;
  return { name: m[1], shape, dtype, space: spaceArg ? spaceArg.replace("pl.Mem.", "") : null,
    base, offset, bytes, op, code: line.trim() };
}

const ALLOC = /^\s*([A-Za-z_]\w*)\s*:\s*pl\.Ptr\s*=\s*pl\.tile\.alloc\(\s*pl\.Mem\.(\w+)\s*,\s*(\d+)\s*\)/;

const stages = STAGES.map(([index, pass, file]) => {
  const body = callableBody(readFileSync(join(RUN_DIR, file), "utf8"));
  if (!body) throw new Error(`${CALLABLE} not found in ${file}`);
  const tiles = new Map(), allocations = [];
  body.lines.forEach((line, i) => {
    const a = ALLOC.exec(line);
    if (a) { allocations.push({ base: a[1], space: a[2], bytes: Number(a[3]), line: body.offset + i + 1 }); return; }
    const t = parseTile(line);
    if (t) tiles.set(t.name, { ...t, line: body.offset + i + 1 });
  });
  return { index, pass, file, tiles, allocations };
});

const names = [...new Set(stages.flatMap((s) => [...s.tiles.keys()]))].sort();

const tiles = names.map((name) => ({
  name,
  perStage: stages.map((s) => {
    const t = s.tiles.get(name);
    return t ? { base: t.base, offset: t.offset, space: t.space, shape: t.shape, dtype: t.dtype,
      bytes: t.bytes, op: t.op, line: t.line } : null;
  }),
  code: (stages.map((s) => s.tiles.get(name)).filter(Boolean).pop() || {}).code || null,
}));

const stageOut = stages.map((s, i) => {
  const prev = i ? stages[i - 1] : null;
  let moved = null;
  if (prev) {
    moved = 0;
    s.tiles.forEach((t, n) => {
      const p = prev.tiles.get(n);
      if (p && p.base && t.base && p.base !== t.base) moved += 1;
    });
  }
  const allocBytes = s.allocations.reduce((a, x) => a + x.bytes, 0);
  const withSpace = [...s.tiles.values()].filter((t) => t.space).length;
  return { index: s.index, pass: s.pass, file: s.file,
    tileCount: s.tiles.size, slotCount: s.allocations.length, allocBytes,
    movedCount: moved, spaceAssigned: withSpace,
    allocations: s.allocations.map((a) => ({ ...a,
      offset: ([...s.tiles.values()].find((t) => t.base === a.base) || {}).offset ?? null })) };
});

// ---- KPIs -----------------------------------------------------------------
const withSlots = stageOut.filter((s) => s.slotCount);
const peak = Math.max(...withSlots.map((s) => s.allocBytes));
const final = withSlots[withSlots.length - 1];
const finalAllocs = final.allocations.slice().sort((a, b) => (a.offset ?? 0) - (b.offset ?? 0));
const highWater = Math.max(...finalAllocs.map((a) => (a.offset ?? 0) + a.bytes));
const packed = finalAllocs.reduce((a, x) => a + x.bytes, 0);
const bySize = final.allocations.slice().sort((a, b) => b.bytes - a.bytes);
const top3 = bySize.slice(0, 3).reduce((a, x) => a + x.bytes, 0);

const kpi = {
  occupancy: final.allocBytes / BUDGET.safeBytes,
  headroomBytes: BUDGET.safeBytes - final.allocBytes,
  peakBytes: peak,
  finalBytes: final.allocBytes,
  reduction: 1 - final.allocBytes / peak,
  slotsBefore: withSlots[0].slotCount,
  slotsAfter: final.slotCount,
  highWater,
  fragmentationBytes: highWater - packed,
  fragmentation: (highWater - packed) / highWater,
  top3Share: top3 / packed,
  largestSlot: { base: bySize[0].base, bytes: bySize[0].bytes, share: bySize[0].bytes / packed },
  tileCount: names.length,
  tenantsPerSlot: names.length / final.slotCount,
  spaces: [...new Set(stageOut.flatMap((s) => s.allocations.map((a) => a.space)))],
};

const doc = {
  schemaVersion: "memory-lane.v1",
  run: RUN_ID, callable: CALLABLE,
  sourceDirectory: `Data/DeepseekV4/${RUN_ID}/passes_dump`,
  budget: BUDGET, kpi, stages: stageOut, tiles,
  caveats: [
    "容量为 tile.alloc 的静态求和，是编译期上界，不是运行时峰值，也不是性能比例。",
    "没有真实生命周期（dump 只有静态行号），因此无法给出复用的理论下界。",
    "pass 36 之后内存线进入 PTOAS（PlanMemory 等），不产出 pass dump，这里取不到。",
    "tile 跨快照按名字匹配。SSA 改名会被当作不同 tile；t__tile_N 这类计数器名还会被不同临时量复用，这些行的跨列连续性不可靠。",
  ],
};

mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, `${JSON.stringify(doc)}\n`, "utf8");

console.log(`stages=${stageOut.length} tiles=${names.length}`);
stageOut.forEach((s) => console.log(
  `  ${String(s.index).padStart(2)} ${s.pass.padEnd(28)} tiles=${String(s.tileCount).padStart(3)}` +
  ` slots=${String(s.slotCount).padStart(2)} bytes=${String(s.allocBytes).padStart(6)}` +
  ` moved=${s.movedCount ?? "—"}`));
console.log("KPI:", JSON.stringify({
  occupancy: `${(kpi.occupancy * 100).toFixed(1)}%`, headroom: kpi.headroomBytes,
  reduction: `${(kpi.reduction * 100).toFixed(1)}%`, frag: `${(kpi.fragmentation * 100).toFixed(2)}%`,
  top3: `${(kpi.top3Share * 100).toFixed(1)}%`, slots: `${kpi.slotsBefore}→${kpi.slotsAfter}`,
}));
console.log(`written: ${OUT}`);
