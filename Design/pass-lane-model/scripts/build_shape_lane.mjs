// Extract the "data shape" lane for one callable from a PyPTO pass-dump run.
//
// Produces data/shape-lane.json conforming to schema/{lanes,subjects,events,manifest}.schema.json.
// Run from the repository root:  node Design/pass-lane-model/scripts/build_shape_lane.mjs
//
// The dumps are valid PyPTO DSL text. memory-pass-replay/scripts/build_data.py parses them with
// Python's `ast`; this script uses a bracket-matching text parser instead so it can run under node.

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "../../..");
const RUN_DIR = join(ROOT, "Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/passes_dump");
const OUT = join(HERE, "../data/shape-lane.json");
const RUN_ID = "_jit_l3_decode_csa_20260903_010617";
const CALLABLE = "csa_merge_pack_publish";

// The shape lane's active window: from the first tile-level IR to stride materialization.
const STAGES = [
  [11, "ConvertTensorToTileOps", "11_after_ConvertTensorToTileOps.py"],
  [13, "LowerCompositeOps", "13_after_LowerCompositeOps.py"],
  [14, "FlattenTileNdTo2D", "14_after_FlattenTileNdTo2D.py"],
  [15, "BlockNzTensorViews", "15_after_BlockNzTensorViews.py"],
  [16, "LegalizeTileCast", "16_after_LegalizeTileCast.py"],
  [17, "AutoTileMatmulL0", "17_after_AutoTileMatmulL0.py"],
  [18, "CanonicalizeTileSlice", "18_after_CanonicalizeTileSlice.py"],
  [19, "InferTileMemorySpace", "19_after_InferTileMemorySpace.py"],
  [21, "ResolveBackendOpLayouts", "21_after_ResolveBackendOpLayouts.py"],
  [32, "MaterializeTensorStrides", "32_after_MaterializeTensorStrides.py"],
];

/** Return the source lines of `def <name>(...)` through the end of its body. */
function callableLines(source, name) {
  const lines = source.split(/\r?\n/);
  const start = lines.findIndex((l) => l.trimStart().startsWith(`def ${name}(`));
  if (start < 0) return null;
  const indent = lines[start].length - lines[start].trimStart().length;
  let end = lines.length;
  for (let i = start + 1; i < lines.length; i += 1) {
    const line = lines[i];
    if (!line.trim()) continue;
    const ind = line.length - line.trimStart().length;
    if (ind <= indent && !line.trimStart().startsWith(")")) {
      end = i;
      break;
    }
  }
  return { lines: lines.slice(start, end), offset: start };
}

/** Match the bracket that opens at `open` and return the index of its partner. */
function matchBracket(text, open) {
  const pairs = { "[": "]", "(": ")" };
  const close = pairs[text[open]];
  let depth = 0;
  for (let i = open; i < text.length; i += 1) {
    if (text[i] === text[open]) depth += 1;
    else if (text[i] === close) {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Split on commas that sit at bracket depth zero. */
function splitTop(text) {
  const parts = [];
  let depth = 0;
  let current = "";
  for (const ch of text) {
    if ("([".includes(ch)) depth += 1;
    else if (")]".includes(ch)) depth -= 1;
    if (ch === "," && depth === 0) {
      parts.push(current.trim());
      current = "";
    } else current += ch;
  }
  if (current.trim()) parts.push(current.trim());
  return parts;
}

/** Parse one `name: pl.Tile[...] = rhs` declaration. */
function parseTileDecl(line) {
  const m = /^\s*([A-Za-z_][\w]*)\s*:\s*pl\.Tile\[/.exec(line);
  if (!m) return null;
  const open = line.indexOf("[", m[0].length - 1);
  const close = matchBracket(line, open);
  if (close < 0) return null;
  const args = splitTop(line.slice(open + 1, close));
  if (!args.length) return null;

  const shapeRaw = args[0];
  const shape = shapeRaw.startsWith("[") ? shapeRaw : `[${shapeRaw}]`;
  const dims = splitTop(shape.slice(1, -1)).filter(Boolean);

  const rest = args.slice(1);
  const dtype = (rest.find((a) => /^pl\.[A-Z]/.test(a)) || "").replace(/^pl\./, "");
  const spaceArg = rest.find((a) => a.startsWith("pl.Mem."));
  const viewArg = rest.find((a) => a.startsWith("pl.TileView("));

  const eq = line.indexOf("=", close);
  const rhs = eq < 0 ? "" : line.slice(eq + 1).trim();
  const opMatch = /^([\w.]+)\s*\(/.exec(rhs);

  return {
    name: m[1],
    shape,
    rank: dims.length,
    dtype,
    space: spaceArg ? spaceArg.replace("pl.Mem.", "") : null,
    hasMemRef: rest.some((a) => a.startsWith("pl.MemRef(")),
    view: viewArg ? summarizeView(viewArg) : null,
    op: opMatch ? opMatch[1] : rhs.slice(0, 40) || null,
    code: line.trim(),
  };
}

/** Keep only the fields of pl.TileView(...) that the lane renders. */
function summarizeView(arg) {
  const inner = arg.slice("pl.TileView(".length, -1);
  const out = {};
  for (const part of splitTop(inner)) {
    const kv = /^(\w+)\s*=\s*(.*)$/.exec(part);
    if (kv) out[kv[1]] = kv[2];
  }
  return out;
}

function readStage([number, passName, file]) {
  const source = readFileSync(join(RUN_DIR, file), "utf8");
  const block = callableLines(source, CALLABLE);
  if (!block) throw new Error(`${CALLABLE} not found in ${file}`);
  const tiles = new Map();
  block.lines.forEach((line, i) => {
    const decl = parseTileDecl(line);
    if (decl) tiles.set(decl.name, { ...decl, line: block.offset + i + 1 });
  });
  return { number, passName, file, tiles };
}

// ---------------------------------------------------------------------------

const stages = STAGES.map(readStage);

// A tile's dtype is part of its type annotation and cannot be mutated in place — a cast
// produces a new value. So an apparent dtype change proves the *name* was reused by a
// different temporary rather than that one tile changed. Counter-style names (`t__tile_7`)
// collide this way across snapshots. Identity is therefore name + dtype, and names that
// carry no semantic prefix are reported at lower confidence.
const AUTO_NAME = /^t__tile_\d+$/;
const identityKey = (t) => `${t.name}|${t.dtype}`;

const TRACKED = ["shape", "rank", "space", "hasMemRef"];
const EVENT_LABEL = {
  shape: "shape-changed",
  rank: "rank-changed",
  space: "space-assigned",
  hasMemRef: "memref-bound",
};

const keys = new Set();
const dtypesByName = new Map();
stages.forEach((s) =>
  s.tiles.forEach((t) => {
    keys.add(identityKey(t));
    if (!dtypesByName.has(t.name)) dtypesByName.set(t.name, new Set());
    dtypesByName.get(t.name).add(t.dtype);
  }),
);

const subjects = [];
const events = [];
const unresolved = {};
let eventSeq = 0;
const pushEvent = (e) => events.push({ id: `ev${String(++eventSeq).padStart(4, "0")}`, ...e });

for (const key of [...keys].sort()) {
  const [name, dtype] = key.split("|");
  const collided = dtypesByName.get(name).size > 1;
  const confidence = collided ? "heuristic" : AUTO_NAME.test(name) ? "name-match" : "exact";
  const seq = stages.map((s) => {
    const t = s.tiles.get(name);
    return t && t.dtype === dtype ? t : null;
  });
  const firstIdx = seq.findIndex(Boolean);
  const lastIdx = seq.reduce((acc, v, i) => (v ? i : acc), -1);
  const first = seq[firstIdx];

  const subjectId = `tile:${name}#${dtype}`;
  subjects.push({
    id: subjectId,
    laneIds: ["shape", "memory"],
    kind: "tile",
    label: name,
    identity: {
      name,
      anchorLine: first.line,
      firstStage: stages[firstIdx].number,
      identityConfidence: confidence,
      nameReused: collided,
    },
    lifespan: { from: stages[firstIdx].number, to: stages[lastIdx].number },
    // Per-stage values, aligned 1:1 with manifest.stages. The timeline must render the value
    // *at that stage*, not the final one — otherwise a changing subject reads as static.
    stageValues: seq.map((t, i) =>
      t
        ? { stage: stages[i].number, shape: t.shape, rank: t.rank, dtype: t.dtype, space: t.space, op: t.op, line: t.line }
        : null,
    ),
    current: {
      shape: seq[lastIdx].shape,
      rank: seq[lastIdx].rank,
      dtype: seq[lastIdx].dtype,
      space: seq[lastIdx].space,
      op: seq[lastIdx].op,
      code: seq[lastIdx].code,
    },
  });

  pushEvent({
    stage: stages[firstIdx].number,
    pass: stages[firstIdx].passName,
    laneId: "shape",
    subjectId: `tile:${name}`,
    kind: "created",
    to: { shape: first.shape, rank: first.rank, dtype: first.dtype, space: first.space },
    detail: first.op,
  });

  let prev = first;
  for (let i = firstIdx + 1; i < stages.length; i += 1) {
    const cur = seq[i];
    if (!cur) {
      if (prev) {
        pushEvent({
          stage: stages[i].number,
          pass: stages[i].passName,
          laneId: "shape",
          subjectId,
          kind: "removed",
          from: { shape: prev.shape, dtype: prev.dtype },
        });
        prev = null;
      }
      continue;
    }
    if (!prev) {
      pushEvent({
        stage: stages[i].number,
        pass: stages[i].passName,
        laneId: "shape",
        subjectId,
        kind: "created",
        to: { shape: cur.shape, rank: cur.rank, dtype: cur.dtype, space: cur.space },
        detail: cur.op,
      });
      prev = cur;
      continue;
    }
    // Change events require a trustworthy identity. For counter-style names the same label is
// reused by unrelated temporaries, so a "change" there is an artefact of the naming, not a
// transformation — count it instead of emitting it.
    if (confidence !== "exact") {
      for (const field of TRACKED) {
        if (String(prev[field]) !== String(cur[field])) unresolved[field] = (unresolved[field] || 0) + 1;
      }
      prev = cur;
      continue;
    }
    for (const field of TRACKED) {
      if (String(prev[field]) !== String(cur[field])) {
        pushEvent({
          stage: stages[i].number,
          pass: stages[i].passName,
          laneId: field === "hasMemRef" ? "memory" : "shape",
          subjectId,
          kind: EVENT_LABEL[field],
          field,
          from: { [field]: prev[field] },
          to: { [field]: cur[field] },
          detail: cur.op,
        });
      }
    }
    const prevView = JSON.stringify(prev.view);
    const curView = JSON.stringify(cur.view);
    if (prevView !== curView) {
      pushEvent({
        stage: stages[i].number,
        pass: stages[i].passName,
        laneId: "shape",
        subjectId,
        kind: prev.view ? "view-changed" : "view-materialized",
        field: "view",
        from: prev.view ? { view: prev.view } : undefined,
        to: { view: cur.view },
        detail: cur.op,
      });
    }
    prev = cur;
  }
}

// The aggregate view is the primary one: a distribution per snapshot needs no per-subject
// identity, so it stays valid even where the names cannot be tracked.
const histogram = (tiles, field) => {
  const out = {};
  tiles.forEach((t) => {
    const k = String(t[field]);
    out[k] = (out[k] || 0) + 1;
  });
  return Object.fromEntries(Object.entries(out).sort((a, b) => b[1] - a[1]));
};

const aggregates = stages.map((s) => {
  const tiles = [...s.tiles.values()];
  return {
    stage: s.number,
    pass: s.passName,
    tileCount: tiles.length,
    totalBytes: tiles.reduce((a, t) => a + (Number(t.bytes) || 0), 0),
    shape: histogram(tiles, "shape"),
    dtype: histogram(tiles, "dtype"),
    rank: histogram(tiles, "rank"),
    space: histogram(tiles, "space"),
    viewMaterialized: tiles.filter((t) => t.view).length,
    memrefBound: tiles.filter((t) => t.hasMemRef).length,
  };
});

const laneDefs = JSON.parse(readFileSync(join(HERE, "../schema/lanes.sample.json"), "utf8"));

const doc = {
  schemaVersion: "pass-lane-model.bundle.v1",
  manifest: {
    schemaVersion: "pass-lane-model.manifest.v1",
    scenarioId: `${RUN_ID}/${CALLABLE}/shape`,
    run: RUN_ID,
    callable: CALLABLE,
    sourceDirectory: "Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/passes_dump",
    activeLanes: ["shape"],
    partialLanes: ["memory"],
    stages: stages.map((s) => ({ index: s.number, pass: s.passName, file: s.file, subjectCount: s.tiles.size })),
    counts: {
      subjects: subjects.length,
      events: events.length,
      byConfidence: subjects.reduce((a, s) => {
        a[s.identity.identityConfidence] = (a[s.identity.identityConfidence] || 0) + 1;
        return a;
      }, {}),
      unresolvedChanges: unresolved,
    },
    caveats: [
      "聚合视图（按快照统计分布）不依赖逐对象身份，始终有效；逐对象时间线只对 identityConfidence=exact 的 subject 开放。",
      "t__tile_N 这类计数器名在快照之间被不同临时量复用，无法从 dump 文本恢复身份。dtype 不可变，因此同名不同 dtype 一定是重名；同名同 dtype 仍可能重名，这部分变化只计数不出事件（unresolvedChanges）。",
      "要让逐对象时间线覆盖全部 tile，dump 需要带上 Span，或改用进程内 instrument 而非文本快照。",
      "本 run 的 dump 共 52 个快照，pass 序列与当前 pass_manager.py 的 Default 策略不完全一致（含 BlockNzTensorViews、LegalizeTileCast、LowerPipelineToSlots 等）。",
      "memory 线此处只含 memref-bound 事件，完整内存线见 Design/memory-pass-replay。",
    ],
  },
  lanes: laneDefs.lanes,
  aggregates,
  subjects,
  events,
};

mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, `${JSON.stringify(doc, null, 1)}\n`, "utf8");

const byKind = {};
events.forEach((e) => (byKind[e.kind] = (byKind[e.kind] || 0) + 1));
const byStage = {};
events.forEach((e) => (byStage[e.stage] = (byStage[e.stage] || 0) + 1));
console.log(`stages=${stages.length} subjects=${subjects.length} events=${events.length}`);
console.log("by kind :", byKind);
console.log("by stage:", byStage);
console.log(`written  : ${OUT}`);

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
