// Address × lifetime memory map of one snapshot.
//
// The same model as Design/memory-inspector/Memory_V2.html and its
// memory-map-abnormality-guide.md: every on-chip Tile becomes a box whose x is
// its byte range inside a memory space and whose y is its static lifetime in
// dump source lines. Overlaps are then classified as reuse, view or conflict.
//
// Lifetimes are static source lines, not device cycles. Loops, branches and
// async pipelines are approximated conservatively, exactly as the guide says.

import { walkExpr, walkStmts } from './pyir.mjs';
import { parseType } from './analyze.mjs';

/**
 * Safe on-chip capacities, read from repo/pto/src/backend/common/soc.cpp
 * (Create910BSoC). Both runs in this explorer were compiled for Ascend910B /
 * a2a3. Vec is the 184 KB safe cap, not the 192 KB physical UB: PTO-ISA
 * reserves the top of the buffer (pto-isa#170).
 */
export const MEM_LIMITS_910B = {
  Vec: 184 * 1024,
  Mat: 512 * 1024,
  Left: 64 * 1024,
  Right: 64 * 1024,
  Acc: 128 * 1024,
};

const MEM_ORDER = ['Vec', 'Mat', 'Left', 'LeftScale', 'Right', 'RightScale', 'Acc', 'Bias'];
const COMPUTE_KINDS = new Set(['AIC', 'AIV', 'InCore']);

/** Names a statement reads, the way Python's ast sees `Name` loads. */
function stmtReads(s) {
  const roots = [];
  if (s.expr) roots.push(s.expr);
  if (s.exprs) roots.push(...s.exprs);
  if (s.iter) roots.push(s.iter);
  if (s.cond) roots.push(s.cond);
  if (s.ctx) roots.push(s.ctx);
  const out = new Set();
  for (const r of roots) walkExpr(r, (e) => { if (e.k === 'name') out.add(e.id); });
  return out;
}

function tileOf(typeNode) {
  const t = parseType(typeNode);
  if (!t || t.ctor !== 'Tile' || !t.memref || !t.memref.buffer || !t.space) return null;
  if (t.space === 'DDR') return null;
  if (typeof t.memref.offset !== 'number' || typeof t.memref.size !== 'number') return null;
  return t;
}

/**
 * One function's boxes. `f` is an analyzed function (analyzeProgram output);
 * only AIC / AIV / InCore kernels carry on-chip tiles worth mapping.
 */
export function functionMemoryMap(f, limits = MEM_LIMITS_910B) {
  if (!f || !COMPUTE_KINDS.has(f.kind)) return null;

  // name -> tile definition; a name rebound in two branches keeps the earliest
  // definition and later widens to the latest read.
  const defs = new Map();
  const define = (name, t, line, op) => {
    const cur = defs.get(name);
    if (cur) { cur.start = Math.min(cur.start, line); return; }
    defs.set(name, {
      name,
      space: t.space,
      base: t.memref.buffer,
      offset: t.memref.offset,
      size: t.memref.size,
      shape: Array.isArray(t.shape) ? t.shape : [],
      dtype: t.dtype || '',
      op: op ? op.replace(/^pl\./, '') : null,
      start: line,
      end: line,
    });
  };

  for (const p of f.params || []) {
    const t = p.ctor === 'Tile' && p.memref && p.space && p.space !== 'DDR' ? p : null;
    if (t && typeof t.memref.offset === 'number') define(p.name, t, f.line, 'param');
  }

  const lastRead = new Map();
  walkStmts(f.body || [], (s) => {
    if (s.kind === 'assign' && s.type && s.targets && s.targets.length === 1) {
      const t = tileOf(s.type);
      if (t) define(s.targets[0], t, s.line, s.op);
    }
    for (const n of stmtReads(s)) lastRead.set(n, Math.max(lastRead.get(n) || 0, s.line));
  });

  const tiles = [...defs.values()];
  for (const t of tiles) {
    const r = lastRead.get(t.name);
    if (r && r > t.end) t.end = r;
  }

  // Same physical slot + touching lifetimes = one value carried through SSA
  // aliases (phi, yield, loop-carried). Disjoint lifetimes stay apart: that is
  // the slot being reused, which is the point of the whole exercise.
  const slots = new Map();
  for (const t of tiles) {
    const key = t.space + '|' + t.base + '|' + t.offset + '|' + t.size;
    if (!slots.has(key)) slots.set(key, []);
    slots.get(key).push(t);
  }
  const boxes = [];
  for (const group of slots.values()) {
    group.sort((a, b) => a.start - b.start || a.end - b.end);
    let cur = null;
    for (const t of group) {
      if (cur && t.start <= cur.end) {
        cur.end = Math.max(cur.end, t.end);
        cur.aliases.push(t.name);
        continue;
      }
      cur = { ...t, aliases: [], view: false, conflict: false };
      boxes.push(cur);
    }
  }
  boxes.sort((a, b) => a.start - b.start || a.offset - b.offset);

  // Address and lifetime both overlapping: same base is a view onto the
  // parent, a different base is two allocations claiming the same bytes.
  for (let i = 0; i < boxes.length; i++) {
    const a = boxes[i];
    for (let j = i + 1; j < boxes.length; j++) {
      const b = boxes[j];
      if (a.space !== b.space) continue;
      if (!(a.offset < b.offset + b.size && b.offset < a.offset + a.size)) continue;
      if (!(a.start <= b.end && b.start <= a.end)) continue;
      if (a.base === b.base) {
        if (a.size < b.size) a.view = true;
        else if (b.size < a.size) b.view = true;
        else b.view = true;
      } else {
        a.conflict = true;
        b.conflict = true;
      }
    }
  }

  const bySpace = new Map();
  for (const b of boxes) {
    if (!bySpace.has(b.space)) bySpace.set(b.space, { space: b.space, hwm: 0, limit: limits[b.space] || 0, tiles: 0, bases: new Set() });
    const s = bySpace.get(b.space);
    s.hwm = Math.max(s.hwm, b.offset + b.size);
    s.tiles++;
    s.bases.add(b.base);
  }
  const spaces = [...bySpace.values()]
    .map((s) => ({ space: s.space, hwm: s.hwm, limit: s.limit, tiles: s.tiles, bases: s.bases.size }))
    .sort((x, y) => spaceRank(x.space) - spaceRank(y.space));

  for (const b of boxes) {
    const lim = limits[b.space];
    b.overflow = !!lim && b.offset + b.size > lim;
  }

  // Before AllocateMemoryAddr every base still sits at offset 0. Two or more
  // distinct bases parked at 0 in one space means addresses are not assigned
  // yet, and the overlaps above are placeholders rather than conflicts.
  const unplaced = spaces.filter((s) => {
    const at0 = new Set(boxes.filter((b) => b.space === s.space && b.offset === 0).map((b) => b.base));
    return at0.size >= 2;
  }).map((s) => s.space);
  for (const b of boxes) {
    b.pending = b.conflict && unplaced.includes(b.space);
    if (b.pending) b.conflict = false;
  }

  return {
    name: f.name,
    ftype: f.kind,
    src_start: f.decoLine,
    src_end: f.endLine,
    spaces,
    boxes,
    unplaced,
  };
}

function spaceRank(s) {
  const i = MEM_ORDER.indexOf(s);
  return i < 0 ? 99 : i;
}

/** Every compute function of an analyzed snapshot, in source order. */
export function programMemoryMap(an, limits = MEM_LIMITS_910B) {
  const functions = [];
  for (const f of an.functions) {
    const m = functionMemoryMap(f, limits);
    if (m && m.boxes.length) functions.push(m);
  }
  functions.sort((a, b) => a.src_start - b.src_start);
  return { functions, limits };
}

/**
 * Counts the UI badges are built from. A box is abnormal on conflict or
 * overflow; `pending` overlaps (bases not yet placed) are counted apart and
 * never as conflicts.
 */
export function memoryMapStats(fns) {
  const out = { functions: fns.length, boxes: 0, views: 0, conflicts: 0, overflow: 0, abnormal: 0, pending: 0, unplaced: 0 };
  for (const fn of fns) {
    if (fn.unplaced.length) out.unplaced++;
    for (const b of fn.boxes) {
      out.boxes++;
      if (b.pending) out.pending++;
      if (b.conflict) out.conflicts++;
      if (b.overflow) out.overflow++;
      if (b.conflict || b.overflow) out.abnormal++;
      else if (b.view) out.views++;
    }
  }
  return out;
}
