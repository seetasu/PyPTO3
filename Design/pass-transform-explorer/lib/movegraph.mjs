// Where a function body went when a Pass moved it.
//
// Three things a Pass can do to function boundaries:
//
//   dissolve  `InlineFunctions` splices a callee's body into its callers and
//             deletes the callee.
//   extract   `OutlineIncoreScopes`, `ExpandMixedKernel` lift a region out of a
//             parent into a brand new function and leave a call behind.
//   wrap      `OutlineClusterScopes` creates a new function that only calls an
//             existing one - an SPMD shell, 1-4 statements. NOTHING moves.
//
// The third case is why this file classifies instead of assuming. Treated as an
// extraction, `OutlineClusterScopes` reads as "109 statements lifted out of
// decode_csa_test" when that function actually GREW by 3 and all 48 new
// functions are shells around bodies that never moved. The test that separates
// them is structural: a new function that calls a name which already existed is
// a wrapper around it, not a body lifted out of its caller. On the real dumps
// that splits 48/48 wrapper for `OutlineClusterScopes` against 60/60 genuine
// for `OutlineIncoreScopes`, with nothing ambiguous in between.
//
// dissolve and extract are the same shape - a body, a source, a destination, a
// statement weight - so one derivation serves both and the viewer draws one
// figure with the arrows reversed.
//
// The derivation is structural, from the call graph, not from diffing text:
//
//   dissolve  a removed function's body landed in the nearest caller that
//             survived. Walk up the BEFORE graph until a surviving name is hit.
//             Several callers means the body was copied into each.
//   extract   an added function's body came from whoever calls it AFTER, as
//             long as that caller already existed.
//
// It is an attribution, not a measurement, so `reconcile()` reports it against
// the destination's real statement growth and the caller shows both. Inlining
// drops the call statement and can fold constants, so the two agree closely
// rather than exactly - and a reader who can see the gap can judge it.

/** Caller lists keyed by callee, from one analyzed program. */
function callerIndex(an) {
  const by = new Map();
  for (const f of an.functions) {
    for (const c of f.calls) {
      if (!by.has(c.callee)) by.set(c.callee, []);
      by.get(c.callee).push({ from: f.name, line: c.line, via: c.via });
    }
  }
  return by;
}

/**
 * Nearest surviving ancestors of `name`, with a copy count each.
 *
 * `keep` decides what counts as a destination. The walk is depth-first through
 * callers that did not survive, so a chain (rms_norm -> decode_csa ->
 * decode_csa_test, where only the last survives) credits the end of the chain.
 * `seen` breaks recursive cycles rather than trusting the IR not to have any.
 */
function resolveHosts(name, callers, keep, seen) {
  if (seen.has(name)) return new Map();
  seen.add(name);
  const out = new Map();
  for (const c of callers.get(name) || []) {
    if (keep.has(c.from)) {
      out.set(c.from, (out.get(c.from) || 0) + 1);
    } else {
      for (const [k, v] of resolveHosts(c.from, callers, keep, seen)) {
        out.set(k, (out.get(k) || 0) + v);
      }
    }
  }
  seen.delete(name);
  return out;
}

/**
 * Build the migration for one Pass step.
 *
 * Returns `null` when no whole body moved, so the caller can fall back to its
 * ordinary rendering instead of drawing an empty figure.
 */
export function moveGraph(before, after) {
  const beforeNames = new Set(before.functions.map((f) => f.name));
  const afterNames = new Set(after.functions.map((f) => f.name));

  const removed = before.functions.filter((f) => !afterNames.has(f.name));
  const addedAll = after.functions.filter((f) => !beforeNames.has(f.name));
  if (!removed.length && !addedAll.length) return null;

  // Split the new functions: a shell around a name that already existed moved
  // nothing, and must not be counted as an extraction.
  const wraps = [];
  const added = [];
  for (const f of addedAll) {
    const around = f.calls.filter((c) => beforeNames.has(c.callee));
    if (around.length) {
      wraps.push({
        body: f.name,
        wrapped: around[0].callee,
        via: around[0].via,
        stmts: f.stmtCount,
        lines: f.srcLineCount,
        kind: f.kind,
        to: f.decoLine,
      });
    } else {
      added.push(f);
    }
  }

  // The dominant direction decides the figure. A Pass that both removes and
  // adds is drawn as whichever moved more statements; the other side still
  // shows up as an unattributed row rather than being dropped.
  const removedWeight = removed.reduce((n, f) => n + f.stmtCount, 0);
  const addedWeight = added.reduce((n, f) => n + f.stmtCount, 0);
  if (!removedWeight && !addedWeight) {
    // Only shells were created. Say exactly that instead of inventing a flow.
    return wraps.length ? { direction: 'wrap', moves: [], wraps, hosts: [] } : null;
  }
  const direction = removedWeight >= addedWeight ? 'dissolve' : 'extract';

  const moves = [];
  if (direction === 'dissolve') {
    const callers = callerIndex(before);
    for (const f of removed) {
      const hosts = resolveHosts(f.name, callers, afterNames, new Set());
      if (!hosts.size) {
        moves.push({ body: f.name, host: null, copies: 0, stmts: f.stmtCount, lines: f.srcLineCount, kind: f.kind, line: f.decoLine, side: 'before' });
        continue;
      }
      for (const [host, copies] of hosts) {
        // A dissolved body exists only in the BEFORE file.
        moves.push({ body: f.name, host, copies, stmts: f.stmtCount, lines: f.srcLineCount, kind: f.kind, line: f.decoLine, side: 'before' });
      }
    }
  } else {
    const callers = callerIndex(after);
    for (const f of added) {
      const hosts = resolveHosts(f.name, callers, beforeNames, new Set());
      if (!hosts.size) {
        moves.push({ body: f.name, host: null, copies: 0, stmts: f.stmtCount, lines: f.srcLineCount, kind: f.kind, line: f.decoLine, side: 'after' });
        continue;
      }
      for (const [host, copies] of hosts) {
        // An extracted body exists only in the AFTER file.
        moves.push({ body: f.name, host, copies, stmts: f.stmtCount, lines: f.srcLineCount, kind: f.kind, line: f.decoLine, side: 'after' });
      }
    }
  }

  if (!moves.some((m) => m.host)) return wraps.length ? { direction: 'wrap', moves: [], wraps, hosts: [] } : null;

  return {
    direction,
    moves,
    wraps,
    hosts: reconcile(direction, moves, before, after),
    bodies: byBody(moves),
    // Totals the viewer must not recompute from `moves`: one entry per
    // (body, host) pair means a body copied into two hosts appears twice, and
    // counting rows there reports 21 bodies where the function census - and the
    // 函数结构 card next to it - says 20.
    counts: {
      bodies: new Set(moves.map((m) => m.body)).size,
      hosts: new Set(moves.filter((m) => m.host).map((m) => m.host)).size,
      // Statements the bodies hold, counted once each...
      stmts: [...new Map(moves.map((m) => [m.body, m.stmts])).values()].reduce((n, v) => n + v, 0),
      // ...and counted again per copy, which is what actually gets written.
      stmtsWithCopies: moves.reduce((n, m) => n + m.stmts * m.copies, 0),
      duplicated: [...new Set(moves.map((m) => m.body))]
        .filter((b) => moves.filter((m) => m.body === b).reduce((n, m) => n + m.copies, 0) > 1).length,
      functionsBefore: before.functions.length,
      functionsAfter: after.functions.length,
    },
  };
}

/**
 * One entry per body, carrying every destination it reached.
 *
 * The figure draws a row per body, not per (body, host) pair: a body that was
 * copied into two callers is still one body, and giving it two rows both
 * inflates the count and makes the reader hunt for the duplicate name.
 */
function byBody(moves) {
  const out = new Map();
  for (const m of moves) {
    if (!out.has(m.body)) {
      out.set(m.body, {
        body: m.body,
        stmts: m.stmts,
        lines: m.lines,
        kind: m.kind,
        line: m.line || 0,
        side: m.side,
        to: [],
        copies: 0,
      });
    }
    const b = out.get(m.body);
    if (m.host) {
      b.to.push({ host: m.host, copies: m.copies });
      b.copies += m.copies;
    }
  }
  // Heaviest first: weight is what the figure is about.
  return [...out.values()].sort((a, b) => b.stmts * Math.max(b.copies, 1) - a.stmts * Math.max(a.copies, 1));
}

/**
 * Per-destination totals: what the attribution claims, against what the
 * snapshots actually measure. The gap is the honest part of this figure.
 */
function reconcile(direction, moves, before, after) {
  const totals = new Map();
  for (const m of moves) {
    if (!m.host) continue;
    if (!totals.has(m.host)) totals.set(m.host, { name: m.host, claimed: 0, bodies: 0, copies: 0 });
    const t = totals.get(m.host);
    t.claimed += m.stmts * m.copies;
    t.bodies += 1;
    t.copies += m.copies;
  }

  for (const t of totals.values()) {
    const wasIn = before.byName.get(t.name);
    const nowIn = after.byName.get(t.name);
    const was = wasIn ? wasIn.stmtCount : 0;
    const now = nowIn ? nowIn.stmtCount : 0;
    t.before = was;
    t.after = now;
    // A dissolve grows its host; an extract shrinks it. Either way the measured
    // move is the size of the change, so `measured` is comparable to `claimed`.
    t.measured = direction === 'dissolve' ? now - was : was - now;
    // A host survives the Pass, so it has a line on both sides. Name the one
    // this number came from - the two files number the same function very
    // differently, and a bare number silently matches the wrong gutter.
    t.line = (nowIn || wasIn || {}).decoLine || null;
    t.side = nowIn ? 'after' : 'before';
    t.kind = (nowIn || wasIn || {}).kind || null;
  }

  return [...totals.values()].sort((a, b) => b.claimed - a.claimed);
}
