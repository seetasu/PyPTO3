/**
 * Build the Tuning Console dataset from one real on-device run dump.
 *
 * Source run: Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617
 *   dfx_outputs/rank{0,1}/d0/  merged swimlane trace, chip_swimlane_records, deps, name_map, host STRACE log
 *   report/perf_hints.log      compiler perf hints (PH001, PH-MR-001)
 *   passes_dump/               52 IR dumps
 *   distributed_meta.json      bound parameter shapes / dtypes
 *
 * Every number in data.js is read from those files.
 * Run:  node build-data.cjs
 */
'use strict';
const fs = require('fs');
const path = require('path');

const RUN = path.resolve(__dirname, '../../Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617');
const OUT = path.join(__dirname, 'data.js');
const rd = (p) => fs.readFileSync(path.join(RUN, p), 'utf8');
const rj = (p) => JSON.parse(rd(p));
const r2 = (n) => Math.round(n * 100) / 100;
const r3 = (n) => Math.round(n * 1000) / 1000;

/* ---------------------------------------------------------------- case */
const distMeta = rj('distributed_meta.json');
const nameMap = rj('dfx_outputs/rank0/d0/name_map.json');
const dispatch = rj('dfx_outputs/rank0/d0/dispatch_program.json');
const csr = rj('dfx_outputs/rank0/d0/chip_swimlane_records.json');
const deps = rj('dfx_outputs/rank0/d0/deps.json');

const binCtx = rj('next_levels/decode_csa_test/cache/binary_context.json');
const coreTypes = csr.metadata.core_types;

/* incore scope names as the compiler outlined them — the kernel inventory */
const incoreNames = Array.from(new Set(
  (fs.readFileSync(path.join(RUN, 'passes_dump/09_after_OutlineIncoreScopes.py'), 'utf8')
    .match(/^\s*def ([a-z_0-9]+)\(/gm) || []).map((s) => s.trim().replace(/^def /, '').replace(/\($/, ''))
)).sort();
const caseInfo = {
  program: dispatch.program,
  model: 'deepseek_v4_flash_dspark / decode_csa',
  backend: 'a2a3',
  ranks: ['rank0', 'rank1'],
  device: 'd0',
  clockHz: csr.metadata.clock_freq_hz,
  numCores: csr.metadata.num_cores,
  aicCount: coreTypes.filter((t) => t === 'aic').length,
  aivCount: coreTypes.filter((t) => t === 'aiv').length,
  threadsPerCore: Math.max.apply(null, csr.metadata.core_to_thread) + 1,
  callables: Object.keys(nameMap.callable_id_to_name).length,
  swimlaneLevel: csr.chip_swimlane_level,
  metaSchema: distMeta.schema,
  params: distMeta.params.map((p) => ({ name: p.name, dir: p.direction, shape: p.shape, dtype: p.dtype })),
  runDir: '_jit_l3_decode_csa_20260903_010617',
  capturedAt: '2026-09-03 01:06:17',
  toolchain: {
    platform: binCtx.platform,
    ptoIsaRevision: binCtx.pto_isa_revision,
    runtimeName: binCtx.runtime_name,
    runtimeRevision: binCtx.runtime_revision,
    schema: binCtx.schema,
  },
  incoreScopes: incoreNames,
  sourceRoot: '/data/w00949750/wzh_pypto_github/pypto/pypto-lib/models/deepseek_v4_flash_dspark',
  /* This dump carries no kernel -> source-file map: perf hints are anchored on
   * source locations, the IR keeps only outlined incore scope names. */
  hasKernelSourceMap: false,
};

/* ------------------------------------------------------------ end-to-end */
function hostSpans(file) {
  const out = {};
  for (const line of rd('dfx_outputs/' + file).split(/\r?\n/)) {
    if (!line.trim()) continue;
    const name = (line.match(/name=(\S+)/) || [])[1];
    const dur = +(line.match(/dur=(\d+)/) || [])[1];
    const inv = +(line.match(/inv=(\d+)/) || [])[1];
    const clk = /clk=dev/.test(line) ? 'device' : 'host';
    if (!name || !Number.isFinite(dur)) continue;
    (out[inv] = out[inv] || {})[name] = { us: r2(dur / 1000), clk: clk };
  }
  return out;
}
const e2e = {
  rank0: hostSpans('rank0/d0/host.2263908.log'),
  rank1: hostSpans('rank1/d0/host.2263922.log'),
};

const parseList = (s) => {
  const m = (s || '').match(/\[([^\]]*)\]/);
  return m && m[1].trim() ? m[1].split(',').map((x) => x.trim()) : [];
};
const sum = (a) => a.reduce((x, y) => x + y, 0);

/* ------------------------------------------------------- swimlane trace
 * One rank at a time: worker lanes, task aggregation, AICPU scheduler lanes,
 * the ready-queue counter, dependency flows and the measured critical path.  */
function buildRank(rank, traceFile) {
const trace = rj('dfx_outputs/' + rank + '/d0/' + traceFile).traceEvents;
const threadName = {};
const processName = {};
trace.filter((e) => e.cat === '__metadata').forEach((e) => {
  if (e.name === 'thread_name') threadName[e.pid + ':' + e.tid] = e.args.name;
  if (e.name === 'process_name') processName[e.pid] = e.args.name;
});

/* pid 4 = "Worker View": one event per block execution on a core.
 * pid 3 = "Scheduler View": the AICPU-side view of the same block
 *         (dispatch-time-us -> finish-time-us, aicpu-duration-us). */
const blockEvents = trace.filter((e) => e.cat === 'event' && e.ph === 'X'
  && e.pid === 4 && /Task:/.test(e.args['event-hint'] || '')
  && !!threadName['4:' + e.tid]);
const schedViewEvents = trace.filter((e) => e.cat === 'event' && e.ph === 'X'
  && e.pid === 3 && e.args['aicpu-duration-us'] !== undefined);
const SPAN = r2(Math.max.apply(null, blockEvents.map((e) => e.ts + e.dur)));

const laneNames = Array.from(new Set(blockEvents.map((e) => threadName['4:' + e.tid]).filter(Boolean)))
  .sort((a, b) => {
    const ka = a.indexOf('AIC') === 0 ? 0 : 1;
    const kb = b.indexOf('AIC') === 0 ? 0 : 1;
    return ka - kb || (+a.split('_')[1] - +b.split('_')[1]);
  });
const laneIdx = {};
laneNames.forEach((n, i) => { laneIdx[n] = i; });

const rankDeps = rj('dfx_outputs/' + rank + '/d0/deps.json');
const depById = {};
rankDeps.tasks.forEach((t) => { depById[t.task_id] = t; });
const taskMap = new Map();
for (const e of blockEvents) {
  const hint = e.args['event-hint'];
  const id = String(e.args.taskId);
  if (!taskMap.has(id)) {
    taskMap.set(id, {
      id: id,
      tag: (hint.match(/Task:(\S+?),/) || [])[1],
      funcId: +(hint.match(/FuncId:(-?\d+)/) || [0, -1])[1],
      rawName: e.name,
      blocks: [],
      fanoutRaw: e.args['fanout-hint'],
      faninRaw: e.args['fanin-hint'],
    });
  }
  const t = taskMap.get(id);
  t.blocks.push({
    lane: laneIdx[threadName['4:' + e.tid]],
    core: +(hint.match(/CoreId:(\d+)/) || [0, -1])[1],
    ts: e.ts,
    dur: e.dur,
    kdur: e.args['kernel-duration-us'] || 0,
    setup: e.args['local_setup_us'] || 0,
  });
}

/* AICPU-side (dispatch -> finish) view of each block, keyed by taskId */
const schedViewByTask = {};
for (const e of schedViewEvents) {
  const id = String(e.args.taskId);
  (schedViewByTask[id] = schedViewByTask[id] || []).push({
    dispatch: e.args['dispatch-time-us'],
    finish: e.args['finish-time-us'],
    aicpu: e.args['aicpu-duration-us'],
  });
}

const tasks = [];
taskMap.forEach((t) => {
  const b = t.blocks;
  const durs = b.map((x) => x.dur).sort((a, c) => a - c);
  const start = Math.min.apply(null, b.map((x) => x.ts));
  const end = Math.max.apply(null, b.map((x) => x.ts + x.dur));
  const lanes = Array.from(new Set(b.map((x) => laneNames[x.lane])));
  const kind = lanes.every((l) => l.indexOf('AIC') === 0) ? 'aic'
    : lanes.every((l) => l.indexOf('AIV') === 0) ? 'aiv' : 'mix';
  const dd = depById[t.id];
  const setupSum = sum(b.map((x) => x.setup));
  const sv = schedViewByTask[t.id] || [];
  const svAicpu = sv.map((x) => x.aicpu).sort((a, c) => a - c);
  const workerMean = sum(b.map((x) => x.dur)) / b.length;
  tasks.push({
    id: t.id, tag: t.tag, funcId: t.funcId,
    callable: nameMap.callable_id_to_name[String(t.funcId)] || t.rawName,
    rawName: t.rawName,
    ring: +(t.tag.match(/r(\d+)/) || [0, 0])[1],
    kind: kind,
    blockCount: b.length,
    coreCount: new Set(b.map((x) => x.core)).size,
    laneCount: lanes.length,
    blockNum: dd ? dd.block_num : null,
    scope: dd ? dd.scope : null,
    earlyDispatch: dd ? !!dd.early_dispatch : null,
    start: r2(start), end: r2(end), span: r2(end - start),
    durMin: r2(durs[0]), durMax: r2(durs[durs.length - 1]),
    durMed: r2(durs[Math.floor(durs.length / 2)]),
    durMean: r2(sum(durs) / durs.length),
    durP90: r2(durs[Math.min(durs.length - 1, Math.floor(durs.length * 0.9))]),
    busySum: r2(sum(durs)),
    kdurSum: r2(sum(b.map((x) => x.kdur))),
    setupSum: r2(setupSum),
    setupMean: r3(setupSum / b.length),
    setupShare: r3(setupSum / sum(durs)),
    imbalance: r2(durs[durs.length - 1] / Math.max(durs[Math.floor(durs.length / 2)], 1e-9)),
    svBlocks: sv.length,
    svAicpuMean: svAicpu.length ? r2(sum(svAicpu) / svAicpu.length) : null,
    svAicpuMax: svAicpu.length ? r2(svAicpu[svAicpu.length - 1]) : null,
    svOverhead: svAicpu.length ? r2(sum(svAicpu) / svAicpu.length - workerMean) : null,
    pred: parseList(t.faninRaw), succ: parseList(t.fanoutRaw),
    args: dd ? dd.args.map((a) => ({ idx: a.idx, type: a.type, dtype: a.dtype, shape: a.shape })) : [],
  });
});
tasks.sort((a, b) => a.start - b.start);
const taskIndex = {};
tasks.forEach((t, i) => { taskIndex[t.tag] = i; });

const laneBlocks = laneNames.map(() => []);
taskMap.forEach((t) => {
  const ti = taskIndex[t.tag];
  t.blocks.forEach((b) => laneBlocks[b.lane].push([r2(b.ts), r2(b.dur), ti]));
});
laneBlocks.forEach((a) => a.sort((x, y) => x[0] - y[0]));

const lanes = laneNames.map((name, i) => {
  const a = laneBlocks[i];
  let busy = 0, idle = 0, nGap = 0, maxGap = 0, cursor = null;
  for (const row of a) {
    busy += row[1];
    if (cursor !== null && row[0] - cursor > 0.5) {
      idle += row[0] - cursor; nGap++; maxGap = Math.max(maxGap, row[0] - cursor);
    }
    cursor = Math.max(cursor === null ? 0 : cursor, row[0] + row[1]);
  }
  return {
    name: name, kind: name.indexOf('AIC') === 0 ? 'aic' : 'aiv', coreId: +name.split('_')[1],
    blocks: a.length, busy: r2(busy), util: r2((busy / SPAN) * 100),
    idle: r2(idle), nGap: nGap, maxGap: r2(maxGap),
    first: a.length ? r2(a[0][0]) : null,
    last: a.length ? r2(a[a.length - 1][0] + a[a.length - 1][1]) : null,
  };
});

/* ------------------------------------------------------ AICPU scheduler */
const schedEv = trace.filter((e) => e.cat === 'scheduler' && e.ph === 'X');
const schedLanes = Array.from(new Set(schedEv.map((e) => threadName['2:' + e.tid] || 'Sched?'))).sort();
const schedPhases = {};
for (const e of schedEv) {
  const p = e.args.phase;
  const s = (schedPhases[p] = schedPhases[p] || { n: 0, us: 0, tasks: 0 });
  s.n++; s.us += e.dur; s.tasks += e.args.tasks_processed || 0;
}
Object.keys(schedPhases).forEach((k) => {
  const s = schedPhases[k];
  s.us = r2(s.us);
  s.usPerTask = s.tasks ? r3(s.us / s.tasks) : null;
});
const schedWindow = {
  lo: r2(Math.min.apply(null, schedEv.map((e) => e.ts))),
  hi: r2(Math.max.apply(null, schedEv.map((e) => e.ts + e.dur))),
};
const schedBusy = r2(sum(schedEv.map((e) => e.dur)));
const schedBlocks = schedLanes.map((ln) => schedEv
  .filter((e) => (threadName['2:' + e.tid] || 'Sched?') === ln)
  .map((e) => [r2(e.ts), r2(e.dur), e.args.phase, e.args.tasks_processed || 0])
  .sort((a, b) => a[0] - b[0]));
const schedLaneStats = schedLanes.map((ln, i) => {
  const busy = sum(schedBlocks[i].map((b) => b[1]));
  return {
    name: ln, events: schedBlocks[i].length, busy: r2(busy),
    util: r2((busy / (schedWindow.hi - schedWindow.lo)) * 100),
  };
});

const orchEv = trace.filter((e) => e.cat === 'orchestrator').map((e) => [r2(e.ts), r2(e.dur)]);
const orch = {
  count: orchEv.length,
  busy: r2(sum(orchEv.map((e) => e[1]))),
  lo: r2(Math.min.apply(null, orchEv.map((e) => e[0]))),
  hi: r2(Math.max.apply(null, orchEv.map((e) => e[0] + e[1]))),
  blocks: orchEv,
};

/* ready-but-undispatched queue counter */
const qEv = trace.filter((e) => e.cat === 'queue').sort((a, b) => a.ts - b.ts);
const readyQueue = qEv.map((e) => [r2(e.ts), e.args.AIC || 0, e.args.AIV || 0, e.args.MIX || 0]);
const KEYS = ['AIC', 'AIV', 'MIX'];
const rqStat = { window: 0, avg: {}, peak: {}, busyTime: {}, busyShare: {} };
KEYS.forEach((k) => { rqStat.avg[k] = 0; rqStat.peak[k] = 0; rqStat.busyTime[k] = 0; });
for (let i = 0; i < readyQueue.length - 1; i++) {
  const dt = readyQueue[i + 1][0] - readyQueue[i][0];
  rqStat.window += dt;
  KEYS.forEach((k, j) => {
    const v = readyQueue[i][j + 1];
    rqStat.avg[k] += v * dt;
    if (v > 0) rqStat.busyTime[k] += dt;
    rqStat.peak[k] = Math.max(rqStat.peak[k], v);
  });
}
KEYS.forEach((k) => {
  rqStat.avg[k] = r3(rqStat.avg[k] / rqStat.window);
  rqStat.busyShare[k] = r2((rqStat.busyTime[k] / rqStat.window) * 100);
  rqStat.busyTime[k] = r2(rqStat.busyTime[k]);
});
rqStat.window = r2(rqStat.window);

const flowCount = {};
trace.filter((e) => e.cat === 'flow').forEach((e) => { flowCount[e.name] = (flowCount[e.name] || 0) + 1; });
const hbPairs = trace.filter((e) => e.cat === 'flow' && e.name === 'hb_violation' && e.ph === 's').map((e) => {
  const fin = trace.find((f) => f.cat === 'flow' && f.name === 'hb_violation' && f.ph === 'f' && f.id === e.id);
  return {
    from: threadName['4:' + e.tid] || String(e.tid),
    to: fin ? (threadName['4:' + fin.tid] || String(fin.tid)) : null,
    ts: r2(e.ts), tsEnd: fin ? r2(fin.ts) : null,
    inputs: e.input_task_count, outputs: e.output_task_count,
  };
});

/* ------------------------------------------------------- critical path */
const byTag = {};
tasks.forEach((t) => { byTag[t.tag] = t; });
const best = {};
for (const t of tasks) {
  let bp = null, bl = 0;
  for (const p of t.pred) {
    const P = byTag[p];
    if (!P) continue;
    const c = (best[p] ? best[p].len : 0) + P.span;
    if (c > bl) { bl = c; bp = p; }
  }
  best[t.tag] = { len: bl, prev: bp };
}
let endTag = null, mx = -1;
for (const t of tasks) {
  const v = best[t.tag].len + t.span;
  if (v > mx) { mx = v; endTag = t.tag; }
}
const critTags = [];
for (let c = endTag; c; c = best[c].prev) critTags.unshift(c);
let cursor = null, posGap = 0, overlap = 0;
const critNodes = critTags.map((tag) => {
  const t = byTag[tag];
  const gap = cursor === null ? 0 : r2(t.start - cursor);
  if (gap > 0) posGap += gap; else overlap += -gap;
  cursor = t.end;
  return { tag: tag, gap: gap };
});
const critical = {
  tags: critTags,
  chainSpan: r2(mx),
  walltime: SPAN,
  nodes: critNodes,
  gapOnPath: r2(posGap),
  overlapOnPath: r2(overlap),
  spanSum: r2(sum(critTags.map((tg) => byTag[tg].span))),
};

const aicLanes = lanes.filter((l) => l.kind === 'aic');
const aivLanes = lanes.filter((l) => l.kind === 'aiv');

return {
  rank: rank,
  traceFile: traceFile,
  swimlane: { spanUs: SPAN, laneNames: laneNames, lanes: lanes, blocks: laneBlocks },
  tasks: tasks,
  taskIndex: taskIndex,
  critical: critical,
  scheduler: {
    processNames: processName, lanes: schedLanes, laneStats: schedLaneStats,
    phases: schedPhases, window: schedWindow, busy: schedBusy, blocks: schedBlocks,
    perLaneUtil: r2(sum(schedLaneStats.map((s) => s.util)) / schedLaneStats.length),
  },
  orchestrator: orch,
  readyQueue: readyQueue,
  readyStat: rqStat,
  flows: flowCount,
  hbViolations: hbPairs,
  occupancy: {
    aicUtil: r2(sum(aicLanes.map((l) => l.util)) / aicLanes.length),
    aivUtil: r2(sum(aivLanes.map((l) => l.util)) / aivLanes.length),
    aicWorst: aicLanes.slice().sort((a, b) => a.util - b.util)[0],
    aivWorst: aivLanes.slice().sort((a, b) => a.util - b.util)[0],
    aicBest: aicLanes.slice().sort((a, b) => b.util - a.util)[0],
    aivBest: aivLanes.slice().sort((a, b) => b.util - a.util)[0],
    maxGapLane: lanes.slice().sort((a, b) => b.maxGap - a.maxGap)[0],
  },
};
}

const RANKS = {
  rank0: buildRank('rank0', 'merged_swimlane_20260903_010746.json'),
  rank1: buildRank('rank1', 'merged_swimlane_20260903_010747.json'),
};

/* Findings and inspector defaults are anchored on rank0's trace. */
const R = RANKS.rank0;
const SPAN = R.swimlane.spanUs;
const tasks = R.tasks;
const lanes = R.swimlane.lanes;
const critTags = R.critical.tags;
const critical = R.critical;
const schedPhases = R.scheduler.phases;
const schedLanes = R.scheduler.lanes;
const schedLaneStats = R.scheduler.laneStats;
const schedWindow = R.scheduler.window;
const schedBusy = R.scheduler.busy;
const rqStat = R.readyStat;
const hbPairs = R.hbViolations;

/* ------------------------------------------------------- compiler hints */
const hintLines = rd('report/perf_hints.log').split(/\r?\n/).filter((l) => l.trim());
const hints = hintLines.map((line) => {
  const at = line.match(/ at (\/\S+):(\d+):(\d+)\s*$/);
  const code = (line.match(/\[perf_hint ([A-Z0-9-]+)\]/) || [])[1];
  const occ = +(line.match(/\((\d+) occurrences at this source location\)/) || [0, 1])[1];
  const h = {
    code: code,
    file: at ? at[1].split('/').pop() : null,
    module: at ? at[1].replace(/^.*pypto-lib\//, '') : null,
    line: at ? +at[2] : null,
    col: at ? +at[3] : null,
    occurrences: occ,
  };
  if (code === 'PH001') {
    const m = line.match(/TileInnermostDimGranularity: (\S+) has innermost dim = (\d+)B \(tile (\w+)\[([^\]]+)\], target_memory=(\w+)\)/);
    if (m) { h.op = m[1]; h.innermostB = +m[2]; h.dtype = m[3]; h.tileShape = m[4]; h.mem = m[5]; }
    const mv = line.match(/moves (\d+)B as (\d+) x (\d+)B rows/);
    if (mv) { h.movesB = +mv[1]; h.rows = +mv[2]; h.rowB = +mv[3]; }
    const rec = line.match(/recommended >= (\d+)B for backend (\w+)/);
    if (rec) { h.recB = +rec[1]; h.backend = rec[2]; }
    const cl = line.match(/L2 cache line = (\d+)B/);
    if (cl) h.cacheLineB = +cl[1];
    h.kind = 'tile-granularity';
  } else if (code === 'PH-MR-001') {
    const m = line.match(/requested depth (\d+) for pipeline group (\d+) in (\w+), but only (\d+) of (\d+) buffers fit \((\d+) B per stage, (\d+) B free\)/);
    if (m) {
      h.reqDepth = +m[1]; h.group = +m[2]; h.unit = m[3];
      h.fit = +m[4]; h.of = +m[5]; h.perStageB = +m[6]; h.freeB = +m[7];
    }
    const own = line.match(/would fit depth (\d+) on its own/);
    if (own) h.ownDepth = +own[1];
    h.kind = 'pipeline-depth';
  }
  return h;
});
const ph001 = hints.filter((h) => h.code === 'PH001');
const phmr = hints.filter((h) => h.code === 'PH-MR-001');

const tileSites = {};
for (const h of ph001) {
  const k = h.file + ':' + h.line;
  const s = (tileSites[k] = tileSites[k] || {
    file: h.file, module: h.module, line: h.line, n: 0, occ: 0,
    minB: Infinity, mems: {}, ops: {}, dtypes: {}, recB: h.recB, cacheLineB: h.cacheLineB, shapes: {},
  });
  s.n++; s.occ += h.occurrences;
  if (h.innermostB) s.minB = Math.min(s.minB, h.innermostB);
  s.mems[h.mem] = (s.mems[h.mem] || 0) + 1;
  s.ops[h.op] = (s.ops[h.op] || 0) + 1;
  s.dtypes[h.dtype] = (s.dtypes[h.dtype] || 0) + 1;
  if (h.dtype) s.shapes[h.dtype + '[' + h.tileShape + ']'] = 1;
}
const tileSiteList = Object.keys(tileSites).map((k) => {
  const s = tileSites[k];
  return {
    key: k, file: s.file, module: s.module, line: s.line, n: s.n, occ: s.occ,
    minB: s.minB === Infinity ? null : s.minB, recB: s.recB, cacheLineB: s.cacheLineB,
    mems: s.mems, ops: s.ops, dtypes: s.dtypes, shapes: Object.keys(s.shapes).slice(0, 6),
  };
}).sort((a, b) => (a.minB || 1e9) - (b.minB || 1e9) || b.occ - a.occ);

const tileByFile = {};
ph001.forEach((h) => {
  const f = (tileByFile[h.file] = tileByFile[h.file] || { file: h.file, module: h.module, n: 0, occ: 0, minB: Infinity, sites: {} });
  f.n++; f.occ += h.occurrences;
  if (h.innermostB) f.minB = Math.min(f.minB, h.innermostB);
  f.sites[h.line] = 1;
});
const tileFileList = Object.keys(tileByFile).map((k) => {
  const f = tileByFile[k];
  return { file: f.file, module: f.module, n: f.n, occ: f.occ, minB: f.minB === Infinity ? null : f.minB, siteCount: Object.keys(f.sites).length };
}).sort((a, b) => b.occ - a.occ);

const depthSites = {};
for (const h of phmr) {
  const k = h.file + ':' + h.line;
  const s = (depthSites[k] = depthSites[k] || { file: h.file, module: h.module, line: h.line, groups: [], units: {} });
  s.groups.push({ group: h.group, unit: h.unit, reqDepth: h.reqDepth, fit: h.fit, perStageB: h.perStageB, freeB: h.freeB, ownDepth: h.ownDepth });
  s.units[h.unit] = (s.units[h.unit] || 0) + 1;
}
const depthSiteList = Object.keys(depthSites).map((k) => {
  const s = depthSites[k];
  return {
    key: k, file: s.file, module: s.module, line: s.line,
    units: Object.keys(s.units), groupCount: s.groups.length,
    maxReqDepth: Math.max.apply(null, s.groups.map((g) => g.reqDepth)),
    fittedDepth: Math.max.apply(null, s.groups.map((g) => g.fit)),
    perStageB: Math.max.apply(null, s.groups.map((g) => g.perStageB)),
    freeB: Math.max.apply(null, s.groups.map((g) => g.freeB)),
    groups: s.groups,
  };
}).sort((a, b) => b.groupCount - a.groupCount);

/* on-chip free space per staging unit, as this run's MemoryReuse pass reported it */
const budgets = {};
for (const h of phmr) {
  const b = (budgets[h.unit] = budgets[h.unit] || { unit: h.unit, freeB: 0, sites: 0, minStageB: Infinity, maxStageB: 0 });
  b.freeB = Math.max(b.freeB, h.freeB);
  b.minStageB = Math.min(b.minStageB, h.perStageB);
  b.maxStageB = Math.max(b.maxStageB, h.perStageB);
  b.sites++;
}

/* ---------------------------------------------------------- IR / passes */
const passDir = path.join(RUN, 'passes_dump');
const passFiles = fs.readdirSync(passDir).filter((f) => f.endsWith('.py')).sort();
let prevLines = null;
const passes = passFiles.map((f) => {
  const text = fs.readFileSync(path.join(passDir, f), 'utf8');
  const lines = text.split('\n').length;
  const row = {
    idx: +f.slice(0, 2),
    name: f.replace(/^\d+_(after_)?/, '').replace(/\.py$/, ''),
    file: f,
    lines: lines,
    delta: prevLines === null ? 0 : lines - prevLines,
    counts: {
      pipeline: (text.match(/pl\.pipeline\(/g) || []).length,
      matmul: (text.match(/pl\.tile\.matmul/g) || []).length,
      spmd: (text.match(/pl\.spmd/g) || []).length,
      range: (text.match(/pl\.range/g) || []).length,
      left: (text.match(/pl\.Mem\.Left/g) || []).length,
      right: (text.match(/pl\.Mem\.Right/g) || []).length,
      acc: (text.match(/pl\.Mem\.Acc/g) || []).length,
      vec: (text.match(/pl\.Mem\.Vec/g) || []).length,
    },
  };
  prevLines = lines;
  return row;
});

const autoTile = fs.readFileSync(path.join(passDir, '17_after_AutoTileMatmulL0.py'), 'utf8');
const pipelineSites = Array.from(autoTile.matchAll(/pl\.pipeline\(([^)]*)\)/g)).map((m) => {
  const args = m[1];
  const stage = +(args.match(/stage=(\d+)/) || [0, 0])[1];
  const trip = args.split(',')[0].trim();
  const carriers = (args.match(/init_values=\(([^)]*)/) || ['', ''])[1].split(',').filter((s) => s.trim()).length;
  const carrier = ((args.match(/init_values=\(([^,)]*)/) || ['', ''])[1] || '').trim();
  return { stage: stage, trip: trip, carriers: carriers, carrier: carrier };
}).filter((s) => s.stage > 0).sort((a, b) => b.stage - a.stage || String(a.trip).localeCompare(String(b.trip)));

const l0Tiles = (function () {
  const agg = {};
  for (const m of autoTile.matchAll(/pl\.Tile\[\[(\d+), (\d+)\], pl\.(\w+), pl\.Mem\.(Left|Right|Acc)/g)) {
    const rows = +m[1], cols = +m[2], dtype = m[3], mem = m[4];
    const bpe = /INT8|FP8/.test(dtype) ? 1 : /BF16|FP16/.test(dtype) ? 2 : 4;
    const key = mem + '|' + dtype + '|' + rows + 'x' + cols;
    const a = (agg[key] = agg[key] || {
      mem: mem, dtype: dtype, rows: rows, cols: cols,
      bytesPerElem: bpe, bytes: rows * cols * bpe, innermostB: cols * bpe, n: 0,
    });
    a.n++;
  }
  return Object.keys(agg).map((k) => agg[k]).sort((a, b) => b.bytes - a.bytes || b.n - a.n);
})();

const frontend = fs.readFileSync(path.join(passDir, '00_frontend.py'), 'utf8');
const finalIr = fs.readFileSync(path.join(passDir, passFiles[passFiles.length - 1]), 'utf8');
const dsl = {
  parallel: (frontend.match(/pl\.parallel/g) || []).length,
  range: (frontend.match(/pl\.range/g) || []).length,
  spmd: (frontend.match(/pl\.spmd/g) || []).length,
  pipeline: (frontend.match(/pl\.pipeline/g) || []).length,
  prefetch: (frontend.match(/pl\.prefetch/g) || []).length,
  functions: (frontend.match(/@pl\.function/g) || []).length,
  matmulFinal: (finalIr.match(/pl\.tile\.matmul/g) || []).length,
  rangeFinal: (finalIr.match(/pl\.range/g) || []).length,
  spmdFinal: (finalIr.match(/pl\.spmd/g) || []).length,
};

/* ----------------------------------------------------- IR excerpt pairs */
function excerpt(file, needle, before, after) {
  const text = fs.readFileSync(path.join(passDir, file), 'utf8').split('\n');
  const i = text.findIndex((l) => l.indexOf(needle) >= 0);
  if (i < 0) return null;
  return text.slice(Math.max(0, i - before), i + after).map((l) => l.replace(/\s+$/, ''));
}
const irPairs = (function () {
  const afterLines = autoTile.split('\n');
  const i = afterLines.findIndex((l) => /_l0_init_storage/.test(l) && /pl\.tile\.create/.test(l));
  if (i < 0) return [];
  const base = (afterLines[i].match(/^\s*(\w+)_l0_init_storage/) || [])[1];
  const beforeLines = fs.readFileSync(path.join(passDir, '16_after_LegalizeTileCast.py'), 'utf8').split('\n');
  const j = beforeLines.findIndex((l) => l.indexOf(base) >= 0 && /pl\.tile\.matmul/.test(l));
  const clean = (a) => a.map((l) => l.replace(/\s+$/, '')).filter((l) => l.length);
  return [{
    pass: 'AutoTileMatmulL0',
    passIdx: 17,
    subject: base,
    beforeFile: '16_after_LegalizeTileCast.py',
    afterFile: '17_after_AutoTileMatmulL0.py',
    before: j < 0 ? [] : clean(beforeLines.slice(j, j + 2)),
    after: clean(afterLines.slice(i, i + 7)),
  }];
})();

/* ------------------------------------------------------------- findings */
const pick = (name) => tasks.filter((t) => t.callable === name)[0];

const waitTasks = tasks.filter((t) => /_wait$/.test(t.callable || '')).sort((a, b) => b.span - a.span);
const waitSpan = r2(sum(waitTasks.map((t) => t.span)));
const qkpv = pick('qk_pv_aic');
const worstImb = tasks.filter((t) => t.blockCount >= 32).sort((a, b) => b.imbalance - a.imbalance)[0];
const worstHandoff = tasks.filter((t) => t.svOverhead != null && t.blockCount >= 8)
  .sort((a, b) => b.svOverhead - a.svOverhead)[0];
const worstSetupShare = tasks.filter((t) => t.blockCount >= 8).sort((a, b) => b.setupShare - a.setupShare)[0];
const aicUtil = R.occupancy.aicUtil;
const aivUtil = R.occupancy.aivUtil;
const schedPerLaneUtil = R.scheduler.perLaneUtil;
const rank0Dev = e2e.rank0[2]['chip.run.runner_run.device_wall'].us;
const rank1Dev = e2e.rank1[2]['chip.run.runner_run.device_wall'].us;
const depthDegraded = depthSiteList.filter((s) => s.fittedDepth < s.maxReqDepth);
const setupHeavy = tasks.filter((t) => t.setupShare > 0.05).sort((a, b) => b.setupSum - a.setupSum);

const findings = [
  {
    id: 'F1', level: 'l2', severity: 'high', axis: 'comm',
    title: '通信等待独占关键路径 ' + r2((waitSpan / SPAN) * 100) + '%',
    metric: waitSpan + ' us / ' + SPAN + ' us',
    claim: waitTasks.length + ' 个 *_wait 任务合计 ' + waitSpan + ' us，全部单块单核，其中 '
      + waitTasks[0].callable + ' 单独 ' + waitTasks[0].span + ' us；关键路径 33 个节点里通信与 publish 段占主导。',
    evidence: [
      { artifact: 'merged_swimlane (rank0/d0)', locator: waitTasks.map((t) => t.tag).join(' / '), value: waitTasks.map((t) => t.callable + '=' + t.span + 'us').join(', ') },
      { artifact: 'critical path (fanin/fanout hints)', locator: 'chain ' + critTags.length + ' nodes', value: 'span sum ' + critical.spanSum + ' us' },
    ],
    focus: { view: 'l2', task: waitTasks[0].tag, critOnly: true },
    lever: '按通信算子优先级推进：先确认算法选择（allgather vs 分块 readback），再做通算重叠，最后才调块大小与乒乓。',
    guardrail: '必须同时报告带宽利用率与 overlap 效率；只压缩 wait 时长而不看带宽，会把等待搬到别处。',
    verify: '重测 device_wall 与该 wait 任务 span，并确认 *_wait 仍在关键路径上。',
  },
  {
    id: 'F2', level: 'l1', severity: 'high', axis: 'launch',
    title: worstHandoff.callable + ' hand-off 比核上计算还贵（+' + worstHandoff.svOverhead + ' us/块）',
    metric: 'AICPU ' + worstHandoff.svAicpuMean + ' us vs 核上 ' + worstHandoff.durMean + ' us',
    claim: '同一个块有两个视角：Worker View 记核上 ' + worstHandoff.durMean + ' us，Scheduler View 记 dispatch→finish '
      + worstHandoff.svAicpuMean + ' us，差 ' + worstHandoff.svOverhead + ' us 是领取与依赖等待。核上那 '
      + worstHandoff.durMean + ' us 里还有 ' + worstHandoff.setupMean + ' us（' + r2(worstHandoff.setupShare * 100)
      + '%）是 local_setup 而非 kernel；' + worstHandoff.blockCount + ' 块累计 setup ' + worstHandoff.setupSum + ' us。',
    evidence: [
      { artifact: 'Scheduler View (pid 3)', locator: worstHandoff.tag + ' (' + worstHandoff.callable + ')', value: 'aicpu-duration mean ' + worstHandoff.svAicpuMean + ' us, max ' + worstHandoff.svAicpuMax + ' us' },
      { artifact: 'Worker View (pid 4)', locator: 'duration − kernel_duration', value: 'setup mean ' + worstHandoff.setupMean + ' us, kernel mean ' + r2(worstHandoff.kdurSum / worstHandoff.blockCount) + ' us' },
      { artifact: 'same trace', locator: 'setup 占比 > 5% 的任务', value: setupHeavy.length + ' 个，最高 ' + worstSetupShare.callable + ' ' + r2(worstSetupShare.setupShare * 100) + '%' },
    ],
    focus: { view: 'l1', task: worstHandoff.tag },
    lever: '把 publish 段与它的生产者合进同一 mixed kernel，消掉一次 AICPU hand-off；setup 里可复用的准备提到核外或跨块复用。',
    guardrail: '合核会拉长单核占用；合并后要复查该核是否变成新的独占瓶颈。',
    verify: '重测该任务的 aicpu-duration、setup mean 与所在核 util；hand-off 差值应收窄且核 util 不恶化。',
  },
  {
    id: 'F3', level: 'e2e', severity: 'high', axis: 'balance',
    title: 'rank0 / rank1 device_wall 偏斜 ' + r2(rank0Dev / rank1Dev) + 'x',
    metric: 'rank0 ' + rank0Dev + ' us vs rank1 ' + rank1Dev + ' us (inv=2)',
    claim: '稳定态第 2 次调用里 rank0 的 device_wall 比 rank1 长 ' + r2(rank0Dev - rank1Dev)
      + ' us，sched 段 ' + e2e.rank0[2]['chip.run.runner_run.device_wall.sched'].us + ' vs '
      + e2e.rank1[2]['chip.run.runner_run.device_wall.sched'].us + ' us。先分 rank 再看内核，否则会把负载不均当成内核问题。',
    evidence: [
      { artifact: 'host.2263908.log (rank0)', locator: 'inv=2 device_wall', value: rank0Dev + ' us, sched ' + e2e.rank0[2]['chip.run.runner_run.device_wall.sched'].us + ' us' },
      { artifact: 'host.2263922.log (rank1)', locator: 'inv=2 device_wall', value: rank1Dev + ' us, sched ' + e2e.rank1[2]['chip.run.runner_run.device_wall.sched'].us + ' us' },
      { artifact: 'both logs', locator: 'inv=1 graph_build', value: 'rank0 ' + e2e.rank0[1]['chip.run.runner_run.device_wall.graph_build'].us + ' us vs rank1 ' + e2e.rank1[1]['chip.run.runner_run.device_wall.graph_build'].us + ' us — 首次 JIT 建图，不能当稳定态' },
      { artifact: 'merged_swimlane (both ranks)', locator: 'device-side trace span', value: 'rank0 ' + RANKS.rank0.swimlane.spanUs + ' us / ' + RANKS.rank0.tasks.length + ' 任务 vs rank1 ' + RANKS.rank1.swimlane.spanUs + ' us / ' + RANKS.rank1.tasks.length + ' 任务' },
      { artifact: 'worker lanes (both ranks)', locator: 'AIC / AIV 平均占用', value: 'rank0 ' + RANKS.rank0.occupancy.aicUtil + '% / ' + RANKS.rank0.occupancy.aivUtil + '%，rank1 ' + RANKS.rank1.occupancy.aicUtil + '% / ' + RANKS.rank1.occupancy.aivUtil + '%' },
    ],
    focus: { view: 'e2e' },
    lever: '先确认偏斜来自数据切分还是通信同步；rank 侧不均衡应在 L2 任务划分里解决，不要先改单核 Tile。',
    guardrail: '本次 dump 只有 2 次调用，mean/median 不成立。要下结论必须补足迭代次数。',
    verify: '固定 case 重跑 ≥10 次，分 rank 记录 device_wall 的 mean/median 与偏斜比。',
  },
  {
    id: 'F4', level: 'compiler', severity: 'high', axis: 'pipeline',
    title: depthDegraded.length + ' 处软流水深度被降到 ' + depthDegraded[0].fittedDepth,
    metric: phmr.length + ' 条 PH-MR-001 / ' + depthDegraded.length + ' 个源码点',
    claim: 'MemoryReuse 报告：请求 depth ' + Array.from(new Set(depthSiteList.map((s) => s.maxReqDepth))).sort().join('/')
      + '，实际只有 1 个 buffer 放得下，相距 1 个 stage 的操作共享存储并串行化。'
      + 'Left/Right 每 stage ' + budgets.Right.minStageB + '–' + budgets.Right.maxStageB + ' B，可用 ' + budgets.Right.freeB + ' B；Vec 每 stage ' + budgets.Vec.minStageB + ' B，可用 ' + budgets.Vec.freeB + ' B。',
    evidence: depthSiteList.slice(0, 4).map((s) => ({
      artifact: 'report/perf_hints.log', locator: s.module + ':' + s.line,
      value: 'depth ' + s.maxReqDepth + '→' + s.fittedDepth + '，' + s.groupCount + ' 组 @' + s.units.join('/') + '，' + s.perStageB + ' B/stage，' + s.freeB + ' B free',
    })),
    focus: { view: 'compiler', pass: 'MemoryReuse' },
    lever: '先减少同驻 tile（更小或更少 co-live 操作数），再谈调 stage；Left/Right 是编译器的 L0A/L0B staging 结果，不要当独立 Tile 预算去调。',
    guardrail: '盲目加大 stage 只会让 MemoryReuse 再降一次深度，并多出一条同样的提示。',
    verify: '改后重跑编译，确认该源码点不再出现 PH-MR-001，并复测该 kernel 的块时长。',
  },
  {
    id: 'F5', level: 'compiler', severity: 'medium', axis: 'granularity',
    title: sum(ph001.map((h) => h.occurrences)) + ' 次搬运末维 < ' + ph001[0].cacheLineB + 'B cache line',
    metric: '最小 ' + tileSiteList[0].minB + 'B，覆盖 ' + tileFileList.length + ' 个算子文件',
    claim: 'TileInnermostDimGranularity 在 ' + tileSiteList.length + ' 个源码点上报 tile.load/tile.store 末维不足一个 L2 cache line，最极端处只有 '
      + tileSiteList[0].minB + 'B。末维碎片化会让每次搬运只拿到 cache line 的一小部分。',
    evidence: tileFileList.slice(0, 5).map((f) => ({
      artifact: 'report/perf_hints.log', locator: f.module,
      value: f.occ + ' 次 / ' + f.siteCount + ' 个源码点，最小末维 ' + f.minB + 'B',
    })),
    focus: { view: 'compiler', tab: 'granularity' },
    lever: '按 dtype 把末维凑到 512B：BF16 → 256 元素倍数，FP32 → 128，INT8 → 512。',
    guardrail: '加大末维会同时抬高 L0/UB 占用，可能触发 F4 的深度回退；两项要一起看。',
    verify: '重编译后核对 PH001 条数与最小末维，并复测对应 kernel 的 MTE 时间。',
  },
  {
    id: 'F6', level: 'l2', severity: 'medium', axis: 'sched',
    title: 'AICPU 调度器平均占用 ' + schedPerLaneUtil + '%',
    metric: schedBusy + ' us busy / ' + schedLanes.length + ' 个调度线程',
    claim: schedLanes.length + ' 个调度线程在 ' + r2(schedWindow.hi - schedWindow.lo) + ' us 窗口内合计 busy ' + schedBusy
      + ' us。complete 阶段最贵：' + schedPhases.complete.us + ' us 处理 ' + schedPhases.complete.tasks
      + ' 次完成，约 ' + schedPhases.complete.usPerTask + ' us/次；dispatch ' + schedPhases.dispatch.usPerTask + ' us/次。',
    evidence: Object.keys(schedPhases).sort((a, b) => schedPhases[b].us - schedPhases[a].us).map((k) => ({
      artifact: 'merged_swimlane scheduler lane', locator: 'phase=' + k,
      value: schedPhases[k].us + ' us / ' + schedPhases[k].n + ' 段 / ' + schedPhases[k].tasks + ' 任务'
        + (schedPhases[k].usPerTask ? ' = ' + schedPhases[k].usPerTask + ' us/任务' : ''),
    })),
    focus: { view: 'l2', overlay: 'sched' },
    lever: '合并相邻核、把外层迭代折进核内，或用 pl.spmd 一次 fan-out 多块，降低完成回收次数。',
    guardrail: 'A3/910C 上「约 50us 级内核」只是实测启发式，不是跨芯片硬规则；合核到多长要按本 case 实测。',
    verify: '重测 complete 段总时长与任务数；单任务代价不变而次数下降才算生效。',
  },
  {
    id: 'F7', level: 'l2', severity: 'medium', axis: 'sched',
    title: 'AIC ready-but-undispatched 占窗口 ' + rqStat.busyShare.AIC + '%',
    metric: 'avg ' + rqStat.avg.AIC + ' / peak ' + rqStat.peak.AIC,
    claim: 'shared_ready_queue 在 ' + rqStat.busyTime.AIC + ' us（窗口的 ' + rqStat.busyShare.AIC
      + '%）里有 AIC 任务已 ready 但未派发，峰值 ' + rqStat.peak.AIC + ' 个；AIV 侧 ' + rqStat.busyShare.AIV + '%，峰值 ' + rqStat.peak.AIV + '。'
      + '同期 AIC 平均占用只有 ' + aicUtil + '%。',
    evidence: [
      { artifact: 'merged_swimlane queue counter', locator: 'shared_ready_queue', value: 'AIC avg ' + rqStat.avg.AIC + ', peak ' + rqStat.peak.AIC + ', >0 占 ' + rqStat.busyShare.AIC + '%' },
      { artifact: 'worker lanes', locator: 'AIC 平均 vs AIV 平均', value: aicUtil + '% vs ' + aivUtil + '%' },
      { artifact: 'flow events', locator: 'hb_violation', value: hbPairs.length + ' 对 happens-before 违例标记' },
    ],
    focus: { view: 'l2', overlay: 'ready' },
    lever: '只针对已观测到的关键路径提前 dispatch 或调整依赖，不要全局提前。',
    guardrail: '提前 dispatch、改依赖、延后非关键任务都可能以吞吐换时延；两个指标都要报。',
    verify: '重测 ready>0 时间占比、AIC 平均占用与 device_wall；三者要同向改善。',
  },
  {
    id: 'F8', level: 'l1', severity: 'medium', axis: 'balance',
    title: worstImb.callable + ' 块时长离散 ' + worstImb.imbalance + 'x',
    metric: 'max ' + worstImb.durMax + ' us / med ' + worstImb.durMed + ' us',
    claim: worstImb.blockCount + ' 个块摊到 ' + worstImb.coreCount + ' 核（约 '
      + r2(worstImb.blockCount / worstImb.coreCount) + ' 波），最慢块 ' + worstImb.durMax
      + ' us 是中位块的 ' + worstImb.imbalance + ' 倍，尾块决定该任务 span ' + worstImb.span + ' us。',
    evidence: [
      { artifact: 'merged_swimlane blocks', locator: worstImb.tag + ' (' + worstImb.callable + ')', value: 'min ' + worstImb.durMin + ' / med ' + worstImb.durMed + ' / p90 ' + worstImb.durP90 + ' / max ' + worstImb.durMax + ' us' },
      { artifact: 'deps.json', locator: 'task ' + worstImb.id, value: 'block_num=' + worstImb.blockNum + ', scope=' + worstImb.scope },
    ],
    focus: { view: 'l1', task: worstImb.tag },
    lever: '独立循环用 pl.parallel 而非 pl.range；尾块偏长说明每块工作量不齐，需要重新切分而不是加核。',
    guardrail: '先确认慢块是工作量差异还是 MTE/UB 争用；PMU 打开会改变调度，不能与 PMU-off 基线直接比较。',
    verify: '重测该任务 durMax/durMed 与 span；离散度下降且 span 缩短才算生效。',
  },
  {
    id: 'F9', level: 'l1', severity: 'medium', axis: 'fusion',
    title: qkpv.callable + ' 混合核跨 ' + qkpv.coreCount + ' 核，单块 ' + qkpv.durMean + ' us',
    metric: 'span ' + qkpv.span + ' us，核上 ' + qkpv.durMean + ' us/块',
    claim: '本 case 只有 ' + tasks.filter((t) => t.kind === 'mix').length + ' 个 mixed kernel 横跨全部 '
      + qkpv.coreCount + ' 个核（AIC+AIV 同核），这是其中最贵的一个：span '
      + qkpv.span + ' us 而单块 ' + qkpv.durMean + ' us，说明 span 几乎等于单块时长，Cube 与 Vec 段是串在块内的。'
      + '同段还有 ' + (pick('qk_pv_aiv') ? pick('qk_pv_aiv').tag : 'qk_pv_aiv') + ' 作为 Vec 侧对应体。',
    evidence: [
      { artifact: 'merged_swimlane', locator: qkpv.tag + ' (' + qkpv.callable + ')', value: qkpv.blockCount + ' 块 / ' + qkpv.coreCount + ' 核，min ' + qkpv.durMin + ' / med ' + qkpv.durMed + ' / max ' + qkpv.durMax + ' us' },
      { artifact: 'deps.json', locator: 'task ' + qkpv.id, value: 'block_num=' + qkpv.blockNum + '，前驱 ' + qkpv.pred.length + ' 个，后继 ' + qkpv.succ.length + ' 个' },
      { artifact: 'critical path', locator: '是否在关键路径上', value: critTags.indexOf(qkpv.tag) >= 0 ? '在，第 ' + (critTags.indexOf(qkpv.tag) + 1) + ' 个节点' : '不在' },
    ],
    focus: { view: 'l1', task: qkpv.tag },
    lever: '按 Flash Attention 的 Cube→Vec→Cube→Vec 解耦思路，用 GM FIFO 让两侧真正并行，而不是块内串行。',
    guardrail: 'S1_TILE、预加载深度、FIFO 槽数与 UB 预算必须一起推导；只单独扫 Tile 会在 F4 的深度回退上撞墙。',
    verify: '重测该任务 span 与单块时长的比值；解耦生效后 span 应显著小于 块数 × 单块时长 / 核数。',
  },
  {
    id: 'F10', level: 'l2', severity: 'low', axis: 'reuse',
    title: '本程序 0 处 pl.prefetch，L2 复用杠杆未启用',
    metric: 'prefetch ' + dsl.prefetch + ' / pipeline ' + dsl.pipeline + ' / spmd ' + dsl.spmd + ' / parallel ' + dsl.parallel,
    claim: '前端 IR 里 pl.prefetch 出现 ' + dsl.prefetch + ' 次。权重类输入确实存在（'
      + caseInfo.params.filter((p) => /^w/.test(p.name)).length + ' 个 w* 参数），但没有任何静态预取，也没有 N-group swizzle 证据。',
    evidence: [
      { artifact: 'passes_dump/00_frontend.py', locator: 'pl.prefetch', value: dsl.prefetch + ' 处' },
      { artifact: 'distributed_meta.json', locator: 'w* 参数', value: caseInfo.params.filter((p) => /^w/.test(p.name)).map((p) => p.name.replace(/__ssa_v0$/, '') + ' ' + p.dtype + JSON.stringify(p.shape)).slice(0, 4).join(', ') },
    ],
    focus: { view: 'l2', overlay: 'none' },
    lever: '只对静态、确定会冷、且所有在途 warm 数据能放进 L2 的权重用 pl.prefetch。',
    guardrail: 'prefetch 占 SDMA；错误预取会拖慢通信或挤掉真正需要的数据。本 case 通信已是瓶颈（F1），风险更高。',
    verify: '加预取后同时看 device_wall、通信段 span 与 SDMA 占用，三者不能互相恶化。',
  },
];

/* --------------------------------------------------- finding -> subjects
 * A finding is only useful if the reader can see it on the stage. Each one
 * names the concrete objects the centre view should mark (tasks, source
 * sites, scheduler phases), so the stage can number them instead of leaving
 * the reader to guess which parts the inspector is talking about.          */
const taskByTag = {};
tasks.forEach((t) => { taskByTag[t.tag] = t; });

const SUBJECTS = {
  F1: {
    view: 'l2',
    tasks: waitTasks.map((t) => t.tag),
  },
  F2: {
    view: 'l1',
    tasks: [worstHandoff.tag],
  },
  F3: {
    view: 'e2e',
    ranks: ['rank0', 'rank1'],
  },
  F4: {
    view: 'compiler', tab: 'depth',
    sites: depthSiteList.map((s) => s.key),
  },
  F5: {
    view: 'compiler', tab: 'granularity',
    sites: tileSiteList.slice(0, 12).map((s) => s.key),
    files: tileFileList.map((f) => f.file),
  },
  F6: {
    view: 'l2', overlay: 'sched',
    schedPhases: ['complete', 'dispatch'],
  },
  F7: {
    view: 'l2', overlay: 'ready',
    lanes: lanes.filter((l) => l.kind === 'aic').sort((a, b) => a.util - b.util).slice(0, 6).map((l) => l.name),
  },
  F8: {
    view: 'l1',
    tasks: [worstImb.tag],
  },
  F9: {
    view: 'l1',
    tasks: [qkpv.tag],
  },
  F10: {
    view: 'l2',
    absent: true,
  },
};
findings.forEach((f) => {
  const s = SUBJECTS[f.id] || {};
  f.subjects = {
    view: s.view || (f.focus && f.focus.view) || 'l2',
    tab: s.tab || null,
    overlay: s.overlay || null,
    tasks: s.tasks || [],
    lanes: s.lanes || [],
    sites: s.sites || [],
    files: s.files || [],
    ranks: s.ranks || [],
    schedPhases: s.schedPhases || [],
    absent: !!s.absent,
  };
  /* chips shown in the centre evidence bar, each one jumpable */
  f.chips = []
    .concat(f.subjects.tasks.map((tag) => {
      const t = taskByTag[tag];
      return { kind: 'task', id: tag, label: t ? t.callable : tag, value: t ? t.span + ' us' : '' };
    }))
    .concat(f.subjects.sites.map((key) => {
      const d = depthSiteList.find((x) => x.key === key);
      const g = tileSiteList.find((x) => x.key === key);
      return {
        kind: 'site', id: key, label: key,
        value: d ? 'depth ' + d.maxReqDepth + '→' + d.fittedDepth : (g ? g.minB + 'B' : ''),
      };
    }))
    .concat(f.subjects.lanes.map((name) => {
      const l = lanes.find((x) => x.name === name);
      return { kind: 'lane', id: name, label: name, value: l ? r2(l.util) + '%' : '' };
    }))
    .concat(f.subjects.ranks.map((r) => ({
      kind: 'rank', id: r, label: r,
      value: e2e[r][2]['chip.run.runner_run.device_wall'].us + ' us',
    })))
    .concat(f.subjects.schedPhases.map((p) => ({
      kind: 'phase', id: p, label: 'phase ' + p,
      value: schedPhases[p] ? schedPhases[p].us + ' us' : '',
    })));
});

/* ---------------------------------------------------------------- write */
const payload = {
  generatedBy: 'Design/operator-tuning-console/build-data.cjs',
  source: caseInfo.runDir,
  case: caseInfo,
  e2e: e2e,
  ranks: RANKS,
  defaultRank: 'rank0',
  hints: hints,
  tileSites: tileSiteList,
  tileFiles: tileFileList,
  depthSites: depthSiteList,
  budgets: budgets,
  passes: passes,
  pipelineSites: pipelineSites,
  l0Tiles: l0Tiles,
  dsl: dsl,
  irPairs: irPairs,
  findings: findings,
  derived: {
    waitTasks: waitTasks.map((t) => t.tag), waitSpan: waitSpan,
    setupHeavy: setupHeavy.map((t) => t.tag),
    worstImb: worstImb.tag, worstHandoff: worstHandoff.tag, worstSetupShare: worstSetupShare.tag,
  },
};

fs.writeFileSync(OUT, 'window.TUNING_RUN = ' + JSON.stringify(payload) + ';\n');
console.log('wrote', OUT, (fs.statSync(OUT).size / 1024).toFixed(1) + ' KB');
Object.keys(RANKS).forEach((k) => {
  const x = RANKS[k];
  console.log(' ', k, 'span', x.swimlane.spanUs, 'us | tasks', x.tasks.length,
    '| blocks', sum(x.swimlane.blocks.map((a) => a.length)),
    '| crit', x.critical.tags.length, '| AIC', x.occupancy.aicUtil + '%', 'AIV', x.occupancy.aivUtil + '%');
});
console.log('hints', hints.length, 'passes', passes.length, 'findings', findings.length);
