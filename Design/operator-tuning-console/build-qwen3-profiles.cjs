/**
 * Build the Qwen3 14B profile layers that build-data.cjs has no concept of.
 *
 * Source: Data/pypto_qwen3_profiles.zip, unzipped in place. Four captures of
 * the same 40-layer fused-host graph:
 *
 *   tp1/swimlane/prefill_records/  merged swimlane WITH task names + a complete
 *                                  dependency graph -> this one feeds the
 *                                  task-level machinery in build-data.cjs
 *   tp1/swimlane/decode_records/   merged swimlane, FuncId = -1 on every task
 *                                  (no names) and fanin/fanout almost empty,
 *                                  but a full 18754-row l2_swimlane_records
 *   tp2/swimlane/{prefill,decode}_records/  l2_swimlane_records only: 643 tasks
 *                                  on 6 cores, no names in the capture — the
 *                                  names are recovered from the collector's own
 *                                  validated 40 x 16 layout
 *   tp{1,2}/torch/prof/.../ASCEND_PROFILER_OUTPUT/
 *                                  step_trace_time, op_statistic, api_statistic
 *                                  and kernel_details (real PMU counters)
 *
 * What this file adds on top of data.js:
 *   - torch attribution: per-step Computing / Free / Preparing, the framework
 *     ops around the fused kernel, and the host ACL API bill
 *   - measured PMU pipe occupancy for every invocation of aicore_kernel_0
 *   - the 40-layer unwrap, with the layer boundary recovered from the
 *     orchestrator submit order the same way the collector recovers it
 *   - the AICPU -> AICore head overhead split (NoC propagation vs dcci + ack)
 *   - a TP=1 / TP=2 comparison built from the two captures side by side
 *
 * Every number here is read from those files. Run:  node build-qwen3-profiles.cjs
 */
'use strict';
const fs = require('fs');
const path = require('path');

const OUT = path.join(__dirname, 'qwen3-profiles-data.js');
const DATA = path.resolve(__dirname, '../../Data');
/* the archive carries its own pypto_qwen3_profiles/ root; accept either nesting */
const ROOT = [
  path.join(DATA, 'pypto_qwen3_profiles/pypto_qwen3_profiles'),
  path.join(DATA, 'pypto_qwen3_profiles'),
].find((p) => fs.existsSync(path.join(p, 'tp1/swimlane')));
if (!ROOT) {
  console.error('pypto_qwen3_profiles not found under ' + DATA
    + '\n  unzip Data/pypto_qwen3_profiles.zip -d Data/pypto_qwen3_profiles');
  process.exit(1);
}

const at = (p) => path.join(ROOT, p);
const rj = (p) => JSON.parse(fs.readFileSync(at(p), 'utf8'));
const r2 = (n) => (n == null || !isFinite(n) ? null : Math.round(n * 100) / 100);
const r3 = (n) => (n == null || !isFinite(n) ? null : Math.round(n * 1000) / 1000);
const sum = (a) => a.reduce((x, y) => x + y, 0);
const sortNum = (a) => a.slice().sort((x, y) => x - y);
const quant = (sorted, q) => (sorted.length
  ? sorted[Math.min(sorted.length - 1, Math.max(0, Math.round((sorted.length - 1) * q)))] : null);

/* stats block reused for every duration / overhead distribution.
 * `refUs` is the core time these per-block numbers are measured against —
 * never wall time: a per-block overhead summed over 18754 blocks has nothing
 * to compare to a 32 ms span. */
function stats(vals, refUs) {
  const v = sortNum(vals.filter((x) => isFinite(x)));
  if (!v.length) return null;
  const s = sum(v);
  return {
    count: v.length,
    sumUs: r2(s),
    meanUs: r3(s / v.length),
    p50Us: r3(quant(v, 0.5)),
    p90Us: r3(quant(v, 0.9)),
    p99Us: r3(quant(v, 0.99)),
    minUs: r3(v[0]),
    maxUs: r3(v[v.length - 1]),
    shareOfKernel: refUs ? r3((s / refUs) * 100) : null,
  };
}

/* Where a layer's body begins. A handful of tasks per layer are dispatched far
 * ahead of the rest (the scheduler's early_dispatch), so min(start) describes
 * those stragglers rather than the layer. Split on the largest gap in the first
 * half of the start times, and only when that gap really stands out. */
function bodyStart(starts) {
  const s = sortNum(starts);
  if (s.length < 8) return { at: s[0], early: 0 };
  const gaps = [];
  for (let i = 1; i < s.length; i++) gaps.push(s[i] - s[i - 1]);
  const medGap = quant(sortNum(gaps), 0.5) || 0;
  let k = -1;
  let best = 0;
  for (let i = 0; i < Math.floor(s.length / 2); i++) {
    if (gaps[i] > best) { best = gaps[i]; k = i; }
  }
  if (k < 0 || best <= Math.max(medGap * 8, 1)) return { at: s[0], early: 0 };
  return { at: s[k + 1], early: k + 1, leadUs: s[k + 1] - s[0] };
}

/* ------------------------------------------------------------------- CSV
 * Ascend profiler CSVs quote fields that contain commas ("""1,1""").      */
function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = '';
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"') {
        if (text[i + 1] === '"') { cell += '"'; i++; } else quoted = false;
      } else cell += c;
      continue;
    }
    if (c === '"') { quoted = true; continue; }
    if (c === ',') { row.push(cell); cell = ''; continue; }
    if (c === '\n') { row.push(cell); rows.push(row); row = []; cell = ''; continue; }
    if (c === '\r') continue;
    cell += c;
  }
  if (cell.length || row.length) { row.push(cell); rows.push(row); }
  if (!rows.length) return [];
  const head = rows[0].map((h) => h.trim());
  return rows.slice(1).filter((r) => r.length > 1).map((r) => {
    const o = {};
    head.forEach((h, i) => { o[h] = (r[i] == null ? '' : r[i]).trim(); });
    return o;
  });
}
const nf = (v) => {
  if (v == null) return null;
  const s = String(v).trim();
  if (!s || s === 'N/A') return null;
  const n = Number(s);
  return isFinite(n) ? n : null;
};

/* =================================================================== capture
 * Two collectors, two validation states. The dataset's own .gitignore keeps
 * tp1/ and scratch/ out of the delivered tree and its README validates only
 * tp2 — the console has to carry that difference, not flatten it.          */
const CAPTURES = {
  tp1: {
    id: 'tp1',
    label: 'TP=1',
    sub: '单卡 fused host · 40 层整网',
    collector: 'collect_qwen3_profiles.py',
    validated: false,
    validateNote: '数据集 .gitignore 把 tp1/ 与 scratch/ 列为历史或临时生成物，'
      + 'README 只对 tp2 给了验收标记；tp1 的数字可用于对照，但不是数据集认定的交付基线。',
    swim: { prefill: 'tp1/swimlane/prefill_records', decode: 'tp1/swimlane/decode_records' },
    merged: {
      prefill: 'merged_swimlane_20260814_092436.json',
      decode: 'merged_swimlane_20260814_093106.json',
    },
    torch: 'tp1/torch/prof/6c42b4b8ccc347b39c28d0410ff20ed7_1334331_20260814094818713_ascend_pt',
    build: {
      prefill: 'scratch/tp1_build_output/_jit_prefill_fwd_20260814_093031',
      decode: 'scratch/tp1_build_output/_jit_decode_fwd_20260814_093048',
    },
  },
  tp2: {
    id: 'tp2',
    label: 'TP=2',
    sub: '2 rank · 40 层整网 · 存 rank0',
    collector: 'collect_qwen3_tp2_profiles.py',
    validated: true,
    validateNote: 'README 验收：每个 stage 恰好一次 ChipWorker.run，整图 643 个 aicore_task'
      + '（AIV 402 / AIC 241），rank0 与 rank1 都过校验，可视化产物取 rank0。',
    swim: { prefill: 'tp2/swimlane/prefill_records', decode: 'tp2/swimlane/decode_records' },
    merged: { prefill: null, decode: null },
    torch: 'tp2/torch/prof/6c42b4b8ccc347b39c28d0410ff20ed7_1828477_20260814152952803_ascend_pt',
    build: { prefill: null, decode: null },
  },
};
const STAGES = ['prefill', 'decode'];
const VID = (c, s) => c + ':' + s;

/* the collector's own 16-step layer layout, used to validate the 40 x 16 split
 * and to name tp2's otherwise anonymous tasks */
const LAYER_STEPS_16 = [
  ['RMS', 'aiv'], ['QKV', 'aic'], ['QKV post', 'aiv'], ['attn prepare', 'aiv'],
  ['QK', 'aic'], ['softmax', 'aiv'], ['PV', 'aic'], ['context cast', 'aiv'],
  ['O proj', 'aic'], ['AR(O)', 'aiv'], ['FFN RMS', 'aiv'], ['gate/up', 'aic'],
  ['SwiGLU', 'aiv'], ['down', 'aic'], ['AR(down)', 'aiv'], ['residual tail', 'aiv'],
];

/* ============================================== 40-layer split from submits
 * The graph is one embed + 40 identical layers + a tail, submitted in order by
 * the AICPU orchestrator. Between two layers the orchestrator submits a fixed
 * number of loop-control tasks that never reach an AICore, so the AICore
 * submit_idx jumps by a constant > 1 exactly 40 times. That jump is the layer
 * boundary; requiring it to appear 40 times at a constant spacing is what
 * makes the split a reading rather than a guess. */
function splitLayers(ordered, submitOf) {
  const idx = ordered.map(submitOf);
  const byGap = {};
  for (let i = 1; i < idx.length; i++) {
    const g = idx[i] - idx[i - 1];
    (byGap[g] = byGap[g] || []).push(i);
  }
  const cands = Object.keys(byGap).map(Number).filter((g) => g > 1 && byGap[g].length === 40)
    .map((g) => {
      const pos = byGap[g];
      const sp = Array.from(new Set(pos.slice(1).map((p, i) => p - pos[i])));
      return { gap: g, pos: pos, per: sp.length === 1 ? sp[0] : null };
    })
    .filter((c) => c.per && c.pos[0] >= 0
      && c.pos[39] + c.per <= ordered.length
      /* the 40 boundaries have to tile the loop exactly */
      && c.pos[39] - c.pos[0] === 39 * c.per);
  if (!cands.length) return null;
  /* prefer the split that leaves the fewest tasks outside the loop */
  cands.sort((a, b) => (ordered.length - 40 * a.per) - (ordered.length - 40 * b.per));
  const c = cands[0];
  const head = ordered.slice(0, c.pos[0]);
  const layers = [];
  for (let i = 0; i < 40; i++) layers.push(ordered.slice(c.pos[i], c.pos[i] + c.per));
  const tail = ordered.slice(c.pos[39] + c.per);
  return {
    gap: c.gap, per: c.per, head: head, layers: layers, tail: tail,
    ambiguous: cands.length > 1,
  };
}

/* ------------------------------------------------- l2_swimlane_records reader
 * aicore_tasks: [core_id, task_token, reg_task_id, start_cycles, end_cycles,
 *                receive_to_start_cycles]
 * aicpu_tasks:  [core_id, reg_task_id, dispatch_cycles, finish_cycles]
 * The trailing receive_to_start_cycles is what lets the per-task head overhead
 * be split into AICPU->AICore NoC propagation (hardware) and the AICore-local
 * dcci + ack pair (software).  Schema per repo/simpler swimlane_converter.py. */
function readRecords(dir) {
  const d = rj(dir + '/l2_swimlane_records.json');
  const hz = d.metadata.clock_freq_hz;
  const c2u = 1e6 / hz;
  const types = d.metadata.core_types;
  const rows = d.aicore_tasks || [];
  const width = rows.length ? rows[0].length : 0;
  /* one entry per task token; a task spread over N cores contributes N rows */
  const byTok = new Map();
  let base = Infinity;
  rows.forEach((r) => { if (r[3] < base) base = r[3]; });
  (d.aicpu_orchestrator_phases || []).forEach((ph) => ph.forEach((p) => {
    if (p.start_cycles && p.start_cycles < base) base = p.start_cycles;
  }));
  rows.forEach((r) => {
    const k = String(r[1]);
    let t = byTok.get(k);
    if (!t) {
      t = {
        tok: k, core: r[0], reg: r[2], kind: types[r[0]],
        start: r[3], end: r[4], recv: r[5] || 0, blocks: 0, cores: new Set(),
      };
      byTok.set(k, t);
    }
    if (r[3] < t.start) t.start = r[3];
    if (r[4] > t.end) t.end = r[4];
    t.recv = Math.max(t.recv, r[5] || 0);
    t.blocks++;
    t.cores.add(r[0]);
  });
  /* scheduler side, joined on (core_id, reg_task_id) as the converter does */
  const sched = {};
  (d.aicpu_tasks || []).forEach((r) => { sched[r[0] + '#' + r[1]] = { disp: r[2], fin: r[3] }; });
  const orch = {};
  (d.aicpu_orchestrator_phases || []).forEach((ph) => ph.forEach((p) => {
    orch[String(p.task_id)] = p.submit_idx;
  }));
  return {
    hz: hz, c2u: c2u, base: base, width: width, types: types,
    numCores: d.metadata.num_cores, level: d.l2_swimlane_level,
    blockRows: rows.length, tasks: Array.from(byTok.values()), sched: sched, orch: orch,
    rows: rows,
    schedPhases: d.aicpu_scheduler_phases || [],
    orchPhases: d.aicpu_orchestrator_phases || [],
  };
}

/* -------------------------------------------------------- merged trace reader
 * Only tp1 ships one. Worker View (pid 4) = one event per block execution,
 * Scheduler View (pid 3) = the AICPU-side dispatch -> finish of the same block. */
function readMerged(dir, file) {
  const ev = rj(dir + '/' + file).traceEvents;
  const threadName = {};
  ev.filter((e) => e.cat === '__metadata' && e.name === 'thread_name')
    .forEach((e) => { threadName[e.pid + ':' + e.tid] = e.args.name; });
  const hint = (e) => e.args['event-hint'] || '';
  const worker = ev.filter((e) => e.pid === 4 && e.cat === 'event' && e.ph === 'X'
    && /Task:/.test(hint(e)) && e.args['duration-us'] !== undefined);
  const schedView = ev.filter((e) => e.pid === 3 && e.cat === 'event' && e.ph === 'X'
    && e.args['dispatch-time-us'] !== undefined);
  const orch = {};
  ev.filter((e) => e.cat === 'orchestrator').forEach((e) => {
    orch[String(e.args.task_id)] = e.args.submit_idx;
  });
  const byTok = new Map();
  worker.forEach((e) => {
    const k = String(e.args.taskId);
    let t = byTok.get(k);
    if (!t) {
      t = {
        tok: k, name: e.name.replace(/\(.*\)$/, ''),
        funcId: +((hint(e).match(/FuncId:(-?\d+)/) || [])[1] || -1),
        startUs: e.ts, endUs: e.ts + e.dur, blocks: 0, cores: new Set(),
        busyUs: 0, kernelUs: 0, setupUs: 0, kinds: new Set(),
      };
      byTok.set(k, t);
    }
    const lane = threadName['4:' + e.tid] || '';
    t.blocks++;
    t.cores.add(+((hint(e).match(/CoreId:(\d+)/) || [])[1] || -1));
    t.kinds.add(lane.indexOf('AIC') === 0 ? 'aic' : 'aiv');
    t.busyUs += e.dur;
    t.kernelUs += e.args['kernel-duration-us'] || 0;
    t.setupUs += e.args['local_setup_us'] || 0;
    if (e.ts < t.startUs) t.startUs = e.ts;
    if (e.ts + e.dur > t.endUs) t.endUs = e.ts + e.dur;
    /* mixed scopes: the AIC and AIV halves share one taskId */
    if (t.name !== e.name.replace(/\(.*\)$/, '')) t.mixed = true;
  });
  const sched = {};
  schedView.forEach((e) => {
    const k = String(e.args.taskId);
    const s = sched[k] || (sched[k] = { disp: Infinity, fin: -Infinity });
    if (e.args['dispatch-time-us'] < s.disp) s.disp = e.args['dispatch-time-us'];
    if (e.args['finish-time-us'] > s.fin) s.fin = e.args['finish-time-us'];
  });
  const laneOf = {};
  worker.forEach((e) => {
    const c = +((hint(e).match(/CoreId:(\d+)/) || [])[1] || -1);
    const lane = threadName['4:' + e.tid] || '';
    laneOf[c] = lane;
  });
  /* the AICPU's own two lanes: scheduler phases (pid 2) and submits (pid 1) */
  const schedEv = ev.filter((e) => e.pid === 2 && e.cat === 'scheduler' && e.ph === 'X');
  const orchEv = ev.filter((e) => e.cat === 'orchestrator' && e.ph === 'X');
  return {
    events: ev, threadName: threadName, worker: worker, orch: orch, sched: sched,
    tasks: Array.from(byTok.values()), laneOf: laneOf,
    schedEv: schedEv, orchEv: orchEv,
    schedLanes: Array.from(new Set(schedEv.map((e) => e.tid))).length,
    spanUs: Math.max.apply(null, worker.map((e) => e.ts + e.dur)),
  };
}

/* ====================================================================== L2 */
function laneStatsFromRecords(rec, spanUs) {
  const busy = {};
  const n = {};
  rec.rows.forEach((r) => {
    busy[r[0]] = (busy[r[0]] || 0) + (r[4] - r[3]) * rec.c2u;
    n[r[0]] = (n[r[0]] || 0) + 1;
  });
  return rec.types.map((kind, core) => ({
    core: core, kind: kind, n: n[core] || 0,
    busyUs: r2(busy[core] || 0),
    util: spanUs ? r3(((busy[core] || 0) / spanUs) * 100) : null,
  }));
}

function headSplit(rec) {
  /* dispatch -> receive : AICPU writes the task id, AICore reads it (NoC)
   * receive  -> start   : the AICore-local dcci + ack pair (software)
   * start    -> end     : the kernel
   * end      -> finish  : completion reported back to the scheduler          */
  const noc = [];
  const dcci = [];
  const kern = [];
  const tail = [];
  let joined = 0;
  rec.rows.forEach((r) => {
    const s = rec.sched[r[0] + '#' + r[2]];
    const recvCyc = r[5] || 0;
    dcci.push(recvCyc * rec.c2u);
    kern.push((r[4] - r[3]) * rec.c2u);
    if (!s) return;
    joined++;
    noc.push(((r[3] - recvCyc) - s.disp) * rec.c2u);
    tail.push((s.fin - r[4]) * rec.c2u);
  });
  const kernelUs = sum(kern);
  return {
    source: 'l2_swimlane_records · aicore_tasks + aicpu_tasks 按 (core_id, reg_task_id) 联结',
    joinable: joined,
    blockRows: rec.rows.length,
    hasReceiveColumn: rec.width >= 6,
    kernelCoreUs: r2(kernelUs),
    noc: stats(noc, kernelUs),
    dcci: stats(dcci, kernelUs),
    kernel: stats(kern, kernelUs),
    tail: stats(tail, kernelUs),
  };
}

/* The merged trace carries the Scheduler View's dispatch -> finish, but not the
 * receive timestamp, so it can only give the head overhead as one lump. */
function headFromMerged(merged) {
  const queue = [];
  const tail = [];
  const kern = [];
  merged.worker.forEach((e) => {
    kern.push(e.dur);
  });
  merged.tasks.forEach((t) => {
    const s = merged.sched[t.tok];
    if (!s || !isFinite(s.disp)) return;
    queue.push(t.startUs - s.disp);
    tail.push(s.fin - t.endUs);
  });
  const kernelUs = sum(kern);
  return {
    source: 'merged swimlane · Scheduler View dispatch/finish 对 Worker View 起止',
    joinable: queue.length,
    blockRows: merged.worker.length,
    hasReceiveColumn: false,
    kernelCoreUs: r2(kernelUs),
    noc: null,
    dcci: null,
    queue: stats(queue, kernelUs),
    kernel: stats(kern, kernelUs),
    tail: stats(tail, kernelUs),
  };
}

function durHistogram(vals) {
  const v = sortNum(vals);
  if (!v.length) return [];
  /* log buckets: the distributions span four decades in every capture */
  const edges = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, Infinity];
  const out = [];
  for (let i = 0; i < edges.length - 1; i++) {
    const n = v.filter((x) => x >= edges[i] && x < edges[i + 1]).length;
    if (!n) continue;
    out.push({
      lo: edges[i], hi: edges[i + 1] === Infinity ? null : edges[i + 1],
      n: n, share: r3((n / v.length) * 100),
    });
  }
  return out;
}

/* ============================================================== torch layer */
function readTorch(cap) {
  const dir = cap.torch + '/ASCEND_PROFILER_OUTPUT';
  const rd = (f) => parseCsv(fs.readFileSync(at(dir + '/' + f), 'utf8'));
  const info = (() => {
    const cands = fs.readdirSync(at(cap.torch)).filter((f) => /^profiler_info/.test(f));
    return cands.length ? rj(cap.torch + '/' + cands[0]) : null;
  })();

  /* --- per-step device timeline. Step 0 is the single prefill, the rest are
   * the configured decode steps: that is what the collector launches.      */
  const stepRows = rd('step_trace_time.csv');
  const steps = stepRows.map((r, i) => {
    const comp = nf(r.Computing) || 0;
    const free = nf(r.Free) || 0;
    const prep = nf(r.Preparing) || 0;
    const stage = nf(r.Stage) || (comp + free);
    return {
      step: nf(r.Step),
      stage: i === 0 ? 'prefill' : 'decode',
      deviceId: nf(r.Device_id),
      stageUs: r2(stage), computingUs: r2(comp), freeUs: r2(free), preparingUs: r2(prep),
      commUs: r2(nf(r['Communication(Not Overlapped)']) || 0),
      freeShare: r3((free / stage) * 100),
      computeShare: r3((comp / stage) * 100),
    };
  });

  /* --- every invocation of the PyPTO fused kernel, with its PMU counters.
   * The *_ratio columns are per-pipe busy fractions of aicore_time and
   * overlap, so they do not partition the kernel: they are read one at a
   * time, never summed.                                                    */
  const kd = rd('kernel_details.csv');
  const fusedName = 'aicore_kernel_0';
  let fi = 0;
  const fused = kd.filter((r) => r.Name === fusedName).map((r) => {
    const stage = fi++ === 0 ? 'prefill' : 'decode';
    return {
      step: nf(r['Step Id']), stage: stage,
      durUs: r2(nf(r['Duration(us)'])), waitUs: r2(nf(r['Wait Time(us)'])),
      blocks: nf(r['Block Num']), mixBlocks: nf(r['Mix Block Num']),
      core: r['Accelerator Core'],
      aicTimeUs: r2(nf(r['aicore_time(us)'])), aivTimeUs: r2(nf(r['aiv_time(us)'])),
      aicCycles: nf(r.aic_total_cycles), aivCycles: nf(r.aiv_total_cycles),
      cubeUtil: r3(nf(r['cube_utilization(%)'])),
      aic: {
        mac: r3(nf(r.aic_mac_ratio)), scalar: r3(nf(r.aic_scalar_ratio)),
        mte1: r3(nf(r.aic_mte1_ratio)), mte2: r3(nf(r.aic_mte2_ratio)),
        fixpipe: r3(nf(r.aic_fixpipe_ratio)), icacheMiss: r3(nf(r.aic_icache_miss_rate)),
        macUs: r2(nf(r['aic_mac_time(us)'])), mte2Us: r2(nf(r['aic_mte2_time(us)'])),
        scalarUs: r2(nf(r['aic_scalar_time(us)'])), mte1Us: r2(nf(r['aic_mte1_time(us)'])),
      },
      aiv: {
        vec: r3(nf(r.aiv_vec_ratio)), scalar: r3(nf(r.aiv_scalar_ratio)),
        mte2: r3(nf(r.aiv_mte2_ratio)), mte3: r3(nf(r.aiv_mte3_ratio)),
        icacheMiss: r3(nf(r.aiv_icache_miss_rate)),
        vecUs: r2(nf(r['aiv_vec_time(us)'])), scalarUs: r2(nf(r['aiv_scalar_time(us)'])),
      },
    };
  });

  /* --- what else ran on the device, i.e. the framework ops the fused kernel
   * did NOT absorb. simpler_aicpu_exec_* is the AI_CPU launcher for the same
   * kernel, so it is marked rather than counted as separate work.          */
  const opRows = rd('op_statistic.csv');
  const ops = opRows.map((r) => ({
    type: r['OP Type'], core: r['Core Type'],
    count: nf(r.Count), totalUs: r2(nf(r['Total Time(us)'])),
    avgUs: r3(nf(r['Avg Time(us)'])), maxUs: r2(nf(r['Max Time(us)'])),
    ratio: r3(nf(r['Ratio(%)'])),
    role: r['OP Type'] === fusedName ? 'fused'
      : /^simpler_aicpu_exec/.test(r['OP Type']) ? 'launcher' : 'framework',
  }));
  const frameworkUs = sum(ops.filter((o) => o.role === 'framework').map((o) => o.totalUs));
  const fusedUs = sum(ops.filter((o) => o.role === 'fused').map((o) => o.totalUs));

  /* --- the host bill. Level=acl rows are host-side CANN calls; the stream
   * synchronize row is the host waiting for the device, not host work, so it
   * is kept separate instead of inflating the total.                       */
  const apiRows = rd('api_statistic.csv');
  const api = apiRows.map((r) => ({
    name: r['API Name'], level: r.Level,
    totalUs: r2(nf(r['Time(us)'])), count: nf(r.Count),
    avgUs: r3(nf(r['Avg(us)'])), maxUs: r2(nf(r['Max(us)'])),
    isWait: /Synchronize/i.test(r['API Name']),
  })).sort((a, b) => b.totalUs - a.totalUs);
  const apiWork = api.filter((a) => a.level === 'acl' && !a.isWait);
  const apiWaitUs = sum(api.filter((a) => a.isWait).map((a) => a.totalUs));

  const decodeSteps = steps.filter((s) => s.stage === 'decode');
  const mean = (a, k) => (a.length ? sum(a.map((x) => x[k])) / a.length : null);

  return {
    profDir: cap.torch,
    deviceId: steps.length ? steps[0].deviceId : null,
    torchVersion: info && info.config ? (info.config.common || {}).torch_version || null : null,
    iters: { prefill: 1, decode: decodeSteps.length },
    steps: steps,
    decodeMean: {
      stageUs: r2(mean(decodeSteps, 'stageUs')),
      computingUs: r2(mean(decodeSteps, 'computingUs')),
      freeUs: r2(mean(decodeSteps, 'freeUs')),
      preparingUs: r2(mean(decodeSteps, 'preparingUs')),
      freeShare: r3(mean(decodeSteps, 'freeShare')),
      computeShare: r3(mean(decodeSteps, 'computeShare')),
    },
    fused: fused,
    ops: ops,
    deviceMix: {
      fusedUs: r2(fusedUs), frameworkUs: r2(frameworkUs),
      fusedShare: r3((fusedUs / (fusedUs + frameworkUs)) * 100),
      frameworkOps: ops.filter((o) => o.role === 'framework').length,
      frameworkCalls: sum(ops.filter((o) => o.role === 'framework').map((o) => o.count)),
    },
    api: api.slice(0, 24),
    apiHost: { workUs: r2(sum(apiWork.map((a) => a.totalUs))), waitUs: r2(apiWaitUs), rows: apiWork.length },
    kernelRows: kd.length,
  };
}

/* ====================================================================== run */
const l2 = {};
const layers = {};
const variants = [];
const torch = {};

Object.keys(CAPTURES).forEach((cid) => {
  const cap = CAPTURES[cid];
  torch[cid] = readTorch(cap);
  STAGES.forEach((stage) => {
    const vid = VID(cid, stage);
    const dir = cap.swim[stage];
    const rec = readRecords(dir);
    const mergedFile = cap.merged[stage];
    const merged = mergedFile ? readMerged(dir, mergedFile) : null;

    /* the swimlane the reader is looking at: the merged trace when the capture
     * ships one (it is the complete run), otherwise the records */
    const recSpanUs = rec.rows.length
      ? (Math.max.apply(null, rec.rows.map((r) => r[4])) - rec.base) * rec.c2u : 0;
    const spanUs = merged ? merged.spanUs : recSpanUs;
    /* tp1's prefill records hold only the last 62 tasks of the ring buffer;
     * the merged trace is the complete capture, so say which one answers what */
    const recordsComplete = !merged || rec.rows.length >= merged.worker.length * 0.9;

    const taskList = merged ? merged.tasks : rec.tasks;
    const durs = merged
      ? merged.tasks.map((t) => t.endUs - t.startUs)
      : rec.tasks.map((t) => (t.end - t.start) * rec.c2u);
    const busyUs = merged
      ? sum(merged.worker.map((e) => e.dur))
      : sum(rec.rows.map((r) => (r[4] - r[3]) * rec.c2u));

    /* ----------------------------------------------------- layer recovery */
    const submitOf = merged
      ? ((t) => merged.orch[t.tok])
      : ((t) => rec.orch[t.tok]);
    const withSubmit = taskList.filter((t) => submitOf(t) !== undefined);
    const ordered = withSubmit.slice().sort((a, b) => submitOf(a) - submitOf(b));
    const split = ordered.length ? splitLayers(ordered, submitOf) : null;

    const startOf = (t) => (merged ? t.startUs : (t.start - rec.base) * rec.c2u);
    const endOf = (t) => (merged ? t.endUs : (t.end - rec.base) * rec.c2u);
    const kindOf = (t) => {
      if (!merged) return t.kind;
      const k = Array.from(t.kinds);
      return k.length > 1 ? 'mix' : k[0];
    };
    const nameOf = (t, stepIdx) => {
      if (merged && t.name && t.name !== 'task' && t.name !== 'task_spmd') return t.name;
      /* tp2: the collector validates a 16-step layout, so the step index is a
       * name. tp1 decode: 265 steps per layer with nothing to name them. */
      if (split && split.per === 16 && stepIdx != null) return LAYER_STEPS_16[stepIdx][0];
      return null;
    };

    let layerInfo = null;
    if (split) {
      /* validate against the collector's layout when the shape matches */
      let layoutOk = null;
      if (split.per === 16) {
        layoutOk = split.layers.every((L) => L.every((t, i) => kindOf(t) === LAYER_STEPS_16[i][1]));
      }
      const rows = split.layers.map((L, i) => {
        const s = Math.min.apply(null, L.map(startOf));
        const e = Math.max.apply(null, L.map(endOf));
        const body = bodyStart(L.map(startOf));
        const perTask = L.map((t) => endOf(t) - startOf(t));
        const aicUs = sum(L.filter((t) => kindOf(t) === 'aic').map((t) => endOf(t) - startOf(t)));
        const aivUs = sum(L.filter((t) => kindOf(t) === 'aiv').map((t) => endOf(t) - startOf(t)));
        const worst = L.reduce((a, t, j) => {
          const d = endOf(t) - startOf(t);
          return a && a.d >= d ? a : { d: d, i: j, name: nameOf(t, split.per === 16 ? j : null) };
        }, null);
        /* per-task offsets inside the layer, so the screen can draw the
         * layer-unwrapped swimlane the dataset's own PNG draws. Only for the
         * coarse layouts: 265 tasks x 40 layers is not a readable row. */
        const blocks = split.per <= 32 ? L.map((t, j) => [
          r2(startOf(t) - body.at), r2(endOf(t) - startOf(t)),
          kindOf(t) === 'aic' ? 1 : 0, j,
        ]) : null;
        return {
          layer: i, startUs: r2(s), endUs: r2(e),
          spanUs: r2(e - s), blocks: blocks,
          /* the layer's own window, with early-dispatched stragglers excluded */
          bodyStartUs: r2(body.at), bodyUs: r2(e - body.at),
          earlyTasks: body.early, earlyLeadUs: r2(body.leadUs || 0),
          busyUs: r2(sum(perTask)),
          tasks: L.length, aicUs: r2(aicUs), aivUs: r2(aivUs),
          worstUs: r2(worst.d), worstStep: worst.i, worstName: worst.name,
        };
      });
      const spans = sortNum(rows.map((r) => r.bodyUs));
      const med = quant(spans, 0.5);
      const stepStats = split.per <= 32 ? Array.from({ length: split.per }, (_, j) => {
        const vals = split.layers.map((L) => endOf(L[j]) - startOf(L[j]));
        const v = sortNum(vals);
        const st = sum(vals);
        return {
          step: j,
          name: split.per === 16 ? LAYER_STEPS_16[j][0] : (nameOf(split.layers[0][j], null) || ('step ' + j)),
          kind: kindOf(split.layers[0][j]),
          meanUs: r3(st / 40), p50Us: r3(quant(v, 0.5)),
          minUs: r3(v[0]), maxUs: r3(v[v.length - 1]),
          spreadX: v[0] > 0 ? r3(v[v.length - 1] / v[0]) : null,
          sumUs: r2(st),
          share: r3((st / sum(rows.map((r) => r.busyUs))) * 100),
        };
      }) : null;
      layerInfo = {
        method: '按 AICPU orchestrator 提交序切层：层与层之间固定有 ' + (split.gap - 1)
          + ' 个不落到 AICore 的循环控制任务，于是相邻 AICore 任务的 submit_idx 跳 '
          + split.gap + ' 的情况恰好出现 40 次，且两次之间恒隔 ' + split.per
          + ' 个任务——这 40 个跳变就是层边界。',
        verified: !split.ambiguous,
        layoutChecked: layoutOk,
        layerCount: 40, perLayer: split.per,
        headTasks: split.head.length, tailTasks: split.tail.length,
        outsideNames: split.head.concat(split.tail)
          .map((t) => nameOf(t, null) || ('task ' + t.tok)),
        steps: stepStats,
        rows: rows,
        steady: {
          metric: 'bodyUs',
          medianSpanUs: r2(med),
          minSpanUs: r2(spans[0]), maxSpanUs: r2(spans[spans.length - 1]),
          spreadPct: med ? r3(((spans[spans.length - 1] - spans[0]) / med) * 100) : null,
          iqrUs: r2(quant(spans, 0.75) - quant(spans, 0.25)),
          firstDeltaPct: med ? r3(((rows[0].bodyUs - med) / med) * 100) : null,
          earlyLayers: rows.filter((r) => r.earlyTasks > 0).length,
          earlyTasksMax: Math.max.apply(null, rows.map((r) => r.earlyTasks)),
        },
        outliers: rows.slice()
          .map((r) => ({ layer: r.layer, spanUs: r.bodyUs, deltaPct: r3(((r.bodyUs - med) / med) * 100) }))
          .sort((a, b) => Math.abs(b.deltaPct) - Math.abs(a.deltaPct))
          .slice(0, 6),
      };
      layers[vid] = layerInfo;
    }

    /* layer index per task, for the top table */
    const layerOfTok = {};
    const stepOfTok = {};
    if (split) {
      split.layers.forEach((L, i) => L.forEach((t, j) => {
        layerOfTok[t.tok] = i; stepOfTok[t.tok] = j;
      }));
    }

    /* ------------------------------------------------------------- lanes */
    const lanes = merged
      ? (() => {
        const busy = {};
        const n = {};
        merged.worker.forEach((e) => {
          const c = +(((e.args['event-hint'] || '').match(/CoreId:(\d+)/) || [])[1]);
          busy[c] = (busy[c] || 0) + e.dur;
          n[c] = (n[c] || 0) + 1;
        });
        return rec.types.map((kind, core) => ({
          core: core, kind: kind, n: n[core] || 0, busyUs: r2(busy[core] || 0),
          util: r3(((busy[core] || 0) / spanUs) * 100),
        }));
      })()
      : laneStatsFromRecords(rec, spanUs);
    const used = lanes.filter((l) => l.n > 0);
    const byKind = (k, arr) => arr.filter((l) => l.kind === k);
    const utilOf = (arr) => (arr.length
      ? r3(sum(arr.map((l) => l.busyUs)) / (arr.length * spanUs) * 100) : null);

    /* ------------------------------------------------- top tasks + blocks */
    const top = ordered.slice()
      .sort((a, b) => (endOf(b) - startOf(b)) - (endOf(a) - startOf(a)))
      .slice(0, 30)
      .map((t) => {
        const s = merged ? merged.sched[t.tok] : rec.sched[t.core + '#' + t.reg];
        const st = startOf(t);
        const en = endOf(t);
        return {
          tok: t.tok, reg: merged ? null : t.reg,
          name: nameOf(t, stepOfTok[t.tok]),
          kind: kindOf(t), layer: layerOfTok[t.tok] == null ? null : layerOfTok[t.tok],
          step: stepOfTok[t.tok] == null ? null : stepOfTok[t.tok],
          blocks: merged ? t.blocks : t.blocks,
          cores: merged ? t.cores.size : t.cores.size,
          startUs: r2(st), durUs: r2(en - st),
          dcciUs: merged ? null : r3((t.recv || 0) * rec.c2u),
          nocUs: (!merged && s) ? r3(((t.start - (t.recv || 0)) - s.disp) * rec.c2u) : null,
          tailUs: merged
            ? (s ? r3(s.fin - en) : null)
            : (s ? r3((s.fin - t.end) * rec.c2u) : null),
        };
      });

    /* A compact lane/block form for every variant that data.js does not carry
     * a full rank for: enough for an occupancy swimlane, no task objects. */
    const blocks = (() => {
      if (stage === 'prefill' && cid === 'tp1') return null;   /* in data.js */
      const rows = rec.rows.map((r) => [r[0], r2((r[3] - rec.base) * rec.c2u), r2((r[4] - r[3]) * rec.c2u)]);
      return { laneCount: rec.numCores, rows: rows };
    })();

    /* ------------------------------------------------------- AICPU phases
     * From the merged trace where there is one, because tp1's prefill records
     * keep only the ring buffer's last 62 tasks and no phases at all. */
    const phaseAgg = {};
    const addPhase = (kind, us) => {
      const a = phaseAgg[kind] || (phaseAgg[kind] = { kind: kind, count: 0, busy: 0 });
      a.count++;
      a.busy += us;
    };
    let schedLaneCount;
    let orchBusy;
    let orchN;
    let phaseSource;
    if (merged && !recordsComplete) {
      merged.schedEv.forEach((e) => addPhase(e.args.phase || e.name.replace(/\(.*\)$/, ''), e.dur));
      schedLaneCount = merged.schedLanes;
      orchBusy = sum(merged.orchEv.map((e) => e.dur));
      orchN = merged.orchEv.length;
      phaseSource = 'merged swimlane · Scheduler View / AICPU Orchestrator';
    } else {
      rec.schedPhases.forEach((ph) => ph.forEach((p) => {
        addPhase(p.kind, (p.end_cycles - p.start_cycles) * rec.c2u);
      }));
      schedLaneCount = rec.schedPhases.filter((p) => p.length).length;
      orchBusy = sum(rec.orchPhases.map((ph) =>
        sum(ph.map((p) => (p.end_cycles - p.start_cycles) * rec.c2u))));
      orchN = sum(rec.orchPhases.map((ph) => ph.length));
      phaseSource = 'l2_swimlane_records · aicpu_scheduler_phases / aicpu_orchestrator_phases';
    }
    const schedBusy = sum(Object.keys(phaseAgg).map((k) => phaseAgg[k].busy));

    l2[vid] = {
      spanUs: r2(spanUs),
      busyUs: r2(busyUs),
      taskCount: taskList.length,
      blockRows: merged ? merged.worker.length : rec.blockRows,
      recordRows: rec.blockRows,
      recordsComplete: recordsComplete,
      coreTotal: rec.numCores,
      coresUsed: used.length,
      clockHz: rec.hz,
      swimlaneLevel: rec.level,
      occ: {
        aicUtil: utilOf(byKind('aic', lanes)), aivUtil: utilOf(byKind('aiv', lanes)),
        aicUsedUtil: utilOf(byKind('aic', used)), aivUsedUtil: utilOf(byKind('aiv', used)),
        aicCores: byKind('aic', lanes).length, aivCores: byKind('aiv', lanes).length,
        aicUsed: byKind('aic', used).length, aivUsed: byKind('aiv', used).length,
        /* core time / wall time = how many cores were busy on average.
         * ~1 means the graph ran one task at a time. */
        busyCores: r3(busyUs / spanUs),
      },
      lanes: lanes,
      head: (merged && !recordsComplete) ? headFromMerged(merged)
        : (rec.rows.length ? headSplit(rec) : null),
      dur: Object.assign(stats(durs, null) || {}, { hist: durHistogram(durs) }),
      top: top,
      sched: {
        source: phaseSource,
        phases: Object.keys(phaseAgg).map((k) => ({
          kind: k, count: phaseAgg[k].count, busyUs: r2(phaseAgg[k].busy),
          share: schedBusy ? r3((phaseAgg[k].busy / schedBusy) * 100) : null,
        })).sort((a, b) => b.busyUs - a.busyUs),
        lanes: schedLaneCount,
        busyUs: r2(schedBusy),
        /* several AICPU threads share the lane budget, so this can exceed 100% */
        util: r3((schedBusy / spanUs) * 100),
      },
      orch: {
        submits: orchN, busyUs: r2(orchBusy),
        meanUs: orchN ? r3(orchBusy / orchN) : null,
        util: r3((orchBusy / spanUs) * 100),
        aicoreTasks: Object.keys(rec.orch).length,
      },
      blocks: blocks,
    };

    const named = merged
      ? merged.tasks.some((t) => t.name && t.name !== 'task' && t.name !== 'task_spmd')
      : !!(split && split.per === 16);
    variants.push({
      id: vid, capture: cid, stage: stage,
      label: cap.label + ' · ' + (stage === 'prefill' ? 'prefill' : 'decode'),
      sub: cap.sub,
      validated: cap.validated,
      primary: vid === 'tp1:prefill',
      spanUs: r2(spanUs),
      taskCount: taskList.length,
      blockRows: merged ? merged.worker.length : rec.blockRows,
      coresUsed: used.length,
      coreTotal: rec.numCores,
      hasMergedTrace: !!merged,
      hasNames: named,
      namesFrom: merged && named ? '采集自带 FuncId / kernel 名'
        : named ? '采集无名；按采集脚本已校验的 40×16 层布局还原步名' : null,
      hasDeps: !!(merged && merged.tasks.length && cid === 'tp1' && stage === 'prefill'),
      hasReceiveSplit: rec.width >= 6 && rec.rows.length > 100,
      hasLayers: !!split,
      gaps: [
        merged ? null : '无 merged swimlane：没有 Worker / Scheduler View，'
          + '任务只有 (core, token, reg_id, 起止, receive_to_start) 五类事实',
        named ? null : '采集里 FuncId 全为 -1：任务没有 kernel 名',
        (merged && merged.tasks.length && cid === 'tp1' && stage === 'prefill')
          ? null : '没有依赖图：关键路径、依赖连线、scope 归因在本采集上不成立',
        recordsComplete ? null : 'l2_swimlane_records 只留下 ' + rec.blockRows
          + ' 条记录（环形缓冲尾部），头开销拆解在本采集上样本不足',
      ].filter(Boolean),
    });
  });
});

/* ============================================================== topology */
function tcmp(label, get, unit, note) {
  return {
    metric: label, unit: unit || null, note: note || null,
    tp1: get('tp1'), tp2: get('tp2'),
  };
}
const topology = {
  rows: [
    tcmp('decode step（torch Stage）', (c) => torch[c].decodeMean.stageUs, 'us', 'step_trace_time.csv · decode 步均值'),
    tcmp('其中 Computing', (c) => torch[c].decodeMean.computingUs, 'us'),
    tcmp('其中 Free（设备空闲）', (c) => torch[c].decodeMean.freeUs, 'us'),
    tcmp('Free 占比', (c) => torch[c].decodeMean.freeShare, '%'),
    tcmp('融合 kernel 单次时长', (c) => {
      const d = torch[c].fused.filter((f) => f.stage === 'decode');
      return r2(sum(d.map((f) => f.durUs)) / d.length);
    }, 'us', 'kernel_details.csv · aicore_kernel_0 decode 均值'),
    tcmp('AIC mac 占比', (c) => {
      const d = torch[c].fused.filter((f) => f.stage === 'decode');
      return r3(sum(d.map((f) => f.aic.mac)) / d.length);
    }, '×', '每流水线相对 aicore_time 的忙占比，互相重叠，不可相加'),
    tcmp('AIC mte2 占比', (c) => {
      const d = torch[c].fused.filter((f) => f.stage === 'decode');
      return r3(sum(d.map((f) => f.aic.mte2)) / d.length);
    }, '×'),
    tcmp('AIC scalar 占比', (c) => {
      const d = torch[c].fused.filter((f) => f.stage === 'decode');
      return r3(sum(d.map((f) => f.aic.scalar)) / d.length);
    }, '×'),
    tcmp('AIV scalar 占比', (c) => {
      const d = torch[c].fused.filter((f) => f.stage === 'decode');
      return r3(sum(d.map((f) => f.aiv.scalar)) / d.length);
    }, '×'),
    tcmp('AIV vec 占比', (c) => {
      const d = torch[c].fused.filter((f) => f.stage === 'decode');
      return r3(sum(d.map((f) => f.aiv.vec)) / d.length);
    }, '×'),
    tcmp('decode 泳道 span', (c) => l2[VID(c, 'decode')].spanUs, 'us', 'l2_swimlane_records / merged swimlane'),
    tcmp('decode 任务数', (c) => l2[VID(c, 'decode')].taskCount, null, '同一张 40 层图，任务粒度不同'),
    tcmp('decode 用到的核', (c) => l2[VID(c, 'decode')].coresUsed, '/ 60'),
    tcmp('decode AIC 占用（全部核）', (c) => l2[VID(c, 'decode')].occ.aicUtil, '%'),
    tcmp('decode 平均忙核数', (c) => l2[VID(c, 'decode')].occ.busyCores, '核',
      '核时 / 墙钟；≈1 表示一次只有一个核在算'),
    tcmp('每层任务数', (c) => (layers[VID(c, 'decode')] || {}).perLayer, null),
    tcmp('框架算子（融合 kernel 之外）', (c) => torch[c].deviceMix.frameworkCalls, '次',
      'op_statistic.csv · 5 步合计'),
    tcmp('主机 ACL 调用耗时', (c) => torch[c].apiHost.workUs, 'us',
      'api_statistic.csv · 不含 Synchronize 等待'),
  ],
};

/* ========================================================== compile notes
 * The 2026-08-14 build ships perf_hints and binary_context but no passes_dump,
 * which is why data.js keeps reading the 2026-06-25 build for the compiler and
 * ISA layers. These are the new build's own hints, kept separate from those. */
function readHints(rel) {
  const p = at(rel + '/report/perf_hints.log');
  if (!fs.existsSync(p)) return null;
  const lines = fs.readFileSync(p, 'utf8').split(/\r?\n/).filter((l) => l.trim());
  const sites = {};
  const codes = {};
  lines.forEach((l) => {
    const code = (l.match(/\[perf_hint ([A-Z0-9-]+)\]/) || [])[1];
    if (!code) return;
    const loc = l.match(/ at (\S+?):(\d+):(\d+)\s*$/);
    const occ = +((l.match(/\((\d+) occurrences/) || [])[1] || 1);
    const rule = (l.match(/\]\s+(\w+):/) || [])[1] || null;
    const file = loc ? loc[1].replace(/^.*\/models\//, 'models/') : '(no location)';
    const line = loc ? +loc[2] : null;
    const key = code + '@' + file + ':' + line;
    const s = sites[key] || (sites[key] = {
      code: code, rule: rule, file: file, module: file.replace(/^.*\//, ''),
      line: line, n: 0, occ: 0, sample: l.replace(/^\[perf_hint [^\]]+\]\s*/, '').slice(0, 260),
    });
    s.n++;
    s.occ += occ;
    codes[code] = (codes[code] || 0) + 1;
  });
  const arr = Object.keys(sites).map((k) => sites[k]).sort((a, b) => b.occ - a.occ || b.n - a.n);
  return {
    lines: lines.length,
    byCode: Object.keys(codes).sort().map((c) => ({ code: c, n: codes[c] })),
    byFile: (() => {
      const f = {};
      arr.forEach((s) => {
        const e = f[s.file] || (f[s.file] = { file: s.file, module: s.module, sites: 0, occ: 0, codes: {} });
        e.sites++;
        e.occ += s.occ;
        e.codes[s.code] = (e.codes[s.code] || 0) + 1;
      });
      return Object.keys(f).map((k) => f[k]).sort((a, b) => b.occ - a.occ);
    })(),
    sites: arr.slice(0, 40),
  };
}
const compile = {
  source: 'scratch/tp1_build_output/_jit_{prefill,decode}_fwd_20260814_*',
  note: '这是 2026-08-14 那次构建自己的 perf_hints 与 binary_context。'
    + '同一次构建没有 passes_dump，所以「编译器」与「ISA / 布局」两层读的仍是 2026-06-25 的旧构建。',
  binaryContext: (() => {
    const p = CAPTURES.tp1.build.prefill + '/cache/binary_context.json';
    return fs.existsSync(at(p)) ? rj(p) : null;
  })(),
  hints: {
    prefill: readHints(CAPTURES.tp1.build.prefill),
    decode: readHints(CAPTURES.tp1.build.decode),
  },
};

/* ================================================================ findings
 * The console's working unit is a queue entry, so the new layers have to land
 * there too or they stay decoration. These are hygiene items, not chains: the
 * captures that produce them carry no dependency graph, so none of them can
 * claim a share of makespan. Each one says which capture it belongs to and
 * what it cannot attribute.                                                 */
const F = [];
function finding(o) {
  F.push({
    id: o.id, kind: 'hygiene', level: o.level, severity: o.severity || 'med',
    axis: o.axis, title: o.title, metric: o.metric, cost: null,
    variant: o.variant,
    unattributed: o.unattributed,
    claim: o.claim,
    evidence: o.evidence,
    focus: { view: o.view, panel: o.panel, variant: o.variant },
    lever: o.lever, guardrail: o.guardrail, verify: o.verify,
    subjects: {
      view: o.view, variant: o.variant, panel: o.panel,
      tab: null, overlay: null, tasks: [], lanes: [], sites: [], files: [],
      ranks: [], schedPhases: [], absent: false,
    },
    chips: [],
    levels: [o.level],
    rootPass: null,
  });
}
const t1 = torch.tp1;
const t2 = torch.tp2;
const d1 = l2['tp1:decode'];
const d2 = l2['tp2:decode'];
const dec1 = t1.fused.filter((f) => f.stage === 'decode');
const dec2 = t2.fused.filter((f) => f.stage === 'decode');
const avg1 = (get) => r3(sum(dec1.map(get)) / dec1.length);
const avg2 = (get) => r3(sum(dec2.map(get)) / dec2.length);
const ly2 = layers['tp2:decode'];
const worstStep2 = ly2 && ly2.steps ? ly2.steps.slice().sort((a, b) => b.sumUs - a.sumUs)[0] : null;

finding({
  id: 'Q1', level: 'e2e', severity: 'high', axis: 'host', view: 'e2e', panel: 'step',
  variant: 'tp1:decode',
  title: 'TP=1 decode 每步 ' + r2(t1.decodeMean.stageUs / 1000) + ' ms 里设备空闲 '
    + r2(t1.decodeMean.freeUs / 1000) + ' ms',
  metric: 'Computing ' + r2(t1.decodeMean.computingUs / 1000) + ' ms / Free '
    + r2(t1.decodeMean.freeUs / 1000) + ' ms（' + t1.decodeMean.freeShare + '%）',
  unattributed: 'step_trace_time 只说设备空闲了多久，不说主机那一侧具体在等什么；'
    + '本次采集没有 host STRACE log，接不上 span 树。',
  claim: '4 个 decode step 的 Free 占比分别是 '
    + t1.steps.filter((s) => s.stage === 'decode').map((s) => s.freeShare + '%').join(' / ')
    + '，稳定在六成上下。同一轮里融合 kernel 之外还跑了 ' + t1.deviceMix.frameworkCalls
    + ' 次框架算子，主机侧 ACL 调用合计 ' + r2(t1.apiHost.workUs / 1000)
    + ' ms（另有 ' + r2(t1.apiHost.waitUs / 1000) + ' ms 是等设备，不算主机开销）。'
    + '在这个比例下，把 kernel 再快一倍也只能省下步时间的四成。',
  evidence: [
    { artifact: 'ASCEND_PROFILER_OUTPUT/step_trace_time.csv', locator: 'decode 步均值',
      value: 'Stage ' + r2(t1.decodeMean.stageUs / 1000) + ' ms = Computing '
        + r2(t1.decodeMean.computingUs / 1000) + ' + Free ' + r2(t1.decodeMean.freeUs / 1000) + ' ms' },
    { artifact: 'ASCEND_PROFILER_OUTPUT/api_statistic.csv', locator: '主机 acl 调用合计',
      value: r2(t1.apiHost.workUs / 1000) + ' ms / ' + t1.apiHost.rows + ' 条' },
    { artifact: 'ASCEND_PROFILER_OUTPUT/op_statistic.csv', locator: '融合 kernel 之外的算子',
      value: t1.deviceMix.frameworkOps + ' 种 / ' + t1.deviceMix.frameworkCalls + ' 次' },
  ],
  lever: '先把主机侧的 per-layer 小算子合并或前移，再谈 kernel 内部；'
    + '融合 kernel 已经占掉设备时间的 ' + t1.deviceMix.fusedShare + '%，那一侧的余量有限。',
  guardrail: '合并主机算子会改变 KV cache 的写入时机，必须先验证输出 token 一致；'
    + 'Free 是设备侧空闲，不能直接当成主机可省的时间，两者只能互相印证方向。',
  verify: '重测 step_trace_time.csv 的 Free 占比，并核对 api_statistic 里被合并调用的次数确实下降。',
});

finding({
  id: 'Q2', level: 'l1', severity: 'high', axis: 'pipeline', view: 'l1', panel: 'pmu',
  variant: 'tp1:decode',
  title: 'TP=1 decode 的 AIC 卡在搬运：mte2 占比 ' + avg1((f) => f.aic.mte2)
    + '，mac 只有 ' + avg1((f) => f.aic.mac),
  metric: 'mte2 ' + avg1((f) => f.aic.mte2) + ' / mte1 ' + avg1((f) => f.aic.mte1)
    + ' / mac ' + avg1((f) => f.aic.mac) + ' / fixpipe ' + avg1((f) => f.aic.fixpipe),
  unattributed: '流水占比是相对 aicore_time 的忙占比，互相重叠；'
    + '它能说明哪条流水线最忙，不能折成墙钟上的一段。',
  claim: '4 次 decode 调用里，AIC 的 GM→L1 搬运（mte2）忙占比稳定在 '
    + Math.min.apply(null, dec1.map((f) => f.aic.mte2)) + '–'
    + Math.max.apply(null, dec1.map((f) => f.aic.mte2))
    + '，而矩阵乘（mac）只有 ' + Math.min.apply(null, dec1.map((f) => f.aic.mac)) + '–'
    + Math.max.apply(null, dec1.map((f) => f.aic.mac))
    + '。decode 每步只出一个 token，权重却要整张搬一遍，这正是 mte2 主导的形状。'
    + ' prefill 那次调用的 mac 是 ' + dec1.length + ' 次 decode 的好几倍（'
    + t1.fused[0].aic.mac + '），同一份 kernel 两种阶段的瓶颈不在一处。',
  evidence: [
    { artifact: 'ASCEND_PROFILER_OUTPUT/kernel_details.csv', locator: 'aicore_kernel_0 · decode',
      value: 'aic_mte2_ratio ' + avg1((f) => f.aic.mte2) + ' vs aic_mac_ratio ' + avg1((f) => f.aic.mac) },
    { artifact: 'ASCEND_PROFILER_OUTPUT/kernel_details.csv', locator: '同一 kernel 的 prefill 调用',
      value: 'aic_mac_ratio ' + t1.fused[0].aic.mac + ' / aic_mte2_ratio ' + t1.fused[0].aic.mte2 },
  ],
  lever: '按 mte2 去调：加大搬运末维、提高 L1 的流水深度、复用已在片上的权重块；'
    + '动 Cube 侧的 tile 形状对这一层没有帮助。',
  guardrail: '加深流水会抬高 L1 / UB 占用，可能反过来把 depth 压回 1（见编译器层的 PH-MR-001）；'
    + 'PMU 开着采的这一轮不能与 PMU 关闭的基线比墙钟。',
  verify: '复测同一 step 的 aic_mte2_ratio 与 aicore_kernel_0 时长，两者要一起降。',
});

finding({
  id: 'Q3', level: 'l1', severity: 'med', axis: 'pipeline', view: 'l1', panel: 'pmu',
  variant: 'tp1:decode',
  title: '40 个 AIV 核在做标量：vec 占比 ' + avg1((f) => f.aiv.vec)
    + '，scalar ' + avg1((f) => f.aiv.scalar),
  metric: 'vec ' + avg1((f) => f.aiv.vec) + ' / scalar ' + avg1((f) => f.aiv.scalar)
    + ' / mte2 ' + avg1((f) => f.aiv.mte2) + ' / mte3 ' + avg1((f) => f.aiv.mte3),
  unattributed: 'PMU 只说各流水线忙不忙，不说标量在算什么；'
    + '本次采集没有 AIV 侧的指令级剖分，落不到具体源码。',
  claim: 'AIV 的向量流水忙占比只有 ' + avg1((f) => f.aiv.vec) + '，而标量流水是 '
    + avg1((f) => f.aiv.scalar) + '，两者差两个数量级。泳道上 AIV 的核占用是 '
    + d1.occ.aivUtil + '%——核确实被占着，占着的时间几乎都不是在做向量计算。'
    + '这不等于 AIV 空转：地址计算、循环控制、同步都会计到标量流水上。',
  evidence: [
    { artifact: 'ASCEND_PROFILER_OUTPUT/kernel_details.csv', locator: 'aicore_kernel_0 · decode',
      value: 'aiv_vec_ratio ' + avg1((f) => f.aiv.vec) + ' / aiv_scalar_ratio ' + avg1((f) => f.aiv.scalar) },
    { artifact: 'l2_swimlane_records.json', locator: 'tp1 decode · AIV 占用',
      value: d1.occ.aivUtil + '%（' + d1.occ.aivCores + ' 核）' },
  ],
  lever: '先确认这些 AIV 任务本来该做什么：如果是搬运 / 归约，改成向量化实现；'
    + '如果只是控制流，考虑把它们从 AIV 上挪走。',
  guardrail: '标量占比高也可能是正常的同步等待，动手前要先定位到具体 scope，'
    + '否则会把一个正常的控制核当成性能问题改。',
  verify: '改完后看 aiv_vec_ratio 是否上升、aiv_scalar_ratio 是否下降，同时核对 AIV 核占用没有变差。',
});

finding({
  id: 'Q4', level: 'l2', severity: 'high', axis: 'dispatch', view: 'l2', panel: 'head',
  variant: 'tp1:decode',
  title: 'decode 的任务太小：dispatch→receive 每块 ' + d1.head.noc.meanUs
    + ' us，合计相当于内核核时的 ' + d1.head.noc.shareOfKernel + '%',
  metric: 'NoC ' + d1.head.noc.meanUs + ' us / dcci+ack ' + d1.head.dcci.meanUs
    + ' us / 内核 ' + d1.head.kernel.meanUs + ' us / 回报 ' + d1.head.tail.meanUs + ' us',
  unattributed: '头开销是每块一次的固定成本，落在墙钟上的哪一段取决于当时有没有别的块在跑，'
    + '本采集没有依赖图，算不出它真正拖长了多少 makespan。',
  claim: 'receive_to_start_cycles 把头开销拆成两段：AICPU→AICore 的 NoC 传播 '
    + d1.head.noc.meanUs + ' us（硬件侧）和 AICore 本地的 dcci + ack '
    + d1.head.dcci.meanUs + ' us（软件侧）。内核本身平均只有 ' + d1.head.kernel.meanUs
    + ' us，所以每块的 dispatch 到 finish 里有相当一部分不是计算。'
    + '对照 TP=2：同一张图切成 643 个大任务时，NoC 只有 '
    + d2.head.noc.meanUs + ' us，占内核核时 ' + d2.head.noc.shareOfKernel + '%。'
    + '任务粒度决定了这笔开销值不值得管。',
  evidence: [
    { artifact: 'l2_swimlane_records.json', locator: 'aicore_tasks[5] = receive_to_start_cycles',
      value: 'NoC ' + d1.head.noc.meanUs + ' us / dcci+ack ' + d1.head.dcci.meanUs + ' us · '
        + d1.head.joinable + ' 块' },
    { artifact: 'l2_swimlane_records.json', locator: '内核时长分布',
      value: 'p50 ' + d1.dur.p50Us + ' us / 均值 ' + d1.dur.meanUs + ' us · ' + d1.taskCount + ' 任务' },
    { artifact: 'tp2 l2_swimlane_records.json', locator: '同图粗粒度对照',
      value: 'NoC ' + d2.head.noc.meanUs + ' us（占内核核时 ' + d2.head.noc.shareOfKernel + '%）' },
  ],
  lever: '合并相邻的小任务，或者把同一 scope 的多次派发折成一次多块派发，'
    + '让固定头开销摊到更长的内核时间上。dcci + ack 那一段是软件可调的，NoC 传播不是。',
  guardrail: '合并任务会降低调度灵活度，可能让关键路径上的任务更晚开始；'
    + '本采集没有依赖图，合并之前要在有依赖图的采集上确认拓扑。',
  verify: '重采后比较 receive_to_start 的合计与内核核时之比，以及泳道 span 是否真的缩短。',
});

if (worstStep2) {
  finding({
    id: 'Q5', level: 'l2', severity: 'high', axis: 'parallelism', view: 'l2', panel: 'swimlane',
    variant: 'tp2:decode',
    title: 'TP=2 基本没并起来：平均只有 ' + d2.occ.busyCores + ' 个核在忙，用到 '
      + d2.coresUsed + '/' + d2.coreTotal,
    metric: '核时 ' + r2(d2.busyUs / 1000) + ' ms / 墙钟 ' + r2(d2.spanUs / 1000)
      + ' ms · ' + ly2.perLayer + ' 任务 × 40 层',
    unattributed: '本采集没有任务名与依赖图，说不出是依赖链逼着串行，还是每个任务本来就只发了一个块。',
    claim: '643 个任务的核时合计 ' + r2(d2.busyUs / 1000) + ' ms，墙钟 '
      + r2(d2.spanUs / 1000) + ' ms，比值 ' + d2.occ.busyCores
      + '——几乎是一个接一个跑完。60 个核里只有 ' + d2.coresUsed + ' 个落上过任务（'
      + d2.occ.aicUsed + ' AIC / ' + d2.occ.aivUsed + ' AIV）。'
      + '每层 ' + r2(ly2.steady.medianSpanUs / 1000) + ' ms 中，' + worstStep2.name
      + ' 一步就占 ' + r2(worstStep2.meanUs / 1000) + ' ms（' + worstStep2.share
      + '% 的层内核时）。同一张图在 TP=1 那次采集里平均有 ' + d1.occ.busyCores + ' 个核在忙。',
    evidence: [
      { artifact: 'tp2 l2_swimlane_records.json', locator: '核时 / 墙钟',
        value: d2.occ.busyCores + ' 核 · 用到 ' + d2.coresUsed + '/' + d2.coreTotal },
      { artifact: 'tp2 l2_swimlane_records.json', locator: '层内最重的一步',
        value: worstStep2.name + ' 均值 ' + r2(worstStep2.meanUs / 1000) + ' ms / 占层内核时 '
          + worstStep2.share + '%' },
      { artifact: 'ASCEND_PROFILER_OUTPUT/kernel_details.csv', locator: 'TP=2 decode PMU',
        value: 'aic_mac_ratio ' + avg2((f) => f.aic.mac) + ' / aic_scalar_ratio '
          + avg2((f) => f.aic.scalar) },
    ],
    lever: '先确认 ' + worstStep2.name + ' 这一步是不是只发了一个块：如果是，把它 pl.spmd 展开到多核；'
      + 'PMU 上 AIC scalar 占比 ' + avg2((f) => f.aic.scalar) + '、mac 只有 '
      + avg2((f) => f.aic.mac) + '，更像在等而不是在算，同步路径也要一起看。',
    guardrail: '展开会增加派发次数，头开销会随之上升（见 Q4）；'
      + 'TP=2 与 TP=1 是两个采集脚本的产物，不能拿 TP=1 的绝对数字当 TP=2 的目标。',
    verify: '重采后看核时 / 墙钟是否从 ' + d2.occ.busyCores + ' 抬起来，以及用到的核数是否超过 '
      + d2.coresUsed + '。',
  });
}

/* ===================================================================== out */
const payload = {
  generatedBy: 'Design/operator-tuning-console/build-qwen3-profiles.cjs',
  source: 'Data/pypto_qwen3_profiles.zip（解压后 Data/pypto_qwen3_profiles/）',
  capturedAt: '2026-08-14',
  dataset: {
    validatedCapture: 'tp2',
    excluded: ['tp1/', 'scratch/'],
    note: '数据集 README 只对 tp2 给出验收标记，.gitignore 把 tp1/ 与 scratch/ 排除在交付之外。'
      + '两次采集来自同一台机器同一天，但用的是两个采集脚本，不能当作跨机器性能基线。',
  },
  captures: Object.keys(CAPTURES).reduce((o, k) => {
    const c = CAPTURES[k];
    o[k] = {
      id: c.id, label: c.label, sub: c.sub, collector: c.collector,
      validated: c.validated, validateNote: c.validateNote,
      hasBuildOutput: !!c.build.prefill,
      torchDevice: torch[k].deviceId,
      iters: torch[k].iters,
    };
    return o;
  }, {}),
  variants: variants,
  l2: l2,
  layers: layers,
  torch: torch,
  topology: topology,
  compile: compile,
  findings: F,
};

const out = '/* Generated by ' + payload.generatedBy + ' — do not edit.\n'
  + ' * Source: ' + payload.source + '\n'
  + ' *\n'
  + ' * Adds the layers build-data.cjs has no concept of: torch host/device\n'
  + ' * attribution, measured PMU pipe occupancy, the 40-layer unwrap, the\n'
  + ' * AICPU->AICore head overhead split, and the TP=1 / TP=2 comparison.\n'
  + ' */\n'
  + '(function () {\n'
  + "  'use strict';\n"
  + '  const run = window.TUNING_RUNS && window.TUNING_RUNS.decode_fwd_layers;\n'
  + '  if (!run) return;\n'
  + '  run.qwen3 = ' + JSON.stringify(payload) + ';\n'
  + '}());\n';
fs.writeFileSync(OUT, out);

/* ------------------------------------------------------------------ report */
variants.forEach((v) => {
  const L = layers[v.id];
  console.log(v.id.padEnd(12)
    + ' span ' + String(v.spanUs).padStart(10) + ' us'
    + ' | tasks ' + String(v.taskCount).padStart(6)
    + ' | blocks ' + String(v.blockRows).padStart(6)
    + ' | cores ' + String(v.coresUsed).padStart(2) + '/' + v.coreTotal
    + ' | AIC ' + String(l2[v.id].occ.aicUtil).padStart(6) + '%'
    + ' | 层 ' + (L ? '40x' + L.perLayer + (L.verified ? '' : '?') : '—')
    + (v.hasNames ? ' | 有名' : ' | 无名')
    + (v.validated ? ' | 已校验' : ''));
});
Object.keys(torch).forEach((k) => {
  const t = torch[k];
  console.log(k + ' torch: decode step ' + t.decodeMean.stageUs + ' us = Computing '
    + t.decodeMean.computingUs + ' + Free ' + t.decodeMean.freeUs
    + '（' + t.decodeMean.freeShare + '%）| 融合 kernel 占设备 '
    + t.deviceMix.fusedShare + '% | 框架算子 ' + t.deviceMix.frameworkCalls + ' 次'
    + ' | 主机 ACL ' + t.apiHost.workUs + ' us');
});
console.log('wrote ' + OUT + ' ' + (fs.statSync(OUT).size / 1024).toFixed(1) + ' KB');
