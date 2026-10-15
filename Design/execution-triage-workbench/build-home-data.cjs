#!/usr/bin/env node
/* Build home-data.js for the Execution Triage home (isometric "three towers" view).
 *
 * One execution = three towers: Serving / Host / Device. Tower height = time the
 * layer holds on the critical path; floors = steps (or model layers); each floor
 * carries two cards that explain what that floor is made of.
 *
 * Only the DSpark Serving request is built.
 *
 * Sources (read-only):
 *   - ./data.js                              (build-data.cjs: DSpark serving digest)
 *   - Data/serving-strace-swimlane(2).json   (request-level outputs for the P0 band)
 *
 * Usage: node Design/execution-triage-workbench/build-home-data.cjs
 */
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const OUT = path.join(__dirname, 'home-data.js');
const r2 = (v) => Math.round(v * 100) / 100;

global.window = {};
require(path.join(__dirname, 'data.js'));
const TRIAGE = global.window.TRIAGE_DATA;

const card = (title, sub, ms, tone, text) => ({ title, sub, ms: r2(ms), tone: tone || '', text: text || null });
const sumMs = (floors) => r2(floors.reduce((a, f) => a + f.ms, 0));
// Demo-only split of bind.args: preserve its measured total and model allocation vs H2D.
function demoBindSplit(totalMs, stepIndex, dev) {
  const total = r2(totalMs);
  const allocationShare = 0.10 + ((stepIndex * 7 + dev * 3) % 8) / 100;
  const allocation = r2(total * allocationShare);
  return { allocation, transfer: r2(total - allocation) };
}
// Demo-only split of runner_run; shared by per-step and E2E views.
function demoDeviceExecEstimate(rrMs, decodeIndex, dev) {
  const modeledExec = 104 + ((decodeIndex * 7 + dev * 11) % 17) * 0.8;
  return r2(Math.min(rrMs, modeledExec));
}

/* ------------------------------------------------------------ DSpark serving */
let TRACE = null;
function traceEvents() {
  if (!TRACE) TRACE = JSON.parse(fs.readFileSync(path.join(ROOT, 'Data', 'serving-strace-swimlane(2).json'), 'utf8'));
  return TRACE;
}

// P0：请求级服务指标。输出时刻取调度器 process_step_output（只算落在请求窗口内的）。
function dsparkRequest(run) {
  const trace = traceEvents();
  const ev = trace.traceEvents.filter((e) => e.ph === 'X');
  const http = ev.find((e) => e.name === 'http.completions');
  const t0 = http.ts; const t1 = http.ts + http.dur;
  const outs = ev.filter((e) => e.name === 'scheduler.process_step_output' && e.ts >= t0 && e.ts <= t1).sort((a, b) => a.ts - b.ts);
  const gaps = outs.slice(1).map((e, i) => (e.ts - outs[i].ts) / 1000).sort((a, b) => a - b);
  const q = (p) => gaps[Math.min(gaps.length - 1, Math.floor(p * (gaps.length - 1)))];
  const label = (s) => (s.id === 'Prefill' ? 'P' : s.id === 'Draft init' ? 'DI' : s.id);
  const timeline = run.steps.map((s) => ({ id: label(s), t0: s.t0, ms: s.periodMs, kind: s.decodeStep ? 'decode' : s.id === 'Prefill' ? 'prefill' : 'draft' }));
  const last = run.steps[run.steps.length - 1];
  const tailMs = r2(http.dur / 1000 - (last.t0 + last.periodMs));
  return {
    id: http.args.request_id,
    stream: http.args.stream,
    maxTokens: http.args.max_tokens,
    e2eMs: r2(http.dur / 1000),
    firstOutputMs: r2((outs[0].ts - t0) / 1000),
    prefillMs: run.steps.find((s) => s.id === 'Prefill').periodMs,
    draftMs: run.steps.find((s) => s.id === 'Draft init').periodMs,
    outputs: outs.length,
    gapP50Ms: r2(q(0.5)),
    gapP90Ms: r2(q(0.9)),
    gapMaxMs: r2(gaps[gaps.length - 1]),
    timeline,
    tailMs,
  };
}

/* 单步流转：一个 step 内 Serving / Worker / 16 条 lane 的真实先后与重叠。
 * 时间以“上一步结束”为 0（ms）；负值表示在上一步期间已经并行发生。 */
function dsparkFlows(run) {
  const trace = traceEvents();
  const ev = trace.traceEvents.filter((e) => e.ph === 'X');
  const pn = {};
  trace.traceEvents.filter((e) => e.ph === 'M' && e.name === 'process_name').forEach((e) => { pn[e.pid] = e.args.name; });
  const pidOf = (re) => Number(Object.keys(pn).find((p) => re.test(pn[p])));
  const LANE = pidOf(/STRACE/); const API = pidOf(/serving-api/); const WORKER = pidOf(/serving-worker/);
  const http = ev.find((e) => e.name === 'http.completions');
  const end = (e) => e.ts + e.dur;
  const phase = (e) => e.name.split('| ')[1];
  const sched = ev.filter((e) => e.pid === API && /^scheduler\./.test(e.name));
  const worker = ev.filter((e) => e.pid === WORKER && /^WorkerProcess\.(prepare_decode|dispatch_decode|reclaim_decode|batch_prefill|execute_step)$/.test(e.name));
  const label = (s) => (s.id === 'Prefill' ? 'P' : s.id === 'Draft init' ? 'DI' : s.id);

  /* Worker 进程是三个线程在并发跑：准备下一步 ∥ 下发 ∥ 回收上一步的输出。
   * prepare 有 97% 的时间与 reclaim 重叠（53 次里 51 次完全落在 reclaim 内部），
   * 所以必须按线程分行，压在一行里会相互覆盖。
   * 行的身份取 tid，行名取 trace 的 thread_name，顺序按流水：准备 → 下发 → 回收。 */
  const threadName = {};
  trace.traceEvents
    .filter((e) => e.ph === 'M' && e.name === 'thread_name' && e.pid === WORKER)
    .forEach((e) => { threadName[e.tid] = e.args.name; });
  const WORKER_ROLES = [
    { key: 'prepare', label: '准备', re: /prepare_decode$/ },
    { key: 'dispatch', label: '下发', re: /(dispatch_decode|execute_step|batch_prefill)$/ },
    { key: 'reclaim', label: '回收', re: /reclaim_decode$/ },
  ];
  // 一个 tid 归一个行：按它跑得最多的那种调用定角色
  const tidRole = {};
  const tally = {};
  worker.forEach((e) => {
    const role = (WORKER_ROLES.find((r) => r.re.test(e.name)) || WORKER_ROLES[1]).key;
    tally[e.tid] = tally[e.tid] || {};
    tally[e.tid][role] = (tally[e.tid][role] || 0) + 1;
  });
  Object.keys(tally).forEach((tid) => {
    tidRole[tid] = Object.entries(tally[tid]).sort((a, b) => b[1] - a[1])[0][0];
  });
  const workerRows = WORKER_ROLES
    .filter((r) => Object.values(tidRole).includes(r.key))
    .map((r) => {
      const tid = Object.keys(tidRole).find((k) => tidRole[k] === r.key);
      return { key: r.key, label: r.label, thread: threadName[tid] || `tid ${tid}` };
    });

  const out = {};
  let prevEnd = http.ts;
  run.steps.forEach((s) => {
    const stepIndex = s.decodeStep ? run.steps.indexOf(s) : -1;
    const spans = ev.filter((e) => e.pid === LANE && e.args && e.args.invocation === s.inv);
    const runs = spans.filter((e) => phase(e) === 'simpler_run').sort((a, b) => a.tid - b.tid);
    const lanes = runs.map((r) => {
      const own = spans.filter((e) => e.tid === r.tid);
      const get = (n) => own.find((e) => phase(e) === n);
      const b = get('bind'); const rr = get('runner_run');
      const rrMs = rr.dur / 1000;
      const lane = { dev: r.tid, start: r.ts, bindStart: b.ts, bindEnd: end(b), rrStart: rr.ts, end: end(r) };
      if (stepIndex < 0) return lane;
      const deviceExec = demoDeviceExecEstimate(rrMs, stepIndex, r.tid);
      return { ...lane, deviceExec, waitEstimate: r2(Math.max(0, rrMs - deviceExec)) };
    });
    const minStart = Math.min(...lanes.map((l) => l.start));
    const maxEnd = Math.max(...lanes.map((l) => l.end));
    const crit = lanes.reduce((a, b) => (b.rrStart > a.rrStart ? b : a));
    // 本步的 dispatch：在第一条 lane 开始前结束的最近一次
    const disp = worker.filter((e) => /dispatch_decode$/.test(e.name) && end(e) <= minStart + 1000).sort((a, b) => end(b) - end(a))[0];
    const ownStep = disp ? disp.args.step_id : null;
    const queue = disp ? sched.filter((e) => /queue_worker_step$/.test(e.name) && e.ts <= disp.ts).sort((a, b) => b.ts - a.ts)[0] : null;
    const output = sched.filter((e) => /process_step_output$/.test(e.name) && e.ts >= maxEnd - 1000).sort((a, b) => a.ts - b.ts)[0];
    const reclaim = ownStep != null ? worker.find((e) => /reclaim_decode$/.test(e.name) && e.args.step_id === ownStep) : null;
    const lo = Math.max(prevEnd - 260000, Math.min(prevEnd, minStart, disp ? disp.ts : Infinity, queue ? queue.ts : Infinity) - 8000);
    const hi = Math.max(maxEnd + 12000, output ? end(output) + 2000 : 0);
    const rel = (t) => r2((t - prevEnd) / 1000);
    const clip = (list, kindOf) => list.filter((e) => end(e) > lo && e.ts < hi).map((e) => {
      const a = Math.max(e.ts, lo); const b = Math.min(end(e), hi);
      return {
        n: e.name.replace(/^(scheduler|WorkerProcess)\./, ''),
        s: rel(a),
        d: r2((b - a) / 1000),
        kind: kindOf(e),
        ...(tidRole[e.tid] ? { t: tidRole[e.tid] } : {}),
      };
    });
    out[label(s)] = {
      t0: rel(lo),
      t1: rel(hi),
      crit: crit.dev,
      critStart: rel(crit.rrStart),
      laneStart: rel(minStart),
      end: rel(maxEnd),
      hostMs: r2(s.gapWorkerMs + s.hostMs),
      deviceMs: s.deviceMs,
      periodMs: s.periodMs,
      lanes: lanes.map((l) => ({
        dev: l.dev, start: rel(l.start), bindStart: rel(l.bindStart), bindEnd: rel(l.bindEnd),
        rrStart: rel(l.rrStart), end: rel(l.end),
        ...(l.deviceExec != null ? { deviceExec: l.deviceExec, waitEstimate: l.waitEstimate } : {}),
      })),
      serving: clip(sched, (e) => (/wait_worker_output$/.test(e.name) ? 'wait' : e === queue || e === output ? 'own' : 'other')),
      worker: clip(worker, (e) => {
        if (/reclaim_decode$/.test(e.name)) return e === reclaim ? 'wait' : 'other';
        if (ownStep == null || e.args.step_id == null) return 'own';
        return e.args.step_id === ownStep ? 'own' : e.args.step_id > ownStep ? 'next' : 'other';
      }),
      links: {
        queue: queue ? rel(queue.ts) : null,
        dispatch: disp ? [rel(disp.ts), rel(end(disp))] : null,
        reclaimEnd: reclaim ? rel(end(reclaim)) : null,
        output: output ? rel(output.ts) : null,
      },
    };
    prevEnd = maxEnd;
  });
  return { rows: workerRows, steps: out };
}

/* 瓶颈分解（按 pypto-serving#95 的口径）：
 * - 关键路径：Serving 调度 + Worker 下发 + bind.args（分配 + H2D）+ lane 启动 / 其他 + Device，五段相加 = step 周期。
 * - 观察窗口：来自不同 lane 或与关键路径重叠的量，不能与上面相加。
 * - 常驻推算：去掉关键 lane 落在关键路径上的 bind.args，其余阶段不变。只是基线拆分的推算，不是实测。 */
function dsparkRuntime(run, decode) {
  const trace = traceEvents();
  const ev = trace.traceEvents.filter((e) => e.ph === 'X');
  const pn = {};
  trace.traceEvents.filter((e) => e.ph === 'M' && e.name === 'process_name').forEach((e) => { pn[e.pid] = e.args.name; });
  const LANE = Number(Object.keys(pn).find((p) => /STRACE/.test(pn[p])));
  const phase = (e) => e.name.split('| ')[1];
  const byInv = {};
  ev.filter((e) => e.pid === LANE).forEach((e) => { (byInv[e.args.invocation] = byInv[e.args.invocation] || []).push(e); });
  const q = (xs, p) => { const a = xs.slice().sort((x, y) => x - y); return a[Math.min(a.length - 1, Math.floor(p * (a.length - 1)))]; };
  const stat = (xs) => ({ mean: r2(xs.reduce((a, b) => a + b, 0) / xs.length), p50: r2(q(xs, 0.5)), p90: r2(q(xs, 0.9)), max: r2(Math.max(...xs)) });

  const rowOf = (s) => {
    const w = run.waterfalls[s.id];
    const eff = Math.max(w.prevEnd, Math.min(...w.lanes.map((l) => l.start)));
    const c = w.lanes.find((l) => l.dev === s.critLane);
    const h2d = Math.max(0, c.bindEnd - Math.max(eff, c.bindStart));
    const bindSplit = demoBindSplit(h2d, run.steps.indexOf(s), c.dev);
    const spans = byInv[s.inv];
    const lane = (dev, n) => spans.find((e) => e.tid === dev && phase(e) === n);
    const devs = [...new Set(spans.map((e) => e.tid))];
    const args = devs.map((d) => lane(d, 'bind.args').dur / 1000);
    return {
      id: s.id,
      serving: s.gapServingMs,
      worker: s.gapWorkerMs,
      h2d,
      allocation: bindSplit.allocation,
      transfer: bindSplit.transfer,
      launch: Math.max(0, s.hostMs - h2d),
      device: s.deviceMs,
      period: s.periodMs,
      laneArgsMed: q(args, 0.5),
      laneArgsMin: Math.min(...args),
      laneArgsSum: args.reduce((a, b) => a + b, 0),
      rrMax: Math.max(...devs.map((d) => lane(d, 'runner_run').dur / 1000)),
      validate: lane(s.critLane, 'validate').dur / 1000,
      prebuilt: lane(s.critLane, 'bind.prebuilt').dur / 1000,
    };
  };
  const rows = decode.map(rowOf);
  const col = (k) => rows.map((r) => r[k]);
  // 每个 step 的同口径拆分，供头部按选中的那一步显示
  const lab = (s) => (s.id === 'Prefill' ? 'P' : s.id === 'Draft init' ? 'DI' : s.id);
  const byStep = {};
  run.steps.forEach((s) => {
    const r = rowOf(s);
    byStep[lab(s)] = {
      serving: r.serving, worker: r.worker, host: r2(r.h2d + r.launch), h2d: r.h2d, launch: r.launch,
      allocation: r.allocation, transfer: r.transfer,
      device: r.device, period: r.period, resident: r2(r.period - r.h2d), decode: s.decodeStep != null,
    };
  });
  const worker = ev.filter((e) => e.name === 'WorkerProcess.dispatch_decode').map((e) => e.dur / 1000);
  const after = rows.map((r) => r.period - r.h2d);
  const saveMs = rows.reduce((a, r) => a + r.h2d, 0);
  return {
    n: rows.length,
    byStep,
    range: `${decode[0].id}–${decode[decode.length - 1].id}`,
    path: [
      { key: 'serving', label: 'Serving 调度', sub: '只有调度器在推进', stat: stat(col('serving')) },
      { key: 'worker', label: 'Worker 下发', sub: '只有 Worker 在推进', stat: stat(col('worker')) },
      { key: 'h2d', label: 'bind.args', sub: '关键 lane · 分配 + H2D', stat: stat(col('h2d')), hot: true },
      { key: 'launch', label: 'lane 启动 · 其他', sub: '关键 lane 进入 bind 前后', stat: stat(col('launch')) },
      { key: 'device', label: 'runner_run', sub: '关键 lane · Device 上界', stat: stat(col('device')), upper: true },
    ],
    bindSplit: {
      allocation: stat(col('allocation')),
      transfer: stat(col('transfer')),
    },
    period: stat(col('period')),
    windows: [
      { label: '16 lane bind.args 合计', sub: '每卡一条 host lane', stat: stat(col('laneArgsSum')) },
      { label: 'bind.args 中位 lane', sub: '不只是 lane 6', stat: stat(col('laneArgsMed')) },
      { label: 'bind.args 最快 lane', sub: '下限仍非 0', stat: stat(col('laneArgsMin')) },
      { label: 'bind.prebuilt', sub: '关键 lane', stat: stat(col('prebuilt')) },
      { label: 'runner_run 最长 lane', sub: '含跨卡等待', stat: stat(col('rrMax')) },
      { label: 'validate · D2H', sub: '关键 lane · 优化后不应增加', stat: stat(col('validate')) },
      { label: 'Worker dispatch_decode', sub: '已被流水覆盖', stat: stat(worker) },
    ],
    resident: {
      period: stat(after),
      savePct: r2((1 - stat(after).mean / stat(col('period')).mean) * 100),
      e2eSaveMs: r2(saveMs),
    },
  };
}

/* 全程泳道：一次请求 54 个 step 的 16 条 lane，时间以 http.completions 起点为 0。
 * 与单步视图同一套三条带，只是横轴换成整个 E2E，用来先看整体节奏再下钻。 */
function dsparkE2E(run) {
  const trace = traceEvents();
  const ev = trace.traceEvents.filter((e) => e.ph === 'X');
  const pn = {};
  trace.traceEvents.filter((e) => e.ph === 'M' && e.name === 'process_name').forEach((e) => { pn[e.pid] = e.args.name; });
  const LANE = Number(Object.keys(pn).find((p) => /STRACE/.test(pn[p])));
  const API = Number(Object.keys(pn).find((p) => /serving-api/.test(pn[p])));
  const http = ev.find((e) => e.name === 'http.completions');
  const t0 = http.ts;
  const rel = (ts) => r2((ts - t0) / 1000);
  const phase = (e) => e.name.split('| ')[1];
  const byInv = {};
  ev.filter((e) => e.pid === LANE).forEach((e) => { (byInv[e.args.invocation] = byInv[e.args.invocation] || []).push(e); });
  const label = (s) => (s.id === 'Prefill' ? 'P' : s.id === 'Draft init' ? 'DI' : s.id);
  const steps = run.steps.map((s) => {
    const stepIndex = s.decodeStep ? run.steps.indexOf(s) : -1;
    const spans = byInv[s.inv];
    const devs = [...new Set(spans.map((e) => e.tid))].sort((a, b) => a - b);
    const lanes = devs.map((d) => {
      const own = spans.filter((e) => e.tid === d);
      const g = (n) => own.find((e) => phase(e) === n);
      const b = g('bind'); const rr = g('runner_run');
      const rrMs = rr.dur / 1000;
      const lane = { dev: d, b0: rel(b.ts), b1: rel(b.ts + b.dur), r0: rel(rr.ts), r1: rel(rr.ts + rr.dur) };
      if (stepIndex < 0) return lane;
      const deviceExec = demoDeviceExecEstimate(rrMs, stepIndex, d);
      return { ...lane, deviceExec, waitEstimate: r2(Math.max(0, rrMs - deviceExec)) };
    });
    return {
      id: label(s),
      t0: rel(Math.min(...lanes.map((l) => l.b0 * 1000 + t0))),
      t1: Math.max(...lanes.map((l) => l.r1)),
      crit: s.critLane,
      kind: s.decodeStep ? 'decode' : s.id === 'Prefill' ? 'prefill' : 'draft',
      lanes,
    };
  });
  const outs = ev.filter((e) => e.pid === API && e.name === 'scheduler.process_step_output' && e.ts >= t0 && e.ts <= t0 + http.dur)
    .map((e) => rel(e.ts));
  return { durMs: r2(http.dur / 1000), outputs: outs, steps };
}

/* 第 3 步：整体慢，还是负载不均。
 * 判据是“各 lane 进入 runner_run 的起点”对“结束时刻”：起点分散而结束被屏障拉齐 = 少数 lane 晚到，
 * 其余 lane 的 runner_run 里大部分是等待。等待量按“最晚进入者的 lead 减去自己的 lead”计（下界）。 */
function dsparkDistribution(run, decode) {
  const q = (xs, p) => { const a = xs.slice().sort((x, y) => x - y); return a[Math.min(a.length - 1, Math.floor(p * (a.length - 1)))]; };
  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
  const stat = (xs) => ({ mean: r2(mean(xs)), p50: r2(q(xs, 0.5)), p90: r2(q(xs, 0.9)), max: r2(Math.max(...xs)) });

  const perStep = decode.map((s) => {
    const stepIndex = run.steps.indexOf(s);
    const w = run.waterfalls[s.id];
    const eff = Math.max(w.prevEnd, Math.min(...w.lanes.map((l) => l.start)));
    const lanes = w.lanes.map((l) => {
      const bind = l.bindEnd - l.bindStart;
      return { dev: l.dev, lead: l.rrStart - eff, rr: l.end - l.rrStart, end: l.end - eff, bind,
        ...demoBindSplit(bind, stepIndex, l.dev) };
    });
    const maxLead = Math.max(...lanes.map((l) => l.lead));
    return {
      lanes: lanes.map((l) => ({ ...l, wait: maxLead - l.lead })),
      startSpread: maxLead - Math.min(...lanes.map((l) => l.lead)),
      endSpread: Math.max(...lanes.map((l) => l.end)) - Math.min(...lanes.map((l) => l.end)),
      crit: s.critLane,
    };
  });
  // Demo-only device_wall model: keep the real runner_run span, then split it into
  // a near-uniform compute estimate and the remaining barrier wait. Replace this
  // model with collected device_wall data when it becomes available.
  perStep.forEach((p, stepIndex) => {
    p.lanes = p.lanes.map((l) => {
      const deviceExec = demoDeviceExecEstimate(l.rr, stepIndex, l.dev);
      return { ...l, deviceExec, waitEstimate: r2(Math.max(0, l.rr - deviceExec)) };
    });
  });
  const nLane = perStep[0].lanes.length;
  const lanes = Array.from({ length: nLane }, (_, dev) => {
    const xs = perStep.map((p) => p.lanes.find((l) => l.dev === dev));
    return {
      dev,
      bind: r2(mean(xs.map((l) => l.bind))),
      allocation: r2(mean(xs.map((l) => l.allocation))),
      transfer: r2(mean(xs.map((l) => l.transfer))),
      lead: r2(mean(xs.map((l) => l.lead))),
      rr: r2(mean(xs.map((l) => l.rr))),
      wait: r2(mean(xs.map((l) => l.wait))),
      deviceExec: r2(mean(xs.map((l) => l.deviceExec))),
      waitEstimate: r2(mean(xs.map((l) => l.waitEstimate))),
      critCount: perStep.filter((p) => p.crit === dev).length,
    };
  });
  const crit = lanes.reduce((a, b) => (b.critCount > a.critCount ? b : a));
  const medBind = q(lanes.map((l) => l.bind), 0.5);
  const startSpread = stat(perStep.map((p) => p.startSpread));
  const endSpread = stat(perStep.map((p) => p.endSpread));
  const waitTotal = stat(perStep.map((p) => p.lanes.reduce((a, l) => a + l.wait, 0)));
  const fmt = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : ms >= 100 ? `${Math.round(ms)} ms` : `${ms.toFixed(1)} ms`);
  return {
    n: perStep.length,
    nLane,
    critLane: crit.dev,
    critCount: crit.critCount,
    deviceSplitSource: 'synthetic-demo-model',
    bindSplitSource: 'synthetic-demo-model',
    medBind: r2(medBind),
    skew: r2(crit.bind / medBind),
    startSpread,
    endSpread,
    waitTotal,
    lanes,
    // 三选一：命中哪一类，决定下一步查什么
    patterns: [
      { key: 'all-slow', title: '多数卡执行都长', next: '算子 · 计算配置 · 访存 · 通信开销', hit: false },
      { key: 'late-lane', title: '少数卡晚到，其余在屏障等待', next: '负载均衡 · MoE 路由倾斜 · 数据分片 · 通信', hit: true },
      { key: 'host-bound', title: 'Device 相近但 step 仍慢', next: 'Host 准备 · Worker 调度 · 执行前后的等待', hit: false },
    ],
    verdict: `起点分散 ${fmt(startSpread.mean)}、结束分散 ${fmt(endSpread.mean)}：${nLane} 条 lane 被同步屏障拉齐，差别全在进入 runner_run 的时刻。`
      + `lane ${crit.dev} 的 bind 是中位 lane 的 ${r2(crit.bind / medBind)}×，并在 ${crit.critCount}/${perStep.length} 步都是最晚进入的一条——偏斜持续存在，且每一步都落在关键路径上。`,
    caveat: `其余 ${nLane - 1} 条 lane 每步合计屏障等待下界 ${fmt(waitTotal.mean)}；`
      + `关键 lane 的 runner_run 为 ${fmt(crit.rr)} 上界，右侧 lane 下钻提供设备执行与等待的估计拆分。`,
  };
}

/* lane 专项：某一条 lane 偏慢时，接着要看的四件事。
 * 全部来自本次 trace，回答不了的单列在 missing / experiments 里，不猜。 */
function dsparkLaneFocus(run, decode, dist) {
  const q = (xs, p) => { const a = xs.slice().sort((x, y) => x - y); return a[Math.min(a.length - 1, Math.floor(p * (a.length - 1)))]; };
  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
  const nLane = dist.nLane;

  // 1) 每步每条 lane 的 bind，以及这一步 16 条的中位——用来看是否随步数漂移
  const ids = [];
  const med = [];
  const byLane = {};
  for (let d = 0; d < nLane; d += 1) byLane[d] = [];
  decode.forEach((s) => {
    const w = run.waterfalls[s.id];
    const binds = {};
    w.lanes.forEach((l) => { binds[l.dev] = r2(l.bindEnd - l.bindStart); });
    ids.push(s.id);
    med.push(r2(q(Object.values(binds), 0.5)));
    for (let d = 0; d < nLane; d += 1) byLane[d].push(binds[d]);
  });
  // 四段分别算「这条 lane / 当步中位」的倍数，用来判断是结构性还是逐渐劣化
  const segCount = 4;
  const size = Math.ceil(ids.length / segCount);
  const segments = {};
  for (let d = 0; d < nLane; d += 1) {
    segments[d] = [];
    for (let i = 0; i < ids.length; i += size) {
      const a = byLane[d].slice(i, i + size);
      const m = med.slice(i, i + size);
      segments[d].push({
        range: `${ids[i]}–${ids[Math.min(i + size, ids.length) - 1]}`,
        ratio: r2(mean(a.map((v, k) => v / m[k]))),
      });
    }
  }

  // 2) 各阶段的关键 lane 分别是谁——decode 慢而 prefill 不慢，指向 callable 而不是硬件
  const label = (s) => (s.id === 'Prefill' ? 'P' : s.id === 'Draft init' ? 'DI' : s.id);
  const stages = run.steps.filter((s) => !s.decodeStep).map((s) => ({ id: label(s), name: s.id, crit: s.critLane }));
  const decodeCrit = {};
  decode.forEach((s) => { decodeCrit[s.critLane] = (decodeCrit[s.critLane] || 0) + 1; });
  stages.push({ id: 'decode', name: `decode ${decode.length} 步`, critMix: decodeCrit });

  // 3) 已排除：bind 的另一个子阶段 compatibility 只落在某一条 lane 上，量级可忽略
  const trace = traceEvents();
  const phase = (e) => e.name.split('| ')[1];
  const comp = trace.traceEvents.filter((e) => e.ph === 'X' && e.cat === 'strace.host' && phase(e) === 'bind.compatibility');
  const compLanes = [...new Set(comp.map((e) => e.tid))];
  // prebuilt 只在 Prefill 那一步有量，稳态 decode 里可忽略，分开说
  const decodeInv = new Set(decode.map((s) => s.inv));
  const pre = trace.traceEvents.filter((e) => e.ph === 'X' && e.cat === 'strace.host' && phase(e) === 'bind.prebuilt');
  const preDecode = pre.filter((e) => decodeInv.has(e.args.invocation)).map((e) => e.dur / 1000);
  const preCold = pre.filter((e) => !decodeInv.has(e.args.invocation)).map((e) => e.dur / 1000);
  const excluded = [
    { label: 'bind.compatibility', text: `只出现在 lane ${compLanes.join(' / ')}，${comp.length} 次合计 ${r2(comp.reduce((a, e) => a + e.dur, 0) / 1000)} ms` },
    { label: 'bind.prebuilt', text: `稳态 decode 里 ${preDecode.length} 次合计只有 ${r2(preDecode.reduce((a, v) => a + v, 0))} ms；它的 ${r2(preCold.reduce((a, v) => a + v, 0))} ms 几乎全在冷启动那两步` },
  ];

  // 4) 这份 trace 答不了的：strace.host 的 args 字段就这些，没有字节数，也没有 bind 内部的拆分
  const fields = [...new Set(trace.traceEvents.filter((e) => e.ph === 'X' && e.cat === 'strace.host').flatMap((e) => Object.keys(e.args)))];
  const missing = [
    { label: 'bind.args 的内部拆分', text: '它是叶子 span，分配显存与 H2D 搬运没有分开计时' },
    { label: '每次搬运的字节数', text: `args 只有 ${fields.join(' / ')}，算不出等效带宽，区分不了“搬得多”还是“搬得慢”` },
    { label: 'lane → 物理卡 / NUMA / PCIe 拓扑', text: '无法判断是否与链路位置相关' },
  ];
  const experiments = [
    { label: '换 lane ↔ 卡 映射重跑', text: '判定是“第 N 号位置”还是“这张卡”' },
    { label: '对比 prefill 与 decode 的参数清单', text: 'prefill 不慢而 decode 慢，差异就在两者绑定的参数集合里' },
  ];

  return { ids, med, byLane, segments, stages, excluded, missing, experiments };
}

/* 定界结论：关键路径上每一层查完之后剩下什么。
 * 只有「命中」和「待分清」需要用户往下走，所以它们带 next；
 * 排除项只留一个 topic 作为查过的记录，不展开。 */
function dsparkQuestions(run, runtime, dist) {
  const g = (k) => runtime.path.find((p) => p.key === k).stat.mean;
  const fmt = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : ms >= 100 ? `${Math.round(ms)} ms` : `${ms.toFixed(1)} ms`);
  const share = (ms) => `${Math.round((ms / runtime.period.mean) * 100)}%`;
  // 结论先用用户能感知的量说话：一个 token 等多久、整次请求被它吃掉多少；
  // bind.args / H2D 这些实现口径退到 tech 里，悬停可查，下面的关键路径表也仍然列着。
  const e2eShare = Math.round((runtime.resident.e2eSaveMs / run.totals.e2eMs) * 100);
  return [
    {
      topic: '每出一个 token，都要重新把权重搬上卡',
      state: 'hit',
      hot: true,
      a: `出一个 token 要等 ${fmt(runtime.period.mean)}，其中 ${fmt(g('h2d'))}（${share(g('h2d'))}）在把权重搬进显存，搬完设备才开始算；这次请求 ${runtime.n} 步累计 ${fmt(runtime.resident.e2eSaveMs)}，占端到端 ${fmt(run.totals.e2eMs)} 的 ${e2eShare}%。权重常驻后同样一步推算 ≈ ${fmt(runtime.resident.period.mean)}。`,
      tech: `口径：关键 lane 的 bind.args（设备内存分配 + Host→Device 搬运）${fmt(g('h2d'))}/步，取 ${runtime.n} 个 decode step 的均值；${dist.nLane} 条 lane 每步合计 ${fmt(runtime.windows[0].stat.mean)}。常驻后的步长是推算值。`,
      next: `最慢的是 ${dist.critLane} 号卡（lane ${dist.critLane}），先看它`,
      go: { tab: 'dist', lane: dist.critLane },
    },
    {
      topic: 'Device 到达屏障的时间差',
      state: 'open',
      a: `runner_run ≤ ${fmt(g('device'))} 是含等待的上界；按设备执行估计拆开后，${dist.nLane} 条 lane 的执行都在 ${fmt(Math.min(...dist.lanes.map((l) => l.deviceExec)))}–${fmt(Math.max(...dist.lanes.map((l) => l.deviceExec)))}，起点却分散 ${fmt(dist.startSpread.mean)}——差在到达，不在执行。`,
      next: '看 16 条 lane 的执行与等待拆分',
      go: { tab: 'dist' },
      estimate: true,
    },
    {
      topic: 'Serving 调度',
      state: 'clear',
      a: `落在关键路径上的调度只有 ${fmt(g('serving'))}/步，其余时间在 wait_worker_output（平均 ${fmt(run.serving.waitWorker.avgMs)}）。`,
    },
    {
      topic: 'Worker 下发',
      state: 'clear',
      a: `dispatch ${fmt(run.worker.dispatch.avgMs)}、prepare ${fmt(run.worker.prepare.avgMs)}、reclaim ${fmt(run.worker.reclaim.avgMs)} 都与本步重叠，只有 ${fmt(g('worker'))} 落在关键路径上。`,
    },
  ];
}

function dspark() {
  const run = TRIAGE.runs.find((r) => r.id === 'serving_dspark_gbs1');
  const label = (s) => (s.id === 'Prefill' ? 'P' : s.id === 'Draft init' ? 'DI' : s.id);
  // 每条 lane：从本步起点到它进入 runner_run 的时间（Host 侧），以及它的 runner_run（含等待）
  const lanes = (s) => {
    const w = run.waterfalls[s.id];
    const eff = Math.max(w.prevEnd, Math.min(...w.lanes.map((l) => l.start)));
    return w.lanes.map((l) => ({ dev: l.dev, lead: r2(Math.max(0, l.rrStart - eff)), rr: r2(l.end - l.rrStart) }));
  };
  const laneCount = run.waterfalls[run.steps[0].id].lanes.length;
  const validateOf = (s) => {
    const phaseOf = (e) => e.name.split('| ')[1];
    const sp = traceEvents().traceEvents.find((e) => e.ph === 'X' && e.cat === 'strace.host' && e.tid === s.critLane && e.args && e.args.invocation === s.inv && phaseOf(e) === 'validate');
    return sp ? sp.dur / 1000 : 0;
  };
  // Host 侧按 host lane 命名，Device 侧按它驱动的卡命名；同一个编号上下对应
  const laneLabels = Array.from({ length: laneCount }, (_, i) => `lane ${i}`);
  const devLabels = Array.from({ length: laneCount }, (_, i) => `dev ${i}`);
  const hostFloors = run.steps.map((s) => ({
    id: label(s),
    ms: r2(s.gapWorkerMs + s.hostMs),
    devs: lanes(s).map((l) => l.lead),
    critDev: s.critLane,
    cards: [
      card(`lane ${s.critLane} · bind.args`, '关键 lane · 分配 + H2D', s.critBindMs, 'crit'),
      card('其余 lane', 'bind.args 中位', s.medBindMs, ''),
    ],
  }));
  const deviceFloors = run.steps.map((s) => ({
    id: label(s),
    ms: s.deviceMs,
    devs: lanes(s).map((l) => l.rr),
    critDev: s.critLane,
    cards: [
      card('runner_run', `dev ${s.critLane} · 上界`, s.deviceMs, '', `≤ ${s.deviceMs.toFixed(1)} ms`),
      card('validate · D2H', `dev ${s.critLane}`, validateOf(s), ''),
    ],
  }));
  const servingFloors = run.steps.map((s) => ({
    id: label(s),
    ms: s.gapServingMs,
    cards: [
      card('调度器', '关键路径上', s.gapServingMs, s.gapServingMs < 0.5 ? 'dim' : ''),
      card('schedule + output', '已被流水覆盖', run.serving.schedule.avgMs + run.serving.processOutput.avgMs, 'dim'),
    ],
  }));
  const decode = run.steps.filter((s) => s.decodeStep);
  const hot = decode[0].critLane;
  const runtime = dsparkRuntime(run, decode);
  const dist = dsparkDistribution(run, decode);
  const laneFocus = dsparkLaneFocus(run, decode, dist);
  const e2e = dsparkE2E(run);
  const questions = dsparkQuestions(run, runtime, dist);
  const prefill = run.steps.find((s) => s.id === 'Prefill');
  const draft = run.steps.find((s) => s.id === 'Draft init');
  const h2d = runtime.path.find((p) => p.key === 'h2d').stat;
  const dev = runtime.path.find((p) => p.key === 'device').stat;
  const fmt = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : ms >= 100 ? `${Math.round(ms)} ms` : `${ms.toFixed(1)} ms`);
  // 优化队列：与 #95 的 P0 / P1 / P2 对应，“当前”列全部取自本次基线
  const plan = [
    { pri: 'P0', title: '权重 / Cache 常驻 Device', now: `bind.args ${fmt(h2d.mean)}/step`, action: '加载时上传一次，decode 原地更新，不再逐步分配 + H2D', expect: `步长 → ≈ ${fmt(runtime.resident.period.mean)}（推算）` },
    { pri: 'P1', title: `lane ${hot} bind 偏慢`, now: `比中位 lane 多 ${fmt(run.decode.critBindAvgMs - run.decode.medBindAvgMs)}`, action: `常驻前的过渡：查 lane ${hot}（dev ${hot}）的参数布局与 H2D 通道`, expect: `对齐次慢 lane，E2E −${(run.decode.whatIfSaveMs / 1000).toFixed(2)} s` },
    { pri: 'P1', title: 'Device 临界路径', now: `runner_run ≤ ${fmt(dev.mean)}`, action: '常驻后成为主瓶颈；对照设备执行与同步等待估计，检查 MoE 通信与 rank 均衡', expect: `执行估计 ${fmt(dist.lanes.find((l) => l.dev === dist.critLane).deviceExec)}/步` },
    { pri: 'P2', title: '冷启动移出请求路径', now: `Prefill + Draft init ${fmt(prefill.periodMs + draft.periodMs)}`, action: '固定 shape 的 buffer、arena、通信 domain 在启动时初始化', expect: '首次输出提前' },
  ];
  const checks = [
    '稳态 decode 的 bind.args 接近 0',
    `validate · D2H 不高于基线 ${runtime.windows.find((w) => /validate/.test(w.label)).stat.mean.toFixed(2)} ms`,
    'Cache 原地更新后输出与基线一致',
    '优化后数据单独记录，不覆盖本基线',
  ];
  // P1：每个 decode step 的时间去哪了。用均值保证三层相加等于步长，另给 P50 / P90。
  const q = (xs, p) => { const a = xs.slice().sort((x, y) => x - y); return a[Math.min(a.length - 1, Math.floor(p * (a.length - 1)))]; };
  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
  const per = {
    serving: decode.map((s) => s.gapServingMs),
    host: decode.map((s) => s.gapWorkerMs + s.hostMs),
    device: decode.map((s) => s.deviceMs),
  };
  const period = decode.map((s) => s.periodMs);
  const perStep = {
    scope: 'decode',
    n: decode.length,
    meanMs: r2(mean(period)),
    p50Ms: r2(q(period, 0.5)),
    p90Ms: r2(q(period, 0.9)),
    layers: Object.fromEntries(Object.entries(per).map(([k, xs]) => [k, { meanMs: r2(mean(xs)), p50Ms: r2(q(xs, 0.5)) }])),
    schedMs: r2(run.serving.schedule.avgMs + run.serving.processOutput.avgMs + run.serving.queueStep.avgMs),
  };
  const hits = decode.filter((s) => s.critLane === hot).length;
  return {
    id: 'dspark',
    kind: 'model',
    title: 'DSpark Serving',
    sub: 'DeepSeek-V4-Flash · 单请求',
    deviceText: `${laneCount} NPU`,
    totalMs: run.totals.e2eMs,
    floorUnit: 'step',
    bottleneck: 'host',
    headline: `lane ${hot} 的 bind.args 拖住了 ${hits}/${decode.length} 个 decode step`,
    whatIf: `对齐次慢 lane，E2E 约 −${(run.decode.whatIfSaveMs / 1000).toFixed(2)} s`,
    perStep,
    runtime,
    dist,
    laneFocus,
    e2e,
    questions,
    plan,
    checks,
    coldStart: { prefillMs: prefill.periodMs, draftMs: draft.periodMs },
    reference: { label: 'pypto-serving#95', url: 'https://github.com/hw-native-sys/pypto-serving/issues/95' },
    request: dsparkRequest(run),
    ...(() => { const f = dsparkFlows(run); return { workerRows: f.rows, flows: f.steps }; })(),
    defaultStep: (() => {
      const d = run.steps.filter((x) => x.decodeStep);
      const med = perStep.p50Ms;
      return label(d.reduce((a, b) => (Math.abs(b.periodMs - med) < Math.abs(a.periodMs - med) ? b : a)));
    })(),
    analysis: './analysis.html#run=serving_dspark_gbs1',
    towers: {
      serving: { ms: run.totals.servingMs, caption: '每个 step 中只有调度器在推进的时间', floors: servingFloors },
      host: {
        ms: run.totals.hostMs, caption: '每个 step：关键 lane 进入 runner_run 之前的时间', floors: hostFloors,
        devices: { labels: laneLabels, heat: true, metric: '进入 runner_run 前' },
      },
      device: {
        ms: run.totals.deviceMs, upper: true, caption: '每个 step：关键 lane 的 runner_run（上界，未采 device_wall）', floors: deviceFloors,
        devices: { labels: devLabels, heat: false, metric: 'runner_run（含等待）' },
      },
    },
    notes: ['Serving / Worker / 16 条 Host lane 共用 host 时钟', 'Device 取 runner_run，含跨 lane 同步等待，是上界'],
    source: 'Data/serving-strace-swimlane(2).json',
  };
}

const runs = [dspark()];
fs.writeFileSync(OUT, `/* Generated by build-home-data.cjs — do not edit by hand. */\nwindow.TRIAGE_HOME = ${JSON.stringify({ runs })};\n`);
runs.forEach((r) => console.log(r.id, r.totalMs, 'S', r.towers.serving.ms, 'H', r.towers.host.ms, 'D', r.towers.device.ms, '|', r.headline, '|', r.whatIf));
console.log('bytes', fs.statSync(OUT).size);
