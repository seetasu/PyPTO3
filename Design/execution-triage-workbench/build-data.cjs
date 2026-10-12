#!/usr/bin/env node
/* Build data.js for the Execution Triage workbench.
 *
 * Sources (read-only):
 *   1. Data/serving-strace-swimlane(2).json
 *      DSpark GBS1 serving request: SA_PROFILE spans (api / scheduler / worker)
 *      + 16 Simpler Host STRACE lanes on one host CLOCK_MONOTONIC.
 *   2. Design/operator-tuning-console/data.js  (TUNING_RUNS.decode_csa)
 *      decode_csa L3 operator run: chip.run host spans + device_wall split
 *      + device swimlane accounting.
 *
 * Usage: node Design/execution-triage-workbench/build-data.cjs
 */
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const OUT = path.join(__dirname, 'data.js');
const r1 = (v) => Math.round(v * 10) / 10;
const r2 = (v) => Math.round(v * 100) / 100;

/* ---------------------------------------------------------------- serving */
function buildServing() {
  const src = path.join(ROOT, 'Data', 'serving-strace-swimlane(2).json');
  const trace = JSON.parse(fs.readFileSync(src, 'utf8'));
  const ev = trace.traceEvents.filter((e) => e.ph === 'X');
  const procName = {};
  trace.traceEvents.filter((e) => e.ph === 'M' && e.name === 'process_name')
    .forEach((e) => { procName[e.pid] = e.args.name; });
  const LANE_PID = Number(Object.keys(procName).find((p) => /STRACE/.test(procName[p])));
  const API_PID = Number(Object.keys(procName).find((p) => /serving-api/.test(procName[p])));
  const WORKER_PID = Number(Object.keys(procName).find((p) => /serving-worker/.test(procName[p])));

  const http = ev.find((e) => e.name === 'http.completions');
  const reqStart = http.ts;
  const reqEnd = http.ts + http.dur;
  const end = (e) => e.ts + e.dur;

  // ---- per invocation (prefill, drafter bootstrap, 53 decode steps) ----
  const lane = ev.filter((e) => e.pid === LANE_PID);
  const phaseOf = (e) => e.name.split('| ')[1];
  const invIds = [...new Set(lane.map((e) => e.args.invocation))].sort((a, b) => a - b);
  const invs = invIds.map((inv) => {
    const spans = lane.filter((e) => e.args.invocation === inv);
    const runs = spans.filter((e) => phaseOf(e) === 'simpler_run');
    const label = runs[0].name.split(' |')[0];
    const lanes = runs.map((run) => {
      const own = spans.filter((e) => e.tid === run.tid);
      const get = (n) => own.find((e) => phaseOf(e) === n);
      const bind = get('bind');
      const rr = get('runner_run');
      const val = get('validate');
      return {
        dev: run.tid,
        start: run.ts,
        end: end(run),
        bindStart: bind.ts,
        bindEnd: end(bind),
        bindArgs: get('bind.args').dur,
        rrStart: rr.ts,
        rrEnd: end(rr),
        valEnd: val ? end(val) : end(rr),
      };
    }).sort((a, b) => a.dev - b.dev);
    return {
      inv,
      label,
      callable: runs[0].args.callable,
      decodeStep: runs[0].args.decode_step,
      lanes,
      minStart: Math.min(...lanes.map((l) => l.start)),
      maxEnd: Math.max(...lanes.map((l) => l.end)),
    };
  });

  // ---- critical path through the chip windows ----
  // All 16 lanes finish together (collective barrier), so the lane that enters
  // runner_run last carries the critical path: host time = until its runner_run,
  // device time <= its runner_run (upper bound, device_wall not collected).
  const workerSpans = ev.filter((e) => e.pid === WORKER_PID);
  const workerBusy = workerSpans.filter((e) => /^WorkerProcess\.(prepare_decode|dispatch_decode|batch_prefill)$/.test(e.name));
  const overlap = (a0, a1, spans) => {
    // union length of spans clipped to [a0, a1]
    const iv = spans.map((s) => [Math.max(a0, s.ts), Math.min(a1, end(s))]).filter(([x, y]) => y > x).sort((p, q) => p[0] - q[0]);
    let total = 0; let cur = null;
    for (const [x, y] of iv) {
      if (!cur || x > cur[1]) { if (cur) total += cur[1] - cur[0]; cur = [x, y]; } else cur[1] = Math.max(cur[1], y);
    }
    if (cur) total += cur[1] - cur[0];
    return total;
  };

  // Steps dispatched ahead by the pipeline may finish after the response is sent;
  // they are outside the request window and reported separately as overrun.
  const inWindow = invs.filter((v) => v.maxEnd <= reqEnd);
  const overrun = invs.filter((v) => v.maxEnd > reqEnd).map((v) => ({ id: v.label.split(' ')[0], startMs: r1((v.minStart - reqStart) / 1000), endMs: r1((v.maxEnd - reqStart) / 1000), durMs: r1((v.maxEnd - Math.max(v.minStart, reqEnd)) / 1000) }));
  const stepId = (v) => (v.decodeStep ? v.label.split(' ')[0] : (/prefill/.test(v.callable) ? 'Prefill' : 'Draft init'));

  const acc = { servingQueue: 0, workerExposed: 0, laneBind: 0, laneLaunch: 0, device: 0, tail: 0 };
  let prevEnd = reqStart;
  const steps = inWindow.map((inv) => {
    const effStart = Math.max(prevEnd, inv.minStart);
    const gap = Math.max(0, inv.minStart - prevEnd);
    const gapWorker = gap > 0 ? overlap(prevEnd, inv.minStart, workerBusy) : 0;
    const gapServing = gap - gapWorker;
    const crit = inv.lanes.reduce((a, b) => (b.rrStart > a.rrStart ? b : a));
    const hostWin = crit.rrStart - effStart;
    const deviceUb = inv.maxEnd - crit.rrStart;
    // split the lane-side host time into launch delay (lane not yet started) and bind
    const laneLaunch = Math.max(0, crit.start - effStart);
    const laneBind = hostWin - laneLaunch;
    acc.servingQueue += gapServing;
    acc.workerExposed += gapWorker;
    acc.laneLaunch += laneLaunch;
    acc.laneBind += laneBind;
    acc.device += deviceUb;
    const binds = inv.lanes.map((l) => l.bindEnd - l.bindStart).sort((a, b) => a - b);
    // what-if: critical lane enters runner_run together with the next-latest lane
    const second = inv.lanes.filter((l) => l !== crit).reduce((a, b) => (b.rrStart > a.rrStart ? b : a));
    const step = {
      id: stepId(inv),
      inv: inv.inv,
      callable: inv.callable,
      decodeStep: inv.decodeStep,
      t0: r1((prevEnd - reqStart) / 1000),
      gapServingMs: r2(gapServing / 1000),
      gapWorkerMs: r2(gapWorker / 1000),
      hostMs: r2((laneLaunch + laneBind) / 1000),
      laneLaunchMs: r2(laneLaunch / 1000),
      deviceMs: r2(deviceUb / 1000),
      periodMs: r2((inv.maxEnd - prevEnd) / 1000),
      critLane: crit.dev,
      critBindMs: r2((crit.bindEnd - crit.bindStart) / 1000),
      medBindMs: r2(binds[Math.floor(binds.length / 2)] / 1000),
      maxBindMs: r2(binds[binds.length - 1] / 1000),
      secondLane: second.dev,
      whatIfSaveMs: r2(Math.max(0, crit.rrStart - Math.max(second.rrStart, effStart)) / 1000),
    };
    inv.prevEnd = prevEnd;
    prevEnd = inv.maxEnd;
    return step;
  });
  const lastEnd = prevEnd;
  const tailWorker = overlap(lastEnd, reqEnd, workerSpans.filter((e) => /^WorkerProcess\./.test(e.name)));
  acc.tail = reqEnd - lastEnd;

  // ---- per step waterfall rows (rich detail) ----
  const sched = ev.filter((e) => e.pid === API_PID && /^scheduler\./.test(e.name));
  const workerTop = workerSpans.filter((e) => /^WorkerProcess\.(prepare_decode|dispatch_decode|reclaim_decode|batch_prefill|execute_step)$/.test(e.name));
  const waterfalls = {};
  inWindow.forEach((inv, i) => {
    const w0 = Math.min(inv.minStart, inv.prevEnd) - 4000;
    const w1 = inv.maxEnd + 4000;
    const clip = (list) => list.filter((e) => end(e) > w0 && e.ts < w1).map((e) => ({
      n: e.name.replace(/^(scheduler|WorkerProcess)\./, ''),
      s: r2((Math.max(e.ts, w0) - w0) / 1000),
      d: r2((Math.min(end(e), w1) - Math.max(e.ts, w0)) / 1000),
      step: e.args && e.args.step_id,
    }));
    const crit = inv.lanes.reduce((a, b) => (b.rrStart > a.rrStart ? b : a));
    waterfalls[steps[i].id] = {
      spanMs: r2((w1 - w0) / 1000),
      prevEnd: r2((inv.prevEnd - w0) / 1000),
      critStart: r2((crit.rrStart - w0) / 1000),
      end: r2((inv.maxEnd - w0) / 1000),
      serving: clip(sched).filter((s) => s.n !== 'wait_worker_output' || s.d > 1),
      worker: clip(workerTop),
      lanes: inv.lanes.map((l) => ({
        dev: l.dev,
        start: r2((l.start - w0) / 1000),
        bindStart: r2((l.bindStart - w0) / 1000),
        bindEnd: r2((l.bindEnd - w0) / 1000),
        bindArgsMs: r2(l.bindArgs / 1000),
        rrStart: r2((l.rrStart - w0) / 1000),
        end: r2((l.end - w0) / 1000),
      })),
    };
  });

  // ---- lane statistics over decode steps ----
  const decodeInvs = inWindow.filter((v) => v.decodeStep !== null && v.decodeStep !== undefined);
  const laneStats = decodeInvs[0].lanes.map((l) => {
    const binds = decodeInvs.map((v) => { const x = v.lanes.find((y) => y.dev === l.dev); return x.bindEnd - x.bindStart; });
    const critCount = steps.filter((s) => s.decodeStep !== null && s.critLane === l.dev).length;
    return { dev: l.dev, bindAvgMs: r2(binds.reduce((a, b) => a + b, 0) / binds.length / 1000), bindMaxMs: r2(Math.max(...binds) / 1000), critCount };
  });

  // ---- serving-side facts ----
  const avgOf = (name) => {
    const xs = ev.filter((e) => e.name === name);
    return { n: xs.length, avgMs: r2(xs.reduce((a, b) => a + b.dur, 0) / xs.length / 1000), maxMs: r2(Math.max(...xs.map((e) => e.dur)) / 1000) };
  };
  const decodeSteps = steps.filter((s) => s.decodeStep !== null && s.decodeStep !== undefined);
  const sum = (xs, k) => xs.reduce((a, b) => a + b[k], 0);
  const firstDecode = decodeSteps[0];

  const totals = {
    e2eMs: r2(http.dur / 1000),
    servingMs: r2((acc.servingQueue + acc.tail - tailWorker) / 1000),
    hostWorkerMs: r2((acc.workerExposed + tailWorker) / 1000),
    hostLaunchMs: r2(acc.laneLaunch / 1000),
    hostBindMs: r2(acc.laneBind / 1000),
    deviceMs: r2(acc.device / 1000),
  };
  totals.hostMs = r2(totals.hostWorkerMs + totals.hostLaunchMs + totals.hostBindMs);

  return {
    id: 'serving_dspark_gbs1',
    kind: 'model',
    title: 'DSpark Serving · 单请求',
    subtitle: 'DeepSeek-V4-Flash w8a8 · 16 NPU',
    workload: trace.metadata.workload,
    request: { id: http.args.request_id, maxTokens: http.args.max_tokens, stream: http.args.stream },
    clock: 'host CLOCK_MONOTONIC（Serving / Worker / 16 STRACE lane 共用）',
    source: 'Data/serving-strace-swimlane(2).json',
    capturedAt: '2026-09-18',
    totals,
    stepCount: { prefill: steps.length - decodeSteps.length, decode: decodeSteps.length },
    decode: {
      periodAvgMs: r2(sum(decodeSteps, 'periodMs') / decodeSteps.length),
      hostAvgMs: r2(sum(decodeSteps, 'hostMs') / decodeSteps.length),
      deviceAvgMs: r2(sum(decodeSteps, 'deviceMs') / decodeSteps.length),
      critBindAvgMs: r2(sum(decodeSteps, 'critBindMs') / decodeSteps.length),
      medBindAvgMs: r2(sum(decodeSteps, 'medBindMs') / decodeSteps.length),
      whatIfSaveMs: r2(sum(decodeSteps, 'whatIfSaveMs')),
    },
    ttftMs: r2((firstDecode.t0 * 1000) / 1000),
    serving: {
      schedule: avgOf('scheduler.schedule'),
      processOutput: avgOf('scheduler.process_step_output'),
      queueStep: avgOf('scheduler.queue_worker_step'),
      waitWorker: avgOf('scheduler.wait_worker_output'),
      addRequest: avgOf('ReplicaEngineCore.add_request'),
    },
    worker: {
      prepare: avgOf('WorkerProcess.prepare_decode'),
      dispatch: avgOf('WorkerProcess.dispatch_decode'),
      reclaim: avgOf('WorkerProcess.reclaim_decode'),
    },
    steps,
    overrun,
    laneStats,
    waterfalls,
  };
}

/* --------------------------------------------------------------- operator */
function buildOperator() {
  global.window = {};
  require(path.join(ROOT, 'Design', 'operator-tuning-console', 'data.js'));
  const run = global.window.TUNING_RUNS.decode_csa;
  const us = (o, k) => (o[k] ? o[k].us : 0);
  const invocations = [];
  for (const rank of Object.keys(run.e2e)) {
    for (const inv of Object.keys(run.e2e[rank])) {
      const e = run.e2e[rank][inv];
      const t0 = e['chip.run'].tsNs;
      const rel = (k) => (e[k] && e[k].clk === 'host' ? r2((e[k].tsNs - t0) / 1e6) : null);
      const chip = us(e, 'chip.run');
      const bind = us(e, 'chip.run.bind');
      const rr = us(e, 'chip.run.runner_run');
      const dw = us(e, 'chip.run.runner_run.device_wall');
      const val = us(e, 'chip.run.validate');
      const rel2 = us(e, 'chip.run.claim_release');
      invocations.push({
        id: `${rank} · #${inv}`,
        rank,
        inv: Number(inv),
        warm: Number(inv) > 1,
        chipMs: r2(chip / 1000),
        bindArgsMs: r2(us(e, 'chip.run.bind.args') / 1000),
        bindPrebuiltMs: r2(us(e, 'chip.run.bind.prebuilt') / 1000),
        bindMs: r2(bind / 1000),
        runnerMs: r2(rr / 1000),
        deviceMs: r2(dw / 1000),
        rrHostMs: r2((rr - dw) / 1000),
        validateMs: r2(val / 1000),
        otherMs: r2((chip - bind - rr - val - rel2) / 1000),
        hostMs: r2((chip - dw) / 1000),
        marks: {
          bind: rel('chip.run.bind'),
          bindPrebuilt: rel('chip.run.bind.prebuilt'),
          runner: rel('chip.run.runner_run'),
          validate: rel('chip.run.validate'),
        },
        deviceSplit: {
          preamble: r2(us(e, 'chip.run.runner_run.device_wall.preamble') / 1000),
          orch: r2(us(e, 'chip.run.runner_run.device_wall.orch') / 1000),
          sched: r2(us(e, 'chip.run.runner_run.device_wall.sched') / 1000),
          postOrch: r2(us(e, 'chip.run.runner_run.device_wall.post_orch') / 1000),
        },
      });
    }
  }
  const rk = run.ranks[run.defaultRank];
  const topTasks = rk.tasks
    .map((t) => ({ name: t.callable, engine: t.kind, coreMs: r2((t.engines.aic ? t.engines.aic.coreTime : 0) / 1000 + (t.engines.aiv ? t.engines.aiv.coreTime : 0) / 1000), blocks: t.blockCount }))
    .reduce((m, t) => { const x = m.get(t.name) || { name: t.name, engine: t.engine, coreMs: 0, blocks: 0, n: 0 }; x.coreMs += t.coreMs; x.blocks += t.blocks; x.n += 1; m.set(t.name, x); return m; }, new Map());
  const top = [...topTasks.values()].sort((a, b) => b.coreMs - a.coreMs).slice(0, 6).map((t) => ({ ...t, coreMs: r2(t.coreMs) }));
  return {
    id: 'operator_decode_csa',
    kind: 'operator',
    title: 'decode_csa · 单算子上板',
    subtitle: 'DeepSeek V4 · L3 · 2 rank',
    workload: `${run.case.program} · ${run.case.backend} · ${run.case.numCores} cores（${run.case.aicCount} AIC + ${run.case.aivCount} AIV）`,
    clock: 'host 与 device 两套时钟：host 段用 CLOCK_MONOTONIC，device_wall 用设备时钟，只比较时长',
    source: `Data/DeepseekV4/${run.source}`,
    capturedAt: run.case.capturedAt,
    invocations,
    device: {
      spanMs: r2(rk.swimlane.spanUs / 1000),
      tasks: rk.tasks.length,
      blocks: rk.accounting.workerBlocks,
      aicUtil: rk.occupancy.aicUtil,
      aivUtil: rk.occupancy.aivUtil,
      idlePct: rk.idlePct,
      critShare: rk.critical.share,
      kernelMs: r2(rk.accounting.workerKernelTime / 1000),
      setupMs: r2(rk.accounting.workerSetupTime / 1000),
      top,
    },
  };
}

const data = { generatedAt: new Date().toISOString().slice(0, 10), runs: [buildServing(), buildOperator()] };
fs.writeFileSync(OUT, `/* Generated by build-data.cjs — do not edit by hand. */\nwindow.TRIAGE_DATA = ${JSON.stringify(data)};\n`);
const s = data.runs[0];
console.log('serving totals', s.totals, 'decode', s.decode, 'ttft', s.ttftMs);
console.log('lane stats', s.laneStats.map((l) => `${l.dev}:${l.bindAvgMs}/${l.critCount}`).join(' '));
console.log('steps head', s.steps.slice(0, 4));
console.log('operator', data.runs[1].invocations.map((i) => [i.id, i.chipMs, i.bindMs, i.runnerMs, i.deviceMs, i.hostMs, i.otherMs]));
console.log('device', data.runs[1].device);
console.log('bytes', fs.statSync(OUT).size);
