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

/* ------------------------------------------------------------ DSpark serving */
// P0：请求级服务指标。输出时刻取调度器 process_step_output（只算落在请求窗口内的）。
function dsparkRequest(run) {
  const trace = JSON.parse(fs.readFileSync(path.join(ROOT, 'Data', 'serving-strace-swimlane(2).json'), 'utf8'));
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
  const laneLabels = Array.from({ length: laneCount }, (_, i) => `dev ${i}`);
  const hostFloors = run.steps.map((s) => ({
    id: label(s),
    ms: r2(s.gapWorkerMs + s.hostMs),
    devs: lanes(s).map((l) => l.lead),
    critDev: s.critLane,
    cards: [
      card(`dev ${s.critLane}`, '关键 lane · bind', s.critBindMs, 'crit'),
      card('其余 lane', 'bind 中位', s.medBindMs, ''),
    ],
  }));
  const deviceFloors = run.steps.map((s) => ({
    id: label(s),
    ms: s.deviceMs,
    devs: lanes(s).map((l) => l.rr),
    critDev: s.critLane,
    cards: [
      card('runner_run', `dev ${s.critLane} · 上界`, s.deviceMs, '', `≤ ${s.deviceMs.toFixed(1)} ms`),
      card('step 周期', 'Host + Device', s.periodMs, ''),
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
    headline: `device ${hot} 的 bind 拖住了 ${hits}/${decode.length} 个 decode step`,
    whatIf: `对齐次慢 lane，E2E 约 −${(run.decode.whatIfSaveMs / 1000).toFixed(2)} s`,
    perStep,
    request: dsparkRequest(run),
    analysis: './analysis.html#run=serving_dspark_gbs1',
    towers: {
      serving: { ms: run.totals.servingMs, caption: '每个 step 中只有调度器在推进的时间', floors: servingFloors },
      host: {
        ms: run.totals.hostMs, caption: '每个 step：关键 lane 进入 runner_run 之前的时间', floors: hostFloors,
        devices: { labels: laneLabels, heat: true, metric: '进入 runner_run 前' },
      },
      device: {
        ms: run.totals.deviceMs, upper: true, caption: '每个 step：关键 lane 的 runner_run（上界，未采 device_wall）', floors: deviceFloors,
        devices: { labels: laneLabels, heat: false, metric: 'runner_run（含等待）' },
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
