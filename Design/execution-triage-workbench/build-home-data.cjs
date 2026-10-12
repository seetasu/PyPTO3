#!/usr/bin/env node
/* Build home-data.js for the Execution Triage home (isometric "three towers" view).
 *
 * One execution = three towers: Serving / Host / Device. Tower height = time the
 * layer holds on the critical path; floors = steps (or model layers); each floor
 * carries two cards that explain what that floor is made of.
 *
 * Sources (read-only):
 *   - ./data.js                                   (build-data.cjs: DSpark serving + decode_csa)
 *   - ../operator-tuning-console/qwen3-profiles-data.js   (Qwen3-14B TP1 / TP2 digest)
 *   - Data/pypto_qwen3_profiles/.../tp1/torch/.../trace_view.json  (TP1 per-step host functions)
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
global.window.TUNING_RUNS = { decode_fwd_layers: {} };
require(path.join(ROOT, 'Design', 'operator-tuning-console', 'qwen3-profiles-data.js'));
const Q = global.window.TUNING_RUNS.decode_fwd_layers.qwen3;

const card = (title, sub, ms, tone, text) => ({ title, sub, ms: r2(ms), tone: tone || '', text: text || null });
const sumMs = (floors) => r2(floors.reduce((a, f) => a + f.ms, 0));

/* ------------------------------------------------------------ DSpark serving */
function dspark() {
  const run = TRIAGE.runs.find((r) => r.id === 'serving_dspark_gbs1');
  const label = (s) => (s.id === 'Prefill' ? 'P' : s.id === 'Draft init' ? 'DI' : s.id);
  const hostFloors = run.steps.map((s) => ({
    id: label(s),
    ms: r2(s.gapWorkerMs + s.hostMs),
    cards: [
      card(`dev ${s.critLane}`, '关键 lane · bind', s.critBindMs, 'crit'),
      card('其余 lane', 'bind 中位', s.medBindMs, ''),
    ],
  }));
  const deviceFloors = run.steps.map((s) => ({
    id: label(s),
    ms: s.deviceMs,
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
  const hits = decode.filter((s) => s.critLane === hot).length;
  return {
    id: 'dspark',
    kind: 'model',
    title: 'DSpark Serving',
    sub: 'DeepSeek-V4-Flash · 16 NPU · 单请求',
    totalMs: run.totals.e2eMs,
    floorUnit: 'step',
    bottleneck: 'host',
    headline: `device ${hot} 的 bind 拖住了 ${hits}/${decode.length} 个 decode step`,
    whatIf: `对齐次慢 lane，E2E 约 −${(run.decode.whatIfSaveMs / 1000).toFixed(2)} s`,
    analysis: './analysis.html#run=serving_dspark_gbs1',
    towers: {
      serving: { ms: run.totals.servingMs, caption: '每个 step 中只有调度器在推进的时间', floors: servingFloors },
      host: { ms: run.totals.hostMs, caption: '每个 step：关键 lane 进入 runner_run 之前的时间', floors: hostFloors },
      device: { ms: run.totals.deviceMs, upper: true, caption: '每个 step：关键 lane 的 runner_run（上界，未采 device_wall）', floors: deviceFloors },
    },
    notes: ['Serving / Worker / 16 条 Host lane 共用 host 时钟', 'Device 取 runner_run，含跨 lane 同步等待，是上界'],
    source: 'Data/serving-strace-swimlane(2).json',
  };
}

/* ------------------------------------------------------------ Qwen3 TP1 / TP2 */
function tp1HostFunctions() {
  const dir = path.join(ROOT, 'Data', 'pypto_qwen3_profiles', 'pypto_qwen3_profiles', Q.torch.tp1.profDir, 'ASCEND_PROFILER_OUTPUT');
  const trace = JSON.parse(fs.readFileSync(path.join(dir, 'trace_view.json'), 'utf8'));
  const ev = Array.isArray(trace) ? trace : trace.traceEvents;
  const pn = {};
  ev.filter((e) => e.ph === 'M' && e.name === 'process_name').forEach((e) => { pn[e.pid] = e.args.name; });
  const py = ev.filter((e) => e.ph === 'X' && pn[e.pid] === 'Python');
  const steps = ev.filter((e) => e.ph === 'X' && /^ProfilerStep#\d+$/.test(e.name)).sort((a, b) => a.ts - b.ts);
  return steps.map((s) => {
    const t0 = Number(s.ts); const t1 = t0 + s.dur;
    const within = (re) => py.filter((e) => re.test(e.name) && Number(e.ts) >= t0 && Number(e.ts) + e.dur <= t1 + 1).reduce((a, e) => a + e.dur, 0) / 1000;
    return {
      compactMs: within(/compact_vllm_kv_for_contract$/),
      scatterMs: within(/scatter_contract_kv_to_vllm$/),
    };
  });
}

function qwen(tp) {
  const T = Q.torch[tp];
  const stepLabel = (s, i) => (s.stage === 'prefill' ? 'P' : `D${i}`);
  const fn = tp === 'tp1' ? tp1HostFunctions() : null;
  const hostFloors = T.steps.map((s, i) => {
    const host = (s.freeUs + s.preparingUs) / 1000;
    const kv = fn ? fn[i].compactMs + fn[i].scatterMs : 0;
    return {
      id: stepLabel(s, i),
      ms: r2(host),
      cards: fn
        ? [card('KV 布局转换', 'compact + scatter', kv, 'crit'), card('设备空闲', 'Free', s.freeUs / 1000, '')]
        : [card('设备空闲', 'Free', s.freeUs / 1000, 'dim'), card('准备', 'Preparing', s.preparingUs / 1000, 'dim')],
    };
  });
  const L = Q.layers[`${tp}:decode`];
  const names = L.steps ? L.steps.map((s) => s.name) : null;
  const spans = L.rows.map((r) => r.spanUs).sort((a, b) => b - a);
  const deviceFloors = L.rows.map((r) => {
    let cards;
    if (names && r.blocks) {
      const top = r.blocks.slice().sort((a, b) => b[1] - a[1]).slice(0, 2);
      cards = top.map((b, k) => card(names[b[3]], `${b[2] ? 'AIC' : 'AIV'} · 第 ${b[3] + 1} 个任务`, b[1] / 1000, k === 0 && b[1] / r.spanUs > 0.35 ? 'crit' : ''));
    } else {
      cards = [
        card('层跨度', `${r.tasks} 个任务`, r.spanUs / 1000, r.spanUs >= spans[2] ? 'crit' : ''),
        card('最慢任务', `第 ${r.worstStep + 1} 个`, r.worstUs / 1000, ''),
      ];
    }
    return { id: `L${String(r.layer).padStart(2, '0')}`, ms: r2(r.spanUs / 1000), cards };
  });
  const totalStage = T.steps.reduce((a, s) => a + s.stageUs, 0) / 1000;
  const devMs = T.steps.reduce((a, s) => a + s.computingUs, 0) / 1000;
  const hostMs = sumMs(hostFloors);
  const occ = Q.l2[`${tp}:decode`].occ;
  const cores = (occ.aicUsed || 0) + (occ.aivUsed || 0);
  const isTp1 = tp === 'tp1';
  let headline; let whatIf;
  if (isTp1) {
    const kv = fn.slice(1).reduce((a, x) => a + x.compactMs + x.scatterMs, 0) / (fn.length - 1);
    headline = `设备 ${T.decodeMean.freeShare.toFixed(0)}% 时间在等 Host，每步约 ${kv.toFixed(0)} ms 花在 KV 布局转换`;
    whatIf = `kernel 再快一倍，也只省 step 时间的 ${(T.decodeMean.computeShare / 2).toFixed(0)}%`;
  } else {
    const gate = L.steps.find((s) => s.name === 'gate/up');
    const layerMean = L.rows.reduce((a, r) => a + r.spanUs, 0) / L.rows.length;
    headline = `decode 只用了 60 核中的 ${cores} 个，gate/up 占每层 ${((gate.meanUs / layerMean) * 100).toFixed(0)}%`;
    whatIf = `Host 已压到 ${T.decodeMean.freeShare.toFixed(1)}%，收益在 kernel 并行度`;
  }
  return {
    id: `qwen3_${tp}`,
    kind: 'model',
    title: `Qwen3-14B ${isTp1 ? 'TP1' : 'TP2'}`,
    sub: isTp1 ? '单卡 · 40 层 fused · 未验收采集' : '2 rank · 40 层 fused · rank0',
    totalMs: r2(totalStage),
    floorUnit: 'step',
    bottleneck: isTp1 ? 'host' : 'device',
    headline,
    whatIf,
    analysis: null,
    towers: {
      serving: { na: true, ms: 0, caption: '测试脚本直接驱动，没有请求调度', floors: [] },
      host: { ms: hostMs, derived: true, caption: '每个 step：设备空闲期（Free + Preparing），由 Host 侧造成', floors: hostFloors },
      device: { ms: r2(devMs), caption: `楼层 = 单次 decode 的 40 层（L2 泳道，与 profiler 不同进程）`, floors: deviceFloors, floorUnit: 'layer', heightMs: r2(devMs) },
    },
    notes: [
      'Host = 设备空闲期，按 step_trace_time 推导',
      '泳道与 profiler 来自两次独立进程，时间轴不对齐',
      ...(isTp1 ? ['TP1 不在数据集验收范围内'] : ['TP 通信在融合 kernel 内部，profiler 不可见']),
    ],
    source: `Data/pypto_qwen3_profiles/${tp}/`,
  };
}

/* ------------------------------------------------------------ decode_csa */
function decodeCsa() {
  const run = TRIAGE.runs.find((r) => r.id === 'operator_decode_csa');
  const lab = (i) => `${i.rank.replace('rank', 'r')}#${i.inv}`;
  const hostFloors = run.invocations.map((i) => ({
    id: lab(i),
    ms: i.hostMs,
    cards: [
      card('runner_run', '不在 device 上', i.rrHostMs, 'crit'),
      card('bind', i.warm ? '稳态' : '首次 · 含 prebuilt', i.bindMs, i.warm ? '' : 'crit'),
    ],
  }));
  const deviceFloors = run.invocations.map((i) => ({
    id: lab(i),
    ms: i.deviceMs,
    cards: [card('device_wall', '设备时钟', i.deviceMs, ''), card('sched', 'kernel 执行', i.deviceSplit.sched, '')],
  }));
  const warm = run.invocations.filter((i) => i.warm);
  const devShare = warm.reduce((a, i) => a + i.deviceMs, 0) / warm.reduce((a, i) => a + i.chipMs, 0) * 100;
  return {
    id: 'decode_csa',
    kind: 'operator',
    title: 'decode_csa',
    sub: 'DeepSeek V4 · L3 单算子 · 2 rank',
    totalMs: r2(run.invocations.reduce((a, i) => a + i.chipMs, 0)),
    floorUnit: 'call',
    bottleneck: 'host',
    headline: `稳态调用里设备只忙 ${devShare.toFixed(1)}%，runner_run 大部分时间不在 device 上`,
    whatIf: `只优化 kernel，chip.run 最多改善 ${devShare.toFixed(1)}%`,
    analysis: './analysis.html#run=operator_decode_csa',
    towers: {
      serving: { na: true, ms: 0, caption: '单算子直接调用，没有 Serving 层', floors: [] },
      host: { ms: sumMs(hostFloors), caption: '每次 chip.run：除 device_wall 外的时间', floors: hostFloors },
      device: { ms: sumMs(deviceFloors), caption: '每次 chip.run：device_wall（设备时钟，只比较时长）', floors: deviceFloors },
    },
    notes: ['Host 与 Device 用不同时钟，只比较时长'],
    source: 'Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617',
  };
}

const runs = [dspark(), qwen('tp1'), qwen('tp2'), decodeCsa()];
fs.writeFileSync(OUT, `/* Generated by build-home-data.cjs — do not edit by hand. */\nwindow.TRIAGE_HOME = ${JSON.stringify({ runs })};\n`);
runs.forEach((r) => console.log(r.id, r.totalMs, 'S', r.towers.serving.ms, 'H', r.towers.host.ms, 'D', r.towers.device.ms, '|', r.headline, '|', r.whatIf));
console.log('bytes', fs.statSync(OUT).size);
