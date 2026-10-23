#!/usr/bin/env node
/* 核级泳道（AIC / AIV）的数据：来自 PyPTO 自己的 L2 swimlane 导出
 * Data/pypto_qwen3_profiles/pypto_qwen3_profiles/tp1/swimlane/prefill.json
 *
 * 这是**第三份**数据，和前两页都不是一回事，必须在页面上写清楚：
 *   1. 模型是 Qwen3（TP1），不是 DeepSeek-V4-Flash / DSpark；
 *   2. 跑的是 prefill 的一次 L2 调用，不是 decode step；
 *   3. 与 GBS1 STRACE、GBS256 profiler trace 都不是同一次采集，不能对齐、不能相加。
 * 它回答的是另一个问题：**一次调用在一张卡的 60 个核上是怎么铺开的**。
 *
 * 为什么用 prefill 而不是同目录的 decode：decode.json 里 `FuncId` 全是 -1，
 * 任务名只有 `task(r2t0)` 这样的 token，映射不到算子名；prefill 的 FuncId 有值，
 * 事件名本身就是 `gate_up_proj_spmd(r1t5)`，19 种算子都能认出来。
 *
 * 用法：node Design/execution-triage-workbench/build-core-data.cjs
 * 需要 --max-old-space-size（源文件 39 MB，解析后几百 MB），脚本内部已 re-exec。 */
'use strict';

const fs = require('fs');
const path = require('path');

if (!process.env.__CORE_DATA_CHILD__) {
  // 解析 39 MB JSON 需要更大的堆，直接重启自己
  const { spawnSync } = require('child_process');
  const r = spawnSync(process.execPath, ['--max-old-space-size=4096', __filename], {
    stdio: 'inherit',
    env: { ...process.env, __CORE_DATA_CHILD__: '1' },
  });
  process.exit(r.status === null ? 1 : r.status);
}

const ROOT = path.resolve(__dirname, '..', '..');
const DIR = path.join(ROOT, 'Data', 'pypto_qwen3_profiles', 'pypto_qwen3_profiles', 'tp1', 'swimlane');
const TRACE = path.join(DIR, 'prefill.json');
const NAME_MAP = path.join(DIR, 'prefill_records', 'name_map_build_output.json');
const RECORDS = path.join(DIR, 'prefill_records', 'l2_swimlane_records.json');
const DEPS = path.join(DIR, 'prefill_records', 'deps.json');
const OUT = path.join(__dirname, 'core-data.js');

const r2 = (v) => Math.round(v * 100) / 100;
const r3 = (v) => Math.round(v * 1000) / 1000;
const r4 = (v) => Math.round(v * 10000) / 10000;

/* 算子归类：和设备详情页用的是同一套语义，但这里是 Qwen3 的 dense 结构，没有 MoE。
 * 顺序即优先级。分组只影响配色与小结，表里给的是 trace 里的原始算子名。 */
const GROUPS = [
  { key: 'attn', label: '注意力', re: /qk_pv|attention|rope|kv_cache|probe/i },
  { key: 'proj', label: '投影 · 矩阵乘', re: /q_proj|kv_proj|out_proj|lm_head|^proj/i },
  { key: 'mlp', label: 'MLP', re: /gate_up|gate_proj|up_proj|down_proj|silu|mlp/i },
  { key: 'norm', label: '归一化 · 量化', re: /rmsnorm|rms|norm|cast|gamma/i },
  { key: 'embed', label: '嵌入 · 采样', re: /embed|sample|token/i },
  { key: 'misc', label: '其他', re: /./ },
];
const groupOf = (name) => GROUPS.find((g) => g.re.test(name)).key;

const stripTok = (s) => s.replace(/\s*\(.*\)\s*$/, '').replace(/_spmd$/, '');
// "2: [t0, r1t2]" → ['t0','r1t2']
function hintTokens(s) {
  if (!s) return [];
  const m = /\[([^\]]*)\]/.exec(s);
  if (!m || !m[1].trim()) return [];
  return m[1].split(',').map((x) => x.trim()).filter(Boolean);
}

const trace = JSON.parse(fs.readFileSync(TRACE, 'utf8'));
const nameMap = JSON.parse(fs.readFileSync(NAME_MAP, 'utf8'));
const records = JSON.parse(fs.readFileSync(RECORDS, 'utf8'));
const deps = JSON.parse(fs.readFileSync(DEPS, 'utf8'));
const all = trace.traceEvents;

/* 核：线程名就是核名（AIC_0 … AIV_59）；核类型以 records 的 metadata 为准 */
const tid2core = {};
all.filter((e) => e.ph === 'M' && e.name === 'thread_name' && e.pid === 4)
  .forEach((e) => { tid2core[e.tid] = e.args.name; });
const coreName = [...new Set(Object.values(tid2core))].filter((n) => /^AI[CV]_\d+$/.test(n))
  .sort((a, b) => Number(a.split('_')[1]) - Number(b.split('_')[1]));
const coreIdx = Object.fromEntries(coreName.map((n, i) => [n, i]));
const coreType = coreName.map((n) => (n.startsWith('AIC') ? 'aic' : 'aiv'));

/* 核上任务：一个 SPMD 任务会同时落在多个核上，按 task token 归并成一组 */
const X = all.filter((e) => e.ph === 'X' && e.pid === 4 && e.args && e.args['kernel-duration-us'] != null);
const tasks = new Map();
let t0 = Infinity;
X.forEach((e) => {
  const hint = e.args['event-hint'] || '';
  const tok = (/Task:(\S+?),/.exec(hint) || [, null])[1];
  const fid = Number((/FuncId:(-?\d+)/.exec(hint) || [, -1])[1]);
  const core = tid2core[e.tid];
  if (!tok || coreIdx[core] == null) return;
  if (e.ts < t0) t0 = e.ts;
  let t = tasks.get(tok);
  if (!t) {
    t = {
      tok,
      name: nameMap.callable_id_to_name[fid] || stripTok(e.name),
      raw: stripTok(e.name),
      fid,
      fanin: hintTokens(e.args['fanin-hint']),
      fanout: hintTokens(e.args['fanout-hint']),
      runs: [],
    };
    tasks.set(tok, t);
  }
  t.runs.push([coreIdx[core], e.ts, e.args['kernel-duration-us'], e.args.local_setup_us]);
});

const list = [...tasks.values()].sort((a, b) => Math.min(...a.runs.map((r) => r[1])) - Math.min(...b.runs.map((r) => r[1])));
const tokIdx = Object.fromEntries(list.map((t, i) => [t.tok, i]));
let t1 = 0;
list.forEach((t) => t.runs.forEach((r) => { t1 = Math.max(t1, r[1] + r[2] + r[3]); }));
const spanMs = (t1 - t0) / 1000;

/* 算子：按名字聚合，顺带记下它用了哪些核、kernel 与 setup 各多少 */
const byOp = new Map();
list.forEach((t) => {
  const o = byOp.get(t.name) || { name: t.name, group: groupOf(t.name), n: 0, nTask: 0, kernel: 0, setup: 0, cores: new Set(), aic: 0, aiv: 0 };
  o.nTask += 1;
  t.runs.forEach(([ci, , k, s]) => {
    o.n += 1; o.kernel += k / 1000; o.setup += s / 1000; o.cores.add(ci);
    if (coreType[ci] === 'aic') o.aic += k / 1000; else o.aiv += k / 1000;
  });
  byOp.set(t.name, o);
});
const ops = [...byOp.values()].sort((a, b) => b.kernel - a.kernel).map((o) => ({
  name: o.name, group: o.group, n: o.n, nTask: o.nTask,
  kernel: r3(o.kernel), setup: r3(o.setup), cores: o.cores.size,
  aic: r3(o.aic), aiv: r3(o.aiv),
}));
const opIdx = Object.fromEntries(ops.map((o, i) => [o.name, i]));

/* 每个核的忙碌与任务数 */
const perCore = coreName.map(() => ({ n: 0, kernel: 0, setup: 0 }));
list.forEach((t) => t.runs.forEach(([ci, , k, s]) => {
  perCore[ci].n += 1; perCore[ci].kernel += k / 1000; perCore[ci].setup += s / 1000;
}));

const sum = (xs) => xs.reduce((a, b) => a + b, 0);
const aicCores = coreType.filter((t) => t === 'aic').length;
const aivCores = coreType.length - aicCores;
const aicKernel = sum(perCore.filter((_, i) => coreType[i] === 'aic').map((c) => c.kernel));
const aivKernel = sum(perCore.filter((_, i) => coreType[i] === 'aiv').map((c) => c.kernel));

const out = {
  source: {
    file: 'Data/pypto_qwen3_profiles/pypto_qwen3_profiles/tp1/swimlane/prefill.json',
    exporter: 'PyPTO L2 swimlane',
    model: 'Qwen3 · TP1',
    phase: 'prefill 的一次 L2 调用',
    notSame: '与 index.html（DSpark GBS1 STRACE）、device.html（DSpark GBS256 profiler trace）都不是同一个模型、不是同一次采集',
    clock: `核上周期计数（${(records.metadata.clock_freq_hz / 1e6).toFixed(0)} MHz），trace 原值`,
    /* 采集口径：核数、时钟、采集等级都取自 l2_swimlane_records.json 自己的 metadata；
     * 采集脚本是数据包里带的那个（collect_qwen3_profiles.py），仓库 skills 里没有对应流程。 */
    workload: `Qwen3-14B/TP1-fused-host/prefill_fwd x1/L2-swimlane-level${records.l2_swimlane_level}/${records.metadata.num_cores}core@${(records.metadata.clock_freq_hz / 1e6).toFixed(0)}MHz`,
    capture: 'collect_qwen3_profiles.py --mode swimlane（数据包自带的 TP=1 采集器，开 enable_l2_swimlane + enable_dep_gen）',
    captureNote: 'prompt 是采集脚本里那条固定的 chat 模板问句，不是压测负载；数据包 README 也写明这是某台机器上的一次真机采集，不能当跨机器基线。',
    // 没有对应的 skill，但 PyPTO 文档里有这类数据的读法
    analysis: 'pypto / pypto-serving 的 skills 里没有 L2 swimlane 流程，读法照 PyPTO 文档 docs/zh/user/performance/（00-swimlane 起：采集等级、dispatch→start 间隙、kernel 宽度、makespan 拆解）。本文件的 l2_swimlane_records.json / enable_l2_swimlane 是旧拼写，现名 chip_swimlane_records.json / enable_chip_swimlane，取值与语义不变。',
    decodeNote: '同目录的 decode.json 里 FuncId 全是 -1，任务映射不到算子名，所以这一页用 prefill',
    deps: `依赖来自每个任务的 fanin / fanout；deps.json 另有 ${deps.tasks.length} 个任务的张量形状与 ${deps.edges ? deps.edges.length : 0} 条边，本页没有用`,
  },
  cores: coreName.map((n, i) => ({ name: n, type: coreType[i], n: perCore[i].n, kernel: r2(perCore[i].kernel), setup: r2(perCore[i].setup) })),
  nCore: coreName.length,
  nAic: aicCores,
  nAiv: aivCores,
  spanMs: r2(spanMs),
  nRun: X.length,
  nTask: list.length,
  kernelMs: { aic: r2(aicKernel), aiv: r2(aivKernel) },
  occupancy: { aic: r2((aicKernel / (aicCores * spanMs)) * 100), aiv: r2((aivKernel / (aivCores * spanMs)) * 100) },
  setupMs: r2(sum(perCore.map((c) => c.setup))),
  groups: GROUPS.map((g) => ({ key: g.key, label: g.label })),
  ops,
  // [算子下标, fanin 任务下标[], fanout 任务下标[], 运行实例[[核, 起点 ms, kernel ms, setup ms]]]
  tasks: list.map((t) => ({
    o: opIdx[t.name],
    k: t.tok,
    in: t.fanin.map((x) => tokIdx[x]).filter((x) => x != null),
    out: t.fanout.map((x) => tokIdx[x]).filter((x) => x != null),
    r: t.runs.sort((a, b) => a[0] - b[0]).map(([ci, ts, k, s]) => [ci, r4((ts - t0) / 1000), r4(k / 1000), r4(s / 1000)]),
  })),
};

fs.writeFileSync(OUT, `/* Generated by build-core-data.cjs — do not edit by hand. */\nwindow.TRIAGE_CORE = ${JSON.stringify(out)};\n`);
console.log(`${out.nCore} 个核（AIC ${out.nAic} / AIV ${out.nAiv}）· 窗口 ${out.spanMs} ms · ${out.nRun} 个核上任务 / ${out.nTask} 个 SPMD 任务 · ${ops.length} 种算子`);
console.log(`AIC kernel ${out.kernelMs.aic} ms（占用 ${out.occupancy.aic}%）· AIV ${out.kernelMs.aiv} ms（${out.occupancy.aiv}%）· setup 合计 ${out.setupMs} ms`);
ops.slice(0, 6).forEach((o) => console.log(`   ${o.name.padEnd(28)} ×${String(o.n).padStart(5)}  kernel ${String(o.kernel).padStart(8)} ms  setup ${String(o.setup).padStart(6)} ms  核 ${o.cores}`));
console.log(`bytes ${fs.statSync(OUT).size}`);
