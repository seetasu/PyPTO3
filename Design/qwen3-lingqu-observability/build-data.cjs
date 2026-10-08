const fs = require('node:fs');
const path = require('node:path');

const repoRoot = path.resolve(__dirname, '../..');
const sourceRoot = path.join(repoRoot, 'Data/pypto_qwen3_profiles/pypto_qwen3_profiles/tp2');
const output = path.join(__dirname, 'data.json');
const layout = ['Embedding', 'RMS', 'QKV', 'QKV post', 'Attention prepare', 'QK', 'Softmax', 'PV', 'Context cast', 'O projection', 'AllReduce(O)', 'FFN RMS', 'Gate / Up', 'SwiGLU', 'Down', 'AllReduce(Down)', 'Residual tail'];

function parseCsv(text) {
  const rows = []; let row = []; let cell = ''; let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"' && quoted && text[i + 1] === '"') { cell += '"'; i++; }
    else if (c === '"') quoted = !quoted;
    else if (c === ',' && !quoted) { row.push(cell); cell = ''; }
    else if ((c === '\n' || c === '\r') && !quoted) {
      if (c === '\r' && text[i + 1] === '\n') i++;
      row.push(cell); if (row.some(value => value !== '')) rows.push(row); row = []; cell = '';
    } else cell += c;
  }
  if (cell !== '' || row.length) { row.push(cell); rows.push(row); }
  const headers = rows.shift();
  return rows.map(values => Object.fromEntries(headers.map((header, i) => [header, values[i] ?? ''])));
}

function loadStage(stage) {
  const file = path.join(sourceRoot, 'swimlane', `${stage}.json`);
  const raw = JSON.parse(fs.readFileSync(file, 'utf8'));
  const submits = raw.aicpu_orchestrator_phases[0];
  const submitById = new Map(submits.map(row => [String(row.task_id), row.submit_idx]));
  const tasks = raw.aicore_tasks.map(row => ({ core: row[0], token: String(row[1]), taskId: row[2], start: row[3], end: row[4], type: raw.metadata.core_types[row[0]] }));
  const ordered = tasks.sort((a, b) => submitById.get(a.token) - submitById.get(b.token));
  const base = Math.min(...ordered.map(t => t.start));
  const duration = t => (t.end - t.start) / raw.metadata.clock_freq_hz * 1e6;
  const startUs = t => (t.start - base) / raw.metadata.clock_freq_hz * 1e6;
  const layers = [];
  layers.push({ index: -1, name: 'Embedding', tasks: [ordered[0]].map(t => ({ ...t, startUs: startUs(t), durationUs: duration(t) })) });
  for (let layer = 0; layer < 40; layer++) {
    const batch = ordered.slice(1 + layer * 16, 1 + (layer + 1) * 16);
    layers.push({ index: layer, name: `Layer ${String(layer).padStart(2, '0')}`, tasks: batch.map((t, i) => ({ ...t, op: layout[i + 2], startUs: startUs(t), durationUs: duration(t) })) });
  }
  layers.push({ index: 40, name: 'LM Head / Tail', tasks: ordered.slice(-2).map(t => ({ ...t, startUs: startUs(t), durationUs: duration(t) })) });
  return { stage, capture: 'rank0', layerCount: 40, rawTaskCount: tasks.length, coreCount: raw.metadata.num_cores, aic: raw.metadata.core_types.filter(x => x === 'aic').length, aiv: raw.metadata.core_types.filter(x => x === 'aiv').length, wallMs: Math.max(...tasks.map(t => startUs(t) + duration(t))) / 1000, layers };
}

const profDir = path.join(sourceRoot, 'torch/prof');
const captureDir = fs.readdirSync(profDir).map(name => path.join(profDir, name)).find(p => fs.statSync(p).isDirectory());
function findOutputDir(dir) {
  if (fs.existsSync(path.join(dir, 'kernel_details.csv'))) return dir;
  for (const name of fs.readdirSync(dir)) {
    const child = path.join(dir, name);
    if (fs.statSync(child).isDirectory()) { const result = findOutputDir(child); if (result) return result; }
  }
  return null;
}
const outputDir = findOutputDir(captureDir);
if (!outputDir) throw new Error(`ASCEND_PROFILER_OUTPUT with kernel_details.csv not found under ${captureDir}`);
const kernelRows = parseCsv(fs.readFileSync(path.join(outputDir, 'kernel_details.csv'), 'utf8'));
const operatorRows = parseCsv(fs.readFileSync(path.join(outputDir, 'operator_details.csv'), 'utf8'));
const stepRows = parseCsv(fs.readFileSync(path.join(outputDir, 'step_trace_time.csv'), 'utf8'));
const trace = JSON.parse(fs.readFileSync(path.join(outputDir, 'trace_view.json'), 'utf8'));
const kernelMap = new Map();
for (const row of kernelRows) {
  const name = row.Name || 'Unknown';
  const item = kernelMap.get(name) || { name, type: row.Type, count: 0, durationUs: 0, maxUs: 0, device: row.Device_id };
  item.count++; item.durationUs += Number(row['Duration(us)']) || 0; item.maxUs = Math.max(item.maxUs, Number(row['Duration(us)']) || 0); kernelMap.set(name, item);
}
const topKernels = [...kernelMap.values()].sort((a,b) => b.durationUs - a.durationUs).slice(0, 12);
const operatorMap = new Map();
for (const row of operatorRows) {
  const name = row.Name || 'Unknown';
  const item = operatorMap.get(name) || { name, count: 0, hostUs: 0, deviceUs: 0 };
  item.count++; item.hostUs += Number(row['Host Total Duration(us)']) || 0; item.deviceUs += Number(row['Device Total Duration(us)']) || 0; operatorMap.set(name, item);
}
const topOperators = [...operatorMap.values()].sort((a,b) => b.deviceUs - a.deviceUs || b.hostUs - a.hostUs).slice(0, 10);
const stages = ['prefill', 'decode'].map(loadStage);
const stepMetrics = stepRows.map(row => ({ step: Number(row.Step), device: Number(row.Device_id), stage: Number(row.Stage), computing: Number(row.Computing), communication: Number(row.Communication), free: Number(row.Free), bubble: Number(row.Bubble) }));
const data = {
  model: { name: 'Qwen3-14B', parallel: 'TP = 2', layers: 40, profileDate: '2026-08-14', note: '真实设备采集；单次 profile，不作为跨机器基线' },
  topology: [
    { id: 'superpod', label: 'SuperPoD / 集群', level: 0, state: 'planned', detail: 'profile 未包含集群拓扑' },
    { id: 'rack', label: '机柜', level: 1, state: 'missing', detail: '无机柜编号或资产映射' },
    { id: 'node', label: '计算节点', level: 2, state: 'missing', detail: '无 Rank → 节点映射' },
    { id: 'rank0', label: 'Rank 0', level: 3, state: 'captured', detail: '泳道与 Torch Profiler 数据来自 rank0' },
    { id: 'device0', label: 'Device 0', level: 4, state: 'captured', detail: '60 AICore（20 AIC + 40 AIV）' },
    { id: 'rank1', label: 'Rank 1', level: 3, state: 'missing', detail: '采集验收经过 Rank 1；详细 rank1 trace 未保留在当前目录' },
    { id: 'planes', label: '7 个平面', level: 2, state: 'missing', detail: '无物理 plane / rail 映射' },
    { id: 'links', label: '节点间链路', level: 3, state: 'missing', detail: '无 link topology、端口计数器或时延' },
    { id: 'l1', label: 'L1 交换芯片', level: 3, state: 'missing', detail: '无交换芯片遥测' },
    { id: 'l2', label: 'L2 交换设备', level: 3, state: 'missing', detail: '无 SNMP / NETCONF / Telemetry 指标' },
  ],
  coverage: { rank0: true, rank1: false, nodeMapping: false, planeMapping: false, linkMetrics: false, l1Metrics: false, l2Metrics: false, clocksAligned: false, separateCaptureProcesses: true },
  sources: ['swimlane/{prefill,decode}.json', 'torch/prof/.../kernel_details.csv', 'operator_details.csv', 'step_trace_time.csv', 'trace_view.json'],
  stages, topKernels, topOperators, stepMetrics,
  traceEventCount: Array.isArray(trace.traceEvents) ? trace.traceEvents.length : 0,
  operatorCount: operatorRows.length,
  kernelCount: kernelRows.length,
  sampling: { prefillMs: 328.098, decodeMs: 329.974, source: 'README 中记录的 rank0 task swimlane wall time' },
};
fs.writeFileSync(output, JSON.stringify(data));
console.log(`Wrote ${path.relative(repoRoot, output)} (${stages.map(s => `${s.stage}: ${s.rawTaskCount} tasks`).join(', ')}, ${kernelRows.length} kernels)`);
