import data from './data.json' with { type: 'json' };

let stageName = 'prefill';
let selected = 'rank0';
let activeOperator = null;
const $ = (selector) => document.querySelector(selector);
const stage = () => data.stages.find(item => item.stage === stageName);
const number = value => new Intl.NumberFormat('zh-CN').format(value);
const escapeHtml = value => String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);

function renderMetrics() {
  const current = stage();
  const values = [
    ['模型与并行', data.model.name, data.model.parallel],
    ['执行阶段', stageName === 'prefill' ? 'Prefill' : 'Decode', `${current.layerCount} 层 · fused host 单次运行`],
    ['设备泳道', `${current.rawTaskCount}`, `Device 0 · 20 AIC cores + 40 AIV cores`],
    ['设备 wall time', current.wallMs.toFixed(2), 'ms · Rank 0 L2 task trace'],
  ];
  $('#metrics').innerHTML = values.map(([label, value, sub], i) => `<div class="metric"><div class="metric-label">${label}</div><div class="metric-value">${value}${i === 0 ? '' : i === 3 ? '<em>ms</em>' : ''}</div><div class="metric-sub">${sub}</div></div>`).join('');
}

function renderHierarchy() {
  $('#hierarchy').innerHTML = data.topology.map(item => `<button class="h-node${item.level ? ' indent' : ''}${selected === item.id ? ' selected' : ''}" style="margin-left:${item.level * 12}px;width:calc(100% - ${item.level * 12}px)" data-id="${item.id}" data-state="${item.state}"><i class="dot"></i><strong>${item.label}</strong><small>${item.state === 'captured' ? '已采集' : item.state === 'planned' ? '规划层级' : '未采集'}</small></button>`).join('');
  const current = data.topology.find(item => item.id === selected) || data.topology[3];
  $('#evidence').innerHTML = activeOperator
    ? `<b>${escapeHtml(activeOperator)}</b>　算子来自 Torch Profiler Device 0；本采集与 L2 task trace 分属独立进程，不做逐层/逐事件配对。`
    : `<b>${escapeHtml(current.label)}</b>　${escapeHtml(current.detail)}`;
  $('#hierarchy').querySelectorAll('.h-node').forEach(button => button.addEventListener('click', () => { selected = button.dataset.id; renderHierarchy(); }));
}

function renderFabric() {
  $('#fabric').innerHTML = Array.from({ length: 7 }, (_, i) => `<div class="plane"><b>P${i + 1}</b><small>未采集</small></div>`).join('');
  $('#timeline-title').textContent = `${stageName === 'prefill' ? 'Prefill' : 'Decode'} · 40 层执行泳道`;
}

function renderTimeline() {
  const current = stage();
  const layerData = current.layers.filter(item => item.index >= 0 && item.index < 40);
  const max = Math.max(...layerData.flatMap(layer => layer.tasks.map(task => task.durationUs)));
  $('#timeline').innerHTML = layerData.map(layer => {
    const span = Math.max(...layer.tasks.map(task => task.startUs + task.durationUs)) - Math.min(...layer.tasks.map(task => task.startUs));
    return `<div class="layer-row"><span class="layer-label">${layer.name}</span><div class="layer-bars">${layer.tasks.map((task, i) => `<i class="task-bar" style="--bar:${task.type === 'aic' ? 'var(--orange)' : 'var(--blue)'};--h:${Math.max(4, 5 + task.durationUs / max * 13)}px" title="${layer.name} · ${task.op || `Task ${i + 1}`} · ${task.type.toUpperCase()} core ${task.core} · ${task.durationUs.toFixed(2)} µs"></i>`).join('')}</div><span class="layer-time">${(span / 1000).toFixed(1)} ms</span></div>`;
  }).join('');
  $('#timeline-total').textContent = `整图 wall time ${current.wallMs.toFixed(2)} ms`;
}

function renderKernels() {
  const kernels = data.topKernels;
  const max = Math.max(...kernels.map(item => item.durationUs));
  $('#kernel-count').textContent = `${number(data.kernelCount)} 次 kernel 记录`;
  $('#kernels').innerHTML = kernels.map(item => `<div class="kernel-row"><span class="kernel-name" title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</span><span class="kernel-track"><i style="width:${Math.max(2, item.durationUs / max * 100)}%"></i></span><span class="kernel-time">${item.durationUs.toFixed(1)} µs</span></div>`).join('');
}

function renderOperators() {
  const rows = data.topOperators;
  const max = Math.max(...rows.map(item => item.deviceUs), ...rows.map(item => item.hostUs), 1);
  $('#operators').innerHTML = rows.map(item => `<button class="kernel-row op-row" data-op="${escapeHtml(item.name)}" title="查看 ${escapeHtml(item.name)} 的采集范围"><span class="kernel-name">${escapeHtml(item.name)}<small class="op-count">×${item.count}</small></span><span class="kernel-track"><i style="width:${Math.max(2, item.deviceUs / max * 100)}%;background:linear-gradient(90deg,#967cf0,#c2a8ff)"></i></span><span class="kernel-time">${item.deviceUs ? `${item.deviceUs.toFixed(1)} µs` : `${item.hostUs.toFixed(1)} µs host`}</span></button>`).join('');
  $('#operators').querySelectorAll('[data-op]').forEach(button => button.addEventListener('click', () => { activeOperator = button.dataset.op; selected = 'device0'; renderHierarchy(); }));
}

function renderSteps() {
  const steps = data.stepMetrics;
  const max = Math.max(...steps.map(item => item.stage));
  $('#steps').innerHTML = steps.map(item => `<div class="step-row"><span class="step-id">S${item.step}</span><span class="step-track"><i style="width:${item.computing / max * 100}%" title="计算"></i><i style="width:${item.communication / max * 100}%" title="通信分类"></i><i style="width:${item.free / max * 100}%" title="空闲"></i></span><span class="step-time">${(item.stage / 1000).toFixed(1)} ms</span></div>`).join('');
}

function render() { renderMetrics(); renderHierarchy(); renderFabric(); renderTimeline(); renderKernels(); renderOperators(); renderSteps(); }
document.querySelectorAll('[data-stage]').forEach(button => button.addEventListener('click', () => {
  stageName = button.dataset.stage;
  activeOperator = null;
  document.querySelectorAll('[data-stage]').forEach(item => item.classList.toggle('active', item === button));
  render();
}));
render();
