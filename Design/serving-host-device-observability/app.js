const data = window.REQUEST_OVERVIEW_DATA;
const { run } = data;
const svg = document.querySelector('#request-chart');
const SVG_NS = 'http://www.w3.org/2000/svg';
let mobileChart = window.innerWidth <= 760;
let chartWidth;
let plot;
let plotWidth;
let plotHeight;
const ordered = [...data.items].sort((a, b) => a.startMs - b.startMs);
const maxTtft = Math.ceil(Math.max(...ordered.map((item) => item.ttftMs), run.p99TtftMs) / 500) * 500;
const itls = ordered.map((item) => item.itlMs).sort((a, b) => a - b);
const itlP50 = itls[Math.floor(itls.length * 0.5)];
const itlP90 = itls[Math.floor(itls.length * 0.9)];

function updateChartGeometry() {
  mobileChart = window.innerWidth <= 760;
  chartWidth = mobileChart ? 430 : 1000;
  plot = { left: mobileChart ? 36 : 58, right: chartWidth - (mobileChart ? 12 : 18), top: 18, bottom: 318 };
  plotWidth = plot.right - plot.left;
  plotHeight = plot.bottom - plot.top;
}

function svgElement(tag, attributes = {}, parent = svg) {
  const element = document.createElementNS(SVG_NS, tag);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
  parent.append(element);
  return element;
}

function formatSeconds(milliseconds, digits = 2) {
  return `${(milliseconds / 1000).toFixed(digits)} s`;
}

function buildChart() {
  svg.replaceChildren();
  svg.setAttribute('viewBox', `0 0 ${chartWidth} 360`);
  for (let tick = 0; tick <= maxTtft; tick += 1000) {
    const y = plot.bottom - (tick / maxTtft) * plotHeight;
    svgElement('line', { x1: plot.left, y1: y, x2: plot.right, y2: y, class: 'grid-line' });
    const label = svgElement('text', { x: plot.left - 12, y: y + 4, class: 'axis-label', 'text-anchor': 'end' });
    label.textContent = tick === 0 ? '0' : `${tick / 1000}s`;
  }

  [64, 128, 192, 256].forEach((tick) => {
    const x = plot.left + (tick / ordered.length) * plotWidth;
    svgElement('line', { x1: x, y1: plot.top, x2: x, y2: plot.bottom, class: 'grid-line vertical-grid' });
    const label = svgElement('text', { x, y: plot.bottom + 20, class: 'axis-label', 'text-anchor': 'middle' });
    label.textContent = String(tick);
  });

  [
    { value: run.medianTtftMs, label: 'P50', className: 'median-line' },
    { value: run.p99TtftMs, label: 'P99', className: 'p99-line' },
  ].forEach(({ value, label, className }) => {
    const y = plot.bottom - (value / maxTtft) * plotHeight;
    svgElement('line', { x1: plot.left, y1: y, x2: plot.right, y2: y, class: `reference-line ${className}` });
    const tag = svgElement('text', { x: plot.right - 4, y: y - 5, class: `reference-label ${className}`, 'text-anchor': 'end' });
    tag.textContent = `${label} ${formatSeconds(value, 2)}`;
  });

  ordered.forEach((item, index) => {
    const x = plot.left + (index / (ordered.length - 1)) * plotWidth;
    const y = plot.bottom - (item.ttftMs / maxTtft) * plotHeight;
    const band = item.itlMs <= itlP50 ? 'low' : item.itlMs <= itlP90 ? 'mid' : 'high';
    const point = svgElement('circle', {
      cx: x,
      cy: y,
      r: index === 0 ? 4 : 3,
      class: `request-point itl-${band}${index === 0 ? ' is-selected' : ''}`,
      tabindex: '0',
      role: 'button',
      'aria-label': `样本 ${item.sample}，TTFT ${Math.round(item.ttftMs)} 毫秒，平均 ITL ${Math.round(item.itlMs)} 毫秒`,
    });
    const title = svgElement('title', {}, point);
    title.textContent = `样本 #${String(item.sample).padStart(3, '0')} · TTFT ${formatSeconds(item.ttftMs)} · 平均 ITL ${Math.round(item.itlMs)} ms`;
    point.addEventListener('click', () => selectSample(index));
    point.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        selectSample(index);
      }
    });
  });
}

function selectSample(index) {
  const item = ordered[index];
  svg.querySelectorAll('.request-point').forEach((point, pointIndex) => {
    point.classList.toggle('is-selected', pointIndex === index);
  });
  document.querySelector('#selected-sample').textContent = `#${String(item.sample).padStart(3, '0')}`;
  document.querySelector('#selected-ttft').textContent = formatSeconds(item.ttftMs);
  document.querySelector('#selected-itl').textContent = `${Math.round(item.itlMs)} ms`;
  document.querySelector('#selected-input').textContent = run.inputTokens;
  document.querySelector('#selected-output').textContent = run.outputTokens;
}

function buildRanks() {
  const hostOne = document.querySelector('#host-one-ranks');
  const hostTwo = document.querySelector('#host-two-ranks');
  const rankDetails = new Map();
  const rankButtons = [];

  function selectRank(rank) {
    rankButtons.forEach((button, index) => {
      button.classList.toggle('is-selected', index === rank);
      button.setAttribute('aria-pressed', String(index === rank));
    });
    const detail = rankDetails.get(rank);
    document.querySelector('#inspector-rank').textContent = `RANK ${String(rank).padStart(2, '0')}`;
    document.querySelector('#inspector-device').textContent = `Ascend 910B · Device ${detail.device} · Host 0${detail.host + 1}`;
    document.querySelector('#inspector-core').textContent = `${detail.core}%`;
    document.querySelector('#inspector-memory').textContent = `${detail.memory}%`;
    document.querySelector('#inspector-duration').textContent = `${720 + (rank * 37) % 120} ms`;
  }

  for (let rank = 0; rank < 16; rank += 1) {
    const core = 64 + ((rank * 13 + 17) % 31);
    const memory = 48 + ((rank * 9 + 15) % 28);
    const host = rank < 8 ? 0 : 1;
    const device = rank % 8;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `rank${rank === 7 ? ' is-selected' : ''}`;
    button.style.setProperty('--load', `${core}%`);
    button.setAttribute('aria-pressed', String(rank === 7));
    button.setAttribute('aria-label', `Rank ${String(rank).padStart(2, '0')}, Device ${device}, AI Core ${core}%, HBM ${memory}%`);
    button.innerHTML = `<span class="rank-top"><span class="rank-id">R${String(rank).padStart(2, '0')}</span><i class="rank-status"></i></span><span class="rank-value"><b>AIC ${core}%</b><span>HBM ${memory}%</span></span><i class="rank-bar"></i>`;
    button.addEventListener('click', () => selectRank(rank));
    rankDetails.set(rank, { core, memory, host, device });
    rankButtons.push(button);
    (host === 0 ? hostOne : hostTwo).append(button);
  }

  selectRank(7);
}

function buildRunSummary() {
  document.querySelector('#run-model').textContent = run.model;
  document.querySelector('#run-config').textContent = run.config;
  document.querySelector('#run-date').textContent = run.date;
  document.querySelector('#completed-count').textContent = run.completed;
  document.querySelector('#total-count').textContent = run.requests;
  document.querySelector('#failed-count').textContent = run.failed;
  document.querySelector('#request-throughput').textContent = run.requestPerSec.toFixed(2);
  document.querySelector('#output-throughput').textContent = Math.round(run.outputTokensPerSec).toLocaleString('en-US');
  document.querySelector('#ttft-p50').textContent = formatSeconds(run.medianTtftMs, 2);
  document.querySelector('#ttft-p99').textContent = formatSeconds(run.p99TtftMs, 2);
  document.querySelector('#total-count').textContent = run.requests;
}

buildRunSummary();
updateChartGeometry();
buildChart();
selectSample(0);
buildRanks();

window.addEventListener('resize', () => {
  if ((window.innerWidth <= 760) !== mobileChart) {
    updateChartGeometry();
    buildChart();
    selectSample(0);
  }
});
