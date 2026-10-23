/* Execution Triage · Home（DSpark Serving）
 * 中间是一个 step 的平面泳道：Serving / Host / Device 三条带，x 只表示时间。
 * Host 与 Device 的 16 行是同 16 条 host lane（每张卡一条）的两个阶段：bind.args 与 runner_run。
 * 红色折线 = 关键路径；灰色箭头 = 非关键的流转；虚线 = 等待；虚框 = 与本步重叠的下一步准备。
 * 数据见 build-home-data.cjs。 */
(function () {
  'use strict';

  const RUN = window.TRIAGE_HOME.runs[0];
  const LAYERS = ['serving', 'host', 'device'];
  const NAME = { serving: 'Serving', host: 'Host', device: 'Device' };
  /* Worker 进程是三个线程在并发跑（准备 ∥ 下发 ∥ 回收），所以单步视图里一个线程一行；
   * 压在一行会相互覆盖——prepare 有 97% 的时间落在 reclaim 的区间内。行名与顺序来自数据。 */
  const WROWS = RUN.workerRows && RUN.workerRows.length
    ? RUN.workerRows
    : [{ key: 'worker', label: 'Worker', thread: '' }];
  const WROW = Object.fromEntries(WROWS.map((r) => [r.key, r]));
  const WORK_TASK = {
    prepare: { label: '准备', desc: 'MainThread 执行 prepare_decode，做 decode 前置准备。' },
    dispatch: { label: '下发', desc: 'pypto-device 执行 dispatch_decode，把当前 step 下发到 device lane。' },
    reclaim: { label: '回收', desc: 'pypto-output 执行 reclaim_decode，等待设备完成并回收 step 输出。' },
  };
  const NS = 'http://www.w3.org/2000/svg';
  const $ = (s) => document.querySelector(s);
  const r2 = (v) => Math.round(v * 100) / 100;
  const esc = (v) => String(v).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

  /* ------------------------------------------------------------ P90 筛查步归因
   * 超过 decode P90 的那几步，各自相对中位步偏高的分项——与中位步逐项相减得到。
   * 「这一步长」在全程图上看得见，「长在哪一层」看不见，所以在这里先算出来：
   * 图上用所属带的淡红块表示，右栏给清单，下钻后给一句话。
   * 全部来自 runtime.byStep 的实测分项，没有新的假设。 */
  const SLOW = (() => {
    const ps = RUN.perStep;
    const B = RUN.runtime.byStep;
    const ids = Object.keys(B).filter((id) => B[id].decode);
    // 分项 → 它属于哪条带（Host 含 Worker 下发，与左上角构成一致）
    const PARTS = [
      { k: 'device', name: 'Device', band: 'device' },
      { k: 'h2d', name: 'bind.args', band: 'host' },
      { k: 'worker', name: 'Worker 下发', band: 'host' },
      { k: 'launch', name: 'lane 启动', band: 'host' },
      { k: 'serving', name: 'Serving 调度', band: 'serving' },
    ];
    const med = {};
    PARTS.forEach(({ k }) => {
      const a = ids.map((id) => B[id][k]).sort((x, y) => x - y);
      med[k] = a[Math.floor(a.length / 2)];
    });
    const items = ids.filter((id) => B[id].period > ps.p90Ms).map((id) => {
      const b = B[id];
      const over = PARTS.map((p) => ({ ...p, d: b[p.k] - med[p.k] })).filter((p) => p.d > 0.5).sort((x, y) => y.d - x.d);
      const lead = over[0];
      const second = over[1];
      // 两项都占大头时一起写，否则只写最大的那一项
      const both = lead && second && second.d >= lead.d * 0.6;
      return {
        id,
        period: b.period,
        excess: b.period - ps.p50Ms,
        over,
        // 图上压暗的带 = 多出来的时间落在哪一层；两项都占大头时两条带都标
        bands: lead ? [...new Set(both ? [lead.band, second.band] : [lead.band])] : ['host'],
        tag: lead ? (both ? `${lead.name} + ${second.name}` : lead.name) : '—',
        // 只写站得住的那一两项：次项小于大头的 20% 就不提，免得把 +2 ms 也念出来
        detail: over.filter((p, i) => i === 0 || p.d >= Math.max(2, over[0].d * 0.2)).slice(0, 2)
          .map((p) => `${p.name} +${fmtMs(p.d)}`).join(' · '),
      };
    }).sort((x, y) => y.excess - x.excess);
    const excess = items.reduce((s, x) => s + x.excess, 0);
    return {
      n: items.length,
      of: ids.length,
      items,
      excess,
      sharePct: (excess / RUN.request.e2eMs) * 100,
      byId: Object.fromEntries(items.map((x) => [x.id, x])),
    };
  })();

  function fmtMs(ms) {
    const v = Math.abs(ms);
    if (v >= 1000) return `${(ms / 1000).toFixed(2)} s`;
    if (v >= 100) return `${Math.round(ms)} ms`;
    if (v >= 10) return `${ms.toFixed(1)} ms`;
    return `${ms.toFixed(2)} ms`;
  }

  /* ------------------------------------------------------------ geometry
   * 平面泳道：x 只表示时间（ms，0 = 上一步结束），y 只表示层与 lane。
   * 三条带从上到下依次是 Serving / Host（Worker + 16 条 lane 的 bind）/ Device（16 条 lane 的 runner_run）。
   * 同一时刻在所有行上垂直对齐，所以一条竖线就能读出“此刻每一层在做什么”。 */
  const PAD = { l: 64, r: 140, t: 52, b: 30 };
  let W = 1000;        // 时间轴长度：按容器比例逐次计算，让泳道左右撑满而不是居中留白
  const ROW_H = 10;    // 每条 lane 的行距：条高之外留出空白，16 条才读得出是 16 条
  const BAR = 4;       // lane 条高：一条线的粗细，lane 之间靠空白分开
  const BAR_R = 1;     // 细条的圆角：再大就从线变成胶囊
  const HEAD_H = 16;   // Serving / Worker 行高（比 lane 行距高一档，行名才站得住）
  const GAP = 32;      // 带间距（含每条带上方的带名行）
  const PAD_IN = 6;    // 带内上下留白
  const SUB_GAP = 8;   // Worker 行与 16 条 lane 之间
  const PAD_R_E2E = 40; // 全程视图不需要右侧注记

  // viewBox 与容器同比例时 meet 缩放刚好铺满；比例不符，短的那一边就会被居中留白
  function fitW(height, padR) {
    const r = svg.getBoundingClientRect();
    if (!r.width || !r.height) return 1000;
    return Math.max(520, Math.round(height * (r.width / r.height) - PAD.l - padR));
  }
  /* Worker 的行：一个线程一行；线程内部还可能有调用栈嵌套（execute_step 里套 batch_prefill），
   * 那就在这个线程下面再铺一行。返回每行占几个子行，以及每个 span 落在第几层。 */
  function workerLayout(spans) {
    const depth = new Map();
    let total = 0;
    const rows = WROWS.map((r) => {
      const mine = (spans || []).filter((w) => (w.t && WROW[w.t] ? w.t : WROWS[0].key) === r.key)
        .sort((a, b) => a.s - b.s || b.d - a.d);
      const open = [];
      let max = 0;
      mine.forEach((w) => {
        let d = 0;
        while (open[d] != null && w.s < open[d] - 0.001) d += 1;
        open.length = d + 1;
        open[d] = w.s + w.d;
        depth.set(w, d);
        max = Math.max(max, d);
      });
      const row = { ...r, at: total, span: max + 1 };
      total += max + 1;
      return row;
    });
    return { rows, depth, total: Math.max(1, total) };
  }
  /* 带内留白一律是 PAD_IN，上下对称。Host 带只有单步视图才有 Worker 的那几行，
   * 全程视图不留它们的位置，否则 lane 组会被顶到底边、顶部空一块。 */
  function layout(nLane, withWorker, nWorkerRows) {
    // Worker 的行比 lane 行高一档（HEAD_H）：行名要和 Serving 一样能站住，11 的行距会让三个行名贴在一起
    const headRoom = withWorker ? (nWorkerRows || WROWS.length) * HEAD_H + SUB_GAP : 0;
    const serving = { y: PAD.t, h: HEAD_H + PAD_IN * 2 };
    const host = { y: serving.y + serving.h + GAP, h: headRoom + nLane * ROW_H + PAD_IN * 2, lanes0: PAD_IN + headRoom };
    const device = { y: host.y + host.h + GAP, h: nLane * ROW_H + PAD_IN * 2, lanes0: PAD_IN };
    return { serving, host, device, total: device.y + device.h, withWorker };
  }
  function rows(L, WL) {
    // Worker 的每个线程一行（准备 / 下发 / 回收），和 Host 的「一条 lane 一行」同一套规则
    const worker = (i) => L.host.y + PAD_IN + i * HEAD_H + HEAD_H / 2;
    const rowOf = (key) => (WL ? WL.rows.find((r) => r.key === key) : null);
    return {
      serving: L.serving.y + L.serving.h / 2,
      worker,
      // 线程行的第一层；嵌套调用由 workerLayout 给出的 depth 再往下挪一层
      workerOf: (key, d) => worker((rowOf(key) ? rowOf(key).at : 0) + (d || 0)),
      host: (dev) => L.host.y + L.host.lanes0 + dev * ROW_H + ROW_H / 2,
      device: (dev) => L.device.y + L.device.lanes0 + dev * ROW_H + ROW_H / 2,
    };
  }
  function el(name, attrs, parent) {
    const n = document.createElementNS(NS, name);
    Object.entries(attrs || {}).forEach(([k, v]) => n.setAttribute(k, v));
    if (parent) parent.appendChild(n);
    return n;
  }
  // SVG 图形会随泳道容器一起缩放；文字按屏幕像素反向补偿，避免超宽或高窗口中注记突然变大。
  function keepSvgTextSize(node, px = 14) {
    const box = svg.getBoundingClientRect();
    const viewHeight = svg.viewBox.baseVal.height;
    if (box.height > 0 && viewHeight > 0) node.style.fontSize = `${(px * viewHeight) / box.height}px`;
    return node;
  }
  function enforceSvgMinTextSize(minPx = 12) {
    const box = svg.getBoundingClientRect();
    const viewHeight = svg.viewBox.baseVal.height;
    if (!(box.height > 0 && viewHeight > 0)) return;
    const scale = box.height / viewHeight;
    svg.querySelectorAll('text').forEach((node) => {
      const currentPx = Number.parseFloat(getComputedStyle(node).fontSize);
      if (Number.isFinite(currentPx) && currentPx * scale < minPx) {
        node.style.fontSize = `${minPx / scale}px`;
      }
    });
  }
  // 泳道轴标签按屏幕像素定在 12px：大视口下不跟着 SVG 一起放大，小视口下仍由上面的最小字号兜底。
  function normalizeLaneAxisTextSize(targetPx = 12) {
    const box = svg.getBoundingClientRect();
    const viewHeight = svg.viewBox.baseVal.height;
    if (!(box.height > 0 && viewHeight > 0)) return;
    const scale = box.height / viewHeight;
    svg.querySelectorAll('.tick-label, .lane-label, .lane-tick').forEach((node) => {
      node.style.fontSize = `${targetPx / scale}px`;
      node.style.fontWeight = '400';
    });
  }

  /* ------------------------------------------------------------ state */
  const stepIds = RUN.request.timeline.map((s) => s.id);
  const TABS = ['path', 'dist', 'plan', 'steps'];
  const VIEWS = ['e2e', 'step'];
  const state = { layer: RUN.bottleneck, step: RUN.defaultStep, picked: false, tab: 'path', view: 'e2e', lane: null, hover: null, timeRange: null, pan: null, opsStep: null };
  const tower = () => RUN.towers[state.layer];

  function fullTimeRange() {
    if (state.view === 'e2e') return { start: 0, end: RUN.e2e.durMs, context: 'e2e' };
    const f = RUN.flows[state.step];
    return { start: f.t0, end: f.t1, context: `step:${state.step}` };
  }
  function visibleTimeRange() {
    const full = fullTimeRange();
    return state.timeRange && state.timeRange.context === full.context ? state.timeRange : full;
  }
  function updateTimeControls() {
    const full = fullTimeRange();
    const visible = visibleTimeRange();
    const zoom = full.end > full.start ? (full.end - full.start) / (visible.end - visible.start) : 1;
    const out = $('#timeZoomLevel');
    if (out) out.textContent = `${zoom >= 10 ? Math.round(zoom) : zoom.toFixed(1)}×`;
    const reset = $('#timeZoomReset');
    if (reset) reset.disabled = zoom < 1.01;
    const minus = $('#timeZoomOut');
    if (minus) minus.disabled = zoom < 1.01;
    const plus = $('#timeZoomIn');
    if (plus) plus.disabled = zoom >= 127;
  }
  function zoomTime(factor, anchorRatio = 0.5) {
    const full = fullTimeRange();
    const current = visibleTimeRange();
    const fullSpan = full.end - full.start;
    const minSpan = fullSpan / 128;
    const nextSpan = Math.max(minSpan, Math.min(fullSpan, (current.end - current.start) / factor));
    const anchor = current.start + (current.end - current.start) * Math.max(0, Math.min(1, anchorRatio));
    let start = anchor - nextSpan * anchorRatio;
    start = Math.max(full.start, Math.min(full.end - nextSpan, start));
    if (nextSpan >= fullSpan * 0.999) state.timeRange = null;
    else state.timeRange = { start, end: start + nextSpan, context: full.context };
    drawLanes();
  }
  function panTime(deltaFraction) {
    const full = fullTimeRange();
    const current = visibleTimeRange();
    const span = current.end - current.start;
    let start = current.start + span * deltaFraction;
    start = Math.max(full.start, Math.min(full.end - span, start));
    if (Math.abs(start - full.start) < fullSpanEpsilon(full)) state.timeRange = null;
    else state.timeRange = { start, end: start + span, context: full.context };
    drawLanes();
  }
  function fullSpanEpsilon(full) { return Math.max(1e-6, (full.end - full.start) * 1e-8); }
  function niceTickStep(span, target = 9) {
    const raw = Math.max(Number.EPSILON, span / target);
    const power = 10 ** Math.floor(Math.log10(raw));
    const unit = raw / power;
    return (unit <= 1 ? 1 : unit <= 2 ? 2 : unit <= 5 ? 5 : 10) * power;
  }
  function formatTick(value, step) {
    const decimals = step >= 1 ? 0 : Math.min(3, Math.ceil(-Math.log10(step)));
    return String(Number(value.toFixed(decimals)));
  }
  function addPlotClip(defs, height) {
    const clip = el('clipPath', { id: 'plotClip' }, defs);
    el('rect', { x: PAD.l, y: 0, width: W, height }, clip);
  }
  function drawMinorTicks(g, range, majorStep, X, y, notable = []) {
    const minorStep = majorStep / 5;
    if (!(minorStep > 0)) return;
    const epsilon = minorStep * 0.015;
    for (let t = Math.ceil(range.start / minorStep) * minorStep; t <= range.end + epsilon; t += minorStep) {
      const onMajor = Math.abs(t / majorStep - Math.round(t / majorStep)) < 0.015;
      const onNotable = notable.some((v) => Math.abs(t - v) < epsilon);
      if (onMajor || onNotable) continue;
      vline(g, X(t), y - 2.5, y, 'tick minor');
    }
  }

  function readHash() {
    const p = new URLSearchParams(location.hash.slice(1));
    if (LAYERS.includes(p.get('layer'))) state.layer = p.get('layer');
    if (stepIds.includes(p.get('step'))) { state.step = p.get('step'); state.picked = true; }
    if (TABS.includes(p.get('tab'))) state.tab = p.get('tab');
    if (VIEWS.includes(p.get('view'))) state.view = p.get('view');
    const lane = Number(p.get('lane'));
    if (p.get('lane') !== null && Number.isInteger(lane) && lane >= 0 && lane < RUN.dist.nLane) state.lane = lane;
  }
  function writeHash() {
    const h = `view=${state.view}&layer=${state.layer}${state.picked ? `&step=${encodeURIComponent(state.step)}` : ''}&tab=${state.tab}${state.lane == null ? '' : `&lane=${state.lane}`}`;
    if (location.hash.slice(1) !== h) history.replaceState(null, '', `#${h}`);
  }

  /* ------------------------------------------------------------ tooltip */
  const tip = $('#tip');
  function showTip(e, text) { tip.textContent = text; tip.hidden = false; moveTip(e); }
  function moveTip(e) {
    const r = tip.getBoundingClientRect();
    let x = e.clientX + 14; let y = e.clientY + 14;
    if (x + r.width > window.innerWidth - 8) x = e.clientX - r.width - 14;
    if (y + r.height > window.innerHeight - 8) y = e.clientY - r.height - 14;
    tip.style.left = `${Math.max(8, x)}px`; tip.style.top = `${Math.max(8, y)}px`;
  }
  function hideTip() { tip.hidden = true; }
  function withTip(node, text) {
    node.addEventListener('mouseenter', (e) => showTip(e, text));
    node.addEventListener('mousemove', moveTip);
    node.addEventListener('mouseleave', hideTip);
    return node;
  }

  /* ------------------------------------------------------------ scene */
  const svg = $('#iso');

  function bar(g, x0, x1, yc, h, cls, tipText) {
    const w = Math.max(1.2, x1 - x0);
    const n = el('rect', { class: `blk ${cls}`, x: x0, y: yc - h / 2, width: w, height: h, rx: h >= 8 ? 1.5 : BAR_R }, g);
    if (tipText) withTip(n, tipText);
    return n;
  }
  function deviceLegend(g, y, x) {
    el('title', {}, g).textContent = 'runner_run 执行与等待';
    el('rect', { class: 'device-legend-exec', x, y: y - 7, width: 8, height: 8, rx: 2 }, g);
    const execLabel = el('text', { class: 'device-legend-label', x: x + 14, y, 'text-anchor': 'start' }, g);
    execLabel.textContent = '执行';
    keepSvgTextSize(execLabel, 12);
    el('rect', { class: 'device-legend-wait', x: x + 48, y: y - 7, width: 8, height: 8, rx: 2 }, g);
    const waitLabel = el('text', { class: 'device-legend-label', x: x + 62, y, 'text-anchor': 'start' }, g);
    waitLabel.textContent = '等待';
    keepSvgTextSize(waitLabel, 12);
  }
  function deviceLaneBars(g, l, y, crit, X, stepId) {
    const start = l.rrStart ?? l.r0;
    const end = l.end ?? l.r1;
    const rrMs = end - start;
    const hasSplit = Number.isFinite(l.deviceExec) && Number.isFinite(l.waitEstimate);
    const classes = crit ? ' crit' : '';
    const label = `dev ${l.dev} · runner_run ${fmtMs(rrMs)}（trace 窗口）`;
    const detail = hasSplit
      ? `${label}＝执行 ${fmtMs(l.deviceExec)} + 等待 ${fmtMs(l.waitEstimate)}`
      : `${label}，含可能的同步等待；本步未拆分显示`;
    const attrs = (node) => {
      node.setAttribute('data-l', l.dev);
      if (stepId != null) node.setAttribute('data-s', stepId);
      return withTip(node, detail);
    };
    if (!hasSplit) {
      return [attrs(bar(g, X(start), X(end), y, BAR, `rr${classes}`, detail))];
    }
    const execEnd = Math.min(end, start + l.deviceExec);
    // 先画一条低对比的完整窗口作为等待底，再覆盖实色执行段；避免每条 lane 的等待段都带虚线边框。
    const window = el('rect', {
      class: `blk rr-window${classes}`, x: X(start), y: y - BAR / 2,
      width: Math.max(0.55, X(end) - X(start)), height: BAR, rx: BAR_R,
    }, g);
    attrs(window);
    if (execEnd - start <= 0.01) return [window];
    const exec = el('rect', {
      class: `blk rr-exec${classes}`, x: X(start), y: y - BAR / 2,
      width: Math.max(0.55, X(execEnd) - X(start)), height: BAR, rx: BAR_R,
    }, g);
    attrs(exec);
    return [window, exec];
  }
  const hline = (g, x0, x1, y, cls) => el('line', { class: cls, x1: x0, y1: y, x2: x1, y2: y }, g);
  const vline = (g, x, y0, y1, cls) => el('line', { class: cls, x1: x, y1: y0, x2: x, y2: y1 }, g);
  const pts = (list) => list.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ');

  /* 三条带的底板与带名。带名独占每条带上方的一行，侧栏只留给 lane 行名，两者不再抢同一块地方。
   * preTo 给“上一步期间”的压暗区右边界（单步视图用），全程视图传 null。 */
  // 行名点进这条 lane 的专项（右栏「设备分布 › lane N」）
  function laneLink(node, dev) {
    node.setAttribute('role', 'button');
    node.setAttribute('tabindex', '0');
    withTip(node, `lane ${dev} / dev ${dev} · 点击查看这条 lane 的专项`);
    const open = () => { selectTab('dist'); selectLane(dev); };
    node.addEventListener('click', open);
    node.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
    return node;
  }

  function drawBands(L, preTo, summary) {
    const x1 = PAD.l + W;
    LAYERS.forEach((layer) => {
      const b = L[layer];
      const on = state.layer === layer;
      const g = el('g', { class: `plate${on ? ' is-on' : ''}`, 'data-layer': layer }, svg);
      const mainFrom = preTo == null ? PAD.l : preTo;
      el('rect', { class: 'plate-main', x: mainFrom, y: b.y, width: x1 - mainFrom, height: b.h }, g);
      if (preTo != null) el('rect', { class: 'plate-pre', x: PAD.l, y: b.y, width: preTo - PAD.l, height: b.h }, g);
      el('rect', { class: 'plate-edge', x: PAD.l, y: b.y, width: W, height: b.h, rx: 3 }, g);
      // 带名行：阶段名称与关键统计靠左，图例靠右
      const ty = b.y - 8;
      const tag = el('g', { class: `tag${on ? ' is-on' : ''}`, tabindex: 0, role: 'button', 'aria-pressed': on, 'aria-label': NAME[layer] }, g);
      const title = el('text', { class: 'tag-label', x: 8, y: ty, 'text-anchor': 'start' }, tag);
      title.textContent = NAME[layer];
      keepSvgTextSize(title, 16);
      if (layer === 'device') {
        const split = state.view === 'e2e' || RUN.flows[state.step].lanes.some((l) => Number.isFinite(l.deviceExec));
        if (split) deviceLegend(g, ty, x1 - 92);
        else el('text', { class: 'tag-cap', x: x1 - 8, y: ty, 'text-anchor': 'end' }, g).textContent = 'runner_run 上界 · 未拆分';
      }
      if (summary) {
        const value = summary[layer];
        const upper = summary[layer + 'Upper'] ? '≤ ' : '';
        const svgHeight = svg.getBoundingClientRect().height;
        const textScale = svgHeight > 0 ? svg.viewBox.baseVal.height / svgHeight : 1;
        const statX = Number(title.getAttribute('x')) + title.getComputedTextLength() + 12 * textScale;
        const stat = el('text', { class: 'band-stat', x: statX, y: ty + 1, 'text-anchor': 'start' }, g);
        const totalLabel = el('tspan', { class: 'band-stat-label' }, stat);
        totalLabel.textContent = '累计 ';
        el('tspan', {}, stat).textContent = `${upper}${fmtMs(summary[layer + 'Total'])}`;
        const avgLabel = el('tspan', { class: 'band-stat-label' }, stat);
        avgLabel.textContent = '  ·  均/步 ';
        el('tspan', {}, stat).textContent = `${upper}${fmtMs(value)}`;
        el('title', {}, stat).textContent = `累计值与均值按 ${summary.count} 个 Decode step 统计；并行阶段的累计时间不能相加作为请求墙钟时长。`;
        keepSvgTextSize(stat, 16);
      }
      g.addEventListener('click', () => selectLayer(layer));
      tag.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectLayer(layer); } });
    });
  }

  function renderScene() {
    const f = RUN.flows[state.step];
    const nLane = f.lanes.length;
    const WL = workerLayout(f.worker);
    const L = layout(nLane, true, WL.total);
    const ROW = rows(L, WL);
    W = fitW(L.total + PAD.b, PAD.r);
    const VW = PAD.l + W + PAD.r;
    svg.innerHTML = '';
    svg.setAttribute('viewBox', `0 0 ${VW} ${L.total + PAD.b}`);
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    const range = visibleTimeRange();
    const rangeSpan = range.end - range.start;
    const X = (t) => PAD.l + ((t - range.start) / rangeSpan) * W;
    const x1 = PAD.l + W;

    const defs = el('defs', {}, svg);
    addPlotClip(defs, L.total + PAD.b);
    const arrow = el('marker', { id: 'arrowGrey', viewBox: '0 0 8 8', refX: 7, refY: 4, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, defs);
    el('path', { d: 'M0,0 L8,4 L0,8 z', fill: 'rgba(255,255,255,0.55)' }, arrow);
    const arrowHot = el('marker', { id: 'arrowHot', viewBox: '0 0 8 8', refX: 7, refY: 4, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, defs);
    el('path', { d: 'M0,0 L8,4 L0,8 z', fill: '#ef4b3c' }, arrowHot);

    // 1) 三条带（带名在每条带上方单行）
    drawBands(L, X(0));

    // 2) 关键 lane 的两行：整行压一层淡红，便于在 16 行里一眼找到
    const gc = el('g', {}, svg);
    [ROW.host(f.crit), ROW.device(f.crit)].forEach((y) => {
      el('rect', { class: 'crit-band', x: PAD.l, y: y - ROW_H / 2, width: W, height: ROW_H }, gc);
    });

    // 每条 Host lane 都会把 bind 交给同编号 Device lane；常驻画淡线，悬停时再强调单条配对。
    const lanePairs = el('g', { class: 'lane-pairs' }, svg);
    f.lanes.forEach((l) => {
      el('line', {
        class: 'lane-pair', 'data-l': l.dev,
        x1: X(l.bindEnd), y1: ROW.host(l.dev), x2: X(l.rrStart), y2: ROW.device(l.dev),
      }, lanePairs);
    });
    const K = f.links;
    const yDisp = ROW.workerOf('dispatch');
    if (K.dispatch) {
      const fanout = el('g', { class: 'flow' }, svg);
      f.lanes.forEach((l) => el('line', {
        class: 'lane-upstream', 'data-l': l.dev,
        x1: X(K.dispatch[1]), y1: yDisp + 5, x2: X(l.bindStart), y2: ROW.host(l.dev) - 4,
      }, fanout));
    }

    // 3) 时间参考线：本步时间原点；关键 lane 的起点由关键路径自身标出
    const guides = el('g', { class: 'guides' }, svg);
    vline(guides, X(0), L.serving.y - 6, L.device.y + L.device.h + 6, 'g-start');

    // 3.5) 每条 lane 铺一层整行透明热区：条细成一条线之后，光标不必正好压在线上。
    //      画在条之前，所以真压在条上时仍然是条自己的详情提示。
    const glh = el('g', { class: 'lane-hits' }, svg);
    f.lanes.forEach((l) => {
      const rrStart = l.rrStart ?? l.r0;
      const rrEnd = l.end ?? l.r1;
      const tip = `lane ${l.dev}（dev ${l.dev}）· bind ${fmtMs(l.bindEnd - l.bindStart)} → 等待 ${fmtMs(Math.max(0, rrStart - l.bindEnd))} → runner_run ${fmtMs(rrEnd - rrStart)}`;
      [['host', ROW.host(l.dev)], ['device', ROW.device(l.dev)]].forEach(([layer, y]) => {
        const hit = el('rect', { class: 'lane-hit', x: PAD.l, y: y - ROW_H / 2, width: W, height: ROW_H }, glh);
        hit.addEventListener('click', () => selectLayer(layer));
        pairHover(withTip(hit, tip), l, ROW, X);
      });
    });

    // 4) Serving 带
    const gs = el('g', {}, svg);
    f.serving.forEach((s) => {
      if (s.kind === 'wait') { hline(gs, X(s.s), X(s.s + s.d), ROW.serving, 'wait'); return; }
      bar(gs, X(s.s), X(s.s + s.d), ROW.serving, 11, s.kind, `Serving · ${s.n} ${fmtMs(s.d)}`);
    });

    // 5) Host 带：Worker 的三个线程各一行 + 16 条 lane 的 bind
    const gh = el('g', {}, svg);
    const reclaimWait = f.worker.find((w) => w.t === 'reclaim' && w.kind === 'wait');
    const currentReclaimTask = reclaimWait
      ? f.worker.find((w) => w.t === 'reclaim' && w.kind !== 'wait' && w.s >= reclaimWait.s + reclaimWait.d - 0.01)
      : null;
    f.worker.forEach((w) => {
      const row = w.t && WROW[w.t] ? w.t : WROWS[0].key;
      const d = WL.depth.get(w) || 0;
      const y = ROW.workerOf(row, d);
      const r = WROW[row];
      const meta = WORK_TASK[row] || { label: w.n, desc: w.n };
      const isWait = w.kind === 'wait';
      const isCurrentReclaimTask = w === currentReclaimTask;
      const isCurrent = w.kind === 'own' || isCurrentReclaimTask;
      const currentClass = isWait ? ' worker-waiting' : isCurrent ? ' worker-current' : ' worker-context';
      const taskClass = `worker-task worker-${row} ${w.kind}${currentClass}`;
      const tip = `${isWait ? '回收等待：等待当前 step 的输出可回收。' : isCurrentReclaimTask ? '当前 step 的回收 Task。' : `${meta.desc} `}事件：${w.n} · ${r.label}${r.thread ? `（${r.thread} 线程）` : ''} · ${fmtMs(w.d)}${d ? ' · 嵌套调用' : ''}${w.kind === 'next' ? ' · 属于下一步' : ''}`;
      if (isWait) {
        withTip(hline(gh, X(w.s), X(w.s + w.d), y, `wait worker-wait worker-${row} worker-waiting`), tip);
        if (X(w.s + w.d) - X(w.s) >= 82) {
          const text = el('text', { class: 'worker-task-label worker-wait-label', x: (X(w.s) + X(w.s + w.d)) / 2, y: y + 4, 'text-anchor': 'middle' }, gh);
          text.textContent = '回收等待';
          el('title', {}, text).textContent = tip;
        }
        return;
      }
      const x0 = X(w.s); const x1 = X(w.s + w.d);
      bar(gh, x0, x1, y, 13, taskClass, tip);
      if (isCurrent && x1 - x0 >= 58) {
        const text = el('text', { class: 'worker-task-label', x: (x0 + x1) / 2, y: y + 4, 'text-anchor': 'middle' }, gh);
        text.textContent = meta.label;
        el('title', {}, text).textContent = tip;
      }
    });
    let argsSum = 0;
    f.lanes.forEach((l) => {
      const y = ROW.host(l.dev); const crit = l.dev === f.crit;
      argsSum += l.bindEnd - l.bindStart;
      if (l.rrStart - l.bindEnd > 0.5) hline(gh, X(l.bindEnd), X(l.rrStart), y, 'wait');
      pairHover(bar(gh, X(l.bindStart), X(l.bindEnd), y, BAR, `bind${crit ? ' crit' : ''}`, `lane ${l.dev}（dev ${l.dev}）· bind.args 分配 + H2D ${fmtMs(l.bindEnd - l.bindStart)}`), l, ROW, X);
    });

    // 6) Device 带：16 张卡的 runner_run，末端是同步屏障
    const gd = el('g', {}, svg);
    f.lanes.forEach((l) => {
      const y = ROW.device(l.dev); const crit = l.dev === f.crit;
      deviceLaneBars(gd, l, y, crit, X).forEach((node) => pairHover(node, l, ROW, X));
    });
    // 同步屏障由关键路径的终点表达；不再用贯穿 Device 带的红色竖线重复标记。

    // 6.5) Worker 行名：一个线程一行，行名下挂线程名
    const gw = el('g', {}, svg);
    WL.rows.forEach((r) => {
      // 三个行名叠在 Host 带顶部，字号比 lane 行名小一档才站得开
      const t = el('text', { class: 'lane-tick worker-tick', x: PAD.l - 8, y: ROW.worker(r.at) + 3, 'text-anchor': 'end' }, gw);
      t.textContent = r.label;
      withTip(t, `${WORK_TASK[r.key]?.desc || 'Worker 任务'}${r.thread ? ` 线程：${r.thread}` : ''}${r.span > 1 ? ` · 存在嵌套调用，占 ${r.span} 行` : ''}`);
    });

    // 7) lane 行名：关键 lane 标红，其余每 4 条标一次
    const gl = el('g', {}, svg);
    f.lanes.forEach((l) => {
      const crit = l.dev === f.crit;
      if (!crit && l.dev % 4) return;
      // Host 行是 host 线程（lane N），Device 行是它驱动的那张卡（dev N），同一行号上下对应
      [[ROW.host(l.dev), `lane ${l.dev}`], [ROW.device(l.dev), `dev ${l.dev}`]].forEach(([y, text]) => {
        const t = el('text', { class: `${crit ? 'lane-label' : 'lane-tick'} is-link`, x: PAD.l - 8, y: y + 3, 'text-anchor': 'end' }, gl);
        t.textContent = text;
        laneLink(t, l.dev);
      });
    });

    // 8) 非关键流转（灰）：调度下发 → 所有 Host lane；屏障 → 回收 → 调度输出
    const gf = el('g', { class: 'flow' }, svg);
    const poly = (list, marker) => el('polyline', { class: 'flow-line', points: pts(list), 'marker-end': marker ? `url(#${marker})` : '' }, gf);
    // 下发与回收各自落在自己那条线程行上，箭头才对得上
    const yRecl = ROW.workerOf('reclaim');
    if (K.queue != null && K.dispatch) poly([[X(K.queue), ROW.serving + 6], [X(K.dispatch[0]), yDisp - 5]], 'arrowGrey');
    if (K.reclaimEnd != null && K.output != null) {
      // 从全 lane 共用的屏障中点连到回收线程，避免看起来像 dev 0 单独触发回收。
      poly([[X(f.end), L.device.y + L.device.h / 2], [X(K.reclaimEnd), yRecl + 5], [X(K.output), ROW.serving + 6]], 'arrowGrey');
    }

    // 9) 关键路径（红）：bind 结束 → 进入 runner_run → 同步屏障
    const c = f.lanes.find((l) => l.dev === f.crit);
    el('polyline', {
      class: 'crit-path',
      points: pts([[X(c.bindStart), ROW.host(f.crit)], [X(c.bindEnd), ROW.host(f.crit)], [X(c.rrStart), ROW.device(f.crit)], [X(f.end), ROW.device(f.crit)]]),
      pathLength: 1,
    }, svg);
    const measure = (xa, xb, y, text) => {
      const t = el('text', { class: 'crit-label', x: (xa + xb) / 2, y: y - 7, 'text-anchor': 'middle' }, svg);
      t.textContent = text;
    };
    measure(X(c.bindStart), X(c.bindEnd), ROW.host(f.crit), `bind / H2D ${fmtMs(f.hostMs)}`);
    measure(X(c.rrStart), X(f.end), ROW.device(f.crit), `Device ≤ ${fmtMs(f.deviceMs)}`);

    // 10) 右侧注记：说明这 16 行是什么，以及它们每步都在重新 bind
    const nx = x1 + 12;
    const note = (y, text, dim) => {
      const t = el('text', { class: `plate-note${dim ? ' dim' : ''}`, x: nx, y, 'text-anchor': 'start' }, svg);
      t.textContent = text;
      keepSvgTextSize(t);
    };
    const hy = ROW.host(Math.round(nLane / 2));
    note(hy - 10, `${nLane} 条 lane 每步重新 bind`);
    note(hy + 10, `本步合计 ${fmtMs(argsSum)}`, true);
    note(ROW.device(Math.round(nLane / 2)), '末端对齐 = 同步屏障', true);

    // 11) 时间轴
    const ga = el('g', { class: 'axis' }, svg);
    const yA = L.device.y + L.device.h + 10;
    const tickStep = niceTickStep(rangeSpan, Math.max(4, Math.floor(W / 85)));
    drawMinorTicks(ga, range, tickStep, X, yA, [range.start, range.end, f.critStart, f.end]);
    const ticks = [range.start, range.end, f.critStart, f.end];
    for (let t = Math.ceil(range.start / tickStep) * tickStep; t <= range.end; t += tickStep) ticks.push(t);
    [...new Set(ticks.map((t) => Number(t.toFixed(6))))]
      .filter((t) => t >= range.start - 1e-6 && t <= range.end + 1e-6)
      .sort((a, b) => a - b).forEach((t) => {
      const strong = t === 0 || t === f.critStart || t === f.end || t === range.start || t === range.end;
      vline(ga, X(t), yA - 6, yA, strong ? 'tick strong major' : 'tick major');
      if (!strong && [f.critStart, f.end].some((s) => Math.abs(X(s) - X(t)) < 22)) return;
      const tx = el('text', { class: `tick-label${strong ? ' strong' : ''}`, x: X(t), y: yA + 13, 'text-anchor': 'middle' }, ga);
      tx.textContent = formatTick(t, tickStep);
    });
    const u = el('text', { class: 'tick-label', x: x1 + 10, y: yA + 13, 'text-anchor': 'start' }, ga);
    u.textContent = 'ms';

    // 12) 当前 step
    const t1 = el('text', { class: 'step-title', x: x1, y: PAD.t - 10, 'text-anchor': 'end' }, svg);
    t1.textContent = `${state.step} STEP · ${fmtMs(f.periodMs)}`;
  }

  /* 全程泳道：整个请求的 54 个 step 画在一条时间轴上，三条带与单步视图一致。
   * 先在这里看整体节奏（冷启动、步长漂移、哪条 lane 一直偏慢），再点任一步下钻。 */
  function renderE2E() {
    const E = RUN.e2e;
    const nLane = E.steps[0].lanes.length;
    const summary = headScope();
    const L = layout(nLane, false, null);
    const ROW = rows(L);
    W = fitW(L.total + PAD.b, PAD_R_E2E);
    const VW = PAD.l + W + PAD_R_E2E;
    svg.innerHTML = '';
    svg.setAttribute('viewBox', `0 0 ${VW} ${L.total + PAD.b}`);
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    const range = visibleTimeRange();
    const rangeSpan = range.end - range.start;
    const X = (t) => PAD.l + ((t - range.start) / rangeSpan) * W;
    const x1 = PAD.l + W;
    const hot = RUN.dist.critLane;

    const defs = el('defs', {}, svg);
    addPlotClip(defs, L.total + PAD.b);

    // 1) 三条带（带名在每条带上方单行）
    drawBands(L, null, summary);

    // 2) 一直最晚进入 runner_run 的那条 lane，整行压淡红
    const gc = el('g', {}, svg);
    [ROW.host(hot), ROW.device(hot)].forEach((y) => {
      el('rect', { class: 'crit-band', x: PAD.l, y: y - ROW_H / 2, width: W, height: ROW_H }, gc);
    });

    // 3) 选中的 step：整列高亮，与下方时间线、右栏联动
    const sel = state.picked ? E.steps.find((s) => s.id === state.step) : null;
    if (sel) {
      el('rect', {
        class: 'step-span', x: X(sel.t0), y: L.serving.y - 6,
        width: Math.max(1.5, X(sel.t1) - X(sel.t0)), height: L.device.y + L.device.h + 6 - (L.serving.y - 6),
      }, svg);
    }

    // 4) Serving 带：每次 process_step_output 一根刻度
    const gs = el('g', {}, svg);
    E.outputs.forEach((t) => vline(gs, X(t), ROW.serving - 6, ROW.serving + 6, 'out-tick'));

    // 5) Host / Device 带：54 个 step × 16 条 lane
    const gh = el('g', {}, svg);
    const gd = el('g', {}, svg);
    E.steps.forEach((s) => {
      s.lanes.forEach((l) => {
        const crit = l.dev === s.crit;
        const hb = bar(gh, X(l.b0), X(l.b1), ROW.host(l.dev), BAR, `bind${crit ? ' crit' : ''}`);
        hb.setAttribute('data-s', s.id); hb.setAttribute('data-l', l.dev);
        deviceLaneBars(gd, l, ROW.device(l.dev), crit, X, s.id);
      });
    });

    // 6) lane 行名
    const gl = el('g', {}, svg);
    E.steps[0].lanes.forEach((l) => {
      const crit = l.dev === hot;
      if (!crit && l.dev % 4) return;
      [[ROW.host(l.dev), `lane ${l.dev}`], [ROW.device(l.dev), `dev ${l.dev}`]].forEach(([y, text]) => {
        laneLink(el('text', { class: `${crit ? 'lane-label' : 'lane-tick'} is-link`, x: PAD.l - 8, y: y + 3, 'text-anchor': 'end' }, gl), l.dev).textContent = text;
      });
    });

    // 7) 慢步：超过 decode P90 的整列压一层淡红，并在「多出来的时间所在的那条带」上再压一层
    //    y 本来就表示层，所以归因不用换颜色，落在哪条带上就是答案
    const top = L.serving.y - 6;
    const bot = L.device.y + L.device.h + 6;
    const slow = el('g', {}, svg);
    E.steps.forEach((s) => {
      const a = SLOW.byId[s.id];
      if (!a) return;
      const x = X(s.t0);
      const w = Math.max(1.5, X(s.t1) - X(s.t0));
      el('rect', { class: 'slow-step', x, y: top, width: w, height: bot - top }, slow);
      a.bands.forEach((k) => {
        const band = L[k];
        el('rect', { class: 'slow-band', x, y: band.y - 3, width: w, height: band.h + 6 }, slow);
      });
      // 顶上一个红角标：54 列里哪几列要看，不用逐个悬停找
    });
    // 8) 每个 step 一块透明热区：悬停给时长，点击下钻到单步视图
    const gz = el('g', { class: 'hits' }, svg);
    E.steps.forEach((s) => {
      const r = el('rect', {
        class: `hit${state.picked && s.id === state.step ? ' is-sel' : ''}`, x: X(s.t0), y: top,
        width: Math.max(1.5, X(s.t1) - X(s.t0)), height: bot - top, 'data-step': s.id,
      }, gz);
      const bs = RUN.runtime.byStep[s.id];
      const a = SLOW.byId[s.id];
      // 高于 P90 的步直接说清相对偏差，避免用户自己做减法
      const slowTag = a ? ` · 高于 P90 · 比中位步长 ${fmtMs(a.excess)}（分项偏高：${a.detail}）` : '';
      const ctx = { X, ROW };
      const tipFor = (dev) => {
        const l = s.lanes.find((x) => x.dev === (dev == null ? s.crit : dev));
        const head = `${s.id} · ${fmtMs(bs ? bs.period : s.t1 - s.t0)}${slowTag}`;
        const chain = `lane ${l.dev} bind ${fmtMs(l.b1 - l.b0)} → dev ${l.dev} runner_run ${fmtMs(l.r1 - l.r0)}`;
        const runParts = Number.isFinite(l.deviceExec)
          ? `runner_run（trace 窗口）＝执行 ${fmtMs(l.deviceExec)} + 等待 ${fmtMs(l.waitEstimate)}`
          : 'runner_run 为 trace 窗口，含可能的同步等待；本步未拆分显示';
        const tail = l.dev === s.crit
          ? `最晚进入 runner_run，这一步由它决定；其余 ${s.lanes.length - 1} 条在屏障等它`
          : `比关键 lane ${s.crit} 早 ${fmtMs(s.lanes.find((x) => x.dev === s.crit).r0 - l.r0)} 进入，这段时间在等屏障`;
        return `${head}
${chain}
${runParts}
${tail}（点击下钻）`;
      };
      r.addEventListener('mousemove', (e) => {
        if (state.pan) return;
        const p = svgPoint(e);
        const dev = p ? laneAt(p.y, L, s.lanes.length) : null;
        focusAt(s, dev, ctx);
        showTip(e, tipFor(dev));
      });
      r.addEventListener('mouseleave', () => { clearFocus(); hideTip(); });
      r.addEventListener('click', () => { selectStep(s.id, false); setView('step'); });
    });

    // 9) 冷启动与首次输出
    const pre = E.steps.filter((s) => s.kind !== 'decode');
    if (pre.length) {
      const xEnd = X(pre[pre.length - 1].t1);
      // 顶部导航已显示冷启动区间；这里仅保留阶段边界线，避免出现第二个不同口径的时长标签。
      vline(svg, xEnd, top, bot, 'g-start');
    }

    // 10) 时间轴（秒）
    const ga = el('g', { class: 'axis' }, svg);
    const yA = bot + 4;
    const tickStep = niceTickStep(rangeSpan, Math.max(4, Math.floor(W / 100)));
    drawMinorTicks(ga, range, tickStep, X, yA, [range.start, range.end]);
    for (let t = Math.ceil(range.start / tickStep) * tickStep; t <= range.end; t += tickStep) {
      vline(ga, X(t), yA - 6, yA, 'tick major');
      el('text', { class: 'tick-label', x: X(t), y: yA + 13, 'text-anchor': 'middle' }, ga).textContent = formatTick(t / 1000, tickStep / 1000);
    }
    el('text', { class: 'tick-label', x: x1 + 8, y: yA + 13, 'text-anchor': 'start' }, ga).textContent = 's';

  }

  // 泳道按当前范围渲染：全程 or 单步
  function drawLanes() {
    if (state.view === 'e2e') renderE2E();
    else renderScene();
    enforceSvgMinTextSize();
    normalizeLaneAxisTextSize();
    updateTimeControls();
    syncTimelineWindow();
    syncNav();
  }
  /* 悬停某一步时：只亮这一步的条，并把关键 lane 的 bind → runner_run 连起来，
   * 说明 Host 的 lane N 与 Device 的 dev N 是同一条 lane 的前后两段。 */
  /* 十字聚焦：纵向是一个 step，横向是一条 lane。
   * 纵向回答“这一步各卡在干什么”，横向回答“这条 lane 一路是什么样”，
   * 交叉处就是这一步这条 lane 的 bind → runner_run 两段。 */
  let focused = null;
  let focusKey = '';
  // 单步视图：悬停某一条，就把这条 lane 的两行连起来看
  function pairHover(node, l, ROW, X) {
    node.setAttribute('data-l', l.dev);
    node.addEventListener('mouseenter', () => {
      svg.querySelectorAll('.row-band').forEach((n) => n.remove());
      svg.querySelectorAll('.lane-pair, .lane-upstream').forEach((n) => n.classList.toggle('is-active', Number(n.dataset.l) === l.dev));
      svg.querySelectorAll(`[data-l="${l.dev}"]`).forEach((n) => n.classList.add('is-row'));
      [ROW.host(l.dev), ROW.device(l.dev)].forEach((y) => {
        el('rect', { class: 'row-band', x: PAD.l, y: y - ROW_H / 2, width: W, height: ROW_H }, svg);
      });
    });
    node.addEventListener('mouseleave', () => {
      svg.querySelectorAll('.lane-pair.is-active, .lane-upstream.is-active').forEach((n) => n.classList.remove('is-active'));
      svg.querySelectorAll('.is-row').forEach((n) => n.classList.remove('is-row'));
      svg.querySelectorAll('.row-band').forEach((n) => n.remove());
    });
    return node;
  }
  function svgPoint(e) {
    const m = svg.getScreenCTM();
    if (!m) return null;
    const p = svg.createSVGPoint();
    p.x = e.clientX; p.y = e.clientY;
    return p.matrixTransform(m.inverse());
  }
  function laneAt(y, L, nLane) {
    const hs = L.host.y + L.host.lanes0;
    const ds = L.device.y + L.device.lanes0;
    if (y >= hs && y < hs + nLane * ROW_H) return Math.floor((y - hs) / ROW_H);
    if (y >= ds && y < ds + nLane * ROW_H) return Math.floor((y - ds) / ROW_H);
    return null;
  }
  function clearFocus() {
    if (focused) focused.forEach((n) => n.classList.remove('is-hit', 'is-row'));
    svg.querySelectorAll('.row-band').forEach((n) => n.remove());
    svg.classList.remove('is-focus');
    focused = null; focusKey = '';
  }
  function focusAt(s, dev, ctx) {
    const key = `${s ? s.id : ''}|${dev == null ? '' : dev}`;
    if (key === focusKey) return;
    clearFocus();
    if (!s) return;
    focusKey = key;
    svg.classList.add('is-focus');
    const marked = [];
    svg.querySelectorAll('[data-s]').forEach((n) => {
      const inStep = n.getAttribute('data-s') === s.id;
      const inRow = dev != null && n.getAttribute('data-l') === String(dev);
      if (!inStep && !inRow) return;
      if (inStep) n.classList.add('is-hit');
      if (inRow) n.classList.add('is-row');
      marked.push(n);
    });
    focused = marked;
    // 横向：这条 lane 的两行整行压一层
    if (dev != null) {
      [ctx.ROW.host(dev), ctx.ROW.device(dev)].forEach((y) => {
        el('rect', { class: 'row-band', x: PAD.l, y: y - ROW_H / 2, width: W, height: ROW_H }, svg);
      });
    }
    // lane 配对关系通过对应行的高亮实线表达，不叠加红色虚线。
  }


  // 把导航条的时间线对齐到泳道的绘图区：直接量已经画出来的板，不再反推缩放
  function syncNav() {
    const nav = document.querySelector('.lane-nav');
    const plate = svg.querySelector('.plate-edge');
    if (!nav || !plate) return;
    const n = nav.getBoundingClientRect();
    const p = plate.getBoundingClientRect();
    if (!n.width || !p.width) return;
    nav.style.setProperty('--plot-l', `${Math.max(0, p.left - n.left)}px`);
    nav.style.setProperty('--plot-r', `${Math.max(0, n.right - p.right)}px`);
  }
  function syncLaneA11y() {
    svg.setAttribute('aria-label', state.view === 'e2e'
      ? `全程泳道：${RUN.e2e.steps.length} 个 step 的 Serving、Host、Device 三条带`
      : `单步泳道 ${state.step}：Serving、Host、Device 三条带与关键路径`);
  }
  function setView(view) {
    if (!VIEWS.includes(view) || view === state.view) return;
    state.view = view;
    state.timeRange = null;
    // 切到单步就等于选定了一步（默认是最接近 P50 的那一步）
    if (view === 'step') state.picked = true;
    drawLanes(); syncLaneA11y(); renderPathHead(); renderRhythm(); writeHash();
  }

  /* ------------------------------------------------------------ text
   * 页面按三步排：① 用户可感知的结果 → ② 沿关键路径拆每个 step → ③ 整体慢还是负载不均。
   * 左栏放 ① 与 ②（含泳道），右栏按同样的顺序分页签。 */

  /* ① 结果层：说明问题有多大，但定位不了根因 */
  function renderResult() {
    const q = RUN.request;
    $('#eyebrow').innerHTML = `${esc(RUN.title)} <span class="dim">· ${esc(RUN.sub)} · ${esc(RUN.deviceText)}</span>`;
    $('#eyebrow').title = `${RUN.request.id} · 数据来源 ${RUN.source}${RUN.capture ? `
固定负载 ${RUN.capture.workload}
采集 ${RUN.capture.skill}（profiler 口径，仅供定界）` : ''}`;
    $('#layerName').innerHTML = `${(q.e2eMs / 1000).toFixed(2)}<small>s</small>`;
    $('#layerName').setAttribute('aria-label', `端到端 ${fmtMs(q.e2eMs)}`);
    $('#layerMeta').innerHTML = `<span>E2E · ${q.outputs} 次输出 · ${q.stream ? '流式' : '非流式'} max_tokens ${q.maxTokens}</span>`;
    $('#reqKpis').innerHTML = `
      <div class="kpi" title="服务端第一次 process_step_output"><span>首次输出</span><b>${fmtMs(q.firstOutputMs)}</b></div>
      <div class="kpi" title="${esc(`${q.outputs - 1} 个输出间隔 · P50 ${fmtMs(q.gapP50Ms)} · P90 ${fmtMs(q.gapP90Ms)} · 最大 ${fmtMs(q.gapMaxMs)}`)}"><span>输出间隔 P50</span><b>${fmtMs(q.gapP50Ms)}</b></div>
      <div class="kpi" title="${esc(`输出间隔 P90 ${fmtMs(q.gapP90Ms)} · 最大 ${fmtMs(q.gapMaxMs)}`)}"><span>P90</span><b>${fmtMs(q.gapP90Ms)}</b></div>
      <div class="kpi" title="${esc(`Prefill ${fmtMs(RUN.coldStart.prefillMs)} + Draft init ${fmtMs(RUN.coldStart.draftMs)}`)}"><span>冷启动</span><b>${fmtMs(RUN.coldStart.prefillMs + RUN.coldStart.draftMs)}</b></div>`;
  }

  /* ② 关键路径：每个 decode step 的时间去哪了 */
  /* 左上角始终描述泳道正在展示的范围：全程给 decode 52 步的均值，单步给那一步。
   * 三行相加等于对应的周期（Host 含 Worker 下发）。 */
  function headScope() {
    const ps = RUN.perStep;
    if (state.view === 'e2e') {
      const R = RUN.runtime;
      const L = ps.layers;
      const decodeRows = Object.values(R.byStep).filter((row) => row.decode);
      const sum = (key) => decodeRows.reduce((total, row) => total + row[key], 0);
      return {
        title: `decode ${R.n} 步`,
        note: `平均 ${fmtMs(ps.meanMs)}（P50 ${fmtMs(ps.p50Ms)} · P90 ${fmtMs(ps.p90Ms)}）`,
        serving: L.serving.meanMs,
        host: L.host.meanMs,
        device: L.device.meanMs,
        servingTotal: r2(sum('serving')),
        hostTotal: r2(decodeRows.reduce((total, row) => total + row.worker + row.host, 0)),
        deviceTotal: r2(sum('device')),
        count: decodeRows.length,
        period: ps.meanMs,
        h2d: R.path.find((p) => p.key === 'h2d').stat.mean,
        launch: R.path.find((p) => p.key === 'launch').stat.mean,
        allocation: R.bindSplit.allocation.mean,
        transfer: R.bindSplit.transfer.mean,
        resident: R.resident.period.mean,
        savePct: R.resident.savePct,
        e2eSaveMs: R.resident.e2eSaveMs,
        deviceUpper: true,
      };
    }
    const b = RUN.runtime.byStep[state.step];
    return {
      title: state.step,
      note: b.decode
        ? `${fmtMs(b.period)}（52 步均值 ${fmtMs(ps.meanMs)}）`
        : `${fmtMs(b.period)}（${state.step === 'P' ? 'prefill' : '草稿初始化'} · 不计入均值）`,
      serving: b.serving,
      host: r2(b.worker + b.host),
      device: b.device,
      period: b.period,
      h2d: b.h2d,
      launch: b.launch,
      allocation: b.allocation,
      transfer: b.transfer,
      resident: b.resident,
    };
  }

  function renderPathHead() {
    const v = headScope();
    $('.scene-head').classList.toggle('is-e2e', state.view === 'e2e');
    // 下钻到 P90 筛查步时，第一眼先回答「我为什么在这一步」
    const a = state.view === 'step' ? SLOW.byId[state.step] : null;
    $('#eyebrow2').innerHTML = `<b>${esc(v.title)}</b> <span class="dim">· ${v.note} · 关键路径构成</span>${a
      ? `<span class="why" title="${esc(`分布筛查提示，不等同故障判定。周期超过本次 decode P90 ${fmtMs(RUN.perStep.p90Ms)}；分项偏差与 52 步的中位分项比较`)}">高于 P90 · 比中位步长 ${fmtMs(a.excess)} · 分项偏高：${esc(a.detail)}</span>`
      : ''}`;
    const max = Math.max(...LAYERS.map((l) => v[l]));
    $('#stepMix').innerHTML = LAYERS.map((l, i) => {
      const t = RUN.towers[l]; const share = (v[l] / v.period) * 100;
      const on = state.layer === l;
      return `
        <button class="mix-row${on ? ' is-on' : ''}" type="button" data-mix="${l}" aria-pressed="${on}" title="${esc(t.caption)}">
          <span class="mix-num">${i + 1}</span>
          <span class="mix-name">${NAME[l]}</span>
          <span class="mix-bar"><span class="mix-fill${t.upper ? ' upper' : ''}" style="width:${Math.max(0.6, (v[l] / max) * 100)}%"></span></span>
          <span class="mix-val">${t.upper ? '≤ ' : ''}${fmtMs(v[l])}</span>
          <span class="mix-pct">${share < 1 ? '≈ 0%' : `${t.upper ? '≤ ' : ''}${Math.round(share)}%`}</span>
        </button>${l === 'host' ? subRow(v) : ''}`;
    }).join('') + residentRow(v);
  }



  // Host 里几乎全是关键 lane 的 bind.args（分配 + H2D），#95 把它判为可消除的 P0
  // 余下的 lane 启动 · 其他单列一行，两行相加 = Host 层减去 Worker 下发
  function subRow(b) {
    const split = RUN.dist.bindSplitSource != null && b.allocation != null;
    return `
      <div class="bind-breakdown">
        <div class="mix-sub">
          <span class="mix-sub-name">└ bind.args</span>
          <span class="mix-val">${fmtMs(b.h2d)}</span>
          <span class="chip hot">可消除</span>
        </div>
        ${split ? `<div class="bind-components">
          <span>分配估计 <b>${fmtMs(b.allocation)}</b></span>
          <span>H2D 估计 <b>${fmtMs(b.transfer)}</b></span>
        </div>` : ''}
        ${b.launch != null ? `<div class="mix-sub is-cold" title="lane 起线程、bind 结束到进入 runner_run 之间的空档">
          <span class="mix-sub-name">└ lane 启动 · 其他</span>
          <span class="mix-val">${fmtMs(b.launch)}</span>
        </div>` : ''}
      </div>`;
  }
  // 常驻推算：去掉关键路径上的 bind.args，其余阶段不变
  function residentRow(b) {
    const cut = b.period - b.resident;
    const pct = b.savePct != null ? b.savePct : (cut / b.period) * 100;
    return `
      <div class="resident" title="逐步减去关键路径上的 bind.args，其余阶段不变后的推算步长">
        <span class="resident-k">权重 / Cache 常驻后</span>
        <span class="mix-bar"><span class="mix-fill ghost" style="width:${(b.resident / b.period) * 100}%"></span></span>
        <span class="mix-val">≈ ${fmtMs(b.resident)}</span>
        <span class="chip">推算</span>
      </div>
      <div class="resident-save">
        <span>每步 <b>−${Math.round(pct)}%</b></span>
        <span>${b.e2eSaveMs != null
    ? `本请求累计 <b>−${(b.e2eSaveMs / 1000).toFixed(2)} s</b>`
    : `这一步 <b>−${fmtMs(cut)}</b>`}</span>
      </div>`;
  }

  /* 主导航：54 个 step 的时间线，宽度 = 耗时，点任一段即在泳道里展开 */
  // 面包屑：全程是父层，某一步是子层；父层常驻且可点返回
  function crumb() {
    if (state.view === 'e2e') {
      return `<div class="request-scope-main">
          <b class="request-scope-title">请求全程</b>
          <div class="request-scope-metrics">
            <span class="request-scope-kpi"><i>总耗时</i><b>${fmtMs(RUN.e2e.durMs)}</b></span>
            <span class="request-scope-kpi"><i>步骤</i><b>${RUN.e2e.steps.length}</b></span>
            ${SLOW.n ? `<button class="request-p90" type="button" data-open-p90 title="P90 是本次样本的分布筛查线，不等于故障告警。筛查步合计比中位步长 ${esc(fmtMs(SLOW.excess))}，占 E2E ${SLOW.sharePct.toFixed(1)}%（点开逐步复核）"><i>Decode P90</i><b>${SLOW.n} / ${SLOW.of} 步</b><span>高于 ${fmtMs(RUN.perStep.p90Ms)}</span><em>›</em></button>` : ''}
          </div>
        </div>
        <span class="time-window-label" id="timeWindowLabel" hidden></span>`;
    }
    const t = RUN.request.timeline.find((x) => x.id === state.step);
    const f = RUN.flows[state.step];
    return `<button class="crumb-up" type="button" data-view="e2e">全程</button>
      <span class="crumb-sep" aria-hidden="true">›</span>
      <b class="crumb-cur">${esc(state.step)}</b>
      <span class="nav-meta">${fmtMs(t ? t.ms : f.periodMs)} · 关键 lane ${f.crit}</span>
      <span class="time-window-label" id="timeWindowLabel" hidden></span>`;
  }
  function renderRhythm() {
    const q = RUN.request;
    const segs = q.timeline.map((s) => {
      const a = SLOW.byId[s.id];
      const tip = `${s.id} · ${fmtMs(s.ms)}${a ? ` · 高于 P90 · 比中位步长 ${fmtMs(a.excess)}（分项偏高：${a.detail}）` : ''}`;
      return `<span class="tl-seg k-${s.kind}${a ? ' is-slow' : ''}${state.picked && s.id === state.step ? ' is-sel' : ''}" data-step="${esc(s.id)}" style="flex:${s.ms} 0 0" title="${esc(tip)}"></span>`;
    }).join('')
      + `<span class="tl-seg k-tail" style="flex:${q.tailMs} 0 0"></span>`;
    const first = (q.firstOutputMs / q.e2eMs) * 100;
    const coldStartMs = RUN.coldStart.prefillMs + RUN.coldStart.draftMs;
    const coldEnd = (coldStartMs / q.e2eMs) * 100;
    $('#request').innerHTML = `
      <nav class="crumb${state.view === 'e2e' ? ' request-scope' : ''}" id="navSel" aria-label="泳道范围">${crumb()}</nav>
      <div class="req-line" role="group" aria-label="请求时间线">
        <span class="tl-mark input" style="left:0%"><i></i>首次输入</span>
        <span class="tl-mark cold" style="left:${coldEnd}%"><i></i>冷启动 ${fmtMs(coldStartMs)}</span>
        <span class="tl-mark output" style="left:${first}%"><i></i>首次输出</span>
        <div class="tl">
          <span class="tl-cold-window" style="width:${coldEnd}%" aria-hidden="true"></span>
          ${segs}<span class="tl-window-overlay" id="tlWindowOverlay" aria-hidden="true" hidden></span>
        </div>
      </div>`;
    syncTimelineWindow();
  }
  function syncTimelineWindow() {
    const overlay = $('#tlWindowOverlay');
    const label = $('#timeWindowLabel');
    if (!overlay || !label) return;
    const full = fullTimeRange();
    const range = visibleTimeRange();
    const zoomed = state.view === 'e2e' && (range.end - range.start) < (full.end - full.start) * 0.995;
    if (!zoomed) {
      overlay.hidden = true;
      label.hidden = true;
      document.querySelectorAll('.tl-seg[data-step]').forEach((node) => node.classList.remove('is-in-window'));
      return;
    }
    const duration = RUN.e2e.durMs;
    const steps = RUN.e2e.steps.filter((step) => step.t1 > range.start && step.t0 < range.end);
    const stepIdsInWindow = new Set(steps.map((step) => step.id));
    const left = Math.max(0, Math.min(100, (range.start / duration) * 100));
    const right = Math.max(left, Math.min(100, (range.end / duration) * 100));
    overlay.style.left = `${left}%`;
    overlay.style.width = `${right - left}%`;
    overlay.hidden = false;
    document.querySelectorAll('.tl-seg[data-step]').forEach((node) => node.classList.toggle('is-in-window', stepIdsInWindow.has(node.dataset.step)));
    const first = steps[0]?.id;
    const last = steps[steps.length - 1]?.id;
    label.textContent = steps.length ? `当前窗口：${first}${first === last ? '' : `–${last}`} · ${steps.length} 步` : '当前窗口：无 step';
    label.hidden = false;
  }
  /* ------------------------------------------------------------ 右栏分析面板
   * 每个面板都是「一句结论 → 一份证据 → 折叠的口径」三段。
   * 分位数压成一个区间列，参照口径一律折叠，首屏只留下判断需要的那几个数。 */

  // 折叠块：参照信息、排除项、盲区都放这里，不占首屏
  const fold = (title, meta, body) => `
    <details class="fold">
      <summary><span>${esc(title)}</span><i>${esc(meta)}</i></summary>
      <div class="fold-body">${body}</div>
    </details>`;

  // 区间：均值已经给了量级，分位只需要一个范围
  function fmtRange(a, b) {
    const [va, ua] = fmtMs(a).split(' ');
    const [vb, ub] = fmtMs(b).split(' ');
    return ua === ub ? `${va}–${vb} ${ub}` : `${va} ${ua}–${vb} ${ub}`;
  }

  /* 账表：均值 + P50–P90 两列就够定界；P50 / P90 / 最大的原值挂在行的 title 上 */
  const ledgerRow = (r, cls) => `
    <tr class="${cls || ''}${r.hot ? ' hot' : ''}" title="${esc(`${r.label} · 均值 ${fmtMs(r.stat.mean)} · P50 ${fmtMs(r.stat.p50)} · P90 ${fmtMs(r.stat.p90)} · 最大 ${fmtMs(r.stat.max)}${r.note ? ` · ${r.note}` : ''}`)}">
      <th scope="row"><b>${esc(r.label)}</b>${r.sub ? `<i>${esc(r.sub)}</i>` : ''}</th>
      <td>${r.upper ? '≤ ' : ''}${fmtMs(r.stat.mean)}</td>
      <td class="spread">${fmtRange(r.stat.p50, r.stat.p90)}</td>
    </tr>`;
  const ledgerHead = (title) => `<thead><tr><th scope="col">${title}</th><th scope="col">均值</th><th scope="col">P50–P90</th></tr></thead>`;

  /* ② 关键路径：只给还需要往下走的结论 + 可相加的五段；观察窗口折叠
   * 查完没问题的那几层不展开——它们不是用户要处理的事，只留一行作为查过的记录。 */
  // 结论卡只说用户感知得到的量；实现口径（bind.args / H2D 等）挂在卡片的 title 上
  const Q_CHIP = { hit: '命中' };
  function renderPathPanel() {
    const R = RUN.runtime;
    const live = RUN.questions.filter((q) => q.state === 'hit');
    const items = live.map((q) => `
        <li class="vd is-${q.state}"${q.tech ? ` title="${esc(q.tech)}"` : ''}>
          <span class="vd-k">${Q_CHIP[q.state]}</span>
          <div class="vd-b">
            <b>${esc(q.topic)}${q.estimate ? '<em class="est-tag" title="含模型推算的拆分，不是实测值">估计</em>' : ''}</b>
            <p>${esc(q.a)}</p>
            ${q.next ? (q.go
    ? `<button class="vd-go" type="button" data-go-tab="${esc(q.go.tab)}"${q.go.lane != null ? ` data-go-lane="${q.go.lane}"` : ''}>${esc(q.next)}</button>`
    : `<span class="vd-go is-static">${esc(q.next)}</span>`) : ''}
          </div>
        </li>`).join('');
    $('#panel-path').innerHTML = `
      <section class="bd-sec">
        <h3>定界结论</h3>
        <ul class="verdicts">${items}</ul>
      </section>
      <section class="bd-sec">
        <h3>关键路径<span>decode ${R.n} 步 · 五段相加 = 步长</span></h3>
        <table class="ledger">
          ${ledgerHead('阶段')}
          <tbody>${R.path.map((r) => ledgerRow(r.key === 'serving'
    ? { ...r, note: `调度本身 ${fmtMs(RUN.perStep.schedMs)}/步，其余已被流水覆盖` }
    : r)).join('')}</tbody>
          <tfoot>${ledgerRow({ label: 'step 周期', sub: R.range, stat: R.period }, 'total')}</tfoot>
        </table>
        ${fold('bind.args 内部拆分', '设备内存分配 + H2D 搬运', `
          <table class="ledger dim">
            ${ledgerHead('组成')}
            <tbody>
              ${ledgerRow({ label: '设备内存分配估计', stat: R.bindSplit.allocation })}
              ${ledgerRow({ label: 'H2D 搬运估计', sub: 'Host → Device', stat: R.bindSplit.transfer })}
            </tbody>
            <tfoot>${ledgerRow({ label: 'bind.args 总计', stat: R.path.find((r) => r.key === 'h2d').stat }, 'total')}</tfoot>
          </table>`)}
        ${fold('观察窗口', `${R.windows.length} 项 · 与上表重叠，不能相加`, `
          <table class="ledger dim">
            ${ledgerHead('指标')}
            <tbody>${R.windows.map((r) => ledgerRow(r)).join('')}</tbody>
          </table>`)}
      </section>`;
  }

  /* ③ 整体慢 还是 负载不均：命中哪一类 + 16 条 lane 的起点分散 */
  function renderDistPanel() {
    if (state.lane != null) { renderLanePanel(state.lane); return; }
    const d = RUN.dist;
    const total = RUN.runtime.period.mean;
    const pct = (ms) => Math.max(0.5, Math.min(100, (ms / total) * 100));
    const critLead = d.lanes.find((l) => l.dev === d.critLane).lead;
    // runner_run 拆成「执行」与「等屏障」——执行几乎等长，差别全在起点
    const split = d.deviceSplitSource != null && d.lanes[0].deviceExec != null;
    // 每条 lane 当了多少步的关键 lane：偏斜是持续的还是偶发的，一列就能看出来
    const hasCrit = d.lanes.some((l) => l.critCount != null);
    const rows = d.lanes.map((l) => {
      const crit = l.dev === d.critLane;
      const base = split
        ? `lane ${l.dev} · bind ${fmtMs(l.bind)} · 进入 runner_run 前 ${fmtMs(l.lead)} · runner_run ${fmtMs(l.rr)}＝执行 ${fmtMs(l.deviceExec)}＋等待 ${fmtMs(l.waitEstimate)}`
        : `lane ${l.dev} · bind ${fmtMs(l.bind)} · 进入 runner_run 前 ${fmtMs(l.lead)} · runner_run ${fmtMs(l.rr)}（含等待 ≥ ${fmtMs(l.wait)}）`;
      const tip = `${base}${l.critCount != null ? ` · ${d.n} 步里有 ${l.critCount} 步最晚进入` : ''} — 点击查看这条 lane`;
      return `
        <div class="lane-row is-link${crit ? ' is-crit' : ''}" role="button" tabindex="0" data-lane="${l.dev}" title="${esc(tip)}">
          <span class="ln">lane ${l.dev}</span>
          <span class="track">
            <span class="seg host" style="width:${pct(l.lead)}%"></span>
            ${split
    ? `<span class="seg exec" style="width:${pct(l.deviceExec)}%"></span><span class="seg hold" style="width:${pct(l.waitEstimate)}%"></span>`
    : `<span class="seg rr" style="width:${pct(l.rr)}%"></span>`}
          </span>
          <span class="lv">${fmtMs(l.bind)}</span>
          ${hasCrit ? `<span class="lc${l.critCount ? ' is-on' : ''}">${l.critCount ? `${l.critCount}/${d.n}` : '—'}</span>` : ''}
        </div>`;
    }).join('');
    $('#panel-dist').innerHTML = `
      <section class="bd-sec">
        <h3>不均的类型<span>比较各 lane 的起点与终点</span></h3>
        <ul class="verdicts">${d.patterns.filter((p) => p.hit).map((p) => `
          <li class="vd is-hit">
            <span class="vd-k">命中</span>
            <div class="vd-b">
              <b>${esc(p.title)}</b>
              <p>${esc(d.verdict)}</p>
              <span class="vd-go is-static">往下查：${esc(p.next)}</span>
            </div>
          </li>`).join('')}
        </ul>
        ${(() => { const c = d.patterns.filter((p) => !p.hit); return c.length ? `<p class="vd-clear" title="${esc(c.map((p) => `${p.title}：${p.next}`).join('\n'))}">已排除 ${c.map((p) => esc(p.title)).join(' · ')}</p>` : ''; })()}
      </section>
      <section class="bd-sec">
        <h3>${d.nLane} 条 lane<span>${split ? '实色 = 进入前 · 斜纹 = 执行 · 空档 = 等屏障' : '实色 = 进入 runner_run 前 · 斜纹 = runner_run'}</span></h3>
        <p class="lane-stat">关键 lane <b>${d.critLane}</b> · <b>${d.critCount}/${d.n}</b> 步最晚进入 · bind 是中位 lane 的 <b>${d.skew}×</b></p>
        <div class="lane-chart${hasCrit ? ' with-crit' : ''}" style="--crit:${pct(critLead)}%">
          ${rows}
          <span class="lane-guide" style="left:calc(50px + (100% - ${hasCrit ? 152 : 108}px) * ${(critLead / total).toFixed(4)})"></span>
        </div>
        <p class="lane-note">${split
    ? `红线 = 最晚进入 runner_run 的时刻。斜纹段几乎等长（${fmtMs(Math.min(...d.lanes.map((l) => l.deviceExec)))}–${fmtMs(Math.max(...d.lanes.map((l) => l.deviceExec)))}），差别全在它前面；右列是该 lane 的 bind 均值${hasCrit ? '，末列是它当关键 lane 的步数' : ''}。点一条看它的专项。`
    : `红线 = 最晚进入 runner_run 的时刻，线左侧的斜纹都在等屏障；右列是该 lane 的 bind 均值${hasCrit ? '，末列是它当关键 lane 的步数' : ''}。点一条看它的专项。`}</p>
        ${fold('口径边界', split ? '执行 / 等待拆分' : '别把上界当执行时间', `
          <p class="caveat">${esc(d.caveat)}</p>
          <table class="ledger dim">
            ${ledgerHead('指标')}
            <tbody>
              ${ledgerRow({ label: '起点分散', sub: '最晚 − 最早进入 runner_run', stat: d.startSpread, hot: true })}
              ${ledgerRow({ label: '终点分散', sub: '同步屏障拉齐', stat: d.endSpread })}
              ${ledgerRow({ label: '屏障等待合计', sub: `${d.nLane - 1} 条 lane / 步`, stat: d.waitTotal })}
            </tbody>
          </table>`)}
      </section>`;
  }

  /* lane 专项：下钻一层回答“为什么是这一条”。
   * 三段——慢在哪一段、偏斜是什么形状、各阶段是不是同一条；排查记录与盲区折叠。 */
  function renderLanePanel(dev) {
    const d = RUN.dist;
    const F = RUN.laneFocus;
    const me = d.lanes.find((l) => l.dev === dev);
    const sorted = d.lanes.slice().sort((a, b) => b.bind - a.bind);
    const rank = sorted.findIndex((l) => l.dev === dev) + 1;
    const max = sorted[0].bind;
    const rrRank = d.lanes.slice().sort((a, b) => a.rr - b.rr).findIndex((l) => l.dev === dev) + 1;
    const execs = d.lanes.map((l) => l.deviceExec).filter((v) => v != null).sort((a, b) => a - b);
    const medExec = execs.length ? execs[Math.floor(execs.length / 2)] : null;

    // 单点还是梯度：前几名加这条 lane 就能看出来，其余压成一行区间
    const shown = sorted.slice(0, 5);
    if (!shown.some((l) => l.dev === dev)) shown.push(me);
    const rest = sorted.filter((l) => !shown.includes(l));
    const bars = shown.map((l) => `
      <div class="lane-row${l.dev === dev ? ' is-crit' : ''}">
        <span class="ln">lane ${l.dev}</span>
        <span class="track"><span class="seg host" style="width:${(l.bind / max) * 100}%"></span></span>
        <span class="lv">${fmtMs(l.bind)}</span>
      </div>`).join('')
      + (rest.length ? `
      <div class="lane-row is-rest">
        <span class="ln" title="其余 ${rest.length} 条 lane">其余 ${rest.length}</span>
        <span class="track"><span class="seg host ghost" style="width:${(rest[0].bind / max) * 100}%"></span></span>
        <span class="lv">${fmtRange(rest[rest.length - 1].bind, rest[0].bind)}</span>
      </div>` : '');

    // 54 步趋势：这条 lane 与当步中位 lane 的 bind
    const series = F.byLane[dev];
    const top = Math.max(...series, ...F.med);
    const W2 = 300; const H2 = 44;
    const px = (i) => (i / (series.length - 1)) * W2;
    const py = (v) => H2 - (v / top) * H2;
    const line = (xs) => xs.map((v, i) => `${px(i).toFixed(1)},${py(v).toFixed(1)}`).join(' ');
    const ratios = F.segments[dev].map((x) => x.ratio);
    const segs = F.segments[dev].map((x) => `
      <div class="seg-row"><span>${esc(x.range)}</span><b>${x.ratio.toFixed(2)}×</b></div>`).join('');

    const stages = F.stages.map((s) => {
      const mix = s.critMix
        ? Object.entries(s.critMix).map(([k, v]) => `lane ${k} × ${v}`).join('、')
        : `lane ${s.crit}`;
      const hit = s.critMix ? Object.keys(s.critMix).includes(String(dev)) : s.crit === dev;
      return `<div class="stage-row${hit ? ' is-hit' : ''}"><span>${esc(s.name)}</span><b>${esc(mix)}</b></div>`;
    }).join('');

    const notes = (title, xs, ruled) => `
      <div class="note-grp"><h4>${esc(title)}</h4>
        <ul class="checks${ruled ? ' ruled' : ''}">${xs.map((x) => `<li><b>${esc(x.label)}</b>${esc(x.text)}</li>`).join('')}</ul>
      </div>`;

    $('#panel-dist').innerHTML = `
      <nav class="crumb bd-crumb">
        <button class="crumb-up" type="button" data-lane="">设备分布</button>
        <span class="crumb-sep" aria-hidden="true">›</span>
        <b class="crumb-cur">lane ${dev}</b>
        <span class="nav-meta">bind 第 ${rank} / ${d.nLane}${me.critCount != null ? ` · 关键 ${me.critCount}/${d.n} 步` : ''}</span>
      </nav>
      <section class="bd-sec">
        <h3>慢在哪一段<span>host bind 对 device 执行</span></h3>
        <table class="ledger">
          <tbody>
            <tr class="hot"><th scope="row"><b>bind.args</b><i>host 侧 · 分配 + H2D</i></th>
              <td>${fmtMs(me.bind)}</td><td class="spread">中位的 ${r2(me.bind / d.medBind)}×</td></tr>
            ${me.allocation != null ? `
            <tr class="sub"><th scope="row"><b>└ 设备内存分配</b><i>估计值</i></th>
              <td>${fmtMs(me.allocation)}</td><td class="spread">—</td></tr>
            <tr class="sub"><th scope="row"><b>└ H2D 搬运</b><i>估计值 · Host → Device</i></th>
              <td>${fmtMs(me.transfer)}</td><td class="spread">—</td></tr>` : ''}
            <tr><th scope="row"><b>runner_run</b><i>device 侧 · 含等屏障</i></th>
              <td>${fmtMs(me.rr)}</td><td class="spread">第 ${rrRank} / ${d.nLane} 短</td></tr>
            ${me.deviceExec != null ? `
            <tr class="sub"><th scope="row"><b>└ 设备执行</b></th>
              <td>${fmtMs(me.deviceExec)}</td><td class="spread">${d.nLane} 条中位 ${fmtMs(medExec)}</td></tr>
            <tr class="sub"><th scope="row"><b>└ 等屏障</b><i>runner_run − 执行</i></th>
              <td>${fmtMs(me.waitEstimate)}</td><td class="spread">—</td></tr>` : `
            <tr><th scope="row"><b>屏障等待</b><i>早到白等的时间</i></th>
              <td>${fmtMs(me.wait)}</td><td class="spread">—</td></tr>`}
          </tbody>
        </table>
        <p class="lane-note">${dev === d.critLane
    ? `runner_run 是 ${d.nLane} 条里最短的${medExec != null ? `，执行 ${fmtMs(me.deviceExec)} 与中位的 ${fmtMs(medExec)} 也基本一致` : ''}——设备执行没有变慢的证据，慢的在它前面 host 侧的 bind。`
    : `它不是关键 lane，runner_run 偏长的部分${medExec != null ? `有 ${fmtMs(me.waitEstimate)} 是在等屏障` : '多半是在屏障等待'}。`}</p>
      </section>
      <section class="bd-sec">
        <h3>偏斜的形状<span>是单点、是梯度，还是逐渐劣化</span></h3>
        <div class="lane-chart">${bars}</div>
        <svg class="spark" viewBox="-2 -6 ${W2 + 4} ${H2 + 12}" role="img" aria-label="bind 随 step 的变化">
          <polyline class="sp-med" points="${line(F.med)}" />
          <polyline class="sp-me" points="${line(series)}" />
        </svg>
        <div class="spark-axis">
          <span>${esc(F.ids[0])}</span>
          <em>实线 lane ${dev} · 虚线当步中位 · 峰值 ${fmtMs(top)}</em>
          <span>${esc(F.ids[F.ids.length - 1])}</span>
        </div>
        <div class="seg-list">${segs}</div>
        <p class="lane-note">单点：比第二名还高 ${fmtMs(sorted[0].bind - sorted[1].bind)}，其余 ${d.nLane - 1} 条连续分布。四段倍数 ${Math.min(...ratios).toFixed(2)}–${Math.max(...ratios).toFixed(2)}×，是结构性的而不是逐渐劣化。</p>
      </section>
      <section class="bd-sec">
        <h3>各阶段是不是同一条<span>谁最晚进入 runner_run</span></h3>
        <div class="stage-list">${stages}</div>
        <p class="lane-note">Prefill 的关键 lane 不是它——偏斜跟 decode 这个 callable 绑定的参数有关，不是这条线程或这张卡永远最慢。</p>
      </section>
      ${renderOpsSection(dev)}
      ${fold('排查记录与盲区', `已排除 ${F.excluded.length} · 待补采 ${F.missing.length} · 待实验 ${F.experiments.length}`,
    notes('已排除', F.excluded, true) + notes('这份 trace 答不了', F.missing) + notes('需要做实验', F.experiments))}`;
  }


  /* ③.5 算子级下钻：runner_run 这一个窗口里到底跑了哪些算子
   * 数据来自另一次采集（GBS256 压测包里的 Ascend profiler trace，只采了 1 张卡），
   * 所以这里是「一张卡上一个 decode step 长什么样」的实例，不是本请求这条 lane 的实测。
   * 页面上一律按 trace 原值显示，不缩放到 runner_run 窗口，也不与左边的估计值混算。 */
  function renderOpsSection(dev) {
    const O = window.TRIAGE_OPS;
    if (!O || !O.steps || !O.steps.length) return '';
    const s = O.steps.find((x) => x.id === state.opsStep) || O.steps[1] || O.steps[0];
    const G = Object.fromEntries(O.groups.map((g, i) => [g.key, { ...g, i }]));
    const busy = s.wait + s.compute;
    const pct = (ms) => (ms / s.ms) * 100;
    // 构成条：等待 / 计算 / 流上空档，三段相加 = 这一步的长度
    const mix = `
      <div class="ops-mix" role="img" aria-label="这一步的设备时间构成">
        <span class="om wait" style="width:${pct(s.wait)}%" title="${esc(`等待同步 ${fmtMs(s.wait)}`)}"></span>
        <span class="om calc" style="width:${pct(s.compute)}%" title="${esc(`算子执行 ${fmtMs(s.compute)}`)}"></span>
        <span class="om idle" style="width:${pct(s.idle)}%" title="${esc(`这条流上没有任务 ${fmtMs(s.idle)}`)}"></span>
      </div>
      <div class="ops-legend">
        <span><i class="om wait"></i>等待同步 <b>${fmtMs(s.wait)}</b></span>
        <span><i class="om calc"></i>算子执行 <b>${fmtMs(s.compute)}</b></span>
        <span><i class="om idle"></i>流上空档 <b>${fmtMs(s.idle)}</b></span>
      </div>`;
    // 任务带：这一步的每个设备任务按时间铺开，颜色是语义分组
    const strip = `
      <div class="ops-strip" role="img" aria-label="${esc(`${s.nTask} 个设备任务按时间铺开`)}">
        ${s.tasks.map(([t, d, gi]) => {
    const g = O.groups[gi];
    return `<i class="ot g-${esc(g.key)}" style="left:${(t / s.ms) * 100}%;width:${Math.max(0.12, (d / s.ms) * 100)}%"></i>`;
  }).join('')}
      </div>
      <div class="ops-axis"><span>0</span><em>${s.nTask} 个任务 · ${s.nOp} 种算子</em><span>${fmtMs(s.ms)}</span></div>`;
    const groupRows = s.groups.map((g) => `
      <div class="ops-grow">
        <span class="ogk"><i class="ot g-${esc(g.key)}"></i>${esc(G[g.key] ? G[g.key].label : g.key)}</span>
        <span class="track"><span class="seg host" style="width:${Math.max(0.6, (g.ms / s.groups[0].ms) * 100)}%"></span></span>
        <span class="ogn">${g.n}</span>
        <span class="lv">${fmtMs(g.ms)}</span>
      </div>`).join('');
    const opRows = s.ops.filter((o) => G[o.group] && o.group !== 'wait' && o.group !== 'marker').slice(0, 10).map((o) => `
      <tr title="${esc(`${o.name} · ${o.type} · ${o.n} 次 · 合计 ${fmtMs(o.ms)}`)}">
        <th scope="row"><b>${esc(o.name.length > 34 ? `${o.name.slice(0, 33)}…` : o.name)}</b><i>${esc(o.type)}</i></th>
        <td>${o.n}</td>
        <td class="spread">${fmtMs(o.ms)}</td>
      </tr>`).join('');
    const picks = O.steps.map((x) => `
      <button type="button" class="ops-pick${x.id === s.id ? ' is-on' : ''}" data-ops-step="${esc(x.id)}"
        title="${esc(`${x.label}的一步 ${x.id} · ${fmtMs(x.ms)}`)}">${esc(x.label)} ${fmtMs(x.ms)}</button>`).join('');
    return `
      <section class="bd-sec ops-sec">
        <h3>这一步在设备上跑了什么<span>算子级 · 另一次采集</span></h3>
        <a class="ops-open" href="./device.html#lane=${dev}&step=${esc(s.id)}">打开这张卡的详细分析：${O.nStep} 步 · 多流甘特 · 算子排行 ↗</a>
        <p class="ops-prov">
          <b>不是本请求的数据</b>：来自 ${esc(O.source.run)} 的 Ascend profiler trace，${esc(O.source.rank)}；
          与 dev ${dev} 的 <code>runner_run</code> 既不同次采集也不同负载，时间按 trace 原值显示，<b>没有缩放</b>到这条 lane 的窗口。
          它回答的是“一个 decode step 在一张卡上长什么样”，不能当作 dev ${dev} 的实测拆分。
        </p>
        <div class="ops-picks">${picks}<span class="ops-meta">${O.nStep} 步中</span></div>
        ${mix}
        ${strip}
        <p class="lane-note">这一步 ${fmtMs(s.ms)} 里，算子真正在算的只有 ${fmtMs(s.compute)}（${Math.round((s.compute / s.ms) * 100)}%），
          其余是等同步和流上空档——${O.nStep} 步平均等待占 ${O.waitShare}%。这正是 GBS1 的 STRACE 量不到、只能靠估计拆分的那一段。</p>
        <div class="ops-groups">${groupRows}</div>
        <table class="ledger ops-top">
          <thead><tr><th scope="col">算子（去掉等待，前 10）</th><th scope="col">次数</th><th scope="col">合计</th></tr></thead>
          <tbody>${opRows}</tbody>
        </table>
        ${fold('这份数据的口径', O.source.stream, `
          <table class="ledger dim">
            <tbody>
              <tr><th scope="row"><b>来源</b></th><td colspan="2">${esc(O.source.zip)} › ${esc(O.source.member)}</td></tr>
              <tr><th scope="row"><b>同一次采集的是</b></th><td colspan="2">${esc(O.source.sameRunAs)}；<b>不是</b> ${esc(O.source.notSameRunAs)}</td></tr>
              <tr><th scope="row"><b>卡数</b></th><td colspan="2">${esc(O.source.rank)}，所以没有 16 张卡的对比</td></tr>
              <tr><th scope="row"><b>画的是哪条流</b></th><td colspan="2">${esc(O.source.stream)}；${esc(O.source.streamNote)}</td></tr>
              <tr><th scope="row"><b>时间</b></th><td colspan="2">${esc(O.source.clock)}；${esc(O.source.profiler)}</td></tr>
              <tr><th scope="row"><b>步的边界</b></th><td colspan="2">设备侧 <code>MODEL_EXECUTE</code> 标记之间，共 ${O.nStep} 步（P50 ${fmtMs(O.period.p50)} · P90 ${fmtMs(O.period.p90)}）</td></tr>
            </tbody>
          </table>`)}
      </section>`;
  }

  /* ④ 优化队列与验证目标 */
  function renderPlanPanel() {
    $('#panel-plan').innerHTML = `
      <section class="bd-sec">
        <h3>优化队列<span>当前值取自本次基线</span></h3>
        <ol class="plan">${RUN.plan.map((p) => `
          <li class="plan-item p-${p.pri.toLowerCase()}">
            <span class="pri">${p.pri}</span>
            <div class="plan-body">
              <div class="plan-top"><b>${esc(p.title)}</b><span class="plan-now">${esc(p.now)}</span></div>
              <p>${esc(p.action)}</p>
              <span class="plan-expect">${esc(p.expect)}</span>
            </div>
          </li>`).join('')}
        </ol>
      </section>
      <section class="bd-sec">
        <h3>验证目标<span>同设备、同 prompt 复测</span></h3>
        <ul class="checks">${RUN.checks.map((c) => `<li><b>${esc(c.label)}</b>${esc(c.text)}</li>`).join('')}</ul>
        ${RUN.capture ? `<p class="capture">固定负载 <code>${esc(RUN.capture.workload)}</code><br>采集方式 <code>${esc(RUN.capture.skill)}</code>（pypto-serving/.agents/skills）</p>
        <p class="caveat">${esc(RUN.capture.note)}</p>` : ''}
        <a class="ref" href="${esc(RUN.reference.url)}" target="_blank" rel="noopener">口径参照 ${esc(RUN.reference.label)} ↗</a>
      </section>`;
  }

  function renderTabs() {
    document.querySelectorAll('.side-tab').forEach((b) => {
      const on = b.dataset.tab === state.tab;
      b.classList.toggle('is-on', on); b.setAttribute('aria-selected', on); b.tabIndex = on ? 0 : -1;
    });
    TABS.forEach((t) => { $(`#panel-${t}`).hidden = state.tab !== t; });
  }
  function selectLane(dev) {
    const next = dev === '' || dev == null ? null : Number(dev);
    if (next === state.lane) return;
    state.lane = next;
    renderDistPanel(); writeHash();
    $('#panel-dist').scrollTop = 0;
  }
  function selectTab(tab) {
    if (!TABS.includes(tab) || tab === state.tab) return;
    state.tab = tab; renderTabs(); writeHash();
    if (tab === 'steps') scrollToStep(false);
  }

  /* ------------------------------------------------------------ floors */
  /* 54 行里每一行的两个指标名都一样，所以名字只在表头写一次，行里只留会变的数。 */
  function renderFloors() {
    const t = tower();
    // 口径跟在标题后面；标题已经写了「每个 step」，所以去掉 caption 里重复的那一截
    const cap = t.caption ? t.caption.replace(new RegExp(`^每个\\s*${RUN.floorUnit}\\s*[：:中]?\\s*`), '') : '';
    $('#floorsTitle').innerHTML = `${NAME[state.layer]} · 每个 ${esc(RUN.floorUnit)}${cap ? `<em>${esc(cap)}</em>` : ''}`;
    // 表头取各步共有的那部分：关键 lane 逐步会变，所以把变的那一截写成 N（具体值在行的 title 上）
    const common = (i, key) => {
      const xs = [...new Set(t.floors.map((f) => f.cards[i] && f.cards[i][key]).filter(Boolean))];
      if (xs.length < 2) return xs[0] || '';
      const parts = xs.map((x) => x.split(' '));
      if (!parts.every((p) => p.length === parts[0].length)) return xs[0];
      return parts[0].map((tok, k) => (parts.every((p) => p[k] === tok) ? tok : (/^\d/.test(tok) ? 'N' : '…'))).join(' ');
    };
    const cols = t.floors[0] ? t.floors[0].cards : [];
    $('#floorHead').innerHTML = `<span class="fh-id">step</span>${cols.map((c, i) => `
      <span class="fh-col"><b>${esc(common(i, 'title'))}</b><i>${esc(common(i, 'sub'))}</i></span>`).join('')}`;
    $('#floorList').innerHTML = t.floors.slice().reverse().map((f) => {
      const a = SLOW.byId[f.id];
      return `
      <div class="floor${a ? ' is-slow' : ''}${state.picked && f.id === state.step ? ' is-sel' : ''}" data-floor="${esc(f.id)}">
        <span class="floor-id"${a ? ` title="${esc(`高于 P90 · 比中位步长 ${fmtMs(a.excess)} · 分项偏高：${a.detail}`)}"` : ''}>${esc(f.id)}</span>
        ${f.cards.map((c) => `
          <button class="cell ${c.tone}" type="button" data-floor="${esc(f.id)}" title="${esc(`${f.id} · ${c.title} · ${c.sub}`)}">${esc(c.text || fmtMs(c.ms))}</button>`).join('')}
        ${strip(t, f)}
      </div>`;
    }).join('');
  }

  // 同一层内与中位数相比越慢越红；只在 heat 为 true 的楼上使用。
  function heatAlpha(f, i) {
    const vals = f.devs.slice().sort((a, b) => a - b);
    const med = vals[Math.floor(vals.length / 2)] || 1;
    return Math.max(0, Math.min(1, (f.devs[i] / med - 1.25) / 0.9)) * 0.85;
  }
  function strip(t, f) {
    if (!t.devices || !f.devs) return '';
    const cells = f.devs.map((v, i) => {
      const a = t.devices.heat ? heatAlpha(f, i) : 0;
      const crit = t.devices.heat && f.critDev === i;
      const tip = `${f.id} · ${t.devices.labels[i]} · ${t.devices.metric || ''} ${fmtMs(v)}${crit ? ' · 本步最晚进入' : ''}`;
      return `<span class="sc${crit ? ' crit' : ''}" data-floor="${esc(f.id)}" data-dev="${i}" style="--a:${a.toFixed(3)}" title="${esc(tip)}"></span>`;
    }).join('');
    return `<div class="strip" style="--n:${f.devs.length}" aria-label="每张卡的耗时">${cells}</div>`;
  }

  /* ------------------------------------------------------------ actions */
  function render() {
    renderResult(); renderPathHead(); renderRhythm(); syncLaneA11y(); renderPathPanel(); renderDistPanel(); renderPlanPanel(); renderFloors(); renderTabs();
    drawLanes();
    syncNav();
    $('#analysisLink').href = `./analysis.html#layer=${state.layer}${state.picked ? `&step=${encodeURIComponent(analysisStep(state.step))}` : ''}`;
    writeHash();
  }
  // 详细分析页的 step id 用的是 Prefill / Draft init
  const analysisStep = (id) => (id === 'P' ? 'Prefill' : id === 'DI' ? 'Draft init' : id);

  function selectLayer(layer) {
    if (state.layer === layer && state.tab === 'steps') return;
    state.layer = layer;
    state.tab = 'steps';
    render();
    scrollToStep(false);
  }
  function selectStep(id, scroll) {
    if (!stepIds.includes(id) || (id === state.step && state.picked)) return;
    state.step = id;
    state.timeRange = null;
    state.picked = true;
    drawLanes();
    syncLaneA11y();
    document.querySelectorAll('.floor').forEach((n) => n.classList.toggle('is-sel', n.dataset.floor === id));
    document.querySelectorAll('.tl-seg[data-step]').forEach((n) => n.classList.toggle('is-sel', n.dataset.step === id));
    syncLaneA11y();
    $('#navSel').innerHTML = crumb();
    renderPathHead();
    $('#analysisLink').href = `./analysis.html#layer=${state.layer}&step=${encodeURIComponent(analysisStep(id))}`;
    if (scroll) scrollToStep(true);
    writeHash();
  }
  function scrollToStep(smooth) {
    if (!state.picked) return;
    const row = document.querySelector(`.floor[data-floor="${CSS.escape(state.step)}"]`);
    const list = $('#floorList');
    if (row && list.scrollHeight > list.clientHeight) {
      const top = row.offsetTop - list.offsetTop - (list.clientHeight - row.offsetHeight) / 2;
      list.scrollTo({ top: Math.max(0, top), behavior: smooth && !document.hidden ? 'smooth' : 'auto' });
    }
  }
  // 从全程图右上角的计数跳过来：把 P90 筛查清单滚进视野并闪一下，省得自己找
  function focusSlowList() {
    const li = $('#slowList');
    if (!li) return;
    li.scrollIntoView({ block: 'nearest', behavior: document.hidden ? 'auto' : 'smooth' });
    li.classList.remove('is-ping');
    void li.offsetWidth;
    li.classList.add('is-ping');
  }
  function setHover(id) {
    state.hover = id;
    document.querySelectorAll('.floor').forEach((n) => n.classList.toggle('is-hover', n.dataset.floor === id));
    document.querySelectorAll('.tl-seg[data-step]').forEach((n) => n.classList.toggle('is-hover', n.dataset.step === id));
  }

  // 定界结论里的下一步：直接跳到该看的那条 lane，或下钻到一个 P90 筛查步
  $('#panel-path').addEventListener('click', (e) => {
    const s = e.target.closest('[data-go-step]');
    if (s) { selectStep(s.dataset.goStep, false); setView('step'); return; }
    const b = e.target.closest('[data-go-tab]');
    if (!b) return;
    if (b.dataset.goLane != null) selectLane(b.dataset.goLane);
    selectTab(b.dataset.goTab);
  });

  // 设备分布：点 lane 行进专项，点面包屑返回
  $('#panel-dist').addEventListener('click', (e) => {
    // 算子级下钻里换一步看：最快 / 中位 / 最慢
    const o = e.target.closest('[data-ops-step]');
    if (o) { state.opsStep = o.dataset.opsStep; renderDistPanel(); return; }
    const n = e.target.closest('[data-lane]');
    if (n) selectLane(n.dataset.lane);
  });
  $('#panel-dist').addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    const n = e.target.closest('[data-lane]');
    if (n) { e.preventDefault(); selectLane(n.dataset.lane); }
  });

  // 右栏页签：点击或 ← / → 切换
  const tabBar = document.querySelector('.side-tabs');
  tabBar.addEventListener('click', (e) => { const b = e.target.closest('[data-tab]'); if (b) selectTab(b.dataset.tab); });
  tabBar.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    e.preventDefault();
    const next = TABS[(TABS.indexOf(state.tab) + (e.key === 'ArrowRight' ? 1 : TABS.length - 1)) % TABS.length];
    selectTab(next);
    document.querySelector(`.side-tab[data-tab="${next}"]`).focus();
  });

  // 泳道范围：全程 ↔ 单步
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && state.view === 'step') { e.preventDefault(); setView('e2e'); }
  });

  // 每步构成 ↔ 板
  $('#stepMix').addEventListener('click', (e) => { const b = e.target.closest('[data-mix]'); if (b) selectLayer(b.dataset.mix); });

  // 请求时间线：每一段就是一个 step
  const reqBox = $('#request');
  $('#timeZoomIn').addEventListener('click', () => zoomTime(1.5));
  $('#timeZoomOut').addEventListener('click', () => zoomTime(1 / 1.5));
  $('#timeZoomReset').addEventListener('click', () => { state.timeRange = null; drawLanes(); });
  svg.addEventListener('wheel', (e) => {
    const p = svgPoint(e);
    if (!p) return;
    const ratio = Math.max(0, Math.min(1, (p.x - PAD.l) / W));
    if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
      e.preventDefault();
      const plotPx = svg.clientWidth * (W / (PAD.l + W + (state.view === 'e2e' ? PAD_R_E2E : PAD.r)));
      panTime((e.shiftKey ? e.deltaY : e.deltaX) / Math.max(1, plotPx));
      return;
    }
    e.preventDefault();
    zoomTime(e.deltaY < 0 ? 1.25 : 1 / 1.25, ratio);
  }, { passive: false });
  svg.addEventListener('pointerdown', (e) => {
    if (e.button !== 1 && !e.shiftKey) return;
    e.preventDefault();
    state.pan = { pointerId: e.pointerId, x: e.clientX };
    svg.classList.add('is-panning');
    svg.setPointerCapture(e.pointerId);
  });
  svg.addEventListener('pointermove', (e) => {
    if (!state.pan || state.pan.pointerId !== e.pointerId) return;
    const dx = e.clientX - state.pan.x;
    state.pan.x = e.clientX;
    const plotPx = svg.clientWidth * (W / (PAD.l + W + (state.view === 'e2e' ? PAD_R_E2E : PAD.r)));
    panTime(-dx / Math.max(1, plotPx));
  });
  const endPan = (e) => {
    if (!state.pan || state.pan.pointerId !== e.pointerId) return;
    state.pan = null;
    svg.classList.remove('is-panning');
    if (svg.hasPointerCapture(e.pointerId)) svg.releasePointerCapture(e.pointerId);
  };
  svg.addEventListener('pointerup', endPan);
  svg.addEventListener('pointercancel', endPan);
  reqBox.addEventListener('mouseover', (e) => { const s = e.target.closest('.tl-seg[data-step]'); if (s && s.dataset.step !== state.hover) setHover(s.dataset.step); });
  reqBox.addEventListener('mouseleave', () => setHover(null));
  reqBox.addEventListener('click', (e) => {
    const p90 = e.target.closest('[data-open-p90]');
    if (p90) { selectTab('path'); focusSlowList(); return; }
    const up = e.target.closest('[data-view]');
    if (up) { setView(up.dataset.view); return; }
    const s = e.target.closest('.tl-seg[data-step]');
    if (s) { setView('step'); selectStep(s.dataset.step, true); }
  });

  // 右侧卡片
  const floorList = $('#floorList');
  floorList.addEventListener('mouseover', (e) => { const f = e.target.closest('.floor'); if (f && f.dataset.floor !== state.hover) setHover(f.dataset.floor); });
  floorList.addEventListener('mouseleave', () => setHover(null));
  floorList.addEventListener('click', (e) => { const c = e.target.closest('.card, .sc'); if (c) { setView('step'); selectStep(c.dataset.floor, false); } });
  floorList.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
    e.preventDefault();
    const i = stepIds.indexOf(state.step);
    const next = stepIds[Math.max(0, Math.min(stepIds.length - 1, i + (e.key === 'ArrowUp' ? 1 : -1)))];
    selectStep(next, true);
    const btn = document.querySelector(`.floor[data-floor="${CSS.escape(next)}"] .card`);
    if (btn) btn.focus({ preventScroll: true });
  });

  if (window.ResizeObserver) {
    let last = '';
    new ResizeObserver(() => {
      const r = svg.getBoundingClientRect();
      const key = `${Math.round(r.width)}x${Math.round(r.height)}`;
      if (key === last) return;
      last = key;
      drawLanes();
    }).observe(svg);
  }

  let resizeTimer = 0;
  window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { drawLanes(); syncNav(); }, 120); });

  // 外部改 hash（粘贴链接、前进后退）时重新读状态；writeHash 用 replaceState，不会触发这里
  window.addEventListener('hashchange', () => { readHash(); render(); scrollToStep(false); });

  readHash();
  render();
  scrollToStep(false);
}());
