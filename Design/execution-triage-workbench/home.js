/* Execution Triage · Home（DSpark Serving）
 * 中间是一个 step 的等轴测泳道：Serving / Host / Device 三块板，时间沿纵深前进。
 * 红色折线 = 关键路径；灰色箭头 = 非关键的流转；虚线 = 等待；虚框 = 与本步重叠的下一步准备。
 * 数据见 build-home-data.cjs。 */
(function () {
  'use strict';

  const RUN = window.TRIAGE_HOME.runs[0];
  const LAYERS = ['serving', 'host', 'device'];
  const NAME = { serving: 'Serving', host: 'Host', device: 'Device' };
  const NS = 'http://www.w3.org/2000/svg';
  const $ = (s) => document.querySelector(s);
  const esc = (v) => String(v).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

  function fmtMs(ms) {
    const v = Math.abs(ms);
    if (v >= 1000) return `${(ms / 1000).toFixed(2)} s`;
    if (v >= 100) return `${Math.round(ms)} ms`;
    if (v >= 10) return `${ms.toFixed(1)} ms`;
    return `${ms.toFixed(2)} ms`;
  }

  /* ------------------------------------------------------------ geometry
   * 斜二测投影：x（时间）水平向右，y（泳道）朝观者向下并略向左倾，z 向上。
   * 三块板从上到下依次是 Serving / Host / Device。 */
  const UX = 60;
  const LX = -14;
  const LY = 44;
  const P = (x, y, z) => [x * UX + y * LX, y * LY - z];
  const pts = (list) => list.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ');
  const X_LEN = 13;
  const N_LANE = 16;
  const PITCH = 0.17;
  const LANE_D = 0.12;
  const PLATES = {
    serving: { y0: 0, y1: 0.7 },
    host: { y0: 1.25, y1: 1.85 + N_LANE * PITCH },
    device: { y0: 1.85 + N_LANE * PITCH + 0.55, y1: 1.85 + N_LANE * PITCH + 0.55 + 0.1 + N_LANE * PITCH },
  };
  const ROW = {
    serving: 0.35,
    worker: 1.55,
    host: (dev) => 1.88 + dev * PITCH + LANE_D / 2,
    device: (dev) => PLATES.device.y0 + 0.08 + dev * PITCH + LANE_D / 2,
  };

  function el(name, attrs, parent) {
    const n = document.createElementNS(NS, name);
    Object.entries(attrs || {}).forEach(([k, v]) => n.setAttribute(k, v));
    if (parent) parent.appendChild(n);
    return n;
  }

  /* ------------------------------------------------------------ state */
  const stepIds = RUN.request.timeline.map((s) => s.id);
  const state = { layer: RUN.bottleneck, step: RUN.defaultStep, hover: null };
  const tower = () => RUN.towers[state.layer];

  function readHash() {
    const p = new URLSearchParams(location.hash.slice(1));
    if (LAYERS.includes(p.get('layer'))) state.layer = p.get('layer');
    if (stepIds.includes(p.get('step'))) state.step = p.get('step');
  }
  function writeHash() {
    const h = `layer=${state.layer}&step=${encodeURIComponent(state.step)}`;
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

  function viewBox() {
    const xs = []; const ys = [];
    [[0, PLATES.serving.y0], [X_LEN, PLATES.serving.y0], [X_LEN, PLATES.device.y1], [0, PLATES.device.y1]].forEach(([x, y]) => {
      [0, 24].forEach((z) => { const p = P(x, y, z); xs.push(p[0]); ys.push(p[1]); });
    });
    const minX = Math.min(...xs); const maxX = Math.max(...xs); const minY = Math.min(...ys); const maxY = Math.max(...ys);
    const padL = 110; const padR = 20; const padT = 34; const padB = 36;
    return [minX - padL, minY - padT, maxX - minX + padL + padR, maxY - minY + padT + padB];
  }

  // 一个低矮的盒子：顶面 + 朝前的两个侧面
  function box(g, x0, x1, y0, y1, h, cls, tipText) {
    if (x1 - x0 < 0.02) x1 = x0 + 0.02;
    const grp = el('g', { class: `blk ${cls}` }, g);
    el('polygon', { class: 'side', points: pts([P(x0, y1, 0), P(x1, y1, 0), P(x1, y1, h), P(x0, y1, h)]) }, grp);
    el('polygon', { class: 'side', points: pts([P(x1, y0, 0), P(x1, y1, 0), P(x1, y1, h), P(x1, y0, h)]) }, grp);
    el('polygon', { class: 'top', points: pts([P(x0, y0, h), P(x1, y0, h), P(x1, y1, h), P(x0, y1, h)]) }, grp);
    if (tipText) withTip(grp, tipText);
    return grp;
  }
  function line(g, a, b, cls) { return el('line', { class: cls, x1: a[0], y1: a[1], x2: b[0], y2: b[1] }, g); }

  function renderScene() {
    const f = RUN.flows[state.step];
    svg.innerHTML = '';
    const vb = viewBox();
    svg.setAttribute('viewBox', vb.join(' '));
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    const X = (t) => ((t - f.t0) / (f.t1 - f.t0)) * X_LEN;
    const half = LANE_D / 2;

    const defs = el('defs', {}, svg);
    const hatch = el('pattern', { id: 'rrHatch', width: 5, height: 5, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, defs);
    el('rect', { width: 1.2, height: 5, fill: 'rgba(25,158,112,0.55)' }, hatch);
    const arrow = el('marker', { id: 'arrowGrey', viewBox: '0 0 8 8', refX: 7, refY: 4, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, defs);
    el('path', { d: 'M0,0 L8,4 L0,8 z', fill: 'rgba(255,255,255,0.55)' }, arrow);
    const arrowHot = el('marker', { id: 'arrowHot', viewBox: '0 0 8 8', refX: 7, refY: 4, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, defs);
    el('path', { d: 'M0,0 L8,4 L0,8 z', fill: '#ef4b3c' }, arrowHot);

    // 1) 三块板：t < 0 的部分是“上一步期间”，画得更暗
    LAYERS.forEach((layer, i) => {
      const pl = PLATES[layer];
      const on = state.layer === layer;
      const g = el('g', { class: `plate${on ? ' is-on' : ''}`, 'data-layer': layer }, svg);
      el('polygon', { class: 'plate-pre', points: pts([P(0, pl.y0, 0), P(X(0), pl.y0, 0), P(X(0), pl.y1, 0), P(0, pl.y1, 0)]) }, g);
      el('polygon', { class: 'plate-main', points: pts([P(X(0), pl.y0, 0), P(X_LEN, pl.y0, 0), P(X_LEN, pl.y1, 0), P(X(0), pl.y1, 0)]) }, g);
      el('polygon', { class: 'plate-edge', points: pts([P(0, pl.y0, 0), P(X_LEN, pl.y0, 0), P(X_LEN, pl.y1, 0), P(0, pl.y1, 0)]) }, g);
      // 板名：左上角的编号方块
      const a = P(0, pl.y0 + 0.3, 0);
      const tag = el('g', { class: `tag${on ? ' is-on' : ''}`, tabindex: 0, role: 'button', 'aria-pressed': on, 'aria-label': NAME[layer] }, g);
      el('rect', { class: 'tag-box', x: a[0] - 30, y: a[1] - 10, width: 20, height: 20, rx: 3 }, tag);
      const num = el('text', { class: 'tag-num', x: a[0] - 20, y: a[1] + 4, 'text-anchor': 'middle' }, tag);
      num.textContent = i + 1;
      const lb = el('text', { class: 'tag-label', x: a[0] - 38, y: a[1] + 4, 'text-anchor': 'end' }, tag);
      lb.textContent = NAME[layer];
      g.addEventListener('click', () => selectLayer(layer));
      tag.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectLayer(layer); } });
    });

    // 2) 时间参考线：本步起点、关键 lane 进入 runner_run、同步屏障
    const guides = el('g', { class: 'guides' }, svg);
    line(guides, P(X(0), PLATES.serving.y0 - 0.15, 0), P(X(0), PLATES.device.y1 + 0.15, 0), 'g-start');
    line(guides, P(X(f.critStart), PLATES.host.y0, 0), P(X(f.critStart), PLATES.device.y1 + 0.15, 0), 'g-crit');

    // 3) Serving 板
    const gs = el('g', {}, svg);
    f.serving.forEach((s) => {
      const y = ROW.serving;
      if (s.kind === 'wait') { line(gs, P(X(s.s), y, 0), P(X(s.s + s.d), y, 0), 'wait'); return; }
      box(gs, X(s.s), X(s.s + Math.max(s.d, 0.6)), y - 0.16, y + 0.16, s.kind === 'own' ? 7 : 4, s.kind, `Serving · ${s.n} ${fmtMs(s.d)}`);
    });

    // 4) Host 板：Worker 一行 + 16 条 lane
    const gh = el('g', {}, svg);
    f.worker.forEach((w) => {
      const y = ROW.worker;
      if (w.kind === 'wait') { line(gh, P(X(w.s), y, 0), P(X(w.s + w.d), y, 0), 'wait'); return; }
      box(gh, X(w.s), X(w.s + w.d), y - 0.18, y + 0.18, w.kind === 'other' ? 3 : 7, w.kind, `Worker · ${w.n} ${fmtMs(w.d)}${w.kind === 'next' ? '（下一步）' : ''}`);
    });
    f.lanes.forEach((l) => {
      const y = ROW.host(l.dev); const crit = l.dev === f.crit;
      if (l.rrStart - l.bindEnd > 0.5) line(gh, P(X(l.bindEnd), y, 0), P(X(l.rrStart), y, 0), 'wait');
      box(gh, X(l.bindStart), X(l.bindEnd), y - half, y + half, crit ? 7 : 4, `bind${crit ? ' crit' : ''}`, `dev ${l.dev} · bind ${fmtMs(l.bindEnd - l.bindStart)}`);
    });

    // 5) Device 板：16 张卡的 runner_run，末端是同步屏障
    const gd = el('g', {}, svg);
    f.lanes.forEach((l) => {
      const y = ROW.device(l.dev); const crit = l.dev === f.crit;
      box(gd, X(l.rrStart), X(l.end), y - half, y + half, crit ? 7 : 4, `rr${crit ? ' crit' : ''}`, `dev ${l.dev} · runner_run ${fmtMs(l.end - l.rrStart)}`);
    });
    line(gd, P(X(f.end), PLATES.device.y0, 0), P(X(f.end), PLATES.device.y1, 0), 'barrier');
    line(gd, P(X(f.end), PLATES.device.y0, 0), P(X(f.end), PLATES.device.y0, 22), 'barrier');

    // 6) 关键 lane 的行名
    const gl = el('g', {}, svg);
    [ROW.host(f.crit), ROW.device(f.crit)].forEach((y) => {
      const p = P(0, y, 0);
      const t = el('text', { class: 'lane-label', x: p[0] - 8, y: p[1] + 3, 'text-anchor': 'end' }, gl);
      t.textContent = `dev ${f.crit}`;
    });

    // 7) 非关键流转（灰）：调度下发 → Worker → lane；屏障 → 回收 → 调度输出
    const gf = el('g', { class: 'flow' }, svg);
    const Z = 10;
    const L = f.links;
    const poly = (list, cls, marker) => el('polyline', { class: cls, points: pts(list), 'marker-end': marker ? `url(#${marker})` : '' }, gf);
    if (L.queue != null && L.dispatch) poly([P(X(L.queue), ROW.serving, Z), P(X(L.dispatch[0]), ROW.worker, Z)], 'flow-line', 'arrowGrey');
    if (L.dispatch) poly([P(X(L.dispatch[1]), ROW.worker, Z), P(X(f.laneStart), ROW.host(0), Z)], 'flow-line', 'arrowGrey');
    if (L.reclaimEnd != null && L.output != null) {
      poly([P(X(f.end), ROW.device(0), Z), P(X(L.reclaimEnd), ROW.worker, Z), P(X(L.output), ROW.serving, Z)], 'flow-line', 'arrowGrey');
    }

    // 8) 关键路径（红）：上一步屏障 → dev N 开始 bind → 进入 runner_run → 同步屏障
    const c = f.lanes.find((l) => l.dev === f.crit);
    const ZC = 14;
    const path = [
      P(X(0), ROW.device(f.crit), ZC),
      P(X(c.start), ROW.host(f.crit), ZC),
      P(X(c.rrStart), ROW.host(f.crit), ZC),
      P(X(c.rrStart), ROW.device(f.crit), ZC),
      P(X(f.end), ROW.device(f.crit), ZC),
    ];
    el('polyline', { class: 'crit-path', points: pts(path), pathLength: 1, 'marker-end': 'url(#arrowHot)' }, svg);
    path.slice(1, 4).forEach((p) => el('circle', { class: 'crit-node', cx: p[0], cy: p[1], r: 3 }, svg));
    const label = (a, b, text, dy) => {
      const t = el('text', { class: 'crit-label', x: (a[0] + b[0]) / 2, y: (a[1] + b[1]) / 2 + dy, 'text-anchor': 'middle' }, svg);
      t.textContent = text;
    };
    label(path[1], path[2], `Host ${fmtMs(f.hostMs)}`, -12);
    label(path[3], path[4], `Device ≤ ${fmtMs(f.deviceMs)}`, -12);

    // 9) 时间轴：沿 Device 板前沿
    const ga = el('g', { class: 'axis' }, svg);
    const yA = PLATES.device.y1 + 0.15;
    const ticks = [0, f.critStart, f.end];
    for (let t = Math.ceil(f.t0 / 100) * 100; t <= f.t1; t += 100) if (t !== 0) ticks.push(t);
    ticks.forEach((t) => {
      const strong = t === 0 || t === f.critStart || t === f.end;
      const a = P(X(t), yA - 0.12, 0); const b = P(X(t), yA, 0);
      line(ga, a, b, strong ? 'tick strong' : 'tick');
      if (!strong && [f.critStart, f.end].some((s) => Math.abs(s - t) < 25)) return;
      const tx = el('text', { class: `tick-label${strong ? ' strong' : ''}`, x: b[0], y: b[1] + 14, 'text-anchor': 'middle' }, ga);
      tx.textContent = `${Math.round(t)}`;
    });
    const unit = P(X_LEN, yA, 0);
    const u = el('text', { class: 'tick-label', x: unit[0] + 4, y: unit[1] + 14, 'text-anchor': 'start' }, ga);
    u.textContent = 'ms';

    // 10) 当前 step
    const st = P(X_LEN, PLATES.serving.y0, 0);
    const t1 = el('text', { class: 'step-title', x: st[0], y: st[1] - 12, 'text-anchor': 'end' }, svg);
    t1.textContent = `${state.step} STEP · ${fmtMs(f.periodMs)}`;
  }

  /* ------------------------------------------------------------ text */
  /* P1：每个 decode step 的时间去哪了 */
  function renderStepHead() {
    const ps = RUN.perStep;
    const L = ps.layers;
    $('#eyebrow').textContent = 'decode step';
    $('#layerName').innerHTML = `${Math.round(ps.meanMs)}<small>ms</small>`;
    $('#layerName').setAttribute('aria-label', `每个 decode step 平均 ${fmtMs(ps.meanMs)}`);
    $('#layerMeta').innerHTML = `<span>P50 <b>${fmtMs(ps.p50Ms)}</b> · P90 <b>${fmtMs(ps.p90Ms)}</b></span>`;
    const max = Math.max(...LAYERS.map((l) => L[l].meanMs));
    $('#stepMix').innerHTML = LAYERS.map((l, i) => {
      const t = RUN.towers[l]; const v = L[l].meanMs; const share = (v / ps.meanMs) * 100;
      const on = state.layer === l;
      return `
        <button class="mix-row${on ? ' is-on' : ''}" type="button" data-mix="${l}" aria-pressed="${on}">
          <span class="mix-num">${i + 1}</span>
          <span class="mix-name">${NAME[l]}</span>
          <span class="mix-bar"><span class="mix-fill${t.upper ? ' upper' : ''}" style="width:${Math.max(0.6, (v / max) * 100)}%"></span></span>
          <span class="mix-val">${t.upper ? '≤ ' : ''}${fmtMs(v)}</span>
          <span class="mix-pct">${share < 1 ? '≈ 0%' : `${t.upper ? '≤ ' : ''}${Math.round(share)}%`}</span>
        </button>`;
    }).join('');
  }

  /* P0：这次请求的服务指标 + 请求时间线 */
  function renderRequest() {
    const q = RUN.request;
    const segs = q.timeline.map((s) => `<span class="tl-seg k-${s.kind}${s.id === state.step ? ' is-sel' : ''}" data-step="${esc(s.id)}" style="flex:${s.ms} 0 0" title="${esc(s.id)} · ${fmtMs(s.ms)}"></span>`).join('')
      + `<span class="tl-seg k-tail" style="flex:${q.tailMs} 0 0"></span>`;
    const first = (q.firstOutputMs / q.e2eMs) * 100;
    $('#request').innerHTML = `
      <div class="req-head"><b>${esc(RUN.title)} · ${esc(RUN.deviceText)}</b></div>
      <div class="req-kpis">
        <div class="kpi"><span>E2E</span><b>${fmtMs(q.e2eMs)}</b></div>
        <div class="kpi"><span>首次输出</span><b>${fmtMs(q.firstOutputMs)}</b></div>
        <div class="kpi"><span>输出间隔 P50</span><b>${fmtMs(q.gapP50Ms)}</b></div>
        <div class="kpi"><span>输出</span><b>${q.outputs} 次</b></div>
      </div>
      <div class="req-line" role="group" aria-label="请求时间线">
        <div class="tl">${segs}</div>
        <span class="tl-mark" style="left:${first}%"><i></i>首次输出</span>
      </div>`;
  }

  /* ------------------------------------------------------------ floors */
  function renderFloors() {
    const t = tower();
    $('#floorsTitle').textContent = `${NAME[state.layer]} · 每个 step`;
    $('#floorList').innerHTML = t.floors.slice().reverse().map((f) => `
      <div class="floor${f.id === state.step ? ' is-sel' : ''}" data-floor="${esc(f.id)}">
        <span class="floor-id">${esc(f.id)}</span>
        ${f.cards.map((c) => `
          <button class="card ${c.tone}" type="button" data-floor="${esc(f.id)}">
            <span class="ct">${esc(c.title)}</span>
            <span class="cs">${esc(c.sub)}</span>
            <span class="cv">${esc(c.text || fmtMs(c.ms))}</span>
          </button>`).join('')}
        ${strip(t, f)}
      </div>`).join('');
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
      return `<span class="sc${crit ? ' crit' : ''}" data-floor="${esc(f.id)}" data-dev="${i}" style="--a:${a.toFixed(3)}" title="${esc(`${t.devices.labels[i]} · ${fmtMs(v)}`)}"></span>`;
    }).join('');
    return `<div class="strip" style="--n:${f.devs.length}" aria-label="每张卡的耗时">${cells}</div>`;
  }

  /* ------------------------------------------------------------ actions */
  function render() {
    renderScene(); renderStepHead(); renderRequest(); renderFloors();
    $('#analysisLink').href = `./analysis.html#layer=${state.layer}&step=${encodeURIComponent(analysisStep(state.step))}`;
    writeHash();
  }
  // 详细分析页的 step id 用的是 Prefill / Draft init
  const analysisStep = (id) => (id === 'P' ? 'Prefill' : id === 'DI' ? 'Draft init' : id);

  function selectLayer(layer) {
    if (state.layer === layer) return;
    state.layer = layer;
    render();
    scrollToStep(false);
  }
  function selectStep(id, scroll) {
    if (!stepIds.includes(id) || id === state.step) return;
    state.step = id;
    renderScene();
    document.querySelectorAll('.floor').forEach((n) => n.classList.toggle('is-sel', n.dataset.floor === id));
    document.querySelectorAll('.tl-seg[data-step]').forEach((n) => n.classList.toggle('is-sel', n.dataset.step === id));
    $('#analysisLink').href = `./analysis.html#layer=${state.layer}&step=${encodeURIComponent(analysisStep(id))}`;
    if (scroll) scrollToStep(true);
    writeHash();
  }
  function scrollToStep(smooth) {
    const row = document.querySelector(`.floor[data-floor="${CSS.escape(state.step)}"]`);
    const list = $('#floorList');
    if (row && list.scrollHeight > list.clientHeight) {
      const top = row.offsetTop - list.offsetTop - (list.clientHeight - row.offsetHeight) / 2;
      list.scrollTo({ top: Math.max(0, top), behavior: smooth && !document.hidden ? 'smooth' : 'auto' });
    }
  }
  function setHover(id) {
    state.hover = id;
    document.querySelectorAll('.floor').forEach((n) => n.classList.toggle('is-hover', n.dataset.floor === id));
    document.querySelectorAll('.tl-seg[data-step]').forEach((n) => n.classList.toggle('is-hover', n.dataset.step === id));
  }

  // 每步构成 ↔ 板
  $('#stepMix').addEventListener('click', (e) => { const b = e.target.closest('[data-mix]'); if (b) selectLayer(b.dataset.mix); });

  // 请求时间线：每一段就是一个 step
  const reqBox = $('#request');
  reqBox.addEventListener('mouseover', (e) => { const s = e.target.closest('.tl-seg[data-step]'); if (s && s.dataset.step !== state.hover) setHover(s.dataset.step); });
  reqBox.addEventListener('mouseleave', () => setHover(null));
  reqBox.addEventListener('click', (e) => { const s = e.target.closest('.tl-seg[data-step]'); if (s) selectStep(s.dataset.step, true); });

  // 右侧卡片
  const floorList = $('#floorList');
  floorList.addEventListener('mouseover', (e) => { const f = e.target.closest('.floor'); if (f && f.dataset.floor !== state.hover) setHover(f.dataset.floor); });
  floorList.addEventListener('mouseleave', () => setHover(null));
  floorList.addEventListener('click', (e) => { const c = e.target.closest('.card, .sc'); if (c) selectStep(c.dataset.floor, false); });
  floorList.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
    e.preventDefault();
    const i = stepIds.indexOf(state.step);
    const next = stepIds[Math.max(0, Math.min(stepIds.length - 1, i + (e.key === 'ArrowUp' ? 1 : -1)))];
    selectStep(next, true);
    const btn = document.querySelector(`.floor[data-floor="${CSS.escape(next)}"] .card`);
    if (btn) btn.focus({ preventScroll: true });
  });

  let resizeTimer = 0;
  window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(renderScene, 120); });

  readHash();
  render();
  scrollToStep(false);
}());
