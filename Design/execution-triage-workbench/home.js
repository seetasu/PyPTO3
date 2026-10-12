/* Execution Triage · Home（DSpark Serving）
 * 一次请求 = 三栋楼（Serving / Host / Device）。楼高 = 该层在关键路径上的耗时，
 * 楼层 = step，楼面每一户 = 一张 NPU；右侧列出每一层由什么构成。数据见 build-home-data.cjs。 */
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
   * 等轴测投影：x 向右下，y 向左下，z 向上。 */
  const K = 34;
  const CX = K * Math.cos(Math.PI / 6);
  const SY = K * Math.sin(Math.PI / 6);
  const P = (x, y, z) => [(x - y) * CX, (x + y) * SY - z];
  const pts = (list) => list.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ');
  const H_MAX = 300;
  const W = 2.8;
  const D = 2.0;
  const SITES = { serving: [0, 5.0], host: [2.7, 2.0], device: [5.4, -1.0] };

  function el(name, attrs, parent) {
    const n = document.createElementNS(NS, name);
    Object.entries(attrs || {}).forEach(([k, v]) => n.setAttribute(k, v));
    if (parent) parent.appendChild(n);
    return n;
  }

  /* ------------------------------------------------------------ state */
  const state = { layer: RUN.bottleneck, floor: null, hover: null, hoverDev: null, grow: 1 };
  const tower = () => RUN.towers[state.layer];
  let floorGeo = [];
  let towerGeo = null;

  /* 楼面分户：device 沿两个可见立面排开——左立面从左到右，转过墙角接右立面。 */
  function facade(x0, y0, n) {
    const nL = n === 1 ? 1 : Math.ceil(n / 2);
    const nR = n - nL;
    const yF = y0 + D; const xF = x0 + W;
    const span = (i) => (i < nL
      ? [[x0 + (W * i) / nL, yF], [x0 + (W * (i + 1)) / nL, yF]]
      : [[xF, yF - (D * (i - nL)) / nR], [xF, yF - (D * (i - nL + 1)) / nR]]);
    const cell = (i, z0, z1) => { const [a, b] = span(i); return [P(a[0], a[1], z0), P(b[0], b[1], z0), P(b[0], b[1], z1), P(a[0], a[1], z1)]; };
    const mid = (i, z) => { const [a, b] = span(i); return P((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, z); };
    const dividers = [];
    for (let i = 1; i < nL; i += 1) dividers.push([x0 + (W * i) / nL, yF]);
    for (let j = 1; j < nR; j += 1) dividers.push([xF, yF - (D * j) / nR]);
    return { n, cell, mid, dividers };
  }

  /* 同一层内与中位数相比越慢越红；只在 heat 为 true 的楼上使用。 */
  function heatAlpha(f, i) {
    const v = f.devs[i];
    const vals = f.devs.slice().sort((a, b) => a - b);
    const med = vals[Math.floor(vals.length / 2)] || 1;
    return Math.max(0, Math.min(1, (v / med - 1.25) / 0.9)) * 0.85;
  }
  const devCount = (t) => (t.devices ? t.devices.labels.length : 1);
  function critMode(t) {
    const c = {};
    t.floors.forEach((f) => { if (f.critDev != null) c[f.critDev] = (c[f.critDev] || 0) + 1; });
    const k = Object.keys(c).sort((a, b) => c[b] - c[a])[0];
    return k == null ? null : Number(k);
  }

  function readHash() {
    const p = new URLSearchParams(location.hash.slice(1));
    if (LAYERS.includes(p.get('layer'))) state.layer = p.get('layer');
    if (p.get('floor')) state.floor = p.get('floor');
  }
  function writeHash() {
    const h = `layer=${state.layer}${state.floor ? `&floor=${encodeURIComponent(state.floor)}` : ''}`;
    if (location.hash.slice(1) !== h) history.replaceState(null, '', `#${h}`);
  }

  /* ------------------------------------------------------------ scene */
  const svg = $('#iso');

  function viewBox() {
    // 固定视窗：按最高楼计算边界
    const xs = []; const ys = [];
    LAYERS.forEach((l) => {
      const [x0, y0] = SITES[l];
      [[x0, y0], [x0 + W, y0], [x0 + W, y0 + D], [x0, y0 + D]].forEach(([x, y]) => {
        [0, H_MAX].forEach((z) => { const p = P(x, y, z); xs.push(p[0]); ys.push(p[1]); });
      });
    });
    const minX = Math.min(...xs); const maxX = Math.max(...xs);
    const minY = Math.min(...ys); const maxY = Math.max(...ys);
    const narrow = svg.clientWidth < 640;
    const padL = narrow ? 30 : 230; const padR = narrow ? 90 : 140; const padT = narrow ? 30 : 70; const padB = narrow ? 70 : 160;
    return [minX - padL, minY - padT, maxX - minX + padL + padR, maxY - minY + padT + padB];
  }

  function renderScene() {
    svg.innerHTML = '';
    const vb = viewBox();
    svg.setAttribute('viewBox', vb.join(' '));
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');

    // 地面网格，四周淡出
    const defs = el('defs', {}, svg);
    const grad = el('radialGradient', { id: 'fade', cx: '50%', cy: '62%', r: '55%' }, defs);
    el('stop', { offset: '0%', 'stop-color': '#fff', 'stop-opacity': 1 }, grad);
    el('stop', { offset: '100%', 'stop-color': '#fff', 'stop-opacity': 0 }, grad);
    const mask = el('mask', { id: 'groundMask' }, defs);
    el('rect', { x: vb[0], y: vb[1], width: vb[2], height: vb[3], fill: 'url(#fade)' }, mask);
    const ground = el('g', { class: 'ground', mask: 'url(#groundMask)' }, svg);
    for (let i = -4; i <= 10; i += 1) {
      let a = P(i, -5, 0); let b = P(i, 9, 0);
      el('line', { x1: a[0], y1: a[1], x2: b[0], y2: b[1] }, ground);
      a = P(-4, i - 1, 0); b = P(10, i - 1, 0);
      el('line', { x1: a[0], y1: a[1], x2: b[0], y2: b[1] }, ground);
    }

    const maxMs = Math.max(...LAYERS.map((l) => RUN.towers[l].ms));
    // 由后往前画（x + y 小的在后）
    const order = LAYERS.slice().sort((a, b) => (SITES[a][0] + SITES[a][1]) - (SITES[b][0] + SITES[b][1]));
    floorGeo = [];
    towerGeo = null;
    order.forEach((layer) => drawTower(layer, maxMs));
    LAYERS.forEach((layer, i) => drawTag(layer, i + 1));
    el('g', { id: 'floorLabel' }, svg);
    updateFloorLabel();
  }

  function drawTower(layer, maxMs) {
    const t = RUN.towers[layer];
    const [x0, y0] = SITES[layer];
    const on = state.layer === layer;
    const A = (z) => P(x0, y0, z); const B = (z) => P(x0 + W, y0, z);
    const C = (z) => P(x0 + W, y0 + D, z); const Dd = (z) => P(x0, y0 + D, z);
    const g = el('g', { class: `tower${on ? ' is-on' : ''}${t.upper ? ' upper' : ''}`, 'data-layer': layer }, svg);
    const e = 0.22;
    el('polygon', { class: 'plinth', points: pts([P(x0 - e, y0 - e, 0), P(x0 + W + e, y0 - e, 0), P(x0 + W + e, y0 + D + e, 0), P(x0 - e, y0 + D + e, 0)]) }, g);

    const h = Math.max(3, (H_MAX * t.ms) / maxMs) * state.grow;
    el('polygon', { class: 'face', points: pts([Dd(0), C(0), C(h), Dd(h)]) }, g);
    el('polygon', { class: 'face', points: pts([C(0), B(0), B(h), C(h)]) }, g);
    el('polygon', { class: 'face top', points: pts([A(h), B(h), C(h), Dd(h)]) }, g);

    if (on && t.floors.length) {
      const total = t.floors.reduce((a, f) => a + f.ms, 0) || 1;
      const n = devCount(t);
      const fc = facade(x0, y0, n);
      let z = 0;
      t.floors.forEach((f) => {
        const z1 = z + (h * f.ms) / total;
        floorGeo.push({ id: f.id, z0: z, z1, B, C, Dd, h, f });
        z = z1;
      });
      towerGeo = { fc, h, n, t };
      // 1) 热度：每户每层按相对中位数着色
      if (n > 1 && t.devices.heat) {
        floorGeo.forEach((fg) => {
          for (let i = 0; i < n; i += 1) {
            const a = heatAlpha(fg.f, i);
            if (a > 0.02) el('polygon', { class: 'heat', points: pts(fc.cell(i, fg.z0, fg.z1)), 'fill-opacity': a.toFixed(3) }, g);
          }
        });
      }
      // 2) 楼层线与分户线
      floorGeo.forEach((fg, i) => { if (i > 0 && h > 12) el('polyline', { class: 'floor-line', points: pts([Dd(fg.z0), C(fg.z0), B(fg.z0)]) }, g); });
      fc.dividers.forEach(([dx, dy]) => { const a = P(dx, dy, 0); const b = P(dx, dy, h); el('line', { class: 'col-line', x1: a[0], y1: a[1], x2: b[0], y2: b[1] }, g); });
      // 3) 户名：只标最常成为关键的那张卡
      const mode = n > 1 && t.devices.heat ? critMode(t) : null;
      if (state.grow >= 1 && mode != null) {
        const p = fc.mid(mode, h);
        const lb = el('text', { class: 'dev-label hot', x: p[0], y: p[1] - 12, 'text-anchor': 'middle' }, g);
        lb.textContent = t.devices.labels[mode];
        el('line', { class: 'dev-tick', x1: p[0], y1: p[1] - 8, x2: p[0], y2: p[1] - 2 }, g);
      }
      // 4) 交互：多卡按户命中，单卡按整层命中（动画结束后再挂）
      if (state.grow >= 1) {
        floorGeo.forEach((fg) => {
          const cells = n > 1 ? t.devices.labels.map((_, i) => i) : [null];
          cells.forEach((i) => {
            const poly = i == null
              ? [fg.Dd(fg.z0), fg.C(fg.z0), fg.B(fg.z0), fg.B(fg.z1), fg.C(fg.z1), fg.Dd(fg.z1)]
              : fc.cell(i, fg.z0, fg.z1);
            const hit = el('polygon', { class: `floor-hit${fg.id === state.floor ? ' is-sel' : ''}`, points: pts(poly), 'data-floor': fg.id }, g);
            if (i != null) hit.setAttribute('data-dev', i);
            hit.addEventListener('mouseenter', () => setHover(fg.id, i));
            hit.addEventListener('mouseleave', () => setHover(null));
            hit.addEventListener('click', (ev) => { ev.stopPropagation(); selectFloor(fg.id, true); });
          });
        });
      }
      // 5) 顶部两角的括号，呼应选中
      const tl = Dd(h); const tr = B(h);
      el('path', { class: 'bracket', d: `M${tl[0] - 10},${tl[1] - 2} v-8 h8` }, g);
      el('path', { class: 'bracket', d: `M${tr[0] + 10},${tr[1] - 2} v-8 h-8` }, g);
    }
    if (!on) g.addEventListener('click', () => selectLayer(layer));
  }

  function drawTag(layer, n) {
    const t = RUN.towers[layer];
    const [x0, y0] = SITES[layer];
    const c = P(x0 + W, y0 + D, 0);
    const on = state.layer === layer;
    const stepMs = RUN.perStep.layers[layer].meanMs;
    const label = `${NAME[layer]} · ${t.upper ? '≤ ' : ''}${fmtMs(stepMs)}${svg.clientWidth < 640 ? '' : '/step'}`;
    const g = el('g', { class: `tag${on ? ' is-on' : ''}`, tabindex: 0, role: 'button', 'aria-pressed': on, 'aria-label': `${NAME[layer]}：每步 ${fmtMs(stepMs)}` }, svg);
    const x = c[0] - 10; const y = c[1] + 18;
    el('rect', { class: 'tag-box', x, y, width: 20, height: 20, rx: 3 }, g);
    const num = el('text', { class: 'tag-num', x: x + 10, y: y + 14, 'text-anchor': 'middle' }, g);
    num.textContent = n;
    const lb = el('text', { class: 'tag-label', x: x + 28, y: y + 14 }, g);
    lb.textContent = label;
    g.addEventListener('click', () => selectLayer(layer));
    g.addEventListener('keydown', (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); selectLayer(layer); } });
  }

  function updateFloorLabel() {
    const g = svg.querySelector('#floorLabel');
    if (!g) return;
    g.innerHTML = '';
    svg.querySelectorAll('.floor-hit').forEach((n) => {
      n.classList.toggle('is-hover', n.dataset.floor === state.hover);
      n.classList.toggle('is-sel', n.dataset.floor === state.floor);
    });
    const id = state.hover || state.floor;
    const fg = floorGeo.find((f) => f.id === id);
    const dev = state.hover ? state.hoverDev : null;
    if (towerGeo && dev != null) {
      // 悬停某一户：整列描边，看这张卡在所有楼层上的表现
      el('polygon', { class: 'col-hover', points: pts(towerGeo.fc.cell(dev, 0, towerGeo.h)) }, g);
      if (fg) el('polygon', { class: 'cell-hover', points: pts(towerGeo.fc.cell(dev, fg.z0, fg.z1)) }, g);
    }
    if (!fg) return;
    const p = fg.B((fg.z0 + fg.z1) / 2);
    el('path', { class: 'bracket', d: `M${p[0] + 6},${p[1]} h14` }, g);
    const t = el('text', { class: 'floor-tag', x: p[0] + 26, y: p[1] + 4 }, g);
    t.textContent = `${id} STEP`;
    if (dev != null && towerGeo) {
      const sub = el('text', { class: 'floor-tag-sub', x: p[0] + 26, y: p[1] + 20 }, g);
      sub.textContent = `${towerGeo.t.devices.labels[dev]} · ${fmtMs(fg.f.devs[dev])}`;
    }
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
    const segs = q.timeline.map((s) => `<span class="tl-seg k-${s.kind}${s.id === state.floor ? ' is-sel' : ''}" data-step="${esc(s.id)}" style="flex:${s.ms} 0 0" title="${esc(s.id)} · ${fmtMs(s.ms)}"></span>`).join('')
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

  function renderText() {
    renderStepHead();
    renderRequest();
    $('#analysisLink').href = `./analysis.html#layer=${state.layer}`;
  }

  /* ------------------------------------------------------------ floors */
  function renderFloors() {
    const t = tower();
    $('#floorsTitle').textContent = `${NAME[state.layer]} · 每个 step`;
    $('#floorList').innerHTML = t.floors.slice().reverse().map((f) => `
      <div class="floor${f.id === state.floor ? ' is-sel' : ''}" data-floor="${esc(f.id)}">
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

  function strip(t, f) {
    const n = devCount(t);
    if (n < 2 || !f.devs) return '';
    const cells = f.devs.map((v, i) => {
      const a = t.devices.heat ? heatAlpha(f, i) : 0;
      const crit = t.devices.heat && f.critDev === i;
      return `<span class="sc${crit ? ' crit' : ''}" data-floor="${esc(f.id)}" data-dev="${i}" style="--a:${a.toFixed(3)}" title="${esc(`${t.devices.labels[i]} · ${fmtMs(v)}`)}"></span>`;
    }).join('');
    return `<div class="strip" style="--n:${n}" aria-label="每张卡的耗时">${cells}</div>`;
  }

  /* ------------------------------------------------------------ actions */
  function render() { renderScene(); renderText(); renderFloors(); writeHash(); }

  function selectLayer(layer) {
    if (state.layer === layer) return;
    state.layer = layer; state.floor = null; state.hover = null;
    render();
  }
  function selectFloor(id, scroll) {
    state.floor = state.floor === id ? null : id;
    updateFloorLabel();
    document.querySelectorAll('.floor').forEach((n) => n.classList.toggle('is-sel', n.dataset.floor === state.floor));
    document.querySelectorAll('.tl-seg[data-step]').forEach((n) => n.classList.toggle('is-sel', n.dataset.step === state.floor));
    if (scroll) scrollToFloor(true);
    writeHash();
  }
  function scrollToFloor(smooth) {
    if (!state.floor) return;
    const row = document.querySelector(`.floor[data-floor="${CSS.escape(state.floor)}"]`);
    const list = $('#floorList');
    if (row && list.scrollHeight > list.clientHeight) {
      const top = row.offsetTop - list.offsetTop - (list.clientHeight - row.offsetHeight) / 2;
      list.scrollTo({ top: Math.max(0, top), behavior: smooth && !document.hidden ? 'smooth' : 'auto' });
    } else if (row) {
      row.scrollIntoView({ block: 'nearest' });
    }
  }
  function setHover(id, dev) {
    state.hover = id;
    state.hoverDev = id == null || dev == null ? null : Number(dev);
    updateFloorLabel();
    document.querySelectorAll('.floor').forEach((n) => n.classList.toggle('is-hover', n.dataset.floor === id));
    document.querySelectorAll('.tl-seg[data-step]').forEach((n) => n.classList.toggle('is-hover', n.dataset.step === id));
    document.querySelectorAll('.strip .sc').forEach((n) => n.classList.toggle('is-col', state.hoverDev != null && Number(n.dataset.dev) === state.hoverDev));
  }
  function grow() {
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce || document.hidden) { state.grow = 1; render(); return; }
    const t0 = performance.now();
    renderText(); renderFloors(); writeHash();
    const tick = (now) => {
      const k = Math.min(1, (now - t0) / 560);
      state.grow = 1 - (1 - k) ** 3;
      renderScene();
      if (k < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    // rAF 在后台标签页会暂停：兜底直接落到最终状态
    setTimeout(() => { if (state.grow < 1) { state.grow = 1; renderScene(); } }, 700);
  }

  // 每步构成 ↔ 楼
  $('#stepMix').addEventListener('click', (e) => { const b = e.target.closest('[data-mix]'); if (b) selectLayer(b.dataset.mix); });

  // 请求时间线 ↔ 楼层：时间线上的每一段就是一个 step
  const reqBox = $('#request');
  const hasStep = (id) => tower().floors.some((f) => f.id === id);
  reqBox.addEventListener('mouseover', (e) => { const s = e.target.closest('.tl-seg[data-step]'); if (s && hasStep(s.dataset.step) && s.dataset.step !== state.hover) setHover(s.dataset.step); });
  reqBox.addEventListener('mouseleave', () => setHover(null));
  reqBox.addEventListener('click', (e) => { const s = e.target.closest('.tl-seg[data-step]'); if (s && hasStep(s.dataset.step)) selectFloor(s.dataset.step, true); });

  // 右侧卡片 ↔ 场景楼层
  const floorList = $('#floorList');
  floorList.addEventListener('mouseover', (e) => {
    const f = e.target.closest('.floor');
    if (!f) return;
    const sc = e.target.closest('.sc');
    const dev = sc ? Number(sc.dataset.dev) : null;
    if (f.dataset.floor !== state.hover || dev !== state.hoverDev) setHover(f.dataset.floor, dev);
  });
  floorList.addEventListener('mouseleave', () => setHover(null));
  floorList.addEventListener('click', (e) => { const c = e.target.closest('.card, .sc'); if (c) selectFloor(c.dataset.floor, false); });
  floorList.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
    const ids = tower().floors.map((f) => f.id);
    e.preventDefault();
    const i = ids.indexOf(state.floor);
    const next = i < 0 ? ids[ids.length - 1] : ids[Math.max(0, Math.min(ids.length - 1, i + (e.key === 'ArrowUp' ? 1 : -1)))];
    state.floor = null;
    selectFloor(next, true);
    const btn = document.querySelector(`.floor[data-floor="${CSS.escape(next)}"] .card`);
    if (btn) btn.focus({ preventScroll: true });
  });

  let resizeTimer = 0;
  window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(renderScene, 120); });

  readHash();
  if (state.floor && !tower().floors.some((f) => f.id === state.floor)) state.floor = null;
  grow();
  scrollToFloor(false);
}());
