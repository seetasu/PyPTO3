/* Execution Triage · Home
 * 一次执行 = 三栋楼（Serving / Host / Device）。楼高 = 该层在关键路径上的耗时，
 * 楼层 = step（或模型层），右侧列出每一层由什么构成。数据见 build-home-data.cjs。 */
(function () {
  'use strict';

  const RUNS = window.TRIAGE_HOME.runs;
  const LAYERS = ['serving', 'host', 'device'];
  const NAME = { serving: 'Serving', host: 'Host', device: 'Device' };
  const UNIT = {
    step: { en: 'STEP', cn: 'step', each: '每个 step' },
    layer: { en: 'LAYER', cn: '层', each: '每一层' },
    call: { en: 'CALL', cn: '次调用', each: '每次调用' },
  };
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
  const fmtPct = (p) => (p > 0 && p < 0.1 ? `${p.toFixed(2)}%` : `${p.toFixed(1)}%`);

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
  const state = { run: RUNS[0].id, layer: null, floor: null, hover: null, grow: 1 };
  const run = () => RUNS.find((r) => r.id === state.run);
  const tower = () => run().towers[state.layer];
  const unitOf = (r, layer) => UNIT[(r.towers[layer].floorUnit) || r.floorUnit];
  let floorGeo = [];

  function readHash() {
    const p = new URLSearchParams(location.hash.slice(1));
    if (RUNS.some((r) => r.id === p.get('run'))) state.run = p.get('run');
    if (LAYERS.includes(p.get('layer'))) state.layer = p.get('layer');
    if (p.get('floor')) state.floor = p.get('floor');
  }
  function writeHash() {
    const h = `run=${state.run}&layer=${state.layer}${state.floor ? `&floor=${encodeURIComponent(state.floor)}` : ''}`;
    if (location.hash.slice(1) !== h) history.replaceState(null, '', `#${h}`);
  }

  /* ------------------------------------------------------------ scene */
  const svg = $('#iso');

  function viewBox() {
    // 固定视窗：按最高楼计算边界，切换执行时楼的比例保持一致
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
    const padL = narrow ? 30 : 230; const padR = narrow ? 90 : 140; const padT = narrow ? 30 : 70; const padB = narrow ? 70 : 110;
    return [minX - padL, minY - padT, maxX - minX + padL + padR, maxY - minY + padT + padB];
  }

  function renderScene() {
    const r = run();
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

    const maxMs = Math.max(...LAYERS.map((l) => (r.towers[l].na ? 0 : r.towers[l].ms)));
    // 由后往前画（x + y 小的在后）
    const order = LAYERS.slice().sort((a, b) => (SITES[a][0] + SITES[a][1]) - (SITES[b][0] + SITES[b][1]));
    floorGeo = [];
    order.forEach((layer) => drawTower(r, layer, maxMs));
    LAYERS.forEach((layer, i) => drawTag(r, layer, i + 1));
    el('g', { id: 'floorLabel' }, svg);
    updateFloorLabel();
  }

  function drawTower(r, layer, maxMs) {
    const t = r.towers[layer];
    const [x0, y0] = SITES[layer];
    const on = state.layer === layer;
    const A = (z) => P(x0, y0, z); const B = (z) => P(x0 + W, y0, z);
    const C = (z) => P(x0 + W, y0 + D, z); const Dd = (z) => P(x0, y0 + D, z);
    const g = el('g', { class: `tower${on ? ' is-on' : ''}${t.upper ? ' upper' : ''}`, 'data-layer': layer }, svg);
    const e = 0.22;
    el('polygon', { class: 'plinth', points: pts([P(x0 - e, y0 - e, 0), P(x0 + W + e, y0 - e, 0), P(x0 + W + e, y0 + D + e, 0), P(x0 - e, y0 + D + e, 0)]) }, g);

    if (t.na) {
      el('polygon', { class: 'ghost', points: pts([A(0), B(0), C(0), Dd(0)]) }, g);
      el('polygon', { class: 'ghost', points: pts([Dd(0), C(0), C(14), Dd(14)]) }, g);
      el('polygon', { class: 'ghost', points: pts([C(0), B(0), B(14), C(14)]) }, g);
      el('polygon', { class: 'ghost', points: pts([A(14), B(14), C(14), Dd(14)]) }, g);
      g.addEventListener('click', () => selectLayer(layer));
      return;
    }
    const h = Math.max(3, (H_MAX * t.ms) / maxMs) * state.grow;
    el('polygon', { class: 'face', points: pts([Dd(0), C(0), C(h), Dd(h)]) }, g);
    el('polygon', { class: 'face', points: pts([C(0), B(0), B(h), C(h)]) }, g);
    el('polygon', { class: 'face top', points: pts([A(h), B(h), C(h), Dd(h)]) }, g);

    if (on && t.floors.length) {
      const total = t.floors.reduce((a, f) => a + f.ms, 0) || 1;
      let z = 0;
      t.floors.forEach((f, i) => {
        const z1 = z + (h * f.ms) / total;
        if (i > 0 && h > 12) el('polyline', { class: 'floor-line', points: pts([Dd(z), C(z), B(z)]) }, g);
        floorGeo.push({ id: f.id, z0: z, z1, B, C, Dd, h });
        z = z1;
      });
      floorGeo.forEach((fg) => {
        const hit = el('polygon', {
          class: `floor-hit${fg.id === state.floor ? ' is-sel' : ''}`,
          points: pts([fg.Dd(fg.z0), fg.C(fg.z0), fg.B(fg.z0), fg.B(fg.z1), fg.C(fg.z1), fg.Dd(fg.z1)]),
          'data-floor': fg.id,
        }, g);
        hit.addEventListener('mouseenter', () => setHover(fg.id));
        hit.addEventListener('mouseleave', () => setHover(null));
        hit.addEventListener('click', (ev) => { ev.stopPropagation(); selectFloor(fg.id, true); });
      });
      // 顶部两角的括号，呼应选中
      const tl = Dd(h); const tr = B(h);
      el('path', { class: 'bracket', d: `M${tl[0] - 10},${tl[1] - 2} v-8 h8` }, g);
      el('path', { class: 'bracket', d: `M${tr[0] + 10},${tr[1] - 2} v-8 h-8` }, g);
    }
    if (!on) g.addEventListener('click', () => selectLayer(layer));
  }

  function drawTag(r, layer, n) {
    const t = r.towers[layer];
    const [x0, y0] = SITES[layer];
    const c = P(x0 + W, y0 + D, 0);
    const on = state.layer === layer;
    const share = t.na ? null : (t.ms / r.totalMs) * 100;
    const label = t.na ? `${NAME[layer]} · 未采集` : `${NAME[layer]} · ${t.upper ? '≤ ' : ''}${fmtPct(share)}`;
    const g = el('g', { class: `tag${on ? ' is-on' : ''}`, tabindex: 0, role: 'button', 'aria-pressed': on, 'aria-label': `${NAME[layer]}：${t.na ? '未采集' : fmtMs(t.ms)}` }, svg);
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
    if (!fg) return;
    const zm = (fg.z0 + fg.z1) / 2;
    const p = fg.B(zm);
    const unit = unitOf(run(), state.layer);
    el('path', { class: 'bracket', d: `M${p[0] + 6},${p[1]} h14` }, g);
    const t = el('text', { class: 'floor-tag', x: p[0] + 26, y: p[1] + 4 }, g);
    t.textContent = `${id} ${unit.en}`;
  }

  /* ------------------------------------------------------------ text + floors */
  function renderText() {
    const r = run(); const t = tower();
    const isHot = state.layer === r.bottleneck;
    const idx = LAYERS.indexOf(state.layer) + 1;
    $('#eyebrow').innerHTML = t.na ? `层 ${idx} / 3 · 未采集` : isHot ? '<span class="hot-dot"></span>瓶颈层' : `层 ${idx} / 3`;
    $('#layerName').textContent = NAME[state.layer];
    const unit = unitOf(r, state.layer);
    $('#layerMeta').innerHTML = t.na
      ? '<span>这次执行没有这一层的数据</span>'
      : `<span><b>${t.upper ? '≤ ' : ''}${fmtPct((t.ms / r.totalMs) * 100)}</b> 关键路径${t.upper ? '（上界）' : ''}${t.derived ? '（推导）' : ''}</span><span>${t.upper ? '≤ ' : ''}${fmtMs(t.ms)} · ${t.floors.length} ${unit.cn}</span>`;
    $('#headline').innerHTML = isHot ? `${esc(r.headline)}<br><span style="color:var(--foreground-muted)">→ ${esc(r.whatIf)}</span>` : esc(t.caption);
    $('#runSummary').innerHTML = `<b>${esc(r.title)}</b><span>E2E ${fmtMs(r.totalMs)} · ${esc(r.sub)}</span>`;
    const link = $('#analysisLink');
    if (r.analysis) { link.href = `${r.analysis}&layer=${state.layer}`; link.removeAttribute('aria-disabled'); } else { link.setAttribute('aria-disabled', 'true'); link.removeAttribute('href'); }
    $('#notes').innerHTML = `<ul>${r.notes.map((n) => `<li>${esc(n)}</li>`).join('')}</ul><div class="src">${esc(r.source)}</div>`;
  }

  function renderFloors() {
    const r = run(); const t = tower();
    const unit = unitOf(r, state.layer);
    $('#floorsTitle').textContent = `${NAME[state.layer]} · ${unit.each}`;
    $('#floorsCaption').textContent = t.caption;
    const list = $('#floorList');
    if (t.na || !t.floors.length) {
      list.innerHTML = `<div class="empty"><strong>没有 ${NAME[state.layer]} 层数据</strong><span>点击左侧 Host 或 Device 查看。</span></div>`;
      return;
    }
    list.innerHTML = t.floors.slice().reverse().map((f) => `
      <div class="floor${f.id === state.floor ? ' is-sel' : ''}" data-floor="${esc(f.id)}">
        <span class="floor-id">${esc(f.id)}</span>
        ${f.cards.map((c) => `
          <button class="card ${c.tone}" type="button" data-floor="${esc(f.id)}">
            <span class="ct">${esc(c.title)}</span>
            <span class="cs">${esc(c.sub)}</span>
            <span class="cv">${esc(c.text || fmtMs(c.ms))}</span>
          </button>`).join('')}
      </div>`).join('');
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
  function setHover(id) {
    state.hover = id;
    updateFloorLabel();
    document.querySelectorAll('.floor').forEach((n) => n.classList.toggle('is-hover', n.dataset.floor === id));
  }
  function selectRun(id) {
    state.run = id; state.layer = run().bottleneck; state.floor = null; state.hover = null;
    closeMenus();
    grow();
  }
  function grow() {
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce || document.hidden) { state.grow = 1; render(); return; }
    const t0 = performance.now();
    const token = (grow.token = (grow.token || 0) + 1);
    renderText(); renderFloors(); writeHash();
    const tick = (now) => {
      if (token !== grow.token) return;
      const k = Math.min(1, (now - t0) / 560);
      state.grow = 1 - (1 - k) ** 3;
      renderScene();
      if (k < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    // rAF 在后台标签页会暂停：兜底直接落到最终状态
    setTimeout(() => { if (token === grow.token && state.grow < 1) { state.grow = 1; renderScene(); } }, 700);
  }

  // 右侧卡片 ↔ 场景楼层联动
  const floorList = $('#floorList');
  floorList.addEventListener('mouseover', (e) => { const f = e.target.closest('.floor'); if (f && f.dataset.floor !== state.hover) setHover(f.dataset.floor); });
  floorList.addEventListener('mouseleave', () => setHover(null));
  floorList.addEventListener('click', (e) => { const c = e.target.closest('.card'); if (c) selectFloor(c.dataset.floor, false); });
  floorList.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
    const ids = tower().floors.map((f) => f.id);
    if (!ids.length) return;
    e.preventDefault();
    const i = ids.indexOf(state.floor);
    const next = i < 0 ? ids[ids.length - 1] : ids[Math.max(0, Math.min(ids.length - 1, i + (e.key === 'ArrowUp' ? 1 : -1)))];
    state.floor = null;
    selectFloor(next, true);
    const btn = document.querySelector(`.floor[data-floor="${CSS.escape(next)}"] .card`);
    if (btn) btn.focus({ preventScroll: true });
  });

  // 执行切换菜单与口径说明
  const runButton = $('#runButton'); const runMenu = $('#runMenu');
  const notesButton = $('#notesButton'); const notes = $('#notes');
  function closeMenus() {
    runMenu.hidden = true; runButton.setAttribute('aria-expanded', 'false');
    notes.hidden = true; notesButton.setAttribute('aria-expanded', 'false');
  }
  runButton.addEventListener('click', (e) => {
    e.stopPropagation();
    const open = runMenu.hidden;
    closeMenus();
    if (!open) return;
    runMenu.innerHTML = RUNS.map((r) => `
      <li><button type="button" role="option" aria-selected="${r.id === state.run}" data-run="${r.id}">
        <span class="k">${r.kind === 'model' ? '模型' : '算子'}</span>
        <span class="t">${esc(r.title)}</span>
        <span class="b">${NAME[r.bottleneck]}</span>
        <span class="s">${esc(r.sub)}</span>
      </button></li>`).join('');
    runMenu.hidden = false; runButton.setAttribute('aria-expanded', 'true');
    const cur = runMenu.querySelector('[aria-selected="true"]');
    if (cur) cur.focus();
  });
  runMenu.addEventListener('click', (e) => { const b = e.target.closest('[data-run]'); if (b) selectRun(b.dataset.run); });
  notesButton.addEventListener('click', (e) => {
    e.stopPropagation();
    const open = notes.hidden;
    closeMenus();
    if (open) { notes.hidden = false; notesButton.setAttribute('aria-expanded', 'true'); }
  });
  document.addEventListener('click', (e) => { if (!e.target.closest('.run-picker, .notes')) closeMenus(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { closeMenus(); runButton.focus(); } });

  let resizeTimer = 0;
  window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(renderScene, 120); });

  readHash();
  if (!state.layer) state.layer = run().bottleneck;
  if (state.floor && !tower().floors.some((f) => f.id === state.floor)) state.floor = null;
  grow();
  scrollToFloor(false);
}());
