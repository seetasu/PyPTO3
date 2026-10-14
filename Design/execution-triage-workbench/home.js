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
  // 每条带里的行代表什么：Host 的 lane 是 host 侧线程（每卡一条），不是 Device 本身
  const CAPTION = { serving: '调度器', host: '每卡一条 host lane', device: '每卡 runner_run' };
  const NS = 'http://www.w3.org/2000/svg';
  const $ = (s) => document.querySelector(s);
  const r2 = (v) => Math.round(v * 100) / 100;
  const esc = (v) => String(v).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

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
  const PAD = { l: 64, r: 170, t: 52, b: 30 };
  let W = 1000;        // 时间轴长度：按容器比例逐次计算，让泳道左右撑满而不是居中留白
  const ROW_H = 11;    // 每条 lane 的行高
  const BAR = 8;       // lane 条高
  const HEAD_H = 16;   // Serving / Worker 行高（贴合 11 的条高）
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
  /* 带内留白一律是 PAD_IN，上下对称。Host 带只有单步视图才有 Worker 行，
   * 全程视图不留它的位置，否则 lane 组会被顶到底边、顶部空一块。 */
  function layout(nLane, withWorker) {
    const headRoom = withWorker ? HEAD_H + SUB_GAP : 0;
    const serving = { y: PAD.t, h: HEAD_H + PAD_IN * 2 };
    const host = { y: serving.y + serving.h + GAP, h: headRoom + nLane * ROW_H + PAD_IN * 2, lanes0: PAD_IN + headRoom };
    const device = { y: host.y + host.h + GAP, h: nLane * ROW_H + PAD_IN * 2, lanes0: PAD_IN };
    return { serving, host, device, total: device.y + device.h, withWorker };
  }
  function rows(L) {
    return {
      serving: L.serving.y + L.serving.h / 2,
      worker: L.host.y + PAD_IN + HEAD_H / 2,
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

  /* ------------------------------------------------------------ state */
  const stepIds = RUN.request.timeline.map((s) => s.id);
  const TABS = ['path', 'dist', 'plan', 'steps'];
  const VIEWS = ['e2e', 'step'];
  const state = { layer: RUN.bottleneck, step: RUN.defaultStep, picked: false, tab: 'path', view: 'e2e', lane: null, hover: null };
  const tower = () => RUN.towers[state.layer];

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
    const n = el('rect', { class: `blk ${cls}`, x: x0, y: yc - h / 2, width: w, height: h, rx: 1.5 }, g);
    if (tipText) withTip(n, tipText);
    return n;
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

  function drawBands(L, preTo) {
    const x1 = PAD.l + W;
    LAYERS.forEach((layer, i) => {
      const b = L[layer];
      const on = state.layer === layer;
      const g = el('g', { class: `plate${on ? ' is-on' : ''}`, 'data-layer': layer }, svg);
      const mainFrom = preTo == null ? PAD.l : preTo;
      el('rect', { class: 'plate-main', x: mainFrom, y: b.y, width: x1 - mainFrom, height: b.h }, g);
      if (preTo != null) el('rect', { class: 'plate-pre', x: PAD.l, y: b.y, width: preTo - PAD.l, height: b.h }, g);
      el('rect', { class: 'plate-edge', x: PAD.l, y: b.y, width: W, height: b.h, rx: 3 }, g);
      // 带名行：[编号] 名称 · 说明，压在带的上沿之上
      const ty = b.y - 8;
      const tag = el('g', { class: `tag${on ? ' is-on' : ''}`, tabindex: 0, role: 'button', 'aria-pressed': on, 'aria-label': NAME[layer] }, g);
      el('rect', { class: 'tag-box', x: PAD.l, y: ty - 13, width: 18, height: 18, rx: 3 }, tag);
      el('text', { class: 'tag-num', x: PAD.l + 9, y: ty, 'text-anchor': 'middle' }, tag).textContent = i + 1;
      el('text', { class: 'tag-label', x: PAD.l + 26, y: ty, 'text-anchor': 'start' }, tag).textContent = NAME[layer];
      // 带内这些行是什么：Host 的 16 行是 host 侧每卡一条线程，Device 的 16 行是同编号的卡
      el('text', { class: 'tag-cap', x: PAD.l + 92, y: ty, 'text-anchor': 'start' }, g).textContent = CAPTION[layer];
      g.addEventListener('click', () => selectLayer(layer));
      tag.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectLayer(layer); } });
    });
  }

  function renderScene() {
    const f = RUN.flows[state.step];
    const nLane = f.lanes.length;
    const L = layout(nLane, true);
    const ROW = rows(L);
    W = fitW(L.total + PAD.b, PAD.r);
    const VW = PAD.l + W + PAD.r;
    svg.innerHTML = '';
    svg.setAttribute('viewBox', `0 0 ${VW} ${L.total + PAD.b}`);
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    const X = (t) => PAD.l + ((t - f.t0) / (f.t1 - f.t0)) * W;
    const x1 = PAD.l + W;

    const defs = el('defs', {}, svg);
    const hatch = el('pattern', { id: 'rrHatch', width: 5, height: 5, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, defs);
    el('rect', { width: 1.2, height: 5, fill: 'rgba(255,255,255,0.3)' }, hatch);
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

    // 3) 时间参考线：本步起点、关键 lane 进入 runner_run、同步屏障
    const guides = el('g', { class: 'guides' }, svg);
    vline(guides, X(0), L.serving.y - 6, L.device.y + L.device.h + 6, 'g-start');
    vline(guides, X(f.critStart), L.host.y, L.device.y + L.device.h + 6, 'g-crit');

    // 4) Serving 带
    const gs = el('g', {}, svg);
    f.serving.forEach((s) => {
      if (s.kind === 'wait') { hline(gs, X(s.s), X(s.s + s.d), ROW.serving, 'wait'); return; }
      bar(gs, X(s.s), X(s.s + s.d), ROW.serving, 11, s.kind, `Serving · ${s.n} ${fmtMs(s.d)}`);
    });

    // 5) Host 带：Worker 一行 + 16 条 lane 的 bind
    const gh = el('g', {}, svg);
    f.worker.forEach((w) => {
      if (w.kind === 'wait') { hline(gh, X(w.s), X(w.s + w.d), ROW.worker, 'wait'); return; }
      bar(gh, X(w.s), X(w.s + w.d), ROW.worker, 11, w.kind, `Worker · ${w.n} ${fmtMs(w.d)}${w.kind === 'next' ? '（下一步）' : ''}`);
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
      pairHover(bar(gd, X(l.rrStart), X(l.end), y, BAR, `rr${crit ? ' crit' : ''}`, `dev ${l.dev} · runner_run ${fmtMs(l.end - l.rrStart)}（含等屏障）`), l, ROW, X);
    });
    vline(gd, X(f.end), L.device.y, L.device.y + L.device.h, 'barrier');

    // 6.5) Worker 行名
    const gw = el('g', {}, svg);
    el('text', { class: 'lane-tick', x: PAD.l - 8, y: ROW.worker + 3, 'text-anchor': 'end' }, gw).textContent = 'Worker';

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

    // 8) 非关键流转（灰）：调度下发 → Worker → 第一条 lane；屏障 → 回收 → 调度输出
    const gf = el('g', { class: 'flow' }, svg);
    const K = f.links;
    const poly = (list, marker) => el('polyline', { class: 'flow-line', points: pts(list), 'marker-end': marker ? `url(#${marker})` : '' }, gf);
    if (K.queue != null && K.dispatch) poly([[X(K.queue), ROW.serving + 6], [X(K.dispatch[0]), ROW.worker - 6]], 'arrowGrey');
    if (K.dispatch) poly([[X(K.dispatch[1]), ROW.worker + 6], [X(f.laneStart), ROW.host(0) - 4]], 'arrowGrey');
    if (K.reclaimEnd != null && K.output != null) {
      poly([[X(f.end), ROW.device(0) - 4], [X(K.reclaimEnd), ROW.worker + 6], [X(K.output), ROW.serving + 6]], 'arrowGrey');
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
      el('text', { class: `plate-note${dim ? ' dim' : ''}`, x: nx, y, 'text-anchor': 'start' }, svg).textContent = text;
    };
    const hy = ROW.host(Math.round(nLane / 2));
    note(hy - 10, `${nLane} 条 lane 每步重新 bind`);
    note(hy + 10, `本步合计 ${fmtMs(argsSum)}`, true);
    note(ROW.device(Math.round(nLane / 2)), '末端对齐 = 同步屏障', true);

    // 11) 时间轴
    const ga = el('g', { class: 'axis' }, svg);
    const yA = L.device.y + L.device.h + 10;
    const ticks = [0, f.critStart, f.end];
    for (let t = Math.ceil(f.t0 / 100) * 100; t <= f.t1; t += 100) if (t !== 0) ticks.push(t);
    ticks.forEach((t) => {
      const strong = t === 0 || t === f.critStart || t === f.end;
      vline(ga, X(t), yA - 4, yA, strong ? 'tick strong' : 'tick');
      if (!strong && [f.critStart, f.end].some((s) => Math.abs(X(s) - X(t)) < 22)) return;
      const tx = el('text', { class: `tick-label${strong ? ' strong' : ''}`, x: X(t), y: yA + 13, 'text-anchor': 'middle' }, ga);
      tx.textContent = `${Math.round(t)}`;
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
    const L = layout(nLane, false);
    const ROW = rows(L);
    W = fitW(L.total + PAD.b, PAD_R_E2E);
    const VW = PAD.l + W + PAD_R_E2E;
    svg.innerHTML = '';
    svg.setAttribute('viewBox', `0 0 ${VW} ${L.total + PAD.b}`);
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    const X = (t) => PAD.l + (t / E.durMs) * W;
    const x1 = PAD.l + W;
    const hot = RUN.dist.critLane;

    const defs = el('defs', {}, svg);
    const hatch = el('pattern', { id: 'rrHatch', width: 5, height: 5, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, defs);
    el('rect', { width: 1.2, height: 5, fill: 'rgba(255,255,255,0.3)' }, hatch);

    // 1) 三条带（带名在每条带上方单行）
    drawBands(L, null);

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
        const db = bar(gd, X(l.r0), X(l.r1), ROW.device(l.dev), BAR, `rr${crit ? ' crit' : ''}`);
        db.setAttribute('data-s', s.id); db.setAttribute('data-l', l.dev);
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

    // 7) 慢步：周期超过 decode P90 的，整列压一层淡红
    const top = L.serving.y - 6;
    const bot = L.device.y + L.device.h + 6;
    const p90 = RUN.perStep.p90Ms;
    const slow = el('g', {}, svg);
    let nSlow = 0;
    E.steps.forEach((s) => {
      const b = RUN.runtime.byStep[s.id];
      if (!b || !b.decode || b.period <= p90) return;
      nSlow += 1;
      el('rect', {
        class: 'slow-step', x: X(s.t0), y: top,
        width: Math.max(1.5, X(s.t1) - X(s.t0)), height: bot - top,
      }, slow);
    });
    if (nSlow) {
      // 与左侧的冷启动注记分列两端
      el('text', { class: 'plate-note', x: PAD.l + W, y: PAD.t - 30, 'text-anchor': 'end' }, svg)
        .textContent = `${nSlow} 步超过 decode P90 ${fmtMs(p90)}`;
    }

    // 8) 每个 step 一块透明热区：悬停给时长，点击下钻到单步视图
    const gz = el('g', { class: 'hits' }, svg);
    E.steps.forEach((s) => {
      const r = el('rect', {
        class: `hit${state.picked && s.id === state.step ? ' is-sel' : ''}`, x: X(s.t0), y: top,
        width: Math.max(1.5, X(s.t1) - X(s.t0)), height: bot - top, 'data-step': s.id,
      }, gz);
      const bs = RUN.runtime.byStep[s.id];
      const slowTag = bs && bs.decode && bs.period > p90 ? ' · 超 P90' : '';
      const ctx = { X, ROW };
      const tipFor = (dev) => {
        const l = s.lanes.find((x) => x.dev === (dev == null ? s.crit : dev));
        const head = `${s.id} · ${fmtMs(bs ? bs.period : s.t1 - s.t0)}${slowTag}`;
        const chain = `lane ${l.dev} bind ${fmtMs(l.b1 - l.b0)} → dev ${l.dev} runner_run ${fmtMs(l.r1 - l.r0)}`;
        const tail = l.dev === s.crit
          ? `最晚进入 runner_run，这一步由它决定；其余 ${s.lanes.length - 1} 条在屏障等它`
          : `比关键 lane ${s.crit} 早 ${fmtMs(s.lanes.find((x) => x.dev === s.crit).r0 - l.r0)} 进入，这段时间在等屏障`;
        return `${head}
${chain}
${tail}（点击下钻）`;
      };
      r.addEventListener('mousemove', (e) => {
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
      vline(svg, xEnd, top, bot, 'g-start');
      // 带名行已经占了带的上沿，冷启动注记再往上一行
      el('text', { class: 'plate-note dim', x: xEnd + 6, y: PAD.t - 30, 'text-anchor': 'start' }, svg)
        .textContent = `冷启动 ${fmtMs(pre.reduce((a, s) => a + (s.t1 - s.t0), 0))}`;
    }

    // 10) 时间轴（秒）
    const ga = el('g', { class: 'axis' }, svg);
    const yA = bot + 4;
    for (let t = 0; t <= E.durMs; t += 1000) {
      vline(ga, X(t), yA - 4, yA, 'tick');
      el('text', { class: 'tick-label', x: X(t), y: yA + 13, 'text-anchor': 'middle' }, ga).textContent = `${Math.round(t / 1000)}`;
    }
    el('text', { class: 'tick-label', x: x1 + 8, y: yA + 13, 'text-anchor': 'start' }, ga).textContent = 's';

    el('text', { class: 'step-title', x: x1, y: PAD.t - 10, 'text-anchor': 'end' }, svg)
      .textContent = `全程 · ${fmtMs(E.durMs)} · ${E.steps.length} 步`;
  }

  // 泳道按当前范围渲染：全程 or 单步
  function drawLanes() {
    if (state.view === 'e2e') renderE2E();
    else renderScene();
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
      svg.querySelectorAll('.row-band, .chain').forEach((n) => n.remove());
      svg.querySelectorAll(`[data-l="${l.dev}"]`).forEach((n) => n.classList.add('is-row'));
      [ROW.host(l.dev), ROW.device(l.dev)].forEach((y) => {
        el('rect', { class: 'row-band', x: PAD.l, y: y - ROW_H / 2, width: W, height: ROW_H }, svg);
      });
      el('polyline', { class: 'chain', points: pts([[X(l.bindEnd), ROW.host(l.dev)], [X(l.rrStart), ROW.device(l.dev)]]) }, svg);
    });
    node.addEventListener('mouseleave', () => {
      svg.querySelectorAll('.is-row').forEach((n) => n.classList.remove('is-row'));
      svg.querySelectorAll('.row-band, .chain').forEach((n) => n.remove());
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
    svg.querySelectorAll('.chain, .row-band').forEach((n) => n.remove());
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
    // 交叉处：把这条 lane 的 bind → runner_run 连起来（没悬在某一行时用关键 lane）
    const l = s.lanes.find((x) => x.dev === (dev == null ? s.crit : dev));
    if (l) {
      el('polyline', {
        class: 'chain',
        points: pts([[ctx.X(l.b1), ctx.ROW.host(l.dev)], [ctx.X(l.r0), ctx.ROW.device(l.dev)]]),
      }, svg);
    }
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
    $('#layerName').innerHTML = `${(q.e2eMs / 1000).toFixed(2)}<small>s</small>`;
    $('#layerName').setAttribute('aria-label', `端到端 ${fmtMs(q.e2eMs)}`);
    $('#layerMeta').innerHTML = `<span>E2E · ${q.outputs} 次输出 · 非流式 max_tokens ${q.maxTokens}</span>`;
    $('#reqKpis').innerHTML = `
      <div class="kpi"><span>首次输出</span><b>${fmtMs(q.firstOutputMs)}</b></div>
      <div class="kpi"><span>输出间隔 P50</span><b>${fmtMs(q.gapP50Ms)}</b></div>
      <div class="kpi"><span>P90</span><b>${fmtMs(q.gapP90Ms)}</b></div>
      <div class="kpi"><span>冷启动</span><b>${fmtMs(RUN.coldStart.prefillMs + RUN.coldStart.draftMs)}</b></div>`;
  }

  /* ② 关键路径：每个 decode step 的时间去哪了 */
  /* 左上角始终描述泳道正在展示的范围：全程给 decode 52 步的均值，单步给那一步。
   * 三行相加等于对应的周期（Host 含 Worker 下发）。 */
  function headScope() {
    const ps = RUN.perStep;
    if (state.view === 'e2e') {
      const R = RUN.runtime;
      const L = ps.layers;
      return {
        title: `decode ${R.n} 步`,
        note: `平均 ${fmtMs(ps.meanMs)}（P50 ${fmtMs(ps.p50Ms)} · P90 ${fmtMs(ps.p90Ms)}）`,
        serving: L.serving.meanMs,
        host: L.host.meanMs,
        device: L.device.meanMs,
        period: ps.meanMs,
        h2d: R.path.find((p) => p.key === 'h2d').stat.mean,
        allocation: R.bindSplit.allocation.mean,
        transfer: R.bindSplit.transfer.mean,
        resident: R.resident.period.mean,
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
      allocation: b.allocation,
      transfer: b.transfer,
      resident: b.resident,
    };
  }

  function renderPathHead() {
    const v = headScope();
    $('#eyebrow2').innerHTML = `<b>${esc(v.title)}</b> <span class="dim">· ${v.note} · 关键路径构成</span>`;
    const max = Math.max(...LAYERS.map((l) => v[l]));
    $('#stepMix').innerHTML = LAYERS.map((l, i) => {
      const t = RUN.towers[l]; const share = (v[l] / v.period) * 100;
      const on = state.layer === l;
      return `
        <button class="mix-row${on ? ' is-on' : ''}" type="button" data-mix="${l}" aria-pressed="${on}">
          <span class="mix-num">${i + 1}</span>
          <span class="mix-name">${NAME[l]}</span>
          <span class="mix-bar"><span class="mix-fill${t.upper ? ' upper' : ''}" style="width:${Math.max(0.6, (v[l] / max) * 100)}%"></span></span>
          <span class="mix-val">${t.upper ? '≤ ' : ''}${fmtMs(v[l])}</span>
          <span class="mix-pct">${share < 1 ? '≈ 0%' : `${t.upper ? '≤ ' : ''}${Math.round(share)}%`}</span>
        </button>${l === 'host' ? subRow(v) : ''}`;
    }).join('') + residentRow(v);
  }



  // Host 里几乎全是关键 lane 的 bind.args（分配 + H2D），#95 把它判为可消除的 P0
  function subRow(b) {
    return `
      <div class="bind-breakdown">
        <div class="mix-sub">
          <span class="mix-sub-name">└ bind.args</span>
          <span class="mix-val">${fmtMs(b.h2d)}</span>
          <span class="chip hot">可消除</span>
        </div>
        <div class="bind-components">
          <span>分配估计 <b>${fmtMs(b.allocation)}</b></span>
          <span>H2D 估计 <b>${fmtMs(b.transfer)}</b></span>
        </div>
      </div>`;
  }
  // 常驻推算：去掉关键路径上的 bind.args，其余阶段不变
  function residentRow(b) {
    return `
      <div class="resident">
        <span class="resident-k">权重 / Cache 常驻后</span>
        <span class="mix-bar"><span class="mix-fill ghost" style="width:${(b.resident / b.period) * 100}%"></span></span>
        <span class="mix-val">≈ ${fmtMs(b.resident)}</span>
        <span class="chip">推算</span>
      </div>`;
  }

  /* 主导航：54 个 step 的时间线，宽度 = 耗时，点任一段即在泳道里展开 */
  // 面包屑：全程是父层，某一步是子层；父层常驻且可点返回
  function crumb() {
    if (state.view === 'e2e') {
      return `<b class="crumb-cur">全程</b>
        <span class="nav-meta">${fmtMs(RUN.e2e.durMs)} · ${RUN.e2e.steps.length} 步</span>`;
    }
    const t = RUN.request.timeline.find((x) => x.id === state.step);
    const f = RUN.flows[state.step];
    return `<button class="crumb-up" type="button" data-view="e2e">全程</button>
      <span class="crumb-sep" aria-hidden="true">›</span>
      <b class="crumb-cur">${esc(state.step)}</b>
      <span class="nav-meta">${fmtMs(t ? t.ms : f.periodMs)} · 关键 lane ${f.crit}</span>`;
  }
  function renderRhythm() {
    const q = RUN.request;
    const segs = q.timeline.map((s) => `<span class="tl-seg k-${s.kind}${state.picked && s.id === state.step ? ' is-sel' : ''}" data-step="${esc(s.id)}" style="flex:${s.ms} 0 0" title="${esc(s.id)} · ${fmtMs(s.ms)}"></span>`).join('')
      + `<span class="tl-seg k-tail" style="flex:${q.tailMs} 0 0"></span>`;
    const first = (q.firstOutputMs / q.e2eMs) * 100;
    $('#request').innerHTML = `
      <nav class="crumb" id="navSel" aria-label="泳道范围">${crumb()}</nav>
      <div class="req-line" role="group" aria-label="请求时间线">
        <div class="tl">${segs}</div>
        <span class="tl-mark" style="left:${first}%"><i></i>首次输出</span>
      </div>`;
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
    <tr class="${cls || ''}${r.hot ? ' hot' : ''}" title="${esc(`${r.label} · 均值 ${fmtMs(r.stat.mean)} · P50 ${fmtMs(r.stat.p50)} · P90 ${fmtMs(r.stat.p90)} · 最大 ${fmtMs(r.stat.max)}`)}">
      <th scope="row"><b>${esc(r.label)}</b>${r.sub ? `<i>${esc(r.sub)}</i>` : ''}</th>
      <td>${r.upper ? '≤ ' : ''}${fmtMs(r.stat.mean)}</td>
      <td class="spread">${fmtRange(r.stat.p50, r.stat.p90)}</td>
    </tr>`;
  const ledgerHead = (title) => `<thead><tr><th scope="col">${title}</th><th scope="col">均值</th><th scope="col">P50–P90</th></tr></thead>`;

  /* ② 关键路径：只给还需要往下走的结论 + 可相加的五段；观察窗口折叠
   * 查完没问题的那几层不展开——它们不是用户要处理的事，只留一行作为查过的记录。 */
  const Q_CHIP = { hit: '命中', open: '待分清' };
  function renderPathPanel() {
    const R = RUN.runtime;
    const live = RUN.questions.filter((q) => q.state !== 'clear');
    const clear = RUN.questions.filter((q) => q.state === 'clear');
    const items = live.map((q) => `
        <li class="vd is-${q.state}">
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
        <h3>定界结论<span>只看落在关键路径的那部分</span></h3>
        <ul class="verdicts">${items}</ul>
        ${clear.length ? `<p class="vd-clear" title="${esc(clear.map((q) => `${q.topic}：${q.a}`).join('\n'))}">已排除 ${clear.map((q) => esc(q.topic)).join(' · ')}</p>` : ''}
      </section>
      <section class="bd-sec">
        <h3>关键路径<span>decode ${R.n} 步 · 五段相加 = 步长</span></h3>
        <table class="ledger">
          ${ledgerHead('阶段')}
          <tbody>${R.path.map((r) => ledgerRow(r)).join('')}</tbody>
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
    // runner_run 再按设备执行的估计拆成「执行」与「等屏障」——执行几乎等长，差别全在起点
    const split = d.deviceSplitSource != null && d.lanes[0].deviceExec != null;
    const rows = d.lanes.map((l) => {
      const crit = l.dev === d.critLane;
      const tip = split
        ? `lane ${l.dev} · bind ${fmtMs(l.bind)} · 进入 runner_run 前 ${fmtMs(l.lead)} · runner_run ${fmtMs(l.rr)}＝执行 ${fmtMs(l.deviceExec)}（估计）＋等待 ${fmtMs(l.waitEstimate)} — 点击查看这条 lane`
        : `lane ${l.dev} · bind ${fmtMs(l.bind)} · 进入 runner_run 前 ${fmtMs(l.lead)} · runner_run ${fmtMs(l.rr)}（含等待 ≥ ${fmtMs(l.wait)}）— 点击查看这条 lane`;
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
        <div class="lane-chart" style="--crit:${pct(critLead)}%">
          ${rows}
          <span class="lane-guide" style="left:calc(50px + (100% - 108px) * ${(critLead / total).toFixed(4)})"></span>
        </div>
        <p class="lane-note">${split
    ? `红线 = 最晚进入 runner_run 的时刻。斜纹段几乎等长（${fmtMs(Math.min(...d.lanes.map((l) => l.deviceExec)))}–${fmtMs(Math.max(...d.lanes.map((l) => l.deviceExec)))}），差别全在它前面；右列是该 lane 的 bind 均值。点一条看它的专项。`
    : '红线 = 最晚进入 runner_run 的时刻，线左侧的斜纹都在等屏障；右列是该 lane 的 bind 均值。点一条看它的专项。'}</p>
        ${fold('口径边界', split ? '执行 / 等待是估计值' : '别把上界当执行时间', `
          <p class="caveat">${esc(d.caveat)}</p>
          ${split ? '<p class="caveat est">执行与等待的拆分是估计：runner_run 的区间来自 trace，切分点是推算的，用来演示读图方式，不能当实测值引用。</p>' : ''}
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
        <span class="nav-meta">bind 第 ${rank} / ${d.nLane}</span>
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
            <tr class="sub"><th scope="row"><b>└ 设备执行</b><i>估计值 · 非实测</i></th>
              <td>${fmtMs(me.deviceExec)}</td><td class="spread">${d.nLane} 条中位 ${fmtMs(medExec)}</td></tr>
            <tr class="sub"><th scope="row"><b>└ 等屏障</b><i>估计值 · runner_run − 执行</i></th>
              <td>${fmtMs(me.waitEstimate)}</td><td class="spread">—</td></tr>` : `
            <tr><th scope="row"><b>屏障等待</b><i>早到白等的时间</i></th>
              <td>${fmtMs(me.wait)}</td><td class="spread">—</td></tr>`}
          </tbody>
        </table>
        <p class="lane-note">${dev === d.critLane
    ? `runner_run 是 ${d.nLane} 条里最短的${medExec != null ? `，执行估计 ${fmtMs(me.deviceExec)} 与中位的 ${fmtMs(medExec)} 也基本一致` : ''}——设备执行没有变慢的证据，慢的在它前面 host 侧的 bind。`
    : `它不是关键 lane，runner_run 偏长的部分${medExec != null ? `有 ${fmtMs(me.waitEstimate)} 是在等屏障` : '多半是在屏障等待'}。`}</p>
      </section>
      <section class="bd-sec">
        <h3>偏斜的形状<span>是单点、是梯度，还是逐渐劣化</span></h3>
        <div class="lane-chart">${bars}</div>
        <svg class="spark" viewBox="-2 -6 ${W2 + 4} ${H2 + 12}" role="img" aria-label="bind 随 step 的变化">
          <polyline class="sp-med" points="${line(F.med)}" />
          <polyline class="sp-me" points="${line(series)}" />
        </svg>
        <div class="seg-list">${segs}</div>
        <p class="lane-note">单点：比第二名还高 ${fmtMs(sorted[0].bind - sorted[1].bind)}，其余 ${d.nLane - 1} 条连续分布。四段倍数 ${Math.min(...ratios).toFixed(2)}–${Math.max(...ratios).toFixed(2)}×，是结构性的而不是逐渐劣化。</p>
      </section>
      <section class="bd-sec">
        <h3>各阶段是不是同一条<span>谁最晚进入 runner_run</span></h3>
        <div class="stage-list">${stages}</div>
        <p class="lane-note">Prefill 的关键 lane 不是它——偏斜跟 decode 这个 callable 绑定的参数有关，不是这条线程或这张卡永远最慢。</p>
      </section>
      ${fold('排查记录与盲区', `已排除 ${F.excluded.length} · 待补采 ${F.missing.length} · 待实验 ${F.experiments.length}`,
    notes('已排除', F.excluded, true) + notes('这份 trace 答不了', F.missing) + notes('需要做实验', F.experiments))}`;
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
        <ul class="checks">${RUN.checks.map((c) => `<li>${esc(c)}</li>`).join('')}</ul>
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
    $('#floorsTitle').textContent = `${NAME[state.layer]} · 每个 step`;
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
    $('#floorList').innerHTML = t.floors.slice().reverse().map((f) => `
      <div class="floor${state.picked && f.id === state.step ? ' is-sel' : ''}" data-floor="${esc(f.id)}">
        <span class="floor-id">${esc(f.id)}</span>
        ${f.cards.map((c) => `
          <button class="cell ${c.tone}" type="button" data-floor="${esc(f.id)}" title="${esc(`${f.id} · ${c.title} · ${c.sub}`)}">${esc(c.text || fmtMs(c.ms))}</button>`).join('')}
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
  function setHover(id) {
    state.hover = id;
    document.querySelectorAll('.floor').forEach((n) => n.classList.toggle('is-hover', n.dataset.floor === id));
    document.querySelectorAll('.tl-seg[data-step]').forEach((n) => n.classList.toggle('is-hover', n.dataset.step === id));
  }

  // 定界结论里的下一步：直接跳到该看的那条 lane
  $('#panel-path').addEventListener('click', (e) => {
    const b = e.target.closest('[data-go-tab]');
    if (!b) return;
    if (b.dataset.goLane != null) selectLane(b.dataset.goLane);
    selectTab(b.dataset.goTab);
  });

  // 设备分布：点 lane 行进专项，点面包屑返回
  $('#panel-dist').addEventListener('click', (e) => {
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
  reqBox.addEventListener('mouseover', (e) => { const s = e.target.closest('.tl-seg[data-step]'); if (s && s.dataset.step !== state.hover) setHover(s.dataset.step); });
  reqBox.addEventListener('mouseleave', () => setHover(null));
  reqBox.addEventListener('click', (e) => {
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
