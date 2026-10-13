/* Execution Triage · 压测总览（DSpark Serving，256 请求）
 * 上图：256 条请求生命线（到达 → 首 token → 逐波输出 → 结束），横轴是压测窗口。
 * 下图：输出波次，柱高 = 这一波有多少请求同时吐出一段。
 * 首 token 批次、输出波段、输出间隔双峰都直接标在图上；点标签即筛选，再点取消。
 * 数据见 build-overview-data.cjs。 */
(function () {
  'use strict';

  const D = window.TRIAGE_OVERVIEW;
  const RUN = D.run;
  const NS = 'http://www.w3.org/2000/svg';
  const $ = (s) => document.querySelector(s);
  const esc = (v) => String(v).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
  const nf = (v) => Number(v).toLocaleString('en-US');

  function fmtMs(ms) {
    const v = Math.abs(ms);
    if (v >= 1000) return `${(ms / 1000).toFixed(2)} s`;
    if (v >= 100) return `${Math.round(ms)} ms`;
    return `${ms.toFixed(1)} ms`;
  }

  /* ------------------------------------------------------------ 布局（正交 2D）
   * x = 压测窗口内的时刻，上图 y = 请求（按首 token 批次分组），下图 y = 这一波的请求数。 */
  const L = 78;               // 左边距：每批的请求数与 TTFT
  const RGT = 18;
  const W = 866;              // 时间轴长度
  const GAP = 4;              // 首 token 批次之间的缝
  const REQ_T = 26;           // 上图顶部
  const WAV_H = 72;           // 下图高度
  const VB_T = -18;           // 顶部留白：分界标签在上图之上，别被 viewBox 切掉
  const VB_W = 984;
  const X = (ms) => L + (ms / RUN.durationMs) * W;

  // 行序：按首 token 时间分批，批内按请求号
  const sorted = [...D.requests].sort((a, b) => (a.batch - b.batch) || (a.first - b.first) || (a.i - b.i));

  /* 画面高度按盒子比例定，避免上下留空白条 */
  function targetH() {
    const boxW = svg.clientWidth || 940;
    const boxH = svg.clientHeight || 470;
    return VB_W * (boxH / Math.max(1, boxW));
  }

  /* 行高按可用高度定：盒子越高，256 行铺得越开。 */
  let PR; let rows = []; let REQ_H; let WAV_T; let BASE; let VB;
  function layoutRows() {
    const fixed = REQ_T + 46 + WAV_H + 32 - VB_T;   // 上图之外固定占掉的高度
    const target = targetH();
    PR = Math.max(0.9, Math.min(2.4, (target - fixed - 6 - 3 * GAP) / RUN.nReq));
    let y = REQ_T + 3; let last = sorted[0].batch;
    rows = sorted.map((q) => {
      if (q.batch !== last) { y += GAP; last = q.batch; }
      const row = { q, y };
      y += PR;
      return row;
    });
    REQ_H = y + 3 - REQ_T;
    WAV_T = REQ_T + REQ_H + 46;       // 两图之间留出波段标注的位置
    BASE = WAV_T + WAV_H;             // 波次柱的基线
    VB = [0, VB_T, VB_W, BASE + 32 - VB_T];
  }

  /* 曲线视图：三条带共用时间轴 */
  const BANDS = [
    {
      k: 'load', title: '在系统内的请求数', unit: '',
      keys: ['gen', 'wait'], names: { gen: '生成中', wait: '排队中' },
      max: () => RUN.nReq, stat: () => `批大小均值 ${D.series.mean.batch}`,
      read: (i) => `生成中 ${D.series.gen[i]} · 排队中 ${D.series.wait[i]}`,
    },
    {
      k: 'rate', title: '输出速率', unit: '段/s', keys: ['rate'], fill: true,
      max: (m) => Math.ceil(m / 500) * 500, stat: () => `均值 ${D.series.mean.rate} 段/s`,
      read: (i) => `${D.series.rate[i]} 段/s`,
    },
    {
      k: 'itlP50', title: '输出间隔 P50', unit: 'ms', keys: ['itlP50'], fill: false,
      max: (m) => Math.ceil(m / 100) * 100, stat: () => `窗口 P50 ${Math.round(D.stats.itl.p50)} ms`,
      read: (i) => (D.series.itlP50[i] == null ? '—' : `${D.series.itlP50[i]} ms`),
    },
    {
      k: 'kv', title: 'KV cache 利用率', unit: '%', keys: ['kv'], fill: true,
      max: () => 100, stat: () => `峰值 ${D.series.peak.kv}%`,
      read: (i) => `${D.series.kv[i]}%`,
    },
  ];

  function el(name, attrs, parent) {
    const n = document.createElementNS(NS, name);
    Object.entries(attrs || {}).forEach(([k, v]) => n.setAttribute(k, v));
    if (parent) parent.appendChild(n);
    return n;
  }
  const pts = (list) => list.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ');
  function line(g, x1, y1, x2, y2, cls) {
    return el('line', { class: cls, x1: x1.toFixed(1), y1: y1.toFixed(1), x2: x2.toFixed(1), y2: y2.toFixed(1) }, g);
  }
  function text(g, x, y, cls, s, anchor) {
    const t = el('text', { class: cls, x: x.toFixed(1), y: y.toFixed(1), 'text-anchor': anchor || 'start' }, g);
    t.textContent = s;
    return t;
  }

  /* ------------------------------------------------------------ state */
  const SELS = ['none', ...D.batches.map((b) => b.id), 'slow', 'fast'];
  const VIEWS = [{ k: 'main', label: '服务 · Rank' }, { k: 'rows', label: '请求生命线' }];
  // rank：选中的 rank，默认整个窗口最慢的那张；scrub：悬停到的那一步（不进 hash）
  const state = { sel: 'none', dist: 'itl', view: 'main', rank: D.ranks.slowest, scrub: null };

  function readHash() {
    const p = new URLSearchParams(location.hash.slice(1));
    if (SELS.includes(p.get('sel'))) state.sel = p.get('sel');
    if (D.dists.some((x) => x.k === p.get('dist'))) state.dist = p.get('dist');
    if (VIEWS.some((v) => v.k === p.get('view'))) state.view = p.get('view');
    const r = Number(p.get('rank'));
    if (p.has('rank') && Number.isInteger(r) && r >= 0 && r < D.ranks.n) state.rank = r;
  }
  function writeHash() {
    const h = `view=${state.view}&sel=${state.sel}&dist=${state.dist}&rank=${state.rank}`;
    if (location.hash.slice(1) !== h) history.replaceState(null, '', `#${h}`);
  }
  function select(id) {
    if (!SELS.includes(id)) return;
    state.sel = state.sel === id ? 'none' : id;
    writeHash();
    renderScene();
    renderDist();
    renderRail();
  }

  /* ------------------------------------------------------------ tooltip */
  const tip = $('#tip');
  function showTip(e, s) { tip.textContent = s; tip.hidden = false; moveTip(e); }
  function moveTip(e) {
    const r = tip.getBoundingClientRect();
    let x = e.clientX + 14; let y = e.clientY + 14;
    if (x + r.width > window.innerWidth - 8) x = e.clientX - r.width - 14;
    if (y + r.height > window.innerHeight - 8) y = e.clientY - r.height - 14;
    tip.style.left = `${Math.max(8, x)}px`; tip.style.top = `${Math.max(8, y)}px`;
  }
  function hideTip() { tip.hidden = true; }
  function withTip(node, s) {
    node.addEventListener('mouseenter', (e) => showTip(e, s));
    node.addEventListener('mousemove', moveTip);
    node.addEventListener('mouseleave', hideTip);
    return node;
  }

  /* ------------------------------------------------------------ 场景 */
  const svg = $('#iso');

  /* 图上文字按屏幕像素定字号：写入 --u = viewBox 单位 / 屏幕像素（见 overview.css）。
   * 夹在 0.6–1.15：特别宽时不缩得过小，窄屏时不把字撑出布局。 */
  function syncTextScale() {
    const w = svg.clientWidth; const h = svg.clientHeight;
    if (!w || !h) return;
    const s = Math.min(w / VB[2], h / VB[3]);
    svg.style.setProperty('--u', Math.min(1.15, Math.max(0.6, 1 / s)).toFixed(3));
  }

  function tagAt(g, x, y, num, label) {
    const t = el('g', { class: 'tag' }, g);
    el('rect', { class: 'tag-box', x, y: y - 15, width: 20, height: 20, rx: 3 }, t);
    text(t, x + 10, y - 1, 'tag-num', num, 'middle');
    text(t, x + 27, y - 1, 'tag-label', label);
  }
  // 图上可点的标签：点一次只看这一组，再点取消
  function pickLabel(g, id, on, build, tipText) {
    const n = el('g', { class: `pick${on ? ' is-on' : ''}`, tabindex: 0, role: 'button', 'aria-pressed': String(on) }, g);
    build(n);
    n.addEventListener('click', () => select(id));
    n.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(id); } });
    if (tipText) withTip(n, tipText);
    return n;
  }

  function renderRows() {
    layoutRows();
    svg.textContent = '';
    svg.setAttribute('viewBox', VB.join(' '));
    syncTextScale();
    svg.setAttribute('preserveAspectRatio', 'xMidYMin meet');
    const selBand = state.sel === 'slow' || state.sel === 'fast' ? state.sel : null;
    const selBatch = D.batches.findIndex((b) => b.id === state.sel);

    el('rect', { class: 'panel', x: L, y: REQ_T, width: W, height: REQ_H }, svg);
    el('rect', { class: 'panel', x: L, y: WAV_T, width: W, height: WAV_H }, svg);
    tagAt(svg, L, REQ_T - 8, 1, `${RUN.nReq} 请求 · 首 token ${D.batches.length} 批`);
    tagAt(svg, L, WAV_T - 8, 2, `${D.waves.length} 波输出`);

    /* 请求生命线 */
    const gr = el('g', {}, svg);
    const byBatch = new Map();
    rows.forEach((row) => {
      const q = row.q;
      let g = byBatch.get(q.batch);
      if (!g) {
        g = el('g', { class: selBatch >= 0 && selBatch !== q.batch ? 'req-dim' : '' }, gr);
        byBatch.set(q.batch, g);
      }
      if (q.first > q.st) line(g, X(q.st), row.y, X(q.first), row.y, 'req-wait');
      if (q.mid > q.first) line(g, X(q.first), row.y, X(q.mid), row.y, 'req-slow');
      if (q.last > q.mid) line(g, X(q.mid), row.y, X(q.last), row.y, 'req-fast');
    });

    /* 两条边缘线：首 token 的阶梯、结束时刻的散尾 */
    el('polyline', { class: 'edge-first', points: pts(rows.map((r) => [X(r.q.first), r.y])) }, svg);
    el('polyline', { class: 'edge-last', points: pts(rows.map((r) => [X(r.q.last), r.y])) }, svg);

    /* 选中某个波段时，另一段压暗 */
    if (selBand) {
      const a = selBand === 'slow' ? X(D.split) : X(0);
      const b = selBand === 'slow' ? X(RUN.durationMs) : X(D.split);
      el('rect', { class: 'dim-cover', x: a, y: REQ_T, width: b - a, height: REQ_H }, svg);
    }

    /* 每批：请求数 + 占比 + 首 token */
    const gb = el('g', {}, svg);
    D.batches.forEach((b, k) => {
      const list = rows.filter((r) => r.q.batch === k);
      const top = list[0].y; const bot = list[list.length - 1].y;
      const mid = (top + bot) / 2;
      const tall = bot - top > 26;
      const on = state.sel === b.id;
      pickLabel(gb, b.id, on, (n) => {
        el('rect', { class: 'pick-hit', x: 0, y: top - 8, width: L - 4, height: Math.max(16, bot - top + 16) }, n);
        text(n, L - 12, mid - 2, 'grp-n', `${b.n}`, 'end');
        text(n, L - 12, mid + 11, 'grp-sub', fmtMs(b.ttftMs), 'end');
        if (tall) line(n, L - 7, top, L - 7, bot, 'grp-bar');
      }, `${b.n} 个请求（${(b.share * 100).toFixed(0)}%）· 首 token ${fmtMs(b.ttftMs)}`);
    });

    /* 波次柱 */
    const gw = el('g', {}, svg);
    D.waves.forEach((w, k) => {
      const h = Math.max(1.5, (w.n / RUN.nReq) * WAV_H);
      const r = el('rect', {
        class: `wave ${w.band}${selBand && selBand !== w.band ? ' is-dim' : ''}`,
        x: (X(w.t) - 1.6).toFixed(1), y: (BASE - h).toFixed(1), width: 3.2, height: h.toFixed(1),
      }, gw);
      withTip(r, `第 ${k + 1} 波 · ${(w.t / 1000).toFixed(2)} s · ${w.n} 个请求${w.gap === null ? '' : ` · 间隔 ${fmtMs(w.gap)}`}`);
    });
    line(svg, L, BASE, L + W, BASE, 'axis-base');

    /* 波段分界：间隔从 777 ms 掉到 122 ms 的那一刻 */
    const sx = X(D.split);
    line(svg, sx, REQ_T - 4, sx, BASE + 4, 'split-line');
    text(svg, sx - 6, REQ_T - 24, 'split-label', `${(D.split / 1000).toFixed(2)} s · ${Math.round(D.bands[0].gapP50)} → ${Math.round(D.bands[1].gapP50)} ms`, 'end');

    /* 每段：波数 / 间隔 / 覆盖窗口 */
    const gbd = el('g', {}, svg);
    D.bands.forEach((b) => {
      const x0 = X(b.t0); const x1 = X(b.t1);
      const mid = (x0 + x1) / 2;
      const on = state.sel === b.id;
      const hot = b.id === 'slow' ? ' hot' : '';
      pickLabel(gbd, b.id, on, (n) => {
        el('rect', { class: 'pick-hit', x: x0, y: WAV_T - 42, width: x1 - x0, height: 40 }, n);
        line(n, x0 + 1, WAV_T - 7, x1 - 1, WAV_T - 7, `band-bar${hot}`);
        text(n, mid, WAV_T - 28, `band-label${hot}`, `${b.nWave} 波 · 间隔 ${Math.round(b.gapP50)} ms`, 'middle');
        text(n, mid, WAV_T - 15, 'band-sub', `${fmtMs(b.spanMs)} · 占窗口 ${(b.spanMs / RUN.durationMs * 100).toFixed(0)}%`, 'middle');
      }, `${b.label} · ${b.nWave} 波 · 间隔 P50 ${Math.round(b.gapP50)} ms / P90 ${Math.round(b.gapP90)} ms`);
    });

    /* 时间轴 */
    const ga = el('g', {}, svg);
    for (let s = 0; s <= RUN.durationMs / 1000; s += 5) {
      const x = X(s * 1000);
      line(ga, x, BASE, x, BASE + 5, 'tick');
      text(ga, x, BASE + 18, 'tick-label', `${s}s`, 'middle');
    }
    text(ga, L + W, BASE + 18, 'tick-label strong', `${(RUN.durationMs / 1000).toFixed(2)} s`, 'end');

    /* 悬停层：盖住上图，按鼠标纵坐标找最近的一行 */
    const hit = el('rect', { class: 'hit', x: L, y: REQ_T, width: W, height: REQ_H }, svg);
    const marker = el('g', {}, svg);

    function rowAt(e) {
      const m = svg.getScreenCTM();
      if (!m) return null;
      const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(m.inverse());
      let best = null; let bd = Infinity;
      rows.forEach((r) => { const d = Math.abs(r.y - p.y); if (d < bd) { bd = d; best = r; } });
      return bd < PR * 6 ? best : null;
    }
    function drawMarker(row) {
      marker.textContent = '';
      if (!row) return;
      const q = row.q;
      line(marker, X(q.first), row.y, X(q.last), row.y, 'req-hot');
      el('circle', { class: 'req-dot', cx: X(q.first).toFixed(1), cy: row.y.toFixed(1), r: 2.6 }, marker);
    }
    hit.addEventListener('mousemove', (e) => {
      const row = rowAt(e);
      drawMarker(row);
      if (!row) { hideTip(); return; }
      const q = row.q;
      showTip(e, `req #${q.i} · TTFT ${fmtMs(q.ttft)} · E2E ${fmtMs(q.e2e)} · ${q.outTok} tokens / ${q.chunks} 段`);
    });
    hit.addEventListener('mouseleave', () => { drawMarker(null); hideTip(); });
    hit.addEventListener('click', () => { location.href = D.drill.href; });
  }

  /* ------------------------------------------------------------ 总览：服务曲线 + 16 rank
   * 上半四条服务曲线共用时间轴（定界），下半 16 个 rank（定位）。
   * rank 沿用 AI_Infra NPU Memory Atlas 的 rank 立方体：3:3:4 线框盒，盒高 = 这段时间，
   * 自下而上堆计算与通信，盒顶留空 = 在同步屏障等待；阵列坐标同样是 X = DP、Y = TP。
   * 悬停曲线时立方体切到那一步，离开回到整个窗口（或选中的波段）。 */
  const S = D.series;
  const K = D.ranks;
  const GL = 92;                  // 左列
  const GW = 600;                 // 时间轴
  const TX = VB_W;                // 右对齐的数字列
  const XM = (ms) => GL + (ms / RUN.durationMs) * GW;
  const iAt = (ms) => Math.max(0, Math.min(S.t.length - 1, Math.round(ms / S.stepMs)));

  // 分层与配色取自 Atlas 的类别色；通信层用它对应并行维度的颜色
  const MIX = [
    { k: 'attn', name: 'Attention', grp: '计算', color: '#D0D0D0' },
    { k: 'moe', name: 'MoE 专家', grp: '计算', color: '#D185E6' },
    { k: 'tp', name: 'TP AllReduce', grp: '通信', color: '#8D89EE' },
    { k: 'ep', name: 'EP AlltoAll', grp: '通信', color: '#79D4F2' },
  ];
  const AXIS = { dp: '#B1EF6B', tp: '#8D89EE' };
  const C30 = Math.sqrt(3) / 2;
  const CW = 3; const CD = 3; const CH = 4;        // rank 立方体 3 : 3 : 4
  const PITCH = 5.0;                                // 阵列里相邻 rank 的间距（≈ 1.7 个立方体边长）

  let BH; let BTOP = []; let CBOT; let RTOP; let RHH;
  function layoutMain() {
    const bottom = Math.max(440, targetH()) + VB_T;
    RHH = Math.max(196, (bottom - 26) * 0.46);      // rank 区
    const curves = bottom - RHH - 26 - 62;          // 去掉结论行、时间轴与波段标签
    BH = Math.max(32, (curves - (BANDS.length - 1) * 8) / BANDS.length);
    BTOP = BANDS.map((_, i) => 26 + i * (BH + 8));
    CBOT = BTOP[BANDS.length - 1] + BH;
    RTOP = CBOT + 62;
    BASE = RTOP + RHH;
    VB = [0, VB_T, VB_W, BASE + 6 - VB_T];
  }

  /* 每个 rank 的构成：忙碌时间按 mix 拆成四项，其余是等待 */
  const allIdx = K.t.map((_, i) => i).filter((i) => K.stepMsAt[i] != null);
  function mixOf(s) {
    const t = MIX.reduce((a, m) => a + s.mix[m.k], 0);
    return MIX.map((m) => s.mix[m.k] / t);
  }
  function summarize(rows, span) {
    rows.forEach((s) => {
      const busy = span * s.share;
      s.parts = mixOf(s).map((p) => p * busy);
      s.computeMs = s.parts[0] + s.parts[1];
      s.commMs = s.parts[2] + s.parts[3];
      s.waitMs = span - busy;
    });
    const sh = rows.map((x) => x.share).sort((a, b) => b - a);
    return {
      rows,
      span,
      slowest: rows.reduce((a, b) => (b.share > a.share ? b : a)),
      imbalance: sh[0] / sh[Math.floor(sh.length / 2)],
      whatIfMs: span * (sh[0] - sh[1]),
    };
  }
  // 整个窗口，或选中的波段
  function rankSummary(band) {
    const idx = allIdx.filter((i) => (band === 'slow' ? K.t[i] < D.split : band === 'fast' ? K.t[i] >= D.split : true));
    const rows = K.sum.map((s) => ({
      ...s,
      share: idx.reduce((a, i) => a + K.heat[s.r][i] / 100, 0) / Math.max(1, idx.length),
      routeTok: Math.round(s.routeTok * idx.length / allIdx.length),
    }));
    return summarize(rows, Math.max(1, idx.length * K.stepMs));
  }
  // 悬停到的那一步
  function rankStep(i) {
    const rows = K.sum.map((s) => ({ ...s, share: K.heat[s.r][i] / 100 }));
    return summarize(rows, K.stepMsAt[i]);
  }

  /* ---------- rank 立方体 ----------
   * 移植自 repo/AI_Infra/ai-infra-research/patterns/npu-memory-atlas/index.html：
   * P（2.5D 投影）、shade、block（三面明暗 .82 / 1.04 / .94）、backFaces、outline、rankWallLabel、sideMemoryLabels。
   * 原实现绑定页面全局状态（ranks、cats、viewMode），这里改为传参，便于在 SVG 里多处绘制。 */
  function shade(hex, f) {
    const n = parseInt(hex.slice(1), 16);
    return `#${[n >> 16, (n >> 8) & 255, n & 255].map((v) => Math.min(255, Math.round(v * f)).toString(16).padStart(2, '0')).join('')}`;
  }
  // o = 立方体原点（后角，地面）在画布上的位置，k = 每个逻辑单位的像素
  const isoAt = (o, k) => (x, y, z) => [o[0] + (x - y) * C30 * k, o[1] + (x + y) * 0.5 * k - z * k];
  const ptStr = (P, list) => list.map((p) => P(...p).map((v) => v.toFixed(1)).join(',')).join(' ');

  // 背面压暗 → 自下而上的分层块 → 白色轮廓；返回每层在右前棱上的上下沿，给侧边标注用
  function rankCube(g, P, parts, span, opt) {
    const pad = opt.pad; const gap = opt.gap;
    ['0,0,0 3,0,0 3,3,0 0,3,0', '0,0,0 3,0,0 3,0,4 0,0,4', '0,0,0 0,3,0 0,3,4 0,0,4'].forEach((q) => {
      el('polygon', { class: 'rk-back', points: ptStr(P, q.split(' ').map((s) => s.split(',').map(Number))) }, g);
    });
    const be = [P(0, 0, 0), P(0, 0, CH)];
    line(g, be[0][0], be[0][1], be[1][0], be[1][1], 'rk-back-edge');

    const shown = parts.filter((v) => v / span > 0.004).length;
    const inner = CH - 2 * pad - gap * Math.max(0, shown - 1);
    const x0 = pad; const y0 = pad; const w = CW - 2 * pad; const d = CD - 2 * pad;
    let z = pad; let drawn = 0;
    const bands = parts.map((v, i) => {
      const h = (v / span) * inner;
      if (h < 0.004 * inner) return null;
      if (drawn) z += gap;
      const c = MIX[i].color;
      el('polygon', { points: ptStr(P, [[x0, y0 + d, z], [x0 + w, y0 + d, z], [x0 + w, y0 + d, z + h], [x0, y0 + d, z + h]]), fill: shade(c, 0.82) }, g);
      el('polygon', { points: ptStr(P, [[x0 + w, y0, z], [x0 + w, y0 + d, z], [x0 + w, y0 + d, z + h], [x0 + w, y0, z + h]]), fill: shade(c, 1.04) }, g);
      el('polygon', { points: ptStr(P, [[x0, y0, z + h], [x0 + w, y0, z + h], [x0 + w, y0 + d, z + h], [x0, y0 + d, z + h]]), fill: shade(c, 0.94) }, g);
      const b = { z0: z, z1: z + h };
      z += h; drawn += 1;
      return b;
    });
    const p = (x, y, zz) => P(x, y, zz).map((v) => v.toFixed(1)).join(',');
    el('path', {
      class: 'rk-outline',
      d: `M${p(0, CD, CH)} L${p(0, 0, CH)} L${p(CW, 0, CH)} L${p(CW, CD, CH)} Z M${p(0, CD, CH)} L${p(0, CD, 0)} L${p(CW, CD, 0)} L${p(CW, 0, 0)} L${p(CW, 0, CH)} M${p(CW, CD, CH)} L${p(CW, CD, 0)}`,
    }, g);
    return { bands, top: z };
  }
  // 沿左上棱的斜贴标签（与 Atlas 的 Rank 墙标一致）
  function wallLabel(g, P, title, sub, size) {
    const a = P(0, CD * 0.92, CH);
    const t = el('g', { class: 'rk-wall', transform: `matrix(${C30} -0.5 0 1 ${a[0].toFixed(1)} ${(a[1] - 4).toFixed(1)})` }, g);
    const t1 = el('text', { class: 'rk-wall-title', 'font-size': size, y: sub ? -13 : 0 }, t); t1.textContent = title;
    if (sub) { const t2 = el('text', { class: 'rk-wall-sub', y: -1 }, t); t2.textContent = sub; }
  }
  // 阵列里的小立方体太密，编号平贴在顶面上，不和前后的 rank 打架
  function topLabel(g, P, s, k) {
    const c = P(CW / 2, CD / 2, CH);
    const t = el('text', {
      class: 'rk-top', 'font-size': (k * 1.7).toFixed(1), 'text-anchor': 'middle', 'dominant-baseline': 'central',
      transform: `matrix(${C30} 0.5 ${-C30} 0.5 ${c[0].toFixed(1)} ${c[1].toFixed(1)})`,
    }, g);
    t.textContent = s;
  }
  const rkName = (r) => `Rank ${String(r).padStart(2, '0')}`;

  /* ---------- 下半：16 rank ---------- */
  let headG; let rankG;
  function drawRanks() {
    const selBand = state.sel === 'slow' || state.sel === 'fast' ? state.sel : null;
    const scrub = state.scrub != null && K.stepMsAt[state.scrub] != null ? state.scrub : null;
    const RS = scrub != null ? rankStep(scrub) : rankSummary(selBand);
    const pick = RS.rows[state.rank];
    const scope = scrub != null ? `${(K.t[scrub] / 1000).toFixed(2)} s 这一步` : selBand ? (selBand === 'slow' ? '慢波段' : '快波段') : '整个窗口';

    /* 结论行 */
    headG.textContent = '';
    text(headG, GL, VB_T + 12, 'rank-head hot', `最慢 r${RS.slowest.r} · 不均衡 ${RS.imbalance.toFixed(2)}× · 与次慢对齐可省 ${fmtMs(RS.whatIfMs)}`);
    text(headG, TX, VB_T + 12, 'rank-topo', `${K.topo} · ${scope}`, 'end');

    rankG.textContent = '';
    line(rankG, 0, RTOP - 14, TX, RTOP - 14, 'rule');
    text(rankG, 4, RTOP + 4, 'col-title', `${K.n} Rank · 到达同步屏障`);
    text(rankG, 4, RTOP + 18, 'rk-note', `盒高 = ${scrub != null ? `这一步 ${fmtMs(RS.span)}` : `${scope} ${fmtMs(RS.span)}`} · 顶部留空 = 等待`);

    /* 阵列：X = DP（右下），Y = TP（左下） */
    const ext = PITCH * 3 + CW;                       // 一条轴上的总长
    const k = Math.min((RHH - 46) / (ext + CH), 330 / (2 * ext * C30));
    const o = [62 + ext * C30 * k, RTOP + 40 + CH * k];     // 左侧留给 Y · TP 轴标签
    const PA = isoAt(o, k);
    // 坐标轴：沿地面后两条边的虚线 + 轴名标签（与 Atlas 的并行关系标签同形）
    [['dp', (t) => [t, -1.2, 0], 'X · DP'], ['tp', (t) => [-1.2, t, 0], 'Y · TP']].forEach(([dim, at, name]) => {
      const a = PA(...at(0)); const b = PA(...at(ext + 1.2));
      el('line', { class: 'rk-axis', x1: a[0].toFixed(1), y1: a[1].toFixed(1), x2: b[0].toFixed(1), y2: b[1].toFixed(1), stroke: AXIS[dim] }, rankG);
      const e = PA(...at(ext + 1.6));
      const tag = el('g', { class: 'rk-tag', transform: `translate(${e[0].toFixed(1)} ${e[1].toFixed(1)}) matrix(${C30} ${dim === 'dp' ? 0.5 : -0.5} 0 1 0 0)` }, rankG);
      el('rect', { x: dim === 'dp' ? 0 : -46, y: -8, width: 46, height: 16, rx: 3, stroke: AXIS[dim] }, tag);
      const tx = el('text', { x: dim === 'dp' ? 23 : -23, y: 3.5, 'text-anchor': 'middle', fill: AXIS[dim] }, tag);
      tx.textContent = name;
    });
    // 从后往前画，前面的 rank 盖住后面的
    const order = [...RS.rows].sort((a, b) => (a.dp + a.tp) - (b.dp + b.tp) || a.dp - b.dp);
    order.forEach((s) => {
      const P = isoAt(PA(s.dp * PITCH, s.tp * PITCH, 0), k);
      const hot = s.r === RS.slowest.r;
      const g = el('g', {
        class: `rk${hot ? ' is-hot' : ''}${s.r === state.rank ? ' is-sel' : ''}`,
        tabindex: 0, role: 'button', 'aria-label': `${rkName(s.r)}，到达屏障 ${Math.round(s.share * 100)}%`,
      }, rankG);
      rankCube(g, P, s.parts, RS.span, { pad: 0.22, gap: 0.12 });
      topLabel(g, P, `r${s.r}`, k);
      g.addEventListener('click', () => pickRank(s.r));
      g.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pickRank(s.r); } });
      withTip(g, `${rkName(s.r)} · DP${s.dp} TP${s.tp} · 到达屏障 ${Math.round(s.share * 100)}% · ${MIX.map((m, i) => `${m.name} ${fmtMs(s.parts[i])}`).join(' · ')} · 等待 ${fmtMs(s.waitMs)}`);
    });

    /* 选中的 rank：单 rank 视图 + 侧边标注 */
    const left = o[0] + ext * C30 * k + 76;              // 中间留给 X · DP 轴标签
    const k2 = (RHH - 58) / (CW * 0.5 + CD * 0.5 + CH);
    const o2 = [left + CD * C30 * k2, RTOP + 44 + CH * k2];
    const P2 = isoAt(o2, k2);
    const g2 = el('g', { class: `rk rk-big${pick.r === RS.slowest.r ? ' is-hot' : ''}` }, rankG);
    const cube = rankCube(g2, P2, pick.parts, RS.span, { pad: 0.12, gap: 0.07 });
    wallLabel(g2, P2, rkName(pick.r), `DP${pick.dp} · TP${pick.tp} · 专家 E${pick.experts[0]}–E${pick.experts[1]}`, 17);

    // 侧边标注：色条贴着右前棱上每层的上下沿，文字避让
    const lx = P2(CW, 0, 0)[0] + 18;
    const rows = MIX.map((m, i) => ({ m, v: pick.parts[i], b: cube.bands[i] }))
      .filter((r) => r.b)
      .map((r) => ({ ...r, top: P2(CW, 0, r.b.z1)[1], bot: P2(CW, 0, r.b.z0)[1] }));
    rows.push({ m: { name: '屏障等待', grp: '等待' }, v: pick.waitMs, free: true, top: P2(CW, 0, CH)[1], bot: P2(CW, 0, cube.top)[1] });
    rows.reverse();
    // 避让算法取自 Atlas 的 drawRank：先自上而下推开，再自下而上收回，色条始终贴着各层的物理上下沿
    let prev = -Infinity;
    rows.forEach((r) => { r.y = Math.max((r.top + r.bot) / 2, prev + 21); prev = r.y; });
    for (let j = rows.length - 2; j >= 0; j -= 1) rows[j].y = Math.min(rows[j].y, rows[j + 1].y - 21);
    const ls = el('g', { class: 'rk-labels' }, rankG);
    rows.forEach((r) => {
      el('rect', { class: r.free ? 'rk-swatch free' : 'rk-swatch', x: lx, y: r.top.toFixed(1), width: 4, height: Math.max(1, r.bot - r.top).toFixed(1), fill: r.free ? 'none' : r.m.color }, ls);
      const t = el('text', { x: lx + 12, y: (r.y + 4).toFixed(1), class: `rk-val${r.free && pick.r !== RS.slowest.r ? ' dim' : ''}` }, ls);
      t.innerHTML = `${esc(fmtMs(r.v).replace(/ (m?s)$/, ''))}<tspan class="rk-unit" dx="3">${/ s$/.test(fmtMs(r.v)) ? 's' : 'ms'}</tspan><tspan class="rk-name" dx="6">${esc(r.m.name)}</tspan>`;
    });

    /* 右列：这个 rank 的读数 + 16 个 rank 的到达占比（可点） */
    const ix = lx + 168;
    const iw = TX - ix;
    const gi = el('g', {}, rankG);
    text(gi, ix, RTOP + 4, 'col-title', `${rkName(pick.r)} · ${scope}`);
    const facts = [
      ['到达屏障', `${Math.round(pick.share * 100)}%`, pick.r === RS.slowest.r],
      ['计算 / 通信', `${fmtMs(pick.computeMs)} / ${fmtMs(pick.commMs)}`],
      ['等待', fmtMs(pick.waitMs)],
      ['专家路由 token', `${nf(pick.routeTok)} · ${pick.rel >= 0 ? '+' : '−'}${Math.abs(Math.round(pick.rel * 100))}%`, pick.rel > 0.15],
    ];
    facts.forEach(([kk, v, bad], j) => {
      const y = RTOP + 26 + j * 19;
      text(gi, ix, y, 'rk-fact-k', kk);
      text(gi, TX, y, `rk-fact-v${bad ? ' hot' : ''}`, v, 'end');
    });
    const by = RTOP + 26 + facts.length * 19 + 12;
    const bh = Math.max(36, BASE - by - 20);
    text(gi, ix, by, 'rk-note', '各 rank 到达屏障的占比');
    const bw = iw / K.n;
    const lo = 0.6;
    RS.rows.forEach((s) => {
      const h = Math.max(1, (s.share - lo) / (1 - lo) * bh);
      const x = ix + s.r * bw;
      const bar = el('rect', {
        class: `rk-bar${s.r === RS.slowest.r ? ' hot' : ''}${s.r === pick.r ? ' is-sel' : ''}`,
        x: (x + 1).toFixed(1), y: (by + 8 + bh - h).toFixed(1), width: (bw - 2).toFixed(1), height: h.toFixed(1),
      }, gi);
      bar.addEventListener('click', () => pickRank(s.r));
      withTip(bar, `r${s.r} · ${Math.round(s.share * 100)}%`);
      if (s.r % 4 === 0) text(gi, x + 1, by + 8 + bh + 12, 'rk-tick dim', `DP${s.dp}`);
    });
  }
  function pickRank(r) {
    state.rank = r;
    writeHash();
    drawRanks();
  }

  function renderMain() {
    layoutMain();
    svg.textContent = '';
    svg.setAttribute('viewBox', VB.join(' '));
    syncTextScale();
    svg.setAttribute('preserveAspectRatio', 'xMidYMin meet');
    const selBand = state.sel === 'slow' || state.sel === 'fast' ? state.sel : null;
    headG = el('g', {}, svg);

    /* ---------- 上半：服务曲线 ---------- */
    BANDS.forEach((b, bi) => {
      const top = BTOP[bi]; const bot = top + BH;
      const vals = b.keys.map((k) => S[k]);
      const peak = Math.max(...vals.flat().filter((v) => v != null));
      const max = b.max(peak) || 1;
      const Y = (v) => bot - (v / max) * (BH - 5);

      el('rect', { class: 'panel', x: GL, y: top, width: GW, height: BH }, svg);
      if (b.keys.length > 1) {
        const stack = S.t.map((_, i) => b.keys.reduce((a, k) => a + (S[k][i] || 0), 0));
        [[stack, 'band-fill wait'], [S[b.keys[0]], 'band-fill gen']].forEach(([arr, cls]) => {
          el('path', { class: cls, d: `M${XM(S.t[0])},${bot} ${arr.map((v, i) => `L${XM(S.t[i]).toFixed(1)},${Y(v).toFixed(1)}`).join(' ')} L${XM(S.t[S.t.length - 1]).toFixed(1)},${bot} Z` }, svg);
        });
      } else {
        const arr = S[b.keys[0]]; const pt = [];
        arr.forEach((v, i) => { if (v != null) pt.push([XM(S.t[i]), Y(v)]); });
        if (b.fill) el('path', { class: 'band-fill', d: `M${pt[0][0].toFixed(1)},${bot} ${pt.map((p) => `L${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ')} L${pt[pt.length - 1][0].toFixed(1)},${bot} Z` }, svg);
        el('polyline', { class: 'band-line', points: pts(pt) }, svg);
      }
      text(svg, GL + 7, top + 14, 'band-title', b.title + (b.unit ? ` · ${b.unit}` : ''));
      text(svg, GL - 8, top + 10, 'band-max', String(max), 'end');
      text(svg, GL - 8, bot - 1, 'band-max dim', '0', 'end');
      text(svg, TX, top + 14, 'band-stat', b.stat(), 'end');
      if (b.keys.length > 1) {
        el('rect', { class: 'key gen', x: GL + 7, y: top + 20, width: 8, height: 8, rx: 2 }, svg);
        text(svg, GL + 19, top + 27, 'band-key', b.names.gen);
        el('rect', { class: 'key wait', x: GL + 58, y: top + 20, width: 8, height: 8, rx: 2 }, svg);
        text(svg, GL + 70, top + 27, 'band-key', b.names.wait);
      }
      if (selBand) {
        const a = selBand === 'slow' ? XM(D.split) : XM(0);
        const c = selBand === 'slow' ? XM(RUN.durationMs) : XM(D.split);
        el('rect', { class: 'dim-cover', x: a, y: top, width: c - a, height: BH }, svg);
      }
    });

    /* 分界线与时间轴 */
    const sx = XM(D.split);
    line(svg, sx, BTOP[0] - 4, sx, CBOT + 4, 'split-line');
    text(svg, sx - 6, BTOP[0] - 9, 'split-label', `${(D.split / 1000).toFixed(2)} s · 间隔 ${Math.round(D.bands[0].gapP50)} → ${Math.round(D.bands[1].gapP50)} ms`, 'end');
    const ga = el('g', {}, svg);
    for (let s = 0; s <= RUN.durationMs / 1000; s += 5) {
      const x = XM(s * 1000);
      line(ga, x, CBOT + 2, x, CBOT + 6, 'tick');
      text(ga, x, CBOT + 18, 'tick-label', `${s}s`, 'middle');
    }
    text(ga, GL + GW, CBOT + 18, 'tick-label strong', `${(RUN.durationMs / 1000).toFixed(2)} s`, 'end');

    /* 两段可点：点了曲线与 rank 一起过滤 */
    const gbd = el('g', {}, svg);
    D.bands.forEach((b) => {
      const x0 = XM(b.t0); const x1 = XM(b.t1); const mid = (x0 + x1) / 2;
      const hot = b.id === 'slow' ? ' hot' : '';
      pickLabel(gbd, b.id, state.sel === b.id, (n) => {
        el('rect', { class: 'pick-hit', x: x0, y: CBOT + 24, width: x1 - x0, height: 24 }, n);
        line(n, x0 + 1, CBOT + 28, x1 - 1, CBOT + 28, `band-bar${hot}`);
        text(n, mid, CBOT + 44, `band-label${hot}`, `${b.nWave} 波 · 间隔 ${Math.round(b.gapP50)} ms · 占窗口 ${(b.spanMs / RUN.durationMs * 100).toFixed(0)}%`, 'middle');
      }, `${b.label} · ${b.nWave} 波 · 间隔 P50 ${Math.round(b.gapP50)} ms`);
    });

    rankG = el('g', {}, svg);
    drawRanks();

    /* ---------- 联动：曲线上的十字光标把 rank 切到那一步 ---------- */
    const cross = el('g', {}, svg);
    const hit = el('rect', { class: 'hit cross-hit', x: GL, y: BTOP[0], width: GW, height: CBOT - BTOP[0] }, svg);
    hit.addEventListener('mousemove', (e) => {
      const m = svg.getScreenCTM();
      if (!m) return;
      const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(m.inverse());
      const i = iAt((p.x - GL) / GW * RUN.durationMs);
      const x = XM(S.t[i]);
      cross.textContent = '';
      line(cross, x, BTOP[0], x, CBOT, 'cross');
      BANDS.forEach((b, bi) => {
        const top = BTOP[bi]; const bot = top + BH;
        const peak = Math.max(...b.keys.map((k) => S[k]).flat().filter((v) => v != null));
        const max = b.max(peak) || 1;
        const v = b.keys.length > 1 ? S.gen[i] : S[b.keys[0]][i];
        if (v == null) return;
        el('circle', { class: 'cross-dot', cx: x.toFixed(1), cy: (bot - (v / max) * (BH - 5)).toFixed(1), r: 2.8 }, cross);
      });
      if (state.scrub !== i) { state.scrub = i; drawRanks(); }
      showTip(e, `${(S.t[i] / 1000).toFixed(2)} s · ${BANDS.map((b) => b.read(i)).join(' · ')}`);
    });
    hit.addEventListener('mouseleave', () => {
      cross.textContent = ''; hideTip();
      state.scrub = null; drawRanks();
    });
  }

  function renderScene() {
    if (state.view === 'rows') renderRows(); else renderMain();
  }

  function renderViewTabs() {
    const box = $('#viewTabs');
    box.innerHTML = VIEWS.map((v) => `<button type="button" data-view="${v.k}" class="${v.k === state.view ? 'is-on' : ''}">${esc(v.label)}</button>`).join('');
    box.querySelectorAll('[data-view]').forEach((b) => b.addEventListener('click', () => {
      state.view = b.dataset.view;
      writeHash(); renderViewTabs(); renderScene();
    }));
  }

  /* ------------------------------------------------------------ 顶部：头条 KPI + 运行配置 */
  function renderHead() {
    const H = D.headline;
    $('#eyebrow').textContent = `${RUN.id} · DSPARK SERVING`;
    $('#runName').innerHTML = `${H.value}<small>${esc(H.unit)}</small>`;
    $('#runSub').innerHTML = `<b>${esc(H.label)}</b> · ${esc(H.foot)}`;
    $('#runMeta').innerHTML = D.meta.map((m) => `<dt>${esc(m.k)}</dt><dd>${esc(m.v)}</dd>`).join('');
    const tg = $('#metaToggle'); const card = $('#runMeta');
    tg.addEventListener('click', () => {
      const on = tg.getAttribute('aria-expanded') === 'true';
      tg.setAttribute('aria-expanded', String(!on));
      card.hidden = on;
    });
    const a = $('#drillLink');
    a.href = D.drill.href;
    a.title = `下钻：${D.drill.label}`;
    a.setAttribute('aria-label', a.title);
  }

  /* ------------------------------------------------------------ 右栏：服务指标
   * 每项配一种编码：分位条（延迟）/ 占比条（容量、健康）/ 迷你曲线（随时间变化的量）。
   * 迷你曲线上标出 23.00 s 分界与当前选中的波段，和主图保持同一套读法。 */
  function sparkline(key) {
    const arr = D.series[key];
    const n = arr.length;
    const max = Math.max(...arr.filter((v) => v != null)) || 1;
    const pt = arr.map((v, i) => `${(i / (n - 1) * 100).toFixed(2)},${(22 - (v || 0) / max * 20).toFixed(2)}`).join(' ');
    const sx = (D.split / RUN.durationMs) * 100;
    const sel = state.sel === 'slow' ? [0, sx] : state.sel === 'fast' ? [sx, 100] : null;
    return `<svg class="spark" viewBox="0 0 100 24" preserveAspectRatio="none" aria-hidden="true">
      ${sel ? `<rect class="spark-sel" x="${sel[0].toFixed(2)}" y="0" width="${(sel[1] - sel[0]).toFixed(2)}" height="24" />` : ''}
      <polyline class="spark-line" points="${pt}" />
      <line class="spark-split" x1="${sx.toFixed(2)}" y1="0" x2="${sx.toFixed(2)}" y2="24" />
    </svg>`;
  }
  // 分位条：min–max 为轨，标出 P50 / P90 / P99 的位置
  function distBar(d) {
    const span = Math.max(1e-6, d.max - d.min);
    const at = (v) => ((v - d.min) / span * 100).toFixed(2);
    return `<div class="qbar" title="min ${fmtMs(d.min)} · P50 ${fmtMs(d.p50)} · P90 ${fmtMs(d.p90)} · P99 ${fmtMs(d.p99)} · max ${fmtMs(d.max)}">
      <i class="qfill" style="width:${at(d.p50)}%"></i>
      <i class="qt q50" style="left:${at(d.p50)}%"></i>
      <i class="qt q90" style="left:${at(d.p90)}%"></i>
      <i class="qt q99" style="left:${at(d.p99)}%"></i>
    </div>
    <div class="qax"><span>${fmtMs(d.min)}</span><span>${fmtMs(d.max)}</span></div>`;
  }
  // 迷你直方图：只给最该看的那一项（输出间隔的双峰）
  function miniHist(key) {
    const d = D.dists.find((x) => x.k === key);
    const max = Math.max(...d.bins.map((b) => b.n));
    const hot = d.hot || null;
    return `<div class="mhist">${d.bins.map((b) => {
      const isHot = hot && b.x0 >= hot[0] && b.n > 0;
      return `<i class="${b.n === 0 ? 'zero' : isHot ? 'hot' : ''}" style="--h:${(b.n / max * 100).toFixed(1)}%"></i>`;
    }).join('')}</div>
    <div class="qax"><span>${fmtMs(d.bins[0].x0)}</span><span>${fmtMs(d.bins[d.bins.length - 1].x1)}</span></div>`;
  }
  function renderRail() {
    $('#kpis').innerHTML = D.rail.map((g) => `
      <section class="rail-group p${g.p}">
        <h3>${esc(g.g)}${g.items.some((x) => x.dist) ? '<i class="qlg"><b class="q50"></b>P50<b class="q90"></b>P90<b class="q99"></b>P99</i>' : ''}</h3>
        ${g.items.map((it) => `
          <div class="rail-item${(it.bar && it.bar.tone === 'bad') || it.tone === 'bad' ? ' is-bad' : ''}">
            <div class="ri-head">
              <span class="ri-label">${esc(it.label)}${it.pill ? '<i class="pill ok">正常</i>' : ''}</span>
              <span class="ri-val">${esc(it.v)}${it.u ? `<small>${esc(it.u)}</small>` : ''}</span>
            </div>
            ${it.hist ? miniHist(it.hist) : ''}
            ${it.dist ? distBar(it.dist) : ''}
            ${it.bar ? `<div class="pbar"><i style="width:${Math.min(100, it.bar.val / it.bar.max * 100).toFixed(1)}%"></i></div>` : ''}
            ${it.spark ? sparkline(it.spark) : ''}
            <div class="ri-foot">${esc(it.foot)}</div>
          </div>`).join('')}
      </section>`).join('');
  }

  /* ------------------------------------------------------------ 底部：分布直方图
   * 输出间隔这一张把两个峰直接标在柱子上方，点峰也能筛选对应波段。 */
  function renderDist() {
    const d = D.dists.find((x) => x.k === state.dist);
    const max = Math.max(...d.bins.map((b) => b.n));
    const hot = d.hot || null;
    const lo = d.bins[0].x0; const hi = d.bins[d.bins.length - 1].x1;
    const at = (v) => ((v - lo) / (hi - lo) * 100).toFixed(1);
    const peaks = d.k === 'itl' ? [
      { id: 'fast', m: D.itlModes.fast, cls: '' },
      { id: 'slow', m: D.itlModes.slow, cls: ' hot' },
    ] : [];
    const wrap = $('#dist');
    wrap.innerHTML = `
      <div class="dist-head">
        <div class="dist-tabs">${D.dists.map((x) => `<button type="button" data-dist="${x.k}" class="${x.k === d.k ? 'is-on' : ''}">${esc(x.label)}</button>`).join('')}</div>
        <div class="dist-stats">
          <span>P50 <b>${fmtMs(d.stats.p50)}</b></span>
          <span>P90 <b>${fmtMs(d.stats.p90)}</b></span>
          <span>P99 <b>${fmtMs(d.stats.p99)}</b></span>
          <span>均值 <b>${fmtMs(d.stats.mean)}</b></span>
          <span>样本 <b>${nf(d.stats.n)}</b></span>
        </div>
      </div>
      <div class="dist-plot">
        <div class="dist-y"><span>${nf(max)}</span><span>${nf(Math.round(max / 2))}</span><span>0</span></div>
        <div class="dist-bars" role="img" aria-label="${esc(d.label)}分布">
          ${d.bins.map((b) => {
            const isHot = hot && b.x0 >= hot[0] && b.x1 <= hot[1] && b.n > 0;
            const cls = b.n === 0 ? 'zero' : isHot ? 'hot' : '';
            return `<i class="${cls}" style="--h:${(b.n / max * 100).toFixed(2)}%" data-tip="${esc(fmtMs(b.x0))} – ${esc(fmtMs(b.x1))} · ${nf(b.n)} 个"></i>`;
          }).join('')}
          ${peaks.map((p) => {
            const x = Number(at(p.m.p50));
            const tf = x < 12 ? 'translateX(-15%)' : x > 88 ? 'translateX(-85%)' : 'translateX(-50%)';
            return `<button type="button" class="peak${p.cls}${state.sel === p.id ? ' is-on' : ''}" data-sel="${p.id}" style="left:${x}%;transform:${tf}"><b>${nf(p.m.n)}</b> 个 · ${fmtMs(p.m.p50)}</button>`;
          }).join('')}
        </div>
      </div>
      <div class="dist-axis"><span>${fmtMs(d.bins[0].x0)}</span><span>${fmtMs(d.bins[d.bins.length - 1].x1)}</span></div>`;
    wrap.querySelectorAll('[data-dist]').forEach((b) => b.addEventListener('click', () => {
      state.dist = b.dataset.dist; writeHash(); renderDist();
    }));
    wrap.querySelectorAll('.peak').forEach((b) => b.addEventListener('click', () => select(b.dataset.sel)));
    wrap.querySelectorAll('.dist-bars i').forEach((b) => withTip(b, b.dataset.tip));
  }

  /* ------------------------------------------------------------ init */
  readHash();
  renderHead();
  renderRail();
  renderViewTabs();
  renderDist();
  renderScene();
  writeHash();

  window.addEventListener('hashchange', () => { readHash(); renderDist(); renderRail(); renderScene(); });
  let rt = null;
  window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(renderScene, 120); });
})();
