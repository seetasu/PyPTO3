/* =============================================================
 * Tuning Console
 *
 * One real on-device run (Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617)
 * opened as a working surface for the tuning loop:
 *   E2E  -> L2 schedule -> L1/L0 core pipeline -> compiler lowering -> ISA / layout
 *
 * Every number rendered here comes from data.js, which build-data.cjs derives
 * from the run's own artifacts. Nothing is modelled or simulated.
 *
 * Shared patterns used:
 *   ide-frame        page shell, panes, bottom dock, status strip
 *   workbench-shell  split resize kernel (through ide-frame)
 *   swimlane-task    every timed task bar + its hover tooltip + colormap
 * ============================================================= */
(function () {
  'use strict';

  const D = window.TUNING_RUN;
  const SW = window.PtoSwimlaneTaskPattern;
  const CYC_PER_US = D.case.clockHz / 1e6;   /* trace clock: 50 MHz */

  /* ------------------------------------------------------------- state */
  const S = {
    rank: D.defaultRank,
    view: 'e2e',
    task: D.derived.worstHandoff,
    finding: null,
    findingLevel: 'all',
    focus: null,               /* 'finding' | 'task' | 'hint' | 'pass' */
    laneFilter: 'all',
    colorMode: 'semantic',
    overlay: 'sched',
    critOnly: false,
    t0: 0, t1: 0,
    compilerTab: 'passes',
    pass: 17,
    hintSite: null,
    hintModule: 'all',
    dockMode: 'sched',
    termTab: 'problems',
    ledger: [],
    tile: null,
  };

  const R = () => D.ranks[S.rank];
  const $ = (sel) => document.querySelector(sel);
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  };
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"]/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const num = (v, d) => (v == null ? '—' : Number(v).toFixed(d == null ? 2 : d));
  const us = (v, d) => (v == null ? '—' : Number(v).toFixed(d == null ? 1 : d) + ' us');
  const kb = (b) => (b == null ? '—' : b >= 1024 ? (b / 1024).toFixed(b % 1024 ? 1 : 0) + ' KB' : b + ' B');
  const pct = (v, d) => (v == null ? '—' : Number(v).toFixed(d == null ? 1 : d) + '%');
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  const LEVELS = [
    { id: 'e2e', label: 'E2E', hint: '端到端与 rank 分解' },
    { id: 'l2', label: 'L2 调度', hint: '任务放置、依赖、关键路径' },
    { id: 'l1', label: 'L1 / L0', hint: '单核流水与片上预算' },
    { id: 'compiler', label: '编译器', hint: 'Pass、流水深度、搬运粒度' },
    { id: 'isa', label: 'ISA / 布局', hint: '布局与指令层证据' },
  ];
  const LEVEL_LABEL = {};
  LEVELS.forEach((l) => { LEVEL_LABEL[l.id] = l.label; });

  /* Colormap: the shared pattern owns every task color decision, including the
   * aic / aiv / aicpu lane-kind colors. No page-local palette. */
  const CMAP = SW.createTaskColormap();

  /* ---------------------------------------------------- derived at boot */
  /* Which invocation does each rank's device trace correspond to?
   * Reconcile the trace span against the host-reported device_wall.sched. */
  const TRACE_MATCH = {};
  Object.keys(D.ranks).forEach((rank) => {
    const span = D.ranks[rank].swimlane.spanUs;
    let best = null;
    Object.keys(D.e2e[rank]).forEach((inv) => {
      const sp = D.e2e[rank][inv]['chip.run.runner_run.device_wall.sched'];
      if (!sp) return;
      const diff = Math.abs(sp.us - span);
      if (!best || diff < best.diff) best = { inv: +inv, hostUs: sp.us, diff: diff };
    });
    TRACE_MATCH[rank] = best;
  });

  const findingById = {};
  D.findings.forEach((f) => { findingById[f.id] = f; });

  const taskByTag = {};
  const tasksOf = {};
  Object.keys(D.ranks).forEach((rank) => {
    tasksOf[rank] = {};
    D.ranks[rank].tasks.forEach((t) => { tasksOf[rank][t.tag] = t; });
  });
  const curTask = () => tasksOf[S.rank][S.task] || R().tasks[0];

  /* seed the ledger with the locked baseline, straight from the artifacts */
  S.ledger.push({
    id: 'B0',
    state: 'baseline',
    title: '基线锁定 · ' + D.case.program,
    findingId: null,
    hypothesis: '固定 shape / dtype / 平台 / 卡数 / 工具链，作为后续所有对比的唯一基准。',
    change: D.case.runDir + '（' + D.case.capturedAt + '，platform ' + D.case.toolchain.platform + '）',
    correctness: 'distributed_meta.json 记录 ' + D.case.params.length + ' 个绑定参数，schema ' + D.case.metaSchema,
    perf: 'rank0 device_wall ' + us(D.e2e.rank0[2]['chip.run.runner_run.device_wall'].us)
      + ' / rank1 ' + us(D.e2e.rank1[2]['chip.run.runner_run.device_wall'].us) + '（inv=2）',
    keep: '保留为基线',
  });
  let ledgerSeq = 0;
  const openExperiment = () => S.ledger.find((r) => r.state === 'open') || null;

  /* ============================================================ tables */
  function table(cols, rows, opts) {
    const o = opts || {};
    const wrap = el('div', 'tc-table-scroll');
    const t = el('table', 'tc-table');
    const thead = el('thead');
    const tr = el('tr');
    cols.forEach((c) => {
      const th = el('th', c.num ? 'num' : null, c.label);
      if (c.width) th.style.width = c.width;
      tr.appendChild(th);
    });
    thead.appendChild(tr);
    t.appendChild(thead);
    const tb = el('tbody');
    rows.forEach((row) => {
      const r = el('tr');
      if (row.__selected) r.className = 'is-selected';
      cols.forEach((c) => {
        const td = el('td', [c.num ? 'num' : null, c.mono ? 'mono' : null].filter(Boolean).join(' ') || null);
        const v = c.cell ? c.cell(row) : row[c.key];
        if (v instanceof Node) td.appendChild(v);
        else td.innerHTML = v == null ? '—' : String(v);
        r.appendChild(td);
      });
      if (o.onPick) r.addEventListener('click', () => o.onPick(row));
      else r.style.cursor = 'default';
      tb.appendChild(r);
    });
    t.appendChild(tb);
    wrap.appendChild(t);
    return wrap;
  }

  function bar(ratio, tone) {
    const b = el('span', 'tc-bar');
    const i = el('i');
    i.style.width = clamp(ratio * 100, 0, 100).toFixed(2) + '%';
    if (tone) i.dataset.tone = tone;
    b.appendChild(i);
    return b;
  }

  function tiles(items) {
    const g = el('div', 'tc-tiles');
    items.forEach((it) => {
      const n = el('div', 'tc-tile');
      if (it.tone) n.dataset.tone = it.tone;
      n.appendChild(el('span', 'k', it.k));
      n.appendChild(el('span', 'v', it.v));
      if (it.u) n.appendChild(el('span', 'u', it.u));
      g.appendChild(n);
    });
    return g;
  }

  function sectionHead(title, sub, right) {
    const h = el('div', 'tc-section-head');
    h.appendChild(el('h2', null, title));
    if (sub) h.appendChild(el('span', 'sub', sub));
    if (right) { h.appendChild(el('span', 'spacer')); h.appendChild(right); }
    return h;
  }

  function btn(label, opts) {
    const o = opts || {};
    const b = el('button', ['btn', o.variant ? 'btn-' + o.variant : null, o.size ? 'btn-' + o.size : null,
      o.selected ? 'is-selected' : null].filter(Boolean).join(' '), label);
    b.type = 'button';
    if (o.title) b.title = o.title;
    if (o.disabled) b.disabled = true;
    if (o.on) b.addEventListener('click', o.on);
    return b;
  }

  function group(cls, items, current, onPick) {
    const g = el('div', cls);
    items.forEach((it) => {
      const cls2 = cls.indexOf('segmented') === 0 ? 'segmented-control-item' : 'tab-control-item';
      const b = el('button', cls2 + (it.id === current ? ' is-selected' : ''), it.label);
      b.type = 'button';
      if (it.hint) b.title = it.hint;
      b.setAttribute('aria-pressed', it.id === current ? 'true' : 'false');
      b.addEventListener('click', () => onPick(it.id));
      g.appendChild(b);
    });
    return g;
  }

  function field(label, control) {
    const f = el('label', 'tc-field');
    f.appendChild(el('span', null, label));
    f.appendChild(control);
    return f;
  }

  function select(options, current, onChange) {
    const s = el('select');
    options.forEach((o) => {
      const opt = el('option', null, o.label);
      opt.value = o.id;
      if (o.id === current) opt.selected = true;
      s.appendChild(opt);
    });
    s.addEventListener('change', () => onChange(s.value));
    return s;
  }

  /* ====================================================== canvas basics */
  function fitCanvas(canvas, cssW, cssH) {
    const dpr = window.devicePixelRatio || 1;
    canvas.style.width = cssW + 'px';
    canvas.style.height = cssH + 'px';
    canvas.width = Math.max(1, Math.round(cssW * dpr));
    canvas.height = Math.max(1, Math.round(cssH * dpr));
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    return ctx;
  }
  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function drawTimeRuler(ctx, x0, w, y, t0, t1) {
    const span = t1 - t0;
    const stepRaw = span / 8;
    const mag = Math.pow(10, Math.floor(Math.log10(stepRaw)));
    const step = [1, 2, 5, 10].map((m) => m * mag).find((v) => v >= stepRaw) || mag * 10;
    ctx.save();
    ctx.font = '500 11px ' + cssVar('--font-sans');
    ctx.fillStyle = cssVar('--foreground-muted');
    ctx.strokeStyle = cssVar('--border-subtle');
    ctx.lineWidth = 1;
    ctx.textBaseline = 'alphabetic';
    for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) {
      const x = x0 + ((t - t0) / span) * w;
      ctx.beginPath();
      ctx.moveTo(Math.round(x) + 0.5, y + 4);
      ctx.lineTo(Math.round(x) + 0.5, y + 10);
      ctx.stroke();
      ctx.textAlign = 'left';
      ctx.fillText(Math.round(t) + '', Math.round(x) + 3, y + 2);
    }
    ctx.restore();
  }

  /* task object handed to the shared pattern (tooltip + bar) */
  function barTask(t, block, laneName) {
    return {
      label: t.callable,
      displayName: t.callable,
      rawName: t.tag + ' · ' + t.callable,
      colorKey: t.callable,
      lane: laneName,
      laneKind: t.kind,
      laneId: laneName,
      totalCycle: Math.round((block ? block[1] : t.span) * CYC_PER_US),
      clcCycle: block ? Math.round((block[1] - (t.setupMean || 0)) * CYC_PER_US) : null,
      status: /_wait$/.test(t.callable) ? 'wait' : (t.kind === 'mix' ? 'overlap' : 'ok'),
      dominantCounter: t.kind.toUpperCase() + ' · ' + t.blockCount + ' blk / ' + t.coreCount + ' core',
      wrapId: 'ring ' + t.ring,
    };
  }

  /* one shared tooltip per canvas host, created by the pattern */
  function attachTooltip(host, canvas, resolve) {
    const hover = SW.initHoverTooltip({
      root: canvas,
      targets: [canvas],
      appendTo: host,
      bounds: host,
      durationUnit: 'cyc',
      getTask: (target, event) => resolve(event),
    });
    if (!hover) return null;
    let last = null;
    canvas.addEventListener('pointermove', (event) => {
      const task = resolve(event);
      if (!task) { SW.hideTooltip(hover.tooltip); last = null; return; }
      const key = task.rawName + '|' + task.totalCycle;
      if (key !== last) {
        last = key;
        SW.showTooltip(hover.tooltip, task, event, { bounds: host, target: canvas, durationUnit: 'cyc' });
      }
    });
    canvas.addEventListener('pointerleave', () => { last = null; });
    return hover;
  }

  /* ======================================================= E2E view */
  const SPAN_TREE = [
    ['chip.run', 0],
    ['chip.run.bind', 1],
    ['chip.run.bind.args', 2],
    ['chip.run.bind.prebuilt', 2],
    ['chip.run.runner_run', 1],
    ['chip.run.runner_run.device_wall', 2],
    ['chip.run.runner_run.device_wall.preamble', 3],
    ['chip.run.runner_run.device_wall.graph_build', 3],
    ['chip.run.runner_run.device_wall.orch', 3],
    ['chip.run.runner_run.device_wall.sched', 3],
    ['chip.run.runner_run.device_wall.post_orch', 3],
    ['chip.run.validate', 1],
  ];

  function viewE2E(stage) {
    /* --- gates: what must be true before any number is trusted --- */
    const invCount = Object.keys(D.e2e.rank0).length;
    const gates = [
      {
        k: 'Case 固定', state: 'pass', v: 'locked',
        d: D.case.params.length + ' 个绑定参数 · ' + D.case.backend + ' · ' + D.case.ranks.length + ' rank · ' + D.case.numCores + ' core',
      },
      {
        k: '工具链', state: 'pass', v: D.case.toolchain.platform,
        d: 'pto-isa ' + D.case.toolchain.ptoIsaRevision.slice(0, 10) + ' · runtime ' + D.case.toolchain.runtimeName,
      },
      {
        k: '迭代次数', state: 'warn', v: 'n = ' + invCount,
        d: '本 dump 只有 ' + invCount + ' 次调用，mean / median 不成立；下结论前需补足迭代。',
      },
      {
        k: 'PMU', state: 'info', v: 'off',
        d: 'trace 内无 PMU counter；PMU 打开会改变调度，不能与本基线直接比较。',
      },
    ];
    const gs = el('div', 'tc-gates');
    gates.forEach((g) => {
      const n = el('div', 'tc-gate');
      n.dataset.state = g.state;
      n.appendChild(el('span', 'k', g.k));
      n.appendChild(el('span', 'v', g.v));
      n.appendChild(el('span', 'd', g.d));
      gs.appendChild(n);
    });
    const secGate = el('section');
    secGate.appendChild(sectionHead('门禁', '每轮调优前先过这一行，否则后面的数字不可比'));
    secGate.appendChild(gs);
    stage.appendChild(secGate);

    /* --- rank x invocation table --- */
    const rows = [];
    Object.keys(D.e2e).forEach((rank) => {
      Object.keys(D.e2e[rank]).forEach((inv) => {
        const sp = D.e2e[rank][inv];
        const get = (n) => (sp[n] ? sp[n].us : null);
        rows.push({
          rank: rank, inv: +inv,
          dev: get('chip.run.runner_run.device_wall'),
          sched: get('chip.run.runner_run.device_wall.sched'),
          build: get('chip.run.runner_run.device_wall.graph_build'),
          orch: get('chip.run.runner_run.device_wall.orch'),
          host: get('chip.run.runner_run'),
          bind: get('chip.run.bind'),
          traced: TRACE_MATCH[rank] && TRACE_MATCH[rank].inv === +inv,
        });
      });
    });
    const maxDev = Math.max.apply(null, rows.map((r) => r.dev));
    const secTab = el('section');
    secTab.appendChild(sectionHead('每 rank / 每次调用', '设备侧 device_wall 优先；host 侧时间包含绑定与校验，不代表内核',
      el('span', 'tc-readout', '点行切换 L2 视图的 rank')));
    secTab.appendChild(table([
      { label: 'Rank', key: 'rank', mono: true },
      { label: 'inv', key: 'inv', num: true },
      {
        label: 'device_wall', num: true,
        cell: (r) => (r.traced ? '<span class="ok">' : '<span>') + num(r.dev, 1) + '</span>',
      },
      { label: '', cell: (r) => bar(r.dev / maxDev, r.dev === maxDev ? 'warn' : 'neutral') },
      { label: 'sched', key: 'sched', num: true, cell: (r) => num(r.sched, 1) },
      { label: 'graph_build', num: true, cell: (r) => (r.build > r.sched ? '<span class="warn">' : '<span>') + num(r.build, 1) + '</span>' },
      { label: 'orch', num: true, cell: (r) => num(r.orch, 2) },
      { label: 'host runner_run', num: true, cell: (r) => num(r.host, 1) },
      { label: 'trace', cell: (r) => (r.traced ? '<span class="ok">已采</span>' : '—') },
    ], rows.map((r) => Object.assign(r, { __selected: r.rank === S.rank && r.traced })), {
      onPick: (r) => { S.rank = r.rank; S.focus = null; render(); },
    }));
    stage.appendChild(secTab);

    /* --- hierarchical span breakdown for the selected rank, traced invocation --- */
    const inv = TRACE_MATCH[S.rank].inv;
    const sp = D.e2e[S.rank][inv];
    const total = sp['chip.run'].us;
    const secBreak = el('section');
    secBreak.appendChild(sectionHead('调用剖分 · ' + S.rank + ' inv=' + inv,
      'host span 层级来自 host log 的 STRACE 记录，device_wall 及其子段用设备时钟'));
    const rowsWrap = el('div', 'tc-spanrows');
    SPAN_TREE.forEach((entry) => {
      const name = entry[0];
      const depth = entry[1];
      const span = sp[name];
      if (!span) return;
      const row = el('div', 'tc-spanrow');
      row.dataset.depth = depth;
      row.appendChild(el('span', 'lbl', name.replace(/^chip\.run\.?/, '') || 'chip.run'));
      const tone = /graph_build/.test(name) ? 'warn' : /device_wall$/.test(name) ? 'good' : /sched/.test(name) ? 'neutral' : null;
      row.appendChild(bar(span.us / total, tone));
      row.appendChild(el('span', 'val', num(span.us, 2) + ' us'));
      rowsWrap.appendChild(row);
    });
    secBreak.appendChild(rowsWrap);
    stage.appendChild(secBreak);

    /* --- reconciliation: host span vs device trace --- */
    const recSec = el('section');
    recSec.appendChild(sectionHead('对账', 'host 报的 sched 段应当与设备 trace 跨度一致，否则 trace 不属于这次调用'));
    const recRows = Object.keys(D.ranks).map((rank) => {
      const m = TRACE_MATCH[rank];
      const sw = D.ranks[rank].swimlane;
      return {
        rank: rank, inv: m.inv, host: m.hostUs, trace: sw.spanUs, diff: m.diff,
        tasks: D.ranks[rank].tasks.length,
        blocks: sw.blocks.reduce((a, b) => a + b.length, 0),
        crit: D.ranks[rank].critical.tags.length,
        aic: D.ranks[rank].occupancy.aicUtil,
        aiv: D.ranks[rank].occupancy.aivUtil,
        __selected: rank === S.rank,
      };
    });
    recSec.appendChild(table([
      { label: 'Rank', key: 'rank', mono: true },
      { label: '匹配 inv', key: 'inv', num: true },
      { label: 'host sched', num: true, cell: (r) => num(r.host, 1) },
      { label: 'trace span', num: true, cell: (r) => num(r.trace, 1) },
      { label: '偏差', num: true, cell: (r) => (r.diff / r.host < 0.02 ? '<span class="ok">' : '<span class="warn">') + num(r.diff, 1) + ' us</span>' },
      { label: '任务', key: 'tasks', num: true },
      { label: '块', key: 'blocks', num: true },
      { label: '关键路径', cell: (r) => r.crit + ' 节点', num: true },
      { label: 'AIC 占用', num: true, cell: (r) => pct(r.aic) },
      { label: 'AIV 占用', num: true, cell: (r) => pct(r.aiv) },
    ], recRows, { onPick: (r) => { S.rank = r.rank; render(); } }));
    stage.appendChild(recSec);
  }

  /* ========================================================= L2 view */
  function laneRows() {
    const all = R().swimlane.lanes;
    if (S.laneFilter === 'aic') return all.filter((l) => l.kind === 'aic');
    if (S.laneFilter === 'aiv') return all.filter((l) => l.kind === 'aiv');
    return all;
  }

  function viewL2(stage) {
    const rank = R();
    const crit = rank.critical;
    const critSet = {};
    crit.tags.forEach((t) => { critSet[t] = 1; });

    /* --- critical path ribbon: the measured chain, on the real time axis --- */
    const ribSec = el('section');
    ribSec.appendChild(sectionHead('关键路径 · ' + crit.tags.length + ' 节点',
      '链上 span 合计 ' + us(crit.spanSum) + '，正向间隙 ' + us(crit.gapOnPath) + '，重叠 ' + us(crit.overlapOnPath),
      el('span', 'tc-readout', '走完 ' + us(rank.swimlane.spanUs))));
    const ribHost = el('div', 'tc-canvas-strip');
    const ribCanvas = el('canvas');
    ribHost.appendChild(ribCanvas);
    ribSec.appendChild(ribHost);
    stage.appendChild(ribSec);

    /* --- worker swimlane --- */
    const laneSec = el('section', 'tc-stage-fill');
    const legend = el('div', 'tc-legend');
    if (S.colorMode === 'engine') {
      [['aic', 'AIC'], ['aiv', 'AIV'], ['mix', 'MIX']].forEach((p) => {
        const s = el('span');
        const i = el('i');
        i.style.background = CMAP.colorForTask({ laneKind: p[0] }, 'engine');
        s.appendChild(i);
        s.appendChild(el('span', null, p[1]));
        legend.appendChild(s);
      });
    } else {
      rank.tasks.slice().sort((a, b) => b.span - a.span).slice(0, 8).forEach((t) => {
        const s = el('span');
        const i = el('i');
        i.style.background = CMAP.colorForTask({ colorKey: t.callable, label: t.callable }, 'semantic');
        s.appendChild(i);
        s.appendChild(el('span', null, t.callable));
        legend.appendChild(s);
      });
    }
    laneSec.appendChild(sectionHead('Chip swimlane · ' + rank.swimlane.lanes.length + ' core lane',
      laneRows().length + ' 泳道 · ' + rank.swimlane.blocks.reduce((a, b) => a + b.length, 0) + ' 块', legend));
    const laneHost = el('div', 'tc-canvas-host');
    const laneCanvas = el('canvas', 'tc-lanes');
    laneCanvas.tabIndex = 0;
    laneHost.appendChild(laneCanvas);
    laneSec.appendChild(laneHost);
    stage.appendChild(laneSec);

    /* ---------- rendering ---------- */
    const LBL = 66;
    function drawRibbon() {
      const w = ribHost.clientWidth || 800;
      const h = 74;
      const ctx = fitCanvas(ribCanvas, w, h);
      const plotX = LBL, plotW = Math.max(40, w - LBL - 10);
      drawTimeRuler(ctx, plotX, plotW, 14, S.t0, S.t1);
      ctx.font = '500 11px ' + cssVar('--font-sans');
      ctx.fillStyle = cssVar('--foreground-muted');
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText('CRIT PATH', 4, 40);
      ctx.fillText('GAP', 4, 62);
      const sx = (t) => plotX + ((t - S.t0) / (S.t1 - S.t0)) * plotW;
      let cursor = null;
      crit.nodes.forEach((node) => {
        const t = tasksOf[S.rank][node.tag];
        if (!t) return;
        const x = sx(t.start), x2 = sx(t.end);
        if (x2 < plotX || x > plotX + plotW) { cursor = t.end; return; }
        SW.drawTaskBar(ctx, {
          task: barTask(t, null, 'critical'),
          x: Math.max(plotX, x), y: 31, width: Math.max(2, Math.min(plotX + plotW, x2) - Math.max(plotX, x)), height: 18,
          baseColor: CMAP.colorForTask({ colorKey: t.callable, label: t.callable }, S.colorMode === 'engine' ? 'engine' : 'semantic'),
          isSelected: t.tag === S.task,
          isEmphasized: true,
          fontFamily: cssVar('--font-sans'),
        });
        /* gap markers are not task bars: page-local data-viz marks */
        if (cursor !== null && t.start > cursor) {
          const gx = sx(cursor), gx2 = sx(t.start);
          ctx.fillStyle = cssVar('--warning');
          ctx.globalAlpha = 0.5;
          ctx.fillRect(Math.max(plotX, gx), 56, Math.max(1, gx2 - gx), 8);
          ctx.globalAlpha = 1;
        } else if (cursor !== null && t.start < cursor) {
          const ox = sx(t.start), ox2 = sx(cursor);
          ctx.fillStyle = cssVar('--success');
          ctx.globalAlpha = 0.4;
          ctx.fillRect(Math.max(plotX, ox), 58, Math.max(1, ox2 - ox), 4);
          ctx.globalAlpha = 1;
        }
        cursor = t.end;
      });
    }

    const ROW_H = 8, ROW_GAP = 1;
    let laneLayout = [];
    function drawLanes() {
      const lanes = laneRows();
      const w = laneHost.clientWidth || 800;
      const plotX = LBL, plotW = Math.max(40, w - LBL - 10);
      const overlayRows = S.overlay === 'sched' ? rank.scheduler.lanes.length : 0;
      const readyH = S.overlay === 'ready' ? 46 : 0;
      const top = 20 + (overlayRows ? overlayRows * (ROW_H + ROW_GAP) + 8 : 0) + readyH;
      const h = top + lanes.length * (ROW_H + ROW_GAP) + 8;
      const ctx = fitCanvas(laneCanvas, w, Math.max(h, laneHost.clientHeight || h));
      const sx = (t) => plotX + ((t - S.t0) / (S.t1 - S.t0)) * plotW;
      drawTimeRuler(ctx, plotX, plotW, 12, S.t0, S.t1);
      ctx.font = '500 10px ' + cssVar('--font-sans');
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'left';
      laneLayout = [];

      /* AICPU scheduler lanes */
      if (overlayRows) {
        rank.scheduler.lanes.forEach((name, i) => {
          const y = 20 + i * (ROW_H + ROW_GAP);
          ctx.fillStyle = cssVar('--foreground-muted');
          ctx.fillText(name, 4, y + ROW_H / 2);
          rank.scheduler.blocks[i].forEach((b) => {
            const x = sx(b[0]), x2 = sx(b[0] + b[1]);
            if (x2 < plotX || x > plotX + plotW) return;
            ctx.fillStyle = CMAP.colorForLaneKind('aicpu');
            ctx.globalAlpha = b[2] === 'complete' ? 0.95 : b[2] === 'dispatch' ? 0.7 : 0.45;
            ctx.fillRect(Math.max(plotX, x), y, Math.max(0.8, Math.min(plotX + plotW, x2) - Math.max(plotX, x)), ROW_H);
            ctx.globalAlpha = 1;
          });
        });
      }

      /* ready-but-undispatched strip */
      if (readyH) {
        const y0 = 22, hh = readyH - 8;
        const peak = Math.max(rank.readyStat.peak.AIC, rank.readyStat.peak.AIV, 1);
        ctx.fillStyle = cssVar('--foreground-muted');
        ctx.fillText('READY', 4, y0 + hh / 2);
        [['AIC', 1, '--danger'], ['AIV', 2, '--warning']].forEach((cfg) => {
          ctx.beginPath();
          ctx.moveTo(plotX, y0 + hh);
          rank.readyQueue.forEach((q) => {
            const x = clamp(sx(q[0]), plotX, plotX + plotW);
            const y = y0 + hh - (q[cfg[1]] / peak) * hh;
            ctx.lineTo(x, y);
          });
          ctx.lineTo(plotX + plotW, y0 + hh);
          ctx.closePath();
          ctx.fillStyle = cssVar(cfg[2]);
          ctx.globalAlpha = 0.28;
          ctx.fill();
          ctx.globalAlpha = 1;
        });
        ctx.fillStyle = cssVar('--foreground-muted');
        ctx.textAlign = 'right';
        ctx.fillText('peak ' + peak, plotX + plotW - 2, y0 + 5);
        ctx.textAlign = 'left';
      }

      /* worker lanes */
      lanes.forEach((lane, i) => {
        const y = top + i * (ROW_H + ROW_GAP);
        const li = rank.swimlane.laneNames.indexOf(lane.name);
        laneLayout.push({ y: y, laneIdx: li, name: lane.name });
        ctx.fillStyle = lane.util > 60 ? cssVar('--foreground-secondary') : cssVar('--foreground-muted');
        ctx.fillText(lane.name, 4, y + ROW_H / 2);
        rank.swimlane.blocks[li].forEach((b) => {
          const t = rank.tasks[b[2]];
          if (S.critOnly && !critSet[t.tag]) return;
          const x = sx(b[0]), x2 = sx(b[0] + b[1]);
          if (x2 < plotX || x > plotX + plotW) return;
          const xa = Math.max(plotX, x);
          const wBar = Math.max(0.8, Math.min(plotX + plotW, x2) - xa);
          if (wBar < 2.2) {
            /* below task-bar legibility: draw a density tick, not a fake bar */
            ctx.fillStyle = CMAP.colorForTask(
              S.colorMode === 'engine' ? { laneKind: t.kind } : { colorKey: t.callable, label: t.callable },
              S.colorMode === 'engine' ? 'engine' : 'semantic');
            ctx.globalAlpha = t.tag === S.task ? 1 : 0.8;
            ctx.fillRect(xa, y, wBar, ROW_H);
            ctx.globalAlpha = 1;
            return;
          }
          SW.drawTaskBar(ctx, {
            task: barTask(t, b, lane.name),
            x: xa, y: y, width: wBar, height: ROW_H, radius: 1,
            baseColor: CMAP.colorForTask(
              S.colorMode === 'engine' ? { laneKind: t.kind } : { colorKey: t.callable, label: t.callable },
              S.colorMode === 'engine' ? 'engine' : 'semantic'),
            isSelected: t.tag === S.task,
            isRelated: t.tag !== S.task && !!critSet[t.tag] && !S.critOnly,
            fontFamily: cssVar('--font-sans'),
          });
        });
      });
    }

    function hitTest(event) {
      const rect = laneCanvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const plotX = LBL, plotW = Math.max(40, (laneHost.clientWidth || 800) - LBL - 10);
      if (x < plotX) return null;
      const t = S.t0 + ((x - plotX) / plotW) * (S.t1 - S.t0);
      const row = laneLayout.find((r) => y >= r.y - 1 && y <= r.y + ROW_H + 1);
      if (!row) return null;
      const tolerance = ((S.t1 - S.t0) / plotW) * 2;
      const blocks = rank.swimlane.blocks[row.laneIdx];
      for (let i = 0; i < blocks.length; i++) {
        const b = blocks[i];
        if (t >= b[0] - tolerance && t <= b[0] + b[1] + tolerance) {
          const task = rank.tasks[b[2]];
          if (S.critOnly && !critSet[task.tag]) continue;
          return { task: task, block: b, lane: row.name };
        }
      }
      return null;
    }

    attachTooltip(laneHost, laneCanvas, (event) => {
      const hit = hitTest(event);
      return hit ? barTask(hit.task, hit.block, hit.lane) : null;
    });
    laneCanvas.addEventListener('click', (event) => {
      const hit = hitTest(event);
      if (!hit) return;
      S.task = hit.task.tag;
      S.focus = 'task';
      renderInspector();
      drawLanes();
      drawRibbon();
    });

    /* horizontal pan by drag */
    let drag = null;
    laneCanvas.addEventListener('pointerdown', (event) => {
      if (event.button !== 0 || !event.shiftKey) return;
      drag = { x: event.clientX, t0: S.t0, t1: S.t1 };
      laneHost.classList.add('is-grabbing');
      laneCanvas.setPointerCapture(event.pointerId);
    });
    laneCanvas.addEventListener('pointermove', (event) => {
      if (!drag) return;
      const plotW = Math.max(40, (laneHost.clientWidth || 800) - LBL - 10);
      const dt = ((event.clientX - drag.x) / plotW) * (drag.t1 - drag.t0);
      setWindow(drag.t0 - dt, drag.t1 - dt);
      drawLanes(); drawRibbon(); renderToolbar();
    });
    const endDrag = () => { drag = null; laneHost.classList.remove('is-grabbing'); };
    laneCanvas.addEventListener('pointerup', endDrag);
    laneCanvas.addEventListener('pointercancel', endDrag);

    const redraw = () => { drawRibbon(); drawLanes(); };
    requestAnimationFrame(redraw);
    stage.__redraw = redraw;
    if (stage.__ro) stage.__ro.disconnect();
    stage.__ro = new ResizeObserver(() => redraw());
    stage.__ro.observe(laneHost);
    stage.__ro.observe(ribHost);
  }

  function setWindow(a, b) {
    const span = R().swimlane.spanUs;
    let t0 = a, t1 = b;
    const w = t1 - t0;
    if (w >= span) { S.t0 = 0; S.t1 = span; return; }
    if (t0 < 0) { t0 = 0; t1 = w; }
    if (t1 > span) { t1 = span; t0 = span - w; }
    S.t0 = t0; S.t1 = t1;
  }
  function zoom(factor) {
    const mid = (S.t0 + S.t1) / 2;
    const half = ((S.t1 - S.t0) / 2) * factor;
    setWindow(mid - half, mid + half);
  }

  /* ====================================================== L1 / L0 view */
  const DTYPES = [
    { id: 'int8', label: 'INT8', bytes: 1, mult: 512 },
    { id: 'bf16', label: 'BF16', bytes: 2, mult: 256 },
    { id: 'fp16', label: 'FP16', bytes: 2, mult: 256 },
    { id: 'fp32', label: 'FP32', bytes: 4, mult: 128 },
  ];
  const ACC_DTYPES = [
    { id: 'int32', label: 'INT32', bytes: 4 },
    { id: 'fp32', label: 'FP32', bytes: 4 },
  ];
  const dt = (id) => DTYPES.find((d) => d.id === id) || DTYPES[1];
  const adt = (id) => ACC_DTYPES.find((d) => d.id === id) || ACC_DTYPES[1];

  function defaultTile() {
    /* seeded from the AutoTileMatmulL0 dump: the qr_acc K-loop this run actually emitted
     * (Left INT8[16,64], Right INT8[64,512], Acc INT32[16,512], stage=2) */
    return { m: 16, n: 512, k: 64, ab: 'int8', acc: 'int32', live: 1, depth: 2 };
  }

  function tileBudget() {
    const T = S.tile || (S.tile = defaultTile());
    const ab = dt(T.ab), ac = adt(T.acc);
    const left = T.m * T.k * ab.bytes;
    const right = T.k * T.n * ab.bytes;
    const acc = T.m * T.n * ac.bytes * Math.max(1, T.live);
    const freeLR = D.budgets.Right ? D.budgets.Right.freeB : null;
    const accObserved = Math.max.apply(null, D.l0Tiles.filter((x) => x.mem === 'Acc').map((x) => x.bytes));
    const innermost = T.n * ab.bytes;
    const cacheLine = (D.hints.find((h) => h.cacheLineB) || {}).cacheLineB || 512;
    const need = Math.max(left, right) * T.depth;
    /* The run's own counter-example: qkv_proj_rope.py:375 asks for depth 2 with
     * 32768 B per stage against 65536 B free — exactly 100% — and MemoryReuse
     * still fits only one buffer, because co-resident tiles take part of that
     * space. So "exactly at the limit" is a fail in practice, not a pass. */
    const ratio = freeLR ? need / freeLR : null;
    return {
      T: T, ab: ab, ac: ac, left: left, right: right, acc: acc,
      freeLR: freeLR, accObserved: accObserved,
      innermost: innermost, cacheLine: cacheLine,
      depthNeed: need, depthRatio: ratio,
      depthState: ratio == null ? 'warn' : ratio > 1 ? 'fail' : ratio === 1 ? 'fail' : ratio > 0.8 ? 'warn' : 'pass',
      atLimit: ratio === 1,
      lineOk: innermost >= cacheLine,
      needElems: Math.ceil(cacheLine / ab.bytes),
    };
  }

  function viewL1(stage) {
    const rank = R();
    const t = curTask();
    const critSet = {};
    rank.critical.tags.forEach((x) => { critSet[x] = 1; });

    /* --- identity + measured split --- */
    const idSec = el('section');
    idSec.appendChild(sectionHead(t.callable, t.tag + ' · ' + t.kind.toUpperCase() + ' · task ' + t.id,
      el('span', 'tc-readout', critSet[t.tag] ? '在关键路径上（第 ' + (rank.critical.tags.indexOf(t.tag) + 1) + ' 节点）' : '不在关键路径上')));
    const kernelMean = t.kdurSum / t.blockCount;
    idSec.appendChild(tiles([
      { k: 'span', v: num(t.span, 1), u: 'us' },
      { k: '块 / 核', v: t.blockCount + ' / ' + t.coreCount, u: 'block_num ' + t.blockNum },
      { k: '块中位', v: num(t.durMed, 2), u: 'us' },
      { k: '块最长', v: num(t.durMax, 2), u: 'us', tone: t.imbalance > 3 ? 'warn' : null },
      { k: '离散度', v: num(t.imbalance, 2) + 'x', u: 'max / med', tone: t.imbalance > 3 ? 'bad' : t.imbalance > 2 ? 'warn' : 'good' },
      { k: 'kernel 均值', v: num(kernelMean, 2), u: 'us' },
      { k: 'setup 均值', v: num(t.setupMean, 2), u: pct(t.setupShare * 100, 0) + ' of block', tone: t.setupShare > 0.3 ? 'bad' : t.setupShare > 0.1 ? 'warn' : null },
      { k: 'AICPU 视角', v: num(t.svAicpuMean, 1), u: t.svOverhead != null ? '+' + num(t.svOverhead, 1) + ' us hand-off' : '', tone: t.svOverhead > 20 ? 'bad' : t.svOverhead > 5 ? 'warn' : null },
    ]));
    stage.appendChild(idSec);

    /* --- three measurements of the same block, side by side --- */
    const splitSec = el('section');
    splitSec.appendChild(sectionHead('一个块的三种口径',
      'kernel（核内计算）→ + local_setup（核上但非 kernel）→ + hand-off（AICPU dispatch→finish）'));
    const splitHost = el('div', 'tc-canvas-strip');
    const splitCanvas = el('canvas');
    splitHost.appendChild(splitCanvas);
    splitSec.appendChild(splitHost);
    stage.appendChild(splitSec);

    /* --- per-core block strip + duration distribution --- */
    const distSec = el('section');
    distSec.appendChild(sectionHead('块分布',
      t.blockCount + ' 块摊在 ' + t.coreCount + ' 核上，约 ' + num(t.blockCount / t.coreCount, 2) + ' 波；尾块决定 span'));
    const distHost = el('div', 'tc-canvas-host');
    const distCanvas = el('canvas');
    distHost.appendChild(distCanvas);
    distSec.appendChild(distHost);
    stage.appendChild(distSec);

    /* --- on-chip tile budget --- */
    const calcSec = el('section');
    calcSec.appendChild(sectionHead('片上预算试算',
      'Left / Right 是编译器选的 L0A / L0B staging，不是可独立调的 DSL Tile 预算',
      el('span', 'tc-readout', '上限取自本 run 的 MemoryReuse 报告')));
    calcSec.appendChild(renderCalc());
    stage.appendChild(calcSec);

    /* --- compiler hints, honestly unlinked --- */
    const hintSec = el('section');
    const mods = ['all'].concat(D.tileFiles.map((f) => f.file));
    hintSec.appendChild(sectionHead('该层可用的编译提示', D.hints.length + ' 条，按源码模块聚合',
      field('模块', select(mods.map((m) => ({ id: m, label: m === 'all' ? '全部模块' : m })), S.hintModule,
        (v) => { S.hintModule = v; render(); }))));
    const note = el('p', 'tc-note');
    note.innerHTML = '本 dump <strong>没有 kernel → 源码映射</strong>：perf hint 挂在源码位置上，IR 只保留 outline 后的 incore scope 名。'
      + '下面按模块聚合，对应关系需人工确认，不要当成自动归因。';
    hintSec.appendChild(note);
    const hintRows = D.hints
      .filter((h) => S.hintModule === 'all' || h.file === S.hintModule)
      .map((h, i) => Object.assign({ __i: i }, h));
    hintSec.appendChild(table([
      { label: 'Code', key: 'code', mono: true },
      { label: '位置', mono: true, cell: (h) => esc(h.file + ':' + h.line) },
      { label: '类型', cell: (h) => (h.kind === 'pipeline-depth' ? '流水深度' : '搬运粒度') },
      {
        label: '事实', cell: (h) => (h.kind === 'pipeline-depth'
          ? 'depth ' + h.reqDepth + ' → ' + h.fit + ' @' + h.unit + '（' + kb(h.perStageB) + '/stage，' + kb(h.freeB) + ' free）'
          : esc(h.op) + ' 末维 <span class="' + (h.innermostB < 128 ? 'bad' : 'warn') + '">' + h.innermostB + 'B</span> · tile ' + esc(h.dtype + '[' + h.tileShape + ']') + ' → ' + esc(h.mem)),
      },
      { label: '次数', key: 'occurrences', num: true },
    ], hintRows.slice(0, 80), {
      onPick: (h) => { S.view = 'compiler'; S.compilerTab = h.kind === 'pipeline-depth' ? 'depth' : 'granularity'; S.hintSite = h.file + ':' + h.line; render(); },
    }));
    if (hintRows.length > 80) {
      hintSec.appendChild(el('p', 'tc-note', '仅列出前 80 条，共 ' + hintRows.length + ' 条；完整列表在底部 Problems 面板。'));
    }
    stage.appendChild(hintSec);

    /* ---------- canvases ---------- */
    function drawSplit() {
      const w = splitHost.clientWidth || 700;
      const h = 82;
      const ctx = fitCanvas(splitCanvas, w, h);
      const rows = [
        { k: 'kernel', v: kernelMean, tone: '--success' },
        { k: '+ setup', v: t.durMean, tone: '--warning' },
        { k: '+ hand-off', v: t.svAicpuMean == null ? t.durMean : t.svAicpuMean, tone: '--danger' },
      ];
      const max = Math.max.apply(null, rows.map((r) => r.v));
      const x0 = 84, plotW = Math.max(40, w - x0 - 110);
      ctx.font = '500 11px ' + cssVar('--font-sans');
      ctx.textBaseline = 'middle';
      rows.forEach((r, i) => {
        const y = 14 + i * 22;
        ctx.textAlign = 'right';
        ctx.fillStyle = cssVar('--foreground-muted');
        ctx.fillText(r.k, x0 - 8, y + 7);
        ctx.fillStyle = cssVar('--surface-3');
        ctx.fillRect(x0, y, plotW, 14);
        ctx.fillStyle = cssVar(r.tone);
        ctx.globalAlpha = 0.85;
        ctx.fillRect(x0, y, Math.max(1, (r.v / max) * plotW), 14);
        ctx.globalAlpha = 1;
        ctx.textAlign = 'left';
        ctx.fillStyle = cssVar('--foreground');
        ctx.font = '500 11px ' + cssVar('--font-mono');
        ctx.fillText(num(r.v, 2) + ' us', x0 + plotW + 8, y + 7);
        ctx.font = '500 11px ' + cssVar('--font-sans');
      });
    }

    function drawDist() {
      const w = distHost.clientWidth || 700;
      /* per-core strip for this task only */
      const lanes = [];
      rank.swimlane.laneNames.forEach((name, li) => {
        const blocks = rank.swimlane.blocks[li].filter((b) => rank.tasks[b[2]].tag === t.tag);
        if (blocks.length) lanes.push({ name: name, blocks: blocks });
      });
      const ROW = 9, GAP = 1;
      const stripH = 22 + lanes.length * (ROW + GAP) + 10;
      const sorted = rank.swimlane.blocks.flat().filter((b) => rank.tasks[b[2]].tag === t.tag)
        .map((b) => b[1]).sort((a, b) => a - b);
      const histH = 96;
      distHost.style.height = Math.min(520, stripH + histH + 4) + 'px';
      const ctx = fitCanvas(distCanvas, w, stripH + histH);
      const x0 = 70, plotW = Math.max(40, w - x0 - 12);
      drawTimeRuler(ctx, x0, plotW, 12, t.start, t.end);
      const sx = (x) => x0 + ((x - t.start) / Math.max(1e-6, t.end - t.start)) * plotW;
      ctx.font = '500 10px ' + cssVar('--font-sans');
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'left';
      lanes.forEach((lane, i) => {
        const y = 22 + i * (ROW + GAP);
        ctx.fillStyle = cssVar('--foreground-muted');
        ctx.fillText(lane.name, 4, y + ROW / 2);
        lane.blocks.forEach((b) => {
          const x = sx(b[0]);
          const wBar = Math.max(1, sx(b[0] + b[1]) - x);
          if (wBar < 2.2) {
            ctx.fillStyle = CMAP.colorForTask({ colorKey: t.callable, label: t.callable }, 'semantic');
            ctx.fillRect(x, y, wBar, ROW);
            return;
          }
          SW.drawTaskBar(ctx, {
            task: barTask(t, b, lane.name),
            x: x, y: y, width: wBar, height: ROW, radius: 1,
            baseColor: CMAP.colorForTask({ colorKey: t.callable, label: t.callable }, 'semantic'),
            isEmphasized: b[1] >= t.durP90,
            fontFamily: cssVar('--font-sans'),
          });
        });
      });

      /* sorted block-duration profile: data-viz, not a task bar */
      const hy = stripH + 16;
      const hh = histH - 34;
      const max = sorted[sorted.length - 1] || 1;
      ctx.fillStyle = cssVar('--foreground-muted');
      ctx.font = '500 11px ' + cssVar('--font-sans');
      ctx.textAlign = 'left';
      ctx.fillText('块时长排序（' + sorted.length + ' 块，' + num(sorted[0], 2) + ' → ' + num(max, 2) + ' us）', 4, hy - 6);
      const bw = plotW / sorted.length;
      sorted.forEach((v, i) => {
        const bh = (v / max) * hh;
        ctx.fillStyle = v >= t.durP90 ? cssVar('--danger') : cssVar('--primary');
        ctx.globalAlpha = v >= t.durP90 ? 0.9 : 0.55;
        ctx.fillRect(x0 + i * bw, hy + hh - bh, Math.max(0.7, bw - 0.4), bh);
        ctx.globalAlpha = 1;
      });
      [['med', t.durMed, '--foreground-secondary'], ['p90', t.durP90, '--warning']].forEach((m) => {
        const y = hy + hh - (m[1] / max) * hh;
        ctx.strokeStyle = cssVar(m[2]);
        ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x0 + plotW, y); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = cssVar(m[2]);
        ctx.textAlign = 'right';
        ctx.fillText(m[0] + ' ' + num(m[1], 2), x0 - 4, y);
        ctx.textAlign = 'left';
      });
    }

    const redraw = () => { drawSplit(); drawDist(); };
    requestAnimationFrame(redraw);
    stage.__redraw = redraw;
    if (stage.__ro) stage.__ro.disconnect();
    stage.__ro = new ResizeObserver(() => redraw());
    stage.__ro.observe(distHost);
    stage.__ro.observe(splitHost);
  }

  function renderCalc() {
    const B = tileBudget();
    const wrap = el('div', 'tc-calc');

    const form = el('div', 'tc-calc-form');
    const numRow = (label, key, min, max, step) => {
      const row = el('div', 'tc-calc-row');
      row.appendChild(el('label', null, label));
      const i = el('input');
      i.type = 'number'; i.min = min; i.max = max; i.step = step || 1;
      i.value = S.tile[key];
      i.addEventListener('change', () => {
        S.tile[key] = clamp(parseInt(i.value, 10) || min, min, max);
        render();
      });
      row.appendChild(i);
      return row;
    };
    form.appendChild(numRow('M_tile', 'm', 8, 512, 8));
    form.appendChild(numRow('N_tile', 'n', 16, 1024, 16));
    form.appendChild(numRow('K_tile', 'k', 16, 1024, 16));
    const abRow = el('div', 'tc-calc-row');
    abRow.appendChild(el('label', null, 'A / B dtype'));
    abRow.appendChild(select(DTYPES.map((d) => ({ id: d.id, label: d.label })), S.tile.ab, (v) => { S.tile.ab = v; render(); }));
    form.appendChild(abRow);
    const accRow = el('div', 'tc-calc-row');
    accRow.appendChild(el('label', null, 'Acc dtype'));
    accRow.appendChild(select(ACC_DTYPES.map((d) => ({ id: d.id, label: d.label })), S.tile.acc, (v) => { S.tile.acc = v; render(); }));
    form.appendChild(accRow);
    form.appendChild(numRow('live acc', 'live', 1, 8, 1));
    const dRow = el('div', 'tc-calc-row');
    dRow.appendChild(el('label', null, 'pipeline stage'));
    dRow.appendChild(select([1, 2, 3, 4].map((n) => ({ id: String(n), label: String(n) })), String(S.tile.depth),
      (v) => { S.tile.depth = +v; render(); }));
    form.appendChild(dRow);
    const presets = el('div', 'tc-actions');
    presets.appendChild(btn('回到本 run 实测值', {
      size: 'sm', on: () => { S.tile = defaultTile(); render(); },
    }));
    form.appendChild(presets);
    wrap.appendChild(form);

    const out = el('div', 'tc-calc-out');
    const budget = el('div', 'tc-budget');
    const row = (name, bytes, cap, fx, tone) => {
      const r = el('div', 'tc-budget-row');
      r.appendChild(el('span', 'nm', name));
      r.appendChild(bar(cap ? bytes / cap : 0, tone));
      r.appendChild(el('span', 'fx', kb(bytes) + (cap ? ' / ' + kb(cap) : '')));
      return r;
    };
    budget.appendChild(row('Left (L0A)', B.left, B.freeLR, B.left > B.freeLR ? 'bad' : 'neutral'));
    budget.appendChild(row('Right (L0B)', B.right, B.freeLR, B.right > B.freeLR ? 'bad' : 'neutral'));
    budget.appendChild(row('Acc (L0C)', B.acc, B.accObserved, B.acc > B.accObserved ? 'warn' : 'good'));
    budget.appendChild(row('stage × max(L,R)', B.depthNeed, B.freeLR, B.depthState === 'pass' ? 'good' : B.depthState === 'warn' ? 'warn' : 'bad'));
    out.appendChild(budget);

    const verdict = el('div', 'tc-verdict');
    const vrow = (state, tag, html) => {
      const r = el('div', 'tc-verdict-row');
      r.dataset.state = state;
      r.appendChild(el('span', 'tag', tag));
      const p = el('p');
      p.innerHTML = html;
      r.appendChild(p);
      return r;
    };
    const depthMsg = B.depthState === 'pass'
      ? '占 ' + pct(B.depthRatio * 100, 0) + '，留有余量；但 free 是与同驻 tile 共享的，实际仍可能被挤掉。'
      : B.atLimit
        ? '正好用满 100%。本 run 的 ' + D.depthSites[0].module + ':' + D.depthSites[0].line
          + ' 就是这个配置，MemoryReuse 仍只放下 1 个——co-resident buffer 会分走这块空间，所以「刚好等于上限」等于放不下。'
        : B.depthState === 'warn'
          ? '占 ' + pct(B.depthRatio * 100, 0) + '，余量不足以容纳同驻 tile，很可能被降深度。'
          : '超出 ' + pct((B.depthRatio - 1) * 100, 0) + '，MemoryReuse 会把深度降到 1 并报一条 PH-MR-001。';
    verdict.appendChild(vrow(B.depthState, B.depthState === 'pass' ? 'depth' : 'depth ↓',
      'stage ' + B.T.depth + ' 需要 <strong>' + kb(B.depthNeed) + '</strong>，本 run 报告 L0A/L0B 可用 <strong>'
      + kb(B.freeLR) + '</strong>。' + depthMsg));
    verdict.appendChild(vrow(B.lineOk ? 'pass' : 'warn', B.lineOk ? 'cache line' : '末维不足',
      'B tile 末维 ' + B.T.n + ' × ' + B.ab.label + ' = <strong>' + B.innermost + 'B</strong>，cache line ' + B.cacheLine + 'B。'
      + (B.lineOk ? '满足一个整行。' : '需要把末维凑到 ' + B.ab.mult + ' 元素的倍数（≥ ' + B.needElems + ' 个 ' + B.ab.label + '）。')));
    verdict.appendChild(vrow(B.acc > B.accObserved ? 'warn' : 'pass', 'Acc',
      'Acc ≈ M × N × ' + B.ac.bytes + 'B × ' + B.T.live + ' = <strong>' + kb(B.acc) + '</strong>。本 run 出现过的最大 Acc tile 是 '
      + kb(B.accObserved) + '，因此容量至少这么大；dump 未报告 Acc 的确切上限，超过实测值只能视为待验证。'));
    out.appendChild(verdict);

    const obs = D.l0Tiles.slice(0, 8).map((x) => ({
      mem: x.mem, shape: x.dtype + '[' + x.rows + ',' + x.cols + ']',
      bytes: x.bytes, innermost: x.innermostB, n: x.n,
    }));
    out.appendChild(el('p', 'tc-note', '下面是 AutoTileMatmulL0 dump 里真实出现的 L0 tile，可直接对照上面的试算：'));
    out.appendChild(table([
      { label: '空间', key: 'mem', mono: true },
      { label: 'tile', key: 'shape', mono: true },
      { label: '字节', num: true, cell: (r) => kb(r.bytes) },
      { label: '末维', num: true, cell: (r) => (r.innermost >= 512 ? '<span class="ok">' : '<span class="warn">') + r.innermost + 'B</span>' },
      { label: '出现', key: 'n', num: true },
    ], obs, {
      onPick: (r) => {
        const m = r.shape.match(/\[(\d+),(\d+)\]/);
        if (!m) return;
        if (r.mem === 'Right') { S.tile.k = +m[1]; S.tile.n = +m[2]; }
        else if (r.mem === 'Left') { S.tile.m = +m[1]; S.tile.k = +m[2]; }
        else { S.tile.m = +m[1]; S.tile.n = +m[2]; }
        render();
      },
    }));
    wrap.appendChild(out);
    return wrap;
  }

  /* ==================================================== compiler view */
  function viewCompiler(stage) {
    const sub = el('section');
    sub.appendChild(sectionHead('编译降级', D.passes.length + ' 个 Pass dump · ' + D.hints.length + ' 条 perf hint',
      group('segmented-control segmented-control-muted', [
        { id: 'passes', label: 'Pass 轨迹' },
        { id: 'depth', label: '流水深度' },
        { id: 'granularity', label: '搬运粒度' },
      ], S.compilerTab, (v) => { S.compilerTab = v; render(); })));
    stage.appendChild(sub);

    if (S.compilerTab === 'passes') {
      const maxLines = Math.max.apply(null, D.passes.map((p) => p.lines));
      const sec = el('section');
      sec.appendChild(sectionHead('IR 规模与改写点', '行数变化是 Pass 实际改写量的代理指标；点行看它引入了什么'));
      const passTable = table([
        { label: '#', key: 'idx', num: true },
        { label: 'Pass', key: 'name', mono: true },
        { label: 'IR 行', key: 'lines', num: true },
        { label: '', cell: (p) => bar(p.lines / maxLines, Math.abs(p.delta) > 500 ? 'warn' : 'neutral') },
        {
          label: 'Δ', num: true,
          cell: (p) => (p.delta === 0 ? '<span style="opacity:.4">0</span>'
            : (p.delta > 0 ? '<span class="warn">+' : '<span class="ok">') + p.delta + '</span>'),
        },
        { label: 'pipeline', cell: (p) => p.counts.pipeline, num: true },
        { label: 'matmul', cell: (p) => p.counts.matmul, num: true },
        { label: 'Left/Right', cell: (p) => p.counts.left + ' / ' + p.counts.right, num: true },
        { label: 'Acc', cell: (p) => p.counts.acc, num: true },
      ], D.passes.map((p) => Object.assign({ __selected: p.idx === S.pass }, p)), {
        onPick: (p) => { S.pass = p.idx; S.focus = 'pass'; render(); },
      });
      passTable.dataset.tall = 'true';
      sec.appendChild(passTable);
      stage.appendChild(sec);

      const pair = D.irPairs[0];
      if (pair) {
        const irSec = el('section');
        irSec.appendChild(sectionHead('AutoTileMatmulL0 做了什么',
          pair.beforeFile + ' → ' + pair.afterFile + '（subject ' + pair.subject + '）'));
        const pre = el('pre', 'tc-term-static');
        pre.textContent = '- ' + pair.before.map((l) => l.trim()).join('\n- ')
          + '\n\n+ ' + pair.after.map((l) => l.trim()).join('\n+ ');
        const box = el('div', 'tc-canvas-strip');
        box.style.padding = 'var(--space-3)';
        box.style.maxHeight = '260px';
        box.style.overflow = 'auto';
        box.appendChild(pre);
        irSec.appendChild(box);
        irSec.appendChild(el('p', 'tc-note',
          '一条 pl.tile.matmul 被换成 K 循环 + pl.pipeline(stage=…) + Left / Right extract + matmul_acc。'
          + 'L0 分块和 ping-pong 是编译器给的，手工调 Tile 与它相互影响，不是独立旋钮。'));
        stage.appendChild(irSec);
      }
    }

    if (S.compilerTab === 'depth') {
      const sec = el('section');
      sec.appendChild(sectionHead('软流水深度回退',
        D.depthSites.length + ' 个源码点，' + D.hints.filter((h) => h.code === 'PH-MR-001').length + ' 条 PH-MR-001'));
      sec.appendChild(table([
        { label: '源码点', mono: true, cell: (s) => esc(s.module + ':' + s.line) },
        { label: '空间', cell: (s) => s.units.join(' / '), mono: true },
        { label: '组数', key: 'groupCount', num: true },
        { label: '请求 → 实得', num: true, cell: (s) => s.maxReqDepth + ' → <span class="bad">' + s.fittedDepth + '</span>' },
        { label: '每 stage', num: true, cell: (s) => kb(s.perStageB) },
        { label: '可用', num: true, cell: (s) => kb(s.freeB) },
        { label: '需求 / 可用', cell: (s) => bar((s.perStageB * s.maxReqDepth) / s.freeB, 'bad') },
      ], D.depthSites.map((s) => Object.assign({ __selected: s.key === S.hintSite }, s)), {
        onPick: (s) => { S.hintSite = s.key; S.focus = 'hint'; render(); },
      }));
      stage.appendChild(sec);

      const stageGroups = {};
      D.pipelineSites.forEach((p) => {
        const k = 'stage=' + p.stage;
        (stageGroups[k] = stageGroups[k] || []).push(p);
      });
      const sec2 = el('section');
      sec2.appendChild(sectionHead('IR 里请求的流水',
        D.pipelineSites.length + ' 个 pl.pipeline 站点（AutoTileMatmulL0 之后）· 前端手写 ' + D.dsl.pipeline + ' 处'));
      sec2.appendChild(table([
        { label: 'stage', key: 'k', mono: true },
        { label: '站点数', cell: (r) => r.n, num: true },
        { label: '循环次数（trip）', cell: (r) => esc(r.trips), mono: true },
        { label: '携带值', cell: (r) => r.carriers, num: true },
      ], Object.keys(stageGroups).sort().map((k) => ({
        k: k, n: stageGroups[k].length,
        trips: Array.from(new Set(stageGroups[k].map((p) => p.trip))).slice(0, 8).join(', '),
        carriers: Math.max.apply(null, stageGroups[k].map((p) => p.carriers)),
      })), {}));
      sec2.appendChild(el('p', 'tc-note',
        '这 ' + D.pipelineSites.length + ' 个站点里请求 stage 2 / 4 的那些，正是上表被降到 '
        + D.depthSites[0].fittedDepth + ' 的来源；加大 stage 会把需求抬高，回退只会更早发生。'));
      stage.appendChild(sec2);
    }

    if (S.compilerTab === 'granularity') {
      const cacheLine = (D.hints.find((h) => h.cacheLineB) || {}).cacheLineB || 512;
      const sec = el('section');
      sec.appendChild(sectionHead('搬运末维粒度',
        D.hints.filter((h) => h.code === 'PH001').reduce((a, h) => a + h.occurrences, 0) + ' 次命中 · cache line ' + cacheLine + 'B · backend ' + D.case.backend));
      sec.appendChild(table([
        { label: '模块', key: 'module', mono: true },
        { label: '源码点', key: 'siteCount', num: true },
        { label: '命中', key: 'occ', num: true },
        { label: '最小末维', num: true, cell: (f) => '<span class="' + (f.minB < 128 ? 'bad' : 'warn') + '">' + f.minB + 'B</span>' },
        { label: '距一行', cell: (f) => bar(f.minB / cacheLine, 'bad') },
      ], D.tileFiles, { onPick: (f) => { S.hintModule = f.file; S.view = 'l1'; render(); } }));
      stage.appendChild(sec);

      const sec2 = el('section');
      const memFilter = ['all', 'Vec', 'Mat', 'Acc'];
      sec2.appendChild(sectionHead('最差的源码点', '按末维字节升序',
        field('目标空间', select(memFilter.map((m) => ({ id: m, label: m === 'all' ? '全部' : m })), S.__mem || 'all',
          (v) => { S.__mem = v; render(); }))));
      const rows = D.tileSites.filter((s) => {
        const m = S.__mem || 'all';
        return m === 'all' || s.mems[m];
      }).slice(0, 60);
      sec2.appendChild(table([
        { label: '源码点', mono: true, cell: (s) => esc(s.module + ':' + s.line) },
        { label: '末维', num: true, cell: (s) => '<span class="' + (s.minB < 128 ? 'bad' : 'warn') + '">' + s.minB + 'B</span>' },
        { label: '算子', cell: (s) => Object.keys(s.ops).join(', '), mono: true },
        { label: '空间', cell: (s) => Object.keys(s.mems).join(', '), mono: true },
        { label: 'tile', cell: (s) => esc(s.shapes.join(' ')), mono: true },
        { label: '命中', key: 'occ', num: true },
      ], rows.map((s) => Object.assign({ __selected: s.key === S.hintSite }, s)), {
        onPick: (s) => { S.hintSite = s.key; S.focus = 'hint'; render(); },
      }));
      stage.appendChild(sec2);

      const guide = el('section');
      guide.appendChild(sectionHead('按 dtype 的末维目标', '凑满一个 ' + cacheLine + 'B cache line 所需的元素倍数'));
      guide.appendChild(table([
        { label: 'dtype', key: 'label', mono: true },
        { label: '每元素', num: true, cell: (d) => d.bytes + 'B' },
        { label: '末维元素倍数', num: true, cell: (d) => d.mult },
        { label: '对应字节', num: true, cell: (d) => kb(d.mult * d.bytes) },
      ], DTYPES, {}));
      stage.appendChild(guide);
    }
  }

  /* ========================================================= ISA view */
  function viewISA(stage) {
    const layoutPass = D.passes.find((p) => p.name === 'ResolveBackendOpLayouts');
    const spacePass = D.passes.find((p) => p.name === 'InferTileMemorySpace');
    const cacheLine = (D.hints.find((h) => h.cacheLineB) || {}).cacheLineB || 512;

    const have = el('section');
    have.appendChild(sectionHead('本 run 能支撑的 ISA / 布局结论', '来自 binary_context、Pass dump 与 L0 tile 清单'));
    have.appendChild(tiles([
      { k: 'platform', v: D.case.toolchain.platform },
      { k: 'pto-isa', v: D.case.toolchain.ptoIsaRevision.slice(0, 8), u: 'revision' },
      { k: 'runtime', v: D.case.toolchain.runtimeName.split('_')[0], u: D.case.toolchain.runtimeRevision.slice(0, 8) },
      { k: 'cache line', v: cacheLine, u: 'B' },
      { k: 'L0 tile 形状', v: D.l0Tiles.length, u: '种（AutoTileMatmulL0）' },
      { k: 'Left/Right 可用', v: kb(D.budgets.Right.freeB), u: 'MemoryReuse 报告' },
    ]));
    stage.appendChild(have);

    const lay = el('section');
    lay.appendChild(sectionHead('布局与内存空间分配', 'ResolveBackendOpLayouts +' + layoutPass.delta + ' 行，InferTileMemorySpace +' + spacePass.delta + ' 行'));
    lay.appendChild(table([
      { label: '空间', key: 'mem', mono: true },
      { label: 'tile 形状', cell: (r) => esc(r.shape), mono: true },
      { label: 'dtype', key: 'dtype', mono: true },
      { label: '字节', num: true, cell: (r) => kb(r.bytes) },
      { label: '末维', num: true, cell: (r) => (r.innermostB >= cacheLine ? '<span class="ok">' : '<span class="warn">') + r.innermostB + 'B</span>' },
      { label: '占 ' + kb(D.budgets.Right.freeB), cell: (r) => (r.mem === 'Acc' ? '—' : bar(r.bytes / D.budgets.Right.freeB, r.bytes > D.budgets.Right.freeB ? 'bad' : 'neutral')) },
      { label: '出现', key: 'n', num: true },
    ], D.l0Tiles.map((r) => Object.assign({ shape: '[' + r.rows + ',' + r.cols + ']' }, r)), {}));
    lay.appendChild(el('p', 'tc-note',
      'Left / Right 是编译器为 L0A / L0B staging 选的结果，Acc 是 L0C 累加器。它们的形状由 AutoTileMatmulL0 与布局 Pass 决定；'
      + '在 DSL 层能动的是 M / N / K 与 dtype，不是这三个空间本身。'));
    stage.appendChild(lay);

    const missing = el('section');
    missing.appendChild(sectionHead('这一层缺什么', '没有这些产物，指令级结论只能停在假设'));
    const box = el('div', 'tc-empty');
    box.appendChild(el('h3', null, '本 dump 不含 PTOAS / VPTO 级产物'));
    const p = el('p');
    p.innerHTML = '目录里只有 DSL → IR → 二进制的编译侧记录和运行侧 trace。要判断 TileLib 模板选择、向量指令排布、'
      + '寄存器压力下的重物化或 cycle cost model 的偏差，需要补以下产物再回到这一层：';
    box.appendChild(p);
    const ul = el('ul');
    [
      'PTOAS 的 TileLib 模板候选与选中记录（哪个模板、为什么、被拒的原因）',
      'VPTO scheduler 的指令排布报告：数据 / 内存 / 同步依赖、延迟、寄存器压力与重物化决策',
      'cycle cost model 的预测值，用于和本 run 的实测块时长对账',
      'PMU counter（Cube / Vec / MTE / FIXPIPE），并且必须单独建一条 PMU-on 基线',
    ].forEach((s) => ul.appendChild(el('li', null, s)));
    box.appendChild(ul);
    const p2 = el('p');
    p2.innerHTML = '已有的 <code>' + esc(D.case.toolchain.ptoIsaRevision.slice(0, 12)) + '</code> 与 <code>'
      + esc(D.case.toolchain.runtimeName) + '@' + esc(D.case.toolchain.runtimeRevision.slice(0, 12))
      + '</code> 足够把新产物对齐到同一工具链，但不能替代它们。';
    box.appendChild(p2);
    missing.appendChild(box);
    stage.appendChild(missing);
  }

  /* ======================================================== inspector */
  function inspectorSection(title, kicker) {
    const s = el('section', 'inspector-section');
    const h = el('div', 'inspector-section-head');
    h.appendChild(el('h3', 'inspector-section-title', title));
    if (kicker) h.appendChild(el('span', 'inspector-section-kicker', kicker));
    s.appendChild(h);
    return s;
  }

  function kv(pairs) {
    const d = el('dl', 'tc-kv');
    pairs.forEach((p) => {
      if (p[1] == null) return;
      d.appendChild(el('dt', null, p[0]));
      d.appendChild(el('dd', null, p[1]));
    });
    return d;
  }

  function renderInspector() {
    const host = $('#inspector');
    host.textContent = '';
    const title = $('[data-bind="inspectorTitle"]');
    const meta = $('[data-bind="inspectorMeta"]');

    const focus = S.focus || defaultFocus();
    if (focus === 'finding' && S.finding) renderFindingInspector(host, title, meta);
    else if (focus === 'hint' && S.hintSite) renderHintInspector(host, title, meta);
    else if (focus === 'pass') renderPassInspector(host, title, meta);
    else if (focus === 'run') renderRunInspector(host, title, meta);
    else renderTaskInspector(host, title, meta);

    host.appendChild(renderLedger());
  }

  /* the inspector follows the view unless the user pinned something else */
  function defaultFocus() {
    if (S.view === 'e2e' || S.view === 'isa') return 'run';
    if (S.view === 'compiler') return S.compilerTab === 'passes' ? 'pass' : (S.hintSite ? 'hint' : 'run');
    return 'task';
  }

  function renderRunInspector(host, title, meta) {
    title.textContent = D.case.program;
    meta.textContent = D.case.toolchain.platform;

    const s1 = inspectorSection('运行对象', D.case.runDir.slice(0, 16) + '…');
    s1.appendChild(kv([
      ['model', D.case.model],
      ['采集时间', D.case.capturedAt],
      ['ranks', D.case.ranks.join(', ') + ' · ' + D.case.device],
      ['核', D.case.numCores + '（AIC ' + D.case.aicCount + ' / AIV ' + D.case.aivCount + '）'],
      ['callables', String(D.case.callables)],
      ['绑定参数', String(D.case.params.length)],
      ['pto-isa', D.case.toolchain.ptoIsaRevision.slice(0, 12)],
      ['runtime', D.case.toolchain.runtimeName],
    ]));
    host.appendChild(s1);

    const s2 = inspectorSection('两卡对比', 'inv=' + TRACE_MATCH.rank0.inv + ' / ' + TRACE_MATCH.rank1.inv);
    const a = D.ranks.rank0, b = D.ranks.rank1;
    s2.appendChild(kv([
      ['device_wall', num(D.e2e.rank0[2]['chip.run.runner_run.device_wall'].us, 1) + ' / '
        + num(D.e2e.rank1[2]['chip.run.runner_run.device_wall'].us, 1) + ' us'],
      ['trace span', num(a.swimlane.spanUs, 1) + ' / ' + num(b.swimlane.spanUs, 1) + ' us'],
      ['AIC 占用', pct(a.occupancy.aicUtil) + ' / ' + pct(b.occupancy.aicUtil)],
      ['AIV 占用', pct(a.occupancy.aivUtil) + ' / ' + pct(b.occupancy.aivUtil)],
      ['关键路径', a.critical.tags.length + ' / ' + b.critical.tags.length + ' 节点'],
      ['调度器占用', pct(a.scheduler.perLaneUtil) + ' / ' + pct(b.scheduler.perLaneUtil)],
    ]));
    const card = el('div', 'inspector-soft-card is-warning');
    card.textContent = 'rank0 更慢却更闲：span 长 ' + num(a.swimlane.spanUs - b.swimlane.spanUs, 0)
      + ' us，核占用反而低 ' + num(b.occupancy.aicUtil - a.occupancy.aicUtil, 1)
      + ' 个百分点。差异在等待，不在单核算力——先分 rank 排查，再谈内核。';
    s2.appendChild(card);
    host.appendChild(s2);

    const s3 = inspectorSection('下一步', '瓶颈队列按实测影响排序');
    D.findings.slice(0, 3).forEach((f) => {
      s3.appendChild(btn(f.id + ' · ' + f.title, {
        size: 'sm',
        on: () => { S.finding = f.id; S.focus = 'finding'; applyFocus(f); render(); },
      }));
    });
    host.appendChild(s3);
  }

  function renderTaskInspector(host, title, meta) {
    const t = curTask();
    const rank = R();
    title.textContent = t.callable;
    meta.textContent = t.tag + ' · ' + S.rank;

    const s1 = inspectorSection('对象', t.kind.toUpperCase());
    s1.appendChild(kv([
      ['task id', t.id],
      ['callable', t.callable + '（funcId ' + t.funcId + '）'],
      ['ring / scope', 'r' + t.ring + ' · ' + t.scope + (t.earlyDispatch ? ' · early_dispatch' : '')],
      ['窗口', num(t.start, 1) + ' → ' + num(t.end, 1) + ' us'],
      ['块 / 核', t.blockCount + ' / ' + t.coreCount + '（block_num ' + t.blockNum + '）'],
      ['块时长', num(t.durMin, 2) + ' / ' + num(t.durMed, 2) + ' / ' + num(t.durP90, 2) + ' / ' + num(t.durMax, 2) + ' us'],
      ['kernel · setup', num(t.kdurSum / t.blockCount, 2) + ' · ' + num(t.setupMean, 2) + ' us'],
      ['AICPU 视角', t.svAicpuMean == null ? null : num(t.svAicpuMean, 1) + ' us（+' + num(t.svOverhead, 1) + '）'],
      ['前驱 / 后继', t.pred.length + ' / ' + t.succ.length],
    ]));
    host.appendChild(s1);

    if (t.args.length) {
      const s2 = inspectorSection('绑定张量', t.args.length + ' 个');
      const list = el('div', 'tc-evidence');
      t.args.slice(0, 8).forEach((a) => {
        const r = el('div', 'tc-evidence-row');
        r.appendChild(el('span', 'a', 'idx ' + a.idx + ' · ' + a.type));
        r.appendChild(el('span', 'l', a.dtype + ' [' + a.shape.join(', ') + ']'));
        list.appendChild(r);
      });
      s2.appendChild(list);
      if (t.args.length > 8) s2.appendChild(el('p', 'tc-note', '另有 ' + (t.args.length - 8) + ' 个参数。'));
      host.appendChild(s2);
    }

    const s3 = inspectorSection('依赖', 'fanin / fanout hints');
    const chips = el('div', 'tc-chipbar');
    t.pred.concat(t.succ).forEach((tag) => {
      const p = tasksOf[S.rank][tag];
      const b = btn(tag + (p ? ' · ' + p.callable : ''), {
        variant: 'ghost', size: 'sm',
        on: () => { if (p) { S.task = tag; S.focus = 'task'; render(); } },
      });
      if (!p) b.disabled = true;
      chips.appendChild(b);
    });
    s3.appendChild(chips);
    host.appendChild(s3);

    const rel = D.findings.filter((f) => f.focus && f.focus.task === t.tag);
    if (rel.length) {
      const s4 = inspectorSection('关联瓶颈', rel.length + ' 条');
      rel.forEach((f) => {
        s4.appendChild(btn(f.id + ' · ' + f.title, {
          size: 'sm', on: () => { S.finding = f.id; S.focus = 'finding'; applyFocus(f); render(); },
        }));
      });
      host.appendChild(s4);
    }

    const s5 = inspectorSection('所在核', '本任务涉及的泳道');
    const laneNames = {};
    rank.swimlane.laneNames.forEach((name, li) => {
      if (rank.swimlane.blocks[li].some((b) => rank.tasks[b[2]].tag === t.tag)) laneNames[name] = 1;
    });
    const laneStats = rank.swimlane.lanes.filter((l) => laneNames[l.name])
      .sort((a, b) => b.util - a.util).slice(0, 6);
    s5.appendChild(table([
      { label: 'lane', key: 'name', mono: true },
      { label: '占用', num: true, cell: (l) => pct(l.util) },
      { label: '', cell: (l) => bar(l.util / 100, l.util > 70 ? 'warn' : 'neutral') },
      { label: '最大空洞', num: true, cell: (l) => num(l.maxGap, 1) },
    ], laneStats, {}));
    host.appendChild(s5);
  }

  function renderFindingInspector(host, title, meta) {
    const f = findingById[S.finding];
    title.textContent = f.id + ' · ' + LEVEL_LABEL[f.level];
    meta.textContent = f.severity;

    const s1 = inspectorSection(f.title, f.metric);
    s1.appendChild(el('p', 'tc-note', f.claim));
    host.appendChild(s1);

    const s2 = inspectorSection('证据', f.evidence.length + ' 项');
    const list = el('div', 'tc-evidence');
    f.evidence.forEach((e) => {
      const r = el('div', 'tc-evidence-row');
      r.appendChild(el('span', 'a', e.artifact));
      r.appendChild(el('span', 'l', e.locator));
      r.appendChild(el('span', 'v', e.value));
      list.appendChild(r);
    });
    s2.appendChild(list);
    host.appendChild(s2);

    const s3 = inspectorSection('杠杆与护栏');
    s3.appendChild(el('div', 'inspector-soft-card is-info', '杠杆：' + f.lever));
    s3.appendChild(el('div', 'inspector-soft-card is-warning', '护栏：' + f.guardrail));
    s3.appendChild(el('div', 'inspector-soft-card', '复测：' + f.verify));
    host.appendChild(s3);

    host.appendChild(renderComposer(f));
  }

  function renderHintInspector(host, title, meta) {
    const depth = D.depthSites.find((s) => s.key === S.hintSite);
    const tile = D.tileSites.find((s) => s.key === S.hintSite);
    const site = depth || tile;
    if (!site) { renderTaskInspector(host, title, meta); return; }
    title.textContent = site.file + ':' + site.line;
    meta.textContent = depth ? 'PH-MR-001' : 'PH001';

    const s1 = inspectorSection('源码点', site.module);
    if (depth) {
      s1.appendChild(kv([
        ['pipeline 组', depth.groupCount + ' 组'],
        ['空间', depth.units.join(' / ')],
        ['请求深度', String(depth.maxReqDepth)],
        ['实得深度', String(depth.fittedDepth)],
        ['每 stage', kb(depth.perStageB)],
        ['可用', kb(depth.freeB)],
        ['需求 / 可用', num((depth.perStageB * depth.maxReqDepth) / depth.freeB, 2) + 'x'],
      ]));
      const list = el('div', 'tc-evidence');
      depth.groups.forEach((g) => {
        const r = el('div', 'tc-evidence-row');
        r.appendChild(el('span', 'a', 'group ' + g.group + ' @' + g.unit));
        r.appendChild(el('span', 'l', 'depth ' + g.reqDepth + ' → ' + g.fit));
        r.appendChild(el('span', 'v', kb(g.perStageB) + ' / stage，' + kb(g.freeB) + ' free'
          + (g.ownDepth ? '；单独放得下 depth ' + g.ownDepth : '')));
        list.appendChild(r);
      });
      s1.appendChild(list);
    } else {
      s1.appendChild(kv([
        ['算子', Object.keys(tile.ops).join(', ')],
        ['目标空间', Object.keys(tile.mems).join(', ')],
        ['dtype', Object.keys(tile.dtypes).join(', ')],
        ['tile', tile.shapes.join(' ')],
        ['最小末维', tile.minB + 'B'],
        ['建议', '≥ ' + tile.recB + 'B（cache line ' + tile.cacheLineB + 'B）'],
        ['命中', tile.occ + ' 次 / ' + tile.n + ' 条'],
      ]));
    }
    host.appendChild(s1);

    const s2 = inspectorSection('这条提示怎么用');
    s2.appendChild(el('p', 'tc-note', depth
      ? '先减少同驻 tile，再谈调 stage。这条提示说的是空间被 co-resident buffer 抢掉，而不是操作数本身太大——'
        + '所以直接调大 stage 只会让 MemoryReuse 再降一次，并多出一条一样的提示。'
      : '把末维凑到一个整 cache line。加大末维会同时抬高 L0 / UB 占用，可能触发流水深度回退，两项要一起看。'));
    host.appendChild(s2);
  }

  function renderPassInspector(host, title, meta) {
    const p = D.passes.find((x) => x.idx === S.pass) || D.passes[0];
    title.textContent = p.name;
    meta.textContent = '#' + p.idx;
    const s1 = inspectorSection('Pass', p.file);
    s1.appendChild(kv([
      ['IR 行数', String(p.lines)],
      ['Δ 行', (p.delta > 0 ? '+' : '') + p.delta],
      ['pl.pipeline', String(p.counts.pipeline)],
      ['tile.matmul', String(p.counts.matmul)],
      ['pl.spmd', String(p.counts.spmd)],
      ['pl.range', String(p.counts.range)],
      ['Mem.Left / Right', p.counts.left + ' / ' + p.counts.right],
      ['Mem.Acc / Vec', p.counts.acc + ' / ' + p.counts.vec],
    ]));
    host.appendChild(s1);

    const prev = D.passes.find((x) => x.idx === p.idx - 1);
    if (prev) {
      const s2 = inspectorSection('相对上一个 Pass', prev.name);
      const diffs = [
        ['pl.pipeline', p.counts.pipeline - prev.counts.pipeline],
        ['tile.matmul', p.counts.matmul - prev.counts.matmul],
        ['Mem.Left', p.counts.left - prev.counts.left],
        ['Mem.Right', p.counts.right - prev.counts.right],
        ['Mem.Acc', p.counts.acc - prev.counts.acc],
        ['pl.range', p.counts.range - prev.counts.range],
      ].filter((d) => d[1] !== 0);
      if (diffs.length) s2.appendChild(kv(diffs.map((d) => [d[0], (d[1] > 0 ? '+' : '') + d[1]])));
      else s2.appendChild(el('p', 'tc-note', '这一 Pass 没有改变上述结构计数，行数变化 ' + ((p.delta > 0 ? '+' : '') + p.delta) + '。'));
      host.appendChild(s2);
    }
  }

  /* ------------------------------------------------------- experiment */
  function renderComposer(f) {
    const open = openExperiment();
    const s = inspectorSection('实验台账', open ? '已有 1 个进行中' : '每轮只验证一个假设');
    if (open && open.findingId !== f.id) {
      const warn = el('div', 'inspector-soft-card is-warning');
      warn.textContent = '当前有进行中的实验 ' + open.id + '（' + open.title + '）。先结论它，再开下一个；'
        + '同时改两处就无法把性能变化归因到任一改动。';
      s.appendChild(warn);
      s.appendChild(btn('查看 ' + open.id, {
        size: 'sm',
        on: () => { if (open.findingId) { S.finding = open.findingId; S.focus = 'finding'; render(); } },
      }));
      return s;
    }
    if (open && open.findingId === f.id) {
      s.appendChild(renderStepper(open));
      return s;
    }
    const form = el('div', 'tc-form');
    const hyp = el('textarea');
    hyp.rows = 3;
    hyp.value = f.lever;
    const chg = el('input');
    chg.type = 'text';
    chg.placeholder = '例如：hc_post.py:51 把 co-live tile 从 5 组降到 2 组';
    const l1 = el('label');
    l1.appendChild(el('span', null, '假设（改什么、为什么会变好）'));
    l1.appendChild(hyp);
    const l2 = el('label');
    l2.appendChild(el('span', null, '改动位置'));
    l2.appendChild(chg);
    form.appendChild(l1);
    form.appendChild(l2);
    const acts = el('div', 'tc-actions');
    acts.appendChild(btn('开始实验', {
      variant: 'solid', size: 'sm',
      on: () => {
        ledgerSeq += 1;
        S.ledger.push({
          id: 'E' + ledgerSeq,
          state: 'open',
          findingId: f.id,
          title: f.id + ' · ' + f.title,
          hypothesis: hyp.value.trim() || f.lever,
          change: chg.value.trim() || '（未填写改动位置）',
          correctness: null, perf: null, keep: null,
          verify: f.verify, guardrail: f.guardrail,
        });
        render();
      },
    }));
    form.appendChild(acts);
    s.appendChild(form);
    return s;
  }

  function renderStepper(row) {
    const wrap = el('div', 'tc-form');
    const steps = el('div', 'tc-ledger-steps');
    const step = (k, v, done) => {
      const r = el('div', 'tc-ledger-step');
      r.dataset.done = done ? 'true' : 'false';
      r.appendChild(el('span', 'k', k));
      r.appendChild(el('span', 'v', v));
      return r;
    };
    steps.appendChild(step('假设', row.hypothesis, true));
    steps.appendChild(step('改动', row.change, true));
    steps.appendChild(step('正确性', row.correctness || '待记录：先过精度阈值，再谈性能', !!row.correctness));
    steps.appendChild(step('性能', row.perf || '待记录：' + row.verify, !!row.perf));
    steps.appendChild(step('结论', row.keep || '待决定：保留或回退', !!row.keep));
    wrap.appendChild(steps);

    const acts = el('div', 'tc-actions');
    if (!row.correctness) {
      const inp = el('input');
      inp.type = 'text';
      inp.placeholder = '正确性结果（精度阈值 / 对比基准）';
      const lb = el('label');
      lb.appendChild(el('span', null, '记录正确性'));
      lb.appendChild(inp);
      wrap.appendChild(lb);
      acts.appendChild(btn('记录正确性', {
        size: 'sm', variant: 'solid',
        on: () => { row.correctness = inp.value.trim() || '通过（未填写细节）'; render(); },
      }));
    } else if (!row.perf) {
      const inp = el('input');
      inp.type = 'text';
      inp.placeholder = '复测结果，例如 device_wall 5132.8 → ? us';
      const lb = el('label');
      lb.appendChild(el('span', null, '记录性能（' + row.verify + '）'));
      lb.appendChild(inp);
      wrap.appendChild(lb);
      acts.appendChild(btn('记录性能', {
        size: 'sm', variant: 'solid',
        on: () => { row.perf = inp.value.trim() || '（未填写复测数值）'; render(); },
      }));
    } else {
      const guard = el('div', 'inspector-soft-card is-warning');
      guard.textContent = '护栏复核：' + row.guardrail;
      wrap.appendChild(guard);
      acts.appendChild(btn('保留', { size: 'sm', variant: 'solid', on: () => { row.keep = '保留'; row.state = 'kept'; render(); } }));
      acts.appendChild(btn('回退', { size: 'sm', on: () => { row.keep = '回退'; row.state = 'reverted'; render(); } }));
    }
    acts.appendChild(btn('放弃这轮', {
      size: 'sm', variant: 'ghost',
      on: () => { S.ledger = S.ledger.filter((r) => r !== row); render(); },
    }));
    wrap.appendChild(acts);
    return wrap;
  }

  function renderLedger() {
    const s = inspectorSection('台账', S.ledger.length + ' 条');
    const list = el('div', 'tc-ledger');
    S.ledger.slice().reverse().forEach((row) => {
      const item = el('div', 'tc-ledger-item');
      item.dataset.state = row.state;
      const hd = el('div', 'hd');
      hd.appendChild(el('span', 'id', row.id));
      hd.appendChild(el('span', 'st', row.state));
      item.appendChild(hd);
      item.appendChild(el('span', 'ti', row.title));
      const steps = el('div', 'tc-ledger-steps');
      [['假设', row.hypothesis], ['改动', row.change], ['正确性', row.correctness],
        ['性能', row.perf], ['结论', row.keep]].forEach((p) => {
        const r = el('div', 'tc-ledger-step');
        r.dataset.done = p[1] ? 'true' : 'false';
        r.appendChild(el('span', 'k', p[0]));
        r.appendChild(el('span', 'v', p[1] || '—'));
        steps.appendChild(r);
      });
      item.appendChild(steps);
      if (row.findingId) {
        item.appendChild(btn('回到 ' + row.findingId, {
          variant: 'ghost', size: 'sm',
          on: () => { S.finding = row.findingId; S.focus = 'finding'; applyFocus(findingById[row.findingId]); render(); },
        }));
      }
      list.appendChild(item);
    });
    s.appendChild(list);
    return s;
  }

  /* ======================================================= bottom dock */
  function renderDock() {
    const body = $('#dockBody');
    body.textContent = '';
    const rank = R();
    $('[data-bind="dockMeta"]').textContent = S.rank + ' · 与上方时间轴同窗口 '
      + num(S.t0, 0) + '–' + num(S.t1, 0) + ' us';
    const modeHost = $('#dockMode');
    modeHost.textContent = '';
    modeHost.appendChild(group('segmented-control segmented-control-muted', [
      { id: 'sched', label: 'AICPU 调度' },
      { id: 'ready', label: 'Ready queue' },
      { id: 'lanes', label: '核占用' },
    ], S.dockMode, (v) => { S.dockMode = v; renderDock(); }));

    if (S.dockMode === 'lanes') {
      const rows = rank.swimlane.lanes.slice().sort((a, b) => b.util - a.util);
      body.appendChild(table([
        { label: 'lane', key: 'name', mono: true },
        { label: '类型', key: 'kind', mono: true },
        { label: '块', key: 'blocks', num: true },
        { label: '占用', num: true, cell: (l) => pct(l.util) },
        { label: '', cell: (l) => bar(l.util / 100, l.util > 70 ? 'warn' : l.util < 30 ? 'bad' : 'neutral') },
        { label: '空洞数', key: 'nGap', num: true },
        { label: '最大空洞', num: true, cell: (l) => num(l.maxGap, 1) },
        { label: '首块 → 末块', cell: (l) => num(l.first, 0) + ' → ' + num(l.last, 0), num: true },
      ], rows, {}));
      return;
    }

    if (S.dockMode === 'sched') {
      const phases = Object.keys(rank.scheduler.phases)
        .map((k) => Object.assign({ phase: k }, rank.scheduler.phases[k]))
        .sort((a, b) => b.us - a.us);
      const maxUs = phases[0].us;
      const head = el('div', 'tc-tiles');
      head.style.flex = '0 0 auto';
      const t = tiles([
        { k: '调度线程', v: rank.scheduler.lanes.length },
        { k: '合计 busy', v: num(rank.scheduler.busy, 0), u: 'us' },
        { k: '单线程平均占用', v: pct(rank.scheduler.perLaneUtil), tone: rank.scheduler.perLaneUtil > 40 ? 'warn' : null },
        { k: 'orchestrator submit', v: rank.orchestrator.count, u: num(rank.orchestrator.busy, 1) + ' us' },
        { k: 'hb_violation', v: rank.hbViolations.length, u: '对', tone: rank.hbViolations.length ? 'warn' : 'good' },
      ]);
      body.appendChild(t);
      body.appendChild(table([
        { label: 'phase', key: 'phase', mono: true },
        { label: '段数', key: 'n', num: true },
        { label: '总时长', num: true, cell: (p) => num(p.us, 1) },
        { label: '', cell: (p) => bar(p.us / maxUs, p.phase === 'complete' ? 'warn' : 'neutral') },
        { label: '处理任务', key: 'tasks', num: true },
        { label: 'us / 任务', num: true, cell: (p) => (p.usPerTask == null ? '—' : num(p.usPerTask, 3)) },
      ], phases, {}));
      body.appendChild(el('p', 'tc-note',
        'complete 段占 ' + pct((rank.scheduler.phases.complete.us / rank.scheduler.busy) * 100)
        + ' 的调度开销。单任务代价已经很小（' + num(rank.scheduler.phases.complete.usPerTask, 3)
        + ' us），要降总量只能减少任务次数：合核、折迭代、或用 pl.spmd 一次 fan-out 多块。'));
      return;
    }

    /* ready queue */
    const host = el('div', 'tc-canvas-strip');
    host.style.flex = '0 0 auto';
    const canvas = el('canvas');
    host.appendChild(canvas);
    body.appendChild(host);
    body.appendChild(tiles([
      { k: 'AIC ready>0', v: pct(rank.readyStat.busyShare.AIC), u: num(rank.readyStat.busyTime.AIC, 0) + ' us', tone: 'warn' },
      { k: 'AIV ready>0', v: pct(rank.readyStat.busyShare.AIV), u: num(rank.readyStat.busyTime.AIV, 0) + ' us' },
      { k: 'MIX ready>0', v: pct(rank.readyStat.busyShare.MIX), u: num(rank.readyStat.busyTime.MIX, 0) + ' us' },
      { k: '峰值', v: rank.readyStat.peak.AIC + ' / ' + rank.readyStat.peak.AIV + ' / ' + rank.readyStat.peak.MIX, u: 'AIC / AIV / MIX' },
      { k: 'AIC 核占用', v: pct(rank.occupancy.aicUtil), tone: rank.occupancy.aicUtil < 40 ? 'bad' : null },
      { k: 'AIV 核占用', v: pct(rank.occupancy.aivUtil) },
    ]));
    body.appendChild(el('p', 'tc-note',
      'ready > 0 且核占用低，说明队列里有活但没派出去——这是 L2 侧的可攻点。'
      + '但提前 dispatch、改依赖、延后非关键任务都可能以吞吐换时延，两个指标都要报。'));

    const draw = () => {
      const w = host.clientWidth || 700;
      const h = 92;
      const ctx = fitCanvas(canvas, w, h);
      const x0 = 46, plotW = Math.max(40, w - x0 - 12), y0 = 18, hh = h - 40;
      drawTimeRuler(ctx, x0, plotW, 10, S.t0, S.t1);
      const peak = Math.max(rank.readyStat.peak.AIC, rank.readyStat.peak.AIV, 1);
      const sx = (t) => x0 + ((t - S.t0) / (S.t1 - S.t0)) * plotW;
      [['AIC', 1, '--danger'], ['AIV', 2, '--warning'], ['MIX', 3, '--accent']].forEach((cfg) => {
        ctx.beginPath();
        ctx.moveTo(x0, y0 + hh);
        rank.readyQueue.forEach((q) => {
          ctx.lineTo(clamp(sx(q[0]), x0, x0 + plotW), y0 + hh - (q[cfg[1]] / peak) * hh);
        });
        ctx.lineTo(x0 + plotW, y0 + hh);
        ctx.closePath();
        ctx.fillStyle = cssVar(cfg[2]);
        ctx.globalAlpha = 0.3;
        ctx.fill();
        ctx.globalAlpha = 0.9;
        ctx.strokeStyle = cssVar(cfg[2]);
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.globalAlpha = 1;
      });
      ctx.font = '500 11px ' + cssVar('--font-sans');
      ctx.fillStyle = cssVar('--foreground-muted');
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(peak), x0 - 6, y0 + 4);
      ctx.fillText('0', x0 - 6, y0 + hh);
    };
    requestAnimationFrame(draw);
    if (body.__ro) body.__ro.disconnect();
    body.__ro = new ResizeObserver(draw);
    body.__ro.observe(host);
  }

  /* -------------------------------------------------------- terminal */
  const TERM_TABS = [
    { id: 'problems', label: 'Problems' },
    { id: 'output', label: 'Output' },
    { id: 'artifacts', label: 'Artifacts' },
  ];

  function renderTerminal() {
    const tabs = $('#terminalTabs');
    tabs.textContent = '';
    TERM_TABS.forEach((t) => {
      const b = el('span', 'pto-ide-frame__terminal-tab' + (t.id === S.termTab ? ' is-selected' : ''),
        t.label + (t.id === 'problems' ? ' (' + D.hints.length + ')' : ''));
      b.tabIndex = 0;
      b.addEventListener('click', () => { S.termTab = t.id; renderTerminal(); });
      tabs.appendChild(b);
    });

    const body = $('#terminalBody');
    body.textContent = '';

    if (S.termTab === 'problems') {
      const list = el('div', 'tc-term-list');
      D.hints.forEach((h) => {
        const row = el('button', 'tc-term-row');
        row.type = 'button';
        row.dataset.sev = h.kind === 'pipeline-depth' ? 'warn' : 'info';
        row.appendChild(el('span', 'sev', h.code));
        row.appendChild(el('span', 'loc', h.file + ':' + h.line));
        row.appendChild(el('span', 'msg', h.kind === 'pipeline-depth'
          ? 'MemoryReuse: depth ' + h.reqDepth + ' → ' + h.fit + ' @' + h.unit + ' group ' + h.group
            + ' (' + h.perStageB + ' B/stage, ' + h.freeB + ' B free)'
          : 'TileInnermostDimGranularity: ' + h.op + ' innermost ' + h.innermostB + 'B, tile '
            + h.dtype + '[' + h.tileShape + '] → ' + h.mem + ', recommended ≥ ' + h.recB + 'B'));
        row.addEventListener('click', () => {
          S.view = 'compiler';
          S.compilerTab = h.kind === 'pipeline-depth' ? 'depth' : 'granularity';
          S.hintSite = h.file + ':' + h.line;
          S.focus = 'hint';
          render();
        });
        list.appendChild(row);
      });
      body.appendChild(list);
      return;
    }

    if (S.termTab === 'output') {
      const lines = [];
      Object.keys(D.e2e).forEach((rank) => {
        Object.keys(D.e2e[rank]).forEach((inv) => {
          lines.push('# ' + rank + ' inv=' + inv);
          Object.keys(D.e2e[rank][inv]).forEach((name) => {
            const sp = D.e2e[rank][inv][name];
            lines.push('  [' + sp.clk.padEnd(6) + '] ' + name.padEnd(46) + ' dur=' + sp.us.toFixed(2) + ' us');
          });
        });
      });
      const pre = el('pre', 'tc-term-static', lines.join('\n'));
      body.appendChild(pre);
      return;
    }

    const rows = [
      ['distributed_meta.json', D.case.params.length + ' 个绑定参数，schema ' + D.case.metaSchema],
      ['dfx_outputs/rank0/d0/merged_swimlane_*.json', 'Worker + Scheduler view，' + D.ranks.rank0.tasks.length + ' 任务 / '
        + D.ranks.rank0.swimlane.blocks.reduce((a, b) => a + b.length, 0) + ' 块'],
      ['dfx_outputs/rank1/d0/merged_swimlane_*.json', D.ranks.rank1.tasks.length + ' 任务 / '
        + D.ranks.rank1.swimlane.blocks.reduce((a, b) => a + b.length, 0) + ' 块'],
      ['dfx_outputs/rank*/d0/deps.json', 'block_num / scope / early_dispatch / 绑定张量'],
      ['dfx_outputs/rank*/d0/name_map.json', D.case.callables + ' 个 callable'],
      ['dfx_outputs/rank*/d0/host.*.log', 'STRACE host span（bind / runner_run / device_wall / sched）'],
      ['report/perf_hints.log', D.hints.length + ' 条 perf hint（PH001 / PH-MR-001）'],
      ['passes_dump/', D.passes.length + ' 个 IR dump，' + D.passes[0].lines + ' → ' + D.passes[D.passes.length - 1].lines + ' 行'],
      ['next_levels/decode_csa_test/cache/', 'binary_context + ' + D.case.callables + ' 个 incore 二进制'],
      ['next_levels/.../binary_context.json', 'platform ' + D.case.toolchain.platform + ' · pto-isa '
        + D.case.toolchain.ptoIsaRevision.slice(0, 12) + ' · runtime ' + D.case.toolchain.runtimeName],
    ];
    body.appendChild(table([
      { label: '产物', cell: (r) => esc(r[0]), mono: true },
      { label: '内容', cell: (r) => esc(r[1]) },
    ], rows, {}));
  }

  /* ========================================================= chrome */
  function renderTabs() {
    const host = $('#levelTabs');
    host.textContent = '';
    LEVELS.forEach((l) => {
      const b = el('button', 'tab-control-item' + (l.id === S.view ? ' is-selected' : ''), l.label);
      b.type = 'button';
      b.title = l.hint;
      b.setAttribute('aria-pressed', l.id === S.view ? 'true' : 'false');
      b.addEventListener('click', () => { S.view = l.id; S.focus = null; render(); });
      host.appendChild(b);
    });
  }

  function renderToolbar() {
    const host = $('#viewToolbar');
    host.textContent = '';
    const level = LEVELS.find((l) => l.id === S.view);
    host.appendChild(el('span', 'pto-ide-frame__pane-title', level.label));
    host.appendChild(el('span', 'pto-ide-frame__pane-meta', level.hint));

    const right = el('div', 'tc-toolbar-right');

    if (S.view === 'l2') {
      right.appendChild(field('泳道', select([
        { id: 'all', label: '全部 ' + R().swimlane.lanes.length },
        { id: 'aic', label: 'AIC ' + D.case.aicCount },
        { id: 'aiv', label: 'AIV ' + D.case.aivCount },
      ], S.laneFilter, (v) => { S.laneFilter = v; render(); })));
      right.appendChild(field('着色', select([
        { id: 'semantic', label: '按算子' },
        { id: 'engine', label: '按引擎' },
      ], S.colorMode, (v) => { S.colorMode = v; render(); })));
      right.appendChild(field('叠加', select([
        { id: 'sched', label: 'AICPU 调度' },
        { id: 'ready', label: 'Ready queue' },
        { id: 'none', label: '无' },
      ], S.overlay, (v) => { S.overlay = v; render(); })));
      right.appendChild(btn('只看关键路径', {
        size: 'sm', selected: S.critOnly,
        on: () => { S.critOnly = !S.critOnly; render(); },
      }));
      const zoomGroup = el('div', 'toolbar-control');
      zoomGroup.appendChild(btn('−', { variant: 'ghost', size: 'icon', title: '缩小', on: () => { zoom(2); redrawStage(); renderToolbar(); renderDock(); } }));
      zoomGroup.appendChild(btn('Fit', { variant: 'ghost', size: 'sm', on: () => { S.t0 = 0; S.t1 = R().swimlane.spanUs; redrawStage(); renderToolbar(); renderDock(); } }));
      zoomGroup.appendChild(btn('+', { variant: 'ghost', size: 'icon', title: '放大', on: () => { zoom(0.5); redrawStage(); renderToolbar(); renderDock(); } }));
      right.appendChild(zoomGroup);
      right.appendChild(el('span', 'tc-readout', num(S.t0, 0) + '–' + num(S.t1, 0) + ' us · shift+拖动平移'));
    }

    if (S.view === 'l1') {
      const ordered = R().tasks.slice().sort((a, b) => b.span - a.span);
      right.appendChild(field('kernel', select(ordered.map((t) => ({
        id: t.tag, label: t.callable + ' · ' + t.tag + '（' + num(t.span, 0) + ' us）',
      })), S.task, (v) => { S.task = v; S.focus = 'task'; render(); })));
    }

    if (S.view === 'e2e' || S.view === 'l2' || S.view === 'l1') {
      right.appendChild(field('rank', select(Object.keys(D.ranks).map((r) => ({ id: r, label: r })), S.rank,
        (v) => {
          S.rank = v;
          S.t0 = 0; S.t1 = R().swimlane.spanUs;
          if (!tasksOf[S.rank][S.task]) S.task = R().tasks[0].tag;
          render();
        })));
    }

    host.appendChild(right);
  }

  function renderExplorer() {
    const tree = $('#runTree');
    tree.textContent = '';
    const row = (depth, label, metaText, opts) => {
      const o = opts || {};
      const b = el('button', 'tc-tree-row' + (o.selected ? ' is-selected' : ''));
      b.type = 'button';
      b.dataset.depth = depth;
      b.appendChild(el('span', 'n', label));
      if (metaText) b.appendChild(el('span', 'm', metaText));
      if (o.on) b.addEventListener('click', o.on);
      else b.disabled = true;
      tree.appendChild(b);
      return b;
    };
    row(0, D.case.program, D.case.backend);
    Object.keys(D.ranks).forEach((rank) => {
      const m = TRACE_MATCH[rank];
      row(1, rank + ' / ' + D.case.device, us(D.ranks[rank].swimlane.spanUs, 0), {
        selected: rank === S.rank,
        on: () => {
          S.rank = rank;
          S.t0 = 0; S.t1 = R().swimlane.spanUs;
          if (!tasksOf[S.rank][S.task]) S.task = R().tasks[0].tag;
          render();
        },
      });
      Object.keys(D.e2e[rank]).forEach((inv) => {
        row(2, 'inv=' + inv + (m.inv === +inv ? ' · traced' : ''),
          us(D.e2e[rank][inv]['chip.run.runner_run.device_wall'].us, 0), {
            on: () => { S.rank = rank; S.view = 'e2e'; render(); },
          });
      });
    });
    row(0, 'artifacts', D.passes.length + ' passes');
    row(1, 'report/perf_hints.log', D.hints.length, {
      on: () => { S.view = 'compiler'; S.compilerTab = 'depth'; render(); },
    });
    row(1, 'passes_dump/', D.passes.length, {
      on: () => { S.view = 'compiler'; S.compilerTab = 'passes'; render(); },
    });
    row(1, 'binary_context.json', D.case.toolchain.platform, {
      on: () => { S.view = 'isa'; render(); },
    });

    $('[data-bind="explorerMeta"]').textContent = D.case.runDir.slice(0, 18) + '…';

    /* findings queue */
    const filterHost = $('#findingFilter');
    filterHost.textContent = '';
    const counts = { all: D.findings.length };
    D.findings.forEach((f) => { counts[f.level] = (counts[f.level] || 0) + 1; });
    [{ id: 'all', label: '全部' }].concat(LEVELS.filter((l) => counts[l.id]).map((l) => ({ id: l.id, label: l.label })))
      .forEach((o) => {
        filterHost.appendChild(btn(o.label + ' ' + (counts[o.id] || 0), {
          size: 'sm', selected: S.findingLevel === o.id,
          on: () => { S.findingLevel = o.id; render(); },
        }));
      });

    const list = $('#findingList');
    list.textContent = '';
    const shown = D.findings.filter((f) => S.findingLevel === 'all' || f.level === S.findingLevel);
    const logged = {};
    S.ledger.forEach((r) => { if (r.findingId) logged[r.findingId] = 1; });
    shown.forEach((f) => {
      const b = el('button', 'tc-finding'
        + (f.id === S.finding && S.focus === 'finding' ? ' is-selected' : '')
        + (logged[f.id] ? ' is-logged' : ''));
      b.type = 'button';
      b.dataset.sev = f.severity;
      const hd = el('div', 'hd');
      hd.appendChild(el('span', 'id', f.id));
      hd.appendChild(el('span', 'lv', LEVEL_LABEL[f.level]));
      b.appendChild(hd);
      b.appendChild(el('span', 'ti', f.title));
      b.appendChild(el('span', 'mt', f.metric));
      b.addEventListener('click', () => {
        S.finding = f.id;
        S.focus = 'finding';
        applyFocus(f);
        render();
      });
      list.appendChild(b);
    });
    $('[data-bind="findingCount"]').textContent = shown.length + ' / ' + D.findings.length;
  }

  function applyFocus(f) {
    if (!f || !f.focus) return;
    if (f.focus.view) S.view = f.focus.view;
    if (f.focus.task) S.task = f.focus.task;
    if (f.focus.overlay) S.overlay = f.focus.overlay;
    if (f.focus.critOnly != null) S.critOnly = f.focus.critOnly;
    if (f.focus.tab) S.compilerTab = f.focus.tab;
    if (f.focus.pass) {
      const p = D.passes.find((x) => x.name === f.focus.pass);
      if (p) S.pass = p.idx;
    }
    if (f.focus.view === 'l2') { S.t0 = 0; S.t1 = R().swimlane.spanUs; }
  }

  function renderStatus() {
    const host = $('#statusStrip');
    host.textContent = '';
    const rank = R();
    const open = openExperiment();
    const items = [
      ['case', D.case.program],
      ['rank', S.rank + ' inv=' + TRACE_MATCH[S.rank].inv],
      ['span', us(rank.swimlane.spanUs, 1)],
      ['tasks', String(rank.tasks.length)],
      ['crit', rank.critical.tags.length + ' 节点'],
      ['AIC / AIV', pct(rank.occupancy.aicUtil, 0) + ' / ' + pct(rank.occupancy.aivUtil, 0)],
      ['sched', pct(rank.scheduler.perLaneUtil, 0)],
      ['hints', String(D.hints.length)],
    ];
    items.forEach((it) => {
      const s = el('span', 'tc-status-item');
      s.appendChild(el('span', 'k', it[0]));
      s.appendChild(el('span', 'v', it[1]));
      host.appendChild(s);
    });
    const pmu = el('span', 'tc-status-item');
    pmu.appendChild(el('span', 'k', 'pmu'));
    const pv = el('span', 'v warn', 'off');
    pmu.appendChild(pv);
    host.appendChild(pmu);
    const exp = el('span', 'tc-status-item');
    exp.appendChild(el('span', 'k', '实验'));
    exp.appendChild(el('span', 'v' + (open ? ' warn' : ' ok'), open ? open.id + ' 进行中' : '无进行中'));
    host.appendChild(exp);
  }

  function renderFingerprint() {
    const host = $('#fingerprint');
    host.textContent = '';
    const head = el('header', 'panel-shell-header');
    head.appendChild(el('h2', 'panel-shell-title', 'Case fingerprint'));
    head.appendChild(el('span', 'panel-shell-meta', D.case.runDir));
    const close = el('button', 'panel-shell-close', '×');
    close.type = 'button';
    close.setAttribute('aria-label', '关闭');
    close.addEventListener('click', () => toggleFingerprint(false));
    head.appendChild(close);
    host.appendChild(head);
    const body = el('div', 'panel-shell-body');
    const dl = el('dl', 'tc-fp-grid');
    const add = (k, v) => { dl.appendChild(el('dt', null, k)); dl.appendChild(el('dd', null, v)); };
    add('program', D.case.program);
    add('model', D.case.model);
    add('platform / backend', D.case.toolchain.platform + ' / ' + D.case.backend);
    add('pto-isa', D.case.toolchain.ptoIsaRevision);
    add('runtime', D.case.toolchain.runtimeName + ' @ ' + D.case.toolchain.runtimeRevision);
    add('ranks / device', D.case.ranks.join(', ') + ' / ' + D.case.device);
    add('cores', D.case.numCores + '（AIC ' + D.case.aicCount + ' + AIV ' + D.case.aivCount + '，每核 ' + D.case.threadsPerCore + ' thread）');
    add('trace clock', (D.case.clockHz / 1e6) + ' MHz');
    add('callables', D.case.callables + ' 个（incore scope ' + D.case.incoreScopes.length + '）');
    add('captured', D.case.capturedAt);
    add('source root', D.case.sourceRoot);
    body.appendChild(dl);
    body.appendChild(el('p', 'tc-note', '绑定参数 ' + D.case.params.length + ' 个，按方向与 dtype 列出前 12 个：'));
    body.appendChild(table([
      { label: 'name', cell: (p) => esc(p.name.replace(/__ssa_v0$/, '')), mono: true },
      { label: 'dir', key: 'dir' },
      { label: 'dtype', key: 'dtype', mono: true },
      { label: 'shape', cell: (p) => '[' + p.shape.join(', ') + ']', mono: true },
    ], D.case.params.slice(0, 12), {}));
    host.appendChild(body);
  }

  function toggleFingerprint(force) {
    const host = $('#fingerprint');
    const chip = document.querySelector('[data-act="toggle-fingerprint"]');
    const open = force == null ? host.hidden : force;
    host.hidden = !open;
    chip.setAttribute('aria-expanded', open ? 'true' : 'false');
  }

  /* ---------------------------------------------------------- search */
  function buildSearchIndex() {
    const idx = [];
    Object.keys(D.ranks).forEach((rank) => {
      D.ranks[rank].tasks.forEach((t) => {
        idx.push({
          kind: 'task', text: t.callable + ' ' + t.tag, name: t.callable,
          value: t.tag + ' · ' + num(t.span, 0) + ' us · ' + rank,
          go: () => { S.rank = rank; S.view = 'l1'; S.task = t.tag; S.focus = 'task'; },
        });
      });
    });
    D.findings.forEach((f) => {
      idx.push({
        kind: 'finding', text: f.id + ' ' + f.title + ' ' + f.axis, name: f.id + ' ' + f.title,
        value: f.metric,
        go: () => { S.finding = f.id; S.focus = 'finding'; applyFocus(f); },
      });
    });
    D.passes.forEach((p) => {
      idx.push({
        kind: 'pass', text: p.name, name: p.name, value: '#' + p.idx + ' · ' + p.lines + ' 行',
        go: () => { S.view = 'compiler'; S.compilerTab = 'passes'; S.pass = p.idx; S.focus = 'pass'; },
      });
    });
    D.depthSites.forEach((s) => {
      idx.push({
        kind: 'hint', text: 'PH-MR-001 ' + s.module + ':' + s.line, name: s.file + ':' + s.line,
        value: 'depth ' + s.maxReqDepth + ' → ' + s.fittedDepth,
        go: () => { S.view = 'compiler'; S.compilerTab = 'depth'; S.hintSite = s.key; S.focus = 'hint'; },
      });
    });
    D.tileSites.forEach((s) => {
      idx.push({
        kind: 'hint', text: 'PH001 ' + s.module + ':' + s.line, name: s.file + ':' + s.line,
        value: '末维 ' + s.minB + 'B',
        go: () => { S.view = 'compiler'; S.compilerTab = 'granularity'; S.hintSite = s.key; S.focus = 'hint'; },
      });
    });
    return idx;
  }
  const SEARCH = buildSearchIndex();

  function runSearch(q) {
    const box = $('#searchResults');
    const input = $('#searchInput');
    box.textContent = '';
    const query = q.trim().toLowerCase();
    if (!query) { box.hidden = true; input.setAttribute('aria-expanded', 'false'); return; }
    const hits = SEARCH.filter((h) => h.text.toLowerCase().indexOf(query) >= 0).slice(0, 24);
    if (!hits.length) {
      box.appendChild(el('div', 'tc-search-empty', '没有匹配的 kernel、任务、提示或 Pass'));
    } else {
      hits.forEach((h) => {
        const b = el('button', 'tc-search-hit');
        b.type = 'button';
        b.appendChild(el('span', 'k', h.kind));
        b.appendChild(el('span', 'n', h.name));
        b.appendChild(el('span', 'v', h.value));
        b.addEventListener('click', () => {
          h.go();
          input.value = '';
          box.hidden = true;
          input.setAttribute('aria-expanded', 'false');
          render();
        });
        box.appendChild(b);
      });
    }
    box.hidden = false;
    input.setAttribute('aria-expanded', 'true');
  }

  /* ============================================================ render */
  function redrawStage() {
    const stage = $('#stage');
    if (stage.__redraw) stage.__redraw();
  }

  function render() {
    const stage = $('#stage');
    if (stage.__ro) { stage.__ro.disconnect(); stage.__ro = null; }
    stage.__redraw = null;
    stage.textContent = '';

    renderTabs();
    renderToolbar();
    renderExplorer();

    if (S.view === 'e2e') viewE2E(stage);
    else if (S.view === 'l2') viewL2(stage);
    else if (S.view === 'l1') viewL1(stage);
    else if (S.view === 'compiler') viewCompiler(stage);
    else viewISA(stage);

    renderInspector();
    renderDock();
    renderTerminal();
    renderStatus();
    $('[data-bind="caseChip"]').textContent = D.case.program + ' · ' + S.rank;
  }

  /* ------------------------------------------------------------- boot */
  function boot() {
    if (window.PtoIdeFrame) window.PtoIdeFrame.initAll();
    S.t0 = 0;
    S.t1 = R().swimlane.spanUs;
    S.tile = defaultTile();
    S.view = 'e2e';
    S.focus = null;
    renderFingerprint();
    render();

    document.querySelector('[data-act="toggle-fingerprint"]').addEventListener('click', () => toggleFingerprint());
    document.querySelector('[data-act="theme"]').addEventListener('click', () => {
      const root = document.documentElement;
      root.dataset.theme = root.dataset.theme === 'light' ? 'dark' : 'light';
      render();
    });
    document.querySelector('[data-act="focus-search"]').addEventListener('click', () => $('#searchInput').focus());
    document.querySelector('[data-act="show-ledger"]').addEventListener('click', () => {
      const btnEl = document.querySelector('[data-ide-toggle="inspector"]');
      if (btnEl && btnEl.getAttribute('aria-expanded') === 'false') btnEl.click();
      $('#inspector').scrollTop = $('#inspector').scrollHeight;
    });
    document.querySelector('[data-act="open-terminal"]').addEventListener('click', () => {
      const btnEl = document.querySelector('[data-ide-toggle="terminal"]');
      if (btnEl && btnEl.getAttribute('aria-expanded') === 'false') btnEl.click();
    });

    const input = $('#searchInput');
    input.addEventListener('input', () => runSearch(input.value));
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') { input.value = ''; runSearch(''); input.blur(); }
      if (e.key === 'Enter') {
        const first = $('#searchResults').querySelector('.tc-search-hit');
        if (first) first.click();
      }
    });
    document.addEventListener('click', (e) => {
      if (!e.target.closest('.tc-search')) { $('#searchResults').hidden = true; }
      if (!e.target.closest('#fingerprint') && !e.target.closest('[data-act="toggle-fingerprint"]')) {
        toggleFingerprint(false);
      }
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === '/' && document.activeElement !== input) { e.preventDefault(); input.focus(); }
      if (e.key >= '1' && e.key <= '5' && !e.metaKey && !e.ctrlKey
        && document.activeElement !== input && document.activeElement.tagName !== 'INPUT'
        && document.activeElement.tagName !== 'TEXTAREA') {
        S.view = LEVELS[+e.key - 1].id;
        render();
      }
    });
    window.addEventListener('resize', () => { redrawStage(); });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot, { once: true });
  else boot();
})();
