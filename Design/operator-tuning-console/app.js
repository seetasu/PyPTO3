/* =============================================================
 * Tuning Console
 *
 * Two real on-device runs, switchable from the topbar case chip, opened as a
 * working surface for the tuning loop:
 *   E2E  -> L2 schedule -> L1/L0 core pipeline -> compiler lowering -> ISA / layout
 *
 * Timing and utilization numbers come from data.js, which build-data.cjs
 * derives from the run's own artifacts.  E2E model-stage labels can be
 * supplied by JSON; when absent, the screen marks its fallback rule grouping.
 *
 * Shared patterns used:
 *   ide-frame        page shell, panes, bottom dock, status strip
 *   workbench-shell  split resize kernel (through ide-frame)
 *   swimlane-task    every timed task bar + its hover tooltip + colormap
 * ============================================================= */
(function () {
  'use strict';

  const RUNS = window.TUNING_RUNS;
  const CASES = window.TUNING_CASES;
  const SW = window.PtoSwimlaneTaskPattern;
  const requestedEmbedView = new URLSearchParams(window.location.search).get('embed');
  const EMBED_VIEW = ['l1', 'l2'].includes(requestedEmbedView) ? requestedEmbedView : null;
  const requestedCase = new URLSearchParams(window.location.search).get('case');
  const initialCase = CASES.some((c) => c.id === requestedCase) ? requestedCase : CASES[0].id;

  /* The qwen3 profile layers ship their own queue entries. Merge them before
   * anything reads a run, so the queue is one list everywhere — including the
   * case menu's counts, which describe runs that are not loaded yet. */
  Object.keys(RUNS).forEach((id) => {
    const run = RUNS[id];
    if (!run.qwen3 || !run.qwen3.findings || !run.findings) return;
    run.findings = run.findings.concat(run.qwen3.findings);
    run.hygieneCount = run.findings.filter((f) => f.kind === 'hygiene').length;
  });

  /* The active case. Everything derived from it is rebuilt by loadCase(),
   * because the two dumps do not carry the same artifacts: one has host
   * STRACE spans and two ranks, the other has neither. */
  let D = RUNS[initialCase];
  let CYC_PER_US = D.case.clockHz ? D.case.clockHz / 1e6 : null;
  let TRACE_MATCH = {};
  let findingById = {};
  let tasksOf = {};
  const hasE2E = () => !!D.e2e;
  const isServingBenchmark = () => D.kind === 'serving-benchmark';
  const multiRank = () => D.case.ranks.length > 1;

  /* ------------------------------------------------------------- state */
  const S = {
    rank: D.defaultRank,
    view: 'e2e',
    task: D.derived.worstHandoff,
    finding: null,
    chainStep: null,          /* 'C2:1' -- which ladder rung the reader is on */
    findingLevel: 'all',
    focus: null,               /* 'finding' | 'task' | 'hint' | 'pass' */
    /* L2 opens on an annotated per-core trace: the full execution remains
     * visible, but only the strongest performance signals are saturated. */
    laneFilter: 'summary',
    colorMode: 'semantic',
    colorOn: true,          /* off => every bar goes neutral grey */
    scopeReturn: null,      /* window + focus to restore when drilling back up */
    folded: {},             /* inspector sections the reader has collapsed */
    overlay: 'none',
    deps: 'off',            /* 'off' | 'sel' | 'path' -- dependency edges on the swimlane */
    critOnly: false,
    pathOnly: 'off',       /* 'off' | 'obs' | 'cpm' -- which path the filter shows */
    pathFocus: false,      /* click an execution-main-path task => mute unrelated work */
    focusEvidence: false,
    scrollToLane: null,
    t0: 0, t1: 0,
    compilerTab: 'passes',
    pass: 17,
    passMode: 'overview',
    hintSite: null,
    hintModule: 'all',
    dockMode: 'sched',
    termTab: 'problems',
    ledger: [],
    tile: null,
    e2ePanel: 'triage',       /* triage | serving | device | samples
                               * qwen3: step | ops | api | topo */
    variant: null,            /* qwen3 case: which capture x stage is armed */
    l2Panel: 'swimlane',      /* swimlane | layers | head */
    l1Panel: 'pipe',          /* pipe | pmu */
    guidedPass: 'MemoryReuse',
    guidedDepth: 2,
  };

  const R = () => D.ranks[S.rank];
  function selectAnalysisRank(rank, view) {
    S.rank = rank;
    S.t0 = 0; S.t1 = R().swimlane.spanUs;
    if (!tasksOf[S.rank][S.task]) S.task = R().tasks[0].tag;
    if (view) S.view = view;
  }
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
  /* title= needs a real newline; a literal one inside a string breaks the file */
  const NL = String.fromCharCode(10);
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  const LEVELS = [
    { id: 'e2e', label: 'E2E', hint: '端到端与 rank 分解' },
    { id: 'l2', label: 'L2 调度', hint: '任务放置、依赖、关键路径' },
    { id: 'l1', label: 'L1 / L0', hint: '单核流水与片上预算' },
    { id: 'compiler', label: '编译器', hint: 'Pass、流水深度、搬运粒度' },
    { id: 'isa', label: 'ISA / 布局', hint: '布局与指令层证据' },
  ];
  /* role -> how the reader should read this rung. 'stop' is deliberately a
   * first-class rung: a chain that cannot go further says so here instead of
   * ending on a guess. */
  const ROLE = {
    observe: { label: '现象', hint: '这一层看到了什么' },
    descend: { label: '下探', hint: '往下一层追什么' },
    root: { label: '落点', hint: '可以直接验证的地方' },
    stop: { label: '止步', hint: '本 dump 到此为止' },
  };

  const LEVEL_LABEL = {};
  LEVELS.forEach((l) => { LEVEL_LABEL[l.id] = l.label; });

  /* Colormap: the shared pattern owns every task color decision, including the
   * aic / aiv / aicpu lane-kind colors. No page-local palette. */
  const CMAP = SW.createTaskColormap();

  /* Task colour has one decision point. With colouring off every bar goes to
   * a flat neutral, which hands the contrast budget to the occupancy band,
   * the idle wash, the critical path and the evidence markers. */
  function taskColor(t) {
    /* must be an opaque hex: the pattern lightens/alpha-blends baseColor and
     * an rgba() token comes back out as white. --surface-4 is the quiet
     * neutral in both themes. */
    if (!S.colorOn) return cssVar('--surface-4');
    return S.colorMode === 'engine'
      ? CMAP.colorForTask({ laneKind: t.kind }, 'engine')
      : CMAP.colorForTask({ colorKey: t.callable, label: t.callable }, 'semantic');
  }

  /* ------------------------------------------------- derived per case */
  const curTask = () => tasksOf[S.rank][S.task] || R().tasks[0];
  let ledgerSeq = 0;

  function loadCase(id) {
    D = RUNS[id];
    CYC_PER_US = D.case.clockHz ? D.case.clockHz / 1e6 : null;
    /* the tab title names the case, so it has to follow the switch */
    document.title = 'Tuning Console · ' + id;

    if (isServingBenchmark()) {
      TRACE_MATCH = {};
      findingById = {};
      tasksOf = {};
      S.rank = 'request';
      S.view = 'e2e';
      S.finding = null;
      S.focus = null;
      S.ledger.length = 0;
      return;
    }

    /* Which invocation does each rank's device trace correspond to?
     * Reconcile the trace span against the host-reported device_wall.sched.
     * Without host spans there is nothing to reconcile against. */
    TRACE_MATCH = {};
    Object.keys(D.ranks).forEach((rank) => {
      if (!D.e2e || !D.e2e[rank]) { TRACE_MATCH[rank] = null; return; }
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

    findingById = {};
    D.findings.forEach((f) => { findingById[f.id] = f; });

    tasksOf = {};
    Object.keys(D.ranks).forEach((rank) => {
      tasksOf[rank] = {};
      D.ranks[rank].tasks.forEach((t) => { tasksOf[rank][t.tag] = t; });
    });

    /* state that only makes sense inside one case */
    S.rank = D.defaultRank;
    S.finding = null;
    S.chainStep = null;
    S.focus = null;
    S.focusEvidence = false;
    S.scopeReturn = null;
    /* the qwen3 case opens on its own E2E panel: torch step attribution, not
     * the serving triage the other case starts from */
    S.e2ePanel = D.qwen3 ? 'step' : 'triage';
    S.variant = D.qwen3
      ? (D.qwen3.variants.find((v) => v.primary) || D.qwen3.variants[0]).id : null;
    S.l2Panel = 'swimlane';
    S.l1Panel = 'pipe';
    S.guidedPass = 'MemoryReuse';
    S.guidedDepth = 2;
    S.findingLevel = 'all';
    S.laneFilter = 'summary';
    S.overlay = 'none';
    S.deps = 'off';
    S.critOnly = false;
    S.pathOnly = 'off';
    S.pathFocus = false;
    S.task = tasksOf[S.rank][D.derived.worstHandoff]
      ? D.derived.worstHandoff : R().tasks[0].tag;
    S.hintSite = D.tileSites.length ? D.tileSites[0].key : null;
    S.pass = D.passes.length ? D.passes[Math.min(17, D.passes.length - 1)].idx : 0;
    S.passMode = 'overview';
    S.t0 = 0;
    S.t1 = R().swimlane.spanUs;
    /* the E2E tab stays selectable: its absence page is the explanation */

    /* the ledger belongs to the case: a baseline for one run is not a
     * baseline for the other */
    S.ledger.length = 0;
    ledgerSeq = 0;
    S.ledger.push({
      id: 'B0',
      state: 'baseline',
      title: '基线锁定 · ' + D.case.program,
      findingId: null,
      hypothesis: '固定 shape / dtype / 平台 / 卡数 / 工具链，作为后续所有对比的唯一基准。',
      change: D.case.runDir + '（' + D.case.capturedAt + '，platform ' + D.case.toolchain.platform + '）',
      correctness: D.case.params.length
        ? 'distributed_meta.json 记录 ' + D.case.params.length + ' 个绑定参数，schema ' + D.case.metaSchema
        : '本 dump 无 distributed_meta.json：绑定参数未记录，正确性基准缺口',
      perf: hasE2E()
        ? D.case.ranks.map((r) => r + ' device_wall '
            + us(D.e2e[r][TRACE_MATCH[r].inv]['chip.run.runner_run.device_wall'].us)).join(' / ')
          + '（inv=' + TRACE_MATCH[D.defaultRank].inv + '）'
        : '本 dump 无 host STRACE log：device_wall 不可得，基线只能用 trace span '
          + us(R().swimlane.spanUs),
      keep: '保留为基线',
    });
  }
  const openExperiment = () => S.ledger.find((r) => r.state === 'open') || null;

  /* ============================================================ tables */
  function table(cols, rows, opts) {
    const o = opts || {};
    const wrap = el('div', 'tc-table-scroll');
    if (o.tall) wrap.dataset.tall = 'true';
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
      if (row.__subject) r.classList.add('is-subject');
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

  /* ============================================== active finding context
   * S.finding stays active while the reader works, independent of what the
   * inspector happens to be showing. Everything below answers one question:
   * "which things on this screen are the evidence for the active finding?" */
  const activeFinding = () => (S.finding ? findingById[S.finding] : null);
  /* the queue's own order, chains first: a hygiene item must never take a
   * slot in a "top N" list while an attributed chain is left out */
  const topChains = (n) => D.findings.filter((f) => f.kind !== 'hygiene').slice(0, n);

  /* When the reader steps onto a rung of a chain, the marked objects are that
   * rung's, not the whole chain's -- otherwise walking down to the compiler
   * layer still leaves L2 tasks numbered on screen. */
  function activeStep() {
    const f = activeFinding();
    if (!f || !S.chainStep || S.chainStep.indexOf(f.id + ':') !== 0) return null;
    return (f.chain || [])[Number(S.chainStep.split(':')[1])] || null;
  }
  const guidedJourneyActive = () => !!(D.case.guidedJourney && activeFinding());
  function activeSubjects() {
    const st = activeStep();
    if (st) return st.subjects;
    const f = activeFinding();
    return f ? f.subjects : null;
  }

  function subjectTaskSet() {
    const s = activeSubjects();
    const set = {};
    if (s) s.tasks.forEach((t, i) => { set[t] = i + 1; });
    return set;
  }
  function subjectSiteSet() {
    const s = activeSubjects();
    const set = {};
    if (s) s.sites.forEach((x, i) => { set[x] = i + 1; });
    return set;
  }
  function subjectLaneSet() {
    const s = activeSubjects();
    const set = {};
    if (s) s.lanes.forEach((x, i) => { set[x] = i + 1; });
    return set;
  }

  /* Jump to one piece of evidence. The inspector deliberately stays on the
   * finding: it is the argument, the stage is where that argument shows up.
   * The object's own detail is still one click away on the canvas. */
  function gotoChip(chip) {
    const f = activeFinding();
    if (f) S.focus = 'finding';
    if (chip.kind === 'task') {
      const t = tasksOf[S.rank][chip.id];
      S.task = chip.id;
      if (!f) S.focus = 'task';
      const home = (activeSubjects() || f.subjects).view;
      if (t && (home === 'l2' || S.view === 'l2')) {
        S.view = 'l2';
        const pad = Math.max(40, t.span * 0.35);
        setWindow(t.start - pad, t.end + pad);
      } else {
        S.view = 'l1';
      }
    } else if (chip.kind === 'site') {
      S.view = 'compiler';
      S.compilerTab = D.depthSites.some((s) => s.key === chip.id) ? 'depth' : 'granularity';
      S.hintSite = chip.id;
      if (!f) S.focus = 'hint';
    } else if (chip.kind === 'lane') {
      S.view = 'l2';
      S.laneFilter = chip.id.indexOf('AIC') === 0 ? 'aic' : 'aiv';
      S.scrollToLane = chip.id;
    } else if (chip.kind === 'rank') {
      S.rank = chip.id;
      S.view = 'e2e';
      S.t0 = 0; S.t1 = R().swimlane.spanUs;
      if (!tasksOf[S.rank][S.task]) S.task = R().tasks[0].tag;
    } else if (chip.kind === 'phase') {
      S.view = 'l2';
      S.overlay = 'sched';
      S.dockMode = 'sched';
    }
    render();
  }

  /* the bar itself, pinned to the top of the centre stage */
  function findingBar(stage) {
    const f = activeFinding();
    if (!f) return;
    const bar = el('div', 'tc-findingbar');
    bar.dataset.sev = f.severity;
    if (f.kind === 'hygiene') bar.classList.add('is-hygiene');

    /* Which rung of the chain the reader is standing on. Without this the bar
     * always speaks for the first layer, and a four-layer chain reads as one
     * flat observation again. */
    const chain = f.chain || [];
    const rung = activeStep();
    const stepIdx = rung ? chain.indexOf(rung) : -1;
    const chips = rung ? rung.chips : f.chips;

    const hd = el('div', 'hd');
    hd.appendChild(el('span', 'id', f.id));
    hd.appendChild(el('span', 'ti', rung ? rung.headline : f.title));
    hd.appendChild(el('span', 'mt', rung
      ? (LEVEL_LABEL[rung.level] || rung.level) + ' · '
        + (ROLE[rung.role] || { hint: '' }).hint
      : f.metric + (f.cost ? ' · ' + f.cost.share + '% of makespan' : ' · 无归因')));
    const acts = el('div', 'acts');
    const onHomeView = f.subjects.view === S.view
      && (!f.subjects.tab || f.subjects.tab === S.compilerTab);
    if (guidedJourneyActive()) {
      acts.appendChild(el('span', 'tc-journey-done', stepIdx === chain.length - 1
        ? '根因已定位 · ' + (f.rootPass || 'Pass')
        : (LEVEL_LABEL[(rung || chain[0]).level] || (rung || chain[0]).level) + ' 证据'));
    } else if (!onHomeView) {
      acts.appendChild(btn('去证据所在页 · ' + LEVEL_LABEL[f.subjects.view], {
        size: 'sm', variant: 'solid',
        on: () => { applyFocus(f); S.view = f.subjects.view; render(); },
      }));
    } else if (chips.length) {
      acts.appendChild(btn('聚焦证据', {
        size: 'sm', selected: S.focusEvidence,
        title: '把非证据对象压暗，只留这条瓶颈牵涉到的部分',
        on: () => { S.focusEvidence = !S.focusEvidence; render(); },
      }));
    }
    acts.appendChild(btn('退出', {
      size: 'sm', variant: 'ghost',
      title: '清除当前瓶颈上下文，回到自由浏览',
      on: () => { S.finding = null; S.focusEvidence = false; S.focus = null; render(); },
    }));
    hd.appendChild(acts);
    bar.appendChild(hd);

    /* the ladder, walkable from the stage itself */
    if (chain.length) {
      const rungs = el('div', 'tc-rungs');
      rungs.appendChild(el('span', 'lead', guidedJourneyActive()
        ? '分析进度 ' + (stepIdx + 1) + ' / ' + chain.length : '链'));
      chain.forEach((st, i) => {
        if (i) rungs.appendChild(el('span', 'arrow', '→'));
        const b = el(guidedJourneyActive() ? 'span' : 'button', 'tc-rung'
          + (i === stepIdx ? ' is-current' : '')
          + (guidedJourneyActive() && i < stepIdx ? ' is-complete' : ''));
        if (!guidedJourneyActive()) {
          b.type = 'button';
          b.addEventListener('click', () => { applyStep(f, st); render(); });
        }
        b.dataset.role = st.role;
        b.title = st.headline;
        if (guidedJourneyActive() && i === stepIdx) b.setAttribute('aria-current', 'step');
        b.appendChild(el('span', 'lv', LEVEL_LABEL[st.level] || st.level));
        b.appendChild(el('span', 'rl', (ROLE[st.role] || { label: st.role }).label));
        rungs.appendChild(b);
      });
      bar.appendChild(rungs);
    }

    if (chips.length) {
      const row = el('div', 'tc-evchips');
      row.appendChild(el('span', 'lead', '证据 ' + chips.length));
      chips.forEach((chip, i) => {
        const b = el(guidedJourneyActive() ? 'span' : 'button', 'tc-evchip');
        if (!guidedJourneyActive()) b.type = 'button';
        const isCurrent = (chip.kind === 'task' && chip.id === S.task)
          || (chip.kind === 'site' && chip.id === S.hintSite)
          || (chip.kind === 'rank' && chip.id === S.rank);
        if (isCurrent) b.classList.add('is-current');
        b.appendChild(el('span', 'mk', String(i + 1)));
        b.appendChild(el('span', 'nm', chip.label));
        if (chip.value) b.appendChild(el('span', 'vl', chip.value));
        if (!guidedJourneyActive()) b.addEventListener('click', () => gotoChip(chip));
        row.appendChild(b);
      });
      bar.appendChild(row);
    } else if (f.subjects.absent) {
      const row = el('div', 'tc-evchips');
      row.appendChild(el('span', 'lead', '证据 0 · 缺席项'));
      bar.appendChild(row);
    }

    stage.appendChild(bar);
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

  function drawTimeRuler(ctx, x0, w, y, t0, t1, opts) {
    const o = opts || {};
    const span = t1 - t0;
    const stepRaw = span / 8;
    const mag = Math.pow(10, Math.floor(Math.log10(stepRaw)));
    const step = [1, 2, 5, 10].map((m) => m * mag).find((v) => v >= stepRaw) || mag * 10;
    ctx.save();
    if (o.dense) {
      const bandH = 26;
      const bandY = y;
      const label = (t) => step >= 1000
        ? (t / 1000).toFixed(1) + 'ms'
        : Math.round(t) + 'us';
      ctx.fillStyle = cssVar('--surface-3');
      ctx.fillRect(x0, bandY, w, bandH);
      ctx.strokeStyle = cssVar('--border-subtle');
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x0, bandY + bandH - 0.5);
      ctx.lineTo(x0 + w, bandY + bandH - 0.5);
      ctx.stroke();
      const minor = step / 10;
      const firstMinor = Math.ceil(t0 / minor) * minor;
      for (let t = firstMinor; t <= t1 + minor * 0.001; t += minor) {
        const x = x0 + ((t - t0) / span) * w;
        const isMajor = Math.abs(t / step - Math.round(t / step)) < 0.0001;
        ctx.strokeStyle = isMajor ? cssVar('--border-default') : cssVar('--border-subtle');
        ctx.globalAlpha = isMajor ? 0.9 : 0.62;
        ctx.beginPath();
        ctx.moveTo(Math.round(x) + 0.5, bandY + (isMajor ? 14 : 18));
        ctx.lineTo(Math.round(x) + 0.5, bandY + bandH - 4);
        ctx.stroke();
        if (isMajor) {
          ctx.globalAlpha = 1;
          ctx.font = '500 11px ' + cssVar('--font-sans');
          ctx.fillStyle = cssVar('--foreground-secondary');
          ctx.textBaseline = 'middle';
          ctx.textAlign = t === t0 ? 'left' : 'center';
          ctx.fillText(label(t), t === t0 ? x + 3 : x, bandY + 8);
        }
      }
      ctx.restore();
      return;
    }
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

  /* ------------------------------------------------- function summary
   * Sigma = N x mean is exact. Comparing against N x median says which of
   * the three causes is in play without inventing a verdict model. */
  function renderFuncSummary(stage) {
    const rank = R();
    const sec = el('section');
    const top = rank.scopes.slice(0, 12);
    sec.appendChild(sectionHead('函数汇总 · 慢在哪一项',
      'Σ = 重复 × 宽度 × 均值 · 前 ' + top.length + ' / ' + rank.scopes.length,
      el('span', 'tc-readout', 'Σ 大 ≠ 拖慢墙钟 —— 末列才是')));
    const obs = {};
    rank.cpath.segments.forEach((sg) => { obs[sg.tag] = 1; });
    sec.appendChild(table([
      { label: 'scope', cell: (r) => esc(r.name), mono: true },
      { label: 'Σ core-time', cell: (r) => num(r.coreTime, 0), mono: true, num: true },
      { label: '块数', cell: (r) => String(r.cost.blocks), num: true },
      { label: '重复', cell: (r) => num(r.cost.repeat, r.cost.repeat % 1 ? 1 : 0), num: true },
      { label: '宽度', cell: (r) => r.cost.width + ' 核', num: true },
      { label: '均值', cell: (r) => num(r.cost.mean, 2), mono: true, num: true },
      { label: 'p90/中位', cell: (r) => (r.cost.spread == null ? '—'
        : '<span class="' + (r.cost.spread > 1.5 ? 'bad' : r.cost.spread > 1.2 ? 'warn' : '') + '">'
          + num(r.cost.spread, 2) + '</span>'), num: true },
      { label: '偏离中位', cell: (r) => (r.cost.skew >= 0 ? '+' : '') + num(r.cost.skew, 0), mono: true, num: true },
      { label: '主因', cell: (r) => causeOf(r, rank) },
      /* the column that answers what this layer cannot */
      { label: '在路径上', cell: (r) => pathCell(r, obs, rank) },
    ], top, {
      onPick: (r) => { S.view = 'l2'; S.focus = 'scope'; S.task = r.tags[0]; render(); },
    }));
    sec.appendChild(el('p', 'tc-note',
      '块数 = 重复 × 宽度，Σ = 块数 × 均值，都是恒等式。'
      + '重复 = launch 次数 × 波数（同一批核跑了几轮），宽度 = 一次铺开占几个核 —— '
      + '把两者混成一个「次数」会把「宽」误读成「调用多」。'
      + '偏离中位为负 = 中位高于均值，少数快块把均值拉低了，不是长尾。'
      + '「主因」的倍数是相对本 run 所有 scope 的中位数。'
      + '末列来自「路径归责」那一层 —— 本层自己证明不了一个 scope 是否拖慢墙钟。'));
    stage.appendChild(sec);
  }

  /* Which factor carries the cost, measured against this run's own median
   * rather than an absolute threshold, and reported as the multiple so the
   * reader can see how lopsided it is instead of trusting a label. */
  function causeOf(sc, rank) {
    const midOf = (f) => {
      const v = rank.scopes.map(f).slice().sort((a, b) => a - b);
      return v[Math.floor(v.length / 2)] || 1;
    };
    const medMid = midOf((x) => x.cost.med);
    const repMid = midOf((x) => x.cost.repeat);
    const slowX = sc.cost.med / medMid;
    const manyX = sc.cost.repeat / repMid;
    const wobbly = sc.cost.spread != null && sc.cost.spread > 1.5;
    const x = (n) => ' ×' + num(n, n >= 10 ? 0 : 1);
    let main;
    if (slowX < 2 && manyX < 2) main = '<span class="muted">无突出项</span>';
    else if (slowX >= manyX * 3) main = '单次慢' + x(slowX);
    else if (manyX >= slowX * 3) main = '次数多' + x(manyX);
    else main = '两者兼有';
    return main + (wobbly ? ' <span class="warn">+ 波动</span>' : '');
  }

  /* wall-clock relevance comes from the path layer, not from this table */
  function pathCell(sc, obs, rank) {
    const onObs = sc.tags.some((t) => obs[t]);
    if (sc.onCrit) return '<span class="bad">依赖关键路径</span>';
    if (onObs) return '<span class="warn">执行主路径（归因）</span>';
    return '<span class="muted">都不在 · slack ' + num(sc.minSlack, 0) + '</span>';
  }

  /* ================================================= E2E model projection
   *
   * The raw dump deliberately has no model-stage taxonomy: it describes
   * calls, ranks and device traces.  This adapter keeps that boundary clear.
   * A future JSON can provide `e2eRuntime.operators[tag].stage` to replace the
   * fallback rule below; until then the stage label is shown as “规则归类”.
   * Timings always stay trace-derived, only the grouping is mocked/inferred.
   */
  const E2E_STAGES = [
    { id: 'input', label: '输入 / 嵌入', hint: 'token、position、embedding' },
    { id: 'attention', label: 'Attention', hint: 'QKV、attention、softmax' },
    { id: 'communication', label: '通信 / 同步', hint: 'collective、wait、all-to-all' },
    { id: 'moe', label: 'MoE 路由', hint: 'gate、expert、route' },
    { id: 'ffn', label: 'FFN / 输出', hint: 'projection、matmul、norm' },
    { id: 'runtime', label: '运行时 / 其他', hint: '未匹配的运行时任务' },
  ];

  function e2eStageOf(task) {
    const supplied = D.e2eRuntime && D.e2eRuntime.operators
      && D.e2eRuntime.operators[task.tag];
    if (supplied && supplied.stage) return { id: supplied.stage, source: 'json' };
    const name = String(task.callable || '').toLowerCase();
    if (/wait|allgather|all_reduce|allreduce|reduce_scatter|a2a|collective|comm/.test(name)) return { id: 'communication', source: 'rule' };
    if (/embed|token|position|rope/.test(name)) return { id: 'input', source: 'rule' };
    if (/attn|attention|softmax|qk|q_proj|k_proj|v_proj|qkv|fa_/.test(name)) return { id: 'attention', source: 'rule' };
    if (/expert|gate|router|route|moe/.test(name)) return { id: 'moe', source: 'rule' };
    if (/proj|matmul|mlp|norm|ffn|down|up_|out_/.test(name)) return { id: 'ffn', source: 'rule' };
    return { id: 'runtime', source: 'rule' };
  }

  function e2eProjection() {
    const ranks = Object.keys(D.ranks);
    const stages = {};
    E2E_STAGES.forEach((s) => { stages[s.id] = { id: s.id, label: s.label, hint: s.hint, ranks: {} }; });
    const operators = {};
    let ruleCount = 0;
    ranks.forEach((rank) => {
      D.ranks[rank].tasks.forEach((task) => {
        const classified = e2eStageOf(task);
        const stage = stages[classified.id] || stages.runtime;
        if (classified.source === 'rule') ruleCount += 1;
        const existing = stage.ranks[rank] || { coreUs: 0, count: 0, max: null, source: classified.source };
        existing.coreUs += task.span;
        existing.count += 1;
        if (!existing.max || task.span > existing.max.span) existing.max = task;
        stage.ranks[rank] = existing;

        const op = operators[task.callable] || {
          name: task.callable, stage: stage.id, ranks: {}, pathRanks: [], source: classified.source,
        };
        const onCritical = D.ranks[rank].critical.tags.indexOf(task.tag) >= 0;
        const stat = op.ranks[rank] || { coreUs: 0, count: 0, max: null, onCritical: false };
        stat.coreUs += task.span;
        stat.count += 1;
        stat.onCritical = stat.onCritical || onCritical;
        if (!stat.max || task.span > stat.max.span) stat.max = task;
        op.ranks[rank] = stat;
        if (onCritical && op.pathRanks.indexOf(rank) < 0) op.pathRanks.push(rank);
        operators[task.callable] = op;
      });
    });
    return { ranks: ranks, stages: E2E_STAGES.map((s) => stages[s.id]), operators: Object.keys(operators).map((k) => operators[k]), ruleCount: ruleCount };
  }

  /* ===================================================== E2E triage adapter
   *
   * `e2e` is a host STRACE / device-span tree, not an independent serving
   * benchmark.  Keep that distinction visible: a future capture may attach
   * `e2eTriage`, while this adapter labels every derived or demo-only field
   * instead of upgrading it to an observed metric.
   *
   * Supported optional JSON shape:
   * e2eTriage: {
   *   benchmark: { e2e_wall_us, host_wall_us, device_wall_us, source },
   *   workers: [{ id, count, avg_us, min_us, max_us, source }],
   *   workerTasks: [{ id, count, avg_us, min_us, max_us, source }],
   *   lanes: [{ id, count, avg_us, min_us, max_us }],
   *   rounds: [{ id, bind_h2d: { state, us, source },
   *              compile_register: { state, us, source },
   *              result_copy: { state, us, source } }]
   * }
   */
  function triageNumber(obj, camel, snake) {
    if (!obj) return null;
    const value = obj[camel] != null ? obj[camel] : obj[snake];
    return value == null ? null : Number(value);
  }

  function e2eTriageData() {
    const supplied = D.e2eTriage || {};
    const benchmark = supplied.benchmark || {};
    const rank = D.defaultRank;
    const invs = Object.keys(D.e2e[rank] || {}).map(Number).sort((a, b) => a - b);
    const tracedInv = TRACE_MATCH[rank] ? TRACE_MATCH[rank].inv : invs[invs.length - 1];
    const sampled = D.e2e[rank][tracedInv] || {};
    const fromSpan = (name) => sampled[name] ? sampled[name].us : null;
    const e2eWall = triageNumber(benchmark, 'e2eWallUs', 'e2e_wall_us');
    const hostWall = triageNumber(benchmark, 'hostWallUs', 'host_wall_us');
    const deviceWall = triageNumber(benchmark, 'deviceWallUs', 'device_wall_us');
    const hasSuppliedBenchmark = Object.keys(benchmark).length > 0;

    const fallbackBenchmark = {
      e2eWallUs: fromSpan('chip.run'),
      hostWallUs: fromSpan('chip.run.runner_run'),
      deviceWallUs: fromSpan('chip.run.runner_run.device_wall'),
      source: 'STRACE / device span proxy',
      measured: false,
    };
    const resolvedBenchmark = hasSuppliedBenchmark ? {
      e2eWallUs: e2eWall != null ? e2eWall : fallbackBenchmark.e2eWallUs,
      hostWallUs: hostWall != null ? hostWall : fallbackBenchmark.hostWallUs,
      /* A partial benchmark must not silently inherit device timing from an
       * unrelated run.  Null is an intentional “not captured” state. */
      deviceWallUs: deviceWall,
      source: benchmark.source || '独立 benchmark',
      scope: benchmark.scope || null,
      measured: e2eWall != null && hostWall != null && deviceWall != null,
    } : fallbackBenchmark;

    const defaultWorkers = [
      { id: 'WorkerProcess-01', count: 104, avgUs: 146800, minUs: 138200, maxUs: 157600, source: 'mock · Strace 未提供 worker 切分' },
      { id: 'WorkerProcess-02', count: 101, avgUs: 148100, minUs: 140600, maxUs: 159200, source: 'mock · Strace 未提供 worker 切分' },
      { id: 'WorkerProcess-03', count: 103, avgUs: 147300, minUs: 139800, maxUs: 160100, source: 'mock · Strace 未提供 worker 切分' },
      { id: 'WorkerProcess-04', count: 102, avgUs: 147900, minUs: 141100, maxUs: 158700, source: 'mock · Strace 未提供 worker 切分' },
    ];
    const normalizeWorkers = (items, fallback) => Array.isArray(items) && items.length ? items.map((worker, i) => ({
      id: worker.id || worker.worker || ('WorkerProcess-' + String(i + 1).padStart(2, '0')),
      count: Number(worker.count || 0),
      avgUs: triageNumber(worker, 'avgUs', 'avg_us'),
      minUs: triageNumber(worker, 'minUs', 'min_us'),
      maxUs: triageNumber(worker, 'maxUs', 'max_us'),
      source: worker.source || 'Serving Strace',
    })) : fallback;
    const workers = normalizeWorkers(supplied.workers, defaultWorkers);
    const workerTasks = normalizeWorkers(supplied.workerTasks || supplied.worker_tasks, workers);
    const lanes = normalizeWorkers(supplied.lanes, []);

    const defaultRounds = invs.slice(0, 3).map((inv, i) => {
      const spans = D.e2e[rank][inv];
      const value = (name) => spans[name] ? spans[name].us : null;
      return {
        id: 'inv ' + inv,
        bindH2d: { state: 'repeat', us: value('chip.run.bind'), source: 'bind 实测 · H2D mock' },
        compileRegister: {
          state: i === 0 && value('chip.run.bind.prebuilt') > 100 ? 'once' : 'none',
          us: value('chip.run.bind.prebuilt'), source: 'bind.prebuilt proxy',
        },
        resultCopy: { state: 'repeat', us: null, source: 'mock · 结果拷回未采集' },
      };
    });
    const rounds = Array.isArray(supplied.rounds) && supplied.rounds.length ? supplied.rounds.map((round, i) => ({
      id: round.id || round.inv || ('round ' + (i + 1)),
      bindH2d: round.bindH2d || round.bind_h2d || {},
      compileRegister: round.compileRegister || round.compile_register || {},
      resultCopy: round.resultCopy || round.result_copy || {},
    })) : defaultRounds;
    const servingWait = supplied.servingWait || supplied.serving_wait || null;
    return { benchmark: resolvedBenchmark, workers: workers, workerTasks: workerTasks, lanes: lanes, rounds: rounds, servingWait: servingWait };
  }

  function triageMedian(values) {
    const sorted = values.filter((v) => v != null).slice().sort((a, b) => a - b);
    if (!sorted.length) return 0;
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }

  function triageWorkerOutlier(workers) {
    if (!workers || workers.length < 2) return null;
    const midCount = triageMedian(workers.map((worker) => worker.count));
    const midMax = triageMedian(workers.map((worker) => worker.maxUs));
    return workers.find((worker) => worker.count > midCount * 1.25 || worker.count < midCount * 0.75 || worker.maxUs > midMax * 1.25) || null;
  }

  function triageDecision(triage) {
    const b = triage.benchmark;
    const outlier = triageWorkerOutlier(triage.workers);
    if (outlier) return {
      domain: 'serving', title: 'Serving 分配待查',
      detail: outlier.id + ' 的 Count / Max 偏离同组',
      action: '排查请求分配、队列与该 worker 的下游依赖', outlier: outlier,
    };
    if (b.deviceWallUs != null && b.deviceWallUs > b.hostWallUs) return {
      domain: 'device', title: 'Device 执行主导',
      detail: 'device_wall_us 占主导',
      action: '采集 Chip Swimlane 与依赖图',
    };
    return {
      domain: 'host', title: 'Host 编排主导',
      detail: b.deviceWallUs == null ? 'Host 已采集；device_wall_us 尚未采集' : 'host_wall_us 高于 device_wall_us',
      action: '常驻 weights / KV cache / workspace，register-once、dispatch-many 后复测',
    };
  }

  function renderE2EFlowMap(stage, triage, decision) {
    const b = triage.benchmark;
    const servingWait = triage.servingWait && triageNumber(triage.servingWait, 'avgUs', 'avg_us');
    const sec = el('section', 'tc-e2e-command-overview');
    sec.id = 'e2e-triage';
    sec.appendChild(sectionHead('端到端运行总览', '把一次真实请求的 Serving、Host 与 Device 信号放到同一张首屏，先定界再下钻',
      el('span', 'tc-readout', b.measured ? b.source + ' · 同一 scope 实测' : b.source + ' · scope 不完整')));
    const grid = el('div', 'tc-e2e-command-grid');
    const percentOfE2E = (value) => b.e2eWallUs && value != null
      ? Math.max(4, Math.min(100, value / b.e2eWallUs * 100)) : 12;
    const domainCard = (item) => {
      const card = el(item.panel ? 'button' : 'div', 'tc-e2e-command-card');
      if (item.panel) card.type = 'button';
      card.dataset.domain = item.id;
      if (item.active) card.dataset.active = 'true';
      card.appendChild(el('span', 'eyebrow', item.label));
      card.appendChild(el('strong', 'value', item.value));
      card.appendChild(el('span', 'caption', item.caption));
      const meter = el('span', 'meter');
      meter.style.setProperty('--tc-meter', item.share + '%');
      card.appendChild(meter);
      card.appendChild(el('small', 'detail', item.detail));
      if (item.panel) card.addEventListener('click', () => { S.e2ePanel = item.panel; render(); });
      return card;
    };
    grid.appendChild(domainCard({
      id: 'serving', label: 'Serving', value: servingWait != null ? msOrUs(servingWait) : 'WorkerProcess',
      caption: servingWait != null ? 'wait_worker_output 平均等待' : '请求分配 / 队列',
      detail: triage.workers.length + ' 个 WorkerProcess · ' + (triage.workers.length < 2 ? '不可比较' : '可查看分布'),
      share: percentOfE2E(servingWait), panel: 'serving', active: decision.domain === 'serving',
    }));
    grid.appendChild(domainCard({
      id: 'host', label: 'Host', value: msOrUs(b.hostWallUs), caption: 'bind / 注册 / 编排',
      detail: b.hostWallUs != null ? '占端到端 ' + num(percentOfE2E(b.hostWallUs), 1) + '%' : 'Host wall 未采集',
      share: percentOfE2E(b.hostWallUs), panel: 'serving', active: decision.domain === 'host',
    }));
    grid.appendChild(domainCard({
      id: 'device', label: 'Device', value: msOrUs(b.deviceWallUs),
      caption: b.deviceWallUs == null ? 'device_wall_us 未采集' : '执行路径',
      detail: b.deviceWallUs == null ? '进入 Device 轨迹前需补采' : '占端到端 ' + num(percentOfE2E(b.deviceWallUs), 1) + '%',
      share: percentOfE2E(b.deviceWallUs), panel: 'device', active: decision.domain === 'device',
    }));
    const action = el('section', 'tc-e2e-action-card');
    action.dataset.domain = decision.domain;
    action.appendChild(el('span', 'eyebrow', '本轮建议'));
    action.appendChild(el('strong', null, decision.title));
    action.appendChild(el('p', null, decision.detail));
    action.appendChild(el('small', null, decision.action));
    const actionButton = btn(decision.domain === 'device' ? '进入 Device 轨迹' : '查看 Serving / Host', {
      size: 'sm', on: () => { S.e2ePanel = decision.domain === 'device' ? 'device' : 'serving'; render(); },
    });
    actionButton.classList.add('tc-e2e-action-button');
    action.appendChild(actionButton);
    grid.appendChild(action);
    sec.appendChild(grid);
    stage.appendChild(sec);
  }

  function renderE2ECriticalPaths(stage) {
    const ranks = D.case.ranks.filter((rank) => D.ranks[rank] && D.ranks[rank].cpath);
    if (!ranks.length) return;
    const sec = el('section', 'tc-e2e-critical-paths');
    sec.appendChild(sectionHead('跨 Rank 关键链路', '横向位置与宽度来自各卡实测 trace；点击关键路径任务进入 L1'));
    const map = el('div', 'tc-e2e-critical-map');
    ranks.forEach((rank) => {
      const data = D.ranks[rank];
      const row = el('div', 'tc-e2e-critical-row' + (rank === S.rank ? ' is-armed' : ''));
      const label = el('button', 'rank');
      label.type = 'button';
      label.appendChild(el('strong', null, rank));
      label.appendChild(el('small', null, msOrUs(data.swimlane.spanUs) + ' trace'));
      label.addEventListener('click', () => { selectAnalysisRank(rank, 'l2'); S.focus = null; render(); });
      row.appendChild(label);
      const track = el('div', 'track');
      data.cpath.segments.forEach((segment) => {
        const task = data.tasks[segment.tag];
        if (!task) return;
        const mark = el('button', 'mark');
        mark.type = 'button';
        mark.dataset.stage = e2eStageOf(task).id;
        mark.style.left = (task.start / data.swimlane.spanUs * 100).toFixed(3) + '%';
        mark.style.width = Math.max(.7, task.span / data.swimlane.spanUs * 100).toFixed(3) + '%';
        mark.title = task.callable + ' · ' + msOrUs(task.span);
        mark.setAttribute('aria-label', mark.title);
        mark.addEventListener('click', () => e2eJump(rank, task));
        track.appendChild(mark);
      });
      row.appendChild(track);
      map.appendChild(row);
    });
    sec.appendChild(map);
    stage.appendChild(sec);
  }

  function renderE2ETriage(stage) {
    const triage = e2eTriageData();
    renderE2EFlowMap(stage, triage, triageDecision(triage));
  }

  function renderE2EWorkers(stage, triage, decision) {
    const sec = el('section');
    sec.id = 'e2e-workers';
    sec.appendChild(sectionHead('Serving Strace · WorkerProcess 任务', '横线 = Min → Max，圆点 = Avg，左侧数字 = 任务数',
      el('span', 'tc-readout', triage.servingWait ? 'wait_worker_output 平均 ' + num(triageNumber(triage.servingWait, 'avgUs', 'avg_us') / 1000, 1) + ' ms · Max ' + num(triageNumber(triage.servingWait, 'maxUs', 'max_us') / 1000, 1) + ' ms' : triage.workerTasks[0].source)));
    sec.appendChild(renderE2ERangeChart(triage.workerTasks, decision.outlier));
    stage.appendChild(sec);
  }

  function renderE2ERangeChart(records, outlier) {
    const min = Math.min.apply(null, records.map((worker) => worker.minUs || 0));
    const max = Math.max.apply(null, records.map((worker) => worker.maxUs || 0));
    const span = Math.max(1, max - min);
    const chart = el('div', 'tc-e2e-workers');
    records.forEach((worker) => {
      const row = el('div', 'tc-e2e-worker');
      if (outlier === worker) row.dataset.outlier = 'true';
      const label = el('div', 'label');
      label.appendChild(el('strong', null, worker.id));
      label.appendChild(el('small', null, 'Count ' + worker.count));
      row.appendChild(label);
      const range = el('div', 'range');
      const line = el('i', 'range-line');
      line.style.left = ((worker.minUs - min) / span * 100).toFixed(2) + '%';
      line.style.width = Math.max(1, (worker.maxUs - worker.minUs) / span * 100).toFixed(2) + '%';
      const avg = el('i', 'average');
      avg.style.left = ((worker.avgUs - min) / span * 100).toFixed(2) + '%';
      range.appendChild(line);
      range.appendChild(avg);
      row.appendChild(range);
      const nums = el('div', 'numbers');
      nums.appendChild(el('span', null, num(worker.minUs, 0)));
      nums.appendChild(el('strong', null, num(worker.avgUs, 0)));
      nums.appendChild(el('span', null, num(worker.maxUs, 0)));
      row.appendChild(nums);
      chart.appendChild(row);
    });
    return chart;
  }

  function renderE2ELanes(stage, triage) {
    if (!triage.lanes.length) return;
    const outlier = triageWorkerOutlier(triage.lanes);
    const sec = el('section');
    sec.id = 'e2e-lanes';
    sec.appendChild(sectionHead('Host STRACE · 16 路 NPU lane', '每路 55 次 chip.run；横线 = Min → Max，圆点 = Avg。它反映 Host 侧 lane 尾部，不等同多 WorkerProcess 分配。',
      el('span', 'tc-readout', outlier ? outlier.id + ' 的 Max 偏离同组' : 'lane 间未见明显偏斜')));
    sec.appendChild(renderE2ERangeChart(triage.lanes, outlier));
    stage.appendChild(sec);
  }

  function renderE2ERoundMatrix(stage, triage) {
    const labels = [
      { key: 'bindH2d', label: 'bind / H2D' },
      { key: 'compileRegister', label: 'compile / register' },
      { key: 'resultCopy', label: '结果拷回' },
    ];
    const sec = el('section');
    sec.id = 'e2e-rounds';
    sec.appendChild(sectionHead('每轮 Host 开销', '识别重复上传、编译 / 注册与结果拷回；每格显示“是否发生”而不是累计文字',
      el('span', 'tc-readout', '按 invocation 采样')));
    const matrix = el('div', 'tc-e2e-round-matrix');
    matrix.style.setProperty('--tc-rounds', triage.rounds.length);
    matrix.appendChild(el('span', 'corner', '开销 / 轮次'));
    triage.rounds.forEach((round) => matrix.appendChild(el('span', 'round', round.id)));
    labels.forEach((label) => {
      matrix.appendChild(el('strong', 'operation', label.label));
      triage.rounds.forEach((round) => {
        const cell = round[label.key] || {};
        const state = cell.state || 'unknown';
        const button = el('button', 'tc-e2e-round-cell');
        button.type = 'button';
        button.dataset.state = state;
        button.title = label.label + ' · ' + (cell.source || '来源未标记') + (cell.us != null ? ' · ' + num(cell.us, 1) + ' us' : '');
        button.appendChild(el('strong', null, state === 'repeat' ? '重复' : state === 'once' ? '仅首轮' : state === 'none' ? '未发生' : '未知'));
        const sourceFlag = /mock/i.test(cell.source || '') ? ' · mock' : /proxy/i.test(cell.source || '') ? ' · proxy' : '';
        button.appendChild(el('small', null, cell.us != null ? num(cell.us, 0) + ' us' + sourceFlag : (cell.source || 'mock')));
        matrix.appendChild(button);
      });
    });
    sec.appendChild(matrix);
    stage.appendChild(sec);
  }

  function e2eJump(rank, task) {
    S.rank = rank;
    S.task = task.tag;
    S.focus = 'task';
    S.view = 'l1';
    render();
  }

  function e2eStageLegend(projection) {
    const legend = el('div', 'tc-e2e-legend');
    projection.stages.filter((s) => projection.ranks.some((rank) => s.ranks[rank])).forEach((s) => {
      const key = el('span', 'tc-e2e-legend-key');
      key.dataset.stage = s.id;
      key.appendChild(el('i'));
      key.appendChild(el('span', null, s.label));
      legend.appendChild(key);
    });
    return legend;
  }

  function e2ePackedTasks(tasks) {
    const lanes = [];
    return tasks.slice().sort((a, b) => a.start - b.start || b.span - a.span).map((task) => {
      let lane = lanes.findIndex((end) => end <= task.start);
      if (lane < 0) { lane = lanes.length; lanes.push(task.end); }
      else lanes[lane] = task.end;
      return { task: task, lane: lane };
    });
  }

  /* Inspired by profiler timelines: rows are ranks, x is device time, and
   * every mark is a real trace task.  The model-stage colour is a grouping
   * layer, never a replacement for the underlying timing. */
  function renderE2ETraceAtlas(stage, projection) {
    const sec = el('section');
    sec.appendChild(sectionHead('执行时间地图', '共用任务色语义 · 点击任一 span 下钻到 L1'));
    sec.appendChild(e2eStageLegend(projection));
    const atlas = el('div', 'tc-e2e-atlas');
    projection.ranks.forEach((rank) => {
      const rankData = D.ranks[rank];
      const packed = e2ePackedTasks(rankData.tasks);
      const laneCount = Math.max.apply(null, packed.map((x) => x.lane)) + 1;
      const row = el('div', 'tc-e2e-atlas-row');
      const label = el('button', 'tc-e2e-atlas-label' + (rank === S.rank ? ' is-armed' : ''));
      label.type = 'button';
      const deviceWall = D.e2e[rank][TRACE_MATCH[rank].inv]['chip.run.runner_run.device_wall'].us;
      label.appendChild(el('strong', null, rank));
      label.appendChild(el('span', null, 'wall ' + num(deviceWall, 0) + ' · trace ' + num(rankData.swimlane.spanUs, 0)));
      label.appendChild(el('small', null, pct(rankData.occupancy.aicUtil, 0) + ' AIC · ' + pct(rankData.occupancy.aivUtil, 0) + ' AIV'));
      label.addEventListener('click', () => { selectAnalysisRank(rank, 'l2'); S.focus = null; render(); });
      row.appendChild(label);
      const track = el('div', 'tc-e2e-atlas-track');
      track.style.height = Math.max(64, laneCount * 16 + 12) + 'px';
      [0, 25, 50, 75, 100].forEach((p) => {
        const tick = el('i', 'tick');
        tick.style.left = p + '%';
        if (p < 100) tick.appendChild(el('span', null, num(rankData.swimlane.spanUs * p / 100, 0)));
        track.appendChild(tick);
      });
      packed.forEach((entry) => {
        const task = entry.task;
        const stageInfo = e2eStageOf(task);
        const mark = el('button', 'tc-e2e-trace-mark');
        mark.type = 'button';
        mark.dataset.stage = stageInfo.id;
        if (rankData.critical.tags.indexOf(task.tag) >= 0) mark.dataset.critical = 'true';
        mark.style.left = clamp(task.start / rankData.swimlane.spanUs * 100, 0, 100).toFixed(3) + '%';
        mark.style.width = Math.max(0.55, task.span / rankData.swimlane.spanUs * 100).toFixed(3) + '%';
        mark.style.top = (entry.lane * 16 + 8) + 'px';
        mark.title = task.callable + ' · ' + num(task.span, 2) + ' us · ' + (rankData.critical.tags.indexOf(task.tag) >= 0 ? '依赖关键路径' : 'trace task');
        mark.setAttribute('aria-label', mark.title);
        mark.addEventListener('click', () => e2eJump(rank, task));
        track.appendChild(mark);
      });
      row.appendChild(track);
      atlas.appendChild(row);
    });
    sec.appendChild(atlas);
    stage.appendChild(sec);
  }

  /* A 100% composition bar separates “where the device spent trace work”
   * from the wall-clock view above.  Each segment opens its longest scope. */
  function renderE2EStageComposition(stage, projection) {
    const sec = el('section');
    sec.appendChild(sectionHead('阶段工作构成', '各卡独立归一 · segment 宽度 = trace core-time'));
    const chart = el('div', 'tc-e2e-composition');
    projection.ranks.forEach((rank) => {
      const cells = projection.stages.filter((s) => s.ranks[rank]);
      const total = cells.reduce((sum, s) => sum + s.ranks[rank].coreUs, 0) || 1;
      const row = el('div', 'tc-e2e-composition-row');
      const label = el('span', 'rank');
      label.appendChild(el('strong', null, rank));
      label.appendChild(el('small', null, num(total, 0) + ' us work'));
      row.appendChild(label);
      const barHost = el('div', 'tc-e2e-composition-bar');
      cells.forEach((s) => {
        const cell = s.ranks[rank];
        const segment = el('button', 'tc-e2e-composition-segment');
        segment.type = 'button';
        segment.dataset.stage = s.id;
        segment.style.flexGrow = cell.coreUs;
        segment.title = s.label + ' · ' + num(cell.coreUs, 1) + ' us / ' + cell.count + ' tasks · 下钻到 ' + cell.max.callable;
        segment.setAttribute('aria-label', segment.title);
        if (D.ranks[rank].critical.tags.indexOf(cell.max.tag) >= 0) segment.dataset.critical = 'true';
        segment.addEventListener('click', () => e2eJump(rank, cell.max));
        barHost.appendChild(segment);
      });
      row.appendChild(barHost);
      chart.appendChild(row);
    });
    sec.appendChild(chart);
    stage.appendChild(sec);
  }

  /* A scatter plot makes cross-card skew visible without forcing the reader
   * to compare two columns of numbers.  Upper-right = expensive on both;
   * off-diagonal = rank-specific work or imbalance. */
  function renderE2EOperatorScatter(stage, projection) {
    const sec = el('section');
    const ranks = projection.ranks.slice(0, 2);
    if (ranks.length < 2) { renderFuncSummary(stage); return; }
    const ops = projection.operators.map((op) => {
      op.x = op.ranks[ranks[0]] ? op.ranks[ranks[0]].coreUs : 0;
      op.y = op.ranks[ranks[1]] ? op.ranks[ranks[1]].coreUs : 0;
      op.maxCore = Math.max(op.x, op.y);
      op.jumpRank = op.x >= op.y ? ranks[0] : ranks[1];
      return op;
    }).filter((op) => op.maxCore > 0).sort((a, b) => a.maxCore - b.maxCore).slice(-32);
    const max = Math.max.apply(null, ops.map((op) => Math.max(op.x, op.y))) || 1;
    sec.appendChild(sectionHead('算子偏斜散点', ranks[0] + ' × ' + ranks[1] + ' · 右上 = 双卡共同热点，偏轴 = rank 偏斜'));
    sec.appendChild(e2eStageLegend(projection));
    const plot = el('div', 'tc-e2e-scatter');
    plot.appendChild(el('span', 'axis axis-y', ranks[1] + ' core-time'));
    plot.appendChild(el('span', 'axis axis-x', ranks[0] + ' core-time'));
    ops.forEach((op) => {
      const point = el('button', 'tc-e2e-scatter-point');
      point.type = 'button';
      point.dataset.stage = op.stage;
      if (op.pathRanks.length) point.dataset.critical = 'true';
      const scale = (value) => Math.log1p(value) / Math.log1p(max);
      point.style.left = (6 + scale(op.x) * 88).toFixed(2) + '%';
      point.style.bottom = (8 + scale(op.y) * 84).toFixed(2) + '%';
      const size = 8 + scale(op.maxCore) * 14;
      point.style.width = size.toFixed(1) + 'px';
      point.style.height = size.toFixed(1) + 'px';
      point.title = op.name + ' · ' + ranks[0] + ' ' + num(op.x, 1) + ' us · ' + ranks[1] + ' ' + num(op.y, 1) + ' us'
        + (op.pathRanks.length ? ' · 依赖关键路径' : '');
      point.setAttribute('aria-label', point.title);
      point.addEventListener('click', () => e2eJump(op.jumpRank, op.ranks[op.jumpRank].max));
      plot.appendChild(point);
    });
    sec.appendChild(plot);
    stage.appendChild(sec);
  }

  function viewE2E(stage) {
    /* No host STRACE log in this dump either, but a torch profiler run of the
     * same program is an end-to-end layer measured a different way. */
    if (!hasE2E() && QW()) { viewE2EQwen3(stage); return; }
    if (!hasE2E()) { viewE2EAbsent(stage); renderFuncSummary(stage); return; }

    const projection = e2eProjection();
    if (S.e2ePanel === 'triage') {
      renderE2ETriage(stage);
      renderE2ECriticalPaths(stage);
      if (multiRank()) renderE2ECrossRankScheduler(stage);
      return;
    }
    if (S.e2ePanel === 'serving') {
      const triage = e2eTriageData();
      const decision = triageDecision(triage);
      renderE2EWorkers(stage, triage, decision);
      renderE2ELanes(stage, triage);
      renderE2ERoundMatrix(stage, triage);
      return;
    }
    if (S.e2ePanel === 'device') {
      renderE2ETraceAtlas(stage, projection);
      renderE2EStageComposition(stage, projection);
      renderE2EOperatorScatter(stage, projection);
      return;
    }

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
    secTab.appendChild(sectionHead('调用采样', 'device_wall 为设备时钟',
      el('span', 'tc-readout', '点行 = 选中要带去 L2 / L1 的 rank')));
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
      onPick: (r) => { selectAnalysisRank(r.rank, 'l2'); S.focus = null; render(); },
    }));
    stage.appendChild(secTab);

    /* --- hierarchical span breakdown, both ranks on one shared scale --- */
    const rankKeys = Object.keys(D.ranks);
    const spans = {};
    rankKeys.forEach((r) => { spans[r] = D.e2e[r][TRACE_MATCH[r].inv]; });
    /* one denominator for every bar, so the two columns compare directly */
    const total = Math.max.apply(null, rankKeys.map((r) => spans[r]['chip.run'].us));
    const secBreak = el('section');
    secBreak.appendChild(sectionHead('调用剖分', rankKeys.length + ' rank · 共用刻度 · STRACE host span，device_wall 及子段为设备时钟'));
    const rowsWrap = el('div', 'tc-spanrows');
    const shead = el('div', 'tc-spanrow tc-spanhead');
    shead.appendChild(el('span', 'lbl', 'span'));
    rankKeys.forEach((r) => {
      shead.appendChild(el('span', 'h' + (r === S.rank ? ' is-armed' : ''), r + ' inv=' + TRACE_MATCH[r].inv));
      shead.appendChild(el('span', 'h val' + (r === S.rank ? ' is-armed' : ''), 'us'));
    });
    rowsWrap.appendChild(shead);
    SPAN_TREE.forEach((entry) => {
      const name = entry[0];
      const depth = entry[1];
      if (!rankKeys.some((r) => spans[r][name])) return;
      const row = el('div', 'tc-spanrow');
      row.dataset.depth = depth;
      row.appendChild(el('span', 'lbl', name.replace(/^chip\.run\.?/, '') || 'chip.run'));
      const tone = /graph_build/.test(name) ? 'warn' : /device_wall$/.test(name) ? 'good' : /sched/.test(name) ? 'neutral' : null;
      rankKeys.forEach((r) => {
        const span = spans[r][name];
        if (!span) {
          row.appendChild(el('span', 'tc-bar-empty'));
          row.appendChild(el('span', 'val muted', '—'));
          return;
        }
        row.appendChild(bar(span.us / total, tone));
        row.appendChild(el('span', 'val' + (r === S.rank ? ' is-armed' : ''), num(span.us, 2)));
      });
      rowsWrap.appendChild(row);
    });
    secBreak.appendChild(rowsWrap);
    stage.appendChild(secBreak);

    /* --- reconciliation: host span vs device trace --- */
    const recSec = el('section');
    recSec.appendChild(sectionHead('对账', 'host sched ↔ device trace'));
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
      { label: '依赖关键路径', cell: (r) => r.crit + ' 节点', num: true },
      { label: 'AIC 占用', num: true, cell: (r) => pct(r.aic) },
      { label: 'AIV 占用', num: true, cell: (r) => pct(r.aiv) },
    ], recRows, { onPick: (r) => { selectAnalysisRank(r.rank, 'l2'); render(); } }));
    stage.appendChild(recSec);
    renderFuncSummary(stage);
  }

  /* The L2 dump has no host STRACE log, so there is no end-to-end layer to
   * show. This is a state, not an error: name the missing artifact, say what
   * it would have answered, and point at the layer that still works. */
  function viewE2EAbsent(stage) {
    const sec = el('section');
    sec.appendChild(sectionHead('端到端', '本 dump 缺少 host STRACE log'));
    sec.appendChild(table([
      { label: '缺失产物', cell: (r) => esc(r[0]), mono: true },
      { label: '本可回答', cell: (r) => esc(r[1]) },
      { label: '状态', cell: () => '<span class="bad">缺失</span>' },
    ], [
      ['dfx_outputs/**/host.*.log', 'chip.run / bind / runner_run / device_wall 的 span 树'],
      ['  └ inv=', '本次录制里程序被调用了几次（迭代次数 n）'],
      ['  └ device_wall', '设备墙钟，调优的主指标与复测基准'],
      ['  └ bind.prebuilt', 'JIT 建图是否命中缓存，第一次调用能不能用'],
      ['distributed_meta.json', '绑定参数的 shape / dtype，case 是否固定'],
    ], {}));
    stage.appendChild(sec);

    const alt = el('section');
    alt.appendChild(sectionHead('仍然可测', '设备侧 trace 完整'));
    const R0 = R();
    alt.appendChild(tiles([
      { k: 'trace span', v: num(R0.swimlane.spanUs, 1), u: 'us（设备钟）' },
      { k: '任务', v: String(R0.tasks.length) },
      { k: '块', v: String(R0.swimlane.blocks.reduce((a, b) => a + b.length, 0)) },
      { k: '依赖关键路径', v: R0.critical.tags.length, u: '节点 · 静态 CPM' },
      { k: 'AIC 占用', v: pct(R0.occupancy.aicUtil), tone: R0.occupancy.aicUtil < 40 ? 'bad' : 'good' },
      { k: 'AIV 占用', v: pct(R0.occupancy.aivUtil), tone: R0.occupancy.aivUtil < 40 ? 'bad' : null },
    ]));
    const jump = el('div', 'tc-actions');
    jump.appendChild(btn('去 L2 调度', { on: () => { S.view = 'l2'; render(); } }));
    jump.appendChild(btn('去 ISA / 布局', { variant: 'ghost', on: () => { S.view = 'isa'; render(); } }));
    alt.appendChild(jump);
    stage.appendChild(alt);
  }

  /* ================================================ Qwen3 profile layers
   * pypto_qwen3_profiles holds four captures of one 40-layer graph. They do
   * not answer the same questions: only tp1/prefill carries task names and a
   * dependency graph, so only it feeds the task-level machinery above. The
   * rest are served here, and every screen states which capture it is reading
   * and what that capture cannot say.
   *
   * The reader picks a capture x stage once; E2E, L2 and L1/L0 all follow it. */
  const QW = () => D.qwen3 || null;
  const qwVariant = () => {
    const q = QW();
    if (!q) return null;
    return q.variants.find((v) => v.id === S.variant)
      || q.variants.find((v) => v.primary) || q.variants[0];
  };
  const qwCapture = () => { const v = qwVariant(); return v ? QW().captures[v.capture] : null; };
  const qwL2 = () => { const v = qwVariant(); return v ? QW().l2[v.id] : null; };
  const qwLayers = () => { const v = qwVariant(); return v ? QW().layers[v.id] : null; };
  const qwTorch = () => { const v = qwVariant(); return v ? QW().torch[v.capture] : null; };
  /* the one variant data.js carries a full rank for */
  const onPrimaryVariant = () => { const v = qwVariant(); return !!(v && v.primary); };

  const msOrUs = (v) => (v == null ? '—'
    : Math.abs(v) >= 1000 ? num(v / 1000, 2) + ' ms' : num(v, 1) + ' us');

  /* The capture chooser. Two axes, because that is how the dataset is laid
   * out: two collectors x two stages. */
  function qwVariantBar(onPick) {
    const q = QW();
    const cur = qwVariant();
    const wrap = el('div', 'tc-variant-bar');
    Object.keys(q.captures).forEach((cid) => {
      const cap = q.captures[cid];
      const grp = el('div', 'tc-variant-group');
      const head = el('div', 'tc-variant-head');
      head.appendChild(el('span', 'n', cap.label));
      const badge = el('span', 'tc-variant-badge');
      badge.dataset.state = cap.validated ? 'ok' : 'warn';
      badge.textContent = cap.validated ? '已校验' : '未校验';
      badge.title = cap.validateNote;
      head.appendChild(badge);
      grp.appendChild(head);
      const row = el('div', 'tc-variant-row');
      q.variants.filter((v) => v.capture === cid).forEach((v) => {
        const b = el('button', 'tc-variant-pill' + (v.id === cur.id ? ' is-selected' : ''));
        b.type = 'button';
        b.appendChild(el('span', 'n', v.stage));
        b.appendChild(el('span', 'm', msOrUs(v.spanUs)));
        b.title = v.taskCount + ' 任务 · ' + v.blockRows + ' 块 · 用到 '
          + v.coresUsed + '/' + v.coreTotal + ' 核'
          + (v.gaps.length ? NL + '缺口：' + v.gaps.join(NL) : '');
        b.addEventListener('click', () => {
          if (v.id === S.variant) return;
          S.variant = v.id;
          if (onPick) onPick(v);
          render();
        });
        row.appendChild(b);
      });
      grp.appendChild(row);
      wrap.appendChild(grp);
    });
    return wrap;
  }

  /* What this capture cannot answer. A list, not a disclaimer: each line names
   * the missing artifact and the layer it takes down. */
  function qwGapNote(host) {
    const v = qwVariant();
    if (!v || !v.gaps.length) return;
    const sec = el('section');
    sec.appendChild(sectionHead('本采集的缺口', v.label + ' · ' + (v.validated ? '数据集已校验' : '数据集未校验'),
      el('span', 'tc-readout', v.hasNames ? (v.namesFrom || '有任务名') : '任务无名')));
    sec.appendChild(table([
      { label: '缺失', cell: (r) => esc(r) },
      { label: '状态', cell: () => '<span class="bad">缺失</span>', width: '72px' },
    ], v.gaps, {}));
    host.appendChild(sec);
  }

  function qwCaptureNote(host) {
    const v = qwVariant();
    const cap = qwCapture();
    const sec = el('section');
    sec.appendChild(sectionHead('采集来源', cap.collector + ' · ' + QW().capturedAt,
      el('span', 'tc-readout', QW().source)));
    const note = el('div', 'tc-note');
    note.dataset.tone = cap.validated ? 'ok' : 'warn';
    note.appendChild(el('strong', null, cap.label + '：' + (cap.validated ? '数据集认定的交付基线' : '数据集未认定为交付基线')));
    note.appendChild(el('span', null, cap.validateNote));
    note.appendChild(el('small', null, QW().dataset.note));
    sec.appendChild(note);
    host.appendChild(sec);
  }

  /* ---------------------------------------------------- E2E: torch profiler
   * The dump has no host STRACE log, so there is no span tree. What it does
   * have is a torch profiler run of the same program: per-step Computing /
   * Free / Preparing, the framework ops around the fused kernel, and the host
   * ACL bill. That is an end-to-end layer, measured differently. */
  function qwStepPanel(stage) {
    const t = qwTorch();
    const v = qwVariant();
    const cap = qwCapture();
    const dm = t.decodeMean;

    const sec = el('section');
    sec.appendChild(sectionHead('步时间归因', 'torch profiler · device ' + t.deviceId
      + ' · 1 次 prefill + ' + t.iters.decode + ' 次 decode',
      el('span', 'tc-readout', 'step_trace_time.csv · Stage = Computing + Free + Preparing')));
    sec.appendChild(tiles([
      { k: 'decode step', v: num(dm.stageUs / 1000, 2), u: 'ms · 均值' },
      { k: 'Computing', v: num(dm.computingUs / 1000, 2), u: 'ms · 设备在算' },
      { k: 'Free', v: num(dm.freeUs / 1000, 2), u: 'ms · 设备空闲', tone: dm.freeShare > 30 ? 'bad' : null },
      { k: 'Free 占比', v: pct(dm.freeShare), tone: dm.freeShare > 30 ? 'bad' : 'good' },
      { k: 'Preparing', v: num(dm.preparingUs, 0), u: 'us' },
      { k: '主机 ACL 调用', v: num(t.apiHost.workUs / 1000, 1), u: 'ms · 5 步合计' },
    ]));

    /* one stacked bar per step, on a shared scale */
    const maxStage = Math.max.apply(null, t.steps.map((s) => s.stageUs));
    const rows = el('div', 'tc-stepbars');
    t.steps.forEach((s) => {
      const row = el('div', 'tc-stepbar');
      row.dataset.stage = s.stage;
      const lab = el('div', 'label');
      lab.appendChild(el('strong', null, 'step ' + s.step));
      lab.appendChild(el('small', null, s.stage));
      row.appendChild(lab);
      const track = el('div', 'track');
      [['computing', s.computingUs, 'Computing'], ['free', s.freeUs, 'Free'],
        ['preparing', s.preparingUs, 'Preparing']].forEach((seg) => {
        if (!seg[1]) return;
        const part = el('span', 'seg');
        part.dataset.kind = seg[0];
        part.style.width = (seg[1] / maxStage * 100).toFixed(3) + '%';
        part.title = seg[2] + ' ' + num(seg[1] / 1000, 2) + ' ms';
        track.appendChild(part);
      });
      row.appendChild(track);
      row.appendChild(el('div', 'val', num(s.stageUs / 1000, 2) + ' ms'));
      row.appendChild(el('div', 'val muted', pct(s.freeShare) + ' free'));
      rows.appendChild(row);
    });
    sec.appendChild(rows);

    const verdict = el('div', 'tc-e2e-triage-verdict');
    verdict.dataset.domain = dm.freeShare > 30 ? 'host' : 'device';
    if (dm.freeShare > 30) {
      verdict.appendChild(el('strong', null, '主机受限：decode 每步 '
        + num(dm.stageUs / 1000, 1) + ' ms 里设备空闲 ' + num(dm.freeUs / 1000, 1) + ' ms'));
      verdict.appendChild(el('span', null, '设备只用掉 ' + pct(dm.computeShare)
        + '，其余是等主机。融合 kernel 之外还有 ' + t.deviceMix.frameworkCalls
        + ' 次框架算子，主机侧 ACL 调用 ' + num(t.apiHost.workUs / 1000, 1)
        + ' ms（不含同步等待 ' + num(t.apiHost.waitUs / 1000, 0) + ' ms）。'));
      verdict.appendChild(el('small', null, '下一步：先看「设备算子」与「主机 API」，'
        + '把 kernel 内部的优化排到主机开销之后。'));
    } else {
      verdict.appendChild(el('strong', null, '设备受限：decode 每步 '
        + num(dm.stageUs / 1000, 1) + ' ms 里空闲只有 ' + pct(dm.freeShare)));
      verdict.appendChild(el('span', null, '融合 kernel 占设备时间 '
        + pct(t.deviceMix.fusedShare) + '，框架算子只有 ' + t.deviceMix.frameworkCalls
        + ' 次。瓶颈在 kernel 内部，去 L2 / L1 继续。'));
      verdict.appendChild(el('small', null, '下一步：看 L2 的 40 层视图与 L1 的 PMU 流水占比。'));
    }
    sec.appendChild(verdict);
    stage.appendChild(sec);

    /* every invocation of the fused kernel, with its own PMU */
    const fsec = el('section');
    fsec.appendChild(sectionHead('融合 kernel 每次调用', 'aicore_kernel_0 · ' + cap.label
      + ' · 整张 40 层图一次调用',
      el('span', 'tc-readout', 'kernel_details.csv · ' + t.kernelRows + ' 行')));
    fsec.appendChild(table([
      { label: 'step', key: 'step', num: true },
      { label: 'stage', cell: (r) => esc(r.stage) },
      { label: '时长', num: true, cell: (r) => num(r.durUs / 1000, 2) + ' ms' },
      { label: 'Wait', num: true, cell: (r) => num(r.waitUs, 0) },
      { label: 'block', cell: (r) => r.blocks + ' + ' + r.mixBlocks + ' mix' },
      { label: 'AIC mac', num: true, cell: (r) => pmuCell(r.aic.mac) },
      { label: 'AIC mte2', num: true, cell: (r) => pmuCell(r.aic.mte2) },
      { label: 'AIC scalar', num: true, cell: (r) => pmuCell(r.aic.scalar) },
      { label: 'AIV vec', num: true, cell: (r) => pmuCell(r.aiv.vec) },
      { label: 'AIV scalar', num: true, cell: (r) => pmuCell(r.aiv.scalar) },
      { label: 'cube 利用率', num: true, cell: (r) => num(r.cubeUtil, 1) },
    ], t.fused.map((f) => Object.assign({ __selected: f.stage === v.stage }, f)), {}));
    fsec.appendChild(el('p', 'tc-fineprint',
      '流水占比是各流水线相对 aicore_time 的忙占比，互相重叠，不能相加成 100%。'
      + 'PMU 开着采的这一轮不能与 PMU 关闭的基线直接比较。'));
    stage.appendChild(fsec);
  }
  const pmuCell = (v) => (v == null ? '—'
    : '<span class="' + (v >= 0.6 ? 'warn' : v <= 0.02 ? 'muted' : '') + '">' + num(v, 3) + '</span>');

  function qwOpsPanel(stage) {
    const t = qwTorch();
    const sec = el('section');
    sec.appendChild(sectionHead('设备算子构成', '融合 kernel 之外还在设备上跑的东西',
      el('span', 'tc-readout', 'op_statistic.csv · 1 prefill + ' + t.iters.decode + ' decode 合计')));
    sec.appendChild(tiles([
      { k: '融合 kernel', v: num(t.deviceMix.fusedUs / 1000, 1), u: 'ms' },
      { k: '框架算子', v: num(t.deviceMix.frameworkUs / 1000, 1), u: 'ms' },
      { k: '融合占比', v: pct(t.deviceMix.fusedShare), tone: t.deviceMix.fusedShare > 95 ? 'good' : null },
      { k: '框架算子种类', v: String(t.deviceMix.frameworkOps) },
      { k: '框架算子调用', v: String(t.deviceMix.frameworkCalls), u: '次',
        tone: t.deviceMix.frameworkCalls > 500 ? 'bad' : null },
    ]));
    const maxUs = Math.max.apply(null, t.ops.map((o) => o.totalUs));
    sec.appendChild(table([
      { label: 'OP', cell: (r) => esc(r.type), mono: true },
      { label: '角色', cell: (r) => ({ fused: '<span class="ok">融合 kernel</span>',
        launcher: '<span class="muted">AI_CPU 启动器</span>' }[r.role] || '框架') },
      { label: 'Core', cell: (r) => esc(r.core) },
      { label: '次数', key: 'count', num: true },
      { label: '合计', num: true, cell: (r) => num(r.totalUs, 1) },
      { label: '', cell: (r) => bar(r.totalUs / maxUs, r.role === 'fused' ? 'good' : r.role === 'launcher' ? 'neutral' : 'warn') },
      { label: '均值', num: true, cell: (r) => num(r.avgUs, 2) },
      { label: '占比', num: true, cell: (r) => pct(r.ratio) },
    ], t.ops, { tall: true }));
    sec.appendChild(el('p', 'tc-fineprint',
      'simpler_aicpu_exec_* 是同一个融合 kernel 的 AI_CPU 启动器，与 aicore_kernel_0 '
      + '是同一份工作的两条记录，不能把两行相加。'));
    stage.appendChild(sec);
  }

  function qwApiPanel(stage) {
    const t = qwTorch();
    const sec = el('section');
    sec.appendChild(sectionHead('主机 API 账单', '哪几个 CANN 调用吃掉了主机时间',
      el('span', 'tc-readout', 'api_statistic.csv · Level=acl / node')));
    sec.appendChild(tiles([
      { k: '主机调用耗时', v: num(t.apiHost.workUs / 1000, 1), u: 'ms · 不含同步等待' },
      { k: '同步等待', v: num(t.apiHost.waitUs / 1000, 1), u: 'ms · 等设备' },
      { k: 'API 条目', v: String(t.apiHost.rows) },
      { k: 'decode 每步 Free', v: num(t.decodeMean.freeUs / 1000, 1), u: 'ms · 对照' },
    ]));
    const maxUs = Math.max.apply(null, t.api.filter((a) => !a.isWait).map((a) => a.totalUs)) || 1;
    sec.appendChild(table([
      { label: 'API', cell: (r) => esc(r.name), mono: true },
      { label: 'Level', cell: (r) => esc(r.level) },
      { label: '合计', num: true, cell: (r) => num(r.totalUs, 1) },
      { label: '', cell: (r) => (r.isWait ? el('span', 'tc-readout', '等设备') : bar(r.totalUs / maxUs, 'warn')) },
      { label: '次数', key: 'count', num: true },
      { label: '均值', num: true, cell: (r) => num(r.avgUs, 2) },
      { label: '最大', num: true, cell: (r) => num(r.maxUs, 1) },
    ], t.api, { tall: true }));
    sec.appendChild(el('p', 'tc-fineprint',
      'Synchronize* 是主机在等设备算完，不是主机开销；它被单列出来，不计入「主机调用耗时」。'
      + '主机调用耗时与 Free 不是同一把尺子：Free 是设备侧空闲，二者只能相互印证方向。'));
    stage.appendChild(sec);
  }

  function qwTopoPanel(stage) {
    const q = QW();
    const rows = q.topology.rows;
    const sec = el('section');
    sec.appendChild(sectionHead('TP=1 ↔ TP=2 对照', '同一个 40 层 Qwen3 14B 图，两种并行拓扑',
      el('span', 'tc-readout', 'tp2 已校验 / tp1 未校验 · 同机同日，两个采集脚本')));
    const fmt = (v, unit) => {
      if (v == null) return '—';
      if (unit === 'us') return num(v / 1000, 2) + ' ms';
      if (unit === '%') return pct(v);
      if (unit === '×' || unit === '核') return num(v, 3);
      return String(v) + (unit ? ' ' + unit : '');
    };
    sec.appendChild(table([
      { label: '指标', cell: (r) => esc(r.metric) },
      { label: 'TP=1', num: true, cell: (r) => fmt(r.tp1, r.unit) },
      { label: 'TP=2', num: true, cell: (r) => fmt(r.tp2, r.unit) },
      { label: 'TP2 / TP1', num: true, cell: (r) => (r.tp1 && r.tp2 && typeof r.tp1 === 'number'
        ? (function () {
          const k = r.tp2 / r.tp1;
          const cls = k >= 2 || k <= 0.5 ? 'warn' : '';
          return '<span class="' + cls + '">' + num(k, 2) + 'x</span>';
        }()) : '—') },
      { label: '口径', cell: (r) => (r.note ? esc(r.note) : '—') },
    ], rows, { tall: true }));
    stage.appendChild(sec);

    const t1 = q.torch.tp1;
    const t2 = q.torch.tp2;
    const d1 = q.l2['tp1:decode'];
    const d2 = q.l2['tp2:decode'];
    const ly2 = q.layers['tp2:decode'];
    const worst = ly2 && ly2.steps
      ? ly2.steps.slice().sort((a, b) => b.sumUs - a.sumUs)[0] : null;
    const vsec = el('section');
    vsec.appendChild(sectionHead('这张对照说了什么', '只写两份数据都能支持的部分'));
    const list = el('div', 'tc-verdicts');
    [
      ['两边卡在不同地方',
        'TP=1 的 decode step 有 ' + pct(t1.decodeMean.freeShare) + ' 是设备空闲，'
        + '瓶颈在主机；TP=2 只有 ' + pct(t2.decodeMean.freeShare) + '，瓶颈在 kernel 内部。'
        + '同一个模型，两套拓扑要用两套优先级。'],
      ['TP=2 基本没并起来',
        'TP=2 的 decode 平均只有 ' + num(d2.occ.busyCores, 2) + ' 个核在忙（核时 / 墙钟），'
        + '用到 ' + d2.coresUsed + '/' + d2.coreTotal + ' 个核；TP=1 同期是 '
        + num(d1.occ.busyCores, 1) + ' 个核、' + d1.coresUsed + '/' + d1.coreTotal + ' 个核。'
        + '643 个任务几乎一个接一个跑完。'],
      ['TP=2 的核在做标量',
        'TP=2 融合 kernel 的 AIC mac 占比只有 '
        + num(q.topology.rows.find((r) => r.metric === 'AIC mac 占比').tp2, 3)
        + '，scalar 占比 '
        + num(q.topology.rows.find((r) => r.metric === 'AIC scalar 占比').tp2, 3)
        + '；AIV vec 占比 '
        + num(q.topology.rows.find((r) => r.metric === 'AIV vec 占比').tp2, 3)
        + '。核是忙的，但忙在标量而不是计算。'],
      worst ? ['单步就占掉半层',
        'TP=2 每层 ' + msOrUs(ly2.steady.medianSpanUs) + ' 里，'
        + worst.name + ' 一步就是 ' + msOrUs(worst.meanUs) + '（' + pct(worst.share) + ' 的层内核时）。']
        : null,
      ['不能做的比较',
        'tp1 被数据集自己的 .gitignore 排除、没有验收标记，两份采集也用了不同脚本。'
        + '这张表能定方向，不能当作 TP=1 / TP=2 的性能结论。'],
    ].filter(Boolean).forEach((item) => {
      const n = el('div', 'tc-verdict');
      n.appendChild(el('strong', null, item[0]));
      n.appendChild(el('span', null, item[1]));
      list.appendChild(n);
    });
    vsec.appendChild(list);
    stage.appendChild(vsec);
  }

  function viewE2EQwen3(stage) {
    const sec = el('section');
    sec.appendChild(sectionHead('采集', '一次选定，E2E / L2 / L1 都跟着走',
      el('span', 'tc-readout', QW().variants.length + ' 份采集 · 2 拓扑 × 2 阶段')));
    sec.appendChild(qwVariantBar());
    stage.appendChild(sec);

    if (S.e2ePanel === 'ops') { qwOpsPanel(stage); return; }
    if (S.e2ePanel === 'api') { qwApiPanel(stage); return; }
    if (S.e2ePanel === 'topo') { qwTopoPanel(stage); return; }
    qwStepPanel(stage);
    qwCaptureNote(stage);
  }

  /* ------------------------------------------------------- L2: 40-layer view
   * The graph is one embed + 40 identical transformer layers + a tail. Every
   * layer row restarts at its own start, on one shared millisecond scale, so
   * the question "is this steady state, and which layer is not" is a look
   * rather than a calculation. */
  function qwLayerPanel(stage) {
    const L = qwLayers();
    const v = qwVariant();
    if (!L) {
      const sec = el('section');
      sec.appendChild(sectionHead('40 层', '本采集无法还原层边界'));
      sec.appendChild(el('p', 'tc-fineprint', '提交序里没有出现 40 次等间距的循环控制间隔。'));
      stage.appendChild(sec);
      return;
    }
    const st = L.steady;
    const sec = el('section');
    sec.appendChild(sectionHead('40 层一致性', v.label + ' · 每层 ' + L.perLayer + ' 个任务',
      /* not the dataset's 已校验 / 未校验 — this is about whether the submit
       * order gave exactly one way to cut the loop */
      el('span', 'tc-readout' + (L.verified ? '' : ' is-crit'),
        L.verified ? '层边界唯一' : '层边界有歧义')));
    sec.appendChild(tiles([
      { k: '层数', v: String(L.layerCount) },
      { k: '每层任务', v: String(L.perLayer) },
      { k: '层窗口中位数', v: msOrUs(st.medianSpanUs) },
      { k: '最快 / 最慢', v: msOrUs(st.minSpanUs) + ' / ' + msOrUs(st.maxSpanUs) },
      { k: '离散度', v: pct(st.spreadPct), tone: st.spreadPct > 30 ? 'bad' : st.spreadPct > 12 ? 'warn' : 'good' },
      { k: '首层偏差', v: pct(st.firstDeltaPct), tone: Math.abs(st.firstDeltaPct) > 20 ? 'warn' : null },
    ]));
    const note = el('p', 'tc-fineprint');
    note.textContent = L.method
      + ' 层窗口用的是层主体：每层有 ' + (st.earlyLayers ? '1–' + st.earlyTasksMax : '0')
      + ' 个任务被提前派发（最多提前 '
      + msOrUs(Math.max.apply(null, L.rows.map((r) => r.earlyLeadUs)))
      + '），若用 min(start) 计窗口会把这些提前量算进层内。'
      + (L.layoutChecked === true ? ' 每层 16 步的 AIC / AIV 次序与采集脚本已校验的布局一致。' : '');
    sec.appendChild(note);

    /* the unwrapped rows */
    const chart = el('div', 'tc-layerchart');
    const canvas = el('canvas');
    chart.appendChild(canvas);
    const maxBody = Math.max.apply(null, L.rows.map((r) => r.bodyUs));
    const drawLayers = () => {
      const w = chart.clientWidth || 720;
      const rowH = 15;
      const padL = 46;
      const padT = 32;
      const h = padT + L.rows.length * rowH + 10;
      const ctx = fitCanvas(canvas, w, h);
      const plotW = w - padL - 12;
      drawTimeRuler(ctx, padL, plotW, 2, 0, maxBody, { dense: true });
      const med = L.steady.medianSpanUs;
      ctx.font = '500 10px ' + cssVar('--font-mono');
      L.rows.forEach((r, i) => {
        const y = padT + i * rowH;
        if (i % 2) {
          ctx.fillStyle = cssVar('--surface-3');
          ctx.globalAlpha = 0.55;
          ctx.fillRect(padL, y, plotW, rowH);
          ctx.globalAlpha = 1;
        }
        ctx.fillStyle = cssVar('--foreground-secondary');
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        ctx.fillText('L' + String(r.layer).padStart(2, '0'), padL - 6, y + rowH / 2);
        const sx = (t) => padL + (t / maxBody) * plotW;
        if (r.blocks) {
          r.blocks.forEach((b) => {
            const x = sx(Math.max(0, b[0]));
            /* leave a hairline between adjacent tasks, otherwise a layer whose
             * tasks tile the window reads as one solid bar */
            const bw = Math.max(1, (b[1] / maxBody) * plotW - 0.8);
            ctx.fillStyle = CMAP.colorForLaneKind(b[2] ? 'aic' : 'aiv');
            ctx.fillRect(x, y + 2, bw, rowH - 4);
          });
        } else {
          /* too many tasks per layer to draw one by one: the row is the window,
           * shaded by how much core time it carried */
          ctx.fillStyle = CMAP.colorForLaneKind('aic');
          ctx.globalAlpha = 0.75;
          ctx.fillRect(sx(0), y + 2, Math.max(1, (r.bodyUs / maxBody) * plotW), rowH - 4);
          ctx.globalAlpha = 1;
        }
        /* the median marker, so an outlier row reads as one */
        ctx.strokeStyle = cssVar('--border-strong');
        ctx.globalAlpha = 0.6;
        ctx.beginPath();
        ctx.moveTo(Math.round(sx(med)) + 0.5, y);
        ctx.lineTo(Math.round(sx(med)) + 0.5, y + rowH);
        ctx.stroke();
        ctx.globalAlpha = 1;
      });
    };
    stage.appendChild(sec);
    sec.appendChild(chart);
    requestAnimationFrame(drawLayers);
    if (stage.__ro) stage.__ro.disconnect();
    stage.__ro = new ResizeObserver(drawLayers);
    stage.__ro.observe(chart);
    sec.appendChild(el('p', 'tc-fineprint',
      '每行从该层主体起点重新计时，共用同一刻度；竖线是层窗口中位数。'
      + (L.rows[0].blocks ? '色块是该层的每个任务，红=AIC / 蓝=AIV，宽度是实测时长。'
        : '每层 ' + L.perLayer + ' 个任务太密，行内只画层窗口本身。')));

    /* per-step statistics: which step inside a layer is the expensive one */
    if (L.steps) {
      const ssec = el('section');
      ssec.appendChild(sectionHead('层内步骤', '40 层同一步骤的分布',
        el('span', 'tc-readout', v.hasNames ? (v.namesFrom || '') : '步名不可得')));
      const maxSum = Math.max.apply(null, L.steps.map((s) => s.sumUs));
      ssec.appendChild(table([
        { label: '#', key: 'step', num: true },
        { label: '步骤', cell: (r) => esc(r.name), mono: true },
        { label: '引擎', cell: (r) => (r.kind || '').toUpperCase() },
        { label: '均值', num: true, cell: (r) => num(r.meanUs, 1) },
        { label: 'min / max', num: true, cell: (r) => num(r.minUs, 1) + ' / ' + num(r.maxUs, 1) },
        { label: '离散', num: true, cell: (r) => (r.spreadX == null ? '—'
          : '<span class="' + (r.spreadX > 2 ? 'warn' : '') + '">' + num(r.spreadX, 2) + 'x</span>') },
        { label: '40 层合计', num: true, cell: (r) => num(r.sumUs, 0) },
        { label: '', cell: (r) => bar(r.sumUs / maxSum, r.sumUs === maxSum ? 'warn' : 'neutral') },
        { label: '占层内核时', num: true, cell: (r) => pct(r.share) },
      ], L.steps, { tall: true }));
      stage.appendChild(ssec);
    }

    /* outliers + what sits outside the loop */
    const osec = el('section');
    osec.appendChild(sectionHead('离群层与环外任务', '偏离中位数最远的 6 层'));
    osec.appendChild(table([
      { label: '层', cell: (r) => 'L' + String(r.layer).padStart(2, '0'), mono: true },
      { label: '层窗口', num: true, cell: (r) => msOrUs(r.spanUs) },
      { label: '相对中位数', num: true, cell: (r) => '<span class="'
        + (Math.abs(r.deltaPct) > 20 ? 'warn' : '') + '">' + num(r.deltaPct, 1) + '%</span>' },
    ], L.outliers, {}));
    const out = el('dl', 'tc-kv');
    [['环外任务', L.headTasks + ' 前 + ' + L.tailTasks + ' 后'],
      ['环外任务名', L.outsideNames.join('、')],
    ].forEach((kv) => {
      out.appendChild(el('dt', null, kv[0]));
      out.appendChild(el('dd', null, kv[1]));
    });
    osec.appendChild(out);
    stage.appendChild(osec);
  }

  /* ------------------------------------------------- L2: head overhead split
   * receive_to_start_cycles in l2_swimlane_records splits the per-task head
   * overhead into the AICPU -> AICore NoC propagation (hardware-bound) and the
   * AICore-local dcci + ack pair (software-tunable). Without that column the
   * two cannot be told apart, and the screen says so. */
  function qwHeadPanel(stage) {
    const l2 = qwL2();
    const v = qwVariant();
    const h = l2.head;
    const sec = el('section');
    sec.appendChild(sectionHead('头开销拆解', v.label + ' · 每个块一次 dispatch → 执行 → finish',
      el('span', 'tc-readout', h ? h.source : '本采集无记录')));
    if (!h) { stage.appendChild(sec); return; }
    const parts = [
      h.noc ? ['dispatch → receive', 'NoC 传播 · 硬件', h.noc] : null,
      h.dcci ? ['receive → start', 'dcci + ack · 可调', h.dcci] : null,
      h.queue ? ['dispatch → start', '头开销合计 · 不可再分', h.queue] : null,
      h.kernel ? ['start → end', '内核本身', h.kernel] : null,
      h.tail ? ['end → finish', '完成回报', h.tail] : null,
    ].filter(Boolean);
    sec.appendChild(tiles(parts.map((p) => ({
      k: p[0], v: num(p[2].meanUs, 2), u: 'us · 均值',
      tone: p[0] === 'start → end' ? null : (p[2].shareOfKernel > 25 ? 'bad' : null),
    }))));
    sec.appendChild(table([
      { label: '区间', cell: (r) => esc(r[0]), mono: true },
      { label: '含义', cell: (r) => esc(r[1]) },
      { label: '块数', num: true, cell: (r) => r[2].count },
      { label: '均值', num: true, cell: (r) => num(r[2].meanUs, 2) },
      { label: 'p50', num: true, cell: (r) => num(r[2].p50Us, 2) },
      { label: 'p90', num: true, cell: (r) => num(r[2].p90Us, 2) },
      { label: 'max', num: true, cell: (r) => num(r[2].maxUs, 1) },
      { label: '合计', num: true, cell: (r) => num(r[2].sumUs, 0) },
      { label: '相对内核核时', num: true, cell: (r) => (r[0] === 'start → end' ? '—'
        : '<span class="' + (r[2].shareOfKernel > 25 ? 'warn' : '') + '">'
          + pct(r[2].shareOfKernel) + '</span>') },
    ], parts, {}));
    sec.appendChild(el('p', 'tc-fineprint',
      '分母是这次采集里所有块的内核核时合计，不是墙钟：一个块的头开销只能和内核时长比，'
      + '不能和整段墙钟比。'
      + (h.hasReceiveColumn
        ? ' receive_to_start_cycles 存在，所以 NoC 传播与 dcci + ack 能分开。'
        : ' 本采集没有 receive 时间戳，头开销只能给一个合计值。')
      + ' 联结到 ' + h.joinable + ' / ' + h.blockRows + ' 块。'));
    stage.appendChild(sec);

    /* task duration distribution: the head overhead only matters relative to
     * how small the tasks are */
    const d = l2.dur;
    const dsec = el('section');
    dsec.appendChild(sectionHead('任务时长分布', '头开销值不值得管，取决于任务有多小',
      el('span', 'tc-readout', l2.taskCount + ' 任务 · p50 ' + num(d.p50Us, 1)
        + ' us · max ' + msOrUs(d.maxUs))));
    const maxN = Math.max.apply(null, d.hist.map((b) => b.n));
    dsec.appendChild(table([
      { label: '时长', cell: (r) => (r.hi == null ? '≥ ' + r.lo + ' us' : r.lo + '–' + r.hi + ' us'), mono: true },
      { label: '任务', key: 'n', num: true },
      { label: '', cell: (r) => bar(r.n / maxN, 'neutral') },
      { label: '占比', num: true, cell: (r) => pct(r.share) },
    ], d.hist, {}));
    stage.appendChild(dsec);

    /* the longest tasks, with their own head overhead */
    const tsec = el('section');
    tsec.appendChild(sectionHead('最长的 30 个任务', v.hasNames ? '带步名' : '本采集无任务名，只能给 token / reg id'));
    tsec.appendChild(table([
      { label: v.hasNames ? '步骤' : 'token', cell: (r) => esc(r.name || r.tok), mono: true },
      { label: '层', cell: (r) => (r.layer == null ? '—' : 'L' + String(r.layer).padStart(2, '0')) },
      { label: '引擎', cell: (r) => (r.kind || '').toUpperCase() },
      { label: '核 / 块', cell: (r) => r.cores + ' / ' + r.blocks },
      { label: '起点', num: true, cell: (r) => msOrUs(r.startUs) },
      { label: '时长', num: true, cell: (r) => msOrUs(r.durUs) },
      { label: 'NoC', num: true, cell: (r) => (r.nocUs == null ? '—' : num(r.nocUs, 2)) },
      { label: 'dcci', num: true, cell: (r) => (r.dcciUs == null ? '—' : num(r.dcciUs, 2)) },
      { label: '完成回报', num: true, cell: (r) => (r.tailUs == null ? '—' : num(r.tailUs, 2)) },
    ], l2.top, { tall: true }));
    stage.appendChild(tsec);
  }

  /* ------------------------------------- L2: occupancy for the thin captures
   * No merged swimlane means no Worker / Scheduler View, so there is no task
   * object to hand the swimlane pattern. What the records do give is every
   * block's core and interval — enough for an occupancy picture, which is
   * exactly the question these captures can answer. */
  function qwOccPanel(stage) {
    const l2 = qwL2();
    const v = qwVariant();
    const sec = el('section');
    sec.appendChild(sectionHead('核占用', v.label + ' · ' + l2.blockRows + ' 块 · 用到 '
      + l2.coresUsed + '/' + l2.coreTotal + ' 核',
      el('span', 'tc-readout', '无 merged swimlane，这里是块级占用而不是任务泳道')));
    sec.appendChild(tiles([
      { k: 'span', v: msOrUs(l2.spanUs) },
      { k: '核时', v: msOrUs(l2.busyUs) },
      { k: '平均忙核', v: num(l2.occ.busyCores, 2), u: '/ ' + l2.coreTotal,
        tone: l2.occ.busyCores < 2 ? 'bad' : null },
      { k: 'AIC 占用', v: pct(l2.occ.aicUtil), u: l2.occ.aicCores + ' 核',
        tone: l2.occ.aicUtil < 20 ? 'bad' : null },
      { k: 'AIV 占用', v: pct(l2.occ.aivUtil), u: l2.occ.aivCores + ' 核',
        tone: l2.occ.aivUtil < 20 ? 'bad' : null },
      { k: '仅用到的核', v: pct(l2.occ.aicUsedUtil) + ' / ' + pct(l2.occ.aivUsedUtil),
        u: l2.occ.aicUsed + ' AIC / ' + l2.occ.aivUsed + ' AIV' },
    ]));
    if (l2.blocks) {
      const chart = el('div', 'tc-occchart');
      const canvas = el('canvas');
      chart.appendChild(canvas);
      const lanesUsed = l2.lanes.filter((l) => l.n > 0);
      const laneRow = {};
      lanesUsed.forEach((l, i) => { laneRow[l.core] = i; });
      const drawOcc = () => {
        const w = chart.clientWidth || 720;
        const rowH = lanesUsed.length > 24 ? 9 : 16;
        const padL = 54;
        const padT = 30;
        const h = padT + lanesUsed.length * rowH + 8;
        const ctx = fitCanvas(canvas, w, h);
        const plotW = w - padL - 12;
        drawTimeRuler(ctx, padL, plotW, 2, 0, l2.spanUs, { dense: true });
        ctx.font = '500 9px ' + cssVar('--font-mono');
        lanesUsed.forEach((l, i) => {
          const y = padT + i * rowH;
          ctx.fillStyle = cssVar('--surface-2');
          ctx.globalAlpha = 0.45;
          ctx.fillRect(padL, y, plotW, rowH - 1);
          ctx.globalAlpha = 1;
          ctx.fillStyle = cssVar('--foreground-muted');
          ctx.textAlign = 'right';
          ctx.textBaseline = 'middle';
          ctx.fillText(l.kind.toUpperCase() + '_' + l.core, padL - 5, y + rowH / 2);
        });
        l2.blocks.rows.forEach((b) => {
          const i = laneRow[b[0]];
          if (i == null) return;
          const y = padT + i * rowH;
          const x = padL + (b[1] / l2.spanUs) * plotW;
          const bw = Math.max(0.6, (b[2] / l2.spanUs) * plotW);
          ctx.fillStyle = CMAP.colorForLaneKind(l2.lanes[b[0]].kind);
          ctx.fillRect(x, y + 1, bw, rowH - 3);
        });
      };
      sec.appendChild(chart);
      requestAnimationFrame(drawOcc);
      if (stage.__ro) stage.__ro.disconnect();
      stage.__ro = new ResizeObserver(drawOcc);
      stage.__ro.observe(chart);
      sec.appendChild(el('p', 'tc-fineprint',
        '一行一个真实物理核，只画有任务落上去的核；红=AIC / 蓝=AIV。'
        + '没有任务名与依赖，所以这里不给关键路径与依赖连线。'));
    }
    stage.appendChild(sec);

    /* lane table + the AICPU side */
    const lsec = el('section');
    lsec.appendChild(sectionHead('核清单', '按核时排序'));
    const used = l2.lanes.filter((l) => l.n > 0).slice().sort((a, b) => b.busyUs - a.busyUs);
    const maxBusy = used.length ? used[0].busyUs : 1;
    lsec.appendChild(table([
      { label: '核', cell: (r) => r.kind.toUpperCase() + '_' + r.core, mono: true },
      { label: '块', key: 'n', num: true },
      { label: '核时', num: true, cell: (r) => num(r.busyUs, 1) },
      { label: '', cell: (r) => bar(r.busyUs / maxBusy, 'neutral') },
      { label: '占用', num: true, cell: (r) => pct(r.util) },
    ], used, { tall: true }));
    stage.appendChild(lsec);

    const asec = el('section');
    asec.appendChild(sectionHead('AICPU 侧', l2.sched.lanes + ' 个调度器 lane · '
      + l2.orch.submits + ' 次提交',
      el('span', 'tc-readout', l2.sched.source)));
    asec.appendChild(tiles([
      { k: '调度器忙时', v: msOrUs(l2.sched.busyUs), u: l2.sched.lanes + ' lane 合计' },
      { k: '相对墙钟', v: pct(l2.sched.util), u: '可 > 100%（多 lane）' },
      { k: '提交次数', v: String(l2.orch.submits) },
      { k: '单次提交', v: num(l2.orch.meanUs, 2), u: 'us · 均值',
        tone: l2.orch.meanUs > 20 ? 'bad' : null },
      { k: '提交占墙钟', v: pct(l2.orch.util) },
    ]));
    const maxPh = Math.max.apply(null, l2.sched.phases.map((p) => p.busyUs)) || 1;
    asec.appendChild(table([
      { label: '阶段', cell: (r) => esc(r.kind), mono: true },
      { label: '次数', key: 'count', num: true },
      { label: '忙时', num: true, cell: (r) => num(r.busyUs, 1) },
      { label: '', cell: (r) => bar(r.busyUs / maxPh, 'neutral') },
      { label: '占调度器', num: true, cell: (r) => pct(r.share) },
    ], l2.sched.phases, {}));
    stage.appendChild(asec);
  }

  /* --------------------------------------------------------- L1: measured PMU
   * The trace can only say how long a block ran. kernel_details.csv says what
   * the pipes were doing while it ran — for the fused kernel, which is the
   * whole 40-layer graph. That is the one place in this case where a pipe
   * claim is measured rather than derived. */
  function qwPmuPanel(stage) {
    const t = qwTorch();
    const v = qwVariant();
    const cap = qwCapture();
    const mine = t.fused.filter((f) => f.stage === v.stage);
    const rows = mine.length ? mine : t.fused;
    const avg = (get) => (rows.length ? sum2(rows.map(get)) / rows.length : null);
    const sec = el('section');
    sec.appendChild(sectionHead('PMU 实测流水占比', 'aicore_kernel_0 · ' + cap.label
      + ' · ' + v.stage + ' ' + rows.length + ' 次调用',
      el('span', 'tc-readout', 'kernel_details.csv · PMU 开启')));
    const pipes = [
      ['AIC', 'mac', avg((f) => f.aic.mac), 'Cube 矩阵乘'],
      ['AIC', 'mte1', avg((f) => f.aic.mte1), 'L1 → L0 搬运'],
      ['AIC', 'mte2', avg((f) => f.aic.mte2), 'GM → L1 搬运'],
      ['AIC', 'fixpipe', avg((f) => f.aic.fixpipe), '定点后处理'],
      ['AIC', 'scalar', avg((f) => f.aic.scalar), '标量 / 控制'],
      ['AIV', 'vec', avg((f) => f.aiv.vec), '向量计算'],
      ['AIV', 'mte2', avg((f) => f.aiv.mte2), 'GM → UB 搬运'],
      ['AIV', 'mte3', avg((f) => f.aiv.mte3), 'UB → GM 搬运'],
      ['AIV', 'scalar', avg((f) => f.aiv.scalar), '标量 / 控制'],
    ];
    const top = pipes.slice().sort((a, b) => (b[2] || 0) - (a[2] || 0))[0];
    sec.appendChild(tiles([
      { k: '单次时长', v: msOrUs(avg((f) => f.durUs)) },
      { k: '最忙流水', v: top[0] + ' ' + top[1], u: num(top[2], 3) + ' 占比', tone: 'warn' },
      { k: 'AIC mac', v: num(avg((f) => f.aic.mac), 3),
        tone: avg((f) => f.aic.mac) < 0.1 ? 'bad' : null },
      { k: 'AIV vec', v: num(avg((f) => f.aiv.vec), 3),
        tone: avg((f) => f.aiv.vec) < 0.05 ? 'bad' : null },
      { k: 'cube 利用率', v: num(avg((f) => f.cubeUtil), 1) },
      { k: 'icache miss', v: num(avg((f) => f.aic.icacheMiss), 3) + ' / '
        + num(avg((f) => f.aiv.icacheMiss), 3), u: 'AIC / AIV' },
    ]));
    const maxPipe = Math.max.apply(null, pipes.map((p) => p[2] || 0)) || 1;
    sec.appendChild(table([
      { label: '引擎', cell: (r) => r[0] },
      { label: '流水线', cell: (r) => esc(r[1]), mono: true },
      { label: '做什么', cell: (r) => esc(r[3]) },
      { label: '占比', num: true, cell: (r) => pmuCell(r[2]) },
      { label: '', cell: (r) => bar((r[2] || 0) / maxPipe, r[2] === top[2] ? 'warn' : 'neutral') },
    ], pipes, {}));
    sec.appendChild(el('p', 'tc-fineprint',
      '每个占比是该流水线相对 aicore_time / aiv_time 的忙占比。它们会重叠，'
      + '加起来可以超过 1，所以这张表只能一行一行读，不能当成时间切分。'
      + 'PMU 开着采的这一轮不能与 PMU 关闭的基线比较墙钟。'));
    stage.appendChild(sec);

    const rsec = el('section');
    rsec.appendChild(sectionHead('逐次调用', '整图一次调用 = 一行', el('span', 'tc-readout',
      '1 次 prefill + ' + t.iters.decode + ' 次 decode')));
    rsec.appendChild(table([
      { label: 'step', key: 'step', num: true },
      { label: 'stage', cell: (r) => esc(r.stage) },
      { label: '时长', num: true, cell: (r) => msOrUs(r.durUs) },
      { label: 'aicore_time', num: true, cell: (r) => num(r.aicTimeUs, 0) },
      { label: 'aiv_time', num: true, cell: (r) => num(r.aivTimeUs, 0) },
      { label: 'mac', num: true, cell: (r) => pmuCell(r.aic.mac) },
      { label: 'mte1', num: true, cell: (r) => pmuCell(r.aic.mte1) },
      { label: 'mte2', num: true, cell: (r) => pmuCell(r.aic.mte2) },
      { label: 'AIC scalar', num: true, cell: (r) => pmuCell(r.aic.scalar) },
      { label: 'vec', num: true, cell: (r) => pmuCell(r.aiv.vec) },
      { label: 'AIV scalar', num: true, cell: (r) => pmuCell(r.aiv.scalar) },
      { label: 'cube', num: true, cell: (r) => num(r.cubeUtil, 1) },
    ], t.fused.map((f) => Object.assign({ __selected: f.stage === v.stage }, f)), {}));
    rsec.appendChild(el('p', 'tc-fineprint',
      'aicore_time / aiv_time 是按核累加的，会超过单次墙钟时长；它们是占比的分母，不是时长。'));
    stage.appendChild(rsec);

    const verdict = el('div', 'tc-verdicts');
    const macAvg = avg((f) => f.aic.mac);
    const mte2Avg = avg((f) => f.aic.mte2);
    const vecAvg = avg((f) => f.aiv.vec);
    const scAvg = avg((f) => f.aiv.scalar);
    const items = [];
    if (mte2Avg > macAvg * 2) {
      items.push(['AIC 受搬运限制，不受算力限制',
        'mte2（GM → L1）占比 ' + num(mte2Avg, 3) + '，mac 只有 ' + num(macAvg, 3)
        + '。Cube 大部分时间在等权重进来，提高 tile 的末维与流水深度比动算法更直接。']);
    }
    if (scAvg > 0.5 && vecAvg < 0.05) {
      items.push(['AIV 在做标量，不在做向量',
        'AIV scalar 占比 ' + num(scAvg, 3) + '，vec 只有 ' + num(vecAvg, 3)
        + '。' + l2Cores() + ' 个 AIV 核是忙的，但忙在控制流与地址计算上。']);
    }
    if (avg((f) => f.aic.scalar) > 0.5) {
      items.push(['AIC 也在标量上',
        'AIC scalar 占比 ' + num(avg((f) => f.aic.scalar), 3)
        + '，而 mac 只有 ' + num(macAvg, 3) + '。这一版更像在等同步而不是在算。']);
    }
    if (items.length) {
      const vs = el('section');
      vs.appendChild(sectionHead('这组计数器说了什么', '只写计数器直接支持的部分'));
      items.forEach((item) => {
        const n = el('div', 'tc-verdict');
        n.appendChild(el('strong', null, item[0]));
        n.appendChild(el('span', null, item[1]));
        verdict.appendChild(n);
      });
      vs.appendChild(verdict);
      stage.appendChild(vs);
    }
  }
  const sum2 = (a) => a.reduce((x, y) => x + (y || 0), 0);
  const l2Cores = () => { const l = qwL2(); return l ? l.occ.aivCores : 40; };

  /* ========================================================= L2 view */
  function laneRows() {
    const all = R().swimlane.lanes;
    if (S.laneFilter === 'aic') return all.filter((l) => l.kind === 'aic');
    if (S.laneFilter === 'aiv') return all.filter((l) => l.kind === 'aiv');
    return all;
  }

  function viewL2(stage) {
    if (QW()) {
      if (S.l2Panel === 'layers') { qwLayerPanel(stage); return; }
      if (S.l2Panel === 'head') { qwHeadPanel(stage); return; }
      /* The task-level swimlane belongs to the one capture that has task names
       * and a dependency graph. The others get the occupancy picture their
       * records can actually support. */
      if (!onPrimaryVariant()) { qwOccPanel(stage); qwGapNote(stage); return; }
    }
    const rank = R();
    const crit = rank.critical;
    const critSet = {};
    crit.tags.forEach((t) => { critSet[t] = 1; });
    /* whichever path the "只看" filter is pointed at */
    const pathSet = {};
    (S.pathOnly === 'cpm' ? crit.tags : rank.cpath.segments.map((sg) => sg.tag))
      .forEach((t) => { pathSet[t] = 1; });
    const subj = subjectTaskSet();      /* tag -> 1-based marker number */
    const subjLane = subjectLaneSet();
    const hasSubjects = Object.keys(subj).length > 0;
    const dim = S.focusEvidence && (hasSubjects || Object.keys(subjLane).length > 0);
    const cp = rank.cpath;
    /* The annotated view is a visual projection of the L2 queue.  C2/C3 are
     * not generic gap scores: each names a zero-slack task and the runnable
     * work that shares its core pool. */
    const c2 = findingById.C2;
    const c3 = findingById.C3;
    const c2Color = CMAP.colorForTask({ colorKey: 'finding-C2', label: 'C2' }, 'semantic');
    const c3Color = CMAP.colorForTask({ colorKey: 'finding-C3', label: 'C3' }, 'semantic');
    const c2TaskSet = {};
    const c3TaskSet = {};
    if (c2 && c2.subjects) (c2.subjects.tasks || []).forEach((tag) => { c2TaskSet[tag] = 1; });
    if (c3 && c3.subjects) (c3.subjects.tasks || []).forEach((tag) => { c3TaskSet[tag] = 1; });
    const c2Contention = (c2 && c2.contention) || null;
    const c3Contention = (c3 && c3.contention) || null;
    const c2FocusTask = c2Contention && tasksOf[S.rank][c2Contention.focus];
    const c2RivalTasks = c2Contention
      ? c2Contention.rivals.map((tag) => tasksOf[S.rank][tag]).filter(Boolean) : [];
    const c2RivalSet = {};
    c2RivalTasks.forEach((task) => { c2RivalSet[task.tag] = 1; });
    const c2FocusTag = c2Contention && c2Contention.focus;
    const c3FocusTag = c3Contention && c3Contention.focus;
    const c3FocusTask = c3FocusTag && tasksOf[S.rank][c3FocusTag];
    const contentionLaneSet = {};
    const c2LaneSet = {};
    (c2Contention && c2Contention.lanes || []).forEach((name) => { c2LaneSet[name] = 1; });
    (c2Contention && c2Contention.lanes || []).concat(c3Contention && c3Contention.lanes || [])
      .forEach((name) => { contentionLaneSet[name] = 1; });
    const signalTaskSet = { [S.task]: 1 };
    Object.keys(c2TaskSet).forEach((tag) => { signalTaskSet[tag] = 1; });
    Object.keys(c3TaskSet).forEach((tag) => { signalTaskSet[tag] = 1; });
    const focusedFinding = activeFinding();
    const focusTaskRoles = focusedFinding && focusedFinding.taskRoles || {};
    const primaryTaskSet = {};
    const secondaryTaskSet = {};
    (focusTaskRoles.primary || []).forEach((tag) => { primaryTaskSet[tag] = 1; });
    (focusTaskRoles.secondary || []).forEach((tag) => { secondaryTaskSet[tag] = 1; });
    if (!Object.keys(primaryTaskSet).length && focusedFinding) {
      const primary = focusedFinding.focus && focusedFinding.focus.task
        || (focusedFinding.contention && focusedFinding.contention.focus);
      if (primary) primaryTaskSet[primary] = 1;
      (focusedFinding.subjects && focusedFinding.subjects.tasks || []).forEach((tag) => {
        if (tag !== primary) secondaryTaskSet[tag] = 1;
      });
    }
    const hasFindingTaskFocus = Object.keys(primaryTaskSet).length > 0;
    Object.keys(subj).forEach((tag) => { signalTaskSet[tag] = 1; });
    /* Path focus gives the selected path node a clear first visual tier. The
     * rest of the execution main path keeps its task colour at reduced opacity; all
     * non-path work stays as a faint scheduling context. */
    const pathFocusSet = {};
    if (S.pathFocus) {
      cp.segments.forEach((seg) => { pathFocusSet[seg.tag] = 1; });
    }

    /* ---- where each task's blocks actually sit -----------------------
     * Built once per view: the ribbon needs it to jump into the swimlane,
     * and the dependency edges need it to anchor on a real block rather
     * than on the task's aggregate [start, end]. A task spread over 72
     * cores has 72 candidate anchors; only two of them mean anything:
     *   - as a producer, the block that finishes LAST (that is what gates)
     *   - as a consumer, the block that starts FIRST (that is what waited)
     * Anchoring anywhere else would draw an edge that tells no truth about
     * the hand-off it is supposed to depict.                            */
    const lanesByTask = {};
    rank.swimlane.blocks.forEach((blocks, li) => {
      const name = rank.swimlane.laneNames[li];
      blocks.forEach((b) => {
        const tg = rank.tasks[b[2]].tag;
        (lanesByTask[tg] || (lanesByTask[tg] = []))
          .push({ li: li, name: name, start: b[0], end: b[0] + b[1] });
      });
    });
    const gateOut = (tag) => (lanesByTask[tag] || [])
      .reduce((a, b) => (a && a.end >= b.end ? a : b), null);
    const gateIn = (tag) => (lanesByTask[tag] || [])
      .reduce((a, b) => (a && a.start <= b.start ? a : b), null);

    const contentionWindows = [];
    [[c2Contention, 'C2', c2Color], [c3Contention, 'C3', c3Color]].forEach(([item, id, color]) => {
      if (!item) return;
      (item.windows || []).forEach((window) => contentionWindows.push({
        id: id, color: color, engine: item.engine, lanes: item.lanes || [], t0: window.t0, t1: window.t1,
      }));
    });

    /* --- execution attribution path ----------------------------------
     * One primary path is shown here: the backward attribution chain that
     * tiles this rank's observed span using dependency predecessors and
     * same-core execution order. It is a runtime explanation candidate, not
     * a proof of ready-queue causality; task-level dispatch evidence is absent. */
    const ribSec = el('section');
    /* live readouts, filled by the draw pass: keeping them in the section head
     * rather than painting them on the canvas avoids fighting the time ruler
     * for the same pixels, and they stay selectable text */
    const pathReadout = el('span', 'tc-readout');
    const ribRight = el('span', 'tc-readout-group');
    ribRight.appendChild(el('span', 'tc-readout', 'rank span ' + us(cp.makespan)
      + ' · Task 覆盖 ' + us(cp.computeTotal) + ' · 路径间隙 ' + us(cp.stallTotal)));
    ribRight.appendChild(pathReadout);
    ribSec.appendChild(sectionHead('执行主路径 · ' + S.rank + ' · ' + cp.segments.length + ' 个 Task',
      '从最晚结束 Task 反向沿依赖 / 同核前序归因 · 点击节点定位泳道 · 同核等待缺 ready / dispatch 证据，因果待证',
      ribRight));
    const ribHost = el('div', 'tc-canvas-strip');
    const ribCanvas = el('canvas');
    ribHost.appendChild(ribCanvas);
    ribSec.appendChild(ribHost);
    stage.appendChild(ribSec);

    /* --- worker swimlane --- */
    const laneSec = el('section', 'tc-stage-fill');
    const legend = el('div', 'tc-legend');
    const annotated = S.laneFilter === 'summary';
    const c2TraceFocus = D.case.id === 'decode_csa' && c2FocusTask
      && (S.finding === 'C2' || S.task === c2FocusTask.tag);
    legend.appendChild(el('span', 'tc-readout', annotated
      ? '任务颜色表示 kernel 身份；C2 / C3 的线框与符号表示调度角色'
      : '每行一核 · 保留完整逐核标签 · 悬停查看详情'));
    if (annotated && c2TraceFocus) {
      const addC2Legend = (mark, label, style, title) => {
        const item = el('span', 'tc-readout');
        const glyph = el('i');
        Object.keys(style).forEach((key) => { glyph.style[key] = style[key]; });
        item.title = title || (mark === '?'
          ? '缺 task 级 ready / enqueue / dispatch 与优先级记录；全局 ready queue 不能证明关键任务当时已具备派发条件。'
          : mark === '→'
            ? '只提高 ' + c2FocusTask.callable + ' 的调度优先级后重跑；比较 rank device span 与 '
              + c2RivalTasks.map((task) => task.callable).join(' / ')
              + ' slack，必须 span 下降且两项 slack 均不小于 0。'
            : '');
        item.appendChild(glyph);
        item.appendChild(document.createTextNode(mark + ' ' + label));
        legend.appendChild(item);
      };
      addC2Legend('◆', c2FocusTask.callable + ' · slack 0', { background: c2Color, outline: '1px solid #fff' }, '关键链任务：零 slack；泳道块使用双层描边和左侧角色色条。');
      c2RivalTasks.forEach((task, i) => {
        const hatch = i === 0
          ? 'repeating-linear-gradient(135deg, transparent 0 2px, ' + c2Color + ' 2px 3px)'
          : 'repeating-linear-gradient(45deg, transparent 0 2px, ' + c2Color + ' 2px 3px), repeating-linear-gradient(135deg, transparent 0 4px, ' + c2Color + ' 4px 5px)';
        addC2Legend(i === 0 ? '╱' : '╳', task.callable + ' · slack ' + num(task.slack, 0),
          { background: hatch, outline: '1px solid ' + cssVar('--border-default') },
          '同池竞争任务。保留 kernel 身份色，以不同斜纹标出其执行块；slack ' + num(task.slack, 2) + ' us。');
      });
      addC2Legend('?', 'ready 未证实', { border: '1px solid ' + cssVar('--warning'), background: 'transparent' });
      addC2Legend('→', '只调 ' + c2FocusTask.callable + ' 优先级', { border: '1px dashed ' + cssVar('--foreground-secondary'), background: 'transparent' });
    }
    if (S.pathFocus) legend.appendChild(el('span', 'tc-readout', '主路径聚焦中 · 当前任务高亮 · 路径其余任务半透明'));
    const depsReadout = el('span', 'tc-readout');
    legend.appendChild(depsReadout);
    const laneHost = el('div', 'tc-canvas-host');
    const laneCanvas = el('canvas', 'tc-lanes');
    laneCanvas.tabIndex = 0;
    laneHost.appendChild(laneCanvas);
    laneSec.appendChild(sectionHead(annotated ? '72 核泳道 · 关键表现' : '原始逐核泳道',
      annotated
        ? 'C2：' + (c2FocusTask ? c2FocusTask.callable : '关键链 AIC')
          + (c2TraceFocus && c2RivalTasks.length
            ? ' 的同池竞争块：' + c2RivalTasks.map((task, i) => task.callable + '（' + (i === 0 ? '╱' : '╳') + '）').join('、')
              + ' · 斜纹对应 AIC_0 / AIC_18 上的实际 Task 块'
            : ' 与有 slack 工作竞争')
          + ' · C3：' + (c3FocusTask ? c3FocusTask.callable : '关键链 AIV') + ' 与有 slack 工作竞争；点选 C2 查看同一时间轴上的证据与验证条件'
        : '每行一核 · ' + rank.swimlane.blocks.reduce((a, b) => a + b.length, 0) + ' 块', legend));
    laneSec.appendChild(laneHost);
    stage.appendChild(laneSec);
    if (guidedJourneyActive() && S.focus === 'task'
      && activeFinding().focus && S.task === activeFinding().focus.task) {
      renderGuidedPassProcess(stage);
    }

    /* ---------- rendering ---------- */
    const LBL = 66;
    function drawRibbon() {
      const w = ribHost.clientWidth || 800;
      const evRow = hasSubjects ? 26 : 0;
      const h = 74 + evRow;
      const ctx = fitCanvas(ribCanvas, w, h);
      const plotX = LBL, plotW = Math.max(40, w - LBL - 10);
      drawTimeRuler(ctx, plotX, plotW, 0, S.t0, S.t1, { dense: true });
      ctx.font = '500 11px ' + cssVar('--font-sans');
      ctx.fillStyle = cssVar('--foreground-muted');
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      const sx = (t) => plotX + ((t - S.t0) / (S.t1 - S.t0)) * plotW;

      /* evidence row: where the active finding's subjects sit on this axis */
      if (evRow) {
        ctx.fillStyle = cssVar('--foreground');
        ctx.fillText('证据', 4, 40);
        Object.keys(subj).forEach((tag) => {
          const t = tasksOf[S.rank][tag];
          if (!t) return;
          const x = clamp(sx(t.start), plotX, plotX + plotW);
          const x2 = clamp(sx(t.end), plotX, plotX + plotW);
          if (x2 <= plotX || x >= plotX + plotW) return;
          SW.drawTaskBar(ctx, {
            task: barTask(t, null, 'evidence'),
            x: x, y: 32, width: Math.max(3, x2 - x), height: 16,
            baseColor: taskColor(t),
            isSelected: true,
            isEmphasized: t.tag === S.task,
            fontFamily: cssVar('--font-sans'),
          });
          /* numbered marker matching the evidence chip above */
          const cx = Math.min(plotX + plotW - 7, Math.max(plotX + 7, x + 7));
          ctx.beginPath();
          ctx.arc(cx, 28, 7, 0, Math.PI * 2);
          ctx.fillStyle = cssVar('--background');
          ctx.fill();
          ctx.strokeStyle = cssVar('--foreground');
          ctx.lineWidth = 1;
          ctx.stroke();
          ctx.fillStyle = cssVar('--foreground');
          ctx.font = '700 9px ' + cssVar('--font-mono');
          ctx.textAlign = 'center';
          ctx.fillText(String(subj[tag]), cx, 28);
          ctx.textAlign = 'left';
          ctx.font = '500 11px ' + cssVar('--font-sans');
        });
      }

      ctx.fillStyle = cssVar('--foreground-muted');
      ctx.fillText('主路径', 4, 40 + evRow);
      ctx.fillText('间隙', 4, 62 + evRow);
      let cursor = null;
      cp.segments.forEach((node) => {
        const t = tasksOf[S.rank][node.tag];
        if (!t) return;
        const x = sx(t.start), x2 = sx(t.end);
        if (x2 < plotX || x > plotX + plotW) { cursor = t.end; return; }
        ctx.globalAlpha = dim && !subj[t.tag] ? 0.28 : 1;
        SW.drawTaskBar(ctx, {
          task: barTask(t, null, 'critical'),
          x: Math.max(plotX, x), y: 31 + evRow, width: Math.max(2, Math.min(plotX + plotW, x2) - Math.max(plotX, x)), height: 18,
          baseColor: taskColor(t),
          isSelected: !!subj[t.tag] || t.tag === S.task,
          isEmphasized: true,
          fontFamily: cssVar('--font-sans'),
        });
        ctx.globalAlpha = 1;
        /* gap markers are not task bars: page-local data-viz marks */
        if (cursor !== null && t.start > cursor) {
          const gx = sx(cursor), gx2 = sx(t.start);
          ctx.fillStyle = cssVar('--warning');
          ctx.globalAlpha = 0.5;
          ctx.fillRect(Math.max(plotX, gx), 56 + evRow, Math.max(1, gx2 - gx), 8);
          ctx.globalAlpha = 1;
        } else if (cursor !== null && t.start < cursor) {
          const ox = sx(t.start), ox2 = sx(cursor);
          ctx.fillStyle = cssVar('--success');
          ctx.globalAlpha = 0.4;
          ctx.fillRect(Math.max(plotX, ox), 58 + evRow, Math.max(1, ox2 - ox), 4);
          ctx.globalAlpha = 1;
        }
        cursor = t.end;
      });

      /* the shared time cursor: the same two dashed rules are drawn on the
       * swimlane below, so a selection made in either canvas is visible in
       * the other one without the reader hunting for a highlighted bar */
      drawSelGuide(ctx, sx, plotX, plotW, 22, h - 4);

      /* say in words what the two dashed rules mean, so the link between the
       * two canvases does not rely on the reader spotting a highlight */
      const selTask = tasksOf[S.rank][S.task];
      if (!selTask) { pathReadout.textContent = ''; return; }
      const at = cp.segments.findIndex((sg) => sg.tag === selTask.tag);
      pathReadout.textContent = at >= 0
        ? '已选 ' + (at + 1) + '/' + cp.segments.length + ' · ' + selTask.callable
        : '已选 ' + selTask.callable + ' · 不在当前主路径';
      pathReadout.classList.toggle('is-muted', at < 0);
    }

    /* ---- ribbon -> swimlane ------------------------------------------
     * The ribbon was read-only: the reader could get from a lane block to
     * the path (click a block, the path bar lights up) but not back. Now a
     * path node is a control -- it selects the task, scrolls the swimlane to
     * the lane that block actually ran on, and frames it if it is off-window.
     */
    const ribHit = (event) => {
      const rect = ribCanvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const plotX = LBL, plotW = Math.max(40, (ribHost.clientWidth || 800) - LBL - 10);
      if (x < plotX || x > plotX + plotW) return null;
      const t = S.t0 + ((x - plotX) / plotW) * (S.t1 - S.t0);
      const tol = ((S.t1 - S.t0) / plotW) * 2;
      for (let i = 0; i < cp.segments.length; i++) {
        const task = tasksOf[S.rank][cp.segments[i].tag];
        if (!task) continue;
        if (t >= task.start - tol && t <= task.end + tol) {
          return { task: task, node: cp.segments[i], idx: i };
        }
      }
      return null;
    };
    attachTooltip(ribHost, ribCanvas, (event) => {
      const hit = ribHit(event);
      if (!hit) return null;
      return barTask(hit.task, null,
        '执行主路径 ' + (hit.idx + 1) + '/' + cp.segments.length
          + (hit.node.onCpm ? ' · 同时在依赖关键路径上' : ''));
    });
    ribCanvas.style.cursor = 'pointer';
    ribCanvas.addEventListener('click', (event) => {
      const hit = ribHit(event);
      if (!hit) return;
      S.task = hit.task.tag;
      S.focus = 'task';
      S.pathFocus = true;
      /* scroll the swimlane to the block this node actually ran on */
      const anchor = gateIn(hit.task.tag);
      if (anchor) S.scrollToLane = anchor.name;
      /* if the node sits outside the current window, bring it in rather than
       * selecting something the reader cannot see */
      if (hit.task.end < S.t0 || hit.task.start > S.t1) {
        const pad = Math.max(40, hit.task.span * 0.35);
        setWindow(hit.task.start - pad, hit.task.end + pad);
        renderToolbar();
      }
      renderInspector();
      drawLanes();
      drawRibbon();
    });

    /* Two dashed rules at the selected task's start and end, drawn on both
     * canvases. This is the whole linkage: one time interval, two views. */
    function drawSelGuide(ctx, sx, plotX, plotW, yTop, yBot) {
      const t = tasksOf[S.rank][S.task];
      if (!t) return;
      const onPath = cp.segments.some((sg) => sg.tag === t.tag);
      ctx.save();
      ctx.strokeStyle = cssVar(onPath ? '--warning' : '--foreground-muted');
      ctx.globalAlpha = onPath ? 0.85 : 0.5;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      [t.start, t.end].forEach((v) => {
        const x = sx(v);
        if (x < plotX || x > plotX + plotW) return;
        ctx.beginPath();
        ctx.moveTo(Math.round(x) + 0.5, yTop);
        ctx.lineTo(Math.round(x) + 0.5, yBot);
        ctx.stroke();
      });
      ctx.restore();
    }

    /* The raw per-core view deliberately keeps its original row rhythm: this
     * is the evidence view where adjacent cores need to remain scannable. */
    const ROW_H = 11, ROW_GAP = 3, ENGINE_GAP = 7;
    const SCHED_ROW_H = 9, SCHED_ROW_GAP = 2;
    let laneLayout = [];

    function drawLanes() {
      const lanes = laneRows();
      const w = laneHost.clientWidth || 800;
      const plotX = LBL, plotW = Math.max(40, w - LBL - 10);
      const overlayRows = S.overlay === 'sched' ? rank.scheduler.lanes.length : 0;
      const readyH = S.overlay === 'ready' ? 46 : 0;
      const OCC_H = 26;
      const c2Diagnostic = D.case.id === 'decode_csa' && annotated && c2FocusTask
        && (S.finding === 'C2' || S.task === c2FocusTask.tag);
      const schedH = overlayRows ? overlayRows * (SCHED_ROW_H + SCHED_ROW_GAP) + 8 : 0;
      const bandTop = 30;
      const top = bandTop + OCC_H + schedH + readyH;
      const workerRows = [];
      let workerBottom = top;
      lanes.forEach((lane, i) => {
        workerRows.push(workerBottom);
        workerBottom += ROW_H + ROW_GAP;
        if (i < lanes.length - 1 && lane.kind !== lanes[i + 1].kind) workerBottom += ENGINE_GAP;
      });
      const h = workerBottom + 8;
      const ctx = fitCanvas(laneCanvas, w, Math.max(h, laneHost.clientHeight || h));
      const sx = (t) => plotX + ((t - S.t0) / (S.t1 - S.t0)) * plotW;
      drawTimeRuler(ctx, plotX, plotW, 0, S.t0, S.t1, { dense: true });
      ctx.font = '500 10px ' + cssVar('--font-sans');
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'left';
      laneLayout = [];

      /* ---- occupancy band ----
       * Two stacked strips (AIC, AIV) whose opacity tracks how many cores were
       * busy in that window. The idle stretches are the point: they get a
       * warning-tinted wash so a low-occupancy window is visible without
       * reading 72 lanes of bars. */
      (function drawOccBand() {
        const wins = rank.occWindows;
        if (!wins || !wins.length) return;
        const ww = rank.occWindowUs;
        const bandY = bandTop;
        const strip = (OCC_H - 4) / 2;
        ctx.fillStyle = cssVar('--foreground-muted');
        ctx.font = '500 10px ' + cssVar('--font-sans');
        ctx.fillText('AIC', 4, bandY + strip / 2);
        ctx.fillText('AIV', 4, bandY + strip + 2 + strip / 2);
        wins.forEach((win) => {
          const x = sx(win[0]), x2 = sx(win[0] + ww);
          if (x2 < plotX || x > plotX + plotW) return;
          const xa = Math.max(plotX, x);
          const wpx = Math.max(0.8, Math.min(plotX + plotW, x2) - xa);
          [[win[1], bandY, 'aic'], [win[2], bandY + strip + 2, 'aiv']].forEach((cfg) => {
            ctx.fillStyle = CMAP.colorForLaneKind(cfg[2]);
            ctx.globalAlpha = 0.12 + (clamp(cfg[0], 0, 100) / 100) * 0.85;
            ctx.fillRect(xa, cfg[1], wpx, strip);
            ctx.globalAlpha = 1;
          });
        });
        /* Raw mode retains the low-occupancy wash. In annotated mode, the
         * queue's C2 / C3 overlays below take precedence over generic signals. */
        const idleHighlights = annotated ? [] : (rank.idleRuns || []);
        idleHighlights.forEach((r) => {
          const x = sx(r.t0), x2 = sx(r.t1);
          if (x2 < plotX || x > plotX + plotW) return;
          const xa = Math.max(plotX, x);
          const wpx = Math.min(plotX + plotW, x2) - xa;
          if (wpx < 1) return;
          ctx.fillStyle = cssVar('--warning');
          ctx.globalAlpha = 0.14;
          ctx.fillRect(xa, bandY, wpx, h - bandY - 4);
          ctx.globalAlpha = 0.75;
          ctx.strokeStyle = cssVar('--warning');
          ctx.setLineDash([2, 2]);
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(Math.round(xa) + 0.5, bandY);
          ctx.lineTo(Math.round(xa) + 0.5, h - 4);
          ctx.moveTo(Math.round(xa + wpx) - 0.5, bandY);
          ctx.lineTo(Math.round(xa + wpx) - 0.5, h - 4);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.globalAlpha = 1;
          if (wpx > 52) {
            ctx.fillStyle = cssVar('--warning');
            ctx.font = '500 10px ' + cssVar('--font-sans');
            ctx.textAlign = 'center';
            ctx.fillText(num(r.us, 0) + ' us 空转', xa + wpx / 2, bandY - 6);
            ctx.textAlign = 'left';
          }
        });
        ctx.font = '500 10px ' + cssVar('--font-sans');
      })();

      /* AICPU scheduler lanes */
      if (overlayRows) {
        rank.scheduler.lanes.forEach((name, i) => {
          const y = bandTop + OCC_H + i * (SCHED_ROW_H + SCHED_ROW_GAP);
          ctx.fillStyle = cssVar('--foreground-muted');
          ctx.fillText(name, 4, y + SCHED_ROW_H / 2);
          rank.scheduler.blocks[i].forEach((b) => {
            const x = sx(b[0]), x2 = sx(b[0] + b[1]);
            if (x2 < plotX || x > plotX + plotW) return;
            ctx.fillStyle = CMAP.colorForLaneKind('aicpu');
            ctx.globalAlpha = b[2] === 'complete' ? 0.95 : b[2] === 'dispatch' ? 0.7 : 0.45;
            ctx.fillRect(Math.max(plotX, x), y, Math.max(0.8, Math.min(plotX + plotW, x2) - Math.max(plotX, x)), SCHED_ROW_H);
            ctx.globalAlpha = 1;
          });
        });
      }

      /* ready-but-undispatched strip */
      if (readyH) {
        const y0 = 32 + bandTop - 30 + OCC_H, hh = readyH - 8;
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
      const markers = [];
      lanes.forEach((lane, i) => {
        const y = workerRows[i];
        const li = rank.swimlane.laneNames.indexOf(lane.name);
        laneLayout.push({ y: y, laneIdx: li, name: lane.name });
        const laneIsSubject = !!subjLane[lane.name];
        const laneIsContention = annotated && !!contentionLaneSet[lane.name];
        const laneIsC2 = c2Diagnostic && !!c2LaneSet[lane.name];
        /* A barely-there band restores the row rhythm in a dense trace while
         * leaving task colour and idle washes as the primary signals. */
        ctx.fillStyle = cssVar('--surface-2');
        ctx.globalAlpha = i % 2 ? 0.32 : 0.16;
        ctx.fillRect(plotX, y - 1, plotW, ROW_H + 2);
        ctx.globalAlpha = 1;
        if (laneIsSubject) {
          ctx.fillStyle = cssVar('--warning');
          ctx.globalAlpha = 0.12;
          ctx.fillRect(plotX, y - 2, plotW, ROW_H + 4);
          ctx.globalAlpha = 1;
        }
        const rowWindows = annotated
          ? contentionWindows.filter((window) => window.lanes.indexOf(lane.name) >= 0) : [];
        rowWindows.forEach((window) => {
          const x0 = sx(window.t0), x1 = sx(window.t1);
          if (x1 < plotX || x0 > plotX + plotW) return;
          const left = Math.max(plotX, x0);
          const width = Math.max(1, Math.min(plotX + plotW, x1) - left);
          ctx.fillStyle = window.color;
          ctx.globalAlpha = window.id === 'C2' && c2Diagnostic ? 0.20 : 0.10;
          ctx.fillRect(left, y - 2, width, ROW_H + 4);
          ctx.globalAlpha = window.id === 'C2' && c2Diagnostic ? 0.9 : 0.62;
          ctx.strokeStyle = window.color;
          ctx.lineWidth = window.id === 'C2' && c2Diagnostic ? 1.2 : 0.8;
          ctx.setLineDash(window.id === 'C2' && c2Diagnostic ? [3, 2] : [2, 3]);
          ctx.strokeRect(left + 0.5, y - 1.5, Math.max(0, width - 1), ROW_H + 2);
          ctx.setLineDash([]);
          ctx.globalAlpha = 1;
        });
        ctx.fillStyle = laneIsSubject ? cssVar('--warning') : laneIsC2 ? c2Color : laneIsContention ? cssVar('--accent')
          : lane.util > 60 ? cssVar('--foreground-secondary') : cssVar('--foreground-muted');
        ctx.font = (laneIsSubject || laneIsContention ? '600' : '500') + ' 10px ' + cssVar('--font-sans');
        ctx.fillText(lane.name, 4, y + ROW_H / 2);
        if (laneIsC2) {
          ctx.save();
          ctx.beginPath();
          ctx.arc(57, y + ROW_H / 2, 4.2, 0, Math.PI * 2);
          ctx.strokeStyle = cssVar('--warning');
          ctx.lineWidth = 1;
          ctx.stroke();
          ctx.fillStyle = cssVar('--warning');
          ctx.font = '700 7px ' + cssVar('--font-mono');
          ctx.textAlign = 'center';
          ctx.fillText('?', 57, y + ROW_H / 2 + 2.5);
          ctx.restore();
          ctx.font = '500 10px ' + cssVar('--font-sans');
          ctx.textAlign = 'left';
        }
        rank.swimlane.blocks[li].forEach((b) => {
          const t = rank.tasks[b[2]];
          if (S.critOnly && !pathSet[t.tag]) return;
          const x = sx(b[0]), x2 = sx(b[0] + b[1]);
          if (x2 < plotX || x > plotX + plotW) return;
          const xa = Math.max(plotX, x);
          const wBar = Math.max(0.8, Math.min(plotX + plotW, x2) - xa);
          const isSubj = !!subj[t.tag];
          const isFindingPrimary = !!primaryTaskSet[t.tag];
          const isFindingSecondary = !!secondaryTaskSet[t.tag];
          const isCurrentPathTask = S.pathFocus && t.tag === S.task;
          const isOtherPathTask = S.pathFocus && !isCurrentPathTask && !!pathFocusSet[t.tag];
          const bEnd = b[0] + b[1];
          const inC2Window = laneIsC2 && (c2Contention.windows || []).some((win) => b[0] < win.t1 && bEnd > win.t0);
          const isC2Rival = c2Diagnostic && inC2Window && !!c2RivalSet[t.tag];
          const isC2FocusBlock = c2Diagnostic && inC2Window && t.tag === c2FocusTag;
          const fadedByPath = S.pathFocus && !isCurrentPathTask && !isOtherPathTask
            && !isC2Rival && !isSubj && !isFindingPrimary && !isFindingSecondary;
          const fadedBySignal = annotated && !S.pathFocus && !signalTaskSet[t.tag] && !isSubj
            && !isFindingPrimary && !isFindingSecondary;
          const faded = fadedByPath || fadedBySignal;
          const isContentionFocus = t.tag === c2FocusTag || t.tag === c3FocusTag;
          if (isSubj) markers.push({ x: xa, y: y, n: subj[t.tag] });
          ctx.globalAlpha = hasFindingTaskFocus
            ? (isFindingPrimary ? 1 : isFindingSecondary ? 0.48 : 0.16)
            : faded ? (fadedByPath ? 0.14 : 0.2)
            : isOtherPathTask ? 0.44 : isC2Rival && S.pathFocus ? 0.62
              : (dim && !isSubj && !laneIsSubject ? 0.16 : 1);
          if (wBar < 2.2) {
            /* below task-bar legibility: draw a density tick, not a fake bar */
            ctx.fillStyle = faded && !hasFindingTaskFocus ? cssVar('--surface-4') : taskColor(t);
            ctx.fillRect(xa, y, wBar, ROW_H);
            ctx.globalAlpha = 1;
            return;
          }
          SW.drawTaskBar(ctx, {
            task: barTask(t, b, lane.name),
            x: xa, y: y, width: wBar, height: ROW_H, radius: 1,
            baseColor: faded && !hasFindingTaskFocus ? cssVar('--surface-4') : taskColor(t),
            isSelected: t.tag === S.task || (!hasFindingTaskFocus && isSubj && !S.pathFocus),
            isRelated: (!hasFindingTaskFocus && isFindingSecondary) || isC2Rival || (!S.pathFocus && !isSubj && t.tag !== S.task
              && !!pathSet[t.tag] && !S.critOnly),
            isEmphasized: (!hasFindingTaskFocus && isFindingPrimary)
              || (isSubj && !S.pathFocus) || isContentionFocus || isCurrentPathTask,
            fontFamily: cssVar('--font-sans'),
          });
          ctx.globalAlpha = 1;
          if (isFindingPrimary && !hasFindingTaskFocus && !isC2FocusBlock) {
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1.5;
            ctx.strokeRect(xa + 0.75, y + 0.75, Math.max(0, wBar - 1.5), ROW_H - 1.5);
            ctx.fillStyle = cssVar('--warning');
            ctx.fillRect(xa, y - 1, Math.min(3, wBar), ROW_H + 2);
          } else if (isFindingSecondary && !hasFindingTaskFocus && !isC2Rival) {
            ctx.strokeStyle = cssVar('--foreground-secondary');
            ctx.globalAlpha = 0.8;
            ctx.lineWidth = 1;
            ctx.setLineDash([2, 2]);
            ctx.strokeRect(xa + 0.5, y + 0.5, Math.max(0, wBar - 1), ROW_H - 1);
            ctx.setLineDash([]);
            ctx.globalAlpha = 1;
          }
          if (isC2FocusBlock) {
            /* Focus gets a high-contrast double frame; the kernel identity fill
             * remains untouched so role emphasis cannot be mistaken for color. */
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.strokeRect(xa + 1, y + 1, Math.max(0, wBar - 2), ROW_H - 2);
            ctx.fillStyle = c2Color;
            ctx.fillRect(xa, y - 1, Math.min(3, wBar), ROW_H + 2);
          }
          if (isC2Rival) {
            /* Rival tasks retain their kernel identity colors. Opposite hatch
             * directions separate the two competitors even when labels clip. */
            ctx.save();
            ctx.beginPath();
            ctx.rect(xa, y, wBar, ROW_H);
            ctx.clip();
            ctx.strokeStyle = c2Color;
            ctx.globalAlpha = 0.9;
            ctx.lineWidth = 1;
            const rivalIndex = c2RivalTasks.findIndex((task) => task.tag === t.tag);
            const drawHatch = (reverse, step) => {
              for (let hx = xa - ROW_H; hx < xa + wBar; hx += step) {
                ctx.beginPath();
                ctx.moveTo(hx, reverse ? y : y + ROW_H);
                ctx.lineTo(hx + ROW_H, reverse ? y + ROW_H : y);
                ctx.stroke();
              }
            };
            drawHatch(rivalIndex > 0, 5);
            if (rivalIndex > 0) drawHatch(false, 7);
            ctx.restore();
          }
        });
        /* A wider separator at each engine boundary makes the two core pools
         * legible even when the chart is scrolled. */
        if (i < lanes.length - 1 && lane.kind !== lanes[i + 1].kind) {
          const separatorY = y + ROW_H + (ROW_GAP + ENGINE_GAP) / 2;
          ctx.strokeStyle = cssVar('--border-default');
          ctx.globalAlpha = 0.75;
          ctx.beginPath();
          ctx.moveTo(4, separatorY + 0.5);
          ctx.lineTo(plotX + plotW, separatorY + 0.5);
          ctx.stroke();
          ctx.globalAlpha = 1;
        }
      });

      /* ---- dependency edges -------------------------------------------
       * deps.json carries pred / succ per task, but until now the swimlane
       * never showed them: the reader could see two bars and had no way to
       * know which one fed the other. Edges are drawn for the SELECTED task
       * only (or along the path), never for all 126 of them -- a full graph
       * over 72 lanes is a hairball, not a diagnosis.
       *
       * The edge that matters is producer-last-block -> consumer-first-block.
       * When the consumer's first block starts BEFORE the producer's last
       * block ends, it is drawn in danger + dashed as a trace-order anomaly.
       * This is supporting context only; C2/C3 are the separately annotated
       * per-lane contention marks in the worker rows. */
      if (S.deps !== 'off') {
        const rowOf = {};
        laneLayout.forEach((r) => { rowOf[r.name] = r; });
        const edges = [];
        if (S.deps === 'path') {
          for (let i = 0; i + 1 < cp.segments.length; i++) {
            edges.push({ from: cp.segments[i].tag, to: cp.segments[i + 1].tag, kind: 'path' });
          }
        } else {
          const sel = tasksOf[S.rank][S.task];
          if (sel) {
            (sel.pred || []).forEach((p) => edges.push({ from: p, to: sel.tag, kind: 'in' }));
            (sel.succ || []).forEach((q) => edges.push({ from: sel.tag, to: q, kind: 'out' }));
          }
        }
        /* two different reasons an edge cannot be drawn, and they must not be
         * reported as one number: `untraced` means deps.json names a task this
         * trace has no blocks for at all (4 of csa_merge_pack_publish's 6
         * predecessors are like this), `filtered` means the block exists but
         * its lane is hidden by the current 泳道 filter. Only the second one
         * goes away by changing the view. */
        let drawn = 0, untraced = 0, filtered = 0, bad = 0;
        ctx.save();
        ctx.lineWidth = 1.25;
        edges.forEach((e) => {
          const a = gateOut(e.from), b = gateIn(e.to);
          if (!a || !b) { untraced += 1; return; }
          const ra = rowOf[a.name], rb = rowOf[b.name];
          if (!ra || !rb) { filtered += 1; return; }
          const x1 = sx(a.end), x2 = sx(b.start);
          if (Math.max(x1, x2) < plotX || Math.min(x1, x2) > plotX + plotW) return;
          const cx1 = clamp(x1, plotX, plotX + plotW);
          const cx2 = clamp(x2, plotX, plotX + plotW);
          const y1 = ra.y + ROW_H / 2, y2 = rb.y + ROW_H / 2;
          const violation = b.start < a.end - 0.01;
          if (violation) bad += 1;
          ctx.strokeStyle = cssVar(violation ? '--danger'
            : e.kind === 'out' ? '--primary' : e.kind === 'path' ? '--warning' : '--accent');
          ctx.globalAlpha = violation ? 0.9 : 0.6;
          ctx.setLineDash(violation ? [4, 3] : []);
          /* a horizontal-tangent cubic keeps the curve out of the rows it
           * crosses instead of cutting diagonally through every bar */
          const bow = Math.max(14, Math.min(60, Math.abs(cx2 - cx1) * 0.4));
          ctx.beginPath();
          ctx.moveTo(cx1, y1);
          ctx.bezierCurveTo(cx1 + bow, y1, cx2 - bow, y2, cx2, y2);
          ctx.stroke();
          /* arrowhead on the consumer side */
          const dir = cx2 >= cx1 ? 1 : -1;
          ctx.setLineDash([]);
          ctx.beginPath();
          ctx.moveTo(cx2, y2);
          ctx.lineTo(cx2 - dir * 5, y2 - 3);
          ctx.lineTo(cx2 - dir * 5, y2 + 3);
          ctx.closePath();
          ctx.fillStyle = ctx.strokeStyle;
          ctx.fill();
          ctx.globalAlpha = 1;
          drawn += 1;
        });
        ctx.restore();
        const bits = [(S.deps === 'path' ? '路径依赖 ' : '依赖连线 ') + drawn + ' 条'];
        if (bad) bits.push(bad + ' 条早发（消费者先上核）');
        if (untraced) bits.push(untraced + ' 条端点在本 trace 里没有块');
        if (filtered) bits.push(filtered + ' 条端点被泳道筛选隐藏');
        depsReadout.textContent = edges.length ? bits.join(' · ') : '';
        depsReadout.classList.toggle('is-danger', !!bad);
      } else {
        depsReadout.textContent = '';
        depsReadout.classList.remove('is-danger');
      }

      drawSelGuide(ctx, sx, plotX, plotW, 26, h - 4);

      /* numbered markers matching the evidence chips, drawn last so nothing covers them */
      const seen = {};
      markers.forEach((m) => {
        if (seen[m.n]) return;
        seen[m.n] = 1;
        const cx = clamp(m.x, plotX + 7, plotX + plotW - 7);
        ctx.beginPath();
        ctx.arc(cx, m.y + ROW_H / 2, 7, 0, Math.PI * 2);
        ctx.fillStyle = cssVar('--background');
        ctx.fill();
        ctx.strokeStyle = cssVar('--foreground');
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.fillStyle = cssVar('--foreground');
        ctx.font = '700 9px ' + cssVar('--font-mono');
        ctx.textAlign = 'center';
        ctx.fillText(String(m.n), cx, m.y + ROW_H / 2);
        ctx.textAlign = 'left';
        ctx.font = '500 10px ' + cssVar('--font-sans');
      });

      /* auto-scroll a chip-selected lane into view */
      if (S.scrollToLane) {
        const row = laneLayout.find((r) => r.name === S.scrollToLane);
        if (row) laneHost.scrollTop = Math.max(0, row.y - laneHost.clientHeight / 2);
        S.scrollToLane = null;
      }
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
          if (S.critOnly && !pathSet[task.tag]) continue;
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
      S.pathFocus = cp.segments.some((seg) => seg.tag === hit.task.tag);
      if (guidedJourneyActive() && activeFinding().focus
        && activeFinding().focus.task === hit.task.tag) {
        S.scrollToLane = hit.lane;
        render();
        return;
      }
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
    if (QW()) {
      if (S.l1Panel === 'pmu') { qwPmuPanel(stage); return; }
      /* the per-kernel pipeline model is built on tp1/prefill's tasks; on any
       * other capture there is no task to point it at */
      if (!onPrimaryVariant()) {
        const sec = el('section');
        sec.appendChild(sectionHead('单核流水', qwVariant().label + ' 没有可用的任务对象',
          el('span', 'tc-readout', '任务无名 / 无依赖')));
        sec.appendChild(el('p', 'tc-fineprint',
          '本层要落到一个具体 kernel 上：需要任务名、块级起止和该 kernel 的编译信息。'
          + '这份采集只有 (core, token, reg_id, 起止, receive_to_start)，落不到 kernel。'));
        const jump = el('div', 'tc-actions');
        jump.appendChild(btn('看 PMU 实测', { on: () => { S.l1Panel = 'pmu'; render(); } }));
        jump.appendChild(btn('切到 TP=1 · prefill', {
          variant: 'ghost',
          on: () => { S.variant = 'tp1:prefill'; render(); },
        }));
        sec.appendChild(jump);
        stage.appendChild(sec);
        qwGapNote(stage);
        return;
      }
    }
    const rank = R();
    const t = curTask();
    const role = pathRole(rank, t.tag);

    /* --- identity + measured split --- */
    const idSec = el('section');
    idSec.appendChild(sectionHead(t.callable, t.tag + ' · ' + t.kind.toUpperCase() + ' · task ' + t.id
      + ' · ' + t.kernelCount + ' kernel',
      (function () {
        const ro = el('span', 'tc-readout' + (role.onCpm ? ' is-crit' : ''), pathLabel(role));
        ro.title = pathTitle(role);
        return ro;
      })()));
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

    /* --- one scope, two kernels: the Cube/Vec pairing, measured --- */
    if (t.kernelCount > 1) {
      const pairSec = el('section');
      pairSec.appendChild(sectionHead('引擎配对',
        t.kernels.map((k) => k.name).join(' + ') + ' · 同一次 Group launch',
        el('span', 'tc-readout', '最长块比 ' + t.pairRatio + 'x')));
      pairSec.appendChild(table([
        { label: 'kernel', cell: (k) => esc(k.name), mono: true },
        { label: 'FuncId', cell: (k) => String(k.funcId), mono: true, num: true },
        { label: '引擎', cell: (k) => (k.engine === 'aic' ? 'AIC (Cube)' : 'AIV (Vec)') },
        { label: '块 / 核', cell: (k) => k.blocks + ' / ' + k.cores, num: true },
        { label: 'core-time', cell: (k) => num(k.coreTime, 1), mono: true, num: true },
        { label: '最长块', cell: (k) => num(k.durMax, 2), mono: true, num: true },
        { label: '均值', cell: (k) => num(k.durMean, 2), mono: true, num: true },
      ], t.kernels, {}));
      pairSec.appendChild(el('p', 'tc-note',
        '两半共用一个 taskId，只有 event-hint 里的 FuncId 能把它们分开 —— 按 taskId 汇总会把 Vec 侧的 '
        + num(t.engines.aiv.coreTime, 0) + ' us 记到 Cube 侧的名下。'
        + '两侧最长块 ' + num(t.engines.aic.durMax, 1) + ' / ' + num(t.engines.aiv.durMax, 1)
        + ' us，而整段 span 只有 ' + num(t.span, 1) + ' us：'
        + (t.pairRatio < 1.3 ? '两半没有错开，是块内串行。' : '慢的一侧决定整块时长。')));
      stage.appendChild(pairSec);
    }

    /* --- three measurements of the same block, side by side --- */
    const splitSec = el('section');
    splitSec.appendChild(sectionHead('一个块的三种口径',
      'kernel · +local_setup · +hand-off (dispatch→finish)'));
    const splitHost = el('div', 'tc-canvas-strip');
    const splitCanvas = el('canvas');
    splitHost.appendChild(splitCanvas);
    splitSec.appendChild(splitHost);
    stage.appendChild(splitSec);

    /* --- per-core block strip + duration distribution --- */
    const distSec = el('section');
    distSec.appendChild(sectionHead('块分布',
      t.blockCount + ' 块 / ' + t.coreCount + ' 核 · ' + num(t.blockCount / t.coreCount, 2) + ' 波'));
    const distHost = el('div', 'tc-canvas-host');
    const distCanvas = el('canvas');
    distHost.appendChild(distCanvas);
    distSec.appendChild(distHost);
    stage.appendChild(distSec);

    /* --- on-chip tile budget --- */
    const calcSec = el('section');
    calcSec.appendChild(sectionHead('片上预算试算',
      'Left / Right = 编译器选的 L0A / L0B staging',
      el('span', 'tc-readout', '上限取自本 run MemoryReuse 报告')));
    calcSec.appendChild(renderCalc());
    stage.appendChild(calcSec);

    /* --- compiler hints, honestly unlinked --- */
    const hintSec = el('section');
    const mods = ['all'].concat(D.tileFiles.map((f) => f.file));
    hintSec.appendChild(sectionHead('编译提示', D.hints.length + ' 条 · 按模块聚合 · '
      + (D.sourceMap ? 'scope→源码已重建，提示仍按 hint 自带的行号' : '无 kernel→源码映射'),
      field('模块', select(mods.map((m) => ({ id: m, label: m === 'all' ? '全部模块' : m })), S.hintModule,
        (v) => { S.hintModule = v; render(); }))));
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
      hintSec.appendChild(el('p', 'tc-note', '前 80 / ' + hintRows.length + ' 条 · 完整列表见 Problems 面板'));
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
    if (B.freeLR == null) {
      verdict.appendChild(vrow('warn', 'depth ?',
        'stage ' + B.T.depth + ' × max(L,R) = <strong>' + kb(B.depthNeed)
        + '</strong>，但本 dump 无 PH-MR-001，Left/Right 可用字节未知 · 无法判定'));
    } else {
      const depthMsg = B.atLimit
        ? ' · 同配置实测回退：' + D.depthSites[0].file + ':' + D.depthSites[0].line
        : B.depthState === 'fail' ? ' · 超出，MemoryReuse 降至 depth 1'
          : B.depthState === 'warn' ? ' · 余量不足以容纳同驻 tile' : '';
      verdict.appendChild(vrow(B.depthState, B.depthState === 'pass' ? 'depth' : 'depth ↓',
        'stage ' + B.T.depth + ' × max(L,R) = <strong>' + kb(B.depthNeed) + '</strong> / '
        + kb(B.freeLR) + ' free = ' + pct(B.depthRatio * 100, 0) + depthMsg));
    }
    verdict.appendChild(vrow(B.lineOk ? 'pass' : 'warn', B.lineOk ? 'cache line' : '末维不足',
      'N × ' + B.ab.label + ' = <strong>' + B.innermost + 'B</strong> / ' + B.cacheLine + 'B'
      + (B.lineOk ? '' : ' · 需 ' + B.ab.mult + ' 元素倍数（≥ ' + B.needElems + ' 个 ' + B.ab.label + '）')));
    verdict.appendChild(vrow(B.acc > B.accObserved ? 'warn' : 'pass', 'Acc',
      'M × N × ' + B.ac.bytes + 'B × ' + B.T.live + ' = <strong>' + kb(B.acc) + '</strong> · 本 run 实测最大 '
      + kb(B.accObserved) + '（dump 未报上限，超出即待验证）'));
    out.appendChild(verdict);

    const obs = D.l0Tiles.slice(0, 8).map((x) => ({
      mem: x.mem, shape: x.dtype + '[' + x.rows + ',' + x.cols + ']',
      bytes: x.bytes, innermost: x.innermostB, n: x.n,
    }));
    out.appendChild(sectionHead('本 run 出现的 L0 tile', D.l0Tiles.length + ' 种 · 点行回填'));
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
  /* When passes_dump / ptoas / perf_hints come from a different build than the
   * trace, the two layers that read them have to open with that fact — an
   * Explorer breadcrumb is not enough to stop someone matching a line number
   * against this run's tasks. */
  function compileSourceNote(stage) {
    const cs = D.case.compileSource;
    if (!cs) return;
    const sec = el('section');
    sec.appendChild(sectionHead('本层读的是另一次构建', cs.runDir + ' · ' + cs.capturedAt,
      el('span', 'tc-readout is-crit', '与上板数据不同源')));
    const note = el('div', 'tc-note');
    note.dataset.tone = 'warn';
    note.appendChild(el('strong', null, '上板数据 ' + D.case.capturedAt.slice(0, 10)
      + ' · 编译产物 ' + cs.capturedAt.slice(0, 10)));
    note.appendChild(el('span', null, cs.note));
    note.appendChild(el('small', null, '那次构建的程序是 ' + cs.model
      + '；本次上板跑的是 ' + D.case.model + '。'
      + '2026-08-14 的采集没有 passes_dump，所以 Pass 轨迹与 IR 对照只能读旧构建。'));
    sec.appendChild(note);
    stage.appendChild(sec);
  }

  function viewCompiler(stage) {
    compileSourceNote(stage);
    if (S.compilerTab === 'passes') {
      const detailByPass = {};
      (D.passEvidence || []).forEach((d) => { detailByPass[d.idx] = d; });
      const selected = D.passes.find((p) => p.idx === S.pass) || D.passes[0];
      const detail = detailByPass[selected.idx] || { add: 0, del: 0, groups: 0, scopes: [], hunks: [], links: [] };
      const changed = D.passes.filter((p) => {
        const d = detailByPass[p.idx];
        return d && (d.add || d.del);
      }).length;
      const sec = el('section', 'tc-pass-workspace');
      sec.appendChild(sectionHead('编译 IR 全流程 · ' + D.case.program,
        changed + ' / ' + Math.max(0, D.passes.length - 1) + ' 个 Pass 改动了 IR'));

      /* The river is a view of the same selected-pass state as the evidence
       * panel below.  Its strata describe IR form, its dots describe actual
       * snapshot changes — no separate, decorative pipeline is introduced. */
      const strata = [
        { id: 's0', label: 'S0 · 前端', form: 'Tensor IR', until: 0 },
        { id: 's1', label: 'S1 · 规范化张量', form: 'SSA / Tensor', until: 9 },
        { id: 's2', label: 'S2 · 层级化', form: 'Structured IR', until: 12 },
        { id: 's3', label: 'S3 · Tile', form: 'Tile IR', until: 21 },
        { id: 's4', label: 'S4 · 双核 Kernel', form: 'AIC / AIV Kernel', until: 28 },
        { id: 's5', label: 'S5 · 物理内存', form: 'MemRef / 物理内存', until: 34 },
        { id: 's6', label: 'S6 · 运行时', form: 'Runtime IR', until: Infinity },
      ];
      const stratumFor = (p) => strata.find((s) => p.idx <= s.until) || strata[strata.length - 1];
      const kindFor = (p, d) => {
        if (!p.idx || !(d.add || d.del)) return 'same';
        if (/MemoryReuse/.test(p.name) && D.depthSites.some((s) => s.fittedDepth < s.maxReqDepth)) return 'bad';
        if (/MemRef|Memory|Addr|Layout/.test(p.name)) return 'memory';
        if (/Runtime|Host|CallDirection|CommDomain/.test(p.name)) return 'runtime';
        if (/Inline|Outline|Unroll|Split|Expand|LowerPipeline/.test(p.name)) return 'struct';
        if (/Tile|Pipeline|Prefetch|Matmul/.test(p.name)) return 'intent';
        return 'touch';
      };
      const groups = [];
      D.passes.forEach((p, i) => {
        const s = stratumFor(p);
        const last = groups[groups.length - 1];
        if (!last || last.id !== s.id) groups.push({ id: s.id, label: s.label, form: s.form, from: i, to: i });
        else last.to = i;
      });
      const river = el('div', 'tc-pass-river');
      const riverHead = el('div', 'tc-pass-river-head');
      riverHead.appendChild(el('span', null, '相邻快照 · ' + D.passes[0].lines + ' → ' + D.passes[D.passes.length - 1].lines + ' 行'));
      riverHead.appendChild(el('span', null, changed + ' 个关键事件 · ' + D.passes.length + ' 个 Pass'));
      river.appendChild(riverHead);
      const riverScroll = el('div', 'tc-pass-river-scroll');
      const riverScene = el('div', 'tc-pass-river-scene');
      riverScene.style.minWidth = Math.max(1040, D.passes.length * 28) + 'px';
      const formLabel = el('span', 'tc-pass-river-label', 'IR 形态');
      riverScene.appendChild(formLabel);
      groups.forEach((group) => {
        const band = el('div', 'tc-pass-form-band');
        band.dataset.stratum = group.id;
        band.style.left = (group.from / D.passes.length * 100) + '%';
        band.style.width = ((group.to - group.from + 1) / D.passes.length * 100) + '%';
        band.textContent = group.form;
        riverScene.appendChild(band);
      });
      const sequenceLabel = el('span', 'tc-pass-river-label tc-pass-river-seq-label', '执行序 →');
      riverScene.appendChild(sequenceLabel);
      const spine = el('i', 'tc-pass-river-spine'); riverScene.appendChild(spine);
      groups.forEach((group) => {
        const box = el('div', 'tc-pass-stage-band');
        box.dataset.stratum = group.id;
        box.style.left = (group.from / D.passes.length * 100) + '%';
        box.style.width = ((group.to - group.from + 1) / D.passes.length * 100) + '%';
        box.appendChild(el('span', null, group.label));
        riverScene.appendChild(box);
      });
      D.passes.forEach((p, i) => {
        const d = detailByPass[p.idx] || {};
        const kind = kindFor(p, d);
        const b = el('button', 'tc-pass-river-node is-' + kind + (p.idx === selected.idx ? ' is-selected' : ''));
        b.type = 'button'; b.dataset.stratum = stratumFor(p).id;
        b.style.left = ((i + 0.5) / D.passes.length * 100) + '%';
        b.title = String(p.idx).padStart(2, '0') + ' · ' + p.name + (d.add || d.del ? ' · +' + d.add + ' / −' + d.del : ' · 未改动 IR');
        b.setAttribute('aria-label', b.title);
        b.appendChild(el('i', 'dot'));
        if (kind !== 'touch' && kind !== 'same') b.appendChild(el('span', 'idx', String(p.idx).padStart(2, '0')));
        b.addEventListener('click', () => { S.pass = p.idx; S.focus = 'pass'; S.passMode = 'overview'; render(); });
        riverScene.appendChild(b);
      });
      riverScroll.appendChild(riverScene); river.appendChild(riverScroll);
      const legend = el('div', 'tc-pass-river-legend');
      [['struct', '结构变换'], ['intent', '意图相关'], ['bad', '意图被破坏'], ['memory', '内存'], ['runtime', '运行时'], ['touch', '普通改动'], ['same', '未改动']]
        .forEach(([kind, label]) => { const item = el('span'); item.appendChild(el('i', 'is-' + kind)); item.appendChild(el('span', null, label)); legend.appendChild(item); });
      river.appendChild(legend);
      sec.appendChild(river);

      const body = el('div', 'tc-pass-workspace-body');

      const work = el('div', 'tc-pass-detail');
      const head = el('div', 'tc-pass-detail-head');
      const title = el('div');
      title.appendChild(el('span', 'eyebrow', selected.idx === 0 ? '流水线输入' : 'Pass #' + selected.idx + ' · 从 ' + detail.from));
      title.appendChild(el('h3', null, selected.name));
      head.appendChild(title);
      const modes = el('div', 'segmented-control segmented-control-muted');
      [['overview', '变化概览'], ['diff', '代码 Diff']].forEach(([id, label]) => {
        modes.appendChild(btn(label, { size: 'sm', selected: S.passMode === id,
          on: () => { S.passMode = id; render(); } }));
      });
      head.appendChild(modes);
      work.appendChild(head);

      const metrics = el('div', 'tc-pass-metrics');
      [
        ['IR 行', selected.lines, selected.delta === 0 ? '与上一快照等长' : (selected.delta > 0 ? '+' : '') + selected.delta],
        ['实测变更', detail.add + ' + / ' + detail.del + ' −', detail.groups + ' 个改写区域'],
        ['受影响作用域', String(detail.scopes.length), detail.scopes.length ? detail.scopes.slice(0, 2).map((s) => s.name).join(' · ') : '无'],
      ].forEach(([k, v, sub]) => {
        const m = el('div', 'tc-pass-metric');
        m.appendChild(el('span', 'k', k)); m.appendChild(el('strong', null, v)); m.appendChild(el('span', 'sub', sub));
        metrics.appendChild(m);
      });
      work.appendChild(metrics);

      if (detail.links && detail.links.length) {
        const links = el('div', 'tc-pass-links');
        links.appendChild(el('span', 'label', '运行内关联'));
        detail.links.forEach((link) => {
          links.appendChild(btn(link.label, { size: 'sm', on: () => {
            if (link.findingId && findingById[link.findingId]) {
              S.finding = link.findingId; S.focus = 'finding'; applyFocus(findingById[link.findingId]);
            } else if (link.view) S.view = link.view;
            render();
          } }));
        });
        work.appendChild(links);
      }

      if (!detail.add && !detail.del) {
        work.appendChild(el('div', 'tc-pass-empty', selected.idx === 0
          ? '前端 IR 是流水线的事实起点；选择后续 Pass 查看相邻快照的改写证据。'
          : '相邻快照逐行一致：这个 Pass 在本次编译输入上是空操作。'));
      } else if (S.passMode === 'overview') {
        const scopes = el('div', 'tc-pass-scopes');
        scopes.appendChild(el('span', 'label', '受影响作用域'));
        detail.scopes.forEach((scope) => {
          const chip = el('span', 'tc-pass-scope');
          chip.appendChild(el('code', null, scope.name));
          chip.appendChild(el('span', null, scope.lines + ' 行变更'));
          scopes.appendChild(chip);
        });
        work.appendChild(scopes);
        const hunkList = el('div', 'tc-pass-hunks');
        detail.hunks.slice(0, 3).forEach((hunk, i) => {
          const card = el('article', 'tc-pass-hunk');
          card.appendChild(el('div', 'h', '改写区域 ' + (i + 1) + ' · ' + hunk.scopes.join(' / ')));
          const pre = el('pre', 'tc-pass-code');
          pre.textContent = hunk.before.map((line) => '− ' + line).join('\n')
            + (hunk.beforeMore ? '\n− … ' + hunk.beforeMore + ' 行' : '')
            + (hunk.before.length && hunk.after.length ? '\n' : '')
            + hunk.after.map((line) => '+ ' + line).join('\n')
            + (hunk.afterMore ? '\n+ … ' + hunk.afterMore + ' 行' : '');
          card.appendChild(pre);
          hunkList.appendChild(card);
        });
        work.appendChild(hunkList);
      } else {
        const diffList = el('div', 'tc-pass-diff-list');
        detail.hunks.forEach((hunk, i) => {
          const card = el('article', 'tc-pass-diff-card');
          card.appendChild(el('div', 'h', '区域 ' + (i + 1) + ' · ' + hunk.scopes.join(' / ')
            + ' · 前 ' + hunk.beforeLine + ' / 后 ' + hunk.afterLine + ' 行'));
          const grid = el('div', 'tc-pass-diff-grid');
          [['删除', hunk.before, hunk.beforeMore, 'before'], ['新增', hunk.after, hunk.afterMore, 'after']].forEach(([label, lines, more, side]) => {
            const sideEl = el('div', 'tc-pass-diff-side'); sideEl.dataset.side = side;
            sideEl.appendChild(el('span', 'label', label));
            const pre = el('pre', 'tc-pass-code');
            pre.textContent = lines.length ? lines.join('\n') + (more ? '\n… ' + more + ' 行' : '') : '—';
            sideEl.appendChild(pre); grid.appendChild(sideEl);
          });
          card.appendChild(grid); diffList.appendChild(card);
        });
        work.appendChild(diffList);
      }
      body.appendChild(work);
      sec.appendChild(body);
      stage.appendChild(sec);
    }

    if (S.compilerTab === 'depth') {
      const sec = el('section');
      if (!D.depthSites.length) {
        /* MemoryReuse never reported a degradation here. That is a different
         * statement from "we found nothing", so spell out what was checked. */
        sec.appendChild(sectionHead('软流水深度回退', '本 run 无 PH-MR-001'));
        sec.appendChild(table([
          { label: '事实', cell: (r) => esc(r[0]), mono: true },
          { label: '含义', cell: (r) => esc(r[1]) },
        ], [
          ['PH-MR-001 × 0', 'MemoryReuse 未报告过任何一次深度回退'],
          ['pl.pipeline × ' + D.pipelineSites.length, '请求的 stage 都放得下，或该 kernel 未进 MemoryReuse'],
          ['Left / Right / Vec 可用字节 缺失', '片上预算试算器无本 run 实测上限可对账'],
        ], {}));
      } else {
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
        ], D.depthSites.map((s) => Object.assign({
          __selected: s.key === S.hintSite, __subject: !!subjectSiteSet()[s.key],
        }, s)), {
          onPick: (s) => { S.hintSite = s.key; S.focus = 'hint'; render(); },
        }));
      }
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
      ], rows.map((s) => Object.assign({
        __selected: s.key === S.hintSite, __subject: !!subjectSiteSet()[s.key],
      }, s)), {
        onPick: (s) => { S.hintSite = s.key; S.focus = 'hint'; render(); },
      }));
      stage.appendChild(sec2);

      const guide = el('section');
      guide.appendChild(sectionHead('末维目标', '凑满 ' + cacheLine + 'B 所需元素倍数'));
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
    compileSourceNote(stage);
    const layoutPass = D.passes.find((p) => p.name === 'ResolveBackendOpLayouts');
    const spacePass = D.passes.find((p) => p.name === 'InferTileMemorySpace');
    const cacheLine = (D.hints.find((h) => h.cacheLineB) || {}).cacheLineB || 512;

    const have = el('section');
    have.appendChild(sectionHead('工具链与约束', 'binary_context · Pass dump · L0 tile'));
    have.appendChild(tiles([
      { k: 'platform', v: D.case.toolchain.platform || D.case.backend },
      { k: 'pto-isa', v: D.case.toolchain.ptoIsaRevision ? D.case.toolchain.ptoIsaRevision.slice(0, 8) : '—', u: 'revision' },
      { k: 'runtime', v: (D.case.toolchain.runtimeName || '—').split('_')[0],
        u: D.case.toolchain.runtimeRevision ? D.case.toolchain.runtimeRevision.slice(0, 8) : (D.case.toolchain.aicpuThreads ? D.case.toolchain.aicpuThreads + ' AICPU 线程' : '') },
      { k: 'cache line', v: cacheLine, u: 'B' },
      { k: 'L0 tile 形状', v: D.l0Tiles.length, u: '种（AutoTileMatmulL0）' },
      { k: 'Left/Right 可用', v: D.budgets.Right ? kb(D.budgets.Right.freeB) : '—',
        u: D.budgets.Right ? 'MemoryReuse 报告' : '无 PH-MR-001' },
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
      { label: D.budgets.Right ? '占 ' + kb(D.budgets.Right.freeB) : '相对最大',
        cell: (r) => {
          const base = D.budgets.Right ? D.budgets.Right.freeB
            : Math.max.apply(null, D.l0Tiles.map((x) => x.bytes));
          return r.mem === 'Acc' ? '—' : bar(r.bytes / base, r.bytes > base ? 'bad' : 'neutral');
        } },
      { label: '出现', key: 'n', num: true },
    ], D.l0Tiles.map((r) => Object.assign({ shape: '[' + r.rows + ',' + r.cols + ']' }, r)), {}));
    stage.appendChild(lay);

    const missing = el('section');
    const A = D.case.artifacts;

    /* kernel -> source: reconstructed, not read out of the dump */
    const SM = D.sourceMap;
    const srcSec = el('section');
    if (SM) {
      srcSec.appendChild(sectionHead('scope → 源码',
        SM.covered + '/' + SM.total + ' 覆盖 · ' + SM.unique + ' 唯一 · ' + SM.ambiguous + ' 多候选'));
      srcSec.appendChild(table([
        { label: '项', cell: (r) => esc(r[0]), mono: true },
        { label: '值', cell: (r) => esc(r[1]) },
      ], [
        ['来源', '不在 dump 内 —— 由 ' + SM.root + '/ 的源码重建'],
        ['入口', SM.entry + ' · 传递导入 ' + SM.modules.length + ' 个模块'],
        ['依据', 'pl.spmd(..., name_hint="X") 与 OutlineIncoreScopes 外联出的函数同名'],
        ['索引到的 name_hint', String(SM.hintCount)],
        ['唯一定位', SM.unique + ' 个 scope'],
        ['多候选', SM.ambiguous + ' 个（同名 hint 出现在多处，名字消不掉歧义）'],
        ['未匹配', String(SM.missing)],
        ['粒度', '映射的是 scope 不是 kernel —— 混合 scope 拆出的 _aic / _aiv '
          + '两个 kernel 共用同一个 name_hint，因此指向同一处源码'],
        ['不能做的事', '只给出 scope 写在哪里，不把实测块时长归到某一行'],
      ], {}));
    } else {
      srcSec.appendChild(sectionHead('scope → 源码', '源码树不在仓库内'));
      srcSec.appendChild(table([
        { label: '项', cell: (r) => esc(r[0]), mono: true },
        { label: '值', cell: (r) => esc(r[1]) },
        { label: '状态', cell: () => '<span class="bad">缺失</span>' },
      ], [
        ['模型源码', D.case.sourceRoot || '未记录'],
        ['可重建性', '拿到源码树后可按 name_hint 重建，方法同 decode_csa'],
      ], {}));
    }
    stage.appendChild(srcSec);

    /* PTOAS sources, when this dump carries them */
    if (A.ptoas) {
      const src = el('section');
      const units = D.case.ptoasUnits;
      const maxPto = Math.max.apply(null, units.map((u) => u.ptoLines));
      src.appendChild(sectionHead('PTOAS 单元', units.length + ' 个 · .pto → .cpp · '
        + Object.keys(A.kernelDirs).map((k) => k + ' ' + A.kernelDirs[k]).join(' / ')));
      src.appendChild(table([
        { label: '单元', key: 'name', mono: true },
        { label: '.pto 行', key: 'ptoLines', num: true },
        { label: '', cell: (r) => bar(r.ptoLines / maxPto, r.ptoLines === maxPto ? 'warn' : 'neutral') },
        { label: '.cpp 行', num: true, cell: (r) => (r.cppLines == null ? '—' : r.cppLines) },
        { label: '展开比', num: true, cell: (r) => (r.cppLines == null ? '—'
          : num(r.cppLines / r.ptoLines, 2) + 'x') },
      ], units.slice().sort((a, b) => b.ptoLines - a.ptoLines), { tall: true }));
      stage.appendChild(src);
    }

    /* what is still missing — computed, not a fixed list */
    const gaps = [
      [A.ptoas ? null : 'ptoas/*.pto', '每个 kernel 的 PTOAS 源与展开后的 cpp'],
      [Object.keys(A.kernelDirs).length ? null : 'kernels/', '实际编译出的 AIC / AIV 二进制'],
      ['PTOAS TileLib 模板记录', '模板候选与选中原因'],
      ['VPTO scheduler 排布报告', '依赖、延迟、寄存器压力、重物化'],
      ['cycle cost model 预测', '与实测块时长对账'],
      ['PMU counter', 'Cube / Vec / MTE / FIXPIPE，需单独建 PMU-on 基线'],
    ].filter((r) => r[0]);
    missing.appendChild(sectionHead('缺失产物', gaps.length + ' 项'));
    missing.appendChild(table([
      { label: '产物', cell: (r) => esc(r[0]), mono: true },
      { label: '用于', cell: (r) => esc(r[1]) },
      { label: '状态', cell: () => '<span class="bad">缺失</span>' },
    ], gaps, {}));
    stage.appendChild(missing);
  }

  /* ======================================================== inspector */
  /* The rail opens the three questions the page exists to answer and folds
   * the supporting detail. With everything expanded L2 was 6.3 screens of
   * scrolling against 1-2 on every other tab. Nothing is removed: a folded
   * section is one click from its full content, and the fold state is kept
   * per title so it survives the rail's re-render. */
  const FOLD_BY_DEFAULT = { '统计口径': 1, '引擎配对': 1, 'spmd 展开': 1 };

  function inspectorSection(title, kicker) {
    const s = el('section', 'inspector-section');
    const h = el('div', 'inspector-section-head');
    if (!FOLD_BY_DEFAULT[title]) {
      h.appendChild(el('h3', 'inspector-section-title', title));
      if (kicker) h.appendChild(el('span', 'inspector-section-kicker', kicker));
      s.appendChild(h);
      return s;
    }
    if (S.folded[title] === undefined) S.folded[title] = true;
    const open = !S.folded[title];
    s.classList.add('is-foldable');
    if (!open) s.classList.add('is-folded');
    const btn = el('button', 'inspector-section-toggle');
    btn.type = 'button';
    btn.setAttribute('aria-expanded', open ? 'true' : 'false');
    btn.appendChild(el('span', 'chev', open ? '−' : '+'));
    btn.appendChild(el('h3', 'inspector-section-title', title));
    if (kicker) btn.appendChild(el('span', 'inspector-section-kicker', kicker));
    btn.addEventListener('click', () => {
      S.folded[title] = !S.folded[title];
      renderInspector();
    });
    h.appendChild(btn);
    s.appendChild(h);
    return s;
  }

  /* [label, value] or [label, value, title] -- the third slot is where an
   * explanation goes that used to cost a whole card. */
  function kv(pairs) {
    const d = el('dl', 'tc-kv');
    pairs.forEach((p) => {
      if (p[1] == null) return;
      const dt = el('dt', p[2] ? 'has-q' : null, p[0]);
      const dd = el('dd', null, p[1]);
      if (p[2]) { dt.title = p[2]; dd.title = p[2]; }
      d.appendChild(dt); d.appendChild(dd);
    });
    return d;
  }

  function renderInspector() {
    const host = $('#inspector');
    host.textContent = '';
    const title = $('[data-bind="inspectorTitle"]');
    const meta = $('[data-bind="inspectorMeta"]');

    const focus = S.focus || defaultFocus();
    const crumb = scopeCrumb();
    if (crumb) host.appendChild(crumb);
    if (focus === 'scope') renderScopeInspector(host, title, meta);
    else if (focus === 'finding' && S.finding) renderFindingInspector(host, title, meta);
    else if (focus === 'hint' && S.hintSite) renderHintInspector(host, title, meta);
    else if (focus === 'pass') renderPassInspector(host, title, meta);
    else if (focus === 'run') renderRunInspector(host, title, meta);
    else renderTaskInspector(host, title, meta);

    host.appendChild(renderLedger());
  }

  /* the inspector follows the view unless the user pinned something else.
   * L2 asks "which scope ate the time / when was the machine idle";
   * L1 asks "what happened inside one kernel". Different panel. */
  function defaultFocus() {
    if (S.view === 'e2e' || S.view === 'isa') return 'run';
    if (S.view === 'compiler') return S.compilerTab === 'passes' ? 'pass' : (S.hintSite ? 'hint' : 'run');
    /* scope and task panels are built from the one capture that has task
     * objects; with another capture armed they would describe a different run
     * than the stage does */
    if (QW() && !onPrimaryVariant()) return 'run';
    if (S.view === 'l2') return 'scope';
    return 'task';
  }

  /* ------------------------------------------------------ L2 inspector
   * Answers the two questions the swimlane alone cannot: where the core-time
   * went by scope, and which stretches of wall time the machine sat idle.
   * Σdur on its own ranks the fat scopes; pairing it with DAG slack separates
   * "fat" from "fat and pinned to the critical path". */
  const lanesOf = () => R().swimlane.lanes.length;

  /* membership in each of the two paths, for one task */
  function pathRole(rank, tag) {
    const onCpm = rank.critical.tags.indexOf(tag);
    const seg = rank.cpath.segments.filter((sg) => sg.tag === tag)[0];
    const obsIdx = seg ? rank.cpath.segments.indexOf(seg) : -1;
    return {
      onCpm: onCpm >= 0, cpmIdx: onCpm + 1, cpmN: rank.critical.tags.length,
      onObs: !!seg, obsIdx: obsIdx + 1, obsN: rank.cpath.segments.length,
      seg: seg || null,
    };
  }
  function pathLabel(r) {
    if (!r.onCpm && !r.onObs) return '不在执行主路径或依赖关键路径';
    const bits = [];
    if (r.onCpm) bits.push('依赖关键路径 ' + r.cpmIdx + '/' + r.cpmN);
    if (r.onObs) bits.push('执行主路径 ' + r.obsIdx + '/' + r.obsN);
    return '在 ' + bits.join(' · ');
  }
  function pathTitle(r) {
    return '执行主路径 = 从当前 rank 最晚完成的任务反向沿依赖与同核前序归因，用于解释实测设备跨度。' + NL
      + '它不是依赖图的静态下界；同核前序说明执行顺序，不足以证明任务 ready 后被调度器延迟。' + NL
      + '依赖关键路径 = 过滤后的依赖图 CPM，只说明依赖约束下界；两条路径可能不同。'
      + (r.seg ? NL + '本节点前的 stall ' + num(r.seg.stall, 2) + ' us（' + r.seg.kind + '）' : '');
  }
  /* scopes are source-level pl.spmd regions; kernels are what launched */
  const scopeCountOf = () => R().scopes.length;

  /* file:line, plus an honest marker when the name matches more than one
   * source site or only matched after a compiler suffix was stripped */
  /* Cube / Vector, and the split when a scope compiled into both */
  function engineChip(sc) {
    if (sc.kind === 'mix') return 'C+V';
    return sc.kind === 'aic' ? 'C' : 'V';
  }
  function engineTitle(sc) {
    const bits = [];
    ['aic', 'aiv'].forEach((en) => {
      const e = sc.engines && sc.engines[en];
      if (!e) return;
      bits.push((en === 'aic' ? 'AIC (Cube)' : 'AIV (Vec)') + ' ' + num(e.coreTime, 0)
        + ' us · ' + e.blocks + ' 块 / ' + e.cores + ' 核 · 最长 ' + num(e.durMax, 1) + ' us');
    });
    if (sc.kernelCount > 1) {
      bits.push('拆成 ' + sc.kernelCount + ' 个 kernel：' + sc.kernels.map((k) => k.name).join('、'));
    }
    return bits.join(String.fromCharCode(10));
  }

  function srcLabel(src) {
    if (!src) return null;
    return src.file + ':' + src.line
      + (src.candidates > 1 ? ' +' + (src.candidates - 1) : '');
  }
  function srcTitle(src) {
    if (!src) return '';
    const bits = ['name_hint="' + src.hint + '"'];
    if (!src.exact) bits.push('（callable 去掉编译器后缀后匹配）');
    if (src.candidates > 1) {
      bits.push(src.candidates + ' 处同名候选：'
        + src.sites.map((x) => x.file + ':' + x.line).join('、'));
    }
    return bits.join('\n');
  }

  /* Restore the window and the panel the drill-down replaced. */
  function scopeBack() {
    const r = S.scopeReturn;
    S.scopeReturn = null;
    if (r) { S.t0 = r.t0; S.t1 = r.t1; S.task = r.task || S.task; }
    S.focus = 'scope';
    S.focusEvidence = false;
    render();
  }

  /* A breadcrumb the drill-down can be undone from. Rendered by whichever
   * panel the drill landed on, so the trail is visible where the user is. */
  function scopeCrumb() {
    const r = S.scopeReturn;
    if (!r || S.view !== 'l2') return null;
    /* a scope drill left the panel behind; an idle drill only moved the window */
    const panelMoved = (S.focus || defaultFocus()) !== 'scope';
    const bar = el('div', 'tc-crumb');
    const back = el('button', 'tc-crumb-back',
      panelMoved ? '← scope 排行' : '← 恢复时间窗');
    back.type = 'button';
    back.title = (panelMoved ? '回到 L2 面板，并恢复 ' : '恢复 ')
      + num(r.t0, 0) + '–' + num(r.t1, 0) + ' us 的时间窗（Esc）';
    back.addEventListener('click', scopeBack);
    bar.appendChild(back);
    bar.appendChild(el('span', 'sep', '/'));
    bar.appendChild(el('span', 'cur',
      r.scope || (num(S.t0, 0) + '–' + num(S.t1, 0) + ' us')));
    return bar;
  }

  function renderScopeInspector(host, title, meta) {
    const rank = R();
    title.textContent = 'L2 · ' + S.rank;
    meta.textContent = rank.scopes.length + ' scope';

    /* --- 1. accounting: two views of the same block, stated once --- */
    const A = rank.accounting;
    const s0 = inspectorSection('统计口径', A.schedCoreTime ? 'Worker / Scheduler' : 'Worker');
    const kernelTotal = rank.scopes.reduce((a, x) => a + x.kernelCount, 0);
    const splitScopes = rank.scopes.filter((x) => x.kernelCount > 1);
    s0.appendChild(kv([
      ['Worker View', num(A.workerCoreTime, 0) + ' us · ' + A.workerBlocks + ' 块'],
      ['  kernel', num(A.workerKernelTime, 0) + ' us'],
      ['  setup', num(A.workerSetupTime, 0) + ' us'],
      ['Scheduler View', A.schedCoreTime
        ? num(A.schedCoreTime, 0) + ' us · ' + A.schedBlocks + ' 块' : '—'],
      ['hand-off 差', A.handoff == null ? '—' : '+' + num(A.handoff, 0) + ' us'],
      ['scope / kernel', rank.scopes.length + ' / ' + kernelTotal,
        splitScopes.length
          ? 'scope = 源码里一个 pl.spmd 区域；kernel = 设备上真正 launch 的函数。' + NL
            + 'ExpandMixedKernel 把 ' + splitScopes.length + ' 个混合 scope 各拆成 AIC + AIV，'
            + '所以多 ' + (kernelTotal - rank.scopes.length) + ' 个：'
            + splitScopes.map((x) => x.name).join('、') + NL
            + '两半共用一次 launch（同一个 taskId），只能靠 FuncId 分开。'
          : 'scope = 源码里一个 pl.spmd 区域；kernel = 设备上真正 launch 的函数。'
            + '本 case 没有混合 scope，两者一一对应。'],
    ]));
    if (A.naiveSum) {
      s0.appendChild(el('div', 'inspector-soft-card is-warning',
        '同一个块在 trace 里出现两次。两边相加得 ' + num(A.naiveSum, 0)
        + ' us —— 这是重复计数，不是总量。下面的 scope 排行只用 Worker View。'));
    }
    /* s0 is built here but appended below: the rail leads with the three
     * questions the L2 page exists to answer, not with its bookkeeping. */

    /* --- 1. where the makespan went, attributed --- */
    host.appendChild(renderCriticalPath(rank));

    /* --- 2. scope ranking: core-time × slack --- */
    const top = rank.scopes.slice(0, 10);
    const maxCore = top[0] ? top[0].coreTime : 1;
    const s1 = inspectorSection('scope 排行', 'Worker core-time · 前 ' + top.length + ' / ' + rank.scopes.length);
    const rows = el('div', 'tc-scoperows');
    const hd = el('div', 'tc-scoperow is-head');
    [['scope', 'l', '外联后的 incore scope，按 Worker core-time 排'],
     ['引擎', 'e', 'C = AIC (Cube)，V = AIV (Vec)，C+V = 混合 scope'],
     ['core-time', 'n', 'Σ 块时长，跨所有核'],
     ['占', 'n', '占本 rank 总 core-time'],
     ['slack', 'n', 'DAG 上这个 scope 最紧的任务能被推迟多久。'
       + '0 = 在依赖关键路径上，动它直接缩短总时长；slack 大 = 它胖但不急，先看并行度。' + NL
       + '不含资源争抢 —— 等核那部分在「关键路径归责」里记作 core-wait。']]
      .forEach((c) => { const x = el('span', c[1], c[0]); x.title = c[2]; hd.appendChild(x); });
    rows.appendChild(hd);
    top.forEach((sc) => {
      const b = el('button', 'tc-scoperow' + (sc.onCrit ? ' is-crit' : ''));
      b.type = 'button';
      b.title = sc.taskCount + ' 任务 / ' + sc.blocks + ' 块 · wall ' + num(sc.wall, 1) + ' us'
        + (sc.onCrit ? ' · 依赖关键路径上 ' + sc.critNodes + ' 个节点' : ' · 不在依赖关键路径上')
        + (sc.src ? '\n' + srcLabel(sc.src) + '\n' + srcTitle(sc.src) : '');
      const nm = el('span', 'l');
      nm.appendChild(el('i', 'bar'));
      nm.lastChild.style.width = ((sc.coreTime / maxCore) * 100).toFixed(1) + '%';
      nm.appendChild(el('span', 'tx', sc.name));
      b.appendChild(nm);
      if (sc.src) {
        const sl = el('span', 'src' + (sc.src.candidates > 1 ? ' is-amb' : ''), srcLabel(sc.src));
        nm.appendChild(sl);
      }
      /* which engine burns this scope's core-time -- C, V, or both */
      const eg = el('span', 'e is-' + sc.kind, engineChip(sc));
      eg.title = engineTitle(sc);
      b.appendChild(eg);
      b.appendChild(el('span', 'n', num(sc.coreTime, 0)));
      b.appendChild(el('span', 'n muted', pct(sc.coreShare, 1)));
      /* zero slack on a fat scope is the actionable combination */
      b.appendChild(el('span', 'n' + (sc.minSlack === 0 ? ' hot' : ''),
        sc.minSlack === 0 ? '0' : num(sc.minSlack, 0)));
      b.addEventListener('click', () => {
        const from = S.scopeReturn || { t0: S.t0, t1: S.t1, task: S.task, focus: S.focus };
        S.scopeReturn = { t0: from.t0, t1: from.t1, scope: sc.name, task: from.task, focus: from.focus };
        S.task = sc.tags[0];
        S.focus = 'task';
        const t = tasksOf[S.rank][sc.tags[0]];
        if (t) { const pad = Math.max(40, t.span * 0.3); setWindow(t.start - pad, t.end + pad); }
        render();
      });
      rows.appendChild(b);
    });
    s1.appendChild(rows);
    host.appendChild(s1);

    /* --- 3. idle windows --- */
    const idle = rank.idleRuns || [];
    const idleUs = idle.reduce((a, r) => a + r.us, 0);
    const s2 = inspectorSection('空转窗口',
      idle.length ? '前 ' + Math.min(5, idle.length) + ' / ' + idle.length + ' 段 · '
        + pct((idleUs / rank.swimlane.spanUs) * 100, 1) : '无');
    if (!idle.length) {
      s2.appendChild(el('div', 'tc-foot',
        '没有 AIC 与 AIV 同时低于 ' + rank.idlePct + '% 的窗口（窗宽 '
        + num(rank.occWindowUs, 1) + ' us）。'));
    } else {
      const ir = el('div', 'tc-scoperows');
      const ih = el('div', 'tc-scoperow is-idle is-head');
      ['窗口', '时长', 'AIC', 'AIV'].forEach((t, i) => ih.appendChild(el('span', i ? 'n' : 'l', t)));
      ir.appendChild(ih);
      idle.slice(0, 5).forEach((r) => {
        const b = el('button', 'tc-scoperow is-idle');
        b.type = 'button';
        b.title = '窗口内实际在跑：' + (r.running.top.map((t) => t.callable + ' ' + num(t.us, 0) + ' us/' + t.blocks + ' 块').join('，') || '无')
          + String.fromCharCode(10) + '核容量占用 ' + pct(r.running.capacityPct, 1);
        b.appendChild(el('span', 'l', num(r.t0, 0) + '–' + num(r.t1, 0) + ' us'));
        b.appendChild(el('span', 'n hot', num(r.us, 0)));
        b.appendChild(el('span', 'n muted', pct(r.aic, 0)));
        b.appendChild(el('span', 'n muted', pct(r.aiv, 0)));
        b.addEventListener('click', () => {
          if (!S.scopeReturn) {
            S.scopeReturn = { t0: S.t0, t1: S.t1, scope: null, task: S.task, focus: S.focus };
          }
          const pad = Math.max(30, r.us * 0.2);
          setWindow(r.t0 - pad, r.t1 + pad);
          redrawStage(); renderToolbar(); renderInspector();
        });
        ir.appendChild(b);
      });
      s2.appendChild(ir);
      const worst = idle[0];
      /* what is actually executing, by block overlap — not by task envelope */
      const hog = worst.spanning.filter((t) => t.blocks <= 2 && t.onCrit)[0];
      /* headline + cause; the block-by-block breakdown moves to hover */
      const ic = el('div', 'inspector-soft-card is-warning');
      ic.appendChild(el('div', 'hd', '最长一段 ' + num(worst.us, 0) + ' us，'
        + lanesOf() + ' 核只用掉 ' + pct(worst.running.capacityPct, 1)));
      ic.appendChild(el('div', 'bd', hog
        ? hog.callable + ' 单块跨越整段（span ' + num(hog.span, 0) + ' us，在依赖关键路径上）——挡住全部核。'
        : (worst.running.top.length ? '窗口内只有零星块在跑。' : '窗口内没有任何块在执行。')));
      ic.title = '占 ' + pct(worst.share, 1) + ' 的 makespan。' + NL
        + (worst.running.top.length
          ? '窗口内在跑：' + worst.running.top.map((t) => t.callable + ' ' + num(t.us, 0) + ' us/' + t.blocks + ' 块').join('、')
          : '窗口内没有任何块在执行。');
      s2.appendChild(ic);
    }
    host.appendChild(s2);

    /* --- supporting detail, folded by default --- */
    host.appendChild(s0);
    host.appendChild(renderEnginePairing(rank));
    host.appendChild(renderSpmdShape(rank));
  }

  /* --------------------------------------------- critical path, attributed
   * Ports simpler_setup.tools.critical_path. The scope ranking answers
   * "what is fat"; this answers "what did the makespan actually consist of",
   * and unlike the structural slack it can name WHY a task waited. */
  const KIND_LABEL = { 'data-wait': '数据', 'core-wait': '核', 'front-gap': '启动' };
  const KIND_FULL = {
    'data-wait': 'data-wait —— 在等上游生产者',
    'core-wait': 'core-wait —— 在等分到的核空出来（资源串行化）',
    'front-gap': 'front-gap —— 第一个任务前的 launch / dispatch 延迟',
  };
  const BOUND_LABEL = {
    dependency: '依赖受限', comm: '通信受限', resource: '资源受限',
    stall: '调度受限', compute: '计算受限',
  };

  function renderCriticalPath(rank) {
    const c = rank.cpath;
    const sec = inspectorSection('路径归责', c.segments.length + ' 节点');

    if (!c.acyclic) {
      sec.appendChild(el('div', 'inspector-soft-card is-warning',
        'happens-before 图有环，依赖关键路径无法计算；下面仍显示执行归因主链，但不要将其视为依赖下界。'));
    }

    sec.appendChild(kv([
      ['makespan', num(c.makespan, 0) + ' us'],
      ['静态 CPM', num(c.cpm.len, 0) + ' us · ' + pct(c.cpm.share, 1),
        '依赖决定的延迟下界（无限核），' + c.cpm.nodes + ' 个节点。'
        + '接近 makespan = 依赖受限，图本身就是地板。'],
      ['真正计算', num(c.workSpan, 0) + ' us · ' + pct(c.workShare, 1)],
      ['通信等待', c.waitNodes
        ? num(c.waitSpan, 0) + ' us · ' + pct(c.waitShare, 1) + '（' + c.waitNodes + ' 个 *_wait）'
        : '—'],
      ['调度 stall', num(c.stallTotal, 0) + ' us · ' + pct(c.stallShare, 1),
        '等上游 data-wait ' + num(c.stallByKind['data-wait'], 1) + ' us' + NL
        + '等核 core-wait ' + num(c.stallByKind['core-wait'], 1) + ' us' + NL
        + '启动 front-gap ' + num(c.stallByKind['front-gap'], 1) + ' us'],
      ['  数据 / 核 / 启动',
        num(c.stallByKind['data-wait'], 0) + ' / ' + num(c.stallByKind['core-wait'], 0)
        + ' / ' + num(c.stallByKind['front-gap'], 0) + ' us'],
    ]));

    /* the invariant that makes the per-task attribution sound */
    /* The verdict, with the validity gate folded into its own line: if the
     * walk did not tile the makespan the verdict is not usable at all. */
    const bd = el('div', 'inspector-soft-card' + (c.bound === 'compute' && c.tiling.exact ? '' : ' is-warning'));
    bd.appendChild(el('div', 'hd', (BOUND_LABEL[c.bound] || c.bound)
      + (c.tiling.exact ? '' : ' · 归责未闭合')));
    bd.appendChild(el('div', 'bd', c.tiling.exact
      ? c.boundWhy
      : '走查没有铺满 makespan（' + num(c.tiling.sum, 2) + ' vs ' + num(c.tiling.makespan, 2)
        + '），逐节点归责不成立，下面的数字不要引用。'));
    const gate = el('div', 'bd gate');
    gate.textContent = (c.tiling.exact ? '✓ ' : '✗ ')
      + '归责闭合 compute + stall = makespan（差 ' + num(c.tiling.delta, 2) + ' us）';
    gate.title = 'compute + stall = ' + num(c.tiling.sum, 2) + ' us vs makespan '
      + num(c.tiling.makespan, 2) + ' us。' + NL
      + '这条不成立，逐节点归责就不成立 —— 它是整段分析能不能用的前提。';
    bd.appendChild(gate);
    /* the floor check: a dependency-limited floor cannot exceed the wall
     * time it is a floor for. Nothing checked this until it was violated. */
    const cr = rank.critical;
    const floor = el('div', 'bd gate');
    floor.textContent = (cr.floorValid ? '✓ ' : '✗ ')
      + '依赖下界 CPM ≤ makespan（' + num(cr.chainSpan, 0) + ' ≤ ' + num(cr.walltime, 0) + ' us）';
    floor.title = '静态 CPM 是「无限核时依赖能压到多短」，它不可能超过实测总时长。' + NL
      + '这条曾经被违反过：未过滤时间戳的最长链算出 102.2%，把实际并行的两段时长相加了。' + NL
      + '路径上残留重叠 ' + num(cr.overlapOnPath, 2) + ' us'
      + (cr.overlapWithinTol ? '（在边保留容差内）' : '（超出容差，要查）');
    bd.appendChild(floor);
    sec.appendChild(bd);

    /* --- path nodes, worst stall first --- */
    const slow = c.segments.filter((sg) => sg.stall > 1)
      .sort((a, b) => b.stall - a.stall);
    const shown = (slow.length ? slow : c.segments.slice().sort((a, b) => b.compute - a.compute))
      .slice(0, 6);
    const rows = el('div', 'tc-scoperows');
    const hd = el('div', 'tc-scoperow is-cpath is-head');
    [['路径节点', 'l'], ['因', 'e'], ['stall', 'n'], ['span', 'n']]
      .forEach((col) => hd.appendChild(el('span', col[1], col[0])));
    rows.appendChild(hd);
    const maxStall = shown[0] ? Math.max.apply(null, shown.map((x) => x.stall)) || 1 : 1;
    shown.forEach((sg) => {
      const b = el('button', 'tc-scoperow is-cpath' + (sg.onCpm ? ' is-crit' : ''));
      b.type = 'button';
      b.title = sg.tag + ' · ' + sg.callable + NL
        + KIND_FULL[sg.kind] + NL
        + 'stall ' + num(sg.stall, 2) + ' us · 本节点 span ' + num(sg.dur, 2)
        + ' us · 非重叠计入 ' + num(sg.compute, 2) + ' us' + NL
        + (sg.onCpm ? '也在静态 CPM 路径上 —— 动它能降依赖下界'
                    : '只在执行归因主链上 —— 优化可能减少当前排程的间隙，但不能据此断言会降低依赖下界')
        + (sg.isWait ? NL + '这是 *_wait，它的 span 是等待不是计算' : '');
      const nm = el('span', 'l');
      nm.appendChild(el('i', 'bar'));
      nm.lastChild.style.width = ((sg.stall / maxStall) * 100).toFixed(1) + '%';
      nm.appendChild(el('span', 'tx', (sg.stall > 1 ? '🐌 ' : '') + sg.callable));
      b.appendChild(nm);
      b.appendChild(el('span', 'e is-' + sg.kind.split('-')[0], KIND_LABEL[sg.kind]));
      b.appendChild(el('span', 'n' + (sg.stall > 1 ? ' hot' : ' muted'), num(sg.stall, 1)));
      b.appendChild(el('span', 'n muted' + (sg.isWait ? ' is-wait' : ''), num(sg.dur, 0)));
      b.addEventListener('click', () => {
        const from = S.scopeReturn || { t0: S.t0, t1: S.t1, task: S.task, focus: S.focus };
        S.scopeReturn = { t0: from.t0, t1: from.t1, scope: null, task: from.task, focus: from.focus };
        S.task = sg.tag;
        S.focus = 'task';
        const t = tasksOf[S.rank][sg.tag];
        if (t) { const pad = Math.max(40, t.span * 0.3); setWindow(t.start - pad, t.end + pad); }
        render();
      });
      rows.appendChild(b);
    });
    sec.appendChild(rows);
    /* One line, not three cards: the marks, and the one distinction that
     * changes what a proposal is actually worth. */
    const note = el('div', 'tc-foot');
    note.appendChild(el('span', null,
      (slow.length ? '🐌 stall > 1 us（' + c.slowNodes + ' 个）' : '无 stall > 1 us')
      + ' · 红边框 = 也在静态 CPM 上'));
    if (c.cpm.onlyOnCpm && c.cpm.onlyOnCpm.length) {
      const more = el('span', 'q', '路径校验 ?');
      more.title = '静态 CPM 的 ' + c.cpm.nodes + ' 个节点里 ' + c.cpm.shared
        + ' 个也在执行归因主链上，另外 ' + c.cpm.onlyOnCpm.length + ' 个只在依赖关键路径上（'
        + c.cpm.onlyOnCpm.join('、') + '）。' + NL
        + '动只在 CPM 上的节点 → 降依赖下界。' + NL
        + '动只在执行归因主链上的节点 → 可能减少当前排程的 stall。' + NL
        + '两类证据用途不同；执行归因主链中的同核等待仍缺 task 级 ready / dispatch 记录。';
      note.appendChild(more);
    }
    const how = el('span', 'q', '怎么算的 ?');
    how.title = '依赖边按实测时间戳过滤：只有 end(前驱) ≤ start(本节点) + ' + num(c.tol, 3)
      + ' us 且 start(前驱) < start(本节点) 才保留（' + c.edgesKept + ' 留 / ' + c.edgesDropped + ' 弃）。' + NL
      + '容差' + (c.tolSource === 'clock' ? '取 2 个时钟 tick。' : '本 case 没记时钟频率，退回时间戳精度的 2 个量子。') + NL
      + 'core-wait 的前驱是同一条泳道上此前被释放的最晚时刻（running max），流水重叠的块也算得对。';
    note.appendChild(how);
    sec.appendChild(note);

    return sec;
  }

  /* ------------------------------------------------------ engine pairing
   * A general trace tool sees 72 identical blocks. It cannot say that 24 of
   * them are the Cube half and 48 the Vector half of ONE source scope, nor
   * that the two ran on paired cores. Split by FuncId and state the ratio. */
  function renderEnginePairing(rank) {
    let aicT = 0, aivT = 0;
    rank.scopes.forEach((sc) => {
      if (sc.engines.aic) aicT += sc.engines.aic.coreTime;
      if (sc.engines.aiv) aivT += sc.engines.aiv.coreTime;
    });
    const tot = Math.max(aicT + aivT, 1e-9);
    const split = rank.scopes.filter((sc) => sc.kernelCount > 1)
      .sort((a, b) => b.coreTime - a.coreTime);
    const sec = inspectorSection('引擎配对', split.length
      ? split.length + ' 个混合 scope · AIC ' + pct((aicT / tot) * 100, 0) + ' / AIV ' + pct((aivT / tot) * 100, 0)
      : 'AIC ' + pct((aicT / tot) * 100, 0) + ' / AIV ' + pct((aivT / tot) * 100, 0));

    sec.appendChild(kv([
      ['AIC (Cube)', num(aicT, 0) + ' us · ' + pct((aicT / tot) * 100, 1)
        + ' · ' + rank.occupancy.aicUtil + '% 占用'],
      ['AIV (Vec)', num(aivT, 0) + ' us · ' + pct((aivT / tot) * 100, 1)
        + ' · ' + rank.occupancy.aivUtil + '% 占用'],
      ['Cube : Vec', aicT <= aivT
        ? '1 : ' + num(aivT / Math.max(aicT, 1e-9), 2)
        : num(aicT / Math.max(aivT, 1e-9), 2) + ' : 1'],
    ]));

    if (!split.length) {
      sec.appendChild(el('div', 'inspector-soft-card',
        '本 case 没有 mixed kernel —— 每个 scope 只编译出一个 kernel，纯 Cube 或纯 Vec。'));
      return sec;
    }

    const rows = el('div', 'tc-scoperows');
    const hd = el('div', 'tc-scoperow is-pair is-head');
    [['kernel', 'l'], ['引擎', 'e'], ['core-time', 'n'], ['块/核', 'n'], ['最长块', 'n']]
      .forEach((c) => hd.appendChild(el('span', c[1], c[0])));
    rows.appendChild(hd);

    split.forEach((sc) => {
      const head = el('div', 'tc-pairhead');
      head.appendChild(el('span', 'nm', sc.name));
      head.appendChild(el('span', 'ct', num(sc.coreTime, 0) + ' us'));
      rows.appendChild(head);
      sc.kernels.forEach((k) => {
        const b = el('button', 'tc-scoperow is-pair is-sub');
        b.type = 'button';
        b.title = k.name + ' · FuncId ' + k.funcId + NL
          + k.blocks + ' 块 / ' + k.cores + ' 核 · ' + num(k.coreTime, 1) + ' us · 占本 scope ' + pct(k.share, 1);
        const nm = el('span', 'l');
        nm.appendChild(el('i', 'bar'));
        nm.lastChild.style.width = k.share.toFixed(1) + '%';
        nm.appendChild(el('span', 'tx', k.name));
        b.appendChild(nm);
        const eg = el('span', 'e is-' + k.engine, k.engine === 'aic' ? 'C' : 'V');
        b.appendChild(eg);
        b.appendChild(el('span', 'n', num(k.coreTime, 0)));
        b.appendChild(el('span', 'n muted', k.blocks + '/' + k.cores));
        b.appendChild(el('span', 'n muted', num(k.durMax, 0)));
        b.addEventListener('click', () => {
          const from = S.scopeReturn || { t0: S.t0, t1: S.t1, task: S.task, focus: S.focus };
          S.scopeReturn = { t0: from.t0, t1: from.t1, scope: sc.name, task: from.task, focus: from.focus };
          S.task = sc.tags[0];
          S.focus = 'task';
          const t = tasksOf[S.rank][sc.tags[0]];
          if (t) { const pad = Math.max(40, t.span * 0.3); setWindow(t.start - pad, t.end + pad); }
          render();
        });
        rows.appendChild(b);
      });
    });
    sec.appendChild(rows);

    const worst = split[0];
    const card = el('div', 'inspector-soft-card' + (worst.pairRatio < 1.3 ? ' is-warning' : ''));
    card.appendChild(el('div', 'hd', worst.name + ' 两侧最长块相差 ' + worst.pairRatio + ' 倍'));
    card.appendChild(el('div', 'bd', worst.pairRatio < 1.3
      ? '几乎相等 = 两半没有错开，Cube 段和 Vec 段在块内串行。解耦成 GM FIFO 才能真正并行。'
      : '差距明显 = 慢的一侧决定整块时长，快的一侧在等。'));
    card.title = num(worst.engines.aic.durMax, 1) + ' / ' + num(worst.engines.aiv.durMax, 1)
      + ' us，整段 span ' + num(worst.wall, 0) + ' us。' + NL
      + '通用 trace 工具只看到 ' + worst.spmd.blocks + ' 个同名块 —— AIC / AIV 的归属只在 '
      + 'event-hint 的 FuncId 里，而两半共用同一个 taskId。';
    sec.appendChild(card);
    return sec;
  }

  /* --------------------------------------------------------- spmd shape
   * pl.spmd(N) fans one scope out to N cores. The trace records blocks and
   * core ids but never the launch shape, so "how wide, how many waves, how
   * evenly" needs the blocks regrouped per scope. */
  function renderSpmdShape(rank) {
    const sc = rank.scopes.slice().sort((a, b) => b.spmd.cores - a.spmd.cores
      || b.coreTime - a.coreTime);
    const wide = sc.filter((x) => x.spmd.cores > 1);
    const single = sc.length - wide.length;
    const waved = sc.filter((x) => x.spmd.waves > 1);
    const sec = inspectorSection('spmd 展开',
      wide.length + ' 个多核 scope · ' + single + ' 个单核');

    sec.appendChild(kv([
      ['最宽展开', sc[0] ? sc[0].spmd.cores + ' 核（' + sc[0].name + '）' : '—'],
      ['多波 scope', waved.length + (waved.length
        ? ' · 最多 ' + num(Math.max.apply(null, waved.map((x) => x.spmd.waves)), 2) + ' 波' : '')],
      ['单核 scope', single + ' 个'],
    ]));

    const rows = el('div', 'tc-scoperows');
    const hd = el('div', 'tc-scoperow is-spmd is-head');
    [['scope', 'l', '外联后的 incore scope'],
     ['核', 'n', '最宽一次 pl.spmd 展开占了几个核'],
     ['块', 'n', '块数'],
     ['波', 'n', '块数 / 核数。1 波 = 一次填满；>1 波 = 同一批核要跑好几轮，'
       + '每轮之间有一次完成回收。'],
     ['离散', 'n', '最长块 / 中位块。>2 = 同一次展开里各块负载不均，'
       + '最慢的那块决定整个 scope 什么时候结束。']]
      .forEach((c) => { const x = el('span', c[1], c[0]); x.title = c[2]; hd.appendChild(x); });
    rows.appendChild(hd);

    /* rank by what makes a fan-out worth looking at: many waves, or uneven */
    const notable = sc.filter((x) => x.spmd.waves > 1 || x.spmd.imbalance > 1.5
      || x.spmd.widths.length > 1);
    const pick = (notable.length ? notable : sc.filter((x) => x.spmd.cores > 1))
      .slice().sort((a, b) =>
        (b.spmd.waves - 1) * b.spmd.imbalance - (a.spmd.waves - 1) * a.spmd.imbalance
        || b.spmd.imbalance - a.spmd.imbalance).slice(0, 10);
    pick.forEach((x) => {
      const b = el('button', 'tc-scoperow is-spmd');
      b.type = 'button';
      b.title = x.name + NL + x.spmd.launches + ' 次 launch · 宽度 ' + x.spmd.widths.join('/')
        + ' 核 · ' + x.spmd.blocks + ' 块 · ' + num(x.spmd.waves, 2) + ' 波' + NL
        + '离散度 ' + x.spmd.imbalance + 'x（最长块 / 中位块）';
      const nm = el('span', 'l');
      nm.appendChild(el('span', 'tx', x.name));
      b.appendChild(nm);
      b.appendChild(el('span', 'n', String(x.spmd.cores)));
      b.appendChild(el('span', 'n muted', String(x.spmd.blocks)));
      b.appendChild(el('span', 'n' + (x.spmd.waves > 1 ? ' hot' : ' muted'), num(x.spmd.waves, x.spmd.waves > 1 ? 1 : 0)));
      b.appendChild(el('span', 'n' + (x.spmd.imbalance > 2 ? ' hot' : ' muted'), num(x.spmd.imbalance, 2)));
      b.addEventListener('click', () => {
        const from = S.scopeReturn || { t0: S.t0, t1: S.t1, task: S.task, focus: S.focus };
        S.scopeReturn = { t0: from.t0, t1: from.t1, scope: x.name, task: from.task, focus: from.focus };
        S.task = x.tags[0];
        S.focus = 'task';
        const t = tasksOf[S.rank][x.tags[0]];
        if (t) { const pad = Math.max(40, t.span * 0.3); setWindow(t.start - pad, t.end + pad); }
        render();
      });
      rows.appendChild(b);
    });
    sec.appendChild(rows);
    if (!notable.length) {
      sec.appendChild(el('div', 'tc-foot',
        '没有多波、不均或变宽的展开；本 case 的形状问题是 ' + single + ' 个 scope 只用 1 个核。'));
    }
    return sec;
  }

  function renderRunInspector(host, title, meta) {
    title.textContent = D.case.program;
    meta.textContent = D.case.toolchain.platform || D.case.backend;

    const s1 = inspectorSection('运行对象', D.case.runDir.slice(0, 16) + '…');
    s1.appendChild(kv([
      ['model', D.case.model],
      ['采集时间', D.case.capturedAt],
      ['ranks', D.case.ranks.join(', ') + ' · ' + D.case.device],
      ['核', D.case.numCores + '（AIC ' + D.case.aicCount + ' / AIV ' + D.case.aivCount + '）'],
      ['kernel / scope', D.case.callables + ' / ' + scopeCountOf()],
      ['绑定参数', D.case.params.length ? String(D.case.params.length) : '未记录'],
      ['pto-isa', D.case.toolchain.ptoIsaRevision ? D.case.toolchain.ptoIsaRevision.slice(0, 12) : '—'],
      ['runtime', D.case.toolchain.runtimeName || '—'],
    ]));
    host.appendChild(s1);

    /* With four captures of the same graph, the inspector has to describe the
     * one that is armed — otherwise it silently reports tp1/prefill's numbers
     * next to another capture's stage. */
    if (QW()) {
      const v = qwVariant();
      const ql = qwL2();
      const ly = qwLayers();
      const sv = inspectorSection('本次采集', v.label + (v.validated ? ' · 已校验' : ' · 未校验'));
      sv.appendChild(kv([
        ['span', msOrUs(ql.spanUs)],
        ['任务 / 块', ql.taskCount + ' / ' + ql.blockRows + (v.hasNames ? '' : ' · 无任务名')],
        ['核', ql.coresUsed + ' / ' + ql.coreTotal + ' 用到'],
        ['平均忙核', num(ql.occ.busyCores, 2)],
        ['AIC / AIV 占用', pct(ql.occ.aicUtil) + ' / ' + pct(ql.occ.aivUtil)],
        ['层结构', ly ? ly.layerCount + ' 层 × ' + ly.perLayer + ' 任务' : '未还原'],
        ['依赖图', v.hasDeps ? '有' : '无'],
      ]));
      if (!v.primary) {
        sv.appendChild(el('div', 'inspector-soft-card is-warning',
          '本页下方的 scope / 关键路径 / 单核流水来自 tp1:prefill——'
          + '只有那一份采集带任务名和依赖图。' + v.label + ' 能回答的是占用、分层与头开销。'));
      }
      host.appendChild(sv);
    }

    if (!multiRank() || !hasE2E()) { renderRunInspectorSingle(host); return; }
    const RK = D.case.ranks;
    const s2 = inspectorSection('两卡对比', 'inv=' + TRACE_MATCH[RK[0]].inv + ' / ' + TRACE_MATCH[RK[1]].inv);
    const a = D.ranks[RK[0]], b = D.ranks[RK[1]];
    s2.appendChild(kv([
      ['device_wall', num(D.e2e.rank0[2]['chip.run.runner_run.device_wall'].us, 1) + ' / '
        + num(D.e2e.rank1[2]['chip.run.runner_run.device_wall'].us, 1) + ' us'],
      ['trace span', num(a.swimlane.spanUs, 1) + ' / ' + num(b.swimlane.spanUs, 1) + ' us'],
      ['AIC 占用', pct(a.occupancy.aicUtil) + ' / ' + pct(b.occupancy.aicUtil)],
      ['AIV 占用', pct(a.occupancy.aivUtil) + ' / ' + pct(b.occupancy.aivUtil)],
      ['依赖关键路径', a.critical.tags.length + ' / ' + b.critical.tags.length + ' 节点'],
      ['调度器占用', pct(a.scheduler.perLaneUtil) + ' / ' + pct(b.scheduler.perLaneUtil)],
    ]));
    /* the observation, then what the host clock says causes it */
    s2.appendChild(el('div', 'inspector-soft-card is-warning',
      'rank0 更慢却更闲：+' + num(a.swimlane.spanUs - b.swimlane.spanUs, 0) + ' us span，'
      + '−' + num(b.occupancy.aicUtil - a.occupancy.aicUtil, 1) + ' pt AIC 占用'));
    const K = D.launchSkew;
    if (K) {
      const cause = el('div', 'inspector-soft-card');
      cause.appendChild(el('span', 'hd', 'rank1 晚发 ' + num(K.runnerUs, 1) + ' us'));
      cause.appendChild(el('span', 'bd', 'AIC busy ' + num(K.work.rank0.aic.busy, 0) + ' / '
        + num(K.work.rank1.aic.busy, 0) + ' us（差 '
        + pct(Math.abs(K.work.rank0.aic.busy - K.work.rank1.aic.busy) / K.work.rank1.aic.busy * 100, 2)
        + '）——计算量相同，多出来的 span 是等待'));
      const rows = el('div', 'tc-skewrows');
      const hd = el('div', 'tc-skewrow is-head');
      ['wait', 'rank0 等', '错峰上界', '占比'].forEach((t, i) => hd.appendChild(el('span', i ? 'n' : 'l', t)));
      rows.appendChild(hd);
      K.checks.forEach((c) => {
        const r = el('button', 'tc-skewrow');
        r.type = 'button';
        r.appendChild(el('span', 'l', c.callable.replace(/_wait$/, '')));
        r.appendChild(el('span', 'n', num(c.measured, 1)));
        r.appendChild(el('span', 'n muted', num(c.bound, 1)));
        r.appendChild(el('span', 'n' + (c.fitPct >= 80 ? ' hot' : ''), pct(c.fitPct, 0)));
        r.addEventListener('click', () => {
          S.rank = 'rank0'; S.view = 'l2'; S.task = c.tag0; S.focus = 'task';
          const t = tasksOf.rank0[c.tag0];
          if (t) { const pad = Math.max(40, t.span * 0.35); setWindow(t.start - pad, t.end + pad); }
          render();
        });
        rows.appendChild(r);
      });
      cause.appendChild(rows);
      cause.appendChild(el('span', 'bd', (K.allUnderBound ? '4 个 wait 全部落在上界内' : '有 wait 超出上界')
        + ' · 合计 ' + num(K.measuredSum, 0) + ' / ' + num(K.boundSum, 0) + ' us'));
      s2.appendChild(cause);
    }
    host.appendChild(s2);

    const s3 = inspectorSection('瓶颈链', topChains(3).length + ' 条');
    topChains(3).forEach((f) => {
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

    const guidedFinding = guidedJourneyActive() ? activeFinding() : null;
    if (guidedFinding && guidedFinding.focus && guidedFinding.focus.task === t.tag) {
      const path = pathRole(rank, t.tag);
      const summary = inspectorSection('任务摘要', pathLabel(path) + ' · ' + t.kind.toUpperCase());
      summary.appendChild(kv([
        ['执行区间', num(t.start, 1) + ' → ' + num(t.end, 1) + ' us · span ' + num(t.span, 2) + ' us'],
        ['块 / 核', t.blockCount + ' / ' + t.coreCount],
        ['块时长 min / med / p90 / max', num(t.durMin, 2) + ' / ' + num(t.durMed, 2) + ' / '
          + num(t.durP90, 2) + ' / ' + num(t.durMax, 2) + ' us'],
        ['依赖', t.pred.length + ' 前驱 · ' + t.succ.length + ' 后继'],
        ['源码', t.src ? srcLabel(t.src) : (D.sourceMap ? '未匹配' : '源码树不在仓库内')],
      ]));
      host.appendChild(summary);
      renderGuidedBudget(host, guidedFinding);
      host.appendChild(el('p', 'tc-fineprint',
        'L1 PMU 与片上预算提示用于提出验证方向；Pass 深度回退是否造成这段 L2 span 增长，仍需单变量重编译与复测确认。'));
      return;
    }

    const s1 = inspectorSection('对象', t.kind.toUpperCase());
    s1.appendChild(kv([
      ['task id', t.id],
      /* a split scope has no single funcId -- name every kernel's */
      ['scope', t.callable + '（funcId ' + (t.kernels || []).map((k) => k.funcId).join(' + ') + '）'],
      ['ring / scope', 'r' + t.ring + ' · ' + t.scope + (t.earlyDispatch ? ' · early_dispatch' : '')],
      ['窗口', num(t.start, 1) + ' → ' + num(t.end, 1) + ' us'],
      ['块 / 核', t.blockCount + ' / ' + t.coreCount + '（block_num ' + t.blockNum + '）'],
      ['块时长', num(t.durMin, 2) + ' / ' + num(t.durMed, 2) + ' / ' + num(t.durP90, 2) + ' / ' + num(t.durMax, 2) + ' us'],
      ['kernel · setup', num(t.kdurSum / t.blockCount, 2) + ' · ' + num(t.setupMean, 2) + ' us'],
      ['AICPU 视角', t.svAicpuMean == null ? null : num(t.svAicpuMean, 1) + ' us（+' + num(t.svOverhead, 1) + '）'],
      ['前驱 / 后继', t.pred.length + ' / ' + t.succ.length],
      (function () { const r = pathRole(rank, t.tag); return ['路径', pathLabel(r), pathTitle(r)]; })(),
      ['kernel', (t.kernels || []).map((k) => k.name).join(' + ') || t.callable],
      ['源码', t.src ? srcLabel(t.src) : (D.sourceMap ? '未匹配' : '源码树不在仓库内')],
    ]));
    /* one scope, two kernels: state the split instead of one merged number */
    if (t.kernelCount > 1 && t.engines.aic && t.engines.aiv) {
      s1.appendChild(el('div', 'inspector-soft-card',
        '这是一个 mixed scope：ExpandMixedKernel 拆成 ' + t.kernelCount
        + ' 个 kernel，共用一次 Group launch。'
        + 'Cube 侧 ' + t.engines.aic.blocks + ' 块 / ' + num(t.engines.aic.coreTime, 0)
        + ' us（最长 ' + num(t.engines.aic.durMax, 1) + '），'
        + 'Vec 侧 ' + t.engines.aiv.blocks + ' 块 / ' + num(t.engines.aiv.coreTime, 0)
        + ' us（最长 ' + num(t.engines.aiv.durMax, 1) + '）。'
        + '上面「块 / 核」是两半合计。'));
    }
    if (t.src && (t.src.candidates > 1 || !t.src.exact)) {
      s1.appendChild(el('div', 'inspector-soft-card',
        (t.src.exact ? '' : 'callable 去掉编译器后缀后按 name_hint="' + t.src.hint + '" 匹配。')
        + (t.src.candidates > 1
          ? t.src.candidates + ' 处同名候选，名字本身消不掉歧义：'
            + t.src.sites.map((x) => x.file + ':' + x.line).join('、')
          : '')));
    }
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
      if (t.args.length > 8) s2.appendChild(el('p', 'tc-note', '另有 ' + (t.args.length - 8) + ' 个'));
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

    /* a task belongs to a chain if ANY rung marks it, so walking down from
     * L2 to a compiler site still shows the task its chain came from */
    const rungsFor = (f) => (f.chain || []).filter((st) => st.subjects.tasks.indexOf(t.tag) >= 0);
    const rel = D.findings.filter((f) => (f.focus && f.focus.task === t.tag)
      || f.subjects.tasks.indexOf(t.tag) >= 0 || rungsFor(f).length);
    if (rel.length) {
      const s4 = inspectorSection('关联瓶颈', rel.length + ' 条');
      rel.forEach((f) => {
        const at = rungsFor(f).map((st) => LEVEL_LABEL[st.level] || st.level);
        s4.appendChild(btn(f.id + ' · ' + f.title
          + (at.length ? '（' + Array.from(new Set(at)).join(' / ') + '）' : ''), {
          size: 'sm', on: () => { S.finding = f.id; S.chainStep = null; S.focus = 'finding'; applyFocus(f); render(); },
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

  function guidedDepthSite(f) {
    const root = (f.chain || []).find((st) => st.level === 'compiler');
    const key = root && root.subjects && root.subjects.sites && root.subjects.sites[0];
    return D.depthSites.find((site) => site.key === key) || null;
  }

  function renderGuidedBudget(host, f) {
    const site = guidedDepthSite(f);
    if (!site) return;
    const section = inspectorSection('L1/L0 · 片上预算试算',
      site.units.join('/') + ' · ' + site.file + ':' + site.line);
    const controls = el('div', 'tc-journey-budget-controls');
    const numberField = (label, key, min, max, step) => {
      const wrap = el('label', 'tc-journey-budget-field');
      wrap.appendChild(el('span', null, label));
      const input = el('input');
      input.type = 'number'; input.min = min; input.max = max; input.step = step || 1;
      input.value = S.tile[key];
      input.addEventListener('change', () => {
        S.tile[key] = clamp(parseInt(input.value, 10) || min, min, max);
        render();
      });
      wrap.appendChild(input);
      return wrap;
    };
    controls.appendChild(numberField('K', 'k', 16, 1024, 16));
    controls.appendChild(numberField('N', 'n', 16, 1024, 16));
    const depthWrap = el('label', 'tc-journey-budget-field');
    depthWrap.appendChild(el('span', null, 'stage'));
    depthWrap.appendChild(select([1, 2, 3, 4].map((depth) => ({ id: String(depth), label: String(depth) })),
      String(S.guidedDepth), (value) => { S.guidedDepth = +value; render(); }));
    controls.appendChild(depthWrap);
    section.appendChild(controls);

    const rightBytes = S.tile.k * S.tile.n * dt(S.tile.ab).bytes;
    const stageBytes = rightBytes * S.guidedDepth;
    const ratio = stageBytes / site.freeB;
    const status = ratio >= 1 ? 'warn' : 'good';
    const budget = el('div', 'tc-budget');
    const row = el('div', 'tc-budget-row');
    row.appendChild(el('span', 'nm', 'Right × stage'));
    row.appendChild(bar(ratio, status === 'good' ? 'good' : 'warn'));
    row.appendChild(el('span', 'fx', kb(stageBytes) + ' / ' + kb(site.freeB)));
    budget.appendChild(row);
    section.appendChild(budget);

    const fitted = el('div', 'tc-verdict-row');
    fitted.dataset.state = status;
    fitted.appendChild(el('span', 'tag', 'PH-MR-001'));
    fitted.appendChild(el('p', null, '本次实测：' + site.groupCount + ' 组请求 depth '
      + site.maxReqDepth + '，MemoryReuse 拟合为 ' + site.fittedDepth
      + '。当前试算 ' + kb(stageBytes) + ' / ' + kb(site.freeB)
      + (ratio >= 1 ? '，已到预算边界；不能据此假定可容纳同驻 tile。' : '，预算内，但仍需重编译确认深度。')));
    section.appendChild(fitted);
    section.appendChild(el('small', 'tc-fineprint',
      '试算只估单组 Right tile × stage；同驻组共享空间与 Pass 拟合结果以 PH-MR-001 为准。'));
    host.appendChild(section);
  }

  function renderGuidedRootEvidence(host, f) {
    const site = guidedDepthSite(f);
    if (!site) return;
    const root = inspectorSection('根因线索 · MemoryReuse', 'PH-MR-001 · 实测编译提示');
    root.appendChild(kv([
      ['源码点', site.file + ':' + site.line],
      ['片上空间', site.units.join('/')],
      ['请求 / 拟合', 'depth ' + site.maxReqDepth + ' → ' + site.fittedDepth],
      ['单级占用', kb(site.perStageB)],
      ['可用预算', kb(site.freeB)],
      ['同驻组', String(site.groupCount)],
    ]));
    root.appendChild(el('p', 'tc-note',
      '该提示与 L2 选中的 kernel 源码范围相邻，说明 MemoryReuse 确实回退了流水深度。它提供了可验证的编译器线索；是否解释了关键任务 span，仍要通过单变量重编译和复测确认。'));
    host.appendChild(root);
  }

  function renderGuidedPassProcess(stage) {
    const f = activeFinding();
    if (!f || !f.rootPass) return;
    const chain = [
      { name: 'AutoTileMatmulL0', label: 'Tile / stage 请求', note: '产生 L0 tile 与流水深度请求。' },
      { name: 'InferTileMemorySpace', label: '片上空间推导', note: '推导 Left / Right / Acc 的目标空间。' },
      { name: 'MemoryReuse', label: '预算拟合', note: '根据同驻 tile 占用拟合可用流水深度。' },
      { name: 'AllocateMemoryAddr', label: '物理地址分配', note: '为存活的片上 buffer 分配地址。' },
    ].map((entry) => Object.assign({}, entry, { pass: D.passes.find((p) => p.name === entry.name) }))
      .filter((entry) => entry.pass);
    const section = el('section', 'tc-guided-pass-process');
    section.appendChild(sectionHead('相关 Pass 作业过程',
      f.focus.task + ' · 展示相关节点，省略无关 Pass · 高亮节点对应右侧 PH-MR-001'));
    const flow = el('div', 'tc-guided-pass-flow');
    chain.forEach((entry, i) => {
      if (i) flow.appendChild(el('span', 'tc-guided-pass-arrow', '→'));
      const selected = S.guidedPass === entry.name;
      const affected = entry.name === f.rootPass;
      const node = btn('', {
        variant: 'ghost',
        selected,
        on: () => { S.guidedPass = entry.name; S.focus = 'task'; render(); },
      });
      node.classList.add('tc-guided-pass-node');
      if (affected) node.classList.add('is-affected');
      node.appendChild(el('span', 'idx', 'Pass ' + entry.pass.idx));
      node.appendChild(el('strong', null, entry.name));
      node.appendChild(el('span', 'lbl', entry.label));
      flow.appendChild(node);
    });
    section.appendChild(flow);

    const selected = chain.find((entry) => entry.name === S.guidedPass) || chain.find((entry) => entry.name === f.rootPass);
    if (selected) {
      const detail = el('div', 'tc-guided-pass-detail' + (selected.name === f.rootPass ? ' is-root' : ''));
      detail.appendChild(el('strong', null, 'Pass ' + selected.pass.idx + ' · ' + selected.name));
      detail.appendChild(el('span', null, selected.note));
      const passDiff = (D.passEvidence || []).find((item) => item.idx === selected.pass.idx);
      if (passDiff) {
        const scopes = (passDiff.scopes || []).map((scope) => scope.name);
        detail.appendChild(el('small', null, 'IR 快照差异 +' + passDiff.add + ' / −' + passDiff.del
          + ' · ' + passDiff.groups + ' 个改写区域'
          + (scopes.length ? ' · 报告涉及 scope：' + scopes.join('、') : '')));
      }
      if (selected.name === f.rootPass) {
        const site = guidedDepthSite(f);
        if (site) {
          const operation = el('div', 'tc-guided-pass-operation');
          operation.appendChild(el('strong', null, '异常作业 · PH-MR-001'));
          const budgetMap = el('div', 'tc-pass-budget-map');
          const mapHead = el('div', 'tc-pass-budget-head');
          mapHead.appendChild(el('span', null, '同驻组'));
          mapHead.appendChild(el('span', null, 'Right stage 请求 · 每块 ' + kb(site.perStageB)));
          mapHead.appendChild(el('span', null, '拟合结果'));
          budgetMap.appendChild(mapHead);
          for (let group = 0; group < site.groupCount; group++) {
            const row = el('div', 'tc-pass-budget-row');
            row.appendChild(el('span', 'group', 'G' + group));
            const track = el('div', 'tc-pass-budget-track');
            track.setAttribute('aria-label', '请求 depth ' + site.maxReqDepth + '，可用空间 ' + kb(site.freeB));
            for (let depth = 0; depth < site.maxReqDepth; depth++) {
              const tile = el('span', 'tile' + (depth >= site.fittedDepth ? ' is-dropped' : ''));
              tile.appendChild(el('i', null, String(depth + 1)));
              tile.appendChild(el('b', null, kb(site.perStageB)));
              track.appendChild(tile);
            }
            track.appendChild(el('span', 'budget-marker', kb(site.freeB) + ' 可用'));
            row.appendChild(track);
            const outcome = el('span', 'fit');
            outcome.appendChild(el('strong', null, 'depth ' + site.fittedDepth));
            outcome.appendChild(el('small', null, '请求 ' + site.maxReqDepth));
            row.appendChild(outcome);
            budgetMap.appendChild(row);
          }
          operation.appendChild(budgetMap);
          const flowCaption = el('div', 'tc-pass-budget-caption');
          flowCaption.appendChild(el('span', null, 'Tile 请求'));
          flowCaption.appendChild(el('i', null, '→'));
          flowCaption.appendChild(el('span', null, 'MemoryReuse 比对片上预算'));
          flowCaption.appendChild(el('i', null, '→'));
          flowCaption.appendChild(el('strong', null, '流水深度回退'));
          operation.appendChild(flowCaption);
          operation.appendChild(el('small', null, site.file + ':' + site.line
            + ' · 单级占用 × 请求深度 = ' + kb(site.perStageB) + ' × ' + site.maxReqDepth
            + ' = ' + kb(site.perStageB * site.maxReqDepth) + '，可用 ' + kb(site.freeB)
            + '。预算等式本身不能解释为什么拟合为 depth 1，需结合 MemoryReuse 的共享分配状态复核。'));
          detail.appendChild(operation);
        }
      }
      section.appendChild(detail);
    }
    stage.appendChild(section);
  }

  function renderFindingInspector(host, title, meta) {
    const f = findingById[S.finding];
    const chain = f.chain || [];
    const guided = guidedJourneyActive();
    const currentStep = guided ? (activeStep() || chain[0]) : null;
    const currentIndex = guided ? chain.indexOf(currentStep) : -1;
    title.textContent = f.id + ' · ' + (f.kind === 'hygiene' ? '体检项' : '瓶颈链');
    meta.textContent = guided
      ? '分析 ' + (currentIndex + 1) + ' / ' + chain.length
      : (f.cost ? f.cost.share + '% of makespan' : '无归因');

    const s1 = inspectorSection(f.title, f.metric);
    if (f.cost) {
      s1.appendChild(el('div', 'inspector-soft-card is-info',
        '代价 ' + us(f.cost.us, 1) + '（makespan 的 ' + f.cost.share + '%）· 口径：' + f.cost.basis));
    } else if (f.unattributed) {
      s1.appendChild(el('div', 'inspector-soft-card is-warning', '不作为瓶颈：' + f.unattributed));
    }
    s1.appendChild(el('p', 'tc-note', f.claim));
    host.appendChild(s1);

    if (guided && currentStep) {
      const current = inspectorSection('当前分析',
        (LEVEL_LABEL[currentStep.level] || currentStep.level) + ' · ' + (ROLE[currentStep.role] || ROLE.observe).label);
      current.appendChild(el('strong', null, currentStep.headline));
      if (currentStep.detail) current.appendChild(el('p', 'tc-note', currentStep.detail));
      host.appendChild(current);
      if (currentStep.level === 'l1') renderGuidedBudget(host, f);
      if (currentStep.level === 'compiler') renderGuidedRootEvidence(host, f);
    } else if (chain.length) {
      const s0 = inspectorSection('跨层链条',
        chain.map((st) => LEVEL_LABEL[st.level] || st.level).join(' → '));
      const lad = el('div', 'tc-ladder');
      chain.forEach((st, i) => {
        const key = f.id + ':' + i;
        const r = el('div', 'tc-ladder-step' + (S.chainStep === key ? ' is-selected' : ''));
        r.dataset.role = st.role;
        const hd = el('div', 'hd');
        hd.appendChild(el('span', 'lv', LEVEL_LABEL[st.level] || st.level));
        hd.appendChild(el('span', 'role', (ROLE[st.role] || { label: st.role }).label));
        r.appendChild(hd);
        r.appendChild(el('span', 'ti', st.headline));
        if (st.detail) r.appendChild(el('span', 'dt', st.detail));
        /* only a rung that names something on the stage is clickable; a stop
         * rung has nothing to jump to and must not pretend otherwise */
        if ((st.chips.length || st.role !== 'stop') && !guidedJourneyActive()) {
          r.tabIndex = 0;
          r.setAttribute('role', 'button');
          r.classList.add('is-linked');
          const go = () => { applyStep(f, st); render(); };
          r.addEventListener('click', go);
          r.addEventListener('keydown', (ev) => {
            if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); go(); }
          });
        }
        lad.appendChild(r);
      });
      s0.appendChild(lad);
      if (f.terminus) {
        s0.appendChild(el('div', 'inspector-soft-card'
          + (f.terminus.level === 'compiler' ? ' is-info' : ' is-warning'),
          '链止于 ' + (LEVEL_LABEL[f.terminus.level] || f.terminus.level) + '：' + f.terminus.reason));
      }
      host.appendChild(s0);
    }

    const shownEvidence = guided && currentStep ? currentStep.evidence : f.evidence;
    const s2 = inspectorSection('证据', shownEvidence.length + ' 项'
      + (guided ? ' · 当前分析阶段' : (f.chips.length ? ' · 已在中间标号' : '')));
    const list = el('div', 'tc-evidence');
    shownEvidence.forEach((e) => {
      /* link an evidence row to the marked objects its locator names, so the
       * inspector text and the numbered markers on the stage are the same thing */
      const keysOf = (c) => {
        const keys = [c.id, c.label];
        if (c.kind === 'site') keys.push(c.id.split(':')[0]);   /* file without line */
        return keys.filter(Boolean);
      };
      const linked = f.chips.filter((c) => keysOf(c)
        .some((k) => e.locator.indexOf(k) >= 0 || e.value.indexOf(k) >= 0));
      const r = el('div', 'tc-evidence-row');
      const a = el('span', 'a', e.artifact);
      if (linked.length && !guidedJourneyActive()) {
        a.appendChild(document.createTextNode(' · '));
        a.appendChild(el('span', 'jump', '标号 ' + linked.map((c) => f.chips.indexOf(c) + 1).join(' / ')));
        r.classList.add('is-linked');
        r.tabIndex = 0;
        r.setAttribute('role', 'button');
        const go = () => gotoChip(linked[0]);
        r.addEventListener('click', go);
        r.addEventListener('keydown', (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); go(); } });
      }
      r.appendChild(a);
      r.appendChild(el('span', 'l', e.locator));
      r.appendChild(el('span', 'v', e.value));
      list.appendChild(r);
    });
    s2.appendChild(list);
    host.appendChild(s2);

    if (guided) {
      const nav = el('div', 'tc-guided-nav');
      if (currentIndex > 0) {
        nav.appendChild(btn('上一步', {
          variant: 'ghost',
          on: () => { applyStep(f, chain[currentIndex - 1]); render(); },
        }));
      }
      if (currentIndex < chain.length - 1) {
        const next = chain[currentIndex + 1];
        nav.appendChild(btn('继续分析：' + (LEVEL_LABEL[next.level] || next.level), {
          variant: 'solid',
          on: () => { applyStep(f, next); render(); },
        }));
      } else {
        nav.appendChild(el('span', 'tc-journey-done', '已定位到 ' + (f.rootPass || '编译器提示')));
      }
      host.appendChild(nav);
    }

    if (!guided || currentIndex === chain.length - 1) {
      const s3 = inspectorSection('杠杆与护栏');
      s3.appendChild(el('div', 'inspector-soft-card is-info', '杠杆：' + f.lever));
      s3.appendChild(el('div', 'inspector-soft-card is-warning', '护栏：' + f.guardrail));
      s3.appendChild(el('div', 'inspector-soft-card', '复测：' + f.verify));
      host.appendChild(s3);
    }

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

    const s2 = inspectorSection('处理');
    s2.appendChild(el('div', 'inspector-soft-card is-info', depth
      ? '减少同驻 tile，而非调大 stage'
      : '末维凑满一个 cache line'));
    s2.appendChild(el('div', 'inspector-soft-card is-warning', depth
      ? '调大 stage 会再触发一次回退'
      : '加大末维会抬高 L0 / UB 占用，可能触发深度回退'));
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
      else s2.appendChild(el('p', 'tc-note', '结构计数无变化 · 行数 ' + (p.delta > 0 ? '+' : '') + p.delta));
      host.appendChild(s2);
    }
  }

  /* ------------------------------------------------------- experiment */
  function renderComposer(f) {
    const open = openExperiment();
    const s = inspectorSection('实验台账', open ? '已有 1 个进行中' : '每轮只验证一个假设');
    if (open && open.findingId !== f.id) {
      s.appendChild(el('div', 'inspector-soft-card is-warning',
        open.id + ' 进行中 · 结论后才能开下一个'));
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
    l1.appendChild(el('span', null, '假设'));
    l1.appendChild(hyp);
    const l2 = el('label');
    l2.appendChild(el('span', null, '改动'));
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
    steps.appendChild(step('正确性', row.correctness || '待记录', !!row.correctness));
    steps.appendChild(step('性能', row.perf || row.verify, !!row.perf));
    steps.appendChild(step('结论', row.keep || '待决定', !!row.keep));
    wrap.appendChild(steps);

    const acts = el('div', 'tc-actions');
    if (!row.correctness) {
      const inp = el('input');
      inp.type = 'text';
      inp.placeholder = '精度阈值 / 对比基准';
      const lb = el('label');
      lb.appendChild(el('span', null, '正确性'));
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
      lb.appendChild(el('span', null, '性能'));
      lb.appendChild(inp);
      wrap.appendChild(lb);
      acts.appendChild(btn('记录性能', {
        size: 'sm', variant: 'solid',
        on: () => { row.perf = inp.value.trim() || '（未填写复测数值）'; render(); },
      }));
    } else {
      const guard = el('div', 'inspector-soft-card is-warning');
      guard.textContent = row.guardrail;
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

  /* ========================================= E2E cross-rank scheduler */
  function renderE2ECrossRankScheduler(stage) {
    const ranks = D.case.ranks.filter((rank) => D.ranks[rank] && TRACE_MATCH[rank]);
    const skew = D.launchSkew;
    const base = ranks[0];
    const sec = el('section');
    sec.appendChild(sectionHead('跨 rank 调度与 ready queue',
      'E2E 对照 · 点行进入该 rank 的 L2 工作台'));

    const waitSum = skew ? skew.measuredSum : 0;
    sec.appendChild(tiles([
      { k: 'rank 启动偏移', v: skew ? num(skew.runnerUs, 1) : '—', u: 'us', tone: skew ? 'warn' : null },
      { k: '集合点等待', v: skew ? num(waitSum, 1) : '—', u: 'us', tone: skew ? 'warn' : null },
      { k: '等待 / 偏移', v: skew && skew.runnerUs ? num(waitSum / skew.runnerUs, 2) : '—', u: '×' },
      { k: '调度线程', v: ranks.map((rank) => D.ranks[rank].scheduler.lanes.length).join(' / '), u: ranks.join(' / ') },
    ]));

    const rows = ranks.map((rank) => {
      const data = D.ranks[rank];
      const isBase = rank === base;
      return {
        rank: rank,
        launch: isBase ? '基线' : (skew ? '+' + num(skew.runnerUs, 1) + ' us' : '—'),
        trace: num(data.swimlane.spanUs, 1),
        sched: pct(data.scheduler.perLaneUtil, 1),
        ready: 'avg ' + num(data.readyStat.avg.AIC, 3) + ' / peak ' + data.readyStat.peak.AIC,
        readyShare: pct(data.readyStat.busyShare.AIC, 1),
        aic: pct(data.occupancy.aicUtil, 1),
        __selected: rank === S.rank,
      };
    });
    sec.appendChild(table([
      { label: 'rank', key: 'rank', mono: true },
      { label: 'runner_run 启动', key: 'launch', num: true },
      { label: 'trace span', num: true, cell: (row) => row.trace + ' us' },
      { label: '调度线程占用', key: 'sched', num: true },
      { label: 'AIC ready', key: 'ready', num: true },
      { label: 'ready>0', key: 'readyShare', num: true },
      { label: 'AIC 占用', key: 'aic', num: true },
    ], rows, {
      onPick: (row) => { selectAnalysisRank(row.rank, 'l2'); S.focus = null; render(); },
    }));

    const note = el('div', 'inspector-soft-card' + (skew ? ' is-warning' : ''));
    note.textContent = skew
      ? 'C1 在此闭合：L2 的 ready / 调度读数用于排除片内调度饱和；根因仍由两卡 Host runner_run 启动偏移确认。'
      : '本 run 缺少可对齐的跨 rank Host 时间戳；只能比较两卡的设备侧调度读数。';
    sec.appendChild(note);
    stage.appendChild(sec);
  }

  function renderDock() {
    const body = $('#dockBody');
    body.textContent = '';
    if (body.__ro) { body.__ro.disconnect(); body.__ro = null; }
    /* the ready-queue tooltip lives outside #dockBody, so clear it by hand */
    const staleTip = document.querySelector('[data-tc-tip="readyq"]');
    if (staleTip) staleTip.remove();
    if (S.view === 'e2e' && multiRank() && hasE2E()) {
      renderE2ECrossRankDock(body);
      return;
    }
    const title = $('#dockTitle');
    if (title) title.textContent = 'Scheduler & ready queue';
    $('#dockMode').hidden = false;
    const rank = R();
    /* The dock is built from the one capture that has a merged swimlane. With
     * another capture armed it would otherwise look like that capture's data,
     * so it names its source instead. */
    $('[data-bind="dockMeta"]').textContent = (QW() && !onPrimaryVariant())
      ? 'tp1:prefill 的调度器（' + qwVariant().label + ' 无 merged swimlane，见「核占用」页）'
      : S.rank + ' · 与上方时间轴同窗口 ' + num(S.t0, 0) + '–' + num(S.t1, 0) + ' us';
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
      return;
    }

    /* ready queue
     * shared_ready_queue is a Chrome-trace counter (ph "C"): each sample says
     * how many tasks are dependency-satisfied but not yet dispatched, split by
     * engine. A counter holds its value until the next sample, so it is drawn
     * as a step — the same semantics build-data.cjs integrates busyTime with. */
    const SERIES = [
      { key: 'AIC', idx: 1, token: '--danger' },
      { key: 'AIV', idx: 2, token: '--warning' },
      { key: 'MIX', idx: 3, token: '--accent' },
    ];
    const legend = el('div', 'tc-legend');
    SERIES.forEach((sr) => {
      const item = el('span');
      const swatch = el('i');
      swatch.style.background = cssVar(sr.token);
      item.appendChild(swatch);
      item.appendChild(el('span', null, sr.key + ' ready · peak ' + rank.readyStat.peak[sr.key]));
      legend.appendChild(item);
    });
    legend.appendChild(el('span', 'tc-readout', rank.readyQueue.length + ' 个采样点 · 悬停读数'));
    body.appendChild(legend);

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

    const q = rank.readyQueue;
    /* last sample whose timestamp is <= t */
    const sampleAt = (t) => {
      let lo = 0, hi = q.length - 1, hit = -1;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        if (q[mid][0] <= t) { hit = mid; lo = mid + 1; } else { hi = mid - 1; }
      }
      return hit;
    };

    let hoverX = null;   /* pointer position, canvas css px */
    let geom = null;     /* written by draw(), read by the pointer handler */

    const draw = () => {
      const w = host.clientWidth || 700;
      const h = 92;
      const ctx = fitCanvas(canvas, w, h);
      const x0 = 46, plotW = Math.max(40, w - x0 - 12), y0 = 18, hh = h - 40;
      geom = { x0: x0, plotW: plotW, y0: y0, hh: hh };
      drawTimeRuler(ctx, x0, plotW, 10, S.t0, S.t1);
      const peak = Math.max(rank.readyStat.peak.AIC, rank.readyStat.peak.AIV, 1);
      const sx = (t) => x0 + ((t - S.t0) / (S.t1 - S.t0)) * plotW;
      const sy = (v) => y0 + hh - (v / peak) * hh;
      SERIES.forEach((sr) => {
        ctx.beginPath();
        ctx.moveTo(x0, y0 + hh);
        let prevY = y0 + hh;
        q.forEach((sample) => {
          const x = clamp(sx(sample[0]), x0, x0 + plotW);
          const y = sy(sample[sr.idx]);
          ctx.lineTo(x, prevY);   /* hold the previous value up to this sample */
          ctx.lineTo(x, y);       /* then step */
          prevY = y;
        });
        ctx.lineTo(x0 + plotW, prevY);
        ctx.lineTo(x0 + plotW, y0 + hh);
        ctx.closePath();
        ctx.fillStyle = cssVar(sr.token);
        ctx.globalAlpha = 0.3;
        ctx.fill();
        ctx.globalAlpha = 0.9;
        ctx.strokeStyle = cssVar(sr.token);
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

      /* crosshair: the read head the tooltip is reporting */
      if (hoverX != null) {
        const hx = clamp(hoverX, x0, x0 + plotW);
        const i = sampleAt(S.t0 + ((hx - x0) / plotW) * (S.t1 - S.t0));
        ctx.save();
        ctx.strokeStyle = cssVar('--foreground-secondary');
        ctx.globalAlpha = 0.55;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(Math.round(hx) + 0.5, y0 - 6);
        ctx.lineTo(Math.round(hx) + 0.5, y0 + hh);
        ctx.stroke();
        ctx.restore();
        if (i >= 0) {
          SERIES.forEach((sr) => {
            ctx.beginPath();
            ctx.arc(hx, sy(q[i][sr.idx]), 2.5, 0, Math.PI * 2);
            ctx.fillStyle = cssVar(sr.token);
            ctx.fill();
          });
        }
      }
    };

    /* the shared pattern owns the tooltip chrome; only the rows are ours */
    const tipRow = (k, v, cls) => '<div class="pto-swimlane-task-tooltip__row">'
      + '<span class="pto-swimlane-task-tooltip__key">' + esc(k) + '</span>'
      + '<span class="pto-swimlane-task-tooltip__value' + (cls ? ' ' + cls : '') + '">'
      + esc(v) + '</span></div>';

    /* the tooltip lives on the frame layer: .tc-canvas-strip clips to its
     * rounded corners, and the strip is only 92px tall. */
    const tipLayer = document.querySelector('.tc-frame');
    const tip = SW.createTooltip();
    tip.dataset.tcTip = 'readyq';
    tipLayer.appendChild(tip);

    canvas.addEventListener('pointermove', (event) => {
      if (!geom) return;
      hoverX = event.clientX - canvas.getBoundingClientRect().left;
      draw();
      const hx = clamp(hoverX, geom.x0, geom.x0 + geom.plotW);
      const t = S.t0 + ((hx - geom.x0) / geom.plotW) * (S.t1 - S.t0);
      const i = sampleAt(t);
      if (i < 0) { SW.hideTooltip(tip); return; }
      const held = (i + 1 < q.length ? q[i + 1][0] : q[q.length - 1][0]) - q[i][0];
      const total = q[i][1] + q[i][2] + q[i][3];
      const html = '<div class="pto-swimlane-task-tooltip__title">t = ' + num(t, 1) + ' us</div>'
        + tipRow('采样', num(q[i][0], 2) + ' us · #' + (i + 1) + '/' + q.length)
        + tipRow('保持', num(held, 2) + ' us')
        + tipRow('AIC ready', String(q[i][1]), q[i][1] > 0 ? 'is-warn' : '')
        + tipRow('AIV ready', String(q[i][2]))
        + tipRow('MIX ready', String(q[i][3]))
        + tipRow('待派发合计', String(total));
      SW.showTooltip(tip, { counterReadout: true }, event,
        { bounds: tipLayer, target: canvas, getTooltipHtml: () => html });
    });
    canvas.addEventListener('pointerleave', () => {
      hoverX = null;
      draw();
      SW.hideTooltip(tip);
    });

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
      Object.keys(D.e2e || {}).forEach((rank) => {
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

    /* Artifact inventory for the active case: present rows carry what they
     * answer, absent rows say so instead of being dropped silently. */
    const A = D.case.artifacts;
    const rankDir = (r) => 'dfx_outputs/' + (multiRank() ? r + '/d0/' : '');
    const rows = [
      ['distributed_meta.json', A.distributedMeta
        ? D.case.params.length + ' 个绑定参数，schema ' + D.case.metaSchema : '缺失'],
    ].concat(D.case.ranks.map((r) => [
      rankDir(r) + 'merged_swimlane_*.json',
      'Worker + Scheduler view，' + D.ranks[r].tasks.length + ' 任务 / '
        + D.ranks[r].swimlane.blocks.reduce((a, b) => a + b.length, 0) + ' 块',
    ])).concat([
      [rankDir(D.defaultRank) + 'deps.json', 'scope / 绑定张量'
        + (D.ranks[D.defaultRank].tasks[0].blockNum != null ? ' / block_num / early_dispatch' : '')],
      [rankDir(D.defaultRank) + 'name_map.json', D.case.callables + ' 个 kernel 名（level '
        + D.case.level + '）—— 对应 ' + scopeCountOf() + ' 个 scope，混合 scope 各拆两个'],
      [rankDir(D.defaultRank) + 'host.*.log', A.hostSpans
        ? 'STRACE host span（bind / runner_run / device_wall / sched）' : '缺失 —— 无 E2E 层'],
      ['report/perf_hints.log', D.hints.length + ' 条 perf hint（'
        + Array.from(new Set(D.hints.map((h) => h.code))).sort().join(' / ') + '）'],
      ['passes_dump/', D.passes.length + ' 个 IR dump，' + D.passes[0].lines
        + ' → ' + D.passes[D.passes.length - 1].lines + ' 行'],
      ['chip_swimlane_records.json', A.chipSwimlaneRecords
        ? '核清单与时钟频率' : '缺失 —— 核数由 trace 线程名推得'],
      ['binary_context.json', A.binaryContext
        ? 'platform ' + D.case.toolchain.platform + ' · pto-isa '
          + D.case.toolchain.ptoIsaRevision.slice(0, 12) + ' · runtime ' + D.case.toolchain.runtimeName
        : '缺失'],
      ['ptoas/', A.ptoas ? A.ptoas + ' 个单元（.pto + .cpp）' : '缺失'],
      ['kernels/', Object.keys(A.kernelDirs).length
        ? Object.keys(A.kernelDirs).map((k) => k + ' ' + A.kernelDirs[k]).join(' · ') : '缺失'],
      ['orchestration/', A.orchestration.length ? A.orchestration.join(' · ') : '缺失'],
      ['kernel_config.py', A.kernelConfig
        ? 'runtime ' + (D.case.toolchain.runtimeName || '—')
          + (D.case.toolchain.aicpuThreads ? ' · ' + D.case.toolchain.aicpuThreads + ' AICPU 线程' : '')
        : '缺失'],
    ]);
    body.appendChild(table([
      { label: '产物', cell: (r) => esc(r[0]), mono: true },
      { label: '内容', cell: (r) => esc(r[1]) },
    ], rows, {}));
  }

  /* One device, or no host spans: there is no cross-rank comparison to make,
   * so the inspector reports this run on its own terms. */
  function renderRunInspectorSingle(host) {
    const R0 = R();
    const s2 = inspectorSection('本次运行', D.case.ranks.length + ' 个执行单元');
    s2.appendChild(kv([
      ['trace span', num(R0.swimlane.spanUs, 1) + ' us'],
      ['device_wall', hasE2E()
        ? num(D.e2e[S.rank][TRACE_MATCH[S.rank].inv]['chip.run.runner_run.device_wall'].us, 1) + ' us'
        : '无 host log'],
      ['AIC 占用', pct(R0.occupancy.aicUtil)],
      ['AIV 占用', pct(R0.occupancy.aivUtil)],
      ['依赖关键路径', R0.critical.tags.length + ' 节点（静态 CPM）'],
      ['调度器占用', pct(R0.scheduler.perLaneUtil)],
    ]));
    if (!hasE2E()) {
      s2.appendChild(el('div', 'inspector-soft-card is-warning',
        '无 host STRACE log：迭代次数、device_wall、bind 缓存命中都不可得，'
        + '基线只能锁在 trace span ' + num(R0.swimlane.spanUs, 1) + ' us 上'));
    }
    host.appendChild(s2);

    const s3 = inspectorSection('瓶颈链', topChains(3).length + ' 条 · 体检项见左栏');
    topChains(3).forEach((f) => {
      s3.appendChild(btn(f.id + ' · ' + f.title, {
        variant: 'ghost', size: 'sm',
        on: () => { S.finding = f.id; S.focus = 'finding'; applyFocus(f); render(); },
      }));
    });
    host.appendChild(s3);
    host.appendChild(renderLedger());
  }

  /* ========================================================= chrome */
  function renderTabs() {
    const host = $('#levelTabs');
    host.textContent = '';
    host.hidden = guidedJourneyActive();
    if (host.parentElement) host.parentElement.hidden = host.hidden;
    if (host.hidden) return;
    const levels = isServingBenchmark() ? LEVELS.filter((l) => l.id === 'e2e') : LEVELS;
    levels.forEach((l) => {
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
    const guided = guidedJourneyActive();
    /* sub-view switch sits where the view's own tab would: on the left */
    if (S.view === 'compiler') {
      host.appendChild(group('segmented-control segmented-control-muted', [
        { id: 'passes', label: 'Pass 轨迹' },
        { id: 'depth', label: '流水深度' },
        { id: 'granularity', label: '搬运粒度' },
      ], S.compilerTab, (v) => { S.compilerTab = v; render(); }));
    }
    if (S.view === 'e2e' && !isServingBenchmark() && QW()) {
      host.appendChild(group('segmented-control segmented-control-muted', [
        { id: 'step', label: '步时间', hint: 'torch step_trace_time：Computing / Free / Preparing' },
        { id: 'ops', label: '设备算子', hint: '融合 kernel 之外还在设备上跑什么' },
        { id: 'api', label: '主机 API', hint: '哪些 CANN 调用吃掉主机时间' },
        { id: 'topo', label: 'TP 对照', hint: 'TP=1 / TP=2 并排' },
      ], S.e2ePanel, (v) => { S.e2ePanel = v; render(); }));
    } else if (S.view === 'e2e' && !isServingBenchmark()) {
      host.appendChild(group('segmented-control segmented-control-muted', [
        { id: 'triage', label: '定界' },
        { id: 'serving', label: 'Serving / Host' },
        { id: 'device', label: 'Device 轨迹' },
        { id: 'samples', label: '调用数据' },
      ], S.e2ePanel, (v) => { S.e2ePanel = v; render(); }));
    }
    if (S.view === 'l2' && QW()) {
      host.appendChild(group('segmented-control segmented-control-muted', [
        { id: 'swimlane', label: onPrimaryVariant() ? '泳道' : '核占用' },
        { id: 'layers', label: '40 层' },
        { id: 'head', label: '头开销' },
      ], S.l2Panel, (v) => { S.l2Panel = v; render(); }));
    }
    if (S.view === 'l1' && QW()) {
      host.appendChild(group('segmented-control segmented-control-muted', [
        { id: 'pipe', label: '单核流水' },
        { id: 'pmu', label: 'PMU 实测' },
      ], S.l1Panel, (v) => { S.l1Panel = v; render(); }));
    }

    const right = el('div', 'tc-toolbar-right');

    if (S.view === 'compiler') {
      right.appendChild(el('span', 'tc-readout', D.passes.length + ' Pass dump · ' + D.hints.length + ' perf hint'));
    }

    /* the swimlane's own controls only mean something on the task-level
     * swimlane; the aggregate panels have nothing to filter or zoom */
    if (S.view === 'l2' && (!QW() || (onPrimaryVariant() && S.l2Panel === 'swimlane'))) {
      right.appendChild(field('视图', select([
        { id: 'summary', label: '关键表现' },
        { id: 'all', label: '原始逐核 ' + R().swimlane.lanes.length + ' 核' },
        { id: 'aic', label: 'AIC ' + D.case.aicCount },
        { id: 'aiv', label: 'AIV ' + D.case.aivCount },
      ], S.laneFilter, (v) => { S.laneFilter = v; render(); })));
      const colorField = el('div', 'tc-field');
      colorField.appendChild(el('span', null, '着色'));
      colorField.appendChild(btn(S.colorOn ? '开' : '关', {
        size: 'sm', selected: S.colorOn,
        title: S.colorOn ? '关闭算子配色，让占用率与空转段更突出' : '恢复算子配色',
        on: () => { S.colorOn = !S.colorOn; render(); },
      }));
      if (S.colorOn) {
        colorField.appendChild(select([
          { id: 'semantic', label: '按算子' },
          { id: 'engine', label: '按引擎' },
        ], S.colorMode, (v) => { S.colorMode = v; render(); }));
      }
      right.appendChild(colorField);
      right.appendChild(field('叠加', select([
        { id: 'none', label: '无' },
        { id: 'sched', label: 'AICPU 调度' },
        { id: 'ready', label: 'Ready queue' },
      ], S.overlay, (v) => { S.overlay = v; render(); })));
      right.appendChild(field('连线', select([
        { id: 'off', label: '关' },
        { id: 'sel', label: '选中任务' },
        { id: 'path', label: '沿执行主路径' },
      ], S.deps, (v) => { S.deps = v; redrawStage(); renderToolbar(); })));
      const zoomGroup = el('div', 'toolbar-control');
      zoomGroup.appendChild(btn('−', { variant: 'ghost', size: 'icon', title: '缩小', on: () => { zoom(2); redrawStage(); renderToolbar(); } }));
      zoomGroup.appendChild(btn('全程', { variant: 'ghost', size: 'sm', title: '恢复完整时间范围', on: () => { S.t0 = 0; S.t1 = R().swimlane.spanUs; redrawStage(); renderToolbar(); } }));
      zoomGroup.appendChild(btn('+', { variant: 'ghost', size: 'icon', title: '放大', on: () => { zoom(0.5); redrawStage(); renderToolbar(); } }));
      right.appendChild(zoomGroup);
      right.appendChild(el('span', 'tc-readout', num(S.t0, 0) + '–' + num(S.t1, 0) + ' us · shift+拖动平移'));
    }

    if (S.view === 'l1' && (!QW() || (onPrimaryVariant() && S.l1Panel === 'pipe'))) {
      const ordered = R().tasks.slice().sort((a, b) => b.span - a.span);
      right.appendChild(field('kernel', select(ordered.map((t) => ({
        id: t.tag, label: t.callable + ' · ' + t.tag + '（' + num(t.span, 0) + ' us）',
      })), S.task, (v) => { S.task = v; S.focus = 'task'; render(); })));
    }

    /* the capture chooser follows the reader across E2E / L2 / L1 */
    if (QW() && (S.view === 'e2e' || S.view === 'l2' || S.view === 'l1')) {
      const cur = qwVariant();
      right.appendChild(field('采集', select(QW().variants.map((v) => ({
        id: v.id, label: v.label + '（' + msOrUs(v.spanUs) + (v.validated ? ' ✓' : '') + '）',
      })), cur.id, (v) => { S.variant = v; render(); })));
      const badge = el('span', 'tc-readout' + (qwCapture().validated ? '' : ' is-crit'));
      badge.textContent = qwCapture().validated ? '数据集已校验' : '数据集未校验';
      badge.title = qwCapture().validateNote;
      right.appendChild(badge);
    }

    /* E2E is always a cross-rank comparison. Every deeper lens carries one
     * selected analysis rank; compiler / ISA keep that context even though
     * their dumped artifacts are shared rather than rank-specific. */
    if (S.view !== 'e2e' && multiRank()) {
      right.appendChild(field('分析 rank', select(Object.keys(D.ranks).map((r) => ({ id: r, label: r })), S.rank,
        (v) => {
          selectAnalysisRank(v);
          render();
        })));
      if (S.view === 'compiler' || S.view === 'isa') {
        right.appendChild(el('span', 'tc-readout', '编译 / ISA 产物为 rank 共享'));
      }
    }

    host.appendChild(right);
    host.hidden = guided || !host.childNodes.length || (host.childNodes.length === 1 && !right.childNodes.length);
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
    if (QW()) {
      /* four captures of one graph, grouped the way the dataset is: two
       * collectors, two stages. The armed one drives E2E / L2 / L1. */
      const q = QW();
      Object.keys(q.captures).forEach((cid) => {
        const cap = q.captures[cid];
        row(1, cap.label + ' / ' + cap.collector.replace(/^collect_|\.py$/g, ''),
          cap.validated ? '已校验' : '未校验');
        q.variants.filter((v) => v.capture === cid).forEach((v) => {
          row(2, v.stage + (v.primary ? ' · 任务级' : ' · 聚合'), msOrUs(v.spanUs), {
            selected: v.id === S.variant,
            on: () => { S.variant = v.id; render(); },
          });
        });
      });
      row(2, 'host.*.log', '缺失');
    } else {
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
        if (D.e2e && D.e2e[rank]) {
          Object.keys(D.e2e[rank]).forEach((inv) => {
            row(2, 'inv=' + inv + (m && m.inv === +inv ? ' · traced' : ''),
              us(D.e2e[rank][inv]['chip.run.runner_run.device_wall'].us, 0), {
                on: () => { S.rank = rank; S.view = 'e2e'; render(); },
              });
          });
        } else {
          row(2, 'host.*.log', '缺失');
        }
      });
    }
    if (QW()) {
      const t = QW().torch[qwVariant().capture];
      row(0, 'torch profiler', 'device ' + t.deviceId);
      row(1, 'step_trace_time.csv', t.steps.length + ' step', {
        on: () => { S.view = 'e2e'; S.e2ePanel = 'step'; render(); },
      });
      row(1, 'kernel_details.csv', t.kernelRows + ' 行', {
        on: () => { S.view = 'l1'; S.l1Panel = 'pmu'; render(); },
      });
      row(1, 'op_statistic.csv', t.ops.length + ' OP', {
        on: () => { S.view = 'e2e'; S.e2ePanel = 'ops'; render(); },
      });
      row(1, 'api_statistic.csv', t.apiHost.rows + ' API', {
        on: () => { S.view = 'e2e'; S.e2ePanel = 'api'; render(); },
      });
    }
    row(0, 'artifacts', D.passes.length + ' passes');
    if (D.case.compileSource) {
      row(1, '↑ 来自 ' + D.case.compileSource.runDir.replace(/^_jit_/, '').slice(0, 22),
        D.case.compileSource.capturedAt.slice(0, 10));
    }
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
    /* Most chains remain reachable from every layer they traverse. A chain
     * may override that with queueLevels when its causal home is elsewhere:
     * C1, for example, is seen in L2 but belongs to E2E triage. */
    const counts = { all: D.findings.length };
    D.findings.forEach((f) => {
      (f.queueLevels || f.levels || [f.level])
        .forEach((lv) => { counts[lv] = (counts[lv] || 0) + 1; });
    });
    [{ id: 'all', label: '全部' }].concat(LEVELS.filter((l) => counts[l.id]).map((l) => ({ id: l.id, label: l.label })))
      .forEach((o) => {
        filterHost.appendChild(btn(o.label + ' ' + (counts[o.id] || 0), {
          size: 'sm', selected: S.findingLevel === o.id,
          on: () => { S.findingLevel = o.id; render(); },
        }));
      });

    const list = $('#findingList');
    list.textContent = '';
    const shown = D.findings.filter((f) => S.findingLevel === 'all'
      || (f.queueLevels || f.levels || [f.level]).indexOf(S.findingLevel) >= 0);
    const logged = {};
    S.ledger.forEach((r) => { if (r.findingId) logged[r.findingId] = 1; });

    const findingRow = (f) => {
      const b = el('button', 'tc-finding'
        + (f.kind === 'hygiene' ? ' is-hygiene' : '')
        + (f.id === S.finding && S.focus === 'finding' ? ' is-selected' : '')
        + (logged[f.id] ? ' is-logged' : ''));
      b.type = 'button';
      b.dataset.sev = f.severity;
      /* The left-rail impact trace is driven by the same measured makespan
       * share shown in the row, rather than a decorative, invented score. */
      b.dataset.hasCost = f.cost ? 'true' : 'false';
      b.style.setProperty('--finding-impact', (f.cost
        ? Math.max(7, Math.min(36, (Number(f.cost.share) || 0) * 0.36)) : 10) + 'px');
      const hd = el('div', 'hd');
      hd.appendChild(el('span', 'id', f.id));
      /* a chain advertises the layers it crosses; that path IS its identity */
      hd.appendChild(el('span', 'lv', (f.levels || [f.level])
        .map((lv) => LEVEL_LABEL[lv] || lv).join(' → ')));
      b.appendChild(hd);
      b.appendChild(el('span', 'ti', f.title));
      const mt = el('div', 'mt');
      mt.appendChild(el('span', 'm', f.metric));
      mt.appendChild(el('span', f.cost ? 'cost' : 'cost is-none',
        f.cost ? f.cost.share + '%' : '无归因'));
      b.appendChild(mt);
      b.addEventListener('click', () => {
        S.finding = f.id;
        S.chainStep = null;
        S.focus = 'finding';
        applyFocus(f);
        const primaryTags = f.taskRoles && f.taskRoles.primary && f.taskRoles.primary.length
          ? f.taskRoles.primary
          : [f.focus && f.focus.task || (f.contention && f.contention.focus)
            || (f.subjects && f.subjects.tasks && f.subjects.tasks[0])].filter(Boolean);
        const relatedTags = (f.taskRoles && f.taskRoles.secondary && f.taskRoles.secondary.length
          ? f.taskRoles.secondary : (f.subjects && f.subjects.tasks || []).filter((tag) => !primaryTags.includes(tag)));
        const focusTasks = primaryTags.concat(relatedTags).map((tag) => tasksOf[S.rank][tag]).filter(Boolean);
        if (focusTasks.length) {
          const lo = Math.min(...focusTasks.map((task) => task.start));
          const hi = Math.max(...focusTasks.map((task) => task.end));
          if (hi > lo) {
            const pad = Math.max(24, (hi - lo) * 0.12);
            setWindow(lo - pad, hi + pad);
          }
          const primary = tasksOf[S.rank][primaryTags[0]];
          if (primary) {
            S.task = primary.tag;
            S.pathFocus = true;
            const taskIndex = R().tasks.indexOf(primary);
            const laneIndex = R().swimlane.blocks.findIndex((blocks) => blocks.some((block) => block[2] === taskIndex));
            if (laneIndex >= 0) S.scrollToLane = R().swimlane.laneNames[laneIndex];
          }
          S.laneFilter = 'summary';
        }
        if (f.id === 'C2' && f.contention && f.contention.windows && f.contention.windows.length) {
          const focusTask = tasksOf[S.rank][f.contention.focus];
          const lo = Math.min(focusTask ? focusTask.start : Infinity,
            ...f.contention.windows.map((win) => win.t0));
          const hi = Math.max(focusTask ? focusTask.end : -Infinity,
            ...f.contention.windows.map((win) => win.t1));
          if (Number.isFinite(lo) && Number.isFinite(hi) && hi > lo) {
            const pad = Math.max(35, (hi - lo) * 0.12);
            setWindow(lo - pad, hi + pad);
          }
          S.laneFilter = 'summary';
          if (focusTask) { S.task = focusTask.tag; S.pathFocus = true; }
        }
        render();
      });
      return b;
    };

    /* Two groups, never interleaved: a chain carries a makespan attribution,
     * a hygiene item carries the reason it does not. Mixing them is how a
     * hint count ends up looking as urgent as a 39% finding. */
    const group = (label, kicker, rows) => {
      if (!rows.length) return;
      const h = el('div', 'tc-finding-group');
      h.appendChild(el('span', 'gl', label));
      h.appendChild(el('span', 'gk', kicker));
      list.appendChild(h);
      rows.forEach((f) => list.appendChild(findingRow(f)));
    };
    const chains = shown.filter((f) => f.kind !== 'hygiene');
    const hyg = shown.filter((f) => f.kind === 'hygiene');
    group('瓶颈链', chains.length
      ? '合计 ' + num(chains.reduce((n, f) => n + (f.cost ? f.cost.us : 0), 0), 0)
        + ' us · 与 makespan 有时间重叠，不可相加'
      : '本层无', chains);
    group('体检项', hyg.length ? '无 makespan 归因' : '本层无', hyg);
    $('[data-bind="findingCount"]').textContent = shown.length + ' / ' + D.findings.length;
  }

  /* Activating a finding puts the stage where its evidence lives and turns on
   * evidence focus, so the reader never has to guess which parts of the screen
   * the inspector is talking about. */
  function applyFocus(f) {
    if (!f) return;
    if (D.case.guidedJourney && f.chain && f.chain.length) {
      applyStep(f, f.chain[0]);
      return;
    }
    applySubjects(f.subjects || {}, f.focus && f.focus.pass);
  }

  /* A chain step is focusable in its own right: the reader walks the ladder
   * and the stage follows one layer at a time, instead of the whole chain
   * always dumping them on its first layer. */
  function applyStep(f, st) {
    if (!f || !st) return;
    S.chainStep = f.id + ':' + (f.chain || []).indexOf(st);
    applySubjects(st.subjects || {}, st.level === 'compiler' ? (f.rootPass || 'MemoryReuse') : null);
    if (D.case.guidedJourney) {
      S.view = 'l2';
      S.l2Panel = 'swimlane';
      if (f.rootPass) S.guidedPass = f.rootPass;
    }
  }

  function applySubjects(s, passName) {
    S.view = s.view || S.view;
    if (s.tab) S.compilerTab = s.tab;
    if (s.overlay) S.overlay = s.overlay;
    /* a qwen3 entry belongs to one capture and one panel; arming the entry has
     * to arm both, or the screen shows a different run than the claim */
    if (s.variant && QW() && QW().variants.some((v) => v.id === s.variant)) S.variant = s.variant;
    if (s.panel) {
      if (s.view === 'e2e') S.e2ePanel = s.panel;
      else if (s.view === 'l2') S.l2Panel = s.panel;
      else if (s.view === 'l1') S.l1Panel = s.panel;
    }
    if (s.tasks && s.tasks.length && tasksOf[S.rank][s.tasks[0]]) S.task = s.tasks[0];
    if (s.sites && s.sites.length) S.hintSite = s.sites[0];
    if (s.lanes && s.lanes.length) S.laneFilter = s.lanes[0].indexOf('AIC') === 0 ? 'aic' : 'aiv';
    if (passName) {
      const p = D.passes.find((x) => x.name === passName);
      if (p) S.pass = p.idx;
    }
    S.critOnly = false;
    S.pathOnly = 'off';
    S.pathFocus = false;
    S.focusEvidence = true;
    if (S.view === 'l2') { S.t0 = 0; S.t1 = R().swimlane.spanUs; }
  }

  function renderStatus() {
    const host = $('#statusStrip');
    host.textContent = '';
    const rank = R();
    const open = openExperiment();
    const qv = QW() ? qwVariant() : null;
    const ql = qv ? qwL2() : null;
    /* On the qwen3 case the status strip has to describe the armed capture,
     * not the single rank data.js happens to carry. */
    const items = qv ? [
      ['case', D.case.program],
      ['采集', qv.label + (qv.validated ? ' ✓' : ' · 未校验')],
      ['span', msOrUs(ql.spanUs)],
      ['tasks', String(ql.taskCount) + (qv.hasNames ? '' : ' · 无名')],
      ['层', (qwLayers() ? '40 × ' + qwLayers().perLayer : '—')],
      ['AIC / AIV', pct(ql.occ.aicUtil, 0) + ' / ' + pct(ql.occ.aivUtil, 0)],
      ['忙核', num(ql.occ.busyCores, 1) + ' / ' + ql.coreTotal],
      ['hints', String(D.hints.length)],
    ] : (S.view === 'e2e' && multiRank()) ? [
      ['case', D.case.program],
      ['ranks', D.case.ranks.join(' ↔ ')],
      ['启动偏移', D.launchSkew ? us(D.launchSkew.runnerUs, 1) : '无同钟证据'],
      ['集合点等待', D.launchSkew ? us(D.launchSkew.measuredSum, 1) : '—'],
      ['调用', D.case.ranks.map((r) => 'inv=' + TRACE_MATCH[r].inv).join(' / ')],
      ['hints', String(D.hints.length)],
    ] : [
      ['case', D.case.program],
      ['rank', S.rank + (TRACE_MATCH[S.rank] ? ' inv=' + TRACE_MATCH[S.rank].inv : ' · 无 host log')],
      ['span', us(rank.swimlane.spanUs, 1)],
      ['tasks', String(rank.tasks.length)],
      ['crit', rank.critical.tags.length + ' CPM / ' + rank.cpath.segments.length + ' 观测'],
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
    /* the torch profiler run has PMU on; the swimlane capture does not */
    const pmuOn = !!qv;
    const pv = el('span', 'v ' + (pmuOn ? 'ok' : 'warn'), pmuOn ? 'torch 侧 on' : 'off');
    if (pmuOn) pv.title = 'kernel_details.csv 带 PMU 计数器；泳道那一轮没有 PMU，两者不能比墙钟';
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
    if (isServingBenchmark()) {
      const b = D.benchmark;
      const dl = el('dl', 'tc-fp-grid');
      const add = (k, v) => { dl.appendChild(el('dt', null, k)); dl.appendChild(el('dd', null, v)); };
      add('model', D.case.model);
      add('topology', 'DP4 / TP4 / EP16 · concurrency ' + b.concurrency);
      add('shape', '256 input / 64 output tokens · ' + b.requests + ' requests');
      add('benchmark', 'result3.json · ' + num(b.durationS, 2) + ' s');
      add('trace', D.benchmark.trace.raw + ' · request-level only');
      body.appendChild(dl);
      host.appendChild(body);
      return;
    }
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
    add('kernel / scope', D.case.callables + ' 个 kernel 名 / ' + scopeCountOf()
      + ' 个 scope（IR incore scope ' + D.case.incoreScopes.length + '）');
    add('captured', D.case.capturedAt);
    add('source root', D.case.sourceRoot);
    body.appendChild(dl);
    body.appendChild(sectionHead('绑定参数', '前 12 / ' + D.case.params.length));
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
        kind: 'finding',
        text: [f.id, f.title, f.axis, f.kind].concat(f.levels || [])
          .concat((f.chain || []).map((st) => st.headline)).join(' '),
        name: f.id + ' ' + f.title,
        value: f.cost ? f.cost.share + '% · ' + f.metric : '无归因 · ' + f.metric,
        go: () => { S.finding = f.id; S.chainStep = null; S.focus = 'finding'; applyFocus(f); },
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

  /* A request benchmark is a legitimate E2E case, even when no device-side
   * artifacts were captured with it.  Keep it in the same case switcher but
   * give it a purpose-built, E2E-only surface rather than borrowing another
   * run's rank, trace, or compiler evidence. */
  function viewServingBenchmark(stage) {
    const b = D.benchmark;
    const sec = el('section');
    sec.appendChild(sectionHead('Serving 压测 · 请求级结果', 'GBS256 · input 256 / output 64 · DP4 / TP4 / EP16',
      el('span', 'tc-readout', b.completed + ' / ' + b.requests + ' completed · ' + b.failed + ' failed')));
    sec.appendChild(tiles([
      { k: '请求吞吐', v: num(b.requestThroughput, 2), u: 'req/s' },
      { k: '输出吞吐', v: num(b.outputThroughput, 1), u: 'token/s' },
      { k: '并发', v: b.concurrency, u: 'requests' },
      { k: '压测窗口', v: num(b.durationS, 2), u: 's' },
    ]));
    stage.appendChild(sec);

    const latency = el('section');
    latency.appendChild(sectionHead('请求时延分布', '圆点 = Mean · 短线 = P50 → P99；每一行使用自己的刻度，避免 TTFT 掩盖 token 时延'));
    const chart = el('div', 'tc-serving-latency');
    b.latency.forEach((metric) => {
      const max = Math.max(metric.mean, metric.median, metric.p99, 1);
      const row = el('div', 'tc-serving-latency-row');
      const label = el('div', 'label');
      label.appendChild(el('strong', null, metric.label));
      label.appendChild(el('small', null, metric.hint));
      row.appendChild(label);
      const range = el('div', 'range');
      const line = el('i', 'line');
      line.style.left = (metric.median / max * 100).toFixed(2) + '%';
      line.style.width = Math.max(1, (metric.p99 - metric.median) / max * 100).toFixed(2) + '%';
      const mean = el('i', 'mean');
      mean.style.left = (metric.mean / max * 100).toFixed(2) + '%';
      range.appendChild(line);
      range.appendChild(mean);
      row.appendChild(range);
      const values = el('div', 'values');
      values.appendChild(el('span', null, 'P50 ' + num(metric.median, 1)));
      values.appendChild(el('strong', null, 'Mean ' + num(metric.mean, 1)));
      values.appendChild(el('span', null, 'P99 ' + num(metric.p99, 1) + ' ms'));
      row.appendChild(values);
      chart.appendChild(row);
    });
    latency.appendChild(chart);
    stage.appendChild(latency);

    const trace = el('section');
    trace.appendChild(sectionHead('随附 Chrome trace 的可用范围', '原始 trace 未打包进 demo；以下是已确认的事件类别与缺失维度'));
    const grid = el('div', 'tc-serving-trace-map');
    [
      ['可用', 'PyTorch CPU op · aten::copy_ · Event::synchronize · gloo:all_reduce'],
      ['不可直接回答', b.trace.missing.join(' · ')],
      ['原始产物', b.trace.raw + ' · ' + (b.trace.bytes / 1024 / 1024 / 1024).toFixed(2) + ' GB'],
    ].forEach((item) => {
      const row = el('div', 'tc-serving-trace-row');
      row.dataset.state = item[0] === '可用' ? 'ok' : item[0] === '不可直接回答' ? 'missing' : 'source';
      row.appendChild(el('span', null, item[0]));
      row.appendChild(el('strong', null, item[1]));
      grid.appendChild(row);
    });
    trace.appendChild(grid);
    stage.appendChild(trace);
  }

  function renderServingBenchmarkExplorer() {
    const b = D.benchmark;
    const tree = $('#runTree');
    tree.textContent = '';
    [['GBS256 压测', 'DP4 / TP4 / EP16'], ['请求', b.completed + ' / ' + b.requests + ' completed'], ['输入 / 输出', b.inputTokens + ' / ' + b.outputTokens + ' tokens'], ['原始 trace', (b.trace.bytes / 1024 / 1024 / 1024).toFixed(2) + ' GB']].forEach((item, i) => {
      const row = el('div', 'tc-tree-row');
      row.dataset.depth = i ? '1' : '0';
      row.appendChild(el('span', 'n', item[0]));
      row.appendChild(el('span', 'm', item[1]));
      tree.appendChild(row);
    });
    $('[data-bind="explorerMeta"]').textContent = 'benchmark';
    $('[data-bind="findingCount"]').textContent = '采集范围';
    $('#findingFilter').textContent = '';
    const list = $('#findingList');
    list.textContent = '';
    list.appendChild(el('div', 'inspector-soft-card is-warning', '仅请求级 benchmark 可用；没有同 scope 的 Host / Device wall 拆分，设备侧下钻保持不可用。'));
  }

  function renderServingBenchmarkInspector() {
    const b = D.benchmark;
    $('[data-bind="inspectorTitle"]').textContent = 'Benchmark';
    $('[data-bind="inspectorMeta"]').textContent = 'request-level';
    const host = $('#inspector');
    host.textContent = '';
    const summary = inspectorSection('负载指纹', '真实压测');
    summary.appendChild(kv([
      ['模型', D.case.model],
      ['并发', b.concurrency + ' requests'],
      ['输入 / 输出', '256 / 64 tokens per request'],
      ['完成率', b.completed + ' / ' + b.requests],
      ['Trace', b.trace.raw],
    ]));
    host.appendChild(summary);
    const caveat = inspectorSection('不能推断', b.trace.missing.length + ' 项');
    b.trace.missing.forEach((item) => caveat.appendChild(el('div', 'inspector-soft-card is-warning', item)));
    host.appendChild(caveat);
  }

  function renderServingBenchmarkDock() {
    const b = D.benchmark;
    $('[data-bind="dockMeta"]').textContent = 'request-level benchmark';
    $('#dockMode').textContent = '';
    const body = $('#dockBody');
    body.textContent = '';
    body.appendChild(tiles([
      { k: 'TTFT P99', v: num(b.latency[0].p99, 1), u: 'ms', tone: 'warn' },
      { k: 'TPOT P99', v: num(b.latency[1].p99, 1), u: 'ms' },
      { k: 'ITL P99', v: num(b.latency[2].p99, 1), u: 'ms', tone: 'warn' },
      { k: '总 token 吞吐', v: num(b.totalTokenThroughput, 1), u: 'token/s' },
    ]));
  }

  function renderServingBenchmarkTerminal() {
    const tabs = $('#terminalTabs');
    tabs.textContent = '';
    tabs.appendChild(el('span', 'pto-ide-frame__terminal-tab is-selected', 'Benchmark output'));
    const body = $('#terminalBody');
    body.textContent = '';
    const b = D.benchmark;
    body.appendChild(el('pre', 'tc-term-static', [
      'result3.json · ' + b.completed + '/' + b.requests + ' requests complete',
      'request throughput  ' + num(b.requestThroughput, 3) + ' req/s',
      'output throughput   ' + num(b.outputThroughput, 3) + ' token/s',
      'TTFT mean / p99     ' + num(b.latency[0].mean, 2) + ' / ' + num(b.latency[0].p99, 2) + ' ms',
      'TPOT mean / p99     ' + num(b.latency[1].mean, 2) + ' / ' + num(b.latency[1].p99, 2) + ' ms',
      'ITL  mean / p99     ' + num(b.latency[2].mean, 2) + ' / ' + num(b.latency[2].p99, 2) + ' ms',
    ].join('\n')));
  }

  function renderServingBenchmarkStatus() {
    const host = $('#statusStrip');
    host.textContent = '';
    [['case', 'GBS256'], ['并发', String(D.benchmark.concurrency)], ['吞吐', num(D.benchmark.outputThroughput, 1) + ' token/s'], ['TTFT P99', num(D.benchmark.latency[0].p99, 1) + ' ms'], ['Device', '未采集']].forEach((item) => {
      const status = el('span', 'tc-status-item');
      status.appendChild(el('span', 'k', item[0]));
      status.appendChild(el('span', 'v', item[1]));
      host.appendChild(status);
    });
  }

  function render() {
    const stage = $('#stage');
    if (stage.__ro) { stage.__ro.disconnect(); stage.__ro = null; }
    stage.__redraw = null;
    stage.textContent = '';
    stage.classList.toggle('tc-stage--e2e', S.view === 'e2e');

    if (isServingBenchmark()) {
      S.view = 'e2e';
      renderTabs();
      renderToolbar();
      renderServingBenchmarkExplorer();
      viewServingBenchmark(stage);
      renderServingBenchmarkInspector();
      renderServingBenchmarkStatus();
      $('[data-bind="caseChip"]').textContent = D.case.program + ' · request-level';
      return;
    }

    renderTabs();
    renderToolbar();
    renderExplorer();
    findingBar(stage);

    if (S.view === 'e2e') viewE2E(stage);
    else if (S.view === 'l2') viewL2(stage);
    else if (S.view === 'l1') viewL1(stage);
    else if (S.view === 'compiler') viewCompiler(stage);
    else viewISA(stage);

    renderInspector();
    renderStatus();
    $('[data-bind="caseChip"]').textContent = D.case.program + ' · '
      + (QW() ? qwVariant().id : S.rank);
  }

  /* ------------------------------------------------------------- boot */
  /* Switching case swaps the whole dataset. The two dumps carry different
   * artifacts, so every layer re-derives what it can and says what it cannot. */
  function switchCase(id) {
    if (id === D.case.id) { toggleCaseMenu(false); return; }
    loadCase(id);
    S.tile = isServingBenchmark() ? null : defaultTile();
    toggleCaseMenu(false);
    renderCaseMenu();
    renderFingerprint();
    render();
  }

  function toggleCaseMenu(force) {
    const menu = $('#caseMenu');
    const chip = document.querySelector('[data-act="toggle-fingerprint"]');
    const open = force === undefined ? menu.hidden : force;
    menu.hidden = !open;
    chip.setAttribute('aria-expanded', open ? 'true' : 'false');
    if (open) toggleFingerprint(false);
  }

  function renderCaseMenu() {
    const menu = $('#caseMenu');
    menu.textContent = '';
    CASES.forEach((c) => {
      const run = RUNS[c.id];
      const b = el('button', 'tc-case-item' + (c.id === D.case.id ? ' is-selected' : ''));
      b.type = 'button';
      const hd = el('div', 'hd');
      hd.appendChild(el('span', 'nm', c.label));
      hd.appendChild(el('span', 'sub', c.sub));
      b.appendChild(hd);
      /* say up front which layers this dump can answer */
      const layers = el('div', 'ly');
      const layerList = run.kind === 'serving-benchmark' ? [
        ['E2E', true], ['L2', false], ['L1/L0', false], ['编译器', false], ['ISA', false],
      ] : [
        ['E2E', !!run.e2e], ['L2', true], ['L1/L0', true],
        ['编译器', run.passes.length > 0], ['ISA', run.case.artifacts.ptoas > 0],
      ];
      layerList.forEach((pair) => {
        layers.appendChild(el('span', pair[1] ? 'on' : 'off', pair[0]));
      });
      b.appendChild(layers);
      const meta = run.kind === 'serving-benchmark'
        ? run.benchmark.completed + '/' + run.benchmark.requests + ' 请求 · '
          + num(run.benchmark.outputThroughput, 1) + ' token/s · TTFT P99 ' + num(run.benchmark.latency[0].p99, 0) + ' ms'
        : run.ranks[run.defaultRank].tasks.length + ' 任务 · '
          + (run.chainCount != null ? run.chainCount + ' 条瓶颈链 · ' + run.hygieneCount + ' 条体检项'
            : run.findings.length + ' 条瓶颈')
          + ' · ' + run.hints.length + ' 条提示';
      b.appendChild(el('div', 'mt', meta));
      b.addEventListener('click', () => switchCase(c.id));
      menu.appendChild(b);
    });
    const fp = el('button', 'tc-case-item is-action', 'Case fingerprint …');
    fp.type = 'button';
    fp.addEventListener('click', () => { toggleCaseMenu(false); toggleFingerprint(true); });
    menu.appendChild(fp);
  }

  function boot() {
    if (window.PtoIdeFrame) window.PtoIdeFrame.initAll();
    if (EMBED_VIEW) document.body.classList.add('tc-embed-view');
    loadCase(initialCase);
    S.tile = isServingBenchmark() ? null : defaultTile();
    S.view = isServingBenchmark() ? 'e2e' : (EMBED_VIEW || 'e2e');
    S.focus = null;
    renderCaseMenu();
    renderFingerprint();
    render();

    document.querySelector('[data-act="toggle-fingerprint"]').addEventListener('click', () => toggleCaseMenu());
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
      if (!e.target.closest('#caseMenu') && !e.target.closest('[data-act="toggle-fingerprint"]')) {
        toggleCaseMenu(false);
      }
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && S.scopeReturn && S.view === 'l2'
        && document.activeElement !== input && document.activeElement.tagName !== 'INPUT') {
        e.preventDefault(); scopeBack(); return;
      }
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
