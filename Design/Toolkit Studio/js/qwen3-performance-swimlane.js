(function registerQwen3PerformanceSwimlane() {
  'use strict';

  const DATA_URL = '../../Data/_jit_decode_fwd_layers_20260625_184941/dfx_outputs/merged_swimlane_20260625_185006.json';
  const PANEL_ID = 'qwen3PerformanceSwimlanePanel';
  const CANVAS_ID = 'qwen3PerformanceSwimlane';
  const LABEL_WIDTH = 74;
  const AXIS_HEIGHT = 18;
  const ROW_HEIGHT = 19;
  const ROW_GAP = 2;
  // Keep the default view readable while showing more of the real trace:
  // 4 AIC + 5 AIV + 1 AICPU = 10 representative hardware lanes.
  const LANE_LIMITS = { AIC: 4, AIV: 5, AICPU: 1 };

  let initialized = false;
  let loaded = false;
  let loading = null;
  let model = null;
  let selectedTask = null;
  let hitBoxes = [];
  let resizeObserver = null;

  const qs = (selector, root = document) => root.querySelector(selector);

  function panel() {
    return document.getElementById(PANEL_ID);
  }

  function canvas() {
    return document.getElementById(CANVAS_ID);
  }

  function cssColor(name, fallback) {
    const value = getComputedStyle(panel() || document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
  }

  function laneKind(name) {
    if (/^AICPU(?:_|$)/i.test(name)) return 'AICPU';
    if (/^AIV(?:_|$)/i.test(name)) return 'AIV';
    if (/^AIC(?:_|$)/i.test(name)) return 'AIC';
    return 'other';
  }

  function laneNumber(name) {
    const match = String(name).match(/_(\d+)$/);
    return match ? Number(match[1]) : 9999;
  }

  function shortLaneLabel(name) {
    const kind = laneKind(name);
    const number = laneNumber(name);
    return `${kind} ${number === 9999 ? '' : String(number).padStart(2, '0')}`.trim();
  }

  function taskOpName(name) {
    return String(name || 'task').replace(/\s*\([^)]*\)\s*$/, '').trim() || 'task';
  }

  function makeTask(event, laneName) {
    const opName = taskOpName(event.name);
    const duration = Number(event.dur) || 0;
    const taskId = event.args?.taskId == null ? '' : String(event.args.taskId);
    return {
      id: `${laneName}:${event.ts}:${event.name}:${taskId}`,
      label: opName,
      displayName: event.name,
      rawName: event.name,
      opName,
      laneKind: laneKind(laneName),
      laneId: laneName,
      lane: shortLaneLabel(laneName),
      start: Number(event.ts) || 0,
      end: (Number(event.ts) || 0) + duration,
      duration,
      totalCycle: duration,
      clcCycle: duration,
      gap: 0,
      gapRatio: 0,
      status: 'complete',
      dominantCounter: event.args?.['event-hint'] || '',
      wrapId: taskId ? `task ${taskId}` : '',
      inputRawMagic: [],
      outputRawMagic: [],
      coreId: event.args?.CoreId,
    };
  }

  function fallbackData() {
    const events = [
      ['AIC_0', 'copy_hidden(r2t0)', 36.26, 15],
      ['AIC_0', 'x_gamma0(r2t1)', 52.1, 18],
      ['AIC_0', 'rms_recip(r2t6)', 78.6, 23],
      ['AIC_0', 'q_proj(r2t12)', 129.5, 44],
      ['AIV_24', 'rope_qkv(r2t58)', 124.14, 26],
      ['AIV_24', 'fa_fused(r2t90)', 189.38, 62],
      ['AICPU_0', 'scheduler', 0, 32.58],
    ];
    return {
      source: 'fallback preview',
      events: events.map(([laneName, name, ts, dur]) => makeTask({ name, ts, dur, args: {} }, laneName)),
      totalEvents: events.length,
    };
  }

  function parseTrace(payload) {
    const traceEvents = Array.isArray(payload?.traceEvents) ? payload.traceEvents : [];
    const threadNames = new Map(
      traceEvents
        .filter((event) => event.ph === 'M' && event.name === 'thread_name' && event.args?.name)
        .map((event) => [event.tid, event.args.name]),
    );
    const allEvents = traceEvents
      .filter((event) => event.ph === 'X' && event.cat === 'event' && Number(event.dur) > 0 && event.name !== 'setup')
      .map((event) => ({ event, laneName: threadNames.get(event.tid) || `thread_${event.tid}` }))
      .filter(({ laneName }) => ['AIC', 'AIV', 'AICPU'].includes(laneKind(laneName)));

    const stats = new Map();
    allEvents.forEach(({ laneName }) => {
      const item = stats.get(laneName) || { laneName, count: 0 };
      item.count += 1;
      stats.set(laneName, item);
    });

    const selectedLanes = [];
    ['AIC', 'AIV', 'AICPU'].forEach((kind) => {
      [...stats.values()]
        .filter((item) => laneKind(item.laneName) === kind)
        .sort((a, b) => laneNumber(a.laneName) - laneNumber(b.laneName) || b.count - a.count)
        .slice(0, LANE_LIMITS[kind])
        .forEach((item) => selectedLanes.push(item.laneName));
    });

    const events = allEvents
      .filter(({ laneName }) => selectedLanes.includes(laneName))
      .map(({ event, laneName }) => makeTask(event, laneName))
      .sort((a, b) => a.start - b.start || a.laneId.localeCompare(b.laneId));
    const traceTasks = allEvents.map(({ event, laneName }) => makeTask(event, laneName));
    const min = traceTasks.reduce((value, task) => Math.min(value, task.start), Infinity);
    const max = traceTasks.reduce((value, task) => Math.max(value, task.end), 0);

    return {
      source: '真实 trace · merged_swimlane',
      events,
      totalEvents: traceTasks.length,
      min: Number.isFinite(min) ? min : 0,
      max: max || 1,
      lanes: selectedLanes,
    };
  }

  async function loadData() {
    if (loaded) return model;
    if (loading) return loading;
    loading = (async () => {
      try {
        const response = await fetch(new URL(DATA_URL, document.baseURI));
        if (!response.ok) throw new Error(`trace HTTP ${response.status}`);
        model = parseTrace(await response.json());
      } catch (error) {
        model = fallbackData();
        model.min = 0;
        model.max = Math.max(...model.events.map((task) => task.end), 1);
        model.lanes = [...new Set(model.events.map((task) => task.laneId))];
        console.warn('[Qwen3 performance] real trace unavailable; using fallback preview.', error);
      }
      loaded = true;
      loading = null;
      updateMeta();
      render();
      return model;
    })();
    return loading;
  }

  function updateMeta() {
    const root = panel();
    if (!root || !model) return;
    const source = qs('[data-performance-swimlane-source]', root);
    const stats = qs('[data-performance-swimlane-stats]', root);
    if (source) source.textContent = `${model.source} · ${model.lanes.length} 条代表泳道`;
    if (stats) stats.textContent = `${model.totalEvents.toLocaleString()} tasks · ${(model.max - model.min).toFixed(1)} μs`;
  }

  function resizeCanvas(node, width, height) {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    node.width = Math.max(1, Math.round(width * dpr));
    node.height = Math.max(1, Math.round(height * dpr));
    const context = node.getContext('2d');
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    return context;
  }

  function drawAxis(ctx, x, width, min, max) {
    const axisY = AXIS_HEIGHT - 4;
    const span = Math.max(1, max - min);
    ctx.strokeStyle = cssColor('--border-subtle', 'rgba(255,255,255,.12)');
    ctx.fillStyle = cssColor('--foreground-muted', '#8b929e');
    ctx.font = '500 9px ui-monospace, SFMono-Regular, Consolas, monospace';
    ctx.textBaseline = 'top';
    ctx.textAlign = 'center';
    for (let index = 0; index <= 4; index += 1) {
      const position = x + width * (index / 4);
      ctx.beginPath();
      ctx.moveTo(position, axisY);
      ctx.lineTo(position, axisY + 4);
      ctx.stroke();
      ctx.fillText(`${(min + span * (index / 4)).toFixed(0)} μs`, position, axisY + 5);
    }
  }

  function render() {
    const node = canvas();
    if (!node || !model || panel()?.hidden) return;
    const width = Math.max(280, node.clientWidth || node.parentElement?.clientWidth || 640);
    const height = Math.max(80, node.clientHeight || 157);
    const ctx = resizeCanvas(node, width, height);
    const chartX = LABEL_WIDTH;
    const chartWidth = Math.max(120, width - chartX - 10);
    const rows = model.lanes.map((laneName) => ({
      laneName,
      tasks: model.events.filter((task) => task.laneId === laneName),
    }));
    const min = model.min || 0;
    const max = Math.max(model.max || 1, min + 1);
    const span = max - min;
    const pattern = window.PtoSwimlaneTaskPattern;
    const colormap = pattern?.createTaskColormap?.({
      laneKindColors: { AIC: '#735bb4', AIV: '#4d70ba', AICPU: '#4a9568' },
    });

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = cssColor('--ide-frame-pane-fill', '#111318');
    ctx.fillRect(0, 0, width, height);
    drawAxis(ctx, chartX, chartWidth, min, max);
    hitBoxes = [];

    for (let index = 0; index < rows.length; index += 1) {
      const row = rows[index];
      const y = AXIS_HEIGHT + index * (ROW_HEIGHT + ROW_GAP);
      ctx.fillStyle = index % 2 ? 'rgba(255,255,255,.018)' : 'rgba(255,255,255,.035)';
      ctx.fillRect(0, y, width, ROW_HEIGHT);
      ctx.fillStyle = cssColor('--foreground-secondary', '#c6c9d1');
      ctx.font = '600 10px ui-monospace, SFMono-Regular, Consolas, monospace';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(shortLaneLabel(row.laneName), 5, y + ROW_HEIGHT / 2);

      ctx.strokeStyle = 'rgba(255,255,255,.06)';
      ctx.beginPath();
      ctx.moveTo(chartX, y + ROW_HEIGHT);
      ctx.lineTo(width, y + ROW_HEIGHT);
      ctx.stroke();

      row.tasks.forEach((task) => {
        const x = chartX + ((task.start - min) / span) * chartWidth;
        const barWidth = Math.max(1.6, (task.duration / span) * chartWidth);
        const isSelected = selectedTask?.id === task.id;
        const isRelated = Boolean(selectedTask && selectedTask.opName === task.opName && !isSelected);
        const color = colormap?.colorForTask(task, 'semantic') || (laneKind(task.laneId) === 'AIV' ? '#4d70ba' : laneKind(task.laneId) === 'AICPU' ? '#4a9568' : '#735bb4');
        pattern?.drawTaskBar?.(ctx, {
          x, y: y + 2, width: barWidth, height: ROW_HEIGHT - 4, task,
          baseColor: color, isSelected, isRelated,
          fontFamily: 'ui-monospace, SFMono-Regular, Consolas, monospace',
        });
        hitBoxes.push({ task, x, y: y + 2, width: barWidth, height: ROW_HEIGHT - 4 });
      });
    }
  }

  function hitTest(event) {
    const node = canvas();
    if (!node) return null;
    const rect = node.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    for (let index = hitBoxes.length - 1; index >= 0; index -= 1) {
      const hit = hitBoxes[index];
      if (x >= hit.x && x <= hit.x + hit.width && y >= hit.y && y <= hit.y + hit.height) return hit;
    }
    return null;
  }

  function handlePointerMove(event) {
    const node = canvas();
    const root = panel();
    const tooltip = qs('[data-performance-swimlane-tooltip]', root);
    const hit = hitTest(event);
    if (!hit || !window.PtoSwimlaneTaskPattern) {
      window.PtoSwimlaneTaskPattern?.hideTooltip?.(tooltip);
      if (selectedTask) { selectedTask = null; render(); }
      return;
    }
    if (selectedTask?.id !== hit.task.id) {
      selectedTask = hit.task;
      render();
    }
    window.PtoSwimlaneTaskPattern.showTooltip(tooltip, hit.task, event, { bounds: root, durationUnit: 'μs' });
  }

  function handlePointerLeave() {
    const tooltip = qs('[data-performance-swimlane-tooltip]', panel());
    window.PtoSwimlaneTaskPattern?.hideTooltip?.(tooltip);
    if (selectedTask) { selectedTask = null; render(); }
  }

  function handleClick(event) {
    const hit = hitTest(event);
    if (!hit) return;
    window.dispatchEvent(new CustomEvent('qwen3-performance-select', { detail: { taskName: hit.task.opName, rawName: hit.task.rawName } }));
  }

  function toggle(event) {
    const root = panel();
    const button = event.target.closest('[data-performance-swimlane-toggle]');
    if (!root || !button) return;
    const collapsed = root.classList.toggle('is-collapsed');
    button.setAttribute('aria-expanded', String(!collapsed));
    button.setAttribute('title', collapsed ? '展开性能泳道图' : '收起性能泳道图');
    requestAnimationFrame(() => {
      render();
      window.PtoQwen3ModelViz?.fit?.();
    });
  }

  function init() {
    if (initialized) return;
    const node = canvas();
    const root = panel();
    if (!node || !root) return;
    root.addEventListener('click', toggle);
    node.addEventListener('pointermove', handlePointerMove);
    node.addEventListener('pointerleave', handlePointerLeave);
    node.addEventListener('click', handleClick);
    resizeObserver = typeof ResizeObserver === 'function' ? new ResizeObserver(() => render()) : null;
    resizeObserver?.observe(node.parentElement || node);
    window.addEventListener('qwen3-graph-selection', (event) => {
      const label = String(event.detail?.label || '').toLowerCase();
      const task = model?.events.find((item) => label.includes(item.opName.toLowerCase()));
      if (task) { selectedTask = task; render(); }
    });
    initialized = true;
  }

  function show() {
    const root = panel();
    if (!root) return;
    init();
    root.hidden = false;
    updateMeta();
    requestAnimationFrame(render);
    loadData();
  }

  function hide() {
    const root = panel();
    if (root) root.hidden = true;
  }

  window.PtoQwen3PerformanceSwimlane = { show, hide, render, loadData };
})();
