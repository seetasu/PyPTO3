/* Task assignment, logical tiling and software pipeline for paged_attention_pypto.py.
 * Source constants: 17–32; task / page / stack mapping: 261–269.
 * Pipeline source: embedded paged-attention-pypto-source.js, PRE_LAUNCH loop and READY events.
 * Sequence lengths are editable scenario inputs, not captured runtime values.
 */
(function (global) {
  'use strict';
  const constants = Object.freeze({ batch: 16, kvHeads: 8, qPerKv: 5, blocks: 24, pageTokens: 128, pagesPerStack: 4, maxSeq: 3584, preLaunch: 2, transferSlots: 3 });
  const defaultSeqLens = Object.freeze([1025, 512, 2048, 128, 1536, 3584, 3072, 256, 769, 1024, 2049, 640, 2560, 129, 1792, 3328]);
  function taskInfo(id) {
    const kv = id % constants.kvHeads;
    return { id, batch: Math.floor(id / constants.kvHeads), kv, block: id % constants.blocks,
      qStart: kv * constants.qPerKv, qEnd: (kv + 1) * constants.qPerKv - 1 };
  }
  function tasksForBlock(block) {
    return Array.from({ length: Math.ceil((constants.batch * constants.kvHeads - block) / constants.blocks) }, (_, i) => block + i * constants.blocks);
  }
  function partition(seqLen) {
    const pageCount = Math.ceil(seqLen / constants.pageTokens);
    const stackCount = Math.ceil(pageCount / constants.pagesPerStack);
    return { seqLen, pageCount, stackCount, stacks: Array.from({ length: stackCount }, (_, id) => ({ id,
      pages: Array.from({ length: Math.min(constants.pagesPerStack, pageCount - id * constants.pagesPerStack) }, (_, i) => {
        const pageId = id * constants.pagesPerStack + i;
        return { id: pageId, validTokens: Math.min(constants.pageTokens, seqLen - pageId * constants.pageTokens) };
      }),
    })) };
  }
  function pipelineFor(seqLen) {
    const tiling = partition(seqLen);
    const stepCount = tiling.stackCount + constants.preLaunch;
    return { ...tiling, stepCount,
      steps: Array.from({ length: stepCount }, (_, tick) => ({ tick,
        produceStack: tick < tiling.stackCount ? tick : null,
        consumeStack: tick >= constants.preLaunch ? tick - constants.preLaunch : null })),
      stacks: tiling.stacks.map((stack) => ({ ...stack, slot: stack.id % constants.transferSlots,
        tokenStart: stack.id * constants.pagesPerStack * constants.pageTokens,
        tokenEnd: Math.min(seqLen, (stack.id + 1) * constants.pagesPerStack * constants.pageTokens) - 1,
        previousStack: stack.id >= constants.transferSlots ? stack.id - constants.transferSlots : null })) };
  }
  // Reconcile the single context after a scenario edit; valid instances keep their tick.
  function reconcileSelection(data, selection) {
    const producing = ['qk', 'softmax'].includes(selection.stage);
    if (selection.stack !== null) {
      const stack = Math.min(selection.stack, data.stackCount - 1);
      const step = data.steps.find((item) => (producing ? item.produceStack : item.consumeStack) === stack);
      return { ...selection, stack, pipelineStep: step.tick };
    }
    const pipelineStep = Math.max(0, Math.min(selection.pipelineStep, data.stepCount - 1));
    const step = data.steps[pipelineStep];
    return { ...selection, pipelineStep, stack: producing ? step.produceStack : step.consumeStack };
  }
  // 详情视图只放当前 Task 与它的左右邻居（同一 KV head、相邻 Batch），不铺满整个矩阵。
  function taskNeighborhood(id, seqLens, taskView = state.taskView) {
    const active = taskInfo(id);
    const height = taskView === 'pipeline' ? 410 : Math.max(260, 164 + partition(seqLens[active.batch]).stackCount * 60);
    const gap = 24, sideWidth = 300, sideHeight = 200;
    const leftId = active.batch > 0 ? id - constants.kvHeads : null;
    const rightId = active.batch < constants.batch - 1 ? id + constants.kvHeads : null;
    const sideY = (height - sideHeight) / 2;
    const rects = [];
    let x = 0;
    if (leftId !== null) { rects.push({ id: leftId, x, y: sideY, w: sideWidth, h: sideHeight }); x += sideWidth + gap; }
    rects.push({ id, x, y: 0, w: 840, h: height });
    x += 840 + gap;
    if (rightId !== null) rects.push({ id: rightId, x, y: sideY, w: sideWidth, h: sideHeight });
    return rects;
  }
  // Kept in this page session; mounting/unmounting does not reset exploration.
  const state = { pipelineMode: 'schedule', stackDetailOpen: false, stage: 'qk', taskView: 'pipeline', pipelineStep: 2, pipelineStack: null, seqLens: [...defaultSeqLens], block: null, task: null, level: 'matrix', view: null, matrixView: null, matrixSize: null, size: null };
  // A single selection owner: the compute map reads/writes this same page-session state.
  const stageNodes = Object.freeze({ qk: 'qk', softmax: 'softmax', pv: 'pv', accumulate: 'context-acc' });
  const stageLabels = Object.freeze({ qk: 'QK', softmax: 'Softmax', pv: 'PV', accumulate: 'Accumulate' });
  const listeners = new Set();
  let publishedScenario = '';
  function publishScenario() {
    const snapshot = JSON.stringify([context.selection, state.seqLens]);
    if (snapshot === publishedScenario) return;
    publishedScenario = snapshot;
    listeners.forEach((listener) => listener(context.scenario));
  }
  const context = {
    get scenario() { return { ...this.selection, seqLens: [...state.seqLens] }; },
    subscribe(listener, { signal } = {}) {
      if (signal?.aborted) return () => {};
      listeners.add(listener);
      const unsubscribe = () => { listeners.delete(listener); signal?.removeEventListener('abort', unsubscribe); };
      signal?.addEventListener('abort', unsubscribe, { once: true });
      return unsubscribe;
    },
    setLength(value) {
      if (state.task === null || !Number.isInteger(value) || value < 1 || value > constants.maxSeq) return false;
      if (state.seqLens[taskInfo(state.task).batch] === value) return true;
      state.seqLens[taskInfo(state.task).batch] = value;
      const selection = reconcileSelection(pipelineFor(value), this.selection);
      state.pipelineStack = selection.stack; state.pipelineStep = selection.pipelineStep;
      mounted?.refresh?.(); publishScenario(); return true;
    },
    stageNodes, stageLabels,
    get selection() {
      const task = state.task === null ? null : taskInfo(state.task);
      return { block: state.block, task: state.task, batch: task?.batch ?? null, kvHead: task?.kv ?? null,
        pipelineStep: state.pipelineStep, stack: state.pipelineStack, stage: state.stage,
        seqLen: task ? state.seqLens[task.batch] : null, pipelineMode: state.pipelineMode };
    },
    selectInstance({ stack, stage }) {
      if (state.task === null || !Object.hasOwn(stageNodes, stage)) return;
      const data = pipelineFor(state.seqLens[taskInfo(state.task).batch]);
      if (!Number.isInteger(stack) || stack < 0 || stack >= data.stackCount) return;
      state.pipelineStack = stack; state.stage = stage;
      state.pipelineStep = data.steps.find((step) => (['qk', 'softmax'].includes(stage) ? step.produceStack : step.consumeStack) === stack).tick;
      mounted?.refresh?.(); publishScenario();
    },
    selectStage(stage) {
      if (!Object.hasOwn(stageNodes, stage)) return;
      if (state.pipelineStack !== null) { this.selectInstance({ stack: state.pipelineStack, stage }); return; }
      state.stage = stage;
      if (state.task !== null && state.pipelineStack === null) {
        const data = pipelineFor(state.seqLens[taskInfo(state.task).batch]);
        const step = data.steps[Math.min(state.pipelineStep, data.stepCount - 1)];
        state.pipelineStack = ['qk', 'softmax'].includes(stage) ? step.produceStack : step.consumeStack;
      }
      mounted?.refresh?.(); publishScenario();
    },
  };
  global.PtoPagedAttentionContext = context;
  const grid = { x: 94, y: 106, dx: 62, dy: 34, w: 54, h: 28, width: 1096, height: 388 };
  // Playback belongs to this page session. Selection stops it until reload,
  // including when the inspector or its tabs are mounted again.
  const playback = { stopped: false, block: 0 };
  let mounted = null;

  function mount(host) {
    mounted?.abort();
    const lifetime = new AbortController();
    const { signal } = lifetime;
    const on = (element, type, listener, options = {}) => element.addEventListener(type, listener, { ...options, signal });
    const query = (selector) => host.querySelector(selector);
    const reducedMotion = global.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let playbackTimer = null, dependencyController = null;
    let hoverTask = null, drag = null, animation = null, transitionUntil = 0, panTimer = null, suppressClick = false;
    host.innerHTML = `<div class="kf-pa-map">
      <section class="kf-pa-map__shelf" aria-label="SPMD Block 分配">
        <div class="kf-pa-map__shelf-label"><b>Block 分配</b><span>点击查看分配 · stride ${constants.blocks}</span></div>
        <div class="kf-pa-map__blocks">${Array.from({ length: 3 }, (_, group) => `<div class="kf-pa-map__block-group">${Array.from({ length: 8 }, (_, i) => {
          const id = group * 8 + i;
          return `<button class="btn btn-ghost kf-pa-map__block" type="button" data-pa-block="${id}" aria-label="SPMD Block ${id}" aria-pressed="false" title="Block ${id} · ${tasksForBlock(id).length} Tasks"><svg viewBox="0 0 28 24" aria-hidden="true"><path d="M4 13 14 18 24 13M4 18 14 23 24 18"/><path class="kf-pa-map__block-face" d="m4 7 10-5 10 5-10 5Z"/></svg><span>${String(id).padStart(2, '0')}</span></button>`;
        }).join('')}</div>`).join('')}</div>
        <p class="kf-pa-map__assignment" aria-live="polite"></p>
      </section>
      <div class="kf-pa-map__context"><div class="kf-pa-map__breadcrumb"><button class="btn btn-ghost btn-sm" type="button" data-pa-action="back" hidden>← 返回矩阵</button><b data-pa-context>Batch × KV Head</b></div><button class="btn btn-ghost btn-sm" type="button" data-pa-action="enter">进入 Task 0 ↗</button><span class="kf-pa-map__detail-meta" data-pa-meta></span></div>
      <div class="kf-pa-map__config" hidden><label>场景配置 · <span data-pa-batch></span> seq_lens <input class="input" type="number" min="1" max="3584" step="1" aria-label="当前 Batch 的序列长度" data-pa-length></label><input type="range" min="1" max="3584" step="1" aria-label="调节当前 Batch 的序列长度" data-pa-range><span>tokens</span></div>
      <div class="kf-pa-map__pipeline-controls" data-pa-controls hidden>
        <div role="group" aria-label="Task 详情视图"><button class="btn btn-ghost btn-sm" data-pa-view="pipeline">Pipeline</button><button class="btn btn-ghost btn-sm" data-pa-view="pages">Pages</button></div>
        <div data-pa-steps role="group" aria-label="Pipeline Step"></div>
        <div data-pa-modes role="group" aria-label="Pipeline 模式"><button class="btn btn-ghost btn-sm" data-pa-mode="schedule">调度</button><button class="btn btn-ghost btn-sm" data-pa-mode="analysis">依赖分析</button></div><span data-pa-pipeline-summary title="源码静态调度，不是运行 Trace"></span>
      </div>
      <div class="kf-pa-map__viewport" tabindex="0" role="group" aria-label="任务矩阵画布，滚轮缩放，按住左键拖拽平移，方向键切换 Task，Escape 返回矩阵">
        <svg class="kf-pa-map__svg" xmlns="http://www.w3.org/2000/svg" aria-label="Batch 与 KV Head 的任务关系"><g data-pa-ghost opacity="0" pointer-events="none"></g><g data-pa-morph-fill opacity="0" pointer-events="none"></g><g class="kf-pa-map__scene" data-pa-scene></g><g data-pa-morph opacity="0" pointer-events="none"></g></svg>
        <div class="kf-pa-map__tooltip" role="tooltip" hidden></div>
      </div>
      <section data-pa-analysis hidden aria-label="当前 Pipeline 的依赖解释"></section>
      <div class="kf-pa-map__pipeline-detail" data-pa-detail hidden aria-live="polite"></div>
      <section class="kf-pa-map__operator" data-pa-operator hidden aria-label="Where in Operator"></section>
      <footer class="kf-pa-map__footer"><div><p data-pa-summary></p><span data-pa-hint>滚轮放大进入 Task · 拖拽平移</span></div><div class="kf-pa-map__tools" role="group" aria-label="简介画布操作"><button class="btn btn-ghost btn-sm" type="button" data-pa-action="out" aria-label="缩小简介">−</button><output data-pa-zoom>100%</output><button class="btn btn-ghost btn-sm" type="button" data-pa-action="in" aria-label="放大简介">＋</button><button class="btn btn-ghost btn-sm" type="button" data-pa-action="fit">适应</button></div></footer>
    </div>`;
    const viewport = query('.kf-pa-map__viewport');
    const scene = query('[data-pa-scene]');
    const ghost = query('[data-pa-ghost]');
    // 入场时的共享元素：morph-fill 是压在底板之下、场景之上的色块（盖住矩阵格子），
    // morph 是浮在场景之上的边框与编号。拆成两层，卡片正文才不会被自己的底色挡住。
    const morph = query('[data-pa-morph]');
    const morphFill = query('[data-pa-morph-fill]');
    const tooltip = query('.kf-pa-map__tooltip');
    const currentTiling = () => partition(state.seqLens[taskInfo(state.task ?? 0).batch]);
    const neighborhood = () => taskNeighborhood(state.task ?? 0, state.seqLens);
    const activeRect = () => neighborhood().find((rect) => rect.id === state.task) || neighborhood()[0];
    const extent = () => {
      if (state.level === 'matrix') return { width: grid.width, height: grid.height };
      const rects = neighborhood();
      const maxX = Math.max(...rects.map((rect) => rect.x + rect.w));
      const maxY = Math.max(...rects.map((rect) => rect.y + rect.h));
      return { width: maxX + 240, height: maxY + 160 };
    };
    const fitted = () => {
      const box = extent();
      const fitBox = state.level === 'detail' ? { width: activeRect().w + 64, height: activeRect().h + 48 } : box;
      const scale = Math.max(0.05, Math.min((viewport.clientWidth - 32) / fitBox.width, (viewport.clientHeight - 24) / fitBox.height, 1.2));
      const center = state.level === 'matrix' ? { x: box.width / 2, y: box.height / 2 }
        : { x: activeRect().x + activeRect().w / 2, y: activeRect().y + activeRect().h / 2 };
      return { scale, x: viewport.clientWidth / 2 - center.x * scale, y: viewport.clientHeight / 2 - center.y * scale };
    };
    const cellRect = (id) => { const t = taskInfo(id); return { x: grid.x + t.batch * grid.dx, y: grid.y + t.kv * grid.dy, w: grid.w, h: grid.h }; };
    const taskDescription = (id) => {
      const t = taskInfo(id);
      return `Task ${id} · Batch ${t.batch} · KV ${t.kv} · Q${t.qStart}–Q${t.qEnd} · Block ${t.block} · ${state.seqLens[t.batch]} tokens`;
    };
    function paintView() {
      if (!state.view) return;
      const { x, y, scale } = state.view;
      scene.setAttribute('transform', `translate(${x} ${y}) scale(${scale})`);
      // Data labels stay legible instead of shrinking with the scene.
      scene.style.setProperty('--pa-svg-label', `${Math.max(12, 11 / scale)}px`);
      scene.style.setProperty('--pa-task-title', `${Math.max(30, 20 / scale)}px`);
      scene.style.setProperty('--pa-neighbor-title', `${Math.max(18, 12 / scale)}px`);
      scene.classList.toggle('is-compact-detail', state.level === 'detail' && scale < 0.65);
      scene.classList.toggle('is-distant', state.level === 'matrix' && grid.dx * scale < 34);
      scene.classList.toggle('is-tiny', state.level === 'matrix' && grid.w * scale < 25);
      query('[data-pa-zoom]').textContent = `${Math.round(scale / fitted().scale * 100)}%`;
      host.dataset.level = state.level;
    }
    function animateFrom(start) {
      animation?.cancel();
      if (reducedMotion || !start) return;
      const end = state.view;
      animation = scene.animate([
        { transform: `translate(${start.x}px, ${start.y}px) scale(${start.scale})`, opacity: 0.35 },
        { transform: `translate(${end.x}px, ${end.y}px) scale(${end.scale})`, opacity: 1 },
      ], { duration: 320, easing: 'cubic-bezier(.2,.8,.2,1)' });
    }
    function sequenceGraphic(len, x, y, width, height = 8) {
      const gap = width > 100 ? 4 : 1.6;
      const segment = (width - gap * 6) / 7;
      return `<g class="kf-pa-map__sequence" aria-label="seq_lens ${len} tokens，每段 512 tokens"><title>${len} / 3584 tokens · 每段 512 tokens</title>${Array.from({ length: 7 }, (_, i) => {
        const fill = Math.max(0, Math.min(1, (len - i * 512) / 512));
        return `<rect class="kf-pa-map__seq-track" x="${x + i * (segment + gap)}" y="${y}" width="${segment}" height="${height}" rx="1"/>${fill ? `<rect class="kf-pa-map__seq-fill" x="${x + i * (segment + gap)}" y="${y}" width="${segment * fill}" height="${height}" rx="1"/>` : ''}`;
      }).join('')}</g>`;
    }
    function drawMatrix() {
      scene.innerHTML = `<text class="kf-pa-map__axis" x="${grid.x}" y="18">BATCH</text><text class="kf-pa-map__axis" x="8" y="75">seq_lens</text>
        ${state.seqLens.map((len, batch) => `<g data-pa-column="${batch}"><text class="kf-pa-map__batch" x="${grid.x + batch * grid.dx + grid.w / 2}" y="44">${batch}</text>${sequenceGraphic(len, grid.x + batch * grid.dx, 59, grid.w, 12)}<text class="kf-pa-map__seq" x="${grid.x + batch * grid.dx + grid.w / 2}" y="90">${len}</text></g>`).join('')}
        ${Array.from({ length: 8 }, (_, kv) => `<text class="kf-pa-map__row" x="8" y="${grid.y + kv * grid.dy + 19}">KV ${kv}</text>`).join('')}
        ${Array.from({ length: 128 }, (_, id) => {
          const r = cellRect(id);
          return `<g class="kf-pa-map__task" data-pa-task="${id}" transform="translate(${r.x} ${r.y})" tabindex="${id === (state.task ?? 0) ? 0 : -1}" role="button" aria-label="${taskDescription(id)}" aria-pressed="false"><rect class="kf-pa-map__cell" width="${r.w}" height="${r.h}" rx="4"/>${Array.from({ length: 5 }, (_, q) => `<rect class="kf-pa-map__q" x="${5 + q * 9}" y="4" width="7" height="20" rx="1"/>`).join('')}<text class="kf-pa-map__task-id" x="27" y="19">${id}</text></g>`;
        }).join('')}`;
    }
    // 详情视图的框：坐标由调用方给出（稳态=布局值，切换中=补间值）。
    // 内容按框宽在「紧凑 / 展开」两套之间交叉淡入，展开内容裁剪在框内，
    // 这样切换时框从右往左滑入、逐渐变大，也不会让 Stack/Page 溢出到框外。
    function frameCompactMarkup(id, rect, active, len) {
      const t = taskInfo(id);
      const x = t.batch < active.batch ? rect.w - 192 : t.batch > active.batch ? 20 : rect.w / 2 - 86;
      const y = rect.h / 2 - 26;
      return `<text class="kf-pa-map__neighbor-title" x="${x}" y="${y}">Task ${id}</text>${sequenceGraphic(len, x, y + 12, 168, 9)}<text class="kf-pa-map__neighbor-length" x="${x}" y="${y + 45}">${len} tokens</text>`;
    }
    function frameExpandedMarkup(id, len) {
      const t = taskInfo(id);
      const tiling = partition(len);
      // 每个 stack 恰好吐一组 <g>；正文/Stack/Page 全部落在 frame-body 里，
      // 这样裁剪窗口和透明度接管才能盖住整块正文（多一个 </g> 就会把后面的 Stack 甩到组外）。
      const stacks = tiling.stacks.map((stack) => {
        const pages = Array.from({ length: 4 }, (_, i) => {
          const page = stack.pages[i], x = 124 + i * 166;
          if (!page) return `<g aria-hidden="true"><rect class="kf-pa-map__unused" x="${x}" y="4" width="150" height="44" rx="3"/><text class="kf-pa-map__unused-label" x="${x + 75}" y="29">—</text></g>`;
          return `<g data-pa-page="${page.id}" aria-label="Logical Page ${page.id}，${page.validTokens}/128 有效 tokens"><title>Page ${page.id} · ${page.validTokens}/128 tokens</title><rect class="kf-pa-map__page" x="${x}" y="4" width="150" height="44" rx="3"/><text class="kf-pa-map__page-label" x="${x + 10}" y="25">P${page.id}</text><text class="kf-pa-map__page-value" text-anchor="end" x="${x + 140}" y="25">${page.validTokens}/128</text><rect class="kf-pa-map__page-track" x="${x + 10}" y="34" width="130" height="6" rx="1"/><rect class="kf-pa-map__page-fill${page.validTokens < 128 ? ' is-tail' : ''}" x="${x + 10}" y="34" width="${130 * page.validTokens / 128}" height="6" rx="1"/></g>`;
        }).join('');
        return `<g transform="translate(24 ${140 + stack.id * 60})" data-pa-stack="${stack.id}" role="button" tabindex="0" aria-label="Stack ${stack.id}，${stack.pages.length} Pages"><rect class="kf-pa-map__stack${stack.id === state.pipelineStack ? ' is-stack-selected' : ''}" width="792" height="52" rx="5"/><text class="kf-pa-map__stack-label" x="14" y="30">S${stack.id}</text>${pages}</g>`;
      }).join('');
      return [
        `<text class="kf-pa-map__frame-title" x="24" y="48">Task ${id}</text>`,
        '<g class="kf-pa-map__frame-body">',
        `<text class="kf-pa-map__frame-meta" x="24" y="80">Batch ${t.batch} · KV ${t.kv} · Q${t.qStart}–Q${t.qEnd}</text>`,
        sequenceGraphic(len, 24, 102, 552, 12),
        `<text class="kf-pa-map__frame-length" x="610" y="114">${len} tokens</text>`,
        state.taskView === 'pipeline' ? pipelineMarkup(len, id === state.task) : stacks,
        '</g>',
      ].join('');
    }
    function pipelineMarkup(len, activeTask) {
      const data = pipelineFor(len), step = data.steps[Math.min(state.pipelineStep, data.stepCount - 1)];
      const x0 = 230, dx = 62;
      const analysis = activeTask && state.pipelineMode === 'analysis';
      const focused = analysis ? dependencyController?.focusedStacks || [] : [];
      const label = (text, x, y) => `<text class="kf-pa-map__detail-label" x="${x}" y="${y}">${text}</text>`;
      const mark = (stackId, x, y, active) => `<g class="kf-pa-map__pipeline-mark${analysis && !focused.includes(stackId) ? ' is-analysis-muted' : ''}${analysis && focused.includes(stackId) ? ' is-analysis-related' : ''}${active && stackId !== null ? ' is-active' : ''}${stackId === state.pipelineStack ? ' is-stack-selected' : ''}"${stackId !== null ? ` data-pa-stack="${stackId}" role="button" tabindex="0" aria-label="Stack ${stackId}"` : ''}><rect x="${x}" y="${y}" width="54" height="38" rx="3"/><text x="${x + 27}" y="${y + 24}" text-anchor="middle">${stackId === null ? '—' : 'S' + stackId}</text></g>`;
      const stageMark = (stage, x, y, stack) => `<g class="kf-pa-map__stage${state.stage === stage ? ' is-selected' : ''}" data-pa-stage="${stage}" data-pa-stage-stack="${stack ?? ''}" role="button" tabindex="${stack === null ? -1 : 0}" aria-disabled="${stack === null}" aria-pressed="${state.stage === stage}" aria-label="${stageLabels[stage]} · ${stack === null ? '当前 Step 无此阶段' : 'Stack ' + stack}"><rect x="${x - 4}" y="${y - 19}" width="${stage === 'accumulate' ? 100 : 76}" height="28" rx="3"/>${label(stageLabels[stage], x, y)}</g>`;
      return label('Pipeline Step →', 24, 154) + label('PRODUCE', 24, 220)
        + stageMark('qk', 24, 244, step.produceStack) + stageMark('softmax', 112, 244, step.produceStack)
        + label('CONSUME', 24, 282) + stageMark('pv', 24, 306, step.consumeStack) + stageMark('accumulate', 112, 306, step.consumeStack)
        + data.steps.map((item) => `<g opacity="${analysis || item.tick === step.tick ? 1 : .4}">${label(item.tick, x0 + item.tick * dx + 23, 154)}${mark(item.produceStack, x0 + item.tick * dx, 208, item.tick === step.tick)}${mark(item.consumeStack, x0 + item.tick * dx, 270, item.tick === step.tick)}</g>`).join('')
        + `<path class="kf-pa-map__pipeline-offset" d="M${x0 + 27} 188 v-12 h${dx * constants.preLaunch} v12"/>`
        + label(`PRE_LAUNCH = ${constants.preLaunch}`, x0 + 30, 172)
        + label('Stack → Slot', 24, 346) + label('stackId % 3', 24, 380)
        + data.stacks.map((stack) => mark(stack.id, x0 + stack.id * dx, 324, [step.produceStack, step.consumeStack, state.pipelineStack].includes(stack.id)) + `<text class="kf-pa-map__slot${stack.id === state.pipelineStack ? ' is-selected' : ''}${[step.produceStack, step.consumeStack, state.pipelineStack].filter((id) => id !== null && id < data.stackCount).map((id) => data.stacks[id].slot).includes(stack.slot) ? ' is-active' : ''}" x="${x0 + stack.id * dx + 27}" y="380" text-anchor="middle">${stack.slot}</text>`).join('')
        + (analysis ? dependencyController?.overlay() || '' : '');
    }
    function syncPipeline() {
      const detail = state.level === 'detail';
      query('[data-pa-controls]').hidden = !detail;
      query('[data-pa-detail]').hidden = !detail || !state.stackDetailOpen;
      const dependencies = detail && state.taskView === 'pipeline' && state.pipelineMode === 'analysis';
      query('[data-pa-operator]').hidden = !detail || dependencies;
      host.dataset.pipelineMode = dependencies ? 'analysis' : 'schedule';
      query('[data-pa-analysis]').hidden = !dependencies;
      query('[data-pa-modes]').hidden = state.taskView !== 'pipeline';
      host.querySelectorAll('[data-pa-mode]').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.paMode === state.pipelineMode)));
      viewport.hidden = false;
      query('.kf-pa-map__footer').hidden = false;
      if (!detail) return;
      const data = pipelineFor(state.seqLens[taskInfo(state.task).batch]);
      query('[data-pa-pipeline-summary]').textContent = `${data.stackCount} Stacks · ${data.stepCount} Steps · ${constants.transferSlots} Slots`;
      state.pipelineStep = Math.max(0, Math.min(state.pipelineStep, data.stepCount - 1));
      if (state.pipelineStack >= data.stackCount) state.pipelineStack = null;
      const step = data.steps[state.pipelineStep];
      if (state.pipelineStack === null) state.pipelineStack = ['qk', 'softmax'].includes(state.stage) ? step.produceStack : step.consumeStack;
      if (state.taskView === 'pipeline') {
        const options = { context, pipeline: data, synchronizations: global.PTO_PYPA_SEMANTIC_GRAPH.synchronizations,
          onChange() { drawDetail(); paintView(); },
          onNavigate(target, line) { host.dispatchEvent(new CustomEvent('pto-pa-navigate', { bubbles: true, detail: { target, nodeId: stageNodes[state.stage], ...(line ? { sourceLine: line } : {}) } })); } };
        if (!dependencyController) dependencyController = global.PtoPagedAttentionStackDependency.mount(query('[data-pa-analysis]'), options);
        else dependencyController.update(data);
      }
      const c = context.selection;
      const stages = [
        ['prep', 'QKV Prep', ''], ['cache', 'Cache Append', ''],
        ['qk', 'QK', 'AIC'], ['softmax', 'Softmax', 'AIV'],
        ['pv', 'PV', 'AIC'], ['accumulate', 'Acc', 'AIV'], ['output', 'Output', ''],
      ];
      const events = { qk: 'QK_READY', softmax: 'SOFTMAX_READY', pv: 'PV_READY' };
      query('[data-pa-operator]').innerHTML = `<header class="kf-pa-map__operator-head"><span class="kf-eyebrow">WHERE IN OPERATOR</span><span class="kf-pa-map__operator-context">Task ${c.task}${c.stack === null ? '' : ` · Stack ${c.stack}`}</span><div class="kf-pa-map__operator-actions"><button class="btn btn-ghost btn-sm" type="button" data-pa-action="compute">Open full map ↗</button><button class="btn btn-ghost btn-sm" type="button" data-pa-action="source">View source ↗</button></div></header>
        <div class="kf-pa-map__mini-scroll"><nav class="kf-pa-map__mini-map" aria-label="算子主计算链">${stages.map(([stage, label, hardware], index) => {
          const selectable = Object.hasOwn(stageNodes, stage), selected = state.stage === stage;
          const body = `<span>${label}</span>${hardware ? `<small>${hardware}</small>` : ''}`;
          const node = `<div class="kf-pa-map__mini-stage">${selectable ? `<button class="btn btn-ghost kf-pa-map__mini-node" type="button" data-pa-stage="${stage}" aria-pressed="${selected}">${body}</button>` : `<div class="kf-pa-map__mini-node">${body}</div>`}<small class="kf-pa-map__mini-current" ${selected ? '' : 'hidden'}>CURRENT</small></div>`;
          return node + (index < stages.length - 1 ? `<div class="kf-pa-map__mini-link${events[stage] ? ' has-event' : ''}" aria-label="${events[stage] || '计算依赖'}">${events[stage] ? `<span>${events[stage]}</span>` : ''}<i aria-hidden="true"></i></div>` : '');
        }).join('')}</nav></div>`;
      host.querySelectorAll('[data-pa-view]').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.paView === state.taskView)));
      query('[data-pa-steps]').hidden = state.taskView !== 'pipeline';
      query('[data-pa-steps]').innerHTML = `<span>Pipeline Step</span><button class="btn btn-ghost btn-sm" data-pa-step="${step.tick - 1}" aria-label="上一个 Pipeline Step" ${step.tick === 0 ? 'disabled' : ''}>‹</button>`
        + data.steps.map((item) => `<button class="btn btn-ghost btn-sm" data-pa-step="${item.tick}" aria-pressed="${item.tick === step.tick}">${item.tick}</button>`).join('')
        + `<button class="btn btn-ghost btn-sm" data-pa-step="${step.tick + 1}" aria-label="下一个 Pipeline Step" ${step.tick === data.stepCount - 1 ? 'disabled' : ''}>›</button>`;
      const stack = data.stacks[state.pipelineStack];
      query('[data-pa-detail]').hidden = dependencies || !state.stackDetailOpen || !stack;
      query('[data-pa-detail]').innerHTML = stack && state.stackDetailOpen ? `<header><b>STACK ${stack.id}</b><button class="btn btn-ghost btn-sm" type="button" data-pa-action="close-stack" aria-label="关闭 Stack 详情">×</button></header>
        <div class="kf-pa-map__stack-data"><div class="kf-pa-map__stack-pages"><span>Pages</span><div>${Array.from({ length: constants.pagesPerStack }, (_, i) => {
          const page = stack.pages[i];
          return page ? `<div class="kf-pa-map__page-cell" title="Page ${page.id} · ${page.validTokens}/128 tokens" aria-label="Page ${page.id}，${page.validTokens}/128 有效 tokens"><i style="width:${page.validTokens / constants.pageTokens * 100}%"></i><b>P${page.id}</b></div>` : '<div class="kf-pa-map__page-cell is-empty" aria-label="无 Page">—</div>';
        }).join('')}</div></div><div class="kf-pa-map__stack-tokens"><span>Tokens</span><div><b>${stack.tokenStart}</b><i aria-hidden="true"></i><b>${stack.tokenEnd}</b></div></div><div class="kf-pa-map__stack-slot"><span>Slot</span><div><b>${stack.slot}</b>${stack.previousStack === null ? '' : `<small>reused from S${stack.previousStack}</small>`}</div></div></div>` : '';
      publishScenario();
    }
    function selectStack(id) {
      state.stackDetailOpen = !(state.pipelineStack === id && state.stackDetailOpen);
      context.selectInstance({ stack: id, stage: state.stage });
    }
    function selectStep(value) {
      const data = pipelineFor(state.seqLens[taskInfo(state.task).batch]);
      state.pipelineStep = Math.max(0, Math.min(value, data.stepCount - 1));
      state.pipelineStack = null;
      state.stackDetailOpen = false;
      syncPipeline(); drawDetail(); paintView();
    }
    function paintDetailFrames(frames) {
      const active = taskInfo(state.task ?? 0);
      const defs = frames.map((f) => `<clipPath id="pa-frame-clip-${f.id}"><rect width="${Math.max(0, f.w).toFixed(2)}" height="${Math.max(0, f.h).toFixed(2)}"/></clipPath>`).join('');
      scene.innerHTML = `<defs>${defs}</defs>${frames.map((f) => {
        const selected = f.id === state.task;
        const len = state.seqLens[taskInfo(f.id).batch];
        // 300 宽 = 纯紧凑卡片，840 宽 = 纯展开内容；中间宽度按比例交叉淡入。
        const big = Math.max(0, Math.min(1, (f.w - 560) / 220));
        const opacity = f.opacity == null ? 1 : f.opacity;
        const w = Math.max(0, f.w).toFixed(2), h = Math.max(0, f.h).toFixed(2);
        // 完全透明的那一套必须 display:none——否则它照样参与包围盒，
        // 点击坐标会落到卡片外面（Playwright 之类按 bbox 中心点击就会点空）。
        const compactGroup = `<g pointer-events="none" opacity="${(1 - big).toFixed(3)}"${(1 - big) === 0 ? ' display="none"' : ''}>${frameCompactMarkup(f.id, f, active, len)}</g>`;
        const expandedGroup = `<g clip-path="url(#pa-frame-clip-${f.id})" opacity="${big.toFixed(3)}"${big === 0 ? ' display="none"' : ''}>${frameExpandedMarkup(f.id, len)}</g>`;
        return `<g class="kf-pa-map__task-frame${selected ? ' is-expanded' : ''}" data-pa-task="${f.id}" transform="translate(${f.x.toFixed(2)} ${f.y.toFixed(2)})" tabindex="${selected ? 0 : -1}" role="button" aria-label="${taskDescription(f.id)}" aria-pressed="${selected}" opacity="${opacity.toFixed(3)}"><rect class="kf-pa-map__task-boundary" width="${w}" height="${h}" rx="10"/>${compactGroup}${expandedGroup}</g>`;
      }).join('')}`;
    }
    function drawDetail() {
      syncPipeline();
      paintDetailFrames(neighborhood().map((r) => ({ id: r.id, x: r.x, y: r.y, w: r.w, h: r.h, opacity: 1 })));
    }
    function paintPlayback() {
      const playing = !playback.stopped && state.level === 'matrix';
      host.dataset.autoplay = playing ? 'playing' : 'stopped';
      if (playing) host.dataset.playbackBlock = String(playback.block);
      else delete host.dataset.playbackBlock;
      const strengthFor = (block) => {
        const distance = Math.abs(block - playback.block);
        const wrappedDistance = Math.min(distance, constants.blocks - distance);
        return wrappedDistance < 5 ? (5 - wrappedDistance) / 5 : 0;
      };
      host.querySelectorAll('[data-pa-block], [data-pa-task]').forEach((element) => {
        const block = element.hasAttribute('data-pa-block')
          ? Number(element.dataset.paBlock) : taskInfo(Number(element.dataset.paTask)).block;
        element.classList.toggle('is-playing', playing);
        element.classList.toggle('is-playback-current', playing && block === playback.block);
        if (playing) element.style.setProperty('--pa-playback-strength', strengthFor(block));
        else element.style.removeProperty('--pa-playback-strength');
      });
      const assignment = query('.kf-pa-map__assignment');
      // Avoid announcing every automatic step as an accessibility live update.
      assignment.setAttribute('aria-live', playing ? 'off' : 'polite');
      if (playing) {
        assignment.textContent = `Block ${playback.block} → ${tasksForBlock(playback.block).join(', ')} · stride ${constants.blocks}`;
        query('[data-pa-hint]').textContent = '自动循环 · 点击元素停止 · 滚轮放大进入 Task';
      }
    }
    function stopPlayback() {
      if (playback.stopped) return;
      playback.stopped = true;
      clearInterval(playbackTimer);
      playbackTimer = null;
      syncSelection();
    }
    function syncSelection() {
      host.querySelectorAll('[data-pa-block]').forEach((button) => {
        const id = Number(button.dataset.paBlock);
        button.classList.toggle('is-selected', id === state.block);
        button.setAttribute('aria-pressed', String(id === state.block));
        button.classList.toggle('is-hovered', hoverTask !== null && taskInfo(hoverTask).block === id);
      });
      host.querySelectorAll('[data-pa-task]').forEach((cell) => {
        const id = Number(cell.dataset.paTask);
        cell.classList.toggle('is-assigned', state.block !== null && id % 24 === state.block);
        cell.classList.toggle('is-selected', id === state.task);
        cell.classList.toggle('is-hovered', id === hoverTask);
        cell.setAttribute('aria-pressed', String(id === state.task));
        cell.setAttribute('tabindex', id === (state.task ?? 0) ? '0' : '-1');
      });
      query('.kf-pa-map__assignment').textContent = state.block === null
        ? '每个 SPMD Block 间隔 24 领取 Task'
        : `Block ${state.block} → ${tasksForBlock(state.block).join(', ')} · stride 24`;
      const detail = state.level === 'detail';
      const t = taskInfo(state.task ?? 0);
      query('[data-pa-action="back"]').hidden = !detail;
      query('[data-pa-action="enter"]').hidden = detail;
      query('[data-pa-action="enter"]').textContent = `进入 Task ${state.task ?? 0} ↗`;
      query('[data-pa-context]').textContent = detail ? `Task ${t.id}` : 'Batch × KV Head';
      query('[data-pa-meta]').textContent = detail ? `Batch ${t.batch} · KV ${t.kv} · Q${t.qStart}–Q${t.qEnd} · Block ${t.block}` : '';
      query('.kf-pa-map__config').hidden = !detail;
      syncPipeline();
      query('[data-pa-batch]').textContent = `Batch ${t.batch}`;
      query('[data-pa-length]').value = state.seqLens[t.batch];
      query('[data-pa-range]').value = state.seqLens[t.batch];
      const tiling = currentTiling();
      query('[data-pa-summary]').textContent = detail ? `${tiling.seqLen} tokens · ${tiling.pageCount} pages · ${tiling.stackCount} stacks · ${tiling.stackCount + constants.preLaunch} steps · ${constants.transferSlots} slots` : '每格 1 个 Task · 5 条底纹 = 5 个 Q Head';
      query('[data-pa-summary]').title = detail ? `pages = ceil(${tiling.seqLen}/128)；stacks = ceil(${tiling.pageCount}/4)；steps = ${tiling.stackCount} + PRE_LAUNCH (${constants.preLaunch})` : '';
      query('[data-pa-hint]').textContent = detail ? '滚轮缩放 · 拖拽平移 · ← / → 切换 Step' : '场景配置 · 滚轮放大进入 Task · 按住左键拖动移动';
      paintPlayback();
      publishScenario();
    }
    function render() {
      cancelLayoutTween();
      host.dataset.level = state.level;
      tooltip.hidden = true;
      hoverTask = null;
      if (state.level === 'matrix') drawMatrix(); else drawDetail();
      const len = state.seqLens[taskInfo(state.task ?? 0).batch];
      query('[data-pa-length]').value = len;
      query('[data-pa-range]').value = len;
      syncSelection();
      paintView();
    }
    const rememberSize = () => { state.size = { width: viewport.clientWidth, height: viewport.clientHeight }; };
    function fit() { animation?.cancel(); state.view = fitted(); rememberSize(); paintView(); }
    /* ---- 入场动效：矩阵 → Task ------------------------------------------------
       被点的格子不跟着底板一起消失：它被提到 morph 层（不带场景变换，纯屏幕坐标），
       从「格子边框 + Task 编号」补间成「卡片外框 + 卡片标题」；底板整帧朝这个格子
       收拢淡出，卡片一侧只淡入不位移，正文等外框落位后再淡进来。
       几层共用一条时间轴，读起来就是「这个格子长成了这张任务卡」。 */
    const viewTransform = (view) => `translate(${view.x}px, ${view.y}px) scale(${view.scale})`;
    const easeMorph = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
    const clamp01 = (value) => Math.max(0, Math.min(1, value));
    const num = (value, fallback = 0) => (Number.isFinite(Number.parseFloat(value)) ? Number.parseFloat(value) : fallback);
    // 以 anchor 为不动点收拢视图（k < 1 就是整块矩阵往被点的格子里缩）。
    const shrinkView = (view, anchor, k) => ({ scale: view.scale * k,
      x: anchor.x - (anchor.x - view.x) * k, y: anchor.y - (anchor.y - view.y) * k });
    let enterTransit = null;
    let morphTween = null, morphTask = null;
    /* morph 不挂场景变换，两端都用屏幕坐标——t=0 对上矩阵里的格子，
       t=1 与 scene 画出来的卡片外框/标题逐像素重合，所以拆层时看不出接缝。 */
    function endMorph() {
      if (morphTween) { cancelAnimationFrame(morphTween); morphTween = null; }
      // 生长窗口借的是展开组自带的裁剪，收尾/中断都要把它还原成整张卡片。
      if (morphTask !== null) {
        const frame = scene.querySelector('.kf-pa-map__task-frame.is-expanded');
        const boundary = frame?.querySelector('.kf-pa-map__task-boundary');
        const rect = scene.querySelector(`#pa-frame-clip-${morphTask} rect`);
        if (rect && boundary) {
          rect.setAttribute('x', '0');
          rect.setAttribute('y', '0');
          rect.setAttribute('width', boundary.getAttribute('width'));
          rect.setAttribute('height', boundary.getAttribute('height'));
        }
        morphTask = null;
      }
      scene.classList.remove('is-morphing');
      scene.style.removeProperty('--pa-morph-handoff');
      scene.style.removeProperty('--pa-morph-body');
      for (const layer of [morph, morphFill]) {
        layer.setAttribute('opacity', '0');
        layer.innerHTML = '';
      }
    }
    function snapshotGhost(view) {
      // 必须在 render() 换掉 scene.innerHTML 之前调用
      ghost.innerHTML = scene.innerHTML;
      ghost.setAttribute('transform', `translate(${view.x} ${view.y}) scale(${view.scale})`);
      ghost.setAttribute('opacity', '1');
    }
    function endGhost() {
      ghost.setAttribute('opacity', '0');
      ghost.removeAttribute('transform');
      ghost.innerHTML = '';
    }
    function cancelEnterTransit() {
      if (enterTransit) {
        enterTransit.scene?.cancel();
        enterTransit.ghost?.cancel();
        enterTransit = null;
        endGhost();
      }
      endMorph();
    }
    // 被点格子在屏幕坐标下的样子：外框 + 编号，连填充/描边/字重一起照抄，交给 morph 层接力。
    function cellMorphSource(id, r, view) {
      const group = scene.querySelector(`[data-pa-task="${id}"]`);
      const rectEl = group?.querySelector('.kf-pa-map__cell');
      const idEl = group?.querySelector('.kf-pa-map__task-id');
      const rectStyle = rectEl ? getComputedStyle(rectEl) : null;
      const idStyle = idEl ? getComputedStyle(idEl) : null;
      return {
        rect: { x: view.x + r.x * view.scale, y: view.y + r.y * view.scale,
          w: r.w * view.scale, h: r.h * view.scale, rx: num(rectEl?.getAttribute('rx')) * view.scale },
        fill: rectStyle?.fill || 'none', stroke: rectStyle?.stroke || 'none',
        strokeWidth: num(rectStyle?.strokeWidth),
        label: { x: view.x + (r.x + num(idEl?.getAttribute('x'))) * view.scale,
          y: view.y + (r.y + num(idEl?.getAttribute('y'))) * view.scale,
          size: num(idStyle?.fontSize, 12) * view.scale, weight: num(idStyle?.fontWeight, 400) },
        idFill: idStyle?.fill || 'currentColor',
      };
    }
    // 卡片一侧的落点。详情视图会收起顶部指标行（画布随之变高），取景因此会重排，
    // 所以每帧按当前 state.view 重算，而不是入场那一刻算死。
    function cardMorphTarget(task) {
      const view = state.view, a = activeRect();
      const boundaryEl = scene.querySelector('.kf-pa-map__task-frame.is-expanded .kf-pa-map__task-boundary');
      const titleEl = scene.querySelector('.kf-pa-map__task-frame.is-expanded .kf-pa-map__frame-title');
      const boundaryStyle = boundaryEl ? getComputedStyle(boundaryEl) : null;
      const target = {
        view: { x: view.x, y: view.y, scale: view.scale }, origin: { x: a.x, y: a.y },
        rect: { x: view.x + a.x * view.scale, y: view.y + a.y * view.scale, w: a.w * view.scale, h: a.h * view.scale,
          rx: num(boundaryEl?.getAttribute('rx'), 10) * view.scale },
        card: { fill: boundaryStyle?.fill || 'none', stroke: boundaryStyle?.stroke || 'none',
          strokeWidth: num(boundaryStyle?.strokeWidth, 1.5),
          titleFill: titleEl ? getComputedStyle(titleEl).fill : 'currentColor' },
        label: { y: 0, size: 0, prefixX: 0 },
      };
      if (titleEl) {
        // 真实标题就在 scene 里，量它比估它准：编号落位时正好嵌进 "Task " 之后。
        target.label = { y: view.y + (a.y + num(titleEl.getAttribute('y'))) * view.scale,
          size: num(getComputedStyle(titleEl).fontSize, 30) * view.scale,
          prefixX: view.x + (a.x + num(titleEl.getAttribute('x'))) * view.scale };
      }
      return target;
    }
    /* 主时间轴：外框的变形/移位逐帧落到属性上；格子的外观在前 60% 交棒给卡片外观；
       编号整串从左缘按同一比例伸缩，落位时嵌进真实标题；正文只在外框围出的窗口里露出来。 */
    function startMorph(plan) {
      const { id, source, target, duration } = plan;
      const initial = target();
      const panelTag = (cls, face) => `<rect class="kf-pa-map__morph-${cls}" fill="${face.fill}"/>`;
      const edgeTag = (cls, face) => `<rect class="kf-pa-map__morph-${cls}" fill="none" stroke="${face.stroke}" stroke-width="${face.strokeWidth}" vector-effect="non-scaling-stroke"/>`;
      morphFill.innerHTML = `${panelTag('panel-cell', source)}${panelTag('panel-card', initial.card)}`;
      morph.innerHTML = `${edgeTag('edge-cell', source)}${edgeTag('edge-card', initial.card)}`
        + '<text class="kf-pa-map__morph-id" text-anchor="middle"></text>'
        + '<text class="kf-pa-map__morph-title" text-anchor="middle"></text>'
        + '<text class="kf-pa-map__morph-prefix" text-anchor="start"></text>';
      const panels = ['cell', 'card'].map((skin) => morphFill.querySelector(`.kf-pa-map__morph-panel-${skin}`));
      const edges = ['cell', 'card'].map((skin) => morph.querySelector(`.kf-pa-map__morph-edge-${skin}`));
      const fromId = morph.querySelector('.kf-pa-map__morph-id');
      const toId = morph.querySelector('.kf-pa-map__morph-title');
      const prefix = morph.querySelector('.kf-pa-map__morph-prefix');
      fromId.textContent = String(id);
      toId.textContent = String(id);
      prefix.textContent = 'Task\u00A0';
      fromId.style.fill = source.idFill;
      fromId.style.fontWeight = String(source.label.weight);
      for (const el of [toId, prefix]) { el.style.fill = initial.card.titleFill; el.style.fontWeight = '700'; }
      morph.setAttribute('opacity', '1');
      morphFill.setAttribute('opacity', '1');
      morphTask = id;
      // 量源字号下 "Task " 与编号各自的步进：整串共享一条左缘轨迹，前缀和编号永远首尾相接。
      prefix.style.fontSize = `${source.label.size}px`;
      fromId.style.fontSize = `${source.label.size}px`;
      const prefixAdvance = prefix.getComputedTextLength();
      const numberAdvance = fromId.getComputedTextLength();
      const left0 = source.label.x - prefixAdvance - numberAdvance / 2;
      const place = (el, rect) => {
        el.setAttribute('x', rect.x.toFixed(2));
        el.setAttribute('y', rect.y.toFixed(2));
        el.setAttribute('width', Math.max(0, rect.w).toFixed(2));
        el.setAttribute('height', Math.max(0, rect.h).toFixed(2));
        el.setAttribute('rx', Math.max(0, rect.rx).toFixed(2));
      };
      const paint = (t) => {
        const e = easeMorph(t), to = target();
        const box = { x: lerp(source.rect.x, to.rect.x, e), y: lerp(source.rect.y, to.rect.y, e),
          w: lerp(source.rect.w, to.rect.w, e), h: lerp(source.rect.h, to.rect.h, e),
          rx: lerp(source.rect.rx, to.rect.rx, e) };
        const skin = clamp01(e * 1.6);
        [...panels, ...edges].forEach((el) => place(el, box));
        // 格子的外观在前 60% 交棒给卡片外观，底色与描边都不会跳。
        for (const [index, el] of panels.entries()) el.setAttribute('opacity', (index ? skin : 1 - skin).toFixed(3));
        for (const [index, el] of edges.entries()) el.setAttribute('opacity', (index ? skin : 1 - skin).toFixed(3));
        const size = lerp(source.label.size, to.label.size, e);
        const grow = source.label.size > 0 ? size / source.label.size : 1;
        const left = lerp(left0, to.label.prefixX, e), baseline = lerp(source.label.y, to.label.y, e);
        const numberX = left + (prefixAdvance + numberAdvance / 2) * grow;
        for (const el of [fromId, toId]) {
          el.setAttribute('x', numberX.toFixed(2));
          el.setAttribute('y', baseline.toFixed(2));
          el.style.fontSize = `${Math.max(0, size).toFixed(2)}px`;
        }
        fromId.style.fontWeight = String(Math.round(lerp(source.label.weight, 700, e)));
        fromId.setAttribute('opacity', (1 - skin).toFixed(3));
        toId.setAttribute('opacity', skin.toFixed(3));
        prefix.setAttribute('x', left.toFixed(2));
        prefix.setAttribute('y', baseline.toFixed(2));
        prefix.style.fontSize = `${Math.max(0, size).toFixed(2)}px`;
        prefix.setAttribute('opacity', clamp01((t - 0.3) / 0.4).toFixed(3));
        // 正文被裁进这个跟外框同步长大的窗口里——边框没到的地方不会先漏出 Stack / Page。
        // 用的是展开组自带的 pa-frame-clip，和普通详情渲染共用一条裁剪，收尾时再还原。
        const window_ = scene.querySelector(`#pa-frame-clip-${id} rect`);
        if (window_) {
          window_.setAttribute('x', ((box.x - to.view.x) / to.view.scale - to.origin.x).toFixed(2));
          window_.setAttribute('y', ((box.y - to.view.y) / to.view.scale - to.origin.y).toFixed(2));
          window_.setAttribute('width', (box.w / to.view.scale).toFixed(2));
          window_.setAttribute('height', (box.h / to.view.scale).toFixed(2));
        }
        // 正文晚一点、缓一点淡进来：窗口还小的时候它几乎还是透明的，
        // 裁剪边扫过字形也看不出来；等外框基本落位，正文正好到满。
        scene.style.setProperty('--pa-morph-body', clamp01((t - 0.46) / 0.34).toFixed(3));
        const handoff = clamp01((t - 0.86) / 0.14);
        scene.style.setProperty('--pa-morph-handoff', handoff.toFixed(3));
        morph.setAttribute('opacity', (1 - handoff).toFixed(3));
        morphFill.setAttribute('opacity', (1 - handoff).toFixed(3));
      };
      scene.classList.add('is-morphing');
      scene.style.setProperty('--pa-morph-handoff', '0');
      scene.style.setProperty('--pa-morph-body', '0');
      paint(0);
      const started = performance.now();
      const step = (now) => {
        const t = Math.min(1, (now - started) / duration);
        paint(t);
        if (t < 1) { morphTween = requestAnimationFrame(step); return; }
        endMorph();
      };
      morphTween = requestAnimationFrame(step);
    }
    function enterTask(id) {
      if (state.level !== 'matrix') return;
      stopPlayback();
      cancelLayoutTween();
      cancelEnterTransit();
      const duration = reducedMotion ? 0 : 460;
      const r = cellRect(id), old = { ...state.view };
      const source = duration ? cellMorphSource(id, r, old) : null;
      const origin = { x: old.x + (r.x + r.w / 2) * old.scale, y: old.y + (r.y + r.h / 2) * old.scale };
      // Restore to the last matrix view below the entry threshold.
      state.matrixView = { ...old };
      state.matrixSize = { width: viewport.clientWidth, height: viewport.clientHeight };
      const maxReturn = fitted().scale * 2.4;
      if (state.matrixView.scale > maxReturn) {
        const ratio = maxReturn / state.matrixView.scale;
        state.matrixView = { scale: maxReturn, x: origin.x - (origin.x - old.x) * ratio, y: origin.y - (origin.y - old.y) * ratio };
      }
      snapshotGhost(old);
      state.pipelineMode = 'schedule';
      state.pipelineStack = null;
      state.stackDetailOpen = false;
      state.task = id;
      state.block = taskInfo(id).block;
      state.level = 'detail';
      render();
      state.view = fitted();
      rememberSize();
      paintView();
      viewport.focus({ preventScroll: true });
      if (!duration) { endGhost(); return; }
      let cachedTarget = null, cacheKey = '';
      const target = () => {
        const view = state.view;
        const key = `${state.task}|${view.x.toFixed(2)}|${view.y.toFixed(2)}|${view.scale.toFixed(4)}`;
        if (cacheKey !== key) { cacheKey = key; cachedTarget = cardMorphTarget(state.task); }
        return cachedTarget;
      };
      startMorph({ id, source, target, duration });
      // 底板整帧往格子里收拢并快速淡出：格子已经交给 morph 层，剩下的只是退场的上下文。
      // 关键帧必须一路排到 offset 1——缺尾的话，浏览器会在最后一帧之后把值插回元素自身的
      // opacity，底板淡到 0 又会自己浮回来（grid ghosting）。
      const ghostAnim = ghost.animate([
        { transform: viewTransform(old), opacity: 1, offset: 0 },
        { transform: viewTransform(shrinkView(old, origin, 0.94)), opacity: 0.32, offset: 0.18 },
        { transform: viewTransform(shrinkView(old, origin, 0.86)), opacity: 0, offset: 0.38 },
        { transform: viewTransform(shrinkView(old, origin, 0.78)), opacity: 0, offset: 1 },
      ], { duration, easing: 'linear' });
      // 卡片一侧只淡入、不自己位移：画面里只有 morph 一条运动线索，叠影就没有了。
      const sceneAnim = scene.animate([
        { opacity: 0, offset: 0 },
        { opacity: 0, offset: 0.15 },
        { opacity: 1, offset: 0.6 },
        { opacity: 1, offset: 1 },
      ], { duration, easing: 'linear' });
      enterTransit = { scene: sceneAnim, ghost: ghostAnim };
      ghostAnim.finished.then(() => {
        if (enterTransit?.ghost !== ghostAnim) return;
        enterTransit = null;
        endGhost();
      }).catch(() => {});
      transitionUntil = performance.now() + duration + 80;
    }
    function back() {
      if (state.level !== 'detail') return;
      cancelEnterTransit();
      state.level = 'matrix';
      render();
      if (state.matrixView && state.matrixSize) {
        const width = viewport.clientWidth, height = viewport.clientHeight;
        const ratio = Math.min(width / state.matrixSize.width, height / state.matrixSize.height);
        state.view = { scale: state.matrixView.scale * ratio,
          x: width / 2 + (state.matrixView.x - state.matrixSize.width / 2) * ratio,
          y: height / 2 + (state.matrixView.y - state.matrixSize.height / 2) * ratio };
      } else state.view = fitted();
      rememberSize(); paintView();
      const r = cellRect(state.task ?? 0);
      const scale = state.view.scale * 1.25;
      animateFrom({ scale, x: state.view.x - r.x * (scale - state.view.scale), y: state.view.y - r.y * (scale - state.view.scale) });
      transitionUntil = performance.now() + 400;
      viewport.focus({ preventScroll: true });
    }
    // 详情视图内切换聚焦的 Task：保留矩阵上下文，把每个框的位置/尺寸连同取景一起补间。
    // 点右边的 Task，就是它从右侧滑到中间，而不是整块跳帧。
    const lerp = (a, b, t) => a + (b - a) * t;
    const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
    let layoutTween = null;
    const cancelLayoutTween = () => { if (layoutTween) { cancelAnimationFrame(layoutTween); layoutTween = null; } };
    function focusTask(id) {
      if (state.level !== 'detail' || id === state.task) return;
      stopPlayback();
      cancelLayoutTween();
      cancelEnterTransit();
      const fromFrames = neighborhood().map((r) => ({ ...r }));
      const fromView = { ...state.view };
      state.pipelineStack = null;
      state.stackDetailOpen = false;
      state.task = id;
      state.block = taskInfo(id).block;
      const toFrames = taskNeighborhood(id, state.seqLens).map((r) => ({ ...r }));
      const toView = fitted();
      // 面包屑/摘要/配置这些文字信息立即切到新 Task，只有几何在补间。
      syncSelection();
      if (!reducedMotion) {
        const fromMap = new Map(fromFrames.map((r) => [r.id, r]));
        const toMap = new Map(toFrames.map((r) => [r.id, r]));
        const GAP = 360;
        // 进来的框从右侧滑入，出去的框往左侧滑出。
        const tracks = [...new Set([...fromMap.keys(), ...toMap.keys()])].map((rid) => {
          const a = fromMap.get(rid), b = toMap.get(rid);
          return {
            id: rid,
            from: a || { x: b.x + GAP, y: b.y, w: b.w, h: b.h },
            to: b || { x: a.x - GAP, y: a.y, w: a.w, h: a.h },
            enter: !a, leave: !b,
          };
        });
        const started = performance.now(), DURATION = 340;
        transitionUntil = started + DURATION + 80;
        const step = (now) => {
          const t = Math.min(1, (now - started) / DURATION), e = easeInOut(t);
          paintDetailFrames(tracks.map((track) => ({
            id: track.id,
            x: lerp(track.from.x, track.to.x, e), y: lerp(track.from.y, track.to.y, e),
            w: lerp(track.from.w, track.to.w, e), h: lerp(track.from.h, track.to.h, e),
            opacity: track.enter ? e : track.leave ? 1 - e : 1,
          })));
          state.view = { x: lerp(fromView.x, toView.x, e), y: lerp(fromView.y, toView.y, e), scale: lerp(fromView.scale, toView.scale, e) };
          paintView();
          if (t < 1) { layoutTween = requestAnimationFrame(step); return; }
          layoutTween = null;
          state.view = toView; rememberSize(); drawDetail(); paintView();
        };
        host.dataset.level = 'detail';
        layoutTween = requestAnimationFrame(step);
      } else {
        drawDetail(); state.view = toView; rememberSize(); paintView();
      }
      transitionUntil = Math.max(transitionUntil, performance.now() + 400);
      viewport.focus({ preventScroll: true });
    }
    function zoomBy(factor, point, target) {
      if (performance.now() < transitionUntil) return;
      animation?.cancel();
      tooltip.hidden = true;
      const anchor = point || { x: viewport.clientWidth / 2, y: viewport.clientHeight / 2 };
      const old = state.view, base = fitted().scale;
      const scale = Math.max(base * 0.45, Math.min(base * 8, old.scale * factor));
      state.view = { scale, x: anchor.x - (anchor.x - old.x) * scale / old.scale, y: anchor.y - (anchor.y - old.y) * scale / old.scale };
      if (state.level === 'matrix' && scale >= base * 3) { enterTask(target ?? state.task ?? 0); return; }
      if (state.level === 'detail' && scale <= base * 0.7) { back(); return; }
      paintView();
    }
    function chooseTask(id) {
      stopPlayback();
      if (state.task !== id) { state.pipelineStack = null; state.stackDetailOpen = false; }
      state.task = id; state.block = taskInfo(id).block; syncSelection();
    }
    on(host, 'click', (event) => {
      // 按住左键拖动过画布 → 这次抬起不算点击，避免"拖完手一松就进到别的 Task"。
      if (suppressClick) { suppressClick = false; return; }
      if (event.target.closest('button, input, [data-pa-task]')) stopPlayback();
      const mode = event.target.closest('[data-pa-mode]');
      if (mode) { state.pipelineMode = mode.dataset.paMode; drawDetail(); paintView(); return; }
      const dependencyEdge = event.target.closest('[data-pa-dependency-edge]');
      if (dependencyEdge) { dependencyController?.selectEdge(dependencyEdge.dataset.paDependencyEdge); return; }
      const stage = event.target.closest('[data-pa-stage]');
      if (stage) {
        if (stage.getAttribute('aria-disabled') === 'true') return;
        const frame = stage.closest('[data-pa-task]');
        if (frame && Number(frame.dataset.paTask) !== state.task) focusTask(Number(frame.dataset.paTask));
        if (stage.hasAttribute('data-pa-stage-stack')) context.selectInstance({ stack: Number(stage.dataset.paStageStack), stage: stage.dataset.paStage });
        else context.selectStage(stage.dataset.paStage);
        return;
      }
      const handoff = event.target.closest('[data-pa-action]')?.dataset.paAction;
      if (handoff === 'compute' || handoff === 'source') {
        host.dispatchEvent(new CustomEvent('pto-pa-navigate', { bubbles: true, detail: { target: handoff, nodeId: stageNodes[state.stage] } }));
        return;
      }
      const view = event.target.closest('[data-pa-view]');
      if (view) { state.taskView = view.dataset.paView; drawDetail(); fit(); return; }
      const step = event.target.closest('[data-pa-step]');
      if (step) { selectStep(Number(step.dataset.paStep)); return; }
      const stack = event.target.closest('[data-pa-stack]');
      if (stack) { selectStack(Number(stack.dataset.paStack)); return; }
      const block = event.target.closest('[data-pa-block]');
      if (block) {
        if (state.level === 'detail') back();
        const id = Number(block.dataset.paBlock);
        state.block = state.block === id ? null : id;
        state.task = null;
        syncSelection(); return;
      }
      const task = event.target.closest('[data-pa-task]');
      if (task) {
        const id = Number(task.dataset.paTask);
        // 详情里点别的 Task = 直接进入它；矩阵里点只是选中。
        if (state.level === 'detail') focusTask(id); else chooseTask(id);
        return;
      }
      const action = event.target.closest('[data-pa-action]')?.dataset.paAction;
      if (action === 'close-stack') { state.stackDetailOpen = false; syncPipeline(); return; }
      if (action === 'enter') enterTask(state.task ?? 0);
      else if (action === 'back') back();
      else if (action === 'fit') fit();
      else if (action === 'in') zoomBy(1.25);
      else if (action === 'out') zoomBy(0.8);
    });
    on(viewport, 'dblclick', (event) => {
      const task = event.target.closest('[data-pa-task]');
      if (task) enterTask(Number(task.dataset.paTask));
      else fit();
    });
    // 画布内滚轮用于缩放；详情及其入口区域保留正常滚动。
    // 移动画布改为按住左键拖拽（见 pointerdown 分支）。
    on(host, 'wheel', (event) => {
      const rect = viewport.getBoundingClientRect();
      const inside = event.clientX >= rect.left && event.clientX <= rect.right
        && event.clientY >= rect.top && event.clientY <= rect.bottom;
      if (!inside) return;
      event.preventDefault();
      const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? viewport.clientHeight : 1;
      const task = event.target.closest('[data-pa-task]');
      zoomBy(Math.exp(Math.max(-0.22, Math.min(0.22, -event.deltaY * unit * 0.003))), { x: event.clientX - rect.left, y: event.clientY - rect.top }, task ? Number(task.dataset.paTask) : null);
    }, { passive: false });
    on(viewport, 'pointerdown', (event) => {
      // 画布上任意位置都能按住左键拖动（含 Task 框上方）；拖动过就不再算点击。
      // 这里刻意不 setPointerCapture：一旦捕获，之后的 click 会被改派到画布本身，
      // 委托到 Task 框上的单击就永远收不到。等真的拖起来（>4px）再捕获。
      if (event.button !== 0) return;
      animation?.cancel();
      suppressClick = false;
      drag = { x: event.clientX, y: event.clientY, startX: event.clientX, startY: event.clientY, moved: false, pointerId: event.pointerId };
    });
    on(viewport, 'pointermove', (event) => {
      if (drag) {
        if (!drag.moved) {
          if (Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) <= 4) return;
          drag.moved = true;
          // 拖动优先于切换动效：把未播完的补间/入场直接落到终态，避免几种位移打架。
          cancelEnterTransit();
          if (layoutTween) { cancelLayoutTween(); state.view = fitted(); rememberSize(); drawDetail(); paintView(); }
          viewport.setPointerCapture(drag.pointerId);
          viewport.classList.add('is-dragging');
        }
        state.view.x += event.clientX - drag.x; state.view.y += event.clientY - drag.y;
        drag.x = event.clientX; drag.y = event.clientY; paintView(); return;
      }
      const cell = event.target.closest('[data-pa-task]');
      const id = cell ? Number(cell.dataset.paTask) : null;
      if (id !== hoverTask) { hoverTask = id; syncSelection(); }
      tooltip.hidden = id === null;
      if (id !== null) {
        tooltip.textContent = taskDescription(id);
        const rect = viewport.getBoundingClientRect();
        tooltip.style.left = `${Math.max(8, Math.min(event.clientX - rect.left + 14, rect.width - tooltip.offsetWidth - 8))}px`;
        tooltip.style.top = `${Math.max(8, Math.min(event.clientY - rect.top + 18, rect.height - tooltip.offsetHeight - 8))}px`;
      }
    });
    const endDrag = () => {
      if (drag?.moved) suppressClick = true;
      drag = null; viewport.classList.remove('is-dragging');
    };
    on(viewport, 'pointerup', endDrag);
    on(viewport, 'pointercancel', endDrag);
    on(viewport, 'lostpointercapture', endDrag);
    on(viewport, 'pointerleave', () => { hoverTask = null; tooltip.hidden = true; syncSelection(); });
    on(query('[data-pa-steps]'), 'keydown', (event) => {
      if (['ArrowLeft', 'ArrowRight'].includes(event.key)) {
        event.preventDefault(); selectStep(state.pipelineStep + (event.key === 'ArrowRight' ? 1 : -1));
        query(`[data-pa-step="${state.pipelineStep}"]`).focus({ preventScroll: true });
      }
    });
    on(viewport, 'keydown', (event) => {
      const dependency = event.target.closest('[data-pa-dependency-edge]');
      if (dependency && ['Enter', ' '].includes(event.key)) { event.preventDefault(); dependencyController?.selectEdge(dependency.dataset.paDependencyEdge); return; }
      const cell = event.target.closest('[data-pa-task]');
      const id = cell ? Number(cell.dataset.paTask) : state.task ?? 0;
      const stage = event.target.closest('[data-pa-stage]');
      if (stage && ['Enter', ' '].includes(event.key)) { event.preventDefault(); stage.dispatchEvent(new MouseEvent('click', { bubbles: true })); return; }
      if (state.level === 'detail' && event.target.closest('[data-pa-stack]') && ['Enter', ' '].includes(event.key)) {
        event.preventDefault(); selectStack(Number(event.target.closest('[data-pa-stack]').dataset.paStack));
      } else if (state.level === 'detail' && state.taskView === 'pipeline' && ['ArrowLeft', 'ArrowRight'].includes(event.key)) {
        event.preventDefault(); selectStep(state.pipelineStep + (event.key === 'ArrowRight' ? 1 : -1));
      } else if (event.key === 'Enter') { event.preventDefault(); enterTask(id); }
      else if (event.key === ' ' && cell) { event.preventDefault(); chooseTask(id); }
      else if (event.key === 'Escape') { event.preventDefault(); back(); }
      else if (event.key === '+' || event.key === '=') { event.preventDefault(); zoomBy(1.25, null, id); }
      else if (event.key === '-') { event.preventDefault(); zoomBy(0.8); }
      else if (event.key === '0') { event.preventDefault(); fit(); }
      else if (state.level === 'matrix' && ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) {
        event.preventDefault();
        const t = taskInfo(id);
        const b = Math.max(0, Math.min(15, t.batch + (event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0)));
        const kv = Math.max(0, Math.min(7, t.kv + (event.key === 'ArrowDown' ? 1 : event.key === 'ArrowUp' ? -1 : 0)));
        chooseTask(b * 8 + kv);
        query(`[data-pa-task="${state.task}"]`).focus({ preventScroll: true });
      }
    });
    function setLength(value) {
      if (state.level !== 'detail' || !Number.isInteger(value) || value < 1 || value > constants.maxSeq) return false;
      animation?.cancel();
      if (!context.setLength(value)) return false;
      state.view = fitted(); rememberSize(); paintView(); return true;
    }
    on(query('[data-pa-range]'), 'input', (event) => setLength(Number(event.target.value)));
    on(query('[data-pa-length]'), 'input', (event) => setLength(Number(event.target.value)));
    on(query('[data-pa-length]'), 'change', (event) => {
      if (!setLength(Number(event.target.value))) event.target.value = state.seqLens[taskInfo(state.task).batch];
    });
    // Mounting in a collapsed inspector must not persist a 0×0 view.
    function resize() {
      const width = viewport.clientWidth, height = viewport.clientHeight;
      if (width <= 1 || height <= 1) return;
      animation?.cancel();
      if (!state.view || !state.size) state.view = fitted();
      else if (state.size.width !== width || state.size.height !== height) {
        const ratio = Math.min(width / state.size.width, height / state.size.height);
        state.view = { scale: state.view.scale * ratio,
          x: width / 2 + (state.view.x - state.size.width / 2) * ratio,
          y: height / 2 + (state.view.y - state.size.height / 2) * ratio };
      }
      state.size = { width, height }; paintView();
      // 详情视图进出时会重排（标题隐去、画布变高），重画后要补回选中态与焦点属性。
      if (state.level === 'detail') { drawDetail(); syncSelection(); }
    }
    render();
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(viewport);
    if (!playback.stopped) {
      playbackTimer = setInterval(() => {
        if (signal.aborted || document.hidden || state.level !== 'matrix'
          || viewport.clientWidth <= 1 || viewport.clientHeight <= 1) return;
        playback.block = (playback.block + 1) % constants.blocks;
        paintPlayback();
      }, 400);
    }
    const controller = { refresh() { if (state.level === 'detail') { drawDetail(); syncSelection(); paintView(); } }, abort() {
      if (signal.aborted) return;
      cancelLayoutTween();
      cancelEnterTransit();
      dependencyController?.abort();
      clearInterval(playbackTimer);
      lifetime.abort(); observer.disconnect(); animation?.cancel();
      if (mounted === controller) mounted = null;
    }, zoomBy, fit, reset: fit, getScale: () => state.view?.scale, setScale: (scale) => zoomBy(scale / state.view.scale) };
    mounted = controller;
    return controller;
  }
  global.PtoPagedAttentionIntro = { mount, model: { constants, defaultSeqLens, taskInfo, tasksForBlock, partition, pipelineFor, taskNeighborhood, reconcileSelection } };
})(window);
