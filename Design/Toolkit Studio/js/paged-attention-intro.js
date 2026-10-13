/* Task assignment and logical tiling for paged_attention_pypto.py.
 * Source constants: 17–32; task / page / stack mapping: 261–269.
 * Sequence lengths are editable scenario inputs, not captured runtime values.
 */
(function (global) {
  'use strict';
  const constants = Object.freeze({ batch: 16, kvHeads: 8, qPerKv: 5, blocks: 24, pageTokens: 128, pagesPerStack: 4, maxSeq: 3584 });
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
  // 详情视图只放当前 Task 与它的左右邻居（同一 KV head、相邻 Batch），不铺满整个矩阵。
  function taskNeighborhood(id, seqLens) {
    const active = taskInfo(id);
    const height = Math.max(260, 164 + partition(seqLens[active.batch]).stackCount * 60);
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
  const state = { seqLens: [...defaultSeqLens], block: null, task: null, level: 'matrix', view: null, matrixView: null, matrixSize: null, size: null };
  const grid = { x: 94, y: 106, dx: 62, dy: 34, w: 54, h: 28, width: 1096, height: 388 };
  let mounted = null;

  function mount(host) {
    mounted?.abort();
    const lifetime = new AbortController();
    const { signal } = lifetime;
    const on = (element, type, listener, options = {}) => element.addEventListener(type, listener, { ...options, signal });
    const query = (selector) => host.querySelector(selector);
    const reducedMotion = global.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let hoverTask = null, drag = null, animation = null, transitionUntil = 0, panTimer = null, suppressClick = false;
    host.innerHTML = `<div class="kf-pa-map">
      <header class="kf-pa-map__metrics">
        <span class="kf-eyebrow">TASK ASSIGNMENT</span>
        <ul class="kf-pa-map__metrics-row">${[
          [constants.batch, 'Batch'],
          [constants.kvHeads, 'KV Heads'],
          [constants.qPerKv, 'Q / KV'],
          [constants.blocks, 'SPMD Blocks'],
          [constants.pageTokens, 'Tokens / Page'],
          [constants.pagesPerStack, 'Pages / Stack'],
        ].map(([value, label]) => `<li><b>${value.toLocaleString('en-US')}</b><span>${label}</span></li>`).join('')}</ul>
      </header>
      <section class="kf-pa-map__shelf" aria-label="SPMD Block 分配">
        <div class="kf-pa-map__shelf-label"><b>SPMD Blocks <span>${constants.blocks}</span></b><span>点击查看分配 · stride ${constants.blocks}</span></div>
        <div class="kf-pa-map__blocks">${Array.from({ length: 3 }, (_, group) => `<div class="kf-pa-map__block-group">${Array.from({ length: 8 }, (_, i) => {
          const id = group * 8 + i;
          return `<button class="btn btn-ghost kf-pa-map__block" type="button" data-pa-block="${id}" aria-label="SPMD Block ${id}" aria-pressed="false" title="Block ${id} · ${tasksForBlock(id).length} Tasks"><svg viewBox="0 0 28 24" aria-hidden="true"><path d="M4 13 14 18 24 13M4 18 14 23 24 18"/><path class="kf-pa-map__block-face" d="m4 7 10-5 10 5-10 5Z"/></svg><span>${String(id).padStart(2, '0')}</span></button>`;
        }).join('')}</div>`).join('')}</div>
        <p class="kf-pa-map__assignment" aria-live="polite"></p>
      </section>
      <div class="kf-pa-map__context"><div class="kf-pa-map__breadcrumb"><button class="btn btn-ghost btn-sm" type="button" data-pa-action="back" hidden>← 返回矩阵</button><b data-pa-context>Batch × KV Head</b></div><button class="btn btn-ghost btn-sm" type="button" data-pa-action="enter">进入 Task 0 ↗</button><span class="kf-pa-map__detail-meta" data-pa-meta></span></div>
      <div class="kf-pa-map__config" hidden><label>场景配置 · <span data-pa-batch></span> seq_lens <input class="input" type="number" min="1" max="3584" step="1" aria-label="当前 Batch 的序列长度" data-pa-length></label><input type="range" min="1" max="3584" step="1" aria-label="调节当前 Batch 的序列长度" data-pa-range><span>tokens</span></div>
      <div class="kf-pa-map__viewport" tabindex="0" role="group" aria-label="任务矩阵画布，滚轮缩放，按住左键拖拽平移，方向键切换 Task，Escape 返回矩阵">
        <svg class="kf-pa-map__svg" xmlns="http://www.w3.org/2000/svg" aria-label="Batch 与 KV Head 的任务关系"><g data-pa-scene></g></svg>
        <div class="kf-pa-map__tooltip" role="tooltip" hidden></div>
      </div>
      <footer class="kf-pa-map__footer"><div><p data-pa-summary></p><span data-pa-hint>滚轮放大进入 Task · 拖拽平移</span></div><div class="kf-pa-map__tools" role="group" aria-label="简介画布操作"><button class="btn btn-ghost btn-sm" type="button" data-pa-action="out" aria-label="缩小简介">−</button><output data-pa-zoom>100%</output><button class="btn btn-ghost btn-sm" type="button" data-pa-action="in" aria-label="放大简介">＋</button><button class="btn btn-ghost btn-sm" type="button" data-pa-action="fit">适应</button></div></footer>
    </div>`;
    const viewport = query('.kf-pa-map__viewport');
    const scene = query('[data-pa-scene]');
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
      const scale = Math.max(0.05, Math.min((viewport.clientWidth - 32) / box.width, (viewport.clientHeight - 24) / box.height, 1.2));
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
      return `<text class="kf-pa-map__frame-title" x="24" y="48">Task ${id}</text><text class="kf-pa-map__frame-meta" x="24" y="80">Batch ${t.batch} · KV ${t.kv} · Q${t.qStart}–Q${t.qEnd}</text>${sequenceGraphic(len, 24, 102, 552, 12)}<text class="kf-pa-map__frame-length" x="610" y="114">${len} tokens</text>
        ${tiling.stacks.map((stack) => `<g transform="translate(24 ${140 + stack.id * 60})" data-pa-stack="${stack.id}" aria-label="Stack ${stack.id}，${stack.pages.length} Pages"><rect class="kf-pa-map__stack" width="792" height="52" rx="5"/><text class="kf-pa-map__stack-label" x="14" y="30">S${stack.id}</text>${Array.from({ length: 4 }, (_, i) => {
          const page = stack.pages[i], x = 124 + i * 166;
          if (!page) return `<g aria-hidden="true"><rect class="kf-pa-map__unused" x="${x}" y="4" width="150" height="44" rx="3"/><text class="kf-pa-map__unused-label" x="${x + 75}" y="29">—</text></g>`;
          return `<g data-pa-page="${page.id}" aria-label="Logical Page ${page.id}，${page.validTokens}/128 有效 tokens"><title>Page ${page.id} · ${page.validTokens}/128 tokens</title><rect class="kf-pa-map__page" x="${x}" y="4" width="150" height="44" rx="3"/><text class="kf-pa-map__page-label" x="${x + 10}" y="25">P${page.id}</text><text class="kf-pa-map__page-value" text-anchor="end" x="${x + 140}" y="25">${page.validTokens}/128</text><rect class="kf-pa-map__page-track" x="${x + 10}" y="34" width="130" height="6" rx="1"/><rect class="kf-pa-map__page-fill${page.validTokens < 128 ? ' is-tail' : ''}" x="${x + 10}" y="34" width="${130 * page.validTokens / 128}" height="6" rx="1"/></g>`;
        }).join('')}</g>`).join('')}`;
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
        const expandedGroup = `<g pointer-events="none" clip-path="url(#pa-frame-clip-${f.id})" opacity="${big.toFixed(3)}"${big === 0 ? ' display="none"' : ''}>${frameExpandedMarkup(f.id, len)}</g>`;
        return `<g class="kf-pa-map__task-frame${selected ? ' is-expanded' : ''}" data-pa-task="${f.id}" transform="translate(${f.x.toFixed(2)} ${f.y.toFixed(2)})" tabindex="${selected ? 0 : -1}" role="button" aria-label="${taskDescription(f.id)}" aria-pressed="${selected}" opacity="${opacity.toFixed(3)}"><rect class="kf-pa-map__task-boundary" width="${w}" height="${h}" rx="10"/>${compactGroup}${expandedGroup}</g>`;
      }).join('')}`;
    }
    function drawDetail() {
      paintDetailFrames(neighborhood().map((r) => ({ id: r.id, x: r.x, y: r.y, w: r.w, h: r.h, opacity: 1 })));
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
      query('[data-pa-batch]').textContent = `Batch ${t.batch}`;
      const tiling = currentTiling();
      query('[data-pa-summary]').textContent = detail ? `${tiling.seqLen} tokens = ${tiling.pageCount} Pages · ${tiling.stackCount} Stacks` : '每格 1 个 Task · 5 条底纹 = 5 个 Q Head';
      query('[data-pa-hint]').textContent = detail ? '逻辑页分组 · 滚轮缩放 · 按住左键拖动移动 · 点击其它 Task 直接进入' : '场景配置 · 滚轮放大进入 Task · 按住左键拖动移动';
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
    function enterTask(id) {
      if (state.level !== 'matrix') return;
      const r = cellRect(id), old = state.view;
      const origin = { x: old.x + (r.x + r.w / 2) * old.scale, y: old.y + (r.y + r.h / 2) * old.scale };
      // Restore to the last matrix view below the entry threshold.
      state.matrixView = { ...old };
      state.matrixSize = { width: viewport.clientWidth, height: viewport.clientHeight };
      const maxReturn = fitted().scale * 2.4;
      if (state.matrixView.scale > maxReturn) {
        const ratio = maxReturn / state.matrixView.scale;
        state.matrixView = { scale: maxReturn, x: origin.x - (origin.x - old.x) * ratio, y: origin.y - (origin.y - old.y) * ratio };
      }
      state.task = id;
      state.block = taskInfo(id).block;
      state.level = 'detail';
      render();
      state.view = fitted();
      rememberSize();
      paintView();
      const smallScale = r.w * old.scale / extent().width;
      animateFrom({ scale: smallScale, x: origin.x - extent().width * smallScale / 2, y: origin.y - extent().height * smallScale / 2 });
      transitionUntil = performance.now() + 400;
      viewport.focus({ preventScroll: true });
    }
    function back() {
      if (state.level !== 'detail') return;
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
      cancelLayoutTween();
      const fromFrames = neighborhood().map((r) => ({ ...r }));
      const fromView = { ...state.view };
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
      state.task = id; state.block = taskInfo(id).block; syncSelection();
    }
    on(host, 'click', (event) => {
      // 按住左键拖动过画布 → 这次抬起不算点击，避免"拖完手一松就进到别的 Task"。
      if (suppressClick) { suppressClick = false; return; }
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
    // 简介面板自身不滚动：滚轮一律不带动页面；落在画布上时用于缩放。
    // 移动画布改为按住左键拖拽（见 pointerdown 分支）。
    on(host, 'wheel', (event) => {
      event.preventDefault();
      const rect = viewport.getBoundingClientRect();
      const inside = event.clientX >= rect.left && event.clientX <= rect.right
        && event.clientY >= rect.top && event.clientY <= rect.bottom;
      if (!inside) return;
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
          // 拖动优先于切换动效：把未播完的补间直接落到终态，避免两种位移打架。
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
    on(viewport, 'keydown', (event) => {
      const cell = event.target.closest('[data-pa-task]');
      const id = cell ? Number(cell.dataset.paTask) : state.task ?? 0;
      if (event.key === 'Enter') { event.preventDefault(); enterTask(id); }
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
      state.seqLens[taskInfo(state.task).batch] = value;
      animation?.cancel();
      drawDetail();
      state.view = fitted();
      rememberSize();
      query('[data-pa-length]').value = value;
      query('[data-pa-range]').value = value;
      syncSelection(); paintView(); return true;
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
    const controller = { abort() {
      if (signal.aborted) return;
      cancelLayoutTween();
      lifetime.abort(); observer.disconnect(); animation?.cancel();
      if (mounted === controller) mounted = null;
    }, zoomBy, fit, reset: fit, getScale: () => state.view?.scale, setScale: (scale) => zoomBy(scale / state.view.scale) };
    mounted = controller;
    return controller;
  }
  global.PtoPagedAttentionIntro = { mount, model: { constants, defaultSeqLens, taskInfo, tasksForBlock, partition } };
})(window);
