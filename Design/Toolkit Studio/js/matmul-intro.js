/* Tiled matmul introduction for matmul.py (pypto-lib examples/beginner/matmul.py).
 * Source constants: 18–22; tiling loops: 31–37.
 * M / N are tiled across core-groups via nested pl.parallel, and K is consumed by
 * a single matmul — so the picture is a 4×4 tile grid over C fed by one shared K
 * lane, instead of a K-reduction loop. Shell layout mirrors PtoPagedAttentionIntro.
 */
(function (global) {
  'use strict';
  const constants = Object.freeze({ M: 256, N: 256, K: 256, mTile: 64, nTile: 64 });
  const mTiles = constants.M / constants.mTile;                          // 4
  const nTiles = constants.N / constants.nTile;                          // 4
  const tileCount = mTiles * nTiles;                                      // 16
  const macsPerTile = constants.mTile * constants.nTile * constants.K;   // 1,048,576
  const defaultTile = 0;

  function tileInfo(id) {
    const mb = Math.floor(id / nTiles);
    const nb = id % nTiles;
    return {
      id, mb, nb,
      m0: mb * constants.mTile,
      n0: nb * constants.nTile,
      aShape: [constants.mTile, constants.K],
      bShape: [constants.K, constants.nTile],
      outShape: [constants.mTile, constants.nTile],
      macs: macsPerTile,
    };
  }
  const tiles = Array.from({ length: tileCount }, (_, id) => tileInfo(id));

  // Scene geometry, in user units. A / B stack on the left, C sits on the right,
  // and the K lane in the middle carries the "no K tiling" fact.
  const A = { x: 56, y: 116, w: 236, h: 236 };
  const B = { x: 56, y: 404, w: 236, h: 236 };
  const C = { x: 712, y: 140, w: 384, h: 384 };
  const K_LANE = { x: 372, y: 300, w: 276, h: 76 };
  const SCENE = { width: 1160, height: 700 };
  const aBand = (mb) => ({ x: A.x, y: A.y + mb * (A.h / mTiles), w: A.w, h: A.h / mTiles });
  const bBand = (nb) => ({ x: B.x + nb * (B.w / nTiles), y: B.y, w: B.w / nTiles, h: B.h });
  const cTileRect = (id) => { const t = tileInfo(id); const s = C.w / nTiles; return { x: C.x + t.nb * s, y: C.y + t.mb * s, w: s, h: s }; };

  const state = { tile: defaultTile, view: null, size: null };
  let mounted = null;

  function mount(host) {
    mounted?.abort();
    const lifetime = new AbortController();
    const { signal } = lifetime;
    const on = (element, type, listener, options = {}) => element.addEventListener(type, listener, { ...options, signal });
    const query = (selector) => host.querySelector(selector);

    const metrics = [
      [constants.M, 'M'], [constants.N, 'N'], [constants.K, 'K'],
      [constants.mTile, 'M_TILE'], [constants.nTile, 'N_TILE'], [tileCount, 'Tiles'],
    ];
    const tileButton = (id) => {
      const t = tileInfo(id);
      return `<button class="btn btn-ghost kf-pa-map__block" type="button" data-mm-tile="${id}" aria-label="Tile ${id} · M band ${t.mb} × N band ${t.nb}" aria-pressed="false" title="Tile ${id} · M[${t.m0}:${t.m0 + constants.mTile}] × N[${t.n0}:${t.n0 + constants.nTile}]"><svg viewBox="0 0 28 24" aria-hidden="true"><rect x="4" y="3" width="9" height="8" rx="1"/><rect x="15" y="3" width="9" height="8" rx="1"/><rect x="4" y="13" width="9" height="8" rx="1"/><rect x="15" y="13" width="9" height="8" rx="1"/></svg><span>${id}</span></button>`;
    };
    const blockGroups = Array.from({ length: mTiles }, (_, mb) => `<div class="kf-pa-map__block-group kf-mm-shelf-band" data-mm-shelf-band="${mb}" aria-label="M band ${mb}">${Array.from({ length: nTiles }, (_, nb) => tileButton(mb * nTiles + nb)).join('')}</div>`).join('');

    host.innerHTML = `<div class="kf-pa-map kf-mm-map">
      <header class="kf-pa-map__metrics">
        <span class="kf-eyebrow">TILED MATMUL</span>
        <ul class="kf-pa-map__metrics-row">${metrics.map(([value, label]) => `<li><b>${value.toLocaleString('en-US')}</b><span>${label}</span></li>`).join('')}</ul>
      </header>
      <section class="kf-pa-map__shelf" aria-label="M / N 分块">
        <div class="kf-pa-map__shelf-label"><b>M × N 分块 <span>${mTiles} × ${nTiles}</span></b><span>每格 = 1 个 CORE_GROUP tile · 点击选中</span></div>
        <div class="kf-pa-map__blocks">${blockGroups}</div>
        <p class="kf-pa-map__assignment" aria-live="polite"></p>
      </section>
      <div class="kf-pa-map__context"><div class="kf-pa-map__breadcrumb"><b data-mm-context>M × N 分块网格</b></div><button class="btn btn-ghost btn-sm" type="button" data-mm-action="focus">定位选中 Tile ↗</button><span class="kf-pa-map__detail-meta" data-mm-meta></span></div>
      <div class="kf-pa-map__viewport" tabindex="0" role="group" aria-label="分块矩阵乘画布：滚轮缩放，按住左键拖拽平移，方向键切换 Tile，0 适应画布">
        <svg class="kf-pa-map__svg kf-mm-svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${SCENE.width} ${SCENE.height}" aria-label="A、B 矩阵分块与 C 输出 tile 的关系"><g data-mm-scene></g></svg>
        <div class="kf-pa-map__tooltip" role="tooltip" hidden></div>
      </div>
      <footer class="kf-pa-map__footer"><div><p data-mm-summary></p><span data-mm-hint>滚轮缩放 · 按住左键拖拽平移 · 点击 Tile 选中</span></div><div class="kf-pa-map__tools" role="group" aria-label="简介画布操作"><button class="btn btn-ghost btn-sm" type="button" data-mm-action="out" aria-label="缩小">−</button><output data-mm-zoom>100%</output><button class="btn btn-ghost btn-sm" type="button" data-mm-action="in" aria-label="放大">＋</button><button class="btn btn-ghost btn-sm" type="button" data-mm-action="fit">适应</button></div></footer>
    </div>`;

    const viewport = query('.kf-pa-map__viewport');
    const scene = query('[data-mm-scene]');
    const tooltip = query('.kf-pa-map__tooltip');
    let drag = null, hoverTile = null, suppressClick = false;

    const fitted = () => {
      const scale = Math.max(0.2, Math.min((viewport.clientWidth - 24) / SCENE.width, (viewport.clientHeight - 24) / SCENE.height, 1.4));
      return { scale, x: (viewport.clientWidth - SCENE.width * scale) / 2, y: (viewport.clientHeight - SCENE.height * scale) / 2 };
    };

    function matrixBlock(which, box, axis, cells) {
      const lines = cells.map((cell) => axis === 'row'
        ? `<line class="kf-mm-gridline" x1="${box.x}" y1="${cell.y}" x2="${box.x + box.w}" y2="${cell.y}"/>`
        : `<line class="kf-mm-gridline" x1="${cell.x}" y1="${box.y}" x2="${cell.x}" y2="${box.y + box.h}"/>`).join('');
      const marks = cells.map((cell, index) => {
        const attr = axis === 'row' ? `data-mm-a-band="${index}"` : `data-mm-b-band="${index}"`;
        const label = axis === 'row'
          ? `<text class="kf-mm-svg-axis" x="${box.x + 10}" y="${cell.y + cell.h / 2 + 4}">M${index}</text>`
          : `<text class="kf-mm-svg-axis" text-anchor="middle" x="${cell.x + cell.w / 2}" y="${box.y + box.h + 18}">N${index}</text>`;
        return `<g class="kf-mm-band" ${attr}><rect class="kf-mm-band-fill" x="${cell.x}" y="${cell.y}" width="${cell.w}" height="${cell.h}"/>${label}</g>`;
      }).join('');
      return `<g class="kf-mm-matrix" data-mm-matrix="${which}"><rect class="kf-mm-matrix-frame" x="${box.x}" y="${box.y}" width="${box.w}" height="${box.h}" rx="6"/>${lines}${marks}</g>`;
    }

    function drawScene() {
      const aCells = Array.from({ length: mTiles }, (_, mb) => aBand(mb));
      const bCells = Array.from({ length: nTiles }, (_, nb) => bBand(nb));
      const cMarks = tiles.map((t) => {
        const r = cTileRect(t.id);
        return `<g class="kf-mm-tile" data-mm-tile="${t.id}" role="button" tabindex="${t.id === state.tile ? 0 : -1}" aria-label="Tile ${t.id}，M band ${t.mb} × N band ${t.nb}" aria-pressed="false"><rect class="kf-mm-tile-frame" x="${r.x}" y="${r.y}" width="${r.w}" height="${r.h}" rx="4"/><text class="kf-mm-svg-tile" text-anchor="middle" x="${r.x + r.w / 2}" y="${r.y + r.h / 2 + 5}">${t.mb},${t.nb}</text></g>`;
      }).join('');
      scene.innerHTML = `
        <text class="kf-mm-svg-title" x="${A.x}" y="${A.y - 14}">A · [M × K] = [${constants.M} × ${constants.K}] · FP32</text>
        <text class="kf-mm-svg-title" x="${B.x}" y="${B.y - 14}">B · [K × N] = [${constants.K} × ${constants.N}] · FP32</text>
        <text class="kf-mm-svg-title" x="${C.x}" y="${C.y - 30}">C · [M × N] = [${constants.M} × ${constants.N}] · FP32</text>
        ${matrixBlock('a', A, 'row', aCells)}
        ${matrixBlock('b', B, 'col', bCells)}
        <path class="kf-mm-flow" d="M ${A.x + A.w} ${A.y + A.h / 2} C ${A.x + A.w + 34} ${A.y + A.h / 2}, ${K_LANE.x - 40} ${K_LANE.y + K_LANE.h / 2}, ${K_LANE.x} ${K_LANE.y + K_LANE.h / 2}"/>
        <path class="kf-mm-flow" d="M ${B.x + B.w} ${B.y + B.h / 2} C ${B.x + B.w + 34} ${B.y + B.h / 2}, ${K_LANE.x - 40} ${K_LANE.y + K_LANE.h / 2}, ${K_LANE.x} ${K_LANE.y + K_LANE.h / 2}"/>
        <path class="kf-mm-flow" d="M ${K_LANE.x + K_LANE.w} ${K_LANE.y + K_LANE.h / 2} C ${K_LANE.x + K_LANE.w + 30} ${K_LANE.y + K_LANE.h / 2}, ${C.x - 30} ${C.y + C.h / 2}, ${C.x} ${C.y + C.h / 2}"/>
        <g class="kf-mm-klane"><rect x="${K_LANE.x}" y="${K_LANE.y}" width="${K_LANE.w}" height="${K_LANE.h}" rx="10"/><text class="kf-mm-svg-title" text-anchor="middle" x="${K_LANE.x + K_LANE.w / 2}" y="${K_LANE.y + 34}">K = ${constants.K}</text><text class="kf-mm-svg-sub" text-anchor="middle" x="${K_LANE.x + K_LANE.w / 2}" y="${K_LANE.y + 58}">单次 matmul · 不切分 K</text></g>
        ${Array.from({ length: nTiles }, (_, nb) => `<text class="kf-mm-svg-axis" text-anchor="middle" x="${C.x + nb * (C.w / nTiles) + (C.w / nTiles) / 2}" y="${C.y - 6}">N${nb}</text>`).join('')}
        ${Array.from({ length: mTiles }, (_, mb) => `<text class="kf-mm-svg-axis" text-anchor="end" x="${C.x - 10}" y="${C.y + mb * (C.h / mTiles) + (C.h / mTiles) / 2 + 4}">M${mb}</text>`).join('')}
        <g class="kf-mm-c-grid">${cMarks}</g>`;
    }

    function paintView() {
      if (!state.view) return;
      const { x, y, scale } = state.view;
      scene.setAttribute('transform', `translate(${x} ${y}) scale(${scale})`);
      // Labels stay legible instead of shrinking with the scene.
      scene.style.setProperty('--pa-svg-label', `${Math.max(12, 11 / scale)}px`);
      query('[data-mm-zoom]').textContent = `${Math.round(scale / fitted().scale * 100)}%`;
    }

    function syncSelection() {
      const t = tileInfo(state.tile);
      host.querySelectorAll('[data-mm-tile]').forEach((element) => {
        const active = Number(element.dataset.mmTile) === state.tile;
        element.classList.toggle('is-selected', active);
        element.setAttribute('aria-pressed', String(active));
        if (element.getAttribute('role') === 'button') element.setAttribute('tabindex', active ? '0' : '-1');
      });
      host.querySelectorAll('[data-mm-a-band]').forEach((element) => element.classList.toggle('is-selected', Number(element.dataset.mmABand) === t.mb));
      host.querySelectorAll('[data-mm-b-band]').forEach((element) => element.classList.toggle('is-selected', Number(element.dataset.mmBBand) === t.nb));
      query('.kf-pa-map__assignment').textContent = `Tile ${t.id} → M band ${t.mb}（A 行 ${t.m0}–${t.m0 + constants.mTile - 1}）· N band ${t.nb}（B 列 ${t.n0}–${t.n0 + constants.nTile - 1}）`;
      query('[data-mm-context]').textContent = `Tile ${t.id} · M${t.mb} × N${t.nb}`;
      query('[data-mm-meta]').textContent = `A[${t.aShape.join('×')}] × B[${t.bShape.join('×')}] → C[${t.outShape.join('×')}]`;
      query('[data-mm-summary]').textContent = `Tile (${t.mb}, ${t.nb}) · 单次 matmul 收缩 K = ${constants.K} · MACs ${t.macs.toLocaleString('en-US')} / tile`;
    }

    function fit() { state.view = fitted(); state.size = { width: viewport.clientWidth, height: viewport.clientHeight }; paintView(); }
    function zoomBy(factor, point) {
      const anchor = point || { x: viewport.clientWidth / 2, y: viewport.clientHeight / 2 };
      const old = state.view, base = fitted().scale;
      const scale = Math.max(base * 0.4, Math.min(base * 6, old.scale * factor));
      state.view = { scale, x: anchor.x - (anchor.x - old.x) * scale / old.scale, y: anchor.y - (anchor.y - old.y) * scale / old.scale };
      paintView();
    }
    function focusSelected() {
      const r = cTileRect(state.tile);
      const scale = Math.max(state.view?.scale || fitted().scale, fitted().scale * 1.35);
      state.view = { scale, x: viewport.clientWidth / 2 - (r.x + r.w / 2) * scale, y: viewport.clientHeight / 2 - (r.y + r.h / 2) * scale };
      state.size = { width: viewport.clientWidth, height: viewport.clientHeight };
      paintView();
    }
    function select(id) {
      if (id < 0 || id >= tileCount || id === state.tile) return;
      state.tile = id;
      syncSelection();
    }
    const move = (delta) => select(Math.max(0, Math.min(tileCount - 1, state.tile + delta)));

    on(host, 'click', (event) => {
      if (suppressClick) { suppressClick = false; return; }
      const tile = event.target.closest('[data-mm-tile]');
      if (tile) { select(Number(tile.dataset.mmTile)); return; }
      const action = event.target.closest('[data-mm-action]')?.dataset.mmAction;
      if (action === 'in') zoomBy(1.25);
      else if (action === 'out') zoomBy(0.8);
      else if (action === 'fit') fit();
      else if (action === 'focus') focusSelected();
    });
    on(viewport, 'wheel', (event) => {
      event.preventDefault();
      const rect = viewport.getBoundingClientRect();
      const inside = event.clientX >= rect.left && event.clientX <= rect.right && event.clientY >= rect.top && event.clientY <= rect.bottom;
      if (!inside) return;
      const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? viewport.clientHeight : 1;
      zoomBy(Math.exp(Math.max(-0.22, Math.min(0.22, -event.deltaY * unit * 0.003))), { x: event.clientX - rect.left, y: event.clientY - rect.top });
    }, { passive: false });
    on(viewport, 'pointerdown', (event) => {
      if (event.button !== 0) return;
      suppressClick = false;
      drag = { x: event.clientX, y: event.clientY, startX: event.clientX, startY: event.clientY, moved: false, pointerId: event.pointerId };
    });
    on(viewport, 'pointermove', (event) => {
      if (drag) {
        if (!drag.moved) {
          if (Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) <= 4) return;
          drag.moved = true;
          viewport.setPointerCapture(drag.pointerId);
          viewport.classList.add('is-dragging');
        }
        state.view.x += event.clientX - drag.x; state.view.y += event.clientY - drag.y;
        drag.x = event.clientX; drag.y = event.clientY; paintView(); return;
      }
      const tile = event.target.closest('[data-mm-tile]');
      const id = tile ? Number(tile.dataset.mmTile) : null;
      if (id !== hoverTile) {
        hoverTile = id;
        host.querySelectorAll('[data-mm-tile]').forEach((element) => element.classList.toggle('is-hovered', Number(element.dataset.mmTile) === id));
      }
      tooltip.hidden = id === null;
      if (id !== null) {
        const t = tileInfo(id);
        tooltip.textContent = `Tile ${t.id} · A[${t.aShape.join('×')}] × B[${t.bShape.join('×')}] → C[${t.outShape.join('×')}]`;
        const rect = viewport.getBoundingClientRect();
        tooltip.style.left = `${Math.max(8, Math.min(event.clientX - rect.left + 14, rect.width - tooltip.offsetWidth - 8))}px`;
        tooltip.style.top = `${Math.max(8, Math.min(event.clientY - rect.top + 18, rect.height - tooltip.offsetHeight - 8))}px`;
      }
    });
    const endDrag = () => { if (drag?.moved) suppressClick = true; drag = null; viewport.classList.remove('is-dragging'); };
    on(viewport, 'pointerup', endDrag);
    on(viewport, 'pointercancel', endDrag);
    on(viewport, 'lostpointercapture', endDrag);
    on(viewport, 'pointerleave', () => {
      hoverTile = null;
      tooltip.hidden = true;
      host.querySelectorAll('[data-mm-tile].is-hovered').forEach((element) => element.classList.remove('is-hovered'));
    });
    on(viewport, 'keydown', (event) => {
      if (event.key === 'ArrowRight') { event.preventDefault(); move(1); }
      else if (event.key === 'ArrowLeft') { event.preventDefault(); move(-1); }
      else if (event.key === 'ArrowDown') { event.preventDefault(); move(nTiles); }
      else if (event.key === 'ArrowUp') { event.preventDefault(); move(-nTiles); }
      else if (event.key === 'Enter') { event.preventDefault(); focusSelected(); }
      else if (event.key === '+' || event.key === '=') { event.preventDefault(); zoomBy(1.25); }
      else if (event.key === '-') { event.preventDefault(); zoomBy(0.8); }
      else if (event.key === '0') { event.preventDefault(); fit(); }
    });

    // Mounting in a collapsed inspector must not persist a 0×0 view.
    function resize() {
      const width = viewport.clientWidth, height = viewport.clientHeight;
      if (width <= 1 || height <= 1) return;
      if (!state.view || !state.size) state.view = fitted();
      else if (state.size.width !== width || state.size.height !== height) {
        const ratio = Math.min(width / state.size.width, height / state.size.height);
        state.view = { scale: state.view.scale * ratio, x: width / 2 + (state.view.x - state.size.width / 2) * ratio, y: height / 2 + (state.view.y - state.size.height / 2) * ratio };
      }
      state.size = { width, height };
      paintView();
    }

    drawScene();
    syncSelection();
    state.view = fitted();
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(viewport);
    const controller = {
      abort() {
        if (signal.aborted) return;
        lifetime.abort(); observer.disconnect();
        if (mounted === controller) mounted = null;
      },
      zoomBy, fit, reset: fit, focusSelected,
      getScale: () => state.view?.scale,
      setScale: (scale) => zoomBy(scale / state.view.scale),
    };
    mounted = controller;
    return controller;
  }
  global.PtoMatmulIntro = { mount, model: { constants, mTiles, nTiles, tileCount, macsPerTile, tileInfo, tiles } };
})(window);
