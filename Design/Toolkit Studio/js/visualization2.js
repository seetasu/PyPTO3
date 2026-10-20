/* Independent visualization workspace. No legacy renderer or selection state is mounted here. */
(function () {
  'use strict';
  const tabs = ['assignment', 'pipeline', 'graph'];
  function createStore() {
    let state = { open: false, activeTab: 'assignment', selection: null,
      views: Object.fromEntries(tabs.map(id => [id, { subview: null, camera: null, ...(id === 'assignment' ? { phase: 'attention', phase0Camera: null } : {}) }])) };
    const listeners = new Set();
    const snapshot = () => structuredClone(state);
    return {
      getState: snapshot,
      dispatch(action) {
        if (action.type === 'open') state = { ...state, open: true, activeTab: 'assignment' };
        else if (action.type === 'close') state = { ...state, open: false };
        else if (action.type === 'tab' && tabs.includes(action.id)) state = { ...state, activeTab: action.id };
        else if (action.type === 'phase' && ['phase0', 'attention'].includes(action.id)) state = { ...state, views: { ...state.views, assignment: { ...state.views.assignment, phase: action.id } } };
        else if (action.type === 'select') state = { ...state, selection: action.selection };
        else if (action.type === 'view' && tabs.includes(action.id)) state = { ...state,
          views: { ...state.views, [action.id]: { ...state.views[action.id], ...action.patch } } };
        else return;
        listeners.forEach(fn => fn(snapshot(), action));
      },
      subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); },
    };
  }
  const store = createStore();
  // The same embedded Python source shown by the Studio editor owns this configuration.
  const source = globalThis.PTO_PAGED_ATTENTION_PYPTO_SOURCE;
  const sourceNumber = name => {
    const match = source?.match(new RegExp(`^${name} = (\\d+)`, 'm'));
    if (!match) throw new Error(`Missing source configuration: ${name}`);
    return Number(match[1]);
  };
  const constants = Object.freeze({ blocks: sourceNumber('ATTN_SPMD_BLOCKS'), kvHeads: sourceNumber('NUM_KV_HEADS'),
    qPerKv: sourceNumber('NUM_HEADS') / sourceNumber('NUM_KV_HEADS'), ropeCores: sourceNumber('ROPE_CORES'), batch: sourceNumber('BATCH'),
    lanesPerBlock: Number(source.match(/for aiv_id in pl\.split_aiv\((\d+)/)[1]) });
  const seqLens = [1025, 512, 2048, 128, 1536, 3584, 3072, 256, 769, 1024, 2049, 640, 2560, 129, 1792, 3328];
  const grid = { x: 94, y: 106, dx: 62, dy: 34, w: 54, h: 28, width: 1096, height: 388 };
  const taskInfo = id => ({ id, batch: Math.floor(id / 8), kv: id % 8, block: id % 24, qStart: id % 8 * 5, qEnd: id % 8 * 5 + 4 });
  const tasksForBlock = block => Array.from({length: Math.ceil((128 - block) / 24)}, (_, i) => block + i * 24);
  const cellRect = id => ({x: grid.x + taskInfo(id).batch * grid.dx, y: grid.y + taskInfo(id).kv * grid.dy, w: grid.w, h: grid.h});
  const taskDescription = id => { const t = taskInfo(id); return `Task ${id} · Batch ${t.batch} · KV ${t.kv} · Q${t.qStart}–Q${t.qEnd} · Block ${t.block} · ${seqLens[t.batch]} tokens`; };
  let root, viewport, scene, tooltip, trigger, hoverTask = null, drag = null, suppressClick = false;
  let previousInert = [], phase0Controller, phaseAnimation, pipelineController, tabAnimation, graphController;
  const phase0Model = window.PtoVisualization2Phase0.createModel({ ...constants, activeBatch: seqLens.length, itemsPerCore: Math.ceil(constants.kvHeads * constants.batch / constants.ropeCores) });
  function mount() {
    root = document.createElement('section');
    root.id = 'visualization2'; root.className = 'pto-ide-frame__pane viz2'; root.hidden = true;
    root.setAttribute('aria-label', '可视化2');
    root.innerHTML = `<header class="pto-ide-frame__pane-header viz2-header">
      <button type="button" class="btn btn-ghost btn-sm" data-viz2-back>← 返回原界面</button>
      <span class="pto-ide-frame__pane-title">可视化2</span>
      <span class="pto-ide-frame__pane-meta">paged_attention_pypto.py</span>
    </header>
    <div class="viz2-content">
      <div class="pto-ide-frame__tabstrip viz2-tabs"><div class="tab-control" role="tablist" aria-label="可视化视图">${tabs.map(id => `<button type="button" class="tab-control-item" role="tab" id="viz2-tab-${id}" aria-controls="viz2-${id}" data-viz2-tab="${id}">${id[0].toUpperCase()+id.slice(1)}</button>`).join('')}</div></div>
      ${tabs.map(id => `<section class="viz2-canvas" id="viz2-${id}" role="tabpanel" aria-labelledby="viz2-tab-${id}" tabindex="0" hidden></section>`).join('')}
    </div>`;
    document.querySelector('#ideWorkarea').append(root);
    const host = root.querySelector('#viz2-assignment');
    host.innerHTML = `<div class="viz2-phase-toolbar"><div class="segmented-control segmented-control-muted" role="group" aria-label="Assignment 阶段"><button class="btn btn-sm" type="button" data-viz2-phase="phase0" aria-pressed="false" aria-controls="viz2-phase0">Phase0</button><button class="btn btn-sm is-selected" type="button" data-viz2-phase="attention" aria-pressed="true" aria-controls="viz2-attention">Attention+Output</button></div></div>
      <div class="viz2-phase-surface" id="viz2-phase0" aria-label="Phase0 工作分配" hidden></div>
      <div class="viz2-phase-surface" id="viz2-attention" aria-label="Attention+Output 工作分配"><div class="kf-pa-map">      <section class="kf-pa-map__shelf" aria-label="SPMD Block 分配">
        <div class="kf-pa-map__shelf-label"><b>Block 分配</b><span>点击查看分配 · stride ${constants.blocks}</span></div>
        <div class="kf-pa-map__blocks">${Array.from({ length: 3 }, (_, group) => `<div class="kf-pa-map__block-group">${Array.from({ length: 8 }, (_, i) => {
          const id = group * 8 + i;
          return `<button class="btn btn-ghost kf-pa-map__block" type="button" data-pa-block="${id}" aria-label="SPMD Block ${id}" aria-pressed="false" title="Block ${id} · ${tasksForBlock(id).length} Tasks"><svg viewBox="0 0 28 24" aria-hidden="true"><path d="M4 13 14 18 24 13M4 18 14 23 24 18"/><path class="kf-pa-map__block-face" d="m4 7 10-5 10 5-10 5Z"/></svg><span>${String(id).padStart(2, '0')}</span></button>`;
        }).join('')}</div>`).join('')}</div>
        <p class="kf-pa-map__assignment" aria-live="polite"></p>
      </section>

      <div class="kf-pa-map__context"><b>Batch × KV Head</b></div>
      <div class="kf-pa-map__viewport" tabindex="0" role="group" aria-label="任务矩阵画布，滚轮缩放，拖拽平移，方向键选择 Task">
        <svg class="kf-pa-map__svg" xmlns="http://www.w3.org/2000/svg" aria-label="Batch 与 KV Head 的任务关系"><g class="kf-pa-map__scene"></g></svg>
        <div class="kf-pa-map__tooltip" role="tooltip" hidden></div>
      </div>
      <footer class="kf-pa-map__footer"><div><p data-viz2-selection aria-live="polite">每格 1 个 Task · 5 条底纹 = 5 个 Q Head</p><span>滚轮缩放 · 拖拽平移 · 方向键选择 Task</span></div>
      <div class="kf-pa-map__tools" role="group" aria-label="矩阵画布操作"><button class="btn btn-ghost btn-sm" data-viz2-zoom="0.8" aria-label="缩小矩阵">−</button><output data-viz2-scale>100%</output><button class="btn btn-ghost btn-sm" data-viz2-zoom="1.25" aria-label="放大矩阵">＋</button><button class="btn btn-ghost btn-sm" data-viz2-fit>适应</button></div></footer></div></div>`;
    viewport = host.querySelector('.kf-pa-map__viewport'); scene = host.querySelector('.kf-pa-map__scene'); tooltip = host.querySelector('#viz2-attention [role="tooltip"]');
    drawMatrix();
    phase0Controller = window.PtoVisualization2Phase0.mount(root.querySelector('#viz2-phase0'), {
      model: phase0Model,
      onSelect: selection => store.dispatch({type:'select',selection}),
      onCamera: camera => store.dispatch({type:'view',id:'assignment',patch:{phase0Camera:camera}}),
    });
    root.addEventListener('click', event => {
      if (event.target.closest('[data-viz2-back]')) { close(); return; }
      const tab = event.target.closest('[data-viz2-tab]');
      if (tab) { store.dispatch({type:'tab', id:tab.dataset.viz2Tab}); return; }
      const phase = event.target.closest('[data-viz2-phase]');
      if (phase) { store.dispatch({type:'phase',id:phase.dataset.viz2Phase}); return; }
      if (suppressClick) { suppressClick = false; return; }
      const block = event.target.closest('[data-pa-block]');
      if (block) { const id = Number(block.dataset.paBlock); store.dispatch({type:'select',selection: store.getState().selection?.kind === 'block' && store.getState().selection.block === id ? null : {kind:'block', block:id, task:null}}); }
      const task = event.target.closest('[data-pa-task]');
      if (task) selectTask(Number(task.dataset.paTask));
      const zoom = event.target.closest('[data-viz2-zoom]');
      if (zoom) zoomBy(Number(zoom.dataset.viz2Zoom));
      if (event.target.closest('[data-viz2-fit]')) fit();
    });
    root.querySelector('[role="tablist"]').addEventListener('keydown', event => {
      if (!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
      event.preventDefault(); const current = tabs.indexOf(store.getState().activeTab);
      const index = event.key === 'Home' ? 0 : event.key === 'End' ? 2 : (current + (event.key === 'ArrowRight' ? 1 : 2)) % 3;
      store.dispatch({type:'tab', id:tabs[index]}); root.querySelector(`[data-viz2-tab="${tabs[index]}"]`).focus();
    });
    viewport.addEventListener('wheel', event => {
      event.preventDefault(); const r = viewport.getBoundingClientRect();
      const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? r.height : 1;
      zoomBy(Math.exp(Math.max(-.22, Math.min(.22, -event.deltaY * unit * .003))), {x:event.clientX-r.left,y:event.clientY-r.top});
    }, {passive:false});
    viewport.addEventListener('pointerdown', event => {
      if (event.button !== 0) return;
      suppressClick = false; drag = {x:event.clientX,y:event.clientY,startX:event.clientX,startY:event.clientY,id:event.pointerId,moved:false};
    });
    viewport.addEventListener('pointermove', event => {
      if (drag) {
        if (!drag.moved && Math.hypot(event.clientX-drag.startX,event.clientY-drag.startY) <= 4) return;
        drag.moved = true; viewport.setPointerCapture(drag.id); viewport.classList.add('is-dragging'); tooltip.hidden = true;
        const camera = store.getState().views.assignment.camera;
        store.dispatch({type:'view',id:'assignment',patch:{camera:{...camera,x:camera.x+event.clientX-drag.x,y:camera.y+event.clientY-drag.y}}});
        drag.x=event.clientX; drag.y=event.clientY; return;
      }
      const cell = event.target.closest('[data-pa-task]'); hoverTask = cell ? Number(cell.dataset.paTask) : null; paintSelection();
      tooltip.hidden = hoverTask === null;
      if (hoverTask !== null) {
        tooltip.textContent = taskDescription(hoverTask); const r=viewport.getBoundingClientRect();
        tooltip.style.left = `${Math.max(8,Math.min(event.clientX-r.left+14,r.width-tooltip.offsetWidth-8))}px`;
        tooltip.style.top = `${Math.max(8,Math.min(event.clientY-r.top+18,r.height-tooltip.offsetHeight-8))}px`;
      }
    });
    const endDrag = () => { suppressClick=!!drag?.moved; drag=null; viewport.classList.remove('is-dragging'); };
    ['pointerup','pointercancel','lostpointercapture'].forEach(type => viewport.addEventListener(type,endDrag));
    viewport.addEventListener('pointerleave', () => { hoverTask=null; tooltip.hidden=true; paintSelection(); });
    viewport.addEventListener('keydown', event => {
      const delta={ArrowLeft:-8,ArrowRight:8,ArrowUp:-1,ArrowDown:1}[event.key];
      if (delta !== undefined) { event.preventDefault(); const id=Math.max(0,Math.min(127,(store.getState().selection?.task ?? 0)+delta)); selectTask(id); root.querySelector(`[data-pa-task="${id}"]`).focus(); }
      else if (['Enter',' '].includes(event.key) && event.target.matches('[data-pa-task]')) { event.preventDefault(); selectTask(Number(event.target.dataset.paTask)); }
      else if (event.key === 'Escape') { store.dispatch({type:'select',selection:null}); viewport.focus(); }
    });
    viewport.addEventListener('dblclick', event => { if (!event.target.closest('[data-pa-task]')) fit(); });
    let lastSize = null;
    new ResizeObserver(() => {
      const width=viewport.clientWidth, height=viewport.clientHeight;
      if (!width || !height || !store.getState().open || store.getState().activeTab !== 'assignment') return;
      const camera=store.getState().views.assignment.camera;
      if (!camera) fit();
      else if (lastSize && (width!==lastSize.width || height!==lastSize.height)) {
        const ratio=Math.min(width/lastSize.width,height/lastSize.height);
        store.dispatch({type:'view',id:'assignment',patch:{camera:{scale:camera.scale*ratio,
          x:width/2+(camera.x-lastSize.width/2)*ratio,y:height/2+(camera.y-lastSize.height/2)*ratio}}});
      }
      lastSize={width,height};
    }).observe(viewport);
    store.subscribe(render);
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
        ${seqLens.map((len, batch) => `<g data-pa-column="${batch}"><text class="kf-pa-map__batch" x="${grid.x + batch * grid.dx + grid.w / 2}" y="44">${batch}</text>${sequenceGraphic(len, grid.x + batch * grid.dx, 59, grid.w, 12)}<text class="kf-pa-map__seq" x="${grid.x + batch * grid.dx + grid.w / 2}" y="90">${len}</text></g>`).join('')}
        ${Array.from({ length: 8 }, (_, kv) => `<text class="kf-pa-map__row" x="8" y="${grid.y + kv * grid.dy + 19}">KV ${kv}</text>`).join('')}
        ${Array.from({ length: 128 }, (_, id) => {
          const r = cellRect(id);
          return `<g class="kf-pa-map__task" data-pa-task="${id}" transform="translate(${r.x} ${r.y})" tabindex="${id === (store.getState().selection?.task ?? 0) ? 0 : -1}" role="button" aria-label="${taskDescription(id)}" aria-pressed="false"><rect class="kf-pa-map__cell" width="${r.w}" height="${r.h}" rx="4"/>${Array.from({ length: 5 }, (_, q) => `<rect class="kf-pa-map__q" x="${5 + q * 9}" y="4" width="7" height="20" rx="1"/>`).join('')}<text class="kf-pa-map__task-id" x="27" y="19">${id}</text></g>`;
        }).join('')}`;
    }

  function selectTask(id) { const t=taskInfo(id); store.dispatch({type:'select',selection:{kind:'task',task:id,block:t.block,batch:t.batch,kvHead:t.kv}}); }
  function fitted() { const scale=Math.max(.05,Math.min((viewport.clientWidth-32)/grid.width,(viewport.clientHeight-24)/grid.height,1.2)); return {scale,x:(viewport.clientWidth-grid.width*scale)/2,y:(viewport.clientHeight-grid.height*scale)/2}; }
  function fit() { if (viewport.clientWidth && viewport.clientHeight) store.dispatch({type:'view',id:'assignment',patch:{camera:fitted()}}); }
  function zoomBy(factor, point={x:viewport.clientWidth/2,y:viewport.clientHeight/2}) {
    const old=store.getState().views.assignment.camera || fitted(), scale=Math.max(.05,Math.min(8,old.scale*factor)), ratio=scale/old.scale;
    store.dispatch({type:'view',id:'assignment',patch:{camera:{scale,x:point.x-(point.x-old.x)*ratio,y:point.y-(point.y-old.y)*ratio}}});
  }
  function paintSelection() {
    const selection=store.getState().selection;
    root.querySelectorAll('[data-pa-block]').forEach(el => { const id=Number(el.dataset.paBlock); el.classList.toggle('is-selected',id===selection?.block); el.classList.toggle('is-hovered',hoverTask!==null && taskInfo(hoverTask).block===id); el.setAttribute('aria-pressed',String(id===selection?.block)); });
    root.querySelectorAll('[data-pa-task]').forEach(el => { const id=Number(el.dataset.paTask); el.classList.toggle('is-assigned',id%24===selection?.block); el.classList.toggle('is-selected',id===selection?.task); el.classList.toggle('is-hovered',id===hoverTask); el.setAttribute('aria-pressed',String(id===selection?.task)); el.setAttribute('tabindex',id===(selection?.task ?? 0)?'0':'-1'); });
    root.querySelector('.kf-pa-map__assignment').textContent=selection ? `Block ${selection.block} → ${tasksForBlock(selection.block).join(', ')} · stride 24` : '每个 SPMD Block 间隔 24 领取 Task';
    root.querySelector('[data-viz2-selection]').textContent=selection?.task != null ? taskDescription(selection.task) : '每格 1 个 Task · 5 条底纹 = 5 个 Q Head';
  }
  function render(state, action) {
    root.hidden=!state.open;
    if (!state.open) return;
    tabs.forEach(id => { const active=id===state.activeTab, tab=root.querySelector(`[data-viz2-tab="${id}"]`); tab.classList.toggle('is-selected',active); tab.setAttribute('aria-selected',String(active)); tab.tabIndex=active?0:-1; root.querySelector(`#viz2-${id}`).hidden=!active; });
    if (action.type === 'tab') {
      tooltip.hidden = true; phase0Controller.hideTooltip(); pipelineController?.hideTooltip(); graphController?.hideTooltip();
      tabAnimation?.cancel();
      if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        tabAnimation = root.querySelector(`#viz2-${state.activeTab}`).animate(
          [{opacity:0,transform:'translateY(6px)'},{opacity:1,transform:'translateY(0)'}],
          {duration:220,easing:'cubic-bezier(.2,.8,.2,1)'});
      }
    }
    if (state.activeTab === 'pipeline' && !pipelineController) {
      const model = window.PtoVisualization2Pipeline.createModel({ ...constants, seqLens,
        pageTokens: sourceNumber('BLOCK_SIZE'), pagesPerStack: sourceNumber('STACK_PAGES'), phase0Model });
      pipelineController = window.PtoVisualization2Pipeline.mount(root.querySelector('#viz2-pipeline'), {
        model,
        onSelect: selection => store.dispatch({type:'select',selection}),
        onCamera: camera => store.dispatch({type:'view',id:'pipeline',patch:{camera}}),
      });
    }
    pipelineController?.update(state.selection,state.views.pipeline.camera,
      {active:state.activeTab === 'pipeline',enter:action.type === 'tab' && state.activeTab === 'pipeline'});
    if (state.activeTab === 'graph' && !graphController) {
      graphController = window.PtoVisualization2Graph.mount(root.querySelector('#viz2-graph'), {
        onSelect: selection => store.dispatch({type:'select',selection}),
        onView: patch => store.dispatch({type:'view',id:'graph',patch}),
      });
    }
    graphController?.update(state.selection,state.views.graph,
      {active:state.activeTab === 'graph',enter:action.type === 'tab' && state.activeTab === 'graph'});
    if (state.activeTab !== 'assignment') return;
    const phase = state.views.assignment.phase;
    root.querySelectorAll('[data-viz2-phase]').forEach(button => {
      const selected = button.dataset.viz2Phase === phase;
      button.classList.toggle('is-selected', selected); button.setAttribute('aria-pressed', String(selected));
    });
    const phase0 = root.querySelector('#viz2-phase0'), attention = root.querySelector('#viz2-attention');
    phase0.hidden = phase !== 'phase0'; attention.hidden = phase !== 'attention';
    if (action.type === 'phase') {
      tooltip.hidden = true; hoverTask = null; phase0Controller.hideTooltip();
      phaseAnimation?.cancel();
      if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        phaseAnimation = (phase === 'phase0' ? phase0 : attention).animate(
          [{opacity:0, transform:'translateY(6px)'},{opacity:1, transform:'translateY(0)'}],
          {duration:220, easing:'cubic-bezier(.2,.8,.2,1)'});
      }
    }
    if (phase === 'phase0') {
      phase0Controller.update(state.selection, state.views.assignment.phase0Camera, {selectionChanged: action.type !== 'view'});
      return;
    }
    if (!state.views.assignment.camera) { paintSelection(); fit(); return; }
    const {x,y,scale}=state.views.assignment.camera;
    scene.setAttribute('transform',`translate(${x} ${y}) scale(${scale})`);
    scene.style.setProperty('--pa-svg-label',`${Math.max(12,11/scale)}px`);
    scene.classList.toggle('is-distant',grid.dx*scale<34); scene.classList.toggle('is-tiny',grid.w*scale<25);
    root.querySelector('[data-viz2-scale]').textContent=`${Math.round(scale/fitted().scale*100)}%`;
    if (action.type !== 'view') paintSelection();
  }
  function open(button) {
    if (!root) mount();
    if (store.getState().open) return;
    trigger=button;
    const workarea=document.querySelector('#ideWorkarea'); workarea.dataset.visualization2='open';
    previousInert=[...workarea.children].filter(el=>el!==root).map(el=>[el,el.inert]); previousInert.forEach(([el])=>{el.inert=true;});
    store.dispatch({type:'open'}); root.querySelector('[data-viz2-tab="assignment"]').focus();
  }
  function close() {
    store.dispatch({type:'close'}); delete document.querySelector('#ideWorkarea').dataset.visualization2;
    previousInert.forEach(([el,inert])=>{el.inert=inert;}); previousInert=[];
    if (trigger?.isConnected) trigger.focus();
  }
  document.addEventListener('click',event=>{ const button=event.target.closest('[data-visualization2-open]'); if(button) open(button); });
  // Read-only snapshots + explicit actions keep future subviews and shared selection extensible.
  window.PtoVisualization2=Object.freeze({getState:store.getState,subscribe:store.subscribe,dispatch:store.dispatch});
})();
