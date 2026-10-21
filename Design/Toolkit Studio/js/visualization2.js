/* Independent visualization workspace. No legacy renderer or selection state is mounted here. */
(function () {
  'use strict';
  const tabs = ['assignment', 'pipeline', 'graph'];
  let graphIntraController, graphIntraModel, graphZoomJob, graphOuterSnapshot;
  let tabTransition, semanticZoom, anatomyController, anatomyModel, pipelineIntraController, pipelineIntraModel, pipelineZoom;
  function createStore() {
    let state = { open: false, activeTab: 'assignment', selection: null,
      views: Object.fromEntries(tabs.map(id => [id, { subview: null, camera: null, ...(id === 'assignment' ? { phase: 'attention', phase0Camera: null } : {}) }])) };
    const listeners = new Set();
    const snapshot = () => structuredClone(state);
    return {
      getState: snapshot,
      dispatch(action) {
        // A wheel gesture emits several events after crossing the semantic-zoom
        // threshold. Those follow-up camera updates must not cancel the morph.
        if(action.type==='view'&&action.id==='assignment'&&semanticZoom?.animating)return;
        if(['tab','close','select','pipeline-intra-task','pipeline-intra-focus'].includes(action.type)||(action.type==='view'&&pipelineZoom?.animating))pipelineZoom?.cancel();
        if(['tab','close','select','graph-intra-focus','graph-intra-task'].includes(action.type)||(action.type==='view'&&graphZoomJob?.running))cancelGraphZoom();
        if(action.type==='tab')semanticZoom?.cancel();
        const transition = action.type === 'tab' ? tabTransition?.prepare(state,action.id) : null;
        if (['tab','close','phase','select','anatomy-focus','anatomy-task'].includes(action.type)) semanticZoom?.cancel();
        if (['close','phase','select'].includes(action.type) || (action.type === 'view' && tabTransition?.animating)) tabTransition?.cancel();
        if (action.type === 'open') state = { ...state, open: true, activeTab: 'assignment',views:{...state.views,assignment:{...state.views.assignment,subview:null}} };
        else if (action.type === 'close') state = { ...state, open: false };
        else if (action.type === 'tab' && tabs.includes(action.id)) {
          // Scope follows the current workspace context, just like Task identity.
          // Each view keeps its own camera and maps shared subobjects on entry.
          const target=state.views[action.id];
          const inner=state.views[state.activeTab].subview==='block'&&state.selection?.kind==='task';
          const task=state.selection?.task;
          const focus=inner?window.PtoVisualization2IntraMorph.mapFocus(state.activeTab,action.id,state.selection.subobject,anatomyModel.tasks[task]):state.selection?.subobject;
          if(action.id==='graph'&&inner&&target.subview!=='block')
            graphOuterSnapshot={camera:target.camera,nodeId:target.nodeId,inspectorOpen:target.inspectorOpen};
          const context=inner?(action.id==='assignment'?{phase:'attention',anatomyTask:task,
            anatomyFocus:{...target.anatomyFocus,[task]:focus},anatomyInspectorOpen:target.anatomyInspectorOpen??!!focus}:
            action.id==='pipeline'?{intraTask:task,intraFocus:{...target.intraFocus,[task]:focus},intraInspectorOpen:target.intraInspectorOpen??!!focus}:
            {intraTask:task,intraFocus:focus,intraInspectorOpen:target.intraInspectorOpen??!!focus}):{};
          state={...state,activeTab:action.id,selection:inner?{...state.selection,subobject:focus}:state.selection,views:{...state.views,[action.id]:{...target,...context,subview:inner?'block':null}}};
        }
        else if (action.type === 'phase' && ['phase0', 'attention'].includes(action.id)) state = { ...state, views: { ...state.views, assignment: { ...state.views.assignment, phase: action.id } } };
        else if (action.type === 'anatomy-enter' || action.type === 'anatomy-task') {
          const task=anatomyModel.tasks[action.id];if(!task)return;
          const assignment=state.views.assignment,shared=state.selection?.task===task.id?state.selection.subobject:null;
          const incoming=['stage','event'].includes(shared?.kind)?{kind:'stack',index:shared.index}:Number.isInteger(shared?.stackIndex)?{kind:'stack',index:shared.stackIndex}:shared;
          const focus=['stack','page','slot','output'].includes(incoming?.kind)?incoming:assignment.anatomyFocus?.[task.id]||null;
          state={...state,selection:{kind:'task',task:task.id,block:task.block,batch:task.batch,kvHead:task.kvHead,subobject:focus},views:{...state.views,assignment:{...assignment,subview:'block',phase:'attention',anatomyTask:task.id,anatomyFocus:{...assignment.anatomyFocus,[task.id]:focus},anatomyInspectorOpen:!!focus}}};
        }
        else if (action.type === 'anatomy-exit') state={...state,views:{...state.views,assignment:{...state.views.assignment,subview:null}}};
        // Subobjects are scoped by the root Task identity; outer views can retain
        // this context while their renderers continue to select the owning Task.
        else if (action.type === 'anatomy-focus') {
          const assignment=state.views.assignment;
          state={...state,selection:{...state.selection,subobject:action.focus},views:{...state.views,assignment:{...assignment,anatomyFocus:{...assignment.anatomyFocus,[assignment.anatomyTask]:action.focus},anatomyInspectorOpen:true}}};
        }
        else if (action.type === 'pipeline-intra-enter' || action.type === 'pipeline-intra-task') {
          const task=pipelineIntraModel.tasks[action.id];if(!task)return;
          const view=state.views.pipeline,shared=state.selection?.task===task.id?state.selection.subobject:null,incoming=Number.isInteger(shared?.stackIndex)&&!['stage','event','page','stack'].includes(shared?.kind)?{kind:'stack',index:shared.stackIndex}:shared;
          const focus=['stack','page','slot','event','stage','tick','output'].includes(incoming?.kind)?incoming:view.intraFocus?.[task.id]||null;
          state={...state,selection:{kind:'task',task:task.id,block:task.block,batch:task.batch,kvHead:task.kvHead,subobject:focus},views:{...state.views,pipeline:{...view,subview:'block',intraTask:task.id,intraFocus:{...view.intraFocus,[task.id]:focus},intraInspectorOpen:!!focus}}};
        }
        else if(action.type==='pipeline-intra-exit')state={...state,views:{...state.views,pipeline:{...state.views.pipeline,subview:null}}};
        else if(action.type==='pipeline-intra-focus'){
          const view=state.views.pipeline;
          state={...state,selection:{...state.selection,subobject:action.focus},views:{...state.views,pipeline:{...view,intraFocus:{...view.intraFocus,[view.intraTask]:action.focus},intraInspectorOpen:true}}};
        }
        else if(action.type==='graph-intra-enter'||action.type==='graph-intra-task'){
          const task=graphIntraModel.tasks[action.id];if(!task)return;
          const v=state.views.graph,focus=state.selection?.task===task.id?state.selection.subobject:null;
          state={...state,selection:{kind:'task',task:task.id,block:task.block,batch:task.batch,kvHead:task.kvHead,subobject:focus},views:{...state.views,graph:{...v,subview:'block',intraTask:task.id,intraFocus:focus,intraInspectorOpen:!!focus}}};
        }
        else if(action.type==='graph-intra-exit')state={...state,views:{...state.views,graph:{...state.views.graph,...graphOuterSnapshot,subview:null}}};
        else if(action.type==='graph-intra-focus')state={...state,selection:{...state.selection,subobject:action.focus},views:{...state.views,graph:{...state.views.graph,intraFocus:action.focus,intraInspectorOpen:true}}};
        else if (action.type === 'select') state = { ...state, selection: action.selection };
        else if (action.type === 'view' && tabs.includes(action.id)) state = { ...state,
          views: { ...state.views, [action.id]: { ...state.views[action.id], ...action.patch } } };
        else return;
        listeners.forEach(fn => fn(snapshot(), action));
        if (transition) tabTransition.play(transition);
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
      <div class="pto-ide-frame__tabstrip viz2-tabs"><div class="tab-control" role="tablist" aria-label="可视化视图">${tabs.map(id => `<button type="button" class="tab-control-item" role="tab" id="viz2-tab-${id}" aria-controls="viz2-${id}" data-viz2-tab="${id}"><span class="viz2-tab-icon viz2-tab-icon--${id}" aria-hidden="true"></span>${id[0].toUpperCase()+id.slice(1)}</button>`).join('')}</div></div>
      ${tabs.map(id => `<section class="viz2-canvas" id="viz2-${id}" role="tabpanel" aria-labelledby="viz2-tab-${id}" tabindex="0" hidden></section>`).join('')}
    </div>`;
    document.querySelector('#ideWorkarea').append(root);
    const host = root.querySelector('#viz2-assignment');
    host.innerHTML = `<div class="viz2-phase-toolbar viz2-assignment-hud"><div data-viz2-phases class="segmented-control segmented-control-muted" role="group" aria-label="Assignment 阶段"><button class="btn btn-sm" type="button" data-viz2-phase="phase0" aria-pressed="false" aria-controls="viz2-phase0">Phase0</button><button class="btn btn-sm is-selected" type="button" data-viz2-phase="attention" aria-pressed="true" aria-controls="viz2-attention">Attention+Output</button></div><div class="segmented-control segmented-control-muted" role="group" aria-label="Assignment 范围"><button class="btn btn-sm is-selected" data-viz2-scope="inter" aria-label="Block 间" title="Block 间" aria-pressed="true"><span class="viz2-block-scope-icon viz2-block-scope-icon--inter" aria-hidden="true"></span></button><button class="btn btn-sm" data-viz2-scope="intra" aria-label="Block 内" title="Block 内" aria-pressed="false" disabled><span class="viz2-block-scope-icon" aria-hidden="true"></span></button></div><nav aria-label="当前对象" data-viz2-breadcrumb></nav><button class="btn btn-ghost btn-sm" data-viz2-matrix-back hidden>← 返回矩阵</button></div>
      <div class="viz2-phase-surface" id="viz2-anatomy" aria-label="Block 内 Task 数据结构" hidden></div>
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
      <footer class="kf-pa-map__footer"><div><p data-viz2-selection aria-live="polite">每格 1 个 Task · 5 条底纹 = 5 个 Q Head</p><span>指向 Task 持续放大进入内部 · 拖拽平移 · 方向键选择</span></div>
      <div class="kf-pa-map__tools" role="group" aria-label="矩阵画布操作"><button class="btn btn-ghost btn-sm" data-viz2-zoom="0.8" aria-label="缩小矩阵">−</button><output data-viz2-scale>100%</output><button class="btn btn-ghost btn-sm" data-viz2-zoom="1.25" aria-label="放大矩阵">＋</button><button class="btn btn-ghost btn-sm" data-viz2-fit>适应</button></div></footer></div></div>`;
    viewport = host.querySelector('.kf-pa-map__viewport'); scene = host.querySelector('.kf-pa-map__scene'); tooltip = host.querySelector('#viz2-attention [role="tooltip"]');
    drawMatrix();
    anatomyModel=window.PtoVisualization2Anatomy.createModel({seqLens,blocks:constants.blocks,numHeads:sourceNumber('NUM_HEADS'),kvHeads:constants.kvHeads,
      headDim:sourceNumber('HEAD_DIM'),rowTile:sourceNumber('ROW_TILE'),pageTokens:sourceNumber('BLOCK_SIZE'),pagesPerStack:sourceNumber('STACK_PAGES'),transferSlots:sourceNumber('PRE_LAUNCH')+1});
    graphIntraModel=window.PtoVisualization2GraphIntra.createModel({anatomy:anatomyModel,source,preLaunch:sourceNumber('PRE_LAUNCH')});
    pipelineIntraModel=window.PtoVisualization2PipelineIntra.createModel({anatomy:anatomyModel,preLaunch:sourceNumber('PRE_LAUNCH')});
    pipelineZoom=window.PtoVisualization2PipelineZoom.create({root,getOuter:()=>pipelineController,getInner:()=>pipelineIntraController});
    semanticZoom=window.PtoVisualization2SemanticZoom.create({root,getMatrix:matrixSnapshot,getAnatomy:()=>anatomyController?.getTransitionSnapshot()});
    phase0Controller = window.PtoVisualization2Phase0.mount(root.querySelector('#viz2-phase0'), {
      model: phase0Model,
      onSelect: selection => store.dispatch({type:'select',selection}),
      onCamera: camera => store.dispatch({type:'view',id:'assignment',patch:{phase0Camera:camera}}),
    });
    root.querySelector('#viz2-pipeline').innerHTML=`<div class="viz2-phase-toolbar viz2-assignment-hud"><div class="segmented-control segmented-control-muted" role="group" aria-label="Pipeline 范围"><button class="btn btn-sm is-selected" data-pipeline-scope="inter" aria-label="Block 间" title="Block 间" aria-pressed="true"><span class="viz2-block-scope-icon viz2-block-scope-icon--inter" aria-hidden="true"></span></button><button class="btn btn-sm" data-pipeline-scope="intra" aria-label="Block 内" title="Block 内" aria-pressed="false" disabled><span class="viz2-block-scope-icon" aria-hidden="true"></span></button></div><nav aria-label="Pipeline 当前对象" data-pipeline-breadcrumb>Pipeline</nav><button class="btn btn-ghost btn-sm" data-pipeline-back hidden>← 返回 Block 间</button></div><div class="viz2-phase-surface" id="viz2-pipeline-inter"></div><div class="viz2-phase-surface" id="viz2-pipeline-intra" hidden></div>`;
    root.querySelector('#viz2-graph').innerHTML=`<div class="viz2-phase-toolbar viz2-assignment-hud"><div class="segmented-control segmented-control-muted" role="group" aria-label="Graph 范围"><button class="btn btn-sm is-selected" data-graph-scope="inter" aria-label="Block 间" title="Block 间" aria-pressed="true"><span class="viz2-block-scope-icon viz2-block-scope-icon--inter" aria-hidden="true"></span></button><button class="btn btn-sm" data-graph-scope="intra" aria-label="Block 内" title="Block 内" aria-pressed="false"><span class="viz2-block-scope-icon" aria-hidden="true"></span></button></div><nav data-graph-breadcrumb aria-label="Graph 当前对象">Graph</nav><button class="btn btn-ghost btn-sm" data-graph-back hidden>← 返回 Block 间</button></div><div class="viz2-phase-surface" id="viz2-graph-inter"></div><div class="viz2-phase-surface" id="viz2-graph-intra" hidden></div>`;
    root.addEventListener('click', event => {
      if (event.target.closest('[data-viz2-back]')) { close(); return; }
      const tab = event.target.closest('[data-viz2-tab]');
      if (tab) { store.dispatch({type:'tab', id:tab.dataset.viz2Tab}); return; }
      const gs=event.target.closest('[data-graph-scope]');if(gs){gs.dataset.graphScope==='intra'?enterGraphIntra():exitGraphIntra();return;}
      if(event.target.closest('[data-graph-back]')){exitGraphIntra();return;}
      if(event.target.closest('[data-pipeline-back]')){exitPipelineIntra();return;}
      const pipelineScope=event.target.closest('[data-pipeline-scope]');if(pipelineScope){if(pipelineScope.dataset.pipelineScope==='inter')exitPipelineIntra();else enterPipelineIntra(store.getState().selection?.task);return;}
      if(event.target.closest('[data-viz2-matrix-back]')){exitAnatomy();return;}
      const scope=event.target.closest('[data-viz2-scope]');if(scope){if(scope.dataset.viz2Scope==='inter')exitAnatomy();else enterAnatomy(store.getState().selection?.task);return;}
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
      const factor=Math.exp(Math.max(-.22, Math.min(.22, -event.deltaY * unit * .003)));
      const task=event.target.closest('[data-pa-task]');
      zoomBy(factor, {x:event.clientX-r.left,y:event.clientY-r.top}, task?Number(task.dataset.paTask):null);
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
    viewport.addEventListener('dblclick', event => { const target=event.target.closest('[data-pa-task]');if(target)enterAnatomy(Number(target.dataset.paTask));else fit(); });
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
    tabTransition = window.PtoVisualization2Transition.create({root,getPipeline:()=>pipelineController,getGraph:()=>graphController,getIntraTask:id=>anatomyModel?.tasks[id]});
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

  function matrixSnapshot(id) {
    const task=anatomyModel.tasks[id],element=scene.querySelector(`[data-pa-task="${id}"]`);
    const strip=scene.querySelector(`[data-pa-column="${task.batch}"] .kf-pa-map__sequence`);
    const rect=element=>{const r=element.getBoundingClientRect();return {x:r.left,y:r.top,width:r.width,height:r.height};};
    return {task:id,model:task,frame:rect(element.querySelector('.kf-pa-map__cell')),title:rect(element.querySelector('text')),
      qRows:[...element.querySelectorAll('.kf-pa-map__q')].map(rect),segments:[...strip.querySelectorAll('.kf-pa-map__seq-track')].map(rect)};
  }
  function enterAnatomy(id) {
    if(!Number.isInteger(id)||!anatomyModel.tasks[id]||store.getState().views.assignment.subview==='block')return;
    tooltip.hidden=true;hoverTask=null;if(store.getState().selection?.task!==id)selectTask(id);tabTransition?.cancel();
    const transition=semanticZoom.prepare('in',id);store.dispatch({type:'anatomy-enter',id});semanticZoom.play(transition);
  }
  function exitAnatomy() {
    const state=store.getState();if(state.views.assignment.subview!=='block')return;
    anatomyController?.hideTooltip();
    const transition=semanticZoom.prepare('out',state.views.assignment.anatomyTask);
    store.dispatch({type:'anatomy-exit'});semanticZoom.play(transition);
  }
  function enterPipelineIntra(id){
    const state=store.getState();if(!Number.isInteger(id)||!pipelineIntraModel.tasks[id]||state.views.pipeline.subview==='block')return;
    pipelineController.hideTooltip();tabTransition?.cancel();
    const transition=pipelineZoom.prepare('in',id);store.dispatch({type:'pipeline-intra-enter',id});pipelineZoom.play(transition);
  }
  function exitPipelineIntra(){
    const state=store.getState();if(state.views.pipeline.subview!=='block')return;
    pipelineIntraController?.hideTooltip();const transition=pipelineZoom.prepare('out',state.views.pipeline.intraTask);
    store.dispatch({type:'pipeline-intra-exit'});pipelineZoom.play(transition);
  }
  function cancelGraphZoom(){if(!graphZoomJob)return;graphZoomJob.animations.forEach(a=>a.cancel());graphZoomJob.overlay.remove();graphZoomJob.hold.remove();graphZoomJob=null;}
  function graphZoomPrepare(entering){
    cancelGraphZoom();if(matchMedia('(prefers-reduced-motion: reduce)').matches)return null;
    const pane=root.querySelector(entering?'#viz2-graph-inter':'#viz2-graph-intra'),r=pane.getBoundingClientRect(),o=root.getBoundingClientRect(),hold=pane.cloneNode(true);
    hold.removeAttribute('id');hold.inert=true;Object.assign(hold.style,{position:'absolute',left:`${r.left-o.left}px`,top:`${r.top-o.top}px`,width:`${r.width}px`,height:`${r.height}px`,zIndex:12,pointerEvents:'none'});root.append(hold);
    const source=entering?graphController?.getAttentionRect():graphIntraController?.getTransitionRect();
    graphZoomJob={hold,source,entering,animations:[],overlay:document.createElementNS('http://www.w3.org/2000/svg','svg')};return graphZoomJob;
  }
  function graphZoomPlay(job){if(!job)return;requestAnimationFrame(()=>requestAnimationFrame(()=>{if(job!==graphZoomJob)return;
    const target=job.entering?graphIntraController?.getTransitionRect():graphController?.getAttentionRect();if(!target||!job.source){cancelGraphZoom();return;}
    job.running=true;const o=root.getBoundingClientRect(),overlay=job.overlay;overlay.classList.add('viz2-semantic-overlay');overlay.setAttribute('width',root.clientWidth);overlay.setAttribute('height',root.clientHeight);overlay.dataset.graphSemanticTask=store.getState().selection?.task;root.append(overlay);
    const rect=document.createElementNS(overlay.namespaceURI,'rect');rect.setAttribute('fill','none');rect.setAttribute('stroke','var(--primary)');rect.setAttribute('rx',8);overlay.append(rect);
    const destination=root.querySelector(job.entering?'#viz2-graph-intra':'#viz2-graph-inter');job.animations.push(job.hold.animate([{opacity:1},{opacity:0}],{duration:600,fill:'both'}),destination.animate([{opacity:0},{opacity:1}],{duration:700,fill:'both'}));
    // Interpolate source graph region to the destination Task canvas in screen space.
    const from=job.source;const localX=from.left-o.left,localY=from.top-o.top;
    job.hold.style.transformOrigin=`${localX-parseFloat(job.hold.style.left)}px ${localY-parseFloat(job.hold.style.top)}px`;
    job.animations.push(job.hold.animate([{transform:'translate(0,0) scale(1,1)',opacity:1},{transform:`translate(${target.left-from.left}px,${target.top-from.top}px) scale(${target.width/from.width},${target.height/from.height})`,opacity:0}],{duration:720,easing:'cubic-bezier(.2,.8,.2,1)',fill:'both'}));
    const start=performance.now();function frame(now){if(job!==graphZoomJob)return;const p=Math.min(1,(now-start)/720),t=1-(1-p)**3;for(const [key,a,b] of [['x',from.left-o.left,target.left-o.left],['y',from.top-o.top,target.top-o.top],['width',from.width,target.width],['height',from.height,target.height]])rect.setAttribute(key,a+(b-a)*t);rect.setAttribute('opacity',Math.sin(Math.PI*p));if(p<1)requestAnimationFrame(frame);else cancelGraphZoom();}frame(start);
  }));}
  function enterGraphIntra(previous,taskId){const s=store.getState();if(s.views.graph.subview==='block')return;const id=taskId??s.selection?.task??85;if(!graphIntraModel.tasks[id])return;graphOuterSnapshot=previous||{camera:s.views.graph.camera,nodeId:s.views.graph.nodeId,inspectorOpen:s.views.graph.inspectorOpen};const job=graphZoomPrepare(true);store.dispatch({type:'graph-intra-enter',id});graphZoomPlay(job);}
  function exitGraphIntra(){if(store.getState().views.graph.subview!=='block')return;const job=graphZoomPrepare(false);store.dispatch({type:'graph-intra-exit'});graphZoomPlay(job);}
  addEventListener('resize',cancelGraphZoom);matchMedia('(prefers-reduced-motion: reduce)').addEventListener('change',cancelGraphZoom);
  function selectTask(id) { const t=taskInfo(id); store.dispatch({type:'select',selection:{kind:'task',task:id,block:t.block,batch:t.batch,kvHead:t.kv}}); }
  function fitted() { const scale=Math.max(.05,Math.min((viewport.clientWidth-32)/grid.width,(viewport.clientHeight-24)/grid.height,1.2)); return {scale,x:(viewport.clientWidth-grid.width*scale)/2,y:(viewport.clientHeight-grid.height*scale)/2}; }
  function fit() { if (viewport.clientWidth && viewport.clientHeight) store.dispatch({type:'view',id:'assignment',patch:{camera:fitted()}}); }
  function zoomBy(factor, point={x:viewport.clientWidth/2,y:viewport.clientHeight/2}, pointedTask=null) {
    const old=store.getState().views.assignment.camera || fitted(), scale=Math.max(.05,Math.min(8,old.scale*factor)), ratio=scale/old.scale;
    if(factor>1&&old.scale<2.8&&scale>=2.8&&store.getState().views.assignment.phase==='attention'){
      const id=pointedTask??store.getState().selection?.task;
      // The threshold is only a trigger. Enter from the currently painted matrix,
      // exactly as the scope button and double-click do, without an extra camera step.
      if(Number.isInteger(id)){enterAnatomy(id);return;}
    }
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
      if (!tabTransition?.active && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        tabAnimation = root.querySelector(`#viz2-${state.activeTab}`).animate(
          [{opacity:0,transform:'translateY(6px)'},{opacity:1,transform:'translateY(0)'}],
          {duration:220,easing:'cubic-bezier(.2,.8,.2,1)'});
      }
    }
    if (state.activeTab === 'pipeline' && !pipelineController) {
      const model = window.PtoVisualization2Pipeline.createModel({ ...constants, seqLens,
        pageTokens: sourceNumber('BLOCK_SIZE'), pagesPerStack: sourceNumber('STACK_PAGES'), phase0Model });
      pipelineController = window.PtoVisualization2Pipeline.mount(root.querySelector('#viz2-pipeline-inter'), {
        model,
        onSelect: selection => store.dispatch({type:'select',selection}),
        onCamera: camera => store.dispatch({type:'view',id:'pipeline',patch:{camera}}),
        onEnter: enterPipelineIntra,
      });
    }
    const pipeline=state.views.pipeline,pipelineInner=pipeline.subview==='block';
    if(state.activeTab==='pipeline'&&pipelineInner&&action.type==='tab'){
      if(state.selection?.kind!=='task'){store.dispatch({type:'pipeline-intra-exit'});return;}
      if(state.selection.task!==pipeline.intraTask||JSON.stringify(state.selection.subobject??null)!==JSON.stringify(pipeline.intraFocus?.[pipeline.intraTask]??null)){store.dispatch({type:'pipeline-intra-task',id:state.selection.task});return;}
    }
    root.querySelector('#viz2-pipeline-inter').hidden=pipelineInner;
    root.querySelector('#viz2-pipeline-intra').hidden=!pipelineInner;
    root.querySelector('[data-pipeline-back]').hidden=!pipelineInner;
    root.querySelectorAll('[data-pipeline-scope]').forEach(button=>{const selected=(button.dataset.pipelineScope==='intra')===pipelineInner;button.classList.toggle('is-selected',selected);button.setAttribute('aria-pressed',String(selected));if(button.dataset.pipelineScope==='intra')button.disabled=!pipelineInner&&state.selection?.kind!=='task';});
    const pf=state.selection?.subobject;
    root.querySelector('[data-pipeline-breadcrumb]').textContent=state.selection?.task!=null?`Pipeline / Block ${String(state.selection.block).padStart(2,'0')} / Task ${state.selection.task}${pipelineInner&&pf?` / ${pf.event||pf.stage||({stack:'Stack',page:'Page',slot:'Slot',tick:'Tick',output:'Output'}[pf.kind]||pf.kind)} ${pf.index}`:''}`:'Pipeline';
    pipelineController?.update(state.selection,pipeline.camera,
      {active:state.activeTab==='pipeline'&&!pipelineInner,enter:action.type==='tab'&&state.activeTab==='pipeline'});
    if(state.activeTab==='pipeline'&&pipelineInner&&!pipelineIntraController){
      pipelineIntraController=window.PtoVisualization2PipelineIntra.mount(root.querySelector('#viz2-pipeline-intra'),{model:pipelineIntraModel,
        onTask:id=>store.dispatch({type:'pipeline-intra-task',id}),onFocus:focus=>store.dispatch({type:'pipeline-intra-focus',focus}),
        onCamera:(id,camera)=>store.dispatch({type:'view',id:'pipeline',patch:{intraCameras:{...store.getState().views.pipeline.intraCameras,[id]:camera}}}),
        onInspector:open=>store.dispatch({type:'view',id:'pipeline',patch:{intraInspectorOpen:open}})});
    }
    pipelineIntraController?.update(pipeline.intraTask,state.selection?.subobject||null,pipeline.intraCameras?.[pipeline.intraTask]||null,{active:state.activeTab==='pipeline'&&pipelineInner,inspectorOpen:pipeline.intraInspectorOpen});
    const gv=state.views.graph,gi=gv.subview==='block';
    if(state.activeTab==='graph'&&gi&&action.type==='tab'&&state.selection?.kind==='task'&&(gv.intraTask!==state.selection.task||JSON.stringify(gv.intraFocus)!==JSON.stringify(state.selection.subobject))){store.dispatch({type:'graph-intra-task',id:state.selection.task});return;}
    root.querySelector('#viz2-graph-inter').hidden=gi;root.querySelector('#viz2-graph-intra').hidden=!gi;root.querySelector('[data-graph-back]').hidden=!gi;
    root.querySelectorAll('[data-graph-scope]').forEach(b=>{const yes=(b.dataset.graphScope==='intra')===gi;b.classList.toggle('is-selected',yes);b.setAttribute('aria-pressed',String(yes));});
    root.querySelector('[data-graph-breadcrumb]').textContent=state.selection?.task!=null?`Graph / Block ${state.selection.block} / Task ${state.selection.task}${gi&&state.selection.subobject?` / ${state.selection.subobject.node||state.selection.subobject.stage||state.selection.subobject.kind} ${state.selection.subobject.index??''}`:''}`:'Graph';
    if(state.activeTab==='graph'&&!graphController)graphController=window.PtoVisualization2Graph.mount(root.querySelector('#viz2-graph-inter'),{
      execution:{blocks:constants.blocks,taskCount:seqLens.length*constants.kvHeads,phase0Model,seqLens},onSelect:selection=>store.dispatch({type:'select',selection}),onView:patch=>store.dispatch({type:'view',id:'graph',patch}),onEnter:enterGraphIntra,
      onBlockEnter:block=>{const selected=store.getState().selection;const first=graphIntraModel.tasks.find(task=>task.block===block);if(first)enterGraphIntra(null,selected?.kind==='task'&&selected.block===block?selected.task:first.id);}});
    graphController?.update(state.selection,gv,{active:state.activeTab==='graph'&&!gi,enter:action.type==='tab'&&state.activeTab==='graph'&&!gi,restore:action.type==='graph-intra-exit'});
    if(state.activeTab==='graph'&&gi&&!graphIntraController)graphIntraController=window.PtoVisualization2GraphIntra.mount(root.querySelector('#viz2-graph-intra'),{model:graphIntraModel,onTask:id=>store.dispatch({type:'graph-intra-task',id}),onFocus:focus=>store.dispatch({type:'graph-intra-focus',focus}),onView:patch=>store.dispatch({type:'view',id:'graph',patch})});
    graphIntraController?.update(gv.intraTask,state.selection?.subobject,gv,{active:state.activeTab==='graph'&&gi});
    if (state.activeTab !== 'assignment') {anatomyController?.update(null,null,null,{active:false});return;}
    const assignment=state.views.assignment,inner=assignment.subview==='block';
    if(inner&&action.type==='tab'){
      if(state.selection?.kind!=='task'){store.dispatch({type:'anatomy-exit'});return;}
      if(state.selection.task!==assignment.anatomyTask||JSON.stringify(state.selection.subobject??null)!==JSON.stringify(assignment.anatomyFocus?.[assignment.anatomyTask]??null)){store.dispatch({type:'anatomy-task',id:state.selection.task});return;}
    }
    root.querySelector('#viz2-anatomy').hidden=!inner;
    root.querySelector('[data-viz2-phases]').hidden=inner;
    root.querySelector('[data-viz2-matrix-back]').hidden=!inner;
    root.querySelectorAll('[data-viz2-scope]').forEach(button=>{const selected=(button.dataset.viz2Scope==='intra')===inner;button.classList.toggle('is-selected',selected);button.setAttribute('aria-pressed',String(selected));if(button.dataset.viz2Scope==='intra')button.disabled=!inner&&(assignment.phase!=='attention'||state.selection?.kind!=='task');});
    const detailFocus=inner?assignment.anatomyFocus?.[assignment.anatomyTask]:null;
    const detailName=detailFocus?({q:'Q Tile',output:'Attention Output'}[detailFocus.kind]||`${{page:'Page',stack:'Stack',slot:'Slot'}[detailFocus.kind]} ${detailFocus.index}`):'';
    root.querySelector('[data-viz2-breadcrumb]').textContent=state.selection?.task!=null?`Assignment / Block ${String(state.selection.block).padStart(2,'0')} / Task ${state.selection.task}${detailName?` / ${detailName}`:''}`:'Assignment';
    if(inner){
      root.querySelector('#viz2-phase0').hidden=true;root.querySelector('#viz2-attention').hidden=true;
      if(!anatomyController)anatomyController=window.PtoVisualization2Anatomy.mount(root.querySelector('#viz2-anatomy'),{model:anatomyModel,
        onTask:id=>store.dispatch({type:'anatomy-task',id}),onFocus:focus=>store.dispatch({type:'anatomy-focus',focus}),
        onExit:exitAnatomy,
        onCamera:(id,camera)=>store.dispatch({type:'view',id:'assignment',patch:{anatomyCameras:{...store.getState().views.assignment.anatomyCameras,[id]:camera}}}),
        onInspector:open=>store.dispatch({type:'view',id:'assignment',patch:{anatomyInspectorOpen:open}})});
      anatomyController.update(assignment.anatomyTask,assignment.anatomyFocus?.[assignment.anatomyTask]||null,assignment.anatomyCameras?.[assignment.anatomyTask]||null,{active:true,inspectorOpen:assignment.anatomyInspectorOpen});return;
    }
    anatomyController?.update(null,null,null,{active:false});
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
