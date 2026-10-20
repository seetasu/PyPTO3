/* Phase0 assignment from paged_attention_pypto.py, lines 127–145 and 253–263.
 * Geometry expresses ownership and a logical barrier; it carries no timing data.
 */
(function (global) {
  'use strict';
  function createModel({ blocks, kvHeads, qPerKv, activeBatch, ropeCores, itemsPerCore, lanesPerBlock }) {
    const total = activeBatch * kvHeads;
    const lanes = Array.from({ length: blocks * lanesPerBlock }, (_, ropeCore) => {
      const groups = ropeCore >= ropeCores ? [] : Array.from({ length: itemsPerCore }, (_, it) => {
        const gIndex = ropeCore + it * ropeCores;
        const kvHead = Math.floor(gIndex / activeBatch), batch = gIndex % activeBatch;
        return { gIndex, it, batch, kvHead, qStart: kvHead * qPerKv, qEnd: (kvHead + 1) * qPerKv - 1 };
      }).filter(group => group.gIndex < total);
      return { ropeCore, block: Math.floor(ropeCore / lanesPerBlock), lane: ropeCore % lanesPerBlock, groups };
    });
    return { blocks, kvHeads, activeBatch, ropeCores, qPerKv, total, lanesPerBlock, itemsPerCore, lanes,
      activeLanes: lanes.filter(lane => lane.groups.length),
      groupsForBlock: block => lanes.filter(lane => lane.block === block).flatMap(lane => lane.groups),
      ownerOf: (batch, kvHead) => lanes[(kvHead * activeBatch + batch) % ropeCores] };
  }
  const size = { width: 1160 }, columns = 8;
  function mount(host, { model, onSelect, onCamera }) {
    host.innerHTML = `<div class="viz2-p0-summary"><span><b>${model.blocks}</b> Blocks × ${model.lanesPerBlock} AIV Lanes</span><span><b>${model.activeLanes.length}</b> 条参与 Lane</span><span><b>${model.total}</b> 组 Batch × KV Head</span><span>每条参与 Lane <b>${model.activeLanes[0]?.groups.length ?? 0}</b> 组</span></div>
      <div class="viz2-p0-body">
        <div class="viz2-p0-stage">
          <div class="viz2-p0-operations"><b>Phase0</b><span>Q/K Norm</span><i>→</i><span>RoPE</span><i>→</i><span>KV Cache Append</span></div>
          <div class="viz2-p0-viewport" tabindex="0" role="group" aria-label="Phase0 分配画布，拖拽平移，滚轮缩放">
            <svg class="viz2-p0-svg" aria-label="24 Blocks 的双 AIV Lane 分配与 SYNCALL 逻辑关系"><g data-p0-scene></g></svg>
            <div class="kf-pa-map__tooltip" role="tooltip" hidden></div>
          </div>
          <footer class="viz2-p0-footer"><span>短条 = 1 组 · 虚线 Block 无预处理工作 · 连接线表示逻辑汇合</span><div class="kf-pa-map__tools"><button class="btn btn-ghost btn-sm" data-p0-zoom="0.8" aria-label="缩小 Phase0">−</button><output data-p0-scale>100%</output><button class="btn btn-ghost btn-sm" data-p0-zoom="1.25" aria-label="放大 Phase0">＋</button><button class="btn btn-ghost btn-sm" data-p0-fit>适应</button></div></footer>
        </div>
        <aside class="pto-ide-frame__pane viz2-p0-inspector" aria-label="Phase0 分配详情" hidden><header class="pto-ide-frame__pane-header"><b data-p0-title></b><button class="btn btn-ghost btn-sm" data-p0-clear aria-label="清除 Phase0 选择">×</button></header><div class="pto-ide-frame__pane-body" data-p0-detail></div></aside>
      </div>`;
    const viewport = host.querySelector('.viz2-p0-viewport'), scene = host.querySelector('[data-p0-scene]'), tooltip = host.querySelector('[role="tooltip"]');
    const rows = Math.ceil(model.blocks / columns), rowStep = 120, startY = 18, bottom = startY + rows * rowStep;
    // The buses collect every Block, including the Blocks without preprocessing work.
    const buses = Array.from({ length: rows }, (_, row) => {
      const y = startY + row * rowStep + 106;
      return `<path class="viz2-p0-link" d="M74 ${y} H1128"/>`;
    }).join('');
    scene.innerHTML = `${buses}<path class="viz2-p0-link" d="M1128 ${startY+106} V${bottom+18} H580 V${bottom+28}"/>
      ${Array.from({ length: model.blocks }, (_, block) => {
        const x = 14 + block % columns * 138, y = startY + Math.floor(block / columns) * rowStep;
        const lanes = model.lanes.filter(lane => lane.block === block), active = lanes.some(lane => lane.groups.length);
        return `<g class="viz2-p0-block${active ? '' : ' is-idle'}" data-p0-block="${block}" transform="translate(${x} ${y})">
          <path class="viz2-p0-link" d="M60 94 V106"/>
          <rect class="viz2-p0-boundary" width="120" height="94" rx="8"/>
          <g class="viz2-p0-block-target" data-p0-pick-block="${block}" role="button" tabindex="0" aria-pressed="false" aria-label="Phase0 Block ${block}，${model.groupsForBlock(block).length} 组工作"><rect class="viz2-p0-hit" width="120" height="32"/><text x="10" y="21">Block ${String(block).padStart(2,'0')}</text></g>
          ${lanes.map(lane => `<g class="viz2-p0-lane" data-p0-lane="${lane.ropeCore}" role="button" tabindex="0" aria-pressed="false" aria-label="Block ${block} AIV Lane ${lane.lane}，rope_core ${lane.ropeCore}，${lane.groups.length} 组工作" transform="translate(8 ${32+lane.lane*28})"><rect class="viz2-p0-lane-fill" width="104" height="24" rx="6"/><text x="6" y="16">AIV ${lane.lane}</text>${Array.from({length:model.itemsPerCore},(_,i)=>`<rect class="viz2-p0-mark${i<lane.groups.length?' is-work':''}" data-p0-mark="${lane.groups[i]?.gIndex ?? ''}" x="${56+i*(44/model.itemsPerCore)}" y="6" width="${Math.max(2,44/model.itemsPerCore-4)}" height="12" rx="2"/>`).join('')}</g>`).join('')}
        </g>`;
      }).join('')}
      <g class="viz2-p0-barrier" transform="translate(365 ${bottom+28})"><rect width="430" height="42" rx="8"/><text x="215" y="26" text-anchor="middle">SYNCALL · 全部 ${model.blocks} Blocks 参与</text><title>cacheinvalid → fence → SYNCALL(MIX) → cacheinvalid；逻辑屏障，不表示真实耗时</title></g>
      <path class="viz2-p0-link" d="M580 ${bottom+70} V${bottom+91}"/><text class="viz2-p0-next" x="580" y="${bottom+112}" text-anchor="middle">Attention · 所有 Blocks 进入下一阶段</text>`;
    let current = null, camera = null, drag = null, suppressClick = false, lastSize = null;
    const fitted = () => { const scale = Math.max(.05, Math.min((viewport.clientWidth-24)/size.width,(viewport.clientHeight-16)/(bottom+132),1.2)); return {scale,x:(viewport.clientWidth-size.width*scale)/2,y:(viewport.clientHeight-(bottom+132)*scale)/2}; };
    const fit = () => { if (viewport.clientWidth && viewport.clientHeight) onCamera(fitted()); };
    function zoom(factor, point = {x:viewport.clientWidth/2,y:viewport.clientHeight/2}) {
      const old = camera || fitted(), scale = Math.max(.08,Math.min(6,old.scale*factor)), ratio = scale/old.scale;
      onCamera({scale,x:point.x-(point.x-old.x)*ratio,y:point.y-(point.y-old.y)*ratio});
    }
    const mapped = selection => {
      if (selection?.kind === 'task') { const lane = model.ownerOf(selection.batch,selection.kvHead); return { ...selection, block:lane.block, lane:lane.lane, ropeCore:lane.ropeCore, gIndex:selection.kvHead*model.activeBatch+selection.batch }; }
      return selection;
    };
    const groupRows = groups => `<table class="viz2-p0-table"><thead><tr><th>工作项 g_idx</th><th>Batch</th><th>KV Head</th></tr></thead><tbody>${groups.map(group=>`<tr${group.gIndex===mapped(current)?.gIndex?' class="is-focused"':''}><td>${group.gIndex}</td><td>${group.batch}</td><td>${group.kvHead}</td></tr>`).join('')}</tbody></table>`;
    function detail(selection) {
      const inspector=host.querySelector('aside'), s=mapped(selection); inspector.hidden=!s;
      if (!s) return;
      const lanes=model.lanes.filter(lane=>lane.block===s.block && (s.lane==null || lane.lane===s.lane));
      const groups=lanes.flatMap(lane=>lane.groups);
      host.querySelector('[data-p0-title]').textContent=`Block ${s.block}${s.lane==null?'':` · AIV ${s.lane}`}`;
      host.querySelector('[data-p0-detail]').innerHTML=`${groups.length ? `<p>${groups.length} 组预处理工作${s.kind==='task'?` · 焦点 Batch ${s.batch} / KV ${s.kvHead}`:''}</p>
        ${lanes.map(lane=>`<section class="viz2-p0-detail-section"><h3>AIV ${lane.lane} <small>rope_core ${lane.ropeCore}</small></h3>${groupRows(lane.groups)}</section>`).join('')}
        <section class="viz2-p0-detail-section"><h3>每组工作</h3><p>Q/K Norm、RoPE；追加当前 token 的 K/V 至 KV Cache，写回旋转后的 Q。</p><p>每个 KV Head 对应 ${model.qPerKv} 个 Q Heads。</p></section>` : `<p>无有效预处理工作</p><p>rope_core ${lanes.map(lane=>lane.ropeCore).join(' / ')} ≥ ROPE_CORES ${model.ropeCores}，跳过 Phase0 工作循环。</p>`}
        <section class="viz2-p0-detail-section"><h3>同步后进入 Attention</h3><p>该 Block 仍参与 SYNCALL。</p><code>cacheinvalid → fence<br>→ SYNCALL(MIX)<br>→ cacheinvalid</code><p>连接线表达逻辑同步，不表示实际执行时间。</p></section>`;
    }
    function update(selection, nextCamera, { selectionChanged = true } = {}) {
      current=selection; camera=nextCamera;
      const s=mapped(selection);
      scene.querySelectorAll('[data-p0-block]').forEach(el=>el.classList.toggle('is-selected',Number(el.dataset.p0Block)===s?.block));
      scene.querySelectorAll('[data-p0-pick-block]').forEach(el=>el.setAttribute('aria-pressed',String(Number(el.dataset.p0PickBlock)===s?.block && s?.lane==null)));
      scene.querySelectorAll('[data-p0-lane]').forEach(el=>{const lane=model.lanes[Number(el.dataset.p0Lane)], active=lane.block===s?.block && (s.lane==null || lane.lane===s.lane); el.classList.toggle('is-selected',active); el.setAttribute('aria-pressed',String(active));});
      scene.querySelectorAll('[data-p0-mark]').forEach(el=>el.classList.toggle('is-focused',el.dataset.p0Mark!=='' && Number(el.dataset.p0Mark)===s?.gIndex));
      if (selectionChanged) detail(selection);
      if (!camera && viewport.clientWidth) { fit(); return; }
      if (camera) { scene.style.setProperty('--p0-label-size', `${Math.max(12, 11/camera.scale)}px`); scene.setAttribute('transform',`translate(${camera.x} ${camera.y}) scale(${camera.scale})`); host.querySelector('[data-p0-scale]').textContent=`${Math.round(camera.scale/fitted().scale*100)}%`; }
    }
    host.addEventListener('click',event=>{
      if(event.target.closest('[data-p0-clear]')) { onSelect(null); return; }
      if(event.target.closest('[data-p0-fit]')) { fit(); return; }
      const z=event.target.closest('[data-p0-zoom]'); if(z) {zoom(Number(z.dataset.p0Zoom));return;}
      if(suppressClick) {suppressClick=false;return;}
      const lane=event.target.closest('[data-p0-lane]'), block=event.target.closest('[data-p0-pick-block]');
      if(lane) {const l=model.lanes[Number(lane.dataset.p0Lane)];onSelect({kind:'lane',block:l.block,lane:l.lane,ropeCore:l.ropeCore,task:null});}
      else if(block) onSelect({kind:'block',block:Number(block.dataset.p0PickBlock),task:null});
    });
    viewport.addEventListener('keydown',event=>{if(['Enter',' '].includes(event.key) && event.target.closest('[role="button"]')) {event.preventDefault();event.target.closest('[role="button"]').dispatchEvent(new MouseEvent('click',{bubbles:true}));} else if(event.key==='Escape') {onSelect(null);viewport.focus();}});
    viewport.addEventListener('wheel',event=>{event.preventDefault();const r=viewport.getBoundingClientRect(),unit=event.deltaMode===1?16:event.deltaMode===2?r.height:1;zoom(Math.exp(Math.max(-.22,Math.min(.22,-event.deltaY*unit*.003))),{x:event.clientX-r.left,y:event.clientY-r.top});},{passive:false});
    viewport.addEventListener('pointerdown',event=>{if(event.button!==0)return;suppressClick=false;drag={x:event.clientX,y:event.clientY,startX:event.clientX,startY:event.clientY,id:event.pointerId,moved:false};});
    viewport.addEventListener('pointermove',event=>{
      if(drag) {if(!drag.moved && Math.hypot(event.clientX-drag.startX,event.clientY-drag.startY)<=4)return;drag.moved=true;viewport.setPointerCapture(drag.id);viewport.classList.add('is-dragging');tooltip.hidden=true;const c=camera||fitted();onCamera({...c,x:c.x+event.clientX-drag.x,y:c.y+event.clientY-drag.y});drag.x=event.clientX;drag.y=event.clientY;return;}
      const lane=event.target.closest('[data-p0-lane]'),block=event.target.closest('[data-p0-pick-block]');tooltip.hidden=!lane&&!block;
      if(lane||block) {const l=lane?model.lanes[Number(lane.dataset.p0Lane)]:null, groups=l?l.groups:model.groupsForBlock(Number(block.dataset.p0PickBlock));tooltip.textContent=`${l?`rope_core ${l.ropeCore} · `:''}${groups.length?groups.map(g=>`g${g.gIndex}: B${g.batch}/KV${g.kvHead}`).join(' · '):'无有效预处理工作 · 仍参与 SYNCALL'}`;const r=viewport.getBoundingClientRect();tooltip.style.left=`${Math.max(8,Math.min(event.clientX-r.left+12,r.width-tooltip.offsetWidth-8))}px`;tooltip.style.top=`${Math.max(8,Math.min(event.clientY-r.top+16,r.height-tooltip.offsetHeight-8))}px`;}
    });
    const endDrag=()=>{if(drag?.moved)suppressClick=true;drag=null;viewport.classList.remove('is-dragging');};
    ['pointerup','pointercancel','lostpointercapture'].forEach(type=>viewport.addEventListener(type,endDrag));viewport.addEventListener('pointerleave',()=>{tooltip.hidden=true;});
    new ResizeObserver(()=>{const width=viewport.clientWidth,height=viewport.clientHeight;if(!width||!height)return;if(!camera)fit();else if(lastSize&&(width!==lastSize.width||height!==lastSize.height)){const previousFit=Math.max(.05,Math.min((lastSize.width-24)/size.width,(lastSize.height-16)/(bottom+132),1.2));const ratio=fitted().scale/previousFit;onCamera({scale:camera.scale*ratio,x:width/2+(camera.x-lastSize.width/2)*ratio,y:height/2+(camera.y-lastSize.height/2)*ratio});}lastSize={width,height};}).observe(viewport);
    return { update, hideTooltip:()=>{tooltip.hidden=true;}, model };
  }
  global.PtoVisualization2Phase0=Object.freeze({createModel,mount});
})(window);
