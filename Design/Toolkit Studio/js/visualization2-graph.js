/* Block-level mathematical graph: share the original semantic model, layout and
 * scene renderer. Runtime ownership and mathematical node identity stay distinct.
 */
(function (global) {
  'use strict';
  const escape = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  function resolveFocus(model, selection, nodeId) {
    const explicit = selection?.graphNodeId || nodeId;
    if (model.nodes.some(node => node.id === explicit)) return { nodeId: explicit, related: [explicit], kind: 'node' };
    if (selection?.kind === 'task') {
      const regions = model.groups.filter(group => ['paged-attention', 'output-region'].includes(group.id));
      return { nodeId: null, related: [...new Set(regions.flatMap(group => group.members))].filter(id => id !== 'cache-append'), kind: 'task' };
    }
    if (selection?.kind === 'lane' && selection.ropeCore < model.constants.ROPE_CORES.value) {
      const prep = model.groups.find(group => group.id === 'qkv-preparation');
      return { nodeId: null, related: [...(prep?.members || []).filter(id => model.nodes.find(node=>node.id===id)?.type === 'composite'), 'cache-append'], kind: 'lane' };
    }
    return { nodeId: null, related: [], kind: 'none' };
  }
  function mount(host, { onSelect, onView }) {
    const model = global.PTO_PYPA_SEMANTIC_GRAPH;
    const graph = global.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize(model, new Set());
    host.classList.add('viz2-graph');
    host.innerHTML = `<div class="viz2-graph-body"><div class="viz2-graph-stage">
      <div class="kf-pypa-graph__viewport viz2-graph-viewport" role="group" tabindex="0" aria-label="Block 间完整计算图，滚轮缩放，拖拽平移"><div class="kf-pypa-graph__scene viz2-graph-scene" data-viz2-graph-scene></div><div class="kf-pa-map__tooltip" role="tooltip" hidden></div></div>
      <footer class="viz2-graph-footer"><span data-viz2-graph-context>完整计算图</span><div class="kf-pa-map__tools"><button class="btn btn-ghost btn-sm" type="button" data-viz2-graph-zoom="0.8" aria-label="缩小 Graph">−</button><output data-viz2-graph-scale>100%</output><button class="btn btn-ghost btn-sm" type="button" data-viz2-graph-zoom="1.25" aria-label="放大 Graph">＋</button><button class="btn btn-ghost btn-sm" type="button" data-viz2-graph-fit>Fit to View</button></div></footer>
      </div><aside class="pto-ide-frame__pane viz2-graph-inspector" aria-label="Graph 节点详情" hidden><header class="pto-ide-frame__pane-header"><b data-viz2-graph-title></b><button class="btn btn-ghost btn-sm" type="button" data-viz2-graph-close aria-label="关闭 Graph 详情">×</button></header><div class="pto-ide-frame__pane-body" data-viz2-graph-detail></div></aside></div>`;
    const viewport = host.querySelector('.viz2-graph-viewport'), scene = host.querySelector('[data-viz2-graph-scene]');
    // Reuse only the original node skin; no legacy pane chrome is mounted.
    scene.classList.add('has-semantic-overlays');
    scene.dataset.overlayControl='false';
    const inspector = host.querySelector('aside'), tooltip = host.querySelector('[role="tooltip"]');
    // Root-level graph only; mathematical composites are not expanded into in-core graphs.
    const rendered = global.PtoStudioComputationGraph.renderScene(scene, graph, {
      view:'compute', allowExpand:false, synchronizations:false, markerPrefix:'viz2-graph',
      cardHtml:node=>global.PtoStudioComputationGraph.cardHtml(node,model,{shape:false,precision:false,memory:false,control:false}),
    });
    const nodes = rendered.placedNodes, nodeMap = new Map(nodes.map(node=>[node.id,node]));
    let selection = null, view = {}, active = false, camera = null, first = true, lastObjectKey, lastFocusKey = '', inspectorKey = '';
    let drag = null, suppressClick = false, lastSize = null, pendingFocus = false;
    const bounds = ids => {
      const items=ids.map(id=>nodeMap.get(id)).filter(Boolean);
      if(!items.length)return null;
      const minX=Math.min(...items.map(n=>n.x)),minY=Math.min(...items.map(n=>n.y));
      return {x:minX,y:minY,w:Math.max(...items.map(n=>n.x+n.w))-minX,h:Math.max(...items.map(n=>n.y+n.h))-minY};
    };
    const fitted = () => {
      const scale=Math.max(.01,Math.min((viewport.clientWidth-48)/rendered.sceneWidth,(viewport.clientHeight-48)/rendered.sceneHeight,1));
      return {scale,x:(viewport.clientWidth-rendered.sceneWidth*scale)/2,y:(viewport.clientHeight-rendered.sceneHeight*scale)/2};
    };
    function fit() {if(viewport.clientWidth&&viewport.clientHeight)onView({camera:fitted()});}
    function paintCamera() {
      if(!camera)return;
      scene.style.transform=`translate(${camera.x}px, ${camera.y}px) scale(${camera.scale})`;
      // Shared compact-density rules hide secondary labels, retaining original node colors/shapes.
      scene.dataset.density='compact';
      host.querySelector('[data-viz2-graph-scale]').textContent=`${Math.round(camera.scale*100)}%`;
    }
    function focusRelated(focus) {
      const box=bounds(focus.related);if(!box||!camera)return;
      // Fit remains explicit. On a later cross-view entry, frame only the mapped objects.
      const scale=Math.max(.01,Math.min((viewport.clientWidth-80)/(box.w+80),(viewport.clientHeight-80)/(box.h+80),.9));
      onView({camera:{scale,x:viewport.clientWidth/2-(box.x+box.w/2)*scale,y:viewport.clientHeight/2-(box.y+box.h/2)*scale}});
    }
    function inspectorHtml(node) {
      const sections=global.PTO_COMPUTATION_PRESENTATION.inspector(node,{model,qualifiedId:node.id});
      const primary=sections.filter(section=>['语义','属性','依赖','源码'].includes(section.title));
      const advanced=sections.filter(section=>!primary.includes(section));
      const sectionHtml=section=>`<section><h3>${escape(section.title)}</h3><dl>${section.rows.map(row=>`<dt>${escape(row.label)}</dt><dd>${escape(row.value)}</dd>`).join('')}</dl></section>`;
      return primary.map(sectionHtml).join('')+(advanced.length?`<details><summary>更多属性</summary>${advanced.map(sectionHtml).join('')}</details>`:'');
    }
    function update(nextSelection,nextView,options={}) {
      selection=nextSelection;view=nextView;active=!!options.active;camera=view.camera;
      const objectKey=JSON.stringify(selection ? [selection.kind,selection.block,selection.task,selection.lane] : null);
      const changed=lastObjectKey!==undefined&&objectKey!==lastObjectKey;lastObjectKey=objectKey;
      if(changed&&!selection?.graphNodeId&&view.nodeId){onView({nodeId:null,inspectorOpen:false});return;}
      const focus=resolveFocus(model,selection,view.nodeId), key=JSON.stringify([focus,objectKey]);
      if(options.enter){tooltip.hidden=true;pendingFocus=true;}
      if(key!==lastFocusKey){lastFocusKey=key;pendingFocus=true;}
      scene.querySelectorAll('[data-pypa-node]').forEach(el=>{
        const selected=el.dataset.pypaNode===focus.nodeId,related=focus.related.includes(el.dataset.pypaNode);
        el.classList.toggle('is-selected',selected);el.classList.toggle('is-viz2-related',related&&!selected);el.setAttribute('aria-pressed',String(selected));
      });
      scene.querySelectorAll('[data-pypa-edge]').forEach(el=>{
        const edge=graph.edges.find(edge=>edge.id===el.dataset.pypaEdge);
        el.classList.toggle('is-viz2-related',!!edge&&focus.related.includes(edge.s)&&focus.related.includes(edge.t));
      });
      const node=nodeMap.get(focus.nodeId), show=!!node&&view.inspectorOpen===true;inspector.hidden=!show;
      const ikey=show?node.id:'';
      if(ikey!==inspectorKey){inspectorKey=ikey;if(node&&show){host.querySelector('[data-viz2-graph-title]').textContent=node.title;host.querySelector('[data-viz2-graph-detail]').innerHTML=inspectorHtml(node);}}
      const context=selection?.kind==='task'?`Task ${selection.task} · Batch ${selection.batch} · KV Head ${selection.kvHead}`:selection?.block!=null?`Block ${String(selection.block).padStart(2,'0')}${selection.kind==='lane'?` · AIV Lane ${selection.lane}`:''}`:'完整计算图';
      host.querySelector('[data-viz2-graph-context]').textContent=context;
      if(!active||!viewport.clientWidth||!viewport.clientHeight)return;
      if(first){first=false;pendingFocus=false;fit();return;}
      if(!camera){fit();return;}
      if(pendingFocus){pendingFocus=false;if(focus.related.length){focusRelated(focus);return;}}
      paintCamera();
    }
    function zoom(factor,point={x:viewport.clientWidth/2,y:viewport.clientHeight/2}) {
      if(!camera)return;const scale=Math.max(.01,Math.min(3,camera.scale*factor)),ratio=scale/camera.scale;
      onView({camera:{scale,x:point.x-(point.x-camera.x)*ratio,y:point.y-(point.y-camera.y)*ratio}});
    }
    host.addEventListener('click',event=>{
      // Graph owns these node events; the legacy page has document-level selectors.
      event.stopPropagation(); tooltip.hidden=true;
      if(event.target.closest('[data-viz2-graph-fit]')){fit();return;}
      if(event.target.closest('[data-viz2-graph-close]')){onView({inspectorOpen:false});return;}
      const zoomButton=event.target.closest('[data-viz2-graph-zoom]');if(zoomButton){zoom(Number(zoomButton.dataset.viz2GraphZoom));return;}
      if(suppressClick){suppressClick=false;return;}
      const target=event.target.closest('[data-pypa-node]');if(!target)return;
      const id=target.dataset.pypaNode;
      onView({nodeId:id,inspectorOpen:true});
      if(selection?.block!=null)onSelect({...selection,graphNodeId:id});
    });
    viewport.addEventListener('wheel',event=>{
      event.preventDefault();tooltip.hidden=true;const rect=viewport.getBoundingClientRect(),unit=event.deltaMode===1?16:event.deltaMode===2?rect.height:1;
      zoom(Math.exp(Math.max(-.22,Math.min(.22,-event.deltaY*unit*.003))),{x:event.clientX-rect.left,y:event.clientY-rect.top});
    },{passive:false});
    viewport.addEventListener('pointerdown',event=>{if(event.button!==0)return;suppressClick=false;drag={id:event.pointerId,x:event.clientX,y:event.clientY,startX:event.clientX,startY:event.clientY,moved:false};});
    viewport.addEventListener('pointermove',event=>{
      if(drag){if(!drag.moved&&Math.hypot(event.clientX-drag.startX,event.clientY-drag.startY)<=4)return;drag.moved=true;viewport.setPointerCapture(drag.id);tooltip.hidden=true;const c=camera||fitted();onView({camera:{...c,x:c.x+event.clientX-drag.x,y:c.y+event.clientY-drag.y}});drag.x=event.clientX;drag.y=event.clientY;return;}
      const target=event.target.closest('[data-pypa-node]'),node=target?nodeMap.get(target.dataset.pypaNode):null;tooltip.hidden=!node;
      if(node){tooltip.textContent=`${node.title} · ${node.detail.length>100?node.detail.slice(0,100)+'…':node.detail}`;const rect=viewport.getBoundingClientRect();tooltip.style.left=`${Math.max(8,Math.min(event.clientX-rect.left+14,rect.width-tooltip.offsetWidth-8))}px`;tooltip.style.top=`${Math.max(8,Math.min(event.clientY-rect.top+16,rect.height-tooltip.offsetHeight-8))}px`;}
    });
    const endDrag=()=>{if(drag?.moved)suppressClick=true;drag=null;};['pointerup','pointercancel','lostpointercapture'].forEach(type=>viewport.addEventListener(type,endDrag));
    viewport.addEventListener('pointerleave',()=>{tooltip.hidden=true;});
    viewport.addEventListener('dblclick',event=>{if(!event.target.closest('[data-pypa-node]'))fit();});
    viewport.addEventListener('keydown',event=>{if(event.key==='Escape'){onView({inspectorOpen:false});viewport.focus();}else if(['+','=','-','0'].includes(event.key)){event.preventDefault();if(event.key==='0')fit();else zoom(event.key === '-' ? .8 : 1.25);}});
    new ResizeObserver(()=>{
      if(!active||!viewport.clientWidth||!viewport.clientHeight)return;
      const width=viewport.clientWidth,height=viewport.clientHeight;
      if(first||!camera){first=false;fit();}
      else if(lastSize&&(lastSize.width!==width||lastSize.height!==height)){
        onView({camera:{...camera,x:camera.x+(width-lastSize.width)/2,y:camera.y+(height-lastSize.height)/2}});
      }
      lastSize={width,height};
    }).observe(viewport);
    return {update,hideTooltip:()=>{tooltip.hidden=true;},graph,model};
  }
  global.PtoVisualization2Graph=Object.freeze({mount,resolveFocus});
})(window);
