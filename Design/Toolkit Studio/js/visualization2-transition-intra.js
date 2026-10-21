/* Block-local semantic morph. All geometry is measured from existing rendered objects. */
(function(global){
  'use strict';
  const NS='http://www.w3.org/2000/svg';
  const stageNode={qk:'qk',softmax:'softmax',pv:'pv',accumulate:'context-acc'};
  const nodeStage=Object.fromEntries(Object.entries(stageNode).map(([stage,node])=>[node,stage]));
  const panes={assignment:'#viz2-anatomy',pipeline:'#viz2-pipeline-intra',graph:'#viz2-graph-intra'};
  const clamp=n=>Math.max(0,Math.min(1,n));
  const ease=n=>n<.5?4*n*n*n:1-Math.pow(-2*n+2,3)/2;
  const rect=element=>{const r=element.getBoundingClientRect();return {x:r.left,y:r.top,width:r.width,height:r.height};};
  const mix=(a,b,t)=>a+(b-a)*t;
  const visible=(item,clip)=>item&&item.box.width>0&&item.box.height>0&&item.box.x<clip.x+clip.width&&item.box.x+item.box.width>clip.x&&item.box.y<clip.y+clip.height&&item.box.y+item.box.height>clip.y;
  const svg=(name,attributes,parent)=>{const element=document.createElementNS(NS,name);for(const [key,value] of Object.entries(attributes))element.setAttribute(key,value);parent.append(element);return element;};
  function stackOf(focus,task){
    if(Number.isInteger(focus?.stackIndex))return focus.stackIndex;
    if(['stack','stage','event'].includes(focus?.kind))return focus.index;
    if(focus?.kind==='page')return task?.pages?.[focus.index]?.stack;
    return null;
  }
  function mapFocus(from,to,focus,task){
    if(!focus)return null;
    const stack=stackOf(focus,task),valid=Number.isInteger(stack)&&stack>=0&&stack<task.stacks.length;
    if(to==='assignment'){
      if(focus.kind==='q'||focus.node==='q')return {kind:'q',index:0};
      if(focus.kind==='output'||['output','output-norm'].includes(focus.node))return {kind:'output',index:0};
      if(focus.kind==='page'&&task.pages[focus.index])return {kind:'page',index:focus.index};
      if(focus.kind==='slot'&&task.slots[focus.index])return {kind:'slot',index:focus.index};
      return valid?{kind:'stack',index:stack}:null;
    }
    if(to==='pipeline'){
      if(focus.kind==='output'||['output','output-norm'].includes(focus.node))return {kind:'output',index:0};
      if(focus.kind==='slot'&&task.slots[focus.index])return focus;
      if(focus.kind==='event'&&valid)return focus;
      const stage=focus.kind==='stage'?focus.stage:nodeStage[focus.node?.split('/')[0]];
      if(stage&&valid)return {kind:'stage',index:stack,stage};
      return valid?{kind:'stack',index:stack}:null;
    }
    if(to==='graph'){
      if(focus.kind==='output')return {kind:'output',index:0,stackIndex:valid?stack:null};
      if(focus.kind==='q')return {kind:'graph',node:'q',index:valid?stack:0,stackIndex:valid?stack:null};
      if(focus.kind==='event'&&valid)return {...focus,stackIndex:stack};
      if(focus.kind==='stage'&&stageNode[focus.stage]&&valid)return {...focus,stackIndex:stack};
      if(focus.kind==='page'&&task.pages[focus.index])return {...focus,stackIndex:stack};
      return valid?{kind:'stack',index:stack,stackIndex:stack}:null;
    }
    return null;
  }
  function create({root,getTask}){
    let job=null,serial=0;
    const media=global.matchMedia('(prefers-reduced-motion: reduce)');
    const pane=tab=>root.querySelector(panes[tab]);
    const origin=()=>{const r=root.getBoundingClientRect();return {x:r.left+root.clientLeft,y:r.top+root.clientTop,width:root.clientWidth,height:root.clientHeight};};
    function snapshot(tab,task){
      const host=pane(tab),viewport=host?.querySelector(tab==='graph'?'.viz2-graph-viewport':'.viz2-anatomy-viewport');
      if(!host||!viewport)return null;
      const objects=new Map(),add=(id,element,paint=element)=>{
        if(!element)return;
        const box=rect(element),style=getComputedStyle(paint),vector=paint instanceof SVGElement,item={id,element,box,
          fill:vector?(style.fill&&style.fill!=='none'?style.fill:'transparent'):style.backgroundColor,
          stroke:vector?(style.stroke&&style.stroke!=='none'?style.stroke:'transparent'):style.borderColor,
          radius:parseFloat(style.rx)||parseFloat(style.borderRadius)||5};
        objects.set(id,item);
      };
      if(tab==='assignment'){
        for(const element of host.querySelectorAll('[data-anatomy-kind]')){
          const {anatomyKind:kind,anatomyIndex:index}=element.dataset;
          add(`${kind}:${index}`,element,element.querySelector('rect')||element);
        }
      }else if(tab==='pipeline'){
        for(const element of host.querySelectorAll('[data-intra-kind="stage"]'))add(`stage:${element.dataset.intraIndex}:${element.dataset.intraStage}`,element,element.querySelector('rect')||element);
        const output=host.querySelector('[data-intra-kind="output"]'),cards=output?.querySelectorAll('.viz2-intra-output');
        if(cards?.[0])add('normalize',cards[0]);if(cards?.[1])add('output',cards[1]);
      }else{
        for(const element of host.querySelectorAll('[data-pypa-node]'))add(`node:${element.dataset.pypaNode}`,element,element);
        const operations=['qk','softmax','pv','context-acc'].map(id=>objects.get(`node:${id}`)).filter(Boolean);
        if(operations.length===4){
          const left=Math.min(...operations.map(item=>item.box.x)),top=Math.min(...operations.map(item=>item.box.y));
          const right=Math.max(...operations.map(item=>item.box.x+item.box.width)),bottom=Math.max(...operations.map(item=>item.box.y+item.box.height));
          objects.set('region:stack-loop',{id:'region:stack-loop',element:operations[0].element,elements:operations.map(item=>item.element),box:{x:left,y:top,width:right-left,height:bottom-top},fill:'transparent',stroke:operations[0].stroke,radius:8});
        }
      }
      return {tab,task,host,clip:rect(viewport),objects};
    }
    function pairs(source,target,focus,task){
      const result=[],add=(a,b,kind,label,priority=0)=>{
        const s=source.objects.get(a),t=target.objects.get(b);
        if(visible(s,source.clip)&&visible(t,target.clip))result.push({source:s,target:t,kind,label,priority});
      };
      const direction=`${source.tab}>${target.tab}`,s=stackOf(focus,task),selected=Number.isInteger(s)?s:0;
      if(direction==='assignment>pipeline'||direction==='pipeline>assignment'){
        task.stacks.forEach(stack=>Object.keys(stageNode).forEach((stage,index)=>{
          const a=`stack:${stack.index}`,b=`stage:${stack.index}:${stage}`;
          if(source.tab==='assignment')add(a,b,'stack-stage',`S${stack.index} · ${stage}`,index*38+stack.index*8);
          else add(b,a,'stage-stack',`S${stack.index}`,index*38+stack.index*8);
        }));
        add(source.tab==='assignment'?'output:0':'output',source.tab==='assignment'?'output':'output:0','output','Output',120);
      }else if(direction==='pipeline>graph'||direction==='graph>pipeline'){
        Object.entries(stageNode).forEach(([stage,node],index)=>{
          const a=`stage:${selected}:${stage}`,b=`node:${node}`;
          if(source.tab==='pipeline')add(a,b,'stage-node',stage.toUpperCase(),index*45);
          else add(b,a,'node-stage',`S${selected} · ${stage}`,index*45);
        });
        if(source.tab==='pipeline'){add('normalize','node:output-norm','operation','Normalize',170);add('output','node:output','output','Output',210);}
        else {add('node:output-norm','normalize','operation','Normalize',170);add('node:output','output','output','Output',210);}
      }else if(direction==='assignment>graph'||direction==='graph>assignment'){
        const forward=source.tab==='assignment';
        add(forward?'q:0':'node:q',forward?'node:q':'q:0','data','Q Tile',0);
        const page=focus?.kind==='page'&&task.pages[focus.index]?focus.index:task.stacks[selected]?.pages?.[0];
        if(Number.isInteger(page))for(const [index,node] of ['k-cache','v-cache'].entries())add(forward?`page:${page}`:`node:${node}`,forward?`node:${node}`:`page:${page}`,'data',node==='k-cache'?'K Pages':'V Pages',45+index*35);
        add(forward?`stack:${selected}`:'region:stack-loop',forward?'region:stack-loop':`stack:${selected}`,'region',`S${selected} · Stack Loop`,130);
        add(forward?'output:0':'node:output',forward?'node:output':'output:0','output','Output',190);
      }
      return result;
    }
    function cancel(){
      if(!job)return;
      cancelAnimationFrame(job.frame);
      job.animations?.forEach(animation=>animation.cancel());
      job.masked?.forEach(({element,previous})=>{if(previous)element.style.visibility=previous;else element.style.removeProperty('visibility');});
      job.destination?.style.removeProperty('opacity');
      job.hold?.remove();job.overlay?.remove();
      root.classList.remove('viz2-morph-preparing');delete root.dataset.morphDestination;
      job=null;
    }
    function prepare(state,to){
      cancel();
      const from=state.activeTab,task=state.selection?.task;
      if(!state.open||from===to||media.matches||state.selection?.kind!=='task'||state.views[from]?.subview!=='block'||!Number.isInteger(task))return null;
      const source=snapshot(from,task);
      if(!source)return null;
      const bounds=rect(source.host),o=origin(),hold=source.host.cloneNode(true);
      hold.removeAttribute('id');hold.setAttribute('aria-hidden','true');hold.inert=true;
      Object.assign(hold.style,{position:'absolute',left:`${bounds.x-o.x}px`,top:`${bounds.y-o.y}px`,width:`${bounds.width}px`,height:`${bounds.height}px`,zIndex:11,pointerEvents:'none',margin:0});
      const originals=source.host.querySelectorAll('canvas');hold.querySelectorAll('canvas').forEach((canvas,index)=>canvas.getContext('2d').drawImage(originals[index],0,0));
      root.append(hold);
      const descendants=hold.querySelectorAll('*');source.host.querySelectorAll('*').forEach((element,index)=>{if(element.scrollTop||element.scrollLeft){descendants[index].scrollTop=element.scrollTop;descendants[index].scrollLeft=element.scrollLeft;}});
      const cloneMap=new WeakMap(),bind=(a,b)=>{cloneMap.set(a,b);for(let i=0;i<a.children.length;i++)bind(a.children[i],b.children[i]);};bind(source.host,hold);
      const current={id:++serial,from,to,task,source,focus:state.selection.subobject,hold,cloneMap,frame:0,phase:'preparing',masked:[],animations:[]};
      job=current;root.classList.add('viz2-morph-preparing');root.dataset.morphDestination=to;
      return current;
    }
    function play(current){if(current&&current===job)current.frame=requestAnimationFrame(()=>{current.frame=requestAnimationFrame(()=>start(current));});}
    function start(current){
      if(current!==job)return;
      const target=snapshot(current.to,current.task);
      if(!target||target.task!==current.task){cancel();return;}
      // The source and destination renderers share Task-local Stack/Page IDs.
      const task=getTask(current.task);
      if(!task){cancel();return;}
      const mapping=pairs(current.source,target,current.focus,task);
      if(!mapping.length){cancel();return;}
      const o=origin(),overlay=svg('svg',{class:'viz2-transition-overlay',width:o.width,height:o.height,'aria-hidden':'true','data-direction':`${current.from}-${current.to}`},root);
      current.overlay=overlay;current.destination=target.host;
      // A cloned source holds unmatched content; matched elements are owned by the overlay.
      for(const pair of mapping)for(const element of pair.source.elements||[pair.source.element])current.cloneMap.get(element)?.style.setProperty('visibility','hidden');
      for(const pair of mapping)for(const element of pair.target.elements||[pair.target.element]){if(current.masked.some(item=>item.element===element))continue;current.masked.push({element,previous:element.style.visibility});element.style.setProperty('visibility','hidden');}
      root.classList.remove('viz2-morph-preparing');delete root.dataset.morphDestination;
      const destination=target.host;
      current.animations=[current.hold.animate([{opacity:1},{opacity:0}],{duration:400,fill:'both',easing:'ease-in-out'}),destination.animate([{opacity:0},{opacity:0,offset:.22},{opacity:1}],{duration:690,fill:'both',easing:'ease-in-out'})];
      const items=mapping.map(pair=>{
        const group=svg('g',{'data-intra-morph':`${pair.source.id}>${pair.target.id}`},overlay);
        const box=svg('rect',{rx:pair.source.radius,fill:pair.source.fill||'transparent',stroke:pair.source.stroke||'none','stroke-width':1.5},group);
        const label=svg('text',{'text-anchor':'middle','dominant-baseline':'central','font-size':11,'font-family':'var(--font-mono)','fill':'var(--foreground)'},group);
        label.textContent=pair.label;
        current.animations.push(box.animate([{fill:pair.source.fill||'transparent',stroke:pair.source.stroke||'none'},{fill:pair.target.fill||'transparent',stroke:pair.target.stroke||'none'}],{duration:650,fill:'both',easing:'ease-in-out'}));
        return {...pair,box,label};
      });
      current.phase='animating';const started=performance.now(),duration=680;
      function frame(now){
        if(job!==current)return;
        const elapsed=now-started;
        for(const item of items){
          const p=ease(clamp((elapsed-item.priority*.25)/650)),a=item.source.box,b=item.target.box;
          const x=mix(a.x,b.x,p)-o.x,y=mix(a.y,b.y,p),width=mix(a.width,b.width,p),height=mix(a.height,b.height,p);
          for(const [key,value] of Object.entries({x,y,width,height,rx:mix(item.source.radius,item.target.radius,p)}))item.box.setAttribute(key,String(Math.max(key==='width'||key==='height'?0.1:0,value)));
          item.label.setAttribute('x',String(x+width/2));item.label.setAttribute('y',String(y+height/2));
          item.label.style.opacity=String(Math.min(1,p*4)*(1-Math.max(0,(p-.78)/.22)));
          item.box.style.opacity=String(1-Math.max(0,(p-.88)/.12));
        }
        if(elapsed<duration)current.frame=requestAnimationFrame(frame);else cancel();
      }
      frame(started);
    }
    global.addEventListener('resize',cancel);media.addEventListener('change',cancel);
    root.addEventListener('scroll',event=>{if(job?.phase==='animating'&&!job.hold?.contains(event.target))cancel();},true);
    return {prepare,play,cancel,get active(){return !!job;},get animating(){return job?.phase==='animating';}};
  }
  global.PtoVisualization2IntraMorph=Object.freeze({create,mapFocus,stackOf});
})(window);
