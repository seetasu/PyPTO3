/* Identity-based transition coordinator. View renderers own camera/layout; this layer
 * only borrows screen-space snapshots, and never writes selection or camera state. */
(function(global){
  'use strict';
  const NS='http://www.w3.org/2000/svg';
  const rect=r=>({x:r.left,y:r.top,width:r.width,height:r.height});
  function paintedClone(svg) {
    const clone=svg.cloneNode(true), originals=[svg,...svg.querySelectorAll('*')], copies=[clone,...clone.querySelectorAll('*')];
    originals.forEach((node,index)=>{
      const style=getComputedStyle(node);
      for (const name of ['fill','stroke','stroke-width','opacity','stroke-linejoin','stroke-linecap']) copies[index].style.setProperty(name,style.getPropertyValue(name));
    });
    return clone;
  }
  const mix=(a,b,t)=>a+(b-a)*t;
  const lerp=(a,b,t)=>Object.fromEntries(['x','y','width','height'].map(k=>[k,mix(a[k],b[k],t)]));
  const ease=t=>t<.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2;
  function pairs(assignment,pipeline){
    const blocks=new Map(assignment.blocks.map(b=>[b.id,b]));
    const tasks=new Map(assignment.tasks.map(t=>[t.id,t]));
    return {blocks:pipeline.blocks.filter(b=>blocks.has(b.id)).map(b=>({source:blocks.get(b.id),target:b})),
      tasks:pipeline.tasks.filter(t=>tasks.has(t.id)).map(t=>({source:tasks.get(t.id),target:t}))};
  }
  function focusBlock(snapshot,selected){
    const tasksByBlock=new Map();
    snapshot.tasks.forEach(task=>{
      const block=task.task?.block??task.block;
      tasksByBlock.set(block,(tasksByBlock.get(block)||0)+1);
    });
    const visible=block=>block.rect.y>=snapshot.clip.y&&
      block.rect.y+block.rect.height<=snapshot.clip.y+snapshot.clip.height;
    const choice=snapshot.blocks.find(block=>block.id===selected&&tasksByBlock.has(block.id)&&visible(block));
    return choice?.id??snapshot.blocks.find(block=>tasksByBlock.has(block.id)&&visible(block))?.id??null;
  }
  function create({root,getPipeline,getGraph,getIntraTask}){
    let job=null,serial=0,lastFocus=null;
    const intra=global.PtoVisualization2IntraMorph.create({root,getTask:getIntraTask});
    const media=global.matchMedia('(prefers-reduced-motion: reduce)');
    const originRect=()=>{
      const bounds=root.getBoundingClientRect();
      return {left:bounds.left+root.clientLeft,top:bounds.top+root.clientTop,width:root.clientWidth,height:root.clientHeight};
    };
    const el=(tag,attrs={},parent)=>{const n=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));parent?.append(n);return n;};
    function assignmentSnapshot(){
      const host=root.querySelector('#viz2-attention'), viewport=host.querySelector('.kf-pa-map__viewport');
      const blockCount=host.querySelectorAll('[data-pa-block]').length;
      return {clip:rect(viewport.getBoundingClientRect()),
        blocks:[...host.querySelectorAll('[data-pa-block]')].map(button=>{
          const frame=button.querySelector('.viz2-am-block-frame'),label=button.querySelector('.viz2-am-block-label');
          const icon=document.createElementNS(NS,'svg');icon.setAttribute('viewBox','0 0 1 1');
          return {id:Number(button.dataset.paBlock),rect:rect(frame.getBoundingClientRect()),icon,
            iconRect:rect(frame.getBoundingClientRect()),labelRect:rect(label.getBoundingClientRect())};
        }),
        tasks:[...host.querySelectorAll('[data-pa-task]')].map(task=>{
          const frame=task.querySelector('.kf-pa-map__cell'),r=rect(frame.getBoundingClientRect());
          return {id:Number(task.dataset.paTask),block:Number(task.dataset.paTask)%blockCount,rect:r,clone:task.cloneNode(true),
            segments:Array.from({length:7},(_,i)=>({x:r.x+i*r.width/7,y:r.y,width:r.width/7,height:r.height}))};
        })};
    }
    function mask(view,value){
      if(view==='pipeline')getPipeline()?.setTransitionMasked(value);
      else if(view==='graph')root.classList.toggle('viz2-morph-graph',value);
      else root.classList.toggle('viz2-morph-assignment',value);
    }
    function cancel(){
      intra.cancel();
      if(!job)return;
      cancelAnimationFrame(job.frame);job.animations?.forEach(animation=>animation.cancel());job.overlay?.remove();job.hold?.remove();
      root.querySelector(`#viz2-${job.to}`)?.style.removeProperty('opacity');
      mask(job.to,false);root.classList.remove('viz2-morph-preparing');delete root.dataset.morphDestination;job=null;
    }
    function prepare(state,to){
      cancel();
      if(state.views[state.activeTab]?.subview==='block')return intra.prepare(state,to);
      if(!state.open||state.activeTab===to||media.matches)return null;
      const from=state.activeTab;
      const assignmentPipeline=(from==='assignment'&&to==='pipeline')||(from==='pipeline'&&to==='assignment');
      const pipelineGraph=from==='pipeline'&&to==='graph';
      if(!assignmentPipeline&&!pipelineGraph)return null;
      if(assignmentPipeline&&state.views.assignment.phase!=='attention')return null;
      const focus=assignmentPipeline&&from==='pipeline'?(state.selection?.block??lastFocus??0):null;
      if(focus!==null)getPipeline()?.ensureTransitionBlockVisible(focus);
      const source=from==='assignment'?assignmentSnapshot():getPipeline()?.getTransitionSnapshot();
      if(!source)return null;
      job={id:++serial,from,to,source,selection:state.selection,focus,frame:0,phase:'preparing'};
      // Keep a frozen source visible while the destination measures itself. Canvas
      // pixels must be copied explicitly: cloneNode alone leaves them empty.
      const pane=root.querySelector(`#viz2-${from}`), bounds=pane.getBoundingClientRect(), origin=originRect();
      const hold=pane.cloneNode(true);hold.removeAttribute('id');hold.setAttribute('aria-hidden','true');hold.inert=true;
      Object.assign(hold.style,{position:'absolute',left:`${bounds.left-origin.left}px`,top:`${bounds.top-origin.top}px`,width:`${bounds.width}px`,height:`${bounds.height}px`,zIndex:11,pointerEvents:'none',margin:0});
      const originals=pane.querySelectorAll('canvas');hold.querySelectorAll('canvas').forEach((canvas,index)=>canvas.getContext('2d').drawImage(originals[index],0,0));
      job.hold=hold;root.append(hold);
      const descendants=hold.querySelectorAll('*');pane.querySelectorAll('*').forEach((node,index)=>{
        if(node.scrollTop||node.scrollLeft){descendants[index].scrollTop=node.scrollTop;descendants[index].scrollLeft=node.scrollLeft;}
      });
      // Hide the destination before its very first paint. Visibility retains layout.
      root.classList.add('viz2-morph-preparing');root.dataset.morphDestination=to;
      return job;
    }
    function play(current){
      if(current?.phase&&current.source?.objects){intra.play(current);return;}
      if(!current||job!==current)return;
      // Two frames settle first-entry camera and Inspector-dependent width, including ResizeObserver.
      current.frame=requestAnimationFrame(()=>{current.frame=requestAnimationFrame(()=>start(current));});
    }
    function startPipelineGraph(current){
      const target=getGraph()?.getBlockTransitionSnapshot();
      if(!target?.blocks.length){cancel();return;}
      const origin=originRect(),style=getComputedStyle(root),color=name=>style.getPropertyValue(name).trim();
      const sources=new Map(current.source.blocks.map(block=>[block.id,block]));
      const visible=block=>block.rect.y+block.rect.height>current.source.clip.y&&block.rect.y<current.source.clip.y+current.source.clip.height;
      const mapping=target.blocks.filter(block=>sources.has(block.id)&&visible(sources.get(block.id))).map(block=>({source:sources.get(block.id),target:block}));
      const overlay=el('svg',{class:'viz2-transition-overlay',width:origin.width,height:origin.height,'aria-hidden':'true','data-direction':'pipeline-graph'},root);
      current.overlay=overlay;
      const items=mapping.map(pair=>{
        const selected=pair.source.id===current.selection?.block;
        const group=el('g',{'data-morph-block':pair.source.id,'data-morph-stage':pair.target.stage||''},overlay);
        const box=el('rect',{rx:4,fill:selected?color('--state-selected'):color('--surface-3'),stroke:selected?color('--primary'):color('--border-subtle')},group);
        const label=el('text',{'text-anchor':'middle','dominant-baseline':'central',fill:selected?color('--foreground'):color('--foreground-secondary'),'font-family':color('--font-mono'),'font-size':12},group);
        label.textContent=String(pair.source.id).padStart(2,'0');return {...pair,box,label};
      });
      mask('graph',true);root.classList.remove('viz2-morph-preparing');
      const destination=root.querySelector('#viz2-graph');
      current.animations=[current.hold.animate([{opacity:1},{opacity:0}],{duration:420,fill:'both',easing:'ease-in-out'}),
        destination.animate([{opacity:0},{opacity:1}],{duration:700,fill:'both',easing:'ease-in-out'})];
      current.phase='animating';const started=performance.now(),duration=760;
      function frame(now){
        if(job!==current)return;
        const progress=Math.min(1,(now-started)/duration),p=ease(progress);
        items.forEach(item=>{
          const r=lerp(item.source.rect,item.target.rect,p);
          for(const key of ['x','y','width','height'])item.box.setAttribute(key,key==='x'?r.x-origin.left:key==='y'?r.y-origin.top:r[key]);
          item.label.setAttribute('x',r.x-origin.left+r.width/2);
          item.label.setAttribute('y',r.y-origin.top+r.height/2);
          item.box.style.opacity=item.label.style.opacity=String(1-Math.max(0,(progress-.82)/.18));
        });
        if(progress<1)current.frame=requestAnimationFrame(frame);else cancel();
      }
      frame(started);
    }
    function start(current){
      if(job!==current)return;
      if(current.from==='pipeline'&&current.to==='graph'){startPipelineGraph(current);return;}
      const forward=current.to==='pipeline';
      let focus=forward?focusBlock(current.source,current.selection?.block):current.focus;
      if(forward && focus!==null) getPipeline()?.ensureTransitionBlockVisible(focus);
      const target=forward?getPipeline()?.getTransitionSnapshot():assignmentSnapshot();
      const destination=root.querySelector(`#viz2-${current.to}`);
      if(!target || !destination){cancel();return;}
      const assignment=forward?current.source:target, pipeline=forward?target:current.source;
      if(!forward && focus===null) focus=focusBlock(assignment,current.selection?.block);
      const sourceTasks=new Map(assignment.tasks.map(task=>[task.id,task]));
      const targetTasks=new Map(pipeline.tasks.map(task=>[task.id,task]));
      const focusPipelineTasks=focus===null?[]:pipeline.tasks.filter(task=>task.task.block===focus);
      const focusAssignmentTasks=focus===null?[]:assignment.tasks.filter(task=>task.block===focus);
      const taskIds=focusPipelineTasks.filter(task=>sourceTasks.has(task.id)).map(task=>task.id);
      const valid=taskIds.length>0 && taskIds.length===focusPipelineTasks.length && taskIds.length===focusAssignmentTasks.length &&
        taskIds.every(id=>targetTasks.has(id) && sourceTasks.get(id).rect.width>0 && targetTasks.get(id).rect.width>0) &&
        assignment.blocks.some(block=>block.id===focus) && pipeline.blocks.some(block=>block.id===focus) &&
        (forward?assignment: pipeline).clip.width>0 && target.clip.width>0;
      if(valid)lastFocus=focus;
      const origin=originRect(), style=getComputedStyle(root), color=name=>style.getPropertyValue(name).trim();
      const overlay=el('svg',{class:'viz2-transition-overlay',width:origin.width,height:origin.height,'aria-hidden':'true',
        'data-direction':forward?'forward':'reverse','data-focus-block':focus??''},root);
      current.overlay=overlay;
      const focusSource=valid?(forward?assignment:pipeline).blocks.find(block=>block.id===focus):null;
      const focusFrame=focusSource?el('rect',{'data-morph-focus-block':focus,rx:3,fill:'none',
        stroke:color('--primary'),'stroke-width':1.5},overlay):null;
      if(focusFrame){const r=focusSource.rect;focusFrame.setAttribute('x',r.x-origin.left);focusFrame.setAttribute('y',r.y-origin.top);
        focusFrame.setAttribute('width',r.width);focusFrame.setAttribute('height',r.height);}
      const shapes=valid?taskIds.map(id=>{
        const source=sourceTasks.get(id), target=targetTasks.get(id);
        const group=el('g',{'data-morph-task':id,'data-morph-block':focus},overlay);
        const bar=el('rect',{rx:2,fill:color('--surface-4')},group);
        const fill=el('rect',{rx:1,fill:color('--primary')},group);
        const label=el('text',{'text-anchor':'middle','dominant-baseline':'central',fill:color('--foreground'),
          'font-family':color('--font-mono'),'font-size':11},group);label.textContent=String(id);
        return {source,target,group,bar,fill,label};
      }):[];
      destination.style.opacity='0';
      current.hold.style.opacity='1';
      root.classList.remove('viz2-morph-preparing');
      current.phase='animating';
      const started=performance.now(),duration=valid?850:350;
      const fromClip=forward?assignment.clip:pipeline.clip,toClip=forward?pipeline.clip:assignment.clip;
      const local=r=>({x:r.x-origin.left,y:r.y-origin.top,width:r.width,height:r.height});
      function frame(now){
        if(job!==current)return;
        const t=Math.max(0,Math.min(1,(now-started)/duration));
        const fadeOut=valid?Math.min(1,t/.2):t;
        const reveal=valid?Math.max(0,Math.min(1,(t-.7)/.3)):t;
        current.hold.style.opacity=String(1-fadeOut);
        destination.style.opacity=String(reveal);
        if(valid){
          const morph=ease(Math.max(0,Math.min(1,(t-.2)/.5)));
          const alpha=t<.2?t/.2:t>.7?1-reveal:1;
          focusFrame.style.opacity=String(Math.max(0,1-t/.35));
          shapes.forEach(({source,target,group,bar,fill,label})=>{
            const from=forward?source.rect:target.rect, to=forward?target.rect:source.rect;
            const r=local(lerp(from,to,morph));
            for(const key of ['x','y','width','height'])bar.setAttribute(key,r[key]);
            fill.setAttribute('x',r.x);fill.setAttribute('y',r.y+r.height-3);
            fill.setAttribute('width',r.width);fill.setAttribute('height',3);
            label.setAttribute('x',r.x+r.width/2);label.setAttribute('y',r.y+r.height/2);
            group.style.opacity=String(alpha);
          });
          const clip=lerp(fromClip,toClip,morph);
          overlay.style.clipPath=`inset(${Math.max(0,clip.y-origin.top)}px ${Math.max(0,origin.left+origin.width-clip.x-clip.width)}px ${Math.max(0,origin.top+origin.height-clip.y-clip.height)}px ${Math.max(0,clip.x-origin.left)}px)`;
        }
        if(t<1)current.frame=requestAnimationFrame(frame);else cancel();
      }
      frame(started);
    }
    global.addEventListener('resize',cancel);media.addEventListener('change',cancel);
    root.addEventListener('scroll',event=>{if(job?.phase==='animating'&&!job.hold?.contains(event.target))cancel();},true);
    return {prepare,play,cancel,get active(){return !!job||intra.active;},get animating(){return job?.phase==='animating'||intra.animating;}};
  }
  global.PtoVisualization2Transition=Object.freeze({create,pairs,focusBlock});
})(window);
