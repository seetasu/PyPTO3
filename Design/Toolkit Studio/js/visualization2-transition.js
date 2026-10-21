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
    const batches=new Map(assignment.batches.map(b=>[b.batch,b]));
    return {blocks:pipeline.blocks.filter(b=>blocks.has(b.id)).map(b=>({source:blocks.get(b.id),target:b})),
      tasks:pipeline.tasks.filter(t=>batches.has(t.batch)).map(t=>({source:batches.get(t.batch),target:t}))};
  }
  function create({root,getPipeline,getGraph}){
    let job=null,serial=0;
    const media=global.matchMedia('(prefers-reduced-motion: reduce)');
    const originRect=()=>{
      const bounds=root.getBoundingClientRect();
      return {left:bounds.left+root.clientLeft,top:bounds.top+root.clientTop,width:root.clientWidth,height:root.clientHeight};
    };
    const el=(tag,attrs={},parent)=>{const n=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));parent?.append(n);return n;};
    function assignmentSnapshot(){
      const host=root.querySelector('#viz2-attention'), viewport=host.querySelector('.kf-pa-map__viewport');
      return {clip:rect(viewport.getBoundingClientRect()),
        blocks:[...host.querySelectorAll('[data-pa-block]')].map(button=>({id:Number(button.dataset.paBlock),rect:rect(button.getBoundingClientRect()),
          icon:paintedClone(button.querySelector('svg')),iconRect:rect(button.querySelector('svg').getBoundingClientRect()),
          labelRect:rect(button.querySelector('span').getBoundingClientRect())})),
        batches:[...host.querySelectorAll('[data-pa-column]')].map(column=>{
          const sequence=column.querySelector('.kf-pa-map__sequence');
          return {batch:Number(column.dataset.paColumn),rect:rect(sequence.getBoundingClientRect()),clone:sequence.cloneNode(true),segments:[...sequence.querySelectorAll('.kf-pa-map__seq-track')].map(track=>rect(track.getBoundingClientRect()))};
        })};
    }
    function mask(view,value){
      if(view==='pipeline')getPipeline()?.setTransitionMasked(value);
      else if(view==='graph')root.classList.toggle('viz2-morph-graph',value);
      else root.classList.toggle('viz2-morph-assignment',value);
    }
    function cancel(){
      if(!job)return;
      cancelAnimationFrame(job.frame);job.animations?.forEach(animation=>animation.cancel());job.overlay?.remove();job.hold?.remove();
      mask(job.to,false);root.classList.remove('viz2-morph-preparing');delete root.dataset.morphDestination;job=null;
    }
    function prepare(state,to){
      cancel();
      if(!state.open||state.activeTab===to||media.matches||state.views.assignment.subview==='block'||state.views.pipeline.subview==='block'||state.views.graph.subview==='block')return null;
      const from=state.activeTab;
      const assignmentPipeline=(from==='assignment'&&to==='pipeline')||(from==='pipeline'&&to==='assignment');
      const pipelineGraph=from==='pipeline'&&to==='graph';
      if(!assignmentPipeline&&!pipelineGraph)return null;
      if(assignmentPipeline&&state.views.assignment.phase!=='attention')return null;
      const source=from==='assignment'?assignmentSnapshot():getPipeline()?.getTransitionSnapshot();
      if(!source)return null;
      job={id:++serial,from,to,source,selection:state.selection,frame:0,phase:'preparing'};
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
      const target=current.to==='assignment'?assignmentSnapshot():getPipeline()?.getTransitionSnapshot();
      if(!target){cancel();return;}
      const forward=current.to==='pipeline', assignment=forward?current.source:target, pipeline=forward?target:current.source;
      const mapping=pairs(assignment,pipeline), origin=originRect();
      const style=getComputedStyle(root), color=n=>style.getPropertyValue(n).trim();
      const overlay=el('svg',{class:'viz2-transition-overlay',width:origin.width,height:origin.height,'aria-hidden':'true','data-direction':forward?'forward':'reverse'});
      current.overlay=overlay;root.append(overlay);
      const defs=el('defs',{},overlay), clip=el('clipPath',{id:`viz2-morph-clip-${current.id}`},defs), clipRect=el('rect',{},clip);
      const taskLayer=el('g',{'clip-path':`url(#viz2-morph-clip-${current.id})`},overlay);
      const blocks=mapping.blocks.map(pair=>{
        const selected=pair.source.id===current.selection?.block;
        const group=el('g',{'data-morph-block':pair.source.id},overlay);
        const background=el('rect',{rx:3,fill:selected?color('--state-selected'):'transparent',stroke:selected?color('--primary'):'none'},group);
        const icon=pair.source.icon;group.append(icon);icon.removeAttribute('class');icon.style.color=selected?color('--primary'):color('--foreground-secondary');
        const label=el('text',{'text-anchor':'middle','dominant-baseline':'central',fill:selected?color('--foreground'):color('--foreground-secondary'),'font-family':color('--font-mono'),'font-size':12},group);label.textContent=String(pair.source.id).padStart(2,'0');
        return {...pair,group,background,icon,label};
      });
      const tasks=mapping.tasks.map(pair=>{
        // Each Batch strip is actually cloned eight times, then bound by Task.batch.
        const group=pair.source.clone.cloneNode(true);group.removeAttribute('class');group.setAttribute('data-morph-task',pair.target.id);group.setAttribute('data-morph-batch',pair.target.batch);taskLayer.append(group);
        group.replaceChildren();
        const segments=Array.from({length:Math.max(pair.source.segments.length,pair.target.task.stacks)},()=>({track:el('rect',{fill:color('--surface-4'),rx:1},group),fill:el('rect',{fill:color('--primary'),rx:1},group),pages:Array.from({length:pair.target.task.segments[0].pages.length-1},()=>el('line',{stroke:color('--surface-2'),'stroke-width':.7},group))}));
        const line=el('rect',{fill:'none',stroke:pair.target.id===current.selection?.task?color('--foreground'):color('--primary'),'stroke-width':pair.target.id===current.selection?.task?2:1},group);
        const backing=el('rect',{fill:color('--surface-2')},group);
        const label=el('text',{'text-anchor':'middle','dominant-baseline':'central',fill:color('--foreground'),'font-family':color('--font-mono')},group);label.textContent=pair.target.id;
        return {...pair,group,segments,line,backing,label};
      });
      current.hold?.remove();mask(current.to,true);root.classList.remove('viz2-morph-preparing');current.phase='animating';
      const attrRect=(node,r)=>{for(const k of ['x','y','width','height'])node.setAttribute(k,r[k]);};
      const local=r=>({...r,x:r.x-origin.left,y:r.y-origin.top});
      const started=performance.now(), duration=700, stagger=22;
      function frame(now){
        if(job!==current)return;
        const elapsed=now-started, progress=Math.min(1,elapsed/(duration+7*stagger));
        // Clip interpolates between real viewport bounds, including horizontal pan and vertical scroll.
        attrRect(clipRect,local(lerp(forward?assignment.clip:pipeline.clip,forward?pipeline.clip:assignment.clip,ease(progress))));
        blocks.forEach((item,index)=>{
          const t=ease(Math.max(0,Math.min(1,(elapsed-index*4)/duration))), p=forward?t:1-t;
          const r=local(lerp(item.source.rect,item.target.rect,p));attrRect(item.background,r);
          const ir=item.source.iconRect,sr=item.source.rect, factor=1-p;
          item.icon.setAttribute('x',r.x+(ir.x-sr.x)*factor);item.icon.setAttribute('y',r.y+(ir.y-sr.y)*factor);
          item.icon.setAttribute('width',Math.max(.01,ir.width*factor));item.icon.setAttribute('height',Math.max(.01,ir.height*factor));item.icon.style.opacity=String(1-p);
          const labelY=item.source.labelRect.y+item.source.labelRect.height/2-item.source.rect.y;
          item.label.setAttribute('x',r.x+r.width/2);item.label.setAttribute('y',r.y+mix(labelY,r.height/2,p));
        });
        tasks.forEach(item=>{
          const delay=item.target.task.kvHead*stagger, t=ease(Math.max(0,Math.min(1,(elapsed-delay)/duration))),p=forward?t:1-t;
          const r=local(lerp(item.source.rect,item.target.rect,p)), n=item.target.task.stacks;
          const destUnit=r.width/n;
          item.segments.forEach((segment,index)=>{
            const sourceSegment=item.source.segments[index];
            const startX=sourceSegment?r.width*(sourceSegment.x-item.source.rect.x)/item.source.rect.width:r.width;
            const sourceWidth=sourceSegment?r.width*sourceSegment.width/item.source.rect.width:0;
            const endX=Math.min(index,n)*destUnit;
            const w=Math.max(0,mix(sourceWidth,index<n?destUnit-Math.min(2,destUnit*.1):0,p));
            const sr={x:r.x+mix(startX,endX,p),y:r.y,width:w,height:r.height};attrRect(segment.track,sr);
            const fraction=item.target.task.segments[index]?.fraction||0;attrRect(segment.fill,{...sr,width:w*fraction});
            segment.pages.forEach((line,page)=>{
              const x=sr.x+w*(page+1)/(segment.pages.length+1);
              line.setAttribute('x1',x);line.setAttribute('x2',x);line.setAttribute('y1',sr.y);line.setAttribute('y2',sr.y+sr.height);
              line.style.opacity=String(index<n&&destUnit>=28?p:0);
            });
          });
          const selected=item.target.id===current.selection?.task, related=item.target.task.block===current.selection?.block;
          attrRect(item.line,{x:r.x+1,y:r.y+1,width:Math.max(0,r.width-2),height:Math.max(0,r.height-2)});item.line.style.opacity=String((selected||related)?p:0);
          item.label.setAttribute('x',r.x+r.width/2);item.label.setAttribute('y',r.y+r.height/2);item.label.setAttribute('font-size',Math.min(11,Math.max(7,(r.width-3)/String(item.target.id).length/.62)));item.label.style.opacity=String(p);
          const bw=Math.min(r.width,String(item.target.id).length*7+4),bh=Math.min(13,r.height);attrRect(item.backing,{x:r.x+(r.width-bw)/2,y:r.y+(r.height-bh)/2,width:bw,height:bh});item.backing.style.opacity=String(p);
        });
        if(progress<1)current.frame=requestAnimationFrame(frame);else cancel();
      }
      frame(started);
    }
    global.addEventListener('resize',cancel);media.addEventListener('change',cancel);
    root.addEventListener('scroll',event=>{if(job?.phase==='animating'&&!job.hold?.contains(event.target))cancel();},true);
    return {prepare,play,cancel,get active(){return !!job;},get animating(){return job?.phase==='animating';}};
  }
  global.PtoVisualization2Transition=Object.freeze({create,pairs});
})(window);
