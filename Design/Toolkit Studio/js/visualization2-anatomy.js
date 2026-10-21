/* Assignment data anatomy. Ownership, logical capacity and modulo resource
 * mapping only: no execution clock or buffer occupancy is inferred. */
(function(global){
  'use strict';
  function createModel({seqLens,blocks,numHeads,kvHeads,headDim,rowTile,pageTokens,pagesPerStack,transferSlots,blockTable=null}){
    const qPerKv=numHeads/kvHeads,stackTokens=pageTokens*pagesPerStack;
    const tasks=seqLens.flatMap((seqLen,batch)=>Array.from({length:kvHeads},(_,kvHead)=>{
      const id=batch*kvHeads+kvHead,block=id%blocks,qStart=kvHead*qPerKv;
      const pages=Array.from({length:Math.ceil(seqLen/pageTokens)},(_,index)=>({index,stack:Math.floor(index/pagesPerStack),
        tokens:Math.min(pageTokens,seqLen-index*pageTokens),physicalId:Number.isInteger(blockTable?.[batch]?.[index])&&blockTable[batch][index]>=0?blockTable[batch][index]:null}));
      const stacks=global.PtoVisualization2TaskEncoding.segments(seqLen,stackTokens,pageTokens).map(segment=>({
        index:segment.index,tokens:segment.tokens,slot:segment.index%transferSlots,
        pages:pages.filter(page=>page.stack===segment.index).map(page=>page.index),capacityPages:segment.pages}));
      return {id,block,batch,kvHead,seqLen,pageTokens,pagesPerStack,stackTokens,transferSlots,qHeads:Array.from({length:qPerKv},(_,q)=>qStart+q),
        q:{shape:[rowTile,headDim],validShape:[qPerKv,headDim]},pages,stacks,
        slots:Array.from({length:transferSlots},(_,index)=>({index,stacks:stacks.filter(stack=>stack.slot===index).map(stack=>stack.index)})),
        output:{shape:[qPerKv,headDim],dtype:'BF16',rowStart:batch*numHeads+qStart,rowEnd:batch*numHeads+qStart+qPerKv-1}};
    }));
    return {tasks,blocks,numHeads,kvHeads,headDim,rowTile,pageTokens,pagesPerStack,stackTokens,transferSlots};
  }
  function related(task,focus){
    let stacks=[];
    if(focus?.kind==='stack')stacks=[focus.index];
    if(focus?.kind==='page')stacks=[task.pages[focus.index]?.stack];
    if(focus?.kind==='slot')stacks=task.slots[focus.index]?.stacks||[];
    stacks=stacks.filter(index=>task.stacks[index]);
    const slots=[...new Set(stacks.map(index=>task.stacks[index].slot))];
    if(focus?.kind==='slot'&&!slots.includes(focus.index))slots.push(focus.index);
    return {stacks,slots,pages:stacks.flatMap(index=>task.stacks[index].pages),
      shared:task.stacks.filter(stack=>slots.includes(stack.slot)).map(stack=>stack.index)};
  }
  function mount(host,{model,onTask,onFocus,onCamera,onInspector,onExit}){
    const esc=value=>String(value).replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
    host.classList.add('viz2-anatomy');
    host.innerHTML=`<div class="viz2-anatomy-taskbar"><span>同一 Block · 持续缩小返回 Block 间</span><div role="group" aria-label="切换 Block 内 Task" data-anatomy-tasks></div><span data-anatomy-context></span></div>
      <div class="viz2-anatomy-body"><div class="viz2-anatomy-stage"><div class="viz2-anatomy-viewport" role="group" tabindex="0" aria-label="Task 数据结构画布，滚轮缩放，拖拽平移"><svg class="viz2-anatomy-svg" aria-label="Q Tile、Stack 内逻辑 Pages 与 Transfer Slots 的映射"><g data-anatomy-scene></g></svg><div class="kf-pa-map__tooltip" role="tooltip" hidden></div></div>
      <footer class="viz2-anatomy-footer"><span>有效数据 <i class="viz2-anatomy-key"></i>未用容量 <i class="viz2-anatomy-key is-capacity"></i><small>静态数据映射</small></span><div class="kf-pa-map__tools"><button type="button" class="btn btn-ghost btn-sm" data-anatomy-zoom="0.8" aria-label="缩小数据结构">−</button><output data-anatomy-scale></output><button type="button" class="btn btn-ghost btn-sm" data-anatomy-zoom="1.25" aria-label="放大数据结构">＋</button><button type="button" class="btn btn-ghost btn-sm" data-anatomy-fit>Fit to View</button></div></footer></div>
      <aside class="pto-ide-frame__pane viz2-anatomy-inspector" aria-label="Task 数据详情" hidden><header class="pto-ide-frame__pane-header"><b data-anatomy-title></b><button type="button" class="btn btn-ghost btn-sm" data-anatomy-close aria-label="关闭数据详情">×</button></header><div class="pto-ide-frame__pane-body" data-anatomy-detail></div></aside></div>`;
    const viewport=host.querySelector('.viz2-anatomy-viewport'),scene=host.querySelector('[data-anatomy-scene]'),inspector=host.querySelector('aside'),tooltip=host.querySelector('[role="tooltip"]'),selector=host.querySelector('[data-anatomy-tasks]');
    let task=null,focus=null,camera=null,active=false,drag=null,suppressClick=false,lastSize=null,diagramHeight=560;
    const positions={qX:40,qY:146,qW:174,qH:model.rowTile*9+16,cacheX:300,cacheW:368,rowY:146,rowH:84,pageH:30,gap:8};
    positions.pageW=(positions.cacheW-24-(model.pagesPerStack-1)*positions.gap)/model.pagesPerStack;
    positions.slotX=positions.cacheX+positions.cacheW+112;positions.width=positions.slotX+220;
    const rect=(x,y,w,h,attrs='')=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="3" ${attrs}/>`;
    const text=(x,y,value,cls='')=>`<text x="${x}" y="${y}" class="${cls}">${esc(value)}</text>`;
    const object=(kind,index,label,body)=>`<g data-anatomy-kind="${kind}" data-anatomy-index="${index}" tabindex="0" role="button" aria-label="${esc(label)}" aria-pressed="false">${body}</g>`;
    function draw(){
      const p=positions,bodyBottom=Math.max(p.qY+p.qH+112,p.rowY+task.stacks.length*p.rowH,p.rowY+model.transferSlots*118);
      const outputY=bodyBottom+30;diagramHeight=outputY+100;
      selector.innerHTML=model.tasks.filter(t=>t.block===task.block).map(t=>`<button type="button" class="btn btn-ghost btn-sm ${t.id===task.id?'is-selected':''}" data-anatomy-task-select="${t.id}" aria-pressed="${t.id===task.id}" aria-label="切换到 Task ${t.id}，Batch ${t.batch}，KV Head ${t.kvHead}">Task ${t.id}</button>`).join('');
      host.querySelector('[data-anatomy-context]').textContent=`Batch ${task.batch} · KV Head ${task.kvHead} · ${task.seqLen} tokens`;
      const qRows=Array.from({length:model.rowTile},(_,row)=>rect(p.qX+18,p.qY+row*9,p.qW-36,7,`class="${row<task.qHeads.length?'viz2-anatomy-valid':'viz2-anatomy-empty'}" ${row<task.qHeads.length?`data-anatomy-q-row="${row}"`:''}`)).join('');
      scene.innerHTML=`${rect(18,36,p.width-36,diagramHeight-52,'class="viz2-anatomy-task-frame" data-anatomy-task-frame')}
        ${text(40,70,`Task ${task.id}`,'viz2-anatomy-task-title')}
        ${text(p.qX,112,'Q Tile','viz2-anatomy-heading')}${text(p.cacheX,112,`KV Cache · ${task.stacks.length} Stacks · ${task.pages.length} Pages`,'viz2-anatomy-heading')}${text(p.slotX,112,`Transfer Slots · ${model.transferSlots}`,'viz2-anatomy-heading')}
        ${object('q',0,`Q Tile，Q Heads ${task.qHeads[0]}–${task.qHeads.at(-1)}`,
          `${rect(p.qX,p.qY-12,p.qW,p.qH+120,'class="viz2-anatomy-hit"')}${rect(p.qX,p.qY-12,p.qW,p.qH+40,'class="viz2-anatomy-q-boundary"')}${qRows}${text(p.qX,p.qY+p.qH+52,`Q Heads ${task.qHeads[0]}–${task.qHeads.at(-1)}`)}${text(p.qX,p.qY+p.qH+76,`Tile ${task.q.shape.join(' × ')}`,'viz2-anatomy-muted')}${text(p.qX,p.qY+p.qH+98,`Valid ${task.q.validShape.join(' × ')}`,'viz2-anatomy-muted')}`)}
        ${task.stacks.map(stack=>{
          const y=p.rowY+stack.index*p.rowH,slotY=p.rowY+stack.slot*118;
          const cells=Array.from({length:model.pagesPerStack},(_,position)=>{
            const index=stack.index*model.pagesPerStack+position,page=task.pages[index];
            const x=p.cacheX+12+position*(p.pageW+p.gap),pageY=y+32,key=`${stack.index}:${position}`;
            if(!page)return rect(x,pageY,p.pageW,p.pageH,`class="viz2-anatomy-unused-page" data-anatomy-cell="${key}" aria-hidden="true"`);
            return object('page',index,`逻辑 Page ${index}，${page.tokens} 有效 tokens`,
              `${rect(x,pageY,p.pageW,p.pageH,`class="viz2-anatomy-empty" data-anatomy-cell="${key}" data-anatomy-page-track="${index}"`)}${rect(x,pageY,p.pageW*page.tokens/model.pageTokens,p.pageH,`class="viz2-anatomy-valid" data-anatomy-page-fill="${index}"`)}${text(x+8,pageY+21,`P${index}`)}`);
          });
          // The Stack hit surface sits behind its Page buttons. Empty positions
          // select the Stack and never acquire fictitious logical Page IDs.
          const emptyCells=cells.filter((_,position)=>!task.pages[stack.index*model.pagesPerStack+position]).join('');
          const logicalPages=cells.filter((_,position)=>!!task.pages[stack.index*model.pagesPerStack+position]).join('');
          return `<path data-anatomy-slot-link="${stack.index}" class="viz2-anatomy-link" d="M ${p.cacheX+p.cacheW+4} ${y+38} C ${p.cacheX+p.cacheW+48} ${y+38}, ${p.slotX-48} ${slotY+35}, ${p.slotX-6} ${slotY+35}"/>
            <g data-anatomy-stack-container="${stack.index}">
              ${object('stack',stack.index,`Stack ${stack.index}，${stack.tokens} 有效 tokens，映射 Slot ${stack.slot}`,
                `${rect(p.cacheX,y,p.cacheW,74,'class="viz2-anatomy-stack-boundary"')}${text(p.cacheX+12,y+21,`Stack ${stack.index} · ${stack.tokens}/${model.stackTokens}`)}${emptyCells}`)}
              ${logicalPages}
            </g>`;
        }).join('')}
        ${task.slots.map(slot=>{const y=p.rowY+slot.index*118;return object('slot',slot.index,`Transfer Slot ${slot.index}，映射 Stack ${slot.stacks.join(', ')||'无'}`,`${rect(p.slotX,y,180,70,'class="viz2-anatomy-slot-boundary"')}${text(p.slotX+14,y+26,`Slot ${slot.index}`)}${text(p.slotX+14,y+52,slot.stacks.length?`Stack ${slot.stacks.join(' · ')}`:'无映射 Stack','viz2-anatomy-muted')}`);}).join('')}
        ${object('output',0,'当前 Task 的 Attention Output',`${rect(40,outputY-14,p.width-80,70,'class="viz2-anatomy-hit"')}${text(40,outputY+8,'Attention Output','viz2-anatomy-heading')}${Array.from({length:task.qHeads.length},(_,row)=>rect(p.cacheX+row*58,outputY-5,48,24,'class="viz2-anatomy-output-row"')).join('')}${text(p.slotX,outputY+10,`${task.output.shape.join(' × ')} · BF16`,'viz2-anatomy-muted')}`)}`;
    }
    const describe=(kind,index)=>{
      if(kind==='page'){const page=task.pages[index];return `逻辑 Page ${index} · ${page.tokens} / ${model.pageTokens} tokens · Stack ${page.stack}${page.physicalId===null?' · 物理 Page 未知':` · 物理 Page ${page.physicalId}`}`;}
      if(kind==='stack'){const stack=task.stacks[index];return `Stack ${index} · ${stack.tokens} / ${model.stackTokens} tokens · Pages ${stack.pages.join(', ')} · Slot ${stack.slot}`;}
      if(kind==='slot')return `Slot ${index} · Stack ${task.slots[index].stacks.join(', ')||'无'} · 静态循环映射`;
      if(kind==='q')return `Q Heads ${task.qHeads[0]}–${task.qHeads.at(-1)} · Tile [${task.q.shape}] · Valid [${task.q.validShape}]`;
      return `Attention Output · Q Heads ${task.qHeads[0]}–${task.qHeads.at(-1)} · [${task.output.shape}] · BF16`;
    };
    function paintFocus(){
      const r=related(task,focus);
      scene.querySelectorAll('[data-anatomy-kind]').forEach(element=>{
        const {anatomyKind:kind,anatomyIndex:raw}=element.dataset,index=Number(raw);
        const selected=focus?.kind===kind&&focus.index===index;
        const linked=kind==='stack'?r.stacks.includes(index):kind==='page'?r.pages.includes(index):kind==='slot'?r.slots.includes(index):false;
        element.classList.toggle('is-selected',selected);element.classList.toggle('is-related',linked);
        element.classList.toggle('is-shared',kind==='stack'&&r.shared.includes(index)&&!linked);
        element.setAttribute('aria-pressed',String(selected));
      });
      scene.querySelectorAll('[data-anatomy-slot-link]').forEach(path=>path.classList.toggle('is-related',r.stacks.includes(Number(path.dataset.anatomySlotLink))));
    }
    function paintInspector(open){
      inspector.hidden=!open||!focus;if(inspector.hidden)return;
      const kind=focus.kind,index=focus.index;
      host.querySelector('[data-anatomy-title]').textContent=kind==='q'?'Q Tile':kind==='output'?'Attention Output':`${kind==='page'?'逻辑 Page':kind==='stack'?'Stack':'Transfer Slot'} ${index}`;
      let rows=[];
      if(kind==='page'){
        const page=task.pages[index];rows=[['逻辑 Page Index',index],['有效 Token',`${page.tokens} / ${model.pageTokens}`],['Stack',page.stack],['Transfer Slot',task.stacks[page.stack].slot]];
        rows.push(page.physicalId===null?['物理页映射','未知 · 未提供实际 block_table 数据']:['物理 Page ID',page.physicalId]);
      }else if(kind==='stack'){
        const stack=task.stacks[index];rows=[['Stack ID',index],['有效 Token',`${stack.tokens} / ${model.stackTokens}`],['逻辑 Pages',stack.pages.join(', ')],['未用容量',`${model.stackTokens-stack.tokens} tokens`],['Transfer Slot',stack.slot],['同 Slot 的 Stacks',task.slots[stack.slot].stacks.join(', ')]];
      }else if(kind==='slot')rows=[['Slot ID',index],['映射规则',`stackId % ${model.transferSlots}`],['共享该 Slot 的 Stacks',task.slots[index].stacks.join(', ')||'无'],['含义','Score / Probability / PV 各自 Tensor 的同号切片；此处仅表示循环映射，不表示同时占用。']];
      else if(kind==='q')rows=[['Query Heads',task.qHeads.join(', ')],['Tile Shape',`[${task.q.shape.join(', ')}]`],['Valid Shape',`[${task.q.validShape.join(', ')}]`],['未用行',model.rowTile-task.qHeads.length],['来源','q2d 中当前 Batch / KV Head 对应的 Query 分组']];
      else rows=[['Query Heads',task.qHeads.join(', ')],['输出区域',`[${task.output.shape.join(', ')}]`],['类型','BF16'],['逻辑 Tensor 行',`${task.output.rowStart}–${task.output.rowEnd}`],['含义','当前 Task 完成 Normalize 后写入的 Attention Output 区域；不推断物理地址。']];
      host.querySelector('[data-anatomy-detail]').innerHTML=`<p>Task ${task.id} · Block ${String(task.block).padStart(2,'0')}<br>Batch ${task.batch} · KV Head ${task.kvHead}</p><dl>${rows.map(([key,value])=>`<dt>${esc(key)}</dt><dd>${esc(value)}</dd>`).join('')}</dl>`;
    }
    const fitted=()=>{const scale=Math.max(.05,Math.min((viewport.clientWidth-32)/positions.width,(viewport.clientHeight-24)/diagramHeight,1.3));return {scale,x:(viewport.clientWidth-positions.width*scale)/2,y:(viewport.clientHeight-diagramHeight*scale)/2};};
    function paint(){if(!active||!task||!viewport.clientWidth||!viewport.clientHeight)return;if(!camera){onCamera(task.id,fitted());return;}
      scene.setAttribute('transform',`translate(${camera.x} ${camera.y}) scale(${camera.scale})`);
      scene.style.setProperty('--anatomy-font',`${Math.max(12,12/camera.scale)}px`);host.querySelector('[data-anatomy-scale]').textContent=`${Math.round(camera.scale*100)}%`;
    }
    function update(id,nextFocus,nextCamera,{active:visible,inspectorOpen=false}={}){
      const entering=visible&&!active;
      active=visible;if(!active){tooltip.hidden=true;return;}const next=model.tasks[id];if(!next)return;
      const changed=task?.id!==id;task=next;focus=nextFocus;camera=nextCamera;
      // Each Task retains its own camera across Block-local tab switches.
      if(entering)lastSize=null;
      if(changed){tooltip.hidden=true;draw();}paintFocus();paintInspector(inspectorOpen);paint();
    }
    function zoom(factor,point={x:viewport.clientWidth/2,y:viewport.clientHeight/2}){
      if(!camera)return;const previousScale=camera.scale,scale=Math.max(.15,Math.min(4,previousScale*factor)),ratio=scale/previousScale;
      const exitScale=Math.max(.15,fitted().scale*.55);
      // Exit from the currently painted Task, just like the Block 间 button.
      if(factor<1&&previousScale>exitScale&&scale<=exitScale){onExit();return;}
      onCamera(task.id,{scale,x:point.x-(point.x-camera.x)*ratio,y:point.y-(point.y-camera.y)*ratio});
    }
    host.addEventListener('click',event=>{
      tooltip.hidden=true;
      const taskButton=event.target.closest('[data-anatomy-task-select]');if(taskButton){onTask(Number(taskButton.dataset.anatomyTaskSelect));return;}
      if(event.target.closest('[data-anatomy-close]')){onInspector(false);return;}
      if(event.target.closest('[data-anatomy-fit]')){onCamera(task.id,fitted());return;}
      const button=event.target.closest('[data-anatomy-zoom]');if(button){zoom(Number(button.dataset.anatomyZoom));return;}
      if(suppressClick){suppressClick=false;return;}
      const target=event.target.closest('[data-anatomy-kind]');if(target)onFocus({kind:target.dataset.anatomyKind,index:Number(target.dataset.anatomyIndex)});
    });
    viewport.addEventListener('wheel',event=>{event.preventDefault();tooltip.hidden=true;const r=viewport.getBoundingClientRect(),unit=event.deltaMode===1?16:event.deltaMode===2?r.height:1;zoom(Math.exp(Math.max(-.22,Math.min(.22,-event.deltaY*unit*.003))),{x:event.clientX-r.left,y:event.clientY-r.top});},{passive:false});
    viewport.addEventListener('pointerdown',event=>{if(event.button!==0)return;suppressClick=false;drag={id:event.pointerId,x:event.clientX,y:event.clientY,startX:event.clientX,startY:event.clientY,moved:false};});
    viewport.addEventListener('pointermove',event=>{
      if(drag){if(!drag.moved&&Math.hypot(event.clientX-drag.startX,event.clientY-drag.startY)<=4)return;drag.moved=true;viewport.setPointerCapture(drag.id);tooltip.hidden=true;onCamera(task.id,{...camera,x:camera.x+event.clientX-drag.x,y:camera.y+event.clientY-drag.y});drag.x=event.clientX;drag.y=event.clientY;return;}
      const target=event.target.closest('[data-anatomy-kind]');tooltip.hidden=!target;if(target){tooltip.textContent=describe(target.dataset.anatomyKind,Number(target.dataset.anatomyIndex));const r=viewport.getBoundingClientRect();tooltip.style.left=`${Math.max(8,Math.min(event.clientX-r.left+14,r.width-tooltip.offsetWidth-8))}px`;tooltip.style.top=`${Math.max(8,Math.min(event.clientY-r.top+16,r.height-tooltip.offsetHeight-8))}px`;}
    });
    const end=()=>{suppressClick=!!drag?.moved;drag=null;};['pointerup','pointercancel','lostpointercapture'].forEach(type=>viewport.addEventListener(type,end));viewport.addEventListener('pointerleave',()=>{tooltip.hidden=true;});
    viewport.addEventListener('keydown',event=>{
      const target=event.target.closest('[data-anatomy-kind]');if(target&&['Enter',' '].includes(event.key)){event.preventDefault();onFocus({kind:target.dataset.anatomyKind,index:Number(target.dataset.anatomyIndex)});}
      if(event.key==='Escape')onInspector(false);
      if(['+','=','-','0'].includes(event.key)){event.preventDefault();if(event.key==='0')onCamera(task.id,fitted());else zoom(event.key==='-'?.8:1.25);}
    });
    new ResizeObserver(()=>{if(!active||!task||!viewport.clientWidth||!viewport.clientHeight)return;const size={w:viewport.clientWidth,h:viewport.clientHeight};if(camera&&lastSize&&(size.w!==lastSize.w||size.h!==lastSize.h)){
      const old=lastSize;lastSize=size;const ratio=Math.min(size.w/old.w,size.h/old.h);onCamera(task.id,{scale:camera.scale*ratio,x:size.w/2+(camera.x-old.w/2)*ratio,y:size.h/2+(camera.y-old.h/2)*ratio});
    }else{lastSize=size;paint();}}).observe(viewport);
    const screenRect=element=>{const r=element.getBoundingClientRect();return {x:r.left,y:r.top,width:r.width,height:r.height};};
    function getTransitionSnapshot(){if(!task||!active)return null;return {task:task.id,model:task,frame:screenRect(scene.querySelector('[data-anatomy-task-frame]')),
      title:screenRect(scene.querySelector('.viz2-anatomy-task-title')),qRows:[...scene.querySelectorAll('[data-anatomy-q-row]')].map(screenRect),
      cells:[...scene.querySelectorAll('[data-anatomy-cell]')].map(element=>({key:element.dataset.anatomyCell,rect:screenRect(element)}))};}
    return {update,model,getTransitionSnapshot,hideTooltip(){tooltip.hidden=true;}};
  }
  global.PtoVisualization2Anatomy=Object.freeze({createModel,related,mount});
})(window);
