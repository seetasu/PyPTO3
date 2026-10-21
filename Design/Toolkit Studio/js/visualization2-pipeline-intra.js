/* Static loop semantics from paged_attention_pypto.py. Tick columns are logical
 * iterations, never measured durations. Slot ribbons are modulo reuse, not edges. */
(function(global){
  'use strict';
  const stages={qk:'QK',softmax:'Softmax',pv:'PV',accumulate:'Context Accumulate'};
  const buffers={qk:{write:'score_transfer'},softmax:{read:'score_transfer',write:'probability_transfer'},pv:{read:'probability_transfer',write:'pv_transfer'},accumulate:{read:'pv_transfer'}};
  const events={QK_READY_EVENT:['qk','softmax'],SOFTMAX_READY_EVENT:['softmax','pv'],PV_READY_EVENT:['pv','accumulate']};
  const transfers=[{buffer:'score_transfer',producer:'qk',consumer:'softmax',event:'QK_READY_EVENT'},
    {buffer:'probability_transfer',producer:'softmax',consumer:'pv',event:'SOFTMAX_READY_EVENT'},
    {buffer:'pv_transfer',producer:'pv',consumer:'accumulate',event:'PV_READY_EVENT'}];
  function createModel({anatomy,preLaunch}){
    return {...anatomy,preLaunch,tasks:anatomy.tasks.map(task=>{
      const transferBase=task.block*anatomy.transferSlots*anatomy.rowTile;
      const stacks=task.stacks.map(stack=>({...stack,produceTick:stack.index,consumeTick:stack.index+preLaunch,
        produceRow:transferBase+stack.slot*anatomy.rowTile,consumeRow:transferBase+stack.slot*anatomy.rowTile}));
      const ticks=Array.from({length:stacks.length+preLaunch},(_,index)=>({index,
        produce:index<stacks.length?index:null,consume:index>=preLaunch?index-preLaunch:null}));
      return {...task,stacks,ticks,transferBase,dependencies:stacks.flatMap(s=>transfers.map(t=>({...t,stack:s.index,slot:s.slot}))),
        reuse:stacks.filter(s=>s.index>=anatomy.transferSlots).flatMap(s=>transfers.map(t=>({...t,slot:s.slot,from:s.index-anatomy.transferSlots,to:s.index}))),
        operations:stacks.flatMap(s=>Object.keys(stages).map(stage=>({stack:s.index,stage,tick:['qk','softmax'].includes(stage)?s.produceTick:s.consumeTick,slot:s.slot,...buffers[stage]})))};
    })};
  }
  function mount(host,{model,onTask,onFocus,onCamera,onInspector}){
    host.classList.add('viz2-anatomy','viz2-intra');
    host.innerHTML=`<div class="viz2-anatomy-taskbar"><span>同一 Block</span><div role="group" aria-label="切换流水线 Task" data-intra-tasks></div><span data-intra-context></span></div><div class="viz2-anatomy-body"><div class="viz2-anatomy-stage"><div class="viz2-anatomy-viewport" tabindex="0" role="group" aria-label="Stack 与 Tick 逻辑流水线，滚轮缩放，拖拽平移"><svg class="viz2-intra-svg" aria-label="Stack × Tick × Stage"><g data-intra-scene></g></svg><div class="kf-pa-map__tooltip" role="tooltip" hidden></div></div><footer class="viz2-anatomy-footer"><span><small>细箭头 = 数据依赖 · 粗弧线 = 同 Buffer Slot 复用 · 无时间比例</small></span><div class="kf-pa-map__tools"><button class="btn btn-ghost btn-sm" data-intra-zoom="0.8" aria-label="缩小逻辑流水线">−</button><output data-intra-scale></output><button class="btn btn-ghost btn-sm" data-intra-zoom="1.25" aria-label="放大逻辑流水线">＋</button><button class="btn btn-ghost btn-sm" data-intra-fit>Fit to View</button></div></footer></div><aside class="pto-ide-frame__pane viz2-anatomy-inspector" aria-label="流水线对象详情" hidden><header class="pto-ide-frame__pane-header"><b data-intra-title></b><button class="btn btn-ghost btn-sm" data-intra-close aria-label="关闭流水线详情">×</button></header><div class="pto-ide-frame__pane-body" data-intra-detail></div></aside></div>`;
    const viewport=host.querySelector('.viz2-anatomy-viewport'),scene=host.querySelector('[data-intra-scene]'),inspector=host.querySelector('aside'),tooltip=host.querySelector('[role="tooltip"]');
    let task=null,focus=null,camera=null,active=false,drag=null,suppress=false,width=0,height=0,inspectorOpen=false;
    const X=116,COL=124,W=94,H=44,rows={qk:150,softmax:234,pv:350,accumulate:434};
    const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const text=(x,y,s,cls='')=>`<text x="${x}" y="${y}" class="${cls}">${esc(s)}</text>`;
    const rect=(x,y,w,h,cls,extra='')=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="4" class="${cls}" ${extra}/>`;
    const object=(kind,index,content,attrs='')=>`<g tabindex="0" role="button" data-intra-kind="${kind}" data-intra-index="${index}" ${attrs} aria-pressed="false">${content}</g>`;
    const stagePosition=(s,stage)=>({x:X+(['qk','softmax'].includes(stage)?s.produceTick:s.consumeTick)*COL,y:rows[stage]});
    const stroke=slot=>`style="--slot-color:var(--intra-slot-${slot%3})"`;
    function dependencyGeometry(link){
      const a=stagePosition(task.stacks[link.stack],link.producer),b=stagePosition(task.stacks[link.stack],link.consumer);
      const x=a.x+W/2,y=a.y+H,tx=b.x+W/2,ty=b.y;
      return {d:link.producer==='softmax'?`M ${x} ${y} C ${x} ${y+34}, ${tx} ${ty-34}, ${tx} ${ty}`:`M ${x} ${y} V ${ty}`,
        event:{x:(x+tx)/2,y:(y+ty)/2}};
    }
    function reuseGeometry(link){
      const a=stagePosition(task.stacks[link.from],link.consumer),b=stagePosition(task.stacks[link.to],link.producer);
      if(link.buffer==='score_transfer'){
        const top=80-link.slot*17,channel=a.x-13;
        return `M ${a.x} ${a.y+H/2} C ${channel} ${a.y+H/2}, ${channel} ${top}, ${a.x+12} ${top} H ${b.x+W-28} Q ${b.x+W-10} ${top} ${b.x+W-10} ${top+18} V ${b.y}`;
      }
      if(link.buffer==='probability_transfer'){
        const channel=304+link.slot*10;
        return `M ${a.x+W/2} ${a.y} C ${a.x+W/2} ${channel}, ${b.x+W/2} ${channel}, ${b.x+W/2} ${b.y+H}`;
      }
      const bottom=518+link.slot*17;
      return `M ${a.x+W/2} ${a.y+H} V ${bottom-12} Q ${a.x+W/2} ${bottom} ${a.x+W/2+12} ${bottom} H ${b.x+W+1} Q ${b.x+W+13} ${bottom} ${b.x+W+13} ${bottom-12} V ${b.y+H/2+12} Q ${b.x+W+13} ${b.y+H/2} ${b.x+W} ${b.y+H/2}`;
    }
    function draw(){
      const last=task.stacks.at(-1),lastX=last?stagePosition(last,'accumulate').x:X;
      const outputX=lastX+W+34,outputY=rows.accumulate;
      width=outputX+210;height=590;
      host.querySelector('[data-intra-tasks]').innerHTML=model.tasks.filter(t=>t.block===task.block).map(t=>`<button class="btn btn-ghost btn-sm ${t.id===task.id?'is-selected':''}" data-intra-task="${t.id}" aria-pressed="${t.id===task.id}">Task ${t.id}</button>`).join('');
      host.querySelector('[data-intra-context]').textContent=`Batch ${task.batch} · KV ${task.kvHead} · ${task.pages.length} Pages · ${task.stacks.length} Stacks · ${task.ticks.length} Ticks`;
      scene.innerHTML=`<defs><marker id="intra-dependency-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke"/></marker></defs>`+
        rect(16,12,width-32,height-24,'viz2-intra-frame','data-intra-frame')+text(30,35,`Task ${task.id} · Stack × Tick`,'viz2-intra-heading')+
        Object.entries(rows).map(([stage,y])=>text(24,y+27,stage==='accumulate'?'Accumulate':stages[stage],'viz2-intra-row-title')).join('')+
        task.ticks.map(t=>object('tick',t.index,rect(X+t.index*COL,113,W,24,'viz2-intra-tick-hit')+text(X+t.index*COL+W/2,130,`T${t.index}`,'viz2-intra-tick-label'),`aria-label="Tick ${t.index}"`)).join('')+
        task.reuse.map(link=>`<path tabindex="0" role="button" aria-pressed="false" aria-label="${link.buffer} Slot ${link.slot}，Stack ${link.from} 到 ${link.to} 复用" data-intra-kind="slot" data-intra-index="${link.slot}" data-intra-buffer="${link.buffer}" data-intra-from="${link.from}" data-intra-to="${link.to}" data-intra-reuse="${link.buffer}:${link.from}:${link.to}" class="viz2-intra-reuse" ${stroke(link.slot)} d="${reuseGeometry(link)}"/>`).join('')+
        task.stacks.map(stack=>`<g data-intra-stack-path="${stack.index}" ${stroke(stack.slot)}>`+
          task.dependencies.filter(link=>link.stack===stack.index).map(link=>{const g=dependencyGeometry(link);return `<path class="viz2-intra-flow" data-intra-dependency="${stack.index}:${link.buffer}" d="${g.d}" marker-end="url(#intra-dependency-arrow)"/>`+
            object('event',stack.index,`<circle cx="${g.event.x}" cy="${g.event.y}" r="6" class="viz2-intra-event"/>`,`data-intra-event="${link.event}" data-intra-buffer="${link.buffer}" aria-label="Stack ${stack.index} ${link.event}"`);}).join('')+
          Object.keys(stages).map(stage=>{const n=stagePosition(stack,stage);return object('stage',stack.index,
            rect(n.x,n.y,W,H,'viz2-intra-operation',`data-intra-operation="${stack.index}:${stage}"`)+text(n.x+10,n.y+18,`S${stack.index}`)+text(n.x+10,n.y+35,`Slot ${stack.slot}`,'viz2-intra-node-slot'),
            `data-intra-stage="${stage}" aria-label="Stack ${stack.index} ${stages[stage]}，Slot ${stack.slot}"`);}).join('')+`</g>`).join('')+
        `<path class="viz2-intra-output-link" d="M ${lastX+W} ${outputY+H/2} H ${outputX}" marker-end="url(#intra-dependency-arrow)"/>`+
        object('output',0,rect(outputX,outputY,198,H,'viz2-intra-output-hit','data-intra-output')+rect(outputX,outputY,90,H,'viz2-intra-output')+text(outputX+9,outputY+18,'Output','viz2-intra-small')+text(outputX+9,outputY+35,'Normalize','viz2-intra-small')+
          `<path class="viz2-intra-output-link" d="M ${outputX+90} ${outputY+H/2} H ${outputX+108}" marker-end="url(#intra-dependency-arrow)"/>`+rect(outputX+108,outputY,90,H,'viz2-intra-output')+text(outputX+119,outputY+18,'BF16','viz2-intra-small')+text(outputX+119,outputY+35,'Store','viz2-intra-small'),`aria-label="全部 Stack 完成后的 Output Normalize 与 BF16 Store"`);
    }
    function objectFocus(el){return el?{kind:el.dataset.intraKind,index:Number(el.dataset.intraIndex),...(el.dataset.intraStage?{stage:el.dataset.intraStage}:{}),...(el.dataset.intraEvent?{event:el.dataset.intraEvent}:{}),...(el.dataset.intraBuffer?{buffer:el.dataset.intraBuffer}:{}),...(el.dataset.intraFrom?{from:Number(el.dataset.intraFrom),to:Number(el.dataset.intraTo)}:{})}:null;}
    function effective(f){if(f?.kind==='page')return {kind:'stack',index:task.pages[f.index]?.stack};return f;}
    function paintFocus(current=focus){
      const f=effective(current),fixed=effective(focus),chosen=f&&['stack','stage','event'].includes(f.kind)?task.stacks[f.index]:null;
      const slot=f?.kind==='slot'?f.index:chosen?.slot;
      const relevantBuffers=f?.buffer?[f.buffer]:f?.stage?Object.values(buffers[f.stage]):transfers.map(t=>t.buffer);
      scene.querySelectorAll('[data-intra-reuse]').forEach(el=>el.classList.toggle('is-highlighted',Number(el.dataset.intraIndex)===slot&&relevantBuffers.includes(el.dataset.intraBuffer)));
      scene.querySelectorAll('[data-intra-stack-path]').forEach(el=>{
        const s=task.stacks[Number(el.dataset.intraStackPath)];el.classList.toggle('is-related',f?.kind!=='event'&&s.index===chosen?.index);el.classList.toggle('is-shared',slot!=null&&s.slot===slot&&s.index!==chosen?.index);
      });
      scene.querySelectorAll('[data-intra-kind]').forEach(el=>{
        const item=objectFocus(el),s=task.stacks[item.index];let related=false;
        if(f?.kind==='tick')related=item.kind==='tick'?item.index===f.index:item.kind==='stage'&&(['qk','softmax'].includes(item.stage)?s.produceTick:s.consumeTick)===f.index;
        if(f?.kind==='event')related=item.kind==='stage'&&item.index===f.index&&events[f.event]?.includes(item.stage);
        if(f?.kind==='slot')related=['stack','stage'].includes(item.kind)&&s?.slot===f.index;
        const selected=!!fixed&&item.kind===fixed.kind&&item.index===fixed.index&&item.stage===fixed.stage&&item.event===fixed.event&&item.buffer===fixed.buffer&&(item.kind!=='slot'||item.from===fixed.from);
        el.classList.toggle('is-selected',selected);el.classList.toggle('is-related',related);el.setAttribute('aria-pressed',String(selected));
        if(fixed?.kind==='stack'&&item.kind==='stage'&&item.index===fixed.index)el.classList.add('is-selected');
        if(item.kind==='slot')el.classList.toggle('is-related',item.index===slot&&relevantBuffers.includes(item.buffer));
      });
    }
    function describe(f){
      f=effective(f);if(!f)return {title:`Task ${task.id}`,facts:[]};
      const s=task.stacks[f.index],slot=f.kind==='slot'?f.index:s?.slot;
      const reuse=task.stacks.filter(item=>item.slot===slot).map(item=>item.index);
      if(f.kind==='tick'){const t=task.ticks[f.index];return {title:`Tick ${t.index}`,facts:[['produce_stack',t.produce??'无'],['consume_stack',t.consume??'无'],['活跃阶段',[t.produce!=null?`Stack ${t.produce} · QK / Softmax`:'',t.consume!=null?`Stack ${t.consume} · PV / Context Accumulate`:''].filter(Boolean).join('；')]]};}
      if(f.kind==='output')return {title:'Attention Output',facts:[['开始条件',`全部 ${task.stacks.length} Stacks 完成累加`],['Normalize','ctx = BF16(o / l_sum)'],['Store',`out2d rows ${task.output.rowStart}–${task.output.rowEnd}`],['Shape',task.output.shape.join(' × ')]]};
      if(f.kind==='slot'){
        const transfer=transfers.find(t=>t.buffer===f.buffer);
        return {title:`${f.buffer||'Transfer buffers'} · Slot ${slot}`,facts:[['Slot ID',slot],['前次 Stack / 后次 Stack',f.from!=null?`S${f.from} → S${f.to}`:reuse.map(i=>`S${i}`).join(' → ')],['Buffer',f.buffer||transfers.map(t=>t.buffer).join(' / ')],['Producer / Consumer',transfer?`${stages[transfer.producer]} / ${stages[transfer.consumer]}`:'按各 Buffer 分别映射'],['READY Event',transfer?.event||'按 Buffer 区分'],['复用端点',transfer?`S${f.from} ${stages[transfer.consumer]} → S${f.to} ${stages[transfer.producer]}`:'前次消费 → 后次写入'],['produce_row / consume_row',`${task.transferBase+slot*model.rowTile} / ${task.transferBase+slot*model.rowTile}`]]};
      }
      const pairs=f.kind==='event'?events[f.event]:null,stage=f.stage;
      const involved=transfers.filter(t=>f.buffer?t.buffer===f.buffer:stage?(t.producer===stage||t.consumer===stage):true);
      return {title:f.kind==='event'?`Stack ${s.index} · ${f.event}`:f.kind==='stage'?`Stack ${s.index} · ${stages[stage]}`:`Stack ${s.index}`,facts:[
        ['stack index',s.index],['逻辑 Pages',`P${s.pages[0]}–P${s.pages.at(-1)}`],['valid tokens',`${s.tokens} / ${task.stackTokens}`],['Slot ID',s.slot],['阶段',stage?stages[stage]:pairs?pairs.map(k=>stages[k]).join(' → '):'完整 Stack 路径'],['Buffer',involved.map(t=>t.buffer).join(' / ')],['Producer / Consumer',involved.map(t=>`${t.buffer}: ${stages[t.producer]} / ${stages[t.consumer]}`).join('；')],['READY Event',involved.map(t=>t.event).join(' / ')],['前后复用 Stack',`S${s.index-model.transferSlots>=0?s.index-model.transferSlots:'—'} ← S${s.index} → S${s.index+model.transferSlots<task.stacks.length?s.index+model.transferSlots:'—'}`],
        ['produce tick / consume tick',`${s.produceTick} / ${s.consumeTick}`],['produce_row / consume_row',`${s.produceRow} / ${s.consumeRow}`],
        ...(pairs?[['生产者 / 消费者',pairs.map(k=>stages[k]).join(' → ')],['Transfer buffer',buffers[pairs[0]].write]]:stage?[['写入',buffers[stage].write||'—'],['读取',buffers[stage].read||'—']]:[['QK → Softmax','写入 / 读取 score_transfer'],['Softmax → PV','写入 / 读取 probability_transfer'],['PV → Accumulate','写入 / 读取 pv_transfer']]),
        ['row 区间',`${s.produceRow}–${s.produceRow+model.rowTile-1}（AIV 按 lane_row 切分）`]]};
    }
    function inspect(){inspector.hidden=!inspectorOpen||!focus;if(inspector.hidden)return;const d=describe(focus);host.querySelector('[data-intra-title]').textContent=d.title;host.querySelector('[data-intra-detail]').innerHTML=`<dl>${d.facts.map(([key,value])=>`<dt>${esc(key)}</dt><dd>${esc(value)}</dd>`).join('')}</dl><p>静态逻辑映射；Tick 不表示实测时长，Slot 连线不表示 runtime 占用或额外依赖。</p>`;}
    function fitted(){const scale=Math.max(.05,Math.min((viewport.clientWidth-32)/width,(viewport.clientHeight-24)/height,1.2));return {scale,x:(viewport.clientWidth-width*scale)/2,y:(viewport.clientHeight-height*scale)/2};}
    function paint(){if(!active||!task||!viewport.clientWidth)return;if(!camera){camera=fitted();onCamera(task.id,camera);}scene.setAttribute('transform',`translate(${camera.x} ${camera.y}) scale(${camera.scale})`);scene.style.setProperty('--intra-font',`${Math.max(12,11/camera.scale)}px`);scene.style.setProperty('--intra-small-font',`${Math.max(11,10/camera.scale)}px`);host.querySelector('[data-intra-scale]').textContent=`${Math.round(camera.scale*100)}%`;paintFocus();}
    function zoom(factor,point={x:viewport.clientWidth/2,y:viewport.clientHeight/2}){if(!camera)return;const scale=Math.max(.05,Math.min(8,camera.scale*factor)),ratio=scale/camera.scale;onCamera(task.id,{scale,x:point.x-(point.x-camera.x)*ratio,y:point.y-(point.y-camera.y)*ratio});}
    host.addEventListener('click',event=>{if(suppress){suppress=false;return;}const target=event.target.closest('[data-intra-kind]');if(target){tooltip.hidden=true;onFocus(objectFocus(target));return;}const button=event.target.closest('[data-intra-task]');if(button){onTask(Number(button.dataset.intraTask));return;}if(event.target.closest('[data-intra-close]')){onInspector(false);return;}if(event.target.closest('[data-intra-fit]')){onCamera(task.id,fitted());return;}const z=event.target.closest('[data-intra-zoom]');if(z)zoom(Number(z.dataset.intraZoom));});
    viewport.addEventListener('keydown',event=>{if(['Enter',' '].includes(event.key)&&event.target.closest('[data-intra-kind]')){event.preventDefault();onFocus(objectFocus(event.target.closest('[data-intra-kind]')));}if(event.key==='Escape'){tooltip.hidden=true;onInspector(false);}});
    viewport.addEventListener('wheel',event=>{event.preventDefault();tooltip.hidden=true;const r=viewport.getBoundingClientRect(),unit=event.deltaMode===1?16:event.deltaMode===2?r.height:1;zoom(Math.exp(Math.max(-.22,Math.min(.22,-event.deltaY*unit*.003))),{x:event.clientX-r.left,y:event.clientY-r.top});},{passive:false});
    viewport.addEventListener('pointerdown',event=>{if(event.button===0){suppress=false;drag={id:event.pointerId,x:event.clientX,y:event.clientY,startX:event.clientX,startY:event.clientY,moved:false};}});
    viewport.addEventListener('pointermove',event=>{if(drag){if(!drag.moved&&Math.hypot(event.clientX-drag.startX,event.clientY-drag.startY)<=4)return;drag.moved=true;viewport.setPointerCapture(drag.id);tooltip.hidden=true;onCamera(task.id,{...camera,x:camera.x+event.clientX-drag.x,y:camera.y+event.clientY-drag.y});drag.x=event.clientX;drag.y=event.clientY;return;}const el=event.target.closest('[data-intra-kind]');if(!el){tooltip.hidden=true;paintFocus();return;}const f=objectFocus(el),d=describe(f);paintFocus(f);tooltip.innerHTML=`<b>${esc(d.title)}</b><br>${d.facts.filter(([key])=>['Slot ID','阶段','Buffer','Producer / Consumer','READY Event','前后复用 Stack','前次 Stack / 后次 Stack','复用端点','活跃阶段','开始条件','Normalize','Store'].includes(key)).map(([k,v])=>`${esc(k)}: ${esc(v)}`).join('<br>')}<br>静态逻辑映射 · 非 runtime 实测占用`;tooltip.hidden=false;const r=viewport.getBoundingClientRect();tooltip.style.left=`${Math.max(8,Math.min(event.clientX-r.left+16,r.width-tooltip.offsetWidth-8))}px`;tooltip.style.top=`${Math.max(8,Math.min(event.clientY-r.top+18,r.height-tooltip.offsetHeight-8))}px`;});
    const end=()=>{suppress=!!drag?.moved;drag=null;};['pointerup','pointercancel','lostpointercapture'].forEach(type=>viewport.addEventListener(type,end));viewport.addEventListener('pointerleave',()=>{tooltip.hidden=true;paintFocus();});
    let lastSize=null;
    new ResizeObserver(()=>{if(!active||!task||!viewport.clientWidth||!viewport.clientHeight)return;
      const size={width:viewport.clientWidth,height:viewport.clientHeight};
      if(camera&&lastSize&&(size.width!==lastSize.width||size.height!==lastSize.height))
        onCamera(task.id,{...camera,x:camera.x+(size.width-lastSize.width)/2,y:camera.y+(size.height-lastSize.height)/2});
      else paint();
      lastSize=size;
    }).observe(viewport);
    return {update(id,nextFocus,nextCamera,options={}){active=!!options.active;if(!active){tooltip.hidden=true;return;}if(task?.id!==id){task=model.tasks[id];draw();}focus=nextFocus;inspectorOpen=options.inspectorOpen;camera=nextCamera;inspect();paint();},hideTooltip(){tooltip.hidden=true;},
      getTransitionSnapshot(){if(!active||!task)return null;const r=el=>{const b=el.getBoundingClientRect();return {x:b.left,y:b.top,width:b.width,height:b.height};};return {task:task.id,model:task,frame:r(scene.querySelector('[data-intra-frame]')),title:r(scene.querySelector('.viz2-intra-heading')),output:r(scene.querySelector('[data-intra-output]')),stacks:task.stacks.map(s=>({index:s.index,rect:r(scene.querySelector(`[data-intra-operation="${s.index}:qk"]`))}))};},
      setTransitionMasked(value){host.classList.toggle('viz2-intra-masked',value);}};
  }
  global.PtoVisualization2PipelineIntra=Object.freeze({createModel,mount});
})(window);
