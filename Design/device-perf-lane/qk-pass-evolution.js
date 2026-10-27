// Continuous candidate compilation story. Geometry expresses structure, never device time.
window.QK_PASS_EVOLUTION=(()=>{
  const phases=[{id:'memory',label:'P19',title:'Q 驻留候选',note:'将重复 Q 加载移到循环外，分片复用同一面板。',file:'19_after_InferTileMemorySpace.py'}, {id:'exchange',label:'P23',title:'拆分 AIC / AIV',note:'显现分数和 P 的双向传递；QK / PV 与向量计算分到不同计算侧。',file:'23_after_ExpandMixedKernel.py'}, {id:'split',label:'P25',title:'明确 AIV 子块',note:'展开 subblock 分支与有效形状，保留已有数据依赖。',file:'25_after_SplitVectorKernel.py'}, {id:'skew',label:'P28',title:'形成跨分片流水',note:'QK₁ 前移到 PV₀ 前，与 AIV₀ 协作；PV₀ 仍等待 P₀。',file:'28_after_SkewCrossCorePipeline.py'}, {id:'lower',label:'P30',title:'展开内部执行结构',note:'保持 QK / PV 主顺序，显现 L0 计算展开。',file:'30_after_LowerPipelineLoops.py'}];
  const sub=n=>String(n).replace(/\d/g,d=>'₀₁₂₃₄₅₆₇₈₉'[+d]),names={q:'Q',mm:'QK',pv:'PV',send_score:'分数',wait_p:'等 P',recv_p:'收 P',scale:'缩放',exp:'exp',reduce:'归约',send_p:'P',write_o:'O',write_ml:'m/l'};
  const controlIcon=kind=>'<svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'+({play:'<path d="M8 5l11 7-11 7z"/>',pause:'<path d="M8 5v14M16 5v14"/>',reset:'<path d="M4 10a8 8 0 1 1 1 8M4 4v6h6"/>'}[kind])+'</svg>';
  let run=null;
  const phaseAt=value=>Math.max(0,Math.min(4,Math.ceil(value)-1)),valueFor=id=>Math.max(1,phases.findIndex(p=>p.id===id)+1);
  function geometry(step,first){
    const kind=step>=4?'skew':'serial',model=window.QK_PIPELINE_PASSES.model(kind,first),x=t=>118+t/model.end*418;
    return model.events.map(ev=>{let row=ev.row>=4?ev.row+1:ev.row;
      if(step<2&&['scale','exp','reduce'].includes(ev.op))row=1;
      let px=x(ev.start),y=36+row*29,w=Math.max(7,x(ev.end)-px),opacity=1;
      if(step<2&&['send_score','send_p','recv_p','wait_p'].includes(ev.op))opacity=.08;
      if(ev.op==='q'&&step>=1){if(ev.iter===first){px=118;y=7;w=95;}else{px=222;y=7;w=78;opacity=.65;}}
      return{...ev,x:px,y,w,opacity};
    });
  }
  function canvas(state,prefix='live'){
    return '<svg class="qk-evolution-canvas" viewBox="0 0 1040 650" role="group" aria-label="QK 编译依赖结构图"></svg>';
  }
  function emphasis(phase,op,iter,first){
    if(phase===0)return op==='q'?'changed':'context';
    if(phase===1)return ['scale','exp','reduce','send_score','send_p','recv_p','wait_p'].includes(op)?'changed':'context';
    if(phase===2)return ['scale','exp','reduce'].includes(op)?'changed':'context';
    if(phase===3){if(op==='mm'&&iter===first+1||op==='pv'&&iter===first)return'changed';return iter===first&&['send_score','scale','exp','reduce','send_p','recv_p','wait_p'].includes(op)?'related':'context';}
    return ['mm','pv'].includes(op)?'changed':'context';
  }
  const resultFor=(phase,first)=>['Q 加载已移到循环外，后续分片改为复用。','向量计算已拆到 AIV，分数与 P 的跨核依赖显现。','子块有效形状已明确，计算顺序没有改变。','QK'+sub(first+1)+' 已前移；PV'+sub(first)+' 仍等待自己的 P'+sub(first)+'。','内部计算已展开，QK/PV 主顺序保持。'][phase];
  function draw(svg,position,state){
    const stage=Math.ceil(position),first=Math.floor(state.iter/2)*2,v=window.VisualSystem;
    const tx=(x,y,t,size=16,color='#bbb')=>`<text x="${x}" y="${y}" font-size="${size}" fill="${color}">${t}</text>`;
    const edge=(x,y,xx,yy)=>v.edge(x,y,xx,yy,'data');
    let body=tx(32,32,stage>=4?`AIC 顺序：QK${sub(first)} → QK${sub(first+1)} → PV${sub(first)} → PV${sub(first+1)}`:`AIC 顺序：QK${sub(first)} → PV${sub(first)} → QK${sub(first+1)} → PV${sub(first+1)}`,19,'#eee');
    body+='<g opacity="'+(stage===1?1:.7)+'">'+v.data({id:'q',label:stage?'Q · 候选驻留':'Q',role:'activation',detail:stage?'循环外 · 两个分片复用':'每个分片加载'},440,62)+'</g>';
    body+='<path d="M520 126V150H220V170M520 150H760V170" fill="none" stroke="#aaa" marker-end="url(#arrow)"/>';
    const op=(key,label,x,y,iter,icon,detail)=>{
      const chosen=state.object===key&&state.iter===iter,focus=emphasis(Math.max(0,stage-1),key,iter,first)==='changed';
      return `<g class="qk-pass-node" opacity="${focus||chosen?1:.65}" data-qk-object="${key}" data-qk-iter="${iter}" tabindex="0" role="button" aria-label="${label} 分片 ${iter}" aria-pressed="${chosen}"><rect x="${x}" y="${y}" width="260" height="48" rx="24" fill="#4f4f4f" stroke="${chosen||focus?'#eee':'none'}"/><circle cx="${x+26}" cy="${y+24}" r="18" fill="#111"/><path d="${window.PASS_LEGEND_ICONS[icon]||''}" transform="translate(${x+14} ${y+12})" fill="none" stroke="#eee" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>${tx(x+54,y+29,label,18,'#fff')}${detail?tx(x+14,y+69,detail,13):''}${stage>=5&&['mm','pv'].includes(key)?`<g transform="translate(${x+38} ${y+51})"><rect width="184" height="22" rx="3" fill="#303030" stroke="#aaa"/>${tx(8,16,'装载  →  矩阵计算  →  累积',12,'#eee')}</g>`:''}</g>`;
    };
    for(let k=0;k<2;k++){
      const x=k?630:90,c=x+130,iter=first+k;
      body+=tx(x,148,'分片 '+iter,15);
      body+=op('mm','QK'+sub(iter),x,170,iter,'MatMul',stage>=5?'':'AIC · 矩阵计算');
      body+='<g opacity="'+(stage===2?1:.7)+'">'+edge(c,238,c,286)+tx(c+14,266,stage>=2?'分数 → AIV':'分数',14)+'</g>';
      body+=op('scale','缩放 · exp · 归约',x,286,iter,'Reduce / Pool',stage>=2?(stage>=3?'AIV0 · 有效向量计算':'AIV · 向量计算'):'混合计算体 · 向量计算');
      body+='<g opacity="'+(stage===2?1:.7)+'">'+edge(c,354,c,400)+tx(c+14,380,stage>=2?'P → AIC':'P',14)+'</g>';
      body+=op('pv','PV'+sub(iter),x,400,iter,'MatMul',stage>=5?'':'等待对应 P · AIC');
      body+=edge(c,468,c,514)+v.data({id:'o'+iter,label:'O'+sub(iter),role:'activation',detail:'输出 / 累积结果'},x+50,514);
    }
    if(stage===4)body+='<path d="M630 194H590V424H350" fill="none" stroke="#eee" stroke-dasharray="5 5"/>'+tx(400,310,'QK'+sub(first+1)+' 前移 · 执行次序',16,'#fff');
    if(stage===3)body+='<rect x="76" y="276" width="828" height="80" rx="8" fill="none" stroke="#aaa" stroke-dasharray="4 4"/>';
    if(stage>=3)body+=tx(370,620,'AIV1 · valid_shape = 0，无有效计算',15,'#aaa');
    svg.innerHTML=v.wrap(body,1040,650).replace(/^.*?<svg[^>]*>/s,'').replace(/<\/svg>$/,'');
  }
  function render(state){return '<section class="qk-pass-evolution"><div class="evo-controls"><div class="evo-transport"><button type="button" class="ctl" data-evo-play aria-label="播放" data-tooltip="播放">'+controlIcon('play')+'</button><button type="button" class="ctl" data-evo-replay aria-label="重置" data-tooltip="重置">'+controlIcon('reset')+'</button><span class="evo-position-label"></span></div><nav class="evo-sequence" aria-label="Pass 阶段"></nav></div><div class="evo-stage-description"><strong></strong><span></span><em class="evo-playback-state"></em><p></p></div>'+canvas(state)+'<details class="evo-evidence"><summary>对照与依据</summary><details class="evo-comparison" data-qk-section="comparison"><summary>展开本阶段前后对照</summary><div class="evo-comparison-body"></div></details><details class="evo-source"><summary>本阶段代码依据</summary><a target="_blank" rel="noopener"></a></details><details class="evo-boundary"><summary>分析口径与归因边界</summary><p></p></details></details><span class="evo-status" role="status" aria-live="polite"></span></section>';}
  function stop(){const r=run;if(!r)return;run=null;r.playing=false;r.last=-1;r.host.removeAttribute('data-holding');const button=r.host.querySelector('[data-evo-play]');if(button){button.innerHTML=controlIcon('play');button.setAttribute('aria-label','播放');}if(r.raf)cancelAnimationFrame(r.raf);r.navigation?.destroy();for(const n of r.linked)n.classList.remove('qk-skew-linked','evo-primary-link');r.host.removeAttribute('aria-busy');}
  function mount(panel,state,onStage){
    stop();const host=panel.querySelector('.qk-pass-evolution');if(!host)return;host.querySelector('.evo-boundary p').textContent=document.querySelector('#pass-boundary')?.textContent||'';const r={host,position:valueFor(state.pass),playing:false,hasRun:false,raf:0,linked:[],last:-1,limit:5,holdRemaining:0};run=r;
    const svg=host.querySelector('.qk-evolution-canvas'),button=host.querySelector('[data-evo-play]');
    const update=()=>{
      if(run!==r)return;if(r.lastDraw!==r.position){draw(svg,r.position,state);r.lastDraw=r.position;}const index=phaseAt(r.position),p=phases[index];r.navigation?.update({index});host.querySelector('.evo-position-label').textContent=r.position===0?'Pass 前':p.label;host.toggleAttribute('data-holding',r.holdRemaining>0);host.querySelector('.evo-playback-state').textContent=r.holdRemaining>0?(r.playing?'阶段停留':'已暂停 · 阶段停留'):r.playing?'演变中':'';
      for(const n of host.querySelectorAll('[data-evo-stage]'))n.setAttribute('aria-pressed',String(Number(n.dataset.evoStage)===index+1));
      if(index!==r.last){r.last=index;onStage(p.id);host.querySelector('.evo-stage-description strong').textContent=p.title;host.querySelector('.evo-stage-description span').textContent=index===0?'候选假设':'后续结构参考';host.querySelector('.evo-stage-description p').textContent=p.note;host.querySelector('.evo-status').textContent=p.label+' '+p.title;
        for(const n of r.linked)n.classList.remove('qk-skew-linked');r.linked=[];const first=Math.floor(state.iter/2)*2,ops=index===0?['q']:index===1?['send_score','recv_p','send_p','wait_p','scale','exp','reduce']:index===2?['scale','exp','reduce']:index===3?['mm','pv','scale','exp','reduce','send_p','recv_p']:['mm','pv','l0'];for(const op of ops)for(const iter of [first,first+1])for(const n of document.querySelectorAll('#lane [data-qk-version][data-qk-inline-event="'+op+'"][data-qk-inline-iter="'+iter+'"]')){n.classList.add('qk-skew-linked');if(emphasis(index,op,iter,first)==='changed')n.classList.add('evo-primary-link');r.linked.push(n);}
        const comparison=host.querySelector('.evo-comparison-body');comparison.innerHTML='<article><h4>'+p.label+' 前</h4>'+canvas(state,'compare-before')+'</article><article><h4>'+p.label+' 后</h4>'+canvas(state,'compare-after')+'</article>';const charts=comparison.querySelectorAll('svg');draw(charts[0],index,state);draw(charts[1],index+1,state);const source=host.querySelector('.evo-source a');source.href='../../Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/passes_dump/'+p.file;source.textContent=p.file+' ↗';
      }
      const firstFragment=Math.floor(state.iter/2)*2;host.querySelector('.evo-stage-description p').textContent=r.holdRemaining>0?resultFor(index,firstFragment):p.note.replace(/QK₁/g,'QK'+sub(firstFragment+1)).replace(/PV₀/g,'PV'+sub(firstFragment)).replace(/AIV₀/g,'AIV'+sub(firstFragment)).replace(/P₀/g,'P'+sub(firstFragment));
      button.innerHTML=controlIcon(r.playing?'pause':'play');button.setAttribute('aria-label',r.playing?'暂停':'播放');button.dataset.tooltip=r.playing?'暂停':'播放';host.toggleAttribute('aria-busy',r.playing);
    };
    const pause=()=>{r.playing=false;if(r.raf)cancelAnimationFrame(r.raf);r.raf=0;update();};
    const start=limit=>{
      r.limit=limit;if(window.matchMedia?.('(prefers-reduced-motion: reduce)').matches){r.position=limit;r.playing=false;r.hasRun=true;r.holdRemaining=0;update();return;}
      r.playing=true;r.hasRun=true;let last=null;update();
      const tick=now=>{
        if(run!==r||!r.playing)return;let dt=last===null?0:Math.max(0,now-last);last=now;
        for(let guard=0;dt>0&&guard<12;guard++){
          if(r.holdRemaining>0){const used=Math.min(dt,r.holdRemaining);r.holdRemaining-=used;dt-=used;if(r.holdRemaining>0)break;if(r.position>=r.limit){r.playing=false;break;}}
          if(r.position>=r.limit){r.playing=false;break;}
          const boundary=Math.min(r.limit,Math.floor(r.position)+1),needed=(boundary-r.position)*1400;
          if(dt<needed){r.position+=dt/1400;dt=0;}else{r.position=boundary;dt-=needed;r.holdRemaining=1200;}
        }
        update();if(!r.playing){r.raf=0;return;}r.raf=requestAnimationFrame(tick);
      };r.raf=requestAnimationFrame(tick);
    };
    host.addEventListener('click',ev=>{const control=ev.target.closest('[data-evo-stage],[data-evo-play],[data-evo-replay]');if(!control)return;ev.stopPropagation();if(run!==r){stop();run=r;}if(control.dataset.evoStage){pause();r.position=Number(control.dataset.evoStage);r.holdRemaining=0;r.hasRun=false;update();return;}if(control.hasAttribute('data-evo-replay')){pause();r.position=1;r.holdRemaining=0;r.hasRun=false;update();return;}if(r.playing){pause();return;}if(!r.hasRun||r.position>=r.limit&&r.holdRemaining===0){r.position=0;r.holdRemaining=0;}start(5);});
    r.navigation=window.SequenceNavigation.mount(host.querySelector('.evo-sequence'),{items:phases.map(p=>({label:p.label,detail:p.title})),variant:'stage',onSelect:index=>{pause();r.position=index+1;r.holdRemaining=0;r.hasRun=false;update();}});
    host.querySelectorAll('.evo-sequence button').forEach(b=>{b.dataset.tooltip=b.title;b.removeAttribute('title');});
    update();
  }
  return{render,mount,stop,geometry,phases};
})();
