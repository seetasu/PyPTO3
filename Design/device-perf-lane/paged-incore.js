window.PAGED_INCORE=(()=>{
  let selected=null,tab='hardware',pass='split',view=null;
  const data=()=>window.PAGED_INCORE_DATA,esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),us=n=>n.toFixed(3)+' μs';
  const passFor=pipe=>pipe==='CUBE'||pipe==='MTE1'?'lower':pipe==='MTE2'?'memory':'split';
  const coreName=s=>s.includes('cubecore')?'仿真 AIC':s.endsWith('veccore0')?'仿真 AIV0':'仿真 AIV1';
  function reset(){selected=null;tab='hardware';pass='split';}
  function selectEvent(id){selected=data().events.find(e=>e.id===Number(id))||null;if(selected){pass=passFor(selected.pipe);tab='pass';}}
  function renderInline(svg,{y,height,W,LEFT,record,ct}){
    const d=data(),ns='http://www.w3.org/2000/svg',make=(tag,a={},p=svg)=>{const n=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(a))n.setAttribute(k,v);p.append(n);return n;},g=make('g',{class:'pa-inline-pipeline','data-pa-core':record.core}),text=(s,x,py,size=10,color='#aaa',p=g)=>{const n=make('text',{x,y:py,'font-size':size,fill:color,'font-family':'var(--mono)'},p);n.textContent=s;return n;};
    make('rect',{x:8,y:y+4,width:W-24,height:height-12,fill:'#151515',stroke:'#383838'},g);
    text('Core '+record.core+' · 硬件 block',LEFT-95,y+25,12,'#eee');text('硬件实测 · '+us(record.computeUs)+' · 不含核内子事件',LEFT+135,y+25,10);
    const close=make('g',{role:'button',tabindex:0,'data-pa-collapse':'true',class:'pa-inline-control'},g);make('rect',{x:W-88,y:y+10,width:60,height:24,rx:12,fill:'#222',stroke:'#555'},close);text('收起',W-72,y+26,10,'#ccc',close);
    text('指令仿真 · 独立时间轴 · core0 的 AIC / AIV0 / AIV1 · 非所选物理核实测',LEFT-95,y+48,10,'#bcbcbc');
    const x=t=>LEFT+(t-d.start)/(d.end-d.start)*(W-LEFT-24);
    for(let i=0;i<=8;i++){const t=d.start+(d.end-d.start)*i/8,px=x(t);make('line',{x1:px,x2:px,y1:y+70,y2:y+74+d.tracks.length*18,stroke:'#292929'},g);text(t.toFixed(2),px,y+66,9,'#888');}
    text('仿真 μs',LEFT-65,y+66,9,'#888');
    const ownType=ct==='AIC'?'cubecore':'veccore';
    d.tracks.forEach((track,index)=>{const py=y+76+index*18,[core,pipe]=track.split('|'),active=core.includes(ownType);make('rect',{x:LEFT,y:py,width:W-LEFT-24,height:16,fill:index%2?'#141414':'#1b1b1b'},g);text(coreName(core)+' · '+pipe,LEFT-8,py+11,9,active?'#bbb':'#777').setAttribute('text-anchor','end');
      for(const ev of d.events.filter(e=>e.core===core&&e.pipe===pipe)){const chosen=selected?.id===ev.id,box=make('g',{class:'pa-sim-event'+(chosen?' selected':''),role:'button',tabindex:0,'data-pa-sim-event':ev.id,'aria-label':coreName(core)+' '+ev.pipe+' '+ev.name,opacity:chosen||active?1:.4},g),px=x(ev.start),w=Math.max(.7,x(ev.end)-px);make('rect',{x:px,y:py+2,width:w,height:12,rx:.6,fill:pipe==='FLOWCTRL'?'#555':'#999',stroke:chosen?'#fff':'none','stroke-width':chosen?2:0},box);make('rect',{x:px-2,y:py,width:Math.max(5,w+4),height:16,fill:'transparent'},box);const title=make('title',{},box);title.textContent=ev.name+' · '+us(ev.start)+' → '+us(ev.end)+' · PC '+ev.pc;}
    });
    text('仿真输入 128/129；10/16 硬件 profile 输入 512/513；当前 block 来自 10/15。仅代码同源，不绑定计时。',LEFT-95,y+height-20,9,'#888');
    const current=svg.querySelector('.task-event[data-core="'+record.core+'"]')?.dataset.id;
    for(const node of svg.querySelectorAll('.task-event[data-id]')){node.classList.toggle('pa-other-side',node.dataset.id!==current);node.classList.toggle('pa-other-record',node.dataset.id===current&&node.getAttribute('data-core')!==String(record.core));node.classList.toggle('pa-selected-record',node.dataset.id===current&&node.getAttribute('data-core')===String(record.core));}
    return g;
  }
  function passDiagram(p,side){const f=p[side].facts;let content='';
    if(p.id==='split')content=side==='before'?'<div class="pa-compile-node">混合计算体<br>QK / PV + 向量处理</div>':'<div class="pa-compile-node">AIC · QK / PV</div><div class="pa-transfer">score_transfer ↓ · probability_transfer ↑<br>pv_transfer ↓ · sync_wait</div><div class="pa-compile-node">AIV · Softmax / 累加 / 输出</div>';
    if(p.id==='skew')content='<div class="pa-compile-ticks">'+[0,1,2,3].map(t=>'<div><b>tick '+t+'</b><span>QK '+t+'</span><span>'+(t<2?'填充':'PV '+(t-2))+'</span></div>').join('')+'</div><small>produce=tick · consume=tick−2 · ring % 3</small><strong class="pa-no-change">主 tick 调度'+(side==='before'?'已存在':'保持不变')+'</strong>';
    if(p.id==='lower')content='<div class="pa-compile-pages">'+(side==='before'?'<span>page 0</span><span>page 1</span><span>page 2</span><span>page 3</span>':'<span>offset 0<br>QK₀ + QK₁</span><span>offset 2<br>QK₂ + QK₃</span>')+'</div><small>'+esc(f.pipelineHeads[0]||'')+'</small><div>matmul 调用位置：'+f.matmulSites+'</div>';
    if(p.id==='memory')content='<div class="pa-compile-pages"><span>Q / K / V<br>Tile</span><span>'+(side==='before'?'待推导存储':'Mat / Left / Right / Acc / Vec')+'</span></div><div>IR Mat 标注数：'+f.memMat+'</div>';
    return '<article class="pa-pass-side"><h4>Pass '+(side==='before'?'前':'后')+'</h4>'+content+'<details><summary>对应 IR 证据</summary>'+p[side].snippets.map(s=>'<div class="pa-code-line"><small>L'+s.line+'</small><code>'+esc(s.text)+'</code></div>').join('')+'</details><a target="_blank" rel="noopener" href="../../'+esc(data().source)+'/passes_dump/'+p[side].file+'">'+p[side].file+' ↗</a></article>';
  }
  function renderPanel(panel,rank,state,select){
    view={panel,rank,state,select};const d=data(),side=rank.launches.find(l=>l.id===state.id),record=side.blocks.items.find(b=>b.core===state.core),ev=selected,p=d.passes.find(p=>p.id===pass);
    const tabs='<nav class="pa-incore-tabs" aria-label="核内分析数据类型">'+[['hardware','硬件实测'],['sim','指令仿真'],['pass','Pass 分析']].map(([id,label])=>'<button type="button" data-pa-tab="'+id+'" aria-pressed="'+(id===tab)+'">'+label+'</button>').join('')+'</nav>';
    const event=ev?'<div class="pa-selected-instr"><span class="pa-provenance sim">指令仿真</span><strong>'+coreName(ev.core)+' · '+ev.pipe+' · '+ev.name+'</strong><small>'+us(ev.start)+' → '+us(ev.end)+' · PC '+esc(ev.pc)+'</small></div>':'';
    let content='';
    if(tab==='hardware')content='<span class="pa-provenance hardware">硬件实测 · 20261015</span><h3>'+side.ct+' · Core '+record.core+'</h3><dl class="pa-hw-values">'+[['执行',record.computeUs],['Setup',record.setupUs],['占用',record.slotUs],['起点',record.startUs],['终点',record.endUs]].map(([label,value])=>'<div><dt>'+label+'</dt><dd>'+us(value)+'</dd></div>').join('')+'</dl><p>这一条硬件记录只覆盖整个 block；展开区是同源生成代码的独立指令仿真参考，不是该物理核的计时分解。</p>';
    if(tab==='sim')content='<span class="pa-provenance sim">指令仿真 · 20261016</span>'+event+'<p>'+d.events.length+' 条提取的搬运、计算与同步指令，'+d.tracks.length+' 条执行引擎泳道。</p><p>'+esc(d.workload.simulation)+'；'+esc(d.workload.hardware)+'。</p>'+(ev?'<code class="pa-instr-detail">'+esc(ev.detail)+'</code>':'<p>点击展开区的指令查看时间、PC 与相关 Pass。</p>');
    if(tab==='pass')content=event+'<span class="pa-provenance compile">编译结构 · 当前算子实际 Pass 编号</span><nav class="pa-pass-nav">'+d.passes.map(item=>'<button type="button" data-pa-pass="'+item.id+'" aria-pressed="'+(item.id===pass)+'">'+item.label+'<small>'+item.name+'</small></button>').join('')+'</nav><p class="pa-pass-note">'+(p.id==='skew'?'该算子的 tick−2 跨栈消费在 P30 前已存在；本对照不虚构新的 QK/PV 重排。':p.id==='lower'?'P32 将 page 流水从 4 次 stage=2 改为步长 2、stage=1 的展开结构；不推导性能收益。':p.id==='split'?'通过 score / probability / pv_transfer 和同步连接 AIC/AIV，不能套用 DeepSeek 的直接 pipe 示例。':'展示实际存储推导前后；不预设当前 Q 重复搬运或存在循环外提缺陷。')+'</p><div class="pa-pass-pair">'+passDiagram(p,'before')+passDiagram(p,'after')+'</div><p class="pa-engine-association">指令按引擎关联到分析环节；PC 尚未逐条映射到 IR，不把 MMAD 自动命名为某次 QK 或 PV。</p>';
    panel.innerHTML='<section class="pa-incore-inspector"><header><h2>Core '+state.core+' · 核内分析</h2><button type="button" class="ctl" data-pa-back>返回 Task</button></header><div class="pa-provenance-row"><span class="pa-provenance hardware">硬件实测</span><span class="pa-provenance sim">指令仿真</span><span class="pa-provenance compile">编译结构</span></div>'+tabs+'<div class="pa-incore-content">'+content+'</div><details class="pa-sources"><summary>来源与匹配范围</summary><p>三个生成代码文件 SHA256 相同：'+esc(d.sourceHash)+'</p><p>'+esc(d.workload.scope)+'</p><a target="_blank" rel="noopener" href="../../'+esc(d.source+'/'+d.simulator)+'/trace.json">原始仿真 trace ↗</a></details></section>';
    for(const node of document.querySelectorAll('#lane .task-event[data-id]')){node.classList.toggle('pa-other-side',node.dataset.id!==state.id);node.classList.toggle('pa-other-record',node.dataset.id===state.id&&node.getAttribute('data-core')!==String(state.core));node.classList.toggle('pa-selected-record',node.dataset.id===state.id&&node.getAttribute('data-core')===String(state.core));}
    panel.onclick=e=>{const b=e.target.closest('[data-pa-tab],[data-pa-pass],[data-pa-back]');if(!b)return;if(b.hasAttribute('data-pa-back')){select(state.id,null);return;}if(b.dataset.paTab)tab=b.dataset.paTab;if(b.dataset.paPass){pass=b.dataset.paPass;tab='pass';}renderPanel(panel,rank,state,select);state.redraw?.();};
  }
  return{reset,selectEvent,renderInline,renderPanel,height:()=>data().tracks.length*18+112};
})();
