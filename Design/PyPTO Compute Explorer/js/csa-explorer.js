/* CSA scene adapter. The shared graph pattern owns node rendering, selection and camera input. */
(function (global) {
  'use strict';
  const model=global.PtoCsaModel, objects=model.objects;
  const state={open:false,tab:'assignment',selection:null,rank:0,
    views:{assignment:{scope:'overall',topic:'gather',camera:null},pipeline:{scope:'overall',topic:'gather',camera:null},graph:{scope:'overall'}}};
  const tabs=['assignment','pipeline','graph'];
  let root, controller, trigger, previousInert=[], graph;
  const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const titles={assignment:'Rank / Data Assignment',pipeline:'Distributed Execution',graph:'CSA Graph'};
  const alias={rows0:'full',rows1:'full',group0:'publish',group1:'publish',mainstate:'main',innerstate:'idxcomp',gather:'push',local:'pre',full:'gather',original:'write',compressed:'main',indexcache:'idxcomp',window:'publish',group:'publish',push:'gather',payload:'gather',readback:'gather',rope:'ready'};
  function node(g,object,rank,x,y,label){const o=objects[object],id='r'+rank+':'+object;
    g.nodes.push({id,object,rank,x,y,width:242,height:62,label:label||o.label,kind:['tensor','state'].includes(o.kind)?'tensor':'module',typeLabel:model.kindLabels[o.kind],colorKey:o.sem,parent:'rank-'+rank});
    return id;
  }
  function edge(g,a,b,type='data'){g.edges.push({source:a,target:b,type:type==='sync'?'control':type==='comm'?'communication':'activation',dashed:type==='sync',color:type==='comm'?'var(--primary)':undefined});}
  function link(g,r,a,b,type){edge(g,'r'+r+':'+a,'r'+r+':'+b,type);}
  function rank(g,r,x,y,w,h,label){g.clusters.push({id:'rank-'+r,label:label||'Rank '+r+' · device '+r,x,y,width:w,height:h,colorKey:'module:decoder',collapsible:false});}
  function blank(w,h){return {width:w,height:h,nodes:[],edges:[],clusters:[]};}
  function assignment(topic){
    const g=blank(1500,topic==='cache'?850:700);
    for(let r=0;r<2;r++){
      const x=30+r*750;rank(g,r,x,25,710,g.height-50,'Rank '+r+' · '+(topic==='output'?'O Groups '+(r?'4–7':'0–3'):'本地 token '+(r?'R1':'R0')));
      if(topic==='gather'){
        node(g,'local',r,x+355,100,'x_normed_t · [T,4096]');
        node(g,'gather',r,x+355,260);
        node(g,'full',r,x+355,410,'x_normed_full · [2T,4096]');
        node(g,'q',r,x+145,570,'Q · 本地 T 行');
        node(g,'kv',r,x+545,570,'KV · 整组 2T 行');
        link(g,r,'local','gather');link(g,r,'gather','full');link(g,r,'local','q');link(g,r,'full','kv');
      } else if(topic==='cache'){
        node(g,'full',r,x+355,105);
        [['kv',145],['main',355],['idxcomp',565]].forEach(([o,c])=>{node(g,o,r,x+c,280);link(g,r,'full',o);});
        [['original',145],['compressed',355],['indexcache',565]].forEach(([o,c])=>node(g,o,r,x+c,460));
        link(g,r,'kv','original');link(g,r,'main','compressed');link(g,r,'idxcomp','indexcache');
        node(g,'attention',r,x+245,690);node(g,'indexer',r,x+555,690);
        link(g,r,'original','attention');link(g,r,'compressed','attention');link(g,r,'indexcache','indexer');link(g,r,'indexer','attention');
      } else {
        node(g,'group',r,x+355,105,'本地 T tokens × 全部 8 Groups');
        node(g,'publish',r,x+355,250);
        node(g,'window',r,x+355,420,'4 Groups × 来自 R0 / R1 的行');
        node(g,'a2a',r,x+355,585,'attention_local_groups');
        link(g,r,'group','publish');link(g,r,'publish','window','comm');link(g,r,'window','a2a','sync');
      }
    }
    if(topic==='gather'){edge(g,'r0:local','r1:gather','comm');edge(g,'r1:local','r0:gather','comm');}
    if(topic==='output'){edge(g,'r0:publish','r1:window','comm');edge(g,'r1:publish','r0:window','comm');}
    return g;
  }
  function pipeline(){
    const g=blank(2070,1040);
    for(let r=0;r<2;r++){
      const y=25+r*515;rank(g,r,25,y,2020,490);
      const positions={pre:[155,120],gather:[440,120],q:[740,70],kv:[740,190],main:[740,310],idxcomp:[740,420],write:[1025,190],indexer:[1025,420],ready:[1290,290],attention:[1510,100],publish:[1800,100],a2a:[1800,230],oproj:[1800,345],rs:[1510,345],post:[1510,450]};
      Object.entries(positions).forEach(([o,[x,dy]])=>node(g,o,r,x,y+dy));
      model.relations.pipeline.forEach(([a,b,t])=>link(g,r,a,b,t));
    }
    edge(g,'r0:gather','r1:gather','comm');edge(g,'r1:gather','r0:gather','comm');
    edge(g,'r0:publish','r1:a2a','comm');edge(g,'r1:publish','r0:a2a','comm');
    edge(g,'r0:oproj','r1:rs','comm');edge(g,'r1:oproj','r0:rs','comm');
    return g;
  }
  function localGraph(topic){
    if(topic==='gather'){
      const g=blank(1500,800);
      for(let r=0;r<2;r++){const x=30+r*750;rank(g,r,x,25,710,750);
        ['local','push','payload','readback','full'].forEach((o,i)=>node(g,o,r,x+355,110+i*145));
        [['local','push'],['push','payload','sync'],['payload','readback','sync'],['readback','full']].forEach(([a,b,t])=>link(g,r,a,b,t));}
      edge(g,'r0:push','r1:payload','comm');edge(g,'r1:push','r0:payload','comm');return g;
    }
    if(topic==='output'){
      const g=blank(1500,1080);
      for(let r=0;r<2;r++){const x=30+r*750;rank(g,r,x,25,710,1030);
        ['attention','publish','window','a2a','oproj','rs','post'].forEach((o,i)=>node(g,o,r,x+355,105+i*145));
        [['attention','publish','sync'],['publish','window','comm'],['window','a2a','sync'],['a2a','oproj'],['oproj','rs','comm'],['rs','post']].forEach(([a,b,t])=>link(g,r,a,b,t));}
      edge(g,'r0:publish','r1:window','comm');edge(g,'r1:publish','r0:window','comm');edge(g,'r0:oproj','r1:rs','comm');edge(g,'r1:oproj','r0:rs','comm');return g;
    }
    const r=state.rank,g=blank(1450,1060);rank(g,r,25,25,1400,1005,'Rank '+r+' · 三条 Cache 分支与两种就绪路径');
    const pos={full:[730,100],local:[160,100],rope:[1150,100],q:[160,305],kv:[460,305],main:[760,305],idxcomp:[1150,305],write:[460,500],compressed:[760,500],indexcache:[1150,500],original:[420,690],ready:[730,690],indexer:[1150,690],attention:[730,930]};
    Object.entries(pos).forEach(([o,[x,y]])=>node(g,o,r,x,y));
    model.relations.cache.forEach(([a,b,t])=>link(g,r,a,b,t));return g;
  }

  function anatomy(topic){
    const r=state.rank;
    if(topic==='gather'){
      const g=blank(1500,800);
      rank(g,0,25,35,440,310,'Source Rank 0 · 本地行');
      rank(g,1,25,430,440,310,'Source Rank 1 · 本地行');
      g.clusters.push({id:'target-rank',label:'目标 Rank '+r+' · 整组副本',x:850,y:35,width:625,height:705,collapsible:false});
      node(g,'local',0,245,185,'R0 · x_normed_t [T,D]');node(g,'local',1,245,580,'R1 · x_normed_t [T,D]');
      node(g,'gather',r,650,380);
      node(g,'full',r,1165,200,'x_normed_full · [2T,D]');
      node(g,'rows0',r,1030,370);node(g,'rows1',r,1300,370);
      node(g,'kv',r,1165,600,'KV / Compressor · 读取整组行');
      edge(g,'r0:local','r'+r+':gather','comm');edge(g,'r1:local','r'+r+':gather','comm');
      link(g,r,'gather','full');link(g,r,'full','kv');
      // Two row-segment marks are a structural decomposition, not two extra copy Tasks.
      g.nodes.filter(n=>['full','rows0','rows1','kv'].includes(n.object)).forEach(n=>n.parent='target-rank');
      return g;
    }
    if(topic==='output'){
      const g=blank(1500,840);
      g.clusters.push({id:'source-attention',label:'Source Rank '+r+' · 本地 T tokens 的全部 8 个 Groups',x:30,y:25,width:1440,height:295,collapsible:false});
      node(g,'group',r,750,105,'64 Heads → 8 O Groups');
      node(g,'group0',r,390,245);node(g,'group1',r,1110,245);
      link(g,r,'group','group0');link(g,r,'group','group1');
      for(let target=0;target<2;target++){const x=30+target*750;rank(g,target,x,405,690,405,'Destination Rank '+target+' · Groups '+(target?'4–7':'0–3'));
        node(g,'window',target,x+345,505,'Window · 4 Groups × 2T_PAD');
        node(g,'a2a',target,x+345,700,'每 Group · [R0 行 | R1 行]');
        link(g,target,'window','a2a');
        edge(g,'r'+r+':group'+target,'r'+target+':window','comm');
      }
      return g;
    }
    const g=blank(1500,900);rank(g,r,25,25,1450,850,'Rank '+r+' · 三种独立 Cache；逻辑 Shape 示意');
    const pos={kv:[260,130],main:[750,130],idxcomp:[1240,130],original:[260,405],compressed:[750,405],indexcache:[1240,405],attention:[500,720],indexer:[1240,720]};
    Object.entries(pos).forEach(([o,[x,y]])=>node(g,o,r,x,y,
      ({original:'Original · [N_ori,128,1,512]',compressed:'Compressed · [N_cmp,128,1,512]',indexcache:'Indexer · INT8 + FP32 scale'})[o]));
    [['kv','original'],['main','compressed'],['idxcomp','indexcache'],['original','attention'],['compressed','attention'],['indexcache','indexer'],['indexer','attention']].forEach(([a,b])=>link(g,r,a,b));
    return g;
  }

  function mount(){
    root=document.createElement('section');root.id='csaExplorer';root.className='pto-ide-frame__pane viz2 csa-explorer';root.hidden=true;
    root.setAttribute('aria-label','DeepSeek V4 CSA Compute Explorer');
    root.innerHTML='<header class="pto-ide-frame__pane-header viz2-header"><button class="btn btn-ghost btn-sm" data-csa-back>← 返回代码</button><span class="pto-ide-frame__pane-title">Compute Explorer</span><span class="pto-ide-frame__pane-meta">decode_csa.py · TP=2</span></header>'+
    '<div class="pto-ide-frame__tabstrip viz2-tabs"><div class="tab-control" role="tablist" aria-label="CSA 可视化视图">'+tabs.map(id=>'<button class="tab-control-item" role="tab" id="csa-tab-'+id+'" data-csa-tab="'+id+'" aria-controls="csa-panel"><span class="viz2-tab-icon viz2-tab-icon--'+id+'"></span>'+id[0].toUpperCase()+id.slice(1)+'</button>').join('')+'</div></div>'+
    '<div class="csa-toolbar"><div class="segmented-control segmented-control-muted" role="group" aria-label="CSA 观察层次"><button class="btn btn-sm" data-csa-scope="overall">整体</button><button class="btn btn-sm" data-csa-scope="local">局部</button></div><nav data-csa-breadcrumb aria-label="当前对象"></nav><button class="btn btn-ghost btn-sm" data-csa-return hidden>← 返回整体</button></div>'+
    '<div class="csa-focusbar" aria-label="关键路径焦点"></div>'+
    '<div class="csa-workspace" id="csa-panel" role="tabpanel"><div class="csa-canvas-column"><div class="csa-caption"></div><div class="csa-stage pto-model-graphviz-pattern-page" role="group" aria-label="CSA 数据与任务关系，拖拽平移，按住 Command 或 Ctrl 滚轮缩放"></div><footer class="csa-footer"><div class="csa-legend"><span>中性矩形 · Tensor / Cache</span><span>彩色模块 · 计算 / 通信 / 同步</span><span>实线 · 数据</span><span class="csa-comm-legend">蓝线 · 跨 Rank</span><span>虚线 · 控制 / 同步</span></div><button class="btn btn-ghost btn-sm" data-csa-fit>全览</button><button class="btn btn-ghost btn-sm" data-csa-read>阅读</button><button class="btn btn-ghost btn-sm" data-csa-zoom="0.8" aria-label="缩小 CSA">−</button><button class="btn btn-ghost btn-sm" data-csa-zoom="1.25" aria-label="放大 CSA">＋</button></footer></div>'+
    '<aside class="pto-ide-frame__pane csa-inspector" aria-label="CSA 对象详情"><header class="pto-ide-frame__pane-header"><span class="pto-ide-frame__pane-title">对象详情</span><button class="btn btn-ghost btn-sm" data-csa-clear>清除选择</button></header><div class="pto-ide-frame__pane-body csa-detail"></div></aside></div>';
    document.getElementById('ideWorkarea').append(root);
    root.addEventListener('click',event=>{
      const b=event.target.closest('button');if(!b)return;
      if(b.hasAttribute('data-csa-back'))return close();
      if(b.dataset.csaTab){saveCamera();const previous=state.views[state.tab];state.tab=b.dataset.csaTab;if(state.tab!=='graph'&&previous.scope==='local'){state.views[state.tab].camera=null;state.views[state.tab].scope='local';state.views[state.tab].topic=selectedTopic(previous.topic);}if(state.tab==='assignment'&&state.selection)state.views.assignment.topic=selectedTopic(previous.topic);render();return;}
      if(b.dataset.csaScope){saveCamera();const v=state.views[state.tab];v.scope=b.dataset.csaScope;if(state.selection)v.topic=selectedTopic(v.topic);v.camera=null;render();return;}
      if(b.hasAttribute('data-csa-return')){state.views[state.tab].scope='overall';state.views[state.tab].camera=null;render();return;}
      if(b.dataset.csaFocus){const topic=b.dataset.csaFocus,v=state.views[state.tab];v.topic=topic;v.camera=null;state.selection={object:{gather:'gather',cache:'ready',output:'publish'}[topic],rank:state.rank};render();return;}
      if(b.hasAttribute('data-csa-clear')){state.selection=null;controller?.clearSelection();details();chrome();return;}
      if(b.hasAttribute('data-csa-drill')){const v=state.views[state.tab];v.scope='local';v.topic=selectedTopic(v.topic);v.camera=null;render();return;}
      if(b.hasAttribute('data-csa-fit')){controller?.fit();return;}if(b.hasAttribute('data-csa-read')){controller?.setTransform({zoom:.85,tx:24,ty:24});return;}
      if(b.dataset.csaZoom){const t=controller?.getTransform();if(t)controller.setTransform({...t,zoom:t.zoom*Number(b.dataset.csaZoom)});return;}
      if(b.dataset.csaSource){const [a,z]=b.dataset.csaSource.split(':').map(Number);global.PtoComputeIDE.showSource(a,z);}
    });
    root.querySelector('[role="tablist"]').addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();const at=tabs.indexOf(state.tab),i=e.key==='Home'?0:e.key==='End'?2:(at+(e.key==='ArrowRight'?1:2))%3;root.querySelector('[data-csa-tab="'+tabs[i]+'"]').click();root.querySelector('[data-csa-tab="'+tabs[i]+'"]').focus();});
  }
  function selectedTopic(fallback){return state.selection?.object==='rank'?(fallback||'gather'):objects[state.selection?.object]?.topic||(fallback||'gather');}
  function saveCamera(){if(controller&&state.tab!=='graph')state.views[state.tab].camera=controller.getTransform();}
  function chrome(){
    root.querySelectorAll('[data-csa-tab]').forEach(b=>{const active=b.dataset.csaTab===state.tab;b.classList.toggle('is-selected',active);b.setAttribute('aria-selected',String(active));b.tabIndex=active?0:-1;});
    root.querySelector('#csa-panel').setAttribute('aria-labelledby','csa-tab-'+state.tab);
    root.querySelectorAll('[data-csa-scope]').forEach(b=>{b.disabled=state.tab==='graph';const active=b.dataset.csaScope===state.views[state.tab].scope;b.classList.toggle('is-selected',active);b.setAttribute('aria-pressed',String(active));});
    root.querySelector('[data-csa-return]').hidden=state.tab==='graph'||state.views[state.tab].scope!=='local';
    root.querySelector('[data-csa-breadcrumb]').textContent=titles[state.tab]+(state.tab==='graph'?'':' / '+(state.views[state.tab].scope==='local'?model.topics[state.views[state.tab].topic]+' / 局部':'整体'))+(state.selection?' / Rank '+state.selection.rank+' / '+objects[state.selection.object].label:'');
    const drill=root.querySelector('[data-csa-drill]');if(drill)drill.disabled=state.tab==='graph';
  }
  function details(){
    const host=root.querySelector('.csa-detail'),selection=state.selection,o=selection&&objects[selection.object];
    if(!o){host.innerHTML='<span class="csa-eyebrow">DEEPSEEK V4 · CSA</span><h2>从数据归属理解计算</h2><p>选择 Rank、Tensor 或任务查看解释；点击“进入局部”查看它所属的关键关系。</p><dl><dt>Assignment</dt><dd>本地输入 → 全组 KV → 按 O Group 重分布</dd><dt>Pipeline</dt><dd>计算分支、Cache 就绪与跨 Rank 通信</dd><dt>默认容量</dt><dd>TP=2，T=256，整组 512 token 行</dd></dl><p class="csa-note">运行时 T 可变。图中两个 Rank 是同一程序的两个设备实例，源码中的 host rank 循环不表示设备串行执行。</p>';host.scrollTop=0;return;}
    host.innerHTML='<span class="csa-eyebrow">RANK '+selection.rank+' · '+model.kindLabels[o.kind]+'</span><h2>'+esc(o.label)+'</h2><dl>'+[['What',o.what],['Why · 静态解释',o.why],['Data · 源码 Shape',o.data],['How · 源码组织',o.how]].map(([k,v])=>'<dt>'+k+'</dt><dd>'+esc(v)+'</dd>').join('')+'</dl><button class="btn btn-sm" data-csa-drill>进入局部 →</button><section class="csa-sources"><h3>Source · 本地源码</h3>'+o.source.map(([f,a,z])=>{
      if(f==='decode_csa.py')return '<button class="btn btn-ghost btn-sm" data-csa-source="'+a+':'+z+'">'+f+' : '+a+'–'+z+' ↗</button>';
      const text=global.PTO_CSA_SOURCE.dependencies[f]?.split('\n').slice(a-1,z).map((l,i)=>(a+i)+'  '+l).join('\n')||'';
      return '<details><summary>'+esc(f)+' : '+a+'–'+z+'</summary><pre>'+esc(text)+'</pre></details>';
    }).join('')+'</section><p class="csa-note">源码事实与静态依赖示意。布局为简化示意，不代表实际调度、耗时或通信带宽。</p>';host.scrollTop=0;
  }
  function render(){
    controller?.destroy();controller=null;const stage=root.querySelector('.csa-stage'),view=state.views[state.tab];stage.replaceChildren();
    root.querySelector('.csa-focusbar').innerHTML=state.tab==='graph'?'':Object.entries(model.topics).map(([key,label])=>'<button class="btn btn-ghost btn-sm '+(view.topic===key?'is-selected':'')+'" data-csa-focus="'+key+'" aria-pressed="'+(view.topic===key)+'">'+label+'</button>').join('');
    if(state.tab==='graph'){
      stage.innerHTML='<div class="csa-unadapted"><h2>CSA Graph 尚未适配</h2><p>本阶段先探索 Assignment 与 Pipeline。可切回查看 Rank 数据分布和通信依赖。</p><p>当前文件：decode_csa.py</p></div>';
      root.querySelector('.csa-caption').textContent='CSA 场景 · Graph';root.querySelector('.csa-footer').hidden=true;
    }else{
      root.querySelector('.csa-footer').hidden=false;
      graph=view.scope==='local'?(state.tab==='assignment'?anatomy(view.topic):localGraph(view.topic)):state.tab==='assignment'?assignment(view.topic):pipeline();
      root.querySelector('.csa-caption').innerHTML='<b>'+esc(view.scope==='local'?(state.tab==='assignment'?'Tensor / Cache Anatomy · ':'Task / Communication Dependency · ')+model.topics[view.topic]:titles[state.tab])+'</b><span>'+esc(state.tab==='assignment'?'T 为本地 token 行；数据块仅示意归属与分组。':'主要静态依赖 · 分支不保证同时执行 · 拖拽平移 / Ctrl 或 ⌘ 滚轮缩放')+'</span>';
      let selectedId=state.selection?'r'+state.selection.rank+':'+state.selection.object:null;
      if(selectedId&&!graph.nodes.some(n=>n.id===selectedId)){let mapped=state.selection.object;for(let i=0;i<4&&!graph.nodes.some(n=>n.id==='r'+state.selection.rank+':'+mapped);i++)mapped=alias[mapped];selectedId=mapped?'r'+state.selection.rank+':'+mapped:null;}
      if(state.selection?.object==='rank')selectedId='rank-'+state.selection.rank;
      controller=global.PtoModelGraphvizPattern.renderController(stage,graph,{fitMode:'full',viewportPadding:24,reportOverlays:false,overlays:{evidence:false,edgeTags:false,edgeHover:false},activeNodeId:selectedId,initialTransform:view.camera||undefined,
        onSelect:({nodeId,source})=>{if(source==='init')return;const n=graph.nodes.find(x=>x.id===nodeId);if(n){state.selection={object:n.object,rank:n.rank};state.rank=n.rank;}else if(nodeId.startsWith('rank-')){state.rank=Number(nodeId.split('-')[1]);state.selection={object:'rank',rank:state.rank};}details();chrome();},
        onClearSelection:()=>{state.selection=null;details();chrome();}});
    }
    if(controller){controller.svg.ptoModelGraphviz.metadata.clusterEntries.forEach(({el,cluster})=>{if(cluster.id.startsWith('rank-')){const label=el.querySelector('.pto-model-graphviz-cluster-label');if(label){label.style.pointerEvents='auto';label.style.cursor='pointer';}}});controller.svg.ptoModelGraphviz.metadata.edgeEntries.forEach(({el,edge})=>{if(edge.type==='communication')el.style.stroke='var(--primary)';});}
    details();chrome();
  }
  function open(button){if(!root)mount();if(state.open)return;trigger=button;state.open=true;root.hidden=false;const area=document.getElementById('ideWorkarea');area.dataset.visualization2='open';previousInert=[...area.children].filter(el=>el!==root).map(el=>[el,el.inert]);previousInert.forEach(([el])=>el.inert=true);render();root.querySelector('[data-csa-tab="'+state.tab+'"]').focus();}
  function close(){if(!state.open)return;saveCamera();state.open=false;root.hidden=true;delete document.getElementById('ideWorkarea').dataset.visualization2;previousInert.forEach(([el,inert])=>el.inert=inert);previousInert=[];trigger?.isConnected&&trigger.focus({preventScroll:true});}
  document.addEventListener('click',e=>{const b=e.target.closest('[data-visualization2-open]');if(b&&global.PtoComputeIDE.getScene()==='csa')open(b);});
  global.PtoCsaExplorer=Object.freeze({open,close,getState:()=>structuredClone(state)});
})(window);
