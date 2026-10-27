/* CSA scene adapter. The shared graph pattern owns node rendering, selection and camera input. */
(function (global) {
  'use strict';
  const model=global.PtoCsaModel, objects=model.objects;
  const state={open:false,tab:'assignment',selection:null,rank:0,qkMode:false,
    views:{assignment:{scope:'overall',topic:'gather',camera:null},pipeline:{scope:'overall',topic:'gather',camera:null},graph:{scope:'overall',camera:null,expanded:[],overlays:{data:true},nodeId:null,edgeId:null,inspectorOpen:false}}};
  const tabs=['assignment','pipeline','graph'];
  let root, controller, semanticController, qkController, trigger, previousInert=[], graph;
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
    '<div class="csa-toolbar"><div class="segmented-control segmented-control-muted" role="group" aria-label="CSA 观察层次" data-csa-scopegroup><button class="btn btn-sm" data-csa-scope="overall">整体</button><button class="btn btn-sm" data-csa-scope="local">局部</button></div><nav data-csa-breadcrumb aria-label="当前对象"></nav><button class="btn btn-ghost btn-sm" data-csa-return hidden>← 返回整体</button><button class="btn btn-sm" data-csa-qk hidden>QK Work Assignment</button></div>'+
    '<div class="csa-focusbar" aria-label="关键路径焦点"></div>'+
    '<div class="csa-workspace" id="csa-panel" role="tabpanel"><div class="csa-canvas-column"><div class="csa-caption"></div><div class="csa-stage pto-model-graphviz-pattern-page" role="group" aria-label="CSA 数据与任务关系，拖拽平移，按住 Command 或 Ctrl 滚轮缩放"></div><footer class="csa-footer"><div class="csa-legend"><span>中性矩形 · Tensor / Cache</span><span>彩色模块 · 计算 / 通信 / 同步</span><span>实线 · 数据</span><span class="csa-comm-legend">蓝线 · 跨 Rank</span><span>虚线 · 控制 / 同步</span></div><button class="btn btn-ghost btn-sm" data-csa-fit>全览</button><button class="btn btn-ghost btn-sm" data-csa-read>阅读</button><button class="btn btn-ghost btn-sm" data-csa-zoom="0.8" aria-label="缩小 CSA">−</button><button class="btn btn-ghost btn-sm" data-csa-zoom="1.25" aria-label="放大 CSA">＋</button></footer></div>'+
    '<aside class="pto-ide-frame__pane csa-inspector" aria-label="CSA 对象详情"><header class="pto-ide-frame__pane-header"><span class="pto-ide-frame__pane-title">对象详情</span><button class="btn btn-ghost btn-sm" data-csa-clear>清除选择</button></header><div class="pto-ide-frame__pane-body csa-detail"></div></aside></div>'+
    '<div class="csa-qk-host" hidden></div>';
    document.getElementById('ideWorkarea').append(root);
    root.addEventListener('click',event=>{
      const b=event.target.closest('button');if(!b)return;
      if(b.hasAttribute('data-csa-back'))return close();
      if(b.hasAttribute('data-csa-qk')){state.qkMode=!state.qkMode;render();return;}
      if(b.dataset.csaTab){saveCamera();state.tab=b.dataset.csaTab;if(state.tab==='assignment'&&state.selection)state.views.assignment.topic=selectedTopic(state.views.assignment.topic);render();return;}
      if(b.dataset.csaScope){saveCamera();const v=state.views[state.tab];if(state.tab==='graph'){if((v.expanded.length?'local':'overall')===b.dataset.csaScope)return;if(b.dataset.csaScope==='local'&&!v.bookmarkedNodeId&&semanticController){semanticController.toggleExpanded(v.nodeId?.split('/')[0]||'input',true);return;}setGraphScope(b.dataset.csaScope);render();return;}if(v.scope===b.dataset.csaScope)return;v.scope=b.dataset.csaScope;if(state.selection)v.topic=selectedTopic(v.topic);v.camera=null;render();return;}
      if(b.hasAttribute('data-csa-return')){if(state.tab==='graph')setGraphScope('overall');else{state.views[state.tab].scope='overall';state.views[state.tab].camera=null;}render();return;}
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
  function setGraphScope(scope){
    const v=state.views.graph;
    if(scope==='overall'){
      if(v.expanded.length&&v.camera)v.localCamera={root:v.expanded[0].split('/')[0],camera:v.camera};
      if(v.nodeId?.includes('/')){v.bookmarkedNodeId=v.nodeId;v.nodeId=v.nodeId.split('/')[0];}
      v.expanded=[];
    }else{
      const target=v.bookmarkedNodeId||v.nodeId||v.expanded[0]||'input';
      const path=target.split('/');
      v.expanded=[...new Set([...v.expanded,...path.slice(0,-1).map((_,i)=>path.slice(0,i+1).join('/')),path[0]])];
      if(v.bookmarkedNodeId){v.nodeId=v.bookmarkedNodeId;v.bookmarkedNodeId=null;}
    }
    v.scope=v.expanded.length?'local':'overall';
    v.camera=scope==='local'&&v.localCamera?.root===v.expanded[0]?.split('/')[0]?v.localCamera.camera:null;
  }
  function saveCamera(){if(controller&&state.tab!=='graph')state.views[state.tab].camera=controller.getTransform();}
  function chrome(){
    root.querySelectorAll('[data-csa-tab]').forEach(b=>{const active=b.dataset.csaTab===state.tab;b.classList.toggle('is-selected',active);b.setAttribute('aria-selected',String(active));b.tabIndex=active?0:-1;});
    root.querySelector('#csa-panel').setAttribute('aria-labelledby','csa-tab-'+state.tab);
    const qk=state.tab==='assignment'&&state.qkMode;
    const scope=state.tab==='graph'?(state.views.graph.expanded.length?'local':'overall'):state.views[state.tab].scope;
    root.querySelectorAll('[data-csa-scope]').forEach(b=>{const active=b.dataset.csaScope===scope;b.classList.toggle('is-selected',active);b.setAttribute('aria-pressed',String(active));});
    const scopeGroup=root.querySelector('[data-csa-scopegroup]');if(scopeGroup)scopeGroup.hidden=qk;
    const focusbar=root.querySelector('.csa-focusbar');if(focusbar)focusbar.hidden=qk;
    root.querySelector('[data-csa-return]').hidden=qk||scope!=='local';
    const qkButton=root.querySelector('[data-csa-qk]');
    if(qkButton){qkButton.hidden=state.tab!=='assignment';qkButton.classList.toggle('is-selected',qk);qkButton.textContent=qk?'← 返回 Assignment':'QK Work Assignment';qkButton.setAttribute('aria-pressed',String(qk));}
    root.querySelector('[data-csa-breadcrumb]').textContent=qk?titles.assignment+' / QK Work Assignment':titles[state.tab]+(state.tab==='graph'?' / '+(scope==='local'?'局部':'整体'):' / '+(scope==='local'?model.topics[state.views[state.tab].topic]+' / 局部':'整体'))+(state.tab!=='graph'&&state.selection?' / Rank '+state.selection.rank+' / '+objects[state.selection.object].label:'');
    const drill=root.querySelector('[data-csa-drill]');if(drill)drill.disabled=state.tab==='graph';
  }
  // Jumping to a source line only reads correctly once the explorer releases the workarea,
  // so leave the local scene first and then scroll the editor to the range.
  function qkSource(file,a,z){
    const go=()=>{if(file==='decode_csa.py')global.PtoComputeIDE.showSource(a,z);else global.PtoComputeIDE.showSourceFile(file,a,z);};
    if(state.open){close();requestAnimationFrame(go);}else go();
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
    controller?.destroy();controller=null;semanticController?.destroy();semanticController=null;const stage=root.querySelector('.csa-stage'),view=state.views[state.tab];stage.replaceChildren();
    const qkHost=root.querySelector('.csa-qk-host'),qkActive=state.tab==='assignment'&&state.qkMode;
    root.querySelector('.csa-workspace').hidden=qkActive;qkHost.hidden=!qkActive;
    if(qkActive){
      // Local experiment: it owns only its own host and keeps its own selection state.
      if(qkController)qkController.render();else qkController=global.PtoCsaQkWork.create(qkHost,{onSource:qkSource});
      chrome();return;
    }
    root.querySelector('.csa-focusbar').innerHTML=state.tab==='graph'?'':Object.entries(model.topics).map(([key,label])=>'<button class="btn btn-ghost btn-sm '+(view.topic===key?'is-selected':'')+'" data-csa-focus="'+key+'" aria-pressed="'+(view.topic===key)+'">'+label+'</button>').join('');
    if(state.tab==='graph'){
      root.querySelector('.csa-footer').hidden=true;
      root.querySelector('.csa-inspector').hidden=true;
      root.querySelector('.csa-caption').innerHTML='<b>CSA Compute Graph</b><span>Stage 是语义分组；点击 Module 的 + 展开源码支持的内部关系。</span>';
      stage.classList.add('csa-semantic-stage');
      const semantic=global.PTO_CSA_SEMANTIC_GRAPH;
      semanticController=global.PtoVisualization2Graph.mount(stage,{onSelect:()=>{},onView:patch=>{
        if(patch.expanded){const opened=new Set(patch.expanded);patch.expanded=patch.expanded.filter(id=>id.split('/').slice(0,-1).every((_,i)=>opened.has(id.split('/').slice(0,i+1).join('/'))));}
        if(patch.nodeId)state.views.graph.edgeId=null;
        Object.assign(state.views.graph,patch);
        if(!semanticController)return;
        if(patch.expanded)state.views.graph.scope=patch.expanded.length?'local':'overall';
        semanticController.update(null,state.views.graph,{active:true});chrome();
      },adapter:{
        getModel:()=>semantic,executionData:()=>null,fitOnEnter:true,layoutOptions:{rootLayerGap:44},labelMinWidth:96,expandedScaleFloor:node=>node.parentId?.62:0,expandedXOffset:node=>node.id==='cache'?140:0,focusExpandedTop:true,stageLabel:g=>'<b>'+esc(g.label)+'</b>',
        context:()=> '源码静态数据、状态和 Task ID 依赖；Stage 不表示严格串行。',
        resolveFocus:(g,selection,v)=>({nodeId:v.nodeId,edgeId:v.edgeId,related:v.nodeId?[v.nodeId]:[],kind:'node'}),
        hasInspector:()=>!!(state.views.graph.nodeId||state.views.graph.edgeId),
        onNode:id=>{state.views.graph.nodeId=id;state.views.graph.bookmarkedNodeId=null;state.views.graph.edgeId=null;state.views.graph.inspectorOpen=true;semanticController.update(null,state.views.graph,{active:true});},
        onEdge:id=>{state.views.graph.edgeId=id;state.views.graph.nodeId=null;state.views.graph.bookmarkedNodeId=null;state.views.graph.inspectorOpen=true;semanticController.update(null,state.views.graph,{active:true});},
        onSync:()=>{},
        onSource:range=>{const [file,a,z]=range.split(':');global.PtoComputeIDE.showSourceFile(file,Number(a),Number(z));},
        inspector:(node,sync,g)=>{
          const selectedEdge=g.edges.find(edge=>edge.id===state.views.graph.edgeId);
          const semanticNode=node?.semantic||semantic.nodes.find(item=>item.id===node?.id?.split('/')[0]);
          const byId=new Map(g.nodes.map(item=>[item.id,item]));
          const label=id=>byId.get(id)?.title||id;
          const semanticLabel=id=>g.nodes.find(item=>item.id===id||item.id.endsWith('/'+id))?.title||id;
          const kind=edge=>edge.communication?'跨 Rank Communication':({data:'Data',state:'State',feedback:'跨 Decode Step State',control:'Control',computation:'Computation'}[edge.type]||edge.type);
          const connection=edge=>`${label(edge.s)} → ${label(edge.t)} · ${kind(edge)}${edge.label?' · '+edge.label:''}`;
          const section=(title,rows)=>rows.length?'<section><h3>'+esc(title)+'</h3><dl>'+rows.map(([key,value])=>'<dt>'+esc(key)+'</dt><dd>'+esc(value)+'</dd>').join('')+'</dl></section>':'';
          if(selectedEdge){
            const sourceNode=byId.get(selectedEdge.s)?.semantic,targetNode=byId.get(selectedEdge.t)?.semantic;
            const html=section('Dependency',[["Type",kind(selectedEdge)],["What",selectedEdge.label||'源码静态关系'],["Producer",label(selectedEdge.s)],["Consumer",label(selectedEdge.t)]])+
              section('Source',[["Producer",sourceNode?.source?.file+' : '+(sourceNode?.source?.ranges||[]).map(range=>range.start+'–'+range.end).join('、')],["Consumer",targetNode?.source?.file+' : '+(targetNode?.source?.ranges||[]).map(range=>range.start+'–'+range.end).join('、')]].filter(([,value])=>value&&!value.startsWith('undefined')));
            return {title:selectedEdge.label||'依赖',html};
          }
          const item=semanticNode,payload=item?.[item.type]||{},incoming=node?g.edges.filter(edge=>edge.t===node.id):[],outgoing=node?g.edges.filter(edge=>edge.s===node.id):[];
          const loc=item?.source?.ranges?.[0],file=item?.source?.file;
          const shape=payload.shape?.allocated?.length?'['+payload.shape.allocated.join(', ')+']':null;
          const facts=(item?.provenance||[]).filter(entry=>entry.kind==='fact');
          const resolved=(item?.provenance||[]).filter(entry=>entry.kind==='resolved');
          const rows=[['What',item?.description||node?.detail||'源码静态关系'],['Kind',item?.type||node?.type]];
          const io=section('Inputs / Outputs',[
            ['Inputs',incoming.length?incoming.map(connection).join('；'):'外部输入或当前展开层级未显示'],
            ['Outputs',outgoing.length?outgoing.map(connection).join('；'):'当前展开层级未显示下游'],
            ...(payload.producer?[['Producer',semanticLabel(payload.producer)]]:[]),...(payload.consumers?.length?[['Consumers',payload.consumers.map(semanticLabel).join('、')]]:[])
          ]);
          const metrics=section('Shape / Dtype / Storage',[
            ...(shape?[['Shape',shape]]:[]),...(payload.dtype?[['Dtype',payload.dtype]]:[]),
            ...(payload.storage?.space?[['Storage',payload.storage.space]]:[]),...(payload.symbol?[['Symbol',payload.symbol]]:[])
          ]);
          const dependencies=section('Dependencies',[...incoming,...outgoing].filter(edge=>['state','feedback','control'].includes(edge.type)||edge.communication).map(edge=>[kind(edge),connection(edge)]));
          const lifecycle=item?.type==='state'?section('State lifecycle',[
            ...(payload.initial?[['跨 Step / 调用前',payload.initial]]:[]),
            ...(payload.update?[['当前调用更新',payload.update]]:[]),
            ...(payload.feedback?[['后续 Decode Step',payload.feedback]]:[])
          ]):'';
          const sourceText=file==='decode_csa.py'?global.PTO_CSA_SOURCE.text:global.PTO_CSA_SOURCE.dependencies[file];
          const excerpt=loc&&sourceText?sourceText.split('\n').slice(loc.start-1,loc.end).map((line,index)=>`${loc.start+index}  ${line}`).join('\n'):'';
          const sourceHtml=loc?'<section><h3>Source</h3><button class="btn btn-ghost btn-sm" data-viz2-graph-source="'+esc(file)+':'+loc.start+':'+loc.end+'">'+esc(file)+' : '+loc.start+'–'+loc.end+' ↗</button>'+
            (excerpt?'<details><summary>查看本地源码行段</summary><pre>'+esc(excerpt)+'</pre></details>':'')+'</section>':'';
          const provenance=section('Evidence',[
            ...facts.map(entry=>['源码事实',entry.origin+(entry.fields.length?' · '+entry.fields.join('、'): '')]),
            ...resolved.map(entry=>['静态推导',entry.origin+(entry.note?' · '+entry.note:'')])
          ]);
          return {title:node?.title||item?.name||'CSA',html:section('What',rows)+io+metrics+lifecycle+dependencies+sourceHtml+provenance};
        },tooltip:(id,edgeId,syncId,g)=>id?(g.nodes.find(n=>n.id===id)?.detail||id):(g.edges.find(e=>e.id===edgeId)?.label||'依赖')
      }});
      semanticController.update(null,view,{active:true});
    }else{
      stage.classList.remove('csa-semantic-stage','viz2-graph');
      root.querySelector('.csa-inspector').hidden=false;
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
    if(state.tab!=='graph')details();chrome();
  }
  function open(button){if(!root)mount();if(state.open)return;trigger=button;state.open=true;root.hidden=false;const area=document.getElementById('ideWorkarea');area.dataset.visualization2='open';previousInert=[...area.children].filter(el=>el!==root).map(el=>[el,el.inert]);previousInert.forEach(([el])=>el.inert=true);render();root.querySelector('[data-csa-tab="'+state.tab+'"]').focus();}
  function close(){if(!state.open)return;saveCamera();semanticController?.destroy();semanticController=null;state.open=false;root.hidden=true;delete document.getElementById('ideWorkarea').dataset.visualization2;previousInert.forEach(([el,inert])=>el.inert=inert);previousInert=[];trigger?.isConnected&&trigger.focus({preventScroll:true});}
  document.addEventListener('click',e=>{const b=e.target.closest('[data-visualization2-open]');if(b&&global.PtoComputeIDE.getScene()==='csa')open(b);});
  global.PtoCsaExplorer=Object.freeze({open,close,getState:()=>structuredClone(state)});
})(window);
