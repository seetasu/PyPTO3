/* Task compute graph. Counts are source-level invocations, never runtime cost. */
(function(global){
'use strict';
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function createModel({anatomy,source,preLaunch}){
 const lines=source.split('\n'),line=s=>lines.findIndex(l=>l.includes(s))+1;
 const tasks=anatomy.tasks.map(t=>{
  const P=t.pages.length,S=t.stacks.length,T=S+preLaunch;
  const n=(id,title,role,count,unit,formula,needle,detail)=>({id,title,role,count,unit,formula,line:line(needle),detail});
  const nodes=[
   n('q','Q Tile','tensor',1,'AIC / Task','1','q_tile = pl.load',`BF16 · Mat · ${t.q.shape.join('×')} / valid ${t.q.validShape.join('×')}`),
   n('k','K Cache Pages','tensor',P,'AIC / Task','ceil(seq_len / BLOCK_SIZE)','qk_k_page = pl.load',`BF16 · GM → Mat · Page ${anatomy.pageTokens}×${anatomy.headDim} · physical ID unknown`),
   n('qk','QK Matmul','operation',P,'AIC / Task','page_count','qk_score_page = pl.matmul','每个有效逻辑 Page 一次 Matmul；尾页仍加载完整 Page。'),
   n('scores','Scores','tensor',P,'AIC stores / Task','page_count','score_transfer,',`score_transfer · FP32 · GM · slot tile ${anatomy.rowTile}×${anatomy.stackTokens}`),
   n('softmax','Online Softmax','operation',S,'每 AIV Lane / Task','stack_count; tick < stack_count','score_scaled = pl.tile.muls','Scale → valid_cols Mask → online max/sum → BF16 Probability。两条 Lane 分别处理 2 / 3 Q Heads。'),
   n('probability','Probability','tensor',S,'每 AIV Lane stores / Task','stack_count','probability_valid = pl.set_validshape',`probability_transfer · BF16 · GM · slot ${anatomy.rowTile}×${anatomy.stackTokens}`),
   n('v','V Cache Pages','tensor',P,'AIC / Task','page_count','value_cache_bsnd,',`BF16 · GM → Mat · ${anatomy.pageTokens}×${anatomy.headDim} / Page`),
   n('pv','PV Matmul','operation',P,'AIC / Task','Σ pages per Stack = page_count','pv0 = pl.matmul','每 Stack 首 Page 为 Matmul，其余 Page 为 Matmul Accumulate。'),
   n('partial','Partial Context','tensor',S,'AIC stores / Task','stack_count','pl.store(pv3,',`pv_transfer · FP32 · GM · ${anatomy.rowTile}×${anatomy.headDim}`),
   n('accumulate','Context Accumulate','operation',S,'每 AIV Lane / Task','stack_count; tick >= PRE_LAUNCH','pl.add(pl.row_expand_mul(o_iter','o_next = o_previous × delayed_rescale + partial_context。'),
   n('normalize','Output Normalize','operation',1,'每 AIV Lane / Task','1, after loop','ctx = pl.cast','完成所有 Stack 后 o / l_sum，再 Cast BF16。'),
   n('output','BF16 Output','tensor',1,'每 AIV Lane store / Task','1','                ctx,',`out2d · GM · valid ${t.output.shape.join('×')} · 两条 Lane 分别写 2 / 3 行`),
   n('m','m · Max','state',S,'每 AIV Lane / Task','next_m = max(local_m, m_iter)','next_m = pl.maximum','FP32 · Vec · per Lane 8×1 / valid 2或3×1；m[-1] = −3e38；m[s] 参与下一 Stack 的 online max。'),
   n('l','l_sum','state',S,'每 AIV Lane / Task','l[s] = exp(m[s−1]−m[s]) × l[s−1] + sum(p[s])','next_l = pl.add','FP32 · Vec · per Lane 8×1 / valid 2或3×1；l[-1] = 0；最终 l_sum 用于 Output Normalize。'),
   n('rescale','Rescale FIFO','state',T,'每 AIV Lane / Task','pending0 ← pending1; pending1 ← rescale_after; ticks = stacks + PRE_LAUNCH','pending0_after = pl.tile.move','初始化 (1,1)。Stack s 的 rescale 经两拍 FIFO，在 consume_stack=s 时由 pending0 读取；drain tick 保持 m/l，rescale_after=pending1。'),
   n('o','o · Context','state',S,'每 AIV Lane / Task','o[s] = o[s−1] × rescale[s] + pv[s]','o_after = pl.yield_','FP32 · Vec · per Lane 8×128 / valid 2或3×128；o[-1] = 0；跨 Stack 累积 FP32 Context，最终归一化。')];
  const task={...t,nodes,pageCount:P,stackCount:S,tickCount:T,preLaunch};
  task.graph=semanticModel(task);return task;
 });return {tasks,source,anatomy};
}

// Adapt Task facts to the same semantic schema used by the outer graph. All
// composite subgraphs, their ports and source evidence come from the source model.
const aliases={k:'k-cache',v:'v-cache',accumulate:'context-acc',normalize:'output-norm'};
const semanticId=id=>aliases[id]||id;
const localId=id=>Object.keys(aliases).find(k=>aliases[k]===id)||id;
function semanticModel(task){
 const base=global.PTO_PYPA_SEMANTIC_GRAPH,copy=v=>JSON.parse(JSON.stringify(v));
 const ids=['q','k-cache','qk','scores','scale-mask','softmax','probability','v-cache','pv','context-acc','output-norm','output'];
 const nodes=ids.map(id=>copy(base.nodes.find(n=>n.id===id)));
 // PV's output tile is also a first-class root tensor; its subgraph retains the
 // Mat-memory partial. No physical page IDs or addresses are synthesized.
 const partial=copy(base.nodes.find(n=>n.id==='pv').composite.subgraph.nodes.find(n=>n.id==='partial'));
 partial.id='partial';partial.name='Partial Context · pv_transfer';
 partial.data={role:'intermediate',granularity:'stack',entity:'context.partial',symbol:'pv_transfer',dtype:'FP32',shape:{allocated:['TRANSFER_ROWS','HEAD_DIM']},storage:{space:'GM'}};
 partial.provenance=[{kind:'resolved',origin:'source',fields:Object.keys(partial.data).map(k=>'data.'+k),note:'FP32 producer and transfer-row addressing in paged_attention_pypto.py.'}];
 nodes.splice(nodes.findIndex(n=>n.id==='context-acc'),0,partial);
 for(const id of ['m','l','rescale','o']){
  const fact=task.nodes.find(n=>n.id===id);
  const state={dtype:'FP32',granularity:'stack',initial:id==='m'?'−3e38':id==='rescale'?'pending0 = 1; pending1 = 1':'0',update:fact.formula,feedback:fact.detail};
  nodes.push({id,type:'state',name:fact.title,description:fact.detail,source:{file:'paged_attention_pypto.py',ranges:[{start:fact.line,end:fact.line}],symbols:[id==='l'?'l_sum':id]},ports:{inputs:{default:{}},outputs:{default:{}}},state,provenance:[{kind:'resolved',origin:'source',fields:Object.keys(state).map(k=>'state.'+k),note:'Loop-carried state from pl.range init_values and pl.yield_.'}]});
 }
 const byId=new Map(nodes.map(n=>[n.id,n]));
 // Extra semantic state ports resolve through the ordinary composite mechanism.
 const bind=(id,side,name,node,port='default')=>{const n=byId.get(id);n.ports[side][name]={};n.composite.portMappings[side][name]={node,port,relation:'control'};};
 bind('softmax','inputs','previous','previous');
 bind('output-norm','inputs','final','normalize');
 const edges=copy(base.edges.filter(e=>ids.includes(e.s)&&ids.includes(e.t)&&!['context-acc','output-norm'].includes(e.t)&&!['softmax>context-acc'].includes(e.s+'>'+e.t)));
 const edge=(s,t,label,type='data',sourcePort='default',targetPort='default',id)=>edges.push({id:id||`${type}:${s}:${t}`,s,t,label,type,sourcePort,targetPort});
 edge('pv','partial','pv_transfer','data','partial');edge('partial','context-acc','Partial Context','data','default','partial');
 edge('softmax','m','next_m','state','state');edge('m','softmax','previous Stack · m','feedback','default','previous');
 edge('softmax','l','next_l','state','state');edge('l','softmax','previous Stack · l_sum','feedback','default','previous');
 edge('softmax','rescale','rescale_after','state','state');edge('rescale','rescale','FIFO shift / Tick','feedback');
 edge('rescale','context-acc','pending0 · delayed rescale','state','default','state');
 edge('context-acc','o','o_next','state','Context');edge('o','context-acc','previous Stack · o','feedback','default','state');
 edge('l','output-norm','final l_sum','data','default','final');edge('o','output-norm','final o','data','default','final');
 const region=(id,name,members)=>({id,type:'region',name,description:name,members,source:{file:'paged_attention_pypto.py',ranges:[{start:759,end:873}],symbols:[]},provenance:[{kind:'resolved',origin:'source',fields:['members'],note:'Task-local semantic regions; not runtime timing.'}]});
 const graph={schemaVersion:2,id:`attention-task-${task.id}`,sourceFiles:base.sourceFiles,constants:base.constants,runtimeDims:base.runtimeDims,nodes,edges,
  groups:[region('task-inputs','Task Inputs',['q','k-cache','v-cache']),region('task-compute',`Stack Loop ×${task.stackCount} · QK / PV Page Loop ×${task.pageCount}`,['qk','scores','scale-mask','softmax','probability','pv','partial','context-acc','m','l','rescale','o']),region('task-output','Task Output',['output-norm','output'])],
  synchronizations:copy(base.synchronizations)};
 // The root Q is the current Task tile, not the full Q projection tensor.
 const q=byId.get('q');q.name='Q Tile';q.data.shape={allocated:task.q.shape,valid:task.q.validShape};
 return graph;
}
function mount(host,{model,onTask,onFocus,onView}){
 host.classList.add('viz2-compute');
 host.innerHTML='<div class="viz2-anatomy-taskbar" data-compute-toolbar></div><div class="viz2-compute-canvas"></div>';
 const toolbar=host.querySelector('[data-compute-toolbar]');
 toolbar.innerHTML=`<div class="segmented-control segmented-control-muted" role="group" aria-label="Graph 显示模式"><button class="btn btn-sm" data-compute-mode="structure">Structure</button><button class="btn btn-sm" data-compute-mode="weight">Execution Weight</button></div><label>Task <select data-compute-task aria-label="Graph Task">${model.tasks.map(t=>`<option value="${t.id}">Task ${t.id} · B${t.block} / Batch ${t.batch} / KV ${t.kvHead}</option>`).join('')}</select></label><span data-compute-context></span><div data-compute-stacks role="group" aria-label="Stack 焦点"></div><div data-compute-pages role="group" aria-label="有效逻辑 Pages"></div>`;
 let task=model.tasks[85]||model.tasks[0],focus,view={},active=false,hover=null;
 const canvas=host.querySelector('.viz2-compute-canvas');
 const stackIndex=()=>Number.isInteger(focus?.stackIndex)?focus.stackIndex:['stack','stage','event'].includes(focus?.kind)?focus.index:focus?.kind==='page'?task.pages[focus.index]?.stack:null;
 const currentNode=()=>focus?.node?semanticId(focus.node):focus?.kind==='stage'?semanticId(focus.stage):focus?.kind==='output'?'output':null;
 const edgeFor=(id,graph)=>graph.edges.find(e=>e.id===id||e.semantic?.id===id);
 function executionData(graph){
  const data=global.PtoPagedAttentionExecution.derive(task.graph,global.PtoPagedAttentionIntro.model,{task:task.id,seqLen:task.seqLen});
  if(!data)return null;
  const entry=(count,unit,formula,source)=>({count,unit,label:`×${count} · ${unit}`,formula,source});
  task.nodes.forEach(n=>{data.nodes[semanticId(n.id)]=entry(n.count,n.unit,n.formula,{file:'paged_attention_pypto.py',ranges:[[n.line,n.line]]});});
  data.nodes['qk/q-tile']=entry(1,'AIC load / Task','once before tick loop');
  data.nodes['qk/matmul']=entry(task.pageCount,'AIC / Task','page_count');
  data.nodes['pv/matmul']=entry(task.pageCount,'Page 贡献 / AIC / Task',`matmul = ${task.stackCount}; matmul_acc = ${task.pageCount-task.stackCount}`);
  data.nodes['scale-mask']=entry(task.stackCount,'每 AIV Lane / Task','stack_count');
  // Descendant counts follow source loop granularity. PV Matmul is a semantic
  // page contribution: distinguish first-page matmul from matmul_acc in Inspector.
  graph.nodes.forEach(n=>{
   if(data.nodes[n.id]||!n.parentId)return;
   const root=n.id.split('/')[0];
   const count=['qk','pv'].includes(root)?(n.semantic?.data?.granularity==='stack'?task.stackCount:task.pageCount):['output-norm'].includes(root)?1:task.stackCount;
   data.nodes[n.id]=entry(count,['qk','pv'].includes(root)?'AIC / Task':'每 AIV Lane / Task','source loop granularity',n.source);
  });
  graph.edges.forEach(e=>{if(e.type!=='data')return;const source=e.semanticSource||e.s,root=source.split('/')[0];let count=['q','output-norm','output','l','o'].includes(root)?1:['qk','k-cache','v-cache'].includes(root)?task.pageCount:task.stackCount;if(root==='pv')count=e.internal?task.pageCount:task.stackCount;data.edges[e.id]={...entry(count,'逻辑传输 / Task','producer loop'),width:global.PtoPagedAttentionExecution.width(count)};});
  return data;
 }
 function resolveFocus(graph,selection,v){
  const f=hover||focus,id=f?.node?semanticId(f.node):hover?null:currentNode();
  const edge=f?.edge?edgeFor(f.edge,graph):null;
  const sync=task.graph.synchronizations.find(s=>s.id===f?.syncId||s.name===f?.event?.replace('_EVENT',''));
  const related=new Set();
  if(id){related.add(id);graph.edges.filter(e=>[e.s,e.t,e.semanticSource,e.semanticTarget].includes(id)).forEach(e=>{related.add(e.s);related.add(e.t);});}
  if(edge){related.add(edge.s);related.add(edge.t);}
  if(sync){related.add(sync.producer.boundary||sync.producer.node);related.add(sync.consumer.boundary||sync.consumer.node);}
  if(!id&&!edge&&!sync&&stackIndex()!=null)graph.nodes.filter(n=>!['output-norm','output'].includes(n.id.split('/')[0])||stackIndex()===task.stackCount-1).forEach(n=>related.add(n.id));
  return {nodeId:graph.nodes.some(n=>n.id===id)?id:graph.nodes.find(n=>n.id===id?.split('/')[0])?.id,related:[...related],edgeId:edge?.id,kind:'task'};
 }
 function description(node,sync,graph,edgeId){
  const e=edgeFor(edgeId||focus?.edge,graph),s=stackIndex(),stack=task.stacks[s],page=focus?.kind==='page'?task.pages[focus.index]:null;
  let title='Task '+task.id,rows=[],line=759;
  if(sync){title=sync.name;rows=[['Buffer',sync.buffer],['生产者',`${sync.producer.coreType} · ${sync.producer.pipe}`],['消费者',`${sync.consumer.coreType} · ${sync.consumer.pipe}`],['静态次数',`${task.stackCount} / Task；${sync.name==='SOFTMAX_READY'?'每 AIV Lane set；AIC wait':'AIC set；每 AIV Lane wait'}`],['语义',sync.explanation]];line=sync.producer.source?.ranges?.[0]?.start||759;}
  else if(node){
   const root=node.id.split('/')[0],n=task.nodes.find(n=>semanticId(n.id)===root),data=executionData(graph),weight=data?.nodes[node.id];
   title=node.title;line=node.sourceLine||n?.line||759;
   rows=[['类型',node.type.toUpperCase()],['静态执行次数',weight?.label||'未推导'],['公式 / 循环边界',weight?.formula||n?.formula||'源码分支'],['语义',node.detail],...(n?[['Task 形状 / 内存',n.detail]]:[]),['生产者',graph.edges.filter(e=>e.t===node.id).map(e=>e.s).join(', ')||'Task input'],['消费者',graph.edges.filter(e=>e.s===node.id).map(e=>e.t).join(', ')||'Task output']];
   if(root==='pv')rows.push(['PV 统计口径',`每 Task / AIC：首 Page matmul ×${task.stackCount}；matmul_acc ×${task.pageCount-task.stackCount}；合计 ${task.pageCount} 次 Page 贡献`]);
  }else if(e){title=e.label||`${e.s} → ${e.t}`;line=graph.nodes.find(n=>n.id===e.t)?.sourceLine||759;rows=[['依赖类型',e.type],['生产者 → 消费者',`${e.s} → ${e.t}`],['状态关系',e.type==='feedback'?`${s===0?'初始化 → S0':s!=null?`S${s-1} → S${s}`:'S(s−1) → S(s)'}；m/l/o 初始 −3e38 / 0 / 0；rescale FIFO 初始 (1,1)`:'按源码数据流与端口映射'],['次数口径','逻辑传输次数；不推断 Bytes 或实测耗时']];}
  else {title=page?`Page ${page.index}`:stack?`Stack ${s}`:title;rows=[['Task',`Block ${task.block} / Batch ${task.batch} / KV Head ${task.kvHead}`],['有效 Pages',stack?stack.pages.map(p=>`P${p} (${task.pages[p].tokens}/${task.pageTokens})`).join(', '):task.pageCount],['有效 Tokens',page?`${page.tokens}/${task.pageTokens}`:stack?`${stack.tokens}/${task.stackTokens}`:task.seqLen],['计算实例',stack?`QK/PV ${stack.pages.length} Pages；Softmax/Accumulate 各 Lane 1 次`:`${task.stackCount} Stacks / ${task.tickCount} logical ticks`],['物理 Page','未提供实际 block_table，未知']];}
  return {title,line,rows};
 }
 const shared=global.PtoVisualization2Graph.mount(canvas,{onView:patch=>{if(patch.overlays){const {execution,...overlays}=patch.overlays;onView({intraOverlays:overlays,intraMode:execution?'weight':'structure'});return;}const mapped={};for(const [k,v] of Object.entries(patch))mapped[{camera:'intraCamera',expanded:'intraExpanded',overlays:'intraOverlays',inspectorOpen:'intraInspectorOpen'}[k]||k]=v;onView(mapped);},adapter:{
  getModel:()=>task.graph,resolveFocus,executionData,
  stageLabel:g=>`<b>${esc(g.label)}</b>`,
  context:()=>view.intraMode==='weight'?'Static Execution Weight · AIC / 每 AIV Lane 分别统计 · 非耗时':'Data / State / Feedback / READY · 源码静态计算关系',
  hasInspector:()=>!!focus,
  inspector(node,sync,graph){const d=description(node,sync,graph);const semantic=node?global.PTO_COMPUTATION_PRESENTATION.inspector(node,{model:task.graph,qualifiedId:node.id}):[];return {title:d.title,html:`<dl>${d.rows.map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')}</dl>${semantic.map(section=>`<section><h3>${esc(section.title)}</h3><dl>${section.rows.map(r=>`<dt>${esc(r.label)}</dt><dd>${esc(r.value)}</dd>`).join('')}</dl></section>`).join('')}<button class="btn btn-ghost btn-sm" data-compute-source="${d.line}">PyPTO 源码 :${d.line}</button><pre data-compute-code hidden></pre>`};},
  onNode(id){const root=localId(id.split('/')[0]),stage=['qk','softmax','pv','accumulate'].includes(root)?root:null;onFocus({kind:stage?'stage':root==='output'?'output':'graph',node:id,index:stackIndex()??0,stackIndex:stackIndex(),...(stage?{stage}:{})});},
  onEdge(id,graph){onFocus({kind:'graph-edge',edge:id,index:stackIndex()??0,stackIndex:stackIndex()});},
  onSync(id){const sync=task.graph.synchronizations.find(s=>s.id===id);onFocus({kind:'event',syncId:id,event:sync.name+'_EVENT',index:stackIndex()??0,stackIndex:stackIndex()});},
  onHover(node,edge,syncId){hover=node?{node}:edge?{edge}:syncId?{syncId}:null;},
  tooltip(id,edge,syncId,graph){const d=description(graph.nodes.find(n=>n.id===id),task.graph.synchronizations.find(s=>s.id===syncId),graph,edge);return `${d.title} · ${d.rows.slice(0,3).map(r=>r.join(': ')).join(' · ')} · source:${d.line}`;},
  decorate(scene,nodes,graph,data){scene.dataset.executionWeight=String(!!data);nodes.forEach(n=>{const el=scene.querySelector(`[data-pypa-node="${n.id}"]`),entry=data?.nodes[n.id];if(el)el.style.setProperty('--execution-weight',entry?String(Math.min(1,entry.count/Math.max(task.pageCount,task.tickCount))):'0');});}
 }});
 host.addEventListener('change',e=>{if(e.target.matches('[data-compute-task]'))onTask(Number(e.target.value));});
 toolbar.addEventListener('click',e=>{const b=e.target.closest('[data-compute-mode],[data-compute-stack],[data-compute-page]');if(!b)return;if(b.dataset.computeMode)onView({intraMode:b.dataset.computeMode});else if(b.dataset.computeStack!==undefined)onFocus({kind:'stack',index:Number(b.dataset.computeStack)});else onFocus({kind:'page',index:Number(b.dataset.computePage)});});
 // Shared controller stops legacy document-level graph handlers; source actions
 // are handled in capture so their Inspector remains local to this Task.
 canvas.addEventListener('click',e=>{const b=e.target.closest('[data-compute-source]');if(!b)return;e.stopPropagation();const line=Number(b.dataset.computeSource),code=canvas.querySelector('[data-compute-code]');code.hidden=false;code.textContent=model.source.split('\n').slice(Math.max(0,line-4),line+20).map((v,i)=>`${Math.max(1,line-3)+i}  ${v}`).join('\n');},true);
 return {update(id,f,v,{active:next}={}){
  if(model.tasks[id])task=model.tasks[id];focus=f;view=v;active=!!next;hover=null;
  if(active){toolbar.querySelector('[data-compute-task]').value=task.id;toolbar.querySelector('[data-compute-context]').textContent=`Batch ${task.batch} · KV ${task.kvHead} · ${task.seqLen} tokens · ${task.pageCount} Pages / ${task.stackCount} Stacks / ${task.tickCount} logical ticks`;
   const s=stackIndex();toolbar.querySelector('[data-compute-stacks]').innerHTML=task.stacks.map(stack=>`<button class="btn btn-ghost btn-sm ${s===stack.index?'is-selected':''}" data-compute-stack="${stack.index}">S${stack.index}</button>`).join('');
   toolbar.querySelector('[data-compute-pages]').innerHTML=task.stacks[s]?.pages.map(p=>`<button class="btn btn-ghost btn-sm ${f?.kind==='page'&&f.index===p?'is-selected':''}" data-compute-page="${p}">P${p} · ${task.pages[p].tokens}/${task.pageTokens}</button>`).join('')||'';
   toolbar.querySelectorAll('[data-compute-mode]').forEach(b=>{const selected=b.dataset.computeMode===(v.intraMode||'structure');b.classList.toggle('is-selected',selected);b.setAttribute('aria-pressed',String(selected));});
  }
  const sync=task.graph.synchronizations.find(s=>s.id===f?.syncId||s.name===f?.event?.replace('_EVENT',''));
  shared.update({kind:'task',task:task.id,block:task.block,batch:task.batch,kvHead:task.kvHead,subobject:f},{camera:v.intraCamera,nodeId:currentNode(),syncId:sync?.id,expanded:v.intraExpanded||[],overlays:{control:true,...v.intraOverlays,execution:v.intraMode==='weight'},inspectorOpen:v.intraInspectorOpen},{active});
 },hideTooltip:shared.hideTooltip,getTransitionRect:shared.getTransitionRect};
}
global.PtoVisualization2GraphIntra=Object.freeze({createModel,mount});
})(window);
