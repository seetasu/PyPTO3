/* Task compute graph. Counts are source-level invocations, never runtime cost. */
(function(global){
'use strict';
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function createModel({anatomy,source,preLaunch}){
 const lines=source.split('\n'),line=s=>lines.findIndex(l=>l.includes(s))+1;
 const tasks=anatomy.tasks.map(t=>{
  const P=t.pages.length,S=t.stacks.length,T=S+preLaunch;
  const n=(id,title,role,x,y,count,unit,formula,needle,detail,ops=[])=>({id,title,role,x,y,w:170,h:66,count,unit,formula,line:line(needle),detail,ops});
  const nodes=[
   n('q','Q Tile','tensor',30,100,1,'AIC / Task','1','q_tile = pl.load',`BF16 · Mat · ${t.q.shape.join('×')} / valid ${t.q.validShape.join('×')}`),
   n('k','K Cache Pages','tensor',30,220,P,'AIC / Task','ceil(seq_len / BLOCK_SIZE)','qk_k_page = pl.load',`BF16 · GM → Mat · Page ${anatomy.pageTokens}×${anatomy.headDim} · physical ID unknown`),
   n('qk','QK Matmul','operation',260,150,P,'AIC / Task','page_count','qk_score_page = pl.matmul','每个有效逻辑 Page 一次 Matmul；尾页仍加载完整 Page。',[['Load K',P],['Transpose View',P],['Matmul',P],['Store Scores',P]]),
   n('scores','Scores','tensor',490,150,P,'AIC stores / Task','page_count','score_transfer,',`score_transfer · FP32 · GM · slot tile ${anatomy.rowTile}×${anatomy.stackTokens}`),
   n('softmax','Online Softmax','operation',720,150,S,'每 AIV Lane / Task','stack_count; tick < stack_count','score_scaled = pl.tile.muls','Scale → valid_cols Mask → online max/sum → BF16 Probability。两条 Lane 分别处理 2 / 3 Q Heads。',[['Load Scores',S],['Scale / Fillpad',S],['Row Max / Maximum',S],['Sub / Exp (rescale)',S],['Row Expand Sub / Exp',S],['Row Sum',S],['Mul / Add (l_sum)',S],['Cast BF16 / Store',S]]),
   n('probability','Probability','tensor',950,150,S,'每 AIV Lane stores / Task','stack_count','probability_valid = pl.set_validshape',`probability_transfer · BF16 · GM · slot ${anatomy.rowTile}×${anatomy.stackTokens}`),
   n('v','V Cache Pages','tensor',950,320,P,'AIC / Task','page_count','value_cache_bsnd,',`BF16 · GM → Mat · ${anatomy.pageTokens}×${anatomy.headDim} / Page`),
   n('pv','PV Matmul','operation',1180,150,P,'AIC / Task','Σ pages per Stack = page_count','pv0 = pl.matmul','每 Stack 首 Page 为 Matmul，其余 Page 为 Matmul Accumulate。',[['Load V / Probability',P],['Matmul',S],['Matmul Accumulate',P-S],['Store PV',S]]),
   n('partial','Partial Context','tensor',1410,150,S,'AIC stores / Task','stack_count','pl.store(pv3,',`pv_transfer · FP32 · GM · ${anatomy.rowTile}×${anatomy.headDim}`),
   n('accumulate','Context Accumulate','operation',1640,150,S,'每 AIV Lane / Task','stack_count; tick >= PRE_LAUNCH','pl.add(pl.row_expand_mul(o_iter','o_next = o_previous × delayed_rescale + partial_context。',[['Load PV',S],['Row Expand Mul',S],['Add',S]]),
   n('normalize','Output Normalize','operation',1870,150,1,'每 AIV Lane / Task','1, after loop','ctx = pl.cast','完成所有 Stack 后 o / l_sum，再 Cast BF16。',[['Reciprocal',1],['Row Expand Mul',1],['Cast BF16',1]]),
   n('output','BF16 Output','tensor',2100,150,1,'每 AIV Lane store / Task','1','                ctx,',`out2d · GM · valid ${t.output.shape.join('×')} · 两条 Lane 分别写 2 / 3 行`),
   n('m','m · Max','state',600,420,S,'每 AIV Lane / Task','next_m = max(local_m, m_iter)','next_m = pl.maximum','FP32 · Vec · per Lane 8×1 / valid 2或3×1；m[-1] = −3e38；m[s] 参与下一 Stack 的 online max。'),
   n('l','l_sum','state',850,420,S,'每 AIV Lane / Task','l[s] = exp(m[s−1]−m[s]) × l[s−1] + sum(p[s])','next_l = pl.add','FP32 · Vec · per Lane 8×1 / valid 2或3×1；l[-1] = 0；最终 l_sum 用于 Output Normalize。'),
   n('rescale','Rescale FIFO','state',1180,420,T,'每 AIV Lane / Task','pending0 ← pending1; pending1 ← rescale_after; ticks = stacks + PRE_LAUNCH','pending0_after = pl.tile.move','初始化 (1,1)。Stack s 的 rescale 经两拍 FIFO，在 consume_stack=s 时由 pending0 读取；drain tick 保持 m/l，rescale_after=pending1。',[['Move pending1 → pending0',T],['Move rescale_after → pending1',T]]),
   n('o','o · Context','state',1640,420,S,'每 AIV Lane / Task','o[s] = o[s−1] × rescale[s] + pv[s]','o_after = pl.yield_','FP32 · Vec · per Lane 8×128 / valid 2或3×128；o[-1] = 0；跨 Stack 累积 FP32 Context，最终归一化。')];
  // Two dataflow tiers leave the lower band for loop-carried state, not time.
  const placement={q:[30,120],k:[30,240],qk:[260,180],scores:[490,180],softmax:[720,180],probability:[950,180],v:[1180,340],pv:[950,340],partial:[720,340],accumulate:[490,340],normalize:[260,340],output:[30,340],m:[260,560],l:[490,560],rescale:[950,560],o:[720,560]};
  nodes.forEach(n=>{[n.x,n.y]=placement[n.id];});
  const edges=[];const e=(s,d,kind='data',label='',count=S)=>edges.push({id:`${kind}:${s}:${d}`,s,t:d,kind,label,count});
  [['q','qk'],['k','qk'],['qk','scores'],['scores','softmax'],['softmax','probability'],['probability','pv'],['v','pv'],['pv','partial'],['partial','accumulate'],['accumulate','normalize'],['normalize','output']].forEach(([a,b])=>e(a,b,'data','',['k','qk'].includes(a)?P:S));
  edges.find(e=>e.s==='q').count=1;edges.filter(e=>['probability','v'].includes(e.s)&&e.t==='pv').forEach(e=>e.count=P);edges.filter(e=>['accumulate','normalize'].includes(e.s)).forEach(e=>e.count=1);
  [['qk','softmax','QK_READY'],['softmax','pv','SOFTMAX_READY'],['pv','accumulate','PV_READY']].forEach(([a,b,label])=>e(a,b,'event',label));
  [['softmax','m'],['m','softmax'],['softmax','l'],['l','softmax'],['softmax','rescale'],['rescale','accumulate'],['accumulate','o'],['o','accumulate'],['l','normalize'],['o','normalize'],['rescale','rescale']].forEach(([a,b])=>e(a,b,'state',a===b?'FIFO shift / Tick':a==='m'||a==='l'||a==='o'?'previous → next Stack':''));
  edges.filter(e=>['l','o'].includes(e.s)&&e.t==='normalize').forEach(e=>{e.kind='data';e.count=1;e.label='final state after all Stacks';});
  return {...t,nodes,edges,pageCount:P,stackCount:S,tickCount:T,preLaunch};
 });return {tasks,source,anatomy};
}
function mount(host,{model,onTask,onFocus,onView}){
 host.classList.add('viz2-anatomy','viz2-compute');
 host.innerHTML=`<div class="viz2-anatomy-taskbar"><div class="segmented-control segmented-control-muted" role="group" aria-label="Graph 显示模式"><button class="btn btn-sm" data-compute-mode="structure">Structure</button><button class="btn btn-sm" data-compute-mode="weight">Execution Weight</button></div><label>Task <select data-compute-task aria-label="Graph Task"></select></label><span data-compute-context></span><div data-compute-stacks role="group" aria-label="Stack 焦点"></div><div data-compute-pages role="group" aria-label="有效逻辑 Pages"></div></div><div class="viz2-anatomy-body"><div class="viz2-anatomy-stage"><div class="viz2-anatomy-viewport" tabindex="0" aria-label="Block 内计算图"><svg class="viz2-anatomy-svg"><g data-compute-scene></g></svg><div class="kf-pa-map__tooltip" role="tooltip" hidden></div></div><footer class="viz2-anatomy-footer"><span data-compute-legend></span><div class="kf-pa-map__tools"><button class="btn btn-ghost btn-sm" data-compute-zoom=".8">−</button><output data-compute-scale></output><button class="btn btn-ghost btn-sm" data-compute-zoom="1.25">＋</button><button class="btn btn-ghost btn-sm" data-compute-fit>Fit to View</button></div></footer></div><aside class="pto-ide-frame__pane viz2-anatomy-inspector" hidden><header class="pto-ide-frame__pane-header"><b data-compute-title></b><button class="btn btn-ghost btn-sm" data-compute-close>×</button></header><div class="pto-ide-frame__pane-body" data-compute-detail></div></aside></div>`;
 const viewport=host.querySelector('.viz2-anatomy-viewport'),scene=host.querySelector('[data-compute-scene]'),tip=host.querySelector('[role=tooltip]'),aside=host.querySelector('aside');
 host.querySelector('[data-compute-task]').innerHTML=model.tasks.map(t=>`<option value="${t.id}">Task ${t.id} · B${t.block} / Batch ${t.batch} / KV ${t.kvHead}</option>`).join('');
 let task,focus,view={},camera,active=false,drag,suppress=false,expanded=new Set(),hover=null;
 const bounds=()=>({w:1390,h:expanded.size?1030:740});
 const fit=()=>{const b=bounds(),scale=Math.max(.05,Math.min((viewport.clientWidth-48)/b.w,(viewport.clientHeight-32)/b.h,1));onView({intraCamera:{scale,x:(viewport.clientWidth-b.w*scale)/2,y:(viewport.clientHeight-b.h*scale)/2}});};
 function paintCamera(){if(!camera)return;scene.setAttribute('transform',`translate(${camera.x} ${camera.y}) scale(${camera.scale})`);scene.style.setProperty('--anatomy-font',`${Math.max(12,10/camera.scale)}px`);host.querySelector('[data-compute-scale]').textContent=`${Math.round(camera.scale*100)}%`;}
 const path=(a,b,kind,i)=>{
  if(a.id===b.id)return `M ${a.x+30} ${a.y+66} C ${a.x-70} 580 ${a.x+220} 580 ${a.x+140} ${a.y+66}`;
  if(kind==='event')return `M ${a.x+85} ${a.y} C ${a.x+85} ${55-i*12} ${b.x+85} ${55-i*12} ${b.x+85} ${b.y}`;
  if(kind==='state')return `M ${a.x+85} ${a.y+(a.y<300?66:0)} C ${a.x+85} ${315+i*5} ${b.x+85} ${315+i*5} ${b.x+85} ${b.y+(b.y<300?66:0)}`;
  if(kind==='data'&&b.x<a.x)return `M ${a.x} ${a.y+33} C ${a.x-30} ${a.y+33} ${b.x+200} ${b.y+33} ${b.x+170} ${b.y+33}`;
  if(kind==='data'&&b.x===a.x)return `M ${a.x+85} ${a.y+66} V${b.y}`;
  return `M ${a.x+170} ${a.y+33} C ${a.x+200} ${a.y+33} ${b.x-30} ${b.y+33} ${b.x} ${b.y+33}`;
 };
 function primitiveGraph(n){
  const S=task.stackCount,P=task.pageCount,T=task.tickCount;
  const spec={
   qk:[['load','Load K',P,0,0,[]],['transpose','Transpose View',P,1,0,['load']],['matmul','Matmul',P,2,0,['transpose']],['store','Store Scores',P,3,0,['matmul']]],
   softmax:[['load','Load / Scale / Fillpad',S,0,0,[]],['max','Row Max / Maximum',S,1,0,['load']],['rescale','Sub / Exp → rescale',S,2,0,['max']],['prob','Row Sub / Exp → P',S,2,1,['load','max']],['sum','Row Sum P',S,3,1,['prob']],['l','Mul / Add → l_sum',S,4,0,['rescale','sum']],['cast','Cast BF16 / Store P',S,4,1,['prob']]],
   pv:[['load','Load V / P',P,0,0,[]],['matmul','Matmul (first Page)',S,1,0,['load']],['acc','Matmul Acc (rest)',P-S,2,0,['matmul','load']],['store','Store PV',S,3,0,['acc']]],
   accumulate:[['load','Load PV',S,0,1,[]],['rescale','Row Mul o × pending0',S,0,0,[]],['add','Add → o_next',S,1,0,['load','rescale']]],
   normalize:[['recip','Reciprocal l_sum',1,0,0,[]],['mul','Row Mul o / l_sum',1,1,0,['recip']],['cast','Cast BF16',1,2,0,['mul']],['store','Store Output',1,3,0,['cast']]],
   rescale:[['p0','Move pending1 → p0',T,0,0,[]],['p1','Move rescale → p1',T,0,1,[]]]
  }[n.id]||[];
  if(n.id==='pv'&&P===S){spec.splice(2,1);spec.at(-1)[5]=['matmul'];}
  const x=40,y=810,w=215,h=40,dx=262,dy=70,map=new Map(spec.map(a=>[a[0],a]));
  return `<g class="compute-expanded"><text x="${x}" y="${y-30}">${esc(n.title)} · Operation Graph · ${esc(n.unit)}</text>${spec.flatMap(a=>a[5].map(id=>{const b=map.get(id);return `<path class="compute-edge data" marker-end="url(#compute-arrow)" d="M${x+b[3]*dx+w} ${y+b[4]*dy+20} C${x+b[3]*dx+w+20} ${y+b[4]*dy+20} ${x+a[3]*dx-20} ${y+a[4]*dy+20} ${x+a[3]*dx} ${y+a[4]*dy+20}"/>`;})).join('')}${spec.map(a=>`<g data-compute-node="${n.id}" role="button" tabindex="0" aria-label="${esc(a[1])}"><rect x="${x+a[3]*dx}" y="${y+a[4]*dy}" width="${w}" height="${h}" rx="3"/><text x="${x+a[3]*dx+8}" y="${y+a[4]*dy+25}">${esc(a[1])} ×${a[2]}</text></g>`).join('')}</g>`;
 }
 function draw(){
  const map=new Map(task.nodes.map(n=>[n.id,n]));
  scene.innerHTML=`<defs><marker id="compute-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="context-stroke"/></marker></defs><text x="30" y="38">Task ${task.id} · Compute Graph</text>${['qk','pv','softmax','accumulate'].map(id=>{const n=map.get(id),page=['qk','pv'].includes(id);return `<g class="compute-loop-group"><rect x="${n.x-12}" y="${n.y-34}" width="194" height="112" rx="6"/><text class="compute-loop" x="${n.x}" y="${n.y-16}">${page?'Page':'Stack'} Loop ×${page?task.pageCount:task.stackCount} · ${page?'AIC':'Lane'}</text></g>`;}).join('')}${task.edges.map((e,i)=>`<path class="compute-edge ${e.kind}" data-compute-edge="${e.id}" d="${path(map.get(e.s),map.get(e.t),e.kind,i%3)}" marker-end="url(#compute-arrow)"/><title>${esc(e.label)}</title>`).join('')}${task.edges.filter(e=>e.kind==='event').map((e,i)=>`<g data-compute-edge="${e.id}" class="compute-event" tabindex="0" role="button" aria-label="${e.label}"><rect x="${(map.get(e.s).x+map.get(e.t).x)/2+15}" y="${8+i*15}" width="140" height="20" rx="3"/><text x="${(map.get(e.s).x+map.get(e.t).x)/2+22}" y="${23+i*15}">${e.label}</text></g>`).join('')}${task.nodes.map(n=>`<g data-compute-node="${n.id}" class="compute-node ${n.role}" tabindex="0" role="button" aria-label="${esc(n.title)}" data-count="${n.count}" data-count-unit="${esc(n.unit)}"><rect x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}" rx="${n.role==='state'?24:4}"/><text x="${n.x+12}" y="${n.y+23}">${esc(n.title.split(' ').slice(0,2).join(' '))}</text><text x="${n.x+12}" y="${n.y+40}">${esc(n.title.split(' ').slice(2).join(' '))}</text><text class="compute-count" x="${n.x+12}" y="${n.y+59}">×${n.count}</text>${n.ops.length?`<text class="compute-expand" x="${n.x+150}" y="${n.y+59}">${expanded.has(n.id)?'−':'+'}</text>`:''}</g>${expanded.has(n.id)?primitiveGraph(n):''}}`).join('')}`;
  host.querySelector('[data-compute-task]').value=task.id;
  host.querySelector('[data-compute-stacks]').innerHTML=task.stacks.map(s=>`<button class="btn btn-ghost btn-sm" data-compute-stack="${s.index}">S${s.index}</button>`).join('');
  host.querySelector('[data-compute-context]').textContent=`Batch ${task.batch} · KV ${task.kvHead} · ${task.seqLen} tokens · ${task.pageCount} Pages / ${task.stackCount} Stacks / ${task.tickCount} logical ticks`;
 }
 function selectedStack(){if(Number.isInteger(focus?.stackIndex))return focus.stackIndex;if(['stack','stage','event'].includes(focus?.kind))return focus.index;if(focus?.kind==='page')return task.pages[focus.index]?.stack;return null;}
 function paint(){
  const f=hover||focus,id=f?.node||({stage:{qk:'qk',softmax:'softmax',pv:'pv',accumulate:'accumulate'}[f?.stage],output:'output'}[f?.kind]),eid=f?.edge|| (f?.kind==='event'?task.edges.find(e=>e.label===f.event?.replace('_EVENT',''))?.id:null);
  const relevant=new Set(id?[id]:[]);task.edges.forEach(e=>{if(e.s===id||e.t===id||e.id===eid){relevant.add(e.s);relevant.add(e.t);}});
  const s=selectedStack();
  const stack=task.stacks[s];
  host.querySelector('[data-compute-context]').textContent=`Batch ${task.batch} · KV ${task.kvHead} · ${task.seqLen} tokens · ${task.pageCount} Pages / ${task.stackCount} Stacks / ${task.tickCount} logical ticks${stack?` · S${s}: ${stack.tokens}/${task.stackTokens} tokens`:''}`;

  host.querySelector('[data-compute-pages]').innerHTML=stack?stack.pages.map(p=>`<button class="btn btn-ghost btn-sm ${focus?.kind==='page'&&focus.index===p?'is-selected':''}" data-compute-page="${p}">P${p} · ${task.pages[p].tokens}/${task.pageTokens}</button>`).join(''):'';
  scene.querySelectorAll('[data-compute-node]').forEach(el=>{el.classList.toggle('is-related',relevant.has(el.dataset.computeNode)||(['stack','page'].includes(f?.kind)&&s!=null&&(!['normalize','output'].includes(el.dataset.computeNode)||s===task.stackCount-1)));el.classList.toggle('is-selected',el.dataset.computeNode===id);});
  scene.querySelectorAll('[data-compute-edge]').forEach(el=>{const e=task.edges.find(e=>e.id===el.dataset.computeEdge);el.classList.toggle('is-related',e.id===eid||e.s===id||e.t===id);});
  host.querySelectorAll('[data-compute-stack]').forEach(b=>b.classList.toggle('is-selected',Number(b.dataset.computeStack)===s));
  host.dataset.mode=view.intraMode||'structure';host.querySelectorAll('[data-compute-mode]').forEach(b=>{b.classList.toggle('is-selected',b.dataset.computeMode===host.dataset.mode);b.setAttribute('aria-pressed',String(b.dataset.computeMode===host.dataset.mode));});
  scene.querySelectorAll('.compute-node').forEach(el=>{const n=task.nodes.find(n=>n.id===el.dataset.computeNode);el.style.setProperty('--weight',.18+.65*n.count/Math.max(task.pageCount,task.tickCount));});
  scene.querySelectorAll('path[data-compute-edge]').forEach(el=>{const e=task.edges.find(e=>e.id===el.dataset.computeEdge);el.style.strokeWidth=host.dataset.mode==='weight'&&e.kind==='data'?1+2*e.count/Math.max(1,task.pageCount):1.2;});
  host.querySelector('[data-compute-legend]').textContent=host.dataset.mode==='weight'?'Static Execution Weight · QK/PV: AIC · Softmax/Acc/Normalize: 每 AIV Lane · 非耗时':'实线 Data · 虚线 Loop State · 点线 READY Event';
 }
 function description(f){
  const n=task.nodes.find(n=>n.id===f?.node),e=task.edges.find(e=>e.id===f?.edge),s=selectedStack();
  if(n)return {title:n.title,line:n.line,rows:[['类型',n.role],['静态次数',`×${n.count} · ${n.unit}`],['公式 / 循环边界',n.formula],['语义 / Shape / Dtype / Memory',n.detail],['生产者',task.edges.filter(e=>e.t===n.id).map(e=>e.s).join(', ')||'Task input'],['消费者',task.edges.filter(e=>e.s===n.id).map(e=>e.t).join(', ')||'Task output'],...(n.ops.length?[['局部展开','再次点击节点展开 / 收起实际操作']]:[])]};
  if(e)return {title:e.label||`${e.s} → ${e.t}`,line:task.nodes.find(n=>n.id===e.t).line,rows:[['关系类型',e.kind],['生产者 → 消费者',`${e.s} → ${e.t}`],['统计口径',e.kind==='event'?`${e.label} ×${task.stackCount} / Task（每 Stack 1 次）；${e.label==='SOFTMAX_READY'?'两条 AIV Lane 各 set；AIC wait':'AIC set；两条 AIV Lane 各 wait'}`:e.kind==='state'?'跨 Stack 状态；FIFO move 按 Tick':`×${e.count} · ${['scores','partial','softmax','normalize','accumulate'].includes(e.s)?'每 AIV Lane':'AIC'} / Task；逻辑调用，不推断 Bytes`],['状态关系',e.kind==='state'?`${s!=null?`S${Math.max(0,s-1)} → S${s}${s===0?'（读取初始化状态）':''}`:'S(s−1) → S(s)'}；m/l/o 初始化 −3e38 / 0 / 0，FIFO (1,1)。${e.s==='rescale'||e.t==='rescale'?'rescale[s] 延迟 PRE_LAUNCH=2 后用于 o[s]。':''}`:'见源代码的 sync_set / sync_wait 与对应 transfer buffer']]};
  const stack=task.stacks[s],page=focus?.kind==='page'?task.pages[focus.index]:null;return {title:page?`Page ${page.index}`:stack?`Stack ${s}`:'Task '+task.id,line:759,rows:[...(page?[['逻辑 Page Index',page.index],['Page 有效 Tokens',`${page.tokens}/${task.pageTokens}`],['所属 Stack',page.stack]]:[]),['当前 Task',`Block ${task.block} / Batch ${task.batch} / KV Head ${task.kvHead}`],['有效 Pages',stack?stack.pages.map(p=>`P${p} (${task.pages[p].tokens}/${task.pageTokens})`).join(', '):`${task.pageCount}`],['有效 Tokens',stack?`${stack.tokens}/${task.stackTokens}`:task.seqLen],['计算实例',stack?`QK/PV ${stack.pages.length} Pages；Softmax/Accumulate 各 Lane 1 次`:'选择 Stack 查看对应实例'],['物理 Page','未提供实际 block_table，未知']]};
 }
 function inspector(){aside.hidden=!view.intraInspectorOpen;if(aside.hidden)return;const d=description(focus);host.querySelector('[data-compute-title]').textContent=d.title;host.querySelector('[data-compute-detail]').innerHTML=`<dl>${d.rows.map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')}</dl><button class="btn btn-ghost btn-sm" data-compute-source="${d.line}">PyPTO 源码 :${d.line}</button><pre data-compute-code hidden></pre>`;}
 function zoom(f,p={x:viewport.clientWidth/2,y:viewport.clientHeight/2}){const scale=Math.max(.08,Math.min(4,camera.scale*f)),r=scale/camera.scale;onView({intraCamera:{scale,x:p.x-(p.x-camera.x)*r,y:p.y-(p.y-camera.y)*r}});}
 host.addEventListener('change',e=>{if(e.target.matches('[data-compute-task]'))onTask(Number(e.target.value));});
 host.addEventListener('click',e=>{e.stopPropagation();tip.hidden=true;hover=null;if(suppress){suppress=false;return;}const b=e.target.closest('[data-compute-mode],[data-compute-stack],[data-compute-page],[data-compute-node],[data-compute-edge],[data-compute-fit],[data-compute-zoom],[data-compute-close],[data-compute-source]');if(!b)return;
  if(b.hasAttribute('data-compute-mode')){onView({intraMode:b.dataset.computeMode});return;}
  if(b.hasAttribute('data-compute-fit')){fit();return;}if(b.dataset.computeZoom){zoom(Number(b.dataset.computeZoom));return;}if(b.hasAttribute('data-compute-close')){onView({intraInspectorOpen:false});return;}
  if(b.dataset.computeSource){const line=Number(b.dataset.computeSource),code=host.querySelector('[data-compute-code]');code.hidden=false;code.textContent=model.source.split('\n').slice(Math.max(0,line-4),line+20).map((v,i)=>`${Math.max(1,line-3)+i}  ${v}`).join('\n');return;}
  if(b.hasAttribute('data-compute-page')){onFocus({kind:'page',index:Number(b.dataset.computePage)});return;}
  if(b.hasAttribute('data-compute-stack')){onFocus({kind:'stack',index:Number(b.dataset.computeStack)});return;}
  if(b.dataset.computeNode){const n=task.nodes.find(n=>n.id===b.dataset.computeNode);if(n.ops.length){const collapse=expanded.has(n.id);expanded.clear();if(!collapse)expanded.add(n.id);draw();paint();if(collapse)fit();else{const scale=Math.min((viewport.clientWidth-40)/1390,(viewport.clientHeight-32)/260,1);onView({intraCamera:{scale,x:(viewport.clientWidth-1390*scale)/2,y:viewport.clientHeight/2-870*scale}});}}const stage=['qk','softmax','pv','accumulate'].includes(n.id)?n.id:null;onFocus({kind:stage?'stage':n.id==='output'?'output':'graph',index:selectedStack()??0,stackIndex:selectedStack(),node:n.id,...(stage?{stage}:{})});return;}
  const edge=task.edges.find(e=>e.id===b.dataset.computeEdge);onFocus({kind:edge.kind==='event'?'event':'graph-edge',...(edge.kind==='event'?{event:edge.label+'_EVENT'}:{}),edge:edge.id,index:selectedStack()??0,stackIndex:selectedStack()});
 });
 viewport.addEventListener('wheel',e=>{e.preventDefault();const r=viewport.getBoundingClientRect();zoom(Math.exp(Math.max(-.22,Math.min(.22,-e.deltaY*.003))),{x:e.clientX-r.left,y:e.clientY-r.top});},{passive:false});
 viewport.addEventListener('pointerdown',e=>{if(e.button===0)drag={x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY,id:e.pointerId,moved:false};});
 viewport.addEventListener('pointermove',e=>{if(drag){if(!drag.moved&&Math.hypot(e.clientX-drag.startX,e.clientY-drag.startY)<4)return;drag.moved=true;viewport.setPointerCapture(drag.id);onView({intraCamera:{...camera,x:camera.x+e.clientX-drag.x,y:camera.y+e.clientY-drag.y}});drag.x=e.clientX;drag.y=e.clientY;tip.hidden=true;return;}const el=e.target.closest('[data-compute-node],[data-compute-edge]');hover=el?el.dataset.computeNode?{node:el.dataset.computeNode}:{edge:el.dataset.computeEdge}:null;paint();tip.hidden=!hover;if(hover){const d=description(hover);tip.textContent=`${d.title} · ${d.rows.slice(0,3).map(r=>r.join(': ')).join(' · ')} · source:${d.line}`;const r=viewport.getBoundingClientRect();tip.style.left=`${Math.max(8,Math.min(e.clientX-r.left+12,r.width-tip.offsetWidth-8))}px`;tip.style.top=`${Math.max(8,Math.min(e.clientY-r.top+12,r.height-tip.offsetHeight-8))}px`;}});
 ['pointerup','pointercancel','lostpointercapture'].forEach(k=>viewport.addEventListener(k,()=>{suppress=!!drag?.moved;drag=null;}));viewport.addEventListener('pointerleave',()=>{hover=null;tip.hidden=true;paint();});
 viewport.addEventListener('keydown',e=>{if(['Enter',' '].includes(e.key)&&e.target.matches('[data-compute-node],[data-compute-edge]')){e.preventDefault();e.target.dispatchEvent(new MouseEvent('click',{bubbles:true}));}if(e.key==='0')fit();});
 let size=null;new ResizeObserver(()=>{if(!active||!viewport.clientWidth)return;const next={w:viewport.clientWidth,h:viewport.clientHeight};if(size&&camera&&(size.w!==next.w||size.h!==next.h)){if(expanded.size){const scale=Math.min((next.w-40)/1390,(next.h-32)/260,1);onView({intraCamera:{scale,x:(next.w-1390*scale)/2,y:next.h/2-870*scale}});}else onView({intraCamera:{...camera,x:camera.x+(next.w-size.w)/2,y:camera.y+(next.h-size.h)/2}});}size=next;}).observe(viewport);
 return {update(id,f,v,{active:next}={}){const entering=next&&!active;active=!!next;view=v;focus=f;camera=v.intraCamera;if(!active)return;const t=model.tasks[id];if(!t)return;const changed=t!==task;if(changed){task=t;expanded=new Set();draw();}paint();inspector();if(entering||changed||!camera){fit();return;}paintCamera();},hideTooltip(){tip.hidden=true;},getTransitionRect(){return viewport.getBoundingClientRect();}};
}
global.PtoVisualization2GraphIntra=Object.freeze({createModel,mount});
})(window);
