import React,{useState,useMemo,useEffect} from 'react';
import {createRoot} from 'react-dom/client';
import ReactFlow,{ReactFlowProvider,Controls,useReactFlow,useNodesInitialized,getTransformForBounds} from 'reactflow';
import 'reactflow/dist/style.css';
import '../../csa-pass-field/src/styles.css';
import {nodeTypes,edgeTypes,DiagramSettings} from '../../csa-pass-field/src/graph-components.jsx';
import {snapshotAt,callableFlow} from '../../csa-pass-field/src/CallableField.jsx';
import * as qk from '../../csa-pass-field/src/qk-graph-state.js';
import mergePasses from '../../csa-pass-field/data/callable/passes.json';
import mergeIsa from '../../csa-pass-field/data/callable/isa.json';
import qkIsa from '../../csa-pass-field/data/csa-qk-pv/isa.json';
import {instructionCollections as mergeCollections} from '../../csa-pass-field/src/instruction-collections.js';
import {instructionCollections as qkCollections} from '../../csa-pass-field/src/qk-instruction-collections.js';
import './panel.css';
const adapter=()=>window.CompileLaneAdapter;
import {sampleFor,kernelsFor as resolveKernels} from './match.mjs';
let activeFinding=null,previewFinding=null;
const escapeHtml=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function findings(){
 const tasks=adapter()?.current().launches.filter(l=>/^qk_pv_(aic|aiv)$/.test(l.leaf))||[];
 // Fixed scenario values exercise the analysis interactions independently of trace metrics.
 const issues=[
  {id:'qk-vector-split',title:'P25 · Tensor 分片工作量不均',evidence:'32 × 128 沿列切分为 32 × 96 / 32 × 32',detail:'Tensor 沿第 1 维（列）切分，两个分片分别含 3072 和 1024 个元素，工作量相差 3×。检查分片边界；不推算设备耗时。',color:'#b69af3',pass:25,nodes:['value.prob'],kernels:['qk_pv_aiv'],kind:'issue',comparison:[['输入 Tensor','32 × 128','沿列维度切分'],['分片 0','未拆分','32 × 96 · 列 [0, 96)'],['分片 1','未拆分','32 × 32 · 列 [96, 128)']],scenario:[{label:'分片 0',shape:[32,96],sub:'列 [0, 96)'},{label:'分片 1',shape:[32,32],sub:'列 [96, 128)'}]},
  {id:'qk-c2v-slot',title:'P28 · C2V 交接 slot 不匹配',evidence:'同一次交接：Push slot 0 → Pop slot 1',detail:'流水重排后，同一迭代的生产端与消费端引用了不同队列 slot。应检查 slot 索引与迭代偏移，使配对交接保持一致。',color:'#efc65d',pass:28,nodes:['handoff.c2v-push','handoff.c2v-pop'],edges:['control.c2v'],kernels:['qk_pv_aic','qk_pv_aiv'],kind:'issue',comparison:[['生产端 Push','同一逻辑队列','迭代 i · slot 0'],['消费端 Pop','同一逻辑队列','迭代 i · slot 1'],['配对关系','Push → Pop','slot 0 ≠ slot 1']],scenario:[{label:'Push · slot 0',sub:'生产端 · 迭代 i'},{label:'Pop · slot 1',sub:'消费端 · 迭代 i'}]}
 ];
 const structures=[23,25,28].map((index,i)=>{const p=qk.passes.passes.find(p=>p.index===index);return {id:'pass-'+index,title:['AIC / AIV 拆分','Vector 子块拆分','跨核流水重排'][i],evidence:p.summary,detail:('结构差异：'+p.delta.filter(d=>d.op==='addNode').length+' 个新增对象，'+p.delta.filter(d=>d.op==='removeNode').length+' 个移除对象，'+p.delta.filter(d=>d.op==='updateNode').length+' 个对象更新；'+p.delta.filter(d=>d.op==='addEdge').length+' 条新增关系。'+(p.sourceRefs?.length?'证据：'+p.sourceRefs[0].file.split('/').pop()+':'+p.sourceRefs[0].line:'') ),color:['#67b9a0','#78b7ff','#efc65d'][i],pass:index,nodes:p.affectedObjectIds.filter(id=>!id.startsWith('data.')&&!id.startsWith('control.')),kernels:['qk_pv_aic','qk_pv_aiv'],kind:'structure'};});
 return [...issues,...structures];
}
function findingById(id){return findings().find(f=>f.id===id);}
function menu(){const rows=items=>items.map(f=>`<button class="ctl result-item" data-compile-finding="${f.id}" aria-pressed="${activeFinding===f.id}"><span class="result-name"><i style="width:20px;height:20px;flex:none;border-radius:50%;background:${f.color}"></i>${escapeHtml(f.title)}</span><small>${escapeHtml(f.evidence)}</small></button>`).join('');const all=findings(),issues=all.filter(f=>f.kind==='issue');return `<details class="bar-menu" data-menu="compile"><summary class="ctl mode-pill${document.body.classList.contains('compile-analysis-mode')?' on':''}" data-compile-menu>编译分析</summary><div class="menu-panel results-menu"><strong>qk_pv · 分析关注点</strong>${issues.length?rows(issues):'<small>当前未发现块耗时异常</small>'}<details><summary class="ctl">结构变化</summary>${rows(all.filter(f=>f.kind==='structure'))}</details><div class="menu-divider"></div><button class="ctl" data-compile-tab>浏览全部 Pass</button></div></details>`;}
window.CompileAnalysis={menu};
function showFinding(id,preview=false){const f=findingById(id);if(!f)return;if(!preview){activeFinding=activeFinding===id?null:id;previewFinding=null;if(activeFinding){document.body.classList.add('compile-analysis-mode');window.CompileDrawer.open('qk_pv',f.pass);}else{close();adapter()?.resize();return;}}else previewFinding=id;
 adapter()?.preview(f.kernels,null,f);window.dispatchEvent(new CustomEvent('compile-finding',{detail:{...f,preview}}));}
let host,root,previousFocus;
const close=()=>{if(host)host.hidden=true;document.body.classList.remove('compile-split');document.querySelector('[data-compile-tab]')?.setAttribute('aria-pressed','false');root?.render(null);adapter()?.preview([]);adapter()?.leaveCompile();activeFinding=null;adapter()?.resize();previousFocus?.focus({preventScroll:true});};
function PassRail({pass,isa,onSelect}){
 return <nav className="linked-pass-rail" aria-label="Pass 阶段导航"><div className="rail-scroll"><div className="phase-passes">{qk.passes.passes.filter(p=>p.delta?.length).map(p=><button key={p.index} className={'pass-tick scope-local '+([25,28].includes(p.index)?'has-issue ':'')+(!isa&&pass===p.index?'is-current':'')} title={p.name} aria-label={'Pass '+p.index+' '+p.name} onClick={()=>onSelect(p.index,false)}><span>{String(p.index).padStart(2,'0')}</span></button>)}<button className={'pass-tick '+(isa?'is-current':'')} aria-label="查看 ISA" onClick={()=>onSelect(51,true)}>ISA</button></div></div></nav>;
}
function Panel({initialSample,initialPass}){
 const [sample,setSample]=useState(initialSample),[pass,setPass]=useState(initialPass),[isa,setIsa]=useState(false),[selected,setSelected]=useState(initialSample==='qk_pv'?'op.qk':'op.merge'),[hover,setHover]=useState(null),[expanded,setExpanded]=useState(null),[task,setTask]=useState(()=>adapter()?.current().selected),[instance,setInstance]=useState(null),[height,setHeight]=useState(host?.clientHeight||480),[finding,setFinding]=useState(()=>findingById(activeFinding));
 const flowApi=useReactFlow(),nodesReady=useNodesInitialized(),isQk=sample==='qk_pv';
 const passes=isQk?qk.passes.passes:mergePasses.passes;
 const current=passes.find(p=>p.index===pass)||passes[0];
 const collections=useMemo(()=>(isQk?qkCollections:mergeCollections)((isQk?qkIsa:mergeIsa).instructions),[isQk]);
 const domain=useMemo(()=>isQk?qk.buildDomainState(pass):snapshotAt(pass),[isQk,pass]);
 const loadData=useMemo(()=>({expandedRegion:expanded,onToggle:id=>{setHover(null);setExpanded(v=>v===id?null:id);setSelected(id);setInstance(null);},onInstanceSelect:i=>{setHover(null);setInstance(i);setSelected(i.collectionId);}}),[expanded]);
 const graph=useMemo(()=>(isQk?qk.toReactFlow:callableFlow)(domain,{currentPass:current,selectedId:selected,hoverId:hover,showIsa:isa,positionOverrides:{},layout:'horizontal',loadExpanded:expanded,loadData}),[domain,current,selected,hover,isa,expanded,loadData,isQk]);
 const activeCollection=collections.find(c=>c.id===(hover||selected));
 const linkedGraph=useMemo(()=>{const ids=new Set(finding?.nodes||[]);const anchor=graph.nodes.find(n=>n.id===finding?.nodes[0]);const nodes=graph.nodes.map(n=>({...n,className:[n.className,n.id===activeCollection?.owner||ids.has(n.id)?'compile-owner':''].filter(Boolean).join(' ')}));if(anchor&&finding){let x=anchor.position.x,y=anchor.position.y,p=anchor.parentNode;while(p){const n=graph.nodes.find(v=>v.id===p);if(!n)break;x+=n.position.x;y+=n.position.y;p=n.parentNode;}nodes.push({id:'finding-tag',position:{x,y:y-52},data:{label:finding.title},style:{background:finding.color,color:'#111',border:0,borderRadius:24,width:290,padding:'9px 14px',fontSize:14,fontWeight:600},draggable:false});}const edges=graph.edges.map(e=>({...e,...(finding?.edges?.includes(e.id)?{animated:true,style:{...e.style,stroke:finding.color,strokeWidth:3}}:{})}));
 if(anchor&&finding?.scenario){let x=anchor.position.x,y=anchor.position.y,p=anchor.parentNode;while(p){const n=graph.nodes.find(v=>v.id===p);if(!n)break;x+=n.position.x;y+=n.position.y;p=n.parentNode;}
 finding.scenario.forEach((item,i)=>{const id='scenario-'+i;nodes.push(item.shape?{id,type:'tensor',position:{x:x+360,y:y+120+i*130},data:{label:item.label,semanticKind:'tensor_value',glyphWidth:i===0?336:112,attributes:{shape:item.shape,dtype:'BF16'},memoryRanges:[item.sub]},style:{width:i===0?336:112,height:64},draggable:false}:{id,position:{x:x+280,y:y+i*110},data:{label:<><strong>{item.label}</strong><div style={{marginTop:8,fontSize:12}}>{item.sub}</div></>},style:{background:'#242424',color:finding.color,border:'2px solid '+finding.color,borderRadius:8,width:210,padding:14},draggable:false});edges.push({id:'scenario-edge-'+i,source:finding.nodes[Math.min(i,finding.nodes.length-1)],target:id,sourceHandle:item.shape?'out':undefined,targetHandle:item.shape?'in':undefined,type:'step',label:item.shape?item.sub:undefined,labelStyle:{fill:'#ddd',fontSize:12},labelBgStyle:{fill:'#222'},style:{stroke:finding.color,strokeWidth:2}});});
 if(finding.id==='qk-c2v-slot')edges.push({id:'scenario-mismatch',source:'scenario-0',target:'scenario-1',type:'step',label:'slot 不匹配',style:{stroke:finding.color,strokeWidth:2,strokeDasharray:'5 4'},labelStyle:{fill:finding.color}});
 }
 return {...graph,nodes,edges};},[graph,activeCollection,finding]);
 useEffect(()=>{const fn=e=>{setFinding(e.detail);if(!e.detail)return;setPass(e.detail.pass);setIsa(false);setSelected(e.detail.nodes[0]||null);setHover(null);};window.addEventListener('compile-finding',fn);return()=>window.removeEventListener('compile-finding',fn);},[]);
 const chosen=graph.nodes.find(n=>n.id===selected)?.data;
 const instructions=collections.filter(c=>c.id===selected||c.owner===selected||c.kernelId===selected);
 const kernelsFor=id=>resolveKernels(id,sample,collections);
 function preview(id){adapter()?.preview(!hover&&instance?.kernel?[instance.kernel]:kernelsFor(id),sampleFor(task)===sample?task.taskId:null);}
 useEffect(()=>{const refresh=()=>finding?adapter()?.preview(finding.kernels,null,finding):preview(hover||selected);refresh();window.addEventListener('compile-lane-resized',refresh);return()=>{window.removeEventListener('compile-lane-resized',refresh);adapter()?.preview([]);};},[hover,selected,sample,task,instance,finding]);
 useEffect(()=>{if(!nodesReady)return;const t=setTimeout(()=>{
   const anchor=finding?.nodes[0]||(isQk?'op.qk':'op.merge');
   const absolute=n=>{let x=n.position.x,y=n.position.y,p=n.parentNode;while(p){const parent=graph.nodes.find(v=>v.id===p);if(!parent)break;x+=parent.position.x;y+=parent.position.y;p=parent.parentNode;}return {...n,position:{x,y}};};
   const nodes=linkedGraph.nodes.filter(n=>finding?(finding.nodes.includes(n.id)||n.id.startsWith('scenario-')||n.id==='finding-tag'):isa?n.id.startsWith('instructions.'):n.id===anchor||(['op','handoff','tensor'].includes(n.type)&&graph.edges.some(e=>(e.source===anchor&&e.target===n.id)||(e.target===anchor&&e.source===n.id)))).map(absolute);
   if(!nodes.length)return;
   const x=Math.min(...nodes.map(n=>n.position.x)),y=Math.min(...nodes.map(n=>n.position.y));
   const width=Math.max(...nodes.map(n=>n.position.x+Number(n.style?.width||180)))-x,height=Math.max(...nodes.map(n=>n.position.y+Number(n.style?.height||70)))-y;
   const area=host.shadowRoot.querySelector('.graph');
   const [tx,ty,zoom]=getTransformForBounds({x,y,width,height},area.clientWidth,area.clientHeight-140,.12,.9,.2);
   flowApi.setViewport({x:tx,y:ty+65,zoom},{duration:0});
 },120);return()=>clearTimeout(t);},[sample,pass,isa,nodesReady,finding?.id]);
 useEffect(()=>{const fn=e=>{const l=e.detail;setTask(l);const next=sampleFor(l);if(next==='qk_pv'){setSample(next);setSelected(next==='qk_pv'?(l.ct==='AIC'?'kernel.aic':'kernel.aiv'):'kernel.main');setInstance(null);}else{setSelected(null);setInstance(null);}};window.addEventListener('compile-task-selected',fn);return()=>window.removeEventListener('compile-task-selected',fn);},[sample]);
 const matching=adapter()?.current().launches.filter(l=>sampleFor(l)===sample)||[];
 const taskMatch=sampleFor(task)===sample;
 function resize(h){h=Math.min(window.innerHeight-300,Math.max(260,h));setHeight(h);host.style.height=h+'px';document.body.style.setProperty('--compile-linked-height',h+'px');}
 return <DiagramSettings.Provider value={{palette:'identity',layout:'horizontal'}}><div className="linked-panel">
 <div className="resize" role="separator" tabIndex={0} aria-label="调整编译面板高度" aria-orientation="horizontal" onKeyDown={e=>{if(['ArrowUp','ArrowDown'].includes(e.key)){e.preventDefault();resize(height+(e.key==='ArrowUp'?30:-30));}}} onPointerDown={e=>{e.currentTarget.setPointerCapture(e.pointerId);e.currentTarget.dataset.start=e.clientY;e.currentTarget.dataset.height=host.clientHeight;}} onPointerMove={e=>{if(e.currentTarget.hasPointerCapture(e.pointerId))resize(+e.currentTarget.dataset.height + +e.currentTarget.dataset.start-e.clientY);}} onPointerUp={e=>e.currentTarget.releasePointerCapture(e.pointerId)}/>
 <header><strong>qk_pv</strong><span>编译分析 · {isa?'PTO-ISA':`P${String(pass).padStart(2,'0')} · ${current.name}`}</span><button onClick={()=>flowApi.fitView({padding:.12})}>适配</button><button onClick={close} aria-label="关闭编译分析">×</button></header>
 {finding&&<div className="finding-evidence"><details><summary style={{background:finding.color}}>{finding.title}</summary><p>{finding.evidence}</p><p>{finding.detail}</p>{finding.comparison&&<table className="finding-comparison"><thead><tr><th>对象</th><th>变换前</th><th>变换后</th></tr></thead><tbody>{finding.comparison.map(row=><tr key={row[0]}>{row.map((cell,i)=><td key={i}>{cell}</td>)}</tr>)}</tbody></table>}<small>P{finding.pass} · {qk.passes.passes.find(p=>p.index===finding.pass)?.name}</small></details></div>}
 {isa&&<div className="mapping-caption">{(hover||selected)?(kernelsFor(hover||selected).length?`所属 kernel：${(!hover&&instance?.kernel?[instance.kernel]:kernelsFor(hover||selected)).join(' / ')} · 非指令耗时`:'无可确认的泳道映射'):'选择指令查看所属 kernel'}{activeCollection&&<details key={activeCollection.id}><summary>{activeCollection.label} · {activeCollection.owner} · 源码位置</summary>{activeCollection.instances.map(i=><button key={i.id} aria-pressed={instance?.id===i.id} onClick={()=>{setSelected(activeCollection.id);setInstance(i);}}>L{i.line} · {i.kernel}</button>)}{instance&&<pre>{instance.text}</pre>}</details>}</div>}
 <PassRail pass={pass} isa={isa} onSelect={(index,instructions)=>{setFinding(null);setPass(index);setIsa(instructions);setSelected(instructions?null:'op.qk');setHover(null);setInstance(null);}}/>
 <div className="workspace"><div className="graph"><ReactFlow nodes={linkedGraph.nodes} edges={linkedGraph.edges} nodeTypes={nodeTypes} edgeTypes={edgeTypes} nodesDraggable={false} minZoom={.08} maxZoom={2} onNodeMouseEnter={(e,n)=>setHover(n.id)} onNodeMouseLeave={()=>setHover(null)} onNodeClick={(e,n)=>{setHover(null);setSelected(n.id);setInstance(null);}} onPaneClick={()=>{setSelected(null);setInstance(null);}}><Controls showInteractive={false}/></ReactFlow></div>
</div></div></DiagramSettings.Provider>;
}
window.CompileDrawer={open(sample,pass){
 sample='qk_pv';pass=pass||23;adapter()?.enterSplit();
 previousFocus=document.activeElement;document.querySelectorAll('.bar-menu[open]').forEach(x=>x.open=false);
 if(!host){host=document.createElement('section');host.className='compile-linked-host';host.setAttribute('aria-label','编译联动实验');document.querySelector('.main-column').append(host);const shadow=host.attachShadow({mode:'open'});const css=document.createElement('link');css.rel='stylesheet';css.href='compile-linked/dist/compile-linked.css';shadow.append(css);const mount=document.createElement('div');mount.style.height='100%';shadow.append(mount);root=createRoot(mount);shadow.addEventListener('keydown',e=>{if(e.key==='Escape')close();});}
 host.hidden=false;document.body.classList.add('compile-split');document.querySelector('[data-compile-tab]')?.setAttribute('aria-pressed','true');host.style.height=Math.min(480,innerHeight*.58)+'px';document.body.style.setProperty('--compile-linked-height',host.style.height);
 root.render(<ReactFlowProvider key={sample+pass+Date.now()}><Panel initialSample={sample} initialPass={pass}/></ReactFlowProvider>);
}};
let resizeFrame;new ResizeObserver(()=>{cancelAnimationFrame(resizeFrame);resizeFrame=requestAnimationFrame(()=>adapter()?.resize());}).observe(document.getElementById('lane-wrap'));
const hostStyle=document.createElement('style');hostStyle.textContent=`.compile-linked-host{position:relative;flex:none;min-height:260px;background:#111;border-top:1px solid #555}.compile-linked-host[hidden]{display:none}body.compile-split .cpi-nav{display:none}body.compile-split .trace-pair{min-height:160px;overflow:hidden;padding-bottom:0}body.compile-split .trace-pair>.sec:last-child{min-height:0;flex:1;padding-top:0;padding-bottom:0}body.compile-split #lane-wrap{overflow:auto;min-height:0}body.compile-split .lane-toolbar,body.compile-split .canvas-tools{bottom:calc(var(--compile-linked-height) + 14px)}body.compile-split .analysis-panel{display:none} [data-compile-tab][aria-pressed=true],body.compile-analysis-mode [data-compile-menu]{background:#eee!important;color:#111!important}body.compile-analysis-mode [data-compile-menu]::after{color:#555!important}`;document.head.append(hostStyle);
// The task inspector entry opens the same linked surface in this experiment.
document.getElementById('pass-analysis-open')?.addEventListener('click',e=>{const l=adapter()?.current().selected;const s=sampleFor(l);if(s){e.preventDefault();e.stopImmediatePropagation();window.CompileDrawer.open(s,s==='qk_pv'?23:35);}},true);

document.addEventListener('keydown',e=>{if(e.key==='Escape'&&host&&!host.hidden)close();});

document.addEventListener('click',e=>{
 const choice=e.target.closest('[data-compile-finding]');if(choice){showFinding(choice.dataset.compileFinding);return;}
 if(e.target.closest('[data-compile-menu]')){e.preventDefault();const wasOpen=!!e.target.closest('details')?.open;if(!host||host.hidden){document.body.classList.add('compile-analysis-mode');window.CompileDrawer.open('qk_pv',23);}else{document.body.classList.add('compile-analysis-mode');adapter()?.enterSplit();}adapter()?.resize();const menu=document.querySelector('[data-menu=compile]');if(menu)menu.open=!wasOpen;return;}
 if(e.target.closest('[data-filter-mode],[data-results],[data-path-mode]')){if(host&&!host.hidden)close();activeFinding=null;adapter()?.leaveCompile();return;}
 if(e.target.closest('[data-compile-tab]')){if(host&&!host.hidden){close();return;}activeFinding=null;document.body.classList.add('compile-analysis-mode');window.CompileDrawer.open('qk_pv',23);}
});
document.addEventListener('pointerover',e=>{const b=e.target.closest('[data-compile-finding]');if(b)showFinding(b.dataset.compileFinding,true);});
document.addEventListener('pointerout',e=>{const b=e.target.closest('[data-compile-finding]');if(b&&!b.contains(e.relatedTarget)){previewFinding=null;const f=findingById(activeFinding);adapter()?.preview(f?.kernels||[],null,f);window.dispatchEvent(new CustomEvent('compile-finding',{detail:f||null}));}});
adapter()?.resize();
