import {useEffect,useMemo,useState} from 'react';
import {ReactFlowProvider} from 'reactflow';
import {GraphCanvas,Icon,PassRail} from './App';
import {DiagramSettings} from './graph-components';
import {variants,checkpoints,source} from './case-data';
import {passes} from './graph-state';
import './case.css';
import {tensorGeometry} from './tensor-glyph';

function projection(variant,step,selected,hover,overrides,layout,actions){
 const nodes=[],edges=[];const active=hover||selected;
 const node=(id,type,label,x,y,w,h,extra={})=>nodes.push({id,type,position:overrides[id]||{x,y},style:{width:w,height:h},draggable:!['scope','memory'].includes(type),data:{id,label,semanticKind:type==='tensor'?'tile_value':type,attributes:{},selected:id===selected,...extra}});
 const edge=(a,b,type='data',extra={})=>edges.push({id:a+'>'+b,source:a,target:b,type,sourceHandle:type==='memoryBinding'?'bottom-out':'right-out',targetHandle:type==='memoryBinding'?'top-in':'left-in',data:{focused:a===active||b===active,...extra}});
 node('scope','scope',variant.id==='separate'?'hc_mix_x · 独立 scope':'hc_post · 多输出融合 scope',70,0,1400,740,{attributes:{level:'SPMD'}});
 node('pre','tensor','pre',100,270,110,74,{attributes:{shape:['TILE','HC_PAD']}});
 node('transpose','op','transpose',260,285,170,48);edge('pre','transpose');
 node('sum','op','Σ head',1250,285,150,48);
 node('out','tensor',variant.id==='separate'?'x_mixed · PASS':'x_mixed · FAIL',1530,270,120,74,{attributes:{shape:['TILE','D']},memoryRanges:[variant.result]});edge('sum','out');
 for(let h=0;h<4;h++){
  const y=65+h*150;
  node('slice'+h,'op',`slice / reshape · h${h}`,470,y,220,48);
  node('scale'+h,'tensor',`pre_${h}`,735,y-10,110,74,{attributes:{shape:['TILE',1]},memoryRanges:step>=4?[`UB base ${variant.addresses[h]}`]:[]});
  node('mul'+h,'op',`row_expand_mul · h${h}`,920,y,260,48);
  node('x'+h,'tensor',`x · head ${h}`,940,y+58,110,74,{attributes:{shape:['TILE','D']}});
  edge('x'+h,'mul'+h); edges.at(-1).sourceHandle='top-out'; edges.at(-1).targetHandle='bottom-in';
  edge('transpose','slice'+h);edge('slice'+h,'scale'+h);edge('scale'+h,'mul'+h);edge('mul'+h,'sum');
 }
 if(variant.id!=='separate'){
  node('post','tensor','post · PASS',110,70,140,74);node('comb','tensor','comb · PASS',110,450,140,74);
 }
 if(step>=4){
  const distinct=[...new Set(variant.addresses)];
  const highlights=[];
  node('ub','memory','UB',470,810,860,130,{spaceContainer:true,gridWidth:828,gridHeight:78,highlights});
  distinct.forEach((address,i)=>{
   const heads=variant.addresses.map((a,h)=>a===address?h:null).filter(h=>h!==null);
   const id='allocation.'+address,x=20+i*204,width=156; const color=variant.id==='separate'?['#3487d4','#70b0e4','#2470b3','#a1c9eb'][heads[0]]:'#0077FF';
   const linked=variant.id==='separate'||variant.id==='fused'||active===id||heads.some(h=>['scale'+h,'slice'+h,'mul'+h].includes(active));
   if(linked)highlights.push({x,y:0,width,height:72,color});
   node(id,'memory',String(address),486+x,846,width,72,{compactBlock:true,bindingLabel:variant.id==='separate'?`h${heads[0]} · ${address}`:String(address),base:String(address),shared:heads.length>1,touched:linked,description:`UB base ${address} · ${heads.map(h=>'pre_'+h).join(', ')} · 大小未知，块宽不表示字节数`});
   if(linked)heads.forEach(h=>edge('scale'+h,id,'memoryBinding',{compact:true,focused:true,bindingColor:color}));
  });
 }
 node('case.warning','caseWarning','UB 告警',470,970,600,52,{fixed:variant.id==='separate',attempted:variant.id==='materialized',onInspect:actions.inspect,onAction:actions.action});
 nodes.at(-1).draggable=false;
 // Use actual glyph dimensions before aligning ports; never route around a stale box.
 for(const n of nodes.filter(n=>n.type==='tensor')){const g=tensorGeometry(n.data);n.style.width=g.width;n.style.height=g.height;}
 const lookup=new Map(nodes.map(n=>[n.id,n]));
 for(let h=0;h<4;h++){
  const mul=lookup.get('mul'+h),x=lookup.get('x'+h),scale=lookup.get('scale'+h);
  scale.position.y=mul.position.y+mul.style.height/2-scale.style.height/2;
  x.position.x=mul.position.x+mul.style.width/2-x.style.width/2;
 }
 const sum=lookup.get('sum'),out=lookup.get('out');
 out.position.y=sum.position.y+sum.style.height/2-out.style.height/2;
 if(variant.id==='separate')for(const n of nodes){if(/^scale[0-3]$/.test(n.id)){n.data.touched=true;n.data.memoryRanges=[`h${n.id.slice(-1)} → UB ${variant.addresses[Number(n.id.slice(-1))]}`];}}
 if(layout==='vertical')for(const n of nodes)n.position={x:n.position.y,y:n.position.x};
 return {nodes,edges};
}
export default function CaseApp(){
 const initial=new URLSearchParams(location.hash.slice(1));
 const [variantId,setVariant]=useState('fused');
 const [passIndex,setPassIndex]=useState(36); const [proposal,setProposal]=useState('separate');
 const step=4; const [attempted,setAttempted]=useState(false);
 const [selected,setSelected]=useState(initial.get('node')||null),[hover,setHover]=useState(null),[panel,setPanel]=useState(null),[playing,setPlaying]=useState(false);
 const [palette,setPalette]=useState('identity'),[layout,setLayout]=useState('horizontal'),[overrides,setOverrides]=useState({});
 const variant=variants.find(v=>v.id===variantId),checkpoint={name:passes.passes[passIndex-1]?.name||'PTO-ISA · 最终产物',detail:'参考管线导航已恢复完整范围；CP-01 对应版本的逐 Pass dump 尚未接入，当前图保留 Issue 报告的最终地址证据，不随参考 Pass 伪造变化。'};
 const fixed=variantId==='separate';
 const applyFix=()=>{setVariant('separate');setSelected(null);setPlaying(false);setPanel(null);};
 const resetCase=()=>{setVariant('fused');setAttempted(false);setSelected(null);setPanel(null);setPlaying(false);setPassIndex(36);};
 const flow=projection(variant,step,selected,hover,overrides,layout,{inspect:()=>{setSelected(fixed?'out':'allocation.'+variant.addresses[0]);setPanel('detail');},action:()=>fixed?resetCase():setPanel('repair')});
 useEffect(()=>{history.replaceState(null,'','#'+new URLSearchParams({case:'CP-01',state:variantId==='separate'?'after':'before',pass:String(passIndex),...(selected?{node:selected}:{})}));},[variantId,passIndex,selected]);
 useEffect(()=>{if(!playing)return;if(passIndex===52){setPlaying(false);return;}const timer=setTimeout(()=>setPassIndex(s=>s+1),1800);return()=>clearTimeout(timer);},[playing,passIndex]);
 useEffect(()=>{const key=e=>{if(e.target.closest('input,select,textarea')||e.metaKey||e.ctrlKey)return;if(e.key==='Escape')setPanel(null);if(false&&e.key==='ArrowRight'){e.preventDefault();setPassIndex(s=>Math.min(52,s+1));}if(false&&e.key==='ArrowLeft'){e.preventDefault();setPassIndex(s=>Math.max(1,s-1));}};window.addEventListener('keydown',key);return()=>window.removeEventListener('keydown',key);},[]);
 const allocation=selected?.startsWith('allocation.')?Number(selected.split('.')[1]):null;
 const heads=allocation!==null?variant.addresses.map((a,h)=>a===allocation?h:null).filter(h=>h!==null):[];
 const selectedNode=flow.nodes.find(n=>n.id===selected);
 return <DiagramSettings.Provider value={{palette,layout}}><main className={`app-shell palette-${palette} layout-${layout}`}><section className="graph-stage" aria-label="CP-01 错误案例计算图">
 <ReactFlowProvider><GraphCanvas nodes={flow.nodes} edges={flow.edges} showIsa={false} loadExpanded={false} onSelect={id=>{setSelected(id);setPanel(id?'detail':null);}} onHover={setHover} onMoveNode={(id,pos)=>setOverrides(o=>({...o,[id]:pos}))} onResetPositions={()=>setOverrides({})}/></ReactFlowProvider>
 <header className="focus-heading"><span className="eyebrow">PASS ERROR CASES · CP-01</span><div className="focus-title"><strong>{String(passIndex).padStart(2,'0')}</strong><div><h1>{checkpoint.name}</h1><p>融合多输出 scope · 单路输出错误</p></div></div><div className="focus-state"><p>{variant.label} · x_mixed {variant.result}</p></div></header>
 <div className="top-actions">{[['settings','页面设置'],['info','图例与说明'],['detail','案例详情']].map(([id,label])=><button key={id} data-liquid-glass="icon" aria-label={label} data-tooltip={label} aria-expanded={panel===id} onClick={()=>setPanel(p=>p===id?null:id)}><Icon name={id}/></button>)}</div>
 <div className="relation-key"><span><i className="key-line data"/>DATA</span><span><i className="key-line binding"/>BINDING</span></div>
 {panel&&<aside className={`inspector overlay-panel inspector-right ${['repair','diff'].includes(panel)?'case-repair-panel':''}`} data-liquid-glass="panel" role="dialog" aria-label={panel==='diff'?'方案 Diff':panel==='repair'?'处理告警':panel==='settings'?'页面设置':panel==='info'?'图例与说明':'案例详情'}><header className="inspector-header"><span>{panel==='diff'?'方案 Diff':panel==='repair'?'处理告警':panel==='settings'?'页面设置':panel==='info'?'图例与说明':'CP-01 · 证据检查'}</span><button aria-label="关闭检查器" onClick={()=>setPanel(null)}>×</button></header>
 {panel==='repair'?<section className="object-block"><div className="repair-intro"><h2>选择处理方案</h2></div><div className="repair-options compact-options" role="radiogroup" aria-label="处理方案">{[['separate','拆分 mix_x scope','推荐'],['materialized','仅分离 buffer','尝试']].map(([id,title,badge])=><button role="radio" aria-checked={proposal===id} key={id} className="repair-option" onClick={()=>setProposal(id)}><small>{badge}</small><strong>{title}</strong><div className="option-analysis"><div className="option-scope">{id==='separate'?<><div className="mini-scope"><label>原 scope</label><span>pre · post · comb</span></div><span className="scope-arrow">→</span><div className="mini-scope moved"><label>独立 scope</label><span>mix_x</span></div></>:<div className="mini-scope"><label>保持融合 scope</label><span>pre · post · comb · mix_x</span></div>}</div><div className="option-bindings"><label>4 个 scale → 独立地址</label><div>{(id==='separate'?[32768,32832,32896,32960]:[512,576,640,704]).map(address=><code key={address}>{address}</code>)}</div></div><div className={`option-result ${id==='separate'?'pass':'fail'}`}><b>{id==='separate'?'✓ PASS':'⚠ 仍 FAIL'}</b><span>{id==='separate'?'精度通过':'约 97.9% 错误'}</span></div></div></button>)}</div><div className="repair-footer"><div className="repair-actions"><button className="isa-toggle" onClick={()=>{if(proposal==='separate')applyFix();else{setVariant('materialized');setAttempted(true);setSelected('out');setPanel('detail');}}}>{proposal==='separate'?'应用推荐方案 · 拆分 scope':'应用尝试 · 仅分离 buffer'}</button></div></div></section>:panel==='diff'?<section className="object-block"><h2>Before → 方案对比</h2><div className="settings-options">{[['separate','拆分 scope · 推荐'],['materialized','仅分离 buffer']].map(([id,label])=><button key={id} aria-pressed={proposal===id} onClick={()=>setProposal(id)}>{label}</button>)}</div><div className="visual-diff"><div className="diff-head"><span>变化项</span><strong>Before · 当前问题</strong><strong>{proposal==='separate'?'方案：拆分 scope':'方案：仅分离 buffer'}</strong></div><div className="diff-row"><span>Scope</span><div className="scope-mini"><small>hc_post</small><b>pre / post / comb</b><b className="diff-problem">mix_x</b></div><div>{proposal==='separate'?<><div className="scope-mini"><small>hc_post</small><b>pre / post / comb</b></div><div className="scope-mini diff-changed"><small>独立 hc_mix_x</small><b>mix_x</b></div></>:<div className="scope-mini"><small>hc_post · 保持融合</small><b>pre / post / comb / mix_x</b></div>}</div></div><div className="diff-row"><span>UB 绑定</span><div className="binding-mini"><small>pre_0 / pre_1 / pre_2 / pre_3</small><span>↓ 共用</span><b>74304</b></div><div className="binding-mini"><small>每个 scale 独立地址</small><span>↓ 分别绑定</span><div className="address-chips">{(proposal==='separate'?[32768,32832,32896,32960]:[512,576,640,704]).map(address=><b key={address}>{address}</b>)}</div></div></div><div className="diff-row diff-verdict"><span>精度结果</span><div>⚠ FAIL <small>97.94% 元素错误</small></div><div>{proposal==='separate'?'✓ PASS':'⚠ 仍 FAIL'}<small>{proposal==='separate'?'拆分 scope 后通过':'约 97.9% 错误，未解决'}</small></div></div></div><p className="diff-takeaway">{proposal==='separate'?'改动：把 mix_x 移出融合 scope。结果：独立地址，精度通过。':'改动：只给 scale 分配独立 buffer。结果：地址分开，输出仍错。'}</p><div className="repair-actions"><button className="isa-toggle" onClick={()=>setPanel('repair')}>返回选方案</button><button className="isa-toggle" onClick={()=>{if(proposal==='separate')applyFix();else{setVariant('materialized');setAttempted(true);setSelected('out');setPanel('detail');}}}>应用此方案</button></div><small>结构与结果来自 Issue 对照记录。</small></section>:panel==='settings'?<><h3>染色方案</h3><div className="settings-options">{[['identity','对象类别'],['semantic','语义角色']].map(([id,label])=><button key={id} aria-pressed={palette===id} onClick={()=>setPalette(id)}>{label}</button>)}</div><h3>布局方向</h3><div className="settings-options">{[['horizontal','横向'],['vertical','纵向']].map(([id,label])=><button key={id} aria-pressed={layout===id} onClick={()=>{setLayout(id);setOverrides({});}}>{label}</button>)}</div></>:panel==='info'?<><h2>逻辑值 → UB 绑定</h2><p>Tensor、计算、scope、存储图元沿用已批准版本。蓝色表示当前内存选择，四个 head 通过标签区分。</p><p>UB 网格不表示容量或 bank。地址块等宽，用于检查身份；大小未知，不绘制虚构区间。</p><p>共址不等同于已证明的生命周期冲突。当前没有完整 live range，也没有逐 Pass dump。</p><a href={source} target="_blank" rel="noreferrer">Issue #1723 ↗</a></>:<section className="object-block"><h2>{selectedNode?.data.label||checkpoint.name}</h2><p>{variant.detail}</p>{!fixed&&<><p className="case-alert">⚠ {attempted?'地址已分离，但输出仍失败。':'4 个逻辑 scale → 同一 UB 地址 74304。'}</p><button className="isa-toggle" onClick={()=>setPanel('repair')}>选择修复方式</button></>}{fixed&&<p>✓ After：采用独立 scope 的已报告通过记录。不是对原编译器缺陷已修复的声明。</p>}<div className="impact-state"><i/>FACT · Issue 报告的地址与数值结果</div>
 {allocation!==null?<dl className="object-data"><dt>UB 起始地址</dt><dd>{allocation}</dd><dt>绑定值</dt><dd>{heads.map(h=>'pre_'+h).join(' / ')||'当前实现无此绑定'}</dd><dt>大小 / 对齐</dt><dd>待获取</dd><dt>生命周期</dt><dd>待获取；不能判定同时存活</dd></dl>:<dl className="object-data"><dt>输出</dt><dd>x_mixed · {variant.result}</dd><dt>其他输出</dt><dd>{variant.id==='separate'?'位于其他 scope':'post PASS / comb PASS'}</dd><dt>四路地址</dt><dd>{variant.addresses.join(' / ')}</dd></dl>}
 <h3>当前检查点</h3><p>{checkpoint.detail}</p><h3>待验证</h3><p>transpose → subview / reshape → row_expand_mul 的 layout、有效区域与依赖。强制分离地址后仍错误，完整根因尚未确认。</p><details><summary>生命周期与来源</summary><p>创建、写入、读取、最后使用：待获取原始产物。此图按 Issue 重建语义结构，未执行硬件复验。</p><p>编译版本：7fe857a895078333dd95fdc8ab915f733664c063；平台 a2a3。地址是报告的起始地址，不是完整区间。</p><a href={source} target="_blank" rel="noreferrer">查看 Issue #1723 ↗</a></details></section>}
 </aside>}
 <PassRail currentIndex={passIndex} onSelect={setPassIndex} playing={playing} onTogglePlay={()=>setPlaying(v=>!v)} onReset={()=>{setPlaying(false);setPassIndex(1);}} evidenceMode caseWarning caseResolved={fixed} />
 <p className="case-pipeline-note">完整参考管线 · CP-01 逐 Pass 快照待接入</p>
 </section></main></DiagramSettings.Provider>;
}
