import { tensorGeometry } from './tensor-glyph';
import { routeOrthogonal } from './orthogonal-route';
import snapshots from '../data/memory.json';
import './memory-field.css';
export const memoryOwner = value => {
 const text=value?.name||'';
 const expr=value?.expression||'';
 if(/tile\.matmul/.test(expr))return /pv|oi/.test(text)?'op.pv':'op.qk';
 if(/exp_bf16/.test(text))return 'value.prob';
 if(/exp/.test(text))return 'op.exp';
 if(/scores|scaled/.test(text))return 'value.scores';
 if(/mi_|max/.test(text))return 'op.rowmax';
 if(/li_|sum/.test(text))return 'op.rowsum';
 if(/prob/.test(text))return 'value.prob';
 if(/oi_|pv_/.test(text))return 'op.pv';
 return null;
};
export const memoryOwners = (value,row) => {
 if(!row.kernel.endsWith('aic'))return [memoryOwner(value)].filter(Boolean);
 const name=value.name||'',expr=value.expression||'';
 if(/^qk_kv_/.test(name))return ['value.kv','op.qk','op.pv'];
 if(/^qk_q_tile_/.test(name))return ['value.q','op.qk'];
 if(/^qk_raw_/.test(name))return ['op.qk'];
 if(/^qk_oi_/.test(name)||/qk_oi_/.test(expr))return ['op.pv'];
 return [memoryOwner(value)].filter(Boolean);
};
const kb=n=>`${Number((n/1024).toFixed(2))} KiB`;
export const memoryRows = pass => pass < 33 ? [] : snapshots[Math.min(pass,36)]?.rows || [];
export function memoryProjection(flow, pass, selectedId, layout, overrides, hoverId = null) {
 const instructionSelection=flow.nodes.find(n=>n.id===selectedId || n.data.instances?.some(i=>i.id===selectedId));
 const bindingId=instructionSelection?.data.owner || selectedId;
 const positions={
 'task.plan':[400,-100], 'task.qk-pv':[1000,-100],
 'kernel.aic':[80,0], 'kernel.aiv':[80,420],
 'value.q':[-100,65], 'value.kv':[-100,185],
 'op.qk':[180,70], 'op.pv':[1430,70], 'value.out':[1850,70],
 'handoff.c2v-push':[480,70], 'handoff.c2v-pop':[480,470],
 'handoff.v2c-push':[1210,470], 'handoff.v2c-pop':[1210,70],
 'value.scores':[220,670], 'op.rowmax':[420,678], 'value.max':[420,680],
 'op.exp':[740,678], 'value.prob':[1040,670], 'op.rowsum':[1200,678], 'value.sum':[1510,670],
 };
 const rows=memoryRows(pass).map(row=>({...row,owners:[...new Set(row.values.flatMap(value=>memoryOwners(value,row)).filter(id=>id&&positions[id]))]}));
 const selected=rows.find(row=>`allocation.${row.id}`===selectedId);
 const related=new Set(selected ? selected.owners : rows.filter(row=>row.owners.includes(bindingId)).flatMap(row=>row.owners));
 const nodes=flow.nodes.filter(n=>positions[n.id]).map(n=>({...n,position:{x:positions[n.id][0],y:positions[n.id][1]},className:related.has(n.id)?'memory-related':'',style:{...n.style,width:n.type==='kernel'?1700:n.type==='task'?340:n.type==='tensor'?110:260,height:n.type==='kernel'?390:n.type==='tensor'?90:n.type==='task'?54:48},data:{...n.data,selected:n.id===selectedId,...(n.type==='kernel'?{role:''}:{})}}));
 for(const node of nodes){
  if(node.type==='tensor'){const geometry=tensorGeometry(node.data);node.style.width=geometry.width;node.style.height=geometry.height;}
  if(node.type==='op'||node.type==='handoff'){
   const label=node.data.label.replace(/ · slot.*$/,'');
   const context=document.createElement('canvas').getContext('2d');
   context.font=node.type==='handoff'?'620 14px "SFMono-Regular", Consolas, "Liberation Mono", monospace':'600 16px Arial, "PingFang SC", sans-serif';
   const textWidth=context.measureText(label).width;
   node.style.width=Math.ceil(textWidth+58);
  }
 }
 // Lay out complete rows from measured glyph widths, never from stale fixed x positions.
 const byId=new Map(nodes.map(node=>[node.id,node]));
 const placeRow=(ids,start,centerY,gap=72)=>{
  let x=start;
  for(const id of ids){const node=byId.get(id);if(!node)continue;node.position={x,y:centerY-node.style.height/2};x+=node.style.width+gap;}
  return x-gap;
 };
 const mainEnd=placeRow(['value.scores','op.rowmax','op.exp','value.prob','op.rowsum','value.sum'],180,582);
 placeRow(['op.qk','handoff.c2v-push','handoff.v2c-pop','op.pv'],180,94);
 const center=id=>{const n=byId.get(id);return n.position.x+n.style.width/2;};
 for(const [id,anchor] of [['handoff.c2v-pop','handoff.c2v-push'],['handoff.v2c-push','value.prob'],['value.max','op.rowmax']]){
  const n=byId.get(id);n.position.x=center(anchor)-n.style.width/2;
 }
 const frameWidth=Math.max(1400,mainEnd-80+100);
 for(const id of ['kernel.aic','kernel.aiv'])byId.get(id).style.width=frameWidth;
 byId.get('value.out').position.x=80+frameWidth+72;
 for(const node of nodes){
  const bindings=rows.filter(row=>row.owners.includes(node.id)&&(bindingId===node.id||selected?.id===row.id));
  if(bindings.length)node.data.memoryRanges=bindings.map(row=>`${row.space} · ${row.offset==null?row.base:`[${row.offset}, ${row.offset+row.size})`}`);
 }
 const ids=new Set(nodes.map(n=>n.id));
 const edges=flow.edges.filter(e=>ids.has(e.source)&&ids.has(e.target)).map(e=>({...e,className:'',data:{...e.data,focused:e.data?.focused || e.source===selectedId || e.target===selectedId}}));
 const groups=Object.groupBy(rows,row=>`${row.kernel}:${row.space}`);
 const storage={Mat:{label:'L1',x:180,y:180,width:190,height:155},Left:{label:'L0A',x:450,y:180,width:220,height:76},Right:{label:'L0B',x:450,y:276,width:220,height:76},Acc:{label:'L0C',x:850,y:180,width:220,height:155},Vec:{label:'UB',x:180,y:800,width:mainEnd-180,height:100}};
 for(const [key,group] of Object.entries(groups)){
  const unit=storage[group[0].space];
  if(!unit)continue;
  const {x,y,width,height}=unit;
  const addressed=pass>=36&&group.every(row=>row.offset!=null);
  const extent=Math.max(...group.map(row=>(row.offset||0)+row.size));
  const highlights=[];
  nodes.push({id:`space.${key}`,type:'memory',selectable:false,draggable:false,position:{x,y},style:{width,height},data:{spaceContainer:true,label:unit.label,gridWidth:width-32,gridHeight:height-52,highlights}});
  group.forEach((row,i)=>{
   const id=`allocation.${row.id}`,active=selectedId===id||row.owners.includes(bindingId);
   const inner=width-32,bodyHeight=height-52,columns=unit.label==='UB'?12:unit.label==='L1'||unit.label==='L0C'?2:4;
   const rowCount=Math.ceil(group.length/columns);
   const bx=addressed&&unit.label==='UB'?row.offset/extent*inner:addressed?0:(i%columns)*inner/columns;
   const by=addressed&&unit.label!=='UB'?row.offset/extent*bodyHeight:addressed?0:Math.floor(i/columns)*bodyHeight/rowCount;
   const bw=addressed?(unit.label==='UB'?Math.max(1,row.size/extent*inner-1):inner):inner/columns-4;
   const bh=addressed?(unit.label==='UB'?bodyHeight:Math.max(1,row.size/extent*bodyHeight-1)):bodyHeight/rowCount-4;
   if(active)highlights.push({x:bx,y:by,width:bw,height:bh});
   nodes.push({id,type:'memory',position:{x:x+16+bx,y:y+36+by},style:{width:bw,height:bh},data:{compactBlock:true,gridWidth:bw,gridHeight:bh,base:row.base,label:row.base,shared:row.values.length>1,selected:selectedId===id,touched:active,description:`${unit.label} · ${row.base} · ${kb(row.size)} · ${row.values.length} 个绑定值${addressed?` · [${row.offset}, ${row.offset+row.size})`:''}`}});
   if(active)for(const owner of row.owners)edges.push({id:`${id}:${owner}`,source:owner,target:id,sourceHandle:'memory',targetHandle:'in',type:'memoryBinding',zIndex:20,data:{focused:true,compact:true}});
  });
 }
 // ISA extends the same storage/computation canvas; it does not restore aggregate buffers.
 const instructionNodes=flow.nodes.filter(n=>n.type==='isa');
 const placedInstructions=[];
 for(const original of instructionNodes){
  const owner=nodes.find(n=>n.id===original.data.owner);
  if(!owner)continue;
  const width=original.style.width,height=original.style.height;
  const x=owner.position.x+owner.style.width/2-width/2;
  let y=950;
  let collision;
  do {
   collision=placedInstructions.find(placed=>x<placed.position.x+placed.style.width+24&&x+width+24>placed.position.x&&y<placed.position.y+placed.style.height+24&&y+height+24>placed.position.y);
   if(collision)y=collision.position.y+collision.style.height+24;
  } while(collision);
  const node={...original,position:{x,y},style:{...original.style}};
  nodes.push(node);placedInstructions.push(node);
 }
 const allIds=new Set(nodes.map(n=>n.id));
 edges.push(...flow.edges.filter(e=>instructionNodes.some(n=>n.id===e.target)&&allIds.has(e.source)).map(e=>({...e,data:{...e.data}})));
 const aiv=nodes.find(n=>n.id==='kernel.aiv');
 if(aiv)aiv.style.height=510;
 for(const node of nodes){
  if(layout==='vertical')node.position={x:node.position.y,y:node.position.x};
  if(overrides[node.id])node.position=overrides[node.id];
 }
 // Keep dragged computation glyphs clear of storage plates as well as default positions.
 const plates=nodes.filter(node=>node.data.spaceContainer);
 for(const node of nodes.filter(node=>['op','tensor','handoff'].includes(node.type))){
  for(const plate of plates){
   const gap=28,w=node.style.width,h=node.style.height;
   if(node.position.x+w>plate.position.x-gap&&node.position.x<plate.position.x+plate.style.width+gap&&node.position.y+h>plate.position.y-gap&&node.position.y<plate.position.y+plate.style.height+gap){
    if(layout==='vertical')node.position.x=plate.position.x-w-gap;
    else node.position.y=plate.position.y-h-gap;
   }
  }
 }
 const lookup=new Map(nodes.map(n=>[n.id,n]));
 const portPoint=(n,side)=>({x:n.position.x+(side==='left'?0:side==='right'?n.style.width:n.style.width/2),y:n.position.y+(side==='top'?0:side==='bottom'?n.style.height:n.style.height/2)});
 for(const edge of edges){
  const source=lookup.get(edge.source),target=lookup.get(edge.target);
  if(!source||!target)continue;
  if(instructionNodes.some(n=>n.id===edge.target)){
   edge.sourceHandle='bottom-out';edge.targetHandle='top-in';
   edge.data={...edge.data,instructionLink:true,hoverFocused:hoverId===edge.source||hoverId===edge.target||target.data.instances?.some(i=>i.id===hoverId),routePoints:[portPoint(source,'bottom'),portPoint(target,'top')]};
   edge.zIndex=edge.data.hoverFocused?1000:2;
   continue;
  }
  let from='right',to='left';
  if(edge.type==='memoryBinding'||target.position.y-source.position.y>130){from='bottom';to='top';}
  if(edge.source==='value.max'){from='right';to='bottom';}
  if(edge.source==='value.scores'&&edge.target==='op.exp'){from='bottom';to='bottom';}
  if(target.position.x<source.position.x&&from==='right'){from='left';to='right';}
  edge.sourceHandle=`${from}-out`;edge.targetHandle=`${to}-in`;
  const obstacles=nodes.filter(n=>n.type!=='kernel'&&!n.data.compactBlock&&!(n.data.spaceContainer&&target.data.compactBlock&&target.position.x>=n.position.x&&target.position.x<n.position.x+n.style.width&&target.position.y>=n.position.y&&target.position.y<n.position.y+n.style.height)).map(n=>({x:n.position.x,y:n.position.y-(n.type==='tensor'?20:0),width:n.style.width,height:n.style.height+(n.type==='tensor'?20:0)}));
  edge.data={...edge.data,routePoints:routeOrthogonal(portPoint(source,from),portPoint(target,to),from,to,obstacles)};
 }
 return {nodes,edges};
}
export function MemoryDetails({row,passIndex,onClose}){
 const snapshot=snapshots[Math.min(passIndex,36)];
 return <aside className="inspector overlay-panel inspector-right" data-liquid-glass="panel" role="dialog" aria-label="分配详情">
 <header className="inspector-header"><span>{row.space} · {row.base}</span><button onClick={onClose} aria-label="关闭分配详情">×</button></header>
 <section className="object-block"><h2>{row.base}</h2><dl className="object-data"><dt>大小</dt><dd>{kb(row.size)}</dd><dt>地址</dt><dd>{row.offset==null?'尚未分配':`[${row.offset}, ${row.offset+row.size})`}</dd><dt>所属函数</dt><dd>{row.kernel}</dd><dt>绑定值</dt><dd>{row.values.length}</dd></dl>
 <details open><summary>相关计算值</summary>{row.values.map(v=><p key={v.name+v.line}><code>{v.name}</code></p>)}</details>
 <details><summary>源码与映射依据</summary><p>{snapshot.file} · L{row.line}</p>{row.values.map(v=><pre className="instruction-source" key={v.name+v.line}>L{v.line} {v.expression}</pre>)}<p>Tile 与分配关系取自 IR；主图按变量与操作语义归并到现有计算节点，只展开已建立对应关系的分配。绑定不表示值同时存活。</p>{passIndex>36&&<p>当前保留 P36 分配快照。</p>}</details></section>
 </aside>;
}
