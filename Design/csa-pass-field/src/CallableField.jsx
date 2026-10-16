import React from 'react';
import snapshots from '../data/callable/snapshots.json';
import {tensorGeometry} from './tensor-glyph.js';
import {instructionCollections} from './instruction-collections.js';
import isa from '../data/callable/isa.json';
import './memory-field.css';

export const snapshotAt=index=>snapshots[Math.min(51,Math.max(0,index))];
export const memoryRows=index=>snapshotAt(index).rows;
const compactWidth=label=>{const context=document.createElement('canvas').getContext('2d');context.font='600 16px Arial, "PingFang SC", sans-serif';return Math.ceil(context.measureText(label).width+62);};
const stageIds=['op.merge','op.normalize','op.rope','op.pack','handoff.publish','handoff.notify'];
const rectPort=(node,side)=>({x:node.position.x+(side==='left'?0:side==='right'?node.style.width:node.style.width/2),y:node.position.y+(side==='top'?0:side==='bottom'?node.style.height:node.style.height/2)});

// Navigation and playback share one comparison of default, visible graph semantics.
// Source locations, SSA names, selection and inspector-only metadata are excluded.
export function visibleSnapshotSignature(snapshot){
 return JSON.stringify({
  nodes:snapshot.nodes.map(n=>{
   if(n.regionOperations){
    const counts=new Map();for(const statement of n.regionOperations)for(const opcode of statement.operations)counts.set(opcode,(counts.get(opcode)||0)+1);
    return [n.id,n.label,n.attributes.domain,[...counts]];
   }
   if(n.renderKind==='kernel')return [n.id,n.label,n.attributes.target,n.attributes.objectKind,n.attributes.detail];
   if(n.renderKind==='tensor')return [n.id,n.label,n.attributes.shape,n.attributes.dtype];
   return [n.id,n.label];
  }),
  controls:(snapshot.nodes.find(n=>n.id==='kernel.main')?.sourceFacts?.loops||[]).map(loop=>[
   (loop.target.match(/[A-Za-z_]\w*/)?.[0]||'').replace(/_inline.*$|__.*$/,''),
   loop.iterator.replace(/_inline\d+__\w+_v\d+/g,'').replace(/__\w+_v\d+/g,'').split(', init_values=')[0].replace(/^pl\./,'')
  ]),
  edges:snapshot.edges.map(e=>[e.source,e.target,e.semanticKind]),
  allocations:snapshot.rows.map(r=>[r.id,r.offset,r.size])
 });
}
export const visibleChangedPasses=snapshots.slice(1).flatMap((snapshot,i)=>visibleSnapshotSignature(snapshot)!==visibleSnapshotSignature(snapshots[i])?[i+1]:[]);

export function callableFlow(state,{currentPass,selectedId,hoverId,showIsa,positionOverrides,layout,loadExpanded,loadData}){
 const snapshot=snapshotAt(currentPass.index),rows=snapshot.rows;
 const collections=showIsa?instructionCollections(isa.instructions):[];
 const chosenInstruction=collections.find(c=>c.id===selectedId||c.instances.some(i=>i.id===selectedId));
 const expanded=state.nodes.find(n=>n.id===loadData.expandedRegion);
 const buckets=[];
 for(const owner of state.nodes.filter(n=>n.regionOperations)){
  const grouped=new Map();
  for(const statement of owner.regionOperations)for(const opcode of statement.operations){
   if(!grouped.has(opcode))grouped.set(opcode,{id:'bucket.'+owner.id+'.'+opcode,owner:owner.id,opcode,statements:[],count:0});
   const bucket=grouped.get(opcode);bucket.count++;if(!bucket.statements.includes(statement))bucket.statements.push(statement);
  }
  buckets.push(...grouped.values());
 }
 const selectedBucket=buckets.find(b=>b.id===selectedId);
 const selectedOperation=expanded?.regionOperations.find(item=>'detail.'+expanded.id+'.'+item.line===selectedId);
 const selectedOwner=chosenInstruction?.owner||selectedBucket?.owner||(selectedOperation?expanded.id:selectedId);
 const selectedValues=selectedBucket?new Set(selectedBucket.statements.map(s=>s.value)):null;
 const maxBucketRows=Math.max(...stageIds.map(id=>buckets.filter(b=>b.owner===id).length));
 const controls=snapshot.nodes.find(n=>n.id==='kernel.main')?.sourceFacts?.loops||[];
 const bodyBottom=400+maxBucketRows*78;
 const verticalRows=stageIds.map(id=>Math.ceil(buckets.filter(b=>b.owner===id).length/3));
 const verticalOffsets=[];let verticalCursor=240;
 verticalRows.forEach(rows=>{verticalOffsets.push(verticalCursor);verticalCursor+=140+rows*78;});
 const replayRow=loadData.memoryReplayId;
 const selectedRow=rows.find(r=>'allocation.'+r.id===(replayRow||selectedId));
 const positions={
 'kernel.main':[40,70],'task.main':[670,-40],
 'value.mi':[-170,170],'value.li':[-170,270],'value.oi':[-170,370],
 'value.sink':[410,340],'value.cos':[650,340],'value.swap':[650,440],
 'value.grouped':[940,340],'value.window':[1250,340],'value.signal':[1480,440],
 };
 stageIds.forEach((id,i)=>positions[id]=[160+i*310,260]);
 ['mi','li','oi','sink','cos','swap'].forEach((key,i)=>positions['value.'+key]=[-180,200+i*110]);
 ['grouped','window','signal'].forEach((key,i)=>positions['value.'+key]=[2100,260+i*180]);
 if(layout==='vertical'){
  Object.assign(positions,{'kernel.main':[300,0],'task.main':[680,-110],'value.mi':[400,180],'value.li':[400,280],'value.oi':[400,380],'value.sink':[400,480],'value.cos':[400,710],'value.swap':[400,810],'value.grouped':[1040,930],'value.window':[1040,1170],'value.signal':[1040,1410]});
  stageIds.forEach((id,i)=>positions[id]=[720,verticalOffsets[i]]);
 ['mi','li','oi','sink','cos','swap'].forEach((key,i)=>positions['value.'+key]=[340,250+i*120]);
 ['grouped','window','signal'].forEach((key,i)=>positions['value.'+key]=[1600,verticalOffsets[i+3]]);
 }
 const contexts=state.nodes.filter(n=>n.id.startsWith('context.'));
 contexts.forEach((n,i)=>positions[n.id]=layout==='vertical'?[650+i*330,verticalCursor-30]:[1390+i*310,bodyBottom-10]);
 const nodes=state.nodes.map(n=>{
  let size=n.renderKind==='kernel'?{width:layout==='vertical'?1200:1960,height:layout==='vertical'?verticalCursor+220:bodyBottom+260}:n.renderKind==='task'?{width:360,height:54}:n.renderKind==='tensor'?tensorGeometry(n):n.id.startsWith('context.')?{width:285,height:68}:{width:150,height:48};
  if(['op','handoff'].includes(n.renderKind)){
   const context=document.createElement('canvas').getContext('2d');context.font='600 16px Arial, "PingFang SC", sans-serif';size.width=Math.ceil(context.measureText(n.label).width+58);
  }
  const label=n.regionOperations?n.label+' · '+n.attributes.domain:n.label;
  if(n.regionOperations)size.width=compactWidth(label);
  return {id:n.id,type:n.renderKind,position:{x:positions[n.id][0],y:positions[n.id][1]},style:{width:size.width,height:size.height,zIndex:n.renderKind==='kernel'?0:3},data:{...n,label,selected:n.id===selectedId,touched:currentPass.affectedObjectIds?.includes(n.id),involvement:[]},selectable:true,draggable:n.renderKind!=='kernel'};
 });
 const edges=state.edges.map(e=>({...e,type:e.semanticKind,data:{...e,focused:[e.source,e.target].includes(hoverId||selectedId)},zIndex:[e.source,e.target].includes(hoverId||selectedId)?1000:2}));
 // Actual operator buckets are visible by default; source statements remain expandable.
 for(const ownerId of stageIds){
  const owner=nodes.find(n=>n.id===ownerId),list=buckets.filter(b=>b.owner===ownerId);
  list.forEach((bucket,i)=>{
   const label=bucket.opcode.replace(/^pld?\./,'')+' ×'+bucket.count;
   const position=layout==='vertical'?{x:550+i%3*300,y:owner.position.y+90+Math.floor(i/3)*78}:{x:owner.position.x,y:350+i*78};
   nodes.push({id:bucket.id,type:bucket.opcode.startsWith('pld.')||bucket.opcode==='pl.system.fence'?'handoff':'op',position,style:{width:compactWidth(label),height:48,zIndex:3},data:{id:bucket.id,label,owner:ownerId,renderKind:'op',semanticKind:'operator_group',role:bucket.count+' 处静态算子 · '+owner.data.label,attributes:{opcode:bucket.opcode,count:bucket.count,operations:bucket.count},regionOperations:bucket.statements,sourceRefs:bucket.statements.slice(0,4).map(item=>({...owner.data.sourceRefs[0],line:item.line})),selected:bucket.id===selectedId,touched:currentPass.affectedObjectIds?.includes(ownerId),involvement:[]}});
  });
 }
 controls.forEach((loop,i)=>{
  const name=(loop.target.match(/[A-Za-z_]\w*/)?.[0]||String(i)).replace(/_inline.*$|__.*$/,'');
  const form=loop.iterator.match(/pl\.(\w+)/)?.[1]||'for';
  const id='control.'+name+'.'+i;
  nodes.push({id,type:'scope',position:layout==='vertical'?{x:320+i%3*350,y:80+Math.floor(i/3)*76}:{x:140+i*310,y:125},style:{width:280,height:68,zIndex:2},data:{id,label:form+' · '+name,bounds:loop.iterator.replace(/_inline\d+__\w+_v\d+/g,'').replace(/__\w+_v\d+/g,'').split(', init_values=')[0].replace(/^pl\./,''),semanticKind:'loop',renderKind:'scope',attributes:{level:'CONTROL',iterator:loop.iterator},sourceText:loop.iterator,role:loop.target,sourceRefs:[{file:snapshot.file,line:loop.line}]}});
 });
 const fence=buckets.find(b=>b.opcode==='pl.system.fence');
 if(fence){
  const put=buckets.find(b=>b.owner===fence.owner&&['pld.tile.put','pld.tensor.put'].includes(b.opcode));
  for(const [source,target] of [[put?.id,fence.id],[fence.id,'handoff.notify']])if(source)edges.push({id:'ordering.'+source+'.'+target,source,target,type:'control',data:{hoverOnly:true,hoverFocused:[source,target].includes(hoverId||selectedId)},zIndex:1000});
 }
 const latestValue=new Map();
 for(const ownerId of stageIds){
  const region=state.nodes.find(n=>n.id===ownerId);
  for(const statement of [...region.regionOperations].sort((a,b)=>a.line-b.line)){
   const target='bucket.'+ownerId+'.'+statement.operations[0];
   for(const read of statement.reads||[]){const source=latestValue.get(read);if(source&&source!==target)edges.push({id:'dependency.'+source+'.'+target+'.'+read,source,target,type:'data',data:{hoverOnly:true,hoverFocused:[source,target].includes(hoverId||selectedId)},zIndex:1000});}
   if(statement.value)latestValue.set(statement.value,target);
  }
 }
 if(rows.length){
  const x=layout==='vertical'?420:150,y=layout==='vertical'?verticalCursor+50:bodyBottom+90,width=layout==='vertical'?1000:1810,height=160,gridWidth=width-32,gridHeight=80;
  const highlights=[];
  nodes.push({id:'space.Vec',type:'memory',selectable:false,draggable:false,position:{x,y},style:{width,height,zIndex:0},data:{spaceContainer:true,gridTop:64,label:'UB · '+rows.length+' 分配',playback:loadData.memoryPlayback,gridWidth,gridHeight,highlights}});
  const extent=Math.max(...rows.map(r=>(r.offset||0)+r.size));
  rows.forEach((row,i)=>{
   const addressed=row.offset!=null;
   const bx=addressed?row.offset/extent*gridWidth:(i%12)*gridWidth/12;
   const by=addressed?0:Math.floor(i/12)*gridHeight/Math.ceil(rows.length/12);
   const bw=addressed?row.size/extent*gridWidth:gridWidth/12-3;
   const bh=addressed?gridHeight:gridHeight/Math.ceil(rows.length/12)-3;
   const owner=selectedOwner===row.id?null:selectedOwner;
   const active=selectedRow?.id===row.id||(!replayRow&&((selectedOperation?row.values.some(v=>v.name===selectedOperation.value):selectedValues?row.values.some(v=>selectedValues.has(v.name)):row.owners.includes(owner))));
   if(active){
    if(addressed&&!selectedRow){
     for(const value of row.values.filter(v=>(selectedOperation?v.name===selectedOperation.value:selectedValues?selectedValues.has(v.name):v.owner===owner)&&v.offset!=null))highlights.push({x:value.offset/extent*gridWidth,y:0,width:value.size/extent*gridWidth,height:gridHeight});
    }else highlights.push({x:bx,y:by,width:bw,height:bh});
   }
   const id='allocation.'+row.id;
   nodes.push({id,type:'memory',position:{x:x+16+bx,y:y+64+by},style:{width:bw,height:bh,zIndex:5},data:{compactBlock:true,base:row.base,shared:row.values.length>1,touched:active,selected:id===selectedId,description:row.base+' · '+row.size+' B'+(addressed?' · ['+row.offset+', '+(row.offset+row.size)+')':'')}});
   if(active&&!replayRow)for(const targetOwner of selectedRow?row.owners:[owner]){
    edges.push({id:id+':'+targetOwner,source:selectedBucket?selectedBucket.id:targetOwner,target:id,type:'memoryBinding',data:{focused:true,compact:true},zIndex:1000});
   }
  });
 }
 let detailBottom=layout==='vertical'?verticalCursor+240:bodyBottom+270;
 if(expanded){
  const operations=expanded.regionOperations||[],lookupValue=new Map();
  const detailNodes=[];
  operations.forEach((operation,i)=>{
   const id='detail.'+expanded.id+'.'+operation.line;
   const label=operation.operations[0].replace(/^pld?\./,'');
   const data={id,label,semanticKind:'op',renderKind:'op',owner:expanded.id,role:operation.value,attributes:{operations:operation.operations.join(', ')},sourceText:operation.expression,sourceRefs:expanded.sourceRefs.map(ref=>({...ref,line:operation.line})),selected:id===selectedId};
   const position=layout==='vertical'?{x:1700+i%4*240,y:verticalCursor+300+Math.floor(i/4)*100}:{x:150+i%6*250,y:bodyBottom+300+Math.floor(i/6)*100};
   const node={id,type:'op',position,style:{width:220,height:48,zIndex:3},data};
   detailNodes.push(node);
   for(const read of operation.reads||[]){const source=lookupValue.get(read);if(source&&source!==id)edges.push({id:source+'->'+id+':'+read,source,target:id,type:'data',data:{focused:[source,id].includes(hoverId||selectedId)}});}
   if(operation.value)lookupValue.set(operation.value,id);
  });
  nodes.push(...detailNodes);
  detailBottom=Math.max(detailBottom,...detailNodes.map(n=>n.position.y+100));
 }
 // Replay follows the actual bound value names, including consuming operators.
 // Keep UI selection independent so pausing does not open/change the Inspector.
 if(replayRow&&selectedRow){
  const values=new Set(selectedRow.values.map(v=>v.name));
  const related=buckets.filter(bucket=>bucket.statements.some(statement=>values.has(statement.value)||(statement.reads||[]).some(read=>values.has(read))));
  const relatedIds=new Set(related.map(bucket=>bucket.id));
  for(const n of nodes){
   if(relatedIds.has(n.id)){n.data.selected=true;n.data.replayRelated=true;}
   if(n.id.startsWith('detail.')){
    const statement=expanded?.regionOperations.find(item=>'detail.'+expanded.id+'.'+item.line===n.id);
    if(statement&&(values.has(statement.value)||(statement.reads||[]).some(read=>values.has(read)))){n.data.selected=true;n.data.replayRelated=true;relatedIds.add(n.id);}
   }
  }
  for(const source of relatedIds)edges.push({id:'replay.'+source+'.'+selectedRow.id,source,target:'allocation.'+selectedRow.id,type:'memoryBinding',data:{focused:true,compact:true},zIndex:1000});
 }
 // Keep each instruction set centered below its source region; pack vertically without overlaps.
 const placed=nodes.filter(n=>n.id.startsWith('detail.'));
 for(const c of collections){
  const owner=nodes.find(n=>n.id===c.owner);if(!owner)continue;
  const width=c.id===loadExpanded?920:310,height=c.id===loadExpanded?210+Math.ceil(c.instances.length/3)*76+176:56;
  let x=layout==='vertical'?2000:owner.position.x+owner.style.width/2-width/2;let y=layout==='vertical'?owner.position.y:detailBottom+60,collision;
  do{collision=placed.find(n=>x<n.position.x+n.style.width+24&&x+width+24>n.position.x&&y<n.position.y+n.style.height+24&&y+height+24>n.position.y);if(collision){if(layout==='vertical')x=collision.position.x+collision.style.width+24;else y=collision.position.y+collision.style.height+24;}}while(collision);
  const n={id:c.id,type:'isa',position:{x,y},style:{width,height,zIndex:c.id===loadExpanded?100:3},data:{...c,ownerLabel:owner.data.label,expanded:c.id===loadExpanded,onToggle:()=>loadData.onToggle(c.id),onInstanceSelect:loadData.onInstanceSelect,selected:c.id===selectedId}};
  nodes.push(n);placed.push(n);
  edges.push({id:'origin.'+c.id,source:c.owner,target:c.id,type:'lowering',data:{instructionLink:true,hoverFocused:hoverId===c.id||hoverId===c.owner},zIndex:hoverId===c.id||hoverId===c.owner?1000:2});
 }
 for(const n of nodes){
  if(positionOverrides[n.id])n.position=positionOverrides[n.id];
 }
 const lookup=new Map(nodes.map(n=>[n.id,n]));
 for(const e of edges){
  const a=lookup.get(e.source),b=lookup.get(e.target);if(!a||!b)continue;
  let from='right',to='left';
  if(e.data.instructionLink||e.type==='memoryBinding'||e.source==='task.main'){from='bottom';to='top';}
  else if(b.position.y>a.position.y+80){from='bottom';to='top';}
  else if(b.position.y<a.position.y-80){from='top';to='bottom';}
  const p=rectPort(a,from),q=rectPort(b,to);e.sourceHandle=from+'-out';e.targetHandle=to+'-in';
  e.data.routePoints=e.data.instructionLink?[p,q]:from==='bottom'||from==='top'?[p,{x:p.x,y:(p.y+q.y)/2},{x:q.x,y:(p.y+q.y)/2},q]:[p,{x:(p.x+q.x)/2,y:p.y},{x:(p.x+q.x)/2,y:q.y},q];
 }
 return {nodes,edges:[...new Map(edges.map(e=>[e.id,e])).values()]};
}
export function MemoryDetails({row,onClose}){
 return <aside className="inspector overlay-panel inspector-right" role="dialog" aria-label="分配详情"><header className="inspector-header"><span>UB · {row.base}</span><button onClick={onClose} aria-label="关闭分配详情">×</button></header><section><h2>{row.base}</h2><p>{row.size.toLocaleString()} B {row.offset!=null?' · ['+row.offset+', '+(row.offset+row.size)+')':''}</p><p>{row.values.length} 个绑定值 · {row.owners.length} 个计算区域</p><details><summary>绑定值与地址</summary>{row.values.map((v,i)=><p key={i}><code>{v.name}</code><br/>{v.offset!=null?'['+v.offset+', '+(v.offset+v.size)+')':''}</p>)}</details></section></aside>;
}
