import {build} from 'esbuild';
import fs from 'node:fs';
import assert from 'node:assert/strict';
const result=await build({entryPoints:['src/CallableField.jsx'],bundle:true,platform:'node',format:'esm',write:false,loader:{'.css':'empty'},logLevel:'silent'});
const module=await import('data:text/javascript;base64,'+Buffer.from(result.outputFiles[0].text).toString('base64'));
// Deterministic text measurement for geometry checks, not browser rendering.
globalThis.document={createElement:()=>({getContext:()=>({measureText:text=>({width:[...text].reduce((sum,ch)=>sum+(/[\u2e80-\uffff]/.test(ch)?16:9),0)})})})};
const snapshots=JSON.parse(fs.readFileSync('data/callable/snapshots.json'));
let checked=0;
for(let index=1;index<=52;index++)for(const layout of ['horizontal','vertical']){
 const state=snapshots[Math.min(index,51)];
 const options={currentPass:{index},selectedId:null,hoverId:null,showIsa:index===52,positionOverrides:{},layout,loadExpanded:false,loadData:{onToggle(){},onInstanceSelect(){}}};
 const flow=module.callableFlow(state,options);
 for(const n of flow.nodes)for(const value of [n.position.x,n.position.y,n.style.width,n.style.height])assert(Number.isFinite(value),n.id);
 const glyphs=flow.nodes.filter(n=>['op','handoff','tensor','isa','task','scope'].includes(n.type));
 for(let i=0;i<glyphs.length;i++)for(let j=i+1;j<glyphs.length;j++){
  const a=glyphs[i],b=glyphs[j];
  const overlaps=a.position.x<b.position.x+b.style.width&&a.position.x+a.style.width>b.position.x&&a.position.y<b.position.y+b.style.height&&a.position.y+a.style.height>b.position.y;
  assert(!overlaps,`Pass ${index} ${layout}: ${a.id} overlaps ${b.id}`);
 }
 for(const edge of flow.edges){assert(flow.nodes.some(n=>n.id===edge.source));assert(flow.nodes.some(n=>n.id===edge.target));}
 checked++;
}
for(const selectedId of ['op.merge','op.normalize','op.rope','op.pack','handoff.publish']){
 const flow=module.callableFlow(snapshots[36],{currentPass:{index:36},selectedId,hoverId:null,showIsa:false,positionOverrides:{},layout:'horizontal',loadExpanded:false,loadData:{}});
 assert(flow.nodes.find(n=>n.id==='space.Vec').data.highlights.length>0,selectedId);
 assert(flow.edges.some(e=>e.type==='memoryBinding'&&e.source===selectedId),selectedId);
}
for(const region of ['op.merge','op.normalize','op.rope','op.pack','handoff.publish','handoff.notify']){
 const options={currentPass:{index:36},selectedId:null,hoverId:null,showIsa:false,positionOverrides:{},layout:'horizontal',loadExpanded:false,loadData:{expandedRegion:region,onToggle(){},onInstanceSelect(){}}};
 const flow=module.callableFlow(snapshots[36],options),details=flow.nodes.filter(n=>n.id.startsWith('detail.'));
 assert(details.length>0,region);assert.equal(new Set(details.map(n=>n.id)).size,details.length);
 for(const detail of details)assert(detail.data.sourceText,detail.id);
}
console.log(`${checked} Pass/layout states checked; no glyph overlaps; UB highlights and six region expansions verified`);

const render=index=>module.callableFlow(snapshots[index],{currentPass:{index},selectedId:null,hoverId:null,showIsa:false,positionOverrides:{},layout:'horizontal',loadExpanded:false,loadData:{}});
const signature=flow=>JSON.stringify(flow.nodes.map(n=>[n.id,n.data.label,n.data.bounds,n.data.attributes?.target,n.data.attributes?.objectKind,n.position,n.style.width,n.style.height]));
for(const [before,after] of [[1,2],[10,11],[22,23],[32,33],[34,35],[35,36]])assert.notEqual(signature(render(before)),signature(render(after)),`Expected visible Pass transition ${before} -> ${after}`);
assert.equal(signature(render(2)),signature(render(3)),'True no-op must retain the visible graph');
for(const index of [1,2,11,23,33,35,36,51]){const flow=render(index);console.log('P'+index,flow.nodes.filter(n=>n.id.startsWith('bucket.')).length+' operator buckets',flow.nodes.filter(n=>n.id.startsWith('control.')).length+' loops',flow.nodes.find(n=>n.id==='space.Vec')?.data.label||'no explicit allocation');}
console.log('Visible pass transitions and no-op stability verified');

for(const index of [33,35,36])for(const row of snapshots[index].rows){
 const flow=module.callableFlow(snapshots[index],{currentPass:{index},selectedId:'op.merge',hoverId:null,showIsa:false,positionOverrides:{},layout:'horizontal',loadData:{memoryReplayId:'allocation.'+row.id}});
 assert.equal(flow.nodes.find(n=>n.id==='space.Vec').data.highlights.length,1);
 assert.equal(flow.nodes.filter(n=>n.data.compactBlock&&n.data.touched).length,1);
 assert.equal(new Set(flow.edges.map(e=>e.id)).size,flow.edges.length);
}
console.log('Allocation replay isolates one range at each step in P33/P35/P36');

const visible=module.visibleChangedPasses;
for(const index of [3,6,8])assert(!visible.includes(index),'No-op cannot be underlined or played: '+index);
const provenanceOnly=structuredClone(snapshots[4]);
for(const node of provenanceOnly.nodes){node.sourceRefs=[{line:999}];for(const op of node.regionOperations||[])op.expression+=' SSA rename';}
assert.equal(module.visibleSnapshotSignature(provenanceOnly),module.visibleSnapshotSignature(snapshots[4]));
console.log('Shared navigation/playback visible changes:',visible.join(', '));

assert(!module.visibleChangedPasses.includes(46));
for(const index of [45,50])assert(module.visibleChangedPasses.includes(index));
assert.equal(snapshots[44].nodes.filter(n=>n.id.startsWith('context.')).length,0);
assert.equal(snapshots[45].nodes.filter(n=>n.id.startsWith('context.')).length,2);
const fenceStatements=snapshots[50].nodes.find(n=>n.id==='handoff.publish').regionOperations.filter(s=>s.operations.includes('pl.system.fence'));
assert.equal(fenceStatements.length,2);
assert.equal(snapshots[49].nodes.find(n=>n.id==='handoff.publish').regionOperations.filter(s=>s.operations.includes('pl.system.fence')).length,0);
const finalFlow=render(50),fenceNode=finalFlow.nodes.find(n=>n.data.attributes?.opcode==='pl.system.fence');
assert(fenceNode);assert.equal(fenceNode.data.attributes.count,2);
assert(finalFlow.edges.some(e=>e.source===fenceNode.id&&e.target==='handoff.notify'));
console.log('P45 contexts and P50 two fences appear in graph and shared playback steps; P46 remains unchanged');

const {nextPassAction}=await import('../src/pass-playback.js');
for(const pass of [33,35,36]){const action=nextPassAction(pass,[33,35,36,45,50,52]);assert.equal(action.type,'pass');assert.equal(action.delay,1800);}
assert.equal(nextPassAction(52,[52]).type,'stop');
console.log('Pass timeline advances without waiting for UB allocation replay');
for(const pass of [33,35,36])for(const row of snapshots[pass].rows){
 const flow=module.callableFlow(snapshots[pass],{currentPass:{index:pass},selectedId:null,hoverId:null,showIsa:false,positionOverrides:{},layout:'horizontal',loadData:{memoryReplayId:'allocation.'+row.id}});
 const names=new Set(row.values.map(value=>value.name));
 const expected=flow.nodes.filter(n=>n.id.startsWith('bucket.')&&n.data.regionOperations.some(s=>names.has(s.value)||(s.reads||[]).some(name=>names.has(name))));
 assert(expected.length>0,'No linked op for '+row.id);
 for(const op of expected){assert(op.data.replayRelated&&op.data.selected);assert(flow.edges.some(e=>e.source===op.id&&e.target==='allocation.'+row.id&&e.data.focused&&e.zIndex===1000));}
 for(const op of flow.nodes.filter(n=>n.id.startsWith('bucket.')&&!expected.includes(n)))assert(!op.data.replayRelated);
}
console.log('Every allocation replay step highlights exact producers/consumers and foreground binding edges');
