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

// Scope containment is semantic, not merely a row of loop captions.
const {controlScopes,scopeForLine}=await import('../src/control-layout.js');
const controlTree=snapshots[28].nodes.find(n=>n.id==='kernel.main').sourceFacts.controlTree;
assert.equal(controlTree.length,2);
assert(controlTree[0].target.includes('pack_work'));
assert(controlTree[1].target.includes('peer_tp'));
assert.equal(controlTree[0].body[0].body.length,2);
assert(controlTree[0].body[0].body[0].iterator.includes('pl.pipeline'));
assert.equal(controlTree[1].body[0].kind,'condition');
for(const index of [0,1,2,11,28,30,36,50])for(const layout of ['horizontal','vertical']){
 const flow=module.callableFlow(snapshots[index],{currentPass:{index},layout,positionOverrides:{},loadData:{}});
 const scopes=controlScopes(snapshots[index].nodes[0].sourceFacts.controlTree);
 const inside=(a,b)=>a.position.x>=b.position.x&&a.position.y>=b.position.y+48&&a.position.x+a.style.width<=b.position.x+b.style.width&&a.position.y+a.style.height<=b.position.y+b.style.height;
 for(const scope of scopes){if(scope.parent)assert(inside(flow.nodes.find(n=>n.id===scope.id),flow.nodes.find(n=>n.id===scope.parent)),scope.id);}
 for(const node of flow.nodes.filter(n=>n.id.startsWith('bucket.'))){
  const scope=scopeForLine(scopes,node.data.regionOperations[0].line);
  if(scope)assert(inside(node,flow.nodes.find(n=>n.id===scope)),node.id);
  for(const statement of node.data.regionOperations)assert.equal(scopeForLine(scopes,statement.line),scope);
 }
}
const branches=controlScopes([{line:1,endLine:8,kind:'condition',test:'x',body:[],elseLine:6,otherwise:[]}]);
assert.equal(scopeForLine(branches,3),branches[0].id);
assert.equal(scopeForLine(branches,6),branches[1].id);
console.log('Source hierarchy, pipeline, condition/else and nested operator containment verified');

for(const index of [28,50]){
 const state=snapshots[index];
 const notify=render(index).nodes.find(n=>n.data.attributes?.opcode==='pld.system.notify');
 const flow=module.callableFlow(state,{currentPass:{index},selectedId:notify.id,layout:'horizontal',positionOverrides:{},loadData:{}});
 const incoming=flow.edges.find(e=>e.target===notify.id&&e.type==='control');
 assert(incoming&&incoming.data.hoverFocused&&incoming.zIndex===1000);
 assert.equal(flow.nodes.find(n=>n.id===incoming.source).data.attributes.opcode,index===50?'pl.system.fence':'pld.tile.put');
 assert(flow.edges.some(e=>e.source===notify.id&&e.target==='value.signal'&&e.data.hoverFocused));
}
console.log('Notify source ordering and signal write verified before/after fence lowering');

for(let index=0;index<52;index++)for(const layout of ['horizontal','vertical']){
 const flow=module.callableFlow(snapshots[index],{currentPass:{index},layout,positionOverrides:{},loadData:{}});
 const kernel=flow.nodes.find(n=>n.id==='kernel.main');
 for(const node of flow.nodes.filter(n=>n.type==='loop'||n.id.startsWith('bucket.')||['op.merge','op.normalize','op.rope','op.pack','handoff.publish','handoff.notify','space.Vec'].includes(n.id)||n.id.startsWith('context.'))){
  assert(node.position.x>=kernel.position.x+24,node.id+' outside kernel left');
  assert(node.position.y>=kernel.position.y+60,node.id+' outside kernel top');
  assert(node.position.x+node.style.width<=kernel.position.x+kernel.style.width-24,node.id+' outside kernel right');
  assert(node.position.y+node.style.height<=kernel.position.y+kernel.style.height-24,node.id+' outside kernel bottom');
 }
}
console.log('All 104 scope/layout states keep stages, loops, operators and UB inside Kernel');

const audit=[];
for(let index=0;index<52;index++){
 const flow=render(index),operators=flow.nodes.filter(n=>n.id.startsWith('bucket.'));
 const links=flow.edges.filter(e=>e.id.startsWith('nested.')||e.id.startsWith('dependency.')||e.id.startsWith('side-effect.')||e.id.startsWith('signal-write.'));
 const isolated=operators.filter(n=>!links.some(e=>e.source===n.id||e.target===n.id));
 const missing=[];
 const sourceLines=new Set(operators.flatMap(n=>n.data.regionOperations.map(s=>s.line)));
 for(const node of operators)for(const statement of node.data.regionOperations)for(const dep of statement.dependencies||[]){if(dep.producerLine!=null&&!sourceLines.has(dep.producerLine))missing.push(dep.producerLine);}
 audit.push({pass:index,operators:operators.length,dependencyEdges:links.length,loopCarriedEdges:links.filter(e=>e.data.loopCarried).length,isolated:isolated.map(n=>({id:n.id,opcode:n.data.attributes.opcode,lines:n.data.regionOperations.map(s=>s.line)})),unprojectedProducerLines:[...new Set(missing)]});
}
fs.mkdirSync('docs/validation',{recursive:true});
fs.writeFileSync('docs/validation/control-dag-audit.json',JSON.stringify(audit,null,2)+'\n');
console.log('DAG audit isolated operators:',audit.filter(a=>a.isolated.length).map(a=>[a.pass,a.isolated.map(n=>n.opcode)]));
console.log('Unprojected producer lines:',audit.filter(a=>a.unprojectedProducerLines.length).map(a=>[a.pass,a.unprojectedProducerLines]));
const load=render(28).nodes.find(n=>n.id==='bucket.op.merge.pl.tile.load.control.root.0.0');
assert(load);
const loadFlow=module.callableFlow(snapshots[28],{currentPass:{index:28},selectedId:load.id,layout:'vertical',positionOverrides:{},loadData:{}});
for(const input of ['value.mi','value.li','value.oi'])assert(loadFlow.edges.some(e=>e.source===input&&e.target===load.id&&e.data.hoverFocused));
assert(loadFlow.edges.some(e=>e.source===load.id&&e.target.includes('control.root.0.0.0')&&e.data.role==='loop-init'&&e.data.hoverFocused));
console.log('P28 outer load: three input tensors and pipeline initialization consumers verified');

for(const row of audit){assert.equal(row.isolated.length,0,'Isolated operator in P'+row.pass);assert.equal(row.unprojectedProducerLines.length,0,'Missing producer in P'+row.pass);}
const probe=JSON.parse(fs.readFileSync('data/callable-probe/analysis.json'));
for(let index=0;index<52;index++){
 const expected=Object.fromEntries(Object.entries(probe.snapshots[index].sections[0].operations).filter(([op])=>/^(pl\.(tensor|tile|system)\.|pld\.(tile|tensor|system)\.)/.test(op)&&!['pl.tile.alloc','pl.tile.get_block_idx','pl.tile.reserve_buffer'].includes(op)));
 const actual={};
 for(const n of snapshots[index].nodes)for(const statement of n.regionOperations||[])for(const op of statement.operations)actual[op]=(actual[op]||0)+1;
 assert.deepEqual(actual,expected,'Operation inventory mismatch P'+index);
}
console.log('All 52 source operation inventories match projected operations; no silent drops');
