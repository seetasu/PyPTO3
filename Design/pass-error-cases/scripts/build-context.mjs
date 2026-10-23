import fs from 'node:fs';
import vm from 'node:vm';
// Reuse workbench's source projection, program expansion and port resolution.
const base=new URL('../../architecture/deepseek_v4_flash_csa/',import.meta.url);
const context=vm.createContext({window:{},structuredClone});
for(const file of ['implementation-programs.js','mapping-registry.js','json-architecture.js'])vm.runInContext(fs.readFileSync(new URL(file,base),'utf8'),context,{filename:file});
const registry=context.window.PtoCsaMappingRegistry,renderer=context.window.PtoCsaImplementationPattern;
const payload=JSON.parse(fs.readFileSync(new URL('json-architecture.json',base)));
const foldable=registry.defaultExpandedPrograms.filter(id=>id!=='sparse_attention');
function build(mask) {
const programs=registry.defaultExpandedPrograms.filter(id=>!foldable.includes(id)||(mask&(1<<foldable.indexOf(id))));
const expanded=renderer.expandImplementationPrograms(renderer.buildGraph(payload,'l3'),new Set(programs));
// Keep the attention/output spine on one axis in every expansion state.
const spine=expanded.nodes.find(n=>n.id==='sparse_qk').x;
const downstream=new Set(['head_merge','head_group_redistribution','attention_alltoall','grouped_output_projection','output_reduce_scatter','attention_hc_post','attention_hc_output']);
for(const n of expanded.nodes)if(downstream.has(n.id)||downstream.has(n.parent))n.x=spine;
for(const c of expanded.clusters)if(downstream.has(c.id))c.x=spine-c.width/2;
for(const n of expanded.nodes){
  if(n.id==='compressor_state')n.x-=32;
  if(n.id==='indexer_state')n.x+=40;
}
const sparse=expanded.clusters.find(c=>c.id==='sparse_attention');
const firstGather=expanded.nodes.find(n=>n.id==='window_gather');
const secondGather=expanded.nodes.find(n=>n.id==='compressed_gather');
// Two independent reads share a row rather than looking like a serial chain.
firstGather.x=spine-150;secondGather.x=spine+150;secondGather.y=firstGather.y;
sparse.x=spine-310;sparse.width=620;
const qkTop=expanded.nodes.find(n=>n.id==='sparse_qk').y;
for(const n of expanded.nodes)if(n.y>=qkTop)n.y+=32;
for(const c of expanded.clusters){if(c.y>=qkTop)c.y+=32;else if(c.y+c.height>=qkTop)c.height+=32;}
const merge=expanded.nodes.find(n=>n.id==='head_merge')||expanded.clusters.find(c=>c.id==='head_merge');
const redistribution=expanded.nodes.find(n=>n.id==='head_group_redistribution');
if(merge&&redistribution){const bottom=merge.y+(merge.id==='head_merge'&&'kind' in merge?merge.height/2:merge.height);redistribution.y=Math.max(redistribution.y,bottom+60);}
expanded.height+=32;
const focusIds=new Set(['sparse_qk','sparse_softmax','sparse_pv']);
const nodes=expanded.nodes.map(n=>{
  const source=registry.getNode(n.id);
  return {id:n.id,label:n.label,x:n.x,y:n.y,width:n.width,height:n.height,parent:n.parent,foldable:foldable.includes(n.id),
    kind:n.kind==='tensor'?'tensor':source?.type==='Communication'?'communication':'op',
    focus:focusIds.has(n.id),detail:source?.reason||n.mappingReason||n.label,sourceRefs:source?.sourceRefs||n.sourceRefs||[],
    subtitle:n.kind==='tensor'?(n.id.includes('cache')?'KV cache':n.id.includes('mapping')||n.id==='block_table'?'Runtime metadata':'State / value'):null};
});
const graph={source:'architecture/deepseek_v4_flash_csa/workbench.html',mapping:'Workbench expanded implementation; qk_pv focus is a semantic correspondence, not compile-run identity.',
  width:expanded.width,height:expanded.height,nodes,edges:expanded.edges.map(e=>({source:e.source,target:e.target,label:e.tensor?.name||'',dashed:e.dashed||false})),
  clusters:expanded.clusters.map(c=>({id:c.id,label:c.label,x:c.x,y:c.y,width:c.width,height:c.height,parent:c.parent,foldable:foldable.includes(c.id)}))};
return graph;
}
const graph=build((1<<foldable.length)-1);
graph.foldable=foldable;
// Intern repeated geometry so arbitrary independent expansion stays compact.
graph.pool={nodes:[],edges:[],clusters:[]};
const lookup={nodes:new Map(),edges:new Map(),clusters:new Map()};
const intern=(kind,value)=>{
  const key=JSON.stringify(value),map=lookup[kind];
  if(!map.has(key)){map.set(key,graph.pool[kind].length);graph.pool[kind].push(value);}
  return map.get(key);
};
graph.variants=Array.from({length:1<<foldable.length},(_,mask)=>{
  const variant=build(mask);
  return {width:variant.width,height:variant.height,...Object.fromEntries(['nodes','edges','clusters'].map(kind=>[kind,variant[kind].map(value=>intern(kind,value))]))};
});
fs.writeFileSync(new URL('../data/csa-context.json',import.meta.url),JSON.stringify(graph,null,2));
