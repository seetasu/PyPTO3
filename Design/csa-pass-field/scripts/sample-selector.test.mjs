import {build} from 'esbuild';
import assert from 'node:assert/strict';
globalThis.document={createElement:()=>({getContext:()=>({measureText:text=>({width:[...text].length*9})})})};
for(const id of ['qk_pv','csa_merge_pack_publish']){
 globalThis.location={hash:'#callable='+id};
 const built=await build({entryPoints:['src/graph-state.js','src/sample-registry.js'],bundle:true,platform:'node',format:'esm',outdir:'/tmp/callable-selector-unused',write:false,loader:{'.css':'empty','.svg':'dataurl'},logLevel:'silent'});
 const modules=[];
 for(const file of built.outputFiles)modules.push(await import('data:text/javascript;base64,'+Buffer.from(file.text+'\n//'+id).toString('base64')));
 const [adapter,registry]=modules;
 assert.equal(registry.sample.id,id);assert.equal(registry.samples.length,2);
 const hash=registry.sampleHash(id,'#pass=33&node=op.exp&stage=isa&palette=identity&layout=vertical');
 const params=new URLSearchParams(hash.slice(1));assert.equal(params.get('callable'),id);assert(!params.has('node'));assert.equal(params.get('pass'),'33');assert.equal(params.get('layout'),'vertical');
 for(const index of [1,23,33,36,52])for(const layout of ['horizontal','vertical']){
  const state=adapter.buildDomainState(index);
  const flow=adapter.toReactFlow(state,{currentPass:{...adapter.passes.passes[Math.min(index,51)-1],index},selectedId:null,hoverId:null,showIsa:index===52,positionOverrides:{},layout,loadExpanded:false,loadData:{onToggle(){},onInstanceSelect(){}}});
  assert(flow.nodes.length);const ids=new Set(flow.nodes.map(n=>n.id));
  assert.equal(ids.size,flow.nodes.length);
  for(const n of flow.nodes)for(const value of [n.position.x,n.position.y,n.style.width,n.style.height])assert(Number.isFinite(value),id+':'+n.id);
  for(const edge of flow.edges){assert(ids.has(edge.source),edge.source);assert(ids.has(edge.target),edge.target);}
  assert.equal(ids.has('op.merge'),id==='csa_merge_pack_publish');
  if(index>=33){assert([...ids].some(id=>id==='space.Vec'||id.endsWith(':Vec')));if(id==='qk_pv')assert([...ids].some(id=>id.endsWith(':Mat')));else assert(![...ids].some(id=>id.endsWith(':Mat')));}
 }
 console.log(id+': correct registry, clean selection URL, 10 stage/layout states and isolated memory spaces');
}
