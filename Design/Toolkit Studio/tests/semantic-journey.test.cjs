const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path'), vm = require('node:vm');
function load() {
  const w = {}, context = vm.createContext({window:w,console});
  for (const name of ['paged-attention-pypto-source','semantic-computation-graph','semantic-computation-presentation','semantic-computation-layout','paged-attention-semantic-graph','matmul-semantic-graph','semantic-computation-journey']) vm.runInContext(fs.readFileSync(path.join(__dirname,'../js',name+'.js'),'utf8'),context);
  return w;
}
const plain = (value) => JSON.parse(JSON.stringify(value));
test('Q downstream reaches output through actual operands and never visits K/V source branches', () => {
  const w=load(), r=w.PTO_COMPUTATION_JOURNEY.trace(w.PTO_PYPA_SEMANTIC_GRAPH,'q','downstream');
  for (const id of ['q','qk/q-tile','qk/matmul','qk/score','scores','scale-mask/scale','scale-mask/mask','softmax/scores','softmax/probability','probability','pv/matmul','context-acc/add','context','output-norm/normalize','output']) assert.ok(r.nodeIds.has(id),id);
  for (const id of ['k-cache','v-cache','k','v','k-proj','v-proj','qk/k-page','pv/v-page','aux']) assert.equal(r.nodeIds.has(id),false,id);
  assert.deepEqual([...r.terminals],['output']);
});
test('bidirectional is union of independent walks, not an undirected connected component', () => {
  const w=load(), m=w.PTO_PYPA_SEMANTIC_GRAPH, api=w.PTO_COMPUTATION_JOURNEY;
  const both=api.trace(m,'q','both'), up=api.trace(m,'q','upstream'), down=api.trace(m,'q','downstream');
  assert.deepEqual([...both.nodeIds].sort(),[...new Set([...up.nodeIds,...down.nodeIds])].sort());
  assert.ok(up.nodeIds.has('q-proj'));
  assert.equal(up.nodeIds.has('qk/matmul'),false);
  assert.equal(both.nodeIds.has('k-proj'),false);
});
test('K channels stay separate across Cache Append and controls are not K payload sources', () => {
  const w=load(), api=w.PTO_COMPUTATION_JOURNEY,m=w.PTO_PYPA_SEMANTIC_GRAPH;
  const k=api.trace(m,'k','downstream');
  for (const id of ['cache-append/write-k','cache-append/key','k-cache','qk/k-page','qk/matmul']) assert.ok(k.nodeIds.has(id),id);
  for (const id of ['cache-append/write-v','cache-append/value','v-cache','cache-append/slot','aux']) assert.equal(k.nodeIds.has(id),false,id);
  const up=api.trace(m,'k-cache','upstream'),down=api.trace(m,'k-cache','downstream');
  assert.ok(up.nodeIds.has('k-proj')); assert.ok(up.nodeIds.has('k')); assert.equal(up.nodeIds.has('q'),false);
  assert.ok(down.nodeIds.has('qk/matmul')); assert.equal(down.nodeIds.has('k-proj'),false);
  for (const e of up.index.edges.filter((e)=>e.type==='control')) assert.equal(up.edgeIds.has(e.id),false);
});
test('feedback visits are bounded and reported as cross-iteration relations',()=>{
  const w=load(),m=w.PTO_PYPA_SEMANTIC_GRAPH,api=w.PTO_COMPUTATION_JOURNEY;
  const r=api.trace(m,'softmax/probability','both');
  assert.ok(r.feedbackEdgeIds.has('softmax/updated>previous:next stack'));
  assert.ok(r.feedbackEdgeIds.has('context-acc/add>old:next stack o'));
  assert.ok(r.nodeIds.size<=api.index(m).nodes.size);
  assert.deepEqual([...r.edgeIds], [...api.trace(m,'softmax/probability','both').edgeIds]);
});
test('canonical journey maps through collapsed, expanded and nested composite boundaries',()=>{
  const w=load(),m=w.PTO_PYPA_SEMANTIC_GRAPH,api=w.PTO_COMPUTATION_JOURNEY;
  const r=api.trace(m,'q','downstream');
  for (const expanded of [new Set(),new Set(['qk']),new Set(['qk','softmax','pv','context-acc','output-norm'])]) {
    const visible=w.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize(m,expanded),view=api.project(r,visible);
    for (const id of ['q','qk','scores','softmax','probability','pv','context-acc','context','output']) assert.ok(view.nodeIds.has(id),id);
    assert.equal(view.nodeIds.has('k-cache'),false);
    assert.ok(view.otherInputIds.has('k-cache'));
    assert.equal(view.anchorVisibleId,'q');
    assert.ok(view.edgeIds.size>0);
    if (expanded.has('qk')) {assert.ok(view.nodeIds.has('qk/q-tile'));assert.equal(view.nodeIds.has('qk/k-page'),false);}
  }
});
test('folding an internal anchor preserves the original qualified ID and maps it to the parent',()=>{
  const w=load(),m=w.PTO_PYPA_SEMANTIC_GRAPH,api=w.PTO_COMPUTATION_JOURNEY;
  const r=api.trace(m,'qk/q-tile','downstream');
  const folded=api.project(r,w.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize(m,new Set()));
  assert.equal(r.anchorId,'qk/q-tile'); assert.equal(folded.anchorVisibleId,'qk'); assert.equal(folded.anchorCollapsed,true); assert.ok(folded.nodeIds.has('output'));
  const open=api.project(r,w.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize(m,new Set(['qk'])));
  assert.equal(open.anchorVisibleId,'qk/q-tile');assert.equal(open.anchorCollapsed,false);
});
test('same-entity views are not unioned; only declared directed slice/aggregate boundaries carry data',()=>{
  const w=load(),api=w.PTO_COMPUTATION_JOURNEY;
  const m={nodes:[{id:'a',type:'data',name:'A',data:{entity:'shared',relations:[{kind:'slice',entity:'shared'}]}},{id:'b',type:'data',name:'B',data:{entity:'shared'}}],edges:[],groups:[]};
  assert.equal(api.trace(m,'a','downstream').nodeIds.has('b'),false);
  const real=api.index(w.PTO_PYPA_SEMANTIC_GRAPH);
  assert.ok(real.edges.some((e)=>e.mappings.some((map)=>map.relation==='slice')));
  assert.ok(real.edges.some((e)=>e.mappings.some((map)=>map.relation==='aggregate')));
});
test('journey is layout-independent, read-only, and shared Matmul ports remain valid',()=>{
  const w=load(),api=w.PTO_COMPUTATION_JOURNEY;
  for (const m of [w.PTO_PYPA_SEMANTIC_GRAPH,w.PTO_MATMUL_SEMANTIC_GRAPH]) {
    const before=JSON.stringify(m),index=api.index(m);
    for (const [id,item] of index.nodes) {
      if(item.node.type==='composite') continue;
      const result=api.trace(m,id,'both',index);
      assert.ok(result); assert.ok(result.nodeIds.has(id));
      for(const edge of index.edges) if(result.edgeIds.has(edge.id)) assert.ok(index.nodes.has(edge.s)&&index.nodes.has(edge.t));
    }
    assert.equal(JSON.stringify(m),before);
  }
  assert.equal(api.trace(w.PTO_MATMUL_SEMANTIC_GRAPH,'q'),null);
});
test('Minimap viewport uses the live affine transform and click inverse preserves its scale',()=>{
  const w=load(),api=w.PTO_COMPUTATION_JOURNEY;
  const graph=w.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize(w.PTO_PYPA_SEMANTIC_GRAPH,new Set(['qk']));
  for(const view of [{scale:.9,tx:50,ty:-600},{scale:.3,tx:-20,ty:150}]) {
    const map=api.minimap(graph,view,{w:800,h:600},150,180);
    const point=api.worldPoint(map,map.viewport.x,map.viewport.y);
    assert.ok(Math.abs(point.x+view.tx/view.scale)<1e-9);
    assert.ok(Math.abs(point.y+view.ty/view.scale)<1e-9);
    assert.ok(Math.abs(map.viewport.w-800/view.scale*map.scale)<1e-9);
    assert.equal(map.groups.length,graph.groups.length);
    assert.equal(map.nodes.length,graph.nodes.length);
  }
});
