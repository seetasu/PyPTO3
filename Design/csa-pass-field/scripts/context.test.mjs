import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
const g=JSON.parse(fs.readFileSync(new URL('../data/csa-context.json',import.meta.url)));
test('CSA context has valid edges and a bounded QK-softmax-PV focus',()=>{
  const ids=new Set(g.nodes.map(n=>n.id));
  assert.equal(ids.size,g.nodes.length);
  for(const e of g.edges)assert.ok(ids.has(e.source)&&ids.has(e.target));
  assert.deepEqual(g.nodes.filter(n=>n.focus).map(n=>n.id),['sparse_qk','sparse_softmax','sparse_pv']);
  for(const id of ['pypto_q_down','cmp_pool','idx_score','topk_leaf','merge_sink','output_projection','attention_hc_output'])assert.ok(ids.has(id));
  for(const id of ['q_projection','main_compressor','lightning_indexer','exact_topk','head_merge','grouped_output_projection'])assert.ok(g.clusters.some(c=>c.id===id));
  assert.ok(!ids.has('ffn_norm'));
  assert.ok(g.edges.some(e=>e.source==='sparse_pv'&&e.target==='merge_statistics'));
  assert.equal(g.nodes.length,58);
  for(const n of g.nodes)assert.ok(Number.isFinite(n.x)&&Number.isFinite(n.y)&&n.width>0&&n.height>0);
  assert.equal(g.nodes.find(n=>n.id==='tp_allgather').kind,'communication');
});
test('context defaults to focused core with every surrounding program collapsed',()=>{
  assert.equal(g.variants.length,1<<g.foldable.length);
  for(const id of ['q_projection','lightning_indexer','exact_topk','head_merge','grouped_output_projection'])assert.ok(g.foldable.includes(id));
  for(const [mask,encoded] of g.variants.entries()){
    const variant={...encoded,...Object.fromEntries(['nodes','edges','clusters'].map(kind=>[kind,encoded[kind].map(index=>g.pool[kind][index])]))};
    const ids=new Set(variant.nodes.map(n=>n.id));
    for(const e of variant.edges)assert.ok(ids.has(e.source)&&ids.has(e.target));
    g.foldable.forEach((id,index)=>{
      assert.equal(ids.has(id),!(mask&(1<<index)));
      assert.equal(variant.clusters.some(c=>c.id===id),!!(mask&(1<<index)));
    });
    assert.equal(variant.nodes.filter(n=>n.focus).length,3);
    const qk=variant.nodes.find(n=>n.id==='sparse_qk');
    for(const id of ['head_group_redistribution','attention_alltoall','output_reduce_scatter','attention_hc_post'])assert.equal(variant.nodes.find(n=>n.id===id).x,qk.x);
    const scope=variant.clusters.find(c=>c.id==='sparse_attention');
    assert.ok(scope.width>=qk.width+100);
    const reads=variant.nodes.filter(n=>['window_gather','compressed_gather'].includes(n.id));
    assert.equal(reads[0].y,reads[1].y);
    assert.ok(Math.abs(reads[0].x-reads[1].x)>=(reads[0].width+reads[1].width)/2+24);
  }
});
