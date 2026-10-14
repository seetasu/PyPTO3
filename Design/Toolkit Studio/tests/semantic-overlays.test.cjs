const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const read = (name) => fs.readFileSync(path.join(__dirname, '..', 'js', name), 'utf8');
function fixture() {
  const context = vm.createContext({ window: {}, console });
  for (const name of ['paged-attention-pypto-source', 'semantic-computation-graph', 'semantic-computation-presentation', 'semantic-computation-layout', 'paged-attention-semantic-graph']) {
    vm.runInContext(read(name + '.js'), context);
  }
  context.activeSemanticModel = () => context.window.PTO_PYPA_SEMANTIC_GRAPH;
  const w = context.window, model = w.PTO_PYPA_SEMANTIC_GRAPH;
  const all = w.PTO_COMPUTATION_PRESENTATION.flatten(model);
  const expanded = new Set(all.filter(({ node }) => node.type === 'composite').map(({ id }) => id));
  const graph = w.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize(model, expanded);
  const demo = read('demo-v2.js');
  const slice = (a, b) => demo.slice(demo.indexOf(a), demo.indexOf(b, demo.indexOf(a)));
  vm.runInContext(`const pypaOverlays = {shape:true, precision:true, memory:false, control:false}; const pypaEscape = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));\n` +
    slice('  function pypaComputeCardHtml(node) {', '  // 缩放读数：') +
    slice('  function pypaInspectorRows(rows) {', '  /* 总览保留任务执行图') +
    'window.render = {card:pypaComputeCardHtml, inspector:pypaInspectorHtml, layers:pypaOverlays};', context);
  return { w, model, all, graph, render: w.render };
}
const plain = (value) => JSON.parse(JSON.stringify(value));
test('overlay projection preserves runtime symbols, local valid shape and absent external dtype', () => {
  const { graph } = fixture(), node = (id) => graph.nodes.find((n) => n.id === id);
  assert.equal(node('qk/q-tile').overlay.shape, '[16, 128]');
  assert.equal(node('qk/q-tile').overlay.valid, '[5, 128]');
  assert.equal(node('k-cache').overlay.shape, '[cache_token_rows, 1024]');
  assert.equal(node('k-cache').overlay.precision, '');
  assert.equal(node('q-proj').overlay.precision, '');
  assert.equal(node('q-proj').overlay.shape, null);
  assert.equal(node('qk/matmul').overlay.precision, '输出 FP32');
});
test('every memory label exactly matches the node storage declaration, without borrowing', () => {
  const { w, model, all } = fixture();
  for (const { node, id } of all) {
    const overlay = w.PTO_COMPUTATION_PRESENTATION.overlays(node, { model, qualifiedId: id });
    assert.equal(overlay.memory, node[node.type]?.storage?.space || '', id);
    if (overlay.memory) assert.ok(overlay.evidence.some((e) => e.fields.includes(`${node.type}.storage`)), id);
  }
});
test('all 16 combinations retain names/types and expose only enabled properties', () => {
  const { graph, render } = fixture();
  for (let mask = 0; mask < 16; mask++) {
    ['shape','precision','memory','control'].forEach((key, i) => { render.layers[key] = !!(mask & (1 << i)); });
    for (const node of graph.nodes.filter((n) => !n.expandedGroup)) {
      const html = render.card(node);
      assert.doesNotMatch(html, /undefined|null|NaN|\[object Object\]/, node.id);
      assert.equal(html.includes('node-shape'), render.layers.shape && !!node.card, node.id);
      assert.equal(html.includes('data-overlay-field="memory"'), render.layers.memory && !!node.overlay.memory, node.id);
      assert.equal(html.includes('data-overlay-field="precision"'), render.layers.precision && !!node.overlay.precision, node.id);
      assert.ok(node.title && node.badge, node.id);
      if (mask === 0) assert.doesNotMatch(html, /node-shape|node-valid|data-overlay-field=/);
    }
  }
});
test('reading overlays cannot mutate topology, geometry, expansion or source mapping', () => {
  const { w, model, all, graph } = fixture();
  const original = JSON.stringify(model), geometry = plain(graph.nodes.map((n) => [n.id,n.x,n.y,n.w,n.h,n.sourceRanges]));
  for (const { node, id } of all) w.PTO_COMPUTATION_PRESENTATION.overlays(node, { model, qualifiedId:id });
  assert.equal(JSON.stringify(model), original);
  const after = w.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize(model, graph.opened);
  assert.deepEqual(plain(after.nodes.map((n) => [n.id,n.x,n.y,n.w,n.h,n.sourceRanges])), geometry);
  assert.deepEqual(plain(after.edges.map((e) => [e.id,e.s,e.t,e.type])), plain(graph.edges.map((e) => [e.id,e.s,e.t,e.type])));
  assert.ok(after.edges.some((e) => e.type === 'feedback'));
  assert.ok(after.edges.some((e) => e.type === 'control'));
});
test('QK inspector keeps transformation and source first, derivations folded and long rows full width', () => {
  const { graph, render } = fixture(), qk = graph.nodes.find((n) => n.id === 'qk');
  const html = render.inspector(qk, qk.sourceRanges);
  assert.match(html, /结果 \[16, 128\] · FP32 输出/);
  assert.ok(html.indexOf('Shape Transformation') < html.indexOf('来源依据 · 完整属性'));
  assert.match(html, /<details[^>]*><summary>计算细节 · Shape 推导与粒度/);
  assert.doesNotMatch(html, /<details[^>]*\bopen\b/);
  assert.match(html, /<dt class="is-long">粒度区别<\/dt><dd>/);
  assert.match(html, /data-pypa-source-node="qk"/);
});
