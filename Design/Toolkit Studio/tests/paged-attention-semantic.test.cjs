const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const js = path.join(__dirname, '..', 'js');
const plain = (value) => JSON.parse(JSON.stringify(value));
function load() {
  const context = vm.createContext({ window: {} });
  for (const name of ['paged-attention-pypto-source', 'semantic-computation-graph', 'semantic-computation-presentation', 'semantic-computation-layout', 'paged-attention-semantic-graph']) {
    vm.runInContext(fs.readFileSync(path.join(js, name + '.js'), 'utf8'), context, { filename: name + '.js' });
  }
  return context.window;
}
function allNodes(graph, prefix = '') {
  return graph.nodes.flatMap((node) => [{ node, id: prefix + node.id }, ...(node.composite ? allNodes(node.composite.subgraph, prefix + node.id + '/') : [])]);
}
const find = (model, id) => allNodes(model).find((item) => item.id === id).node;

// Geometry recorded before this contract migration; protects the existing layout.
const collapsedGeometry = [
  ['q-proj',72,76,170,88], ['k-proj',314,76,170,88], ['v-proj',556,76,170,88],
  ['q-prep',166,276,188,88], ['k-prep',426,276,188,88], ['v-scale',686,276,188,88],
  ['q',193,476,170,88], ['k',435,476,170,88], ['v',677,476,170,88], ['aux',798,76,170,88],
  ['cache-append',426,676,188,88], ['k-cache',314,876,170,88], ['v-cache',556,876,170,88],
  ['qk',426,1076,188,88], ['scores',435,1276,170,88], ['scale-mask',426,1476,188,88],
  ['softmax',426,1676,188,88], ['probability',435,1876,170,88], ['pv',426,2076,188,88],
  ['context-acc',426,2276,188,88], ['context',435,2476,170,88], ['output-norm',426,2676,188,88], ['output',435,2876,170,88],
];
test('all Paged Attention semantic nodes validate without legacy display fields', () => {
  const w = load(), model = w.PTO_PYPA_SEMANTIC_GRAPH;
  assert.equal(w.PTO_SEMANTIC_COMPUTATION_GRAPH.validate(model), model);
  assert.equal(allNodes(model).length, 69);
  for (const { node } of allNodes(model)) {
    for (const legacy of ['sub', 'title', 'kind', 'badge', 'lines', 'sourceLine', 'api', 'semanticType']) assert.equal(legacy in node, false);
    assert.ok(node.source.ranges.length);
    assert.ok(node.provenance.length);
  }
});
test('collapsed geometry and every root dependency direction and port are retained', () => {
  const w = load(), model = w.PTO_PYPA_SEMANTIC_GRAPH;
  const placed = w.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize(model, new Set());
  assert.deepEqual(plain(placed.nodes.map((n) => [n.id,n.x,n.y,n.w,n.h])), collapsedGeometry);
  const pairs = ['q-proj>q-prep','q-prep>q','k-proj>k-prep','k-prep>k','v-proj>v-scale','v-scale>v','k>cache-append','v>cache-append',
    'aux>cache-append','aux>qk','aux>scale-mask','cache-append>k-cache','cache-append>v-cache','q>qk','k-cache>qk','qk>scores','scores>scale-mask',
    'scale-mask>softmax','softmax>probability','probability>pv','v-cache>pv','pv>context-acc','softmax>context-acc','softmax>context',
    'context-acc>context','context>output-norm','output-norm>output'];
  assert.deepEqual(plain(model.edges.map((e) => e.s + '>' + e.t)), pairs);
  const scores = model.edges.find((e) => e.s === 'qk' && e.t === 'scores');
  assert.equal(scores.sourcePort, 'Scores');
  assert.equal(model.edges.find((e) => e.s === 'softmax' && e.t === 'context-acc').targetPort, 'state');
});
test('every individual and nested expansion resolves to declared endpoint ports without mutation', () => {
  const w = load(), model = w.PTO_PYPA_SEMANTIC_GRAPH, before = JSON.stringify(model);
  const ids = allNodes(model).filter(({ node }) => node.type === 'composite').map(({ id }) => id);
  const cases = ids.map((id) => new Set(id.split('/').map((_, i, parts) => parts.slice(0,i+1).join('/'))));
  cases.push(new Set(ids));
  for (const expanded of cases) {
    const placed = w.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize(model, expanded);
    const byId = new Map(placed.nodes.map((n) => [n.id,n]));
    for (const e of placed.edges) {
      assert.ok(e.sourcePort in byId.get(e.s).ports.outputs, e.id + ' source port');
      assert.ok(e.targetPort in byId.get(e.t).ports.inputs, e.id + ' target port');
      assert.ok(e.semantic);
      assert.ok(e.semanticSource && e.semanticTarget);
    }
    assert.ok(placed.edgeRoutes.every((r) => !r.path.includes('NaN')));
  }
  assert.equal(JSON.stringify(model), before);
  const nested = w.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize(model, new Set(['q-prep','q-prep/norm']));
  const input = nested.edges.find((e) => e.semanticSource === 'q-proj');
  assert.equal(input.t, 'q-prep/norm/sum');
  assert.equal(input.targetPort, 'default');
  assert.equal(input.semantic.targetPort, 'Q');
});
test('Page Scores refine Stack Scores explicitly and reject incompatible boundary dtype', () => {
  const w = load(), model = w.PTO_PYPA_SEMANTIC_GRAPH;
  const parent = find(model, 'scores'), child = find(model, 'qk/score'), composite = find(model, 'qk');
  assert.equal(parent.data.dtype, 'FP32'); assert.equal(parent.data.granularity, 'stack');
  assert.equal(child.data.dtype, 'FP32'); assert.equal(child.data.granularity, 'page');
  assert.equal(parent.data.entity, child.data.entity);
  assert.equal(composite.composite.portMappings.outputs.Scores.relation, 'aggregate');
  assert.match(composite.composite.portMappings.outputs.Scores.description, /offset/);
  const wrong = plain(model); find(wrong, 'qk').ports.outputs.Scores.dtype = 'BF16';
  assert.throws(() => w.PTO_SEMANTIC_COMPUTATION_GRAPH.validate(wrong), /dtype mismatch/);
});
test('feedback is semantic even without visual route, and Context differs from PV Partial', () => {
  const w = load(), model = plain(w.PTO_PYPA_SEMANTIC_GRAPH);
  const graph = find(model, 'softmax').composite.subgraph;
  const feedback = graph.edges.find((e) => e.s === 'updated' && e.t === 'previous');
  assert.equal(feedback.type, 'feedback'); delete feedback.route;
  assert.doesNotThrow(() => w.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize(model, new Set(['softmax','context-acc'])));
  assert.equal(find(model, 'pv/partial').type, 'data');
  assert.equal(find(model, 'context-acc/partial').type, 'data');
  assert.equal(find(model, 'context-acc/old').type, 'state');
  assert.equal(find(model, 'context').type, 'state');
  assert.match(find(model, 'context-acc/pending').state.update, /pending1_iter/);
});
test('source-backed Q Tile, physical cache pages, and unknown inputs remain distinct', () => {
  const w = load(), model = w.PTO_PYPA_SEMANTIC_GRAPH;
  const q = find(model, 'qk/q-tile').data, k = find(model, 'qk/k-page').data, v = find(model, 'pv/v-page').data;
  assert.equal(q.role, 'input-view'); assert.equal(q.granularity, 'tile');
  assert.deepEqual(plain(q.shape.valid), ['GROUP','HEAD_DIM']);
  assert.equal(q.storage.space, 'Mat'); assert.equal('dtype' in q, false);
  assert.equal(k.role, 'cache-view'); assert.equal(k.granularity, 'page'); assert.equal('dtype' in k, false);
  assert.equal(v.dtype, 'BF16'); assert.match(k.addressing.physicalPage, /block_table_2d/);
  assert.match(k.addressing.row, /physical_page \* BLOCK_SIZE/);
  for (const id of ['q-proj','k-proj','v-proj']) {
    const data = find(model,id).data;
    assert.equal('dtype' in data, false); assert.equal('shape' in data, false);
    const display = w.PTO_COMPUTATION_PRESENTATION.node(find(model,id));
    assert.equal(display.sub, 'tensor'); assert.equal(display.api, '');
  }
});
test('every symbol and API is present in cited source; source ranges are structured and bounded', () => {
  const w = load(), source = w.PTO_PAGED_ATTENTION_PYPTO_SOURCE.trimEnd().split('\n');
  for (const { node, id } of allNodes(w.PTO_PYPA_SEMANTIC_GRAPH)) {
    const text = node.source.ranges.map(({ start, end }) => {
      assert.ok(start >= 1 && end <= source.length, id);
      return source.slice(start-1,end).join('\n');
    }).join('\n');
    for (const symbol of node.source.symbols) assert.ok(text.includes(symbol), id + ': ' + symbol);
    for (const api of node.operation?.api || node.composite?.api || []) assert.ok(text.includes(api), id + ': ' + api);
  }
});
test('Inspector derives connections and exposes addressing, mapping, and provenance without fabricated API', () => {
  const w = load(), model = w.PTO_PYPA_SEMANTIC_GRAPH;
  const score = w.PTO_COMPUTATION_PRESENTATION.node(find(model,'qk/score'), { id:'qk/score' });
  const sections = w.PTO_COMPUTATION_PRESENTATION.inspector(score, { model, qualifiedId:'qk/score' });
  const rows = sections.flatMap((s) => s.rows);
  assert.equal(rows.find((r) => r.label === '生产者').value, 'matmul');
  assert.ok(rows.some((r) => r.value.includes('FP32')));
  assert.equal(rows.some((r) => r.label === 'api'), false);
  const cacheRows = w.PTO_COMPUTATION_PRESENTATION.inspector(find(model,'k-cache'), { model }).flatMap((s) => s.rows);
  assert.ok(cacheRows.some((r) => r.value.includes('block_table_2d')));
  assert.equal(w.PTO_COMPUTATION_PRESENTATION.sourceLine(find(model,'k-cache')), 194);
});

test('visual feedback styling cannot hide a missing semantic feedback dependency', () => {
  const w = load(), model = plain(w.PTO_PYPA_SEMANTIC_GRAPH);
  const feedback = find(model,'softmax').composite.subgraph.edges.find((e) => e.type === 'feedback');
  feedback.type = 'state'; feedback.route = 'feedback';
  assert.throws(() => w.PTO_SEMANTIC_COMPUTATION_GRAPH.validate(model), /forward dependency cycle/);
});
test('all five provenance levels remain compatible while undeclared guesses are rejected', () => {
  const w = load();
  for (const kind of ['fact','resolved','compiled','estimated','measured']) {
    const model = plain(w.PTO_PYPA_SEMANTIC_GRAPH);
    model.nodes[0].provenance[0].kind = kind;
    assert.doesNotThrow(() => w.PTO_SEMANTIC_COMPUTATION_GRAPH.validate(model));
  }
  const model = plain(w.PTO_PYPA_SEMANTIC_GRAPH);
  model.nodes[0].provenance[0].kind = 'guess';
  assert.throws(() => w.PTO_SEMANTIC_COMPUTATION_GRAPH.validate(model), /invalid provenance kind/);
});
