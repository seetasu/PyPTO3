const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const js = path.join(__dirname, '..', 'js');
function load() {
  const context = vm.createContext({ window: {} });
  for (const name of ['paged-attention-pypto-source', 'semantic-computation-graph',
    'semantic-computation-presentation', 'semantic-computation-layout',
    'paged-attention-semantic-graph', 'paged-attention-intro',
    'paged-attention-stack-dependency-model', 'visualization2-transition-intra']) {
    vm.runInContext(fs.readFileSync(path.join(js, `${name}.js`), 'utf8'), context, { filename: name });
  }
  return context.window;
}

test('READY identity and Stack survive Graph → Pipeline → Graph mapping', () => {
  const w = load(), task = w.PtoPagedAttentionIntro.model.pipelineFor(2049);
  for (const sync of w.PTO_PYPA_SEMANTIC_GRAPH.synchronizations) {
    const graph = { kind: 'event', syncId: sync.id, event: `${sync.name}_EVENT`, index: 2, stackIndex: 2 };
    const pipeline = w.PtoVisualization2IntraMorph.mapFocus('graph', 'pipeline', graph, task);
    assert.equal(pipeline.syncId, sync.id);
    assert.equal(pipeline.event, graph.event);
    assert.equal(pipeline.index, 2);
    const returned = w.PtoVisualization2IntraMorph.mapFocus('pipeline', 'graph', pipeline, task);
    assert.equal(returned.syncId, sync.id);
    assert.equal(returned.stackIndex, 2);
    const dependency = w.PtoPagedAttentionStackDependencyModel.build(task, w.PTO_PYPA_SEMANTIC_GRAPH.synchronizations);
    const edge = dependency.edges.find(item => item.id === `sync:s2:${sync.id}`);
    assert.equal(edge.synchronization, sync);
    assert.equal(edge.kind, 'sync');
    assert.equal(edge.source.file, 'paged_attention_pypto.py');
    assert.ok(dependency.edges.some(item => item.kind === 'data' && item.id.startsWith('data:s2:')));
  }
});

test('QK WAIT evidence names the Scores load boundary and matches the same logical Tick', () => {
  const w = load(), sync = w.PTO_PYPA_SEMANTIC_GRAPH.synchronizations.find(item => item.id === 'qk-ready');
  assert.equal(sync.consumer.boundary, 'scale-mask');
  assert.equal(sync.consumer.anchor, 'scale-mask/scale');
  const detail = w.PTO_COMPUTATION_PRESENTATION.synchronization(sync, w.PTO_PYPA_SEMANTIC_GRAPH);
  assert.match(detail.rows.find(row => row.label === 'Consumer').value, /Scaling \+ Mask/);
  assert.match(detail.rows.find(row => row.label === 'WAIT 位置').value, /scale-mask\/scale/);
  assert.equal(detail.sources.filter(item => item.action === 'set').length, 1);
  assert.equal(detail.sources.filter(item => item.action === 'wait').length, 1);
  const dependencies = w.PtoPagedAttentionStackDependencyModel.build(w.PtoPagedAttentionIntro.model.pipelineFor(2049), w.PTO_PYPA_SEMANTIC_GRAPH.synchronizations);
  const edge = dependencies.edges.find(item => item.id === 'sync:s2:qk-ready');
  const producer = dependencies.nodes.find(item => item.id === edge.from);
  const consumer = dependencies.nodes.find(item => item.id === edge.to);
  assert.equal(producer.tick, 2);
  assert.equal(consumer.tick, 2);
  assert.ok(dependencies.edges.some(item => item.kind === 'reuse'));
  assert.ok(dependencies.edges.some(item => item.kind === 'state'));
  assert.notEqual(edge.kind, 'reuse');
  assert.notEqual(edge.kind, 'state');
});
