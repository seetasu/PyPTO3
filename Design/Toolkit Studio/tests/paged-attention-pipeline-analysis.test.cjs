const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const base = path.join(__dirname, '..');
function load() {
  const context = vm.createContext({ window: {} });
  for (const name of ['paged-attention-pypto-source', 'semantic-computation-graph', 'semantic-computation-presentation', 'semantic-computation-layout', 'paged-attention-semantic-graph', 'paged-attention-intro', 'paged-attention-stack-dependency-model', 'paged-attention-stack-dependency']) {
    vm.runInContext(fs.readFileSync(path.join(base, 'js', `${name}.js`), 'utf8'), context, { filename: `${name}.js` });
  }
  return context.window;
}
const plain = (value) => JSON.parse(JSON.stringify(value));

test('Task 80 at 2049 tokens projects only real dependencies across its five pipeline stacks', () => {
  const w = load();
  const pipeline = w.PtoPagedAttentionIntro.model.pipelineFor(2049);
  const model = w.PtoPagedAttentionStackDependencyModel.build(pipeline, w.PTO_PYPA_SEMANTIC_GRAPH.synchronizations);
  const selection = { task: 80, stack: 1, stage: 'softmax', pipelineStep: 1 };
  const modelBefore = plain(model), selectionBefore = plain(selection);

  assert.equal(model.stackCount, 5);
  assert.equal(model.stepCount, 7);
  assert.deepEqual(plain(model.stacks.map((stack) => stack.slot)), [0, 1, 2, 0, 1]);
  assert.equal(model.edges.filter((edge) => edge.kind === 'reuse').length, 6);

  for (const category of ['state', 'reuse', 'sync']) {
    const projection = w.PtoPagedAttentionStackDependency.project(model, selection, { category });
    assert.ok(projection.edges.every((edge) => edge.kind === category));
    assert.ok(projection.edges.every((edge) => edge.kind !== 'data'));
    assert.ok(projection.selectedEdge === null || projection.edges.includes(projection.selectedEdge));
  }

  const ml = model.edges.find((edge) => edge.id === 'state:m-l:s0:s1');
  const o = model.edges.find((edge) => edge.id === 'state:o:s0:s1');
  assert.deepEqual([ml.from, ml.to], ['s0:softmax', 's1:softmax']);
  assert.deepEqual([o.from, o.to], ['s0:accumulate', 's1:accumulate']);

  const syncProjection = w.PtoPagedAttentionStackDependency.project(model, selection, { category: 'sync' });
  for (const eventId of ['qk-ready', 'softmax-ready', 'pv-ready']) {
    const edges = syncProjection.edges.filter((edge) => edge.id.endsWith(`:${eventId}`));
    assert.equal(edges.length, 5);
    assert.ok(edges.every((edge) => edge.kind === 'sync'));
    assert.ok(edges.every((edge) => !syncProjection.edges.some((candidate) => candidate.kind === 'data' && candidate.from === edge.from && candidate.to === edge.to)));
  }

  const explicit = w.PtoPagedAttentionStackDependency.project(model, selection, { category: 'state', edgeId: 'state:o:s0:s1' });
  assert.equal(explicit.selectedEdge.id, 'state:o:s0:s1');
  assert.deepEqual(plain(model), modelBefore);
  assert.deepEqual(plain(selection), selectionBefore);
});

test('reuse projection is empty before the first three-slot wrap', () => {
  const w = load();
  const model = w.PtoPagedAttentionStackDependencyModel.build(w.PtoPagedAttentionIntro.model.pipelineFor(1025), w.PTO_PYPA_SEMANTIC_GRAPH.synchronizations);
  const projection = w.PtoPagedAttentionStackDependency.project(model, { task: 0, stack: 0, stage: 'qk', pipelineStep: 0 }, { category: 'reuse' });
  assert.deepEqual(plain(projection.edges), []);
  assert.equal(projection.selectedEdge, null);
});

test('every stage anchor stays on its existing produce or consume tick position', () => {
  const w = load();
  const model = w.PtoPagedAttentionStackDependencyModel.build(w.PtoPagedAttentionIntro.model.pipelineFor(2049), w.PTO_PYPA_SEMANTIC_GRAPH.synchronizations);
  const selection = { task: 80, stack: 2, stage: 'pv', pipelineStep: 4 };
  const { anchors } = w.PtoPagedAttentionStackDependency.project(model, selection);
  assert.equal(Object.keys(anchors).length, model.nodes.length);
  for (const node of model.nodes) {
    const produce = node.stage === 'qk' || node.stage === 'softmax';
    assert.deepEqual(plain(anchors[node.id]), {
      x: 257 + node.tick * 62,
      y: produce ? 227 : 289,
      tick: node.tick,
      row: produce ? 'produce' : 'consume',
    });
  }
});

test('sequence edits preserve valid Task/Stage/mode context and reconcile a removed Stack to its real tick', () => {
  const w = load(), intro = w.PtoPagedAttentionIntro.model;
  const original = { task: 80, stack: 4, stage: 'accumulate', pipelineStep: 6, pipelineMode: 'analysis' };
  const shortened = intro.reconcileSelection(intro.pipelineFor(1025), original);
  assert.deepEqual(plain(shortened), { ...original, stack: 2, pipelineStep: 4 });
  assert.deepEqual(original, { task: 80, stack: 4, stage: 'accumulate', pipelineStep: 6, pipelineMode: 'analysis' });
  for (const seqLen of [1025, 2049, 3584]) {
    const selected = { task: 80, stack: 1, stage: 'softmax', pipelineStep: 1, pipelineMode: 'schedule' };
    assert.deepEqual(plain(intro.reconcileSelection(intro.pipelineFor(seqLen), selected)), selected);
  }
  const emptyPhase = intro.reconcileSelection(intro.pipelineFor(1), { task: 80, stack: null, stage: 'qk', pipelineStep: 6 });
  assert.equal(emptyPhase.pipelineStep, 2);
  assert.equal(emptyPhase.stack, null, 'drain tick does not fabricate a producing Stack');
});

test('the shared timeline draws one selected cause and no parallel data/sync duplicate', () => {
  const w = load(), pipeline = w.PtoPagedAttentionIntro.model.pipelineFor(2049);
  const model = w.PtoPagedAttentionStackDependencyModel.build(pipeline, w.PTO_PYPA_SEMANTIC_GRAPH.synchronizations);
  for (const [category, edgeId] of [['state','state:m-l:s0:s1'], ['state','state:o:s0:s1'], ['state','state:rescale:s0'], ['reuse','reuse:score_transfer:s0:s3'], ['sync','sync:s0:qk-ready'], ['sync','sync:s0:softmax-ready'], ['sync','sync:s0:pv-ready']]) {
    const projection = w.PtoPagedAttentionStackDependency.project(model, { task: 80, stack: 1, stage: 'softmax', pipelineStep: 1 }, { category, edgeId });
    const markup = w.PtoPagedAttentionStackDependency.overlay(model, projection);
    assert.equal((markup.match(/data-pa-dependency-edge=/g) || []).length, 1);
    assert.equal((markup.match(/class="kf-pa-analysis-edge__line"/g) || []).length, 1);
    assert.doesNotMatch(markup, /data-dep-node|data-dep-svg|is-data/);
    assert.match(markup, /y="265"/, 'annotation stays between existing Produce/Consume cells, clear of PRE_LAUNCH');
  }
});
