const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const base = path.join(__dirname, '..');
const js = path.join(base, 'js');
function load() {
  const context = vm.createContext({ window: {} });
  for (const name of ['paged-attention-pypto-source', 'semantic-computation-graph', 'semantic-computation-presentation', 'semantic-computation-layout', 'paged-attention-semantic-graph', 'paged-attention-intro', 'paged-attention-stack-dependency-model']) {
    vm.runInContext(fs.readFileSync(path.join(js, `${name}.js`), 'utf8'), context, { filename: `${name}.js` });
  }
  return context.window;
}
const plain = (value) => JSON.parse(JSON.stringify(value));
function isDag(nodes, edges) {
  const degree = new Map(nodes.map(({ id }) => [id, 0]));
  const outgoing = new Map(nodes.map(({ id }) => [id, []]));
  for (const edge of edges) { degree.set(edge.to, degree.get(edge.to) + 1); outgoing.get(edge.from).push(edge.to); }
  const ready = [...degree].filter(([, count]) => count === 0).map(([id]) => id);
  let visited = 0;
  while (ready.length) {
    const id = ready.pop(); visited += 1;
    for (const next of outgoing.get(id)) { degree.set(next, degree.get(next) - 1); if (degree.get(next) === 0) ready.push(next); }
  }
  return visited === nodes.length;
}

test('build uses pipeline steps and yields expected nodes, edges, tail tokens, and reuse constraints', () => {
  const w = load(), model = w.PtoPagedAttentionStackDependencyModel;
  assert.deepEqual(plain(model.stages), ['qk', 'softmax', 'pv', 'accumulate']);
  for (const [seqLen, expectedStacks, expectedSteps, reuseCount, tailTokens] of [[1, 1, 3, 0, 1], [512, 1, 3, 0, 512], [513, 2, 4, 0, 1], [1024, 2, 4, 0, 512], [1025, 3, 5, 0, 1], [1536, 3, 5, 0, 512], [1537, 4, 6, 3, 1], [3584, 7, 9, 12, 512]]) {
    const pipeline = w.PtoPagedAttentionIntro.model.pipelineFor(seqLen);
    const before = plain(pipeline), sync = w.PTO_PYPA_SEMANTIC_GRAPH.synchronizations, syncBefore = plain(sync);
    const result = model.build(pipeline, sync);
    assert.equal(result.stackCount, expectedStacks);
    assert.equal(result.stepCount, expectedSteps);
    assert.equal(result.nodes.length, expectedStacks * 4);
    assert.equal(result.edges.filter((edge) => edge.kind === 'data').length, expectedStacks * 3);
    assert.equal(result.edges.filter((edge) => edge.kind === 'sync').length, expectedStacks * 3);
    assert.equal(result.edges.filter((edge) => edge.kind === 'reuse').length, reuseCount);
    assert.equal(result.nodes.find((node) => node.stack === expectedStacks - 1 && node.stage === 'qk').validTokens, tailTokens);
    for (const stack of pipeline.stacks) {
      const stepProduce = pipeline.steps.find((step) => step.produceStack === stack.id);
      const stepConsume = pipeline.steps.find((step) => step.consumeStack === stack.id);
      for (const node of result.nodes.filter((item) => item.stack === stack.id)) {
        assert.equal(node.tick, ['qk', 'softmax'].includes(node.stage) ? stepProduce.tick : stepConsume.tick);
        assert.equal(node.slot, stack.slot);
      }
      const fifo = result.edges.find((edge) => edge.id === `state:rescale:s${stack.id}`);
      assert.deepEqual(plain(fifo.versions), [`tick ${stepProduce.tick}: pending1=r${stack.id}`, `tick ${stepProduce.tick + 1}: pending0=r${stack.id}`, `tick ${stepProduce.tick + 2}: consume r${stack.id}`]);
    }
    assert.equal(isDag(result.nodes, result.edges), true);
    assert.deepEqual(plain(pipeline), before);
    assert.deepEqual(plain(sync), syncBefore);
  }
});

test('synchronization edges reuse semantic records and cite their SET/WAIT ranges', () => {
  const w = load(), pipeline = w.PtoPagedAttentionIntro.model.pipelineFor(1025);
  const result = w.PtoPagedAttentionStackDependencyModel.build(pipeline, w.PTO_PYPA_SEMANTIC_GRAPH.synchronizations);
  for (const [eventId, expectedName] of [['qk-ready', 'QK_READY'], ['softmax-ready', 'SOFTMAX_READY'], ['pv-ready', 'PV_READY']]) {
    const edge = result.edges.find((item) => item.id === `sync:s0:${eventId}`);
    const record = w.PTO_PYPA_SEMANTIC_GRAPH.synchronizations.find((item) => item.id === eventId);
    assert.equal(edge.synchronization, record);
    assert.equal(edge.synchronization.name, expectedName);
    assert.match(edge.explanation, /SET|写完/);
    assert.match(edge.explanation, /WAIT|读取前/);
    assert.equal(edge.source.file, 'paged_attention_pypto.py');
    assert.deepEqual(plain(edge.source.ranges), plain(record.producer.source.ranges.concat(record.consumer.source.ranges)));
  }
});

test('state and reuse evidence points to matching source statements and all edges form a DAG', () => {
  const w = load(), source = w.PTO_PAGED_ATTENTION_PYPTO_SOURCE.split('\n');
  const pipeline = w.PtoPagedAttentionIntro.model.pipelineFor(1537);
  const result = w.PtoPagedAttentionStackDependencyModel.build(pipeline, w.PTO_PYPA_SEMANTIC_GRAPH.synchronizations);
  const checks = [
    ['state:rescale:s0', /pending0_after = pl\.tile\.move/],
    ['reuse:score_transfer:s0:s3', /score_transfer/],
    ['reuse:probability_transfer:s0:s3', /probability_transfer/],
    ['reuse:pv_transfer:s0:s3', /pv_transfer/],
  ];
  for (const [id, pattern] of checks) {
    const edge = result.edges.find((item) => item.id === id);
    assert.ok(edge, `${id} exists`);
    assert.equal(edge.source.file, 'paged_attention_pypto.py');
    const cited = edge.source.ranges.map(({ start, end }) => {
      assert.ok(start >= 1 && end >= start && end <= source.length, `${id} range in source`);
      return source.slice(start - 1, end).join('\n');
    }).join('\n');
    assert.match(cited, pattern, `${id} source range mentions its buffer/state`);
  }
  assert.match(result.edges.find((edge) => edge.id === 'state:rescale:s0').explanation, /tick 0.*tick 1.*tick 2/);
  assert.equal(isDag(result.nodes, result.edges), true);
});


test('slot reuse waits for each buffer’s own last reader, and state versions stay on their stages', () => {
  const w = load(), result = w.PtoPagedAttentionStackDependencyModel.build(w.PtoPagedAttentionIntro.model.pipelineFor(3584), w.PTO_PYPA_SEMANTIC_GRAPH.synchronizations);
  for (const stack of result.stacks.filter((s) => s.previousStack !== null)) {
    const previous = stack.previousStack;
    for (const [buffer, reader, writer] of [['score_transfer', 'softmax', 'qk'], ['probability_transfer', 'pv', 'softmax'], ['pv_transfer', 'accumulate', 'pv']]) {
      const edge = result.edges.find((e) => e.id === `reuse:${buffer}:s${previous}:s${stack.id}`);
      assert.equal(edge.from, `s${previous}:${reader}`);
      assert.equal(edge.to, `s${stack.id}:${writer}`);
      assert.equal(result.stacks[previous].slot, stack.slot);
      assert.match(edge.explanation, /静态/);
    }
  }
  const crossState = result.edges.filter((e) => e.kind === 'state' && e.from.split(':')[0] !== e.to.split(':')[0]);
  for (const edge of crossState) {
    assert.equal(edge.from.split(':')[1], edge.to.split(':')[1]);
    assert.ok(['softmax', 'accumulate'].includes(edge.from.split(':')[1]));
  }
  assert.equal(new Set(result.nodes.map((n) => n.id)).size, result.nodes.length);
  assert.equal(new Set(result.edges.map((e) => e.id)).size, result.edges.length);
});

test('each Stack points to its own full or tail PV wait branch using shared synchronization metadata', () => {
  const w = load();
  for (const [seqLen, pages, line] of [[1025, 1, 522], [1153, 2, 576], [1281, 3, 662], [1536, 4, 430]]) {
    const result = w.PtoPagedAttentionStackDependencyModel.build(w.PtoPagedAttentionIntro.model.pipelineFor(seqLen), w.PTO_PYPA_SEMANTIC_GRAPH.synchronizations);
    const edge = result.edges.find((e) => e.id === 'sync:s2:softmax-ready');
    assert.equal(result.stacks[2].pages.length, pages);
    assert.equal(edge.consumerSource.ranges.length, 1);
    assert.equal(edge.consumerSource.ranges[0].start, line);
    assert.match(edge.explanation, new RegExp(`${pages} Pages`));
  }
});
