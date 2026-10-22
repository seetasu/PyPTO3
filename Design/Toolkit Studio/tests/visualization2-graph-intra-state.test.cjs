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
    'paged-attention-stack-dependency-model', 'visualization2-graph-intra']) {
    vm.runInContext(fs.readFileSync(path.join(js, `${name}.js`), 'utf8'), context, { filename: name });
  }
  return context.window;
}

test('Task Stack evolution follows source-backed state versions and delayed FIFO ticks', () => {
  const w = load();
  for (const [seqLen, count] of [[128, 1], [1025, 3], [3584, 7]]) {
    const task = { seqLen, stackCount: count, graph: { synchronizations: w.PTO_PYPA_SEMANTIC_GRAPH.synchronizations } };
    for (const index of [0, 1, 2, count - 1].filter(i => i < count)) {
      const step = w.PtoVisualization2GraphIntra.stackEvolution(task, index);
      assert.equal(step.produceTick, index);
      assert.equal(step.consumeTick, index + 2);
      assert.equal(step.states.m.previous, index ? `m${index - 1}` : 'm初始 = −3e38');
      assert.equal(step.states.l.previous, index ? `l${index - 1}` : 'l初始 = 0');
      assert.equal(step.states.o.previous, index ? `o${index - 1}` : 'o初始 = 0');
      assert.equal(step.states.o.updated, `o${index}`);
      assert.equal(step.next, index + 1 < count ? `S${index + 1}` : null);
      assert.deepEqual(JSON.parse(JSON.stringify(step.fifo.versions)), [
        `tick ${index}: pending1=r${index}`,
        `tick ${index + 1}: pending0=r${index}`,
        `tick ${index + 2}: consume r${index}`,
      ]);
      assert.equal(step.stateEdges.length, index ? 3 : 1);
      assert.equal(step.dataEdges.length, 3);
      assert.equal(step.fifo.source.file, 'paged_attention_pypto.py');
    }
    assert.equal(w.PtoVisualization2GraphIntra.stackEvolution(task, count), null);
  }
});
