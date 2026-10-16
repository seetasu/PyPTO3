/* Synchronization metadata is source-backed execution information, separate from data-flow edges. */
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const js = path.join(__dirname, '..', 'js');
function load() {
  const context = vm.createContext({ window: {} });
  for (const name of [
    'paged-attention-pypto-source',
    'semantic-computation-graph',
    'semantic-computation-presentation',
    'semantic-computation-layout',
    'paged-attention-semantic-graph',
  ]) {
    vm.runInContext(fs.readFileSync(path.join(js, name + '.js'), 'utf8'), context, { filename: name + '.js' });
  }
  return context.window;
}

function allNodes(graph, prefix = '') {
  return graph.nodes.flatMap((node) => [
    { node, id: prefix + node.id },
    ...(node.composite ? allNodes(node.composite.subgraph, prefix + node.id + '/') : []),
  ]);
}
const plain = (value) => JSON.parse(JSON.stringify(value));
const endpointId = (endpoint) => endpoint.anchor;

const EXPECTED = [
  {
    id: 'qk-ready', eventId: 0, name: 'QK_READY', buffer: 'score_transfer', tensor: 'scores', dtype: 'FP32',
    producer: { node: 'qk', anchor: 'qk/score', coreType: 'AIC', pipe: 'FIX', ranges: [[357, 362]], fftsMode: 2 },
    consumer: { node: 'softmax', anchor: 'scale-mask/scale', boundary: 'scale-mask', coreType: 'AIV', pipe: 'MTE2', ranges: [[765, 769]] },
    sourceChecks: [
      ['producer', /pl\.system\.sync_set\(\s*QK_READY_EVENT[\s\S]*?pipe=pl\.PipeType\.FIX[\s\S]*?ffts_mode=2[\s\S]*?core_type=pl\.KernelType\.AIC/],
      ['consumer', /pl\.system\.sync_wait\(\s*QK_READY_EVENT[\s\S]*?pipe=pl\.PipeType\.MTE2[\s\S]*?core_type=pl\.KernelType\.AIV/],
    ],
  },
  {
    id: 'softmax-ready', eventId: 1, name: 'SOFTMAX_READY', buffer: 'probability_transfer', tensor: 'probability', dtype: 'BF16',
    producer: { node: 'softmax', anchor: 'softmax/probability', coreType: 'AIV', pipe: 'MTE3', ranges: [[817, 822]], fftsMode: 2 },
    consumer: { node: 'pv', anchor: 'pv/probability', coreType: 'AIC', pipe: 'MTE2', ranges: [[430, 434], [522, 526], [576, 580], [662, 666]] },
    sourceChecks: [
      ['producer', /pl\.system\.sync_set\(\s*SOFTMAX_READY_EVENT[\s\S]*?pipe=pl\.PipeType\.MTE3[\s\S]*?ffts_mode=2[\s\S]*?core_type=pl\.KernelType\.AIV/],
      ['consumer', /pl\.system\.sync_wait\(\s*SOFTMAX_READY_EVENT[\s\S]*?pipe=pl\.PipeType\.MTE2[\s\S]*?core_type=pl\.KernelType\.AIC/],
    ],
  },
  {
    id: 'pv-ready', eventId: 2, name: 'PV_READY', buffer: 'pv_transfer', tensor: 'pv/partial', dtype: 'FP32',
    producer: { node: 'pv', anchor: 'pv/partial', coreType: 'AIC', pipe: 'FIX', ranges: [[719, 724]], fftsMode: 2 },
    consumer: { node: 'context-acc', anchor: 'context-acc/partial', coreType: 'AIV', pipe: 'MTE2', ranges: [[830, 834]] },
    sourceChecks: [
      ['producer', /pl\.system\.sync_set\(\s*PV_READY_EVENT[\s\S]*?pipe=pl\.PipeType\.FIX[\s\S]*?ffts_mode=2[\s\S]*?core_type=pl\.KernelType\.AIC/],
      ['consumer', /pl\.system\.sync_wait\(\s*PV_READY_EVENT[\s\S]*?pipe=pl\.PipeType\.MTE2[\s\S]*?core_type=pl\.KernelType\.AIV/],
    ],
  },
];

test('three synchronization records match their source APIs, parameters, and exact line ranges', () => {
  const w = load();
  const model = w.PTO_PYPA_SEMANTIC_GRAPH;
  const sourceLines = w.PTO_PAGED_ATTENTION_PYPTO_SOURCE.trimEnd().split('\n');
  assert.equal(model.synchronizations.length, EXPECTED.length);

  for (const expected of EXPECTED) {
    const event = model.synchronizations.find((item) => item.id === expected.id);
    assert.ok(event, `${expected.id} exists`);
    for (const key of ['id', 'eventId', 'name', 'buffer', 'tensor', 'dtype']) assert.equal(event[key], expected[key], `${expected.id}.${key}`);
    assert.ok(event.explanation.trim().length > 0, `${expected.id} has an explanation`);

    for (const side of ['producer', 'consumer']) {
      const want = expected[side], actual = event[side];
      assert.equal(actual.node, want.node, `${expected.id}.${side}.node`);
      assert.equal(actual.anchor, want.anchor, `${expected.id}.${side}.anchor`);
      assert.equal(actual.coreType, want.coreType, `${expected.id}.${side}.coreType`);
      assert.equal(actual.pipe, want.pipe, `${expected.id}.${side}.pipe`);
      assert.equal(actual.source.file, 'paged_attention_pypto.py', `${expected.id}.${side}.source.file`);
      assert.deepEqual(plain(actual.source.ranges.map(({ start, end }) => [start, end])), want.ranges, `${expected.id}.${side}.source.ranges`);
      if (want.fftsMode === undefined) assert.equal('fftsMode' in actual, false, `${expected.id}.${side} has no fftsMode`);
      else assert.equal(actual.fftsMode, want.fftsMode, `${expected.id}.${side}.fftsMode`);
      if (want.boundary) assert.equal(actual.boundary, want.boundary, `${expected.id}.${side}.boundary`);

      const citedText = actual.source.ranges.map(({ start, end }) => {
        assert.ok(start >= 1 && end >= start && end <= sourceLines.length, `${expected.id}.${side} line range is in source`);
        return sourceLines.slice(start - 1, end).join('\n');
      }).join('\n');
      const check = expected.sourceChecks.find(([which]) => which === side)[1];
      assert.match(citedText, check, `${expected.id}.${side} cites the expected sync API and hardware parameters`);
    }
  }
});

test('synchronization anchors name declared graph nodes and never become graph nodes or edges', () => {
  const model = load().PTO_PYPA_SEMANTIC_GRAPH;
  const ids = new Set(allNodes(model).map(({ id }) => id));
  for (const event of model.synchronizations) {
    for (const endpoint of [event.producer, event.consumer]) {
      assert.ok(ids.has(endpoint.node), `${event.id}: node ${endpoint.node} exists`);
      assert.ok(ids.has(endpoint.anchor), `${event.id}: anchor ${endpoint.anchor} exists`);
    }
  }
  assert.ok(model.nodes.every((node) => !/sync|event/i.test(node.type)), 'root nodes contain no synchronization type');
  assert.ok(model.edges.every((edge) => !/sync|event/i.test(edge.type)), 'root edges contain no synchronization type');
  assert.equal(model.nodes.some((node) => node.id === 'qk-ready' || node.id === 'softmax-ready' || node.id === 'pv-ready'), false);
  assert.equal(model.edges.some((edge) => model.synchronizations.some((event) => event.id === edge.id)), false);
});

test('all QK, Softmax, and PV expanded or collapsed combinations can resolve both endpoints', () => {
  const w = load();
  const model = w.PTO_PYPA_SEMANTIC_GRAPH;
  const stages = ['qk', 'softmax', 'pv'];
  for (let mask = 0; mask < 2 ** stages.length; mask += 1) {
    const expanded = new Set(stages.filter((_, index) => mask & (1 << index)));
    const visible = w.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize(model, expanded);
    const visibleIds = new Set(visible.nodes.map((node) => node.id));
    for (const event of model.synchronizations) {
      for (const side of ['producer', 'consumer']) {
        const endpoint = event[side];
        const resolved = visibleIds.has(endpointId(endpoint))
          ? endpointId(endpoint)
          : (endpoint.boundary || endpoint.node);
        assert.ok(visibleIds.has(resolved), `${[...expanded].join(',') || 'all collapsed'}: ${event.id}.${side} resolves to ${resolved}`);
      }
    }
    const bridges = w.PTO_COMPUTATION_LAYOUT.synchronizationBridges({ ...visible, synchronizations: model.synchronizations });
    assert.equal(bridges.length, model.synchronizations.length);
    for (const bridge of bridges) {
      const event = model.synchronizations.find((item) => item.id === bridge.event.id);
      const endpoint = (side) => {
        const ep = event[side];
        return visibleIds.has(ep.anchor) ? ep.anchor : (ep.boundary || ep.node);
      };
      assert.equal(bridge.producer, endpoint('producer'), `${event.id} producer bridge anchor`);
      assert.equal(bridge.consumer, endpoint('consumer'), `${event.id} consumer bridge anchor`);
    }
  }
});

test('all 32 boundary expansion combinations keep markers clear of computational nodes without mutating geometry', () => {
  const w = load(), model = w.PTO_PYPA_SEMANTIC_GRAPH;
  const stages = ['qk', 'scale-mask', 'softmax', 'pv', 'context-acc'];
  for (let mask = 0; mask < 32; mask++) {
    const graph = w.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize(model, new Set(stages.filter((_, i) => mask & (1 << i))));
    const before = JSON.stringify(graph);
    const bridges = w.PTO_COMPUTATION_LAYOUT.synchronizationBridges(graph);
    assert.equal(bridges.length, 3);
    assert.equal(new Set(bridges.map((b) => b.event.id)).size, 3);
    assert.equal(JSON.stringify(graph), before);
    for (const bridge of bridges) {
      for (const node of graph.nodes.filter((n) => !n.expandedGroup)) {
        const intersects = bridge.x < node.x + node.w && bridge.x + bridge.w > node.x && bridge.y < node.y + node.h && bridge.y + bridge.h > node.y;
        assert.equal(intersects, false, `mask ${mask}: ${bridge.event.name} overlaps ${node.id}`);
      }
      const detail = w.PTO_COMPUTATION_PRESENTATION.synchronization(bridge.event, model);
      assert.ok(detail.rows.some((row) => row.value.includes(bridge.event.buffer)));
      assert.equal(detail.sources.length, bridge.event.producer.source.ranges.length + bridge.event.consumer.source.ranges.length);
    }
  }
});
