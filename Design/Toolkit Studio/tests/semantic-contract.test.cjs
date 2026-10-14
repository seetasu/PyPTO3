const test = require('node:test');
const assert = require('node:assert/strict');
require('../js/semantic-computation-graph.js');
require('../js/semantic-computation-presentation.js');
const contract = globalThis.PTO_SEMANTIC_COMPUTATION_GRAPH;
const presentation = globalThis.PTO_COMPUTATION_PRESENTATION;
const clone = (value) => structuredClone(value);
const src = { file: 'kernel.py', ranges: [{ start: 2, end: 3 }], symbols: ['tile'] };
const data = (id, entity) => ({ id, type: 'data', name: id, description: '数据实体', source: clone(src),
  provenance: [{ kind: 'fact', origin: 'source', fields: ['data.role', 'data.granularity', 'data.entity'] }],
  ports: { inputs: { in: { entity } }, outputs: { out: { entity } } },
  data: { role: 'intermediate', granularity: 'tile', entity } });
const base = () => ({ schemaVersion: 2, id: 'root', sourceFiles: { 'kernel.py': { lineCount: 10 } },
  nodes: [data('a', 'score'), data('b', 'score')],
  edges: [{ id: 'flow', s: 'a', t: 'b', type: 'data', sourcePort: 'out', targetPort: 'in' }] });

test('valid graph and derived presentation retain source and hide missing attributes', () => {
  const graph = base();
  assert.equal(contract.validate(graph), graph);
  const card = presentation.node(graph.nodes[0]);
  assert.equal(card.title, 'a');
  assert.equal(card.api, '');
  assert.equal(card.sourceLine, 2);
  assert.deepEqual(card.sourceRanges, [[2, 3]]);
  const inspector = presentation.inspector(card, { model: graph, qualifiedId: 'a' });
  assert.match(JSON.stringify(inspector), /消费者/);
  assert.doesNotMatch(JSON.stringify(inspector), /undefined|null/);
});

test('rejects unknown values, source overflow and missing provenance coverage', () => {
  const unknown = base(); unknown.nodes[0].data.dtype = '待确认';
  assert.throws(() => contract.validate(unknown), /unknown placeholder/);
  const overflow = base(); overflow.nodes[0].source.ranges[0].end = 11;
  assert.throws(() => contract.validate(overflow), /lineCount/);
  const uncovered = base(); uncovered.nodes[0].data.dtype = 'fp16';
  assert.throws(() => contract.validate(uncovered), /data.dtype/);
});

test('checks edge ports and composite child mappings', () => {
  const broken = base(); broken.edges[0].sourcePort = 'missing';
  assert.throws(() => contract.validate(broken), /missing edge port/);
  const graph = base();
  const child = data('inner', 'score');
  graph.nodes = [{ id: 'shell', type: 'composite', name: '组合', description: '展开计算', source: clone(src),
    provenance: [{ kind: 'resolved', origin: 'source', fields: ['composite.semantics'] }], ports: { inputs: { in: { entity: 'score' } }, outputs: { out: { entity: 'score' } } },
    composite: { semantics: 'compose', category: 'compute', api: [],
      subgraph: { id: 'child', nodes: [child], edges: [] },
      portMappings: { inputs: { in: { node: 'inner', port: 'in', relation: 'identity' } },
        outputs: { out: { node: 'inner', port: 'out', relation: 'identity' } } } } }];
  graph.edges = [];
  assert.equal(contract.validate(graph), graph);
  graph.nodes[0].composite.portMappings.outputs.out.port = 'missing';
  assert.throws(() => contract.validate(graph), /missing child endpoint port/);
});

test('identity mappings require matching entity and dtype when both provided', () => {
  const graph = base();
  graph.nodes = [{ id: 'shell', type: 'composite', name: '组合', description: '展开计算', source: clone(src),
    provenance: [{ kind: 'resolved', origin: 'source', fields: ['composite.semantics'] }], ports: { inputs: { in: { entity: 'score', dtype: 'fp16' } }, outputs: {} },
    composite: { semantics: 'compose', category: 'compute', api: [],
      subgraph: { id: 'child', nodes: [data('inner', 'score')], edges: [] },
      portMappings: { inputs: { in: { node: 'inner', port: 'in', relation: 'identity' } }, outputs: {} } } }];
  graph.edges = [];
  graph.nodes[0].composite.subgraph.nodes[0].ports.inputs.in.dtype = 'fp32';
  assert.throws(() => contract.validate(graph), /dtype mismatch/);
  graph.nodes[0].composite.subgraph.nodes[0].ports.inputs.in.dtype = 'fp16';
  graph.nodes[0].composite.subgraph.nodes[0].ports.inputs.in.entity = 'other';
  assert.throws(() => contract.validate(graph), /entity mismatch/);
});
