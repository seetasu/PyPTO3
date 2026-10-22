const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const js = path.join(__dirname, '..', 'js');
function load() {
  const context = vm.createContext({ window: {} });
  for (const name of ['paged-attention-pypto-source', 'semantic-computation-presentation',
    'semantic-computation-layout', 'semantic-computation-graph', 'paged-attention-semantic-graph',
    'matmul-semantic-graph', 'semantic-computation-visibility']) {
    vm.runInContext(fs.readFileSync(path.join(js, `${name}.js`), 'utf8'), context);
  }
  return context.window;
}

function intersects(a, b, node) {
  if (Math.abs(a.x - b.x) < .5) return a.x > node.x + 1 && a.x < node.x + node.w - 1 &&
    Math.max(a.y, b.y) > node.y + 1 && Math.min(a.y, b.y) < node.y + node.h - 1;
  return a.y > node.y + 1 && a.y < node.y + node.h - 1 &&
    Math.max(a.x, b.x) > node.x + 1 && Math.min(a.x, b.x) < node.x + node.w - 1;
}

function verifyRoutes(graph) {
  assert.equal(graph.edgeRoutes.length, graph.edges.length);
  graph.edges.forEach((edge, index) => {
    const route = graph.edgeRoutes[index];
    assert.ok(route.points.length >= 2, edge.id);
    assert.ok(!route.path.includes('NaN'), edge.id);
    for (const node of graph.nodes) {
      if ([edge.s, edge.t].includes(node.id) || node.expandedGroup) continue;
      for (let i = 1; i < route.points.length; i++) {
        assert.equal(intersects(route.points[i - 1], route.points[i], node), false,
          `${edge.id} crosses ${node.id}`);
      }
    }
  });
}

test('flow layout keeps the computation spine aligned and state in a side rail', () => {
  const w = load(), model = w.PTO_PYPA_SEMANTIC_GRAPH;
  const graph = w.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize(model, new Set(), { flowLayout: true });
  const node = (id) => graph.nodes.find((item) => item.id === id);
  const center = (id) => node(id).x + node(id).w / 2;
  for (const id of ['qk', 'softmax', 'pv', 'context-acc', 'output-norm']) {
    assert.equal(center(id), center('qk'), `${id} remains on the computation spine`);
  }
  assert.ok(node('context').x > node('context-acc').x + node('context-acc').w);
  assert.equal(graph.edges.length, model.edges.length);
  verifyRoutes(graph);
  const bridges = w.PTO_COMPUTATION_LAYOUT.synchronizationBridges(graph);
  assert.equal(bridges.length, model.synchronizations.length);
  for (const bridge of bridges) {
    assert.ok(!bridge.path.includes('NaN'));
    for (const item of graph.nodes) {
      assert.ok(bridge.x >= item.x + item.w || bridge.x + bridge.w <= item.x ||
        bridge.y >= item.y + item.h || bridge.y + bridge.h <= item.y,
      `${bridge.event.id} marker overlaps ${item.id}`);
    }
  }
});

test('routes avoid unrelated nodes after expansion and data contraction', () => {
  const w = load(), model = w.PTO_PYPA_SEMANTIC_GRAPH;
  const expanded = new Set(['q-prep', 'q-prep/norm', 'qk', 'softmax', 'pv', 'context-acc']);
  const graph = w.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize(model, expanded, { flowLayout: true });
  verifyRoutes(graph);
  const contracted = w.PtoComputationVisibility.withoutData(graph, { flowLayout: true });
  verifyRoutes(contracted);
  assert.ok(contracted.edges.some((edge) => edge.sourceEdgeIds?.length > 1));
});

test('flow routing also works for the Matmul semantic graph', () => {
  const w = load(), model = w.PTO_MATMUL_SEMANTIC_GRAPH;
  assert.ok(model);
  const graph = w.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize(model, new Set(), { flowLayout: true });
  verifyRoutes(graph);
  const expanded = new Set();
  const collect = (scope, prefix = '') => scope.nodes.forEach((node) => {
    if (!node.composite) return;
    const id = prefix + node.id;
    expanded.add(id);
    collect(node.composite.subgraph, `${id}/`);
  });
  collect(model);
  verifyRoutes(w.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize(model, expanded, { flowLayout: true }));
});

test('Level 1 contracts transient data without losing dependency provenance or expansion', () => {
  const w = load();
  for (const model of [w.PTO_PYPA_SEMANTIC_GRAPH, w.PTO_MATMUL_SEMANTIC_GRAPH]) {
    const overview = w.PtoComputationVisibility.overview(model);
    const graph = w.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize(overview, new Set(),
      { flowLayout: true, overviewLayout: true });
    verifyRoutes(graph);
    if (model.nodes.some((node) => node.type === 'data' && node.data.role === 'intermediate'))
      assert.ok(graph.edges.some((edge) => edge.sourceEdgeIds?.length > 1));
    assert.ok(graph.nodes.every((node) => node.type !== 'data' || node.data.role !== 'intermediate'));
    for (const edge of graph.edges.filter((entry) => entry.sourceEdgeIds)) {
      assert.ok(edge.sourceEdgeIds.every((id) => model.edges.some((original) => original.id === id)));
    }
    const first = overview.nodes.find((node) => node.type === 'composite');
    if (first) verifyRoutes(w.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize(overview,
      new Set([first.id]), { flowLayout: true, overviewLayout: true }));
  }
});
