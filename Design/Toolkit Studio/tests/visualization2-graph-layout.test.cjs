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

function verifyOverviewGeometry(graph) {
  const overlap = (a, b) => Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x) > 1 &&
    Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y) > 1;
  graph.nodes.forEach((a, index) => graph.nodes.slice(index + 1).forEach((b) => {
    if (a.id.startsWith(`${b.id}/`) || b.id.startsWith(`${a.id}/`)) return;
    assert.equal(overlap(a, b), false, `nodes ${a.id} / ${b.id} overlap`);
  }));
  const topRegions = graph.groups.filter((group) => !group.id.includes('/'));
  topRegions.forEach((a, index) => topRegions.slice(index + 1).forEach((b) =>
    assert.equal(overlap(a, b), false, `regions ${a.id} / ${b.id} overlap`)));
  for (const region of topRegions) for (const node of graph.nodes) {
    if (node.id.includes('/')) continue;
    assert.ok(node.y >= region.y + 48 || !overlap(node, region),
      `region title ${region.id} covers ${node.id}`);
  }
  const byId = new Map(graph.nodes.map((node) => [node.id, node]));
  graph.edgeRoutes.forEach((route, index) => {
    const edge = graph.edges[index];
    [[route.points[0], byId.get(edge.s)], [route.points.at(-1), byId.get(edge.t)]].forEach(([point, node]) => {
      assert.ok(node, `missing endpoint for ${edge.id}`);
      const near = Math.min(Math.abs(point.x - node.x), Math.abs(point.x - node.x - node.w),
        Math.abs(point.y - node.y), Math.abs(point.y - node.y - node.h));
      assert.ok(near < 1, `${edge.id} misses ${node.id} boundary by ${near}`);
    });
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
    verifyOverviewGeometry(graph);
    if (model.nodes.some((node) => node.type === 'data' && node.data.role === 'intermediate'))
      assert.ok(graph.edges.some((edge) => edge.sourceEdgeIds?.length > 1));
    assert.ok(graph.nodes.every((node) => node.type !== 'data' || node.data.role !== 'intermediate'));
    for (const edge of graph.edges.filter((entry) => entry.sourceEdgeIds)) {
      assert.ok(edge.sourceEdgeIds.every((id) => model.edges.some((original) => original.id === id)));
    }
    const first = overview.nodes.find((node) => node.type === 'composite');
    if (first) {
      const expanded = w.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize(overview,
        new Set([first.id]), { flowLayout: true, overviewLayout: true });
      verifyRoutes(expanded);
      verifyOverviewGeometry(expanded);
    }
  }
});

test('state evolution expansion retains current inputs, feedback, and phase order', () => {
  const w = load(), model = w.PTO_PYPA_SEMANTIC_GRAPH;
  const overview = w.PtoComputationVisibility.overview(model);
  for (const composite of ['softmax', 'context-acc']) {
    const graph = w.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize(overview, new Set([composite]),
      { flowLayout: true, overviewLayout: true });
    const node = (id) => graph.nodes.find((item) => item.id === `${composite}/${id}`);
    const phases = model.nodes.find((item) => item.id === composite).composite.subgraph.stateEvolution;
    const bottom = (ids) => Math.max(...ids.map((id) => node(id).y + node(id).h));
    const top = (ids) => Math.min(...ids.map((id) => node(id).y));
    assert.ok(bottom(phases.previous) < top(phases.current), composite);
    assert.ok(bottom(phases.current) < top(phases.updated), composite);
    assert.ok(graph.edges.some((edge) => edge.s === `${composite}/updated` && edge.type === 'feedback'));
    assert.ok(graph.edges.some((edge) => edge.s === `${composite}/${composite === 'softmax' ? 'scores' : 'partial'}`));
    verifyRoutes(graph);
    verifyOverviewGeometry(graph);
    const feedback = graph.edges.find((edge) => edge.s === `${composite}/updated` && edge.type === 'feedback');
    const sections = w.PTO_COMPUTATION_PRESENTATION.inspectorEdge(feedback, { model });
    assert.ok(sections.some((section) => section.rows.some((row) => row.label === '更新公式')));
    assert.ok(sections.some((section) => section.title === '源码'));
  }
});

test('execution decks reserve region and routing space for 9, 17 and 28 instances',()=>{
 const w=load(),model=w.PtoComputationVisibility.overview(w.PTO_PYPA_SEMANTIC_GRAPH);
 for(const count of [9,17,28])for(const expanded of [new Set(),new Set(['qk','pv'])]){
  const base=w.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize(model,expanded,{flowLayout:true,overviewLayout:true});
  const footprints=Object.fromEntries(base.nodes.filter(n=>!n.expandedGroup&&['operation','composite'].includes(n.type)).map(n=>[n.id,{right:3*(count-1),bottom:1.5*(count-1)}]));
  const graph=w.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize(model,expanded,{flowLayout:true,overviewLayout:true,executionFootprints:footprints});
  const occupied={...graph,nodes:graph.nodes.map(n=>({...n,w:n.w+(n.executionFootprint?.right||0),h:n.h+(n.executionFootprint?.bottom||0)}))};
  verifyOverviewGeometry(graph);verifyRoutes(occupied);
  occupied.nodes.forEach((a,i)=>occupied.nodes.slice(i+1).forEach(b=>{
   if(a.id.startsWith(b.id+'/')||b.id.startsWith(a.id+'/'))return;
   assert.ok(Math.min(a.x+a.w,b.x+b.w)-Math.max(a.x,b.x)<=1||Math.min(a.y+a.h,b.y+b.h)-Math.max(a.y,b.y)<=1,`deck ${a.id} overlaps ${b.id}`);
  }));
 }
});
