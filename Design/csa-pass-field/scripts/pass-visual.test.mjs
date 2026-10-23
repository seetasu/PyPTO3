import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { layoutVisibleGraph } from '../src/layout.js';
import { controlFacts, csaLines, structuralFacts } from './visual-facts.mjs';

const read = (name) => JSON.parse(fs.readFileSync(new URL(`../data/csa-qk-pv/${name}.json`, import.meta.url)));
const graph = read('graph'), passes = read('passes').passes, view = read('views').views.front;
const dumpRoot = new URL('../../../Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/passes_dump/', import.meta.url);
function snapshot(index) {
  const file = fs.readdirSync(dumpRoot).find((name) => name.startsWith(String(index).padStart(2, '0') + '_'));
  return { index, fileRel: file, sourceText: fs.readFileSync(new URL(file, dumpRoot), 'utf8') };
}
function stateAt(index) {
  const nodes = new Map(graph.nodes.map((n) => [n.id, structuredClone(n)]));
  const active = new Set(graph.initialState.nodeIds), edges = new Set(graph.initialState.edgeIds);
  for (const pass of passes.filter((p) => p.index <= index)) for (const d of pass.delta) {
    if (d.op === 'addNode') active.add(d.id);
    if (d.op === 'removeNode') active.delete(d.id);
    if (d.op === 'updateNode') Object.assign(nodes.get(d.id), d.after);
    if (d.op === 'addEdge') edges.add(d.id);
    if (d.op === 'removeEdge') edges.delete(d.id);
  }
  return { nodes: [...active].map(id => nodes.get(id)).filter(n => n.renderKind !== 'context'), edges: graph.edges.filter(e => edges.has(e.id)) };
}
test('early loop unroll and SSA come from CSA source', () => {
  assert.equal(controlFacts(snapshot(1)).gathers.length, 1);
  assert.equal(controlFacts(snapshot(1)).runCount, 5);
  assert.equal(controlFacts(snapshot(2)).gathers.length, 5);
  assert.equal(controlFacts(snapshot(4)).ssa, true);
  assert.equal(controlFacts(snapshot(5)).lowerBound, '0');
});
test('TFREE split and pipeline stage changes are observable', () => {
  const facts = i => structuralFacts(csaLines(snapshot(i)));
  assert.ok(facts(25).freeSplits.every(x => x === null));
  assert.ok(facts(26).freeSplits.every(x => x === '0'));
  assert.ok(facts(29).loops.some(x => x.stage === 2));
  assert.ok(!facts(30).loops.some(x => x.stage === 2));
});
test('no-op and changes outside CSA never invent a graph rewrite', () => {
  for (const p of passes) if (p.noOp || (!p.familySourceChanged && p.index > 10)) assert.equal(p.delta.length, 0, p.name);
  for (const p of passes) if (p.familySourceChanged) assert.ok(p.delta.some(d => d.id !== 'control.qk'), p.name);
});
test('all stages have finite layouts, and mixed-kernel extraction keeps OP axes', () => {
  let previous;
  for (let i = 1; i <= 51; i++) {
    const state = stateAt(i);
    const result = layoutVisibleGraph(state.nodes, state.edges, view, { referenceNodes: graph.nodes, referenceEdges: graph.edges });
    for (const n of state.nodes) assert.ok(Number.isFinite(result.positions.get(n.id)?.x) && Number.isFinite(result.positions.get(n.id)?.y), `${i}: ${n.id}`);
    const point = result.positions.get('op.qk');
    if (previous) assert.deepEqual(point, previous, `Pass ${i} moves QK`);
    previous = point;
  }
});
