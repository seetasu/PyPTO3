import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '../../..');
const DATA = path.join(HERE, '../data/csa-qk-pv');
const read = (name) => JSON.parse(fs.readFileSync(path.join(DATA, name), 'utf8'));
const fail = (message) => { throw new Error(message); };
const assert = (condition, message) => { if (!condition) fail(message); };

const manifest = read('manifest.json');
const graph = read('graph.json');
const passes = read('passes.json');
const isa = read('isa.json');
const views = read('views.json');

assert(manifest.schemaVersion === 'csa-pass-field.manifest.v1', 'manifest schemaVersion');
assert(graph.schemaVersion === 'csa-pass-field.graph.v1', 'graph schemaVersion');
assert(passes.schemaVersion === 'csa-pass-field.passes.v1', 'passes schemaVersion');
assert(isa.schemaVersion === 'csa-pass-field.isa.v1', 'isa schemaVersion');
assert(views.schemaVersion === 'csa-pass-field.views.v1', 'views schemaVersion');
assert(manifest.passCount === 51, `expected 51 Passes, got ${manifest.passCount}`);
assert(passes.passes.length === manifest.passCount, 'manifest/pass count mismatch');
assert(passes.passes.every((pass, index) => pass.index === index + 1), 'Pass indices must be continuous 1..51');

const unique = (items, label) => {
  const ids = items.map((item) => item.id);
  assert(new Set(ids).size === ids.length, `${label} IDs must be unique`);
  return new Set(ids);
};
const nodeIds = unique(graph.nodes, 'node');
const edgeIds = unique(graph.edges, 'edge');
for (const edge of graph.edges) {
  assert(nodeIds.has(edge.source), `${edge.id}: missing source ${edge.source}`);
  assert(nodeIds.has(edge.target), `${edge.id}: missing target ${edge.target}`);
}
for (const id of graph.initialState.nodeIds) assert(nodeIds.has(id), `initial node missing: ${id}`);
for (const id of graph.initialState.edgeIds) assert(edgeIds.has(id), `initial edge missing: ${id}`);
for (const id of graph.projections.isa.nodeIds) assert(nodeIds.has(id), `ISA projection node missing: ${id}`);
for (const id of graph.projections.isa.edgeIds) assert(edgeIds.has(id), `ISA projection edge missing: ${id}`);
for (const id of nodeIds) assert(views.views.front.nodes[id], `front view missing node anchor: ${id}`);

for (const pass of passes.passes) {
  if (pass.noOp) assert(pass.delta.length === 0, `${pass.name}: a no-op cannot carry a visual delta`);
  for (const op of pass.delta) {
    if (op.op.endsWith('Node')) assert(nodeIds.has(op.id), `${pass.name}: missing node ${op.id}`);
    if (op.op.endsWith('Edge')) assert(edgeIds.has(op.id), `${pass.name}: missing edge ${op.id}`);
  }
  for (const sourceRef of pass.sourceRefs) assert(fs.existsSync(path.join(ROOT, sourceRef.file)), `${pass.name}: missing source ${sourceRef.file}`);
}

function initialState() {
  return {
    nodes: new Set(graph.initialState.nodeIds),
    edges: new Set(graph.initialState.edgeIds),
    nodeData: new Map(graph.nodes.map((item) => [item.id, structuredClone(item)])),
  };
}

function apply(state, operation, reverse = false) {
  const before = reverse ? operation.after : operation.before;
  const after = reverse ? operation.before : operation.after;
  if (operation.op === 'addNode' || operation.op === 'removeNode') {
    if (after?.active) state.nodes.add(operation.id); else state.nodes.delete(operation.id);
  } else if (operation.op === 'addEdge' || operation.op === 'removeEdge') {
    if (after?.active) state.edges.add(operation.id); else state.edges.delete(operation.id);
  } else if (operation.op === 'updateNode') {
    const target = state.nodeData.get(operation.id);
    const patch = after || {};
    const other = before || {};
    for (const key of new Set([...Object.keys(other), ...Object.keys(patch)])) {
      if (!(key in patch)) delete target[key];
    }
    Object.assign(target, structuredClone(patch));
  }
}

const hashState = (state) => crypto.createHash('sha256').update(JSON.stringify({
  nodes: [...state.nodes].sort(), edges: [...state.edges].sort(),
  nodeData: [...state.nodeData].sort(([a], [b]) => a.localeCompare(b)),
})).digest('hex');
const replay = initialState();
const initialHash = hashState(replay);
for (const pass of passes.passes) for (const operation of pass.delta) apply(replay, operation);
for (const pass of [...passes.passes].reverse()) for (const operation of [...pass.delta].reverse()) apply(replay, operation, true);
assert(hashState(replay) === initialHash, 'forward/reverse Pass replay is not reversible');

const pto = fs.readFileSync(path.join(ROOT, isa.source.pto), 'utf8');
const cpp = fs.readFileSync(path.join(ROOT, isa.source.cpp), 'utf8');
for (const instruction of isa.instructions) {
  const escaped = instruction.mlirName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const count = (pto.match(new RegExp(`\\b${escaped}(?![\\w.])`, 'g')) || []).length;
  assert(count === instruction.occurrenceCount, `${instruction.mlirName}: expected ${instruction.occurrenceCount}, found ${count}`);
  if (instruction.apiName) assert(cpp.includes(`${instruction.apiName}(`), `${instruction.apiName} not verified in C++`);
}

console.log(`validated ${graph.nodes.length} nodes, ${graph.edges.length} edges, ${passes.passes.length} Passes`);
console.log(`qk_pv: ${passes.summary.sourceChangedForQkPvFamily} family-source changes / ${passes.summary.changedForProgram} program changes / ${passes.summary.noOpForProgram} no-op`);
console.log(`replay hash: ${initialHash.slice(0, 12)}`);
