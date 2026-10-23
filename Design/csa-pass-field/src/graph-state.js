import mergeGraph from '../data/callable/graph.json';
import mergePasses from '../data/callable/passes.json';
import mergeViews from '../data/callable/views.json';
import isa from '../data/callable/isa.json';
import { instructionCollections } from './instruction-collections.js';
import * as qk from './qk-graph-state.js';
import {isQk} from './sample-registry.js';
import {memoryProjection} from './MemoryField.jsx';
const passes=isQk?qk.passes:mergePasses;
const graph=isQk?qk.graph:mergeGraph;
const views=isQk?qk.views:mergeViews;
import {callableFlow,snapshotAt} from './CallableField.jsx';

const nodeRegistry = new Map(graph.nodes.map((item) => [item.id, item]));
const edgeRegistry = new Map(graph.edges.map((item) => [item.id, item]));

function cloneNode(id) {
  return structuredClone(nodeRegistry.get(id));
}

export function buildDomainState(index) {
 if(isQk)return qk.buildDomainState(index);
 const snapshot=snapshotAt(index);
 return {nodes:snapshot.nodes,edges:snapshot.edges,registry:new Map(snapshot.nodes.map(n=>[n.id,n]))};
}

export function neighborsOf(id, edges) {
  if (!id) return new Set();
  const result = new Set([id]);
  for (const edge of edges) {
    if (edge.source === id) result.add(edge.target);
    if (edge.target === id) result.add(edge.source);
  }
  return result;
}

const edgeTypeMap = {
  data: 'data',
  control: 'control',
  taskDependency: 'taskDependency',
  memoryBinding: 'memoryBinding',
  lowering: 'lowering',
  lineage: 'lineage',
};

export function toReactFlow(state,options){if(!isQk)return callableFlow(state,options);const flow=qk.toReactFlow(state,options);return options.currentPass.index>=33?memoryProjection(flow,options.currentPass.index,options.loadData.memoryReplayId||options.selectedId,options.layout,options.positionOverrides,options.hoverId):flow;}

export function groupPassesForRail(expandedRuns) {
  const phaseIndex = new Map(passes.phases.map((phase) => [phase.id, phase]));
  const segments = [];
  for (const pass of passes.passes) {
    const current = segments.at(-1);
    if (!current || current.id !== pass.phase) segments.push({ ...phaseIndex.get(pass.phase), passes: [] });
    segments.at(-1).passes.push(pass);
  }
  return segments.map((phase) => {
    const phasePasses = phase.passes;
    const items = [];
    let index = 0;
    while (index < phasePasses.length) {
      const pass = phasePasses[index];
      if (!pass.noOp) {
        items.push({ type: 'pass', pass });
        index += 1;
        continue;
      }
      const run = [];
      while (index < phasePasses.length && phasePasses[index].noOp) run.push(phasePasses[index++]);
      const key = `${phase.id}:${run[0].index}-${run.at(-1).index}`;
      if (run.length === 1 || expandedRuns.has(key)) run.forEach((item) => items.push({ type: 'pass', pass: item, runKey: key }));
      else items.push({ type: 'noop-run', key, passes: run });
    }
    return { ...phase, segmentKey: `${phase.id}:${phasePasses[0].index}-${phasePasses.at(-1).index}`, passes: phasePasses, items };
  });
}

export { graph, passes, views };
