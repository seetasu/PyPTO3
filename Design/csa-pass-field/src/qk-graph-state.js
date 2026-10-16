import graph from '../data/csa-qk-pv/graph.json';
import passes from '../data/csa-qk-pv/passes.json';
import views from '../data/csa-qk-pv/views.json';
import isa from '../data/csa-qk-pv/isa.json';
import { instructionCollections } from './qk-instruction-collections.js';
import { layoutVisibleGraph } from './layout.js';

const nodeRegistry = new Map(graph.nodes.map((item) => [item.id, item]));
const edgeRegistry = new Map(graph.edges.map((item) => [item.id, item]));

function cloneNode(id) {
  return structuredClone(nodeRegistry.get(id));
}

export function buildDomainState(passIndex) {
  const activeNodeIds = new Set(graph.initialState.nodeIds);
  const activeEdgeIds = new Set(graph.initialState.edgeIds);
  const nodeData = new Map(graph.nodes.map((item) => [item.id, structuredClone(item)]));

  for (const pass of passes.passes) {
    if (pass.index > passIndex) break;
    for (const operation of pass.delta) {
      if (operation.op === 'addNode') activeNodeIds.add(operation.id);
      if (operation.op === 'removeNode') activeNodeIds.delete(operation.id);
      if (operation.op === 'addEdge') activeEdgeIds.add(operation.id);
      if (operation.op === 'removeEdge') activeEdgeIds.delete(operation.id);
      if (operation.op === 'updateNode') Object.assign(nodeData.get(operation.id), structuredClone(operation.after));
    }
  }

  return {
    nodes: [...activeNodeIds].map((id) => nodeData.get(id) || cloneNode(id)),
    edges: [...activeEdgeIds].map((id) => structuredClone(edgeRegistry.get(id))),
    registry: nodeData,
  };
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

export function toReactFlow(state, { currentPass, selectedId, hoverId, showIsa, positionOverrides, layout = 'horizontal', loadExpanded = false, loadData = {} }) {
  const projectedNodes = showIsa && currentPass?.index >= graph.projections.isa.availableAfterPass
    ? instructionCollections(isa.instructions)
    : [];
  const projectedEdges = showIsa && currentPass?.index >= graph.projections.isa.availableAfterPass
    ? projectedNodes.map(node => ({ id: `origin.${node.id}`, source: node.owner, target: node.id, semanticKind: node.owner.startsWith('kernel.') ? 'lineage' : 'lowering', sourcePort: node.owner.startsWith('kernel.') ? 'out' : 'memory', targetPort: 'in' }))
    : [];
  const visibleNodes = [...state.nodes, ...projectedNodes].filter((item) => item.renderKind !== 'context' && item.attributes?.present !== false && (showIsa || item.attributes?.detailLevel !== 'isa'));
  const visibleIds = new Set(visibleNodes.map((item) => item.id));
  // Whole-program TLOAD totals cannot be attributed to a single QK OP.
  const visibleEdges = [...state.edges, ...projectedEdges].filter((item) => item.target !== 'isa.load' && visibleIds.has(item.source) && visibleIds.has(item.target));
  const focusId = hoverId || selectedId?.replace(/\.L\d+$/, '') || (loadExpanded && showIsa ? loadExpanded : null);
  const neighborhood = neighborsOf(focusId, visibleEdges);
  const touched = new Set(currentPass?.affectedObjectIds || []);
  const reasons = new Map();
  const addReason = (id, reason) => reasons.set(id, [...(reasons.get(id) || []), reason]);
  for (const item of currentPass?.delta || []) {
    if (item.op === 'addNode') addReason(item.id, '本 Pass 新增此节点');
    if (item.op === 'updateNode') {
      const names = {label:'名称或计算形式',semanticKind:'对象类型',attributes:'对象属性',sourceFacts:'源码结构与属性',sourceRefs:'来源位置'};
      addReason(item.id, `本 Pass 改写：${Object.keys(item.after || {}).map(key => names[key] || key).join('、')}`);
    }
    const targetEdge = edgeRegistry.get(item.id);
    if (targetEdge) {
      const action = item.op === 'removeEdge' ? '移除' : '新增';
      const label = id => state.registry.get(id)?.label || id;
      const relation = `${action}${targetEdge.semanticKind === 'memoryBinding' ? '存储绑定' : '关联连线'}：${label(targetEdge.source)} → ${label(targetEdge.target)}`;
      addReason(targetEdge.source, relation);
      addReason(targetEdge.target, relation);
      touched.add(targetEdge.source);
      touched.add(targetEdge.target);
    }
  }

  const direct = new Set((currentPass?.delta || []).filter(op => ['addNode', 'updateNode', 'addEdge'].includes(op.op)).map(op => op.id));
  const passFocus = !focusId && !currentPass?.noOp && (visibleNodes.some(item => direct.has(item.id)) || visibleEdges.some(item => direct.has(item.id)));
  const context = new Set(touched);
  for (const edge of visibleEdges) if (touched.has(edge.source) || touched.has(edge.target)) { context.add(edge.source); context.add(edge.target); }
  const { positions, sizes } = layoutVisibleGraph(visibleNodes, visibleEdges, views.views.front, { layout, loadExpanded, referenceNodes: graph.nodes, referenceEdges: graph.edges });

  const nodes = visibleNodes.map((item) => {
    const override = positionOverrides[item.id];
    const isContainer = ['scope', 'kernel', 'loop'].includes(item.renderKind);
    const size = sizes.get(item.id);
    return {
      id: item.id,
      type: item.renderKind,
      position: override || positions.get(item.id),
      data: { ...item, ...(item.instances ? { ownerLabel: state.registry.get(item.owner)?.label || item.kernel, columnFirst: projectedNodes.find(node => node.owner === item.owner)?.id === item.id, onToggle: () => loadData.onToggle(item.id), onInstanceSelect: loadData.onInstanceSelect, expanded: loadExpanded === item.id } : {}), selected: selectedId === item.id, touched: touched.has(item.id), involvement: reasons.get(item.id) || [], directChange: direct.has(item.id) },
      style: {
        width: size.width,
        height: size.height,
        zIndex: item.id === loadExpanded ? 100 : item.renderKind === 'scope' ? 0 : ['kernel', 'loop'].includes(item.renderKind) ? 1 : item.renderKind === 'task' ? 4 : 3,
      },
      className: item.id === loadExpanded ? 'is-expanded-collection' : !isContainer && (focusId ? !neighborhood.has(item.id) : passFocus && !context.has(item.id)) ? 'is-dimmed' : passFocus && !isContainer && !touched.has(item.id) ? 'is-context' : '',
      draggable: !isContainer,
      selectable: true,
    };
  });

  const edges = visibleEdges.map((item) => ({
    id: item.id,
    type: edgeTypeMap[item.semanticKind] || 'data',
    source: item.source,
    target: item.target,
    sourceHandle: layout === 'vertical' && ['lowering','lineage'].includes(item.semanticKind) ? 'side-out' : item.sourcePort || (item.target.startsWith('isa.') ? 'memory' : 'out'),
    targetHandle: layout === 'vertical' && ['lowering','lineage'].includes(item.semanticKind) ? 'side-in' : item.targetPort || 'in',
    data: {
      ...item,
      touched: touched.has(item.id),
      focused: Boolean(focusId && (item.source === focusId || item.target === focusId)),
    },
    className: focusId ? (!(neighborhood.has(item.source) && neighborhood.has(item.target)) ? 'is-dimmed' : '') : passFocus && !touched.has(item.id) ? 'is-context' : '',
    zIndex: 2,
  }));
  return { nodes, edges };
}

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
