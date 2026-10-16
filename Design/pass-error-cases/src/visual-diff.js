// Compare displayed semantics, excluding provenance, selection and layout.
export function visualPassDiff(graph, passes) {
  const nodes = new Map(graph.nodes.map(node => [node.id, structuredClone(node)]));
  const active = new Set(graph.initialState.nodeIds);
  const edges = new Set(graph.initialState.edgeIds);
  function snapshot(index) {
    return JSON.stringify({ nodes: [...active].sort().map(id => nodes.get(id)).filter(n => n.renderKind !== 'context').map(n => ({
      id:n.id, kind:n.semanticKind, label:n.label, attributes:n.attributes,
      facts:n.renderKind === 'loop' || n.renderKind === 'op' && n.attributes?.auxiliary ? n.sourceFacts : n.renderKind === 'tensor' ? {spaces:n.sourceFacts?.spaces} : undefined,
    })), edges:[...edges].sort() });
  }
  let previous = snapshot(0);
  return passes.map(pass => {
    for (const change of pass.delta) {
      if (change.op === 'addNode') active.add(change.id);
      if (change.op === 'removeNode') active.delete(change.id);
      if (change.op === 'updateNode') Object.assign(nodes.get(change.id), structuredClone(change.after));
      if (change.op === 'addEdge') edges.add(change.id);
      if (change.op === 'removeEdge') edges.delete(change.id);
    }
    const current = snapshot(pass.index);
    const changed = current !== previous;
    previous = current;
    return {index:pass.index,changed};
  });
}
