/* View-only contraction of DATA cards. Original semantic objects and positions survive. */
(function (root) {
  'use strict';
  function withoutData(graph, options) {
    const hidden = new Set(graph.nodes.filter((node) => node.type === 'data').map((node) => node.id));
    const nodes = graph.nodes.filter((node) => !hidden.has(node.id));
    const outgoing = new Map(graph.nodes.map((node) => [node.id, []]));
    graph.edges.forEach((edge) => outgoing.get(edge.s).push(edge));
    const edges = [];
    function follow(chain, seen) {
      const first = chain[0], last = chain.at(-1);
      if (!hidden.has(last.t)) {
        if (chain.length === 1) { edges.push(first); return; }
        const sourceEdgeIds = chain.flatMap((edge) => edge.sourceEdgeIds || [edge.id]);
        const type = chain.every((edge) => edge.type === 'data') ? 'data'
          : chain.find((edge) => edge.type !== 'data').type;
        edges.push({ ...first, id: 'data-hidden:' + JSON.stringify(sourceEdgeIds),
          t: last.t, targetPort: last.targetPort, semanticTarget: last.semanticTarget,
          type, route: type, sourceEdgeIds,
          label: [...new Set(chain.map((edge) => edge.label).filter(Boolean))].join(' → ') });
        return;
      }
      if (seen.has(last.t)) return;
      const nextSeen = new Set(seen).add(last.t);
      for (const next of outgoing.get(last.t)) follow([...chain, next], nextSeen);
    }
    for (const edge of graph.edges) if (!hidden.has(edge.s)) follow([edge], new Set([edge.s]));
    return { ...graph, nodes, edges,
      edgeRoutes: root.PTO_COMPUTATION_LAYOUT.routeEdges(nodes, edges, graph.width - 88, options) };
  }
  root.PtoComputationVisibility = { withoutData };
})(typeof window === 'undefined' ? globalThis : window);
