/* Semantic schema and validation. Geometry is computed by semantic-computation-layout.js. */
(function (root) {
  'use strict';
  const NODE_TYPES = new Set(['operation', 'data', 'state', 'composite', 'region']);
  const EDGE_TYPES = new Set(['data', 'computation', 'state', 'control']);

  function validate(model) {
    const visit = (scope, nodes, edges, groups = []) => {
      const byId = new Map();
      nodes.forEach((node) => {
        if (!node.id || byId.has(node.id)) throw new Error(`${scope}: duplicate or missing node id`);
        if (!NODE_TYPES.has(node.semanticType || node.type)) throw new Error(`${scope}/${node.id}: invalid node type`);
        byId.set(node.id, node);
        if (node.subgraph) {
          const child = node.subgraph;
          visit(`${scope}/${node.id}`, child.nodes, child.edges, child.groups);
          for (const side of ['inputs', 'outputs']) {
            Object.values(child[side] || {}).forEach((id) => {
              if (!child.nodes.some((item) => item.id === id)) throw new Error(`${scope}/${node.id}: invalid ${side} port ${id}`);
            });
          }
        }
      });
      edges.forEach((edge) => {
        const source = byId.get(edge.s), target = byId.get(edge.t);
        if (!source || !target) throw new Error(`${scope}: broken edge ${edge.s} → ${edge.t}`);
        if (!EDGE_TYPES.has(edge.type)) throw new Error(`${scope}: invalid edge type`);
        if (source.subgraph && !source.subgraph.outputs?.[edge.sourcePort || 'default']) throw new Error(`${scope}: missing output port on ${edge.s}`);
        if (target.subgraph && !target.subgraph.inputs?.[edge.targetPort || 'default']) throw new Error(`${scope}: missing input port on ${edge.t}`);
      });
      groups.forEach((group) => {
        if (group.semanticType !== 'region' || !group.id || !group.members?.length) throw new Error(`${scope}: invalid region`);
        (group.members || []).forEach((id) => {
          if (!byId.has(id)) throw new Error(`${scope}/${group.id}: unknown member ${id}`);
        });
      });
    };
    visit(model.id || 'graph', model.nodes, model.edges, model.groups);
    return model;
  }

  function materialize(model, expandedIds) {
    validate(model);
    return root.PTO_COMPUTATION_LAYOUT.layout(model, expandedIds);
  }

  root.PTO_SEMANTIC_COMPUTATION_GRAPH = { validate, materialize };
})(typeof window === 'undefined' ? globalThis : window);
