/* Source-derived, directed dependency journeys. No geometry or UI state.
 * Composite endpoints resolve through declared ports, never through entity-name
 * equality: two views of one entity can describe different slices/generations. */
(function (root) {
  'use strict';
  const FLOW = new Set(['data', 'computation', 'state', 'feedback']);
  function index(model) {
    const nodes = new Map(), rawEdges = [];
    const walk = (graph, prefix = '', stage = null) => {
      const membership = new Map((graph.groups || []).flatMap((group) => group.members.map((id) => [id, { id: prefix + group.id, name: group.name }])));
      for (const node of graph.nodes) {
        const id = prefix + node.id, region = membership.get(node.id) || stage;
        nodes.set(id, { id, node, stage: region });
        if (node.composite) walk(node.composite.subgraph, id + '/', region);
      }
      for (const edge of graph.edges) rawEdges.push({ ...edge, id: prefix + edge.id, s: prefix + edge.s, t: prefix + edge.t });
    };
    walk(model);
    function endpoint(id, port, side) {
      const mappings = [], seen = new Set(); let current = id, name = port || 'default';
      while (nodes.get(current)?.node.composite) {
        if (seen.has(current)) throw new Error('Composite port cycle: ' + current);
        seen.add(current);
        const mapping = nodes.get(current).node.composite.portMappings[side]?.[name];
        if (!mapping) throw new Error(`Missing ${side} mapping: ${current}.${name}`);
        mappings.push({ composite: current, port: name, ...mapping });
        current += '/' + mapping.node; name = mapping.port;
      }
      return { id: current, port: name, mappings };
    }
    const edges = rawEdges.map((edge) => {
      const from = endpoint(edge.s, edge.sourcePort, 'outputs'), to = endpoint(edge.t, edge.targetPort, 'inputs');
      return { ...edge, semanticSource: edge.s, semanticTarget: edge.t,
        s: from.id, t: to.id, sourcePort: from.port, targetPort: to.port,
        mappings: [...from.mappings, ...to.mappings] };
    });
    return { model, nodes, edges, endpoint };
  }
  function trace(model, anchorId, direction = 'both', compiled = index(model)) {
    if (!['both', 'upstream', 'downstream'].includes(direction)) throw new Error('Invalid journey direction');
    const anchor = compiled.nodes.get(anchorId);
    if (!anchor) return null;
    const payload = anchor.node.composite;
    const seeds = payload ? [...new Set(['inputs', 'outputs'].flatMap((side) => Object.keys(payload.portMappings[side]).map((port) => compiled.endpoint(anchorId, port, side).id)))] : [anchorId];
    const nodeIds = new Set(seeds), edgeIds = new Set(), feedbackEdgeIds = new Set();
    // Both = union of two independent walks. Never reverse direction at a merge,
    // which would pull K/V inputs into Q's downstream influence path.
    const walk = (side) => {
      const visited = new Set(seeds), queue = [...seeds];
      const adjacency = new Map();
      for (const edge of compiled.edges) {
        if (!FLOW.has(edge.type)) continue;
        const key = side === 'downstream' ? edge.s : edge.t;
        if (!adjacency.has(key)) adjacency.set(key, []);
        adjacency.get(key).push(edge);
      }
      for (let i = 0; i < queue.length; i++) {
        for (const edge of adjacency.get(queue[i]) || []) {
          const next = side === 'downstream' ? edge.t : edge.s;
          edgeIds.add(edge.id); if (edge.type === 'feedback') feedbackEdgeIds.add(edge.id);
          nodeIds.add(next);
          if (!visited.has(next)) { visited.add(next); queue.push(next); }
        }
      }
    };
    if (direction !== 'upstream') walk('downstream');
    if (direction !== 'downstream') walk('upstream');
    const terminals = [...nodeIds].filter((id) => !compiled.edges.some((edge) => (direction === 'upstream' ? edge.t : edge.s) === id && edgeIds.has(edge.id) && edge.type !== 'feedback'));
    const branches = [...nodeIds].filter((id) => compiled.edges.filter((edge) => edge.s === id && edgeIds.has(edge.id) && edge.type !== 'feedback').length > 1);
    return { anchorId, direction, nodeIds, edgeIds, feedbackEdgeIds, terminals, branches, index: compiled };
  }
  function project(result, visible) {
    const visibleIds = new Set(visible.nodes.map((node) => node.id));
    const representative = (id) => {
      let current = id;
      while (current && !visibleIds.has(current)) current = current.includes('/') ? current.slice(0, current.lastIndexOf('/')) : '';
      return current || null;
    };
    const nodeIds = new Set(), edgeIds = new Set(), otherInputIds = new Set();
    for (const id of result.nodeIds) {
      let current = id;
      while (current) {
        if (visibleIds.has(current)) nodeIds.add(current);
        current = current.includes('/') ? current.slice(0, current.lastIndexOf('/')) : '';
      }
    }
    for (const edge of visible.edges) {
      if ((edge.sourceEdgeIds || [edge.id]).some((id) => result.edgeIds.has(id))) edgeIds.add(edge.id);
      else if (FLOW.has(edge.type) && nodeIds.has(edge.t) && !nodeIds.has(edge.s)) otherInputIds.add(edge.s);
    }
    // Other operand branches stay separate from the anchor's influence path.
    // Resolve them canonically as well, so expanding QK still identifies K Cache.
    const other = new Set(), queue = [];
    for (const edge of result.index.edges) {
      if (FLOW.has(edge.type) && result.nodeIds.has(edge.t) && !result.nodeIds.has(edge.s)) { other.add(edge.s); queue.push(edge.s); }
    }
    for (let i = 0; i < queue.length; i++) {
      for (const edge of result.index.edges) {
        if (!FLOW.has(edge.type) || edge.t !== queue[i] || result.nodeIds.has(edge.s) || other.has(edge.s)) continue;
        other.add(edge.s); queue.push(edge.s);
      }
    }
    for (const id of other) { const visibleId = representative(id); if (visibleId && !nodeIds.has(visibleId)) otherInputIds.add(visibleId); }
    const anchorVisibleId = representative(result.anchorId);
    const rows = visible.nodes.filter((node) => nodeIds.has(node.id)).map((node) => {
      const meta = result.index.nodes.get(node.id);
      const outgoing = visible.edges.filter((edge) => edge.s === node.id && edgeIds.has(edge.id));
      return { id: node.id, name: node.title, type: node.type, stage: meta?.stage?.name || '',
        shape: node.overlay?.shape || null, branch: outgoing.filter((edge) => edge.type !== 'feedback').length > 1,
        feedback: outgoing.some((edge) => edge.type === 'feedback'),
        terminal: result.terminals.some((id) => representative(id) === node.id) };
    });
    return { nodeIds, edgeIds, otherInputIds, rows, anchorVisibleId, anchorCollapsed: anchorVisibleId !== result.anchorId };
  }
  /* Minimap reads the exact layout and the controller transform. */
  function minimap(graph, view, viewport, width = 150, height = 150) {
    const scale = Math.min(width / graph.width, height / graph.height);
    const offsetX = (width - graph.width * scale) / 2, offsetY = (height - graph.height * scale) / 2;
    const point = (x, y) => ({ x: offsetX + x * scale, y: offsetY + y * scale });
    const start = point(-view.tx / view.scale, -view.ty / view.scale);
    return { width, height, scale, offsetX, offsetY,
      nodes: graph.nodes.map((node) => ({ id: node.id, ...point(node.x, node.y), w: node.w * scale, h: (node.groupHeight || node.h) * scale })),
      groups: graph.groups.map((group) => ({ id: group.id, ...point(group.x, group.y), w: group.w * scale, h: group.h * scale })),
      viewport: { x: start.x, y: start.y, w: viewport.w / view.scale * scale, h: viewport.h / view.scale * scale } };
  }
  const worldPoint = (map, x, y) => ({ x: (x - map.offsetX) / map.scale, y: (y - map.offsetY) / map.scale });
  root.PTO_COMPUTATION_JOURNEY = { index, trace, project, minimap, worldPoint };
})(typeof window === 'undefined' ? globalThis : window);
