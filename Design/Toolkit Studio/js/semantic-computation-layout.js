/* Deterministic top-to-bottom layout for typed hierarchical computation graphs.
   It reads semantic nodes, edges and regions; it never adds or deletes them. */
(function (root) {
  'use strict';

  const BASE = { rootX: 72, rootTop: 76, childX: 34, childTop: 28,
    columnGap: 72, layerGap: 112, childColumnGap: 60, childLayerGap: 82,
    regionSide: 28, regionTop: 55, regionBottom: 28,
    compositeSide: 22, compositeHeader: 62, compositeBottom: 24 };
  const fullId = (prefix, local) => prefix ? `${prefix}/${local}` : local;
  const forward = (edge) => edge.type !== 'feedback';

  function nodeSize(node, child) {
    const fallbackWidth = child ? 194 : ['data', 'state'].includes(node.type) ? 170 : 188;
    return { w: node.w || fallbackWidth, h: node.h || (child ? 80 : 88) };
  }

  function rankNodes(nodes, edges, scope) {
    const byId = new Map(nodes.map((item) => [item.node.id, item]));
    const incoming = new Map(nodes.map((item) => [item.node.id, []]));
    const outgoing = new Map(nodes.map((item) => [item.node.id, []]));
    edges.filter(forward).forEach((edge) => {
      incoming.get(edge.t).push(edge.s);
      outgoing.get(edge.s).push(edge.t);
    });
    const degree = new Map(nodes.map((item) => [item.node.id, incoming.get(item.node.id).length]));
    const ready = nodes.filter((item) => !degree.get(item.node.id)).map((item) => item.node.id);
    let visited = 0;
    while (ready.length) {
      const id = ready.shift();
      const item = byId.get(id);
      item.rank = Math.max(0, ...incoming.get(id).map((source) => byId.get(source).rank + 1));
      visited += 1;
      outgoing.get(id).forEach((target) => {
        degree.set(target, degree.get(target) - 1);
        if (!degree.get(target)) { ready.push(target); ready.sort((a, b) => byId.get(a).index - byId.get(b).index); }
      });
    }
    if (visited !== nodes.length) throw new Error(`${scope}: forward graph has a cycle; mark feedback edges explicitly`);
    return { incoming, outgoing };
  }

  function layoutScope(graph, prefix, expanded, childScope) {
    const nodes = graph.nodes.map((node, index) => {
      const id = fullId(prefix, node.id);
      const inner = node.composite?.subgraph && expanded.has(id)
        ? layoutScope(node.composite.subgraph, id, expanded, true) : null;
      const base = nodeSize(node, childScope);
      return { node, id, inner, index,
        boxW: inner ? Math.max(base.w, inner.width + BASE.compositeSide * 2) : base.w,
        boxH: inner ? BASE.compositeHeader + inner.height + BASE.compositeBottom : base.h };
    });
    const byId = new Map(nodes.map((item) => [item.node.id, item]));
    if (!nodes.length) return { graph, prefix, nodes, regions: [], width: 120, height: 90 };
    const { incoming, outgoing } = rankNodes(nodes, graph.edges, prefix || graph.id || 'graph');
    const membership = new Map();
    (graph.groups || []).forEach((group, index) => group.members.forEach((id) => membership.set(id, index)));
    const regionKey = (item) => membership.get(item.node.id) ?? (graph.groups || []).length;
    const ranks = new Map();
    nodes.forEach((item) => {
      if (!ranks.has(item.rank)) ranks.set(item.rank, []);
      ranks.get(item.rank).push(item);
    });
    const levels = [...ranks.keys()].sort((a, b) => a - b);
    const order = new Map();
    levels.forEach((rank) => {
      const row = ranks.get(rank);
      row.sort((a, b) => {
        const region = regionKey(a) - regionKey(b);
        if (region) return region;
        const barycenter = (item) => {
          const predecessors = incoming.get(item.node.id).filter((id) => order.has(id));
          return predecessors.length ? predecessors.reduce((sum, id) => sum + order.get(id), 0) / predecessors.length : Infinity;
        };
        return barycenter(a) - barycenter(b) || a.index - b.index;
      });
      row.forEach((item, index) => order.set(item.node.id, index));
    });
    // A reverse sweep reduces crossings where one producer fans out to several rows.
    [...levels].reverse().forEach((rank) => {
      const row = ranks.get(rank);
      row.sort((a, b) => {
        const region = regionKey(a) - regionKey(b);
        if (region) return region;
        const avg = (item) => {
          const successors = outgoing.get(item.node.id).filter((id) => order.has(id));
          return successors.length ? successors.reduce((sum, id) => sum + order.get(id), 0) / successors.length : Infinity;
        };
        return avg(a) - avg(b) || order.get(a.node.id) - order.get(b.node.id);
      });
      row.forEach((item, index) => order.set(item.node.id, index));
    });

    const side = childScope ? BASE.childX : BASE.rootX;
    const top = childScope ? BASE.childTop : BASE.rootTop;
    const columnGap = childScope ? BASE.childColumnGap : BASE.columnGap;
    const layerGap = childScope ? BASE.childLayerGap : BASE.layerGap;
    const rowWidth = (row) => row.reduce((sum, item) => sum + item.boxW, 0) + Math.max(0, row.length - 1) * columnGap;
    const widest = Math.max(0, ...levels.map((rank) => rowWidth(ranks.get(rank))));
    let width = widest + side * 2;
    let cursorY = top;
    levels.forEach((rank) => {
      const row = ranks.get(rank);
      let cursorX = side + (widest - rowWidth(row)) / 2;
      row.forEach((item) => { item.x = cursorX; item.y = cursorY; cursorX += item.boxW + columnGap; });
      cursorY += Math.max(...row.map((item) => item.boxH)) + layerGap;
    });
    const regions = (graph.groups || []).map((group) => {
      const members = group.members.map((id) => byId.get(id));
      return { ...group,
        label: group.name,
        id: fullId(prefix, group.id),
        x: Math.min(...members.map((item) => item.x)) - BASE.regionSide,
        y: Math.min(...members.map((item) => item.y)) - BASE.regionTop,
        w: Math.max(...members.map((item) => item.x + item.boxW)) - Math.min(...members.map((item) => item.x)) + BASE.regionSide * 2,
        h: Math.max(...members.map((item) => item.y + item.boxH)) - Math.min(...members.map((item) => item.y)) + BASE.regionTop + BASE.regionBottom,
      };
    });
    const left = Math.min(0, ...regions.map((group) => group.x));
    if (left < 0) {
      const shift = -left + 16;
      nodes.forEach((item) => { item.x += shift; });
      regions.forEach((group) => { group.x += shift; });
      width += shift;
    }
    const bottom = Math.max(cursorY - layerGap + (childScope ? 34 : 56), ...regions.map((group) => group.y + group.h + 28));
    width = Math.max(width, ...regions.map((group) => group.x + group.w + 28));
    return { graph, prefix, nodes, regions, width, height: bottom };
  }

  function layout(model, expandedIds = new Set()) {
    const expanded = new Set(expandedIds);
    const hierarchy = layoutScope(model, '', expanded, false);
    const nodes = [], groups = [], rawEdges = [], opened = new Set();
    const visit = (scope, offsetX, offsetY) => {
      scope.regions.forEach((group) => groups.push({ ...group, x: offsetX + group.x, y: offsetY + group.y }));
      scope.nodes.forEach((item) => {
        const node = item.node;
        const id = item.id;
        // 投影需要 model 才能解析常量与运行时维度；几何仍只由语义节点决定。
        nodes.push({ ...root.PTO_COMPUTATION_PRESENTATION.node(node, { id, parentId: scope.prefix || undefined, model }),
          x: offsetX + item.x, y: offsetY + item.y, w: item.boxW,
          h: item.inner ? 42 : item.boxH,
          childOperation: !!scope.prefix, expandable: !!node.composite?.subgraph,
          expandedGroup: !!item.inner, groupWidth: item.inner ? item.boxW : undefined,
          groupHeight: item.inner ? item.boxH : undefined,
        });
        if (item.inner) {
          opened.add(id);
          visit(item.inner, offsetX + item.x + BASE.compositeSide, offsetY + item.y + BASE.compositeHeader);
        }
      });
      scope.graph.edges.forEach((edge) => rawEdges.push(root.PTO_COMPUTATION_PRESENTATION.edge(edge, {
        id: fullId(scope.prefix, edge.id), sourceId: fullId(scope.prefix, edge.s), targetId: fullId(scope.prefix, edge.t),
        internal: !!scope.prefix })));
    };
    visit(hierarchy, 0, 0);
    const byId = new Map(nodes.map((node) => [node.id, node]));
    const endpoint = (id, port, side) => {
      let current = byId.get(id), name = port || 'default';
      while (current?.expandedGroup) {
        const mapping = current.composite.portMappings[side]?.[name];
        if (!mapping) throw new Error(`${current.id}: missing ${side} port ${name}`);
        current = byId.get(`${current.id}/${mapping.node}`);
        name = mapping.port;
      }
      return { id: current?.id, port: name };
    };
    const edges = rawEdges.map((edge) => {
      const source = endpoint(edge.s, edge.sourcePort, 'outputs'), target = endpoint(edge.t, edge.targetPort, 'inputs');
      return { ...edge, semanticSource: edge.s, semanticTarget: edge.t,
        s: source.id, t: target.id, sourcePort: source.port, targetPort: target.port };
    });
    const right = Math.max(hierarchy.width, ...nodes.map((node) => node.x + node.w), ...groups.map((group) => group.x + group.w));
    const bottom = Math.max(hierarchy.height, ...nodes.map((node) => node.y + (node.groupHeight || node.h)), ...groups.map((group) => group.y + group.h));
    const geometry = routeEdges(nodes, edges, right);
    return { ...model, semantic: model, nodes, edges, groups, opened, edgeRoutes: geometry,
      width: Math.max(right + 88, ...geometry.map((route) => route.maxX + 32)), height: bottom + 100 };
  }

  function routeEdges(nodes, edges, rightBound) {
    const byId = new Map(nodes.map((node) => [node.id, node]));
    const incoming = new Map(nodes.map((node) => [node.id, []]));
    const outgoing = new Map(nodes.map((node) => [node.id, []]));
    edges.forEach((edge, index) => { incoming.get(edge.t).push(index); outgoing.get(edge.s).push(index); });
    incoming.forEach((list) => list.sort((a, b) => byId.get(edges[a].s).x - byId.get(edges[b].s).x || a - b));
    outgoing.forEach((list) => list.sort((a, b) => byId.get(edges[a].t).x - byId.get(edges[b].t).x || a - b));
    const port = (node, list, index) => node.x + node.w * (list.indexOf(index) + 1) / (list.length + 1);
    let sideLane = 0;
    const routes = edges.map((edge, index) => {
      const source = byId.get(edge.s), target = byId.get(edge.t);
      const sx = port(source, outgoing.get(edge.s), index), tx = port(target, incoming.get(edge.t), index);
      const sy = source.y + source.h, ty = target.y;
      if (edge.route === 'feedback') {
        const frames = nodes.filter((node) => node.expandedGroup &&
          source.x >= node.x && target.x >= node.x &&
          source.x + source.w <= node.x + node.groupWidth && target.x + target.w <= node.x + node.groupWidth &&
          source.y >= node.y && target.y >= node.y &&
          source.y + source.h <= node.y + node.groupHeight && target.y + target.h <= node.y + node.groupHeight);
        frames.sort((a, b) => a.groupWidth * a.groupHeight - b.groupWidth * b.groupHeight);
        const left = frames.length && target.x + target.w / 2 < frames[0].x + frames[0].groupWidth / 2;
        const lane = frames.length ? left ? frames[0].x + 10 : frames[0].x + frames[0].groupWidth - 10 : rightBound + 42;
        const x1 = left ? source.x : source.x + source.w;
        const x2 = left ? target.x : target.x + target.w;
        const y1 = source.y + source.h / 2, y2 = target.y + target.h / 2;
        return { path: `M ${x1} ${y1} H ${lane} V ${y2} H ${x2}`,
          labelX: lane + (left ? -8 : 8), labelY: (y1 + y2) / 2,
          labelAxis: 'y', labelMin: Math.min(y1, y2), labelMax: Math.max(y1, y2), maxX: Math.max(lane, x1, x2) };
      }
      if (ty - sy > 190) {
        const leftBound = Math.min(...nodes.map((node) => node.x));
        const useLeft = (source.x + target.x) / 2 < rightBound / 2;
        const lane = useLeft ? Math.max(22, leftBound - 42 - sideLane * 14) : rightBound + 42 + sideLane * 14;
        sideLane = (sideLane + 1) % 4;
        return { path: `M ${sx} ${sy} V ${sy + 28} H ${lane} V ${ty - 28} H ${tx} V ${ty}`,
          labelX: lane + (useLeft ? -8 : 8), labelY: (sy + ty) / 2,
          labelAxis: 'y', labelMin: sy + 28, labelMax: ty - 28, maxX: Math.max(lane, sx, tx) };
      }
      const middle = (sy + ty) / 2;
      return { path: `M ${sx} ${sy} V ${middle} H ${tx} V ${ty}`,
        labelX: Math.abs(sx - tx) > 38 ? (sx + tx) / 2 : tx + 12,
        labelY: middle - 7, labelAxis: Math.abs(sx - tx) > 38 ? 'x' : 'y',
        labelMin: Math.abs(sx - tx) > 38 ? Math.min(sx, tx) : sy,
        labelMax: Math.abs(sx - tx) > 38 ? Math.max(sx, tx) : ty,
        maxX: Math.max(sx, tx) };
    });
    const placed = [];
    routes.forEach((route, index) => {
      if (!edges[index].label) return;
      const width = edges[index].label.length * 7.2 + 8;
      const candidates = [0, 22, -22, 44, -44, 66, -66];
      for (const shift of candidates) {
        const x = route.labelX + (route.labelAxis === 'x' ? shift : 0);
        const y = route.labelY + (route.labelAxis === 'y' ? shift : 0);
        if ((route.labelAxis === 'x' ? x : y) < route.labelMin || (route.labelAxis === 'x' ? x : y) > route.labelMax) continue;
        const box = { x: x - width / 2, y: y - 14, w: width, h: 19 };
        const intersects = (other) => box.x < other.x + other.w && box.x + box.w > other.x && box.y < other.y + other.h && box.y + box.h > other.y;
        if (placed.some(intersects) || nodes.filter((node) => !node.expandedGroup).some(intersects)) continue;
        route.labelX = x; route.labelY = y; placed.push(box); return;
      }
      placed.push({ x: route.labelX - width / 2, y: route.labelY - 14, w: width, h: 19 });
    });
    return routes;
  }

  // Project static synchronization metadata after layout; never rank it as data flow.
  function synchronizationBridges(graph) {
    const byId = new Map(graph.nodes.map((node) => [node.id, node]));
    const right = Math.max(...graph.nodes.map((node) => node.x + (node.groupWidth || node.w)));
    const resolve = (end) => byId.get(end.anchor) || byId.get(end.boundary || end.node);
    return (graph.synchronizations || []).map((event) => {
      const producer = resolve(event.producer), consumer = resolve(event.consumer);
      if (!producer || !consumer) throw new Error(`${event.name}: unresolved synchronization boundary`);
      const sx = producer.x + producer.w / 2, sy = producer.y + producer.h;
      const tx = consumer.x + consumer.w / 2, ty = consumer.y;
      const lane = right + 24 + event.eventId * 12;
      return { event, producer: producer.id, consumer: consumer.id,
        x: producer.x + producer.w + 24, y: sy + 22, w: 254, h: 48,
        path: `M ${sx} ${sy} V ${sy + 12} H ${lane} V ${ty - 20} H ${tx} V ${ty}`,
        maxX: Math.max(lane, producer.x + producer.w + 278) };
    });
  }

  root.PTO_COMPUTATION_LAYOUT = { layout, routeEdges, synchronizationBridges };
})(typeof window === 'undefined' ? globalThis : window);
