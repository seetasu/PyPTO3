(function registerPtoModelArchitectureFrontView(global) {
  'use strict';

  const SCHEMA_VERSION = 'model_architecture_front_view.v1';
  const DEFAULTS = Object.freeze({
    width: 1600,
    top: 36,
    outerGap: 22,
    rowGap: 18,
    clusterTitle: 36,
    repeatClusterHeader: 112,
    clusterPadding: 20,
    clusterWidth: 1240,
    nodeWidth: 300,
    moduleNodeWidth: 336,
    tensorWidth: 210,
    nodeHeight: 42,
    laneInset: 180,
    expertCell: 14,
    expertGap: 3,
    expertColumns: 16,
  });
  const FRONT_VIEW_COLORS = Object.freeze({
    dark: Object.freeze({
      'front:embedding': '#14B8A6',
      'front:attention': '#3B82F6',
      'front:norm': '#38BDF8',
      'front:ffn': '#8B5CF6',
      'front:moe': '#F59E0B',
      'front:expert': '#EA580C',
      'front:residual': '#5273FF',
      'front:head': '#4F46E5',
    }),
    light: Object.freeze({
      'front:embedding': '#99F6E4',
      'front:attention': '#93C5FD',
      'front:norm': '#BAE6FD',
      'front:ffn': '#C4B5FD',
      'front:moe': '#FDE68A',
      'front:expert': '#FDBA74',
      'front:residual': '#C7D2FE',
      'front:head': '#A5B4FC',
    }),
  });

  const asElement = (value) => (typeof value === 'string' ? document.querySelector(value) : value);
  const invariant = (condition, message) => {
    if (!condition) throw new Error(`[model-architecture-front-view] ${message}`);
  };

  function normalizedItem(value) {
    return typeof value === 'string' ? { id: value } : { ...(value || {}) };
  }

  function collectStructureIds(item, output = new Set()) {
    if (!item) return output;
    if (item.type === 'row') {
      (item.items || []).forEach((entry) => output.add(normalizedItem(entry).id));
      return output;
    }
    if (item.type === 'cluster') {
      output.add(item.id);
      (item.children || []).forEach((child) => collectStructureIds(child, output));
      return output;
    }
    if (item.type === 'expert-bank') {
      output.add(item.id);
      (item.expertIds || []).forEach((id) => output.add(id));
      (item.aside || []).forEach((entry) => output.add(normalizedItem(entry).id));
    }
    return output;
  }

  function validate(data) {
    invariant(data?.schemaVersion === SCHEMA_VERSION, `expected ${SCHEMA_VERSION}`);
    invariant(data.nodes && typeof data.nodes === 'object', 'nodes map is required');
    invariant(Array.isArray(data.structure), 'structure array is required');
    const referenced = new Set();
    data.structure.forEach((item) => collectStructureIds(item, referenced));
    referenced.forEach((id) => {
      if ((data.structure || []).some((item) => item.id === id)) return;
      invariant(data.nodes[id] || id === 'routed_expert_bank', `missing node definition: ${id}`);
    });
    (data.edges || []).forEach((edge) => {
      invariant(edge.id && edge.source && edge.target, 'every edge requires id/source/target');
    });
    return data;
  }

  function nodeSize(definition, override, config) {
    const kind = definition.kind || 'op';
    const tensorKinds = new Set(['tensor', 'input', 'output', 'constant', 'parameter']);
    const defaultWidth = tensorKinds.has(kind)
      ? config.tensorWidth
      : kind === 'module' ? config.moduleNodeWidth : config.nodeWidth;
    const width = override.width || definition.width || defaultWidth;
    return {
      width,
      height: override.height || definition.height || config.nodeHeight,
    };
  }

  function createLayout(data, collapsedIds) {
    const config = { ...DEFAULTS, ...(data.layout || {}) };
    const nodes = [];
    const clusters = [];
    const visibleIds = new Set();
    const aliasById = new Map();
    const nodeParent = new Map();

    function clusterTitleHeight(item) {
      return item.repeatCount ? config.repeatClusterHeader : config.clusterTitle;
    }

    function itemHeight(item) {
      if (item.type === 'spacer') return item.height || config.outerGap;
      if (item.type === 'row') {
        return Math.max(config.nodeHeight, ...(item.items || []).map((entry) => {
          const spec = normalizedItem(entry);
          return nodeSize(data.nodes[spec.id] || {}, spec, config).height;
        }));
      }
      if (item.type === 'cluster') {
        if (collapsedIds.has(item.id)) return config.nodeHeight;
        const children = item.children || [];
        const content = children.reduce((sum, child) => sum + itemHeight(child), 0);
        const gaps = Math.max(0, children.length - 1) * (item.gap || config.rowGap);
        return clusterTitleHeight(item) + config.clusterPadding + content + gaps;
      }
      if (item.type === 'expert-bank') {
        if (collapsedIds.has(item.id)) return config.nodeHeight;
        const columns = item.columns || config.expertColumns;
        const rows = Math.ceil((item.expertIds || []).length / columns);
        return config.clusterTitle + 44 + rows * config.expertCell + Math.max(0, rows - 1) * config.expertGap + config.clusterPadding;
      }
      return config.nodeHeight;
    }

    function markAlias(item, alias) {
      collectStructureIds(item).forEach((id) => aliasById.set(id, alias));
    }

    function pushNode(id, x, y, override = {}, parent = null) {
      const definition = data.nodes[id] || { id, label: id, kind: 'module' };
      const size = nodeSize(definition, override, config);
      const node = {
        ...definition,
        ...override,
        id,
        x,
        y,
        width: size.width,
        height: size.height,
        parent,
        typeLabel: override.typeLabel || definition.typeLabel || definition.kind || 'Op',
        colorKey: override.colorKey || definition.colorKey || `sem:${definition.kind || 'op'}`,
      };
      nodes.push(node);
      visibleIds.add(id);
      nodeParent.set(id, parent);
      return node;
    }

    function rowPositions(entries, bounds) {
      const result = new Map();
      const lanes = { left: [], center: [], right: [] };
      entries.forEach((entry) => lanes[entry.lane || 'center'].push(entry));
      const laneX = {
        left: bounds.left + Math.min(config.laneInset, bounds.width * 0.19),
        center: bounds.center,
        right: bounds.right - Math.min(config.laneInset, bounds.width * 0.19),
      };
      Object.entries(lanes).forEach(([lane, laneEntries]) => {
        const gap = config.rowGap + 8;
        const widths = laneEntries.map((entry) => nodeSize(data.nodes[entry.id] || {}, entry, config).width);
        const totalWidth = widths.reduce((sum, width) => sum + width, 0) + Math.max(0, widths.length - 1) * gap;
        let cursor = laneX[lane] - totalWidth / 2;
        laneEntries.forEach((entry, index) => {
          result.set(entry.id, cursor + widths[index] / 2);
          cursor += widths[index] + gap;
        });
      });
      return result;
    }

    function place(item, top, bounds, parent = null) {
      const height = itemHeight(item);
      if (item.type === 'spacer') return top + height;

      if (item.type === 'row') {
        const entries = (item.items || []).map(normalizedItem);
        const positions = rowPositions(entries, bounds);
        entries.forEach((entry) => {
          const definition = data.nodes[entry.id] || {};
          const size = nodeSize(definition, entry, config);
          pushNode(entry.id, positions.get(entry.id), top + height / 2, entry, parent);
          if (entry.collapsed || definition.collapsed) {
            nodes[nodes.length - 1].collapsed = true;
          }
          if (size.height < height) nodes[nodes.length - 1].y = top + height / 2;
        });
        return top + height;
      }

      if (item.type === 'cluster') {
        if (collapsedIds.has(item.id)) {
          pushNode(item.id, bounds.center, top + height / 2, {
            label: item.label || data.nodes[item.id]?.label || item.id,
            kind: 'module',
            typeLabel: item.typeLabel || 'Module',
            colorKey: item.colorKey || data.nodes[item.id]?.colorKey || 'module:decoder',
            collapsed: true,
          }, parent);
          markAlias(item, item.id);
          return top + height;
        }
        const width = Math.min(item.width || config.clusterWidth, bounds.width);
        const left = bounds.center - width / 2;
        const cluster = {
          ...(data.nodes[item.id] || {}),
          id: item.id,
          label: item.label || data.nodes[item.id]?.label || item.id,
          x: left,
          y: top,
          width,
          height,
          parent,
          colorKey: item.colorKey || data.nodes[item.id]?.colorKey || 'module:decoder',
          collapsible: item.collapsible !== false,
          repeat: Boolean(item.repeatCount),
          repeatCount: item.repeatCount,
          selectedInstanceIndex: item.selectedInstanceIndex,
          instanceIndices: item.repeatCount ? Array.from({ length: item.repeatCount }, (_, index) => index) : undefined,
        };
        clusters.push(cluster);
        visibleIds.add(item.id);
        let cursor = top + clusterTitleHeight(item);
        const childBounds = {
          left: left + config.clusterPadding,
          right: left + width - config.clusterPadding,
          center: bounds.center,
          width: width - config.clusterPadding * 2,
        };
        (item.children || []).forEach((child, index) => {
          cursor = place(child, cursor, childBounds, item.id);
          if (index < item.children.length - 1) cursor += item.gap || config.rowGap;
        });
        return top + height;
      }

      if (item.type === 'expert-bank') {
        if (collapsedIds.has(item.id)) {
          pushNode(item.id, bounds.center, top + height / 2, {
            label: item.label || data.nodes[item.id]?.label || item.id,
            kind: 'module',
            typeLabel: `${(item.expertIds || []).length} logical experts`,
            colorKey: item.colorKey || 'sem:expert',
            collapsed: true,
          }, parent);
          markAlias(item, item.id);
          return top + height;
        }
        const aside = (item.aside || []).map(normalizedItem);
        const asideWidths = aside.map((entry) => nodeSize(data.nodes[entry.id] || {}, entry, config).width);
        const asideWidth = asideWidths.length ? Math.max(...asideWidths) : 0;
        const asideGap = aside.length ? config.rowGap + 8 : 0;
        const asideReserve = asideWidth + asideGap;
        const width = Math.min(item.width || bounds.width - asideReserve, bounds.width - asideReserve);
        const groupWidth = width + asideReserve;
        const left = bounds.center - groupWidth / 2;
        clusters.push({
          id: item.id,
          label: item.label || data.nodes[item.id]?.label || item.id,
          x: left,
          y: top,
          width,
          height,
          parent,
          colorKey: item.colorKey || 'front:moe',
          collapsible: true,
          repeat: false,
          expertCount: (item.expertIds || []).length,
        });
        visibleIds.add(item.id);
        const anchorId = `${item.id}/__anchor`;
        pushNode(anchorId, left + width / 2, top + config.clusterTitle + 10, {
          label: `${item.label || 'Routed Experts'} · ${(item.expertIds || []).length}`,
          kind: 'module',
          typeLabel: 'Parent module',
          colorKey: item.colorKey || 'front:moe',
          width: Math.min(config.nodeWidth, width - 44),
          height: 34,
          hideTypeLabel: true,
          overlayKind: 'expert-bank-anchor',
        }, item.id);
        aliasById.set(item.id, anchorId);
        const columns = item.columns || config.expertColumns;
        const gridWidth = columns * config.expertCell + Math.max(0, columns - 1) * config.expertGap;
        const gridLeft = left + (width - gridWidth) / 2;
        const gridTop = top + config.clusterTitle + 42;
        (item.expertIds || []).forEach((id, index) => {
          const column = index % columns;
          const row = Math.floor(index / columns);
          pushNode(id,
            gridLeft + column * (config.expertCell + config.expertGap) + config.expertCell / 2,
            gridTop + row * (config.expertCell + config.expertGap) + config.expertCell / 2,
            {
              width: config.expertCell,
              height: config.expertCell,
              hideTypeLabel: true,
              overlayKind: 'expert-cell',
              colorKey: data.nodes[id]?.colorKey || 'front:expert',
            }, item.id);
        });
        aside.forEach((entry, index) => {
          const width = asideWidths[index];
          pushNode(entry.id, left + groupWidth - width / 2, top + height / 2 + index * 58, entry, parent);
        });
        return top + height;
      }

      return top + height;
    }

    const bounds = { left: 36, right: config.width - 36, center: config.width / 2, width: config.width - 72 };
    let cursor = config.top;
    data.structure.forEach((item, index) => {
      cursor = place(item, cursor, bounds, null);
      if (index < data.structure.length - 1) cursor += config.outerGap;
    });

    function endpoint(id) {
      if (visibleIds.has(id) && nodes.some((node) => node.id === id)) return id;
      return aliasById.get(id) || null;
    }

    const edgePairs = new Set();
    const edges = [];
    (data.edges || []).forEach((edge) => {
      const source = endpoint(edge.source);
      const target = endpoint(edge.target);
      if (!source || !target || source === target) return;
      const pair = `${source}->${target}`;
      if (edgePairs.has(pair)) return;
      edgePairs.add(pair);
      edges.push({
        ...edge,
        source,
        target,
        edgeType: edge.edgeType || edge.semanticEdgeType,
        edgeTypeLabel: edge.edgeTypeLabel || edge.semanticEdgeType,
        dashed: edge.dashed ?? ['parameter', 'state'].includes(edge.semanticEdgeType),
        sourceAnchor: edge.sourceAnchor || 'bottom',
        targetAnchor: edge.targetAnchor || 'top',
        curve: edge.curve || 'vertical',
      });
    });

    const evidenceMap = {};
    nodes.forEach((node) => {
      evidenceMap[node.id] = data.evidence?.[node.id] || {
        what: node.label,
        description: node.typeLabel || node.kind,
        evidence: node.sourceRef || data.provenance || '',
      };
    });

    return {
      width: config.width,
      height: Math.ceil(cursor + config.top),
      nodes,
      clusters,
      edges,
      bundles: (data.residualBundles || []).map((bundle) => ({
        ...bundle,
        source: endpoint(bundle.source),
        target: endpoint(bundle.target),
      })).filter((bundle) => bundle.source && bundle.target && bundle.source !== bundle.target),
      evidenceMap,
      metadata: data.metadata || {},
    };
  }

  let bundleSequence = 0;

  function frontViewColormap(graph, theme) {
    const resolvedTheme = theme || global.document?.documentElement?.dataset?.theme || 'dark';
    const colormap = global.PtoModelGraphvizPattern.modelArchitectureColormap(graph, { theme: resolvedTheme });
    const palette = FRONT_VIEW_COLORS[resolvedTheme] || FRONT_VIEW_COLORS.dark;
    Object.entries(palette).forEach(([key, color]) => {
      colormap.semanticColors[key] = { raw: color };
    });
    return colormap;
  }

  function roundedBundlePath(start, end, laneX) {
    const direction = start.y <= end.y ? 1 : -1;
    const radius = Math.min(18, Math.max(8, Math.abs(end.y - start.y) * 0.05));
    return [
      `M ${start.x} ${start.y}`,
      `H ${laneX + radius}`,
      `Q ${laneX} ${start.y} ${laneX} ${start.y + radius * direction}`,
      `V ${end.y - radius * direction}`,
      `Q ${laneX} ${end.y} ${laneX + radius} ${end.y}`,
      `H ${end.x}`,
    ].join(' ');
  }

  function renderResidualBundles(svg, graph) {
    if (!svg || !(graph.bundles || []).length) return;
    const nodeById = new Map(graph.nodes.map((node) => [node.id, node]));
    const clusterById = new Map(graph.clusters.map((cluster) => [cluster.id, cluster]));
    const defs = svg.querySelector('defs');
    const markerId = `pto-front-residual-arrow-${bundleSequence += 1}`;
    const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
    marker.setAttribute('id', markerId);
    marker.setAttribute('viewBox', '0 0 10 10');
    marker.setAttribute('refX', '8.5');
    marker.setAttribute('refY', '5');
    marker.setAttribute('markerWidth', '7');
    marker.setAttribute('markerHeight', '7');
    marker.setAttribute('markerUnits', 'userSpaceOnUse');
    marker.setAttribute('orient', 'auto');
    const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    arrow.setAttribute('d', 'M 0 0 L 10 5 L 0 10 z');
    arrow.setAttribute('fill', 'context-stroke');
    marker.appendChild(arrow);
    defs?.appendChild(marker);
    const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    group.setAttribute('class', 'pto-model-front__residual-bundles');
    (graph.bundles || []).forEach((bundle) => {
      const source = nodeById.get(bundle.source);
      const target = nodeById.get(bundle.target);
      if (!source || !target) return;
      const count = Math.max(1, Number(bundle.count) || 1);
      const spacing = Number(bundle.spacing) || 5;
      const side = bundle.side === 'right' ? 'right' : 'left';
      const clearance = Number(bundle.clearance) || 130;
      const container = clusterById.get(bundle.containerId);
      const containerInset = Number(bundle.containerInset) || 72;
      const start = { x: source.x + (side === 'left' ? -source.width / 2 : source.width / 2), y: source.y };
      const end = { x: target.x + (side === 'left' ? -target.width / 2 : target.width / 2), y: target.y };
      const laneX = container
        ? side === 'left' ? container.x + containerInset : container.x + container.width - containerInset
        : side === 'left' ? Math.min(start.x, end.x) - clearance : Math.max(start.x, end.x) + clearance;
      for (let index = 0; index < count; index += 1) {
        const offset = (index - (count - 1) / 2) * spacing;
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('class', 'pto-model-front__residual-line');
        path.setAttribute('d', roundedBundlePath(
          { x: start.x, y: start.y + offset },
          { x: end.x, y: end.y + offset },
          laneX + offset,
        ));
        path.setAttribute('marker-end', `url(#${markerId})`);
        group.appendChild(path);
      }
      if (bundle.label) {
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('class', 'pto-model-front__residual-label');
        label.setAttribute('x', String(laneX + (side === 'left' ? -8 : 8)));
        label.setAttribute('y', String((start.y + end.y) / 2));
        label.setAttribute('text-anchor', side === 'left' ? 'end' : 'start');
        label.textContent = bundle.label;
        group.appendChild(label);
      }
    });
    const firstNode = svg.querySelector('.pto-model-graphviz-node');
    svg.insertBefore(group, firstNode || null);
  }

  function render(rootInput, dataInput, options = {}) {
    const root = asElement(rootInput);
    invariant(root, 'render root not found');
    invariant(global.PtoModelGraphvizPattern?.renderController, 'model-graphviz pattern is required');
    let data = validate(dataInput);
    let collapsedIds = new Set(options.collapsedIds || data.initialCollapsedIds || []);
    let selectedNodeId = options.selectedNodeId || data.selectedNodeId || null;
    let controller = null;
    let destroyed = false;

    function renderScene({ fit = true, preserveTransform = false } = {}) {
      if (destroyed) return null;
      const previousTransform = preserveTransform ? controller?.getTransform?.() : null;
      controller?.destroy?.();
      const graph = createLayout(data, collapsedIds);
      controller = global.PtoModelGraphvizPattern.renderController(root, graph, {
        ariaLabel: options.ariaLabel || `${data.model?.label || 'Model'} architecture front view`,
        className: ['pto-model-architecture-front-view', options.className].filter(Boolean).join(' '),
        colormap: frontViewColormap(graph, options.theme),
        evidenceMap: graph.evidenceMap,
        reportOverlays: false,
        edgeRouting: 'orthogonal',
        fitMode: 'full',
        autoFit: fit,
        initialTransform: previousTransform || undefined,
        activeNodeId: selectedNodeId,
        interaction: { panZoom: true, selectable: true, relatedHighlight: true },
        overlays: { evidence: true, edgeTags: false },
        onToggle(payload) {
          if (payload.collapsed) collapsedIds.delete(payload.nodeId);
          else collapsedIds.add(payload.nodeId);
          renderScene({ fit: false, preserveTransform: true });
          options.onToggle?.({ ...payload, collapsedIds: new Set(collapsedIds) });
        },
        onSelect(payload) {
          selectedNodeId = payload.nodeId;
          options.onSelect?.(payload);
        },
        onRepeatInstanceChange(payload) {
          options.onLayerChange?.({
            nodeId: payload.nodeId,
            layerIndex: payload.instanceIndex,
            source: payload.source,
          });
        },
      });
      renderResidualBundles(controller?.svg, graph);
      return graph;
    }

    const api = {
      fit() { controller?.fit?.(); return api; },
      selectNode(nodeId, selectOptions = {}) {
        selectedNodeId = nodeId || null;
        if (nodeId) controller?.selectNode?.(nodeId, selectOptions);
        else controller?.clearSelection?.();
        return api;
      },
      setCollapsed(nodeId, collapsed) {
        if (collapsed) collapsedIds.add(nodeId);
        else collapsedIds.delete(nodeId);
        renderScene({ fit: false, preserveTransform: true });
        return api;
      },
      setData(nextData, nextOptions = {}) {
        data = validate(nextData);
        collapsedIds = new Set(nextOptions.collapsedIds || data.initialCollapsedIds || []);
        selectedNodeId = nextOptions.selectedNodeId ?? data.selectedNodeId ?? null;
        renderScene({ fit: nextOptions.fit !== false });
        return api;
      },
      getState() {
        return { selectedNodeId, collapsedIds: new Set(collapsedIds), transform: controller?.getTransform?.() };
      },
      destroy() {
        destroyed = true;
        controller?.destroy?.();
        root.innerHTML = '';
      },
    };

    renderScene({ fit: options.autoFit !== false });
    return api;
  }

  const api = Object.freeze({ schemaVersion: SCHEMA_VERSION, validate, createLayout, render });
  global.PtoModelArchitectureFrontView = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
