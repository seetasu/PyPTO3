import dagre from 'dagre';
import { tensorGeometry } from './tensor-glyph.js';

const BACKGROUND_KINDS = new Set(['scope', 'kernel', 'loop']);
const CORE_KINDS = new Set(['tensor', 'op', 'handoff']);

function sizeOf(node, view) {
  if (node.renderKind === 'tensor') return tensorGeometry(node);
  if (node.instances) return { width: 310, height: 56 };
  if (node.renderKind === 'op') {
    const label = node.label.replace(/^tensor\.|^tile\./, 'tile.');
    const textWidth = [...label].reduce((width, char) => width + (/[ilI .·]/.test(char) ? 5 : /[MWQO]/.test(char) ? 13 : 9), 0);
    return { width: Math.max(124, Math.ceil(textWidth + 72)), height: 48 };
  }
  if (node.renderKind === 'memory') return { width: 300, height: 106 };
  const fallback = view.nodes?.[node.id] || {};
  return {
    width: fallback.width || 160,
    height: fallback.height || 64,
  };
}

function centerOf(position, size) {
  return {
    x: position.x + size.width / 2,
    y: position.y + size.height / 2,
  };
}

function boundsOf(ids, positions, sizes, padding = {}) {
  const active = ids.filter((id) => positions.has(id));
  if (!active.length) return null;
  const left = Math.min(...active.map((id) => positions.get(id).x));
  const top = Math.min(...active.map((id) => positions.get(id).y));
  const right = Math.max(...active.map((id) => positions.get(id).x + sizes.get(id).width));
  const bottom = Math.max(...active.map((id) => positions.get(id).y + sizes.get(id).height));
  return {
    x: left - (padding.left || 0),
    y: top - (padding.top || 0),
    width: right - left + (padding.left || 0) + (padding.right || 0),
    height: bottom - top + (padding.top || 0) + (padding.bottom || 0),
  };
}

function layoutDagre(nodes, edges, sizes, config) {
  const model = new dagre.graphlib.Graph();
  model.setDefaultEdgeLabel(() => ({}));
  model.setGraph({
    rankdir: config.direction || 'LR',
    ranksep: config.rankSep || 104,
    nodesep: config.nodeSep || 58,
    edgesep: config.edgeSep || 24,
    marginx: 0,
    marginy: 0,
    ranker: 'network-simplex',
  });
  nodes.forEach((node) => model.setNode(node.id, sizes.get(node.id)));
  const nodeIds = new Set(nodes.map((node) => node.id));
  edges
    .filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target))
    .forEach((edge) => model.setEdge(edge.source, edge.target));
  dagre.layout(model);
  return new Map(nodes.map((node) => {
    const point = model.node(node.id);
    const size = sizes.get(node.id);
    return [node.id, { x: point.x - size.width / 2, y: point.y - size.height / 2 }];
  }));
}

function spreadLane(ids, positions, sizes, top, height) {
  const active = ids.filter((id) => positions.has(id));
  if (!active.length) return;
  const sourceCenters = active.map((id) => centerOf(positions.get(id), sizes.get(id)).y);
  const min = Math.min(...sourceCenters);
  const max = Math.max(...sourceCenters);
  const innerTop = top + 58;
  const innerBottom = top + height - 32;
  active.forEach((id) => {
    const size = sizes.get(id);
    const sourceCenter = centerOf(positions.get(id), size).y;
    const ratio = max === min ? 0.5 : (sourceCenter - min) / (max - min);
    const targetCenter = innerTop + ratio * Math.max(0, innerBottom - innerTop);
    positions.set(id, { ...positions.get(id), y: targetCenter - size.height / 2 });
  });
}

function placeAlignedBand(nodes, edges, positions, sizes, y, gap = 28, preserveOrder = false) {
  if (!nodes.length) return;
  const preferred = nodes.map((node) => {
    const sources = edges
      .filter((edge) => edge.target === node.id && positions.has(edge.source))
      .map((edge) => centerOf(positions.get(edge.source), sizes.get(edge.source)).x);
    const width = sizes.get(node.id).width;
    return {
      id: node.id,
      width,
      x: sources.length ? sources.reduce((sum, value) => sum + value, 0) / sources.length - width / 2 : 0,
    };
  });
  if (preserveOrder) {
    const startX = Math.min(...preferred.map((item) => item.x));
    preferred[0].x = startX;
  } else {
    preferred.sort((a, b) => a.x - b.x);
  }

  preferred.forEach((item, index) => {
    const previous = preferred[index - 1];
    if (previous) item.x = Math.max(item.x, previous.x + previous.width + gap);
    positions.set(item.id, { x: item.x, y });
  });
}

export function layoutVisibleGraph(nodes, edges, view, { layout = 'horizontal', loadExpanded = false, referenceNodes = nodes, referenceEdges = edges } = {}) {
  const config = view.layout || {};
  const positions = new Map();
  const sizes = new Map(nodes.map((node) => [node.id, sizeOf(node, view)]));
  const expanded = nodes.find(node => node.id === loadExpanded && node.instances);
  if (expanded) sizes.set(expanded.id, { width: 920, height: 210 + Math.ceil(expanded.instances.length / 3) * 76 + Math.max(...expanded.instances.map(item => Math.max(item.inputs?.length || 0, item.outputs?.length || 0))) * 88 });
  if (loadExpanded && sizes.has('isa.load')) sizes.set('isa.load', { width: 1100, height: 850 });
  if (layout === 'vertical') return layoutVertical(nodes, edges, view, sizes);
  const activeIds = new Set(nodes.map((node) => node.id));
  const coreNodes = referenceNodes.filter((node) => CORE_KINDS.has(node.renderKind) && !node.attributes?.auxiliary);
  coreNodes.forEach((node) => { if (!sizes.has(node.id)) sizes.set(node.id, sizeOf(node, view)); });
  // Reserve the lowered graph's ranks from the start. Nodes retain their axes
  // when AIC/AIV handoffs appear; reserved nodes are never rendered early.
  const coreEdges = referenceEdges.filter((edge) => (edge.semanticKind === 'data' || edge.semanticKind === 'control') && !['data.qk-scores', 'data.prob-pv'].includes(edge.id));
  const dagrePositions = layoutDagre(coreNodes, coreEdges, sizes, config);
  dagrePositions.forEach((position, id) => positions.set(id, position));

  const containerMembers = config.containerMembers || {};
  const activeKernels = referenceNodes.filter((node) => node.renderKind === 'kernel');
  {
    const aicTop = 70;
    const aivTop = 350;
    spreadLane(containerMembers['kernel.aic'] || [], positions, sizes, aicTop, 170);
    spreadLane(containerMembers['kernel.aiv'] || [], positions, sizes, aivTop, 240);

    const aicCenter = aicTop + 98;
    const aivCenter = aivTop + 126;
    const externalSlots = config.externalSlots || {};
    coreNodes.forEach((node) => {
      const slot = externalSlots[node.id];
      if (!slot || !positions.has(node.id)) return;
      const size = sizes.get(node.id);
      const targetCenter = slot === 'aic' ? aicCenter : slot === 'aiv' ? aivCenter : (aicCenter + aivCenter) / 2;
      positions.set(node.id, { ...positions.get(node.id), y: targetCenter - size.height / 2 });
    });
    // Preserve Dagre ranks, but align the main chains inside each semantic lane.
    // Branches keep a separate baseline so their edges need only two bends.
    const align = (ids, center) => ids.forEach((id) => {
      if (positions.has(id)) positions.set(id, { ...positions.get(id), y: center - sizes.get(id).height / 2 });
    });
    align(['value.q', 'op.qk', 'handoff.c2v-push', 'handoff.v2c-pop', 'op.pv', 'value.out'], aicCenter);
    align(['handoff.c2v-pop', 'value.scores', 'op.exp', 'value.prob', 'handoff.v2c-push'], aivCenter);
    align(['op.rowmax', 'value.max', 'op.rowsum', 'value.sum'], aivCenter - 100);
    // Align the incoming tensor's center with the actual multi-input port.
    for (const edge of edges) {
      if (!['lhs', 'rhs'].includes(edge.targetPort) || !positions.has(edge.source) || !positions.has(edge.target)) continue;
      if (edge.source !== 'value.q') continue;
      const fraction = edge.targetPort === 'lhs' ? .5 : .8;
      positions.set(edge.source, { ...positions.get(edge.source), y: positions.get(edge.target).y + sizes.get(edge.target).height * fraction - sizes.get(edge.source).height / 2 });
    }
    // Separation takes precedence over port alignment. Include room for the
    // domain label above each value; never stack inputs to save a bend.
    const placed = [];
    for (const node of coreNodes) {
      const p = positions.get(node.id), size = sizes.get(node.id);
      if (!p) continue;
      for (let attempt = 0; attempt < coreNodes.length; attempt++) {
        const hit = placed.find(box => p.x < box.x + box.width + 12 && p.x + size.width + 12 > box.x && p.y - 26 < box.y + box.height + 18 && p.y + size.height + 18 > box.y - 26);
        if (!hit) break;
        p.y = hit.y + hit.height + 48;
      }
      placed.push({...p,...size});
    }
  }

  const kernelBounds = [];
  activeKernels.forEach((kernel) => {
    const bounds = boundsOf(containerMembers[kernel.id] || [], positions, sizes, {
      left: 72, right: 72, top: 156, bottom: 80,
    });
    if (!bounds) return;
    positions.set(kernel.id, { x: bounds.x, y: bounds.y });
    sizes.set(kernel.id, { width: bounds.width, height: bounds.height });
    kernelBounds.push(kernel.id);
  });
  // Container padding is part of lane geometry, not just decoration.
  // Separate complete kernel bounds and translate every member together.
  const aic = positions.get('kernel.aic'), aiv = positions.get('kernel.aiv');
  if (aic && aiv) {
    const shift = Math.max(0, aic.y + sizes.get('kernel.aic').height + 72 - aiv.y);
    if (shift) {
      for (const id of ['kernel.aiv', ...(containerMembers['kernel.aiv'] || [])]) {
        const position = positions.get(id);
        if (position) positions.set(id, {...position, y:position.y + shift});
      }
    }
  }
  // External values are outside the complete kernel boundary, including its
  // control-region padding, not just outside its first/last OP.
  if (aic) {
    for (const id of ['value.q','value.kv']) {
      const p=positions.get(id);
      if(p) positions.set(id,{...p,x:Math.min(p.x,aic.x-sizes.get(id).width-40)});
    }
    const out=positions.get('value.out');
    if(out) positions.set('value.out',{...out,x:Math.max(out.x,aic.x+sizes.get('kernel.aic').width+40)});
  }
  for (const core of ['aic','aiv']) {
    const kernel = `kernel.${core}`, p = positions.get(kernel), s = sizes.get(kernel);
    if (!p) continue;
    for (const [kind,inset,top,bottom] of [['iter',18,42,20],['guard',36,88,36]]) {
      const id = `region.${core}.${kind}`;
      if (!activeIds.has(id)) continue;
      positions.set(id,{x:p.x+inset,y:p.y+top});
      sizes.set(id,{width:s.width-inset*2,height:s.height-top-bottom});
    }
  }

  const scope = nodes.find((node) => node.renderKind === 'scope');
  if (scope) {
    const scopeChildren = [...coreNodes.map((node) => node.id), ...kernelBounds];
    const bounds = boundsOf(scopeChildren, positions, sizes, {
      left: 100, right: 100, top: 370, bottom: 110,
    });
    if (bounds) {
      positions.set(scope.id, { x: bounds.x, y: bounds.y });
      sizes.set(scope.id, { width: bounds.width, height: bounds.height });
    }
  }

  const scopeBounds = scope && positions.has(scope.id)
    ? { ...positions.get(scope.id), ...sizes.get(scope.id) }
    : boundsOf(coreNodes.map((node) => node.id), positions, sizes, { top: 40, bottom: 40, left: 40, right: 40 });

  const gatherNodes = nodes.filter((node) => node.id.startsWith('gather.'));
  const gatherY = scopeBounds.y + 220;
  gatherNodes.forEach((node, i) => positions.set(node.id, { x: scopeBounds.x + 180 + i * 268, y: gatherY }));
  for (const [id, inset, top] of [['region.iter', 26, 62], ['region.guard', 52, 112]]) {
    if (!activeIds.has(id)) continue;
    positions.set(id, { x: scopeBounds.x + inset, y: scopeBounds.y + top });
    sizes.set(id, { width: scopeBounds.width - inset * 2, height: scopeBounds.height - top - (id === 'region.iter' ? 32 : 58) });
  }
  if (activeIds.has('region.runs')) {
    positions.set('region.runs', { x: scopeBounds.x + 152, y: gatherY - 42 });
    sizes.set('region.runs', { width: 1360, height: 132 });
  }

  const taskNodes = nodes.filter((node) => node.renderKind === 'task');
  if (taskNodes.length) {
    const taskEdges = edges.filter((edge) => edge.semanticKind === 'taskDependency');
    const taskPositions = layoutDagre(taskNodes, taskEdges, sizes, { ...config, rankSep: 300, nodeSep: 32 });
    const taskBounds = boundsOf(taskNodes.map((node) => node.id), taskPositions, sizes);
    const offsetX = scopeBounds.x + scopeBounds.width / 2 - (taskBounds.x + taskBounds.width / 2);
    const offsetY = scopeBounds.y - taskBounds.height - 104 - taskBounds.y;
    taskPositions.forEach((position, id) => positions.set(id, { x: position.x + offsetX, y: position.y + offsetY }));
  }
  const context = nodes.find((node) => node.renderKind === 'context');
  if (context) positions.set(context.id, { x: scopeBounds.x + (scopeBounds.width - sizes.get(context.id).width) / 2, y: scopeBounds.y + scopeBounds.height + 50 });

  const memoryNodes = nodes.filter((node) => node.renderKind === 'memory');
  placeAlignedBand(
    memoryNodes,
    edges.filter((edge) => edge.semanticKind === 'memoryBinding'),
    positions,
    sizes,
    scopeBounds.y + scopeBounds.height + 88,
    46,
  );

  const isaNodes = nodes.filter((node) => node.renderKind === 'isa');
  const memoryBottom = memoryNodes.length
    ? Math.max(...memoryNodes.map((node) => positions.get(node.id).y + sizes.get(node.id).height))
    : scopeBounds.y + scopeBounds.height;
  // Preserve the producer's x axis. Resolve width conflicts in y, never by
  // pushing an instruction sideways and adding an artificial routing dogleg.
  const placedIsa = [];
  const columns = [];
  const ownerGroups = new Map();
  isaNodes.filter((node) => node.id !== 'isa.load').forEach((node) => {
    const edge = edges.find((edge) => ['lowering', 'lineage'].includes(edge.semanticKind) && edge.target === node.id);
    if (!edge || !positions.has(edge.source)) return;
    if (!ownerGroups.has(edge.source)) ownerGroups.set(edge.source, []);
    ownerGroups.get(edge.source).push(node);
  });
  // Pack complete owner columns, not individual instructions. A neighboring
  // column must never split one owner's consecutive instruction stack.
  [...ownerGroups].sort(([a], [b]) => centerOf(positions.get(a), sizes.get(a)).x - centerOf(positions.get(b), sizes.get(b)).x).forEach(([owner, members]) => {
    const center = centerOf(positions.get(owner), sizes.get(owner)).x;
    columns.push(center);
    const width = Math.max(...members.map(node => sizes.get(node.id).width));
    const height = members.reduce((sum, node) => sum + sizes.get(node.id).height, 0) + (members.length - 1) * 16;
    const x = center - width / 2;
    let y = memoryBottom + 104;
    for (const previous of placedIsa) {
      if (x < previous.x + previous.width + 28 && x + width + 28 > previous.x) y = Math.max(y, previous.y + previous.height + 56);
    }
    let cursor = y;
    members.forEach(node => {
      const size = sizes.get(node.id);
      positions.set(node.id, { x:center-size.width/2, y:cursor });
      cursor += size.height + 16;
    });
    placedIsa.push({ x, y, width, height });
  });
  // Keep MemRefs out of the straight producer-to-instruction corridors.
  // Their many-to-one binding edges retain orthogonal routing.
  const placedMemory = [];
  memoryNodes.forEach((node) => {
    const position = positions.get(node.id);
    const size = sizes.get(node.id);
    let x = position.x;
    for (let i = 0; i <= columns.length + memoryNodes.length; i++) {
      const column = columns.find((column) => column >= x - 24 && column <= x + size.width + 24);
      const obstacle = placedMemory.find((box) => x < box.x + box.width + 28 && x + size.width + 28 > box.x);
      if (column == null && !obstacle) break;
      x = Math.max(x, column == null ? x : column + 32, obstacle ? obstacle.x + obstacle.width + 46 : x);
    }
    positions.set(node.id, { ...position, x });
    placedMemory.push({ x, ...size });
  });
  if (sizes.has('isa.load')) {
    // This whole-program aggregate has no single producer; keep it separate
    // rather than attributing all loads to the QK matmul.
    positions.set('isa.load', { x: scopeBounds.x + (scopeBounds.width - sizes.get('isa.load').width) / 2, y: Math.max(memoryBottom + 104, ...placedIsa.map((box) => box.y + box.height)) + 70 });
  }

  nodes.filter((node) => BACKGROUND_KINDS.has(node.renderKind)).forEach((node) => {
    if (!positions.has(node.id)) positions.set(node.id, { x: 0, y: 0 });
  });

  const positionedIds = [...positions.keys()].filter((id) => activeIds.has(id));
  const minX = Math.min(...positionedIds.map((id) => positions.get(id).x));
  const minY = Math.min(...positionedIds.map((id) => positions.get(id).y));
  const shiftX = 84 - minX;
  const shiftY = 82 - minY;
  positionedIds.forEach((id) => {
    const position = positions.get(id);
    positions.set(id, { x: position.x + shiftX, y: position.y + shiftY });
  });

  return { positions, sizes };
}

// Upright computation spines, with local side branches and separate support
// columns. Container geometry is derived afresh, never transposed from LR.
function layoutVertical(nodes, edges, view, sizes) {
  const positions = new Map();
  const ids = new Set(nodes.map(n => n.id));
  const put = (id, cx, y) => { if (ids.has(id)) positions.set(id, {x:cx-sizes.get(id).width/2,y}); };
  const collections = nodes.filter(n => n.instances);
  const isaWidth = Math.max(310, ...collections.map(n => sizes.get(n.id).width));
  const aic = 500, aiv = aic + 650 + (collections.length ? isaWidth + 80 : 0);
  const rows = [
    ['value.q',aic-110,260],['value.kv',aic+110,260],
    ['op.qk',aic,500],['handoff.c2v-push',aic,650],
    ['handoff.c2v-pop',aiv,650],['value.scores',aiv,780],
    ['op.rowmax',aiv,920],['value.max',aiv,1050],
    ['op.exp',aiv,1190],['value.prob',aiv,1320],
    ['op.rowsum',aiv-210,1460],['value.sum',aiv-210,1590],
    ['handoff.v2c-push',aiv,1740],['handoff.v2c-pop',aic,1740],
    ['op.pv',aic,1880],['value.out',aic,2120],
  ];
  rows.forEach(args => put(...args));
  const members = view.layout.containerMembers;
  for (const core of ['aic','aiv']) {
    const id = `kernel.${core}`;
    if (!ids.has(id)) continue;
    const b = boundsOf(members[id], positions, sizes, {left:66,right:66,top:144,bottom:64});
    positions.set(id,{x:b.x,y:b.y}); sizes.set(id,{width:b.width,height:b.height});
    for (const [kind,inset,top] of [['iter',16,42],['guard',32,86]]) {
      const region=`region.${core}.${kind}`;
      if (!ids.has(region)) continue;
      positions.set(region,{x:b.x+inset,y:b.y+top});
      sizes.set(region,{width:b.width-2*inset,height:b.height-top-inset});
    }
  }
  const scopeIds=nodes.filter(n => CORE_KINDS.has(n.renderKind) && !n.attributes?.auxiliary || n.renderKind==='kernel').map(n=>n.id);
  const b=boundsOf(scopeIds,positions,sizes,{left:60,right:60,top:180,bottom:64});
  for (const n of nodes.filter(n=>n.renderKind==='scope')) {
    positions.set(n.id,{x:b.x,y:b.y});sizes.set(n.id,{width:b.width,height:b.height});
  }
  for (const [id,inset,top] of [['region.iter',16,42],['region.guard',32,86]]) {
    if (!ids.has(id)) continue;
    positions.set(id,{x:b.x+inset,y:b.y+top}); sizes.set(id,{width:b.width-2*inset,height:b.height-top-inset});
  }
  const gathers=nodes.filter(n=>n.id.startsWith('gather.'));
  gathers.forEach((n,i)=>put(n.id,aic+350,380+i*80));
  if(ids.has('region.runs')) {
    const label=nodes.find(n=>n.id==='region.runs').label;
    const width=Math.max(320,label.length*10+48,...gathers.map(n=>sizes.get(n.id).width+48));
    positions.set('region.runs',{x:aic+350-width/2,y:330});
    sizes.set('region.runs',{width,height:Math.max(140,gathers.length*80+80)});
  }
  put('task.plan',(aic+aiv)/2,-200);put('task.qk-pv',(aic+aiv)/2,-50);
  // Memory uses its own left rail. ISA groups remain consecutive beside their
  // producer's kernel; expanded groups reserve their complete measured bounds.
  nodes.filter(n=>n.renderKind==='memory').forEach((n,i)=>put(n.id,80,850+i*190));
  for(const core of ['aic','aiv']) {
    const cx=core==='aic'?aic:aiv;
    const groups=new Map();
    collections.filter(n=>n.kernel?.endsWith(core)).forEach(n=>{
      if(!groups.has(n.owner)) groups.set(n.owner,[]);
      groups.get(n.owner).push(n);
    });
    let bottom=330;
    [...groups].sort(([a],[b])=>(positions.get(a)?.y||0)-(positions.get(b)?.y||0)).forEach(([owner,items])=>{
      const ownerTop = positions.get(owner)?.y ?? 330;
      // Leaf-to-ISA links join side-center ports, not top edges. Keep kernel
      // collections near their header rather than the center of a tall scope.
      const alignedTop = owner.startsWith('kernel.') ? ownerTop
        : ownerTop + sizes.get(owner).height / 2 - sizes.get(items[0].id).height / 2;
      let y=Math.max(bottom,alignedTop);
      for(const n of items) {positions.set(n.id,{x:cx+300,y});y+=sizes.get(n.id).height+16;}
      bottom=y+32;
    });
  }
  // Registry-only auxiliary objects retain deterministic, disjoint fallback slots.
  let fallbackY=2350;
  for(const n of nodes) if(!positions.has(n.id)) {put(n.id,aic,fallbackY);fallbackY+=sizes.get(n.id).height+48;}
  const minX=Math.min(...[...positions.values()].map(p=>p.x));
  const minY=Math.min(...[...positions.values()].map(p=>p.y));
  for(const [id,p] of positions) positions.set(id,{x:p.x-minX+84,y:p.y-minY+82});
  return {positions,sizes};
}
