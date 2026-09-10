(function () {
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const relationClass = relation => `dg-edge--${relation}`;
  const typeTitle = {finding: '01 / Finding', reasoning: '02 / Abnormal dimension', entity: '03 / Promoted object', cause: 'Cause exploration', evidence: 'Observed evidence', hypothesis: 'Hypothesis / pending validation', diagnosis: 'Diagnosis', experiment: 'Recommended experiment'};

  function svg(tag, attrs = {}) { const el = document.createElementNS(SVG_NS, tag); Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value)); return el; }
  function el(tag, className, text) { const node = document.createElement(tag); if (className) node.className = className; if (text !== undefined) node.textContent = text; return node; }
  function clamp(value, min, max) { return Math.min(max, Math.max(min, value)); }

  function makeThumbnail(id, evidence, detail) {
    const button = el('button', 'dg-evidence-thumb');
    button.type = 'button'; button.dataset.dgEvidence = id; button.setAttribute('aria-label', `查看 ${detail.title} 详细证据`);
    if (detail.type === 'bars') {
      const bars = el('span', 'dg-thumb-bars'); const max = Math.max(...detail.values);
      detail.values.forEach((value, index) => { const bar = el('i', (detail.hotIndex === index || detail.hotIndices?.includes(index)) ? 'is-hot' : ''); bar.style.setProperty('--h', `${Math.max(19, Math.round(value / max * 100))}%`); bars.append(bar); });
      button.append(bars);
    } else {
      const table = el('span', 'dg-thumb-table'); detail.rows.slice(0, 2).forEach((row, index) => row.forEach((cell, cellIndex) => table.append(el(cellIndex === 0 && index === 1 ? 'b' : 'span', '', cell)))); button.append(table);
    }
    button.append(el('span', 'dg-thumb-caption', evidence.signal || detail.note));
    return button;
  }

  function renderNote(detail, anchor, overlay, viewport) {
    const note = el('section', 'dg-note'); note.dataset.dgNote = 'true';
    const close = el('button', 'dg-note__close', '×'); close.type = 'button'; close.setAttribute('aria-label', '关闭证据便利贴'); close.addEventListener('click', () => note.remove());
    note.append(close, el('h3', '', detail.title));
    if (detail.type === 'bars') {
      const bars = el('div', 'dg-note__bars'); const max = Math.max(...detail.values);
      detail.values.forEach((value, index) => {
        const row = el('div', `dg-note__bar${detail.hotIndex === index || detail.hotIndices?.includes(index) ? ' is-hot' : ''}`);
        const bar = el('i'); bar.style.setProperty('--w', `${Math.round(value / max * 100)}%`);
        row.append(el('span', '', detail.labels[index]), bar, el('strong', '', Number.isInteger(value) ? String(value) : `${value.toFixed(2)}×`)); bars.append(row);
      }); note.append(bars);
    } else {
      const table = el('div', 'dg-note__table'); table.append(el('span', '', ''), el('span', '', '基线'), el('span', '', '当前'));
      detail.rows.forEach(row => row.forEach((cell, index) => table.append(el(index === 0 ? 'strong' : 'span', '', cell)))); note.append(table);
    }
    note.append(el('p', '', detail.note)); overlay.replaceChildren(note);
    const viewportRect = viewport.getBoundingClientRect(); const anchorRect = anchor.getBoundingClientRect();
    note.style.maxHeight = `${Math.max(160, viewport.clientHeight - 24)}px`;
    const width = note.offsetWidth; const height = note.offsetHeight; let left = anchorRect.right - viewportRect.left + 14; let top = anchorRect.top - viewportRect.top - 20;
    if (left + width > viewport.clientWidth - 12) left = anchorRect.left - viewportRect.left - width - 14;
    if (top + height > viewport.clientHeight - 12) top = viewport.clientHeight - height - 12;
    if (top < 12) top = 12; if (left < 12) left = 12;
    note.style.left = `${left}px`; note.style.top = `${top}px`; note.style.setProperty('--dg-note-angle', `${(anchorRect.left % 3) - 1}deg`);
  }

  class DiagnosticGraphController {
    constructor(root, data, actions = {}) {
      this.root = root; this.data = data; this.actions = actions; this.nodesById = new Map(data.nodes.map(node => [node.id, node]));
      this.manualOffsets = new Map(); this.positions = new Map(); this.camera = {x: 0, y: 0, k: 1}; this.selectedId = null; this.layoutMode = false; this.noteEvidenceId = null;
      this.renderShell(); this.renderNodes(); this.layout(); this.bind(); this.layoutWithElk().finally(() => requestAnimationFrame(() => this.fit()));
    }
    renderShell() {
      this.root.className = 'diagnostic-graph'; this.root.innerHTML = '';
      this.toolbar = el('div', 'diagnostic-graph__toolbar'); this.toolbar.setAttribute('aria-label', '诊断图画布控制');
      const controls = [['−', '缩小画布', 'zoom-out'], ['+', '放大画布', 'zoom-in'], ['Fit', '适应画布', 'fit'], ['100%', '恢复 100% 比例', 'hundred'], ['调整布局', '调整节点布局', 'layout'], ['重置', '重置自动布局', 'reset']];
      controls.forEach(([text, label, action]) => { const button = el('button', action === 'fit' || action === 'hundred' || action === 'layout' || action === 'reset' ? 'btn btn-ghost btn-compact' : 'btn btn-ghost btn-icon', text); button.type = 'button'; button.dataset.dgAction = action; button.setAttribute('aria-label', label); this.toolbar.append(button); });
      this.zoomOutput = el('output', '', '100%'); this.zoomOutput.setAttribute('aria-live', 'polite'); this.toolbar.insertBefore(this.zoomOutput, this.toolbar.querySelector('[data-dg-action="fit"]'));
      this.viewport = el('div', 'diagnostic-graph__viewport'); this.viewport.tabIndex = 0; this.viewport.setAttribute('aria-label', '诊断推理画布');
      this.world = el('div', 'diagnostic-graph__world'); this.edgeSvg = svg('svg', {class: 'diagnostic-graph__edges', 'aria-hidden': 'true'}); this.edgeSvg.append(this.markerDefs()); this.nodeLayer = el('div', 'diagnostic-graph__nodes'); this.world.append(this.edgeSvg, this.nodeLayer); this.overlay = el('div', 'diagnostic-graph__overlay');
      this.hint = el('div', 'dg-canvas-hint'); this.hint.innerHTML = '拖动空白处平移 · <kbd>Ctrl</kbd> + 滚轮缩放'; this.viewport.append(this.world, this.overlay, this.hint); this.root.append(this.toolbar, this.viewport);
    }
    markerDefs() { const defs = svg('defs'); const main = svg('marker', {id: 'dg-arrow', markerWidth: 8, markerHeight: 8, refX: 6, refY: 4, orient: 'auto'}); main.append(svg('path', {d: 'M1 1 L7 4 L1 7', fill: 'none', stroke: '#c29570', 'stroke-width': 1.2})); const muted = svg('marker', {id: 'dg-arrow-muted', markerWidth: 8, markerHeight: 8, refX: 6, refY: 4, orient: 'auto'}); muted.append(svg('path', {d: 'M1 1 L7 4 L1 7', fill: 'none', stroke: '#7d887e', 'stroke-width': 1.1})); defs.append(main, muted); return defs; }
    renderNodes() {
      this.nodeElements = new Map(); this.nodeLayer.replaceChildren();
      this.data.nodes.forEach(node => { const item = el('article', `dg-node dg-node--${node.kind}${node.state ? ` is-${node.state}` : ''}`); item.id = `dg-node-${node.id}`; item.dataset.dgNode = node.id; item.tabIndex = 0; item.setAttribute('aria-label', `${node.title} ${node.signal}`); item.append(el('span', 'dg-node__eyebrow', typeTitle[node.kind]), el('h3', 'dg-node__title', node.title), el('strong', 'dg-node__signal', node.signal), el('p', 'dg-node__summary', node.summary));
        (node.evidenceRefs || []).forEach(evidenceId => item.append(makeThumbnail(evidenceId, node, this.data.evidenceById[evidenceId])));
        if (node.kind === 'experiment') { const action = el('button', 'btn btn-solid btn-compact', '创建验证实验'); action.type = 'button'; action.dataset.dgExperiment = 'true'; action.addEventListener('click', event => { event.stopPropagation(); this.actions.openExperiment?.(); }); item.append(action); }
        this.nodeLayer.append(item); this.nodeElements.set(node.id, item); });
    }
    measure() { return new Map([...this.nodeElements].map(([id, item]) => [id, {width: item.offsetWidth, height: item.offsetHeight}])); }
    ranks() {
      const ranks = new Map(this.data.nodes.map(node => [node.id, 0]));
      const incoming = new Map(this.data.nodes.map(node => [node.id, 0]));
      const outgoing = new Map(this.data.nodes.map(node => [node.id, []]));
      this.data.edges.forEach(edge => {
        if (!outgoing.has(edge.source) || !incoming.has(edge.target)) throw new Error(`Unknown graph edge: ${edge.id}`);
        outgoing.get(edge.source).push(edge); incoming.set(edge.target, incoming.get(edge.target) + 1);
      });
      const queue = this.data.nodes.filter(node => incoming.get(node.id) === 0).map(node => node.id);
      while (queue.length) {
        const source = queue.shift();
        outgoing.get(source).forEach(edge => {
          const advance = edge.relation === 'diagnose' || edge.relation === 'validate' ? 0 : 1;
          ranks.set(edge.target, Math.max(ranks.get(edge.target), ranks.get(source) + advance));
          incoming.set(edge.target, incoming.get(edge.target) - 1);
          if (incoming.get(edge.target) === 0) queue.push(edge.target);
        });
      }
      return ranks;
    }
    layout() {
      this.elkSections = null;
      const sizes = this.measure(); const ranks = this.ranks(); const layers = new Map(); this.data.nodes.forEach(node => { const rank = ranks.get(node.id) ?? 0; if (!layers.has(rank)) layers.set(rank, []); layers.get(rank).push(node); });
      const layerWidths = [...layers.entries()].map(([rank, nodes]) => [rank, Math.max(...nodes.map(node => sizes.get(node.id).width))]); let x = 72; const xByRank = new Map(); layerWidths.forEach(([rank, width]) => { xByRank.set(rank, x); x += width + 56; });
      this.positions.clear(); const mainY = 390;
      layers.forEach((nodes, rank) => { const gap = nodes.some(node => node.kind === 'evidence') ? 42 : nodes.length > 1 ? 32 : 0; const total = nodes.reduce((sum, node) => sum + sizes.get(node.id).height, 0) + Math.max(0, nodes.length - 1) * gap; let y = mainY - total / 2; nodes.forEach(node => { const size = sizes.get(node.id); const offset = this.manualOffsets.get(node.id) || {x: 0, y: 0}; this.positions.set(node.id, {x: xByRank.get(rank) + offset.x, y: y + offset.y, width: size.width, height: size.height}); y += size.height + gap; }); });
      const firstBounds = [...this.positions.values()].reduce((box, pos) => ({left: Math.min(box.left, pos.x), top: Math.min(box.top, pos.y), right: Math.max(box.right, pos.x + pos.width), bottom: Math.max(box.bottom, pos.y + pos.height)}), {left: Infinity, top: Infinity, right: 0, bottom: 0});
      const shiftX = 72 - firstBounds.left; const shiftY = 72 - firstBounds.top;
      this.positions.forEach(pos => { pos.x += shiftX; pos.y += shiftY; });
      const bounds = [...this.positions.values()].reduce((box, pos) => ({right: Math.max(box.right, pos.x + pos.width), bottom: Math.max(box.bottom, pos.y + pos.height)}), {right: 0, bottom: 0});
      this.world.style.setProperty('--dg-world-width', `${bounds.right + 72}px`); this.world.style.setProperty('--dg-world-height', `${Math.max(620, bounds.bottom + 72)}px`); this.renderGeometry();
    }
    async layoutWithElk() {
      if (typeof window.ELK !== 'function') return;
      const sizes = this.measure(); const elk = new window.ELK();
      const graph = await elk.layout({id: 'diagnostic-root', layoutOptions: {
        'elk.algorithm': 'layered', 'elk.direction': 'RIGHT', 'elk.edgeRouting': 'ORTHOGONAL',
        'elk.spacing.nodeNode': '32', 'elk.layered.spacing.nodeNodeBetweenLayers': '56',
        'elk.layered.nodePlacement.strategy': 'NETWORK_SIMPLEX', 'elk.padding': '[top=72,left=72,bottom=72,right=72]'
      }, children: this.data.nodes.map(node => ({id: node.id, width: sizes.get(node.id).width, height: sizes.get(node.id).height})), edges: this.data.edges.map(edge => ({id: edge.id, sources: [edge.source], targets: [edge.target]}))});
      const compactWidth = parseFloat(this.world.style.getPropertyValue('--dg-world-width'));
      if (Number.isFinite(compactWidth) && graph.width > compactWidth * 1.12) return;
      this.positions.clear(); graph.children.forEach(node => {
        const offset = this.manualOffsets.get(node.id) || {x: 0, y: 0}; const size = sizes.get(node.id);
        this.positions.set(node.id, {x: node.x + offset.x, y: node.y + offset.y, width: size.width, height: size.height});
      });
      this.elkSections = new Map(graph.edges.map(edge => [edge.id, edge.sections?.[0]]).filter(([, section]) => section));
      this.world.style.setProperty('--dg-world-width', `${Math.ceil(graph.width)}px`); this.world.style.setProperty('--dg-world-height', `${Math.ceil(graph.height)}px`); this.renderGeometry();
    }
    renderGeometry() {
      this.nodeElements.forEach((item, id) => { const pos = this.positions.get(id); item.style.transform = `translate(${pos.x}px,${pos.y}px)`; }); this.renderEdges(); this.applyCamera();
    }
    renderEdges() {
      const defs = this.edgeSvg.querySelector('defs'); this.edgeSvg.replaceChildren(defs);
      this.data.edges.forEach(edge => { const from = this.positions.get(edge.source); const to = this.positions.get(edge.target); if (!from || !to) return; const section = this.elkSections?.get(edge.id); const vertical = Math.abs(from.x - to.x) < 1 && to.y > from.y; const start = section?.startPoint || (vertical ? {x: from.x + from.width / 2, y: from.y + from.height} : {x: from.x + from.width, y: from.y + from.height / 2}); const end = section?.endPoint || (vertical ? {x: to.x + to.width / 2, y: to.y} : {x: to.x, y: to.y + to.height / 2}); const bends = section?.bendPoints || []; const channel = vertical ? Math.round((start.y + end.y) / 2) : Math.round((start.x + end.x) / 2); const d = section ? `M ${start.x} ${start.y} ${bends.map(point => `L ${point.x} ${point.y}`).join(' ')} L ${end.x} ${end.y}` : vertical ? `M ${start.x} ${start.y} V ${end.y}` : `M ${start.x} ${start.y} H ${channel} V ${end.y} H ${end.x}`; const path = svg('path', {class: `dg-edge ${relationClass(edge.relation)}`, d}); path.dataset.dgEdge = edge.id; path.append(svg('title')); path.querySelector('title').textContent = edge.label || edge.relation; this.edgeSvg.append(path);
        if (edge.label) { const group = svg('g', {class: 'dg-edge-label'}); const width = Math.max(60, edge.label.length * 7 + 16); const labelX = vertical ? start.x : channel; const labelY = vertical ? channel : (start.y + end.y) / 2; group.setAttribute('transform', `translate(${labelX} ${labelY})`); group.append(svg('rect', {x: -width / 2, y: -10, width, height: 20, rx: 4}), svg('text', {y: 4})); group.lastChild.textContent = edge.label; this.edgeSvg.append(group); }
      }); this.updateFocus();
    }
    applyCamera() { this.world.style.transform = `translate(${this.camera.x}px,${this.camera.y}px) scale(${this.camera.k})`; this.zoomOutput.value = `${Math.round(this.camera.k * 100)}%`; this.zoomOutput.textContent = this.zoomOutput.value; }
    fit() { const width = parseFloat(this.world.style.getPropertyValue('--dg-world-width')) || 1600; const height = parseFloat(this.world.style.getPropertyValue('--dg-world-height')) || 900; const pad = 34; const k = Math.min((this.viewport.clientWidth - pad * 2) / width, (this.viewport.clientHeight - pad * 2) / height); this.camera.k = Math.max(.08, k); this.camera.x = (this.viewport.clientWidth - width * this.camera.k) / 2; this.camera.y = (this.viewport.clientHeight - height * this.camera.k) / 2; this.applyCamera(); }
    setScale(next, point) { const old = this.camera.k; const k = clamp(next, .1, 3); const anchor = point || {x: this.viewport.clientWidth / 2, y: this.viewport.clientHeight / 2}; this.camera.x = anchor.x - (anchor.x - this.camera.x) * k / old; this.camera.y = anchor.y - (anchor.y - this.camera.y) * k / old; this.camera.k = k; this.overlay.replaceChildren(); this.noteEvidenceId = null; this.applyCamera(); }
    setMode(mode) { this.layoutMode = mode === 'layout'; this.viewport.dataset.dgLayout = String(this.layoutMode); this.toolbar.querySelector('[data-dg-action="layout"]').classList.toggle('is-active', this.layoutMode); this.toolbar.querySelector('[data-dg-action="layout"]').textContent = this.layoutMode ? '完成布局' : '调整布局'; }
    focusFor(id) { const node = this.nodesById.get(id); return new Set([id, ...(node?.focusRefs || []), ...(node?.supportRefs || []), ...(node?.parentRefs || [])]); }
    updateFocus() { const focused = this.selectedId ? this.focusFor(this.selectedId) : null; this.nodeElements.forEach((item, id) => { item.classList.toggle('is-selected', id === this.selectedId); item.classList.toggle('is-dimmed', Boolean(focused && !focused.has(id))); }); this.edgeSvg.querySelectorAll('.dg-edge').forEach(path => { const edge = this.data.edges.find(item => item.id === path.dataset.dgEdge); const related = !focused || (focused.has(edge.source) && focused.has(edge.target)); path.classList.toggle('is-related', Boolean(focused && related)); path.classList.toggle('is-dimmed', Boolean(focused && !related)); }); this.edgeSvg.querySelectorAll('.dg-edge-label').forEach(label => label.classList.toggle('is-dimmed', Boolean(focused && !label.previousElementSibling.classList.contains('is-related')))); }
    select(id) { this.selectedId = this.selectedId === id ? null : id; this.updateFocus(); }
    isOverlap(id) { const current = this.positions.get(id); return [...this.positions.entries()].some(([otherId, other]) => otherId !== id && current.x < other.x + other.width && current.x + current.width > other.x && current.y < other.y + other.height && current.y + current.height > other.y); }
    bind() {
      this.toolbar.addEventListener('click', event => { const action = event.target.closest('[data-dg-action]')?.dataset.dgAction; if (!action) return; if (action === 'zoom-in') this.setScale(this.camera.k + .12); if (action === 'zoom-out') this.setScale(this.camera.k - .12); if (action === 'fit') this.fit(); if (action === 'hundred') { this.camera = {x: 42, y: 42, k: 1}; this.applyCamera(); } if (action === 'layout') this.setMode(this.layoutMode ? 'browse' : 'layout'); if (action === 'reset') { this.manualOffsets.clear(); this.layout(); this.layoutWithElk().finally(() => this.fit()); } });
      this.viewport.addEventListener('wheel', event => { event.preventDefault(); const rect = this.viewport.getBoundingClientRect(); const point = {x: event.clientX - rect.left, y: event.clientY - rect.top}; if (event.ctrlKey || event.metaKey) this.setScale(this.camera.k * (event.deltaY > 0 ? .9 : 1.1), point); else { this.camera.x -= event.deltaX; this.camera.y -= event.deltaY; this.overlay.replaceChildren(); this.noteEvidenceId = null; this.applyCamera(); } }, {passive: false});
      this.viewport.addEventListener('pointerdown', event => { const node = event.target.closest('[data-dg-node]'); const thumb = event.target.closest('[data-dg-evidence]'); if (thumb) return; if (node && this.layoutMode) { this.dragNode(event, node); return; } if (node) return; this.pan(event); });
      this.viewport.addEventListener('click', event => { if (event.target.closest('[data-dg-evidence]')) return; const node = event.target.closest('[data-dg-node]'); if (node) this.select(node.dataset.dgNode); else { this.selectedId = null; this.overlay.replaceChildren(); this.noteEvidenceId = null; this.updateFocus(); } });
      this.nodeLayer.addEventListener('click', event => {
        const experiment = event.target.closest('[data-dg-experiment]');
        if (experiment) { event.stopPropagation(); this.actions.openExperiment?.(); return; }
        const thumb = event.target.closest('[data-dg-evidence]'); if (!thumb) return; event.stopPropagation(); const id = thumb.dataset.dgEvidence; if (this.noteEvidenceId === id) { this.overlay.replaceChildren(); this.noteEvidenceId = null; return; } this.noteEvidenceId = id; renderNote(this.data.evidenceById[id], thumb, this.overlay, this.viewport);
      });
      this.viewport.addEventListener('keydown', event => { if (event.key === 'Escape') { if (this.noteEvidenceId) { this.overlay.replaceChildren(); this.noteEvidenceId = null; } else if (this.selectedId) { this.selectedId = null; this.updateFocus(); } } if (event.key === ' ' && event.target === this.viewport) event.preventDefault(); });
      new ResizeObserver(() => { if (!this.userCameraMoved) this.fit(); }).observe(this.viewport);
    }
    pan(event) { const start = {x: event.clientX, y: event.clientY, cameraX: this.camera.x, cameraY: this.camera.y}; this.viewport.dataset.dgPanning = 'true'; this.viewport.setPointerCapture(event.pointerId); const move = moveEvent => { this.userCameraMoved = true; this.camera.x = start.cameraX + moveEvent.clientX - start.x; this.camera.y = start.cameraY + moveEvent.clientY - start.y; this.overlay.replaceChildren(); this.noteEvidenceId = null; this.applyCamera(); }; const end = endEvent => { this.viewport.dataset.dgPanning = 'false'; this.viewport.removeEventListener('pointermove', move); this.viewport.removeEventListener('pointerup', end); this.viewport.removeEventListener('pointercancel', end); if (this.viewport.hasPointerCapture(endEvent.pointerId)) this.viewport.releasePointerCapture(endEvent.pointerId); }; this.viewport.addEventListener('pointermove', move); this.viewport.addEventListener('pointerup', end); this.viewport.addEventListener('pointercancel', end); }
    dragNode(event, item) { const id = item.dataset.dgNode; const start = {x: event.clientX, y: event.clientY, offset: this.manualOffsets.get(id) || {x: 0, y: 0}}; item.dataset.dgDragging = 'true'; item.setPointerCapture(event.pointerId); const move = moveEvent => { const offset = {x: start.offset.x + (moveEvent.clientX - start.x) / this.camera.k, y: start.offset.y + (moveEvent.clientY - start.y) / this.camera.k}; this.manualOffsets.set(id, offset); this.layout(); }; const end = endEvent => { if (this.isOverlap(id)) this.manualOffsets.set(id, start.offset); this.layout(); this.layoutWithElk(); delete item.dataset.dgDragging; item.removeEventListener('pointermove', move); item.removeEventListener('pointerup', end); item.removeEventListener('pointercancel', end); if (item.hasPointerCapture(endEvent.pointerId)) item.releasePointerCapture(endEvent.pointerId); }; item.addEventListener('pointermove', move); item.addEventListener('pointerup', end); item.addEventListener('pointercancel', end); }
    ensureLayout() { if (!this.positions.size) { this.layout(); this.fit(); } }
    resetLayout() { this.manualOffsets.clear(); this.layout(); this.layoutWithElk().finally(() => this.fit()); }
    resize() { if (!this.userCameraMoved) this.fit(); }
    destroy() { this.root.replaceChildren(); }
  }
  window.DiagnosticGraphController = DiagnosticGraphController;
}());
