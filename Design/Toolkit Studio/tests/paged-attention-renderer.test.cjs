const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const JS = path.join(__dirname, '..', 'js');
const read = (file) => fs.readFileSync(path.join(JS, file), 'utf8');

function functionSource(source, name) {
  const start = source.indexOf(`function ${name}(`);
  assert.notEqual(start, -1, `missing ${name}`);
  const boundaries = {
    pypaVisibleGraph: '  function renderPyptoTaskGraph()',
    renderPyptoTaskGraph: '  // 缩放读数：',
    syncPypaGraphZoom: '  let pypaSelectedNodeId = null;',
    pypaSourceRanges: '  function revealPypaSource(',
    revealPypaSource: '  function selectPypaGraphNode(',
    selectPypaGraphNode: '  /* 总览保留任务执行图',
  };
  const end = source.indexOf(boundaries[name], start);
  assert.notEqual(end, -1, `missing end boundary for ${name}`);
  return source.slice(start, end).trimEnd();
}

class Element {
  constructor(tag = 'div') {
    this.tagName = tag;
    this.children = [];
    this.style = {};
    this.dataset = {};
    this.attributes = {};
    this.listeners = {};
    this.classList = {
      values: new Set(),
      add: (...xs) => xs.forEach((x) => this.classList.values.add(x)),
      toggle: (x, on) => { if (on) this.classList.values.add(x); else this.classList.values.delete(x); },
      contains: (x) => this.classList.values.has(x),
    };
    this.clientWidth = tag === 'viewport' ? 960 : 0;
    this.clientHeight = tag === 'viewport' ? 640 : 0;
    this.innerHTML = '';
    this.textContent = '';
  }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  appendChild(child) { this.children.push(child); return child; }
  replaceChildren(...children) { this.children = children; }
  addEventListener(type, listener) { this.listeners[type] = listener; }
  setPointerCapture() {}
  getBoundingClientRect() { return { left: 0, top: 0, width: this.clientWidth, height: this.clientHeight }; }
  closest() { return null; }
}

function setup() {
  const demo = read('demo-v2.js');
  const viewport = new Element('viewport');
  const scene = new Element('scene');
  const zoom = new Element('zoom');
  const inspector = new Element('inspector');
  const rows = Array.from({ length: 450 }, (_, i) => {
    const row = new Element('source-row');
    row.dataset.pagedAttentionLine = String(i + 1);
    return row;
  });
  const elements = { '#pypaTaskViewport': viewport, '#pypaTaskScene': scene, '#pypaGraphZoom': zoom, '#pypaNodeInspector': inspector };
  const document = {
    createElement: (tag) => new Element(tag),
    createElementNS: (_ns, tag) => new Element(tag),
  };
  const window = {};
  const calls = [];
  const context = vm.createContext({ window, document, console, AbortController, ResizeObserver: class { observe() {} disconnect() {} } });
  for (const file of ['semantic-computation-presentation.js', 'paged-attention-pypto-source.js', 'paged-attention-semantic-graph.js', 'semantic-computation-layout.js', 'semantic-computation-graph.js']) {
    vm.runInContext(read(file), context, { filename: file });
  }
  window.pagedAttentionPyptoFocusForLine = (line) => ({ line });
  window.focusSourceRegion = (focus) => calls.push(['focus', focus.line]);
  window.revealSourceRegionLine = (line) => calls.push(['reveal', line]);
  window.state = { sourceRegionFocus: null };
  const picked = ['pypaVisibleGraph', 'renderPyptoTaskGraph', 'syncPypaGraphZoom', 'pypaSourceRanges', 'revealPypaSource', 'selectPypaGraphNode'];
  const extracted = picked.map((name) => functionSource(demo, name)).join('\n');
  const bootstrap = `
    const $ = (selector) => elements[selector] || null;
    const $$ = (selector) => selector === '#pypaTaskScene [data-pypa-node]' ? scene.children.filter((x) => x.dataset.pypaNode) : selector === '#pypaTaskScene [data-pypa-group]' ? scene.children.filter((x) => x.dataset.pypaGroup) : selector === '#dslEditor [data-paged-attention-line]' ? rows : [];
    let pypaGraphController = null;
    let pypaGraphView = 'compute';
    let pypaGraphTransform = null;
    let pypaGraphRenderedView = null;
    let pypaGraphSize = null;
    let pypaSelectedNodeId = null;
    const pypaExpandedStages = new Set();
    const PYPA_SCENE_WIDTH = 2260, PYPA_SCENE_HEIGHT = 1480;
    const pypaEscape = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const state = window.state;
    const pagedAttentionPyptoFocusForLine = window.pagedAttentionPyptoFocusForLine;
    const focusSourceRegion = window.focusSourceRegion;
    const revealSourceRegionLine = window.revealSourceRegionLine;
    const elements = window.__elements, scene = window.__scene, rows = window.__rows;
    window.__test = { get controller() { return pypaGraphController; }, get transform() { return { ...pypaGraphTransform }; }, get expanded() { return [...pypaExpandedStages]; }, get selected() { return pypaSelectedNodeId; }, pypaVisibleGraph, renderPyptoTaskGraph, pypaSourceRanges, revealPypaSource, selectPypaGraphNode, expand(id) { pypaExpandedStages.has(id) ? pypaExpandedStages.delete(id) : pypaExpandedStages.add(id); renderPyptoTaskGraph(); } };
  `;
  Object.assign(window, { __elements: elements, __scene: scene, __rows: rows });
  vm.runInContext(bootstrap + extracted, context);
  window.__test.renderPyptoTaskGraph();
  return { window, scene, viewport, zoom, inspector, rows, calls };
}

test('compute renderer shows typed nodes and never renders missing values as undefined', () => {
  const { window, scene } = setup();
  window.__test.expand('qk');
  const text = scene.children.map((node) => node.innerHTML || '').join(' ');
  assert.match(text, /<em>DATA<\/em>/);
  assert.match(text, /<em>OPERATION<\/em>/);
  assert.doesNotMatch(text, /undefined|null/);
});

test('expanding qk reveals q-tile detail and collapsing restores the compact graph', () => {
  const { window, scene } = setup();
  const compact = scene.children.length;
  const graph = window.__test.pypaVisibleGraph();
  const qk = graph.nodes.find((node) => node.id === 'qk');
  assert.ok(qk?.expandable);
  window.__test.expanded; // establish the isolated VM state before rerender
  const ctx = window.__test;
  assert.equal(ctx.pypaVisibleGraph().nodes.some((node) => node.id.includes('q-tile')), false);
  // The renderer's public interaction state is exercised by the extracted function's VM hook.
  window.__test.expand('qk');
  assert.ok(scene.children.some((node) => node.dataset.pypaNode?.includes('q-tile')));
  window.__test.expand('qk');
  assert.equal(scene.children.length, compact);
});

test('nested q-prep and norm expansion retains port-bearing child nodes', () => {
  const { window } = setup();
  window.__test.expand('q-prep');
  const qPrep = window.__test.pypaVisibleGraph().nodes;
  const norm = qPrep.find((item) => item.id === 'q-prep/norm');
  assert.ok(norm, 'q-prep exposes its nested RMSNorm composite');
  assert.ok(qPrep.filter((item) => item.parentId === 'q-prep').some((item) => item.ports), 'q-prep children retain ports');
  window.__test.expand('q-prep/norm');
  const normChildren = window.__test.pypaVisibleGraph().nodes.filter((item) => item.parentId === 'q-prep/norm');
  assert.ok(normChildren.length > 0);
  assert.ok(normChildren.some((item) => item.ports), 'norm child operations retain port metadata');
});

test('actual controller wheel and pointer listeners update zoom and preserve panning state', () => {
  const { window, viewport } = setup();
  const controller = window.__test.controller;
  const initial = controller.getScale();
  let prevented = false;
  viewport.listeners.wheel({ preventDefault() { prevented = true; }, clientX: 400, clientY: 300, deltaY: -1 });
  assert.ok(prevented);
  assert.ok(controller.getScale() > initial);
  const afterWheel = controller.getScale();
  viewport.listeners.pointerdown({ target: { closest: () => null }, clientX: 20, clientY: 30, pointerId: 1 });
  viewport.listeners.pointermove({ clientX: 50, clientY: 70 });
  viewport.listeners.pointerup({});
  const transformAfterPan = window.__test.transform;
  const beforeRebuild = window.__test.controller;
  window.__test.renderPyptoTaskGraph();
  assert.equal(window.__test.controller.getScale(), afterWheel);
  assert.deepEqual(window.__test.transform, transformAfterPan);
  assert.notEqual(window.__test.controller, beforeRebuild);
  assert.ok(window.__test.controller);
});

test('selecting qk and score fills the inspector with their structure and source evidence', () => {
  const { window, inspector, rows } = setup();
  window.__test.expand('qk');
  for (const id of ['qk', 'qk/score']) {
    const node = window.__test.pypaVisibleGraph().nodes.find((item) => item.id === id);
    window.__test.selectPypaGraphNode(node);
    assert.equal(inspector.hidden, false);
    assert.match(inspector.innerHTML, new RegExp(node.title));
    assert.match(inspector.innerHTML, /<dt>.*<\/dt><dd>.*<\/dd>/);
    assert.ok(rows.some((row) => row.classList.contains('is-pypa-source-evidence')));
  }
});

test('source range selection reveals the requested multi-range line and highlights only evidence rows', () => {
  const { window, rows, calls } = setup();
  const graph = window.__test.pypaVisibleGraph();
  const qk = graph.nodes.find((node) => node.id === 'qk');
  const cacheAppend = graph.nodes.find((node) => node.id === 'cache-append');
  const ranges = window.__test.pypaSourceRanges(qk);
  assert.ok(ranges.length > 1);
  window.__test.revealPypaSource(cacheAppend, 1);
  assert.deepEqual(calls.slice(-2), [['focus', 210], ['reveal', 210]]);
  window.__test.revealPypaSource(qk);
  assert.deepEqual(calls.slice(-2), [['focus', 285], ['reveal', 285]]);
  window.__test.selectPypaGraphNode(qk);
  for (const row of rows) {
    const line = Number(row.dataset.pagedAttentionLine);
    const expected = ranges.some(([start, end]) => line >= start && line <= end);
    assert.equal(row.classList.contains('is-pypa-source-evidence'), expected, `line ${line}`);
  }
});
