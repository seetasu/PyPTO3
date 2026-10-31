const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(require('node:path').join(__dirname, '../index-focus.html'), 'utf8');
const source = html.slice(html.indexOf('function applySingleChain('), html.indexOf('function singleTaskName('));

function node(tag, attributes = {}) {
  return {tag, attributes, children: [], classes: new Set(),
    classList: {toggle(name, on) { if (on) this.owner.classes.add(name); else this.owner.classes.delete(name); }},
    append(...items) { this.children.push(...items); }};
}
function item(id, x, y) {
  const element = node('g'); element.classList.owner = element;
  return {element, task: {taskId:id}, box: {x, w:12, h:8, cy:y}};
}
function render(entries, chain) {
  const svg = node('svg'); svg.classList.owner = svg; svg.viewBox = {baseVal:{width:800,height:400}};
  const byTask = new Map();
  for (const entry of entries) {
    const id = String(entry.task.taskId);
    if (!byTask.has(id)) byTask.set(id, []);
    byTask.get(id).push(entry);
  }
  const context = {$: () => svg, singleSelected:0, singleChains:[chain],
    laneRelationIndex:{byElement:new Map(entries.map(e => [e.element,e])),byTask},
    el:node, laneLeft:()=>76, singleTaskName:id=>'Task '+id};
  vm.runInNewContext(source + '\napplySingleChain();', context);
  return svg;
}

test('highlights all blocks and renders one label per task and one arrow per dependency', () => {
  const entries = [item('a',100,80), item('a',110,100), item('b',300,140), item('other',500,180)];
  const svg = render(entries, ['a','b']);
  assert(entries[0].element.classes.has('single-hit'));
  assert(entries[1].element.classes.has('single-hit'));
  assert(entries[2].element.classes.has('single-hit'));
  assert(entries[3].element.classes.has('single-muted'));
  const overlay = svg.children.find(c => c.attributes.class === 'single-chain-annotation');
  assert.equal(overlay.children.filter(c => c.attributes.class === 'single-task-label').length, 2);
  assert.equal(overlay.children.filter(c => c.attributes['marker-end']).length, 1);
  assert.equal(overlay.children.filter(c => c.attributes.class === 'single-task-boundary').length, 2);
  assert.equal(overlay.children.find(c => c.attributes.class === 'single-task-label').textContent, 'Task a');
});

test('only draws visible tasks without connecting across an offscreen intermediate task', () => {
  const svg = render([item('a',100,80), item('c',300,140)], ['a','b','c']);
  const overlay = svg.children.find(c => c.attributes.class === 'single-chain-annotation');
  assert.equal(overlay.children.filter(c => c.attributes['marker-end']).length, 0);
  assert.equal(overlay.children.filter(c => c.attributes.class === 'single-task-label').length, 2);
  assert.equal(overlay.children.filter(c => c.attributes.class === 'single-task-boundary').length, 2);
});

test('centers selected rows in the visible area below the sticky axis', () => {
  const centerSource = html.slice(html.indexOf('function centerSingleChain()'), html.indexOf('function updateSingleGrip()'));
  const wrap = {scrollTop:200, scrollHeight:1200, clientHeight:400, clientTop:0,
    getBoundingClientRect:()=>({top:100}), querySelector:()=>({getBoundingClientRect:()=>({height:50})})};
  const entry = (top,bottom) => ({element:{getBoundingClientRect:()=>({top,bottom})}});
  const context = {$:()=>wrap, singleSelected:0, singleChains:[['a','b']],
    laneRelationIndex:{byTask:new Map([['a',[entry(500,510)]],['b',[entry(530,540)]]])}};
  vm.runInNewContext(centerSource + '\ncenterSingleChain();', context);
  assert.equal(wrap.scrollTop, 395);
});

test('adds scroll space while a chain is selected and removes it when cleared', () => {
  const syncSource = html.slice(html.indexOf('function syncSingleScrollSpace()'), html.indexOf('function centerSingleChain()'));
  const nodes = {'lane-wrap':{clientHeight:700}, 'single-scroll-before':{style:{}}, 'single-scroll-after':{style:{}}};
  const context = {$:id=>nodes[id],singleSelected:0};
  vm.runInNewContext(syncSource + '\nsyncSingleScrollSpace();',context);
  assert.equal(nodes['single-scroll-before'].style.height,'350px');
  assert.equal(nodes['single-scroll-after'].style.height,'350px');
  context.singleSelected = null;
  vm.runInNewContext('syncSingleScrollSpace();',context);
  assert.equal(nodes['single-scroll-before'].style.height,'0px');
  assert.equal(nodes['single-scroll-after'].style.height,'0px');
});
