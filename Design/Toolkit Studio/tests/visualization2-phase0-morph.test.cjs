const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const context = vm.createContext({ window: {} });
for (const file of ['paged-attention-pypto-source', 'visualization2-phase0', 'visualization2-morph-player']) vm.runInContext(fs.readFileSync(path.join(__dirname, '../js', file + '.js'), 'utf8'), context);
const { createModel, createLayout } = context.window.PtoVisualization2Phase0;
const source = context.window.PTO_PAGED_ATTENTION_PYPTO_SOURCE;
const n = name => Number(source.match(new RegExp(`^${name} = (\\d+)`, 'm'))[1]);
const config = { blocks: n('ATTN_SPMD_BLOCKS'), kvHeads: n('NUM_KV_HEADS'), qPerKv: n('NUM_HEADS') / n('NUM_KV_HEADS'), activeBatch: n('BATCH'), ropeCores: n('ROPE_CORES'), lanesPerBlock: Number(source.match(/for aiv_id in pl\.split_aiv\((\d+)/)[1]) };
const model = createModel(config), layout = createLayout(model);

test('source configuration generates 128 unique items, 32 working lanes and 16 working Blocks', () => {
  assert.equal(model.items.length, 128);
  assert.equal(new Set(model.items.map(i => i.id)).size, 128);
  assert.equal(model.lanes.length, 48);
  assert.equal(model.activeLanes.length, 32);
  assert.ok(model.activeLanes.every(l => l.groups.length === 4));
  assert.equal(Array.from({ length: model.blocks }, (_, b) => model.groupsForBlock(b)).filter(g => g.length).length, 16);
  for (let b = 16; b < 24; b++) assert.equal(model.groupsForBlock(b).length, 0);
  for (const i of model.items) {
    assert.equal(i.gIndex, i.kvHead * model.activeBatch + i.batch);
    assert.equal(i.ropeCore, i.gIndex % model.ropeCores);
    assert.equal(i.block, Math.floor(i.ropeCore / model.lanesPerBlock));
    assert.equal(i.lane, i.ropeCore % model.lanesPerBlock);
    assert.equal(i.iteration, Math.floor(i.gIndex / model.ropeCores));
    assert.equal(model.itemFor(i.batch, i.kvHead), i);
  }
});

test('known Lane membership and cross-phase identities are correct', () => {
  assert.deepEqual(Array.from(model.lanes[0].groups, g => g.gIndex), [0, 32, 64, 96]);
  assert.deepEqual(Array.from(model.lanes[1].groups, g => g.gIndex), [1, 33, 65, 97]);
  assert.deepEqual(Array.from(model.lanes[16].groups, g => g.gIndex), [16, 48, 80, 112]);
  assert.deepEqual(Array.from(model.lanes[31].groups, g => g.gIndex), [31, 63, 95, 127]);
  const g = model.items[32];
  assert.deepEqual([g.batch, g.kvHead, g.block, g.lane, g.attentionTaskId, g.attentionTaskId % model.blocks], [0, 2, 0, 0, 2, 2]);
});

test('every item keeps one size and its Block slot follows the source mapping', () => {
  model.items.forEach((item, id) => {
    const sourceRect = layout.sources[id], targetRect = layout.targets[id];
    assert.equal(sourceRect.w, targetRect.w);
    assert.equal(sourceRect.h, targetRect.h);
    assert.equal(item.block, Math.floor((item.gIndex % config.ropeCores) / 2));
    assert.equal(item.lane, item.gIndex % 2);
    assert.equal(item.iteration, Math.floor(item.gIndex / config.ropeCores));
    assert.equal(item.block < 8, item.kvHead % 2 === 0);
    const lane = layout.laneRects[item.ropeCore];
    assert.ok(targetRect.x >= lane.x && targetRect.x + targetRect.w <= lane.x + lane.w);
  });
  assert.ok(model.lanes.slice(32).every(lane => lane.groups.length === 0));
  for (let kv = 0; kv < config.kvHeads; kv++) {
    const column = model.items.filter(item => item.kvHead === kv);
    assert.equal(new Set(column.map(item => layout.sources[item.gIndex].x)).size, 1);
    assert.equal(new Set(column.map(item => layout.targets[item.gIndex].x)).size, 1);
  }
  for (let pair = 0; pair < config.activeBatch / 2; pair++) {
    const even = layout.blockRects[pair], odd = layout.blockRects[pair + 8];
    assert.equal(even.y, odd.y);
    assert.ok(even.x + even.w <= odd.x);
  }
});

test('layout contains every item in its true Lane and keeps both matrices separate', () => {
  assert.equal(new Set(layout.sources.map(r => r.x)).size, 8);
  assert.equal(new Set(layout.sources.map(r => r.y)).size, 16);
  model.items.forEach((i, id) => {
    const r = layout.targets[id], lane = layout.laneRects[i.ropeCore], block = layout.blockRects[i.block];
    assert.ok(r.x > lane.x && r.x + r.w < lane.x + lane.w);
    assert.ok(r.y > lane.y && r.y + r.h < lane.y + lane.h);
    assert.ok(lane.x > block.x && lane.x + lane.w < block.x + block.w);
    assert.ok(lane.y > block.y && lane.y + lane.h < block.y + block.h);
  });
  for (const rects of [layout.sources, layout.targets, layout.blockRects.filter(Boolean)]) {
    rects.forEach((a, id) => {
      assert.ok(a.w > 0 && a.h > 0 && a.x >= 0 && a.x + a.w < layout.width && a.y >= 0 && a.y + a.h < layout.height);
      rects.slice(id + 1).forEach(b => assert.equal(a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y, false));
    });
  }
  assert.ok(layout.blockRects.slice(16).every(rect => rect === null));
  assert.deepEqual(Array.from(layout.columnOrder), [0, 2, 4, 6, 1, 3, 5, 7]);
  assert.equal(new Set(layout.sources.map(rect => rect.x)).size, 8);
  assert.equal(new Set(layout.sources.map(rect => rect.y)).size, 16);
  assert.equal(new Set(layout.blockRects.filter(Boolean).map(rect => rect.y)).size, 8);
});

test('runtime Batch sizes and partial final lanes use the same mapping formulas', () => {
  const m = createModel({ ...config, activeBatch: 7, ropeCores: 10 });
  assert.equal(m.items.length, 56);
  assert.ok(m.activeLanes.every(l => l.groups.length === 5 || l.groups.length === 6));
  assert.equal(new Set(m.items.map(i => i.attentionTaskId)).size, 56);
  const l = createLayout(m);
  m.items.forEach((item, id) => {
    const r = l.targets[id], owner = m.ownerOf(item.batch, item.kvHead);
    assert.equal(owner.ropeCore, item.gIndex % 10);
    assert.ok(r.y + r.h < l.laneRects[owner.ropeCore].y + l.laneRects[owner.ropeCore].h);
  });
});

function eventTarget() {
  const listeners = new Map();
  return { hidden: false, matches: false, addEventListener(k, fn) { if (!listeners.has(k)) listeners.set(k, new Set()); listeners.get(k).add(fn); }, removeEventListener(k, fn) { listeners.get(k)?.delete(fn); }, emit(k) { listeners.get(k)?.forEach(fn => fn()); }, get count() { return [...listeners.values()].reduce((sum, set) => sum + set.size, 0); } };
}
function playerFixture(options = {}) {
  let now = 0, next = 0;
  const jobs = new Map(), paints = [], changes = [];
  const add = (fn, ms) => { const id = ++next; jobs.set(id, { at: now + ms, fn }); return id; };
  const clock = { now: () => now, frame: fn => add(fn, 16), cancelFrame: id => jobs.delete(id), delay: add, cancelDelay: id => jobs.delete(id) };
  const media = eventTarget(), visibility = eventTarget();
  const player = context.window.PtoVisualization2MorphPlayer.create({ clock, media, visibility, draw: p => paints.push(p), onChange: s => changes.push(s), ...options });
  function advance(ms) {
    const end = now + ms;
    for (let guard = 0; guard < 10000; guard++) {
      const job = [...jobs.entries()].filter(([, j]) => j.at <= end).sort((a, b) => a[1].at - b[1].at)[0];
      if (!job) { now = end; return; }
      jobs.delete(job[0]); now = job[1].at; job[1].fn(now);
    }
    throw new Error('timer accumulation');
  }
  return { player, advance, paints, changes, jobs, media, visibility };
}
test('shared clock follows 1000/1000/1500/900 cadence and owns at most one pending job', () => {
  const f = playerFixture(); f.player.setActive(true);
  f.advance(999); assert.equal(f.player.progress, 0);
  f.advance(1020); assert.equal(f.player.progress, 1);
  f.advance(1488); assert.equal(f.player.progress, 1);
  f.advance(930); assert.equal(f.player.progress, 0);
  assert.equal(f.jobs.size, 1);
  f.player.pause(); f.advance(10000); assert.equal(f.jobs.size, 0); assert.equal(f.player.progress, 0);
});
test('mid-animation reversal and repeated replay cancel prior work without losing continuity', () => {
  const f = playerFixture(); f.player.setActive(true); f.player.goTo(1); f.advance(300);
  const p = f.player.progress; assert.ok(p > 0 && p < 1);
  f.player.goTo(0); assert.equal(f.player.progress, p); f.advance(720); assert.equal(f.player.progress, 0); assert.equal(f.jobs.size, 0);
  f.player.replay(); f.player.replay(); f.player.replay(); assert.equal(f.jobs.size, 1);
  f.advance(2100); assert.equal(f.player.progress, 1); assert.equal(f.jobs.size, 1);
});
test('Phase0 hides at a stable endpoint, returns paused, and visibility/reduced-motion/destroy release jobs', () => {
  const f = playerFixture({ pauseOnReturn: true }); f.player.setActive(true); f.player.goTo(1); f.advance(500);
  f.player.setActive(false); assert.equal(f.player.progress, 1); assert.equal(f.jobs.size, 0);
  f.player.setActive(true); f.advance(5000); assert.equal(f.player.progress, 1); assert.equal(f.jobs.size, 0);
  f.player.play(); f.advance(100); f.visibility.hidden = true; f.visibility.emit('visibilitychange'); assert.equal(f.jobs.size, 0);
  f.visibility.hidden = false; f.media.matches = true; f.media.emit('change'); f.player.goTo(0); assert.equal(f.player.progress, 0); f.player.goTo(1); assert.equal(f.player.progress, 1); assert.equal(f.jobs.size, 0);
  f.player.destroy(); const n = f.paints.length; f.advance(5000); f.player.goTo(0); assert.equal(f.paints.length, n); assert.equal(f.media.count + f.visibility.count, 0);
});
test('Attention retains its existing automatic-return policy and manual pause remains sticky', () => {
  const f = playerFixture(); f.player.setActive(true); f.advance(1400);
  assert.ok(f.player.progress > 0 && f.player.progress < 1);
  f.player.setActive(false); assert.equal(f.jobs.size, 0);
  f.player.setActive(true); assert.equal(f.jobs.size, 1); f.advance(2100); assert.equal(f.player.progress, 1);
  f.player.pause(); f.player.setActive(false); f.player.setActive(true); assert.equal(f.jobs.size, 0);
});
test('disabling loop finishes the current morph without scheduling another cycle', () => {
  const f = playerFixture(); f.player.setActive(true); f.player.play(); f.advance(300); f.player.setLoop(false);
  assert.equal(f.jobs.size, 1); f.advance(900); assert.equal(f.player.progress, 1); assert.equal(f.jobs.size, 0);
  f.player.replay(); f.advance(720); assert.equal(f.player.progress, 0); assert.equal(f.jobs.size, 0);
});
