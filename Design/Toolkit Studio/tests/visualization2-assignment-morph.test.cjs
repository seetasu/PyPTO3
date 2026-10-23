const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const context = vm.createContext({ window: {} });
vm.runInContext(fs.readFileSync(path.join(__dirname, '../js/visualization2-assignment-morph.js'), 'utf8'), context);
const createLayout = context.window.PtoVisualization2AssignmentMorph.createLayout;
vm.runInContext(fs.readFileSync(path.join(__dirname, '../js/paged-attention-pypto-source.js'), 'utf8'), context);
const source = context.window.PTO_PAGED_ATTENTION_PYPTO_SOURCE;
const number = name => Number(source.match(new RegExp(`^${name} = (\\d+)`, 'm'))[1]);
const config = { batch: number('BATCH'), kvHeads: number('NUM_KV_HEADS'), blocks: number('ATTN_SPMD_BLOCKS') };
const layout = createLayout(config);
const { partitionWidth, sequenceSegments } = context.window.PtoVisualization2AssignmentMorph;

test('Q Head partitions equally span the available Task width at different sizes', () => {
  for (const width of [62, layout.sources[0].w - 4, 180]) {
    const partitions = partitionWidth(width, 5);
    assert.equal(partitions.length, 5);
    assert.equal(new Set(partitions.map(p => p.width)).size, 1);
    assert.equal(partitions[0].x, 0);
    assert.ok(Math.abs(partitions.at(-1).x + partitions.at(-1).width - width) < 1e-9);
    partitions.slice(1).forEach((p, i) => assert.ok(p.x > partitions[i].x + partitions[i].width));
  }
});

test('Batch strips and Task interiors use identical 512-token fractions on a shared seven-segment scale', () => {
  for (const [len, full, partial] of [[512, 1, 0], [1025, 2, 1 / 512], [2048, 4, 0], [3584, 7, 0]]) {
    const strip = sequenceSegments(len, 68, 7), task = sequenceSegments(len, layout.sources[0].w - 4, 7);
    assert.equal(task.length, 7);
    assert.equal(task.filter(s => s.fraction === 1).length, full);
    assert.equal(task[full]?.fraction || 0, partial);
    assert.deepEqual(Array.from(strip, s => s.fraction), Array.from(task, s => s.fraction));
    assert.equal(new Set(task.map(s => s.width)).size, 1);
    assert.ok(Math.abs(task.reduce((sum, s) => sum + s.fraction * 512, 0) - len) < 1e-9);
    task.forEach(s => assert.ok(s.fillWidth >= 0 && s.fillWidth <= s.width));
  }
});

test('source is 16 Batch rows by 8 persistent KV columns, with no X or size change at the destination', () => {
  assert.equal(layout.sources.length, 128);
  assert.equal(layout.columns.length, 8);
  assert.equal(new Set(layout.sources.map(r => r.y)).size, 16);
  assert.equal(new Set(layout.sources.map(r => r.x)).size, 8);
  layout.sources.forEach((from, id) => {
    const to = layout.targets[id];
    assert.equal(from.x, layout.columns[id % config.kvHeads].x + 7);
    assert.equal(from.x, to.x, `Task ${id} stays in its KV column`);
    assert.equal(from.w, to.w);
    assert.equal(from.h, to.h);
    assert.ok(from.x > 0 && from.x + from.w < layout.width);
    assert.ok(from.y > 44 && from.y + from.h < layout.height);
    for (let step = 0; step <= 100; step++) {
      const p = step / 100, y = from.y + (to.y - from.y) * p;
      assert.ok(y >= Math.min(from.y, to.y) && y <= Math.max(from.y, to.y) + 1e-9);
    }
  });
});

test('24 thin Block frames contain every Task once, ordered by Batch residue and quotient', () => {
  assert.equal(layout.blockRects.length, 24);
  assert.equal(layout.groups.length, 3);
  assert.equal(new Set(layout.assignments.flat()).size, 128);
  assert.equal(layout.assignments.flat().length, 128);
  assert.deepEqual(Array.from(layout.assignments[3]), [3, 27, 51, 75, 99, 123]);
  assert.deepEqual(Array.from(layout.assignments[8]), [8, 32, 56, 80, 104]);
  layout.assignments.forEach((members, block) => {
    assert.ok(members.length === 5 || members.length === 6);
    const frame = layout.blockRects[block];
    members.forEach((id, index) => {
      const r = layout.targets[id], batch = Math.floor(id / config.kvHeads);
      assert.equal(id % config.blocks, block);
      assert.equal(batch % 3, Math.floor(block / config.kvHeads));
      assert.equal(Math.floor(batch / 3), index);
      assert.ok(r.x > frame.x && r.x + r.w < frame.x + frame.w);
      assert.ok(r.y >= frame.y + 24 && r.y + r.h < frame.y + frame.h);
      if (index) assert.ok(r.y > layout.targets[members[index - 1]].y + r.h);
    });
  });
});

test('each endpoint has no overlapping Tasks or Block frames and reverse restores every Batch row', () => {
  for (const rects of [layout.sources, layout.targets, layout.blockRects]) {
    rects.forEach((a, i) => rects.slice(i + 1).forEach(b => {
      const overlaps = a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
      assert.equal(overlaps, false);
    }));
  }
  layout.sources.forEach((from, id) => {
    const to = layout.targets[id];
    assert.ok(Math.abs(to.y + (from.y - to.y) - from.y) < 1e-9);
  });
});

test('layout derives Batch-residue groups from configuration and preserves modulo ownership in fallback', () => {
  for (const config of [{ batch: 10, kvHeads: 4, blocks: 8 }, { batch: 7, kvHeads: 4, blocks: 6 }]) {
    const l = createLayout(config);
    assert.equal(l.sources.length, config.batch * config.kvHeads);
    assert.equal(l.blockRects.length, config.blocks);
    l.assignments.forEach((members, block) => members.forEach(id => {
      const from = l.sources[id], to = l.targets[id], frame = l.blockRects[block];
      assert.equal(id % config.blocks, block);
      assert.equal(from.x, to.x);
      assert.ok(to.x >= frame.x && to.x + to.w <= frame.x + frame.w);
      assert.ok(to.y >= frame.y && to.y + to.h <= frame.y + frame.h);
    }));
  }
});
