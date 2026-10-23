const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const context = vm.createContext({ window: {} });
vm.runInContext(fs.readFileSync(path.join(__dirname, '../js/visualization2-transition.js'), 'utf8'), context);
const { focusBlock, pairs } = context.window.PtoVisualization2Transition;

test('focus prefers a selected fully visible Block with Tasks', () => {
  const snapshot = {
    clip: { x: 0, y: 0, width: 300, height: 100 },
    blocks: [0, 1, 2, 3].map(id => ({ id, rect: { x: 0, y: id * 30, width: 100, height: 30 } })),
    tasks: [{ id: 0, block: 0 }, { id: 24, block: 0 }, { id: 1, block: 1 }, { id: 2, block: 2 }],
  };
  assert.equal(focusBlock(snapshot, 2), 2);
  assert.equal(focusBlock(snapshot, 3), 0);
  snapshot.clip.y = 30;
  assert.equal(focusBlock(snapshot, null), 1);
  snapshot.clip.height = 10;
  assert.equal(focusBlock(snapshot, null), null);
});

test('pairing preserves real Task identities across different Block sizes', () => {
  const assignment = { blocks: [{ id: 0 }, { id: 1 }], tasks: [{ id: 0 }, { id: 24 }, { id: 48 }, { id: 1 }] };
  const pipeline = { blocks: [{ id: 0 }, { id: 1 }], tasks: [{ id: 0 }, { id: 24 }, { id: 48 }, { id: 1 }] };
  const mapping = pairs(assignment, pipeline);
  assert.deepEqual(Array.from(mapping.tasks, pair => pair.source.id), [0, 24, 48, 1]);
  assert.equal(new Set(mapping.tasks.map(pair => pair.target.id)).size, 4);
});
