const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function load() {
  const context = vm.createContext({ window: {} });
  const source = fs.readFileSync(path.join(__dirname, '../js/paged-attention-intro.js'), 'utf8');
  vm.runInContext(source, context, { filename: 'paged-attention-intro.js' });
  return context.window.PtoPagedAttentionIntro.model;
}

function plain(value) {
  return JSON.parse(JSON.stringify(value));
}

test('Paged Attention introduction exposes the documented model constants and default lengths', () => {
  const model = load();
  assert.deepEqual(plain(model.constants), {
    batch: 16,
    kvHeads: 8,
    qPerKv: 5,
    blocks: 24,
    pageTokens: 128,
    pagesPerStack: 4,
    maxSeq: 3584,
  });
  assert.deepEqual(plain(model.defaultSeqLens), [
    1025, 512, 2048, 128, 1536, 3584, 3072, 256,
    769, 1024, 2049, 640, 2560, 129, 1792, 3328,
  ]);
});

test('128 (batch, kv-head) tasks map uniquely to query-head groups and blocks', () => {
  const model = load();
  const seen = new Set();
  for (let id = 0; id < 128; id += 1) {
    const task = plain(model.taskInfo(id));
    assert.equal(task.id, id);
    assert.equal(task.batch, Math.floor(id / 8));
    assert.equal(task.kv, id % 8);
    assert.equal(task.qStart, task.kv * 5);
    assert.equal(task.qEnd - task.qStart + 1, 5);
    assert.equal(task.qEnd, task.qStart + 4);
    assert.ok(task.block >= 0 && task.block < 24);
    assert.equal(seen.has(task.id), false);
    seen.add(task.id);
  }
  assert.equal(seen.size, 128);
});

test('24 blocks allocate all 128 tasks once, including the first and last blocks', () => {
  const model = load();
  const all = [];
  for (let block = 0; block < 24; block += 1) {
    const tasks = plain(model.tasksForBlock(block));
    assert.ok(tasks.length > 0);
    assert.ok(tasks.every(id => model.taskInfo(id).block === block));
    all.push(...tasks);
  }
  assert.equal(all.length, 128);
  assert.equal(new Set(all).size, 128);
  assert.deepEqual([...all].sort((a, b) => a - b), Array.from({ length: 128 }, (_, i) => i));
  assert.deepEqual(plain(model.tasksForBlock(0)), [0, 24, 48, 72, 96, 120]);
  assert.deepEqual(plain(model.tasksForBlock(23)), [23, 47, 71, 95, 119]);
});

test('sequence boundaries produce correctly filled pages and stacks', () => {
  const model = load();
  for (const seqLen of [1, 128, 129, 512, 513, 1024, 1025, 3584]) {
    const result = plain(model.partition(seqLen));
    const pageCount = Math.ceil(seqLen / 128);
    const stackCount = Math.ceil(pageCount / 4);
    assert.equal(result.seqLen, seqLen);
    assert.equal(result.pageCount, pageCount);
    assert.equal(result.stackCount, stackCount);
    assert.equal(result.stacks.length, stackCount);

    let pageIndex = 0;
    for (let stackIndex = 0; stackIndex < result.stacks.length; stackIndex += 1) {
      const stack = result.stacks[stackIndex];
      assert.equal(stack.id, stackIndex);
      assert.ok(stack.pages.length >= 1 && stack.pages.length <= 4);
      if (stackIndex < stackCount - 1) assert.equal(stack.pages.length, 4);
      for (const page of stack.pages) {
        assert.equal(page.id, pageIndex);
        assert.equal(page.validTokens, Math.min(128, seqLen - pageIndex * 128));
        assert.ok(page.validTokens >= 1 && page.validTokens <= 128);
        pageIndex += 1;
      }
    }
    assert.equal(pageIndex, pageCount);
    assert.equal(result.stacks.flatMap(stack => stack.pages).reduce((sum, page) => sum + page.validTokens, 0), seqLen);
  }
});

test('default task 0 has nine pages, three stacks, and a one-page tail', () => {
  const model = load();
  const task0 = plain(model.partition(model.defaultSeqLens[0]));
  assert.equal(task0.seqLen, 1025);
  assert.equal(task0.pageCount, 9);
  assert.equal(task0.stackCount, 3);
  assert.deepEqual(task0.stacks.map(stack => stack.pages.length), [4, 4, 1]);
  assert.equal(task0.stacks[2].pages[0].validTokens, 1);
  assert.ok(task0.stacks.every(stack => stack.pages.length <= 4));
});
