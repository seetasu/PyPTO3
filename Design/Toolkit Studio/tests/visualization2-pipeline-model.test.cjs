const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function load() {
  const context = vm.createContext({ window: {} });
  for (const name of [
    'visualization2-task-encoding',
    'visualization2-anatomy',
    'visualization2-pipeline-intra',
  ]) {
    vm.runInContext(
      fs.readFileSync(path.join(__dirname, '..', 'js', `${name}.js`), 'utf8'),
      context,
      { filename: name },
    );
  }
  return context.window;
}

function makeModel(w, seqLens, { transferSlots = 2, preLaunch = 2 } = {}) {
  const anatomy = w.PtoVisualization2Anatomy.createModel({
    seqLens,
    blocks: seqLens.length,
    numHeads: 4,
    kvHeads: 2,
    headDim: 8,
    rowTile: 16,
    pageTokens: 8,
    pagesPerStack: 2,
    transferSlots,
  });
  return { anatomy, model: w.PtoVisualization2PipelineIntra.createModel({ anatomy, preLaunch }) };
}

const transfers = [
  { buffer: 'score_transfer', producer: 'qk', consumer: 'softmax' },
  { buffer: 'probability_transfer', producer: 'softmax', consumer: 'pv' },
  { buffer: 'pv_transfer', producer: 'pv', consumer: 'accumulate' },
];

function assertTaskInvariants(model, task, preLaunch) {
  const operationsByStack = new Map();
  for (const operation of task.operations) {
    const operations = operationsByStack.get(operation.stack) || [];
    operations.push(operation);
    operationsByStack.set(operation.stack, operations);
    const producer = ['qk', 'softmax'].includes(operation.stage);
    assert.equal(operation.tick, operation.stack + (producer ? 0 : preLaunch));
    assert.equal(operation.slot, task.stacks[operation.stack].slot);
  }
  assert.equal(task.operations.length, task.stacks.length * 4);
  for (const stack of task.stacks) {
    assert.equal(stack.produceTick, stack.index);
    assert.equal(stack.consumeTick, stack.index + preLaunch);
    assert.equal(operationsByStack.get(stack.index).length, 4);
  }

  assert.equal(task.dependencies.length, task.stacks.length * transfers.length);
  const dependencyKeys = new Set();
  for (const dependency of task.dependencies) {
    const key = `${dependency.stack}:${dependency.buffer}`;
    assert.equal(dependencyKeys.has(key), false, `duplicate dependency ${key}`);
    dependencyKeys.add(key);
    const expected = transfers.find(item => item.buffer === dependency.buffer);
    assert.ok(expected, `unknown dependency buffer: ${dependency.buffer}`);
    assert.equal(dependency.producer, expected.producer);
    assert.equal(dependency.consumer, expected.consumer);
    assert.ok(task.stacks[dependency.stack], `missing dependency stack ${dependency.stack}`);
    assert.equal(dependency.slot, task.stacks[dependency.stack].slot);
  }

  const expectedReuseCount = Math.max(0, task.stacks.length - model.transferSlots) * transfers.length;
  assert.equal(task.reuse.length, expectedReuseCount);
  for (const reuse of task.reuse) {
    const expected = transfers.find(item => item.buffer === reuse.buffer);
    assert.ok(expected, `unknown reuse buffer: ${reuse.buffer}`);
    assert.equal(reuse.producer, expected.producer);
    assert.equal(reuse.consumer, expected.consumer);
    assert.equal(reuse.to - reuse.from, model.transferSlots);
    assert.equal(reuse.slot, task.stacks[reuse.from].slot);
    assert.equal(reuse.slot, task.stacks[reuse.to].slot);
  }

  const expectedBase = task.block * model.transferSlots * model.rowTile;
  assert.equal(task.transferBase, expectedBase);
  for (const stack of task.stacks) {
    const expectedRow = expectedBase + stack.slot * model.rowTile;
    assert.equal(stack.produceRow, expectedRow);
    assert.equal(stack.consumeRow, expectedRow);
  }
}

test('pipeline model preserves offsets, dependencies, reuse slots and output across task sizes', () => {
  const w = load();
  const preLaunch = 3;
  const { anatomy, model } = makeModel(w, [8, 40], { transferSlots: 2, preLaunch });
  const noOffset = w.PtoVisualization2PipelineIntra.createModel({ anatomy, preLaunch: 0 });

  assert.equal(model.tasks.length, 4, 'multiple batches and KV heads produce separate tasks');
  assert.deepEqual(
    model.tasks.map(task => task.stacks.length),
    [1, 1, 3, 3],
    'fixture covers a single Stack and multiple Stacks in multiple Tasks',
  );

  for (let index = 0; index < model.tasks.length; index++) {
    const task = model.tasks[index];
    const anatomyTask = anatomy.tasks[index];
    assertTaskInvariants(model, task, preLaunch);
    assert.deepEqual(task.output, anatomyTask.output, 'pipeline projection keeps output metadata');
    assert.deepEqual(task.output, noOffset.tasks[index].output, 'output metadata is independent of tick offset');
  }
});
