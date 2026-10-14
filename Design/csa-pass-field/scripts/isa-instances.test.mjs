import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { loadInstances, groupLoadInstances } from '../src/isa-instances.js';

const isa = JSON.parse(fs.readFileSync(new URL('../data/csa-qk-pv/isa.json', import.meta.url)));
const instruction = isa.instructions.find((item) => item.mlirName === 'pto.tload');
const instances = loadInstances(instruction);
test('TLOAD expansion preserves every source identity exactly once', () => {
  assert.equal(instances.length, 12);
  assert.equal(new Set(instances.map((item) => item.id)).size, 12);
  assert.deepEqual(instances.map((item) => item.line), instruction.sourceRefs.map((item) => item.line));
});
test('kernel and output Tile types form 6/3/2/1 groups', () => {
  const groups = groupLoadInstances(instances);
  assert.deepEqual(groups.map((item) => item.kernel), ['qk_pv_aic', 'qk_pv_aiv']);
  assert.deepEqual(groups.flatMap((item) => item.groups.map((group) => group.instances.length)), [6, 3, 2, 1]);
  assert.deepEqual(instances.find((item) => item.line === 217).inputShape, ['1', '512']);
  assert.deepEqual(instances.find((item) => item.line === 217).shape, ['16', '512']);
  assert.equal(instances.at(-1).dtype, 'FP32');
});
