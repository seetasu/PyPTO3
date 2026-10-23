const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function load() {
  const context = vm.createContext({ window: {} });
  const source = fs.readFileSync(path.join(__dirname, '../js/matmul-intro.js'), 'utf8');
  vm.runInContext(source, context, { filename: 'matmul-intro.js' });
  return context.window.PtoMatmulIntro.model;
}

function plain(value) {
  return JSON.parse(JSON.stringify(value));
}

test('Matmul introduction exposes the source constants and derived tile counts', () => {
  const model = load();
  assert.deepEqual(plain(model.constants), { M: 256, N: 256, K: 256, mTile: 64, nTile: 64 });
  assert.equal(model.mTiles, 4);
  assert.equal(model.nTiles, 4);
  assert.equal(model.tileCount, 16);
  assert.equal(model.macsPerTile, 64 * 64 * 256);
});

test('16 tiles map uniquely to full M / N bands without overlap', () => {
  const model = load();
  const seen = new Set();
  for (let id = 0; id < model.tileCount; id += 1) {
    const tile = plain(model.tileInfo(id));
    assert.equal(tile.id, id);
    assert.equal(tile.mb, Math.floor(id / 4));
    assert.equal(tile.nb, id % 4);
    assert.equal(tile.m0, tile.mb * 64);
    assert.equal(tile.n0, tile.nb * 64);
    assert.ok(tile.m0 >= 0 && tile.m0 + 64 <= model.constants.M);
    assert.ok(tile.n0 >= 0 && tile.n0 + 64 <= model.constants.N);
    assert.deepEqual(tile.aShape, [64, 256]);
    assert.deepEqual(tile.bShape, [256, 64]);
    assert.deepEqual(tile.outShape, [64, 64]);
    assert.equal(tile.macs, model.macsPerTile);
    assert.equal(seen.has(tile.id), false);
    seen.add(tile.id);
  }
  assert.equal(seen.size, 16);
});

test('tile grid covers every M band exactly nTiles times', () => {
  const model = load();
  const byBand = new Map();
  for (let id = 0; id < model.tileCount; id += 1) {
    const { mb, nb } = plain(model.tileInfo(id));
    byBand.set(mb, (byBand.get(mb) || 0) + 1);
    assert.equal(id, mb * model.nTiles + nb);
  }
  assert.equal(byBand.size, model.mTiles);
  for (const count of byBand.values()) assert.equal(count, model.nTiles);
});

test('total MAC work equals the untiled M×N×K contraction', () => {
  const model = load();
  const total = model.macsPerTile * model.tileCount;
  assert.equal(total, model.constants.M * model.constants.N * model.constants.K);
});
