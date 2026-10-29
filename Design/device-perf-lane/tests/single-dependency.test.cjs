const { test } = require('node:test');
const assert = require('node:assert/strict');
const { findChains, focusRange, blockPhases } = require('../single-dependency.js');

test('joins maximal chains and deduplicates dependency records', () => {
  const edges = [['a','b'], ['a','b'], ['b','c'], ['c','d'], ['c','e'], ['f','g'], ['h','g']]
    .map(([pred, succ]) => ({ pred, succ }));
  assert.deepEqual(findChains(edges, ['a','b','c','d','e','f','g','h']), [['a','b','c']]);
});

test('includes chain endpoints with other incoming or outgoing edges', () => {
  const edges = [['x','a'], ['y','a'], ['a','b'], ['b','c'], ['c','d'], ['c','e']]
    .map(([pred, succ]) => ({ pred, succ }));
  assert.deepEqual(findChains(edges, ['x','y','a','b','c','d','e']), [['a','b','c']]);
});

test('handles empty input, self loops and cycles without inventing a chain', () => {
  assert.deepEqual(findChains([]), []);
  assert.deepEqual(findChains([{pred: 1, succ: 1}, {pred: 2, succ: 3}, {pred: 3, succ: 2}], [1,2,3]), []);
});

test('excludes dependencies involving tasks outside Worker View', () => {
  const edges = [['a','b'], ['a','alloc'], ['dummy','b']].map(([pred, succ]) => ({pred, succ}));
  assert.deepEqual(findChains(edges, ['a','b']), [['a','b']]);
});

test('focuses on the full chain with context and clamps to the run window', () => {
  const tasks = [{taskId:'a',startUs:100,endUs:120},{taskId:'b',startUs:130,endUs:150},{taskId:'other',startUs:0,endUs:900}];
  assert.deepEqual(focusRange(['a','b'], tasks, 1000), [94,156]);
  assert.deepEqual(focusRange(['a'], [{taskId:'a',startUs:0,endUs:20}], 1000), [0,24.8]);
  assert.equal(focusRange(['missing'], tasks, 1000), null);
});

test('limits zoom and includes every launch for the same task', () => {
  const tasks = [{taskId:'a',startUs:100,endUs:101},{taskId:'a',startUs:120,endUs:121}];
  const range = focusRange(['a'], tasks, 10000);
  assert.ok(range[0] <= 100 && range[1] >= 121);
  assert.equal(range[1] - range[0], 10000/256);
});

test('places setup and computation on the time axis and clips to the visible range', () => {
  const block = {startUs:100, endUs:200, setupUs:70, computeUs:20};
  assert.deepEqual(blockPhases(block, 120, 190), [
    {kind:'setup',start:120,end:170}, {kind:'other',start:170,end:180}, {kind:'compute',start:180,end:190},
  ]);
  assert.deepEqual(blockPhases(block, 200, 250), []);
});

test('keeps phases disjoint when rounded timing fields overlap', () => {
  const block = {startUs:100, endUs:110, setupUs:9, computeUs:2};
  assert.deepEqual(blockPhases(block, 0, 200), [
    {kind:'setup',start:100,end:108}, {kind:'compute',start:108,end:110},
  ]);
  assert.deepEqual(blockPhases({...block, computeUs:NaN}, 0, 200), []);
});

test('matches the eight chains in the bundled trace', () => {
  const fs = require('node:fs');
  const html = fs.readFileSync(require('node:path').join(__dirname, '../index-focus.html'), 'utf8');
  const data = JSON.parse(html.match(/<script type="application\/json" id="lane-data">([\s\S]*?)<\/script>/)[1]);
  const expected = [
    ['hc_pre_linear', 'hc_pre_linear_reduce'], ['mix_x', 'rms_norm'],
    ['rmsnorm_rope', 'kv_hadamard', 'kv_and_cache_write'],
    ['qr_rope', 'qr_hadamard_matmul', 'qr_hadamard_quant'],
    ['weights_proj', 'weights_proj_reduce'], ['qr_proj_matmul', 'qr_rms_norm_quant'],
    ['kv_rms_norm_rope', 'csa_cache_writeback', 'kv_touch'],
    ['indexer_score_leaf_wave_aic + indexer_score_leaf_wave_aiv', 'indexer_topk_group_wave', 'indexer_topk_query_merge', 'csa_slots_build_valid_qk_plan'],
  ];
  for (const rank of data.ranks) {
    const chains = findChains(rank.edges, rank.launches.map(task => task.taskId));
    assert.deepEqual(chains.map(chain => chain.map(id => rank.taskNames[id])), expected);
  }
});
