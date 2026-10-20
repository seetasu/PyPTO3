const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const read = (name) => fs.readFileSync(path.join(__dirname, '../js', name + '.js'), 'utf8');
function fixture() {
  const vmContext = vm.createContext({ window: {}, console });
  for (const name of ['paged-attention-pypto-source', 'semantic-computation-graph', 'semantic-computation-presentation', 'semantic-computation-layout', 'paged-attention-semantic-graph', 'paged-attention-intro', 'paged-attention-execution']) {
    const source = read(name).replace('global.PtoPagedAttentionContext = context;', 'global.PtoPagedAttentionContext = context; global.fixtureState = state;');
    vm.runInContext(source, vmContext);
  }
  const w = vmContext.window;
  const model = w.PTO_PYPA_SEMANTIC_GRAPH;
  return { w, model, derive: (seqLen, task = 0) => w.PtoPagedAttentionExecution.derive(model, w.PtoPagedAttentionIntro.model, { task, seqLen }) };
}
test('counts match explicit page, stack, tail and PV API scenarios', () => {
  const { derive } = fixture();
  // seqLen, Pages, Stacks, tail Pages, last valid tokens, matmul_acc
  for (const [len, pages, stacks, tail, valid, acc] of [
    [1,1,1,1,1,0], [128,1,1,1,128,0], [129,2,1,2,1,1], [384,3,1,3,128,2],
    [512,4,1,4,128,3], [513,5,2,1,1,3], [1025,9,3,1,1,6], [3584,28,7,4,128,21],
  ]) {
    const d = derive(len);
    assert.deepEqual([d.pages,d.stacks,d.ticks,d.pvMatmul,d.pvMatmulAcc], [pages,stacks,stacks+2,stacks,acc]);
    assert.equal(d.stacksDetail.at(-1).pages.length, tail);
    assert.equal(d.stacksDetail.at(-1).branch, tail === 4 ? '完整 Stack 分支' : `tail_pages == ${tail}`);
    assert.equal(d.stacksDetail.at(-1).pages.at(-1).validTokens, valid);
  }
  for (const len of [null,0,-1,3585,1.5,NaN,Infinity,'512']) assert.equal(derive(len), null);
  for (const task of [null,-1,128,0.5]) assert.equal(derive(512,task), null);
});
test('only grounded nodes/data edges are weighted and expansion preserves identities and geometry', () => {
  const { w, model, derive } = fixture(), original = JSON.stringify(model), d = derive(1025);
  const all = w.PTO_COMPUTATION_PRESENTATION.flatten(model);
  const expanded = new Set(all.filter(({node}) => node.type === 'composite').map(({id}) => id));
  const graph = w.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize(model, expanded);
  const before = JSON.stringify(graph);
  assert.deepEqual(Object.keys(d.nodes).sort(), ['qk','pv','scale-mask','softmax','context-acc','output-norm','qk/matmul','pv/matmul'].sort());
  for (const [id, entry] of Object.entries(d.nodes)) {
    assert.ok(all.some((node) => node.id === id), id);
    for (const [start,end] of entry.source.ranges) assert.ok(start > 0 && end <= model.sourceFiles['paged_attention_pypto.py'].lineCount);
  }
  for (const [id, entry] of Object.entries(d.edges)) {
    assert.equal(graph.edges.find((edge) => edge.id === id)?.type, 'data', id);
    assert.equal(entry.width, 1.5 + 0.7 * (entry.count - 1));
  }
  assert.equal(d.edges['qk>scores:Scores'].count, 9);
  assert.equal(d.edges['scores>scale-mask:Scores'].count, 3);
  assert.equal(d.edges['output-norm>output:BF16 Output'].count, 1);
  assert.equal(JSON.stringify(model), original);
  assert.equal(JSON.stringify(w.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize(model, expanded)), before);
});
test('scene updates validate inputs, reconcile selection, share Batch lengths and release subscribers', () => {
  const { w } = fixture(), context = w.PtoPagedAttentionContext, state = w.fixtureState;
  let received = 0;
  const abort = new AbortController();
  context.subscribe(() => received++, {signal:abort.signal});
  assert.equal(context.setLength(512), false, 'no selected Task');
  state.task = 0; state.pipelineStack = 2; state.stage = 'pv';
  assert.equal(context.setLength(129), true);
  assert.equal(context.selection.stack, 0);
  assert.equal(context.selection.pipelineStep, 2);
  assert.equal(received, 1);
  assert.equal(context.setLength(129), true);
  assert.equal(received, 1, 'unchanged input emits nothing');
  for (const invalid of [0,3585,2.5,NaN]) assert.equal(context.setLength(invalid), false);
  assert.equal(received, 1);
  state.task = 7;
  assert.equal(context.selection.seqLen,129, 'other KV head of same Batch');
  state.task = 8;
  assert.equal(context.selection.seqLen,512, 'different Batch keeps its own length');
  const snapshot = context.scenario;
  snapshot.seqLens[0] = 3000;
  assert.equal(context.scenario.seqLens[0],129, 'snapshot does not expose mutable state');
  abort.abort();
  context.setLength(513);
  assert.equal(received,1);
  for (let i=0;i<5;i++) { const unsubscribe = context.subscribe(() => received++); unsubscribe(); }
  context.setLength(1025);
  assert.equal(received,1,'remount cleanup leaves no duplicate listeners');
});
test('execution labels use semantic constants matching the Task partition model', () => {
  const { w, model } = fixture(), c = w.PtoPagedAttentionIntro.model.constants;
  for (const [key, val] of [['BLOCK_SIZE',c.pageTokens],['STACK_PAGES',c.pagesPerStack],['PRE_LAUNCH',c.preLaunch],['SEQ_LEN',c.maxSeq]]) assert.equal(model.constants[key].value,val);
  const source = w.PTO_PAGED_ATTENTION_PYPTO_SOURCE.split('\n');
  assert.match(source[264], /page_count = .*BLOCK_SIZE/);
  assert.match(source[265], /stack_count = .*STACK_TOKENS/);
  assert.match(source[278], /pl.range\(stack_count \+ PRE_LAUNCH\)/);
  const pv = source.slice(368,718).join('\n');
  assert.equal((pv.match(/= pl.matmul\(/g) || []).length,4,'full, one-, two-, three-Page branch initializers');
  assert.equal((pv.match(/= pl.matmul_acc\(/g) || []).length,6,'3 + 0 + 1 + 2 accumulation sites');
});
