/* Tensor / Tile 元数据校验：常量表与源码互校、shape 期望表、维度可追溯性、
 * allocated / valid 不混淆、运行时 shape 不写死、provenance 依据正确、
 * 以及节点清单、端口与源码映射不回归。 */
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const js = path.join(__dirname, '..', 'js');
function load() {
  const context = vm.createContext({ window: {} });
  for (const name of ['paged-attention-pypto-source', 'semantic-computation-graph', 'semantic-computation-presentation', 'semantic-computation-layout', 'paged-attention-semantic-graph']) {
    vm.runInContext(fs.readFileSync(path.join(js, name + '.js'), 'utf8'), context, { filename: name + '.js' });
  }
  return context.window;
}
function allNodes(graph, prefix = '') {
  return graph.nodes.flatMap((node) => [{ node, id: prefix + node.id }, ...(node.composite ? allNodes(node.composite.subgraph, prefix + node.id + '/') : [])]);
}
const find = (model, id) => allNodes(model).find((item) => item.id === id).node;
const payload = (node) => node[node.type];
/* 语义图在 vm 里执行，跨 realm 的数组 / 对象需要先转成宿主 realm 的普通值再比较。 */
const plain = (value) => JSON.parse(JSON.stringify(value));
const sourceText = (w, node) => {
  const lines = w.PTO_PAGED_ATTENTION_PYPTO_SOURCE.trimEnd().split('\n');
  return node.source.ranges.map(({ start, end }) => lines.slice(start - 1, end).join('\n')).join('\n');
};
const kindsCovering = (node, key) => node.provenance
  .filter((entry) => entry.fields.some((field) => field === `${node.type}.${key}` || field.startsWith(`${node.type}.${key}.`)))
  .map((entry) => entry.kind);

/* 源码第 17–50 行的常量按 Python 语法书写，// 为整除：这里逐行取表达式求值，
   // 归一化为整除后再与模型常量表互校，避免常量表与源码漂移。 */
function evaluateSourceConstants(text) {
  const values = {}, external = {};
  for (const raw of text.split('\n')) {
    const body = raw.split('#')[0].trim();
    const match = /^([A-Z][A-Z0-9_]*) = (.+)$/.exec(body);
    if (!match) continue;
    const [, name, expression] = match;
    if (expression.includes('M.')) { external[name] = expression; continue; }
    const integer = expression.includes('//');
    const normalized = integer ? expression.replace(/\/\//g, '/') : expression;
    const names = Object.keys(values);
    const compute = new Function('math', ...names, `return ${integer ? 'Math.floor' : ''}(${normalized});`);
    values[name] = compute(Math, ...names.map((key) => values[key]));
  }
  return { values, external };
}
const resolveDims = (model, dims) => dims.map((dim) => (typeof dim === 'string' && model.constants[dim] ? model.constants[dim].value : dim));

/* 关键 Tensor / Tile 的期望元数据：数值维度写成解析后的值，
   运行时维度保留符号名；allocated 省略 valid 表示源码未声明 valid。 */
const NODE_METADATA = [
  { id: 'q-proj', granularity: 'tensor', storage: 'GM', absent: ['shape', 'dtype'] },
  { id: 'k-proj', granularity: 'tensor', storage: 'GM', absent: ['shape', 'dtype'] },
  { id: 'v-proj', granularity: 'tensor', storage: 'GM', absent: ['shape', 'dtype'] },
  { id: 'q', granularity: 'tensor', storage: 'GM', dtype: 'BF16', allocated: ['query_rows', 128], valid: [5, 128] },
  { id: 'k', granularity: 'token', dtype: 'BF16', allocated: [1, 128], absent: ['storage'] },
  { id: 'v', granularity: 'token', dtype: 'BF16', allocated: [1, 128], absent: ['storage'] },
  { id: 'aux', granularity: 'bundle', storage: 'GM', absent: ['shape', 'dtype'] },
  { id: 'k-cache', granularity: 'tensor', storage: 'GM', allocated: ['cache_token_rows', 1024], absent: ['dtype'] },
  { id: 'v-cache', granularity: 'tensor', storage: 'GM', dtype: 'BF16', allocated: ['cache_token_rows', 1024] },
  { id: 'cache-append/slot', granularity: 'scalar', dtype: 'INDEX', absent: ['shape', 'storage'] },
  { id: 'cache-append/key', granularity: 'tensor', storage: 'GM', allocated: ['cache_token_rows', 1024], absent: ['dtype'] },
  { id: 'cache-append/value', granularity: 'tensor', storage: 'GM', dtype: 'BF16', allocated: ['cache_token_rows', 1024] },
  { id: 'qk/q-tile', granularity: 'tile', storage: 'Mat', allocated: [16, 128], valid: [5, 128], absent: ['dtype'] },
  { id: 'qk/k-page', granularity: 'page', storage: 'Mat', allocated: [128, 128], absent: ['dtype', 'valid'] },
  { id: 'qk/score', granularity: 'page', dtype: 'FP32', allocated: [16, 128], absent: ['storage', 'valid'] },
  { id: 'scores', granularity: 'stack', storage: 'GM', dtype: 'FP32', allocated: [1152, 512], absent: ['valid'] },
  { id: 'softmax/scores', granularity: 'stack', storage: 'Vec', allocated: [8, 512], valid: ['lane_rows', 512], absent: ['dtype'] },
  { id: 'softmax/previous', granularity: 'stack', dtype: 'FP32', absent: ['shape'] },
  { id: 'softmax/probability', granularity: 'stack', dtype: 'BF16', absent: ['shape', 'storage'] },
  { id: 'softmax/updated', granularity: 'stack', dtype: 'FP32', absent: ['shape'] },
  { id: 'probability', granularity: 'stack', storage: 'GM', dtype: 'BF16', allocated: [1152, 512], absent: ['valid'] },
  { id: 'pv/probability', granularity: 'page', storage: 'Mat', dtype: 'BF16', allocated: [16, 128], valid: [5, 128] },
  { id: 'pv/v-page', granularity: 'page', storage: 'Mat', dtype: 'BF16', allocated: [128, 128], absent: ['valid'] },
  { id: 'pv/partial', granularity: 'stack', storage: 'GM', dtype: 'FP32', allocated: [16, 128], absent: ['valid'] },
  { id: 'context-acc/partial', granularity: 'stack', storage: 'Vec', dtype: 'FP32', allocated: [8, 128], valid: ['lane_rows', 128] },
  { id: 'context-acc/pending', granularity: 'stack', absent: ['shape', 'dtype', 'storage'] },
  { id: 'context-acc/old', granularity: 'stack', dtype: 'FP32', allocated: [8, 128], valid: ['lane_rows', 128], absent: ['storage'] },
  { id: 'context', granularity: 'stack', dtype: 'FP32', absent: ['shape'] },
  { id: 'output', granularity: 'tensor', storage: 'GM', dtype: 'BF16', allocated: ['query_rows', 128], valid: ['lane_rows', 128] },
];
const DATA_STATE_IDS = [
  'q-proj', 'k-proj', 'v-proj', 'q', 'k', 'v', 'aux',
  'cache-append/slot', 'cache-append/key', 'cache-append/value',
  'k-cache', 'v-cache',
  'qk/q-tile', 'qk/k-page', 'qk/score',
  'scores',
  'softmax/scores', 'softmax/previous', 'softmax/probability', 'softmax/updated',
  'probability',
  'pv/probability', 'pv/v-page', 'pv/partial',
  'context-acc/partial', 'context-acc/pending', 'context-acc/old',
  'context', 'output',
];
const EMPTY_SUBGRAPHS = ['q-prep', 'k-prep', 'v-scale', 'scale-mask', 'output-norm', 'q-prep/norm'];
/* 形状直接引自 load / reshape / full / set_validshape 参数的节点。 */
const SHAPE_QUOTED = ['qk/q-tile', 'qk/k-page', 'pv/v-page', 'pv/probability', 'softmax/scores', 'context-acc/partial', 'context-acc/old', 'k-cache', 'v-cache', 'cache-append/key', 'cache-append/value'];
/* 形状由 matmul 操作数、写入地址或切片行数推导的节点。 */
const SHAPE_DERIVED = ['q', 'k', 'v', 'qk/score', 'scores', 'probability', 'pv/partial', 'output'];
/* 维度名必须字面出现在被引用源码范围内的节点（k / v 的行数取自 1 行切片表达式）。 */
const SHAPE_IN_RANGE = [...SHAPE_QUOTED, 'k', 'v'];
/* storage 由 target_memory 直接给出 / 由外部 pl.Tensor 位于 GM 推导的节点。 */
const STORAGE_QUOTED = ['qk/q-tile', 'qk/k-page', 'pv/probability', 'pv/v-page', 'softmax/scores', 'context-acc/partial'];
const STORAGE_DERIVED = ['q-proj', 'k-proj', 'v-proj', 'q', 'aux', 'k-cache', 'v-cache', 'cache-append/key', 'cache-append/value', 'scores', 'probability', 'pv/partial', 'output'];
const DTYPE_DERIVED = ['v-cache', 'cache-append/value', 'context-acc/partial'];

test('constants table is a faithful static evaluation of the cited source lines', () => {
  const w = load(), model = w.PTO_PYPA_SEMANTIC_GRAPH, lines = w.PTO_PAGED_ATTENTION_PYPTO_SOURCE.split('\n');
  const { values, external } = evaluateSourceConstants(w.PTO_PAGED_ATTENTION_PYPTO_SOURCE);
  assert.deepEqual(Object.keys(model.constants).sort(), Object.keys(values).sort());
  for (const [name, entry] of Object.entries(model.constants)) {
    assert.equal(typeof entry.value, 'number', name);
    assert.ok(Math.abs(entry.value - values[name]) < 1e-12, `${name}: ${entry.value} != ${values[name]}`);
    assert.match(lines[entry.line - 1], new RegExp(`^${name} = `), name);
  }
  assert.deepEqual(Object.keys(model.externalConstants).sort(), Object.keys(external).sort());
  for (const [name, entry] of Object.entries(model.externalConstants)) {
    assert.equal(entry.expression, external[name], name);
    assert.match(lines[entry.line - 1], new RegExp(`^${name} = `), name);
  }
  // 外部常量 Q_HEAD_PAD 由第 63 行 assert 钉为 16，模型记录该依据但不写进常量表。
  const qHeadPad = model.externalConstants.Q_HEAD_PAD;
  assert.equal(qHeadPad.assert, 'assert Q_HEAD_PAD == 16');
  assert.ok(lines[qHeadPad.assertLine - 1].includes(qHeadPad.assert));
  assert.equal('Q_HEAD_PAD' in model.constants, false);
});

test('every key Tensor and Tile carries source-consistent granularity, shape, dtype and storage', () => {
  const w = load(), model = w.PTO_PYPA_SEMANTIC_GRAPH;
  assert.equal(NODE_METADATA.length, DATA_STATE_IDS.length);
  for (const spec of NODE_METADATA) {
    const node = find(model, spec.id), data = payload(node);
    assert.equal(node.type === 'data' || node.type === 'state', true, spec.id);
    assert.equal(data.granularity, spec.granularity, spec.id);
    for (const key of spec.absent || []) assert.equal(key in data, false, `${spec.id}: ${key} 不应声明`);
    if (spec.allocated) assert.deepEqual(plain(resolveDims(model, data.shape.allocated)), spec.allocated, `${spec.id}.allocated`);
    if (spec.valid) assert.deepEqual(plain(resolveDims(model, data.shape.valid)), spec.valid, `${spec.id}.valid`);
    for (const key of ['dtype', 'storage']) if (spec[key]) assert.equal(key === 'dtype' ? data.dtype : data.storage.space, spec[key], `${spec.id}.${key}`);
    if (data.shape && !spec.valid) assert.equal('valid' in data.shape, false, `${spec.id}: 源码未声明 valid`);
  }
});

test('allocated and valid are never conflated and both keep the allocated rank', () => {
  const w = load(), model = w.PTO_PYPA_SEMANTIC_GRAPH;
  let withValid = 0;
  for (const { node, id } of allNodes(model)) {
    const data = payload(node);
    if (node.type !== 'data' && node.type !== 'state') continue;
    if (!data.shape) continue;
    assert.ok(Array.isArray(data.shape.allocated), `${id}: 必须声明 allocated`);
    if (!('valid' in data.shape)) continue;
    withValid += 1;
    assert.equal(data.shape.allocated.length, data.shape.valid.length, `${id}: 维度不一致`);
    assert.notDeepEqual(plain(data.shape.allocated), plain(data.shape.valid), `${id}: allocated 与 valid 不应相同`);
  }
  // 只有源码显式给出 valid_shape / set_validshape 的对象才声明 valid。
  assert.equal(withValid, 7);
});

test('every shape dimension is a declared constant or a declared runtime symbol', () => {
  const w = load(), model = w.PTO_PYPA_SEMANTIC_GRAPH, source = w.PTO_PAGED_ATTENTION_PYPTO_SOURCE;
  const literals = [];
  for (const { node, id } of allNodes(model)) {
    const data = payload(node);
    if (!data.shape) continue;
    for (const dim of [...data.shape.allocated, ...(data.shape.valid || [])]) {
      if (typeof dim === 'number') { literals.push([id, dim]); continue; }
      assert.ok(dim in model.constants || model.runtimeDims.includes(dim), `${id}: 未声明的维度名 ${dim}`);
      assert.ok(source.includes(dim), `${id}: 维度名 ${dim} 不在源码中`);
    }
  }
  // 唯一允许的字面量维度是单行载荷的 1 行：其余维度都必须走常量名或运行时符号，
  // 防止把典型场景的配置值（BATCH = 16 / MAX_BLOCKS = 28 等）直接写死。
  assert.deepEqual(literals, [['k', 1], ['v', 1]]);
  for (const dim of model.runtimeDims) {
    assert.equal(dim in model.constants, false, dim);
    assert.ok(source.includes(`${dim} = `), `${dim} 应有源码定义`);
  }
});

test('runtime-determined shapes stay symbolic instead of fixed scenario values', () => {
  const w = load(), model = w.PTO_PYPA_SEMANTIC_GRAPH;
  for (const id of ['k-cache', 'v-cache', 'cache-append/key', 'cache-append/value']) {
    const shape = find(model, id).data.shape.allocated;
    assert.equal(shape[0], 'cache_token_rows', id);
    assert.equal(shape[1], 'KV_HIDDEN', id);
    assert.equal(model.constants.KV_HIDDEN.value, 1024, id);
  }
  const aux = find(model, 'aux').data;
  assert.equal('shape' in aux, false);
  assert.match(aux.addressing.blockTable, /block_table_2d = pl\.reshape\(block_table, \[active_batch, max_blocks_per_seq\]\)/);
  assert.match(aux.addressing.activeBatch, /pl\.tensor\.dim\(seq_lens, 0\)/);
  // 典型场景的 BATCH = 16 / MAX_BLOCKS = 28 / SEQ_LEN = 3584 不得进入行数与页表描述。
  assert.doesNotMatch(aux.addressing.blockTable, /\b(16|28|3584)\b/);
  assert.doesNotMatch(find(model, 'output').data.shape.allocated.join(','), /\b(16|3584)\b/);
});

test('shape, dtype and storage provenance name the actual basis', () => {
  const w = load(), model = w.PTO_PYPA_SEMANTIC_GRAPH;
  const kinds = (id, key) => plain(kindsCovering(find(model, id), key));
  for (const id of SHAPE_QUOTED) assert.deepEqual(kinds(id, 'shape'), ['fact'], `${id} shape`);
  for (const id of SHAPE_DERIVED) assert.deepEqual(kinds(id, 'shape'), ['resolved'], `${id} shape`);
  for (const id of STORAGE_QUOTED) assert.deepEqual(kinds(id, 'storage'), ['fact'], `${id} storage`);
  for (const id of STORAGE_DERIVED) assert.deepEqual(kinds(id, 'storage'), ['resolved'], `${id} storage`);
  for (const id of DTYPE_DERIVED) assert.deepEqual(kinds(id, 'dtype'), ['resolved'], `${id} dtype`);
  // 其余已声明 dtype 的数据节点都直接来自源码声明，不借推导。
  for (const spec of NODE_METADATA.filter((item) => item.dtype && !DTYPE_DERIVED.includes(item.id))) {
    assert.deepEqual(kinds(spec.id, 'dtype'), ['fact'], `${spec.id} dtype`);
  }
  // 每个 data 节点的事实部分与每条 resolved 依据都必须写明出处。
  for (const { node, id } of allNodes(model)) {
    if (node.type !== 'data') continue;
    const fact = node.provenance.filter((entry) => entry.kind === 'fact');
    assert.equal(fact.length, 1, `${id} fact`);
    assert.ok(fact[0].note && fact[0].note.length > 8, `${id} fact note`);
  }
  for (const { node, id } of allNodes(model)) for (const entry of node.provenance.filter((item) => item.kind === 'resolved')) {
    assert.ok(entry.note && entry.note.length > 8, `${id} resolved note`);
  }
});

test('quoted shapes and storage spaces are literally present in the cited source ranges', () => {
  const w = load(), model = w.PTO_PYPA_SEMANTIC_GRAPH;
  for (const id of SHAPE_IN_RANGE) {
    const node = find(model, id), data = payload(node), text = sourceText(w, node);
    for (const dim of [...data.shape.allocated, ...(data.shape.valid || [])]) {
      if (typeof dim !== 'number') assert.ok(text.includes(dim), `${id}: ${dim} 不在引用范围内`);
    }
  }
  for (const id of STORAGE_QUOTED) {
    const node = find(model, id);
    assert.ok(sourceText(w, node).includes(`pl.MemorySpace.${payload(node).storage.space}`), `${id}: storage`);
  }
  const matmul = find(model, 'qk/score');
  assert.ok(sourceText(w, matmul).includes('out_dtype=pl.FP32'), 'qk/score dtype');
  assert.ok(sourceText(w, find(model, 'qk/q-tile')).includes('valid_shape=[GROUP, HEAD_DIM]'), 'q-tile valid');
  assert.ok(sourceText(w, find(model, 'pv/probability')).includes('valid_shape=[GROUP, BLOCK_SIZE]'), 'pv probability valid');
  assert.ok(sourceText(w, find(model, 'context-acc/partial')).includes('valid_shape=[lane_rows, HEAD_DIM]'), 'context-acc partial valid');
});

test('enrichment preserves the node inventory, empty-subgraph inventory and source mapping', () => {
  const w = load(), model = w.PTO_PYPA_SEMANTIC_GRAPH;
  const all = allNodes(model);
  assert.equal(all.length, 69);
  assert.deepEqual(plain(all.filter(({ node }) => node.type === 'data' || node.type === 'state').map(({ id }) => id)), DATA_STATE_IDS);
  assert.deepEqual(plain(all.filter(({ node }) => node.type === 'composite').map(({ id }) => id)),
    ['q-prep', 'q-prep/norm', 'k-prep', 'k-prep/norm', 'v-scale', 'cache-append', 'qk', 'scale-mask', 'softmax', 'pv', 'context-acc', 'output-norm']);
  for (const id of EMPTY_SUBGRAPHS) {
    const graph = find(model, id).composite.subgraph;
    assert.equal(graph.nodes.some((node) => node.type === 'data' || node.type === 'state'), false, id);
  }
  for (const [id, primaryRange] of [['qk', 1], ['k-cache', 2], ['v-cache', 2], ['context', 1]]) {
    assert.equal(find(model, id).source.primaryRange, primaryRange, id);
  }
  assert.deepEqual(plain(model.groups.map((group) => group.members.length)), [9, 10, 3]);
  assert.deepEqual(plain(model.groups.map((group) => group.id)), ['qkv-preparation', 'paged-attention', 'output-region']);
});

test('main graph and every composite subgraph validate on their own after enrichment', () => {
  const w = load(), model = w.PTO_PYPA_SEMANTIC_GRAPH, validate = w.PTO_SEMANTIC_COMPUTATION_GRAPH.validate;
  assert.doesNotThrow(() => validate(plain(model)));
  for (const { node, id } of allNodes(model)) {
    if (node.type !== 'composite') continue;
    const standalone = { schemaVersion: 2, id: id.split('/').join('-'), ...plain(node.composite.subgraph) };
    assert.doesNotThrow(() => validate(standalone), id);
  }
});

test('enriched metadata still materializes every expansion without geometry drift', () => {
  const w = load(), model = w.PTO_PYPA_SEMANTIC_GRAPH;
  const ids = allNodes(model).filter(({ node }) => node.type === 'composite').map(({ id }) => id);
  const before = JSON.stringify(model);
  for (const expanded of [new Set(), ...ids.map((id) => new Set([id])), new Set(ids)]) {
    const placed = w.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize(model, expanded);
    assert.ok(placed.nodes.every((node) => Number.isFinite(node.x) && Number.isFinite(node.y)), '节点坐标');
    assert.ok(placed.edgeRoutes.every((route) => !route.path.includes('NaN')), '连线');
    const byId = new Map(placed.nodes.map((node) => [node.id, node]));
    for (const edge of placed.edges) {
      assert.ok(edge.sourcePort in byId.get(edge.s).ports.outputs, edge.id);
      assert.ok(edge.targetPort in byId.get(edge.t).ports.inputs, edge.id);
    }
  }
  assert.equal(JSON.stringify(model), before, 'materialize 不得修改语义图');
});
