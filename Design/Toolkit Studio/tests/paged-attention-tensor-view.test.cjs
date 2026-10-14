/* Step 2 验收：Tensor 节点显示 Shape / Tensor Inspector 解释 Shape 与有效区域 /
 * QK Matmul 的输入 → 操作 → 输出 Shape。全部断言只读语义图与 Presentation 投影，
 * 不在测试里重算形状；渲染层用真实 HTML 输出验证「不显示 undefined / null」。 */
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const JS = path.join(__dirname, '..', 'js');
const read = (file) => fs.readFileSync(path.join(JS, file), 'utf8');
const MODULES = ['paged-attention-pypto-source', 'semantic-computation-graph',
  'semantic-computation-presentation', 'semantic-computation-layout', 'paged-attention-semantic-graph'];

function load(extra = {}) {
  const context = vm.createContext({ window: {}, console, ...extra });
  for (const name of MODULES) vm.runInContext(read(`${name}.js`), context, { filename: `${name}.js` });
  return context.window;
}
const plain = (value) => JSON.parse(JSON.stringify(value));
function allNodes(graph, prefix = '') {
  return graph.nodes.flatMap((node) => [{ node, id: prefix + node.id }, ...(node.composite ? allNodes(node.composite.subgraph, `${prefix}${node.id}/`) : [])]);
}
const find = (model, id) => allNodes(model).find((item) => item.id === id).node;
/* 投影层接口只接受语义节点，这里按限定 id 取节点再交给投影，避免测试自带形状表。 */
const project = (window, id, name) => {
  const model = window.PTO_PYPA_SEMANTIC_GRAPH;
  return plain(window.PTO_COMPUTATION_PRESENTATION[name]({ id, semantic: find(model, id) }, { model, qualifiedId: id }));
};
const rowValue = (detail, label) => detail.groups.flatMap((group) => group.rows).find((row) => row.label === label)?.value;

/* ================= 1. Q Tile ================= */
test('Q Tile shows the allocated 16 × 128 shape and the inspector treats 5 × 128 as the valid region', () => {
  const w = load(), model = w.PTO_PYPA_SEMANTIC_GRAPH;
  const card = plain(w.PTO_COMPUTATION_PRESENTATION.card(find(model, 'qk/q-tile'), model));
  assert.equal(card.shapeText, '[16, 128]');
  assert.equal(card.shapeSymbol, '[ROW_TILE, HEAD_DIM]');
  assert.equal(card.meta, 'Tile · Mat');
  assert.equal(card.undeclared, false);
  assert.match(card.shapeTitle, /valid \[GROUP, HEAD_DIM\] = \[5, 128\]/);

  const detail = project(w, 'qk/q-tile', 'tensorDetail');
  assert.equal(rowValue(detail, 'Granularity'), 'Tile');
  assert.equal(rowValue(detail, 'Allocated Shape'), '[16, 128]');
  assert.equal(rowValue(detail, 'Valid Shape'), '[5, 128]');
  assert.equal(rowValue(detail, 'Memory Space'), 'Mat');
  assert.equal(rowValue(detail, '源码变量'), 'q_tile');
  // 有效区域按 Allocated 的尺度绘制：16 行里前 5 行有效，列 128 全宽。
  assert.equal(detail.region.rows, 16);
  assert.equal(detail.region.cols, 128);
  assert.equal(detail.region.validRows, 5);
  assert.equal(detail.region.validCols, 128);
  assert.equal(detail.region.full, false);
  assert.equal(detail.region.headline, 'Allocated 16 × 128 · Valid 5 × 128');
  assert.match(detail.region.rowLine, /行 0–4 属于当前计算有效区域（GROUP = 5 \/ ROW_TILE = 16）/);
  assert.match(detail.region.rowLine, /行 5–15 不属于当前计算/);
  assert.deepEqual(detail.region.legend, { valid: '当前计算有效区域', invalid: '非当前计算有效区域' });
  assert.match(detail.region.disclaimer, /非有效区域不等于 Padding/);
  assert.match(detail.region.disclaimer, /也不默认为 0/);
  assert.equal(detail.regionNote, '');
});

/* ================= 2. K Page ================= */
test('K Page shows a full 128 × 128 page in Mat memory and never invents a valid region', () => {
  const w = load(), model = w.PTO_PYPA_SEMANTIC_GRAPH;
  const card = plain(w.PTO_COMPUTATION_PRESENTATION.card(find(model, 'qk/k-page'), model));
  assert.equal(card.shapeText, '[128, 128]');
  assert.equal(card.shapeSymbol, '[BLOCK_SIZE, HEAD_DIM]');
  assert.match(card.meta, /Page · Mat/);

  const detail = project(w, 'qk/k-page', 'tensorDetail');
  assert.equal(rowValue(detail, 'Granularity'), 'Page');
  assert.equal(rowValue(detail, 'Allocated Shape'), '[128, 128]');
  assert.equal(rowValue(detail, 'Memory Space'), 'Mat');
  // 源码只在 pl.load 之外声明 valid；这里没有，所以既不画有效区域也不写 Valid Shape 行。
  assert.equal(detail.region, null);
  assert.equal(rowValue(detail, 'Valid Shape'), undefined);
  assert.match(detail.regionNote, /只声明了 Allocated Shape，没有声明 Valid Shape/);
  // 整页有效是结论而不是源码事实：不得据此补一个 [128, 128] 的 Valid。
  assert.equal(detail.shapeTitle, 'allocated [BLOCK_SIZE, HEAD_DIM] = [128, 128]');
});

/* ================= 3. K Cache 动态维度 ================= */
test('K Cache keeps its runtime row count instead of a fixed row number', () => {
  const w = load(), model = w.PTO_PYPA_SEMANTIC_GRAPH;
  const card = plain(w.PTO_COMPUTATION_PRESENTATION.card(find(model, 'k-cache'), model));
  assert.equal(card.shapeText, '[cache_token_rows, 1024]');
  assert.equal(card.shapeSymbol, '[cache_token_rows, KV_HIDDEN]');
  assert.equal(card.runtime, undefined);
  assert.doesNotMatch(card.shapeText, /\b\d+\b(?!\])/);

  const detail = project(w, 'k-cache', 'tensorDetail');
  assert.equal(rowValue(detail, 'Allocated Shape'), '[cache_token_rows, 1024]');
  assert.equal(rowValue(detail, '动态维度'), 'cache_token_rows 由运行时输入决定，不写成固定值。');
  assert.equal(rowValue(detail, 'Memory Space'), 'GM');
  // 动态尺度与 Valid 的比例不可靠，因此不给比例图，只给拒绝理由。
  assert.equal(detail.region, null);
  assert.match(detail.regionNote, /只声明了 Allocated Shape，没有声明 Valid Shape/);
  const other = project(w, 'cache-append/key', 'tensorDetail');
  assert.equal(rowValue(other, 'Allocated Shape'), '[cache_token_rows, 1024]');
});

/* ================= 4. 未声明字段不被伪造 ================= */
test('nodes without a declared shape or dtype keep that fact instead of borrowing one', () => {
  const w = load(), model = w.PTO_PYPA_SEMANTIC_GRAPH;
  for (const id of ['q-proj', 'k-proj', 'v-proj']) {
    const card = plain(w.PTO_COMPUTATION_PRESENTATION.card(find(model, id), model));
    assert.equal(card.shapeText, null, id);
    assert.equal(card.shapeSymbol, null, id);
    assert.equal(card.shapeTitle, '', id);
    assert.equal(card.undeclared, true, id);
    assert.equal(card.tensor, true, id);
    const detail = project(w, id, 'tensorDetail');
    assert.equal(rowValue(detail, 'Shape'), '源码未声明', id);
    assert.equal(rowValue(detail, 'Dtype'), '源码未声明', id);
    assert.equal(rowValue(detail, 'Allocated Shape'), undefined, id);
    assert.equal(detail.region, null, id);
    assert.equal(detail.regionNote, '', id);
    assert.equal(detail.shapeTitle, '', id);
  }
  // aux 是控制数据 bundle，既没有 shape 也不该被当成普通 Tensor。
  const aux = project(w, 'aux', 'tensorDetail');
  assert.equal(rowValue(aux, 'Granularity'), 'Bundle');
  assert.equal(rowValue(aux, 'Shape'), '源码未声明');
  assert.equal(rowValue(aux, 'Dtype'), '源码未声明');
  assert.equal(aux.region, null);
  // 单行载荷的 k / v 有 shape 但没有 pl.Tensor 位置，Memory Space 行整行不得出现。
  const k = project(w, 'k', 'tensorDetail');
  assert.equal(rowValue(k, 'Allocated Shape'), '[1, 128]');
  assert.equal(rowValue(k, 'Memory Space'), undefined);
});

/* ================= 5. QK Matmul 的输入输出 Shape ================= */
test('the QK Matmul transformation derives the shapes the source declares and matches them', () => {
  const w = load();
  for (const id of ['qk', 'qk/matmul']) {
    const t = project(w, id, 'transformation');
    assert.equal(t.name, 'QK Matmul', id);
    assert.equal(t.title, 'QK Matmul', id);
    assert.deepEqual(plain(t.operation.api), ['pl.matmul', 'pl.tile.transpose_view'], id);
    assert.equal(t.operands.length, 2, id);
    assert.deepEqual(plain(t.operands.map((entry) => [entry.name, entry.shape.text, entry.transposed])), [
      ['Q Tile', '[16, 128]', false],
      ['K Page', '[128, 128]', true],
    ], id);
    assert.equal(t.results.length, 1, id);
    assert.deepEqual(plain(t.results.map((entry) => [entry.name, entry.shape.text, entry.dtype])), [['FP32 Scores', '[16, 128]', 'FP32']], id);
    // 收缩维取自右侧操作数被转置后的首维，不是 K Page 被 load 出来的 [BLOCK_SIZE, HEAD_DIM] 首维。
    assert.equal(t.rule.contract, 'HEAD_DIM', id);
    assert.equal(t.rule.text, '[16, 128]', id);
    assert.equal(t.rule.symbol, '[ROW_TILE, BLOCK_SIZE]', id);
    assert.equal(t.rule.declared, '[16, 128]', id);
    assert.equal(t.rule.consistent, true, id);
    assert.match(t.rule.steps[0], /Q Tile 的末维 HEAD_DIM 与 K Page 的首维 HEAD_DIM 收缩/);
    // 操作数没有在源码里声明 dtype：不写、也不从邻节点推断。
    assert.deepEqual(plain(t.operands.map((entry) => entry.dtype)), [null, null], id);
    assert.match(t.transposeNote, /pl\.tile\.transpose_view/);
    // 只有前 GROUP 行参与本次计算。
    assert.match(t.notes.find((note) => note.label === '有效范围').value, /只有 Valid Shape \[5, 128\]/);
  }
  // Composite 只在自己确实是聚合边界时才列边界端口。
  const composite = project(w, 'qk', 'transformation');
  assert.deepEqual(plain(composite.boundary.filter((edge) => edge.direction === 'in').map((edge) => [edge.label, edge.peer])), [
    ['block_table', 'Aux Inputs'], ['Q', 'Q'], ['K · block_table', 'K Cache'],
  ]);
  assert.deepEqual(plain(composite.boundary.filter((edge) => edge.direction === 'out').map((edge) => [edge.label, edge.peer])), [['Scores', 'Scores']]);
  assert.deepEqual(plain(project(w, 'qk/matmul', 'transformation').boundary), []);
});

/* ================= 6. 单页 Scores 与 Stack Scores ================= */
test('a single-page score result is never presented as the stack-level scores tensor', () => {
  const w = load(), model = w.PTO_PYPA_SEMANTIC_GRAPH;
  const t = project(w, 'qk', 'transformation');
  assert.deepEqual(plain(t.results.map((entry) => [entry.id, entry.shape.text])), [['score', '[16, 128]']]);
  assert.equal(t.results.every((entry) => entry.shape.text !== '[1152, 512]'), true);
  const granularity = t.notes.find((note) => note.label === '粒度区别');
  assert.ok(granularity, '必须给出单页与 Stack 的粒度区别');
  assert.match(granularity.value, /FP32 Scores 是实体 attention\.scores 在 \[16, 128\] 上的单页切片/);
  assert.match(granularity.value, /父图 Scores（score_transfer）粒度是 Stack，Allocated \[1152, 512\]/);
  assert.match(granularity.value, /两者不是同一个 Tensor/);
  assert.match(t.notes.find((note) => note.label === 'FP32 Scores 写入语义').value, /写入 score_transfer 的当前 Stack 列区间/);
  // 两个节点在卡片上必须是两个不同的 Shape。
  assert.equal(plain(w.PTO_COMPUTATION_PRESENTATION.card(find(model, 'qk/score'), model)).shapeText, '[16, 128]');
  assert.equal(plain(w.PTO_COMPUTATION_PRESENTATION.card(find(model, 'scores'), model)).shapeText, '[1152, 512]');
});

/* ================= 7. 渲染层不出现 undefined / null ================= */
/* 直接抽取 demo-v2.js 的渲染函数，用真实模型跑一遍全部 DATA / STATE 节点。 */
function renderer() {
  const demo = read('demo-v2.js');
  const slice = (start, end) => {
    const from = demo.indexOf(start);
    assert.notEqual(from, -1, `missing ${start}`);
    const to = demo.indexOf(end, from);
    assert.notEqual(to, -1, `missing ${end}`);
    return demo.slice(from, to);
  };
  const window = {};
  const context = vm.createContext({ window, console });
  for (const name of MODULES) vm.runInContext(read(`${name}.js`), context, { filename: `${name}.js` });
  vm.runInContext([
    `const pypaEscape = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));`,
    slice('  function pypaComputeCardHtml(node) {', '  // 缩放读数：'),
    slice('  function pypaInspectorRows(rows) {', '  /* 总览保留任务执行图'),
    'window.__tensorView = { pypaComputeCardHtml, pypaInspectorHtml, pypaRegionHtml, pypaTransformHtml };',
  ].join('\n'), context);
  const model = window.PTO_PYPA_SEMANTIC_GRAPH;
  const expanded = new Set(allNodes(model).filter(({ node }) => node.type === 'composite').map(({ id }) => id));
  const placed = window.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize(model, expanded);
  return { window, render: window.__tensorView, placed };
}

test('every rendered card and inspector section avoids undefined, null and NaN', () => {
  const { render, placed } = renderer();
  const banned = /undefined|null|NaN|\[object Object\]/;
  const dataNodes = placed.nodes.filter((node) => node.kind === 'data');
  assert.ok(dataNodes.length >= 29, '所有 DATA / STATE 节点都应被投影');
  for (const node of dataNodes) {
    const card = render.pypaComputeCardHtml(node);
    assert.doesNotMatch(card, banned, `${node.id} card`);
    assert.match(card, /kf-pypa-graph__node-shape|kf-pypa-graph__node-meta-line/, `${node.id} card 内容`);
    const html = render.pypaInspectorHtml(node, node.sourceRanges);
    assert.doesNotMatch(html, banned, `${node.id} inspector`);
    assert.match(html, /NODE INSPECTOR/, node.id);
  }
  for (const node of placed.nodes.filter((item) => item.kind === 'operation')) {
    assert.doesNotMatch(render.pypaInspectorHtml(node, node.sourceRanges), banned, `${node.id} inspector`);
  }
  // 缺字段的节点用的是明确文案，而不是空值占位。
  const qProj = placed.nodes.find((node) => node.id === 'q-proj');
  assert.match(render.pypaComputeCardHtml(qProj), /Shape 未声明/);
  assert.match(render.pypaInspectorHtml(qProj, qProj.sourceRanges), /源码未声明/);
  // 卡片 Shape 用 <wbr> 换行，维名不会被拆开。
  const kCache = placed.nodes.find((node) => node.id === 'k-cache');
  assert.match(render.pypaComputeCardHtml(kCache), /\[cache_token_rows,<wbr> 1024\]/);
});

test('the rendered inspector shows the valid-region diagram and the QK transformation cards', () => {
  const { render, placed } = renderer();
  const qTile = placed.nodes.find((node) => node.id === 'qk/q-tile');
  const html = render.pypaInspectorHtml(qTile, qTile.sourceRanges);
  assert.match(html, /有效区域 · Allocated \/ Valid/);
  assert.match(html, /kf-pypa-graph__region-grid/);
  assert.match(html, /当前计算有效区域：5 行 × 128 列/);
  assert.match(html, /非当前计算有效区域：11 行 × 128 列/);
  assert.match(html, /非有效区域不等于 Padding/);
  // 有效区域按行铺 16 条带，前 5 条为有效；列满宽时不画分割线。
  assert.equal((html.match(/<i class="is-valid"><\/i>/g) || []).length, 5 + 1);
  assert.doesNotMatch(html, /region-split/);

  const kPage = placed.nodes.find((node) => node.id === 'qk/k-page');
  const kPageHtml = render.pypaInspectorHtml(kPage, kPage.sourceRanges);
  assert.doesNotMatch(kPageHtml, /region-grid/);
  assert.match(kPageHtml, /没有声明 Valid Shape，因此不绘制有效区域/);

  const qk = placed.nodes.find((node) => node.id === 'qk');
  const transformHtml = render.pypaInspectorHtml(qk, qk.sourceRanges);
  assert.match(transformHtml, /Shape Transformation · QK Matmul/);
  assert.match(transformHtml, /kf-pypa-graph__transform-card/);
  assert.match(transformHtml, /\[16, 128\]/);
  assert.match(transformHtml, /\[128, 128\]/);
  assert.match(transformHtml, /class="kf-pypa-graph__transform-mark">ᵀ/);
  assert.match(transformHtml, /收缩维/);
  assert.match(transformHtml, /粒度区别/);
  assert.match(transformHtml, /dtype 源码未声明/);
  assert.match(transformHtml, /Composite 边界/);
});

/* ================= 8. Composite 展开与收起 ================= */
test('expanding and collapsing a composite only changes the projected node set', () => {
  const w = load(), model = w.PTO_PYPA_SEMANTIC_GRAPH;
  const materialize = w.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize;
  const compact = materialize(model, new Set());
  assert.equal(compact.nodes.some((node) => node.id === 'qk/q-tile'), false);
  assert.equal(compact.opened.has('qk'), false);
  const expanded = materialize(model, new Set(['qk']));
  assert.equal(expanded.opened.has('qk'), true);
  const head = expanded.nodes.find((node) => node.id === 'qk');
  assert.equal(head.expandedGroup, true);
  assert.equal(head.kind, 'operation');
  assert.ok(head.groupWidth > 0 && head.groupHeight > 0);
  for (const id of ['qk/q-tile', 'qk/k-page', 'qk/matmul', 'qk/score']) {
    const node = expanded.nodes.find((item) => item.id === id);
    assert.ok(node, `${id} 应在展开后出现`);
    assert.equal(node.childOperation, true, id);
    assert.ok(node.type === 'data' || node.type === 'operation', id);
  }
  const collapsed = materialize(model, new Set());
  assert.deepEqual(plain(collapsed.nodes.map((node) => node.id)), plain(compact.nodes.map((node) => node.id)));
});

/* ================= 9. 选择 / 联动 / 视图变换 ================= */
test('selection, source linkage and viewport geometry stay exactly as the layout produced them', () => {
  const w = load(), model = w.PTO_PYPA_SEMANTIC_GRAPH;
  const all = allNodes(model);
  const expanded = new Set(all.filter(({ node }) => node.type === 'composite').map(({ id }) => id));
  const lineCount = Object.values(model.sourceFiles)[0].lineCount;
  // 源码联动：每个节点都能给出落在源码行数内的范围，Selection 才有可定位之处。
  for (const { node, id } of all) {
    assert.ok(node.source.ranges.length > 0, id);
    for (const { start, end } of node.source.ranges) {
      assert.ok(start >= 1 && end >= start && end <= lineCount, `${id}: ${start}–${end}`);
    }
  }
  // 每个节点的限定 id 都能被 Inspector 重新定位（选择与详情共用同一条路径）。
  for (const { node, id } of all) {
    const detail = w.PTO_COMPUTATION_PRESENTATION.tensorDetail({ id, semantic: node }, { model, qualifiedId: id });
    assert.equal(detail.id, node.id, id);
    assert.ok(detail.title, id);
  }
  // Step 2 只加展示字段：几何必须与纯布局输出逐节点相等，缩放 / 平移 / Fit 的取景因此不变。
  const projected = w.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize(model, expanded);
  const laid = w.PTO_COMPUTATION_LAYOUT.layout(model, expanded);
  const geometry = (graph) => plain(graph.nodes.map((node) => [node.id, node.x, node.y, node.w, node.h, node.parentId || '']));
  assert.deepEqual(geometry(projected), geometry(laid));
  assert.deepEqual(plain(projected.nodes.map((node) => node.id)), plain(laid.nodes.map((node) => node.id)));
  assert.equal(projected.width, laid.width);
  assert.equal(projected.height, laid.height);
  // 布局预留的卡片盒高不变，Shape 行不会把节点撑开（Composite 收起时用盒高，展开时用头部高度）。
  const collapsed = w.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize(model, new Set());
  for (const [id, w_, h_] of [['q-proj', 170, 88], ['k-cache', 170, 88], ['q', 170, 88], ['scores', 170, 88], ['qk', 188, 88]]) {
    const node = collapsed.nodes.find((item) => item.id === id);
    assert.deepEqual([node.w, node.h], [w_, h_], id);
  }
  const qkHead = projected.nodes.find((item) => item.id === 'qk');
  assert.equal(qkHead.expandedGroup, true);
  assert.equal(qkHead.h, 42);
});

/* ================= 10. 其他视图无回归 ================= */
test('the new projections are read-only and stay out of the other views', () => {
  const w = load(), model = w.PTO_PYPA_SEMANTIC_GRAPH;
  const P = w.PTO_COMPUTATION_PRESENTATION;
  const before = JSON.stringify(model);
  for (const { node, id } of allNodes(model)) {
    P.card({ id, semantic: node }, model);
    P.tensorDetail({ id, semantic: node }, { model, qualifiedId: id });
  }
  assert.equal(JSON.stringify(model), before, '投影不得修改语义图');
  // 卡片内容只由 compute 视图的分支消费；其他视图仍读 node.sub 那一行。
  const demo = read('demo-v2.js');
  assert.match(demo, /pypaGraphView === 'compute' \? pypaComputeCardHtml\(node\) : `<small>\$\{pypaEscape\(node\.sub\)\}<\/small>`/);
  // 语义契约与节点清单未受影响。
  assert.doesNotThrow(() => w.PTO_SEMANTIC_COMPUTATION_GRAPH.validate(plain(model)));
  assert.equal(allNodes(model).length, 69);
  assert.equal(allNodes(model).filter(({ node }) => node.type === 'data' || node.type === 'state').length, 29);
});
