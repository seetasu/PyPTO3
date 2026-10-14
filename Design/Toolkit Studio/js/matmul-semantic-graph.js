/* Source-grounded matmul semantics; geometry is supplied by the shared layout. */
(function (root) {
  'use strict';
  const source = (start, end, symbols) => ({ file: 'matmul.py', ranges: [{ start, end }], symbols });
  const ports = () => ({ inputs: { default: {} }, outputs: { default: {} } });
  function node(id, type, name, start, end, symbols, description, payload) {
    const inferred = ['role', 'granularity', 'entity', 'relations', 'shape'];
    const fact = Object.keys(payload).filter((key) => !inferred.includes(key));
    const resolved = Object.keys(payload).filter((key) => inferred.includes(key));
    return { id, type, name, description, source: source(start, end, symbols), ports: ports(),
      provenance: [
        ...(fact.length ? [{ kind: 'fact', origin: 'source', fields: fact.map((key) => `${type}.${key}`), note: '来自 matmul.py 的张量契约、切片和计算表达式。' }] : []),
        ...(resolved.length ? [{ kind: 'resolved', origin: 'source', fields: resolved.map((key) => `${type}.${key}`), note: '由源码契约与 M/N 切片静态推导；不代表设备运行采集。' }] : []),
      ], [type]: payload };
  }
  const data = (id, name, symbol, shape, role, granularity, start, end, description, extra = {}) =>
    node(id, 'data', name, start, end, [symbol], description,
      { role, granularity, entity: `matmul.${id}`, symbol, dtype: 'FP32', shape: { allocated: shape }, ...extra });
  const op = (id, name, start, end, symbols, description, category, api) =>
    node(id, 'operation', name, start, end, symbols, description, { semantics: description, category, api });
  const edge = (s, t, label, extra = {}) => ({ id: `${s}>${t}`, s, t, type: 'data', sourcePort: 'default', targetPort: 'default', label, ...extra });
  const children = [
    op('slice-a', 'A 行切片', 34, 34, ['a', 'mb', 'tile_a'], '按 mb 取 M_TILE 行，保留完整 K 维。', 'view', []),
    op('slice-b', 'B 列切片', 35, 35, ['b', 'nb', 'tile_b'], '按 nb 取 N_TILE 列，保留完整 K 维。', 'view', []),
    data('tile-a', 'A Tile', 'tile_a', ['M_TILE', 'K'], 'input-view', 'tile', 34, 34,
      'tile_a = a[mb : mb + M_TILE, :]，64 × 256；同一 M 块供 4 个 N 块使用。',
      { relations: [{ kind: 'slice', entity: 'matmul.a', description: 'a[mb : mb + M_TILE, :]' }], addressing: { rows: 'mb : mb + M_TILE', columns: ':' } }),
    data('tile-b', 'B Tile', 'tile_b', ['K', 'N_TILE'], 'input-view', 'tile', 35, 35,
      'tile_b = b[:, nb : nb + N_TILE]，256 × 64；同一 N 块供 4 个 M 块使用。',
      { relations: [{ kind: 'slice', entity: 'matmul.b', description: 'b[:, nb : nb + N_TILE]' }], addressing: { rows: ':', columns: 'nb : nb + N_TILE' } }),
    op('multiply', 'Tile Matmul', 36, 36, ['tile_a', 'tile_b'],
      'C_tile[i, j] = Σₖ tile_a[i, k] × tile_b[k, j]；一次 pl.matmul 收缩全部 K = 256，结果为 64 × 64。源码没有 K 分块、显式累加器或硬件存储层声明。', 'matmul', ['pl.matmul']),
    data('result', 'C Tile', 'pl.matmul(tile_a, tile_b)', ['M_TILE', 'N_TILE'], 'intermediate', 'tile', 36, 36,
      '矩阵乘法的局部结果，写入 FP32 输出 c 的对应 64 × 64 区域；源码未为中间结果显式指定 dtype。'),
    op('write', '写回 C 切片', 36, 36, ['c', 'mb', 'nb'],
      '通过索引赋值写入 c[mb : mb + M_TILE, nb : nb + N_TILE]；不同 M/N 块写入互不重叠的区域。', 'store', []),
  ];
  // The intermediate expression has no explicit dtype declaration.
  delete children.find((item) => item.id === 'result').data.dtype;
  children.find((item) => item.id === 'result').provenance.forEach((entry) => { entry.fields = entry.fields.filter((field) => field !== 'data.dtype'); });
  const tile = node('tile', 'composite', 'M/N Tile 计算 × 16', 31, 36, ['mb', 'nb', 'matmul_tile'],
    '两层 pl.parallel 枚举 4 × 4 个输出分块；每个 CORE_GROUP scope 执行相同的切片、矩阵乘法与写回。展开查看一个通用 Tile 的计算依赖，不推断物理核数量或设备执行时序。',
    { semantics: '每个 (mb, nb) 独立计算一个 64 × 64 输出区域；K 不切分。', category: 'matmul', api: ['pl.parallel', 'pl.at', 'pl.matmul'],
      subgraph: { schemaVersion: 2, id: 'matmul-tile', nodes: children, edges: [edge('slice-a', 'tile-a', 'A 切片'), edge('slice-b', 'tile-b', 'B 切片'), edge('tile-a', 'multiply', '64 × 256'), edge('tile-b', 'multiply', '256 × 64'), edge('multiply', 'result', '64 × 64'), edge('result', 'write', '局部结果')], groups: [] },
      portMappings: { inputs: { A: { node: 'slice-a', port: 'default', relation: 'view' }, B: { node: 'slice-b', port: 'default', relation: 'view' } }, outputs: { C: { node: 'write', port: 'default', relation: 'aggregate', description: '16 个互不重叠的输出切片组成 C。' } } } });
  tile.ports = { inputs: { A: {}, B: {} }, outputs: { C: {} } };
  root.PTO_MATMUL_SEMANTIC_GRAPH = {
    schemaVersion: 2, id: 'matmul', sourceFiles: { 'matmul.py': { lineCount: 85 } },
    constants: { M: { value: 256, line: 18 }, N: { value: 256, line: 19 }, K: { value: 256, line: 20 }, M_TILE: { value: 64, line: 21 }, N_TILE: { value: 64, line: 22 } },
    nodes: [data('a', '输入 A', 'a', ['M', 'K'], 'input', 'tensor', 27, 27, 'FP32 输入 A，256 × 256；每个 Tile 读取所需的 M 行块与全部 K 列。'),
      data('b', '输入 B', 'b', ['K', 'N'], 'input', 'tensor', 28, 28, 'FP32 输入 B，256 × 256；每个 Tile 读取全部 K 行与所需的 N 列块。'), tile,
      data('c', '输出 C', 'c', ['M', 'N'], 'output', 'tensor', 29, 37, '16 个 64 × 64 切片组成 FP32 输出 C，256 × 256；C = A @ B。')],
    edges: [edge('a', 'tile', 'A · M 行块 / 完整 K', { targetPort: 'A' }), edge('b', 'tile', 'B · 完整 K / N 列块', { targetPort: 'B' }), edge('tile', 'c', '16 个输出切片', { sourcePort: 'C' })], groups: [],
  };
})(typeof window === 'undefined' ? globalThis : window);
