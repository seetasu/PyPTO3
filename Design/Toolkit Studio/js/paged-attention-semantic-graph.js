/* Source-grounded semantic model. Display strings are derived by the presentation
 * adapter; graph identity, ports and evidence are independent of geometry. */
(function (root) {
  'use strict';
  const FILE = 'paged_attention_pypto.py';
  const parseRanges = (ranges) => ranges.split('、').map((part) => {
    const [start, end = start] = part.split('–').map(Number);
    return { start, end };
  });
  const source = (ranges, symbols = []) => ({ file: FILE, ranges: parseRanges(ranges), symbols });
  const evidence = (type, payload, note, resolved = []) => {
    if (type === 'data') resolved = [...new Set([...resolved, ...['role', 'granularity', 'entity', 'relations', 'addressing'].filter((key) => key in payload)])];
    const fact = Object.keys(payload).filter((key) => !resolved.includes(key));
    return [
      ...(fact.length ? [{ kind: 'fact', origin: 'source', fields: fact.map((key) => `${type}.${key}`), ...(note ? { note } : {}) }] : []),
      ...(resolved.length ? [{ kind: 'resolved', origin: 'source', fields: resolved.map((key) => `${type}.${key}`), note: '由本算子已声明的输入、计算和视图映射推导；不代表运行采集结果。' }] : []),
    ];
  };
  const ordinaryPorts = (entity) => ({ inputs: { default: {} }, outputs: { default: entity ? { entity } : {} } });
  function node(id, type, name, ranges, symbols, description, payload, resolved = []) {
    return { id, type, name, description, source: source(ranges, symbols),
      provenance: evidence(type, payload, undefined, resolved), ports: ordinaryPorts(payload.entity), [type]: payload };
  }
  const D = (id, name, ranges, symbols, description, data, resolved) => node(id, 'data', name, ranges, symbols, description, data, resolved);
  const O = (id, name, ranges, api, description, category, symbols = []) => node(id, 'operation', name, ranges, symbols, description,
    { semantics: description, category, api });
  const S = (id, name, ranges, symbols, description, state) => node(id, 'state', name, ranges, symbols, description, state);
  const E = (s, t, label, type = 'data', extra = {}) => ({ id: `${s}>${t}:${label}`, s, t, type,
    sourcePort: 'default', targetPort: 'default', label, ...extra });
  function frame(id, nodes, edges, inputs, outputs) {
    const ports = { inputs: {}, outputs: {} }, portMappings = { inputs: {}, outputs: {} };
    for (const side of ['inputs', 'outputs']) {
      for (const [name, binding] of Object.entries(side === 'inputs' ? inputs : outputs)) {
        const spec = typeof binding === 'string' ? { node: binding, relation: 'view' } : binding;
        const { entity, ...mapping } = spec;
        ports[side][name] = entity ? { entity } : {};
        portMappings[side][name] = { port: 'default', ...mapping };
      }
    }
    return { graph: { schemaVersion: 2, id, nodes, edges, groups: [] }, ports, portMappings };
  }
  function C(id, name, ranges, symbols, description, category, api, child) {
    const composite = { semantics: description, category, api, subgraph: child.graph, portMappings: child.portMappings };
    return { ...node(id, 'composite', name, ranges, symbols, description, composite), ports: child.ports };
  }
  const view = (node, entity) => ({ node, entity, relation: 'view' });
  const slice = (node, entity, description) => ({ node, entity, relation: 'slice', description });
  const control = (node, entity, description) => ({ node, entity, relation: 'control', description });
  const aggregate = (node, entity, description) => ({ node, entity, relation: 'aggregate', description });
  const cachePayload = (key) => ({ role: 'cache', granularity: 'tensor', entity: `cache.${key}`,
    symbol: key === 'key' ? 'key_cache_bsnd' : 'value_cache_bsnd',
    shape: { allocated: ['cache_token_rows', 'KV_HIDDEN'] },
    addressing: { logicalPage: 'stack * STACK_PAGES + page_in_stack', physicalPage: 'block_table_2d[batch, logical_page]',
      row: 'cache_base + physical_page * BLOCK_SIZE + token_offset', column: 'kv_head * HEAD_DIM + dim' } });
  const keyCache = (id, name = 'K Cache') => D(id, name, '17–25、101–107、194–198、261–295', ['key_cache_bsnd', 'block_table_2d', 'cache_base', 'col'],
    '分页 Key Cache 的二维访问视图。逻辑页经 block_table 映射到物理页；物理页号和外部 Cache dtype 未在此源码中声明为常量。', cachePayload('key'));
  const valueCache = (id, name = 'V Cache') => D(id, name, '17–25、101–107、210–213、261–267、364–429', ['value_cache_bsnd', 'block_table_2d', 'cache_base', 'col'],
    '分页 Value Cache 的二维访问视图，与 K Cache 使用相同逻辑页表和行列寻址。外部 Cache dtype 未在函数签名中声明。', cachePayload('value'));
  const normFrame = (isQ) => frame(isQ ? 'q-rmsnorm' : 'k-rmsnorm', [
    O('sum', 'Row Sum', isQ ? '229–229' : '175–175', ['pl.row_sum', 'pl.mul'], '按行求平方和。', 'reduction', [isQ ? 'q_ss' : 'k_ss']),
    O('inv', 'Inverse RMS', isQ ? '230–230' : '176–176', ['pl.recip', 'pl.sqrt', 'pl.add', 'pl.mul'], '由平方和、HEAD_DIM_INV 和 EPS 求逆 RMS。', 'normalization', [isQ ? 'q_inv' : 'k_inv']),
    O('weight', 'Norm Weight', isQ ? '231–234' : '177–180', ['pl.row_expand_mul', 'pl.col_expand_mul'], '应用 head 权重与逆 RMS。', 'normalization', [isQ ? 'q_heads' : 'k_normed']),
  ], [E('sum', 'inv', 'sum'), E('inv', 'weight', 'inv_rms')],
  { default: view('sum', isQ ? 'projection.q' : 'projection.k') }, { default: view('weight', isQ ? 'normalized.q' : 'normalized.k') });

  const subgraphs = {
    'q-prep': frame('q-preparation', [
      C('norm', 'Q RMSNorm', '215–234', ['q_raw', 'q_heads'], '按 head 对 Q 应用 inv_rms、RMSNorm 与 q_norm_w。', 'normalization', ['pl.row_sum', 'pl.recip', 'pl.row_expand_mul'], normFrame(true)),
      O('rope', 'Q RoPE', '235–246', ['pl.concat', 'pl.col_expand_mul', 'pl.sub', 'pl.add'], 'cos/sin 旋转 Q 的半维。', 'position', ['q_rotated']),
      O('write', '写入 Q', '247–251', ['pl.cast'], 'BF16 Q 写入 q2d；不写 KV Cache。', 'store', ['q2d', 'q_rotated']),
    ], [E('norm', 'rope', 'normalized Q'), E('rope', 'write', 'rotated Q')],
    { Q: view('norm', 'projection.q') }, { Q: view('write', 'attention.q') }),
    'k-prep': frame('k-preparation', [
      C('norm', 'K RMSNorm', '162–180', ['k_raw', 'k_normed'], '按 head 对 K 应用 inv_rms、RMSNorm 与 k_norm_w。', 'normalization', ['pl.row_sum', 'pl.recip', 'pl.row_expand_mul'], normFrame(false)),
      O('rope', 'K RoPE', '181–193', ['pl.concat', 'pl.col_expand_mul', 'pl.sub', 'pl.add'], 'cos/sin 旋转 K 的半维。', 'position', ['k_rotated']),
      O('cast', 'BF16 K', '194–198', ['pl.cast'], 'K 转 BF16 后写入 Key Cache。', 'cast', ['k_rotated']),
    ], [E('norm', 'rope', 'normalized K'), E('rope', 'cast', 'rotated K')],
    { K: view('norm', 'projection.k') }, { K: view('cast', 'attention.k') }),
    'v-scale': frame('v-scaling', [
      O('scale', 'V × inv_rms', '200–207', ['pl.mul'], 'V 投影与 inv_rms 相乘。', 'elementwise', ['v_proj', 'inv_rms']),
      O('cast', 'BF16 V', '200–213', ['pl.cast'], '缩放后的 V 转 BF16。', 'cast', ['v_row_bf16']),
    ], [E('scale', 'cast', 'scaled V')], { V: view('scale', 'projection.v') }, { V: view('cast', 'attention.v') }),
    'cache-append': frame('cache-append', [
      D('slot', '写入位置', '146–155、194–194', ['slot_mapping', 'write_slot', 'cache_row'], 'slot_mapping 先钳制到非负值，再加当前层基址，决定当前 token 的 Cache 行。',
        { role: 'metadata', granularity: 'scalar', symbol: 'write_slot', entity: 'metadata.slot', dtype: 'INDEX' }),
      O('write-k', '追加 K', '194–198', [], 'K 载荷写入 Key Cache；这是索引赋值。', 'store', ['key_cache_bsnd', 'k_rotated']),
      O('write-v', '追加 V', '210–213', [], 'V 载荷写入 Value Cache；这是索引赋值。', 'store', ['value_cache_bsnd', 'v_row_bf16']),
      keyCache('key', 'Key Cache'), valueCache('value', 'Value Cache'),
    ], [E('slot', 'write-k', 'slot', 'control'), E('slot', 'write-v', 'slot', 'control'), E('write-k', 'key', 'K'), E('write-v', 'value', 'V')],
    { K: view('write-k', 'attention.k'), V: view('write-v', 'attention.v'), slot: view('slot', 'metadata.slot') },
    { K: view('key', 'cache.key'), V: view('value', 'cache.value') }),
    qk: frame('qk', [
      D('q-tile', 'Q Tile', '269–276', ['q_tile', 'q2d'], '从 q2d 加载当前 Q 组到 Mat；外部 Tensor 的 dtype 未由此 load 显式声明。',
        { role: 'input-view', granularity: 'tile', entity: 'attention.q', symbol: 'q_tile', shape: { allocated: ['ROW_TILE', 'HEAD_DIM'], valid: ['GROUP', 'HEAD_DIM'] }, storage: { space: 'Mat' } }),
      D('k-page', 'K Page', '261–267、282–295', ['qk_k_page', 'qk_ph', 'block_table_2d'], '通过逻辑页表读物理页号，再将一个 KV head 的 K Page 加载为 Mat Tile。Page 不是 SPMD Block。',
        { role: 'cache-view', granularity: 'page', entity: 'cache.key.page', symbol: 'qk_k_page', shape: { allocated: ['BLOCK_SIZE', 'HEAD_DIM'] }, storage: { space: 'Mat' }, addressing: cachePayload('key').addressing }),
      O('matmul', 'QK Matmul', '296–300', ['pl.matmul', 'pl.tile.transpose_view'], 'Q Tile 与 K Page 转置相乘，显式输出 FP32。', 'matmul', ['qk_score_page']),
      D('score', 'FP32 Scores', '296–305', ['qk_score_page', 'score_transfer'], '每 Page 的 FP32 Scores 写入 score_transfer 的当前 Stack 列区间；它是父图 Stack Scores 的切片，不是整个 Stack。',
        { role: 'intermediate', granularity: 'page', entity: 'attention.scores', symbol: 'qk_score_page', dtype: 'FP32',
          relations: [{ kind: 'slice', entity: 'attention.scores', description: 'pl.store(..., [produce_row, qk_page_offset * BLOCK_SIZE], score_transfer) 将每页写入当前 Stack。' }] }),
    ], [E('q-tile', 'matmul', 'Q'), E('k-page', 'matmul', 'Kᵀ'), E('matmul', 'score', 'Scores')],
    { Q: view('q-tile', 'attention.q'), K: slice('k-page', 'cache.key', '从 K Cache 取当前逻辑页对应的 KV head 切片。'), block: control('k-page', 'metadata.block-table', '页表提供物理页号，不是 K 载荷。') },
    { Scores: aggregate('score', 'attention.scores', '每页 FP32 Scores 按 page offset 写入同一个 Stack 的 score_transfer。') }),
    'scale-mask': frame('scaling-mask', [
      O('scale', 'Scaling', '777–777', ['pl.tile.muls'], 'Scores 乘 SCALE = 1/√HEAD_DIM。', 'elementwise', ['score_scaled', 'SCALE']),
      O('mask', 'Valid Columns Mask', '778–788', ['pl.set_validshape', 'pl.fillpad'], '根据 seq_len 处理当前 Stack 的有效列和填充值。', 'masking', ['score_masked', 'valid_cols']),
    ], [E('scale', 'mask', 'scaled Scores')],
    { Scores: view('scale', 'attention.scores'), seq: control('mask', 'metadata.seq-lens', 'seq_len 决定当前 Stack 的有效列数。') }, { Scores: view('mask', 'attention.masked-scores') }),
    softmax: frame('online-softmax', [
      D('scores', 'Masked Scores', '770–788', ['score_masked'], '当前 Stack 的 score_aiv 经过 SCALE 与有效列处理后的视图。',
        { role: 'intermediate', granularity: 'stack', entity: 'attention.masked-scores', symbol: 'score_masked', storage: { space: 'Vec' } }),
      S('previous', 'Previous m / l_sum', '745–760', ['m_iter', 'l_iter'], '上轮 Stack 的 running max 和归一化和。',
        { entity: 'softmax.state', granularity: 'stack', dtype: 'FP32', initial: 'm = -3.0e38; l_sum = 0', update: '读取上轮 m_after / l_after', feedback: 'updated → previous，下一轮 Stack' }),
      O('local-max', 'Row Maximum', '789–789', ['pl.row_max'], '计算当前 Stack 的局部最大值。', 'reduction', ['local_m']),
      O('next-max', 'Update Maximum', '790–790', ['pl.maximum'], 'next_m = max(local_m, m_iter)。', 'reduction', ['next_m']),
      O('rescale', 'Rescale', '791–791', ['pl.exp', 'pl.sub'], 'rescale = exp(m_iter − next_m)。', 'elementwise', ['rescale']),
      O('exp', 'Exponential', '792–792', ['pl.exp', 'pl.row_expand_sub'], '计算未除以全局 l_sum 的概率权重。', 'elementwise', ['probability_aiv']),
      D('probability', 'Probability P', '802–815', ['probability_bf16', 'probability_transfer'], '当前 Stack 的 BF16 概率权重写入 probability_transfer，供 PV 消费。',
        { role: 'intermediate', granularity: 'stack', entity: 'attention.probability', symbol: 'probability_bf16', dtype: 'BF16' }),
      O('sum', 'Normalization Sum', '793–801', ['pl.row_sum', 'pl.add', 'pl.mul'], 'next_l = rescale × l_iter + row_sum(probability_reduce)。', 'reduction', ['next_l']),
      S('updated', 'Updated m / l_sum', '791–801、823–825、856–861', ['m_after', 'l_after', 'rescale_after'], 'm/l 更新后反馈到下一轮，rescale 被送入延迟 FIFO，l_sum 最终参与输出归一化。',
        { entity: 'softmax.state', granularity: 'stack', dtype: 'FP32', initial: 'm = -3.0e38; l_sum = 0; pending rescale = 1', update: 'm_after=next_m; l_after=next_l; rescale_after=rescale', feedback: 'updated → previous；rescale → Pending Rescale FIFO' }),
    ], [E('scores', 'local-max', 'Scores'), E('scores', 'exp', 'Scores'), E('local-max', 'next-max', 'local_m'),
      E('previous', 'next-max', 'm_iter', 'state'), E('previous', 'rescale', 'm_iter', 'state'), E('next-max', 'rescale', 'next_m'), E('next-max', 'exp', 'next_m'),
      E('exp', 'probability', 'P'), E('exp', 'sum', 'ΣP'), E('rescale', 'sum', 'rescale'), E('previous', 'sum', 'l_iter', 'state'),
      E('next-max', 'updated', 'next_m', 'state'), E('sum', 'updated', 'next_l', 'state'), E('updated', 'previous', 'next stack', 'feedback', { route: 'feedback' })],
    { Scores: view('scores', 'attention.masked-scores') }, { P: view('probability', 'attention.probability'), state: view('updated', 'softmax.state') }),
    pv: frame('pv', [
      D('probability', 'Probability P', '435–446', ['probability0', 'probability_transfer'], '从 probability_transfer 加载当前 Page 的 BF16 概率 Tile。',
        { role: 'input-view', granularity: 'page', entity: 'attention.probability', symbol: 'probability0', dtype: 'BF16', shape: { allocated: ['ROW_TILE', 'BLOCK_SIZE'], valid: ['GROUP', 'BLOCK_SIZE'] }, storage: { space: 'Mat' } }),
      D('v-page', 'V Page', '364–407', ['v0', 'pv_ph0', 'block_table_2d'], '通过 block_table 读物理页号，加载一个 KV head 的 V Page 到 Mat；v0 显式声明 BF16。',
        { role: 'cache-view', granularity: 'page', entity: 'cache.value.page', symbol: 'v0', dtype: 'BF16', shape: { allocated: ['BLOCK_SIZE', 'HEAD_DIM'] }, storage: { space: 'Mat' }, addressing: cachePayload('value').addressing }),
      O('matmul', 'PV Matmul', '447–718', ['pl.matmul', 'pl.matmul_acc'], 'P 与 V Page 相乘；同一 Stack 的 Page 结果在 FP32 accumulator 中累加。', 'matmul', ['pv0', 'pv3']),
      D('partial', 'PV Partial', '447–451、503–718', ['pv3', 'pv_transfer'], '当前 Stack 的 FP32 Context partial 写入 pv_transfer；这是单轮数据结果，不是跨 Stack 状态。',
        { role: 'intermediate', granularity: 'stack', entity: 'context.partial', symbol: 'pv_transfer', dtype: 'FP32' }),
    ], [E('probability', 'matmul', 'P'), E('v-page', 'matmul', 'V'), E('matmul', 'partial', 'partial')],
    { P: slice('probability', 'attention.probability', '从当前 Stack 的 probability_transfer 取一个 Page 的概率列。'), V: slice('v-page', 'cache.value', '从 V Cache 取当前物理页的 KV head 切片。') }, { partial: view('partial', 'context.partial') }),
    'context-acc': frame('context-accumulate', [
      D('partial', 'PV Partial', '835–841', ['pv_aiv', 'pv_transfer'], '从当前 Stack 的 PV 传输区加载局部 partial 到 Vec；此处 load 未显式声明外部 buffer dtype。',
        { role: 'input-view', granularity: 'stack', entity: 'context.partial', symbol: 'pv_aiv', shape: { allocated: ['AIV_ROW_TILE', 'HEAD_DIM'], valid: ['lane_rows', 'HEAD_DIM'] }, storage: { space: 'Vec' } }),
      S('pending', 'Pending Rescale', '754–760、848–861', ['pending0_iter', 'pending1_iter', 'rescale_after'], '两个 rescale 槽位的 FIFO 状态；它承担延迟对齐，不是 PV 结果数据。',
        { entity: 'context.pending-rescale', granularity: 'stack', initial: 'pending0 = 1; pending1 = 1', update: 'pending0_after=pending1_iter; pending1_after=rescale_after', feedback: 'pl.range init_values 与 pl.yield_ 传递到下一 tick' }),
      S('old', 'Previous Context', '748–760、842–861', ['o_iter', 'o_after'], '跨 Stack 保存的 Context accumulator。',
        { entity: 'context.o', granularity: 'stack', dtype: 'FP32', initial: 'o = 0', update: 'o_after = o_iter × pending0_iter + pv_aiv', feedback: 'add → old，下一轮 Stack' }),
      O('rescale', 'Rescale Context', '842–843', ['pl.row_expand_mul'], '旧 Context 乘 pending0。', 'elementwise', ['o_iter', 'pending0_iter']),
      O('add', 'Accumulate', '842–844', ['pl.add'], '把当前 Stack 的 PV partial 加到重缩放后的 Context。', 'accumulation', ['o_after']),
    ], [E('pending', 'rescale', 'pending0', 'state'), E('old', 'rescale', 'o', 'state'), E('rescale', 'add', 'rescaled o'), E('partial', 'add', 'PV partial'), E('add', 'old', 'next stack o', 'feedback', { route: 'feedback' })],
    { partial: view('partial', 'context.partial'), state: aggregate('pending', 'softmax.state', 'rescale_after 经双槽 FIFO 延迟为 pending0_iter/pending1_iter。') }, { Context: view('add', 'context.o') }),
    'output-norm': frame('output-normalization', [
      O('normalize', 'Final Normalize', '864–865', ['pl.recip', 'pl.row_expand_mul'], 'Context o 除以最终 l_sum。', 'normalization', ['o', 'l_sum']),
      O('cast', 'Cast BF16', '864–868', ['pl.cast'], '归一化 Context 转 BF16。', 'cast', ['ctx']),
      O('store', 'Store Output', '869–873', ['pl.store'], '按当前 Batch/KV 组及 AIV lane 写回 Attention Output。', 'store', ['ctx', 'out2d']),
    ], [E('normalize', 'cast', 'Context'), E('cast', 'store', 'BF16')],
    { Context: control('normalize', 'context.normalization-state', '输出归一化同时读取最终 o 与 l_sum。') }, { Output: view('store', 'attention.output') }),
  };

  const nodes = [
    D('q-proj', 'Q Projection', '68–94、215–227', ['q_proj'], '外部投影结果，作为 Q 归一化输入；此函数签名未声明 dtype 或完整 shape。', { role: 'input', granularity: 'tensor', entity: 'projection.q', symbol: 'q_proj' }),
    D('k-proj', 'K Projection', '68–94、162–173', ['k_proj'], '外部投影结果，作为 K 归一化输入；dtype 或完整 shape 未显式声明。', { role: 'input', granularity: 'tensor', entity: 'projection.k', symbol: 'k_proj' }),
    D('v-proj', 'V Projection', '68–94、200–207', ['v_proj'], '外部 V 投影结果，经 inv_rms 缩放后转 BF16。', { role: 'input', granularity: 'tensor', entity: 'projection.v', symbol: 'v_proj' }),
    C('q-prep', 'Q Norm → RoPE', '215–251', ['q_raw', 'q_heads', 'q_rotated', 'q2d'], 'Q 乘 inv_rms、按 head 归一化，再施加 RoPE，写入 q2d。', 'normalization', ['pl.row_sum', 'pl.recip', 'pl.concat'], subgraphs['q-prep']),
    C('k-prep', 'K Norm → RoPE', '162–198', ['k_raw', 'k_normed', 'k_rotated'], 'K 乘 inv_rms、按 head 归一化，再施加 RoPE。', 'normalization', ['pl.row_sum', 'pl.recip', 'pl.concat'], subgraphs['k-prep']),
    C('v-scale', 'V Scaling', '200–213', ['v_row_bf16'], 'V 按 inv_rms 缩放并转 BF16。', 'elementwise', ['pl.mul', 'pl.cast'], subgraphs['v-scale']),
    D('q', 'Q', '247–251', ['q_rotated', 'q2d'], '当前 token 的 Q 组转 BF16 后写入 q2d；后续由 q_tile 消费。dtype 描述写入载荷，不声明外部 q2d 存储的类型。', { role: 'intermediate', granularity: 'token-group', entity: 'attention.q', dtype: 'BF16' }),
    D('k', 'K', '181–198', ['k_rotated'], '当前 token 的 K 载荷转 BF16 并追加到 Key Cache。', { role: 'intermediate', granularity: 'token', entity: 'attention.k', dtype: 'BF16' }),
    D('v', 'V', '200–213', ['v_row_bf16'], '当前 token 的 BF16 V 载荷将追加到 Value Cache。', { role: 'intermediate', granularity: 'token', entity: 'attention.v', symbol: 'v_row_bf16', dtype: 'BF16' }),
    D('aux', 'Aux Inputs', '68–94、146–155、261–267', ['seq_lens', 'block_table', 'slot_mapping'], 'seq_lens 决定有效长度，slot_mapping 决定当前 token 写入位置，block_table 提供逻辑页到物理页映射。', { role: 'metadata', granularity: 'bundle', entity: 'metadata.bundle' }),
    C('cache-append', 'Cache Append', '194–198、210–213', ['key_cache_bsnd', 'value_cache_bsnd'], '当前 token 的 K/V 载荷按 cache_row 写入对应 Cache。', 'store', [], subgraphs['cache-append']),
    keyCache('k-cache'), valueCache('v-cache'),
    C('qk', 'QK Matmul', '269–276、285–305、306–356', ['q_tile', 'qk_k_page', 'qk_score_page', 'score_transfer'], 'Q 与映射后的 K Page 相乘；每页 FP32 结果写入当前 Stack 的 Scores 区间。', 'matmul', ['pl.load', 'pl.matmul'], subgraphs.qk),
    D('scores', 'Scores', '278–305、770–776', ['score_transfer', 'score_aiv'], '多个 Page 的 QK Scores 按列区间写入 score_transfer，再按整个 Stack 加载到 AIV。', { role: 'intermediate', granularity: 'stack', entity: 'attention.scores', symbol: 'score_transfer', dtype: 'FP32',
      relations: [{ kind: 'aggregate', entity: 'attention.scores', description: 'qk/score 每页结果按 qk_page_offset * BLOCK_SIZE 写入本 Stack。' }] }),
    C('scale-mask', 'Scaling + Mask', '777–788', ['score_scaled', 'score_masked', 'valid_cols'], '缩放 Scores，再根据 seq_len 处理末 Stack 的无效列。', 'masking', ['pl.tile.muls', 'pl.set_validshape', 'pl.fillpad'], subgraphs['scale-mask']),
    C('softmax', 'Online Softmax', '745–825、856–861', ['m_iter', 'l_iter', 'probability_bf16', 'm_after', 'l_after'], '跨 Stack 更新 running max 和归一化和，输出 BF16 概率权重及 rescale 状态。', 'normalization', ['pl.row_max', 'pl.exp', 'pl.row_sum'], subgraphs.softmax),
    D('probability', 'Probability P', '802–815', ['probability_bf16', 'probability_transfer'], '本 Stack 的 BF16 权重尚未除以全局 l_sum，供 PV 消费。', { role: 'intermediate', granularity: 'stack', entity: 'attention.probability', symbol: 'probability_bf16', dtype: 'BF16' }),
    C('pv', 'PV Matmul', '364–718', ['probability0', 'v0', 'pv3', 'pv_transfer'], 'P 与映射后的 V Page 相乘，Stack 内各页结果累加为 FP32 partial。', 'matmul', ['pl.load', 'pl.matmul', 'pl.matmul_acc'], subgraphs.pv),
    C('context-acc', 'Context Accumulate', '827–861', ['pv_aiv', 'o_iter', 'o_after', 'pending0_iter'], '旧 Context 乘延迟 rescale，再加当前 Stack 的 PV partial。', 'accumulation', ['pl.row_expand_mul', 'pl.add'], subgraphs['context-acc']),
    S('context', 'Context', '745–760、842–861', ['o', 'l_sum'], '跨 Stack 累积的 o 与归一化和 l_sum，是最终输出归一化的状态，不是单轮 PV 数据。',
      { entity: 'context.normalization-state', granularity: 'stack', dtype: 'FP32', initial: 'o = 0; l_sum = 0', update: 'o = o_after; l_sum = l_after', feedback: 'pl.range init_values / pl.yield_ 跨 tick 传递' }),
    C('output-norm', 'Output Normalize', '864–873', ['o', 'l_sum', 'ctx', 'out2d'], '所有 Stack 完成后，以最终 l_sum 归一化 o，转 BF16 并写回 out2d。', 'normalization', ['pl.recip', 'pl.row_expand_mul', 'pl.cast', 'pl.store'], subgraphs['output-norm']),
    D('output', 'Attention Output', '864–873', ['ctx', 'out2d'], '最终 BF16 Attention 载荷按 Batch、KV head、GROUP 和 lane 对应行写回。dtype 描述 ctx 载荷。', { role: 'output', granularity: 'token-group', entity: 'attention.output', symbol: 'ctx', dtype: 'BF16' }),
  ];
  const aux = nodes.find((node) => node.id === 'aux');
  aux.ports.outputs = { slot: { entity: 'metadata.slot' }, block: { entity: 'metadata.block-table' }, seq: { entity: 'metadata.seq-lens' } };
  const ports = {
    'q-proj>q-prep': { targetPort: 'Q' }, 'q-prep>q': { sourcePort: 'Q' }, 'k-proj>k-prep': { targetPort: 'K' }, 'k-prep>k': { sourcePort: 'K' },
    'v-proj>v-scale': { targetPort: 'V' }, 'v-scale>v': { sourcePort: 'V' }, 'k>cache-append': { targetPort: 'K' }, 'v>cache-append': { targetPort: 'V' },
    'aux>cache-append': { sourcePort: 'slot', targetPort: 'slot' }, 'cache-append>k-cache': { sourcePort: 'K' }, 'cache-append>v-cache': { sourcePort: 'V' },
    'aux>qk': { sourcePort: 'block', targetPort: 'block' }, 'q>qk': { targetPort: 'Q' }, 'k-cache>qk': { targetPort: 'K' },
    'qk>scores': { sourcePort: 'Scores' }, 'scores>scale-mask': { targetPort: 'Scores' }, 'aux>scale-mask': { sourcePort: 'seq', targetPort: 'seq' },
    'scale-mask>softmax': { sourcePort: 'Scores', targetPort: 'Scores' }, 'softmax>probability': { sourcePort: 'P' },
    'softmax>context-acc': { sourcePort: 'state', targetPort: 'state' }, 'softmax>context': { sourcePort: 'state' },
    'probability>pv': { targetPort: 'P' }, 'v-cache>pv': { targetPort: 'V' }, 'pv>context-acc': { sourcePort: 'partial', targetPort: 'partial' },
    'context-acc>context': { sourcePort: 'Context' }, 'context>output-norm': { targetPort: 'Context' }, 'output-norm>output': { sourcePort: 'Output' },
  };
  const edge = (s, t, label, type = 'data', extra = {}) => E(s, t, label, type, { ...ports[`${s}>${t}`], ...extra });
  const edges = [
    edge('q-proj', 'q-prep', 'Q Projection'), edge('q-prep', 'q', 'Q'), edge('k-proj', 'k-prep', 'K Projection'), edge('k-prep', 'k', 'K'), edge('v-proj', 'v-scale', 'V Projection'), edge('v-scale', 'v', 'V'),
    edge('k', 'cache-append', 'K'), edge('v', 'cache-append', 'V'), edge('aux', 'cache-append', 'slot_mapping', 'control'), edge('aux', 'qk', 'block_table', 'control'), edge('aux', 'scale-mask', 'seq_lens', 'control'),
    edge('cache-append', 'k-cache', 'K'), edge('cache-append', 'v-cache', 'V'), edge('q', 'qk', 'Q'), edge('k-cache', 'qk', 'K · block_table'), edge('qk', 'scores', 'Scores'), edge('scores', 'scale-mask', 'Scores'),
    edge('scale-mask', 'softmax', 'scaled / masked Scores'), edge('softmax', 'probability', 'P'), edge('probability', 'pv', 'P'), edge('v-cache', 'pv', 'V · block_table'), edge('pv', 'context-acc', 'PV partial'),
    edge('softmax', 'context-acc', 'rescale', 'state', { route: 'state' }), edge('softmax', 'context', 'l_sum', 'state', { route: 'state' }),
    edge('context-acc', 'context', 'Context', 'state'), edge('context', 'output-norm', 'Context', 'state'), edge('output-norm', 'output', 'BF16 Output'),
  ];
  // Boundary contracts inherit only known properties of the same semantic entity.
  // Slice / aggregate mappings then validate the refinement inside each Composite.
  edges.forEach((edge) => {
    const from = nodes.find((node) => node.id === edge.s), to = nodes.find((node) => node.id === edge.t);
    for (const [boundary, peer, side, port] of [[from, to, 'outputs', edge.sourcePort], [to, from, 'inputs', edge.targetPort]]) {
      if (boundary.type !== 'composite' || !peer.data) continue;
      const contract = boundary.ports[side][port];
      if (contract.entity !== peer.data.entity) continue;
      for (const key of ['dtype', 'granularity']) if (peer.data[key]) contract[key] = peer.data[key];
    }
  });
  for (const [id, primaryRange] of [['qk', 1], ['k-cache', 2], ['v-cache', 2], ['context', 1]]) {
    nodes.find((node) => node.id === id).source.primaryRange = primaryRange;
  }
  const region = (id, name, members) => ({ id, type: 'region', name, description: name, members, source: source('68–874'), provenance: [{ kind: 'resolved', origin: 'source', fields: ['members'], note: '按算子语义组织的现有区域分组；不是硬件执行分区。' }] });
  const model = { schemaVersion: 2, id: 'paged-attention', sourceFiles: { [FILE]: { lineCount: (root.PTO_PAGED_ATTENTION_PYPTO_SOURCE || '').trimEnd().split('\n').length } }, nodes, edges, groups: [
    region('qkv-preparation', 'QKV Preparation', ['q-proj', 'k-proj', 'v-proj', 'q-prep', 'k-prep', 'v-scale', 'q', 'k', 'v']),
    region('paged-attention', 'Paged Attention', ['cache-append', 'k-cache', 'v-cache', 'qk', 'scores', 'scale-mask', 'softmax', 'probability', 'pv', 'context-acc']),
    region('output-region', 'Output', ['context', 'output-norm', 'output']),
  ] };
  root.PTO_PYPA_SEMANTIC_GRAPH = model;
  root.PTO_PYPA_SEMANTIC_SUBGRAPHS = Object.fromEntries(Object.entries(subgraphs).map(([id, bundle]) => [id, bundle.graph]));
})(typeof window === 'undefined' ? globalThis : window);
