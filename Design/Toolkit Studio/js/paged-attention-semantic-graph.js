/* Source-grounded semantic refinements for paged_attention_pypto.py.
   Nodes, typed edges and ports carry the meaning; layout is computed separately. */
(function (root) {
  'use strict';
  const n = (id, semanticType, title, lines, api, detail, extra = {}) => ({
    id, semanticType, title, sub: api, lines, api, detail,
    evidence: semanticType === 'data' || semanticType === 'state' ? '源码数据或状态对象'
      : semanticType === 'composite' ? '源码操作的语义归并' : '源码显式计算',
    ...extra,
  });
  const e = (s, t, label, type = 'data', extra = {}) => ({ s, t, label, type, ...extra });
  const frame = (nodes, edges, inputs, outputs) => ({ nodes, edges, inputs, outputs });

  const qNorm = frame([
    n('sum', 'operation', 'Row Sum', '229–229', 'pl.row_sum', 'Q 的逐行平方和。'),
    n('inv', 'operation', 'Inverse RMS', '230–230', 'pl.recip · pl.sqrt', '由平方和得到归一化系数。'),
    n('weight', 'operation', 'Norm Weight', '231–234', 'pl.mul', '应用 q_norm_w。'),
  ], [e('sum','inv','sum'), e('inv','weight','inv_rms')],
  { default: 'sum' }, { default: 'weight' });
  const kNorm = frame([
    n('sum', 'operation', 'Row Sum', '175–175', 'pl.row_sum', 'K 的逐行平方和。'),
    n('inv', 'operation', 'Inverse RMS', '176–176', 'pl.recip · pl.sqrt', '由平方和得到归一化系数。'),
    n('weight', 'operation', 'Norm Weight', '177–180', 'pl.mul', '应用 k_norm_w。'),
  ], [e('sum','inv','sum'), e('inv','weight','inv_rms')],
  { default: 'sum' }, { default: 'weight' });

  const subgraphs = {
    'q-prep': frame([
      n('norm', 'composite', 'Q RMSNorm', '216–234', 'pl.row_sum · pl.recip · pl.mul', '按 head 对 Q 做 RMSNorm。', { subgraph: qNorm }),
      n('rope', 'operation', 'Q RoPE', '235–246', 'pl.concat · pl.mul', 'cos/sin 旋转 Q 的半维。'),
      n('write', 'operation', '写入 Q', '247–250', 'pl.cast · q2d', 'BF16 Q 写入 q2d；不写 KV Cache。'),
    ], [e('norm','rope','normalized Q'), e('rope','write','rotated Q')],
    { Q: 'norm' }, { Q: 'write' }),

    'k-prep': frame([
      n('norm', 'composite', 'K RMSNorm', '162–180', 'pl.row_sum · pl.recip · pl.mul', '按 head 对 K 做 RMSNorm。', { subgraph: kNorm }),
      n('rope', 'operation', 'K RoPE', '181–193', 'pl.concat · pl.mul', 'cos/sin 旋转 K 的半维。'),
      n('cast', 'operation', 'BF16 K', '194–198', 'pl.cast', 'K 转 BF16 后写入 Key Cache。'),
    ], [e('norm','rope','normalized K'), e('rope','cast','rotated K')],
    { K: 'norm' }, { K: 'cast' }),

    'v-scale': frame([
      n('scale', 'operation', 'V × inv_rms', '200–204', 'pl.mul', 'V 投影与 inv_rms 相乘。'),
      n('cast', 'operation', 'BF16 V', '200–213', 'pl.cast', '缩放后的 V 转 BF16。'),
    ], [e('scale','cast','scaled V')], { V: 'scale' }, { V: 'cast' }),

    'cache-append': frame([
      n('slot', 'data', '写入位置', '146–155', 'slot_mapping', 'slot_mapping 确定当前 token 的 cache 行。'),
      n('write-k', 'operation', '追加 K', '194–198', 'key_cache_bsnd[...]', '仅 K 写入 Key Cache。'),
      n('write-v', 'operation', '追加 V', '210–213', 'value_cache_bsnd[...]', '仅 V 写入 Value Cache。'),
      n('key', 'data', 'Key Cache', '194–198', 'key_cache_bsnd', '更新后的分页 Key Cache。'),
      n('value', 'data', 'Value Cache', '210–213', 'value_cache_bsnd', '更新后的分页 Value Cache。'),
    ], [e('slot','write-k','slot'), e('slot','write-v','slot'), e('write-k','key','K'), e('write-v','value','V')],
    { K: 'write-k', V: 'write-v', slot: 'slot' }, { K: 'key', V: 'value' }),

    qk: frame([
      n('q-tile', 'data', 'Q Tile', '270–275', 'pl.load · q_tile', 'QK 使用的当前 Q tile。'),
      n('k-page', 'data', 'K Page', '285–295', 'block_table_2d · key_cache_bsnd', '通过 block_table 读取物理 K page。'),
      n('matmul', 'operation', 'QK Matmul', '296–300', 'pl.matmul', 'Q tile 与 K page 转置相乘。'),
      n('score', 'data', 'FP32 Scores', '301–305', 'score_transfer', '当前 page 的 QK scores。'),
    ], [e('q-tile','matmul','Q'), e('k-page','matmul','Kᵀ'), e('matmul','score','Scores')],
    { Q: 'q-tile', K: 'k-page', block: 'k-page' }, { Scores: 'score' }),

    'scale-mask': frame([
      n('scale', 'operation', 'Scaling', '777–777', 'pl.tile.muls', 'Scores 乘 1/√HEAD_DIM。'),
      n('mask', 'operation', 'Valid Columns Mask', '778–788', 'pl.set_validshape · pl.fillpad', '按 seq_len 屏蔽末 stack 无效列。'),
    ], [e('scale','mask','scaled Scores')],
    { Scores: 'scale', seq: 'mask' }, { Scores: 'mask' }),

    softmax: frame([
      n('scores', 'data', 'Masked Scores', '777–788', 'score_masked', '当前 stack 的有效 score。'),
      n('previous', 'state', 'Previous m / l_sum', '745–760', 'm_iter · l_iter', '上轮 stack 保存的最大值与归一化和。'),
      n('local-max', 'operation', 'Row Maximum', '789–789', 'pl.row_max', '计算当前 stack 的局部最大值。'),
      n('next-max', 'operation', 'Update Maximum', '790–790', 'pl.maximum', 'next_m = max(local_m, m_iter)。'),
      n('rescale', 'operation', 'Rescale', '791–791', 'pl.exp · pl.sub', '旧状态按 exp(m_iter−next_m) 重缩放。'),
      n('exp', 'operation', 'Exponential', '792–792', 'pl.exp · pl.row_expand_sub', '计算当前 score 的未归一化概率。'),
      n('probability', 'data', 'Probability P', '802–815', 'probability_transfer', '当前 stack 的 BF16 P，供 PV 消费。'),
      n('sum', 'operation', 'Normalization Sum', '794–800', 'pl.row_sum · pl.add', 'next_l = rescale·l_iter + ΣP。'),
      n('updated', 'state', 'Updated m / l_sum', '823–861', 'm_after · l_after', '状态传递到后续 stack；l_sum 也用于输出归一化。'),
    ], [
      e('scores','local-max','Scores'), e('scores','exp','Scores'),
      e('local-max','next-max','local_m'), e('previous','next-max','m_iter','state'),
      e('previous','rescale','m_iter','state'), e('next-max','rescale','next_m'),
      e('next-max','exp','next_m'), e('exp','probability','P'),
      e('exp','sum','ΣP'), e('rescale','sum','rescale'), e('previous','sum','l_iter','state'),
      e('next-max','updated','next_m','state'), e('sum','updated','next_l','state'),
      e('updated','previous','next stack','state', { route: 'feedback' }),
    ], { Scores: 'scores' }, { P: 'probability', state: 'updated' }),

    pv: frame([
      n('probability', 'data', 'Probability P', '435–447', 'pl.load · probability_transfer', '从 probability_transfer 读取当前 stack 的 BF16 概率。'),
      n('v-page', 'data', 'V Page', '369–405', 'block_table_2d · value_cache_bsnd', '按 block_table 读取物理 V page。'),
      n('matmul', 'operation', 'PV Matmul', '405–718', 'pl.matmul · pl.matmul_acc', 'P 与 V page 相乘，stack 内累积。'),
      n('partial', 'data', 'PV Partial', '503–718', 'pv_transfer', '当前 stack 的 FP32 context partial。'),
    ], [e('probability','matmul','P'), e('v-page','matmul','V'), e('matmul','partial','partial')],
    { P: 'probability', V: 'v-page' }, { partial: 'partial' }),

    'context-acc': frame([
      n('partial', 'data', 'PV Partial', '830–840', 'pv_transfer · pv_aiv', '当前 stack 的 PV partial。'),
      n('pending', 'state', 'Pending Rescale', '848–855', 'pending0_iter · pending1_iter', '跨 stack 延迟的 rescale 状态。'),
      n('old', 'state', 'Previous Context', '758–760', 'o_iter', '上轮累计 context。'),
      n('rescale', 'operation', 'Rescale Context', '842–843', 'pl.row_expand_mul', '旧 context 乘 pending0。'),
      n('add', 'operation', 'Accumulate', '843–844', 'pl.add', '加上当前 PV partial。'),
    ], [e('pending','rescale','pending0','state'), e('old','rescale','o','state'), e('rescale','add','rescaled o'), e('partial','add','PV partial'), e('add','old','next stack o','state', { route: 'feedback' })],
    { partial: 'partial', state: 'pending' }, { Context: 'add' }),

    'output-norm': frame([
      n('normalize', 'operation', 'Final Normalize', '864–865', 'pl.recip · pl.row_expand_mul', 'Context o 除以最终 l_sum。'),
      n('cast', 'operation', 'Cast BF16', '864–868', 'pl.cast', '归一化 context 转 BF16。'),
      n('store', 'operation', 'Store Output', '869–872', 'pl.store · out2d', '写回最终 Attention Output。'),
    ], [e('normalize','cast','Context'), e('cast','store','BF16')],
    { Context: 'normalize' }, { Output: 'store' }),
  };

  root.PTO_PYPA_SEMANTIC_SUBGRAPHS = subgraphs;
})(typeof window === 'undefined' ? globalThis : window);
