/**
 * 推理性能分析 · 模拟采集数据
 *
 * 数据口径（全部为演示用模拟值，但内部自洽）：
 *   Qwen3-14B · hidden 5120 · 40 layers · Q40/KV8 · head_dim 128 · FFN 17408 · vocab 152064
 *   BF16 权重 · batch 16 · 平均 seq 1614 · page 128 token
 *   Ascend 950B · HBM 64 GB / 3.6 TB/s peak · Cube BF16 800 TFLOPS peak
 *
 * 关键恒等式（改数时请一起改）：
 *   每层 360.0 us x 40 = 14.400 ms + 边界 0.605 ms + Host 空隙 0.195 ms = TPOT 15.200 ms
 *   权重 27.99 GB + KV 4.23 GB + 激活 0.42 GB = 32.64 GB / step
 *   32.64 GB / 15.2 ms = 2.147 TB/s 达成 = 59.6% 峰值；理论下界 32.64 / 3.6 = 9.07 ms
 */
(function registerInferenceProfileData() {
  'use strict';

  const PEAK_BW = 3.6;       // TB/s
  const PEAK_FLOPS = 800;    // TFLOPS · BF16 Cube
  const TPOT = 15.2;         // ms · p50

  // 每请求 seq_len（16 路），合计 25,824 -> 平均 1,614
  const seqLens = [1893, 2048, 1367, 1544, 2048, 1211, 1832, 1207, 2048, 743, 1655, 2048, 1389, 1920, 866, 2005];

  /** 确定性伪随机，保证每次刷新图形一致 */
  function jitter(seed, n, base, spread) {
    const out = [];
    let s = seed;
    for (let i = 0; i < n; i += 1) {
      s = (s * 1103515245 + 12345) % 2147483648;
      out.push(base * (1 + ((s / 2147483648) - 0.5) * spread));
    }
    return out;
  }

  /** 逐层曲线：首层 cache 冷、尾层要喂给输出边界，两端偏高 */
  function perLayer(seed, baseUs, spread, headBoost, tailBoost) {
    const values = jitter(seed, 40, baseUs, spread);
    values[0] *= headBoost;
    values[1] *= 1 + (headBoost - 1) * 0.35;
    values[39] *= tailBoost;
    return values.map((v) => Number(v.toFixed(2)));
  }

  const ops = [
    {
      id: 'gate-up-proj', name: 'gate_proj · up_proj', scope: 'Scope 3 · MLP', group: 'mlp',
      calls: 40, totalMs: 5.244, perLayerUs: 131.1, share: 34.5,
      units: { cube: 5.1, vector: 3.2, mte2: 89.4, mte3: 1.1, sync: 1.2 },
      bound: 'mte2', boundLabel: 'MTE2', efficiency: 94,
      bytesIn: 14.264, bytesOut: 0.084, reuse: 1.0,
      gflop: 228.0, achievedTflops: 43.5, achievedBw: 2.72, ai: 16.0,
      cores: 64, imbalance: 0.06,
      perLayer: perLayer(7, 131.1, 0.05, 1.09, 1.02),
      source: 'decode_layer.py:612-688',
      static: [
        ['UB 峰值占用', '58%（编译期预算）', '63%', '+5 pt', 'warn'],
        ['Split-K tile 数', '5', '5', '一致', 'ok'],
        ['达成带宽', '2.66 TB/s（估）', '2.72 TB/s', '+2.3%', 'ok'],
      ],
      note: '权重搬运完全主导：89.4% 时间花在 MTE2。已贴近内存屋顶，除非改精度或做权重复用，否则无进一步空间。',
    },
    {
      id: 'down-proj', name: 'down_proj', scope: 'Scope 3 · MLP', group: 'mlp',
      calls: 40, totalMs: 2.640, perLayerUs: 66.0, share: 17.4,
      units: { cube: 5.0, vector: 2.8, mte2: 88.7, mte3: 2.1, sync: 1.4 },
      bound: 'mte2', boundLabel: 'MTE2', efficiency: 93,
      bytesIn: 7.132, bytesOut: 0.131, reuse: 1.0,
      gflop: 114.0, achievedTflops: 43.2, achievedBw: 2.70, ai: 16.0,
      cores: 64, imbalance: 0.07,
      perLayer: perLayer(11, 66.0, 0.05, 1.08, 1.02),
      source: 'decode_layer.py:742-801',
      static: [
        ['UB 峰值占用', '54%（编译期预算）', '55%', '+1 pt', 'ok'],
        ['Split-K/N 拆分', '17 × 5', '17 × 5', '一致', 'ok'],
        ['达成带宽', '2.64 TB/s（估）', '2.70 TB/s', '+2.3%', 'ok'],
      ],
      note: '与 gate/up 同形态，FP32 atomic 累加的写回量略高，MTE3 占比 2.1%。',
    },
    {
      id: 'fa-fused', name: 'fa_fused', scope: 'Scope 2 · Attention', group: 'attn',
      calls: 65920, totalMs: 2.276, perLayerUs: 56.9, share: 15.0,
      units: { cube: 12.8, vector: 21.4, mte2: 58.1, mte3: 3.2, sync: 4.5 },
      bound: 'mte2', boundLabel: 'MTE2', efficiency: 52,
      bytesIn: 4.240, bytesOut: 0.211, reuse: 3.2,
      gflop: 21.2, achievedTflops: 9.3, achievedBw: 1.86, ai: 5.0,
      cores: 64, imbalance: 0.18,
      perLayer: perLayer(23, 56.9, 0.09, 1.14, 1.05),
      source: 'decode_layer.py:301-398',
      static: [
        ['真实 KV block 数', '256（MCB 静态上界）', '206', '−19.5%', 'ok'],
        ['UB 峰值占用', '58%（编译期预算）', '63%', '+5 pt', 'warn'],
        ['单 work item 时延', '1.76 μs（估）', '2.21 μs', '+25.6%', 'warn'],
        ['达成带宽', '2.60 TB/s（估）', '1.86 TB/s', '−28.5%', 'bad'],
      ],
      note: 'paged K/V 的非连续访存使 MTE2 只跑到 51.7% 峰值，而同批投影算子在 70–76%。这是当前最大的可回收项。',
    },
    {
      id: 'qkv-proj', name: 'q_proj · k_proj · v_proj', scope: 'Scope 1 · 投影', group: 'proj',
      calls: 40, totalMs: 1.124, perLayerUs: 28.1, share: 7.4,
      units: { cube: 4.9, vector: 3.1, mte2: 87.2, mte3: 2.6, sync: 2.2 },
      bound: 'mte2', boundLabel: 'MTE2', efficiency: 92,
      bytesIn: 2.936, bytesOut: 0.094, reuse: 1.0,
      gflop: 47.0, achievedTflops: 41.8, achievedBw: 2.61, ai: 16.0,
      cores: 64, imbalance: 0.09,
      perLayer: perLayer(31, 28.1, 0.06, 1.10, 1.01),
      source: 'decode_layer.py:429-505',
      static: [
        ['并行拆分', 'Q 10×5 · K/V 2×5', '一致', '一致', 'ok'],
        ['UB 峰值占用', '49%（编译期预算）', '51%', '+2 pt', 'ok'],
        ['达成带宽', '2.58 TB/s（估）', '2.61 TB/s', '+1.2%', 'ok'],
      ],
      note: '读取上一层 dcr_xgamma 预生成的 BF16 normed_in，省掉一次层内 cast。',
    },
    {
      id: 'out-proj', name: 'out_proj', scope: 'Scope 3 · 投影', group: 'proj',
      calls: 40, totalMs: 0.824, perLayerUs: 20.6, share: 5.4,
      units: { cube: 4.8, vector: 3.0, mte2: 86.5, mte3: 3.1, sync: 2.6 },
      bound: 'mte2', boundLabel: 'MTE2', efficiency: 91,
      bytesIn: 2.096, bytesOut: 0.131, reuse: 1.0,
      gflop: 33.6, achievedTflops: 40.8, achievedBw: 2.54, ai: 16.0,
      cores: 64, imbalance: 0.10,
      perLayer: perLayer(41, 20.6, 0.06, 1.09, 1.01),
      source: 'decode_layer.py:518-577',
      static: [
        ['并行拆分', '10 × 5 split-N/K', '一致', '一致', 'ok'],
        ['UB 峰值占用', '46%（编译期预算）', '47%', '+1 pt', 'ok'],
        ['达成带宽', '2.55 TB/s（估）', '2.54 TB/s', '−0.4%', 'ok'],
      ],
      note: '规模最小的投影，固定开销占比略高，达成带宽因此比 gate/up 低约 7%。',
    },
    {
      id: 'rms-lm-head', name: 'rms_lm_head', scope: '输出边界', group: 'boundary',
      calls: 1, totalMs: 0.598, perLayerUs: null, share: 3.9,
      units: { cube: 6.2, vector: 8.4, mte2: 84.9, mte3: 0.3, sync: 0.2 },
      bound: 'mte2', boundLabel: 'MTE2', efficiency: 91,
      bytesIn: 1.556, bytesOut: 0.005, reuse: 1.0,
      gflop: 24.9, achievedTflops: 41.6, achievedBw: 2.60, ai: 16.0,
      cores: 64, imbalance: 0.05,
      perLayer: null,
      source: 'decode_layer.py:1204-1266',
      static: [
        ['词表投影规模', '5120 × 152064', '一致', '一致', 'ok'],
        ['达成带宽', '2.58 TB/s（估）', '2.60 TB/s', '+0.8%', 'ok'],
      ],
      note: '整个 step 只执行一次，但 1.56 GB 的词表权重让它单独占了 3.9%。',
    },
    {
      id: 'silu', name: 'silu', scope: 'Scope 3 · MLP', group: 'mlp',
      calls: 40, totalMs: 0.352, perLayerUs: 8.8, share: 2.3,
      units: { cube: 0, vector: 62.3, mte2: 24.1, mte3: 11.4, sync: 2.2 },
      bound: 'vector', boundLabel: 'Vector', efficiency: 78,
      bytesIn: 0.136, bytesOut: 0.045, reuse: 1.4,
      gflop: 0.13, achievedTflops: 0.37, achievedBw: 0.51, ai: 0.96,
      cores: 64, imbalance: 0.12,
      perLayer: perLayer(53, 8.8, 0.07, 1.06, 1.01),
      source: 'decode_layer.py:704-736',
      static: [
        ['延迟应用 post_inv_rms', '合并进 SwiGLU', '一致', '一致', 'ok'],
        ['UB 峰值占用', '38%（编译期预算）', '39%', '+1 pt', 'ok'],
      ],
      note: 'Vector-bound 的纯 elementwise 段，已把 post-RMS 缩放折叠进来，省掉一次全量读写。',
    },
    {
      id: 'online-softmax', name: 'online_softmax', scope: 'Scope 2 · Attention', group: 'attn',
      calls: 5120, totalMs: 0.344, perLayerUs: 8.6, share: 2.3,
      units: { cube: 0, vector: 71.5, mte2: 18.2, mte3: 6.1, sync: 4.2 },
      bound: 'vector', boundLabel: 'Vector', efficiency: 64,
      bytesIn: 0.104, bytesOut: 0.021, reuse: 2.1,
      gflop: 0.50, achievedTflops: 1.45, achievedBw: 0.36, ai: 4.8,
      cores: 64, imbalance: 0.31,
      perLayer: perLayer(61, 8.6, 0.11, 1.12, 1.03),
      source: 'decode_layer.py:404-460',
      static: [
        ['work items', 'BATCH × NUM_KV_HEADS = 128', '128', '一致', 'ok'],
        ['负载不均衡度 CV', '0.10（假设均匀）', '0.31', '+0.21', 'bad'],
        ['UB 峰值占用', '31%（编译期预算）', '34%', '+3 pt', 'warn'],
      ],
      note: '每个 work item 要归并的块数与该请求 seq_len 成正比，ragged 输入直接变成 0.31 的负载倾斜。',
    },
    {
      id: 'residual-cast', name: 'residual_rms_cast', scope: 'Scope 3 · Norm', group: 'norm',
      calls: 40, totalMs: 0.272, perLayerUs: 6.8, share: 1.8,
      units: { cube: 0, vector: 51.4, mte2: 33.2, mte3: 12.1, sync: 3.3 },
      bound: 'vector', boundLabel: 'Vector', efficiency: 72,
      bytesIn: 0.072, bytesOut: 0.031, reuse: 1.2,
      gflop: 0.05, achievedTflops: 0.18, achievedBw: 0.38, ai: 0.48,
      cores: 64, imbalance: 0.09,
      perLayer: perLayer(67, 6.8, 0.06, 1.05, 1.01),
      source: 'decode_layer.py:589-606',
      static: [['UB 峰值占用', '29%（编译期预算）', '30%', '+1 pt', 'ok']],
      note: '一次读取同时产出 FP32 残差与 BF16 的 MLP 输入，避免二次扫描。',
    },
    {
      id: 'dcr-xgamma', name: 'dcr_xgamma', scope: 'Scope 3 · Carry', group: 'norm',
      calls: 200, totalMs: 0.264, perLayerUs: 6.6, share: 1.7,
      units: { cube: 0, vector: 48.2, mte2: 39.4, mte3: 9.8, sync: 2.6 },
      bound: 'vector', boundLabel: 'Vector', efficiency: 74,
      bytesIn: 0.061, bytesOut: 0.038, reuse: 1.1,
      gflop: 0.04, achievedTflops: 0.15, achievedBw: 0.37, ai: 0.40,
      cores: 64, imbalance: 0.04,
      perLayer: perLayer(71, 6.6, 0.05, 1.04, 1.16),
      source: 'decode_layer.py:812-869',
      static: [
        ['SPMD 路数', '5', '5', '一致', 'ok'],
        ['跨层输出', 'out FP32 + normed BF16', '一致', '一致', 'ok'],
      ],
      note: '层尾单次读取产出两份跨层输出，L39 偏高是因为要额外喂给输出边界。',
    },
    {
      id: 'rope-qkv', name: 'rope_qkv', scope: 'Scope 2 · Attention', group: 'attn',
      calls: 40, totalMs: 0.256, perLayerUs: 6.4, share: 1.7,
      units: { cube: 0, vector: 54.8, mte2: 31.0, mte3: 11.5, sync: 2.7 },
      bound: 'vector', boundLabel: 'Vector', efficiency: 69,
      bytesIn: 0.124, bytesOut: 0.106, reuse: 1.0,
      gflop: 0.05, achievedTflops: 0.20, achievedBw: 0.90, ai: 0.40,
      cores: 64, imbalance: 0.14,
      perLayer: perLayer(79, 6.4, 0.07, 1.06, 1.01),
      source: 'decode_layer.py:212-288',
      static: [
        ['Q pad', '5 real → 16 tile rows', '一致', '一致', 'ok'],
        ['paged 写入对齐', 'page 边界对齐', '未对齐 2 处', '偏差', 'warn'],
      ],
      note: 'MTE3 占比 11.5%，是全表最高：当前 token 的 K/V 要散写进 paged cache。',
    },
    {
      id: 'qk-norm', name: 'qk_norm', scope: 'Scope 1 · Norm', group: 'norm',
      calls: 320, totalMs: 0.232, perLayerUs: 5.8, share: 1.5,
      units: { cube: 0, vector: 68.4, mte2: 22.7, mte3: 6.2, sync: 2.7 },
      bound: 'vector', boundLabel: 'Vector', efficiency: 70,
      bytesIn: 0.048, bytesOut: 0.024, reuse: 1.3,
      gflop: 0.03, achievedTflops: 0.13, achievedBw: 0.31, ai: 0.42,
      cores: 64, imbalance: 0.08,
      perLayer: perLayer(83, 5.8, 0.06, 1.05, 1.01),
      source: 'decode_layer.py:508-556',
      static: [['task 数', '8（每 KV head 一路）', '8', '一致', 'ok']],
      note: 'gamma 与 reciprocal 合并在同一趟里做，避免二次读 q_proj / k_proj。',
    },
    {
      id: 'rms-recip', name: 'rms_recip', scope: 'Scope 1 · Norm', group: 'norm',
      calls: 40, totalMs: 0.168, perLayerUs: 4.2, share: 1.1,
      units: { cube: 0, vector: 74.2, mte2: 21.1, mte3: 1.4, sync: 3.3 },
      bound: 'vector', boundLabel: 'Vector', efficiency: 66,
      bytesIn: 0.026, bytesOut: 0.001, reuse: 1.0,
      gflop: 0.01, achievedTflops: 0.06, achievedBw: 0.16, ai: 0.38,
      cores: 64, imbalance: 0.05,
      perLayer: perLayer(89, 4.2, 0.06, 1.05, 1.01),
      source: 'decode_layer.py:118-164',
      static: [['pipeline stage', '4', '4', '一致', 'ok']],
      note: '只算倒数标量，与 QKV 投影重叠执行，实际暴露在关键路径上的不足 1 μs。',
    },
    {
      id: 'post-rms-reduce', name: 'post_rms_reduce', scope: 'Scope 3 · Norm', group: 'norm',
      calls: 40, totalMs: 0.136, perLayerUs: 3.4, share: 0.9,
      units: { cube: 0, vector: 72.8, mte2: 22.4, mte3: 1.5, sync: 3.3 },
      bound: 'vector', boundLabel: 'Vector', efficiency: 67,
      bytesIn: 0.021, bytesOut: 0.001, reuse: 1.0,
      gflop: 0.01, achievedTflops: 0.05, achievedBw: 0.15, ai: 0.36,
      cores: 64, imbalance: 0.05,
      perLayer: perLayer(97, 3.4, 0.06, 1.04, 1.01),
      source: 'decode_layer.py:171-204',
      static: [['与 residual_cast 并行', '是', '是', '一致', 'ok']],
      note: '与 residual_rms_cast 并行，reciprocal 延迟到 silu 里才应用。',
    },
    {
      id: 'fa-work-build', name: 'fa_work_build', scope: 'Scope 2 · Attention', group: 'attn',
      calls: 40, totalMs: 0.088, perLayerUs: 2.2, share: 0.6,
      units: { cube: 0, vector: 58.1, mte2: 12.4, mte3: 24.2, sync: 5.3 },
      bound: 'vector', boundLabel: 'Vector', efficiency: 61,
      bytesIn: 0.001, bytesOut: 0.002, reuse: 1.0,
      gflop: 0.00, achievedTflops: 0.01, achievedBw: 0.03, ai: 0.20,
      cores: 64, imbalance: 0.03,
      perLayer: perLayer(101, 2.2, 0.05, 1.03, 1.01),
      source: 'decode_layer.py:88-112',
      static: [
        ['work table 稠密率', '—（静态无法知道）', '80.5%', '206 / 256', 'ok'],
        ['去除的空块', '—', '50', '省 19.5% 迭代', 'ok'],
      ],
      note: '花 2.2 μs 把 ragged 请求压紧成无空洞工作表，为 fa_fused 省掉 19.5% 的空块迭代。',
    },
    {
      id: 'copy-hidden', name: 'copy_hidden', scope: '输入边界', group: 'boundary',
      calls: 1, totalMs: 0.003, perLayerUs: null, share: 0.02,
      units: { cube: 0, vector: 12.4, mte2: 61.2, mte3: 24.1, sync: 2.3 },
      bound: 'mte2', boundLabel: 'MTE2', efficiency: 84,
      bytesIn: 0.00016, bytesOut: 0.00033, reuse: 1.0,
      gflop: 0.00, achievedTflops: 0.00, achievedBw: 0.16, ai: 0.00,
      cores: 64, imbalance: 0.02,
      perLayer: null,
      source: 'decode_layer.py:1155',
      static: [['精度转换点', '1（入口唯一）', '1', '一致', 'ok']],
      note: '整个 step 唯一的入口精度边界，BF16 → FP32 只做一次。',
    },
    {
      id: 'cast-lmhead', name: 'cast_lmhead_in', scope: '输出边界', group: 'boundary',
      calls: 1, totalMs: 0.002, perLayerUs: null, share: 0.01,
      units: { cube: 0, vector: 14.1, mte2: 59.8, mte3: 24.0, sync: 2.1 },
      bound: 'mte2', boundLabel: 'MTE2', efficiency: 83,
      bytesIn: 0.00033, bytesOut: 0.00016, reuse: 1.0,
      gflop: 0.00, achievedTflops: 0.00, achievedBw: 0.25, ai: 0.00,
      cores: 64, imbalance: 0.02,
      perLayer: null,
      source: 'decode_layer.py:1189',
      static: [['精度转换点', '1（出口唯一）', '1', '一致', 'ok']],
      note: '40 层循环之后唯一的 FP32 → BF16 转换。',
    },
    {
      id: 'x-gamma0', name: 'x_gamma0', scope: '输入边界', group: 'boundary',
      calls: 1, totalMs: 0.002, perLayerUs: null, share: 0.01,
      units: { cube: 0, vector: 56.2, mte2: 30.1, mte3: 11.4, sync: 2.3 },
      bound: 'vector', boundLabel: 'Vector', efficiency: 70,
      bytesIn: 0.00033, bytesOut: 0.00016, reuse: 1.0,
      gflop: 0.00, achievedTflops: 0.00, achievedBw: 0.25, ai: 0.00,
      cores: 64, imbalance: 0.02,
      perLayer: null,
      source: 'decode_layer.py:1168',
      static: [['仅 layer 0', '是', '是', '一致', 'ok']],
      note: '只为 layer 0 补一份预缩放输入，其余各层由上一层的 dcr_xgamma 直接产出。',
    },
    {
      id: '__idle__', name: '同步与 Host 空隙', scope: '未归属', group: 'idle',
      calls: null, totalMs: 0.375, perLayerUs: 4.5, share: 2.5,
      units: { cube: 0, vector: 0, mte2: 0, mte3: 0, sync: 100 },
      bound: 'idle', boundLabel: 'Idle', efficiency: null,
      bytesIn: 0, bytesOut: 0, reuse: null,
      gflop: 0, achievedTflops: 0, achievedBw: 0, ai: null,
      cores: null, imbalance: null,
      perLayer: null,
      source: '—',
      static: [],
      note: '层内 barrier 0.180 ms + Host dispatch 0.195 ms。时间线检出 12 处 > 2 μs 的空隙。',
    },
  ];

  const groups = [
    { id: 'mlp', label: 'MLP', detail: 'gate/up · silu · down', ms: 8.236, share: 54.2 },
    { id: 'attn', label: 'Attention', detail: 'work_build · rope · fa_fused · softmax', ms: 2.964, share: 19.5 },
    { id: 'proj', label: 'QKV / Out 投影', detail: 'q/k/v_proj · out_proj', ms: 1.948, share: 12.8 },
    { id: 'norm', label: 'Norm / Carry', detail: 'rms · qk_norm · residual · dcr_xgamma', ms: 1.072, share: 7.1 },
    { id: 'boundary', label: 'LM Head 与边界', detail: 'copy · cast · rms_lm_head', ms: 0.605, share: 4.0 },
    { id: 'idle', label: '同步与空隙', detail: 'barrier · host dispatch', ms: 0.375, share: 2.5 },
  ];

  // ITL 直方图：512 steps，14.4 -> 18.8 ms，右偏长尾
  const itlBins = [
    [14.4, 6], [14.8, 41], [15.2, 168], [15.6, 131], [16.0, 74],
    [16.4, 41], [16.8, 22], [17.2, 13], [17.6, 8], [18.0, 5], [18.4, 2], [18.8, 1],
  ];

  const profiles = {
    'run-0803-a': {
      id: 'run-0803-a',
      title: 'Decode Fused · 全链路 Profiling',
      token: 'ptok://qwen3-14b/decode-fused@run-0803-a',
      meta: {
        model: 'Qwen3-14B', params: '14.8 B', dtype: 'BF16',
        batch: 16, layers: 40, seqAvg: 1614, page: 128,
        device: 'Ascend 950B', hbm: 64, peakBw: PEAK_BW, peakFlops: PEAK_FLOPS,
        env: 'env:8da1bf09', envMatch: true,
        steps: 512, capturedAt: '2026-08-03 14:32:08', duration: '7.8 s',
        collector: 'PyPTO DFX · msprof v9.0',
      },
      summary: {
        tpot: { p50: 15.2, p90: 16.4, p99: 18.1 }, tpotDelta: 3.4,
        tps: 1053, tpsDelta: -3.3,
        ttft: 184, ttftDelta: 1.2,
        kvUsed: 4.26, kvPool: 6.55, kvPct: 65.0,
        preempt: 0, batchAvg: 14.7,
        sol: [
          { id: 'cube', label: 'Cube (AIC)', pct: 3.7, detail: '29.4 / 800 TFLOPS' },
          { id: 'vector', label: 'Vector (AIV)', pct: 17.9, detail: 'elementwise + 归约' },
          { id: 'mte2', label: 'MTE2 · HBM → 片上', pct: 70.4, detail: '32.64 GB / step · 2.15 TB/s' },
          { id: 'mte3', label: 'MTE3 · 片上 → HBM', pct: 5.1, detail: '0.79 GB / step' },
        ],
        bound: 'memory',
        lowerBoundMs: 9.07, efficiency: 59.7,
        traffic: { weights: 27.99, kv: 4.23, act: 0.42, total: 32.64 },
        flops: { total: 447.5, achieved: 29.4, ai: 13.7, ridge: 222.2 },
      },
      groups,
      ops,
      itlBins,
      seqLens,
      baseline: { id: 'b8160fd', token: 'ptok://qwen3-14b/decode-fused@b8160fd', tpot: 14.7, tps: 1088, label: '可信基线 · 08/01' },
    },
  };

  window.PtoInferenceProfile = {
    profiles,
    current: 'run-0803-a',
    constants: { PEAK_BW, PEAK_FLOPS, TPOT },
    get: (id) => profiles[id || 'run-0803-a'],
  };
})();
