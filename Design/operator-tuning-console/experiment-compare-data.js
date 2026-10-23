/*
 * Experiment re-run fixtures for the 修改对比 view.
 *
 * None of the dumps under Data/ contains a second, post-change capture of the
 * same program, so every candidate number here is SIMULATED.  What is real:
 *   - the baseline side: app.js resolves every `ref` against the loaded run
 *     (data.js / qwen3-profiles-data.js), never against this file;
 *   - the code context lines: copied from the model source at the quoted line
 *     numbers (Data/DeepseekV4/deepseek_v4_flash_dspark/, repo/pypto-lib/models/qwen3_14b/);
 *   - the APIs the edits use (pl.system.task_dummy, DistributedConfig.num_sub_workers,
 *     pl.pipeline) exist in repo/pto.
 * The screen labels each candidate as 模拟复测 so it cannot be read as a measurement.
 *
 * Keyed by '<case key in TUNING_RUNS>:<finding id>'.
 *
 * metric.ref   where app.js reads the baseline value (see compareBase in app.js)
 * metric.cand  simulated candidate mean over `candidate.repeats` runs
 * metric.sigma candidate run-to-run standard deviation; a delta inside 2σ is 持平
 * metric.role  target  — what the hypothesis says should move
 *              guard   — what the finding's 护栏 says must not break
 *              observe — context, never decides
 * metric.limit guard only: { min } / { max } is an absolute bound on the candidate,
 *              { tol } is the largest allowed relative regression
 */
(function () {
  'use strict';

  window.TUNING_EXPERIMENTS = {

    /* ---------------------------------------------------------------- C1 */
    'decode_csa:C1': {
      summary: '每个 rank 一个 sub-worker，两卡 runner_run 同时离开 host',
      candidate: {
        runDir: '_jit_l3_decode_csa_20261009_101544',
        capturedAt: '2026-10-09 10:15:44',
        repeats: 10,
        pmu: false,
        swimlaneLevel: 4,
      },
      correctness: {
        check: 'golden_decode_csa · attn_out',
        value: 'max |Δ| 3.1e-4',
        limit: '≤ 1e-3（bf16）',
        pass: true,
      },
      code: [{
        file: 'decode_csa.py',
        hunks: [{
          at: [2041, 2041],
          lines: [
            '     compile_cfg = dict(',
            '         dump_passes=args.dump_passes,',
            '-        distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),',
            '+        # C1: one sub-worker per rank, so rank1 no longer queues behind',
            '+        # rank0\'s submission on the host before its runner_run starts.',
            '+        distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=len(device_ids)),',
            '     )',
            '     result = run(',
          ],
        }],
      }],
      compile: {
        note: '只改 host 侧下发配置：IR、Pass 轨迹和 perf hint 与基线逐项一致。',
        hints: [],
      },
      rank: 'rank0',
      metrics: [
        { layer: 'e2e', role: 'target', label: 'rank 启动偏移', ref: 'skew', cand: 182.4, sigma: 61.7, better: 'lower', unit: 'us' },
        { layer: 'e2e', role: 'target', label: 'rank0 集合点等待 Σ', ref: 'waitSum:rank0', cand: 268.5, sigma: 74.0, better: 'lower', unit: 'us' },
        { layer: 'e2e', role: 'target', label: 'rank0 device_wall', ref: 'wall:rank0', cand: 3698.2, sigma: 88.5, better: 'lower', unit: 'us' },
        { layer: 'e2e', role: 'guard', label: 'rank1 device_wall', ref: 'wall:rank1', cand: 3986.7, sigma: 40.3, better: 'lower', unit: 'us', limit: { tol: 0.02 } },
        { layer: 'e2e', role: 'guard', label: 'rank0 host bind', ref: 'e2e:rank0:chip.run.bind', cand: 3318.4, sigma: 52.6, better: 'lower', unit: 'us', limit: { tol: 0.10 } },
        { layer: 'l2', role: 'observe', label: 'rank0 trace span', ref: 'span:rank0', cand: 3610.4, sigma: 84.2, better: 'lower', unit: 'us' },
        { layer: 'l2', role: 'observe', label: 'rank0 AIC 占用', ref: 'aicUtil:rank0', cand: 43.1, sigma: 1.2, better: 'higher', unit: '%' },
      ],
      window: {
        rank: 'rank0', t0: 0, t1: 4900,
        caption: 'rank0 四个 *_wait 任务：错峰收窄后，等待段应同步缩短',
        tasks: {
          r2t13: { start: 270.3, end: 415.8 },
          r2t15: { start: 431.2, end: 449.6 },
          r2t56: { start: 3108.4, end: 3166.1 },
          r2t60: { start: 3540.7, end: 3542.1 },
        },
      },
    },

    /* ---------------------------------------------------------------- C2 */
    'decode_csa:C2': {
      summary: 'indexer_compressor 的 kv_score_proj 晚一跳下发，让 kv_score_proj_0 先拿到 AIC',
      candidate: {
        runDir: '_jit_l3_decode_csa_20261009_113207',
        capturedAt: '2026-10-09 11:32:07',
        repeats: 10,
        pmu: false,
        swimlaneLevel: 4,
      },
      correctness: {
        check: 'golden_decode_csa · attn_out',
        value: 'max |Δ| 0（逐位一致）',
        limit: '≤ 1e-3（bf16）',
        pass: true,
      },
      code: [{
        file: 'decode_csa.py',
        hunks: [{
          at: [869, 869],
          lines: [
            '     with pl.scope():',
            '         # Dispatch barrier: kv_proj_matmul, kv_score_proj and weights_proj resolve one hop',
            '         # after rms_norm, leaving qr_proj_matmul first.',
            '         late_dep = pl.system.task_dummy(deps=[rope_tid])',
            '+        # C2: the indexer compressor\'s kv_score_proj has 172 us of slack; push it one',
            '+        # more hop so the zero-slack kv_score_proj_0 wins the AIC when both are ready.',
            '+        idx_late_dep = pl.system.task_dummy(deps=[late_dep])',
            '         qkv_proj_rope(',
          ],
        }, {
          at: [903, 906],
          lines: [
            '         idx_cache_write_tid = indexer_compressor(',
            '             x_normed_t, idx_kv_unused,',
            '             inner_compress_state, inner_compress_state_block_table,',
            '             inner_wkv, inner_wgate, inner_ape, inner_norm_w,',
            '             cmp_cos_il, cmp_sin_signed, hadamard_idx,',
            '             idx_kv_cache, idx_kv_scale,',
            '             position_ids, idx_slot_mapping, inner_state_slot_mapping,',
            '-            late_dep,',
            '+            idx_late_dep,',
            '         )',
          ],
        }],
      }],
      compile: {
        note: '只多一个 task_dummy：编排图 +1 个调度节点，kernel IR 与 perf hint 不变。',
        hints: [],
        graph: [['rank0 任务数', '84', '85'], ['kernel / scope', '64 / 62', '64 / 62']],
      },
      rank: 'rank0',
      metrics: [
        { layer: 'l2', role: 'target', label: 'kv_score_proj_0 span', ref: 'task:r2t28.span', cand: 318.2, sigma: 6.1, better: 'lower', unit: 'us', task: 'r2t28' },
        { layer: 'l2', role: 'target', label: '关键路径 makespan', ref: 'makespan:rank0', cand: 4803.5, sigma: 21.4, better: 'lower', unit: 'us' },
        { layer: 'l2', role: 'guard', label: 'kv_score_proj slack', ref: 'task:r2t23.slack', cand: 95.1, sigma: 8.7, better: 'higher', unit: 'us', task: 'r2t23', limit: { min: 0 } },
        { layer: 'l2', role: 'guard', label: 'kv_proj_matmul slack', ref: 'task:r3t11.slack', cand: 812.4, sigma: 11.9, better: 'higher', unit: 'us', task: 'r3t11', limit: { min: 0 } },
        { layer: 'l2', role: 'observe', label: 'kv_score_proj span', ref: 'task:r2t23.span', cand: 762.5, sigma: 9.0, better: 'lower', unit: 'us', task: 'r2t23' },
        { layer: 'l2', role: 'observe', label: 'kv_proj_matmul span', ref: 'task:r3t11.span', cand: 185.2, sigma: 7.4, better: 'lower', unit: 'us', task: 'r3t11' },
        { layer: 'l2', role: 'observe', label: 'AIC 占用', ref: 'aicUtil:rank0', cand: 32.7, sigma: 0.4, better: 'higher', unit: '%' },
        { layer: 'l1', role: 'observe', label: 'kv_score_proj_0 块中位', ref: 'task:r2t28.durMed', cand: 6.26, sigma: 0.05, better: 'lower', unit: 'us', task: 'r2t28' },
        { layer: 'e2e', role: 'observe', label: 'rank0 device_wall', ref: 'wall:rank0', cand: 5058.1, sigma: 24.0, better: 'lower', unit: 'us' },
      ],
      window: {
        rank: 'rank0', t0: 1400, t1: 2460,
        caption: '就绪窗口 1428–1824 us：kv_score_proj_0 先拿核，两个 rival 后移但仍在 latest-finish 之前',
        tasks: {
          r2t28: { start: 1428.4, end: 1746.6 },
          r2t23: { start: 1508.9, end: 2271.4 },
          r3t11: { start: 1431.5, end: 1616.7 },
        },
      },
    },

    /* ---------------------------------------------------------------- C3 */
    'decode_csa:C3': {
      summary: 'scatter_softmax_pool 晚一跳下发，给 indexer_topk_query_merge 让出 AIV',
      candidate: {
        runDir: '_jit_l3_decode_csa_20261009_140518',
        capturedAt: '2026-10-09 14:05:18',
        repeats: 10,
        pmu: false,
        swimlaneLevel: 4,
      },
      correctness: {
        check: 'golden_decode_csa · attn_out',
        value: 'max |Δ| 0（逐位一致）',
        limit: '≤ 1e-3（bf16）',
        pass: true,
      },
      code: [{
        file: 'decode_compressor_ratio4.py',
        hunks: [{
          at: [143, 143],
          lines: [
            '     # One block per request: each c_idx owns its own state ring and writes only',
            '     # its own pooled_kv rows, so the whole nest is parallel over requests.',
            '-    with pl.spmd(b_dim, name_hint="scatter_softmax_pool", deps=[_kv_score_tid]) as pool_tid:',
            '+    # C3: hold the pool one hop so indexer_topk_query_merge keeps the AIV.',
            '+    pool_gate = pl.system.task_dummy(deps=[_kv_score_tid])',
            '+    with pl.spmd(b_dim, name_hint="scatter_softmax_pool", deps=[pool_gate]) as pool_tid:',
            '         c_idx = pl.tile.get_block_idx()',
            '         first_pos_b = pl.read(position_ids, [c_idx * s_dim])',
          ],
        }],
      }],
      compile: {
        note: '只多一个 task_dummy：kernel IR 与 perf hint 不变；后移量由调度决定，源码里无法设上限。',
        hints: [],
        graph: [['rank0 任务数', '84', '85'], ['kernel / scope', '64 / 62', '64 / 62']],
      },
      rank: 'rank0',
      metrics: [
        { layer: 'l2', role: 'target', label: 'indexer_topk_query_merge span', ref: 'task:r3t16.span', cand: 128.7, sigma: 4.3, better: 'lower', unit: 'us', task: 'r3t16' },
        { layer: 'l2', role: 'target', label: '关键路径 makespan', ref: 'makespan:rank0', cand: 4911.6, sigma: 19.8, better: 'lower', unit: 'us' },
        { layer: 'l2', role: 'guard', label: 'scatter_softmax_pool slack', ref: 'task:r2t24.slack', cand: 38.5, sigma: 9.6, better: 'higher', unit: 'us', task: 'r2t24', limit: { min: 0 } },
        { layer: 'l2', role: 'guard', label: 'rmsnorm_rope_cache_write slack', ref: 'task:r2t26.slack', cand: -41.6, sigma: 10.2, better: 'higher', unit: 'us', task: 'r2t26', limit: { min: 0 } },
        { layer: 'l2', role: 'guard', label: 'compress_state_commit slack', ref: 'task:r2t25.slack', cand: 905.3, sigma: 12.5, better: 'higher', unit: 'us', task: 'r2t25', limit: { min: 0 } },
        { layer: 'l2', role: 'observe', label: 'AIV 占用', ref: 'aivUtil:rank0', cand: 37.2, sigma: 0.5, better: 'higher', unit: '%' },
        { layer: 'e2e', role: 'observe', label: 'rank0 device_wall', ref: 'wall:rank0', cand: 5161.4, sigma: 23.1, better: 'lower', unit: 'us' },
      ],
      window: {
        rank: 'rank0', t0: 2280, t1: 2700,
        caption: 'r3t16 变短了，但 rmsnorm_rope_cache_write 被推过 latest-finish：尾延迟转移给了它的后继',
        tasks: {
          r3t16: { start: 2340.4, end: 2469.1 },
          r2t24: { start: 2412.8, end: 2513.4 },
          r2t26: { start: 2630.1, end: 2662.94 },
          r2t25: { start: 2560.1, end: 2603.1 },
        },
      },
    },

    /* ------------------------------------------------------------ MOCK-C1 */
    'MOCK-C1:MOCK-C1': {
      summary: 'kv_score_proj 的 K 切分 512 → 256，Right 每级 32 KB → 16 KB，MemoryReuse 保住 depth 2',
      candidate: {
        runDir: '_jit_l3_decode_csa_mock_ktile256',
        capturedAt: '2026-10-09 15:40:12',
        repeats: 10,
        pmu: false,
        swimlaneLevel: 4,
      },
      correctness: {
        check: 'golden_decode_csa · cmp_out',
        value: 'max |Δ| 6.0e-5（累加顺序变化）',
        limit: '≤ 1e-3（fp32 累加）',
        pass: true,
      },
      code: [{
        file: 'decode_compressor_ratio4.py',
        hunks: [{
          at: [62, 62],
          lines: [
            ' # tiling',
            ' K_TILE = 512',
            '+# kv_score_proj only: halve the K slice so each Right stage is 16 KB and two',
            '+# stages of all five groups fit the 64 KB MemoryReuse budget (PH-MR-001).',
            '+KV_K_TILE = 256',
            ' OUT_TILE = 64',
            ' MM_B_TILE = 16',
          ],
        }, {
          at: [118, 121],
          lines: [
            '-        for kb in pl.pipeline(0, D // K_TILE, stage=2):',
            '-            k0 = kb * K_TILE',
            '+        for kb in pl.pipeline(0, D // KV_K_TILE, stage=2):',
            '+            k0 = kb * KV_K_TILE',
            '             x_rows = pl.min(MM_B_TILE, bs - global_row0)',
            '-            x_tile = pl.slice(x_flat, [MM_B_TILE, K_TILE], [global_row0, k0], valid_shape=[x_rows, K_TILE])',
            '+            x_tile = pl.slice(x_flat, [MM_B_TILE, KV_K_TILE], [global_row0, k0], valid_shape=[x_rows, KV_K_TILE])',
          ],
        }, {
          at: [126, 129],
          lines: [
            '-            wkv_tile = wkv[o0 : o0 + OUT_TILE, k0 : k0 + K_TILE]',
            '-            wgate_tile = wgate[o0 : o0 + OUT_TILE, k0 : k0 + K_TILE]',
            '+            wkv_tile = wkv[o0 : o0 + OUT_TILE, k0 : k0 + KV_K_TILE]',
            '+            wgate_tile = wgate[o0 : o0 + OUT_TILE, k0 : k0 + KV_K_TILE]',
            '             if k0 == 0:',
          ],
        }],
      }],
      compile: {
        note: 'PH-MR-001 在该源码点消失；MemoryReuse 之后的 IR 只在 kv_score_proj 作用域内变化。',
        hints: [
          { code: 'PH-MR-001', site: 'decode_compressor_ratio4.py:110', base: 'Right · depth 2→1 · 5 组 · 32 KB/stage', cand: '未触发 · depth 2 保持 · 16 KB/stage', state: 'resolved' },
        ],
        graph: [['该源码点 PH-MR-001（每组一条）', '5', '0'], ['perf hint 总数', '230', '225']],
      },
      rank: 'rank0',
      metrics: [
        { layer: 'compiler', role: 'target', label: '该点拟合流水深度', ref: 'hint:decode_compressor_ratio4.py:110.fittedDepth', cand: 2, sigma: 0, better: 'higher', unit: '级' },
        { layer: 'l1', role: 'target', label: 'kv_score_proj_0 块中位', ref: 'task:r2t28.durMed', cand: 5.12, sigma: 0.08, better: 'lower', unit: 'us', task: 'r2t28' },
        { layer: 'l2', role: 'target', label: 'kv_score_proj_0 span', ref: 'task:r2t28.span', cand: 341.7, sigma: 5.8, better: 'lower', unit: 'us', task: 'r2t28' },
        { layer: 'l2', role: 'target', label: '关键路径 makespan', ref: 'makespan:rank0', cand: 4836.1, sigma: 20.9, better: 'lower', unit: 'us' },
        { layer: 'l1', role: 'guard', label: 'kv_score_proj_0 setup 占比', ref: 'task:r2t28.setupShare', cand: 0.061, sigma: 0.004, better: 'lower', unit: '', task: 'r2t28', limit: { tol: 0.25 } },
        { layer: 'l2', role: 'guard', label: 'kv_score_proj slack', ref: 'task:r2t23.slack', cand: 160.8, sigma: 9.1, better: 'higher', unit: 'us', task: 'r2t23', limit: { min: 0 } },
        { layer: 'l1', role: 'observe', label: 'mte2 占比（PMU 模拟）', base: 0.42, cand: 0.31, sigma: 0.02, better: 'lower', unit: '' },
      ],
      window: {
        rank: 'rank0', t0: 1400, t1: 2260,
        caption: '块内流水恢复 depth 2 后，kv_score_proj_0 的 256 块整体变短，竞争窗口随之收窄',
        tasks: {
          r2t28: { start: 1428.2, end: 1769.9 },
          r2t23: { start: 1430.9, end: 2205.7 },
          r3t11: { start: 1431.3, end: 1534.0 },
        },
      },
    },

    /* ---------------------------------------------------------------- C4 */
    'decode_fwd_layers:C4': {
      summary: 'MLP_M_TILE 128 → 64 缩小 gate_up_proj 单块工作量，并打开块内 PMU 重采',
      candidate: {
        runDir: 'pypto_qwen3_profiles/tp1 · _jit_prefill_fwd_20261009_162210',
        capturedAt: '2026-10-09 16:22:10',
        repeats: 3,
        pmu: true,
        swimlaneLevel: 4,
      },
      correctness: {
        check: 'golden_prefill_fwd · hidden_out',
        value: 'max |Δ| 2.2e-3',
        limit: '≤ 4e-3（bf16，40 层累积）',
        pass: true,
      },
      code: [{
        file: 'prefill_fwd.py',
        hunks: [{
          at: [90, 90],
          lines: [
            ' KV_OUT_CHUNK = 256',
            '-TOK_TILE = 128',
            '+TOK_TILE = 64  # C4: must track MLP_M_TILE (assert at line 180)',
            ' X_GAMMA_HIDDEN_CHUNK = 128',
          ],
        }, {
          at: [144, 144],
          lines: [
            '-MLP_M_TILE = 128',
            '+# C4: halve the per-block M so one gate_up_proj block carries half the work.',
            '+MLP_M_TILE = 64',
            ' SILU_OUT_CHUNK = 64',
          ],
        }],
      }],
      compile: {
        note: 'line 180 断言 MLP_M_TILE == TOK_TILE，只改一处编不过；TOK_TILE 也切分 MLP 之前的算子，改动面超出 gate_up_proj。',
        hints: [],
        graph: [['gate_up_proj 块数', '20', '40'], ['受影响常量', 'MLP_M_TILE', 'MLP_M_TILE + TOK_TILE']],
      },
      rank: 'tp1:prefill',
      metrics: [
        { layer: 'l2', role: 'target', label: 'gate_up_proj span', ref: 'task:r3t688.span', cand: 268.9, sigma: 14.6, better: 'lower', unit: 'us', task: 'r3t688' },
        { layer: 'l1', role: 'target', label: 'gate_up_proj 最长块', ref: 'task:r3t688.durMax', cand: 121.4, sigma: 6.2, better: 'lower', unit: 'us', task: 'r3t688' },
        { layer: 'l2', role: 'guard', label: 'prefill makespan', ref: 'makespan:tp1:prefill', cand: 44102.7, sigma: 410.5, better: 'lower', unit: 'us', limit: { tol: 0.01 } },
        { layer: 'l2', role: 'observe', label: 'AIC 占用', ref: 'aicUtil:tp1:prefill', cand: 71.9, sigma: 1.4, better: 'higher', unit: '%' },
      ],
      window: {
        rank: 'tp1:prefill', t0: 30250, t1: 30700,
        caption: '单块工作量减半后块数翻倍：一波变两波，span 只缩短到原来的 85%',
        tasks: {
          r3t688: { start: 30322.1, end: 30591.0 },
        },
      },
    },
  };
})();
