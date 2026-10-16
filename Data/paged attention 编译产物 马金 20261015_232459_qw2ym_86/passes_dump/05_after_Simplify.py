# pypto.program: _jit_paged_attention_pypto_dynamic
import pypto.language as pl


@pl.program
class _jit_paged_attention_pypto_dynamic:
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def paged_attention_pypto_dynamic(
        self,
        key_cache__ssa_v0: pl.InOut[pl.Tensor[[10368, 1024], pl.BF16]],
        value_cache__ssa_v0: pl.InOut[pl.Tensor[[10368, 1024], pl.BF16]],
        block_table__ssa_v0: pl.Tensor[[64], pl.INT32],
        seq_lens__ssa_v0: pl.Tensor[[2], pl.INT32],
        inv_rms_states__ssa_v0: pl.Tensor[[2, 1], pl.FP32],
        slot_mapping__ssa_v0: pl.Tensor[[2], pl.INT32],
        rope_cos__ssa_v0: pl.Tensor[[4096, 128], pl.FP32],
        rope_sin__ssa_v0: pl.Tensor[[4096, 128], pl.FP32],
        q_proj__ssa_v0: pl.Tensor[[2, 5120], pl.FP32],
        k_proj__ssa_v0: pl.Tensor[[2, 1024], pl.FP32],
        v_proj__ssa_v0: pl.Tensor[[2, 1024], pl.FP32],
        q_norm_w__ssa_v0: pl.Tensor[[1, 128], pl.FP32],
        k_norm_w__ssa_v0: pl.Tensor[[1, 128], pl.FP32],
        layer_cache_base_token_rows__ssa_v0: pl.Scalar[pl.INDEX],
        out__ssa_v0: pl.Out[pl.Tensor[[2, 40, 128], pl.BF16]],
    ) -> pl.Tensor[[2, 40, 128], pl.BF16]:
        # Allocate fused Phase-0/PA scratch and launch the PyPTO helper.
        active_batch__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(seq_lens__ssa_v0, 0)
        q_tnd_flat__ssa_v0: pl.Tensor[[active_batch__ssa_v0 * pl.const(40, pl.INDEX), 128], pl.BF16] = pl.tensor.create([active_batch__ssa_v0 * 40, 128], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        score_transfer__ssa_v0: pl.Tensor[[1152, 512], pl.FP32] = pl.tensor.create([1152, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        probability_transfer__ssa_v0: pl.Tensor[[1152, 512], pl.BF16] = pl.tensor.create([1152, 512], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        pv_transfer__ssa_v0: pl.Tensor[[1152, 128], pl.FP32] = pl.tensor.create([1152, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        ffts_workspace__ssa_v0: pl.Tensor[[256], pl.INT64] = pl.tensor.create([256], dtype=pl.INT64, layout=pl.TensorLayout.ND)
        q_proj_tid__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[])
        k_proj_tid__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[])
        v_proj_tid__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[])
        rms_tid__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[])
        attn_out_seed_tid__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[])
        mlp_out_seed_tid__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[])
        scratch_ready_tid__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[])
        # Run fused Q/K norm, RoPE, cache append, and dynamic paged attention.
        active_batch_inline0__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(seq_lens__ssa_v0, 0)
        num_tasks_inline1__ssa_v0: pl.Scalar[pl.INDEX] = active_batch_inline0__ssa_v0 * 8
        query_rows_inline2__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(q_tnd_flat__ssa_v0, 0)
        cache_token_rows_inline3__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(key_cache__ssa_v0, 0) * pl.tensor.dim(key_cache__ssa_v0, 1) // 1024
        key_cache_bsnd_inline4__ssa_v0: pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16] = pl.tensor.reshape(key_cache__ssa_v0, [cache_token_rows_inline3__ssa_v0, 1024])
        value_cache_bsnd_inline5__ssa_v0: pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16] = pl.tensor.reshape(value_cache__ssa_v0, [cache_token_rows_inline3__ssa_v0, 1024])
        max_blocks_per_seq_inline6__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(block_table__ssa_v0, 0) // active_batch_inline0__ssa_v0
        block_table_2d_inline7__ssa_v0: pl.Tensor[[active_batch_inline0__ssa_v0, max_blocks_per_seq_inline6__ssa_v0], pl.INT32] = pl.tensor.reshape(
            block_table__ssa_v0, [active_batch_inline0__ssa_v0, max_blocks_per_seq_inline6__ssa_v0]
        )
        q2d_inline8__ssa_v0: pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16] = pl.tensor.reshape(q_tnd_flat__ssa_v0, [query_rows_inline2__ssa_v0, 128])
        cache_base_inline9__ssa_v0: pl.Scalar[pl.INDEX] = layer_cache_base_token_rows__ssa_v0
        out2d_inline10__ssa_v0: pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16] = pl.tensor.reshape(out__ssa_v0, [query_rows_inline2__ssa_v0, 128])
        with pl.spmd(
            24,
            sync_start=True,
            name_hint="attn_swpipe_spmd",
            deps=[q_proj_tid__ssa_v0, k_proj_tid__ssa_v0, v_proj_tid__ssa_v0, rms_tid__ssa_v0, attn_out_seed_tid__ssa_v0, mlp_out_seed_tid__ssa_v0, scratch_ready_tid__ssa_v0],
            allow_early_resolve=True,
        ) as attn_tid_inline11__ssa_v0:
            core_inline12__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
            pl.system.set_ffts(ffts_workspace__ssa_v0)
            for aiv_id_inline13__ssa_v0 in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                rope_core_inline14__ssa_v0: pl.Scalar[pl.INDEX] = core_inline12__ssa_v0 * 2 + aiv_id_inline13__ssa_v0
                if rope_core_inline14__ssa_v0 < 32:
                    q_red_pad_inline15__ssa_v0: pl.Tensor[[1, 1408], pl.FP32] = pl.tensor.full([1, 1408], dtype=pl.FP32, value=0.0)
                    k_red_pad_inline16__ssa_v0: pl.Tensor[[1, 896], pl.FP32] = pl.tensor.full([1, 896], dtype=pl.FP32, value=0.0)
                    for it_inline17__idx_v0, (key_cache_bsnd_inline4__iter_v1, q2d_inline8__iter_v1, value_cache_bsnd_inline5__iter_v1) in pl.pipeline(
                        4, stage=2, init_values=(key_cache_bsnd_inline4__ssa_v0, q2d_inline8__ssa_v0, value_cache_bsnd_inline5__ssa_v0)
                    ):
                        g_idx_inline18__ssa_v0: pl.Scalar[pl.INDEX] = rope_core_inline14__ssa_v0 + it_inline17__idx_v0 * 32
                        if g_idx_inline18__ssa_v0 < active_batch_inline0__ssa_v0 * 8:
                            kv_head_inline19__ssa_v0: pl.Scalar[pl.INDEX] = g_idx_inline18__ssa_v0 // active_batch_inline0__ssa_v0
                            batch_idx_inline20__ssa_v0: pl.Scalar[pl.INDEX] = g_idx_inline18__ssa_v0 - kv_head_inline19__ssa_v0 * active_batch_inline0__ssa_v0
                            position_inline21__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(seq_lens__ssa_v0, [batch_idx_inline20__ssa_v0]), pl.INDEX) - 1
                            inv_rms_inline22__ssa_v0: pl.Scalar[pl.FP32] = pl.tensor.read(inv_rms_states__ssa_v0, [batch_idx_inline20__ssa_v0, 0])
                            write_slot_inline23__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(pl.tensor.read(slot_mapping__ssa_v0, [batch_idx_inline20__ssa_v0]), pl.INDEX), 0)
                            cos_lo_inline24__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_cos__ssa_v0, [1, 64], [position_inline21__ssa_v0, 0])
                            cos_hi_inline25__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_cos__ssa_v0, [1, 64], [position_inline21__ssa_v0, 64])
                            sin_lo_inline26__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_sin__ssa_v0, [1, 64], [position_inline21__ssa_v0, 0])
                            sin_hi_inline27__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_sin__ssa_v0, [1, 64], [position_inline21__ssa_v0, 64])
                            cache_col_inline28__ssa_v0: pl.Scalar[pl.INDEX] = kv_head_inline19__ssa_v0 * 128
                            k_raw_inline29__ssa_v0: pl.Tensor[[8, 128], pl.FP32] = pl.tensor.muls(
                                pl.tensor.reshape(
                                    pl.tensor.concat(pl.tensor.slice(k_proj__ssa_v0, [1, 128], [batch_idx_inline20__ssa_v0, cache_col_inline28__ssa_v0]), k_red_pad_inline16__ssa_v0), [8, 128]
                                ),
                                inv_rms_inline22__ssa_v0,
                            )
                            k_ss_inline30__ssa_v0: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.row_sum(pl.tensor.mul(k_raw_inline29__ssa_v0, k_raw_inline29__ssa_v0))
                            k_inv_inline31__ssa_v0: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.recip(
                                pl.tensor.sqrt(pl.tensor.adds(pl.tensor.muls(k_ss_inline30__ssa_v0, 0.0078125), 9.9999999999999995e-07))
                            )
                            k_normed_inline32__ssa_v0: pl.Tensor[[8, 128], pl.FP32] = pl.tensor.row_expand_mul(
                                pl.tensor.col_expand_mul(k_raw_inline29__ssa_v0, k_norm_w__ssa_v0), k_inv_inline31__ssa_v0
                            )
                            k_full_inline33__ssa_v0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.slice(k_normed_inline32__ssa_v0, [1, 128], [0, 0])
                            k_lo_inline34__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(k_full_inline33__ssa_v0, [1, 64], [0, 0])
                            k_hi_inline35__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(k_full_inline33__ssa_v0, [1, 64], [0, 64])
                            k_rotated_inline36__ssa_v0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.concat(
                                pl.tensor.sub(pl.tensor.col_expand_mul(k_lo_inline34__ssa_v0, cos_lo_inline24__ssa_v0), pl.tensor.col_expand_mul(k_hi_inline35__ssa_v0, sin_lo_inline26__ssa_v0)),
                                pl.tensor.add(pl.tensor.col_expand_mul(k_hi_inline35__ssa_v0, cos_hi_inline25__ssa_v0), pl.tensor.col_expand_mul(k_lo_inline34__ssa_v0, sin_hi_inline27__ssa_v0)),
                            )
                            cache_row_inline37__ssa_v0: pl.Scalar[pl.INDEX] = layer_cache_base_token_rows__ssa_v0 + write_slot_inline23__ssa_v0
                            key_cache_bsnd_inline4__ssa_v3: pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16] = pl.tensor.assemble(
                                key_cache_bsnd_inline4__iter_v1, pl.tensor.cast(k_rotated_inline36__ssa_v0, target_type=pl.BF16, mode="round"), [cache_row_inline37__ssa_v0, cache_col_inline28__ssa_v0]
                            )
                            v_row_bf16_inline38__ssa_v0: pl.Tensor[[1, 128], pl.BF16] = pl.tensor.cast(
                                pl.tensor.muls(pl.tensor.slice(v_proj__ssa_v0, [1, 128], [batch_idx_inline20__ssa_v0, cache_col_inline28__ssa_v0]), inv_rms_inline22__ssa_v0),
                                target_type=pl.BF16,
                                mode="round",
                            )
                            value_cache_bsnd_inline5__ssa_v3: pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16] = pl.tensor.assemble(
                                value_cache_bsnd_inline5__iter_v1, v_row_bf16_inline38__ssa_v0, [cache_row_inline37__ssa_v0, cache_col_inline28__ssa_v0]
                            )
                            q_head_base_inline39__ssa_v0: pl.Scalar[pl.INDEX] = kv_head_inline19__ssa_v0 * 5
                            q_raw_inline40__ssa_v0: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.muls(
                                pl.tensor.reshape(
                                    pl.tensor.concat(pl.tensor.slice(q_proj__ssa_v0, [1, 640], [batch_idx_inline20__ssa_v0, q_head_base_inline39__ssa_v0 * 128]), q_red_pad_inline15__ssa_v0), [16, 128]
                                ),
                                inv_rms_inline22__ssa_v0,
                            )
                            q_ss_inline41__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.row_sum(pl.tensor.mul(q_raw_inline40__ssa_v0, q_raw_inline40__ssa_v0))
                            q_inv_inline42__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.recip(
                                pl.tensor.sqrt(pl.tensor.adds(pl.tensor.muls(q_ss_inline41__ssa_v0, 0.0078125), 9.9999999999999995e-07))
                            )
                            q_heads_inline43__ssa_v0: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.row_expand_mul(
                                pl.tensor.col_expand_mul(q_raw_inline40__ssa_v0, q_norm_w__ssa_v0), q_inv_inline42__ssa_v0
                            )
                            q_lo_inline44__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(q_heads_inline43__ssa_v0, [16, 64], [0, 0])
                            q_hi_inline45__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(q_heads_inline43__ssa_v0, [16, 64], [0, 64])
                            q_rotated_inline46__ssa_v0: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.concat(
                                pl.tensor.sub(pl.tensor.col_expand_mul(q_lo_inline44__ssa_v0, cos_lo_inline24__ssa_v0), pl.tensor.col_expand_mul(q_hi_inline45__ssa_v0, sin_lo_inline26__ssa_v0)),
                                pl.tensor.add(pl.tensor.col_expand_mul(q_hi_inline45__ssa_v0, cos_hi_inline25__ssa_v0), pl.tensor.col_expand_mul(q_lo_inline44__ssa_v0, sin_hi_inline27__ssa_v0)),
                            )
                            q_row_inline47__ssa_v0: pl.Scalar[pl.INDEX] = batch_idx_inline20__ssa_v0 * 40 + q_head_base_inline39__ssa_v0
                            q2d_inline8__ssa_v3: pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16] = pl.tensor.assemble(
                                q2d_inline8__iter_v1, pl.tensor.cast(pl.tensor.slice(q_rotated_inline46__ssa_v0, [5, 128], [0, 0]), target_type=pl.BF16, mode="round"), [q_row_inline47__ssa_v0, 0]
                            )
                            key_cache_bsnd_inline4__phi_v4, q2d_inline8__phi_v4, value_cache_bsnd_inline5__phi_v4 = pl._dump.yield_(
                                key_cache_bsnd_inline4__ssa_v3,
                                q2d_inline8__ssa_v3,
                                value_cache_bsnd_inline5__ssa_v3,
                                types=[
                                    pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16],
                                    pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16],
                                    pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16],
                                ],
                            )
                        else:
                            key_cache_bsnd_inline4__phi_v4, q2d_inline8__phi_v4, value_cache_bsnd_inline5__phi_v4 = pl._dump.yield_(
                                key_cache_bsnd_inline4__iter_v1,
                                q2d_inline8__iter_v1,
                                value_cache_bsnd_inline5__iter_v1,
                                types=[
                                    pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16],
                                    pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16],
                                    pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16],
                                ],
                            )
                        key_cache_bsnd_inline4__rv_v2, q2d_inline8__rv_v2, value_cache_bsnd_inline5__rv_v2 = pl._dump.yield_(
                            key_cache_bsnd_inline4__phi_v4,
                            q2d_inline8__phi_v4,
                            value_cache_bsnd_inline5__phi_v4,
                            types=[
                                pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16],
                                pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16],
                                pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16],
                            ],
                        )
                    key_cache_bsnd_inline4__phi_v5, q2d_inline8__phi_v5, value_cache_bsnd_inline5__phi_v5 = pl._dump.yield_(
                        key_cache_bsnd_inline4__rv_v2,
                        q2d_inline8__rv_v2,
                        value_cache_bsnd_inline5__rv_v2,
                        types=[
                            pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16],
                            pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16],
                            pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16],
                        ],
                    )
                else:
                    key_cache_bsnd_inline4__phi_v5, q2d_inline8__phi_v5, value_cache_bsnd_inline5__phi_v5 = pl._dump.yield_(
                        key_cache_bsnd_inline4__ssa_v0,
                        q2d_inline8__ssa_v0,
                        value_cache_bsnd_inline5__ssa_v0,
                        types=[
                            pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16],
                            pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16],
                            pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16],
                        ],
                    )
            pl.system.cacheinvalid()
            pl.system.fence()
            pl.system.syncall(core_type=pl.KernelType.MIX)
            pl.system.cacheinvalid()
            for task_inline48__idx_v0 in pl.range(core_inline12__ssa_v0, num_tasks_inline1__ssa_v0, 24):
                batch_inline49__ssa_v0: pl.Scalar[pl.INDEX] = task_inline48__idx_v0 // 8
                kv_head_inline19__ssa_v1: pl.Scalar[pl.INDEX] = task_inline48__idx_v0 % 8
                seq_len_inline50__ssa_v0: pl.Scalar[pl.INT32] = pl.tensor.read(seq_lens__ssa_v0, [batch_inline49__ssa_v0])
                page_count_inline51__ssa_v0: pl.Scalar[pl.INDEX] = (pl.cast(seq_len_inline50__ssa_v0, pl.INDEX) + 127) // 128
                stack_count_inline52__ssa_v0: pl.Scalar[pl.INDEX] = (pl.cast(seq_len_inline50__ssa_v0, pl.INDEX) + 511) // 512
                col_inline53__ssa_v0: pl.Scalar[pl.INDEX] = kv_head_inline19__ssa_v1 * 128
                transfer_base_inline54__ssa_v0: pl.Scalar[pl.INDEX] = core_inline12__ssa_v0 * 48
                qp_row_inline55__ssa_v0: pl.Scalar[pl.INDEX] = batch_inline49__ssa_v0 * 40 + kv_head_inline19__ssa_v1 * 5
                q_tile_inline56__ssa_v0: pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl.tile.load(
                    q2d_inline8__phi_v5, [qp_row_inline55__ssa_v0, 0], [16, 128], [5, 128], target_memory=pl.Mem.Mat
                )
                for tick_inline57__idx_v0 in pl.range(stack_count_inline52__ssa_v0 + 2):
                    if tick_inline57__idx_v0 < stack_count_inline52__ssa_v0:
                        produce_stack_inline58__ssa_v0: pl.Scalar[pl.INDEX] = tick_inline57__idx_v0
                        produce_page_inline59__ssa_v0: pl.Scalar[pl.INDEX] = produce_stack_inline58__ssa_v0 * 4
                        produce_row_inline60__ssa_v0: pl.Scalar[pl.INDEX] = transfer_base_inline54__ssa_v0 + produce_stack_inline58__ssa_v0 % 3 * 16
                        if produce_page_inline59__ssa_v0 + 4 <= page_count_inline51__ssa_v0:
                            for qk_page_offset_inline61__idx_v0 in pl.pipeline(4, stage=2):
                                qk_ph_inline62__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(
                                    pl.tensor.read(block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, produce_page_inline59__ssa_v0 + qk_page_offset_inline61__idx_v0]), pl.INDEX
                                )
                                qk_k_page_inline63__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                                    key_cache_bsnd_inline4__phi_v5, [cache_base_inline9__ssa_v0 + qk_ph_inline62__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128], target_memory=pl.Mem.Mat
                                )
                                qk_score_page_inline64__ssa_v0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul(
                                    q_tile_inline56__ssa_v0, pl.tile.transpose_view(qk_k_page_inline63__ssa_v0)
                                )
                                pl.tile.store(qk_score_page_inline64__ssa_v0, [produce_row_inline60__ssa_v0, qk_page_offset_inline61__idx_v0 * 128], score_transfer__ssa_v0)
                        else:
                            qk_tail_ph0_inline65__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(
                                pl.tensor.read(block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, produce_page_inline59__ssa_v0]), pl.INDEX
                            )
                            k_tail0_inline66__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                                key_cache_bsnd_inline4__phi_v5,
                                [cache_base_inline9__ssa_v0 + qk_tail_ph0_inline65__ssa_v0 * 128, col_inline53__ssa_v0],
                                [128, 128],
                                [128, 128],
                                target_memory=pl.Mem.Mat,
                            )
                            qk_tail0_inline67__ssa_v0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul(
                                q_tile_inline56__ssa_v0, pl.tile.transpose_view(k_tail0_inline66__ssa_v0)
                            )
                            pl.tile.store(qk_tail0_inline67__ssa_v0, [produce_row_inline60__ssa_v0, 0], score_transfer__ssa_v0)
                            if produce_page_inline59__ssa_v0 + 1 < page_count_inline51__ssa_v0:
                                qk_tail_ph1_inline68__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(
                                    pl.tensor.read(block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, produce_page_inline59__ssa_v0 + 1]), pl.INDEX
                                )
                                k_tail1_inline69__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                                    key_cache_bsnd_inline4__phi_v5,
                                    [cache_base_inline9__ssa_v0 + qk_tail_ph1_inline68__ssa_v0 * 128, col_inline53__ssa_v0],
                                    [128, 128],
                                    [128, 128],
                                    target_memory=pl.Mem.Mat,
                                )
                                qk_tail1_inline70__ssa_v0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul(
                                    q_tile_inline56__ssa_v0, pl.tile.transpose_view(k_tail1_inline69__ssa_v0)
                                )
                                pl.tile.store(qk_tail1_inline70__ssa_v0, [produce_row_inline60__ssa_v0, 128], score_transfer__ssa_v0)
                            if produce_page_inline59__ssa_v0 + 2 < page_count_inline51__ssa_v0:
                                qk_tail_ph2_inline71__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(
                                    pl.tensor.read(block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, produce_page_inline59__ssa_v0 + 2]), pl.INDEX
                                )
                                k_tail2_inline72__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                                    key_cache_bsnd_inline4__phi_v5,
                                    [cache_base_inline9__ssa_v0 + qk_tail_ph2_inline71__ssa_v0 * 128, col_inline53__ssa_v0],
                                    [128, 128],
                                    [128, 128],
                                    target_memory=pl.Mem.Mat,
                                )
                                qk_tail2_inline73__ssa_v0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul(
                                    q_tile_inline56__ssa_v0, pl.tile.transpose_view(k_tail2_inline72__ssa_v0)
                                )
                                pl.tile.store(qk_tail2_inline73__ssa_v0, [produce_row_inline60__ssa_v0, 256], score_transfer__ssa_v0)
                        pl.system.sync_set(0, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                    if 2 <= tick_inline57__idx_v0:
                        consume_stack_inline74__ssa_v0: pl.Scalar[pl.INDEX] = tick_inline57__idx_v0 - 2
                        consume_page_inline75__ssa_v0: pl.Scalar[pl.INDEX] = consume_stack_inline74__ssa_v0 * 4
                        consume_row_inline76__ssa_v0: pl.Scalar[pl.INDEX] = transfer_base_inline54__ssa_v0 + consume_stack_inline74__ssa_v0 % 3 * 16
                        if consume_page_inline75__ssa_v0 + 4 <= page_count_inline51__ssa_v0:
                            pv_ph0_inline77__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, consume_page_inline75__ssa_v0]), pl.INDEX)
                            pv_ph1_inline78__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(
                                pl.tensor.read(block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, consume_page_inline75__ssa_v0 + 1]), pl.INDEX
                            )
                            pv_ph2_inline79__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(
                                pl.tensor.read(block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, consume_page_inline75__ssa_v0 + 2]), pl.INDEX
                            )
                            pv_ph3_inline80__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(
                                pl.tensor.read(block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, consume_page_inline75__ssa_v0 + 3]), pl.INDEX
                            )
                            v0_inline81__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[0], pl.Mem.Mat] = pl._dump.call(
                                "tile.load",
                                [value_cache_bsnd_inline5__phi_v5, [cache_base_inline9__ssa_v0 + pv_ph0_inline77__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                                pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            v1_inline82__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[1], pl.Mem.Mat] = pl._dump.call(
                                "tile.load",
                                [value_cache_bsnd_inline5__phi_v5, [cache_base_inline9__ssa_v0 + pv_ph1_inline78__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                                pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            v2_inline83__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[2], pl.Mem.Mat] = pl._dump.call(
                                "tile.load",
                                [value_cache_bsnd_inline5__phi_v5, [cache_base_inline9__ssa_v0 + pv_ph2_inline79__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                                pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            v3_inline84__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[3], pl.Mem.Mat] = pl._dump.call(
                                "tile.load",
                                [value_cache_bsnd_inline5__phi_v5, [cache_base_inline9__ssa_v0 + pv_ph3_inline80__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                                pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                            probability0_inline85__ssa_v0: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                "tile.load",
                                [probability_transfer__ssa_v0, [consume_row_inline76__ssa_v0, 0], [16, 128], [5, 128]],
                                pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            pv0_inline86__ssa_v0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul(
                                pl._dump.call(
                                    "tile.move", [probability0_inline85__ssa_v0], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}
                                ),
                                pl._dump.call("tile.move", [v0_inline81__ssa_v0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}),
                            )
                            probability1_inline87__ssa_v0: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[1], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                "tile.load",
                                [probability_transfer__ssa_v0, [consume_row_inline76__ssa_v0, 128], [16, 128], [5, 128]],
                                pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            pv1_inline88__ssa_v0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                                pv0_inline86__ssa_v0,
                                pl._dump.call(
                                    "tile.move", [probability1_inline87__ssa_v0], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}
                                ),
                                pl._dump.call("tile.move", [v1_inline82__ssa_v0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}),
                            )
                            probability2_inline89__ssa_v0: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                "tile.load",
                                [probability_transfer__ssa_v0, [consume_row_inline76__ssa_v0, 256], [16, 128], [5, 128]],
                                pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            pv2_inline90__ssa_v0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                                pv1_inline88__ssa_v0,
                                pl._dump.call(
                                    "tile.move", [probability2_inline89__ssa_v0], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}
                                ),
                                pl._dump.call("tile.move", [v2_inline83__ssa_v0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}),
                            )
                            probability3_inline91__ssa_v0: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[1], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                "tile.load",
                                [probability_transfer__ssa_v0, [consume_row_inline76__ssa_v0, 384], [16, 128], [5, 128]],
                                pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            pv3_inline92__ssa_v0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                                pv2_inline90__ssa_v0,
                                pl._dump.call(
                                    "tile.move", [probability3_inline91__ssa_v0], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}
                                ),
                                pl._dump.call("tile.move", [v3_inline84__ssa_v0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}),
                            )
                            pl.tile.store(pv3_inline92__ssa_v0, [consume_row_inline76__ssa_v0, 0], pv_transfer__ssa_v0)
                        else:
                            tail_pages_inline93__ssa_v0: pl.Scalar[pl.INDEX] = page_count_inline51__ssa_v0 - consume_page_inline75__ssa_v0
                            if tail_pages_inline93__ssa_v0 == 1:
                                pv1_ph0_inline94__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(
                                    pl.tensor.read(block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, consume_page_inline75__ssa_v0]), pl.INDEX
                                )
                                v1_0_inline95__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[0], pl.Mem.Mat] = pl._dump.call(
                                    "tile.load",
                                    [value_cache_bsnd_inline5__phi_v5, [cache_base_inline9__ssa_v0 + pv1_ph0_inline94__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                                    pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                                probability1_0_inline96__ssa_v0: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                    "tile.load",
                                    [probability_transfer__ssa_v0, [consume_row_inline76__ssa_v0, 0], [16, 128], [5, 128]],
                                    pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                pv_tail1_inline97__ssa_v0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul(
                                    pl._dump.call(
                                        "tile.move",
                                        [probability1_0_inline96__ssa_v0],
                                        pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])],
                                        kwargs={"target_memory": pl.Mem.Left},
                                    ),
                                    pl._dump.call("tile.move", [v1_0_inline95__ssa_v0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}),
                                )
                                pl.tile.store(pv_tail1_inline97__ssa_v0, [consume_row_inline76__ssa_v0, 0], pv_transfer__ssa_v0)
                            if tail_pages_inline93__ssa_v0 == 2:
                                pv2_ph0_inline98__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(
                                    pl.tensor.read(block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, consume_page_inline75__ssa_v0]), pl.INDEX
                                )
                                pv2_ph1_inline99__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(
                                    pl.tensor.read(block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, consume_page_inline75__ssa_v0 + 1]), pl.INDEX
                                )
                                v2_0_inline100__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[0], pl.Mem.Mat] = pl._dump.call(
                                    "tile.load",
                                    [value_cache_bsnd_inline5__phi_v5, [cache_base_inline9__ssa_v0 + pv2_ph0_inline98__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                                    pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                v2_1_inline101__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[1], pl.Mem.Mat] = pl._dump.call(
                                    "tile.load",
                                    [value_cache_bsnd_inline5__phi_v5, [cache_base_inline9__ssa_v0 + pv2_ph1_inline99__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                                    pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                                probability2_0_inline102__ssa_v0: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                    "tile.load",
                                    [probability_transfer__ssa_v0, [consume_row_inline76__ssa_v0, 0], [16, 128], [5, 128]],
                                    pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                pv_tail2_0_inline103__ssa_v0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul(
                                    pl._dump.call(
                                        "tile.move",
                                        [probability2_0_inline102__ssa_v0],
                                        pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])],
                                        kwargs={"target_memory": pl.Mem.Left},
                                    ),
                                    pl._dump.call("tile.move", [v2_0_inline100__ssa_v0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}),
                                )
                                probability2_1_inline104__ssa_v0: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[1], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                    "tile.load",
                                    [probability_transfer__ssa_v0, [consume_row_inline76__ssa_v0, 128], [16, 128], [5, 128]],
                                    pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                pv_tail2_1_inline105__ssa_v0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                                    pv_tail2_0_inline103__ssa_v0,
                                    pl._dump.call(
                                        "tile.move",
                                        [probability2_1_inline104__ssa_v0],
                                        pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])],
                                        kwargs={"target_memory": pl.Mem.Left},
                                    ),
                                    pl._dump.call("tile.move", [v2_1_inline101__ssa_v0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}),
                                )
                                pl.tile.store(pv_tail2_1_inline105__ssa_v0, [consume_row_inline76__ssa_v0, 0], pv_transfer__ssa_v0)
                            if tail_pages_inline93__ssa_v0 == 3:
                                pv3_ph0_inline106__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(
                                    pl.tensor.read(block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, consume_page_inline75__ssa_v0]), pl.INDEX
                                )
                                pv3_ph1_inline107__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(
                                    pl.tensor.read(block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, consume_page_inline75__ssa_v0 + 1]), pl.INDEX
                                )
                                pv3_ph2_inline108__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(
                                    pl.tensor.read(block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, consume_page_inline75__ssa_v0 + 2]), pl.INDEX
                                )
                                v3_0_inline109__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[0], pl.Mem.Mat] = pl._dump.call(
                                    "tile.load",
                                    [value_cache_bsnd_inline5__phi_v5, [cache_base_inline9__ssa_v0 + pv3_ph0_inline106__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                                    pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                v3_1_inline110__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[1], pl.Mem.Mat] = pl._dump.call(
                                    "tile.load",
                                    [value_cache_bsnd_inline5__phi_v5, [cache_base_inline9__ssa_v0 + pv3_ph1_inline107__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                                    pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                v3_2_inline111__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[2], pl.Mem.Mat] = pl._dump.call(
                                    "tile.load",
                                    [value_cache_bsnd_inline5__phi_v5, [cache_base_inline9__ssa_v0 + pv3_ph2_inline108__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                                    pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                                probability3_0_inline112__ssa_v0: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                    "tile.load",
                                    [probability_transfer__ssa_v0, [consume_row_inline76__ssa_v0, 0], [16, 128], [5, 128]],
                                    pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                pv_tail3_0_inline113__ssa_v0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul(
                                    pl._dump.call(
                                        "tile.move",
                                        [probability3_0_inline112__ssa_v0],
                                        pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])],
                                        kwargs={"target_memory": pl.Mem.Left},
                                    ),
                                    pl._dump.call("tile.move", [v3_0_inline109__ssa_v0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}),
                                )
                                probability3_1_inline114__ssa_v0: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[1], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                    "tile.load",
                                    [probability_transfer__ssa_v0, [consume_row_inline76__ssa_v0, 128], [16, 128], [5, 128]],
                                    pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                pv_tail3_1_inline115__ssa_v0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                                    pv_tail3_0_inline113__ssa_v0,
                                    pl._dump.call(
                                        "tile.move",
                                        [probability3_1_inline114__ssa_v0],
                                        pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])],
                                        kwargs={"target_memory": pl.Mem.Left},
                                    ),
                                    pl._dump.call("tile.move", [v3_1_inline110__ssa_v0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}),
                                )
                                probability3_2_inline116__ssa_v0: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                    "tile.load",
                                    [probability_transfer__ssa_v0, [consume_row_inline76__ssa_v0, 256], [16, 128], [5, 128]],
                                    pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                pv_tail3_2_inline117__ssa_v0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                                    pv_tail3_1_inline115__ssa_v0,
                                    pl._dump.call(
                                        "tile.move",
                                        [probability3_2_inline116__ssa_v0],
                                        pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])],
                                        kwargs={"target_memory": pl.Mem.Left},
                                    ),
                                    pl._dump.call("tile.move", [v3_2_inline111__ssa_v0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}),
                                )
                                pl.tile.store(pv_tail3_2_inline117__ssa_v0, [consume_row_inline76__ssa_v0, 0], pv_transfer__ssa_v0)
                        pl.system.sync_set(2, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                for aiv_id_inline118__ssa_v0 in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    pl.system.set_ffts(ffts_workspace__ssa_v0)
                    lane_row_inline119__ssa_v0: pl.Scalar[pl.INDEX] = aiv_id_inline118__ssa_v0 * 2
                    lane_rows_inline120__ssa_v0: pl.Scalar[pl.INDEX] = aiv_id_inline118__ssa_v0 + 2
                    tmp_inline121__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    zero_scores_inline122__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.full([8, 512], dtype=pl.FP32, value=pl.const(0.0, pl.FP32))
                    zero_scores_valid_inline123__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 512])] = pl.tile.set_validshape(
                        zero_scores_inline122__ssa_v0, lane_rows_inline120__ssa_v0, 512
                    )
                    zrow_inline124__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = pl.tile.muls(
                        pl.tile.row_max(zero_scores_valid_inline123__ssa_v0, tmp_inline121__ssa_v0), 0.0
                    )
                    m_inline125__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = pl.tile.adds(zrow_inline124__ssa_v0, -3.0000000000000001e38)
                    l_sum_inline126__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = zrow_inline124__ssa_v0
                    o_init_inline127__ssa_v0: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.full([8, 128], dtype=pl.FP32, value=pl.const(0.0, pl.FP32))
                    o_inline128__ssa_v0: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 128])] = pl.tile.set_validshape(
                        o_init_inline127__ssa_v0, lane_rows_inline120__ssa_v0, 128
                    )
                    rescale_pending0_inline129__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = pl.tile.adds(zrow_inline124__ssa_v0, 1.0)
                    rescale_pending1_inline130__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = pl.tile.adds(zrow_inline124__ssa_v0, 1.0)
                    for tick_inline131__idx_v0, (m_iter, l_iter, o_iter, pending0_iter, pending1_iter) in pl.range(
                        stack_count_inline52__ssa_v0 + 2,
                        init_values=(m_inline125__ssa_v0, l_sum_inline126__ssa_v0, o_inline128__ssa_v0, rescale_pending0_inline129__ssa_v0, rescale_pending1_inline130__ssa_v0),
                    ):
                        if tick_inline131__idx_v0 < stack_count_inline52__ssa_v0:
                            produce_stack_inline58__ssa_v1: pl.Scalar[pl.INDEX] = tick_inline131__idx_v0
                            produce_row_inline60__ssa_v1: pl.Scalar[pl.INDEX] = transfer_base_inline54__ssa_v0 + produce_stack_inline58__ssa_v1 % 3 * 16
                            pl.system.sync_wait(0, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                            score_aiv_inline140__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 512])] = pl.tile.load(
                                score_transfer__ssa_v0, [produce_row_inline60__ssa_v1 + lane_row_inline119__ssa_v0, 0], [8, 512], [lane_rows_inline120__ssa_v0, 512], target_memory=pl.Mem.Vec
                            )
                            score_scaled_inline141__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 512])] = pl.tile.muls(
                                score_aiv_inline140__ssa_v0, 0.088388347648318433
                            )
                            valid_cols_inline142__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(pl.cast(seq_len_inline50__ssa_v0, pl.INDEX) - produce_stack_inline58__ssa_v1 * 512, 512)
                            score_valid_inline143__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, valid_cols_inline142__ssa_v0])] = (
                                pl.tile.set_validshape(score_scaled_inline141__ssa_v0, lane_rows_inline120__ssa_v0, valid_cols_inline142__ssa_v0)
                            )
                            score_filled_inline144__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(
                                score_valid_inline143__ssa_v0, pad_value=pl.PadValue.min
                            )
                            score_masked_inline145__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 512], pad=pl.PadValue.min)] = (
                                pl.tile.set_validshape(score_filled_inline144__ssa_v0, lane_rows_inline120__ssa_v0, 512)
                            )
                            local_m_inline146__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = pl.tile.row_max(
                                score_masked_inline145__ssa_v0, tmp_inline121__ssa_v0
                            )
                            next_m_inline147__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = pl.tile.maximum(
                                local_m_inline146__ssa_v0, m_iter
                            )
                            rescale_inline148__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = pl.tile.exp(
                                pl.tile.sub(m_iter, next_m_inline147__ssa_v0)
                            )
                            probability_aiv_inline149__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 512], pad=pl.PadValue.min)] = pl.tile.exp(
                                pl.tile.row_expand_sub(score_masked_inline145__ssa_v0, next_m_inline147__ssa_v0)
                            )
                            probability_reduce_inline150__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 512], pad=pl.PadValue.min)] = (
                                pl.tile.move(probability_aiv_inline149__ssa_v0, target_memory=pl.Mem.Vec, slayout=pl.TileLayout.none_box)
                            )
                            next_l_inline151__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = pl.tile.add(
                                pl.tile.mul(rescale_inline148__ssa_v0, l_iter), pl.tile.row_sum(probability_reduce_inline150__ssa_v0, tmp_inline121__ssa_v0)
                            )
                            probability_bf16_inline152__ssa_v0: pl.Tile[[8, 512], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 512], pad=pl.PadValue.min)] = pl.tile.cast(
                                probability_aiv_inline149__ssa_v0, target_type=pl.BF16, mode="rint"
                            )
                            probability_valid_inline153__ssa_v0: pl.Tile[[8, 512], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 512], pad=pl.PadValue.min)] = (
                                pl.tile.set_validshape(probability_bf16_inline152__ssa_v0, lane_rows_inline120__ssa_v0, 512)
                            )
                            pl.tile.store(probability_valid_inline153__ssa_v0, [produce_row_inline60__ssa_v1 + lane_row_inline119__ssa_v0, 0], probability_transfer__ssa_v0)
                            pl.system.sync_set(1, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
                            m_after_inline137__rv_v0, l_after_inline138__rv_v0, rescale_after_inline139__rv_v0 = pl._dump.yield_(
                                next_m_inline147__ssa_v0,
                                next_l_inline151__ssa_v0,
                                rescale_inline148__ssa_v0,
                                types=[
                                    pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])],
                                    pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])],
                                    pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])],
                                ],
                            )
                        else:
                            m_after_inline137__rv_v0, l_after_inline138__rv_v0, rescale_after_inline139__rv_v0 = pl._dump.yield_(
                                m_iter,
                                l_iter,
                                pending1_iter,
                                types=[
                                    pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])],
                                    pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])],
                                    pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])],
                                ],
                            )
                        if 2 <= tick_inline131__idx_v0:
                            consume_stack_inline74__ssa_v1: pl.Scalar[pl.INDEX] = tick_inline131__idx_v0 - 2
                            consume_row_inline76__ssa_v1: pl.Scalar[pl.INDEX] = transfer_base_inline54__ssa_v0 + consume_stack_inline74__ssa_v1 % 3 * 16
                            pl.system.sync_wait(2, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                            pv_aiv_inline155__ssa_v0: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 128])] = pl.tile.load(
                                pv_transfer__ssa_v0, [consume_row_inline76__ssa_v1 + lane_row_inline119__ssa_v0, 0], [8, 128], [lane_rows_inline120__ssa_v0, 128], target_memory=pl.Mem.Vec
                            )
                            o_after_inline154__rv_v0: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 128])] = pl.yield_(
                                pl.tile.add(pl.tile.row_expand_mul(o_iter, pending0_iter), pv_aiv_inline155__ssa_v0)
                            )
                        else:
                            o_after_inline154__rv_v0: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 128])] = pl.yield_(o_iter)
                        pending0_after_inline156__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = pl.tile.move(
                            pending1_iter, target_memory=pl.Mem.Vec
                        )
                        pending1_after_inline157__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = pl.tile.move(
                            rescale_after_inline139__rv_v0, target_memory=pl.Mem.Vec
                        )
                        m_inline132, l_sum_inline133, o_inline134, rescale_pending0_inline135, rescale_pending1_inline136 = pl._dump.yield_(
                            m_after_inline137__rv_v0,
                            l_after_inline138__rv_v0,
                            o_after_inline154__rv_v0,
                            pending0_after_inline156__ssa_v0,
                            pending1_after_inline157__ssa_v0,
                            types=[
                                pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])],
                                pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])],
                                pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 128])],
                                pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])],
                                pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])],
                            ],
                        )
                    ctx_inline158__ssa_v0: pl.Tile[[8, 128], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 128])] = pl.tile.cast(
                        pl.tile.row_expand_mul(o_inline134, pl.tile.recip(l_sum_inline133)), target_type=pl.BF16, mode="rint"
                    )
                    pl.tile.store(ctx_inline158__ssa_v0, [batch_inline49__ssa_v0 * 40 + kv_head_inline19__ssa_v1 * 5 + lane_row_inline119__ssa_v0, 0], out2d_inline10__ssa_v0)
        return out__ssa_v0
