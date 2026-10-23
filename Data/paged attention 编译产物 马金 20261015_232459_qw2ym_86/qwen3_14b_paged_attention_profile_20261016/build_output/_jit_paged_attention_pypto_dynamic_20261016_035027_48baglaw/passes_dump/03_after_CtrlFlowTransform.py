# pypto.program: _jit_paged_attention_pypto_dynamic
import pypto.language as pl


@pl.program
class _jit_paged_attention_pypto_dynamic:
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def paged_attention_pypto_dynamic(
        self,
        key_cache: pl.InOut[pl.Tensor[[10368, 1024], pl.BF16]],
        value_cache: pl.InOut[pl.Tensor[[10368, 1024], pl.BF16]],
        block_table: pl.Tensor[[64], pl.INT32],
        seq_lens: pl.Tensor[[2], pl.INT32],
        inv_rms_states: pl.Tensor[[2, 1], pl.FP32],
        slot_mapping: pl.Tensor[[2], pl.INT32],
        rope_cos: pl.Tensor[[4096, 128], pl.FP32],
        rope_sin: pl.Tensor[[4096, 128], pl.FP32],
        q_proj: pl.Tensor[[2, 5120], pl.FP32],
        k_proj: pl.Tensor[[2, 1024], pl.FP32],
        v_proj: pl.Tensor[[2, 1024], pl.FP32],
        q_norm_w: pl.Tensor[[1, 128], pl.FP32],
        k_norm_w: pl.Tensor[[1, 128], pl.FP32],
        layer_cache_base_token_rows: pl.Scalar[pl.INDEX],
        out: pl.Out[pl.Tensor[[2, 40, 128], pl.BF16]],
    ) -> pl.Tensor[[2, 40, 128], pl.BF16]:
        # Allocate fused Phase-0/PA scratch and launch the PyPTO helper.
        active_batch: pl.Scalar[pl.INDEX] = pl.tensor.dim(seq_lens, 0)
        q_tnd_flat: pl.Tensor[[active_batch * pl.const(40, pl.INDEX), 128], pl.BF16] = pl.tensor.create([active_batch * 40, 128], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        score_transfer: pl.Tensor[[1152, 512], pl.FP32] = pl.tensor.create([1152, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        probability_transfer: pl.Tensor[[1152, 512], pl.BF16] = pl.tensor.create([1152, 512], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        pv_transfer: pl.Tensor[[1152, 128], pl.FP32] = pl.tensor.create([1152, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        ffts_workspace: pl.Tensor[[256], pl.INT64] = pl.tensor.create([256], dtype=pl.INT64, layout=pl.TensorLayout.ND)
        q_proj_tid: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[])
        k_proj_tid: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[])
        v_proj_tid: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[])
        rms_tid: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[])
        attn_out_seed_tid: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[])
        mlp_out_seed_tid: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[])
        scratch_ready_tid: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[])
        # Run fused Q/K norm, RoPE, cache append, and dynamic paged attention.
        active_batch_inline0: pl.Scalar[pl.INDEX] = pl.tensor.dim(seq_lens, 0)
        num_tasks_inline1: pl.Scalar[pl.INDEX] = active_batch_inline0 * 8
        query_rows_inline2: pl.Scalar[pl.INDEX] = pl.tensor.dim(q_tnd_flat, 0)
        cache_token_rows_inline3: pl.Scalar[pl.INDEX] = pl.tensor.dim(key_cache, 0) * pl.tensor.dim(key_cache, 1) // 1024
        key_cache_bsnd_inline4: pl.Tensor[[cache_token_rows_inline3, 1024], pl.BF16] = pl.tensor.reshape(key_cache, [cache_token_rows_inline3, 1024])
        value_cache_bsnd_inline5: pl.Tensor[[cache_token_rows_inline3, 1024], pl.BF16] = pl.tensor.reshape(value_cache, [cache_token_rows_inline3, 1024])
        max_blocks_per_seq_inline6: pl.Scalar[pl.INDEX] = pl.tensor.dim(block_table, 0) // active_batch_inline0
        block_table_2d_inline7: pl.Tensor[[active_batch_inline0, max_blocks_per_seq_inline6], pl.INT32] = pl.tensor.reshape(block_table, [active_batch_inline0, max_blocks_per_seq_inline6])
        q2d_inline8: pl.Tensor[[query_rows_inline2, 128], pl.BF16] = pl.tensor.reshape(q_tnd_flat, [query_rows_inline2, 128])
        cache_base_inline9: pl.Scalar[pl.INDEX] = layer_cache_base_token_rows
        out2d_inline10: pl.Tensor[[query_rows_inline2, 128], pl.BF16] = pl.tensor.reshape(out, [query_rows_inline2, 128])
        with pl.spmd(
            24, sync_start=True, name_hint="attn_swpipe_spmd", deps=[q_proj_tid, k_proj_tid, v_proj_tid, rms_tid, attn_out_seed_tid, mlp_out_seed_tid, scratch_ready_tid], allow_early_resolve=True
        ) as attn_tid_inline11:
            core_inline12: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
            pl.system.set_ffts(ffts_workspace)
            for aiv_id_inline13 in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                rope_core_inline14: pl.Scalar[pl.INDEX] = core_inline12 * 2 + aiv_id_inline13
                if rope_core_inline14 < 32:
                    q_red_pad_inline15: pl.Tensor[[1, 1408], pl.FP32] = pl.tensor.full([1, 1408], dtype=pl.FP32, value=0.0)
                    k_red_pad_inline16: pl.Tensor[[1, 896], pl.FP32] = pl.tensor.full([1, 896], dtype=pl.FP32, value=0.0)
                    for it_inline17 in pl.pipeline(4, stage=2):
                        g_idx_inline18: pl.Scalar[pl.INDEX] = rope_core_inline14 + it_inline17 * 32
                        if g_idx_inline18 < 8 * active_batch_inline0:
                            kv_head_inline19: pl.Scalar[pl.INDEX] = g_idx_inline18 // active_batch_inline0
                            batch_idx_inline20: pl.Scalar[pl.INDEX] = g_idx_inline18 - kv_head_inline19 * active_batch_inline0
                            position_inline21: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(seq_lens, [batch_idx_inline20]), pl.INDEX) - 1
                            inv_rms_inline22: pl.Scalar[pl.FP32] = pl.tensor.read(inv_rms_states, [batch_idx_inline20, 0])
                            write_slot_inline23: pl.Scalar[pl.INDEX] = pl.max(pl.cast(pl.tensor.read(slot_mapping, [batch_idx_inline20]), pl.INDEX), 0)
                            cos_lo_inline24: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_cos, [1, 64], [position_inline21, 0])
                            cos_hi_inline25: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_cos, [1, 64], [position_inline21, 64])
                            sin_lo_inline26: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_sin, [1, 64], [position_inline21, 0])
                            sin_hi_inline27: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_sin, [1, 64], [position_inline21, 64])
                            cache_col_inline28: pl.Scalar[pl.INDEX] = kv_head_inline19 * 128
                            k_raw_inline29: pl.Tensor[[8, 128], pl.FP32] = pl.tensor.muls(
                                pl.tensor.reshape(pl.tensor.concat(pl.tensor.slice(k_proj, [1, 128], [batch_idx_inline20, cache_col_inline28]), k_red_pad_inline16), [8, 128]), inv_rms_inline22
                            )
                            k_ss_inline30: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.row_sum(pl.tensor.mul(k_raw_inline29, k_raw_inline29))
                            k_inv_inline31: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(pl.tensor.adds(pl.tensor.muls(k_ss_inline30, 0.0078125), 9.9999999999999995e-07)))
                            k_normed_inline32: pl.Tensor[[8, 128], pl.FP32] = pl.tensor.row_expand_mul(pl.tensor.col_expand_mul(k_raw_inline29, k_norm_w), k_inv_inline31)
                            k_full_inline33: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.slice(k_normed_inline32, [1, 128], [0, 0])
                            k_lo_inline34: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(k_full_inline33, [1, 64], [0, 0])
                            k_hi_inline35: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(k_full_inline33, [1, 64], [0, 64])
                            k_rotated_inline36: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.concat(
                                pl.tensor.sub(pl.tensor.col_expand_mul(k_lo_inline34, cos_lo_inline24), pl.tensor.col_expand_mul(k_hi_inline35, sin_lo_inline26)),
                                pl.tensor.add(pl.tensor.col_expand_mul(k_hi_inline35, cos_hi_inline25), pl.tensor.col_expand_mul(k_lo_inline34, sin_hi_inline27)),
                            )
                            cache_row_inline37: pl.Scalar[pl.INDEX] = layer_cache_base_token_rows + write_slot_inline23
                            key_cache_bsnd_inline4: pl.Tensor[[cache_token_rows_inline3, 1024], pl.BF16] = pl.tensor.assemble(
                                key_cache_bsnd_inline4, pl.tensor.cast(k_rotated_inline36, target_type=pl.BF16, mode="round"), [cache_row_inline37, cache_col_inline28]
                            )
                            v_row_bf16_inline38: pl.Tensor[[1, 128], pl.BF16] = pl.tensor.cast(
                                pl.tensor.muls(pl.tensor.slice(v_proj, [1, 128], [batch_idx_inline20, cache_col_inline28]), inv_rms_inline22), target_type=pl.BF16, mode="round"
                            )
                            value_cache_bsnd_inline5: pl.Tensor[[cache_token_rows_inline3, 1024], pl.BF16] = pl.tensor.assemble(
                                value_cache_bsnd_inline5, v_row_bf16_inline38, [cache_row_inline37, cache_col_inline28]
                            )
                            q_head_base_inline39: pl.Scalar[pl.INDEX] = kv_head_inline19 * 5
                            q_raw_inline40: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.muls(
                                pl.tensor.reshape(pl.tensor.concat(pl.tensor.slice(q_proj, [1, 640], [batch_idx_inline20, q_head_base_inline39 * 128]), q_red_pad_inline15), [16, 128]),
                                inv_rms_inline22,
                            )
                            q_ss_inline41: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.row_sum(pl.tensor.mul(q_raw_inline40, q_raw_inline40))
                            q_inv_inline42: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(pl.tensor.adds(pl.tensor.muls(q_ss_inline41, 0.0078125), 9.9999999999999995e-07)))
                            q_heads_inline43: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.row_expand_mul(pl.tensor.col_expand_mul(q_raw_inline40, q_norm_w), q_inv_inline42)
                            q_lo_inline44: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(q_heads_inline43, [16, 64], [0, 0])
                            q_hi_inline45: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(q_heads_inline43, [16, 64], [0, 64])
                            q_rotated_inline46: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.concat(
                                pl.tensor.sub(pl.tensor.col_expand_mul(q_lo_inline44, cos_lo_inline24), pl.tensor.col_expand_mul(q_hi_inline45, sin_lo_inline26)),
                                pl.tensor.add(pl.tensor.col_expand_mul(q_hi_inline45, cos_hi_inline25), pl.tensor.col_expand_mul(q_lo_inline44, sin_hi_inline27)),
                            )
                            q_row_inline47: pl.Scalar[pl.INDEX] = batch_idx_inline20 * 40 + q_head_base_inline39
                            q2d_inline8: pl.Tensor[[query_rows_inline2, 128], pl.BF16] = pl.tensor.assemble(
                                q2d_inline8, pl.tensor.cast(pl.tensor.slice(q_rotated_inline46, [5, 128], [0, 0]), target_type=pl.BF16, mode="round"), [q_row_inline47, 0]
                            )
            pl.system.cacheinvalid()
            pl.system.fence()
            pl.system.syncall(core_type=pl.KernelType.MIX)
            pl.system.cacheinvalid()
            for task_inline48 in pl.range(core_inline12, num_tasks_inline1, 24):
                batch_inline49: pl.Scalar[pl.INDEX] = task_inline48 // 8
                kv_head_inline19: pl.Scalar[pl.INDEX] = task_inline48 % 8
                seq_len_inline50: pl.Scalar[pl.INT32] = pl.tensor.read(seq_lens, [batch_inline49])
                page_count_inline51: pl.Scalar[pl.INDEX] = (pl.cast(seq_len_inline50, pl.INDEX) + 128 - 1) // 128
                stack_count_inline52: pl.Scalar[pl.INDEX] = (pl.cast(seq_len_inline50, pl.INDEX) + 512 - 1) // 512
                col_inline53: pl.Scalar[pl.INDEX] = kv_head_inline19 * 128
                transfer_base_inline54: pl.Scalar[pl.INDEX] = core_inline12 * 3 * 16
                qp_row_inline55: pl.Scalar[pl.INDEX] = batch_inline49 * 40 + kv_head_inline19 * 5
                q_tile_inline56: pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl.tile.load(
                    q2d_inline8, [qp_row_inline55, 0], [16, 128], [5, 128], target_memory=pl.Mem.Mat
                )
                for tick_inline57 in pl.range(stack_count_inline52 + 2):
                    if tick_inline57 < stack_count_inline52:
                        produce_stack_inline58: pl.Scalar[pl.INDEX] = tick_inline57
                        produce_page_inline59: pl.Scalar[pl.INDEX] = produce_stack_inline58 * 4
                        produce_row_inline60: pl.Scalar[pl.INDEX] = transfer_base_inline54 + produce_stack_inline58 % 3 * 16
                        if produce_page_inline59 + 4 <= page_count_inline51:
                            for qk_page_offset_inline61 in pl.pipeline(4, stage=2):
                                qk_ph_inline62: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(block_table_2d_inline7, [batch_inline49, produce_page_inline59 + qk_page_offset_inline61]), pl.INDEX)
                                qk_k_page_inline63: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                                    key_cache_bsnd_inline4, [cache_base_inline9 + qk_ph_inline62 * 128, col_inline53], [128, 128], [128, 128], target_memory=pl.Mem.Mat
                                )
                                qk_score_page_inline64: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul(
                                    q_tile_inline56, pl.tile.transpose_view(qk_k_page_inline63)
                                )
                                pl.tile.store(qk_score_page_inline64, [produce_row_inline60, qk_page_offset_inline61 * 128], score_transfer)
                        else:
                            qk_tail_ph0_inline65: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(block_table_2d_inline7, [batch_inline49, produce_page_inline59]), pl.INDEX)
                            k_tail0_inline66: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                                key_cache_bsnd_inline4, [cache_base_inline9 + qk_tail_ph0_inline65 * 128, col_inline53], [128, 128], [128, 128], target_memory=pl.Mem.Mat
                            )
                            qk_tail0_inline67: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul(
                                q_tile_inline56, pl.tile.transpose_view(k_tail0_inline66)
                            )
                            pl.tile.store(qk_tail0_inline67, [produce_row_inline60, 0], score_transfer)
                            if produce_page_inline59 + 1 < page_count_inline51:
                                qk_tail_ph1_inline68: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(block_table_2d_inline7, [batch_inline49, produce_page_inline59 + 1]), pl.INDEX)
                                k_tail1_inline69: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                                    key_cache_bsnd_inline4, [cache_base_inline9 + qk_tail_ph1_inline68 * 128, col_inline53], [128, 128], [128, 128], target_memory=pl.Mem.Mat
                                )
                                qk_tail1_inline70: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul(
                                    q_tile_inline56, pl.tile.transpose_view(k_tail1_inline69)
                                )
                                pl.tile.store(qk_tail1_inline70, [produce_row_inline60, 128], score_transfer)
                            if produce_page_inline59 + 2 < page_count_inline51:
                                qk_tail_ph2_inline71: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(block_table_2d_inline7, [batch_inline49, produce_page_inline59 + 2]), pl.INDEX)
                                k_tail2_inline72: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                                    key_cache_bsnd_inline4, [cache_base_inline9 + qk_tail_ph2_inline71 * 128, col_inline53], [128, 128], [128, 128], target_memory=pl.Mem.Mat
                                )
                                qk_tail2_inline73: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul(
                                    q_tile_inline56, pl.tile.transpose_view(k_tail2_inline72)
                                )
                                pl.tile.store(qk_tail2_inline73, [produce_row_inline60, 256], score_transfer)
                        pl.system.sync_set(0, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                    if tick_inline57 >= 2:
                        consume_stack_inline74: pl.Scalar[pl.INDEX] = tick_inline57 - 2
                        consume_page_inline75: pl.Scalar[pl.INDEX] = consume_stack_inline74 * 4
                        consume_row_inline76: pl.Scalar[pl.INDEX] = transfer_base_inline54 + consume_stack_inline74 % 3 * 16
                        if consume_page_inline75 + 4 <= page_count_inline51:
                            pv_ph0_inline77: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(block_table_2d_inline7, [batch_inline49, consume_page_inline75]), pl.INDEX)
                            pv_ph1_inline78: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(block_table_2d_inline7, [batch_inline49, consume_page_inline75 + 1]), pl.INDEX)
                            pv_ph2_inline79: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(block_table_2d_inline7, [batch_inline49, consume_page_inline75 + 2]), pl.INDEX)
                            pv_ph3_inline80: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(block_table_2d_inline7, [batch_inline49, consume_page_inline75 + 3]), pl.INDEX)
                            v0_inline81: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[0], pl.Mem.Mat] = pl._dump.call(
                                "tile.load",
                                [value_cache_bsnd_inline5, [cache_base_inline9 + pv_ph0_inline77 * 128, col_inline53], [128, 128], [128, 128]],
                                pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            v1_inline82: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[1], pl.Mem.Mat] = pl._dump.call(
                                "tile.load",
                                [value_cache_bsnd_inline5, [cache_base_inline9 + pv_ph1_inline78 * 128, col_inline53], [128, 128], [128, 128]],
                                pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            v2_inline83: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[2], pl.Mem.Mat] = pl._dump.call(
                                "tile.load",
                                [value_cache_bsnd_inline5, [cache_base_inline9 + pv_ph2_inline79 * 128, col_inline53], [128, 128], [128, 128]],
                                pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            v3_inline84: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[3], pl.Mem.Mat] = pl._dump.call(
                                "tile.load",
                                [value_cache_bsnd_inline5, [cache_base_inline9 + pv_ph3_inline80 * 128, col_inline53], [128, 128], [128, 128]],
                                pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                            probability0_inline85: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                "tile.load",
                                [probability_transfer, [consume_row_inline76, 0], [16, 128], [5, 128]],
                                pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            pv0_inline86: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul(
                                pl._dump.call("tile.move", [probability0_inline85], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}),
                                pl._dump.call("tile.move", [v0_inline81], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}),
                            )
                            probability1_inline87: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[1], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                "tile.load",
                                [probability_transfer, [consume_row_inline76, 128], [16, 128], [5, 128]],
                                pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            pv1_inline88: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                                pv0_inline86,
                                pl._dump.call("tile.move", [probability1_inline87], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}),
                                pl._dump.call("tile.move", [v1_inline82], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}),
                            )
                            probability2_inline89: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                "tile.load",
                                [probability_transfer, [consume_row_inline76, 256], [16, 128], [5, 128]],
                                pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            pv2_inline90: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                                pv1_inline88,
                                pl._dump.call("tile.move", [probability2_inline89], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}),
                                pl._dump.call("tile.move", [v2_inline83], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}),
                            )
                            probability3_inline91: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[1], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                "tile.load",
                                [probability_transfer, [consume_row_inline76, 384], [16, 128], [5, 128]],
                                pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            pv3_inline92: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                                pv2_inline90,
                                pl._dump.call("tile.move", [probability3_inline91], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}),
                                pl._dump.call("tile.move", [v3_inline84], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}),
                            )
                            pl.tile.store(pv3_inline92, [consume_row_inline76, 0], pv_transfer)
                        else:
                            tail_pages_inline93: pl.Scalar[pl.INDEX] = page_count_inline51 - consume_page_inline75
                            if tail_pages_inline93 == 1:
                                pv1_ph0_inline94: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(block_table_2d_inline7, [batch_inline49, consume_page_inline75]), pl.INDEX)
                                v1_0_inline95: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[0], pl.Mem.Mat] = pl._dump.call(
                                    "tile.load",
                                    [value_cache_bsnd_inline5, [cache_base_inline9 + pv1_ph0_inline94 * 128, col_inline53], [128, 128], [128, 128]],
                                    pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                                probability1_0_inline96: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                    "tile.load",
                                    [probability_transfer, [consume_row_inline76, 0], [16, 128], [5, 128]],
                                    pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                pv_tail1_inline97: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul(
                                    pl._dump.call(
                                        "tile.move", [probability1_0_inline96], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}
                                    ),
                                    pl._dump.call("tile.move", [v1_0_inline95], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}),
                                )
                                pl.tile.store(pv_tail1_inline97, [consume_row_inline76, 0], pv_transfer)
                            if tail_pages_inline93 == 2:
                                pv2_ph0_inline98: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(block_table_2d_inline7, [batch_inline49, consume_page_inline75]), pl.INDEX)
                                pv2_ph1_inline99: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(block_table_2d_inline7, [batch_inline49, consume_page_inline75 + 1]), pl.INDEX)
                                v2_0_inline100: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[0], pl.Mem.Mat] = pl._dump.call(
                                    "tile.load",
                                    [value_cache_bsnd_inline5, [cache_base_inline9 + pv2_ph0_inline98 * 128, col_inline53], [128, 128], [128, 128]],
                                    pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                v2_1_inline101: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[1], pl.Mem.Mat] = pl._dump.call(
                                    "tile.load",
                                    [value_cache_bsnd_inline5, [cache_base_inline9 + pv2_ph1_inline99 * 128, col_inline53], [128, 128], [128, 128]],
                                    pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                                probability2_0_inline102: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                    "tile.load",
                                    [probability_transfer, [consume_row_inline76, 0], [16, 128], [5, 128]],
                                    pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                pv_tail2_0_inline103: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul(
                                    pl._dump.call(
                                        "tile.move", [probability2_0_inline102], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}
                                    ),
                                    pl._dump.call("tile.move", [v2_0_inline100], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}),
                                )
                                probability2_1_inline104: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[1], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                    "tile.load",
                                    [probability_transfer, [consume_row_inline76, 128], [16, 128], [5, 128]],
                                    pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                pv_tail2_1_inline105: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                                    pv_tail2_0_inline103,
                                    pl._dump.call(
                                        "tile.move", [probability2_1_inline104], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}
                                    ),
                                    pl._dump.call("tile.move", [v2_1_inline101], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}),
                                )
                                pl.tile.store(pv_tail2_1_inline105, [consume_row_inline76, 0], pv_transfer)
                            if tail_pages_inline93 == 3:
                                pv3_ph0_inline106: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(block_table_2d_inline7, [batch_inline49, consume_page_inline75]), pl.INDEX)
                                pv3_ph1_inline107: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(block_table_2d_inline7, [batch_inline49, consume_page_inline75 + 1]), pl.INDEX)
                                pv3_ph2_inline108: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(block_table_2d_inline7, [batch_inline49, consume_page_inline75 + 2]), pl.INDEX)
                                v3_0_inline109: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[0], pl.Mem.Mat] = pl._dump.call(
                                    "tile.load",
                                    [value_cache_bsnd_inline5, [cache_base_inline9 + pv3_ph0_inline106 * 128, col_inline53], [128, 128], [128, 128]],
                                    pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                v3_1_inline110: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[1], pl.Mem.Mat] = pl._dump.call(
                                    "tile.load",
                                    [value_cache_bsnd_inline5, [cache_base_inline9 + pv3_ph1_inline107 * 128, col_inline53], [128, 128], [128, 128]],
                                    pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                v3_2_inline111: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[2], pl.Mem.Mat] = pl._dump.call(
                                    "tile.load",
                                    [value_cache_bsnd_inline5, [cache_base_inline9 + pv3_ph2_inline108 * 128, col_inline53], [128, 128], [128, 128]],
                                    pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                                probability3_0_inline112: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                    "tile.load",
                                    [probability_transfer, [consume_row_inline76, 0], [16, 128], [5, 128]],
                                    pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                pv_tail3_0_inline113: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul(
                                    pl._dump.call(
                                        "tile.move", [probability3_0_inline112], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}
                                    ),
                                    pl._dump.call("tile.move", [v3_0_inline109], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}),
                                )
                                probability3_1_inline114: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[1], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                    "tile.load",
                                    [probability_transfer, [consume_row_inline76, 128], [16, 128], [5, 128]],
                                    pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                pv_tail3_1_inline115: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                                    pv_tail3_0_inline113,
                                    pl._dump.call(
                                        "tile.move", [probability3_1_inline114], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}
                                    ),
                                    pl._dump.call("tile.move", [v3_1_inline110], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}),
                                )
                                probability3_2_inline116: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                    "tile.load",
                                    [probability_transfer, [consume_row_inline76, 256], [16, 128], [5, 128]],
                                    pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                pv_tail3_2_inline117: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                                    pv_tail3_1_inline115,
                                    pl._dump.call(
                                        "tile.move", [probability3_2_inline116], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}
                                    ),
                                    pl._dump.call("tile.move", [v3_2_inline111], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}),
                                )
                                pl.tile.store(pv_tail3_2_inline117, [consume_row_inline76, 0], pv_transfer)
                        pl.system.sync_set(2, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                for aiv_id_inline118 in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    pl.system.set_ffts(ffts_workspace)
                    lane_row_inline119: pl.Scalar[pl.INDEX] = aiv_id_inline118 * 2
                    lane_rows_inline120: pl.Scalar[pl.INDEX] = 2 + aiv_id_inline118
                    tmp_inline121: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    zero_scores_inline122: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.full([8, 512], dtype=pl.FP32, value=pl.const(0.0, pl.FP32))
                    zero_scores_valid_inline123: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 512])] = pl.tile.set_validshape(
                        zero_scores_inline122, lane_rows_inline120, 512
                    )
                    zrow_inline124: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 1])] = pl.tile.muls(
                        pl.tile.row_max(zero_scores_valid_inline123, tmp_inline121), 0.0
                    )
                    m_inline125: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 1])] = pl.tile.adds(zrow_inline124, -3.0000000000000001e38)
                    l_sum_inline126: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 1])] = zrow_inline124
                    o_init_inline127: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.full([8, 128], dtype=pl.FP32, value=pl.const(0.0, pl.FP32))
                    o_inline128: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 128])] = pl.tile.set_validshape(o_init_inline127, lane_rows_inline120, 128)
                    rescale_pending0_inline129: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 1])] = pl.tile.adds(zrow_inline124, 1.0)
                    rescale_pending1_inline130: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 1])] = pl.tile.adds(zrow_inline124, 1.0)
                    for tick_inline131, (m_iter, l_iter, o_iter, pending0_iter, pending1_iter) in pl.range(
                        stack_count_inline52 + 2, init_values=(m_inline125, l_sum_inline126, o_inline128, rescale_pending0_inline129, rescale_pending1_inline130)
                    ):
                        if tick_inline131 < stack_count_inline52:
                            produce_stack_inline58: pl.Scalar[pl.INDEX] = tick_inline131
                            produce_row_inline60: pl.Scalar[pl.INDEX] = transfer_base_inline54 + produce_stack_inline58 % 3 * 16
                            pl.system.sync_wait(0, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                            score_aiv_inline140: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 512])] = pl.tile.load(
                                score_transfer, [produce_row_inline60 + lane_row_inline119, 0], [8, 512], [lane_rows_inline120, 512], target_memory=pl.Mem.Vec
                            )
                            score_scaled_inline141: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 512])] = pl.tile.muls(
                                score_aiv_inline140, 0.088388347648318433
                            )
                            valid_cols_inline142: pl.Scalar[pl.INDEX] = pl.min(512, pl.cast(seq_len_inline50, pl.INDEX) - produce_stack_inline58 * 512)
                            score_valid_inline143: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, valid_cols_inline142])] = pl.tile.set_validshape(
                                score_scaled_inline141, lane_rows_inline120, valid_cols_inline142
                            )
                            score_filled_inline144: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(score_valid_inline143, pad_value=pl.PadValue.min)
                            score_masked_inline145: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 512], pad=pl.PadValue.min)] = pl.tile.set_validshape(
                                score_filled_inline144, lane_rows_inline120, 512
                            )
                            local_m_inline146: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 1])] = pl.tile.row_max(score_masked_inline145, tmp_inline121)
                            next_m_inline147: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 1])] = pl.tile.maximum(local_m_inline146, m_iter)
                            rescale_inline148: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 1])] = pl.tile.exp(pl.tile.sub(m_iter, next_m_inline147))
                            probability_aiv_inline149: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 512], pad=pl.PadValue.min)] = pl.tile.exp(
                                pl.tile.row_expand_sub(score_masked_inline145, next_m_inline147)
                            )
                            probability_reduce_inline150: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 512], pad=pl.PadValue.min)] = pl.tile.move(
                                probability_aiv_inline149, target_memory=pl.Mem.Vec, slayout=pl.TileLayout.none_box
                            )
                            next_l_inline151: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 1])] = pl.tile.add(
                                pl.tile.mul(rescale_inline148, l_iter), pl.tile.row_sum(probability_reduce_inline150, tmp_inline121)
                            )
                            probability_bf16_inline152: pl.Tile[[8, 512], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 512], pad=pl.PadValue.min)] = pl.tile.cast(
                                probability_aiv_inline149, target_type=pl.BF16, mode="rint"
                            )
                            probability_valid_inline153: pl.Tile[[8, 512], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 512], pad=pl.PadValue.min)] = pl.tile.set_validshape(
                                probability_bf16_inline152, lane_rows_inline120, 512
                            )
                            pl.tile.store(probability_valid_inline153, [produce_row_inline60 + lane_row_inline119, 0], probability_transfer)
                            pl.system.sync_set(1, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
                            m_after_inline137, l_after_inline138, rescale_after_inline139 = pl._dump.yield_(
                                next_m_inline147,
                                next_l_inline151,
                                rescale_inline148,
                                types=[
                                    pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 1])],
                                    pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 1])],
                                    pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 1])],
                                ],
                            )
                        else:
                            m_after_inline137, l_after_inline138, rescale_after_inline139 = pl._dump.yield_(
                                m_iter,
                                l_iter,
                                pending1_iter,
                                types=[
                                    pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 1])],
                                    pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 1])],
                                    pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 1])],
                                ],
                            )
                        if tick_inline131 >= 2:
                            consume_stack_inline74: pl.Scalar[pl.INDEX] = tick_inline131 - 2
                            consume_row_inline76: pl.Scalar[pl.INDEX] = transfer_base_inline54 + consume_stack_inline74 % 3 * 16
                            pl.system.sync_wait(2, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                            pv_aiv_inline155: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 128])] = pl.tile.load(
                                pv_transfer, [consume_row_inline76 + lane_row_inline119, 0], [8, 128], [lane_rows_inline120, 128], target_memory=pl.Mem.Vec
                            )
                            o_after_inline154: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 128])] = pl.yield_(
                                pl.tile.add(pl.tile.row_expand_mul(o_iter, pending0_iter), pv_aiv_inline155)
                            )
                        else:
                            o_after_inline154: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 128])] = pl.yield_(o_iter)
                        pending0_after_inline156: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 1])] = pl.tile.move(pending1_iter, target_memory=pl.Mem.Vec)
                        pending1_after_inline157: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 1])] = pl.tile.move(
                            rescale_after_inline139, target_memory=pl.Mem.Vec
                        )
                        m_inline132, l_sum_inline133, o_inline134, rescale_pending0_inline135, rescale_pending1_inline136 = pl._dump.yield_(
                            m_after_inline137,
                            l_after_inline138,
                            o_after_inline154,
                            pending0_after_inline156,
                            pending1_after_inline157,
                            types=[
                                pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 1])],
                                pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 1])],
                                pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 128])],
                                pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 1])],
                                pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 1])],
                            ],
                        )
                    ctx_inline158: pl.Tile[[8, 128], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120, 128])] = pl.tile.cast(
                        pl.tile.row_expand_mul(o_inline134, pl.tile.recip(l_sum_inline133)), target_type=pl.BF16, mode="rint"
                    )
                    pl.tile.store(ctx_inline158, [batch_inline49 * 40 + kv_head_inline19 * 5 + lane_row_inline119, 0], out2d_inline10)
        return out
