# pypto.program: _jit_paged_attention_pypto_dynamic
import pypto.language as pl

active_batch_inline0__ssa_v0 = pl.dynamic("active_batch_inline0__ssa_v0")


@pl.program
class _jit_paged_attention_pypto_dynamic:
    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def attn_swpipe_aic(
        ffts_workspace__ssa_v0: pl.Tensor[[256], pl.INT64],
        key_cache_bsnd_inline4__ssa_v0: pl.InOut[pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16]],
        q2d_inline8__ssa_v0: pl.InOut[pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16]],
        value_cache_bsnd_inline5__ssa_v0: pl.InOut[pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16]],
        active_batch_inline0__ssa_v0: pl.Scalar[pl.INDEX],
        seq_lens__ssa_v0: pl.Tensor[[2], pl.INT32],
        inv_rms_states__ssa_v0: pl.Tensor[[2, 1], pl.FP32],
        slot_mapping__ssa_v0: pl.Tensor[[2], pl.INT32],
        rope_cos__ssa_v0: pl.Tensor[[4096, 128], pl.FP32],
        rope_sin__ssa_v0: pl.Tensor[[4096, 128], pl.FP32],
        k_proj__ssa_v0: pl.Tensor[[2, 1024], pl.FP32],
        k_norm_w__ssa_v0: pl.Tensor[[1, 128], pl.FP32],
        layer_cache_base_token_rows__ssa_v0: pl.Scalar[pl.INDEX],
        v_proj__ssa_v0: pl.Tensor[[2, 1024], pl.FP32],
        q_proj__ssa_v0: pl.Tensor[[2, 5120], pl.FP32],
        q_norm_w__ssa_v0: pl.Tensor[[1, 128], pl.FP32],
        num_tasks_inline1__ssa_v0: pl.Scalar[pl.INDEX],
        block_table_2d_inline7__ssa_v0: pl.Tensor[[active_batch_inline0__ssa_v0, max_blocks_per_seq_inline6__ssa_v0], pl.INT32],
        cache_base_inline9__ssa_v0: pl.Scalar[pl.INDEX],
        score_transfer__ssa_v0: pl.InOut[pl.Tensor[[1152, 512], pl.FP32]],
        probability_transfer__ssa_v0: pl.InOut[pl.Tensor[[1152, 512], pl.BF16]],
        pv_transfer__ssa_v0: pl.InOut[pl.Tensor[[1152, 128], pl.FP32]],
        out2d_inline10__ssa_v0: pl.Out[pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16]],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True, "split_aiv": True})
        core_inline12__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        pl.system.set_ffts(ffts_workspace__ssa_v0)
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.syncall(core_type=pl.KernelType.MIX)
        pl.system.cacheinvalid()
        for task_inline48__idx_v0 in pl.range(core_inline12__ssa_v0, num_tasks_inline1__ssa_v0, 24):
            batch_inline49__ssa_v0: pl.Scalar[pl.INDEX] = task_inline48__idx_v0 // 8
            kv_head_inline19__ssa_v1: pl.Scalar[pl.INDEX] = task_inline48__idx_v0 % 8
            seq_len_inline50__tile: pl.Scalar[pl.INT32] = pl.tensor.read(seq_lens__ssa_v0, [batch_inline49__ssa_v0])
            page_count_inline51__ssa_v0: pl.Scalar[pl.INDEX] = (pl.cast(seq_len_inline50__tile, pl.INDEX) + 127) // 128
            stack_count_inline52__ssa_v0: pl.Scalar[pl.INDEX] = (pl.cast(seq_len_inline50__tile, pl.INDEX) + 511) // 512
            col_inline53__ssa_v0: pl.Scalar[pl.INDEX] = kv_head_inline19__ssa_v1 * 128
            transfer_base_inline54__ssa_v0: pl.Scalar[pl.INDEX] = core_inline12__ssa_v0 * 48
            qp_row_inline55__ssa_v0: pl.Scalar[pl.INDEX] = batch_inline49__ssa_v0 * 40 + kv_head_inline19__ssa_v1 * 5
            q_tile_inline56__ssa_v0: pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl.tile.load(
                q2d_inline8__ssa_v0, [qp_row_inline55__ssa_v0, 0], [16, 128], [5, 128], target_memory=pl.Mem.Mat
            )
            for tick_inline57__idx_v0 in pl.range(stack_count_inline52__ssa_v0 + 2):
                if tick_inline57__idx_v0 < stack_count_inline52__ssa_v0:
                    produce_stack_inline58__ssa_v0: pl.Scalar[pl.INDEX] = tick_inline57__idx_v0
                    produce_page_inline59__ssa_v0: pl.Scalar[pl.INDEX] = produce_stack_inline58__ssa_v0 * 4
                    produce_row_inline60__ssa_v0: pl.Scalar[pl.INDEX] = transfer_base_inline54__ssa_v0 + produce_stack_inline58__ssa_v0 % 3 * 16
                    if produce_page_inline59__ssa_v0 + 4 <= page_count_inline51__ssa_v0:
                        for qk_page_offset_inline61__idx_v0 in pl.pipeline(0, 4, 2, stage=1):
                            t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, produce_page_inline59__ssa_v0 + qk_page_offset_inline61__idx_v0])
                            qk_ph_inline62__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile, pl.INDEX)
                            qk_k_page_inline63__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                                key_cache_bsnd_inline4__ssa_v0,
                                [cache_base_inline9__ssa_v0 + qk_ph_inline62__ssa_v0 * 128, col_inline53__ssa_v0],
                                [128, 128],
                                [128, 128],
                                target_memory=pl.Mem.Mat,
                                attrs={"pipeline_membership": "0:0"},
                            )
                            t__tmp_v39: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(
                                qk_k_page_inline63__ssa_v0, attrs={"pipeline_membership": "0:0"}
                            )
                            q_tile_inline56__ssa_v0_Left: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])] = pl.tile.move(
                                q_tile_inline56__ssa_v0, target_memory=pl.Mem.Left, attrs={"pipeline_membership": "0:0"}
                            )
                            t__tmp_v39_Right: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.move(t__tmp_v39, target_memory=pl.Mem.Right, attrs={"pipeline_membership": "0:0"})
                            qk_score_page_inline64__ssa_v0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul(
                                q_tile_inline56__ssa_v0_Left, t__tmp_v39_Right
                            )
                            score_transfer__store: pl.Tensor[[1152, 512], pl.FP32] = pl.tile.store(
                                qk_score_page_inline64__ssa_v0, [produce_row_inline60__ssa_v0, qk_page_offset_inline61__idx_v0 * 128], score_transfer__ssa_v0
                            )
                            t__tile_1: pl.Scalar[pl.INT32] = pl.tensor.read(
                                block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, produce_page_inline59__ssa_v0 + (qk_page_offset_inline61__idx_v0 + 1)]
                            )
                            qk_ph_inline62__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.cast(t__tile_1, pl.INDEX)
                            qk_k_page_inline63__ssa_v0_1: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                                key_cache_bsnd_inline4__ssa_v0,
                                [cache_base_inline9__ssa_v0 + qk_ph_inline62__ssa_v0_1 * 128, col_inline53__ssa_v0],
                                [128, 128],
                                [128, 128],
                                target_memory=pl.Mem.Mat,
                                attrs={"pipeline_membership": "0:1"},
                            )
                            t__tmp_v39_1: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(
                                qk_k_page_inline63__ssa_v0_1, attrs={"pipeline_membership": "0:1"}
                            )
                            q_tile_inline56__ssa_v0_Left_1: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])] = pl.tile.move(
                                q_tile_inline56__ssa_v0, target_memory=pl.Mem.Left, attrs={"pipeline_membership": "0:1"}
                            )
                            t__tmp_v39_Right_1: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.move(t__tmp_v39_1, target_memory=pl.Mem.Right, attrs={"pipeline_membership": "0:1"})
                            qk_score_page_inline64__ssa_v0_1: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul(
                                q_tile_inline56__ssa_v0_Left_1, t__tmp_v39_Right_1
                            )
                            score_transfer__store_1: pl.Tensor[[1152, 512], pl.FP32] = pl.tile.store(
                                qk_score_page_inline64__ssa_v0_1, [produce_row_inline60__ssa_v0, (qk_page_offset_inline61__idx_v0 + 1) * 128], score_transfer__ssa_v0
                            )
                    else:
                        t__tile_2: pl.Scalar[pl.INT32] = pl.tensor.read(block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, produce_page_inline59__ssa_v0])
                        qk_tail_ph0_inline65__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_2, pl.INDEX)
                        k_tail0_inline66__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                            key_cache_bsnd_inline4__ssa_v0, [cache_base_inline9__ssa_v0 + qk_tail_ph0_inline65__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128], target_memory=pl.Mem.Mat
                        )
                        t__tmp_v41: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(
                            k_tail0_inline66__ssa_v0
                        )
                        q_tile_inline56__ssa_v0_Left_2: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])] = pl.tile.move(q_tile_inline56__ssa_v0, target_memory=pl.Mem.Left)
                        t__tmp_v41_Right: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.move(t__tmp_v41, target_memory=pl.Mem.Right)
                        qk_tail0_inline67__ssa_v0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul(
                            q_tile_inline56__ssa_v0_Left_2, t__tmp_v41_Right
                        )
                        score_transfer__store_v0: pl.Tensor[[1152, 512], pl.FP32] = pl.tile.store(qk_tail0_inline67__ssa_v0, [produce_row_inline60__ssa_v0, 0], score_transfer__ssa_v0)
                        if produce_page_inline59__ssa_v0 + 1 < page_count_inline51__ssa_v0:
                            t__tile_3: pl.Scalar[pl.INT32] = pl.tensor.read(block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, produce_page_inline59__ssa_v0 + 1])
                            qk_tail_ph1_inline68__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_3, pl.INDEX)
                            k_tail1_inline69__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                                key_cache_bsnd_inline4__ssa_v0,
                                [cache_base_inline9__ssa_v0 + qk_tail_ph1_inline68__ssa_v0 * 128, col_inline53__ssa_v0],
                                [128, 128],
                                [128, 128],
                                target_memory=pl.Mem.Mat,
                            )
                            t__tmp_v43: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(
                                k_tail1_inline69__ssa_v0
                            )
                            t__tmp_v43_Right: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.move(t__tmp_v43, target_memory=pl.Mem.Right)
                            qk_tail1_inline70__ssa_v0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul(
                                q_tile_inline56__ssa_v0_Left_2, t__tmp_v43_Right
                            )
                            score_transfer__store_v1: pl.Tensor[[1152, 512], pl.FP32] = pl.tile.store(qk_tail1_inline70__ssa_v0, [produce_row_inline60__ssa_v0, 128], score_transfer__ssa_v0)
                        if produce_page_inline59__ssa_v0 + 2 < page_count_inline51__ssa_v0:
                            t__tile_4: pl.Scalar[pl.INT32] = pl.tensor.read(block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, produce_page_inline59__ssa_v0 + 2])
                            qk_tail_ph2_inline71__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_4, pl.INDEX)
                            k_tail2_inline72__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                                key_cache_bsnd_inline4__ssa_v0,
                                [cache_base_inline9__ssa_v0 + qk_tail_ph2_inline71__ssa_v0 * 128, col_inline53__ssa_v0],
                                [128, 128],
                                [128, 128],
                                target_memory=pl.Mem.Mat,
                            )
                            t__tmp_v45: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(
                                k_tail2_inline72__ssa_v0
                            )
                            t__tmp_v45_Right: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.move(t__tmp_v45, target_memory=pl.Mem.Right)
                            qk_tail2_inline73__ssa_v0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul(
                                q_tile_inline56__ssa_v0_Left_2, t__tmp_v45_Right
                            )
                            score_transfer__store_v2: pl.Tensor[[1152, 512], pl.FP32] = pl.tile.store(qk_tail2_inline73__ssa_v0, [produce_row_inline60__ssa_v0, 256], score_transfer__ssa_v0)
                    pl.system.sync_set(0, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                if 2 <= tick_inline57__idx_v0:
                    consume_stack_inline74__ssa_v0: pl.Scalar[pl.INDEX] = tick_inline57__idx_v0 - 2
                    consume_page_inline75__ssa_v0: pl.Scalar[pl.INDEX] = consume_stack_inline74__ssa_v0 * 4
                    consume_row_inline76__ssa_v0: pl.Scalar[pl.INDEX] = transfer_base_inline54__ssa_v0 + consume_stack_inline74__ssa_v0 % 3 * 16
                    if consume_page_inline75__ssa_v0 + 4 <= page_count_inline51__ssa_v0:
                        t__tile_5: pl.Scalar[pl.INT32] = pl.tensor.read(block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, consume_page_inline75__ssa_v0])
                        pv_ph0_inline77__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_5, pl.INDEX)
                        t__tile_6: pl.Scalar[pl.INT32] = pl.tensor.read(block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, consume_page_inline75__ssa_v0 + 1])
                        pv_ph1_inline78__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_6, pl.INDEX)
                        t__tile_7: pl.Scalar[pl.INT32] = pl.tensor.read(block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, consume_page_inline75__ssa_v0 + 2])
                        pv_ph2_inline79__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_7, pl.INDEX)
                        t__tile_8: pl.Scalar[pl.INT32] = pl.tensor.read(block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, consume_page_inline75__ssa_v0 + 3])
                        pv_ph3_inline80__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_8, pl.INDEX)
                        v0_inline81__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[0], pl.Mem.Mat] = pl._dump.call(
                            "tile.load",
                            [value_cache_bsnd_inline5__ssa_v0, [cache_base_inline9__ssa_v0 + pv_ph0_inline77__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                            pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[0], pl.Mem.Mat],
                            kwargs={"target_memory": pl.Mem.Mat},
                        )
                        v1_inline82__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[1], pl.Mem.Mat] = pl._dump.call(
                            "tile.load",
                            [value_cache_bsnd_inline5__ssa_v0, [cache_base_inline9__ssa_v0 + pv_ph1_inline78__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                            pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[1], pl.Mem.Mat],
                            kwargs={"target_memory": pl.Mem.Mat},
                        )
                        v2_inline83__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[2], pl.Mem.Mat] = pl._dump.call(
                            "tile.load",
                            [value_cache_bsnd_inline5__ssa_v0, [cache_base_inline9__ssa_v0 + pv_ph2_inline79__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                            pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[2], pl.Mem.Mat],
                            kwargs={"target_memory": pl.Mem.Mat},
                        )
                        v3_inline84__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[3], pl.Mem.Mat] = pl._dump.call(
                            "tile.load",
                            [value_cache_bsnd_inline5__ssa_v0, [cache_base_inline9__ssa_v0 + pv_ph3_inline80__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                            pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[3], pl.Mem.Mat],
                            kwargs={"target_memory": pl.Mem.Mat},
                        )
                        pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                        probability0_inline85__ssa_v0: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                            "tile.load",
                            [probability_transfer__ssa_v0, [consume_row_inline76__ssa_v0, 0], [16, 128], [5, 128]],
                            pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                            kwargs={"target_memory": pl.Mem.Mat},
                        )
                        t__tmp_v50: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                            "tile.move", [probability0_inline85__ssa_v0], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}
                        )
                        t__tmp_v51: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl._dump.call(
                            "tile.move", [v0_inline81__ssa_v0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}
                        )
                        pv0_inline86__ssa_v0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul(t__tmp_v50, t__tmp_v51)
                        probability1_inline87__ssa_v0: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[1], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                            "tile.load",
                            [probability_transfer__ssa_v0, [consume_row_inline76__ssa_v0, 128], [16, 128], [5, 128]],
                            pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[1], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                            kwargs={"target_memory": pl.Mem.Mat},
                        )
                        t__tmp_v52: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                            "tile.move", [probability1_inline87__ssa_v0], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}
                        )
                        t__tmp_v53: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl._dump.call(
                            "tile.move", [v1_inline82__ssa_v0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}
                        )
                        pv1_inline88__ssa_v0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                            pv0_inline86__ssa_v0, t__tmp_v52, t__tmp_v53
                        )
                        probability2_inline89__ssa_v0: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                            "tile.load",
                            [probability_transfer__ssa_v0, [consume_row_inline76__ssa_v0, 256], [16, 128], [5, 128]],
                            pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                            kwargs={"target_memory": pl.Mem.Mat},
                        )
                        t__tmp_v54: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                            "tile.move", [probability2_inline89__ssa_v0], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}
                        )
                        t__tmp_v55: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl._dump.call(
                            "tile.move", [v2_inline83__ssa_v0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}
                        )
                        pv2_inline90__ssa_v0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                            pv1_inline88__ssa_v0, t__tmp_v54, t__tmp_v55
                        )
                        probability3_inline91__ssa_v0: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[1], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                            "tile.load",
                            [probability_transfer__ssa_v0, [consume_row_inline76__ssa_v0, 384], [16, 128], [5, 128]],
                            pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[1], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                            kwargs={"target_memory": pl.Mem.Mat},
                        )
                        t__tmp_v56: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                            "tile.move", [probability3_inline91__ssa_v0], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}
                        )
                        t__tmp_v57: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl._dump.call(
                            "tile.move", [v3_inline84__ssa_v0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}
                        )
                        pv3_inline92__ssa_v0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                            pv2_inline90__ssa_v0, t__tmp_v56, t__tmp_v57
                        )
                        pv_transfer__store: pl.Tensor[[1152, 128], pl.FP32] = pl.tile.store(pv3_inline92__ssa_v0, [consume_row_inline76__ssa_v0, 0], pv_transfer__ssa_v0)
                    else:
                        tail_pages_inline93__ssa_v0: pl.Scalar[pl.INDEX] = page_count_inline51__ssa_v0 - consume_page_inline75__ssa_v0
                        if tail_pages_inline93__ssa_v0 == 1:
                            t__tile_9: pl.Scalar[pl.INT32] = pl.tensor.read(block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, consume_page_inline75__ssa_v0])
                            pv1_ph0_inline94__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_9, pl.INDEX)
                            v1_0_inline95__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[0], pl.Mem.Mat] = pl._dump.call(
                                "tile.load",
                                [value_cache_bsnd_inline5__ssa_v0, [cache_base_inline9__ssa_v0 + pv1_ph0_inline94__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                                pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[0], pl.Mem.Mat],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                            probability1_0_inline96__ssa_v0: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                "tile.load",
                                [probability_transfer__ssa_v0, [consume_row_inline76__ssa_v0, 0], [16, 128], [5, 128]],
                                pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            t__tmp_v59: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                "tile.move", [probability1_0_inline96__ssa_v0], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}
                            )
                            t__tmp_v60: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl._dump.call(
                                "tile.move", [v1_0_inline95__ssa_v0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}
                            )
                            pv_tail1_inline97__ssa_v0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul(
                                t__tmp_v59, t__tmp_v60
                            )
                            pv_transfer__store_v0: pl.Tensor[[1152, 128], pl.FP32] = pl.tile.store(pv_tail1_inline97__ssa_v0, [consume_row_inline76__ssa_v0, 0], pv_transfer__ssa_v0)
                        if tail_pages_inline93__ssa_v0 == 2:
                            t__tile_10: pl.Scalar[pl.INT32] = pl.tensor.read(block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, consume_page_inline75__ssa_v0])
                            pv2_ph0_inline98__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_10, pl.INDEX)
                            t__tile_11: pl.Scalar[pl.INT32] = pl.tensor.read(block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, consume_page_inline75__ssa_v0 + 1])
                            pv2_ph1_inline99__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_11, pl.INDEX)
                            v2_0_inline100__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[0], pl.Mem.Mat] = pl._dump.call(
                                "tile.load",
                                [value_cache_bsnd_inline5__ssa_v0, [cache_base_inline9__ssa_v0 + pv2_ph0_inline98__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                                pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[0], pl.Mem.Mat],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            v2_1_inline101__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[1], pl.Mem.Mat] = pl._dump.call(
                                "tile.load",
                                [value_cache_bsnd_inline5__ssa_v0, [cache_base_inline9__ssa_v0 + pv2_ph1_inline99__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                                pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[1], pl.Mem.Mat],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                            probability2_0_inline102__ssa_v0: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                "tile.load",
                                [probability_transfer__ssa_v0, [consume_row_inline76__ssa_v0, 0], [16, 128], [5, 128]],
                                pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            t__tmp_v63: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                "tile.move", [probability2_0_inline102__ssa_v0], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}
                            )
                            t__tmp_v64: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl._dump.call(
                                "tile.move", [v2_0_inline100__ssa_v0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}
                            )
                            pv_tail2_0_inline103__ssa_v0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul(
                                t__tmp_v63, t__tmp_v64
                            )
                            probability2_1_inline104__ssa_v0: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[1], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                "tile.load",
                                [probability_transfer__ssa_v0, [consume_row_inline76__ssa_v0, 128], [16, 128], [5, 128]],
                                pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[1], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            t__tmp_v65: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                "tile.move", [probability2_1_inline104__ssa_v0], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}
                            )
                            t__tmp_v66: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl._dump.call(
                                "tile.move", [v2_1_inline101__ssa_v0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}
                            )
                            pv_tail2_1_inline105__ssa_v0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                                pv_tail2_0_inline103__ssa_v0, t__tmp_v65, t__tmp_v66
                            )
                            pv_transfer__store_v1: pl.Tensor[[1152, 128], pl.FP32] = pl.tile.store(pv_tail2_1_inline105__ssa_v0, [consume_row_inline76__ssa_v0, 0], pv_transfer__ssa_v0)
                        if tail_pages_inline93__ssa_v0 == 3:
                            t__tile_12: pl.Scalar[pl.INT32] = pl.tensor.read(block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, consume_page_inline75__ssa_v0])
                            pv3_ph0_inline106__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_12, pl.INDEX)
                            t__tile_13: pl.Scalar[pl.INT32] = pl.tensor.read(block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, consume_page_inline75__ssa_v0 + 1])
                            pv3_ph1_inline107__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_13, pl.INDEX)
                            t__tile_14: pl.Scalar[pl.INT32] = pl.tensor.read(block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, consume_page_inline75__ssa_v0 + 2])
                            pv3_ph2_inline108__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_14, pl.INDEX)
                            v3_0_inline109__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[0], pl.Mem.Mat] = pl._dump.call(
                                "tile.load",
                                [value_cache_bsnd_inline5__ssa_v0, [cache_base_inline9__ssa_v0 + pv3_ph0_inline106__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                                pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[0], pl.Mem.Mat],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            v3_1_inline110__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[1], pl.Mem.Mat] = pl._dump.call(
                                "tile.load",
                                [value_cache_bsnd_inline5__ssa_v0, [cache_base_inline9__ssa_v0 + pv3_ph1_inline107__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                                pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[1], pl.Mem.Mat],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            v3_2_inline111__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[2], pl.Mem.Mat] = pl._dump.call(
                                "tile.load",
                                [value_cache_bsnd_inline5__ssa_v0, [cache_base_inline9__ssa_v0 + pv3_ph2_inline108__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                                pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[2], pl.Mem.Mat],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                            probability3_0_inline112__ssa_v0: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                "tile.load",
                                [probability_transfer__ssa_v0, [consume_row_inline76__ssa_v0, 0], [16, 128], [5, 128]],
                                pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            t__tmp_v70: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                "tile.move", [probability3_0_inline112__ssa_v0], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}
                            )
                            t__tmp_v71: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl._dump.call(
                                "tile.move", [v3_0_inline109__ssa_v0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}
                            )
                            pv_tail3_0_inline113__ssa_v0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul(
                                t__tmp_v70, t__tmp_v71
                            )
                            probability3_1_inline114__ssa_v0: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[1], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                "tile.load",
                                [probability_transfer__ssa_v0, [consume_row_inline76__ssa_v0, 128], [16, 128], [5, 128]],
                                pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[1], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            t__tmp_v72: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                "tile.move", [probability3_1_inline114__ssa_v0], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}
                            )
                            t__tmp_v73: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl._dump.call(
                                "tile.move", [v3_1_inline110__ssa_v0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}
                            )
                            pv_tail3_1_inline115__ssa_v0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                                pv_tail3_0_inline113__ssa_v0, t__tmp_v72, t__tmp_v73
                            )
                            probability3_2_inline116__ssa_v0: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                "tile.load",
                                [probability_transfer__ssa_v0, [consume_row_inline76__ssa_v0, 256], [16, 128], [5, 128]],
                                pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            t__tmp_v74: pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                "tile.move", [probability3_2_inline116__ssa_v0], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}
                            )
                            t__tmp_v75: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl._dump.call(
                                "tile.move", [v3_2_inline111__ssa_v0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}
                            )
                            pv_tail3_2_inline117__ssa_v0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                                pv_tail3_1_inline115__ssa_v0, t__tmp_v74, t__tmp_v75
                            )
                            pv_transfer__store_v2: pl.Tensor[[1152, 128], pl.FP32] = pl.tile.store(pv_tail3_2_inline117__ssa_v0, [consume_row_inline76__ssa_v0, 0], pv_transfer__ssa_v0)
                    pl.system.sync_set(2, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
            pl.system.set_ffts(ffts_workspace__ssa_v0)

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def attn_swpipe_aiv(
        ffts_workspace__ssa_v0: pl.Tensor[[256], pl.INT64],
        key_cache_bsnd_inline4__ssa_v0: pl.InOut[pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16]],
        q2d_inline8__ssa_v0: pl.InOut[pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16]],
        value_cache_bsnd_inline5__ssa_v0: pl.InOut[pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16]],
        active_batch_inline0__ssa_v0: pl.Scalar[pl.INDEX],
        seq_lens__ssa_v0: pl.Tensor[[2], pl.INT32],
        inv_rms_states__ssa_v0: pl.Tensor[[2, 1], pl.FP32],
        slot_mapping__ssa_v0: pl.Tensor[[2], pl.INT32],
        rope_cos__ssa_v0: pl.Tensor[[4096, 128], pl.FP32],
        rope_sin__ssa_v0: pl.Tensor[[4096, 128], pl.FP32],
        k_proj__ssa_v0: pl.Tensor[[2, 1024], pl.FP32],
        k_norm_w__ssa_v0: pl.Tensor[[1, 128], pl.FP32],
        layer_cache_base_token_rows__ssa_v0: pl.Scalar[pl.INDEX],
        v_proj__ssa_v0: pl.Tensor[[2, 1024], pl.FP32],
        q_proj__ssa_v0: pl.Tensor[[2, 5120], pl.FP32],
        q_norm_w__ssa_v0: pl.Tensor[[1, 128], pl.FP32],
        num_tasks_inline1__ssa_v0: pl.Scalar[pl.INDEX],
        block_table_2d_inline7__ssa_v0: pl.Tensor[[active_batch_inline0__ssa_v0, max_blocks_per_seq_inline6__ssa_v0], pl.INT32],
        cache_base_inline9__ssa_v0: pl.Scalar[pl.INDEX],
        score_transfer__ssa_v0: pl.InOut[pl.Tensor[[1152, 512], pl.FP32]],
        probability_transfer__ssa_v0: pl.InOut[pl.Tensor[[1152, 512], pl.BF16]],
        pv_transfer__ssa_v0: pl.InOut[pl.Tensor[[1152, 128], pl.FP32]],
        out2d_inline10__ssa_v0: pl.Out[pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16]],
    ) -> tuple[pl.Tensor[[1152, 512], pl.FP32], pl.Tensor[[1152, 512], pl.BF16], pl.Tensor[[1152, 128], pl.FP32], pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16]]:
        pl.func_attr({"mx_tensor_views_blocked": True, "split_aiv": True, "dual_aiv_dispatch": True})
        core_inline12__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        pl.system.set_ffts(ffts_workspace__ssa_v0)
        aiv_id_inline13__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
        rope_core_inline14__ssa_v0: pl.Scalar[pl.INDEX] = core_inline12__ssa_v0 * 2 + aiv_id_inline13__ssa_v0
        if rope_core_inline14__ssa_v0 < 32:
            q_red_pad_inline15__tile: pl.Tile[[1, 1408], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 1408], dtype=pl.FP32, value=pl.const(0.0, pl.FP32))
            k_red_pad_inline16__tile: pl.Tile[[1, 896], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 896], dtype=pl.FP32, value=pl.const(0.0, pl.FP32))
            for it_inline17__idx_v0, (key_cache_bsnd_inline4__iter_v1, q2d_inline8__iter_v1, value_cache_bsnd_inline5__iter_v1) in pl.pipeline(
                0, 4, 2, stage=1, init_values=(key_cache_bsnd_inline4__ssa_v0, q2d_inline8__ssa_v0, value_cache_bsnd_inline5__ssa_v0)
            ):
                g_idx_inline18__ssa_v0: pl.Scalar[pl.INDEX] = rope_core_inline14__ssa_v0 + it_inline17__idx_v0 * 32
                if g_idx_inline18__ssa_v0 < active_batch_inline0__ssa_v0 * 8:
                    kv_head_inline19__ssa_v0: pl.Scalar[pl.INDEX] = g_idx_inline18__ssa_v0 // active_batch_inline0__ssa_v0
                    batch_idx_inline20__ssa_v0: pl.Scalar[pl.INDEX] = g_idx_inline18__ssa_v0 - kv_head_inline19__ssa_v0 * active_batch_inline0__ssa_v0
                    t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(seq_lens__ssa_v0, [batch_idx_inline20__ssa_v0])
                    position_inline21__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile, pl.INDEX) - 1
                    inv_rms_inline22__tile: pl.Scalar[pl.FP32] = pl.tensor.read(inv_rms_states__ssa_v0, [batch_idx_inline20__ssa_v0, 0])
                    t__tile_1: pl.Scalar[pl.INT32] = pl.tensor.read(slot_mapping__ssa_v0, [batch_idx_inline20__ssa_v0])
                    write_slot_inline23__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(t__tile_1, pl.INDEX), 0)
                    cos_lo_inline24__tile: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                        rope_cos__ssa_v0, [position_inline21__ssa_v0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec, attrs={"pipeline_membership": "0:0"}
                    )
                    cos_hi_inline25__tile: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                        rope_cos__ssa_v0, [position_inline21__ssa_v0, 64], [1, 64], [1, 64], target_memory=pl.Mem.Vec, attrs={"pipeline_membership": "0:0"}
                    )
                    sin_lo_inline26__tile: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                        rope_sin__ssa_v0, [position_inline21__ssa_v0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec, attrs={"pipeline_membership": "0:0"}
                    )
                    sin_hi_inline27__tile: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                        rope_sin__ssa_v0, [position_inline21__ssa_v0, 64], [1, 64], [1, 64], target_memory=pl.Mem.Vec, attrs={"pipeline_membership": "0:0"}
                    )
                    cache_col_inline28__ssa_v0: pl.Scalar[pl.INDEX] = kv_head_inline19__ssa_v0 * 128
                    t__tile_2: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                        k_proj__ssa_v0, [batch_idx_inline20__ssa_v0, cache_col_inline28__ssa_v0], [1, 128], [1, 128], target_memory=pl.Mem.Vec, attrs={"pipeline_membership": "0:0"}
                    )
                    t__tile_3: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.concat(t__tile_2, k_red_pad_inline16__tile, attrs={"pipeline_membership": "0:0"})
                    t__tile_4: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__tile_3, [8, 128], attrs={"pipeline_membership": "0:0"})
                    k_raw_inline29__tile: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.muls(t__tile_4, inv_rms_inline22__tile, attrs={"pipeline_membership": "0:0"})
                    t__tile_5: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.mul(k_raw_inline29__tile, k_raw_inline29__tile, attrs={"pipeline_membership": "0:0"})
                    tmp_tile: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec, attrs={"pipeline_membership": "0:0"})
                    k_ss_inline30__tile: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(t__tile_5, tmp_tile, attrs={"pipeline_membership": "0:0"})
                    t__rm_a0_tmp_v0: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(k_ss_inline30__tile, [1, 8], attrs={"pipeline_membership": "0:0"})
                    t__row_major_tmp_v1: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(t__rm_a0_tmp_v0, 0.0078125, attrs={"pipeline_membership": "0:0"})
                    t__tile_6: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v1, [8, 1], attrs={"pipeline_membership": "0:0"})
                    t__rm_a0_tmp_v2: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__tile_6, [1, 8], attrs={"pipeline_membership": "0:0"})
                    t__row_major_tmp_v3: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(t__rm_a0_tmp_v2, 9.9999999999999995e-07, attrs={"pipeline_membership": "0:0"})
                    t__tile_7: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v3, [8, 1], attrs={"pipeline_membership": "0:0"})
                    t__rm_a0_tmp_v4: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__tile_7, [1, 8], attrs={"pipeline_membership": "0:0"})
                    t__row_major_tmp_v5: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.sqrt(t__rm_a0_tmp_v4, attrs={"pipeline_membership": "0:0"})
                    t__tile_8: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v5, [8, 1], attrs={"pipeline_membership": "0:0"})
                    k_inv_inline31__rm_a0_tmp_v6: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__tile_8, [1, 8], attrs={"pipeline_membership": "0:0"})
                    k_inv_inline31__row_major_tmp_v7: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.recip(k_inv_inline31__rm_a0_tmp_v6, attrs={"pipeline_membership": "0:0"})
                    k_inv_inline31__tile: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(k_inv_inline31__row_major_tmp_v7, [8, 1], attrs={"pipeline_membership": "0:0"})
                    k_norm_w__tile: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.load(k_norm_w__ssa_v0, [0, 0], [1, 128], [1, 128], target_memory=pl.Mem.Vec, attrs={"pipeline_membership": "0:0"})
                    t__tile_9: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(k_raw_inline29__tile, k_norm_w__tile, attrs={"pipeline_membership": "0:0"})
                    k_normed_inline32__tile: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(t__tile_9, k_inv_inline31__tile, attrs={"pipeline_membership": "0:0"})
                    k_full_inline33__tile: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.slice(k_normed_inline32__tile, [1, 128], [0, 0], attrs={"pipeline_membership": "0:0"})
                    k_lo_inline34__tile: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(k_full_inline33__tile, [1, 64], [0, 0], attrs={"pipeline_membership": "0:0"})
                    k_hi_inline35__tile: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(k_full_inline33__tile, [1, 64], [0, 64], attrs={"pipeline_membership": "0:0"})
                    t__tile_10: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(k_lo_inline34__tile, cos_lo_inline24__tile, attrs={"pipeline_membership": "0:0"})
                    t__tile_11: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(k_hi_inline35__tile, sin_lo_inline26__tile, attrs={"pipeline_membership": "0:0"})
                    t__tile_12: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(t__tile_10, t__tile_11, attrs={"pipeline_membership": "0:0"})
                    t__tile_13: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(k_hi_inline35__tile, cos_hi_inline25__tile, attrs={"pipeline_membership": "0:0"})
                    t__tile_14: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(k_lo_inline34__tile, sin_hi_inline27__tile, attrs={"pipeline_membership": "0:0"})
                    t__tile_15: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(t__tile_13, t__tile_14, attrs={"pipeline_membership": "0:0"})
                    k_rotated_inline36__tile: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.concat(t__tile_12, t__tile_15, attrs={"pipeline_membership": "0:0"})
                    cache_row_inline37__ssa_v0: pl.Scalar[pl.INDEX] = layer_cache_base_token_rows__ssa_v0 + write_slot_inline23__ssa_v0
                    t__tile_16: pl.Tile[[1, 128], pl.BF16, pl.Mem.Vec] = pl.tile.cast(k_rotated_inline36__tile, target_type=pl.BF16, mode="round", attrs={"pipeline_membership": "0:0"})
                    key_cache_bsnd_inline4__tile: pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16] = pl.tile.store(
                        t__tile_16, [cache_row_inline37__ssa_v0, cache_col_inline28__ssa_v0], key_cache_bsnd_inline4__iter_v1
                    )
                    t__tile_17: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                        v_proj__ssa_v0, [batch_idx_inline20__ssa_v0, cache_col_inline28__ssa_v0], [1, 128], [1, 128], target_memory=pl.Mem.Vec, attrs={"pipeline_membership": "0:0"}
                    )
                    t__tile_18: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.muls(t__tile_17, inv_rms_inline22__tile, attrs={"pipeline_membership": "0:0"})
                    v_row_bf16_inline38__tile: pl.Tile[[1, 128], pl.BF16, pl.Mem.Vec] = pl.tile.cast(t__tile_18, target_type=pl.BF16, mode="round", attrs={"pipeline_membership": "0:0"})
                    value_cache_bsnd_inline5__tile: pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16] = pl.tile.store(
                        v_row_bf16_inline38__tile, [cache_row_inline37__ssa_v0, cache_col_inline28__ssa_v0], value_cache_bsnd_inline5__iter_v1
                    )
                    q_head_base_inline39__ssa_v0: pl.Scalar[pl.INDEX] = kv_head_inline19__ssa_v0 * 5
                    t__tile_19: pl.Tile[[1, 640], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                        q_proj__ssa_v0, [batch_idx_inline20__ssa_v0, q_head_base_inline39__ssa_v0 * 128], [1, 640], [1, 640], target_memory=pl.Mem.Vec, attrs={"pipeline_membership": "0:0"}
                    )
                    t__tile_20: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.concat(t__tile_19, q_red_pad_inline15__tile, attrs={"pipeline_membership": "0:0"})
                    t__tile_21: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__tile_20, [16, 128], attrs={"pipeline_membership": "0:0"})
                    q_raw_inline40__tile: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.muls(t__tile_21, inv_rms_inline22__tile, attrs={"pipeline_membership": "0:0"})
                    t__tile_22: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.mul(q_raw_inline40__tile, q_raw_inline40__tile, attrs={"pipeline_membership": "0:0"})
                    tmp_tile_1: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec, attrs={"pipeline_membership": "0:0"})
                    q_ss_inline41__tile: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(t__tile_22, tmp_tile_1, attrs={"pipeline_membership": "0:0"})
                    t__rm_a0_tmp_v8: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(q_ss_inline41__tile, [1, 16], attrs={"pipeline_membership": "0:0"})
                    t__row_major_tmp_v9: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.muls(t__rm_a0_tmp_v8, 0.0078125, attrs={"pipeline_membership": "0:0"})
                    t__tile_23: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v9, [16, 1], attrs={"pipeline_membership": "0:0"})
                    t__rm_a0_tmp_v10: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__tile_23, [1, 16], attrs={"pipeline_membership": "0:0"})
                    t__row_major_tmp_v11: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.adds(t__rm_a0_tmp_v10, 9.9999999999999995e-07, attrs={"pipeline_membership": "0:0"})
                    t__tile_24: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v11, [16, 1], attrs={"pipeline_membership": "0:0"})
                    t__rm_a0_tmp_v12: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__tile_24, [1, 16], attrs={"pipeline_membership": "0:0"})
                    t__row_major_tmp_v13: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.sqrt(t__rm_a0_tmp_v12, attrs={"pipeline_membership": "0:0"})
                    t__tile_25: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v13, [16, 1], attrs={"pipeline_membership": "0:0"})
                    q_inv_inline42__rm_a0_tmp_v14: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__tile_25, [1, 16], attrs={"pipeline_membership": "0:0"})
                    q_inv_inline42__row_major_tmp_v15: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.recip(q_inv_inline42__rm_a0_tmp_v14, attrs={"pipeline_membership": "0:0"})
                    q_inv_inline42__tile: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(q_inv_inline42__row_major_tmp_v15, [16, 1], attrs={"pipeline_membership": "0:0"})
                    q_norm_w__tile: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.load(q_norm_w__ssa_v0, [0, 0], [1, 128], [1, 128], target_memory=pl.Mem.Vec, attrs={"pipeline_membership": "0:0"})
                    t__tile_26: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(q_raw_inline40__tile, q_norm_w__tile, attrs={"pipeline_membership": "0:0"})
                    q_heads_inline43__tile: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(t__tile_26, q_inv_inline42__tile, attrs={"pipeline_membership": "0:0"})
                    q_lo_inline44__tile_textract: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.extract(
                        q_heads_inline43__tile, 0, 0, [16, 64], target_memory=pl.Mem.Vec, attrs={"pipeline_membership": "0:0"}
                    )
                    q_hi_inline45__tile_textract: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.extract(
                        q_heads_inline43__tile, 0, 64, [16, 64], target_memory=pl.Mem.Vec, attrs={"pipeline_membership": "0:0"}
                    )
                    t__tile_27: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(q_lo_inline44__tile_textract, cos_lo_inline24__tile, attrs={"pipeline_membership": "0:0"})
                    t__tile_28: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(q_hi_inline45__tile_textract, sin_lo_inline26__tile, attrs={"pipeline_membership": "0:0"})
                    t__tile_29: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(t__tile_27, t__tile_28, attrs={"pipeline_membership": "0:0"})
                    t__tile_30: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(q_hi_inline45__tile_textract, cos_hi_inline25__tile, attrs={"pipeline_membership": "0:0"})
                    t__tile_31: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(q_lo_inline44__tile_textract, sin_hi_inline27__tile, attrs={"pipeline_membership": "0:0"})
                    t__tile_32: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(t__tile_30, t__tile_31, attrs={"pipeline_membership": "0:0"})
                    q_rotated_inline46__tile: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.concat(t__tile_29, t__tile_32, attrs={"pipeline_membership": "0:0"})
                    q_row_inline47__ssa_v0: pl.Scalar[pl.INDEX] = batch_idx_inline20__ssa_v0 * 40 + q_head_base_inline39__ssa_v0
                    t__tile_33: pl.Tile[[5, 128], pl.FP32, pl.Mem.Vec] = pl.tile.slice(q_rotated_inline46__tile, [5, 128], [0, 0], attrs={"pipeline_membership": "0:0"})
                    t__tile_34: pl.Tile[[5, 128], pl.BF16, pl.Mem.Vec] = pl.tile.cast(t__tile_33, target_type=pl.BF16, mode="round", attrs={"pipeline_membership": "0:0"})
                    q2d_inline8__tile: pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16] = pl.tile.store(t__tile_34, [q_row_inline47__ssa_v0, 0], q2d_inline8__iter_v1)
                    key_cache_bsnd_inline4__phi_v4, q2d_inline8__phi_v4, value_cache_bsnd_inline5__phi_v4 = pl._dump.yield_(
                        key_cache_bsnd_inline4__tile,
                        q2d_inline8__tile,
                        value_cache_bsnd_inline5__tile,
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
                g_idx_inline18__ssa_v0_1: pl.Scalar[pl.INDEX] = rope_core_inline14__ssa_v0 + (it_inline17__idx_v0 + 1) * 32
                if g_idx_inline18__ssa_v0_1 < active_batch_inline0__ssa_v0 * 8:
                    kv_head_inline19__ssa_v0_1: pl.Scalar[pl.INDEX] = g_idx_inline18__ssa_v0_1 // active_batch_inline0__ssa_v0
                    batch_idx_inline20__ssa_v0_1: pl.Scalar[pl.INDEX] = g_idx_inline18__ssa_v0_1 - kv_head_inline19__ssa_v0_1 * active_batch_inline0__ssa_v0
                    t__tile_35: pl.Scalar[pl.INT32] = pl.tensor.read(seq_lens__ssa_v0, [batch_idx_inline20__ssa_v0_1])
                    position_inline21__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.cast(t__tile_35, pl.INDEX) - 1
                    inv_rms_inline22__tile_1: pl.Scalar[pl.FP32] = pl.tensor.read(inv_rms_states__ssa_v0, [batch_idx_inline20__ssa_v0_1, 0])
                    t__tile_36: pl.Scalar[pl.INT32] = pl.tensor.read(slot_mapping__ssa_v0, [batch_idx_inline20__ssa_v0_1])
                    write_slot_inline23__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.max(pl.cast(t__tile_36, pl.INDEX), 0)
                    cos_lo_inline24__tile_1: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                        rope_cos__ssa_v0, [position_inline21__ssa_v0_1, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec, attrs={"pipeline_membership": "0:1"}
                    )
                    cos_hi_inline25__tile_1: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                        rope_cos__ssa_v0, [position_inline21__ssa_v0_1, 64], [1, 64], [1, 64], target_memory=pl.Mem.Vec, attrs={"pipeline_membership": "0:1"}
                    )
                    sin_lo_inline26__tile_1: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                        rope_sin__ssa_v0, [position_inline21__ssa_v0_1, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec, attrs={"pipeline_membership": "0:1"}
                    )
                    sin_hi_inline27__tile_1: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                        rope_sin__ssa_v0, [position_inline21__ssa_v0_1, 64], [1, 64], [1, 64], target_memory=pl.Mem.Vec, attrs={"pipeline_membership": "0:1"}
                    )
                    cache_col_inline28__ssa_v0_1: pl.Scalar[pl.INDEX] = kv_head_inline19__ssa_v0_1 * 128
                    t__tile_37: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                        k_proj__ssa_v0, [batch_idx_inline20__ssa_v0_1, cache_col_inline28__ssa_v0_1], [1, 128], [1, 128], target_memory=pl.Mem.Vec, attrs={"pipeline_membership": "0:1"}
                    )
                    t__tile_38: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.concat(t__tile_37, k_red_pad_inline16__tile, attrs={"pipeline_membership": "0:1"})
                    t__tile_39: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__tile_38, [8, 128], attrs={"pipeline_membership": "0:1"})
                    k_raw_inline29__tile_1: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.muls(t__tile_39, inv_rms_inline22__tile_1, attrs={"pipeline_membership": "0:1"})
                    t__tile_40: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.mul(k_raw_inline29__tile_1, k_raw_inline29__tile_1, attrs={"pipeline_membership": "0:1"})
                    tmp_tile_2: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec, attrs={"pipeline_membership": "0:1"})
                    k_ss_inline30__tile_1: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(t__tile_40, tmp_tile_2, attrs={"pipeline_membership": "0:1"})
                    t__rm_a0_tmp_v0_1: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(k_ss_inline30__tile_1, [1, 8], attrs={"pipeline_membership": "0:1"})
                    t__row_major_tmp_v1_1: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(t__rm_a0_tmp_v0_1, 0.0078125, attrs={"pipeline_membership": "0:1"})
                    t__tile_41: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v1_1, [8, 1], attrs={"pipeline_membership": "0:1"})
                    t__rm_a0_tmp_v2_1: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__tile_41, [1, 8], attrs={"pipeline_membership": "0:1"})
                    t__row_major_tmp_v3_1: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.adds(t__rm_a0_tmp_v2_1, 9.9999999999999995e-07, attrs={"pipeline_membership": "0:1"})
                    t__tile_42: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v3_1, [8, 1], attrs={"pipeline_membership": "0:1"})
                    t__rm_a0_tmp_v4_1: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__tile_42, [1, 8], attrs={"pipeline_membership": "0:1"})
                    t__row_major_tmp_v5_1: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.sqrt(t__rm_a0_tmp_v4_1, attrs={"pipeline_membership": "0:1"})
                    t__tile_43: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v5_1, [8, 1], attrs={"pipeline_membership": "0:1"})
                    k_inv_inline31__rm_a0_tmp_v6_1: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__tile_43, [1, 8], attrs={"pipeline_membership": "0:1"})
                    k_inv_inline31__row_major_tmp_v7_1: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.recip(k_inv_inline31__rm_a0_tmp_v6_1, attrs={"pipeline_membership": "0:1"})
                    k_inv_inline31__tile_1: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(k_inv_inline31__row_major_tmp_v7_1, [8, 1], attrs={"pipeline_membership": "0:1"})
                    k_norm_w__tile_1: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                        k_norm_w__ssa_v0, [0, 0], [1, 128], [1, 128], target_memory=pl.Mem.Vec, attrs={"pipeline_membership": "0:1"}
                    )
                    t__tile_44: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(k_raw_inline29__tile_1, k_norm_w__tile_1, attrs={"pipeline_membership": "0:1"})
                    k_normed_inline32__tile_1: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(t__tile_44, k_inv_inline31__tile_1, attrs={"pipeline_membership": "0:1"})
                    k_full_inline33__tile_1: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.slice(k_normed_inline32__tile_1, [1, 128], [0, 0], attrs={"pipeline_membership": "0:1"})
                    k_lo_inline34__tile_1: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(k_full_inline33__tile_1, [1, 64], [0, 0], attrs={"pipeline_membership": "0:1"})
                    k_hi_inline35__tile_1: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(k_full_inline33__tile_1, [1, 64], [0, 64], attrs={"pipeline_membership": "0:1"})
                    t__tile_45: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(k_lo_inline34__tile_1, cos_lo_inline24__tile_1, attrs={"pipeline_membership": "0:1"})
                    t__tile_46: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(k_hi_inline35__tile_1, sin_lo_inline26__tile_1, attrs={"pipeline_membership": "0:1"})
                    t__tile_47: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(t__tile_45, t__tile_46, attrs={"pipeline_membership": "0:1"})
                    t__tile_48: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(k_hi_inline35__tile_1, cos_hi_inline25__tile_1, attrs={"pipeline_membership": "0:1"})
                    t__tile_49: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(k_lo_inline34__tile_1, sin_hi_inline27__tile_1, attrs={"pipeline_membership": "0:1"})
                    t__tile_50: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(t__tile_48, t__tile_49, attrs={"pipeline_membership": "0:1"})
                    k_rotated_inline36__tile_1: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.concat(t__tile_47, t__tile_50, attrs={"pipeline_membership": "0:1"})
                    cache_row_inline37__ssa_v0_1: pl.Scalar[pl.INDEX] = layer_cache_base_token_rows__ssa_v0 + write_slot_inline23__ssa_v0_1
                    t__tile_51: pl.Tile[[1, 128], pl.BF16, pl.Mem.Vec] = pl.tile.cast(k_rotated_inline36__tile_1, target_type=pl.BF16, mode="round", attrs={"pipeline_membership": "0:1"})
                    key_cache_bsnd_inline4__tile_1: pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16] = pl.tile.store(
                        t__tile_51, [cache_row_inline37__ssa_v0_1, cache_col_inline28__ssa_v0_1], key_cache_bsnd_inline4__phi_v4
                    )
                    t__tile_52: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                        v_proj__ssa_v0, [batch_idx_inline20__ssa_v0_1, cache_col_inline28__ssa_v0_1], [1, 128], [1, 128], target_memory=pl.Mem.Vec, attrs={"pipeline_membership": "0:1"}
                    )
                    t__tile_53: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.muls(t__tile_52, inv_rms_inline22__tile_1, attrs={"pipeline_membership": "0:1"})
                    v_row_bf16_inline38__tile_1: pl.Tile[[1, 128], pl.BF16, pl.Mem.Vec] = pl.tile.cast(t__tile_53, target_type=pl.BF16, mode="round", attrs={"pipeline_membership": "0:1"})
                    value_cache_bsnd_inline5__tile_1: pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16] = pl.tile.store(
                        v_row_bf16_inline38__tile_1, [cache_row_inline37__ssa_v0_1, cache_col_inline28__ssa_v0_1], value_cache_bsnd_inline5__phi_v4
                    )
                    q_head_base_inline39__ssa_v0_1: pl.Scalar[pl.INDEX] = kv_head_inline19__ssa_v0_1 * 5
                    t__tile_54: pl.Tile[[1, 640], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                        q_proj__ssa_v0, [batch_idx_inline20__ssa_v0_1, q_head_base_inline39__ssa_v0_1 * 128], [1, 640], [1, 640], target_memory=pl.Mem.Vec, attrs={"pipeline_membership": "0:1"}
                    )
                    t__tile_55: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.concat(t__tile_54, q_red_pad_inline15__tile, attrs={"pipeline_membership": "0:1"})
                    t__tile_56: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__tile_55, [16, 128], attrs={"pipeline_membership": "0:1"})
                    q_raw_inline40__tile_1: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.muls(t__tile_56, inv_rms_inline22__tile_1, attrs={"pipeline_membership": "0:1"})
                    t__tile_57: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.mul(q_raw_inline40__tile_1, q_raw_inline40__tile_1, attrs={"pipeline_membership": "0:1"})
                    tmp_tile_3: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec, attrs={"pipeline_membership": "0:1"})
                    q_ss_inline41__tile_1: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(t__tile_57, tmp_tile_3, attrs={"pipeline_membership": "0:1"})
                    t__rm_a0_tmp_v8_1: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(q_ss_inline41__tile_1, [1, 16], attrs={"pipeline_membership": "0:1"})
                    t__row_major_tmp_v9_1: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.muls(t__rm_a0_tmp_v8_1, 0.0078125, attrs={"pipeline_membership": "0:1"})
                    t__tile_58: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v9_1, [16, 1], attrs={"pipeline_membership": "0:1"})
                    t__rm_a0_tmp_v10_1: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__tile_58, [1, 16], attrs={"pipeline_membership": "0:1"})
                    t__row_major_tmp_v11_1: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.adds(t__rm_a0_tmp_v10_1, 9.9999999999999995e-07, attrs={"pipeline_membership": "0:1"})
                    t__tile_59: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v11_1, [16, 1], attrs={"pipeline_membership": "0:1"})
                    t__rm_a0_tmp_v12_1: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__tile_59, [1, 16], attrs={"pipeline_membership": "0:1"})
                    t__row_major_tmp_v13_1: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.sqrt(t__rm_a0_tmp_v12_1, attrs={"pipeline_membership": "0:1"})
                    t__tile_60: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v13_1, [16, 1], attrs={"pipeline_membership": "0:1"})
                    q_inv_inline42__rm_a0_tmp_v14_1: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__tile_60, [1, 16], attrs={"pipeline_membership": "0:1"})
                    q_inv_inline42__row_major_tmp_v15_1: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.recip(q_inv_inline42__rm_a0_tmp_v14_1, attrs={"pipeline_membership": "0:1"})
                    q_inv_inline42__tile_1: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(q_inv_inline42__row_major_tmp_v15_1, [16, 1], attrs={"pipeline_membership": "0:1"})
                    q_norm_w__tile_1: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                        q_norm_w__ssa_v0, [0, 0], [1, 128], [1, 128], target_memory=pl.Mem.Vec, attrs={"pipeline_membership": "0:1"}
                    )
                    t__tile_61: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(q_raw_inline40__tile_1, q_norm_w__tile_1, attrs={"pipeline_membership": "0:1"})
                    q_heads_inline43__tile_1: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(t__tile_61, q_inv_inline42__tile_1, attrs={"pipeline_membership": "0:1"})
                    q_lo_inline44__tile_textract_1: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.extract(
                        q_heads_inline43__tile_1, 0, 0, [16, 64], target_memory=pl.Mem.Vec, attrs={"pipeline_membership": "0:1"}
                    )
                    q_hi_inline45__tile_textract_1: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.extract(
                        q_heads_inline43__tile_1, 0, 64, [16, 64], target_memory=pl.Mem.Vec, attrs={"pipeline_membership": "0:1"}
                    )
                    t__tile_62: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(q_lo_inline44__tile_textract_1, cos_lo_inline24__tile_1, attrs={"pipeline_membership": "0:1"})
                    t__tile_63: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(q_hi_inline45__tile_textract_1, sin_lo_inline26__tile_1, attrs={"pipeline_membership": "0:1"})
                    t__tile_64: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(t__tile_62, t__tile_63, attrs={"pipeline_membership": "0:1"})
                    t__tile_65: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(q_hi_inline45__tile_textract_1, cos_hi_inline25__tile_1, attrs={"pipeline_membership": "0:1"})
                    t__tile_66: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(q_lo_inline44__tile_textract_1, sin_hi_inline27__tile_1, attrs={"pipeline_membership": "0:1"})
                    t__tile_67: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(t__tile_65, t__tile_66, attrs={"pipeline_membership": "0:1"})
                    q_rotated_inline46__tile_1: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.concat(t__tile_64, t__tile_67, attrs={"pipeline_membership": "0:1"})
                    q_row_inline47__ssa_v0_1: pl.Scalar[pl.INDEX] = batch_idx_inline20__ssa_v0_1 * 40 + q_head_base_inline39__ssa_v0_1
                    t__tile_68: pl.Tile[[5, 128], pl.FP32, pl.Mem.Vec] = pl.tile.slice(q_rotated_inline46__tile_1, [5, 128], [0, 0], attrs={"pipeline_membership": "0:1"})
                    t__tile_69: pl.Tile[[5, 128], pl.BF16, pl.Mem.Vec] = pl.tile.cast(t__tile_68, target_type=pl.BF16, mode="round", attrs={"pipeline_membership": "0:1"})
                    q2d_inline8__tile_1: pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16] = pl.tile.store(t__tile_69, [q_row_inline47__ssa_v0_1, 0], q2d_inline8__phi_v4)
                    key_cache_bsnd_inline4__phi_v4_1, q2d_inline8__phi_v4_1, value_cache_bsnd_inline5__phi_v4_1 = pl._dump.yield_(
                        key_cache_bsnd_inline4__tile_1,
                        q2d_inline8__tile_1,
                        value_cache_bsnd_inline5__tile_1,
                        types=[
                            pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16],
                            pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16],
                            pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16],
                        ],
                    )
                else:
                    key_cache_bsnd_inline4__phi_v4_1, q2d_inline8__phi_v4_1, value_cache_bsnd_inline5__phi_v4_1 = pl._dump.yield_(
                        key_cache_bsnd_inline4__phi_v4,
                        q2d_inline8__phi_v4,
                        value_cache_bsnd_inline5__phi_v4,
                        types=[
                            pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16],
                            pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16],
                            pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16],
                        ],
                    )
                key_cache_bsnd_inline4__rv_v2, q2d_inline8__rv_v2, value_cache_bsnd_inline5__rv_v2 = pl._dump.yield_(
                    key_cache_bsnd_inline4__phi_v4_1,
                    q2d_inline8__phi_v4_1,
                    value_cache_bsnd_inline5__phi_v4_1,
                    types=[
                        pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16],
                        pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16],
                        pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16],
                    ],
                )
        else:
            pass
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.syncall(core_type=pl.KernelType.MIX)
        pl.system.cacheinvalid()
        for task_inline48__idx_v0 in pl.range(core_inline12__ssa_v0, num_tasks_inline1__ssa_v0, 24):
            batch_inline49__ssa_v0: pl.Scalar[pl.INDEX] = task_inline48__idx_v0 // 8
            kv_head_inline19__ssa_v1: pl.Scalar[pl.INDEX] = task_inline48__idx_v0 % 8
            seq_len_inline50__tile: pl.Scalar[pl.INT32] = pl.tensor.read(seq_lens__ssa_v0, [batch_inline49__ssa_v0])
            stack_count_inline52__ssa_v0: pl.Scalar[pl.INDEX] = (pl.cast(seq_len_inline50__tile, pl.INDEX) + 511) // 512
            transfer_base_inline54__ssa_v0: pl.Scalar[pl.INDEX] = core_inline12__ssa_v0 * 48
            aiv_id_inline118__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
            pl.system.set_ffts(ffts_workspace__ssa_v0)
            lane_row_inline119__ssa_v0: pl.Scalar[pl.INDEX] = aiv_id_inline118__ssa_v0 * 2
            lane_rows_inline120__ssa_v0: pl.Scalar[pl.INDEX] = aiv_id_inline118__ssa_v0 + 2
            tmp_inline121__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            zero_scores_inline122__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.full([8, 512], dtype=pl.FP32, value=pl.const(0.0, pl.FP32))
            zero_scores_valid_inline123__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 512])] = pl.tile.set_validshape(
                zero_scores_inline122__ssa_v0, lane_rows_inline120__ssa_v0, 512
            )
            t__tmp_v76: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = pl.tile.row_max(zero_scores_valid_inline123__ssa_v0, tmp_inline121__ssa_v0)
            zrow_inline124__rm_a0_tmp_v16: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = pl.tile.reshape(t__tmp_v76, [1, 8])
            zrow_inline124__row_major_tmp_v17: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = pl.tile.muls(zrow_inline124__rm_a0_tmp_v16, 0.0)
            zrow_inline124__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = pl.tile.reshape(zrow_inline124__row_major_tmp_v17, [8, 1])
            m_inline125__rm_a0_tmp_v18: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = pl.tile.reshape(zrow_inline124__ssa_v0, [1, 8])
            m_inline125__row_major_tmp_v19: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = pl.tile.adds(
                m_inline125__rm_a0_tmp_v18, -3.0000000000000001e38
            )
            m_inline125__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = pl.tile.reshape(m_inline125__row_major_tmp_v19, [8, 1])
            l_sum_inline126__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = zrow_inline124__ssa_v0
            o_init_inline127__ssa_v0: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.full([8, 128], dtype=pl.FP32, value=pl.const(0.0, pl.FP32))
            o_inline128__ssa_v0: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 128])] = pl.tile.set_validshape(
                o_init_inline127__ssa_v0, lane_rows_inline120__ssa_v0, 128
            )
            rescale_pending0_inline129__rm_a0_tmp_v20: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = pl.tile.reshape(zrow_inline124__ssa_v0, [1, 8])
            rescale_pending0_inline129__row_major_tmp_v21: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = pl.tile.adds(
                rescale_pending0_inline129__rm_a0_tmp_v20, 1.0
            )
            rescale_pending0_inline129__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = pl.tile.reshape(
                rescale_pending0_inline129__row_major_tmp_v21, [8, 1]
            )
            rescale_pending1_inline130__rm_a0_tmp_v22: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = pl.tile.reshape(zrow_inline124__ssa_v0, [1, 8])
            rescale_pending1_inline130__row_major_tmp_v23: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = pl.tile.adds(
                rescale_pending1_inline130__rm_a0_tmp_v22, 1.0
            )
            rescale_pending1_inline130__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = pl.tile.reshape(
                rescale_pending1_inline130__row_major_tmp_v23, [8, 1]
            )
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
                    valid_cols_inline142__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(pl.cast(seq_len_inline50__tile, pl.INDEX) - produce_stack_inline58__ssa_v1 * 512, 512)
                    score_valid_inline143__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, valid_cols_inline142__ssa_v0])] = (
                        pl.tile.set_validshape(score_scaled_inline141__ssa_v0, lane_rows_inline120__ssa_v0, valid_cols_inline142__ssa_v0)
                    )
                    score_filled_inline144__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(score_valid_inline143__ssa_v0, pad_value=pl.PadValue.min)
                    score_masked_inline145__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 512], pad=pl.PadValue.min)] = pl.tile.set_validshape(
                        score_filled_inline144__ssa_v0, lane_rows_inline120__ssa_v0, 512
                    )
                    local_m_inline146__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = pl.tile.row_max(
                        score_masked_inline145__ssa_v0, tmp_inline121__ssa_v0
                    )
                    next_m_inline147__rm_a0_tmp_v24: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = pl.tile.reshape(
                        local_m_inline146__ssa_v0, [1, 8]
                    )
                    next_m_inline147__rm_a1_tmp_v25: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = pl.tile.reshape(m_iter, [1, 8])
                    next_m_inline147__row_major_tmp_v26: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = pl.tile.maximum(
                        next_m_inline147__rm_a0_tmp_v24, next_m_inline147__rm_a1_tmp_v25
                    )
                    next_m_inline147__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = pl.tile.reshape(
                        next_m_inline147__row_major_tmp_v26, [8, 1]
                    )
                    t__rm_a0_tmp_v27: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = pl.tile.reshape(m_iter, [1, 8])
                    t__rm_a1_tmp_v28: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = pl.tile.reshape(next_m_inline147__ssa_v0, [1, 8])
                    t__row_major_tmp_v29: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = pl.tile.sub(t__rm_a0_tmp_v27, t__rm_a1_tmp_v28)
                    t__tmp_v77: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = pl.tile.reshape(t__row_major_tmp_v29, [8, 1])
                    rescale_inline148__rm_a0_tmp_v30: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = pl.tile.reshape(t__tmp_v77, [1, 8])
                    rescale_inline148__row_major_tmp_v31: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = pl.tile.exp(
                        rescale_inline148__rm_a0_tmp_v30
                    )
                    rescale_inline148__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = pl.tile.reshape(
                        rescale_inline148__row_major_tmp_v31, [8, 1]
                    )
                    t__tmp_v78: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 512], pad=pl.PadValue.min)] = pl.tile.row_expand_sub(
                        score_masked_inline145__ssa_v0, next_m_inline147__ssa_v0
                    )
                    probability_aiv_inline149__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 512], pad=pl.PadValue.min)] = pl.tile.exp(
                        t__tmp_v78
                    )
                    probability_reduce_inline150__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 512], pad=pl.PadValue.min)] = pl.tile.move(
                        probability_aiv_inline149__ssa_v0, target_memory=pl.Mem.Vec, slayout=pl.TileLayout.none_box
                    )
                    t__rm_a0_tmp_v32: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = pl.tile.reshape(rescale_inline148__ssa_v0, [1, 8])
                    t__rm_a1_tmp_v33: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = pl.tile.reshape(l_iter, [1, 8])
                    t__row_major_tmp_v34: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = pl.tile.mul(t__rm_a0_tmp_v32, t__rm_a1_tmp_v33)
                    t__tmp_v79: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = pl.tile.reshape(t__row_major_tmp_v34, [8, 1])
                    t__tmp_v80: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = pl.tile.row_sum(
                        probability_reduce_inline150__ssa_v0, tmp_inline121__ssa_v0
                    )
                    next_l_inline151__rm_a0_tmp_v35: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = pl.tile.reshape(t__tmp_v79, [1, 8])
                    next_l_inline151__rm_a1_tmp_v36: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = pl.tile.reshape(t__tmp_v80, [1, 8])
                    next_l_inline151__row_major_tmp_v37: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = pl.tile.add(
                        next_l_inline151__rm_a0_tmp_v35, next_l_inline151__rm_a1_tmp_v36
                    )
                    next_l_inline151__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = pl.tile.reshape(
                        next_l_inline151__row_major_tmp_v37, [8, 1]
                    )
                    probability_bf16_inline152__ssa_v0: pl.Tile[[8, 512], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 512], pad=pl.PadValue.min)] = pl.tile.cast(
                        probability_aiv_inline149__ssa_v0, target_type=pl.BF16, mode="rint"
                    )
                    probability_valid_inline153__ssa_v0: pl.Tile[[8, 512], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 512], pad=pl.PadValue.min)] = (
                        pl.tile.set_validshape(probability_bf16_inline152__ssa_v0, lane_rows_inline120__ssa_v0, 512)
                    )
                    probability_transfer__store: pl.Tensor[[1152, 512], pl.BF16] = pl.tile.store(
                        probability_valid_inline153__ssa_v0, [produce_row_inline60__ssa_v1 + lane_row_inline119__ssa_v0, 0], probability_transfer__ssa_v0
                    )
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
                    t__tmp_v81: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 128])] = pl.tile.row_expand_mul(o_iter, pending0_iter)
                    t__tmp_v82: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 128])] = pl.tile.add(t__tmp_v81, pv_aiv_inline155__ssa_v0)
                    o_after_inline154__rv_v0: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 128])] = pl.yield_(t__tmp_v82)
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
            t__rm_a0_tmp_v38: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = pl.tile.reshape(l_sum_inline133, [1, 8])
            t__row_major_tmp_v39: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = pl.tile.recip(t__rm_a0_tmp_v38)
            t__tmp_v83: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = pl.tile.reshape(t__row_major_tmp_v39, [8, 1])
            t__tmp_v84: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 128])] = pl.tile.row_expand_mul(o_inline134, t__tmp_v83)
            ctx_inline158__ssa_v0: pl.Tile[[8, 128], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 128])] = pl.tile.cast(t__tmp_v84, target_type=pl.BF16, mode="rint")
            out2d_inline10__store: pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16] = pl.tile.store(
                ctx_inline158__ssa_v0, [batch_inline49__ssa_v0 * 40 + kv_head_inline19__ssa_v1 * 5 + lane_row_inline119__ssa_v0, 0], out2d_inline10__ssa_v0
            )
        return score_transfer__ssa_v0, probability_transfer__ssa_v0, pv_transfer__ssa_v0, out2d_inline10__ssa_v0

    @pl.function(type=pl.FunctionType.Group, level=pl.Level.CORE_GROUP, role=pl.Role.SubWorker)
    def attn_swpipe(
        ffts_workspace__ssa_v0: pl.Tensor[[256], pl.INT64],
        key_cache_bsnd_inline4__ssa_v0: pl.InOut[pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16]],
        q2d_inline8__ssa_v0: pl.InOut[pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16]],
        value_cache_bsnd_inline5__ssa_v0: pl.InOut[pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16]],
        active_batch_inline0__ssa_v0: pl.Scalar[pl.INDEX],
        seq_lens__ssa_v0: pl.Tensor[[2], pl.INT32],
        inv_rms_states__ssa_v0: pl.Tensor[[2, 1], pl.FP32],
        slot_mapping__ssa_v0: pl.Tensor[[2], pl.INT32],
        rope_cos__ssa_v0: pl.Tensor[[4096, 128], pl.FP32],
        rope_sin__ssa_v0: pl.Tensor[[4096, 128], pl.FP32],
        k_proj__ssa_v0: pl.Tensor[[2, 1024], pl.FP32],
        k_norm_w__ssa_v0: pl.Tensor[[1, 128], pl.FP32],
        layer_cache_base_token_rows__ssa_v0: pl.Scalar[pl.INDEX],
        v_proj__ssa_v0: pl.Tensor[[2, 1024], pl.FP32],
        q_proj__ssa_v0: pl.Tensor[[2, 5120], pl.FP32],
        q_norm_w__ssa_v0: pl.Tensor[[1, 128], pl.FP32],
        num_tasks_inline1__ssa_v0: pl.Scalar[pl.INDEX],
        block_table_2d_inline7__ssa_v0: pl.Tensor[[active_batch_inline0__ssa_v0, max_blocks_per_seq_inline6__ssa_v0], pl.INT32],
        cache_base_inline9__ssa_v0: pl.Scalar[pl.INDEX],
        score_transfer__ssa_v0: pl.InOut[pl.Tensor[[1152, 512], pl.FP32]],
        probability_transfer__ssa_v0: pl.InOut[pl.Tensor[[1152, 512], pl.BF16]],
        pv_transfer__ssa_v0: pl.InOut[pl.Tensor[[1152, 128], pl.FP32]],
        out2d_inline10__ssa_v0: pl.Out[pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16]],
    ) -> tuple[pl.Tensor[[1152, 512], pl.FP32], pl.Tensor[[1152, 512], pl.BF16], pl.Tensor[[1152, 128], pl.FP32], pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16]]:
        pl.func_attr({"mx_tensor_views_blocked": True, "split_aiv": True})
        self.attn_swpipe_aic(
            ffts_workspace__ssa_v0,
            key_cache_bsnd_inline4__ssa_v0,
            q2d_inline8__ssa_v0,
            value_cache_bsnd_inline5__ssa_v0,
            active_batch_inline0__ssa_v0,
            seq_lens__ssa_v0,
            inv_rms_states__ssa_v0,
            slot_mapping__ssa_v0,
            rope_cos__ssa_v0,
            rope_sin__ssa_v0,
            k_proj__ssa_v0,
            k_norm_w__ssa_v0,
            layer_cache_base_token_rows__ssa_v0,
            v_proj__ssa_v0,
            q_proj__ssa_v0,
            q_norm_w__ssa_v0,
            num_tasks_inline1__ssa_v0,
            block_table_2d_inline7__ssa_v0,
            cache_base_inline9__ssa_v0,
            score_transfer__ssa_v0,
            probability_transfer__ssa_v0,
            pv_transfer__ssa_v0,
            out2d_inline10__ssa_v0,
        )
        self.attn_swpipe_aiv(
            ffts_workspace__ssa_v0,
            key_cache_bsnd_inline4__ssa_v0,
            q2d_inline8__ssa_v0,
            value_cache_bsnd_inline5__ssa_v0,
            active_batch_inline0__ssa_v0,
            seq_lens__ssa_v0,
            inv_rms_states__ssa_v0,
            slot_mapping__ssa_v0,
            rope_cos__ssa_v0,
            rope_sin__ssa_v0,
            k_proj__ssa_v0,
            k_norm_w__ssa_v0,
            layer_cache_base_token_rows__ssa_v0,
            v_proj__ssa_v0,
            q_proj__ssa_v0,
            q_norm_w__ssa_v0,
            num_tasks_inline1__ssa_v0,
            block_table_2d_inline7__ssa_v0,
            cache_base_inline9__ssa_v0,
            score_transfer__ssa_v0,
            probability_transfer__ssa_v0,
            pv_transfer__ssa_v0,
            out2d_inline10__ssa_v0,
        )
        return score_transfer__ssa_v0, probability_transfer__ssa_v0, pv_transfer__ssa_v0, out2d_inline10__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def attn_swpipe_spmd(
        self,
        ffts_workspace__ssa_v0: pl.Tensor[[256], pl.INT64],
        key_cache_bsnd_inline4__ssa_v0: pl.InOut[pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16]],
        q2d_inline8__ssa_v0: pl.InOut[pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16]],
        value_cache_bsnd_inline5__ssa_v0: pl.InOut[pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16]],
        active_batch_inline0__ssa_v0: pl.Scalar[pl.INDEX],
        seq_lens__ssa_v0: pl.Tensor[[2], pl.INT32],
        inv_rms_states__ssa_v0: pl.Tensor[[2, 1], pl.FP32],
        slot_mapping__ssa_v0: pl.Tensor[[2], pl.INT32],
        rope_cos__ssa_v0: pl.Tensor[[4096, 128], pl.FP32],
        rope_sin__ssa_v0: pl.Tensor[[4096, 128], pl.FP32],
        k_proj__ssa_v0: pl.Tensor[[2, 1024], pl.FP32],
        k_norm_w__ssa_v0: pl.Tensor[[1, 128], pl.FP32],
        layer_cache_base_token_rows__ssa_v0: pl.Scalar[pl.INDEX],
        v_proj__ssa_v0: pl.Tensor[[2, 1024], pl.FP32],
        q_proj__ssa_v0: pl.Tensor[[2, 5120], pl.FP32],
        q_norm_w__ssa_v0: pl.Tensor[[1, 128], pl.FP32],
        num_tasks_inline1__ssa_v0: pl.Scalar[pl.INDEX],
        block_table_2d_inline7__ssa_v0: pl.Tensor[[active_batch_inline0__ssa_v0, max_blocks_per_seq_inline6__ssa_v0], pl.INT32],
        cache_base_inline9__ssa_v0: pl.Scalar[pl.INDEX],
        score_transfer__ssa_v0: pl.InOut[pl.Tensor[[1152, 512], pl.FP32]],
        probability_transfer__ssa_v0: pl.InOut[pl.Tensor[[1152, 512], pl.BF16]],
        pv_transfer__ssa_v0: pl.InOut[pl.Tensor[[1152, 128], pl.FP32]],
        out2d_inline10__ssa_v0: pl.Out[pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16]],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[pl.Tensor[[1152, 512], pl.FP32], pl.Tensor[[1152, 512], pl.BF16], pl.Tensor[[1152, 128], pl.FP32], pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16]] = (
            self.attn_swpipe(
                ffts_workspace__ssa_v0,
                key_cache_bsnd_inline4__ssa_v0,
                q2d_inline8__ssa_v0,
                value_cache_bsnd_inline5__ssa_v0,
                active_batch_inline0__ssa_v0,
                seq_lens__ssa_v0,
                inv_rms_states__ssa_v0,
                slot_mapping__ssa_v0,
                rope_cos__ssa_v0,
                rope_sin__ssa_v0,
                k_proj__ssa_v0,
                k_norm_w__ssa_v0,
                layer_cache_base_token_rows__ssa_v0,
                v_proj__ssa_v0,
                q_proj__ssa_v0,
                q_norm_w__ssa_v0,
                num_tasks_inline1__ssa_v0,
                block_table_2d_inline7__ssa_v0,
                cache_base_inline9__ssa_v0,
                score_transfer__ssa_v0,
                probability_transfer__ssa_v0,
                pv_transfer__ssa_v0,
                out2d_inline10__ssa_v0,
            )
        )
        score_transfer__ssa_v1: pl.Tensor[[1152, 512], pl.FP32] = ret__tmp_v0[0]
        probability_transfer__ssa_v1: pl.Tensor[[1152, 512], pl.BF16] = ret__tmp_v0[1]
        pv_transfer__ssa_v1: pl.Tensor[[1152, 128], pl.FP32] = ret__tmp_v0[2]
        out2d_inline10__ssa_v1: pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16] = ret__tmp_v0[3]

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
        pl.func_attr({"mx_tensor_views_blocked": True})
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
        t__tmp_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(key_cache__ssa_v0, 0)
        t__tmp_v1: pl.Scalar[pl.INDEX] = pl.tensor.dim(key_cache__ssa_v0, 1)
        cache_token_rows_inline3__ssa_v0: pl.Scalar[pl.INDEX] = t__tmp_v0 * t__tmp_v1 // 1024
        key_cache_bsnd_inline4__ssa_v0: pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16] = pl.tensor.reshape(key_cache__ssa_v0, [cache_token_rows_inline3__ssa_v0, 1024])
        value_cache_bsnd_inline5__ssa_v0: pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16] = pl.tensor.reshape(value_cache__ssa_v0, [cache_token_rows_inline3__ssa_v0, 1024])
        t__tmp_v2: pl.Scalar[pl.INDEX] = pl.tensor.dim(block_table__ssa_v0, 0)
        max_blocks_per_seq_inline6__ssa_v0: pl.Scalar[pl.INDEX] = t__tmp_v2 // active_batch_inline0__ssa_v0
        block_table_2d_inline7__ssa_v0: pl.Tensor[[active_batch_inline0__ssa_v0, max_blocks_per_seq_inline6__ssa_v0], pl.INT32] = pl.tensor.reshape(
            block_table__ssa_v0, [active_batch_inline0__ssa_v0, max_blocks_per_seq_inline6__ssa_v0]
        )
        q2d_inline8__ssa_v0: pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16] = pl.tensor.reshape(q_tnd_flat__ssa_v0, [query_rows_inline2__ssa_v0, 128])
        cache_base_inline9__ssa_v0: pl.Scalar[pl.INDEX] = layer_cache_base_token_rows__ssa_v0
        out2d_inline10__ssa_v0: pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16] = pl.tensor.reshape(out__ssa_v0, [query_rows_inline2__ssa_v0, 128])
        ret__tmp_v0: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
            self.attn_swpipe_spmd,
            ffts_workspace__ssa_v0,
            key_cache_bsnd_inline4__ssa_v0,
            q2d_inline8__ssa_v0,
            value_cache_bsnd_inline5__ssa_v0,
            active_batch_inline0__ssa_v0,
            seq_lens__ssa_v0,
            inv_rms_states__ssa_v0,
            slot_mapping__ssa_v0,
            rope_cos__ssa_v0,
            rope_sin__ssa_v0,
            k_proj__ssa_v0,
            k_norm_w__ssa_v0,
            layer_cache_base_token_rows__ssa_v0,
            v_proj__ssa_v0,
            q_proj__ssa_v0,
            q_norm_w__ssa_v0,
            num_tasks_inline1__ssa_v0,
            block_table_2d_inline7__ssa_v0,
            cache_base_inline9__ssa_v0,
            score_transfer__ssa_v0,
            probability_transfer__ssa_v0,
            pv_transfer__ssa_v0,
            out2d_inline10__ssa_v0,
            deps=[q_proj_tid__ssa_v0, k_proj_tid__ssa_v0, v_proj_tid__ssa_v0, rms_tid__ssa_v0, attn_out_seed_tid__ssa_v0, mlp_out_seed_tid__ssa_v0, scratch_ready_tid__ssa_v0],
            core_num=24,
            sync_start=True,
            allow_early_resolve=True,
        )
        attn_tid_inline11__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0[0]
        return out__ssa_v0
