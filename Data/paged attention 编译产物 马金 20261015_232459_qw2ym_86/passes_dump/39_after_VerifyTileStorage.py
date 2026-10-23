# pypto.program: _jit_paged_attention_pypto_dynamic
import pypto.language as pl

active_batch_inline0__ssa_v0 = pl.dynamic("active_batch_inline0__ssa_v0")


@pl.program
class _jit_paged_attention_pypto_dynamic:
    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def attn_swpipe_aic(
        ffts_workspace__ssa_v0: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 2048)],
        key_cache_bsnd_inline4__ssa_v0: pl.InOut[pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        q2d_inline8__ssa_v0: pl.InOut[pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]],
        value_cache_bsnd_inline5__ssa_v0: pl.InOut[pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        active_batch_inline0__ssa_v0: pl.Scalar[pl.INDEX],
        seq_lens__ssa_v0: pl.Tensor[[2], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 8)],
        inv_rms_states__ssa_v0: pl.Tensor[[2, 1], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 8)],
        slot_mapping__ssa_v0: pl.Tensor[[2], pl.INT32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 8)],
        rope_cos__ssa_v0: pl.Tensor[[4096, 128], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 2097152)],
        rope_sin__ssa_v0: pl.Tensor[[4096, 128], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 2097152)],
        k_proj__ssa_v0: pl.Tensor[[2, 1024], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 8192)],
        k_norm_w__ssa_v0: pl.Tensor[[1, 128], pl.FP32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 512)],
        layer_cache_base_token_rows__ssa_v0: pl.Scalar[pl.INDEX],
        v_proj__ssa_v0: pl.Tensor[[2, 1024], pl.FP32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 8192)],
        q_proj__ssa_v0: pl.Tensor[[2, 5120], pl.FP32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 40960)],
        q_norm_w__ssa_v0: pl.Tensor[[1, 128], pl.FP32, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 512)],
        num_tasks_inline1__ssa_v0: pl.Scalar[pl.INDEX],
        block_table_2d_inline7__ssa_v0: pl.Tensor[[active_batch_inline0__ssa_v0, max_blocks_per_seq_inline6__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)],
        cache_base_inline9__ssa_v0: pl.Scalar[pl.INDEX],
        score_transfer__ssa_v0: pl.InOut[pl.Tensor[[1152, 512], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 2359296)]],
        probability_transfer__ssa_v0: pl.InOut[pl.Tensor[[1152, 512], pl.BF16, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 1179648)]],
        pv_transfer__ssa_v0: pl.InOut[pl.Tensor[[1152, 128], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 589824)]],
        out2d_inline10__ssa_v0: pl.Out[pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)]],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True, "split_aiv": True, "buffer_ir_selected": False})
        mem_mat_19: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 4096)
        mem_mat_20: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 32768)
        mem_mat_21: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 32768)
        mem_left_22: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 4096)
        mem_right_23: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_acc_24: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 8192)
        mem_left_25: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 4096)
        mem_right_26: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        pv_v_l1: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 131072, pinned=True)
        pv_p_l1: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 8192, pinned=True)
        core_inline12__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        pl._dump.call("system.set_ffts", [ffts_workspace__ssa_v0], pl._dump.Unknown)
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.syncall(core_type=pl.KernelType.MIX)
        pl.system.cacheinvalid()
        for task_inline48__idx_v0 in pl.range(core_inline12__ssa_v0, num_tasks_inline1__ssa_v0, 24):
            batch_inline49__ssa_v0: pl.Scalar[pl.INDEX] = task_inline48__idx_v0 // 8
            kv_head_inline19__ssa_v1: pl.Scalar[pl.INDEX] = task_inline48__idx_v0 % 8
            seq_len_inline50__tile: pl.Scalar[pl.INT32] = pl._dump.call("tensor.read", [seq_lens__ssa_v0, [batch_inline49__ssa_v0]], pl.Scalar[pl.INT32])
            page_count_inline51__ssa_v0: pl.Scalar[pl.INDEX] = (pl.cast(seq_len_inline50__tile, pl.INDEX) + 127) // 128
            stack_count_inline52__ssa_v0: pl.Scalar[pl.INDEX] = (pl.cast(seq_len_inline50__tile, pl.INDEX) + 511) // 512
            col_inline53__ssa_v0: pl.Scalar[pl.INDEX] = kv_head_inline19__ssa_v1 * 128
            transfer_base_inline54__ssa_v0: pl.Scalar[pl.INDEX] = core_inline12__ssa_v0 * 48
            qp_row_inline55__ssa_v0: pl.Scalar[pl.INDEX] = batch_inline49__ssa_v0 * 40 + kv_head_inline19__ssa_v1 * 5
            q_tile_inline56__ssa_v0: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_mat_19, pl.const(0, pl.INT64), 4096), pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                "tile.load",
                [q2d_inline8__ssa_v0, [qp_row_inline55__ssa_v0, 0], [16, 128], [5, 128]],
                pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                kwargs={"target_memory": pl.Mem.Mat},
            )
            for tick_inline57__idx_v0 in pl.range(stack_count_inline52__ssa_v0 + 2):
                if tick_inline57__idx_v0 < stack_count_inline52__ssa_v0:
                    produce_stack_inline58__ssa_v0: pl.Scalar[pl.INDEX] = tick_inline57__idx_v0
                    produce_page_inline59__ssa_v0: pl.Scalar[pl.INDEX] = produce_stack_inline58__ssa_v0 * 4
                    produce_row_inline60__ssa_v0: pl.Scalar[pl.INDEX] = transfer_base_inline54__ssa_v0 + produce_stack_inline58__ssa_v0 % 3 * 16
                    if produce_page_inline59__ssa_v0 + 4 <= page_count_inline51__ssa_v0:
                        for qk_page_offset_inline61__idx_v0 in pl.range(0, 4, 2):
                            t__tile: pl.Scalar[pl.INT32] = pl._dump.call(
                                "tensor.read", [block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, produce_page_inline59__ssa_v0 + qk_page_offset_inline61__idx_v0]], pl.Scalar[pl.INT32]
                            )
                            qk_ph_inline62__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile, pl.INDEX)
                            t__tile_1: pl.Scalar[pl.INT32] = pl._dump.call(
                                "tensor.read", [block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, produce_page_inline59__ssa_v0 + (qk_page_offset_inline61__idx_v0 + 1)]], pl.Scalar[pl.INT32]
                            )
                            qk_ph_inline62__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.cast(t__tile_1, pl.INDEX)
                            qk_k_page_inline63__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_mat_20, pl.const(0, pl.INT64), 32768), pl.Mem.Mat] = pl._dump.call(
                                "tile.load",
                                [key_cache_bsnd_inline4__ssa_v0, [cache_base_inline9__ssa_v0 + qk_ph_inline62__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                                pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            qk_k_page_inline63__ssa_v0_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_mat_21, pl.const(0, pl.INT64), 32768), pl.Mem.Mat] = pl._dump.call(
                                "tile.load",
                                [key_cache_bsnd_inline4__ssa_v0, [cache_base_inline9__ssa_v0 + qk_ph_inline62__ssa_v0_1 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                                pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            t__tmp_v39: pl.Tile[
                                [128, 128], pl.BF16, pl.MemRef(mem_mat_20, pl.const(0, pl.INT64), 32768), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                            ] = pl._dump.call(
                                "tile.transpose_view",
                                [qk_k_page_inline63__ssa_v0],
                                pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)],
                            )
                            q_tile_inline56__ssa_v0_Left: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_left_22, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(valid_shape=[5, 128])] = (
                                pl._dump.call(
                                    "tile.move", [q_tile_inline56__ssa_v0], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}
                                )
                            )
                            t__tmp_v39_Right: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_23, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl._dump.call(
                                "tile.move", [t__tmp_v39], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}
                            )
                            qk_score_page_inline64__ssa_v0: pl.Tile[
                                [16, 128], pl.FP32, pl.MemRef(mem_acc_24, pl.const(0, pl.INT64), 8192), pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)
                            ] = pl._dump.call(
                                "tile.matmul",
                                [q_tile_inline56__ssa_v0_Left, t__tmp_v39_Right],
                                pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)],
                            )
                            score_transfer__store: pl.Tensor[[1152, 512], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 2359296)] = pl._dump.call(
                                "tile.store",
                                [qk_score_page_inline64__ssa_v0, [produce_row_inline60__ssa_v0, qk_page_offset_inline61__idx_v0 * 128], score_transfer__ssa_v0],
                                pl.Tensor[[1152, 512], pl.FP32],
                            )
                            t__tmp_v39_1: pl.Tile[
                                [128, 128], pl.BF16, pl.MemRef(mem_mat_21, pl.const(0, pl.INT64), 32768), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                            ] = pl._dump.call(
                                "tile.transpose_view",
                                [qk_k_page_inline63__ssa_v0_1],
                                pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)],
                            )
                            q_tile_inline56__ssa_v0_Left_1: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_left_25, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(valid_shape=[5, 128])] = (
                                pl._dump.call(
                                    "tile.move", [q_tile_inline56__ssa_v0], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}
                                )
                            )
                            t__tmp_v39_Right_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_26, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl._dump.call(
                                "tile.move", [t__tmp_v39_1], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}
                            )
                            qk_score_page_inline64__ssa_v0_1: pl.Tile[
                                [16, 128], pl.FP32, pl.MemRef(mem_acc_24, pl.const(0, pl.INT64), 8192), pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)
                            ] = pl._dump.call(
                                "tile.matmul",
                                [q_tile_inline56__ssa_v0_Left_1, t__tmp_v39_Right_1],
                                pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)],
                            )
                            score_transfer__store_1: pl.Tensor[[1152, 512], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 2359296)] = pl._dump.call(
                                "tile.store",
                                [qk_score_page_inline64__ssa_v0_1, [produce_row_inline60__ssa_v0, (qk_page_offset_inline61__idx_v0 + 1) * 128], score_transfer__ssa_v0],
                                pl.Tensor[[1152, 512], pl.FP32],
                            )
                    else:
                        t__tile_2: pl.Scalar[pl.INT32] = pl._dump.call("tensor.read", [block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, produce_page_inline59__ssa_v0]], pl.Scalar[pl.INT32])
                        qk_tail_ph0_inline65__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_2, pl.INDEX)
                        k_tail0_inline66__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_mat_20, pl.const(0, pl.INT64), 32768), pl.Mem.Mat] = pl._dump.call(
                            "tile.load",
                            [key_cache_bsnd_inline4__ssa_v0, [cache_base_inline9__ssa_v0 + qk_tail_ph0_inline65__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                            pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                            kwargs={"target_memory": pl.Mem.Mat},
                        )
                        t__tmp_v41: pl.Tile[
                            [128, 128], pl.BF16, pl.MemRef(mem_mat_20, pl.const(0, pl.INT64), 32768), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                        ] = pl._dump.call(
                            "tile.transpose_view", [k_tail0_inline66__ssa_v0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)]
                        )
                        q_tile_inline56__ssa_v0_Left_2: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_left_22, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(valid_shape=[5, 128])] = (
                            pl._dump.call("tile.move", [q_tile_inline56__ssa_v0], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left})
                        )
                        t__tmp_v41_Right: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_23, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl._dump.call(
                            "tile.move", [t__tmp_v41], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}
                        )
                        qk_tail0_inline67__ssa_v0: pl.Tile[
                            [16, 128], pl.FP32, pl.MemRef(mem_acc_24, pl.const(0, pl.INT64), 8192), pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)
                        ] = pl._dump.call(
                            "tile.matmul", [q_tile_inline56__ssa_v0_Left_2, t__tmp_v41_Right], pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)]
                        )
                        score_transfer__store_v0: pl.Tensor[[1152, 512], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 2359296)] = pl._dump.call(
                            "tile.store", [qk_tail0_inline67__ssa_v0, [produce_row_inline60__ssa_v0, 0], score_transfer__ssa_v0], pl.Tensor[[1152, 512], pl.FP32]
                        )
                        if produce_page_inline59__ssa_v0 + 1 < page_count_inline51__ssa_v0:
                            t__tile_3: pl.Scalar[pl.INT32] = pl._dump.call(
                                "tensor.read", [block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, produce_page_inline59__ssa_v0 + 1]], pl.Scalar[pl.INT32]
                            )
                            qk_tail_ph1_inline68__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_3, pl.INDEX)
                            k_tail1_inline69__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_mat_20, pl.const(0, pl.INT64), 32768), pl.Mem.Mat] = pl._dump.call(
                                "tile.load",
                                [key_cache_bsnd_inline4__ssa_v0, [cache_base_inline9__ssa_v0 + qk_tail_ph1_inline68__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                                pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            t__tmp_v43: pl.Tile[
                                [128, 128], pl.BF16, pl.MemRef(mem_mat_20, pl.const(0, pl.INT64), 32768), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                            ] = pl._dump.call(
                                "tile.transpose_view",
                                [k_tail1_inline69__ssa_v0],
                                pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)],
                            )
                            t__tmp_v43_Right: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_23, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl._dump.call(
                                "tile.move", [t__tmp_v43], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}
                            )
                            qk_tail1_inline70__ssa_v0: pl.Tile[
                                [16, 128], pl.FP32, pl.MemRef(mem_acc_24, pl.const(0, pl.INT64), 8192), pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)
                            ] = pl._dump.call(
                                "tile.matmul",
                                [q_tile_inline56__ssa_v0_Left_2, t__tmp_v43_Right],
                                pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)],
                            )
                            score_transfer__store_v1: pl.Tensor[[1152, 512], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 2359296)] = pl._dump.call(
                                "tile.store", [qk_tail1_inline70__ssa_v0, [produce_row_inline60__ssa_v0, 128], score_transfer__ssa_v0], pl.Tensor[[1152, 512], pl.FP32]
                            )
                        if produce_page_inline59__ssa_v0 + 2 < page_count_inline51__ssa_v0:
                            t__tile_4: pl.Scalar[pl.INT32] = pl._dump.call(
                                "tensor.read", [block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, produce_page_inline59__ssa_v0 + 2]], pl.Scalar[pl.INT32]
                            )
                            qk_tail_ph2_inline71__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_4, pl.INDEX)
                            k_tail2_inline72__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_mat_20, pl.const(0, pl.INT64), 32768), pl.Mem.Mat] = pl._dump.call(
                                "tile.load",
                                [key_cache_bsnd_inline4__ssa_v0, [cache_base_inline9__ssa_v0 + qk_tail_ph2_inline71__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                                pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            t__tmp_v45: pl.Tile[
                                [128, 128], pl.BF16, pl.MemRef(mem_mat_20, pl.const(0, pl.INT64), 32768), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                            ] = pl._dump.call(
                                "tile.transpose_view",
                                [k_tail2_inline72__ssa_v0],
                                pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)],
                            )
                            t__tmp_v45_Right: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_23, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl._dump.call(
                                "tile.move", [t__tmp_v45], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}
                            )
                            qk_tail2_inline73__ssa_v0: pl.Tile[
                                [16, 128], pl.FP32, pl.MemRef(mem_acc_24, pl.const(0, pl.INT64), 8192), pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)
                            ] = pl._dump.call(
                                "tile.matmul",
                                [q_tile_inline56__ssa_v0_Left_2, t__tmp_v45_Right],
                                pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)],
                            )
                            score_transfer__store_v2: pl.Tensor[[1152, 512], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 2359296)] = pl._dump.call(
                                "tile.store", [qk_tail2_inline73__ssa_v0, [produce_row_inline60__ssa_v0, 256], score_transfer__ssa_v0], pl.Tensor[[1152, 512], pl.FP32]
                            )
                    pl.system.sync_set(0, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                if 2 <= tick_inline57__idx_v0:
                    consume_stack_inline74__ssa_v0: pl.Scalar[pl.INDEX] = tick_inline57__idx_v0 - 2
                    consume_page_inline75__ssa_v0: pl.Scalar[pl.INDEX] = consume_stack_inline74__ssa_v0 * 4
                    consume_row_inline76__ssa_v0: pl.Scalar[pl.INDEX] = transfer_base_inline54__ssa_v0 + consume_stack_inline74__ssa_v0 % 3 * 16
                    if consume_page_inline75__ssa_v0 + 4 <= page_count_inline51__ssa_v0:
                        t__tile_5: pl.Scalar[pl.INT32] = pl._dump.call("tensor.read", [block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, consume_page_inline75__ssa_v0]], pl.Scalar[pl.INT32])
                        pv_ph0_inline77__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_5, pl.INDEX)
                        t__tile_6: pl.Scalar[pl.INT32] = pl._dump.call(
                            "tensor.read", [block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, consume_page_inline75__ssa_v0 + 1]], pl.Scalar[pl.INT32]
                        )
                        pv_ph1_inline78__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_6, pl.INDEX)
                        t__tile_7: pl.Scalar[pl.INT32] = pl._dump.call(
                            "tensor.read", [block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, consume_page_inline75__ssa_v0 + 2]], pl.Scalar[pl.INT32]
                        )
                        pv_ph2_inline79__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_7, pl.INDEX)
                        t__tile_8: pl.Scalar[pl.INT32] = pl._dump.call(
                            "tensor.read", [block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, consume_page_inline75__ssa_v0 + 3]], pl.Scalar[pl.INT32]
                        )
                        pv_ph3_inline80__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_8, pl.INDEX)
                        v0_inline81__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(pv_v_l1, pl.const(0, pl.INT64), 32768, slots=4)[0], pl.Mem.Mat] = pl._dump.call(
                            "tile.load",
                            [value_cache_bsnd_inline5__ssa_v0, [cache_base_inline9__ssa_v0 + pv_ph0_inline77__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                            pl.Tile[[128, 128], pl.BF16, pl.MemRef(pv_v_l1, slots=4)[0], pl.Mem.Mat],
                            kwargs={"target_memory": pl.Mem.Mat},
                        )
                        v1_inline82__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(pv_v_l1, pl.const(32768, pl.INT64), 32768, slots=4)[1], pl.Mem.Mat] = pl._dump.call(
                            "tile.load",
                            [value_cache_bsnd_inline5__ssa_v0, [cache_base_inline9__ssa_v0 + pv_ph1_inline78__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                            pl.Tile[[128, 128], pl.BF16, pl.MemRef(pv_v_l1, slots=4)[1], pl.Mem.Mat],
                            kwargs={"target_memory": pl.Mem.Mat},
                        )
                        v2_inline83__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(pv_v_l1, pl.const(65536, pl.INT64), 32768, slots=4)[2], pl.Mem.Mat] = pl._dump.call(
                            "tile.load",
                            [value_cache_bsnd_inline5__ssa_v0, [cache_base_inline9__ssa_v0 + pv_ph2_inline79__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                            pl.Tile[[128, 128], pl.BF16, pl.MemRef(pv_v_l1, slots=4)[2], pl.Mem.Mat],
                            kwargs={"target_memory": pl.Mem.Mat},
                        )
                        v3_inline84__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(pv_v_l1, pl.const(98304, pl.INT64), 32768, slots=4)[3], pl.Mem.Mat] = pl._dump.call(
                            "tile.load",
                            [value_cache_bsnd_inline5__ssa_v0, [cache_base_inline9__ssa_v0 + pv_ph3_inline80__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                            pl.Tile[[128, 128], pl.BF16, pl.MemRef(pv_v_l1, slots=4)[3], pl.Mem.Mat],
                            kwargs={"target_memory": pl.Mem.Mat},
                        )
                        pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                        probability0_inline85__ssa_v0: pl.Tile[[16, 128], pl.BF16, pl.MemRef(pv_p_l1, pl.const(0, pl.INT64), 4096, slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = (
                            pl._dump.call(
                                "tile.load",
                                [probability_transfer__ssa_v0, [consume_row_inline76__ssa_v0, 0], [16, 128], [5, 128]],
                                pl.Tile[[16, 128], pl.BF16, pl.MemRef(pv_p_l1, slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                        )
                        t__tmp_v50: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_left_22, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                            "tile.move", [probability0_inline85__ssa_v0], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}
                        )
                        t__tmp_v51: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_23, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl._dump.call(
                            "tile.move", [v0_inline81__ssa_v0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}
                        )
                        pv0_inline86__ssa_v0: pl.Tile[
                            [16, 128], pl.FP32, pl.MemRef(mem_acc_24, pl.const(0, pl.INT64), 8192), pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)
                        ] = pl._dump.call("tile.matmul", [t__tmp_v50, t__tmp_v51], pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)])
                        probability1_inline87__ssa_v0: pl.Tile[[16, 128], pl.BF16, pl.MemRef(pv_p_l1, pl.const(4096, pl.INT64), 4096, slots=2)[1], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = (
                            pl._dump.call(
                                "tile.load",
                                [probability_transfer__ssa_v0, [consume_row_inline76__ssa_v0, 128], [16, 128], [5, 128]],
                                pl.Tile[[16, 128], pl.BF16, pl.MemRef(pv_p_l1, slots=2)[1], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                        )
                        t__tmp_v52: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_left_22, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                            "tile.move", [probability1_inline87__ssa_v0], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}
                        )
                        t__tmp_v53: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_23, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl._dump.call(
                            "tile.move", [v1_inline82__ssa_v0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}
                        )
                        pv1_inline88__ssa_v0: pl.Tile[
                            [16, 128], pl.FP32, pl.MemRef(mem_acc_24, pl.const(0, pl.INT64), 8192), pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)
                        ] = pl._dump.call(
                            "tile.matmul_acc", [pv0_inline86__ssa_v0, t__tmp_v52, t__tmp_v53], pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)]
                        )
                        probability2_inline89__ssa_v0: pl.Tile[[16, 128], pl.BF16, pl.MemRef(pv_p_l1, pl.const(0, pl.INT64), 4096, slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = (
                            pl._dump.call(
                                "tile.load",
                                [probability_transfer__ssa_v0, [consume_row_inline76__ssa_v0, 256], [16, 128], [5, 128]],
                                pl.Tile[[16, 128], pl.BF16, pl.MemRef(pv_p_l1, slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                        )
                        t__tmp_v54: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_left_22, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                            "tile.move", [probability2_inline89__ssa_v0], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}
                        )
                        t__tmp_v55: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_23, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl._dump.call(
                            "tile.move", [v2_inline83__ssa_v0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}
                        )
                        pv2_inline90__ssa_v0: pl.Tile[
                            [16, 128], pl.FP32, pl.MemRef(mem_acc_24, pl.const(0, pl.INT64), 8192), pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)
                        ] = pl._dump.call(
                            "tile.matmul_acc", [pv1_inline88__ssa_v0, t__tmp_v54, t__tmp_v55], pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)]
                        )
                        probability3_inline91__ssa_v0: pl.Tile[[16, 128], pl.BF16, pl.MemRef(pv_p_l1, pl.const(4096, pl.INT64), 4096, slots=2)[1], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = (
                            pl._dump.call(
                                "tile.load",
                                [probability_transfer__ssa_v0, [consume_row_inline76__ssa_v0, 384], [16, 128], [5, 128]],
                                pl.Tile[[16, 128], pl.BF16, pl.MemRef(pv_p_l1, slots=2)[1], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                        )
                        t__tmp_v56: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_left_22, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                            "tile.move", [probability3_inline91__ssa_v0], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}
                        )
                        t__tmp_v57: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_23, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl._dump.call(
                            "tile.move", [v3_inline84__ssa_v0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}
                        )
                        pv3_inline92__ssa_v0: pl.Tile[
                            [16, 128], pl.FP32, pl.MemRef(mem_acc_24, pl.const(0, pl.INT64), 8192), pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)
                        ] = pl._dump.call(
                            "tile.matmul_acc", [pv2_inline90__ssa_v0, t__tmp_v56, t__tmp_v57], pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)]
                        )
                        pv_transfer__store: pl.Tensor[[1152, 128], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 589824)] = pl._dump.call(
                            "tile.store", [pv3_inline92__ssa_v0, [consume_row_inline76__ssa_v0, 0], pv_transfer__ssa_v0], pl.Tensor[[1152, 128], pl.FP32]
                        )
                    else:
                        tail_pages_inline93__ssa_v0: pl.Scalar[pl.INDEX] = page_count_inline51__ssa_v0 - consume_page_inline75__ssa_v0
                        if tail_pages_inline93__ssa_v0 == 1:
                            t__tile_9: pl.Scalar[pl.INT32] = pl._dump.call(
                                "tensor.read", [block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, consume_page_inline75__ssa_v0]], pl.Scalar[pl.INT32]
                            )
                            pv1_ph0_inline94__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_9, pl.INDEX)
                            v1_0_inline95__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(pv_v_l1, pl.const(0, pl.INT64), 32768, slots=4)[0], pl.Mem.Mat] = pl._dump.call(
                                "tile.load",
                                [value_cache_bsnd_inline5__ssa_v0, [cache_base_inline9__ssa_v0 + pv1_ph0_inline94__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                                pl.Tile[[128, 128], pl.BF16, pl.MemRef(pv_v_l1, slots=4)[0], pl.Mem.Mat],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                            probability1_0_inline96__ssa_v0: pl.Tile[[16, 128], pl.BF16, pl.MemRef(pv_p_l1, pl.const(0, pl.INT64), 4096, slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = (
                                pl._dump.call(
                                    "tile.load",
                                    [probability_transfer__ssa_v0, [consume_row_inline76__ssa_v0, 0], [16, 128], [5, 128]],
                                    pl.Tile[[16, 128], pl.BF16, pl.MemRef(pv_p_l1, slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                            )
                            t__tmp_v59: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_left_22, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                "tile.move", [probability1_0_inline96__ssa_v0], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}
                            )
                            t__tmp_v60: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_23, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl._dump.call(
                                "tile.move", [v1_0_inline95__ssa_v0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}
                            )
                            pv_tail1_inline97__ssa_v0: pl.Tile[
                                [16, 128], pl.FP32, pl.MemRef(mem_acc_24, pl.const(0, pl.INT64), 8192), pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)
                            ] = pl._dump.call("tile.matmul", [t__tmp_v59, t__tmp_v60], pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)])
                            pv_transfer__store_v0: pl.Tensor[[1152, 128], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 589824)] = pl._dump.call(
                                "tile.store", [pv_tail1_inline97__ssa_v0, [consume_row_inline76__ssa_v0, 0], pv_transfer__ssa_v0], pl.Tensor[[1152, 128], pl.FP32]
                            )
                        if tail_pages_inline93__ssa_v0 == 2:
                            t__tile_10: pl.Scalar[pl.INT32] = pl._dump.call(
                                "tensor.read", [block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, consume_page_inline75__ssa_v0]], pl.Scalar[pl.INT32]
                            )
                            pv2_ph0_inline98__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_10, pl.INDEX)
                            t__tile_11: pl.Scalar[pl.INT32] = pl._dump.call(
                                "tensor.read", [block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, consume_page_inline75__ssa_v0 + 1]], pl.Scalar[pl.INT32]
                            )
                            pv2_ph1_inline99__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_11, pl.INDEX)
                            v2_0_inline100__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(pv_v_l1, pl.const(0, pl.INT64), 32768, slots=4)[0], pl.Mem.Mat] = pl._dump.call(
                                "tile.load",
                                [value_cache_bsnd_inline5__ssa_v0, [cache_base_inline9__ssa_v0 + pv2_ph0_inline98__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                                pl.Tile[[128, 128], pl.BF16, pl.MemRef(pv_v_l1, slots=4)[0], pl.Mem.Mat],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            v2_1_inline101__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(pv_v_l1, pl.const(32768, pl.INT64), 32768, slots=4)[1], pl.Mem.Mat] = pl._dump.call(
                                "tile.load",
                                [value_cache_bsnd_inline5__ssa_v0, [cache_base_inline9__ssa_v0 + pv2_ph1_inline99__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                                pl.Tile[[128, 128], pl.BF16, pl.MemRef(pv_v_l1, slots=4)[1], pl.Mem.Mat],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                            probability2_0_inline102__ssa_v0: pl.Tile[
                                [16, 128], pl.BF16, pl.MemRef(pv_p_l1, pl.const(0, pl.INT64), 4096, slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])
                            ] = pl._dump.call(
                                "tile.load",
                                [probability_transfer__ssa_v0, [consume_row_inline76__ssa_v0, 0], [16, 128], [5, 128]],
                                pl.Tile[[16, 128], pl.BF16, pl.MemRef(pv_p_l1, slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            t__tmp_v63: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_left_22, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                "tile.move", [probability2_0_inline102__ssa_v0], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}
                            )
                            t__tmp_v64: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_23, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl._dump.call(
                                "tile.move", [v2_0_inline100__ssa_v0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}
                            )
                            pv_tail2_0_inline103__ssa_v0: pl.Tile[
                                [16, 128], pl.FP32, pl.MemRef(mem_acc_24, pl.const(0, pl.INT64), 8192), pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)
                            ] = pl._dump.call("tile.matmul", [t__tmp_v63, t__tmp_v64], pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)])
                            probability2_1_inline104__ssa_v0: pl.Tile[
                                [16, 128], pl.BF16, pl.MemRef(pv_p_l1, pl.const(4096, pl.INT64), 4096, slots=2)[1], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])
                            ] = pl._dump.call(
                                "tile.load",
                                [probability_transfer__ssa_v0, [consume_row_inline76__ssa_v0, 128], [16, 128], [5, 128]],
                                pl.Tile[[16, 128], pl.BF16, pl.MemRef(pv_p_l1, slots=2)[1], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            t__tmp_v65: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_left_22, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                "tile.move", [probability2_1_inline104__ssa_v0], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}
                            )
                            t__tmp_v66: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_23, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl._dump.call(
                                "tile.move", [v2_1_inline101__ssa_v0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}
                            )
                            pv_tail2_1_inline105__ssa_v0: pl.Tile[
                                [16, 128], pl.FP32, pl.MemRef(mem_acc_24, pl.const(0, pl.INT64), 8192), pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)
                            ] = pl._dump.call(
                                "tile.matmul_acc",
                                [pv_tail2_0_inline103__ssa_v0, t__tmp_v65, t__tmp_v66],
                                pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)],
                            )
                            pv_transfer__store_v1: pl.Tensor[[1152, 128], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 589824)] = pl._dump.call(
                                "tile.store", [pv_tail2_1_inline105__ssa_v0, [consume_row_inline76__ssa_v0, 0], pv_transfer__ssa_v0], pl.Tensor[[1152, 128], pl.FP32]
                            )
                        if tail_pages_inline93__ssa_v0 == 3:
                            t__tile_12: pl.Scalar[pl.INT32] = pl._dump.call(
                                "tensor.read", [block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, consume_page_inline75__ssa_v0]], pl.Scalar[pl.INT32]
                            )
                            pv3_ph0_inline106__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_12, pl.INDEX)
                            t__tile_13: pl.Scalar[pl.INT32] = pl._dump.call(
                                "tensor.read", [block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, consume_page_inline75__ssa_v0 + 1]], pl.Scalar[pl.INT32]
                            )
                            pv3_ph1_inline107__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_13, pl.INDEX)
                            t__tile_14: pl.Scalar[pl.INT32] = pl._dump.call(
                                "tensor.read", [block_table_2d_inline7__ssa_v0, [batch_inline49__ssa_v0, consume_page_inline75__ssa_v0 + 2]], pl.Scalar[pl.INT32]
                            )
                            pv3_ph2_inline108__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_14, pl.INDEX)
                            v3_0_inline109__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(pv_v_l1, pl.const(0, pl.INT64), 32768, slots=4)[0], pl.Mem.Mat] = pl._dump.call(
                                "tile.load",
                                [value_cache_bsnd_inline5__ssa_v0, [cache_base_inline9__ssa_v0 + pv3_ph0_inline106__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                                pl.Tile[[128, 128], pl.BF16, pl.MemRef(pv_v_l1, slots=4)[0], pl.Mem.Mat],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            v3_1_inline110__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(pv_v_l1, pl.const(32768, pl.INT64), 32768, slots=4)[1], pl.Mem.Mat] = pl._dump.call(
                                "tile.load",
                                [value_cache_bsnd_inline5__ssa_v0, [cache_base_inline9__ssa_v0 + pv3_ph1_inline107__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                                pl.Tile[[128, 128], pl.BF16, pl.MemRef(pv_v_l1, slots=4)[1], pl.Mem.Mat],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            v3_2_inline111__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(pv_v_l1, pl.const(65536, pl.INT64), 32768, slots=4)[2], pl.Mem.Mat] = pl._dump.call(
                                "tile.load",
                                [value_cache_bsnd_inline5__ssa_v0, [cache_base_inline9__ssa_v0 + pv3_ph2_inline108__ssa_v0 * 128, col_inline53__ssa_v0], [128, 128], [128, 128]],
                                pl.Tile[[128, 128], pl.BF16, pl.MemRef(pv_v_l1, slots=4)[2], pl.Mem.Mat],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                            probability3_0_inline112__ssa_v0: pl.Tile[
                                [16, 128], pl.BF16, pl.MemRef(pv_p_l1, pl.const(0, pl.INT64), 4096, slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])
                            ] = pl._dump.call(
                                "tile.load",
                                [probability_transfer__ssa_v0, [consume_row_inline76__ssa_v0, 0], [16, 128], [5, 128]],
                                pl.Tile[[16, 128], pl.BF16, pl.MemRef(pv_p_l1, slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            t__tmp_v70: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_left_22, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                "tile.move", [probability3_0_inline112__ssa_v0], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}
                            )
                            t__tmp_v71: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_23, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl._dump.call(
                                "tile.move", [v3_0_inline109__ssa_v0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}
                            )
                            pv_tail3_0_inline113__ssa_v0: pl.Tile[
                                [16, 128], pl.FP32, pl.MemRef(mem_acc_24, pl.const(0, pl.INT64), 8192), pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)
                            ] = pl._dump.call("tile.matmul", [t__tmp_v70, t__tmp_v71], pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)])
                            probability3_1_inline114__ssa_v0: pl.Tile[
                                [16, 128], pl.BF16, pl.MemRef(pv_p_l1, pl.const(4096, pl.INT64), 4096, slots=2)[1], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])
                            ] = pl._dump.call(
                                "tile.load",
                                [probability_transfer__ssa_v0, [consume_row_inline76__ssa_v0, 128], [16, 128], [5, 128]],
                                pl.Tile[[16, 128], pl.BF16, pl.MemRef(pv_p_l1, slots=2)[1], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            t__tmp_v72: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_left_22, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                "tile.move", [probability3_1_inline114__ssa_v0], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}
                            )
                            t__tmp_v73: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_23, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl._dump.call(
                                "tile.move", [v3_1_inline110__ssa_v0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}
                            )
                            pv_tail3_1_inline115__ssa_v0: pl.Tile[
                                [16, 128], pl.FP32, pl.MemRef(mem_acc_24, pl.const(0, pl.INT64), 8192), pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)
                            ] = pl._dump.call(
                                "tile.matmul_acc",
                                [pv_tail3_0_inline113__ssa_v0, t__tmp_v72, t__tmp_v73],
                                pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)],
                            )
                            probability3_2_inline116__ssa_v0: pl.Tile[
                                [16, 128], pl.BF16, pl.MemRef(pv_p_l1, pl.const(0, pl.INT64), 4096, slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])
                            ] = pl._dump.call(
                                "tile.load",
                                [probability_transfer__ssa_v0, [consume_row_inline76__ssa_v0, 256], [16, 128], [5, 128]],
                                pl.Tile[[16, 128], pl.BF16, pl.MemRef(pv_p_l1, slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            t__tmp_v74: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_left_22, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                "tile.move", [probability3_2_inline116__ssa_v0], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}
                            )
                            t__tmp_v75: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_23, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl._dump.call(
                                "tile.move", [v3_2_inline111__ssa_v0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}
                            )
                            pv_tail3_2_inline117__ssa_v0: pl.Tile[
                                [16, 128], pl.FP32, pl.MemRef(mem_acc_24, pl.const(0, pl.INT64), 8192), pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)
                            ] = pl._dump.call(
                                "tile.matmul_acc",
                                [pv_tail3_1_inline115__ssa_v0, t__tmp_v74, t__tmp_v75],
                                pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)],
                            )
                            pv_transfer__store_v2: pl.Tensor[[1152, 128], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 589824)] = pl._dump.call(
                                "tile.store", [pv_tail3_2_inline117__ssa_v0, [consume_row_inline76__ssa_v0, 0], pv_transfer__ssa_v0], pl.Tensor[[1152, 128], pl.FP32]
                            )
                    pl.system.sync_set(2, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
            pl._dump.call("system.set_ffts", [ffts_workspace__ssa_v0], pl._dump.Unknown)

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def attn_swpipe_aiv(
        ffts_workspace__ssa_v0: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 2048)],
        key_cache_bsnd_inline4__ssa_v0: pl.InOut[pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        q2d_inline8__ssa_v0: pl.InOut[pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]],
        value_cache_bsnd_inline5__ssa_v0: pl.InOut[pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        active_batch_inline0__ssa_v0: pl.Scalar[pl.INDEX],
        seq_lens__ssa_v0: pl.Tensor[[2], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 8)],
        inv_rms_states__ssa_v0: pl.Tensor[[2, 1], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 8)],
        slot_mapping__ssa_v0: pl.Tensor[[2], pl.INT32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 8)],
        rope_cos__ssa_v0: pl.Tensor[[4096, 128], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 2097152)],
        rope_sin__ssa_v0: pl.Tensor[[4096, 128], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 2097152)],
        k_proj__ssa_v0: pl.Tensor[[2, 1024], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 8192)],
        k_norm_w__ssa_v0: pl.Tensor[[1, 128], pl.FP32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 512)],
        layer_cache_base_token_rows__ssa_v0: pl.Scalar[pl.INDEX],
        v_proj__ssa_v0: pl.Tensor[[2, 1024], pl.FP32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 8192)],
        q_proj__ssa_v0: pl.Tensor[[2, 5120], pl.FP32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 40960)],
        q_norm_w__ssa_v0: pl.Tensor[[1, 128], pl.FP32, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 512)],
        num_tasks_inline1__ssa_v0: pl.Scalar[pl.INDEX],
        block_table_2d_inline7__ssa_v0: pl.Tensor[[active_batch_inline0__ssa_v0, max_blocks_per_seq_inline6__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)],
        cache_base_inline9__ssa_v0: pl.Scalar[pl.INDEX],
        score_transfer__ssa_v0: pl.InOut[pl.Tensor[[1152, 512], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 2359296)]],
        probability_transfer__ssa_v0: pl.InOut[pl.Tensor[[1152, 512], pl.BF16, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 1179648)]],
        pv_transfer__ssa_v0: pl.InOut[pl.Tensor[[1152, 128], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 589824)]],
        out2d_inline10__ssa_v0: pl.Out[pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[pl.Tensor[[1152, 512], pl.FP32], pl.Tensor[[1152, 512], pl.BF16], pl.Tensor[[1152, 128], pl.FP32], pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16]]:
        pl.func_attr({"mx_tensor_views_blocked": True, "split_aiv": True, "dual_aiv_dispatch": True, "buffer_ir_selected": False})
        mem_vec_19: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 5632)
        mem_vec_20: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 3584)
        mem_vec_21: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_22: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_23: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_24: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_26: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 512)
        mem_vec_27: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 512)
        mem_vec_29: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 512)
        mem_vec_34: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_65: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_75: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_76: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_77: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_78: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_80: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 512)
        mem_vec_81: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 512)
        mem_vec_83: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 512)
        mem_vec_88: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_105: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_107: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_108: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_119: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_132: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_133: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_149: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        core_inline12__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        pl._dump.call("system.set_ffts", [ffts_workspace__ssa_v0], pl._dump.Unknown)
        aiv_id_inline13__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
        rope_core_inline14__ssa_v0: pl.Scalar[pl.INDEX] = core_inline12__ssa_v0 * 2 + aiv_id_inline13__ssa_v0
        if rope_core_inline14__ssa_v0 < 32:
            q_red_pad_inline15__tile: pl.Tile[[1, 1408], pl.FP32, pl.MemRef(mem_vec_19, pl.const(0, pl.INT64), 5632), pl.Mem.Vec] = pl._dump.call(
                "tile.full", [[1, 1408], 0.0], pl.Tile[[1, 1408], pl.FP32, pl.Mem.Vec], kwargs={"dtype": pl.FP32}
            )
            k_red_pad_inline16__tile: pl.Tile[[1, 896], pl.FP32, pl.MemRef(mem_vec_20, pl.const(0, pl.INT64), 3584), pl.Mem.Vec] = pl._dump.call(
                "tile.full", [[1, 896], 0.0], pl.Tile[[1, 896], pl.FP32, pl.Mem.Vec], kwargs={"dtype": pl.FP32}
            )
            for it_inline17__idx_v0, (key_cache_bsnd_inline4__iter_v1, q2d_inline8__iter_v1, value_cache_bsnd_inline5__iter_v1) in pl.range(
                0, 4, 2, init_values=(key_cache_bsnd_inline4__ssa_v0, q2d_inline8__ssa_v0, value_cache_bsnd_inline5__ssa_v0)
            ):
                g_idx_inline18__ssa_v0: pl.Scalar[pl.INDEX] = rope_core_inline14__ssa_v0 + it_inline17__idx_v0 * 32
                g_idx_inline18__ssa_v0_1: pl.Scalar[pl.INDEX] = rope_core_inline14__ssa_v0 + (it_inline17__idx_v0 + 1) * 32
                if g_idx_inline18__ssa_v0 < active_batch_inline0__ssa_v0 * 8:
                    kv_head_inline19__ssa_v0: pl.Scalar[pl.INDEX] = g_idx_inline18__ssa_v0 // active_batch_inline0__ssa_v0
                    batch_idx_inline20__ssa_v0: pl.Scalar[pl.INDEX] = g_idx_inline18__ssa_v0 - kv_head_inline19__ssa_v0 * active_batch_inline0__ssa_v0
                    t__tile: pl.Scalar[pl.INT32] = pl._dump.call("tensor.read", [seq_lens__ssa_v0, [batch_idx_inline20__ssa_v0]], pl.Scalar[pl.INT32])
                    position_inline21__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile, pl.INDEX) - 1
                    inv_rms_inline22__tile: pl.Scalar[pl.FP32] = pl._dump.call("tensor.read", [inv_rms_states__ssa_v0, [batch_idx_inline20__ssa_v0, 0]], pl.Scalar[pl.FP32])
                    t__tile_1: pl.Scalar[pl.INT32] = pl._dump.call("tensor.read", [slot_mapping__ssa_v0, [batch_idx_inline20__ssa_v0]], pl.Scalar[pl.INT32])
                    write_slot_inline23__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.cast(t__tile_1, pl.INDEX), 0)
                    cache_col_inline28__ssa_v0: pl.Scalar[pl.INDEX] = kv_head_inline19__ssa_v0 * 128
                    cache_row_inline37__ssa_v0: pl.Scalar[pl.INDEX] = layer_cache_base_token_rows__ssa_v0 + write_slot_inline23__ssa_v0
                    q_head_base_inline39__ssa_v0: pl.Scalar[pl.INDEX] = kv_head_inline19__ssa_v0 * 5
                    q_row_inline47__ssa_v0: pl.Scalar[pl.INDEX] = batch_idx_inline20__ssa_v0 * 40 + q_head_base_inline39__ssa_v0
                    cos_lo_inline24__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_21, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl._dump.call(
                        "tile.load", [rope_cos__ssa_v0, [position_inline21__ssa_v0, 0], [1, 64], [1, 64]], pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec], kwargs={"target_memory": pl.Mem.Vec}
                    )
                    cos_hi_inline25__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_22, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl._dump.call(
                        "tile.load", [rope_cos__ssa_v0, [position_inline21__ssa_v0, 64], [1, 64], [1, 64]], pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec], kwargs={"target_memory": pl.Mem.Vec}
                    )
                    sin_lo_inline26__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_23, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl._dump.call(
                        "tile.load", [rope_sin__ssa_v0, [position_inline21__ssa_v0, 0], [1, 64], [1, 64]], pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec], kwargs={"target_memory": pl.Mem.Vec}
                    )
                    sin_hi_inline27__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl._dump.call(
                        "tile.load", [rope_sin__ssa_v0, [position_inline21__ssa_v0, 64], [1, 64], [1, 64]], pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec], kwargs={"target_memory": pl.Mem.Vec}
                    )
                    t__tile_2: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INDEX), 512), pl.Mem.Vec] = pl._dump.call(
                        "tile.load",
                        [k_proj__ssa_v0, [batch_idx_inline20__ssa_v0, cache_col_inline28__ssa_v0], [1, 128], [1, 128]],
                        pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec],
                        kwargs={"target_memory": pl.Mem.Vec},
                    )
                    k_norm_w__tile: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_26, pl.const(0, pl.INT64), 512), pl.Mem.Vec] = pl._dump.call(
                        "tile.load", [k_norm_w__ssa_v0, [0, 0], [1, 128], [1, 128]], pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec], kwargs={"target_memory": pl.Mem.Vec}
                    )
                    t__tile_3: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_27, pl.const(0, pl.INT64), 512), pl.Mem.Vec] = pl._dump.call(
                        "tile.load",
                        [v_proj__ssa_v0, [batch_idx_inline20__ssa_v0, cache_col_inline28__ssa_v0], [1, 128], [1, 128]],
                        pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec],
                        kwargs={"target_memory": pl.Mem.Vec},
                    )
                    t__tile_4: pl.Tile[[1, 640], pl.FP32, pl.MemRef(mem_vec_65, pl.const(0, pl.INDEX), 2560), pl.Mem.Vec] = pl._dump.call(
                        "tile.load",
                        [q_proj__ssa_v0, [batch_idx_inline20__ssa_v0, q_head_base_inline39__ssa_v0 * 128], [1, 640], [1, 640]],
                        pl.Tile[[1, 640], pl.FP32, pl.Mem.Vec],
                        kwargs={"target_memory": pl.Mem.Vec},
                    )
                    q_norm_w__tile: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_29, pl.const(0, pl.INT64), 512), pl.Mem.Vec] = pl._dump.call(
                        "tile.load", [q_norm_w__ssa_v0, [0, 0], [1, 128], [1, 128]], pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec], kwargs={"target_memory": pl.Mem.Vec}
                    )
                    t__tile_5: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_132, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec] = pl._dump.call(
                        "tile.concat", [t__tile_2, k_red_pad_inline16__tile], pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_6: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_132, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [t__tile_5, [8, 128]], pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec]
                    )
                    k_raw_inline29__tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_132, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec] = pl._dump.call(
                        "tile.muls", [t__tile_6, inv_rms_inline22__tile], pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_7: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec] = pl._dump.call(
                        "tile.mul", [k_raw_inline29__tile, k_raw_inline29__tile], pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec]
                    )
                    tmp_tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_149, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec] = pl._dump.call(
                        "tile.create", [[8, 128]], pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec], kwargs={"dtype": pl.FP32, "target_memory": pl.Mem.Vec}
                    )
                    k_ss_inline30__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_34, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl._dump.call(
                        "tile.row_sum", [t__tile_7, tmp_tile], pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec]
                    )
                    t__rm_a0_tmp_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_34, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [k_ss_inline30__tile, [1, 8]], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec]
                    )
                    t__row_major_tmp_v1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INDEX), 32), pl.Mem.Vec] = pl._dump.call(
                        "tile.muls", [t__rm_a0_tmp_v0, 0.0078125], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_8: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INDEX), 32), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [t__row_major_tmp_v1, [8, 1]], pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec]
                    )
                    t__rm_a0_tmp_v2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INDEX), 32), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [t__tile_8, [1, 8]], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec]
                    )
                    t__row_major_tmp_v3: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INDEX), 32), pl.Mem.Vec] = pl._dump.call(
                        "tile.adds", [t__rm_a0_tmp_v2, 9.9999999999999995e-07], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_9: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INDEX), 32), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [t__row_major_tmp_v3, [8, 1]], pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec]
                    )
                    t__rm_a0_tmp_v4: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INDEX), 32), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [t__tile_9, [1, 8]], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec]
                    )
                    t__row_major_tmp_v5: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INDEX), 32), pl.Mem.Vec] = pl._dump.call(
                        "tile.sqrt", [t__rm_a0_tmp_v4], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_10: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INDEX), 32), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [t__row_major_tmp_v5, [8, 1]], pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec]
                    )
                    k_inv_inline31__rm_a0_tmp_v6: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INDEX), 32), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [t__tile_10, [1, 8]], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec]
                    )
                    k_inv_inline31__row_major_tmp_v7: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_149, pl.const(0, pl.INDEX), 32), pl.Mem.Vec] = pl._dump.call(
                        "tile.recip", [k_inv_inline31__rm_a0_tmp_v6], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec]
                    )
                    k_inv_inline31__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_149, pl.const(0, pl.INDEX), 32), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [k_inv_inline31__row_major_tmp_v7, [8, 1]], pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_11: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_132, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec] = pl._dump.call(
                        "tile.col_expand_mul", [k_raw_inline29__tile, k_norm_w__tile], pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec]
                    )
                    k_normed_inline32__tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_132, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec] = pl._dump.call(
                        "tile.row_expand_mul", [t__tile_11, k_inv_inline31__tile], pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec]
                    )
                    k_full_inline33__tile: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_132, pl.const(0, pl.INDEX), 512), pl.Mem.Vec] = pl._dump.call(
                        "tile.slice", [k_normed_inline32__tile, [1, 128], [0, 0]], pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec]
                    )
                    k_lo_inline34__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_132, pl.const(0, pl.INDEX), 256), pl.Mem.Vec] = pl._dump.call(
                        "tile.slice", [k_full_inline33__tile, [1, 64], [0, 0]], pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec]
                    )
                    k_hi_inline35__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_132, pl.const(256, pl.INDEX), 256), pl.Mem.Vec] = pl._dump.call(
                        "tile.slice", [k_full_inline33__tile, [1, 64], [0, 64]], pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_12: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INDEX), 256), pl.Mem.Vec] = pl._dump.call(
                        "tile.col_expand_mul", [k_lo_inline34__tile, cos_lo_inline24__tile], pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_13: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_149, pl.const(0, pl.INDEX), 256), pl.Mem.Vec] = pl._dump.call(
                        "tile.col_expand_mul", [k_hi_inline35__tile, sin_lo_inline26__tile], pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_14: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INDEX), 256), pl.Mem.Vec] = pl._dump.call(
                        "tile.sub", [t__tile_12, t__tile_13], pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_15: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_149, pl.const(0, pl.INDEX), 256), pl.Mem.Vec] = pl._dump.call(
                        "tile.col_expand_mul", [k_hi_inline35__tile, cos_hi_inline25__tile], pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_16: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_132, pl.const(0, pl.INDEX), 256), pl.Mem.Vec] = pl._dump.call(
                        "tile.col_expand_mul", [k_lo_inline34__tile, sin_hi_inline27__tile], pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_17: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_149, pl.const(0, pl.INDEX), 256), pl.Mem.Vec] = pl._dump.call(
                        "tile.add", [t__tile_15, t__tile_16], pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec]
                    )
                    k_rotated_inline36__tile: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_132, pl.const(0, pl.INDEX), 512), pl.Mem.Vec] = pl._dump.call(
                        "tile.concat", [t__tile_14, t__tile_17], pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_18: pl.Tile[[1, 128], pl.BF16, pl.MemRef(mem_vec_26, pl.const(0, pl.INDEX), 256), pl.Mem.Vec] = pl._dump.call(
                        "tile.cast", [k_rotated_inline36__tile], pl.Tile[[1, 128], pl.BF16, pl.Mem.Vec], kwargs={"target_type": pl.BF16, "mode": 2}
                    )
                    t__tile_19: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_132, pl.const(0, pl.INDEX), 512), pl.Mem.Vec] = pl._dump.call(
                        "tile.muls", [t__tile_3, inv_rms_inline22__tile], pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec]
                    )
                    v_row_bf16_inline38__tile: pl.Tile[[1, 128], pl.BF16, pl.MemRef(mem_vec_27, pl.const(0, pl.INDEX), 256), pl.Mem.Vec] = pl._dump.call(
                        "tile.cast", [t__tile_19], pl.Tile[[1, 128], pl.BF16, pl.Mem.Vec], kwargs={"target_type": pl.BF16, "mode": 2}
                    )
                    t__tile_20: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_132, pl.const(0, pl.INDEX), 8192), pl.Mem.Vec] = pl._dump.call(
                        "tile.concat", [t__tile_4, q_red_pad_inline15__tile], pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_21: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_132, pl.const(0, pl.INDEX), 8192), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [t__tile_20, [16, 128]], pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec]
                    )
                    q_raw_inline40__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_132, pl.const(0, pl.INDEX), 8192), pl.Mem.Vec] = pl._dump.call(
                        "tile.muls", [t__tile_21, inv_rms_inline22__tile], pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_22: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INDEX), 8192), pl.Mem.Vec] = pl._dump.call(
                        "tile.mul", [q_raw_inline40__tile, q_raw_inline40__tile], pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec]
                    )
                    tmp_tile_1: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_149, pl.const(0, pl.INDEX), 8192), pl.Mem.Vec] = pl._dump.call(
                        "tile.create", [[16, 128]], pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec], kwargs={"dtype": pl.FP32, "target_memory": pl.Mem.Vec}
                    )
                    q_ss_inline41__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_65, pl.const(0, pl.INDEX), 64), pl.Mem.Vec] = pl._dump.call(
                        "tile.row_sum", [t__tile_22, tmp_tile_1], pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec]
                    )
                    t__rm_a0_tmp_v8: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_65, pl.const(0, pl.INDEX), 64), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [q_ss_inline41__tile, [1, 16]], pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec]
                    )
                    t__row_major_tmp_v9: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INDEX), 64), pl.Mem.Vec] = pl._dump.call(
                        "tile.muls", [t__rm_a0_tmp_v8, 0.0078125], pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_23: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INDEX), 64), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [t__row_major_tmp_v9, [16, 1]], pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec]
                    )
                    t__rm_a0_tmp_v10: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INDEX), 64), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [t__tile_23, [1, 16]], pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec]
                    )
                    t__row_major_tmp_v11: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INDEX), 64), pl.Mem.Vec] = pl._dump.call(
                        "tile.adds", [t__rm_a0_tmp_v10, 9.9999999999999995e-07], pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_24: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INDEX), 64), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [t__row_major_tmp_v11, [16, 1]], pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec]
                    )
                    t__rm_a0_tmp_v12: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INDEX), 64), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [t__tile_24, [1, 16]], pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec]
                    )
                    t__row_major_tmp_v13: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INDEX), 64), pl.Mem.Vec] = pl._dump.call(
                        "tile.sqrt", [t__rm_a0_tmp_v12], pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_25: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INDEX), 64), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [t__row_major_tmp_v13, [16, 1]], pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec]
                    )
                    q_inv_inline42__rm_a0_tmp_v14: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INDEX), 64), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [t__tile_25, [1, 16]], pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec]
                    )
                    q_inv_inline42__row_major_tmp_v15: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_149, pl.const(0, pl.INDEX), 64), pl.Mem.Vec] = pl._dump.call(
                        "tile.recip", [q_inv_inline42__rm_a0_tmp_v14], pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec]
                    )
                    q_inv_inline42__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_149, pl.const(0, pl.INDEX), 64), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [q_inv_inline42__row_major_tmp_v15, [16, 1]], pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_26: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_132, pl.const(0, pl.INDEX), 8192), pl.Mem.Vec] = pl._dump.call(
                        "tile.col_expand_mul", [q_raw_inline40__tile, q_norm_w__tile], pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec]
                    )
                    q_heads_inline43__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_132, pl.const(0, pl.INDEX), 8192), pl.Mem.Vec] = pl._dump.call(
                        "tile.row_expand_mul", [t__tile_26, q_inv_inline42__tile], pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec]
                    )
                    q_lo_inline44__tile_textract: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec] = pl._dump.call(
                        "tile.extract", [q_heads_inline43__tile, 0, 0, [16, 64]], pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec], kwargs={"target_memory": pl.Mem.Vec}
                    )
                    q_hi_inline45__tile_textract: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_149, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec] = pl._dump.call(
                        "tile.extract", [q_heads_inline43__tile, 0, 64, [16, 64]], pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec], kwargs={"target_memory": pl.Mem.Vec}
                    )
                    t__tile_27: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_132, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec] = pl._dump.call(
                        "tile.col_expand_mul", [q_lo_inline44__tile_textract, cos_lo_inline24__tile], pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_28: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_65, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl._dump.call(
                        "tile.col_expand_mul", [q_hi_inline45__tile_textract, sin_lo_inline26__tile], pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_29: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_65, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl._dump.call(
                        "tile.sub", [t__tile_27, t__tile_28], pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_30: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_132, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec] = pl._dump.call(
                        "tile.col_expand_mul", [q_hi_inline45__tile_textract, cos_hi_inline25__tile], pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_31: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec] = pl._dump.call(
                        "tile.col_expand_mul", [q_lo_inline44__tile_textract, sin_hi_inline27__tile], pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_32: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec] = pl._dump.call(
                        "tile.add", [t__tile_30, t__tile_31], pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec]
                    )
                    q_rotated_inline46__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_132, pl.const(0, pl.INDEX), 8192), pl.Mem.Vec] = pl._dump.call(
                        "tile.concat", [t__tile_29, t__tile_32], pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_33: pl.Tile[[5, 128], pl.FP32, pl.MemRef(mem_vec_132, pl.const(0, pl.INDEX), 2560), pl.Mem.Vec] = pl._dump.call(
                        "tile.slice", [q_rotated_inline46__tile, [5, 128], [0, 0]], pl.Tile[[5, 128], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_34: pl.Tile[[5, 128], pl.BF16, pl.MemRef(mem_vec_132, pl.const(0, pl.INDEX), 1280), pl.Mem.Vec] = pl._dump.call(
                        "tile.cast", [t__tile_33], pl.Tile[[5, 128], pl.BF16, pl.Mem.Vec], kwargs={"target_type": pl.BF16, "mode": 2}
                    )
                    key_cache_bsnd_inline4__tile: pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl._dump.call(
                        "tile.store",
                        [t__tile_18, [cache_row_inline37__ssa_v0, cache_col_inline28__ssa_v0], key_cache_bsnd_inline4__iter_v1],
                        pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16],
                    )
                    value_cache_bsnd_inline5__tile: pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl._dump.call(
                        "tile.store",
                        [v_row_bf16_inline38__tile, [cache_row_inline37__ssa_v0, cache_col_inline28__ssa_v0], value_cache_bsnd_inline5__iter_v1],
                        pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16],
                    )
                    q2d_inline8__tile: pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl._dump.call(
                        "tile.store", [t__tile_34, [q_row_inline47__ssa_v0, 0], q2d_inline8__iter_v1], pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16]
                    )
                    key_cache_bsnd_inline4__phi_v4, q2d_inline8__phi_v4, value_cache_bsnd_inline5__phi_v4 = pl._dump.yield_(
                        key_cache_bsnd_inline4__tile,
                        q2d_inline8__tile,
                        value_cache_bsnd_inline5__tile,
                        types=[
                            pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16, pl.MemRef("mem_ddr_72", pl.const(0, pl.INT64), 0)],
                            pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16, pl.MemRef("mem_ddr_73", pl.const(0, pl.INT64), 0)],
                            pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16, pl.MemRef("mem_ddr_74", pl.const(0, pl.INT64), 0)],
                        ],
                    )
                else:
                    key_cache_bsnd_inline4__phi_v4, q2d_inline8__phi_v4, value_cache_bsnd_inline5__phi_v4 = pl._dump.yield_(
                        key_cache_bsnd_inline4__iter_v1,
                        q2d_inline8__iter_v1,
                        value_cache_bsnd_inline5__iter_v1,
                        types=[
                            pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16, pl.MemRef("mem_ddr_72", pl.const(0, pl.INT64), 0)],
                            pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16, pl.MemRef("mem_ddr_73", pl.const(0, pl.INT64), 0)],
                            pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16, pl.MemRef("mem_ddr_74", pl.const(0, pl.INT64), 0)],
                        ],
                    )
                if g_idx_inline18__ssa_v0_1 < active_batch_inline0__ssa_v0 * 8:
                    kv_head_inline19__ssa_v0_1: pl.Scalar[pl.INDEX] = g_idx_inline18__ssa_v0_1 // active_batch_inline0__ssa_v0
                    batch_idx_inline20__ssa_v0_1: pl.Scalar[pl.INDEX] = g_idx_inline18__ssa_v0_1 - kv_head_inline19__ssa_v0_1 * active_batch_inline0__ssa_v0
                    t__tile_35: pl.Scalar[pl.INT32] = pl._dump.call("tensor.read", [seq_lens__ssa_v0, [batch_idx_inline20__ssa_v0_1]], pl.Scalar[pl.INT32])
                    position_inline21__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.cast(t__tile_35, pl.INDEX) - 1
                    inv_rms_inline22__tile_1: pl.Scalar[pl.FP32] = pl._dump.call("tensor.read", [inv_rms_states__ssa_v0, [batch_idx_inline20__ssa_v0_1, 0]], pl.Scalar[pl.FP32])
                    t__tile_36: pl.Scalar[pl.INT32] = pl._dump.call("tensor.read", [slot_mapping__ssa_v0, [batch_idx_inline20__ssa_v0_1]], pl.Scalar[pl.INT32])
                    write_slot_inline23__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.max(pl.cast(t__tile_36, pl.INDEX), 0)
                    cache_col_inline28__ssa_v0_1: pl.Scalar[pl.INDEX] = kv_head_inline19__ssa_v0_1 * 128
                    cache_row_inline37__ssa_v0_1: pl.Scalar[pl.INDEX] = layer_cache_base_token_rows__ssa_v0 + write_slot_inline23__ssa_v0_1
                    q_head_base_inline39__ssa_v0_1: pl.Scalar[pl.INDEX] = kv_head_inline19__ssa_v0_1 * 5
                    q_row_inline47__ssa_v0_1: pl.Scalar[pl.INDEX] = batch_idx_inline20__ssa_v0_1 * 40 + q_head_base_inline39__ssa_v0_1
                    cos_lo_inline24__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_75, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl._dump.call(
                        "tile.load", [rope_cos__ssa_v0, [position_inline21__ssa_v0_1, 0], [1, 64], [1, 64]], pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec], kwargs={"target_memory": pl.Mem.Vec}
                    )
                    cos_hi_inline25__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_76, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl._dump.call(
                        "tile.load", [rope_cos__ssa_v0, [position_inline21__ssa_v0_1, 64], [1, 64], [1, 64]], pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec], kwargs={"target_memory": pl.Mem.Vec}
                    )
                    sin_lo_inline26__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_77, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl._dump.call(
                        "tile.load", [rope_sin__ssa_v0, [position_inline21__ssa_v0_1, 0], [1, 64], [1, 64]], pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec], kwargs={"target_memory": pl.Mem.Vec}
                    )
                    sin_hi_inline27__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_78, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl._dump.call(
                        "tile.load", [rope_sin__ssa_v0, [position_inline21__ssa_v0_1, 64], [1, 64], [1, 64]], pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec], kwargs={"target_memory": pl.Mem.Vec}
                    )
                    t__tile_37: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 512), pl.Mem.Vec] = pl._dump.call(
                        "tile.load",
                        [k_proj__ssa_v0, [batch_idx_inline20__ssa_v0_1, cache_col_inline28__ssa_v0_1], [1, 128], [1, 128]],
                        pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec],
                        kwargs={"target_memory": pl.Mem.Vec},
                    )
                    k_norm_w__tile_1: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_80, pl.const(0, pl.INT64), 512), pl.Mem.Vec] = pl._dump.call(
                        "tile.load", [k_norm_w__ssa_v0, [0, 0], [1, 128], [1, 128]], pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec], kwargs={"target_memory": pl.Mem.Vec}
                    )
                    t__tile_38: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_81, pl.const(0, pl.INT64), 512), pl.Mem.Vec] = pl._dump.call(
                        "tile.load",
                        [v_proj__ssa_v0, [batch_idx_inline20__ssa_v0_1, cache_col_inline28__ssa_v0_1], [1, 128], [1, 128]],
                        pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec],
                        kwargs={"target_memory": pl.Mem.Vec},
                    )
                    t__tile_39: pl.Tile[[1, 640], pl.FP32, pl.MemRef(mem_vec_119, pl.const(0, pl.INDEX), 2560), pl.Mem.Vec] = pl._dump.call(
                        "tile.load",
                        [q_proj__ssa_v0, [batch_idx_inline20__ssa_v0_1, q_head_base_inline39__ssa_v0_1 * 128], [1, 640], [1, 640]],
                        pl.Tile[[1, 640], pl.FP32, pl.Mem.Vec],
                        kwargs={"target_memory": pl.Mem.Vec},
                    )
                    q_norm_w__tile_1: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_83, pl.const(0, pl.INT64), 512), pl.Mem.Vec] = pl._dump.call(
                        "tile.load", [q_norm_w__ssa_v0, [0, 0], [1, 128], [1, 128]], pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec], kwargs={"target_memory": pl.Mem.Vec}
                    )
                    t__tile_40: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_105, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec] = pl._dump.call(
                        "tile.concat", [t__tile_37, k_red_pad_inline16__tile], pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_41: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_105, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [t__tile_40, [8, 128]], pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec]
                    )
                    k_raw_inline29__tile_1: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_105, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec] = pl._dump.call(
                        "tile.muls", [t__tile_41, inv_rms_inline22__tile_1], pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_42: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec] = pl._dump.call(
                        "tile.mul", [k_raw_inline29__tile_1, k_raw_inline29__tile_1], pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec]
                    )
                    tmp_tile_2: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_108, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec] = pl._dump.call(
                        "tile.create", [[8, 128]], pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec], kwargs={"dtype": pl.FP32, "target_memory": pl.Mem.Vec}
                    )
                    k_ss_inline30__tile_1: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_88, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl._dump.call(
                        "tile.row_sum", [t__tile_42, tmp_tile_2], pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec]
                    )
                    t__rm_a0_tmp_v0_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_88, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [k_ss_inline30__tile_1, [1, 8]], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec]
                    )
                    t__row_major_tmp_v1_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 32), pl.Mem.Vec] = pl._dump.call(
                        "tile.muls", [t__rm_a0_tmp_v0_1, 0.0078125], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_43: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 32), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [t__row_major_tmp_v1_1, [8, 1]], pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec]
                    )
                    t__rm_a0_tmp_v2_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 32), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [t__tile_43, [1, 8]], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec]
                    )
                    t__row_major_tmp_v3_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 32), pl.Mem.Vec] = pl._dump.call(
                        "tile.adds", [t__rm_a0_tmp_v2_1, 9.9999999999999995e-07], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_44: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 32), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [t__row_major_tmp_v3_1, [8, 1]], pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec]
                    )
                    t__rm_a0_tmp_v4_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 32), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [t__tile_44, [1, 8]], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec]
                    )
                    t__row_major_tmp_v5_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 32), pl.Mem.Vec] = pl._dump.call(
                        "tile.sqrt", [t__rm_a0_tmp_v4_1], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_45: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 32), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [t__row_major_tmp_v5_1, [8, 1]], pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec]
                    )
                    k_inv_inline31__rm_a0_tmp_v6_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 32), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [t__tile_45, [1, 8]], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec]
                    )
                    k_inv_inline31__row_major_tmp_v7_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_108, pl.const(0, pl.INDEX), 32), pl.Mem.Vec] = pl._dump.call(
                        "tile.recip", [k_inv_inline31__rm_a0_tmp_v6_1], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec]
                    )
                    k_inv_inline31__tile_1: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_108, pl.const(0, pl.INDEX), 32), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [k_inv_inline31__row_major_tmp_v7_1, [8, 1]], pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_46: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_105, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec] = pl._dump.call(
                        "tile.col_expand_mul", [k_raw_inline29__tile_1, k_norm_w__tile_1], pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec]
                    )
                    k_normed_inline32__tile_1: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_105, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec] = pl._dump.call(
                        "tile.row_expand_mul", [t__tile_46, k_inv_inline31__tile_1], pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec]
                    )
                    k_full_inline33__tile_1: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_105, pl.const(0, pl.INDEX), 512), pl.Mem.Vec] = pl._dump.call(
                        "tile.slice", [k_normed_inline32__tile_1, [1, 128], [0, 0]], pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec]
                    )
                    k_lo_inline34__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_105, pl.const(0, pl.INDEX), 256), pl.Mem.Vec] = pl._dump.call(
                        "tile.slice", [k_full_inline33__tile_1, [1, 64], [0, 0]], pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec]
                    )
                    k_hi_inline35__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_105, pl.const(256, pl.INDEX), 256), pl.Mem.Vec] = pl._dump.call(
                        "tile.slice", [k_full_inline33__tile_1, [1, 64], [0, 64]], pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_47: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 256), pl.Mem.Vec] = pl._dump.call(
                        "tile.col_expand_mul", [k_lo_inline34__tile_1, cos_lo_inline24__tile_1], pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_48: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_108, pl.const(0, pl.INDEX), 256), pl.Mem.Vec] = pl._dump.call(
                        "tile.col_expand_mul", [k_hi_inline35__tile_1, sin_lo_inline26__tile_1], pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_49: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 256), pl.Mem.Vec] = pl._dump.call(
                        "tile.sub", [t__tile_47, t__tile_48], pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_50: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_108, pl.const(0, pl.INDEX), 256), pl.Mem.Vec] = pl._dump.call(
                        "tile.col_expand_mul", [k_hi_inline35__tile_1, cos_hi_inline25__tile_1], pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_51: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_105, pl.const(0, pl.INDEX), 256), pl.Mem.Vec] = pl._dump.call(
                        "tile.col_expand_mul", [k_lo_inline34__tile_1, sin_hi_inline27__tile_1], pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_52: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_108, pl.const(0, pl.INDEX), 256), pl.Mem.Vec] = pl._dump.call(
                        "tile.add", [t__tile_50, t__tile_51], pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec]
                    )
                    k_rotated_inline36__tile_1: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_105, pl.const(0, pl.INDEX), 512), pl.Mem.Vec] = pl._dump.call(
                        "tile.concat", [t__tile_49, t__tile_52], pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_53: pl.Tile[[1, 128], pl.BF16, pl.MemRef(mem_vec_80, pl.const(0, pl.INDEX), 256), pl.Mem.Vec] = pl._dump.call(
                        "tile.cast", [k_rotated_inline36__tile_1], pl.Tile[[1, 128], pl.BF16, pl.Mem.Vec], kwargs={"target_type": pl.BF16, "mode": 2}
                    )
                    t__tile_54: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_105, pl.const(0, pl.INDEX), 512), pl.Mem.Vec] = pl._dump.call(
                        "tile.muls", [t__tile_38, inv_rms_inline22__tile_1], pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec]
                    )
                    v_row_bf16_inline38__tile_1: pl.Tile[[1, 128], pl.BF16, pl.MemRef(mem_vec_81, pl.const(0, pl.INDEX), 256), pl.Mem.Vec] = pl._dump.call(
                        "tile.cast", [t__tile_54], pl.Tile[[1, 128], pl.BF16, pl.Mem.Vec], kwargs={"target_type": pl.BF16, "mode": 2}
                    )
                    t__tile_55: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_105, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl._dump.call(
                        "tile.concat", [t__tile_39, q_red_pad_inline15__tile], pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_56: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_105, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [t__tile_55, [16, 128]], pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec]
                    )
                    q_raw_inline40__tile_1: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_105, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl._dump.call(
                        "tile.muls", [t__tile_56, inv_rms_inline22__tile_1], pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_57: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl._dump.call(
                        "tile.mul", [q_raw_inline40__tile_1, q_raw_inline40__tile_1], pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec]
                    )
                    tmp_tile_3: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_108, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl._dump.call(
                        "tile.create", [[16, 128]], pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec], kwargs={"dtype": pl.FP32, "target_memory": pl.Mem.Vec}
                    )
                    q_ss_inline41__tile_1: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_119, pl.const(0, pl.INDEX), 64), pl.Mem.Vec] = pl._dump.call(
                        "tile.row_sum", [t__tile_57, tmp_tile_3], pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec]
                    )
                    t__rm_a0_tmp_v8_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_119, pl.const(0, pl.INDEX), 64), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [q_ss_inline41__tile_1, [1, 16]], pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec]
                    )
                    t__row_major_tmp_v9_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 64), pl.Mem.Vec] = pl._dump.call(
                        "tile.muls", [t__rm_a0_tmp_v8_1, 0.0078125], pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_58: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 64), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [t__row_major_tmp_v9_1, [16, 1]], pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec]
                    )
                    t__rm_a0_tmp_v10_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 64), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [t__tile_58, [1, 16]], pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec]
                    )
                    t__row_major_tmp_v11_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 64), pl.Mem.Vec] = pl._dump.call(
                        "tile.adds", [t__rm_a0_tmp_v10_1, 9.9999999999999995e-07], pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_59: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 64), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [t__row_major_tmp_v11_1, [16, 1]], pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec]
                    )
                    t__rm_a0_tmp_v12_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 64), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [t__tile_59, [1, 16]], pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec]
                    )
                    t__row_major_tmp_v13_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 64), pl.Mem.Vec] = pl._dump.call(
                        "tile.sqrt", [t__rm_a0_tmp_v12_1], pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_60: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 64), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [t__row_major_tmp_v13_1, [16, 1]], pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec]
                    )
                    q_inv_inline42__rm_a0_tmp_v14_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 64), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [t__tile_60, [1, 16]], pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec]
                    )
                    q_inv_inline42__row_major_tmp_v15_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_108, pl.const(0, pl.INDEX), 64), pl.Mem.Vec] = pl._dump.call(
                        "tile.recip", [q_inv_inline42__rm_a0_tmp_v14_1], pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec]
                    )
                    q_inv_inline42__tile_1: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_108, pl.const(0, pl.INDEX), 64), pl.Mem.Vec] = pl._dump.call(
                        "tile.reshape", [q_inv_inline42__row_major_tmp_v15_1, [16, 1]], pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_61: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_105, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl._dump.call(
                        "tile.col_expand_mul", [q_raw_inline40__tile_1, q_norm_w__tile_1], pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec]
                    )
                    q_heads_inline43__tile_1: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_105, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl._dump.call(
                        "tile.row_expand_mul", [t__tile_61, q_inv_inline42__tile_1], pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec]
                    )
                    q_lo_inline44__tile_textract_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec] = pl._dump.call(
                        "tile.extract", [q_heads_inline43__tile_1, 0, 0, [16, 64]], pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec], kwargs={"target_memory": pl.Mem.Vec}
                    )
                    q_hi_inline45__tile_textract_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_108, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec] = pl._dump.call(
                        "tile.extract", [q_heads_inline43__tile_1, 0, 64, [16, 64]], pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec], kwargs={"target_memory": pl.Mem.Vec}
                    )
                    t__tile_62: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_105, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec] = pl._dump.call(
                        "tile.col_expand_mul", [q_lo_inline44__tile_textract_1, cos_lo_inline24__tile_1], pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_63: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_119, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl._dump.call(
                        "tile.col_expand_mul", [q_hi_inline45__tile_textract_1, sin_lo_inline26__tile_1], pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_64: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_119, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl._dump.call(
                        "tile.sub", [t__tile_62, t__tile_63], pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_65: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_105, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec] = pl._dump.call(
                        "tile.col_expand_mul", [q_hi_inline45__tile_textract_1, cos_hi_inline25__tile_1], pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_66: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec] = pl._dump.call(
                        "tile.col_expand_mul", [q_lo_inline44__tile_textract_1, sin_hi_inline27__tile_1], pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_67: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec] = pl._dump.call(
                        "tile.add", [t__tile_65, t__tile_66], pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec]
                    )
                    q_rotated_inline46__tile_1: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_105, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl._dump.call(
                        "tile.concat", [t__tile_64, t__tile_67], pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_68: pl.Tile[[5, 128], pl.FP32, pl.MemRef(mem_vec_105, pl.const(0, pl.INDEX), 2560), pl.Mem.Vec] = pl._dump.call(
                        "tile.slice", [q_rotated_inline46__tile_1, [5, 128], [0, 0]], pl.Tile[[5, 128], pl.FP32, pl.Mem.Vec]
                    )
                    t__tile_69: pl.Tile[[5, 128], pl.BF16, pl.MemRef(mem_vec_105, pl.const(0, pl.INDEX), 1280), pl.Mem.Vec] = pl._dump.call(
                        "tile.cast", [t__tile_68], pl.Tile[[5, 128], pl.BF16, pl.Mem.Vec], kwargs={"target_type": pl.BF16, "mode": 2}
                    )
                    key_cache_bsnd_inline4__tile_1: pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16, pl.MemRef("mem_ddr_72", pl.const(0, pl.INT64), 0)] = pl._dump.call(
                        "tile.store",
                        [t__tile_53, [cache_row_inline37__ssa_v0_1, cache_col_inline28__ssa_v0_1], key_cache_bsnd_inline4__phi_v4],
                        pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16],
                    )
                    value_cache_bsnd_inline5__tile_1: pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16, pl.MemRef("mem_ddr_74", pl.const(0, pl.INT64), 0)] = pl._dump.call(
                        "tile.store",
                        [v_row_bf16_inline38__tile_1, [cache_row_inline37__ssa_v0_1, cache_col_inline28__ssa_v0_1], value_cache_bsnd_inline5__phi_v4],
                        pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16],
                    )
                    q2d_inline8__tile_1: pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16, pl.MemRef("mem_ddr_73", pl.const(0, pl.INT64), 0)] = pl._dump.call(
                        "tile.store", [t__tile_69, [q_row_inline47__ssa_v0_1, 0], q2d_inline8__phi_v4], pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16]
                    )
                    key_cache_bsnd_inline4__phi_v4_1, q2d_inline8__phi_v4_1, value_cache_bsnd_inline5__phi_v4_1 = pl._dump.yield_(
                        key_cache_bsnd_inline4__tile_1,
                        q2d_inline8__tile_1,
                        value_cache_bsnd_inline5__tile_1,
                        types=[
                            pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16, pl.MemRef("mem_ddr_126", pl.const(0, pl.INT64), 0)],
                            pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16, pl.MemRef("mem_ddr_127", pl.const(0, pl.INT64), 0)],
                            pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16, pl.MemRef("mem_ddr_128", pl.const(0, pl.INT64), 0)],
                        ],
                    )
                else:
                    key_cache_bsnd_inline4__phi_v4_1, q2d_inline8__phi_v4_1, value_cache_bsnd_inline5__phi_v4_1 = pl._dump.yield_(
                        key_cache_bsnd_inline4__phi_v4,
                        q2d_inline8__phi_v4,
                        value_cache_bsnd_inline5__phi_v4,
                        types=[
                            pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16, pl.MemRef("mem_ddr_126", pl.const(0, pl.INT64), 0)],
                            pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16, pl.MemRef("mem_ddr_127", pl.const(0, pl.INT64), 0)],
                            pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16, pl.MemRef("mem_ddr_128", pl.const(0, pl.INT64), 0)],
                        ],
                    )
                key_cache_bsnd_inline4__rv_v2, q2d_inline8__rv_v2, value_cache_bsnd_inline5__rv_v2 = pl._dump.yield_(
                    key_cache_bsnd_inline4__phi_v4_1,
                    q2d_inline8__phi_v4_1,
                    value_cache_bsnd_inline5__phi_v4_1,
                    types=[
                        pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16, pl.MemRef("mem_ddr_129", pl.const(0, pl.INT64), 0)],
                        pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16, pl.MemRef("mem_ddr_130", pl.const(0, pl.INT64), 0)],
                        pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16, pl.MemRef("mem_ddr_131", pl.const(0, pl.INT64), 0)],
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
            seq_len_inline50__tile: pl.Scalar[pl.INT32] = pl._dump.call("tensor.read", [seq_lens__ssa_v0, [batch_inline49__ssa_v0]], pl.Scalar[pl.INT32])
            stack_count_inline52__ssa_v0: pl.Scalar[pl.INDEX] = (pl.cast(seq_len_inline50__tile, pl.INDEX) + 511) // 512
            transfer_base_inline54__ssa_v0: pl.Scalar[pl.INDEX] = core_inline12__ssa_v0 * 48
            aiv_id_inline118__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
            pl._dump.call("system.set_ffts", [ffts_workspace__ssa_v0], pl._dump.Unknown)
            lane_row_inline119__ssa_v0: pl.Scalar[pl.INDEX] = aiv_id_inline118__ssa_v0 * 2
            lane_rows_inline120__ssa_v0: pl.Scalar[pl.INDEX] = aiv_id_inline118__ssa_v0 + 2
            tmp_inline121__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_132, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl._dump.call(
                "tile.create", [[8, 512]], pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec], kwargs={"dtype": pl.FP32, "target_memory": pl.Mem.Vec}
            )
            zero_scores_inline122__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl._dump.call(
                "tile.full", [[8, 512], 0.0], pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec], kwargs={"dtype": pl.FP32}
            )
            zero_scores_valid_inline123__ssa_v0: pl.Tile[
                [8, 512], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 512])
            ] = pl._dump.call(
                "tile.set_validshape",
                [zero_scores_inline122__ssa_v0, lane_rows_inline120__ssa_v0, 512],
                pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 512])],
            )
            t__tmp_v76: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_149, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = pl._dump.call(
                "tile.row_max", [zero_scores_valid_inline123__ssa_v0, tmp_inline121__ssa_v0], pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])]
            )
            zrow_inline124__rm_a0_tmp_v16: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_149, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = (
                pl._dump.call("tile.reshape", [t__tmp_v76, [1, 8]], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])])
            )
            zrow_inline124__row_major_tmp_v17: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = (
                pl._dump.call("tile.muls", [zrow_inline124__rm_a0_tmp_v16, 0.0], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])])
            )
            zrow_inline124__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = pl._dump.call(
                "tile.reshape", [zrow_inline124__row_major_tmp_v17, [8, 1]], pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])]
            )
            m_inline125__rm_a0_tmp_v18: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = (
                pl._dump.call("tile.reshape", [zrow_inline124__ssa_v0, [1, 8]], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])])
            )
            m_inline125__row_major_tmp_v19: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_108, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = (
                pl._dump.call("tile.adds", [m_inline125__rm_a0_tmp_v18, -3.0000000000000001e38], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])])
            )
            m_inline125__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_108, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = pl._dump.call(
                "tile.reshape", [m_inline125__row_major_tmp_v19, [8, 1]], pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])]
            )
            l_sum_inline126__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = (
                zrow_inline124__ssa_v0
            )
            o_init_inline127__ssa_v0: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_105, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec] = pl._dump.call(
                "tile.full", [[8, 128], 0.0], pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec], kwargs={"dtype": pl.FP32}
            )
            o_inline128__ssa_v0: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_105, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 128])] = (
                pl._dump.call(
                    "tile.set_validshape",
                    [o_init_inline127__ssa_v0, lane_rows_inline120__ssa_v0, 128],
                    pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 128])],
                )
            )
            rescale_pending0_inline129__rm_a0_tmp_v20: pl.Tile[
                [1, 8], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])
            ] = pl._dump.call("tile.reshape", [zrow_inline124__ssa_v0, [1, 8]], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])])
            rescale_pending0_inline129__row_major_tmp_v21: pl.Tile[
                [1, 8], pl.FP32, pl.MemRef(mem_vec_19, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])
            ] = pl._dump.call("tile.adds", [rescale_pending0_inline129__rm_a0_tmp_v20, 1.0], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])])
            rescale_pending0_inline129__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_19, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = (
                pl._dump.call("tile.reshape", [rescale_pending0_inline129__row_major_tmp_v21, [8, 1]], pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])])
            )
            rescale_pending1_inline130__rm_a0_tmp_v22: pl.Tile[
                [1, 8], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])
            ] = pl._dump.call("tile.reshape", [zrow_inline124__ssa_v0, [1, 8]], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])])
            rescale_pending1_inline130__row_major_tmp_v23: pl.Tile[
                [1, 8], pl.FP32, pl.MemRef(mem_vec_65, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])
            ] = pl._dump.call("tile.adds", [rescale_pending1_inline130__rm_a0_tmp_v22, 1.0], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])])
            rescale_pending1_inline130__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_65, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = (
                pl._dump.call("tile.reshape", [rescale_pending1_inline130__row_major_tmp_v23, [8, 1]], pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])])
            )
            for tick_inline131__idx_v0, (m_iter, l_iter, o_iter, pending0_iter, pending1_iter) in pl.range(
                stack_count_inline52__ssa_v0 + 2,
                init_values=(m_inline125__ssa_v0, l_sum_inline126__ssa_v0, o_inline128__ssa_v0, rescale_pending0_inline129__ssa_v0, rescale_pending1_inline130__ssa_v0),
            ):
                if tick_inline131__idx_v0 < stack_count_inline52__ssa_v0:
                    produce_stack_inline58__ssa_v1: pl.Scalar[pl.INDEX] = tick_inline131__idx_v0
                    produce_row_inline60__ssa_v1: pl.Scalar[pl.INDEX] = transfer_base_inline54__ssa_v0 + produce_stack_inline58__ssa_v1 % 3 * 16
                    pl.system.sync_wait(0, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                    score_aiv_inline140__ssa_v0: pl.Tile[
                        [8, 512], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 512])
                    ] = pl._dump.call(
                        "tile.load",
                        [score_transfer__ssa_v0, [produce_row_inline60__ssa_v1 + lane_row_inline119__ssa_v0, 0], [8, 512], [lane_rows_inline120__ssa_v0, 512]],
                        pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 512])],
                        kwargs={"target_memory": pl.Mem.Vec},
                    )
                    score_scaled_inline141__ssa_v0: pl.Tile[
                        [8, 512], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 512])
                    ] = pl._dump.call(
                        "tile.muls", [score_aiv_inline140__ssa_v0, 0.088388347648318433], pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 512])]
                    )
                    valid_cols_inline142__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(pl.cast(seq_len_inline50__tile, pl.INDEX) - produce_stack_inline58__ssa_v1 * 512, 512)
                    score_valid_inline143__ssa_v0: pl.Tile[
                        [8, 512], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, valid_cols_inline142__ssa_v0])
                    ] = pl._dump.call(
                        "tile.set_validshape",
                        [score_scaled_inline141__ssa_v0, lane_rows_inline120__ssa_v0, valid_cols_inline142__ssa_v0],
                        pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, valid_cols_inline142__ssa_v0])],
                    )
                    score_filled_inline144__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl._dump.call(
                        "tile.fillpad", [score_valid_inline143__ssa_v0], pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)], kwargs={"pad_value": pl.PadValue.min}
                    )
                    score_masked_inline145__ssa_v0: pl.Tile[
                        [8, 512], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 512], pad=pl.PadValue.min)
                    ] = pl._dump.call(
                        "tile.set_validshape",
                        [score_filled_inline144__ssa_v0, lane_rows_inline120__ssa_v0, 512],
                        pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 512], pad=pl.PadValue.min)],
                    )
                    local_m_inline146__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_149, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = (
                        pl._dump.call(
                            "tile.row_max", [score_masked_inline145__ssa_v0, tmp_inline121__ssa_v0], pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])]
                        )
                    )
                    next_m_inline147__rm_a0_tmp_v24: pl.Tile[
                        [1, 8], pl.FP32, pl.MemRef(mem_vec_149, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])
                    ] = pl._dump.call("tile.reshape", [local_m_inline146__ssa_v0, [1, 8]], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])])
                    next_m_inline147__rm_a1_tmp_v25: pl.Tile[
                        [1, 8], pl.FP32, pl.MemRef(mem_vec_108, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])
                    ] = pl._dump.call("tile.reshape", [m_iter, [1, 8]], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])])
                    next_m_inline147__row_major_tmp_v26: pl.Tile[
                        [1, 8], pl.FP32, pl.MemRef(mem_vec_119, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])
                    ] = pl._dump.call(
                        "tile.maximum",
                        [next_m_inline147__rm_a0_tmp_v24, next_m_inline147__rm_a1_tmp_v25],
                        pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])],
                    )
                    next_m_inline147__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_119, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = (
                        pl._dump.call("tile.reshape", [next_m_inline147__row_major_tmp_v26, [8, 1]], pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])])
                    )
                    t__rm_a0_tmp_v27: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_108, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = (
                        pl._dump.call("tile.reshape", [m_iter, [1, 8]], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])])
                    )
                    t__rm_a1_tmp_v28: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_119, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = (
                        pl._dump.call("tile.reshape", [next_m_inline147__ssa_v0, [1, 8]], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])])
                    )
                    t__row_major_tmp_v29: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_149, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = (
                        pl._dump.call("tile.sub", [t__rm_a0_tmp_v27, t__rm_a1_tmp_v28], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])])
                    )
                    t__tmp_v77: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_149, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = pl._dump.call(
                        "tile.reshape", [t__row_major_tmp_v29, [8, 1]], pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])]
                    )
                    rescale_inline148__rm_a0_tmp_v30: pl.Tile[
                        [1, 8], pl.FP32, pl.MemRef(mem_vec_149, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])
                    ] = pl._dump.call("tile.reshape", [t__tmp_v77, [1, 8]], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])])
                    rescale_inline148__row_major_tmp_v31: pl.Tile[
                        [1, 8], pl.FP32, pl.MemRef(mem_vec_20, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])
                    ] = pl._dump.call("tile.exp", [rescale_inline148__rm_a0_tmp_v30], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])])
                    rescale_inline148__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_20, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = (
                        pl._dump.call("tile.reshape", [rescale_inline148__row_major_tmp_v31, [8, 1]], pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])])
                    )
                    t__tmp_v78: pl.Tile[
                        [8, 512], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 512], pad=pl.PadValue.min)
                    ] = pl._dump.call(
                        "tile.row_expand_sub",
                        [score_masked_inline145__ssa_v0, next_m_inline147__ssa_v0],
                        pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 512], pad=pl.PadValue.min)],
                    )
                    probability_aiv_inline149__ssa_v0: pl.Tile[
                        [8, 512], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 512], pad=pl.PadValue.min)
                    ] = pl._dump.call("tile.exp", [t__tmp_v78], pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 512], pad=pl.PadValue.min)])
                    probability_reduce_inline150__ssa_v0: pl.Tile[
                        [8, 512], pl.FP32, pl.MemRef(mem_vec_149, pl.const(0, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 512], pad=pl.PadValue.min)
                    ] = pl._dump.call(
                        "tile.move",
                        [probability_aiv_inline149__ssa_v0],
                        pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 512], pad=pl.PadValue.min)],
                        kwargs={"target_memory": pl.Mem.Vec, "slayout": pl.TileLayout.none_box},
                    )
                    t__rm_a0_tmp_v32: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_20, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = pl._dump.call(
                        "tile.reshape", [rescale_inline148__ssa_v0, [1, 8]], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])]
                    )
                    t__rm_a1_tmp_v33: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = (
                        pl._dump.call("tile.reshape", [l_iter, [1, 8]], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])])
                    )
                    t__row_major_tmp_v34: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_26, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = (
                        pl._dump.call("tile.mul", [t__rm_a0_tmp_v32, t__rm_a1_tmp_v33], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])])
                    )
                    t__tmp_v79: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_26, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = pl._dump.call(
                        "tile.reshape", [t__row_major_tmp_v34, [8, 1]], pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])]
                    )
                    t__tmp_v80: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_27, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = pl._dump.call(
                        "tile.row_sum", [probability_reduce_inline150__ssa_v0, tmp_inline121__ssa_v0], pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])]
                    )
                    next_l_inline151__rm_a0_tmp_v35: pl.Tile[
                        [1, 8], pl.FP32, pl.MemRef(mem_vec_26, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])
                    ] = pl._dump.call("tile.reshape", [t__tmp_v79, [1, 8]], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])])
                    next_l_inline151__rm_a1_tmp_v36: pl.Tile[
                        [1, 8], pl.FP32, pl.MemRef(mem_vec_27, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])
                    ] = pl._dump.call("tile.reshape", [t__tmp_v80, [1, 8]], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])])
                    next_l_inline151__row_major_tmp_v37: pl.Tile[
                        [1, 8], pl.FP32, pl.MemRef(mem_vec_26, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])
                    ] = pl._dump.call(
                        "tile.add", [next_l_inline151__rm_a0_tmp_v35, next_l_inline151__rm_a1_tmp_v36], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])]
                    )
                    next_l_inline151__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_26, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = (
                        pl._dump.call("tile.reshape", [next_l_inline151__row_major_tmp_v37, [8, 1]], pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])])
                    )
                    probability_bf16_inline152__ssa_v0: pl.Tile[
                        [8, 512], pl.BF16, pl.MemRef(mem_vec_133, pl.const(0, pl.INDEX), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 512], pad=pl.PadValue.min)
                    ] = pl._dump.call(
                        "tile.cast",
                        [probability_aiv_inline149__ssa_v0],
                        pl.Tile[[8, 512], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 512], pad=pl.PadValue.min)],
                        kwargs={"target_type": pl.BF16, "mode": 1},
                    )
                    probability_valid_inline153__ssa_v0: pl.Tile[
                        [8, 512], pl.BF16, pl.MemRef(mem_vec_133, pl.const(0, pl.INDEX), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 512], pad=pl.PadValue.min)
                    ] = pl._dump.call(
                        "tile.set_validshape",
                        [probability_bf16_inline152__ssa_v0, lane_rows_inline120__ssa_v0, 512],
                        pl.Tile[[8, 512], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 512], pad=pl.PadValue.min)],
                    )
                    probability_transfer__store: pl.Tensor[[1152, 512], pl.BF16, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 1179648)] = pl._dump.call(
                        "tile.store",
                        [probability_valid_inline153__ssa_v0, [produce_row_inline60__ssa_v1 + lane_row_inline119__ssa_v0, 0], probability_transfer__ssa_v0],
                        pl.Tensor[[1152, 512], pl.BF16],
                    )
                    pl.system.sync_set(1, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
                    m_after_inline137__rv_v0, l_after_inline138__rv_v0, rescale_after_inline139__rv_v0 = pl._dump.yield_(
                        next_m_inline147__ssa_v0,
                        next_l_inline151__ssa_v0,
                        rescale_inline148__ssa_v0,
                        types=[
                            pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_119, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])],
                            pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_26, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])],
                            pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_20, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])],
                        ],
                    )
                else:
                    m_iter_mv: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_119, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = pl._dump.call(
                        "tile.move", [m_iter], pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])], kwargs={"target_memory": pl.Mem.Vec}
                    )
                    l_iter_mv: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_26, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = pl._dump.call(
                        "tile.move", [l_iter], pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])], kwargs={"target_memory": pl.Mem.Vec}
                    )
                    pending1_iter_mv: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_20, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = pl._dump.call(
                        "tile.move", [pending1_iter], pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])], kwargs={"target_memory": pl.Mem.Vec}
                    )
                    m_after_inline137__rv_v0, l_after_inline138__rv_v0, rescale_after_inline139__rv_v0 = pl._dump.yield_(
                        m_iter_mv,
                        l_iter_mv,
                        pending1_iter_mv,
                        types=[
                            pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_119, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])],
                            pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_26, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])],
                            pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_20, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])],
                        ],
                    )
                if 2 <= tick_inline131__idx_v0:
                    consume_stack_inline74__ssa_v1: pl.Scalar[pl.INDEX] = tick_inline131__idx_v0 - 2
                    consume_row_inline76__ssa_v1: pl.Scalar[pl.INDEX] = transfer_base_inline54__ssa_v0 + consume_stack_inline74__ssa_v1 % 3 * 16
                    pl.system.sync_wait(2, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                    pv_aiv_inline155__ssa_v0: pl.Tile[
                        [8, 128], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 128])
                    ] = pl._dump.call(
                        "tile.load",
                        [pv_transfer__ssa_v0, [consume_row_inline76__ssa_v1 + lane_row_inline119__ssa_v0, 0], [8, 128], [lane_rows_inline120__ssa_v0, 128]],
                        pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 128])],
                        kwargs={"target_memory": pl.Mem.Vec},
                    )
                    t__tmp_v81: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_149, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 128])] = (
                        pl._dump.call("tile.row_expand_mul", [o_iter, pending0_iter], pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 128])])
                    )
                    t__tmp_v82: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_105, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 128])] = (
                        pl._dump.call("tile.add", [t__tmp_v81, pv_aiv_inline155__ssa_v0], pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 128])])
                    )
                    o_after_inline154__rv_v0: pl.Tile[
                        [8, 128], pl.FP32, pl.MemRef(mem_vec_105, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 128])
                    ] = pl.yield_(t__tmp_v82)
                else:
                    o_after_inline154__rv_v0: pl.Tile[
                        [8, 128], pl.FP32, pl.MemRef(mem_vec_105, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 128])
                    ] = pl.yield_(o_iter)
                pending0_after_inline156__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_19, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = (
                    pl._dump.call("tile.move", [pending1_iter], pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])], kwargs={"target_memory": pl.Mem.Vec})
                )
                pending1_after_inline157__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_65, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = (
                    pl._dump.call(
                        "tile.move",
                        [rescale_after_inline139__rv_v0],
                        pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])],
                        kwargs={"target_memory": pl.Mem.Vec},
                    )
                )
                l_after_inline138__rv_v0_mv: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = (
                    pl._dump.call(
                        "tile.move", [l_after_inline138__rv_v0], pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])], kwargs={"target_memory": pl.Mem.Vec}
                    )
                )
                m_after_inline137__rv_v0_mv: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_108, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = (
                    pl._dump.call(
                        "tile.move", [m_after_inline137__rv_v0], pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])], kwargs={"target_memory": pl.Mem.Vec}
                    )
                )
                m_inline132, l_sum_inline133, o_inline134, rescale_pending0_inline135, rescale_pending1_inline136 = pl._dump.yield_(
                    m_after_inline137__rv_v0_mv,
                    l_after_inline138__rv_v0_mv,
                    o_after_inline154__rv_v0,
                    pending0_after_inline156__ssa_v0,
                    pending1_after_inline157__ssa_v0,
                    types=[
                        pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_108, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])],
                        pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])],
                        pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_105, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 128])],
                        pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_19, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])],
                        pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_65, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])],
                    ],
                )
            t__rm_a0_tmp_v38: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_107, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = pl._dump.call(
                "tile.reshape", [l_sum_inline133, [1, 8]], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])]
            )
            t__row_major_tmp_v39: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])] = pl._dump.call(
                "tile.recip", [t__rm_a0_tmp_v38], pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_rows_inline120__ssa_v0])]
            )
            t__tmp_v83: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_133, pl.const(0, pl.INDEX), 32), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])] = pl._dump.call(
                "tile.reshape", [t__row_major_tmp_v39, [8, 1]], pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 1])]
            )
            t__tmp_v84: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_132, pl.const(0, pl.INDEX), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 128])] = pl._dump.call(
                "tile.row_expand_mul", [o_inline134, t__tmp_v83], pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 128])]
            )
            ctx_inline158__ssa_v0: pl.Tile[[8, 128], pl.BF16, pl.MemRef(mem_vec_132, pl.const(0, pl.INDEX), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 128])] = (
                pl._dump.call(
                    "tile.cast", [t__tmp_v84], pl.Tile[[8, 128], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows_inline120__ssa_v0, 128])], kwargs={"target_type": pl.BF16, "mode": 1}
                )
            )
            out2d_inline10__store: pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)] = pl._dump.call(
                "tile.store",
                [ctx_inline158__ssa_v0, [batch_inline49__ssa_v0 * 40 + kv_head_inline19__ssa_v1 * 5 + lane_row_inline119__ssa_v0, 0], out2d_inline10__ssa_v0],
                pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16],
            )
        return score_transfer__ssa_v0, probability_transfer__ssa_v0, pv_transfer__ssa_v0, out2d_inline10__ssa_v0

    @pl.function(type=pl.FunctionType.Group, level=pl.Level.CORE_GROUP, role=pl.Role.SubWorker)
    def attn_swpipe(
        ffts_workspace__ssa_v0: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 2048)],
        key_cache_bsnd_inline4__ssa_v0: pl.InOut[pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        q2d_inline8__ssa_v0: pl.InOut[pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]],
        value_cache_bsnd_inline5__ssa_v0: pl.InOut[pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        active_batch_inline0__ssa_v0: pl.Scalar[pl.INDEX],
        seq_lens__ssa_v0: pl.Tensor[[2], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 8)],
        inv_rms_states__ssa_v0: pl.Tensor[[2, 1], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 8)],
        slot_mapping__ssa_v0: pl.Tensor[[2], pl.INT32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 8)],
        rope_cos__ssa_v0: pl.Tensor[[4096, 128], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 2097152)],
        rope_sin__ssa_v0: pl.Tensor[[4096, 128], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 2097152)],
        k_proj__ssa_v0: pl.Tensor[[2, 1024], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 8192)],
        k_norm_w__ssa_v0: pl.Tensor[[1, 128], pl.FP32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 512)],
        layer_cache_base_token_rows__ssa_v0: pl.Scalar[pl.INDEX],
        v_proj__ssa_v0: pl.Tensor[[2, 1024], pl.FP32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 8192)],
        q_proj__ssa_v0: pl.Tensor[[2, 5120], pl.FP32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 40960)],
        q_norm_w__ssa_v0: pl.Tensor[[1, 128], pl.FP32, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 512)],
        num_tasks_inline1__ssa_v0: pl.Scalar[pl.INDEX],
        block_table_2d_inline7__ssa_v0: pl.Tensor[[active_batch_inline0__ssa_v0, max_blocks_per_seq_inline6__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)],
        cache_base_inline9__ssa_v0: pl.Scalar[pl.INDEX],
        score_transfer__ssa_v0: pl.InOut[pl.Tensor[[1152, 512], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 2359296)]],
        probability_transfer__ssa_v0: pl.InOut[pl.Tensor[[1152, 512], pl.BF16, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 1179648)]],
        pv_transfer__ssa_v0: pl.InOut[pl.Tensor[[1152, 128], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 589824)]],
        out2d_inline10__ssa_v0: pl.Out[pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[pl.Tensor[[1152, 512], pl.FP32], pl.Tensor[[1152, 512], pl.BF16], pl.Tensor[[1152, 128], pl.FP32], pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16]]:
        pl.func_attr({"mx_tensor_views_blocked": True, "split_aiv": True})
        pl._dump.call(
            self.attn_swpipe_aic,
            [
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
            ],
            pl._dump.Unknown,
        )
        pl._dump.call(
            self.attn_swpipe_aiv,
            [
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
            ],
            pl.Tuple[pl.Tensor[[1152, 512], pl.FP32], pl.Tensor[[1152, 512], pl.BF16], pl.Tensor[[1152, 128], pl.FP32], pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16]],
        )
        return score_transfer__ssa_v0, probability_transfer__ssa_v0, pv_transfer__ssa_v0, out2d_inline10__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def attn_swpipe_spmd(
        self,
        ffts_workspace__ssa_v0: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 2048)],
        key_cache_bsnd_inline4__ssa_v0: pl.InOut[pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        q2d_inline8__ssa_v0: pl.InOut[pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]],
        value_cache_bsnd_inline5__ssa_v0: pl.InOut[pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        active_batch_inline0__ssa_v0: pl.Scalar[pl.INDEX],
        seq_lens__ssa_v0: pl.Tensor[[2], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 8)],
        inv_rms_states__ssa_v0: pl.Tensor[[2, 1], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 8)],
        slot_mapping__ssa_v0: pl.Tensor[[2], pl.INT32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 8)],
        rope_cos__ssa_v0: pl.Tensor[[4096, 128], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 2097152)],
        rope_sin__ssa_v0: pl.Tensor[[4096, 128], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 2097152)],
        k_proj__ssa_v0: pl.Tensor[[2, 1024], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 8192)],
        k_norm_w__ssa_v0: pl.Tensor[[1, 128], pl.FP32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 512)],
        layer_cache_base_token_rows__ssa_v0: pl.Scalar[pl.INDEX],
        v_proj__ssa_v0: pl.Tensor[[2, 1024], pl.FP32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 8192)],
        q_proj__ssa_v0: pl.Tensor[[2, 5120], pl.FP32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 40960)],
        q_norm_w__ssa_v0: pl.Tensor[[1, 128], pl.FP32, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 512)],
        num_tasks_inline1__ssa_v0: pl.Scalar[pl.INDEX],
        block_table_2d_inline7__ssa_v0: pl.Tensor[[active_batch_inline0__ssa_v0, max_blocks_per_seq_inline6__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)],
        cache_base_inline9__ssa_v0: pl.Scalar[pl.INDEX],
        score_transfer__ssa_v0: pl.InOut[pl.Tensor[[1152, 512], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 2359296)]],
        probability_transfer__ssa_v0: pl.InOut[pl.Tensor[[1152, 512], pl.BF16, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 1179648)]],
        pv_transfer__ssa_v0: pl.InOut[pl.Tensor[[1152, 128], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 589824)]],
        out2d_inline10__ssa_v0: pl.Out[pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)]],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[pl.Tensor[[1152, 512], pl.FP32], pl.Tensor[[1152, 512], pl.BF16], pl.Tensor[[1152, 128], pl.FP32], pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16]] = pl._dump.call(
            self.attn_swpipe,
            [
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
            ],
            pl.Tuple[pl.Tensor[[1152, 512], pl.FP32], pl.Tensor[[1152, 512], pl.BF16], pl.Tensor[[1152, 128], pl.FP32], pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16]],
        )
        score_transfer__ssa_v1: pl.Tensor[[1152, 512], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 2359296)] = ret__tmp_v0[0]
        probability_transfer__ssa_v1: pl.Tensor[[1152, 512], pl.BF16, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 1179648)] = ret__tmp_v0[1]
        pv_transfer__ssa_v1: pl.Tensor[[1152, 128], pl.FP32, pl.MemRef("mem_ddr_21", pl.const(0, pl.INT64), 589824)] = ret__tmp_v0[2]
        out2d_inline10__ssa_v1: pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16, pl.MemRef("mem_ddr_22", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[3]

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def paged_attention_pypto_dynamic(
        self,
        key_cache__ssa_v0: pl.InOut[pl.Tensor[[10368, 1024], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 21233664)]],
        value_cache__ssa_v0: pl.InOut[pl.Tensor[[10368, 1024], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 21233664)]],
        block_table__ssa_v0: pl.Tensor[[64], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 256)],
        seq_lens__ssa_v0: pl.Tensor[[2], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 8)],
        inv_rms_states__ssa_v0: pl.Tensor[[2, 1], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 8)],
        slot_mapping__ssa_v0: pl.Tensor[[2], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 8)],
        rope_cos__ssa_v0: pl.Tensor[[4096, 128], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 2097152)],
        rope_sin__ssa_v0: pl.Tensor[[4096, 128], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 2097152)],
        q_proj__ssa_v0: pl.Tensor[[2, 5120], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 40960)],
        k_proj__ssa_v0: pl.Tensor[[2, 1024], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 8192)],
        v_proj__ssa_v0: pl.Tensor[[2, 1024], pl.FP32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 8192)],
        q_norm_w__ssa_v0: pl.Tensor[[1, 128], pl.FP32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 512)],
        k_norm_w__ssa_v0: pl.Tensor[[1, 128], pl.FP32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 512)],
        layer_cache_base_token_rows__ssa_v0: pl.Scalar[pl.INDEX],
        out__ssa_v0: pl.Out[pl.Tensor[[2, 40, 128], pl.BF16, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 20480)]],
    ) -> pl.Tensor[[2, 40, 128], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        active_batch__ssa_v0: pl.Scalar[pl.INDEX] = pl._dump.call("tensor.dim", [seq_lens__ssa_v0, 0], pl.Scalar[pl.INDEX])
        q_tnd_flat__ssa_v0: pl.Tensor[[active_batch__ssa_v0 * pl.const(40, pl.INDEX), 128], pl.BF16, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)] = pl._dump.call(
            "tensor.create", [[active_batch__ssa_v0 * 40, 128]], pl.Tensor[[active_batch__ssa_v0 * pl.const(40, pl.INDEX), 128], pl.BF16], kwargs={"dtype": pl.BF16, "layout": pl.TensorLayout.ND}
        )
        score_transfer__ssa_v0: pl.Tensor[[1152, 512], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 2359296)] = pl._dump.call(
            "tensor.create", [[1152, 512]], pl.Tensor[[1152, 512], pl.FP32], kwargs={"dtype": pl.FP32, "layout": pl.TensorLayout.ND}
        )
        probability_transfer__ssa_v0: pl.Tensor[[1152, 512], pl.BF16, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 1179648)] = pl._dump.call(
            "tensor.create", [[1152, 512]], pl.Tensor[[1152, 512], pl.BF16], kwargs={"dtype": pl.BF16, "layout": pl.TensorLayout.ND}
        )
        pv_transfer__ssa_v0: pl.Tensor[[1152, 128], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 589824)] = pl._dump.call(
            "tensor.create", [[1152, 128]], pl.Tensor[[1152, 128], pl.FP32], kwargs={"dtype": pl.FP32, "layout": pl.TensorLayout.ND}
        )
        ffts_workspace__ssa_v0: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 2048)] = pl._dump.call(
            "tensor.create", [[256]], pl.Tensor[[256], pl.INT64], kwargs={"dtype": pl.INT64, "layout": pl.TensorLayout.ND}
        )
        q_proj_tid__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[])
        k_proj_tid__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[])
        v_proj_tid__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[])
        rms_tid__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[])
        attn_out_seed_tid__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[])
        mlp_out_seed_tid__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[])
        scratch_ready_tid__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[])
        active_batch_inline0__ssa_v0: pl.Scalar[pl.INDEX] = pl._dump.call("tensor.dim", [seq_lens__ssa_v0, 0], pl.Scalar[pl.INDEX])
        num_tasks_inline1__ssa_v0: pl.Scalar[pl.INDEX] = active_batch_inline0__ssa_v0 * 8
        query_rows_inline2__ssa_v0: pl.Scalar[pl.INDEX] = pl._dump.call("tensor.dim", [q_tnd_flat__ssa_v0, 0], pl.Scalar[pl.INDEX])
        t__tmp_v0: pl.Scalar[pl.INDEX] = pl._dump.call("tensor.dim", [key_cache__ssa_v0, 0], pl.Scalar[pl.INDEX])
        t__tmp_v1: pl.Scalar[pl.INDEX] = pl._dump.call("tensor.dim", [key_cache__ssa_v0, 1], pl.Scalar[pl.INDEX])
        cache_token_rows_inline3__ssa_v0: pl.Scalar[pl.INDEX] = t__tmp_v0 * t__tmp_v1 // 1024
        key_cache_bsnd_inline4__ssa_v0: pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 21233664)] = pl._dump.call(
            "tensor.reshape", [key_cache__ssa_v0, [cache_token_rows_inline3__ssa_v0, 1024]], pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16]
        )
        value_cache_bsnd_inline5__ssa_v0: pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 21233664)] = pl._dump.call(
            "tensor.reshape", [value_cache__ssa_v0, [cache_token_rows_inline3__ssa_v0, 1024]], pl.Tensor[[cache_token_rows_inline3__ssa_v0, 1024], pl.BF16]
        )
        t__tmp_v2: pl.Scalar[pl.INDEX] = pl._dump.call("tensor.dim", [block_table__ssa_v0, 0], pl.Scalar[pl.INDEX])
        max_blocks_per_seq_inline6__ssa_v0: pl.Scalar[pl.INDEX] = t__tmp_v2 // active_batch_inline0__ssa_v0
        block_table_2d_inline7__ssa_v0: pl.Tensor[[active_batch_inline0__ssa_v0, max_blocks_per_seq_inline6__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 256)] = pl._dump.call(
            "tensor.reshape",
            [block_table__ssa_v0, [active_batch_inline0__ssa_v0, max_blocks_per_seq_inline6__ssa_v0]],
            pl.Tensor[[active_batch_inline0__ssa_v0, max_blocks_per_seq_inline6__ssa_v0], pl.INT32],
        )
        q2d_inline8__ssa_v0: pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)] = pl._dump.call(
            "tensor.reshape", [q_tnd_flat__ssa_v0, [query_rows_inline2__ssa_v0, 128]], pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16]
        )
        cache_base_inline9__ssa_v0: pl.Scalar[pl.INDEX] = layer_cache_base_token_rows__ssa_v0
        out2d_inline10__ssa_v0: pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 20480)] = pl._dump.call(
            "tensor.reshape", [out__ssa_v0, [query_rows_inline2__ssa_v0, 128]], pl.Tensor[[query_rows_inline2__ssa_v0, 128], pl.BF16]
        )
        ret__tmp_v0: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl._dump.submit(
            pl.spmd_submit(
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
            ),
            pl.Tuple[pl.Scalar[pl.TASK_ID]],
        )
        attn_tid_inline11__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0[0]
        return out__ssa_v0
