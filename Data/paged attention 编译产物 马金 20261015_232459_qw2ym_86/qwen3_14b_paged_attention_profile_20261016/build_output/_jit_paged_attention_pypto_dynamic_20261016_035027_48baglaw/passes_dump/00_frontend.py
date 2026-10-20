# pypto.program: _jit_paged_attention_pypto_dynamic
import pypto.language as pl


@pl.program
class _jit_paged_attention_pypto_dynamic:
    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def paged_attention_pypto_swpipe(
        self,
        q_tnd_flat: pl.Tensor[[80, 128], pl.BF16],
        key_cache: pl.Tensor[[10368, 1024], pl.BF16],
        value_cache: pl.Tensor[[10368, 1024], pl.BF16],
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
        out: pl.Tensor[[2, 40, 128], pl.BF16],
        score_transfer: pl.Tensor[[1152, 512], pl.FP32],
        probability_transfer: pl.Tensor[[1152, 512], pl.BF16],
        pv_transfer: pl.Tensor[[1152, 128], pl.FP32],
        ffts_workspace: pl.Tensor[[256], pl.INT64],
        q_proj_tid: pl.Scalar[pl.TASK_ID],
        k_proj_tid: pl.Scalar[pl.TASK_ID],
        v_proj_tid: pl.Scalar[pl.TASK_ID],
        rms_tid: pl.Scalar[pl.TASK_ID],
        attn_out_seed_tid: pl.Scalar[pl.TASK_ID],
        mlp_out_seed_tid: pl.Scalar[pl.TASK_ID],
        scratch_ready_tid: pl.Scalar[pl.TASK_ID],
    ) -> pl.Scalar[pl.TASK_ID]:
        # Run fused Q/K norm, RoPE, cache append, and dynamic paged attention.
        active_batch: pl.Scalar[pl.INDEX] = pl.tensor.dim(seq_lens, 0)
        num_tasks: pl.Scalar[pl.INDEX] = active_batch * 8
        query_rows: pl.Scalar[pl.INDEX] = pl.tensor.dim(q_tnd_flat, 0)
        cache_token_rows: pl.Scalar[pl.INDEX] = pl.tensor.dim(key_cache, 0) * pl.tensor.dim(key_cache, 1) // 1024
        key_cache_bsnd: pl.Tensor[[cache_token_rows, 1024], pl.BF16] = pl.tensor.reshape(key_cache, [cache_token_rows, 1024])
        value_cache_bsnd: pl.Tensor[[cache_token_rows, 1024], pl.BF16] = pl.tensor.reshape(value_cache, [cache_token_rows, 1024])
        max_blocks_per_seq: pl.Scalar[pl.INDEX] = pl.tensor.dim(block_table, 0) // active_batch
        block_table_2d: pl.Tensor[[active_batch, max_blocks_per_seq], pl.INT32] = pl.tensor.reshape(block_table, [active_batch, max_blocks_per_seq])
        q2d: pl.Tensor[[query_rows, 128], pl.BF16] = pl.tensor.reshape(q_tnd_flat, [query_rows, 128])
        cache_base: pl.Scalar[pl.INDEX] = layer_cache_base_token_rows
        out2d: pl.Tensor[[query_rows, 128], pl.BF16] = pl.tensor.reshape(out, [query_rows, 128])
        with pl.spmd(
            24, sync_start=True, name_hint="attn_swpipe_spmd", deps=[q_proj_tid, k_proj_tid, v_proj_tid, rms_tid, attn_out_seed_tid, mlp_out_seed_tid, scratch_ready_tid], allow_early_resolve=True
        ) as attn_tid:
            core: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
            pl.system.set_ffts(ffts_workspace)
            for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                rope_core: pl.Scalar[pl.INDEX] = core * 2 + aiv_id
                if rope_core < 32:
                    q_red_pad: pl.Tensor[[1, 1408], pl.FP32] = pl.tensor.full([1, 1408], dtype=pl.FP32, value=0.0)
                    k_red_pad: pl.Tensor[[1, 896], pl.FP32] = pl.tensor.full([1, 896], dtype=pl.FP32, value=0.0)
                    for it in pl.pipeline(4, stage=2):
                        g_idx: pl.Scalar[pl.INDEX] = rope_core + it * 32
                        if g_idx < 8 * active_batch:
                            kv_head: pl.Scalar[pl.INDEX] = g_idx // active_batch
                            batch_idx: pl.Scalar[pl.INDEX] = g_idx - kv_head * active_batch
                            position: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(seq_lens, [batch_idx]), pl.INDEX) - 1
                            inv_rms: pl.Scalar[pl.FP32] = pl.tensor.read(inv_rms_states, [batch_idx, 0])
                            write_slot: pl.Scalar[pl.INDEX] = pl.max(pl.cast(pl.tensor.read(slot_mapping, [batch_idx]), pl.INDEX), 0)
                            cos_lo: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_cos, [1, 64], [position, 0])
                            cos_hi: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_cos, [1, 64], [position, 64])
                            sin_lo: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_sin, [1, 64], [position, 0])
                            sin_hi: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_sin, [1, 64], [position, 64])
                            cache_col: pl.Scalar[pl.INDEX] = kv_head * 128
                            k_raw: pl.Tensor[[8, 128], pl.FP32] = pl.tensor.muls(
                                pl.tensor.reshape(pl.tensor.concat(pl.tensor.slice(k_proj, [1, 128], [batch_idx, cache_col]), k_red_pad), [8, 128]), inv_rms
                            )
                            k_ss: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.row_sum(pl.tensor.mul(k_raw, k_raw))
                            k_inv: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(pl.tensor.adds(pl.tensor.muls(k_ss, 0.0078125), 9.9999999999999995e-07)))
                            k_normed: pl.Tensor[[8, 128], pl.FP32] = pl.tensor.row_expand_mul(pl.tensor.col_expand_mul(k_raw, k_norm_w), k_inv)
                            k_full: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.slice(k_normed, [1, 128], [0, 0])
                            k_lo: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(k_full, [1, 64], [0, 0])
                            k_hi: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(k_full, [1, 64], [0, 64])
                            k_rotated: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.concat(
                                pl.tensor.sub(pl.tensor.col_expand_mul(k_lo, cos_lo), pl.tensor.col_expand_mul(k_hi, sin_lo)),
                                pl.tensor.add(pl.tensor.col_expand_mul(k_hi, cos_hi), pl.tensor.col_expand_mul(k_lo, sin_hi)),
                            )
                            cache_row: pl.Scalar[pl.INDEX] = layer_cache_base_token_rows + write_slot
                            key_cache_bsnd: pl.Tensor[[cache_token_rows, 1024], pl.BF16] = pl.tensor.assemble(
                                key_cache_bsnd, pl.tensor.cast(k_rotated, target_type=pl.BF16, mode="round"), [cache_row, cache_col]
                            )
                            v_row_bf16: pl.Tensor[[1, 128], pl.BF16] = pl.tensor.cast(
                                pl.tensor.muls(pl.tensor.slice(v_proj, [1, 128], [batch_idx, cache_col]), inv_rms), target_type=pl.BF16, mode="round"
                            )
                            value_cache_bsnd: pl.Tensor[[cache_token_rows, 1024], pl.BF16] = pl.tensor.assemble(value_cache_bsnd, v_row_bf16, [cache_row, cache_col])
                            q_head_base: pl.Scalar[pl.INDEX] = kv_head * 5
                            q_raw: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.muls(
                                pl.tensor.reshape(pl.tensor.concat(pl.tensor.slice(q_proj, [1, 640], [batch_idx, q_head_base * 128]), q_red_pad), [16, 128]), inv_rms
                            )
                            q_ss: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.row_sum(pl.tensor.mul(q_raw, q_raw))
                            q_inv: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(pl.tensor.adds(pl.tensor.muls(q_ss, 0.0078125), 9.9999999999999995e-07)))
                            q_heads: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.row_expand_mul(pl.tensor.col_expand_mul(q_raw, q_norm_w), q_inv)
                            q_lo: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(q_heads, [16, 64], [0, 0])
                            q_hi: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(q_heads, [16, 64], [0, 64])
                            q_rotated: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.concat(
                                pl.tensor.sub(pl.tensor.col_expand_mul(q_lo, cos_lo), pl.tensor.col_expand_mul(q_hi, sin_lo)),
                                pl.tensor.add(pl.tensor.col_expand_mul(q_hi, cos_hi), pl.tensor.col_expand_mul(q_lo, sin_hi)),
                            )
                            q_row: pl.Scalar[pl.INDEX] = batch_idx * 40 + q_head_base
                            q2d: pl.Tensor[[query_rows, 128], pl.BF16] = pl.tensor.assemble(
                                q2d, pl.tensor.cast(pl.tensor.slice(q_rotated, [5, 128], [0, 0]), target_type=pl.BF16, mode="round"), [q_row, 0]
                            )
            pl.system.cacheinvalid()
            pl.system.fence()
            pl.system.syncall(core_type=pl.KernelType.MIX)
            pl.system.cacheinvalid()
            for task in pl.range(core, num_tasks, 24):
                batch: pl.Scalar[pl.INDEX] = task // 8
                kv_head: pl.Scalar[pl.INDEX] = task % 8
                seq_len: pl.Scalar[pl.INT32] = pl.tensor.read(seq_lens, [batch])
                page_count: pl.Scalar[pl.INDEX] = (pl.cast(seq_len, pl.INDEX) + 128 - 1) // 128
                stack_count: pl.Scalar[pl.INDEX] = (pl.cast(seq_len, pl.INDEX) + 512 - 1) // 512
                col: pl.Scalar[pl.INDEX] = kv_head * 128
                transfer_base: pl.Scalar[pl.INDEX] = core * 3 * 16
                qp_row: pl.Scalar[pl.INDEX] = batch * 40 + kv_head * 5
                q_tile: pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl.tile.load(q2d, [qp_row, 0], [16, 128], [5, 128], target_memory=pl.Mem.Mat)
                for tick in pl.range(stack_count + 2):
                    if tick < stack_count:
                        produce_stack: pl.Scalar[pl.INDEX] = tick
                        produce_page: pl.Scalar[pl.INDEX] = produce_stack * 4
                        produce_row: pl.Scalar[pl.INDEX] = transfer_base + produce_stack % 3 * 16
                        if produce_page + 4 <= page_count:
                            for qk_page_offset in pl.pipeline(4, stage=2):
                                qk_ph: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(block_table_2d, [batch, produce_page + qk_page_offset]), pl.INDEX)
                                qk_k_page: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(key_cache_bsnd, [cache_base + qk_ph * 128, col], [128, 128], [128, 128], target_memory=pl.Mem.Mat)
                                qk_score_page: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul(
                                    q_tile, pl.tile.transpose_view(qk_k_page)
                                )
                                pl.tile.store(qk_score_page, [produce_row, qk_page_offset * 128], score_transfer)
                        else:
                            qk_tail_ph0: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(block_table_2d, [batch, produce_page]), pl.INDEX)
                            k_tail0: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(key_cache_bsnd, [cache_base + qk_tail_ph0 * 128, col], [128, 128], [128, 128], target_memory=pl.Mem.Mat)
                            qk_tail0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul(
                                q_tile, pl.tile.transpose_view(k_tail0)
                            )
                            pl.tile.store(qk_tail0, [produce_row, 0], score_transfer)
                            if produce_page + 1 < page_count:
                                qk_tail_ph1: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(block_table_2d, [batch, produce_page + 1]), pl.INDEX)
                                k_tail1: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                                    key_cache_bsnd, [cache_base + qk_tail_ph1 * 128, col], [128, 128], [128, 128], target_memory=pl.Mem.Mat
                                )
                                qk_tail1: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul(
                                    q_tile, pl.tile.transpose_view(k_tail1)
                                )
                                pl.tile.store(qk_tail1, [produce_row, 128], score_transfer)
                            if produce_page + 2 < page_count:
                                qk_tail_ph2: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(block_table_2d, [batch, produce_page + 2]), pl.INDEX)
                                k_tail2: pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                                    key_cache_bsnd, [cache_base + qk_tail_ph2 * 128, col], [128, 128], [128, 128], target_memory=pl.Mem.Mat
                                )
                                qk_tail2: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul(
                                    q_tile, pl.tile.transpose_view(k_tail2)
                                )
                                pl.tile.store(qk_tail2, [produce_row, 256], score_transfer)
                        pl.system.sync_set(0, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                    if tick >= 2:
                        consume_stack: pl.Scalar[pl.INDEX] = tick - 2
                        consume_page: pl.Scalar[pl.INDEX] = consume_stack * 4
                        consume_row: pl.Scalar[pl.INDEX] = transfer_base + consume_stack % 3 * 16
                        if consume_page + 4 <= page_count:
                            pv_ph0: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(block_table_2d, [batch, consume_page]), pl.INDEX)
                            pv_ph1: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(block_table_2d, [batch, consume_page + 1]), pl.INDEX)
                            pv_ph2: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(block_table_2d, [batch, consume_page + 2]), pl.INDEX)
                            pv_ph3: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(block_table_2d, [batch, consume_page + 3]), pl.INDEX)
                            v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[0], pl.Mem.Mat] = pl._dump.call(
                                "tile.load",
                                [value_cache_bsnd, [cache_base + pv_ph0 * 128, col], [128, 128], [128, 128]],
                                pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            v1: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[1], pl.Mem.Mat] = pl._dump.call(
                                "tile.load",
                                [value_cache_bsnd, [cache_base + pv_ph1 * 128, col], [128, 128], [128, 128]],
                                pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            v2: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[2], pl.Mem.Mat] = pl._dump.call(
                                "tile.load",
                                [value_cache_bsnd, [cache_base + pv_ph2 * 128, col], [128, 128], [128, 128]],
                                pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            v3: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[3], pl.Mem.Mat] = pl._dump.call(
                                "tile.load",
                                [value_cache_bsnd, [cache_base + pv_ph3 * 128, col], [128, 128], [128, 128]],
                                pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                            probability0: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                "tile.load",
                                [probability_transfer, [consume_row, 0], [16, 128], [5, 128]],
                                pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            pv0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul(
                                pl._dump.call("tile.move", [probability0], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}),
                                pl._dump.call("tile.move", [v0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}),
                            )
                            probability1: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[1], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                "tile.load",
                                [probability_transfer, [consume_row, 128], [16, 128], [5, 128]],
                                pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            pv1: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                                pv0,
                                pl._dump.call("tile.move", [probability1], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}),
                                pl._dump.call("tile.move", [v1], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}),
                            )
                            probability2: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                "tile.load",
                                [probability_transfer, [consume_row, 256], [16, 128], [5, 128]],
                                pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            pv2: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                                pv1,
                                pl._dump.call("tile.move", [probability2], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}),
                                pl._dump.call("tile.move", [v2], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}),
                            )
                            probability3: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[1], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                "tile.load",
                                [probability_transfer, [consume_row, 384], [16, 128], [5, 128]],
                                pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                kwargs={"target_memory": pl.Mem.Mat},
                            )
                            pv3: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                                pv2,
                                pl._dump.call("tile.move", [probability3], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}),
                                pl._dump.call("tile.move", [v3], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}),
                            )
                            pl.tile.store(pv3, [consume_row, 0], pv_transfer)
                        else:
                            tail_pages: pl.Scalar[pl.INDEX] = page_count - consume_page
                            if tail_pages == 1:
                                pv1_ph0: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(block_table_2d, [batch, consume_page]), pl.INDEX)
                                v1_0: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[0], pl.Mem.Mat] = pl._dump.call(
                                    "tile.load",
                                    [value_cache_bsnd, [cache_base + pv1_ph0 * 128, col], [128, 128], [128, 128]],
                                    pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                                probability1_0: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                    "tile.load",
                                    [probability_transfer, [consume_row, 0], [16, 128], [5, 128]],
                                    pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                pv_tail1: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul(
                                    pl._dump.call("tile.move", [probability1_0], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}),
                                    pl._dump.call("tile.move", [v1_0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}),
                                )
                                pl.tile.store(pv_tail1, [consume_row, 0], pv_transfer)
                            if tail_pages == 2:
                                pv2_ph0: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(block_table_2d, [batch, consume_page]), pl.INDEX)
                                pv2_ph1: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(block_table_2d, [batch, consume_page + 1]), pl.INDEX)
                                v2_0: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[0], pl.Mem.Mat] = pl._dump.call(
                                    "tile.load",
                                    [value_cache_bsnd, [cache_base + pv2_ph0 * 128, col], [128, 128], [128, 128]],
                                    pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                v2_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[1], pl.Mem.Mat] = pl._dump.call(
                                    "tile.load",
                                    [value_cache_bsnd, [cache_base + pv2_ph1 * 128, col], [128, 128], [128, 128]],
                                    pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                                probability2_0: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                    "tile.load",
                                    [probability_transfer, [consume_row, 0], [16, 128], [5, 128]],
                                    pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                pv_tail2_0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul(
                                    pl._dump.call("tile.move", [probability2_0], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}),
                                    pl._dump.call("tile.move", [v2_0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}),
                                )
                                probability2_1: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[1], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                    "tile.load",
                                    [probability_transfer, [consume_row, 128], [16, 128], [5, 128]],
                                    pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                pv_tail2_1: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                                    pv_tail2_0,
                                    pl._dump.call("tile.move", [probability2_1], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}),
                                    pl._dump.call("tile.move", [v2_1], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}),
                                )
                                pl.tile.store(pv_tail2_1, [consume_row, 0], pv_transfer)
                            if tail_pages == 3:
                                pv3_ph0: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(block_table_2d, [batch, consume_page]), pl.INDEX)
                                pv3_ph1: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(block_table_2d, [batch, consume_page + 1]), pl.INDEX)
                                pv3_ph2: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(block_table_2d, [batch, consume_page + 2]), pl.INDEX)
                                v3_0: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[0], pl.Mem.Mat] = pl._dump.call(
                                    "tile.load",
                                    [value_cache_bsnd, [cache_base + pv3_ph0 * 128, col], [128, 128], [128, 128]],
                                    pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                v3_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[1], pl.Mem.Mat] = pl._dump.call(
                                    "tile.load",
                                    [value_cache_bsnd, [cache_base + pv3_ph1 * 128, col], [128, 128], [128, 128]],
                                    pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                v3_2: pl.Tile[[128, 128], pl.BF16, pl.MemRef("pv_v_l1", slots=4)[2], pl.Mem.Mat] = pl._dump.call(
                                    "tile.load",
                                    [value_cache_bsnd, [cache_base + pv3_ph2 * 128, col], [128, 128], [128, 128]],
                                    pl.Tile[[128, 128], pl.BF16, pl.Mem.Mat],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                                probability3_0: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                    "tile.load",
                                    [probability_transfer, [consume_row, 0], [16, 128], [5, 128]],
                                    pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                pv_tail3_0: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul(
                                    pl._dump.call("tile.move", [probability3_0], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}),
                                    pl._dump.call("tile.move", [v3_0], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}),
                                )
                                probability3_1: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[1], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                    "tile.load",
                                    [probability_transfer, [consume_row, 128], [16, 128], [5, 128]],
                                    pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                pv_tail3_1: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                                    pv_tail3_0,
                                    pl._dump.call("tile.move", [probability3_1], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}),
                                    pl._dump.call("tile.move", [v3_1], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}),
                                )
                                probability3_2: pl.Tile[[16, 128], pl.BF16, pl.MemRef("pv_p_l1", slots=2)[0], pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])] = pl._dump.call(
                                    "tile.load",
                                    [probability_transfer, [consume_row, 256], [16, 128], [5, 128]],
                                    pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[5, 128])],
                                    kwargs={"target_memory": pl.Mem.Mat},
                                )
                                pv_tail3_2: pl.Tile[[16, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[5, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                                    pv_tail3_1,
                                    pl._dump.call("tile.move", [probability3_2], pl.Tile[[16, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[5, 128])], kwargs={"target_memory": pl.Mem.Left}),
                                    pl._dump.call("tile.move", [v3_2], pl.Tile[[128, 128], pl.BF16, pl.Mem.Right], kwargs={"target_memory": pl.Mem.Right}),
                                )
                                pl.tile.store(pv_tail3_2, [consume_row, 0], pv_transfer)
                        pl.system.sync_set(2, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                for aiv_id_1 in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    pl.system.set_ffts(ffts_workspace)
                    lane_row: pl.Scalar[pl.INDEX] = aiv_id_1 * 2
                    lane_rows: pl.Scalar[pl.INDEX] = 2 + aiv_id_1
                    tmp: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    zero_scores: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.full([8, 512], dtype=pl.FP32, value=pl.const(0.0, pl.FP32))
                    zero_scores_valid: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 512])] = pl.tile.set_validshape(zero_scores, lane_rows, 512)
                    zrow: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 1])] = pl.tile.muls(pl.tile.row_max(zero_scores_valid, tmp), 0.0)
                    m: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 1])] = pl.tile.adds(zrow, -3.0000000000000001e38)
                    l_sum: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 1])] = zrow
                    o_init: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.full([8, 128], dtype=pl.FP32, value=pl.const(0.0, pl.FP32))
                    o: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 128])] = pl.tile.set_validshape(o_init, lane_rows, 128)
                    rescale_pending0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 1])] = pl.tile.adds(zrow, 1.0)
                    rescale_pending1: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 1])] = pl.tile.adds(zrow, 1.0)
                    for tick_1, (m_iter, l_iter, o_iter, pending0_iter, pending1_iter) in pl.range(stack_count + 2, init_values=(m, l_sum, o, rescale_pending0, rescale_pending1)):
                        if tick_1 < stack_count:
                            produce_stack: pl.Scalar[pl.INDEX] = tick_1
                            produce_row: pl.Scalar[pl.INDEX] = transfer_base + produce_stack % 3 * 16
                            pl.system.sync_wait(0, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                            score_aiv: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 512])] = pl.tile.load(
                                score_transfer, [produce_row + lane_row, 0], [8, 512], [lane_rows, 512], target_memory=pl.Mem.Vec
                            )
                            score_scaled: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 512])] = pl.tile.muls(score_aiv, 0.088388347648318433)
                            valid_cols: pl.Scalar[pl.INDEX] = pl.min(512, pl.cast(seq_len, pl.INDEX) - produce_stack * 512)
                            score_valid: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, valid_cols])] = pl.tile.set_validshape(score_scaled, lane_rows, valid_cols)
                            score_filled: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(score_valid, pad_value=pl.PadValue.min)
                            score_masked: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 512], pad=pl.PadValue.min)] = pl.tile.set_validshape(score_filled, lane_rows, 512)
                            local_m: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 1])] = pl.tile.row_max(score_masked, tmp)
                            next_m: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 1])] = pl.tile.maximum(local_m, m_iter)
                            rescale: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 1])] = pl.tile.exp(pl.tile.sub(m_iter, next_m))
                            probability_aiv: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 512], pad=pl.PadValue.min)] = pl.tile.exp(
                                pl.tile.row_expand_sub(score_masked, next_m)
                            )
                            probability_reduce: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 512], pad=pl.PadValue.min)] = pl.tile.move(
                                probability_aiv, target_memory=pl.Mem.Vec, slayout=pl.TileLayout.none_box
                            )
                            next_l: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 1])] = pl.tile.add(pl.tile.mul(rescale, l_iter), pl.tile.row_sum(probability_reduce, tmp))
                            probability_bf16: pl.Tile[[8, 512], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 512], pad=pl.PadValue.min)] = pl.tile.cast(
                                probability_aiv, target_type=pl.BF16, mode="rint"
                            )
                            probability_valid: pl.Tile[[8, 512], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 512], pad=pl.PadValue.min)] = pl.tile.set_validshape(
                                probability_bf16, lane_rows, 512
                            )
                            pl.tile.store(probability_valid, [produce_row + lane_row, 0], probability_transfer)
                            pl.system.sync_set(1, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
                            m_after, l_after, rescale_after = pl._dump.yield_(
                                next_m,
                                next_l,
                                rescale,
                                types=[
                                    pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 1])],
                                    pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 1])],
                                    pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 1])],
                                ],
                            )
                        else:
                            m_after, l_after, rescale_after = pl._dump.yield_(
                                m_iter,
                                l_iter,
                                pending1_iter,
                                types=[
                                    pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 1])],
                                    pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 1])],
                                    pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 1])],
                                ],
                            )
                        if tick_1 >= 2:
                            consume_stack: pl.Scalar[pl.INDEX] = tick_1 - 2
                            consume_row: pl.Scalar[pl.INDEX] = transfer_base + consume_stack % 3 * 16
                            pl.system.sync_wait(2, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                            pv_aiv: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 128])] = pl.tile.load(
                                pv_transfer, [consume_row + lane_row, 0], [8, 128], [lane_rows, 128], target_memory=pl.Mem.Vec
                            )
                            o_after: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 128])] = pl.yield_(pl.tile.add(pl.tile.row_expand_mul(o_iter, pending0_iter), pv_aiv))
                        else:
                            o_after: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 128])] = pl.yield_(o_iter)
                        pending0_after: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 1])] = pl.tile.move(pending1_iter, target_memory=pl.Mem.Vec)
                        pending1_after: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 1])] = pl.tile.move(rescale_after, target_memory=pl.Mem.Vec)
                        m_1, l_sum_1, o_1, rescale_pending0_1, rescale_pending1_1 = pl._dump.yield_(
                            m_after,
                            l_after,
                            o_after,
                            pending0_after,
                            pending1_after,
                            types=[
                                pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 1])],
                                pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 1])],
                                pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 128])],
                                pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 1])],
                                pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 1])],
                            ],
                        )
                    ctx: pl.Tile[[8, 128], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[lane_rows, 128])] = pl.tile.cast(
                        pl.tile.row_expand_mul(o_1, pl.tile.recip(l_sum_1)), target_type=pl.BF16, mode="rint"
                    )
                    pl.tile.store(ctx, [batch * 40 + kv_head * 5 + lane_row, 0], out2d)
        return attn_tid

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
        self.paged_attention_pypto_swpipe(
            q_tnd_flat,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            inv_rms_states,
            slot_mapping,
            rope_cos,
            rope_sin,
            q_proj,
            k_proj,
            v_proj,
            q_norm_w,
            k_norm_w,
            layer_cache_base_token_rows,
            out,
            score_transfer,
            probability_transfer,
            pv_transfer,
            ffts_workspace,
            q_proj_tid,
            k_proj_tid,
            v_proj_tid,
            rms_tid,
            attn_out_seed_tid,
            mlp_out_seed_tid,
            scratch_ready_tid,
        )
        return out
