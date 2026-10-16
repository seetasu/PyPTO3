    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def csa_merge_pack_publish(attention_grouped_inline1276__ssa_v0: pl.InOut[pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)]], pack_work_count_inline1228__ssa_v0: pl.Scalar[pl.INDEX], sparse_blk_mi_inline1234__ssa_v0: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], sparse_blk_li_inline1283__ssa_v0: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)], sparse_blk_oi_inline1233__ssa_v0: pl.Tensor[[t_blk_inline2373__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)], attn_sink__ssa_v0: pl.Tensor[[64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 256)], rope_cos_il_inline1232__ssa_v0: pl.Tensor[[256, 64], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 65536)], rope_sin_signed_inline1231__ssa_v0: pl.Tensor[[256, 64], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 65536)], rope_swap_idx_inline1230__ssa_v0: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 4096)], tp_rank__ssa_v0: pl.Scalar[pl.INT32], attention_window__ssa_v0: pl.Out[pld.DistributedTensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 16777216)]], group_base__ssa_v0: pl.Scalar[pl.INT32], attention_signal__ssa_v0: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 8)]], attention_window_ctx: pld.CommCtx, attention_signal_ctx: pld.CommCtx):
        mem_vec_15: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_17: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        mem_vec_18: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_19: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        mem_vec_20: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        mem_vec_23: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        mem_vec_27: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_52: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_53: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_56: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_57: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_66: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 65536)
        worker_inline1227__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for pack_work_inline1270__idx_v0, (attention_grouped_inline1276__iter_v1,) in pl.range(worker_inline1227__ssa_v0, pack_work_count_inline1228__ssa_v0, 48, init_values=(attention_grouped_inline1276__ssa_v0,)):
            token_block_inline1226__ssa_v0: pl.Scalar[pl.INDEX] = pack_work_inline1270__idx_v0 // 4
            m_h_idx_inline1303__ssa_v0: pl.Scalar[pl.INDEX] = pack_work_inline1270__idx_v0 - token_block_inline1226__ssa_v0 * 4
            m_t0_inline1278__ssa_v0: pl.Scalar[pl.INDEX] = token_block_inline1226__ssa_v0 * 8
            m_h0_inline1224__ssa_v0: pl.Scalar[pl.INDEX] = m_h_idx_inline1303__ssa_v0 * 16
            global_group0_inline1314__ssa_v0: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 // 8
            destination_rank_inline1223__ssa_v0: pl.Scalar[pl.INDEX] = global_group0_inline1314__ssa_v0 // 4
            local_group0_inline1222__ssa_v0: pl.Scalar[pl.INDEX] = global_group0_inline1314__ssa_v0 - destination_rank_inline1223__ssa_v0 * 4
            for m_dt_inline1287__idx_v0, (attention_grouped_inline1276__iter_v3,) in pl.range(8, init_values=(attention_grouped_inline1276__iter_v1,)):
                m_t_inline1290__ssa_v0: pl.Scalar[pl.INDEX] = m_t0_inline1278__ssa_v0 + m_dt_inline1287__idx_v0
                m_idx_inline1289__ssa_v0: pl.Scalar[pl.INDEX] = m_t_inline1290__ssa_v0 * 4 + m_h_idx_inline1303__ssa_v0
                m_blk_base_inline1306__ssa_v0: pl.Scalar[pl.INDEX] = m_idx_inline1289__ssa_v0 * 80
                m_mi_inline1221__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_52, pl.const(98560, pl.INT64), 64), pl.Mem.Vec] = pl.tile.load(sparse_blk_mi_inline1234__ssa_v0, [m_blk_base_inline1306__ssa_v0, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec)
                m_li_inline1302__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_53, pl.const(98816, pl.INT64), 64), pl.Mem.Vec] = pl.tile.load(sparse_blk_li_inline1283__ssa_v0, [m_blk_base_inline1306__ssa_v0, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec)
                m_oi_inline1220__tile: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_66, pl.const(99584, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(sparse_blk_oi_inline1233__ssa_v0, [m_blk_base_inline1306__ssa_v0, 0], [16, 512], [16, 512], target_memory=pl.Mem.Vec)
                for m_sb_inline1219__idx_v0, (m_li_inline1302__iter_v1, m_mi_inline1221__iter_v1, m_oi_inline1220__iter_v1) in pl.range(1, 5, 2, init_values=(m_li_inline1302__tile, m_mi_inline1221__tile, m_oi_inline1220__tile)):
                    m_row_inline1218__ssa_v0: pl.Scalar[pl.INDEX] = m_blk_base_inline1306__ssa_v0 + m_sb_inline1219__idx_v0 * 16
                    m_row_inline1218__ssa_v0_1: pl.Scalar[pl.INDEX] = m_blk_base_inline1306__ssa_v0 + (m_sb_inline1219__idx_v0 + 1) * 16
                    m_cur_mi_inline1217__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 64), pl.Mem.Vec] = pl.tile.load(sparse_blk_mi_inline1234__ssa_v0, [m_row_inline1218__ssa_v0, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec)
                    m_cur_li_inline1293__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_56, pl.const(99072, pl.INT64), 64), pl.Mem.Vec] = pl.tile.load(sparse_blk_li_inline1283__ssa_v0, [m_row_inline1218__ssa_v0, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec)
                    m_cur_oi_inline1216__tile: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(sparse_blk_oi_inline1233__ssa_v0, [m_row_inline1218__ssa_v0, 0], [16, 512], [16, 512], target_memory=pl.Mem.Vec)
                    m_cur_mi_inline1217__tile_1: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_57, pl.const(99328, pl.INT64), 64), pl.Mem.Vec] = pl.tile.load(sparse_blk_mi_inline1234__ssa_v0, [m_row_inline1218__ssa_v0_1, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec)
                    m_cur_li_inline1293__tile_1: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_17, pl.const(32768, pl.INT64), 64), pl.Mem.Vec] = pl.tile.load(sparse_blk_li_inline1283__ssa_v0, [m_row_inline1218__ssa_v0_1, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec)
                    m_cur_oi_inline1216__tile_1: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_18, pl.const(32832, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(sparse_blk_oi_inline1233__ssa_v0, [m_row_inline1218__ssa_v0_1, 0], [16, 512], [16, 512], target_memory=pl.Mem.Vec)
                    m_mi_new_inline1215__rm_a0_tmp_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_52, pl.const(98560, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_inline1221__iter_v1, [1, 16])
                    m_mi_new_inline1215__rm_a1_tmp_v1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_cur_mi_inline1217__tile, [1, 16])
                    m_mi_new_inline1215__row_major_tmp_v2: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_19, pl.const(65600, pl.INT64), 64), pl.Mem.Vec] = pl.tile.maximum(m_mi_new_inline1215__rm_a0_tmp_v0, m_mi_new_inline1215__rm_a1_tmp_v1)
                    m_mi_new_inline1215__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_19, pl.const(65600, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_new_inline1215__row_major_tmp_v2, [16, 1])
                    m_mi_inline1221__ssa_v3: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_19, pl.const(65600, pl.INT64), 64), pl.Mem.Vec] = m_mi_new_inline1215__tile
                    t__rm_a0_tmp_v3: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_52, pl.const(98560, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_inline1221__iter_v1, [1, 16])
                    t__rm_a1_tmp_v4: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_19, pl.const(65600, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_new_inline1215__tile, [1, 16])
                    t__row_major_tmp_v5: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_20, pl.const(65664, pl.INT64), 64), pl.Mem.Vec] = pl.tile.sub(t__rm_a0_tmp_v3, t__rm_a1_tmp_v4)
                    t__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_20, pl.const(65664, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v5, [16, 1])
                    m_alpha_inline1214__rm_a0_tmp_v6: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_20, pl.const(65664, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile, [1, 16])
                    m_alpha_inline1214__row_major_tmp_v7: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_20, pl.const(65664, pl.INT64), 64), pl.Mem.Vec] = pl.tile.exp(m_alpha_inline1214__rm_a0_tmp_v6)
                    m_alpha_inline1214__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_20, pl.const(65664, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_alpha_inline1214__row_major_tmp_v7, [16, 1])
                    t__rm_a0_tmp_v8: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_cur_mi_inline1217__tile, [1, 16])
                    t__rm_a1_tmp_v9: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_19, pl.const(65600, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_new_inline1215__tile, [1, 16])
                    t__row_major_tmp_v10: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 64), pl.Mem.Vec] = pl.tile.sub(t__rm_a0_tmp_v8, t__rm_a1_tmp_v9)
                    t__tile_1: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v10, [16, 1])
                    m_beta_inline1213__rm_a0_tmp_v11: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_1, [1, 16])
                    m_beta_inline1213__row_major_tmp_v12: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_23, pl.const(65728, pl.INT64), 64), pl.Mem.Vec] = pl.tile.exp(m_beta_inline1213__rm_a0_tmp_v11)
                    m_beta_inline1213__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_23, pl.const(65728, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_beta_inline1213__row_major_tmp_v12, [16, 1])
                    t__rm_a0_tmp_v13: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_20, pl.const(65664, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_alpha_inline1214__tile, [1, 16])
                    t__rm_a1_tmp_v14: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_53, pl.const(98816, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_li_inline1302__iter_v1, [1, 16])
                    t__row_major_tmp_v15: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 64), pl.Mem.Vec] = pl.tile.mul(t__rm_a0_tmp_v13, t__rm_a1_tmp_v14)
                    t__tile_2: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v15, [16, 1])
                    t__rm_a0_tmp_v16: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_23, pl.const(65728, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_beta_inline1213__tile, [1, 16])
                    t__rm_a1_tmp_v17: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_56, pl.const(99072, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_cur_li_inline1293__tile, [1, 16])
                    t__row_major_tmp_v18: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_56, pl.const(99072, pl.INT64), 64), pl.Mem.Vec] = pl.tile.mul(t__rm_a0_tmp_v16, t__rm_a1_tmp_v17)
                    t__tile_3: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_56, pl.const(99072, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v18, [16, 1])
                    m_li_inline1302__rm_a0_tmp_v19: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_2, [1, 16])
                    m_li_inline1302__rm_a1_tmp_v20: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_56, pl.const(99072, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_3, [1, 16])
                    m_li_inline1302__row_major_tmp_v21: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_56, pl.const(99072, pl.INT64), 64), pl.Mem.Vec] = pl.tile.add(m_li_inline1302__rm_a0_tmp_v19, m_li_inline1302__rm_a1_tmp_v20)
                    m_li_inline1302__tile_1: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_56, pl.const(99072, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_li_inline1302__row_major_tmp_v21, [16, 1])
                    t__tile_4: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(m_oi_inline1220__iter_v1, m_alpha_inline1214__tile)
                    t__tile_5: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(m_cur_oi_inline1216__tile, m_beta_inline1213__tile)
                    m_oi_inline1220__tile_1: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.add(t__tile_4, t__tile_5)
                    m_mi_new_inline1215__rm_a0_tmp_v0_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_19, pl.const(65600, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_inline1221__ssa_v3, [1, 16])
                    m_mi_new_inline1215__rm_a1_tmp_v1_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_57, pl.const(99328, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_cur_mi_inline1217__tile_1, [1, 16])
                    m_mi_new_inline1215__row_major_tmp_v2_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 64), pl.Mem.Vec] = pl.tile.maximum(m_mi_new_inline1215__rm_a0_tmp_v0_1, m_mi_new_inline1215__rm_a1_tmp_v1_1)
                    m_mi_new_inline1215__tile_1: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_new_inline1215__row_major_tmp_v2_1, [16, 1])
                    m_mi_inline1221__ssa_v3_1: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 64), pl.Mem.Vec] = m_mi_new_inline1215__tile_1
                    t__rm_a0_tmp_v3_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_19, pl.const(65600, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_inline1221__ssa_v3, [1, 16])
                    t__rm_a1_tmp_v4_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_new_inline1215__tile_1, [1, 16])
                    t__row_major_tmp_v5_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_19, pl.const(65600, pl.INT64), 64), pl.Mem.Vec] = pl.tile.sub(t__rm_a0_tmp_v3_1, t__rm_a1_tmp_v4_1)
                    t__tile_6: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_19, pl.const(65600, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v5_1, [16, 1])
                    m_alpha_inline1214__rm_a0_tmp_v6_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_19, pl.const(65600, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_6, [1, 16])
                    m_alpha_inline1214__row_major_tmp_v7_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_19, pl.const(65600, pl.INT64), 64), pl.Mem.Vec] = pl.tile.exp(m_alpha_inline1214__rm_a0_tmp_v6_1)
                    m_alpha_inline1214__tile_1: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_19, pl.const(65600, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_alpha_inline1214__row_major_tmp_v7_1, [16, 1])
                    t__rm_a0_tmp_v8_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_57, pl.const(99328, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_cur_mi_inline1217__tile_1, [1, 16])
                    t__rm_a1_tmp_v9_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_new_inline1215__tile_1, [1, 16])
                    t__row_major_tmp_v10_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_57, pl.const(99328, pl.INT64), 64), pl.Mem.Vec] = pl.tile.sub(t__rm_a0_tmp_v8_1, t__rm_a1_tmp_v9_1)
                    t__tile_7: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_57, pl.const(99328, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v10_1, [16, 1])
                    m_beta_inline1213__rm_a0_tmp_v11_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_57, pl.const(99328, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_7, [1, 16])
                    m_beta_inline1213__row_major_tmp_v12_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_57, pl.const(99328, pl.INT64), 64), pl.Mem.Vec] = pl.tile.exp(m_beta_inline1213__rm_a0_tmp_v11_1)
                    m_beta_inline1213__tile_1: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_57, pl.const(99328, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_beta_inline1213__row_major_tmp_v12_1, [16, 1])
                    t__rm_a0_tmp_v13_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_19, pl.const(65600, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_alpha_inline1214__tile_1, [1, 16])
                    t__rm_a1_tmp_v14_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_56, pl.const(99072, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_li_inline1302__tile_1, [1, 16])
                    t__row_major_tmp_v15_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_56, pl.const(99072, pl.INT64), 64), pl.Mem.Vec] = pl.tile.mul(t__rm_a0_tmp_v13_1, t__rm_a1_tmp_v14_1)
                    t__tile_8: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_56, pl.const(99072, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v15_1, [16, 1])
                    t__rm_a0_tmp_v16_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_57, pl.const(99328, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_beta_inline1213__tile_1, [1, 16])
                    t__rm_a1_tmp_v17_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_17, pl.const(32768, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_cur_li_inline1293__tile_1, [1, 16])
                    t__row_major_tmp_v18_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_17, pl.const(32768, pl.INT64), 64), pl.Mem.Vec] = pl.tile.mul(t__rm_a0_tmp_v16_1, t__rm_a1_tmp_v17_1)
                    t__tile_9: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_17, pl.const(32768, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v18_1, [16, 1])
                    m_li_inline1302__rm_a0_tmp_v19_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_56, pl.const(99072, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_8, [1, 16])
                    m_li_inline1302__rm_a1_tmp_v20_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_17, pl.const(32768, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_9, [1, 16])
                    m_li_inline1302__row_major_tmp_v21_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_56, pl.const(99072, pl.INT64), 64), pl.Mem.Vec] = pl.tile.add(m_li_inline1302__rm_a0_tmp_v19_1, m_li_inline1302__rm_a1_tmp_v20_1)
                    m_li_inline1302__tile_2: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_56, pl.const(99072, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_li_inline1302__row_major_tmp_v21_1, [16, 1])
                    t__tile_10: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(m_oi_inline1220__tile_1, m_alpha_inline1214__tile_1)
                    t__tile_11: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_18, pl.const(32832, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(m_cur_oi_inline1216__tile_1, m_beta_inline1213__tile_1)
                    m_oi_inline1220__tile_2: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_66, pl.const(99584, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.add(t__tile_10, t__tile_11)
                    m_mi_inline1221__ssa_v3_mv: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_52, pl.const(98560, pl.INT64), 64), pl.Mem.Vec] = pl.tile.move(m_mi_inline1221__ssa_v3_1, target_memory=pl.Mem.Vec)
                    m_li_inline1302__tile_mv: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_53, pl.const(98816, pl.INT64), 64), pl.Mem.Vec] = pl.tile.move(m_li_inline1302__tile_2, target_memory=pl.Mem.Vec)
                    m_li_inline1302__rv_v2, m_mi_inline1221__rv_v2, m_oi_inline1220__rv_v2 = pl.yield_(m_li_inline1302__tile_mv, m_mi_inline1221__ssa_v3_mv, m_oi_inline1220__tile_2)
                t__tile_12: pl.Tile[[16], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.load(attn_sink__ssa_v0, [m_h0_inline1224__ssa_v0], [16], [16], target_memory=pl.Mem.Vec)
                n_sink_bias_inline1266__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_12, [16, 1])
                t__rm_a0_tmp_v22: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_52, pl.const(98560, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_inline1221__rv_v2, [1, 16])
                t__rm_a1_tmp_v23: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_52, pl.const(98560, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_inline1221__rv_v2, [1, 16])
                t__row_major_tmp_v24: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_18, pl.const(32832, pl.INT64), 64), pl.Mem.Vec] = pl.tile.sub(t__rm_a0_tmp_v22, t__rm_a1_tmp_v23)
                t__tile_13: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_18, pl.const(32832, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v24, [16, 1])
                n_sink_tile_inline1212__rm_a0_tmp_v25: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_18, pl.const(32832, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_13, [1, 16])
                n_sink_tile_inline1212__rm_a1_tmp_v26: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(n_sink_bias_inline1266__tile, [1, 16])
                n_sink_tile_inline1212__row_major_tmp_v27: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.add(n_sink_tile_inline1212__rm_a0_tmp_v25, n_sink_tile_inline1212__rm_a1_tmp_v26)
                n_sink_tile_inline1212__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(n_sink_tile_inline1212__row_major_tmp_v27, [16, 1])
                t__rm_a0_tmp_v28: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(n_sink_tile_inline1212__tile, [1, 16])
                t__rm_a1_tmp_v29: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_52, pl.const(98560, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_inline1221__rv_v2, [1, 16])
                t__row_major_tmp_v30: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.sub(t__rm_a0_tmp_v28, t__rm_a1_tmp_v29)
                t__tile_14: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v30, [16, 1])
                t__rm_a0_tmp_v31: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_14, [1, 16])
                t__row_major_tmp_v32: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.exp(t__rm_a0_tmp_v31)
                t__tile_15: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v32, [16, 1])
                n_denom_inline1254__rm_a0_tmp_v33: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_53, pl.const(98816, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_li_inline1302__rv_v2, [1, 16])
                n_denom_inline1254__rm_a1_tmp_v34: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_15, [1, 16])
                n_denom_inline1254__row_major_tmp_v35: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.add(n_denom_inline1254__rm_a0_tmp_v33, n_denom_inline1254__rm_a1_tmp_v34)
                n_denom_inline1254__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(n_denom_inline1254__row_major_tmp_v35, [16, 1])
                t__tile_16: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_66, pl.const(99584, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_div(m_oi_inline1220__rv_v2, n_denom_inline1254__tile)
                n_full_inline1211__tile: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_66, pl.const(99584, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.slice(t__tile_16, [16, 512], [0, 0])
                n_bf16_inline1210__tile: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(n_full_inline1211__tile, target_type=pl.BF16, mode='rint')
                m_rope_inline1208__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_66, pl.const(101376, pl.INT64), 30976), pl.Mem.Vec] = pl.tile.slice(n_full_inline1211__tile, [16, 64], [0, 448])
                m_cos_il_inline1286__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_52, pl.const(98560, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(rope_cos_il_inline1232__ssa_v0, [m_t_inline1290__ssa_v0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                m_sin_signed_inline1225__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_53, pl.const(98816, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(rope_sin_signed_inline1231__ssa_v0, [m_t_inline1290__ssa_v0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                t__tile_17: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_18, pl.const(32832, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(rope_swap_idx_inline1230__ssa_v0, [0, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
                gather_acc_init: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                for gather_lv, (gather_ia,) in pl.range(16, init_values=(gather_acc_init,)):
                    gather_inp_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_66, pl.const(99584, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(m_rope_inline1208__tile, [1, 64], [gather_lv, 0], [1, 64])
                    gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_18, pl.const(32832, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(t__tile_17, [1, 64], [gather_lv, 0], [1, 64])
                    gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_56, pl.const(99072, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                    gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_57, pl.const(99328, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
                    gather_asmbl: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
                    gather_rv: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 4096), pl.Mem.Vec] = pl.yield_(gather_asmbl)
                m_swapped_inline1207__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 4096), pl.Mem.Vec] = gather_rv
                m_rope_inline1208__tile_textract: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(32832, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.extract(t__tile_16, 0, 448, [16, 64], target_memory=pl.Mem.Vec)
                t__tile_18: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_66, pl.const(99584, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(m_rope_inline1208__tile_textract, m_cos_il_inline1286__tile)
                t__tile_19: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(32832, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(m_swapped_inline1207__tile, m_sin_signed_inline1225__tile)
                m_rot_inline1206__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_66, pl.const(99584, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(t__tile_18, t__tile_19)
                n_rope_bf16_inline1205__tile: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_18, pl.const(32832, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(m_rot_inline1206__tile, target_type=pl.BF16, mode='rint')
                t__tile_20: pl.Tile[[16, 448], pl.BF16, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 16256), pl.Mem.Vec] = pl.tile.slice(n_bf16_inline1210__tile, [16, 448], [0, 0])
                n_full_bf16_inline1285__tile: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(99584, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.concat(t__tile_20, n_rope_bf16_inline1205__tile)
                n_head_inline1244__ssa_v0: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0
                source_row_inline1238__ssa_v0: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v0 // 8 * 256 + m_t_inline1290__ssa_v0
                source_col_inline1204__ssa_v0: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v0 % 8 * 512
                t__tile_21: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(99584, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(n_full_bf16_inline1285__tile, [1, 512], [0, 0])
                attention_grouped_inline1276__tile: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(t__tile_21, [source_row_inline1238__ssa_v0, source_col_inline1204__ssa_v0], attention_grouped_inline1276__iter_v3)
                n_head_inline1244__ssa_v1: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 1
                source_row_inline1238__ssa_v1: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v1 // 8 * 256 + m_t_inline1290__ssa_v0
                source_col_inline1204__ssa_v1: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v1 % 8 * 512
                t__tile_22: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(100608, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(n_full_bf16_inline1285__tile, [1, 512], [1, 0])
                attention_grouped_inline1276__tile_1: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(t__tile_22, [source_row_inline1238__ssa_v1, source_col_inline1204__ssa_v1], attention_grouped_inline1276__tile)
                n_head_inline1244__ssa_v2: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 2
                source_row_inline1238__ssa_v2: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v2 // 8 * 256 + m_t_inline1290__ssa_v0
                source_col_inline1204__ssa_v2: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v2 % 8 * 512
                t__tile_23: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(101632, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(n_full_bf16_inline1285__tile, [1, 512], [2, 0])
                attention_grouped_inline1276__tile_2: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(t__tile_23, [source_row_inline1238__ssa_v2, source_col_inline1204__ssa_v2], attention_grouped_inline1276__tile_1)
                n_head_inline1244__ssa_v3: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 3
                source_row_inline1238__ssa_v3: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v3 // 8 * 256 + m_t_inline1290__ssa_v0
                source_col_inline1204__ssa_v3: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v3 % 8 * 512
                t__tile_24: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(102656, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(n_full_bf16_inline1285__tile, [1, 512], [3, 0])
                attention_grouped_inline1276__tile_3: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(t__tile_24, [source_row_inline1238__ssa_v3, source_col_inline1204__ssa_v3], attention_grouped_inline1276__tile_2)
                n_head_inline1244__ssa_v4: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 4
                source_row_inline1238__ssa_v4: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v4 // 8 * 256 + m_t_inline1290__ssa_v0
                source_col_inline1204__ssa_v4: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v4 % 8 * 512
                t__tile_25: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(103680, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(n_full_bf16_inline1285__tile, [1, 512], [4, 0])
                attention_grouped_inline1276__tile_4: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(t__tile_25, [source_row_inline1238__ssa_v4, source_col_inline1204__ssa_v4], attention_grouped_inline1276__tile_3)
                n_head_inline1244__ssa_v5: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 5
                source_row_inline1238__ssa_v5: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v5 // 8 * 256 + m_t_inline1290__ssa_v0
                source_col_inline1204__ssa_v5: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v5 % 8 * 512
                t__tile_26: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(104704, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(n_full_bf16_inline1285__tile, [1, 512], [5, 0])
                attention_grouped_inline1276__tile_5: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(t__tile_26, [source_row_inline1238__ssa_v5, source_col_inline1204__ssa_v5], attention_grouped_inline1276__tile_4)
                n_head_inline1244__ssa_v6: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 6
                source_row_inline1238__ssa_v6: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v6 // 8 * 256 + m_t_inline1290__ssa_v0
                source_col_inline1204__ssa_v6: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v6 % 8 * 512
                t__tile_27: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(105728, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(n_full_bf16_inline1285__tile, [1, 512], [6, 0])
                attention_grouped_inline1276__tile_6: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(t__tile_27, [source_row_inline1238__ssa_v6, source_col_inline1204__ssa_v6], attention_grouped_inline1276__tile_5)
                n_head_inline1244__ssa_v7: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 7
                source_row_inline1238__ssa_v7: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v7 // 8 * 256 + m_t_inline1290__ssa_v0
                source_col_inline1204__ssa_v7: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v7 % 8 * 512
                t__tile_28: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(106752, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(n_full_bf16_inline1285__tile, [1, 512], [7, 0])
                attention_grouped_inline1276__tile_7: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(t__tile_28, [source_row_inline1238__ssa_v7, source_col_inline1204__ssa_v7], attention_grouped_inline1276__tile_6)
                n_head_inline1244__ssa_v8: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 8
                source_row_inline1238__ssa_v8: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v8 // 8 * 256 + m_t_inline1290__ssa_v0
                source_col_inline1204__ssa_v8: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v8 % 8 * 512
                t__tile_29: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(107776, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(n_full_bf16_inline1285__tile, [1, 512], [8, 0])
                attention_grouped_inline1276__tile_8: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(t__tile_29, [source_row_inline1238__ssa_v8, source_col_inline1204__ssa_v8], attention_grouped_inline1276__tile_7)
                n_head_inline1244__ssa_v9: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 9
                source_row_inline1238__ssa_v9: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v9 // 8 * 256 + m_t_inline1290__ssa_v0
                source_col_inline1204__ssa_v9: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v9 % 8 * 512
                t__tile_30: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(108800, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(n_full_bf16_inline1285__tile, [1, 512], [9, 0])
                attention_grouped_inline1276__tile_9: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(t__tile_30, [source_row_inline1238__ssa_v9, source_col_inline1204__ssa_v9], attention_grouped_inline1276__tile_8)
                n_head_inline1244__ssa_v10: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 10
                source_row_inline1238__ssa_v10: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v10 // 8 * 256 + m_t_inline1290__ssa_v0
                source_col_inline1204__ssa_v10: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v10 % 8 * 512
                t__tile_31: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(109824, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(n_full_bf16_inline1285__tile, [1, 512], [10, 0])
                attention_grouped_inline1276__tile_10: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(t__tile_31, [source_row_inline1238__ssa_v10, source_col_inline1204__ssa_v10], attention_grouped_inline1276__tile_9)
                n_head_inline1244__ssa_v11: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 11
                source_row_inline1238__ssa_v11: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v11 // 8 * 256 + m_t_inline1290__ssa_v0
                source_col_inline1204__ssa_v11: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v11 % 8 * 512
                t__tile_32: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(110848, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(n_full_bf16_inline1285__tile, [1, 512], [11, 0])
                attention_grouped_inline1276__tile_11: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(t__tile_32, [source_row_inline1238__ssa_v11, source_col_inline1204__ssa_v11], attention_grouped_inline1276__tile_10)
                n_head_inline1244__ssa_v12: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 12
                source_row_inline1238__ssa_v12: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v12 // 8 * 256 + m_t_inline1290__ssa_v0
                source_col_inline1204__ssa_v12: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v12 % 8 * 512
                t__tile_33: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(111872, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(n_full_bf16_inline1285__tile, [1, 512], [12, 0])
                attention_grouped_inline1276__tile_12: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(t__tile_33, [source_row_inline1238__ssa_v12, source_col_inline1204__ssa_v12], attention_grouped_inline1276__tile_11)
                n_head_inline1244__ssa_v13: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 13
                source_row_inline1238__ssa_v13: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v13 // 8 * 256 + m_t_inline1290__ssa_v0
                source_col_inline1204__ssa_v13: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v13 % 8 * 512
                t__tile_34: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(112896, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(n_full_bf16_inline1285__tile, [1, 512], [13, 0])
                attention_grouped_inline1276__tile_13: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(t__tile_34, [source_row_inline1238__ssa_v13, source_col_inline1204__ssa_v13], attention_grouped_inline1276__tile_12)
                n_head_inline1244__ssa_v14: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 14
                source_row_inline1238__ssa_v14: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v14 // 8 * 256 + m_t_inline1290__ssa_v0
                source_col_inline1204__ssa_v14: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v14 % 8 * 512
                t__tile_35: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(113920, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(n_full_bf16_inline1285__tile, [1, 512], [14, 0])
                attention_grouped_inline1276__tile_14: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(t__tile_35, [source_row_inline1238__ssa_v14, source_col_inline1204__ssa_v14], attention_grouped_inline1276__tile_13)
                n_head_inline1244__ssa_v15: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 15
                source_row_inline1238__ssa_v15: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v15 // 8 * 256 + m_t_inline1290__ssa_v0
                source_col_inline1204__ssa_v15: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v15 % 8 * 512
                t__tile_36: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(114944, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(n_full_bf16_inline1285__tile, [1, 512], [15, 0])
                attention_grouped_inline1276__tile_15: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(t__tile_36, [source_row_inline1238__ssa_v15, source_col_inline1204__ssa_v15], attention_grouped_inline1276__tile_14)
                attention_grouped_inline1276__rv_v4: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_65", pl.const(0, pl.INT64), 16777216)] = pl.yield_(attention_grouped_inline1276__tile_15)
            source_row_inline1238__ssa_v16: pl.Scalar[pl.INDEX] = global_group0_inline1314__ssa_v0 * 256 + m_t0_inline1278__ssa_v0
            target_row_inline1203__ssa_v0: pl.Scalar[pl.INDEX] = local_group0_inline1222__ssa_v0 * 512 + pl.cast(tp_rank__ssa_v0, pl.INDEX) * 256 + m_t0_inline1278__ssa_v0
            tput_stage: pl.Tile[[8, 4096], pl.BF16, pl.MemRef(mem_vec_66, pl.const(99584, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.create([8, 4096], dtype=pl.BF16, target_memory=pl.Mem.Vec)
            pld.tile.put(attention_window__ssa_v0, pl.cast(group_base__ssa_v0, pl.INDEX) + destination_rank_inline1223__ssa_v0, attention_grouped_inline1276__rv_v4, tput_stage, [target_row_inline1203__ssa_v0, 0], [source_row_inline1238__ssa_v16, 0], [8, 4096], atomic=pl.AtomicType.None_)
            source_row_inline1238__ssa_v17: pl.Scalar[pl.INDEX] = global_group0_inline1314__ssa_v0 * 256 + m_t0_inline1278__ssa_v0 + 256
            target_row_inline1203__ssa_v1: pl.Scalar[pl.INDEX] = local_group0_inline1222__ssa_v0 * 512 + pl.cast(tp_rank__ssa_v0, pl.INDEX) * 256 + m_t0_inline1278__ssa_v0 + 512
            tput_stage_1: pl.Tile[[8, 4096], pl.BF16, pl.MemRef(mem_vec_66, pl.const(99584, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.create([8, 4096], dtype=pl.BF16, target_memory=pl.Mem.Vec)
            pld.tile.put(attention_window__ssa_v0, pl.cast(group_base__ssa_v0, pl.INDEX) + destination_rank_inline1223__ssa_v0, attention_grouped_inline1276__rv_v4, tput_stage_1, [target_row_inline1203__ssa_v1, 0], [source_row_inline1238__ssa_v17, 0], [8, 4096], atomic=pl.AtomicType.None_)
            attention_grouped_inline1276__rv_v2: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_68", pl.const(0, pl.INT64), 16777216)] = pl.yield_(attention_grouped_inline1276__rv_v4)
        for peer_tp_inline1273__idx_v0 in pl.range(2):
            if peer_tp_inline1273__idx_v0 != pl.cast(tp_rank__ssa_v0, pl.INDEX):
                pld.system.notify(attention_signal__ssa_v0, pl.cast(group_base__ssa_v0, pl.INDEX) + peer_tp_inline1273__idx_v0, [tp_rank__ssa_v0, 0], pl.const(1, pl.INT32), op=0)
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def csa_merge_pack_publish_spmd(self, attention_grouped_inline1276__ssa_v0: pl.InOut[pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)]], pack_work_count_inline1228__ssa_v0: pl.Scalar[pl.INDEX], sparse_blk_mi_inline1234__ssa_v0: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], sparse_blk_li_inline1283__ssa_v0: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)], sparse_blk_oi_inline1233__ssa_v0: pl.Tensor[[t_blk_inline2373__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)], attn_sink__ssa_v0: pl.Tensor[[64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 256)], rope_cos_il_inline1232__ssa_v0: pl.Tensor[[256, 64], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 65536)], rope_sin_signed_inline1231__ssa_v0: pl.Tensor[[256, 64], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 65536)], rope_swap_idx_inline1230__ssa_v0: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 4096)], tp_rank__ssa_v0: pl.Scalar[pl.INT32], attention_window__ssa_v0: pl.Out[pld.DistributedTensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 16777216)]], group_base__ssa_v0: pl.Scalar[pl.INT32], attention_signal__ssa_v0: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 8)]], attention_window_ctx: pld.CommCtx, attention_signal_ctx: pld.CommCtx):
        self.csa_merge_pack_publish(attention_grouped_inline1276__ssa_v0, pack_work_count_inline1228__ssa_v0, sparse_blk_mi_inline1234__ssa_v0, sparse_blk_li_inline1283__ssa_v0, sparse_blk_oi_inline1233__ssa_v0, attn_sink__ssa_v0, rope_cos_il_inline1232__ssa_v0, rope_sin_signed_inline1231__ssa_v0, rope_swap_idx_inline1230__ssa_v0, tp_rank__ssa_v0, attention_window__ssa_v0, group_base__ssa_v0, attention_signal__ssa_v0, attention_window_ctx, attention_signal_ctx, attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.output_existing, pl.adir.scalar, pl.adir.inout, pl.adir.scalar, pl.adir.scalar]})
