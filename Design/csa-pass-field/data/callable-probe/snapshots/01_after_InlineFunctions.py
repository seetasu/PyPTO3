            with pl.spmd(48, name_hint="csa_merge_pack_publish_spmd", deps=[qk_tid_inline1229, attn_rope_tid_inline1294]) as publish_tid_inline1260:
                worker_inline1227: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                for pack_work_inline1270 in pl.range(worker_inline1227, pack_work_count_inline1228, 48):
                    token_block_inline1226: pl.Scalar[pl.INDEX] = pack_work_inline1270 // 4
                    m_h_idx_inline1303: pl.Scalar[pl.INDEX] = pack_work_inline1270 - token_block_inline1226 * 4
                    m_t0_inline1278: pl.Scalar[pl.INDEX] = token_block_inline1226 * 8
                    m_h0_inline1224: pl.Scalar[pl.INDEX] = m_h_idx_inline1303 * 16
                    global_group0_inline1314: pl.Scalar[pl.INDEX] = m_h0_inline1224 // 8
                    destination_rank_inline1223: pl.Scalar[pl.INDEX] = global_group0_inline1314 // 4
                    local_group0_inline1222: pl.Scalar[pl.INDEX] = global_group0_inline1314 - destination_rank_inline1223 * 4
                    for m_dt_inline1287 in pl.range(8):
                        m_t_inline1290: pl.Scalar[pl.INDEX] = m_t0_inline1278 + m_dt_inline1287
                        m_idx_inline1289: pl.Scalar[pl.INDEX] = m_t_inline1290 * 4 + m_h_idx_inline1303
                        m_blk_base_inline1306: pl.Scalar[pl.INDEX] = m_idx_inline1289 * 5 * 16
                        m_mi_inline1221: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.slice(sparse_blk_mi_inline1234, [16, 1], [m_blk_base_inline1306, 0])
                        m_li_inline1302: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.slice(sparse_blk_li_inline1283, [16, 1], [m_blk_base_inline1306, 0])
                        m_oi_inline1220: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.slice(sparse_blk_oi_inline1233, [16, 512], [m_blk_base_inline1306, 0])
                        for m_sb_inline1219 in pl.pipeline(1, 5, stage=2):
                            m_row_inline1218: pl.Scalar[pl.INDEX] = m_blk_base_inline1306 + m_sb_inline1219 * 16
                            m_cur_mi_inline1217: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.slice(sparse_blk_mi_inline1234, [16, 1], [m_row_inline1218, 0])
                            m_cur_li_inline1293: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.slice(sparse_blk_li_inline1283, [16, 1], [m_row_inline1218, 0])
                            m_cur_oi_inline1216: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.slice(sparse_blk_oi_inline1233, [16, 512], [m_row_inline1218, 0])
                            m_mi_new_inline1215: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.maximum(m_mi_inline1221, m_cur_mi_inline1217)
                            m_alpha_inline1214: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.exp(pl.tensor.sub(m_mi_inline1221, m_mi_new_inline1215))
                            m_beta_inline1213: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.exp(pl.tensor.sub(m_cur_mi_inline1217, m_mi_new_inline1215))
                            m_li_inline1302: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.add(pl.tensor.mul(m_alpha_inline1214, m_li_inline1302), pl.tensor.mul(m_beta_inline1213, m_cur_li_inline1293))
                            m_oi_inline1220: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.add(pl.tensor.row_expand_mul(m_oi_inline1220, m_alpha_inline1214), pl.tensor.row_expand_mul(m_cur_oi_inline1216, m_beta_inline1213))
                            m_mi_inline1221: pl.Tensor[[16, 1], pl.FP32] = m_mi_new_inline1215
                        n_sink_bias_inline1266: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(attn_sink, [16], [m_h0_inline1224]), [16, 1])
                        n_sink_tile_inline1212: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.add(pl.tensor.sub(m_mi_inline1221, m_mi_inline1221), n_sink_bias_inline1266)
                        n_denom_inline1254: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.add(m_li_inline1302, pl.tensor.exp(pl.tensor.sub(n_sink_tile_inline1212, m_mi_inline1221)))
                        n_full_inline1211: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.slice(pl.tensor.row_expand_div(m_oi_inline1220, n_denom_inline1254), [16, 512], [0, 0])
                        n_bf16_inline1210: pl.Tensor[[16, 512], pl.BF16] = pl.tensor.cast(n_full_inline1211, target_type=pl.BF16, mode='rint')
                        m_rope_inline1208: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(n_full_inline1211, [16, 64], [0, 448])
                        m_cos_il_inline1286: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_cos_il_inline1232, [1, 64], [m_t_inline1290, 0])
                        m_sin_signed_inline1225: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_sin_signed_inline1231, [1, 64], [m_t_inline1290, 0])
                        m_swapped_inline1207: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.gather(m_rope_inline1208, pl.tensor.slice(rope_swap_idx_inline1230, [16, 64], [0, 0]), dim=-1)
                        m_rot_inline1206: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.add(pl.tensor.col_expand_mul(m_rope_inline1208, m_cos_il_inline1286), pl.tensor.col_expand_mul(m_swapped_inline1207, m_sin_signed_inline1225))
                        n_rope_bf16_inline1205: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.cast(m_rot_inline1206, target_type=pl.BF16, mode='rint')
                        n_full_bf16_inline1285: pl.Tensor[[16, 512], pl.BF16] = pl.tensor.concat(pl.tensor.slice(n_bf16_inline1210, [16, 448], [0, 0]), n_rope_bf16_inline1205)
                        for n_hi_inline1209 in pl.unroll(16):
                            n_head_inline1244: pl.Scalar[pl.INDEX] = m_h0_inline1224 + n_hi_inline1209
                            source_row_inline1238: pl.Scalar[pl.INDEX] = n_head_inline1244 // 8 * 256 + m_t_inline1290
                            source_col_inline1204: pl.Scalar[pl.INDEX] = n_head_inline1244 % 8 * 512
                            attention_grouped_inline1276: pl.Tensor[[2048, 4096], pl.BF16] = pl.tensor.assemble(attention_grouped_inline1276, pl.tensor.slice(n_full_bf16_inline1285, [1, 512], [n_hi_inline1209, 0]), [source_row_inline1238, source_col_inline1204])
                    for group_slot_inline1262 in pl.unroll(2):
                        source_row_inline1238: pl.Scalar[pl.INDEX] = (global_group0_inline1314 + group_slot_inline1262) * 256 + m_t0_inline1278
                        target_row_inline1203: pl.Scalar[pl.INDEX] = (local_group0_inline1222 + group_slot_inline1262) * 512 + pl.cast(tp_rank, pl.INDEX) * 256 + m_t0_inline1278
                        pld.tensor.put(attention_window, pl.cast(group_base, pl.INDEX) + destination_rank_inline1223, attention_grouped_inline1276, [target_row_inline1203, 0], [source_row_inline1238, 0], [8, 4096], atomic=pl.AtomicType.None_, chunk_rows=8, chunk_cols=4096)
                for peer_tp_inline1273 in pl.range(2):
                    if peer_tp_inline1273 != pl.cast(tp_rank, pl.INDEX):
                        pld.system.notify(attention_signal, pl.cast(group_base, pl.INDEX) + peer_tp_inline1273, [tp_rank, 0], pl.const(1, pl.INT32), op=0)
