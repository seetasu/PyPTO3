            with pl.spmd(48, name_hint="csa_merge_pack_publish_spmd", deps=[qk_tid, attn_rope_tid]) as publish_tid:
                worker: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                for pack_work in pl.range(worker, pack_work_count, 48):
                    token_block: pl.Scalar[pl.INDEX] = pack_work // 4
                    m_h_idx: pl.Scalar[pl.INDEX] = pack_work - token_block * 4
                    m_t0: pl.Scalar[pl.INDEX] = token_block * 8
                    m_h0: pl.Scalar[pl.INDEX] = m_h_idx * 16
                    global_group0: pl.Scalar[pl.INDEX] = m_h0 // 8
                    destination_rank: pl.Scalar[pl.INDEX] = global_group0 // 4
                    local_group0: pl.Scalar[pl.INDEX] = global_group0 - destination_rank * 4
                    for m_dt in pl.range(8):
                        m_t: pl.Scalar[pl.INDEX] = m_t0 + m_dt
                        m_idx: pl.Scalar[pl.INDEX] = m_t * 4 + m_h_idx
                        m_blk_base: pl.Scalar[pl.INDEX] = m_idx * 5 * 16
                        m_mi: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.slice(sparse_blk_mi, [16, 1], [m_blk_base, 0])
                        m_li: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.slice(sparse_blk_li, [16, 1], [m_blk_base, 0])
                        m_oi: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.slice(sparse_blk_oi, [16, 512], [m_blk_base, 0])
                        for m_sb in pl.pipeline(1, 5, stage=2):
                            m_row: pl.Scalar[pl.INDEX] = m_blk_base + m_sb * 16
                            m_cur_mi: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.slice(sparse_blk_mi, [16, 1], [m_row, 0])
                            m_cur_li: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.slice(sparse_blk_li, [16, 1], [m_row, 0])
                            m_cur_oi: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.slice(sparse_blk_oi, [16, 512], [m_row, 0])
                            m_mi_new: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.maximum(m_mi, m_cur_mi)
                            m_alpha: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.exp(pl.tensor.sub(m_mi, m_mi_new))
                            m_beta: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.exp(pl.tensor.sub(m_cur_mi, m_mi_new))
                            m_li: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.add(pl.tensor.mul(m_alpha, m_li), pl.tensor.mul(m_beta, m_cur_li))
                            m_oi: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.add(pl.tensor.row_expand_mul(m_oi, m_alpha), pl.tensor.row_expand_mul(m_cur_oi, m_beta))
                            m_mi: pl.Tensor[[16, 1], pl.FP32] = m_mi_new
                        n_sink_bias: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(attn_sink, [16], [m_h0]), [16, 1])
                        n_sink_tile: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.add(pl.tensor.sub(m_mi, m_mi), n_sink_bias)
                        n_denom: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.add(m_li, pl.tensor.exp(pl.tensor.sub(n_sink_tile, m_mi)))
                        n_full: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.slice(pl.tensor.row_expand_div(m_oi, n_denom), [16, 512], [0, 0])
                        n_bf16: pl.Tensor[[16, 512], pl.BF16] = pl.tensor.cast(n_full, target_type=pl.BF16, mode='rint')
                        m_rope: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(n_full, [16, 64], [0, 448])
                        m_cos_il: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_cos_il, [1, 64], [m_t, 0])
                        m_sin_signed: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_sin_signed, [1, 64], [m_t, 0])
                        m_swapped: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.gather(m_rope, pl.tensor.slice(rope_swap_idx, [16, 64], [0, 0]), dim=-1)
                        m_rot: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.add(pl.tensor.col_expand_mul(m_rope, m_cos_il), pl.tensor.col_expand_mul(m_swapped, m_sin_signed))
                        n_rope_bf16: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.cast(m_rot, target_type=pl.BF16, mode='rint')
                        n_full_bf16: pl.Tensor[[16, 512], pl.BF16] = pl.tensor.concat(pl.tensor.slice(n_bf16, [16, 448], [0, 0]), n_rope_bf16)
                        for n_hi in pl.unroll(16):
                            n_head: pl.Scalar[pl.INDEX] = m_h0 + n_hi
                            source_row: pl.Scalar[pl.INDEX] = n_head // 8 * 256 + m_t
                            source_col: pl.Scalar[pl.INDEX] = n_head % 8 * 512
                            attention_grouped: pl.Tensor[[2048, 4096], pl.BF16] = pl.tensor.assemble(attention_grouped, pl.tensor.slice(n_full_bf16, [1, 512], [n_hi, 0]), [source_row, source_col])
                    for group_slot in pl.unroll(2):
                        source_row: pl.Scalar[pl.INDEX] = (global_group0 + group_slot) * 256 + m_t0
                        target_row: pl.Scalar[pl.INDEX] = (local_group0 + group_slot) * 512 + pl.cast(tp_rank, pl.INDEX) * 256 + m_t0
                        pld.tensor.put(attention_window, pl.cast(group_base, pl.INDEX) + destination_rank, attention_grouped, [target_row, 0], [source_row, 0], [8, 4096], atomic=pl.AtomicType.None_, chunk_rows=8, chunk_cols=4096)
                for peer_tp in pl.range(2):
                    if peer_tp != pl.cast(tp_rank, pl.INDEX):
                        pld.system.notify(attention_signal, pl.cast(group_base, pl.INDEX) + peer_tp, [tp_rank, 0], pl.const(1, pl.INT32), op=0)
