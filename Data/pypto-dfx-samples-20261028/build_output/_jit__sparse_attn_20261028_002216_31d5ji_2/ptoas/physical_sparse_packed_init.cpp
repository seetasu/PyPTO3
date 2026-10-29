#include "pto/pto-inst.hpp"
using namespace pto;

template <typename Tensor>
static AICORE inline auto PTOAS__GLOBAL_TENSOR_DATA(Tensor &tensor)
    -> decltype(tensor.data()) {
  return tensor.data();
}


enum class PTOAutoSyncTailMode : int {
  kBarrierAll = 0,
  kSetWaitMte3ToSEvent0 = 1,
};

static AICORE inline void ptoas_auto_sync_tail(
    PTOAutoSyncTailMode mode = PTOAutoSyncTailMode::kBarrierAll) {
  switch (mode) {
  case PTOAutoSyncTailMode::kSetWaitMte3ToSEvent0:
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    break;
  case PTOAutoSyncTailMode::kBarrierAll:
  default:
    pipe_barrier(PIPE_ALL);
    break;
  }
}

template <typename Ptr>
static AICORE inline void PTOAS__DCCI_SINGLE_CACHE_LINE(Ptr ptr) {
  dcci((__gm__ void*)ptr, cache_line_t::SINGLE_CACHE_LINE);
}

AICORE void physical_sparse_packed_init(__gm__ bfloat16_t* v1, int64_t v2, int32_t v3, int32_t v4) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v5 = 0;
  // pto: %c512_index
  const int64_t v6 = 512;
  // pto: %c1_index
  const int64_t v7 = 1;
  // pto: %c0_index
  const int64_t v8 = 0;
  // pto: %c4_index
  const int64_t v9 = 4;
  // pto: %c128_index
  const int64_t v10 = 128;
  // pto: %cst_6
  const bfloat16_t v11 = 0.0f;
  // pto: %o_packed_heads_inline365__ssa_v0_view
  const int64_t v12 = 1;
  // pto: %o_packed_heads_inline365__ssa_v0_view
  const int64_t v13 = 1;
  // pto: %o_packed_heads_inline365__ssa_v0_view
  const int64_t v14 = 1;
  // pto: %o_packed_heads_inline365__ssa_v0_view
  int64_t v15 = (int64_t) v2;
  // pto: %o_packed_heads_inline365__ssa_v0_view
  int64_t v16 = v15 * v6;
  // pto: %o_packed_heads_inline365__ssa_v0_view
  int64_t v17 = v14 * v16;
  // pto: %o_packed_heads_inline365__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v18 = pto::Shape<1, 1, 1, -1, -1>(v12, v13, v14, v15, v6);
  // pto: %o_packed_heads_inline365__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v19 = pto::Stride<-1, -1, -1, -1, -1>(v13 * v17, v17, v16, v6, v7);
  // pto: %o_packed_heads_inline365__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v20 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v18, v19);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  for (int64_t v21 = v8; v21 < v9; v21 += v7) {
    // pto: %packed_blk_inline367__ssa_v0, %0
    ;
    int64_t v22 = (int64_t) ((uint64_t) ((int64_t) v3) * (uint64_t) v9);
    // pto: %1
    ;
    int64_t v23 = (int64_t) ((uint64_t) v22 + (uint64_t) v21);
    // pto: %2
    ;
    int64_t v24 = (int64_t) ((uint64_t) v23 * (uint64_t) v10);
    // pto: %t__tile
    ;
    Tile<TileType::Vec, bfloat16_t, 128, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v25 = Tile<TileType::Vec, bfloat16_t, 128, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v6);
    // pto: %t__tile
    ;
    uint64_t v26 = (uint64_t) v5;
    TASSIGN(v25, v26);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TEXPANDS(v25, v11);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %3
    ;
    int64_t v27 = v24 < v8 ? v8 : v24;
    // pto: %o_packed_heads_inline365__iter_v1_pview
    ;
    __gm__ bfloat16_t* v28 = PTOAS__GLOBAL_TENSOR_DATA(v20);
    // pto: %o_packed_heads_inline365__iter_v1_pview
    ;
    const int64_t v29 = 0;
    // pto: %o_packed_heads_inline365__iter_v1_pview
    ;
    const int64_t v30 = 512;
    // pto: %o_packed_heads_inline365__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 128, 512> v31 = pto::Shape<1, 1, 1, 128, 512>();
    // pto: %o_packed_heads_inline365__iter_v1_pview
    ;
    pto::Stride<65536, 65536, 65536, 512, 1> v32 = pto::Stride<65536, 65536, 65536, 512, 1>();
    // pto: %o_packed_heads_inline365__iter_v1_pview
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 512>, pto::Stride<65536, 65536, 65536, 512, 1>, pto::Layout::ND> v33 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 512>, pto::Stride<65536, 65536, 65536, 512, 1>, pto::Layout::ND>(v28 + (v29 + v27 * v30), v31, v32);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v33, v25);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}