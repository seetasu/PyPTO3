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

AICORE void kv_touch(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, int64_t v3, int64_t v4) {
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
  // pto: %c128_index
  const int64_t v8 = 128;
  // pto: %3
  int64_t v9 = (int64_t) ((uint64_t) v3 * (uint64_t) v8);
  // pto: %ori_kv_flat__ssa_v0_view
  const int64_t v10 = 1;
  // pto: %ori_kv_flat__ssa_v0_view
  const int64_t v11 = 1;
  // pto: %ori_kv_flat__ssa_v0_view
  const int64_t v12 = 1;
  // pto: %ori_kv_flat__ssa_v0_view
  int64_t v13 = v9 * v6;
  // pto: %ori_kv_flat__ssa_v0_view
  int64_t v14 = v12 * v13;
  // pto: %ori_kv_flat__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v15 = pto::Shape<1, 1, 1, -1, -1>(v10, v11, v12, v9, v6);
  // pto: %ori_kv_flat__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v16 = pto::Stride<-1, -1, -1, -1, -1>(v11 * v14, v14, v13, v6, v7);
  // pto: %ori_kv_flat__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v17 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v15, v16);
  // pto: %cmp_kv_flat__ssa_v0_view
  const int64_t v18 = 1;
  // pto: %cmp_kv_flat__ssa_v0_view
  const int64_t v19 = 1;
  // pto: %cmp_kv_flat__ssa_v0_view
  const int64_t v20 = 1;
  // pto: %cmp_kv_flat__ssa_v0_view
  int64_t v21 = (int64_t) v4;
  // pto: %cmp_kv_flat__ssa_v0_view
  int64_t v22 = v21 * v6;
  // pto: %cmp_kv_flat__ssa_v0_view
  int64_t v23 = v20 * v22;
  // pto: %cmp_kv_flat__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v24 = pto::Shape<1, 1, 1, -1, -1>(v18, v19, v20, v21, v6);
  // pto: %cmp_kv_flat__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v25 = pto::Stride<-1, -1, -1, -1, -1>(v19 * v23, v23, v22, v6, v7);
  // pto: %cmp_kv_flat__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v26 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v24, v25);
  // pto: %t__tile
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v27 = Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v7, v6);
  // pto: %t__tile
  uint64_t v28 = (uint64_t) v5;
  TASSIGN(v27, v28);
  // pto: %ori_kv_flat__ssa_v0_pview
  __gm__ bfloat16_t* v29 = PTOAS__GLOBAL_TENSOR_DATA(v17);
  // pto: %ori_kv_flat__ssa_v0_pview
  pto::Shape<1, 1, 1, 1, 512> v30 = pto::Shape<1, 1, 1, 1, 512>();
  // pto: %ori_kv_flat__ssa_v0_pview
  pto::Stride<512, 512, 512, 512, 1> v31 = pto::Stride<512, 512, 512, 512, 1>();
  // pto: %ori_kv_flat__ssa_v0_pview
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v32 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v29, v30, v31);
  TLOAD(v27, v32);
  set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  TSTORE(v32, v27);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  // pto: %0
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v33 = Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v7, v6);
  // pto: %0
  uint64_t v34 = (uint64_t) v5;
  TASSIGN(v33, v34);
  // pto: %cmp_kv_flat__ssa_v0_pview
  __gm__ bfloat16_t* v35 = PTOAS__GLOBAL_TENSOR_DATA(v26);
  // pto: %cmp_kv_flat__ssa_v0_pview
  pto::Shape<1, 1, 1, 1, 512> v36 = pto::Shape<1, 1, 1, 1, 512>();
  // pto: %cmp_kv_flat__ssa_v0_pview
  pto::Stride<512, 512, 512, 512, 1> v37 = pto::Stride<512, 512, 512, 512, 1>();
  // pto: %cmp_kv_flat__ssa_v0_pview
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v38 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v35, v36, v37);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  TLOAD(v33, v38);
  set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
  wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
  TSTORE(v38, v33);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}