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

AICORE void hc_pre_linear_reduce(__gm__ float* v1, __gm__ float* v2, int64_t v3, int64_t v4, int64_t v5, int32_t v6, int32_t v7) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v8 = 0;
  // pto: %c2048_i64
  const int64_t v9 = 2048;
  // pto: %c32_index
  const int64_t v10 = 32;
  // pto: %c1_index
  const int64_t v11 = 1;
  // pto: %c16_index
  const int64_t v12 = 16;
  // pto: %c0_index
  const int64_t v13 = 0;
  // pto: %c4_index
  const int64_t v14 = 4;
  // pto: %8
  int64_t v15 = (int64_t) ((uint64_t) v4 * (uint64_t) v14);
  // pto: %mixes_partials_inline64__rv_v2_view
  const int64_t v16 = 1;
  // pto: %mixes_partials_inline64__rv_v2_view
  const int64_t v17 = 1;
  // pto: %mixes_partials_inline64__rv_v2_view
  const int64_t v18 = 1;
  // pto: %mixes_partials_inline64__rv_v2_view
  int64_t v19 = v15 * v10;
  // pto: %mixes_partials_inline64__rv_v2_view
  int64_t v20 = v18 * v19;
  // pto: %mixes_partials_inline64__rv_v2_view
  pto::Shape<1, 1, 1, -1, -1> v21 = pto::Shape<1, 1, 1, -1, -1>(v16, v17, v18, v15, v10);
  // pto: %mixes_partials_inline64__rv_v2_view
  pto::Stride<-1, -1, -1, -1, -1> v22 = pto::Stride<-1, -1, -1, -1, -1>(v17 * v20, v20, v19, v10, v11);
  // pto: %mixes_partials_inline64__rv_v2_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v23 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v21, v22);
  // pto: %hc_mixes_raw_inline21__ssa_v0_view
  const int64_t v24 = 1;
  // pto: %hc_mixes_raw_inline21__ssa_v0_view
  const int64_t v25 = 1;
  // pto: %hc_mixes_raw_inline21__ssa_v0_view
  const int64_t v26 = 1;
  // pto: %hc_mixes_raw_inline21__ssa_v0_view
  int64_t v27 = (int64_t) v5;
  // pto: %hc_mixes_raw_inline21__ssa_v0_view
  int64_t v28 = v27 * v10;
  // pto: %hc_mixes_raw_inline21__ssa_v0_view
  int64_t v29 = v26 * v28;
  // pto: %hc_mixes_raw_inline21__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v30 = pto::Shape<1, 1, 1, -1, -1>(v24, v25, v26, v27, v10);
  // pto: %hc_mixes_raw_inline21__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v31 = pto::Stride<-1, -1, -1, -1, -1>(v25 * v29, v29, v28, v10, v11);
  // pto: %hc_mixes_raw_inline21__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v32 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v30, v31);
  // pto: %linear_block_inline77__ssa_v0, %1
  int64_t v33 = (int64_t) ((uint64_t) ((int64_t) v6) * (uint64_t) v12);
  // pto: %mixes_total_inline79__tile
  Tile<TileType::Vec, float, 16, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v34 = Tile<TileType::Vec, float, 16, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v10);
  // pto: %mixes_total_inline79__tile
  uint64_t v35 = (uint64_t) v8;
  TASSIGN(v34, v35);
  // pto: %2
  int64_t v36 = v33 < v13 ? v13 : v33;
  // pto: %mixes_partials_inline64__rv_v2_pview
  __gm__ float* v37 = PTOAS__GLOBAL_TENSOR_DATA(v23);
  // pto: %mixes_partials_inline64__rv_v2_pview
  const int64_t v38 = 0;
  // pto: %mixes_partials_inline64__rv_v2_pview
  const int64_t v39 = 32;
  // pto: %mixes_partials_inline64__rv_v2_pview
  pto::Shape<1, 1, 1, 16, 32> v40 = pto::Shape<1, 1, 1, 16, 32>();
  // pto: %mixes_partials_inline64__rv_v2_pview
  pto::Stride<512, 512, 512, 32, 1> v41 = pto::Stride<512, 512, 512, 32, 1>();
  // pto: %mixes_partials_inline64__rv_v2_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 32>, pto::Stride<512, 512, 512, 32, 1>, pto::Layout::ND> v42 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 32>, pto::Stride<512, 512, 512, 32, 1>, pto::Layout::ND>(v37 + (v38 + v36 * v39), v40, v41);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  TLOAD(v34, v42);
  set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  for (int64_t v43 = v11; v43 < v14; v43 += v11) {
    // pto: %3
    ;
    int64_t v44 = (int64_t) ((uint64_t) v43 * (uint64_t) v4);
    // pto: %4
    ;
    int64_t v45 = (int64_t) ((uint64_t) v44 + (uint64_t) v33);
    // pto: %partial_tile_inline82__tile
    ;
    Tile<TileType::Vec, float, 16, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v46 = Tile<TileType::Vec, float, 16, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v10);
    // pto: %partial_tile_inline82__tile
    ;
    uint64_t v47 = (uint64_t) v9;
    TASSIGN(v46, v47);
    // pto: %5
    ;
    int64_t v48 = v45 < v13 ? v13 : v45;
    // pto: %6
    ;
    __gm__ float* v49 = PTOAS__GLOBAL_TENSOR_DATA(v23);
    // pto: %6
    ;
    const int64_t v50 = 0;
    // pto: %6
    ;
    const int64_t v51 = 32;
    // pto: %6
    ;
    pto::Shape<1, 1, 1, 16, 32> v52 = pto::Shape<1, 1, 1, 16, 32>();
    // pto: %6
    ;
    pto::Stride<512, 512, 512, 32, 1> v53 = pto::Stride<512, 512, 512, 32, 1>();
    // pto: %6
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 32>, pto::Stride<512, 512, 512, 32, 1>, pto::Layout::ND> v54 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 32>, pto::Stride<512, 512, 512, 32, 1>, pto::Layout::ND>(v49 + (v50 + v48 * v51), v52, v53);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v46, v54);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %0
    ;
    Tile<TileType::Vec, float, 16, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v55 = Tile<TileType::Vec, float, 16, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v10);
    // pto: %0
    ;
    uint64_t v56 = (uint64_t) v8;
    TASSIGN(v55, v56);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADD(v55, v34, v46);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  }
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  // pto: %hc_mixes_raw_inline21__ssa_v0_pview
  __gm__ float* v57 = PTOAS__GLOBAL_TENSOR_DATA(v32);
  // pto: %hc_mixes_raw_inline21__ssa_v0_pview
  const int64_t v58 = 0;
  // pto: %hc_mixes_raw_inline21__ssa_v0_pview
  const int64_t v59 = 32;
  // pto: %hc_mixes_raw_inline21__ssa_v0_pview
  pto::Shape<1, 1, 1, 16, 32> v60 = pto::Shape<1, 1, 1, 16, 32>();
  // pto: %hc_mixes_raw_inline21__ssa_v0_pview
  pto::Stride<512, 512, 512, 32, 1> v61 = pto::Stride<512, 512, 512, 32, 1>();
  // pto: %hc_mixes_raw_inline21__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 32>, pto::Stride<512, 512, 512, 32, 1>, pto::Layout::ND> v62 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 32>, pto::Stride<512, 512, 512, 32, 1>, pto::Layout::ND>(v57 + (v58 + v36 * v59), v60, v61);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  TSTORE(v62, v34);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}