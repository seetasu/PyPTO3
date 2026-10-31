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

AICORE void gate_pre_route(__gm__ float* v1, __gm__ int32_t* v2, __gm__ float* v3, __gm__ float* v4, int64_t v5, int64_t v6, int64_t v7, int64_t v8) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v9 = 0;
  // pto: %c1_index
  const int64_t v10 = 1;
  // pto: %c6_index
  const int64_t v11 = 6;
  // pto: %c256_index
  const int64_t v12 = 256;
  // pto: %c0_index
  const int64_t v13 = 0;
  // pto: %cst_6
  const float v14 = 0.0f;
  // pto: %c0_i32
  const int32_t v15 = 0;
  // pto: %c16_index
  const int64_t v16 = 16;
  // pto: %c192_index
  const int64_t v17 = 192;
  // pto: %cst_10
  const float v18 = -3.40282347E+38f;
  // pto: %biased_scores_buf_inline174__ssa_v0_view
  const int64_t v19 = 1;
  // pto: %biased_scores_buf_inline174__ssa_v0_view
  const int64_t v20 = 1;
  // pto: %biased_scores_buf_inline174__ssa_v0_view
  const int64_t v21 = 1;
  // pto: %biased_scores_buf_inline174__ssa_v0_view
  int64_t v22 = (int64_t) v8;
  // pto: %biased_scores_buf_inline174__ssa_v0_view
  int64_t v23 = v22 * v12;
  // pto: %biased_scores_buf_inline174__ssa_v0_view
  int64_t v24 = v21 * v23;
  // pto: %biased_scores_buf_inline174__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v25 = pto::Shape<1, 1, 1, -1, -1>(v19, v20, v21, v22, v12);
  // pto: %biased_scores_buf_inline174__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v26 = pto::Stride<-1, -1, -1, -1, -1>(v20 * v24, v24, v23, v12, v10);
  // pto: %biased_scores_buf_inline174__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v27 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v25, v26);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  for (int64_t v28 = v13; v28 < v5; v28 += v10) {
    // pto: %0
    ;
    if (v6 <= v28) {
      (v1)[v28] = v14;
      for (int64_t v29 = v13; v29 < v11; v29 += v10) {
        // pto: %2
        ;
        int64_t v30 = (int64_t) ((uint64_t) v28 * (uint64_t) v11);
        // pto: %3
        ;
        int64_t v31 = (int64_t) ((uint64_t) v30 + (uint64_t) v29);
        (v2)[v31] = v15;
        (v3)[v31] = v14;
      };
    };
  }
  // pto: %7
  for (int64_t v32 = v13; v32 < (v8 / v16); v32 += v10) {
    // pto: %8
    ;
    int64_t v33 = (int64_t) ((uint64_t) v32 * (uint64_t) v16);
    // pto: %t__tile
    ;
    Tile<TileType::Vec, float, 16, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v34 = Tile<TileType::Vec, float, 16, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v17);
    // pto: %t__tile
    ;
    uint64_t v35 = (uint64_t) v9;
    TASSIGN(v34, v35);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TEXPANDS(v34, v18);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %9
    ;
    int64_t v36 = v33 < v13 ? v13 : v33;
    // pto: %biased_scores_buf_inline174__iter_v1_pview
    ;
    __gm__ float* v37 = PTOAS__GLOBAL_TENSOR_DATA(v27);
    // pto: %biased_scores_buf_inline174__iter_v1_pview
    ;
    const int64_t v38 = 64;
    // pto: %biased_scores_buf_inline174__iter_v1_pview
    ;
    const int64_t v39 = 256;
    // pto: %biased_scores_buf_inline174__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 16, 192> v40 = pto::Shape<1, 1, 1, 16, 192>();
    // pto: %biased_scores_buf_inline174__iter_v1_pview
    ;
    pto::Stride<4096, 4096, 4096, 256, 1> v41 = pto::Stride<4096, 4096, 4096, 256, 1>();
    // pto: %biased_scores_buf_inline174__iter_v1_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 192>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND> v42 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 192>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND>(v37 + (v38 + v36 * v39), v40, v41);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v42, v34);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  pipe_barrier(PIPE_ALL);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  dsb((mem_dsb_t)0);
  #endif // __DAV_VEC__

  pipe_barrier(PIPE_ALL);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  dsb((mem_dsb_t)0);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}