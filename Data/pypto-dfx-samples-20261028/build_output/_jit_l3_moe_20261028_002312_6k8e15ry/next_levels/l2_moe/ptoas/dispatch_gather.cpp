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

AICORE void dispatch_gather(__gm__ int8_t* v1, __gm__ int32_t* v2, __gm__ int8_t* v3, __gm__ float* v4, __gm__ float* v5, __gm__ float* v6, __gm__ int32_t* v7, __gm__ int64_t* v8, __gm__ int64_t* v9, int32_t v10, int32_t v11) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v12 = 0;
  // pto: %c512_index
  const int64_t v13 = 512;
  // pto: %c4096_index
  const int64_t v14 = 4096;
  // pto: %c1_index
  const int64_t v15 = 1;
  // pto: %c2_index
  const int64_t v16 = 2;
  // pto: %c32_index
  const int64_t v17 = 32;
  // pto: %c8_index
  const int64_t v18 = 8;
  // pto: %c16_index
  const int64_t v19 = 16;
  // pto: %c0_index
  const int64_t v20 = 0;
  // pto: %recv_x_out_flat_inline335__ssa_v0_view
  const int64_t v21 = 1;
  // pto: %recv_x_out_flat_inline335__ssa_v0_view
  const int64_t v22 = 1;
  // pto: %recv_x_out_flat_inline335__ssa_v0_view
  const int64_t v23 = 1;
  // pto: %recv_x_out_flat_inline335__ssa_v0_view
  int64_t v24 = v13 * v14;
  // pto: %recv_x_out_flat_inline335__ssa_v0_view
  int64_t v25 = v23 * v24;
  // pto: %recv_x_out_flat_inline335__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v26 = pto::Shape<1, 1, 1, -1, -1>(v21, v22, v23, v13, v14);
  // pto: %recv_x_out_flat_inline335__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v27 = pto::Stride<-1, -1, -1, -1, -1>(v22 * v25, v25, v24, v14, v15);
  // pto: %recv_x_out_flat_inline335__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v28 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v26, v27);
  // pto: %recv_x__ssa_v0_view
  const int64_t v29 = 1;
  // pto: %recv_x__ssa_v0_view
  const int64_t v30 = 1;
  // pto: %recv_x__ssa_v0_view
  const int64_t v31 = 1;
  // pto: %recv_x__ssa_v0_view
  int64_t v32 = v13 * v14;
  // pto: %recv_x__ssa_v0_view
  int64_t v33 = v31 * v32;
  // pto: %recv_x__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v34 = pto::Shape<1, 1, 1, -1, -1>(v29, v30, v31, v13, v14);
  // pto: %recv_x__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v35 = pto::Stride<-1, -1, -1, -1, -1>(v30 * v33, v33, v32, v14, v15);
  // pto: %recv_x__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v36 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v34, v35);
  // pto: %e_inline361__ssa_v1
  int64_t v37 = (int64_t) v10;
  // pto: %0
  int64_t v38 = (int64_t) ((uint64_t) v37 * (uint64_t) v19);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  // pto: %b_inline378__rv_v2
  int64_t v39;
  v39 = v20;
  for (int64_t v40 = v20; v40 < v16; v40 += v15) {
    // pto: %flat_offset_mul
    ;
    int64_t v41 = (int64_t) ((uint64_t) v40 * (uint64_t) v17);
    // pto: %flat_offset, %1
    ;
    int32_t v42 = (v2)[(int64_t) ((uint64_t) v41 + (uint64_t) v37)];
    // pto: %2
    ;
    int64_t v43 = (int64_t) v42;
    // pto: %3
    ;
    int64_t v44 = (int64_t) ((uint64_t) v40 * (uint64_t) v18);
    // pto: %4
    ;
    int64_t v45 = (int64_t) ((uint64_t) v38 + (uint64_t) v44);
    for (int64_t v46 = v20; v46 < v43; v46 += v15) {
      // pto: %5
      ;
      int64_t v47 = (int64_t) ((uint64_t) v45 + (uint64_t) v46);
      // pto: %6
      ;
      uint64_t v48 = (uint64_t) v39;
      // pto: %6
      ;
      int64_t v49 = (int64_t) (v48 + (uint64_t) v46);
      // pto: %7
      ;
      int64_t v50 = (int64_t) ((uint64_t) v38 + (uint64_t) v49);
      // pto: %t__tile
      ;
      Tile<TileType::Vec, int8_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v51 = Tile<TileType::Vec, int8_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v14);
      // pto: %t__tile
      ;
      uint64_t v52 = (uint64_t) v12;
      TASSIGN(v51, v52);
      // pto: %8
      ;
      int64_t v53 = v47 < v20 ? v20 : v47;
      // pto: %recv_x__ssa_v0_pview
      ;
      __gm__ int8_t* v54 = PTOAS__GLOBAL_TENSOR_DATA(v36);
      // pto: %recv_x__ssa_v0_pview
      ;
      const int64_t v55 = 0;
      // pto: %recv_x__ssa_v0_pview
      ;
      const int64_t v56 = 4096;
      // pto: %recv_x__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 4096> v57 = pto::Shape<1, 1, 1, 1, 4096>();
      // pto: %recv_x__ssa_v0_pview
      ;
      pto::Stride<4096, 4096, 4096, 4096, 1> v58 = pto::Stride<4096, 4096, 4096, 4096, 1>();
      // pto: %recv_x__ssa_v0_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v59 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v54 + (v55 + v53 * v56), v57, v58);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      TLOAD(v51, v59);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      // pto: %9
      ;
      int64_t v60 = v50 < v20 ? v20 : v50;
      // pto: %recv_x_out_flat_inline335__iter_v3_pview
      ;
      __gm__ int8_t* v61 = PTOAS__GLOBAL_TENSOR_DATA(v28);
      // pto: %recv_x_out_flat_inline335__iter_v3_pview
      ;
      const int64_t v62 = 0;
      // pto: %recv_x_out_flat_inline335__iter_v3_pview
      ;
      const int64_t v63 = 4096;
      // pto: %recv_x_out_flat_inline335__iter_v3_pview
      ;
      pto::Shape<1, 1, 1, 1, 4096> v64 = pto::Shape<1, 1, 1, 1, 4096>();
      // pto: %recv_x_out_flat_inline335__iter_v3_pview
      ;
      pto::Stride<4096, 4096, 4096, 4096, 1> v65 = pto::Stride<4096, 4096, 4096, 4096, 1>();
      // pto: %recv_x_out_flat_inline335__iter_v3_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v66 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v61 + (v62 + v60 * v63), v64, v65);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      TSTORE(v66, v51);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      // pto: %11
      ;
      int64_t v67 = (int64_t) ((uint64_t) v47 * (uint64_t) v18);
      // pto: %10
      ;
      float v68 = (v4)[v67];
      (v5)[v50] = v68;
      // pto: %17, %15
      ;
      float v69 = (v4)[(int64_t) ((uint64_t) v67 + (uint64_t) v15)];
      (v6)[v50] = v69;
      // pto: %21, %gather_route_inline386__tile
      ;
      float v70 = (v4)[(int64_t) ((uint64_t) v67 + (uint64_t) v16)];
      // pto: %22
      ;
      int32_t v71 = (int32_t) v70;
      (v7)[v50] = v71;
    };
    // pto: %25
    ;
    uint64_t v72 = (uint64_t) v39;
    // pto: %25
    ;
    v39 = (int64_t) (v72 + (uint64_t) v43);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
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