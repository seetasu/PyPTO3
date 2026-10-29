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

AICORE void expert_scatter(__gm__ int32_t* v1, __gm__ int32_t* v2, __gm__ bfloat16_t* v3, __gm__ bfloat16_t* v4, int64_t v5, int64_t v6, int64_t v7, __gm__ int64_t* v8) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v9 = 0;
  // pto: %c2_index
  const int64_t v10 = 2;
  // pto: %c32_index
  const int64_t v11 = 32;
  // pto: %c1_index
  const int64_t v12 = 1;
  // pto: %c16_index
  const int64_t v13 = 16;
  // pto: %c48_index
  const int64_t v14 = 48;
  // pto: %c4096_index
  const int64_t v15 = 4096;
  // pto: %c0_index
  const int64_t v16 = 0;
  // pto: %c4_index
  const int64_t v17 = 4;
  // pto: %c2_i64
  const int64_t v18 = 2;
  // pto: %recv_y_tile_inline395__rv_v2_view
  const int64_t v19 = 1;
  // pto: %recv_y_tile_inline395__rv_v2_view
  const int64_t v20 = 1;
  // pto: %recv_y_tile_inline395__rv_v2_view
  const int64_t v21 = 1;
  // pto: %recv_y_tile_inline395__rv_v2_view
  int64_t v22 = v13 * v15;
  // pto: %recv_y_tile_inline395__rv_v2_view
  int64_t v23 = v21 * v22;
  // pto: %recv_y_tile_inline395__rv_v2_view
  pto::Shape<1, 1, 1, -1, -1> v24 = pto::Shape<1, 1, 1, -1, -1>(v19, v20, v21, v13, v15);
  // pto: %recv_y_tile_inline395__rv_v2_view
  pto::Stride<-1, -1, -1, -1, -1> v25 = pto::Stride<-1, -1, -1, -1, -1>(v20 * v23, v23, v22, v15, v12);
  // pto: %recv_y_tile_inline395__rv_v2_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v26 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v24, v25);
  // pto: %0
  int64_t v27 = (int64_t) ((uint64_t) v5 + (uint64_t) v6);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
  // pto: %source_begin_inline399__rv_v2
  int64_t v28;
  v28 = v16;
  for (int64_t v29 = v16; v29 < v10; v29 += v12) {
    // pto: %flat_offset_mul
    ;
    int64_t v30 = (int64_t) ((uint64_t) v29 * (uint64_t) v11);
    // pto: %flat_offset, %t__tile
    ;
    int32_t v31 = (v1)[(int64_t) ((uint64_t) v30 + (uint64_t) v7)];
    // pto: %2
    ;
    uint64_t v32 = (uint64_t) v28;
    // pto: %1, %2
    ;
    int64_t v33 = (int64_t) (v32 + (uint64_t) ((int64_t) v31));
    // pto: %3
    ;
    bool v34 = v5 < v28;
    // pto: %3
    ;
    int64_t v35 = v34 ? v28 : v5;
    // pto: %4
    ;
    for (int64_t v36 = v35; v36 < (v27 < v33 ? v27 : v33); v36 += v12) {
      // pto: %6
      ;
      int64_t v37 = (int64_t) ((uint64_t) v7 * (uint64_t) v13);
      // pto: %7, %5
      ;
      int32_t v38 = (v2)[(int64_t) ((uint64_t) v37 + (uint64_t) v36)];
      // pto: %8
      ;
      int64_t v39 = (int64_t) v38;
      // pto: %tput_stage
      ;
      Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v40 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v15);
      // pto: %tput_stage
      ;
      uint64_t v41 = (uint64_t) v9;
      TASSIGN(v40, v41);
      // pto: %9
      ;
      int64_t v42 = v39 < v16 ? v16 : v39;
      // pto: %10
      ;
      int64_t v43 = (int64_t) ((uint64_t) v36 - (uint64_t) v5);
      // pto: %11
      ;
      int64_t v44 = v43 < v16 ? v16 : v43;
      // pto: %12
      ;
      int64_t v45 = (v8)[v10];
      // pto: %13, %14, %15, %16
      ;
      int64_t v46 = (v8)[(int64_t) ((uint64_t) ((int64_t) (int32_t) v45) + (uint64_t) v17)];
      // pto: %17, %18
      ;
      int64_t v47 = (v8)[(int64_t) ((uint64_t) v29 + (uint64_t) v17)];
      // pto: %23
      ;
      const int64_t v48 = 1;
      // pto: %23
      ;
      const int64_t v49 = 1;
      // pto: %23
      ;
      const int64_t v50 = 1;
      // pto: %23
      ;
      int64_t v51 = v14 * v15;
      // pto: %23
      ;
      int64_t v52 = v50 * v51;
      // pto: %23
      ;
      pto::Shape<1, 1, 1, -1, -1> v53 = pto::Shape<1, 1, 1, -1, -1>(v48, v49, v50, v14, v15);
      // pto: %23
      ;
      pto::Stride<-1, -1, -1, -1, -1> v54 = pto::Stride<-1, -1, -1, -1, -1>(v49 * v52, v52, v51, v15, v12);
      // pto: %19, %20, %22, %23
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v55 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3 + (int64_t) ((uint64_t) v47 - (uint64_t) v46) / v18, v53, v54);
      // pto: %routed_y_buf__ssa_v0_peer_pview
      ;
      __gm__ bfloat16_t* v56 = PTOAS__GLOBAL_TENSOR_DATA(v55);
      // pto: %routed_y_buf__ssa_v0_peer_pview
      ;
      const int64_t v57 = 0;
      // pto: %routed_y_buf__ssa_v0_peer_pview
      ;
      const int64_t v58 = 4096;
      // pto: %routed_y_buf__ssa_v0_peer_pview
      ;
      pto::Shape<1, 1, 1, 1, 4096> v59 = pto::Shape<1, 1, 1, 1, 4096>();
      // pto: %routed_y_buf__ssa_v0_peer_pview
      ;
      pto::Stride<4096, 4096, 4096, 4096, 1> v60 = pto::Stride<4096, 4096, 4096, 4096, 1>();
      // pto: %routed_y_buf__ssa_v0_peer_pview
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v61 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v56 + (v57 + v42 * v58), v59, v60);
      // pto: %recv_y_tile_inline395__rv_v2_local_pview
      ;
      __gm__ bfloat16_t* v62 = PTOAS__GLOBAL_TENSOR_DATA(v26);
      // pto: %recv_y_tile_inline395__rv_v2_local_pview
      ;
      const int64_t v63 = 0;
      // pto: %recv_y_tile_inline395__rv_v2_local_pview
      ;
      const int64_t v64 = 4096;
      // pto: %recv_y_tile_inline395__rv_v2_local_pview
      ;
      pto::Shape<1, 1, 1, 1, 4096> v65 = pto::Shape<1, 1, 1, 1, 4096>();
      // pto: %recv_y_tile_inline395__rv_v2_local_pview
      ;
      pto::Stride<4096, 4096, 4096, 4096, 1> v66 = pto::Stride<4096, 4096, 4096, 4096, 1>();
      // pto: %recv_y_tile_inline395__rv_v2_local_pview
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v67 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v62 + (v63 + v44 * v64), v65, v66);
      pipe_barrier(PIPE_ALL);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      pipe_barrier(PIPE_MTE2);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
      pto::comm::TPUT(v61, v67, v40);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      pipe_barrier(PIPE_ALL);
      __gm__ bfloat16_t* v68 = PTOAS__GLOBAL_TENSOR_DATA(v61);
      PTOAS__DCCI_SINGLE_CACHE_LINE(v68);
      pipe_barrier(PIPE_ALL);
      dsb(DSB_DDR);
    };
    v28 = v33;
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}