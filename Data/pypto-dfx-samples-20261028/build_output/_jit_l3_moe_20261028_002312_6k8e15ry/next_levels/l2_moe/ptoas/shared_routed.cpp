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

AICORE void shared_routed(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, int64_t v4, __gm__ int64_t* v5, int32_t v6, int32_t v7) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c24576_i64
  const int64_t v8 = 24576;
  // pto: %c0_i64
  const int64_t v9 = 0;
  // pto: %c16384_i64
  const int64_t v10 = 16384;
  // pto: %c8_index
  const int64_t v11 = 8;
  // pto: %c4096_index
  const int64_t v12 = 4096;
  // pto: %c1_index
  const int64_t v13 = 1;
  // pto: %c48_index
  const int64_t v14 = 48;
  // pto: %c0_index
  const int64_t v15 = 0;
  // pto: %c6_index
  const int64_t v16 = 6;
  // pto: %sh_inline7__rv_v2_view
  const int64_t v17 = 1;
  // pto: %sh_inline7__rv_v2_view
  const int64_t v18 = 1;
  // pto: %sh_inline7__rv_v2_view
  const int64_t v19 = 1;
  // pto: %sh_inline7__rv_v2_view
  int64_t v20 = v11 * v12;
  // pto: %sh_inline7__rv_v2_view
  int64_t v21 = v19 * v20;
  // pto: %sh_inline7__rv_v2_view
  pto::Shape<1, 1, 1, -1, -1> v22 = pto::Shape<1, 1, 1, -1, -1>(v17, v18, v19, v11, v12);
  // pto: %sh_inline7__rv_v2_view
  pto::Stride<-1, -1, -1, -1, -1> v23 = pto::Stride<-1, -1, -1, -1, -1>(v18 * v21, v21, v20, v12, v13);
  // pto: %sh_inline7__rv_v2_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v24 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v22, v23);
  // pto: %routed_y_buf__ssa_v0_view
  const int64_t v25 = 1;
  // pto: %routed_y_buf__ssa_v0_view
  const int64_t v26 = 1;
  // pto: %routed_y_buf__ssa_v0_view
  const int64_t v27 = 1;
  // pto: %routed_y_buf__ssa_v0_view
  int64_t v28 = v14 * v12;
  // pto: %routed_y_buf__ssa_v0_view
  int64_t v29 = v27 * v28;
  // pto: %routed_y_buf__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v30 = pto::Shape<1, 1, 1, -1, -1>(v25, v26, v27, v14, v12);
  // pto: %routed_y_buf__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v31 = pto::Stride<-1, -1, -1, -1, -1>(v26 * v29, v29, v28, v12, v13);
  // pto: %routed_y_buf__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v32 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v30, v31);
  // pto: %ffn_out_inline16__ssa_v0_view
  const int64_t v33 = 1;
  // pto: %ffn_out_inline16__ssa_v0_view
  const int64_t v34 = 1;
  // pto: %ffn_out_inline16__ssa_v0_view
  const int64_t v35 = 1;
  // pto: %ffn_out_inline16__ssa_v0_view
  int64_t v36 = v11 * v12;
  // pto: %ffn_out_inline16__ssa_v0_view
  int64_t v37 = v35 * v36;
  // pto: %ffn_out_inline16__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v38 = pto::Shape<1, 1, 1, -1, -1>(v33, v34, v35, v11, v12);
  // pto: %ffn_out_inline16__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v39 = pto::Stride<-1, -1, -1, -1, -1>(v34 * v37, v37, v36, v12, v13);
  // pto: %ffn_out_inline16__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v40 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v38, v39);
  // pto: %t_inline521__ssa_v0
  int64_t v41 = (int64_t) v6;
  // pto: %5
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  if (v41 < v4) {
    // pto: %t__tile
    ;
    Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v42 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v12);
    // pto: %t__tile
    ;
    uint64_t v43 = (uint64_t) v8;
    TASSIGN(v42, v43);
    // pto: %6
    ;
    int64_t v44 = v41 < v15 ? v15 : v41;
    // pto: %sh_inline7__rv_v2_pview
    ;
    __gm__ bfloat16_t* v45 = PTOAS__GLOBAL_TENSOR_DATA(v24);
    // pto: %sh_inline7__rv_v2_pview
    ;
    const int64_t v46 = 0;
    // pto: %sh_inline7__rv_v2_pview
    ;
    const int64_t v47 = 4096;
    // pto: %sh_inline7__rv_v2_pview
    ;
    pto::Shape<1, 1, 1, 1, 4096> v48 = pto::Shape<1, 1, 1, 1, 4096>();
    // pto: %sh_inline7__rv_v2_pview
    ;
    pto::Stride<4096, 4096, 4096, 4096, 1> v49 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %sh_inline7__rv_v2_pview
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v50 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v45 + (v46 + v44 * v47), v48, v49);
    TLOAD(v42, v50);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %acc_inline522__tile
    ;
    Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v51 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v12);
    // pto: %acc_inline522__tile
    ;
    uint64_t v52 = (uint64_t) v9;
    TASSIGN(v51, v52);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    RoundMode v53 = RoundMode::CAST_ROUND;
    SaturationMode v54 = SaturationMode::OFF;
    TCVT(v51, v42, v53, v54);
    for (int64_t v55 = v15; v55 < v16; v55 += v13) {
      // pto: %7
      ;
      int64_t v56 = (int64_t) ((uint64_t) v41 * (uint64_t) v16);
      // pto: %8
      ;
      int64_t v57 = (int64_t) ((uint64_t) v56 + (uint64_t) v55);
      // pto: %0
      ;
      Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v58 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v12);
      // pto: %0
      ;
      uint64_t v59 = (uint64_t) v10;
      TASSIGN(v58, v59);
      // pto: %9
      ;
      int64_t v60 = v57 < v15 ? v15 : v57;
      // pto: %routed_y_buf__ssa_v0_pview
      ;
      __gm__ bfloat16_t* v61 = PTOAS__GLOBAL_TENSOR_DATA(v32);
      // pto: %routed_y_buf__ssa_v0_pview
      ;
      const int64_t v62 = 0;
      // pto: %routed_y_buf__ssa_v0_pview
      ;
      const int64_t v63 = 4096;
      // pto: %routed_y_buf__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 4096> v64 = pto::Shape<1, 1, 1, 1, 4096>();
      // pto: %routed_y_buf__ssa_v0_pview
      ;
      pto::Stride<4096, 4096, 4096, 4096, 1> v65 = pto::Stride<4096, 4096, 4096, 4096, 1>();
      // pto: %routed_y_buf__ssa_v0_pview
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v66 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v61 + (v62 + v60 * v63), v64, v65);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TLOAD(v58, v66);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      // pto: %1
      ;
      Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v67 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v12);
      // pto: %1
      ;
      uint64_t v68 = (uint64_t) v8;
      TASSIGN(v67, v68);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      pipe_barrier(PIPE_V);
      RoundMode v69 = RoundMode::CAST_ROUND;
      SaturationMode v70 = SaturationMode::OFF;
      TCVT(v67, v58, v69, v70);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      // pto: %2
      ;
      Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v71 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v12);
      // pto: %2
      ;
      uint64_t v72 = (uint64_t) v9;
      TASSIGN(v71, v72);
      pipe_barrier(PIPE_V);
      TADD(v71, v51, v67);
    };
    // pto: %3
    ;
    Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v73 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v12);
    // pto: %3
    ;
    uint64_t v74 = (uint64_t) v9;
    TASSIGN(v73, v74);
    pipe_barrier(PIPE_V);
    RoundMode v75 = RoundMode::CAST_RINT;
    SaturationMode v76 = SaturationMode::OFF;
    TCVT(v73, v51, v75, v76);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %ffn_out_inline16__ssa_v0_pview
    ;
    __gm__ bfloat16_t* v77 = PTOAS__GLOBAL_TENSOR_DATA(v40);
    // pto: %ffn_out_inline16__ssa_v0_pview
    ;
    const int64_t v78 = 0;
    // pto: %ffn_out_inline16__ssa_v0_pview
    ;
    const int64_t v79 = 4096;
    // pto: %ffn_out_inline16__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 1, 4096> v80 = pto::Shape<1, 1, 1, 1, 4096>();
    // pto: %ffn_out_inline16__ssa_v0_pview
    ;
    pto::Stride<4096, 4096, 4096, 4096, 1> v81 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %ffn_out_inline16__ssa_v0_pview
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v82 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v77 + (v78 + v44 * v79), v80, v81);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v82, v73);
  } else {
    // pto: %4
    ;
    Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v83 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v12);
    // pto: %4
    ;
    uint64_t v84 = (uint64_t) v9;
    TASSIGN(v83, v84);
    // pto: %11
    ;
    int64_t v85 = v41 < v15 ? v15 : v41;
    // pto: %12
    ;
    __gm__ bfloat16_t* v86 = PTOAS__GLOBAL_TENSOR_DATA(v24);
    // pto: %12
    ;
    const int64_t v87 = 0;
    // pto: %12
    ;
    const int64_t v88 = 4096;
    // pto: %12
    ;
    pto::Shape<1, 1, 1, 1, 4096> v89 = pto::Shape<1, 1, 1, 1, 4096>();
    // pto: %12
    ;
    pto::Stride<4096, 4096, 4096, 4096, 1> v90 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %12
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v91 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v86 + (v87 + v85 * v88), v89, v90);
    TLOAD(v83, v91);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    // pto: %15
    ;
    __gm__ bfloat16_t* v92 = PTOAS__GLOBAL_TENSOR_DATA(v40);
    // pto: %15
    ;
    const int64_t v93 = 0;
    // pto: %15
    ;
    const int64_t v94 = 4096;
    // pto: %15
    ;
    pto::Shape<1, 1, 1, 1, 4096> v95 = pto::Shape<1, 1, 1, 1, 4096>();
    // pto: %15
    ;
    pto::Stride<4096, 4096, 4096, 4096, 1> v96 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %15
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v97 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v92 + (v93 + v85 * v94), v95, v96);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v97, v83);
  }
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}