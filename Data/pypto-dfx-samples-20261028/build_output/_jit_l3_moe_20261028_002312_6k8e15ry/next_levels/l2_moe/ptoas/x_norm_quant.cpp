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

AICORE void x_norm_quant(__gm__ float* v1, __gm__ int8_t* v2, __gm__ float* v3, int64_t v4, int32_t v5, int32_t v6) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v7 = 0;
  // pto: %c32_i64
  const int64_t v8 = 32;
  // pto: %c8224_i64
  const int64_t v9 = 8224;
  // pto: %c1_index
  const int64_t v10 = 1;
  // pto: %c8_index
  const int64_t v11 = 8;
  // pto: %c4096_index
  const int64_t v12 = 4096;
  // pto: %c0_index
  const int64_t v13 = 0;
  // pto: %c512_index
  const int64_t v14 = 512;
  // pto: %c256_index
  const int64_t v15 = 256;
  // pto: %xn_scale_buf_inline172__ssa_v0_view
  const int64_t v16 = 1;
  // pto: %xn_scale_buf_inline172__ssa_v0_view
  const int64_t v17 = 1;
  // pto: %xn_scale_buf_inline172__ssa_v0_view
  const int64_t v18 = 1;
  // pto: %xn_scale_buf_inline172__ssa_v0_view
  int64_t v19 = (int64_t) v4;
  // pto: %xn_scale_buf_inline172__ssa_v0_view
  int64_t v20 = v19 * v10;
  // pto: %xn_scale_buf_inline172__ssa_v0_view
  int64_t v21 = v18 * v20;
  // pto: %xn_scale_buf_inline172__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v22 = pto::Shape<1, 1, 1, -1, -1>(v16, v17, v18, v19, v10);
  // pto: %xn_scale_buf_inline172__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v23 = pto::Stride<-1, -1, -1, -1, -1>(v17 * v21, v21, v20, v10, (int64_t) v4);
  // pto: %xn_scale_buf_inline172__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v24 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v1, v22, v23);
  // pto: %x_norm_i8_inline3__ssa_v0_view
  const int64_t v25 = 1;
  // pto: %x_norm_i8_inline3__ssa_v0_view
  const int64_t v26 = 1;
  // pto: %x_norm_i8_inline3__ssa_v0_view
  const int64_t v27 = 1;
  // pto: %x_norm_i8_inline3__ssa_v0_view
  int64_t v28 = v11 * v12;
  // pto: %x_norm_i8_inline3__ssa_v0_view
  int64_t v29 = v27 * v28;
  // pto: %x_norm_i8_inline3__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v30 = pto::Shape<1, 1, 1, -1, -1>(v25, v26, v27, v11, v12);
  // pto: %x_norm_i8_inline3__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v31 = pto::Stride<-1, -1, -1, -1, -1>(v26 * v29, v29, v28, v12, v10);
  // pto: %x_norm_i8_inline3__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v32 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v30, v31);
  // pto: %xg_buf_inline170__ssa_v0_view
  const int64_t v33 = 1;
  // pto: %xg_buf_inline170__ssa_v0_view
  const int64_t v34 = 1;
  // pto: %xg_buf_inline170__ssa_v0_view
  const int64_t v35 = 1;
  // pto: %xg_buf_inline170__ssa_v0_view
  int64_t v36 = (int64_t) v4;
  // pto: %xg_buf_inline170__ssa_v0_view
  int64_t v37 = v36 * v12;
  // pto: %xg_buf_inline170__ssa_v0_view
  int64_t v38 = v35 * v37;
  // pto: %xg_buf_inline170__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v39 = pto::Shape<1, 1, 1, -1, -1>(v33, v34, v35, v36, v12);
  // pto: %xg_buf_inline170__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v40 = pto::Stride<-1, -1, -1, -1, -1>(v34 * v38, v38, v37, v12, v10);
  // pto: %xg_buf_inline170__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v41 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v39, v40);
  // pto: %quant_block_inline209__ssa_v0, %6
  int64_t v42 = (int64_t) ((uint64_t) ((int64_t) v5) * (uint64_t) v11);
  // pto: %xn_sq_col_inline211__tile
  Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v43 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v10);
  // pto: %xn_sq_col_inline211__tile
  uint64_t v44 = (uint64_t) v7;
  TASSIGN(v43, v44);
  // pto: %7
  int64_t v45 = v42 < v13 ? v13 : v42;
  // pto: %xn_scale_buf_inline172__ssa_v0_pview
  int64_t v46 = (int64_t) v4;
  // pto: %xn_scale_buf_inline172__ssa_v0_pview
  const int64_t v47 = 0;
  // pto: %xn_scale_buf_inline172__ssa_v0_pview
  __gm__ float* v48 = PTOAS__GLOBAL_TENSOR_DATA(v24);
  // pto: %xn_scale_buf_inline172__ssa_v0_pview
  const int64_t v49 = 1;
  // pto: %xn_scale_buf_inline172__ssa_v0_pview
  const int64_t v50 = 1;
  // pto: %xn_scale_buf_inline172__ssa_v0_pview
  const int64_t v51 = 1;
  // pto: %xn_scale_buf_inline172__ssa_v0_pview
  int64_t v52 = v11 * v10;
  // pto: %xn_scale_buf_inline172__ssa_v0_pview
  int64_t v53 = v51 * v52;
  // pto: %xn_scale_buf_inline172__ssa_v0_pview
  pto::Shape<1, 1, 1, 8, 1> v54 = pto::Shape<1, 1, 1, 8, 1>(v49, v50, v51, v11, v10);
  // pto: %xn_scale_buf_inline172__ssa_v0_pview
  pto::Stride<-1, -1, -1, -1, -1> v55 = pto::Stride<-1, -1, -1, -1, -1>(v50 * v53, v53, v52, v10, v46);
  // pto: %xn_scale_buf_inline172__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v56 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v48 + (v47 + v45 * v10 + v13 * v46), v54, v55);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  TLOAD(v43, v56);
  for (int64_t v57 = v13; v57 < v12; v57 += v14) {
    // pto: %t__tile
    ;
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v58 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v15);
    // pto: %t__tile
    ;
    uint64_t v59 = (uint64_t) v8;
    TASSIGN(v58, v59);
    // pto: %9
    ;
    int64_t v60 = v57 < v13 ? v13 : v57;
    // pto: %xg_buf_inline170__ssa_v0_pview
    ;
    __gm__ float* v61 = PTOAS__GLOBAL_TENSOR_DATA(v41);
    // pto: %xg_buf_inline170__ssa_v0_pview
    ;
    const int64_t v62 = 0;
    // pto: %xg_buf_inline170__ssa_v0_pview
    ;
    const int64_t v63 = 4096;
    // pto: %xg_buf_inline170__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 8, 256> v64 = pto::Shape<1, 1, 1, 8, 256>();
    // pto: %xg_buf_inline170__ssa_v0_pview
    ;
    pto::Stride<32768, 32768, 32768, 4096, 1> v65 = pto::Stride<32768, 32768, 32768, 4096, 1>();
    // pto: %xg_buf_inline170__ssa_v0_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND> v66 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>(v61 + (v62 + v45 * v63 + v60), v64, v65);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v58, v66);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %0
    ;
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v67 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v15);
    // pto: %0
    ;
    uint64_t v68 = (uint64_t) v9;
    TASSIGN(v67, v68);
    // pto: %11
    ;
    int64_t v69 = (int64_t) ((uint64_t) v57 + (uint64_t) v15);
    // pto: %12
    ;
    int64_t v70 = v69 < v13 ? v13 : v69;
    // pto: %13
    ;
    __gm__ float* v71 = PTOAS__GLOBAL_TENSOR_DATA(v41);
    // pto: %13
    ;
    const int64_t v72 = 0;
    // pto: %13
    ;
    const int64_t v73 = 4096;
    // pto: %13
    ;
    pto::Shape<1, 1, 1, 8, 256> v74 = pto::Shape<1, 1, 1, 8, 256>();
    // pto: %13
    ;
    pto::Stride<32768, 32768, 32768, 4096, 1> v75 = pto::Stride<32768, 32768, 32768, 4096, 1>();
    // pto: %13
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND> v76 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>(v71 + (v72 + v45 * v73 + v70), v74, v75);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    TLOAD(v67, v76);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    // pto: %xn_q_scaled_inline213__tile
    ;
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v77 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v15);
    // pto: %xn_q_scaled_inline213__tile
    ;
    uint64_t v78 = (uint64_t) v8;
    TASSIGN(v77, v78);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TROWEXPANDMUL(v77, v58, v43);
    // pto: %xn_q_i32_inline214__tile
    ;
    Tile<TileType::Vec, int32_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v79 = Tile<TileType::Vec, int32_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v15);
    // pto: %xn_q_i32_inline214__tile
    ;
    uint64_t v80 = (uint64_t) v8;
    TASSIGN(v79, v80);
    pipe_barrier(PIPE_V);
    RoundMode v81 = RoundMode::CAST_RINT;
    SaturationMode v82 = SaturationMode::ON;
    TCVT(v79, v77, v81, v82);
    // pto: %xn_q_half_inline215__tile
    ;
    Tile<TileType::Vec, half, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v83 = Tile<TileType::Vec, half, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v15);
    // pto: %xn_q_half_inline215__tile
    ;
    uint64_t v84 = (uint64_t) v8;
    TASSIGN(v83, v84);
    pipe_barrier(PIPE_V);
    RoundMode v85 = RoundMode::CAST_ROUND;
    SaturationMode v86 = SaturationMode::OFF;
    TCVT(v83, v79, v85, v86);
    // pto: %1
    ;
    Tile<TileType::Vec, int8_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v87 = Tile<TileType::Vec, int8_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v15);
    // pto: %1
    ;
    uint64_t v88 = (uint64_t) v8;
    TASSIGN(v87, v88);
    pipe_barrier(PIPE_V);
    RoundMode v89 = RoundMode::CAST_TRUNC;
    SaturationMode v90 = SaturationMode::ON;
    TCVT(v87, v83, v89, v90);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %x_norm_i8_inline3__iter_v1_pview
    ;
    __gm__ int8_t* v91 = PTOAS__GLOBAL_TENSOR_DATA(v32);
    // pto: %x_norm_i8_inline3__iter_v1_pview
    ;
    const int64_t v92 = 0;
    // pto: %x_norm_i8_inline3__iter_v1_pview
    ;
    const int64_t v93 = 4096;
    // pto: %x_norm_i8_inline3__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 8, 256> v94 = pto::Shape<1, 1, 1, 8, 256>();
    // pto: %x_norm_i8_inline3__iter_v1_pview
    ;
    pto::Stride<32768, 32768, 32768, 4096, 1> v95 = pto::Stride<32768, 32768, 32768, 4096, 1>();
    // pto: %x_norm_i8_inline3__iter_v1_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND> v96 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>(v91 + (v92 + v45 * v93 + v60), v94, v95);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v96, v87);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    // pto: %2
    ;
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v97 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v15);
    // pto: %2
    ;
    uint64_t v98 = (uint64_t) v9;
    TASSIGN(v97, v98);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TROWEXPANDMUL(v97, v67, v43);
    // pto: %3
    ;
    Tile<TileType::Vec, int32_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v99 = Tile<TileType::Vec, int32_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v15);
    // pto: %3
    ;
    uint64_t v100 = (uint64_t) v9;
    TASSIGN(v99, v100);
    pipe_barrier(PIPE_V);
    RoundMode v101 = RoundMode::CAST_RINT;
    SaturationMode v102 = SaturationMode::ON;
    TCVT(v99, v97, v101, v102);
    // pto: %4
    ;
    Tile<TileType::Vec, half, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v103 = Tile<TileType::Vec, half, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v15);
    // pto: %4
    ;
    uint64_t v104 = (uint64_t) v9;
    TASSIGN(v103, v104);
    pipe_barrier(PIPE_V);
    RoundMode v105 = RoundMode::CAST_ROUND;
    SaturationMode v106 = SaturationMode::OFF;
    TCVT(v103, v99, v105, v106);
    // pto: %5
    ;
    Tile<TileType::Vec, int8_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v107 = Tile<TileType::Vec, int8_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v15);
    // pto: %5
    ;
    uint64_t v108 = (uint64_t) v9;
    TASSIGN(v107, v108);
    pipe_barrier(PIPE_V);
    RoundMode v109 = RoundMode::CAST_TRUNC;
    SaturationMode v110 = SaturationMode::ON;
    TCVT(v107, v103, v109, v110);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    // pto: %x_norm_i8_inline3__tile_pview
    ;
    __gm__ int8_t* v111 = PTOAS__GLOBAL_TENSOR_DATA(v32);
    // pto: %x_norm_i8_inline3__tile_pview
    ;
    const int64_t v112 = 0;
    // pto: %x_norm_i8_inline3__tile_pview
    ;
    const int64_t v113 = 4096;
    // pto: %x_norm_i8_inline3__tile_pview
    ;
    pto::Shape<1, 1, 1, 8, 256> v114 = pto::Shape<1, 1, 1, 8, 256>();
    // pto: %x_norm_i8_inline3__tile_pview
    ;
    pto::Stride<32768, 32768, 32768, 4096, 1> v115 = pto::Stride<32768, 32768, 32768, 4096, 1>();
    // pto: %x_norm_i8_inline3__tile_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND> v116 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>(v111 + (v112 + v45 * v113 + v70), v114, v115);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v116, v107);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}