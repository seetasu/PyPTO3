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

AICORE void exp_h_q(__gm__ float* v1, __gm__ float* v2, __gm__ int8_t* v3) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c32832_i64
  const int64_t v4 = 32832;
  // pto: %c32896_i64
  const int64_t v5 = 32896;
  // pto: %c65664_i64
  const int64_t v6 = 65664;
  // pto: %c98432_i64
  const int64_t v7 = 98432;
  // pto: %c131200_i64
  const int64_t v8 = 131200;
  // pto: %c0_i64
  const int64_t v9 = 0;
  // pto: %c32768_i64
  const int64_t v10 = 32768;
  // pto: %c16_index
  const int64_t v11 = 16;
  // pto: %c2048_index
  const int64_t v12 = 2048;
  // pto: %c1_index
  const int64_t v13 = 1;
  // pto: %cst_10
  const float v14 = 9.99999974E-5f;
  // pto: %c0_index
  const int64_t v15 = 0;
  // pto: %c1024_index
  const int64_t v16 = 1024;
  // pto: %c512_index
  const int64_t v17 = 512;
  // pto: %cst_14
  const float v18 = 127.0f;
  // pto: %h_tile_fp32_inline464__rv_v2_view
  const int64_t v19 = 1;
  // pto: %h_tile_fp32_inline464__rv_v2_view
  const int64_t v20 = 1;
  // pto: %h_tile_fp32_inline464__rv_v2_view
  const int64_t v21 = 1;
  // pto: %h_tile_fp32_inline464__rv_v2_view
  int64_t v22 = v11 * v12;
  // pto: %h_tile_fp32_inline464__rv_v2_view
  int64_t v23 = v21 * v22;
  // pto: %h_tile_fp32_inline464__rv_v2_view
  pto::Shape<1, 1, 1, -1, -1> v24 = pto::Shape<1, 1, 1, -1, -1>(v19, v20, v21, v11, v12);
  // pto: %h_tile_fp32_inline464__rv_v2_view
  pto::Stride<-1, -1, -1, -1, -1> v25 = pto::Stride<-1, -1, -1, -1, -1>(v20 * v23, v23, v22, v12, v13);
  // pto: %h_tile_fp32_inline464__rv_v2_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v26 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v24, v25);
  // pto: %h_tile_scale_dq_inline445__ssa_v0_view
  const int64_t v27 = 1;
  // pto: %h_tile_scale_dq_inline445__ssa_v0_view
  const int64_t v28 = 1;
  // pto: %h_tile_scale_dq_inline445__ssa_v0_view
  const int64_t v29 = 1;
  // pto: %h_tile_scale_dq_inline445__ssa_v0_view
  int64_t v30 = v11 * v13;
  // pto: %h_tile_scale_dq_inline445__ssa_v0_view
  int64_t v31 = v29 * v30;
  // pto: %h_tile_scale_dq_inline445__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v32 = pto::Shape<1, 1, 1, -1, -1>(v27, v28, v29, v11, v13);
  // pto: %h_tile_scale_dq_inline445__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v33 = pto::Stride<-1, -1, -1, -1, -1>(v28 * v31, v31, v30, v13, v11);
  // pto: %h_tile_scale_dq_inline445__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v34 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v2, v32, v33);
  // pto: %h_tile_i8_inline444__ssa_v0_view
  const int64_t v35 = 1;
  // pto: %h_tile_i8_inline444__ssa_v0_view
  const int64_t v36 = 1;
  // pto: %h_tile_i8_inline444__ssa_v0_view
  const int64_t v37 = 1;
  // pto: %h_tile_i8_inline444__ssa_v0_view
  int64_t v38 = v11 * v12;
  // pto: %h_tile_i8_inline444__ssa_v0_view
  int64_t v39 = v37 * v38;
  // pto: %h_tile_i8_inline444__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v40 = pto::Shape<1, 1, 1, -1, -1>(v35, v36, v37, v11, v12);
  // pto: %h_tile_i8_inline444__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v41 = pto::Stride<-1, -1, -1, -1, -1>(v36 * v39, v39, v38, v12, v13);
  // pto: %h_tile_i8_inline444__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v42 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v40, v41);
  // pto: %eh_amax_inline483__tile
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v43 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v11);
  // pto: %eh_amax_inline483__tile
  uint64_t v44 = (uint64_t) v4;
  TASSIGN(v43, v44);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  TEXPANDS(v43, v14);
  for (int64_t v45 = v15; v45 < v12; v45 += v16) {
    // pto: %eh_a_f32_inline485__tile
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v46 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v17);
    // pto: %eh_a_f32_inline485__tile
    ;
    uint64_t v47 = (uint64_t) v5;
    TASSIGN(v46, v47);
    // pto: %18
    ;
    int64_t v48 = v45 < v15 ? v15 : v45;
    // pto: %h_tile_fp32_inline464__rv_v2_pview
    ;
    __gm__ float* v49 = PTOAS__GLOBAL_TENSOR_DATA(v26);
    // pto: %h_tile_fp32_inline464__rv_v2_pview
    ;
    const int64_t v50 = 0;
    // pto: %h_tile_fp32_inline464__rv_v2_pview
    ;
    pto::Shape<1, 1, 1, 16, 512> v51 = pto::Shape<1, 1, 1, 16, 512>();
    // pto: %h_tile_fp32_inline464__rv_v2_pview
    ;
    pto::Stride<32768, 32768, 32768, 2048, 1> v52 = pto::Stride<32768, 32768, 32768, 2048, 1>();
    // pto: %h_tile_fp32_inline464__rv_v2_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<32768, 32768, 32768, 2048, 1>, pto::Layout::ND> v53 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<32768, 32768, 32768, 2048, 1>, pto::Layout::ND>(v49 + (v50 + v48), v51, v52);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v46, v53);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %0
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v54 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v17);
    // pto: %0
    ;
    uint64_t v55 = (uint64_t) v6;
    TASSIGN(v54, v55);
    // pto: %19
    ;
    int64_t v56 = (int64_t) ((uint64_t) v45 + (uint64_t) v17);
    // pto: %20
    ;
    int64_t v57 = v56 < v15 ? v15 : v56;
    // pto: %21
    ;
    __gm__ float* v58 = PTOAS__GLOBAL_TENSOR_DATA(v26);
    // pto: %21
    ;
    const int64_t v59 = 0;
    // pto: %21
    ;
    pto::Shape<1, 1, 1, 16, 512> v60 = pto::Shape<1, 1, 1, 16, 512>();
    // pto: %21
    ;
    pto::Stride<32768, 32768, 32768, 2048, 1> v61 = pto::Stride<32768, 32768, 32768, 2048, 1>();
    // pto: %21
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<32768, 32768, 32768, 2048, 1>, pto::Layout::ND> v62 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<32768, 32768, 32768, 2048, 1>, pto::Layout::ND>(v58 + (v59 + v57), v60, v61);
    TLOAD(v54, v62);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    // pto: %t__tile
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v63 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v17);
    // pto: %t__tile
    ;
    uint64_t v64 = (uint64_t) v7;
    TASSIGN(v63, v64);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TNEG(v63, v46);
    // pto: %eh_a_abs_inline486__tile
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v65 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v17);
    // pto: %eh_a_abs_inline486__tile
    ;
    uint64_t v66 = (uint64_t) v5;
    TASSIGN(v65, v66);
    pipe_barrier(PIPE_V);
    TMAX(v65, v46, v63);
    // pto: %tmp_tile
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v67 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v17);
    // pto: %tmp_tile
    ;
    uint64_t v68 = (uint64_t) v7;
    TASSIGN(v67, v68);
    // pto: %1
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v69 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v13);
    // pto: %1
    ;
    uint64_t v70 = (uint64_t) v8;
    TASSIGN(v69, v70);
    pipe_barrier(PIPE_V);
    TROWMAX(v69, v65, v67);
    // pto: %eh_a_max_inline487__tile
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v71 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v11);
    // pto: %eh_a_max_inline487__tile
    ;
    uint64_t v72 = (uint64_t) v8;
    TASSIGN(v71, v72);
    // pto: %2
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v73 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v11);
    // pto: %2
    ;
    uint64_t v74 = (uint64_t) v5;
    TASSIGN(v73, v74);
    pipe_barrier(PIPE_V);
    TMAX(v73, v43, v71);
    // pto: %3
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v75 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v17);
    // pto: %3
    ;
    uint64_t v76 = (uint64_t) v9;
    TASSIGN(v75, v76);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TNEG(v75, v54);
    // pto: %4
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v77 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v17);
    // pto: %4
    ;
    uint64_t v78 = (uint64_t) v6;
    TASSIGN(v77, v78);
    pipe_barrier(PIPE_V);
    TMAX(v77, v54, v75);
    // pto: %5
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v79 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v17);
    // pto: %5
    ;
    uint64_t v80 = (uint64_t) v9;
    TASSIGN(v79, v80);
    // pto: %6
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v81 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v13);
    // pto: %6
    ;
    uint64_t v82 = (uint64_t) v10;
    TASSIGN(v81, v82);
    pipe_barrier(PIPE_V);
    TROWMAX(v81, v77, v79);
    // pto: %7
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v83 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v11);
    // pto: %7
    ;
    uint64_t v84 = (uint64_t) v10;
    TASSIGN(v83, v84);
    // pto: %8
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v85 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v11);
    // pto: %8
    ;
    uint64_t v86 = (uint64_t) v4;
    TASSIGN(v85, v86);
    pipe_barrier(PIPE_V);
    TMAX(v85, v73, v83);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  }
  // pto: %9
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v87 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v11);
  // pto: %9
  uint64_t v88 = (uint64_t) v5;
  TASSIGN(v87, v88);
  pipe_barrier(PIPE_V);
  TEXPANDS(v87, v18);
  // pto: %eh_sq_row_inline488__tile
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v89 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v11);
  // pto: %eh_sq_row_inline488__tile
  uint64_t v90 = (uint64_t) v7;
  TASSIGN(v89, v90);
  pipe_barrier(PIPE_V);
  TDIV(v89, v87, v43);
  // pto: %10
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v91 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v11);
  // pto: %10
  uint64_t v92 = (uint64_t) v5;
  TASSIGN(v91, v92);
  pipe_barrier(PIPE_V);
  TRECIP(v91, v89);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  // pto: %11
  Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v93 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v13);
  // pto: %11
  uint64_t v94 = (uint64_t) v5;
  TASSIGN(v93, v94);
  // pto: %h_tile_scale_dq_inline445__ssa_v0_pview
  __gm__ float* v95 = PTOAS__GLOBAL_TENSOR_DATA(v34);
  // pto: %h_tile_scale_dq_inline445__ssa_v0_pview
  pto::Shape<1, 1, 1, 16, 1> v96 = pto::Shape<1, 1, 1, 16, 1>();
  // pto: %h_tile_scale_dq_inline445__ssa_v0_pview
  pto::Stride<16, 16, 16, 1, 16> v97 = pto::Stride<16, 16, 16, 1, 16>();
  // pto: %h_tile_scale_dq_inline445__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 16>, pto::Layout::DN> v98 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 16>, pto::Layout::DN>(v95, v96, v97);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v98, v93);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  // pto: %eh_sq_col_inline489__tile
  Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v99 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v13);
  // pto: %eh_sq_col_inline489__tile
  uint64_t v100 = (uint64_t) v7;
  TASSIGN(v99, v100);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (int64_t v101 = v15; v101 < v12; v101 += v16) {
    // pto: %eh_q_f32_inline491__tile
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v102 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v17);
    // pto: %eh_q_f32_inline491__tile
    ;
    uint64_t v103 = (uint64_t) v5;
    TASSIGN(v102, v103);
    // pto: %22
    ;
    int64_t v104 = v101 < v15 ? v15 : v101;
    // pto: %23
    ;
    __gm__ float* v105 = PTOAS__GLOBAL_TENSOR_DATA(v26);
    // pto: %23
    ;
    const int64_t v106 = 0;
    // pto: %23
    ;
    pto::Shape<1, 1, 1, 16, 512> v107 = pto::Shape<1, 1, 1, 16, 512>();
    // pto: %23
    ;
    pto::Stride<32768, 32768, 32768, 2048, 1> v108 = pto::Stride<32768, 32768, 32768, 2048, 1>();
    // pto: %23
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<32768, 32768, 32768, 2048, 1>, pto::Layout::ND> v109 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<32768, 32768, 32768, 2048, 1>, pto::Layout::ND>(v105 + (v106 + v104), v107, v108);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    TLOAD(v102, v109);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    // pto: %12
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v110 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v17);
    // pto: %12
    ;
    uint64_t v111 = (uint64_t) v6;
    TASSIGN(v110, v111);
    // pto: %24
    ;
    int64_t v112 = (int64_t) ((uint64_t) v101 + (uint64_t) v17);
    // pto: %25
    ;
    int64_t v113 = v112 < v15 ? v15 : v112;
    // pto: %26
    ;
    __gm__ float* v114 = PTOAS__GLOBAL_TENSOR_DATA(v26);
    // pto: %26
    ;
    const int64_t v115 = 0;
    // pto: %26
    ;
    pto::Shape<1, 1, 1, 16, 512> v116 = pto::Shape<1, 1, 1, 16, 512>();
    // pto: %26
    ;
    pto::Stride<32768, 32768, 32768, 2048, 1> v117 = pto::Stride<32768, 32768, 32768, 2048, 1>();
    // pto: %26
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<32768, 32768, 32768, 2048, 1>, pto::Layout::ND> v118 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<32768, 32768, 32768, 2048, 1>, pto::Layout::ND>(v114 + (v115 + v113), v116, v117);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    TLOAD(v110, v118);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    // pto: %eh_q_scaled_inline492__tile
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v119 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v17);
    // pto: %eh_q_scaled_inline492__tile
    ;
    uint64_t v120 = (uint64_t) v5;
    TASSIGN(v119, v120);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    TROWEXPANDMUL(v119, v102, v99);
    // pto: %eh_q_i32_inline493__tile
    ;
    Tile<TileType::Vec, int32_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v121 = Tile<TileType::Vec, int32_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v17);
    // pto: %eh_q_i32_inline493__tile
    ;
    uint64_t v122 = (uint64_t) v5;
    TASSIGN(v121, v122);
    pipe_barrier(PIPE_V);
    RoundMode v123 = RoundMode::CAST_RINT;
    SaturationMode v124 = SaturationMode::ON;
    TCVT(v121, v119, v123, v124);
    // pto: %eh_q_half_inline494__tile
    ;
    Tile<TileType::Vec, half, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v125 = Tile<TileType::Vec, half, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v17);
    // pto: %eh_q_half_inline494__tile
    ;
    uint64_t v126 = (uint64_t) v5;
    TASSIGN(v125, v126);
    pipe_barrier(PIPE_V);
    RoundMode v127 = RoundMode::CAST_ROUND;
    SaturationMode v128 = SaturationMode::OFF;
    TCVT(v125, v121, v127, v128);
    // pto: %13
    ;
    Tile<TileType::Vec, int8_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v129 = Tile<TileType::Vec, int8_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v17);
    // pto: %13
    ;
    uint64_t v130 = (uint64_t) v5;
    TASSIGN(v129, v130);
    pipe_barrier(PIPE_V);
    RoundMode v131 = RoundMode::CAST_TRUNC;
    SaturationMode v132 = SaturationMode::ON;
    TCVT(v129, v125, v131, v132);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    // pto: %h_tile_i8_inline444__iter_v1_pview
    ;
    __gm__ int8_t* v133 = PTOAS__GLOBAL_TENSOR_DATA(v42);
    // pto: %h_tile_i8_inline444__iter_v1_pview
    ;
    const int64_t v134 = 0;
    // pto: %h_tile_i8_inline444__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 16, 512> v135 = pto::Shape<1, 1, 1, 16, 512>();
    // pto: %h_tile_i8_inline444__iter_v1_pview
    ;
    pto::Stride<32768, 32768, 32768, 2048, 1> v136 = pto::Stride<32768, 32768, 32768, 2048, 1>();
    // pto: %h_tile_i8_inline444__iter_v1_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<32768, 32768, 32768, 2048, 1>, pto::Layout::ND> v137 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<32768, 32768, 32768, 2048, 1>, pto::Layout::ND>(v133 + (v134 + v104), v135, v136);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v137, v129);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    // pto: %14
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v138 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v17);
    // pto: %14
    ;
    uint64_t v139 = (uint64_t) v6;
    TASSIGN(v138, v139);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    TROWEXPANDMUL(v138, v110, v99);
    // pto: %15
    ;
    Tile<TileType::Vec, int32_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v140 = Tile<TileType::Vec, int32_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v17);
    // pto: %15
    ;
    uint64_t v141 = (uint64_t) v6;
    TASSIGN(v140, v141);
    pipe_barrier(PIPE_V);
    RoundMode v142 = RoundMode::CAST_RINT;
    SaturationMode v143 = SaturationMode::ON;
    TCVT(v140, v138, v142, v143);
    // pto: %16
    ;
    Tile<TileType::Vec, half, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v144 = Tile<TileType::Vec, half, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v17);
    // pto: %16
    ;
    uint64_t v145 = (uint64_t) v6;
    TASSIGN(v144, v145);
    pipe_barrier(PIPE_V);
    RoundMode v146 = RoundMode::CAST_ROUND;
    SaturationMode v147 = SaturationMode::OFF;
    TCVT(v144, v140, v146, v147);
    // pto: %17
    ;
    Tile<TileType::Vec, int8_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v148 = Tile<TileType::Vec, int8_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v17);
    // pto: %17
    ;
    uint64_t v149 = (uint64_t) v6;
    TASSIGN(v148, v149);
    pipe_barrier(PIPE_V);
    RoundMode v150 = RoundMode::CAST_TRUNC;
    SaturationMode v151 = SaturationMode::ON;
    TCVT(v148, v144, v150, v151);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    // pto: %h_tile_i8_inline444__tile_pview
    ;
    __gm__ int8_t* v152 = PTOAS__GLOBAL_TENSOR_DATA(v42);
    // pto: %h_tile_i8_inline444__tile_pview
    ;
    const int64_t v153 = 0;
    // pto: %h_tile_i8_inline444__tile_pview
    ;
    pto::Shape<1, 1, 1, 16, 512> v154 = pto::Shape<1, 1, 1, 16, 512>();
    // pto: %h_tile_i8_inline444__tile_pview
    ;
    pto::Stride<32768, 32768, 32768, 2048, 1> v155 = pto::Stride<32768, 32768, 32768, 2048, 1>();
    // pto: %h_tile_i8_inline444__tile_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<32768, 32768, 32768, 2048, 1>, pto::Layout::ND> v156 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<32768, 32768, 32768, 2048, 1>, pto::Layout::ND>(v152 + (v153 + v113), v154, v155);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v156, v148);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  }
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}