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

AICORE void build_valid(__gm__ int32_t* v1, __gm__ float* v2, __gm__ int32_t* v3, __gm__ int32_t* v4, int64_t v5, int64_t v6, int64_t v7, int32_t v8, int32_t v9) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c4640_i64
  const int64_t v10 = 4640;
  // pto: %c0_i64
  const int64_t v11 = 0;
  // pto: %c544_i64
  const int64_t v12 = 544;
  // pto: %c32_i64
  const int64_t v13 = 32;
  // pto: %c8_index
  const int64_t v14 = 8;
  // pto: %c128_index
  const int64_t v15 = 128;
  // pto: %c1_index
  const int64_t v16 = 1;
  // pto: %c0_index
  const int64_t v17 = 0;
  // pto: %cst_9
  const float v18 = 1.0f;
  // pto: %cst_10
  const float v19 = 0.0f;
  // pto: %cst_11
  const float v20 = 1.00000002E+20f;
  // pto: %c2_index
  const int64_t v21 = 2;
  // pto: %c192_index
  const int64_t v22 = 192;
  // pto: %c0_i32
  const int32_t v23 = 0;
  // pto: %window_swa_indices__ssa_v0_view
  const int64_t v24 = 1;
  // pto: %window_swa_indices__ssa_v0_view
  const int64_t v25 = 1;
  // pto: %window_swa_indices__ssa_v0_view
  const int64_t v26 = 1;
  // pto: %window_swa_indices__ssa_v0_view
  int64_t v27 = v14 * v15;
  // pto: %window_swa_indices__ssa_v0_view
  int64_t v28 = v26 * v27;
  // pto: %window_swa_indices__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v29 = pto::Shape<1, 1, 1, -1, -1>(v24, v25, v26, v14, v15);
  // pto: %window_swa_indices__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v30 = pto::Stride<-1, -1, -1, -1, -1>(v25 * v28, v28, v27, v15, v16);
  // pto: %window_swa_indices__ssa_v0_view
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v31 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v29, v30);
  // pto: %sparse_bias__ssa_v0_view
  const int64_t v32 = 1;
  // pto: %sparse_bias__ssa_v0_view
  const int64_t v33 = 1;
  // pto: %sparse_bias__ssa_v0_view
  const int64_t v34 = 1;
  // pto: %sparse_bias__ssa_v0_view
  int64_t v35 = (int64_t) v7;
  // pto: %sparse_bias__ssa_v0_view
  int64_t v36 = v14 * v35;
  // pto: %sparse_bias__ssa_v0_view
  int64_t v37 = v34 * v36;
  // pto: %sparse_bias__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v38 = pto::Shape<1, 1, 1, -1, -1>(v32, v33, v34, v14, (int64_t) v7);
  // pto: %sparse_bias__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v39 = pto::Stride<-1, -1, -1, -1, -1>(v33 * v37, v37, v36, v35, v16);
  // pto: %sparse_bias__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v40 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v38, v39);
  // pto: %v_blk__ssa_v0, %11
  int64_t v41 = (int64_t) ((uint64_t) ((int64_t) v8) * (uint64_t) v14);
  // pto: %t__tile
  Tile<TileType::Vec, int32_t, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v42 = Tile<TileType::Vec, int32_t, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v15);
  // pto: %t__tile
  uint64_t v43 = (uint64_t) v10;
  TASSIGN(v42, v43);
  // pto: %12
  int64_t v44 = v41 < v17 ? v17 : v41;
  // pto: %window_swa_indices__ssa_v0_pview
  __gm__ int32_t* v45 = PTOAS__GLOBAL_TENSOR_DATA(v31);
  // pto: %window_swa_indices__ssa_v0_pview
  const int64_t v46 = 0;
  // pto: %window_swa_indices__ssa_v0_pview
  const int64_t v47 = 128;
  // pto: %window_swa_indices__ssa_v0_pview
  pto::Shape<1, 1, 1, 8, 128> v48 = pto::Shape<1, 1, 1, 8, 128>();
  // pto: %window_swa_indices__ssa_v0_pview
  pto::Stride<1024, 1024, 1024, 128, 1> v49 = pto::Stride<1024, 1024, 1024, 128, 1>();
  // pto: %window_swa_indices__ssa_v0_pview
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 128>, pto::Stride<1024, 1024, 1024, 128, 1>, pto::Layout::ND> v50 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 128>, pto::Stride<1024, 1024, 1024, 128, 1>, pto::Layout::ND>(v45 + (v46 + v44 * v47), v48, v49);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  TLOAD(v42, v50);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  // pto: %v_win_f__tile
  Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v51 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v15);
  // pto: %v_win_f__tile
  uint64_t v52 = (uint64_t) v10;
  TASSIGN(v51, v52);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  RoundMode v53 = RoundMode::CAST_ROUND;
  SaturationMode v54 = SaturationMode::OFF;
  TCVT(v51, v42, v53, v54);
  // pto: %0
  Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v55 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v15);
  // pto: %0
  uint64_t v56 = (uint64_t) v10;
  TASSIGN(v55, v56);
  pipe_barrier(PIPE_V);
  TADDS(v55, v51, v18);
  // pto: %1
  Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v57 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v15);
  // pto: %1
  uint64_t v58 = (uint64_t) v10;
  TASSIGN(v57, v58);
  pipe_barrier(PIPE_V);
  TMAXS(v57, v55, v19);
  // pto: %v_win_valid__tile
  Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v59 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v15);
  // pto: %v_win_valid__tile
  uint64_t v60 = (uint64_t) v10;
  TASSIGN(v59, v60);
  pipe_barrier(PIPE_V);
  TMINS(v59, v57, v18);
  // pto: %2
  Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v61 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v15);
  // pto: %2
  uint64_t v62 = (uint64_t) v10;
  TASSIGN(v61, v62);
  pipe_barrier(PIPE_V);
  TSUBS(v61, v59, v18);
  // pto: %3
  Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v63 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v15);
  // pto: %3
  uint64_t v64 = (uint64_t) v10;
  TASSIGN(v63, v64);
  pipe_barrier(PIPE_V);
  TMULS(v63, v61, v20);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  // pto: %sparse_bias__ssa_v0_pview
  int64_t v65 = (int64_t) v7;
  // pto: %sparse_bias__ssa_v0_pview
  const int64_t v66 = 0;
  // pto: %sparse_bias__ssa_v0_pview
  __gm__ float* v67 = PTOAS__GLOBAL_TENSOR_DATA(v40);
  // pto: %sparse_bias__ssa_v0_pview
  const int64_t v68 = 1;
  // pto: %sparse_bias__ssa_v0_pview
  const int64_t v69 = 1;
  // pto: %sparse_bias__ssa_v0_pview
  const int64_t v70 = 1;
  // pto: %sparse_bias__ssa_v0_pview
  int64_t v71 = v14 * v65;
  // pto: %sparse_bias__ssa_v0_pview
  int64_t v72 = v70 * v71;
  // pto: %sparse_bias__ssa_v0_pview
  pto::Shape<1, 1, 1, 8, 128> v73 = pto::Shape<1, 1, 1, 8, 128>(v68, v69, v70, v14, v15);
  // pto: %sparse_bias__ssa_v0_pview
  pto::Stride<-1, -1, -1, -1, -1> v74 = pto::Stride<-1, -1, -1, -1, -1>(v69 * v72, v72, v71, v65, v16);
  // pto: %sparse_bias__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 8, 128>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v75 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 128>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v67 + (v66 + v44 * v65 + v17 * v16), v73, v74);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v75, v63);
  set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  // pto: %v_rows_i32__tile
  Tile<TileType::Vec, int32_t, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v76 = Tile<TileType::Vec, int32_t, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v16);
  // pto: %v_rows_i32__tile
  uint64_t v77 = (uint64_t) v10;
  TASSIGN(v76, v77);
  wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
  for (int64_t v78 = v17; v78 < v14; v78 += v16) {
    // pto: %14
    ;
    int64_t v79 = (int64_t) ((uint64_t) v41 + (uint64_t) v78);
    // pto: %v_pos__tile
    ;
    int32_t v80 = (v3)[v79];
    // pto: %15, %v_len__tile
    ;
    int32_t v81 = (v4)[v79 / v21];
    // pto: %16, %17
    ;
    int64_t v82 = (int64_t) ((uint64_t) ((int64_t) v80) + (uint64_t) v16);
    // pto: %18
    ;
    int64_t v83 = (int64_t) v81;
    // pto: %19
    ;
    int64_t v84 = v82 < v83 ? v82 : v83;
    // pto: %20
    ;
    int64_t v85 = v84 / v15;
    // pto: %21
    ;
    int64_t v86 = v85 < v5 ? v85 : v5;
    // pto: %22
    ;
    int32_t v87 = (int32_t) v86;
    v76.SetValue(v78, v87);
  }
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  // pto: %v_rows__rm_a0_tmp_v0
  Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v88 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v14);
  // pto: %v_rows__rm_a0_tmp_v0
  uint64_t v89 = (uint64_t) v10;
  TASSIGN(v88, v89);
  // pto: %v_rows__row_major_tmp_v1
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v90 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v14);
  // pto: %v_rows__row_major_tmp_v1
  uint64_t v91 = (uint64_t) v11;
  TASSIGN(v90, v91);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
  pipe_barrier(PIPE_V);
  RoundMode v92 = RoundMode::CAST_ROUND;
  SaturationMode v93 = SaturationMode::OFF;
  TCVT(v90, v88, v92, v93);
  // pto: %v_rows__tile
  Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v94 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v16);
  // pto: %v_rows__tile
  uint64_t v95 = (uint64_t) v11;
  TASSIGN(v94, v95);
  // pto: %v_zero__tile
  Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v96 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v15);
  // pto: %v_zero__tile
  uint64_t v97 = (uint64_t) v10;
  TASSIGN(v96, v97);
  pipe_barrier(PIPE_V);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  TEXPANDS(v96, v19);
  for (int64_t v98 = v17; v98 < v6; v98 += v16) {
    // pto: %23
    ;
    int64_t v99 = (int64_t) ((uint64_t) v98 * (uint64_t) v15);
    // pto: %24
    ;
    int64_t v100 = (int64_t) ((uint64_t) v99 + (uint64_t) v15);
    // pto: %t__ci_tmp_v0
    ;
    Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v101 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v22);
    // pto: %t__ci_tmp_v0
    ;
    uint64_t v102 = (uint64_t) v12;
    TASSIGN(v101, v102);
    // pto: %4
    ;
    Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v103 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
    // pto: %4
    ;
    uint64_t v104 = (uint64_t) v13;
    TASSIGN(v103, v104);
    // pto: %ci_tmp_view
    ;
    Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v105;
    TRESHAPE(v105, v101);
    // pto: %ci_dst_view
    ;
    Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v106;
    TRESHAPE(v106, v103);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    TCI<Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v106, v23, v105);
    // pto: %v_col_i32__tile
    ;
    Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v107 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
    // pto: %v_col_i32__tile
    ;
    uint64_t v108 = (uint64_t) v12;
    TASSIGN(v107, v108);
    // pto: %26
    ;
    int32_t v109 = (int32_t) v99;
    pipe_barrier(PIPE_V);
    TADDS(v107, v103, v109);
    // pto: %5
    ;
    Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v110 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
    // pto: %5
    ;
    uint64_t v111 = (uint64_t) v13;
    TASSIGN(v110, v111);
    pipe_barrier(PIPE_V);
    RoundMode v112 = RoundMode::CAST_ROUND;
    SaturationMode v113 = SaturationMode::OFF;
    TCVT(v110, v107, v112, v113);
    // pto: %v_cols__tile
    ;
    Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v114 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v15);
    // pto: %v_cols__tile
    ;
    uint64_t v115 = (uint64_t) v12;
    TASSIGN(v114, v115);
    pipe_barrier(PIPE_V);
    TCOLEXPANDADD(v114, v96, v110);
    // pto: %6
    ;
    Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v116 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v15);
    // pto: %6
    ;
    uint64_t v117 = (uint64_t) v12;
    TASSIGN(v116, v117);
    pipe_barrier(PIPE_V);
    TROWEXPANDSUB(v116, v114, v94);
    // pto: %7
    ;
    Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v118 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v15);
    // pto: %7
    ;
    uint64_t v119 = (uint64_t) v12;
    TASSIGN(v118, v119);
    pipe_barrier(PIPE_V);
    TNEG(v118, v116);
    // pto: %8
    ;
    Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v120 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v15);
    // pto: %8
    ;
    uint64_t v121 = (uint64_t) v12;
    TASSIGN(v120, v121);
    pipe_barrier(PIPE_V);
    TMAXS(v120, v118, v19);
    // pto: %v_cmp_mask__tile
    ;
    Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v122 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v15);
    // pto: %v_cmp_mask__tile
    ;
    uint64_t v123 = (uint64_t) v12;
    TASSIGN(v122, v123);
    pipe_barrier(PIPE_V);
    TMINS(v122, v120, v18);
    // pto: %9
    ;
    Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v124 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v15);
    // pto: %9
    ;
    uint64_t v125 = (uint64_t) v12;
    TASSIGN(v124, v125);
    pipe_barrier(PIPE_V);
    TSUBS(v124, v122, v18);
    // pto: %10
    ;
    Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v126 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v15);
    // pto: %10
    ;
    uint64_t v127 = (uint64_t) v12;
    TASSIGN(v126, v127);
    pipe_barrier(PIPE_V);
    TMULS(v126, v124, v20);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    // pto: %29
    ;
    int64_t v128 = v100 < v17 ? v17 : v100;
    // pto: %sparse_bias__iter_v2_pview
    ;
    int64_t v129 = (int64_t) v7;
    // pto: %sparse_bias__iter_v2_pview
    ;
    const int64_t v130 = 0;
    // pto: %sparse_bias__iter_v2_pview
    ;
    __gm__ float* v131 = PTOAS__GLOBAL_TENSOR_DATA(v40);
    // pto: %sparse_bias__iter_v2_pview
    ;
    const int64_t v132 = 1;
    // pto: %sparse_bias__iter_v2_pview
    ;
    const int64_t v133 = 1;
    // pto: %sparse_bias__iter_v2_pview
    ;
    const int64_t v134 = 1;
    // pto: %sparse_bias__iter_v2_pview
    ;
    int64_t v135 = v14 * v129;
    // pto: %sparse_bias__iter_v2_pview
    ;
    int64_t v136 = v134 * v135;
    // pto: %sparse_bias__iter_v2_pview
    ;
    pto::Shape<1, 1, 1, 8, 128> v137 = pto::Shape<1, 1, 1, 8, 128>(v132, v133, v134, v14, v15);
    // pto: %sparse_bias__iter_v2_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v138 = pto::Stride<-1, -1, -1, -1, -1>(v133 * v136, v136, v135, v129, v16);
    // pto: %sparse_bias__iter_v2_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 128>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v139 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 128>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v131 + (v130 + v44 * v129 + v128 * v16), v137, v138);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TSTORE(v139, v126);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  }
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}