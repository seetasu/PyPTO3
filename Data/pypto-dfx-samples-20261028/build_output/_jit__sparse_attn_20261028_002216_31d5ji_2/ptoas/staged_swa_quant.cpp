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

AICORE void staged_swa_quant(__gm__ float* v1, __gm__ float* v2, __gm__ int8_t* v3, int64_t v4, int64_t v5, int64_t v6, int32_t v7, int32_t v8) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v9 = 0;
  // pto: %c128_i64
  const int64_t v10 = 128;
  // pto: %c32896_i64
  const int64_t v11 = 32896;
  // pto: %c65664_i64
  const int64_t v12 = 65664;
  // pto: %c8192_index
  const int64_t v13 = 8192;
  // pto: %c1_index
  const int64_t v14 = 1;
  // pto: %c8_index
  const int64_t v15 = 8;
  // pto: %c32_index
  const int64_t v16 = 32;
  // pto: %cst_8
  const float v17 = 9.99999974E-5f;
  // pto: %c0_index
  const int64_t v18 = 0;
  // pto: %c1024_index
  const int64_t v19 = 1024;
  // pto: %c256_index
  const int64_t v20 = 256;
  // pto: %cst_12
  const float v21 = 127.0f;
  // pto: %o_r_inline655__rv_v2_view
  const int64_t v22 = 1;
  // pto: %o_r_inline655__rv_v2_view
  const int64_t v23 = 1;
  // pto: %o_r_inline655__rv_v2_view
  const int64_t v24 = 1;
  // pto: %o_r_inline655__rv_v2_view
  int64_t v25 = (int64_t) v6;
  // pto: %o_r_inline655__rv_v2_view
  int64_t v26 = v25 * v13;
  // pto: %o_r_inline655__rv_v2_view
  int64_t v27 = v24 * v26;
  // pto: %o_r_inline655__rv_v2_view
  pto::Shape<1, 1, 1, -1, -1> v28 = pto::Shape<1, 1, 1, -1, -1>(v22, v23, v24, v25, v13);
  // pto: %o_r_inline655__rv_v2_view
  pto::Stride<-1, -1, -1, -1, -1> v29 = pto::Stride<-1, -1, -1, -1, -1>(v23 * v27, v27, v26, v13, v14);
  // pto: %o_r_inline655__rv_v2_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v30 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v28, v29);
  // pto: %act_scale_dq_inline657__iter_v1_view
  const int64_t v31 = 1;
  // pto: %act_scale_dq_inline657__iter_v1_view
  const int64_t v32 = 1;
  // pto: %act_scale_dq_inline657__iter_v1_view
  const int64_t v33 = 1;
  // pto: %act_scale_dq_inline657__iter_v1_view
  int64_t v34 = (int64_t) v6;
  // pto: %act_scale_dq_inline657__iter_v1_view
  int64_t v35 = v15 * v34;
  // pto: %act_scale_dq_inline657__iter_v1_view
  int64_t v36 = v33 * v35;
  // pto: %act_scale_dq_inline657__iter_v1_view
  pto::Shape<1, 1, 1, -1, -1> v37 = pto::Shape<1, 1, 1, -1, -1>(v31, v32, v33, v15, (int64_t) v6);
  // pto: %act_scale_dq_inline657__iter_v1_view
  pto::Stride<-1, -1, -1, -1, -1> v38 = pto::Stride<-1, -1, -1, -1, -1>(v32 * v36, v36, v35, v34, v14);
  // pto: %act_scale_dq_inline657__iter_v1_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v39 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v37, v38);
  // pto: %o_r_i8_inline656__iter_v1_view
  const int64_t v40 = 1;
  // pto: %o_r_i8_inline656__iter_v1_view
  const int64_t v41 = 1;
  // pto: %o_r_i8_inline656__iter_v1_view
  const int64_t v42 = 1;
  // pto: %o_r_i8_inline656__iter_v1_view
  int64_t v43 = (int64_t) v6;
  // pto: %o_r_i8_inline656__iter_v1_view
  int64_t v44 = v43 * v13;
  // pto: %o_r_i8_inline656__iter_v1_view
  int64_t v45 = v42 * v44;
  // pto: %o_r_i8_inline656__iter_v1_view
  pto::Shape<1, 1, 1, -1, -1> v46 = pto::Shape<1, 1, 1, -1, -1>(v40, v41, v42, v43, v13);
  // pto: %o_r_i8_inline656__iter_v1_view
  pto::Stride<-1, -1, -1, -1, -1> v47 = pto::Stride<-1, -1, -1, -1, -1>(v41 * v45, v45, v44, v13, v14);
  // pto: %o_r_i8_inline656__iter_v1_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v48 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v46, v47);
  // pto: %t__tmp_v88, %4
  int64_t v49 = (int64_t) ((uint64_t) ((int64_t) v7) * (uint64_t) v16);
  // pto: %g_amax_inline683__tile
  Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v50 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v16);
  // pto: %g_amax_inline683__tile
  uint64_t v51 = (uint64_t) v9;
  TASSIGN(v50, v51);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  TEXPANDS(v50, v17);
  for (int64_t v52 = v18; v52 < v19; v52 += v20) {
    // pto: %oc_inline685__tile
    ;
    Tile<TileType::Vec, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v53 = Tile<TileType::Vec, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v20);
    // pto: %oc_inline685__tile
    ;
    uint64_t v54 = (uint64_t) v10;
    TASSIGN(v53, v54);
    // pto: %5
    ;
    int64_t v55 = v49 < v18 ? v18 : v49;
    // pto: %6
    ;
    int64_t v56 = (int64_t) ((uint64_t) v4 + (uint64_t) v52);
    // pto: %7
    ;
    int64_t v57 = v56 < v18 ? v18 : v56;
    // pto: %o_r_inline655__rv_v2_pview
    ;
    __gm__ float* v58 = PTOAS__GLOBAL_TENSOR_DATA(v30);
    // pto: %o_r_inline655__rv_v2_pview
    ;
    const int64_t v59 = 0;
    // pto: %o_r_inline655__rv_v2_pview
    ;
    const int64_t v60 = 8192;
    // pto: %o_r_inline655__rv_v2_pview
    ;
    pto::Shape<1, 1, 1, 32, 256> v61 = pto::Shape<1, 1, 1, 32, 256>();
    // pto: %o_r_inline655__rv_v2_pview
    ;
    pto::Stride<262144, 262144, 262144, 8192, 1> v62 = pto::Stride<262144, 262144, 262144, 8192, 1>();
    // pto: %o_r_inline655__rv_v2_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<262144, 262144, 262144, 8192, 1>, pto::Layout::ND> v63 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<262144, 262144, 262144, 8192, 1>, pto::Layout::ND>(v58 + (v59 + v55 * v60 + v57), v61, v62);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v53, v63);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %t__tile
    ;
    Tile<TileType::Vec, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v64 = Tile<TileType::Vec, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v20);
    // pto: %t__tile
    ;
    uint64_t v65 = (uint64_t) v11;
    TASSIGN(v64, v65);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TNEG(v64, v53);
    // pto: %oc_abs_inline686__tile
    ;
    Tile<TileType::Vec, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v66 = Tile<TileType::Vec, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v20);
    // pto: %oc_abs_inline686__tile
    ;
    uint64_t v67 = (uint64_t) v10;
    TASSIGN(v66, v67);
    pipe_barrier(PIPE_V);
    TMAX(v66, v53, v64);
    // pto: %tmp_tile
    ;
    Tile<TileType::Vec, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v68 = Tile<TileType::Vec, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v20);
    // pto: %tmp_tile
    ;
    uint64_t v69 = (uint64_t) v11;
    TASSIGN(v68, v69);
    // pto: %0
    ;
    Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v70 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v14);
    // pto: %0
    ;
    uint64_t v71 = (uint64_t) v12;
    TASSIGN(v70, v71);
    pipe_barrier(PIPE_V);
    TROWMAX(v70, v66, v68);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    // pto: %oc_amax_inline687__tile
    ;
    Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v72 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v16);
    // pto: %oc_amax_inline687__tile
    ;
    uint64_t v73 = (uint64_t) v12;
    TASSIGN(v72, v73);
    // pto: %1
    ;
    Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v74 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v16);
    // pto: %1
    ;
    uint64_t v75 = (uint64_t) v9;
    TASSIGN(v74, v75);
    pipe_barrier(PIPE_V);
    TMAX(v74, v50, v72);
  }
  // pto: %g_scale_num_inline688__tile
  Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v76 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v16);
  // pto: %g_scale_num_inline688__tile
  uint64_t v77 = (uint64_t) v10;
  TASSIGN(v76, v77);
  TEXPANDS(v76, v21);
  // pto: %g_sq_row_inline689__tile
  Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v78 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v16);
  // pto: %g_sq_row_inline689__tile
  uint64_t v79 = (uint64_t) v11;
  TASSIGN(v78, v79);
  pipe_barrier(PIPE_V);
  TDIV(v78, v76, v50);
  // pto: %2
  Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v80 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v16);
  // pto: %2
  uint64_t v81 = (uint64_t) v10;
  TASSIGN(v80, v81);
  pipe_barrier(PIPE_V);
  TRECIP(v80, v78);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  // pto: %8
  int64_t v82 = v5 < v18 ? v18 : v5;
  // pto: %9
  int64_t v83 = v49 < v18 ? v18 : v49;
  // pto: %act_scale_dq_inline657__iter_v1_pview
  int64_t v84 = (int64_t) v6;
  // pto: %act_scale_dq_inline657__iter_v1_pview
  const int64_t v85 = 0;
  // pto: %act_scale_dq_inline657__iter_v1_pview
  __gm__ float* v86 = PTOAS__GLOBAL_TENSOR_DATA(v39);
  // pto: %act_scale_dq_inline657__iter_v1_pview
  const int64_t v87 = 1;
  // pto: %act_scale_dq_inline657__iter_v1_pview
  const int64_t v88 = 1;
  // pto: %act_scale_dq_inline657__iter_v1_pview
  const int64_t v89 = 1;
  // pto: %act_scale_dq_inline657__iter_v1_pview
  int64_t v90 = v14 * v84;
  // pto: %act_scale_dq_inline657__iter_v1_pview
  int64_t v91 = v89 * v90;
  // pto: %act_scale_dq_inline657__iter_v1_pview
  pto::Shape<1, 1, 1, 1, 32> v92 = pto::Shape<1, 1, 1, 1, 32>(v87, v88, v89, v14, v16);
  // pto: %act_scale_dq_inline657__iter_v1_pview
  pto::Stride<-1, -1, -1, -1, -1> v93 = pto::Stride<-1, -1, -1, -1, -1>(v88 * v91, v91, v90, v84, v14);
  // pto: %act_scale_dq_inline657__iter_v1_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v94 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v86 + (v85 + v82 * v84 + v83 * v14), v92, v93);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v94, v80);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  // pto: %g_sq_col_inline690__tile
  Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v95 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v14);
  // pto: %g_sq_col_inline690__tile
  uint64_t v96 = (uint64_t) v11;
  TASSIGN(v95, v96);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (int64_t v97 = v18; v97 < v19; v97 += v20) {
    // pto: %3
    ;
    Tile<TileType::Vec, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v98 = Tile<TileType::Vec, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v20);
    // pto: %3
    ;
    uint64_t v99 = (uint64_t) v10;
    TASSIGN(v98, v99);
    // pto: %11
    ;
    int64_t v100 = (int64_t) ((uint64_t) v4 + (uint64_t) v97);
    // pto: %12
    ;
    int64_t v101 = v100 < v18 ? v18 : v100;
    // pto: %13
    ;
    __gm__ float* v102 = PTOAS__GLOBAL_TENSOR_DATA(v30);
    // pto: %13
    ;
    const int64_t v103 = 0;
    // pto: %13
    ;
    const int64_t v104 = 8192;
    // pto: %13
    ;
    pto::Shape<1, 1, 1, 32, 256> v105 = pto::Shape<1, 1, 1, 32, 256>();
    // pto: %13
    ;
    pto::Stride<262144, 262144, 262144, 8192, 1> v106 = pto::Stride<262144, 262144, 262144, 8192, 1>();
    // pto: %13
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<262144, 262144, 262144, 8192, 1>, pto::Layout::ND> v107 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<262144, 262144, 262144, 8192, 1>, pto::Layout::ND>(v102 + (v103 + v83 * v104 + v101), v105, v106);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    TLOAD(v98, v107);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    // pto: %oq_scaled_inline692__tile
    ;
    Tile<TileType::Vec, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v108 = Tile<TileType::Vec, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v20);
    // pto: %oq_scaled_inline692__tile
    ;
    uint64_t v109 = (uint64_t) v10;
    TASSIGN(v108, v109);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TROWEXPANDMUL(v108, v98, v95);
    // pto: %oq_i32_inline693__tile
    ;
    Tile<TileType::Vec, int32_t, 32, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v110 = Tile<TileType::Vec, int32_t, 32, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v20);
    // pto: %oq_i32_inline693__tile
    ;
    uint64_t v111 = (uint64_t) v10;
    TASSIGN(v110, v111);
    pipe_barrier(PIPE_V);
    RoundMode v112 = RoundMode::CAST_RINT;
    SaturationMode v113 = SaturationMode::ON;
    TCVT(v110, v108, v112, v113);
    // pto: %oq_half_inline694__tile
    ;
    Tile<TileType::Vec, half, 32, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v114 = Tile<TileType::Vec, half, 32, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v20);
    // pto: %oq_half_inline694__tile
    ;
    uint64_t v115 = (uint64_t) v10;
    TASSIGN(v114, v115);
    pipe_barrier(PIPE_V);
    RoundMode v116 = RoundMode::CAST_ROUND;
    SaturationMode v117 = SaturationMode::OFF;
    TCVT(v114, v110, v116, v117);
    // pto: %oq_i8_inline695__tile
    ;
    Tile<TileType::Vec, int8_t, 32, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v118 = Tile<TileType::Vec, int8_t, 32, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v20);
    // pto: %oq_i8_inline695__tile
    ;
    uint64_t v119 = (uint64_t) v10;
    TASSIGN(v118, v119);
    pipe_barrier(PIPE_V);
    RoundMode v120 = RoundMode::CAST_TRUNC;
    SaturationMode v121 = SaturationMode::ON;
    TCVT(v118, v114, v120, v121);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    // pto: %o_r_i8_inline656__iter_v3_pview
    ;
    __gm__ int8_t* v122 = PTOAS__GLOBAL_TENSOR_DATA(v48);
    // pto: %o_r_i8_inline656__iter_v3_pview
    ;
    const int64_t v123 = 0;
    // pto: %o_r_i8_inline656__iter_v3_pview
    ;
    const int64_t v124 = 8192;
    // pto: %o_r_i8_inline656__iter_v3_pview
    ;
    pto::Shape<1, 1, 1, 32, 256> v125 = pto::Shape<1, 1, 1, 32, 256>();
    // pto: %o_r_i8_inline656__iter_v3_pview
    ;
    pto::Stride<262144, 262144, 262144, 8192, 1> v126 = pto::Stride<262144, 262144, 262144, 8192, 1>();
    // pto: %o_r_i8_inline656__iter_v3_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<262144, 262144, 262144, 8192, 1>, pto::Layout::ND> v127 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<262144, 262144, 262144, 8192, 1>, pto::Layout::ND>(v122 + (v123 + v83 * v124 + v101), v125, v126);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TSTORE(v127, v118);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  }
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}