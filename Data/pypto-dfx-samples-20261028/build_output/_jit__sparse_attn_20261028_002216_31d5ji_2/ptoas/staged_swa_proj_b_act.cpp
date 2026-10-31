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

AICORE void staged_swa_proj_b_act(__gm__ float* v1, __gm__ int32_t* v2, __gm__ float* v3, __gm__ bfloat16_t* v4, int64_t v5, int64_t v6, int64_t v7, int64_t v8, int32_t v9, int32_t v10) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v11 = 0;
  // pto: %c2048_i64
  const int64_t v12 = 2048;
  // pto: %c34816_i64
  const int64_t v13 = 34816;
  // pto: %c67584_i64
  const int64_t v14 = 67584;
  // pto: %c4096_index
  const int64_t v15 = 4096;
  // pto: %c1_index
  const int64_t v16 = 1;
  // pto: %c32768_index
  const int64_t v17 = 32768;
  // pto: %c8_index
  const int64_t v18 = 8;
  // pto: %c512_index
  const int64_t v19 = 512;
  // pto: %c32_index
  const int64_t v20 = 32;
  // pto: %c0_index
  const int64_t v21 = 0;
  // pto: %c16_index
  const int64_t v22 = 16;
  // pto: %cst_12
  const float v23 = 0.0f;
  // pto: %wo_b_scale__ssa_v0_view
  const int64_t v24 = 1;
  // pto: %wo_b_scale__ssa_v0_view
  const int64_t v25 = 1;
  // pto: %wo_b_scale__ssa_v0_view
  const int64_t v26 = 1;
  // pto: %wo_b_scale__ssa_v0_view
  const int64_t v27 = 1;
  // pto: %wo_b_scale__ssa_v0_view
  int64_t v28 = v15 * v16;
  // pto: %wo_b_scale__ssa_v0_view
  int64_t v29 = v27 * v28;
  // pto: %wo_b_scale__ssa_v0_view
  int64_t v30 = v26 * v29;
  // pto: %wo_b_scale__ssa_v0_view
  pto::Shape<1, 1, 1, 1, -1> v31 = pto::Shape<1, 1, 1, 1, -1>(v24, v25, v26, v27, v15);
  // pto: %wo_b_scale__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v32 = pto::Stride<-1, -1, -1, -1, -1>(v25 * v30, v30, v29, v28, v16);
  // pto: %wo_b_scale__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v33 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v31, v32);
  // pto: %partials_inline658__rv_v2_view
  const int64_t v34 = 1;
  // pto: %partials_inline658__rv_v2_view
  const int64_t v35 = 1;
  // pto: %partials_inline658__rv_v2_view
  const int64_t v36 = 1;
  // pto: %partials_inline658__rv_v2_view
  int64_t v37 = (int64_t) v7;
  // pto: %partials_inline658__rv_v2_view
  int64_t v38 = v37 * v17;
  // pto: %partials_inline658__rv_v2_view
  int64_t v39 = v36 * v38;
  // pto: %partials_inline658__rv_v2_view
  pto::Shape<1, 1, 1, -1, -1> v40 = pto::Shape<1, 1, 1, -1, -1>(v34, v35, v36, v37, v17);
  // pto: %partials_inline658__rv_v2_view
  pto::Stride<-1, -1, -1, -1, -1> v41 = pto::Stride<-1, -1, -1, -1, -1>(v35 * v39, v39, v38, v17, v16);
  // pto: %partials_inline658__rv_v2_view
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v42 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v40, v41);
  // pto: %act_scale_dq_inline657__rv_v2_view
  const int64_t v43 = 1;
  // pto: %act_scale_dq_inline657__rv_v2_view
  const int64_t v44 = 1;
  // pto: %act_scale_dq_inline657__rv_v2_view
  const int64_t v45 = 1;
  // pto: %act_scale_dq_inline657__rv_v2_view
  int64_t v46 = (int64_t) v7;
  // pto: %act_scale_dq_inline657__rv_v2_view
  int64_t v47 = v18 * v46;
  // pto: %act_scale_dq_inline657__rv_v2_view
  int64_t v48 = v45 * v47;
  // pto: %act_scale_dq_inline657__rv_v2_view
  pto::Shape<1, 1, 1, -1, -1> v49 = pto::Shape<1, 1, 1, -1, -1>(v43, v44, v45, v18, (int64_t) v7);
  // pto: %act_scale_dq_inline657__rv_v2_view
  pto::Stride<-1, -1, -1, -1, -1> v50 = pto::Stride<-1, -1, -1, -1, -1>(v44 * v48, v48, v47, v46, v16);
  // pto: %act_scale_dq_inline657__rv_v2_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v51 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v49, v50);
  // pto: %attn_out_view_inline653__ssa_v0_view
  const int64_t v52 = 1;
  // pto: %attn_out_view_inline653__ssa_v0_view
  const int64_t v53 = 1;
  // pto: %attn_out_view_inline653__ssa_v0_view
  const int64_t v54 = 1;
  // pto: %attn_out_view_inline653__ssa_v0_view
  int64_t v55 = (int64_t) v8;
  // pto: %attn_out_view_inline653__ssa_v0_view
  int64_t v56 = v55 * v15;
  // pto: %attn_out_view_inline653__ssa_v0_view
  int64_t v57 = v54 * v56;
  // pto: %attn_out_view_inline653__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v58 = pto::Shape<1, 1, 1, -1, -1>(v52, v53, v54, v55, v15);
  // pto: %attn_out_view_inline653__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v59 = pto::Stride<-1, -1, -1, -1, -1>(v53 * v57, v57, v56, v15, v16);
  // pto: %attn_out_view_inline653__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v60 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v58, v59);
  // pto: %act_idx_inline720__ssa_v0
  int64_t v61 = (int64_t) v9;
  // pto: %1
  int64_t v62 = v61 / v5;
  // pto: %2
  int64_t v63 = (int64_t) ((uint64_t) v62 * (uint64_t) v5);
  // pto: %3
  int64_t v64 = (int64_t) ((uint64_t) v61 - (uint64_t) v63);
  // pto: %4
  int64_t v65 = (int64_t) ((uint64_t) v62 * (uint64_t) v19);
  // pto: %5
  int64_t v66 = (int64_t) ((uint64_t) v64 * (uint64_t) v20);
  // pto: %wb_scale_inline725__tile
  Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v67 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v19);
  // pto: %wb_scale_inline725__tile
  uint64_t v68 = (uint64_t) v11;
  TASSIGN(v67, v68);
  // pto: %6
  int64_t v69 = v65 < v21 ? v21 : v65;
  // pto: %wo_b_scale__ssa_v0_pview
  __gm__ float* v70 = PTOAS__GLOBAL_TENSOR_DATA(v33);
  // pto: %wo_b_scale__ssa_v0_pview
  const int64_t v71 = 0;
  // pto: %wo_b_scale__ssa_v0_pview
  pto::Shape<1, 1, 1, 1, 512> v72 = pto::Shape<1, 1, 1, 1, 512>();
  // pto: %wo_b_scale__ssa_v0_pview
  pto::Stride<512, 512, 512, 512, 1> v73 = pto::Stride<512, 512, 512, 512, 1>();
  // pto: %wo_b_scale__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v74 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v70 + (v71 + v69), v72, v73);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  TLOAD(v67, v74);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  // pto: %wb_scale_chunk_inline726__tile
  Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v75 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v19);
  // pto: %wb_scale_chunk_inline726__tile
  uint64_t v76 = (uint64_t) v11;
  TASSIGN(v75, v76);
  // pto: %7
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  for (int64_t v77 = v66; v77 < ((int64_t) ((uint64_t) v66 + (uint64_t) v20)); v77 += v22) {
    // pto: %acc_inline728__tile
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v78 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v19);
    // pto: %acc_inline728__tile
    ;
    uint64_t v79 = (uint64_t) v12;
    TASSIGN(v78, v79);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    pipe_barrier(PIPE_V);
    TEXPANDS(v78, v23);
    for (int64_t v80 = v21; v80 < v18; v80 += v16) {
      // pto: %p_g_inline730__tile
      ;
      Tile<TileType::Vec, int32_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v81 = Tile<TileType::Vec, int32_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v19);
      // pto: %p_g_inline730__tile
      ;
      uint64_t v82 = (uint64_t) v13;
      TASSIGN(v81, v82);
      // pto: %8
      ;
      int64_t v83 = v77 < v21 ? v21 : v77;
      // pto: %9
      ;
      int64_t v84 = (int64_t) ((uint64_t) v80 * (uint64_t) v15);
      // pto: %10
      ;
      int64_t v85 = (int64_t) ((uint64_t) v84 + (uint64_t) v65);
      // pto: %11
      ;
      int64_t v86 = v85 < v21 ? v21 : v85;
      // pto: %partials_inline658__rv_v2_pview
      ;
      __gm__ int32_t* v87 = PTOAS__GLOBAL_TENSOR_DATA(v42);
      // pto: %partials_inline658__rv_v2_pview
      ;
      const int64_t v88 = 0;
      // pto: %partials_inline658__rv_v2_pview
      ;
      const int64_t v89 = 32768;
      // pto: %partials_inline658__rv_v2_pview
      ;
      pto::Shape<1, 1, 1, 16, 512> v90 = pto::Shape<1, 1, 1, 16, 512>();
      // pto: %partials_inline658__rv_v2_pview
      ;
      pto::Stride<524288, 524288, 524288, 32768, 1> v91 = pto::Stride<524288, 524288, 524288, 32768, 1>();
      // pto: %partials_inline658__rv_v2_pview
      ;
      GlobalTensor<int32_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<524288, 524288, 524288, 32768, 1>, pto::Layout::ND> v92 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<524288, 524288, 524288, 32768, 1>, pto::Layout::ND>(v87 + (v88 + v83 * v89 + v86), v90, v91);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TLOAD(v81, v92);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      // pto: %g_scale_row_inline731__tile
      ;
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v93 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v22);
      // pto: %g_scale_row_inline731__tile
      ;
      uint64_t v94 = (uint64_t) v14;
      TASSIGN(v93, v94);
      // pto: %12
      ;
      int64_t v95 = v80 < v21 ? v21 : v80;
      // pto: %act_scale_dq_inline657__rv_v2_pview
      ;
      int64_t v96 = (int64_t) v7;
      // pto: %act_scale_dq_inline657__rv_v2_pview
      ;
      const int64_t v97 = 0;
      // pto: %act_scale_dq_inline657__rv_v2_pview
      ;
      __gm__ float* v98 = PTOAS__GLOBAL_TENSOR_DATA(v51);
      // pto: %act_scale_dq_inline657__rv_v2_pview
      ;
      const int64_t v99 = 1;
      // pto: %act_scale_dq_inline657__rv_v2_pview
      ;
      const int64_t v100 = 1;
      // pto: %act_scale_dq_inline657__rv_v2_pview
      ;
      const int64_t v101 = 1;
      // pto: %act_scale_dq_inline657__rv_v2_pview
      ;
      int64_t v102 = v16 * v96;
      // pto: %act_scale_dq_inline657__rv_v2_pview
      ;
      int64_t v103 = v101 * v102;
      // pto: %act_scale_dq_inline657__rv_v2_pview
      ;
      pto::Shape<1, 1, 1, 1, 16> v104 = pto::Shape<1, 1, 1, 1, 16>(v99, v100, v101, v16, v22);
      // pto: %act_scale_dq_inline657__rv_v2_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v105 = pto::Stride<-1, -1, -1, -1, -1>(v100 * v103, v103, v102, v96, v16);
      // pto: %act_scale_dq_inline657__rv_v2_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v106 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v98 + (v97 + v95 * v96 + v83 * v16), v104, v105);
      TLOAD(v93, v106);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      // pto: %g_scale_inline732__tile
      ;
      Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v107 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v16);
      // pto: %g_scale_inline732__tile
      ;
      uint64_t v108 = (uint64_t) v14;
      TASSIGN(v107, v108);
      // pto: %p_g_f32_inline733__tile
      ;
      Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v109 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v19);
      // pto: %p_g_f32_inline733__tile
      ;
      uint64_t v110 = (uint64_t) v13;
      TASSIGN(v109, v110);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      RoundMode v111 = RoundMode::CAST_NONE;
      SaturationMode v112 = SaturationMode::OFF;
      TCVT(v109, v81, v111, v112);
      // pto: %t__tile
      ;
      Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v113 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v19);
      // pto: %t__tile
      ;
      uint64_t v114 = (uint64_t) v13;
      TASSIGN(v113, v114);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      TROWEXPANDMUL(v113, v109, v107);
      // pto: %0
      ;
      Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v115 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v19);
      // pto: %0
      ;
      uint64_t v116 = (uint64_t) v12;
      TASSIGN(v115, v116);
      pipe_barrier(PIPE_V);
      TADD(v115, v78, v113);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    };
    // pto: %out_t_inline734__tile
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v117 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v19);
    // pto: %out_t_inline734__tile
    ;
    uint64_t v118 = (uint64_t) v12;
    TASSIGN(v117, v118);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v117, v78, v75);
    // pto: %out_bf16_inline735__tile
    ;
    Tile<TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v119 = Tile<TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v19);
    // pto: %out_bf16_inline735__tile
    ;
    uint64_t v120 = (uint64_t) v12;
    TASSIGN(v119, v120);
    pipe_barrier(PIPE_V);
    RoundMode v121 = RoundMode::CAST_RINT;
    SaturationMode v122 = SaturationMode::OFF;
    TCVT(v119, v117, v121, v122);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %14
    ;
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    if (v77 < v8) {
      // pto: %15
      ;
      int64_t v123 = (int64_t) ((uint64_t) v8 - (uint64_t) v77);
      // pto: %16
      ;
      int64_t v124 = v123 < v22 ? v123 : v22;
      // pto: %output_tile_inline737__tile
      ;
      Tile<TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v125 = Tile<TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v124, v19);
      // pto: %output_tile_inline737__tile
      ;
      uint64_t v126 = (uint64_t) v12;
      TASSIGN(v125, v126);
      // pto: %slice_view
      ;
      Tile<TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v127 = Tile<TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v124);
      // pto: %slice_view
      ;
      uint64_t v128 = (uint64_t) v12;
      TASSIGN(v127, v128);
      // pto: %17
      ;
      int64_t v129 = v77 < v21 ? v21 : v77;
      // pto: %attn_out_view_inline653__ssa_v0_pview
      ;
      const int64_t v130 = 0;
      // pto: %attn_out_view_inline653__ssa_v0_pview
      ;
      __gm__ bfloat16_t* v131 = PTOAS__GLOBAL_TENSOR_DATA(v60);
      // pto: %attn_out_view_inline653__ssa_v0_pview
      ;
      const int64_t v132 = 1;
      // pto: %attn_out_view_inline653__ssa_v0_pview
      ;
      const int64_t v133 = 1;
      // pto: %attn_out_view_inline653__ssa_v0_pview
      ;
      const int64_t v134 = 1;
      // pto: %attn_out_view_inline653__ssa_v0_pview
      ;
      int64_t v135 = v124 * v15;
      // pto: %attn_out_view_inline653__ssa_v0_pview
      ;
      int64_t v136 = v134 * v135;
      // pto: %attn_out_view_inline653__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, -1, 512> v137 = pto::Shape<1, 1, 1, -1, 512>(v132, v133, v134, v124, v19);
      // pto: %attn_out_view_inline653__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v138 = pto::Stride<-1, -1, -1, -1, -1>(v133 * v136, v136, v135, v15, v16);
      // pto: %attn_out_view_inline653__ssa_v0_pview
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v139 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v131 + (v130 + v129 * v15 + v69 * v16), v137, v138);
      TSTORE(v139, v127);
    };
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}