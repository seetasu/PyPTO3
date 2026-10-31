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

AICORE void ffn_norm(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ float* v3, __gm__ float* v4, __gm__ float* v5, __gm__ float* v6, int64_t v7, int32_t v8, int32_t v9) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c32800_i64
  const int64_t v10 = 32800;
  // pto: %c16416_i64
  const int64_t v11 = 16416;
  // pto: %c0_i64
  const int64_t v12 = 0;
  // pto: %c16384_i64
  const int64_t v13 = 16384;
  // pto: %c8_index
  const int64_t v14 = 8;
  // pto: %c4096_index
  const int64_t v15 = 4096;
  // pto: %c1_index
  const int64_t v16 = 1;
  // pto: %c0_index
  const int64_t v17 = 0;
  // pto: %c512_index
  const int64_t v18 = 512;
  // pto: %cst_9
  const float v19 = 2.44140625E-4f;
  // pto: %cst_10
  const float v20 = 9.99999997E-7f;
  // pto: %cst_11
  const float v21 = 9.99999974E-5f;
  // pto: %cst_12
  const float v22 = 127.0f;
  // pto: %cst_13
  const float v23 = 0.00787401571f;
  // pto: %x_mixed_inline0__rv_v2_view
  const int64_t v24 = 1;
  // pto: %x_mixed_inline0__rv_v2_view
  const int64_t v25 = 1;
  // pto: %x_mixed_inline0__rv_v2_view
  const int64_t v26 = 1;
  // pto: %x_mixed_inline0__rv_v2_view
  int64_t v27 = v14 * v15;
  // pto: %x_mixed_inline0__rv_v2_view
  int64_t v28 = v26 * v27;
  // pto: %x_mixed_inline0__rv_v2_view
  pto::Shape<1, 1, 1, -1, -1> v29 = pto::Shape<1, 1, 1, -1, -1>(v24, v25, v26, v14, v15);
  // pto: %x_mixed_inline0__rv_v2_view
  pto::Stride<-1, -1, -1, -1, -1> v30 = pto::Stride<-1, -1, -1, -1, -1>(v25 * v28, v28, v27, v15, v16);
  // pto: %x_mixed_inline0__rv_v2_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v31 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v29, v30);
  // pto: %norm_w_2d_inline178__ssa_v0_view
  const int64_t v32 = 1;
  // pto: %norm_w_2d_inline178__ssa_v0_view
  const int64_t v33 = 1;
  // pto: %norm_w_2d_inline178__ssa_v0_view
  const int64_t v34 = 1;
  // pto: %norm_w_2d_inline178__ssa_v0_view
  int64_t v35 = v16 * v15;
  // pto: %norm_w_2d_inline178__ssa_v0_view
  int64_t v36 = v34 * v35;
  // pto: %norm_w_2d_inline178__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v37 = pto::Shape<1, 1, 1, -1, -1>(v32, v33, v34, v16, v15);
  // pto: %norm_w_2d_inline178__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v38 = pto::Stride<-1, -1, -1, -1, -1>(v33 * v36, v36, v35, v15, v16);
  // pto: %norm_w_2d_inline178__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v39 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v37, v38);
  // pto: %xg_buf_inline170__ssa_v0_view
  const int64_t v40 = 1;
  // pto: %xg_buf_inline170__ssa_v0_view
  const int64_t v41 = 1;
  // pto: %xg_buf_inline170__ssa_v0_view
  const int64_t v42 = 1;
  // pto: %xg_buf_inline170__ssa_v0_view
  int64_t v43 = (int64_t) v7;
  // pto: %xg_buf_inline170__ssa_v0_view
  int64_t v44 = v43 * v15;
  // pto: %xg_buf_inline170__ssa_v0_view
  int64_t v45 = v42 * v44;
  // pto: %xg_buf_inline170__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v46 = pto::Shape<1, 1, 1, -1, -1>(v40, v41, v42, v43, v15);
  // pto: %xg_buf_inline170__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v47 = pto::Stride<-1, -1, -1, -1, -1>(v41 * v45, v45, v44, v15, v16);
  // pto: %xg_buf_inline170__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v48 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v46, v47);
  // pto: %inv_rms_buf_inline171__ssa_v0_view
  const int64_t v49 = 1;
  // pto: %inv_rms_buf_inline171__ssa_v0_view
  const int64_t v50 = 1;
  // pto: %inv_rms_buf_inline171__ssa_v0_view
  const int64_t v51 = 1;
  // pto: %inv_rms_buf_inline171__ssa_v0_view
  int64_t v52 = (int64_t) v7;
  // pto: %inv_rms_buf_inline171__ssa_v0_view
  int64_t v53 = v52 * v16;
  // pto: %inv_rms_buf_inline171__ssa_v0_view
  int64_t v54 = v51 * v53;
  // pto: %inv_rms_buf_inline171__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v55 = pto::Shape<1, 1, 1, -1, -1>(v49, v50, v51, v52, v16);
  // pto: %inv_rms_buf_inline171__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v56 = pto::Stride<-1, -1, -1, -1, -1>(v50 * v54, v54, v53, v16, (int64_t) v7);
  // pto: %inv_rms_buf_inline171__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v57 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v4, v55, v56);
  // pto: %x_norm_scale_inline4__ssa_v0_view
  const int64_t v58 = 1;
  // pto: %x_norm_scale_inline4__ssa_v0_view
  const int64_t v59 = 1;
  // pto: %x_norm_scale_inline4__ssa_v0_view
  const int64_t v60 = 1;
  // pto: %x_norm_scale_inline4__ssa_v0_view
  int64_t v61 = v14 * v16;
  // pto: %x_norm_scale_inline4__ssa_v0_view
  int64_t v62 = v60 * v61;
  // pto: %x_norm_scale_inline4__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v63 = pto::Shape<1, 1, 1, -1, -1>(v58, v59, v60, v14, v16);
  // pto: %x_norm_scale_inline4__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v64 = pto::Stride<-1, -1, -1, -1, -1>(v59 * v62, v62, v61, v16, v14);
  // pto: %x_norm_scale_inline4__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v65 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v5, v63, v64);
  // pto: %xn_scale_buf_inline172__ssa_v0_view
  const int64_t v66 = 1;
  // pto: %xn_scale_buf_inline172__ssa_v0_view
  const int64_t v67 = 1;
  // pto: %xn_scale_buf_inline172__ssa_v0_view
  const int64_t v68 = 1;
  // pto: %xn_scale_buf_inline172__ssa_v0_view
  int64_t v69 = (int64_t) v7;
  // pto: %xn_scale_buf_inline172__ssa_v0_view
  int64_t v70 = v69 * v16;
  // pto: %xn_scale_buf_inline172__ssa_v0_view
  int64_t v71 = v68 * v70;
  // pto: %xn_scale_buf_inline172__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v72 = pto::Shape<1, 1, 1, -1, -1>(v66, v67, v68, v69, v16);
  // pto: %xn_scale_buf_inline172__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v73 = pto::Stride<-1, -1, -1, -1, -1>(v67 * v71, v71, v70, v16, (int64_t) v7);
  // pto: %xn_scale_buf_inline172__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v74 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v6, v72, v73);
  // pto: %tok_inline179__ssa_v0
  int64_t v75 = (int64_t) v8;
  // pto: %t__tmp_v51
  Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v76 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
  // pto: %t__tmp_v51
  uint64_t v77 = (uint64_t) v10;
  TASSIGN(v76, v77);
  // pto: %0
  int64_t v78 = v75 < v17 ? v17 : v75;
  // pto: %x_mixed_inline0__rv_v2_pview
  __gm__ bfloat16_t* v79 = PTOAS__GLOBAL_TENSOR_DATA(v31);
  // pto: %x_mixed_inline0__rv_v2_pview
  const int64_t v80 = 0;
  // pto: %x_mixed_inline0__rv_v2_pview
  const int64_t v81 = 4096;
  // pto: %x_mixed_inline0__rv_v2_pview
  pto::Shape<1, 1, 1, 1, 4096> v82 = pto::Shape<1, 1, 1, 1, 4096>();
  // pto: %x_mixed_inline0__rv_v2_pview
  pto::Stride<4096, 4096, 4096, 4096, 1> v83 = pto::Stride<4096, 4096, 4096, 4096, 1>();
  // pto: %x_mixed_inline0__rv_v2_pview
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v84 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v79 + (v80 + v78 * v81), v82, v83);
  TLOAD(v76, v84);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  // pto: %rms_x_inline180__ssa_v0
  Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v85 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
  // pto: %rms_x_inline180__ssa_v0
  uint64_t v86 = (uint64_t) v11;
  TASSIGN(v85, v86);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  RoundMode v87 = RoundMode::CAST_ROUND;
  SaturationMode v88 = SaturationMode::OFF;
  TCVT(v85, v76, v87, v88);
  // pto: %t__tmp_v52
  Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v89 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
  // pto: %t__tmp_v52
  uint64_t v90 = (uint64_t) v12;
  TASSIGN(v89, v90);
  // pto: %norm_w_2d_inline178__ssa_v0_pview
  __gm__ bfloat16_t* v91 = PTOAS__GLOBAL_TENSOR_DATA(v39);
  // pto: %norm_w_2d_inline178__ssa_v0_pview
  pto::Shape<1, 1, 1, 1, 4096> v92 = pto::Shape<1, 1, 1, 1, 4096>();
  // pto: %norm_w_2d_inline178__ssa_v0_pview
  pto::Stride<4096, 4096, 4096, 4096, 1> v93 = pto::Stride<4096, 4096, 4096, 4096, 1>();
  // pto: %norm_w_2d_inline178__ssa_v0_pview
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v94 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v91, v92, v93);
  TLOAD(v89, v94);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
  // pto: %rms_w_inline181__ssa_v0
  Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v95 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
  // pto: %rms_w_inline181__ssa_v0
  uint64_t v96 = (uint64_t) v10;
  TASSIGN(v95, v96);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
  pipe_barrier(PIPE_V);
  RoundMode v97 = RoundMode::CAST_ROUND;
  SaturationMode v98 = SaturationMode::OFF;
  TCVT(v95, v89, v97, v98);
  // pto: %xg_inline182__ssa_v0
  Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v99 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
  // pto: %xg_inline182__ssa_v0
  uint64_t v100 = (uint64_t) v10;
  TASSIGN(v99, v100);
  pipe_barrier(PIPE_V);
  TMUL(v99, v85, v95);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  // pto: %xg_buf_inline170__ssa_v0_pview
  __gm__ float* v101 = PTOAS__GLOBAL_TENSOR_DATA(v48);
  // pto: %xg_buf_inline170__ssa_v0_pview
  const int64_t v102 = 0;
  // pto: %xg_buf_inline170__ssa_v0_pview
  const int64_t v103 = 4096;
  // pto: %xg_buf_inline170__ssa_v0_pview
  pto::Shape<1, 1, 1, 1, 4096> v104 = pto::Shape<1, 1, 1, 1, 4096>();
  // pto: %xg_buf_inline170__ssa_v0_pview
  pto::Stride<4096, 4096, 4096, 4096, 1> v105 = pto::Stride<4096, 4096, 4096, 4096, 1>();
  // pto: %xg_buf_inline170__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v106 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v101 + (v102 + v78 * v103), v104, v105);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v106, v99);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  // pto: %t__tmp_v53
  Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v107 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
  // pto: %t__tmp_v53
  uint64_t v108 = (uint64_t) v11;
  TASSIGN(v107, v108);
  pipe_barrier(PIPE_V);
  TMUL(v107, v85, v85);
  // pto: %sq_rows_inline183__ssa_v0
  Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v109 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v18);
  // pto: %sq_rows_inline183__ssa_v0
  uint64_t v110 = (uint64_t) v11;
  TASSIGN(v109, v110);
  // pto: %sq_partial_tmp_inline184__ssa_v0
  Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v111 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v18);
  // pto: %sq_partial_tmp_inline184__ssa_v0
  uint64_t v112 = (uint64_t) v12;
  TASSIGN(v111, v112);
  // pto: %sq_partial_inline185__ssa_v0
  Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v113 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v16);
  // pto: %sq_partial_inline185__ssa_v0
  uint64_t v114 = (uint64_t) v13;
  TASSIGN(v113, v114);
  pipe_barrier(PIPE_V);
  TROWSUM(v113, v109, v111);
  // pto: %sq_reduce_inline186__ssa_v0
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v115 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v14);
  // pto: %sq_reduce_inline186__ssa_v0
  uint64_t v116 = (uint64_t) v11;
  TASSIGN(v115, v116);
  // pto: %t__tmp_v54
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v117 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v14);
  // pto: %t__tmp_v54
  uint64_t v118 = (uint64_t) v13;
  TASSIGN(v117, v118);
  // pto: %assemble_view
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v119;
  // pto: %assemble_view
  uint64_t v120 = (uint64_t) v11;
  TASSIGN(v119, v120);
  pipe_barrier(PIPE_V);
  TMOV(v119, v117);
  v115.SetValidShape(v16, v14);
  // pto: %sq_sum_tmp_inline188__ssa_v0
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v121 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v14);
  // pto: %sq_sum_tmp_inline188__ssa_v0
  uint64_t v122 = (uint64_t) v12;
  TASSIGN(v121, v122);
  // pto: %sq_sum_inline189__ssa_v0
  Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v123 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
  // pto: %sq_sum_inline189__ssa_v0
  uint64_t v124 = (uint64_t) v13;
  TASSIGN(v123, v124);
  pipe_barrier(PIPE_V);
  TROWSUM(v123, v115, v121);
  // pto: %t__tmp_v55
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v125 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
  // pto: %t__tmp_v55
  uint64_t v126 = (uint64_t) v13;
  TASSIGN(v125, v126);
  v125.SetValidShape(v16, v16);
  // pto: %t__tmp_v56
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v127 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
  // pto: %t__tmp_v56
  uint64_t v128 = (uint64_t) v11;
  TASSIGN(v127, v128);
  pipe_barrier(PIPE_V);
  TMULS(v127, v125, v19);
  // pto: %t__tmp_v57
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v129 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
  // pto: %t__tmp_v57
  uint64_t v130 = (uint64_t) v11;
  TASSIGN(v129, v130);
  pipe_barrier(PIPE_V);
  TADDS(v129, v127, v20);
  // pto: %t__tmp_v58
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v131 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
  // pto: %t__tmp_v58
  uint64_t v132 = (uint64_t) v11;
  TASSIGN(v131, v132);
  pipe_barrier(PIPE_V);
  TSQRT(v131, v129);
  // pto: %inv_rms_inline191__ssa_v0
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v133 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
  // pto: %inv_rms_inline191__ssa_v0
  uint64_t v134 = (uint64_t) v12;
  TASSIGN(v133, v134);
  pipe_barrier(PIPE_V);
  TRECIP(v133, v131);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  // pto: %inv_rms_buf_inline171__ssa_v0_pview
  int64_t v135 = (int64_t) v7;
  // pto: %inv_rms_buf_inline171__ssa_v0_pview
  const int64_t v136 = 0;
  // pto: %inv_rms_buf_inline171__ssa_v0_pview
  __gm__ float* v137 = PTOAS__GLOBAL_TENSOR_DATA(v57);
  // pto: %inv_rms_buf_inline171__ssa_v0_pview
  const int64_t v138 = 1;
  // pto: %inv_rms_buf_inline171__ssa_v0_pview
  const int64_t v139 = 1;
  // pto: %inv_rms_buf_inline171__ssa_v0_pview
  const int64_t v140 = 1;
  // pto: %inv_rms_buf_inline171__ssa_v0_pview
  int64_t v141 = v16 * v16;
  // pto: %inv_rms_buf_inline171__ssa_v0_pview
  int64_t v142 = v140 * v141;
  // pto: %inv_rms_buf_inline171__ssa_v0_pview
  pto::Shape<1, 1, 1, 1, 1> v143 = pto::Shape<1, 1, 1, 1, 1>(v138, v139, v140, v16, v16);
  // pto: %inv_rms_buf_inline171__ssa_v0_pview
  pto::Stride<-1, -1, -1, -1, -1> v144 = pto::Stride<-1, -1, -1, -1, -1>(v139 * v142, v142, v141, v16, v135);
  // pto: %inv_rms_buf_inline171__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v145 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v137 + (v136 + v78 * v16 + v17 * v135), v143, v144);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  TSTORE(v145, v133);
  // pto: %t__tmp_v59
  Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v146 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
  // pto: %t__tmp_v59
  uint64_t v147 = (uint64_t) v11;
  TASSIGN(v146, v147);
  pipe_barrier(PIPE_V);
  TABS(v146, v99);
  // pto: %xg_abs_rows_inline192__ssa_v0
  Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v148 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v18);
  // pto: %xg_abs_rows_inline192__ssa_v0
  uint64_t v149 = (uint64_t) v11;
  TASSIGN(v148, v149);
  // pto: %amax_partial_tmp_inline193__ssa_v0
  Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v150 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v18);
  // pto: %amax_partial_tmp_inline193__ssa_v0
  uint64_t v151 = (uint64_t) v10;
  TASSIGN(v150, v151);
  // pto: %amax_partial_inline194__ssa_v0
  Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v152 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v16);
  // pto: %amax_partial_inline194__ssa_v0
  uint64_t v153 = (uint64_t) v13;
  TASSIGN(v152, v153);
  pipe_barrier(PIPE_V);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  TROWMAX(v152, v148, v150);
  // pto: %amax_reduce_inline195__ssa_v0
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v154 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v14);
  // pto: %amax_reduce_inline195__ssa_v0
  uint64_t v155 = (uint64_t) v11;
  TASSIGN(v154, v155);
  // pto: %t__tmp_v60
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v156 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v14);
  // pto: %t__tmp_v60
  uint64_t v157 = (uint64_t) v13;
  TASSIGN(v156, v157);
  // pto: %3
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v158;
  // pto: %3
  uint64_t v159 = (uint64_t) v11;
  TASSIGN(v158, v159);
  pipe_barrier(PIPE_V);
  TMOV(v158, v156);
  v154.SetValidShape(v16, v14);
  // pto: %amax_tmp_inline197__ssa_v0
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v160 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v14);
  // pto: %amax_tmp_inline197__ssa_v0
  uint64_t v161 = (uint64_t) v10;
  TASSIGN(v160, v161);
  // pto: %xg_amax_inline198__ssa_v0
  Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v162 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
  // pto: %xg_amax_inline198__ssa_v0
  uint64_t v163 = (uint64_t) v13;
  TASSIGN(v162, v163);
  pipe_barrier(PIPE_V);
  TROWMAX(v162, v154, v160);
  // pto: %t__tmp_v61
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v164 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
  // pto: %t__tmp_v61
  uint64_t v165 = (uint64_t) v13;
  TASSIGN(v164, v165);
  v164.SetValidShape(v16, v16);
  // pto: %amax_eps_inline200__ssa_v0
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v166 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v14);
  // pto: %amax_eps_inline200__ssa_v0
  uint64_t v167 = (uint64_t) v11;
  TASSIGN(v166, v167);
  pipe_barrier(PIPE_V);
  TEXPANDS(v166, v21);
  v166.SetValidShape(v16, v16);
  // pto: %xg_amax_v2_inline202__ssa_v0
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v168 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
  // pto: %xg_amax_v2_inline202__ssa_v0
  uint64_t v169 = (uint64_t) v11;
  TASSIGN(v168, v169);
  pipe_barrier(PIPE_V);
  TMAX(v168, v164, v166);
  // pto: %scale_max_inline203__ssa_v0
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v170 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v14);
  // pto: %scale_max_inline203__ssa_v0
  uint64_t v171 = (uint64_t) v10;
  TASSIGN(v170, v171);
  TEXPANDS(v170, v22);
  v170.SetValidShape(v16, v16);
  // pto: %xg_sq_inline205__ssa_v0
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v172 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
  // pto: %xg_sq_inline205__ssa_v0
  uint64_t v173 = (uint64_t) v10;
  TASSIGN(v172, v173);
  pipe_barrier(PIPE_V);
  TDIV(v172, v170, v168);
  // pto: %xg_dequant_scale_inline206__ssa_v0
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v174 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
  // pto: %xg_dequant_scale_inline206__ssa_v0
  uint64_t v175 = (uint64_t) v11;
  TASSIGN(v174, v175);
  pipe_barrier(PIPE_V);
  TMULS(v174, v168, v23);
  // pto: %x_norm_dequant_scale_inline207__ssa_v0
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v176 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
  // pto: %x_norm_dequant_scale_inline207__ssa_v0
  uint64_t v177 = (uint64_t) v11;
  TASSIGN(v176, v177);
  pipe_barrier(PIPE_V);
  TMUL(v176, v174, v133);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
  // pto: %x_norm_scale_inline4__ssa_v0_pview
  __gm__ float* v178 = PTOAS__GLOBAL_TENSOR_DATA(v65);
  // pto: %x_norm_scale_inline4__ssa_v0_pview
  const int64_t v179 = 0;
  // pto: %x_norm_scale_inline4__ssa_v0_pview
  pto::Shape<1, 1, 1, 1, 1> v180 = pto::Shape<1, 1, 1, 1, 1>();
  // pto: %x_norm_scale_inline4__ssa_v0_pview
  pto::Stride<1, 1, 1, 1, 8> v181 = pto::Stride<1, 1, 1, 1, 8>();
  // pto: %x_norm_scale_inline4__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 8>, pto::Layout::DN> v182 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 8>, pto::Layout::DN>(v178 + (v179 + v78), v180, v181);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
  TSTORE(v182, v176);
  // pto: %xn_scale_buf_inline172__ssa_v0_pview
  int64_t v183 = (int64_t) v7;
  // pto: %xn_scale_buf_inline172__ssa_v0_pview
  const int64_t v184 = 0;
  // pto: %xn_scale_buf_inline172__ssa_v0_pview
  __gm__ float* v185 = PTOAS__GLOBAL_TENSOR_DATA(v74);
  // pto: %xn_scale_buf_inline172__ssa_v0_pview
  const int64_t v186 = 1;
  // pto: %xn_scale_buf_inline172__ssa_v0_pview
  const int64_t v187 = 1;
  // pto: %xn_scale_buf_inline172__ssa_v0_pview
  const int64_t v188 = 1;
  // pto: %xn_scale_buf_inline172__ssa_v0_pview
  int64_t v189 = v16 * v16;
  // pto: %xn_scale_buf_inline172__ssa_v0_pview
  int64_t v190 = v188 * v189;
  // pto: %xn_scale_buf_inline172__ssa_v0_pview
  pto::Shape<1, 1, 1, 1, 1> v191 = pto::Shape<1, 1, 1, 1, 1>(v186, v187, v188, v16, v16);
  // pto: %xn_scale_buf_inline172__ssa_v0_pview
  pto::Stride<-1, -1, -1, -1, -1> v192 = pto::Stride<-1, -1, -1, -1, -1>(v187 * v190, v190, v189, v16, v183);
  // pto: %xn_scale_buf_inline172__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v193 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v185 + (v184 + v78 * v16 + v17 * v183), v191, v192);
  TSTORE(v193, v172);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}