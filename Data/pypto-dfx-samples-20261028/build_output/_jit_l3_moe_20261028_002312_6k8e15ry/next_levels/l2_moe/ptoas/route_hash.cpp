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

AICORE void route_hash(__gm__ int64_t* v1, __gm__ int32_t* v2, __gm__ float* v3, __gm__ int32_t* v4, __gm__ float* v5, int64_t v6, int64_t v7, int64_t v8, int32_t v9, int32_t v10) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  const int64_t v11 = 32;
  const int64_t v12 = 32;
  const int64_t v13 = 1024;
  const int64_t v14 = 1024;
  // pto: %c256_i64
  const int64_t v15 = 256;
  // pto: %c512_i64
  const int64_t v16 = 512;
  // pto: %c8704_i64
  const int64_t v17 = 8704;
  // pto: %c16896_i64
  const int64_t v18 = 16896;
  // pto: %c0_i64
  const int64_t v19 = 0;
  // pto: %c1_index
  const int64_t v20 = 1;
  // pto: %c6_index
  const int64_t v21 = 6;
  // pto: %c256_index
  const int64_t v22 = 256;
  // pto: %c8_index
  const int64_t v23 = 8;
  // pto: %c0_index
  const int64_t v24 = 0;
  // pto: %c0_i32
  const int32_t v25 = 0;
  // pto: %c512_index
  const int64_t v26 = 512;
  // pto: %c128_index
  const int64_t v27 = 128;
  // pto: %cst_15
  const float v28 = 1.5f;
  // pto: %route_scores_buf_inline173__ssa_v1_view
  const int64_t v29 = 1;
  // pto: %route_scores_buf_inline173__ssa_v1_view
  const int64_t v30 = 1;
  // pto: %route_scores_buf_inline173__ssa_v1_view
  const int64_t v31 = 1;
  // pto: %route_scores_buf_inline173__ssa_v1_view
  int64_t v32 = (int64_t) v8;
  // pto: %route_scores_buf_inline173__ssa_v1_view
  int64_t v33 = v32 * v22;
  // pto: %route_scores_buf_inline173__ssa_v1_view
  int64_t v34 = v31 * v33;
  // pto: %route_scores_buf_inline173__ssa_v1_view
  pto::Shape<1, 1, 1, -1, -1> v35 = pto::Shape<1, 1, 1, -1, -1>(v29, v30, v31, v32, v22);
  // pto: %route_scores_buf_inline173__ssa_v1_view
  pto::Stride<-1, -1, -1, -1, -1> v36 = pto::Stride<-1, -1, -1, -1, -1>(v30 * v34, v34, v33, v22, v20);
  // pto: %route_scores_buf_inline173__ssa_v1_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v37 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v35, v36);
  // pto: %th_idx_inline243__ssa_v0, %5
  int64_t v38 = (int64_t) ((uint64_t) ((int64_t) v9) * (uint64_t) v23);
  // pto: %hs_idx_tile_inline244__tile
  Tile<TileType::Vec, int32_t, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v39 = Tile<TileType::Vec, int32_t, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v23);
  // pto: %hs_idx_tile_inline244__tile
  uint64_t v40 = (uint64_t) v15;
  TASSIGN(v39, v40);
  for (int64_t v41 = v24; v41 < v23; v41 += v20) {
    // pto: %7, %6
    ;
    int64_t v42 = (v1)[(int64_t) ((uint64_t) v38 + (uint64_t) v41)];
    for (int64_t v43 = v24; v43 < v21; v43 += v20) {
      // pto: %flat_offset_mul
      ;
      int64_t v44 = (int64_t) ((uint64_t) v42 * (uint64_t) v21);
      // pto: %flat_offset, %9
      ;
      int32_t v45 = (v2)[(int64_t) ((uint64_t) v44 + (uint64_t) v43)];
      // pto: %10
      ;
      int64_t v46 = (int64_t) ((uint64_t) v41 * (uint64_t) v23);
      // pto: %11
      ;
      int64_t v47 = (int64_t) ((uint64_t) v46 + (uint64_t) v43);
      v39.SetValue(v47, v45);
    };
    for (int64_t v48 = v21; v48 < v23; v48 += v20) {
      // pto: %12
      ;
      int64_t v49 = (int64_t) ((uint64_t) v41 * (uint64_t) v23);
      // pto: %13
      ;
      int64_t v50 = (int64_t) ((uint64_t) v49 + (uint64_t) v48);
      v39.SetValue(v50, v25);
    };
  }
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  set_flag(PIPE_S, PIPE_V, EVENT_ID1);
  // pto: %local_scores_inline249__tile
  Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v51 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v22);
  // pto: %local_scores_inline249__tile
  uint64_t v52 = (uint64_t) v16;
  TASSIGN(v51, v52);
  // pto: %t__tile
  Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v53 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v22);
  // pto: %t__tile
  uint64_t v54 = (uint64_t) v17;
  TASSIGN(v53, v54);
  // pto: %14
  int64_t v55 = v38 < v24 ? v24 : v38;
  // pto: %route_scores_buf_inline173__ssa_v1_pview
  __gm__ float* v56 = PTOAS__GLOBAL_TENSOR_DATA(v37);
  // pto: %route_scores_buf_inline173__ssa_v1_pview
  const int64_t v57 = 0;
  // pto: %route_scores_buf_inline173__ssa_v1_pview
  const int64_t v58 = 256;
  // pto: %route_scores_buf_inline173__ssa_v1_pview
  pto::Shape<1, 1, 1, 8, 256> v59 = pto::Shape<1, 1, 1, 8, 256>();
  // pto: %route_scores_buf_inline173__ssa_v1_pview
  pto::Stride<2048, 2048, 2048, 256, 1> v60 = pto::Stride<2048, 2048, 2048, 256, 1>();
  // pto: %route_scores_buf_inline173__ssa_v1_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<2048, 2048, 2048, 256, 1>, pto::Layout::ND> v61 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<2048, 2048, 2048, 256, 1>, pto::Layout::ND>(v56 + (v57 + v55 * v58), v59, v60);
  TLOAD(v53, v61);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  // pto: %assemble_view
  Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, 8, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v62;
  // pto: %assemble_view
  uint64_t v63 = (uint64_t) v16;
  TASSIGN(v62, v63);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TMOV(v62, v53);
  // pto: %gather_acc_init
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v64 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v23);
  // pto: %gather_acc_init
  uint64_t v65 = (uint64_t) v17;
  TASSIGN(v64, v65);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
  for (int64_t v66 = v24; v66 < v23; v66 += v20) {
    int64_t v67 = (int64_t) ((uint64_t) v66 * (uint64_t) v14);
    // pto: %17
    ;
    int64_t v68 = (int64_t) ((uint64_t) v67 + (uint64_t) v26);
    // pto: %gather_inp_row
    ;
    Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v69 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v20, v22);
    // pto: %gather_inp_row
    ;
    uint64_t v70 = (uint64_t) v68;
    TASSIGN(v69, v70);
    int64_t v71 = (int64_t) ((uint64_t) v66 * (uint64_t) v13);
    // pto: %slice_view
    ;
    int64_t v72 = (int64_t) ((uint64_t) v71 + (uint64_t) v16);
    // pto: %slice_view
    ;
    Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v73;
    // pto: %slice_view
    ;
    uint64_t v74 = (uint64_t) v72;
    TASSIGN(v73, v74);
    int64_t v75 = (int64_t) ((uint64_t) v66 * (uint64_t) v12);
    // pto: %21
    ;
    int64_t v76 = (int64_t) ((uint64_t) v75 + (uint64_t) v22);
    // pto: %gather_idx_row
    ;
    Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v77 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v20, v23);
    // pto: %gather_idx_row
    ;
    uint64_t v78 = (uint64_t) v76;
    TASSIGN(v77, v78);
    // pto: %23
    ;
    int64_t v79 = (int64_t) ((uint64_t) v66 * (uint64_t) v11);
    // pto: %23
    ;
    int64_t v80 = (int64_t) ((uint64_t) v79 + (uint64_t) v15);
    // pto: %23
    ;
    Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v81;
    // pto: %23
    ;
    uint64_t v82 = (uint64_t) v80;
    TASSIGN(v81, v82);
    // pto: %gather_row_tmp
    ;
    Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v83 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v20, v23);
    // pto: %gather_row_tmp
    ;
    uint64_t v84 = (uint64_t) v18;
    TASSIGN(v83, v84);
    // pto: %gather_row
    ;
    Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v85 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v20, v23);
    // pto: %gather_row
    ;
    uint64_t v86 = (uint64_t) v19;
    TASSIGN(v85, v86);
    pipe_barrier(PIPE_V);
    TGATHER(v85, v73, v81, v83);
    // pto: %24
    ;
    int64_t v87 = (int64_t) ((uint64_t) v79 + (uint64_t) v17);
    // pto: %24
    ;
    Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v88;
    // pto: %24
    ;
    uint64_t v89 = (uint64_t) v87;
    TASSIGN(v88, v89);
    pipe_barrier(PIPE_V);
    TMOV(v88, v85);
  }
  // pto: %gather_all_inline250__tile
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v90 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v23);
  // pto: %gather_all_inline250__tile
  uint64_t v91 = (uint64_t) v17;
  TASSIGN(v90, v91);
  v90.SetValidShape(v23, v21);
  // pto: %hs_vals_pad_inline252__tile
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v92 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v23, v23);
  // pto: %hs_vals_pad_inline252__tile
  uint64_t v93 = (uint64_t) v17;
  TASSIGN(v92, v93);
  pipe_barrier(PIPE_V);
  TFILLPAD<pto::TFillPadMode::InPlace>(v92, v90);
  // pto: %hs_idx_read_inline253__tile
  Tile<TileType::Vec, int32_t, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v94 = Tile<TileType::Vec, int32_t, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v23);
  // pto: %hs_idx_read_inline253__tile
  uint64_t v95 = (uint64_t) v19;
  TASSIGN(v94, v95);
  // pto: %1
  Tile<TileType::Vec, int32_t, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v96 = Tile<TileType::Vec, int32_t, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v23);
  // pto: %1
  uint64_t v97 = (uint64_t) v15;
  TASSIGN(v96, v97);
  // pto: %25
  Tile<TileType::Vec, int32_t, 8, 8, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v98;
  // pto: %25
  uint64_t v99 = (uint64_t) v15;
  TASSIGN(v98, v99);
  // pto: %26
  Tile<TileType::Vec, int32_t, 8, 8, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v100;
  // pto: %26
  uint64_t v101 = (uint64_t) v19;
  TASSIGN(v100, v101);
  TMOV(v100, v98);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  // pto: %tmp_tile
  Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v102 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v27);
  // pto: %tmp_tile
  uint64_t v103 = (uint64_t) v16;
  TASSIGN(v102, v103);
  // pto: %3
  Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v104 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v20);
  // pto: %3
  uint64_t v105 = (uint64_t) v15;
  TASSIGN(v104, v105);
  pipe_barrier(PIPE_V);
  TROWSUM(v104, v92, v102);
  // pto: %hs_denom_inline254__tile
  Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v106 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v20);
  // pto: %hs_denom_inline254__tile
  uint64_t v107 = (uint64_t) v15;
  TASSIGN(v106, v107);
  // pto: %4
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v108 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v23, v23);
  // pto: %4
  uint64_t v109 = (uint64_t) v16;
  TASSIGN(v108, v109);
  pipe_barrier(PIPE_V);
  TROWEXPANDDIV(v108, v92, v106);
  // pto: %hs_weights_pad_inline255__tile
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v110 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v23, v23);
  // pto: %hs_weights_pad_inline255__tile
  uint64_t v111 = (uint64_t) v16;
  TASSIGN(v110, v111);
  pipe_barrier(PIPE_V);
  TMULS(v110, v108, v28);
  set_flag(PIPE_V, PIPE_S, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID1);
  for (int64_t v112 = v24; v112 < v23; v112 += v20) {
    // pto: %27
    ;
    int64_t v113 = (int64_t) ((uint64_t) v38 + (uint64_t) v112);
    // pto: %28
    ;
    if (v113 < v6) {
      for (int64_t v114 = v24; v114 < v21; v114 += v20) {
        // pto: %30
        ;
        int64_t v115 = (int64_t) ((uint64_t) v112 * (uint64_t) v23);
        // pto: %31
        ;
        int64_t v116 = (int64_t) ((uint64_t) v115 + (uint64_t) v114);
        // pto: %29
        ;
        int32_t v117 = v94.GetValue(v116);
        // pto: %33
        ;
        int64_t v118 = (int64_t) ((uint64_t) v113 * (uint64_t) v21);
        // pto: %34
        ;
        int64_t v119 = (int64_t) ((uint64_t) v118 + (uint64_t) v114);
        (v4)[v119] = v117;
        // pto: %35
        ;
        float v120 = v110.GetValue(v116);
        (v5)[v119] = v120;
      };
    };
  }
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