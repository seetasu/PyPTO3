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

AICORE void physical_sparse_build_bias(__gm__ int32_t* v1, __gm__ float* v2, __gm__ int32_t* v3, int64_t v4, int64_t v5, int64_t v6, int32_t v7, int32_t v8) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v9 = 0;
  // pto: %c128_index
  const int64_t v10 = 128;
  // pto: %c1_index
  const int64_t v11 = 1;
  // pto: %c640_index
  const int64_t v12 = 640;
  // pto: %c512_index
  const int64_t v13 = 512;
  // pto: %c16_index
  const int64_t v14 = 16;
  // pto: %c0_index
  const int64_t v15 = 0;
  // pto: %cst_7
  const float v16 = 1.0f;
  // pto: %cst_8
  const float v17 = 0.0f;
  // pto: %cst_9
  const float v18 = 3.40282347E+38f;
  // pto: %swa_indices__ssa_v0_view
  const int64_t v19 = 1;
  // pto: %swa_indices__ssa_v0_view
  const int64_t v20 = 1;
  // pto: %swa_indices__ssa_v0_view
  const int64_t v21 = 1;
  // pto: %swa_indices__ssa_v0_view
  int64_t v22 = (int64_t) v5;
  // pto: %swa_indices__ssa_v0_view
  int64_t v23 = v22 * v10;
  // pto: %swa_indices__ssa_v0_view
  int64_t v24 = v21 * v23;
  // pto: %swa_indices__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v25 = pto::Shape<1, 1, 1, -1, -1>(v19, v20, v21, v22, v10);
  // pto: %swa_indices__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v26 = pto::Stride<-1, -1, -1, -1, -1>(v20 * v24, v24, v23, v10, v11);
  // pto: %swa_indices__ssa_v0_view
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v27 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v25, v26);
  // pto: %sparse_bias_inline385__ssa_v0_view
  const int64_t v28 = 1;
  // pto: %sparse_bias_inline385__ssa_v0_view
  const int64_t v29 = 1;
  // pto: %sparse_bias_inline385__ssa_v0_view
  const int64_t v30 = 1;
  // pto: %sparse_bias_inline385__ssa_v0_view
  int64_t v31 = (int64_t) v6;
  // pto: %sparse_bias_inline385__ssa_v0_view
  int64_t v32 = v31 * v12;
  // pto: %sparse_bias_inline385__ssa_v0_view
  int64_t v33 = v30 * v32;
  // pto: %sparse_bias_inline385__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v34 = pto::Shape<1, 1, 1, -1, -1>(v28, v29, v30, v31, v12);
  // pto: %sparse_bias_inline385__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v35 = pto::Stride<-1, -1, -1, -1, -1>(v29 * v33, v33, v32, v12, v11);
  // pto: %sparse_bias_inline385__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v36 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v34, v35);
  // pto: %cmp_indices__ssa_v0_view
  const int64_t v37 = 1;
  // pto: %cmp_indices__ssa_v0_view
  const int64_t v38 = 1;
  // pto: %cmp_indices__ssa_v0_view
  const int64_t v39 = 1;
  // pto: %cmp_indices__ssa_v0_view
  int64_t v40 = (int64_t) v5;
  // pto: %cmp_indices__ssa_v0_view
  int64_t v41 = v40 * v13;
  // pto: %cmp_indices__ssa_v0_view
  int64_t v42 = v39 * v41;
  // pto: %cmp_indices__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v43 = pto::Shape<1, 1, 1, -1, -1>(v37, v38, v39, v40, v13);
  // pto: %cmp_indices__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v44 = pto::Stride<-1, -1, -1, -1, -1>(v38 * v42, v42, v41, v13, v11);
  // pto: %cmp_indices__ssa_v0_view
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v45 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v43, v44);
  // pto: %bias_block_inline542__ssa_v0, %5
  int64_t v46 = (int64_t) ((uint64_t) ((int64_t) v7) * (uint64_t) v14);
  // pto: %6
  if (v46 < v4) {
    // pto: %7
    ;
    int64_t v47 = (int64_t) ((uint64_t) v4 - (uint64_t) v46);
    // pto: %8
    ;
    int64_t v48 = v47 < v14 ? v47 : v14;
    // pto: %raw_rows_inline546__tile
    ;
    Tile<TileType::Vec, int32_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v49 = Tile<TileType::Vec, int32_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48, v10);
    // pto: %raw_rows_inline546__tile
    ;
    uint64_t v50 = (uint64_t) v9;
    TASSIGN(v49, v50);
    // pto: %9
    ;
    int64_t v51 = v46 < v15 ? v15 : v46;
    // pto: %swa_indices__ssa_v0_pview
    ;
    const int64_t v52 = 0;
    // pto: %swa_indices__ssa_v0_pview
    ;
    __gm__ int32_t* v53 = PTOAS__GLOBAL_TENSOR_DATA(v27);
    // pto: %swa_indices__ssa_v0_pview
    ;
    const int64_t v54 = 1;
    // pto: %swa_indices__ssa_v0_pview
    ;
    const int64_t v55 = 1;
    // pto: %swa_indices__ssa_v0_pview
    ;
    const int64_t v56 = 1;
    // pto: %swa_indices__ssa_v0_pview
    ;
    int64_t v57 = v48 * v10;
    // pto: %swa_indices__ssa_v0_pview
    ;
    int64_t v58 = v56 * v57;
    // pto: %swa_indices__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, -1, 128> v59 = pto::Shape<1, 1, 1, -1, 128>(v54, v55, v56, v48, v10);
    // pto: %swa_indices__ssa_v0_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v60 = pto::Stride<-1, -1, -1, -1, -1>(v55 * v58, v58, v57, v10, v11);
    // pto: %swa_indices__ssa_v0_pview
    ;
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, 128>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v61 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, 128>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v53 + (v52 + v51 * v10 + v15 * v11), v59, v60);
    TLOAD(v49, v61);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %raw_index_inline547__tile
    ;
    Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v62 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48, v10);
    // pto: %raw_index_inline547__tile
    ;
    uint64_t v63 = (uint64_t) v9;
    TASSIGN(v62, v63);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    RoundMode v64 = RoundMode::CAST_ROUND;
    SaturationMode v65 = SaturationMode::OFF;
    TCVT(v62, v49, v64, v65);
    // pto: %t__tile
    ;
    Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v66 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48, v10);
    // pto: %t__tile
    ;
    uint64_t v67 = (uint64_t) v9;
    TASSIGN(v66, v67);
    pipe_barrier(PIPE_V);
    TADDS(v66, v62, v16);
    // pto: %0
    ;
    Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v68 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48, v10);
    // pto: %0
    ;
    uint64_t v69 = (uint64_t) v9;
    TASSIGN(v68, v69);
    pipe_barrier(PIPE_V);
    TMAXS(v68, v66, v17);
    // pto: %raw_flag_inline548__tile
    ;
    Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v70 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48, v10);
    // pto: %raw_flag_inline548__tile
    ;
    uint64_t v71 = (uint64_t) v9;
    TASSIGN(v70, v71);
    pipe_barrier(PIPE_V);
    TMINS(v70, v68, v16);
    // pto: %1
    ;
    Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v72 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48, v10);
    // pto: %1
    ;
    uint64_t v73 = (uint64_t) v9;
    TASSIGN(v72, v73);
    pipe_barrier(PIPE_V);
    TSUBS(v72, v70, v16);
    // pto: %raw_bias_inline549__tile
    ;
    Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v74 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48, v10);
    // pto: %raw_bias_inline549__tile
    ;
    uint64_t v75 = (uint64_t) v9;
    TASSIGN(v74, v75);
    pipe_barrier(PIPE_V);
    TMULS(v74, v72, v18);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %raw_bias_valid_inline550__tile
    ;
    Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v76 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48, v10);
    // pto: %raw_bias_valid_inline550__tile
    ;
    uint64_t v77 = (uint64_t) v9;
    TASSIGN(v76, v77);
    // pto: %slice_view
    ;
    Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v78 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48);
    // pto: %slice_view
    ;
    uint64_t v79 = (uint64_t) v9;
    TASSIGN(v78, v79);
    // pto: %sparse_bias_inline385__ssa_v0_pview
    ;
    const int64_t v80 = 0;
    // pto: %sparse_bias_inline385__ssa_v0_pview
    ;
    __gm__ float* v81 = PTOAS__GLOBAL_TENSOR_DATA(v36);
    // pto: %sparse_bias_inline385__ssa_v0_pview
    ;
    const int64_t v82 = 1;
    // pto: %sparse_bias_inline385__ssa_v0_pview
    ;
    const int64_t v83 = 1;
    // pto: %sparse_bias_inline385__ssa_v0_pview
    ;
    const int64_t v84 = 1;
    // pto: %sparse_bias_inline385__ssa_v0_pview
    ;
    int64_t v85 = v48 * v12;
    // pto: %sparse_bias_inline385__ssa_v0_pview
    ;
    int64_t v86 = v84 * v85;
    // pto: %sparse_bias_inline385__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, -1, 128> v87 = pto::Shape<1, 1, 1, -1, 128>(v82, v83, v84, v48, v10);
    // pto: %sparse_bias_inline385__ssa_v0_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v88 = pto::Stride<-1, -1, -1, -1, -1>(v83 * v86, v86, v85, v12, v11);
    // pto: %sparse_bias_inline385__ssa_v0_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, -1, 128>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v89 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 128>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v81 + (v80 + v51 * v12 + v15 * v11), v87, v88);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v89, v78);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    // pto: %cmp_rows_inline551__tile
    ;
    Tile<TileType::Vec, int32_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v90 = Tile<TileType::Vec, int32_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48, v13);
    // pto: %cmp_rows_inline551__tile
    ;
    uint64_t v91 = (uint64_t) v9;
    TASSIGN(v90, v91);
    // pto: %cmp_indices__ssa_v0_pview
    ;
    const int64_t v92 = 0;
    // pto: %cmp_indices__ssa_v0_pview
    ;
    __gm__ int32_t* v93 = PTOAS__GLOBAL_TENSOR_DATA(v45);
    // pto: %cmp_indices__ssa_v0_pview
    ;
    const int64_t v94 = 1;
    // pto: %cmp_indices__ssa_v0_pview
    ;
    const int64_t v95 = 1;
    // pto: %cmp_indices__ssa_v0_pview
    ;
    const int64_t v96 = 1;
    // pto: %cmp_indices__ssa_v0_pview
    ;
    int64_t v97 = v48 * v13;
    // pto: %cmp_indices__ssa_v0_pview
    ;
    int64_t v98 = v96 * v97;
    // pto: %cmp_indices__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, -1, 512> v99 = pto::Shape<1, 1, 1, -1, 512>(v94, v95, v96, v48, v13);
    // pto: %cmp_indices__ssa_v0_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v100 = pto::Stride<-1, -1, -1, -1, -1>(v95 * v98, v98, v97, v13, v11);
    // pto: %cmp_indices__ssa_v0_pview
    ;
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v101 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v93 + (v92 + v51 * v13 + v15 * v11), v99, v100);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v90, v101);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    // pto: %cmp_index_inline552__tile
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v102 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48, v13);
    // pto: %cmp_index_inline552__tile
    ;
    uint64_t v103 = (uint64_t) v9;
    TASSIGN(v102, v103);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    RoundMode v104 = RoundMode::CAST_ROUND;
    SaturationMode v105 = SaturationMode::OFF;
    TCVT(v102, v90, v104, v105);
    // pto: %2
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v106 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48, v13);
    // pto: %2
    ;
    uint64_t v107 = (uint64_t) v9;
    TASSIGN(v106, v107);
    pipe_barrier(PIPE_V);
    TADDS(v106, v102, v16);
    // pto: %3
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v108 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48, v13);
    // pto: %3
    ;
    uint64_t v109 = (uint64_t) v9;
    TASSIGN(v108, v109);
    pipe_barrier(PIPE_V);
    TMAXS(v108, v106, v17);
    // pto: %cmp_flag_inline553__tile
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v110 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48, v13);
    // pto: %cmp_flag_inline553__tile
    ;
    uint64_t v111 = (uint64_t) v9;
    TASSIGN(v110, v111);
    pipe_barrier(PIPE_V);
    TMINS(v110, v108, v16);
    // pto: %4
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v112 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48, v13);
    // pto: %4
    ;
    uint64_t v113 = (uint64_t) v9;
    TASSIGN(v112, v113);
    pipe_barrier(PIPE_V);
    TSUBS(v112, v110, v16);
    // pto: %cmp_bias_inline554__tile
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v114 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48, v13);
    // pto: %cmp_bias_inline554__tile
    ;
    uint64_t v115 = (uint64_t) v9;
    TASSIGN(v114, v115);
    pipe_barrier(PIPE_V);
    TMULS(v114, v112, v18);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    // pto: %cmp_bias_valid_inline555__tile
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v116 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48, v13);
    // pto: %cmp_bias_valid_inline555__tile
    ;
    uint64_t v117 = (uint64_t) v9;
    TASSIGN(v116, v117);
    // pto: %12
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v118 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48);
    // pto: %12
    ;
    uint64_t v119 = (uint64_t) v9;
    TASSIGN(v118, v119);
    // pto: %14
    ;
    const int64_t v120 = 0;
    // pto: %14
    ;
    __gm__ float* v121 = PTOAS__GLOBAL_TENSOR_DATA(v36);
    // pto: %14
    ;
    const int64_t v122 = 1;
    // pto: %14
    ;
    const int64_t v123 = 1;
    // pto: %14
    ;
    const int64_t v124 = 1;
    // pto: %14
    ;
    int64_t v125 = v48 * v12;
    // pto: %14
    ;
    int64_t v126 = v124 * v125;
    // pto: %14
    ;
    pto::Shape<1, 1, 1, -1, 512> v127 = pto::Shape<1, 1, 1, -1, 512>(v122, v123, v124, v48, v13);
    // pto: %14
    ;
    pto::Stride<-1, -1, -1, -1, -1> v128 = pto::Stride<-1, -1, -1, -1, -1>(v123 * v126, v126, v125, v12, v11);
    // pto: %14
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v129 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v121 + (v120 + v51 * v12 + v10 * v11), v127, v128);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TSTORE(v129, v118);
  }
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}