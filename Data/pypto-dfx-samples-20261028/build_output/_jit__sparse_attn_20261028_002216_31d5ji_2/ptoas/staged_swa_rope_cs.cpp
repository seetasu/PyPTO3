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

AICORE void staged_swa_rope_cs(__gm__ int32_t* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, __gm__ float* v4, __gm__ float* v5, int64_t v6, int64_t v7, int64_t v8, int32_t v9, int32_t v10) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  const int64_t v11 = 128;
  const int64_t v12 = 64;
  const int64_t v13 = 128;
  const int64_t v14 = 4;
  // pto: %c15616_i64
  const int64_t v15 = 15616;
  // pto: %c0_i64
  const int64_t v16 = 0;
  // pto: %c4096_i64
  const int64_t v17 = 4096;
  // pto: %c8192_i64
  const int64_t v18 = 8192;
  // pto: %c11392_i64
  const int64_t v19 = 11392;
  // pto: %c9216_i64
  const int64_t v20 = 9216;
  // pto: %c11264_i64
  const int64_t v21 = 11264;
  // pto: %c15488_i64
  const int64_t v22 = 15488;
  // pto: %c16_index
  const int64_t v23 = 16;
  const int64_t v24 = 64;
  // pto: %c1_index
  const int64_t v25 = 1;
  // pto: %c2_index
  const int64_t v26 = 2;
  // pto: %cst_12
  const float v27 = 1.0f;
  // pto: %c192_index
  const int64_t v28 = 192;
  // pto: %c0_i32
  const int32_t v29 = 0;
  // pto: %cst_15
  const float v30 = 0.5f;
  // pto: %cst_16
  const float v31 = 2.0f;
  // pto: %c0_index
  const int64_t v32 = 0;
  const int64_t v33 = 32;
  // pto: %c15616_index
  const int64_t v34 = 15616;
  // pto: %c11392_index
  const int64_t v35 = 11392;
  // pto: %c4096_index
  const int64_t v36 = 4096;
  // pto: %c9216_index
  const int64_t v37 = 9216;
  // pto: %rope_swap_idx_inline391__ssa_v0_view
  const int64_t v38 = 1;
  // pto: %rope_swap_idx_inline391__ssa_v0_view
  const int64_t v39 = 1;
  // pto: %rope_swap_idx_inline391__ssa_v0_view
  const int64_t v40 = 1;
  // pto: %rope_swap_idx_inline391__ssa_v0_view
  int64_t v41 = v23 * v24;
  // pto: %rope_swap_idx_inline391__ssa_v0_view
  int64_t v42 = v40 * v41;
  // pto: %rope_swap_idx_inline391__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v43 = pto::Shape<1, 1, 1, -1, -1>(v38, v39, v40, v23, v24);
  // pto: %rope_swap_idx_inline391__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v44 = pto::Stride<-1, -1, -1, -1, -1>(v39 * v42, v42, v41, v24, v25);
  // pto: %rope_swap_idx_inline391__ssa_v0_view
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v45 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v43, v44);
  // pto: %freqs_cos__ssa_v0_view
  const int64_t v46 = 1;
  // pto: %freqs_cos__ssa_v0_view
  const int64_t v47 = 1;
  // pto: %freqs_cos__ssa_v0_view
  const int64_t v48 = 1;
  // pto: %freqs_cos__ssa_v0_view
  int64_t v49 = (int64_t) v7;
  // pto: %freqs_cos__ssa_v0_view
  int64_t v50 = v49 * v24;
  // pto: %freqs_cos__ssa_v0_view
  int64_t v51 = v48 * v50;
  // pto: %freqs_cos__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v52 = pto::Shape<1, 1, 1, -1, -1>(v46, v47, v48, v49, v24);
  // pto: %freqs_cos__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v53 = pto::Stride<-1, -1, -1, -1, -1>(v47 * v51, v51, v50, v24, v25);
  // pto: %freqs_cos__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v54 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v52, v53);
  // pto: %freqs_sin__ssa_v0_view
  const int64_t v55 = 1;
  // pto: %freqs_sin__ssa_v0_view
  const int64_t v56 = 1;
  // pto: %freqs_sin__ssa_v0_view
  const int64_t v57 = 1;
  // pto: %freqs_sin__ssa_v0_view
  int64_t v58 = (int64_t) v7;
  // pto: %freqs_sin__ssa_v0_view
  int64_t v59 = v58 * v24;
  // pto: %freqs_sin__ssa_v0_view
  int64_t v60 = v57 * v59;
  // pto: %freqs_sin__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v61 = pto::Shape<1, 1, 1, -1, -1>(v55, v56, v57, v58, v24);
  // pto: %freqs_sin__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v62 = pto::Stride<-1, -1, -1, -1, -1>(v56 * v60, v60, v59, v24, v25);
  // pto: %freqs_sin__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v63 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v61, v62);
  // pto: %rope_cos_il_view_inline397__ssa_v0_view
  const int64_t v64 = 1;
  // pto: %rope_cos_il_view_inline397__ssa_v0_view
  const int64_t v65 = 1;
  // pto: %rope_cos_il_view_inline397__ssa_v0_view
  const int64_t v66 = 1;
  // pto: %rope_cos_il_view_inline397__ssa_v0_view
  int64_t v67 = (int64_t) v8;
  // pto: %rope_cos_il_view_inline397__ssa_v0_view
  int64_t v68 = v67 * v24;
  // pto: %rope_cos_il_view_inline397__ssa_v0_view
  int64_t v69 = v66 * v68;
  // pto: %rope_cos_il_view_inline397__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v70 = pto::Shape<1, 1, 1, -1, -1>(v64, v65, v66, v67, v24);
  // pto: %rope_cos_il_view_inline397__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v71 = pto::Stride<-1, -1, -1, -1, -1>(v65 * v69, v69, v68, v24, v25);
  // pto: %rope_cos_il_view_inline397__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v72 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v70, v71);
  // pto: %rope_sin_signed_view_inline398__ssa_v0_view
  const int64_t v73 = 1;
  // pto: %rope_sin_signed_view_inline398__ssa_v0_view
  const int64_t v74 = 1;
  // pto: %rope_sin_signed_view_inline398__ssa_v0_view
  const int64_t v75 = 1;
  // pto: %rope_sin_signed_view_inline398__ssa_v0_view
  int64_t v76 = (int64_t) v8;
  // pto: %rope_sin_signed_view_inline398__ssa_v0_view
  int64_t v77 = v76 * v24;
  // pto: %rope_sin_signed_view_inline398__ssa_v0_view
  int64_t v78 = v75 * v77;
  // pto: %rope_sin_signed_view_inline398__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v79 = pto::Shape<1, 1, 1, -1, -1>(v73, v74, v75, v76, v24);
  // pto: %rope_sin_signed_view_inline398__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v80 = pto::Stride<-1, -1, -1, -1, -1>(v74 * v78, v78, v77, v24, v25);
  // pto: %rope_sin_signed_view_inline398__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v81 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v5, v79, v80);
  // pto: %cs_block_inline400__ssa_v0
  int64_t v82 = (int64_t) v9;
  // pto: %15
  int64_t v83 = v82 % v26;
  // pto: %16
  int64_t v84 = v82 / v26;
  // pto: %17
  int64_t v85 = (int64_t) ((uint64_t) v83 * (uint64_t) v23);
  int64_t v86 = (int64_t) ((uint64_t) v83 * (uint64_t) v33);
  // pto: %swap_ones_inline405__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v87 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v24);
  // pto: %swap_ones_inline405__tile
  uint64_t v88 = (uint64_t) v15;
  TASSIGN(v87, v88);
  TEXPANDS(v87, v27);
  // pto: %t__ci_tmp_v0
  Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v89 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v28);
  // pto: %t__ci_tmp_v0
  uint64_t v90 = (uint64_t) v16;
  TASSIGN(v89, v90);
  // pto: %t__tile
  Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v91 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v24);
  // pto: %t__tile
  uint64_t v92 = (uint64_t) v17;
  TASSIGN(v91, v92);
  // pto: %ci_tmp_view
  Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v93;
  TRESHAPE(v93, v89);
  // pto: %ci_dst_view
  Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v94;
  TRESHAPE(v94, v91);
  TCI<Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v94, v29, v93);
  // pto: %swap_ramp_inline406__tile
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v95 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v24);
  // pto: %swap_ramp_inline406__tile
  uint64_t v96 = (uint64_t) v16;
  TASSIGN(v95, v96);
  pipe_barrier(PIPE_V);
  RoundMode v97 = RoundMode::CAST_ROUND;
  SaturationMode v98 = SaturationMode::OFF;
  TCVT(v95, v91, v97, v98);
  // pto: %swap_col_inline407__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v99 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v24);
  // pto: %swap_col_inline407__tile
  uint64_t v100 = (uint64_t) v15;
  TASSIGN(v99, v100);
  pipe_barrier(PIPE_V);
  TCOLEXPANDMUL(v99, v87, v95);
  // pto: %0
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v101 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v24);
  // pto: %0
  uint64_t v102 = (uint64_t) v16;
  TASSIGN(v101, v102);
  pipe_barrier(PIPE_V);
  TMULS(v101, v99, v30);
  // pto: %swap_dup_i32_inline408__tile
  Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v103 = Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v24);
  // pto: %swap_dup_i32_inline408__tile
  uint64_t v104 = (uint64_t) v16;
  TASSIGN(v103, v104);
  pipe_barrier(PIPE_V);
  RoundMode v105 = RoundMode::CAST_TRUNC;
  SaturationMode v106 = SaturationMode::ON;
  TCVT(v103, v101, v105, v106);
  // pto: %swap_dup_f_inline409__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v107 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v24);
  // pto: %swap_dup_f_inline409__tile
  uint64_t v108 = (uint64_t) v16;
  TASSIGN(v107, v108);
  pipe_barrier(PIPE_V);
  RoundMode v109 = RoundMode::CAST_ROUND;
  SaturationMode v110 = SaturationMode::OFF;
  TCVT(v107, v103, v109, v110);
  // pto: %1
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v111 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v24);
  // pto: %1
  uint64_t v112 = (uint64_t) v16;
  TASSIGN(v111, v112);
  pipe_barrier(PIPE_V);
  TMULS(v111, v107, v31);
  // pto: %swap_lane_inline410__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v113 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v24);
  // pto: %swap_lane_inline410__tile
  uint64_t v114 = (uint64_t) v16;
  TASSIGN(v113, v114);
  pipe_barrier(PIPE_V);
  TSUB(v113, v99, v111);
  // pto: %2
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v115 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v24);
  // pto: %2
  uint64_t v116 = (uint64_t) v15;
  TASSIGN(v115, v116);
  pipe_barrier(PIPE_V);
  TADDS(v115, v99, v27);
  // pto: %3
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v117 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v24);
  // pto: %3
  uint64_t v118 = (uint64_t) v16;
  TASSIGN(v117, v118);
  TMULS(v117, v113, v31);
  // pto: %swap_pair_f_inline411__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v119 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v24);
  // pto: %swap_pair_f_inline411__tile
  uint64_t v120 = (uint64_t) v15;
  TASSIGN(v119, v120);
  pipe_barrier(PIPE_V);
  TSUB(v119, v115, v117);
  // pto: %swap_idx_inline412__tile
  Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v121 = Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v24);
  // pto: %swap_idx_inline412__tile
  uint64_t v122 = (uint64_t) v15;
  TASSIGN(v121, v122);
  pipe_barrier(PIPE_V);
  RoundMode v123 = RoundMode::CAST_ROUND;
  SaturationMode v124 = SaturationMode::ON;
  TCVT(v121, v119, v123, v124);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  // pto: %19
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  if (v84 == v32) {
    int64_t v125 = (int64_t) ((uint64_t) v83 * (uint64_t) v13);
    // pto: %21
    ;
    int64_t v126 = (int64_t) ((uint64_t) v125 + (uint64_t) v34);
    // pto: %4
    ;
    Tile<TileType::Vec, int32_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v127 = Tile<TileType::Vec, int32_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v33);
    // pto: %4
    ;
    uint64_t v128 = (uint64_t) v126;
    TASSIGN(v127, v128);
    // pto: %slice_view
    ;
    int64_t v129 = (int64_t) ((uint64_t) v86 * (uint64_t) v14);
    // pto: %slice_view
    ;
    int64_t v130 = (int64_t) ((uint64_t) v129 + (uint64_t) v15);
    // pto: %slice_view
    ;
    Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, 16, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v131;
    // pto: %slice_view
    ;
    uint64_t v132 = (uint64_t) v130;
    TASSIGN(v131, v132);
    // pto: %23
    ;
    int64_t v133 = v86 < v32 ? v32 : v86;
    // pto: %rope_swap_idx_inline391__ssa_v0_pview
    ;
    __gm__ int32_t* v134 = PTOAS__GLOBAL_TENSOR_DATA(v45);
    // pto: %rope_swap_idx_inline391__ssa_v0_pview
    ;
    const int64_t v135 = 0;
    // pto: %rope_swap_idx_inline391__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 16, 32> v136 = pto::Shape<1, 1, 1, 16, 32>();
    // pto: %rope_swap_idx_inline391__ssa_v0_pview
    ;
    pto::Stride<1024, 1024, 1024, 64, 1> v137 = pto::Stride<1024, 1024, 1024, 64, 1>();
    // pto: %rope_swap_idx_inline391__ssa_v0_pview
    ;
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, 16, 32>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND> v138 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 16, 32>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND>(v134 + (v135 + v133), v136, v137);
    TSTORE(v138, v131);
  }
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  // pto: %cs_ones_inline413__tile
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v139 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v33);
  // pto: %cs_ones_inline413__tile
  uint64_t v140 = (uint64_t) v15;
  TASSIGN(v139, v140);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  pipe_barrier(PIPE_V);
  TEXPANDS(v139, v27);
  // pto: %cs_ramp_i32_inline414__ci_tmp_v1
  Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v141 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v28);
  // pto: %cs_ramp_i32_inline414__ci_tmp_v1
  uint64_t v142 = (uint64_t) v16;
  TASSIGN(v141, v142);
  // pto: %cs_ramp_i32_inline414__tile
  Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v143 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v33);
  // pto: %cs_ramp_i32_inline414__tile
  uint64_t v144 = (uint64_t) v17;
  TASSIGN(v143, v144);
  // pto: %24
  Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v145;
  TRESHAPE(v145, v141);
  // pto: %25
  Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v146;
  TRESHAPE(v146, v143);
  TCI<Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v146, v29, v145);
  // pto: %cs_ramp_inline415__tile
  Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v147 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v33);
  // pto: %cs_ramp_inline415__tile
  uint64_t v148 = (uint64_t) v16;
  TASSIGN(v147, v148);
  pipe_barrier(PIPE_V);
  RoundMode v149 = RoundMode::CAST_ROUND;
  SaturationMode v150 = SaturationMode::OFF;
  TCVT(v147, v143, v149, v150);
  // pto: %cs_col_inline416__tile
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v151 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v33);
  // pto: %cs_col_inline416__tile
  uint64_t v152 = (uint64_t) v15;
  TASSIGN(v151, v152);
  pipe_barrier(PIPE_V);
  TCOLEXPANDMUL(v151, v139, v147);
  // pto: %5
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v153 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v33);
  // pto: %5
  uint64_t v154 = (uint64_t) v16;
  TASSIGN(v153, v154);
  pipe_barrier(PIPE_V);
  TMULS(v153, v151, v30);
  // pto: %cs_dup_i32_inline417__tile
  Tile<TileType::Vec, int32_t, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v155 = Tile<TileType::Vec, int32_t, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v33);
  // pto: %cs_dup_i32_inline417__tile
  uint64_t v156 = (uint64_t) v16;
  TASSIGN(v155, v156);
  pipe_barrier(PIPE_V);
  RoundMode v157 = RoundMode::CAST_TRUNC;
  SaturationMode v158 = SaturationMode::ON;
  TCVT(v155, v153, v157, v158);
  // pto: %cs_dup_f_inline418__tile
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v159 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v33);
  // pto: %cs_dup_f_inline418__tile
  uint64_t v160 = (uint64_t) v16;
  TASSIGN(v159, v160);
  pipe_barrier(PIPE_V);
  RoundMode v161 = RoundMode::CAST_ROUND;
  SaturationMode v162 = SaturationMode::OFF;
  TCVT(v159, v155, v161, v162);
  // pto: %cs_dup_idx_inline419__tile
  Tile<TileType::Vec, int32_t, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v163 = Tile<TileType::Vec, int32_t, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v33);
  // pto: %cs_dup_idx_inline419__tile
  uint64_t v164 = (uint64_t) v17;
  TASSIGN(v163, v164);
  pipe_barrier(PIPE_V);
  RoundMode v165 = RoundMode::CAST_ROUND;
  SaturationMode v166 = SaturationMode::ON;
  TCVT(v163, v159, v165, v166);
  // pto: %6
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v167 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v33);
  // pto: %6
  uint64_t v168 = (uint64_t) v16;
  TASSIGN(v167, v168);
  pipe_barrier(PIPE_V);
  TMULS(v167, v159, v31);
  // pto: %cs_lane_inline420__tile
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v169 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v33);
  // pto: %cs_lane_inline420__tile
  uint64_t v170 = (uint64_t) v15;
  TASSIGN(v169, v170);
  pipe_barrier(PIPE_V);
  TSUB(v169, v151, v167);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  // pto: %7
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v171 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v33);
  // pto: %7
  uint64_t v172 = (uint64_t) v15;
  TASSIGN(v171, v172);
  pipe_barrier(PIPE_V);
  TMULS(v171, v169, v31);
  // pto: %8
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v173 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v33);
  // pto: %8
  uint64_t v174 = (uint64_t) v15;
  TASSIGN(v173, v174);
  pipe_barrier(PIPE_V);
  TSUBS(v173, v171, v27);
  // pto: %cs_sign_inline421__tile
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v175 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v33);
  // pto: %cs_sign_inline421__tile
  uint64_t v176 = (uint64_t) v15;
  TASSIGN(v175, v176);
  pipe_barrier(PIPE_V);
  TNEG(v175, v173);
  // pto: %26
  int64_t v177 = (int64_t) ((uint64_t) v84 * (uint64_t) v33);
  // pto: %27
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  if (v177 < v6) {
    // pto: %28
    ;
    int64_t v178 = (int64_t) ((uint64_t) v6 - (uint64_t) v177);
    // pto: %29
    ;
    int64_t v179 = v178 < v33 ? v178 : v33;
    // pto: %cs_cos_rows_inline424__tile
    ;
    Tile<TileType::Vec, bfloat16_t, 32, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v180 = Tile<TileType::Vec, bfloat16_t, 32, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v179, v23);
    // pto: %cs_cos_rows_inline424__tile
    ;
    uint64_t v181 = (uint64_t) v16;
    TASSIGN(v180, v181);
    // pto: %30
    ;
    int64_t v182 = v177 < v32 ? v32 : v177;
    // pto: %31
    ;
    int64_t v183 = v85 < v32 ? v32 : v85;
    // pto: %freqs_cos__ssa_v0_pview
    ;
    const int64_t v184 = 0;
    // pto: %freqs_cos__ssa_v0_pview
    ;
    __gm__ bfloat16_t* v185 = PTOAS__GLOBAL_TENSOR_DATA(v54);
    // pto: %freqs_cos__ssa_v0_pview
    ;
    const int64_t v186 = 1;
    // pto: %freqs_cos__ssa_v0_pview
    ;
    const int64_t v187 = 1;
    // pto: %freqs_cos__ssa_v0_pview
    ;
    const int64_t v188 = 1;
    // pto: %freqs_cos__ssa_v0_pview
    ;
    int64_t v189 = v179 * v24;
    // pto: %freqs_cos__ssa_v0_pview
    ;
    int64_t v190 = v188 * v189;
    // pto: %freqs_cos__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, -1, 16> v191 = pto::Shape<1, 1, 1, -1, 16>(v186, v187, v188, v179, v23);
    // pto: %freqs_cos__ssa_v0_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v192 = pto::Stride<-1, -1, -1, -1, -1>(v187 * v190, v190, v189, v24, v25);
    // pto: %freqs_cos__ssa_v0_pview
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v193 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v185 + (v184 + v182 * v24 + v183 * v25), v191, v192);
    TLOAD(v180, v193);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %cs_sin_rows_inline425__tile
    ;
    Tile<TileType::Vec, bfloat16_t, 32, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v194 = Tile<TileType::Vec, bfloat16_t, 32, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v179, v23);
    // pto: %cs_sin_rows_inline425__tile
    ;
    uint64_t v195 = (uint64_t) v18;
    TASSIGN(v194, v195);
    // pto: %freqs_sin__ssa_v0_pview
    ;
    const int64_t v196 = 0;
    // pto: %freqs_sin__ssa_v0_pview
    ;
    __gm__ bfloat16_t* v197 = PTOAS__GLOBAL_TENSOR_DATA(v63);
    // pto: %freqs_sin__ssa_v0_pview
    ;
    const int64_t v198 = 1;
    // pto: %freqs_sin__ssa_v0_pview
    ;
    const int64_t v199 = 1;
    // pto: %freqs_sin__ssa_v0_pview
    ;
    const int64_t v200 = 1;
    // pto: %freqs_sin__ssa_v0_pview
    ;
    int64_t v201 = v179 * v24;
    // pto: %freqs_sin__ssa_v0_pview
    ;
    int64_t v202 = v200 * v201;
    // pto: %freqs_sin__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, -1, 16> v203 = pto::Shape<1, 1, 1, -1, 16>(v198, v199, v200, v179, v23);
    // pto: %freqs_sin__ssa_v0_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v204 = pto::Stride<-1, -1, -1, -1, -1>(v199 * v202, v202, v201, v24, v25);
    // pto: %freqs_sin__ssa_v0_pview
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v205 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v197 + (v196 + v182 * v24 + v183 * v25), v203, v204);
    TLOAD(v194, v205);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    // pto: %cs_cos_inline426__tile
    ;
    Tile<TileType::Vec, float, 32, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v206 = Tile<TileType::Vec, float, 32, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v179, v23);
    // pto: %cs_cos_inline426__tile
    ;
    uint64_t v207 = (uint64_t) v19;
    TASSIGN(v206, v207);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    RoundMode v208 = RoundMode::CAST_ROUND;
    SaturationMode v209 = SaturationMode::OFF;
    TCVT(v206, v180, v208, v209);
    // pto: %cs_sin_inline427__tile
    ;
    Tile<TileType::Vec, float, 32, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v210 = Tile<TileType::Vec, float, 32, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v179, v23);
    // pto: %cs_sin_inline427__tile
    ;
    uint64_t v211 = (uint64_t) v20;
    TASSIGN(v210, v211);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    RoundMode v212 = RoundMode::CAST_ROUND;
    SaturationMode v213 = SaturationMode::OFF;
    TCVT(v210, v194, v212, v213);
    // pto: %gather_acc_init
    ;
    Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v214 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v33);
    // pto: %gather_acc_init
    ;
    uint64_t v215 = (uint64_t) v16;
    TASSIGN(v214, v215);
    for (int64_t v216 = v32; v216 < v33; v216 += v25) {
      // pto: %34
      ;
      int64_t v217 = (int64_t) ((uint64_t) v179 - (uint64_t) v216);
      // pto: %35
      ;
      int64_t v218 = v217 < v32 ? v32 : v217;
      // pto: %36
      ;
      int64_t v219 = v218 < v25 ? v218 : v25;
      int64_t v220 = (int64_t) ((uint64_t) v216 * (uint64_t) v24);
      // pto: %39
      ;
      int64_t v221 = (int64_t) ((uint64_t) v220 + (uint64_t) v35);
      // pto: %gather_inp_row
      ;
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v222 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v219, v23);
      // pto: %gather_inp_row
      ;
      uint64_t v223 = (uint64_t) v221;
      TASSIGN(v222, v223);
      // pto: %41
      ;
      int64_t v224 = (int64_t) ((uint64_t) v216 * (uint64_t) v12);
      // pto: %41
      ;
      int64_t v225 = (int64_t) ((uint64_t) v224 + (uint64_t) v19);
      // pto: %41
      ;
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v226;
      // pto: %41
      ;
      uint64_t v227 = (uint64_t) v225;
      TASSIGN(v226, v227);
      int64_t v228 = (int64_t) ((uint64_t) v216 * (uint64_t) v13);
      // pto: %44
      ;
      int64_t v229 = (int64_t) ((uint64_t) v228 + (uint64_t) v36);
      // pto: %gather_idx_row
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v230 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v33);
      // pto: %gather_idx_row
      ;
      uint64_t v231 = (uint64_t) v229;
      TASSIGN(v230, v231);
      // pto: %46
      ;
      int64_t v232 = (int64_t) ((uint64_t) v216 * (uint64_t) v11);
      // pto: %46
      ;
      int64_t v233 = (int64_t) ((uint64_t) v232 + (uint64_t) v17);
      // pto: %46
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v234;
      // pto: %46
      ;
      uint64_t v235 = (uint64_t) v233;
      TASSIGN(v234, v235);
      // pto: %gather_row_tmp
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v236 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v33);
      // pto: %gather_row_tmp
      ;
      uint64_t v237 = (uint64_t) v21;
      TASSIGN(v236, v237);
      // pto: %gather_row
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v238 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v33);
      // pto: %gather_row
      ;
      uint64_t v239 = (uint64_t) v18;
      TASSIGN(v238, v239);
      pipe_barrier(PIPE_V);
      TGATHER(v238, v226, v234, v236);
      // pto: %assemble_view
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v240;
      // pto: %assemble_view
      ;
      uint64_t v241 = (uint64_t) v232;
      TASSIGN(v240, v241);
      pipe_barrier(PIPE_V);
      TMOV(v240, v238);
    };
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    // pto: %cs_cos_il_inline428__tile
    ;
    Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v242 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v33);
    // pto: %cs_cos_il_inline428__tile
    ;
    uint64_t v243 = (uint64_t) v16;
    TASSIGN(v242, v243);
    // pto: %9
    ;
    Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v244 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v33);
    // pto: %9
    ;
    uint64_t v245 = (uint64_t) v19;
    TASSIGN(v244, v245);
    for (int64_t v246 = v32; v246 < v33; v246 += v25) {
      // pto: %48
      ;
      int64_t v247 = (int64_t) ((uint64_t) v179 - (uint64_t) v246);
      // pto: %49
      ;
      int64_t v248 = v247 < v32 ? v32 : v247;
      // pto: %50
      ;
      int64_t v249 = v248 < v25 ? v248 : v25;
      int64_t v250 = (int64_t) ((uint64_t) v246 * (uint64_t) v24);
      // pto: %53
      ;
      int64_t v251 = (int64_t) ((uint64_t) v250 + (uint64_t) v37);
      // pto: %10
      ;
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v252 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v249, v23);
      // pto: %10
      ;
      uint64_t v253 = (uint64_t) v251;
      TASSIGN(v252, v253);
      // pto: %55
      ;
      int64_t v254 = (int64_t) ((uint64_t) v246 * (uint64_t) v12);
      // pto: %55
      ;
      int64_t v255 = (int64_t) ((uint64_t) v254 + (uint64_t) v20);
      // pto: %55
      ;
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v256;
      // pto: %55
      ;
      uint64_t v257 = (uint64_t) v255;
      TASSIGN(v256, v257);
      int64_t v258 = (int64_t) ((uint64_t) v246 * (uint64_t) v13);
      // pto: %58
      ;
      int64_t v259 = (int64_t) ((uint64_t) v258 + (uint64_t) v36);
      // pto: %11
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v260 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v33);
      // pto: %11
      ;
      uint64_t v261 = (uint64_t) v259;
      TASSIGN(v260, v261);
      // pto: %60
      ;
      int64_t v262 = (int64_t) ((uint64_t) v246 * (uint64_t) v11);
      // pto: %60
      ;
      int64_t v263 = (int64_t) ((uint64_t) v262 + (uint64_t) v17);
      // pto: %60
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v264;
      // pto: %60
      ;
      uint64_t v265 = (uint64_t) v263;
      TASSIGN(v264, v265);
      // pto: %12
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v266 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v33);
      // pto: %12
      ;
      uint64_t v267 = (uint64_t) v22;
      TASSIGN(v266, v267);
      // pto: %13
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v268 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v33);
      // pto: %13
      ;
      uint64_t v269 = (uint64_t) v18;
      TASSIGN(v268, v269);
      pipe_barrier(PIPE_V);
      TGATHER(v268, v256, v264, v266);
      // pto: %61
      ;
      int64_t v270 = (int64_t) ((uint64_t) v262 + (uint64_t) v19);
      // pto: %61
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v271;
      // pto: %61
      ;
      uint64_t v272 = (uint64_t) v270;
      TASSIGN(v271, v272);
      pipe_barrier(PIPE_V);
      TMOV(v271, v268);
    };
    // pto: %cs_sin_il_inline429__tile
    ;
    Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v273 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v33);
    // pto: %cs_sin_il_inline429__tile
    ;
    uint64_t v274 = (uint64_t) v19;
    TASSIGN(v273, v274);
    // pto: %cs_sin_signed_inline430__tile
    ;
    Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v275 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v33);
    // pto: %cs_sin_signed_inline430__tile
    ;
    uint64_t v276 = (uint64_t) v15;
    TASSIGN(v275, v276);
    pipe_barrier(PIPE_V);
    TMUL(v275, v273, v175);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    // pto: %63
    ;
    int64_t v277 = v86 < v32 ? v32 : v86;
    // pto: %rope_cos_il_view_inline397__ssa_v0_pview
    ;
    __gm__ float* v278 = PTOAS__GLOBAL_TENSOR_DATA(v72);
    // pto: %rope_cos_il_view_inline397__ssa_v0_pview
    ;
    const int64_t v279 = 0;
    // pto: %rope_cos_il_view_inline397__ssa_v0_pview
    ;
    const int64_t v280 = 64;
    // pto: %rope_cos_il_view_inline397__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 32, 32> v281 = pto::Shape<1, 1, 1, 32, 32>();
    // pto: %rope_cos_il_view_inline397__ssa_v0_pview
    ;
    pto::Stride<2048, 2048, 2048, 64, 1> v282 = pto::Stride<2048, 2048, 2048, 64, 1>();
    // pto: %rope_cos_il_view_inline397__ssa_v0_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND> v283 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND>(v278 + (v279 + v182 * v280 + v277), v281, v282);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v283, v242);
    // pto: %rope_sin_signed_view_inline398__ssa_v0_pview
    ;
    __gm__ float* v284 = PTOAS__GLOBAL_TENSOR_DATA(v81);
    // pto: %rope_sin_signed_view_inline398__ssa_v0_pview
    ;
    const int64_t v285 = 0;
    // pto: %rope_sin_signed_view_inline398__ssa_v0_pview
    ;
    const int64_t v286 = 64;
    // pto: %rope_sin_signed_view_inline398__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 32, 32> v287 = pto::Shape<1, 1, 1, 32, 32>();
    // pto: %rope_sin_signed_view_inline398__ssa_v0_pview
    ;
    pto::Stride<2048, 2048, 2048, 64, 1> v288 = pto::Stride<2048, 2048, 2048, 64, 1>();
    // pto: %rope_sin_signed_view_inline398__ssa_v0_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND> v289 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND>(v284 + (v285 + v182 * v286 + v277), v287, v288);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    TSTORE(v289, v275);
  }
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}