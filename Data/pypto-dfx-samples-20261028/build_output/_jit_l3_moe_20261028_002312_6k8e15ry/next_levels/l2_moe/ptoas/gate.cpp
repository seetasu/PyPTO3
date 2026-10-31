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

AICORE void gate_aic(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ float* v4, __gm__ float* v5, __gm__ float* v6, __gm__ float* v7, int32_t v8, int64_t v9, int64_t v10, int32_t v11, int32_t v12) {
  using T = float;

  #if defined(__DAV_CUBE__)
  const int32_t v13 = 0;
  // pto: %c0_i64
  const int64_t v14 = 0;
  // pto: %c262144_i64
  const int64_t v15 = 262144;
  // pto: %c393216_i64
  const int64_t v16 = 393216;
  // pto: %c131072_i64
  const int64_t v17 = 131072;
  // pto: %c32768_i64
  const int64_t v18 = 32768;
  // pto: %c64_index
  const int64_t v19 = 64;
  // pto: %c1_index
  const int64_t v20 = 1;
  // pto: %c4096_index
  const int64_t v21 = 4096;
  // pto: %c4_index
  const int64_t v22 = 4;
  // pto: %c16_index
  const int64_t v23 = 16;
  // pto: %c2048_index
  const int64_t v24 = 2048;
  // pto: %c0_index
  const int64_t v25 = 0;
  // pto: %c1024_index
  const int64_t v26 = 1024;
  // pto: %c512_index
  const int64_t v27 = 512;
  // pto: %cn512_index
  const int64_t v28 = -512;
  // pto: %xg_buf_inline170__ssa_v0_view
  const int64_t v29 = 1;
  // pto: %xg_buf_inline170__ssa_v0_view
  const int64_t v30 = 1;
  // pto: %xg_buf_inline170__ssa_v0_view
  const int64_t v31 = 1;
  // pto: %xg_buf_inline170__ssa_v0_view
  int64_t v32 = (int64_t) v9;
  // pto: %xg_buf_inline170__ssa_v0_view
  int64_t v33 = v32 * v21;
  // pto: %xg_buf_inline170__ssa_v0_view
  int64_t v34 = v31 * v33;
  // pto: %xg_buf_inline170__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v35 = pto::Shape<1, 1, 1, -1, -1>(v29, v30, v31, v32, v21);
  // pto: %xg_buf_inline170__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v36 = pto::Stride<-1, -1, -1, -1, -1>(v30 * v34, v34, v33, v21, v20);
  // pto: %xg_buf_inline170__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v37 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v35, v36);
  // pto: %gate_w__ssa_v0_view
  const int64_t v38 = 1;
  // pto: %gate_w__ssa_v0_view
  const int64_t v39 = 1;
  // pto: %gate_w__ssa_v0_view
  const int64_t v40 = 1;
  // pto: %gate_w__ssa_v0_view
  int64_t v41 = v19 * v21;
  // pto: %gate_w__ssa_v0_view
  int64_t v42 = v40 * v41;
  // pto: %gate_w__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v43 = pto::Shape<1, 1, 1, -1, -1>(v38, v39, v40, v19, v21);
  // pto: %gate_w__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v44 = pto::Stride<-1, -1, -1, -1, -1>(v39 * v42, v42, v41, v21, v20);
  // pto: %gate_w__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v45 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v43, v44);
  const int32_t v46 = 0;
  auto v47 = TPipe<0, Direction::DIR_C2V, 1024, 2, 2, true>(v7, v13, v46);
  // pto: %gb_idx_inline221__ssa_v0
  int64_t v48 = (int64_t) v11;
  // pto: %12, %14
  int64_t v49 = (int64_t) ((uint64_t) (v48 / v22) * (uint64_t) v23);
  // pto: %13, %15
  int64_t v50 = (int64_t) ((uint64_t) (v48 % v22) * (uint64_t) v23);
  // pto: %gate_logits_tile_inline227__tile
  Tile<TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v51 = Tile<TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v23, v23);
  // pto: %gate_logits_tile_inline227__tile
  uint64_t v52 = (uint64_t) v14;
  TASSIGN(v51, v52);
  // pto: %gd_x_inline230__tile
  Tile<TileType::Mat, float, 16, 2048, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v53 = Tile<TileType::Mat, float, 16, 2048, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v23, v24);
  // pto: %gd_x_inline230__tile
  uint64_t v54 = (uint64_t) v15;
  TASSIGN(v53, v54);
  // pto: %16
  int64_t v55 = v49 < v25 ? v25 : v49;
  // pto: %xg_buf_inline170__ssa_v0_pview
  __gm__ float* v56 = PTOAS__GLOBAL_TENSOR_DATA(v37);
  // pto: %xg_buf_inline170__ssa_v0_pview
  const int64_t v57 = 0;
  // pto: %xg_buf_inline170__ssa_v0_pview
  const int64_t v58 = 4096;
  // pto: %xg_buf_inline170__ssa_v0_pview
  pto::Shape<1, 1, 1, 16, 2048> v59 = pto::Shape<1, 1, 1, 16, 2048>();
  // pto: %xg_buf_inline170__ssa_v0_pview
  pto::Stride<65536, 65536, 65536, 4096, 1> v60 = pto::Stride<65536, 65536, 65536, 4096, 1>();
  // pto: %xg_buf_inline170__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND> v61 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>(v56 + (v57 + v55 * v58), v59, v60);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
  TLOAD(v53, v61);
  // pto: %gd_w_inline231__tile
  Tile<TileType::Mat, float, 16, 2048, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v62 = Tile<TileType::Mat, float, 16, 2048, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v23, v24);
  // pto: %gd_w_inline231__tile
  uint64_t v63 = (uint64_t) v16;
  TASSIGN(v62, v63);
  // pto: %17
  int64_t v64 = v50 < v25 ? v25 : v50;
  // pto: %gate_w__ssa_v0_pview
  __gm__ float* v65 = PTOAS__GLOBAL_TENSOR_DATA(v45);
  // pto: %gate_w__ssa_v0_pview
  const int64_t v66 = 0;
  // pto: %gate_w__ssa_v0_pview
  const int64_t v67 = 4096;
  // pto: %gate_w__ssa_v0_pview
  pto::Shape<1, 1, 1, 16, 2048> v68 = pto::Shape<1, 1, 1, 16, 2048>();
  // pto: %gate_w__ssa_v0_pview
  pto::Stride<65536, 65536, 65536, 4096, 1> v69 = pto::Stride<65536, 65536, 65536, 4096, 1>();
  // pto: %gate_w__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND> v70 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>(v65 + (v66 + v64 * v67), v68, v69);
  __gm__ float* v71 = PTOAS__GLOBAL_TENSOR_DATA(v70);
  __gm__ uint8_t* v72 = reinterpret_cast<__gm__ uint8_t*>(v71);
  __gm__ float* v73 = (__gm__ float*) (v72 + v10);
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND> v74(nullptr);
  v74 = v70;
  TASSIGN(v74, v73);
  TLOAD(v62, v74);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  // pto: %0
  Tile<TileType::Mat, float, 16, 2048, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v75 = Tile<TileType::Mat, float, 16, 2048, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v23, v24);
  // pto: %0
  uint64_t v76 = (uint64_t) v14;
  TASSIGN(v75, v76);
  // pto: %19
  __gm__ float* v77 = PTOAS__GLOBAL_TENSOR_DATA(v37);
  // pto: %19
  const int64_t v78 = 2048;
  // pto: %19
  const int64_t v79 = 4096;
  // pto: %19
  pto::Shape<1, 1, 1, 16, 2048> v80 = pto::Shape<1, 1, 1, 16, 2048>();
  // pto: %19
  pto::Stride<65536, 65536, 65536, 4096, 1> v81 = pto::Stride<65536, 65536, 65536, 4096, 1>();
  // pto: %19
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND> v82 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>(v77 + (v78 + v55 * v79), v80, v81);
  TLOAD(v75, v82);
  // pto: %1
  Tile<TileType::Mat, float, 16, 2048, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v83 = Tile<TileType::Mat, float, 16, 2048, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v23, v24);
  // pto: %1
  uint64_t v84 = (uint64_t) v17;
  TASSIGN(v83, v84);
  // pto: %21
  __gm__ float* v85 = PTOAS__GLOBAL_TENSOR_DATA(v45);
  // pto: %21
  const int64_t v86 = 2048;
  // pto: %21
  const int64_t v87 = 4096;
  // pto: %21
  pto::Shape<1, 1, 1, 16, 2048> v88 = pto::Shape<1, 1, 1, 16, 2048>();
  // pto: %21
  pto::Stride<65536, 65536, 65536, 4096, 1> v89 = pto::Stride<65536, 65536, 65536, 4096, 1>();
  // pto: %21
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND> v90 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>(v85 + (v86 + v64 * v87), v88, v89);
  __gm__ float* v91 = PTOAS__GLOBAL_TENSOR_DATA(v90);
  __gm__ uint8_t* v92 = reinterpret_cast<__gm__ uint8_t*>(v91);
  __gm__ float* v93 = (__gm__ float*) (v92 + v10);
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND> v94(nullptr);
  v94 = v90;
  TASSIGN(v94, v93);
  TLOAD(v83, v94);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  // pto: %gd_w_inline231__tile_t
  Tile<TileType::Mat, float, 2048, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v95 = Tile<TileType::Mat, float, 2048, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v24, v23);
  // pto: %gd_w_inline231__tile_t
  uint64_t v96 = (uint64_t) v16;
  TASSIGN(v95, v96);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  for (int64_t v97 = v25; v97 < v24; v97 += v26) {
    // pto: %gate_logits_tile_inline227__tile_l0_a
    ;
    Tile<TileType::Left, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v98 = Tile<TileType::Left, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v23, v27);
    // pto: %gate_logits_tile_inline227__tile_l0_a
    ;
    uint64_t v99 = (uint64_t) v14;
    TASSIGN(v98, v99);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    pipe_barrier(PIPE_MTE1);
    uint16_t v100 = static_cast<uint16_t>(v25);
    uint16_t v101 = static_cast<uint16_t>(v97);
    TEXTRACT(v98, v53, v100, v101);
    // pto: %gate_logits_tile_inline227__tile_l0_b
    ;
    Tile<TileType::Right, float, 512, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v102 = Tile<TileType::Right, float, 512, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v27, v23);
    // pto: %gate_logits_tile_inline227__tile_l0_b
    ;
    uint64_t v103 = (uint64_t) v14;
    TASSIGN(v102, v103);
    uint16_t v104 = static_cast<uint16_t>(v97);
    uint16_t v105 = static_cast<uint16_t>(v25);
    TEXTRACT(v102, v95, v104, v105);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    // pto: %2
    ;
    Tile<TileType::Left, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v106 = Tile<TileType::Left, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v23, v27);
    // pto: %2
    ;
    uint64_t v107 = (uint64_t) v18;
    TASSIGN(v106, v107);
    // pto: %22
    ;
    int64_t v108 = (int64_t) ((uint64_t) v97 + (uint64_t) v27);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    uint16_t v109 = static_cast<uint16_t>(v25);
    uint16_t v110 = static_cast<uint16_t>(v108);
    TEXTRACT(v106, v53, v109, v110);
    // pto: %3
    ;
    Tile<TileType::Right, float, 512, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v111 = Tile<TileType::Right, float, 512, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v27, v23);
    // pto: %3
    ;
    uint64_t v112 = (uint64_t) v18;
    TASSIGN(v111, v112);
    uint16_t v113 = static_cast<uint16_t>(v108);
    uint16_t v114 = static_cast<uint16_t>(v25);
    TEXTRACT(v111, v95, v113, v114);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    // pto: %24
    ;
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    if (v97 == v25) {
      pipe_barrier(PIPE_M);
      TMATMUL(v51, v98, v102);
    } else {
      pipe_barrier(PIPE_M);
      TMATMUL_ACC(v51, v51, v98, v102);
    };
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    // pto: %25
    ;
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    if (v97 == v28) {
      pipe_barrier(PIPE_M);
      TMATMUL(v51, v106, v111);
    } else {
      pipe_barrier(PIPE_M);
      TMATMUL_ACC(v51, v51, v106, v111);
    };
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  }
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  // pto: %5
  Tile<TileType::Mat, float, 2048, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v115 = Tile<TileType::Mat, float, 2048, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v24, v23);
  // pto: %5
  uint64_t v116 = (uint64_t) v17;
  TASSIGN(v115, v116);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  for (int64_t v117 = v25; v117 < v24; v117 += v26) {
    // pto: %6
    ;
    Tile<TileType::Left, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v118 = Tile<TileType::Left, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v23, v27);
    // pto: %6
    ;
    uint64_t v119 = (uint64_t) v14;
    TASSIGN(v118, v119);
    pipe_barrier(PIPE_MTE1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    uint16_t v120 = static_cast<uint16_t>(v25);
    uint16_t v121 = static_cast<uint16_t>(v117);
    TEXTRACT(v118, v75, v120, v121);
    // pto: %7
    ;
    Tile<TileType::Right, float, 512, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v122 = Tile<TileType::Right, float, 512, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v27, v23);
    // pto: %7
    ;
    uint64_t v123 = (uint64_t) v14;
    TASSIGN(v122, v123);
    uint16_t v124 = static_cast<uint16_t>(v117);
    uint16_t v125 = static_cast<uint16_t>(v25);
    TEXTRACT(v122, v115, v124, v125);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
    // pto: %8
    ;
    Tile<TileType::Left, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v126 = Tile<TileType::Left, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v23, v27);
    // pto: %8
    ;
    uint64_t v127 = (uint64_t) v18;
    TASSIGN(v126, v127);
    // pto: %27
    ;
    int64_t v128 = (int64_t) ((uint64_t) v117 + (uint64_t) v27);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
    uint16_t v129 = static_cast<uint16_t>(v25);
    uint16_t v130 = static_cast<uint16_t>(v128);
    TEXTRACT(v126, v75, v129, v130);
    // pto: %9
    ;
    Tile<TileType::Right, float, 512, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v131 = Tile<TileType::Right, float, 512, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v27, v23);
    // pto: %9
    ;
    uint64_t v132 = (uint64_t) v18;
    TASSIGN(v131, v132);
    uint16_t v133 = static_cast<uint16_t>(v128);
    uint16_t v134 = static_cast<uint16_t>(v25);
    TEXTRACT(v131, v115, v133, v134);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
    pipe_barrier(PIPE_M);
    TMATMUL_ACC(v51, v51, v118, v122);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    pipe_barrier(PIPE_M);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
    TMATMUL_ACC(v51, v51, v126, v131);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
  }
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TPUSH<TPipe<0, Direction::DIR_C2V, 1024, 2, 2, true>, Tile<TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_NO_SPLIT>(v47, v51);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
  #endif // __DAV_CUBE__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}

AICORE void gate_aiv(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ float* v4, __gm__ float* v5, __gm__ float* v6, __gm__ float* v7, int32_t v8, int64_t v9, int32_t v10, int32_t v11, int32_t v12) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  const int32_t v13 = 0;
  // pto: %c4096_i64
  const int64_t v14 = 4096;
  // pto: %c2048_i64
  const int64_t v15 = 2048;
  // pto: %c4160_i64
  const int64_t v16 = 4160;
  // pto: %c3072_i64
  const int64_t v17 = 3072;
  // pto: %c64_index
  const int64_t v18 = 64;
  // pto: %c1_index
  const int64_t v19 = 1;
  // pto: %c256_index
  const int64_t v20 = 256;
  // pto: %c0_index
  const int64_t v21 = 0;
  // pto: %c4_index
  const int64_t v22 = 4;
  // pto: %c16_index
  const int64_t v23 = 16;
  // pto: %cst_12
  const float v24 = 0.0f;
  // pto: %cst_13
  const float v25 = 1.0f;
  // pto: %cst_14
  const float v26 = 10.0f;
  // pto: %c3_index
  const int64_t v27 = 3;
  // pto: %gate_bias__ssa_v0_view
  const int64_t v28 = 1;
  // pto: %gate_bias__ssa_v0_view
  const int64_t v29 = 1;
  // pto: %gate_bias__ssa_v0_view
  const int64_t v30 = 1;
  // pto: %gate_bias__ssa_v0_view
  const int64_t v31 = 1;
  // pto: %gate_bias__ssa_v0_view
  int64_t v32 = v18 * v19;
  // pto: %gate_bias__ssa_v0_view
  int64_t v33 = v31 * v32;
  // pto: %gate_bias__ssa_v0_view
  int64_t v34 = v30 * v33;
  // pto: %gate_bias__ssa_v0_view
  pto::Shape<1, 1, 1, 1, -1> v35 = pto::Shape<1, 1, 1, 1, -1>(v28, v29, v30, v31, v18);
  // pto: %gate_bias__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v36 = pto::Stride<-1, -1, -1, -1, -1>(v29 * v34, v34, v33, v32, v19);
  // pto: %gate_bias__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v37 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v35, v36);
  // pto: %inv_rms_buf_inline171__ssa_v0_view
  const int64_t v38 = 1;
  // pto: %inv_rms_buf_inline171__ssa_v0_view
  const int64_t v39 = 1;
  // pto: %inv_rms_buf_inline171__ssa_v0_view
  const int64_t v40 = 1;
  // pto: %inv_rms_buf_inline171__ssa_v0_view
  int64_t v41 = (int64_t) v9;
  // pto: %inv_rms_buf_inline171__ssa_v0_view
  int64_t v42 = v41 * v19;
  // pto: %inv_rms_buf_inline171__ssa_v0_view
  int64_t v43 = v40 * v42;
  // pto: %inv_rms_buf_inline171__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v44 = pto::Shape<1, 1, 1, -1, -1>(v38, v39, v40, v41, v19);
  // pto: %inv_rms_buf_inline171__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v45 = pto::Stride<-1, -1, -1, -1, -1>(v39 * v43, v43, v42, v19, (int64_t) v9);
  // pto: %inv_rms_buf_inline171__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v46 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v4, v44, v45);
  // pto: %route_scores_buf_inline173__ssa_v0_view
  const int64_t v47 = 1;
  // pto: %route_scores_buf_inline173__ssa_v0_view
  const int64_t v48 = 1;
  // pto: %route_scores_buf_inline173__ssa_v0_view
  const int64_t v49 = 1;
  // pto: %route_scores_buf_inline173__ssa_v0_view
  int64_t v50 = (int64_t) v9;
  // pto: %route_scores_buf_inline173__ssa_v0_view
  int64_t v51 = v50 * v20;
  // pto: %route_scores_buf_inline173__ssa_v0_view
  int64_t v52 = v49 * v51;
  // pto: %route_scores_buf_inline173__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v53 = pto::Shape<1, 1, 1, -1, -1>(v47, v48, v49, v50, v20);
  // pto: %route_scores_buf_inline173__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v54 = pto::Stride<-1, -1, -1, -1, -1>(v48 * v52, v52, v51, v20, v19);
  // pto: %route_scores_buf_inline173__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v55 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v5, v53, v54);
  // pto: %biased_scores_buf_inline174__rv_v2_view
  const int64_t v56 = 1;
  // pto: %biased_scores_buf_inline174__rv_v2_view
  const int64_t v57 = 1;
  // pto: %biased_scores_buf_inline174__rv_v2_view
  const int64_t v58 = 1;
  // pto: %biased_scores_buf_inline174__rv_v2_view
  int64_t v59 = (int64_t) v9;
  // pto: %biased_scores_buf_inline174__rv_v2_view
  int64_t v60 = v59 * v20;
  // pto: %biased_scores_buf_inline174__rv_v2_view
  int64_t v61 = v58 * v60;
  // pto: %biased_scores_buf_inline174__rv_v2_view
  pto::Shape<1, 1, 1, -1, -1> v62 = pto::Shape<1, 1, 1, -1, -1>(v56, v57, v58, v59, v20);
  // pto: %biased_scores_buf_inline174__rv_v2_view
  pto::Stride<-1, -1, -1, -1, -1> v63 = pto::Stride<-1, -1, -1, -1, -1>(v57 * v61, v61, v60, v20, v19);
  // pto: %biased_scores_buf_inline174__rv_v2_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v64 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v6, v62, v63);
  const int32_t v65 = 0;
  auto v66 = TPipe<0, Direction::DIR_C2V, 1024, 2, 2, true>(v7, v13, v65);
  // pto: %subblock_idx, %36
  if ((int64_t) v12 == v21) {
    // pto: %gb_idx_inline221__ssa_v0
    ;
    int64_t v67 = (int64_t) v10;
    // pto: %37, %39
    ;
    int64_t v68 = (int64_t) ((uint64_t) (v67 / v22) * (uint64_t) v23);
    // pto: %38, %40
    ;
    int64_t v69 = (int64_t) ((uint64_t) (v67 % v22) * (uint64_t) v23);
    // pto: %t__tile
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v70 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v19, v23);
    // pto: %t__tile
    ;
    uint64_t v71 = (uint64_t) v14;
    TASSIGN(v70, v71);
    // pto: %41
    ;
    int64_t v72 = v69 < v21 ? v21 : v69;
    // pto: %gate_bias__ssa_v0_pview
    ;
    __gm__ float* v73 = PTOAS__GLOBAL_TENSOR_DATA(v37);
    // pto: %gate_bias__ssa_v0_pview
    ;
    const int64_t v74 = 0;
    // pto: %gate_bias__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 1, 16> v75 = pto::Shape<1, 1, 1, 1, 16>();
    // pto: %gate_bias__ssa_v0_pview
    ;
    pto::Stride<16, 16, 16, 16, 1> v76 = pto::Stride<16, 16, 16, 16, 1>();
    // pto: %gate_bias__ssa_v0_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v77 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v73 + (v74 + v72), v75, v76);
    TLOAD(v70, v77);
    // pto: %gp_bias_row_inline226__tile
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v78 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v19, v23);
    // pto: %gp_bias_row_inline226__tile
    ;
    uint64_t v79 = (uint64_t) v14;
    TASSIGN(v78, v79);
    // pto: %0
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v80 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v19);
    // pto: %0
    ;
    uint64_t v81 = (uint64_t) v15;
    TASSIGN(v80, v81);
    // pto: %42
    ;
    int64_t v82 = v68 < v21 ? v21 : v68;
    // pto: %inv_rms_buf_inline171__ssa_v0_pview
    ;
    int64_t v83 = (int64_t) v9;
    // pto: %inv_rms_buf_inline171__ssa_v0_pview
    ;
    const int64_t v84 = 0;
    // pto: %inv_rms_buf_inline171__ssa_v0_pview
    ;
    __gm__ float* v85 = PTOAS__GLOBAL_TENSOR_DATA(v46);
    // pto: %inv_rms_buf_inline171__ssa_v0_pview
    ;
    const int64_t v86 = 1;
    // pto: %inv_rms_buf_inline171__ssa_v0_pview
    ;
    const int64_t v87 = 1;
    // pto: %inv_rms_buf_inline171__ssa_v0_pview
    ;
    const int64_t v88 = 1;
    // pto: %inv_rms_buf_inline171__ssa_v0_pview
    ;
    int64_t v89 = v23 * v19;
    // pto: %inv_rms_buf_inline171__ssa_v0_pview
    ;
    int64_t v90 = v88 * v89;
    // pto: %inv_rms_buf_inline171__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 16, 1> v91 = pto::Shape<1, 1, 1, 16, 1>(v86, v87, v88, v23, v19);
    // pto: %inv_rms_buf_inline171__ssa_v0_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v92 = pto::Stride<-1, -1, -1, -1, -1>(v87 * v90, v90, v89, v19, v83);
    // pto: %inv_rms_buf_inline171__ssa_v0_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v93 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v85 + (v84 + v82 * v19 + v21 * v83), v91, v92);
    TLOAD(v80, v93);
    // pto: %gate_logits_tile_inline227__rv_v2_Vec
    ;
    const int32_t v94 = 16;
    // pto: %gate_logits_tile_inline227__rv_v2_Vec
    ;
    const int32_t v95 = 16;
    // pto: %gate_logits_tile_inline227__rv_v2_Vec
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v96 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v94, v95);
    TPOP<TPipe<0, Direction::DIR_C2V, 1024, 2, 2, true>, Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_NO_SPLIT>(v66, v96);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %gate_logits_tile_v1_inline232__tile
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v97 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v23);
    // pto: %gate_logits_tile_v1_inline232__tile
    ;
    uint64_t v98 = (uint64_t) v16;
    TASSIGN(v97, v98);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TROWEXPANDMUL(v97, v96, v80);
    TFREE<TPipe<0, Direction::DIR_C2V, 1024, 2, 2, true>, TileSplitAxis::TILE_NO_SPLIT>(v66);
    // pto: %gp_relu_inline233__tile
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v99 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v23);
    // pto: %gp_relu_inline233__tile
    ;
    uint64_t v100 = (uint64_t) v15;
    TASSIGN(v99, v100);
    pipe_barrier(PIPE_V);
    TMAXS(v99, v97, v24);
    // pto: %1
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v101 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v23);
    // pto: %1
    ;
    uint64_t v102 = (uint64_t) v17;
    TASSIGN(v101, v102);
    TNEG(v101, v97);
    // pto: %gp_abs_inline234__tile
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v103 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v23);
    // pto: %gp_abs_inline234__tile
    ;
    uint64_t v104 = (uint64_t) v17;
    TASSIGN(v103, v104);
    pipe_barrier(PIPE_V);
    TMAX(v103, v97, v101);
    // pto: %2
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v105 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v23);
    // pto: %2
    ;
    uint64_t v106 = (uint64_t) v17;
    TASSIGN(v105, v106);
    pipe_barrier(PIPE_V);
    TNEG(v105, v103);
    // pto: %3
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v107 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v23);
    // pto: %3
    ;
    uint64_t v108 = (uint64_t) v17;
    TASSIGN(v107, v108);
    pipe_barrier(PIPE_V);
    TEXP(v107, v105);
    // pto: %4
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v109 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v23);
    // pto: %4
    ;
    uint64_t v110 = (uint64_t) v17;
    TASSIGN(v109, v110);
    pipe_barrier(PIPE_V);
    TADDS(v109, v107, v25);
    // pto: %5
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v111 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v23);
    // pto: %5
    ;
    uint64_t v112 = (uint64_t) v17;
    TASSIGN(v111, v112);
    pipe_barrier(PIPE_V);
    TLOG(v111, v109);
    // pto: %gp_softplus_log_inline235__tile
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v113 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v23);
    // pto: %gp_softplus_log_inline235__tile
    ;
    uint64_t v114 = (uint64_t) v15;
    TASSIGN(v113, v114);
    pipe_barrier(PIPE_V);
    TADD(v113, v99, v111);
    // pto: %6
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v115 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v23);
    // pto: %6
    ;
    uint64_t v116 = (uint64_t) v17;
    TASSIGN(v115, v116);
    pipe_barrier(PIPE_V);
    TNEG(v115, v97);
    // pto: %7
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v117 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v23);
    // pto: %7
    ;
    uint64_t v118 = (uint64_t) v17;
    TASSIGN(v117, v118);
    pipe_barrier(PIPE_V);
    TSUBS(v117, v115, v26);
    // pto: %8
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v119 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v23);
    // pto: %8
    ;
    uint64_t v120 = (uint64_t) v17;
    TASSIGN(v119, v120);
    pipe_barrier(PIPE_V);
    TMAXS(v119, v117, v24);
    // pto: %gp_neg_floor_mask_inline236__tile
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v121 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v23);
    // pto: %gp_neg_floor_mask_inline236__tile
    ;
    uint64_t v122 = (uint64_t) v17;
    TASSIGN(v121, v122);
    pipe_barrier(PIPE_V);
    TMINS(v121, v119, v25);
    // pto: %9
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v123 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v23);
    // pto: %9
    ;
    uint64_t v124 = (uint64_t) v16;
    TASSIGN(v123, v124);
    TMINS(v123, v97, v24);
    // pto: %10
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v125 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v23);
    // pto: %10
    ;
    uint64_t v126 = (uint64_t) v16;
    TASSIGN(v125, v126);
    pipe_barrier(PIPE_V);
    TEXP(v125, v123);
    // pto: %gp_neg_floor_inline237__tile
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v127 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v23);
    // pto: %gp_neg_floor_inline237__tile
    ;
    uint64_t v128 = (uint64_t) v16;
    TASSIGN(v127, v128);
    pipe_barrier(PIPE_V);
    TMUL(v127, v121, v125);
    // pto: %gp_softplus_inline238__tile
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v129 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v23);
    // pto: %gp_softplus_inline238__tile
    ;
    uint64_t v130 = (uint64_t) v16;
    TASSIGN(v129, v130);
    pipe_barrier(PIPE_V);
    TMAX(v129, v113, v127);
    // pto: %gp_score_inline239__tile
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v131 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v23);
    // pto: %gp_score_inline239__tile
    ;
    uint64_t v132 = (uint64_t) v16;
    TASSIGN(v131, v132);
    pipe_barrier(PIPE_V);
    TSQRT(v131, v129);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %route_scores_buf_inline173__ssa_v0_pview
    ;
    __gm__ float* v133 = PTOAS__GLOBAL_TENSOR_DATA(v55);
    // pto: %route_scores_buf_inline173__ssa_v0_pview
    ;
    const int64_t v134 = 0;
    // pto: %route_scores_buf_inline173__ssa_v0_pview
    ;
    const int64_t v135 = 256;
    // pto: %route_scores_buf_inline173__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 16, 16> v136 = pto::Shape<1, 1, 1, 16, 16>();
    // pto: %route_scores_buf_inline173__ssa_v0_pview
    ;
    pto::Stride<4096, 4096, 4096, 256, 1> v137 = pto::Stride<4096, 4096, 4096, 256, 1>();
    // pto: %route_scores_buf_inline173__ssa_v0_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND> v138 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND>(v133 + (v134 + v82 * v135 + v72), v136, v137);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v138, v131);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    // pto: %45, %46
    ;
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    if ((int64_t) v8 >= v27) {
      // pto: %11
      ;
      Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v139 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v23);
      // pto: %11
      ;
      uint64_t v140 = (uint64_t) v15;
      TASSIGN(v139, v140);
      TEXPANDS(v139, v25);
      // pto: %gp_bias_inline240__tile
      ;
      Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v141 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v23);
      // pto: %gp_bias_inline240__tile
      ;
      uint64_t v142 = (uint64_t) v15;
      TASSIGN(v141, v142);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v141, v139, v78);
      // pto: %gp_biased_inline241__tile
      ;
      Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v143 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v23, v23);
      // pto: %gp_biased_inline241__tile
      ;
      uint64_t v144 = (uint64_t) v16;
      TASSIGN(v143, v144);
      pipe_barrier(PIPE_V);
      TADD(v143, v131, v141);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      // pto: %biased_scores_buf_inline174__rv_v2_pview
      ;
      __gm__ float* v145 = PTOAS__GLOBAL_TENSOR_DATA(v64);
      // pto: %biased_scores_buf_inline174__rv_v2_pview
      ;
      const int64_t v146 = 0;
      // pto: %biased_scores_buf_inline174__rv_v2_pview
      ;
      const int64_t v147 = 256;
      // pto: %biased_scores_buf_inline174__rv_v2_pview
      ;
      pto::Shape<1, 1, 1, 16, 16> v148 = pto::Shape<1, 1, 1, 16, 16>();
      // pto: %biased_scores_buf_inline174__rv_v2_pview
      ;
      pto::Stride<4096, 4096, 4096, 256, 1> v149 = pto::Stride<4096, 4096, 4096, 256, 1>();
      // pto: %biased_scores_buf_inline174__rv_v2_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND> v150 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND>(v145 + (v146 + v82 * v147 + v72), v148, v149);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      TSTORE(v150, v143);
    };
  } else {
    // pto: %12
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v151 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v19, v21);
    // pto: %12
    ;
    uint64_t v152 = (uint64_t) v14;
    TASSIGN(v151, v152);
    // pto: %13
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v153 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v21);
    // pto: %13
    ;
    uint64_t v154 = (uint64_t) v14;
    TASSIGN(v153, v154);
    // pto: %14
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v155 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v21);
    // pto: %14
    ;
    uint64_t v156 = (uint64_t) v15;
    TASSIGN(v155, v156);
    // pto: %50
    ;
    const int32_t v157 = 16;
    // pto: %50
    ;
    const int32_t v158 = 16;
    // pto: %50
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v159 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v157, v158);
    v159.SetValidShape(v21, v21);
    TPOP<TPipe<0, Direction::DIR_C2V, 1024, 2, 2, true>, Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_NO_SPLIT>(v66, v159);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    // pto: %15
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v160 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v21);
    // pto: %15
    ;
    uint64_t v161 = (uint64_t) v16;
    TASSIGN(v160, v161);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TROWEXPANDMUL(v160, v159, v155);
    TFREE<TPipe<0, Direction::DIR_C2V, 1024, 2, 2, true>, TileSplitAxis::TILE_NO_SPLIT>(v66);
    // pto: %16
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v162 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v21);
    // pto: %16
    ;
    uint64_t v163 = (uint64_t) v15;
    TASSIGN(v162, v163);
    pipe_barrier(PIPE_V);
    TMAXS(v162, v160, v24);
    // pto: %17
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v164 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v21);
    // pto: %17
    ;
    uint64_t v165 = (uint64_t) v17;
    TASSIGN(v164, v165);
    TNEG(v164, v160);
    // pto: %18
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v166 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v21);
    // pto: %18
    ;
    uint64_t v167 = (uint64_t) v17;
    TASSIGN(v166, v167);
    pipe_barrier(PIPE_V);
    TMAX(v166, v160, v164);
    // pto: %19
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v168 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v21);
    // pto: %19
    ;
    uint64_t v169 = (uint64_t) v17;
    TASSIGN(v168, v169);
    pipe_barrier(PIPE_V);
    TNEG(v168, v166);
    // pto: %20
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v170 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v21);
    // pto: %20
    ;
    uint64_t v171 = (uint64_t) v17;
    TASSIGN(v170, v171);
    pipe_barrier(PIPE_V);
    TEXP(v170, v168);
    // pto: %21
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v172 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v21);
    // pto: %21
    ;
    uint64_t v173 = (uint64_t) v17;
    TASSIGN(v172, v173);
    pipe_barrier(PIPE_V);
    TADDS(v172, v170, v25);
    // pto: %22
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v174 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v21);
    // pto: %22
    ;
    uint64_t v175 = (uint64_t) v17;
    TASSIGN(v174, v175);
    pipe_barrier(PIPE_V);
    TLOG(v174, v172);
    // pto: %23
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v176 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v21);
    // pto: %23
    ;
    uint64_t v177 = (uint64_t) v15;
    TASSIGN(v176, v177);
    pipe_barrier(PIPE_V);
    TADD(v176, v162, v174);
    // pto: %24
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v178 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v21);
    // pto: %24
    ;
    uint64_t v179 = (uint64_t) v17;
    TASSIGN(v178, v179);
    pipe_barrier(PIPE_V);
    TNEG(v178, v160);
    // pto: %25
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v180 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v21);
    // pto: %25
    ;
    uint64_t v181 = (uint64_t) v17;
    TASSIGN(v180, v181);
    pipe_barrier(PIPE_V);
    TSUBS(v180, v178, v26);
    // pto: %26
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v182 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v21);
    // pto: %26
    ;
    uint64_t v183 = (uint64_t) v17;
    TASSIGN(v182, v183);
    pipe_barrier(PIPE_V);
    TMAXS(v182, v180, v24);
    // pto: %27
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v184 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v21);
    // pto: %27
    ;
    uint64_t v185 = (uint64_t) v17;
    TASSIGN(v184, v185);
    pipe_barrier(PIPE_V);
    TMINS(v184, v182, v25);
    // pto: %28
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v186 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v21);
    // pto: %28
    ;
    uint64_t v187 = (uint64_t) v16;
    TASSIGN(v186, v187);
    TMINS(v186, v160, v24);
    // pto: %29
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v188 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v21);
    // pto: %29
    ;
    uint64_t v189 = (uint64_t) v16;
    TASSIGN(v188, v189);
    pipe_barrier(PIPE_V);
    TEXP(v188, v186);
    // pto: %30
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v190 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v21);
    // pto: %30
    ;
    uint64_t v191 = (uint64_t) v16;
    TASSIGN(v190, v191);
    pipe_barrier(PIPE_V);
    TMUL(v190, v184, v188);
    // pto: %31
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v192 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v21);
    // pto: %31
    ;
    uint64_t v193 = (uint64_t) v16;
    TASSIGN(v192, v193);
    pipe_barrier(PIPE_V);
    TMAX(v192, v176, v190);
    // pto: %32
    ;
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v194 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v21);
    // pto: %32
    ;
    uint64_t v195 = (uint64_t) v16;
    TASSIGN(v194, v195);
    pipe_barrier(PIPE_V);
    TSQRT(v194, v192);
    // pto: %51, %52
    ;
    if ((int64_t) v8 >= v27) {
      // pto: %33
      ;
      Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v196 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v21);
      // pto: %33
      ;
      uint64_t v197 = (uint64_t) v15;
      TASSIGN(v196, v197);
      TEXPANDS(v196, v25);
      // pto: %34
      ;
      Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v198 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v21);
      // pto: %34
      ;
      uint64_t v199 = (uint64_t) v15;
      TASSIGN(v198, v199);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v198, v196, v153);
      // pto: %35
      ;
      Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v200 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v21);
      // pto: %35
      ;
      uint64_t v201 = (uint64_t) v16;
      TASSIGN(v200, v201);
      pipe_barrier(PIPE_V);
      TADD(v200, v194, v198);
    };
  }
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}