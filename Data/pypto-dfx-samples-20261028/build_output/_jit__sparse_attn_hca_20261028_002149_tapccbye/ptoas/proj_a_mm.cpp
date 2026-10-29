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

AICORE void proj_a_mm(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ float* v3, int64_t v4, int64_t v5, int64_t v6, int32_t v7, int32_t v8) {
  using T = float;

  #if defined(__DAV_CUBE__)
  // pto: %c0_i64
  const int64_t v9 = 0;
  // pto: %c8192_i64
  const int64_t v10 = 8192;
  // pto: %c73728_i64
  const int64_t v11 = 73728;
  // pto: %c81920_i64
  const int64_t v12 = 81920;
  // pto: %c12288_i64
  const int64_t v13 = 12288;
  // pto: %c32768_i64
  const int64_t v14 = 32768;
  // pto: %c4096_i64
  const int64_t v15 = 4096;
  // pto: %c64_index
  const int64_t v16 = 64;
  // pto: %c4096_index
  const int64_t v17 = 4096;
  // pto: %c1_index
  const int64_t v18 = 1;
  // pto: %c8_index
  const int64_t v19 = 8;
  // pto: %c256_index
  const int64_t v20 = 256;
  // pto: %c16_index
  const int64_t v21 = 16;
  // pto: %c4194304_index
  const int64_t v22 = 4194304;
  // pto: %c16384_index
  const int64_t v23 = 16384;
  // pto: %c8192_index
  const int64_t v24 = 8192;
  // pto: %c128_index
  const int64_t v25 = 128;
  // pto: %c0_index
  const int64_t v26 = 0;
  // pto: %c2_index
  const int64_t v27 = 2;
  // pto: %cn1_index
  const int64_t v28 = -1;
  // pto: %o_packed__ssa_v16_view
  const int64_t v29 = 1;
  // pto: %o_packed__ssa_v16_view
  const int64_t v30 = 1;
  // pto: %o_packed__ssa_v16_view
  const int64_t v31 = 1;
  // pto: %o_packed__ssa_v16_view
  int64_t v32 = v16 * v17;
  // pto: %o_packed__ssa_v16_view
  int64_t v33 = v31 * v32;
  // pto: %o_packed__ssa_v16_view
  pto::Shape<1, 1, 1, -1, -1> v34 = pto::Shape<1, 1, 1, -1, -1>(v29, v30, v31, v16, v17);
  // pto: %o_packed__ssa_v16_view
  pto::Stride<-1, -1, -1, -1, -1> v35 = pto::Stride<-1, -1, -1, -1, -1>(v30 * v33, v33, v32, v17, v18);
  // pto: %o_packed__ssa_v16_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v36 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v34, v35);
  // pto: %wo_a__ssa_v0_view
  pto::Shape<-1, -1, -1, -1, -1> v37 = pto::Shape<-1, -1, -1, -1, -1>(v19, v20, v16, v21, v21);
  // pto: %wo_a__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v38 = pto::Stride<-1, -1, -1, -1, -1>(v22, v23, v20, v21, v18);
  // pto: %wo_a__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<-1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::NZ> v39 = GlobalTensor<bfloat16_t, pto::Shape<-1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::NZ>(v2, v37, v38);
  // pto: %o_r_pad__iter_v1_view
  const int64_t v40 = 1;
  // pto: %o_r_pad__iter_v1_view
  const int64_t v41 = 1;
  // pto: %o_r_pad__iter_v1_view
  const int64_t v42 = 1;
  // pto: %o_r_pad__iter_v1_view
  int64_t v43 = v21 * v24;
  // pto: %o_r_pad__iter_v1_view
  int64_t v44 = v42 * v43;
  // pto: %o_r_pad__iter_v1_view
  pto::Shape<1, 1, 1, -1, -1> v45 = pto::Shape<1, 1, 1, -1, -1>(v40, v41, v42, v21, v24);
  // pto: %o_r_pad__iter_v1_view
  pto::Stride<-1, -1, -1, -1, -1> v46 = pto::Stride<-1, -1, -1, -1, -1>(v41 * v44, v44, v43, v24, v18);
  // pto: %o_r_pad__iter_v1_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v47 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v45, v46);
  // pto: %nf__ssa_v0, %12
  int64_t v48 = (int64_t) ((uint64_t) ((int64_t) v7) * (uint64_t) v25);
  // pto: %acc_a__tile
  Tile<TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v49 = Tile<TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v21, v25);
  // pto: %acc_a__tile
  uint64_t v50 = (uint64_t) v9;
  TASSIGN(v49, v50);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  for (int64_t v51 = v26; v51 < v21; v51 += v27) {
    // pto: %13
    ;
    int64_t v52 = (int64_t) ((uint64_t) v51 * (uint64_t) v20);
    // pto: %15
    ;
    int64_t v53 = (int64_t) ((uint64_t) v52 + (uint64_t) v20);
    // pto: %xa_k_chunk__tile
    ;
    Tile<TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v54 = Tile<TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v19, v20);
    // pto: %xa_k_chunk__tile
    ;
    uint64_t v55 = (uint64_t) v9;
    TASSIGN(v54, v55);
    // pto: %16
    ;
    int64_t v56 = v4 < v26 ? v26 : v4;
    // pto: %17
    ;
    int64_t v57 = v52 < v26 ? v26 : v52;
    // pto: %o_packed__ssa_v16_pview
    ;
    __gm__ bfloat16_t* v58 = PTOAS__GLOBAL_TENSOR_DATA(v36);
    // pto: %o_packed__ssa_v16_pview
    ;
    const int64_t v59 = 0;
    // pto: %o_packed__ssa_v16_pview
    ;
    const int64_t v60 = 4096;
    // pto: %o_packed__ssa_v16_pview
    ;
    pto::Shape<1, 1, 1, 8, 256> v61 = pto::Shape<1, 1, 1, 8, 256>();
    // pto: %o_packed__ssa_v16_pview
    ;
    pto::Stride<32768, 32768, 32768, 4096, 1> v62 = pto::Stride<32768, 32768, 32768, 4096, 1>();
    // pto: %o_packed__ssa_v16_pview
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND> v63 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>(v58 + (v59 + v56 * v60 + v57), v61, v62);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    TLOAD(v54, v63);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    // pto: %wa_k_chunk__tile
    ;
    Tile<TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v64 = Tile<TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v25, v20);
    // pto: %wa_k_chunk__tile
    ;
    uint64_t v65 = (uint64_t) v10;
    TASSIGN(v64, v65);
    // pto: %18
    ;
    int64_t v66 = v5 < v26 ? v26 : v5;
    // pto: %19
    ;
    int64_t v67 = v52 / v21;
    // pto: %20
    ;
    int64_t v68 = v67 < v26 ? v26 : v67;
    // pto: %21
    ;
    int64_t v69 = v48 / v21;
    // pto: %22
    ;
    int64_t v70 = v69 < v26 ? v26 : v69;
    // pto: %wo_a__ssa_v0_pview
    ;
    __gm__ bfloat16_t* v71 = PTOAS__GLOBAL_TENSOR_DATA(v39);
    // pto: %wo_a__ssa_v0_pview
    ;
    const int64_t v72 = 0;
    // pto: %wo_a__ssa_v0_pview
    ;
    const int64_t v73 = 4194304;
    // pto: %wo_a__ssa_v0_pview
    ;
    const int64_t v74 = 16384;
    // pto: %wo_a__ssa_v0_pview
    ;
    const int64_t v75 = 256;
    // pto: %wo_a__ssa_v0_pview
    ;
    pto::Shape<1, 16, 8, 16, 16> v76 = pto::Shape<1, 16, 8, 16, 16>();
    // pto: %wo_a__ssa_v0_pview
    ;
    pto::Stride<4194304, 16384, 256, 16, 1> v77 = pto::Stride<4194304, 16384, 256, 16, 1>();
    // pto: %wo_a__ssa_v0_pview
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 16, 8, 16, 16>, pto::Stride<4194304, 16384, 256, 16, 1>, pto::Layout::NZ> v78 = GlobalTensor<bfloat16_t, pto::Shape<1, 16, 8, 16, 16>, pto::Stride<4194304, 16384, 256, 16, 1>, pto::Layout::NZ>(v71 + ((v72 + v66 * v73) + v68 * v74 + v70 * v75), v76, v77);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    TLOAD(v64, v78);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    // pto: %0
    ;
    Tile<TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v79 = Tile<TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v19, v20);
    // pto: %0
    ;
    uint64_t v80 = (uint64_t) v11;
    TASSIGN(v79, v80);
    // pto: %24
    ;
    int64_t v81 = v53 < v26 ? v26 : v53;
    // pto: %25
    ;
    __gm__ bfloat16_t* v82 = PTOAS__GLOBAL_TENSOR_DATA(v36);
    // pto: %25
    ;
    const int64_t v83 = 0;
    // pto: %25
    ;
    const int64_t v84 = 4096;
    // pto: %25
    ;
    pto::Shape<1, 1, 1, 8, 256> v85 = pto::Shape<1, 1, 1, 8, 256>();
    // pto: %25
    ;
    pto::Stride<32768, 32768, 32768, 4096, 1> v86 = pto::Stride<32768, 32768, 32768, 4096, 1>();
    // pto: %25
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND> v87 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>(v82 + (v83 + v56 * v84 + v81), v85, v86);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    TLOAD(v79, v87);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
    // pto: %1
    ;
    Tile<TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v88 = Tile<TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v25, v20);
    // pto: %1
    ;
    uint64_t v89 = (uint64_t) v12;
    TASSIGN(v88, v89);
    // pto: %27
    ;
    int64_t v90 = v53 / v21;
    // pto: %28
    ;
    int64_t v91 = v90 < v26 ? v26 : v90;
    // pto: %31
    ;
    __gm__ bfloat16_t* v92 = PTOAS__GLOBAL_TENSOR_DATA(v39);
    // pto: %31
    ;
    const int64_t v93 = 0;
    // pto: %31
    ;
    const int64_t v94 = 4194304;
    // pto: %31
    ;
    const int64_t v95 = 16384;
    // pto: %31
    ;
    const int64_t v96 = 256;
    // pto: %31
    ;
    pto::Shape<1, 16, 8, 16, 16> v97 = pto::Shape<1, 16, 8, 16, 16>();
    // pto: %31
    ;
    pto::Stride<4194304, 16384, 256, 16, 1> v98 = pto::Stride<4194304, 16384, 256, 16, 1>();
    // pto: %31
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 16, 8, 16, 16>, pto::Stride<4194304, 16384, 256, 16, 1>, pto::Layout::NZ> v99 = GlobalTensor<bfloat16_t, pto::Shape<1, 16, 8, 16, 16>, pto::Stride<4194304, 16384, 256, 16, 1>, pto::Layout::NZ>(v92 + ((v93 + v66 * v94) + v91 * v95 + v70 * v96), v97, v98);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    TLOAD(v88, v99);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
    // pto: %wa_k_chunk__tile_t
    ;
    Tile<TileType::Mat, bfloat16_t, 256, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v100 = Tile<TileType::Mat, bfloat16_t, 256, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v20, v25);
    // pto: %wa_k_chunk__tile_t
    ;
    uint64_t v101 = (uint64_t) v10;
    TASSIGN(v100, v101);
    // pto: %acc_a__iter_v1_l0_a
    ;
    Tile<TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v102 = Tile<TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v19, v25);
    // pto: %acc_a__iter_v1_l0_a
    ;
    uint64_t v103 = (uint64_t) v13;
    TASSIGN(v102, v103);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    uint16_t v104 = static_cast<uint16_t>(v26);
    uint16_t v105 = static_cast<uint16_t>(v26);
    TEXTRACT(v102, v54, v104, v105);
    // pto: %acc_a__iter_v1_l0_b
    ;
    Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v106 = Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v25, v25);
    // pto: %acc_a__iter_v1_l0_b
    ;
    uint64_t v107 = (uint64_t) v14;
    TASSIGN(v106, v107);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    pipe_barrier(PIPE_MTE1);
    uint16_t v108 = static_cast<uint16_t>(v26);
    uint16_t v109 = static_cast<uint16_t>(v26);
    TEXTRACT(v106, v100, v108, v109);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    // pto: %2
    ;
    Tile<TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v110 = Tile<TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v19, v25);
    // pto: %2
    ;
    uint64_t v111 = (uint64_t) v9;
    TASSIGN(v110, v111);
    uint16_t v112 = static_cast<uint16_t>(v26);
    uint16_t v113 = static_cast<uint16_t>(v25);
    TEXTRACT(v110, v54, v112, v113);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    // pto: %3
    ;
    Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v114 = Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v25, v25);
    // pto: %3
    ;
    uint64_t v115 = (uint64_t) v9;
    TASSIGN(v114, v115);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    uint16_t v116 = static_cast<uint16_t>(v25);
    uint16_t v117 = static_cast<uint16_t>(v26);
    TEXTRACT(v114, v100, v116, v117);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    // pto: %32
    ;
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    if (v51 == v26) {
      pipe_barrier(PIPE_M);
      TMATMUL(v49, v102, v106);
    } else {
      pipe_barrier(PIPE_M);
      TMATMUL_ACC(v49, v49, v102, v106);
    };
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    pipe_barrier(PIPE_M);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    TMATMUL_ACC(v49, v49, v110, v114);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    // pto: %5
    ;
    Tile<TileType::Mat, bfloat16_t, 256, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v118 = Tile<TileType::Mat, bfloat16_t, 256, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v20, v25);
    // pto: %5
    ;
    uint64_t v119 = (uint64_t) v12;
    TASSIGN(v118, v119);
    // pto: %6
    ;
    Tile<TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v120 = Tile<TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v19, v25);
    // pto: %6
    ;
    uint64_t v121 = (uint64_t) v15;
    TASSIGN(v120, v121);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
    uint16_t v122 = static_cast<uint16_t>(v26);
    uint16_t v123 = static_cast<uint16_t>(v26);
    TEXTRACT(v120, v79, v122, v123);
    // pto: %7
    ;
    Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v124 = Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v25, v25);
    // pto: %7
    ;
    uint64_t v125 = (uint64_t) v14;
    TASSIGN(v124, v125);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    pipe_barrier(PIPE_MTE1);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
    uint16_t v126 = static_cast<uint16_t>(v26);
    uint16_t v127 = static_cast<uint16_t>(v26);
    TEXTRACT(v124, v118, v126, v127);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
    // pto: %8
    ;
    Tile<TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v128 = Tile<TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v19, v25);
    // pto: %8
    ;
    uint64_t v129 = (uint64_t) v10;
    TASSIGN(v128, v129);
    uint16_t v130 = static_cast<uint16_t>(v26);
    uint16_t v131 = static_cast<uint16_t>(v25);
    TEXTRACT(v128, v79, v130, v131);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    // pto: %9
    ;
    Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v132 = Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v25, v25);
    // pto: %9
    ;
    uint64_t v133 = (uint64_t) v9;
    TASSIGN(v132, v133);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    uint16_t v134 = static_cast<uint16_t>(v25);
    uint16_t v135 = static_cast<uint16_t>(v26);
    TEXTRACT(v132, v118, v134, v135);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
    // pto: %33
    ;
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
    if (v51 == v28) {
      pipe_barrier(PIPE_M);
      TMATMUL(v49, v120, v124);
    } else {
      pipe_barrier(PIPE_M);
      TMATMUL_ACC(v49, v49, v120, v124);
    };
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    pipe_barrier(PIPE_M);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
    TMATMUL_ACC(v49, v49, v128, v132);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  }
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  // pto: %34
  int64_t v136 = (int64_t) ((uint64_t) v6 + (uint64_t) v48);
  // pto: %35
  int64_t v137 = v136 < v26 ? v26 : v136;
  // pto: %o_r_pad__iter_v1_pview
  __gm__ float* v138 = PTOAS__GLOBAL_TENSOR_DATA(v47);
  // pto: %o_r_pad__iter_v1_pview
  const int64_t v139 = 0;
  // pto: %o_r_pad__iter_v1_pview
  pto::Shape<1, 1, 1, 16, 128> v140 = pto::Shape<1, 1, 1, 16, 128>();
  // pto: %o_r_pad__iter_v1_pview
  pto::Stride<131072, 131072, 131072, 8192, 1> v141 = pto::Stride<131072, 131072, 131072, 8192, 1>();
  // pto: %o_r_pad__iter_v1_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 128>, pto::Stride<131072, 131072, 131072, 8192, 1>, pto::Layout::ND> v142 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 128>, pto::Stride<131072, 131072, 131072, 8192, 1>, pto::Layout::ND>(v138 + (v139 + v137), v140, v141);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v142, v49);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  #endif // __DAV_CUBE__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}