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

AICORE void staged_swa_proj_a_mm(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ float* v3, int64_t v4, int64_t v5, int64_t v6, int64_t v7, int64_t v8, int32_t v9, int32_t v10) {
  using T = float;

  #if defined(__DAV_CUBE__)
  // pto: %c0_i64
  const int64_t v11 = 0;
  // pto: %c131072_i64
  const int64_t v12 = 131072;
  // pto: %c196608_i64
  const int64_t v13 = 196608;
  // pto: %c327680_i64
  const int64_t v14 = 327680;
  // pto: %c32768_i64
  const int64_t v15 = 32768;
  // pto: %c49152_i64
  const int64_t v16 = 49152;
  // pto: %c16384_i64
  const int64_t v17 = 16384;
  // pto: %c4096_index
  const int64_t v18 = 4096;
  // pto: %c1_index
  const int64_t v19 = 1;
  // pto: %c8_index
  const int64_t v20 = 8;
  // pto: %c256_index
  const int64_t v21 = 256;
  // pto: %c64_index
  const int64_t v22 = 64;
  // pto: %c16_index
  const int64_t v23 = 16;
  // pto: %c4194304_index
  const int64_t v24 = 4194304;
  // pto: %c16384_index
  const int64_t v25 = 16384;
  // pto: %c8192_index
  const int64_t v26 = 8192;
  // pto: %c128_index
  const int64_t v27 = 128;
  // pto: %c0_index
  const int64_t v28 = 0;
  // pto: %c2_index
  const int64_t v29 = 2;
  // pto: %cn64_index
  const int64_t v30 = -64;
  // pto: %cn1_index
  const int64_t v31 = -1;
  // pto: %o_packed_inline654__ssa_v0_view
  const int64_t v32 = 1;
  // pto: %o_packed_inline654__ssa_v0_view
  const int64_t v33 = 1;
  // pto: %o_packed_inline654__ssa_v0_view
  const int64_t v34 = 1;
  // pto: %o_packed_inline654__ssa_v0_view
  int64_t v35 = (int64_t) v7;
  // pto: %o_packed_inline654__ssa_v0_view
  int64_t v36 = v35 * v18;
  // pto: %o_packed_inline654__ssa_v0_view
  int64_t v37 = v34 * v36;
  // pto: %o_packed_inline654__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v38 = pto::Shape<1, 1, 1, -1, -1>(v32, v33, v34, v35, v18);
  // pto: %o_packed_inline654__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v39 = pto::Stride<-1, -1, -1, -1, -1>(v33 * v37, v37, v36, v18, v19);
  // pto: %o_packed_inline654__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v40 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v38, v39);
  // pto: %wo_a__ssa_v0_view
  pto::Shape<-1, -1, -1, -1, -1> v41 = pto::Shape<-1, -1, -1, -1, -1>(v20, v21, v22, v23, v23);
  // pto: %wo_a__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v42 = pto::Stride<-1, -1, -1, -1, -1>(v24, v25, v21, v23, v19);
  // pto: %wo_a__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<-1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::NZ> v43 = GlobalTensor<bfloat16_t, pto::Shape<-1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::NZ>(v2, v41, v42);
  // pto: %o_r_inline655__iter_v1_view
  const int64_t v44 = 1;
  // pto: %o_r_inline655__iter_v1_view
  const int64_t v45 = 1;
  // pto: %o_r_inline655__iter_v1_view
  const int64_t v46 = 1;
  // pto: %o_r_inline655__iter_v1_view
  int64_t v47 = (int64_t) v8;
  // pto: %o_r_inline655__iter_v1_view
  int64_t v48 = v47 * v26;
  // pto: %o_r_inline655__iter_v1_view
  int64_t v49 = v46 * v48;
  // pto: %o_r_inline655__iter_v1_view
  pto::Shape<1, 1, 1, -1, -1> v50 = pto::Shape<1, 1, 1, -1, -1>(v44, v45, v46, v47, v26);
  // pto: %o_r_inline655__iter_v1_view
  pto::Stride<-1, -1, -1, -1, -1> v51 = pto::Stride<-1, -1, -1, -1, -1>(v45 * v49, v49, v48, v26, v19);
  // pto: %o_r_inline655__iter_v1_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v52 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v50, v51);
  // pto: %pa_blk_inline666__ssa_v0
  int64_t v53 = (int64_t) v9;
  // pto: %12, %13
  int64_t v54 = (int64_t) ((uint64_t) (v53 % v20) * (uint64_t) v27);
  // pto: %14, %15
  int64_t v55 = (int64_t) ((uint64_t) (v53 / v20) * (uint64_t) v21);
  // pto: %16
  int64_t v56 = (int64_t) ((uint64_t) v4 + (uint64_t) v55);
  // pto: %acc_a_inline672__tile
  Tile<TileType::Acc, float, 256, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v57 = Tile<TileType::Acc, float, 256, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v21, v27);
  // pto: %acc_a_inline672__tile
  uint64_t v58 = (uint64_t) v11;
  TASSIGN(v57, v58);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
  for (int64_t v59 = v28; v59 < v23; v59 += v29) {
    // pto: %17
    ;
    int64_t v60 = (int64_t) ((uint64_t) v59 * (uint64_t) v21);
    // pto: %19
    ;
    int64_t v61 = (int64_t) ((uint64_t) v60 + (uint64_t) v21);
    // pto: %xa_k_chunk_inline675__tile
    ;
    Tile<TileType::Mat, bfloat16_t, 256, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v62 = Tile<TileType::Mat, bfloat16_t, 256, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v21, v21);
    // pto: %xa_k_chunk_inline675__tile
    ;
    uint64_t v63 = (uint64_t) v11;
    TASSIGN(v62, v63);
    // pto: %20
    ;
    int64_t v64 = v56 < v28 ? v28 : v56;
    // pto: %21
    ;
    int64_t v65 = v60 < v28 ? v28 : v60;
    // pto: %o_packed_inline654__ssa_v0_pview
    ;
    __gm__ bfloat16_t* v66 = PTOAS__GLOBAL_TENSOR_DATA(v40);
    // pto: %o_packed_inline654__ssa_v0_pview
    ;
    const int64_t v67 = 0;
    // pto: %o_packed_inline654__ssa_v0_pview
    ;
    const int64_t v68 = 4096;
    // pto: %o_packed_inline654__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 256, 256> v69 = pto::Shape<1, 1, 1, 256, 256>();
    // pto: %o_packed_inline654__ssa_v0_pview
    ;
    pto::Stride<1048576, 1048576, 1048576, 4096, 1> v70 = pto::Stride<1048576, 1048576, 1048576, 4096, 1>();
    // pto: %o_packed_inline654__ssa_v0_pview
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<1048576, 1048576, 1048576, 4096, 1>, pto::Layout::ND> v71 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<1048576, 1048576, 1048576, 4096, 1>, pto::Layout::ND>(v66 + (v67 + v64 * v68 + v65), v69, v70);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    TLOAD(v62, v71);
    // pto: %wa_k_chunk_inline676__tile
    ;
    Tile<TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v72 = Tile<TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v27, v21);
    // pto: %wa_k_chunk_inline676__tile
    ;
    uint64_t v73 = (uint64_t) v12;
    TASSIGN(v72, v73);
    // pto: %22
    ;
    int64_t v74 = v5 < v28 ? v28 : v5;
    // pto: %23
    ;
    int64_t v75 = v60 / v23;
    // pto: %24
    ;
    int64_t v76 = v75 < v28 ? v28 : v75;
    // pto: %25
    ;
    int64_t v77 = v54 / v23;
    // pto: %26
    ;
    int64_t v78 = v77 < v28 ? v28 : v77;
    // pto: %wo_a__ssa_v0_pview
    ;
    __gm__ bfloat16_t* v79 = PTOAS__GLOBAL_TENSOR_DATA(v43);
    // pto: %wo_a__ssa_v0_pview
    ;
    const int64_t v80 = 0;
    // pto: %wo_a__ssa_v0_pview
    ;
    const int64_t v81 = 4194304;
    // pto: %wo_a__ssa_v0_pview
    ;
    const int64_t v82 = 16384;
    // pto: %wo_a__ssa_v0_pview
    ;
    const int64_t v83 = 256;
    // pto: %wo_a__ssa_v0_pview
    ;
    pto::Shape<1, 16, 8, 16, 16> v84 = pto::Shape<1, 16, 8, 16, 16>();
    // pto: %wo_a__ssa_v0_pview
    ;
    pto::Stride<4194304, 16384, 256, 16, 1> v85 = pto::Stride<4194304, 16384, 256, 16, 1>();
    // pto: %wo_a__ssa_v0_pview
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 16, 8, 16, 16>, pto::Stride<4194304, 16384, 256, 16, 1>, pto::Layout::NZ> v86 = GlobalTensor<bfloat16_t, pto::Shape<1, 16, 8, 16, 16>, pto::Stride<4194304, 16384, 256, 16, 1>, pto::Layout::NZ>(v79 + ((v80 + v74 * v81) + v76 * v82 + v78 * v83), v84, v85);
    TLOAD(v72, v86);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    // pto: %0
    ;
    Tile<TileType::Mat, bfloat16_t, 256, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v87 = Tile<TileType::Mat, bfloat16_t, 256, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v21, v21);
    // pto: %0
    ;
    uint64_t v88 = (uint64_t) v13;
    TASSIGN(v87, v88);
    // pto: %28
    ;
    int64_t v89 = v61 < v28 ? v28 : v61;
    // pto: %29
    ;
    __gm__ bfloat16_t* v90 = PTOAS__GLOBAL_TENSOR_DATA(v40);
    // pto: %29
    ;
    const int64_t v91 = 0;
    // pto: %29
    ;
    const int64_t v92 = 4096;
    // pto: %29
    ;
    pto::Shape<1, 1, 1, 256, 256> v93 = pto::Shape<1, 1, 1, 256, 256>();
    // pto: %29
    ;
    pto::Stride<1048576, 1048576, 1048576, 4096, 1> v94 = pto::Stride<1048576, 1048576, 1048576, 4096, 1>();
    // pto: %29
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<1048576, 1048576, 1048576, 4096, 1>, pto::Layout::ND> v95 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<1048576, 1048576, 1048576, 4096, 1>, pto::Layout::ND>(v90 + (v91 + v64 * v92 + v89), v93, v94);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    TLOAD(v87, v95);
    // pto: %1
    ;
    Tile<TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v96 = Tile<TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v27, v21);
    // pto: %1
    ;
    uint64_t v97 = (uint64_t) v14;
    TASSIGN(v96, v97);
    // pto: %31
    ;
    int64_t v98 = v61 / v23;
    // pto: %32
    ;
    int64_t v99 = v98 < v28 ? v28 : v98;
    // pto: %35
    ;
    __gm__ bfloat16_t* v100 = PTOAS__GLOBAL_TENSOR_DATA(v43);
    // pto: %35
    ;
    const int64_t v101 = 0;
    // pto: %35
    ;
    const int64_t v102 = 4194304;
    // pto: %35
    ;
    const int64_t v103 = 16384;
    // pto: %35
    ;
    const int64_t v104 = 256;
    // pto: %35
    ;
    pto::Shape<1, 16, 8, 16, 16> v105 = pto::Shape<1, 16, 8, 16, 16>();
    // pto: %35
    ;
    pto::Stride<4194304, 16384, 256, 16, 1> v106 = pto::Stride<4194304, 16384, 256, 16, 1>();
    // pto: %35
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 16, 8, 16, 16>, pto::Stride<4194304, 16384, 256, 16, 1>, pto::Layout::NZ> v107 = GlobalTensor<bfloat16_t, pto::Shape<1, 16, 8, 16, 16>, pto::Stride<4194304, 16384, 256, 16, 1>, pto::Layout::NZ>(v100 + ((v101 + v74 * v102) + v99 * v103 + v78 * v104), v105, v106);
    TLOAD(v96, v107);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    // pto: %wa_k_chunk_inline676__tile_t
    ;
    Tile<TileType::Mat, bfloat16_t, 256, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v108 = Tile<TileType::Mat, bfloat16_t, 256, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v21, v27);
    // pto: %wa_k_chunk_inline676__tile_t
    ;
    uint64_t v109 = (uint64_t) v12;
    TASSIGN(v108, v109);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    for (int64_t v110 = v28; v110 < v21; v110 += v27) {
      // pto: %acc_a_inline672__iter_v1_l0_a
      ;
      Tile<TileType::Left, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v111 = Tile<TileType::Left, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v21, v22);
      // pto: %acc_a_inline672__iter_v1_l0_a
      ;
      uint64_t v112 = (uint64_t) v15;
      TASSIGN(v111, v112);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      pipe_barrier(PIPE_MTE1);
      uint16_t v113 = static_cast<uint16_t>(v28);
      uint16_t v114 = static_cast<uint16_t>(v110);
      TEXTRACT(v111, v62, v113, v114);
      // pto: %acc_a_inline672__iter_v1_l0_b
      ;
      Tile<TileType::Right, bfloat16_t, 64, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v115 = Tile<TileType::Right, bfloat16_t, 64, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v22, v27);
      // pto: %acc_a_inline672__iter_v1_l0_b
      ;
      uint64_t v116 = (uint64_t) v16;
      TASSIGN(v115, v116);
      uint16_t v117 = static_cast<uint16_t>(v110);
      uint16_t v118 = static_cast<uint16_t>(v28);
      TEXTRACT(v115, v108, v117, v118);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      // pto: %2
      ;
      Tile<TileType::Left, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v119 = Tile<TileType::Left, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v21, v22);
      // pto: %2
      ;
      uint64_t v120 = (uint64_t) v11;
      TASSIGN(v119, v120);
      // pto: %36
      ;
      int64_t v121 = (int64_t) ((uint64_t) v110 + (uint64_t) v22);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
      uint16_t v122 = static_cast<uint16_t>(v28);
      uint16_t v123 = static_cast<uint16_t>(v121);
      TEXTRACT(v119, v62, v122, v123);
      // pto: %3
      ;
      Tile<TileType::Right, bfloat16_t, 64, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v124 = Tile<TileType::Right, bfloat16_t, 64, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v22, v27);
      // pto: %3
      ;
      uint64_t v125 = (uint64_t) v11;
      TASSIGN(v124, v125);
      uint16_t v126 = static_cast<uint16_t>(v121);
      uint16_t v127 = static_cast<uint16_t>(v28);
      TEXTRACT(v124, v108, v126, v127);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      // pto: %38
      ;
      bool v128 = v59 == v28;
      // pto: %39, %40
      ;
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v128 & v110 == v28) {
        pipe_barrier(PIPE_M);
        TMATMUL(v57, v111, v115);
      } else {
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v57, v57, v111, v115);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      // pto: %42, %43
      ;
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      if (v128 & v110 == v30) {
        pipe_barrier(PIPE_M);
        TMATMUL(v57, v119, v124);
      } else {
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v57, v57, v119, v124);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    };
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    // pto: %5
    ;
    Tile<TileType::Mat, bfloat16_t, 256, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v129 = Tile<TileType::Mat, bfloat16_t, 256, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v21, v27);
    // pto: %5
    ;
    uint64_t v130 = (uint64_t) v14;
    TASSIGN(v129, v130);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    for (int64_t v131 = v28; v131 < v21; v131 += v27) {
      // pto: %6
      ;
      Tile<TileType::Left, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v132 = Tile<TileType::Left, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v21, v22);
      // pto: %6
      ;
      uint64_t v133 = (uint64_t) v15;
      TASSIGN(v132, v133);
      pipe_barrier(PIPE_MTE1);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
      uint16_t v134 = static_cast<uint16_t>(v28);
      uint16_t v135 = static_cast<uint16_t>(v131);
      TEXTRACT(v132, v87, v134, v135);
      // pto: %7
      ;
      Tile<TileType::Right, bfloat16_t, 64, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v136 = Tile<TileType::Right, bfloat16_t, 64, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v22, v27);
      // pto: %7
      ;
      uint64_t v137 = (uint64_t) v17;
      TASSIGN(v136, v137);
      uint16_t v138 = static_cast<uint16_t>(v131);
      uint16_t v139 = static_cast<uint16_t>(v28);
      TEXTRACT(v136, v129, v138, v139);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
      // pto: %8
      ;
      Tile<TileType::Left, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v140 = Tile<TileType::Left, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v21, v22);
      // pto: %8
      ;
      uint64_t v141 = (uint64_t) v11;
      TASSIGN(v140, v141);
      // pto: %45
      ;
      int64_t v142 = (int64_t) ((uint64_t) v131 + (uint64_t) v22);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
      uint16_t v143 = static_cast<uint16_t>(v28);
      uint16_t v144 = static_cast<uint16_t>(v142);
      TEXTRACT(v140, v87, v143, v144);
      // pto: %9
      ;
      Tile<TileType::Right, bfloat16_t, 64, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v145 = Tile<TileType::Right, bfloat16_t, 64, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v22, v27);
      // pto: %9
      ;
      uint64_t v146 = (uint64_t) v15;
      TASSIGN(v145, v146);
      uint16_t v147 = static_cast<uint16_t>(v142);
      uint16_t v148 = static_cast<uint16_t>(v28);
      TEXTRACT(v145, v129, v147, v148);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
      // pto: %47
      ;
      bool v149 = v59 == v31;
      // pto: %48, %49
      ;
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
      if (v149 & v131 == v28) {
        pipe_barrier(PIPE_M);
        TMATMUL(v57, v132, v136);
      } else {
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v57, v57, v132, v136);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
      // pto: %51, %52
      ;
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
      if (v149 & v131 == v30) {
        pipe_barrier(PIPE_M);
        TMATMUL(v57, v140, v145);
      } else {
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v57, v57, v140, v145);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
    };
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  }
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  // pto: %53
  int64_t v150 = v55 < v28 ? v28 : v55;
  // pto: %54
  int64_t v151 = (int64_t) ((uint64_t) v6 + (uint64_t) v54);
  // pto: %55
  int64_t v152 = v151 < v28 ? v28 : v151;
  // pto: %o_r_inline655__iter_v1_pview
  __gm__ float* v153 = PTOAS__GLOBAL_TENSOR_DATA(v52);
  // pto: %o_r_inline655__iter_v1_pview
  const int64_t v154 = 0;
  // pto: %o_r_inline655__iter_v1_pview
  const int64_t v155 = 8192;
  // pto: %o_r_inline655__iter_v1_pview
  pto::Shape<1, 1, 1, 256, 128> v156 = pto::Shape<1, 1, 1, 256, 128>();
  // pto: %o_r_inline655__iter_v1_pview
  pto::Stride<2097152, 2097152, 2097152, 8192, 1> v157 = pto::Stride<2097152, 2097152, 2097152, 8192, 1>();
  // pto: %o_r_inline655__iter_v1_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 256, 128>, pto::Stride<2097152, 2097152, 2097152, 8192, 1>, pto::Layout::ND> v158 = GlobalTensor<float, pto::Shape<1, 1, 1, 256, 128>, pto::Stride<2097152, 2097152, 2097152, 8192, 1>, pto::Layout::ND>(v153 + (v154 + v150 * v155 + v152), v156, v157);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v158, v57);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
  #endif // __DAV_CUBE__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}