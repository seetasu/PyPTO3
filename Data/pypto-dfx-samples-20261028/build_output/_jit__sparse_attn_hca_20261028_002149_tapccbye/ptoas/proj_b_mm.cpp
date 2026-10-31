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

AICORE void proj_b_mm(__gm__ int32_t* v1, __gm__ int8_t* v2, __gm__ int8_t* v3, int64_t v4, int32_t v5, int32_t v6) {
  using T = float;

  #if defined(__DAV_CUBE__)
  // pto: %c0_i64
  const int64_t v7 = 0;
  // pto: %c4096_i64
  const int64_t v8 = 4096;
  // pto: %c69632_i64
  const int64_t v9 = 69632;
  // pto: %c73728_i64
  const int64_t v10 = 73728;
  // pto: %c6144_i64
  const int64_t v11 = 6144;
  // pto: %c32768_i64
  const int64_t v12 = 32768;
  // pto: %c2048_i64
  const int64_t v13 = 2048;
  // pto: %c16_index
  const int64_t v14 = 16;
  // pto: %c32768_index
  const int64_t v15 = 32768;
  // pto: %c1_index
  const int64_t v16 = 1;
  // pto: %c8192_index
  const int64_t v17 = 8192;
  // pto: %c8_index
  const int64_t v18 = 8;
  // pto: %c32_index
  const int64_t v19 = 32;
  // pto: %c256_index
  const int64_t v20 = 256;
  // pto: %c4194304_index
  const int64_t v21 = 4194304;
  // pto: %c131072_index
  const int64_t v22 = 131072;
  // pto: %c512_index
  const int64_t v23 = 512;
  // pto: %c0_index
  const int64_t v24 = 0;
  // pto: %c2_index
  const int64_t v25 = 2;
  // pto: %c4_index
  const int64_t v26 = 4;
  // pto: %c1024_index
  const int64_t v27 = 1024;
  // pto: %c128_index
  const int64_t v28 = 128;
  // pto: %cn1_index
  const int64_t v29 = -1;
  // pto: %c4096_index
  const int64_t v30 = 4096;
  // pto: %partials__iter_v1_view
  const int64_t v31 = 1;
  // pto: %partials__iter_v1_view
  const int64_t v32 = 1;
  // pto: %partials__iter_v1_view
  const int64_t v33 = 1;
  // pto: %partials__iter_v1_view
  int64_t v34 = v14 * v15;
  // pto: %partials__iter_v1_view
  int64_t v35 = v33 * v34;
  // pto: %partials__iter_v1_view
  pto::Shape<1, 1, 1, -1, -1> v36 = pto::Shape<1, 1, 1, -1, -1>(v31, v32, v33, v14, v15);
  // pto: %partials__iter_v1_view
  pto::Stride<-1, -1, -1, -1, -1> v37 = pto::Stride<-1, -1, -1, -1, -1>(v32 * v35, v35, v34, v15, v16);
  // pto: %partials__iter_v1_view
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v38 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v36, v37);
  // pto: %o_r_i8_pad__rv_v4_view
  const int64_t v39 = 1;
  // pto: %o_r_i8_pad__rv_v4_view
  const int64_t v40 = 1;
  // pto: %o_r_i8_pad__rv_v4_view
  const int64_t v41 = 1;
  // pto: %o_r_i8_pad__rv_v4_view
  int64_t v42 = v14 * v17;
  // pto: %o_r_i8_pad__rv_v4_view
  int64_t v43 = v41 * v42;
  // pto: %o_r_i8_pad__rv_v4_view
  pto::Shape<1, 1, 1, -1, -1> v44 = pto::Shape<1, 1, 1, -1, -1>(v39, v40, v41, v14, v17);
  // pto: %o_r_i8_pad__rv_v4_view
  pto::Stride<-1, -1, -1, -1, -1> v45 = pto::Stride<-1, -1, -1, -1, -1>(v40 * v43, v43, v42, v17, v16);
  // pto: %o_r_i8_pad__rv_v4_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v46 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v44, v45);
  // pto: %wo_b__ssa_v0_view
  pto::Shape<-1, -1, -1, -1, -1> v47 = pto::Shape<-1, -1, -1, -1, -1>(v18, v19, v20, v14, v19);
  // pto: %wo_b__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v48 = pto::Stride<-1, -1, -1, -1, -1>(v21, v22, v23, v19, v16);
  // pto: %wo_b__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<-1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::NZ> v49 = GlobalTensor<int8_t, pto::Shape<-1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::NZ>(v3, v47, v48);
  // pto: %dc__ssa_v0, %13
  int64_t v50 = (int64_t) ((uint64_t) ((int64_t) v5) * (uint64_t) v23);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  for (int64_t v51 = v24; v51 < v25; v51 += v16) {
    // pto: %14
    ;
    int64_t v52 = (int64_t) ((uint64_t) v51 * (uint64_t) v20);
    // pto: %15
    ;
    int64_t v53 = (int64_t) ((uint64_t) v50 + (uint64_t) v52);
    // pto: %acc_b__tile
    ;
    Tile<TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v54 = Tile<TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v14, v20);
    // pto: %acc_b__tile
    ;
    uint64_t v55 = (uint64_t) v7;
    TASSIGN(v54, v55);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int64_t v56 = v24; v56 < v26; v56 += v25) {
      // pto: %16
      ;
      int64_t v57 = (int64_t) ((uint64_t) v56 * (uint64_t) v20);
      // pto: %18
      ;
      int64_t v58 = (int64_t) ((uint64_t) v57 + (uint64_t) v20);
      // pto: %t__tile
      ;
      Tile<TileType::Mat, int8_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v59 = Tile<TileType::Mat, int8_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v14, v20);
      // pto: %t__tile
      ;
      uint64_t v60 = (uint64_t) v7;
      TASSIGN(v59, v60);
      // pto: %19
      ;
      int64_t v61 = (int64_t) ((uint64_t) v4 * (uint64_t) v27);
      // pto: %20
      ;
      int64_t v62 = (int64_t) ((uint64_t) v61 + (uint64_t) v57);
      // pto: %21
      ;
      int64_t v63 = v62 < v24 ? v24 : v62;
      // pto: %o_r_i8_pad__rv_v4_pview
      ;
      __gm__ int8_t* v64 = PTOAS__GLOBAL_TENSOR_DATA(v46);
      // pto: %o_r_i8_pad__rv_v4_pview
      ;
      const int64_t v65 = 0;
      // pto: %o_r_i8_pad__rv_v4_pview
      ;
      pto::Shape<1, 1, 1, 16, 256> v66 = pto::Shape<1, 1, 1, 16, 256>();
      // pto: %o_r_i8_pad__rv_v4_pview
      ;
      pto::Stride<131072, 131072, 131072, 8192, 1> v67 = pto::Stride<131072, 131072, 131072, 8192, 1>();
      // pto: %o_r_i8_pad__rv_v4_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<131072, 131072, 131072, 8192, 1>, pto::Layout::ND> v68 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<131072, 131072, 131072, 8192, 1>, pto::Layout::ND>(v64 + (v65 + v63), v66, v67);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v59, v68);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      // pto: %0
      ;
      Tile<TileType::Mat, int8_t, 256, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v69 = Tile<TileType::Mat, int8_t, 256, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v20, v20);
      // pto: %0
      ;
      uint64_t v70 = (uint64_t) v8;
      TASSIGN(v69, v70);
      // pto: %22
      ;
      int64_t v71 = v4 < v24 ? v24 : v4;
      // pto: %23
      ;
      int64_t v72 = v57 / v19;
      // pto: %24
      ;
      int64_t v73 = v72 < v24 ? v24 : v72;
      // pto: %25
      ;
      int64_t v74 = v53 / v14;
      // pto: %26
      ;
      int64_t v75 = v74 < v24 ? v24 : v74;
      // pto: %wo_b__ssa_v0_pview
      ;
      __gm__ int8_t* v76 = PTOAS__GLOBAL_TENSOR_DATA(v49);
      // pto: %wo_b__ssa_v0_pview
      ;
      const int64_t v77 = 0;
      // pto: %wo_b__ssa_v0_pview
      ;
      const int64_t v78 = 4194304;
      // pto: %wo_b__ssa_v0_pview
      ;
      const int64_t v79 = 131072;
      // pto: %wo_b__ssa_v0_pview
      ;
      const int64_t v80 = 512;
      // pto: %wo_b__ssa_v0_pview
      ;
      pto::Shape<1, 8, 16, 16, 32> v81 = pto::Shape<1, 8, 16, 16, 32>();
      // pto: %wo_b__ssa_v0_pview
      ;
      pto::Stride<4194304, 131072, 512, 32, 1> v82 = pto::Stride<4194304, 131072, 512, 32, 1>();
      // pto: %wo_b__ssa_v0_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 8, 16, 16, 32>, pto::Stride<4194304, 131072, 512, 32, 1>, pto::Layout::NZ> v83 = GlobalTensor<int8_t, pto::Shape<1, 8, 16, 16, 32>, pto::Stride<4194304, 131072, 512, 32, 1>, pto::Layout::NZ>(v76 + ((v77 + v71 * v78) + v73 * v79 + v75 * v80), v81, v82);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      TLOAD(v69, v83);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      // pto: %1
      ;
      Tile<TileType::Mat, int8_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v84 = Tile<TileType::Mat, int8_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v14, v20);
      // pto: %1
      ;
      uint64_t v85 = (uint64_t) v9;
      TASSIGN(v84, v85);
      // pto: %28
      ;
      int64_t v86 = (int64_t) ((uint64_t) v61 + (uint64_t) v58);
      // pto: %29
      ;
      int64_t v87 = v86 < v24 ? v24 : v86;
      // pto: %30
      ;
      __gm__ int8_t* v88 = PTOAS__GLOBAL_TENSOR_DATA(v46);
      // pto: %30
      ;
      const int64_t v89 = 0;
      // pto: %30
      ;
      pto::Shape<1, 1, 1, 16, 256> v90 = pto::Shape<1, 1, 1, 16, 256>();
      // pto: %30
      ;
      pto::Stride<131072, 131072, 131072, 8192, 1> v91 = pto::Stride<131072, 131072, 131072, 8192, 1>();
      // pto: %30
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<131072, 131072, 131072, 8192, 1>, pto::Layout::ND> v92 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<131072, 131072, 131072, 8192, 1>, pto::Layout::ND>(v88 + (v89 + v87), v90, v91);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v84, v92);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
      // pto: %2
      ;
      Tile<TileType::Mat, int8_t, 256, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v93 = Tile<TileType::Mat, int8_t, 256, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v20, v20);
      // pto: %2
      ;
      uint64_t v94 = (uint64_t) v10;
      TASSIGN(v93, v94);
      // pto: %32
      ;
      int64_t v95 = v58 / v19;
      // pto: %33
      ;
      int64_t v96 = v95 < v24 ? v24 : v95;
      // pto: %36
      ;
      __gm__ int8_t* v97 = PTOAS__GLOBAL_TENSOR_DATA(v49);
      // pto: %36
      ;
      const int64_t v98 = 0;
      // pto: %36
      ;
      const int64_t v99 = 4194304;
      // pto: %36
      ;
      const int64_t v100 = 131072;
      // pto: %36
      ;
      const int64_t v101 = 512;
      // pto: %36
      ;
      pto::Shape<1, 8, 16, 16, 32> v102 = pto::Shape<1, 8, 16, 16, 32>();
      // pto: %36
      ;
      pto::Stride<4194304, 131072, 512, 32, 1> v103 = pto::Stride<4194304, 131072, 512, 32, 1>();
      // pto: %36
      ;
      GlobalTensor<int8_t, pto::Shape<1, 8, 16, 16, 32>, pto::Stride<4194304, 131072, 512, 32, 1>, pto::Layout::NZ> v104 = GlobalTensor<int8_t, pto::Shape<1, 8, 16, 16, 32>, pto::Stride<4194304, 131072, 512, 32, 1>, pto::Layout::NZ>(v97 + ((v98 + v71 * v99) + v96 * v100 + v75 * v101), v102, v103);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
      TLOAD(v93, v104);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
      // pto: %t__tile_t
      ;
      Tile<TileType::Mat, int8_t, 256, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v105 = Tile<TileType::Mat, int8_t, 256, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v20, v20);
      // pto: %t__tile_t
      ;
      uint64_t v106 = (uint64_t) v8;
      TASSIGN(v105, v106);
      // pto: %acc_b__iter_v1_l0_a
      ;
      Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v107 = Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v14, v28);
      // pto: %acc_b__iter_v1_l0_a
      ;
      uint64_t v108 = (uint64_t) v11;
      TASSIGN(v107, v108);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      uint16_t v109 = static_cast<uint16_t>(v24);
      uint16_t v110 = static_cast<uint16_t>(v24);
      TEXTRACT(v107, v59, v109, v110);
      // pto: %acc_b__iter_v1_l0_b
      ;
      Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v111 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v28, v20);
      // pto: %acc_b__iter_v1_l0_b
      ;
      uint64_t v112 = (uint64_t) v12;
      TASSIGN(v111, v112);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      uint16_t v113 = static_cast<uint16_t>(v24);
      uint16_t v114 = static_cast<uint16_t>(v24);
      TEXTRACT(v111, v105, v113, v114);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      // pto: %3
      ;
      Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v115 = Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v14, v28);
      // pto: %3
      ;
      uint64_t v116 = (uint64_t) v7;
      TASSIGN(v115, v116);
      uint16_t v117 = static_cast<uint16_t>(v24);
      uint16_t v118 = static_cast<uint16_t>(v28);
      TEXTRACT(v115, v59, v117, v118);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      // pto: %4
      ;
      Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v119 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v28, v20);
      // pto: %4
      ;
      uint64_t v120 = (uint64_t) v7;
      TASSIGN(v119, v120);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      uint16_t v121 = static_cast<uint16_t>(v28);
      uint16_t v122 = static_cast<uint16_t>(v24);
      TEXTRACT(v119, v105, v121, v122);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      // pto: %37
      ;
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v56 == v24) {
        pipe_barrier(PIPE_M);
        TMATMUL(v54, v107, v111);
      } else {
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v54, v54, v107, v111);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
      pipe_barrier(PIPE_M);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      TMATMUL_ACC(v54, v54, v115, v119);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
      // pto: %6
      ;
      Tile<TileType::Mat, int8_t, 256, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v123 = Tile<TileType::Mat, int8_t, 256, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v20, v20);
      // pto: %6
      ;
      uint64_t v124 = (uint64_t) v10;
      TASSIGN(v123, v124);
      // pto: %7
      ;
      Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v125 = Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v14, v28);
      // pto: %7
      ;
      uint64_t v126 = (uint64_t) v13;
      TASSIGN(v125, v126);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
      uint16_t v127 = static_cast<uint16_t>(v24);
      uint16_t v128 = static_cast<uint16_t>(v24);
      TEXTRACT(v125, v84, v127, v128);
      // pto: %8
      ;
      Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v129 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v28, v20);
      // pto: %8
      ;
      uint64_t v130 = (uint64_t) v12;
      TASSIGN(v129, v130);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
      pipe_barrier(PIPE_MTE1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
      uint16_t v131 = static_cast<uint16_t>(v24);
      uint16_t v132 = static_cast<uint16_t>(v24);
      TEXTRACT(v129, v123, v131, v132);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
      // pto: %9
      ;
      Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v133 = Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v14, v28);
      // pto: %9
      ;
      uint64_t v134 = (uint64_t) v8;
      TASSIGN(v133, v134);
      uint16_t v135 = static_cast<uint16_t>(v24);
      uint16_t v136 = static_cast<uint16_t>(v28);
      TEXTRACT(v133, v84, v135, v136);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      // pto: %10
      ;
      Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v137 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v28, v20);
      // pto: %10
      ;
      uint64_t v138 = (uint64_t) v7;
      TASSIGN(v137, v138);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
      uint16_t v139 = static_cast<uint16_t>(v28);
      uint16_t v140 = static_cast<uint16_t>(v24);
      TEXTRACT(v137, v123, v139, v140);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
      // pto: %38
      ;
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
      if (v56 == v29) {
        pipe_barrier(PIPE_M);
        TMATMUL(v54, v125, v129);
      } else {
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v54, v54, v125, v129);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_M);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
      TMATMUL_ACC(v54, v54, v133, v137);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    // pto: %39
    ;
    int64_t v141 = (int64_t) ((uint64_t) v4 * (uint64_t) v30);
    // pto: %40
    ;
    int64_t v142 = (int64_t) ((uint64_t) v141 + (uint64_t) v53);
    // pto: %41
    ;
    int64_t v143 = v142 < v24 ? v24 : v142;
    // pto: %partials__iter_v3_pview
    ;
    __gm__ int32_t* v144 = PTOAS__GLOBAL_TENSOR_DATA(v38);
    // pto: %partials__iter_v3_pview
    ;
    const int64_t v145 = 0;
    // pto: %partials__iter_v3_pview
    ;
    pto::Shape<1, 1, 1, 16, 256> v146 = pto::Shape<1, 1, 1, 16, 256>();
    // pto: %partials__iter_v3_pview
    ;
    pto::Stride<524288, 524288, 524288, 32768, 1> v147 = pto::Stride<524288, 524288, 524288, 32768, 1>();
    // pto: %partials__iter_v3_pview
    ;
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<524288, 524288, 524288, 32768, 1>, pto::Layout::ND> v148 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<524288, 524288, 524288, 32768, 1>, pto::Layout::ND>(v144 + (v145 + v143), v146, v147);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v148, v54);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
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