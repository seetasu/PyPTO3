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

AICORE void staged_swa_proj_b_mm(__gm__ int32_t* v1, __gm__ int8_t* v2, __gm__ int8_t* v3, int64_t v4, int64_t v5, int64_t v6, int64_t v7, int32_t v8, int32_t v9) {
  using T = float;

  #if defined(__DAV_CUBE__)
  // pto: %c0_i64
  const int64_t v10 = 0;
  // pto: %c65536_i64
  const int64_t v11 = 65536;
  // pto: %c196608_i64
  const int64_t v12 = 196608;
  // pto: %c262144_i64
  const int64_t v13 = 262144;
  // pto: %c49152_i64
  const int64_t v14 = 49152;
  // pto: %c32768_i64
  const int64_t v15 = 32768;
  // pto: %c16384_i64
  const int64_t v16 = 16384;
  // pto: %c32768_index
  const int64_t v17 = 32768;
  // pto: %c1_index
  const int64_t v18 = 1;
  // pto: %c8192_index
  const int64_t v19 = 8192;
  // pto: %c8_index
  const int64_t v20 = 8;
  // pto: %c32_index
  const int64_t v21 = 32;
  // pto: %c256_index
  const int64_t v22 = 256;
  const int64_t v23 = 16;
  // pto: %c4194304_index
  const int64_t v24 = 4194304;
  // pto: %c131072_index
  const int64_t v25 = 131072;
  // pto: %c512_index
  const int64_t v26 = 512;
  // pto: %c0_index
  const int64_t v27 = 0;
  // pto: %c2_index
  const int64_t v28 = 2;
  // pto: %c128_index
  const int64_t v29 = 128;
  // pto: %cn128_index
  const int64_t v30 = -128;
  // pto: %c4096_index
  const int64_t v31 = 4096;
  // pto: %partials_inline658__iter_v1_view
  const int64_t v32 = 1;
  // pto: %partials_inline658__iter_v1_view
  const int64_t v33 = 1;
  // pto: %partials_inline658__iter_v1_view
  const int64_t v34 = 1;
  // pto: %partials_inline658__iter_v1_view
  int64_t v35 = (int64_t) v7;
  // pto: %partials_inline658__iter_v1_view
  int64_t v36 = v35 * v17;
  // pto: %partials_inline658__iter_v1_view
  int64_t v37 = v34 * v36;
  // pto: %partials_inline658__iter_v1_view
  pto::Shape<1, 1, 1, -1, -1> v38 = pto::Shape<1, 1, 1, -1, -1>(v32, v33, v34, v35, v17);
  // pto: %partials_inline658__iter_v1_view
  pto::Stride<-1, -1, -1, -1, -1> v39 = pto::Stride<-1, -1, -1, -1, -1>(v33 * v37, v37, v36, v17, v18);
  // pto: %partials_inline658__iter_v1_view
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v40 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v38, v39);
  // pto: %o_r_i8_inline656__rv_v2_view
  const int64_t v41 = 1;
  // pto: %o_r_i8_inline656__rv_v2_view
  const int64_t v42 = 1;
  // pto: %o_r_i8_inline656__rv_v2_view
  const int64_t v43 = 1;
  // pto: %o_r_i8_inline656__rv_v2_view
  int64_t v44 = (int64_t) v7;
  // pto: %o_r_i8_inline656__rv_v2_view
  int64_t v45 = v44 * v19;
  // pto: %o_r_i8_inline656__rv_v2_view
  int64_t v46 = v43 * v45;
  // pto: %o_r_i8_inline656__rv_v2_view
  pto::Shape<1, 1, 1, -1, -1> v47 = pto::Shape<1, 1, 1, -1, -1>(v41, v42, v43, v44, v19);
  // pto: %o_r_i8_inline656__rv_v2_view
  pto::Stride<-1, -1, -1, -1, -1> v48 = pto::Stride<-1, -1, -1, -1, -1>(v42 * v46, v46, v45, v19, v18);
  // pto: %o_r_i8_inline656__rv_v2_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v49 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v47, v48);
  // pto: %wo_b__ssa_v0_view
  pto::Shape<-1, -1, -1, -1, -1> v50 = pto::Shape<-1, -1, -1, -1, -1>(v20, v21, v22, v23, v21);
  // pto: %wo_b__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v51 = pto::Stride<-1, -1, -1, -1, -1>(v24, v25, v26, v21, v18);
  // pto: %wo_b__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<-1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::NZ> v52 = GlobalTensor<int8_t, pto::Shape<-1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::NZ>(v3, v50, v51);
  // pto: %dc_inline700__ssa_v0, %12
  int64_t v53 = (int64_t) ((uint64_t) ((int64_t) v8) * (uint64_t) v26);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
  for (int64_t v54 = v27; v54 < v28; v54 += v18) {
    // pto: %13
    ;
    int64_t v55 = (int64_t) ((uint64_t) v54 * (uint64_t) v22);
    // pto: %14
    ;
    int64_t v56 = (int64_t) ((uint64_t) v53 + (uint64_t) v55);
    // pto: %15
    ;
    for (int64_t v57 = v27; v57 < (v7 / v29); v57 += v18) {
      // pto: %16
      ;
      int64_t v58 = (int64_t) ((uint64_t) v57 * (uint64_t) v29);
      // pto: %acc_b_inline705__tile
      ;
      Tile<TileType::Acc, int32_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v59 = Tile<TileType::Acc, int32_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v29, v22);
      // pto: %acc_b_inline705__tile
      ;
      uint64_t v60 = (uint64_t) v10;
      TASSIGN(v59, v60);
      // pto: %b_act_inline707__tile
      ;
      Tile<TileType::Mat, int8_t, 128, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v61 = Tile<TileType::Mat, int8_t, 128, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v29, v26);
      // pto: %b_act_inline707__tile
      ;
      uint64_t v62 = (uint64_t) v10;
      TASSIGN(v61, v62);
      // pto: %17
      ;
      int64_t v63 = v58 < v27 ? v27 : v58;
      // pto: %19
      ;
      int64_t v64 = v5 < v27 ? v27 : v5;
      // pto: %o_r_i8_inline656__rv_v2_pview
      ;
      __gm__ int8_t* v65 = PTOAS__GLOBAL_TENSOR_DATA(v49);
      // pto: %o_r_i8_inline656__rv_v2_pview
      ;
      const int64_t v66 = 0;
      // pto: %o_r_i8_inline656__rv_v2_pview
      ;
      const int64_t v67 = 8192;
      // pto: %o_r_i8_inline656__rv_v2_pview
      ;
      pto::Shape<1, 1, 1, 128, 512> v68 = pto::Shape<1, 1, 1, 128, 512>();
      // pto: %o_r_i8_inline656__rv_v2_pview
      ;
      pto::Stride<1048576, 1048576, 1048576, 8192, 1> v69 = pto::Stride<1048576, 1048576, 1048576, 8192, 1>();
      // pto: %o_r_i8_inline656__rv_v2_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 128, 512>, pto::Stride<1048576, 1048576, 1048576, 8192, 1>, pto::Layout::ND> v70 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 128, 512>, pto::Stride<1048576, 1048576, 1048576, 8192, 1>, pto::Layout::ND>(v65 + (v66 + v63 * v67 + v64), v68, v69);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v61, v70);
      // pto: %b_weight_inline708__tile
      ;
      Tile<TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v71 = Tile<TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v22, v26);
      // pto: %b_weight_inline708__tile
      ;
      uint64_t v72 = (uint64_t) v11;
      TASSIGN(v71, v72);
      // pto: %20
      ;
      int64_t v73 = v6 < v27 ? v27 : v6;
      // pto: %21
      ;
      int64_t v74 = v56 / v23;
      // pto: %22
      ;
      int64_t v75 = v74 < v27 ? v27 : v74;
      // pto: %wo_b__ssa_v0_pview
      ;
      __gm__ int8_t* v76 = PTOAS__GLOBAL_TENSOR_DATA(v52);
      // pto: %wo_b__ssa_v0_pview
      ;
      const int64_t v77 = 0;
      // pto: %wo_b__ssa_v0_pview
      ;
      const int64_t v78 = 4194304;
      // pto: %wo_b__ssa_v0_pview
      ;
      const int64_t v79 = 512;
      // pto: %wo_b__ssa_v0_pview
      ;
      pto::Shape<1, 16, 16, 16, 32> v80 = pto::Shape<1, 16, 16, 16, 32>();
      // pto: %wo_b__ssa_v0_pview
      ;
      pto::Stride<4194304, 131072, 512, 32, 1> v81 = pto::Stride<4194304, 131072, 512, 32, 1>();
      // pto: %wo_b__ssa_v0_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 16, 16, 16, 32>, pto::Stride<4194304, 131072, 512, 32, 1>, pto::Layout::NZ> v82 = GlobalTensor<int8_t, pto::Shape<1, 16, 16, 16, 32>, pto::Stride<4194304, 131072, 512, 32, 1>, pto::Layout::NZ>(v76 + (v77 + v73 * v78 + v75 * v79), v80, v81);
      TLOAD(v71, v82);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      // pto: %0
      ;
      Tile<TileType::Mat, int8_t, 128, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v83 = Tile<TileType::Mat, int8_t, 128, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v29, v26);
      // pto: %0
      ;
      uint64_t v84 = (uint64_t) v12;
      TASSIGN(v83, v84);
      // pto: %24
      ;
      int64_t v85 = (int64_t) ((uint64_t) v5 + (uint64_t) v26);
      // pto: %25
      ;
      int64_t v86 = v85 < v27 ? v27 : v85;
      // pto: %26
      ;
      __gm__ int8_t* v87 = PTOAS__GLOBAL_TENSOR_DATA(v49);
      // pto: %26
      ;
      const int64_t v88 = 0;
      // pto: %26
      ;
      const int64_t v89 = 8192;
      // pto: %26
      ;
      pto::Shape<1, 1, 1, 128, 512> v90 = pto::Shape<1, 1, 1, 128, 512>();
      // pto: %26
      ;
      pto::Stride<1048576, 1048576, 1048576, 8192, 1> v91 = pto::Stride<1048576, 1048576, 1048576, 8192, 1>();
      // pto: %26
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 128, 512>, pto::Stride<1048576, 1048576, 1048576, 8192, 1>, pto::Layout::ND> v92 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 128, 512>, pto::Stride<1048576, 1048576, 1048576, 8192, 1>, pto::Layout::ND>(v87 + (v88 + v63 * v89 + v86), v90, v91);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      TLOAD(v83, v92);
      // pto: %1
      ;
      Tile<TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v93 = Tile<TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v22, v26);
      // pto: %1
      ;
      uint64_t v94 = (uint64_t) v13;
      TASSIGN(v93, v94);
      // pto: %32
      ;
      __gm__ int8_t* v95 = PTOAS__GLOBAL_TENSOR_DATA(v52);
      // pto: %32
      ;
      const int64_t v96 = 2097152;
      // pto: %32
      ;
      const int64_t v97 = 4194304;
      // pto: %32
      ;
      const int64_t v98 = 512;
      // pto: %32
      ;
      pto::Shape<1, 16, 16, 16, 32> v99 = pto::Shape<1, 16, 16, 16, 32>();
      // pto: %32
      ;
      pto::Stride<4194304, 131072, 512, 32, 1> v100 = pto::Stride<4194304, 131072, 512, 32, 1>();
      // pto: %32
      ;
      GlobalTensor<int8_t, pto::Shape<1, 16, 16, 16, 32>, pto::Stride<4194304, 131072, 512, 32, 1>, pto::Layout::NZ> v101 = GlobalTensor<int8_t, pto::Shape<1, 16, 16, 16, 32>, pto::Stride<4194304, 131072, 512, 32, 1>, pto::Layout::NZ>(v95 + (v96 + v73 * v97 + v75 * v98), v99, v100);
      TLOAD(v93, v101);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      // pto: %b_weight_inline708__tile_t
      ;
      Tile<TileType::Mat, int8_t, 512, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v102 = Tile<TileType::Mat, int8_t, 512, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v26, v22);
      // pto: %b_weight_inline708__tile_t
      ;
      uint64_t v103 = (uint64_t) v11;
      TASSIGN(v102, v103);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      for (int64_t v104 = v27; v104 < v26; v104 += v22) {
        // pto: %acc_b_inline705__iter_v1_l0_a
        ;
        Tile<TileType::Left, int8_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v105 = Tile<TileType::Left, int8_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v29, v29);
        // pto: %acc_b_inline705__iter_v1_l0_a
        ;
        uint64_t v106 = (uint64_t) v14;
        TASSIGN(v105, v106);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        pipe_barrier(PIPE_MTE1);
        uint16_t v107 = static_cast<uint16_t>(v27);
        uint16_t v108 = static_cast<uint16_t>(v104);
        TEXTRACT(v105, v61, v107, v108);
        // pto: %acc_b_inline705__iter_v1_l0_b
        ;
        Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v109 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v29, v22);
        // pto: %acc_b_inline705__iter_v1_l0_b
        ;
        uint64_t v110 = (uint64_t) v15;
        TASSIGN(v109, v110);
        uint16_t v111 = static_cast<uint16_t>(v104);
        uint16_t v112 = static_cast<uint16_t>(v27);
        TEXTRACT(v109, v102, v111, v112);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        // pto: %2
        ;
        Tile<TileType::Left, int8_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v113 = Tile<TileType::Left, int8_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v29, v29);
        // pto: %2
        ;
        uint64_t v114 = (uint64_t) v10;
        TASSIGN(v113, v114);
        // pto: %33
        ;
        int64_t v115 = (int64_t) ((uint64_t) v104 + (uint64_t) v29);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        uint16_t v116 = static_cast<uint16_t>(v27);
        uint16_t v117 = static_cast<uint16_t>(v115);
        TEXTRACT(v113, v61, v116, v117);
        // pto: %3
        ;
        Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v118 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v29, v22);
        // pto: %3
        ;
        uint64_t v119 = (uint64_t) v10;
        TASSIGN(v118, v119);
        uint16_t v120 = static_cast<uint16_t>(v115);
        uint16_t v121 = static_cast<uint16_t>(v27);
        TEXTRACT(v118, v102, v120, v121);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        // pto: %35
        ;
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        if (v104 == v27) {
          pipe_barrier(PIPE_M);
          TMATMUL(v59, v105, v109);
        } else {
          pipe_barrier(PIPE_M);
          TMATMUL_ACC(v59, v59, v105, v109);
        };
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        // pto: %36
        ;
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        if (v104 == v30) {
          pipe_barrier(PIPE_M);
          TMATMUL(v59, v113, v118);
        } else {
          pipe_barrier(PIPE_M);
          TMATMUL_ACC(v59, v59, v113, v118);
        };
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      // pto: %5
      ;
      Tile<TileType::Mat, int8_t, 512, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v122 = Tile<TileType::Mat, int8_t, 512, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v26, v22);
      // pto: %5
      ;
      uint64_t v123 = (uint64_t) v13;
      TASSIGN(v122, v123);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      for (int64_t v124 = v27; v124 < v26; v124 += v22) {
        // pto: %6
        ;
        Tile<TileType::Left, int8_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v125 = Tile<TileType::Left, int8_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v29, v29);
        // pto: %6
        ;
        uint64_t v126 = (uint64_t) v16;
        TASSIGN(v125, v126);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
        uint16_t v127 = static_cast<uint16_t>(v27);
        uint16_t v128 = static_cast<uint16_t>(v124);
        TEXTRACT(v125, v83, v127, v128);
        // pto: %7
        ;
        Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v129 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v29, v22);
        // pto: %7
        ;
        uint64_t v130 = (uint64_t) v15;
        TASSIGN(v129, v130);
        pipe_barrier(PIPE_MTE1);
        uint16_t v131 = static_cast<uint16_t>(v124);
        uint16_t v132 = static_cast<uint16_t>(v27);
        TEXTRACT(v129, v122, v131, v132);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
        // pto: %8
        ;
        Tile<TileType::Left, int8_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v133 = Tile<TileType::Left, int8_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v29, v29);
        // pto: %8
        ;
        uint64_t v134 = (uint64_t) v15;
        TASSIGN(v133, v134);
        // pto: %38
        ;
        int64_t v135 = (int64_t) ((uint64_t) v124 + (uint64_t) v29);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
        uint16_t v136 = static_cast<uint16_t>(v27);
        uint16_t v137 = static_cast<uint16_t>(v135);
        TEXTRACT(v133, v83, v136, v137);
        // pto: %9
        ;
        Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v138 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v29, v22);
        // pto: %9
        ;
        uint64_t v139 = (uint64_t) v10;
        TASSIGN(v138, v139);
        uint16_t v140 = static_cast<uint16_t>(v135);
        uint16_t v141 = static_cast<uint16_t>(v27);
        TEXTRACT(v138, v122, v140, v141);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v59, v59, v125, v129);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
        pipe_barrier(PIPE_M);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
        TMATMUL_ACC(v59, v59, v133, v138);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
      };
      set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      // pto: %41
      ;
      int64_t v142 = (int64_t) ((uint64_t) v6 * (uint64_t) v31);
      // pto: %42
      ;
      int64_t v143 = (int64_t) ((uint64_t) v142 + (uint64_t) v56);
      // pto: %43
      ;
      int64_t v144 = v143 < v27 ? v27 : v143;
      // pto: %partials_inline658__iter_v5_pview
      ;
      __gm__ int32_t* v145 = PTOAS__GLOBAL_TENSOR_DATA(v40);
      // pto: %partials_inline658__iter_v5_pview
      ;
      const int64_t v146 = 0;
      // pto: %partials_inline658__iter_v5_pview
      ;
      const int64_t v147 = 32768;
      // pto: %partials_inline658__iter_v5_pview
      ;
      pto::Shape<1, 1, 1, 128, 256> v148 = pto::Shape<1, 1, 1, 128, 256>();
      // pto: %partials_inline658__iter_v5_pview
      ;
      pto::Stride<4194304, 4194304, 4194304, 32768, 1> v149 = pto::Stride<4194304, 4194304, 4194304, 32768, 1>();
      // pto: %partials_inline658__iter_v5_pview
      ;
      GlobalTensor<int32_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<4194304, 4194304, 4194304, 32768, 1>, pto::Layout::ND> v150 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<4194304, 4194304, 4194304, 32768, 1>, pto::Layout::ND>(v145 + (v146 + v63 * v147 + v144), v148, v149);
      wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      pipe_barrier(PIPE_FIX);
      TSTORE(v150, v59);
      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    };
  }
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
  #endif // __DAV_CUBE__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}