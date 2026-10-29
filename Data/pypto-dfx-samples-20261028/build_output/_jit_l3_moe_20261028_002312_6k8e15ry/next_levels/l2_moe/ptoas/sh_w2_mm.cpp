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

AICORE void sh_w2_mm(__gm__ int8_t* v1, __gm__ int8_t* v2, __gm__ int32_t* v3, int64_t v4, int32_t v5, int32_t v6) {
  using T = float;

  #if defined(__DAV_CUBE__)
  // pto: %c0_i64
  const int64_t v7 = 0;
  // pto: %c8192_i64
  const int64_t v8 = 8192;
  // pto: %c139264_i64
  const int64_t v9 = 139264;
  // pto: %c147456_i64
  const int64_t v10 = 147456;
  // pto: %c6144_i64
  const int64_t v11 = 6144;
  // pto: %c32768_i64
  const int64_t v12 = 32768;
  // pto: %c2048_i64
  const int64_t v13 = 2048;
  // pto: %c4096_i64
  const int64_t v14 = 4096;
  // pto: %c16_index
  const int64_t v15 = 16;
  // pto: %c2048_index
  const int64_t v16 = 2048;
  // pto: %c1_index
  const int64_t v17 = 1;
  // pto: %c64_index
  const int64_t v18 = 64;
  // pto: %c256_index
  const int64_t v19 = 256;
  // pto: %c32_index
  const int64_t v20 = 32;
  // pto: %c8388608_index
  const int64_t v21 = 8388608;
  // pto: %c131072_index
  const int64_t v22 = 131072;
  // pto: %c512_index
  const int64_t v23 = 512;
  // pto: %c4096_index
  const int64_t v24 = 4096;
  // pto: %c0_index
  const int64_t v25 = 0;
  // pto: %c1024_index
  const int64_t v26 = 1024;
  // pto: %c128_index
  const int64_t v27 = 128;
  // pto: %cn128_index
  const int64_t v28 = -128;
  // pto: %cn512_index
  const int64_t v29 = -512;
  // pto: %h_tile_i8_inline291__rv_v3_view
  const int64_t v30 = 1;
  // pto: %h_tile_i8_inline291__rv_v3_view
  const int64_t v31 = 1;
  // pto: %h_tile_i8_inline291__rv_v3_view
  const int64_t v32 = 1;
  // pto: %h_tile_i8_inline291__rv_v3_view
  int64_t v33 = v15 * v16;
  // pto: %h_tile_i8_inline291__rv_v3_view
  int64_t v34 = v32 * v33;
  // pto: %h_tile_i8_inline291__rv_v3_view
  pto::Shape<1, 1, 1, -1, -1> v35 = pto::Shape<1, 1, 1, -1, -1>(v30, v31, v32, v15, v16);
  // pto: %h_tile_i8_inline291__rv_v3_view
  pto::Stride<-1, -1, -1, -1, -1> v36 = pto::Stride<-1, -1, -1, -1, -1>(v31 * v34, v34, v33, v16, v17);
  // pto: %h_tile_i8_inline291__rv_v3_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v37 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v35, v36);
  // pto: %shared_w2__ssa_v0_view
  pto::Shape<-1, -1, -1, -1, -1> v38 = pto::Shape<-1, -1, -1, -1, -1>(v17, v18, v19, v15, v20);
  // pto: %shared_w2__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v39 = pto::Stride<-1, -1, -1, -1, -1>(v21, v22, v23, v20, v17);
  // pto: %shared_w2__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<-1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::NZ> v40 = GlobalTensor<int8_t, pto::Shape<-1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::NZ>(v2, v38, v39);
  // pto: %y_i32_inline318__ssa_v0_view
  const int64_t v41 = 1;
  // pto: %y_i32_inline318__ssa_v0_view
  const int64_t v42 = 1;
  // pto: %y_i32_inline318__ssa_v0_view
  const int64_t v43 = 1;
  // pto: %y_i32_inline318__ssa_v0_view
  int64_t v44 = v15 * v24;
  // pto: %y_i32_inline318__ssa_v0_view
  int64_t v45 = v43 * v44;
  // pto: %y_i32_inline318__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v46 = pto::Shape<1, 1, 1, -1, -1>(v41, v42, v43, v15, v24);
  // pto: %y_i32_inline318__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v47 = pto::Stride<-1, -1, -1, -1, -1>(v42 * v45, v45, v44, v24, v17);
  // pto: %y_i32_inline318__ssa_v0_view
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v48 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v46, v47);
  // pto: %db_idx_inline319__ssa_v0, %12
  int64_t v49 = (int64_t) ((uint64_t) ((int64_t) v5) * (uint64_t) v19);
  // pto: %y_acc_inline321__tile
  Tile<TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v50 = Tile<TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v15, v19);
  // pto: %y_acc_inline321__tile
  uint64_t v51 = (uint64_t) v7;
  TASSIGN(v50, v51);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
  for (int64_t v52 = v25; v52 < v16; v52 += v26) {
    // pto: %hs_k_inline323__tile
    ;
    Tile<TileType::Mat, int8_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v53 = Tile<TileType::Mat, int8_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v15, v23);
    // pto: %hs_k_inline323__tile
    ;
    uint64_t v54 = (uint64_t) v7;
    TASSIGN(v53, v54);
    // pto: %13
    ;
    int64_t v55 = v52 < v25 ? v25 : v52;
    // pto: %h_tile_i8_inline291__rv_v3_pview
    ;
    __gm__ int8_t* v56 = PTOAS__GLOBAL_TENSOR_DATA(v37);
    // pto: %h_tile_i8_inline291__rv_v3_pview
    ;
    const int64_t v57 = 0;
    // pto: %h_tile_i8_inline291__rv_v3_pview
    ;
    pto::Shape<1, 1, 1, 16, 512> v58 = pto::Shape<1, 1, 1, 16, 512>();
    // pto: %h_tile_i8_inline291__rv_v3_pview
    ;
    pto::Stride<32768, 32768, 32768, 2048, 1> v59 = pto::Stride<32768, 32768, 32768, 2048, 1>();
    // pto: %h_tile_i8_inline291__rv_v3_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<32768, 32768, 32768, 2048, 1>, pto::Layout::ND> v60 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<32768, 32768, 32768, 2048, 1>, pto::Layout::ND>(v56 + (v57 + v55), v58, v59);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    TLOAD(v53, v60);
    // pto: %sw2_k_inline324__tile
    ;
    Tile<TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v61 = Tile<TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v19, v23);
    // pto: %sw2_k_inline324__tile
    ;
    uint64_t v62 = (uint64_t) v8;
    TASSIGN(v61, v62);
    // pto: %14
    ;
    int64_t v63 = v52 / v20;
    // pto: %15
    ;
    int64_t v64 = v63 < v25 ? v25 : v63;
    // pto: %16
    ;
    int64_t v65 = v49 / v15;
    // pto: %17
    ;
    int64_t v66 = v65 < v25 ? v25 : v65;
    // pto: %shared_w2__ssa_v0_pview
    ;
    __gm__ int8_t* v67 = PTOAS__GLOBAL_TENSOR_DATA(v40);
    // pto: %shared_w2__ssa_v0_pview
    ;
    const int64_t v68 = 0;
    // pto: %shared_w2__ssa_v0_pview
    ;
    const int64_t v69 = 131072;
    // pto: %shared_w2__ssa_v0_pview
    ;
    const int64_t v70 = 512;
    // pto: %shared_w2__ssa_v0_pview
    ;
    pto::Shape<1, 16, 16, 16, 32> v71 = pto::Shape<1, 16, 16, 16, 32>();
    // pto: %shared_w2__ssa_v0_pview
    ;
    pto::Stride<8388608, 131072, 512, 32, 1> v72 = pto::Stride<8388608, 131072, 512, 32, 1>();
    // pto: %shared_w2__ssa_v0_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 16, 16, 16, 32>, pto::Stride<8388608, 131072, 512, 32, 1>, pto::Layout::NZ> v73 = GlobalTensor<int8_t, pto::Shape<1, 16, 16, 16, 32>, pto::Stride<8388608, 131072, 512, 32, 1>, pto::Layout::NZ>(v67 + (v68 + v64 * v69 + v66 * v70), v71, v72);
    __gm__ int8_t* v74 = PTOAS__GLOBAL_TENSOR_DATA(v73);
    __gm__ uint8_t* v75 = reinterpret_cast<__gm__ uint8_t*>(v74);
    __gm__ int8_t* v76 = (__gm__ int8_t*) (v75 + v4);
    GlobalTensor<int8_t, pto::Shape<1, 16, 16, 16, 32>, pto::Stride<8388608, 131072, 512, 32, 1>, pto::Layout::NZ> v77(nullptr);
    v77 = v73;
    TASSIGN(v77, v76);
    TLOAD(v61, v77);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    // pto: %0
    ;
    Tile<TileType::Mat, int8_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v78 = Tile<TileType::Mat, int8_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v15, v23);
    // pto: %0
    ;
    uint64_t v79 = (uint64_t) v9;
    TASSIGN(v78, v79);
    // pto: %18
    ;
    int64_t v80 = (int64_t) ((uint64_t) v52 + (uint64_t) v23);
    // pto: %19
    ;
    int64_t v81 = v80 < v25 ? v25 : v80;
    // pto: %20
    ;
    __gm__ int8_t* v82 = PTOAS__GLOBAL_TENSOR_DATA(v37);
    // pto: %20
    ;
    const int64_t v83 = 0;
    // pto: %20
    ;
    pto::Shape<1, 1, 1, 16, 512> v84 = pto::Shape<1, 1, 1, 16, 512>();
    // pto: %20
    ;
    pto::Stride<32768, 32768, 32768, 2048, 1> v85 = pto::Stride<32768, 32768, 32768, 2048, 1>();
    // pto: %20
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<32768, 32768, 32768, 2048, 1>, pto::Layout::ND> v86 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<32768, 32768, 32768, 2048, 1>, pto::Layout::ND>(v82 + (v83 + v81), v84, v85);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    TLOAD(v78, v86);
    // pto: %1
    ;
    Tile<TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v87 = Tile<TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v19, v23);
    // pto: %1
    ;
    uint64_t v88 = (uint64_t) v10;
    TASSIGN(v87, v88);
    // pto: %22
    ;
    int64_t v89 = (int64_t) ((uint64_t) v63 + (uint64_t) v15);
    // pto: %23
    ;
    int64_t v90 = v89 < v25 ? v25 : v89;
    // pto: %26
    ;
    __gm__ int8_t* v91 = PTOAS__GLOBAL_TENSOR_DATA(v40);
    // pto: %26
    ;
    const int64_t v92 = 0;
    // pto: %26
    ;
    const int64_t v93 = 131072;
    // pto: %26
    ;
    const int64_t v94 = 512;
    // pto: %26
    ;
    pto::Shape<1, 16, 16, 16, 32> v95 = pto::Shape<1, 16, 16, 16, 32>();
    // pto: %26
    ;
    pto::Stride<8388608, 131072, 512, 32, 1> v96 = pto::Stride<8388608, 131072, 512, 32, 1>();
    // pto: %26
    ;
    GlobalTensor<int8_t, pto::Shape<1, 16, 16, 16, 32>, pto::Stride<8388608, 131072, 512, 32, 1>, pto::Layout::NZ> v97 = GlobalTensor<int8_t, pto::Shape<1, 16, 16, 16, 32>, pto::Stride<8388608, 131072, 512, 32, 1>, pto::Layout::NZ>(v91 + (v92 + v90 * v93 + v66 * v94), v95, v96);
    __gm__ int8_t* v98 = PTOAS__GLOBAL_TENSOR_DATA(v97);
    __gm__ uint8_t* v99 = reinterpret_cast<__gm__ uint8_t*>(v98);
    __gm__ int8_t* v100 = (__gm__ int8_t*) (v99 + v4);
    GlobalTensor<int8_t, pto::Shape<1, 16, 16, 16, 32>, pto::Stride<8388608, 131072, 512, 32, 1>, pto::Layout::NZ> v101(nullptr);
    v101 = v97;
    TASSIGN(v101, v100);
    TLOAD(v87, v101);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    // pto: %sw2_k_inline324__tile_t
    ;
    Tile<TileType::Mat, int8_t, 512, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v102 = Tile<TileType::Mat, int8_t, 512, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v23, v19);
    // pto: %sw2_k_inline324__tile_t
    ;
    uint64_t v103 = (uint64_t) v8;
    TASSIGN(v102, v103);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    for (int64_t v104 = v25; v104 < v23; v104 += v19) {
      // pto: %y_acc_inline321__tile_l0_a
      ;
      Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v105 = Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v15, v27);
      // pto: %y_acc_inline321__tile_l0_a
      ;
      uint64_t v106 = (uint64_t) v11;
      TASSIGN(v105, v106);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      pipe_barrier(PIPE_MTE1);
      uint16_t v107 = static_cast<uint16_t>(v25);
      uint16_t v108 = static_cast<uint16_t>(v104);
      TEXTRACT(v105, v53, v107, v108);
      // pto: %y_acc_inline321__tile_l0_b
      ;
      Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v109 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v27, v19);
      // pto: %y_acc_inline321__tile_l0_b
      ;
      uint64_t v110 = (uint64_t) v12;
      TASSIGN(v109, v110);
      uint16_t v111 = static_cast<uint16_t>(v104);
      uint16_t v112 = static_cast<uint16_t>(v25);
      TEXTRACT(v109, v102, v111, v112);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      // pto: %2
      ;
      Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v113 = Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v15, v27);
      // pto: %2
      ;
      uint64_t v114 = (uint64_t) v7;
      TASSIGN(v113, v114);
      // pto: %27
      ;
      int64_t v115 = (int64_t) ((uint64_t) v104 + (uint64_t) v27);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
      uint16_t v116 = static_cast<uint16_t>(v25);
      uint16_t v117 = static_cast<uint16_t>(v115);
      TEXTRACT(v113, v53, v116, v117);
      // pto: %3
      ;
      Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v118 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v27, v19);
      // pto: %3
      ;
      uint64_t v119 = (uint64_t) v7;
      TASSIGN(v118, v119);
      uint16_t v120 = static_cast<uint16_t>(v115);
      uint16_t v121 = static_cast<uint16_t>(v25);
      TEXTRACT(v118, v102, v120, v121);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      // pto: %29
      ;
      bool v122 = v52 == v25;
      // pto: %30, %31
      ;
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v122 & v104 == v25) {
        pipe_barrier(PIPE_M);
        TMATMUL(v50, v105, v109);
      } else {
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v50, v50, v105, v109);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      // pto: %33, %34
      ;
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      if (v122 & v104 == v28) {
        pipe_barrier(PIPE_M);
        TMATMUL(v50, v113, v118);
      } else {
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v50, v50, v113, v118);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    };
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    // pto: %5
    ;
    Tile<TileType::Mat, int8_t, 512, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v123 = Tile<TileType::Mat, int8_t, 512, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v23, v19);
    // pto: %5
    ;
    uint64_t v124 = (uint64_t) v10;
    TASSIGN(v123, v124);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    for (int64_t v125 = v25; v125 < v23; v125 += v19) {
      // pto: %6
      ;
      Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v126 = Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v15, v27);
      // pto: %6
      ;
      uint64_t v127 = (uint64_t) v13;
      TASSIGN(v126, v127);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
      uint16_t v128 = static_cast<uint16_t>(v25);
      uint16_t v129 = static_cast<uint16_t>(v125);
      TEXTRACT(v126, v78, v128, v129);
      // pto: %7
      ;
      Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v130 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v27, v19);
      // pto: %7
      ;
      uint64_t v131 = (uint64_t) v12;
      TASSIGN(v130, v131);
      pipe_barrier(PIPE_MTE1);
      uint16_t v132 = static_cast<uint16_t>(v125);
      uint16_t v133 = static_cast<uint16_t>(v25);
      TEXTRACT(v130, v123, v132, v133);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
      // pto: %8
      ;
      Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v134 = Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v15, v27);
      // pto: %8
      ;
      uint64_t v135 = (uint64_t) v14;
      TASSIGN(v134, v135);
      // pto: %36
      ;
      int64_t v136 = (int64_t) ((uint64_t) v125 + (uint64_t) v27);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
      uint16_t v137 = static_cast<uint16_t>(v25);
      uint16_t v138 = static_cast<uint16_t>(v136);
      TEXTRACT(v134, v78, v137, v138);
      // pto: %9
      ;
      Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v139 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v27, v19);
      // pto: %9
      ;
      uint64_t v140 = (uint64_t) v7;
      TASSIGN(v139, v140);
      uint16_t v141 = static_cast<uint16_t>(v136);
      uint16_t v142 = static_cast<uint16_t>(v25);
      TEXTRACT(v139, v123, v141, v142);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
      // pto: %38
      ;
      bool v143 = v52 == v29;
      // pto: %39, %40
      ;
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
      if (v143 & v125 == v25) {
        pipe_barrier(PIPE_M);
        TMATMUL(v50, v126, v130);
      } else {
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v50, v50, v126, v130);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
      // pto: %42, %43
      ;
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
      if (v143 & v125 == v28) {
        pipe_barrier(PIPE_M);
        TMATMUL(v50, v134, v139);
      } else {
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v50, v50, v134, v139);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
    };
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  }
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  // pto: %44
  int64_t v144 = v49 < v25 ? v25 : v49;
  // pto: %y_i32_inline318__ssa_v0_pview
  __gm__ int32_t* v145 = PTOAS__GLOBAL_TENSOR_DATA(v48);
  // pto: %y_i32_inline318__ssa_v0_pview
  const int64_t v146 = 0;
  // pto: %y_i32_inline318__ssa_v0_pview
  pto::Shape<1, 1, 1, 16, 256> v147 = pto::Shape<1, 1, 1, 16, 256>();
  // pto: %y_i32_inline318__ssa_v0_pview
  pto::Stride<65536, 65536, 65536, 4096, 1> v148 = pto::Stride<65536, 65536, 65536, 4096, 1>();
  // pto: %y_i32_inline318__ssa_v0_pview
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND> v149 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>(v145 + (v146 + v144), v147, v148);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v149, v50);
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