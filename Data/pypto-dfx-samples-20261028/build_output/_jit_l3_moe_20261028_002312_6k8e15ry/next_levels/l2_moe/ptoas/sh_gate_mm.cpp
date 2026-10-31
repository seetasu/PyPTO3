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

AICORE void sh_gate_mm(__gm__ int8_t* v1, __gm__ int8_t* v2, __gm__ int32_t* v3, int64_t v4, int64_t v5, int64_t v6, int32_t v7, int32_t v8) {
  using T = float;

  #if defined(__DAV_CUBE__)
  // pto: %c0_i64
  const int64_t v9 = 0;
  // pto: %c8192_i64
  const int64_t v10 = 8192;
  // pto: %c139264_i64
  const int64_t v11 = 139264;
  // pto: %c147456_i64
  const int64_t v12 = 147456;
  // pto: %c6144_i64
  const int64_t v13 = 6144;
  // pto: %c32768_i64
  const int64_t v14 = 32768;
  // pto: %c2048_i64
  const int64_t v15 = 2048;
  // pto: %c4096_i64
  const int64_t v16 = 4096;
  // pto: %c8_index
  const int64_t v17 = 8;
  // pto: %c4096_index
  const int64_t v18 = 4096;
  // pto: %c1_index
  const int64_t v19 = 1;
  // pto: %c128_index
  const int64_t v20 = 128;
  // pto: %c16_index
  const int64_t v21 = 16;
  // pto: %c32_index
  const int64_t v22 = 32;
  // pto: %c8388608_index
  const int64_t v23 = 8388608;
  // pto: %c65536_index
  const int64_t v24 = 65536;
  // pto: %c512_index
  const int64_t v25 = 512;
  // pto: %c2048_index
  const int64_t v26 = 2048;
  // pto: %c256_index
  const int64_t v27 = 256;
  // pto: %c0_index
  const int64_t v28 = 0;
  // pto: %c1024_index
  const int64_t v29 = 1024;
  // pto: %cn128_index
  const int64_t v30 = -128;
  // pto: %cn512_index
  const int64_t v31 = -512;
  // pto: %x_norm_i8_inline3__rv_v2_view
  const int64_t v32 = 1;
  // pto: %x_norm_i8_inline3__rv_v2_view
  const int64_t v33 = 1;
  // pto: %x_norm_i8_inline3__rv_v2_view
  const int64_t v34 = 1;
  // pto: %x_norm_i8_inline3__rv_v2_view
  int64_t v35 = v17 * v18;
  // pto: %x_norm_i8_inline3__rv_v2_view
  int64_t v36 = v34 * v35;
  // pto: %x_norm_i8_inline3__rv_v2_view
  pto::Shape<1, 1, 1, -1, -1> v37 = pto::Shape<1, 1, 1, -1, -1>(v32, v33, v34, v17, v18);
  // pto: %x_norm_i8_inline3__rv_v2_view
  pto::Stride<-1, -1, -1, -1, -1> v38 = pto::Stride<-1, -1, -1, -1, -1>(v33 * v36, v36, v35, v18, v19);
  // pto: %x_norm_i8_inline3__rv_v2_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v39 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v37, v38);
  // pto: %shared_w1__ssa_v0_view
  pto::Shape<-1, -1, -1, -1, -1> v40 = pto::Shape<-1, -1, -1, -1, -1>(v19, v20, v20, v21, v22);
  // pto: %shared_w1__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v41 = pto::Stride<-1, -1, -1, -1, -1>(v23, v24, v25, v22, v19);
  // pto: %shared_w1__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<-1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::NZ> v42 = GlobalTensor<int8_t, pto::Shape<-1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::NZ>(v2, v40, v41);
  // pto: %gate_i32_inline278__ssa_v0_view
  const int64_t v43 = 1;
  // pto: %gate_i32_inline278__ssa_v0_view
  const int64_t v44 = 1;
  // pto: %gate_i32_inline278__ssa_v0_view
  const int64_t v45 = 1;
  // pto: %gate_i32_inline278__ssa_v0_view
  int64_t v46 = v21 * v26;
  // pto: %gate_i32_inline278__ssa_v0_view
  int64_t v47 = v45 * v46;
  // pto: %gate_i32_inline278__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v48 = pto::Shape<1, 1, 1, -1, -1>(v43, v44, v45, v21, v26);
  // pto: %gate_i32_inline278__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v49 = pto::Stride<-1, -1, -1, -1, -1>(v44 * v47, v47, v46, v26, v19);
  // pto: %gate_i32_inline278__ssa_v0_view
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v50 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v48, v49);
  // pto: %nb_idx_inline280__ssa_v0, %12
  int64_t v51 = (int64_t) ((uint64_t) ((int64_t) v7) * (uint64_t) v27);
  // pto: %gate_acc_inline282__tile
  Tile<TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v52 = Tile<TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v21, v27);
  // pto: %gate_acc_inline282__tile
  uint64_t v53 = (uint64_t) v9;
  TASSIGN(v52, v53);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
  for (int64_t v54 = v28; v54 < v18; v54 += v29) {
    // pto: %xs_k_inline284__tile
    ;
    Tile<TileType::Mat, int8_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v55 = Tile<TileType::Mat, int8_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v5, v25);
    // pto: %xs_k_inline284__tile
    ;
    uint64_t v56 = (uint64_t) v9;
    TASSIGN(v55, v56);
    // pto: %13
    ;
    int64_t v57 = v4 < v28 ? v28 : v4;
    // pto: %14
    ;
    int64_t v58 = v54 < v28 ? v28 : v54;
    // pto: %x_norm_i8_inline3__rv_v2_pview
    ;
    const int64_t v59 = 0;
    // pto: %x_norm_i8_inline3__rv_v2_pview
    ;
    __gm__ int8_t* v60 = PTOAS__GLOBAL_TENSOR_DATA(v39);
    // pto: %x_norm_i8_inline3__rv_v2_pview
    ;
    const int64_t v61 = 1;
    // pto: %x_norm_i8_inline3__rv_v2_pview
    ;
    const int64_t v62 = 1;
    // pto: %x_norm_i8_inline3__rv_v2_pview
    ;
    const int64_t v63 = 1;
    // pto: %x_norm_i8_inline3__rv_v2_pview
    ;
    int64_t v64 = (int64_t) v5;
    // pto: %x_norm_i8_inline3__rv_v2_pview
    ;
    int64_t v65 = v64 * v18;
    // pto: %x_norm_i8_inline3__rv_v2_pview
    ;
    int64_t v66 = v63 * v65;
    // pto: %x_norm_i8_inline3__rv_v2_pview
    ;
    pto::Shape<1, 1, 1, -1, 512> v67 = pto::Shape<1, 1, 1, -1, 512>(v61, v62, v63, v64, v25);
    // pto: %x_norm_i8_inline3__rv_v2_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v68 = pto::Stride<-1, -1, -1, -1, -1>(v62 * v66, v66, v65, v18, v19);
    // pto: %x_norm_i8_inline3__rv_v2_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v69 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v60 + (v59 + v57 * v18 + v58 * v19), v67, v68);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    TLOAD(v55, v69);
    // pto: %sw1_k_inline285__tile
    ;
    Tile<TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v70 = Tile<TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v27, v25);
    // pto: %sw1_k_inline285__tile
    ;
    uint64_t v71 = (uint64_t) v10;
    TASSIGN(v70, v71);
    // pto: %15
    ;
    int64_t v72 = v54 / v22;
    // pto: %16
    ;
    int64_t v73 = v72 < v28 ? v28 : v72;
    // pto: %17
    ;
    int64_t v74 = v51 / v21;
    // pto: %18
    ;
    int64_t v75 = v74 < v28 ? v28 : v74;
    // pto: %shared_w1__ssa_v0_pview
    ;
    __gm__ int8_t* v76 = PTOAS__GLOBAL_TENSOR_DATA(v42);
    // pto: %shared_w1__ssa_v0_pview
    ;
    const int64_t v77 = 0;
    // pto: %shared_w1__ssa_v0_pview
    ;
    const int64_t v78 = 65536;
    // pto: %shared_w1__ssa_v0_pview
    ;
    const int64_t v79 = 512;
    // pto: %shared_w1__ssa_v0_pview
    ;
    pto::Shape<1, 16, 16, 16, 32> v80 = pto::Shape<1, 16, 16, 16, 32>();
    // pto: %shared_w1__ssa_v0_pview
    ;
    pto::Stride<8388608, 65536, 512, 32, 1> v81 = pto::Stride<8388608, 65536, 512, 32, 1>();
    // pto: %shared_w1__ssa_v0_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 16, 16, 16, 32>, pto::Stride<8388608, 65536, 512, 32, 1>, pto::Layout::NZ> v82 = GlobalTensor<int8_t, pto::Shape<1, 16, 16, 16, 32>, pto::Stride<8388608, 65536, 512, 32, 1>, pto::Layout::NZ>(v76 + (v77 + v73 * v78 + v75 * v79), v80, v81);
    __gm__ int8_t* v83 = PTOAS__GLOBAL_TENSOR_DATA(v82);
    __gm__ uint8_t* v84 = reinterpret_cast<__gm__ uint8_t*>(v83);
    __gm__ int8_t* v85 = (__gm__ int8_t*) (v84 + v6);
    GlobalTensor<int8_t, pto::Shape<1, 16, 16, 16, 32>, pto::Stride<8388608, 65536, 512, 32, 1>, pto::Layout::NZ> v86(nullptr);
    v86 = v82;
    TASSIGN(v86, v85);
    TLOAD(v70, v86);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    // pto: %0
    ;
    Tile<TileType::Mat, int8_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v87 = Tile<TileType::Mat, int8_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v5, v25);
    // pto: %0
    ;
    uint64_t v88 = (uint64_t) v11;
    TASSIGN(v87, v88);
    // pto: %20
    ;
    int64_t v89 = (int64_t) ((uint64_t) v54 + (uint64_t) v25);
    // pto: %21
    ;
    int64_t v90 = v89 < v28 ? v28 : v89;
    // pto: %22
    ;
    const int64_t v91 = 0;
    // pto: %22
    ;
    __gm__ int8_t* v92 = PTOAS__GLOBAL_TENSOR_DATA(v39);
    // pto: %22
    ;
    const int64_t v93 = 1;
    // pto: %22
    ;
    const int64_t v94 = 1;
    // pto: %22
    ;
    const int64_t v95 = 1;
    // pto: %22
    ;
    int64_t v96 = (int64_t) v5;
    // pto: %22
    ;
    int64_t v97 = v96 * v18;
    // pto: %22
    ;
    int64_t v98 = v95 * v97;
    // pto: %22
    ;
    pto::Shape<1, 1, 1, -1, 512> v99 = pto::Shape<1, 1, 1, -1, 512>(v93, v94, v95, v96, v25);
    // pto: %22
    ;
    pto::Stride<-1, -1, -1, -1, -1> v100 = pto::Stride<-1, -1, -1, -1, -1>(v94 * v98, v98, v97, v18, v19);
    // pto: %22
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v101 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v92 + (v91 + v57 * v18 + v90 * v19), v99, v100);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    TLOAD(v87, v101);
    // pto: %1
    ;
    Tile<TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v102 = Tile<TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v27, v25);
    // pto: %1
    ;
    uint64_t v103 = (uint64_t) v12;
    TASSIGN(v102, v103);
    // pto: %24
    ;
    int64_t v104 = (int64_t) ((uint64_t) v72 + (uint64_t) v21);
    // pto: %25
    ;
    int64_t v105 = v104 < v28 ? v28 : v104;
    // pto: %28
    ;
    __gm__ int8_t* v106 = PTOAS__GLOBAL_TENSOR_DATA(v42);
    // pto: %28
    ;
    const int64_t v107 = 0;
    // pto: %28
    ;
    const int64_t v108 = 65536;
    // pto: %28
    ;
    const int64_t v109 = 512;
    // pto: %28
    ;
    pto::Shape<1, 16, 16, 16, 32> v110 = pto::Shape<1, 16, 16, 16, 32>();
    // pto: %28
    ;
    pto::Stride<8388608, 65536, 512, 32, 1> v111 = pto::Stride<8388608, 65536, 512, 32, 1>();
    // pto: %28
    ;
    GlobalTensor<int8_t, pto::Shape<1, 16, 16, 16, 32>, pto::Stride<8388608, 65536, 512, 32, 1>, pto::Layout::NZ> v112 = GlobalTensor<int8_t, pto::Shape<1, 16, 16, 16, 32>, pto::Stride<8388608, 65536, 512, 32, 1>, pto::Layout::NZ>(v106 + (v107 + v105 * v108 + v75 * v109), v110, v111);
    __gm__ int8_t* v113 = PTOAS__GLOBAL_TENSOR_DATA(v112);
    __gm__ uint8_t* v114 = reinterpret_cast<__gm__ uint8_t*>(v113);
    __gm__ int8_t* v115 = (__gm__ int8_t*) (v114 + v6);
    GlobalTensor<int8_t, pto::Shape<1, 16, 16, 16, 32>, pto::Stride<8388608, 65536, 512, 32, 1>, pto::Layout::NZ> v116(nullptr);
    v116 = v112;
    TASSIGN(v116, v115);
    TLOAD(v102, v116);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    // pto: %sw1_k_inline285__tile_t
    ;
    Tile<TileType::Mat, int8_t, 512, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v117 = Tile<TileType::Mat, int8_t, 512, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v25, v27);
    // pto: %sw1_k_inline285__tile_t
    ;
    uint64_t v118 = (uint64_t) v10;
    TASSIGN(v117, v118);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    for (int64_t v119 = v28; v119 < v25; v119 += v27) {
      // pto: %29
      ;
      int64_t v120 = v5 < v21 ? v5 : v21;
      // pto: %gate_acc_inline282__tile_l0_a
      ;
      Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v121 = Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v120, v20);
      // pto: %gate_acc_inline282__tile_l0_a
      ;
      uint64_t v122 = (uint64_t) v13;
      TASSIGN(v121, v122);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      pipe_barrier(PIPE_MTE1);
      uint16_t v123 = static_cast<uint16_t>(v28);
      uint16_t v124 = static_cast<uint16_t>(v119);
      TEXTRACT(v121, v55, v123, v124);
      // pto: %gate_acc_inline282__tile_l0_b
      ;
      Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v125 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v20, v27);
      // pto: %gate_acc_inline282__tile_l0_b
      ;
      uint64_t v126 = (uint64_t) v14;
      TASSIGN(v125, v126);
      uint16_t v127 = static_cast<uint16_t>(v119);
      uint16_t v128 = static_cast<uint16_t>(v28);
      TEXTRACT(v125, v117, v127, v128);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      // pto: %2
      ;
      Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v129 = Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v120, v20);
      // pto: %2
      ;
      uint64_t v130 = (uint64_t) v9;
      TASSIGN(v129, v130);
      // pto: %31
      ;
      int64_t v131 = (int64_t) ((uint64_t) v119 + (uint64_t) v20);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
      uint16_t v132 = static_cast<uint16_t>(v28);
      uint16_t v133 = static_cast<uint16_t>(v131);
      TEXTRACT(v129, v55, v132, v133);
      // pto: %3
      ;
      Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v134 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v20, v27);
      // pto: %3
      ;
      uint64_t v135 = (uint64_t) v9;
      TASSIGN(v134, v135);
      uint16_t v136 = static_cast<uint16_t>(v131);
      uint16_t v137 = static_cast<uint16_t>(v28);
      TEXTRACT(v134, v117, v136, v137);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      // pto: %33
      ;
      bool v138 = v54 == v28;
      // pto: %34, %35
      ;
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v138 & v119 == v28) {
        pipe_barrier(PIPE_M);
        TMATMUL(v52, v121, v125);
      } else {
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v52, v52, v121, v125);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      // pto: %37, %38
      ;
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      if (v138 & v119 == v30) {
        pipe_barrier(PIPE_M);
        TMATMUL(v52, v129, v134);
      } else {
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v52, v52, v129, v134);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    };
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    // pto: %5
    ;
    Tile<TileType::Mat, int8_t, 512, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v139 = Tile<TileType::Mat, int8_t, 512, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v25, v27);
    // pto: %5
    ;
    uint64_t v140 = (uint64_t) v12;
    TASSIGN(v139, v140);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    for (int64_t v141 = v28; v141 < v25; v141 += v27) {
      // pto: %40
      ;
      int64_t v142 = v5 < v21 ? v5 : v21;
      // pto: %6
      ;
      Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v143 = Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v142, v20);
      // pto: %6
      ;
      uint64_t v144 = (uint64_t) v15;
      TASSIGN(v143, v144);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
      uint16_t v145 = static_cast<uint16_t>(v28);
      uint16_t v146 = static_cast<uint16_t>(v141);
      TEXTRACT(v143, v87, v145, v146);
      // pto: %7
      ;
      Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v147 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v20, v27);
      // pto: %7
      ;
      uint64_t v148 = (uint64_t) v14;
      TASSIGN(v147, v148);
      pipe_barrier(PIPE_MTE1);
      uint16_t v149 = static_cast<uint16_t>(v141);
      uint16_t v150 = static_cast<uint16_t>(v28);
      TEXTRACT(v147, v139, v149, v150);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
      // pto: %8
      ;
      Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v151 = Tile<TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v142, v20);
      // pto: %8
      ;
      uint64_t v152 = (uint64_t) v16;
      TASSIGN(v151, v152);
      // pto: %42
      ;
      int64_t v153 = (int64_t) ((uint64_t) v141 + (uint64_t) v20);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
      uint16_t v154 = static_cast<uint16_t>(v28);
      uint16_t v155 = static_cast<uint16_t>(v153);
      TEXTRACT(v151, v87, v154, v155);
      // pto: %9
      ;
      Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v156 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v20, v27);
      // pto: %9
      ;
      uint64_t v157 = (uint64_t) v9;
      TASSIGN(v156, v157);
      uint16_t v158 = static_cast<uint16_t>(v153);
      uint16_t v159 = static_cast<uint16_t>(v28);
      TEXTRACT(v156, v139, v158, v159);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
      // pto: %44
      ;
      bool v160 = v54 == v31;
      // pto: %45, %46
      ;
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
      if (v160 & v141 == v28) {
        pipe_barrier(PIPE_M);
        TMATMUL(v52, v143, v147);
      } else {
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v52, v52, v143, v147);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
      // pto: %48, %49
      ;
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
      if (v160 & v141 == v30) {
        pipe_barrier(PIPE_M);
        TMATMUL(v52, v151, v156);
      } else {
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v52, v52, v151, v156);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
    };
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  }
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  // pto: %50
  int64_t v161 = v51 < v28 ? v28 : v51;
  // pto: %gate_i32_inline278__ssa_v0_pview
  __gm__ int32_t* v162 = PTOAS__GLOBAL_TENSOR_DATA(v50);
  // pto: %gate_i32_inline278__ssa_v0_pview
  const int64_t v163 = 0;
  // pto: %gate_i32_inline278__ssa_v0_pview
  pto::Shape<1, 1, 1, 16, 256> v164 = pto::Shape<1, 1, 1, 16, 256>();
  // pto: %gate_i32_inline278__ssa_v0_pview
  pto::Stride<32768, 32768, 32768, 2048, 1> v165 = pto::Stride<32768, 32768, 32768, 2048, 1>();
  // pto: %gate_i32_inline278__ssa_v0_pview
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<32768, 32768, 32768, 2048, 1>, pto::Layout::ND> v166 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<32768, 32768, 32768, 2048, 1>, pto::Layout::ND>(v162 + (v163 + v161), v164, v165);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v166, v52);
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