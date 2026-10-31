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

AICORE void dispatch_push(__gm__ int32_t* v1, __gm__ float* v2, __gm__ int8_t* v3, __gm__ int8_t* v4, __gm__ float* v5, __gm__ float* v6, __gm__ int32_t* v7, int32_t v8, int32_t v9, __gm__ int64_t* v10, __gm__ int64_t* v11, __gm__ int64_t* v12, int32_t v13, int32_t v14) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c4096_i64
  const int64_t v15 = 4096;
  // pto: %c4352_i64
  const int64_t v16 = 4352;
  // pto: %c4608_i64
  const int64_t v17 = 4608;
  // pto: %c0_i64
  const int64_t v18 = 0;
  // pto: %c8_index
  const int64_t v19 = 8;
  // pto: %c6_index
  const int64_t v20 = 6;
  // pto: %c1_index
  const int64_t v21 = 1;
  // pto: %c512_index
  const int64_t v22 = 512;
  // pto: %c4096_index
  const int64_t v23 = 4096;
  // pto: %c2_index
  const int64_t v24 = 2;
  // pto: %c16_index
  const int64_t v25 = 16;
  // pto: %c0_index
  const int64_t v26 = 0;
  // pto: %c32_index
  const int64_t v27 = 32;
  // pto: %c0_i32
  const int32_t v28 = 0;
  // pto: %cst_14
  const float v29 = 0.0f;
  // pto: %c4_index
  const int64_t v30 = 4;
  // pto: %c4_i64
  const int64_t v31 = 4;
  // pto: %c1_i32
  const int32_t v32 = 1;
  // pto: %indices_inline5__ssa_v0_view
  const int64_t v33 = 1;
  // pto: %indices_inline5__ssa_v0_view
  const int64_t v34 = 1;
  // pto: %indices_inline5__ssa_v0_view
  const int64_t v35 = 1;
  // pto: %indices_inline5__ssa_v0_view
  int64_t v36 = v19 * v20;
  // pto: %indices_inline5__ssa_v0_view
  int64_t v37 = v35 * v36;
  // pto: %indices_inline5__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v38 = pto::Shape<1, 1, 1, -1, -1>(v33, v34, v35, v19, v20);
  // pto: %indices_inline5__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v39 = pto::Stride<-1, -1, -1, -1, -1>(v34 * v37, v37, v36, v20, v21);
  // pto: %indices_inline5__ssa_v0_view
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v40 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v38, v39);
  // pto: %weights_inline6__ssa_v0_view
  const int64_t v41 = 1;
  // pto: %weights_inline6__ssa_v0_view
  const int64_t v42 = 1;
  // pto: %weights_inline6__ssa_v0_view
  const int64_t v43 = 1;
  // pto: %weights_inline6__ssa_v0_view
  int64_t v44 = v19 * v20;
  // pto: %weights_inline6__ssa_v0_view
  int64_t v45 = v43 * v44;
  // pto: %weights_inline6__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v46 = pto::Shape<1, 1, 1, -1, -1>(v41, v42, v43, v19, v20);
  // pto: %weights_inline6__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v47 = pto::Stride<-1, -1, -1, -1, -1>(v42 * v45, v45, v44, v20, v21);
  // pto: %weights_inline6__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v48 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v46, v47);
  // pto: %x_norm_i8_inline3__rv_v2_view
  const int64_t v49 = 1;
  // pto: %x_norm_i8_inline3__rv_v2_view
  const int64_t v50 = 1;
  // pto: %x_norm_i8_inline3__rv_v2_view
  const int64_t v51 = 1;
  // pto: %x_norm_i8_inline3__rv_v2_view
  int64_t v52 = v19 * v23;
  // pto: %x_norm_i8_inline3__rv_v2_view
  int64_t v53 = v51 * v52;
  // pto: %x_norm_i8_inline3__rv_v2_view
  pto::Shape<1, 1, 1, -1, -1> v54 = pto::Shape<1, 1, 1, -1, -1>(v49, v50, v51, v19, v23);
  // pto: %x_norm_i8_inline3__rv_v2_view
  pto::Stride<-1, -1, -1, -1, -1> v55 = pto::Stride<-1, -1, -1, -1, -1>(v50 * v53, v53, v52, v23, v21);
  // pto: %x_norm_i8_inline3__rv_v2_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v56 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v54, v55);
  // pto: %push_block_inline356__ssa_v0
  int64_t v57 = (int64_t) v13;
  // pto: %0
  int64_t v58 = v57 / v25;
  // pto: %1
  int64_t v59 = (int64_t) ((uint64_t) v58 * (uint64_t) v25);
  // pto: %3
  int64_t v60 = (int64_t) v8;
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
  // pto: %4, %active_tokens_v1_inline359__phi_v2
  int64_t v61 = v60 < v26 ? v26 : v60;
  // pto: %5, %active_tokens_v1_inline359__phi_v4
  // pto: %slot_ctr_inline360__ssa_v0
  int32_t v62[32];
  for (int64_t v63 = v26; v63 < v27; v63 += v21) {
    v62[v63] = v28;
  }
  // pto: %indices_tile_inline362__ssa_v0
  Tile<TileType::Vec, int32_t, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v64 = Tile<TileType::Vec, int32_t, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v19, v20);
  // pto: %indices_tile_inline362__ssa_v0
  uint64_t v65 = (uint64_t) v15;
  TASSIGN(v64, v65);
  // pto: %indices_inline5__ssa_v0_pview
  __gm__ int32_t* v66 = PTOAS__GLOBAL_TENSOR_DATA(v40);
  // pto: %indices_inline5__ssa_v0_pview
  pto::Shape<1, 1, 1, 8, 6> v67 = pto::Shape<1, 1, 1, 8, 6>();
  // pto: %indices_inline5__ssa_v0_pview
  pto::Stride<48, 48, 48, 6, 1> v68 = pto::Stride<48, 48, 48, 6, 1>();
  // pto: %indices_inline5__ssa_v0_pview
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 6>, pto::Stride<48, 48, 48, 6, 1>, pto::Layout::ND> v69 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 6>, pto::Stride<48, 48, 48, 6, 1>, pto::Layout::ND>(v66, v67, v68);
  TLOAD(v64, v69);
  // pto: %weights_tile_inline363__ssa_v0
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v70 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v19, v20);
  // pto: %weights_tile_inline363__ssa_v0
  uint64_t v71 = (uint64_t) v16;
  TASSIGN(v70, v71);
  // pto: %weights_inline6__ssa_v0_pview
  __gm__ float* v72 = PTOAS__GLOBAL_TENSOR_DATA(v48);
  // pto: %weights_inline6__ssa_v0_pview
  pto::Shape<1, 1, 1, 8, 6> v73 = pto::Shape<1, 1, 1, 8, 6>();
  // pto: %weights_inline6__ssa_v0_pview
  pto::Stride<48, 48, 48, 6, 1> v74 = pto::Stride<48, 48, 48, 6, 1>();
  // pto: %weights_inline6__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 8, 6>, pto::Stride<48, 48, 48, 6, 1>, pto::Layout::ND> v75 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 6>, pto::Stride<48, 48, 48, 6, 1>, pto::Layout::ND>(v72, v73, v74);
  TLOAD(v70, v75);
  set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
  // pto: %aux_tile_inline364__ssa_v0
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v76 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v19);
  // pto: %aux_tile_inline364__ssa_v0
  uint64_t v77 = (uint64_t) v17;
  TASSIGN(v76, v77);
  TEXPANDS(v76, v29);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
  for (int64_t v78 = v26; v78 < (v61 > v19 ? v19 : v61); v78 += v21) {
    for (int64_t v79 = v26; v79 < v20; v79 += v21) {
      // pto: %flat_offset_mul
      ;
      int64_t v80 = (int64_t) ((uint64_t) v78 * (uint64_t) v19);
      // pto: %flat_offset
      ;
      int64_t v81 = (int64_t) ((uint64_t) v80 + (uint64_t) v79);
      // pto: %eid_inline344__ssa_v1
      ;
      int32_t v82 = v64.GetValue(v81);
      // pto: %6
      ;
      int64_t v83 = (int64_t) v82;
      // pto: %7
      ;
      int64_t v84 = v83 / v27;
      // pto: %9
      ;
      int64_t v85 = (int64_t) ((uint64_t) v84 * (uint64_t) v27);
      // pto: %10
      ;
      int64_t v86 = (int64_t) ((uint64_t) v83 - (uint64_t) v85);
      // pto: %12
      ;
      if (v84 == v58) {
        // pto: %11, %2, %13
        ;
        if (v86 % v25 == (int64_t) ((uint64_t) v57 - (uint64_t) v59)) {
          // pto: %slot_inline369__ssa_v0
          ;
          // pto: %slot_inline369__ssa_v0
          ;
          int32_t v87;
          v87 = v62[v86];
          // pto: %14
          ;
          int64_t v88 = (int64_t) v87;
          // pto: %15
          ;
          int64_t v89 = (int64_t) ((uint64_t) v88 + (uint64_t) v21);
          // pto: %16
          ;
          v62[v86] = (int32_t) v89;
          // pto: %18
          ;
          int64_t v90 = (int64_t) ((uint64_t) v86 * (uint64_t) v25);
          // pto: %19, %20
          ;
          int64_t v91 = (int64_t) ((uint64_t) ((int64_t) v9) * (uint64_t) v19);
          // pto: %21
          ;
          int64_t v92 = (int64_t) ((uint64_t) v90 + (uint64_t) v91);
          // pto: %14
          ;
          int64_t v93 = (int64_t) v87;
          // pto: %23
          ;
          int64_t v94 = (int64_t) ((uint64_t) v92 + (uint64_t) v93);
          // pto: %tput_stage
          ;
          Tile<TileType::Vec, int8_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v95 = Tile<TileType::Vec, int8_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v23);
          // pto: %tput_stage
          ;
          uint64_t v96 = (uint64_t) v18;
          TASSIGN(v95, v96);
          // pto: %24
          ;
          int64_t v97 = v94 < v26 ? v26 : v94;
          // pto: %25
          ;
          int64_t v98 = v78 < v26 ? v26 : v78;
          // pto: %26
          ;
          int64_t v99 = (v10)[v24];
          // pto: %27, %28, %29, %30
          ;
          int64_t v100 = (v10)[(int64_t) ((uint64_t) ((int64_t) (int32_t) v99) + (uint64_t) v30)];
          // pto: %31
          ;
          int64_t v101 = (int64_t) ((uint64_t) v58 + (uint64_t) v30);
          // pto: %32
          ;
          int64_t v102 = (v10)[v101];
          // pto: %37
          ;
          const int64_t v103 = 1;
          // pto: %37
          ;
          const int64_t v104 = 1;
          // pto: %37
          ;
          const int64_t v105 = 1;
          // pto: %37
          ;
          int64_t v106 = v22 * v23;
          // pto: %37
          ;
          int64_t v107 = v105 * v106;
          // pto: %37
          ;
          pto::Shape<1, 1, 1, -1, -1> v108 = pto::Shape<1, 1, 1, -1, -1>(v103, v104, v105, v22, v23);
          // pto: %37
          ;
          pto::Stride<-1, -1, -1, -1, -1> v109 = pto::Stride<-1, -1, -1, -1, -1>(v104 * v107, v107, v106, v23, v21);
          // pto: %33, %36, %37
          ;
          GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v110 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3 + (int64_t) ((uint64_t) v102 - (uint64_t) v100), v108, v109);
          // pto: %recv_x__ssa_v0_peer_pview
          ;
          __gm__ int8_t* v111 = PTOAS__GLOBAL_TENSOR_DATA(v110);
          // pto: %recv_x__ssa_v0_peer_pview
          ;
          const int64_t v112 = 0;
          // pto: %recv_x__ssa_v0_peer_pview
          ;
          const int64_t v113 = 4096;
          // pto: %recv_x__ssa_v0_peer_pview
          ;
          pto::Shape<1, 1, 1, 1, 4096> v114 = pto::Shape<1, 1, 1, 1, 4096>();
          // pto: %recv_x__ssa_v0_peer_pview
          ;
          pto::Stride<4096, 4096, 4096, 4096, 1> v115 = pto::Stride<4096, 4096, 4096, 4096, 1>();
          // pto: %recv_x__ssa_v0_peer_pview
          ;
          GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v116 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v111 + (v112 + v97 * v113), v114, v115);
          // pto: %x_norm_i8_inline3__rv_v2_local_pview
          ;
          __gm__ int8_t* v117 = PTOAS__GLOBAL_TENSOR_DATA(v56);
          // pto: %x_norm_i8_inline3__rv_v2_local_pview
          ;
          const int64_t v118 = 0;
          // pto: %x_norm_i8_inline3__rv_v2_local_pview
          ;
          const int64_t v119 = 4096;
          // pto: %x_norm_i8_inline3__rv_v2_local_pview
          ;
          pto::Shape<1, 1, 1, 1, 4096> v120 = pto::Shape<1, 1, 1, 1, 4096>();
          // pto: %x_norm_i8_inline3__rv_v2_local_pview
          ;
          pto::Stride<4096, 4096, 4096, 4096, 1> v121 = pto::Stride<4096, 4096, 4096, 4096, 1>();
          // pto: %x_norm_i8_inline3__rv_v2_local_pview
          ;
          GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v122 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v117 + (v118 + v98 * v119), v120, v121);
          pipe_barrier(PIPE_ALL);
          wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
          pipe_barrier(PIPE_MTE2);
          wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
          pto::comm::TPUT(v116, v122, v95);
          set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
          set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
          pipe_barrier(PIPE_ALL);
          __gm__ int8_t* v123 = PTOAS__GLOBAL_TENSOR_DATA(v116);
          PTOAS__DCCI_SINGLE_CACHE_LINE(v123);
          pipe_barrier(PIPE_ALL);
          dsb(DSB_DDR);
          // pto: %t__tile
          ;
          float v124 = (v5)[v78];
          wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
          v76.SetValue(v26, v124);
          // pto: %t__tmp_v120
          ;
          float v125 = v70.GetValue(v81);
          v76.SetValue(v21, v125);
          // pto: %42
          ;
          int64_t v126 = (int64_t) ((uint64_t) v78 * (uint64_t) v20);
          // pto: %43
          ;
          int64_t v127 = (int64_t) ((uint64_t) v126 + (uint64_t) v79);
          // pto: %44, %45
          ;
          float v128 = (float) ((int32_t) v127);
          v76.SetValue(v24, v128);
          set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
          // pto: %46
          ;
          int64_t v129 = (v11)[v24];
          // pto: %47, %48, %49, %50
          ;
          int64_t v130 = (v11)[(int64_t) ((uint64_t) ((int64_t) (int32_t) v129) + (uint64_t) v30)];
          // pto: %52
          ;
          int64_t v131 = (v11)[v101];
          // pto: %57
          ;
          const int64_t v132 = 1;
          // pto: %57
          ;
          const int64_t v133 = 1;
          // pto: %57
          ;
          const int64_t v134 = 1;
          // pto: %57
          ;
          int64_t v135 = v22 * v19;
          // pto: %57
          ;
          int64_t v136 = v134 * v135;
          // pto: %57
          ;
          pto::Shape<1, 1, 1, -1, -1> v137 = pto::Shape<1, 1, 1, -1, -1>(v132, v133, v134, v22, v19);
          // pto: %57
          ;
          pto::Stride<-1, -1, -1, -1, -1> v138 = pto::Stride<-1, -1, -1, -1, -1>(v133 * v136, v136, v135, v19, v21);
          // pto: %53, %54, %56, %57
          ;
          GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v139 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v6 + (int64_t) ((uint64_t) v131 - (uint64_t) v130) / v31, v137, v138);
          // pto: %recv_aux__ssa_v0_peer_pview
          ;
          __gm__ float* v140 = PTOAS__GLOBAL_TENSOR_DATA(v139);
          // pto: %recv_aux__ssa_v0_peer_pview
          ;
          const int64_t v141 = 0;
          // pto: %recv_aux__ssa_v0_peer_pview
          ;
          const int64_t v142 = 8;
          // pto: %recv_aux__ssa_v0_peer_pview
          ;
          pto::Shape<1, 1, 1, 1, 8> v143 = pto::Shape<1, 1, 1, 1, 8>();
          // pto: %recv_aux__ssa_v0_peer_pview
          ;
          pto::Stride<8, 8, 8, 8, 1> v144 = pto::Stride<8, 8, 8, 8, 1>();
          // pto: %recv_aux__ssa_v0_peer_pview
          ;
          GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<8, 8, 8, 8, 1>, pto::Layout::ND> v145 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<8, 8, 8, 8, 1>, pto::Layout::ND>(v140 + (v141 + v97 * v142), v143, v144);
          wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
          TSTORE(v145, v76);
          set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
          __gm__ float* v146 = PTOAS__GLOBAL_TENSOR_DATA(v145);
          PTOAS__DCCI_SINGLE_CACHE_LINE(v146);
          pipe_barrier(PIPE_ALL);
          dsb(DSB_DDR);
        };
      };
    };
  }
  // pto: %59
  int64_t v147 = (int64_t) v9;
  // pto: %60
  if (v58 != v147) {
    // pto: %61
    ;
    int64_t v148 = (v12)[v24];
    // pto: %62, %63, %64, %65
    ;
    int64_t v149 = (v12)[(int64_t) ((uint64_t) ((int64_t) (int32_t) v148) + (uint64_t) v30)];
    // pto: %66, %67
    ;
    int64_t v150 = (v12)[(int64_t) ((uint64_t) v58 + (uint64_t) v30)];
    // pto: %72
    ;
    const int64_t v151 = 1;
    // pto: %72
    ;
    const int64_t v152 = 1;
    // pto: %72
    ;
    const int64_t v153 = 1;
    // pto: %72
    ;
    int64_t v154 = v24 * v21;
    // pto: %72
    ;
    int64_t v155 = v153 * v154;
    // pto: %72
    ;
    pto::Shape<1, 1, 1, -1, -1> v156 = pto::Shape<1, 1, 1, -1, -1>(v151, v152, v153, v24, v21);
    // pto: %72
    ;
    pto::Stride<-1, -1, -1, -1, -1> v157 = pto::Stride<-1, -1, -1, -1, -1>(v152 * v155, v155, v154, v21, v24);
    // pto: %68, %69, %71, %72
    ;
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v158 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v7 + (int64_t) ((uint64_t) v150 - (uint64_t) v149) / v31, v156, v157);
    // pto: %73
    ;
    int64_t v159 = v147 < v26 ? v26 : v147;
    // pto: %data_arrived__ssa_v0_peer_pview
    ;
    __gm__ int32_t* v160 = PTOAS__GLOBAL_TENSOR_DATA(v158);
    // pto: %data_arrived__ssa_v0_peer_pview
    ;
    const int64_t v161 = 0;
    // pto: %data_arrived__ssa_v0_peer_pview
    ;
    pto::Shape<1, 1, 1, 1, 1> v162 = pto::Shape<1, 1, 1, 1, 1>();
    // pto: %data_arrived__ssa_v0_peer_pview
    ;
    pto::Stride<1, 1, 1, 1, 2> v163 = pto::Stride<1, 1, 1, 1, 2>();
    // pto: %data_arrived__ssa_v0_peer_pview
    ;
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN> v164 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN>(v160 + (v161 + v159), v162, v163);
    pto::comm::NotifyOp v165 = pto::comm::NotifyOp::AtomicAdd;
    pto::comm::TNOTIFY(v164, v32, v165);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}