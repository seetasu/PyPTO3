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

AICORE void dispatch_meta(__gm__ int32_t* v1, __gm__ int32_t* v2, __gm__ int32_t* v3, __gm__ int32_t* v4, __gm__ int32_t* v5, int32_t v6, int32_t v7, int32_t v8, __gm__ int64_t* v9, __gm__ int64_t* v10) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v11 = 0;
  // pto: %c8_index
  const int64_t v12 = 8;
  // pto: %c6_index
  const int64_t v13 = 6;
  // pto: %c1_index
  const int64_t v14 = 1;
  // pto: %c2_index
  const int64_t v15 = 2;
  // pto: %c32_index
  const int64_t v16 = 32;
  // pto: %c0_index
  const int64_t v17 = 0;
  // pto: %c0_i32
  const int32_t v18 = 0;
  // pto: %c4_index
  const int64_t v19 = 4;
  // pto: %c4_i64
  const int64_t v20 = 4;
  // pto: %c1_i32
  const int32_t v21 = 1;
  // pto: %indices_inline5__ssa_v0_view
  const int64_t v22 = 1;
  // pto: %indices_inline5__ssa_v0_view
  const int64_t v23 = 1;
  // pto: %indices_inline5__ssa_v0_view
  const int64_t v24 = 1;
  // pto: %indices_inline5__ssa_v0_view
  int64_t v25 = v12 * v13;
  // pto: %indices_inline5__ssa_v0_view
  int64_t v26 = v24 * v25;
  // pto: %indices_inline5__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v27 = pto::Shape<1, 1, 1, -1, -1>(v22, v23, v24, v12, v13);
  // pto: %indices_inline5__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v28 = pto::Stride<-1, -1, -1, -1, -1>(v23 * v26, v26, v25, v13, v14);
  // pto: %indices_inline5__ssa_v0_view
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v29 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v27, v28);
  // pto: %arrived__ssa_v0_view
  const int64_t v30 = 1;
  // pto: %arrived__ssa_v0_view
  const int64_t v31 = 1;
  // pto: %arrived__ssa_v0_view
  const int64_t v32 = 1;
  // pto: %arrived__ssa_v0_view
  int64_t v33 = v15 * v14;
  // pto: %arrived__ssa_v0_view
  int64_t v34 = v32 * v33;
  // pto: %arrived__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v35 = pto::Shape<1, 1, 1, -1, -1>(v30, v31, v32, v15, v14);
  // pto: %arrived__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v36 = pto::Stride<-1, -1, -1, -1, -1>(v31 * v34, v34, v33, v14, v15);
  // pto: %arrived__ssa_v0_view
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v37 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v3, v35, v36);
  // pto: %0
  int64_t v38 = (int64_t) v6;
  set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
  // pto: %1, %active_tokens_inline337__phi_v2
  int64_t v39 = v38 < v17 ? v17 : v38;
  // pto: %2, %active_tokens_inline337__phi_v4
  // pto: %cursor_inline338__ssa_v0
  int32_t v40[64];
  for (int64_t v41 = v17; v41 < v15; v41 += v14) {
    for (int64_t v42 = v17; v42 < v16; v42 += v14) {
      // pto: %3
      ;
      int64_t v43 = (int64_t) ((uint64_t) v41 * (uint64_t) v16);
      // pto: %4
      ;
      int64_t v44 = (int64_t) ((uint64_t) v43 + (uint64_t) v42);
      v40[v44] = v18;
    };
  }
  // pto: %meta_indices_tile_inline341__ssa_v0
  Tile<TileType::Vec, int32_t, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v45 = Tile<TileType::Vec, int32_t, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v13);
  // pto: %meta_indices_tile_inline341__ssa_v0
  uint64_t v46 = (uint64_t) v11;
  TASSIGN(v45, v46);
  // pto: %indices_inline5__ssa_v0_pview
  __gm__ int32_t* v47 = PTOAS__GLOBAL_TENSOR_DATA(v29);
  // pto: %indices_inline5__ssa_v0_pview
  pto::Shape<1, 1, 1, 8, 6> v48 = pto::Shape<1, 1, 1, 8, 6>();
  // pto: %indices_inline5__ssa_v0_pview
  pto::Stride<48, 48, 48, 6, 1> v49 = pto::Stride<48, 48, 48, 6, 1>();
  // pto: %indices_inline5__ssa_v0_pview
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 6>, pto::Stride<48, 48, 48, 6, 1>, pto::Layout::ND> v50 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 6>, pto::Stride<48, 48, 48, 6, 1>, pto::Layout::ND>(v47, v48, v49);
  TLOAD(v45, v50);
  set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
  for (int64_t v51 = v17; v51 < (v39 > v12 ? v12 : v39); v51 += v14) {
    for (int64_t v52 = v17; v52 < v13; v52 += v14) {
      // pto: %flat_offset_mul
      ;
      int64_t v53 = (int64_t) ((uint64_t) v51 * (uint64_t) v12);
      // pto: %flat_offset, %eid_inline344__ssa_v0
      ;
      int32_t v54 = v45.GetValue((int64_t) ((uint64_t) v53 + (uint64_t) v52));
      // pto: %5
      ;
      int64_t v55 = (int64_t) v54;
      // pto: %t__tmp_v117
      ;
      // pto: %t__tmp_v117
      ;
      int32_t v56;
      v56 = v40[v55];
      // pto: %14
      ;
      int64_t v57 = (int64_t) v56;
      // pto: %15
      ;
      int64_t v58 = (int64_t) ((uint64_t) v57 + (uint64_t) v14);
      // pto: %16
      ;
      v40[v55] = (int32_t) v58;
    };
  }
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  // pto: %meta_tile_inline347__ssa_v0
  Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v59 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v16);
  // pto: %meta_tile_inline347__ssa_v0
  uint64_t v60 = (uint64_t) v11;
  TASSIGN(v59, v60);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TEXPANDS(v59, v18);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  for (int64_t v61 = v17; v61 < v15; v61 += v14) {
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    for (int64_t v62 = v17; v62 < v16; v62 += v14) {
      // pto: %17
      ;
      int64_t v63 = (int64_t) ((uint64_t) v61 * (uint64_t) v16);
      // pto: %18
      ;
      int64_t v64 = (int64_t) ((uint64_t) v63 + (uint64_t) v62);
      // pto: %t__tmp_v118
      ;
      // pto: %t__tmp_v118
      ;
      int32_t v65;
      v65 = v40[v64];
      v59.SetValue(v62, v65);
    };
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    // pto: %21
    ;
    int64_t v66 = (v9)[v15];
    // pto: %22, %23, %24, %25
    ;
    int64_t v67 = (v9)[(int64_t) ((uint64_t) ((int64_t) (int32_t) v66) + (uint64_t) v19)];
    // pto: %26
    ;
    int64_t v68 = (int64_t) ((uint64_t) v61 + (uint64_t) v19);
    // pto: %27
    ;
    int64_t v69 = (v9)[v68];
    // pto: %32
    ;
    const int64_t v70 = 1;
    // pto: %32
    ;
    const int64_t v71 = 1;
    // pto: %32
    ;
    const int64_t v72 = 1;
    // pto: %32
    ;
    int64_t v73 = v15 * v16;
    // pto: %32
    ;
    int64_t v74 = v72 * v73;
    // pto: %32
    ;
    pto::Shape<1, 1, 1, -1, -1> v75 = pto::Shape<1, 1, 1, -1, -1>(v70, v71, v72, v15, v16);
    // pto: %32
    ;
    pto::Stride<-1, -1, -1, -1, -1> v76 = pto::Stride<-1, -1, -1, -1, -1>(v71 * v74, v74, v73, v16, v14);
    // pto: %28, %29, %31, %32
    ;
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v77 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2 + (int64_t) ((uint64_t) v69 - (uint64_t) v67) / v20, v75, v76);
    // pto: %my_rank__ssa_v0_idx
    ;
    int64_t v78 = (int64_t) v7;
    // pto: %33
    ;
    int64_t v79 = v78 < v17 ? v17 : v78;
    // pto: %recv_meta__ssa_v0_peer_pview
    ;
    __gm__ int32_t* v80 = PTOAS__GLOBAL_TENSOR_DATA(v77);
    // pto: %recv_meta__ssa_v0_peer_pview
    ;
    const int64_t v81 = 0;
    // pto: %recv_meta__ssa_v0_peer_pview
    ;
    const int64_t v82 = 32;
    // pto: %recv_meta__ssa_v0_peer_pview
    ;
    pto::Shape<1, 1, 1, 1, 32> v83 = pto::Shape<1, 1, 1, 1, 32>();
    // pto: %recv_meta__ssa_v0_peer_pview
    ;
    pto::Stride<32, 32, 32, 32, 1> v84 = pto::Stride<32, 32, 32, 32, 1>();
    // pto: %recv_meta__ssa_v0_peer_pview
    ;
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v85 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v80 + (v81 + v79 * v82), v83, v84);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v85, v59);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    __gm__ int32_t* v86 = PTOAS__GLOBAL_TENSOR_DATA(v85);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v86);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %35
    ;
    if (v61 != v78) {
      // pto: %36
      ;
      int64_t v87 = (v10)[v15];
      // pto: %37, %38, %39, %40
      ;
      int64_t v88 = (v10)[(int64_t) ((uint64_t) ((int64_t) (int32_t) v87) + (uint64_t) v19)];
      // pto: %42
      ;
      int64_t v89 = (v10)[v68];
      // pto: %47
      ;
      const int64_t v90 = 1;
      // pto: %47
      ;
      const int64_t v91 = 1;
      // pto: %47
      ;
      const int64_t v92 = 1;
      // pto: %47
      ;
      int64_t v93 = v15 * v14;
      // pto: %47
      ;
      int64_t v94 = v92 * v93;
      // pto: %47
      ;
      pto::Shape<1, 1, 1, -1, -1> v95 = pto::Shape<1, 1, 1, -1, -1>(v90, v91, v92, v15, v14);
      // pto: %47
      ;
      pto::Stride<-1, -1, -1, -1, -1> v96 = pto::Stride<-1, -1, -1, -1, -1>(v91 * v94, v94, v93, v14, v15);
      // pto: %43, %44, %46, %47
      ;
      GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v97 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v3 + (int64_t) ((uint64_t) v89 - (uint64_t) v88) / v20, v95, v96);
      // pto: %arrived__ssa_v0_peer_pview
      ;
      __gm__ int32_t* v98 = PTOAS__GLOBAL_TENSOR_DATA(v97);
      // pto: %arrived__ssa_v0_peer_pview
      ;
      const int64_t v99 = 0;
      // pto: %arrived__ssa_v0_peer_pview
      ;
      pto::Shape<1, 1, 1, 1, 1> v100 = pto::Shape<1, 1, 1, 1, 1>();
      // pto: %arrived__ssa_v0_peer_pview
      ;
      pto::Stride<1, 1, 1, 1, 2> v101 = pto::Stride<1, 1, 1, 1, 2>();
      // pto: %arrived__ssa_v0_peer_pview
      ;
      GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN> v102 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN>(v98 + (v99 + v79), v100, v101);
      pto::comm::NotifyOp v103 = pto::comm::NotifyOp::AtomicAdd;
      pto::comm::TNOTIFY(v102, v21, v103);
    };
  }
  set_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
  for (int64_t v104 = v17; v104 < v15; v104 += v14) {
    // pto: %50, %51
    ;
    if (v104 != (int64_t) v7) {
      // pto: %52
      ;
      int64_t v105 = v104 < v17 ? v17 : v104;
      // pto: %arrived__ssa_v0_local_pview
      ;
      __gm__ int32_t* v106 = PTOAS__GLOBAL_TENSOR_DATA(v37);
      // pto: %arrived__ssa_v0_local_pview
      ;
      const int64_t v107 = 0;
      // pto: %arrived__ssa_v0_local_pview
      ;
      pto::Shape<1, 1, 1, 1, 1> v108 = pto::Shape<1, 1, 1, 1, 1>();
      // pto: %arrived__ssa_v0_local_pview
      ;
      pto::Stride<1, 1, 1, 1, 2> v109 = pto::Stride<1, 1, 1, 1, 2>();
      // pto: %arrived__ssa_v0_local_pview
      ;
      GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN> v110 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN>(v106 + (v107 + v105), v108, v109);
      pto::comm::WaitCmp v111 = pto::comm::WaitCmp::GE;
      pto::comm::TWAIT(v110, v8, v111);
    };
  }
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
  for (int64_t v112 = v17; v112 < v16; v112 += v14) {
    // pto: %acc_inline352__rv_v2
    ;
    int32_t v113;
    v113 = v18;
    for (int64_t v114 = v17; v114 < v15; v114 += v14) {
      // pto: %53
      ;
      int64_t v115 = (int64_t) ((uint64_t) v114 * (uint64_t) v16);
      // pto: %54
      ;
      int64_t v116 = (int64_t) ((uint64_t) v115 + (uint64_t) v112);
      // pto: %count_inline354__tile
      ;
      int32_t v117 = (v2)[v116];
      (v4)[v116] = v117;
      // pto: %57
      ;
      uint32_t v118 = (uint32_t) v113;
      // pto: %57
      ;
      v113 = (int32_t) (v118 + (uint32_t) v117);
    };
    (v5)[v112] = v113;
  }
  wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
  pipe_barrier(PIPE_ALL);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  dsb((mem_dsb_t)0);
  #endif // __DAV_VEC__

  pipe_barrier(PIPE_ALL);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  dsb((mem_dsb_t)0);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}