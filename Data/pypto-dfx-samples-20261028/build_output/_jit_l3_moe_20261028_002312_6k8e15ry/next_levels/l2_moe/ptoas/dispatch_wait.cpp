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

AICORE void dispatch_wait(__gm__ int32_t* v1, __gm__ int32_t* v2, int32_t v3, int32_t v4, __gm__ int64_t* v5) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c1_index
  const int64_t v6 = 1;
  // pto: %c2_index
  const int64_t v7 = 2;
  // pto: %c0_index
  const int64_t v8 = 0;
  // pto: %c16_index
  const int64_t v9 = 16;
  // pto: %data_arrived__ssa_v0_view
  const int64_t v10 = 1;
  // pto: %data_arrived__ssa_v0_view
  const int64_t v11 = 1;
  // pto: %data_arrived__ssa_v0_view
  const int64_t v12 = 1;
  // pto: %data_arrived__ssa_v0_view
  int64_t v13 = v7 * v6;
  // pto: %data_arrived__ssa_v0_view
  int64_t v14 = v12 * v13;
  // pto: %data_arrived__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v15 = pto::Shape<1, 1, 1, -1, -1>(v10, v11, v12, v7, v6);
  // pto: %data_arrived__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v16 = pto::Stride<-1, -1, -1, -1, -1>(v11 * v14, v14, v13, v6, v7);
  // pto: %data_arrived__ssa_v0_view
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v17 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v2, v15, v16);
  for (int64_t v18 = v8; v18 < v7; v18 += v6) {
    // pto: %0, %1
    ;
    if (v18 != (int64_t) v3) {
      // pto: %2
      ;
      int64_t v19 = v18 < v8 ? v8 : v18;
      // pto: %data_arrived__ssa_v0_local_pview
      ;
      __gm__ int32_t* v20 = PTOAS__GLOBAL_TENSOR_DATA(v17);
      // pto: %data_arrived__ssa_v0_local_pview
      ;
      const int64_t v21 = 0;
      // pto: %data_arrived__ssa_v0_local_pview
      ;
      pto::Shape<1, 1, 1, 1, 1> v22 = pto::Shape<1, 1, 1, 1, 1>();
      // pto: %data_arrived__ssa_v0_local_pview
      ;
      pto::Stride<1, 1, 1, 1, 2> v23 = pto::Stride<1, 1, 1, 1, 2>();
      // pto: %data_arrived__ssa_v0_local_pview
      ;
      GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN> v24 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN>(v20 + (v21 + v19), v22, v23);
      // pto: %3, %4
      ;
      int64_t v25 = (int64_t) ((uint64_t) ((int64_t) v4) * (uint64_t) v9);
      // pto: %5
      ;
      int32_t v26 = (int32_t) v25;
      pto::comm::WaitCmp v27 = pto::comm::WaitCmp::GE;
      pto::comm::TWAIT(v24, v26, v27);
    };
  }
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}