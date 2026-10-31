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

AICORE void sh_h_tile_i8_init(__gm__ int8_t* v1) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v2 = 0;
  // pto: %c16_index
  const int64_t v3 = 16;
  // pto: %c2048_index
  const int64_t v4 = 2048;
  // pto: %c1_index
  const int64_t v5 = 1;
  // pto: %cst_4
  const half v6 = 0.0f;
  // pto: %h_tile_i8_inline291__ssa_v0_view
  const int64_t v7 = 1;
  // pto: %h_tile_i8_inline291__ssa_v0_view
  const int64_t v8 = 1;
  // pto: %h_tile_i8_inline291__ssa_v0_view
  const int64_t v9 = 1;
  // pto: %h_tile_i8_inline291__ssa_v0_view
  int64_t v10 = v3 * v4;
  // pto: %h_tile_i8_inline291__ssa_v0_view
  int64_t v11 = v9 * v10;
  // pto: %h_tile_i8_inline291__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v12 = pto::Shape<1, 1, 1, -1, -1>(v7, v8, v9, v3, v4);
  // pto: %h_tile_i8_inline291__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v13 = pto::Stride<-1, -1, -1, -1, -1>(v8 * v11, v11, v10, v4, v5);
  // pto: %h_tile_i8_inline291__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v14 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v12, v13);
  // pto: %t__tile
  Tile<TileType::Vec, half, 16, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v15 = Tile<TileType::Vec, half, 16, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v3, v4);
  // pto: %t__tile
  uint64_t v16 = (uint64_t) v2;
  TASSIGN(v15, v16);
  TEXPANDS(v15, v6);
  // pto: %0
  Tile<TileType::Vec, int8_t, 16, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v17 = Tile<TileType::Vec, int8_t, 16, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v3, v4);
  // pto: %0
  uint64_t v18 = (uint64_t) v2;
  TASSIGN(v17, v18);
  pipe_barrier(PIPE_V);
  RoundMode v19 = RoundMode::CAST_TRUNC;
  SaturationMode v20 = SaturationMode::ON;
  TCVT(v17, v15, v19, v20);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  // pto: %h_tile_i8_inline291__ssa_v0_pview
  __gm__ int8_t* v21 = PTOAS__GLOBAL_TENSOR_DATA(v14);
  // pto: %h_tile_i8_inline291__ssa_v0_pview
  pto::Shape<1, 1, 1, 16, 2048> v22 = pto::Shape<1, 1, 1, 16, 2048>();
  // pto: %h_tile_i8_inline291__ssa_v0_pview
  pto::Stride<32768, 32768, 32768, 2048, 1> v23 = pto::Stride<32768, 32768, 32768, 2048, 1>();
  // pto: %h_tile_i8_inline291__ssa_v0_pview
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<32768, 32768, 32768, 2048, 1>, pto::Layout::ND> v24 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<32768, 32768, 32768, 2048, 1>, pto::Layout::ND>(v21, v22, v23);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v24, v17);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}