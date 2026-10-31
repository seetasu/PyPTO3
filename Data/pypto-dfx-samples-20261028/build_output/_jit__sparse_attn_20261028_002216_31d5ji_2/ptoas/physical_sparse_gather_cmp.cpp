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

AICORE void physical_sparse_gather_cmp(__gm__ bfloat16_t* v1, __gm__ int32_t* v2, __gm__ int32_t* v3, __gm__ int32_t* v4, __gm__ bfloat16_t* v5, __gm__ int32_t* v6, __gm__ bfloat16_t* v7, int64_t v8, int64_t v9, int32_t v10, int32_t v11, int64_t v12, int64_t v13, int64_t v14, int64_t v15, int64_t v16, int64_t v17, int64_t v18, int32_t v19, int32_t v20) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c2048_i64
  const int64_t v21 = 2048;
  // pto: %c133120_i64
  const int64_t v22 = 133120;
  // pto: %c141312_i64
  const int64_t v23 = 141312;
  // pto: %c0_i64
  const int64_t v24 = 0;
  // pto: %c512_i64
  const int64_t v25 = 512;
  const int64_t v26 = 1024;
  // pto: %c1536_i64
  const int64_t v27 = 1536;
  // pto: %c149504_i64
  const int64_t v28 = 149504;
  // pto: %c512_index
  const int64_t v29 = 512;
  // pto: %c1_index
  const int64_t v30 = 1;
  // pto: %c8_index
  const int64_t v31 = 8;
  // pto: %c8192_index
  const int64_t v32 = 8192;
  // pto: %c4_index
  const int64_t v33 = 4;
  // pto: %c128_index
  const int64_t v34 = 128;
  // pto: %c0_index
  const int64_t v35 = 0;
  const int64_t v36 = 640;
  // pto: %c1_i32
  const int32_t v37 = 1;
  // pto: %c0_i32
  const int32_t v38 = 0;
  // pto: %c7_i32
  const int32_t v39 = 7;
  // pto: %c128_i32
  const int32_t v40 = 128;
  // pto: %cn1_i32
  const int32_t v41 = -1;
  // pto: %c8192_i32
  const int32_t v42 = 8192;
  // pto: %c8191_i32
  const int32_t v43 = 8191;
  // pto: %c192_index
  const int64_t v44 = 192;
  // pto: %cst_24
  const bfloat16_t v45 = 0.0f;
  // pto: %c16_index
  const int64_t v46 = 16;
  // pto: %c64_index
  const int64_t v47 = 64;
  const half v48 = 0.0f;
  // pto: %source_view_inline433__ssa_v0_view
  const int64_t v49 = 1;
  // pto: %source_view_inline433__ssa_v0_view
  const int64_t v50 = 1;
  // pto: %source_view_inline433__ssa_v0_view
  const int64_t v51 = 1;
  // pto: %source_view_inline433__ssa_v0_view
  int64_t v52 = (int64_t) v14;
  // pto: %source_view_inline433__ssa_v0_view
  int64_t v53 = v52 * v29;
  // pto: %source_view_inline433__ssa_v0_view
  int64_t v54 = v51 * v53;
  // pto: %source_view_inline433__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v55 = pto::Shape<1, 1, 1, -1, -1>(v49, v50, v51, v52, v29);
  // pto: %source_view_inline433__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v56 = pto::Stride<-1, -1, -1, -1, -1>(v50 * v54, v54, v53, v29, v30);
  // pto: %source_view_inline433__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v57 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v55, v56);
  // pto: %cmp_indices__ssa_v0_view
  const int64_t v58 = 1;
  // pto: %cmp_indices__ssa_v0_view
  const int64_t v59 = 1;
  // pto: %cmp_indices__ssa_v0_view
  const int64_t v60 = 1;
  // pto: %cmp_indices__ssa_v0_view
  int64_t v61 = (int64_t) v15;
  // pto: %cmp_indices__ssa_v0_view
  int64_t v62 = v61 * v29;
  // pto: %cmp_indices__ssa_v0_view
  int64_t v63 = v60 * v62;
  // pto: %cmp_indices__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v64 = pto::Shape<1, 1, 1, -1, -1>(v58, v59, v60, v61, v29);
  // pto: %cmp_indices__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v65 = pto::Stride<-1, -1, -1, -1, -1>(v59 * v63, v63, v62, v29, v30);
  // pto: %cmp_indices__ssa_v0_view
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v66 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v64, v65);
  // pto: %cmp_block_table_view_inline458__ssa_v0_view
  const int64_t v67 = 1;
  // pto: %cmp_block_table_view_inline458__ssa_v0_view
  const int64_t v68 = 1;
  // pto: %cmp_block_table_view_inline458__ssa_v0_view
  const int64_t v69 = 1;
  // pto: %cmp_block_table_view_inline458__ssa_v0_view
  int64_t v70 = v30 * v32;
  // pto: %cmp_block_table_view_inline458__ssa_v0_view
  int64_t v71 = v69 * v70;
  // pto: %cmp_block_table_view_inline458__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v72 = pto::Shape<1, 1, 1, -1, -1>(v67, v68, v69, v30, v32);
  // pto: %cmp_block_table_view_inline458__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v73 = pto::Stride<-1, -1, -1, -1, -1>(v68 * v71, v71, v70, v32, v30);
  // pto: %cmp_block_table_view_inline458__ssa_v0_view
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v74 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v72, v73);
  // pto: %cmp_kv_flat_inline439__ssa_v0_view
  const int64_t v75 = 1;
  // pto: %cmp_kv_flat_inline439__ssa_v0_view
  const int64_t v76 = 1;
  // pto: %cmp_kv_flat_inline439__ssa_v0_view
  const int64_t v77 = 1;
  // pto: %cmp_kv_flat_inline439__ssa_v0_view
  int64_t v78 = (int64_t) v16;
  // pto: %cmp_kv_flat_inline439__ssa_v0_view
  int64_t v79 = v78 * v29;
  // pto: %cmp_kv_flat_inline439__ssa_v0_view
  int64_t v80 = v77 * v79;
  // pto: %cmp_kv_flat_inline439__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v81 = pto::Shape<1, 1, 1, -1, -1>(v75, v76, v77, v78, v29);
  // pto: %cmp_kv_flat_inline439__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v82 = pto::Stride<-1, -1, -1, -1, -1>(v76 * v80, v80, v79, v29, v30);
  // pto: %cmp_kv_flat_inline439__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v83 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v5, v81, v82);
  // pto: %cmp_kv__ssa_v0_s0
  int64_t v84 = (int64_t) ((uint64_t) v18 * (uint64_t) v29);
  // pto: %cmp_kv__ssa_v0_view
  const int64_t v85 = 1;
  // pto: %cmp_kv__ssa_v0_view
  int64_t v86 = (int64_t) v17;
  // pto: %cmp_kv__ssa_v0_view
  pto::Shape<1, -1, -1, -1, -1> v87 = pto::Shape<1, -1, -1, -1, -1>(v85, v86, (int64_t) v18, v30, v29);
  // pto: %cmp_kv__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v88 = pto::Stride<-1, -1, -1, -1, -1>(v86 * v84, v84, v29, v29, v30);
  // pto: %cmp_kv__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v89 = GlobalTensor<bfloat16_t, pto::Shape<1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v7, v87, v88);
  // pto: %stage_inline532__phi_v7
  Tile<TileType::Vec, bfloat16_t, 128, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v90 = Tile<TileType::Vec, bfloat16_t, 128, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v29);
  // pto: %stage_inline532__phi_v7
  uint64_t v91 = (uint64_t) v21;
  TASSIGN(v90, v91);
  // pto: %stage_inline532__phi_v6
  Tile<TileType::Vec, bfloat16_t, 128, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v92 = Tile<TileType::Vec, bfloat16_t, 128, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v29);
  // pto: %stage_inline532__phi_v6
  uint64_t v93 = (uint64_t) v21;
  TASSIGN(v92, v93);
  // pto: %stage_inline532__phi_v5
  Tile<TileType::Vec, bfloat16_t, 128, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v94 = Tile<TileType::Vec, bfloat16_t, 128, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v29);
  // pto: %stage_inline532__phi_v5
  uint64_t v95 = (uint64_t) v21;
  TASSIGN(v94, v95);
  // pto: %stage_inline532__phi_v4
  Tile<TileType::Vec, bfloat16_t, 128, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v96 = Tile<TileType::Vec, bfloat16_t, 128, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v29);
  // pto: %stage_inline532__phi_v4
  uint64_t v97 = (uint64_t) v21;
  TASSIGN(v96, v97);
  // pto: %gather_block_inline460__ssa_v0
  int64_t v98 = (int64_t) v19;
  // pto: %1
  int64_t v99 = v98 / v33;
  // pto: %2
  int64_t v100 = (int64_t) ((uint64_t) v8 - (uint64_t) v30);
  // pto: %3
  int64_t v101 = (int64_t) ((uint64_t) v100 - (uint64_t) v99);
  // pto: %4
  int64_t v102 = (int64_t) ((uint64_t) v99 * (uint64_t) v33);
  // pto: %5
  int64_t v103 = (int64_t) ((uint64_t) v98 - (uint64_t) v102);
  // pto: %6
  int64_t v104 = (int64_t) ((uint64_t) v103 + (uint64_t) v30);
  // pto: %7
  int64_t v105 = (int64_t) ((uint64_t) v101 * (uint64_t) v31);
  // pto: %8
  int64_t v106 = (int64_t) ((uint64_t) v104 * (uint64_t) v34);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
  set_flag(PIPE_S, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_S, PIPE_V, EVENT_ID1);
  set_flag(PIPE_S, PIPE_V, EVENT_ID2);
  set_flag(PIPE_S, PIPE_V, EVENT_ID3);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  for (int64_t v107 = v35; v107 < v31; v107 += v30) {
    // pto: %9
    ;
    int64_t v108 = (int64_t) ((uint64_t) v105 + (uint64_t) v107);
    // pto: %10
    ;
    if (v108 < v9) {
      // pto: %flat_offset_mul
      ;
      int64_t v109 = (int64_t) ((uint64_t) v108 * (uint64_t) v31);
      // pto: %flat_offset, %gather_block_valid_inline467__tile
      ;
      int32_t v110 = (v2)[(int64_t) ((uint64_t) v109 + (uint64_t) v104)];
      // pto: %11, %12
      ;
      if ((int64_t) v110 > v35) {
        // pto: %13
        ;
        int64_t v111 = (int64_t) ((uint64_t) v108 * (uint64_t) v36);
        // pto: %14
        ;
        int64_t v112 = (int64_t) ((uint64_t) v111 + (uint64_t) v106);
        // pto: %15, %16
        ;
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID1);
        if ((int64_t) v10 == v34) {
          // pto: %17
          ;
          int64_t v113 = (int64_t) ((uint64_t) v106 - (uint64_t) v34);
          int64_t v114 = (int64_t) ((uint64_t) v36 - (uint64_t) v106);
          // pto: %19
          ;
          int64_t v115 = v114 < v34 ? v114 : v34;
          // pto: %gather_slots_inline470__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v116 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v115);
          // pto: %gather_slots_inline470__ssa_v0
          ;
          uint64_t v117 = (uint64_t) v21;
          TASSIGN(v116, v117);
          // pto: %20
          ;
          int64_t v118 = v108 < v35 ? v35 : v108;
          // pto: %21
          ;
          int64_t v119 = v113 < v35 ? v35 : v113;
          // pto: %cmp_indices__ssa_v0_pview
          ;
          const int64_t v120 = 0;
          // pto: %cmp_indices__ssa_v0_pview
          ;
          __gm__ int32_t* v121 = PTOAS__GLOBAL_TENSOR_DATA(v66);
          // pto: %cmp_indices__ssa_v0_pview
          ;
          const int64_t v122 = 1;
          // pto: %cmp_indices__ssa_v0_pview
          ;
          const int64_t v123 = 1;
          // pto: %cmp_indices__ssa_v0_pview
          ;
          const int64_t v124 = 1;
          // pto: %cmp_indices__ssa_v0_pview
          ;
          int64_t v125 = v30 * v29;
          // pto: %cmp_indices__ssa_v0_pview
          ;
          int64_t v126 = v124 * v125;
          // pto: %cmp_indices__ssa_v0_pview
          ;
          pto::Shape<1, 1, 1, 1, -1> v127 = pto::Shape<1, 1, 1, 1, -1>(v122, v123, v124, v30, v115);
          // pto: %cmp_indices__ssa_v0_pview
          ;
          pto::Stride<-1, -1, -1, -1, -1> v128 = pto::Stride<-1, -1, -1, -1, -1>(v123 * v126, v126, v125, v29, v30);
          // pto: %cmp_indices__ssa_v0_pview
          ;
          GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v129 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v121 + (v120 + v118 * v29 + v119 * v30), v127, v128);
          wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
          wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
          wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
          TLOAD(v116, v129);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
          // pto: %gather_slots_v1_inline471__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v130 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v30, v34);
          // pto: %gather_slots_v1_inline471__ssa_v0
          ;
          uint64_t v131 = (uint64_t) v21;
          TASSIGN(v130, v131);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
          TFILLPAD<pto::TFillPadMode::InPlace>(v130, v116);
          // pto: %gather_slots_plus_one_inline472__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v132 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v30, v34);
          // pto: %gather_slots_plus_one_inline472__ssa_v0
          ;
          uint64_t v133 = (uint64_t) v22;
          TASSIGN(v132, v133);
          pipe_barrier(PIPE_V);
          wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
          TADDS(v132, v130, v37);
          // pto: %gather_slots_positive_inline473__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v134 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v30, v34);
          // pto: %gather_slots_positive_inline473__ssa_v0
          ;
          uint64_t v135 = (uint64_t) v22;
          TASSIGN(v134, v135);
          pipe_barrier(PIPE_V);
          TMAXS(v134, v132, v38);
          // pto: %gather_slot_valid_inline474__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v136 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v30, v34);
          // pto: %gather_slot_valid_inline474__ssa_v0
          ;
          uint64_t v137 = (uint64_t) v22;
          TASSIGN(v136, v137);
          pipe_barrier(PIPE_V);
          TMINS(v136, v134, v37);
          // pto: %gather_slots_nonnegative_inline475__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v138 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v30, v34);
          // pto: %gather_slots_nonnegative_inline475__ssa_v0
          ;
          uint64_t v139 = (uint64_t) v21;
          TASSIGN(v138, v139);
          TMAXS(v138, v130, v38);
          // pto: %gather_logical_blocks_inline476__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v140 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v30, v34);
          // pto: %gather_logical_blocks_inline476__ssa_v0
          ;
          uint64_t v141 = (uint64_t) v23;
          TASSIGN(v140, v141);
          pipe_barrier(PIPE_V);
          TSHRS(v140, v138, v39);
          // pto: %gather_page_starts_inline477__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v142 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v30, v34);
          // pto: %gather_page_starts_inline477__ssa_v0
          ;
          uint64_t v143 = (uint64_t) v24;
          TASSIGN(v142, v143);
          pipe_barrier(PIPE_V);
          wait_flag(PIPE_S, PIPE_V, EVENT_ID2);
          wait_flag(PIPE_S, PIPE_V, EVENT_ID3);
          TMULS(v142, v140, v40);
          // pto: %gather_rows_inline478__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v144 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v30, v34);
          // pto: %gather_rows_inline478__ssa_v0
          ;
          uint64_t v145 = (uint64_t) v24;
          TASSIGN(v144, v145);
          pipe_barrier(PIPE_V);
          TSUB(v144, v138, v142);
          // pto: %gather_negative_blocks_inline479__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v146 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v30, v34);
          // pto: %gather_negative_blocks_inline479__ssa_v0
          ;
          uint64_t v147 = (uint64_t) v21;
          TASSIGN(v146, v147);
          pipe_barrier(PIPE_V);
          TMULS(v146, v140, v41);
          // pto: %gather_table_distance_inline480__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v148 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v30, v34);
          // pto: %gather_table_distance_inline480__ssa_v0
          ;
          uint64_t v149 = (uint64_t) v21;
          TASSIGN(v148, v149);
          pipe_barrier(PIPE_V);
          TADDS(v148, v146, v42);
          // pto: %gather_table_nonnegative_inline481__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v150 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v30, v34);
          // pto: %gather_table_nonnegative_inline481__ssa_v0
          ;
          uint64_t v151 = (uint64_t) v21;
          TASSIGN(v150, v151);
          pipe_barrier(PIPE_V);
          TMAXS(v150, v148, v38);
          // pto: %gather_table_valid_inline482__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v152 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v30, v34);
          // pto: %gather_table_valid_inline482__ssa_v0
          ;
          uint64_t v153 = (uint64_t) v25;
          TASSIGN(v152, v153);
          pipe_barrier(PIPE_V);
          TMINS(v152, v150, v37);
          // pto: %gather_nonnegative_blocks_inline483__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v154 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v30, v34);
          // pto: %gather_nonnegative_blocks_inline483__ssa_v0
          ;
          uint64_t v155 = (uint64_t) v21;
          TASSIGN(v154, v155);
          pipe_barrier(PIPE_V);
          TMAXS(v154, v140, v38);
          // pto: %gather_table_indices_inline484__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v156 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v30, v34);
          // pto: %gather_table_indices_inline484__ssa_v0
          ;
          uint64_t v157 = (uint64_t) v23;
          TASSIGN(v156, v157);
          pipe_barrier(PIPE_V);
          TMINS(v156, v154, v43);
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
          // pto: %gather_table_inline487__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v158 = Tile<TileType::Vec, int32_t, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v32);
          // pto: %gather_table_inline487__ssa_v0
          ;
          uint64_t v159 = (uint64_t) v21;
          TASSIGN(v158, v159);
          // pto: %cmp_block_table_view_inline458__ssa_v0_pview
          ;
          __gm__ int32_t* v160 = PTOAS__GLOBAL_TENSOR_DATA(v74);
          // pto: %cmp_block_table_view_inline458__ssa_v0_pview
          ;
          pto::Shape<1, 1, 1, 1, 8192> v161 = pto::Shape<1, 1, 1, 1, 8192>();
          // pto: %cmp_block_table_view_inline458__ssa_v0_pview
          ;
          pto::Stride<8192, 8192, 8192, 8192, 1> v162 = pto::Stride<8192, 8192, 8192, 8192, 1>();
          // pto: %cmp_block_table_view_inline458__ssa_v0_pview
          ;
          GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 8192>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND> v163 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 8192>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND>(v160, v161, v162);
          wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
          TLOAD(v158, v163);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
          // pto: %gather_table_tmp_inline488__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v164 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v34);
          // pto: %gather_table_tmp_inline488__ssa_v0
          ;
          uint64_t v165 = (uint64_t) v26;
          TASSIGN(v164, v165);
          TEXPANDS(v164, v38);
          // pto: %gather_pages_inline486__ssa_v1
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v166 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v34);
          // pto: %gather_pages_inline486__ssa_v1
          ;
          uint64_t v167 = (uint64_t) v27;
          TASSIGN(v166, v167);
          pipe_barrier(PIPE_V);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
          TGATHER(v166, v158, v156, v164);
          // pto: %gather_pages_plus_one_inline489__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v168 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v34);
          // pto: %gather_pages_plus_one_inline489__ssa_v0
          ;
          uint64_t v169 = (uint64_t) v21;
          TASSIGN(v168, v169);
          pipe_barrier(PIPE_V);
          TADDS(v168, v166, v37);
          // pto: %gather_pages_positive_inline490__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v170 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v34);
          // pto: %gather_pages_positive_inline490__ssa_v0
          ;
          uint64_t v171 = (uint64_t) v21;
          TASSIGN(v170, v171);
          pipe_barrier(PIPE_V);
          TMAXS(v170, v168, v38);
          // pto: %gather_page_nonnegative_inline491__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v172 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v34);
          // pto: %gather_page_nonnegative_inline491__ssa_v0
          ;
          uint64_t v173 = (uint64_t) v21;
          TASSIGN(v172, v173);
          pipe_barrier(PIPE_V);
          TMINS(v172, v170, v37);
          // pto: %gather_negative_pages_inline492__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v174 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v34);
          // pto: %gather_negative_pages_inline492__ssa_v0
          ;
          uint64_t v175 = (uint64_t) v23;
          TASSIGN(v174, v175);
          TMULS(v174, v166, v41);
          // pto: %gather_page_distance_inline493__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v176 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v34);
          // pto: %gather_page_distance_inline493__ssa_v0
          ;
          uint64_t v177 = (uint64_t) v23;
          TASSIGN(v176, v177);
          pipe_barrier(PIPE_V);
          TADDS(v176, v174, v11);
          // pto: %gather_page_distance_positive_inline494__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v178 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v34);
          // pto: %gather_page_distance_positive_inline494__ssa_v0
          ;
          uint64_t v179 = (uint64_t) v23;
          TASSIGN(v178, v179);
          pipe_barrier(PIPE_V);
          TMAXS(v178, v176, v38);
          // pto: %gather_page_in_range_inline495__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v180 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v34);
          // pto: %gather_page_in_range_inline495__ssa_v0
          ;
          uint64_t v181 = (uint64_t) v23;
          TASSIGN(v180, v181);
          pipe_barrier(PIPE_V);
          TMINS(v180, v178, v37);
          // pto: %gather_logical_valid_inline496__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v182 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v30, v34);
          // pto: %gather_logical_valid_inline496__ssa_v0
          ;
          uint64_t v183 = (uint64_t) v22;
          TASSIGN(v182, v183);
          TMUL(v182, v136, v152);
          // pto: %gather_physical_valid_inline497__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v184 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v34);
          // pto: %gather_physical_valid_inline497__ssa_v0
          ;
          uint64_t v185 = (uint64_t) v21;
          TASSIGN(v184, v185);
          pipe_barrier(PIPE_V);
          TMUL(v184, v172, v180);
          // pto: %gather_valid_inline498__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v186 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v30, v34);
          // pto: %gather_valid_inline498__ssa_v0
          ;
          uint64_t v187 = (uint64_t) v25;
          TASSIGN(v186, v187);
          pipe_barrier(PIPE_V);
          TMUL(v186, v182, v184);
          // pto: %gather_physical_page_starts_inline499__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v188 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v34);
          // pto: %gather_physical_page_starts_inline499__ssa_v0
          ;
          uint64_t v189 = (uint64_t) v21;
          TASSIGN(v188, v189);
          pipe_barrier(PIPE_V);
          TMULS(v188, v166, v40);
          // pto: %gather_physical_rows_inline500__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v190 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v34);
          // pto: %gather_physical_rows_inline500__ssa_v0
          ;
          uint64_t v191 = (uint64_t) v21;
          TASSIGN(v190, v191);
          pipe_barrier(PIPE_V);
          TADD(v190, v188, v144);
          // pto: %gather_rows_plus_one_inline501__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v192 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v34);
          // pto: %gather_rows_plus_one_inline501__ssa_v0
          ;
          uint64_t v193 = (uint64_t) v21;
          TASSIGN(v192, v193);
          pipe_barrier(PIPE_V);
          TADDS(v192, v190, v37);
          // pto: %gather_masked_rows_inline502__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v194 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v30, v34);
          // pto: %gather_masked_rows_inline502__ssa_v0
          ;
          uint64_t v195 = (uint64_t) v21;
          TASSIGN(v194, v195);
          pipe_barrier(PIPE_V);
          TMUL(v194, v186, v192);
          // pto: %gather_vector_rows_inline503__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v196 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v30, v34);
          // pto: %gather_vector_rows_inline503__ssa_v0
          ;
          uint64_t v197 = (uint64_t) v21;
          TASSIGN(v196, v197);
          pipe_barrier(PIPE_V);
          TSUBS(v196, v194, v37);
          // pto: %gather_safe_rows_inline504__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v198 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v30, v34);
          // pto: %gather_safe_rows_inline504__ssa_v0
          ;
          uint64_t v199 = (uint64_t) v24;
          TASSIGN(v198, v199);
          pipe_barrier(PIPE_V);
          TMAXS(v198, v196, v38);
          // pto: %gather_reduce_ones_inline505__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v200 = Tile<TileType::Vec, int32_t, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v34);
          // pto: %gather_reduce_ones_inline505__ssa_v0
          ;
          uint64_t v201 = (uint64_t) v21;
          TASSIGN(v200, v201);
          pipe_barrier(PIPE_V);
          TEXPANDS(v200, v37);
          // pto: %gather_valid_matrix_inline506__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v202 = Tile<TileType::Vec, int32_t, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v34);
          // pto: %gather_valid_matrix_inline506__ssa_v0
          ;
          uint64_t v203 = (uint64_t) v22;
          TASSIGN(v202, v203);
          pipe_barrier(PIPE_V);
          TCOLEXPANDMUL(v202, v200, v186);
          // pto: %gather_reduce_tmp_inline507__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v204 = Tile<TileType::Vec, int32_t, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v34);
          // pto: %gather_reduce_tmp_inline507__ssa_v0
          ;
          uint64_t v205 = (uint64_t) v23;
          TASSIGN(v204, v205);
          TEXPANDS(v204, v38);
          // pto: %gather_count_tile_inline508__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v206 = Tile<TileType::Vec, int32_t, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
          // pto: %gather_count_tile_inline508__ssa_v0
          ;
          uint64_t v207 = (uint64_t) v27;
          TASSIGN(v206, v207);
          pipe_barrier(PIPE_V);
          TROWSUM(v206, v202, v204);
          set_flag(PIPE_V, PIPE_S, EVENT_ID1);
          wait_flag(PIPE_V, PIPE_S, EVENT_ID1);
          // pto: %gather_count_inline509__ssa_v0
          ;
          int32_t v208 = v206.GetValue(v35);
          set_flag(PIPE_S, PIPE_V, EVENT_ID4);
          // pto: %22
          ;
          int64_t v209 = (int64_t) v208;
          // pto: %gather_lane_ends_inline511__ci_tmp_v0
          ;
          Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v210 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v44);
          // pto: %gather_lane_ends_inline511__ci_tmp_v0
          ;
          uint64_t v211 = (uint64_t) v22;
          TASSIGN(v210, v211);
          // pto: %gather_lane_ends_inline511__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v212 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v34);
          // pto: %gather_lane_ends_inline511__ssa_v0
          ;
          uint64_t v213 = (uint64_t) v27;
          TASSIGN(v212, v213);
          // pto: %ci_tmp_view
          ;
          Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v214;
          TRESHAPE(v214, v210);
          // pto: %ci_dst_view
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v215;
          TRESHAPE(v215, v212);
          wait_flag(PIPE_S, PIPE_V, EVENT_ID4);
          TCI<Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v215, v37, v214);
          // pto: %gather_valid_ends_inline512__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v216 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v34);
          // pto: %gather_valid_ends_inline512__ssa_v0
          ;
          uint64_t v217 = (uint64_t) v22;
          TASSIGN(v216, v217);
          pipe_barrier(PIPE_V);
          TMUL(v216, v212, v186);
          // pto: %gather_end_matrix_inline513__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v218 = Tile<TileType::Vec, int32_t, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v34);
          // pto: %gather_end_matrix_inline513__ssa_v0
          ;
          uint64_t v219 = (uint64_t) v21;
          TASSIGN(v218, v219);
          pipe_barrier(PIPE_V);
          TCOLEXPANDMUL(v218, v200, v216);
          // pto: %gather_end_tile_inline514__ssa_v0
          ;
          Tile<TileType::Vec, int32_t, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v220 = Tile<TileType::Vec, int32_t, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
          // pto: %gather_end_tile_inline514__ssa_v0
          ;
          uint64_t v221 = (uint64_t) v22;
          TASSIGN(v220, v221);
          pipe_barrier(PIPE_V);
          TROWMAX(v220, v218, v204);
          set_flag(PIPE_V, PIPE_S, EVENT_ID2);
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
          wait_flag(PIPE_V, PIPE_S, EVENT_ID2);
          // pto: %gather_end_inline515__ssa_v0
          ;
          int32_t v222 = v220.GetValue(v35);
          set_flag(PIPE_S, PIPE_V, EVENT_ID5);
          set_flag(PIPE_S, PIPE_V, EVENT_ID1);
          // pto: %24
          ;
          wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
          wait_flag(PIPE_S, PIPE_V, EVENT_ID5);
          if (v209 == v35) {
            // pto: %gather_empty_stage_inline517__ssa_v0
            ;
            Tile<TileType::Vec, bfloat16_t, 128, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v223 = Tile<TileType::Vec, bfloat16_t, 128, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v29);
            // pto: %gather_empty_stage_inline517__ssa_v0
            ;
            uint64_t v224 = (uint64_t) v21;
            TASSIGN(v223, v224);
            pipe_barrier(PIPE_V);
            TEXPANDS(v223, v45);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            // pto: %25
            ;
            int64_t v225 = v112 < v35 ? v35 : v112;
            // pto: %source_view_inline433__iter_v1_pview
            ;
            __gm__ bfloat16_t* v226 = PTOAS__GLOBAL_TENSOR_DATA(v57);
            // pto: %source_view_inline433__iter_v1_pview
            ;
            const int64_t v227 = 0;
            // pto: %source_view_inline433__iter_v1_pview
            ;
            const int64_t v228 = 512;
            // pto: %source_view_inline433__iter_v1_pview
            ;
            pto::Shape<1, 1, 1, 128, 512> v229 = pto::Shape<1, 1, 1, 128, 512>();
            // pto: %source_view_inline433__iter_v1_pview
            ;
            pto::Stride<65536, 65536, 65536, 512, 1> v230 = pto::Stride<65536, 65536, 65536, 512, 1>();
            // pto: %source_view_inline433__iter_v1_pview
            ;
            GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 512>, pto::Stride<65536, 65536, 65536, 512, 1>, pto::Layout::ND> v231 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 512>, pto::Stride<65536, 65536, 65536, 512, 1>, pto::Layout::ND>(v226 + (v227 + v225 * v228), v229, v230);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(v231, v223);
          } else {
            // pto: %23, %26
            ;
            if ((int64_t) v222 == v209) {
              v198.SetValidShape(v30, v209);
              // pto: %gather_prefix_stage_inline519__ssa_v0
              ;
              Tile<TileType::Vec, bfloat16_t, 128, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v232 = Tile<TileType::Vec, bfloat16_t, 128, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v209, v29);
              // pto: %gather_prefix_stage_inline519__ssa_v0
              ;
              uint64_t v233 = (uint64_t) v21;
              TASSIGN(v232, v233);
              // pto: %cmp_kv_flat_inline439__ssa_v0_pview
              ;
              const int64_t v234 = 0;
              // pto: %cmp_kv_flat_inline439__ssa_v0_pview
              ;
              __gm__ bfloat16_t* v235 = PTOAS__GLOBAL_TENSOR_DATA(v83);
              // pto: %cmp_kv_flat_inline439__ssa_v0_pview
              ;
              const int64_t v236 = 1;
              // pto: %cmp_kv_flat_inline439__ssa_v0_pview
              ;
              const int64_t v237 = 1;
              // pto: %cmp_kv_flat_inline439__ssa_v0_pview
              ;
              const int64_t v238 = 1;
              // pto: %cmp_kv_flat_inline439__ssa_v0_pview
              ;
              int64_t v239 = (int64_t) v16;
              // pto: %cmp_kv_flat_inline439__ssa_v0_pview
              ;
              int64_t v240 = v239 * v29;
              // pto: %cmp_kv_flat_inline439__ssa_v0_pview
              ;
              int64_t v241 = v238 * v240;
              // pto: %cmp_kv_flat_inline439__ssa_v0_pview
              ;
              pto::Shape<1, 1, 1, -1, 512> v242 = pto::Shape<1, 1, 1, -1, 512>(v236, v237, v238, v239, v29);
              // pto: %cmp_kv_flat_inline439__ssa_v0_pview
              ;
              pto::Stride<-1, -1, -1, -1, -1> v243 = pto::Stride<-1, -1, -1, -1, -1>(v237 * v241, v241, v240, v29, v30);
              // pto: %cmp_kv_flat_inline439__ssa_v0_pview
              ;
              GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v244 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v235 + (v234 + v35 * v29 + v35 * v30), v242, v243);
              MGATHER<pto::Coalesce::Row>(v232, v244, v198);
              set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
              // pto: %gather_prefix_filled_inline520__ssa_v0
              ;
              Tile<TileType::Vec, bfloat16_t, 128, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v245 = Tile<TileType::Vec, bfloat16_t, 128, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v34, v29);
              // pto: %gather_prefix_filled_inline520__ssa_v0
              ;
              uint64_t v246 = (uint64_t) v21;
              TASSIGN(v245, v246);
              wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
              TFILLPAD<pto::TFillPadMode::InPlace>(v245, v232);
              set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
              v245.SetValidShape(v34, v29);
              // pto: %27
              ;
              int64_t v247 = v112 < v35 ? v35 : v112;
              // pto: %28
              ;
              __gm__ bfloat16_t* v248 = PTOAS__GLOBAL_TENSOR_DATA(v57);
              // pto: %28
              ;
              const int64_t v249 = 0;
              // pto: %28
              ;
              const int64_t v250 = 512;
              // pto: %28
              ;
              pto::Shape<1, 1, 1, 128, 512> v251 = pto::Shape<1, 1, 1, 128, 512>();
              // pto: %28
              ;
              pto::Stride<65536, 65536, 65536, 512, 1> v252 = pto::Stride<65536, 65536, 65536, 512, 1>();
              // pto: %28
              ;
              GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 512>, pto::Stride<65536, 65536, 65536, 512, 1>, pto::Layout::ND> v253 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 512>, pto::Stride<65536, 65536, 65536, 512, 1>, pto::Layout::ND>(v248 + (v249 + v247 * v250), v251, v252);
              wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
              TSTORE(v253, v245);
            } else {
              // pto: %gather_mask_ones_inline522__ssa_v0
              ;
              Tile<TileType::Vec, int32_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v254 = Tile<TileType::Vec, int32_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v34);
              // pto: %gather_mask_ones_inline522__ssa_v0
              ;
              uint64_t v255 = (uint64_t) v21;
              TASSIGN(v254, v255);
              pipe_barrier(PIPE_V);
              TEXPANDS(v254, v41);
              // pto: %gather_mask_columns_inline523__ssa_v0
              ;
              Tile<TileType::Vec, int32_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v256 = Tile<TileType::Vec, int32_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v34);
              // pto: %gather_mask_columns_inline523__ssa_v0
              ;
              uint64_t v257 = (uint64_t) v21;
              TASSIGN(v256, v257);
              pipe_barrier(PIPE_V);
              TCOLEXPANDMUL(v256, v254, v186);
              // pto: %transpose_tmp
              ;
              Tile<TileType::Vec, int32_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v258 = Tile<TileType::Vec, int32_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v34);
              // pto: %transpose_tmp
              ;
              uint64_t v259 = (uint64_t) v22;
              TASSIGN(v258, v259);
              // pto: %gather_mask_words_inline524__ssa_v0
              ;
              Tile<TileType::Vec, int32_t, 128, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v260 = Tile<TileType::Vec, int32_t, 128, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v46);
              // pto: %gather_mask_words_inline524__ssa_v0
              ;
              uint64_t v261 = (uint64_t) v23;
              TASSIGN(v260, v261);
              pipe_barrier(PIPE_V);
              TTRANS(v260, v256, v258);
              set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
              // pto: %gather_mask_inline525__ssa_v0
              ;
              Tile<TileType::Vec, uint8_t, 128, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v262 = Tile<TileType::Vec, uint8_t, 128, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v47);
              // pto: %gather_mask_inline525__ssa_v0
              ;
              uint64_t v263 = (uint64_t) v23;
              TASSIGN(v262, v263);
              // pto: %vector_stage_inline526__ssa_v0
              ;
              Tile<TileType::Vec, bfloat16_t, 128, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v264 = Tile<TileType::Vec, bfloat16_t, 128, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v29);
              // pto: %vector_stage_inline526__ssa_v0
              ;
              uint64_t v265 = (uint64_t) v21;
              TASSIGN(v264, v265);
              // pto: %29
              ;
              const int64_t v266 = 0;
              // pto: %29
              ;
              __gm__ bfloat16_t* v267 = PTOAS__GLOBAL_TENSOR_DATA(v83);
              // pto: %29
              ;
              const int64_t v268 = 1;
              // pto: %29
              ;
              const int64_t v269 = 1;
              // pto: %29
              ;
              const int64_t v270 = 1;
              // pto: %29
              ;
              int64_t v271 = (int64_t) v16;
              // pto: %29
              ;
              int64_t v272 = v271 * v29;
              // pto: %29
              ;
              int64_t v273 = v270 * v272;
              // pto: %29
              ;
              pto::Shape<1, 1, 1, -1, 512> v274 = pto::Shape<1, 1, 1, -1, 512>(v268, v269, v270, v271, v29);
              // pto: %29
              ;
              pto::Stride<-1, -1, -1, -1, -1> v275 = pto::Stride<-1, -1, -1, -1, -1>(v269 * v273, v273, v272, v29, v30);
              // pto: %29
              ;
              GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v276 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v267 + (v266 + v35 * v29 + v35 * v30), v274, v275);
              wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
              MGATHER<pto::Coalesce::Row>(v264, v276, v198);
              set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
              // pto: %vector_stage_fp16_inline527__ssa_v0
              ;
              Tile<TileType::Vec, half, 128, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v277 = Tile<TileType::Vec, half, 128, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v29);
              // pto: %vector_stage_fp16_inline527__ssa_v0
              ;
              uint64_t v278 = (uint64_t) v21;
              TASSIGN(v277, v278);
              // pto: %gather_select_tmp_inline528__ssa_v0
              ;
              Tile<TileType::Vec, half, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v279 = Tile<TileType::Vec, half, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v29);
              // pto: %gather_select_tmp_inline528__ssa_v0
              ;
              uint64_t v280 = (uint64_t) v28;
              TASSIGN(v279, v280);
              TEXPANDS(v279, v48);
              // pto: %vector_selected_fp16_inline530__ssa_v0
              ;
              Tile<TileType::Vec, half, 128, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v281 = Tile<TileType::Vec, half, 128, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v29);
              // pto: %vector_selected_fp16_inline530__ssa_v0
              ;
              uint64_t v282 = (uint64_t) v21;
              TASSIGN(v281, v282);
              pipe_barrier(PIPE_V);
              wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
              TSELS(v281, v262, v277, v279, v48);
              set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
              // pto: %vector_selected_inline531__ssa_v0
              ;
              Tile<TileType::Vec, bfloat16_t, 128, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v283 = Tile<TileType::Vec, bfloat16_t, 128, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v29);
              // pto: %vector_selected_inline531__ssa_v0
              ;
              uint64_t v284 = (uint64_t) v21;
              TASSIGN(v283, v284);
              // pto: %31
              ;
              int64_t v285 = v112 < v35 ? v35 : v112;
              // pto: %32
              ;
              __gm__ bfloat16_t* v286 = PTOAS__GLOBAL_TENSOR_DATA(v57);
              // pto: %32
              ;
              const int64_t v287 = 0;
              // pto: %32
              ;
              const int64_t v288 = 512;
              // pto: %32
              ;
              pto::Shape<1, 1, 1, 128, 512> v289 = pto::Shape<1, 1, 1, 128, 512>();
              // pto: %32
              ;
              pto::Stride<65536, 65536, 65536, 512, 1> v290 = pto::Stride<65536, 65536, 65536, 512, 1>();
              // pto: %32
              ;
              GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 512>, pto::Stride<65536, 65536, 65536, 512, 1>, pto::Layout::ND> v291 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 512>, pto::Stride<65536, 65536, 65536, 512, 1>, pto::Layout::ND>(v286 + (v287 + v285 * v288), v289, v290);
              wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
              TSTORE(v291, v283);
            };
          };
          set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
          set_flag(PIPE_S, PIPE_V, EVENT_ID2);
          set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
          set_flag(PIPE_S, PIPE_V, EVENT_ID3);
          set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
        } else {
          // pto: %stage_inline532__tile
          ;
          Tile<TileType::Vec, bfloat16_t, 128, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v292 = Tile<TileType::Vec, bfloat16_t, 128, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v29);
          // pto: %stage_inline532__tile
          ;
          uint64_t v293 = (uint64_t) v21;
          TASSIGN(v292, v293);
          wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
          TEXPANDS(v292, v45);
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
          for (int64_t v294 = v35; v294 < v34; v294 += v30) {
            // pto: %33
            ;
            int64_t v295 = (int64_t) ((uint64_t) v106 + (uint64_t) v294);
            // pto: %34
            ;
            int64_t v296 = (int64_t) ((uint64_t) v295 - (uint64_t) v34);
            // pto: %35
            ;
            if (v296 < v29) {
              // pto: %36
              ;
              int64_t v297 = (int64_t) ((uint64_t) v108 * (uint64_t) v29);
              // pto: %37, %logical_slot_inline535__tile
              ;
              int32_t v298 = (v3)[(int64_t) ((uint64_t) v297 + (uint64_t) v296)];
              // pto: %38
              ;
              int64_t v299 = (int64_t) v298;
              // pto: %39
              ;
              if (v299 >= v35) {
                // pto: %41
                ;
                int64_t v300 = v299 / v12;
                // pto: %42
                ;
                if (v300 < v32) {
                  // pto: %physical_block_inline537__tile
                  ;
                  int32_t v301 = (v6)[v300];
                  // pto: %43
                  ;
                  int64_t v302 = (int64_t) v301;
                  // pto: %44, %46, %47
                  ;
                  if (v302 >= v35 & v302 < v13) {
                    // pto: %50
                    ;
                    int64_t v303 = v299 % v12;
                    // pto: %kv_row_inline540__tile
                    ;
                    Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v304 = Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v29);
                    // pto: %kv_row_inline540__tile
                    ;
                    uint64_t v305 = (uint64_t) v22;
                    TASSIGN(v304, v305);
                    // pto: %51
                    ;
                    int64_t v306 = v302 < v35 ? v35 : v302;
                    // pto: %52
                    ;
                    int64_t v307 = v303 < v35 ? v35 : v303;
                    // pto: %cmp_kv__ssa_v0_pview
                    ;
                    const int64_t v308 = 0;
                    // pto: %cmp_kv__ssa_v0_pview
                    ;
                    __gm__ bfloat16_t* v309 = PTOAS__GLOBAL_TENSOR_DATA(v89);
                    // pto: %cmp_kv__ssa_v0_pview
                    ;
                    const int64_t v310 = 1;
                    // pto: %cmp_kv__ssa_v0_pview
                    ;
                    pto::Shape<1, 1, 1, 1, 512> v311 = pto::Shape<1, 1, 1, 1, 512>(v310, v30, v30, v30, v29);
                    // pto: %cmp_kv__ssa_v0_pview
                    ;
                    pto::Stride<-1, -1, -1, -1, -1> v312 = pto::Stride<-1, -1, -1, -1, -1>(v30 * v84, v84, v29, v29, v30);
                    // pto: %cmp_kv__ssa_v0_pview
                    ;
                    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v313 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v309 + ((v308 + v306 * v84 + v307 * v29) + v35 * v29 + v35 * v30), v311, v312);
                    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
                    TLOAD(v304, v313);
                    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
                    // pto: %t__tile
                    ;
                    Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v314 = Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v29);
                    // pto: %t__tile
                    ;
                    uint64_t v315 = (uint64_t) v22;
                    TASSIGN(v314, v315);
                    int64_t v316 = (int64_t) ((uint64_t) v294 * (uint64_t) v26);
                    // pto: %assemble_view
                    ;
                    int64_t v317 = (int64_t) ((uint64_t) v316 + (uint64_t) v21);
                    // pto: %assemble_view
                    ;
                    Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v318;
                    // pto: %assemble_view
                    ;
                    uint64_t v319 = (uint64_t) v317;
                    TASSIGN(v318, v319);
                    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
                    pipe_barrier(PIPE_V);
                    TMOV(v318, v314);
                    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
                  };
                };
              };
            };
          };
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
          // pto: %53
          ;
          int64_t v320 = v112 < v35 ? v35 : v112;
          // pto: %54
          ;
          __gm__ bfloat16_t* v321 = PTOAS__GLOBAL_TENSOR_DATA(v57);
          // pto: %54
          ;
          const int64_t v322 = 0;
          // pto: %54
          ;
          const int64_t v323 = 512;
          // pto: %54
          ;
          pto::Shape<1, 1, 1, 128, 512> v324 = pto::Shape<1, 1, 1, 128, 512>();
          // pto: %54
          ;
          pto::Stride<65536, 65536, 65536, 512, 1> v325 = pto::Stride<65536, 65536, 65536, 512, 1>();
          // pto: %54
          ;
          GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 512>, pto::Stride<65536, 65536, 65536, 512, 1>, pto::Layout::ND> v326 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 512>, pto::Stride<65536, 65536, 65536, 512, 1>, pto::Layout::ND>(v321 + (v322 + v320 * v323), v324, v325);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
          TSTORE(v326, v292);
          set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
        };
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      };
    };
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
  wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID2);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID3);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}