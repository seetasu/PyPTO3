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

AICORE void merge_norm(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ float* v4, __gm__ float* v5, __gm__ float* v6, __gm__ int32_t* v7, __gm__ bfloat16_t* v8, int64_t v9, int64_t v10, int32_t v11, int32_t v12) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  const int64_t v13 = 256;
  const int64_t v14 = 256;
  // pto: %c164352_i64
  const int64_t v15 = 164352;
  // pto: %c164864_i64
  const int64_t v16 = 164864;
  // pto: %c0_i64
  const int64_t v17 = 0;
  // pto: %c98624_i64
  const int64_t v18 = 98624;
  // pto: %c32768_i64
  const int64_t v19 = 32768;
  // pto: %c32832_i64
  const int64_t v20 = 32832;
  // pto: %c131584_i64
  const int64_t v21 = 131584;
  // pto: %c65600_i64
  const int64_t v22 = 65600;
  // pto: %c65664_i64
  const int64_t v23 = 65664;
  // pto: %c98432_i64
  const int64_t v24 = 98432;
  // pto: %c98496_i64
  const int64_t v25 = 98496;
  // pto: %c98560_i64
  const int64_t v26 = 98560;
  // pto: %c131392_i64
  const int64_t v27 = 131392;
  // pto: %c131456_i64
  const int64_t v28 = 131456;
  // pto: %c131520_i64
  const int64_t v29 = 131520;
  // pto: %c164608_i64
  const int64_t v30 = 164608;
  const int64_t v31 = 1024;
  const int64_t v32 = 2048;
  const int64_t v33 = 3072;
  const int64_t v34 = 4096;
  const int64_t v35 = 5120;
  const int64_t v36 = 6144;
  const int64_t v37 = 7168;
  const int64_t v38 = 8192;
  const int64_t v39 = 9216;
  const int64_t v40 = 10240;
  const int64_t v41 = 11264;
  const int64_t v42 = 12288;
  const int64_t v43 = 13312;
  const int64_t v44 = 14336;
  const int64_t v45 = 15360;
  // pto: %c1_index
  const int64_t v46 = 1;
  // pto: %c512_index
  const int64_t v47 = 512;
  // pto: %c64_index
  const int64_t v48 = 64;
  // pto: %c8_index
  const int64_t v49 = 8;
  // pto: %c16_index
  const int64_t v50 = 16;
  // pto: %c4096_index
  const int64_t v51 = 4096;
  // pto: %c4_index
  const int64_t v52 = 4;
  // pto: %c0_index
  const int64_t v53 = 0;
  // pto: %c2_index
  const int64_t v54 = 2;
  // pto: %c448_index
  const int64_t v55 = 448;
  // pto: %c65664_index
  const int64_t v56 = 65664;
  // pto: %c3_index
  const int64_t v57 = 3;
  // pto: %c5_index
  const int64_t v58 = 5;
  // pto: %c6_index
  const int64_t v59 = 6;
  // pto: %c7_index
  const int64_t v60 = 7;
  // pto: %c9_index
  const int64_t v61 = 9;
  // pto: %c10_index
  const int64_t v62 = 10;
  // pto: %c11_index
  const int64_t v63 = 11;
  // pto: %c12_index
  const int64_t v64 = 12;
  // pto: %c13_index
  const int64_t v65 = 13;
  // pto: %c14_index
  const int64_t v66 = 14;
  // pto: %c15_index
  const int64_t v67 = 15;
  // pto: %sparse_blk_mi__rv_v2_view
  const int64_t v68 = 1;
  // pto: %sparse_blk_mi__rv_v2_view
  const int64_t v69 = 1;
  // pto: %sparse_blk_mi__rv_v2_view
  const int64_t v70 = 1;
  // pto: %sparse_blk_mi__rv_v2_view
  int64_t v71 = (int64_t) v10;
  // pto: %sparse_blk_mi__rv_v2_view
  int64_t v72 = v71 * v46;
  // pto: %sparse_blk_mi__rv_v2_view
  int64_t v73 = v70 * v72;
  // pto: %sparse_blk_mi__rv_v2_view
  pto::Shape<1, 1, 1, -1, -1> v74 = pto::Shape<1, 1, 1, -1, -1>(v68, v69, v70, v71, v46);
  // pto: %sparse_blk_mi__rv_v2_view
  pto::Stride<-1, -1, -1, -1, -1> v75 = pto::Stride<-1, -1, -1, -1, -1>(v69 * v73, v73, v72, v46, (int64_t) v10);
  // pto: %sparse_blk_mi__rv_v2_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v76 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v1, v74, v75);
  // pto: %sparse_blk_li__rv_v2_view
  const int64_t v77 = 1;
  // pto: %sparse_blk_li__rv_v2_view
  const int64_t v78 = 1;
  // pto: %sparse_blk_li__rv_v2_view
  const int64_t v79 = 1;
  // pto: %sparse_blk_li__rv_v2_view
  int64_t v80 = (int64_t) v10;
  // pto: %sparse_blk_li__rv_v2_view
  int64_t v81 = v80 * v46;
  // pto: %sparse_blk_li__rv_v2_view
  int64_t v82 = v79 * v81;
  // pto: %sparse_blk_li__rv_v2_view
  pto::Shape<1, 1, 1, -1, -1> v83 = pto::Shape<1, 1, 1, -1, -1>(v77, v78, v79, v80, v46);
  // pto: %sparse_blk_li__rv_v2_view
  pto::Stride<-1, -1, -1, -1, -1> v84 = pto::Stride<-1, -1, -1, -1, -1>(v78 * v82, v82, v81, v46, (int64_t) v10);
  // pto: %sparse_blk_li__rv_v2_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v85 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v2, v83, v84);
  // pto: %sparse_blk_oi__rv_v2_view
  const int64_t v86 = 1;
  // pto: %sparse_blk_oi__rv_v2_view
  const int64_t v87 = 1;
  // pto: %sparse_blk_oi__rv_v2_view
  const int64_t v88 = 1;
  // pto: %sparse_blk_oi__rv_v2_view
  int64_t v89 = (int64_t) v10;
  // pto: %sparse_blk_oi__rv_v2_view
  int64_t v90 = v89 * v47;
  // pto: %sparse_blk_oi__rv_v2_view
  int64_t v91 = v88 * v90;
  // pto: %sparse_blk_oi__rv_v2_view
  pto::Shape<1, 1, 1, -1, -1> v92 = pto::Shape<1, 1, 1, -1, -1>(v86, v87, v88, v89, v47);
  // pto: %sparse_blk_oi__rv_v2_view
  pto::Stride<-1, -1, -1, -1, -1> v93 = pto::Stride<-1, -1, -1, -1, -1>(v87 * v91, v91, v90, v47, v46);
  // pto: %sparse_blk_oi__rv_v2_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v94 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v92, v93);
  // pto: %attn_sink__ssa_v0_view
  const int64_t v95 = 1;
  // pto: %attn_sink__ssa_v0_view
  const int64_t v96 = 1;
  // pto: %attn_sink__ssa_v0_view
  const int64_t v97 = 1;
  // pto: %attn_sink__ssa_v0_view
  const int64_t v98 = 1;
  // pto: %attn_sink__ssa_v0_view
  int64_t v99 = v48 * v46;
  // pto: %attn_sink__ssa_v0_view
  int64_t v100 = v98 * v99;
  // pto: %attn_sink__ssa_v0_view
  int64_t v101 = v97 * v100;
  // pto: %attn_sink__ssa_v0_view
  pto::Shape<1, 1, 1, 1, -1> v102 = pto::Shape<1, 1, 1, 1, -1>(v95, v96, v97, v98, v48);
  // pto: %attn_sink__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v103 = pto::Stride<-1, -1, -1, -1, -1>(v96 * v101, v101, v100, v99, v46);
  // pto: %attn_sink__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v104 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v102, v103);
  // pto: %rope_cos_il__ssa_v1_view
  const int64_t v105 = 1;
  // pto: %rope_cos_il__ssa_v1_view
  const int64_t v106 = 1;
  // pto: %rope_cos_il__ssa_v1_view
  const int64_t v107 = 1;
  // pto: %rope_cos_il__ssa_v1_view
  int64_t v108 = v49 * v48;
  // pto: %rope_cos_il__ssa_v1_view
  int64_t v109 = v107 * v108;
  // pto: %rope_cos_il__ssa_v1_view
  pto::Shape<1, 1, 1, -1, -1> v110 = pto::Shape<1, 1, 1, -1, -1>(v105, v106, v107, v49, v48);
  // pto: %rope_cos_il__ssa_v1_view
  pto::Stride<-1, -1, -1, -1, -1> v111 = pto::Stride<-1, -1, -1, -1, -1>(v106 * v109, v109, v108, v48, v46);
  // pto: %rope_cos_il__ssa_v1_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v112 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v5, v110, v111);
  // pto: %rope_sin_signed__ssa_v1_view
  const int64_t v113 = 1;
  // pto: %rope_sin_signed__ssa_v1_view
  const int64_t v114 = 1;
  // pto: %rope_sin_signed__ssa_v1_view
  const int64_t v115 = 1;
  // pto: %rope_sin_signed__ssa_v1_view
  int64_t v116 = v49 * v48;
  // pto: %rope_sin_signed__ssa_v1_view
  int64_t v117 = v115 * v116;
  // pto: %rope_sin_signed__ssa_v1_view
  pto::Shape<1, 1, 1, -1, -1> v118 = pto::Shape<1, 1, 1, -1, -1>(v113, v114, v115, v49, v48);
  // pto: %rope_sin_signed__ssa_v1_view
  pto::Stride<-1, -1, -1, -1, -1> v119 = pto::Stride<-1, -1, -1, -1, -1>(v114 * v117, v117, v116, v48, v46);
  // pto: %rope_sin_signed__ssa_v1_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v120 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v6, v118, v119);
  // pto: %rope_swap_idx__ssa_v1_view
  const int64_t v121 = 1;
  // pto: %rope_swap_idx__ssa_v1_view
  const int64_t v122 = 1;
  // pto: %rope_swap_idx__ssa_v1_view
  const int64_t v123 = 1;
  // pto: %rope_swap_idx__ssa_v1_view
  int64_t v124 = v50 * v48;
  // pto: %rope_swap_idx__ssa_v1_view
  int64_t v125 = v123 * v124;
  // pto: %rope_swap_idx__ssa_v1_view
  pto::Shape<1, 1, 1, -1, -1> v126 = pto::Shape<1, 1, 1, -1, -1>(v121, v122, v123, v50, v48);
  // pto: %rope_swap_idx__ssa_v1_view
  pto::Stride<-1, -1, -1, -1, -1> v127 = pto::Stride<-1, -1, -1, -1, -1>(v122 * v125, v125, v124, v48, v46);
  // pto: %rope_swap_idx__ssa_v1_view
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v128 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v7, v126, v127);
  // pto: %o_packed__ssa_v0_view
  const int64_t v129 = 1;
  // pto: %o_packed__ssa_v0_view
  const int64_t v130 = 1;
  // pto: %o_packed__ssa_v0_view
  const int64_t v131 = 1;
  // pto: %o_packed__ssa_v0_view
  int64_t v132 = v48 * v51;
  // pto: %o_packed__ssa_v0_view
  int64_t v133 = v131 * v132;
  // pto: %o_packed__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v134 = pto::Shape<1, 1, 1, -1, -1>(v129, v130, v131, v48, v51);
  // pto: %o_packed__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v135 = pto::Stride<-1, -1, -1, -1, -1>(v130 * v133, v133, v132, v51, v46);
  // pto: %o_packed__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v136 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v8, v134, v135);
  // pto: %m_li__rv_v2
  Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v137 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
  // pto: %m_li__rv_v2
  uint64_t v138 = (uint64_t) v18;
  TASSIGN(v137, v138);
  // pto: %m_mi__rv_v2
  Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v139 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
  // pto: %m_mi__rv_v2
  uint64_t v140 = (uint64_t) v21;
  TASSIGN(v139, v140);
  // pto: %m_oi__rv_v2
  Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v141 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v47);
  // pto: %m_oi__rv_v2
  uint64_t v142 = (uint64_t) v20;
  TASSIGN(v141, v142);
  // pto: %m_idx__ssa_v0
  int64_t v143 = (int64_t) v11;
  // pto: %106
  int64_t v144 = v143 / v52;
  // pto: %107
  int64_t v145 = (int64_t) ((uint64_t) v144 * (uint64_t) v52);
  // pto: %108
  int64_t v146 = (int64_t) ((uint64_t) v143 - (uint64_t) v145);
  // pto: %109
  int64_t v147 = (int64_t) ((uint64_t) v146 * (uint64_t) v50);
  // pto: %110
  int64_t v148 = (int64_t) ((uint64_t) v143 * (uint64_t) v9);
  // pto: %111
  int64_t v149 = (int64_t) ((uint64_t) v148 * (uint64_t) v50);
  // pto: %m_mi__tile
  Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v150 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
  // pto: %m_mi__tile
  uint64_t v151 = (uint64_t) v15;
  TASSIGN(v150, v151);
  // pto: %112
  int64_t v152 = v149 < v53 ? v53 : v149;
  // pto: %sparse_blk_mi__rv_v2_pview
  int64_t v153 = (int64_t) v10;
  // pto: %sparse_blk_mi__rv_v2_pview
  const int64_t v154 = 0;
  // pto: %sparse_blk_mi__rv_v2_pview
  __gm__ float* v155 = PTOAS__GLOBAL_TENSOR_DATA(v76);
  // pto: %sparse_blk_mi__rv_v2_pview
  const int64_t v156 = 1;
  // pto: %sparse_blk_mi__rv_v2_pview
  const int64_t v157 = 1;
  // pto: %sparse_blk_mi__rv_v2_pview
  const int64_t v158 = 1;
  // pto: %sparse_blk_mi__rv_v2_pview
  int64_t v159 = v50 * v46;
  // pto: %sparse_blk_mi__rv_v2_pview
  int64_t v160 = v158 * v159;
  // pto: %sparse_blk_mi__rv_v2_pview
  pto::Shape<1, 1, 1, 16, 1> v161 = pto::Shape<1, 1, 1, 16, 1>(v156, v157, v158, v50, v46);
  // pto: %sparse_blk_mi__rv_v2_pview
  pto::Stride<-1, -1, -1, -1, -1> v162 = pto::Stride<-1, -1, -1, -1, -1>(v157 * v160, v160, v159, v46, v153);
  // pto: %sparse_blk_mi__rv_v2_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v163 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v155 + (v154 + v152 * v46 + v53 * v153), v161, v162);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
  TLOAD(v150, v163);
  // pto: %m_li__tile
  Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v164 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
  // pto: %m_li__tile
  uint64_t v165 = (uint64_t) v16;
  TASSIGN(v164, v165);
  // pto: %sparse_blk_li__rv_v2_pview
  int64_t v166 = (int64_t) v10;
  // pto: %sparse_blk_li__rv_v2_pview
  const int64_t v167 = 0;
  // pto: %sparse_blk_li__rv_v2_pview
  __gm__ float* v168 = PTOAS__GLOBAL_TENSOR_DATA(v85);
  // pto: %sparse_blk_li__rv_v2_pview
  const int64_t v169 = 1;
  // pto: %sparse_blk_li__rv_v2_pview
  const int64_t v170 = 1;
  // pto: %sparse_blk_li__rv_v2_pview
  const int64_t v171 = 1;
  // pto: %sparse_blk_li__rv_v2_pview
  int64_t v172 = v50 * v46;
  // pto: %sparse_blk_li__rv_v2_pview
  int64_t v173 = v171 * v172;
  // pto: %sparse_blk_li__rv_v2_pview
  pto::Shape<1, 1, 1, 16, 1> v174 = pto::Shape<1, 1, 1, 16, 1>(v169, v170, v171, v50, v46);
  // pto: %sparse_blk_li__rv_v2_pview
  pto::Stride<-1, -1, -1, -1, -1> v175 = pto::Stride<-1, -1, -1, -1, -1>(v170 * v173, v173, v172, v46, v166);
  // pto: %sparse_blk_li__rv_v2_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v176 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v168 + (v167 + v152 * v46 + v53 * v166), v174, v175);
  TLOAD(v164, v176);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
  // pto: %m_oi__tile
  Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v177 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v47);
  // pto: %m_oi__tile
  uint64_t v178 = (uint64_t) v17;
  TASSIGN(v177, v178);
  // pto: %sparse_blk_oi__rv_v2_pview
  __gm__ float* v179 = PTOAS__GLOBAL_TENSOR_DATA(v94);
  // pto: %sparse_blk_oi__rv_v2_pview
  const int64_t v180 = 0;
  // pto: %sparse_blk_oi__rv_v2_pview
  const int64_t v181 = 512;
  // pto: %sparse_blk_oi__rv_v2_pview
  pto::Shape<1, 1, 1, 16, 512> v182 = pto::Shape<1, 1, 1, 16, 512>();
  // pto: %sparse_blk_oi__rv_v2_pview
  pto::Stride<8192, 8192, 8192, 512, 1> v183 = pto::Stride<8192, 8192, 8192, 512, 1>();
  // pto: %sparse_blk_oi__rv_v2_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v184 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(v179 + (v180 + v152 * v181), v182, v183);
  TLOAD(v177, v184);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
  // pto: %115, %116, %117
  int64_t v185 = (int64_t) ((uint64_t) ((int64_t) ((uint64_t) v9 - (uint64_t) v46) / v54) * (uint64_t) v54);
  // pto: %118
  int64_t v186 = (int64_t) ((uint64_t) v185 + (uint64_t) v46);
  for (int64_t v187 = v46; v187 < v186; v187 += v54) {
    // pto: %119
    ;
    int64_t v188 = (int64_t) ((uint64_t) v187 * (uint64_t) v50);
    // pto: %120
    ;
    int64_t v189 = (int64_t) ((uint64_t) v149 + (uint64_t) v188);
    // pto: %122
    ;
    int64_t v190 = (int64_t) ((uint64_t) v188 + (uint64_t) v50);
    // pto: %123
    ;
    int64_t v191 = (int64_t) ((uint64_t) v149 + (uint64_t) v190);
    // pto: %m_cur_mi__tile
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v192 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %m_cur_mi__tile
    ;
    uint64_t v193 = (uint64_t) v18;
    TASSIGN(v192, v193);
    // pto: %124
    ;
    int64_t v194 = v189 < v53 ? v53 : v189;
    // pto: %125
    ;
    int64_t v195 = (int64_t) v10;
    // pto: %125
    ;
    const int64_t v196 = 0;
    // pto: %125
    ;
    __gm__ float* v197 = PTOAS__GLOBAL_TENSOR_DATA(v76);
    // pto: %125
    ;
    const int64_t v198 = 1;
    // pto: %125
    ;
    const int64_t v199 = 1;
    // pto: %125
    ;
    const int64_t v200 = 1;
    // pto: %125
    ;
    int64_t v201 = v50 * v46;
    // pto: %125
    ;
    int64_t v202 = v200 * v201;
    // pto: %125
    ;
    pto::Shape<1, 1, 1, 16, 1> v203 = pto::Shape<1, 1, 1, 16, 1>(v198, v199, v200, v50, v46);
    // pto: %125
    ;
    pto::Stride<-1, -1, -1, -1, -1> v204 = pto::Stride<-1, -1, -1, -1, -1>(v199 * v202, v202, v201, v46, v195);
    // pto: %125
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v205 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v197 + (v196 + v194 * v46 + v53 * v195), v203, v204);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v192, v205);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %m_cur_li__tile
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v206 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %m_cur_li__tile
    ;
    uint64_t v207 = (uint64_t) v19;
    TASSIGN(v206, v207);
    // pto: %127
    ;
    int64_t v208 = (int64_t) v10;
    // pto: %127
    ;
    const int64_t v209 = 0;
    // pto: %127
    ;
    __gm__ float* v210 = PTOAS__GLOBAL_TENSOR_DATA(v85);
    // pto: %127
    ;
    const int64_t v211 = 1;
    // pto: %127
    ;
    const int64_t v212 = 1;
    // pto: %127
    ;
    const int64_t v213 = 1;
    // pto: %127
    ;
    int64_t v214 = v50 * v46;
    // pto: %127
    ;
    int64_t v215 = v213 * v214;
    // pto: %127
    ;
    pto::Shape<1, 1, 1, 16, 1> v216 = pto::Shape<1, 1, 1, 16, 1>(v211, v212, v213, v50, v46);
    // pto: %127
    ;
    pto::Stride<-1, -1, -1, -1, -1> v217 = pto::Stride<-1, -1, -1, -1, -1>(v212 * v215, v215, v214, v46, v208);
    // pto: %127
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v218 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v210 + (v209 + v194 * v46 + v53 * v208), v216, v217);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    TLOAD(v206, v218);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    // pto: %m_cur_oi__tile
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v219 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v47);
    // pto: %m_cur_oi__tile
    ;
    uint64_t v220 = (uint64_t) v20;
    TASSIGN(v219, v220);
    // pto: %129
    ;
    __gm__ float* v221 = PTOAS__GLOBAL_TENSOR_DATA(v94);
    // pto: %129
    ;
    const int64_t v222 = 0;
    // pto: %129
    ;
    const int64_t v223 = 512;
    // pto: %129
    ;
    pto::Shape<1, 1, 1, 16, 512> v224 = pto::Shape<1, 1, 1, 16, 512>();
    // pto: %129
    ;
    pto::Stride<8192, 8192, 8192, 512, 1> v225 = pto::Stride<8192, 8192, 8192, 512, 1>();
    // pto: %129
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v226 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(v221 + (v222 + v194 * v223), v224, v225);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    TLOAD(v219, v226);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    // pto: %0
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v227 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %0
    ;
    uint64_t v228 = (uint64_t) v21;
    TASSIGN(v227, v228);
    // pto: %130
    ;
    int64_t v229 = v191 < v53 ? v53 : v191;
    // pto: %131
    ;
    int64_t v230 = (int64_t) v10;
    // pto: %131
    ;
    const int64_t v231 = 0;
    // pto: %131
    ;
    __gm__ float* v232 = PTOAS__GLOBAL_TENSOR_DATA(v76);
    // pto: %131
    ;
    const int64_t v233 = 1;
    // pto: %131
    ;
    const int64_t v234 = 1;
    // pto: %131
    ;
    const int64_t v235 = 1;
    // pto: %131
    ;
    int64_t v236 = v50 * v46;
    // pto: %131
    ;
    int64_t v237 = v235 * v236;
    // pto: %131
    ;
    pto::Shape<1, 1, 1, 16, 1> v238 = pto::Shape<1, 1, 1, 16, 1>(v233, v234, v235, v50, v46);
    // pto: %131
    ;
    pto::Stride<-1, -1, -1, -1, -1> v239 = pto::Stride<-1, -1, -1, -1, -1>(v234 * v237, v237, v236, v46, v230);
    // pto: %131
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v240 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v232 + (v231 + v229 * v46 + v53 * v230), v238, v239);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    TLOAD(v227, v240);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    // pto: %1
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v241 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %1
    ;
    uint64_t v242 = (uint64_t) v22;
    TASSIGN(v241, v242);
    // pto: %133
    ;
    int64_t v243 = (int64_t) v10;
    // pto: %133
    ;
    const int64_t v244 = 0;
    // pto: %133
    ;
    __gm__ float* v245 = PTOAS__GLOBAL_TENSOR_DATA(v85);
    // pto: %133
    ;
    const int64_t v246 = 1;
    // pto: %133
    ;
    const int64_t v247 = 1;
    // pto: %133
    ;
    const int64_t v248 = 1;
    // pto: %133
    ;
    int64_t v249 = v50 * v46;
    // pto: %133
    ;
    int64_t v250 = v248 * v249;
    // pto: %133
    ;
    pto::Shape<1, 1, 1, 16, 1> v251 = pto::Shape<1, 1, 1, 16, 1>(v246, v247, v248, v50, v46);
    // pto: %133
    ;
    pto::Stride<-1, -1, -1, -1, -1> v252 = pto::Stride<-1, -1, -1, -1, -1>(v247 * v250, v250, v249, v46, v243);
    // pto: %133
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v253 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v245 + (v244 + v229 * v46 + v53 * v243), v251, v252);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    TLOAD(v241, v253);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
    // pto: %2
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v254 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v47);
    // pto: %2
    ;
    uint64_t v255 = (uint64_t) v23;
    TASSIGN(v254, v255);
    // pto: %135
    ;
    __gm__ float* v256 = PTOAS__GLOBAL_TENSOR_DATA(v94);
    // pto: %135
    ;
    const int64_t v257 = 0;
    // pto: %135
    ;
    const int64_t v258 = 512;
    // pto: %135
    ;
    pto::Shape<1, 1, 1, 16, 512> v259 = pto::Shape<1, 1, 1, 16, 512>();
    // pto: %135
    ;
    pto::Stride<8192, 8192, 8192, 512, 1> v260 = pto::Stride<8192, 8192, 8192, 512, 1>();
    // pto: %135
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v261 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(v256 + (v257 + v229 * v258), v259, v260);
    TLOAD(v254, v261);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
    // pto: %m_mi_new__rm_a0_tmp_v0
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v262 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %m_mi_new__rm_a0_tmp_v0
    ;
    uint64_t v263 = (uint64_t) v15;
    TASSIGN(v262, v263);
    // pto: %m_mi_new__rm_a1_tmp_v1
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v264 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %m_mi_new__rm_a1_tmp_v1
    ;
    uint64_t v265 = (uint64_t) v18;
    TASSIGN(v264, v265);
    // pto: %m_mi_new__row_major_tmp_v2
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v266 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %m_mi_new__row_major_tmp_v2
    ;
    uint64_t v267 = (uint64_t) v24;
    TASSIGN(v266, v267);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pipe_barrier(PIPE_V);
    TMAX(v266, v262, v264);
    // pto: %m_mi_new__tile
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v268 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %m_mi_new__tile
    ;
    uint64_t v269 = (uint64_t) v24;
    TASSIGN(v268, v269);
    // pto: %m_mi__ssa_v3
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v270 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %m_mi__ssa_v3
    ;
    uint64_t v271 = (uint64_t) v24;
    TASSIGN(v270, v271);
    // pto: %t__rm_a0_tmp_v3
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v272 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %t__rm_a0_tmp_v3
    ;
    uint64_t v273 = (uint64_t) v15;
    TASSIGN(v272, v273);
    // pto: %t__rm_a1_tmp_v4
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v274 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %t__rm_a1_tmp_v4
    ;
    uint64_t v275 = (uint64_t) v24;
    TASSIGN(v274, v275);
    // pto: %t__row_major_tmp_v5
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v276 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %t__row_major_tmp_v5
    ;
    uint64_t v277 = (uint64_t) v25;
    TASSIGN(v276, v277);
    pipe_barrier(PIPE_V);
    TSUB(v276, v272, v274);
    // pto: %t__tile
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v278 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %t__tile
    ;
    uint64_t v279 = (uint64_t) v25;
    TASSIGN(v278, v279);
    // pto: %m_alpha__rm_a0_tmp_v6
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v280 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %m_alpha__rm_a0_tmp_v6
    ;
    uint64_t v281 = (uint64_t) v25;
    TASSIGN(v280, v281);
    // pto: %m_alpha__row_major_tmp_v7
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v282 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %m_alpha__row_major_tmp_v7
    ;
    uint64_t v283 = (uint64_t) v25;
    TASSIGN(v282, v283);
    pipe_barrier(PIPE_V);
    TEXP(v282, v280);
    // pto: %m_alpha__tile
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v284 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %m_alpha__tile
    ;
    uint64_t v285 = (uint64_t) v25;
    TASSIGN(v284, v285);
    // pto: %t__rm_a0_tmp_v8
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v286 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %t__rm_a0_tmp_v8
    ;
    uint64_t v287 = (uint64_t) v18;
    TASSIGN(v286, v287);
    // pto: %t__rm_a1_tmp_v9
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v288 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %t__rm_a1_tmp_v9
    ;
    uint64_t v289 = (uint64_t) v24;
    TASSIGN(v288, v289);
    // pto: %t__row_major_tmp_v10
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v290 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %t__row_major_tmp_v10
    ;
    uint64_t v291 = (uint64_t) v18;
    TASSIGN(v290, v291);
    TSUB(v290, v286, v288);
    // pto: %3
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v292 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %3
    ;
    uint64_t v293 = (uint64_t) v18;
    TASSIGN(v292, v293);
    // pto: %m_beta__rm_a0_tmp_v11
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v294 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %m_beta__rm_a0_tmp_v11
    ;
    uint64_t v295 = (uint64_t) v18;
    TASSIGN(v294, v295);
    // pto: %m_beta__row_major_tmp_v12
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v296 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %m_beta__row_major_tmp_v12
    ;
    uint64_t v297 = (uint64_t) v26;
    TASSIGN(v296, v297);
    pipe_barrier(PIPE_V);
    TEXP(v296, v294);
    // pto: %m_beta__tile
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v298 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %m_beta__tile
    ;
    uint64_t v299 = (uint64_t) v26;
    TASSIGN(v298, v299);
    // pto: %t__rm_a0_tmp_v13
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v300 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %t__rm_a0_tmp_v13
    ;
    uint64_t v301 = (uint64_t) v25;
    TASSIGN(v300, v301);
    // pto: %t__rm_a1_tmp_v14
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v302 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %t__rm_a1_tmp_v14
    ;
    uint64_t v303 = (uint64_t) v16;
    TASSIGN(v302, v303);
    // pto: %t__row_major_tmp_v15
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v304 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %t__row_major_tmp_v15
    ;
    uint64_t v305 = (uint64_t) v18;
    TASSIGN(v304, v305);
    pipe_barrier(PIPE_V);
    TMUL(v304, v300, v302);
    // pto: %4
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v306 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %4
    ;
    uint64_t v307 = (uint64_t) v18;
    TASSIGN(v306, v307);
    // pto: %t__rm_a0_tmp_v16
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v308 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %t__rm_a0_tmp_v16
    ;
    uint64_t v309 = (uint64_t) v26;
    TASSIGN(v308, v309);
    // pto: %t__rm_a1_tmp_v17
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v310 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %t__rm_a1_tmp_v17
    ;
    uint64_t v311 = (uint64_t) v19;
    TASSIGN(v310, v311);
    // pto: %t__row_major_tmp_v18
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v312 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %t__row_major_tmp_v18
    ;
    uint64_t v313 = (uint64_t) v19;
    TASSIGN(v312, v313);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TMUL(v312, v308, v310);
    // pto: %5
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v314 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %5
    ;
    uint64_t v315 = (uint64_t) v19;
    TASSIGN(v314, v315);
    // pto: %m_li__rm_a0_tmp_v19
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v316 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %m_li__rm_a0_tmp_v19
    ;
    uint64_t v317 = (uint64_t) v18;
    TASSIGN(v316, v317);
    // pto: %m_li__rm_a1_tmp_v20
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v318 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %m_li__rm_a1_tmp_v20
    ;
    uint64_t v319 = (uint64_t) v19;
    TASSIGN(v318, v319);
    // pto: %m_li__row_major_tmp_v21
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v320 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %m_li__row_major_tmp_v21
    ;
    uint64_t v321 = (uint64_t) v19;
    TASSIGN(v320, v321);
    pipe_barrier(PIPE_V);
    TADD(v320, v316, v318);
    // pto: %6
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v322 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %6
    ;
    uint64_t v323 = (uint64_t) v19;
    TASSIGN(v322, v323);
    // pto: %7
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v324 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v47);
    // pto: %7
    ;
    uint64_t v325 = (uint64_t) v18;
    TASSIGN(v324, v325);
    pipe_barrier(PIPE_V);
    TROWEXPANDMUL(v324, v177, v284);
    // pto: %8
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v326 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v47);
    // pto: %8
    ;
    uint64_t v327 = (uint64_t) v20;
    TASSIGN(v326, v327);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    TROWEXPANDMUL(v326, v219, v298);
    // pto: %9
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v328 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v47);
    // pto: %9
    ;
    uint64_t v329 = (uint64_t) v20;
    TASSIGN(v328, v329);
    pipe_barrier(PIPE_V);
    TADD(v328, v324, v326);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    // pto: %10
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v330 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %10
    ;
    uint64_t v331 = (uint64_t) v24;
    TASSIGN(v330, v331);
    // pto: %11
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v332 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %11
    ;
    uint64_t v333 = (uint64_t) v21;
    TASSIGN(v332, v333);
    // pto: %12
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v334 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %12
    ;
    uint64_t v335 = (uint64_t) v27;
    TASSIGN(v334, v335);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    TMAX(v334, v330, v332);
    // pto: %13
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v336 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %13
    ;
    uint64_t v337 = (uint64_t) v27;
    TASSIGN(v336, v337);
    // pto: %14
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v338 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %14
    ;
    uint64_t v339 = (uint64_t) v27;
    TASSIGN(v338, v339);
    // pto: %15
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v340 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %15
    ;
    uint64_t v341 = (uint64_t) v24;
    TASSIGN(v340, v341);
    // pto: %16
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v342 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %16
    ;
    uint64_t v343 = (uint64_t) v27;
    TASSIGN(v342, v343);
    // pto: %17
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v344 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %17
    ;
    uint64_t v345 = (uint64_t) v28;
    TASSIGN(v344, v345);
    pipe_barrier(PIPE_V);
    TSUB(v344, v340, v342);
    // pto: %18
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v346 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %18
    ;
    uint64_t v347 = (uint64_t) v28;
    TASSIGN(v346, v347);
    // pto: %19
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v348 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %19
    ;
    uint64_t v349 = (uint64_t) v28;
    TASSIGN(v348, v349);
    // pto: %20
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v350 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %20
    ;
    uint64_t v351 = (uint64_t) v28;
    TASSIGN(v350, v351);
    pipe_barrier(PIPE_V);
    TEXP(v350, v348);
    // pto: %21
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v352 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %21
    ;
    uint64_t v353 = (uint64_t) v28;
    TASSIGN(v352, v353);
    // pto: %22
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v354 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %22
    ;
    uint64_t v355 = (uint64_t) v21;
    TASSIGN(v354, v355);
    // pto: %23
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v356 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %23
    ;
    uint64_t v357 = (uint64_t) v27;
    TASSIGN(v356, v357);
    // pto: %24
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v358 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %24
    ;
    uint64_t v359 = (uint64_t) v21;
    TASSIGN(v358, v359);
    TSUB(v358, v354, v356);
    // pto: %25
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v360 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %25
    ;
    uint64_t v361 = (uint64_t) v21;
    TASSIGN(v360, v361);
    // pto: %26
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v362 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %26
    ;
    uint64_t v363 = (uint64_t) v21;
    TASSIGN(v362, v363);
    // pto: %27
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v364 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %27
    ;
    uint64_t v365 = (uint64_t) v29;
    TASSIGN(v364, v365);
    pipe_barrier(PIPE_V);
    TEXP(v364, v362);
    // pto: %28
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v366 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %28
    ;
    uint64_t v367 = (uint64_t) v29;
    TASSIGN(v366, v367);
    // pto: %29
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v368 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %29
    ;
    uint64_t v369 = (uint64_t) v28;
    TASSIGN(v368, v369);
    // pto: %30
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v370 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %30
    ;
    uint64_t v371 = (uint64_t) v19;
    TASSIGN(v370, v371);
    // pto: %31
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v372 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %31
    ;
    uint64_t v373 = (uint64_t) v21;
    TASSIGN(v372, v373);
    pipe_barrier(PIPE_V);
    TMUL(v372, v368, v370);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    // pto: %32
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v374 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %32
    ;
    uint64_t v375 = (uint64_t) v21;
    TASSIGN(v374, v375);
    // pto: %33
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v376 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %33
    ;
    uint64_t v377 = (uint64_t) v29;
    TASSIGN(v376, v377);
    // pto: %34
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v378 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %34
    ;
    uint64_t v379 = (uint64_t) v22;
    TASSIGN(v378, v379);
    // pto: %35
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v380 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %35
    ;
    uint64_t v381 = (uint64_t) v22;
    TASSIGN(v380, v381);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
    TMUL(v380, v376, v378);
    // pto: %36
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v382 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %36
    ;
    uint64_t v383 = (uint64_t) v22;
    TASSIGN(v382, v383);
    // pto: %37
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v384 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %37
    ;
    uint64_t v385 = (uint64_t) v21;
    TASSIGN(v384, v385);
    // pto: %38
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v386 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %38
    ;
    uint64_t v387 = (uint64_t) v22;
    TASSIGN(v386, v387);
    // pto: %39
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v388 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %39
    ;
    uint64_t v389 = (uint64_t) v22;
    TASSIGN(v388, v389);
    pipe_barrier(PIPE_V);
    TADD(v388, v384, v386);
    // pto: %40
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v390 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %40
    ;
    uint64_t v391 = (uint64_t) v22;
    TASSIGN(v390, v391);
    // pto: %41
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v392 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v47);
    // pto: %41
    ;
    uint64_t v393 = (uint64_t) v21;
    TASSIGN(v392, v393);
    pipe_barrier(PIPE_V);
    TROWEXPANDMUL(v392, v328, v352);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    // pto: %42
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v394 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v47);
    // pto: %42
    ;
    uint64_t v395 = (uint64_t) v23;
    TASSIGN(v394, v395);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
    TROWEXPANDMUL(v394, v254, v366);
    // pto: %43
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v396 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v47);
    // pto: %43
    ;
    uint64_t v397 = (uint64_t) v17;
    TASSIGN(v396, v397);
    pipe_barrier(PIPE_V);
    TADD(v396, v392, v394);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    // pto: %m_mi__ssa_v3_mv
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v398 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %m_mi__ssa_v3_mv
    ;
    uint64_t v399 = (uint64_t) v15;
    TASSIGN(v398, v399);
    TMOV(v398, v338);
    // pto: %m_li__tile_mv
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v400 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %m_li__tile_mv
    ;
    uint64_t v401 = (uint64_t) v16;
    TASSIGN(v400, v401);
    TMOV(v400, v390);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
  }
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
  // pto: %136, %137
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
  if ((int64_t) ((uint64_t) v9 - (uint64_t) v186) == v46) {
    // pto: %138
    ;
    int64_t v402 = (int64_t) ((uint64_t) v186 * (uint64_t) v50);
    // pto: %139
    ;
    int64_t v403 = (int64_t) ((uint64_t) v149 + (uint64_t) v402);
    // pto: %44
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v404 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %44
    ;
    uint64_t v405 = (uint64_t) v23;
    TASSIGN(v404, v405);
    // pto: %140
    ;
    int64_t v406 = v403 < v53 ? v53 : v403;
    // pto: %141
    ;
    int64_t v407 = (int64_t) v10;
    // pto: %141
    ;
    const int64_t v408 = 0;
    // pto: %141
    ;
    __gm__ float* v409 = PTOAS__GLOBAL_TENSOR_DATA(v76);
    // pto: %141
    ;
    const int64_t v410 = 1;
    // pto: %141
    ;
    const int64_t v411 = 1;
    // pto: %141
    ;
    const int64_t v412 = 1;
    // pto: %141
    ;
    int64_t v413 = v50 * v46;
    // pto: %141
    ;
    int64_t v414 = v412 * v413;
    // pto: %141
    ;
    pto::Shape<1, 1, 1, 16, 1> v415 = pto::Shape<1, 1, 1, 16, 1>(v410, v411, v412, v50, v46);
    // pto: %141
    ;
    pto::Stride<-1, -1, -1, -1, -1> v416 = pto::Stride<-1, -1, -1, -1, -1>(v411 * v414, v414, v413, v46, v407);
    // pto: %141
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v417 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v409 + (v408 + v406 * v46 + v53 * v407), v415, v416);
    TLOAD(v404, v417);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %45
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v418 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %45
    ;
    uint64_t v419 = (uint64_t) v18;
    TASSIGN(v418, v419);
    // pto: %143
    ;
    int64_t v420 = (int64_t) v10;
    // pto: %143
    ;
    const int64_t v421 = 0;
    // pto: %143
    ;
    __gm__ float* v422 = PTOAS__GLOBAL_TENSOR_DATA(v85);
    // pto: %143
    ;
    const int64_t v423 = 1;
    // pto: %143
    ;
    const int64_t v424 = 1;
    // pto: %143
    ;
    const int64_t v425 = 1;
    // pto: %143
    ;
    int64_t v426 = v50 * v46;
    // pto: %143
    ;
    int64_t v427 = v425 * v426;
    // pto: %143
    ;
    pto::Shape<1, 1, 1, 16, 1> v428 = pto::Shape<1, 1, 1, 16, 1>(v423, v424, v425, v50, v46);
    // pto: %143
    ;
    pto::Stride<-1, -1, -1, -1, -1> v429 = pto::Stride<-1, -1, -1, -1, -1>(v424 * v427, v427, v426, v46, v420);
    // pto: %143
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v430 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v422 + (v421 + v406 * v46 + v53 * v420), v428, v429);
    TLOAD(v418, v430);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    // pto: %46
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v431 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v47);
    // pto: %46
    ;
    uint64_t v432 = (uint64_t) v20;
    TASSIGN(v431, v432);
    // pto: %145
    ;
    __gm__ float* v433 = PTOAS__GLOBAL_TENSOR_DATA(v94);
    // pto: %145
    ;
    const int64_t v434 = 0;
    // pto: %145
    ;
    const int64_t v435 = 512;
    // pto: %145
    ;
    pto::Shape<1, 1, 1, 16, 512> v436 = pto::Shape<1, 1, 1, 16, 512>();
    // pto: %145
    ;
    pto::Stride<8192, 8192, 8192, 512, 1> v437 = pto::Stride<8192, 8192, 8192, 512, 1>();
    // pto: %145
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v438 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(v433 + (v434 + v406 * v435), v436, v437);
    TLOAD(v431, v438);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    // pto: %47
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v439 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %47
    ;
    uint64_t v440 = (uint64_t) v15;
    TASSIGN(v439, v440);
    // pto: %48
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v441 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %48
    ;
    uint64_t v442 = (uint64_t) v23;
    TASSIGN(v441, v442);
    // pto: %49
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v443 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %49
    ;
    uint64_t v444 = (uint64_t) v21;
    TASSIGN(v443, v444);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pipe_barrier(PIPE_V);
    TMAX(v443, v439, v441);
    // pto: %50
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v445 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %50
    ;
    uint64_t v446 = (uint64_t) v21;
    TASSIGN(v445, v446);
    // pto: %51
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v447 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %51
    ;
    uint64_t v448 = (uint64_t) v15;
    TASSIGN(v447, v448);
    // pto: %52
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v449 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %52
    ;
    uint64_t v450 = (uint64_t) v21;
    TASSIGN(v449, v450);
    // pto: %53
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v451 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %53
    ;
    uint64_t v452 = (uint64_t) v19;
    TASSIGN(v451, v452);
    pipe_barrier(PIPE_V);
    TSUB(v451, v447, v449);
    // pto: %54
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v453 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %54
    ;
    uint64_t v454 = (uint64_t) v19;
    TASSIGN(v453, v454);
    // pto: %55
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v455 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %55
    ;
    uint64_t v456 = (uint64_t) v19;
    TASSIGN(v455, v456);
    // pto: %56
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v457 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %56
    ;
    uint64_t v458 = (uint64_t) v19;
    TASSIGN(v457, v458);
    pipe_barrier(PIPE_V);
    TEXP(v457, v455);
    // pto: %57
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v459 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %57
    ;
    uint64_t v460 = (uint64_t) v19;
    TASSIGN(v459, v460);
    // pto: %58
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v461 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %58
    ;
    uint64_t v462 = (uint64_t) v23;
    TASSIGN(v461, v462);
    // pto: %59
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v463 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %59
    ;
    uint64_t v464 = (uint64_t) v21;
    TASSIGN(v463, v464);
    // pto: %60
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v465 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %60
    ;
    uint64_t v466 = (uint64_t) v23;
    TASSIGN(v465, v466);
    TSUB(v465, v461, v463);
    // pto: %61
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v467 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %61
    ;
    uint64_t v468 = (uint64_t) v23;
    TASSIGN(v467, v468);
    // pto: %62
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v469 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %62
    ;
    uint64_t v470 = (uint64_t) v23;
    TASSIGN(v469, v470);
    // pto: %63
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v471 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %63
    ;
    uint64_t v472 = (uint64_t) v22;
    TASSIGN(v471, v472);
    pipe_barrier(PIPE_V);
    TEXP(v471, v469);
    // pto: %64
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v473 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %64
    ;
    uint64_t v474 = (uint64_t) v22;
    TASSIGN(v473, v474);
    // pto: %65
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v475 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %65
    ;
    uint64_t v476 = (uint64_t) v19;
    TASSIGN(v475, v476);
    // pto: %66
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v477 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %66
    ;
    uint64_t v478 = (uint64_t) v16;
    TASSIGN(v477, v478);
    // pto: %67
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v479 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %67
    ;
    uint64_t v480 = (uint64_t) v23;
    TASSIGN(v479, v480);
    pipe_barrier(PIPE_V);
    TMUL(v479, v475, v477);
    // pto: %68
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v481 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %68
    ;
    uint64_t v482 = (uint64_t) v23;
    TASSIGN(v481, v482);
    // pto: %69
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v483 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %69
    ;
    uint64_t v484 = (uint64_t) v22;
    TASSIGN(v483, v484);
    // pto: %70
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v485 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %70
    ;
    uint64_t v486 = (uint64_t) v18;
    TASSIGN(v485, v486);
    // pto: %71
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v487 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %71
    ;
    uint64_t v488 = (uint64_t) v18;
    TASSIGN(v487, v488);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TMUL(v487, v483, v485);
    // pto: %72
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v489 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %72
    ;
    uint64_t v490 = (uint64_t) v18;
    TASSIGN(v489, v490);
    // pto: %73
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v491 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %73
    ;
    uint64_t v492 = (uint64_t) v23;
    TASSIGN(v491, v492);
    // pto: %74
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v493 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %74
    ;
    uint64_t v494 = (uint64_t) v18;
    TASSIGN(v493, v494);
    // pto: %75
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v495 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
    // pto: %75
    ;
    uint64_t v496 = (uint64_t) v18;
    TASSIGN(v495, v496);
    pipe_barrier(PIPE_V);
    TADD(v495, v491, v493);
    // pto: %76
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v497 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %76
    ;
    uint64_t v498 = (uint64_t) v18;
    TASSIGN(v497, v498);
    // pto: %77
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v499 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v47);
    // pto: %77
    ;
    uint64_t v500 = (uint64_t) v23;
    TASSIGN(v499, v500);
    pipe_barrier(PIPE_V);
    TROWEXPANDMUL(v499, v177, v459);
    // pto: %78
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v501 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v47);
    // pto: %78
    ;
    uint64_t v502 = (uint64_t) v20;
    TASSIGN(v501, v502);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    TROWEXPANDMUL(v501, v431, v473);
    // pto: %79
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v503 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v47);
    // pto: %79
    ;
    uint64_t v504 = (uint64_t) v20;
    TASSIGN(v503, v504);
    pipe_barrier(PIPE_V);
    TADD(v503, v499, v501);
    // pto: %80
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v505 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %80
    ;
    uint64_t v506 = (uint64_t) v21;
    TASSIGN(v505, v506);
  } else {
    // pto: %m_li__rv_v2_main_mv
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v507 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %m_li__rv_v2_main_mv
    ;
    uint64_t v508 = (uint64_t) v18;
    TASSIGN(v507, v508);
    pipe_barrier(PIPE_V);
    TMOV(v507, v164);
    // pto: %m_mi__rv_v2_main_mv
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v509 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
    // pto: %m_mi__rv_v2_main_mv
    ;
    uint64_t v510 = (uint64_t) v21;
    TASSIGN(v509, v510);
    TMOV(v509, v150);
    // pto: %m_oi__rv_v2_main_mv
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v511 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v47);
    // pto: %m_oi__rv_v2_main_mv
    ;
    uint64_t v512 = (uint64_t) v20;
    TASSIGN(v511, v512);
    TMOV(v511, v177);
  }
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
  // pto: %81
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v513 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
  // pto: %81
  uint64_t v514 = (uint64_t) v17;
  TASSIGN(v513, v514);
  // pto: %146
  int64_t v515 = v147 < v53 ? v53 : v147;
  // pto: %attn_sink__ssa_v0_pview
  __gm__ float* v516 = PTOAS__GLOBAL_TENSOR_DATA(v104);
  // pto: %attn_sink__ssa_v0_pview
  const int64_t v517 = 0;
  // pto: %attn_sink__ssa_v0_pview
  pto::Shape<1, 1, 1, 1, 16> v518 = pto::Shape<1, 1, 1, 1, 16>();
  // pto: %attn_sink__ssa_v0_pview
  pto::Stride<16, 16, 16, 16, 1> v519 = pto::Stride<16, 16, 16, 16, 1>();
  // pto: %attn_sink__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v520 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v516 + (v517 + v515), v518, v519);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
  TLOAD(v513, v520);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  // pto: %n_sink_bias__tile
  Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v521 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
  // pto: %n_sink_bias__tile
  uint64_t v522 = (uint64_t) v17;
  TASSIGN(v521, v522);
  // pto: %t__rm_a0_tmp_v22
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v523 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
  // pto: %t__rm_a0_tmp_v22
  uint64_t v524 = (uint64_t) v21;
  TASSIGN(v523, v524);
  // pto: %t__rm_a1_tmp_v23
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v525 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
  // pto: %t__rm_a1_tmp_v23
  uint64_t v526 = (uint64_t) v21;
  TASSIGN(v525, v526);
  // pto: %t__row_major_tmp_v24
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v527 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
  // pto: %t__row_major_tmp_v24
  uint64_t v528 = (uint64_t) v23;
  TASSIGN(v527, v528);
  pipe_barrier(PIPE_V);
  TSUB(v527, v523, v525);
  // pto: %82
  Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v529 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
  // pto: %82
  uint64_t v530 = (uint64_t) v23;
  TASSIGN(v529, v530);
  // pto: %n_sink_tile__rm_a0_tmp_v25
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v531 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
  // pto: %n_sink_tile__rm_a0_tmp_v25
  uint64_t v532 = (uint64_t) v23;
  TASSIGN(v531, v532);
  // pto: %n_sink_tile__rm_a1_tmp_v26
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v533 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
  // pto: %n_sink_tile__rm_a1_tmp_v26
  uint64_t v534 = (uint64_t) v17;
  TASSIGN(v533, v534);
  // pto: %n_sink_tile__row_major_tmp_v27
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v535 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
  // pto: %n_sink_tile__row_major_tmp_v27
  uint64_t v536 = (uint64_t) v17;
  TASSIGN(v535, v536);
  pipe_barrier(PIPE_V);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TADD(v535, v531, v533);
  // pto: %n_sink_tile__tile
  Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v537 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
  // pto: %n_sink_tile__tile
  uint64_t v538 = (uint64_t) v17;
  TASSIGN(v537, v538);
  // pto: %t__rm_a0_tmp_v28
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v539 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
  // pto: %t__rm_a0_tmp_v28
  uint64_t v540 = (uint64_t) v17;
  TASSIGN(v539, v540);
  // pto: %t__rm_a1_tmp_v29
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v541 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
  // pto: %t__rm_a1_tmp_v29
  uint64_t v542 = (uint64_t) v21;
  TASSIGN(v541, v542);
  // pto: %t__row_major_tmp_v30
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v543 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
  // pto: %t__row_major_tmp_v30
  uint64_t v544 = (uint64_t) v17;
  TASSIGN(v543, v544);
  pipe_barrier(PIPE_V);
  TSUB(v543, v539, v541);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
  // pto: %83
  Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v545 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
  // pto: %83
  uint64_t v546 = (uint64_t) v17;
  TASSIGN(v545, v546);
  // pto: %t__rm_a0_tmp_v31
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v547 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
  // pto: %t__rm_a0_tmp_v31
  uint64_t v548 = (uint64_t) v17;
  TASSIGN(v547, v548);
  // pto: %t__row_major_tmp_v32
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v549 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
  // pto: %t__row_major_tmp_v32
  uint64_t v550 = (uint64_t) v17;
  TASSIGN(v549, v550);
  pipe_barrier(PIPE_V);
  TEXP(v549, v547);
  // pto: %84
  Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v551 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
  // pto: %84
  uint64_t v552 = (uint64_t) v17;
  TASSIGN(v551, v552);
  // pto: %n_denom__rm_a0_tmp_v33
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v553 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
  // pto: %n_denom__rm_a0_tmp_v33
  uint64_t v554 = (uint64_t) v18;
  TASSIGN(v553, v554);
  // pto: %n_denom__rm_a1_tmp_v34
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v555 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
  // pto: %n_denom__rm_a1_tmp_v34
  uint64_t v556 = (uint64_t) v17;
  TASSIGN(v555, v556);
  // pto: %n_denom__row_major_tmp_v35
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v557 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v50);
  // pto: %n_denom__row_major_tmp_v35
  uint64_t v558 = (uint64_t) v23;
  TASSIGN(v557, v558);
  pipe_barrier(PIPE_V);
  TADD(v557, v553, v555);
  // pto: %n_denom__tile
  Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v559 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v46);
  // pto: %n_denom__tile
  uint64_t v560 = (uint64_t) v23;
  TASSIGN(v559, v560);
  // pto: %85
  Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v561 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v47);
  // pto: %85
  uint64_t v562 = (uint64_t) v17;
  TASSIGN(v561, v562);
  pipe_barrier(PIPE_V);
  TROWEXPANDDIV(v561, v141, v559);
  // pto: %n_full__tile
  Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v563 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v47);
  // pto: %n_full__tile
  uint64_t v564 = (uint64_t) v17;
  TASSIGN(v563, v564);
  // pto: %slice_view
  Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, 16, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v565;
  // pto: %slice_view
  uint64_t v566 = (uint64_t) v17;
  TASSIGN(v565, v566);
  // pto: %n_bf16__tile
  Tile<TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v567 = Tile<TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v47);
  // pto: %n_bf16__tile
  uint64_t v568 = (uint64_t) v20;
  TASSIGN(v567, v568);
  pipe_barrier(PIPE_V);
  RoundMode v569 = RoundMode::CAST_RINT;
  SaturationMode v570 = SaturationMode::OFF;
  TCVT(v567, v565, v569, v570);
  // pto: %m_rope__tile_textract
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v571 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v48);
  // pto: %m_rope__tile_textract
  uint64_t v572 = (uint64_t) v23;
  TASSIGN(v571, v572);
  uint16_t v573 = static_cast<uint16_t>(v53);
  uint16_t v574 = static_cast<uint16_t>(v55);
  TEXTRACT(v571, v561, v573, v574);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
  // pto: %m_cos_il__tile
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v575 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v48);
  // pto: %m_cos_il__tile
  uint64_t v576 = (uint64_t) v21;
  TASSIGN(v575, v576);
  // pto: %147
  int64_t v577 = v144 < v53 ? v53 : v144;
  // pto: %rope_cos_il__ssa_v1_pview
  __gm__ float* v578 = PTOAS__GLOBAL_TENSOR_DATA(v112);
  // pto: %rope_cos_il__ssa_v1_pview
  const int64_t v579 = 0;
  // pto: %rope_cos_il__ssa_v1_pview
  const int64_t v580 = 64;
  // pto: %rope_cos_il__ssa_v1_pview
  pto::Shape<1, 1, 1, 1, 64> v581 = pto::Shape<1, 1, 1, 1, 64>();
  // pto: %rope_cos_il__ssa_v1_pview
  pto::Stride<64, 64, 64, 64, 1> v582 = pto::Stride<64, 64, 64, 64, 1>();
  // pto: %rope_cos_il__ssa_v1_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v583 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v578 + (v579 + v577 * v580), v581, v582);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
  TLOAD(v575, v583);
  // pto: %m_sin_signed__tile
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v584 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v48);
  // pto: %m_sin_signed__tile
  uint64_t v585 = (uint64_t) v15;
  TASSIGN(v584, v585);
  // pto: %rope_sin_signed__ssa_v1_pview
  __gm__ float* v586 = PTOAS__GLOBAL_TENSOR_DATA(v120);
  // pto: %rope_sin_signed__ssa_v1_pview
  const int64_t v587 = 0;
  // pto: %rope_sin_signed__ssa_v1_pview
  const int64_t v588 = 64;
  // pto: %rope_sin_signed__ssa_v1_pview
  pto::Shape<1, 1, 1, 1, 64> v589 = pto::Shape<1, 1, 1, 1, 64>();
  // pto: %rope_sin_signed__ssa_v1_pview
  pto::Stride<64, 64, 64, 64, 1> v590 = pto::Stride<64, 64, 64, 64, 1>();
  // pto: %rope_sin_signed__ssa_v1_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v591 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v586 + (v587 + v577 * v588), v589, v590);
  TLOAD(v584, v591);
  // pto: %86
  Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v592 = Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v48);
  // pto: %86
  uint64_t v593 = (uint64_t) v17;
  TASSIGN(v592, v593);
  // pto: %rope_swap_idx__ssa_v1_pview
  __gm__ int32_t* v594 = PTOAS__GLOBAL_TENSOR_DATA(v128);
  // pto: %rope_swap_idx__ssa_v1_pview
  pto::Shape<1, 1, 1, 16, 64> v595 = pto::Shape<1, 1, 1, 16, 64>();
  // pto: %rope_swap_idx__ssa_v1_pview
  pto::Stride<1024, 1024, 1024, 64, 1> v596 = pto::Stride<1024, 1024, 1024, 64, 1>();
  // pto: %rope_swap_idx__ssa_v1_pview
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND> v597 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND>(v594, v595, v596);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
  TLOAD(v592, v597);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  // pto: %gather_acc_init
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v598 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v48);
  // pto: %gather_acc_init
  uint64_t v599 = (uint64_t) v18;
  TASSIGN(v598, v599);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  for (int64_t v600 = v53; v600 < v50; v600 += v46) {
    int64_t v601 = (int64_t) ((uint64_t) v600 * (uint64_t) v14);
    // pto: %151
    ;
    int64_t v602 = (int64_t) ((uint64_t) v601 + (uint64_t) v56);
    // pto: %gather_inp_row
    ;
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v603 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v48);
    // pto: %gather_inp_row
    ;
    uint64_t v604 = (uint64_t) v602;
    TASSIGN(v603, v604);
    // pto: %153
    ;
    int64_t v605 = (int64_t) ((uint64_t) v600 * (uint64_t) v13);
    // pto: %153
    ;
    int64_t v606 = (int64_t) ((uint64_t) v605 + (uint64_t) v23);
    // pto: %153
    ;
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v607;
    // pto: %153
    ;
    uint64_t v608 = (uint64_t) v606;
    TASSIGN(v607, v608);
    // pto: %gather_idx_row
    ;
    Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v609 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v48);
    // pto: %gather_idx_row
    ;
    uint64_t v610 = (uint64_t) v601;
    TASSIGN(v609, v610);
    // pto: %158
    ;
    Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v611;
    // pto: %158
    ;
    uint64_t v612 = (uint64_t) v605;
    TASSIGN(v611, v612);
    // pto: %gather_row_tmp
    ;
    Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v613 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v48);
    // pto: %gather_row_tmp
    ;
    uint64_t v614 = (uint64_t) v30;
    TASSIGN(v613, v614);
    // pto: %gather_row
    ;
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v615 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v48);
    // pto: %gather_row
    ;
    uint64_t v616 = (uint64_t) v16;
    TASSIGN(v615, v616);
    pipe_barrier(PIPE_V);
    TGATHER(v615, v607, v611, v613);
    // pto: %assemble_view
    ;
    int64_t v617 = (int64_t) ((uint64_t) v605 + (uint64_t) v18);
    // pto: %assemble_view
    ;
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v618;
    // pto: %assemble_view
    ;
    uint64_t v619 = (uint64_t) v617;
    TASSIGN(v618, v619);
    pipe_barrier(PIPE_V);
    TMOV(v618, v615);
  }
  // pto: %m_swapped__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v620 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v48);
  // pto: %m_swapped__tile
  uint64_t v621 = (uint64_t) v18;
  TASSIGN(v620, v621);
  // pto: %87
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v622 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v48);
  // pto: %87
  uint64_t v623 = (uint64_t) v17;
  TASSIGN(v622, v623);
  TCOLEXPANDMUL(v622, v571, v575);
  // pto: %88
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v624 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v48);
  // pto: %88
  uint64_t v625 = (uint64_t) v23;
  TASSIGN(v624, v625);
  pipe_barrier(PIPE_V);
  TCOLEXPANDMUL(v624, v620, v584);
  // pto: %m_rot__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v626 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v48);
  // pto: %m_rot__tile
  uint64_t v627 = (uint64_t) v17;
  TASSIGN(v626, v627);
  pipe_barrier(PIPE_V);
  TADD(v626, v622, v624);
  // pto: %n_rope_bf16__tile
  Tile<TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v628 = Tile<TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v48);
  // pto: %n_rope_bf16__tile
  uint64_t v629 = (uint64_t) v23;
  TASSIGN(v628, v629);
  pipe_barrier(PIPE_V);
  RoundMode v630 = RoundMode::CAST_RINT;
  SaturationMode v631 = SaturationMode::OFF;
  TCVT(v628, v626, v630, v631);
  // pto: %89
  Tile<TileType::Vec, bfloat16_t, 16, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v632 = Tile<TileType::Vec, bfloat16_t, 16, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v55);
  // pto: %89
  uint64_t v633 = (uint64_t) v20;
  TASSIGN(v632, v633);
  // pto: %159
  Tile<TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, 16, 448, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v634;
  // pto: %159
  uint64_t v635 = (uint64_t) v20;
  TASSIGN(v634, v635);
  // pto: %n_full_bf16__tile
  Tile<TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v636 = Tile<TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v50, v47);
  // pto: %n_full_bf16__tile
  uint64_t v637 = (uint64_t) v17;
  TASSIGN(v636, v637);
  pipe_barrier(PIPE_V);
  TCONCAT(v636, v634, v628);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  // pto: %160, %161
  int64_t v638 = (int64_t) ((uint64_t) (v147 / v49) * (uint64_t) v49);
  // pto: %162
  int64_t v639 = (int64_t) ((uint64_t) v638 + (uint64_t) v144);
  // pto: %163, %164
  int64_t v640 = (int64_t) ((uint64_t) (v147 % v49) * (uint64_t) v47);
  // pto: %90
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v641 = Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v47);
  // pto: %90
  uint64_t v642 = (uint64_t) v17;
  TASSIGN(v641, v642);
  // pto: %165
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v643;
  // pto: %165
  uint64_t v644 = (uint64_t) v17;
  TASSIGN(v643, v644);
  // pto: %166
  int64_t v645 = v639 < v53 ? v53 : v639;
  // pto: %167
  int64_t v646 = v640 < v53 ? v53 : v640;
  // pto: %o_packed__ssa_v0_pview
  __gm__ bfloat16_t* v647 = PTOAS__GLOBAL_TENSOR_DATA(v136);
  // pto: %o_packed__ssa_v0_pview
  const int64_t v648 = 0;
  // pto: %o_packed__ssa_v0_pview
  const int64_t v649 = 4096;
  // pto: %o_packed__ssa_v0_pview
  pto::Shape<1, 1, 1, 1, 512> v650 = pto::Shape<1, 1, 1, 1, 512>();
  // pto: %o_packed__ssa_v0_pview
  pto::Stride<4096, 4096, 4096, 4096, 1> v651 = pto::Stride<4096, 4096, 4096, 4096, 1>();
  // pto: %o_packed__ssa_v0_pview
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v652 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v647 + (v648 + v645 * v649 + v646), v650, v651);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v652, v643);
  // pto: %168
  int64_t v653 = (int64_t) ((uint64_t) v147 + (uint64_t) v46);
  // pto: %169, %170
  int64_t v654 = (int64_t) ((uint64_t) (v653 / v49) * (uint64_t) v49);
  // pto: %171
  int64_t v655 = (int64_t) ((uint64_t) v654 + (uint64_t) v144);
  // pto: %173, %174
  int64_t v656 = (int64_t) ((uint64_t) (v653 % v49) * (uint64_t) v47);
  // pto: %91
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v657 = Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v47);
  // pto: %91
  uint64_t v658 = (uint64_t) v31;
  TASSIGN(v657, v658);
  // pto: %175
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v659;
  // pto: %175
  uint64_t v660 = (uint64_t) v31;
  TASSIGN(v659, v660);
  // pto: %177
  int64_t v661 = v655 < v53 ? v53 : v655;
  // pto: %178
  int64_t v662 = v656 < v53 ? v53 : v656;
  // pto: %o_packed__tile_pview
  __gm__ bfloat16_t* v663 = PTOAS__GLOBAL_TENSOR_DATA(v136);
  // pto: %o_packed__tile_pview
  const int64_t v664 = 0;
  // pto: %o_packed__tile_pview
  const int64_t v665 = 4096;
  // pto: %o_packed__tile_pview
  pto::Shape<1, 1, 1, 1, 512> v666 = pto::Shape<1, 1, 1, 1, 512>();
  // pto: %o_packed__tile_pview
  pto::Stride<4096, 4096, 4096, 4096, 1> v667 = pto::Stride<4096, 4096, 4096, 4096, 1>();
  // pto: %o_packed__tile_pview
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v668 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v663 + (v664 + v661 * v665 + v662), v666, v667);
  pipe_barrier(PIPE_MTE3);
  TSTORE(v668, v659);
  // pto: %179
  int64_t v669 = (int64_t) ((uint64_t) v147 + (uint64_t) v54);
  // pto: %180, %181
  int64_t v670 = (int64_t) ((uint64_t) (v669 / v49) * (uint64_t) v49);
  // pto: %182
  int64_t v671 = (int64_t) ((uint64_t) v670 + (uint64_t) v144);
  // pto: %184, %185
  int64_t v672 = (int64_t) ((uint64_t) (v669 % v49) * (uint64_t) v47);
  // pto: %92
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v673 = Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v47);
  // pto: %92
  uint64_t v674 = (uint64_t) v32;
  TASSIGN(v673, v674);
  // pto: %186
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v675;
  // pto: %186
  uint64_t v676 = (uint64_t) v32;
  TASSIGN(v675, v676);
  // pto: %188
  int64_t v677 = v671 < v53 ? v53 : v671;
  // pto: %189
  int64_t v678 = v672 < v53 ? v53 : v672;
  // pto: %190
  __gm__ bfloat16_t* v679 = PTOAS__GLOBAL_TENSOR_DATA(v136);
  // pto: %190
  const int64_t v680 = 0;
  // pto: %190
  const int64_t v681 = 4096;
  // pto: %190
  pto::Shape<1, 1, 1, 1, 512> v682 = pto::Shape<1, 1, 1, 1, 512>();
  // pto: %190
  pto::Stride<4096, 4096, 4096, 4096, 1> v683 = pto::Stride<4096, 4096, 4096, 4096, 1>();
  // pto: %190
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v684 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v679 + (v680 + v677 * v681 + v678), v682, v683);
  pipe_barrier(PIPE_MTE3);
  TSTORE(v684, v675);
  // pto: %191
  int64_t v685 = (int64_t) ((uint64_t) v147 + (uint64_t) v57);
  // pto: %192, %193
  int64_t v686 = (int64_t) ((uint64_t) (v685 / v49) * (uint64_t) v49);
  // pto: %194
  int64_t v687 = (int64_t) ((uint64_t) v686 + (uint64_t) v144);
  // pto: %196, %197
  int64_t v688 = (int64_t) ((uint64_t) (v685 % v49) * (uint64_t) v47);
  // pto: %93
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v689 = Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v47);
  // pto: %93
  uint64_t v690 = (uint64_t) v33;
  TASSIGN(v689, v690);
  // pto: %198
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v691;
  // pto: %198
  uint64_t v692 = (uint64_t) v33;
  TASSIGN(v691, v692);
  // pto: %200
  int64_t v693 = v687 < v53 ? v53 : v687;
  // pto: %201
  int64_t v694 = v688 < v53 ? v53 : v688;
  // pto: %202
  __gm__ bfloat16_t* v695 = PTOAS__GLOBAL_TENSOR_DATA(v136);
  // pto: %202
  const int64_t v696 = 0;
  // pto: %202
  const int64_t v697 = 4096;
  // pto: %202
  pto::Shape<1, 1, 1, 1, 512> v698 = pto::Shape<1, 1, 1, 1, 512>();
  // pto: %202
  pto::Stride<4096, 4096, 4096, 4096, 1> v699 = pto::Stride<4096, 4096, 4096, 4096, 1>();
  // pto: %202
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v700 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v695 + (v696 + v693 * v697 + v694), v698, v699);
  pipe_barrier(PIPE_MTE3);
  TSTORE(v700, v691);
  // pto: %203
  int64_t v701 = (int64_t) ((uint64_t) v147 + (uint64_t) v52);
  // pto: %204, %205
  int64_t v702 = (int64_t) ((uint64_t) (v701 / v49) * (uint64_t) v49);
  // pto: %206
  int64_t v703 = (int64_t) ((uint64_t) v702 + (uint64_t) v144);
  // pto: %208, %209
  int64_t v704 = (int64_t) ((uint64_t) (v701 % v49) * (uint64_t) v47);
  // pto: %94
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v705 = Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v47);
  // pto: %94
  uint64_t v706 = (uint64_t) v34;
  TASSIGN(v705, v706);
  // pto: %210
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v707;
  // pto: %210
  uint64_t v708 = (uint64_t) v34;
  TASSIGN(v707, v708);
  // pto: %212
  int64_t v709 = v703 < v53 ? v53 : v703;
  // pto: %213
  int64_t v710 = v704 < v53 ? v53 : v704;
  // pto: %214
  __gm__ bfloat16_t* v711 = PTOAS__GLOBAL_TENSOR_DATA(v136);
  // pto: %214
  const int64_t v712 = 0;
  // pto: %214
  const int64_t v713 = 4096;
  // pto: %214
  pto::Shape<1, 1, 1, 1, 512> v714 = pto::Shape<1, 1, 1, 1, 512>();
  // pto: %214
  pto::Stride<4096, 4096, 4096, 4096, 1> v715 = pto::Stride<4096, 4096, 4096, 4096, 1>();
  // pto: %214
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v716 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v711 + (v712 + v709 * v713 + v710), v714, v715);
  pipe_barrier(PIPE_MTE3);
  TSTORE(v716, v707);
  // pto: %215
  int64_t v717 = (int64_t) ((uint64_t) v147 + (uint64_t) v58);
  // pto: %216, %217
  int64_t v718 = (int64_t) ((uint64_t) (v717 / v49) * (uint64_t) v49);
  // pto: %218
  int64_t v719 = (int64_t) ((uint64_t) v718 + (uint64_t) v144);
  // pto: %220, %221
  int64_t v720 = (int64_t) ((uint64_t) (v717 % v49) * (uint64_t) v47);
  // pto: %95
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v721 = Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v47);
  // pto: %95
  uint64_t v722 = (uint64_t) v35;
  TASSIGN(v721, v722);
  // pto: %222
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v723;
  // pto: %222
  uint64_t v724 = (uint64_t) v35;
  TASSIGN(v723, v724);
  // pto: %224
  int64_t v725 = v719 < v53 ? v53 : v719;
  // pto: %225
  int64_t v726 = v720 < v53 ? v53 : v720;
  // pto: %226
  __gm__ bfloat16_t* v727 = PTOAS__GLOBAL_TENSOR_DATA(v136);
  // pto: %226
  const int64_t v728 = 0;
  // pto: %226
  const int64_t v729 = 4096;
  // pto: %226
  pto::Shape<1, 1, 1, 1, 512> v730 = pto::Shape<1, 1, 1, 1, 512>();
  // pto: %226
  pto::Stride<4096, 4096, 4096, 4096, 1> v731 = pto::Stride<4096, 4096, 4096, 4096, 1>();
  // pto: %226
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v732 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v727 + (v728 + v725 * v729 + v726), v730, v731);
  pipe_barrier(PIPE_MTE3);
  TSTORE(v732, v723);
  // pto: %227
  int64_t v733 = (int64_t) ((uint64_t) v147 + (uint64_t) v59);
  // pto: %228, %229
  int64_t v734 = (int64_t) ((uint64_t) (v733 / v49) * (uint64_t) v49);
  // pto: %230
  int64_t v735 = (int64_t) ((uint64_t) v734 + (uint64_t) v144);
  // pto: %232, %233
  int64_t v736 = (int64_t) ((uint64_t) (v733 % v49) * (uint64_t) v47);
  // pto: %96
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v737 = Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v47);
  // pto: %96
  uint64_t v738 = (uint64_t) v36;
  TASSIGN(v737, v738);
  // pto: %234
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v739;
  // pto: %234
  uint64_t v740 = (uint64_t) v36;
  TASSIGN(v739, v740);
  // pto: %236
  int64_t v741 = v735 < v53 ? v53 : v735;
  // pto: %237
  int64_t v742 = v736 < v53 ? v53 : v736;
  // pto: %238
  __gm__ bfloat16_t* v743 = PTOAS__GLOBAL_TENSOR_DATA(v136);
  // pto: %238
  const int64_t v744 = 0;
  // pto: %238
  const int64_t v745 = 4096;
  // pto: %238
  pto::Shape<1, 1, 1, 1, 512> v746 = pto::Shape<1, 1, 1, 1, 512>();
  // pto: %238
  pto::Stride<4096, 4096, 4096, 4096, 1> v747 = pto::Stride<4096, 4096, 4096, 4096, 1>();
  // pto: %238
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v748 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v743 + (v744 + v741 * v745 + v742), v746, v747);
  pipe_barrier(PIPE_MTE3);
  TSTORE(v748, v739);
  // pto: %239
  int64_t v749 = (int64_t) ((uint64_t) v147 + (uint64_t) v60);
  // pto: %240, %241
  int64_t v750 = (int64_t) ((uint64_t) (v749 / v49) * (uint64_t) v49);
  // pto: %242
  int64_t v751 = (int64_t) ((uint64_t) v750 + (uint64_t) v144);
  // pto: %244, %245
  int64_t v752 = (int64_t) ((uint64_t) (v749 % v49) * (uint64_t) v47);
  // pto: %97
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v753 = Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v47);
  // pto: %97
  uint64_t v754 = (uint64_t) v37;
  TASSIGN(v753, v754);
  // pto: %246
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v755;
  // pto: %246
  uint64_t v756 = (uint64_t) v37;
  TASSIGN(v755, v756);
  // pto: %248
  int64_t v757 = v751 < v53 ? v53 : v751;
  // pto: %249
  int64_t v758 = v752 < v53 ? v53 : v752;
  // pto: %250
  __gm__ bfloat16_t* v759 = PTOAS__GLOBAL_TENSOR_DATA(v136);
  // pto: %250
  const int64_t v760 = 0;
  // pto: %250
  const int64_t v761 = 4096;
  // pto: %250
  pto::Shape<1, 1, 1, 1, 512> v762 = pto::Shape<1, 1, 1, 1, 512>();
  // pto: %250
  pto::Stride<4096, 4096, 4096, 4096, 1> v763 = pto::Stride<4096, 4096, 4096, 4096, 1>();
  // pto: %250
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v764 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v759 + (v760 + v757 * v761 + v758), v762, v763);
  pipe_barrier(PIPE_MTE3);
  TSTORE(v764, v755);
  // pto: %254
  int64_t v765 = (int64_t) ((uint64_t) v639 + (uint64_t) v49);
  // pto: %98
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v766 = Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v47);
  // pto: %98
  uint64_t v767 = (uint64_t) v38;
  TASSIGN(v766, v767);
  // pto: %257
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v768;
  // pto: %257
  uint64_t v769 = (uint64_t) v38;
  TASSIGN(v768, v769);
  // pto: %259
  int64_t v770 = v765 < v53 ? v53 : v765;
  // pto: %261
  __gm__ bfloat16_t* v771 = PTOAS__GLOBAL_TENSOR_DATA(v136);
  // pto: %261
  const int64_t v772 = 0;
  // pto: %261
  const int64_t v773 = 4096;
  // pto: %261
  pto::Shape<1, 1, 1, 1, 512> v774 = pto::Shape<1, 1, 1, 1, 512>();
  // pto: %261
  pto::Stride<4096, 4096, 4096, 4096, 1> v775 = pto::Stride<4096, 4096, 4096, 4096, 1>();
  // pto: %261
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v776 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v771 + (v772 + v770 * v773 + v646), v774, v775);
  pipe_barrier(PIPE_MTE3);
  TSTORE(v776, v768);
  // pto: %262, %263, %264
  int64_t v777 = (int64_t) ((uint64_t) ((int64_t) ((uint64_t) v147 + (uint64_t) v61) / v49) * (uint64_t) v49);
  // pto: %265
  int64_t v778 = (int64_t) ((uint64_t) v777 + (uint64_t) v144);
  // pto: %99
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v779 = Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v47);
  // pto: %99
  uint64_t v780 = (uint64_t) v39;
  TASSIGN(v779, v780);
  // pto: %269
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v781;
  // pto: %269
  uint64_t v782 = (uint64_t) v39;
  TASSIGN(v781, v782);
  // pto: %271
  int64_t v783 = v778 < v53 ? v53 : v778;
  // pto: %273
  __gm__ bfloat16_t* v784 = PTOAS__GLOBAL_TENSOR_DATA(v136);
  // pto: %273
  const int64_t v785 = 0;
  // pto: %273
  const int64_t v786 = 4096;
  // pto: %273
  pto::Shape<1, 1, 1, 1, 512> v787 = pto::Shape<1, 1, 1, 1, 512>();
  // pto: %273
  pto::Stride<4096, 4096, 4096, 4096, 1> v788 = pto::Stride<4096, 4096, 4096, 4096, 1>();
  // pto: %273
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v789 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v784 + (v785 + v783 * v786 + v662), v787, v788);
  pipe_barrier(PIPE_MTE3);
  TSTORE(v789, v781);
  // pto: %274, %275, %276
  int64_t v790 = (int64_t) ((uint64_t) ((int64_t) ((uint64_t) v147 + (uint64_t) v62) / v49) * (uint64_t) v49);
  // pto: %277
  int64_t v791 = (int64_t) ((uint64_t) v790 + (uint64_t) v144);
  // pto: %100
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v792 = Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v47);
  // pto: %100
  uint64_t v793 = (uint64_t) v40;
  TASSIGN(v792, v793);
  // pto: %281
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v794;
  // pto: %281
  uint64_t v795 = (uint64_t) v40;
  TASSIGN(v794, v795);
  // pto: %283
  int64_t v796 = v791 < v53 ? v53 : v791;
  // pto: %285
  __gm__ bfloat16_t* v797 = PTOAS__GLOBAL_TENSOR_DATA(v136);
  // pto: %285
  const int64_t v798 = 0;
  // pto: %285
  const int64_t v799 = 4096;
  // pto: %285
  pto::Shape<1, 1, 1, 1, 512> v800 = pto::Shape<1, 1, 1, 1, 512>();
  // pto: %285
  pto::Stride<4096, 4096, 4096, 4096, 1> v801 = pto::Stride<4096, 4096, 4096, 4096, 1>();
  // pto: %285
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v802 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v797 + (v798 + v796 * v799 + v678), v800, v801);
  pipe_barrier(PIPE_MTE3);
  TSTORE(v802, v794);
  // pto: %286, %287, %288
  int64_t v803 = (int64_t) ((uint64_t) ((int64_t) ((uint64_t) v147 + (uint64_t) v63) / v49) * (uint64_t) v49);
  // pto: %289
  int64_t v804 = (int64_t) ((uint64_t) v803 + (uint64_t) v144);
  // pto: %101
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v805 = Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v47);
  // pto: %101
  uint64_t v806 = (uint64_t) v41;
  TASSIGN(v805, v806);
  // pto: %293
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v807;
  // pto: %293
  uint64_t v808 = (uint64_t) v41;
  TASSIGN(v807, v808);
  // pto: %295
  int64_t v809 = v804 < v53 ? v53 : v804;
  // pto: %297
  __gm__ bfloat16_t* v810 = PTOAS__GLOBAL_TENSOR_DATA(v136);
  // pto: %297
  const int64_t v811 = 0;
  // pto: %297
  const int64_t v812 = 4096;
  // pto: %297
  pto::Shape<1, 1, 1, 1, 512> v813 = pto::Shape<1, 1, 1, 1, 512>();
  // pto: %297
  pto::Stride<4096, 4096, 4096, 4096, 1> v814 = pto::Stride<4096, 4096, 4096, 4096, 1>();
  // pto: %297
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v815 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v810 + (v811 + v809 * v812 + v694), v813, v814);
  pipe_barrier(PIPE_MTE3);
  TSTORE(v815, v807);
  // pto: %298, %299, %300
  int64_t v816 = (int64_t) ((uint64_t) ((int64_t) ((uint64_t) v147 + (uint64_t) v64) / v49) * (uint64_t) v49);
  // pto: %301
  int64_t v817 = (int64_t) ((uint64_t) v816 + (uint64_t) v144);
  // pto: %102
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v818 = Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v47);
  // pto: %102
  uint64_t v819 = (uint64_t) v42;
  TASSIGN(v818, v819);
  // pto: %305
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v820;
  // pto: %305
  uint64_t v821 = (uint64_t) v42;
  TASSIGN(v820, v821);
  // pto: %307
  int64_t v822 = v817 < v53 ? v53 : v817;
  // pto: %309
  __gm__ bfloat16_t* v823 = PTOAS__GLOBAL_TENSOR_DATA(v136);
  // pto: %309
  const int64_t v824 = 0;
  // pto: %309
  const int64_t v825 = 4096;
  // pto: %309
  pto::Shape<1, 1, 1, 1, 512> v826 = pto::Shape<1, 1, 1, 1, 512>();
  // pto: %309
  pto::Stride<4096, 4096, 4096, 4096, 1> v827 = pto::Stride<4096, 4096, 4096, 4096, 1>();
  // pto: %309
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v828 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v823 + (v824 + v822 * v825 + v710), v826, v827);
  pipe_barrier(PIPE_MTE3);
  TSTORE(v828, v820);
  // pto: %310, %311, %312
  int64_t v829 = (int64_t) ((uint64_t) ((int64_t) ((uint64_t) v147 + (uint64_t) v65) / v49) * (uint64_t) v49);
  // pto: %313
  int64_t v830 = (int64_t) ((uint64_t) v829 + (uint64_t) v144);
  // pto: %103
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v831 = Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v47);
  // pto: %103
  uint64_t v832 = (uint64_t) v43;
  TASSIGN(v831, v832);
  // pto: %317
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v833;
  // pto: %317
  uint64_t v834 = (uint64_t) v43;
  TASSIGN(v833, v834);
  // pto: %319
  int64_t v835 = v830 < v53 ? v53 : v830;
  // pto: %321
  __gm__ bfloat16_t* v836 = PTOAS__GLOBAL_TENSOR_DATA(v136);
  // pto: %321
  const int64_t v837 = 0;
  // pto: %321
  const int64_t v838 = 4096;
  // pto: %321
  pto::Shape<1, 1, 1, 1, 512> v839 = pto::Shape<1, 1, 1, 1, 512>();
  // pto: %321
  pto::Stride<4096, 4096, 4096, 4096, 1> v840 = pto::Stride<4096, 4096, 4096, 4096, 1>();
  // pto: %321
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v841 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v836 + (v837 + v835 * v838 + v726), v839, v840);
  pipe_barrier(PIPE_MTE3);
  TSTORE(v841, v833);
  // pto: %322, %323, %324
  int64_t v842 = (int64_t) ((uint64_t) ((int64_t) ((uint64_t) v147 + (uint64_t) v66) / v49) * (uint64_t) v49);
  // pto: %325
  int64_t v843 = (int64_t) ((uint64_t) v842 + (uint64_t) v144);
  // pto: %104
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v844 = Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v47);
  // pto: %104
  uint64_t v845 = (uint64_t) v44;
  TASSIGN(v844, v845);
  // pto: %329
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v846;
  // pto: %329
  uint64_t v847 = (uint64_t) v44;
  TASSIGN(v846, v847);
  // pto: %331
  int64_t v848 = v843 < v53 ? v53 : v843;
  // pto: %333
  __gm__ bfloat16_t* v849 = PTOAS__GLOBAL_TENSOR_DATA(v136);
  // pto: %333
  const int64_t v850 = 0;
  // pto: %333
  const int64_t v851 = 4096;
  // pto: %333
  pto::Shape<1, 1, 1, 1, 512> v852 = pto::Shape<1, 1, 1, 1, 512>();
  // pto: %333
  pto::Stride<4096, 4096, 4096, 4096, 1> v853 = pto::Stride<4096, 4096, 4096, 4096, 1>();
  // pto: %333
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v854 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v849 + (v850 + v848 * v851 + v742), v852, v853);
  pipe_barrier(PIPE_MTE3);
  TSTORE(v854, v846);
  // pto: %334, %335, %336
  int64_t v855 = (int64_t) ((uint64_t) ((int64_t) ((uint64_t) v147 + (uint64_t) v67) / v49) * (uint64_t) v49);
  // pto: %337
  int64_t v856 = (int64_t) ((uint64_t) v855 + (uint64_t) v144);
  // pto: %105
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v857 = Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v46, v47);
  // pto: %105
  uint64_t v858 = (uint64_t) v45;
  TASSIGN(v857, v858);
  // pto: %341
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v859;
  // pto: %341
  uint64_t v860 = (uint64_t) v45;
  TASSIGN(v859, v860);
  // pto: %343
  int64_t v861 = v856 < v53 ? v53 : v856;
  // pto: %345
  __gm__ bfloat16_t* v862 = PTOAS__GLOBAL_TENSOR_DATA(v136);
  // pto: %345
  const int64_t v863 = 0;
  // pto: %345
  const int64_t v864 = 4096;
  // pto: %345
  pto::Shape<1, 1, 1, 1, 512> v865 = pto::Shape<1, 1, 1, 1, 512>();
  // pto: %345
  pto::Stride<4096, 4096, 4096, 4096, 1> v866 = pto::Stride<4096, 4096, 4096, 4096, 1>();
  // pto: %345
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v867 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v862 + (v863 + v861 * v864 + v758), v865, v866);
  pipe_barrier(PIPE_MTE3);
  TSTORE(v867, v859);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}