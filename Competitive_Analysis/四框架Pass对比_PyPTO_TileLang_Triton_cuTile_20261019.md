# 四框架编译 Pass 对比：PyPTO / TileLang-Ascend / Triton / cuTile

- **整理日期**：2026-10-19
- **取材方式**：全部直接读源码，非文档摘要。各框架对应的代码位置见文末「来源与核实方式」。
- **标注约定**（沿用本目录既有写法）：【事实·内】= 本仓库内可核实；【事实·外】= 外部仓库或官方文档可核实；【推断】= 基于前两者的判断。

---

## 0. 口径说明（先读）

**计数单位不可直接比较。** 四家统计粒度不同：

- Triton 数的是 `add_stages()` 注册的各 stage 内去重后的 `add_*` pass 调用名；
- TileLang-Ascend 与 PTOAS 数的是 transform 源文件数；
- PyPTO 数的是 `Default` 策略里的流水线条目。

因此下表的数字**只用于看量级与可见性，不能用来论证"谁的优化更强"**。一个 46 行的 pass 文件和一个 46 个 pass 的 stage 不是一个意思。

**另一个前提**：Triton-Ascend 的 pass（`TritonToLinalg` / `TritonToHIVM`）未并入本文。它复用上游 `make_ttir` 之后走 Linalg / HIVM 另一条链，与上游 NVIDIA 后端不是同一组 pass；并入会使"Triton"一列语义混淆。

---

## 1. 规模与中间态可见性

| 框架 | pass 数 | 计数依据 | 中间态可见性 |
| ---- | ------- | -------- | ------------ |
| **Triton**（NVIDIA 后端） | ≈81 | ttir 8 + ttgir 46 + llir 27 | **最高**：缓存目录里直接存有 `.ttir` / `.ttgir` / `.llir` / `.ptx` / `.cubin` / `.json`；`MLIR_ENABLE_DUMP` 可 dump 每个 pass 前的 IR |
| **TileLang-Ascend**（`ascendc_pto`） | 51 | `src/transform/*.cc`，其中 `ascend_*` 17 个 | 高：全部开源，pass 有名有文件 |
| **PyPTO**（+ PTOAS） | 42 + 70 以上 | `Default` 策略 42 条；PTOAS 侧 `Passes/` 31 个命名 pass、`Lowering/` 11 个，另有 `InsertSync/`、`TileFusion/`、`VPTO*/` 等 67 个源文件 | 高：48 篇 pass 文档、逐 pass IR 快照、IRProperty 逐 pass 校验、printer↔parser roundtrip |
| **cuTile**（`NVIDIA/cuda-tile`） | **4** | `include/cuda_tile/Dialect/CudaTile/Transforms/Passes.td` 的全部内容 | 低：tile 级优化在闭源 `tileiras` 内，外部只能看到 Tile IR 字节码 |

---

## 2. 按职责对照

### 2.1 职责归属总览

每一行是一件**必须有人做**的事；"闭源"表示该职责确实存在，但实现不在公开代码里。

| 职责 | Triton | TileLang-Ascend | PyPTO | cuTile |
| ---- | ------ | --------------- | ----- | ------ |
| 语言规范化 / 控制流 | 自己（13） | 自己（8） | 自己（8） | **公开的 3 个实质 pass 全在这里** |
| 布局 / 编码推导 | 自己（7） | 自己（1） | 自己（3）+ PTOAS | 闭源 |
| Tile 切分 / 矩阵加速 | 自己（6） | 自己（2） | 自己（4）+ PTOAS | 闭源 |
| 向量化 / 掩码 | 无独立 pass | **自己（9，最重）** | 自己（2） | 闭源 |
| 内存空间推导 | 并入分配 | 自己（2） | 自己（2）+ PTOAS | 闭源 |
| 内存复用 / 地址分配 | 自己（4） | 自己（6） | 自己（3）+ PTOAS（5） | 闭源 |
| **同步插入** | 自己（8） | 自己（8） | **整体外包给 PTOAS** | 闭源 |
| 软件流水 / 异步 | 自己（7） | 自己（5） | 自己（3）+ PTOAS（3） | 闭源 |
| warp / 核特化 | 自己（5） | 自己（3） | 自己（3）+ PTOAS（3） | 闭源 |
| 跨核数据交换 | 无 | 自己（2） | 自己（2） | 闭源 |
| **任务图 / 编排** | 无 | 无 | **自己（11）** | 无 |
| **分布式 / 通信** | 无 | 无 | **自己（4）** | 无 |
| 调试 / 校验 pass | **sanitizer（4）+ debug info（3）** | 1 | **IRProperty 契约 + PTOAS 校验（5）** | 1（公开 4 个之一） |

### 2.2 逐项明细

#### 语言规范化 / 控制流

- **Triton**：`inliner`、`canonicalizer`、`cse`、`sccp`、`symbol_dce`、`loop_unroll`、`combine`、`reorder_broadcast`、`triton_licm`、`loop_aware_cse`、`fuse_nested_loops`、`combine_tensor_select_and_if`、`scf_to_cf`
- **TileLang-Ascend**：`frontend_legalize`、`simplify`、`flatten_buffer`、`loop_partition`、`merge_if_stmt`、`if_stmt_binding`、`config_index_bitwidth`、`ascend_lower_opaque_block`
- **PyPTO**：`inline_functions`、`unroll_loops`、`ctrl_flow_transform`、`convert_to_ssa`、`simplify`、`normalize_stmt_structure`、`flatten_call_expr`、`fold_no_op_reshape`
- **cuTile**：`CudaTileLICM`、`LoopSplit`、`FuseFMA`

#### 布局 / 编码推导

- **Triton**：`assign_cga_layouts`、`remove_layout_conversions`（流水线中出现三次）、`infer_coalesced_encodings`、`resolve_auto_encodings`、`optimize_descriptor_encoding`、`optimize_tmem_layouts`、`coalesce`
- **TileLang-Ascend**：`layout_inference`
- **PyPTO**：`resolve_backend_op_layouts`、`materialize_tensor_strides`、`canonicalize_tile_slice`；PTOAS 侧 `InferPTOLayout`
- **cuTile**：闭源

#### Tile 切分 / 矩阵加速

- **Triton**：`accelerate_matmul`、`check_matmul_two_cta`、`optimize_dot_operands`、`lower_mma`、`f32_dot_tc`、`promote_lhs_to_tmem`
- **TileLang-Ascend**：`lower_tile_op`、`cluster_planning`
- **PyPTO**：`convert_tensor_to_tile_ops`、`lower_composite_ops`、`flatten_tile_nd_to_2d`、`auto_tile_matmul_l0`；PTOAS 侧 `PTOMaterializeTileOpSections`、`TileFusion/`
- **cuTile**：闭源

#### 向量化 / 掩码

- **Triton**：无独立 pass，由 layout 与 `coalesce` 承担
- **TileLang-Ascend**：`loop_vectorize`、`loop_vectorize_dynamic`、`vectorize_loop`、`legalize_vectorized_loop`、`ascend_lower_parallel_to_vector`、`ascend_vector_instruction_selection`、`ascend_vector_mask`、`ascend_vector_mask_legalize`、`ascend_tail_mask_propagation`
- **PyPTO**：`lower_auto_vector_split`、`split_vector_kernel`
- **cuTile**：闭源

#### 内存空间推导

- **Triton**：并入分配阶段
- **TileLang-Ascend**：`ascend_infer_buffer_scope`、`ascend_collect_buffer_shape`
- **PyPTO**：`infer_tile_memory_space`、`init_mem_ref`；PTOAS 侧 `InferPTOMemScope`
- **cuTile**：闭源

#### 内存复用 / 地址分配

- **Triton**：`allocate_shared_memory_nv`、`allocate_tensor_memory`、`allocate_global_scratch_memory`、`reduce_data_duplication`
- **TileLang-Ascend**：`ascend_memory_planning`、`ascend_storage_rewrite`、`merge_shared_memory_allocations`、`allocate_tmp_buffer`、`multi_version_buffer_rewriter`、`ascend_workspace_reduction`
- **PyPTO**：`materialize_semantic_aliases`、`memory_reuse`、`allocate_memory_addr`；PTOAS 侧 `PTOPlanMemoryModern`、`OptMemPlanForPipeline`、`PTOResolveBufferSelect`、`PTOResolveReservedBuffersPass`、`SlotAffineAnalysis`
- **cuTile**：闭源

> 补充【事实·内】：PyPTO 的这三个 pass 可通过 `memory_planner=PTOAS` 跳过，改由 ptoas 的 `PlanMemory` 在 `--pto-level=level2` 下接管；默认 `PYPTO` 模式则烘出物理地址并用 `--pto-level=level3`。

#### 同步插入

- **Triton**：`fence_insertion`、`proxy_fence_insertion`、`membar`、`tmem_barrier_insertion`、`optimize_mbarrier_arrivals`、`cluster_barrier_mbar_allocator`、`initialize_ws_cluster_barriers`、`hoist_mbarrier_lifecycle`
- **TileLang-Ascend**：`ascend_sync_insert`、`ascend_sync_insert_vs`、`thread_storage_sync`、`thread_partial_sync`、`storage_access`、`eliminate_storage_sync_for_mbarrier`、`inject_fence_proxy`、`wgmma_sync_rewriter`
- **PyPTO**：**自身不做**。由 PTOAS 的 `InsertSync/` 子系统承担（12 个源文件：`MemoryDependentAnalyzer`、`SyncEventIdAllocation`、`RemoveRedundantSync`、`RemoveSlotRedundantSync`、`SyncMacroModel`、`SyncCodegen` 等），另有 `LoweringSyncToPipe`、`PTOInjectBarrierAllSync`、`PTORemoveRedundantBarrier`
- **cuTile**：闭源

#### 软件流水 / 异步

- **Triton**：`assign_latencies`、`schedule_loops`、`pipeline`、`prefetch`、`coalesce_async_copy`、`tma_lowering`、`interleave_tmem`
- **TileLang-Ascend**：`inject_pipeline`、`pipeline_planning`、`cross_core_pipeline`、`inject_ptx_async_copy`、`multi_version_buffer_rewriter`
- **PyPTO**：`lower_pipeline_loops`、`skew_cross_core_pipeline`、`inject_gm_pipe_buffer`；PTOAS 侧 `OptMemPlanForPipeline`、`PTOAssignDefaultFrontendPipeIdPass`、`PTOInferValidatePipeInitPass`
- **cuTile**：闭源

#### warp / 核特化

- **Triton**：`hopper_warpspec`、`warp_specialize`、`optimize_partition_warps`、`allocate_warp_groups`、`warp_specialize_to_llvm`
- **TileLang-Ascend**：`warp_specialized_rewriter`（CUDA 侧）、`ascend_combinecv`、`annotate_device_regions`
- **PyPTO**：`expand_mixed_kernel`（拆 AIC / AIV / Group）、`split_vector_kernel`、`stamp_tfree_split`；PTOAS 侧 `PTOOutlineSIMTSections`、`PTOMaterializeSIMTPersistentFragment`、`PTOPromotePersistentFragmentLoopsPass`
- **cuTile**：闭源

#### 跨核数据交换

- **Triton**：无——SM 之间经 global memory，不做核间配对
- **TileLang-Ascend**：`cross_core_pipeline`、`ascend_pto_save_buffer_shape`
- **PyPTO**：`skew_cross_core_pipeline`、`stamp_tfree_split`（`tpush` / `tpop` 配对）
- **cuTile**：闭源

#### 任务图 / 编排（仅 PyPTO）

`outline_hierarchy_scopes`、`outline_incore_scopes`、`outline_cluster_scopes`、`optimize_orch_tensors`、`derive_call_directions`、`auto_derive_task_dependencies`、`expand_manual_phase_fence`、`normalize_return_order`、`canonicalize_io_order`、`materialize_runtime_scopes`、`classify_iter_arg_carry`

#### 分布式 / 通信（仅 PyPTO）

`synthesize_allreduce_signals`、`materialize_comm_domain_scopes`、`lower_host_tensor_collectives`、`materialize_dist_tensor_ctx`

#### 调试 / 校验 pass

- **Triton**：`fp_sanitizer`、`global_sanitizer`、`concurrency_sanitizer`、`prepare_consan_captures`、`di_scope`、`di_local_variable`、`version_info`
- **TileLang-Ascend**：`legalize_safe_memory_access`
- **PyPTO**：IRProperty verifier（25 个编译期不变量）+ 逐 pass dump + roundtrip 校验；PTOAS 侧 `PTOValidateVMIIR`、`PTOValidateVPTOIR`、`PTOVerifyTFreePass`、`PTOValidatePhysicalSectionBoundaries`、`PTOPrintAddressAnalysis`
- **cuTile**：`SynthesizeDebugInfoScopes`

---

## 3. 完整 pass 清单

### 3.1 Triton · NVIDIA 后端

**ttir（8）**

```
inliner · rewrite_tensor_descriptor_to_pointer · canonicalizer · combine
reorder_broadcast · cse · symbol_dce · loop_unroll
```

**ttgir（46）**

```
convert_to_ttgpuir · to_clc · coalesce · f32_dot_tc · assign_cga_layouts
remove_layout_conversions · optimize_cta_locality · optimize_thread_locality
accelerate_matmul · check_matmul_two_cta · optimize_dot_operands
optimize_descriptor_encoding · loop_aware_cse · fuse_nested_loops · canonicalizer
triton_licm · combine_tensor_select_and_if · hopper_warpspec · assign_latencies
schedule_loops · pipeline · optimize_accumulator_init · hoist_tmem_alloc
promote_lhs_to_tmem · warp_specialize · optimize_partition_warps
remove_tmem_tokens · prefetch · coalesce_async_copy · optimize_tmem_layouts
tmem_load_reduce · tma_lowering · lower_clc · interleave_tmem
reduce_data_duplication · reorder_instructions · symbol_dce · fence_insertion
lower_mma · hoist_mbarrier_lifecycle · sccp · cse · fp_sanitizer · inliner
infer_coalesced_encodings · resolve_auto_encodings
```

**llir（27）**

```
global_sanitizer · combine_tensor_select_and_if · allocate_warp_groups
scf_to_cf · inliner · prepare_consan_captures · allocate_shared_memory_nv
allocate_tensor_memory · proxy_fence_insertion · membar · tmem_barrier_insertion
optimize_mbarrier_arrivals · concurrency_sanitizer · canonicalizer · cse
cluster_barrier_mbar_allocator · allocate_global_scratch_memory · to_llvmir
initialize_ws_cluster_barriers · canonicalize_llvm_ir · warp_specialize_to_llvm
nvgpu_to_llvm · symbol_dce · nvvm_to_llvm · di_scope · di_local_variable
version_info
```

**其后**：`make_ptx`（LLVM NVPTX 出 PTX）→ `make_cubin`（`subprocess` 调 `ptxas`）。
**旁路**：`language == Language.GLUON` 时 stage 变为 `glir → ttgir → llir → ptx → cubin`，跳过 ttir。

### 3.2 TileLang-Ascend（`ascendc_pto`）

**Ascend 专用（17）**

```
ascend_collect_buffer_shape · ascend_combinecv · ascend_host
ascend_infer_buffer_scope · ascend_lower_opaque_block
ascend_lower_parallel_to_vector · ascend_memory_planning
ascend_pto_save_buffer_shape · ascend_storage_rewrite · ascend_sync_insert
ascend_sync_insert_vs · ascend_tail_mask_propagation
ascend_vector_instruction_selection · ascend_vector_mask
ascend_vector_mask_legalize · ascend_vid_reduction · ascend_workspace_reduction
```

**通用 / 跨后端共享**

```
layout_inference · lower_tile_op · flatten_buffer · simplify · loop_vectorize
loop_vectorize_dynamic · vectorize_loop · legalize_vectorized_loop
loop_partition · inject_pipeline · pipeline_planning · cluster_planning
cross_core_pipeline · merge_shared_memory_allocations
multi_version_buffer_rewriter · allocate_tmp_buffer · storage_access
thread_storage_sync · thread_partial_sync · frontend_legalize · merge_if_stmt
if_stmt_binding · config_index_bitwidth · annotate_device_regions
legalize_safe_memory_access · lower_device_storage_access_info · make_packed_api
```

**CUDA 专用（仍保留在同一目录）**

```
warp_specialized_rewriter · inject_tma_barrier · inject_ptx_async_copy
lower_hopper_intrin · wgmma_sync_rewriter · eliminate_storage_sync_for_mbarrier
inject_fence_proxy
```

### 3.3 PyPTO（`Default` 策略，按执行顺序）

**通用前缀（7）**

```
inline_functions · unroll_loops · ctrl_flow_transform · convert_to_ssa
simplify · normalize_stmt_structure · flatten_call_expr
```

**Tensor 段（5）**

```
outline_hierarchy_scopes · outline_incore_scopes · outline_cluster_scopes
convert_tensor_to_tile_ops · optimize_orch_tensors
```

**Tile / PTO 段（30）**

```
lower_composite_ops · flatten_tile_nd_to_2d · auto_tile_matmul_l0
canonicalize_tile_slice · infer_tile_memory_space · resolve_backend_op_layouts
lower_auto_vector_split · expand_mixed_kernel · inject_gm_pipe_buffer
split_vector_kernel · stamp_tfree_split · normalize_return_order
skew_cross_core_pipeline · lower_pipeline_loops · canonicalize_io_order
materialize_tensor_strides · init_mem_ref · materialize_semantic_aliases
memory_reuse · allocate_memory_addr · fold_no_op_reshape
fuse_create_assemble_to_slice · derive_call_directions
auto_derive_task_dependencies · expand_manual_phase_fence
synthesize_allreduce_signals · materialize_comm_domain_scopes
lower_host_tensor_collectives · materialize_dist_tensor_ctx · simplify
materialize_runtime_scopes · classify_iter_arg_carry
```

**PTOAS 侧 · `Passes/`（31）**

```
InferPTOLayout · InferPTOMemScope · LoweringSyncToPipe · OptMemPlanForPipeline
PTOAnalyzeSIMTPersistentFragment · PTOAssignDefaultFrontendPipeIdPass
PTOConvertSCFToCFWithLoopHintsPass · PTOInferValidatePipeInitPass
PTOInjectBarrierAllSync · PTOMaterializeImplicitTmp
PTOMaterializeSIMTPersistentFragment · PTOMaterializeTileOpSections
PTONarrowVPTOLoopCounters · PTONormalizeUncoveredTileSections
PTOOutlineSIMTSections · PTOPlanMemoryModern · PTOPrintAddressAnalysis
PTOPromotePersistentFragmentLoopsPass · PTORemoveRedundantBarrier
PTOResolveBufferSelect · PTOResolveReservedBuffersPass · PTOUnrollLoopsPass
PTOVPTOPtrBoundary · PTOValidatePhysicalSectionBoundaries · PTOValidateVMIIR
PTOValidateVPTOIR · PTOVerifyTFreePass · PTOWrapFunctionsInSectionsPass
SIMTPersistentFragmentAnalysis · SlotAffineAnalysis · SubviewSlotAnalysis
```

**PTOAS 侧 · 其他子系统**：`InsertSync/`（12 个源文件）、`Lowering/`（11）、`TileFusion/`、`VMI/`、`VMIToVPTO/`、`VPTO/`、`VPTOScheduler/`、`BufidSync/`、`PTOToEmitC/`

### 3.4 cuTile · CUDA Tile IR

**公开 pass（全部 4 个）**

| Pass | 作用（官方 summary 转述） |
| ---- | ------------------------ |
| `SynthesizeDebugInfoScopes` | 为 module 合成 debug info scope 信息 |
| `FuseFMA` | 把乘加 / 乘减融合为 FMA（非数值保持） |
| `LoopSplit` | 当 if 条件把归纳变量与循环不变量比较时拆分循环（带 `split-threshold` 选项） |
| `CudaTileLICM` | 把循环不变量提出 `cuda_tile.for` / `cuda_tile.loop` |

**闭源部分**：`tileiras`——字节码到 cubin，官方类比 `ptxas`；布局、同步、内存规划、Tensor Core 映射均在其中，pass 列表未公开。

**配套工具**：`cuda-tile-opt`（MLIR opt 驱动）、`cuda-tile-optimize`、`cuda-tile-translate`（MLIR → 字节码）、`cuda-tile-tblgen`。

---

## 4. 结论

**结论一｜cuTile 不是 pass 少，是不公开。**
【事实·外】`NVIDIA/cuda-tile` 的 `Passes.td` 只有四个 pass，全部是通用循环与标量优化。布局、同步、内存规划、Tensor Core 映射一个都不在其中。
【推断】NVIDIA 开源的是**方言与契约**（dialect、字节码格式、Python 绑定、一致性测试），不是优化实现。对外宣称的"开放"应理解为"可以对接"，而非"可以审查优化决策"。

**结论二｜Triton 把新硬件特性直接 pass 化。**
【事实·外】ttgir + llir 中可数出 `tmem_*` 六个、`mbarrier` 相关三个、`warp_specialize` 四个，外加 `tma_lowering`、`cluster_barrier_mbar_allocator`。
【推断】这是其 pass 总数最高的直接原因，代价是后端与具体架构强耦合——每代新硬件都要新增一批 pass。

**结论三｜同步插入的归属，四家各不相同。**
【事实】Triton 与 TileLang 各自用 8 个 pass 自己做；cuTile 在闭源编译器内做；PyPTO 自身不做，整体外包给 PTOAS 的 `InsertSync/` 子系统。
【推断】对 PyPTO 是双刃：pass 流水线无需理解硬件事件模型，但这段能力不在自己仓内，跨仓版本必须对齐（当前由 `toolchain/versions.env` 钉 ptoas 版本、`runtime/pto_isa.pin` 钉 ISA 版本）。

**结论四｜有两类 pass 只有 PyPTO 有。**
【事实·内】任务图编排 11 个、分布式通信 4 个，其余三家对应行全空。
【推断】这是 pass 层面唯一无人竞争的地带。差异化叙事应钉在"编排与通信被编译进产物"这一点上，而非"有 tile 虚拟 ISA"或"抽象层级高"——后两者已被 cuTile / CUDA Tile IR 独立验证，不再独占。

**结论五｜"验证"有两条互不替代的路线。**
【事实】Triton 有运行时 sanitizer（`fp_sanitizer`、`global_sanitizer`、`concurrency_sanitizer`）；PyPTO 有编译期契约（IRProperty + 25 个不变量 + printer↔parser roundtrip）。
【推断】前者抓的是真实执行中的数值与并发错误，后者抓的是编译器自身的 bug，不能互相替代。**PyPTO 缺前者，Triton 缺后者**；若要在"可信证据链"上立论，补运行时 sanitizer 是最直接的缺口。

---

## 5. 来源与核实方式

| 框架 | 代码位置 |
| ---- | -------- |
| Triton | `triton-lang/triton` → `third_party/nvidia/backend/compiler.py`（`add_stages`、`make_ttir`、`make_ttgir`、`make_llir`、`make_ptx`、`make_cubin`） |
| TileLang-Ascend | `tile-ai/tilelang-ascend`（`ascendc_pto` 分支）→ `src/transform/`、`tilelang/engine/phase.py`、`tilelang/engine/lower.py` |
| PyPTO | 本仓 `repo/pto` → `python/pypto/ir/pass_manager.py`、`docs/zh-cn/dev/passes/` |
| PTOAS | 本仓 `repo/PTOAS` → `lib/PTO/Transforms/{Passes,Lowering,InsertSync,TileFusion,VPTO*}/` |
| cuTile | `NVIDIA/cuda-tile` → `include/cuda_tile/Dialect/CudaTile/Transforms/Passes.td`、`tools/` |

复核命令示例（Triton 侧）：

```bash
sed -n '/def make_ttgir/,/def make_llir/p' third_party/nvidia/backend/compiler.py \
  | grep -oE "add_[a-z_0-9]+" | awk '!seen[$0]++'
```

---

## 6. 未覆盖与待核实

1. **Triton-Ascend 的 pass 未并入**。其链路为 TTIR → `TritonToLinalg` / `TritonToHIVM` → `triton-mlir-opt` → `bishengir-opt` → `hivmc` / `bisheng`，`third_party/ascend/AscendNPU-IR` 作为子模块。需要时应单独成列，不要并入"Triton"列。
2. **TileLang-Ascend 的 `npuir` 分支未计入**。该分支 `3rdparty/` 只有 `AscendNPU-IR`（无 `pto-isa`），codegen 为 `codegen_npuir_api.cc` / `codegen_npuir_dev.cc`，pass 分布与 `ascendc_pto` 分支不同。
3. **cuTile 闭源部分的 pass 规模未知**。只能确认职责存在，无法计数。
4. **PyPTO 与 PTOAS 的 pass 数存在重叠语义**：部分职责（内存规划）可按 `memory_planner` 配置在两边切换，计数时不应简单相加。
5. **TileLang-Ascend 的 51 个文件中，部分为 analysis 而非 transform**，未逐个区分。

---

## 7. 相关材料

- 本目录：`PyPTO3.0_对位NVIDIA_TileLang_Triton_深度对比.md`（战略层对比）、`PyPTO3_竞品资料收集审核稿.md`（竞品分层与来源清单）
- PyPTO 自身流水线：`repo/pto/docs/zh-cn/dev/passes/00-pass_manager.md` 及同目录逐 pass 文档（编号与执行顺序一致）
