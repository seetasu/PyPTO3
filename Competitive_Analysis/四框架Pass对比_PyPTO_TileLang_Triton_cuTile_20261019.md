# 四框架编译 Pass 对比：PyPTO / TileLang-Ascend / Triton / cuTile

- **整理日期**：2026-10-19
- **取材方式**：全部直接读源码，非文档摘要。各框架对应的代码位置见「来源与核实方式」。
- **标注约定**（沿用本目录既有写法）：【事实·内】= 本仓库内可核实；【事实·外】= 外部仓库或官方文档可核实；【推断】= 基于前两者的判断。

---

## 0. 口径说明（先读）

**必须区分两种口径，混用会得出相反结论。**

| 口径 | 含义 | 适合回答 |
| ---- | ---- | -------- |
| **A. 默认流水线调用序列** | 编译一次实际跑过的 pass 条目数（同一个 pass 可重复出现） | "默认编译一个算子经过多少步" |
| **B. 注册的 pass 定义数** | 框架里一共定义了多少个可用 pass（TableGen `def` 或 pass 工厂） | "这个编译器的工具箱有多大" |

两者差距很大：Triton 口径 A 约 81、口径 B 为 56；PTOAS 口径 B 为 83，但其口径 A 随 `--pto-level` 组装而变。**源文件数是第三种口径，既不等于 A 也不等于 B**（一个文件可注册多个 pass，也可能只含 analysis），本文不再用它作为主指标。

因此下表的数字**只用于看量级与可见性，不能用来论证"谁的优化更强"**。

> **勘误（2026-10-19 复核）**：本文初版把 PyPTO 记为 42、PTOAS 记为"31 个命名 pass + 70 以上源文件"，两处均不准确。前者因去重命令吞掉了重复出现的 `simplify`、并漏掉被 `lambda` 包裹的 `auto_derive_task_dependencies`；后者只统计了 `lib/PTO/Transforms/Passes/` 一个目录的源文件，漏掉 VMI / VPTO / 融合等全部系列。已按下表更正。

**另一个前提**：Triton-Ascend 的 pass（`TritonToLinalg` / `TritonToHIVM`）未并入本文。它复用上游 `make_ttir` 之后走 Linalg / HIVM 另一条链，与上游 NVIDIA 后端不是同一组 pass；并入会使"Triton"一列语义混淆。

---

## 1. 规模与中间态可见性

| 框架 | 口径 A<br>流水线条目 | 口径 B<br>注册 pass 数 | 中间态可见性 |
| ---- | ------------------- | --------------------- | ------------ |
| **Triton**（NVIDIA 后端） | **≈81**<br><small>ttir 8 + ttgir 46 + llir 27，去重后</small> | **56**<br><small>Triton 6 + TritonGPU 23 + TritonNvidiaGPU 20 + Conversion 2 + NVGPUToLLVM 5</small> | **最高**：缓存目录里直接存有 `.ttir` / `.ttgir` / `.llir` / `.ptx` / `.cubin` / `.json`；`MLIR_ENABLE_DUMP` 可 dump 每个 pass 前的 IR |
| **TileLang-Ascend**（`ascendc_pto`） | 未逐一核实顺序 | **51 个 transform 源文件**<br><small>其中 `ascend_*` 17 个；该仓用 C++ 注册而非单一 TableGen 清单，故此格为源文件口径</small> | 高：全部开源，pass 有名有文件 |
| **PyPTO** | **44**<br><small>`Default` 策略：前缀 7 + Tensor 5 + Tile/PTO 32；其中 `simplify` 出现两次，故不同 pass 为 43</small> | 43 个 pass 工厂 | 高：48 个 pass 文档文件（逐 pass 文档 01–42 共 42 篇）、逐 pass IR 快照、IRProperty 逐 pass 校验、printer↔parser roundtrip |
| **PTOAS**（PyPTO 的汇编层） | 随 `--pto-level` 组装 | **83**<br><small>`include/PTO/Transforms/Passes.td` 的 `def` 计数</small> | 高：开源，pass 有 TableGen 定义 |
| **cuTile**（`NVIDIA/cuda-tile`） | 未公开 | **4**<br><small>`Dialect/CudaTile/Transforms/Passes.td` 的全部内容</small> | 低：tile 级优化在闭源 `tileiras` 内，外部只能看到 Tile IR 字节码 |

**【推断】由此得出一个与初版相反的结论**：按口径 B，**PyPTO 生态（43 + 83 = 126）是四家里 pass 定义最多的**，而非初版暗示的"比 Triton 少"。但这个"多"一半来自 PTOAS——即一半的工具箱不在 pypto 仓内（见结论三）。

**另一处修正**：`DebugTileOptimization` 策略 = 前缀 7 + Tile/PTO 32 = **39 条**，跳过 Tensor 段 5 条。

---

## 2. 五类抽象

### 2.1 分类标准

把约 180 个 pass 归类时，**不按它在流水线的哪个阶段，而按它做的是哪种决策**。前者是各家的工程划分（阶段名彼此不可比），后者是四家共有的语义，因此才能横向对照。

判据是三个问题：

1. 它改变的是 IR 的**形状**、**词汇表**，还是**与硬件的绑定关系**？
2. 做错了会**慢**，还是会**错**？
3. 它与目标硬件**有关**还是**无关**？

按此可分为五类；其中第 III 类可再拆三个子类，细到七类为止，再细就失去抽象价值。

> **与上一版的分类差异**（便于对照）：本节把原「向量化 / 掩码」一行拆开——向量宽度选择归入第 III 类，掩码与尾块合法化归入第 IV 类，理由见第 2 问判据（掩码做错是**错**，不是慢）。原「跨核数据交换」一行同样拆开：资源绑定部分归 III，配对与同步部分归 IV。

### 2.2 五类定义

| 类别 | 它决定什么 | 做错的后果 | 与硬件相关 | 能否跨后端共享 |
| ---- | ---------- | ---------- | ---------- | -------------- |
| **I. 规范化** | IR 的形状——去掉它程序仍正确，只是后续 pass 写不下去 | 编译失败或后续 pass 失效 | 无关 | **可共享**，四家高度同质，多从 LLVM / MLIR / TVM 继承 |
| **II. 降级** | IR 的词汇表——一次单向、不可逆的表示跃迁 | 语义丢失 | 弱相关 | **部分可共享**，关节位置由目标决定 |
| **III. 硬件映射** | 抽象实体与硬件资源的绑定关系 | **慢** | 强相关 | **不可共享** |
| **IV. 正确性保障** | 什么时候必须等、必须屏蔽、必须别名 | **错，或 hang** | 强相关 | **不可共享** |
| **V. 组织与交付** | 代码怎么被切开、被调用、被观察 | 接口错、无法诊断 | 弱相关 | 取决于交付形态 |

**第 III 类的三个子类**：

| 子类 | 绑定的硬件资源 |
| ---- | -------------- |
| **III-a 空间** | 布局与编码、内存空间归属、物理地址、buffer 复用 |
| **III-b 时间** | 软件流水、异步搬运、延迟标注、多缓冲 |
| **III-c 并行单元** | warp / 核的特化与分工、tile 切分、矩阵单元映射、向量宽度 |
| **III-d 融合与指令调度** | 算子 / 循环融合的边界，以及最终指令序的排布 |

### 2.3 四家的重心分布

【推断】按各家 pass 清单逐条归类后的相对权重（非精确统计）：

| 框架 | 重心 | 说明 |
| ---- | ---- | ---- |
| **Triton** | **III 硬件映射** | 约一半 pass 在 III，且 `tmem_*`、`mbarrier`、`warp_specialize` 等 Hopper / Blackwell 专属项直接 pass 化；V 类里独有 sanitizer |
| **TileLang-Ascend** | **III + IV** | `ascend_*` 17 个几乎全落在这两类，其中向量化相关 9 个为四家最重 |
| **PyPTO**（仅 pypto 仓） | **V 组织与交付** | 编排 11 + 分布式 4；III 相对偏薄，**III-d 与 IV 整体外包给 PTOAS** |
| **PTOAS**（PyPTO 的汇编层） | **III + IV** | 83 个注册 pass 里，融合与调度 12 个、VMI 布局 12 个、同步与校验一整套；是 PyPTO 生态真正的硬件映射层 |
| **cuTile** | 公开部分只有 **I** | II / III / IV 全在闭源 `tileiras` 内 |

**三条由此得出的判断**：

1. **可共享的只有 I 和 II，III 与 IV 永远不能共享。** 这解释了 TileLang 为何能"一套前端多后端"——它共享的正是 I + II 的骨架（`LayoutInference`、`LowerTileOp`、向量化、流水注入），各后端自己补 III + IV；也解释了换平台时 kernel 要改写内存层级与同步，因为用户写的东西喂的就是 III。
2. **第 IV 类是唯一不能被 autotune 替代的。** 搜索能找到更快的配置（III 的事），但不能帮你插对同步。
3. **四家最重的那一类彼此不重叠**，所以单算子性能对标与整网能力对标会得出相反结论。

---

## 3. 按五类展开的职责矩阵

表格列出职责归属与数量，完整 pass 名在每节的明细中。"闭源"表示职责确实存在，但实现不在公开代码里。

### 3.1 第 I 类 · 规范化

| 职责 | Triton | TileLang-Ascend | PyPTO（+ PTOAS） | cuTile |
| ---- | ------ | --------------- | ---------------- | ------ |
| 语言规范化 / 控制流 | 自己（13） | 自己（8） | 自己（8） | **公开的 3 个实质 pass 全在这里** |

- **Triton**：`inliner`、`canonicalizer`、`cse`、`sccp`、`symbol_dce`、`loop_unroll`、`combine`、`reorder_broadcast`、`triton_licm`、`loop_aware_cse`、`fuse_nested_loops`、`combine_tensor_select_and_if`、`scf_to_cf`
- **TileLang-Ascend**：`frontend_legalize`、`simplify`、`flatten_buffer`、`loop_partition`、`merge_if_stmt`、`if_stmt_binding`、`config_index_bitwidth`、`ascend_lower_opaque_block`
- **PyPTO**：`inline_functions`、`unroll_loops`、`ctrl_flow_transform`、`convert_to_ssa`、`simplify`、`normalize_stmt_structure`、`flatten_call_expr`、`fold_no_op_reshape`
- **cuTile**：`CudaTileLICM`、`LoopSplit`、`FuseFMA`

### 3.2 第 II 类 · 降级

| 职责 | Triton | TileLang-Ascend | PyPTO（+ PTOAS） | cuTile |
| ---- | ------ | --------------- | ---------------- | ------ |
| 表示跃迁（换词汇表） | 自己（6） | 自己（2） | 自己（3）+ PTOAS | 闭源 |

- **Triton**：`convert_to_ttgpuir`、`to_llvmir`、`nvvm_to_llvm`、`nvgpu_to_llvm`、`lower_clc`、`canonicalize_llvm_ir`
- **TileLang-Ascend**：`lower_tile_op`、`lower_device_storage_access_info`
- **PyPTO**：`convert_tensor_to_tile_ops`、`lower_composite_ops`、`flatten_tile_nd_to_2d`；PTOAS 侧 `PTOLowerGenericOps`、`LowerPTOToUBufOps`、`PTOToEmitC/`
- **cuTile**：闭源（Tile IR → NVVM / PTX 在 `tileiras` 内）

> 【推断】这一类的位置定义了各家的阶段边界：Triton 的 `ttir → ttgir → llir` 三个关节、PyPTO 的 `IR → .pto → C++` 两个关节、cuTile 的 `Tile dialect → 字节码 → cubin` 两个关节。关节前后的 pass 集合几乎不重叠。

### 3.3 第 III 类 · 硬件映射

#### III-a 空间

| 职责 | Triton | TileLang-Ascend | PyPTO（+ PTOAS） | cuTile |
| ---- | ------ | --------------- | ---------------- | ------ |
| 布局 / 编码推导 | 自己（7） | 自己（1） | 自己（3）+ PTOAS（1） | 闭源 |
| 内存空间推导 | 并入分配 | 自己（2） | 自己（2）+ PTOAS（1） | 闭源 |
| 内存复用 / 地址分配 | 自己（4） | 自己（6） | 自己（3）+ PTOAS（5） | 闭源 |

- **布局 / 编码**
  - Triton：`assign_cga_layouts`、`remove_layout_conversions`（流水线中出现三次）、`infer_coalesced_encodings`、`resolve_auto_encodings`、`optimize_descriptor_encoding`、`optimize_tmem_layouts`、`coalesce`
  - TileLang-Ascend：`layout_inference`
  - PyPTO：`resolve_backend_op_layouts`、`materialize_tensor_strides`、`canonicalize_tile_slice`；PTOAS 侧 `InferPTOLayout`
- **内存空间推导**
  - TileLang-Ascend：`ascend_infer_buffer_scope`、`ascend_collect_buffer_shape`
  - PyPTO：`infer_tile_memory_space`、`init_mem_ref`；PTOAS 侧 `InferPTOMemScope`
- **内存复用 / 地址分配**
  - Triton：`allocate_shared_memory_nv`、`allocate_tensor_memory`、`allocate_global_scratch_memory`、`reduce_data_duplication`
  - TileLang-Ascend：`ascend_memory_planning`、`ascend_storage_rewrite`、`merge_shared_memory_allocations`、`allocate_tmp_buffer`、`multi_version_buffer_rewriter`、`ascend_workspace_reduction`
  - PyPTO：`memory_reuse`、`allocate_memory_addr`、`fuse_create_assemble_to_slice`；PTOAS 侧 `PTOPlanMemoryModern`、`OptMemPlanForPipeline`、`PTOResolveBufferSelect`、`PTOResolveReservedBuffersPass`、`SlotAffineAnalysis`

> 【事实·内】PyPTO 的 `memory_reuse` 与 `allocate_memory_addr` 可通过 `memory_planner=PTOAS` 跳过，改由 ptoas 的 `PlanMemory` 在 `--pto-level=level2` 下接管；默认 `PYPTO` 模式烘出物理地址并用 `--pto-level=level3`。**因此这一行的两家数字不可相加。**

#### III-b 时间

| 职责 | Triton | TileLang-Ascend | PyPTO（+ PTOAS） | cuTile |
| ---- | ------ | --------------- | ---------------- | ------ |
| 软件流水 / 异步 | 自己（7） | 自己（5） | 自己（3）+ PTOAS（3） | 闭源 |

- **Triton**：`assign_latencies`、`schedule_loops`、`pipeline`、`prefetch`、`coalesce_async_copy`、`tma_lowering`、`interleave_tmem`
- **TileLang-Ascend**：`inject_pipeline`、`pipeline_planning`、`cross_core_pipeline`、`inject_ptx_async_copy`、`multi_version_buffer_rewriter`
- **PyPTO**：`lower_pipeline_loops`、`skew_cross_core_pipeline`、`inject_gm_pipe_buffer`；PTOAS 侧 `OptMemPlanForPipeline`、`PTOAssignDefaultFrontendPipeIdPass`、`PTOInferValidatePipeInitPass`

#### III-c 并行单元

| 职责 | Triton | TileLang-Ascend | PyPTO（+ PTOAS） | cuTile |
| ---- | ------ | --------------- | ---------------- | ------ |
| Tile 切分 / 矩阵加速 | 自己（5） | 自己（1） | 自己（1）+ PTOAS | 闭源 |
| 向量宽度选择 | 无独立 pass | **自己（5，四家最重）** | 自己（2） | 闭源 |
| warp / 核特化 | 自己（5） | 自己（3） | 自己（3）+ PTOAS（3） | 闭源 |
| 跨核资源绑定 | 无（SM 间经 global） | 自己（2） | 自己（1） | 闭源 |

- **Tile 切分 / 矩阵加速**
  - Triton：`accelerate_matmul`、`check_matmul_two_cta`、`optimize_dot_operands`、`lower_mma`、`f32_dot_tc`、`promote_lhs_to_tmem`
  - TileLang-Ascend：`cluster_planning`
  - PyPTO：`auto_tile_matmul_l0`；PTOAS 侧 `PTOMaterializeTileOpSections`、`TileFusion/`
- **向量宽度选择**
  - TileLang-Ascend：`loop_vectorize`、`loop_vectorize_dynamic`、`vectorize_loop`、`ascend_lower_parallel_to_vector`、`ascend_vector_instruction_selection`
  - PyPTO：`lower_auto_vector_split`、`split_vector_kernel`
  - Triton：无独立 pass，由 layout 与 `coalesce` 承担
- **warp / 核特化**
  - Triton：`hopper_warpspec`、`warp_specialize`、`optimize_partition_warps`、`allocate_warp_groups`、`warp_specialize_to_llvm`
  - TileLang-Ascend：`warp_specialized_rewriter`（CUDA 侧）、`ascend_combinecv`、`annotate_device_regions`
  - PyPTO：`expand_mixed_kernel`（拆 AIC / AIV / Group）、`split_vector_kernel`、`stamp_tfree_split`；PTOAS 侧 `PTOOutlineSIMTSections`、`PTOMaterializeSIMTPersistentFragment`、`PTOPromotePersistentFragmentLoopsPass`
- **跨核资源绑定**
  - TileLang-Ascend：`cross_core_pipeline`、`ascend_pto_save_buffer_shape`
  - PyPTO：`skew_cross_core_pipeline`

#### III-d 融合与指令调度

| 职责 | Triton | TileLang-Ascend | PyPTO（+ PTOAS） | cuTile |
| ---- | ------ | --------------- | ---------------- | ------ |
| 算子 / 循环融合 | 无独立子系统（靠 `combine` 等局部改写） | 自己（`TileFusion/` 目录） | **PTOAS 侧完整子系统（10）** | `FuseFMA`（仅 FMA 级） |
| 指令调度 | 自己（2） | 无显式调度器 | **PTOAS 侧（2）** | 闭源 |

- **算子 / 循环融合**
  - PTOAS：`PreFusionAnalysis`、`PrintPreFusionAnalysis`、`FusionPlan`、`PTOFusionRegionGen`、`PTOLowLevelLoopFusion`、`PTOFusionPredicateElision`、`PTOFusionLoadStoreElision`、`PTOVexpdifFusion`、`PTOUnrollAfterLoopFusion`、`PTOFlattenFusionRegion`，另有 `VPTOStatefulStreamFusion`、`VPTOCombineReductions`
  - TileLang-Ascend：`TileFusion/` 目录（源码在 PTOAS 之外独立实现）
  - Triton：无独立融合子系统；`combine`、`combine_tensor_select_and_if`、`fuse_nested_loops` 属局部改写
  - cuTile：`FuseFMA` 仅做乘加融合
- **指令调度**
  - Triton：`schedule_loops`、`reorder_instructions`
  - PTOAS：`OpScheduling`、`VPTOScheduler`
  - PyPTO 自身：无（调度权在 PTOAS）

> 【推断】这一子类是初版遗漏的。它的分布很特别：**融合与指令调度在 PyPTO 生态里完全由 PTOAS 承担，pypto 仓内一个都没有**——与第 IV 类（同步插入）同样外包。两者相加意味着"排布指令序 + 保证执行正确"这两件最接近硬件的事都不在 pypto 仓里。

### 3.4 第 IV 类 · 正确性保障

| 职责 | Triton | TileLang-Ascend | PyPTO（+ PTOAS） | cuTile |
| ---- | ------ | --------------- | ---------------- | ------ |
| **同步插入** | 自己（8） | 自己（8） | **整体外包给 PTOAS** | 闭源 |
| 掩码 / 尾块与访存合法化 | 无（mask 在语言层） | 自己（3） | 自己（1） | 闭源 |
| 语义别名 / 原地约束 | 无 | 无 | 自己（1） | 闭源 |
| 跨核配对与同步 | 无 | 并入 sync_insert | 自己（1） | 闭源 |

- **同步插入**
  - Triton：`fence_insertion`、`proxy_fence_insertion`、`membar`、`tmem_barrier_insertion`、`optimize_mbarrier_arrivals`、`cluster_barrier_mbar_allocator`、`initialize_ws_cluster_barriers`、`hoist_mbarrier_lifecycle`
  - TileLang-Ascend：`ascend_sync_insert`、`ascend_sync_insert_vs`、`thread_storage_sync`、`thread_partial_sync`、`storage_access`、`eliminate_storage_sync_for_mbarrier`、`inject_fence_proxy`、`wgmma_sync_rewriter`
  - PyPTO：**自身不做**。由 PTOAS 的 `InsertSync/` 子系统承担（12 个源文件：`MemoryDependentAnalyzer`、`SyncEventIdAllocation`、`RemoveRedundantSync`、`RemoveSlotRedundantSync`、`SyncMacroModel`、`SyncCodegen` 等），另有 `LoweringSyncToPipe`、`PTOInjectBarrierAllSync`、`PTORemoveRedundantBarrier`
- **掩码 / 尾块与访存合法化**
  - TileLang-Ascend：`ascend_vector_mask`、`ascend_vector_mask_legalize`、`ascend_tail_mask_propagation`；另有 `legalize_safe_memory_access`、`legalize_vectorized_loop`
  - PyPTO：`canonicalize_io_order`（访存次序约束）
- **语义别名 / 原地约束**
  - PyPTO：`materialize_semantic_aliases`（循环 carry 与原地算子必须共享同一 MemRef，语义强制，**即便跳过 MemoryReuse 也必须运行**）
- **跨核配对与同步**
  - PyPTO：`stamp_tfree_split`（把跨核 `tpop` 的 split / pipe-id 复制到配对的 `tfree`）

> 【推断】这一类是四家分工差异最大的一行，也是最值得单独论述的一行：Triton 与 TileLang 自己做、cuTile 锁在闭源编译器内、**只有 PyPTO 整体外包**。好处是 pypto 的 pass 流水线无需理解硬件事件模型；代价是这段最不可替代的能力不在自己仓内，跨仓版本必须对齐。

### 3.5 第 V 类 · 组织与交付

| 职责 | Triton | TileLang-Ascend | PyPTO（+ PTOAS） | cuTile |
| ---- | ------ | --------------- | ---------------- | ------ |
| 函数边界 / 调用约定 | 无（单 kernel） | 自己（1） | 自己（6） | 无 |
| **任务图 / 编排** | 无 | 无 | **自己（5）** | 无 |
| **分布式 / 通信** | 无（交给 NCCL） | 无（shmem 为核间） | **自己（4）** | 无（交给 NCCL） |
| 调试 / 校验 pass | **sanitizer 4 + debug info 3** | 1 | **IRProperty 契约 + PTOAS 校验（5）** | 1 |

- **函数边界 / 调用约定**
  - TileLang-Ascend：`ascend_host`
  - PyPTO：`outline_hierarchy_scopes`、`outline_incore_scopes`、`outline_cluster_scopes`、`derive_call_directions`、`normalize_return_order`、`optimize_orch_tensors`；PTOAS 侧 `PTOWrapFunctionsInSectionsPass`
- **任务图 / 编排**
  - PyPTO：`auto_derive_task_dependencies`、`expand_manual_phase_fence`、`materialize_runtime_scopes`、`classify_iter_arg_carry`、`fold_no_op_reshape`（配合 IO 规范化）
- **分布式 / 通信**
  - PyPTO：`synthesize_allreduce_signals`、`materialize_comm_domain_scopes`、`lower_host_tensor_collectives`、`materialize_dist_tensor_ctx`
- **调试 / 校验**
  - Triton：`fp_sanitizer`、`global_sanitizer`、`concurrency_sanitizer`、`prepare_consan_captures`、`di_scope`、`di_local_variable`、`version_info`
  - TileLang-Ascend：`legalize_safe_memory_access`（兼具 IV 类属性）
  - PyPTO：IRProperty verifier（25 个编译期不变量）+ 逐 pass dump + roundtrip 校验；PTOAS 侧 `PTOValidateVMIIR`、`PTOValidateVPTOIR`、`PTOVerifyTFreePass`、`PTOValidatePhysicalSectionBoundaries`、`PTOPrintAddressAnalysis`
  - cuTile：`SynthesizeDebugInfoScopes`

---

## 4. 完整 pass 清单

### 4.1 Triton · NVIDIA 后端

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

### 4.2 TileLang-Ascend（`ascendc_pto`）

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

### 4.3 PyPTO（`Default` 策略，按执行顺序）

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

**Tile / PTO 段（32）**

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

### 4.4 PTOAS（PyPTO 的汇编层）

`include/PTO/Transforms/Passes.td` 注册的 **83 个 pass**，按 TableGen 中的声明顺序（大致即流水线顺序）：

**同步与基础降级**

```
PTOInsertSync · PTOInjectBarrierAllSync · PTOBufidSync · PTOLowerGenericOps
InferPTOLayout · PTOA5NormalizeTMov · PTORemoveIdentityTMov · InferPTOMemScope
PTORematerializeFixpipeVectorQuant · PTOMaterializeImplicitTmp · PlanMemory
PTOLoweringSyncToPipe
```

**融合与调度**

```
PreFusionAnalysis · PrintPreFusionAnalysis · FusionPlan · OpScheduling
PTOMarkLastUse · PTOFusionRegionGen
```

**Section 与流水 id**

```
PTOWrapFunctionsInSections · PTONormalizeUncoveredTileSections
PTOValidatePhysicalSectionBoundaries · PTOMaterializeTileOpSections
VPTOSplitCVModule · VPTONormalizeContainer · PTOLowerFrontendPipeOps
PTOAssignDefaultFrontendPipeId · PTOInferValidatePipeInit
PTOResolveReservedBuffers
```

**Tile op 展开与规范化**

```
InsertTemplateAttributes · ExpandTileOp · FoldTileBufIntrinsics
PTOCanonicalizeIR · PTOArithRangeOptimize · LowerPTOToUBufOps
PTOInlineLibCall · PTOInlineBackendHelpers · PTOVerifyTFree
PTOResolveBufferSelect
```

**循环与 SIMT 片段**

```
PTOUnrollLoops · PTOUnrollSIMTFor · PTOConvertSCFToCFWithLoopHints
PTOPromotePersistentFragmentLoops · PTONarrowVPTOLoopCounters
PTOAnalyzeSIMTPersistentFragment · PTOMaterializeSIMTPersistentFragment
PTOOutlineSIMTSections · PTOInferVPTOVecScope
```

**校验**

```
PTOValidateVPTOIR · PTOValidateVMIIR · PTOValidateVMILayoutIR
PTOValidateVPTOEmissionIR
```

**VMI 层（布局赋值与下降，12）**

```
VMIPreAssignmentCombine · VMILayoutAssignment
VMILayoutRematerializeWeakProducers · VMIMaskGranularityAssignment
VMILayoutFold · VMILayoutRematerialize · VMILayoutSinkMaterialization
VMILegalizeArithSelect · VMINormalizeSignlessIntToUnsigned
VMILowerUnifiedToLegacy · VMIExpandImplicitEnsureLayouts · VMIToVPTO
```

**VPTO 层（调度与指令级优化）**

```
VPTOStatefulStreamFusion · PTOExpandSoftLib · VPTOScheduler
VPTOExpandWrapperOps · VPTOSoftPostUpdate · VPTOGuardedLICM
PTOPrintAddressAnalysis · PTOVPTOPtrBoundary · VPTOPtrNormalize
VPTOPtrCastCleanup · VPTOOptimizeVcvt · VPTOOptimizeCtrlState
VPTOCombineReductions · VPTOMaskSimplify
```

**低层融合**

```
PTOLowLevelLoopFusion · PTOFusionPredicateElision · PTOFusionLoadStoreElision
PTOVexpdifFusion · PTOUnrollAfterLoopFusion · PTOFlattenFusionRegion
```

**对应源码目录**：`lib/PTO/Transforms/` 下 `Passes/`（31 个源文件）、`Lowering/`（11）、`InsertSync/`（12）、`TileFusion/`、`VMI/`、`VMIToVPTO/`、`VPTO/`、`VPTOScheduler/`、`BufidSync/`、`PTOToEmitC/`。源文件数与 pass 数不是一一对应——`PTOInsertSync` 一个 pass 由 `InsertSync/` 的 12 个文件实现，反之一个文件也可能注册多个 pass。

### 4.5 cuTile · CUDA Tile IR

**公开 pass（全部 4 个）**

| Pass | 作用（官方 summary 转述） | 归类 |
| ---- | ------------------------ | ---- |
| `CudaTileLICM` | 把循环不变量提出 `cuda_tile.for` / `cuda_tile.loop` | I |
| `LoopSplit` | 当 if 条件把归纳变量与循环不变量比较时拆分循环（带 `split-threshold` 选项） | I |
| `FuseFMA` | 把乘加 / 乘减融合为 FMA（非数值保持） | I |
| `SynthesizeDebugInfoScopes` | 为 module 合成 debug info scope 信息 | V |

**闭源部分**：`tileiras`——字节码到 cubin，官方类比 `ptxas`；布局、同步、内存规划、Tensor Core 映射均在其中，pass 列表未公开。

**配套工具**：`cuda-tile-opt`（MLIR opt 驱动）、`cuda-tile-optimize`、`cuda-tile-translate`（MLIR → 字节码）、`cuda-tile-tblgen`。

---

## 5. 结论

**结论一｜cuTile 不是 pass 少，是不公开——而且公开的那几个属于最不值钱的一类。**
【事实·外】`NVIDIA/cuda-tile` 的 `Passes.td` 只有四个 pass，按第 2 节分类，三个落在第 I 类（规范化），一个落在第 V 类（debug info）。第 II、III、IV 类**一个都没有**。
【推断】NVIDIA 开源的是**方言与契约**（dialect、字节码格式、Python 绑定、一致性测试），不是优化实现。对外宣称的"开放"应理解为"可以对接"，而非"可以审查优化决策"。

**结论二｜Triton 把新硬件特性直接 pass 化，重心压在第 III 类。**
【事实·外】ttgir + llir 中可数出 `tmem_*` 六个、`mbarrier` 相关三个、`warp_specialize` 四个，外加 `tma_lowering`、`cluster_barrier_mbar_allocator`。
【推断】这是其 pass 总数最高的直接原因，代价是后端与具体架构强耦合——每代新硬件都要新增一批 III 类 pass。

**结论三｜PyPTO 把三件最贴硬件的事整体外包给了 PTOAS：同步插入、算子融合、指令调度。**
【事实】第 IV 类同步插入：Triton 与 TileLang 各用 8 个 pass 自己做，cuTile 在闭源编译器内做，**PyPTO 自身一个都没有**，由 PTOAS 的 `PTOInsertSync`（`InsertSync/` 12 个源文件实现）承担。第 III-d 类同理：融合 10 个、调度 2 个，全部在 PTOAS，pypto 仓内为零。
【推断】这一类是**唯一不能被 autotune 替代**的能力——搜索能找到更快的配置，但不能帮你插对同步。按口径 B，PyPTO 生态 126 个 pass 里 83 个在 PTOAS，**三分之二的工具箱不在 pypto 仓内**，跨仓版本必须对齐（当前由 `toolchain/versions.env` 钉 ptoas 版本、`runtime/pto_isa.pin` 钉 ISA 版本）。这既是分工优势，也是单仓能力评估时最容易被高估或低估的地方。

**结论四｜第 V 类里有两个职责只有 PyPTO 有。**
【事实·内】任务图编排 5 个 + 函数边界 6 个、分布式通信 4 个，其余三家对应行全空。
【推断】这是 pass 层面唯一无人竞争的地带。差异化叙事应钉在"编排与通信被编译进产物"这一点上，而非"有 tile 虚拟 ISA"或"抽象层级高"——后两者已被 cuTile / CUDA Tile IR 独立验证，不再独占。

**结论五｜"验证"有两条互不替代的路线。**
【事实】Triton 有运行时 sanitizer（`fp_sanitizer`、`global_sanitizer`、`concurrency_sanitizer`）；PyPTO 有编译期契约（IRProperty + 25 个不变量 + printer↔parser roundtrip）。
【推断】前者抓真实执行中的数值与并发错误，后者抓编译器自身的 bug，不能互相替代。**PyPTO 缺前者，Triton 缺后者**；若要在"可信证据链"上立论，补运行时 sanitizer 是最直接的缺口。

**结论六｜四家最重的那一类彼此不重叠，所以"对标"必须先说清对标哪一类。**
【推断】Triton 与 TileLang 的重心在第 III 类（把硬件资源榨干），PyPTO 的重心在第 V 类（把算子之间与卡之间编译进来），cuTile 公开的只有第 I 类。因此单算子性能对标与整网能力对标会得出相反结论——两种结论都成立，因为量的根本不是同一类能力。

---

## 6. 来源与核实方式

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

## 7. 未覆盖与待核实

1. **Triton-Ascend 的 pass 未并入**。其链路为 TTIR → `TritonToLinalg` / `TritonToHIVM` → `triton-mlir-opt` → `bishengir-opt` → `hivmc` / `bisheng`，`third_party/ascend/AscendNPU-IR` 作为子模块。需要时应单独成列，不要并入"Triton"列。
2. **TileLang-Ascend 的 `npuir` 分支未计入**。该分支 `3rdparty/` 只有 `AscendNPU-IR`（无 `pto-isa`），codegen 为 `codegen_npuir_api.cc` / `codegen_npuir_dev.cc`，pass 分布与 `ascendc_pto` 分支不同。
3. **cuTile 闭源部分的 pass 规模未知**。只能确认职责存在，无法计数，故第 2.3 节四家重心分布中 cuTile 一行只能描述公开部分。
4. **PyPTO 与 PTOAS 的 pass 数存在重叠语义**：内存规划可按 `memory_planner` 在两边切换，计数不应简单相加（见 3.3 节注）。
5. **TileLang-Ascend 的 51 个文件中部分为 analysis 而非 transform**，未逐个区分。
6. **第 2 节的五类归属含判断成分**。少数 pass 跨类（如 `legalize_safe_memory_access` 兼具 IV / V 属性、`multi_version_buffer_rewriter` 兼具 III-a / III-b 属性），归类时取其主要职责，重新统计时可能出现 ±1 的差异。
7. **TileLang-Ascend 与 PTOAS 的口径仍不对齐**。前者为源文件数（51），后者为 TableGen 注册数（83）；TileLang 用 C++ 直接注册、无单一 `.td` 清单，因此无法给出与 PTOAS 同口径的数字。若要严格对齐，需逐个源文件统计其注册的 pass 数。
8. **PTOAS 的 83 个 pass 中哪些进入默认流水线未核实**。`--pto-level=level1/2/3` 组装的 pass 序列不同（level3 跳过 `PlanMemory`），因此其口径 A 数字本文留空。
9. **Triton 的 56 只统计了 NVIDIA 相关的五个 `.td`**，未含 AMD、CPU 等后端，也未含 `Conversion/` 下其余目录。

---

## 8. 相关材料

- 本目录：`PyPTO3.0_对位NVIDIA_TileLang_Triton_深度对比.md`（战略层对比）、`PyPTO3_竞品资料收集审核稿.md`（竞品分层与来源清单）
- PyPTO 自身流水线：`repo/pto/docs/zh-cn/dev/passes/00-pass_manager.md` 及同目录逐 pass 文档（编号与执行顺序一致）
