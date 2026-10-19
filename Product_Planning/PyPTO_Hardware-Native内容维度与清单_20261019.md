# PyPTO 的 Hardware-Native 内容：维度体系与内容清单

> 整理日期：2026-10-19
> 回答的问题：站在 hw-native-sys「Hardware-Native」的定位上，基于昇腾硬件，**可以从哪些维度**梳理内容、**哪些内容应该放进 PyPTO**，才能让开发者真正做到硬件原生的模型 / 算子开发。
> 证据范围：`repo/pypto`（较新镜像，含 `python/pypto/pypto_core/backend.pyi`、`docs/zh` 全量）、`repo/pto`、`repo/.github/profile/README.md`、`Insight/` 与 `Product_Planning/` 既有结论、`github_issues/` 归档。
> 标注约定（沿用本仓库写法）：【事实】可在仓库文件中直接核对；【统计】由本文方法得出，方法已给出；【推断】基于前两者的判断；【建议】产品动作。

---

## 摘要

**一句话结论**：Hardware-Native 不是「把昇腾细节写进 DSL」，而是让每一条会裁决正确性或性能的昇腾事实，在 PyPTO 里**至少有一个明确的承载形态**。因此内容梳理的主维度不是「有哪些硬件知识」，而是 **「硬件事实域 × 承载形态」** 的交叉——前者决定内容是什么，后者决定它该以语言原语、可查 API、诊断码、归因记录、观测产物还是范式库的形式存在。

本文给出：

- **1 条定义与 3 条判据**（第 1 节）：什么算 PyPTO 里的 Hardware-Native 内容。
- **5 个维度**（第 2 节）：主维度 2 个（硬件事实域、承载形态），调节量 3 个（作用域层级、抽象剂量、代际变量）。
- **9 × 6 内容矩阵与现状盘点**（第 3–5 节）：逐格标注已有 / 部分 / 缺失。
- **7 项优先内容清单**（第 8 节）与**边界**（第 9 节）。

**三条最值得先看的判断**：

1. **PyPTO 已经把硬件事实「写进了编译器」，但没有把它「交给开发者」。** `BackendHandler` 已在 Python 侧暴露 26 个方法——L0A/L0B/L0C/Bias/Mat 容量、GM 访问粒度、L2 cache line、推荐最内维、分形对齐、Die/Cluster/Core 拓扑，以及 7 条代际行为开关；但 `pypto/__init__.py` 不导出 `backend`，全部 `docs/` 中检索不到任何一个容量 getter 的名字。**这是成本最低、收益最直接的一格。**
2. **九个硬件事实域里，只有「访存粒度」一域有硬件性能提示。** 全仓库唯一的 PerfHint 码仍是 `PH001`。其余八域做错了只会「慢」，而编译器一声不吭。
3. **现有 Hardware-Native 内容在 Tile / InCore / Kernel 三层最厚，在「整网」一层最薄。** 而模型开发者恰好站在整网层——这是「模型开发」相对「算子开发」的结构性缺口，不是文档数量问题。

---

## 1. 先定义：什么算 PyPTO 里的 Hardware-Native 内容

### 1.1 组织自己的定义

**【事实】** `repo/.github/profile/README.md` 写明组织哲学：system software——from compilers to kernels——must be designed with a deep awareness of the underlying hardware architecture。

**【事实】** `pto-isa` 的设计目标被归纳为「不是隐藏底层能力，而是抬高抽象层级同时保留调优空间」（转引自 [Insight/hw-native-sys-仓库功能洞察报告.md](../Insight/hw-native-sys-仓库功能洞察报告.md) 对 `pto-isa` README 的摘录）。

**【推断】** 这两句合起来给出的不是「暴露硬件」，而是一组更强的要求：**抽象可以抬高，但硬件事实不许消失**。消失的定义是：开发者既无法表达它、也无法查询它、也无法知道编译器替他做了什么。

### 1.2 三条判据

据此，一条内容算不算 PyPTO 的 Hardware-Native 内容，用三问判定：

| 判据 | 问题 | 不满足时的后果 |
| ---- | ---- | -------------- |
| **J1 裁决性** | 这条昇腾事实会裁决正确性或性能吗？ | 不裁决的事实属于背景知识，不必进 PyPTO |
| **J2 不可自得性** | 开发者能从 DSL 源码或通用知识推出它吗？ | 能推出的不必额外承载 |
| **J3 有承载形态** | 它在 PyPTO 里以什么形式存在？ | **没有形态 = 抽象泄漏** |

**【推断】** [PyPTO3.0_昇腾亲和产品方向分析.md](PyPTO3.0_昇腾亲和产品方向分析.md) 已论证 J1 与 J2（语言层刻意去昇腾化、但裁决权 100% 在微架构），并给出 8 个泄漏点。**本文补的是 J3**：同一条事实放在语言层、规格 API、诊断、归因、观测还是范式库里，是完全不同的产品决策，而既有分析没有区分这一层。

---

## 2. 五个维度

```text
                  维度二：承载形态（怎么放进去）
                  E 可表达 / Q 可查询 / C 可校验 / A 可归因 / O 可观测 / R 可复用
                        ┌───┬───┬───┬───┬───┬───┐
  维度一    1 存储层级  │   │   │   │   │   │   │
  硬件      2 计算单元  │   │   │   │   │   │   │
  事实域    3 布局分形  │   │   │   │   │   │   │   ← 主矩阵：决定内容清单
  （是什么）4 访存粒度  │   │   │   │   │   │   │
            5 跨核同步  │   │   │   │   │   │   │
            6 时间流水  │   │   │   │   │   │   │
            7 数值精度  │   │   │   │   │   │   │
            8 任务运行时│   │   │   │   │   │   │
            9 多卡通信  │   │   │   │   │   │   │
                        └───┴───┴───┴───┴───┴───┘
                               ↑        ↑        ↑
       维度三 作用域层级 ───────┘        │        │   指令→Tile→InCore→Kernel→整网→多卡
       维度四 抽象剂量 ──────────────────┘        │   Tensor / Tile / Block 三档深度
       维度五 代际变量 ───────────────────────────┘   910B(A2/A3) vs 950(A5) 横切每一格
```

| 维度 | 回答什么 | 产出什么 |
| ---- | -------- | -------- |
| **一 · 硬件事实域** | 内容是什么 | 9 个域的事实清单 |
| **二 · 承载形态** | 以什么形式放进 PyPTO | 每条内容的落点 |
| **三 · 作用域层级** | 这条内容服务算子开发还是模型开发 | 算子侧 / 模型侧分工 |
| **四 · 抽象剂量** | 给谁看、默认多深 | 呈现深度分档 |
| **五 · 代际变量** | 是否需要按芯片代际版本化 | 哪些内容必须带 target |

维度一 × 维度二是**主矩阵**，决定「梳理哪些内容」；维度三、四、五是**调节量**，决定每格内容的归属、深度与版本化方式。维度四沿用既有结论，本文不重复展开。

---

## 3. 维度一：九个昇腾硬件事实域

每个域按「硬件事实 → PyPTO 中的表达 → 做错的后果」三要素组织。后果一栏区分**错**（含 hang、静默数据损坏）与**慢**——这决定它该进 C（校验）还是 A/O（归因与观测）。

| # | 域 | 关键昇腾事实（已核实数值） | PyPTO 中的表达 | 做错 |
| - | -- | -------------------------- | -------------- | ---- |
| 1 | **存储层级与容量** | L0A/L0B 各 64 KiB；L0C 910B 128 KiB / 950 256 KiB；Mat(L1) 512 KiB；另有 Bias、ScalarLocal | `MemorySpace`（`DDR/Vec/Mat/Left/Right/Acc/Bias/ScalarLocal`）、`target_memory`、`pl.create_l1` | 错（放不下）→ 慢（复用不当） |
| 2 | **计算单元与分工** | Cluster = 1 Cube + 2 Vector；Die / Cluster / Core 三级拓扑 | `aic`/`aiv` 作用域、`ExpandMixedKernel`、`SplitVectorKernel` | 错（切分不合法）+ 慢 |
| 3 | **布局与分形** | Cube 要 NZ/ZN fractal、16 对齐；`fractalABSize` 在 PTOAS 侧硬编码 512（PTOAS#57） | `TileLayout`（`row_major/col_major/none_box`）、`blayout/slayout/fractal/stride/valid_shape/pad`、`ResolveBackendOpLayouts` | 错（校验失败）→ 慢（隐式布局转换） |
| 4 | **访存粒度与带宽** | GM 粒度 910B 512 B / 950 128 B；L2 cache line 均 512 B | `PH001`、`tile.load/store` 最内维 | **慢**（唯一有提示的一域） |
| 5 | **跨核通道与同步** | 每对等核心每方向 8 个硬件 flag，每 Cluster 共 32；ring 单向 8 槽、双向每向 4 槽；A2/A3 ring 在 GM、A5 在消费者片上 SRAM | `tpush/tpop`、`pl.cross_core_slot`、`system.syncall`（hard/soft）、`InjectGMPipeBuffer` | **错或 hang**（#828：identity tmov 在 A5 挂死） |
| 6 | **时间与流水** | `PipeType` = MTE1/MTE2/MTE3/M/V/S；搬运-计算 overlap 决定峰值 | `pl.pipeline(stage=N)`、`LowerPipelineLoops`、`SkewCrossCorePipeline`、L0 指令级 trace | 慢（深度不足）→ 错（跨 stage 误复用） |
| 7 | **数值与精度** | `cast` 饱和模式在 A2/A3 为硬件原生；950 的 `INT32→FP16` 需经 FP32 多跳；MX/FP4 仅 950 | `pl.cast(saturation_mode=)`、`LegalizeTileCast`、`InsertMxScaleAddr`、`precision/` 两篇 | 错（精度不对齐） |
| 8 | **任务与运行时** | 三程序模型（Host/AICPU/AICore）、MPMD；四层 scope-depth ring；`ring_task_window` 需 2 的幂且 ≥4，`ring_heap` ≥1024 B | `pl.submit` / `Submit` IR 节点、`manual_scope`、`RunConfig`、persistent L3、JIT cache | 慢（派发开销、ring 饿死） |
| 9 | **多卡与通信** | window buffer、comm domain、集合通信多形态 | `pld.*`、`DistributedTensorType`、`WindowBuffer`、22 篇 distributed 文档 | 错（同步漏插）+ 慢 |

**【统计】** 既有分析从 632 条 Issue 标题中统计出 178 条（28%）落在微架构映射上，分布为：AIC/AIV 切分 45、matmul 与 L0 分块 42、layout 33、内存复用与地址分配 31、跨核 ring 23、valid_shape 与边界 19、同步与 flag 15、内存空间语义 10（方法见 [PyPTO3.0_昇腾亲和产品方向分析.md](PyPTO3.0_昇腾亲和产品方向分析.md) 第 2 节）。

**【推断】** 把这 8 个泄漏点映射到上表：它们**全部落在域 1、2、3、5**，即「存储 / 分工 / 布局 / 同步」四域。域 4 有提示但只有一条；域 6、7、8、9 在 Issue 标题里几乎不出现——**这不代表它们没问题，而更可能代表开发者还没走到那一步，或者走到了也说不清**。模型级（域 8、9）的沉默尤其值得警惕。

---

## 4. 维度二：六种承载形态，以及分配判据

### 4.1 六种形态

| 代号 | 形态 | 在 PyPTO 里长什么样 | 现有例子 |
| ---- | ---- | ------------------- | -------- |
| **E** | **可表达** | 类型、枚举、原语、directive——开发者能写出这个硬件决策 | `pl.Mem.*`、`TileLayout`、`pl.pipeline(stage=)`、`pl.cross_core_slot` |
| **Q** | **可查询** | 平台规格作为**数据**而非文档数字，开发者能在代码里算 | `BackendHandler` 的 26 个方法（**仅 C++ / stub，未对用户开放**） |
| **C** | **可校验** | 诊断码、`CHECK`、写码期体检——用昇腾语言说出「这样不行」 | `PH001`、`CHECK_SPAN`、`report/perf_hints.log` |
| **A** | **可归因** | 编译器硬件决策的记录 + 回到用户源码行 | 逐 pass 文档（60 篇）、pass dump、IR lowering trace、memory map |
| **O** | **可观测** | 按昇腾结构组织的执行证据 | L2 泳道图、`dfx_outputs`、L0 指令级 trace、runtime replay |
| **R** | **可复用** | 把「唯一正确的硬件用法」固化成范式 | `examples/`（beginner/intermediate/advanced/models）、6 篇 tutorials、`pypto-lib` kernel |

### 4.2 分配判据：一条事实该进哪一格

**【推断】** 关键的一问是：**这条昇腾事实，开发者是「要决定它」，还是「要知道编译器怎么替他决定的」？**

```text
这条昇腾事实……
├─ 开发者必须亲自决定 ───────────────→ E（给原语）+ C（给校验）
├─ 开发者要据它算出自己的决定 ───────→ Q（给可查规格）
├─ 编译器替他决定了 ─────────────────→ A（给归因）
├─ 只有真跑一次才知道 ───────────────→ O（给观测）
└─ 有唯一正确写法，不该让人重新发明 ─→ R（给范式）
```

**三条推论**：

1. **E 和 A 是同一件事的两面，互斥。** 一条事实要么开发者决定（E），要么编译器决定（A）；最糟的状态是编译器决定了却不说——既不是 E 也不是 A。域 1 的 `MemoryReuse` 正处在这个状态，它的典型失败是**静默数据损坏**（#585 混叠 `gate_acc`/`up_acc`、#768 复用存活 yield 输出）。
2. **Q 是 C 和 R 的前置。** 校验规则的阈值、范式里的 tile 尺寸，都必须引自同一份规格；否则工具与编译器双源不一致，用户照着改反而编译不过。
3. **C 覆盖「可静态判定」的部分，O 覆盖剩下的。** 分界线就是「能不能在不跑的情况下算出来」——L0 容量能（C），ring 槽位阻塞不能（O）。

---

## 5. 主矩阵：九域 × 六形态的现状盘点

标注：**✓** 已有且面向用户；**△** 存在但只在内部 / 只覆盖一部分；**✗** 缺失。依据为 `repo/pypto` 实际文件检索（方法见第 11 节）。

| 域 | E 可表达 | Q 可查询 | C 可校验 | A 可归因 | O 可观测 | R 可复用 |
| -- | :------: | :------: | :------: | :------: | :------: | :------: |
| 1 存储层级与容量 | ✓ | **△** | △ | ✓ | ✓ | ✓ |
| 2 计算单元与分工 | ✓ | **△** | ✗ | ✓ | ✓ | ✓ |
| 3 布局与分形 | ✓ | **△** | △ | ✓ | △ | △ |
| 4 访存粒度 | — | **△** | ✓ | ✓ | ✓ | △ |
| 5 跨核同步 | ✓ | **△** | ✗ | ✓ | ✓ | ✓ |
| 6 时间与流水 | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ |
| 7 数值与精度 | ✓ | △ | △ | ✓ | △ | ✓ |
| 8 任务与运行时 | ✓ | △ | ✗ | ✓ | ✓ | △ |
| 9 多卡与通信 | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ |

### 5.1 读这张表的三个结论

#### 结论一：A 列整列是 PyPTO 最强的地方，E 列次之

**【事实】** `docs/zh/dev/passes/` 下 60 篇 pass 文档（含 00–54 逐 pass 页），`dev/ir/` 12 篇，加上逐 pass IR 快照、IRProperty 校验、printer↔parser roundtrip。**【推断】** 按 [四框架Pass对比](../Competitive_Analysis/四框架Pass对比_PyPTO_TileLang_Triton_cuTile_20261019.md) 的口径，这是四家里中间态可见性最完整的一套。A 列不是缺口，是资产。

**【事实】** `docs/zh/user/language/03-memory.md` 已给出完整双语对照：`Mat(L1)`、`Vec(UB)`、`Left(L0A)`、`Right(L0B)`、`Acc(L0C)`，并配有内存层级示意图；`pl.create_l1(...)` 这一 API 名甚至直接用了硬件名。**【推断】** 既有分析列为首要抓手之一的「双语对照」，在 `repo/pypto` 这一版上**已经落地**，应从缺口清单中移除。

#### 结论二：Q 列整列是最大、也是最便宜的缺口

**【事实】** `python/pypto/pypto_core/backend.pyi` 的 `BackendHandler` 已暴露 26 个方法，包括：

- 容量：`get_l0a_capacity_bytes` / `get_l0b_capacity_bytes` / `get_l0c_capacity_bytes` / `get_bias_capacity_bytes` / `get_mat_capacity_bytes`
- 访存：`get_gm_access_granularity_bytes`、`get_l2_cache_line_bytes`、`get_recommended_innermost_dim_bytes`
- 分形：`get_l0_fractal_alignment`、`get_l0c_m_alignment(dtype)`、`get_min_l0_tile_dim`
- 拓扑：`Mem` / `Core` / `Cluster` / `Die` 四级类，带 `total_core_count()` 等聚合
- 代际行为开关 7 条：`requires_gm_pipe_buffer`、`requires_split_load_tpop_workaround`、`requires_level3_tmp_scratch`、`requires_vto_c_fractal_adapt`、`requires_runtime_subblock_bridge`、`requires_no_split_dual_aiv_dispatch`、`splits_cube_to_vector_transport_in_hardware`
- 其他：`get_pto_target_arch`、`get_default_sim_platform`、`honors_high_precision_algorithm`、`supports_mat_to_bias_move(src, dst)`

**【事实】** 同时：`python/pypto/__init__.py` 中检索不到 `backend` 的导出；在 `repo/pypto/docs/` 全量中检索 `get_l0a_capacity_bytes` / `get_gm_access_granularity` **零命中**；`docs/zh/dev/backend/00-backend_handler.md` 把它完全定位为「pass 与 codegen 的后端分发机制」，不含任何用户视角。

**【推断】** 规格**已经是数据了**，只是这份数据只流向编译器，不流向开发者。开发者今天要知道 L0C 容量，只能去读 `Insight/` 的整理表或 Ascend C 文档——那是第二份、会过期的规格。既有分析担心的「双源不一致」风险，其实可以**通过开放已有单源来消除**，而不是新建一份档案。

#### 结论三：C 列是「做错会错」的域里最空的一列

**【事实】** 在 `repo/pypto/src` 与 `include` 中检索 `PH0\d\d`，唯一命中仍是 `PH001`（`TileInnermostDimGranularity`）。**【事实】** hw-native-sys/pypto#1305 仍 open：`PH001` 的 span 指向流水线后的 IR 文本位置（`<string>:line:col`），而非用户 DSL 源位置。

**【推断】** 域 2、5、6、8、9 做错的后果是「错 / hang / 慢」，但编译期一句提示都没有。这不是「再加几条 hint」的问题——**C 列能做多深，取决于 Q 列是否开放（阈值从哪来）与 #1305 是否解决（结论落到哪一行）**。三者是一条链。

---

## 6. 维度三：作用域层级——这一维回答「模型 vs 算子」

**【推断】** 前两维足以梳理**算子**开发的内容，但「模型开发」不是「更多算子」，它换了作用域。按作用域展开，硬件事实出现在六个层级：

| 层级 | 这一层的硬件事实 | PyPTO 现有承载 | 主要服务谁 |
| ---- | ---------------- | -------------- | ---------- |
| **L1 指令** | `PipeType`、ISA 语义、flag / ring 原语 | `reference/pto-isa/` 3 篇、L0 指令级 trace | 系统开发者 |
| **L2 Tile** | memory space、layout、valid_shape、分形 | `user/language/` 7 篇（含 03-memory） | 性能专家 |
| **L3 InCore 函数** | L0 分块、double buffer、向量宽度、最内维 | `performance/04-incore`、`tutorials/02-matmul` | 性能专家 |
| **L4 Kernel / Task** | AIC/AIV 切分、ring 槽位、syncall、任务粒度 | `user/tasks/` 4 篇、`performance/01–03` | 性能专家 |
| **L5 整网（单卡）** | 算子边界与融合、KV cache、L3 常驻、JIT 缓存、整网片上预算 | `dev/06-persistent-l3`、`dev/10-jit-cache`（**均在 dev 侧**） | **算法 / 模型开发者** |
| **L6 多卡** | window buffer、comm domain、集合通信形态 | `user/distributed/` 22 篇 | 模型 / 系统开发者 |

**【统计】** 用户侧文档页数（`docs/zh/user/`，不计 index）：distributed 22、language 7、performance 7、tools 7、tutorials 6、tasks 4、precision 2、ops 2、execution 2、appendix 2。

**【推断】** 两处不对称值得注意：

1. **L5 在用户侧几乎是空的。** 与整网硬件预算直接相关的 persistent L3、JIT cache、artifact store 全部只在 `dev/` 下，没有面向模型开发者的入口。模型开发者问的问题是「我这一层 Transformer 在 910B 上，片上放得下几个 KV 分块 / 要切几次 / 哪个算子该融」——这个问题今天在 PyPTO 里没有任何一页正面回答。
2. **L6 的 22 篇与 L5 的 0 篇形成反差。** 多卡通信被当成一等公民写满了，单卡整网反而是断层。**这很可能是因为 L6 有明确的 API 面（`pld.*`），而 L5 的硬件事实分散在编译选项、缓存机制与算子边界里，没有一个 API 把它们收拢——L5 缺的首先是一个对象，其次才是文档。**

**【建议】** 把「整网片上预算」作为 L5 的承载对象来设计：输入是模型结构与 target，输出是逐层的 Vec/Mat/L0 占用、搬运量与核占用估算。它天然横跨 Q（规格）、C（放不下就报）、A（编译器的分块选择）三列，是把三列串起来的最短路径。既有真实数据可直接验证原型：`Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/dfx_outputs/`。

---

## 7. 维度五：代际变量——横切每一格，不是第十个域

**【事实】** `docs/zh/user/appendix/00-feature-matrix.md` 已有一张 910B(A2/A3) vs 950(A5) 对照表，含 6 行：平台字符串、mix 作用域、跨核环、GM pipe buffer 注入、多跳 `tile.cast`、MX matmul 系列。

**【事实】** 但代际差异实际横跨九域，且数值差异未进该表：

| 域 | 910B（A2/A3） | 950（A5） |
| -- | ------------- | --------- |
| 1 容量 | L0C 128 KiB | L0C 256 KiB |
| 4 访存 | GM 粒度 512 B | GM 粒度 128 B |
| 5 同步 | ring 在 GM，需 `__gm_pipe_buffer` | ring 在消费者片上 SRAM，不需要 |
| 7 精度 | `cast` 饱和为硬件原生 | `INT32→FP16` 需经 FP32 多跳；支持 MX |

**【事实】** `BackendHandler` 里的 7 条 `requires_*` / `splits_*` 布尔，每一条都对应一个用户可感知的行为差异（例如 `requires_split_load_tpop_workaround`、`requires_no_split_dual_aiv_dispatch`），**目前零用户文档**。

**【推断】** 代际不是第十个事实域，而是**每一格的一个属性**。处理上有一条硬约束：**代际差异必须随 target 版本化、由规格源头生成，绝不能写死在文案里**——否则每出一代芯片，所有页面的数字都要人工同步一遍，而漏掉的那一个不会报错。

---

## 8. 内容清单：七项，按优先级

排序依据：**是否解除其他项的阻塞 > 实现成本 > 覆盖的域数**。

| 顺位 | 内容 | 形态 | 覆盖域 | 依据 |
| :--: | ---- | ---- | ------ | ---- |
| 1 | **把平台档案开放成用户一等公民** | Q | 全部 9 | 规格已是数据（`BackendHandler` 26 方法），只差导出 + 一页用户文档；是 C 与 R 的前置 |
| 2 | **源码映射：IR 位置 ↔ 用户 DSL 行** | A/C 基础设施 | 全部 9 | #1305 仍 open；不做则所有校验结论停在「IR 某处」 |
| 3 | **按域补 PerfHint，先补「错」后补「慢」** | C | 1/2/3/5 优先 | 占 Issue 标题 28% 的四域今天零提示；域 5 的失败是 hang，域 1 是静默损坏 |
| 4 | **整网片上预算对象** | Q+C+A | 1/3/6/8 | L5 作用域在用户侧为空，模型开发者无入口（第 6 节） |
| 5 | **buffer 生命周期与别名可视化** | O | 1 | `MemoryReuse` 的失败是静默数据损坏且三次回归（#673→#1310→#1352），读 IR 文本发现不了 |
| 6 | **代际差异按域版本化** | 横切 | 全部 9 | feature-matrix 只有 6 行特性级；7 条 `requires_*` 零文档 |
| 7 | **硬件范式库：每域一个「唯一正确写法」** | R | 3/5/6/8 | `examples/` 已按难度分层，但不是按硬件决策点组织 |

**【建议】** 1 与 2 并行启动：第 1 项是纯暴露工作，不依赖任何 IR 改造；第 2 项涉及跨层 IR 元数据改造、周期长，应尽早开始并准备工具侧自建映射的兜底。3–7 在这两项之上展开。

**【建议】不建议先做的**：通用 autotune / 搜索类能力。按既有统计，当前痛点分布是正确性（layout 33 + 内存复用 31 + 同步 15 = 79 条）压倒性高于性能（1 条 hint）。更重要的是：**搜索能找到更快的配置（域 1/3/6 的事），但不能帮你插对同步（域 5）**——正确性保障是唯一不能被 autotune 替代的一类。

---

## 9. 边界：哪些内容不该放进 PyPTO

**【推断】** Hardware-Native 不等于什么都往 PyPTO 塞。三条边界：

| 内容 | 该放哪 | 理由 |
| ---- | ------ | ---- |
| 昇腾硬件名进入 DSL 语义（把 `Mat` 改回 `L1`） | **不做** | #268 已明确否决别名方案；双语对照属显示层，`03-memory.md` 的做法是正确形态 |
| 硬件映射与指令调度的实现（融合边界、同步插入、指令排布） | **PTOAS** | PTOAS 83 个注册 pass 才是真正的硬件映射层；PyPTO 重心在「组织与交付」，重复实现会造成两套决策 |
| 常驻的全量硬件面板 | **工具侧（Toolkit / Studio），且分档** | 对算法开发者，过度亲和本身是负担；PyPTO 内默认给最浅一档，由用户主动下钻 |

**【推断】** 一条简明分工：**PyPTO 负责让硬件事实「可表达、可查询、可归因」，PTOAS 负责让它「被正确实现」，Toolkit 负责让它「被看见和被解释」。** 三者共用同一份规格源头，这是不可让步的。

---

## 10. 风险

| 风险 | 说明 | 应对 |
| ---- | ---- | ---- |
| 规格双源 | 工具 / 文档自行维护第二份硬件数字 | 第 8 节第 1 项：开放已有单源，禁止新建档案 |
| Q 开放后被误用为稳定 API | 用户把 `requires_*` 写进生产代码，后续重构受阻 | 明确区分「可查询的规格值」与「内部行为开关」，后者只读不承诺 |
| 校验结论无落点 | #1305 不解决，PerfHint 增加反而增噪 | 第 3 项排在第 2 项之后；或先只补「错」类（有明确 op 位置），后补「慢」类 |
| 代际内容过期 | 数字写死在文案中，新芯片出时无声失效 | 第 7 节：随 target 版本化，由规格生成 |
| L5 做成文档而非对象 | 补几页「整网最佳实践」，无法随模型与 target 变化 | 第 4 项定义为可计算对象，用 `dfx_outputs` 真实数据验证 |
| 过度亲和压垮算法开发者 | 九域全量平铺，抵消分层抽象设计意图 | 维度四剂量原则：Tensor / Tile / Block 分档，默认最浅 |

---

## 11. 证据索引与复核方法

### 仓库文件（均位于 `repo/pypto`，为较新镜像）

- `python/pypto/pypto_core/backend.pyi`（约 100–130 行）：`BackendHandler` 26 个方法，含容量、访存、分形、拓扑、7 条代际开关。
- `python/pypto/__init__.py`：检索不到 `backend` 导出。
- `include/pypto/backend/common/backend_handler.h`：C++ 侧同一接口的虚函数定义。
- `docs/zh/dev/backend/00-backend_handler.md`：将其定位为 pass / codegen 分发机制，无用户视角。
- `docs/zh/user/language/03-memory.md`（15–22、73–76、106 行）：`Mat(L1)` / `Vec(UB)` / `Left(L0A)` / `Right(L0B)` / `Acc(L0C)` 对照表与层级图；`pl.create_l1`。
- `docs/zh/user/appendix/00-feature-matrix.md`：910B vs 950 六行特性差异；`PipeType` 取值；内存规划器三选项。
- `docs/zh/user/performance/index.md`：三列时间模型（host / 编排 / AICore）、`report/perf_hints.log`。
- `docs/zh/dev/passes/`：60 篇 pass 文档；`92-diagnostics.md` 为诊断体系。
- `docs/zh/reference/pto-isa/`：3 篇（cluster architecture、tpush_tpop、buffer management）。
- `repo/.github/profile/README.md`：组织 Hardware-Native 定义。

### 本仓库既有分析（本文在其基础上补 J3 / 承载形态维度）

- [Product_Planning/PyPTO3.0_昇腾亲和产品方向分析.md](PyPTO3.0_昇腾亲和产品方向分析.md)：阶段轴 × 泄漏点轴、8 个泄漏点统计、六个抓手、剂量原则。
- [Insight/PTO3写算子过程中的硬件内存感知与配置.md](../Insight/PTO3写算子过程中的硬件内存感知与配置.md)：内存层级、容量表、降低链路、写算子检查清单。
- [Insight/hw-native-sys-仓库功能洞察报告.md](../Insight/hw-native-sys-仓库功能洞察报告.md)：七仓分层、各仓定位。
- [Competitive_Analysis/四框架Pass对比_PyPTO_TileLang_Triton_cuTile_20261019.md](../Competitive_Analysis/四框架Pass对比_PyPTO_TileLang_Triton_cuTile_20261019.md)：五类 pass 抽象、「III 与 IV 不可共享」、「正确性保障不可被 autotune 替代」。
- [Insight/05Operator-hardware.md](../Insight/05Operator-hardware.md)：Ascend GM→L1/UB/L0→Cube/Vector 执行链路与十层案例。

### 复核方法

```bash
# Q 列缺口：规格 getter 在文档中零命中
grep -rln "get_l0a_capacity_bytes\|get_gm_access_granularity" repo/pypto/docs
```

```bash
# C 列缺口：唯一的 PerfHint 码
grep -rn "PH00" repo/pypto/src repo/pypto/include | grep -o "PH0[0-9][0-9]" | sort -u
```

```bash
# 作用域层级统计：用户侧各章页数
for d in repo/pypto/docs/zh/user/*/; do echo "$d $(ls $d*.md | grep -v index | wc -l)"; done
```

> 本文只做维度梳理与内容建议，不包含代码或产品实现；所有数值均标注了可复核位置。若 `repo/pypto` 镜像更新，应按上述方法重新核对后再引用。
