# PyPTO3.0 工具昇腾亲和产品方向分析

> 调研日期：2026-09-14
> 证据范围：`repo/pto` 源码与开发者文档、`github_issues/pto` 与 `github_issues/PTOAS` 的 Issue 归档、`Insight/PTO3写算子过程中的硬件内存感知与配置.md`
> 阅读对象：PyPTO3.0 工具的产品与体验设计决策者

## 摘要

本文回答一个问题：PyPTO3.0 工具要做「昇腾亲和」，应该从哪些方面入手。

核心判断是：**昇腾亲和的着力点不在开发阶段的划分上，而在「抽象泄漏点」上。** PyPTO 语言层刻意做了硬件无关抽象，但正确性和性能仍由昇腾微架构裁决；语言层抽象掉的昇腾语义，在编译期和运行期又必须被还原。工具的价值就是在这些泄漏点上按需还原昇腾语义，而不是再叠加一层硬件特色卖点。

基于 632 条 Issue 标题的统计，泄漏点集中在七处，占全部 Issue 标题的 28%。据此给出六个产品抓手，并建议以「源码映射」为第一优先级。

## 0. 本文的证据约定

为避免把假设写成结论，全文按以下方式标注：

- **【事实】**：可在仓库文件或 Issue 归档中直接核对，给出路径或 Issue 号。
- **【统计】**：由本文的检索方法得出，方法在第 2 节说明，可复现。
- **【推断】**：基于上述证据的产品判断，非既有结论。
- **【建议】**：产品动作建议。

本文只做分析与建议，不包含代码或产品实现。

## 1. 核心判断：语言层去昇腾化，裁决权仍在昇腾

### 1.1 语言层是刻意去昇腾化的

**【事实】** hw-native-sys/pypto#268（2026-02-26 创建，已 closed）把 `MemorySpace` 枚举从昇腾硬件名改为逻辑名：

| 原名 | 新名 | 含义 |
|---|---|---|
| `L1` | `Mat` | 矩阵 / L1 缓冲区 |
| `UB` | `Vec` | Unified Buffer / 向量操作数暂存 |
| `L0A` | `Left` | Matmul 左操作数 |
| `L0B` | `Right` | Matmul 右操作数 |
| `L0C` | `Acc` | 累加器 |

该 Issue 给出的理由是：可移植（抽象可映射到昇腾以外的硬件）、可读（名字反映逻辑角色）、可维护（不需要了解 Ascend internals 即可推理）。Issue 中明确记录：「保留昇腾名作为别名」的方案被评估后否决。

### 1.2 但裁决权完全在昇腾微架构

**【事实】** 编译器内部与运行时是彻底昇腾化的：

- Cluster 拓扑为 1 个 Cube 核 + 2 个伙伴 Vector 核；每对等核心每方向 8 个硬件 flag，每 Cluster 共 32 个（[00-cluster_architecture.md](../repo/pto/docs/zh-cn/reference/pto-isa/00-cluster_architecture.md)）。
- 跨核 ring buffer 单向 8 槽、双向每向 4 槽；A2/A3 平台 ring 位于 GM，A5 位于消费者片上 SRAM（同上）。
- GM 访问粒度 Ascend910B 为 512 B、Ascend950 为 128 B；L2 cache line 均为 512 B（[92-diagnostics.md](../repo/pto/docs/zh-cn/dev/passes/92-diagnostics.md)）。
- PTOAS 侧仍有昇腾硬编码与编译宏：`TileConfig.fractalABSize` 硬编码 512（hw-native-sys/PTOAS#57）、需要生成 `__DAV_C220_VEC__` 类毕昇编译宏（hw-native-sys/PTOAS#36）。

### 1.3 结论

**【推断】** 语言层承诺了硬件无关，正确性与性能却 100% 由昇腾微架构决定。这道缝隙不会因为语言层继续抽象而消失——它只会转移到编译期和运行期。

**工具正是站在这道缝隙上的产品。** 因此昇腾亲和对工具的定义是：

> 在抽象泄漏的地方，按需把昇腾语义还原回来。

这也解释了为什么按开发阶段（表达 / 编译 / 执行 / 优化）来切分昇腾亲和是不够的——那套划分套用在任何编译器工具上都成立，无法回答「PyPTO 的昇腾亲和特殊在哪」。按泄漏点切分才具备 PyPTO 的专属性。

## 2. 证据：七个泄漏点

### 2.1 统计方法

**【统计】** 从 `github_issues/pto/pypto_issues.csv` 中提取全部 Issue 标题共 **632 条**，按关键词归入各家族。说明三点口径：

1. 仅匹配标题，不匹配正文，因此结论为保守下界。
2. 家族之间**存在重叠**（例如一条 Issue 可能同时涉及 matmul 与 layout），各家族计数之和大于并集。
3. 首轮统计中 `ring` 关键词误匹配了 `during` / `string` / `ordering` 等词，已收紧为 `tpush|tpop|pipe_buffer|ring buffer|ring_task|cross-core|跨核` 后重新计数。

### 2.2 分布

| 泄漏点 | 命中 | 昇腾根因 |
|---|---:|---|
| AIC / AIV 切分 | 45 | Cluster 为 1 Cube + 2 Vector，跨核即需 ring + flag |
| matmul 与 L0 分块 | 42 | L0A / L0B / L0C 容量决定 m / n / k |
| layout 与布局 | 33 | Cube 要求 NZ / ZN fractal、16 对齐、row/col_major |
| 内存复用与地址分配 | 31 | 片上空间稀缺，必须复用 |
| 跨核 ring 通道 | 23 | TPUSH / TPOP 多槽环形缓冲与流控 |
| valid_shape 与边界 | 19 | 非整除 shape 的 padding 与有效区域 |
| 同步与 flag | 15 | 硬件 SET / WAIT flag |
| 内存空间语义 | 10 | `MemorySpace` / `target_memory` 的表达与序列化 |

**【统计】** 去重并集 **178 条，占全部 632 条标题的 28%**。

**【推断】** 近三成的 Issue 标题落在昇腾微架构映射上。这几处就是昇腾语义在 PyPTO 中的主要泄漏口，也是工具做亲和的优先靶区。

### 2.3 三条需要单独拎出来的证据

#### 证据一：硬件性能提示至今只有一条，且无法回到用户源码

**【事实】** 全系统注册的 diagnostic check 共 3 条，其中 `PerfHint` 级别**只有 1 条**：`TileInnermostDimGranularity`（PH001），检查 `tile.load` / `tile.store` 最内维字节数是否低于平台建议粒度。来源：[diagnostic_check_registry.cpp](../repo/pto/src/ir/verifier/diagnostic_check_registry.cpp) 第 79 至 90 行，全仓库仅出现 `PH001` 一个 hint 码。

**【事实】** hw-native-sys/pypto#1305（2026-05-07 创建，**目前仍 open**）记录了该检查的关键缺陷：其 span 指向流水线后的 IR 文本位置（`<string>:line:col`），而非用户编写的 DSL 源位置；`TileType`、`Call` op、`Span` 均未携带该元数据。

**【推断】** 这意味着当前唯一一条昇腾性能建议，用户拿到后无法定位到自己写的那一行。**源码映射缺失是整个昇腾亲和的公共地基问题**，它不解决，其余亲和能力的结论都只能停在「IR 某处有问题」。

#### 证据二：做昇腾决策的 pass，需要专门立项才补齐文档

**【事实】** 承担昇腾硬件决策的几个 pass，其文档是通过专门的 Docs Issue 在 2026-04 至 2026-05 期间补齐的。这些 Issue 目前**均已 closed**，其中三个已产出对应文档文件（`ResolveTransposeLayout` 未在 `docs/zh-cn/dev/passes/` 下检索到同名文档）：

| Issue | Pass | 现有文档 | 状态 |
|---|---|---|---|
| #1164（2026-04-25） | `InferTileMemorySpace` | `16-infer_tile_memory_space.md` | closed |
| #1165（2026-04-25） | `ResolveTransposeLayout` | — | closed |
| #1166（2026-04-25） | `ResolveBackendOpLayouts` | `17-resolve_backend_op_layouts.md` | closed |
| #1271（2026-05-06） | `AutoTileMatmulL0` | `14-auto_tile_matmul_l0.md` | closed |

**【推断】** 这些 pass 需要单独立项补文档，说明其硬件决策逻辑不具备自解释性。文档补齐解决了「团队内部可查」，但没有解决「用户在自己的代码上下文里理解这些决策」。后者是工具的职责。

#### 证据三：内存复用的失败模式是静默数据损坏

**【事实】** `MemoryReuse` 家族的典型故障并非报错，而是产生错误结果：

- #585（2026-03-17）：错误地将 `gate_acc` 与 `up_acc` 混叠。
- #768（2026-03-28）：为存活的 loop yield 输出复用 buffer，**导致数据损坏**。
- #673（2026-03-23）→ #1310（2026-05-08，标题明确记为 #673 的 regression）→ #1352（2026-05-12）：acc→acc 非法 `pto.tmov` 问题反复出现三次。

**【推断】** 这类问题靠阅读 IR 文本无法发现，且存在回归反复。它需要的不是解释文案，而是 buffer 生命周期与别名关系的可视化。

#### 补充：代际差异会直接导致挂死

**【事实】** #828（2026-04-01）：一个 dst == src 的 identity `pto.tmov` 导致 ptoas 插入多余同步，**在 A5 上挂死**。

**【推断】** A2/A3 与 A5 的 ring buffer 放置策略不同（GM 对片上 SRAM），同一段代码在两代芯片上的瓶颈与故障模式不同。跨代解释能力是刚需，且无法靠人工读代码获得。

## 3. 六个产品抓手

以下按泄漏点组织，而非按开发阶段组织。

### A. 双语对照：把逻辑名还原为昇腾硬件名

**【建议】** 在工具中提供 `Mat / Vec / Left / Right / Acc` 与 `L1 / UB / L0A / L0B / L0C` 的随时可切换对照（悬浮提示、侧栏标注、按 target 开关）。

**理由【推断】**：语言层因 #268 的可移植性承诺，不能把 `Mat` 叫回 `L1`；但工具没有这层约束——工具本就应当按 target 切换显示。开发者已有的昇腾知识（Ascend C、CANN 文档、团队经验）全部以 UB / L1 / L0A 组织，当前这套存量经验与 PyPTO 概念之间隔着一次手工翻译。**这次翻译只能由工具承担。**

**性价比**：实现成本约等于一张映射表加显示层，收益是开发者存量昇腾经验立即可用。本文认为这是被低估程度最高的一条。

### B. 把硬件约束从编译期报错前移为写码期契约

**【建议】** 在提交编译之前运行一次「昇腾约束体检」，覆盖泄漏点中可静态判定的部分：

- L0A / L0B / L0C 容量能否容纳当前 m × n × k；
- shape 是否满足 NZ / ZN fractal 的 16 对齐要求（对应 #724）；
- 最内维字节数是否达到目标平台建议粒度（910B 512 B / 950 128 B）；
- 跨核边界两侧的 layout 要求是否一致（对应 #737、#763）；
- Tile 生命周期是否重叠、是否存在 no-alias 要求。

**理由【事实 + 推断】**：现有约束以 `CHECK` 抛 `ValueError` 或 pass 内部断言的形式暴露（[02-error-handling.md](../repo/pto/docs/zh-cn/dev/02-error-handling.md)），时机晚且使用编译器内部语言。体检结论应当用昇腾语言表述，例如「L0C 容量不足以容纳 m×n，建议在 N 方向切分」，而非断言堆栈。

约束清单可直接复用 [PTO3写算子过程中的硬件内存感知与配置.md](../Insight/PTO3写算子过程中的硬件内存感知与配置.md) 第 12 节已整理的检查清单。

### C. 让编译器的昇腾决策可见、可归因、可干预

**【建议】** 对下列决策点，每一个都要能回答三句话——**编译器选了什么 / 依据哪条昇腾规格 / 我可以怎么改**：

`InferTileMemorySpace`、`AutoTileMatmulL0`、`ResolveBackendOpLayouts`、`ResolveTransposeLayout`、`ExpandMixedKernel`、`MemoryReuse`、`AllocateMemoryAddr`、`InsertSync`。

**其中 `MemoryReuse` 需要区别对待**：依据证据三，它的失败是静默数据损坏且反复回归，因此它需要的交付物是 buffer 生命周期与别名关系图，而不是解释文本。

### D. 源码映射：当前最硬的断点

**【建议】** 建立 IR 位置与用户 DSL 源位置的映射，使所有昇腾结论都能落到用户写的那一行。两条路径：推动 IR 携带 DSL span（即 #1305 所指方向），或由工具侧自建映射索引。

**理由【事实 + 推断】**：#1305 仍 open，明确记录 `TileType` / `Call` / `Span` 均不携带 DSL 源元数据。在此之前，抓手 A、B、C、E 的结论都无法精确落点，价值均需打折。本文认为这是整个昇腾亲和的地基。

### E. 执行证据按昇腾结构组织

**【建议】** 运行时视图不做通用 timeline，而按昇腾执行结构组织：

- 泳道按 Cluster / AIC / AIV 排布；
- 显示 flag 的 SET / WAIT 配对关系；
- 显示 ring 槽位占用与流控阻塞点；
- 区分 `syncall` 的 hard 与 soft 形态；
- 支持 A2/A3 与 A5 的对照解释（ring 在 GM 对 ring 在消费者 SRAM）。

**【事实】** 仓库中 `Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/dfx_outputs/` 下已有 `chip_swimlane_records.json`、`deps.json`、`merged_swimlane_*.json` 等真实运行数据，足以支撑原型验证，无需等待新数据采集。

### F. 规格作为数据，且必须与编译器同源

**【建议】** 建立「目标平台档案」，容纳 GM 访问粒度、L2 cache line、L0 / L1 / UB 容量、Cluster 拓扑、flag 数量、ring 槽数、fractal 尺寸等，所有页面的阈值、建议文案与检查规则统一引用它。

**关键约束【推断】**：该档案必须从 `BackendHandler` 等编译器源头生成，不可由工具侧手工维护第二份。若工具与编译器对同一规格给出不同数值，用户会照工具修改而编译不过——**这比不做亲和更糟**。

## 4. 优先级建议

**【建议】** 推荐顺序：**D → A → C → B → E → F**。

| 顺位 | 抓手 | 依据 |
|---|---|---|
| 1 | D 源码映射 | 地基；#1305 仍 open，缺失明确；不做则其余全部打折 |
| 2 | A 双语对照 | 成本近乎为零，立刻激活开发者存量昇腾经验 |
| 3 | C 决策可见 | 针对静默损坏等最难自查的问题，信息增益最高 |
| 4 | B 约束前移 | 依赖 D 与 F 提供落点和阈值 |
| 5 | E 执行视图 | 已有真实数据，但价值依赖 D 的源码回溯 |
| 6 | F 规格同源 | 工程性强，可与 B 并行推进 |

### 不建议先做的

**【建议】** 不建议优先建设通用性能调优顾问或 roofline 类能力。

**理由【统计】**：证据显示主要痛点是**正确性**——layout（33）、内存复用（31）、同步（15）合计 79 条标题；而硬件性能提示至今只有 PH001 一条。先做性能建议会偏离真实痛点分布。

## 5. 需要正视的设计张力

**【推断】** 工具的昇腾亲和做得越深，与语言层 #268 所确立的「硬件无关」方向张力越大。

建议的处理方式是：**把昇腾亲和全部落在可按 target 开关的视图层与检查层，不写入 DSL 语义。** 这样语言层保留其可移植性承诺，昇腾亲和由工具承担，两侧均无需让步。

这一原则同时决定了抓手 A 的实现边界：双语对照是显示层能力，不是给 DSL 增加别名——后者已在 #268 中被明确否决。

## 6. 风险

| 风险 | 说明 | 应对 |
|---|---|---|
| 规格双源不一致 | 工具与编译器对同一硬件参数给出不同值 | 抓手 F：单一来源生成，禁止手工维护第二份 |
| 源码映射久拖不决 | #1305 涉及 IR 元数据改造，跨层协同成本高 | 准备工具侧自建映射的兜底方案，不把全部亲和能力押在 IR 改造上 |
| 统计口径被质疑 | 本文统计仅覆盖标题，且家族间重叠 | 第 2.1 节已公开方法与口径，结论以「下界」表述 |
| 亲和能力随芯片迭代过期 | A2/A3/A5、910B/950 差异持续演进 | 规格与规则随 target 版本化，不写死在页面文案中 |

## 7. 证据索引

### Issue（仓库 hw-native-sys/pypto，除注明外）

| 编号 | 标题要点 | 状态 | 创建日期 |
|---|---|---|---|
| #268 | MemorySpace 改为硬件无关命名 | closed | 2026-02-26 |
| #585 | MemoryReuse 混叠 gate_acc / up_acc | closed | 2026-03-17 |
| #673 | acc→acc 非法 tmov | closed | 2026-03-23 |
| #724 | NZ/ZN fractal 未做 16 对齐 padding | closed | 2026-03-25 |
| #746 | shape [M, 1] 的 layout 不匹配 | closed | 2026-03-27 |
| #768 | 复用存活 yield 输出导致数据损坏 | closed | 2026-03-28 |
| #828 | identity tmov 引发多余同步，A5 挂死 | closed | 2026-04-01 |
| #1164 / #1165 / #1166 | 补齐三个硬件决策 pass 文档 | closed | 2026-04-25 |
| #1229 | 数据流经 GM 后 blayout 校验失败 | closed | 2026-04-30 |
| #1271 | 补齐 AutoTileMatmulL0 文档 | closed | 2026-05-06 |
| #1305 | PH001 缺少源码映射等改进 | **open** | 2026-05-07 |
| #1310 / #1352 | acc→acc 问题回归 | closed | 2026-05-08 / 2026-05-12 |
| PTOAS#36 | 需生成毕昇 `__DAV_C220_VEC__` 宏 | — | — |
| PTOAS#57 | fractalABSize 硬编码 512 | — | — |

### 仓库文件

- [repo/pto/docs/zh-cn/reference/pto-isa/00-cluster_architecture.md](../repo/pto/docs/zh-cn/reference/pto-isa/00-cluster_architecture.md)：Cluster 拓扑、flag、ring buffer、平台差异。
- [repo/pto/docs/zh-cn/dev/passes/92-diagnostics.md](../repo/pto/docs/zh-cn/dev/passes/92-diagnostics.md)：诊断体系、各 backend 阈值、PH001。
- [repo/pto/src/ir/verifier/diagnostic_check_registry.cpp](../repo/pto/src/ir/verifier/diagnostic_check_registry.cpp)：已注册的 3 条检查。
- [repo/pto/docs/zh-cn/dev/02-error-handling.md](../repo/pto/docs/zh-cn/dev/02-error-handling.md)：CHECK / INTERNAL_CHECK 与 span。
- [Insight/PTO3写算子过程中的硬件内存感知与配置.md](../Insight/PTO3写算子过程中的硬件内存感知与配置.md)：内存层级、降低链路、写算子检查清单。
- `Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/dfx_outputs/`：可用于原型验证的真实运行数据。
- `github_issues/pto/pypto_issues.csv`：632 条 Issue 归档，本文统计来源。

### 统计复现方法

从 CSV 中以 `^"[0-9]+","[^"]{0,200}"` 提取标题行，再按第 2.2 节各家族关键词分别计数；并集统计使用全部家族关键词的合并正则。`ring` 一词需收紧匹配，以排除 `during` / `string` / `ordering` 等误命中。
