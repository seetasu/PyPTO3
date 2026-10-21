# Parallel Lane

单个 program 的并行结构线：覆盖 8 个决定函数切分与核归属的 pass，顶部是能判好坏的 KPI，主画布是全部函数的核归属矩阵，下方是最终调用树。

- 入口：`index.html`；已加入 `launch.html`。
- 数据：`data/parallel-lane.json`，由 `scripts/build_parallel_lane.mjs` 从 pass dump 直接生成。
- 直接双击打开即可：数据已内联进 `index.html`，不依赖 `fetch`。想走 HTTP 服务也行，仓库已有 `.claude/launch.json` 的 `design-static`。

## 重建数据

```bash
node Design/parallel-lane/scripts/build_parallel_lane.mjs
```

解析 `Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/passes_dump/` 下 8 个快照里 `@pl.function(...)` 的 `FunctionType` / `Level` / `Role`，以及 `pl.submit` / `pl.spmd_submit` / `self.X()` 三种调用边和 `core_num`。

构建脚本写两份：`data/parallel-lane.json`（源数据，可被其它工具复用）和 `index.html` 里 `<script type="application/json" id="lane-data">` 的内联副本。两份由同一条命令同时刷新，不要手改其中一份。内联是因为相对 `fetch` 在 `file://` 和桌面端文件预览用的 `data:` URL 下都会失败。

## 覆盖的 8 个 pass

| # | pass | 函数 | 本步新建 | 内联消失 | 改类型 |
| - | ---- | ---- | -------- | -------- | ------ |
| 0 | frontend（基线） | 24 | — | — | — |
| 1 | InlineFunctions | 4 | 0 | **20** | 0 |
| 8 | OutlineHierarchyScopes | 4 | 0 | 0 | 0 |
| 9 | OutlineIncoreScopes | 64 | **60** | 0 | 0 |
| 10 | OutlineClusterScopes | 112 | **48** | 0 | 0 |
| 22 | LowerAutoVectorSplit | 112 | 0 | 0 | 0 |
| 23 | ExpandMixedKernel | 116 | 4 | 0 | **62** |
| 25 | SplitVectorKernel | 116 | 0 | 0 | 0 |

并行结构在这个 run 里由 4 个 pass 完全决定（1 / 9 / 10 / 23）；其余 44 个快照的函数集合逐字相同，加进来只增加列、不增加信息。`OutlineHierarchyScopes`、`LowerAutoVectorSplit`、`SplitVectorKernel` 对本 program 是空操作。

类型演化：

```
InCore=2 Inline=20  →  InCore=2  →  InCore=62  →  InCore=62 Spmd=48  →  AIC=14 AIV=50 Group=2 Spmd=48
```

## KPI 与判据

| KPI | 本 program | 判据 |
| --- | ---------- | ---- |
| **Cube : Vector 函数配比** | **1 : 3.57**（14 AIC / 50 AIV） | 910B 一个 die 是 24 AIC + 48 AIV，即 1:2（`repo/pto/src/backend/common/soc.cpp:144`）。本 program 的 vector 侧函数比硬件配比多 79%，说明计算被切成了更多 vector kernel |
| 函数数 | 116（24 → 4 → 116） | 先被内联压到 4，再被外提放大 29 倍 |
| SPMD 任务 / 提交点 | 48 / 53 | 41 个 `spmd_submit` + 12 个 `submit` |
| 调用树深度 | 5 | HOST → CHIP → CORE_GROUP → AIC → AIV |
| 叶子 kernel | 64 | 占 55%，其余 52 个是编排脚手架 |
| 静态并行宽度 | 13 / 41 | 只有 13 个 `core_num` 是常量（最大 48），28 个是动态符号 |

**结论：切分本身是健康的——层级完整、无孤立节点、脚手架占 45% 属正常；唯一可疑的是 1:3.57 的 Cube:Vector 配比偏离硬件 1:2。但这是函数个数不是执行时间，要判断是否真的失衡得看每个 kernel 的耗时，dump 里没有。**

## 读图

- **主画布**：每行一个函数（136 个主体），8 列是 8 个 pass。行按最终类型分带，带标签在左侧。
  - 填充色 = 该步这个函数绑在哪类核上。蓝 = 编排、灰 = 未定核 InCore、橙 = Cube AIC、绿 = Vector AIV。
  - 格子左缘亮边 = 这一步该函数的类型被改写了。填充与改写是两个通道，不混用。
  - 最暗的格子 = 该步函数尚不存在（或已被内联掉）。
  - 列头标 `+N 新建` / `−N 内联` / `↻N 改类型`。
- **调用树**：最终 116 个函数按深度分列。实线 = `pl.submit`，虚线 = `pl.spmd_submit`（悬停看 `core_num`），细线 = `self.X()` 直接调用。选中节点的进出边高亮。
- **详情表**：选中函数在 8 个 pass 的类型、层级、角色、体量、dump 行号；变化的格子着色。

顶部下拉、点矩阵行、点树节点都可以换观测对象。

## 口径与限制

1. 主体身份用函数名匹配。本 run 每个快照内函数名互不重复，pass 09 诞生的 64 个名字到最后一字未改，因此跨列连续性可靠；换一个 run 需重新核对。
2. 源码里的 `Inline` helper `hc_post` 和 `rms_norm` 在 pass 01 被内联掉，pass 09 又把同名字给了新外提的 `InCore` 函数。这两个名字各拆成两个主体（`name` 与 `name#2`），不是一条连续的线。
3. 并行宽度 `core_num` 多数是动态符号表达式，编译期取不到具体核数，页面只统计常量宽度。
4. `ExpandMixedKernel` 之后的核内拆分（`PTOOutlineSIMTSections`、`VPTOSplitCVModule`）在 PTOAS，不产出 pass dump，这里取不到。
5. 函数体行数是 dump 文本行数，不是指令条数，只能横比不能当代价。
6. Cube:Vector 比的是函数个数，不是执行时间。
