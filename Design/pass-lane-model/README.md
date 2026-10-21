# Pass Lane Model

把 pass 可视化的主角从「pass」换成「被观测对象」：pass 退化成时间刻度，纵向看的是某一类对象跨 pass 的整体变化。

- 入口：`index.html`；已加入仓库 `launch.html`。
- 对象模型：`schema/`（四个 JSON Schema + 六条线的定义数据）。
- 本页实现的是六条线中的**数据形态线**；内存线见 [`../memory-pass-replay`](../memory-pass-replay)。
- 通过本地 HTTP 服务打开页面（`fetch` 在 `file://` 下被浏览器拒绝）。仓库已有 `.claude/launch.json` 的 `design-static` 配置。

## 对象模型

三元组，关系是 `PassRun --emits--> LaneEvent --on--> Subject`：

| 对象 | 文件 | 作用 |
| ---- | ---- | ---- |
| **Lane** | `schema/lanes.schema.json` | 关注对象的类别，也是事件频道。**一个 pass 可以向多条 lane 发事件，所以 lane 不是对 pass 的划分。** |
| **Subject** | `schema/subjects.schema.json` | 被观测对象的实例（tile / function / op / task…），是可视化的主角 |
| **LaneEvent** | `schema/events.schema.json` | 某个 pass 在某条 lane 上对某个 Subject 做的一件事 |
| **Manifest** | `schema/manifest.schema.json` | 一次投影的范围、刻度表与口径声明 |

六条线的定义在 `schema/lanes.sample.json`：内存、数据形态、算子形态、并行结构、时序、依赖，外加一条默认折叠的「规范化」。每条线声明了 `emitters`（pypto 侧的 pass）与 `downstreamEmitters`（PTOAS 侧延续该线的 pass，dump 通常取不到）。

几个刻意的设计点：

- `visual.form` 与语义解耦——换渲染形态不改数据。
- `metricKind` 区分 count / bytes / ratio / cardinality，**不同口径不得混成一个「大小」轴**（tile 定义数与分配槽数差一个数量级）。
- `cardinality` 字段显式表达 1:3（`ExpandMixedKernel` 拆核）、4:1（`MemoryReuse` 合槽）这类基数变化——**它们不是 diff，不能用「changed + 数量差」表达**。
- `stages[].skipped` 让被配置门控跳过的 pass 仍占刻度，避免两次编译的时间轴对不齐。
- `identity.identityConfidence` 决定一个 Subject 能不能看时间线（见下）。

## 数据来源与重建

```bash
node Design/pass-lane-model/scripts/build_shape_lane.mjs
```

从 `Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/passes_dump/` 的 10 个快照（pass 11–32）抽取 callable `csa_merge_pack_publish` 的全部 `pl.Tile` 声明，产出 `data/shape-lane.json`。

脚本用 node 写（`memory-pass-replay` 的同类脚本用 Python `ast`）：这里改为括号匹配的文本解析，不依赖 Python 环境。

## 这条线上实际看到了什么

**tile 定义数：64 → 65 → 101 → 138。** 增量集中在三步：`CanonicalizeTileSlice` +1、`ResolveBackendOpLayouts` +36、`MaterializeTensorStrides` +37。而形状与秩的分布几乎不变（始终是 1 个 rank-1、其余 rank-2）。

也就是说，**这个 callable 在数据形态线上的变化不是「形状被改写」，而是「tile 数量翻倍」**——这正是逐 pass diff 看不出、跨 pass 聚合才能看到的东西。

## 身份问题（这条线最重要的发现）

跨快照追一个 tile 要靠名字，而名字并不总是可靠：

| 置信度 | 数量 | 含义 | 本页行为 |
| ------ | ---- | ---- | -------- |
| `exact` | 102 | 带语义前缀的名字 | 开放逐对象时间线 |
| `name-match` | 29 | `t__tile_N` 计数器名 | 只发 created / removed |
| `heuristic` | 14 | 同名但 dtype 不同，**已确认是重名** | 只发 created / removed |

判据：**tile 的 dtype 是类型标注的一部分，不可变**（cast 产生新值而非原地改写），所以同名不同 dtype 一定是不同的临时量复用了同一个计数器名。按此判定后，还剩 6 处属性变化（shape 4、rank 2）全部落在计数器名上，无法证明连续性，因此**只计数、不出事件**（`manifest.counts.unresolvedChanges`）。

结论：**要让逐对象时间线覆盖全部 tile，dump 需要带上 Span，或改用进程内 instrument 而非文本快照。** 聚合视图不受此影响——按快照统计分布不需要逐对象身份，所以它是主视图。

## 口径与限制

1. 柱高是 callable 内 `pl.Tile` 的**定义数**，与 `memory-pass-replay` 的**分配槽数**是两个口径。
2. 本 run 的 dump 共 52 个快照，pass 序列与当前 `pass_manager.py` 的 `Default` 策略不完全一致（含 `BlockNzTensorViews`、`LegalizeTileCast`、`LowerPipelineToSlots` 等本 run 特有的 pass）。
3. `memory` 线此处只带 `memref-bound` 一类事件；MemRef 与 TileView 要到 pass 33 之后才出现，不在本窗口内。
4. 行号是静态 dump 行号，不是源码行号。
5. 配色经 `dataviz/scripts/validate_palette.js --mode dark` 六项检查全部通过；超出前 5 种的形状归入中性「其他」，不生成新色相。
