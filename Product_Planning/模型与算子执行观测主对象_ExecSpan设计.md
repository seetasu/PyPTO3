# 模型与算子执行观测主对象：ExecSpan

> **目的**：为"模型 / 算子执行过程可视化观测 Demo"确定唯一的观测主对象，使模型级和算子级不是两个页面，而是同一个对象的两个缩放层级。
>
> **依据**：[PTO3 源码、调度与硬件执行概念术语手册](./PTO3_源码_调度_硬件执行概念术语手册.md) 的三条线模型；`Data/pypto_qwen3_profiles/` 的 Qwen3-14B 真实采集数据（40 层 / 643 个 aicore_tasks / 60 核）。
>
> **状态**：设计稿，尚未实现。字段语义已对照采集脚本核实，标注见第 8 节。
>
> **定位修订（2026-09-29）**：ExecSpan 是**执行侧**观测对象，只覆盖"源码 → 编译 → 性能"全链的后三分之一。全链主体是 [Callable](./模型与算子全链路观测主体_Callable设计.md)，本对象降级为它的执行面与下钻视图。另：第 8 节把 `dep` 边与张量信息标为 `inferred`，因发现 `dfx_outputs/deps.json` 提供了 `tensor_id` 与完整 shape，应升级为 `derived`。

---

## 1. 结论

**观测主对象 = `ExecSpan`（执行跨度）。**

一句话定义：

> **ExecSpan 是"一次真实发生的执行"所占据的一段「资源 × 时间」，并同时携带它在源码线、调度线、设备线上的三重身份。**

它不是 trace event 的换名。区别在三点：

| | Chrome trace event | **ExecSpan** |
|---|---|---|
| 身份 | 只有设备身份（tid、ts、dur、name 字符串） | **三重身份**：源码 / 调度 / 设备，可互相回溯 |
| 时长 | 一个数 `dur` | **状态分解**：等依赖 / 等资源 / 核上执行 / 核上空转 |
| 层级 | 靠 name 前缀人工分组 | **递归自相似**：同一结构表达 step → layer → task → block → 阶段 |
| 关系 | 无（或仅 flow 箭头） | **因果边**：依赖 / 同步 / 数据流 / 展开，可算关键路径 |

选择它作为主对象的理由：现有仓库资产里，`Timeline` / `TimelineView` 是**视图原语**，`Run` / `Task` / `Event` 是**散落的记录**，中间缺的正是一个能被点选、下钻、追溯、比对的**领域对象**。Demo 的一切交互都应该是"对 ExecSpan 做操作"。

---

## 2. 为什么模型级和算子级可以共用一个对象

三条线在手册里已经写清：`scope → Function → kernel → task → block → core`。这条链的每一环都在回答同一个问题——"**这段工作，在什么资源上，占了哪段时间，为什么是这么长**"。

因此把 ExecSpan 设计成**递归**的：一个 ExecSpan 的 children 仍是 ExecSpan。

| 层级 | ExecSpan 实例 | 资源轴 | 典型跨度 | 谁关心 |
|---|---|---|---|---|
| L0 `step` | 一次 prefill / 一个 decode step | 整卡 | ms | 模型工程师 |
| L1 `layer` | 第 17 层 / 一个融合区域 | 整卡 | 百 µs | 模型工程师 |
| L2 `task` | 一次 `Submit` 产生的 task（如 `q_proj`） | core group | 十 µs | 两者交界 |
| L3 `block` | SPMD 展开的 `block[i]` 实例 | 单个 AIC/AIV 核 | µs | 算子工程师 |
| L4 `stage` | block 内 load / compute / store 流水阶段 | 核内单元 | 百 ns | 算子工程师 |
| L5 `tileop` | `tile.load` / `tile.matmul` 等 Tile 操作 | 核内单元 + MemorySpace | ns | 算子工程师 |

**缩放（zoom）就是唯一的主交互**。用户从"decode 第 3 步慢"一路下钻到"`q_proj` 的 block 12 在等 K 搬运"，全程不换页面、不换心智模型。

> L0–L3 可从现有真实数据直接构造；L4–L5 目前需要更细的采集或从 IR 推导，Demo 中应标注为 `derived` / `inferred`（见第 6 节与第 8 节）。

---

## 3. 对象模型

```jsonc
{
  "id": "span:decode3/L17/q_proj/blk12",
  "level": "block",                  // step | layer | task | block | stage | tileop
  "parentId": "span:decode3/L17/q_proj",
  "childIds": [],

  // ── 三重身份：主对象的核心，缺一不可 ──
  "identity": {
    "source": {                      // 用户源码线：稳定、可跨 Run 对比
      "nameHint": "q_proj",
      "scopeKind": "InCore",         // InCore | Spmd | Runtime | CommDomain
      "sourceRef": "qwen3_decode.py:184",
      "callChain": ["decode_fwd", "attn_block", "q_proj"]
    },
    "schedule": {                    // 编译与调度线：随 Run 变化
      "callableId": 7,
      "functionName": "q_proj",
      "kernelKind": "AIC",           // AIC | AIV | Group
      "taskToken": 12884901896,
      "ringId": 3,                   // taskToken >> 32
      "localTaskId": 8,              // taskToken & 0xFFFFFFFF
      "submitIdx": 41,
      "loopIter": 17,
      "blockIdx": 12,
      "blockNum": 48
    },
    "device": {                      // 真实设备执行线：实测事实
      "coreId": 5,
      "coreType": "aic",
      "coreThread": 2,
      "startCycle": 93134581864053,
      "endCycle": 93134581865295,
      "clockFreqHz": 50000000
    }
  },

  // ── 时间：永远是分解，不是一个数 ──
  "time": {
    "t0": 41230.4,
    "t1": 41255.2,                   // µs，对齐到 Run 起点
    "breakdown": {
      "depWait": 6.1,                // 依赖未满足
      "queueWait": 2.3,              // 依赖已满足、等资源（ready queue）
      "occupied": 16.4,              // 核上占用，以下为其细分
      "occupiedDetail": {
        "compute": 11.2,             // Cube / Vector 实际算
        "mte": 3.8,                  // 搬运 MTE1/2/3
        "syncStall": 1.1,            // barrier / fence 等待
        "idle": 0.3                  // 核上但未产出
      }
    },
    "onCriticalPath": true
  },

  // ── 资源占位 ──
  "resource": {
    "lane": "AIC_5",                 // 泳道键，与时间轴共同定位
    "memory": [                      // 该 span 期间的 MemorySpace 占用
      { "space": "L1", "bytes": 131072, "tensor": "q_tile" },
      { "space": "L0A", "bytes": 32768, "tensor": "qa" }
    ]
  },

  // ── 因果边：见第 5 节 ──
  "edges": { "in": [], "out": [] },

  // ── 证据分级：区分实测 / 派生 / 推断 ──
  "evidence": {
    "grade": "measured",             // measured | derived | inferred
    "sources": ["tp1/swimlane/decode_records/l2_swimlane_records.json#aicore_tasks[412]"]
  },

  // ── 观测结论（可为空；由分析或 Agent 填充） ──
  "findings": [
    { "kind": "bottleneck", "label": "等 K 搬运", "confidence": 0.8, "evidenceRef": "edge:dep/k_proj" }
  ]
}
```

**必填最小集**（Demo 首版）：`id` / `level` / `parentId` / `identity.device` / `time.t0,t1` / `resource.lane`。其余字段允许缺失，UI 必须能优雅降级——真实采集永远是不完整的。

---

## 4. 时间分解：主对象真正的价值所在

一条只有 `dur` 的 span 只能回答"多久"，回答不了"为什么"。ExecSpan 强制把时长拆成**四个互斥段**：

```text
submit ──┬── depWait ──┬── queueWait ──┬────────── occupied ──────────┬──> finish
         │  等前驱完成  │  等核资源      │  compute │ mte │ sync │ idle │
         └─ 调度线证据 ─┴─ 调度线证据 ──┴───────── 设备线证据 ──────────┘
```

这条分解直接决定页面的颜色编码和结论语言：

| 主导段 | 用户该看的结论 | 该跳转到哪 |
|---|---|---|
| `depWait` 大 | 依赖链问题，不是算子慢 | 因果图上游 span |
| `queueWait` 大 | 资源不足 / 分波（wave） | 同时段的核占用全景 |
| `mte` 大 | 访存瓶颈 | 该 span 的 MemorySpace 占用 |
| `syncStall` 大 | 同步设计问题 | 同步边与配对 span |
| `compute` 大 | 真·计算密集，看 tiling | 源码 + Tile 结构 |

> **数据现状**：`depWait` / `queueWait` 需要 `aicpu_scheduler_phases` 与 `aicore_tasks` 对齐后派生，标记 `derived`；`occupied` 内部四分目前 L2 记录不直接提供，首版 Demo 应**只展示 occupied 总量**，把内部细分留作后续采集，不得编造。

---

## 5. 因果边 `CausalEdge`

主对象的第二等公民。四类，足以覆盖观测需求：

| 类型 | 含义 | 数据来源 | 视觉 |
|---|---|---|---|
| `dep` | 前驱 task 必须先完成 | `deps` / AUTO 追踪 / scheduler phases | 实线箭头 |
| `sync` | barrier / fence / 跨核同步配对 | SyncOp、trace 配对事件 | 虚线双向 |
| `data` | 张量生产→消费、buffer 复用 | IR 数据流、MemRef 共享 | 细线 + 张量名 |
| `fanout` | task → N 个 logical block 的展开 | `blockNum` | 括号 / 收束线 |

有了边，**关键路径**就是可计算的，而不是靠人眼在泳道上找最长的那条。`time.onCriticalPath` 由此填充，也是 Demo 最有说服力的一个开关。

---

## 6. 视觉形态：Span 条

主对象在页面上的**唯一图元**是一根"分层胶囊"：

```text
          ┌──────┬───┬───────────────────────┐
 AIC_5    │▒▒▒▒▒▒│░░░│███████████████████████│   ← 一根 Span 条
          └──────┴───┴───────────────────────┘
           depWait queueWait      occupied
           斜纹     浅底          实色（按 coreType 分 AIC / AIV 两色）

          关键路径加描边；带 finding 时右上角挂角标
```

三条固定规则：

1. **横轴永远是真实时间**，宽度不得用最小显示宽度夸大短 span（qwen3 采集脚本的 PNG 已遵循此约定，Demo 应保持一致）。
2. **纵轴是资源泳道**：L0–L1 聚合视图下泳道为"层 / 阶段"，L2–L5 下泳道为真实 core。
3. **颜色只编码两件事**：核类型（AIC / AIV）和时间分解段。不要再用颜色表达第三个维度。

页面围绕主对象的**三个联动视图**（选中任一 span，三者同步）：

- **时间轴**（主视图）：span 在「资源 × 时间」上的分布
- **因果图**：选中 span 的上下游链与关键路径
- **身份卡**：三重身份 + 源码定位 + 证据来源 + 时间分解条

---

## 7. 主对象的交互契约

Demo 中所有操作都应表述为对 ExecSpan 的动词，便于后续封装为 Agent 工具：

| 动词 | 含义 | 返回 |
|---|---|---|
| `zoom(span, level)` | 上钻 / 下钻 | 子或父 ExecSpan 集合 |
| `locate(span, line)` | 跨线定位到源码 / 调度 / 设备 | 对应身份与证据 |
| `explain(span)` | 时间去哪了 | 分解 + 主导段结论 |
| `trace(span, dir)` | 沿因果边回溯 / 前推 | 边 + 邻接 span |
| `criticalPath(scope)` | 计算关键路径 | span 序列 |
| `compare(spanA, spanB)` | 同 `nameHint` 跨 Run / 跨 block 比对 | 差异分解 |
| `aggregate(spans, by)` | 按 nameHint / coreType / layer 聚合 | 统计 span |

> `compare` 必须按 **source 身份**（`nameHint`）对齐，绝不能按 `taskToken`——手册明确指出 TaskId 随 Run 变化，跨 Run 比数值是错的。

---

## 8. 与真实数据的映射（已核实）

以 `Data/pypto_qwen3_profiles/pypto_qwen3_profiles/tp1/swimlane/decode_records/` 为例：

| ExecSpan 字段 | 数据来源 | 核实情况 |
|---|---|---|
| `identity.device.coreId` / `startCycle` / `endCycle` | `l2_swimlane_records.json → aicore_tasks[i]` 第 0 / 3 / 4 列 | 已核实，见 `collect_qwen3_tp2_profiles.py:111-127` |
| `identity.schedule.taskToken` / `ringId` / `localTaskId` | 第 1 列，`ringId = token >> 32`、`localTaskId = token & 0xFFFFFFFF` | 已核实，同上 |
| `identity.schedule.callableId` | 第 2 列 `registered_task_id` | 已核实（脚本字段命名） |
| `identity.source.nameHint` | `name_map__jit_decode_fwd_*.json → callable_id_to_name` | 已核实，含 `q_proj` / `paged_attention_rope_cce_aic` 等 |
| `identity.device.coreType` | `metadata.core_types[coreId]`（tp1：20 aic + 40 aiv） | 已核实 |
| 时间换算 | `metadata.clock_freq_hz = 50_000_000` | 已核实 |
| `time.breakdown.depWait` / `queueWait` | `aicpu_scheduler_phases` / `aicpu_orchestrator_phases` | 字段存在，需对齐后派生，标 `derived` |
| `resource.memory` | 无对应字段 | 首版留空或从 IR 推导，标 `inferred` |
| `edges.dep` | 无显式记录 | 可由 `submit_idx` 顺序 + 调用链近似，须标 `inferred` |

**关键判断：三重身份在真实数据里是可以真正打通的**——`aicore_tasks` 的第 2 列经 `callable_id_to_name` 能回到源码 `name_hint`。这是整个设计成立的支点，也是这个 Demo 相比通用 Perfetto 的核心差异。

`Data/serving-strace-swimlane(2).json` 与 `tp1/swimlane/decode.json` 是 Chrome Trace 格式，可作为 L0–L1 的宿主侧补充；但它们缺少 callable 映射，单独使用会退化成普通 trace 查看器。

---

## 9. 与仓库现有资产的关系

| 已有资产 | 关系 |
|---|---|
| [算子作业可视化基础对象与原语封装规划](./算子作业可视化基础对象与原语封装规划.md) 的 `Run` / `Task` / `Event` / `DepEdge` | ExecSpan 是这些记录的**统一承载对象**，不替代它们，而是给它们共同的身份与时间模型 |
| 同文档的 `TimelineView` / `EvidenceChain` / `BreakdownView` 原语 | 它们是渲染 ExecSpan 的三个视图；本设计补齐了它们共享的数据契约 |
| [PyPTO3 算子开发可视化基础作业对象](./PyPTO3算子开发可视化基础作业对象.md) 的 `Timeline` / `Bottleneck` | ExecSpan 把二者合并：`Bottleneck` 降级为 span 上的 `findings` |
| `Design/runtime-trace-swimlane`、`Design/single-rank-inference-observatory` | 现有原型只做了"设备身份 + 单一时长"，正好是 ExecSpan 的最小子集，可作为渲染层参考 |

---

## 10. MVP 边界与下一步

首版 Demo 建议只做**四个层级 L0–L3**、**两类边（dep / fanout）**、**三段时间分解（depWait / queueWait / occupied）**，数据源用 tp1 decode。

待确认（影响实现，不影响本对象设计）：

1. Demo 落在 `Design/`（PTO Design System）还是 `inference/`（shadcn 风格工作台，且禁用 PTO DS 与 IDE Frame）；
2. 是否要求跨 Run 比对（决定是否同时加载 prefill + decode、tp1 + tp2）；
3. `occupied` 内部四分是否值得补采集——这是"为什么慢"能否回答到算子内部的分水岭。
