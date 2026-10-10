# 模型与算子全链路观测主体：Callable

> **目的**：确定"一个图看到算子源码 → 编译过程 → 性能结果"的观测主体与主视图。
>
> **与 [ExecSpan 设计](./模型与算子执行观测主对象_ExecSpan设计.md) 的关系**：ExecSpan 是**执行侧**的观测对象，只覆盖全链的后三分之一。本文确定的 `Callable` 是**全链主体**，ExecSpan 降级为它的执行面与下钻视图。两者是主 + 次，不是替代。
>
> **状态**：设计稿。数据可得性已逐项核实，见第 5 节；缺口在第 6 节。

---

## 1. 为什么 ExecSpan 不能当全链主体

ExecSpan 的身份锚是"一次执行实例"，骨架是**时间轴**。但全链的前半段没有时间轴：

| 环节 | 有时间轴吗 | 它的本质 |
|---|---|---|
| 源码 scope | 没有 | 静态结构 |
| 编译决策（perf_hint） | 没有 | **属性**——不发生在任何时刻 |
| kernel 产物 | 没有 | 静态产物 |
| task / block 执行 | 有 | 运行时实例 |
| 性能结果 | 没有 | 执行实例的**聚合** |

把源码和编译决策塞进时间轴，只能做成"点选 span 弹出源码面板"——那是跳转，不是"一个图看到"。

---

## 2. 主体：`Callable`

**主体 = 一段被命名的计算，身份锚是 callable 名**（`q_proj`、`gate_proj`、`out_proj_aic`）。

选它的理由只有一条，但足够硬：

> **`taskToken` 每次运行都变，callable 名不变。它是唯一一个既跨 Run 稳定、又能落到每一层的锚。**

它在每一层的真实落点（均已核实，见第 5 节）：

```text
源码      pl.at(name_hint="q_proj")            decode_fwd.py:567
  ↓
编译决策   perf_hint（按源码位置挂载）          report/perf_hints.log
  ↓
kernel    kernels/aic/q_proj.cpp               文件名即 callable 名
  ↓
提交      deps.json.tasks[].kernel_ids         含 block_num、张量 shape
  ↓
执行      aicore_tasks[i][2] = registered_task_id
            → name_map.callable_id_to_name["7"] = "q_proj"
  ↓
性能      该 callable 全部 block 的耗时分布
```

### 对象骨架

```jsonc
{
  "id": "callable:q_proj",
  "name": "q_proj",
  "coreType": "AIC",                  // AIC | AIV | mixed
  "siblings": ["q_proj_0"],           // 编译器加后缀产生的同源变体

  "facets": {
    "source":   { "file": "decode_fwd.py", "line": 567, "scopeKind": "InCore" },
    "compile":  { "hints": [ /* 见下 */ ], "kernelFiles": ["kernels/aic/q_proj.cpp"] },
    "schedule": { "callableId": 7, "taskIds": ["..."], "blockNum": 48 },
    "exec":     { "spanIds": ["..."] },        // ← ExecSpan 在此接入
    "perf":     { "p50": 12.4, "p99": 31.2, "coreTimeShare": 0.061 }
  }
}
```

### 编译决策 `CompileHint`

图上"因"的一侧，结构直接来自 `perf_hints.log`：

```jsonc
{
  "code": "PH-MR-001",
  "category": "MemoryReuse",          // MemoryReuse | TileInnermostDimGranularity
  "site": { "file": "prefill_fwd.py", "line": 1240, "col": 21 },
  "decision": "pipeline depth 2 → 1",
  "cause": "only 1 of 2 buffers fit (32768 B per stage, 65536 B free)",
  "consequence": "stages 1 apart share storage and serialize",
  "advice": "relieve co-residents, or reduce pipeline depth to 1",
  "occurrences": 212
}
```

`consequence` 字段是关键——编译器**已经用自然语言说出了性能后果**，这正是归因边的文字依据，不需要我们推断。

---

## 3. 主视图：Callable 全链路归因图

**纵轴 = 变换阶段，不是时间。** 这是让源码和编译进图的唯一办法；代价是失去时序，时序留给下钻。

```text
① 源码 scope    [ q_proj @ decode_fwd.py:567 ]                 ← 收敛：1 个
② 编译决策      [MemoryReuse ×212] [Tile 粒度 ×104]      ─┐
③ kernel 产物   [q_proj.cpp · AIC]  [mixed 拆 AIC+AIV]     │ 归因边
④ task 提交     ▭ ▭ ▭ ▭ ▭ … 共 643                         │
⑤ block 实例    ▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮ × block_num                │
⑥ 性能分布      ├──▨▨▨▨▨──────┤ ←──────────────────────────┘ ← 发散：上千
```

三条设计规则：

1. **左收敛、右发散 = 因在左、果在右。** 用户的动作永远是"从右边的果，找左边的因"。
2. **两类边**：
   - **化身边**（纵向）：scope → kernel → task → block，表达 1:1 / 1:2 / 1:N 的基数关系；
   - **归因边**（斜跨，红色虚线）：某条 `CompileHint` 直连性能分布的右尾。
3. **归因边是这个图的灵魂。** 没有它，图只是一张血缘图。有了它，`PH-MR-001` 的 `consequence`（"stage 会串行化"）就直接解释了分布右尾为什么长。

### AIC/AIV 拆分在图上是可见的，不是推断

`kernels/` 目录里 `out_proj_aic` / `out_proj_aiv`、`qk_pv_online_phase_0_aic` / `_aiv` 成对出现。手册说的"mixed InCore 拆为 AIC + AIV 两个 kernel"在文件系统层面就是**实测事实**，第 ③ 行的分叉可以直接画，标 `measured`。

---

## 4. 下钻：ExecSpan 在哪里接入

| 用户问题 | 用哪个视图 |
|---|---|
| 这个 callable 为什么慢？ | **主图**（本文） |
| 这一次具体哪个 block 慢、卡在哪一段？ | **ExecSpan 时间轴**（上一份设计） |

从主图第 ⑤ 行点进去，就是 ExecSpan 的「资源 × 时间」泳道。ExecSpan 的 `identity.source.nameHint` 正好是本主体的 `id`，两者天然咬合。

---

## 5. 数据可得性（已逐项核实）

数据根目录：`Data/pypto_qwen3_profiles/pypto_qwen3_profiles/scratch/tp1_build_output/`

| 图上的层 | 数据来源 | 等级 |
|---|---|---|
| ③ kernel 产物、AIC/AIV 拆分 | `kernels/aic/*.cpp`（33 个）、`kernels/aiv/*.cpp`（44 个），文件名即 callable 名 | `measured` |
| ② 编译决策 | `report/perf_hints.log`：316 条，`MemoryReuse` 212 + `TileInnermostDimGranularity` 104，覆盖 70 个源码位置 | `measured` |
| ① 源码位置 | perf_hint 行尾的 `at <file>:<line>:<col>` | `measured` |
| ④ task、依赖、张量 | `dfx_outputs/deps.json`：`task_id` / `kernel_ids` / `block_num` / `args[]`（含 dtype、shape、strides、INOUT/INPUT） | `measured` |
| ④→① 身份桥 | `dfx_outputs/name_map_build_output.json`、`decode_records/name_map__jit_decode_fwd_*.json` 的 `callable_id_to_name` | `measured` |
| ⑤ block 执行 | `l2_swimlane_records.json` 的 `aicore_tasks`（见 ExecSpan 文档第 8 节） | `measured` |
| ⑥ 性能分布 | 由 ⑤ 按 callable 聚合 | `derived` |
| 数据依赖边 | `deps.json` 的 `args[].tensor_id` 按 INOUT→INPUT 匹配 | `derived`（比 ExecSpan 文档里标的 `inferred` 更好） |

> **对上一份设计的修正**：ExecSpan 文档第 8 节把 `dep` 边和张量信息标成了 `inferred`。`deps.json` 提供了 `tensor_id` 与完整 shape，这两项应升级为 `derived`。

---

## 6. 三个必须说清的缺口

1. **perf_hint 锚在源码行号，不在 callable 名——这是最大缺口。**
   要把 212 条 `MemoryReuse` 挂到 `q_proj` 上，需要"源码行 → scope 归属"的映射，现有数据里**没有现成的**。两条路：从 IR 侧补 dump（得到 `measured`），或按文件 + 行号区间近似（只能标 `inferred`）。**这个缺口不补，归因边就是猜的**，是首版最该先解决的事。

2. **右端画不下。** 643 个 task × block_num 是上千个实例。第 ⑤⑥ 行只能画分布，不能画实例。想看具体哪个 block 慢，必须下钻到 ExecSpan。

3. **`perf_hints.log` 不分 stage。** 它是整次编译的输出。好在源码文件名区分了（`decode_fwd.py` / `prefill_fwd.py` / `rms_lm_head.py`），按文件可以切开 prefill 与 decode。

---

## 7. MVP 边界

首版建议：

- 主体取 **decode 链路**的 callable（按 `decode_fwd.py` 过滤 perf_hint，28 个源码位置）；
- 六层全画，但 ⑤⑥ 只画分布；
- 归因边**只画能确证的那几条**——缺口 1 未解决前，宁可少画，并在每条边上标注证据等级；
- 点击第 ⑤ 行下钻到 ExecSpan 时间轴。

承接 ExecSpan 文档第 10 节的待确认项，仍需你定：Demo 落 `Design/`（PTO Design System）还是 `inference/`（shadcn 工作台风格，禁用 PTO DS 与 IDE Frame）。
