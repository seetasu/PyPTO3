# PyPTO 编译 Pass 问题典型案例

> 用途：用于 PyPTO3 诊断与调优工作台的真实问题案例设计。  
> 筛选口径：只保留归档中能关联到具体 pass，或明确 pass 链的案例。  
> 证据边界：Issue 的硬件、版本和误差数据仅描述当时复现；Pass 链中标为“待验证”的部分不能作为已确认根因。

## 案例索引

| Case | 用户看到的问题 | 关联 Pass | 关键可视化对象 | Issue |
|---|---|---|---|---|
| CP-01 | 融合多输出 scope 后一路输出大面积错误 | `InitMemref`、`MemoryReuse`、`LegalizePtoBufferReuse`、`AllocateMemoryAddr` | 融合前后 IR、tile 生命周期、UB 地址与别名 | [pypto#1723](https://github.com/hw-native-sys/pypto/issues/1723) |
| CP-02 | `UP_DOWN` 切分下行归约结果错误 | `SplitVectorKernel` | split 前后 IR、sub-block 分片、partial sum 合并路径 | [pypto#1864](https://github.com/hw-native-sys/pypto/issues/1864) |

---

## CP-01：融合多输出 Scope 后单路输出错误

- **真实来源**：[pypto#1723](https://github.com/hw-native-sys/pypto/issues/1723)，已关闭。
- **涉及平台**：Issue 报告在 a2a3 上观察到。
- **问题对象**：多输出 `pl.spmd` scope 中的 per-head reduction 输出 `mix_x`。
- **证据状态**：多个 scale buffer 地址相同是已观察到的现象；强制让地址不同后结果仍错误，因此完整根因尚未确认。

### 1. 问题描述与可视化呈现

用户把 `mix_x` 与 `pre`、`post` 等输出放入同一个多输出 `pl.spmd` scope 后，`mix_x` 约 97.9% 的元素错误，而其他输出正确。将相同计算拆到独立 scope 后，`mix_x` 正确。

建议工作台呈现：

| 可视化区域 | 展示内容 |
|---|---|
| 结果概览 | 按输出分别展示 PASS/FAIL、错误元素比例、最大误差和容差；避免整体状态掩盖单个输出失败。 |
| Scope 图 | 并排显示融合前后的 scope 边界、输入输出、共享中间 tile 和消费者。点击 `mix_x` 可沿数据流回溯。 |
| Tile 生命周期 | 每个 per-head scale tile 显示创建、写入、读取和释放时间，以及作用域和消费者。 |
| UB 地址图 | 用不同颜色表示各 head 的 buffer 地址区间；地址重叠且生命周期重叠时标记冲突。 |
| 结果 A/B | 比较 fused 与 separate 两种实现的 IR、地址分配和逐输出误差分布。 |

### 2. 问题分析与 Pass 过程可视化

Issue 报告指出，融合场景中四个 per-head scale buffer 被分配到同一 UB 地址；单独 scope 中则具有不同地址。Issue 进一步指出，即使强制使用不同地址，输出仍错误。因此，地址复用是已观察到的问题线索，但不能单独解释全部错误。

需要重点检查并呈现的 pass 链：

```text
InitMemref
    → MemoryReuse
    → LegalizePtoBufferReuse
    → AllocateMemoryAddr
```

建议每个 pass 提供以下快照：

1. **IR 与 tile 状态**：scope 中的 tile、view、reshape、subview、transpose 和消费者关系。
2. **生命周期与别名**：buffer 的 live range、读写区间、别名对象，以及 pass 前后关系变化。
3. **物理地址变化**：pass 前后 UB 地址、大小、对齐和地址复用决策；定位多个 per-head tile 首次共址的阶段。
4. **语义与数值对照**：将每个 pass 快照与 `mix_x` 中间值或最终输出误差关联。

界面应把“观察到地址共用”标为 `FACT`，把“转置 tile 的内存复用或 view 处理导致剩余错误”标为 `待验证`。不应将某个 pass 显示为已确认的唯一根因。

### 3. 解决方案

- 检查内存复用的别名分析和生命周期判定，禁止仍有读写冲突的 per-head tile 共用地址。
- 检查 `transpose → reshape/subview → row_expand_mul` 链上的 tile layout、有效区域和地址计算是否一致。
- 对 fused 与 separate scope 做逐 pass 对照，继续定位“地址已分离但结果仍错”的剩余缺陷。
- 在根因完全确认前，将该类结构作为融合安全检查对象；无法证明安全时，禁止相关 tile 复用或回退为拆分 scope，并明确告知用户原因。
- 增加测试：逐 head 地址唯一性、buffer 生命周期不冲突、融合/拆分结果一致性，以及硬件数值回归。

### 面向产品的结论

诊断主线应是“单路输出失败 → scope 融合差异 → tile 生命周期与地址复用 → 逐 pass 追踪”。地址冲突是强线索，但该 Issue 的记录表明仍需排查其他融合路径缺陷。

---

## CP-02：UP_DOWN 切分下行归约产生错误数值

- **真实来源**：[pypto#1864](https://github.com/hw-native-sys/pypto/issues/1864)，已关闭。
- **问题对象**：带 `pl.row_sum` 的 `pl.split(SplitMode.UP_DOWN)` scope。
- **对照条件**：相同计算在 `SplitMode.NONE` 下正确，在 `UP_DOWN` 下错误。
- **证据状态**：错误与 split 路径关联明确；Issue 对缺少跨 sub-block partial sum 合并的描述属于根因分析，需结合 pass 输出验证。

### 1. 问题描述与可视化呈现

典型计算为逐行归一化：

```text
y[i, :] = x[i, :] / sum_j(x[i, j])
```

程序在 `NONE` 模式下结果正确，在 `UP_DOWN` 模式下编译运行成功但输出错误，没有编译期诊断。

建议工作台呈现：

| 可视化区域 | 展示内容 |
|---|---|
| 正确性 A/B | 对照 NONE 与 UP_DOWN 的 PASS/FAIL、最大绝对/相对误差、错误元素分布。 |
| Split 分片图 | 显示两个 AIV sub-block 各自负责的行区间、tile shape、输入分片和输出写回位置。 |
| Reduction 数据流 | 展示每个 `row_sum` 的归约轴、输入元素集合、各 sub-block 的 partial sum，以及合并和广播路径。 |
| 数学覆盖检查 | 把期望的整行归约范围与 split 后实际访问范围叠加，指出漏算、重复计算或未合并区域。 |
| IR 对照 | 并排查看 split 前后 IR，突出归约轴、shape、offset、sub-block id 和跨块通信/合并操作。 |

### 2. 问题分析与 Pass 过程可视化

归档将问题定位到 `pl.split(SplitMode.UP_DOWN)` 的切分处理路径，重点 pass 为 **`SplitVectorKernel`**。分析认为，切分把 reduction 的自由轴分给两个 AIV sub-block 后，可能没有生成正确的跨 sub-block 合并；另一种正确行为是，如果当前组合不受支持，就在编译期拒绝。

建议围绕 `SplitVectorKernel` 展示：

1. **Pass 输入**：原始 `row_sum` 的归约轴、tile shape、scope 属性和 split 模式。
2. **Pass 输出**：每个 sub-block 的 tile shape、数据偏移和归约轴变化。
3. **合并路径**：显示 partial sum 是否产生、由谁合并、结果是否广播到对应数据分片。
4. **合法性结果**：明确当前 reduction 与 split 方向的组合是“支持并正确处理”“不支持并拒绝”还是“未覆盖”。
5. **中间值对照**：比较 split 前后的归约输入、partial sum、最终 denominator 与输出，定位首次偏离位置。

推荐将“跨 sub-block 合并缺失”显示为待验证假设，直到 pass 输出或运行时中间值证据能够确认。

### 3. 解决方案

- 若支持 `UP_DOWN + row_sum`，在切分后正确生成 partial reduction、跨 sub-block 合并和结果广播。
- 若当前实现不支持该组合，增加 pass 级合法性检查并在编译期报错，指出 split 模式、归约轴和可选替代方式。
- 明确 reduction 类型、归约轴与 `UP_DOWN` / `LEFT_RIGHT` 等 split 方向的兼容规则，避免部分组合静默通过。
- 增加 `NONE` 与 `UP_DOWN` 的语义一致性测试，覆盖不同 tile 尺寸、归约轴、reshape 形式和边界情况。

### 面向产品的结论

诊断主线应是“同一计算仅 split 模式不同 → `SplitVectorKernel` 输入/输出变化 → sub-block 归约覆盖与合并检查 → 通过修复或编译期拒绝保证语义安全”。

---

## 统一的工作台交互建议

两个案例适合复用同一条编译诊断流程：

```text
用户症状
  → 选择失败的输出或算子
  → 查看关联 pass 及输入/输出快照
  → 对照 shape、layout、生命周期或分片语义
  → 验证根因假设
  → 应用修复/规避方式
  → 重跑正确性回归
```

每个结论应带证据状态：`FACT` 表示 Issue 或产物直接支持；`待验证` 表示需要通过 pass dump、运行时中间值或复现实验进一步确认。Pass 可视化应同时支持 IR、tile、内存/分片信息和数值结果，帮助用户把编译器内部变化与最终错误连接起来。
