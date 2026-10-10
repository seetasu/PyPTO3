# PyPTO Debug & Tune：真实 Issue 案例卡库（用于 UX Demo）

> 版本：2026-09-28  
> 用途：为「诊断 / 调优工作台」制作可走查、可追溯的 UX Demo。  
> 证据范围：`hw-native-sys/pypto-lib` 公开 Issue；Issue 状态经 GitHub API 于 2026-09-28 核验。  
> 重要边界：性能数值、设备配置和代码版本只说明当时案例，不能在 Demo 中表述为通用性能承诺。

---

## 1. 使用方式

每张案例卡都可单独成为一个 Demo 任务。产品不应只展示“报错”和“答案”，而应让用户完整经历：

```text
触发信号 → 选择正确证据 → 缩小问题范围 → 形成可验证假设
→ 单变量改动 / 运行方案 → 正确性与性能复验 → 沉淀适用条件
```

### 证据标签

| 标签 | 含义 | Demo 表达要求 |
| --- | --- | --- |
| `FACT` | Issue 中直接描述、日志或复现步骤可支持 | 可以用肯定陈述，必须保留来源链接 |
| `INFER` | 根据证据得到的合理诊断方向 | 以“可能”“待验证”表达，显示依据 |
| `MOCK` | 为补齐交互而构造的数据或状态 | 明确标注“模拟演示”，不得伪装为运行结果 |

### 建议的首版 Demo 主线

优先实现以下四张，分别覆盖编译、正确性、精度、性能四种入口：

1. `DT-01`：K-chunk matmul 编译失败；
2. `DT-03`：KV Cache 偶发错误（缺失 WAR 依赖）；
3. `DT-05`：长 decode 尾部质量劣化；
4. `DT-07`：完整 layer 中被掩盖的 attention 气泡。

其余案例可作为任务历史、案例库筛选和“继续调查”入口，避免首版承载过多互不相干的数据对象。

---

## 2. 案例索引

| ID | 用户看到的现象 | 已确认根因 / 结论 | 首要产物 | 来源 |
| --- | --- | --- | --- | --- |
| DT-01 | K 分块 matmul 无法编译 | AIC↔AIV PTO IR 的属性语法无法被 PTOAS 解析 | PTOAS 报错、`.pto` | [#32](https://github.com/hw-native-sys/pypto-lib/issues/32)（已关闭） |
| DT-02 | Prefill simulator 卡死 | task ring 满；同时有生成编排输出索引越界 | runtime log、scope/ring 数据 | [#122](https://github.com/hw-native-sys/pypto-lib/issues/122)（已关闭） |
| DT-03 | 同输入偶发错误，输出约 7% 随机失败 | 两个 reshape view 间缺少 WAR 依赖 | `deps.json`、生成 orchestration | [#481](https://github.com/hw-native-sys/pypto-lib/issues/481)（开放） |
| DT-04 | 结果多数元素仅差 1 ULP，却无法稳定回归 | `SplitMode.UP_DOWN` 改变 FP32 归约/窄化路径 | `error_distribution`、split A/B | [#362](https://github.com/hw-native-sys/pypto-lib/issues/362)（已关闭） |
| DT-05 | 长输出前段正常、尾部逐渐重复或乱码 | RoPE profile 路由错误；split-K 原子累加不确定 | 长度阶梯回归、逐层路由、重复运行 | [#951](https://github.com/hw-native-sys/pypto-lib/issues/951)（已关闭） |
| DT-06 | 固定输入通过，A→B→A 多轮执行失败 | persistent worker 复用下存在跨 epoch 状态污染 | 多轮压力回归、epoch 对比 | [#929](https://github.com/hw-native-sys/pypto-lib/issues/929)（已关闭） |
| DT-07 | 完整层看起来很满，单独 attention 却很慢 | 其他算子掩盖了 attention 的内在 memory / AIC↔AIV 串行气泡 | chip trace、standalone trace | [#607](https://github.com/hw-native-sys/pypto-lib/issues/607)（已关闭） |
| DT-08 | 想确认真实 NPU 上 task 是否完整且正确地执行 | golden 已通过；真实 AICPU/AICore trace 可用于后续调度分析 | Perfetto trace、验证结果 | [#590](https://github.com/hw-native-sys/pypto-lib/issues/590)（已关闭） |
| DT-09 | Decode 代码可运行但吞吐不理想 | GM 往返、过度拆分、tile / 融合选择仍有改进空间 | scope 清单、memory report、PMU | [#70](https://github.com/hw-native-sys/pypto-lib/issues/70)（已关闭） |

---

## 3. 案例卡

## DT-01：K-chunk Matmul 编译失败

- **真实来源**：[#32](https://github.com/hw-native-sys/pypto-lib/issues/32)，2026-03-23，已关闭。
- **角色与目标**：算子开发者把单次 full-K matmul 改为 K 分块累加，期望降低片上压力并形成标准的 `matmul + accumulate` 写法。
- **触发信号（FACT）**：AIC 与 AIV 生成的 `.pto` 都在 PTOAS parse 阶段失败；错误直接指向 `tpush_to_aiv` / `tpop_from_aic` 的属性语法。
- **误区**：把它当作 tile 参数问题，持续调整 `K_TILE`、N/M fragment；这不会改变解析器对 IR 语法的接受能力。

### 产品调查路径

1. 展示失败 phase、报错的 kernel 名、`.pto` 源码定位。
2. 提示“这是 PTOAS parse failure，不是 device runtime failure”。
3. 让用户选择“保留 PTO IR 并跳过 PTOAS”作为隔离动作（`skip_ptoas`）。
4. 对比 full-K 与 K-chunk 两份生成 IR，仅高亮跨 AIC/AIV 的传递 op。
5. 形成结论：应提交编译器/汇编器兼容性问题，而不是继续扫 tile。

### Demo 界面与状态

| 区域 | 内容 |
| --- | --- |
| Run 概览 | `Compile failed`、阶段 PTOAS、两个受影响 kernel |
| IR Inspector | 报错行、AIC/AIV 双栏 `.pto`、被标红的传递指令 |
| 决策卡 | `FACT：语法解析失败`；`INFER：生成器与 PTOAS 版本不兼容` |
| 下一步 | 导出最小复现、查看工具链版本、保留 `.pto` 作为附件 |

**成功标准**：用户能在 1 分钟内区分“源码/形状错误”和“生成 IR / 汇编器兼容性错误”。

---

## DT-02：Prefill 任务 Ring 满导致 simulator 失败

- **真实来源**：[#122](https://github.com/hw-native-sys/pypto-lib/issues/122)，2026-04-17，已关闭。
- **角色与目标**：模型开发者运行 Qwen3-32B prefill scope 的 simulator 验证。
- **触发信号（FACT）**：日志报告 `Task Allocator Deadlock - Task Ring Full`；131072 个 task slot 全部活跃，`last_task_alive=0`。随后还出现 `TaskOutputTensors::get_ref()` 输出索引越界。
- **关键结论（FACT）**：扩大 task window 只能推迟 deadlock；问题还涉及生成 orchestration 对 task output 的错误访问。

### 产品调查路径

1. 从 host 侧泛化错误跳入“运行时根因”页，而非只停在 ACL 错误码。
2. 读取 `orch_error_code`、task window、heap、dep-pool 的实时占用。
3. 标出第一个无法推进的 task / scope，并关联其生成 orchestration 行。
4. 分开呈现两个并存问题：ring 背压，以及 `get_ref` 输出索引越界。
5. 给出两类不同的行动：缩短 scope 生命周期 / 修复生成编排；禁止将“增大 ring”标成最终修复。

### Demo 界面与状态

| 区域 | 内容 |
| --- | --- |
| Health 面板 | Task window `100%`，Heap 仍有余量，危险状态为 `SCOPE / TASK WINDOW` |
| Scope 时间线 | scope 提交持续累积、没有可回收 watermark 的位置 |
| 证据抽屉 | 原始 fatal 日志、生成 C++ 的 `get_ref(index)` 源码行 |
| 行动卡 | `P0：定位 output index 越界`；`P1：检查 / 切分 scope`；`不建议：仅扩大容量` |

**成功标准**：用户理解“容量满”是现象，scope 生命周期或编排错误才可能是根因。

---

## DT-03：KV Cache 偶发错误——缺失 WAR 依赖

- **真实来源**：[#481](https://github.com/hw-native-sys/pypto-lib/issues/481)，2026-06-09，仍开放。
- **角色与目标**：模型开发者将 KV cache 改为原地写回，希望避免额外输出 buffer。
- **触发信号（FACT）**：多 token decode 下，`kv_cache` 检查通过，但 `x_out` 大约 7% 元素在不同运行中以不同索引失败。
- **已确认根因（FACT）**：gather 与 writeback 对同一外部 buffer 分别产生两个 reshape value；自动依赖按 value 跟踪，未插入 WAR 边，写回可能在读取前执行。
- **已确认实验（FACT）**：手动把生成 C++ 中的两处参数改成 `add_inout` 并经 `--runtime-dir` 重跑，结果通过。

### 产品调查路径

1. 把“同输入但不稳定”的信号自动归为 Race 候选，而不是直接推荐放宽 tolerance。
2. 在 `deps.json` 中同时显示逻辑 tensor、view 和底层 buffer 身份。
3. 高亮 gather → writeback 应存在、却不存在的 WAR 边。
4. 展示临时 workaround（no-op self-copy）及成本：共享 kernel 每次多了无意义 copy。
5. 创建更窄的候选改动：capture gather TaskId，writeback 使用 `deps=[gather_tid]`。
6. 用固定输入多次运行验证：正确性稳定性优先于平均耗时。

### Demo 界面与状态

| 区域 | 内容 |
| --- | --- |
| Symptom 卡 | “固定输入 / 非确定性失败 / 失败位置漂移” |
| Dependency Graph | 两个 view 指向同一 buffer；缺失边用虚线黄色显示 |
| 解释卡 | `FACT：RAW/WAW 会自动推断，WAR 不会` |
| 方案比较 | `self-copy workaround` vs `explicit deps`，分别展示正确性、额外 task/copy 与风险 |
| 实验结果 | 至少 20 次重复运行的 pass/fail 条带；没有真实数据时标记 `MOCK` |

**成功标准**：用户不会把“偶发正确性错误”错误归因成浮点误差，且能追溯一条显式依赖的必要性。

---

## DT-04：UP_DOWN Split 造成 1 ULP BF16 漂移

- **真实来源**：[#362](https://github.com/hw-native-sys/pypto-lib/issues/362)，2026-05-23，已关闭。
- **角色与目标**：算子开发者希望以 `SplitMode.UP_DOWN` 让合并的 matmul + dequant + SwiGLU scope 适配片上容量并提高吞吐。
- **触发信号（FACT）**：相同输入、相同代码结构下，仅切换 split mode，约 27.6% 输出元素出现不超过 1 ULP 的 BF16 差异。
- **诊断结论（INFER，Issue 明确为假设）**：上下半区拆分改变 FP32 归约或 dequant 的计算次序，随后窄化为 BF16 时放大为可见差异。

### 产品调查路径

1. 展示精度分布，而不是单个 allclose 红/绿灯。
2. 建立 A/B 版本树：`UP_DOWN`、`NONE`、拆为两个 scope。
3. 同时展示“内存是否适配”和“误差分布是否扩大”，避免只选最快的版本。
4. 给出可验证的 workaround：更小 fragment + `SplitMode.NONE`；将潜在性能损失标为待测。
5. 记录适用约束：含 matmul、FP32 dequant、最终 BF16 cast 的融合 scope 不能假定 split 是无损优化。

### Demo 界面与状态

| 区域 | 内容 |
| --- | --- |
| 精度面板 | rel-L2、cosine、ULP 分布和 bad-point 占比 |
| 版本对比 | Split mode、UB 占用、数值结果、性能状态 |
| 事实 / 推断 | 明确标出“归约重排是推断，不是已证明 compiler 缺陷” |
| 建议 | “先建立数值基线，再尝试结构性 split” |

**成功标准**：用户能识别“很小的每点误差”为什么仍会阻塞严格回归，并知道不能用性能理由掩盖数值变更。

---

## DT-05：长 Decode 尾部质量劣化

- **真实来源**：[#951](https://github.com/hw-native-sys/pypto-lib/issues/951)，2026-08-13，已关闭。
- **角色与目标**：服务/模型团队希望 DeepSeek-V4-Flash 的长输出保持稳定、可复现。
- **触发信号（FACT）**：48、96 token 输出正常，128 token 的后段逐渐重复或变成结构化乱码；请求没有 runtime error。
- **已拆分的根因（FACT）**：
  1. CSA/HCA 层错误使用基础 RoPE，而非压缩 RoPE，位置误差随 decode 位置累积；
  2. split-K 的并发 FP32 `AtomicAdd` 使请求或 rank 的执行顺序影响舍入结果。
- **关键验证（FACT）**：改为固定顺序 reduction 后，多次请求变为字节一致，但长序列尾部仍劣化；这证明“确定性问题”和“RoPE 路由问题”是两件事。

### 产品调查路径

1. 用 token-length 阶梯而不是单一 case 展示质量退化边界。
2. 从异常 token 段反查层级配置：SWA、CSA/HCA、MTP 分别应使用的 RoPE profile。
3. 并排展示两条假设链：profile 路由与非确定性 reduction。
4. 允许用户对每条链单独创建实验，并显示“修复了一部分现象，但未修复另一部分”。
5. 将最终验收定义为：长长度正确、跨重复/rank 一致、短长度不回归。

### Demo 界面与状态

| 区域 | 内容 |
| --- | --- |
| Quality Timeline | 48 / 96 / 128 token 阶梯，标记开始劣化的 token 区间 |
| Layer Inspector | 43 层 RoPE profile 路由矩阵，错误层高亮 |
| Determinism 面板 | 多次请求 hash / token diff；AtomicAdd 与固定序归约 A/B |
| 实验结论 | “字节一致：已改善；长尾质量：仍失败”——禁止合并成单一成功状态 |

**成功标准**：用户能够接受“一个用户症状可能包含两个独立根因”，并完成分而治之的验证闭环。

---

## DT-06：Persistent Worker 的跨轮输入污染

- **真实来源**：[#929](https://github.com/hw-native-sys/pypto-lib/issues/929)，2026-08-10，已关闭。
- **角色与目标**：分布式 MoE 开发者复用已 prepared 的 worker，降低多轮执行开销。
- **触发信号（FACT）**：固定输入连续 20 个 host round 全部通过；交替输入 A→B→A 时第三轮第一 epoch 失败，71.4783% 点超过相对误差阈值。
- **关键结论（FACT）**：即使每轮重置共享 window，问题仍出现；不能简单归因为 window 未清空。

### 产品调查路径

1. 在 Run History 中将“单次 pass”与“协议压力回归”分开呈现。
2. 自动生成最小状态机：worker prepared → A epoch → B epoch → A epoch。
3. 通过输入指纹、epoch、rank、窗口代次标识定位首次错误转移。
4. 建议两条对照实验：固定输入控制组；每轮 reset window 的诊断组。
5. 以跨轮正确性矩阵作为验收，而不是单个 golden 结果。

### Demo 界面与状态

| 区域 | 内容 |
| --- | --- |
| Run Matrix | host round × epoch 的 pass/fail 矩阵，A/B 输入用不同 pattern 标识 |
| State Inspector | worker / communication window 的生命周期与 generation（缺少真实采集时为 `MOCK`） |
| 对照组 | Fixed、Alternating、Reset-window 三列结果 |
| 建议 | “将多轮、变输入、跨 rank 的协议回归设为一等测试” |

**成功标准**：用户明白一次性 golden 通过并不等价于长驻执行协议正确。

---

## DT-07：完整 Layer 掩盖了 Attention 的真正瓶颈

- **真实来源**：[#607](https://github.com/hw-native-sys/pypto-lib/issues/607)，2026-06-24，已关闭。
- **角色与目标**：性能工程师评估 Qwen3-14B decode attention 是否值得优化。
- **触发信号（FACT）**：完整 decode layer 中约有 30 个 kernel，projection/MLP 持续占用 Cube；attention 自身的停顿被其他独立工作隐藏。
- **已确认结论（FACT）**：standalone paged attention 的瓶颈更清晰：KV cache 低算术强度、QK→softmax→SV 的 AIC/AIV 串行、decode 并行度不足。完整 layer 的稠密时间线不代表 attention 已被优化。

### 产品调查路径

1. 默认展示端到端视图，并标记“被 overlap 隐藏”的 scope。
2. 用户点选 attention 后打开 standalone 对照，不在同一尺度下直接比较总时长。
3. 列出三类可归因气泡：memory-bound、跨引擎 handoff、并行度不足。
4. 让用户选择优化目标：降低 isolated latency，或提升 end-to-end throughput；二者的测量口径不同。
5. 对候选改动要求同时保留 standalone 指标和整层/整网指标。

### Demo 界面与状态

| 区域 | 内容 |
| --- | --- |
| 双 Trace | Full layer 与 standalone attention 同步缩放、不同的时间边界说明 |
| Bottleneck Breakdown | HBM / AIC↔AIV / parallelism 三类证据 |
| 目标选择 | “单算子优化”或“端到端吞吐”；切换后改变推荐指标 |
| 防误判提示 | `FACT：被隐藏的等待不等于消失的等待` |

**成功标准**：用户不会因为完整 layer 看起来 busy 就否定 standalone 算子优化价值。

---

## DT-08：真机 Trace 的可信度核验

- **真实来源**：[#590](https://github.com/hw-native-sys/pypto-lib/issues/590)，2026-06-23，已关闭。
- **角色与目标**：性能工程师要确认采集到的 AICPU/AICore trace 对应真实、完整、数值正确的 CSA attention 执行。
- **触发信号（FACT）**：DeepSeek V4 CSA standalone 在 910B2 上通过 torch golden；TraCR 记录了调度与 task 执行事件，并导出 Perfetto 文件。
- **产品机会**：用户现在需要自行确认“trace 是否来自正确 workload、是否可解释”；这应成为采集后的强制检查，而不是专家经验。

### 产品调查路径

1. 将 build revision、设备、输入摘要、golden 状态与 trace 绑定为一个不可拆分的 Run manifest。
2. 在打开 Perfetto 风格视图前显示可信度检查：验证是否通过、task 数、采集通道、时钟域、缺失数据。
3. 若缺失输入/版本/校验，禁止把 trace 标记成“性能结论”。
4. 允许从一个 task 回到生成函数、原始验证和运行配置。

**成功标准**：Demo 的 trace 不只是漂亮的时间线，而是可追溯的诊断证据。

---

## DT-09：可运行的 Decode 仍存在结构性性能机会

- **真实来源**：[#70](https://github.com/hw-native-sys/pypto-lib/issues/70)，2026-04-07，已关闭。
- **角色与目标**：维护 Qwen3 decode layer，在不破坏正确性的前提下降低 GM 往返和调度开销。
- **观察（FACT）**：Issue 指出了多段 attention 之间的 GM 中间张量往返、QK/softmax/SV 拆为多个 `pl.incore()`、padding matmul 浪费、以及可用 `matmul_acc` / 融合的机会。
- **注意**：该 Issue 中部分建议是候选优化，不应在 Demo 中伪装成已测得收益。

### 产品调查路径

1. 从 scope 图找“写 GM 后马上被下一 task 读”的中间值。
2. 区分可以融合的相邻任务，与因容量、同步或可观测性必须保留边界的任务。
3. 联动 memory report：Mat / Acc / Vec 是否允许融合后的 tile。
4. 将“融合”创建为单变量候选实验，并默认要求数值验证、task 数变化和端到端测量。

**成功标准**：产品把“可能值得融合”表达为带约束的假设，而不是一键式、无条件的优化建议。

---

## 4. 未直接对应公开 Issue 的文档能力

下列 `debug-and-tune` 文档包含真实模型/实验背景，但本轮公开 Issue 检索未找到单独、直接对应的 Issue。若做 Demo，建议作为能力卡或由真实构建产物驱动，不能伪造为某个 Issue 的完整复现：

| 能力 | 建议 Demo 定位 | 需要的真实数据 |
| --- | --- | --- |
| L2 Prefetch | “缓存 warm 候选评审” | L2 working set、warm anchor、stage timing、端到端 A/B |
| CCE In-Core Profiling | “融合 kernel 内部阶段分解” | 每 core timing tensor、活跃 core mask、L2 task 边界 |
| Cube Tile Tuning | “约束驱动的 tile 搜索” | Mat/Acc/Vec report、候选 tile、PMU、重复 benchmark |
| In-Core Simulator Profiling | “单核流水验证” | 生成的 `.cpp`/`.pto`、代表性 control input、模拟器 trace |

---

## 5. 统一交互骨架

所有案例应复用同一套页面结构，差异放在数据与证据对象上：

```text
任务列表 / Case Library
  → 现象摘要（用户为什么来）
  → 证据工作区（日志、图、Trace、误差、内存等）
  → 调查步骤（事实 → 假设 → 可修改入口）
  → 实验定义（单变量、预检查、正确性、性能）
  → 结果与适用条件（成立 / 不成立 / 未知）
```

最低限度的真实性约束：

- 每个建议均能回链至少一条 `FACT` 证据；
- 没有 `ready/dispatch` 或资源快照时，不断言某个 task “被调度器卡住”；
- 没有重复测量时，不将一次 wall-time 差异称为性能收益；
- 没有真实候选 run 时，所有“优化后”图、数值和状态都标为 `MOCK`；
- 正确性、确定性和性能必须是分开的验收状态，不能用其中之一替代另两个。

