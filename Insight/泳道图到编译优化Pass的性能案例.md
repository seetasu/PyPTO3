# 从泳道图追到编译优化：PyPTO 性能案例整理

> 整理日期：2026-10-17  
> 目的：筛选“泳道先暴露运行时性能现象，再通过 pass dump、编译提示或代码生成产物定位原因”的案例。Pass 可以按设计正常工作；重点是把运行时现象与编译过程建立证据链，不要求修改 pass。  
> 证据标注：**归因有证据**表示存在泳道/trace 与 pass 前后产物或编译输出的对应证据；**改动已验证**表示来源还记录了调整后的性能或正确性结果。

## 快速结论

按“泳道现象 → 检查 pass/编译产物 → 找到原因”的标准，已有 3 个可用案例：#1475（`MemoryReuse` 缓冲区复用引入 WAR 等待）、#1598（`OptimizeOrchTensors` 的保守 guard 阻止窗口外提）、#2040（检查 `InjectGMPipeBuffer` 和跨核同步，随后用最小复现修正初始归因）。前两个是 Pass 行为导致/暴露问题的直接例子；#2040 展示编译路径如何参与排查、并通过实验证伪初始假设。它们都不需要被包装成 pass bug。

改动验证另作一列：#2040 有 focused reproducer 的性能与 golden 结果；#1475/#1598 的公开记录主要证明定位链，未要求把它们说成已修复。

## 案例一：Qwen3-14B `fa_qks` 中 MemoryReuse 复用缓冲区，造成流水串行

**来源**：[pypto#1475](https://github.com/hw-native-sys/pypto/issues/1475)，状态：open；记录的 PyPTO commit `21f11ecb`，pypto-lib commit `5bee71a`。

### 现象与定位

Qwen3-14B fused decode attention 的 `fa_qks` 中，两个本可独立的 QK matmul 在 Cube（AIC）轨迹上串行执行；第二个 matmul 的 MTE2 K-load 等待第一个 matmul。Vector（AIV）侧也观察到类似停顿。

通过 `28_after_InitMemRef.py` 与 `29_after_MemoryReuse.py` 对照，发现两个 matmul 原本使用不同的 Mat 缓冲区；`MemoryReuse` pass 将它们合并复用。生成代码随后为共享缓冲区插入 `set_flag/wait_flag`，第二个 K-load 被第一个 matmul 的读取阻塞。归因是 pass 的合法生命周期复用引入跨流水阶段 WAR 依赖，牺牲重叠。

### 定位证据与验证状态

Issue 给出复现步骤与 IR、生成 C++、模拟器 trace 的证据。原因链已定位到 `MemoryReuse` 合并缓冲区后新增的跨 pipe 等待。归档状态仍为 **open**，但这不影响它作为“泳道现象如何追到 pass 行为”的案例；只需避免额外声称该 pass 已修复。

### 汇报时可讲

> 泳道/trace 看到独立 matmul 没有重叠后，我们把问题从设备流水下钻到编译产物；`MemoryReuse` 前后的 IR 显示两个输入缓冲区被合并，生成代码出现由复用引发的等待。问题定位到了具体 pass 和依赖，但目前公开记录没有最终修复结果。

**待补证**：泳道图原始截图/trace、修复 PR、修复前后同 shape 的多轮无观测扰动 benchmark、精度和内存占用结果。

## 案例二：Qwen3-14B q/k/v 投影在 OptimizeOrchTensors 后退化为单核串行

**来源**：[pypto#1598](https://github.com/hw-native-sys/pypto/issues/1598)，状态：open；关联修复 [pypto#1468](https://github.com/hw-native-sys/pypto/issues/1468)。

### 现象与定位

在 a2a3 真机、两层 `decode_fwd` 的 L2 泳道中，`q_proj` 的 20 个 tile task 全落在同一核、约 47 μs 一个且无重叠；`k_proj/v_proj` 也出现串行链。报告折算每层每个投影约 940 μs 的串行跨度。

检查 `14_after_OptimizeOrchTensors.py` 后发现，`OptimizeOrchTensors` 的 Pattern 5（`OutWindowExternalizer`）没有执行窗口外提：后续 `qk_norm` 对整个 q/k 输出的读取触发了 #1468 加入的保守 guard。窗口外提不执行，循环携带的 buffer 链没有被打断，导致并行循环不能在投影内部展开到多个核。Issue 分析认为该 full-parent read 在该编排顺序下实际有先后关系、并不构成竞态，因此 guard 可能过度保守。

### 修改与验证状态

#1598 提出了顺序感知 guard、按 Out 分别判定等修复方向；也记录了把 scope-1 改为 `pl.spmd` 的绕行方案。但该 Issue 仍为 **open**，没有证据表明 guard 已修改并在同一真机 case 上完成 A/B。故应称为**Pass 回归定位案例**，不能称为修复完成案例。

### 汇报时可讲

> 泳道图暴露 q/k/v tile 全部落在单核串行；pass dump 显示窗口外提没有发生；继续追到 `OptimizeOrchTensors` 的保守安全判断，发现一个后续整张量读取阻断了本可安全的并行化。根因定位清楚，但公开记录还缺少修复后的验证闭环。

**待补证**：guard 改动的 PR/commit；同一模型、设备、shape 下的 task 落核分布、投影耗时、整图 wall；golden 正确性及其他 alias 场景回归。

## 案例三：PTOAS 自动同步插入过于保守，GEMM 流水被打断（编译闭环对照）

**来源**：[PTOAS#643](https://github.com/hw-native-sys/PTOAS/issues/643)，状态：closed。

> 该来源有完整的编译原因分析和修改验证，但资料没有证明问题最初由泳道图发现；作为“编译优化导致性能问题并完成验证”的对照案例，不应包装成泳道起因。

### 现象与定位

启用 `ptoas --enable-insert-sync` 后，生成 GEMM 在多个 shape 上只有 torch_npu 的 0.37～0.41 倍。6144³ 上生成版本为 3.836 ms，手写 kernel 约 1.506 ms，torch_npu 为 1.408 ms。对照生成 C++ 与手写 kernel 后，发现内层 K 循环存在大量 `wait_flag`、`pipe_barrier`，包括 `PIPE_ALL`，切断 TLOAD、TEXTRACT 与 TMATMUL 的重叠。

### 修改与验证

先将 ping/pong 判定改为单个 if/else，避免跨分支依赖被同步外提；使用新版 PTOAS 后达到 torch_npu 的约 83%。随后把循环完全展开并使用 ping-pong buffer，消除控制流造成的同步外提，性能达到与手写实现持平。Issue 记录了维护者给出的修改方案和阶段性性能验证。

### 汇报时可讲

> 生成 GEMM 明显落后；通过对照自动生成代码，定位到同步插入切断流水重叠；调整 DSL 分支结构并展开循环后，性能从对标 0.37×提升至 83%，最终追平手写实现。它是完整的编译性能闭环，但来源未记载泳道图发现环节。

**待补证**：若要讲成泳道驱动故事，需找到原始泳道/性能时间线、同机基线、编译器版本和完整最终数值；现有 issue 摘要只支持代码生成对照与最终性能结论。

## 建议的案例组合

- **Pass 前后证据最清楚**：#1475，直接对比 `InitMemRef` 与 `MemoryReuse` 后 IR，再对应生成代码中的等待。
- **编排优化是否生效**：#1598，泳道串行现象与 `OptimizeOrchTensors` Pattern 5 guard 未触发相对应。
- **排查过程与实验闭环最好**：#2040，泳道发现 attention 差距后检查 `InjectGMPipeBuffer`/跨核边界，并通过最小 reproducer 修正原因判断、验证依赖距离优化。
- PTOAS#643 可作为补充的自动同步插入案例，但原始材料没有明确泳道是发现入口。

## 新发现：Qwen3 decode attention 的泳道归因被后续实验修正

**来源**：[pypto#2040](https://github.com/hw-native-sys/pypto/issues/2040)，2026-07-14 创建、2026-08-27 关闭；关联 pypto-lib#622、#765、#1031。

这是本次在线检索发现、可用来补充案例叙事的一手记录。它不是 DeepSeek-V4，也不是 pass 修复案例；价值在于展示如何用泳道和最小对照实验修正一个看似合理的编译归因。

### 现象与初始判断

Qwen3-14B decode 中，原生 `fa_fused` 的注意力耗时比 CANN FAI 路径高。Issue 初始假设是 A2/A3 上 `InjectGMPipeBuffer` 对 910B 关闭，Cube/Vector 边界经 GM 缓冲交接，融合没有消除 GM 流量。受控 TraCR/PMU 记录显示 CCE 约 548 µs、PyPTO 约 1263 µs；泳道复现命令使用 L2 swimlane。

### 后续证据如何修正根因

7 月 24 日的补充使用相同设备和输入做 decode 对照，golden 均通过。PyPTO attention 的 AIC/AIV 执行时间略短于 CCE，剩余全层差异集中在独立 RoPE/fusion 与调度边界。作者据此收窄最初的 GM 往返判断：两边都存在 GM payload transport 和跨核 flag，关键差别是 wait 消费的是哪一轮 producer。

最小 reproducer 只改变 producer/consumer 的依赖距离，其他 workspace 与 sync event 保持一致：same-step 100-round median 为 27.9 µs，explicit distance-2 为 18.8 µs，约 1.48×；两者 golden 均通过。这个结果支持“显式表达 skewed cross-core dependency 能减少等待”的判断，而不是“换掉 GM transport 就能优化”。

### 如何使用这个案例

- 可以讲成：泳道暴露注意力流水等待 → 提出 GM 往返假设 → 用同设备对照和最小 reproducer 缩小变量 → 将焦点改为依赖时序 → distance-2 实验提速且正确性通过。
- 不要讲成：已确认 `InjectGMPipeBuffer` pass 有缺陷并已修复。Issue 没有证明 pass 被修改；验证采用手工 `sync_set/sync_wait` 调度，属编程/调度表达的实验。
- 也不要把 27.9→18.8 µs 的小 reproducer 数据替代成整层收益。Issue 明确说明 swimlane 采集会扰动绝对时长，pair 内比较才适用。

### 证据边界

Issue 关闭状态为 completed，记录了真机 decode golden 对照、两个 focused reproducer 的计时与 golden 结果；但这不等于 PyPTO pass 修复。该案例仍符合当前目标，因为 pass/编译路径是定位链的一环，后续用实验确认了真正可控的性能原因。

## 资料索引

- 本地归档：`github_issues/pto/pypto_issues.md`，#1475、#1598 的 Issue 正文。
- 已有性能案例整理：`Insight/PyPTO性能调优典型用户案例_基于GitHub_20260923.md`，PTOAS#643 案例段。
- 在线补充：PyPTO#2040 的 Issue 正文及 2026-07-24 补充评论；在线状态核对日期：2026-10-17。
- 本文按本地归档截至 2026-10-17 的状态整理；GitHub issue 状态可能之后变化。
