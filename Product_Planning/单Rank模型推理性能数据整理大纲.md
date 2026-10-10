# 单 Rank 模型推理性能数据整理大纲

> **用途**：将一次推理压测的请求级结果与单 Rank Runtime Trace 整理为可审阅、可比较、可下钻的性能分析材料。  
> **当前样本**：`dsv4-flash-dspark-a3-dp4tp4ep16-gbs256-in256o64.zip`，采集日期字段为 `2026-09-08`。  
> **范围边界**：当前只有一个进程 / NPU 0 视角的 Trace，未提供逻辑 rank ID、其他 rank Trace、网络与集群资源指标。本文所有“瓶颈”均指**单 Rank 证据或待验证假设**，不等同于整网归因。

---

## 1. Run 身份与可比性

### 1.1 Run 标识

| 项目 | 当前样本值 | 来源 | 可信度 / 说明 |
|---|---|---|---|
| 模型 | `DeepSeek-V4-Flash-0731-w8a8` | `result3.json.model_id` | 已记录 |
| 服务接口 | OpenAI-compatible | `endpoint_type` / `backend` | 已记录 |
| 并行配置 | `dp4tp4ep16` | 文件名 | 文件名推测，待用启动配置核验 |
| 批大小 | `gbs256` | 文件名 | 与请求数一致，但语义待核验 |
| 输入 / 输出配置 | 256 / 64 token | `input_lens` / `output_lens` | 已记录为目标长度 |
| 请求数 | 256 | `num_prompts` | 已记录 |
| 到达模型 | `request_rate=inf`、`burstiness=1.0` | `result3.json` | 全量 burst，不代表平稳流量 |
| 最大并发 | 256 | `max_concurrency` | 已记录 |
| Trace 范围 | Python 进程 + NPU 0 | Trace metadata | 单设备视角 |

### 1.2 对比前必须一致的条件

- 模型权重、tokenizer、量化精度与采样参数。
- 引擎、框架、CANN / runtime、驱动与算子包版本。
- 硬件型号、节点数、网络拓扑、DP / TP / PP / EP 配置。
- warmup 次数、图缓存状态、输入 / 输出长度分布与请求到达模型。
- 是否开启投机解码、draft 模型、draft 长度、KV Cache 与 batch 参数。
- Trace 采集级别及是否影响被测性能。

> 若以上条件不一致，结论应命名为“观察到的差异”，而不是“优化收益”或“性能回退”。

---

## 2. 数据资产清单

### 2.1 文件级结构

| 文件 | 解压后规模 | 内容 | 主要用途 |
|---|---:|---|---|
| `result3.json` | 425,092 B | 压测摘要、逐请求时延、流式间隔、生成文本、投机解码统计 | 服务体验、吞吐、请求分布分析 |
| `dsv4-...json` | 1,377,413,488 B | Chrome Trace / PyTorch Profiler / CANN 事件 | CPU、NPU、通信、等待、线程与算子下钻 |

### 2.2 请求与压测摘要字段

| 字段组 | 关键字段 | 可回答的问题 |
|---|---|---|
| 配置 | `model_id`、`num_prompts`、`request_rate`、`burstiness`、`max_concurrency` | 这是什么负载、能否与另一 Run 比较？ |
| 完成情况 | `duration`、`completed`、`failed`、`errors` | 是否成功完成、失败是否集中于某类请求？ |
| 吞吐 | `request_throughput`、`output_throughput`、`total_token_throughput`、`max_output_tokens_per_s` | 峰值与平均服务能力分别是多少？ |
| 请求形状 | `input_lens`、`output_lens` | 时延差异是否来自输入 / 输出形状？ |
| 首 token | `ttfts`、`mean_ttft_ms`、`median_ttft_ms`、`p99_ttft_ms` | 用户多久看到首次响应，长尾如何？ |
| 流式输出 | `itls`、`mean_itl_ms`、`median_itl_ms`、`p99_itl_ms`、`tpot` | 输出是否平稳，卡顿发生在哪个阶段？ |
| 生成文本 | `generated_texts` | 是否为空、截断、重复或异常长度？ |
| 投机解码 | `spec_decode_*` | 草稿开销是否换来了实际收益？ |

### 2.3 Trace 事件字段

| 事件层 | 已见类别 / 示例 | 可回答的问题 |
|---|---|---|
| 框架 CPU | `cpu_op`、`aten::*`、`vllm::*`、`npu_fx_compiler inference` | Host 是否处于关键路径，哪些算子或封装最重？ |
| Runtime | `enqueue`、`dequeue`、`HostToDevice`、`AscendCL@*` | Host-Device 派发、同步、拷贝是否异常？ |
| NPU 计算 | `Computing`、`AivKernel`、`Node@launch`、量化 matmul | 核函数、计算片段与 stream 是否饱和？ |
| MoE / 模型模块 | `vllm::moe_forward_shared`、`vllm::dsa_forward`、`MoeDistributeDispatchV2` | MoE、attention、矩阵乘分别贡献何种负担？ |
| 通信 | `Communication`、`Communication(Not Overlapped)`、all-gather / all-to-all、`gloo:all_reduce` | 集体通信是否长尾、是否和计算重叠？ |
| 同步 / 等待 | `EVENT_WAIT`、`NOTIFY_WAIT*`、`aclrtSynchronize*` | 依赖、流同步、队列反压发生在哪里？ |
| 元数据 | `process_name`、`thread_name`、NPU / stream labels | 事件属于哪个进程、线程、stream、设备？ |

---

## 3. 指标口径与基础分析

### 3.1 请求级体验指标

| 指标 | 建议计算 | 当前样本 | 解读边界 |
|---|---|---:|---|
| 成功率 | `completed / num_prompts` | 100%（256 / 256） | 不等于文本质量正确 |
| 请求吞吐 | `completed / duration` | 9.34 req/s | burst 全程平均值 |
| 输出吞吐 | `total_output_tokens / duration` | 597.55 tok/s | 需核对 token 是目标值还是实际 usage |
| 总 token 吞吐 | `(input + output tokens) / duration` | 2987.73 tok/s | 适合容量量级比较 |
| TTFT | `ttfts` 分位数 | P50 2.59 s，P99 3.43 s | 含服务端排队、prefill、首包链路等成分 |
| 首 token 后完成时间 | `sum(itls)` | P50 24.36 s | 仅按现有间隔数组计算 |
| 端到端时间 | `ttft + sum(itls)` | P50 26.84 s，P99 27.38 s | 不是服务端完整日志时间线 |

### 3.2 流式与 Decode 指标

- 全部观测间隔：14,240 条；均值 432.94 ms，P50 134.57 ms，P99 874.83 ms。
- 前 24 个记录位置：6,144 条，均值 783.11 ms。
- 第 28 个记录位置以后：7,336 条，均值 121.14 ms。
- 单请求 `itls` 长度为 24–63，未稳定等于目标输出 64。

**可分析维度**：

1. 慢输出是否集中于统一阶段、少数请求、特定 batch，还是特定流式 chunk。
2. 早入场 / 晚入场 cohort 的 TTFT、端到端时间与完成顺序是否存在系统性差异。
3. TTFT 与 decode 是否为同一瓶颈，还是应分别归因到 admission / prefill 与 decode / 通信。
4. `itls` 是 token 间隔、chunk 间隔还是客户端观测间隔；该语义必须在解释 TPOT 前核验。

### 3.3 投机解码指标

| 指标 | 当前样本 | 建议分析 |
|---|---:|---|
| Draft token 数 | 99,680 | 草稿侧计算规模 |
| Accepted token 数 | 1,990 | 实际被主模型接受的 token |
| 重算接受率 | 1.996% | `accepted / draft × 100%`，与结果字段一致 |
| Acceptance length | 1.14 | 需结合引擎口径理解 |
| 位置接受率 | 7 个位置字段 | 观察草稿窗口位置衰减 |

**推荐 A/B**：保持并发、输入/输出形状和 warmup 相同，比较 speculative on/off、draft 长度、draft model、采样策略，并同时报告 TTFT、TPOT、吞吐和实际 usage。

---

## 4. Trace 分析大纲

### 4.1 时间线与阶段识别

1. 标记 profiler enable / disable，排除采集启动和收尾时间。
2. 确定与请求压测时长对应的活跃窗口。
3. 将活跃窗口按 prefill、decode、尾部 draining、同步 / 清理分段。
4. 对齐请求 cohort 的 TTFT / 完成分布与 Trace 中的关键事件。

当前样本中，主要模型活动约位于 profiler 相对时间 19.1–48.1 s；完整 Trace 跨度约 74.2 s，不能误读为 74.2 s 的持续模型推理。

### 4.2 Host / CPU 关键路径

| 观察对象 | 当前样本证据 | 下一步问题 |
|---|---|---|
| `npu_fx_compiler inference` | 29 次，累计 19.49 s，单次 P50 665.5 ms，最大 994.3 ms | 是图编译、图执行封装，还是动态 shape / cache miss？ |
| `vllm::moe_forward_shared` | 1,535 次，累计 14.32 s | 是 CPU 包装时间还是设备执行等待？ |
| `vllm::dsa_forward` | 1,535 次，累计 6.30 s | 与 attention / decode 阶段如何对应？ |
| `aten::item` / scalar 读取 | 高调用与累计时间 | 是否因频繁同步破坏流水线？ |

**注意**：Trace 的 inclusive duration 可嵌套，不能把多个算子的累计时间直接相加为端到端耗时。

### 4.3 NPU 计算、通信与重叠

| 分类 | 当前样本累计值 | 可回答的问题 |
|---|---:|---|
| `Computing` | 9.40 s | NPU 计算片段是否连续、是否存在空泡？ |
| `Communication` | 12.41 s | 通信总体规模和分布如何？ |
| `Communication(Not Overlapped)` | 12.31 s | 通信是否缺少与计算的重叠？ |
| `gloo:all_reduce` | 192 次，累计 1.26 s | collective 是否存在长尾？ |

当前 Trace 的 profiler overlap 分类中，`Communication(Not Overlapped)` 接近全部 `Communication`。这是一个优先验证的**单 Rank 现象**：应补充全 rank 对齐 Trace、通信 group / 拓扑与网络计数器，才能判断是否为系统级通信瓶颈。

### 4.4 等待与依赖

- 对 `EVENT_WAIT`、`NOTIFY_WAIT*`、`aclrtSynchronize*` 按 stream、时间窗、前驱事件聚合。
- 区分 Host 等待 Device、Device 等待通信、stream 内依赖与队列反压。
- 将最长等待事件关联到同一时间窗的 compute / communication 事件。
- 不将跨 stream 的 inclusive wait 累计值直接解释为设备空闲百分比。

---

## 5. 推荐分析切片

### 5.1 按请求 / cohort 切片

- 最快、P50、P95、P99 请求。
- 早提交与晚提交 cohort。
- 完成时间 `<22 s`、`22–25 s`、`25–26.5 s`、`26.5–27 s`、`27–28 s`。
- 文本为空、长度异常、错误或提前结束的请求。

### 5.2 按模型阶段切片

- Admission / 排队。
- Prefill / TTFT。
- Decode 早期慢速段。
- Decode 稳定段。
- 尾部 draining 与资源回收。

### 5.3 按执行资源切片

- Python 主线程与其他 CPU 线程。
- NPU stream、通信 plane、CANN runtime。
- 计算、通信、同步、HostToDevice、enqueue / dequeue。
- MoE dispatch / expert compute / combine，attention，量化 matmul。

### 5.4 按实验变量切片

- 并发：1 / 8 / 32 / 64 / 128 / 256。
- 到达模型：burst、固定速率、泊松与真实流量回放。
- 输入 / 输出 token 长度：短 prompt、长上下文、短输出、长 decode。
- 并行策略、batch 策略、量化、KV Cache、投机解码。
- 冷态 / 热态图缓存与不同 profiling 级别。

---

## 6. 标准输出物模板

### 6.1 Run 摘要卡

- Run ID、采集时间、模型 / 权重 / tokenizer、引擎 / runtime、硬件与并行配置。
- 负载：请求数、并发、到达模型、输入 / 输出分布、warmup。
- 结果：成功率、QPS、输出 TPS、TTFT P50 / P95 / P99、TPOT / ITL、端到端 P50 / P95 / P99。
- 结论状态：已证实、待验证、无法判断。

### 6.2 体验分布页

- TTFT 直方图与端到端完成时间分布。
- 请求 cohort 的提交、首 token、完成时间带。
- 流式间隔热图或位置分布。
- 错误、空输出、截断、文本长度异常统计。

### 6.3 单 Rank Trace 页

- 进程 / NPU / stream 范围与时间基准。
- 关键路径泳道：Host graph、compute、communication、wait。
- 可点选事件：名称、持续时间、线程、前驱 / 后继、原始 args。
- 明确标注“Rank-local evidence”，避免替代多 rank 对比。

### 6.4 归因与验证页

| 假设 | 证据 | 尚缺数据 | 最小验证实验 | 预期指标 |
|---|---|---|---|---|
| 图缓存 / 动态 shape 影响 decode | 主线程 graph 路径长且重复 | graph cache hit/miss、shape 序列 | 冷 / 热态重跑，记录 shape | TTFT、早期 ITL、CPU critical path |
| 通信缺少重叠 | 单 Rank non-overlapped communication 高 | 全 rank Trace、通信拓扑、NIC 指标 | 改 overlap / batch 参数 | communication overlap、TPOT |
| 投机解码收益低 | 接受率 1.996% | draft 配置、实际 token usage | speculative on/off A/B | TPOT、吞吐、成本 |

---

## 7. 当前数据不能得出的结论

- 不能确认该 Trace 属于逻辑 `rank 0`；仅能确认它是 NPU 0 的单进程 / 单设备视角。
- 不能判断哪个 rank 最慢，或确认整个 DP / TP / EP 作业存在负载不均。
- 不能以单 Rank 的通信事件直接证明网络、拓扑或某个 collective 是全局根因。
- 不能确认 `output_lens=64` 等于每条请求的实际模型输出 token 数。
- 不能把 Trace 的 inclusive 时间、跨 stream wait 时间简单相加为墙钟时间或空闲率。
- 不能从非空且不重复的 `generated_texts` 推断模型回答质量、正确性或安全性。

---

## 8. 下一轮采集契约

### P0：建立可比较基线

- 每个 Run 保存完整 manifest：版本、硬件、并行、启动参数、环境变量、采集开关。
- 保存实际服务端 usage：输入 / 输出 token、TTFT、完成时间、错误码、请求 ID、batch ID。
- 拆分 cold / warm 两类 Run，并记录 warmup / graph cache 状态。
- 至少输出 burst 与一个固定速率负载。

### P1：补齐全局归因

- 每个 rank 采集对齐 Trace，保留 rank、pid、tid、device、node 与时间基准。
- 同步采集通信 group、collective、网络带宽 / 拥塞、NPU 利用率、显存与 KV Cache 指标。
- 建立请求 ID → batch → rank / stream → Trace event 的关联键。

### P2：形成优化闭环

- 每项优化只改变一个变量，并保留基线 / 候选 Trace。
- 同时验收性能、错误、实际输出 token 与任务质量。
- 输出相对基线的置信区间或重复 Run 波动，而非单次绝对差值。
- 将通过验证的配置、适用负载与副作用沉淀为可检索的实验记录。

---

## 9. 使用检查表

- [ ] Run 身份、负载与环境已记录，可判断是否可比。
- [ ] 请求级指标与 Trace 时间基准已分别说明。
- [ ] TTFT、decode、端到端时延均报告分位数，而非只有均值。
- [ ] 流式间隔的 token / chunk 语义已核验。
- [ ] Trace 的累计耗时已按 inclusive / overlap 口径解释。
- [ ] 单 Rank 结论没有外推为整网结论。
- [ ] 每个“瓶颈”都附有原始证据、缺失数据和最小验证实验。
- [ ] 优化结果已与可比基线、正确性和实际输出 usage 一起复测。
