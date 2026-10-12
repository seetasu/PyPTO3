# Execution Triage · DSpark Serving

DSpark Serving 单请求（DeepSeek-V4-Flash w8a8，16 NPU，GBS1）的性能定界 demo。只加载这一份数据。

| 页面 | 作用 |
|---|---|
| `index.html` · 首页 | 单步流转：Serving / Host / Device 三块板与关键路径 |
| `analysis.html` · 详细分析 | 关键路径三段条、Step 节奏、单步瀑布、层详情 |

数据来源：`Data/serving-strace-swimlane(2).json`。Serving、Worker 和 16 条 Host STRACE lane 共用 host `CLOCK_MONOTONIC`。

## 首页

页面上只放数字、标签和图形，口径说明都记录在本文件中。

- **左上 · 每个 decode step 的时间去哪了（P1）**
  - 大字为 decode 步长均值 233 ms，附 P50 224 ms / P90 265 ms。
  - 下方三行构成：Serving 0.11 ms ≈ 0%、Host 108 ms 47%、Device ≤ 124 ms ≤ 53%，三者相加等于步长。点击某一行即选中对应那块板，右侧列表随之切换。
- **中间 · 单步流转（2.5D 泳道）**：一次只画一个 step，默认是步长最接近 P50 的 D47。
  - 三块板从上到下：Serving（调度器）、Host（Worker 一行 + 16 条 lane 的 bind）、Device（16 张卡的 runner_run）。时间水平向右，0 = 上一步结束；0 左侧较暗的部分是“上一步期间已经并行发生的事”。
  - **红色折线 = 关键路径**：上一步的同步屏障 → 关键 lane（多为 dev 6）开始 bind → 进入 runner_run → 16 张卡在同步屏障处一起结束，并标出 Host / Device 两段毫秒数。
  - **灰色虚线箭头 = 非关键流转**：调度器 `queue_worker_step` → Worker `dispatch_decode` → 第一条 lane；同步屏障 → Worker 回收 → 调度器 `process_step_output`。
  - 实线框是本步的工作，**虚框**是与本步重叠的下一步 prepare / dispatch，**虚线**是等待（`wait_worker_output`、reclaim、bind 结束到进入 runner_run 之间的空档）。
  - 悬停任意块显示名称与时长，例如 `dev 6 · bind 98.0 ms`；点击板或板名选择那一层。
- **底部 · 这次请求的服务指标（P0）**
  - 指标：E2E 12.57 s，首次输出 421 ms，输出间隔 P50 225 ms，输出 53 次。
  - 下方是请求时间线：每段一个 step，宽度等于耗时，标出“首次输出”；点击某一段即在中间显示那一步。
- **右侧**：选中那一层按 step 从后往前列出 2 张构成卡片，红色数值为关键项；每行下方有一条 16 格的分卡色带（Host 层越红越慢，device 6 那一格在几乎每一行都是红的）。点击卡片即在中间显示那一步，支持 ↑ / ↓ 切换。
- **右上 ↗**：打开详细分析页。
- 状态写在 URL hash 里（`#layer=…&step=…`），可以直接分享。
- 窄屏下泳道整体缩小，细节需要在桌面宽度查看。

## 口径

- **step 的边界**：每个 step 从上一步结束算到本步 16 条 lane 全部结束。16 条 lane 在 runner_run 末尾同步结束，所以最晚进入 runner_run 的 lane（关键 lane）决定一步的长度。
  - 它进入 runner_run 之前的时间计为 **Host**；step 之间只有 Worker 在工作的空档也计为 Host。
  - 它的 runner_run 计为 **Device**。没有采集 device_wall，runner_run 里含跨卡同步等待，所以是**上界**。
  - 只有调度器在推进的时间计为 **Serving**。调度本身约 0.47 ms/step，已被流水覆盖。
- **分卡数值**：Host 层每格 = 这条 lane 从本步起点到进入 runner_run 的时间；Device 层每格 = 这条 lane 的 runner_run（含等待，所以不着色）。
- **单步流转的归属**：本步的 dispatch = 在第一条 lane 开始前结束的最近一次 `dispatch_decode`，其 `step_id` 即本步；`step_id` 更大的 prepare / dispatch 视为“下一步准备”。prefill 与草稿初始化两步没有 dispatch，不画下发箭头。
- **请求指标**：
  - 请求是非流式的（`stream: false`，`max_tokens: 128`）。首次输出取服务端第一次 `scheduler.process_step_output`，不是客户端 TTFT。
  - 每次输出都记录为 `new_tokens=1`，每次实际输出的 token 数无法确认，所以给的是“输出间隔”，不是 TPOT。
  - 只统计落在请求窗口内的输出。流水预下发的 D53 在响应返回后才结束，不计入 E2E。
- **暂未放在首屏（P2）**：device 6 与次慢 lane 同时进入 runner_run 时，E2E 预计约 −1.88 s（−15%）。这条结论保留在详细分析页。

## 数据生成

```bash
node Design/execution-triage-workbench/build-data.cjs        # data.js：详细分析页
node Design/execution-triage-workbench/build-home-data.cjs   # home-data.js：首页（依赖 data.js）
```

两个脚本都只读取 `Data/serving-strace-swimlane(2).json`。
