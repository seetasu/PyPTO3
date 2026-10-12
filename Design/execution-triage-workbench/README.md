# Execution Triage · 执行定界工作台

面向昇腾平台上模型 / 算子的一次上板执行。打开页面后，开发者先看到这次执行的性能瓶颈落在 **Serving、Host 还是 Device**，再按需下钻。

| 页面 | 作用 |
|---|---|
| `index.html` · 首页 | 极简定界：三栋楼，楼高即耗时 |
| `analysis.html` · 详细分析 | 关键路径三段条、Step 节奏、单步瀑布和证据覆盖（仅 DSpark 与 decode_csa） |

## 首页：三栋楼

- **场景**：一次执行画成三栋等轴测线框楼——① Serving ② Host ③ Device。
  - **楼高 = 该层在关键路径上的耗时**，所有执行共用同一比例尺。瓶颈那栋楼默认选中并标红。
  - 没有采集的层只画虚线地基，标为“未采集”。
  - Device 是上界时，楼顶用虚线表示。
- **楼层**：选中那栋楼的每一层是一个 step（Qwen3 的 Device 楼是模型的 40 层，decode_csa 是每次调用），层高与该层耗时成正比。悬停显示 `D10 STEP` / `L17 LAYER` 标签，点击可选中。
- **左上**：层名、关键路径占比、耗时与楼层数；选中瓶颈层时给出一句话结论和假设收益。
- **右侧**：从顶层往下逐层列出构成卡片，每层 2 张；红色数值是关键项，灰色表示已被流水覆盖或可以忽略。与场景双向联动，支持 ↑ / ↓ 键切换楼层。
- **左下**：切换执行；**右下 i**：数据口径与来源。
- 状态写在 URL hash 里（`#run=…&layer=…&floor=…`），可以直接分享。

## 内置执行

| 执行 | 瓶颈 | 首页结论 | 来源 |
|---|---|---|---|
| DSpark Serving · 单请求 | Host | device 6 的 bind 拖住了 52/52 个 decode step | `Data/serving-strace-swimlane(2).json` |
| Qwen3-14B TP1 | Host | 设备 60% 时间在等 Host，每步约 45 ms 花在 KV 布局转换 | `Data/pypto_qwen3_profiles/…/tp1/` |
| Qwen3-14B TP2 | Device | decode 只用了 60 核中的 6 个，gate/up 占每层约一半 | `Data/pypto_qwen3_profiles/…/tp2/` |
| decode_csa · 单算子 | Host | 稳态调用里设备只忙约 3% | `Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617` |

## 口径

- **DSpark**：Serving、Worker 和 16 条 Host STRACE lane 共用 host 时钟。16 条 lane 在 runner_run 末尾同步结束，所以最晚进入 runner_run 的 lane 决定一步的长度。
  - 它进入 runner_run 之前的时间计为 Host。
  - 它的 runner_run 计为 Device。这部分包含同步等待，是**上界**。
- **Qwen3**：
  - Host = 每个 step 的设备空闲期（`step_trace_time.csv` 的 Free + Preparing），属于推导值。
  - Device = Computing。
  - KV 布局转换取自 `trace_view.json` 中 `compact_vllm_kv_for_contract` 与 `scatter_contract_kv_to_vllm` 两个函数。
  - Device 楼的 40 层来自 L2 泳道。泳道与 profiler 是两次独立进程，时间轴不对齐。
  - TP1 不在数据集验收范围内；TP2 只保留了 rank0，TP 通信在融合 kernel 内部，profiler 看不到。
- **decode_csa**：Device = `device_wall`（设备时钟），Host = `chip.run − device_wall`，两者只比较时长。

## 数据生成

```bash
node Design/execution-triage-workbench/build-data.cjs        # data.js：详细分析页（DSpark + decode_csa）
node Design/execution-triage-workbench/build-home-data.cjs   # home-data.js：首页四次执行
```

`build-home-data.cjs` 依赖 `data.js`，以及 `Design/operator-tuning-console/qwen3-profiles-data.js` 中已有的 Qwen3 解析结果（只读）；TP1 的逐步 Python 函数耗时直接从原始 `trace_view.json` 读取。
