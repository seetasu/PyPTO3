# Execution Triage V2 · GBS256 压测

V2 保留原单请求 Demo，并基于 `dsv4-flash-dspark-a3-dp4tp4ep16-gbs256-in256o64.zip` 展示一轮批量推理压测与对应的 PyTorch profiler 执行跟踪。

- `result3.json`：256 个请求，256 并发上限，输入 256 tokens、输出 64 tokens；汇总吞吐及 TTFT / TPOT / ITL 数组。
- `trace-summary.json`：对压缩包内 1.38 GB profiler JSON 流式扫描后生成的事件汇总（6,458,204 条事件），页面展示累计耗时靠前的事件。
- `result-data.js` 与 `trace-data.js`：静态页面读取的数据副本；不需要浏览器直接加载大体积 trace。
- `build-data.ps1`：从 ZIP 重新提取基准摘要并重建 trace 汇总和页面数据。

## 数据口径

- benchmark 持续 27.42 秒，256/256 请求完成，失败数为 0；请求速率设置为 `inf`，表示客户端突发提交。
- request throughput 是每秒完成请求数；output throughput 是生成 token 吞吐；总 token throughput 同时计入输入与输出。
- TTFT / TPOT 汇总字段以毫秒计。原始 `ttfts` / `itls` 数组以秒计；页面转换成毫秒绘图。
- profiler trace 的时间跨度约 74.2 秒，和 benchmark 27.42 秒不是同一个时间口径。事件持续时间互相重叠，累计时长仅用于定位高频/耗时事件，不能相加成墙钟时间，也不能直接当成设备利用率。
- 原始 trace 留在 `Data/` 压缩包中；此 Demo 只保留可读的基准摘要和压缩后的事件聚合。

重新生成：

```powershell
powershell -ExecutionPolicy Bypass -File Design/execution-triage-workbench-v2/build-data.ps1
```
