# PyPTO Distributed Run prototype

基于 `_jit_l3_decode_csa_20260903_010617` 的跨 Rank 观测界面原型。Run 总览突出同一 Task ID 的 Rank 本地执行跨度差异，并展示差异任务附近的真实设备 trace 过程、依赖数量和各通信序列实际包含的任务阶段；Rank 详情页可查看全部任务时间线、调度提交、声明依赖和采集产物。

## Rebuild data

```powershell
node Design/distributed-observability-model/build-subjects.cjs
node Design/distributed-observability-model/build-runtime-data.cjs
```

运行时详情数据生成在 `src/run-data.json`。所有时钟跨度按 Rank 分开，当前不作跨卡对齐；同名前缀通信 kernel 只代表各 Rank 均有相关任务记录，不代表已配对 collective。

## Run locally

```powershell
npm run dev
```
