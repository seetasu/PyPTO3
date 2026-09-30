# Qwen3 × LingQu 性能观测原型

基于 `Data/pypto_qwen3_profiles/pypto_qwen3_profiles/tp2/` 的本地交互原型，可从 Qwen3-14B 的 prefill/decode 下钻到 rank0、Device 0、40 层与 AICore task，并查看独立 Torch Profiler 的 operator、kernel 与 step 指标。profile 没有平台资产映射，因此 LingQu 层级用于标出需要接入的数据，不代表本次运行已确认发生在 LingQu 超节点。

## 构建数据与打开

```powershell
npm run data
```

启动开发预览：

```powershell
npm run dev
```

生产构建：`npm run build`。

原始数据保留在 `Data/`；派生数据写入本目录的 `data.json`。

## 数据覆盖边界

- prefill / decode L2 swimlane 各含 643 个 AICore task；根据采集脚本验证过的提交顺序还原为 embedding、40 层 × 16 个 task、LM head/tail。
- profiler 数据包含 Device 0 的 kernel、operator、step 与 trace 产物；Profiler 与 L2 swimlane 是独立进程，不能按时间戳逐事件匹配。
- 原采集 README 表明 TP=2 的两个 rank 均经过采集验收，但可视化产物保留 rank0。原始目录没有 rank1 的可视化 trace。
- 没有 Rank → 节点映射、机柜资产映射、plane/link 拓扑及 L1/L2 Telemetry。界面将这些标作“未采集”，不显示推测的性能值。
- Torch operator 与 L2 task swimlane 是两个独立进程产生的 profile；可以从 operator 定位到已采集的 Rank / Device 范围，不能可靠关联到具体 transformer layer 或 AICore task。
- 这是单次采集观察面板，不是集群告警系统；真实异常判断需要同期的设备状态、同步时钟和 LingQu 网络遥测。
