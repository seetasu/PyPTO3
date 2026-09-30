# decode_csa_test 跨 rank 性能观测主体图

这是针对 `Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/` 的第一阶段数据模型：只建立可观测主体及其原始声明关系，不对慢点、等待原因、通信时延或关键路径做推断。

构建命令：

```powershell
node Design/distributed-observability-model/build-subjects.cjs
```

产物为 `subjects.json`。它有两个顶层集合：

- `subjects`：分布式 Run、设备组、设备、Rank、AICPU 调度器、任务调用、kernel 定义，以及从命名一致性识别出的通信序列。
- `relations`：包含、运行于、调用、参与通信序列，以及 `deps.json` 已声明的**同 rank**任务依赖。

本采集明确能观察到两张卡（device 1、3）和两套 DFX 输出；`allgather`、`all-to-all`、`reduce-scatter` 序列在两张卡上各有任务成员。原始文件没有 collective/communicator ID、消息字节数、链路拓扑、显式跨 rank 依赖边，也没有可直接证明两张卡时钟已对齐的字段。生成器会在 `coverage.unavailable` 保留这些缺口，避免将同名任务误写为已证实的传输或阻塞因果关系。
