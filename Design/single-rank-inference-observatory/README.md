# 单 Rank 模型推理观测 Demo

基于 `dsv4-flash-dspark-a3-dp4tp4ep16-gbs256-in256o64.zip` 的单 Rank 视角 / NPU 0 Trace 数据构建的交互原型。源数据没有给出逻辑 rank 编号，因此页面将其标为“观测 Rank”。

入口：`index.html`。

核心使用流程不是引导页：性能工程师从一个慢 decode 告警进入，选择请求 cohort 或问题切片，在时间线上点选证据，再把当前假设标记为待验证。
