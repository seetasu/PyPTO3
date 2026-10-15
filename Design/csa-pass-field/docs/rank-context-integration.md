# Rank 上下文与 qk_pv 编译联动

2026-10-15

## 官方依据

- https://www.pypto.ai/pypto-lib/pypto-coding/l3-programming/ ：Host 按 rank 分派 per-rank orchestration，device=r 在调用处绑定；rank 内沿用 L2 计算表达，跨卡通过 window、put/get、notify/wait。数据传输与信号依赖应分别解释。
- https://github.com/hw-native-sys/pypto/blob/main/docs/en/user/03-programming-model.md ：源程序、编译层次、runtime 调度是不同维度。

设计判断：rank 是计算图的部署上下文，不是某个 Pass 生成的新层次。相比两个平行 tab，计算图面板展开 rank 并联动共享编译画布更合理。官方文档支持语义边界，界面组织是本项目的设计判断。

## 实现

计算图面板增加展开/收起 rank，复用原始图节点、边、折叠变体与稳定 sourceId。两个 rank 的逻辑副本增加独立 rankId，三类集合通信显示双向静态连线。旧 deployment URL 自动打开 rank 面板，主画布继续显示当前 Pass。点击任一 rank 的 QK、Softmax、PV，关联主画布的 rank 上下文，不重置 Pass；所有 Pass 仍展示同一份共享编译证据。

这是按相同 per-rank 程序进行的静态 Mock 投影；尚无逐节点真实分片范围、任务实例、独立 rank 编译 dump 或完整信号协议。不能据此认定每个算子均实际执行两次、精确数据切分或完成时序。图中的通信线表示聚合通信关系，不是完整协议边。

## 验证

生产构建、数据校验、2 项计算图测试通过；所有折叠变体均检查 rank 节点唯一性、通信端点和 focus 源身份。此前 file:// 浏览器访问被策略拒绝，本次没有绕过限制，视觉验收尚未完成。
