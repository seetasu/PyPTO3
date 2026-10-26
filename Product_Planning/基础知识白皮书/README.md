# PyPTO 基础知识白皮书

本组白皮书面向第一次接触 PyPTO、模型推理优化、算子开发与硬件原生编程的工程师和产品/工具团队。内容以仓库内研究、规划、源码镜像和术语材料为依据；涉及产品规划的内容会标为设计建议，不代表已经交付的能力。

## 阅读顺序

[打开交互式白皮书](./index.html)

1. [PyPTO 基础：从 Python DSL 到设备执行](./PyPTO基础_从PythonDSL到设备执行.md)
2. [模型推理基础：从请求阶段到性能指标](./模型推理基础_从请求阶段到性能指标.md)
3. [算子开发基础：从数学规格到可验证 Kernel](./算子开发基础_从数学规格到可验证Kernel.md)
4. [Hardware-Native System 基础：让硬件事实进入编程模型](./Hardware-Native基础_让硬件事实进入编程模型.md)
5. [PyPTO Pass 流水线详解：从 Tensor 到设备任务](./PyPTO-Pass流水线详解_从Tensor到设备任务.md)

## 使用说明

- 文中“事实/机制”用于解释仓库材料描述的概念；“建议/检查项”属于工程实践总结。
- PyPTO 的具体 API、支持范围和编译行为可能随版本变化。实施时应对照目标版本的官方文档、源码和测试。
- 性能结论必须绑定设备、软件版本、shape、dtype、输入与计时口径；示意流程不构成性能承诺。
- 交互页面由同目录五份 Markdown 生成数据；正文有调整后，在该目录运行 `node build-reader-data.cjs` 同步页面内容。

## 仓库内主要依据

- `Product_Planning/PTO3_源码_调度_硬件执行概念术语手册.md`
- `Insight/PTO3写算子过程中的硬件内存感知与配置.md`
- `Insight/PyPTO性能调优方法论_基于Toolkit与案例.md`
- `Product_Planning/大模型推理优化全旅程.md`
- `Product_Planning/单Rank模型推理性能数据整理大纲.md`
- `Product_Planning/模型与算子全链路观测主体_Callable设计.md`
- `Product_Planning/人-Agent-Toolkit协作模式_算子开发流程.md`
- `Product_Planning/PyPTO3算子开发可视化基础作业对象.md`
- `Product_Planning/PyPTO_Hardware-Native内容维度与清单_20261019.md` 与配套九域详解
- `repo/pypto/python/pypto/ir/pass_manager.py` 与 `repo/pypto/docs/zh/dev/passes/`

