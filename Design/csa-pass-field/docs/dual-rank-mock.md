# 双 rank 静态部署 Mock · 修订

2026-10-15 · 待视觉审阅

部署视图独立于 Pass 进度，不再复制各 Pass 的局部图。隐藏 Pass 导航及其键盘切换，保留返回编译视图。

两卡分别展示 token all-gather、Q/KV 准备、qk_pv、merge/pack/publish、O-group 接收、输出投影 reduce-scatter、HC post。骨架聚合了分支与细粒度依赖，实线仅表示阶段顺序，不是严格 task happens-before。

跨 rank 虚线展示三类通信的双向静态关系。依据 decode_csa.py 的 allgather 调用、tensor.put/notify、o_group_a2a 与 o_proj_reduce_scatter 调用。未展开 peer 分支、信号协议或迭代；此图不是动态执行 DAG，未声称测得每条传输。

点击阶段可查看源码位置；点击 qk_pv 的详情入口返回 Pass 23 编译结构。资源数量来自记录，具体物理设备、Task/Core 绑定和耗时未接入。

构建、数据验证、部署阶段独立性和通信端点测试通过。此前 file:// 预览被策略拒绝，视觉与交互验收未完成。

## 图元复用修订

阶段集合使用现有 Scope 组件，内部使用共享 OpNodeBase（ƒₓ）与 HandoffNodeBase（⇄），不再单独定义紫色或绿色阶段卡片。部署包装仅提供上下和跨 rank 端口，不改变编译视图端口。Q/KV 准备展示源码调用，publish 与 output 将计算和通信分开；这些是聚合操作摘要，不是完整算子展开。跨 rank 连线连接 all-gather、tensor.put、o_group_a2a 和 reduce-scatter 通信节点。双视图与资源清单维持原样。

定向检查通过：38 个唯一节点、14 个 Scope、6 条跨 rank 连线且所有端点均为通信节点；构建和数据校验通过。浏览器预览仍受此前 file:// 策略限制，未完成视觉验收。

图标修订：计算节点使用 vendor/pto-design-system/assets/fx.svg 的本地副本，替换文字字符徽章；共享计算和通信节点文字居中、类别标记靠右。部署端口复用透明 port 样式，保留连接锚点但不显示圆点。构建、数据校验及端点测试通过；视觉验收仍未完成。
