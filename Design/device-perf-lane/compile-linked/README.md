# 编译联动实验

入口：`../index-compile-linked.html`。主页面不依赖本实验。

直接导入 `Design/csa-pass-field/src` 的 `graph-components.jsx`、`CallableField.jsx`、`qk-graph-state.js` 与指令集合解析器；使用原图元、折线路由和阶段数据，不嵌入完整页面。Shadow DOM 隔离图样式，产物位于 dist，浏览时无需开发服务器。

## 关联契约

- 归档运行：`_jit_l3_decode_csa_20260903_010617`，与 Pass Field 数据的 source 路径一致。
- 当前 rank 下按 launch.leaf 精确关联 `csa_merge_pack_publish`、`qk_pv_aic`、`qk_pv_aiv`；按 taskId 缩小至选中提交。
- QK Op/指令使用原 instructionCollections 的 owner/kernelId/kernel。MatMul 只关联 AIC，exp/归约只关联 AIV；无法细化的值和控制结构显示整个 callable 关联。
- 指令位置来自 PTOAS 静态 occurrence 的 line/text；不推算指令执行时间，也不把重复源码数量作为运行次数。
- 点击泳道提交反向选中对应 kernel；图悬停为临时预览，点击保持选择。右侧事件按钮选中对应提交，不改变时间窗口。
- 无直接指令映射的节点不补造映射。阶段切换退出最终指令视图。

构建（仓库根目录）：
`rtk proxy node Design/csa-pass-field/node_modules/vite/bin/vite.js build --config Design/device-perf-lane/compile-linked/vite.config.mjs`

验证：真实指令数据的 MatMul/AIC、exp/AIV、merge/kernel 映射断言；浏览器检验两入口、阶段/指令切换、事件反向定位、关闭。

## 分屏交互（2026-10-27）

编译分析从底部独立入口开启，采用上下分屏，不覆盖泳道。分隔线支持拖动及方向键调整；紧凑行随可用高度适配，展开行使用泳道区域独立滚动。移除编译图右侧信息栏，保留图节点与泳道的 kernel 级联动。编译图目前覆盖 qk_pv 和 csa_merge_pack_publish，非泳道全部 callable。

## 单 callable 版本

界面固定为 qk_pv，作为标题；移除 callable selector。Pass rail 沿用原版 phase-group、pass-tick、noop-run 结构与样式。按当前 qk 数据 delta 是否非空确定默认显示阶段，零 delta 阶段可展开；这仅表示未呈现图形变化，不表示编译器没有执行该 pass。进入分屏默认关闭并行宽度与路径导航。

### Analysis interaction scenarios
Two independent synthetic compiler fixtures exercise analysis interaction without changing the trace: P25 splits a 32x128 Tensor along columns into 32x96 and 32x32, and P28 pairs C2V Push slot 0 with Pop slot 1 in the same iteration. Scenario nodes and before/after tables encode these static inconsistencies. Lane links identify owning kernels only; no runtime wait percentage or instruction timing is fabricated.
