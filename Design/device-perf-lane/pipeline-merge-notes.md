# 核内流水合并对照

入口：index-pipeline-merged.html。主版与归档版均未改动。

保留 focus 主页面、核内展开及编译分屏；单独加入“流水演变”对照入口，核内 Pass 分析同样进入连续演变。演变以 foreignObject 嵌入所选 Core 行下方，独占该行展开高度；不打开 dialog。返回核内事件时恢复原事件展开，重绘时清理播放 RAF。

静态核对：两版 phases、geometry、qk-pipeline-passes.js 相同；差异在画布渲染和播放控件。复用归档的稳定节点身份、连续几何插值、滑块、本阶段重播；沿用 focus 的当前数据与模型。

P19 是 Q 外提候选；P23 是 AIC/AIV 拆分；P25 显示原有 subblock 有效形状（不与编译分屏中的 96/32 合成异常混合）；P28 前移下一分片 QK，PV 保持对本分片 P 的依赖；P30 为内部 L0 展开。条带宽度与节点位置不用于表示设备计时。

更新：移除 foreignObject 演变画布。Pass 控件位于原重复 Q 提示条；进入时缩放到当前两个迭代的事件区间，在原 MTE/CUBE/AIV 执行引擎行上插值变换事件并更新依赖。结构变换不改写底层 trace 指标。

2026-10-28：经确认，合并副本提升为 index-focus.html；launch 高保真卡片指向该主入口。
