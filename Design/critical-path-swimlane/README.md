# Critical Path Swimlane

Rendered pattern · 基于 Execution Triage 的单步路径视图；Spatial Systems UI 呈现。

## 设计意图

开发者先判断时间消耗在哪里，再展开事件证据。默认将大量 task 收敛成 Serving、Host、Device 三层共享时间轴，保留 16 条 lane 的持续时间、起点差异、间隙和并行关系。文字仅标识层、稀疏行号和当前最晚进入路径；不逐条铺满 task 名称。此形式用于定界和提出调查方向，不替代原始 trace 的算子与因果验证。

## 来源链

固定版本和每个输入文件的 SHA256 在 `provenance.json`。

1. `PyPTO3/Data/serving-strace-swimlane(2).json`：DSpark GBS1 服务端采集；API / Scheduler / Worker 和 16 条 Simpler Host STRACE lane 使用同一 CLOCK_MONOTONIC。
2. `Design/execution-triage-workbench/build-data.cjs`：按 `args.invocation` 聚合；按 `tid` 对应 lane；从 `simpler_run`、`bind`、`runner_run` 事件提取窗口并确定 step。
3. `build-home-data.cjs → home-data.js → home.js`：转为相对上一步结束的毫秒、Worker 行与前后步骤上下文；原图由 JS 绘制 SVG，不能只提取 CSS。
4. `export-data.cjs`：读取这些产物并逐一回查原始事件；导出 54 个 step / 864 条 lane。额外从原始 runner_run 取 `rrEnd`，避免将外层 simpler_run 结束冒充 runner_run 结束。`end` 保留原外层结束语义。

重新提取：`node export-data.cjs /path/to/PyPTO3-checkout`。脚本只写本目录，不运行或改动来源仓库脚本。独立预览使用普通 script 加载数据，支持 file://，不依赖来源 checkout。

## 数据契约

`{version:1, run:{title,sub,workerRows}, steps:[{id,t0,t1,periodMs,end,lanes,serving,worker}]}`。

- 所有时间为毫秒；0 是上一步 16 条外层 simpler_run 全部结束。负数表示本步准备与上一步重叠。
- lane：`{dev,bindStart,bindEnd,rrStart,rrEnd,end}`。`dev` 在 step 内唯一；时间必须有限；各窗口起点不晚于终点。`end` 为外层 simpler_run 结束。
- Serving / Worker：`{n,s,d,kind,t?}`；`s` 是开始，`d` 为持续时间。Worker 的 `t` 是 prepare / dispatch / reclaim；未知角色须在上游适配，不猜测因果。
- 最晚进入 lane 按 `max(rrStart)` 重算。红色表示调查关注路径，不证明该 lane 是唯一因果瓶颈。step 结束取全 lane 外层 end 的最大值。
- 所有 lane 同一线性时间比例。bind 合计并行时长不可加到周期；Device 行仅表示 Host 记录的 runner_run 调用窗，不是 device_wall。
- 不导入原 demo 的 `deviceExec` / `waitEstimate` 公式拆分，也不导入分配 / H2D 比例。没有独立采集字段就不画执行与等待分段。

## 交互与复用

见 `pattern.html` 的嵌入示例。`CriticalPathSwimlane.mount(root,data,{step:'D09',syncHash:false})` 返回 `setStep(id)`、`getState()`、`destroy()`。`syncHash` 只应由独占 URL 的顶层页面启用。可重复 mount 不同容器；clip ID 独立，监听随 destroy 清理。

- Hover：预览 lane 配对与精确读数；离开后回到固定选择。背景只保留最晚进入路径。
- Click / Enter / Space：固定 lane，检查器可持续读取。选择为白色轮廓，关注路径同时保留红色。
- 检查器关闭保留 lane、step、缩放。检查器默认收起，打开为右侧悬浮面板；关闭不改变主体边界，窄屏使用底部浮层。运行信息、图例、快捷键与数据口径统一收进右上 info 按钮。
- ← / → 与文字结合的阶段导航：切换 step，重置至该 step 完整范围并保留 lane 身份。
- 滚轮或 ＋ / − 缩放；拖动空白平移；完整范围恢复所有关键端点。没有动画，减少动态效果设置不影响使用。
- 视图始终保留全部层；缩放与 Fit 独立悬浮于右下，与导航垂直对齐。

## 与来源差异及验证

保留原始三层、16 lane、单时间轴、上下配对、按需标签及最晚进入路径；改为中性灰、白色选择、橙红关注。去掉来源图中常驻的 16 条配对连线，仅强调当前 lane 与关键 lane；不复制整套业务工作台。修正默认范围裁切、bind/H2D 混称、等待混称和同步屏障的确定性断言。与来源是语义和交互提取，不要求像素一致。

`node verify.cjs` 验证全 step 数据约束、D09 读数、关键路径和范围边界。`export-data.cjs` 回查原始事件。浏览器交互、桌面 / 窄屏、视觉对比度尚待用户验收（遵循仓库约定，未运行浏览器自动化）。内容成熟度为 reviewed：2026-10-20 用户多轮截图评审后明确要求收录 Library / Pattern；不代表已完成全量浏览器回归。主库通过 catalog/assets.json 登记并由 scripts/build.py 生成。

## 视口与控件布局

画布填满宿主视口（默认 100dvh，最小 560px），SVG 按实际容器宽高重算几何，不以固定高度居中留空。顶部与底部预留悬浮操作安全区。字体保持屏幕像素大小，行间距随可用高度伸展。底部为紧凑操作栏与真实周期导航，右上为检查器与 info 按钮。复用库内 liquid-glass-controls 的 control / panel / icon 预设和 MIT 许可证。面板默认收起且互斥打开，不影响视角。浏览器视觉验收待完成。

阶段导航合并 selector：选中段固定加宽并显示英文名称，其他段悬停或键盘聚焦显示名称。未选中段之间保留周期比例，选中段宽度用于交互焦点，不能以其屏幕宽度比较耗时。

## All Steps Overview

总览交互更新：数据区 hover 同时定位当前 lane 行及 step 列；上下同号 lane 配对高亮。行头 hover 仅定位行，列头 hover 仅定位列。红色 `!` 行头表示至少一个 step 最晚进入 runner_run，不代表已确认硬件故障。点击行头或数据行保持总览并打开 lane 专项分析；只有点击列头或底部阶段导航进入单步。专项分析包含全 54 step 的 bind 均值排名、逐步中位均值对比、runner_run 调用窗均值、最晚进入次数、bind 趋势和阶段分布，不使用执行或等待估计。视觉及真实指针交互待用户验收。

右上 All Steps 切换完整请求泳道。按上一步外层结束累积绝对时间，共享轴保留流水重叠；每 step 展示 16 lane 的 bind 与 runner_run 窗口，红色为该步最晚进入 lane。Serving 只显示各步 own 事件，避免把前后上下文重复画入。悬停给 step 读数，点击或 Enter 进入单步；导航也可直接进入。全程支持缩放、平移和 Fit，hash view=all 可恢复。总览不拆分执行与等待。

视觉密度调整：三层标题 23px、右对齐并配结构图标；移除横向分隔线和 step 竖向网格，改为交替 step 底色。底色只编码 step 分区，不编码性能或异常。hover 行列高亮在分区底色上叠加。

### Continuous timeline

Overview and step detail share one absolute time axis. Wheel zoom is anchored to the pointer; dragging pans. Step navigation animates to a local range and Fit returns to the full run. Critical and hovered lane connectors fade in as each step gains screen width. Device rows retain Host-observed runner_run windows.

### Server outputs and slow steps

Serving diamonds/ticks come from scheduler.process_step_output start timestamps inside http.completions. Times share the request origin with lane events; these are server processing events, not client delivery or token counts. Navigation dots mark Decode periods strictly above the empirical P90 (sorted index floor(0.9*(n-1))); hover compares to the same-scope median. Prefill and DI are excluded from this baseline.

### Worker phase linkage

Raw Worker intervals are preserved without merging. Phase membership uses Worker step_id. Cross-layer candidates use STRACE group start after dispatch start and latest completion before reclaim end, checked one-to-one across all 53 Worker groups. These are temporal candidates, not causal flow IDs. Prefill and Draft Init share a batch containment context. D53 is retained as an unshown boundary group. Hover links both directions; clicking a Worker event pins its group and opens a shared-axis phase chart.
