# PyPTO Agent Decision Explorer

直接打开 agent-decision-lineage-softmax.html 即可，无构建依赖。数据通过 lineage-data.js 加载，lineage-data.json 为同一模型的可读版本。

## 工程与模型

- HTML / page.css / app.js：三栏基础框架、阶段和类型筛选、节点详情与证据链接。
- nodes：id、type、title、status、stage、reason、relatedNodeIds、evidenceRefs、assertionBasis。
- edges：source、target、type；当前仅 expresses relates_to，不推断时间顺序或因果。
- sources：文件 ID、快照路径、SHA256；evidenceRefs.location 记录章节或 JSON 字段。
- evidence/：10 份原始材料快照，不改原始材料。
- assets/design-system/：从 V3 同源设计系统复制 tokens、CSS、ide-frame 与 workbench-shell。

状态区分 completed / recorded / adopted / considered_not_adopted / attempted_failed / not_attempted / validated。采用与验证是不同概念，验证结果通过证据节点和引用表达。列表数量只是本页人工整理节点数量，不代表 Agent 实验或候选总数。

## V3 复用审计

原 V3 为单文件 HTML，业务 CSS/JS 内联；外链包含 IDE Frame、workbench shell、AIC/AIV、memory/hardware、矩阵/张量、播放条和 candidate-population-lineage。当前仅复用基础 shell 依赖；不移植 MatMul、Cube、播放条或候选图。

映射：V3 topbar → ide-frame topbar；pane/header/body/status → 同名共享类；业务控件 → btn btn-ghost；旧业务画布 → 节点索引。未创建新视觉组件或 preview page。直接嵌入 ide-frame，三栏内容与深色主题依据原 V3 和本阶段需求推断。页面不覆写 pane 背景与阴影，无私有装饰边框。

## 信息边界

无法可靠恢复完整 Agent 对话、每次思考时间、完整候选总数、实验总次数和未记录失败。stage_retry_count=0 不代表零实验。Stage 3 约束分析不是编译记录。KB_SELECTION、DESIGN_BINDINGS、KB_USAGE 为知识选择/绑定/使用材料，不是执行候选账本。

Stage 5 基线与 Final 正式采集、quick A/B 和 Golden 的范围不能混合。performance.json 的 repeat_bound_diagnoses 是路由诊断；terminal_evidence 才是报告停止证据。原始报告判定 compute_bound，小 case 仍有启动开销主导的限定。timeline preflight 失败不等于流水线失败。

未运行 NPU 测试，也未新增测试。未实现 Lineage 图或性能图表。

## Agent Decision Lineage 主交互

中央改为 SVG 节点群与曲线连接，复制 V3 的 candidate-population-lineage CSS 作为视觉基础；不使用其候选数量驱动的 JS。默认显示少量关键决策、失败与未尝试分支；点击阶段展开全部该阶段业务节点，点击节点高亮上游/下游语义关系，空白和「返回全局」重置。支持 Tab/Enter/Space 操作节点。

新增知识选择、API 适配、设计约束、Precision、Tiling、Memory、Scheduling、Kernel、正确性和 Final 节点，均引用既有材料。关系有独立 evidenceRefs；workflow、stage_membership 和实际决策关系分开。原始 evidence 快照未改动。

浏览器检查受工具 file:// 协议安全策略阻止，未绕过策略；尚未实际确认布局和点击表现。未新增或运行测试。

## 阶段 2 信息架构重构（2026-10-31，取代前述三栏/阶段发散布局）

- Workflow-first 改为 Decision-first + Experiment-first；五阶段成为轻量标尺，不再绘制阶段归属边。
- 默认全宽图谱：SPEC → 知识/API/硬件约束 → 采用设计节点群 → Kernel → Correctness → Baseline → Mask Lifting → Final。
- 保留 ide-frame direct embedding 与原深色主题；复用 V3 连续带、局部节点群和分流视觉原则。带宽固定，不表示数量。没有引入硬件视图或性能 Dashboard。
- `lineage-layout.js` 为语义展示投影：groups/items、branches/parent/relation、links/source citation。原 `lineage-data.json/js` 与 evidence 快照未修改。主线中工作流交接不是实验因果；所有展示关系记录文件与位置。
- 同一布局坐标用于 Overview / Design / Optimize；通过 viewBox 和 clipPath 聚焦，不根据选择重新排列节点。任务导航默认折叠，Inspector 在下方按需打开。
- 单击节点默认高亮直接上下游；可切换完整语义溯源。Fit to View 恢复当前范围，返回全局清除选择并恢复全局缩放。Focus Selection 适配选中节点及其关联边界。
- 正式大用例 kernel：5.926 → 4.946 μs（1.198×）；保留小用例 3.250 → 3.428 μs 未获益限定。Golden E2E 比值未改为优化加速比。

### HTTP 与可视验收

本机 HTTP 入口：`http://127.0.0.1:8765/agent-decision-lineage-softmax.html`。若服务已停止，可在本目录执行 `python3 -m http.server 8765 --bind 127.0.0.1`。

已在 1440×1000 视口实际打开 Overview、Design、Optimize 并截图；页面 scrollWidth=1440，与视口相等。检查了 Runtime if 失败 Inspector、TILE_ROWS 未尝试 Inspector、Focus Selection、Fit to View、返回全局。截图见 `previews/overview-1440.png`、`design-1440.png`、`optimize-1440.png`。未重跑 NPU 性能或正确性测试。

### 仍有限制

这是单次 Softmax Run 的语义布局，不是支持任意拓扑的通用自动布局器。默认 Overview 为保留完整故事采用较密文字；聚焦视图使用原坐标裁切，留白较多。低于 900px 的窄屏保留画布滚动。记录不足的因果不补画；timeline 工具失败仅在说明中披露。硬件、性能详细视图仍未开发。
