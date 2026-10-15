# CSA Pass Field：语义修复、双卡 DAG、执行与内存性能计划

日期：2026-10-15  
状态：M0 语义修复与 M0.5 计算图/rank 编译联动已实现；用户反馈“ok”，本轮同步计划。真实执行与内存性能接入待实施；完整浏览器验收尚未完成。  
读者：产品、设计、前端与编译/runtime 数据接入开发者。

## 1. 目标与范围

将现有“qk_pv 编译过程解释”扩展成可追溯的工作流：**计算节点 → 编译变化 → 双卡任务与数据分工 → 实际执行 → 内存性能线索 → 对照验证**。

本计划回应四项需求：

1. 列出当前版本需要修复的问题，区分已确认问题、可读性缺口和新增能力。
2. 阅读并评估本仓库已有泳道 Demo，明确复用边界。
3. 规划有真实数据支撑的双卡 DAG，区分 kernel 类型、资源池和执行实例。
4. 将本次 CSA 的 PH-MR-001 内存性能问题纳入新增特性。

产品目标是让开发者判断“哪里发生了什么，以及证据支持到哪一步”。不以增加图的数量为目标，不把编译提示直接等同于运行瓶颈。

### 1.1 源码与展示副本

| 对象 | 路径 | 本轮处理 |
| --- | --- | --- |
| 开发源码 | `Design/csa-pass-field/` | 已实现语义修复、rank 展开与编译联动；构建至 dist/index.html |
| 用户指定展示副本 | `/Users/yin/AI_Infra/ai-infra-research/patterns/csa-pass-field/index.html` | 上轮已读取内嵌数据；本轮不修改 |
| 原始运行 | `Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/` | 只读证据基线 |

AI_Infra 目录记录该页面来自本仓库 `dist/index.html` 的独立 HTML 快照。实施时先比较源码生成物与展示副本的数据及功能版本；不能假设二者实时同步。新能力先在开发入口验证，再按单独授权同步展示副本，提交、推送和发布均不属于本文档任务。

## 2. 本轮核验结果

### 2.1 双卡与资源

- `distributed_meta.json`：设备列表为 `[1, 3]`，runtime 为 `tensormap_and_ringbuffer`，backend 为 Ascend910B。
- `orchestration/host_orch.py`：按 rank 切分参数，并提交 `decode_csa_test`。
- `rank0/d0` 与 `rank1/d0` 都包含芯片记录、依赖、名称表、dispatch 程序标识和 merged trace。
- 两个 rank 的记录均报告 72 个核条目：24 AIC、48 AIV；各有 4038 条 `aicore_tasks` 记录。记录条数不是独立逻辑 Task 数，也不是两卡合计。
- 两个 rank 的名称表都将 FuncId 48/49 映射到 `qk_pv_aic` / `qk_pv_aiv`。
- 两个 rank 中出现相同 taskId `8589934645`，但运行时间不同；taskId 必须带 rank/dispatch 命名空间。
- trace 中 `Task:r2t53` 的 `r2` 不能当作设备 rank；设备归属来自运行目录和 dispatch 元数据。

上述资源数量是该次记录报告的资源清单。某个 kernel 实际用了多少核，必须由它的执行块统计；不能直接标为“qk_pv 使用全部 72 核”。rank→物理设备编号还需在接入时用 dispatch/运行配置交叉核对，不能把 `d0` 理解为物理 Device 0。

### 2.2 当前图的真实语义

- `kernel.aic`、`kernel.aiv` 在生成器中是 `qk_pv_aic`、`qk_pv_aiv` 两种 kernel，属性为目标 AIC/AIV；并非两个物理核实例。
- Pass 23 `ExpandMixedKernel` 改写 qk_pv family，产生 AIC/AIV 拆核及交接。
- Pass 22 `LowerAutoVectorSplit`、Pass 40 `AutoDeriveTaskDependencies` 在此次输入中是 no-op。
- Pass 43、45、50 改写整份程序，但不改写 qk_pv family。分布式相关 Pass 出现，不等于局部 DAG 已呈现双卡策略。
- 当前源码已将最终 ISA 设为独立 result stage，并兼容 `stage=isa` URL。不能再把“ISA 被归因于 Pass 51”列为当前已确认缺陷；此项作为回归约束保留。
- `build-context.mjs` 明确声明上下文图是语义对应，不是该次编译运行的对象身份映射。现有上下文图不能直接当作已验证的 runtime DAG。

## 3. 当前版本 Fix 清单

| 编号 / 优先级 | 分类 | 问题与证据 | 修复要求 | 验收 |
| --- | --- | --- | --- | --- |
| F01 / P0 | 已确认语义问题 | 生成器 fallback 用整程序 `changed` 推导“qk_pv 源 IR 发生可观测改写”；部分 program-source-only Pass 的 intent 与 summary 相矛盾 | 分开保存 Pass 的一般职责、本次整程序变化、本次局部变化；不得用程序变化冒充 qk_pv 变化 | 对 43/45/50 分别检查：整体变化、局部不变、详情文案一致；22/40/51 保持 no-op |
| F02 / P0 | 表达缺口 | 两个 kernel 类型缺少物理部署范围，易读成一张卡上各一个核 | 在局部图标明 kernel/目标核类型；显示“完整运行 2 ranks，当前查看某 rank 或共享编译结构”；硬件数量由记录驱动 | 能区分 2 个 kernel 类型、每卡 24/48 资源条目、所选任务实际用核数 |
| F03 / P1 | 导航缺口 | 分布式 Pass 改变发生在局部图外，主图静止，用户无法追到变化对象 | 给出“变化位于完整 CSA”定位入口；未接入前可定位原始快照，接入后定位双卡图对象 | 43/45/50 可追到真实改变范围，不新增虚假的局部动画 |
| F04 / P0（接入前） | 身份缺口 | 静态上下文仅有语义对应，缺少 run/rank/dispatch/task/block 的关联契约 | 建立可追溯映射表；缺少运行身份的节点只展示结构，不能附上近似任务的耗时 | 不依赖名字近似来伪造映射；未匹配对象有具体原因 |
| F05 / P1 | 交付一致性 | 开发源码与 AI_Infra HTML 是不同资产，新增能力可能只落到一侧 | 生成 manifest，记录数据 run、schema、生成版本/哈希；实施前比对两份产物 | 开发入口与最终交付副本的功能、来源和版本可对账 |

### 3.1 已实施状态

- F01：已区分整程序与 qk_pv family 变化；相关数据断言通过。
- F02：已明确 kernel 目标类型与资源数量的不同含义。资源清单不表示当前任务实际用核；当前计算图不铺开核格。
- F03：已提供整程序变化证据片段；Pass 导航不再将局部无变化全部置灰，区分局部变化、全局变化与 no-op。真实双卡对象定位仍依赖后续映射。
- F04：已建立 object-links 身份契约和 unresolved 说明；尚未接入动态实例关联。
- F05：已生成版本/输入哈希和独立 HTML；AI_Infra 展示副本未同步，不视为同一版本。
- 图元：计算与通信复用共享组件；计算图标采用设计系统 fx.svg，文字居中、图标靠右。隐藏连线端口圆点，移除重复的“CSA 阶段 SCOPE”。

详细历史记录见 [Fix 验收记录](fix-review.md)；当前融合方向以本文及 [rank 联动记录](rank-context-integration.md) 为准。

### 3.2 已有正确行为：保持并回归

- 最终 PTO-ISA 是代码生成产物；runtime 不是第 52 个编译 Pass。
- 指令 occurrence 为静态源码计数，不是动态执行次数或时间。
- C2V/V2C 属于片内跨核交接；跨 rank 通信采用不同边类型与标识。
- MemoryReuse 的复用量或地址变化不能直接折算成加速。
- 保持已有滚轮缩放、空白拖动、局部展开、对象选择与语义配色。

## 4. 已阅读的泳道 Demo 与复用策略

### 4.1 Runtime Trace 泳道图

入口：[runtime-trace-swimlane/index.html](../../runtime-trace-swimlane/index.html)。

实际实现：

- `normalizeTrace` 读取 Chrome Trace `M` 元数据和 `X` 完整时长事件，按 `pid:tid` 形成轨道。
- Canvas 绘制调用 `PtoSwimlaneTaskPattern.drawTaskBar`；已有 hover、点击 Inspector、事件搜索、线程选择和 1–12× 滑条缩放。
- 缩放方式是改变 Canvas 宽度并横向滚动；不是已经具备完整时间窗状态的双卡执行工作台。
- 默认数据是 vLLM/PyPTO trace，不是本次 CSA trace。

已发现的接入前问题：

1. 第 330 行 `TRACE_URL` 包含多余的单引号：`../../'Data/...`。该路径在仓库中不存在，无单引号的对应文件存在。已修正路径；HTTP 加载与实际交互仍待验证。
2. `eventKind` 把名称含 kernel/chip 的事件映射为 matmul，含 dispatch/submit 的事件映射为 mte_load。这种配色启发式不能用作 CSA 的真实执行单元分类。
3. 当前只处理 M/X，未接入此次 trace 已有的 C 计数器与 s/f flow；不能原样保留就声称支持任务依赖与调度队列。
4. lane ID 没有 run/rank/dispatch 前缀；合并两卡会发生身份碰撞。
5. `fetch` 外部 JSON 的方式不能直接满足当前 CSA 独立 `file://` HTML 交付方式。

复用：任务条、tooltip、Inspector、轨道筛选和坐标绘制思路。将这些封装为可嵌入视图；保留 CSA 的页面框架，不移植一套重复导航。

### 4.2 同一份 CSA 数据的调优控制台

入口：[operator-tuning-console/index.html](../../operator-tuning-console/index.html)。已阅读 `build-data.cjs`、`app.js` 和 README 的相关部分。

它比通用 Trace Demo 更适合作为数据适配参考：

- `buildRank` 已读取同一 run 的 rank0/rank1、Worker/Scheduler、deps 和名称表。
- 明确处理 mixed kernel：一个 group taskId 下可以同时有 AIC/AIV，必须再按 FuncId 归组，避免把 Vector 时间算到 Cube 名下。
- 已有依赖连线、时间窗、任务定位、核占用、kernel/setup 分项及 PH-MR-001 解析。
- 核占用与任务墙钟时间采用不同口径；已有实现也明确讨论了依赖图和 happens-before 的区别。

复用方式：提炼纯数据转换和 shared swimlane pattern，增加 CSA adapter 测试；不要整体复制大型 app.js 或复用其场景构造数据。已有实现有特定 pid、名称和单程序假设，接入时用 process/thread 元数据、dispatch_program 与运行 schema 验证，避免照搬硬编码。

### 4.3 执行视图交互契约

- 一级入口：编译 / 执行；内存问题作为对象关联的诊断入口，避免增加独立且断开的工作台。
- 选择同一对象跨视图保持：静态 kernel → 对应任务列表 → block；从 block 回到 kernel、Pass 与源码。
- 默认按 rank 分组，再按 AIC/AIV 折叠；保留完整资源清单和“仅显示活跃核”选择。
- 时间采用微秒，原始 tick 使用记录中的 `clock_freq_hz` 转换；不硬编码另一种设备频率。
- 默认分别显示两卡相对起点；跨卡绝对对齐只有在时钟域与偏移证据成立后启用。相对起点相同不能解释为同时启动。
- Worker、Scheduler 是同一工作的不同观测口径，不能累加成两次耗时；setup 是否已包含在 duration 中须按本 run schema 确认。
- trace 缺失、名称未匹配、数据未加载与筛选为空分别处理；没有精确指令 trace 时，指令面板显示静态关联而非编造执行条。

## 5. 双卡 DAG 方案

### 5.1 可行性结论

可构建由这次运行支撑的**双 rank 任务 DAG + kernel/block 展开**。不能把现有 qk_pv 图简单复制两份后直接称为完整双卡运行图。

需要同时表达三层身份：

| 层 | 对象 | 数据依据 |
| --- | --- | --- |
| 编译结构 | qk_pv、AIC/AIV kernel、静态指令、MemRef | pass dumps、.pto、代码生成产物 |
| 双卡部署 | rank、dispatch、卡内资源池、张量分片、通信端点 | distributed_meta、host_orch、前端/最终 IR、通信参数 |
| 执行实例 | task、FuncId、block、CoreId、时间 | deps、name_map、chip records、merged trace |

kernel 定义可以被多次调度并在多核执行。物理核与 kernel 是映射关系，不能以“一个 kernel 节点”等价为“一个硬件核”。

### 5.2 图的组织

当前采用“同一计算图的 rank 展开 + qk_pv 编译联动”，撤掉独立“双卡部署”tab，避免重复表达同一计算流程。

- 计算图面板提供“展开 rank / 收起 rank”，复用原图节点、连线和局部折叠，不另画一套阶段卡片。
- 展开后以 Rank 0、Rank 1 为上下文容器。当前依据同一 per-rank 程序作静态 Mock 投影；sourceId 保持语义身份，rankId 区分部署上下文，不代表已有真实任务/分片映射。
- CP Token All-Gather、Attention All-to-All、Output Reduce-Scatter 展示聚合的双向跨 rank 通信。连线按节点包围盒避障，从上下空隙绕行；不穿过其他 op，双向路线分离。
- 点击任一 rank 的 QK / Softmax / PV，主画布关联该 rank 的 qk_pv 编译上下文，保留当前 Pass。展示的是同一份共享编译证据，不伪造两份 rank 独立编译记录。
- rank 配置独立于 Pass 进度；从 Pass01 可查看部署上下文，不表示 Pass01 生成或完成了分卡。旧 view=deployment URL 转入计算图 rank 展开，主画布保持编译过程。
- 当前不提供真实分片范围、task/core 绑定或动态执行时间。下一阶段在现有图补充来源明确的映射，不能把逻辑副本直接升格为已验证双卡执行图。

官方 L3 文档的 Host → per-rank orchestration → core compute 分层支持这种语义组织；面板与按钮布局是产品设计判断，并非官方规定的 UI。

后续接入真实执行后，从 qk_pv 选择 mixed group、kernel/task，再列出实际 CoreId 或进入泳道。默认不展开全部核与原始记录。

边类型分开保存：

- 张量数据依赖；任务提交依赖；片内 C2V/V2C；跨 rank payload；signal/wait 控制；kernel 到 core 的部署映射。
- 数据、通信、控制、存储与任务维持不同类别编码。rank 使用容器和文字，不占用对象语义色。
- 仅有“同名 tensor”不足以建立跨 rank 边。由 peer、通信域、窗口/offset、signal 协议等匹配通信端点。
- 对缺少精确动态对应的跨卡关系，只呈现有源码证据的静态通信连线，不显示传输开始/结束或耗时。
- 循环协议的静态图可能含环。只有按迭代/任务实例展开并验证无环的依赖子图才称为执行 DAG；不能通过删边强行做成 DAG。

qk_pv 内部继续在主编译画布呈现，并与计算图中的 rank 选择联动。跨卡通信在完整 CSA 的真实边界处出现，不应凭空添加 `qk_pv(rank0) → qk_pv(rank1)`。

### 5.3 数据契约（拟新增）

沿用现有 JSON → domain state → React Flow adapter 分层，新增运行投影而非把 runtime 数据写进 Pass reducer。

| 文件 | 内容 |
| --- | --- |
| `run-manifest.json` | run、输入产物哈希、schema、平台、rank/dispatch、时钟与对齐状态 |
| `deployment.json` | rank/设备关联、每卡 core 清单、张量分片与通信端点 |
| `execution.json` | tasks、kernel instances、blocks、dependencies、counter/flow 与计时口径 |
| `object-links.json` | 静态对象与动态实例的多对多关联、来源及匹配方式 |
| `memory-findings.json` | 编译提示、buffer/group、源码、关联任务与验证状态 |

任务主键：`runId/rankId/dispatchId/taskId`。kernel 实例追加 FuncId；block 追加 CoreId、block/record 身份。静态 instruction ID 不包含伪造的运行次数。

关联按 dispatch_program 解析对应程序的名称表，保留 taskId 字符串，避免大整数精度丢失。每条跨层关联记录 `sourceRefs` 和 `observed / derived / unresolved` 状态；UI 用具体缺失项解释未匹配。

## 6. 新特性：CSA 内存压力与流水深度退化

### 6.1 首个真实案例

`report/perf_hints.log:1` 报告：

- Pass：MemoryReuse；诊断：PH-MR-001。
- 源码：`decode_csa.py:397:9`。
- 当前本地源码对应 `name_hint="csa_merge_pack_publish"` 的 SPMD scope，依赖 qk_tid 与 attn_rope_tid。
- Vec pipeline group 0 请求 depth 2，只容纳 1/2 个 buffer；每 stage 32768 B，日志报告 free 188416 B。
- 日志解释：单独的 operand 能放下两份，但共存 buffer / 其他 pipeline group 使空间超额，阶段共享存储并串行化。

这是编译器报告的规划退化，属于 qk_pv 下游的 merge/pack/publish 阶段，不能直接标在 qk_pv 上，也不是内存越界或数值错误。源码映射正式入库前仍须核对归档版本/哈希与编译 loc；当前行号和 name_hint 一致是映射证据的一部分。

### 6.2 产品体验

1. 完整 CSA 图在 merge/pack/publish 标出“流水深度 2 → 1”。点击打开对应原始提示。
2. 联动定位 MemoryReuse 前后 IR，展示该 group 的 buffer 身份、复用关系与所处内存空间。
3. Memory Map 同时表达地址区间和静态生命周期；可定位共存对象。静态生命周期轴与 runtime 时间轴明确区分。
4. 关联到该 scope 的任务与执行区间，提出可验证问题：“该规划退化是否解释了这里的串行/等待？”
5. 调整 tile 或共存对象的方案进入实验记录；重编译检查提示与分配变化，重跑检查精度和时间，再给出效果结论。

不能用 `floor(188416 / 32768)` 直接推导可用深度：日志已指出共存竞争，这不是该 buffer 独占的预算。分配表、预留区、对齐和冲突集合不完整时，只展示已知值与缺失项，不生成貌似精确的可用容量。

### 6.3 证据与结论门槛

| 层级 | 可交付判断 | 需要的证据 |
| --- | --- | --- |
| 编译事实 | depth 2 → 1、提示定位、规划结果 | 原始日志、IR、源码对应 |
| 运行关联 | 某 scope 对应哪些 task/block | 名称表、group/FuncId、dispatch 与源码映射 |
| 致因假设 | 内存复用约束可能降低重叠 | 依赖、生命周期、执行现象；不足时保留假设 |
| 优化成立 | 具体修改减少耗时且精度通过 | 同输入/硬件/采集口径的前后编译和多次运行 |

不从时间轴空白直接诊断内存瓶颈；降低 depth 可能只消除不合理请求，并不保证提速。将历史 #2510/#1118 留作后续独立正确性案例，不混进这次运行的已发生问题。

## 7. 实施阶段与交付物

| 阶段 | 工作 | 交付物 | 退出条件 |
| --- | --- | --- | --- |
| M0：语义修复（已实现） | F01/F02；核对产物版本；保留已正确的 ISA/no-op 语义 | 修订生成器/文案、版本清单、针对性断言 | 局部/全局变化不矛盾；kernel 与核数量可区分 |
| M0.5：计算图融合（已实现） | rank 展开、共享图元、qk_pv 当前 Pass 联动、跨 rank 避障 | context-ranks 投影与计算图联动；详见 rank-context-integration.md | 构建/数据/结构检查通过；浏览器视觉验收待补 |
| M1：执行证据接入 | 从控制台提炼 adapter；封装泳道 pattern；修通通用泳道的数据 URL（独立小修） | 运行 schema、数据生成器、单 rank 执行视图、对象关联表 | 两个 mixed kernel 不串账；任务/块/核可往返定位 |
| M2：真实双卡 DAG（待接入） | 在现有 rank 展开上接入真实分片、通信端点、实例与资源关联 | 双卡 deployment/execution 投影、双 rank 泳道 | rank ID 不碰撞；跨卡边有证据；无依据时不做绝对对时 |
| M3：真实内存案例 | 接入 publish scope 的 PH-MR-001、Memory Map、编译/运行关联 | 内存性能诊断与实验入口 | 提示定位准确；规划退化与运行因果分开；无虚构收益 |
| M4：交付验证 | 独立 HTML 打包、浏览器烟测、授权后的副本同步 | 可复现构建与验证记录 | 离线资源、URL 恢复、关键状态和数据口径通过 |

M0 与 M0.5 已实现；后续按 M1 → M2 → M3 推进，并补齐 M4 浏览器验收。M1 先完成一张卡的链路，再扩展两卡，可以优先发现计时和身份问题；M0.5 已显式呈现双 rank 静态上下文，M2 才交付真实双卡执行关联。

### 7.1 当前验证与剩余工作

- 已通过：生产构建、数据校验、2 项计算图测试；所有折叠变体的 rank 节点唯一性、通信端点和 focus 身份检查；256 种折叠组合共 1,536 条跨 rank 路由避障检查。
- 用户通过截图反馈过连线遮挡；已修复并收到“ok”。这不替代完整浏览器交互回归。
- 尚未完成：1440×900/窄屏视觉烟测、旧 URL 与交互状态完整回归。此前工具 file:// 访问被策略拒绝，未绕过该限制。
- 尚未实施：真实 task/block/core 关联、双 rank 泳道与时钟对齐、细粒度通信/信号协议、PH-MR-001 诊断和 Memory Map。
- 未提交、推送、发布或同步 AI_Infra 副本；不把当前 Mock 判定为 M1/M2/M3 完成。

### 7.2 验证清单

- 数据：从原始文件重新统计每卡 core/task/block；4038 是本 run 单 rank 记录基线，检查失败时报告变化，不把常量当通用 schema。
- 身份：相同 taskId 出现在不同 rank 不合并；同 taskId 的 AIC/AIV 以 FuncId 区分；多 dispatch 的名称解析不串表。
- 时间：tick→µs 正确；任务墙钟与多核累计时间不同；Worker/Scheduler/setup 不重复计数；对齐方式可追溯。
- 图：验证节点、边、端口与对象映射；区分提交依赖与严格完成后启动；早发任务不生成伪造关键路径。
- Pass：22/40/51 no-op、23 拆核、43/45/50 全局变化；最终产物 stage 及原 URL 可恢复。
- 内存：首个 PH-MR-001 指向 publish scope，数值与日志一致；不宣称已解决或已提速。
- UI：在 1440×900 作为拟定基准视口做最终烟测，另查窄屏折叠；覆盖切 rank、选择 kernel/task/block、双向定位、缩放平移、折叠、筛选空态、缺失数据和旧 URL。
- 交付：运行现有数据校验、相关 Node 测试及构建；新增 adapter 用真实固定样本测试。最终 `file://` HTML 不依赖外部 fetch 或本机绝对数据路径。

## 8. 来源与本轮验证边界

### 本地证据

- [已有方案 A](solution-a-data-and-react-flow-plan.md)：延续 JSON/domain state/React Flow 分层。
- [CSA App](../src/App.jsx)：独立 ISA result stage、URL 兼容与 Inspector。
- [CSA 数据生成器](../scripts/build-csa-qk-pv.mjs)：kernel 类型、Pass 变化与 contract fallback。
- [上下文生成器](../scripts/build-context.mjs)：语义对应的范围声明。
- [通用泳道](../../runtime-trace-swimlane/index.html)：数据路径、解析、Canvas 与交互。
- [控制台数据生成器](../../operator-tuning-console/build-data.cjs)：双 rank、mixed group、任务和内存提示转换。
- [控制台交互](../../operator-tuning-console/app.js)：执行视图、时间窗与依赖连线。
- [运行配置](../../../Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/distributed_meta.json)、[Host 编排](../../../Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/orchestration/host_orch.py)、[运行记录目录](../../../Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/dfx_outputs)。
- [内存提示](../../../Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/report/perf_hints.log)、[CSA 源码](../../../Data/DeepseekV4/deepseek_v4_flash_dspark/decode_csa.py)。

### 社区官方依据

以下资料于 2026-10-15 的前序分析中检索；main 文档可能与归档 run 版本不同，实施以 run 对应 schema 为准。

- [L3 编程模型](https://www.pypto.ai/pypto-lib/pypto-coding/l3-programming/)：2026-10-15 已核对 Host 按 rank 分派、device=r、window 数据传输与 notify/wait 分离。
- [PyPTO 编程模型](https://github.com/hw-native-sys/pypto/blob/main/docs/en/user/03-programming-model.md)：2026-10-15 已核对编译层次与 runtime 调度的区别。
- [工具链与双代码生成路径](https://github.com/hw-native-sys/pypto/blob/main/docs/en/dev/00-ecosystem.md)。
- [Runtime DFX 与分 rank/dispatch 产物](https://github.com/hw-native-sys/pypto/blob/main/docs/en/dev/03-runtime-dfx.md)。
- [编译诊断与性能提示](https://github.com/hw-native-sys/pypto/blob/main/docs/en/dev/passes/92-diagnostics.md)。
- [PTO codegen](https://github.com/hw-native-sys/pypto/blob/main/docs/en/dev/codegen/00-pto_codegen.md)。
- [历史布局错误 #2510](https://github.com/hw-native-sys/pypto/issues/2510)、[历史多槽同步错误 #1118](https://github.com/hw-native-sys/PTOAS/issues/1118)：前序查询时均已关闭，本文不将其视为当前 CSA 已发生的问题。

已完成源码与原始数据核对、上述产品代码修改、构建和结构检查；本次仅更新计划文档。未完成工具浏览器验收，未执行硬件实验，未验证性能收益。当前实现细节与静态映射边界见 [rank 联动记录](rank-context-integration.md)；此前独立部署骨架记录保留为历史，不能作为当前入口说明。
