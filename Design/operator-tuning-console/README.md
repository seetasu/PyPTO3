# 算子调优控制台 · Tuning Console

把「端到端 → L2 调度 → L1/L0 单核流水 → 编译降级 → ISA / 布局」这套调优闭环，做成一个可操作的产品界面，
而不是一篇讲解或一个向导。对象是一次**真实上板执行**：

```
Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/
```

打开入口：`Design/operator-tuning-console/index.html`，也在 `launch.html` 的「内存与性能」分类里。

---

## 这个 Demo 是什么形态

一个 IDE 形态的工作台，工作单元是**瓶颈队列里的一条条目**：

| 区域 | 内容 |
|---|---|
| Explorer | 运行树（program → rank/device → invocation → 产物）+ **瓶颈队列**（10 条，按实测影响排序，可按层筛选） |
| 中心 | 五个层级视图，用页面级 Tab 切换：E2E / L2 调度 / L1·L0 / 编译器 / ISA·布局 |
| Inspector | 跟随当前视图的对象（Run / 任务 / 提示 / Pass / 瓶颈条目）+ **实验台账**编辑器 |
| 底部 Dock | Visualization（AICPU 调度 / Ready queue / 核占用）与 Terminal（Problems / Output / Artifacts）互斥切换 |
| 状态条 | case、rank、trace span、关键路径节点数、AIC/AIV 占用、调度器占用、PMU 状态、进行中实验数 |

闭环被做成了产品约束，而不是提示语：

- **门禁先行**：E2E 视图第一屏是四个门禁（Case 固定 / 工具链 / 迭代次数 / PMU）。本次 dump 只有 2 次调用，
  「迭代次数」门禁直接是 warn，并写明 mean/median 不成立。
- **每轮只验证一个假设**：台账同时只允许一条 `open` 实验。已有进行中实验时，点开另一条瓶颈只会看到拦截提示和
  「回到 Ex」按钮，不给第二个「开始实验」。
- **顺序是固定的**：实验步进器必须先记录正确性，才出现性能输入框；两者齐了才出现「保留 / 回退」，
  并在决定前再显示一次该条目的护栏。
- **每条结论都带护栏和复测口径**：瓶颈条目由 `证据 → 杠杆 → 护栏 → 复测` 四段组成，护栏写的是这条杠杆的代价
  （例如 prefetch 占 SDMA、提前 dispatch 以吞吐换时延、PMU-on 不能与 PMU-off 基线比较）。

---

## 瓶颈 → 屏幕：证据是怎么对上的

点一条瓶颈之后，中间不会只是「换了个页签」。**证据条**会钉在中心区顶部，把这条结论和屏幕上的具体对象绑起来。
它只有两行，没有解说：

```
┌ F1  通信等待独占关键路径 36.72%   1791.82 us / 4879.82 us   [聚焦证据 ✓] [退出] ┐
│ 证据 4  ①cp_token_allgather_payload_wait 1080.52us  ②o_group_a2a_wait 663us … │
└──────────────────────────────────────────────────────────────────────────────┘
```

三处用**同一套编号**串起来：

| 位置 | 表现 |
|---|---|
| 证据条的 chip | `① ② ③ ④`，点一下把时间轴缩放到那个对象上 |
| 中心画布 | 关键路径 ribbon 多出一条「证据」行，泳道里对应的块加白框 + 同号圆标；非证据对象在「聚焦证据」开启时压暗到 16% |
| 右侧 Inspector | 证据小节标题变成「N 项 · 已在中间标号」，每条证据行带 `标号 1 / 2 / 3`，整行可点，跳到同一个对象 |

编译器层的瓶颈（F4 / F5）没有画布，改成**表格行高亮**：`is-subject` 行加主色左条和淡底，9 / 12 行一眼可辨。
F10 这种「缺席证据」的条目不给 chip，明确写出「本页没有可标记的对象」。

`聚焦证据` 只压暗、不隐藏——被压暗的东西仍然可悬停、可点击，避免把上下文一起删掉。
`退出` 清除瓶颈上下文，回到自由浏览；切页签不会清掉它，因为一条瓶颈常常要跨两层看。

### 页面上不写解释

这是工具不是教程：中心区没有任何叙述性段落，五个页签下的 `<p>` 数量是 0 / 0 / 4 / 0 / 0
（L1 的 4 条是试算器判定，全部是带数字的短读数）。说明性内容只存在三处，且都是可执行的：

- 瓶颈条目的 `杠杆 / 护栏 / 复测` 三张卡——这是要照着做的动作，不是背景介绍
- 编译提示 Inspector 的「处理」两张卡——`减少同驻 tile，而非调大 stage` 这类一句话结论
- 实验台账的拦截消息——`E1 进行中 · 结论后才能开下一个`

页签名不重复出现在页签下面：工具条只放控件（编译器的 `Pass 轨迹 / 流水深度 / 搬运粒度` 子切换、
L2 的泳道 / 着色 / 叠加、L1 的 kernel 选择、rank 选择），没有控件的页签（ISA / 布局）工具条整行隐藏。

其余位置一律用「标题 + 计数 / 单位」做小标题，例如 `门禁 · 4 项`、`IR 规模与改写点 · 1794 → 4950 行`、
`缺失产物 · 本 dump 不含 PTOAS / VPTO 级记录`。原来的 ISA 空状态是一段散文加项目符号，现在是一张三列表
（产物 / 用于 / 状态），四行全部标红「缺失」。

---

## 数据来源：每个数字都来自这次执行

`data.js` 由 `build-data.cjs` 从 dump 目录直接生成，没有任何建模、估算或造数。

```bash
node Design/operator-tuning-console/build-data.cjs
```

| 产物 | 提取出来的东西 |
|---|---|
| `distributed_meta.json` | 55 个绑定参数的 shape / dtype / 方向 → Case fingerprint |
| `dfx_outputs/rank{0,1}/d0/host.*.log` | STRACE host span（bind / runner_run / device_wall / graph_build / sched / orch）→ E2E 剖分 |
| `dfx_outputs/rank{0,1}/d0/merged_swimlane_*.json` | Worker View（pid 4）每块的 `duration-us` / `kernel-duration-us` / `local_setup_us` / CoreId；Scheduler View（pid 3）同一块的 `dispatch-time-us → finish-time-us`；AICPU scheduler phase；`shared_ready_queue` 计数器；dependency / hb_violation flow |
| `dfx_outputs/rank*/d0/deps.json` | `block_num` / `scope` / `early_dispatch` / 每个任务的绑定张量 |
| `dfx_outputs/rank*/d0/name_map.json` | 64 个 callable id → 名字 |
| `report/perf_hints.log` | 230 条 perf hint：197 条 PH001（搬运末维粒度，累计 261 次命中）+ 33 条 PH-MR-001（软流水深度回退） |
| `passes_dump/` | 52 份 IR dump 的行数、Δ 行、`pl.pipeline` / `tile.matmul` / `Mem.Left·Right·Acc·Vec` 计数；AutoTileMatmulL0 的真实 L0 tile 形状与 55 个 pipeline 站点；一段真实 before/after IR |
| `next_levels/.../binary_context.json` | platform、pto-isa revision、runtime 名与 revision → 工具链指纹 |

### 两个视角，不要混

Trace 里同一个块有两份记录，工具把它们分开显示，因为它们回答的是不同问题：

- **Worker View（pid 4）**：核上发生了什么。`duration − kernel_duration = local_setup`。
- **Scheduler View（pid 3）**：AICPU 看到的 `dispatch → finish`。它减去核上时长就是领取与依赖等待。

L1 视图的「一个块的三种口径」就是这三层：`kernel` → `+ setup` → `+ hand-off`。

### 关键路径怎么算的

用每个任务的 `fanin-hint` 建图，按实测 start 排序做 DP，取加权最长链（权重是任务自身的 span）。
rank0 得到 33 个节点、链上 span 合计 4985.0 us；链上正向间隙 378.3 us，重叠 539.7 us
（重叠为负间隙，说明后继在前驱最后一块结束前就起来了）。

### 对账

E2E 视图会把 host 报的 `device_wall.sched` 与设备 trace 的跨度对齐，确认这份 trace 属于哪一次调用。
两个 rank 都命中 `inv=2`，偏差 < 1%。不对账就无法保证「看的 trace 和读的数字是同一次运行」。

---

## 十条瓶颈条目（全部由数据推导，非人工填写）

| ID | 层 | 结论 | 实测依据 |
|---|---|---|---|
| F1 | L2 | 通信等待独占关键路径 36.72% | 4 个 `*_wait` 任务合计 1791.8 / 4879.8 us，全部单块单核 |
| F2 | L1 | `csa_merge_pack_publish` hand-off 比核上计算还贵 | AICPU 1032.1 us vs 核上 387.4 us（+644.6）；核上 57.8% 是 setup |
| F3 | E2E | rank0 / rank1 device_wall 偏斜 1.28x | 5132.8 vs 4017.3 us；rank0 更慢但核占用更低（32.2% vs 41.4%） |
| F4 | 编译器 | 9 处软流水深度被降到 1 | 33 条 PH-MR-001；Left/Right 32–64 KB/stage vs 64 KB free |
| F5 | 编译器 | 261 次搬运末维 < 512B cache line | 最小 4B，覆盖 9 个算子文件、192 个源码点 |
| F6 | L2 | AICPU 调度器平均占用 42.3% | 3 线程合计 busy 6143.9 us；complete 3452.6 us / 4038 次 = 0.855 us/次 |
| F7 | L2 | AIC ready-but-undispatched 占窗口 26.6% | `shared_ready_queue` avg 0.433、peak 6，同期 AIC 核占用仅 32.2% |
| F8 | L1 | `qr_hadamard_matmul` 块时长离散 7.82x | 256 块 / 24 核，max 19.7 vs med 2.52 us |
| F9 | L1 | `qk_pv_aic` 混合核 span ≈ 单块时长 | 72 块 / 72 核，span 886.4 us，单块 648.1 us |
| F10 | L2 | 全程 0 处 `pl.prefetch` | 前端 IR 计数；10 个 `w*` 权重参数存在但无静态预取 |

---

## 片上预算试算器

L1 视图里的试算器不是通用公式演示，它对着**本 run 的实测上限**校验：

- `Left = M × K × bytes(AB)`，`Right = K × N × bytes(AB)`，`Acc = M × N × bytes(Acc) × live`
- Left / Right 的可用空间取本 run MemoryReuse 报告里的 `65536 B`；Vec 侧是 `188416 B`
- Acc 不报上限，所以只给「本 run 出现过的最大 Acc tile = 128 KB」作为**下界**，并明确说明超过它属于待验证
- 末维检查：`N × bytes(AB)` 对 512B cache line；不足时给出该 dtype 的元素倍数（INT8 512 / BF16·FP16 256 / FP32 128）

默认值就是 dump 里真实存在的那条 K 循环：`Left INT8[16,64]` + `Right INT8[64,512]` + `Acc INT32[16,512]`，`stage=2`。
在这个配置下 `stage × max(L,R) = 65536 B` 正好等于 free，但本 run 的 `qkv_proj_rope.py:375` 仍然只放下 1 个 buffer
——同驻 tile 会分走这块空间。试算器因此把「刚好等于上限」判为**放不下**，而不是刚好通过。这是用实测反例校准的判据，
不是理论公式。

点右侧「AutoTileMatmulL0 dump 里真实出现的 L0 tile」任意一行，可以把该形状回填到试算器。

---

## 诚实的空缺

- **没有 kernel → 源码映射**。perf hint 挂在源码位置上，IR 只保留 outline 后的 incore scope 名，两者之间
  这份 dump 不提供可验证的对应关系。L1 视图因此不做自动归因，只提供模块选择器 + 明确说明「对应关系需人工确认」。
- **没有 PTOAS / VPTO 级产物**。ISA 视图只给这份 dump 能支撑的结论（工具链指纹、布局与内存空间分配、L0 tile 清单、
  512B 约束），并列出需要补齐哪些产物（TileLib 模板选择记录、VPTO 指令排布报告、cycle cost model 预测、PMU counter）
  才能把结论推进到指令层。
- **没有 PMU**。trace 里没有硬件 counter，门禁标为 `off`，并注明 PMU 打开会改变调度，不能与本基线直接比较。
- **只有 2 次调用**。所以工具不显示 mean/median，只显示每次调用的值，并在门禁里把这点标成 warn。

---

## 设计系统落地

页面是 `vendor/pto-design-system` 的消费者，不自建视觉语言。

- Shell：`patterns/ide-frame`（standalone host、activity rail 四键、explorer/inspector toggle、bottom dock 互斥、status strip），
  resize 交给它委派的 `patterns/workbench-shell`。保留共享渐变 / aura / pane 磨砂皮肤，只放宽 4:3 为全视口
  （与仓库里其它全屏 Demo 一致）。
- 所有计时任务条与 hover tooltip 走 `patterns/swimlane-task`：`drawTaskBar`、`createTaskColormap`、
  `initHoverTooltip` + `showTooltip` / `hideTooltip`。没有本地重写任务条几何、配色哈希或 tooltip 行为。
- 不使用播放条：这个页面没有 demo 时间轴 / step / scrubber 语义。
- 颜色：`styles.css` 里 0 个硬编码色值，全部走 token；`app.js` 里 0 个私有调色板，lane/任务配色全部由
  `createTaskColormap()` 决定。
- 默认 dark（新建 standalone PTO 页面的默认），右上角可切 light。

### 一条对齐基线

所有容器标题与内容区文字落在同一条竖线上：pane body 取与 `.pto-ide-frame__pane-header` 相同的
`--tc-gutter: 10px`，而带内边距的表面（表格滚动区、canvas、gate / tile 卡、台账条目、树行、瓶颈卡）
用 `--tc-bleed` 反向外溢自己的 cell padding，于是表面边缘压进 gutter、表面内的文字正好回到 gutter 上。
这是 VS Code 侧栏的老做法：行的高亮满幅，文字对齐。

### 已知例外

1. **全视口**：`.tc-frame` 放宽 `aspect-ratio` 为 `auto`、去掉圆角与投影。渐变、aura、pane fill、blur 全部保留，
   没有用页面私有面板替换共享皮肤。与 `deepseek-gap-investigator`、`pass-decision-studio` 的处理一致。
2. **Canvas 内 data-viz 标注低于 11px**：72 泳道的 lane label 用 10px，时间轴刻度用 11px。行高只有 8px，
   放大字号会互相压盖；这属于设计系统允许的「可缩放 data-viz 内部标注」例外，且所有信息都有 hover tooltip
   与 Inspector 作为 fallback。`swimlane-task` pattern 自身的条内文字也是 8–9px。
3. **inspector-section 家族在本页实现**：`.inspector-rail / .inspector-section / -head / -title / -kicker`
   与 `.inspector-soft-card` 写在 `references/quick-reference.md` 里、token 定义在 `tokens/components.css`，
   但设计系统没有发布对应的 CSS 规则（仓库里每个 Demo 都各自实现）。本页按那组 token 实现，不引入新值。
   在此之前它们完全没有样式，`<h3>` 回落到浏览器默认的 16px 粗体 block，导致「瓶颈队列 / 10 / 10」换行。
4. **封面是卡片不是截图**：`Design/assets/launch-previews/tuning-console.svg` 是一张说明性封面卡，
   里面内联了 token 色值（`<img>` 加载的 SVG 拿不到宿主页面的 CSS 变量）。它没有伪装成产品截图。

### 审计

```bash
cd vendor/pto-design-system
node scripts/audit-typography.mjs ../../Design/operator-tuning-console/styles.css
node scripts/audit-theme.mjs      ../../Design/operator-tuning-console/styles.css
```

两项均无告警。

---

## 交互速查

| 操作 | 结果 |
|---|---|
| 点瓶颈条目 | 跳到证据所在层级，钉出证据条，给证据对象编号并压暗其余部分 |
| 点证据 chip / Inspector 证据行 | 跳到同一个编号对象；时间轴自动缩放过去，Inspector 保持停在这条瓶颈上 |
| 证据条「聚焦证据」 | 压暗非证据对象（不隐藏，仍可悬停点击）；再点一次恢复 |
| 证据条「退出」 | 清除瓶颈上下文；切页签不会清除，方便跨层追同一条 |
| `1`–`5` | 切换五个层级视图 |
| `/` | 聚焦搜索（kernel / 任务 / 编译提示 / Pass，回车跳第一条） |
| L2 画布点击 | 选中任务 → Inspector 显示绑定张量、依赖链、关联瓶颈、所在核占用 |
| L2 画布 shift + 拖动 | 水平平移；工具栏 `+` / `−` / `Fit` 缩放，Dock 时间轴跟随同一窗口 |
| Dock「核占用」 | 72 条泳道的占用、空洞数、最大空洞、首末块 |
| Terminal「Problems」 | 230 条编译提示当作 IDE 问题列表，点击跳到对应源码点 |
| Inspector 依赖 chip | 沿 fanin / fanout 在任务图里走 |
| 试算器里点 L0 tile 行 | 把该真实形状回填进试算器 |

---

## 文件

```
Design/operator-tuning-console/
├── index.html        ide-frame shell 与槽位
├── styles.css        仅页面级布局；颜色全部走 token
├── app.js            状态机、五个视图、canvas 渲染、台账逻辑
├── build-data.cjs    从 dump 目录生成 data.js（可重跑）
├── data.js           生成产物，勿手改
└── README.md
```
