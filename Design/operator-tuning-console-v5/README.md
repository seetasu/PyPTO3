# 算子调优控制台 · Tuning Console

把「端到端 → L2 调度 → L1/L0 单核流水 → 编译降级 → ISA / 布局」这套调优闭环，做成一个可操作的产品界面，
而不是一篇讲解或一个向导。对象是**真实上板执行**：

```
Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/   decode_csa
Data/pypto_qwen3_profiles/                            qwen3_14b_fwd（由 Data/pypto_qwen3_profiles.zip 解压）
Data/_jit_decode_fwd_layers_20260625_184941/          qwen3_14b_fwd 的编译产物（旧构建，见「编译产物来自另一次构建」）
```

打开入口：`Design/operator-tuning-console/index.html`，也在 `launch.html` 的「内存与性能」分类里。

Case 菜单另有一个 **「模拟 PH-MR-001 · MemoryReuse」** 教学案例。它复用 `decode_csa` 的真实 L2 轨迹与 `decode_compressor_ratio4.py:110` 的真实 PH-MR-001（Right、depth 2→1、5 组、32 KB/stage、64 KB free），把它们关联到 `MemoryReuse`。L1 PMU 桥接是构造数据；提示与 L2 的因果关系仍是待验证假设，不代表已确认的编译器缺陷。案例集中在 `mock-pass-case.js`，真实数据生成器不会读取或覆盖它。

生成器有两支，都只读 dump、不造数：

```bash
node build-data.cjs            # -> data.js               任务级泳道 / 关键路径 / 编译器 / ISA
node build-qwen3-profiles.cjs  # -> qwen3-profiles-data.js torch 归因 / PMU / 40 层 / 头开销 / TP 对照
```

---

## 这个 Demo 是什么形态

一个 IDE 形态的工作台，三个真实 case 可切换（`decode_csa` / `qwen3_14b_fwd` / `serving_gbs256`），
工作单元是**性能发现里的一条条目**：

| 区域 | 内容 |
|---|---|
| Explorer | 运行树（program → rank/device → invocation → 产物）+ **性能发现**：分两组，上面是**诊断路径**（每条带 makespan 归因、跨层阶梯和明确的证据不足层），下面是**待归因信号**（真实读数但无归因）。按层筛选时，一条链在它经过的每一层都会出现 |
| 中心 | 五个层级视图，用页面级 Tab 切换：E2E / L2 调度 / L1·L0 / 编译器 / ISA·布局 |
| Inspector | 跟随当前视图的对象（Run / 任务 / 提示 / Pass / 瓶颈条目）+ **实验记录**编辑器 |

信息架构以证据作用域划分：**E2E 始终并排比较全部 rank**，用于界定跨 rank 的 Host / Device 问题；
进入 L2、L1/L0、编译器或 ISA 后，工具栏选择一个“分析 rank”继续下钻。编译器与 ISA 的 dump
当前为 rank 共享产物，rank 只保留分析上下文，不会伪造不同的编译结果。E2E 页签内直接呈现
“跨 rank 调度与 ready queue”对照，点任一 rank 才进入该 rank 的 L2 工作台；不再保留底部 Dock 或 Terminal 面板。
| 状态条 | case、rank、trace span、关键路径节点数、AIC/AIV 占用、调度器占用、PMU 状态、进行中实验数 |

闭环被做成了产品约束，而不是提示语：

- **每轮只验证一个假设**：实验记录同时只允许一条 `open` 实验。已有进行中实验时，点开另一条瓶颈只会看到拦截提示和
  「回到 Ex」按钮，不给第二个「开始实验」。
- **顺序是固定的**：实验步进器必须先记录正确性，才出现性能输入框；两者齐了才出现「保留 / 回退」，
  并在决定前再显示一次该条目的约束与风险。
- **每条结论都带约束与风险和验证标准**：瓶颈条目由 `证据 → 优化动作 → 约束与风险 → 验证` 四段组成，约束与风险写的是这条优化动作的代价
  （例如 prefetch 占 SDMA、提前 dispatch 以吞吐换时延、PMU-on 不能与 PMU-off 基线比较）。

---

## 瓶颈 → 屏幕：证据是怎么对上的

点一条瓶颈之后，中间不会只是「换了个页签」。**证据条**会钉在中心区顶部，把这条结论和屏幕上的具体对象绑起来。
它只有两行，没有解说：

```
┌ C1  集合点等待 1791.82 us…   L2 发现 → E2E 可验证原因 → L1 证据不足   [聚焦证据 ✓] [退出] ┐
│ 证据 4  ①cp_token_allgather_payload_wait 1080.52us  ②o_group_a2a_wait 663us … │
└──────────────────────────────────────────────────────────────────────────────┘
```

三处用**同一套编号**串起来：

| 位置 | 表现 |
|---|---|
| 证据条的 chip | `① ② ③ ④`，点一下把时间轴缩放到那个对象上 |
| 中心画布 | 关键路径 ribbon 多出一条「证据」行，泳道里对应的块加白框 + 同号圆标；非证据对象在「聚焦证据」开启时压暗到 16% |
| 右侧 Inspector | 证据小节标题变成「N 项 · 已在中间标号」，每条证据行带 `标号 1 / 2 / 3`，整行可点，跳到同一个对象 |

编译器层的链可验证原因（C2 / C3 的 `root` 一格）没有画布，改成**表格行高亮**：`is-subject` 行加主色左条和淡底，9 / 12 行一眼可辨。
F10 这种「缺席证据」的条目不给 chip，明确写出「本页没有可标记的对象」。

`聚焦证据` 只压暗、不隐藏——被压暗的东西仍然可悬停、可点击，避免把上下文一起删掉。
`退出` 清除瓶颈上下文，回到自由浏览；切页签不会清掉它，因为一条瓶颈常常要跨两层看。

### 页面上不写解释

这是工具不是教程：中心区没有任何叙述性段落，五个页签下的 `<p>` 数量是 0 / 0 / 4 / 0 / 0
（L1 的 4 条是评估判定，全部是带数字的短读数）。说明性内容只存在三处，且都是可执行的：

- 瓶颈条目的 `优化动作 / 约束与风险 / 验证` 三张卡——这是要照着做的动作，不是背景介绍
- 编译提示 Inspector 的「处理」两张卡——`减少同驻 tile，而非调大 stage` 这类一句话结论
- 实验记录的拦截消息——`E1 进行中 · 结论后才能开下一个`

页签名不重复出现在页签下面：工具条只放控件（编译器的 `Pass 轨迹 / 流水深度 / 搬运粒度` 子切换、
L2 的泳道 / 着色 / 叠加、L1 的 kernel 选择、L2 / L1 的 rank 选择），没有控件的页签（E2E、ISA / 布局）
工具条整行隐藏。

### E2E 没有 rank 切换

L2 和 L1 整页都属于一个 rank——泳道、关键路径、任务表、占用率全部换掉，所以工具条上有 rank 选择。

E2E 不是：它的职责就是**比较**。四个区块里三个本来就同时列两个 rank（`每 rank / 每次调用` 4 行、
`测量一致性校验` 2 行），只有 `调用剖分` 过去跟着全局 rank 走——切一次 rank，整页只有那一块的数字变。
现在 `调用剖分` 也改成两列并排、**共用一个分母**，于是 `runner_run.device_wall` 这一行直接读出
`5132.80 / 4017.34`，F3 的 1.28x 偏斜不用切页签就在同一行里。E2E 因此不再依赖 `S.rank`，rank 选择从它的
工具条上去掉了。

`S.rank` 仍然存在，含义变成「带去 L2 / L1 的那个 rank」：在两张表里点行即可选中，被选中的 rank 在
`调用剖分` 的列头和数值上加重，状态条上也始终写着 `rank rank0 inv=2`。

其余位置一律用「标题 + 计数 / 单位」做小标题，例如 `编译 IR 全流程 · 51 / 51 个 Pass 改动了 IR`、
`缺失产物 · 本 dump 不含 PTOAS / VPTO 级记录`。原来的 ISA 空状态是一段散文加项目符号，现在是一张三列表
（产物 / 用于 / 状态），四行全部标红「缺失」。

---

## E2E 页：多级多卡运行视图

### 先性能定界，再下钻

E2E 首屏不再从一组汇总读数开始，而是用一个 **Serving / Host / Device 性能定界台** 回答“应该先去哪里调”：

页面分成四个工作态：默认的「性能定界」只保留主判断；「Serving / Host」放 WorkerProcess、lane 与轮次开销；「Device 轨迹」放多卡时间地图、构成带和偏斜散点；「原始测量数据」才显示原始采样、剖分与测量一致性校验表。这样表格是下钻材料，而不是首屏内容。

「性能定界」的主体是一张运行流图：`E2E Request → Serving → Host → Device`。节点展示各层已采集的关键时延，边展示请求数量、队列等待与 bind / H2D 状态；主导路径用高亮线连接。点 Serving / Host / Device 节点即可进入对应工作态。未采集的 `device_wall_us` 仍以虚线节点和“未采集”显示，而不会被其他 run 的设备数值填充。

1. **边界时间带**：把同一 scope 的 `host_wall_us`、`device_wall_us` 放到同一刻度，并把端到端时间单列为参照；端到端时间不能与两者相加。若数据跨 scope，明确写出，避免错误比较。
2. **WorkerProcess 任务范围图**：每个 Serving worker task 一行；左侧是任务数，横线是 `Min → Max`，圆点是 `Avg`。多个 WorkerProcess 时，Count 或 Max 相比同组中位数偏离 25% 以上，优先判定为 Serving 分配 / 尾部阻塞问题。
3. **每轮 Host 开销矩阵**：按 invocation 展开 `bind / H2D`、`compile / register`、结果拷回，以“重复 / 仅首轮 / 未发生 / 未知”呈现，不把时序问题堆成文字列表。

性能定界结论映射到下一步：

| 信号 | 性能定界 | 下一步 |
| --- | --- | --- |
| `host_wall_us` 主导 | Host 搬运、注册或编排 | 常驻 weights / KV cache / workspace，register-once、dispatch-many，然后验证 Host / Device 分离是否符合预期 |
| 某个 WorkerProcess 的 Count 或 Max 偏离 | Serving 调度或尾部阻塞 | 排查请求分配、队列与对应 worker 的下游依赖 |
| `device_wall_us` 主导 | Device 执行路径 | 进入 L2，采集 Chip Swimlane 与依赖图 |

`serving-trace-data.js` 汇总了 `serving-strace-swimlane(2).json` 的实测数据：一个完整请求的 E2E、单个 `serving-worker` 的 WorkerProcess task 统计、`scheduler.wait_worker_output` 的 54 次等待统计、16 路 NPU lane 的 55 次 `chip.run`，以及四个有代表性的 bind / prebuilt 轮次。它还带来一个清晰的 Host lane 尾部信号：`device 0` 的平均 `chip.run` 为 302.6 ms，其余 lane 大多约 224–225 ms。

该文件仍**没有**独立 benchmark 的 `device_wall_us`、H2D 和明确的结果拷回事件；页面把这些显示为“未采集”，而不是沿用别的 run 的数值。完整请求 E2E 与每 decode step 的 Host wall 也不属于同一 scope，首屏会显式说明这一点。

接入真实数据时，可在 case JSON 中添加可选的 `e2eTriage`；缺失字段会继续使用上述明确标识的 fallback：

```js
e2eTriage: {
  benchmark: {
    e2e_wall_us: 159091,
    host_wall_us: 127060,
    device_wall_us: 4017,
    source: "independent benchmark"
  },
  workers: [
    { id: "WorkerProcess-01", count: 104, avg_us: 146800, min_us: 138200, max_us: 157600 }
  ],
  workerTasks: [
    { id: "WorkerProcess.prepare_decode", count: 53, avg_us: 90174, min_us: 75236, max_us: 156732 }
  ],
  lanes: [
    { id: "device 0", count: 55, avg_us: 302602, min_us: 65898, max_us: 375424 }
  ],
  rounds: [
    {
      id: "inv 1",
      bind_h2d: { state: "repeat", us: 3106, source: "Serving Strace" },
      compile_register: { state: "once", us: 3, source: "Serving Strace" },
      result_copy: { state: "repeat", us: 812, source: "Serving Strace" }
    }
  ]
}
```

`state` 支持 `repeat`、`once`、`none` 与 `unknown`。

### GBS256 Serving benchmark case

左上角 case 菜单额外提供 `serving_gbs256`，数据来自 `dsv4-flash-dspark-a3-dp4tp4ep16-gbs256-in256o64.zip` 中的 `result3.json`。它是一个独立的请求级 case：256 并发请求全部完成，显示请求 / 输出吞吐，以及 TTFT、TPOT、ITL 的 Mean、P50、P99。随包 Chrome/PyTorch trace 已确认含 CPU op、`aten::copy_`、同步与 gloo 通信事件，但没有同 scope 的 Host / Device wall、WorkerProcess 统计或可归因 H2D / D2H，因此该 case **只开放 E2E**，不能下钻到 L2、L1、编译器或 ISA，也不会复用其他 case 的设备数据。

### 多级多卡运行投影

完成性能定界后，E2E 继续提供三层可下钻的多卡运行数据：

1. **执行时间地图**：每一行是一张卡，横轴为该卡的设备 trace；每个 span 是一条真实 task，关键路径以描边高亮。点任一 span 可进入该算子的 L1 视图。
2. **阶段工作构成带**：每卡一条 100% 堆叠带，宽度表示 trace core-time；用颜色快速对比 Attention、通信、MoE 等阶段的工作构成，点 segment 下钻到该阶段最长 scope。
3. **算子偏斜散点**：横纵轴分别是两张卡的 scope core-time；右上角是共同热点，偏离对角线的是单卡偏斜，点位可继续下钻。

现有 dump 没有“模型阶段 / 算子归属”的字段，因此时延、trace 和利用率仍是实测；**阶段归类**在 UI 中明确标为「规则归类」，是为 demo 补齐的可替换 mock，不会制造假的性能数值。接入完整 JSON 时，在对应 case 上附加下面的可选字段即可覆盖默认规则：

```js
e2eRuntime: {
  operators: {
    "r2t93": { stage: "attention" },
    "r2t94": { stage: "attention" }
  }
}
```

支持的 `stage` 是 `input`、`attention`、`communication`、`moe`、`ffn`、`runtime`。未映射的 task 会继续使用 fallback，不会丢失。

## 函数汇总：Σ = 重复 × 宽度 × 均值

「执行特征来自单次 Block 时长偏高、重复次数偏高，还是离散度高」这三项要能分开，Σ 就得先拆对。

第一版我拆成 `块数 × 均值`，错了：**块数把两件事混在一起**。

```
fa_fused    72 块  =  重复 1  × 宽 72 核     ← 一次铺开占满 72 核，不是调用了 72 次
up_proj     85 块  =  重复 85 × 宽  1 核     ← 真的是 85 次 launch
```

按块数算，两个都是「次数多 ×72 / ×85」，而第一个其实是**单次慢**。
所以拆成三项：

| 项 | 定义 | 对应问题 |
|---|---|---|
| 重复 | launch 次数 × 执行波次数（同一批核跑了几轮） | 重复次数偏高 |
| 宽度 | 一次铺开占几个核 | 并行度，不是成本 |
| 均值 | 单块平均时长 | 单次 Block 时长偏高 |
| p90 / 中位 | 分布的宽窄 | 离散度高 |

两个恒等式都精确成立（最大偏差 0.17 us，来自执行波次数的 r2 舍入）：

```
块数 = 重复 × 宽度
Σ    = 块数 × 均值
```

还有一列 `偏离中位` = `Σ − 块数 × 中位`，**带符号**。
为负说明中位高于均值，是少数快块把均值拉低了，不是长尾问题 ——
`qk_pv` 就是 −9667（中位 782 vs 均值 648）。第一版我把它 clamp 到 0，
恰好把更有意思的那种情况藏了。

「执行特征」列给的是**相对本 run 所有 scope 中位数的倍数**，不是绝对阈值：

```
qk_pv               Σ46662  重复   1  宽72核  均值 648.08  单次 Block 时长偏高 ×69     依赖关键路径
kv_score_proj       Σ 5654  重复21.3 宽24核  均值  11.04  重复次数偏高 ×21     观测路径
tp_o_a              Σ 2540  重复   8  宽16核  均值  19.85  重复次数偏高 ×8.0 + 离散度高  观测路径
qr_hadamard_quant   Σ 2265  重复 5.3 宽48核  均值   8.85  重复次数偏高 ×5.3 + 离散度高  都不在 · slack 640
qproj_matmul        Σ 2218  重复 2.7 宽24核  均值  34.65  时长与重复次数均偏高  都不在 · slack 1794
```

### 末列就是这一层证明不了的那件事

`在路径上` 一列来自「关键路径归因」那一层。这不是装饰 ——
`qr_hadamard_quant` 是个 2265 us、离散度 2.35 倍的 scope，看 Σ 排行会想去动它，
但它**两条路径都不在，slack 640 us**：把它优化掉不缩短墙钟。

把「不能单独证明」直接做成表里的一列，比只写一句话更难被忽略。

## 两 rank 怎么对齐的（F3 的推导）

两份 host log 的 `ts=` 来自**同一台主机的 CLOCK_MONOTONIC**（pid 2263908 / 2263922，连号），
所以两份各自从 t=0 起算的设备 trace 可以摆到同一条绝对轴上：

```
rank0  chip.run.runner_run  ts = 1717978439221135 ns
rank1  chip.run.runner_run  ts = 1717978440374926 ns   →  rank1 晚 1153.79 us
```

把这个偏移加到 rank1 各 `*_wait` 的到达时刻上，就得到 **rank0 在该点最多能等多久的上界**
（rank0 可以在 rank1 的数据落地时就被释放，而落地不晚于 rank1 自己走到 wait，所以是上界不是等式）：

| wait | rank0 到达 | rank1 到达（对齐后） | 上界 | 实测 | 占上界 |
|---|---|---|---|---|---|
| `cp_token_allgather_payload_wait` | 268.10 | 1385.33 | 1117.23 | **1080.52** | 96.7% |
| `o_group_a2a_wait` | 3732.34 | 4450.71 | 718.37 | **663.00** | 92.3% |
| `cp_token_allgather_readback_wait` | 1371.56 | 1491.09 | 119.53 | 47.02 | 39.3% |
| `tp_o_rs_wait` | 4779.52 | 4822.25 | 42.73 | 1.28 | 3.0% |
| | | | **1997.86** | **1791.82** | 89.7% |

四个全部落在上界内，两个主导项贴到 92–97%。配合「两卡 AIC busy 只差 0.67%、任务与块数完全相同」，
负载不均的零假设被排除——**rank0 不是慢，是先到**。

这条推导有两处不是实测:

1. 跨 rank 时钟同步是从同一主机的 mono `ts` 推的，dump 里没有显式的同步记录
2. 上界成立只说明「等待可以被Rank 启动偏移解释」，不证明Rank 启动偏移是唯一成因

两条都写进了 F3 的约束与风险，不在正文里冒充结论。Rank 启动偏移本身产生在
`runner_run` 123109.06 us（host 钟）对 `device_wall` 5132.80 us（设备钟）这段未拆解的主机时间里，
dump 内没有更细的 span 可归因。

---

## Ready queue 图表

`shared_ready_queue` 是 trace 里的 counter 事件（`ph: "C"`），每个采样点给出当刻
「依赖已满足但尚未被 AICPU 派发」的任务数，按引擎分三路。rank0 有 1128 个采样点，rank1 有 1016 个。

- **三条线**：AIC（danger）/ AIV（warning）/ MIX（accent），图例里各自带本 rank 的峰值
- **阶梯而非折线**：counter 的值保持到下一个采样点，所以两点之间画成水平段再跳变。
  这跟 `build-data.cjs` 积分 `busyTime` 的口径一致——如果画成线性插值，图和数字就是两套说法
- **悬停读数**：十字准星 + 三个系列的点，tooltip 给出光标时刻 `t`、命中的采样点编号与时间戳、
  这个值**保持多久**、三路各自的值和待派发合计。tooltip 用 `swimlane-task` pattern 的
  `createTooltip / showTooltip / hideTooltip`，只有行内容是本页的
- 图表与 L2 页签共用时间窗口：在 L2 缩放 / 平移后，dock 的刻度、曲线和悬停换算一起跟着变

---

## L2 调度页：时间花在哪个 scope，哪段时间核在空转

这一页替掉的是「Perfetto 打开 36k 事件 + 临时脚本按 kernel 汇总 core-time」那套流程。
右侧面板原本显示 L1 的 kernel 详情，现在换成 L2 自己的三段。

### 统计口径先说清楚

同一个块在 trace 里出现两次(Worker View pid 4 / Scheduler View pid 3)。
面板第一段把两个总量并排列出，并直接点名相加是错的：

```
Worker View     125572 us · 4038 块      ← Scope 性能排序只用这一份
  kernel        108940 us
  setup          16632 us
Scheduler View  194708 us · 4038 块
hand-off 差     +69135 us
⚠ 两边相加得 320280 us —— 这是重复计数，不是总量
```

### Scope 性能排序带 slack，不只是 Σdur

按 callable 归并成 scope，每行给 core-time、占比和 **slack**：

| scope | core-time | 占 | slack |
|---|---|---|---|
| `qk_pv_aic` | 46662 | 37.2% | **0** |
| `csa_merge_pack_publish` | 18597 | 14.8% | **0** |
| `indexer_score_leaf_wave_aic` | 6494 | 5.2% | **0** |
| `kv_score_proj` | 5654 | 4.5% | 172 |
| `qr_rms_norm_quant` | 3183 | 2.5% | 517 |

slack 来自 fanin/fanout DAG 上的前推/后推(ES/EF → LF/LS，用实测 span)，
`slack = LS − ES`。**这是结构 slack，不含资源争抢** —— 两个 slack=0 的任务仍可能在抢同一个核。

没有 slack 只看 Σdur 的话，`kv_score_proj`(4.5%)和 `qr_rms_norm_quant`(2.5%)会排在
前列看着值得动；加上 slack 就能看出它们分别有 172 us 和 517 us 余量，
真正卡住总时长的是前三个 slack=0 的。关键路径上的 scope 左侧有红色条。

### Core 空闲区间

把 run 切成 240 个等宽窗口(rank0 每窗 20.33 us)，统计每窗 AIC / AIV 的核占用。
AIC 与 AIV **同时**低于 15% 的连续窗口合并成一段：

| 窗口 | 时长 | AIC | AIV | 实测核容量占用 |
|---|---|---|---|---|
| 3721–4392 us | **671** | 0% | 2.2% | **1.5%** |
| 1159–1423 us | 264 | 0% | 2.8% | 1.9% |
| 2501–2623 us | 122 | 0% | 2.0% | 1.3% |

rank0 共 8 段、1199.6 us，占 span 的 24.6%。最长那段 671 us 里 72 个核只用掉 1.5% 容量，
而 `o_group_a2a_wait` 用 **1 个块**占了 660 us —— 单块任务挡住全部核，
这正是 C1 在 L2 层的具体形态。

点任意一行会把泳道时间窗收到那一段。

### 下钻可返回

scope 行和空转行都会改写时间窗，所以两种下钻都留了回路：

| 动作 | 改了什么 | 面包屑 | 返回方式 |
|---|---|---|---|
| 点 scope 行 | 面板 → kernel 详情 + 时间窗 | `← Scope 性能排序 / csa_merge_pack_publish` | 点面包屑，或 Esc |
| 点空转行 | 只改时间窗，面板不动 | `← 恢复时间窗 / 3587–4526 us` | 同上 |

返回时时间窗恢复到下钻前的值（不是重置成全量）。连续下钻不会覆盖最初的返回点。

### 泳道上的占用率色带

工具条上 `着色` 是一个开关。**关掉之后所有任务条变中性灰**（`--surface-4`，
深色 `#313131` / 浅色 `#CCCCCC`），对比度全部让给占用率色带、空转底色和关键路径，
图例也跟着换成「任务（配色已关）/ Core 空闲区间 N 段 / 关键路径 N 节点」。

泳道顶部固定两条色带(AIC / AIV)，透明度跟着每窗占用率走；
被判定为空转的窗口整列打上 warning 色底、两侧虚线、上方标 `671 us 空转`。
不用读 72 条泳道就能看出机器什么时候闲着。

**校验**:把 240 个窗口的占用率取均值，必须等于泳道法算出的平均占用 ——
rank0 `32.22% / 37.50%`，qwen3_14b_fwd（TP=1 prefill）`73.87% / 7.16%`，两边逐位相等。

---

## 三个 case

顶栏的 case chip 打开切换菜单。几份 dump 不是同一个程序,也不带同样的产物 ——
菜单里每一行直接标出它能回答哪几层:

| | decode_csa | qwen3_14b_fwd | serving_gbs256 |
|---|---|---|---|
| 模型 | DeepSeek V4 flash_dspark | Qwen3 14B, 40 层整网 | DeepSeek V4 Flash w8a8 |
| 层级 | L3,2 rank | L2,TP1 / TP2 × prefill / decode | 请求级 benchmark |
| 采集 | 2026-09-03 | 2026-08-14 | 2026-09-28 |
| span | 4879.82 / 3774.04 us | 44.47 ms（TP=1 prefill，另三份见下） | 27.4 s |
| 任务 / 块 | 84 / 4038(每 rank) | 684 / 15228（TP=1 prefill） | — |
| AIC / AIV 占用 | 32.2% / 37.5% | 73.9% / 7.2% | — |
| **E2E** | ✅ host STRACE,2 次调用 | ✅ **torch profiler**（无 host STRACE） | ✅ 请求级 |
| L2 调度 | ✅ | ✅ 仅 TP=1 prefill 有任务名与依赖 | ❌ |
| L1 / L0 | ✅ | ✅ + **实测 PMU** | ❌ |
| 编译器 | ✅ 52 pass,PH001 + PH-MR-001 | ⚠️ 42 pass,**来自另一次构建** | ❌ |
| **ISA / 布局** | ❌ 无 PTOAS 产物 | ⚠️ 38 个 .pto + .cpp,**同上** | ❌ |
| 瓶颈条目 | 3 条链 + 2 条待归因信号 | **1 条链** + 9 条待归因信号 | — |

**不能拼在一起读**:不同模型、不同卡数、不同采集时间。

### 一个 case 里的四份采集

`qwen3_14b_fwd` 不是一次采集，是**同一张 40 层图的四份**：两个采集脚本 × prefill / decode。
它们能回答的问题不一样，所以采集是一个**贯穿 E2E / L2 / L1 的选择**，选定之后三层都跟着它走
（工具条的「采集」下拉、Explorer 的运行树、E2E 页顶部的采集卡片是同一个状态）。

| | TP=1 prefill | TP=1 decode | TP=2 prefill | TP=2 decode |
|---|---|---|---|---|
| 数据集校验 | ❌ | ❌ | ✅ | ✅ |
| span | 44.47 ms | 31.99 ms | 328.18 ms | 330.05 ms |
| 任务 / 块 | 684 / 15228 | 10607 / 18754 | 643 / 643 | 643 / 643 |
| 用到的核 | 60 / 60 | 60 / 60 | **6 / 60** | **6 / 60** |
| 平均忙核（核时/墙钟） | 17.64 | 19.43 | **1.00** | **1.00** |
| merged swimlane | ✅ | ✅ | ❌ | ❌ |
| 任务名 | ✅ 采集自带 | ❌ FuncId 全 −1 | ⚠️ 按已校验层布局还原 | ⚠️ 同左 |
| 依赖图 | ✅ deps.json | ❌ | ❌ | ❌ |
| 40 层还原 | ✅ 40 × 17 | ✅ 40 × 265 | ✅ 40 × 16 | ✅ 40 × 16 |
| 头开销 NoC / dcci 分离 | ❌ 记录只剩 62 条 | ✅ 18754 块 | ✅ 643 块 | ✅ 643 块 |

数据集自己的 `.gitignore` 把 `tp1/` 与 `scratch/` 列为历史或临时生成物，README 只对 `tp2/` 给了验收标记。
界面不抹平这件事：采集卡片、工具条徽标、状态条都写「已校验 / 未校验」，`TP 对照`页的最后一条结论
直接写「这张表能定方向，不能当作 TP=1 / TP=2 的性能结论」。

### 缺席是一种状态,不是错误

只有 TP=1 prefill 带任务名和依赖图，所以只有它进 `data.js` 的任务级机器。切到另外三份采集时:

- **L2 的「泳道」页签改名成「核占用」**：没有 merged swimlane 就没有 Worker / Scheduler View，
  画的是块级占用（一行一个真实物理核），并写明「没有任务名与依赖，所以这里不给关键路径与依赖连线」
- **L1 的「单核流水」整页让位**：本层要落到一个具体 kernel 上，这几份采集落不到，
  页面给出原因 + 两个出口（看 PMU 实测 / 切回 TP=1 prefill），不画空表
- **Inspector 多一段「本次采集」**，写当前这份的 span / 任务 / 核 / 层结构 / 有没有依赖图；
  不是主采集时再加一张卡，说明下面的 scope 与关键路径来自 TP=1 prefill
- **底部 Dock** 的标题直接写「tp1:prefill 的调度器（TP=2 · decode 无 merged swimlane…）」，
  不让另一份采集的调度器数据看起来像当前这份的
- **E2E 页签有真内容了**：这个 case 仍然没有 host STRACE log，但有同一程序的 torch profiler 采集，
  见下面「E2E 换了一把尺子」

### 编译产物来自另一次构建

2026-08-14 那次采集**没有 `passes_dump/`**，所以「编译器」与「ISA / 布局」两层读的是
`_jit_decode_fwd_layers_20260625_184941`（2026-06-25 构建）。这是一个显式选择，不是回退：

- `build-data.cjs` 里新增 `CASE.compileRoot`，编译侧的 `passes_dump` / `ptoas` / `kernels` /
  `orchestration` / `report/perf_hints.log` / `kernel_config.py` 全部走第二个根目录
- `case.compileSource` 带上那次构建的目录名、日期和一句话：
  「行号与 kernel 名不能直接对到本次 trace 上」
- Explorer 的 `artifacts` 节点下多一行 `↑ 来自 decode_fwd_layers_2026… 2026-06-25`
- `incoreScopes` 改从本次运行的 `name_map` 读（46 个），不用旧构建的 kernel 清单冒充
- `binary_context.json` 用的是**新构建**的（`pto-isa 83d01313…`、`tensormap_and_ringbuffer`），
  因为它描述的是这次跑的二进制

新构建自己的 `report/perf_hints.log`（prefill 316 行 / decode 亦有）被单独收在
`qwen3.compile.hints` 里，没有混进编译器层的 16 条——那 16 条属于旧构建，混在一起两边都不可信。

生成器里这是一张 `CAN` 表。门槛有两种:一种是**有没有这份证据**,一种是**这个量级够不够格**——
第二种是这轮补上的,它挡住了「12 us 的 hand-off 被摆在 39% 的发现旁边」这类事:

```js
const CAN = {
  C1: !!(launchSkew && waitTasks.length && launchSkew.checks.length >= 2),
  C2: waveRows.length >= 2 && waveGapSum > 0,
  /* 既要存在,也要够大:≥3% makespan 且 ≥10% 核上时间 */
  C3: !!(stallHost && stallGap >= SPAN * 0.03 && stallHost.setupShare >= 0.1),
  C4: !!longBlock,
  ...
};
const findings = [ CAN.C1 && {...}, CAN.C2 && {...}, ... ].filter(Boolean);
```

### 两份 trace 的格式差异

同一个 `merged_swimlane`,不同采集的 swimlane level 不同,generator 里做了归一:

| | decode_csa | qwen3 TP=1 prefill | 旧的 decode_fwd_layers dump |
|---|---|---|---|
| setup 时间 | 块事件上的 `local_setup_us` | 同左 | **独立的 `setup` 事件** |
| `duration-us` | setup + kernel | setup + kernel | **kernel only** |
| `kernel-duration-us` | 有 | 有 | 无 |
| ready 计数器 | `shared_ready_queue` | `shared_ready_queue` | 多一组 `local_ready_buf_T{0,1,2}`(已过滤) |

生成器按「有没有独立 `setup` 事件」自动分支（`hasOwnSetupEvents`），两边都归一成
`{dur = setup + kernel, kdur = kernel, setup}`,所以"Block 时延分解"在每个 case 下是同一个意思。

---

## E2E 换了一把尺子：torch profiler

`qwen3_14b_fwd` 没有 host STRACE log，所以没有 `chip.run` 的 span 树。它有的是**同一个程序的
torch profiler 采集**：一次 prefill + 四次 decode，整张 40 层图在设备上是**一次** `aicore_kernel_0`
调用。这让端到端有了另一种问法——不是「span 树里哪一段长」，而是「这一步里设备在算还是在等」。

四个页签：

| 页签 | 读的文件 | 回答 |
|---|---|---|
| 步时间 | `step_trace_time.csv` | 每步 Stage = Computing + Free + Preparing，外加融合 kernel 的逐次 PMU |
| 设备算子 | `op_statistic.csv` | 融合 kernel 之外还在设备上跑什么 |
| 主机 API | `api_statistic.csv` | 哪些 CANN 调用吃掉主机时间；Synchronize 单列，不算主机开销 |
| TP 对照 | 以上三者 × 两次采集 | TP=1 / TP=2 并排 |

TP=1 decode 的读数直接给出性能定界结论：

```
Stage 80.82 ms = Computing 32.55 ms + Free 48.28 ms   Free 占比 59.7%
融合 kernel 占设备时间 93.0%，其外有 1597 次框架算子
主机 acl 调用合计 77.25 ms（另有 171 ms 是等设备，不计入）
```

页面给的是「主机受限」这一句，加一句能落地的推论：**在这个比例下，把 kernel 再快一倍也只能省下
步时间的四成**。TP=2 是反过来的（Free 占比 1.5%），同一个 verdict 组件换一边。

口径上被显式挡住的两件事：
`simpler_aicpu_exec_*` 是同一个融合 kernel 的 AI_CPU 启动器，与 `aicore_kernel_0` 是同一份工作的
两条记录，页面把它标成「AI_CPU 启动器」并写明不能与融合行相加；
`Free` 是设备侧空闲、主机 acl 耗时是主机侧读数，两者不是同一把尺子，只能互相印证方向。

---

## L1 / L0 多了一层实测：PMU

原来这一层的流水判断来自 trace 时长和编译信息，是推的。`kernel_details.csv` 里
**每一次** `aicore_kernel_0` 调用都带真实计数器，于是有了一页不用推的：

```
TP=1 prefill   mac 0.240  mte1 0.200  mte2 0.551  scalar 0.203   |  vec 0.017  scalar 0.753
TP=1 decode    mac 0.055  mte1 0.159  mte2 0.428  scalar 0.083   |  vec 0.004  scalar 0.777
TP=2 decode    mac 0.008  mte1 0.014  mte2 0.037  scalar 0.661   |  vec 0.001  scalar 0.699
```

三条都进了性能发现（Q2 / Q3 / Q5），因为它们指向三件不同的事：decode 的 AIC 卡在 **GM→L1 搬运**
而不是算力；40 个 AIV 核忙在**标量**上（vec 0.004）；TP=2 连 AIC 都是 scalar 0.661、mac 0.008——
核是忙的，忙在标量而不是计算。

页面上写死的两句口径：**这些占比是各流水线相对 `aicore_time` 的忙占比，互相重叠，加起来可以超过 1**，
所以只能一行一行读，不能当成时间切分；**`aicore_time` / `aiv_time` 是按核累加的**，会超过单次墙钟，
它们是占比的分母不是时长。状态条的 `pmu` 从 `off` 变成 `torch 侧 on`，并注明泳道那一轮没有 PMU，
两者不能比墙钟。

---

## 40 层：把重复摊开

程序是 1 个 embed + 40 个同构 transformer 层 + 一段尾巴。`decode_fwd_layers` 这个名字本来就指这件事，
但旧 dump 里看不出来。现在四份采集都能把层还原出来，并且**不是靠猜**：

层与层之间，AICPU orchestrator 会提交若干个不落到 AICore 的循环控制任务。于是按提交序排好之后，
相邻 AICore 任务的 `submit_idx` 会出现一个固定的跳变，**恰好 40 次，且两次之间恒隔 N 个任务**。
这 40 个跳变就是层边界。四份采集各自解出来：

```
TP=1 prefill  跳 5，每层 17 个任务，环外 1 前 + 3 后 = 684
TP=1 decode   跳 4，每层 265 个任务，环外 4 前 + 3 后 = 10607
TP=2 prefill  跳 3，每层 16 个任务，环外 1 前 + 2 后 = 643
TP=2 decode   同上
```

TP=2 的 40 × 16 与采集脚本自己校验过的布局完全一致（`_LAYER_TASK_LAYOUT`，16 步的 AIC / AIV 次序
逐位相符），所以那两份采集虽然没有任务名，**步名是可以还原的**——页面标注来源是「按采集脚本已校验的
40×16 层布局还原步名」，不冒充采集自带。

### 层窗口不能用 min(start) 算

第一版用 `max(end) − min(start)` 当层窗口，TP=1 decode 的结果是 869 us 到 19250 us、离散度 175%，
一看就不对。原因是**每层都有少量任务被提前派发**（scheduler 的 `early_dispatch`），最远的提前了 9.8 ms，
`min(start)` 描述的是这几个掉队的，不是这一层。

改成按起始时间里**前半段最大的那个间隔**切（且要求这个间隔明显大于该层的中位间隔），
得到「层主体」窗口，同时把提前派发的任务数和提前量单独报出来：

```
             层窗口中位数   最快 / 最慢      离散度  每层提前派发
TP=1 prefill    1.04 ms    974 us / 1.13 ms  15.3%   0 个
TP=1 decode    735.6 us    635 / 780 us      19.7%   1–10 个，最远提前 9.8 ms
TP=2 decode    6.69 ms     6.63 / 7.09 ms     6.8%   2 个
```

画法照抄数据集自己的 PNG：每行从该层主体起点重新计时，共用同一毫秒刻度，竖线是层窗口中位数；
色块是该层的每个任务（红 AIC / 蓝 AIV），宽度是实测时长，**不用最小显示宽度夸大短 task**。
每层 265 个任务的那份太密，行内只画层窗口本身，并在正文里说明为什么。

层内步骤表按「40 层同一步骤」聚合，TP=2 上一眼能看到 `gate/up` 一步就占掉层内核时的 50.6%。

---

## 头开销：NoC 传播 vs dcci + ack

`l2_swimlane_records.json` 的 `aicore_tasks` 第 6 列是 `receive_to_start_cycles`
（schema 见 `repo/simpler/simpler_setup/tools/swimlane_converter.py`）：AICore 读到新 task_id 之后、
执行开始之前的那段，也就是**本地 dcci + ack 的代价**。有了它，每块的头开销能拆成两段：

```
dispatch → receive   AICPU 写下 task_id 到 AICore 读到  — NoC 传播，硬件侧
receive  → start     dcci + ack                        — 软件侧，可调
start    → end       内核本身
end      → finish    完成回报
```

TP=1 decode 上这笔账很重：

```
NoC 14.60 us  dcci+ack 3.66 us  内核 29.49 us  回报 6.35 us     18754 块
NoC 合计相当于内核核时的 49.5%
```

同一张图在 TP=2 那份采集里被切成 643 个大任务，NoC 只有 0.35 us、占内核核时 0.07%——
**任务粒度决定了这笔固定开销值不值得管**，这正是页面把两者放在一起的理由。

分母是这次采集里所有块的**内核核时合计**，不是墙钟：一个块的头开销只能和内核时长比。
TP=1 prefill 的 `l2_swimlane_records` 只留下环形缓冲尾部的 62 条，样本不足，
页面改用 merged trace 的 Scheduler View 给一个不可再分的 `dispatch → start` 合计，并写明原因。

---

## 数据来源：每个数字都来自这次执行

`data.js` 由 `build-data.cjs` 从 dump 目录直接生成，没有任何建模、估算或造数。

```bash
node Design/operator-tuning-console/build-data.cjs
```

| 产物 | 提取出来的东西 |
|---|---|
| `distributed_meta.json` | 55 个绑定参数的 shape / dtype / 方向 → Case fingerprint |
| `dfx_outputs/rank{0,1}/d0/host.*.log` | STRACE host span（bind / runner_run / device_wall / graph_build / sched / orch）→ E2E 剖分；`ts=` 还用来对齐两 rank，见下 |
| `dfx_outputs/rank{0,1}/d0/merged_swimlane_*.json` | Worker View（pid 4）每块的 `duration-us` / `kernel-duration-us` / `local_setup_us` / CoreId；Scheduler View（pid 3）同一块的 `dispatch-time-us → finish-time-us`；AICPU scheduler phase；`shared_ready_queue` 计数器；dependency / hb_violation flow |
| `dfx_outputs/rank*/d0/deps.json` | `block_num` / `scope` / `early_dispatch` / 每个任务的绑定张量 |
| `dfx_outputs/rank*/d0/name_map.json` | 64 个 callable id → 名字 |
| `report/perf_hints.log` | 230 条 perf hint：197 条 PH001（搬运末维粒度，累计 261 次命中）+ 33 条 PH-MR-001（软流水深度回退） |
| `passes_dump/` | 52 份 IR dump 的行数、DSL / 内存空间计数；每个相邻快照的真实增删行、受影响函数和最多 6 个改写片段；AutoTileMatmulL0 的真实 L0 tile 形状与 55 个 pipeline 站点 |
| `next_levels/.../binary_context.json` | platform、pto-isa revision、runtime 名与 revision → 工具链指纹 |

### 两个视角，不要混

Trace 里同一个块有两份记录，工具把它们分开显示，因为它们回答的是不同问题：

- **Worker View（pid 4）**：核上发生了什么。`duration − kernel_duration = local_setup`。
- **Scheduler View（pid 3）**：AICPU 看到的 `dispatch → finish`。它减去核上时长就是领取与依赖等待。

L1 视图的「Block 时延分解」就是这三层：`kernel` → `+ setup` → `+ hand-off`。

### 关键路径怎么算的

用每个任务的 `fanin-hint` 建图，按实测 start 排序做 DP，取加权最长链（权重是任务自身的 span）。
rank0 得到 33 个节点、链上 span 合计 4985.0 us；链上正向间隙 378.3 us，重叠 539.7 us
（重叠为负间隙，说明后继在前驱最后一块结束前就起来了）。

### 测量一致性校验

E2E 视图会把 host 报的 `device_wall.sched` 与设备 trace 的跨度对齐，确认这份 trace 属于哪一次调用。
两个 rank 都命中 `inv=2`，偏差 < 1%。不测量一致性校验就无法保证「看的 trace 和读的数字是同一次运行」。

---

## 诊断路径（全部由数据推导，非人工填写）

一条**链**是一次完整的定位：在某一层看到发现，往下一层追，最后要么落到一个可编译验证的
源码点，要么**明确证据不足**并说出缺什么数据。判一条读数是不是瓶颈只有一条标准：
**它能不能从这一份 run 里折算出 makespan 归因**。折不出来的进待归因信号，并写清折不出来的原因。

下表是 **decode_csa** 的 3 条可行动诊断路径。

C1 的 `*_wait` 证据在 L2 可见，但其主归属是 **E2E**：归因与改动点都在跨 rank 的 Host
启动 / 下发对齐。因此它不计入 L2 队列；从 L2 的关联任务进入 C1 时，默认回到 E2E 的归因证据。

| ID | 链路 | 结论 | 归因 | 证据不足 |
|---|---|---|---|---|
| C1 | L2 → E2E → *L1 证据不足* | 集合点等待 1791.82 us，可由 Rank 启动偏移解释 1153.79 us | 36.72% | L1：4 个 wait 全是单块单核，核上没有可调对象 |
| C2 | L2 → L1/L0 → 编译器 | 多执行波次小块任务有 1907.52 us 是块间间隙，不是核上工作 | 39.09% | 编译器：`decode_compressor_ratio4.py:110` 的 MemoryReuse |
| C3 | L2 → L1/L0 → 编译器 | `csa_merge_pack_publish` 每块 224.06 us 在核上空等生产者 | 11.22% | 编译器：`decode_csa.py:397`，但「放得下却降级」本身仍需复现 |

三条在时间上仍可能重叠，
所以**归因不可相加** —— 队列头上就这么写着。

qwen3_14b_fwd 只生成得出 C4 一条，理由见上面的「三个 case」；它另外带 5 条由 qwen3 层生成的待归因信号（Q1–Q5）。

### 每条链的三个量

| 链 | 发现层量的是什么 | 继续分析层量的是什么 | 可验证原因 / 证据不足 |
|---|---|---|---|
| C1 | Σ(`*_wait`) = 1791.82 us | rank 启动偏移 1153.79 us，两个主导 wait 贴上界 96.71% / 92.29% | 核上无对象 → 不继续分析 |
| C2 | Σ(span − 执行波次数 × 块中位) = 1907.52 us | 每块 dispatch 0.647 + complete 0.855 = 1.5 us，2368 块 / 3 线程 ≈ 1184 us（占间隙 62%）；Right 可用 64 KB，一级 stage 就要 32 KB | depth 2→1 是算术上被逼出来的 |
| C3 | trace 自带 `hb_violation` 2634.28–3018.5 us，正是生产者未完成 | 核上 387.43 us 里 224.06 us 不是 kernel | Vec 可用 184 KB、两级只要 64 KB，仍被降到 1 |

## 待归因信号（真实读数，但没有 makespan 归因）

这里只保留能辅助链上决策的待归因信号；无归因的 L2 观察不会进入队列。

| ID | 读数 | 为什么不计入性能发现 |
|---|---|---|
| H1 | 搬运末维 < cache line 261 次 | 131 次末维已 ≥ 半条 line；28 次是 `fp32[1]`/`int32[1]` 标量访问，padding 补不成一条 line。真正可调的 102 次里只有 11 次落在链上任务的源码邻域。而且 PH001 是静态提示，本 run 没有 MTE 级计数 |
| H2 | `qr_hadamard_matmul` Block 时长离散度 7.82x | 离散度只值 17.18 us；该任务 span 152.54 us 里的大头是 125.66 us 的**间隙**，已归入 C2。它也不在依赖关键路径上（slack 640.46 us） |

---

## 片上内存容量评估

L1 视图里的评估不是通用公式演示，它对着**本 run 的实测上限**校验：

- `Left = M × K × bytes(AB)`，`Right = K × N × bytes(AB)`，`Acc = M × N × bytes(Acc) × live`
- Left / Right 的可用空间取本 run MemoryReuse 报告里的 `65536 B`；Vec 侧是 `188416 B`
- Acc 不报上限，所以只给「本 run 出现过的最大 Acc tile = 128 KB」作为**下界**，并明确说明超过它属于待验证
- 末维检查：`N × bytes(AB)` 对 512B cache line；不足时给出该 dtype 的元素倍数（INT8 512 / BF16·FP16 256 / FP32 128）

默认值就是 dump 里真实存在的那条 K 循环：`Left INT8[16,64]` + `Right INT8[64,512]` + `Acc INT32[16,512]`，`stage=2`。
在这个配置下 `stage × max(L,R) = 65536 B` 正好等于 free，但本 run 的 `qkv_proj_rope.py:375` 仍然只放下 1 个 buffer
——同驻 tile 会分走这块空间。评估因此把「刚好等于上限」判为**放不下**，而不是刚好通过。这是用实测反例校准的判据，
不是理论公式。

点右侧「AutoTileMatmulL0 dump 里真实出现的 L0 tile」任意一行，可以把该形状回填到评估。

---

## 关键路径归因：照着 simpler 的算法重写了一遍

原来这里只有**结构 slack** —— 在 `deps.json` 上前推 ES/EF、后推 LS/LF。
它能说「这个 scope 动了能缩短总时长」，说不了「这段时间到底在等什么」。
面板里那句「不含资源争抢」就是在认这个账。

`repo/simpler/simpler_setup/tools/critical_path.py` 正好补这一块。把它的算法搬了过来。

### 两条路径，不是一条

**静态 CPM** —— 依赖决定的延迟下界（无限核）。

依赖边先按实测时间戳过滤，只有**真的先完成**的才算 happens-before：

```js
if (en[ptag] <= st[t.tag] + TOL && st[ptag] < st[t.tag]) keep.push(ptag);
```

`start(前驱) < start(本节点)` 这个严格条件让边保留是反对称的 ——
**即使时间戳打平，保留下来的图也可证明无环**，后面的 DP 和反向走查才安全。
decode_csa rank0 上 133 条边留 126、弃 7。

**观测路径** —— 从最后完成的任务往回「归因」走，每步在两类前驱里挑卡得最紧的：

| 前驱 | 记为 |
|---|---|
| 数据依赖（happens-before 边） | `data-wait` |
| 同核资源（这条泳道上此前被释放的最晚时刻，running max） | `core-wait` |
| 一个都没有 | `front-gap` |

同核前驱用 running max 而不是「前一块」，所以流水重叠的切片也算得对。

### 第一版关键路径是错的，这次修了

demo 从第一次提交（2026-09-21）起就有一条关键路径：在 **原始 deps 图**上走最长链。
它把每条依赖边都当成 happens-before，于是消费者实际先于生产者结束就开始时
（早发、或者只表达所有权不表达顺序的 lifetime 边），两段时长被相加而不是取其一。

```
decode_csa / rank0
  老：33 节点，chainSpan 4985.02 us  =  makespan 的 102.2%
```

**一个「依赖决定的延迟下界」不可能超过它所界定的墙钟时间。** 查下去，那条链上有
5 条边的后继在前驱结束之前就开始了，重叠合计 539.74 us：

```
r2t53 qk_pv          end=3520.66  →  r2t55 csa_merge_pack_publish start=3018.50   重叠 502.16
r2t12 allgather_push end=288.16   →  r2t13 payload_wait          start=268.10   重叠  20.06
r2t11 rms_norm       end=255.28   →  r2t12 allgather_push        start=245.60   重叠   9.68
...
```

过滤后的 CPM 是 14 节点 / 3066.8 us / 62.85%，而且这 14 个是老的 33 个的**严格子集**
（新独有 0 个）——老的不是算岔，是多吞了 19 个本不该串起来的节点。

现在 `critical` 直接由 `cpath.cpm` 派生，全 demo 只有一条关键路径定义。

### slack 也在同一张图上跑了

修完第一处，第二处不一致立刻暴露：关键路径 14 节点，而 `slack == 0` 的任务有 33 个。
因为 slack 的前推/后推还在用原始 `fanin/fanout` 边——没有真正卡住消费者的边照样把
ES 往前推，slack 因此偏小，太多任务读成零 slack。

改成同一张过滤图后：

```
rank0    路径 14 节点   slack==0  14   onCrit  14
rank1    路径 11 节点   slack==0  11   onCrit  11
device0  路径 14 节点   slack==0  14   onCrit  14
```

**这改变了实际建议。** 第二大 scope `csa_merge_pack_publish`（18597 us，14.8%）
原来标 `slack 0 / 在关键路径上`，现在是 `slack 1341.94 us / 不在` ——
从「动它直接缩短总时长」变成「它不在关键路径上，先看并行度」。
Scope 性能排序里被误标红边框的一共 19 个。

### 下游读数也跟着改了（这一步之前漏了）

改完数据层不等于改完。过滤后 CPM 从 33 掉到 14 节点，而观测路径是 32 节点 ——
页面上两个东西都叫「关键路径」，于是直接自相矛盾：

```
L2 关键路径归因表   第一行    q_rope_prepare   stall 88.1 us（全路径最大）
L1 页签         同一任务   「不在关键路径上」
```

观测路径 32 个节点里有 23 个不在 CPM 上，所以这不是偶发。现在两条分别命名：

| 名字 | 是什么 | 动它得到什么 |
|---|---|---|
| **依赖关键路径**（静态 CPM） | 依赖决定的延迟下界 | 降下界 |
| **观测路径**（反向归因） | 计算 + stall 精确铺满 makespan | 去掉 stall |

逐项改动：

- **泳道顶部的色带**改画观测路径 —— 它本来就画在真实时间轴上，就该显示铺满这根轴的那条。
  原来画 CPM，表头却写「走完 4879.8 us」，而那条链只有 3066.8 + 185.8 = 3252.6 us。
  现在表头是 `计算 4597.5 us + stall 282.3 us = 4879.8 us`，
  CPM 节点在条上用红色 tick 标出，行标签从 `CRIT PATH / GAP` 改成 `OBS PATH / STALL`。
- **L1 任务头和 Inspector** 报两条的归属：`在 依赖关键路径 1/14 · 观测路径 6/32`，
  或者 `两条路径都不在`。
- **工具栏「只看关键路径」开关**变成三选：全部任务 / 观测路径 / 依赖关键路径。
  原来那个开关在修复后只剩 14 个任务，而且没说是哪一条。
- **F1 标题改了**。原来是「通信等待独占关键路径 36.72%」—— 那个 36.72% 是
  wait span 占 **makespan**，走的是观测路径；4 个 `*_wait` 里只有 2 个在 CPM 上。
  现在是「通信等待占 makespan 36.72%」，claim 里把两条的归属分别说清。
- **F1 的证据来源改名**：不再写 `critical path (fanin/fanout hints)` ——
  图已经按实测时间戳过滤过，不是原始 hint 图了。证据里同时列出两条路径各自的数。
- **F9 的路径证据**从「在/不在关键路径上」改成 `依赖关键路径第 14 节点 · 观测路径第 16 节点`。
- E2E 对比、ISA、状态栏、Scope 性能排序 tooltip 里所有裸的「关键路径」都加了限定词。
  回归脚本里加了一条检查：页面文本中不允许出现不带限定词的「关键路径」。

### 两条不变量，现在是算出来摆在界面上的

```
✓ 归因闭合   compute + stall = makespan（差 0.00 us）
✓ 依赖下界   CPM ≤ makespan（3067 ≤ 4880 us）
```

第二条就是这次被违反的那条。它一直成立不了，只是**之前没有人检查**。
过滤生效的另一个证据：路径上的残留重叠从 539.74 us 变成 **精确 0**。

### 归因闭合检查

正向 frontier sweep 保证 compute + stall **精确铺满** makespan：

```js
const gap = Math.max(0, a - frontier);
const eff = Math.max(0, b - Math.max(a, frontier));
frontier = Math.max(frontier, b);
```

三个 rank 全部 `delta 0.00`。**这个检查不过，下面的逐节点归因就不成立** ——
所以它是算出来摆在面板上的，不是拿文字声称的。

```
decode_csa / rank0   4879.82 = 4879.82   ✓
decode_csa / rank1   3774.04 = 3774.04   ✓
qwen3_14b_fwd        44474.42 = 44474.42 ✓
```

### 照搬会错的一处：模型把等待算成了 compute

`critical_path.py` 的 `dur = end - start` 是节点的 **wall span**。
路径上的 `*_wait` 是通信等待，它的 span 占着路径但根本不是计算。直接套判据会得到：

```
rank0  compute 94.21%  stall 5.79%   →  "compute-bound"
```

而这 4597.5 us「compute」里有 **1791.82 us（39%）是 4 个 `*_wait`**。
所以这里把 compute 拆成**真正计算**和**通信等待**，并让它参与判据：

```
rank0   真算 57.5%   等待 36.7%(4 个)   stall 5.8%   →  通信受限
rank1   真算 93.4%   等待  0.1%(2 个)   stall 6.5%   →  计算受限
```

**这和 F3 是两条独立推导，结论一致。** F3 是从 host CLOCK_MONOTONIC 对齐推出
rank1 晚启动 1153.79 us、rank0 在空等；归因走查从设备侧时间戳独立得到
rank0 通信受限 / rank1 计算受限。而且 F1 的 36.72% 与归因的 `waitShare` 精确相等，
4 个 `*_wait` 全部落在归因路径上。

### 两条路径不能互换

静态 CPM 的 14 个节点里，9 个也在观测路径上，**另外 5 个观测路径从不经过**。

- 动只在 CPM 上的节点 → 降**依赖下界**
- 动只在观测路径上的节点 → 去掉 **stall**

面板上用红色左边框区分，并把 5 个 CPM-only 的 tag 直接列出来。
提优化建议时必须说清在动哪一条。

### 容差

参考工具用 2 个时钟 tick。decode_csa 有 `clock_freq_hz = 50 MHz` → **0.04 us**。
qwen3_14b_fwd 的 `l2_swimlane_records` 也带 `clock_freq_hz = 50 MHz`（旧的 decode_fwd_layers dump
没有，那时退回到时间戳精度的 2 个量子 0.02 us）。界面上标明用的是哪一种，不假装有时钟。

### 还没有的

- 参考工具按 `(task, core-block)` 切片建图，这里按 task 建图、用泳道块算同核前驱。
  块级的 core-wait 比任务级更细。
- 没有 `CPM_static.json` / `CPM_observed.json` 那种把 off-path 任务改名的 Perfetto 导出。
- 归因用的 span 含 setup（泳道上块从 setup 起画），参考工具用的是裸 kernel tick。
  对 core-wait 而言含 setup 更对 —— 核在 setup 期间确实被占着。

---

## scope 和 kernel 不是一回事

这两个词在通用 trace 工具里会混成一个，在 PTO 里不是。

```
源码        with pl.spmd(NUM_QK_CORES, name_hint="qk_pv") as qk_tid:
              └─ 一个 InCoreScopeStmt            ← 这是 scope
                 │
OutlineIncoreScopes (09_after_…)
                 └─ Function(InCore) 名叫 qk_pv
                    │
ExpandMixedKernel (23_after_…)
                    ├─ qk_pv_aic  FunctionType::AIC   ← 这是 kernel
                    └─ qk_pv_aiv  FunctionType::AIV   ← 这也是 kernel
                       两个被一个 Group 函数依次调用
```

**一个 scope 编译出 1 个或 2 个 kernel。** 纯 Cube 或纯 Vec 的 scope，
`ExpandMixedKernel` 只把 `FunctionType::InCore` 改成 `AIC` / `AIV`，
不拆不改名；**同时含 Cube 和 Vec 算子的混合 scope 才被拆成两半**，各带 `_aic` / `_aiv` 后缀。

这个 dump 直接印证：

| | decode_csa | qwen3_14b_fwd（TP=1 prefill） |
|---|---|---|
| scope（源码 pl.spmd 区域） | 62 | 18 |
| kernel（`name_map.json` 里的名字） | **64** | **46** |
| 混合 scope（一个 task 两个 FuncId） | 2（`qk_pv`、`indexer_score_leaf_wave`） | 1（`qk_pv_online_phase`） |

qwen3 那一列的差额不等于混合 scope 数：`name_map` 收的是整张 40 层图的 46 个 callable，
而这次 prefill 只跑到其中 18 个 scope。差额只有在同一张图被完整执行时才等于混合数。

### 之前这里是错的

两半**共用一次 Group launch，所以在 trace 里是同一个 `taskId`**，
只有 `event-hint` 里的 `FuncId` 能把它们分开：

```
taskId 8589934645  tag r2t53  FuncId 48+49  72 blocks
taskId 12884901902 tag r3t14  FuncId 42+43  72 blocks
（84 个 task 里只有这 2 个带双 FuncId）
```

原来的 `build-data.cjs` 按 `taskId` 分组、`funcId` 只取第一个事件的，
于是 Vec 侧的 31626 us 被记到了 Cube 侧 `qk_pv_aic` 的名下。现在按 FuncId 重新分组，
每个 task 带一个 `kernels[]`，不变量 `Σkernel.coreTime == scope.coreTime` 在两个 case 上精确成立。

## 引擎配对：AIC / AIV

L2 右栏的「引擎配对」分区回答的是「Cube 和 Vec 各花了多少、混合核的两半谁拖谁」。

```
AIC (Cube)   37740 us · 30.1% · 32.22% 占用
AIV (Vec)    87833 us · 69.9% · 37.50% 占用
Cube : Vec   1 : 2.33

qk_pv                         46662 us
  qk_pv_aiv   V   31626   48/48   最长 881
  qk_pv_aic   C   15036   24/24   最长 830
```

**两侧最长块 830.3 / 881.2 us，而整段 span 只有 886.4 us。**
886.4 ≈ max(830.3, 881.2)，不是 830.3 + 881.2 = 1711.5 —— 所以**两半是并行的**，
24 个 Cube 块和 48 个 Vec 块跑在不同核上。

> 这一段原来写反了：早先的 F9 从「两侧最长块相差只有 1.061 倍」推出「块内串行」，
> 但那个比值恰恰是并行时该有的样子。由于没有块内 PMU / pipe 证据，这个读数不进入当前队列；
> 现阶段只能得出「1 个执行波次跑完，span 就等于一个块」，不能选择具体优化方向。
> 拆两半量出来的数没问题，错的是从这个数推出的结论。

Scope 性能排序也多了一列引擎标记：`C` / `V` / `C+V`，悬停给两侧的 core-time、块数和最长块。

## spmd 展开

`pl.spmd(N)` 把一个 scope 铺到 N 个核上。trace 只记块和 core id，
不记这次 launch 的形状，所以「铺了多宽、跑了几个执行波次、铺得匀不匀」要按 scope 重新汇总。

| 列 | 含义 |
|---|---|
| 核 | 这个 scope 最宽一次展开占了几个核 |
| 块 | 块数 |
| 执行波次 | 块数 / 核数。1 个执行波次 = 一次填满；>1 个执行波次 = 同一批核要跑好几轮，每轮之间有一次完成回收 |
| Block 时长离散度 | 最长块 / 中位块。>2 = 同一次展开里各块负载不均，最慢的那块决定 scope 什么时候结束 |

两个 case 的形状完全不同，这一维一眼能看出来：

```
decode_csa          48 个多核 scope / 14 个单核，最宽 72 核
                    22 个多执行波次 scope，最多 21.33 个执行波次
                    qr_hadamard_matmul  24 核 256 块 10.7 个执行波次 Block 时长离散度 7.82
                    kv_score_proj       24 核 512 块 21.3 个执行波次 Block 时长离散度 2.97

qwen3_14b_fwd       16 个多核 scope / 2 个单核，最宽 60 核
（TP=1 prefill）     6 个多执行波次 scope，最多 1.54 个执行波次
                    rmsnorm        8 核   8 块 1.00 个执行波次 Block 时长离散度 13.03
                    post_rmsnorm   8 核   8 块 1.00 个执行波次 Block 时长离散度  9.94
```

没有值得看的展开时，这一段不拿 `1/1/1` 的行凑数，直接说「本 case 的形状问题在别处」。

## scope → 源码：dump 里没有，但能重建

泳道上的算子名不是编译器发明的。每个外联 scope 都以源码里的
`pl.spmd(..., name_hint="X")` 命名：

```python
# decode_sparse_attn_csa.py:208
with pl.spmd(NUM_QK_CORES, name_hint="qk_pv", deps=[qk_plan_tid, cache_ready_dep],
             allow_early_resolve=True) as qk_tid:
```

从入口 `decode_csa.py` 追传递导入(13 个模块)、索引其中所有 `name_hint`，
就能把 trace 里的 scope 打回源码。**映射的粒度是 scope 不是 kernel** ——
混合 scope 拆出的 `_aic` / `_aiv` 共用一个 `name_hint`，指向同一处源码：

| | |
|---|---|
| 覆盖 | **62 / 62** scope |
| 唯一定位到 文件:行 | **54** |
| 多候选(同名 hint 出现在多处) | **8** |
| 未匹配 | 0 |

编译器加的两级后缀要先折回去：

- `_aic` / `_aiv` —— `ExpandMixedKernel` 拆 mixed kernel。
  `qk_pv_aic` → `qk_pv`。现在 scope 本身就叫 `qk_pv`，这一层是精确命中，不用去后缀
- `_0` —— **同一处源码被实例化两次**，不是两处源码。
  `decode_csa.py` 只 import 了 `compressor_ratio4`，却在 line 353 与 892 各调一次，
  两次 tile 常量不同，实测块数 512 / 256 正好对上 —— 于是有
  `kv_score_proj` 与 `kv_score_proj_0`

另外 `_spmd` 是**前端追踪时追加**的：源码写 `name_hint="csa_merge_pack_publish"`，
`00_frontend.py` 里才变成 `..._spmd`。别拿前端 IR 当源码读。

### 界面上怎么呈现

- L2 Scope 性能排序每行第二列给 `文件:行`，多候选标 `+N` 并染成 warning 色
- L1 kernel inspector 多一行 `源码`；多候选或去后缀匹配的，下面补一张卡说明
- ISA 页签新增 `kernel → 源码` 段，写明**来源不在 dump 内**、依据是什么、
  以及**不能做的事：只给出 scope 写在哪里，不把实测块时长归到某一行**
- qwen3_14b_fwd 的 Qwen3 源码树不在仓库里，同一段显示为缺失并说明可重建

### 泳道图例去掉了

原来按算子着色时画 8 个色块，是 `slice(0, 8)` 的显示上限，不是统计量 ——
62 个 scope 列 8 个既不完整，在 qwen3_14b_fwd 上还会重名(684 个任务只有 18 个
scope，40 层里同名任务反复出现，颜色还完全一样)。

62 路分类本来就没有可读的图例。现在改成一行读数：

```
62 scope 各一色 · 颜色只用于区分相邻块，名字看悬停或右侧排行
```

按引擎着色仍保留 AIC / AIV / MIX 三项 —— 那是真能查的图例。

---

## L2 页的两条联动

### 观测路径 ⇄ chip swimlane

原来只有单向：点泳道里的块 → 上方观测路径条上对应的节点亮起来。反过来点不动 ——
路径条是一张图片。现在两边都是控件：

- **点路径节点** → 选中该任务、Inspector 切过去、泳道**滚到这个节点真正跑的那条 lane**；
  如果节点落在当前时间窗之外，先把窗口挪过去再选，不会选中一个看不见的东西
- **点泳道块** → 路径条上的节点亮起（原有行为）
- 两个画布都画**同一对虚线**：选中任务的 start / end。这才是联动本身 ——
  一段时间区间，两个视图。节点在观测路径上时虚线是 warning 色，不在时是灰色
- 路径条标题右侧实时写出 `已选 第 16/32 节点 · qk_pv`，或者 `已选 xxx · 不在观测路径上`

滚到哪条 lane 不是随便挑的：用的是这个任务**最早开始的那个块**所在的 lane ——
作为消费者，它才是等过的那一个。

### 依赖连线

`deps.json` 里一直有 pred / succ，但泳道从来没画过。现在画了，工具栏「依赖连线」三档：

| 档位 | 画什么 |
|---|---|
| 选中任务 | 选中任务的 pred → 它、它 → succ |
| 沿观测路径 | 路径上相邻节点之间的 31 条边 |
| 关 | 不画 |

**不画全图**。126 条边铺在 72 条 lane 上是一团毛线，不是诊断。

连线的两个端点也不是任务的聚合 `[start, end]`，而是具体的块：

- 生产者一侧取**最后结束**的块（那才是 gate 住下游的那个）
- 消费者一侧取**最早开始**的块（那才是等过的那个）

**消费者的第一个块比生产者的最后一个块先开始时，边画成红色虚线** ——
这就是 trace 自己标的 `hb_violation`，也正是 C3 那条链在讲的事。
它本来只活在文字里，现在在画布上能直接看到：选中 `csa_merge_pack_publish`，
那条指向它的红色虚线就是「消费者在生产者跑完之前就上核了」。
切到「沿观测路径」，32 个节点的 31 条边里有 **6 条**是这种早发。

画不出来的边分两类报，不合并成一个数：

- `N 条端点在本 trace 里没有块` —— deps.json 提到的任务在这份 trace 里根本没有块
  （`csa_merge_pack_publish` 的 6 个前驱里有 4 个是这样）。换视图也不会出现
- `N 条端点被泳道筛选隐藏` —— 块是有的，只是当前「泳道」筛选把那条 lane 滤掉了。
  这个改筛选就能看到

---

## 这一轮：从 10 条读数收敛为 3 条可行动链

原来的队列是 10 条平铺的发现（F1–F10）。逐条对着 `data.js` 的原始聚合复核之后，
**只有 3 条同时具备 makespan 归因和下一步动作**；数据不足的单执行波次长任务不进队列，
其余发现为真但不足以单列为瓶颈。
改动不是重新排版，是把归因重算了一遍。

### 被推翻的三条归因

- **F9「`qk_pv` 的 Cube / Vec 两半块内串行」—— 结论是错的。**
  24 个 AIC 块（最长 830.26 us）和 48 个 AIV 块（最长 881.22 us）跑在**不同核**上，
  整段 span 886.38 us ≈ max(830.26, 881.22)。真串行应该接近 1711.48 us。
  两半本来就是并行的。这个读数仅保留为补采块内证据的依据，不作为当前优化方向。
- **F2「hand-off 比核上计算还贵」—— 归因是错的。**
  trace 自带的 `hb_violation` 标出 `AIC_4→AIV_41` 区间 2634.28–3018.5 us，正好是生产者
  `qk_pv`（到 3520.66 us 才结束）还在跑、而消费者的块已经在 3018.5 us 上核了。
  那 57.8% 的「setup」是**在核上等数据**，不是可复用的准备工作。原来的优化动作
  「把 setup 提到核外复用」是对错误前提开的药。
  顺带发现：这条早发边被 CPM 的时间戳过滤丢掉（`edgesDropped=7`），
  所以它之后整条尾巴的 slack 都是同一个 1341.94 us —— 那个 slack 不能当「可以不管」的依据。
- **F6「调度器平均占用 42.29%」—— 口径是错的。**
  3 条线程都没饱和，平均占用本身不构成约束。它的真实代价是
  `dispatch 0.647 + complete 0.855 = 1.502 us/块`，摊在 2368 个多执行波次小块上 ≈ 1184 us。
  这个数现在是 C2 的继续分析层，不是一条独立发现。

### 被降级的三条

- **F5「261 次末维 < 512B」被夸大**：131 次末维已经 ≥ 半条 cache line；最极端的 28 次 4B
  全是 `fp32[1]` / `int32[1]` 标量访问，padding 补不成一条 line。真正可调的 102 次里
  只有 11 次落在链上任务的源码邻域。
- **F7「ready-but-undispatched 26.57%」立不住**：平均队列深度只有 0.433。
- **F8「Block 时长离散度 7.82x」不是瓶颈**：span 152.54 us（makespan 的 3.13%），slack 640.46 us，
  两条路径都不在。而且「尾块决定 span」算不过来 —— 10.67 个执行波次 × 中位 2.52 us = 26.88 us 的
  工作量下界，span 却是 152.54 us，差的 125.66 us 是**间隙**。
- **F10「0 处 pl.prefetch」** 是缺失项，不是已发生的损失。

### 新增的一条链

F1–F10 里没有人把「间隙」算出来。按「span − 执行波次数 × 块中位时长」逐任务算，
10 个多执行波次任务有 **1907.52 us（makespan 的 39.09%）** 不在算 —— 这是本 case 最大的一块，
而且能一路接到 `decode_compressor_ratio4.py:110` 的 MemoryReuse 上。它现在是 C2。

### 数据结构上的变化

```js
{
  id: 'C2', kind: 'chain',              // 'chain' | 'hygiene'
  cost: { us, share, basis },           // hygiene 恒为 null
  unattributed: '...',                  // chain 恒为 undefined
  chain: [{ level, role, headline, detail, evidence, subjects, chips }],
  terminus: { level, reason },          // 链在哪一层停，为什么
  levels: ['l2', 'l1', 'compiler'],     // 供按层筛选
  rootPass: 'MemoryReuse' | null,       // Pass 视图反向链接用
}
```

`role` 只有四种：`observe`（发现）/ `descend`（继续分析）/ `root`（可验证原因）/ `stop`（证据不足）。
**`stop` 是一等状态** —— 一条走不下去的链要说出缺什么数据，而不是收在一个猜测上。
界面上 `stop` 那一格是虚线边框，点不动，因为它在屏幕上没有对应的对象可标。

读者可以在中心栏的链条上一格一格往下走：点「编译器 · 可验证原因」，舞台会切到编译器视图的
流水深度页签，证据标号重新编成那一层的对象（3 个能接到链上的 PH-MR-001 源码点），
而不是继续标着 L2 的任务。

---

## 诚实的空缺

- **dump 内没有 kernel → 源码映射**。decode_csa 的这一条已由模型源码按 name_hint 重建（62/62 覆盖、54 唯一、
  8 个多候选，见上一节）；qwen3_14b_fwd 的 Qwen3 源码树不在仓库里，仍然缺。重建出来的是「scope 写在哪里」，
  **不是**「哪一行耗了多少时间」——perf hint 自带的行号与它是两条独立证据，不要互相当作确认。
- **没有 PTOAS / VPTO 级产物**。ISA 视图只给这份 dump 能支撑的结论（工具链指纹、布局与内存空间分配、L0 tile 清单、
  512B 约束），并列出需要补齐哪些产物（TileLib 模板选择记录、VPTO 指令排布报告、cycle cost model 预测、PMU counter）
  才能把结论推进到指令层。
- **没有 PMU**。trace 里没有硬件 counter；PMU 打开会改变调度，不能与本基线直接比较。
- **只有 2 次调用**。因此调用采样不显示 mean/median，只显示每次调用的值。
- **只有一次采集**。每个百分比都来自单次 run。同一负载两次采集的 stall 占比能差几个百分点，所以不要拿「各采一次」的两个配置做对比 —— 参考 skill 的原话是 one capture, one sample。另外 makespan 含首轮 warm-up，不是稳态。
- **归因是任务级，不是块级**。参考工具按 (task, core-block) 切片建 happens-before 图；这里按 task 建图，只有同核前驱用到了泳道块。块级的 core-wait 会比任务级更细。

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
`--tc-gutter: 10px`，而带内边距的表面（表格滚动区、canvas、tile 卡、实验记录条目、树行、瓶颈卡）
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
   在此之前它们完全没有样式，`<h3>` 回落到浏览器默认的 16px 粗体 block，导致「性能发现 / 10 / 10」换行。
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

## 右栏的密度：默认开三个，其余折叠

L2 页曾经把 7 个分区全部展开、配 12 段说明文字，右栏要滚 6.3 屏，
而其他每个页签只要 1–2 屏。分区本身不是问题，**一次全给**才是。

现在按「这一页是来回答哪三个问题」排，只开这三个：

| 开 | 回答 |
|---|---|
| 关键路径归因 | 这次为什么花了这么久 |
| Scope 性能排序 | 时间花在哪个 scope |
| Core 空闲区间 | 哪段时间核在空转 |

统计口径、引擎配对、spmd 展开折起来，各占一行，点一下展开，
折叠状态记在 `S.folded` 里，跨页签和重绘都保持。**没有删任何内容。**

说明文字从 12 段降到 4 段。去掉的不是信息，是位置：

- 列的定义搬进表头的 `title=`（slack 怎么算、执行波次和离散度什么意思）
- 推导过程搬进脚注上的 `两条路怎么选 ?` / `怎么算的 ?`，悬停可见
- 归因闭合检查从整张卡压成结论卡里的一行 `✓ compute + stall = makespan`

留下的 4 段卡都是**会让人读错数字的那种**：通信受限的判据、
最长Core 空闲区间是谁挡住的、混合核两半的比值、Worker/Scheduler 重复计数。

表格行数也收了：Scope 性能排序 14→10、Core 空闲区间 8→5、路径节点 10→6，
表头上写清是「前 N / 共 M」，点进去可以下钻。

结果是 L2 从 6.3 屏降到 2.6 屏，和其余页签（1.0–2.5 屏）齐平。

---

## 交互速查

| 操作 | 结果 |
|---|---|
| Inspector 分区标题的 `+` / `−` | 展开 / 折叠该分区（统计口径、引擎配对、spmd 展开默认折叠） |
| 悬停表头 / 带点下划线的字段 | 该列或该行的定义与推导，正文里不再重复 |
| 点瓶颈条目 | 跳到证据所在层级，钉出证据条，给证据对象编号并压暗其余部分 |
| 点证据 chip / Inspector 证据行 | 跳到同一个编号对象；时间轴自动缩放过去，Inspector 保持停在这条瓶颈上 |
| 证据条「聚焦证据」 | 压暗非证据对象（不隐藏，仍可悬停点击）；再点一次恢复 |
| 证据条「退出」 | 清除瓶颈上下文；切页签不会清除，方便跨层追同一条 |
| `1`–`5` | 切换五个层级视图 |
| `/` | 聚焦搜索（kernel / 任务 / 编译提示 / Pass，回车跳第一条） |
| L2 泳道点击 | 选中任务 → Inspector 显示绑定张量、依赖链、关联瓶颈、所在核占用；观测路径条上该节点亮起 |
| L2 观测路径点击 | 选中该节点 → 泳道滚到它真正跑的那条 lane；节点在窗口外时先把窗口挪过去 |
| L2 两画布的虚线 | 选中任务的 start / end，两个画布画同一对；节点在观测路径上是 warning 色，不在是灰色 |
| L2 工具栏「依赖连线」 | 选中任务 / 沿观测路径 / 关；红色虚线边 = 消费者比生产者先上核（`hb_violation`） |
| L2 画布 shift + 拖动 | 水平平移；工具栏 `+` / `−` / `Fit` 缩放，Dock 时间轴跟随同一窗口 |
| Dock「核占用」 | 72 条泳道的占用、空洞数、最大空洞、首末块 |
| Terminal「Problems」 | 230 条编译提示当作 IDE 问题列表，点击跳到对应源码点 |
| Inspector 依赖 chip | 沿 fanin / fanout 在任务图里走 |
| 评估里点 L0 tile 行 | 把该真实形状回填进评估 |

---

## 文件

```
Design/operator-tuning-console/
├── index.html        ide-frame shell 与槽位
├── styles.css        仅页面级布局；颜色全部走 token
├── app.js            状态机、五个视图、canvas 渲染、实验记录逻辑
├── build-data.cjs    从 dump 目录生成 data.js（可重跑）
├── data.js           生成产物，勿手改
└── README.md
```
