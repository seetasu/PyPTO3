# 由 Hardware-Native 没做好引起的八类问题

> 整理日期：2026-10-19
> 数据来源：`github_issues/` 下六个仓库的 Issue 归档（`*_issues.json`，GitHub API 原始字段），已排除 PR，共 **1463 条**。
> 本文回答：**哪些问题是因为「硬件事实没被恰当承载」而产生的，它们长什么样，各有多少。**
> 配套：问题类型 → 维度缺口的映射见 [PyPTO_Hardware-Native内容维度与清单](../Product_Planning/PyPTO_Hardware-Native内容维度与清单_20261019.md) 与 [九域详解](../Product_Planning/PyPTO_Hardware-Native九域详解_20261019.md)。
> 标注：【事实】可核对到 Issue 号与原文；【统计】本文检索方法所得，方法见第 7 节；【推断】基于前两者的判断。

---

## 摘要

**【统计】1463 条 Issue 中，454 条（31.0%）带有「硬症状」**——即静默错算（编译通过、不报错、结果错）或设备挂死 / 超时。其中 73 条仍 open。

**【推断】这 454 条不是「bug 多」，而是一类特定的失效**：它们几乎都不是算法写错，而是**某条昇腾硬件约束被破坏，而整条链路上没有任何一环负责检查它**。按「链路上哪一环缺位」归类，得到八种类型：

| # | 类型 | 一句话判据 | 规模【统计】 | 缺的是哪一环 |
| - | ---- | ---------- | -----------: | ------------ |
| **P1** | **静默错算** | 编译干净、不报错、结果错 | 标题含 `silent` **34**；全文 **239**；「no error/warning」**68** | **检查（C）** |
| **P2** | **设备挂死 / 超时** | 同步是正确性问题，却没有编译期检查 | `507018`/`507015` 错误码 **86** | **检查（C）** |
| **P3** | **规格不可信** | 文档或预算里的数字与硅片不符 | 硬编码 / 规格质疑类标题 **9**，含 1 条 open 的高危 | **可查规格（Q）** |
| **P4** | **硬件旋钮在抽象里消失** | ISA 和 IR 都有这个参数，DSL 没有 | 典型案例少但后果重 | **可表达（E）** |
| **P5** | **跨平台 / 代际 / 仿真器分歧** | 同一段代码换平台就变 | 提仿真器 **166**，明说 sim≠device **20** | **代际变量（维度五）** |
| **P6** | **工具链版本回归** | 同一硬件约束被反复破坏 | 回归措辞 **196**；提 ptoas 版本号 **100** | **跨仓契约** |
| **P7** | **诊断无落点 / 决策不可见** | 报错了，但不在你写的那一行 | 诊断质量类标题 **13** | **归因（A）** |
| **P8** | **性能悄悄劣化** | 没变慢的报告，只有被偶然发现的百分数 | 性能类 **231**（全文） | **检查（C）+ 观测（O）** |

**三条最值得先看的判断**：

1. **P1 与 P2 合计占全部 Issue 的三成，而它们共同缺的是同一样东西**：一条「做错会错 / 会 hang」的硬件约束，在编译期**本来可判定**却没有检查。最锋利的证据是 pypto#1935 的原话——「编译器已经知道启动 block 数和目标 SoC 每种核型的物理核数，**所以这是可静态检查的**」。
2. **规格本身可能是错的。** pto-isa#170（**仍 open**）：Ascend910B1 上 Vec UB 的**实际可用约 184 KB，而不是 192 KB**；PyPTO 的分配器预算不扣这 8 KB，于是放到 188.8 KB 时编译毫无报错、运行产出 NaN。这不是「没把规格给开发者」，而是「**给的规格是错的**」——比第一种严重一级。
3. **仿真器的失效方向是双向的，而且它恰好在最该起作用的地方失效。** pto-isa#149 的原文：CPU sim **有** inactive-lane guard，而 device 路径**没有**——于是 sim 通过、真机静默全零。整个栈把「无硬件也能验证」当作获客设计，但一个比硬件更宽松的模拟器，会把 Hardware-Native 的缺陷**藏起来**而不是暴露。

---

## 0. 口径与方法

**【统计】口径声明**（与本仓库既有分析一致，结论按**下界与量级**表述，不作精确统计）：

1. 匹配**标题 + 正文**。正文含堆栈、环境表、复现脚本，因此关键词命中率偏高；凡本文给出「标题命中」与「全文命中」两个数的，标题数是更保守的那个。
2. **类型之间存在重叠**。一条 Issue 可以同时是 P1 和 P5（如 pto-isa#149：静默全零 + sim/device 分歧），各类型计数之和大于并集。
3. 已排除 PR（`pull_request` 字段非空者），仅统计 Issue。
4. 六仓分布：pypto 632、simpler 310、PTOAS 301、pypto-lib 105、pto-isa 76、pypto-serving 39。

**什么不算「Hardware-Native 引起的问题」**（本文排除）：文档错字、CI 配置、构建脚本、纯语言前端 bug（SSA、控制流）、功能缺失类 Feature。判据是第 1.2 节主文的 J1——**这条问题的根因是否是一条会裁决正确性或性能的昇腾硬件事实**。

---

## 1. 硬症状在九域上的分布

**【统计】** 把「静默错算 + 挂死」作为硬症状，与九个硬件事实域交叉：

| 域 | 硬症状命中 | 其中 open |
| -- | ---------: | --------: |
| 8 任务 / 运行时环 | 233 | 45 |
| 7 数值 / dtype | 180 | 32 |
| 6 流水 / slot | 175 | 41 |
| **5 跨核 / 同步** | **174** | **28** |
| 3 布局 / valid_shape | 153 | 26 |
| 1 存储 / 复用 | 143 | 28 |
| 2 AIC / AIV 分工 | 105 | 18 |
| 9 多卡 / 通信 | 89 | 20 |
| 4 访存粒度 | 31 | 5 |
| **合计（去重）** | **454** | **73** |

**【推断】与九域详解的 PerfHint 覆盖表一起看，结论很直接**：

- **域 4（访存粒度）硬症状最少（31），却是唯一一个 PerfHint 做到成熟的域**（`PH001`）。
- **域 5（跨核 / 同步）硬症状 174 条、28 条 open，PerfHint 为零。**
- 域 8 数字最高（233）部分是关键词偏宽（`scope`、`task` 命中面大），但它与域 6 的高位一致地指向同一件事：**资源与时序类的硬件事实，今天靠运行时报错兜底，而不是编译期拦截**。

---

## 2. P1 · 静默错算：硬件约束被破坏而无人检查

### 2.1 判据

**三个条件同时成立**：编译干净 → 运行不报错 / 不抛异常 → 结果错。Issue 原文的典型措辞是「compiles clean, raises nothing, produces no NaN — the destination just contains the wrong values」（pypto#2007）。

**【统计】** 标题直接含 `silent*` 的 **34** 条；全文 **239** 条；「no error / without any warning / compiles clean / raises nothing」类措辞 **68** 条。

### 2.2 四个子型（按被破坏的硬件约束归类）

#### 子型 a：原地安全性 / 别名约束未声明

**【事实】pypto#2007** —— `pl.concat` 静默返回错数据。根因一句话：`tile.concat` 注册时**漏了 `.not_inplace_safe()`**（`src/ir/op/tile_ops/transform.cpp:763`），`MemoryReuse` 因此可以把它的输出 buffer 合并到自己的某个输入上；但 `pto.tconcat` **不能原地做**——`dst` 的行 stride 是 `C0 + C1`，与两个源的行 stride **都不同**，而它是逐行向前拷贝的，所以一旦 `dst` 与某个 src 共享基址，写 `dst` 第 j 行就会落到还没读的源数据上。

**这条 Issue 还暴露了测试体系的一个盲点**：「in-repo golden 看不到它，因为那个 golden 是同一条代码路径的第二次设备派发」。

**同族**：pypto#768（为存活的 loop yield 输出复用 buffer → 数据损坏）、pypto#585（错误混叠 `gate_acc`/`up_acc`）、pypto#1821（MemoryReuse 全局最大优先打包丢掉 `PIPE_MTE3↔PIPE_V` 同步 → 读到陈旧值，**约 17.5% 输出错误**）、PTOAS#713。

#### 子型 b：车道 / 分片语义未守

**【事实】pto-isa#149** —— 这是整个归档里机制最清楚的一条。A2A3（及 A5）的 `pushVec2GMFiFo` 在 `TILE_NO_SPLIT` 路径上**缺少 inactive-lane guard**。`SplitMode.NONE` 下硬件在**两个**子核（AIV0 + AIV1）上都跑 Vec kernel；`TILE_NO_SPLIT` 让两条 lane 都把完整静态 tile 存到 `subAIVOffset = 0`，于是 AIV1（那条非活跃的 replay lane，其 tile 携带运行时 `valid_shape=[0,0]`）把**一整块零**写进 AIV0 刚写了真实结果的同一个 v2c 槽位，把它覆盖掉。Cube 随后 pop 到零 → **静默全零输出，编译干净、无断言**。

这正是[九域详解](../Product_Planning/PyPTO_Hardware-Native九域详解_20261019.md)第 2.3 节「编译器明确不检查」三条里第一条的**设备侧根因**，且它由 pypto#1525 一路追到 pto-isa。

**同族**：pypto#2006（`pl.split_aiv(2, NONE)` 静默丢掉一半工作——每个 block 有一条 AIV lane 从不写回）、pypto#1864（`pl.split(UP_DOWN)` 的 scope 里含 `row_sum` 时静默误编译，**wrong numerics, no error**）、pypto#1448（UP_DOWN + 动态 `valid_rows > HALF` 静默丢掉 subblock 1 的写）、pypto#1525、simpler#767。

#### 子型 c：视图元数据在降级链路中丢失

**【事实】** 同一形态在不同 pass 上反复出现：

| Issue | 丢了什么 |
| ----- | -------- |
| pypto#1370 | Tile 一元算子**丢掉 TileView 的 `valid_shape`**，在 NPU 上生成不匹配的 `validCol` |
| pypto#1401 | `tile.cast` 与 `tile.row_sum` 拿到错的 `valid_shape` |
| pypto#1510 | `pto.alloc_tile` 的 addr 在「reshape-of-subview 链」上**丢掉 slice 偏移**（静默错值） |
| pypto#2010 | `t[:, a:b]` 在多行 tile 上静默返回错数据——`pl.tile.slice` 在自己的活源上编码了一个密集 MemRef |
| pypto#2087（**open**） | 动态 pitch 张量的列切片经 orch→incore 传递后静默读错数据（stride 被密集化） |
| pypto#739 | `[M, 1]` 张量缺 `pl.DN` 时**静默生成错 stride** |
| pto-isa#200（**open**） | A5 的 `TMov` 在 Acc ValidShape strip C2V 用了 `validRow` 而非 `Rows` 作 NZ srcStride → 从第 2 个 strip 起静默错 |

**【推断】** 这一子型是域 3 的签名失效：**`valid_shape` / `stride` / `pad` 是正确性参数，却长得像性能参数**。`shape` 放不下会报错，这三个写错不会。而它们只在非整除、尾块、动态形状时才生效，所以单测很容易全绿。

#### 子型 d：cache line 粒度

**【事实】pypto#2005** —— 在一个 `pl.at` scope 里，整张量初始化与标量覆写打到同一个 GM 张量时，a2a3 上静默损坏 `write_dst_map`。这与 `03-memory.md` 记载的 64 B 核内 cache line 争用是同一机制。

**【事实】pto-isa#157** —— A2A3 的 `copy_ubuf_to_gm_align_b8` 在 dst 是 peer-rank-mapped GM 时**静默丢写**，b16/b32 路径不受影响。

### 2.3 缺的是什么

**【推断】** P1 的每一个子型都对应一条**可声明、可判定**的约束：算子的原地安全性（注册表里的一个标记）、车道分片义务、视图元数据的保真传播、cache line 独占。这不是「要做静态分析」，而是「**有一张表没填满**」——pypto#2007 的修法就是给 `tile.concat` 加一个 `.not_inplace_safe()`。

---

## 3. P2 · 设备挂死：同步是正确性，却没有编译期检查

### 3.1 判据

运行时错误码 **507018**（AICore / AICPU timeout）或 **507015**（AICORE_EXCEPTION），或 `TENSOR_WAIT_TIMEOUT`、`deadlock`、`stall`。

**【统计】** 正文出现这两个错误码的 **86** 条，13 条仍 open。挂死类措辞（标题）**83** 条。

### 3.2 代表案例

**【事实】pypto#1935** —— 本文最锋利的一条证据。hard（FFTS）形式的 `pl.system.syncall` 在外层 `pl.spmd(N)` **未填满**该 core_type 的全部物理核时，设备上挂死（507018）。硬屏障等**每一个**该类型的物理核到达，未启动的核永远到不了，FFTS wait 永不完成。Ascend 910B 有 48 个物理 AIV 核：`pl.spmd(24)` → 507018，**只把 24 改成 48** → PASS，占用率是唯一变量。

Issue 的期望行为写得像一份规格：

> 编译器应当拒绝一个 hard 模式的 `syncall`，当它外层的 `pl.spmd` 启动数未填满该 core_type 的全部物理核，并给出清晰的编译期错误，例如：
> `hard pl.system.syncall(core_type="aiv_only") requires the spmd launch to fill all 48 AIV cores, but pl.spmd(24) launches 24 blocks. Use mode="soft" (GM-polling) for partial occupancy.`

而原文给出的理由是本文的核心论点：**「编译器已经知道启动 block 数（`pl.spmd(N)`）和目标 SoC 每种核型的物理核数，所以这是可静态检查的。」**

它还记录了一个被低估的二次代价：挂死会让**设备处于 draining/reset 状态，之后在该设备上的运行都返回 `code 13` 直到重置**。即一次挂死污染的不止一次运行。

**【事实】** 同族，且根因分散在三个仓：

| Issue | 机制 |
| ----- | ---- |
| PTOAS#823 | ptoas 0.45 的 `EVENT_ID` 分配器把**一个全局 event 复用给了重叠的 fence** → AIV deadlock（507018） |
| pto-isa#188 | `TPUSH` 析构的 drain 在**偶数 push 次数**上差一 → AICore 507018 |
| pto-isa#127 | `TPipe` 析构**泄漏 FFTS free signal** |
| PTOAS#797 | `TNOTIFY(Set)` 对同一信号单元的第二次写不传播 → 在 `TWAIT` 挂死 |
| PTOAS#1026 | `fence_scope<gm>` 在 vector kernel 里发出 `pipe_barrier(PIPE_FIX)`，TPUT 之后挂死设备 |
| pypto#1592 / #1650 / #1761 / #1854 / #2097 | 混合 cube+vec scope、spmd 自 RAW、UP_DOWN 产出给后续非切分 scope——五种不同形状，同一个 507018 |
| pypto#1789（**open**） | **删掉**一个 vector 算子（`pl.transpose`）导致 MTE3→V fence 的 `EVENT_ID` 被重新分配并在运行时 deadlock |

**【推断】pypto#1789 值得单独看**：删代码引入挂死。这说明同步资源（event id、flag）的分配对**无关的代码改动敏感**，而开发者手上没有任何工具能看到这种敏感性。

### 3.3 缺的是什么

**【推断】** 域 5 的 C 列为零（见九域详解第 5.4 节）。而 P2 的 86 条里，至少 `pypto#1935` 这一类（占用率 vs 硬屏障）是纯算术可判定的。更难但更值钱的是「GM 交接四步序列是否完整」——`cacheinvalid → fence → syncall → cacheinvalid`，顺序固定、漏一步就读旧数据，今天完全靠文档里的一段散文。

---

## 4. P3 · 规格不可信：给出的数字与硅片不符

### 4.1 判据

不是「规格查不到」，而是「**查到的规格是错的，或被硬编码成了一个数**」。

### 4.2 最高危的一条（仍 open）

**【事实】pto-isa#170** ——

> Ascend910B1 / CANN 9.0.0（target `a2a3`）上，一个纯 Vector kernel 的 `pto.alloc_tile` 分配达到 **188.8 KB** 的 Vec 高水位——**仍在 192 KB Vec UB 之内**——在设备上输出 **NaN**。同一个 kernel 降到 **183.3 KB** 高水位则正确。损坏的开关点在 183.3 KB（干净）与 188.8 KB（NaN）之间，**即约 184 KB 附近**。两种情况编译都不报错；188.8 KB 的那一个在运行时**静默损坏**。

| case | Vec 占用 | buffer 数 | 设备输出 |
| ---- | -------- | --------- | -------- |
| HD512 | **188.8 KB** / 192 KB | 81 | **NaN**（112 个） |
| HD496 | **183.3 KB** / 192 KB | 81 | 正确（0 个 NaN） |

Issue 的假设：真实可用 Vec UB 约 **184 KB 而非 192 KB**，顶部约 8 KB 被保留；**上游（PyPTO）分配器的 192 KB 预算没有减掉这部分**，于是它会发出一个达到 188.8 KB 的 `.pto` 而不报任何错。

它还排除了另外两种解释：NaN **全部集中在最高地址的累加器 tile**（低地址 tile 干净）；把高水位降到 184 KB 以下（更小 tile，或内层流水 `stage=1`）→ 0 NaN；用保证除数严格为正的输入 → 188.8 KB 的那个**仍然** NaN。

**【推断】这一条把「Q 列未开放」升级成了一个更严重的问题**：维度框架里建议「开放已有单源」，前提是那个单源**正确**。本例中 `BackendHandler` 的容量数字与硅片不符，而且**差异不在任何一方的文档里**。因此 Q 列的正确形态不只是暴露数字，还要能表达「保留区 / 可用上界」这种不等于物理容量的量。

### 4.3 硬编码族

**【事实】**

| Issue | 被写死的硬件数 |
| ----- | -------------- |
| PTOAS#57 | `TileConfig.fractalABSize` 硬编码 **512** |
| PTOAS#709 | GM-ring 的 `slot_num` 在 `aic/aiv_initialize_pipe` 上硬编码 **8 / 4** |
| pypto#1173 | `execute_on_device` 里硬编码 `block_dim=24`，在可用核更少的设备上**挂死** |
| PTOAS#427 | `pto.tmrgsort` format2 硬编码 4-way |
| simpler#1561 | 四个 notify demo 测试硬编码平台与 device ID |

**【推断】** pypto#1173 是这一族的代表形态：**一个硬编码的硬件数字，失效方式是挂死而不是报错**。8 / 4 这两个数字恰好是 `00-cluster_architecture.md` 里记载的单向 8 槽 / 双向每向 4 槽——即**文档已经把它当规格写了，代码却把它当常量写了**，两边没有连起来。

---

## 5. P4 · 硬件旋钮在抽象里消失

### 5.1 判据

这个参数在 **ISA 里有、IR op 里也有，唯独 DSL 不暴露**，于是用户无法设置，编译器只能取一个保守默认值。

### 5.2 代表案例

**【事实】pypto#2287** —— 自动生成的跨核 pipe 把消费者侧片上 FIFO 预留为 `common_slot_size * slot_num`，**两个因子都错**：

- **深度错**：按 PTOAS 规格，应当用 `local_slot_num`（一个更小的独立旋钮）。**PyPTO 从不发出 `local_slot_num`**，所以每条 local-FIFO pipe 都按完整的 GM 环深度（单向 8 / 双向 4）预留。参数**在 IR op 里已经存在**（`python/pypto/ir/op/system_ops.py:412-426` 同时接受 `slot_num` 与 `local_slot_num`），但 DSL 旋钮 `pl.cross_core_slot(*, slot_num: int)` 只暴露 `slot_num`，**所以用户也设不了**。
- **槽大小错**：`GetCommonSlotSizeBytes` 取 `max(c2v, v2c)`，而 ISA 按**各方向自己的**消费者 tile 大小索引本地落地区。于是**一个方向上的大载荷会把另一个方向的 buffer 撑大**——而 C2V 的 buffer 住在 UB，是这块芯片上最紧的空间。

后果：在载荷不对称的双向 pipe 上**过量预留 UB 一个数量级**，让合法 kernel 过不了 `AllocateMemoryAddr` 的容量检查。

**【推断】** 这是 E 列缺项的教科书案例，而它的症状**伪装成了 P3**：开发者看到的是「容量不够」，真因是「有个旋钮我碰不到」。这类问题最容易被误诊为「片上空间不足，去缩小 tile」——而那是错的修法。

---

## 6. P5–P8 · 另外四类

### 6.1 P5 · 跨平台 / 跨代际 / 仿真器与真机分歧

**判据**：同一段代码，换平台（A2/A3 ↔ A5）或换执行环境（仿真器 ↔ 真机）就改变行为。

**【统计】** 提到仿真器（`a2a3sim` / `a5sim` / `cpu_sim` / `simulator`）的 **166** 条，其中明说 sim 与 device 不一致的 **20** 条。提到代际 / 平台字样的 **846** 条（偏宽，仅作量级）。

**【事实】两个方向都发生**：

| 方向 | Issue | 情形 |
| ---- | ----- | ---- |
| 换代际 | PTOAS#485 | `decode_attention_incore_7` 在 **A5 挂死、A3 通过**（ptoas 0.24） |
| 换代际 | PTOAS#441 | A5 target 在一个 **A2A3 能过**的 RMSNorm vector kernel 上挂死 |
| 换代际 | pypto#828 / PTOAS#421 | identity `pto.tmov`（dst == src）引发多余同步，**在 A5 挂死** |
| sim 过、真机错 | PTOAS#652 | `tfillpad + trowexpandmul` 在 **a2a3 真机挂死，a2a3sim 通过** |
| sim 过、真机错 | simpler#900 | `spmd_paged_attention_highperf` **硬件超时或部分零输出，a2a3sim 通过** |
| sim 过、真机错 | **pto-isa#149** | **CPU sim 有 inactive-lane guard，device 路径没有** |
| 真机过、sim 错 | simpler#581 | a2a3sim 无法 replay **硬件上能通过**的跨核 kernel |
| 真机过、sim 错 | PTOAS#660 | a2a3 结果错，a2a3sim **SIGSEGV** |
| 数值不一致 | pto-isa#88 | A5 上 `expand_clone(d2)` 在仿真器与硬件之间**数值不一致** |
| 语义不一致 | pto-isa#101 | `cpu/TPush.hpp` 的仿真**没有正确建模 a5 跨核 pipe 语义** |

**【推断】** 这一类的产品含义比它的数量更重：**「提供 CPU 模拟器，使得没有昇腾硬件也能做功能验证」是这个栈公开宣称的获客设计**（见 [hw-native-sys 仓库功能洞察报告](hw-native-sys-仓库功能洞察报告.md)）。而 pto-isa#149 这种「sim 比 device 更宽容」的分歧，会让模拟器**系统性地藏起**恰恰最该被它抓到的那一类缺陷——车道、同步、cache 语义。模拟器与硬件的**语义等价性本身需要被当成一等交付物来验证**，否则它的存在是负资产。

### 6.2 P6 · 工具链版本回归：同一硬件约束被反复破坏

**判据**：上下游版本号一变，硬件约束就被重新打破。

**【统计】** 回归措辞 **196** 条（40 open）；正文出现 ptoas 版本号的 **100** 条；出现 CANN / driver 版本的 **63** 条。

**【事实】PTOAS#1111** 是方法论最严的一条：DeepSeek V4-Pro 的 35 个 case 从 PTOAS v0.48 升到 v0.54，做了一次 **2×2 对照**（两个 pypto revision × 两个 PTOAS 二进制），固定 pypto / pypto-lib / simpler / pto-isa / CANN / driver / 硬件，**只动 PTOAS 二进制**：

> **v0.48 上 12/12 PASS，v0.54 上 0/12**，两个 pypto arm 都是如此。**零算术 opcode 变化**——auto-sync 的 flag 对在 v0.49 掉了，dcci scope 在 v0.50 从 OUT 变成 ALL。

**【事实】同族**：

| Issue | 回归 |
| ----- | ---- |
| PTOAS#450 | `--enable-insert-sync` 在 v0.22 挂死，**v0.17 正常** |
| PTOAS#564 | insert-sync 在 K-loop 流水化 GEMM 里丢掉 MTE1→MTE2 的 wait（**0.29 相对 0.27 的回归**） |
| PTOAS#884 | 0.47 把 `func.func private` 降成外部链接 → 重复符号 → AICORE_EXCEPTION 507015 |
| pypto#1310 → #1352 | acc→acc 非法 `pto.tmov`，标题明记为 **#673 的 regression**，此后再现一次 |
| simpler#1839（**open**） | EP2 comm-window 轮询在 a2a3 上死锁，**自 #1644 的 pto-isa pin bump 起**（83d01313 → 0cefc9a5） |
| pto-isa#188 | `TPUSH` drain off-by-one，**014920a8 引入** |
| pto-isa#172 | `TPipe::SyncPeriod` 改动带来 flash_atten 大 S 场景 **10–12% 性能回归** |

**【推断】** 回归集中在 **insert-sync / auto-sync / event 分配**这一块，也就是域 5。这不是巧合：**同步插入是四家 pass 框架里唯一「做错会 hang、且不能被 autotune 替代」的一类**（见[四框架 Pass 对比](../Competitive_Analysis/四框架Pass对比_PyPTO_TileLang_Triton_cuTile_20261019.md)的第 IV 类判断）。它跨 pypto / PTOAS / pto-isa / simpler 四个仓，每仓独立演进、各有 pin，而没有一处**跨仓的硬件契约测试**来守住它。

### 6.3 P7 · 诊断无落点 / 编译器决策不可见

**判据**：问题被发现了，但信息落不到用户写的那一行，或编译器的硬件决策不在任何产物里。

**【事实】**

| Issue | 断点 |
| ----- | ---- |
| pypto#1305（**open**） | `PH001` 的 span 指向流水线后 IR 文本位置（`<string>:line:col`），非原始 DSL；`TileType` / `Call` / `Span` 都不携带 DSL 源元数据 |
| pypto#1612 | `@pl.jit` 诊断报告真实源出处，而非 `"<string>"` |
| pypto#1200 | 前端诊断需要包含源文件与行号 |
| pypto#477 | `INTERNAL_CHECK` / `InternalError` 需要 span 信息 |
| pypto#1558 | `pl.at(deps=...)` 的形状拒绝诊断**插入符指向错误的源行** |
| pypto#824 | `FormatShape` 在错误信息里**打印 ExprPtr 地址** |
| **pypto#2088** | **Pass dump 省略隐式 tile layout，使 layout bug 难以定位** |
| pypto#2232 | `AutoTileMatmulL0` 的 matmul_acc M/N 延后丢掉了**本来受支持的 K 分块**，然后报了一个**误导性的 buffer 错误** |
| PTOAS#546 | ptoas 降级 `pto.tfillpad` 遇到 src/dst tile 类型不匹配时**错误信息不清** |
| simpler#731 | Stall 诊断对几乎所有卡住的任务都打印 `kernel_id=-1` |
| simpler#1133 | trb 的背压 / 死锁日志**误导**（把堆游标当占用率打印，且无 ring 索引） |

**【推断】** pypto#2088 与 pypto#2232 是这一类里最值得注意的两条，因为它们是 **A 列（归因）本身的缺陷**，不是 C 列的缺陷：

- #2088：PyPTO 的 60 篇 pass 文档 + 逐 pass dump 是它相对四家竞品最强的资产（见四框架对比），但**dump 省略了隐式 layout**——即最需要被看见的那个决策恰好不在 dump 里。
- #2232：编译器放弃优化后报了一个**误导性**错误。这与[九域详解](../Product_Planning/PyPTO_Hardware-Native九域详解_20261019.md)第 10.2 节的「归因型 PerfHint」是同一机制的两面：说清楚放弃原因是资产，说错就是负债。

### 6.4 P8 · 性能悄悄劣化

**判据**：没有「变慢了」的报告，只有被偶然发现的百分数。

**【统计】** 性能类措辞全文 **231** 条（75 open）。

**【事实】有确切数字的几条**：

| Issue | 劣化 |
| ----- | ---- |
| pto-isa#172 | `TPipe::SyncPeriod` 改动 → flash_atten 大 S **10–12%** 回归 |
| simpler#1791 | A2A3 每次运行开新 AICore stream → HCA/SWA 有效时延回归 **3.8–4.4%** |
| pypto#1821 | MemoryReuse 打包改动丢掉同步 → **约 17.5% 输出错误**（兼属 P1） |
| PTOAS#586 | MLIR 编译时间从 0.27 的 ~1–2 分钟涨到 0.29+ 的 **~8 分钟** |
| PTOAS#950 | ptoas 0.50 在某 kernel 上编译 **>160 倍**变慢 |
| （文档）`PH001` | `b_trans` matmul 的 128 B 行 vs 推荐 512 B，实测 **16–25%** |

**【推断】** 这一类的特征是**发现方式全是偶然**：有人恰好做了版本对照、有人恰好跑了基线。域 4 以外的八域没有任何编译期性能提示，也没有性能基线回归门禁（PTOAS#586 / #950 两条编译时性能回归能被发现，恰恰因为编译时间是开发者天天感知的量）。

---

## 7. 三条横切观察

### 7.1 「编译器已经知道，但选择不说」是最大的单一浪费

**【推断】** 把 P1 / P2 / P4 的代表案例并排看，一个共同结构浮现出来：**判定所需的全部信息在编译期都已具备**。

| Issue | 编译期已知的信息 | 却没有做的事 |
| ----- | ---------------- | ------------ |
| pypto#1935 | `pl.spmd(N)` 的 N；SoC 每核型物理核数 | 比较这两个数 |
| pypto#2007 | `tile.concat` 的 dst/src stride 关系 | 在算子注册表打一个 `.not_inplace_safe()` |
| pypto#2287 | `local_slot_num` 已在 IR op 签名里 | 在 DSL 暴露它、并按方向分别算槽大小 |
| pypto#2006 | 区域的 `mode` 与 `aiv_id` 是否进入实参 | 比较这两者 |
| pto-isa#170 | （缺）保留区大小 | —— 这一条**真的缺信息**，属 P3 |

**【推断】** 八条里有七条不需要新信息，只需要**把已有的两个数放在一起比一次**。这与[内容清单](../Product_Planning/PyPTO_Hardware-Native内容维度与清单_20261019.md)第 8 节第 3 项（把编译器放弃优化的静默分支改成归因型提示）是同一条路径，而 Issue 数据给了它一个更强的理由：**这不是"增加诊断能力"，而是"把已经做过的判断说出来"。**

### 7.2 一条硬件约束的根因会穿透四个仓库

**【事实】** 以车道守卫为例，一条失效的追溯路径：

```
pypto#1525  (DX: 静默全零，应当是编译错误)
   ↓ 板卡 owner 在真机 a2a3 上复现并移交
pto-isa#149 (设备侧根因：pushVec2GMFiFo 的 TILE_NO_SPLIT 缺 inactive-lane guard)
   ↓ 同一语义在模拟器上被 guard 住
          → sim 通过、真机错（兼属 P5）
   ↑ 而 PyPTO 侧对应的「编译器不检查 lane 分片」仍记在 04-scopes.md 的散文里
```

**【事实】** 另一条：`pypto#828`（PyPTO codegen 发出 identity `tmov`）↔ `PTOAS#421`（ptoas 据此插入多余同步）→ A5 挂死。**同一个缺陷在两个仓各开了一条 Issue，标题几乎一样。**

**【推断】** 这解释了 P6 为什么集中在同步：**跨仓的硬件约束没有单一归属**。pypto 认为是 ptoas 的同步插入问题，ptoas 认为是 pypto 发出了无意义的 op，pto-isa 认为自己的 intrinsic 契约没变。而四个仓各自 pin 版本独立演进。

### 7.3 测试体系对这一类失效系统性失明

**【事实】** 三处彼此独立的记载：

- **pypto#2007**：「in-repo golden 看不到它，因为那个 golden 是**同一条代码路径的第二次设备派发**」——自己和自己比。
- **pto-isa#149**：CPU sim 有 guard 而 device 没有 → **sim 通过**。
- **pto-isa#170**：183.3 KB 干净、188.8 KB NaN —— 只有**跨越某个占用阈值**才触发，而测试通常用小 shape。

**【推断】** 这三条合起来说明：P1 / P2 这一类失效需要的验证形态是**真机 + 独立 golden + 足够大的占用**，而这三项任意缺一都会放过它。这也反向解释了为什么 454 条硬症状里有相当一部分是在 DeepSeek V4 / qwen3-14b 这类**真实大模型 kernel**上被发现的——那是唯一同时满足三项条件的工作负载。

---

## 8. 八类问题 → 维度框架的映射

| 类型 | 缺的形态 | 缺的域（最集中） | 对应内容清单的哪一项 |
| ---- | -------- | ---------------- | -------------------- |
| P1 静默错算 | **C** | 1、2、3 | 第 3 项（归因型提示）；算子注册表补全 |
| P2 设备挂死 | **C** | **5** | 第 3 项；域 5 的交接序列校验 |
| P3 规格不可信 | **Q** | 1、5 | 第 1 项（开放平台档案）**+ 一条新约束：档案必须能表达保留区 / 可用上界** |
| P4 旋钮消失 | **E** | 5、6 | （新）DSL 旋钮与 IR op 签名的一致性审计 |
| P5 跨平台 / sim 分歧 | **维度五** | 全部 | 第 6 项（代际按域版本化）**+ 模拟器语义等价性作为交付物** |
| P6 版本回归 | **跨仓契约** | **5** | （新）跨仓硬件契约测试 |
| P7 诊断无落点 | **A** | 全部 | 第 2 项（源码映射）；pass dump 补全隐式 layout |
| P8 性能悄悄劣化 | **C + O** | 2、5、6、7、8、9 | 第 3 项；性能基线门禁 |

**【推断】三项清单之外的新增项**（本文数据支持，原清单未覆盖）：

1. **P3 揭示平台档案不能只是「暴露已有数字」。** pto-isa#170 的根因是 PyPTO 的 192 KB 预算不减保留区，所以档案需要区分**物理容量**与**可用上界**，并且可用上界本身要能被真机验证。
2. **P4 需要一次 DSL ↔ IR op 签名的一致性审计。** `local_slot_num` 在 IR 里有而 DSL 里没有，这类缺口无法靠文档发现，只能靠比对签名。
3. **P6 需要跨仓硬件契约测试。** 四个仓各自 pin 独立演进，而同步正确性跨这四个仓；PTOAS#1111 的 2×2 方法论应当从「出事后的排查手段」变成「常设的门禁」。

---

## 9. 复核方法

数据与脚本：Issue 归档在 `github_issues/<repo>/<repo>_issues.json`（GitHub API 原始字段）。本文的统计用 node 直接读取这些 JSON，排除 `pull_request` 字段非空者。

```bash
# 总量（排除 PR）
node -e "
const fs=require('fs');
let n=0;
for(const f of ['pto/pypto','PTOAS/PTOAS','simpler/simpler','pypto-lib/pypto-lib','pto-isa/pto-isa','pypto-serving/pypto-serving'])
  n+=JSON.parse(fs.readFileSync('github_issues/'+f+'_issues.json','utf8').replace(/^﻿/,''))
       .filter(i=>!i.pull_request).length;
console.log(n);   // 1463
"
```

```bash
# 某一类的命中（示例：运行时挂死错误码），输出 repo#number 与标题
node -e "
const fs=require('fs');const out=[];
for(const [f,r] of [['pto/pypto','pypto'],['PTOAS/PTOAS','PTOAS'],['simpler/simpler','simpler'],['pypto-lib/pypto-lib','pypto-lib'],['pto-isa/pto-isa','pto-isa'],['pypto-serving/pypto-serving','pypto-serving']])
  for(const i of JSON.parse(fs.readFileSync('github_issues/'+f+'_issues.json','utf8').replace(/^﻿/,'')))
    if(!i.pull_request && /507018|507015/.test((i.title||'')+(i.body||''))) out.push(r+'#'+i.number+' '+i.title);
console.log(out.length); out.forEach(s=>console.log(s));
"
```

把上面的正则替换为第 0 节各类型的判据，即可复现本文每一个【统计】数字。各类型正则：

| 类型 | 正则要点 |
| ---- | -------- |
| P1 | `silent` ／ `no (error\|warning\|diagnostic)` ／ `compiles clean` ／ `raises nothing` ／ `wrong result` ／ `miscompil` ／ `all-?zero` ／ `golden mismatch` |
| P2 | `507018\|507015` ／ `hang\|deadlock\|TENSOR_WAIT_TIMEOUT\|stall` |
| P3 | `hard-?cod\|hardcod` ／ `appears to be` ／ `not \d+ ?KB` ／ `expose .* as .* param` |
| P5 | `a2a3sim\|a5sim\|cpu[_ ]sim\|simulator` ／ `but (passes\|works)` |
| P6 | `regress` ／ `ptoas 0\.\d\|v0\.\d\d` ／ `pin bump` |
| P7 | `source location\|<string>\|caret\|misleading\|unclear error\|hard to locate` |
| P8 | `perf\w* (regress\|drop)` ／ `\d+(\.\d+)?% ` ／ `slower` |

> 口径再次提醒：匹配覆盖标题 + 正文，正文含堆栈与环境表，故为**宽匹配**；类型之间重叠；全部结论按下界与量级表述。本文只做分析，不含代码或产品实现。
