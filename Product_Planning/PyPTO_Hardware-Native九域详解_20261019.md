# PyPTO Hardware-Native：九个昇腾硬件事实域详解

> 整理日期：2026-10-19
> 本文是 [PyPTO_Hardware-Native内容维度与清单_20261019.md](PyPTO_Hardware-Native内容维度与清单_20261019.md) 第 3 节的展开。主文给出维度框架与优先级，本文逐域给出事实、承载与缺口。
> 证据范围：`repo/pypto`（较新镜像）的 `src/`、`include/`、`python/`、`docs/zh/` 实际检索，逐条给出位置。
> 标注约定：【事实】可在仓库文件中直接核对；【推断】基于事实的判断；【缺口】未被承载的部分。

---

## 0. 先更正主文的两处事实

展开过程中有两条主文的判断被证据推翻，先列出，主文已同步修正。

### 0.1 PerfHint 不止 `PH001`，而是 11 个码

主文判断「九个域里只有访存粒度一域有硬件性能提示」。这来自只检索 `DiagnosticCheckRegistry` 和 `PH00` 字面量。**实际上性能提示有两套发出路径**：

| 路径 | 码 | 发出位置 |
| ---- | -- | -------- |
| `DiagnosticCheckRegistry` 注册的检查 | `PH001` | `src/ir/verifier/perf_hint_tile_innermost_dim.cpp` |
| **pass 直接 emplace 的 hint** | `PH-AT-003/005/006/007/008/009/010/011` | `src/ir/transforms/auto_tile_matmul_l0_pass.cpp` |
| 同上 | `PH-MR-001` | `src/ir/transforms/memory_reuse_pass.cpp` |
| 同上 | `PH-DSA-001` | `src/ir/transforms/allocate_memory_addr_pass.cpp` |

**【事实】** 共 **11 个 hint 码**。复核：

```bash
grep -rhno "PH[-A-Z0-9]\{3,12\}" repo/pypto/src repo/pypto/include | sed 's/.*://' | sort -u
```

**【推断】修正后的结论更有意思**：PerfHint 覆盖的是**域 1（存储容量）+ 域 3（分形对齐）+ 域 4（访存粒度）+ 域 6（流水深度）** 四域，而 **域 2、5、7、8、9 仍为零**。而且这 11 条里有 8 条（`PH-AT-*`）属于一个特殊类别——**「编译器放弃替你做决定」的归因型提示**，而不是「你写慢了」的建议型提示。这一类在主文的六形态里横跨 C 与 A，值得单独命名（见第 10.2 节）。

### 0.2 `MemorySpace` 是 10 个值，不是 8 个

**【事实】** `include/pypto/ir/memory_space.h` 的枚举有 10 个值：`DDR`、`Vec`、`Mat`、`Left`、`Right`、`Acc`、`Bias`、`ScalarLocal`、`LeftScale`、`RightScale`。后两个是 **A5 的 MX block-scale 缓冲区**（L0A 侧 / L0B 侧），既有 Insight 整理稿的 8 空间表写于 MX 支持之前。

---

## 1. 域 1 · 存储层级与容量

### 1.1 硬件事实

**十个空间，六个在片上，彼此并列而非嵌套。**

**【事实】**（`docs/zh/user/language/03-memory.md`）「六个片上空间是彼此并列的缓冲区，不是嵌套关系：`Left` 不是 `Mat` 里的一块区域，`Acc` 也不在 `Right` 里面。数据在它们**之间**搬运，而哪些搬运合法是硬件属性，不是编译器策略。」

| 空间 | 硬件 | 从 DDR 直达 |
| ---- | ---- | ----------- |
| `DDR` | 片外全局内存 | 它本身 |
| `Vec` | UB 统一缓冲区 | **能**（`pl.load` 默认目标） |
| `Mat` | L1 | **能**（`target_memory=pl.Mem.Mat`） |
| `Left` | L0A（matmul 左操作数） | 不能，只能 `pl.move` |
| `Right` | L0B（matmul 右操作数） | 不能，只能 `pl.move` |
| `Acc` | L0C（累加器） | 不能，**由 `pl.matmul` 写入** |
| `Bias` | AIC 核上 bias 缓冲区 | 不能，只能 `pl.move` |
| `ScalarLocal` | 核上标量寄存器 / C 栈 | — |
| `LeftScale` / `RightScale` | A5 的 MX block-scale 缓冲 | 不能 |

**三条硬约束**：

1. **【事实】** DDR load 只能落到 `Vec` 或 `Mat`。这是「矩阵乘通路是两段式、逐元素代码不是」的根源。
2. **【事实】**（`memory_space.h` 注释）**没有任何 target 能把数据 move 进 `Acc`**——只有 MAD 单元写 L0C，所以累加器必须在 `Acc` 里**创建**，无法由拷贝得到。
3. **【事实】** 合法搬运是**逐 target** 的：`Backend::GetSoC().GetMemoryGraph()` 给出精确的每目标邻接表，`Backend::FindMemPath` 走它；**A5 实现 `Vec → Mat`，A2/A3 不实现**。IR 层的 `memory_space.h` 只持有「所有后端的并集」，用于在未选 backend 时（类型推导发生在解析期）拒绝「任何地方都不可能」的组合。

**容量**（来自 `BackendHandler`）：

| 参数 | Ascend910B | Ascend950 |
| ---- | ---------: | --------: |
| L0A / L0B | 64 KiB | 64 KiB |
| L0C | **128 KiB** | **256 KiB** |
| Mat (L1) | 512 KiB | 512 KiB |

**【事实】三个不同的「粒度」数字不要混用**（本域一个，域 4 两个）：核内数据 cache line = **64 B**（`pl.read`/`pl.write` 通路）；GM 访问粒度 = 512 B / 128 B；L2 cache line = 512 B。

### 1.2 PyPTO 承载

| 形态 | 内容 |
| ---- | ---- |
| **E** | `pl.Mem.*`（= `pl.MemorySpace`）、`target_memory=`、`pl.move`、`pl.create_tile`、`pl.create_l1`、`pl.MemRef("name", slots=N)` |
| **Q** | `get_l0a/l0b/l0c/bias/mat_capacity_bytes`、`GetSoC().GetMemoryGraph()`、`FindMemPath` —— **全部未对用户开放** |
| **C** | `PH-AT-*` 8 条（L0 容量 / 分形判定）、`PH-MR-001`（流水深度 vs 容量）、`PH-DSA-001`（DSA-RP 放弃流水分离）、`ScalarWriteLineShared`（64 B line 争用，Warning） |
| **A** | `InferTileMemorySpace`(20)、`InitMemRef`(34)、`MaterializeSemanticAliases`(35)、`MemoryReuse`(36)、`AllocateMemoryAddr`(37)；三种 `memory_planner`（`PYPTO` / `DSA_RP` / `PTOAS`） |
| **O** | `python -m pypto.tools.memory_map <pass_dump> -o map.html` —— 横轴地址、纵轴生命期、旁边是 IR |
| **R** | `tutorials/02-matmul`、`examples/intermediate/01_fused_linear.py`、`examples/runtime/multi_program_kv_cache.py` |

### 1.3 做错的后果

| 失败模式 | 表现 | 证据 |
| -------- | ---- | ---- |
| 容量放不下 | 编译期报错或流水降级 | `PH-MR-001` |
| **复用别名错误** | **静默数据损坏** | #585 错误混叠 `gate_acc`/`up_acc`；#768 为存活 loop yield 输出复用 buffer |
| acc→acc 非法 `tmov` | 反复回归三次 | #673 → #1310（标题记为 #673 regression）→ #1352 |
| 并发标量写同一条 64 B line | **大多数写消失，且每次运行消失的不一样** | `03-memory.md`「来自并发任务实例的标量写」 |

**【事实】** 64 B line 这条已经有检查：`ScalarWriteLineShared`（Warning，`PrePipeline`）。它分两种情形报——下标可分析时给出实测跨步（「consecutive blocks write 4 bytes apart, so 16 of them share each 64-byte cache line」），下标不可分析时如实说明而不猜。**它明确有两个盲区**：两个不同任务写同一张量**完全不报**（缺任务依赖图，报就会在每对有序生产者/消费者上误报）；被 `if blk == 0:` 保护的写**会误报**（检查不读谓词）。

### 1.4 缺口

- **Q 整列未开放**，包括比容量数字更有价值的 `GetMemoryGraph()`——它回答的正是「在这个 target 上，从 X 能不能直接搬到 Y」。
- **`MemoryReuse` 的别名关系没有可视化。** `memory_map` 画的是「地址 × 生命期」，看得出谁活得太久、余量还有多少，但看不出**谁和谁被判定为可以共用物理 slot**。而静默数据损坏恰恰是别名判断错，不是生命期算错。

---

## 2. 域 2 · 计算单元与分工

### 2.1 硬件事实

**【事实】** Cluster = **1 个 Cube 核 + 2 个伙伴 Vector 核**，共享基于硬件 flag 的同步机制（`reference/pto-isa/00-cluster_architecture.md`）。拓扑在 `BackendHandler` 里是 `Die` / `Cluster` / `Core` / `Mem` 四级类。

三条**硬件级**约束，每一条都会改变代码怎么写：

1. **跨核 pipe 的 split/no-split 属性被硬件固定。** 【事实】「函数里所有 `pl.aiv_shard` / `pl.aic_gather` 都跑在同一条跨核 pipe 上，而硬件在这条 pipe 的整个生命周期内把它固定为 split 或 no-split 之一。」因此同一函数内，一个会跨越边界的 `mode=NONE` 区域不能与同样跨越的 `UP_DOWN`/`LEFT_RIGHT` 区域并存。但**两种不同的切分轴没问题**——轴是逐次传输选的，只有 split/no-split 属于 pipe。
2. **`mode=NONE` 的 V→C gather 只能搬 lane-uniform 的值。** 【事实】「硬件要求两条子 lane 都参与 no-split 握手，而它们共用同一个目标槽位、没有每 lane 偏移。两者之间没有任何仲裁：两条 lane 都会 push，因此当它们持有**不同**的值时，cube 收到的是二者之一，且**不确定是哪一个**——不是 lane 0 的，而是不确定的。」没有办法指定 lane；把产出限定到某条 lane 也无效，因为另一条 lane 依然执行到 push。
3. **TNOTIFY 两种核都能跑。** 【事实】因此在混合 kernel 中编译器会把 `pld.system.notify` **同时发到 AIC 与 AIV 两条 lane**。危险的是 cube 那份副本：AIC 可能在 AIV 的 `put` 尚未落盘前就发布信号。

### 2.2 PyPTO 承载

| 形态 | 内容 |
| ---- | ---- |
| **E** | `pl.at(level=pl.Level.CORE_GROUP)`、`pl.cluster()`、`pl.spmd(n)`（dispatch / loop 两形式）、`pl.split_aiv(2, mode=)`、`pl.SplitMode.UP_DOWN/LEFT_RIGHT/NONE`、`pl.aiv_shard` / `pl.aic_gather`、`optimizations=[pl.split(mode)]` |
| **Q** | **已开放（DSL 内）**：`pl.system.available_cluster_count()`、`pl.system.available_aiv_count()`；未开放：`Die`/`Cluster`/`Core` 拓扑类 |
| **C** | 一组结构性错误（`src/ir/verifier/verify_aiv_split.cpp`），见下表；**无 PerfHint** |
| **A** | `OutlineIncoreScopes`(09)、`OutlineClusterScopes`(10)、`ExpandMixedKernel`(24)、`SplitVectorKernel`(26)、`LowerAutoVectorSplit`(23)、`StampTfreeSplit`(27) |
| **O** | L2 泳道按 AIC / AIV 排布 |
| **R** | `tutorials/03-mixed-kernel`、`examples/advanced/03_mixed_kernel.py` |

**【事实】已有的结构性诊断**（摘自 `04-scopes.md` 症状表）：

- `Misplaced tensor op ... should be inside InCore block`
- `vector op '...' sits outside every pl.split_aiv region`
- `cube op '...' inside a pl.split_aiv region`
- `'x' is produced on the CUBE lane ... reads it on the VECTOR lane inside one`（未写明的 C→V 跨越）
- `'x' is defined inside a pl.split_aiv region but ... reads it on the CUBE lane outside`（未写明的 V→C 跨越）
- `'tile.aiv_shard' operand is in Vec, but it transfers a cube-produced value`
- `'pl.aiv_shard' crosses the AIC/AIV boundary under ... but ... earlier in this function crosses it under ...`（split/no-split 混用）

**【推断】** 这是九域里结构性检查最密的一域——编译器**强制**你写明每一次跨核边界，哪怕不写也能工作（「去掉其中任何一个调用，这次跨越依然能工作……但那样一来边界就成了没人选择的边界，而这恰恰与手动模式的意义相反」）。这是 Hardware-Native 在语言层做得最到位的一处：**它把「硬件事实必须被表达」变成了编译期义务**。

### 2.3 三处「编译器明确不检查」

**【事实】** 文档逐条标明了编译器不检查的地方，这是本域真正的风险面：

| 不检查什么 | 症状 | 文档原文 |
| ---------- | ---- | -------- |
| `mode=NONE` 区域向外的 V→C gather 是否 lane-uniform | cube 随机读到其中一条 lane 的值 | 「编译器不会检查上述任何一点。」 |
| lane 分片（两条 lane 是否重复执行只应发生一次的副作用） | 对端信号计数器读到应有值的 **2 倍**；`wait` 一次就放行 → 该 rank 抢跑读到传输途中的数据 | 「两条 AIV lane 对同一个 peer 发 notify 能正常编译，且**不会有任何诊断**」 |
| cube 写与 vector 读同一块 GM 的定序 | 读到旧数据 | `ExpandMixedKernel` 只自动处理一种窄特例（唯一 cube `tile.store` producer 与同 origin 的 vector `tile.load` 在同一 body 或嵌套 body） |

**【事实】** 第二条为什么无法诊断，文档说得很清楚：「正确写法与错误写法在 AIV 函数中生成的是同一条语句，唯一的差别是 `aiv_id` 有没有进入该调用的实参。」

**【推断】** 第二条的症状——「多 rank 运行时某个 rank 间歇性算错；单 rank 复现不了，看上去也不像同步问题」——是整个九域里**最难自查**的一个。它跨域 2 与域 9，而两域的 C 列都是空的。

---

## 3. 域 3 · 布局与分形

### 3.1 硬件事实

**【事实】** Cube 要求 NZ / ZN 分形布局与 16 对齐。`pl.NZ` 的完整前提条件（`user/language/00-types.md`）：

- dtype 为整字节
- 张量形状静态
- `shape[-2] % 16 == 0`
- `shape[-1] % (256 / dtype 位宽) == 0`
- **逻辑秩为 2 或 3**（`[R, C]` 或 `[B, R, C]`——底层 NZ 描述符只有一个 batch 槽位）
- 作为 matmul 操作数读取

其余情形一律报错。

**两条容易踩的语义**：

1. **【事实】layout 注解是对内存中字节序的断言，不是请求转换。** 因此把 ND 实参传给声明为 `pl.NZ` 的形参会在**第一个 pass 运行之前**被类型检查拒绝：`Layout mismatch at argument 1 of call to 'nz_helper': parameter 'b' is declared NZ but the argument is ND.` 两端都标注，或者都不标注。
2. **【事实】NZ 切片的追溯是逐矩阵的。** `[N_RANKS, R, C]` 堆叠张量取 rank 分片 `w[r]` 可以（保留了最后两轴）；对整体打包一次的 `[N_RANKS * R, C]` 取 `w[r*R : (r+1)*R]` **被拒绝**——「这些行在内存中连续，但它们是若干分形列块中的片段，而不是 NZ 打包的 `[R, C]`」。

**MX scale 布局**：**【事实】** `pl.MX_A_ZZ` / `pl.MX_B_NN` 标注 Ascend950 上 MX 操作数的 GM scale 张量。「这是唯一一处**要求**在 `pl.Tensor` 注解上写布局标记、而非不建议写的场景。」

**【事实】** PTOAS 侧 `TileConfig.fractalABSize` 硬编码 512（PTOAS#57）。

### 3.2 PyPTO 承载

| 形态 | 内容 |
| ---- | ---- |
| **E** | `pl.ND` / `pl.NZ` / `pl.MX_A_ZZ` / `pl.MX_B_NN`（`pl.TensorLayout` 枚举）；DN 在使用处由 `pl.transpose(x, -2, -1)` 导出，**形参不能声明 DN**；`pl.TileLayout.row_major/col_major/none_box`；`pl.TensorView(stride=, layout=, valid_shape=, pad=)`；`blayout` / `slayout` / `fractal`；`pl.set_validshape`、`pl.fillpad`、`pl.fillpad_expand`、`pl.PadValue`；`matmul(a_trans=, b_trans=)` |
| **Q** | `get_l0_fractal_alignment`、`get_l0c_m_alignment(dtype)`、`get_min_l0_tile_dim` —— 未开放 |
| **C** | 类型检查在第一个 pass 之前拒绝 layout 不一致；`PH-AT-007`（K 非 cube 分形倍数）；`ParserTypeError`（`pl.Tensor[..., pl.DN]` 简写已移除）；**pad 值选错则静默** |
| **A** | `BlockNzTensorViews`(15)、`BlockMxScaleTensorViews`(16)、`CanonicalizeTileSlice`(19)、`ResolveBackendOpLayouts`(22)、`MaterializeTensorStrides`(33)、`MaterializeValidShapeSymbols`(53) |
| **O** | △ 布局转换的实际代价只能从泳道 / L0 trace 间接推 |
| **R** | △ `tutorials/02-matmul` 覆盖 matmul 侧；NZ 预打包与 MX 侧无独立范式 |

### 3.3 做错的后果：一个被文档标为「致命陷阱」的静默错

**【事实】**（`03-memory.md`）

> **致命陷阱：** 对一块用零填充的部分有效 tile 做 `max` 规约，全负数的行会返回 0。没有报错也没有告警——只是恰好那些不满的行数字错了。请根据规约类型选择填充值：`max` 用 `PadValue.min`，`sum` 用 `0`。

**【推断】** 这是 `valid_shape` / `pad` 这组参数的典型形态：**它们是正确性参数，却长得像性能参数**。`shape` 放不下会报错，`pad` 选错不会。而尾块只在不整除时出现，所以单测很容易全绿。

### 3.4 缺口

- NZ 的六条前提条件今天只在文档里，**不是可查询的谓词**——开发者无法在代码里问「我这个张量能不能标 NZ」。
- 规约 × pad 值的配对是可静态判定的（`row_max` 的输入 tile 若 `pad` 为 0 且有 `valid_shape`，就该提示），但**没有检查**。这是 C 列一个成本低、收益明确的空位。

---

## 4. 域 4 · 访存粒度与带宽

### 4.1 硬件事实

**【事实】** 三个数，分属三条通路，不可混用：

| 数 | Ascend910B | Ascend950 | 属于哪条通路 |
| -- | ---------: | --------: | ------------ |
| GM access granularity | 512 B | **128 B** | `tile.load` / `tile.store` 的 DMA |
| L2 cache line | 512 B | 512 B | GM 侧穿越 L2 |
| 推荐最内维字节 | 512 B | **128 B** | `PH001` 的阈值 |
| 核内数据 cache line | 64 B | 64 B | `pl.read` / `pl.write` 的标量通路（域 1） |

**【事实】惩罚量级是实测的**：`b_trans` matmul 的 GM→Mat 权重 load 搬 128 B 的行而推荐值是 512 B，**实测 16–25% 性能损失**。

### 4.2 PyPTO 承载

| 形态 | 内容 |
| ---- | ---- |
| **E** | `pl.set_cache_policy(tensor, pl.CachePolicy.BYPASS)` —— 让流式操作数绕过末级缓存。写在 `pl.at(...)` / `pl.spmd(...)` 作用域体的**顶层**（不必第一行），覆盖该作用域内对该张量的每一次读取 |
| **Q** | `get_gm_access_granularity_bytes`、`get_l2_cache_line_bytes`、`get_recommended_innermost_dim_bytes` —— 未开放 |
| **C** | **`PH001` / `TileInnermostDimGranularity`** —— 本域是 C 列唯一做到成熟的一域，形态见下 |
| **A** | span 指向流水线后 IR 文本位置（`<string>:line:col`），非原始 DSL（#1305 第 2/3 项） |
| **O** | L0 trace 的 MTE2 泳道；`instr_metrics.json` 的逐指令 cycle |
| **R** | △ |

**【推断】`PH001` 应当作为其余八域 C 列的模板**，它已经解决了「性能提示如何不变成噪声」的四个工程问题：

1. **阈值来自单源。** 经 `PassContext::Current()->GetBackendHandler()` 读 `GetRecommendedInnermostDimBytes()`，不写死在检查里。
2. **报传输规模，不只报形状。** `moves <total>B as <rows> x <row>B rows`。文档明确「按 `moves …` 子句排序，不要按条数」——`[1024, 64]` 权重面板与 `[16, 64]` 激活面板的 span、op、dtype、最内维、内存空间完全相同，只差 64 倍流量。
3. **按 11 元组去重**：`(file, line, col, op, dtype, innermost_bytes, innermost_elems, target_memory, rows, row_bytes, transfer_bytes)`，即消息渲染的全部事实。循环/分片展开折叠为一条带计数的 hint，不同传输各自保留。
4. **输出分流**：有 `ReportInstrument`（`compile()` / `pl.jit` 默认）时写 `perf_hints.log`，stderr 只一行摘要；可用 `disabled_diagnostics` 整体关闭。

**【事实】** 它还纠正过一次自己的覆盖范围：早期版本以 L2 局部性为由跳过 `Mat`/`Left`/`Right`/`Acc` tile，#2309 取消了这个跳过——理由是本检查只看 Tile load/store 及对应 Buffer 操作，而它们的非 tile 一侧总是 `TensorType`（片外），所以无论 tile 落在哪个空间都是 GM↔片上传输；真正的片内搬移（`tile.move` / `tile.extract`）本来就不在检查范围内。那个跳过只会压制真阳性。

### 4.3 缺口

- **span 落点**（#1305）：提示指不到用户写的那一行，也没有命名控制最内维的 chunk 常量。
- 主文把本域的 E 列记为「—」，这是错的：`set_cache_policy` 正是本域的 E 列内容。已在主文修正。

---

## 5. 域 5 · 跨核通道与同步

### 5.1 硬件事实

**【事实】**（`reference/pto-isa/00-cluster_architecture.md`）

**Flag：**

| 属性 | 值 |
| ---- | -- |
| 每方向每对等核心的 flag 数 | 8 |
| 每对等核心的方向数 | 2（V→C、C→V） |
| 每个 Vector-Cube 对 | 16 |
| **每个 Cluster 总计** | **32**（2 对等核心 × 16） |

**Ring buffer：** 多槽环形缓冲，每槽容纳一个固定大小 Tile，带流控。

| 属性 | 值 |
| ---- | -- |
| 单向 | 8 槽 |
| 双向 | 每方向 4 槽 |
| **默认（每活跃方向）** | **2 槽** —— 刚够对这次交接做双缓冲，同时给 tile 本身留出片上空间 |

**位置随平台变**：

| 平台 | 位置 |
| ---- | ---- |
| A2/A3 | **全局内存（GM）** —— 需要额外的 `__gm_pipe_buffer` |
| A5 | **消费者片上 SRAM** —— Vector 消费者用 UB，Cube 消费者用 L1；消费者可直接操作本地槽，无需从 GM 显式加载 |

**同步的边界**：**【事实】**「`syncall` 本身只对齐**到达**」。cache 维护是另一回事，必须自己做。

**【事实】** soft `syncall` 的 GM workspace 约需 `used_cores × 8 × sizeof(INT32)`；`mix` 形态还要 Vec scratch + Mat scratch，且 Mat scratch 必须 `flat_layout=True`，否则计数槽位置会错位。

### 5.2 PyPTO 承载

| 形态 | 内容 |
| ---- | ---- |
| **E** | `tpush` / `tpop`；`optimizations=[pl.cross_core_slot(slot_num=N)]`；`pl.system.syncall(mode=pl.SyncAllMode.SOFT/HARD, core_type=pl.KernelType.MIX/AIC/AIV, gm_workspace=, used_cores=)`；`pl.system.sync_set/sync_wait(id, pipe=pl.PipeType.*, core_type=)`；`pl.system.cacheinvalid()`、`pl.system.fence()` |
| **Q** | `requires_gm_pipe_buffer`、`requires_split_load_tpop_workaround`、`splits_cube_to_vector_transport_in_hardware` —— 未开放 |
| **C** | **✗ 无 PerfHint**；GM 交接的定序**完全由用户负责** |
| **A** | `InjectGMPipeBuffer`(25)、`StampTfreeSplit`(27)、`SkewCrossCorePipeline`(29)、`AutoDeriveTaskDependencies`(43)、`ExpandManualPhaseFence`(44)、`InsertCommFence`(52) |
| **O** | 泳道可显示 flag SET/WAIT 配对与 ring 槽占用；`dfx_outputs` 下已有真实数据 |
| **R** | `distributed/09-barrier`；`04-scopes.md` 给出的保守序列 |

**【事实】保守的 GM 交接序列**（文档原文给出，可直接复用）：

```python
pl.system.cacheinvalid()  # 发布 producer 的全部 cache line
pl.system.fence()         # 等待它们对 GM 可见
pl.system.syncall(
    mode=pl.SyncAllMode.SOFT,
    core_type=pl.KernelType.MIX,
    gm_workspace=sync_ws,      # 独占且已清零的 16 元素 INT32 GM tensor
    used_cores=participant_count,
)                         # 只同步到达
pl.system.cacheinvalid()  # consumer 读之前使 cache 失效
```

### 5.3 做错的后果

| 失败模式 | 表现 | 证据 |
| -------- | ---- | ---- |
| 多余同步 | **A5 上挂死** | #828：一个 `dst == src` 的 identity `pto.tmov` 导致 ptoas 插入多余同步 |
| cache 维护用了窄形式 | 跨多条 cache line 的 buffer 读到旧数据 | 【事实】「tensor-region overload 当前只覆盖 view 基地址所在的那一条 cache line」——跨多 line 必须用全 GM 的 `pl.system.cacheinvalid()` |
| 硬 `syncall` 的启动规模写成字面量 | 不满占用时行为不符 | 【事实】应传 `pl.system.available_cluster_count()`（混合 / 纯 cube）或 `available_aiv_count()`（纯 vector），**并在调用点内联书写** |
| ring 槽不足 | 生产侧核跑不超前 | 默认每方向 2 槽；调高需要片上空间 |

### 5.4 缺口

**【推断】本域是「做错会 hang」且「C 列为空」的唯一组合**，风险权重最高：

- 没有任何检查回答「我这段 GM 交接的 publish / fence / barrier / invalidate 序列是否完整」。而这个序列有四步、顺序固定、漏一步就是读旧数据。
- 没有任何检查回答「ring 槽数 × tile 大小是否超了片上余量」——这与域 6 的 `PH-MR-001` 是同一类问题，但跨核环不在它覆盖范围内。
- 代际差异在本域最大（ring 在 GM 对片上 SRAM），而 `requires_gm_pipe_buffer` 这条开关对用户不可见。

---

## 6. 域 6 · 时间与流水

### 6.1 硬件事实

**【事实】** 六条硬件流水，`PipeType` 枚举为 `MTE1` / `MTE2` / `MTE3` / `M` / `V` / `S`。L0 trace 的泳道按**数据流顺序**排列：

```text
MTE2 → MTE1 → CUBE → VECTOR → FIXPIPE → MTE3
```

**【事实】** 流水深度花的是片上缓冲空间：「循环体所暂存的每一块 buffer 都要有 `stage` 份同时存活。**这是片上内存耗尽最常见的单一原因。**」深度通常取 2–4；外层循环以 `stage * step` 推进，尾部有一次 tail dispatch 覆盖余数。

### 6.2 PyPTO 承载

| 形态 | 内容 |
| ---- | ---- |
| **E** | `pl.pipeline(n, stage=N)`（编译器托管）；`pl.MemRef("name", slots=N)[i % N]`（手工放置）；`pl.parallel(n)`；`sync_set/wait` 的 `pipe=` |
| **Q** | **✗** 没有接口回答「这个形状在这个 target 上放得下几级」 |
| **C** | **`PH-MR-001`**（见下）、`PH-DSA-001` |
| **A** | `LowerPipelineToSlots`(30)、`LowerPipelineLoops`(31)、`SkewCrossCorePipeline`(29) |
| **O** | **L0 指令级 trace**：`incore-profiling` skill（`pypto-user` 插件）→ `python -m pypto.tools.clean_sim_trace` → `trace.clean.json`（Perfetto）+ `instr_metrics.json`（逐指令 pipe / cycle / vector 利用率） |
| **R** | `tutorials/05-scheduling-tuning`、`performance/04-incore`、`examples/advanced/01_split_k.py` |

**【事实】`PH-MR-001` 的消息形态**（`memory_reuse_pass.cpp:3927` 附近构造）——它是 11 条 hint 里写得最完整的一条：

```text
software pipelining requested depth 4 for pipeline group <g> in <space>, but only 2
of 4 buffers fit (<slot> B per stage, <free> B free [after <reserved> B reserved])
— stages 2 apart share storage and serialize.
```

后接二分支的可执行建议：

- **slot-bound**（这一个操作数自己就放不下）：「This operand alone needs <need> B for depth 4; shrink the per-stage tile to <= <free/requested> B, or reduce the pipeline depth to 2.」
- **空间争用**（它自己放得下，是共存 buffer 把空间占了）：「relieve the co-residents (smaller or fewer co-live tiles) or reduce the pipeline depth to 2.」

**【事实】** 措辞刻意是来源无关的：`pipeline_membership` 既由显式 `pl.pipeline(stage=)` 打上，也由编译器合成的流水（如跨核 skew 克隆）打上，所以消息归咎于「software pipelining」而非用户的 `stage=`，只把 `pl.pipeline(stage=)` 作为例子给出。

**【推断】** 这条提示的读法——「你要了 4，你拿到 2」——是**归因型**而非建议型：它报告的不是「你写慢了」，而是「编译器没能满足你写的东西，以及为什么」。同一形态在 `PH-AT-*` 里出现 8 次。见第 10.2 节。

### 6.3 做错的后果：显式 slot 与内存规划器的交叉风险

**【事实】** `pl.MemRef(slots=N)` 的下降形式取决于 `memory_planner`：

| 规划器 | slot 下降形式 | 一次迭代内两个 slot 同时存活 |
| ------ | ------------- | ---------------------------- |
| `PYPTO`（默认） | 烘焙地址（`alloc_tile`） | 支持 |
| `PTOAS` | `alloc_multi_tile` 区域 + 每次使用一个 `multi_tile_get` | **codegen 拒绝** |

**【事实】** 这个拒绝是刻意的：ptoas 在循环里同步两个同时存活的 slot 时出过错。较早版本不保护第二个 slot 的 load，使它与下一次迭代的写入竞争，**在设备上被实测出错**。当前版本仍会把其中一种形状——「比读取提前一轮迭代填充的 slot」——同步错，结果是**算错或设备挂死**（PTOAS#1519，ptoas 0.63 修复，**0.64 起又回退**）。

**【推断】** 这是一条跨仓库的、带版本窗口的正确性约束，而它今天只存在于一页用户文档的一段散文里。它满足主文的三条判据（裁决正确性、开发者推不出、无结构化承载），是 Q 列应当承载的典型内容——`memory_planner` × ptoas 版本 × slot 形状的可查矩阵。

**【事实】** 另一条易被误解的：显式 slot **不是** `pl.pipeline` 的等价写法。「`pl.pipeline` 把循环重构成一个调度，而 slot 只是消掉了会阻止重叠的同缓冲冲突。循环本身仍是顺序的」——重叠与否要去 L0 trace 确认，而不是假定换个写法就买到了。

### 6.4 缺口

- **Q 列全空。** `PH-MR-001` 已经在编译期算出了「每级 B / 空闲 B」，但开发者在写码期拿不到同一个数。
- L0 trace 依赖插件 + 支持 TL 的 CANN + `msopprof` worker 三项前置，`clean_sim_trace` 在仓内但 `incore_profile.py` 在插件里——**观测链路跨了仓库边界**。

---

## 7. 域 7 · 数值与精度

### 7.1 硬件事实

**【事实】一次 `pl.cast` 是否是一条指令，取决于目标架构。** `INT32 → FP16` 在 Ascend910B 上是一条硬件 `pto.tcvt`，在 Ascend950 上降为 `INT32 → FP32 → FP16`。每一跳花一次 `tcvt`；当中间类型比源类型更窄时，结果可能与直接舍入相差目标类型的 **1 ULP**。「**这是预期行为，不是缺陷**」，各架构对照表在 `LegalizeTileCast`(17)。

**【事实】饱和模式的默认值由硬件支持决定。** `saturation_mode` 对**整数**目标默认 `"on"`——理由有两条：没有任何标准规定「转换到整数时溢出」该产生什么，钳制更安全；而且「在 A2/A3 上它正是硬件原生支持的转换」。**浮点**目标保持 IEEE（窄化溢出产生无穷），除非显式要求 `"on"`。多跳 cast 只把该模式作用于最后一跳。

**【事实】MX（Ascend950）的组合是枚举式的，不是正交的**：

- `pl.matmul_mx` 及其 `_acc` / `_bias` 变体，每个操作数各接一块 data tile + 一块 scale tile
- 进入算子的两块 **data tile 必须都是 `FP8E4M3FN`**
- FP4 输入**仅支持左侧 FP4 × 右侧 FP8**，且必须在 `matmul_mx` 前显式写 `pl.cast(fp4_tile, pl.FP8E4M3FN)`；A5 的 cast legalization 将其展开为 `FP4 → BF16 → FP32 → FP8E4M3FN`
- **原生 FP4×FP4 与反向 FP8×FP4 均不支持**
- `pl.quant_mx` 本版本仅 MXFP8，`group_axis` 对齐 PTOAS 的 `grpAxis`
- 常规 MX 子视图（`slice`、`reshape`、`transpose`、`reinterpret_view`）与 MX `remote_load` 被拒绝；例外是 FP8E8M0 的 `pl.tensor.view` 可在 packed ND backing 与 `MX_A_ZZ`/`MX_B_NN` 之间建立逻辑 rank-2 alias（PTOAS v0.60 负责物理打包）

**【事实】** `pl.FP4E2M1X2` 是**打包 carrier**（双 nibble，8 bit），物理末维对齐 `torch.float4_e2m1fn_x2` / `!pto.f4E2M1x2`；FP16↔INT4 转换没有配套的 packed load/store carrier ABI。

**【事实】** 远程原子累加 `AtomicType.Add` 要求目标为 `fp32`/`bf16`/`fp16`/`int32`/`int16`/`int8`，其中 **`bf16` 目标仅 Ascend910B（A2/A3）支持**。

### 7.2 PyPTO 承载

| 形态 | 内容 |
| ---- | ---- |
| **E** | dtype 常量；`pl.cast(..., saturation_mode=)`；`pl.matmul_mx` / `_acc` / `_bias`；`pl.quant_mx(group_axis=)`；`out_dtype=`；`pl.MX_A_ZZ` / `pl.MX_B_NN` |
| **Q** | `honors_high_precision_algorithm`、`supports_mat_to_bias_move(src_dtype, bias_dtype)` —— 未开放。后者尤其值得开放：它是**逐 dtype 对**的后端能力查询 |
| **C** | △ 类型检查拒绝不支持的 MX 组合与 layout；`PH-AT-003`（sub-byte dtype 导致 AutoTile 放弃）、`PH-AT-011`（后端不支持该 Mat bias → Bias dtype 对）；**无精度类 PerfHint** |
| **A** | `LegalizeTileCast`(17)（带各架构对照表）、`InsertMxScaleAddr`(21) |
| **O** | `user/precision/00-workflow`（与 torch 对齐流程）、`01-cases` |
| **R** | `user/precision/01-cases`、`dev/fp4.md` |

### 7.3 做错的后果

**【事实】** split-K 的累加顺序与非切分版本不同，**用了原子加之后这个顺序甚至逐次运行都不固定**。文档的处理是：「要预期末位差异；在把它当 bug 之前，先核对累加顺序。」

**【推断】** 本域的失败模式与其他八域不同：它**不产生错误，产生差异**。因此 C 列在这里的正确形态不是「拒绝」，而是「声明预期误差量级」——而这正是今天缺的那一块。`precision/00-workflow` 给了对齐流程，但没有「这个 cast 链 / 这个 MX 组合在这个 target 上，预期误差上界是多少」的可查答案。

---

## 8. 域 8 · 任务与运行时

### 8.1 运行时事实

**【事实】** 运行时**没有一个统一的任务资源池，它有四个互相独立的**，由作用域嵌套深度选择：

```text
ring_idx = min(scope_depth, 3)

作用域深度 0 ──► ring 0 ─┐
作用域深度 1 ──► ring 1  │  各自拥有任务槽窗口、输出堆、依赖边池，
作用域深度 2 ──► ring 2  │  各自按 FIFO 独立回收
作用域深度 3+ ─► ring 3 ─┘
```

每个环是独立映射，有自己的游标与 FIFO 回收指针——「深作用域里的一个短命任务，不必等着顶层的长命分配才能被回收」。

每环三种资源：

| 资源 | 装什么 | 耗尽时 |
| ---- | ------ | ------ |
| `task_window` | 在飞的任务槽 | 一条点名 task window 的容量错误 |
| `heap` | 输出自动分配的字节 | 分配失败 |
| `dep_pool` | 依赖边条目 | 一条点名 dep pool 的容量错误 |

**定尺寸约束**（`RunConfig`，各接受标量或**恰好 4 个** int 对应 ring 0..3，`0` 表示该环保持默认）：

| 字段 | 单位 | 每项约束 |
| ---- | ---- | -------- |
| `ring_task_window` | 在飞任务槽 | **2 的幂，≥ 4** |
| `ring_heap` | **字节** | **2 的幂，≥ 1024** |
| `ring_dep_pool` | 依赖边条目 | `[4, INT32_MAX]` |

**【事实】** 进程级 `PTO2_RING_*` 环境变量已退役、不再被读取，`RunConfig` 是唯一途径。

### 8.2 两头都会翻车的饱和映射

**【事实】默认放置由编译器拥有**：`MaterializeRuntimeScopes`(50) 把整个函数体、以及每个 `for` 体和每个 `if` 的 then/else 体各自包进一个 AUTO 作用域。「这是个合理的默认——但它意味着你的环分配是**控制流形状的副作用**，而不是你决定的任何事情。」

两种失效，方向相反：

| 形态 | 后果 | 文档原文 |
| ---- | ---- | -------- |
| **扁平 kernel**（一个函数体，没有值得包裹的循环或分支） | 所有东西塞进 ring 0，ring 1–3 完全闲置但**仍然被映射** | 「你付了钱，什么也没拿到。失效方式是不对称且不友好的：ring 0 撞到天花板报出容量错误，而隔壁四分之三的资源空着。」 |
| **嵌套过深** | 深度 3、4、5… 全落到 ring 3，而那正是数量最多、最会溢出的那批任务 | 「更深的作用域全都堆到同一个环上」 |

**【事实】** 手工再平衡的关键是**深度而非并列**：「两个**同级**作用域做不到这一点——`ring_idx` 只由深度决定。」`auto_scope=False` 下该 pass 什么都不插，这个函数里每一个作用域都归你放；但它是**纯放置决策**，AUTO 作用域仍保持自动依赖跟踪开启，再平衡环不改变依赖语义。

### 8.3 PyPTO 承载

| 形态 | 内容 |
| ---- | ---- |
| **E** | `pl.scope()`、`auto_scope=False`（`@pl.jit` / `.host` / `.inline` 接受；`.incore` / `.opaque` 拒绝）、`pl.submit(deps=)`、`pl.manual_scope`、`RunConfig(ring_task_window=, ring_heap=, ring_dep_pool=)` |
| **Q** | △ 三个字段的约束（2 的幂、下界）写在文档里，不是可查数据 |
| **C** | **✗ 无 PerfHint**；溢出时运行时自己报错并给建议——「raise `ring_task_window`（`runtime_env.ring_task_window`）or split the scope」 |
| **A** | `MaterializeRuntimeScopes`(50)、`DeriveCallDirections`(42)、`AutoDeriveTaskDependencies`(43)、`ClassifyIterArgCarry`(51)、`LegalizeGraphBoundary`(49) |
| **O** | **本域 O 列最成熟**：`RunConfig(enable_scope_stats=True)` → `<work_dir>/dfx_outputs/scope_stats/scope_stats.jsonl`（NDJSON：第 1 行运行元数据，之后每行一个作用域样本；元数据行的 `task_window_max` / `heap_max` / `dep_pool_max` 是**按 ring 0..3 索引的数组**）→ `python runtime/simpler_setup/tools/scope_stats_plot.py` |
| **R** | `user/tasks/03-tuning`、`performance/01-task-granularity`、`02-runtime-overhead`、`03-dependencies`、`tutorials/04-task-graph` |

**【推断】** 本域是九域里**唯一形成闭环**的一个：度量（`scope_stats`）→ 再平衡（`auto_scope=False` + 深度）→ 确认（新一份 `scope_stats.jsonl` 的峰值不再顶容量）。文档还明确了修法的优先级——「给环加尺寸也是**第二顺位**的修法：一个因为某个作用域里塞了上千个任务而溢出的任务窗口，拆成两个作用域比把它撑大更好」。

### 8.4 缺口

- **C 列为空，但这一域的信息在编译期是已知的。** 作用域深度在 `MaterializeRuntimeScopes` 之后完全确定，「这个 kernel 的任务全落在 ring 0，ring 1–3 闲置」是可静态判定的——这是 C 列又一个成本低、收益明确的空位。
- 依赖推导的盲区（`tasks/00-model.md`「推导看不见什么」）与域 2 的 GM 定序盲区是同一个问题在不同层级的投影，但两处文档互不交叉引用。

---

## 9. 域 9 · 多卡与通信

### 9.1 系统事实

**【事实】** `window buffer` 是**每 rank 一份**的 HCCL 缓冲区，`alloc_window_buffer(size)` 的 size **以字节为单位**（有 `(shape, dtype=)` 便捷重载自动算 `prod(shape) × dtype.get_byte()`）。`name` 由解析器从赋值左侧注入，不要手动传。

**【事实】API 的可用位置是硬分的**：

| 原语 | 只能在哪 |
| ---- | -------- |
| `pld.system.world_size` | **仅 host** |
| `pld.system.rank` / `nranks` | **仅 InCore** |
| `pld.system.notify` / `wait` | **仅 InCore**，仅副作用 |
| `pld.system.defer_wait` | **仅专用的顶层 `pl.at(CORE_GROUP)` waiter** |
| `pld.system.get_comm_ctx` | host 编排器与 InCore 均可 |

**枚举语义**：

| 枚举 | 取值与语义 |
| ---- | ---------- |
| `NotifyOp` | `AtomicAdd`（原子递增对端信号槽，多 rank 屏障）/ `Set`（覆盖，1:1 握手） |
| `WaitCmp` | `Eq` / `Ge` |
| `ReduceOp` | `Sum` / `Max` / `Min` / `Prod`；`allreduce` 支持全部四种，`reduce_scatter` 在 InCore 通道支持四种而 **HOST builtin 通道目前仅 lower `Sum`** |
| `AtomicType` | `None_` / `Add`；`Add` 的目标 dtype 受限，**`bf16` 仅 Ascend910B** |

**【事实】分块与流水的耦合约束**：`chunk_rows` / `chunk_cols`（`0` 表示完整范围）缩小 staging tile，让超过片上 staging 预算的传输仍能一次调用完成。

> **致命陷阱：** `pipeline=True` **要求 `chunk_rows > 0` 且 `chunk_cols > 0` 同时成立**——双缓冲的收益只有在传输确实被分块时才存在。若任一仍为 `0`，会在派发前抛 `ValueError`。

**动态**传输范围必须由匹配的**静态** chunk 限定：动态最内维要求设 `chunk_cols`，动态最外维要求设 `chunk_rows`——**staging tile 是静态分配的，无法按运行时值确定大小**。

**【事实】** `defer_wait` 的价值是「释放物理核，保留逻辑任务」：注册 `signal[offsets] >= expected` 完成条件并返回，不让 AIV 自旋；条件满足前所在任务的 TaskId 保持未完成。

### 9.2 PyPTO 承载

| 形态 | 内容 |
| ---- | ---- |
| **E** | `pld.system.*`（`world_size` / `rank` / `nranks` / `get_comm_ctx` / `notify` / `wait` / `defer_wait`）；`pld.tensor.*`（`window` / `alloc_window_buffer` / `put` / `get`）；`pld.tile.*` RMA；`DistributedTensor`、`CommCtx`、`ReduceOp`、`AtomicType` |
| **Q** | **✗** |
| **C** | **✗**（`pipeline=True` 的 chunk 前置是派发前 `ValueError`，属于参数校验而非硬件诊断） |
| **A** | `LowerL2TensorCollectives`(40)、`LegalizeSpmdLaunches`(41)、`SynthesizeAllreduceSignals`(45)、`MaterializeCommDomainScopes`(46)、`LowerHostTensorCollectives`(47)、`MaterializeDistTensorCtx`(48)、`InsertCommFence`(52) |
| **O** | `user/distributed/04-debugging` |
| **R** | **九域里 R 列最厚的一域**：22 篇文档，含 `13-allreduce_mesh`、`14-allreduce_two_phase`、`15-allreduce_ring`、`16-allreduce_reveal`、`17-broadcast`、`18-allgather`、`19-reduce_scatter`、`20-all_to_all`、`21-putting_it_together`，并明确「每个内置集合通信都是底层原语的组合」（mesh allreduce = stage-in → barrier → remote-accumulate → stage-out） |

**【推断】** R 列为什么在这里独厚，而 Q/C 两列为什么空，是同一个原因：**集合通信有「唯一正确写法」，所以范式库是对的承载形态**；而通信的硬件裁决量（window 大小、信号槽、staging 预算）跨 host 与 device 两侧，没有一个对象把它们收拢。

### 9.3 做错的后果

**【事实】** 本域最严重的失败模式继承自域 2：两条 AIV lane 都执行同一条 `pld.system.notify`，`NotifyOp.AtomicAdd` 让对端计数器在只有一个 rank 到达时就读到 `2`，原本要等两次到达的 `pld.system.wait` 被一次放行，该 rank 抢跑读到数据尚未落盘的缓冲区。

> 症状是多 rank 运行时某个 rank 间歇性算错；**单 rank 复现不了，看上去也不像同步问题。**

**【事实】** 而「限定到 lane 0」这种修法带有分片写法所没有的定序义务：两条 AIV lane 异步运行，没有机制为它们定序，所以被限定在 lane 0 的 notify 可能在 lane 1 的写入落盘**之前**就发布信号。只有当该信号释放的数据全部由 lane 0 自己写入时才安全。

---

## 10. 三张横切表

### 10.1 PerfHint 的域覆盖

**【事实】** 11 个码按域归位：

| 域 | 码 | 覆盖什么 |
| -- | -- | -------- |
| 1 存储容量 | `PH-AT-003/005/006/008/009/010/011`、`PH-MR-001`、`PH-DSA-001` | L0 容量选择、流水深度 vs 容量、地址分配放弃流水分离 |
| 3 布局分形 | `PH-AT-007` | K 非 cube 分形（16）倍数 |
| 4 访存粒度 | **`PH001`** | 最内维字节数低于推荐值 |
| 6 时间流水 | `PH-MR-001` | 「你要了 N 级，拿到 M 级」 |
| **2 计算单元** | **—** | |
| **5 跨核同步** | **—** | |
| **7 数值精度** | **—** | |
| **8 任务运行时** | **—** | |
| **9 多卡通信** | **—** | |

**【推断】** 覆盖与风险严重**错配**：有提示的四域，做错的后果主要是「慢」或「编译器放弃优化」；没提示的五域里，域 2 / 5 / 9 做错的后果是**静默错或挂死**。

### 10.2 一个值得单独命名的提示类别：归因型 PerfHint

**【推断】** 11 条里有 9 条（8 条 `PH-AT-*` + `PH-DSA-001`）说的不是「你写得慢」，而是「**编译器本来要替你优化，但放弃了，原因是 X**」。`PH-AT-*` 的消息几乎都以 `— left untouched` 结尾：

| 码 | 放弃的原因 |
| -- | ---------- |
| `PH-AT-003` | 操作数或累加器是 sub-byte dtype |
| `PH-AT-005` | `ChooseL0Tile` 拒绝该配置 |
| `PH-AT-006` | 需要 N 切分，但 bias 源已在架构 Bias 缓冲里（形成 N 子窗口需要不支持的 Bias→Bias 传输） |
| `PH-AT-007` | K 不是 cube 分形 16 的倍数，无合法 K 分块 |
| `PH-AT-008` | `ChooseL0Tile` 走了 fallback |
| `PH-AT-011` | 多种契约外情形：Vec 驻留的左矩阵操作数、后端不支持该 Mat bias → Bias dtype 对、Mat bias 不是单次使用的完整矩形 `[1, N]` load…… |

**【推断】** 这类提示在主文的六形态里横跨 **C（它是编译期发出的诊断）与 A（它的内容是编译器决策的归因）**，并且它比两者单独都更有价值：它同时告诉开发者「这里没被优化」和「把输入改成什么样就会被优化」。**建议把它作为 C 列扩展时的首选形态**——比凭空新增「你写慢了」类提示更容易做对，因为编译器放弃的那个分支本来就知道原因，只是今天大多数分支选择了静默返回。

### 10.3 代际差异按域展开

**【事实】** 主文指出 `feature-matrix` 只有 6 行特性级差异。按域展开后：

| 域 | Ascend910B（A2/A3） | Ascend950（A5） |
| -- | ------------------- | --------------- |
| 1 存储 | L0C **128 KiB**；无 `LeftScale`/`RightScale`；`Vec → Mat` **不实现** | L0C **256 KiB**；有 MX scale 空间；`Vec → Mat` 实现 |
| 2 单元 | — | `splits_cube_to_vector_transport_in_hardware` 等开关取值不同 |
| 3 布局 | — | MX scale 布局 `MX_A_ZZ` / `MX_B_NN` 仅此代 |
| 4 访存 | GM 粒度 / 推荐最内维 **512 B** | **128 B**（L2 line 两代同为 512 B） |
| 5 同步 | ring 在 **GM**，需 `__gm_pipe_buffer`（`requires_gm_pipe_buffer`） | ring 在**消费者片上 SRAM**（Vector 消费者 UB / Cube 消费者 L1），不需要 |
| 6 流水 | — | `requires_level3_tmp_scratch` 等开关取值不同 |
| 7 精度 | `cast` 饱和为硬件原生；`AtomicType.Add` 支持 **bf16** 目标 | `INT32→FP16` 多跳展开；MX / FP4 系列仅此代；`bf16` 原子累加不支持 |
| 8 运行时 | `requires_runtime_subblock_bridge` 等开关取值不同 | 同上 |
| 9 通信 | `reduce_scatter` HOST 通道仅 `Sum`（两代同） | 同左 |

**【事实】** `BackendHandler` 的 7 条代际开关（`requires_gm_pipe_buffer`、`requires_split_load_tpop_workaround`、`requires_level3_tmp_scratch`、`requires_vto_c_fractal_adapt`、`requires_runtime_subblock_bridge`、`requires_no_split_dual_aiv_dispatch`、`splits_cube_to_vector_transport_in_hardware`）**无一条有用户文档**，而上表每一行的「—」都可能只是因为没人把对应开关翻译成用户语言。

---

## 11. 逐域小结：谁最该先补

按「做错的后果 × C 列是否为空 × 能否静态判定」排序：

| 域 | 做错 | C 列 | 优先级依据 |
| -- | ---- | ---- | ---------- |
| **5 跨核同步** | **hang / 静默错** | **空** | 四步交接序列漏一步就错，完全无检查；代际差异最大 |
| **2 计算单元** | **静默错** | 有结构检查，但三处明确不检查 | 「两条 lane 重复 notify」在域 9 放大成最难自查的失败 |
| **9 多卡通信** | **静默错** | **空** | 继承域 2 的风险；R 列最厚但 Q/C 全空 |
| 8 任务运行时 | 容量错误（会报） | 空 | 作用域深度编译期已知，可静态判定，成本低 |
| 3 布局分形 | 静默错（pad 值） | 部分 | 规约 × pad 配对可静态判定 |
| 1 存储容量 | 静默损坏（别名） | 已有 9 条 hint | 缺的是别名可视化，不是提示 |
| 6 时间流水 | 慢 / 跨规划器正确性 | 有 `PH-MR-001` | 缺 Q 列与跨仓版本矩阵 |
| 7 数值精度 | 差异（非错） | 部分 | 正确形态是误差上界，不是拒绝 |
| 4 访存粒度 | 慢 | **成熟** | 已是其余八域的模板 |

**【推断】** 这个排序与主文第 8 节的七项清单一致，并补上了一个主文没说清的点：**C 列扩展不应从「加性能提示」开始，而应从「把编译器已经知道、但选择静默返回的那些放弃原因打印出来」开始**（第 10.2 节）。`PH-AT-*` 证明这条路可行，且域 5 的 `ExpandMixedKernel`「只自动处理一种窄特例」——那个「其余情形都需要你处理」的分支，正是下一批归因型提示该出现的地方。

---

## 12. 复核方法

```bash
# 全部 perf hint 码（修正主文的关键命令）
grep -rhno "PH[-A-Z0-9]\{3,12\}" repo/pypto/src repo/pypto/include | sed 's/.*://' | sort -u
```

```bash
# hint 发出位置与所属 pass
grep -rn "DiagnosticSeverity::PerfHint" repo/pypto/src | sed 's/:.*PerfHint/ -> PerfHint/'
```

```bash
# 注册表里的 6 条检查及其 severity / phase
# （用 "DiagnosticCheck::" 过滤，否则会多出 Register 方法本身的定义那一行）
grep -n "Register(DiagnosticCheck::" -A2 repo/pypto/src/ir/verifier/diagnostic_check_registry.cpp
```

```bash
# MemorySpace 的 10 个值
grep -n "= [0-9]," repo/pypto/include/pypto/ir/memory_space.h
```

```bash
# BackendHandler 的全部用户可查项
sed -n '/^class BackendHandler/,/^def /p' repo/pypto/python/pypto/pypto_core/backend.pyi
```

> 本文只做事实梳理与缺口标注，不包含代码或产品实现。所有【事实】均给出可核对位置；`repo/pypto` 为镜像，更新后应按上述命令重新核对再引用。
