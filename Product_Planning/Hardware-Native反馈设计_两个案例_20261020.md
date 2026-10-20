# 把硬件约束翻译成反馈：两个案例

> 整理日期：2026-10-20
> 指导思路：**Hardware-Native 不是把硬件细节全部倒给用户，而是把硬件约束翻译成开发过程中及时、可理解、可行动的反馈。**
> 取材：`github_issues/` 的真实 Issue + `repo/pypto` 源码核实。Issue 未写到的部分按该思路补充与扩展，文中以【扩展】标注，与【事实】区分。
> 关联：[九域详解](PyPTO_Hardware-Native九域详解_20261019.md)、[失效类型分析](../Insight/Hardware-Native失效类型_基于1463条Issue_20261019.md)

---

## 0. 先把「好反馈」的判据定下来

这条思路里有三个词，每一个都是一道独立的关卡，缺一条反馈就失效：

| 判据 | 问题 | 不满足的典型样子 |
| ---- | ---- | ---------------- |
| **及时** | 反馈出现在用户还能便宜地改的时候吗？ | 写码期能判定的事，拖到设备运行时才以错误码出现 |
| **可理解** | 反馈说的是用户写的东西，还是编译器内部的东西？ | 报 IR 里某个 buffer 超限，而用户不知道那个 buffer 是自己哪一行造出来的 |
| **可行动** | 照着它做，问题真的会消失吗？ | 给了建议，用户照做，仍然失败 |

**第三条最容易被高估。** 一条措辞友好、解释清楚、还给了两个修法的错误信息，如果那两个修法都解决不了问题，它比沉默更糟——它消耗了用户的信任和时间。Case 2 就是这种。

### 一个已落地的正面参照：把不可信的规格折成安全上界

**【事实】** `repo/pypto/src/backend/common/soc.cpp:131-139`，910B 的 AIV 核配置：

```cpp
// NOTE: the physical Vec UB is 192KB, but PTO-ISA reserves ~8KB at the top of the
// buffer that silently corrupts any tile placed there (pto-isa#170). We therefore
// cap the *safe* usable UB at 184KB so AllocateMemoryAddr raises an error before an
// allocation can reach the bad region, instead of producing NaNs on device.
// TODO(pto-isa#170): restore to 192ULL * 1024 (physical size) once PTO-ISA is fixed.
Core aiv_core(ir::CoreType::VECTOR,
              {
                  Mem(ir::MemorySpace::Vec, 184ULL * 1024, 128),  // 184KB safe (192KB physical)
              });
```

950 侧同样处理：物理 248 KB → 安全 240 KB。

**【推断】这段注释就是这条思路的教科书实现**，它做对了四件事：

1. **没有把细节倒给用户。** 用户从不需要知道「ISA 在 UB 顶部保留了约 8 KB」。
2. **把硬件事实折算进了一个用户会撞到的数。** 保留区变成了分配器上界。
3. **把失效时机从运行期提前到编译期。** 注释里写得很直白——「so AllocateMemoryAddr raises an error **before** an allocation can reach the bad region, **instead of** producing NaNs on device」。
4. **留了回退路径。** `TODO(pto-isa#170)` 指明了这是对一个上游缺陷的补偿，而不是真实规格。

下面两个案例，一个是这件事做成了（并且做得比原始诉求更远），一个是还没做成，且反面思路正在实际发生。

---

## Case 1 · 硬屏障遇上部分占用：设备挂死，而编译器本来就知道答案

**来源**：hw-native-sys/pypto#1935（closed）
**所属域**：域 2（计算单元与分工）× 域 5（跨核与同步）
**失效类型**：P2 设备挂死（见[失效类型分析](../Insight/Hardware-Native失效类型_基于1463条Issue_20261019.md)）

### 1.1 场景

一位开发者在 Ascend 910B 上写一个 SPMD 的逐元素加法。中间需要一道跨核屏障，确保所有 block 都读完输入再开始算。他按文档写了最自然的一版：

```python
TR = TC = 128
N = 24                                   # 这块卡上他习惯用的 block 数

@pl.function(type=pl.FunctionType.InCore)
def add(self, a, b, out):
    i = pl.tile.get_block_idx(); o = i * TR
    ta = pl.load(a, [o, 0], [TR, TC])
    tb = pl.load(b, [o, 0], [TR, TC])
    pl.system.syncall(core_type="aiv_only")      # 默认就是 hard（FFTS）形式
    out = pl.store(pl.add(ta, tb), [o, 0], out)
    return out

@pl.function(type=pl.FunctionType.Orchestration)
def orchestrator(self, a, b, out):
    with pl.spmd(N):                              # N = 24
        out = self.add(a, b, out)
    return out
```

### 1.2 他遇到的问题

**【事实】** 编译通过。上板运行：

```text
RuntimeError: run_prepared failed with code 507018
```

507018 是 AICore 超时。更难受的是二次代价——**【事实】** Issue 原文记载：「it leaves the device draining/reset（subsequent runs on that device fail with `code 13` until reset）」。也就是说，**一次挂死污染的不止这一次运行**，开发者接下来在这张卡上做的任何事都会失败，直到他意识到要重置设备。

**根因（硬件事实）**：hard 形式的 `syncall` 降级为 FFTS 屏障 `SYNC_AIV_ONLY_ALL`，它等待该 core_type 的**每一个物理核**到达。Ascend 910B 有 **48 个物理 AIV 核**。`pl.spmd(24)` 只启动 24 个 block，另外 24 个核永远不会到达屏障，FFTS wait 永不完成。

**【事实】** 这个诊断只有一个变量：把 `pl.spmd(24)` 改成 `pl.spmd(48)` → PASS，其余一字不动。

### 1.3 他当时能看到什么、看不到什么

| 他能看到 | 他看不到 |
| -------- | -------- |
| 一个六位数字 `507018` | 这个数字和 `syncall` 有关 |
| 设备超时 | 「hard 屏障要求满占用」这条规则 |
| 后续运行莫名返回 `code 13` | 48 这个数字，以及它来自哪里 |
| — | `mode="soft"` 这个出路的存在 |

**【事实】而这条规则其实已经写下来了**——在 `tests/st/runtime/cross_core/test_syncall.py` 的模块 docstring 里：

> This imposes two hard requirements on the launch: (1) **full occupancy** — it must fill **all** physical AIV cores…

**【推断】** 这是 Hardware-Native 失效的一个典型形态：**知识存在，但不在用户的路径上。** 它在一个 ST 测试的文件头注释里，而用户是在写 kernel，不是在读测试。

**【事实】更关键的是：编译器手上有全部判定材料。** `repo/pypto/src/backend/common/soc.cpp:143` 的 910B SoC 描述：

```cpp
Die die({{aic_cluster, 24}, {aiv_cluster, 48}});  // 24 AIC cores and 48 AIV cores per die
```

`48` 就在那里。`pl.spmd(N)` 的 `N` 也在 IR 里。Issue 原文把这句话说得很重：

> The compiler already knows both the launch block count (`pl.spmd(N)`) and the target SoC's physical core count per core type, **so this is statically checkable.**

### 1.4 反面解决思路

#### 反面 A：写一篇《FFTS 屏障语义》文档，让用户记住 48

**它看起来合理**：这确实是一条真实的硬件规则，写清楚它是负责任的做法。

**它错在哪**：

- **把硬件细节倒给了用户**，正是这条指导思路要避免的。用户为了写一个加法，被要求记住 FFTS 的 all-arrive 语义和一个芯片相关的常数。
- **48 是会变的。** 950 的几何完全不同（**【事实】** `soc.cpp:192`：18 个 mix cluster，每 cluster 1 AIC + 2 AIV），文档里写死 48 的那一刻，它对 A5 就是错的。
- **文档不在出错的那一刻出现。** 用户出错时看到的是 `507018`，不是这篇文档。违反「及时」。

#### 反面 B：改进运行时报错，把 507018 翻译成人话

**它看起来合理**：至少用户知道发生了什么了。这是很多团队的默认改进方向。

**它错在哪**：**时机不对，而且代价不对称。** 到这一步，用户已经付出了完整的编译 + 上板 + 运行成本，而且设备已经进入 draining 状态——他还要先学会重置设备。把一个编译期可判定的问题留到运行期解释，等于把判定成本转嫁成了用户的设备时间。违反「及时」。

**【推断】** 这条反面思路最危险，因为它**看起来像在改进反馈**，实际上固化了错误的反馈时机。

#### 反面 C：干脆默认改成 soft 模式，或者禁用 hard

**它看起来合理**：soft 模式在部分占用下能工作，默认安全。

**它错在哪**：**用安全换掉了能力。** hard 形式是 FFTS 硬件屏障，soft 形式是 GM 轮询 + 共享计数器——**【事实】** 后者需要一块独占且清零的 GM workspace（约 `used_cores × 8 × sizeof(INT32)`），并且每个参与者要轮询。在满占用场景下，为了防一个本可静态拦截的错误而让所有人付这个代价，是把个案成本摊给了全体。

**【推断】** 这是「抽象泄漏修补」的常见误区：**不是把漏洞堵上，而是把下面那层能力也一起埋掉**。与 `pto-isa` 的设计哲学（「不是隐藏底层能力，而是抬高抽象层级同时保留调优空间」）直接冲突。

#### 反面 D：编译器自动把 `pl.spmd(24)` 改写成 `pl.spmd(48)`

**它看起来合理**：编译器知道正确答案，直接修好不是更省事？

**它错在哪**：**它替用户改了语义。** `pl.spmd(N)` 不只是一个性能旋钮，它决定每个 block 处理哪一段数据——用户的 kernel 体里有 `i * TR` 这样的索引，block 数一变，数据划分就变了。编译器改 24 为 48，要么越界，要么默默改变了计算。

**【推断】** 判断边界很清晰：**编译器可以告诉用户「这两个数必须相等」，但不能替用户决定该动哪一个。** 24 是用户对数据划分的决定，48 是硬件事实；冲突的解法可能是改 24，也可能是改屏障模式，这是用户的取舍。

### 1.5 合理解决思路

**【事实】PyPTO 实际采取的做法**：新增 `HardSyncallOccupancy` 验证器（`repo/pypto/src/ir/verifier/verify_hard_syncall_occupancy.cpp`），把运行时的脚坑变成编译期错误。以下是它的设计要点，每一点对应一条判据。

#### 要点一（及时）：放在信息刚好齐备的那个节点

**【事实】** 验证器运行在 `ExpandMixedKernel` **之后**。原因写在文件头：

> It runs after ExpandMixedKernel, where each launched kernel's FunctionType is resolved (AIV / AIC / Group), which is what lets us map SPMD blocks to physical cores per launch shape.

**【推断】** 这是一个容易被忽略的设计决定：**「尽早」不等于「最早」。** 在 `ExpandMixedKernel` 之前，编译器还不知道这个 kernel 最终跑在 AIV、AIC 还是 Group 上，也就无法把 block 映射到物理核。放得太早会得到一个只能说「可能有问题」的检查，而那种检查会被用户关掉。**正确的时机是「信息刚好齐备的第一个节点」。**

#### 要点二（可理解）：消息里只出现用户写过的东西

**【事实】** 实际发出的消息（`OccupancyMessage`，verifier 第 212-216 行）：

```text
hard pl.system.syncall(core_type="aiv_only") requires the SPMD launch to fill all 48 AIV
cores (one block per physical core), but the launch has 24 blocks. Use mode="soft"
(GM-polling) for partial occupancy.
```

拆开看这条消息的结构：

| 成分 | 文本 | 作用 |
| ---- | ---- | ---- |
| **谁** | `hard pl.system.syncall(core_type="aiv_only")` | 用户写过的那一行，带着他写的参数 |
| **要求什么** | `fill all 48 AIV cores (one block per physical core)` | 硬件事实，**带上具体数字和单位换算** |
| **你给了什么** | `but the launch has 24 blocks` | 两个数并排，差距自明 |
| **怎么办** | `Use mode="soft" (GM-polling) for partial occupancy` | 一条可执行的替代路径 |

**【推断】** 注意它**没有**说的东西：没有提 FFTS、没有提 `SYNC_AIV_ONLY_ALL`、没有解释屏障的 all-arrive 语义、没有提 507018 的成因。这些都是真的，但用户不需要它们就能改对。**「不倒给用户」在这里是具体的删减动作，不是口号。**

#### 要点三（可行动）：给出的出路必须真的能走

**【推断】** 消息给的 `mode="soft"` 不是泛泛的「调一下参数」，而是一个确实能在部分占用下工作的模式——这是 soft 模式存在的理由。**可行动的判据是：用户照做，这条诊断会消失，且不会换来另一个问题。**

#### 要点四：诊断把问题想得比提问者更完整

**【事实】** 验证器覆盖的情形**超出了原 Issue 的诉求**。原 Issue 只说「部分占用」，验证器实际检查四种：

| 情形 | 诊断 |
| ---- | ---- |
| **占用不足或超额** | 上文的 `OccupancyMessage`。注意「over-occupancy 同样报错」——比要求更多的 block 也不对 |
| **满占用但缺 `sync_start`** | 见下 |
| **用错了查询** | 用 `available_cluster_count()` 给一个等 AIV 的屏障定规模 → 「the launch sizes itself on …, use … instead」 |
| **根本不可能满足** | 在纯 AIV kernel 里写 `core_type="mix"` → 「can never complete in a vector-only (AIV) launch: it waits for all CUBE cores, but a single-core-type SPMD launch provides none of them」 |

**【事实】其中「满占用但缺 `sync_start`」是原 Issue 没提到的第二个条件**，验证器的注释解释得很完整：

> Full occupancy fills every physical core, but the FFTS barrier also needs all those blocks *co-resident* — every core must be running this kernel at the same time. Only a sync_start launch guarantees that; without it the runtime may dispatch blocks in waves, so an early wave hits the barrier and waits for a later wave that cannot dispatch until the early one retires -> deadlock (507018). **Occupancy is necessary but not sufficient.**

对应的消息：

```text
hard pl.system.syncall(core_type="aiv_only") requires the SPMD launch to set sync_start=True
so all blocks are co-resident at the FFTS barrier; without it the runtime may dispatch blocks
in waves and the barrier deadlocks on device (507018). Add sync_start=True to the launch, or
use mode="soft" (GM-polling).
```

**【推断】** 这一条的价值在于：如果只按原 Issue 实现，用户把 24 改成 48 之后**仍然可能挂死**——然后他会认定这个检查没用。**一个只拦住一半的检查，比没有检查更损伤信任。**

#### 要点五：顺便把用户推向可移植的写法

**【事实】** 验证器接受两种「宽度」：字面量 `N`，或**启动形状查询** `pl.system.available_aiv_count()` / `available_cluster_count()`。文件头注释给出了一个本案例最有价值的判断：

> These resolve on device to the run's own geometry, so they fill their core type by construction and need no count comparison. **They are the portable spelling: a literal matching this SoC's static counts is wrong on any run whose device reports a different usable count** — too few blocks when more cores are available, too many when fewer are (which the runtime rejects outright).

**【推断】** 这是把反馈从「拦截错误」升级到「引导正确写法」：即使用户把 24 改成 48 并通过了检查，那个 48 在下一代芯片上、或在可用核更少的设备上**仍然是错的**（**【事实】** 对照 pypto#1173：硬编码 `block_dim=24` 在可用核更少的设备上挂死）。查询写法从构造上就对，所以检查对它直接放行、不作数值比较——**用放行本身给出了偏好**。

#### 要点六：不能判定时，明说不判定

**【事实】** 文件头列出了三类「不检查」的情形：启动宽度是 host 标量或算术表达式时「not classifiable, not checked」；没有 SPMD 启动的裸 hard-syncall kernel 不检查，因为「occupancy is a launch-time property」。

**【推断】** 这是诊断可信度的基础：**宁可漏报，不可误报。** 一个会在合法代码上报错的检查，最终会被整体关掉，连同它本该拦住的那些真问题。

### 1.6 这个案例的可复用模式

```text
硬件事实（48 个 AIV 核，FFTS 等全部到达）
   ↓ 不直接暴露
折算成一条编译期可判定的关系（launch 宽度 == 物理核数，且 co-resident）
   ↓ 在信息刚好齐备的 pass 之后判定
用用户的词汇报告差距（他写的那一行 + 两个数 + 一条出路）
   ↓ 同时
对天然正确的写法（查询）放行，用放行表达偏好
   ↓ 并且
判定不了的情形明说不判定
```

---

## Case 2 · 跨核环过量预留：报错了，但两条建议都没用

**来源**：hw-native-sys/pypto#2287（closed；**但本文核实：DSL 旋钮与分方向计量在当前镜像上仍未落地**）
**所属域**：域 5（跨核通道）× 域 1（存储容量）
**失效类型**：P4 硬件旋钮在抽象里消失 + P7 诊断把用户推向错误动作

### 2.1 场景

一位开发者在 DeepSeek v4-flash 的稀疏解码注意力上调 `qk_pv` 这个混合 kernel（cube + vector）。这个 kernel 的跨核流量是**高度不对称**的：

**【事实】** Issue 给出的两个方向的载荷：

| 方向 | 消费者 tile | 字节 |
| ---- | ----------- | ---- |
| V2C（AIV → AIC） | staged KV tile | 131,072 |
| C2V（AIC → AIV） | `qk_raw` `[32, 128]` FP32 | **16,384** |

一个方向比另一个方向大 **8 倍**。而 C2V 的消费者缓冲区住在 **UB**——**【事实】** 用 Issue 的原话，「the tightest space on the part」。

### 2.2 他遇到的问题

**【事实】** 编译在 `AllocateMemoryAddr` 处失败：

```text
ValueError: Vec buffer usage (213760 bytes) exceeds platform limit (188416 bytes).
The first 131072 bytes of that space are reserved by system.reserve_buffer, so tiles are
allocated above them - this is the cross-core pipe ring. Lower its depth with
optimizations=[pl.cross_core_slot(slot_num=N)] on the enclosing pl.at(...), or shrink the
tile that crosses the cube/vector boundary
```

**先给这条消息一个公道的评价**：它比绝大多数编译器错误都好。它说了超多少、上限是多少、**那块保留区是什么**（跨核 pipe 环），还给了两条具体的修法。按「可理解」这一关，它及格。

**但它在「可行动」这一关上失败了。**

**【事实】** Issue 记录了用户照做的结果：

> Even after forcing the minimum depth with `optimizations=[pl.cross_core_slot(slot_num=1)]`, the reserve is still 131,072（each per-direction buffer is sized by the common max），and the kernel remains **25,344 bytes over budget**。

**即：按第一条建议做到极限，预留一个字节都没少。**

### 2.3 为什么两条建议都没用

**【事实】** 真因有两层，用户两条都碰不到：

**第一层 · 深度算错了。** PTOAS 的规格（`ptoas/docs/PTO_IR_manual.md`）规定，消费者侧本地 FIFO 的字节数应当是 `slot_size × effective_local_slot_num`，其中 `local_slot_num` 是一个**独立的、更小的旋钮**。而 PyPTO 从不发出 `local_slot_num`，于是每条 local-FIFO pipe 都按**完整的 GM 环深度**预留（单向 8 / 双向 4）。

**这个参数不是不存在——它存在于除了 DSL 之外的每一层**：

| 层 | `local_slot_num` 在不在 | 本文核实位置 |
| -- | ---------------------- | ------------ |
| pto-isa（ISA） | **在** —— `DataFIFO<..., LocalDepth = 0>`，带 `localFiFoDepth` / `useLocalFiFo` | Issue 引 `pto-isa/include/pto/common/fifo.hpp:71-95` |
| PTOAS（规格） | **在** —— `pto.reserve_buffer.size` 的定义直接引用它 | Issue 引 `PTO_IR_manual.md:9199-9205` |
| PyPTO IR op | **在** —— `aic/aiv_initialize_pipe` 接受它 | 【事实】`python/pypto/ir/op/system_ops.py:558,570-571,584,616` |
| PyPTO codegen | **在** —— 能发出 `local_slot_num = N` | 【事实】`src/backend/common/pto_ops_crosscore.cpp:241-245` |
| 自动 pipe 构造 | **不发** | 【事实】`src/ir/transforms/utils/cross_core_pipe.cpp:349-357`，注释明说「local footprint = slot_num when local_slot_num is omitted」 |
| **DSL 旋钮** | **不在** | 【事实】`python/pypto/language/optimizations.py:136` —— `def cross_core_slot(*, slot_num: int)` |

**第二层 · 槽大小也算错了。** **【事实】** `GetCommonSlotSizeBytes`（`src/ir/transforms/utils/cross_core_pipe.cpp:190-204`，**本文核实当前仍是这个实现**）对两个方向取 `max`：

```cpp
common_slot_size = std::max(common_slot_size.value(), direction->slot_size_bytes.value());
```

而 ISA 按**各方向自己的**消费者 tile 大小索引本地落地区（Issue 引 `a2a3/TPush.hpp:446-448` 与 `:464-466`）。于是**一个方向上的大载荷把另一个方向的缓冲区撑大了**：V2C 的 131,072 决定了 C2V 在 UB 里也按同一个尺度预留，而 C2V 真正需要的只有 16,384 —— **8 倍过量，而且发生在最紧的那块内存上**。

**【推断】所以那条错误信息的两条建议，在这个 kernel 上分别是：**

| 建议 | 实际效果 |
| ---- | -------- |
| `pl.cross_core_slot(slot_num=N)` 降深度 | **无效**。深度确实被 `slot_num` 控制，但预留的**槽大小**取的是两方向 max，降到 1 也还是 131,072 |
| 缩小跨 cube/vector 边界的 tile | **方向错误**。真正该缩的不是 tile，而是那个被错误放大的预留；照做会损失性能去迁就一个计算错误 |

### 2.4 反面解决思路

#### 反面 A：照着错误信息做（用户实际会做的）

**它看起来合理**：错误信息是编译器给的，它应该知道。

**它错在哪**：这正是「可行动」失败的代价。用户会先降 `slot_num`（无效），再开始缩 tile（损失性能且仍然不够 25,344 字节），然后开始怀疑是自己的 kernel 设计有问题。**【推断】** 这个过程可能消耗数小时到数天，而终点是一个错误的结论——「这个 kernel 在 910B 上放不下」。

#### 反面 B：把 FIFO 预留公式和 `LOCAL_SLOT_NUM` 语义写进用户文档

**它看起来合理**：既然真因是用户不知道有两个独立的深度概念，那就讲清楚。

**它错在哪**：**讲清楚了也没用，因为用户设不了。** `pl.cross_core_slot` 只接受 `slot_num`。把一个用户无法触碰的参数解释给他听，只会把挫败感从「不知道怎么办」升级成「知道怎么办但做不到」。**【推断】** 这是「把硬件细节倒给用户」最坏的一种形态——**倒了细节，却没给对应的操作面**。

#### 反面 C：把 Vec 上限调大，或放宽这个容量检查

**它看起来合理**：明明物理上放得下，是检查太严了。

**它错在哪**：**它会把一个编译期错误换成设备上的 NaN。** 本文第 0 节已经核实，188,416 这个数（184 KB）不是保守，而是对 pto-isa#170 的补偿——UB 顶部约 8 KB 会静默损坏。调大上限，等于把第 0 节那个做对了的翻译撤销掉。

**【推断】** 这条反面思路的诱惑在于：当一个检查**因为错误的原因**挡住了合法代码时，最省事的反应是削弱检查。但这里挡错人的不是检查，是喂给检查的那个数。

#### 反面 D：让编译器自动按方向算对，但不暴露旋钮

**它看起来合理**：真因是计量错了，那就修计量——这确实能让这个 kernel 编过去，而且用户什么都不用学。

**它错在哪**：**修掉了这一次，没修掉这一类。** **【事实】** Issue 自己在「Additional Context」里点明了下一个问题：

> `pl.cross_core_slot(*, slot_num: int)` sets ring depth for both directions at once. **With asymmetric payloads the correct configuration is asymmetric** — a shallow ring for the large direction, a deeper one for the small latency-critical direction.

**【推断】** 也就是说，即便计量修对，**载荷不对称时的最优配置本身就是不对称的**，而现在的旋钮是全局的。自动修计量会让这个 kernel 通过，但把「我想给小而关键的方向更深的环」这个真实诉求继续关在门外。这是 Case 1 反面 C 的同一种错误——用自动化掩盖了能力的缺失。

**【事实】** Issue 还记录了同一形态在别处重演：#2269 是 GM 侧的对偶（`__gm_pipe_buffer` 在双向 pipe 上按一个环定尺寸，且**显式给的 `slot_num` 被忽略**），#1471 是同一个 `max` 在另一条路径上的旧案。**同一个计量错误已经出现三次。**

### 2.5 合理解决思路

**【扩展】** 以下按三条判据给出设计。Issue 给了前两步，第三、四步是本文按该思路的补充。

#### 第一步（可行动的前提）：把消失的旋钮补回来

**【事实】Issue 的三条诉求**：

1. 按**各方向自己的**消费者 tile 字节数给该方向的 `reserve_buffer` 定尺寸，不要用 `max(c2v, v2c)`。
2. 按 `effective_local_slot_num` 定尺寸，并在它与 `slot_num` 不同时把 `local_slot_num` 发到 `aic/aiv_initialize_pipe` 上。
3. **在 `pl.cross_core_slot` 上暴露 `local_slot_num`**（IR op 已经接受它）。

**【事实】** 修正后的账：C2V 的 UB 预留 = `local_slot_num × 16,384` = 32,768（深度 2），加上 82,688 字节的 tile = 115,456 / 188,416 —— 宽裕通过。

**【扩展】第 3 条还应顺带解决不对称**。既然真实诉求是分方向配置，旋钮的形态应当允许：

```python
# 保持现有写法可用（两个方向同深度）
optimizations=[pl.cross_core_slot(slot_num=2)]

# 扩展：按方向给，并区分 GM 环深度与片上本地深度
optimizations=[pl.cross_core_slot(c2v=pl.Ring(slot_num=4, local_slot_num=2),
                                  v2c=pl.Ring(slot_num=2, local_slot_num=1))]
```

**【扩展】** 两个设计约束必须同时满足：① 旧写法不能失效；② `local_slot_num` 是 **a2/a3 专有**语义（**【事实】** `system_ops.py:597-599` 的文档字符串：a2/a3 的预留是 `slot_size × local_slot_num`，**a5 上是 `slot_size × slot_num`**），因此它在 A5 上要么被忽略、要么报一条明确的「本平台不适用」，**不能静默接受又不生效**——那会制造一个新的代际陷阱（见[九域详解](PyPTO_Hardware-Native九域详解_20261019.md)第 7 节）。

#### 第二步（可理解）：让诊断指向真正的杠杆，并给出它的量

**【扩展】** 修好计量之后，容量仍可能不够——那时才需要这条诊断。它应当说出**用户能动的那个量**，而不只是总数：

```text
Vec buffer usage (213760 bytes) exceeds the safe limit (188416 bytes, 192KB physical minus
an 8KB reserved region).

Breakdown:
  cross-core pipe ring (C2V, reserved below your tiles)   131072 B
      slot size           32768 B   = max(c2v, v2c), NOT this direction's own
                                      16384 B payload (qk_raw [32,128] FP32)
      local depth              4    = the GM ring depth, reused because
                                      local_slot_num is not set
  your tiles                                               82688 B

Largest lever: the C2V local ring. At local_slot_num=2 it is 32768 B, which brings the
total to 115456 B — within budget.
  optimizations=[pl.cross_core_slot(c2v=pl.Ring(local_slot_num=2))]
```

**【扩展】这条消息相对原版改了三处，每一处对应一条判据**：

| 改动 | 对应判据 |
| ---- | -------- |
| 把总数拆成「环 + 你的 tile」，并标出环的 payload 与深度 | **可理解**——用户看得到哪一项大、为什么大 |
| 直接点名「最大的杠杆」并算出它改到某个值后的总数 | **可行动**——不是让用户试，是告诉他结果 |
| 说明 184 KB 不是物理上限而是安全上限 | **可理解**——避免用户去找「为什么不是 192」，也避免他尝试绕过 |

#### 第三步（可行动的自检）：诊断不应给出自己都无法兑现的建议

**【扩展】这是本案例最可推广的一条。** 原诊断之所以有害，不是因为措辞差，而是因为它**推荐了一个在当前代码路径下不可能生效的动作**——`slot_num` 根本不控制那个数。

因此建议引入一条诊断自检：**当诊断要推荐一个旋钮时，它应当能算出「把这个旋钮调到极限值，节省多少字节」；如果这个数是 0，或者仍然不足以解决问题，就不要把它作为建议给出，而要明说。**

```text
Note: lowering the ring depth with pl.cross_core_slot(slot_num=...) will not help here —
at slot_num=1 the reserve is still 131072 B, because each direction's buffer is currently
sized by max(c2v, v2c) rather than by its own payload. See issue #2287.
```

**【推断】** 这段话即使在修复落地**之前**就很有价值：它把用户从一条死路上拉回来，把时间还给他。**一条诚实承认自己无能为力的诊断，比一条给出无效建议的诊断好得多。**

#### 第四步（及时）：把同一个计量错误从「再犯一次」变成「不可能再犯」

**【扩展】** 同一个 `max(c2v, v2c)` 已经出现三次（#1471 → #2287 → #2269 的 GM 侧对偶）。单点修复挡不住第四次。两条结构性措施：

1. **把「分方向计量」做成一个不变量并让验证器守住它**：任何按方向成对出现的资源（C2V / V2C 的槽大小、深度、预留基址），其计量函数不得跨方向取 `max` 或 `min`。这与 Case 1 的 `HardSyncallOccupancy` 是同一类——**把一条口头规则变成一个会报错的检查**。
2. **把 PTOAS 的规格条款作为测试断言引入**：`reserve_buffer.size == slot_size × effective_local_slot_num` 是 PTOAS 文档里的一句话，今天靠人读。它应当是 PyPTO 侧的一条 lit 断言——这正是[失效类型分析](../Insight/Hardware-Native失效类型_基于1463条Issue_20261019.md)第 8 节提出的「跨仓硬件契约测试」在一个具体点上的落法。

### 2.6 这个案例的可复用模式

```text
硬件有两个独立的深度概念（GM 环深度 / 片上本地深度）
   ↓ DSL 只暴露了一个
用户碰不到真正的杠杆
   ↓ 而诊断推荐的恰好是那个碰得到但没用的
用户照做 → 无效 → 转向错误的修法（缩 tile）→ 得出错误结论（放不下）
   ↓
修法：① 补旋钮（可行动的前提）
     ② 诊断按构成拆账、点名最大杠杆、给出改后的数（可理解 + 可行动）
     ③ 诊断自检：算不出收益的建议不要给，明说无效
     ④ 把计量规则变成验证器不变量 + 跨仓契约断言（防再犯）
```

---

## 3. 两例对照

| | Case 1 · 硬屏障占用 | Case 2 · 跨核环预留 |
| --- | --- | --- |
| **用户看到的** | 一个错误码 `507018`，外加设备进入异常状态 | 一条措辞良好的编译错误 |
| **坏在哪一条判据** | **及时**——反馈晚了一整个运行周期 | **可行动**——反馈到了，但指向错误的动作 |
| **根因性质** | 检查缺位（信息齐备却不判定） | 旋钮缺位（信息齐备、杠杆不存在） |
| **缺的形态**（六形态） | **C 可校验** | **E 可表达** + **C 可校验** |
| **最具诱惑的反面思路** | 改进运行时报错（固化了错误时机） | 自动修计量但不补旋钮（掩盖能力缺失） |
| **共同的反面思路** | 写文档让用户记住硬件常数；或自动替用户改决定 | 同左 |
| **当前状态** | **已落地**，且覆盖面超出原诉求 | 计量与旋钮在本文核实的镜像上**仍未落地** |

### 三条贯穿两例的判断

**【推断】一 · 「编译器已经知道」是识别这类问题最快的探针。**
两例的判定材料都在编译期齐备：Case 1 的 `48` 在 `soc.cpp:143`，`N` 在 IR 里；Case 2 的两个方向的 payload 在 `CrossCorePipeMetadata` 里，`local_slot_num` 在 IR op 签名里。**它们不需要新信息，只需要把已有的两个数放在一起比一次。** 这与[失效类型分析](../Insight/Hardware-Native失效类型_基于1463条Issue_20261019.md)第 7.1 节的统计一致。

**【推断】二 · 「不倒给用户」是删减动作，不是克制态度。**
Case 1 的消息里没有 FFTS、没有 `SYNC_AIV_ONLY_ALL`、没有 all-arrive 语义——这些都是真的，删掉它们消息反而更有用。判断标准很具体：**一条硬件事实，如果用户不知道它也能改对，就不要出现在消息里；如果用户必须知道它才能改对，那缺的多半不是解释，而是一个旋钮**（Case 2）。

**【推断】三 · 自动化是一条有边界的路，边界在「这是不是用户的取舍」。**
两例的反面思路里各有一条「让编译器自动处理」：Case 1 自动改 `spmd(24)→(48)`，Case 2 自动修计量而不补旋钮。两者都越界了，但越的是不同的界——前者替用户改了**数据划分决定**，后者替用户关闭了**不对称配置的可能**。可用的判据是：**编译器可以自动修正「只有一个正确答案」的事（分方向计量就是其一），不能自动修正「有多个合理取舍」的事（改 block 数还是改屏障模式、两个方向各要多深）。**

---

## 4. 证据索引

### Issue

| 编号 | 要点 | 状态 |
| ---- | ---- | ---- |
| pypto#1935 | hard syncall 在部分占用下挂死 507018，无编译期检查；原文指出可静态判定 | closed（已实现验证器） |
| pypto#2287 | 跨核 FIFO 片上预留在深度与槽大小上双重过量；`local_slot_num` 未暴露 | closed（**计量与旋钮在本文核实的镜像上未落地**） |
| pypto#2269 | 上条的 GM 侧对偶：`__gm_pipe_buffer` 按一个环定尺寸，显式 `slot_num` 被忽略 | 见 #2287 引用 |
| pypto#1471 | 同一个 `GetCommonSlotSizeBytes` 的 `max` 的旧案——那次是用未切分的生产者 tile 撑大了消费者环，而非跨方向 | closed |
| pto-isa#170 | Vec UB 实际可用约 184 KB 而非 192 KB，超出则静默 NaN | **open** |
| pypto#1173 | 硬编码 `block_dim=24` 在可用核更少的设备上挂死 | closed |

### 仓内源码（`repo/pypto`，本文逐条核实）

- `src/backend/common/soc.cpp:119-156`（910B SoC）/ `:161-200`（950 SoC）：`Die die({{aic_cluster, 24}, {aiv_cluster, 48}})`；Vec 安全上界 184 KB / 240 KB 及其 `TODO(pto-isa#170)` 注释；两代的 memory graph。
- `src/ir/verifier/verify_hard_syncall_occupancy.cpp`：文件头 12-67 行的设计说明；`OccupancyMessage`(212)、`SyncStartMessage`(225)、`WrongQueryMessage`(235)、`OccupancyGateMessage`(249)、`UnsatisfiableMessage`(261)；四种情形的分发在 415-475 行。
- `src/ir/verifier/property_verifier_registry.cpp:118-122`：`HardSyncallOccupancyValid` 的注册。
- `python/pypto/language/op/system_ops.py:150-180`：`syncall` 的 docstring，已写明编译期拒绝部分占用并引用 #1935。
- `tests/st/runtime/cross_core/test_syncall.py`：模块 docstring，满占用 + `sync_start` 两条要求。
- `src/ir/transforms/utils/cross_core_pipe.cpp:190-204`（`GetCommonSlotSizeBytes` 的 `max`，**当前仍是**）、`:344-357`（深度取 override 或 `kDefaultAutoPipeSlotNum`）。
- `include/pypto/ir/transforms/utils/cross_core_pipe.h:66`：`kDefaultAutoPipeSlotNum = 2`。
- `python/pypto/language/optimizations.py:136`：`def cross_core_slot(*, slot_num: int)` —— 无 `local_slot_num`。
- `python/pypto/ir/op/system_ops.py:558,570-571,584,597-599,616,629-631`：IR op 接受 `local_slot_num`，并记载其 a2/a3 专有语义。
- `src/backend/common/pto_ops_crosscore.cpp:241-245`：codegen 能发出 `local_slot_num`。
- `src/ir/transforms/allocate_memory_addr_pass.cpp:124-139`（`ReservedBytesNote`，即 Case 2 那条建议文本的来源）、`:480-486`（容量 CHECK）。

> 本文只做案例梳理与设计建议，不含代码或产品实现。【事实】均可按上列位置核对；【扩展】为按指导思路所作的补充设计，未在仓库中实现。`repo/pypto` 为镜像，更新后应重新核对再引用。
