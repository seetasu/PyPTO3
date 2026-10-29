# decode_csa 反向推导 · PyPTO 算子生成

拿一份已经跑在卡上的算子，把它钉死的每个数字推回设计空间：当初还有什么选法，为什么没选。

对象是 [`Data/DeepseekV4/deepseek_v4_flash_dspark/decode_csa.py`](../../Data/DeepseekV4/deepseek_v4_flash_dspark/decode_csa.py) —— DeepSeek-V4 的 CSA decode 层，一个 TP rank 的完整编排。

入口：[`index.html`](index.html)。本地静态服务打开，或从仓库根的 `launch.html` 进入。

## 为什么是反向

正向生成要先假设一个搜索过程，候选数量、耗时、余量都只能估。反向推导不需要：成品里每个常量都是一次选择留下的痕迹，而 PyPTO 这套代码把理由大多写在了旁边——

```python
NUM_QK_CORES = 24        # qk_pv dispatch lanes = a2a3 AIC count; re-sweep for other AIC counts
O_RS_REDUCE_WORKERS = 48 # 128 row blocks; 8 left 40 of 48 AIV idle
O_RS_PUBLISH_WORKERS = 24 # put is fabric-bound; more workers only burn cores
PROJ_B_D_TILE = 512      # proj_b_mm D chunk per task; coarser starves the 24 AIC cores
```

每一条注释都记录了一个被否掉的取值和它的后果。再加上 11 条 `raise ValueError` 整除契约、两处「inline 返回句柄跨不过调用边界」的防回归注释、一条编译器缺陷的规避说明，被拒绝的选项基本不用编。

## 这一层在做什么

TP = 2、FLASH 配置下：D = 4096，H = 64，HEAD_DIM = 512，每卡 B = 32 个请求 × S = 8 个 token（1 + 7 个 DSpark 投机 token）= T = 256，all-gather 之后的全组 token 流 KV_T = 512。

子内核链：

```
hc_pre → csa_rope_interleave → rms_norm
       → decode_cp_token_allgather_step            ← 跨卡 ①
       → q_proj_rope / kv_proj_rope → csa_cache_writeback
       → compressor_ratio4 → indexer_compressor → indexer
       → sparse_attn_csa → csa_merge_pack_publish  ← 跨卡 ②
       → o_group_a2a → o_proj_reduce_scatter       ← 跨卡 ③
       → hc_post
```

`@pl.jit.inline(auto_scope=False)` 意味着这十几段之间的依赖边全由作者用 `pl.at` / `pl.spmd` 返回的 task id 和 `pl.system.task_dummy` 显式串出来，编译器不替你推。

## 页面结构

| 区域 | 内容 |
| --- | --- |
| 左栏 Kernel Intent | 形状派生、子内核链、PyPTO 构件（点击看作用）、910B 硬件边界、6 块跨卡窗口 |
| 候选谱系 | TILE SPACE → WORK PLAN → CONTRACT LEGAL → BUFFER FIT → COMM PLAN → PTO EMITTED → SHORTLIST |
| 决策点矩阵 | 7 个被钉死的常量 × 它们的候选取值，占用现算并与真实容量比较，★ 标出源码取值 |
| 拒绝浏览器 | 四次收敛各自的分类与案例，每条都带源码原文 |
| 候选对比 | TP = 1 / 2 / 4 三个世界，16 行决策因子按 config.py 现算 |
| 工程视图 | decode_csa.py 的编排、publish 段、常量表、host 层，以及推导出的 .pto 与 perf_hints |

## 哪些是真的，哪些是构造的

**源码原文，可逐条核对：**

| 事实 | 出处 |
| --- | --- |
| `_TP_CHOICES = (1, 2, 4)`、`_TP_DEFAULT = 2`、TP 形状 import 时冻结 | `decode_csa.py:20-36` |
| 两处「a returned inline handle cannot cross into …」 | `decode_csa.py:282-284, 487-488` |
| 压缩器要 scalar-extent 视图，否则符号不可证相等 | `decode_csa.py:335-337` |
| indexer cache 是 per-request 状态，两半跑不同 token 流 | `decode_csa.py:363-364` |
| `if T != LOCAL_T: raise ValueError(...)` | `decode_csa.py:145-148` |
| `H_TILE = 16`、`QK_M_TILE = 32`、`ATTN_K_TILE = 128`、`NUM_QK_CORES = 24` 及其注释 | `decode_sparse_attn_csa.py:63-75` |
| 单块 sparse-K 误编译、floor 到 2 后逐位相同 | `decode_sparse_attn_csa.py:77-79` |
| `BIAS_T_TILE = min(T, 8)` 与「whole-T form … past the Vec limit」 | `decode_sparse_attn_csa.py:89-91` |
| 五条整除契约 | `decode_sparse_attn_csa.py:92-101` |
| worker 数的四条调优注释、「Keep the INT32 proj-b accumulator within the A2/A3 tile buffer」 | `decode_o_proj.py:63-103` |
| 另外六条整除契约 | `decode_o_proj.py:110-132` |
| 模型与部署常量（hidden 4096 / heads 64 / head_dim 512 / topk 512 / batch 64 / spec 7） | `config.py` 的 `FLASH` |
| 片上容量 Mat 512 KB · L0A/L0B 64 KB · L0C 128 KB | `repo/pypto/include/pypto/backend/910B/backend_910b_handler.h` |
| Vec 184 KB 可用（物理 192 KB，PTO-ISA 保留约 8 KB，pto-isa#170） | `repo/pypto/src/backend/common/soc.cpp` |

**现场算出来的：** 决策点矩阵里每个取值的占用（如 `PROJ_B_MM_T_TILE × N_TILE × 4 B`、`BIAS_T_TILE × IDX_TOPK × 4 B`），以及候选对比表里 TP = 1 / 2 / 4 的 B、T、T_PAD、LOCAL_O_GROUPS、窗口行数、QK 工作项、每核轮次、publish 波次——全部由 [`src/data.js`](src/data.js) 按 config.py 的常量实算。两个结果值得留意：`128 × 256 × 4 B = 128 KB` 正好等于 L0C 容量；`256 × 512 × 4 B = 512 KB` 正好是注释里说的「well past the Vec limit」。

**为把收敛过程画出来而构造的：** 谱系各阶段的候选数量（144 / 288 / 120 / 48 / 96 / 12 / 3）和每个拒绝分类的条数。这些是量级示意，不是真实的编译统计。被拒案例本身是真的，条数不是。页脚写明了这一点。

## 来源与依赖

页面结构与视觉语言改造自 [yinyucheng0601/compute-graph-viewer](https://github.com/yinyucheng0601/compute-graph-viewer) 的 `operator-planning/recommendation-trace-V3.html`（main 分支，2026-10-22）。完整本地镜像见 [`repo/yinyucheng0601.github.io/`](../../repo/yinyucheng0601.github.io/DOWNLOAD.md)。

- [`src/base.css`](src/base.css) —— 从 V3 抽出本 Demo 用到的样式规则，基于 PTO Design System 语义 token。
- [`components/candidate-population-lineage/`](components/candidate-population-lineage) —— 上游抽出的共享 SVG 组件，原样复制。本 Demo 只通过 `render()` 传数据、用 `onSelect` 在组件外渲染详情。
- PTO Design System 走仓库自己的 `vendor/pto-design-system/`。

## 文件

```
index.html           页面骨架 + Demo 自有样式
src/base.css         从 V3 抽出的样式基线
src/data.js          全部内容数据（谱系、详情、拒绝案例、决策矩阵、候选、源码节选）
src/app.js           交互
components/          上游 candidate-population-lineage 组件
```

换算子只需改 `src/data.js`。谱系的数量守恒由组件校验：某一步的拒绝分类条数之和必须正好等于前后两阶段的差，改数字时会直接抛错。
