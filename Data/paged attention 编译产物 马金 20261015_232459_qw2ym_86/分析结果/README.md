# paged attention 编译产物 · 分析结果

分析对象：`../`（`paged attention 编译产物 马金 20261015_232459_qw2ym_86`）
模型 / 算子：Qwen3-14B paged attention 融合算子（Q/K RMSNorm + RoPE + KV cache 写入 + 分页注意力）
编译形态：PyPTO JIT，目标 `a2a3 / 910B`，24 个 block → 24 AIC + 48 AIV SPMD kernel

目录内含**两次采集**：

| 采集 | 目录 | 内容 |
|---|---|---|
| 20261015（冷启动） | `dfx_outputs/` | trace / swimlane / deps / passes_dump / kernels |
| 20261016（第一次带核内数据） | `qwen3_14b_paged_attention_profile_20261016/` | 上述全部 + **PMU 计数器** + **指令级仿真** + runtime head/tail 日志 |

---

## 1. 交付物

| 文件 | 是什么 | 怎么用 |
|---|---|---|
| `dfx_outputs_chrome_tracing.html` | **交互式 trace 查看器**（单文件，数据全内联，离线可开，约 323 KB） | 用浏览器打开。6 个 tab：`时间线` / `核负载` / `核内` / `PMU 与仿真` / `任务与依赖` / `输入定义` |
| `paged_attention_编译产物分析.html` | **静态分析报告**：编译链路、pass 管线、片上内存、编排任务图、DFX 时间线、优化建议 | 用浏览器打开 |
| `scripts/_extract_incore.py` | 核内结构抽取器（PTO IR / kernel C++ / ELF / DFX 四源合一） | `python3 _extract_incore.py [SRC] [--out FILE]` |
| `scripts/_extract_profile.py` | 20261016 profiling bundle 抽取器（PMU / trace / 仿真 / runtime 日志，按可信度分层） | `python3 _extract_profile.py [BUNDLE] [--out FILE]` |
| `scripts/_extract_inputs.py` | 输入定义抽取器（compiled_meta / deps / 前端签名 / debug run.py / run README 五源合一） | `python3 _extract_inputs.py "<产物根目录>"` |
| `scripts/_gen_trace_viewer.py` | 查看器生成器（读 `dfx_outputs/*.json` + profiling bundle → 产出自包含 HTML） | `python3 _gen_trace_viewer.py` |

---

## 2. 查看器怎么读

### tab ① 时间线（Chrome Tracing）
151 条 lane / 4 个进程 / 836 条事件 / 238 条 flow 箭头，chrome://tracing 风格。

- `Ctrl + 滚轮` 缩放，拖拽平移，`Cmd + F` 适配宽度
- 单击 slice 出详情（ts / dur / cat / lane + 该事件关联的 flow 列表），选中后其他 flow 自动淡出
- 238 条箭头按类型着色：`dispatch` 144（蓝）、`submit→dispatch` 72（绿）、`dependency` 18（黄）、`complete` 4（红）
- 右下角可下载原始 trace JSON

### tab ② 核负载
72 核窗口甘特 + 每核 `dur` 排行 + `scheduler_records` + orchestrator 10 次 submit 明细。

### tab ③ 核内
DFX 里**没有**核内数据（72 个核 lane 每核只有 1 条 slice），这一页全部从静态产物重建：

1. 核内信息分布（哪一层有什么）
2. AIC vs AIV 程序对照
3. **片上缓冲：硬件通路 × 容量占用** ← 数据通路图（**附「这张图怎么读」三步说明 + 5 块缓冲的类比对照表**）
4. **单个 tile 尺寸谱：AIC 少而大 · AIV 多而碎** ← 方块阵列图
5. 核内流水 set_flag / wait_flag 配对图
6. 核内二进制段实测
7. 还缺什么 / 怎么补

### tab ④ PMU 与仿真 ★（20261016 新增）
**四份数据、三种可信度**必须先分清再读：

| 数据 | 性质 | 能回答 | 不能回答 |
|---|---|---|---|
| `pmu.csv` | 硬件实测 | 72 核各自 cycles + 8 个带宽计数 | 无 pipe 占用率、无 stall 原因 |
| `merged_swimlane` | 硬件实测 | dispatch→finish 时间轴 | 核内无子事件 |
| `simulator/` | **仿真模型** | pipe 级 / 指令级 cycles 分布 | 输入与硬件不同（单核、合成、`seq_lens=[128,129]`） |
| `costmodel_attn_swpipe` | **无结果** | — | PTO ISA 头文件编不过，只有报错日志 |

内容：
1. 四份数据可信度表
2. **72 核 PMU 条带图**（上 cycles / 下 Σ带宽，对数轴）+ 三分结构表
3. **核内仿真 pipe 堆叠条** + AIC/AIV 单指令 top8 占比
4. **两次运行对比甘特** + 指标表
5. 对之前结论的修正表

### tab ⑤ 任务与依赖
`deps.json` 8 个任务表 + 真实 kernel 的 19 个 args 明细 + `name_map` + 事件名时长排行。

### tab ⑥ 输入定义 ★
shape 从哪来、值有没有落盘，一页讲清：

1. **这次跑的是什么** —— 由 shape 反推：batch 2、每序列 1 token（decode）、40 head × 128、8 kv_head（GQA 组 5）、24 AIC + 48 AIV、16 任务、页 128 / 分栈 512
2. **对外 API 参数 15 个** —— shape / dtype / 方向 / 说明
3. **kernel 独有实参 5 个** —— `q_tnd_flat`（摊平视图）+ 4 个 workspace，附 15−1+1+4=19 的分解
4. **派生量 10 条** —— 源码表达式原样摘录，能与分块数对上
5. **运行期 workload** —— 20261016 的完整命令行；20261015 明确标注「未记录」
6. **哪些数字由输入决定** —— tile 谱 / flag 配对换输入不变，cycles / 利用率会变

---

## 3. 核心结论速览

### 3.1 时间线（20261015，端到端 177.2 µs，核利用率 29.9%）

| 阶段 | 起止 (µs) | 说明 |
|---|---|---|
| Orchestrator 提交 | 0 → 46.0 | 10 次 submit |
| 空档 | 46.0 → 79.2 | 无 trace 覆盖，原因未定 |
| dummy ×7 | 79.2 → 84.0 | 7 个纯依赖空壳 |
| Scheduler drain | 98.3 → 121.4 | 核开跑之前 |
| **核执行** | **119.46 → 172.44** | **52.98 µs**（dispatch→finish 窗口，AIC 均值 35.2 / AIV 36.1） |

**编排**：10 个提交里只有 1 个真计算（#9，24 AIC + 48 AIV），#2–#8 是 7 个 `phase_fence_barrier_0..6` 空壳，白占 7 次调度往返。

### 3.2 片上内存（由 `pto.alloc_tile` 逐条复算）

| 核 | 空间 | 累计峰值 / 容量 | 占用 | 单块最大 | 复用槽位 |
|---|---|---|---|---|---|
| AIC | L1 `mat` | 204 KB / 512 KB | 39.8% | 32 KB | 6.38 |
| AIC | L0A `left` | 8 KB / 64 KB | 12.5% | 4 KB | **2.00**（双缓冲） |
| AIC | L0B `right` | 64 KB / 64 KB | **100%** | 32 KB | **2.00**（双缓冲） |
| AIC | L0C `acc` | 8 KB / 128 KB | 6.25% | 8 KB | 1.00 |
| AIV | UB `vec` | 94.1 KB / 192 KB | 49.0% | 16 KB | 5.88 |

→ **L0B 是硬资源边界**（打满），L0C 只用 6.25%，负载极不均衡。

**tile 碎片化**：AIV 20 种形状里 **7 种 ≤ 512 B，却占 142/212 = 67% 的分配次数**（`[1,8]` / `[8,1]` / `[1,16]` 这些是 RMSNorm / RoPE 的位置参数）。全局最大 tile 32 KB 与最小 32 B 相差 **1024×**。

### 3.3 PMU：三分之一的 block 在空转（20261016，硬件实测）

| 调度线程 | block | AIC cycles 均值 | AIC `main_mem_write_bw` | AIV cycles 均值 | AIV `ub_read_bw_vector` |
|---|---|---|---|---|---|
| t0 | 8 AIC + 16 AIV | 47028.6 | 29 | 45835.5 | 932.6 |
| **t1** | 8 AIC + 16 AIV | **26227.1** | **2** | **28905.1** | **0** |
| t2 | 8 AIC + 16 AIV | 50726.1 | 41 | 51069.6 | 509.0 |

**两条独立证据链**：

1. PMU —— t1 的 cycles 只有其他组的 0.52–0.57×，带宽计数为 0（AIV）/ 2（AIC 写）。若计数器本身没工作，`main_mem_write_bw` 也该是 0，而它是 1–2 → **计数器在工作**。
2. trace —— t1 的核 slice 明显更短（core 1 = 19.68 µs vs core 0 = 41.36 µs，比值 0.48，与 cycles 的 0.48 一致）。

→ 中间 8 个 block 拿到了几乎为空的 KV 分片，**仍在花 26–29k cycles 等 flag**。

### 3.4 核内仿真：瓶颈不是 matmul（模型值，只看比例）

| 核 | 最大项 | 次高 | 关键数字 |
|---|---|---|---|
| AIC (cube) | `SCALAR` **37.7%** | `MTE2` 19.3% | **`CUBE` 只有 7.5%**（3374 cyc） |
| AIV subcore 0 | `VECTOR` 51.5% | `SCALAR` 29.1% | **`BAR` 单项 39.3%**（37955 cyc / 93 次，均 408 cyc） |
| AIV subcore 1 | `VECTOR` 51.9% | `SCALAR` 28.8% | 同上 |

- AIC 上 `SCALAR` 37.7% + `WAIT_FLAG` 26.5% + `LD/ST_XD_XN_IMM` 27.8%（标量地址簿记）→ **矩阵乘引擎大部分时间在等**
- AIV 上 `BAR`（barrier）一项就比任何单个向量算子都重 → **同步开销压过计算**

⚠️ 仿真用 `seq_lens=[128,129]`（硬件 512/513）、单核回放、输入由 `golden.py` 生成。AIC 仿真 44.8k cyc 与硬件 PMU 均值 41.3k 接近**只是巧合**，不是校准。

### 3.5 两次运行对比：同一个二进制，慢了一倍

`.text` 段 4664 / 11884 B 完全一致，`compiled_meta.json` 与 `kernel_config.py` 一致，deps 同为 8 task →
差别只在输入（`--batch 2 --seq-lens 512,513 --page-mapping reverse`）。

| 指标 | 20261015 | 20261016 | 变化 |
|---|---|---|---|
| 总窗口 (µs) | 177.20 | 345.90 | **+95%** |
| 核窗口 (µs) | 52.98 | 79.60 | **+50%** |
| 首核开跑 (µs) | 119.46 | 259.00 | **+117%** |
| 核 slice 均值 (µs) | 41.26 | 53.60 | — |
| 核 slice 最大 (µs) | 52.8 | 79.6 | — |

**runtime 日志给出 head/tail 拆分**（以前只能从 trace 的 slice 跨度猜）：

| 函数 | 核数 | exec (µs) | latency (µs) | Head OH | **Tail OH** | Exec/Latency |
|---|---|---|---|---|---|---|
| `attn_swpipe_aic` | 24 | 22.33 | 51.41 | 5.28 | **23.81** | 43.4% |
| `attn_swpipe_aiv` | 48 | 22.66 | 54.70 | 6.27 | **25.76** | 41.4% |

→ **Tail OH 比 exec 还大，收尾开销约等于整个计算时长。** 汇总 `Exec/Latency = 42.07%`，与「每核 exec 22.55 / 窗口 79.6 = 28.3%」是两件事，别混。

### 3.6 对早期结论的修正

| 之前的说法 | 现在 |
|---|---|
| 「核内时序采不到，需要另采一次 profiling」 | ✅ 已补：PMU 72 核 + 指令级仿真；仍缺 **pipe 占用率** 与 **stall 归因** |
| 「24 block 只摊了 16 个 task，8 个 block 空转」 | ✅ 已证实：8 个 block 带宽计数为 0，cycles 只有 0.52–0.57× |
| 「trace 的核执行窗口 52.98 / 79.6 µs」 | ⚠️ 修正：那是 **dispatch→finish 的 latency 窗口**，不是核上执行时间；真 exec 只有 22.3 / 22.7 µs/核 |
| 「本次产物没有正确性校验」 | ✅ 已补：`key_cache` / `value_cache` / `out` 全部 PASS（`out` 还做了数值比较） |
| 「`chip_swimlane` 时基与 trace 差 1.55×」 | ⚠️ 升级为确定问题：逐核拟合 ns/unit 在 **28.6–67.9** 之间飘，非常数 → 该字段**不是线性时间，不能用** |

### 3.7 仍然缺的

- **冷启动 vs 热态**：20261016 不是热态重采（同二进制 + 不同输入 + 首次 PMU/仿真），这条缺口**仍未补上**，建议补一次 `--no-recompile` 同输入连跑两次
- **pipe 占用率 / stall 原因**：PMU group 7 只给带宽计数，没有 `PIPE_RATIO` 一类计数器
- **核内源码级归因**：`.o` 是 DaVinci ISA（`e_machine 0x1029`），本机 `objdump -d` 为空；DWARF `.debug_line` 完整，有该 ISA 的反汇编器即可把 `.text` 逐行映射回源码
- **20261015 的输入值没有落盘**：`seq_lens` / `block_table` / `slot_mapping` 的实际取值只存在于当次进程内存里（20261016 例外，命令行记在 profile README）。`debug/run.py` 里的 `randint(0,8,2)` 只是重放默认值。要补需带 `--dump-args 2` 重跑
- **PMU 是线程口径，不是逐核口径**：`pmu.csv` 的 `thread_id` 按 `core_id % 3` 分配，且同一线程内 8 个 AIC 核的 `main_mem_write_bw` 完全相同（29 / 2 / 41）→ 带宽类计数器不能当逐核事实读，「8 个 block 空转」的 PMU 证据不成立（结构上 `num_tasks=16 < 24` 仍成立）

---

## 4. 重新生成

```bash
cd scripts
python3 _gen_trace_viewer.py            # → 重新产出查看器 HTML（需 data/ 下存在两次采集目录）
python3 _extract_incore.py "<产物根目录>" --out incore.json
python3 _extract_profile.py "<profile bundle 目录>" --out profile.json
python3 _extract_inputs.py "<产物根目录>" > inputs.json
```

`_gen_trace_viewer.py` 顶部的 `SRC` 常量指向 20261015 的 `dfx_outputs`；`PROFILE` 与 `INPUTS` 由同级的
`qwen3_14b_paged_attention_profile_20261016/` 和根目录下的 `compiled_meta.json` 自动定位。换成别的产物目录即可复用。

**换产物时唯一要改的**是 `SRC`；三份抽取都是顺着 `compiled_meta.json` / `deps.json` / `passes_dump/` /
`debug/run.py` 自己找，不依赖写死的 shape。
