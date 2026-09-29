# Pass Transform Explorer

把 PyPTO 编译器从「输入算子、输出二进制的黑盒」变成一条可以逐站检查的流水线。

工具直接解析 `Data/**/passes_dump/*.py`——每个 Pass 执行后的真实 IR 快照。

左栏顶部切换两条**正交主线**，它们看的是同一批快照，只是主角不同：

### 按 Pass（横向）——「这个 Pass 做了什么」

| 问题 | 视图 |
| --- | --- |
| 变化发生在哪个 Pass？ | **Pass 时间线**：按实际改动行数排序，空操作 Pass 直接置灰 |
| 具体改了哪几行？ | **代码 Diff**：按函数拆分，token 级高亮，并排 / 统一双模式 |
| 这个 Pass 到底优化了什么？ | **变化概览**：从 AST 实测出的结构增量 + **结构图**：五种视角的前后对照 |

### 按 Callable（纵向）——「这个算子经历了什么」

选定一个 callable，看它穿过全部 Pass 的一生。索引是 `pass.functions` / `pass.changedFunctions`
的转置，不引入任何新事实；左右两区直接复用 Pass 主线的结构图与 Diff，只是锁定到这个 callable。

页面分三区，点历程条上任意一步，左右两区同时跟随：

| 区域 | 内容 |
| --- | --- |
| **顶部** | **成果条** + **历程条**：只列真正改动过它的 Pass，每步一句话说清做了什么 |
| **左下** | **这个 Pass 做了什么**：该 Pass 的推荐视角，锁定在这个 callable 上 |
| **右下** | **代码怎么变的**：该 Pass 对这个 callable 的 token 级 Diff |

**行数不是语义。** 一个 Pass 让函数长了 50 行，这只说明体积，不说明它做了什么优化。
历程条的主徽章因此用索引里现成的**结构计数器**——tile 分配、循环、任务——它们才说明
Pass 做了什么：`AutoTileMatmulL0` 加循环是因为它把 matmul 分块，`MemoryReuse` 减分配是因为
它让缓冲区共享存储。语句数保留但降级为次徽章，并在结构计数器已经解释了这一步时直接隐去。
两个计数器都没动、但 IR 确实被改写的 Pass（填地址、改名、重排）标成「就地改写 N 行」，
具体改了什么交给下面两区。

以 `#l3_decode_csa/c/qk_pv/35` 为例，16 个 Pass 改动过它：

```text
09 OutlineIncoreScopes     循环 +5           外提为独立函数
17 AutoTileMatmulL0        循环 +2           matmul 分块
23 ExpandMixedKernel       循环 +2           InCore → AIC+AIV+Group
25 SplitVectorKernel       循环 +3
28 SkewCrossCorePipeline   循环 +2           跨核流水错位
33 InitMemRef              tile 分配 +53     建 MemRef
35 MemoryReuse             tile 分配 −41     缓冲复用
36 AllocateMemoryAddr      就地改写 190 行   只填地址，不改容量
46 Simplify                循环 −1
```

选中 `MemoryReuse` 时，左下的内存视图给出这一步的实际收益：**合计 274 KiB → 49 KiB，
缓冲 30 → 5 个**。顶部成果条给出全程：**tile 分配峰值 53、终值 12，最终省下 41**。

### Pass 专属图

通用视角只能说「IR 变了」。读者想知道的是**这一类 Pass 在做什么**——外提是把一段区域抬成函数、
必须定下边界；下降是把 tensor 语义里隐含的搬运写成显式指令。这是两张完全不同的图，所以按 Pass
类型各画各的，排在视角条最左、默认推荐。

| Pass | 专属图 | 画什么 |
| --- | --- | --- |
| `Outline*Scopes` | **外提** | 宿主区域 → 独立函数 + 一次调用；逐个列出跨边界的值 |
| `ConvertTensorToTileOps` | **语义下降** | 两条算子流的序列比对，换域 / 新增 / 退场分色 |

**外提图**的重点是边界。它从 IR 反查宿主函数与调用行号，从函数签名取参数，从函数体末尾的
`return` 取返回值，两者**求交集**就得到跨循环携带的累加状态——这类值既是参数又是返回值，
外提必须认出来，否则语义就断了。`qk_pv` 的三个是 online softmax 的 l / m / o：

```text
qk_pv                     14 参数 · 3 返回 · 3 个循环携带
                          sparse_blk_li / mi / oi —— 进去再出来
hc_post                    6 参数 · 0 返回        —— 纯写出型，结果走 Out 参数
indexer_score_leaf_wave    9 参数 · 1 返回 · 1 个循环携带
```

**语义下降图**把两条算子流做序列比对。关键是先**归一化算子名**（`pl.tensor.exp` 与
`pl.tile.exp` 都归到 `exp`），这样同一个算子换域会对齐成一列，而不是被当成无关的一删一增。
对齐之后，断口处就是新插入的显式搬运：

```text
qk_pv:  25 个换域 · +19 新增 · −12 退场
        新出现的显式搬运：store ×9、create ×3、load ×2、reshape ×2、transpose_view ×1
```

判断「新出现」也要用归一化名——否则 `tile.gather_row` 会被误报成新增搬运，而它在 tensor 域
本来就有，只是换了域。

没有专属图的 Pass 不显示这个视角，落到下面两个差分视角。

### 两个差分视角

原有的五种结构视角回答「这份 IR 长什么样」，其中两种是 **program 作用域**——在 callable 视图里，
不管选哪个 callable 都渲染出字节级相同的一张全程序图。所以这里补了两个**差分且函数作用域**的视角，
它们回答的是「这一步对**这个** callable 做了什么」：

| 视角 | 回答 | 适合 |
| --- | --- | --- |
| **算子迁移** | 退掉了哪些算子、换上了哪些，同名的左右配对 | 下降类（`Convert*` / `Lower*` / `Legalize*`） |
| **缓冲生命期** | 每个缓冲的存活区间画成一条横条，横轴是语句序号 | 内存类（`InitMemRef` / `MemoryReuse` / `AllocateMemoryAddr`） |

**算子迁移**把直方图增量变成一句话。`ConvertTensorToTileOps` 对 `qk_pv` 不是「多了 6 行」，而是：

```text
pl.tensor.gather_row  9  ⟶  pl.tile.gather_row  9
pl.tensor.slice       8  ⟶  pl.tile.slice       6
pl.tensor.matmul      2  ⟶  pl.tile.matmul      2
pl.tensor.exp / row_max / row_sum / cast …  ⟶  tile 同名
未配对新增：pl.tile.store 0→9、pl.tile.load 0→2   （tensor 域里隐含的搬运，现在显式了）
未配对移除：pl.tensor.assemble 7→0
```

**缓冲生命期**把「复用」画成字面意思。复用不是「缓冲变少了」，而是**一个缓冲活得更久、替多个短命
缓冲干活**。所以每个缓冲画一条横条，`MemoryReuse` 之后许多短条并成少数长条，被拉长的那几条高亮：

```text
qk_pv          40 → 15 个缓冲   274 KiB →  49 KiB    4 条被拉长
hc_post       156 → 12 个缓冲  2.32 MiB →  96 KiB
kv_proj_matmul 23 → 13 个缓冲   440 KiB → 232 KiB
```

推荐视角按 Pass 自动选：有专属图的 → 专属图；内存阶段 → 缓冲生命期；原本推荐调用图或数据流的 → 算子迁移；
控制流与任务 DAG 保持不变。**如果自动选中的视角在这一步没有数据**（`InitMemRef` 之前没有缓冲、
计算 kernel 里没有 task），会自动退回算子迁移——它要么有迁移可看，要么能明确说出「算子构成没有变化，
这一步改的是语句内部」。手动切换过视角后不再自动退回。

`ExpandMixedKernel` 的家族识别不靠名字前缀：只有当某个函数在**同一个 Pass** 里 kind 翻成
`Group`、且同时新增了 `<name>_aic` / `_aiv` 时才认作拆核。纯前缀匹配会把 `foo` 和 `foo_0`
这种编译器加后缀区分出的**不同** callable 误当成父子（两份 run 实测误伤 0 例）。

设计依据是仓库里已有的结论 [`Design/llvm-flow/Pass_Diff_对话记录_20260908.md`](../llvm-flow/Pass_Diff_对话记录_20260908.md)：
文本 IR Diff 是事实来源，Graph 负责快速理解结构，两者同步；并且**不存在一种计算图能解释所有 Pass**，
所以每个 Pass 自带「推荐视角」。

## 运行

数据是生成的（约 50 MB，已 gitignore），第一次使用需要先构建：

```bash
node Design/pass-transform-explorer/build.mjs
```

然后用任意静态服务器打开 `Design/pass-transform-explorer/index.html`，或从 `launch.html` 进入。
（页面通过 `<script>` 标签按需加载快照，`file://` 下通常也能直接打开，但静态服务器更可靠。）

```bash
npx http-server . -p 4178 -c-1
```

构建约 6 秒，产出：

| 产物 | 大小 | 内容 |
| --- | --- | --- |
| `data/index.js` | ~1.5 MB | Pass 时间线、实测增量、证据卡片 |
| `data/docs.js` | ~210 KB | 从 `repo/pto/docs/zh-cn/dev/passes/` 提取的 Pass 说明 |
| `data/<run>/NN.js` | ~50 MB | 每份 IR 快照的原文，按需加载 |
| `lib/bundle.js` | ~75 KB | 解析器 / 分析器的浏览器版 |

`build.mjs --no-src` 只重建索引，跳过快照（改分析逻辑时用它更快）。

## 独立 demo（单文件，无需构建和服务器）

要把工具发给别人、或在没有这个仓库的机器上打开：

```bash
node Design/pass-transform-explorer/build-demo.mjs
```

产出 `demo.html`——**一个文件，双击即开**。样式、解析器、应用代码、Pass 索引、Pass 文档和
全部 94 份 IR 快照都 gzip + base64 内嵌在里面：

| 命令 | 产出 | 大小 | 内容 |
| --- | --- | --- | --- |
| `node build-demo.mjs` | `demo.html` | 6.0 MB | 两份 run，94 份快照（46.96 MB IR） |
| `node build-demo.mjs --runs decode_fwd_layers --out demo-lite.html` | 自定义 | 1.7 MB | 单份 run，42 份快照 |

快照是**逐份压缩、按需解压**的，所以打开时只付索引的代价（约 110 KB），
不是 47 MB 的 IR。切到某个 Pass 的 Diff 或结构图时才解压对应的两份快照（每份约 10 ms）。

需要浏览器支持 `DecompressionStream`：Chrome / Edge 80+、Firefox 113+、Safari 16.4+。
不支持时会显示明确的提示而不是白屏。

demo 里没有仓库，所以 Pass 源码路径和文档来源会显示成带 tooltip 的纯文本而不是死链接；
其余功能与完整版完全一致（同一份 `app.js` 和 `lib/bundle.js`）。

## 为什么 Diff 和图在浏览器里现算

`lib/*.mjs` 既是构建期代码，也被打包进 `lib/bundle.js` 供页面使用——**同一份解析器**。
页面加载一份快照原文后自己解析、自己 diff、自己建图，因此任意函数、任意视角、任意 Pass
都能全保真查看，而不需要把几十万行预计算结果冻进数据包。解析一份 5000 行快照约 50ms，最近 8 份缓存在内存里。

## 五种结构图

节点按 union 布局**一次性排好**，再按状态着色（新增 / 删除 / 属性改变 / 未变）。
分别给前后两版单独布局会让插入一个节点就把整张图挪位，真正的变化反而被淹没。

| 视角 | 节点 | 最适合看 |
| --- | --- | --- |
| 调用 / 作用域 | 函数 | `InlineFunctions`、`Outline*`、`ExpandMixedKernel` |
| 控制流 | for / if / with 区域 | `UnrollLoops`、`LowerPipelineLoops`、`SkewCrossCorePipeline` |
| 数据流 | SSA def-use | `ConvertTensorToTileOps`、`LowerCompositeOps`、`ResolveBackendOpLayouts` |
| 任务 DAG | `pl.submit` / `pl.at` 及依赖边 | `AutoDeriveTaskDependencies`、通信相关 Pass |
| 内存布局 | 缓冲区、空间、地址 | `InitMemRef`、`MemoryReuse`、`AllocateMemoryAddr` |

任务依赖是抽象解释 `pl.array.create` / `update_element` 链恢复出来的——依赖在 IR 里是通过
`_submit_deps_buf` 数组传递的，不解释这些语句就拿不到真实的 task DAG。

## 「高频改写」如何工作

有些 Pass 不改任何结构指标，只是就地改写每条语句——`AllocateMemoryAddr` 填 MemRef 偏移、
`ConvertToSSA` 追加版本后缀、`CanonicalizeIOOrder` 把 `pipeline` 降成 `range`。
这类 Pass 如果只报「735 行变化」等于什么都没说。

所以构建期会把被改写的行按 token 重叠度**配对**（相似度低于 0.5 的算纯增删，不硬凑），
再对每一对做 token 级 diff，聚合出最高频的替换。例如：

```
AllocateMemoryAddr   735 行就地改写，最高频 `0` → `1024`（119 处）
DeriveCallDirections  42 行就地改写，最高频 插入 `, attrs={"arg_directions": [...]}`
MemoryReuse          tile.alloc 668 → 211，合计 11.0 MiB → 4.59 MiB
```

## 自检

```bash
node lib/test.mjs      # 220 项单元检查：解析、类型、diff、改写配对、markdown、bundle 导出
node lib/validate.mjs  # 94 份快照全量解析，要求 0 行未覆盖、0 处表达式失败
node lib/smoke.mjs     # 6321 个函数 × 5 种视角，检查 id 唯一性、悬空边、内存一致性
```

`validate.mjs` 是这套工具可信度的底座：dump 文件是 Python 的一个极规则子集，
解析器必须对 **273,630 条语句一行不漏**，任何分析才谈得上准确。

## 已知边界

- `repo/pto` 镜像里缺少部分 Pass 的文档与源码（`BlockNzTensorViews`、`InsertCommFence`、
  `LowerPipelineToSlots` 等）。这些 Pass 的说明面板会明确标注「镜像里没有」，实测证据不受影响。
- MemRef 的 `size` 为 0 且 shape 含动态维时，内存视图显示「动态」而不是「0 B」——
  编译期确实算不出字节数，不应伪装成 0。
- 数据流视角每个函数最多 600 个节点、控制流 900 个，超出会标注「已截断」。
- 两份 run 的 Pass 序列不同（`l3_decode_csa` 52 个、`decode_fwd_layers` 42 个），
  这是编译配置差异，不是数据缺失。

## 新增一份 run

在 `build.mjs` 顶部的 `RUNS` 数组里加一项，指向仓库内的 `passes_dump` 目录即可：

```js
{ id: 'my_run', title: '…', subtitle: '…', dir: 'Data/…/passes_dump' }
```

Pass 名到文档 / 源码的映射走大小写与下划线无关的归一化（`FlattenTileNdTo2D` ↔
`flatten_tile_nd_to_2d`），新 Pass 通常不需要额外配置；
阶段分组和推荐视角在 `lib/passinfo.mjs` 里，未登记的 Pass 会落到合理默认值。
