# Hardware-Native Lens

IDE 内的单页设计原型，用一个真实算子演示昇腾 Hardware-Native 体系的两件事：

- **可表达** —— Tensor/Tile、Scope、并行请求、内部布局这些硬件语义，本来就写在源码里，因此可以被读取、被标注、被可视化。
- **约束前移** —— 这些语义同时携带硬件约束。约束是纯几何/容量条件，所以可以在编码期判定，不必等编译或上板。

入口：[`index.html`](index.html)（也在仓库根的 `launch.html` 列表里）。

## 取材

全部结构、常量、依赖关系与约束来自 [`../Toolkit Studio/decode_layer.py`](../Toolkit%20Studio/decode_layer.py)
（Qwen3-14B fused decode，FP32 跨层 carry）。编辑器里是节选，**行号与源文件一致**，折叠区标出跳过的行数，便于回溯原文。

参考 Demo `Design/Toolkit Studio/` 的其余能力（模型结构图、Pass 流水线、Correctness Lab、运行详情、SOLO 模式等）在本设计中全部去掉，只保留 IDE 主页面这一条线。

## 页面结构

结构沿用 `vendor/pto-design-system` 的 `ide-frame` + `workbench-shell` pattern；视觉语言参考
`repo/AI_Infra` 的 demo（见下节）。

| 区域 | 内容 |
| --- | --- |
| 左栏 | 语义目录，按五个观察视角分组 |
| 中栏 | `decode_layer.py` 节选，顶部透镜开关控制行内语义标注与 scope 色带，并显示约束波浪线与 gutter 标记 |
| 右栏 | 9 条硬件约束的实时判定与选中约束的详情 |
| 底部 | 按工作负载、模型维度、运行时和算子策略分组的 10 个配置项；模型维度通常由模型决定，此处可调用于演示 |
| 状态栏 | 当前配置下的约束总判定 |

### 一个选中模型

左栏目录条目、源码里的高亮符号和右栏约束行共同控制当前选择：

- 选中**语义对象** —— 在对应代码行下方展开关键事实和 SVG 可视化；多个代码落点可在卡片中切换。右栏高亮相关约束，其余约束压暗但仍可点。没有关联约束时显示完整约束列表。
- 选中**约束** —— 右栏下钻显示规则、当前判定、硬件事实和代码落点，并链接相关语义对象；可返回原约束列表，保留筛选和滚动位置。
- **总览** —— 顶部显示四个关键派生量，下方列全 9 条。

约束区可按违反、有代价、已满足筛选：选中语义对象时筛选复位（约束区在讲的东西换了），在约束之间跳转时筛选保留（同一列表内的导航）。选中条目会定位源码，右侧面板收起时会自动展开。底部「快速体验」可从默认配置切换到容量与分页、切片对齐或并行分配场景；每个场景先恢复默认值再应用对应改动。手动调整时显示相对默认配置的修改项数。顶部重置恢复配置、透镜、筛选和总览。

## 视觉与代码渲染

视觉参考 `repo/AI_Infra`（`ai-infra-research/reports/llm-compute/*`、`atlas-950-explorer`）的做法：

- 近黑灰阶：`#111` 底 / `#131313` 画布 / `#171717` 面板 / `#202020` 抬升，1px 细线 `#353535`，平面无阴影
- 四色强调，与四个透镜一一对应：lime `#B1EF6B`（内部布局）、lavender `#BCA0F3`（Scope）、
  amber `#F2C45F`（并行请求）、sky `#79D4F2`（Tensor/Tile）；单一告警红 `#fd5644`
- 标签一律等宽小字 + 宽字距大写；选中态为浅底深字（`#e0e0e0` / `#111`）

代码渲染按常规编辑器口径分词后再按上下文着色，而不是只认关键字：

| 类别 | 判定方式 |
| --- | --- |
| 函数调用 / 定义 | 标识符后紧跟 `(`；`def` 之后的名字另算 |
| 关键字实参 | 标识符紧贴 `=`（PEP8 下赋值两侧有空格，实参没有），如 `name_hint=` `deps=` |
| 类型 / dtype | 首字母大写，或在 dtype 表内（`FP32` `BF16` `TASK_ID` …） |
| 常量 | 全大写标识符 |
| 模块 / 属性 | `pl` 限定符与点号链上的非调用成员 |
| 其余 | 关键字、字面量、装饰器、字符串、数字、运算符、标点各自分色 |

点号链（`pl.Level.CORE_GROUP`）整段作为一个可点击对象，内部各段仍分别着色。

## 五个观察视角

| 透镜 | 在代码里的落点 | 可视化 |
| --- | --- | --- |
| **Tensor / Tile** | `pl.Tensor[[...], dtype]`、`pl.create_tensor`、切片表达式、`pl.matmul` 的 A/B/C | 张量切块图、tile 去向与字节数、AIC/AIV 流水 |
| **Scope** | `pl.manual_scope()`、`pl.at(level=pl.Level.CORE_GROUP)`、`deps=[...]`、`pl.Array[_, pl.TASK_ID]` | 显式依赖边 vs 被抑制的 tensormap |
| **并行请求** | `pl.parallel`、`pl.spmd`、`pl.pipeline`、`pl.range(core, total, stride)` | 核上分布、栅格整除性、流水重叠窗口、work table 压实 |
| **内部布局** | `Q_HEAD_PAD` / `set_validshape` / `fillpad`、`cache_row` 算术、split-K 切片宽度 | 填充行 vs 有效行、逻辑块→物理页、切片与 slab 对齐 |
| **数值与精度** | Attention 与残差路径的 FP32 / BF16 转换 | dtype 转换边界与累加输出 |

依赖与 Task ID、Split-K 原子累加、K/V 页、Cube/Vector 交接和有效区域处理均可从目录或代码标注进入。流水与交接图表达源码意图，不是实测时间线；源码说明该版本在目标设备上预期存在代码生成失败。

嵌套 scope 以左侧色带表示（`manual_scope` → `pl.at` / `pl.parallel` / `pl.spmd` → `pl.pipeline`）。

## 九条硬件约束

每条都标注三件事：**规则**、背后的**硬件事实**、以及**若不前移会在哪一环暴露**。

1. `cube-m-fractal` —— Q 头打包行数对齐 cube M fractal 16（填充占比过半时给警告）
2. `page-identity` —— `SEQ_TILE == runtime.page_size`
3. `l0b-capacity` —— K tile 双缓冲 ≤ Right(L0B) 64 KB
4. `slab-1to1` —— `QKV_K_SLICE == DOWN_TN`（`deps=[prev_normed_tids[q_ks]]` 成立的几何前提）
5. `down-chunk-align` —— `HIDDEN % DOWN_TN == 0` 且 `DOWN_TN % 256 == 0`
6. `splitk-divides-k` —— `HIDDEN % QKV_OK == 0`
7. `gp-head-cover` —— `GP_SIZE == NUM_KV_HEADS`（work table 只编码 `(b, block)`）
8. `spmd-even-split` —— `(KV heads × batch) % ROPE_CORES == 0`
9. `block-balance` —— grid-stride 的整数打包偏差 ≲ 1.25×

## 怎么演示「约束前移」

底部旋钮改一个值，源码里的形状、Inspector 的布局图、波浪线与状态栏会同时重算。几条立刻出效果的路径：

- `SEQ_TILE` 改 256 → `l0b-capacity` 红（双缓冲 128 KB 超出 64 KB），`page-identity` 红（逻辑块跨页）
- `runtime.page_size` 改 64 → `page-identity` 红，paged 图上画出跨页区域
- `QKV_OK` 改 4 → `slab-1to1` 红，split-K 对齐图的连线变红虚线
- `HEAD_DIM` 改 64 + `DOWN_TN` 保持 1024 → `down-chunk-align` 红（2560 不被整除）
- `GP_SIZE` 改 4 → `gp-head-cover` 红（work table 缺头维度）
- `ROPE_CORES` 改 24 → `spmd-even-split` 红（128 items 不被 24 整除）
- `NUM_KV_HEADS` 改 4 → Q 填充占比升到 75%，`cube-m-fractal` 转警告

## 文件

```
index.html          IDE 页面骨架
styles/app.css      AI_Infra 配色映射、编辑器装饰、代码着色、SVG 配色
js/model.js         旋钮、派生量、九条约束、源码节选与语义标注、符号词典
js/source-extra.js  从原文件摘录的计算主链路，按原始行号与语义标注合并
js/viz.js           8 个 SVG 可视化 + 15 个概念面板
js/app.js           分词与上下文着色、渲染编排、选中与联动
```

无构建步骤、无外部依赖，直接用浏览器打开 `index.html` 即可。
