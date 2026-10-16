# 从 qk_pv 场景 Demo 到 Callable Explorer：本地数据分析

日期：2026-10-16。范围：复制探索稿、提取目标样本并评估通用化；未改动正式页面、未实现上传界面、未宣称已自动生成目标计算图。

## 结论

可以做，但不是将 qk_pv 字符串替换成另一个名字。现有图元、React Flow 交互和 Pass 导航可以保留；场景模型、内存绑定、指令归属和布局需要从硬编码改为数据驱动。

`csa_merge_pack_publish` 是很好的第二样本：单 AIV、无 Matmul、无 AIC↔AIV handoff，却有循环携带状态、共享 UB、子视图偏移及跨 rank 通信。把它正确接入，才能验证通用化能力。

## 本地输入与可追溯性

运行目录：`Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617`。

| 数据 | 路径 | 用途 |
|---|---|---|
| 前端与 Pass IR | `passes_dump/00_frontend.py` 至 `51_after_MaterializeValidShapeSymbols.py` | 52 个阶段的 scope、callable、算子、控制和内存变化 |
| PTO IR | `next_levels/decode_csa_test/ptoas/csa_merge_pack_publish.pto` | vector kernel、tile/通信指令、loc、地址 |
| 生成 C++ | `next_levels/decode_csa_test/ptoas/csa_merge_pack_publish.cpp` | PTO 代码生成核对 |
| AIV kernel C++ | `next_levels/decode_csa_test/kernels/aiv/csa_merge_pack_publish.cpp` | 后端产物核对 |
| Rank 上下文 | `dfx_outputs/rank{0,1}/d0/{chip_swimlane_records.json,dispatch_program.json,name_map.json,deps.json}` | 可选运行态身份关联；本轮没有建立精确实例映射 |

新脚本 `scripts/probe-callable.py` 用 Python AST 读取静态文本，不执行源代码。输出 52 份所选片段、每阶段结构、显式分配与绑定、PTO 静态出现位置及哈希。函数名按 `csa_merge_pack_publish` 和其后缀 family 匹配，早期 scope 按明确 name_hint 匹配。名称选择不是全局子串替换。

## 目标 callable 实际做什么

前端 `00_frontend.py:1710–1763` 的主体是：

1. 接收 sparse block 的 mᵢ、ℓᵢ、Oᵢ，按 worker 分发 pack_work。
2. 在分块循环中用新的最大值重标定 α/β，合并分母与输出累加值。
3. 加入 attention sink，计算归一化结果并转换 BF16。
4. 对尾部 64 维做 RoPE：gather、列扩展乘、相加，再 concat 回 512 维。
5. 将头维结果打包写入 attention_grouped。
6. 根据 destination_rank/local_group 的地址计算发布到 attention_window，满足条件后更新 attention_signal。

SPMD scope 显式声明 48 workers，并依赖 qk_tid、attn_rope_tid。48 是该份 IR 的提交参数，不能直接解释成每个运行实例独占 48 个物理核。

## 必须按目标数据重新生成的阶段

| 阶段 | 观察 | 对新工具的要求 |
|---|---|---|
| P0–8 | 计算仍在名为 csa_merge_pack_publish_spmd 的 With scope 中 | 支持 scope 选择及 scope→callable 追踪，不能显示空图 |
| P9 | 抽取 csa_merge_pack_publish InCore callable | 跟踪抽取关系 |
| P10 | 增加同名 family 的 SPMD wrapper | 区分主计算 callable 和 wrapper，保留任务依赖 |
| P11 | Tensor 运算下降为 Tile | 保留类型、shape、迭代与值边 |
| P23 | 同名 callable 的装饰器变为 AIV；没有分成 AIC/AIV 两个函数 | 比较必须覆盖装饰器/签名，不能只比较函数体 |
| P33/34 | 53 个 Vec 显式分配，合计 520,960 B（508.75 KiB） | 按实际空间生成 UB，不生成空的 AIC/L0 面板 |
| P35 | 12 个 Vec 分配，165,120 B（161.25 KiB） | 以 buffer 身份展示共享关系；分配字节合计下降约 68.3% |
| P36 | 地址显式化，局部地址最大端点 165,120 B | 区分分配范围与绑定视图范围 |
| P39/45/46/50 | 目标 family 源码仍有变化 | 不能沿用 qk_pv 的变化/no-op 标签和证据 |
| P51→ISA | 单 vector kernel，Tile 与通信操作共同出现 | 不添加虚构的 Cube、Matmul 或跨核 handoff |

53/12 是函数内静态 alloc 语句个数；字节是这些分配的 size 求和。复用前合计不是运行时峰值，复用比例也不是测得的硬件性能改善。P36 的 161.25 KiB 是当前样本分配地址范围，不能称作 UB 硬件容量。

所选 scope/family 文本（含装饰器、签名）在 51 个相邻 Pass 中有 21 次变化：1、2、4、5、7、9、10、11、18、21、23、30、31、33、34、35、36、39、45、46、50。文本变化不等于 DAG 拓扑改变。

## 内存细化比 qk_pv 更能暴露解析问题

P36 的 12 个 Vec buffer 覆盖 [0,165120)，共找到 140 个局部带 MemRef 的值声明；这是静态绑定声明数量，不是同时存活值数量。`mem_vec_66` 是 65,536 B，起点 99,584，同时承载多种视图，例如 [99584,165120)、[100608,101632)、[101376,132352)。

因此通用数据层必须同时记录：

- allocation：buffer、空间、所属 callable、分配大小；
- binding：value、buffer、offset、size、shape、dtype、源码位置；
- use/def：谁写入/读取该值；
- 控制路径和循环携带关系：避免把互斥或跨迭代的值当成同时存活。

不能继续使用 `memoryOwner()` 里 `/max/`、`/sum/`、`/oi/` 这类名称猜测；这个样本有多个 exp/max/mul，同类型算子也不能合并成一个节点。全局 Tensor 的 mem_ddr 字符串引用和动态大小 0 必须与局部 Vec 分开；0 不能解释成真实零容量。

## ISA：不能漏掉通信命名空间

本地 PTO 有 68 处 `pto.t*` Tile 指令静态出现位置，另有 2 处 `pto.comm.tput` 和 1 处 `pto.comm.tnotify`。还有 barrier/fence 等同步 IR，不能用只匹配 `pto.t\w+` 的规则当作完整 ISA 清单。

Tile 操作中包括 TLOAD 13、TSTORE 16、TEXP 5、TADD 7、TSUB 6、TMUL 4，以及 TGATHER/TCONCAT/TCVT 等。它们是静态源码出现次数，不是执行次数。

指令到 Op 应综合输入输出值、loc 和 callable 归属。507 个 loc 出现位置仅有 46 个唯一位置，单独按 loc 或操作类型会产生歧义。对无法精确匹配的指令保留 kernel 归属，不强行连到一个代表 Op。

## 当前代码哪些可复用、哪些不能

| 层 | 可复用 | 必须改造 |
|---|---|---|
| 解析 | `pass-transform-explorer/lib/pyir.mjs` 的 tokenizer/parser，`analyze.mjs` 的结构分析 | 上传输入清单、scope family 追踪、诊断与未知语法处理 |
| 数据模型 | graph/passes/views 的职责划分、对象身份和来源字段 | 删除固定 51 Pass、qk_pv-only 路径与人工 Pass changeset |
| 主体图 | React Flow、已确认图元、选中/悬停、缩放/拖动 | 为任意 callable、核心数量、循环和操作集生成图，不固定两条 lane |
| 内存 | 灰色底板、范围 grid 高亮和分配详情 | 从真实 def-use/alias 建立绑定，按实际空间生成组件 |
| ISA | 指令集合/展开、静态出现次数、悬停映射 | 去掉 qk_raw/qk_oi 固定归属和固定指令类型清单 |
| Inspector | 分组卡片、渐进展开 | 由所选对象能力决定内容，不复述 qk_pv 的结论 |

现有 `build-csa-qk-pv.mjs` 的 allNodes/allEdges 和 changesByPass 是人工叙事；`MemoryField.jsx` 固定 kernel.aic/kernel.aiv、节点名和坐标；`instruction-collections.js` 固定 softmax/Matmul/handoff 对应；App/graph-state 直接 import qk_pv JSON。只换输入文件会导致错误图、遗漏和空指针。

现有结构解析器在 P0、9、11、33、35、51 六份完整程序上都报告 0 expression parse errors；P51 主 callable 生成 251 个语法数据流节点，未触及默认 600 节点截断，另有 17 个控制树节点。这个结果验证了基本语法覆盖，不证明分支/循环回边、别名或跨 Pass 身份已经正确重建。

## 建议的上传产品边界

首版是“上传已有编译产物，分析指定 callable”，不负责在线执行或编译上传的 Python。

必需输入：Pass 快照集合（文件夹或 ZIP）和所选 callable；可选 PTO 文件、原始源文件、运行 manifest/dispatch/trace。没有 PTO 就只显示编译视图，不伪造 ISA；只有最终 PTO 就提供最终结构，不能还原缺失 Pass。

导入流程：识别文件→生成可选 callable/scope 清单→用户选择→提取 family 和阶段变化→生成通用图模型。文件名没有明确序号时应允许配置顺序，不能默认为当前 51 Pass 顺序。分析器产出能力/缺失字段；界面只展示可用能力。

阶段关系：前端 scope→InCore callable→SPMD wrapper/AIV callable→buffer/view→PTO occurrences。身份映射以显式名字、调用关系、属性和来源为证；改名/拆分未能自动解决时要求选择对应关系，而不是静默猜测。

布局建议：这个样本只生成 AIV frame + UB 和必要的全局输入/输出/发布对象。默认按真实数据流组织，密集的 scalar 地址计算和循环逐级展开。可以用“合并→归一化→RoPE→打包→发布”作为人工核对后的语义分组，但当前解析脚本不会自动理解这些阶段名。

## 下一步实施顺序

1. 在副本把 `analysis.json` 适配为独立 callable 数据入口，先不做上传外壳。
2. 用单 AIV+UB 样本替换 qk_pv 的硬编码主图，核对 def-use、循环、子区间和通信。
3. 对比 qk_pv 与 merge 两个样本，验证同一适配器能生成单/双核图；人工语义分组作为可选配置。
4. 最后加入本地上传、callable selector、能力诊断与增量缓存。上传本身不作为“通用化已完成”的证明。

## 本轮交付与验证

- 独立副本 `Design/csa-callable-explorer`，不复制 dist 和 node_modules，依赖通过本地链接复用。
- `scripts/probe-callable.py`：参数化只读提取，不执行输入代码。
- `data/callable-probe/analysis.json`：52 个阶段、显式内存绑定、71 个 Tile/通信出现位置和来源哈希。
- `data/callable-probe/snapshots/`：52 个目标片段；`parser-check.json`：现有解析器六阶段检查。
- 断言通过：52 阶段均有目标片段、P23 核类型变化被记录、53→12 分配与字节数、68 Tile+3通信数量。
- 当前 UI 仍是复制来的 qk_pv 基线；本轮没有将分析数据冒充已完成的自动可视化。

## 2026-10-16 接入更新

副本已接入 `data/callable/`，替换了 qk_pv 的数据入口与双核布局。计算区域支持双击展开实际算子；UB 使用逐阶段数据；ISA 使用显式源码区域映射，含 TPUT/TNOTIFY。最初“UI 尚未接入”的描述属于本报告前一阶段，现在以本节和 README 为准。

验证：构建与场景校验通过；104 个 Pass/方向几何状态无图元重叠，五类有存储访问的区域均有 UB 高亮；六区域展开结构通过。浏览器读取被访问策略拒绝，因此未宣称已做视觉验收。
