# Memory Pass Replay

用一次真实 PyPTO 编译 dump 构建两层内存观测：先看单个 callable 的内存池变化，再下钻到单个 Tile/分配对象。

- 第一层入口：`index.html`（Callable）；通过页面按钮可进入 `../memory-object-replay/index.html`（Object），对象层也可带着选中的 Tile 返回 callable 层。
- `launch.html` 的内存观测主入口现指向新版 `../callable-memory-observatory/index.html`；本目录的 Callable 页面与 `../memory-object-replay/index.html` 仍可直接访问。
- 数据：`data/memory-flow.json`，从 `Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/passes_dump/` 的连续快照提取。
- 更新数据：从仓库根目录运行 `python Design/memory-pass-replay/scripts/build_data.py`。
- 样例 callable：`csa_merge_pack_publish`（AIV）与 `tp_o_a`（AIC），页面顶部可以切换。

通过本地 HTTP 服务打开页面。Callable 层可以在 AIV `csa_merge_pack_publish` 和 AIC `tp_o_a` 之间切换，并按四个连续快照回放 `InitMemRef → MaterializeSemanticAliases → MemoryReuse → AllocateMemoryAddr`，对照静态容量、槽数和最终地址布局；AIC 的 Mat、Acc、Left、Right 地址空间分别展示。Object 层把 8 个相关 pass 横向展开到每条 Tile 定义，支持选择 Tile 并追踪其槽位、空间、偏移和 IR 行。Callable 层选中槽后可直接下钻到相关 Tile，对象层保留该 Tile 并可返回。AIV 样例中 Tile 数量为 138/140 条定义，分配槽数量为 53→12，二者是不同统计口径。

该 callable 的静态 `tile.alloc` 总量从 520,960 B 降至 165,120 B；这不是性能提升比例。地址布局只在 `AllocateMemoryAddr` 快照中展示；之前阶段尚无最终 offset，不绘制物理地址。

界面中的源码行号是静态 dump 行号。Tile 引用成员由每个 Pass 的 MemRef base 与定义记录关联；相邻快照的关系按 Tile 名称交集追踪，表达 IR 引用映射，不代表这些 Tile 同时存活或运行时并发。字节统计为静态 `tile.alloc` 总量，不推导运行速度。
