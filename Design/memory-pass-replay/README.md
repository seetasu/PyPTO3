# Memory Pass Replay

用一次真实 PyPTO 编译 dump 回放单个 callable 的内存处理阶段。

- 入口：`index.html`；已加入仓库 `launch.html`。
- 数据：`data/memory-flow.json`，从 `Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/passes_dump/` 的连续快照提取。
- 更新数据：从仓库根目录运行 `python Design/memory-pass-replay/scripts/build_data.py`。
- 样例 callable：`csa_merge_pack_publish`（AIV）。

通过本地 HTTP 服务打开页面。唯一观测对象是整个 callable `csa_merge_pack_publish`。页面以四个连续快照回放 `InitMemRef → MaterializeSemanticAliases → MemoryReuse → AllocateMemoryAddr`：每阶段展示 callable 的 `tile.alloc` 槽位数和静态容量；地址分配阶段按字节偏移绘制整个 Vec 布局。选择分配槽后，关系视图按相邻 Pass 中共同的 Tile 名称列出上游/下游 MemRef base 与交集数量，便于观察别名改写和多槽汇入复用槽；侧栏可查看容量、偏移、关联 Tile IR 定义与源码行，清单支持按 MemRef 或 Tile 名筛选。Pass dump 中 Tile 数量为 138/140 条定义，分配槽数量为 53→12，二者是不同统计口径。

该 callable 的静态 `tile.alloc` 总量从 520,960 B 降至 165,120 B；这不是性能提升比例。地址布局只在 `AllocateMemoryAddr` 快照中展示；之前阶段尚无最终 offset，不绘制物理地址。

界面中的源码行号是静态 dump 行号。Tile 引用成员由每个 Pass 的 MemRef base 与定义记录关联；相邻快照的关系按 Tile 名称交集追踪，表达 IR 引用映射，不代表这些 Tile 同时存活或运行时并发。字节统计为静态 `tile.alloc` 总量，不推导运行速度。
