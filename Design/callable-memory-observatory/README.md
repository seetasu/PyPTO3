# Callable Memory Observatory

基于一次真实 PyPTO pass dump 的两层内存观测 demo：

1. **Callable 之间**：横向比较同一编译运行中 116 个 callable 在 `InitMemRef`、`MaterializeSemanticAliases`、`MemoryReuse`、`AllocateMemoryAddr` 四个快照下的静态分配规模，可按内存空间和 callable 类型筛选。
2. **单个 Callable**：打开一个含片上分配的 callable，逐 Pass 查看槽位、容量和最终地址，并检查槽关联的 Tile 定义。

## 数据来源

- Run：`Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/`
- Pass dump：`33_after_InitMemRef.py`、`34_after_MaterializeSemanticAliases.py`、`35_after_MemoryReuse.py`、`36_after_AllocateMemoryAddr.py`
- 数据文件：`data/memory-observatory.json`
- 重建：从仓库根目录运行 `python Design/callable-memory-observatory/scripts/build_data.py`

当前快照中 57 个 callable 含片上 `tile.alloc`。例如 `csa_merge_pack_publish` 从 53 个槽、520,960 B 变为 12 个槽、165,120 B。

通过本地 HTTP 服务打开 `index.html`。跨 callable 容量是每个 callable 静态 `tile.alloc` 的横向对比；不同 callable 的调用关系和运行时并发数据不在这些 pass dump 中，因此不能把跨 callable 求和当作设备同时存活峰值。偏移只在 `AllocateMemoryAddr` 阶段表示实际分配地址。静态源码行号也不能替代真实生命周期区间。
