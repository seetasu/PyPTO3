# Memory Lane

单个 callable 的内存线工具：覆盖全部 8 个内存相关 pass，顶部是可判好坏的 KPI，主画布是全部 tile 的槽位归属。

- 入口：`index.html`；已加入 `launch.html`（挂在 `Memory Pass Replay` 卡的「02 Object」变体下）。
- 数据：`data/memory-lane.json`，由 `scripts/build_memory_lane.mjs` 从 pass dump 直接生成。
- 通过本地 HTTP 服务打开；仓库已有 `.claude/launch.json` 的 `design-static`。

## 重建数据

```bash
node Design/memory-object-replay/scripts/build_memory_lane.mjs
```

解析 `Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/passes_dump/` 下 8 个快照里 `csa_merge_pack_publish` 的全部 `pl.Tile` 声明与 `pl.tile.alloc`，算出每步的 tile 数、槽数、静态容量、改写量，以及页面顶部的 KPI。

## 覆盖的 8 个 pass

| # | pass | tile | 槽 | 静态容量 | 本步改写 |
| - | ---- | ---- | -- | -------- | -------- |
| 12 | OptimizeOrchTensors | 64 | — | — | — |
| 19 | InferTileMemorySpace | 65 | — | — | 0 |
| 24 | InjectGMPipeBuffer | 101 | — | — | 0 |
| 33 | InitMemRef | 138 | 53 | 508.8 KiB | 0 |
| 34 | MaterializeSemanticAliases | 138 | 53 | 508.8 KiB | **1** |
| 35 | MemoryReuse | 140 | 12 | 161.3 KiB | **115** |
| 36 | AllocateMemoryAddr | 140 | 12 | 161.3 KiB | 0 |
| 38 | FuseCreateAssembleToSlice | 140 | 12 | 161.3 KiB | 0 |

槽位的概念在 33 才诞生；24 与 38 对本 callable 是空操作。**pass 36 之后内存线进入 PTOAS（`PlanMemory` 等），那部分不产出 pass dump，取不到。**

## KPI 与判据

| KPI | 本 callable | 判据 |
| --- | ----------- | ---- |
| **Vec 占用率** | **87.6%**（161.3 / 184 KiB，余量 22.8 KiB） | 预算取自 `repo/pto/src/backend/common/soc.cpp:138`——184 KB 安全、192 KB 物理（PTO-ISA 保留顶部约 8 KB，见 pto-isa#170）。超 80% 即无余量，是红线维度 |
| 静态总量 | −68.3%（508.8 → 161.3 KiB） | 无绝对好坏，需与同 run 其它函数横比 |
| 分配槽 | 53 → 12，每槽均 11.7 个 tile | 密度低的大槽是可疑点 |
| 碎片率 | **0.00%**（空洞 0 B） | 地址完全紧凑，这一维度没有问题 |
| 集中度 | 前 3 槽占 **79.4%**，最大 `mem_vec_66` 39.7% | 优化只能盯大槽；小槽合计 0.8%，优化无意义 |

**结论：内存紧张（87.6%）但原因不在碎片、不在小槽、也不在复用没做——瓶颈是几个大 tile 本身的尺寸，要往上游 tiling 策略找。**

## 读图

- **主画布**：每行一个 tile（140 个），8 列是 8 个 pass。行按最终所在的槽排序，同槽住户相邻。
  - 深浅交替条带 = 槽的分界。33/34 列被切成五十多条，35 之后并成十几块。
  - 格子左缘亮边 = 这一步该 tile 被改写了槽位。填充只表示"在哪个槽"，两个通道分开。
  - 最暗的格子 = 该步尚未建立槽位（pass 12/19/24）。
  - 列头的 `↻N` = 本步改写了多少个 tile。
- **地址布局**：最终 12 个槽在 Vec 上的实际区间，选中的槽高亮。
- **详情表**：选中 tile 在 8 个 pass 的槽位、偏移、空间、形状、dump 行号；变化的格子着色。

顶部下拉或点击任意行可换观测对象。默认选一个语义名、且槽位发生过改写的 tile。

## 口径与限制

1. 容量为 `tile.alloc` 的静态求和，是编译期上界，不是运行时峰值，也不是性能比例。
2. 没有真实生命周期（dump 只有静态行号），因此**算不出复用的理论下界**——无法判断 68.3% 是接近最优还是仍有空间。
3. pass 36 之后进入 PTOAS，取不到。
4. tile 跨快照按名字匹配。SSA 改名会被当作不同 tile；`t__tile_N` 这类计数器名还会被不同临时量复用，这些行的跨列连续性不可靠。
