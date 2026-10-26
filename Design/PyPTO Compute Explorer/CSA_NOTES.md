# DeepSeek V4 CSA · 第一阶段

入口：index.html → 资源管理器 decode_csa.py → **Compute Explorer**。

本场景用于阅读源码中的数据归属与依赖。默认 TP=2。T 是运行时本地 token 数；数字 Shape 按本地默认容量 T=256、整组 2T=512 说明，不是每次运行的固定输入规模。

| 视角 | 整体 | 局部 |
|---|---|---|
| Assignment | 两个 Rank 的 Local / Gathered 数据、三种 Cache、Attention Group 重分布 | rank-major 行段、Cache 生产者与消费者、Group 到目标 Rank 的归属 |
| Pipeline | 按 Rank 分组的主要计算与通信依赖 | AllGather 信号阶段、Cache TaskID 就绪、Publish / All-to-All / ReduceScatter |
| Graph | 显示尚未适配状态 | 本轮不展开 |

选择节点后，Inspector 提供 What / Why / Data / How / Source。Assignment 与 Pipeline 保留语义对象和 Rank；某个 Tensor 不直接出现在目标视角时，选择映射到对应模块。源码按钮返回 decode_csa.py 的真实行段；子算子证据可在详情中展开。两个文件的源码位置与两个场景的可视化状态分别保存。

## 关键阅读口径

- Q 与 Indexer query 读取 local token 行；KV 和两个 Compressor 读取 gathered stream。local 数据独立不意味着 Q 已与 AllGather 重叠：Q 调用位于后续 scope。
- AllGather 得到的是每个 Rank 上的全组副本，顺序为 [Rank 0 rows | Rank 1 rows]。
- Original、Compressed、Indexer KV Cache 独立。Main 与 Indexer Compressor 也各自维护请求状态。
- cache_ready_dep 合并 Original 和 Main 压缩 Cache 的写入 TaskID；Indexer 另外等待 idx_cache_write_tid。图中不添加虚构的全局 barrier。
- Attention 输出先按 Group 归属融合 Put / Notify，再由 o_group_a2a 等待、读回和回收信号；它不是再发送一次相同 payload。
- O Projection 的各 Rank 部分和发回 token owner，owner 做 FP32 累加。ReduceScatter 不是单纯切分 token。
- l3_decode_csa 对数据取 [rank]，传 tp_rank=rank 与 device=rank。host rank 循环不代表两个设备串行执行。

## 源码与工程边界

证据来自本地相对路径 ../../Data/DeepseekV4/deepseek_v4_flash_dspark/decode_csa.py，以及同目录的 decode_cp_token_allgather.py、decode_o_proj.py、config.py。

js/decode-csa-source.js 保存上述源码的本地快照，主文件包含原始相对路径与 SHA-256。源码展示和详情不需要运行时 Fetch、CDN 或外部站点；原始源码更新后需同步快照及行段。运行静态服务器时以 PyPTO3/ 为根，保留现有 ../../vendor/pto-design-system/ 相对依赖。

复用既有 IDE Frame、深色主题、按钮和 Tab；图形通过共享 model-graphviz.renderController 直接嵌入，使用其节点、选择、拖拽和缩放。CSA 的数据和状态独立于 Paged Attention 引擎。

布局、连线和行段块为静态简化示意，不推断运行耗时、实际重叠、带宽、物理地址或 Tile 指令。Pipeline 全览密度较高，可用“阅读”、缩放及局部视图查看细节。未接入真实编译器或 Runtime，未做 CSA Graph、Morph 或播放模拟。
