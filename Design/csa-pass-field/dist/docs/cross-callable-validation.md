# 跨 callable 源码与数据验证

日期：2026-10-16。输入：`Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617`。全部来自同一真实编译运行，读取原始 52 份 passes_dump 和各自 PTO 文件，未执行输入 Python。

## 验证范围

两个已有样本作为回归基线，五个其他 callable 用作结构反例，共 7 × 52 = 364 个阶段。此处验证 callable 发现、阶段源码变化、核装饰器、全操作清单、分配及通信信息；不是新样本 UI 接入或浏览器验收。

| Callable | P23 执行归属 | P33 → P35 显式分配数 | P35 已知分配字节和 | 对通用化的约束 |
| --- | --- | --- | --- | --- |
| qk_pv（基线） | AIC + AIV，含 Group/SPMD 包装 | 53 → 12 | 361216 B | 分核、多个空间和跨核关系；不能只取家族第一函数 |
| csa_merge_pack_publish（基线） | AIV + SPMD 包装 | 53 → 12 | 165120 B | CommCtx、put、notify、两个 fence |
| csa_cache_writeback | AIV + SPMD 包装 | 1 → 1 | 1024 B | 访存样本不应套六个计算业务阶段 |
| idx_qr_proj_matmul | AIC + SPMD 包装 | 23 → 11 | 405504 B | Acc/Mat/Left/Right 四空间，没有 Vec；不得生成 UB |
| cp_token_allgather_payload_wait | AIV | 0 → 0 | 0 B（无显式本地分配） | 等待同步，PTO 含 twait；无分配不代表无行为 |
| comb_sinkhorn | AIV + SPMD 包装 | 132 → 18 | 4608 B | 大量分配身份与复用；不应按节点数量任意改变容量纹理 |
| csa_rope_interleave | AIV | 34 → 6 | 4096 B | gather、视图、多个循环；不应套 merge 样本区域分类 |

字节和按家族各函数声明求和，不能解释为同时占用、芯片容量、峰值或性能收益。qk_pv 的不同核/空间必须分开解释。PTO 的 alloc_tile 与 Python IR 的显式 allocation 处于不同抽象层，数量不要求相等。

## 实际发现与修复

第一次审计发现 idx_qr_proj_matmul 和 comb_sinkhorn 在 P0–8 为空。核查 frontend 后发现它们存在明确 name_hint，但写作 `for ... in pl.spmd(...)`。原 probe 只识别 `With`，漏掉 `For`。

已在 probe 和审计脚本中共同补充 For.iter 的 name_hint 识别，并重新读取全部快照。七个样本最终均在 52 个阶段有候选 scope/function，且附函数名、装饰器、行号、原始文件摘要。该结果只证明当前七个样本的发现覆盖，不证明前缀匹配就是可靠跨 Pass lineage。

新增回归测试使用最小 For scope → AIV 函数的两阶段输入，验证早期发现、outline 转换、保留 fence、读取 twait。真实数据断言同时覆盖单 AIC 四空间、零分配等待、132→18 分配复用和 P50 两个 fence。

## 规则得到的验证

1. **核归属必须来自输入。** 同一运行里同时存在单 AIC、单 AIV、混合核；固定 AIV 框或 UB 无法通用。
2. **需要通用操作库存与专用显示分层。** twait、fence、CommCtx 即使没有算术或本地分配也有重要语义，不能被 tensor/tile 白名单过滤。
3. **输入发现必须覆盖不同 scope 语法。** 同一个 name_hint 可以出现在 With 或 For 中；解析失败/缺失要报告，不能默认为 no-op。
4. **内存表达不能固定为 UB。** 单 AIC 样本同时使用 Acc/Mat/Left/Right；等待样本没有本地显式分配，应不出现虚构的内存面板。
5. **变化稀疏是允许的，但需证据。** JSON 中 changedPasses 仅是去掉位置属性的 AST 家族变化，不是主图变化，绝不可直接用于下划线/播放。
6. **聚合数量不足以验证图。** 本次尚未重建这些样本完整定义/使用、CFG、视图重叠和跨阶段 lineage；不将统计审计当成图结构正确。

## 尚未通过的通用化验收

- scene builder 仍按 merge 的六业务区、固定布局和显式源码范围配置。这五个新样本没有被其真正接入，因此“只换输入、不改 renderer”的验收尚未通过。
- 新审计脚本记录全部 pl/pld 调用，但尚无完整语义 IR 与未知节点覆盖率；操作库存不是语义解析。
- 家族前缀候选可能误收兄弟函数；本次未证明稳定对象匹配或所有作用域边界。
- 多样本来自同一 run，未验证其他模型/编译器版本/不完整输入包。
- 未做新样本的浏览器默认、展开、hover、选择及播放验收。

因此实施优先级仍是统一 IR → 模型驱动布局与交互 → 多样本只换数据接入 → 上传入口。原页面与其运行数据未替换。

## 复现

在仓库根目录执行：

```sh
rtk proxy python3 Design/csa-callable-explorer/scripts/validate-other-callables.py --run Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617 --output Design/csa-callable-explorer/docs/validation/cross-callable-audit.json
rtk proxy python3 Design/csa-callable-explorer/scripts/test-callable-audit.py
rtk proxy npm --prefix Design/csa-callable-explorer run test:callable
```

机器结果：[cross-callable-audit.json](validation/cross-callable-audit.json)。规则：[callable-explorer-rules.md](callable-explorer-rules.md)。JSON 包含逐阶段文件摘要、函数范围、装饰器、参数、全部操作计数及 MemRef/分配信息，可定位到原始源码复核。
