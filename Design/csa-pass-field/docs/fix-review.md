# Fix 候选审阅 · 2026-10-15

状态：代码与数据验证通过，等待用户 approve；浏览器验收未完成。

## 已完成

- F01：43/45/50 明确整程序变化、qk_pv 不变；observations 分开记录两个范围；22/40/51 no-op 保持。
- F02：图中标明共享编译结构、完整运行 2 ranks；kernel 标识目标类型；检查器显示原始记录驱动的每 rank AIC/AIV 资源清单。当前任务实际用核数明确为尚未关联。
- F03：Pass 详情新增完整 CSA 前后源码变化摘录，含原始文件名与行号；离线内嵌，无外部 fetch。摘录采用独有行加上下文，不等同完整序列 diff。
- F04：生成 object-links.json 身份契约，每个静态对象明确 unresolved、原因与空运行身份；定义 run/rank/dispatch/task/func/core/record 命名空间。精确运行映射仍属于 M1，未伪造耗时或任务关联。
- F05：数据 manifest 增加 run、输入 SHA-256、生成器哈希；产物 manifest 增加 HTML SHA-256。已比较 AI_Infra 旧副本，版本不同，尚未覆盖。比较结果见 fix-artifact-comparison.json。
- 通用泳道：修正 TRACE_URL 多余单引号；目标 JSON 文件存在。HTTP 页面加载仍待浏览器验收。

## 验证

- npm run build：通过（包括数据重建、数据校验、Vite、独立 HTML 内联）。
- node --test scripts/*.test.mjs：15/15 通过，包含 pass no-op、ISA 静态身份、横纵布局与上下文图。
- 新增数据断言：43/45/50 范围与证据、22/40/51 no-op、23 拆核、资源计数、未匹配身份、输入哈希。
- 独立 HTML 约 1.8 MiB；Vite 保留单文件 bundle 大小提示。
- 浏览器工具拒绝 file:// 协议，未绕过策略；实际渲染、点击与旧 URL 恢复未完成验收。

## 审阅入口

[本轮候选页面](../dist/index.html)：查看 Pass 43/45/50 的标题与详情；展开源码变化、运行范围与版本；切换最终 ISA 查看 Kernel 目标类型。

双卡 DAG、真实执行泳道、内存性能诊断未实施。源码候选与 AI_Infra 展示副本尚未同步。用户 approve 后再推进后续阶段。

## 导航状态补修

Pass 导航不再按可视动画差异降低透明度。所有编号保持可读；实线标记局部变化，虚线标记全局变化，空心点标记 no-op，悬停与无障碍名称包含解释。选中态与播放步骤保持独立。已重新构建并验证 visual-diff 回归；file:// 浏览器验收仍受先前策略限制。
