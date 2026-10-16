# Pass Error Cases · CP-01

## Memory Inspector · MI-01

独立入口：`index.html#case=MI-01&pass=32`。迁移 Memory_V2 的 Function Memory Map，外围使用现有 CP-01 页面组件；未修改任何 CSS 或共享图元样式。
数据由 `Design/memory-inspector/memory_map.py` 在开发与构建前解析 `32_after_AllocateMemoryAddr.py` 生成，构建检查要求结果与源数据精确一致；该快照独立于 CP-01。
支持全部函数与单函数视图、Buffer 总览、缩放，以及内存空间与地址比例尺选择；「内存证据」提供 Tile 列表、静态生命周期及逐行存活联动。
横向表示字节地址，纵向表示源码行；恢复各空间的地址轴、源码行轴和 Tile 矩形。Tile 仅使用 CP-01 已有的存储中性色与选中蓝色，不增加空间配色、描边或分隔线；View 和异常通过数据详情识别。窄 Tile 和重叠 View 可通过列表选择。
只接入一个真实快照，不提供虚构的跨 Pass 播放。上游 Python 分析器不在本目录，本页消费解析 JSON，不执行 Python。
验证：`node scripts/inspector.test.mjs`。

独立探索副本，复用 CSA Pass Field 已批准的组件与样式；原目录未修改。

- 入口：`index.html`（内联独立页面，支持 file://）。
- 源码入口：`entry.html` → `src/CaseApp.jsx`；数据：`src/case-data.js`。
- 构建：`npm run build`；开发：`npm run dev`。
- 三种实现、四路 scale、共址分组与双向绑定选择、六个检查点、证据检查器。
- 来源：本地归档 pypto #1723，报告版本 7fe857a895078333dd95fdc8ab915f733664c063。局部图按 Issue 重建，不代表完整 dump。
- 待获取：Pass 快照、精确分配大小、live range、中间值与硬件复验。无证据的检查点不生成编译变换。

## 2026-10-16 交互修订
默认 Before 告警，用户通过「处理告警 → 应用：拆分 mix_x scope」才进入 After；仅分离 buffer 的失败尝试保留。撤销原三态 tabs 与六检查点替代导航。恢复 51 Pass 参考管线（来源为原 CSA 页面，不能作为 CP-01 真实版本顺序）；逐 Pass 快照尚未接入，图保持最终地址证据，不伪造中间变换。
