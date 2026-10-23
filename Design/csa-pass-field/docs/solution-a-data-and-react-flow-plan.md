# CSA Pass Field · 方案 A 数据与 React Flow 技术计划

状态：方案 A 已实施  
日期：2026-10-13  
范围：方案 A 主体——以一张持续存在的 CSA DAG 表达 `qk_pv` 如何逐 Pass 从高层计算下降为 Kernel、Task、Memory 与 PTO-ISA 指令。  
不在本计划内：方案 B 的 2.5D Pass 深度栈、完整工作台外围 UI、复用 Pass Transform Explorer 的任何界面元素。

实施入口：[`../index.html`](../index.html)  
离线成品：[`../dist/index.html`](../dist/index.html)，生产构建会内联 JS/CSS，可直接通过 Chrome 的 `file://` 打开  
数据目录：[`../data/csa-qk-pv/`](../data/csa-qk-pv/)，源码侧继续以独立 JSON 为数据真源  

实施后的源数据核验结果：51 个 Pass 中，33 个改变整份程序 IR，18 个是前后快照完全一致的 no-op；`qk_pv` callable family 自身在 17 个 Pass 中变化。`AutoDeriveTaskDependencies` 在此 run 上属于真实 no-op，Task 与前置依赖实际在 `OutlineClusterScopes` 后已经出现。ISA 被实现为最终 Pass 状态上的 PTOAS 投影，不被错误归因给 `MaterializeValidShapeSymbols`。

## 1. 已确定的方向

### 1.1 主体是什么

画面主体是 **一张保持对象身份连续的 CSA DAG**，但叙事主体是 **当前 Pass 对 DAG 的改写**。

它不是：

- 每个 Pass 一张互不关联的快照；
- Before / After 两张图并排；
- 把 Tensor、Kernel、Task、ISA 都画成同一种通用卡片；
- 只用新增/删除数量解释 Pass。

它应当是：

- 同一个对象在 Pass 前后尽量保持位置和稳定 ID；
- 当前 Pass 只增强其实际触及的局部子图；
- 数据流、控制流、Scope、Kernel、Task、Memory、ISA 保留各自语义；
- 默认显示经过提炼的 `qk_pv` 核心问题场景，复杂细节通过局部展开进入同一张图；
- 51 个 Pass 均可到达，真实无变化的 Pass 明确呈现为 no-op，连续 no-op 可压缩但必须能展开。

### 1.2 前端技术结论

复用 `llvmcfg-standalone` 所采用的 **React Flow 节点/边技术框架**，不复用其工作台、四阶段流程或 LLVM Basic Block 组件。

| 技术部分 | 决策 | 说明 |
| --- | --- | --- |
| React Flow 受控画布 | 复用 | Pan、Zoom、Fit、选择、Handle、Node/Edge API 已满足方案 A 基础能力 |
| Custom Node 机制 | 复用 | 重写为 Tensor、Op、Scope、Kernel、Task、Memory、ISA 图元 |
| Custom Edge 机制 | 复用 | 建立数据流、控制流、任务依赖、内存绑定和下降投影等边类型 |
| LLVM `CustomNode.tsx` | 不复用 | 强绑定 Basic Block、`isSame` 与 Graphviz label |
| LLVM `GraphJSON` | 不复用 | 无法表达对象层级、谱系、Scope 和 Pass delta |
| Dagre | 运行时纯几何布局 | 复用 LLVM Flow 的 `rankdir + node size + smoothstep` 技术路线；只在可见结构集合变化时求解，no-op 与纯属性 Pass 不重排 |
| `llvmcfg-standalone` compiled runtime | 不复用 | 它是保留的编译产物，不是新页面的依赖库 |

现有实现证据：

- React Flow、Handle、Dagre 和受控 nodes/edges：[`BaseLayoutFlow.tsx`](../../llvm-flow/llvm-flow-frontend/src/components/modules/llvmcfg/BaseLayoutFlow.tsx)
- LLVM 专用节点：[`CustomNode.tsx`](../../llvm-flow/llvm-flow-frontend/src/components/modules/llvmcfg/CustomNode.tsx)
- LLVM Graphviz 数据形状：[`graph.ts`](../../llvm-flow/llvm-flow-frontend/src/types/graph.ts)
- canonical standalone 与编译 runtime 的边界：[`frontend-architecture.md`](../../llvm-flow/docs/frontend-architecture.md)

实现时优先使用仓库已经声明的 `reactflow` 11.4.2，而不是继续绑定 `react-flow-renderer` 10.1.0。第一阶段不在线升级依赖；先以仓库锁定版本完成最小验证。

## 2. 场景与事实来源

### 2.1 场景

第一版只做一个经过提炼的问题场景：

> `qk_pv` 中 Q/KV → QK → online softmax → PV → output 的混合 Cube / Vector 计算，如何经过语义下降、拆核、跨核传递、内存实体化、任务依赖推导，最终投影为真实 PTO-ISA 指令。

核心问题不是“Pass 总共改了多少行”，而是：

1. 高层 Tensor/Op 在哪一步变为 Tile 语义；
2. 为什么同一个计算被拆成 AIC/Cube 与 AIV/Vector 两部分；
3. 两部分之间的数据和控制如何通过 TPUSH/TPOP 等机制连接；
4. Tile 在哪一步获得 memory space、MemRef、复用关系与地址；
5. Task 与依赖在何时出现；
6. 最终可以在真实 `qk_pv.pto` 中找到哪些指令。

### 2.2 本地事实源

| 事实层 | 来源 | 用途 |
| --- | --- | --- |
| 逐 Pass IR | `Data/**/passes_dump/*.py` | Pass 前后事实、对象出现/消失、属性变化 |
| Pass 分期与解释 | [`passinfo.mjs`](../../pass-transform-explorer/lib/passinfo.mjs) | phase、推荐分析视角、Pass 文档索引 |
| 结构分析 | [`analyze.mjs`](../../pass-transform-explorer/lib/analyze.mjs) 等现有分析器 | Callable、Scope、控制流、SSA def-use、Task、Memory 派生证据 |
| 已核验的变化例子 | [`Pass Transform Explorer README`](../../pass-transform-explorer/README.md) | `qk_pv` 的关键 Pass、内存复用、拆核等交叉检查 |
| 最终 PTO MLIR | [`qk_pv.pto`](../../../Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/next_levels/decode_csa_test/ptoas/qk_pv.pto) | 真实 `pto.*` 指令、类型、memory location、源码 loc |
| PTOAS 生成结果 | [`qk_pv.cpp`](../../../Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/next_levels/decode_csa_test/ptoas/qk_pv.cpp) | `pto.*` 到 PTO-ISA C++ API 的投影验证 |
| 编译器源码与文档 | `repo/pto/` | Pass contract、Codegen 和架构语义 |

### 2.3 ISA 来源

真实指令名称不得由界面文案猜测，使用两级来源：

1. 本地 `qk_pv.pto` 中的 `pto.*` MLIR op；
2. PTO-ISA 源仓库和生成的 `qk_pv.cpp` 中的公开 API 名称。

外部权威来源：

- [PTO-ISA 源仓库](https://github.com/hw-native-sys/pto-isa)
- [PTO-ISA Tile 指令文档](https://github.com/hw-native-sys/pto-isa/blob/main/docs/isa/tile/README.md)
- [PyPTO PTOAS op status](https://github.com/hw-native-sys/pypto/blob/main/docs/en/dev/ptoas-op-status.md)
- [PyPTO Codegen](https://github.com/hw-native-sys/pypto/blob/main/docs/en/dev/codegen/00-pto_codegen.md)

第一版已确认应覆盖的实际 `qk_pv.pto` 指令族包括：

- 搬运：`pto.tload`、`pto.tstore`；
- Cube：`pto.tmatmul`、`pto.tmatmul.acc`；
- Vector：`pto.texp`、`pto.trowmax`、`pto.trowsum`、转换与逐元素算子；
- 跨核：`pto.tpush_to_aiv`、`pto.tpop_from_aic`、`pto.tfree_from_aic`，以及相反方向的配对指令。

这些名称表示静态源码事实。出现次数默认也是静态 occurrence count，不表述为 runtime 执行次数。

当前 `qk_pv.pto` 的静态抽取基线如下，后续生成器应重新计算并用它发现源数据漂移，而不是把数字写死进前端：

| MLIR op | 静态出现次数 |
| --- | ---: |
| `pto.tload` | 12 |
| `pto.tstore` | 11 |
| `pto.tmatmul` | 8 |
| `pto.tmatmul.acc` | 8 |
| `pto.texp` | 2 |
| `pto.trowmax` | 2 |
| `pto.trowsum` | 2 |
| `pto.tpush_to_aiv` / `pto.tpush_to_aic` | 2 / 2 |
| `pto.tpop_from_aiv` / `pto.tpop_from_aic` | 2 / 2 |
| `pto.tfree_from_aiv` / `pto.tfree_from_aic` | 2 / 2 |

### 2.4 事实、派生与编排必须分开

所有可视化对象和结论必须带 `provenance`：

| 值 | 含义 | 示例 |
| --- | --- | --- |
| `observed` | 直接存在于 IR、PTO 或 C++ 生成物 | `pto.tmatmul`、某个 `!pto.tile_buf<loc=acc>` |
| `derived` | 可由确定性分析器从事实源恢复 | SSA def-use、Task dependency、buffer lifetime |
| `curated` | 为叙事提炼的合并或命名 | “QK / online softmax / PV” 三段问题结构 |

`curated` 不得冒充自动抽取；每个 curated 对象仍必须列出其覆盖的 observed/derived 对象 ID。

## 3. 数据管线

```text
passes_dump/*.py ─┐
Pass docs/source ─┼─→ extract / normalize / correlate ─→ scenario JSON
qk_pv.pto ────────┤                                  ├→ schema validation
qk_pv.cpp ────────┘                                  └→ evidence report
```

### 3.1 构建步骤

1. 找到 `l3_decode_csa` 的真实 Pass 顺序及相邻快照。
2. 对每个 Pass 解析 `qk_pv` 及其拆分前后的 callable family。
3. 提取 Tensor/Tile、Op、Scope、Callable、Kernel、Task、MemRef 和 ISA 对象。
4. 以来源、语义角色和谱系建立稳定 ID，不直接把易变 SSA 名作为对象 ID。
5. 为相邻 Pass 计算可逆 delta。
6. 从最终 `qk_pv.pto` 和 `qk_pv.cpp` 提取 ISA evidence。
7. 生成 JSON，并执行 schema、引用完整性、可逆性和来源存在性检查。

### 3.2 输出目录

```text
Design/csa-pass-field/
├── data/
│   └── csa-qk-pv/
│       ├── manifest.json
│       ├── graph.json
│       ├── passes.json
│       ├── isa.json
│       └── views.json
├── schema/
│   ├── manifest.schema.json
│   ├── graph.schema.json
│   ├── passes.schema.json
│   ├── isa.schema.json
│   └── views.schema.json
├── scripts/
│   ├── build-csa-qk-pv.mjs
│   └── validate-data.mjs
└── docs/
    └── solution-a-data-and-react-flow-plan.md
```

## 4. JSON 契约

### 4.1 `manifest.json`

描述场景、数据版本、来源、生成器和检查结果，不承载图节点。

```json
{
  "schemaVersion": "csa-pass-field.manifest.v1",
  "scenarioId": "l3_decode_csa.qk_pv",
  "title": "CSA qk_pv mixed Cube/Vector handoff",
  "sourceRoots": {
    "passDumps": "Data/.../passes_dump",
    "pto": "Data/.../ptoas/qk_pv.pto",
    "cpp": "Data/.../ptoas/qk_pv.cpp"
  },
  "passCount": 51,
  "artifacts": {
    "graph": "graph.json",
    "passes": "passes.json",
    "isa": "isa.json",
    "views": "views.json"
  }
}
```

`passCount` 由构建结果写入并校验，前端不得写死。若当前 dump 抽取结果与 51 不一致，构建失败并报告真实列表，不通过删除 Pass 对齐数字。

### 4.2 `graph.json`

`graph.json` 保存跨 Pass 可追踪的语义对象注册表和初始关系，不保存 React Flow UI 状态。

```json
{
  "schemaVersion": "csa-pass-field.graph.v1",
  "scenarioId": "l3_decode_csa.qk_pv",
  "entryPassId": "frontend",
  "nodes": [
    {
      "id": "value.q",
      "semanticKind": "tensor_value",
      "renderKind": "tensor",
      "label": "Q",
      "role": "query input",
      "attributes": {
        "shape": [32, 512],
        "dtype": "bf16"
      },
      "identity": {
        "lineageRoot": "value.q",
        "aliases": []
      },
      "provenance": "curated",
      "sourceRefs": []
    }
  ],
  "edges": [
    {
      "id": "data.q-to-qk",
      "semanticKind": "data",
      "source": "value.q",
      "target": "op.qk",
      "sourcePort": "out",
      "targetPort": "lhs",
      "provenance": "curated",
      "sourceRefs": []
    }
  ]
}
```

#### 语义对象类型

不能以一个通用 `node` 类型抹平编译层级。第一版至少支持：

- `tensor_value`
- `tile_value`
- `op`
- `scope`
- `callable`
- `kernel`
- `task`
- `memref`
- `isa_group`
- `isa_instruction`

`semanticKind` 表示真实对象类型；`renderKind` 只选择 React Flow 图元。多个语义类型可以共享一种图元，但 Inspector 和关系计算必须继续识别真实类型。

#### 边类型

主 DAG 边：

- `data`
- `control`
- `task_dependency`

关系与投影边：

- `contains`
- `lowered_to`
- `memory_binding`
- `same_identity`
- `source_evidence`

只有主 DAG 边参与拓扑阅读。关系与投影边属于可切换 overlay，不应混入主布局的拓扑约束，否则整图会因“对象谱系”产生伪循环。

### 4.3 稳定 ID 与谱系

稳定 ID 是 Pass 连续性的基础：

- 不使用数组索引；
- 不直接使用会变化的 `_ssa_vN` 名称；
- 不因 renderKind 变化而改变 ID；
- 一个对象被真实拆分时才建立新 ID，并显式记录 `split_from`；
- 多对象融合时记录 `merged_from`；
- 高层 Tensor 到低层 Tile 是谱系关系，不假装为同一种对象。

```json
{
  "id": "tile.qk.acc",
  "semanticKind": "tile_value",
  "identity": {
    "lineageRoot": "value.qk_scores",
    "createdBy": "ConvertTensorToTileOps",
    "derivedFrom": ["value.qk_scores"]
  }
}
```

### 4.4 `passes.json`

不保存 51 份完整图，而保存一个基础图和 51 个可逆 delta。

```json
{
  "schemaVersion": "csa-pass-field.passes.v1",
  "passes": [
    {
      "index": 23,
      "id": "pass.expand-mixed-kernel",
      "name": "ExpandMixedKernel",
      "phase": "pipeline",
      "changed": true,
      "focusIds": ["callable.qk_pv", "kernel.qk_pv.aic", "kernel.qk_pv.aiv"],
      "contract": {
        "preconditions": [],
        "producedProperties": ["aic_aiv_kernel_family"],
        "invalidatedProperties": [],
        "verification": []
      },
      "delta": {
        "addNodes": [],
        "removeNodes": [],
        "updateNodes": [],
        "addEdges": [],
        "removeEdges": [],
        "updateEdges": [],
        "lineage": []
      },
      "evidence": []
    }
  ]
}
```

`updateNodes` 和 `updateEdges` 必须同时保存 `before` 与 `after`；`remove*` 必须保存 tombstone，因此前端可以向前、向后逐 Pass 重放，并验证回到初始状态后哈希一致。

no-op Pass 的规则：

- `changed: false`；
- 所有 delta 数组为空；
- 保留 Pass contract、来源与“为什么在此场景未命中”的证据；
- 前端不得为了动画伪造节点亮起或属性变化。

### 4.5 `isa.json`

ISA 数据需要区分三个名称层级：

```json
{
  "schemaVersion": "csa-pass-field.isa.v1",
  "instructions": [
    {
      "id": "isa.qk.matmul.0",
      "mlirName": "pto.tmatmul",
      "apiName": "TMATMUL",
      "family": "cube",
      "executionUnit": "AIC",
      "occurrence": 1,
      "operands": [],
      "results": [],
      "sourceRefs": [
        {
          "artifact": "qk_pv.pto",
          "line": 250
        }
      ],
      "provenance": "observed"
    }
  ]
}
```

- `mlirName`：`qk_pv.pto` 里的真实 `pto.*` 名称；
- `apiName`：PTO-ISA C++ API 名称，只有在生成物或官方源中确认后填写；
- `family`：为过滤与聚合而派生的类别，不替代真实名称；
- `occurrence`：静态源码 occurrence，不等于动态执行次数。

### 4.6 `views.json`

业务数据与布局规则分离。第一版只实现 `front`，但数据结构为方案 B 预留视图定义。JSON 不存 React Flow 的临时状态，而是声明节点尺寸、Dagre 参数、容器成员与外部槽位；前端据此确定性地计算几何。

```json
{
  "schemaVersion": "csa-pass-field.views.v1",
  "views": {
    "front": {
      "layout": {
        "engine": "dagre@0.8.5",
        "direction": "LR",
        "rankSep": 48,
        "nodeSep": 52,
        "edgeSep": 18
      },
      "dimensions": {
        "tensor": { "width": 160, "height": 64 },
        "op": { "width": 260, "height": 60 },
        "task": { "width": 260, "height": 60 }
      },
      "containerMembers": {
        "kernel.qk.aic": ["value.q", "op.qk", "value.qk"],
        "kernel.qk.aiv": ["op.softmax", "value.prob"]
      },
      "externalSlots": {
        "task": "north",
        "memory": "south",
        "isa": "south"
      },
      "locked": false
    }
  }
}
```

React Flow 的拖动结果只更新本地 view override；“恢复自动布局”会丢弃 override，重新按 JSON 规则求解。除非进入明确的“编辑布局”流程，否则不写回业务 JSON。

## 5. 前端架构

### 5.1 目录建议

```text
Design/csa-pass-field/
├── src/
│   ├── app/
│   │   └── CsaPassField.tsx
│   ├── domain/
│   │   ├── graph-types.ts
│   │   ├── pass-reducer.ts
│   │   ├── graph-selectors.ts
│   │   └── provenance.ts
│   ├── react-flow/
│   │   ├── graph-adapter.ts
│   │   ├── layout-controller.ts
│   │   ├── node-types.ts
│   │   ├── edge-types.ts
│   │   ├── nodes/
│   │   └── edges/
│   ├── interaction/
│   │   ├── selection-state.ts
│   │   ├── pass-navigation.ts
│   │   └── hover-preview.ts
│   └── data/
│       └── load-scenario.ts
├── data/
├── schema/
├── scripts/
└── docs/
```

### 5.2 单一事实源

领域 `GraphState` 是唯一事实源；React Flow nodes/edges 是派生视图。

```text
Scenario JSON
    ↓
PassReducer(currentPass)
    ↓
Domain GraphState
    ↓ selectors
ReactFlowAdapter
    ↓
Node[] + Edge[]
```

禁止同时维护一份领域图和一份可独立修改的 React Flow 图。否则 Pass 回退、Inspector 和未来 B 会出现状态分叉。

React Flow 只拥有：

- viewport；
- 临时 hover；
- 用户拖动产生的 view position 草稿；
- 节点测量结果。

领域 store 拥有：

- 当前 Pass；
- 节点和边的真实存在性及属性；
- selection ID；
- lineage；
- provenance；
- scope 展开状态。

第一版使用 reducer + selectors 即可，不先引入 Redux。只有性能测量证明全图更新成为瓶颈时，再增加细粒度 store。

## 6. React Flow Node 计划

### 6.1 Node Types

```ts
const nodeTypes = {
  tensor: TensorNode,
  op: OpNode,
  scope: ScopeNode,
  callable: CallableNode,
  kernel: KernelNode,
  task: TaskNode,
  memory: MemoryNode,
  isa: IsaNode,
};
```

| Node | 表达对象 | 关键视觉信息 | 默认层级 |
| --- | --- | --- | --- |
| `TensorNode` | Tensor / Tile value | 批准的 folded-value 色带、shape、dtype、memory space | 主图元 |
| `OpNode` | 高层或 Tile Op | 胶囊、op 名、执行域 | 主图元 |
| `ScopeNode` | hierarchy/incore/cluster/pipeline scope | 低对比容器、scope label | 背景容器 |
| `CallableNode` | 编译器 Callable | 签名边界、输入输出端口 | 局部展开 |
| `KernelNode` | AIC/AIV kernel | execution unit、kernel family | 中后期主图元 |
| `TaskNode` | runtime task | task identity、dependency ports | 后期局部展开 |
| `MemoryNode` | MemRef / address binding | space、size、offset、lifetime | 局部展开 |
| `IsaNode` | 单条或小组真实指令 | `pto.*`、API name、source loc | 最终投影 |

节点必须使用稳定 `id` 作为 React key。Pass 造成属性变化时更新 `data`，不能通过生成新 key 强制 remount。

### 6.2 Handle 语义

Handle 不是装饰，端口必须有领域含义：

- Tensor：`producer`、`consumer`；
- Op：`lhs`、`rhs`、`acc`、`result`、`predicate`；
- Kernel：`gm-in`、`gm-out`、`aic-aiv-push`、`aic-aiv-pop`；
- Task：`depends-on`、`unblocks`；
- ISA：`ins`、`outs`。

边的 `sourceHandle` / `targetHandle` 由 JSON port 决定，不由节点在屏幕上的上下关系猜测。

### 6.3 Scope 与包含关系

Scope 与 Kernel 使用“成员集合 + 派生边界”表达，不把复合容器交给 Dagre：

- `views.json.containerMembers` 明确声明成员，不从屏幕重叠反推层级；
- Dagre 只求解叶子节点，Scope / Kernel 边界由成员包围盒加 padding 派生；
- AIC / AIV 成员在求解后进入稳定的语义 lane，避免容器与节点堆叠；
- Scope 的 z-index 低于数据边与业务节点；
- Scope 折叠后保留边界端口和内部对象数量；
- 展开改变可见结构集合时重新求解；纯属性变化、no-op 和 Inspector 开关不触发布局。

`contains` 可存在于领域 JSON，但默认不画成普通箭头。

### 6.4 节点状态

节点外观由“对象类型”和“当前 Pass 状态”两个正交维度组成：

- 对象类型决定基本 glyph；
- `added / removed / modified / unchanged / focused` 决定局部状态；
- memory space、AIC/AIV、provenance 等属性用标签、端口或局部色块表达；
- 不用整张节点的颜色承担全部分类。

状态色只作用于当前 Pass 的局部变化。未变化上下文降低对比度，但仍保持可读；异常或不满足 contract 才使用橙红。

## 7. React Flow Edge 计划

### 7.1 Edge Types

```ts
const edgeTypes = {
  data: DataEdge,
  control: ControlEdge,
  taskDependency: TaskDependencyEdge,
  memoryBinding: MemoryBindingEdge,
  lowering: LoweringEdge,
  lineage: LineageEdge,
};
```

| Edge | 语义 | 默认形式 | 是否参与主拓扑 |
| --- | --- | --- | --- |
| `DataEdge` | producer → consumer | 实线、有向、按端口进入 | 是 |
| `ControlEdge` | predicate/region/control transfer | 细虚线、有条件标签 | 是 |
| `TaskDependencyEdge` | task dependency | 折线路由、dependency label | 是，仅 task 展开态 |
| `MemoryBindingEdge` | value/tile → MemRef/address | 细连接线、可切换 | 否 |
| `LoweringEdge` | 高层对象 → 低层对象 | 投影线、只在当前 Pass 显示 | 否 |
| `LineageEdge` | split/merge/same lineage | 短时变换 overlay | 否 |

### 7.2 Edge 实现

- 使用 React Flow Custom Edge API；
- 基础数据流优先 `getSmoothStepPath`，必要时以固定中继点避免穿过 Tensor/Scope；
- label 使用 `EdgeLabelRenderer`，只显示条件、跨核协议或关键 shape，不重复节点已有信息；
- marker 由 edge type 决定；
- 选中节点时只增强一跳上下游，其他边降噪；
- 禁用无意义的永久流动动画；Pass 预览只允许一次性、短时的方向提示；
- `prefers-reduced-motion` 下直接切换状态。

### 7.3 边与 Pass delta

当前 Pass 切换采用三段状态，而不是整图闪烁：

1. `before`：显示稳定上下文和将受影响的局部边；
2. `transition`：移除边保留短时 ghost，新边从关联端口出现；
3. `after`：清除 ghost，保留当前 Pass 影响标记。

回退 Pass 使用同一 delta 的逆操作，不维护另一套专用动画数据。

## 8. 稳定布局策略

### 8.1 Dagre 的运行边界

LLVM Flow 已证明 React Flow + Dagre 适合根据真实节点尺寸做 LR/TB 自适应布局。方案 A 复用这套节点/边几何技术，但为 Pass 回放增加稳定性约束：布局输入只取“当前可见的结构节点与主拓扑边”，二级投影和面板状态不参与求解。

方案 A 采用：

1. Dagre 以真实 node width / height、`rankdir: LR`、`rankSep`、`nodeSep` 和 `edgeSep` 计算叶子 DAG；
2. 只让数据流、控制流等主拓扑边参与 rank，memory binding 与 lowering projection 不扰动核心 DAG；
3. Kernel / Scope 根据 JSON 中的显式成员集合派生边界，再把 AIC / AIV 成员铺入语义 lane；
4. Task 自动排在 Scope 北侧，Memory 与 ISA 依据来源对象排在南侧独立 band；
5. 同一可见结构集合得到相同几何，所以 no-op 与纯属性 Pass 保持稳定；结构变化时允许重新求解以换取可读性；
6. 用户拖动只形成本地 view override，不改变业务事实；“恢复自动布局”清除 override。

### 8.2 局部展开

默认只显示提炼后的主干：

```text
Q/KV → QK → online softmax → PV → output
```

随着 Pass 推进，同一位置逐步展开：

- Tensor → Tile；
- mixed callable → AIC + AIV kernel family；
- 跨核交接 → TPUSH/TPOP；
- Tile → MemRef/address；
- Kernel → Task dependency；
- Kernel/Tile op → ISA instruction cluster。

展开只占用当前对象附近的预留空间；关闭后回到原 anchor，不把用户带到另一张图或另一页。

## 9. 交互状态

### 9.1 Pass 导航

- 前进/后退逐 Pass 可逆；
- hover Pass 只预览 focus 和摘要，不提交 GraphState；
- click Pass 持久选择并重放 delta；
- 连续 no-op 可压缩为区间，展开后每个 Pass 仍可单独选择；
- URL state 至少包含 scenario、pass、selected object 和 scope expansion。

### 9.2 对象选择

- hover：临时增强一跳邻域和当前 Pass 关系；
- click：设置唯一 `selectedObjectId`；
- Inspector 读取 domain object，不读取节点 DOM；
- 选择在 Pass 前后尽量保持；对象真实消失时显示 tombstone 和去向；
- A 与未来 B 共用同一 selection ID。

### 9.3 ISA 投影

ISA 不作为独立页面。选择 Kernel、Tile Op 或交接边时，在主 DAG 内局部展开其真实指令簇：

- 显示 `pto.*` 名称；
- 可显示已核实的 PTO-ISA API 名；
- 显示 AIC/AIV 和 memory location；
- 显示源码位置与静态 occurrence；
- 不把 instruction cluster 冒充完整 runtime trace。

## 10. 性能策略

`qk_pv` 的完整分析图可能超过 100 个节点。第一版保持“核心 DAG 默认、证据局部展开”：

- React Node 组件使用 `memo`；
- adapter 对未变化对象复用引用；
- edge path 和邻域索引按 graph revision 缓存；
- 不在 pointer move 中重建全图；
- 仅渲染当前展开 scope 的细节对象；
- 大规模 ISA occurrence 聚合为 cluster，选择后再展开单条指令；
- 以浏览器 performance profile 决定是否启用 visible-element 优化，不先凭感觉牺牲语义。

## 11. 验证计划

### 11.1 数据验证

- 所有 edge endpoint 和 port 存在；
- 所有 sourceRef 指向存在的文件/行或明确的外部 URL；
- stable ID 全局唯一；
- lineage 不存在未声明的环；
- 51 个 Pass 顺序与 source manifest 一致；
- no-op Pass 的 delta 为空；
- 从初始态正向重放，再逆向重放，最终 graph hash 一致；
- `isa.json` 的 `pto.*` occurrence 可在 `qk_pv.pto` 逐条定位；
- `apiName` 只有在 `qk_pv.cpp` 或 PTO-ISA 源中确认后出现。

### 11.2 前端单元验证

- 每个 semanticKind 映射到正确 renderKind；
- 每种 edge type 映射到正确端口和 renderer；
- Pass 切换不改变未触及节点的位置；
- 选择在属性更新时不丢失；
- split/merge 后 lineage 可追溯；
- no-op 不产生虚假视觉变化；
- scope 展开/收起不修改领域图；
- reduced motion 行为正确。

### 11.3 页面验收

- canonical desktop viewport 下 DAG 占主体；
- 页面框架为全视口无限画布，Liquid Glass 只承载悬浮控制、Pass rail 和按需二级面板；
- 默认、hover、selected、no-op、局部展开、ISA 投影均实际检查；
- 数据流、控制流、Task dependency 可视觉区分；
- Tensor、Tile、Kernel、Task、ISA 不被同一通用卡片抹平；
- no-op 与纯属性 Pass 切换时无整图跳位，结构改变时自动布局不产生遮挡；
- Inspector 默认关闭，以覆盖层按需出现且不改变画布尺寸；
- 窄屏下主图仍优先，Inspector 变为可关闭的底部抽屉；
- 控制台无错误，JSON 加载失败有明确错误态。

## 12. 实施里程碑

### M0 · 数据合同

- 建立 schema 和空数据目录；
- 确认 source manifest 与真实 Pass 顺序；
- 实现 stable ID、provenance、sourceRef 规则；
- 产出最小 `graph.json` / `passes.json` / `isa.json` / `views.json`。

验收：JSON 全部通过 schema 与引用检查，尚不要求页面。

### M1 · React Flow 骨架

- 建立最小 React Flow 页面；
- 完成 `graph-adapter`；
- 实现 Tensor、Op、Scope 三类节点；
- 实现 Data、Control 两类边；
- 从外部 JSON 加载，不在组件内硬编码业务节点。

验收：初始 CSA 核心 DAG 可交互，选择与 Inspector 数据一致。

### M2 · 关键 Pass 检查点

优先接入以下命名 Pass，序号从 manifest 解析，不在代码中硬编码：

- `ConvertTensorToTileOps`；
- `ExpandMixedKernel`；
- `InitMemRef`；
- `MemoryReuse`；
- `AutoDeriveTaskDependencies`；
- `MaterializeValidShapeSymbols`。

验收：六个检查点之间对象身份连续，前后回放可逆，无全图重排。

### M3 · 51 Pass 与 no-op

- 接入完整 Pass delta；
- 完成 phase 与 no-op 区间；
- 添加 Pass contract 和 evidence；
- 补齐局部展开规则。

验收：所有 Pass 可到达；无证据的 Pass 不生成虚假变化。

### M4 · Kernel / Task / Memory / ISA

- 完成剩余 Node/Edge types；
- 接入 `qk_pv.pto` 与 `qk_pv.cpp` evidence；
- 实现 Kernel/Tile Op 的局部 ISA 展开；
- 完成 AIC/AIV 跨核交接关系。

验收：最终视图可从核心计算追溯到真实 PTO 指令和 source loc。

### M5 · 视觉与交互验收

- 按 Spatial Systems UI 规范完成主体比例、层级、状态和窄屏；
- 运行数据检查、前端测试和实际浏览器 smoke；
- 记录已验证状态与仍属 curated 的部分。

验收：方案 A 成为可独立使用的主视图，并能在不改写数据契约的前提下进入方案 B。

## 13. 方案 B 兼容边界

未来方案 B 使用 Pangu layer-stack 的多视图/投影思路，但不复制其 UI 元素。

A 与 B 必须共享：

- `GraphState`；
- stable object ID；
- Pass delta；
- provenance 与 source refs；
- selection 与 scope expansion；
- `views.json` 中的语义坐标。

B 可以使用不同 renderer。不得为了 2.5D 表现把 React Flow 的屏幕坐标、DOM 节点或 CSS class 写入业务 JSON。

## 14. 完成定义

方案 A 只有同时满足以下条件才算完成：

1. DAG 业务数据全部来自独立 JSON，React 组件内无场景节点硬编码；
2. React Flow 仅承担节点、边、viewport 与命中交互；
3. 当前 Pass 的变化可以正向和反向重放；
4. 未变化对象在连续 Pass 间保持稳定位置和 ID；
5. 数据流、控制流、Scope、Kernel、Task、Memory、ISA 语义没有被合并；
6. PTO 指令名能够追溯到真实源文件；
7. 51 个 Pass 全部可到达，no-op 如实呈现；
8. 没有复用 `pass-transform-explorer` 的 UI 元素；
9. 未来增加 Front / Strata / Side 视图不需要重写 graph 与 passes 数据；
10. 页面经过真实浏览器中的默认、选择、局部展开、回退、no-op 和 ISA 投影检查。
