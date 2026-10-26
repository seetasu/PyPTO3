const fs = require('node:fs');
const path = require('node:path');

const root = __dirname;
const entries = [
  {
    file: 'PyPTO基础_从PythonDSL到设备执行.md',
    title: 'PyPTO 基础：从 Python DSL 到设备执行',
    shortTitle: 'PyPTO 基础',
    lead: '从 Python 表达、Tile 数据流与编译转换，一路理解 PyPTO 程序如何成为运行时任务与设备执行。',
    tags: ['编程模型', '编译与执行', '内存层级'],
    visuals: [
      { rows: [[{label:'Python DSL',tone:'lime'},{label:'IR / Scope'},{label:'Kernel',tone:'violet'}],[{label:'Task / Block',tone:'amber'},{label:'设备执行',tone:'blue'}]], caption:'源码意图经过编译和运行时映射，形成实际设备工作。', note:'阅读提示：scope、kernel、task 与 block 属于不同阶段的对象。' },
      { rows: [[{label:'Tensor / DDR',tone:'blue'},{label:'Tile.load'},{label:'Vec / Mat',tone:'lime'}],[{label:'Left / Right'},{label:'Matmul / Acc',tone:'violet'},{label:'Tile.store'}]], caption:'Tile 数据流与片上内存空间共同决定算子的数据移动。', note:'阅读提示：逻辑视图和真实搬运需要结合编译后表示确认。' },
      { rows: [[{label:'scope'},{label:'Function',tone:'lime'},{label:'Kernel',tone:'violet'}],[{label:'Submit'},{label:'Task',tone:'amber'},{label:'Block'}]], caption:'上层静态结构与运行时动态对象并非一一对应。', note:'阅读提示：用名称映射和 trace 保留对象血缘。' },
      { rows: [[{label:'固定环境'},{label:'编译正确',tone:'lime'},{label:'数值验证'}],[{label:'依赖 / 时间线'},{label:'关键路径',tone:'amber'},{label:'可复现收益'}]], caption:'从规格到测量建立一条可核查的工程证据链。', note:'阅读提示：性能报告必须说明适用条件与计时范围。' }
    ]
  },
  {
    file: '模型推理基础_从请求阶段到性能指标.md',
    title: '模型推理基础：从请求阶段到性能指标',
    shortTitle: '模型推理',
    lead: '从请求进入服务，到 Prefill、Decode 与 KV Cache，再到 TTFT、TPOT 和端到端性能归因。',
    tags: ['请求生命周期', '性能指标', 'Serving 与设备'],
    visuals: [
      { rows: [[{label:'接入 / 排队',tone:'amber'},{label:'调度 / 批处理'}],[{label:'Prefill',tone:'lime'},{label:'Decode 循环',tone:'violet'},{label:'Token 返回',tone:'blue'}],[{label:'KV Cache 读写',tone:'blue'}]], caption:'生成请求跨服务调度、模型计算与状态缓存多个阶段。', note:'阅读提示：服务路径和模型计算路径要用一致的请求标识关联。' },
      { rows: [[{label:'输入上下文'},{label:'Prefill',tone:'lime'},{label:'KV 状态',tone:'blue'}],[{label:'KV Cache'},{label:'Decode step',tone:'violet'},{label:'下一个 Token',tone:'amber'}]], caption:'Prefill 建立上下文状态，Decode 逐步消费并更新它。', note:'阅读提示：KV Cache 用容量换取计算复用，缓存策略存在权衡。' },
      { rows: [[{label:'TTFT',tone:'lime'},{label:'TPOT',tone:'violet'},{label:'吞吐',tone:'blue'}],[{label:'P95 / P99',tone:'amber'},{label:'质量 / 稳定性'}]], caption:'指标描述不同用户体验与服务目标，需要明确起止边界。', note:'阅读提示：同名指标可能有不同采样、窗口和统计定义。' },
      { rows: [[{label:'Serving / Queue'},{label:'Host 编排',tone:'amber'},{label:'Device kernel',tone:'lime'}],[{label:'请求分布'},{label:'关键路径',tone:'violet'},{label:'端到端 SLO',tone:'blue'}]], caption:'性能瓶颈可能跨服务、Host、设备与模型策略多个层面。', note:'阅读提示：单算子提速不自动意味着端到端体验改善。' }
    ]
  },
  {
    file: '算子开发基础_从数学规格到可验证Kernel.md',
    title: '算子开发基础：从数学规格到可验证 Kernel',
    shortTitle: '算子开发',
    lead: '把数学定义变成可编译、可验证、可测量的设备 Kernel：从接口约束到分块、内存、同步与性能证据。',
    tags: ['规格与约束', 'Tile 与内存', '正确性与调优'],
    visuals: [
      { rows: [[{label:'数学定义',tone:'lime'},{label:'输入输出契约'},{label:'shape / dtype'}],[{label:'边界与精度',tone:'amber'},{label:'目标工作负载',tone:'blue'}]], caption:'完整规格把算法语义、边界行为和验收目标放在一起。', note:'阅读提示：先写清规格，才有可靠的参考实现和测试门槛。' },
      { rows: [[{label:'Tensor / DDR',tone:'blue'},{label:'Tile.load'},{label:'片上缓冲',tone:'lime'}],[{label:'分块 / 布局'},{label:'计算 / 累加',tone:'violet'},{label:'Tile.store'}]], caption:'Tile 规划要兼顾数据复用、容量、并行度和尾块。', note:'阅读提示：更大的 Tile 未必更快，也可能降低可用并发。' },
      { rows: [[{label:'SPMD 分片',tone:'lime'},{label:'依赖与同步',tone:'amber'}],[{label:'AIC / AIV',tone:'violet'},{label:'Pipeline',tone:'blue'},{label:'Buffer 生命周期'}]], caption:'并行与流水带来重叠机会，也带来依赖和资源约束。', note:'阅读提示：每条依赖边都应对应可解释的读写或同步关系。' },
      { rows: [[{label:'编译门禁'},{label:'参考值 / 精度',tone:'lime'},{label:'边界与并发'}],[{label:'干净基线'},{label:'A/B 归因',tone:'amber'},{label:'适用范围'}]], caption:'正确性和性能必须在同一组可追溯条件下验收。', note:'阅读提示：优化收益需要精度、资源和回归数据共同支持。' }
    ]
  },
  {
    file: 'Hardware-Native基础_让硬件事实进入编程模型.md',
    title: 'Hardware-Native System 基础：让硬件事实进入编程模型',
    shortTitle: 'Hardware-Native',
    lead: '从存储、计算、布局、同步到多卡通信，理解硬件事实如何进入表达、查询、校验和观测。',
    tags: ['硬件事实域', '可表达与可查询', '可校验与可观测'],
    visuals: [
      { rows: [[{label:'硬件事实域',tone:'blue'},{label:'表达 / 查询',tone:'lime'},{label:'验证与观察',tone:'violet'}],[{label:'Tile 与布局'},{label:'存储 / 同步',tone:'amber'},{label:'Kernel / Task'}]], caption:'关键硬件事实要有开发者可以使用和核对的承载形态。', note:'阅读提示：仓库规划总结了九个事实域和六种承载方式。' },
      { rows: [[{label:'E · 可表达',tone:'lime'},{label:'Q · 可查询',tone:'blue'}],[{label:'C · 可校验',tone:'amber'},{label:'A · 可归因',tone:'violet'}],[{label:'O · 可观测',tone:'blue'},{label:'R · 可复用',tone:'lime'}]], caption:'同一条硬件事实可能需要多种承载形态。', note:'阅读提示：编译成功只说明已执行的检查通过，并非完整硬件语义证明。' },
      { rows: [[{label:'Tensor / 规格',tone:'lime'},{label:'Tile / 布局',tone:'blue'},{label:'Cube / 累加',tone:'violet'}],[{label:'尾块 / mask',tone:'amber'},{label:'IR / Trace',tone:'blue'}]], caption:'矩阵乘把数据布局、访存、计算与边界验证连接起来。', note:'阅读提示：示意图不能替代目标后端文档和实际编译产物。' },
      { rows: [[{label:'发现异常',tone:'amber'},{label:'确定首次偏离',tone:'blue'}],[{label:'映射到硬件事实',tone:'violet'},{label:'修复并复验',tone:'lime'}]], caption:'把症状转换为可复核证据，再做针对性实验。', note:'阅读提示：IR/Trace 是证据，观察到差异本身不等于确认根因。' }
    ]
  },
  {
    file: 'PyPTO-Pass流水线详解_从Tensor到设备任务.md',
    title: 'PyPTO Pass 流水线详解：从 Tensor 到设备任务',
    shortTitle: 'Pass 流水线',
    lead: '把规范化、Scope 提取、Tensor 到 Tile、设备合法化、内存分配与 Runtime 建图串成可观察的编译旅程。',
    tags: ['IR 变换', 'Pass 属性', 'Dump 与诊断'],
    visuals: [
      { rows: [[{label:'Python DSL',tone:'lime'},{label:'规范 IR',tone:'violet'},{label:'Tile / 设备 IR',tone:'blue'}],[{label:'存储 / 执行结构',tone:'amber'},{label:'Kernel / Task',tone:'lime'}]], caption:'Pass 以有序变换把高层程序逐步变成可执行设备任务。', note:'阅读提示：静态编译对象与动态运行时实例并非一一对应。' },
      { rows: [[{label:'控制流 / SSA',tone:'lime'},{label:'Scope 拆分',tone:'violet'}],[{label:'Tensor → Tile',tone:'blue'},{label:'设备合法化',tone:'amber'}],[{label:'流水 / MemRef',tone:'violet'},{label:'Task 依赖',tone:'blue'}]], caption:'六类职责帮助定位阶段目标，不代表流水线仅含六个 Pass。', note:'阅读提示：每次实际顺序以目标版本的 PassManager 和运行 dump 为准。' },
      { rows: [[{label:'required',tone:'blue'},{label:'Pass 变换',tone:'violet'}],[{label:'produced',tone:'lime'},{label:'invalidated',tone:'amber'}]], caption:'Pass 属性声明提供前置条件、保证产出和失效信息。', note:'阅读提示：属性只有在对应验证器注册且实际执行时才会被检查。' },
      { rows: [[{label:'MatMul Tensor',tone:'lime'},{label:'InCore / SSA',tone:'violet'},{label:'Tile MatMul',tone:'blue'}],[{label:'布局 / 流水',tone:'amber'},{label:'MemRef / Task',tone:'lime'}]], caption:'沿同一个算子追踪阶段 IR，能把结构变化和故障症状关联起来。', note:'阅读提示：图中是概念路径，不是固定 Pass 名单或实测代码。' }
    ]
  }
];

const papers = entries.map(entry => ({ ...entry, markdown: fs.readFileSync(path.join(root, entry.file), 'utf8') }));
fs.writeFileSync(path.join(root, 'reader-data.js'), `window.PYPTO_WHITEPAPERS = ${JSON.stringify(papers)};\n`, 'utf8');
console.log(`Generated reader-data.js for ${papers.length} whitepapers.`);
