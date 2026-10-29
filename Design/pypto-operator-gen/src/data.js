/**
 * PyPTO 算子生成 Demo · 内容数据
 *
 * 对象：DeepSeek-V4 CSA decode 层（一个 TP rank），
 *       Data/DeepseekV4/deepseek_v4_flash_dspark/decode_csa.py
 *
 * 做法是反向推导：拿一份已经落地的实现，把它钉死的每个决策推回设计空间——
 * 这个数字当初还有什么选法、为什么没选。所有"被拒绝的选项"都出自源码里的
 * 契约断言、注释里写明的理由，或已知缺陷的规避说明，不是编的。
 *
 * 事实来源：
 *  - 编排与分支：Data/DeepseekV4/deepseek_v4_flash_dspark/decode_csa.py
 *  - 模型与部署常量：同目录 config.py（FLASH 配置）
 *  - tile / worker 常量与理由：decode_sparse_attn_csa.py、decode_o_proj.py
 *  - 整除契约：decode_sparse_attn_csa.py、decode_o_proj.py 的 raise ValueError
 *  - 片上容量：repo/pypto/include/pypto/backend/910B/backend_910b_handler.h
 *  - 内存可达图与 Vec 184 KB：repo/pypto/src/backend/common/soc.cpp
 *
 * 候选谱系里的"候选数量"是为把收敛过程画出来而构造的量级示意；
 * 三个候选本身（TP = 1 / 2 / 4）是源码里真实支持的三种世界。
 */

window.PyptoGenData = (() => {
  /* ── 片上预算（Ascend910B，真实值，byte） ──────────────────────── */
  const CAPACITY = {
    Left: 64 * 1024,   // L0A
    Right: 64 * 1024,  // L0B
    Acc: 128 * 1024,   // L0C
    Mat: 512 * 1024,   // L1
    Vec: 184 * 1024    // UB 可用（物理 192 KB，PTO-ISA 保留约 8 KB）
  };

  // A2/A3 核数：源码按它来定 worker 数
  const CORES = { aic: 24, aiv: 48 };

  /* ── FLASH 配置派生量（TP = 2） ────────────────────────────────── */
  const CFG = {
    D: 4096,            // hidden_size
    H: 64,              // num_attention_heads
    HEAD_DIM: 512,      // MLA value-head dim
    ROPE: 64,           // qk_rope_head_dim
    NOPE: 448,          // head_dim - qk_rope_head_dim
    Q_LORA: 1024,
    O_LORA: 1024,
    O_GROUPS: 8,
    WIN: 128,           // sliding_window
    IDX_TOPK: 512,
    IDX_HEAD_DIM: 128,
    HC_MULT: 4,
    BLOCK_SIZE: 32,
    COMPRESS_RATIO: 4,
    DECODE_BATCH: 64,
    DECODE_SEQ: 8,      // 1 + 7 个 DSpark 投机 token
    TP: 2,
    H_TILE: 16,
    QK_M_TILE: 32,
    ATTN_K_TILE: 128,
    SPARSE_BLOCKS: 5,   // max(2, ceil((128 + 512) / 128))
    T: 256,             // (64 / 2) * 8
    KV_T: 512           // all-gather 之后的全组 token 流
  };

  /* ── 左栏 API / 结构说明 ────────────────────────────────────────── */
  const apiNotes = {
    inline: {
      title: '@pl.jit.inline(auto_scope=False)',
      copy: 'decode_csa 本体。inline 让它被调用方展开，auto_scope=False 表示作用域由作者显式写出——整层十几个子内核的依赖边都要自己排，编译器不替你推。'
    },
    host: {
      title: '@pl.jit.host',
      copy: 'l3_decode_csa。在 HOST 层为 TP 组里每个 rank 分派一次 decode_csa_test（device=rank），并用 pld.alloc_window_buffer 开出 6 块跨卡窗口。'
    },
    dynamic: {
      title: 'pl.dynamic · 动态轴',
      copy: '7 个动态符号：T_DYN（本 rank token）、KV_T_DYN（全组 token）、B_DYN / KV_B_DYN（请求轴）以及四个 paged cache 的块数。本 rank 行数与全组行数是两个不同符号，混用会让编译器无法证明相等。'
    },
    scope: {
      title: 'pl.scope() / pl.at(level=...)',
      copy: 'scope 划出一段作用域，at(level=pl.Level.CORE_GROUP) 把一段固定到核组层级并返回 task id。decode_csa 用 task id 显式串起 rope → KV → cache 写回 → 压缩器 → indexer → 稀疏注意力的依赖链。'
    },
    spmd: {
      title: 'pl.spmd(n, name_hint=..., deps=[...])',
      copy: '开 n 路 SPMD 任务，用 pl.tile.get_block_idx() 取本路编号。csa_cache_writeback 开 kv_dim/8 路，csa_merge_pack_publish 开 48 路——正好是 A2/A3 的 AIV 数。'
    },
    pipeline: {
      title: 'pl.pipeline(start, stop, stage=2)',
      copy: '带软件流水的循环。merge 阶段用它扫 5 个稀疏块做在线 softmax 合并，stage=2 让下一块的读取与当前块的合并重叠。'
    },
    put: {
      title: 'pld.tensor.put',
      copy: '单边写到对端的窗口缓冲：指定 peer、dst_offsets、src_offsets 与 chunk 形状。publish 阶段用它把本 rank 算出的注意力分组直接写进目标 rank 的 attention_window。'
    },
    notify: {
      title: 'pld.system.notify',
      copy: '向对端信号量发通知，op=pld.NotifyOp.AtomicAdd。put 不自带完成语义，写完要显式 notify，对端据此知道哪一份数据已经到齐。'
    },
    taskdummy: {
      title: 'pl.system.task_dummy(deps=[...])',
      copy: '把多个 task id 并成一个依赖句柄。cache_ready_dep 就是把原始 KV 写回和压缩 KV 写回两条任务合成一个，交给稀疏注意力等。'
    },
    createtensor: {
      title: 'pl.create_tensor',
      copy: '按动态维度开中间张量。注意 decode_csa 里有两处注释特地说明：子内核原地写入的张量要保留原句柄，因为 inline 返回的句柄跨不进下一个调用。'
    }
  };

  /* ── 候选谱系：从成品倒推出的设计空间，再收敛回这份实现 ────────── */
  const lineage = {
    title: '候选谱系',
    stages: [
      {
        id: 'tile-space',
        label: 'TILE SPACE',
        kind: 'partial plan',
        count: 144,
        summary: '144 个切分方案：把 decode_csa 链路上 6 个 tile 常量（H_TILE、QK_M_TILE、ATTN_K_TILE、PROJ_B_MM 的 T/N、ATTENTION_PUBLISH_T_TILE、BIAS_T_TILE）的可选取值做组合。',
        next: {
          id: 'bind-dispatch',
          type: 'diverge',
          label: 'EXPAND ×2',
          summary: '同一份切分，可以按 Cube 侧 24 个 AIC 分派，也可以按 Vector 侧 48 个 AIV 分派。整层既有 QK/PV 这种 Cube 重的段，也有 merge / 量化 / 反量化这种 Vector 重的段，两套分派都要展开试。',
          branches: [
            { label: 'AIC 24 路分派 · 144', count: 144, tone: 1 },
            { label: 'AIV 48 路分派 · 144', count: 144, tone: 2 }
          ]
        }
      },
      {
        id: 'work-plan',
        label: 'WORK PLAN',
        kind: 'realization',
        count: 288,
        segments: [144, 144],
        summary: '288 个完整工作计划：切分 × 分派。此时还没检查这些数字满不满足源码里那一串整除契约。',
        next: {
          id: 'divisibility',
          type: 'converge',
          label: 'REJECT −168',
          summary: '整除契约：源码里 11 条 raise ValueError 加上 L0 的 16 行分形下限，在这一步一次性判死。',
          rejected: [
            { label: 'TP 与分组整除 · 64', count: 64 },
            { label: 'tile 与维度整除 · 58', count: 58 },
            { label: 'L0 16 行分形下限 · 46', count: 46 }
          ]
        }
      },
      {
        id: 'contract-legal',
        label: 'CONTRACT LEGAL',
        kind: 'realization',
        count: 120,
        summary: '120 个工作计划通过全部整除契约：token 容量、输出分组、head tile、publish tile 都能被整齐切开。',
        next: {
          id: 'onchip-budget',
          type: 'converge',
          label: 'REJECT −72',
          summary: '片上预算：按 tile 形状算出的累加器与向量工作集，必须同时放进 L0C 128 KB 和 Vec 184 KB。',
          rejected: [
            { label: 'L0C 128 KB 放不下 INT32 累加器 · 30', count: 30 },
            { label: 'Vec 184 KB 放不下 bias / topk 向量 · 28', count: 28 },
            { label: 'Mat 512 KB 放不下 KV tile · 14', count: 14 }
          ]
        }
      },
      {
        id: 'buffer-fit',
        label: 'BUFFER FIT',
        kind: 'realization',
        count: 48,
        summary: '48 个工作计划的片上驻留在预算之内，可以开始排跨卡通信。',
        next: {
          id: 'comm-shape',
          type: 'diverge',
          label: 'EXPAND ×2',
          summary: '跨卡部分有三种排法：单边 put + notify 的窗口缓冲、两段式 all-to-all、或者干脆把整条 token 流 all-gather 到每张卡各算各的。',
          branches: [
            { label: 'put + notify 窗口 · 36', count: 36, tone: 1 },
            { label: '两段 all-to-all · 36', count: 36, tone: 2 },
            { label: '全量 all-gather · 24', count: 24, tone: 3 }
          ]
        }
      },
      {
        id: 'comm-plan',
        label: 'COMM PLAN',
        kind: 'executable plan',
        count: 96,
        segments: [36, 36, 24],
        summary: '96 个完整方案，带着各自的窗口布局、信号量与依赖边，等待编译器验收。',
        next: {
          id: 'compiler-reality',
          type: 'converge',
          label: 'REJECT −84',
          summary: '编译器现实：inline 句柄的跨界规则、动态符号的可证相等、以及一条已知的稀疏块误编译，在这一步把大半方案筛掉。',
          rejected: [
            { label: 'inline 句柄跨不过调用边界 · 26', count: 26 },
            { label: '动态符号不可证相等 · 24', count: 24 },
            { label: '单块 sparse-K 误编译 · 20', count: 20 },
            { label: 'TP 形状 import 时已冻结 · 14', count: 14 }
          ]
        }
      },
      {
        id: 'pto-emitted',
        label: 'PTO EMITTED',
        kind: 'verified plan',
        count: 12,
        summary: '12 个方案编得出来、跑得通，精度也对得上。',
        next: {
          id: 'rank',
          type: 'converge',
          label: 'RANK −9',
          summary: '排序：这一步不判对错，只看核利用率与通信代价。源码注释里留下的三条调优结论正是在这里起作用。',
          rejected: [
            { label: 'worker 开太少，核闲置 · 4', count: 4 },
            { label: 'worker 开太多，只烧核 · 3', count: 3 },
            { label: 'D 切太粗，饿着 AIC · 2', count: 2 }
          ]
        }
      },
      {
        id: 'shortlist',
        label: 'SHORTLIST',
        kind: 'decision',
        count: 3,
        summary: '3 个候选就是源码真实支持的三个 TP 世界：_TP_CHOICES = (1, 2, 4)，默认 _TP_DEFAULT = 2。'
      }
    ]
  };

  /* ── 阶段 / 转换详情 ────────────────────────────────────────────── */
  const traceViews = {
    'tile-space': {
      paneMeta: 'Stage 01 · Tile space',
      status: 'Reverse trace · 切分取值空间',
      title: '先把成品里钉死的数字松开',
      listTitle: '从源码里反推出的 6 个切分点',
      list: [
        ['H_TILE = 16', '每次 QK/PV 处理的 head 数'],
        ['QK_M_TILE = 32', 'QK/PV 的 M 行数，决定 KV 的 L1→L0 复用倍数'],
        ['ATTN_K_TILE = 128', '稀疏注意力的 K 块宽度'],
        ['PROJ_B_MM 128 × 256', 'O 投影第二段的 INT32 累加窗口'],
        ['ATTENTION_PUBLISH_T_TILE = 8', 'publish 阶段每次打包的 token 数'],
        ['BIAS_T_TILE = min(T, 8)', 'indexer bias / slot 向量的 token 块']
      ],
      outputTitle: '144 是怎么来的',
      outputStrong: '6 个切分点各 2–3 个合理取值的组合',
      outputCopy: '这一步只把数字松开，不判对错。下面的"决策点矩阵"按行列出每个切分点的候选取值，以及它们各自的片上占用。',
      outputList: [
        ['契约检查', '未执行'],
        ['容量精算', '未执行'],
        ['通信编排', '未绑定']
      ],
      footerStrong: '反向推导的起点',
      footerCopy: '成品里每个常量都是一次选择的结果；先把选择面还原出来，才谈得上解释。',
      matrix: true
    },
    'bind-dispatch': {
      paneMeta: 'Transition · Expand ×2',
      status: 'Reverse trace · 绑定分派单元',
      title: '一层里同时有 Cube 重和 Vector 重的段',
      listTitle: 'decode_csa 的段落分布',
      list: [
        ['Cube 重', 'q_proj / kv_proj / QK / PV / O 投影两段'],
        ['Vector 重', 'rope 交织、rms_norm、在线 softmax 合并、INT8 量化与反量化'],
        ['搬运重', 'KV cache 写回、压缩器状态页、跨卡 put'],
        ['AIC 数', `${CORES.aic}（A2/A3）`],
        ['AIV 数', `${CORES.aiv}（A2/A3）`]
      ],
      outputTitle: '因此派生两套分派',
      outputStrong: '按 24 路 AIC 分派，或按 48 路 AIV 分派',
      outputCopy: '最终实现两套都用了：QK/PV 按 NUM_QK_CORES = 24 分派，publish 与 reduce 按 48 分派。但在枚举阶段，哪一段归哪一侧还没定。',
      outputList: [
        ['扩张倍数', '×2'],
        ['数量', '144 → 288'],
        ['合法性检查', '未执行']
      ],
      footerStrong: '空间扩张',
      footerCopy: '源码里 24 和 48 这两个数字反复出现，正是因为它们是这块芯片的两种核数。'
    },
    'work-plan': {
      paneMeta: 'Stage 02 · Work plan',
      status: 'Reverse trace · 工作计划待校验',
      title: '288 个完整工作计划',
      listTitle: '每个计划已确定的内容',
      list: [
        ['切分', '6 个 tile 常量取值'],
        ['分派', '每段归 AIC 还是 AIV，开几路'],
        ['token 容量', 'T = DECODE_BATCH / TP × DECODE_SEQ'],
        ['输出分组', 'LOCAL_O_GROUPS = O_GROUPS / TP'],
        ['尚未检查', '整除 / 容量 / 通信 / 编译']
      ],
      outputTitle: '下一步要回答的问题',
      outputStrong: '这些数字能把维度整齐切开吗',
      outputCopy: '源码在 import 期就用 11 条 raise ValueError 把这些整除关系钉死——不满足的组合连模块都加载不起来，根本到不了编译。',
      outputList: [
        ['判据', 'decode_sparse_attn_csa.py / decode_o_proj.py 的断言'],
        ['性质', '硬契约'],
        ['可否用性能抵消', '否']
      ],
      footerStrong: '事实层',
      footerCopy: '这些断言本身就是反向推导最好的材料：每一条都对应一类当初被排除的取值。'
    },
    divisibility: {
      paneMeta: 'Transition · Reject −168',
      status: 'Reverse trace · 整除契约',
      title: '整除契约：168 个组合连模块都加载不起来',
      browser: 'divisibility'
    },
    'contract-legal': {
      paneMeta: 'Stage 03 · Contract legal',
      status: 'Reverse trace · 契约合法',
      title: '120 个工作计划切得整齐',
      listTitle: '已满足的契约',
      list: [
        ['O_GROUPS % TP == 0', '8 / 2 = 4 个本地输出分组'],
        ['H_TILE % HEADS_PER_GROUP == 0', '16 / 8 = 2 个 publish 分组'],
        ['T % ATTENTION_PUBLISH_T_TILE == 0', '256 / 8 = 32 个 token 块'],
        ['T % BIAS_T_TILE == 0', '256 / 8 = 32 个 bias 块'],
        ['T_PAD = ceil(T / 16) × 16', '256，对齐到 Cube 的 16 行 M 下限']
      ],
      outputTitle: '下一步要回答的问题',
      outputStrong: '这些 tile 放得进片上吗',
      outputCopy: '切得整齐不代表放得下。INT32 累加器压 L0C，bias 与 topk 向量压 Vec，KV tile 压 Mat——三个预算要同时成立。',
      outputList: [
        ['L0C', '128 KB'],
        ['Vec', '184 KB 可用'],
        ['Mat', '512 KB']
      ],
      footerStrong: '契约层结束',
      footerCopy: '接下来的判断从"切不切得开"转到"放不放得下"。'
    },
    'onchip-budget': {
      paneMeta: 'Transition · Reject −72',
      status: 'Reverse trace · 片上预算',
      title: '片上预算：72 个计划放不下',
      browser: 'onchip-budget'
    },
    'buffer-fit': {
      paneMeta: 'Stage 04 · Buffer fit',
      status: 'Reverse trace · 片上驻留可行',
      title: '48 个计划的片上驻留在预算内',
      listTitle: '最终实现里的实际占用',
      list: [
        ['PROJ_B INT32 累加器', '128 × 256 × 4 B = 128 KB，L0C 正好打满'],
        ['indexer bias 块', '8 × 512 × 4 B = 16 KB'],
        ['稀疏 KV tile', '128 × 512 × 2 B = 128 KB'],
        ['L0 M 行下限', '16，T_PAD 因此取 256'],
        ['压缩器状态页', `C4A_COMPRESSOR_BLOCK_SIZE = 2 行 / 页`]
      ],
      outputTitle: '交给通信编排的内容',
      outputStrong: '48 个可行驻留',
      outputCopy: 'decode_csa 是一个 TP rank 的实现，整层有三处跨卡：token all-gather、注意力分组 all-to-all、O 投影 reduce-scatter。怎么排是下一步。',
      outputList: [
        ['跨卡点', '3 处'],
        ['窗口缓冲', '6 块（3 数据 + 3 信号）'],
        ['同步原语', 'pld.tensor.put + pld.system.notify']
      ],
      footerStrong: '容量层结束',
      footerCopy: '源码注释里那句「Keep the INT32 proj-b accumulator within the A2/A3 tile buffer」，说的就是这一层。'
    },
    'comm-shape': {
      paneMeta: 'Transition · Expand ×2',
      status: 'Reverse trace · 通信编排',
      title: '同一份驻留，三种跨卡排法',
      listTitle: '三条分支',
      list: [
        ['put + notify 窗口', '单边写进对端窗口缓冲，写完发原子加信号'],
        ['两段 all-to-all', '先按分组交换，再按 token 交换'],
        ['全量 all-gather', '每张卡拿到全部 token 各算各的，不做分组交换'],
        ['窗口容量', 'ATTENTION_WINDOW_ROWS = LOCAL_O_GROUPS × GROUP_T_PAD'],
        ['信号布局', '[TP_SIZE, 1] INT32，每个 peer 一格']
      ],
      outputTitle: '96 个方案',
      outputStrong: '36 put/notify + 36 两段 a2a + 24 全量 all-gather',
      outputCopy: '最终实现选了 put + notify：publish 阶段算完一个分组就直接写到目标 rank，不等整层算完再统一交换。代价是依赖边要自己排——auto_scope=False 正是为此。',
      outputList: [
        ['数量', '48 → 96'],
        ['合法性检查', '未执行'],
        ['排序', '未执行']
      ],
      footerStrong: '空间扩张',
      footerCopy: '通信排法决定了依赖图的形状，也决定了哪些编译器限制会被碰到。'
    },
    'comm-plan': {
      paneMeta: 'Stage 05 · Comm plan',
      status: 'Reverse trace · 方案待编译验收',
      title: '96 个方案等待编译器验收',
      listTitle: '每个方案已确定的内容',
      list: [
        ['跨卡原语', 'put / notify 或集合通信'],
        ['窗口布局', '行数与偏移算法'],
        ['依赖边', 'task_dummy 串起的任务图'],
        ['张量句柄', '哪些子内核原地写、哪些返回新句柄'],
        ['动态符号', '本 rank 轴与全组轴怎么绑']
      ],
      outputTitle: '下一步要回答的问题',
      outputStrong: '编译器认不认',
      outputCopy: '这一步筛掉的不是"算得慢"，而是"编不出来"或"编出来是错的"。decode_csa 的注释里有四处专门记录了这类坑。',
      outputList: [
        ['判据', 'inline 句柄规则 / 动态符号相等性 / 已知缺陷'],
        ['典型现象', '跨调用句柄失效、符号不可证相等、跨 token 输出串味'],
        ['结果', '96 → 12']
      ],
      footerStrong: '编译层',
      footerCopy: '整条链路里淘汰率最高的一步，也是注释写得最密的一步。'
    },
    'compiler-reality': {
      paneMeta: 'Transition · Reject −84',
      status: 'Reverse trace · 编译器现实',
      title: '编译器现实：84 个方案编不出来或编错',
      browser: 'compiler-reality'
    },
    'pto-emitted': {
      paneMeta: 'Stage 06 · PTO emitted',
      status: 'Reverse trace · 已发出 PTO',
      title: '12 个方案编得出、跑得通、精度对',
      listTitle: '产物',
      list: [
        ['passes_dump/', '逐 pass 的 IR 快照'],
        ['ptoas/*.pto', '每个 InCore 函数一份 MLIR'],
        ['ptoas/*.cpp', 'ptoas 产出的设备代码'],
        ['kernels/', '编译好的设备 kernel'],
        ['report/perf_hints.log', '编译期性能提示']
      ],
      outputTitle: '下一步不再判对错',
      outputStrong: '12 → 3 是排序',
      outputCopy: '剩下的差别只在核利用率与通信代价。源码在三个 worker 常量旁边留下的注释，正是这一轮排序的结论。',
      outputList: [
        ['判据', 'AIC / AIV 利用率、fabric 带宽'],
        ['证据', '源码注释里的调优结论'],
        ['结果', '3 个候选']
      ],
      footerStrong: '验收层结束',
      footerCopy: '能编出来是硬门槛；之后的取舍写在注释里。'
    },
    rank: {
      paneMeta: 'Transition · Rank −9',
      status: 'Reverse trace · 排序',
      title: '排序：9 个可执行方案没有进入对比',
      browser: 'rank'
    },
    shortlist: {
      paneMeta: 'Candidates · TP 世界',
      status: 'Reverse trace · decode_csa.py · _TP_CHOICES = (1, 2, 4)',
      title: '三个候选',
      candidates: true
    }
  };

  /* ── 拒绝浏览器：全部出自源码 ──────────────────────────────────── */
  const rejectionBrowsers = {
    divisibility: {
      title: '整除契约拒绝',
      copy: '源码在 import 期就用 raise ValueError 把这些关系钉死。不满足的取值连模块都加载不起来，根本到不了编译。',
      status: '168 / 288 rejected',
      defaultCategory: 'tp-groups',
      categories: [
        { id: 'tp-groups', label: 'TP 与分组整除', hint: 'O_GROUPS / TP / token', count: 64 },
        { id: 'tile-dim', label: 'tile 与维度整除', hint: 'K / N / token tile', count: 58 },
        { id: 'fractal', label: 'L0 16 行分形下限', hint: 'Cube 的 M 对齐', count: 46 }
      ],
      cases: {
        'tp-groups': [
          {
            id: 'TP=3',
            title: 'TP 取 3',
            rule: 'raise ValueError(f"output groups {O_GROUPS} must be divisible by TP size {TP_SIZE}")',
            copy: 'O_GROUPS = 8 是模型侧固定的分组输出投影宽度。TP = 3 时 8 / 3 不整除，LOCAL_O_GROUPS 没有整数解，注意力分组无法均匀分到各 rank。源码把可选 TP 直接限定成 (1, 2, 4)。',
            reasons: [
              'LOCAL_O_GROUPS = O_GROUPS // TP 必须是整数。',
              'DECODE_TOKENS 也要能被 TP 整除，否则 token 流切不齐。',
              '这两条都写成 import 期断言，不是运行期检查。'
            ]
          },
          {
            id: 'HT=24',
            title: 'H_TILE 取 24',
            rule: 'raise ValueError(f"CSA head tile {H_TILE} must contain complete output groups")',
            copy: 'HEADS_PER_GROUP = H / O_GROUPS = 8。H_TILE = 24 虽然能整除 64 个 head，但 24 / 8 = 3 个分组——而 publish 阶段按 PUBLISH_GROUPS = H_TILE / HEADS_PER_GROUP 整块发送，分组数必须能被 LOCAL_O_GROUPS 容纳。',
            reasons: [
              'H_TILE 必须包含完整的输出分组，否则一个 head tile 会跨两个目标 rank。',
              'LOCAL_O_GROUPS % PUBLISH_GROUPS == 0 是另一条独立断言。',
              'H_TILE = 16 给出 PUBLISH_GROUPS = 2，正好整除 LOCAL_O_GROUPS = 4。'
            ]
          },
          {
            id: 'T!=LOCAL_T',
            title: 'CSA token 容量与 TP 本地容量不一致',
            rule: 'raise ValueError(f"CSA token capacity {T} must equal TP local token capacity {LOCAL_T}")',
            copy: 'decode_csa.py 在 import 期直接比对两个模块各自算出的 token 容量。稀疏注意力模块和 O 投影模块都从 config 推导 T，但走的是不同路径；两边对不上说明有一侧的配置读错了。',
            reasons: [
              'T 与 T_PAD 两个量都要逐一比对。',
              '这是跨模块的一致性检查，不是单个模块内部的约束。',
              '失败时整个程序拒绝加载，而不是编出一个错的 kernel。'
            ]
          }
        ],
        'tile-dim': [
          {
            id: 'OAK=192',
            title: 'O_A_K_TILE 取 192',
            rule: 'raise ValueError(f"O-A input {O_GROUP_IN} must be divisible by K tile {O_A_K_TILE}")',
            copy: 'O_GROUP_IN = H × HEAD_DIM / O_GROUPS = 64 × 512 / 8 = 4096。K tile 取 192 时 4096 / 192 不整除，最后一个 K 块是残块，分组投影的 reduce 会多算或少算。',
            reasons: [
              'O 投影第一段按 K 方向分块累加，残块没有单独的处理路径。',
              '256 能整除 4096，取 16 个完整 K 块。',
              '源码对 O-A 的 K / N 和 O-B 的 K / D 各有一条独立断言。'
            ]
          },
          {
            id: 'BIAS_T=6',
            title: 'BIAS_T_TILE 取 6',
            rule: 'raise ValueError("CSA token capacity must contain complete bias tiles")',
            copy: 'T = 256，6 不整除 256。bias / slot 向量按 token 块处理，残块会让最后一轮读到越界或重复的 topk 索引。',
            reasons: [
              'BIAS_T_TILE = min(T, 8)，8 整除 256。',
              'T 本身由 DECODE_BATCH / TP × DECODE_SEQ 决定，是 8 的倍数。',
              'ROPE_CS_T_TILE 同样取 8，注释写明「T is a multiple of 8 by the batch contract」。'
            ]
          }
        ],
        fractal: [
          {
            id: 'T_PAD',
            title: 'token 轴不补到 16 的倍数',
            rule: 'T_PAD = ((T + 16 - 1) // 16) * 16  # T padded up to the 16-row cube M floor',
            copy: 'Cube 的 M 方向以 16 行为最小单位。token 数不补齐到 16 的倍数时，QK/PV 的最后一行块填不满分形，搬运和 Mmad 都要按 16 对齐补齐，声明的形状与实际占用对不上。',
            reasons: [
              'L0 按 16 × 16 分形块组织，M 必须是 16 的倍数。',
              'T = 256 本身已经对齐，T_PAD 在 TP = 4（T = 128）时同样成立。',
              '窗口行数 GROUP_T_PAD = TP × T_PAD 也继承这个对齐。'
            ]
          }
        ]
      }
    },
    'onchip-budget': {
      title: '片上预算拒绝',
      copy: '累加器压 L0C 128 KB，向量工作集压 Vec 184 KB，KV tile 压 Mat 512 KB。三个预算同时成立才算过。',
      status: '72 / 120 rejected',
      defaultCategory: 'l0c',
      categories: [
        { id: 'l0c', label: 'L0C 放不下 INT32 累加器', hint: 'PROJ_B_MM', count: 30 },
        { id: 'vec', label: 'Vec 放不下 bias / topk 向量', hint: 'indexer', count: 28 },
        { id: 'mat', label: 'Mat 放不下 KV tile', hint: 'QK/PV', count: 14 }
      ],
      cases: {
        l0c: [
          {
            id: 'PROJ_B 128×512',
            title: 'PROJ_B_MM_N_TILE 取 512',
            rule: '128 × 512 × 4 B = 256 KB > 128 KB',
            copy: 'O 投影第二段在 L0C 上以 INT32 累加。T tile 128 配 N tile 512 需要 256 KB，是 L0C 容量的两倍。源码注释直接点名了这条约束：「Keep the INT32 proj-b accumulator within the A2/A3 tile buffer」。',
            reasons: [
              'INT8 矩阵乘的累加器是 INT32，每个元素 4 字节。',
              '128 × 256 × 4 B = 128 KB，正好等于 L0C 容量——这就是源码取 256 的原因。',
              '再大一档就越界，不存在"稍微超一点"的余地。'
            ]
          },
          {
            id: 'PROJ_B 256×256',
            title: 'PROJ_B_MM_T_TILE 取 256',
            rule: '256 × 256 × 4 B = 256 KB > 128 KB',
            copy: '把 token 方向的 tile 翻倍同样越界。T tile 与 N tile 对 L0C 的压力是乘性的，只能二选一放大，而放大 N 对 K 方向的复用更有利。',
            reasons: [
              'L0C 占用 = T_TILE × N_TILE × 4 B，两个方向等价地贵。',
              '放大 N 能摊薄权重的 L1 → L0 搬运，放大 T 不能。',
              '最终取 T = 128、N = 256。'
            ]
          }
        ],
        vec: [
          {
            id: 'BIAS whole-T',
            title: 'bias 向量按整个 T 做',
            rule: '256 × 512 × 4 B = 512 KB > 184 KB',
            copy: 'indexer 的 slot / bias 向量形状是 [T, IDX_TOPK] FP32。按整个 token 轴一次做完需要 512 KB，远超 Vec 可用的 184 KB。源码注释写得很直接：「the whole-T form would put [T, IDX_TOPK] FP32 tiles well past the Vec limit」。',
            reasons: [
              'IDX_TOPK = 512，FP32 每元素 4 字节。',
              'BIAS_T_TILE = min(T, 8) 把它压到 8 × 512 × 4 B = 16 KB。',
              'Vec 可用 184 KB（物理 192 KB，PTO-ISA 保留约 8 KB）。'
            ]
          },
          {
            id: 'BIAS_T=128',
            title: 'bias 块取 128 token',
            rule: '128 × 512 × 4 B = 256 KB > 184 KB',
            copy: '折中到 128 个 token 仍然越界。要留出 softmax 中间量和 topk 索引的空间，这个块只能开到几十 KB 量级。',
            reasons: [
              '单是 bias 张量就 256 KB，已经超过整个 Vec 预算。',
              '同时还要放下 exp / max / sum 的中间量。',
              '32 token（64 KB）理论上放得下，但留给其余算子的余量过小。'
            ]
          }
        ],
        mat: [
          {
            id: 'ATTN_K=512',
            title: 'ATTN_K_TILE 取 512',
            rule: '512 × 512 × 2 B = 512 KB，Mat 正好打满',
            copy: '稀疏注意力的 KV tile 形状是 [ATTN_K_TILE, HEAD_DIM] BF16。K tile 取 512 时单块就填满整个 L1，没有给 Q、权重和 double buffer 留任何空间。',
            reasons: [
              'HEAD_DIM = 512，BF16 每元素 2 字节。',
              'ATTN_K_TILE = 128 给出 128 KB，L1 还剩四分之三。',
              'QK_M_TILE / H_TILE = 32 / 16 = 2，注释说明这带来 2 路 KV 的 L1 → L0 复用。'
            ]
          }
        ]
      }
    },
    'compiler-reality': {
      title: '编译器现实拒绝',
      copy: '这一类筛掉的不是"算得慢"，而是"编不出来"或"编出来是错的"。四条都在 decode_csa.py 与兄弟模块的注释里有原话。',
      status: '84 / 96 rejected',
      defaultCategory: 'handle',
      categories: [
        { id: 'handle', label: 'inline 句柄跨不过调用边界', hint: '原地写入', count: 26 },
        { id: 'symbol', label: '动态符号不可证相等', hint: '两条 token 流', count: 24 },
        { id: 'miscompile', label: '单块 sparse-K 误编译', hint: '已知缺陷', count: 20 },
        { id: 'import-freeze', label: 'TP 形状 import 时已冻结', hint: '不能运行期切', count: 14 }
      ],
      cases: {
        handle: [
          {
            id: 'allgather-handle',
            title: '用 all-gather 的返回句柄喂给 kv_proj_rope',
            rule: 'a returned inline handle cannot cross into kv_proj_rope',
            copy: 'decode_cp_token_allgather_step 是原地写 x_normed_full 的。很自然的写法是接住它的返回值再往下传，但 inline 函数返回的句柄跨不进下一个调用。源码特地保留原始句柄，并把这条写进注释。',
            reasons: [
              '子内核原地写入时，调用方手里的原句柄才是有效的那个。',
              '返回的 inline 句柄只在当前作用域内成立。',
              'o_proj_reduce_scatter → hc_post 之间有完全相同的一处，注释措辞也一样。'
            ]
          },
          {
            id: 'oproj-handle',
            title: '用 reduce-scatter 的返回句柄喂给 hc_post',
            rule: 'a returned inline handle cannot cross into hc_post',
            copy: '同一类问题的第二处。o_proj_reduce_scatter 原地写 attn_out，源码用 _o_reduced 接住返回值但不使用，继续把原始的 attn_out 传给 hc_post。下划线前缀本身就是这个意图的标记。',
            reasons: [
              '两处都用 _ 前缀接住返回值表示刻意丢弃。',
              '真正往下传的是 create_tensor 时拿到的原句柄。',
              '这是作者踩过之后留下的防回归注释。'
            ]
          }
        ],
        symbol: [
          {
            id: 'compressor-view',
            title: '把 gathered 流的符号直接传给压缩器',
            rule: 'mixing them with the gathered stream\'s symbols leaves the two not provably equal across the call',
            copy: '压缩器的 token 轴和请求轴在一次调用里绑定到一个行数。如果把全组 token 流的动态符号直接带进去，编译器无法证明调用两侧的行数相等，依赖关系就建立不起来。源码的做法是先 reshape 成标量范围的视图再传。',
            reasons: [
              'cmp_positions / cmp_slots / cmp_state_slots 全部先 reshape 到 [kv_dim]。',
              'position_ids_local 还要同时提供 [t_dim] 和 [t_dim, 1] 两种视图——indexer 要平的，稀疏注意力要二维的。',
              '本 rank 轴 T_DYN 与全组轴 KV_T_DYN 是两个独立符号，不能混。'
            ]
          },
          {
            id: 'two-streams',
            title: '让 indexer 的两半跑同一条 token 流',
            rule: 'the indexer cache is per-request state',
            copy: 'indexer 的 cache 是按请求存的状态，它的压缩器那一半必须跑在全组 token 流上；而查询那一半只看本 rank 的行。写成同一条流会让 cache 的请求维度对不上。',
            reasons: [
              'indexer_compressor 吃 x_normed_full（全组流）。',
              'indexer 吃 x_normed_t（本 rank 流）。',
              '两者通过 idx_cache_write_tid 这个 task id 串起先后，而不是靠共享张量。'
            ]
          }
        ],
        miscompile: [
          {
            id: 'SPARSE_BLOCKS=1',
            title: '稀疏块数取 1',
            rule: 'a single sparse-K block miscompiles in pypto (S-stride cross-token output mixup)',
            copy: '当 TOPK 小到一个 K 块就装得下时，最自然的选择是 SPARSE_BLOCKS = 1。但 pypto 在这个形状上会误编译，产生跨 token 的输出串味。源码把它 floor 到 2，并说明「a 2-block build with an all-invalid 2nd block is bit-exact」。',
            reasons: [
              '这是编译器缺陷，不是算法问题——算出来的结果是错的，不是慢。',
              '补一个全无效的第二块，结果与正确实现逐位相同。',
              '当前配置 TOPK = WIN + CMP_TOPK = 640，SPARSE_BLOCKS = 5，这个 floor 并不生效，但它保证了小 TOPK 配置也安全。'
            ]
          }
        ],
        'import-freeze': [
          {
            id: 'runtime-tp',
            title: '把 TP 做成运行期参数',
            rule: 'Sub-kernels freeze TP-derived shapes at import time',
            copy: '希望同一份编译产物服务不同 TP 规模，是很自然的想法。但所有子内核在 import 时就把 TP 派生的形状算死了——LOCAL_O_GROUPS、T、T_PAD、窗口行数全都是模块级常量。所以 decode_csa.py 必须在 import 子内核之前先解析 --tp 并写回 config。',
            reasons: [
              '文件开头用手写的 _parse_tp_argv 解析 argv，而不是用 argparse——因为它必须在 import 之前跑完。',
              'config.TP = TP_SIZE 这一行的位置是刻意的：它在 import pypto.language 之前。',
              '换 TP 等于换一份编译产物，不是换一个运行期开关。'
            ]
          }
        ]
      }
    },
    rank: {
      title: '排序淘汰',
      copy: '这一步不判对错，只看核利用率与通信代价。三条判据都能在源码的常量注释里找到原话。',
      status: '9 / 12 ranked out',
      defaultCategory: 'under',
      categories: [
        { id: 'under', label: 'worker 开太少，核闲置', hint: 'AIV 利用率', count: 4 },
        { id: 'over', label: 'worker 开太多，只烧核', hint: 'fabric 受限', count: 3 },
        { id: 'coarse', label: 'D 切太粗，饿着 AIC', hint: 'AIC 利用率', count: 2 }
      ],
      cases: {
        under: [
          {
            id: 'RS_REDUCE=8',
            title: 'reduce-scatter 只开 8 个 worker',
            rule: 'O_RS_REDUCE_WORKERS = 48   # 128 row blocks; 8 left 40 of 48 AIV idle',
            copy: 'reduce 阶段有 128 个行块要处理。只开 8 个 worker 时，48 个 AIV 里有 40 个全程闲着。源码把这个数字直接提到 48，让一波就吃满整块芯片的 Vector 侧。',
            reasons: [
              '工作项 128 个，远多于 48，开满不会出现尾部浪费。',
              '注释把被否掉的取值和它的后果一起写了下来。',
              '同一文件里 O_RS_DEQUANT_WORKERS = 12，注释说明 4 个 owner × 12 = 48，也是一波。'
            ]
          },
          {
            id: 'QUANT=3',
            title: 'O-A 量化每组只开 3 路',
            rule: 'O_A_QUANT_WORKERS = 6   # per owner-group; 4 x 2 x 6 -> one AIV wave',
            copy: '量化按 owner-group 分派。每组 6 路时 4 × 2 × 6 = 48，正好一波 AIV；开 3 路就只用掉半块芯片，而量化本身是纯 Vector 工作，没有理由不开满。',
            reasons: [
              '4 个 owner × 2 × 6 = 48 = AIV 数。',
              '注释把这个乘法写了出来，说明数字是倒推出来的。',
              'ATTENTION_PUBLISH_WORKERS 同样取 48，出于一样的理由。'
            ]
          }
        ],
        over: [
          {
            id: 'RS_PUBLISH=48',
            title: 'reduce-scatter publish 开到 48',
            rule: 'O_RS_PUBLISH_WORKERS = 24    # put is fabric-bound; more workers only burn cores',
            copy: 'publish 阶段做的是 pld.tensor.put，瓶颈在卡间 fabric 带宽而不是核。开到 48 不会更快，只是多占 24 个核去等同一条链路。源码取 24，并把理由写在旁边。',
            reasons: [
              'put 是单边写，速度由 fabric 决定。',
              'O_RS_PUT_T_TILE = 8 的注释补齐了算术：4 owner × 16 行块 = 64 个 put 分到 24 个 worker。',
              '省下的核留给同时在跑的 Vector 段。'
            ]
          }
        ],
        coarse: [
          {
            id: 'PROJ_B_D=1024',
            title: 'proj_b 的 D 块取 1024',
            rule: 'PROJ_B_D_TILE = 512  # proj_b_mm D chunk per task; coarser starves the 24 AIC cores',
            copy: 'O 投影第二段按 D 方向切任务。D = 4096，切成 512 一块得到 8 个任务；切成 1024 只剩 4 个任务，喂不满 24 个 AIC。注释里 starves 这个词说的就是这件事。',
            reasons: [
              '任务数要足够多才能填满 24 路 Cube 分派。',
              '切得更细会增加权重的重复搬运，512 是两者的平衡点。',
              '同一侧的 NUM_QK_CORES = 24 注释还补了一句「re-sweep for other AIC counts」——换芯片要重扫。'
            ]
          }
        ]
      }
    }
  };

  /* ── 决策点矩阵：每个常量的取值、占用与结论 ─────────────────────── */
  const KB = 1024;

  const decisionMatrix = [
    {
      id: 'proj-b-acc',
      label: 'PROJ_B 累加器',
      hint: 'T_TILE × N_TILE × INT32',
      budget: 'L0C 128 KB',
      capacity: CAPACITY.Acc,
      source: 'decode_o_proj.py · PROJ_B_MM_T_TILE / PROJ_B_MM_N_TILE',
      options: [
        { label: '64 × 256', bytes: 64 * 256 * 4, verdict: '放得下，但 N 方向复用不足' },
        { label: '128 × 256', bytes: 128 * 256 * 4, chosen: true, verdict: '正好打满 L0C —— 源码取值' },
        { label: '128 × 512', bytes: 128 * 512 * 4, verdict: '超 L0C 一倍' },
        { label: '256 × 256', bytes: 256 * 256 * 4, verdict: '超 L0C 一倍' }
      ]
    },
    {
      id: 'bias-tile',
      label: 'indexer bias 块',
      hint: 'T_TILE × IDX_TOPK × FP32',
      budget: 'Vec 184 KB',
      capacity: CAPACITY.Vec,
      source: 'decode_sparse_attn_csa.py · BIAS_T_TILE',
      options: [
        { label: '8 × 512', bytes: 8 * 512 * 4, chosen: true, verdict: '源码取值 min(T, 8)' },
        { label: '32 × 512', bytes: 32 * 512 * 4, verdict: '放得下，但挤占 softmax 中间量' },
        { label: '128 × 512', bytes: 128 * 512 * 4, verdict: '超 Vec' },
        { label: '256 × 512', bytes: 256 * 512 * 4, verdict: '整个 T，注释点名否掉的写法' }
      ]
    },
    {
      id: 'kv-tile',
      label: '稀疏 KV tile',
      hint: 'ATTN_K_TILE × HEAD_DIM × BF16',
      budget: 'Mat 512 KB',
      capacity: CAPACITY.Mat,
      source: 'decode_sparse_attn_csa.py · ATTN_K_TILE',
      options: [
        { label: 'K = 64', bytes: 64 * 512 * 2, verdict: '块太碎，稀疏块数翻倍' },
        { label: 'K = 128', bytes: 128 * 512 * 2, chosen: true, verdict: '源码取值，L1 还剩四分之三' },
        { label: 'K = 256', bytes: 256 * 512 * 2, verdict: '放得下，但 double buffer 余量减半' },
        { label: 'K = 512', bytes: 512 * 512 * 2, verdict: '正好打满 L1，无余量' }
      ]
    },
    {
      id: 'qk-cores',
      label: 'QK/PV 分派路数',
      hint: '对照 AIC 数',
      budget: `AIC ${CORES.aic}`,
      capacity: CORES.aic,
      countMode: true,
      source: 'decode_sparse_attn_csa.py · NUM_QK_CORES',
      options: [
        { label: '12 路', count: 12, verdict: '只用半数 Cube 核' },
        { label: '24 路', count: 24, chosen: true, verdict: '源码取值 = A2/A3 的 AIC 数' },
        { label: '48 路', count: 48, verdict: '超过 AIC 数，排队而非并行' }
      ]
    },
    {
      id: 'publish-workers',
      label: 'publish worker 数',
      hint: '对照 AIV 数',
      budget: `AIV ${CORES.aiv}`,
      capacity: CORES.aiv,
      countMode: true,
      source: 'decode_sparse_attn_csa.py · ATTENTION_PUBLISH_WORKERS',
      options: [
        { label: '24 路', count: 24, verdict: '半波 AIV，merge 是纯 Vector 工作' },
        { label: '48 路', count: 48, chosen: true, verdict: '源码取值 = 一波 AIV' },
        { label: '96 路', count: 96, verdict: '超过 AIV 数' }
      ]
    },
    {
      id: 'rs-publish',
      label: 'RS publish worker',
      hint: 'put 受 fabric 限制',
      budget: `AIV ${CORES.aiv}`,
      capacity: CORES.aiv,
      countMode: true,
      source: 'decode_o_proj.py · O_RS_PUBLISH_WORKERS',
      options: [
        { label: '8 路', count: 8, verdict: '64 个 put 排不开' },
        { label: '24 路', count: 24, chosen: true, verdict: '源码取值，够喂满 fabric' },
        { label: '48 路', count: 48, verdict: '注释：more workers only burn cores' }
      ]
    },
    {
      id: 'sparse-blocks',
      label: '稀疏块数下限',
      hint: 'max(2, ceil(TOPK / K_TILE))',
      budget: '已知缺陷',
      capacity: 0,
      countMode: true,
      source: 'decode_sparse_attn_csa.py · SPARSE_BLOCKS',
      options: [
        { label: '允许 1 块', count: 1, defect: true, verdict: '单块会误编译，跨 token 输出串味' },
        { label: 'floor 到 2', count: 2, chosen: true, verdict: '源码取值，补全无效块后逐位相同' }
      ]
    }
  ];

  /* ── 三个候选：源码真实支持的三个 TP 世界 ──────────────────────── */
  const candidates = {
    tp1: {
      key: 'tp1',
      name: 'A · TP = 1',
      kicker: '无跨卡',
      entry: 'decode_csa_tp1',
      tp: 1,
      note: '整层跑在一张卡上，没有 all-gather、没有 a2a、没有 reduce-scatter。依赖图最简单，但单卡要扛全部 64 个请求。'
    },
    tp2: {
      key: 'tp2',
      name: 'B · TP = 2',
      kicker: '默认',
      entry: 'decode_csa',
      tp: 2,
      note: '_TP_DEFAULT = 2。三处跨卡都在，窗口与信号布局最规整：LOCAL_O_GROUPS = 4 正好被 PUBLISH_GROUPS = 2 整除。'
    },
    tp4: {
      key: 'tp4',
      name: 'C · TP = 4',
      kicker: '最深切分',
      entry: 'decode_csa',
      tp: 4,
      note: '每卡只剩 16 个请求、128 个 token。单卡算得最少，但 publish 工作项掉到 64 个、LOCAL_O_GROUPS 只剩 2，通信占比明显上升。'
    }
  };

  const candidateOrder = ['tp1', 'tp2', 'tp4'];

  // 下表数值全部按 config.py 的 FLASH 配置现算
  const perTp = (tp) => {
    const B = CFG.DECODE_BATCH / tp;
    const T = B * CFG.DECODE_SEQ;
    const tPad = Math.ceil(T / 16) * 16;
    const localOGroups = CFG.O_GROUPS / tp;
    const groupTPad = tp * tPad;
    return {
      B,
      T,
      tPad,
      localOGroups,
      groupTPad,
      attentionWindowRows: localOGroups * groupTPad,
      oWindowRows: tp * tPad,
      qkItems: T * CFG.SPARSE_BLOCKS,
      qkRounds: Math.ceil((T * CFG.SPARSE_BLOCKS) / CORES.aic),
      publishItems: (T / 8) * (CFG.H / CFG.H_TILE),
      publishWaves: Math.ceil(((T / 8) * (CFG.H / CFG.H_TILE)) / CORES.aiv),
      publishGroups: CFG.H_TILE / (CFG.H / CFG.O_GROUPS)
    };
  };

  const TPS = [1, 2, 4].map(perTp);

  const candidateRows = [
    { label: '入口函数', values: ['decode_csa_tp1', 'decode_csa', 'decode_csa'] },
    { label: 'TP_SIZE', values: ['1', '2（默认）', '4'] },
    { label: '每卡请求数 B', values: TPS.map((t) => String(t.B)) },
    { label: '每卡 token 数 T', values: TPS.map((t) => String(t.T)) },
    { label: 'T_PAD（16 行对齐）', values: TPS.map((t) => String(t.tPad)) },
    { label: 'LOCAL_O_GROUPS', values: TPS.map((t) => String(t.localOGroups)) },
    { label: 'PUBLISH_GROUPS 整除', values: TPS.map((t) => `${t.localOGroups} / ${t.publishGroups} = ${t.localOGroups / t.publishGroups}`) },
    { label: 'attention_window 行数', values: TPS.map((t) => String(t.attentionWindowRows)) },
    { label: '跨卡段数', values: ['0', '3 段', '3 段'], tones: ['positive', '', 'warning'] },
    { label: 'QK/PV 工作项', values: TPS.map((t) => String(t.qkItems)) },
    { label: `每 AIC 轮次（${CORES.aic} 核）`, values: TPS.map((t) => String(t.qkRounds)), tones: ['warning', '', 'positive'] },
    { label: 'publish 工作项', values: TPS.map((t) => String(t.publishItems)) },
    { label: `publish 波次（${CORES.aiv} 核）`, values: TPS.map((t) => String(t.publishWaves)) },
    { label: '依赖图复杂度', values: ['低', '中', '中'] },
    { label: '主要风险', values: ['单卡算力成为上限', '无阻塞风险', '通信占比上升，分组切碎'], tones: ['warning', '', 'warning'] },
    { label: '证据来源', values: ['源码变体', '源码默认值', '源码变体'] }
  ];

  /* ── 流水示意 ───────────────────────────────────────────────────── */
  const pipelines = {
    tp1: {
      meta: 'TP = 1 · 512 token / 卡 · 无跨卡',
      note: '一条直线：rope → norm → q/kv 投影 → cache 写回 → 压缩器 → indexer → 稀疏注意力 → merge → O 投影。没有窗口缓冲，也没有信号量，但每一段的工作量都是 TP = 2 的两倍。',
      lanes: [
        { label: 'Cube', blocks: ['cube:qkv', 'cube:QK', 'cube:PV', 'cube:QK', 'cube:PV', 'cube:O-A', 'cube:O-B', 'idle:'] },
        { label: 'Vector', blocks: ['vector:rope', 'vector:norm', 'idle:', 'vector:merge', 'idle:', 'vector:merge', 'vector:quant', 'vector:act'] },
        { label: '跨卡', blocks: ['idle:', 'idle:', 'idle:', 'idle:', 'idle:', 'idle:', 'idle:', 'idle:'] }
      ]
    },
    tp2: {
      meta: 'TP = 2 · 256 token / 卡 · 3 段跨卡',
      note: 'token all-gather 在 KV 分支之前完成，publish 阶段算完一个分组就 put 到目标 rank，a2a 与 reduce-scatter 跟在后面。跨卡与计算交错，auto_scope=False 让这些依赖边由作者显式排出。',
      lanes: [
        { label: 'Cube', blocks: ['idle:', 'cube:qkv', 'cube:QK', 'cube:PV', 'cube:QK', 'cube:O-A', 'cube:O-B', 'idle:'] },
        { label: 'Vector', blocks: ['vector:rope', 'vector:norm', 'idle:', 'vector:merge', 'vector:merge', 'vector:quant', 'vector:act', 'idle:'] },
        { label: '跨卡', blocks: ['idle:', 'drain:gather', 'idle:', 'drain:put', 'drain:put', 'drain:a2a', 'idle:', 'drain:RS'] }
      ]
    },
    tp4: {
      meta: 'TP = 4 · 128 token / 卡 · 3 段跨卡',
      note: '计算段缩到最短，但三段跨卡的次数没变、分片更碎：LOCAL_O_GROUPS 只剩 2，publish 工作项掉到 64 个、不足两波 AIV。通信在整层里的占比明显上升。',
      lanes: [
        { label: 'Cube', blocks: ['idle:', 'cube:qkv', 'cube:QK', 'idle:', 'cube:PV', 'cube:O-A', 'idle:', 'cube:O-B'] },
        { label: 'Vector', blocks: ['vector:rope', 'vector:norm', 'idle:', 'vector:merge', 'idle:', 'vector:quant', 'idle:', 'vector:act'] },
        { label: '跨卡', blocks: ['drain:gather', 'idle:', 'drain:put', 'drain:put', 'drain:a2a', 'idle:', 'drain:RS', 'drain:RS'] }
      ]
    }
  };

  /* ── 工程文件 ───────────────────────────────────────────────────── */
  const orchestrationSource = () => `# Data/DeepseekV4/deepseek_v4_flash_dspark/decode_csa.py（节选）
# 一个 TP rank 的 CSA decode 层编排。

# 子内核在 import 时就把 TP 派生的形状算死，
# 所以要先解析 --tp 并写回 config，再 import 它们。
_TP_CHOICES = (1, 2, 4)
_TP_DEFAULT = 2

TP_SIZE = _parse_tp_argv()
if TP_SIZE not in _TP_CHOICES:
    raise ValueError(f"--tp must be one of {_TP_CHOICES} (got {TP_SIZE})")
config.TP = TP_SIZE

import pypto.language as pl
import pypto.language.distributed as pld

# 本 rank token 轴与全组 token 轴是两个独立符号。
B_DYN = pl.dynamic("B_DYN")
T_DYN = pl.dynamic("T_DYN")

B = DECODE_BATCH // TP_SIZE      # 32
S = DECODE_SEQ                   # 8
T = B * S                        # 256
CSA_WB_TOKEN_TILE = 8

if T != LOCAL_T:
    raise ValueError(f"CSA token capacity {T} must equal TP local token capacity {LOCAL_T}")


@pl.jit.inline(auto_scope=False)
def decode_csa(x_hc, ..., x_out, gather_window, gather_signal,
               attention_window, attention_signal, o_window, o_signal,
               group_base, tp_rank, local_t):
    """Run one rank of the context-parallel CSA layer."""
    t_dim = pl.tensor.dim(x_hc, 0)
    kv_dim = pl.tensor.dim(ori_slot_mapping, 0)

    with pl.scope():
        hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed, post_t, comb_t)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_rope_interleave") as rope_tid:
        ...                                   # 预先构造交织后的 cos / sin

    with pl.scope():
        rms_norm(x_mixed, attn_norm_w, x_normed_t)

    with pl.scope():
        # 原地写 x_normed_full：保留原句柄，
        # inline 返回的句柄跨不进 kv_proj_rope。
        _gathered_normed, gather_signal = decode_cp_token_allgather_step(
            x_normed_t, x_normed_full, gather_window, gather_signal,
            group_base, tp_rank,
        )

    with pl.scope():
        late_dep = pl.system.task_dummy(deps=[rope_tid])
        q_proj_rope(x_normed_t, wq_a, wq_b, wq_b_scale, gamma_cq, ..., q, qr, qr_scale)
        kv_proj_rope(x_normed_full, wkv, gamma_ckv, ..., kv_full, late_dep)

        with pl.spmd(kv_dim // CSA_WB_TOKEN_TILE, name_hint="csa_cache_writeback") as ori_cache_write_tid:
            wb_blk = pl.tile.get_block_idx()
            for write_dt in pl.range(CSA_WB_TOKEN_TILE):
                write_row_i64 = pl.read(ori_slot_mapping, [wb_blk * CSA_WB_TOKEN_TILE + write_dt])
                if write_row_i64 >= 0:
                    ...                       # 按 slot mapping 散写进 paged KV cache

        # 压缩器的 token / 请求轴每次调用绑一个行数；
        # 混进 gathered 流的符号会让两边不可证相等。
        cmp_positions = pl.reshape(position_ids, [kv_dim])
        cmp_slots = pl.reshape(cmp_slot_mapping, [kv_dim])
        cmp_out, cmp_cache_write_tid = compressor_ratio4(...)
        cache_ready_dep = pl.system.task_dummy(deps=[ori_cache_write_tid, cmp_cache_write_tid])

        # indexer 的 cache 是按请求存的状态：压缩器那一半跑全组流，
        # 查询那一半只看本 rank 的行。
        idx_cache_write_tid = indexer_compressor(x_normed_full, ...)
        idx_topk_scores, idx_topk = indexer(x_normed_t, qr, qr_scale, ...,
                                            late_dep, idx_cache_write_tid)

        (sparse_blk_mi, sparse_blk_li, sparse_blk_oi,
         rope_cos_il, rope_sin_signed, rope_swap_idx,
         qk_tid, attn_rope_tid) = sparse_attn_csa(
            q, kv_cache, window_swa_indices, cmp_kv, cmp_block_table, idx_topk,
            position_ids_t1, freqs_cos_local, freqs_sin_local, cache_ready_dep,
        )
        ...
    return x_out
`;

  const publishSource = () => `# decode_csa.py · csa_merge_pack_publish（节选）
# 48 路 = A2/A3 的 AIV 数。一次打包 8 个 token × 16 个 head。

with pl.spmd(ATTENTION_PUBLISH_WORKERS, name_hint="csa_merge_pack_publish",
             deps=[qk_tid, attn_rope_tid]) as publish_tid:
    worker = pl.tile.get_block_idx()
    for pack_work in pl.range(worker, pack_work_count, ATTENTION_PUBLISH_WORKERS):
        token_block = pack_work // (H // H_TILE)
        m_h0 = (pack_work - token_block * (H // H_TILE)) * H_TILE
        global_group0 = m_h0 // HEADS_PER_GROUP
        destination_rank = global_group0 // LOCAL_O_GROUPS

        for m_dt in pl.range(ATTENTION_PUBLISH_T_TILE):
            ...
            # 在线 softmax 合并：扫 SPARSE_BLOCKS 个分块，stage=2 软件流水
            for m_sb in pl.pipeline(1, SPARSE_BLOCKS, stage=2):
                m_mi_new = pl.maximum(m_mi, m_cur_mi)
                m_alpha = pl.exp(pl.sub(m_mi, m_mi_new))
                m_beta = pl.exp(pl.sub(m_cur_mi, m_mi_new))
                m_li = pl.add(pl.mul(m_alpha, m_li), pl.mul(m_beta, m_cur_li))
                m_oi = pl.add(pl.row_expand_mul(m_oi, m_alpha),
                              pl.row_expand_mul(m_cur_oi, m_beta))
                m_mi = m_mi_new

            # attention sink 进分母，再做输出侧的逆 RoPE
            n_denom = pl.add(m_li, pl.exp(pl.sub(n_sink_tile, m_mi)))
            n_full = pl.row_expand_div(m_oi, n_denom)[0:H_TILE, 0:HEAD_DIM]
            ...

        # 算完一个分组就单边写到目标 rank 的窗口
        for group_slot in pl.unroll(PUBLISH_GROUPS):
            pld.tensor.put(
                dst=attention_window, peer=group_base + destination_rank,
                src=attention_grouped,
                dst_offsets=[target_row, 0], src_offsets=[source_row, 0],
                shape=[ATTENTION_PUBLISH_T_TILE, O_GROUP_IN],
                chunk_rows=ATTENTION_PUBLISH_T_TILE, chunk_cols=O_GROUP_IN,
            )

    # put 不自带完成语义，写完显式通知每个 peer
    for peer_tp in pl.range(TP_SIZE):
        if peer_tp != tp_rank:
            pld.system.notify(target=attention_signal, peer=group_base + peer_tp,
                              offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
`;

  const constantsSource = () => `# decode_sparse_attn_csa.py / decode_o_proj.py（节选）
# 每个常量旁边的注释，就是这次反向推导的证据。

# ── 切分 ─────────────────────────────────────────────────────────
H_TILE = 16
QK_M_TILE = 32           # qk_pv M rows per QK/PV matmul; QK_M_TILE/H_TILE-way KV L1->L0 reuse
ATTN_K_TILE = 128
NUM_QK_CORES = 24        # qk_pv dispatch lanes = a2a3 AIC count; re-sweep for other AIC counts
T_PAD = ((T + 16 - 1) // 16) * 16  # T padded up to the 16-row cube M floor
ATTENTION_PUBLISH_WORKERS = 48
ATTENTION_PUBLISH_T_TILE = 8
LOCAL_O_GROUPS = O_GROUPS // TP
PUBLISH_GROUPS = H_TILE // HEADS_PER_GROUP
ROPE_CS_T_TILE = 8       # rope cos/sin row block; T is a multiple of 8 by the batch contract

TOPK = WIN + CMP_TOPK
# Floor to 2: a single sparse-K block miscompiles in pypto (S-stride cross-token
# output mixup); a 2-block build with an all-invalid 2nd block is bit-exact.
SPARSE_BLOCKS = max(2, (TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE)
QK_ITEMS = T * SPARSE_BLOCKS   # qk_pv work items: one per (token, sparse block)

# Token tile for the slot / bias vector work; the whole-T form would put
# [T, IDX_TOPK] FP32 tiles well past the Vec limit.
BIAS_T_TILE = min(T, 8)

# ── worker 数 ────────────────────────────────────────────────────
O_RS_REDUCE_WORKERS = 48     # 128 row blocks; 8 left 40 of 48 AIV idle
O_RS_PUBLISH_WORKERS = 24    # put is fabric-bound; more workers only burn cores
O_RS_DEQUANT_WORKERS = 12    # per owner; 4 owners -> one AIV wave
O_RS_PUT_T_TILE = 8          # 4 owners x 16 row blocks -> 64 puts over 24 workers
O_A_QUANT_WORKERS = 6        # per owner-group; 4 x 2 x 6 -> one AIV wave

# Keep the INT32 proj-b accumulator within the A2/A3 tile buffer.
PROJ_B_MM_T_TILE = 128
PROJ_B_MM_N_TILE = 256
PROJ_B_D_TILE = 512          # proj_b_mm D chunk per task; coarser starves the 24 AIC cores

# ── 整除契约 ─────────────────────────────────────────────────────
if T % BIAS_T_TILE != 0:
    raise ValueError("CSA token capacity must contain complete bias tiles")
if H_TILE % HEADS_PER_GROUP != 0:
    raise ValueError(f"CSA head tile {H_TILE} must contain complete output groups")
if O_GROUPS % TP != 0:
    raise ValueError(f"output groups {O_GROUPS} must be divisible by TP size {TP}")
if LOCAL_O_GROUPS % PUBLISH_GROUPS != 0:
    raise ValueError("local output groups must contain complete CSA publish tiles")
if T % ATTENTION_PUBLISH_T_TILE != 0:
    raise ValueError("local token capacity must contain complete attention publish tiles")
`;

  const hostSource = () => `# decode_csa.py · l3_decode_csa（节选）
# HOST 层：为 TP 组里每个 rank 分派一次，并开出 6 块跨卡窗口。

@pl.jit.host
def l3_decode_csa(x_hc, ..., x_out, local_t):
    """Launch the complete CSA layer on one physical TP group."""
    x_hc.bind_dynamic(1, T_DYN)
    freqs_cos.bind_dynamic(1, KV_T_DYN)
    kv_cache.bind_dynamic(1, ORI_BLOCK_NUM_DYN)
    ...                                      # 26 个动态轴逐一绑定

    gather_window_buf = pld.alloc_window_buffer([DECODE_GROUP_CAP, D], dtype=pl.BF16)
    gather_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    attention_window_buf = pld.alloc_window_buffer([ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
    attention_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    o_window_buf = pld.alloc_window_buffer([O_WINDOW_ROWS, D], dtype=pl.BF16)
    o_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        gather_window = pld.window(gather_window_buf, [DECODE_GROUP_CAP, D], dtype=pl.BF16)
        attention_window = pld.window(attention_window_buf,
                                      [ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
        o_window = pld.window(o_window_buf, [O_WINDOW_ROWS, D], dtype=pl.BF16)
        ...
        decode_csa_test(
            x_hc[rank], ..., x_out[rank],
            gather_window, gather_signal,
            attention_window, attention_signal, o_window, o_signal,
            0, rank, local_t,
            device=rank,
        )
    return x_out
`;

  const perfHints = (key) => {
    const hints = {
      tp1: [
        '[info] 无跨卡段：窗口缓冲与信号量全部未使用。',
        '[warn] QK/PV 工作项 2560，24 个 AIC 要跑 107 轮，单卡算力是上限。',
        '[info] publish 工作项 256，48 个 AIV 跑 6 波。'
      ],
      tp2: [
        '[info] 三段跨卡：token all-gather / attention a2a / O reduce-scatter。',
        '[info] LOCAL_O_GROUPS = 4，被 PUBLISH_GROUPS = 2 整除，窗口偏移连续。',
        '[info] QK/PV 工作项 1280，24 个 AIC 跑 54 轮；publish 128 项，48 个 AIV 跑 3 波。'
      ],
      tp4: [
        '[warn] publish 工作项降到 64，不足两波 AIV，尾部浪费上升。',
        '[warn] LOCAL_O_GROUPS = 2，与 PUBLISH_GROUPS = 2 相等，每次 publish 就是整个本地分组。',
        '[info] QK/PV 工作项 640，24 个 AIC 跑 27 轮，计算段最短。'
      ]
    };
    return `# build/report/perf_hints.log
# ${candidates[key].name} · 编译期性能提示（静态分析，非实测）

${hints[key].join('\n')}
`;
  };

  const ptoSource = (key) => {
    const c = candidates[key];
    const t = perTp(c.tp);
    return `// build/ptoas/decode_csa_qk_pv.pto
// ${c.name} · T = ${t.T} · SPARSE_BLOCKS = ${CFG.SPARSE_BLOCKS} · 工作项 ${t.qkItems}
// ptoas 会把这份 MLIR 编译成设备侧 .cpp 与 kernel 二进制。

module @decode_csa {
  pto.incore @qk_pv_m${CFG.QK_M_TILE}_k${CFG.ATTN_K_TILE}(
      %q:  !pto.tensor<${CFG.QK_M_TILE}x${CFG.HEAD_DIM}xbf16, #ddr>,
      %kv: !pto.tensor<${CFG.ATTN_K_TILE}x${CFG.HEAD_DIM}xbf16, #ddr>,
      %o:  !pto.tensor<${CFG.QK_M_TILE}x${CFG.HEAD_DIM}xf32,  #ddr>) {

    // KV 进 L1 一次，供 QK_M_TILE / H_TILE = ${CFG.QK_M_TILE / CFG.H_TILE} 组 M 行复用
    %l1_kv = pto.tload %kv [%blk, 0] : <${CFG.ATTN_K_TILE}x${CFG.HEAD_DIM}xbf16, #mat>
    %l0b   = pto.tmove %l1_kv        : <${CFG.ATTN_K_TILE}x${CFG.HEAD_DIM}xbf16, #right>

    %l1_q  = pto.tload %q  [%m0, 0]  : <${CFG.QK_M_TILE}x${CFG.HEAD_DIM}xbf16, #mat>
    %l0a   = pto.tmove %l1_q         : <${CFG.QK_M_TILE}x${CFG.HEAD_DIM}xbf16, #left>

    %l0c   = pto.tmmad %l0a, %l0b, %acc_in : <${CFG.QK_M_TILE}x${CFG.ATTN_K_TILE}xf32, #acc>

    // 每块输出 (m_i, l_i, o_i) 三元组，留给 merge 阶段做在线 softmax 合并
    pto.tstore %l0c -> %o [%m0, %blk] : <${CFG.QK_M_TILE}x${CFG.ATTN_K_TILE}xf32, #acc> -> #ddr
  }
}
`;
  };

  const projectFiles = (key) => [
    { id: 'orchestration', path: 'deepseek_v4_flash_dspark/decode_csa.py', code: orchestrationSource() },
    { id: 'publish', path: 'deepseek_v4_flash_dspark/decode_csa.py · publish', code: publishSource() },
    { id: 'constants', path: 'deepseek_v4_flash_dspark/decode_sparse_attn_csa.py', code: constantsSource() },
    { id: 'host', path: 'deepseek_v4_flash_dspark/decode_csa.py · host', code: hostSource() },
    { id: 'pto', path: 'build/ptoas/decode_csa_qk_pv.pto', code: ptoSource(key) },
    { id: 'hints', path: 'build/report/perf_hints.log', code: perfHints(key) }
  ];

  const projectTree = [
    { type: 'folder', name: 'deepseek_v4_flash_dspark', depth: 0 },
    { type: 'file', id: 'orchestration', name: 'decode_csa.py', depth: 1 },
    { type: 'file', id: 'publish', name: '└ csa_merge_pack_publish', depth: 1 },
    { type: 'file', id: 'host', name: '└ l3_decode_csa', depth: 1 },
    { type: 'file', id: 'constants', name: 'decode_sparse_attn_csa.py', depth: 1 },
    { type: 'folder', name: 'build', depth: 0 },
    { type: 'folder', name: 'ptoas', depth: 1 },
    { type: 'file', id: 'pto', name: 'decode_csa_qk_pv.pto', depth: 2 },
    { type: 'folder', name: 'report', depth: 1 },
    { type: 'file', id: 'hints', name: 'perf_hints.log', depth: 2 }
  ];

  return {
    CAPACITY,
    CORES,
    CFG,
    KB,
    apiNotes,
    lineage,
    traceViews,
    rejectionBrowsers,
    decisionMatrix,
    candidates,
    candidateOrder,
    candidateRows,
    pipelines,
    projectFiles,
    projectTree
  };
})();
