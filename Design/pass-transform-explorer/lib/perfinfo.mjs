// Which Passes can change how fast the operator runs, and how strong that
// claim is.
//
// The tiers are not all the same kind of statement, and the UI says so:
//
//   `decide`  - FACT. These three Passes are the only ones in PyPTO that
//               register their own performance-hint codes (`PH-MR-*`,
//               `PH-AT-*`, `PH-DSA-*`). The compiler itself calls them out.
//   `shape`   - READING of the pass docs. Each entry cites the doc that says
//               what gets faster or slower, so the claim is checkable.
//   `traffic` - READING. These change how many bytes move or how many
//               instructions issue, without choosing between strategies.
//   `form`    - the residual: rewrites that normalize shape without moving
//               work. Performance-neutral until shown otherwise.
//
// A Pass sitting in `form` is a claim too, just a weaker one, so `form` carries
// no badge in the UI: an unmarked Pass means "nothing to report here", not
// "verified neutral".

/** Tier order is strongest-claim-first; `rank` drives the rail badges. */
export const PERF_TIERS = [
  {
    id: 'decide',
    rank: 3,
    label: '性能决策点',
    short: '决策',
    basis: '事实',
    hint: '在明显不同的性能结果之间做选择，且编译器为它注册了专属性能提示码。判定常卡在容量或对齐的整数边界上，输入稍变结果就翻转。',
  },
  {
    id: 'shape',
    rank: 2,
    label: '调度与落位成形',
    short: '成形',
    basis: '文档推断',
    hint: '决定指令顺序、核间重叠，或数据落在哪块片上内存。没有自检，成形得不好不会有任何告警。',
  },
  {
    id: 'traffic',
    rank: 1,
    label: '搬运与指令量',
    short: '搬运',
    basis: '文档推断',
    hint: '改变要搬多少字节、发多少条指令。方向通常是单向的（下降一定会变多），但量级取决于算子。',
  },
  {
    id: 'form',
    rank: 0,
    label: '形式变换',
    short: '',
    basis: '其余',
    hint: '把写法收敛成规范形式，不搬动工作量。IR 体积可能变化很大，但对性能中性。',
  },
];

/**
 * Per-Pass performance profile.
 *
 * `why`   - the mechanism, in one sentence.
 * `lever` - what a kernel author can change to flip the outcome. Omitted when
 *           the Pass gives the author no handle.
 * `doc`   - the doc that states it, so the claim can be checked. Named after
 *           the English pass docs under `repo/pypto/docs/en/dev/passes`.
 */
const PERF = {
  // -- decide: the compiler's own perf-hint owners -------------------------
  MemoryReuse: {
    tier: 'decide',
    why: '并发流水阶段能不能各占一块 buffer，取决于 F_g = min(depth, ⌊空间容量 / 每阶段字节⌋)。这个整数除法掉到 1，就意味着相邻阶段共用存储、串行执行。',
    lever: '缩小单阶段 tile，或减少同时存活的共驻 buffer；也可以直接把 pl.pipeline(stage=) 降到装得下的深度。',
    doc: '36-memory_reuse.md',
  },
  AutoTileMatmulL0: {
    tier: 'decide',
    why: '决定 matmul 的 M/N/K 切分，以及结果直接回写 GM 还是留在 Mat scratch。所有失败路径都是静默放弃优化，只留下一条 perf hint。',
    lever: '把 K 对齐到 cube fractal 16；链式 matmul 的中间量降到 bf16/f16 并用 mode="rint" 做 cast，downcast 才能折进 cube 的 FIXPIPE。',
    doc: '18-auto_tile_matmul_l0.md',
  },
  AllocateMemoryAddr: {
    tier: 'decide',
    why: '在容量约束下给每个 buffer 定物理地址。IR 体积完全不变，但选中的摆放会不会踩到被放宽的冲突对，决定了是否发 PH-DSA-001。',
    doc: '37-allocate_memory_addr.md',
  },

  // -- shape: decide schedule / placement, with no self-check --------------
  CanonicalizeIOOrder: {
    tier: 'shape',
    why: '只重排语句顺序：把所有 load 聚到前面，让 MTE 搬运引擎跑在计算引擎之前。不聚的话，每个副本的 load 只能等上一个副本 store 完才发出，完全没有 prefetch 重叠。',
    doc: '32-canonicalize_io_order.md',
  },
  SkewCrossCorePipeline: {
    tier: 'shape',
    why: '二值决策：错位软流水（AIC 与 AIV 真正重叠），或 DemoteToSequential（退回顺序执行）。退化条件很脆，单方向多于一组 tpush/tpop、动态边界、trip 小于 2 都会触发。',
    lever: '让跨核循环每个方向只有一次 tpush 和一次 tpop，并保证循环边界静态可知。',
    doc: '29-skew_cross_core_pipeline.md',
  },
  InferTileMemorySpace: {
    tier: 'shape',
    why: '决定每个 tile 落在 Vec / Mat / Acc / L0 的哪一块，并尝试把循环不变的 Mat 载入提到 preheader。decline 条件有二十余项，命中任一项就静默放弃提升。',
    doc: '20-infer_tile_memory_space.md',
  },
  LowerPipelineToSlots: {
    tier: 'shape',
    why: '在 PTOAS 内存规划下接管流水循环：给每个 load 一个 slots=F 的 MemRef，按 iv % F 索引，循环体不复制。走这条路的循环不再经过 MemoryReuse 的深度门。',
    doc: '30-lower_pipeline_to_slots.md',
  },
  LowerPipelineLoops: {
    tier: 'shape',
    why: '把循环体复制 F 份做 ping-pong，是全流水线体积改写最大的一步。它本身不做性能取舍，但它制造的片上压力，正是 MemoryReuse 之后要裁的那份。',
    doc: '31-lower_pipeline_loops.md',
  },
  ExpandMixedKernel: {
    tier: 'shape',
    why: '把混合 cube + vector 的核拆成 AIC / AIV 两侧，决定两个计算单元各自承担哪些语句，也就决定了后面还剩多少可重叠的空间。',
    doc: '24-expand_mixed_kernel.md',
  },
  SplitVectorKernel: {
    tier: 'shape',
    why: '按 vector 核切分工作，划定每个核的份额。切不均会直接表现为尾部空转。',
    doc: '26-split_vector_kernel.md',
  },
  LowerAutoVectorSplit: {
    tier: 'shape',
    why: '把过大的向量算子沿轴自动切块。切块大小同时决定单次搬运的粒度和 Vec 上的驻留峰值。',
    doc: '23-lower_auto_vector_split.md',
  },
  InjectGMPipeBuffer: {
    tier: 'shape',
    why: '为跨核往返注入 GM 中转缓冲。多一次 GM 往返就多一份带宽和延迟，但不注入两侧就无法解耦。',
    doc: '25-inject_gm_pipe_buffer.md',
  },
  AutoDeriveTaskDependencies: {
    tier: 'shape',
    why: '推导任务间依赖。推得比实际保守就会过度同步，任务级并行被白白串起来；这一步没有性能提示码，过度同步不会告警。',
    doc: '42-auto_derive_task_dependencies.md',
  },
  OptimizeOrchTensors: {
    tier: 'shape',
    why: '消掉编排层多余的 tensor 中转，减少 GM 往返。不命中时只是维持现状，没有任何信号。',
    doc: '12-optimize_orch_tensors.md',
  },

  // -- traffic: changes bytes moved or instructions issued ----------------
  ConvertTensorToTileOps: {
    tier: 'traffic',
    why: 'tensor 语义下降成 tile 语义，把全部 load / store 显式化。这一步之后，搬运量才第一次变得可数。',
    doc: '11-convert_tensor_to_tile_ops.md',
  },
  ResolveBackendOpLayouts: {
    tier: 'traffic',
    why: '按后端要求定 layout（ND / NZ）。不匹配的地方要插入转换，转换本身就是额外搬运。',
    doc: '22-resolve_backend_op_layouts.md',
  },
  FlattenTileNdTo2D: {
    tier: 'traffic',
    why: '把高维 tile 摊平成 2D。摊平后最内维的字节数，决定单次搬运能不能吃满 cache line。',
    doc: '14-flatten_tile_nd_to_2d.md',
  },
  LowerCompositeOps: {
    tier: 'traffic',
    why: '把复合算子拆成原语。核内集合通信在这里展开成一整套 credit-barrier 协议（notify / wait 加自清零收尾），是真正的核内同步插入点。',
    doc: '13-lower_composite_ops.md',
  },
  UnrollLoops: {
    tier: 'traffic',
    why: '展开循环，消掉循环开销但成倍放大指令数。本算子上这一步的膨胀比例值得单独看一眼。',
    doc: '02-unroll_loops.md',
  },
  InlineFunctions: {
    tier: 'traffic',
    why: '内联掉函数边界，为后续所有跨函数优化创造条件，同时也放大代码体积。',
    doc: '01-inline_functions.md',
  },
  LegalizeTileCast: {
    tier: 'traffic',
    why: '合法化 cast。落在 Vector 上的 cast，对 cube 链路意味着一次 cube 到 vector 再回 cube 的往返。',
    doc: '17-legalize_tile_cast.md',
  },
  FoldNoOpReshape: {
    tier: 'traffic',
    why: '折掉不产生实际搬运的 reshape，直接减少指令数。',
    doc: '38-fold_no_op_reshape.md',
  },
  FuseCreateAssembleToSlice: {
    tier: 'traffic',
    why: '把 create 加 assemble 融成 slice，省掉一次中间拷贝。',
    doc: '39-fuse_create_assemble_to_slice.md',
  },
  BlockNzTensorViews: {
    tier: 'traffic',
    why: '为 NZ 分形布局建立视图，决定矩阵操作数按什么物理块读取。',
    doc: '15-block_nz_tensor_views.md',
  },
  InsertCommFence: {
    tier: 'traffic',
    why: '按两侧契约插入 GM 栅栏与 cache 作废。纯 wait 循环会被合并成一次整体 cache 作废，否则 (P-1) 次全 cache 冲刷会逐个发出。',
    doc: '51-insert_comm_fence.md',
  },
  LowerHostTensorCollectives: {
    tier: 'traffic',
    why: '把 host 侧集合通信下降成内置 kernel 调用，通信量与分块策略在这里定型。',
    doc: '46-lower_host_tensor_collectives.md',
  },
  SynthesizeAllReduceSignals: {
    tier: 'traffic',
    why: '为隐式 allreduce 合成信号缓冲。同一数据缓冲谱系共用一个信号，避免每次调用都新开一块。',
    doc: '44-synthesize_allreduce_signals.md',
  },
};

/** The profile for a Pass; every Pass has one, defaulting to `form`. */
export function perfOf(pass) {
  return PERF[pass] || { tier: 'form' };
}

export function perfTier(id) {
  return PERF_TIERS.find((t) => t.id === id) || PERF_TIERS[PERF_TIERS.length - 1];
}

// ---------------------------------------------------------------------------
// perf_hints.log
// ---------------------------------------------------------------------------

// `[perf_hint <code>] <emitter>: <message> at <path>:<line>:<col>`
//
// The emitter name is in the log, so a hint is attributed to a Pass by what the
// compiler wrote rather than by guessing from the code prefix. An emitter that
// is not a Pass in this run (the post-pipeline verifier checks) is kept and
// marked as such, because "this hint is not any Pass's doing" is the single
// most useful thing to say about the 197 `PH001`s.
const HINT_RE = /^\[perf_hint (\S+)\]\s+([A-Za-z0-9_]+):\s+([\s\S]*?)\s+at\s+(\S+?):(\d+):(\d+)\s*$/;

/**
 * Drop the absolute build prefix, keeping the part that identifies the source.
 * These logs were produced on someone's build machine and carry their home
 * directory; none of that belongs in a data file that ships with the demo.
 */
function trimLocation(p) {
  const norm = p.replace(/\\/g, '/');
  const m = norm.match(/\/(pypto-lib|pypto|pto|models|tests|examples)\/(.*)$/);
  return m ? m[1] + '/' + m[2] : norm.split('/').slice(-2).join('/');
}

/** `(N occurrences at this source location)` - the compiler already folded these. */
function takeOccurrences(message) {
  const m = message.match(/\s*\((\d+) occurrences at this source location\)\s*$/);
  if (!m) return { text: message.trim(), occurrences: 1 };
  return { text: message.slice(0, m.index).trim(), occurrences: Number(m[1]) };
}

/**
 * Parse a `perf_hints.log`. Returns one record per line plus the unparsed
 * remainder: a silently dropped line would understate the count, and that
 * count is the one number this whole feature rests on.
 */
export function parsePerfHints(text) {
  const records = [];
  const unparsed = [];
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) continue;
    const m = HINT_RE.exec(line);
    if (!m) { unparsed.push(line); continue; }
    const taken = takeOccurrences(m[3]);
    records.push({
      code: m[1],
      emitter: m[2],
      message: taken.text,
      occurrences: taken.occurrences,
      at: trimLocation(m[4]) + ':' + m[5],
    });
  }
  return { records, unparsed };
}

/**
 * Group parsed hints by emitter, splitting the ones a Pass in this run produced
 * from the ones that came from somewhere else.
 *
 * `passNames` is the set of Pass names actually in this run, so the split is
 * decided by the run itself rather than by a hardcoded list of verifier names.
 */
export function groupPerfHints(records, passNames) {
  const byEmitter = new Map();
  for (const r of records) {
    if (!byEmitter.has(r.emitter)) {
      byEmitter.set(r.emitter, {
        emitter: r.emitter,
        isPass: passNames.has(r.emitter),
        lines: 0,
        occurrences: 0,
        codes: {},
        sites: [],
      });
    }
    const g = byEmitter.get(r.emitter);
    g.lines += 1;
    g.occurrences += r.occurrences;
    g.codes[r.code] = (g.codes[r.code] || 0) + 1;
    g.sites.push(r);
  }
  // Busiest emitter first; a tie keeps name order so rebuilds are stable.
  return [...byEmitter.values()].sort((a, b) => b.lines - a.lines || a.emitter.localeCompare(b.emitter));
}
