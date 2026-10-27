/* QK Work Assignment for decode_sparse_attn_csa.py.
   Work items -> qk_plan reorder -> qk_order -> SPMD block assignment.
   Pure data functions first; one renderer that owns only its own host element. */
(function (global) {
  'use strict';

  const TOKENS = 5;
  const SOURCE_TOKENS = 256;
  const TOKENS_PER_REQUEST = 8;
  const SPARSE_BLOCKS = 5;
  const ITEM_COUNT = TOKENS * SPARSE_BLOCKS;
  const WINDOW_SB = 0;                       // SB0 is the sliding-window block, always live
  const NUM_QK_CORES = 24;                   // decode_sparse_attn_csa.py L67
  const WIN = 128, IDX_TOPK = 512, ATTN_K_TILE = 128, COMPRESS_RATIO = 4;
  const POSITIONS = [7, 1023, 2047, 8192];

  // Fixed teaching mask (1 = valid). Rows are tokens, columns are sparse blocks.
  const DEFAULT_MASK = Object.freeze([
    Object.freeze([1, 1, 0, 0, 0]),
    Object.freeze([1, 0, 0, 0, 0]),
    Object.freeze([1, 1, 1, 0, 0]),
    Object.freeze([1, 0, 1, 0, 0]),
    Object.freeze([1, 1, 0, 0, 0])
  ]);
  const DEFAULT_SELECTION = 11;              // Token 2 / Sparse Block 1

  const MODES = Object.freeze({
    teaching: Object.freeze({ tokens: TOKENS, blocks: NUM_QK_CORES, button: 'Teaching Example', badge: '5 Query 教学窗口 · Synthetic idx_topk' }),
    source: Object.freeze({ tokens: SOURCE_TOKENS, blocks: NUM_QK_CORES, button: 'Source-scale Example', badge: '默认规模 · Synthetic idx_topk · 非 Runtime Trace' })
  });

  const SUB_FILE = 'decode_sparse_attn_csa.py';
  const REPO_BLOB = 'https://github.com/yuanfeng219-arch/PyPTO3/blob/main/Data/DeepseekV4/deepseek_v4_flash_dspark/';

  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  /* ------------------------------------------------------------------ data */

  function syntheticTopk(position) {
    const upper = Math.max(0, Math.floor((position + 1) / COMPRESS_RATIO));
    const count = Math.min(IDX_TOPK, upper);
    const values = Array(IDX_TOPK).fill(-1);
    for (let i = 0; i < count; i++) values[i] = (i * 37) % upper;
    return values;
  }

  function buildWorkItems(input) {
    const legacyMask = Array.isArray(input) ? input : null;
    const count = legacyMask ? legacyMask.length : MODES[input.mode].tokens;
    const items = [];
    for (let token = 0; token < count; token++) {
      const requestIndex = Math.floor(token / TOKENS_PER_REQUEST);
      const positionId = legacyMask ? null : input.position + token % TOKENS_PER_REQUEST;
      const indices = legacyMask ? null : syntheticTopk(positionId);
      const upper = positionId == null ? null : Math.floor((positionId + 1) / COMPRESS_RATIO);
      for (let sparseBlock = 0; sparseBlock < SPARSE_BLOCKS; sparseBlock++) {
        const slotStart = (sparseBlock - 1) * ATTN_K_TILE;
        const validKvCount = sparseBlock === WINDOW_SB ? Math.min(WIN, (positionId ?? WIN - 1) + 1)
          : legacyMask ? (legacyMask[token][sparseBlock] ? ATTN_K_TILE : 0)
          : indices.slice(slotStart, slotStart + ATTN_K_TILE).filter(index => index >= 0 && index < upper).length;
        const override = !legacyMask && input.overrides?.get(token * SPARSE_BLOCKS + sparseBlock);
        const valid = legacyMask ? !!legacyMask[token][sparseBlock] : override == null ? sparseBlock === WINDOW_SB || validKvCount > 0 : override;
        items.push({
          id: token * SPARSE_BLOCKS + sparseBlock,   // workItemId = token * SPARSE_BLOCKS + sb
          token,
          requestIndex,
          positionId,
          rank: legacyMask ? null : input.rank,
          scenarioId: legacyMask ? 'legacy' : input.mode + '-position-' + input.position,
          sparseBlock,
          valid,
          validKvCount,
          validityReason: legacyMask ? '教学 mask' : override == null ? sparseBlock === WINDOW_SB ? 'SB0：滑动窗口固定有效' : validKvCount ? 'Synthetic idx_topk 在本块含 ' + validKvCount + ' 个满足范围的索引' : '本块没有满足范围的 Synthetic idx_topk 索引' : 'Advanced Simulation 手动覆盖',
          originalPosition: token * SPARSE_BLOCKS + sparseBlock
        });
      }
    }
    return items;
  }

  function deriveScenario(input) {
    const items = buildWorkItems(input), queue = buildQueue(items), byId = new Map(items.map(item => [item.id, item]));
    const blocks = assignToBlocks(queue, NUM_QK_CORES);
    blocks.forEach(block => { block.validCount = block.entries.filter(entry => byId.get(entry.id).valid).length; block.invalidCount = block.entries.length - block.validCount; });
    return { items, queue, blocks, blockCount: NUM_QK_CORES, byId, validCount: items.filter(item => item.valid).length, input };
  }

  // qk_plan: pass 1 writes valid tiles in token-major order, pass 2 appends the rest.
  // Invalid items are never dropped, only pushed to the tail.
  function buildQueue(items) {
    const valid = items.filter(item => item.valid).map(item => item.id);
    const invalid = items.filter(item => !item.valid).map(item => item.id);
    return [...valid, ...invalid];
  }

  // qk_flat = qk_core + qk_it * blockCount  =>  block = pos % N, iteration = floor(pos / N).
  function assignToBlocks(queue, blockCount) {
    const blocks = Array.from({ length: blockCount }, (_, index) => ({ index, entries: [] }));
    queue.forEach((id, queuePosition) => {
      const block = queuePosition % blockCount;
      blocks[block].entries.push({ id, queuePosition, iteration: Math.floor(queuePosition / blockCount) });
    });
    return blocks;
  }

  function getWorkItemDetails(id, queue, blockCount) {
    const queuePosition = queue.indexOf(id);
    return {
      id,
      token: Math.floor(id / SPARSE_BLOCKS),
      sparseBlock: id % SPARSE_BLOCKS,
      originalPosition: id,
      queuePosition,
      block: queuePosition % blockCount,
      iteration: Math.floor(queuePosition / blockCount),
      blockCount,
      qkFlat: (queuePosition % blockCount) + Math.floor(queuePosition / blockCount) * blockCount
    };
  }

  function getBlockDetails(index, items, queue, blockCount) {
    const byId = new Map(items.map(item => [item.id, item]));
    const entries = [];
    for (let queuePosition = index; queuePosition < queue.length; queuePosition += blockCount) {
      entries.push({
        id: queue[queuePosition],
        queuePosition,
        iteration: Math.floor(queuePosition / blockCount),
        valid: !!byId.get(queue[queuePosition])?.valid
      });
    }
    return {
      index,
      blockCount,
      entries,
      // qk_lane_iters = (qk_items - qk_core + NUM_QK_CORES - 1) // NUM_QK_CORES  (L213)
      laneIterations: Math.max(0, Math.floor((queue.length - index + blockCount - 1) / blockCount)),
      validCount: entries.filter(entry => entry.valid).length,
      invalidCount: entries.filter(entry => !entry.valid).length
    };
  }

  /* --------------------------------------------------------------- excerpts */

  const excerptCache = new Map();
  function excerpt(file, start, end) {
    const key = file + ':' + start + ':' + end;
    if (excerptCache.has(key)) return excerptCache.get(key);
    const text = global.PTO_CSA_SOURCE?.dependencies?.[file];
    const out = text
      ? text.split('\n').slice(start - 1, end).map((line, i) => (start + i) + '  ' + line).join('\n')
      : '';
    excerptCache.set(key, out);
    return out;
  }

  /* -------------------------------------------------------------- rendering */

  const CHAIN_VALID = ['KV Gather', 'QK Matmul', 'Scale / Bias / Softmax', 'PV Matmul', 'Save m/l/o'];
  const CHAIN_INVALID = ['Write Neutral m/l/o'];

  function create(host, options = {}) {
    const onSource = options.onSource || (() => {});
    const onSelection = options.onSelection || (() => {});
    const onContext = options.onContext || (() => {});
    const state = {
      mode: options.mode || 'source',
      position: options.position ?? 8192,
      rank: options.rank ?? 0,
      requestIndex: 0,
      selectedToken: null,
      overrides: new Map(),
      advanced: false,
      selection: options.selection === undefined ? null : options.selection,
      openLane: null,
      queueStart: 0,
      depOpen: true
    };
    let pendingFocus = null;
    function scenarioId() { return state.mode + '-position-' + state.position; }
    function select(selection) {
      const d = derive(); state.selection = selection;
      if (selection.type === 'item') {
        const item = d.byId.get(selection.id), details = getWorkItemDetails(selection.id, d.queue, d.blockCount);
        state.requestIndex = item.requestIndex; state.selectedToken = item.token; state.openLane = details.block;
        state.queueStart = Math.max(0, details.queuePosition - 12);
        onSelection({ ...selection, rank: state.rank, scenarioId: scenarioId(), requestIndex: item.requestIndex, tokenIndex: item.token, positionId: item.positionId, sparseBlockIndex: item.sparseBlock, valid: item.valid, validKvCount: item.validKvCount, queuePosition: details.queuePosition, laneId: details.block, laneIteration: details.iteration });
      } else { state.selectedToken = null; onSelection({ ...selection, rank: state.rank, scenarioId: scenarioId(), laneId: selection.block }); }
      render();
    }

    const blockCount = () => NUM_QK_CORES;

    function derive() {
      return deriveScenario({mode:state.mode,position:state.position,rank:state.rank,overrides:state.overrides});
    }

    /* --- A. Work matrix ------------------------------------------------- */

    function matrixCell(item, ctx) {
      const classes = ['csa-qk-cell', item.valid ? 'is-valid' : 'is-invalid'];
      if (ctx.selectedId === item.id) classes.push('is-selected');
      else if (ctx.relatedIds.has(item.id)) classes.push('is-related');
      const label = 'Work Item ' + item.id + ' · Request ' + item.requestIndex + ' · Query ' + item.token + ' · Position ' + item.positionId + ' · SB' + item.sparseBlock + ' · ' + (item.valid ? 'valid' : 'invalid');
      const hover = 'Work Item ' + item.id + ' · T' + item.token + ' / SB' + item.sparseBlock + ' · ' + (item.valid ? 'valid' : 'invalid');
      const control = item.sparseBlock === WINDOW_SB
        ? '<span class="csa-qk-cell__lock" title="SB0 = Sliding Window，恒为 valid，不可切换" aria-hidden="true"><svg viewBox="0 0 12 12" width="9" height="9"><rect x="2" y="5.2" width="8" height="5.6" rx="1.2" fill="currentColor" opacity=".6"/><path d="M4 5.2V4a2 2 0 0 1 4 0v1.2" fill="none" stroke="currentColor" stroke-width="1.1" opacity=".6"/></svg></span>'
        : state.advanced ? '<button class="csa-qk-cell__toggle" type="button" data-qk-toggle="' + item.id + '" aria-pressed="' + item.valid + '" aria-label="Advanced Simulation：切换 Work Item ' + item.id + ' 的 valid / invalid 状态" title="Advanced Simulation：手动覆盖 valid / invalid"><span></span></button>' : '';
      return '<div class="' + classes.join(' ') + '" role="gridcell">' +
        '<button class="csa-qk-cell__select" type="button" data-qk-item="' + item.id + '" title="' + esc(hover) + '" aria-label="' + esc(label) + '" aria-pressed="' + (ctx.selectedId === item.id) + '"><b>' + item.id + '</b></button>' +
        control + '</div>';
    }

    function matrixSection(d, ctx) {
      let head = '<div class="csa-qk-matrix__corner" aria-hidden="true"></div>';
      for (let sb = 0; sb < SPARSE_BLOCKS; sb++) {
        head += '<div class="csa-qk-matrix__head' + (sb === WINDOW_SB ? ' is-window' : '') + '"><b>SB' + sb + '</b><span>' + (sb === WINDOW_SB ? 'Original KV' : 'Compressed KV') + '</span></div>';
      }
      let rows = '';
      const start=state.mode==='source'?state.requestIndex*TOKENS_PER_REQUEST:0;
      const end=Math.min(start+(state.mode==='source'?TOKENS_PER_REQUEST:TOKENS),d.items.length/SPARSE_BLOCKS);
      for (let token = start; token < end; token++) {
        rows += '<button class="csa-qk-matrix__row'+(state.selectedToken===token?' is-selected':'')+'" type="button" data-qk-token="'+token+'" aria-pressed="'+(state.selectedToken===token)+'" title="选择 Query '+token+' · Position '+d.byId.get(token*SPARSE_BLOCKS).positionId+'">Q'+token+'</button>';
        for (let sb = 0; sb < SPARSE_BLOCKS; sb++) rows += matrixCell(d.byId.get(token * SPARSE_BLOCKS + sb), ctx);
      }
      return '<section class="csa-qk-card csa-qk-card--matrix" aria-label="A · Work Generation">' +
        '<header><span class="csa-qk-step">A</span><h3>Work Generation · Request '+state.requestIndex+'</h3>' +
        '<p>每行一个 Query，每列一个 128 行 KV Sparse Block；格内数字是原始 Work Item ID。'+(state.mode==='source'?'当前显示 32 个 Request 中的 1 个，共 8 个 Query。':'教学窗口显示 5 个 Query。')+'</p></header>' +
        '<div class="csa-qk-matrix" role="grid" aria-label="'+(end-start)*SPARSE_BLOCKS+' 个可见 Work Item">' + head + rows + '</div>' +
        '<p class="csa-qk-hint">SB0 读取 Sliding Window / Original KV，固定有效；SB1–SB4 读取 Indexer 选出的 Compressed KV 索引。'+(state.advanced?' 当前为 Advanced Simulation，可手动覆盖有效性。':' 有效性由 Synthetic idx_topk 和 Position 推导。')+'</p>' +
        '</section>';
    }

    /* --- B. Reordered queue --------------------------------------------- */

    function chip(item, position, ctx, cls) {
      const classes = ['csa-qk-chip', item.valid ? 'is-valid' : 'is-invalid'];
      if (ctx.selectedId === item.id) classes.push('is-selected');
      else if (ctx.relatedIds.has(item.id)) classes.push('is-related');
      const label = '队列位置 ' + position + ' · Work Item ' + item.id + ' · Token ' + item.token + ' · SB' + item.sparseBlock + ' · ' + (item.valid ? 'valid' : 'invalid');
      const hover = 'queue position ' + position + ' · Work Item ' + item.id + ' · T' + item.token + ' / SB' + item.sparseBlock + ' · ' + (item.valid ? 'valid' : 'invalid');
      return '<button class="' + classes.join(' ') + (cls ? ' ' + cls : '') + '" type="button" data-qk-item="' + item.id + '" title="' + esc(hover) + '" aria-label="' + esc(label) + '" aria-pressed="' + (ctx.selectedId === item.id) + '">' +
        '<i>' + position + '</i><b>' + item.id + '</b></button>';
    }

    function queueSection(d, ctx) {
      const pageSize = state.mode === 'source' ? 48 : d.queue.length;
      const start = Math.min(state.queueStart, Math.max(0, d.queue.length - pageSize));
      const end = Math.min(d.queue.length, start + pageSize);
      const boundary = d.validCount;
      const visible = d.queue.slice(start, end);
      const chips = visible.map((id, offset) => chip(d.byId.get(id), start + offset, ctx)).join('');
      return '<section class="csa-qk-card csa-qk-card--queue" aria-label="B · qk_order">' +
        '<header><span class="csa-qk-step">B</span><h3>qk_order · valid 优先</h3>' +
        '<p>qk_plan 第一轮按原始顺序写 valid，第二轮追加 invalid；两类工作项都保留。</p></header>' +
        '<div class="csa-qk-queue-summary"><span class="is-valid">valid · ' + (boundary ? '0–' + (boundary - 1) : '空') + ' · ' + boundary + ' 项</span><span class="is-invalid">invalid · ' + (boundary < d.queue.length ? boundary + '–' + (d.queue.length - 1) : '空') + ' · ' + (d.queue.length - boundary) + ' 项</span></div>' +
        '<div class="csa-qk-queue-nav"><button type="button" data-qk-queue-step="-48"' + (start === 0 ? ' disabled' : '') + '>上一段</button><span>队列位置 ' + start + '–' + (end - 1) + ' / ' + d.queue.length + '</span><button type="button" data-qk-queue-step="48"' + (end === d.queue.length ? ' disabled' : '') + '>下一段</button></div>' +
        '<div class="csa-qk-lane__chips">' + chips + '</div>' +
        '<p class="csa-qk-hint">小字是 queue position，大字是 Work Item ID。选择 A 或 C 中的工作项会定位到这里。</p></section>';
    }

    /* --- C. Block assignment ------------------------------------------- */

    function blockSection(d, ctx) {
      const rows = d.blocks.map(block => {
        const selected = ctx.selectedBlock === block.index;
        const open = state.openLane === block.index;
        const first = block.entries[0], last = block.entries[block.entries.length - 1];
        const entries = open ? '<div class="csa-qk-block__lane">' + block.entries.map(entry =>
          '<span class="csa-qk-it">it ' + entry.iteration + '</span>' + chip(d.byId.get(entry.id), entry.queuePosition, ctx, 'csa-qk-chip--lane')).join('') + '</div>' : '';
        return '<div class="csa-qk-block' + (selected ? ' is-selected' : '') + (open ? ' is-open' : '') + '">' +
          '<button class="csa-qk-block__head" type="button" data-qk-block="' + block.index + '" aria-expanded="' + open + '" aria-pressed="' + selected + '"><b>Lane ' + block.index + '</b><i>' + block.entries.length + ' 项 · ' + block.validCount + ' valid / ' + block.invalidCount + ' invalid</i><small>qk_order[' + first.queuePosition + '] … [' + last.queuePosition + '] · ' + (open ? '收起' : '展开') + '</small></button>' + entries + '</div>';
      }).join('');
      return '<section class="csa-qk-card csa-qk-card--blocks" aria-label="C · SPMD Lane Assignment">' +
        '<header><span class="csa-qk-step">C</span><h3>SPMD Lane Assignment · 24 lanes</h3>' +
        '<p>一次 qk_pv Dispatch 展开 24 个逻辑 SPMD Block。Lane k 依次领取 qk_order[k + iteration × 24]。</p></header>' +
        '<div class="csa-qk-blocks">' + rows + '</div>' +
        '<p class="csa-qk-hint">Work Item 不是独立 Runtime Task；Iteration 是同一 Lane 循环领取的序号。</p></section>';
    }

    /* --- Inspector ------------------------------------------------------ */

    function itemInspector(d, details) {
      const item = d.byId.get(details.id), n = d.blockCount;
      const facts = [
        ['Scenario / Rank', item.scenarioId + ' / Rank ' + item.rank],
        ['Request / Query', 'Request ' + item.requestIndex + ' / Q' + item.token],
        ['Position ID', String(item.positionId)],
        ['Work Item / Sparse Block', item.id + ' / SB' + item.sparseBlock],
        ['KV 来源', item.sparseBlock === WINDOW_SB ? 'Original KV · Sliding Window' : 'Compressed KV · idx_topk slots ' + ((item.sparseBlock - 1) * ATTN_K_TILE) + '–' + (item.sparseBlock * ATTN_K_TILE - 1)],
        ['有效索引数', String(item.validKvCount)],
        ['Validity', item.valid ? 'valid' : 'invalid'],
        ['判定依据', item.validityReason],
        ['原始位置', String(item.originalPosition)],
        ['qk_order 位置', String(details.queuePosition) + '（qk_order[' + details.queuePosition + '] = ' + item.id + '）'],
        ['SPMD Lane / Iteration', details.block + ' / ' + details.iteration]
      ];
      const formula = [
        'workItemId = query × 5 + sb = ' + item.token + ' × 5 + ' + item.sparseBlock + ' = ' + item.id,
        'queuePosition = qk_order.indexOf(' + item.id + ') = ' + details.queuePosition,
        'lane = queuePosition % ' + n + ' = ' + details.block,
        'iteration = ⌊queuePosition / ' + n + '⌋ = ' + details.iteration,
        'qk_flat = lane + iteration × ' + n + ' = ' + details.qkFlat
      ].join('\n');
      return '<span class="csa-eyebrow">WORK ITEM ' + item.id + ' · ' + (item.valid ? 'VALID' : 'INVALID') + '</span>' +
        '<h2>Q' + item.token + ' · SB' + item.sparseBlock + '</h2>' +
        '<dl class="csa-qk-facts">' + facts.map(([k, v]) => '<dt>' + esc(k) + '</dt><dd>' + esc(v) + '</dd>').join('') + '</dl>' +
        '<h3 class="csa-qk-subhead">A → B → C · 同一 Work Item</h3><pre class="csa-qk-formula">' + esc(formula) + '</pre>' +
        (state.advanced && item.sparseBlock !== WINDOW_SB ? '<button class="btn btn-sm" type="button" data-qk-toggle="' + item.id + '">Advanced Simulation：切换为 ' + (item.valid ? 'invalid' : 'valid') + '</button>' : '') + pathBlock(item);
    }

    function pathBlock(item) {
      const chain = item.valid ? CHAIN_VALID : CHAIN_INVALID;
      const nodes = chain.map((step, i) => (i ? '<em>→</em>' : '') + '<span' + (item.valid ? '' : ' class="is-neutral"') + '>' + esc(step) + '</span>').join('');
      return '<h3 class="csa-qk-subhead">Block 内执行路径</h3>' +
        '<div class="csa-qk-chain">' + nodes + '</div>' +
        (item.valid
          ? '<p class="csa-qk-hint">含 <code>pl.pipeline(H // QK_M_TILE, stage=2)</code>（L281）· 源码声明的软件流水结构，不表示硬件时间、重叠时长或性能。</p>'
          : '<p class="csa-qk-hint">invalid 分支（L305–313）写入中性统计量：<code>mi = -3.0e38</code>、<code>li = 0</code>、<code>oi = 0</code>，不走 QK / PV 计算。</p>');
    }

    function blockInspector(d, details) {
      const positions = details.entries.map(entry => entry.queuePosition).join(', ') || '—';
      const ids = details.entries.map(entry => entry.id).join(', ') || '—';
      return '<span class="csa-eyebrow">SPMD BLOCK ' + details.index + ' · 共 ' + details.blockCount + '</span>' +
        '<h2>Block ' + details.index + '</h2>' +
        '<dl class="csa-qk-facts">' +
        [['领取的 Queue Position', positions],
         ['领取的 Work Items', ids],
         ['Iterations', String(details.entries.length) + '（qk_lane_iters = ' + details.laneIterations + '）'],
         ['valid / invalid', details.validCount + ' / ' + details.invalidCount]].map(([k, v]) => '<dt>' + esc(k) + '</dt><dd>' + esc(v) + '</dd>').join('') +
        '</dl>' +
        '<h3 class="csa-qk-subhead">采用的规则</h3><pre class="csa-qk-formula">' + esc([
          'block   = queuePosition % ' + details.blockCount,
          'iteration = ⌊queuePosition / ' + details.blockCount + '⌋',
          'qk_flat = qk_core + qk_it × ' + details.blockCount,
          'qk_lane_iters = (qk_items - qk_core + ' + details.blockCount + ' - 1) // ' + details.blockCount
        ].join('\n')) + '</pre>' +
        '<p class="csa-qk-hint">valid 数量可以数出来，但不能用来估算真实执行时间：invalid 项同样领取、同样写结果。</p>';
    }

    function evidence(d, details) {
      const isItem = state.selection?.type === 'item';
      const valid = isItem ? d.byId.get(state.selection.id).valid : true;
      const ranges = [
        [134, 196, 'qk_plan · 有效性、qk_order 两轮写入'],
        [208, 224, 'qk_pv 派发、显式依赖、Block 领取'],
        [224, 314, isItem ? (valid ? '内部计算路径 · valid 分支（224–304）' : '内部计算路径 · invalid 分支（305–313）') : '内部计算路径 · valid / invalid 两个分支']
      ];
      return '<h3 class="csa-qk-subhead">Source Evidence · 真实源码行段</h3>' +
        ranges.map(([a, z, note]) =>
          '<details class="csa-qk-source"><summary><span>' + SUB_FILE + ' : ' + a + '–' + z + '</span><i>' + esc(note) + '</i></summary>' +
          '<pre>' + esc(excerpt(SUB_FILE, a, z)) + '</pre>' +
          '<button class="csa-qk-node__link" type="button" data-qk-source="' + SUB_FILE + ':' + a + ':' + z + '">在编辑器中打开 ' + a + '–' + z + ' ↗</button></details>').join('') +
        '<a class="csa-qk-link" href="' + REPO_BLOB + SUB_FILE + '#L134-L196" target="_blank" rel="noopener noreferrer">在 GitHub 查看 ' + SUB_FILE + ' ↗</a>';
    }

    function terms(d) {
      return '<h3 class="csa-qk-subhead">术语</h3><dl class="csa-qk-facts csa-qk-facts--terms">' +
        [['Dispatch', 'qk_pv 的一次 pl.spmd 派发（逻辑任务）'],
         ['SPMD Block', '该派发展开的逻辑执行块 · 当前 ' + d.blockCount],
         ['Work Item', '(token, sparseBlock)，不是独立 Runtime Task'],
         ['Iteration', '某个 Block 领取工作项的第几次循环']]
          .map(([k, v]) => '<dt>' + esc(k) + '</dt><dd>' + esc(v) + '</dd>').join('') + '</dl>';
    }

    function inspector(d, ctx) {
      const body = !state.selection
        ? '<h3>qk_pv Work Assignment</h3><p class="csa-qk-hint">选择 Work Item 或 SPMD Block，查看 qk_order 中的位置与分配关系。</p>'
        : state.selection.type === 'block'
        ? blockInspector(d, getBlockDetails(state.selection.block, d.items, d.queue, d.blockCount))
        : itemInspector(d, getWorkItemDetails(state.selection.id, d.queue, d.blockCount));
      const dispatch = '<p class="csa-qk-dispatch"><b>DISPATCH</b> qk_pv · ' + d.blockCount + ' SPMD Blocks' +
        (state.mode === 'teaching' ? ' · 教学示意配置' : ' · NUM_QK_CORES = 24') + '</p>';
      return '<aside class="csa-qk-inspector" aria-label="QK Work 对象详情">' + dispatch + body + evidence(d) + terms(d) + '</aside>';
    }

    /* --- Dependency context -------------------------------------------- */

    function sourceLink(label, file, a, z) {
      return '<button class="csa-qk-srclink" type="button" data-qk-source="' + file + ':' + a + ':' + z + '">' + esc(label) + '</button>';
    }

    function dependencyBar() {
      return '<section class="csa-qk-depbar" aria-label="Dependency Context">' +
        '<span class="csa-qk-depbar__label"><b>显式任务依赖</b><i>静态源码模型</i></span>' +
        '<div class="csa-qk-pill"><b>qk_plan_tid</b><small>csa_slots_build_valid_qk_plan · L143</small></div>' +
        '<em class="csa-qk-arrow">→</em>' +
        '<div class="csa-qk-pill is-target"><b>qk_pv</b><small>pl.spmd · 一次 Dispatch</small></div>' +
        '<em class="csa-qk-arrow">←</em>' +
        '<div class="csa-qk-pill"><b>cache_ready_dep</b><small>pl.system.task_dummy · L361</small></div>' +
        '<button class="csa-qk-depbar__toggle" type="button" data-qk-dep-toggle aria-expanded="' + state.depOpen + '">' +
        (state.depOpen ? '收起 2 个输入 ▴' : '展开 2 个输入 ▾') + '</button>' +
        sourceLink('decode_csa.py : 384–393 · 调用处 ↗', 'decode_csa.py', 384, 393) +
        '<div class="csa-qk-depbar__detail"' + (state.depOpen ? '' : ' hidden') + '>' +
        '<span class="csa-qk-depbar__sub"><b>cache_ready_dep</b> = task_dummy(deps=[<b>ori_cache_write_tid</b>, <b>cmp_cache_write_tid</b>])</span>' +
        sourceLink('L361 ↗', 'decode_csa.py', 361, 361) +
        '<span class="csa-qk-depbar__sub">ori_cache_write_tid · csa_cache_writeback</span>' +
        sourceLink('L323 ↗', 'decode_csa.py', 323, 333) +
        '<span class="csa-qk-depbar__sub">cmp_cache_write_tid · compressor_ratio4</span>' +
        sourceLink('L353 ↗', 'decode_csa.py', 353, 360) +
        '</div>' +
        '<p class="csa-qk-hint">只画本段源码显式声明的依赖，不是完整 Runtime Task Dependency Graph；indexer cache 的写入走 idx_cache_write_tid，不并入 cache_ready_dep。箭头长度与位置不代表耗时。</p>' +
        '</section>';
    }

    /* --- shell ---------------------------------------------------------- */

    function controls(d) {
      const buttons = Object.entries(MODES).map(([key, mode]) =>
        '<button class="btn btn-sm' + (state.mode === key ? ' is-selected' : '') + '" type="button" data-qk-mode="' + key + '" aria-pressed="' + (state.mode === key) + '">' + esc(mode.button) + '</button>').join('');
      const positions = POSITIONS.map(position => '<button class="btn btn-sm' + (state.position === position ? ' is-selected' : '') + '" type="button" data-qk-position="' + position + '" aria-pressed="' + (state.position === position) + '">' + position + '</button>').join('');
      const requests = state.mode === 'source' ? '<label>Request <select data-qk-request>' + Array.from({length:32},(_,i)=>'<option value="'+i+'"'+(state.requestIndex===i?' selected':'')+'>'+i+' · Q'+(i*8)+'–'+(i*8+7)+'</option>').join('')+'</select></label>' : '';
      const first=d.byId.get(state.requestIndex*TOKENS_PER_REQUEST*SPARSE_BLOCKS) || d.items[0];
      const visibleCompressed=Math.floor((first.positionId+1)/COMPRESS_RATIO);
      const firstMask=Array.from({length:SPARSE_BLOCKS},(_,sb)=>Number(d.byId.get(first.id+sb).valid)).join(' ');
      return '<div class="csa-qk-controls">' +
        '<div class="segmented-control segmented-control-muted" role="group" aria-label="Query 规模">' + buttons + '</div>' +
        '<span class="csa-qk-badge">' + esc(MODES[state.mode].badge) + '</span>' +
        '<div class="csa-qk-control-row"><span>Position 起点</span>' + positions + '<span>Rank</span>' + [0,1].map(rank=>'<button class="btn btn-sm'+(state.rank===rank?' is-selected':'')+'" type="button" data-qk-rank="'+rank+'" aria-pressed="'+(state.rank===rank)+'">'+rank+'</button>').join('') + requests + '</div>' +
        '<span class="csa-qk-meta">' + (d.items.length/SPARSE_BLOCKS) + ' Queries × 5 Sparse Blocks = ' + d.items.length + ' Work Items · ' + d.validCount + ' valid / ' + (d.items.length-d.validCount) + ' invalid · 1 Dispatch → 24 SPMD Lanes</span>' +
        '<span class="csa-qk-meta">模型配置：TP=2，H=64；Workload / Runtime 输入示例：DECODE_BATCH=64、DECODE_SEQ=8、Position；算子常量：WIN=128、IDX_TOPK=512、ATTN_K_TILE=128、QK_M_TILE=32、NUM_QK_CORES=24。</span>' +
        '<span class="csa-qk-meta">推导：64 / 2 = 32 Requests / Rank；32 × 8 = 256 Queries / Rank；(128 + 512) / 128 = 5 SB；' + (d.items.length/SPARSE_BLOCKS) + ' × 5 = ' + d.items.length + ' Work Items。T 随 Runtime 输入变化。</span>' +
        '<span class="csa-qk-meta">Position ' + first.positionId + ' → 可用压缩位置 ' + visibleCompressed + ' → Synthetic idx_topk 有效槽 ' + Math.min(IDX_TOPK,visibleCompressed) + ' → Q' + first.token + ' mask [' + firstMask + '] → qk_order ' + d.validCount + ' valid / ' + (d.items.length-d.validCount) + ' invalid → 24 Lane 分配。</span>' +
        '<span class="csa-qk-meta">idx_topk 为可复现 Synthetic Example：有效槽依次使用确定性范围内索引，其余填 -1。真实索引由 Indexer 相关性计算产生；-1 表示无索引，不是 KV 值 0 或低相关性。这里不是 Runtime Trace。</span>' +
        '<button class="btn btn-ghost btn-sm" type="button" data-qk-advanced aria-pressed="'+state.advanced+'">Advanced Simulation '+(state.advanced?'开':'关')+'</button>' +
        (state.advanced?'<button class="btn btn-ghost btn-sm" type="button" data-qk-reset>清除手动覆盖</button>':'') + '</div>';
    }

    function render() {
      const d = derive();
      const details = state.selection?.type === 'item' ? getWorkItemDetails(state.selection.id, d.queue, d.blockCount) : null;
      const selectedBlock = state.selection?.type === 'block' ? state.selection.block : details?.block;
      // Selecting a block highlights all of its items; selecting an item only itself.
      const relatedIds = new Set(state.selection?.type === 'block' && d.blocks[selectedBlock]
        ? d.blocks[selectedBlock].entries.map(entry => entry.id) : []);
      const ctx = { selectedId: state.selection?.type === 'item' ? state.selection.id : null, selectedBlock, relatedIds };

      const scroller = host.querySelector('.csa-qk-analysis');
      const scrollTop = scroller ? scroller.scrollTop : 0;

      host.innerHTML = '<div class="csa-qk">' +
        '<div class="csa-qk-analysis">' + dependencyBar() +
        '<div class="csa-qk-top">' + matrixSection(d, ctx) + queueSection(d, ctx) + '</div>' +
        blockSection(d, ctx) + '</div>' +
        '<div class="csa-qk-side">' + inspector(d, ctx) + '</div>' +
        '</div>';
      host.insertAdjacentHTML('afterbegin', controls(d));

      const nextScroller = host.querySelector('.csa-qk-analysis');
      if (nextScroller) nextScroller.scrollTop = scrollTop;
      if (pendingFocus) {
        const target = host.querySelector(pendingFocus);
        if (target) target.focus({ preventScroll: true });
        pendingFocus = null;
      }
    }

    /* --- interaction ---------------------------------------------------- */

    function toggleValid(id) {
      if (!state.advanced || id % SPARSE_BLOCKS === WINDOW_SB) return;
      const item = derive().byId.get(id);
      state.overrides.set(id, !item.valid);
      pendingFocus = '[data-qk-item="' + id + '"]';
      if (state.selection?.type === 'item' && state.selection.id === id) select(state.selection);
      else render();
    }
    function changeContext() {
      state.overrides.clear(); state.selection = null; state.selectedToken = null;
      state.openLane = null; state.queueStart = 0; state.requestIndex = 0;
      onContext({rank:state.rank,scenarioId:scenarioId()}); render();
    }
    host.addEventListener('click', event => {
      const target = event.target;
      const toggle = target.closest('[data-qk-toggle]');
      if (toggle) { toggleValid(Number(toggle.dataset.qkToggle)); return; }
      const source = target.closest('[data-qk-source]');
      if (source) { const [file,a,z] = source.dataset.qkSource.split(':'); onSource(file,Number(a),Number(z)); return; }
      if (target.closest('[data-qk-dep-toggle]')) { state.depOpen = !state.depOpen; render(); return; }
      if (target.closest('[data-qk-reset]')) { state.overrides.clear(); if(state.selection?.type==='item')select(state.selection);else render(); return; }
      if (target.closest('[data-qk-advanced]')) { state.advanced = !state.advanced; if (!state.advanced) state.overrides.clear(); if(state.selection?.type==='item')select(state.selection);else render(); return; }
      const mode = target.closest('[data-qk-mode]');
      if (mode) { if(state.mode!==mode.dataset.qkMode){state.mode=mode.dataset.qkMode;changeContext();} return; }
      const position = target.closest('[data-qk-position]');
      if (position) { const value=Number(position.dataset.qkPosition); if(state.position!==value){state.position=value;changeContext();} return; }
      const rank = target.closest('[data-qk-rank]');
      if (rank) { const value=Number(rank.dataset.qkRank); if(state.rank!==value){state.rank=value;changeContext();} return; }
      const page = target.closest('[data-qk-queue-step]');
      if (page) { state.queueStart=Math.max(0,Math.min(derive().queue.length-48,state.queueStart+Number(page.dataset.qkQueueStep)));render();return; }
      const token = target.closest('[data-qk-token]');
      if (token) { state.selectedToken=Number(token.dataset.qkToken); state.requestIndex=Math.floor(state.selectedToken/TOKENS_PER_REQUEST); render(); return; }
      const block = target.closest('[data-qk-block]');
      if (block) { const id=Number(block.dataset.qkBlock); state.openLane=state.openLane===id?null:id; select({type:'block',block:id}); return; }
      const item = target.closest('[data-qk-item]');
      if (item) { select({type:'item',id:Number(item.dataset.qkItem)}); }
    });
    host.addEventListener('change', event=>{
      if(event.target.matches('[data-qk-request]')){state.requestIndex=Number(event.target.value);state.selectedToken=null;render();}
    });

    render();

    return {
      render,
      setSelection: selection => {
        if (JSON.stringify(selection) === JSON.stringify(state.selection)) return;
        state.selection = selection ? {...selection} : null;
        if (selection?.type === 'item') {
          const d=derive(), item=d.byId.get(selection.id);
          if(item){ const details=getWorkItemDetails(selection.id,d.queue,d.blockCount);state.requestIndex=item.requestIndex;state.selectedToken=item.token;state.openLane=details.block;state.queueStart=Math.max(0,details.queuePosition-12); }
        } else if (selection?.type === 'block') state.openLane=selection.block;
        render();
      },
      getState: () => ({mode:state.mode,position:state.position,rank:state.rank,requestIndex:state.requestIndex,selection:state.selection?{...state.selection}:null})
    };
  }

  global.PtoCsaQkWork = Object.freeze({
    create,
    buildWorkItems,
    buildQueue,
    assignToBlocks,
    getWorkItemDetails,
    getBlockDetails,
    DEFAULT_MASK,
    DEFAULT_SELECTION,
    syntheticTopk,
    deriveScenario,
    MODES,
    constants: { TOKENS, SOURCE_TOKENS, TOKENS_PER_REQUEST, SPARSE_BLOCKS, ITEM_COUNT, NUM_QK_CORES, WINDOW_SB, WIN, IDX_TOPK, ATTN_K_TILE, COMPRESS_RATIO, POSITIONS }
  });
})(typeof window !== 'undefined' ? window : globalThis);
