/* QK Work Assignment — local experiment for decode_sparse_attn_csa.py.
   Work items -> qk_plan reorder -> qk_order -> SPMD block assignment.
   Pure data functions first; one renderer that owns only its own host element. */
(function (global) {
  'use strict';

  const TOKENS = 5;
  const SPARSE_BLOCKS = 5;
  const ITEM_COUNT = TOKENS * SPARSE_BLOCKS;
  const WINDOW_SB = 0;                       // SB0 is the sliding-window block, always live
  const NUM_QK_CORES = 24;                   // decode_sparse_attn_csa.py L67

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
    teaching: Object.freeze({ blocks: 4, button: '教学示例 · 4 Blocks', badge: '教学示例 · 4 Blocks · 非真实 CSA 配置' }),
    source: Object.freeze({ blocks: 24, button: '源码 Block 数 · 24', badge: '源码 Block 数 · 24 · 示例输入' })
  });

  const SUB_FILE = 'decode_sparse_attn_csa.py';
  const REPO_BLOB = 'https://github.com/yuanfeng219-arch/PyPTO3/blob/main/Data/DeepseekV4/deepseek_v4_flash_dspark/';

  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  /* ------------------------------------------------------------------ data */

  function buildWorkItems(mask) {
    const items = [];
    for (let token = 0; token < TOKENS; token++) {
      for (let sparseBlock = 0; sparseBlock < SPARSE_BLOCKS; sparseBlock++) {
        items.push({
          id: token * SPARSE_BLOCKS + sparseBlock,   // workItemId = token * SPARSE_BLOCKS + sb
          token,
          sparseBlock,
          valid: !!mask[token][sparseBlock],
          originalPosition: token * SPARSE_BLOCKS + sparseBlock
        });
      }
    }
    return items;
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
    const state = {
      mode: 'teaching',
      mask: DEFAULT_MASK.map(row => row.slice()),
      selection: { type: 'item', id: DEFAULT_SELECTION },
      depOpen: true
    };
    let pendingFocus = null;

    const blockCount = () => MODES[state.mode].blocks;

    function derive() {
      const items = buildWorkItems(state.mask);
      const queue = buildQueue(items);
      const byId = new Map(items.map(item => [item.id, item]));
      const blocks = assignToBlocks(queue, blockCount());
      blocks.forEach(block => {
        block.validCount = block.entries.filter(entry => byId.get(entry.id).valid).length;
        block.invalidCount = block.entries.length - block.validCount;
      });
      return { items, queue, blocks, blockCount: blockCount(), byId };
    }

    /* --- A. Work matrix ------------------------------------------------- */

    function matrixCell(item, ctx) {
      const classes = ['csa-qk-cell', item.valid ? 'is-valid' : 'is-invalid'];
      if (ctx.selectedId === item.id) classes.push('is-selected');
      else if (ctx.relatedIds.has(item.id)) classes.push('is-related');
      const label = 'Work Item ' + item.id + ' · Token ' + item.token + ' · SB' + item.sparseBlock + ' · ' + (item.valid ? 'valid' : 'invalid');
      const hover = 'Work Item ' + item.id + ' · T' + item.token + ' / SB' + item.sparseBlock + ' · ' + (item.valid ? 'valid' : 'invalid');
      const control = item.sparseBlock === WINDOW_SB
        ? '<span class="csa-qk-cell__lock" title="SB0 = Sliding Window，恒为 valid，不可切换" aria-hidden="true"><svg viewBox="0 0 12 12" width="9" height="9"><rect x="2" y="5.2" width="8" height="5.6" rx="1.2" fill="currentColor" opacity=".6"/><path d="M4 5.2V4a2 2 0 0 1 4 0v1.2" fill="none" stroke="currentColor" stroke-width="1.1" opacity=".6"/></svg></span>'
        : '<button class="csa-qk-cell__toggle" type="button" data-qk-toggle="' + item.id + '" aria-pressed="' + item.valid + '" aria-label="切换 Work Item ' + item.id + ' 的 valid / invalid 状态" title="切换 valid / invalid"><span></span></button>';
      return '<div class="' + classes.join(' ') + '" role="gridcell">' +
        '<button class="csa-qk-cell__select" type="button" data-qk-item="' + item.id + '" title="' + esc(hover) + '" aria-label="' + esc(label) + '" aria-pressed="' + (ctx.selectedId === item.id) + '"><b>' + item.id + '</b></button>' +
        control + '</div>';
    }

    function matrixSection(d, ctx) {
      let head = '<div class="csa-qk-matrix__corner" aria-hidden="true"></div>';
      for (let sb = 0; sb < SPARSE_BLOCKS; sb++) {
        head += '<div class="csa-qk-matrix__head' + (sb === WINDOW_SB ? ' is-window' : '') + '"><b>SB' + sb + '</b><span>' + (sb === WINDOW_SB ? 'Window' : 'Sparse') + '</span></div>';
      }
      let rows = '';
      for (let token = 0; token < TOKENS; token++) {
        rows += '<div class="csa-qk-matrix__row" aria-hidden="true">T' + token + '</div>';
        for (let sb = 0; sb < SPARSE_BLOCKS; sb++) rows += matrixCell(d.byId.get(token * SPARSE_BLOCKS + sb), ctx);
      }
      return '<section class="csa-qk-card csa-qk-card--matrix" aria-label="A · Original Work Matrix">' +
        '<header><span class="csa-qk-step">A</span><h3>Original Work Matrix</h3>' +
        '<p>行 = Token，列 = Sparse Block；每格是一个 (token, sparseBlock) Work Item，格内数字为扁平 ID。</p></header>' +
        '<div class="csa-qk-matrix" role="grid" aria-label="25 个 Work Item">' + head + rows + '</div>' +
        '<p class="csa-qk-hint">SB0 为 Sliding Window，恒为 valid；SB1–SB4 可在格内点右上角状态点切换 valid / invalid。</p>' +
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
      const validPart = d.queue.slice(0, d.queue.filter(id => d.byId.get(id).valid).length);
      const invalidPart = d.queue.slice(validPart.length);
      const lane = (title, range, ids, kind, offset) => '<div class="csa-qk-lane is-' + kind + '">' +
        '<div class="csa-qk-lane__label"><b>' + title + '</b><i>' + range + '</i><span>' + ids.length + ' 项</span></div>' +
        '<div class="csa-qk-lane__chips">' + (ids.length
          ? ids.map((id, i) => chip(d.byId.get(id), offset + i, ctx)).join('')
          : '<span class="csa-qk-empty">无</span>') + '</div></div>';
      return '<section class="csa-qk-card csa-qk-card--queue" aria-label="B · Reordered Work Queue">' +
        '<header><span class="csa-qk-step">B</span><h3>Reordered Work Queue · qk_order</h3>' +
        '<p>qk_plan 用同一个写游标：第一轮写入 valid，第二轮把 invalid 追加到尾部。</p></header>' +
        '<div class="csa-qk-lanes">' +
        lane('valid 段', (validPart.length ? '0–' + (validPart.length - 1) : '空'), validPart, 'valid', 0) +
        '<div class="csa-qk-boundary"><span>valid 段结束 · invalid 从这里开始追加</span></div>' +
        lane('invalid 段', (invalidPart.length ? validPart.length + '–' + (d.queue.length - 1) : '空'), invalidPart, 'invalid', validPart.length) +
        '</div>' +
        '<p class="csa-qk-hint">小字 = queue position（qk_order 下标），大字 = workItemId。invalid 项仍在队列里，qk_pv 仍会领取并走 invalid 分支。</p>' +
        '</section>';
    }

    /* --- C. Block assignment ------------------------------------------- */

    function blockSection(d, ctx) {
      const rows = d.blocks.map(block => {
        const selected = ctx.selectedBlock === block.index;
        const lane = block.entries.length
          ? block.entries.map(entry => {
            const item = d.byId.get(entry.id);
            return '<span class="csa-qk-it">it ' + entry.iteration + '</span>' + chip(item, entry.queuePosition, ctx, 'csa-qk-chip--lane');
          }).join('')
          : '<span class="csa-qk-empty">qk_lane_iters = 0 · 不领取任何工作项</span>';
        return '<div class="csa-qk-block' + (selected ? ' is-selected' : '') + '">' +
          '<button class="csa-qk-block__head" type="button" data-qk-block="' + block.index + '" aria-pressed="' + selected + '" title="Block ' + block.index + ' · 领取 ' + block.entries.length + ' 个工作项 · queue position ' + (block.entries.map(entry => entry.queuePosition).join(', ') || '—') + '">' +
          '<b>Block ' + block.index + '</b><i>' + block.entries.length + ' 项 · ' + block.validCount + ' valid / ' + block.invalidCount + ' invalid</i></button>' +
          '<div class="csa-qk-block__lane">' + lane + '</div></div>';
      }).join('');
      return '<section class="csa-qk-card csa-qk-card--blocks" aria-label="C · SPMD Block Assignment">' +
        '<header><span class="csa-qk-step">C</span><h3>SPMD Block Assignment</h3>' +
        '<p>1 次 Dispatch <code>qk_pv</code> → ' + d.blockCount + ' 个逻辑 SPMD Block；Block k 领取 qk_order[k]、qk_order[k + ' + d.blockCount + ']、…</p></header>' +
        '<div class="csa-qk-blocks">' + rows + '</div>' +
        '<p class="csa-qk-hint">Work Item 不是 Runtime Task：一个 Block 在循环里依次处理多个工作项，<code>iteration</code> 就是它的第几次领取。</p>' +
        '</section>';
    }

    /* --- Inspector ------------------------------------------------------ */

    function itemInspector(d, details) {
      const item = d.byId.get(details.id);
      const n = d.blockCount;
      const facts = [
        ['Work Item 扁平 ID', String(item.id)],
        ['Token Index', 'T' + item.token],
        ['Sparse Block', 'SB' + item.sparseBlock + (item.sparseBlock === WINDOW_SB ? ' · Window' : ' · Sparse')],
        ['Validity', item.valid ? 'valid（valid_block_mask = 1）' : 'invalid（valid_block_mask = 0）'],
        ['Original Position', String(item.originalPosition)],
        ['Queue Position', String(details.queuePosition) + '（qk_order[' + details.queuePosition + '] = ' + item.id + '）'],
        ['Assigned Block', 'Block ' + details.block + ' / 共 ' + n],
        ['Iteration', String(details.iteration) + '（第 ' + (details.iteration + 1) + ' 轮领取）']
      ];
      const formula = [
        'workItemId   = token × SPARSE_BLOCKS + sb = ' + item.token + ' × 5 + ' + item.sparseBlock + ' = ' + item.id,
        'queuePosition = qk_order.indexOf(' + item.id + ') = ' + details.queuePosition,
        'block        = queuePosition % ' + n + ' = ' + details.block,
        'iteration    = ⌊queuePosition / ' + n + '⌋ = ' + details.iteration,
        'qk_flat      = block + iteration × ' + n + ' = ' + details.qkFlat
      ].join('\n');
      return '<span class="csa-eyebrow">WORK ITEM ' + item.id + ' · ' + (item.valid ? 'VALID' : 'INVALID') + '</span>' +
        '<h2>Token ' + item.token + ' · Sparse Block ' + item.sparseBlock + '</h2>' +
        '<dl class="csa-qk-facts">' + facts.map(([k, v]) => '<dt>' + esc(k) + '</dt><dd>' + esc(v) + '</dd>').join('') + '</dl>' +
        '<h3 class="csa-qk-subhead">采用的规则</h3><pre class="csa-qk-formula">' + esc(formula) + '</pre>' +
        (item.sparseBlock === WINDOW_SB
          ? '<p class="csa-qk-hint">SB0 是滑动窗口块，源码在 qk_plan 中直接把 valid_block_mask[*, 0] 写为 1，因此不可切换。</p>'
          : '<button class="btn btn-sm" type="button" data-qk-toggle="' + item.id + '">切换为 ' + (item.valid ? 'invalid' : 'valid') + '</button>') +
        pathBlock(item);
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
      const isItem = state.selection.type === 'item';
      const valid = isItem ? d.byId.get(state.selection.id).valid : true;
      const ranges = [
        [134, 196, 'qk_plan · 有效性、qk_order 两轮写入'],
        [208, 224, 'qk_pv 派发、显式依赖、Block 领取'],
        [224, 314, isItem ? (valid ? '内部计算路径 · valid 分支（224–304）' : '内部计算路径 · invalid 分支（305–313）') : '内部计算路径 · valid / invalid 两个分支']
      ];
      return '<h3 class="csa-qk-subhead">Source Evidence · 真实源码行段</h3>' +
        ranges.map(([a, z, note]) =>
          '<details class="csa-qk-source"><summary><span>' + SUB_FILE + ' : ' + a + '–' + z + '</span><i>' + esc(note) + '</i></summary>' +
          '<pre>' + esc(excerpt(SUB_FILE, a, z)) + '</pre></details>').join('') +
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
      const body = state.selection.type === 'block'
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
      const valid = d.items.filter(item => item.valid).length;
      return '<div class="csa-qk-controls">' +
        '<div class="segmented-control segmented-control-muted" role="group" aria-label="Block 数模式">' + buttons + '</div>' +
        '<span class="csa-qk-badge">' + esc(MODES[state.mode].badge) + '</span>' +
        '<span class="csa-qk-meta">5 Tokens × 5 Sparse Blocks = ' + ITEM_COUNT + ' Work Items · ' + valid + ' valid · ' + (ITEM_COUNT - valid) + ' invalid · 1 Dispatch → ' + d.blockCount + ' SPMD Blocks</span>' +
        '<button class="btn btn-ghost btn-sm" type="button" data-qk-reset>Reset Example</button>' +
        '</div>';
    }

    function render() {
      const d = derive();
      const details = state.selection.type === 'item' ? getWorkItemDetails(state.selection.id, d.queue, d.blockCount) : null;
      const selectedBlock = state.selection.type === 'block' ? state.selection.block : details.block;
      // Selecting a block highlights all of its items; selecting an item only itself.
      const relatedIds = new Set(state.selection.type === 'block' && d.blocks[selectedBlock]
        ? d.blocks[selectedBlock].entries.map(entry => entry.id) : []);
      const ctx = { selectedId: state.selection.type === 'item' ? state.selection.id : null, selectedBlock, relatedIds };

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
      const token = Math.floor(id / SPARSE_BLOCKS);
      const sb = id % SPARSE_BLOCKS;
      if (sb === WINDOW_SB) return;                       // SB0 is always live in the source
      state.mask[token][sb] = state.mask[token][sb] ? 0 : 1;
      pendingFocus = '[data-qk-item="' + id + '"], [data-qk-toggle="' + id + '"]';
      render();
    }

    host.addEventListener('click', event => {
      const toggle = event.target.closest('[data-qk-toggle]');
      if (toggle) { event.preventDefault(); toggleValid(Number(toggle.dataset.qkToggle)); return; }
      const source = event.target.closest('[data-qk-source]');
      if (source) {
        const [file, a, z] = source.dataset.qkSource.split(':');
        onSource(file, Number(a), Number(z));
        return;
      }
      const depToggle = event.target.closest('[data-qk-dep-toggle]');
      if (depToggle) { state.depOpen = !state.depOpen; render(); return; }
      if (event.target.closest('[data-qk-reset]')) {
        state.mask = DEFAULT_MASK.map(row => row.slice());
        state.selection = { type: 'item', id: DEFAULT_SELECTION };
        pendingFocus = '[data-qk-item="' + DEFAULT_SELECTION + '"]';
        render();
        return;
      }
      const mode = event.target.closest('[data-qk-mode]');
      if (mode) {
        state.mode = mode.dataset.qkMode;
        // A block selection cannot survive into a mode with fewer blocks.
        if (state.selection.type === 'block' && state.selection.block >= blockCount()) state.selection = { type: 'block', block: blockCount() - 1 };
        render();
        return;
      }
      const block = event.target.closest('[data-qk-block]');
      if (block) { state.selection = { type: 'block', block: Number(block.dataset.qkBlock) }; pendingFocus = '[data-qk-block="' + block.dataset.qkBlock + '"]'; render(); return; }
      const item = event.target.closest('[data-qk-item]');
      if (item) { state.selection = { type: 'item', id: Number(item.dataset.qkItem) }; pendingFocus = '[data-qk-item="' + item.dataset.qkItem + '"]'; render(); }
    });

    render();

    return {
      render,
      getState: () => ({ mode: state.mode, mask: state.mask.map(row => row.slice()), selection: { ...state.selection } })
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
    MODES,
    constants: { TOKENS, SPARSE_BLOCKS, ITEM_COUNT, NUM_QK_CORES, WINDOW_SB }
  });
})(typeof window !== 'undefined' ? window : globalThis);
