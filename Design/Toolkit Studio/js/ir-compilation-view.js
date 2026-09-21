/* Run → Compilation 内容区。

   信息架构来自参考稿 pypto_compilation_redesign_v6.html：
     摘要 → 需要关注 → 工作区（Kernel 列表 + Source → 关键 Pass → Kernel｜编译 IR 全流程）
   视觉语法走项目已有 token，不引入第二套色板；没有第二份 Object Inspector，
   Pass 下钻就在主视图里就地展开。

   ── 数据来源（全部真实，无伪造）────────────────────────────────────────
   window.PTO_IR_KERNELS
     .kernels[]        45 个 kernel：type / mem(按存储空间字节) / reuse / intent /
                       split / diags / perf(窄搬运提示) / passes[{i,st,n}]
     .limits / .spaces 各存储空间上限（L0A=Left / L0B=Right / L0C=Acc / UB=Vec / L1=Mat）
     .passNames        42 个 pass 名，kernels[].passes[].i 是这个数组的下标
   window.PTO_IR_PIPELINE
     .strata[]         7 个编译层（前端 / 规范化张量 / 层级化 / Tile / 双核 Kernel / 物理内存 / 运行时）
     .passes[]         每个 pass 的中文 desc、gain / lose、函数与算子计数、IR 变更 hunks
   window.PTO_DECODE_LAYER_SOURCE
     decode_layer.py 全文，用 name_hint="<kernel>" 定位 kernel 的源码块

   ── 三个「变化」概念的区别（参考稿第十节明确要求）───────────────────
   1. 「Pass 改变了整体 IR」      → PIPE.passes[i].hunks / f / o（所有 kernel 共用）
   2. 「Pass 对当前 Kernel 有变化」→ K.kernels[].passes[i].st / .n（逐 kernel，真实）
   3. 「Pass 对解释当前 Finding 有价值」→ EXPLAIN 权重表（下面，人工策展的排序函数，
      不是数据；它只决定「哪几个 Pass 值得画进关键链」，不改变任何事实）
   三者不混用：2 决定节点是否高亮，3 决定哪些进入横向关键链。

   ── 已知缺口（没有伪造，明确说明）──────────────────────────────────
   - 最终 Kernel 的「设备代码」在数据源里不存在（inventory 里的 PTO-ISA .pt 未加载）。
     Kernel 阶段因此展示的是真实的资源/切分/复用事实，不是伪代码。
   - `k_seed` / `v_seed` 的 name_hint 块只有 2 行；`decode_fwd_layers` 是编排根，
     定位不到独立源码块，此时 Source 阶段会明确说明，而不是编一段源码。
   - k.passes[i].n 是「该 pass 在该 kernel 上产生的变更条数」，数据源没有给出
     变更内容本身，所以 Pass 阶段展示的 IR 前后差异来自 PIPE.passes[i].hunks
     （那是整个 IR 的 diff，不是这个 kernel 的 diff）—— UI 上已分别标注。 */
(function () {
  'use strict';

  const K = window.PTO_IR_KERNELS;
  const PIPE = window.PTO_IR_PIPELINE;
  const SOURCE = window.PTO_DECODE_LAYER_SOURCE;
  if (!K) return;

  const LIM = K.limits || {};
  const SPACES = K.spaces || [];
  const PASSNAMES = K.passNames || [];
  const PASSMETA = (PIPE && PIPE.passes) || [];
  const STRATA = (PIPE && PIPE.strata) || [];
  const SRC_LINES = SOURCE ? SOURCE.split('\n') : [];
  const SRC_FILE = 'decode_layer.py';

  /* 存储空间在硬件上的叫法与参考稿一致：Left=L0A / Right=L0B / Acc=L0C / Vec=UB / Mat=L1 */
  const SPACE_LABEL = { Vec: 'UB', Mat: 'L1', Acc: 'L0C', Left: 'L0A', Right: 'L0B', Bias: 'Bias' };

  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.prototype.slice.call((r || document).querySelectorAll(s));
  const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g,
    c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const kb = b => (b >= 1024 ? (b / 1024).toFixed(b >= 10240 ? 0 : 1) + ' KB' : b + ' B');
  const pct = (a, b) => (b ? Math.round((a / b) * 100) : 0);

  /* ---------- 派生量（与 ir-compile-guard.js 口径一致，避免同一界面两套算法） ---------- */
  function worstMem(k) {
    let w = 0, space = null;
    for (const s of SPACES) {
      const p = pct((k.mem || {})[s] || 0, LIM[s]);
      if (p > w) { w = p; space = s; }
    }
    return { p: w, space };
  }
  function reuseGain(k) {
    const b = (k.reuse && k.reuse.before.b) || 0;
    return b ? Math.round((1 - k.reuse.after.b / b) * 100) : 0;
  }
  function diagOf(k) {
    const c = { error: 0, warn: 0, perf: 0 };
    (k.diags || []).forEach(d => { c[d.sev] = (c[d.sev] || 0) + 1; });
    return c;
  }
  const perfHintsOf = k => (k.perf || []).length;

  /* 三个 Finding 的真实判定 */
  function findingSets() {
    const mem = K.kernels.filter(k => {
      const w = worstMem(k);
      return w.space === 'Right' && w.p >= 100;
    });
    const intent = K.kernels.filter(k => k.intent && k.intent.demoted > 0);
    const perf = K.kernels.filter(k => perfHintsOf(k) > 0);
    return { mem, intent, perf };
  }
  const FINDINGS = [
    { id: 'mem', tone: 'warn', icon: '!', kind: 'mem', title: 'L0B 达到平台上限',
      route: 'resources', routeLabel: '去 Resources 验证实际资源压力',
      next: 'L0B 在编译阶段已经达到平台上限，但是否形成真实瓶颈仍需结合运行时数据判断。' },
    { id: 'intent', tone: 'warn', icon: '!', kind: 'intent', title: '流水线意图发生变化',
      route: 'performance', routeLabel: '去 Performance 验证运行时影响',
      next: '当前只有编译阶段的意图信号，是否真的拖慢执行，需要映射到 Task 和运行时数据继续验证。' },
    { id: 'perf', tone: 'info', icon: '◇', kind: 'perf', title: '存在编译性能提示',
      route: 'performance', routeLabel: '去 Performance 验证运行时影响',
      next: '编译器在这里给出了静态提示，需要映射到实际 Task 与运行期时间才能判断是否形成瓶颈。' }
  ];

  /* 解释价值权重：人工策展，只回答「这个 Pass 多能解释当前 Kernel」。
     key 必须是 passNames 里真实存在的名字；表里没有的 pass 不会进入关键链。 */
  const EXPLAIN = {
    mem: { AutoTileMatmulL0: 90, MemoryReuse: 80, AllocateMemoryAddr: 80, InitMemRef: 60,
           ConvertTensorToTileOps: 55, OutlineIncoreScopes: 35, OptimizeOrchTensors: 30,
           LowerPipelineLoops: 42, CanonicalizeIOOrder: 38 },
    intent: { SkewCrossCorePipeline: 90, LowerPipelineLoops: 80, SplitVectorKernel: 70,
              ExpandMixedKernel: 70, InjectGMPipeBuffer: 60, ConvertTensorToTileOps: 45, UnrollLoops: 40,
              CanonicalizeIOOrder: 38, InitMemRef: 28 },
    perf: { LowerVectorTransfer: 90, ConvertTensorToTileOps: 70, ResolveBackendOpLayouts: 60,
            CanonicalizeTileSlice: 50, InferTileMemorySpace: 45, OptimizeOrchTensors: 40,
            LowerPipelineLoops: 42, CanonicalizeIOOrder: 38, MemoryReuse: 32, AllocateMemoryAddr: 32 },
    none: { ConvertTensorToTileOps: 70, AutoTileMatmulL0: 60, MemoryReuse: 60,
            AllocateMemoryAddr: 60, OutlineIncoreScopes: 40, SkewCrossCorePipeline: 40,
            LowerPipelineLoops: 42, CanonicalizeIOOrder: 38, InitMemRef: 28 }
  };

  /* 每个 kernel 归到哪一个 Finding —— 决定它的关键链用哪套「解释价值」权重 */
  function kindOf(k) {
    const w = worstMem(k);
    if (w.space === 'Right' && w.p >= 100) return 'mem';
    if (k.intent && k.intent.demoted > 0) return 'intent';
    if (perfHintsOf(k) > 0) return 'perf';
    return 'none';
  }

  /* ---------- 源码定位 ---------- */
  const rxEsc = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  function srcCandidates(name) {
    const out = [name];
    out.push(name.replace(/_(spmd|aic|aiv)$/, ''));
    out.push(name.replace(/_\d+$/, ''));
    return out.filter((v, i, a) => v && a.indexOf(v) === i);
  }
  function locateBlock(name) {
    if (!SRC_LINES.length) return null;
    let hit = -1, used = null;
    for (const c of srcCandidates(name)) {
      const i = SRC_LINES.findIndex(l => l.indexOf('name_hint="' + c + '"') >= 0);
      if (i >= 0) { hit = i; used = c; break; }
    }
    if (hit < 0) {
      for (const c of srcCandidates(name)) {
        const rx = new RegExp('^\\s*def\\s+' + rxEsc(c) + '\\s*\\(');
        const i = SRC_LINES.findIndex(l => rx.test(l));
        if (i >= 0) { hit = i; used = c; break; }
      }
    }
    if (hit < 0) return null;

    /* pl.at(...) 往往跨多行：块首是上一处 `with pl.at(`，块体从 `) as x:` 之后开始 */
    let open = -1;
    for (let j = hit; j >= Math.max(0, hit - 40); j--) {
      if (SRC_LINES[j].indexOf('with pl.at(') >= 0) { open = j; break; }
    }
    let as = -1;
    for (let j = hit; j < Math.min(SRC_LINES.length, hit + 10); j++) {
      if (/\)\s*as\s+[A-Za-z_]\w*\s*:/.test(SRC_LINES[j])) { as = j; break; }
    }
    if (open < 0) open = as >= 0 ? as : hit;
    const head = as >= 0 ? as : open;
    const ind = (SRC_LINES[open].match(/^\s*/) || [''])[0].length;
    let end = head;
    for (let j = head + 1; j < SRC_LINES.length && j < head + 80; j++) {
      const l = SRC_LINES[j];
      if (!l.trim()) { end = j; continue; }
      if ((l.match(/^\s*/) || [''])[0].length <= ind) break;
      end = j;
    }
    return { from: open, to: end, key: used, hint: hit };
  }

  function srcSnippet(name, block, maxLines) {
    if (!block) return null;
    const from = block.from, to = Math.min(block.to, from + maxLines - 1);
    const out = [];
    for (let i = from; i <= to; i++) {
      let line = esc(SRC_LINES[i]);
      if (i === block.hint) line = '<span class="kc-hl">' + line + '</span>';
      else if (/pl\.pipeline\(/.test(SRC_LINES[i])) line = '<span class="kc-hl-warn">' + line + '</span>';
      out.push(line);
    }
    if (block.to > to) out.push('<span class="kc-dim">… 共 ' + (block.to - from + 1) + ' 行</span>');
    return out.join('\n');
  }

  /* 关键 Pass：先按解释价值排序取前 3，再按真实先后顺序展示（编译链是时间序） */
  function keyPasses(k) {
    const w = EXPLAIN[kindOf(k)] || EXPLAIN.none;
    const cand = (k.passes || []).filter(p => p.st === 'changed' || p.st === 'born');
    if (!cand.length) return [];
    const scored = cand.map(p => {
      const name = PASSNAMES[p.i] || ('Pass ' + p.i);
      return { i: p.i, name, st: p.st, n: p.n, w: w[name] || 0, score: (w[name] || 0) * 1000 + Math.min(p.n, 999) };
    });
    const curated = scored.filter(s => s.w > 0);
    const top = (curated.length ? curated : scored)
      .slice().sort((a, b) => b.score - a.score).slice(0, 3);
    /* 策展表没覆盖到的 Kernel（例如 SPMD 包壳）：用剩下的真实变更 Pass 按变更量补到 3 个。
       补进来的每个节点仍然是「真的改过这个 Kernel」的 Pass，只是排序依据换成了变更量。 */
    if (top.length < 3) {
      const rest = scored.filter(s => top.indexOf(s) < 0).sort((a, b) => b.n - a.n);
      while (top.length < 3 && rest.length) top.push(rest.shift());
    }
    return top.sort((a, b) => a.i - b.i);
  }

  /* ---------- IR 前后差异（来自 PIPE.passes[i].hunks，是整个 IR 的 diff） ---------- */
  function irExcerpt(meta, side, maxLines) {
    if (!meta || !meta.hunks || !meta.hunks.length) return null;
    const h = meta.hunks[0];
    const arr = (side === 'before' ? h.b : h.a) || [];
    if (!arr.length) return null;
    return arr.slice(0, maxLines).map(l => esc(String(l).replace(/^\s+/, m => m))).join('\n');
  }
  /* hunks 缺失时用真实的 IR 规模变化顶上，不编造代码 */
  function irScaleFacts(meta) {
    if (!meta) return [];
    const f = meta.f || {}, o = meta.o || {};
    const out = [];
    const fn = [];
    if (f.aic) fn.push('AIC ' + f.aic);
    if (f.aiv) fn.push('AIV ' + f.aiv);
    if (f.incore) fn.push('InCore ' + f.incore);
    if (f.group) fn.push('Group ' + f.group);
    if (f.spmd) fn.push('SPMD ' + f.spmd);
    if (fn.length) out.push(['函数', fn.join(' · ')]);
    const op = [];
    if (o.te != null) op.push('tensor ' + o.te);
    if (o.ti != null) op.push('tile ' + o.ti);
    if (o.mr != null) op.push('memref ' + o.mr);
    if (o.al != null) op.push('alloc ' + o.al);
    if (op.length) out.push(['算子', op.join(' · ')]);
    if (meta.l) out.push(['IR 行数', String(meta.l)]);
    return out;
  }

  /* ---------- state ---------- */
  const st = {
    kernel: null, filter: 'issues', compact: false, findOn: null,
    /* 工作区页签：kernel = Kernel 列表 + 详情；trace = 借用来的编译 IR 全流程 */
    pane: 'kernel'
  };
  let host = null;

  function kernelNames() { return K.kernels.map(k => k.name); }
  function issueSet() {
    const s = findingSets();
    const set = {};
    ['mem', 'intent', 'perf'].forEach(id => s[id].forEach(k => { set[k.name] = true; }));
    return set;
  }
  function byName(n) { return K.kernels.find(k => k.name === n) || null; }
  function current() { return byName(st.kernel) || K.kernels[0] || null; }

  function listRows() {
    const set = issueSet();
    let rows = K.kernels.slice();
    if (st.filter === 'issues') rows = rows.filter(k => set[k.name]);
    const order = { mem: 0, intent: 1, perf: 2, none: 3 };
    rows.sort((a, b) => {
      const d = order[kindOf(a)] - order[kindOf(b)];
      if (d) return d;
      return worstMem(b).p - worstMem(a).p || a.name.localeCompare(b.name);
    });
    return rows;
  }

  /* ---------- 渲染 ---------- */
  function summaryHTML() {
    const changed = PASSMETA.filter(p => p.d === null || p.d > 0).length;
    let e = 0;
    K.kernels.forEach(k => { e += diagOf(k).error; });
    const pass = e === 0;
    /* 设备侧的 AIC / AIV 计数直接数 kernels[].type，不写死 8 / 31 */
    const dev = K.kernels.reduce((a, k) => {
      if (k.type === 'AIC' || k.type === 'AIV') a[k.type] = (a[k.type] || 0) + 1;
      return a;
    }, {});
    const devValue = ['AIC', 'AIV'].filter(t => dev[t]).map(t => t + ' ' + dev[t]).join(' · ') || '—';
    const stratValue = STRATA.length
      ? STRATA[0].id + '–' + STRATA[STRATA.length - 1].id + ' · ' + STRATA.length + ' 层'
      : '—';

    /* 判定块与字段组都跟 Correctness 的 .kf-dg-verdict / .kf-dg-cell 同一套版式：
       标签 + 值两行，不再往格子里塞第三行说明。 */
    const cell = (label, value, tone) =>
      '<div class="kc-scell"><span>' + esc(label) + '</span><b' + (tone ? ' class="is-' + tone + '"' : '') + '>' +
      esc(value) + '</b></div>';

    return '<div class="kc-summary">' +
      '<div class="kc-summary__outcome"><span>编译</span><b class="' + (pass ? 'is-ok' : 'is-bad') + '">' + (pass ? 'PASS' : 'FAIL') + '</b></div>' +
      cell('阻塞诊断', e ? e + ' 个 error' : '无阻塞', e ? 'bad' : 'ok') +
      cell('IR 变化', changed + ' / ' + PASSMETA.length, changed ? 'warn' : 'ok') +
      cell('受影响 Kernel', String(K.kernels.length), '') +
      cell('设备 Kernel', devValue, '') +
      cell('编译层', stratValue, '') +
      cell('IR 快照', PASSMETA.length + ' 个', '') +
    '</div>';
  }

  function findingsHTML() {
    const s = findingSets();
    return '<div class="kc-findings">' + FINDINGS.map(f => {
      const hits = s[f.id];
      if (!hits.length) return '';
      let desc, count;
      if (f.id === 'mem') {
        const top = hits.slice().sort((a, b) => worstMem(b).p - worstMem(a).p)[0];
        desc = top.name + ' · ' + kb(top.mem.Right) + ' / ' + kb(LIM.Right) +
          '。编译通过，当前尚不能判断是否影响运行性能。';
        if (hits.length > 1) desc += ' 另有 ' + (hits.length - 1) + ' 个 Kernel 同样打满。';
        count = hits.length + ' Kernel';
      } else if (f.id === 'intent') {
        const names = hits.slice(0, 3).map(k => k.name).join(' · ');
        desc = hits.length + ' 个 Kernel 的 pl.pipeline 在编译过程中被降级为顺序执行：' + names + '。';
        count = hits.length + ' Kernel';
      } else {
        /* 计数和「最小内层」都取自 kernels[].perf（逐 kernel 的窄搬运记录），
           不混用 perfHints —— 两者口径不同，混在一起会对不上。 */
        const recs = hits.reduce((a, k) => a.concat(k.perf || []), []);
        const minInner = recs.reduce((a, r) => Math.min(a, r.innermost), Infinity);
        desc = '检测到窄数据搬运 / 向量化机会' +
          (isFinite(minInner) ? '（最小内层 ' + minInner + ' 元素，目标 ' + K.perfMinInnermost + ' 元素）' : '') +
          '，需要到 Performance 验证实际影响。';
        count = hits.length + ' Kernel · ' + recs.length + ' 处';
      }
      return '<button type="button" class="kc-finding is-' + f.tone + (st.findOn === f.id ? ' is-on' : '') +
        '" data-kc-find="' + f.id + '">' +
        '<span class="kc-fico">' + esc(f.icon) + '</span>' +
        '<span class="kc-fbody"><b>' + esc(f.title) + '</b><small>' + esc(desc) + '</small></span>' +
        '<em>' + esc(count) + '</em></button>';
    }).join('') + '</div>';
  }

  /* Orchestration 这类长类型名放不进 38px 的 chip，给一个显示缩写，完整名进 title */
  const TYPE_ABBR = { Orchestration: 'Orch' };

  function itemHTML(k) {
    const kind = kindOf(k);
    const w = worstMem(k);
    const c = diagOf(k);
    const bits = [];
    if (w.space) bits.push((SPACE_LABEL[w.space] || w.space) + ' ' + w.p + '%');
    if (k.intent && k.intent.demoted) bits.push('pipeline 降级 ×' + k.intent.demoted);
    if (perfHintsOf(k)) bits.push(perfHintsOf(k) + ' 条性能提示');
    const dc = c.error + c.warn + c.perf;
    if (dc) bits.push(dc + ' 条诊断');
    if (!bits.length && k.reuse && k.reuse.before.b) bits.push('复用 −' + reuseGain(k) + '%');
    const tyCls = k.type === 'AIC' ? 'is-aic' : k.type === 'AIV' ? 'is-aiv' : 'is-other';
    return '<button type="button" class="kc-item' + (k.name === st.kernel ? ' is-on' : '') +
      '" data-kc-kernel="' + esc(k.name) + '" title="' + esc(k.name + ' · ' + (k.type || '') + '') + '">' +
      '<span class="kc-ty ' + tyCls + '" title="' + esc(k.type || '') + '">' +
      esc(TYPE_ABBR[k.type] || k.type || '—') + '</span>' +
      '<span><span class="kc-iname">' + esc(k.name) + '</span>' +
      '<span class="kc-imeta">' + esc(bits.join(' · ')) + '</span></span>' +
      '<span class="kc-istate is-' + (kind === 'none' ? 'ok' : 'warn') + '">' +
      (kind === 'none' ? '✓ 正常' : '● 关注') + '</span></button>';
  }

  function listHTML() {
    const rows = listRows();
    if (!rows.length) return '<p class="kc-list-empty">没有命中的 kernel。</p>';
    return rows.map(itemHTML).join('');
  }

  /* --- Source 阶段 --- */
  function srcStageHTML(k) {
    const block = locateBlock(k.name);
    const range = block ? SRC_FILE + ':' + (block.from + 1) + '–' + (block.to + 1) : SRC_FILE;
    const snippet = block ? srcSnippet(k.name, block, 22) : null;
    const tags = ['PyPTO DSL'];
    if (block && block.key !== k.name) tags.push('name_hint=' + block.key);
    if (k.intent && k.intent.declared.length) {
      tags.push(k.intent.declared.map(d => '声明 pl.pipeline(stage=' + d + ')').join(' '));
    }
    const inner = snippet
      ? '<div class="kc-code"><div class="kc-code-head"><span>' + esc(SRC_FILE) + '</span><span>Source</span></div>' +
        '<div class="kc-code-body">' + snippet + '</div></div>'
      : '<div class="kc-code"><div class="kc-code-head"><span>' + esc(SRC_FILE) + '</span><span>Source</span></div>' +
        '<div class="kc-src-missing">该 Kernel 是编排/合成结果，源码里没有独立的 name_hint 块。<br>' +
        '它由 ' + esc(SRC_FILE) + ' 中的多个 scope 组合而成，此处不展示片段以免误导。</div></div>';

    return '<section class="kc-stage is-source">' +
      '<div class="kc-stage-head"><span class="kc-icon">&lt;/&gt;</span>' +
      '<span class="kc-kind">源码</span><b>' + esc(range) + '</b></div>' +
      '<div class="kc-desc">' + esc(sourceSummary(k, block)) + '</div>' +
      '<div class="kc-meta">' + tags.map(t => '<span>' + esc(t) + '</span>').join('') + '</div>' +
      inner + '</section>';
  }

  function sourceSummary(k, block) {
    if (!block) return '该 Kernel 没有独立的源码块，它是编排层合成的结果。';
    const kind = kindOf(k);
    if (kind === 'mem') return '这段源码描述 ' + k.name + ' 的矩阵分块计算与结果写回，' +
      '后面的 L0 切分与 Buffer 分配都从它推导出来。';
    if (kind === 'intent') return '这段源码声明了流水线执行意图，是后续「意图是否被兑现」的分析起点。';
    if (kind === 'perf') return '这段源码包含数据搬运与向量计算，后续会被降低成 AIV 执行路径。';
    return '这段源码是 ' + k.name + ' 的直接来源，关键编译意图均已兑现。';
  }

  /* --- Pass 阶段 --- */
  function passStageHTML(k, p, kind) {
    const meta = PASSMETA[p.i];
    const warn = p.name === 'SkewCrossCorePipeline' || p.name === 'AllocateMemoryAddr' ||
      p.name === 'AutoTileMatmulL0';
    const before = irExcerpt(meta, 'before', 7);
    const after = irExcerpt(meta, 'after', 7);
    const facts = irScaleFacts(meta);
    let code;
    if (before || after) {
      code = '<div class="kc-code">' +
        '<div class="kc-code-head"><span>IR · 整体变更 hunk</span><span>' + esc(p.name) + '</span></div>' +
        '<div class="kc-code-body">' +
        (before ? '<span class="kc-dim">变化前</span>\n' + before : '<span class="kc-dim">变化前 · 该 hunk 无删除行</span>') +
        '\n\n<span class="kc-dim">→</span>\n\n' +
        (after ? '<span class="kc-dim">变化后</span>\n' + after : '<span class="kc-dim">变化后 · 该 hunk 无新增行</span>') +
        '</div></div>';
    } else {
      code = '<div class="kc-code">' +
        '<div class="kc-code-head"><span>IR 规模变化</span><span>' + esc(p.name) + '</span></div>' +
        '<div class="kc-code-body">' +
        '<span class="kc-dim">该 Pass 的 dump 没有行级 diff，以下是整体 IR 的真实计数</span>\n\n' +
        (facts.length ? facts.map(f => f[0].padEnd(8, ' ') + f[1]).join('\n') : '无可用计数') +
        '</div></div>';
    }

    return '<section class="kc-stage is-pass' + (warn ? ' is-warn' : '') + '">' +
      '<div class="kc-stage-head"><span class="kc-icon">P</span>' +
      '<span class="kc-kind">Pass</span><b>' + esc(p.name) + '</b></div>' +
      '<div class="kc-desc">' + esc((meta && meta.desc) || '该 Pass 改变了当前 Kernel 的中间表示。') + '</div>' +
      '<div class="kc-meta">' +
        (meta && meta.s ? '<span>' + esc(stratumName(meta.s)) + '</span>' : '') +
        '<span class="' + (p.st === 'born' ? 'is-ok' : 'is-warn') + '">' +
          (p.st === 'born' ? '在此 Pass 诞生' : '变更 ' + p.n + ' 处') + '</span>' +
        (meta && meta.gain && meta.gain.length ? '<span>' + esc('+' + meta.gain.join(' ')) + '</span>' : '') +
      '</div>' +
      code + '</section>';
  }

  function stratumName(id) {
    const s = STRATA.find(x => x.id === id);
    return s ? s.name : id;
  }

  /* --- Kernel 阶段 --- */
  function kernelStageHTML(k) {
    const tagCls = k.type === 'AIC' ? 'is-aic' : k.type === 'AIV' ? 'is-aiv' : 'is-other';
    const lines = ['kernel ' + k.name + '(' + (k.type || '?') + ')'];
    if (k.split) lines.push('  split      ' + k.split);
    SPACES.forEach(s => {
      const v = (k.mem || {})[s];
      if (!v) return;
      const p = pct(v, LIM[s]);
      lines.push('  ' + (SPACE_LABEL[s] || s).padEnd(9, ' ') + kb(v) + ' / ' + kb(LIM[s]) + '   ' + p + '%');
    });
    if (k.reuse && k.reuse.after.b != null) {
      lines.push('  reuse      ' + k.reuse.before.n + ' → ' + k.reuse.after.n + ' buffers · ' +
        kb(k.reuse.before.b) + ' → ' + kb(k.reuse.after.b));
    }
    const declared = km => (km.intent && km.intent.declared.length)
      ? km.intent.declared.map(d => 'pipeline(stage=' + d + ')').join(' ')
      : (km.intent && km.intent.l0 && km.intent.l0.length
          ? km.intent.l0.map(l => 'L0 split ' + l.from + '→' + l.to + ' step ' + l.step).join(' · ')
          : '—');
    lines.push('  intent     ' + declared(k));
    if (k.intent && k.intent.demoted) lines.push('  demoted    pipeline ×' + k.intent.demoted + ' → sequential');
    const c = diagOf(k);
    const dc = c.error + c.warn + c.perf;
    lines.push('  diag       ' + (dc ? dc + ' 条（' + [c.error && c.error + 'E', c.warn && c.warn + 'W',
      c.perf && c.perf + 'P'].filter(Boolean).join(' ') + '）' : '无'));
    if (perfHintsOf(k)) {
      const h = k.perf[0];
      lines.push('  hint       最小内层 ' + h.innermost + ' 元素 · ' + h.bytes + ' B · 目标 ' + K.perfMinInnermost + ' B');
    }

    const metaTags = [];
    const w = worstMem(k);
    if (w.space) metaTags.push([(SPACE_LABEL[w.space] || w.space) + ' ' + kb(k.mem[w.space]) + ' / ' + kb(LIM[w.space]),
      w.p >= 90 ? 'is-bad' : w.p >= 70 ? 'is-warn' : 'is-ok']);
    if (k.reuse && k.reuse.before.b) metaTags.push(['复用 −' + reuseGain(k) + '%', 'is-ok']);
    metaTags.push([k.type === 'AIC' ? 'AIC Kernel' : k.type === 'AIV' ? 'AIV Kernel' : (k.type || '—') + ' Kernel',
      tagCls === 'is-aic' ? 'is-ok' : 'is-ok']);

    return '<section class="kc-stage is-kernel">' +
      '<div class="kc-stage-head"><span class="kc-icon">K</span>' +
      '<span class="kc-kind">最终 Kernel</span><b>' + esc(k.name) + '</b></div>' +
      '<div class="kc-desc">' + esc(kernelSummary(k)) + '</div>' +
      '<div class="kc-meta">' + metaTags.map(t =>
        '<span class="' + t[1] + '">' + esc(t[0]) + '</span>').join('') + '</div>' +
      '<div class="kc-code"><div class="kc-code-head"><span>' + esc(k.name) + '</span>' +
      '<span>' + esc((k.type || '') + ' Kernel') + '</span></div>' +
      '<div class="kc-code-body">' + esc(lines.join('\n')) + '</div></div></section>';
  }

  function kernelSummary(k) {
    const kind = kindOf(k);
    if (kind === 'mem') return '最终生成合法 AIC Kernel，但片上内存已经达到平台上限。';
    if (kind === 'intent') return '编译完成，但声明的流水线意图未被完全兑现，已降级为顺序执行。';
    if (kind === 'perf') return '编译完成，没有阻塞，但编译器标记了更宽搬运 / 向量化的潜在机会。';
    return '编译完成，没有需要优先处理的阻塞或风险信号。';
  }

  function detailHTML() {
    const k = current();
    if (!k) return '<p class="kc-list-empty">没有可展示的 Kernel。</p>';
    const block = locateBlock(k.name);
    const range = block ? SRC_FILE + ':' + (block.from + 1) + '–' + (block.to + 1) : SRC_FILE;
    const kind = kindOf(k);
    const keys = keyPasses(k);

    const badges = [];
    const w = worstMem(k);
    if (w.space) badges.push([(SPACE_LABEL[w.space] || w.space) + ' ' + w.p + '%',
      w.p >= 100 ? 'is-bad' : w.p >= 70 ? 'is-warn' : 'is-ok']);
    if (k.intent && k.intent.demoted) badges.push(['意图未兑现 ×' + k.intent.demoted, 'is-warn']);
    if (perfHintsOf(k)) badges.push([perfHintsOf(k) + ' 条性能提示', 'is-warn']);
    const c = diagOf(k);
    if (c.error + c.warn + c.perf) badges.push([(c.error + c.warn + c.perf) + ' 条诊断', 'is-warn']);
    if (!badges.length) badges.push(['资源正常 · 意图兑现', 'is-ok']);

    const track = [srcStageHTML(k), '<div class="kc-arrow">→</div>', '<div class="kc-gap">…</div>'];
    keys.forEach((p, i) => {
      track.push(passStageHTML(k, p, kind));
      if (i < keys.length - 1) track.push('<div class="kc-gap">…</div>', '<div class="kc-arrow">→</div>');
    });
    track.push('<div class="kc-gap">…</div>', '<div class="kc-arrow">→</div>',
      '<div class="kc-arrow is-pair">→</div>', kernelStageHTML(k));

    const F = FINDINGS.find(f => f.kind === kind) || FINDINGS[0];

    return '<div class="kc-dhead"><div><h3>' + esc(k.name) + '</h3>' +
      '<p>' + esc(range) + ' · 直接按 源码 → 关键 Pass → 最终 Kernel 阅读整个编译变化</p></div>' +
      '<div class="kc-badges">' + badges.map(b =>
        '<span class="kc-badge ' + b[1] + '">' + esc(b[0]) + '</span>').join('') + '</div></div>' +
      '<div class="kc-flow">' +
        '<div class="kc-flow-top"><div class="kc-flow-title">' +
          '<b>源码 → 关键 Pass → 最终 Kernel</b>' +
          '<small>关键 Pass 只保留能解释当前 Kernel 为什么形成现在结构 / 资源占用 / 诊断结果的步骤，' +
          '未影响当前 Kernel 的 Pass 用「…」折叠。左右拖动查看完整链路。</small></div>' +
          '<div class="kc-flow-actions"><button type="button" class="kc-compare" data-kc-compact>' +
          (st.compact ? '展开关键 Pass' : '折叠 Pass，只看源码 / Kernel') + '</button></div></div>' +
        '<div class="kc-journey-scroll" data-kc-scroll>' +
          '<div class="kc-journey-track' + (st.compact ? ' is-compact' : '') + '" data-kc-track>' +
          track.join('') + '</div></div></div>' +
      '<div class="kc-next"><b>下一步</b><span>' + esc(F.next) + '</span>' +
      '<button type="button" data-kc-route="' + esc(F.route) + '">' + esc(F.routeLabel) + ' →</button></div>';
  }

  /* ---------- 装配 ---------- */
  function shellHTML() {
    const rows = listRows();
    return '<section class="kc" data-kc>' +
      summaryHTML() +
      '<div class="kc-sect"><div><h2>需要关注</h2>' +
        '<p>把底层编译信号转成可定位、可解释、可继续验证的发现</p></div></div>' +
      findingsHTML() +
      /* 工作区两个页签：Kernel 列表 + 详情 / 编译 IR 全流程（借用 #kgTrace）。
         页签 1 把原来的小节标题和两栏网格收进同一个容器。 */
      '<nav class="kc-tabs" role="tablist" aria-label="编译工作区">' +
        '<button type="button" role="tab" class="kc-tab' + (st.pane === 'kernel' ? ' is-on' : '') + '"' +
          ' data-kc-tab="kernel" aria-selected="' + (st.pane === 'kernel') + '">Kernel 工作区</button>' +
        '<button type="button" role="tab" class="kc-tab' + (st.pane === 'trace' ? ' is-on' : '') + '"' +
          ' data-kc-tab="trace" aria-selected="' + (st.pane === 'trace') + '">编译 IR 全流程</button>' +
      '</nav>' +
      '<section class="kc-pane" data-kc-pane="kernel">' +
        '<div class="kc-sect"><div><h2>Kernel</h2>' +
          '<p>从最终设备执行单元回溯：源码 → 关键编译变化 → 最终结果</p></div></div>' +
        '<div class="kc-work">' +
          '<section class="kc-list-panel"><div class="kc-lhead"><b>Kernel 列表</b>' +
            '<span class="kc-filter">' +
              '<button type="button" data-kc-filter="issues" class="' + (st.filter === 'issues' ? 'is-on' : '') + '">需要关注</button>' +
              '<button type="button" data-kc-filter="all" class="' + (st.filter === 'all' ? 'is-on' : '') + '">全部 ' + K.kernels.length + '</button>' +
            '</span></div>' +
            '<div class="kc-list" data-kc-list>' + listHTML() + '</div></section>' +
          '<section class="kc-detail" data-kc-detail>' + detailHTML() + '</section>' +
        '</div>' +
      '</section>' +
      '<section class="kc-pane" data-kc-pane="trace"></section>' +
      '</section>';
  }

  /* 页签切换只改 class，不重画 —— 页签 2 里挂的是从 kernelGuard 借来的 #kgTrace
     实体节点，重画会把它冲掉。 */
  function showPane() {
    if (!host) return;
    $$('[data-kc-pane]', host).forEach(p => p.classList.toggle('is-on', p.dataset.kcPane === st.pane));
    $$('[data-kc-tab]', host).forEach(b => {
      const on = b.dataset.kcTab === st.pane;
      b.classList.toggle('is-on', on);
      b.setAttribute('aria-selected', String(on));
    });
    if (st.pane === 'trace') attachTrace();
  }

  /* 把 compile guard 的编译 IR 全流程搬进页签 2。同一时刻只有一个宿主：
     这里 appendChild(e) 之后它就不在 #kernelGuard 里了，归还见 release()。 */
  function attachTrace() {
    const box = $('[data-kc-pane="trace"]', host);
    if (!box) return;
    const G = window.PTO_GUARD;
    const el = G && G.traceEl ? G.traceEl() : null;
    if (!el) {
      box.innerHTML = '<p class="kc-pane-missing">编译 IR 全流程需要 compile guard 的 Pass 数据，本次视图没有加载。</p>';
      return;
    }
    if (el.parentElement !== box) {
      box.innerHTML = '';              // 清掉上一次的占位文案
      box.appendChild(el);
    }
    // 两个页签看的是同一个 Kernel：切过来时把 guard 的选中态对齐，
    // activate() 会顺带把还没画过的河流图补上。
    if (G.select && st.kernel && byName(st.kernel)) G.select(st.kernel);
    else if (G.activate) G.activate();
  }

  function paint() {
    if (!host) return;
    host.innerHTML = shellHTML();
    showPane();
  }
  function paintList() {
    const box = $('[data-kc-list]', host);
    if (box) box.innerHTML = listHTML();
  }
  function paintDetail() {
    const box = $('[data-kc-detail]', host);
    if (box) box.innerHTML = detailHTML();
  }

  function pickKernel(name, opts) {
    if (!byName(name)) return;
    st.kernel = name;
    opts = opts || {};
    if (!issueSet()[name]) st.filter = 'all';
    paint();
    if (opts.scroll) {
      const on = $('.kc-item.is-on', host);
      if (on && on.scrollIntoView) on.scrollIntoView({ block: 'nearest' });
    }
  }

  function markFind(id) {
    st.findOn = st.findOn === id ? null : id;
    if (st.findOn) {
      const set = findingSets();
      const hits = set[st.findOn] || [];
      if (hits.length) {
        const top = hits.slice().sort((a, b) => worstMem(b).p - worstMem(a).p)[0];
        st.kernel = top.name;
        st.filter = issueSet()[top.name] ? st.filter : 'all';
      }
    }
    st.pane = 'kernel';          // 命中结果在列表里，先切回工作区页签
    paint();
    const box = $('.kc-work', host);
    if (box && box.scrollIntoView) box.scrollIntoView({ block: 'nearest' });
  }

  function onClick(e) {
    const tb = e.target.closest('[data-kc-tab]');
    if (tb) { st.pane = tb.dataset.kcTab === 'trace' ? 'trace' : 'kernel'; showPane(); return; }
    const fid = e.target.closest('[data-kc-find]');
    if (fid) { markFind(fid.dataset.kcFind); return; }
    const kid = e.target.closest('[data-kc-kernel]');
    if (kid) { pickKernel(kid.dataset.kcKernel, { scroll: true }); return; }
    const flt = e.target.closest('[data-kc-filter]');
    if (flt) {
      st.filter = flt.dataset.kcFilter;
      $$('[data-kc-filter]', host).forEach(b => b.classList.toggle('is-on', b.dataset.kcFilter === st.filter));
      paintList();
      return;
    }
    const cp = e.target.closest('[data-kc-compact]');
    if (cp) {
      st.compact = !st.compact;
      const track = $('[data-kc-track]', host);
      if (track) track.classList.toggle('is-compact', st.compact);
      cp.textContent = st.compact ? '展开关键 Pass' : '折叠 Pass，只看源码 / Kernel';
      const sc = $('[data-kc-scroll]', host);
      if (st.compact && sc) sc.scrollLeft = 0;
      return;
    }
    const rt = e.target.closest('[data-kc-route]');
    if (rt) {
      const btn = document.querySelector('[data-th-tab="' + rt.dataset.kcRoute + '"]');
      if (btn) btn.click();
      return;
    }
  }

  /* ---------- 对外 ---------- */
  window.PTO_COMPILATION = {
    /* root 由 task-history 传入（#runTabPanel），事件用委托，重复 render 不会叠加监听 */
    render(root) {
      host = root;
      if (!host) return false;
      if (!st.kernel || !byName(st.kernel)) {
        const rows = listRows();
        st.kernel = rows.length ? rows[0].name : (kernelNames()[0] || null);
      }
      if (!host.dataset.kcBound) {
        host.addEventListener('click', onClick);
        host.dataset.kcBound = '1';
      }
      paint();
      return true;
    },
    /* Execution 页签的「在 Compilation 查看」落到这里 */
    selectKernel(name) {
      if (!host) return false;
      pickKernel(String(name), { scroll: true });
      return true;
    },
    /* task-history 的 syncPanel() 靠它判断这块面板现在归谁：是本视图自绘的
       .kc，还是从 stage 2 搬来的 DOM。判错就会把新视图冲掉。 */
    owns(node) { return !!node && host === node; },
    /* task-history 在重写 #runTabPanel 之前必须先调这个：页签 2 里挂的
       #kgTrace 是从 kernelGuard 借来的实体节点，被 innerHTML 冲掉就成了游离
       节点，归还后再也长不回 stage 2 的 Kernel Guard 里。 */
    release() {
      if (window.PTO_GUARD && window.PTO_GUARD.traceHome) window.PTO_GUARD.traceHome();
      host = null;
    },
    /* 供调试与验收用：当前状态快照 */
    state() {
      return { kernel: st.kernel, filter: st.filter, compact: st.compact,
        findOn: st.findOn, pane: st.pane,
        keyPasses: (current() ? keyPasses(current()).map(p => p.name) : []) };
    },
    ready: true
  };
})();
