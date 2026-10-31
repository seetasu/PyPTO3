/* ──────────────────────────────────────────────────────────────────────────────
 * Hardware-Native Lens · 交互
 * ────────────────────────────────────────────────────────────────────────────── */
(function () {
  'use strict';

  const M = window.HNL;
  const V = window.HNV;
  const esc = V.esc;
  const $ = function (sel, root) { return (root || document).querySelector(sel); };

  // 透镜：Tensor/Tile 合并为一组，其余各一组
  const LENSES = [
    { id: 'tensor', label: 'Tensor / Tile', members: ['tensor', 'tile'], desc: '形状 · dtype · 片上空间' },
    { id: 'scope', label: 'Scope', members: ['scope'], desc: '任务粒度 · 依赖成边' },
    { id: 'parallel', label: '并行请求', members: ['parallel'], desc: 'parallel · spmd · pipeline' },
    { id: 'layout', label: '内部布局', members: ['layout'], desc: '填充 · 分页 · 对齐' }
  ];
  const LENS_OF = {};
  LENSES.forEach(function (L) { L.members.forEach(function (m) { LENS_OF[m] = L.id; }); });

  // 约束 → 代表性可视化面板
  const CONSTRAINT_CONCEPT = {
    'cube-m-fractal': 'layout:q-pad',
    'page-identity': 'layout:paged-kv',
    'l0b-capacity': 'tile:k-tile',
    'slab-1to1': 'layout:split-k',
    'down-chunk-align': 'layout:split-k',
    'splitk-divides-k': 'layout:split-k',
    'gp-head-cover': 'parallel:grid-stride',
    'spmd-even-split': 'parallel:spmd',
    'block-balance': 'parallel:grid-stride'
  };

  const STATUS_LABEL = { ok: '已满足', warn: '可通过 · 有代价', error: '违反' };

  // ── 状态 ───────────────────────────────────────────────────────────────────
  const state = {
    cfg: Object.assign({}, M.DEFAULTS),
    lens: { tensor: true, scope: true, parallel: true, layout: true },
    sel: { type: 'overview', id: null },
    anchor: null,
    filter: 'all',
    d: null,
    results: {}
  };

  // ════════════════════════════════════════════════════════════════════════════
  // 代码着色
  // ════════════════════════════════════════════════════════════════════════════
  const KEYWORDS = ['with', 'for', 'in', 'as', 'def', 'return', 'assert', 'and', 'or', 'not',
    'if', 'elif', 'else', 'while', 'lambda', 'import', 'from', 'class', 'yield', 'pass',
    'break', 'continue', 'try', 'except', 'finally', 'raise', 'is', 'del', 'global', 'await', 'async'];
  const LANG_CONST = ['True', 'False', 'None'];
  const BUILTINS = ['range', 'len', 'min', 'max', 'int', 'float', 'str', 'bool', 'sum', 'abs',
    'enumerate', 'zip', 'print', 'list', 'dict', 'tuple'];
  const DTYPES = ['FP32', 'FP16', 'BF16', 'INT4', 'INT8', 'INT32', 'INT64', 'UINT8',
    'INDEX', 'TASK_ID', 'BOOL'];

  // 1 注释 · 2 字符串 · 3 装饰器 · 4 数字 · 5 标识符 · 6 空白 · 7 标点 · 8 运算符
  const TOKEN_RE = /(#[^\n]*)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(@[A-Za-z_][A-Za-z0-9_.]*)|(\b\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\b)|([A-Za-z_][A-Za-z0-9_]*)|(\s+)|([(){}\[\].,:;])|([^\s])/g;

  function fill(tpl, d) {
    return String(tpl).replace(/\{\{([A-Z_0-9]+)\}\}/g, function (_, k) {
      const v = d[k];
      if (v == null) return '?';
      return typeof v === 'number' && !Number.isInteger(v) ? String(Math.round(v * 100) / 100) : String(v);
    });
  }

  function symbolOf(name) {
    if (M.SYMBOLS[name]) return M.SYMBOLS[name];
    // pl.a.b.c → 尝试更短的前缀
    const parts = name.split('.');
    for (let i = parts.length - 1; i > 1; i--) {
      const p = parts.slice(0, i).join('.');
      if (M.SYMBOLS[p]) return M.SYMBOLS[p];
    }
    return null;
  }

  // ── 分词 ──────────────────────────────────────────────────────────────────
  function tokenize(line) {
    const toks = [];
    let m;
    TOKEN_RE.lastIndex = 0;
    while ((m = TOKEN_RE.exec(line)) !== null) {
      if (m[1] != null) toks.push({ k: 'cmt', v: m[1] });
      else if (m[2] != null) toks.push({ k: 'str', v: m[2] });
      else if (m[3] != null) toks.push({ k: 'dec', v: m[3] });
      else if (m[4] != null) toks.push({ k: 'num', v: m[4] });
      else if (m[5] != null) toks.push({ k: 'id', v: m[5] });
      else if (m[6] != null) toks.push({ k: 'ws', v: m[6] });
      else if (m[7] != null) toks.push({ k: 'punc', v: m[7] });
      else toks.push({ k: 'op', v: m[8] });
    }
    return toks;
  }

  function prevIdx(toks, i) { for (let j = i - 1; j >= 0; j--) if (toks[j].k !== 'ws') return j; return -1; }
  function nextIdx(toks, i) { for (let j = i + 1; j < toks.length; j++) if (toks[j].k !== 'ws') return j; return -1; }

  // ── 标识符分类 ────────────────────────────────────────────────────────────
  function classifyId(toks, i, inChain) {
    const v = toks[i].v;
    if (KEYWORDS.indexOf(v) >= 0) return 'tk-kw';
    if (LANG_CONST.indexOf(v) >= 0) return 'tk-lang';

    const p = prevIdx(toks, i);
    const n = nextIdx(toks, i);
    const isCall = n >= 0 && toks[n].k === 'punc' && toks[n].v === '(';

    // def foo(...)
    if (p >= 0 && toks[p].k === 'id' && toks[p].v === 'def') return 'tk-fndef';
    if (isCall) return 'tk-fn';
    if (!inChain && BUILTINS.indexOf(v) >= 0) return 'tk-fn';
    if (DTYPES.indexOf(v) >= 0) return 'tk-type';

    // 关键字实参：紧贴等号（PEP8 下赋值两侧有空格，实参没有）
    const adj = toks[i + 1];
    if (adj && adj.k === 'op' && adj.v === '=' && !(toks[i + 2] && toks[i + 2].k === 'op' && toks[i + 2].v === '=')) {
      return 'tk-param';
    }

    if (v === 'pl') return 'tk-mod';
    if (/^[A-Z][A-Z0-9_]*$/.test(v) && v.length > 1) return 'tk-const';
    if (/^[A-Z]/.test(v)) return 'tk-type';
    if (inChain) return 'tk-attr';
    return 'tk-var';
  }

  // 点号链：pl.Level.CORE_GROUP 整段作为一个可点击对象
  function chainAt(toks, i) {
    let j = i;
    const parts = [toks[i].v];
    const idx = [i];
    while (toks[j + 1] && toks[j + 1].k === 'punc' && toks[j + 1].v === '.'
      && toks[j + 2] && toks[j + 2].k === 'id') {
      parts.push(toks[j + 2].v);
      idx.push(j + 2);
      j += 2;
    }
    return { parts: parts, idx: idx, end: j };
  }

  function span(cls, text) { return '<span class="' + cls + '">' + esc(text) + '</span>'; }

  function highlight(line) {
    const toks = tokenize(line);
    let out = '';
    for (let i = 0; i < toks.length; i++) {
      const tk = toks[i];
      if (tk.k === 'ws') { out += esc(tk.v); continue; }
      if (tk.k === 'cmt') { out += span('tk-cmt', tk.v); continue; }
      if (tk.k === 'str') { out += span('tk-str', tk.v); continue; }
      if (tk.k === 'dec') { out += span('tk-dec', tk.v); continue; }
      if (tk.k === 'num') { out += span('tk-num', tk.v); continue; }
      if (tk.k === 'punc') { out += span('tk-punc', tk.v); continue; }
      if (tk.k === 'op') { out += span('tk-op', tk.v); continue; }

      // 标识符（可能是点号链）
      const ch = chainAt(toks, i);
      const cid = symbolOf(ch.parts.join('.'));
      let inner = '';
      for (let c = 0; c < ch.idx.length; c++) {
        if (c > 0) inner += span('tk-punc', '.');
        inner += span(classifyId(toks, ch.idx[c], c > 0), toks[ch.idx[c]].v);
      }
      out += cid
        ? '<span class="tk-link" data-concept="' + cid + '" role="link" tabindex="0">' + inner + '</span>'
        : inner;
      i = ch.end;
    }
    return out;
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 约束求值
  // ════════════════════════════════════════════════════════════════════════════
  function evaluate() {
    state.d = M.derive(state.cfg);
    state.results = {};
    M.CONSTRAINTS.forEach(function (c) {
      state.results[c.id] = c.eval(state.d);
    });
  }

  function lineIssues() {
    // 行号 → 最严重的约束
    const map = {};
    M.CONSTRAINTS.forEach(function (c) {
      const res = state.results[c.id];
      c.lines.forEach(function (no) {
        const cur = map[no];
        const rank = { ok: 0, warn: 1, error: 2 };
        if (!cur || rank[res.status] > rank[cur.status]) {
          map[no] = { status: res.status, cid: c.id, title: c.title, detail: res.detail };
        }
      });
    });
    return map;
  }

  function statusSummary() {
    let ok = 0, warn = 0, error = 0;
    M.CONSTRAINTS.forEach(function (c) {
      const s = state.results[c.id].status;
      if (s === 'ok') ok++; else if (s === 'warn') warn++; else error++;
    });
    return { ok: ok, warn: warn, error: error };
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 编辑器
  // ════════════════════════════════════════════════════════════════════════════
  function buildScopeRanges() {
    // 以「渲染行序号」为单位计算 scope 覆盖
    const rows = [];
    SOURCE_ROWS.forEach(function (row, i) {
      if (row.entry.scope) {
        rows.push({ i: i, kind: row.entry.scope.kind, label: row.entry.scope.label, end: i + row.entry.scope.span - 1 });
      }
    });
    return rows;
  }

  let SOURCE_ROWS = [];

  function conceptLines(id) {
    return SOURCE_ROWS.map(function (row) { return row.entry; }).filter(function (e) {
      return e.no && (e.focus === id || highlight(fill(e.t, state.d)).indexOf('data-concept="' + id + '"') !== -1);
    }).map(function (e) { return e.no; });
  }

  function inlineConceptHtml(id) {
    const c = V.CONCEPTS[id];
    if (!c) return '';
    const lines = conceptLines(id);
    return '<section class="inline-concept inline-' + LENS_OF[c.lens] + '" aria-label="' + esc(c.title) + '详情">'
      + '<div class="inline-concept-head"><div><span class="inline-kind"><i></i>' + esc(c.kind) + '</span>'
      + '<h3>' + esc(c.title) + '</h3><p>' + esc(c.sub) + '</p></div><button class="inline-close" type="button" data-close-concept aria-label="关闭对象详情">×</button></div>'
      + '<div class="inline-content"><div class="inline-viz"><div class="inline-viz-label"><span>语义视图</span><span>示意图 · 随配置更新</span></div>' + c.viz(state.d) + '</div>'
      + '<dl class="insp-facts">' + c.facts(state.d).map(function (f) {
        return '<div><dt>' + esc(f[0]) + '</dt><dd>' + esc(f[1]) + '</dd></div>';
      }).join('') + '</dl></div>'
      + (lines.length > 1 ? '<div class="inline-locations"><span>代码落点</span>' + lines.map(function (no) {
        return '<button type="button" class="lineref" data-anchor="' + no + '"' + (no === state.anchor ? ' aria-current="location"' : '') + '>L' + no + '</button>';
      }).join('') + '</div>' : '') + '</section>';
  }

  function renderEditor() {
    const d = state.d;
    const issues = lineIssues();
    const ranges = buildScopeRanges();
    const maxDepth = 4;
    let html = '';

    SOURCE_ROWS.forEach(function (row, i) {
      const e = row.entry;

      if (e.sec) {
        html += '<div class="ed-sec"><span class="ed-sec-label">' + esc(e.sec) + '</span>'
          + '<span class="ed-sec-hint">' + esc(e.hint || '') + '</span></div>';
        return;
      }
      if (e.fold) {
        html += '<div class="ed-fold"><span class="ed-fold-no">⋯</span>'
          + '<span class="ed-fold-body">' + e.fold + ' 行已折叠</span></div>';
        return;
      }

      // ribbons
      let rb = '';
      let depth = 0;
      ranges.forEach(function (rg) {
        if (i >= rg.i && i <= rg.end) {
          rb += '<i class="rb rb-' + rg.kind + (i === rg.i ? ' is-start' : '')
            + (i === rg.end ? ' is-end' : '') + '" style="--rb-depth:' + depth + '"></i>';
          depth++;
        }
      });
      for (let k = depth; k < maxDepth; k++) rb += '<i class="rb rb-void"></i>';

      // inlays
      const inlays = (e.inlay || []).slice();
      if (e.scope) inlays.unshift({ lens: 'scope', text: e.scope.label, strong: true });
      let chips = '';
      inlays.forEach(function (c) {
        const group = LENS_OF[c.lens] || c.lens;
        chips += '<span class="chip chip-' + c.lens + (c.strong ? ' is-strong' : '')
          + '" data-lens="' + group + '">' + esc(fill(c.text, d)) + '</span>';
      });

      const iss = issues[e.no];
      const cls = ['ed-line'];
      if (iss && iss.status !== 'ok') cls.push('has-' + iss.status);
      if (e.focus && state.sel.type === 'concept' && state.sel.id === e.focus) cls.push('is-linked');
      if (iss && state.sel.type === 'constraint' && state.sel.id === iss.cid) cls.push('is-flagged');

      html += '<div class="' + cls.join(' ') + '" data-no="' + e.no + '"'
        + (e.focus ? ' data-focus="' + e.focus + '"' : '') + '>'
        + '<span class="ed-no">' + e.no + '</span>'
        + '<span class="ed-mark">'
        + (iss && iss.status !== 'ok'
          ? '<button class="mark mark-' + iss.status + '" type="button" data-constraint="' + iss.cid
            + '" title="' + esc(iss.title) + '" aria-label="' + esc(iss.title) + '">'
            + (iss.status === 'error' ? '✕' : '!') + '</button>'
          : '')
        + '</span>'
        + '<span class="ed-rb">' + rb + '</span>'
        + '<code class="ed-code">' + highlight(fill(e.t, d)) + '</code>'
        + '<span class="ed-inlay">' + chips + '</span>'
        + '</div>';
      if (state.sel.type === 'concept' && e.no === state.anchor) html += inlineConceptHtml(state.sel.id);
    });

    $('#editorBody').innerHTML = html;
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 左栏
  // ════════════════════════════════════════════════════════════════════════════
  function renderLensPanel() {
    let html = '';
    LENSES.forEach(function (L) {
      const on = state.lens[L.id];
      html += '<button class="lens-row' + (on ? ' is-on' : '') + '" type="button" data-lens-toggle="' + L.id + '"'
        + ' aria-pressed="' + on + '" title="' + esc(L.desc) + ' · 点击' + (on ? '隐藏' : '显示') + '代码标注">'
        + '<i class="lens-swatch lens-' + L.id + '"></i>'
        + '<span class="lens-text"><b>' + esc(L.label) + '</b></span>'
        + '</button>';
    });
    $('#lensPanel').innerHTML = html;
  }

  function renderCatalog() {
    const groups = {};
    Object.keys(V.CONCEPTS).forEach(function (id) {
      const c = V.CONCEPTS[id];
      const g = LENS_OF[c.lens];
      (groups[g] = groups[g] || []).push({ id: id, c: c });
    });
    let html = '';
    LENSES.forEach(function (L) {
      const items = groups[L.id] || [];
      html += '<div class="cat-group"><span class="cat-head"><i class="lens-swatch lens-' + L.id + '"></i>'
        + esc(L.label) + '</span>';
      items.forEach(function (it) {
        const on = state.sel.type === 'concept' && state.sel.id === it.id;
        html += '<button class="cat-row' + (on ? ' is-active' : '') + '" type="button" data-concept="' + it.id + '">'
          + '<b>' + esc(it.c.title) + '</b></button>';
      });
      html += '</div>';
    });
    $('#catalog').innerHTML = html;
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 底部：场景与算子配置
  // ════════════════════════════════════════════════════════════════════════════
  function renderKnobs() {
    let html = '';
    const groups = [
      { title: '工作负载', ids: ['BATCH'] },
      { title: '模型维度', note: '通常由模型决定；此处可调用于演示。', ids: ['NUM_KV_HEADS', 'HEAD_DIM'] },
      { title: '运行时', ids: ['PAGE_SIZE'] },
      { title: '算子策略', ids: ['SEQ_TILE', 'QKV_OK', 'DOWN_TN', 'GP_SIZE', 'NUM_CORES', 'ROPE_CORES'] }
    ];
    groups.forEach(function (group) {
      html += '<section class="knob-group" aria-label="' + esc(group.title) + '"><div class="knob-group-title">'
        + esc(group.title) + (group.note ? '<span>' + esc(group.note) + '</span>' : '') + '</div><div class="knob-group-fields">';
      group.ids.forEach(function (id) {
        const k = M.KNOBS.find(function (item) { return item.id === id; });
        html += '<div class="knob"><div class="knob-label"><label for="knob-' + k.id + '">' + esc(k.label)
          + '</label><span class="knob-help"><button type="button" class="knob-help-button" aria-label="' + esc(k.label) + '说明" aria-describedby="help-' + k.id + '">?</button>'
          + '<span class="knob-tooltip" role="tooltip" id="help-' + k.id + '">' + esc(k.note) + '</span></span></div>'
          + '<select id="knob-' + k.id + '" data-knob="' + k.id + '">'
          + k.options.map(function (o) {
            return '<option value="' + o + '"' + (state.cfg[k.id] === o ? ' selected' : '') + '>' + o + '</option>';
          }).join('')
          + '</select></div>';
      });
      html += '</div></section>';
    });
    $('#knobGrid').innerHTML = html;
  }

  // 面板头的判定计数 + 底部状态栏
  function renderStatus() {
    const sum = statusSummary();
    $('#problemCount').innerHTML =
      '<i class="dot dot-error"></i>' + sum.error
      + '<i class="dot dot-warn"></i>' + sum.warn
      + '<i class="dot dot-ok"></i>' + sum.ok;

    const st = sum.error ? 'error' : (sum.warn ? 'warn' : 'ok');
    const strip = $('#statusMain');
    strip.className = 'status-main is-' + st;
    strip.textContent = sum.error
      ? sum.error + ' 条硬件约束在编码期即被判定违反'
      : (sum.warn ? sum.warn + ' 条约束可通过但有代价' : '全部 ' + sum.ok + ' 条硬件约束在编码期已满足');
  }

  // 当前选中对象携带的约束；没有则返回 null
  function relatedIds() {
    if (state.sel.type !== 'concept') return null;
    const ids = M.CONSTRAINTS
      .filter(function (k) { return CONSTRAINT_CONCEPT[k.id] === state.sel.id; })
      .map(function (k) { return k.id; });
    return ids.length ? ids : null;
  }

  // 右栏下段：约束区（始终列全 9 条，与当前对象相关的那几条高亮，其余压暗）
  function consHtml() {
    const sum = statusSummary();
    const total = M.CONSTRAINTS.length;
    const rel = relatedIds();
    const order = { error: 0, warn: 1, ok: 2 };
    const list = M.CONSTRAINTS.slice().sort(function (a, b) {
      return order[state.results[a.id].status] - order[state.results[b.id].status];
    });

    let html = '<div class="hn-sec-label">' + (rel ? '相关约束' : '硬件约束')
      + '<small>' + (rel ? rel.length + ' / ' + total : total + ' 条') + '</small></div>';

    html += '<div class="hn-filters" role="group" aria-label="筛选硬件约束">'
      + [['all', '全部', total], ['error', '违反', sum.error], ['warn', '有代价', sum.warn], ['ok', '已满足', sum.ok]]
        .map(function (it) {
          return '<button type="button" data-filter="' + it[0] + '" aria-pressed="'
            + (state.filter === it[0]) + '">' + it[1] + ' <span>' + it[2] + '</span></button>';
        }).join('')
      + '</div>';

    let rows = '';
    list.filter(function (c) {
      return state.filter === 'all' || state.results[c.id].status === state.filter;
    }).forEach(function (c) {
      const res = state.results[c.id];
      const on = state.sel.type === 'constraint' && state.sel.id === c.id;
      const isRel = !!rel && rel.indexOf(c.id) >= 0;
      const cls = ['prob'];
      if (on) cls.push('is-active');
      if (isRel) cls.push('is-related');
      if (rel && !isRel) cls.push('is-faded');
      rows += '<button class="' + cls.join(' ') + '" type="button" data-constraint="' + c.id + '">'
        + '<i class="dot dot-' + res.status + '"></i>'
        + '<span class="prob-main"><b>' + esc(c.title)
        + (isRel ? '<em class="prob-rel">相关</em>' : '') + '</b>'
        + '<small>' + esc(res.detail) + '</small></span>'
        + '<span class="prob-side"><code>' + esc(res.value) + '</code>'
        + '<em class="tag tag-' + res.status + '">' + STATUS_LABEL[res.status] + '</em></span>'
        + '</button>';
    });
    html += '<div class="hn-problem-list">'
      + (rows || '<p class="hn-empty">当前没有此状态的约束。</p>') + '</div>';
    return html;
  }

  function renderCons() {
    const host = $('#consSection');
    if (host) host.innerHTML = consHtml();
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 右栏：一列到底 —— 当前对象在上，约束区在下
  // ════════════════════════════════════════════════════════════════════════════
  function metric(label, value, note) {
    return '<div class="metric"><span>' + esc(label) + '</span><b>' + esc(value) + '</b>'
      + '<small>' + esc(note) + '</small></div>';
  }

  function overviewHtml() {
    const d = state.d;
    let html = '';
    html += '<div class="insp-metrics">'
      + metric('真实块 / 核', d.REAL_BLOCKS + ' / ' + d.NUM_CORES, d.SKEW_FMT + '× 偏差')
      + metric('Q 填充占比', d.PAD_WASTE_PCT + '%', d.Q_HEAD_BATCH + ' 行有效 / ' + d.Q_HEAD_PAD + ' 行物理')
      + metric('K tile 双缓冲', d.K_TILE_DB_KB + ' KB', 'Right 上限 64 KB')
      + metric('q_proj task', String(d.Q_TASK_TOTAL), d.Q_ON + ' N-tile × ' + d.QKV_OK + ' split-K')
      + '</div>';

    return html;
  }

  function constraintHtml(id) {
    const c = M.CONSTRAINTS.filter(function (k) { return k.id === id; })[0];
    if (!c) return null;
    const res = state.results[id];
    const d = state.d;
    const conceptId = CONSTRAINT_CONCEPT[id];
    const concept = conceptId ? V.CONCEPTS[conceptId] : null;
    let html = '';

    html += '<div class="insp-tag"><i class="dot dot-' + res.status + '"></i>'
      + STATUS_LABEL[res.status] + '</div>';
    html += '<h3 class="insp-title">' + esc(c.title) + '</h3>';
    html += '<pre class="insp-rule">' + esc(c.rule) + '</pre>';
    html += '<div class="insp-verdict is-' + res.status + '"><b>' + esc(res.value) + '</b>'
      + '<span>' + esc(res.detail) + '</span></div>';
    if (concept) {
      html += '<div class="insp-viz">' + concept.viz(d) + '</div>';
      html += '<p class="insp-sub insp-sub--link">相关对象：'
        + '<button class="inlink" type="button" data-concept="' + conceptId + '">'
        + esc(concept.title) + '</button></p>';
    }

    html += '<div class="insp-block"><span class="eyebrow">硬件事实</span>'
      + '<p class="insp-p">' + esc(c.origin) + '</p></div>';

    html += '<div class="insp-block"><span class="eyebrow">代码落点</span><div class="insp-lines">'
      + c.lines.map(function (no) {
        return '<button class="lineref" type="button" data-goto="' + no + '">L' + no + '</button>';
      }).join('') + '</div></div>';
    return html;
  }

  // keepScroll：改旋钮引起的重算不该把右栏滚回顶部；换选中对象才回顶部
  function renderRight(keepScroll) {
    let body = null;
    let title = '总览';
    if (state.sel.type === 'concept') { body = ''; title = '硬件约束'; }
    else if (state.sel.type === 'constraint') { body = constraintHtml(state.sel.id); title = '当前约束'; }
    if (body == null) {
      state.sel = { type: 'overview', id: null };
      body = overviewHtml();
      title = '总览';
    }
    const host = $('#rightBody');
    const top = keepScroll ? host.scrollTop : 0;
    $('#rightTitle').textContent = title;
    host.innerHTML = '<div class="hn-obj">' + body + '</div>'
      + '<section class="hn-cons-sec" id="consSection" aria-label="硬件约束"></section>';
    renderCons();
    host.scrollTop = top;
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 选择与渲染编排
  // ════════════════════════════════════════════════════════════════════════════
  function select(type, id, opts) {
    const active = document.activeElement;
    const focusAttr = active && active.hasAttribute
      ? (active.hasAttribute('data-constraint') ? 'data-constraint'
        : active.hasAttribute('data-concept') ? 'data-concept' : null)
      : null;
    const focusValue = focusAttr && active.getAttribute(focusAttr);
    // 重渲染后焦点要落回「点击来源那一栏」的同名元素，而不是任意第一个
    const focusHost = focusAttr && active.closest('#rightBody, #hnExplorer, #editorBody');
    const focusHostId = focusHost ? focusHost.id : null;

    state.sel = { type: type, id: id };
    const lines = type === 'concept' ? conceptLines(id) : [];
    state.anchor = lines.indexOf(opts && opts.anchor) >= 0 ? opts.anchor : (lines[0] || null);
    // 选中语义对象会换掉「约束区在讲什么」，筛选随之复位；
    // 在约束之间跳转只是同一个列表内的导航，筛选保留。
    if (type === 'concept') state.filter = 'all';
    const inspectorToggle = $('[data-ide-toggle="inspector"]');
    if (type === 'constraint' && inspectorToggle.getAttribute('aria-expanded') === 'false') inspectorToggle.click();
    renderEditor();
    renderCatalog();
    renderRight();
    if (focusAttr && focusValue) {
      const q = '[' + focusAttr + '="' + focusValue + '"]';
      const scope = focusHostId ? $('#' + focusHostId) : null;
      const replacement = (scope && $(q, scope)) || document.querySelector(q);
      if (replacement) replacement.focus({ preventScroll: true });
    }
    if (opts && opts.scroll) scrollToSelection();
  }

  function scrollToSelection() {
    let target = null;
    if (state.sel.type === 'constraint') {
      const c = M.CONSTRAINTS.filter(function (k) { return k.id === state.sel.id; })[0];
      if (c) target = $('#editorBody .ed-line[data-no="' + c.lines[0] + '"]');
    } else if (state.sel.type === 'concept') {
      target = $('#editorBody .ed-line[data-no="' + state.anchor + '"]');
    }
    if (target) target.scrollIntoView({ block: 'center', behavior: 'smooth' });
  }

  function gotoLine(no) {
    const el = $('#editorBody .ed-line[data-no="' + no + '"]');
    if (el) {
      el.scrollIntoView({ block: 'center', behavior: 'smooth' });
      el.classList.add('is-ping');
      setTimeout(function () { el.classList.remove('is-ping'); }, 900);
    }
  }

  function applyLensAttrs() {
    LENSES.forEach(function (L) {
      document.documentElement.setAttribute('data-lens-' + L.id, state.lens[L.id] ? 'on' : 'off');
    });
  }

  function renderAll() {
    evaluate();
    renderEditor();
    renderLensPanel();
    renderCatalog();
    renderKnobs();
    renderStatus();
    renderRight(true);
    const changed = M.KNOBS.filter(function (k) { return state.cfg[k.id] !== M.DEFAULTS[k.id]; }).length;
    $('#configChanges').textContent = changed ? '已修改 ' + changed + ' 项' : '默认配置';
    $('#envShape').textContent = state.d.BATCH + '×' + state.d.HIDDEN + ' · ' + state.d.NUM_KV_HEADS + ' KV heads';
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 事件
  // ════════════════════════════════════════════════════════════════════════════
  function wire() {
    document.addEventListener('click', function (ev) {
      const close = ev.target.closest('[data-close-concept]');
      if (close) {
        const id = state.sel.id;
        select('overview', null);
        const item = $('#catalog [data-concept="' + id + '"]');
        if (item) item.focus({ preventScroll: true });
        return;
      }
      const anchor = ev.target.closest('[data-anchor]');
      if (anchor) {
        const no = Number(anchor.getAttribute('data-anchor'));
        select('concept', state.sel.id, { anchor: no, scroll: true });
        const replacement = $('[data-anchor="' + no + '"]');
        if (replacement) replacement.focus({ preventScroll: true });
        return;
      }
      const filter = ev.target.closest('[data-filter]');
      if (filter) {
        state.filter = filter.getAttribute('data-filter');
        renderCons();
        const back = $('[data-filter="' + state.filter + '"]');
        if (back) back.focus({ preventScroll: true });
        return;
      }
      const scenario = ev.target.closest('[data-scenario]');
      if (scenario) {
        const presets = { default: {}, capacity: { SEQ_TILE: 256 }, alignment: { QKV_OK: 4 }, parallel: { ROPE_CORES: 24 } };
        state.cfg = Object.assign({}, M.DEFAULTS, presets[scenario.getAttribute('data-scenario')]);
        renderAll(); pulse();
        return;
      }
      const lensBtn = ev.target.closest('[data-lens-toggle]');
      if (lensBtn) {
        const id = lensBtn.getAttribute('data-lens-toggle');
        state.lens[id] = !state.lens[id];
        applyLensAttrs();
        renderLensPanel();
        $('[data-lens-toggle="' + id + '"]').focus();
        return;
      }
      const consBtn = ev.target.closest('[data-constraint]');
      if (consBtn) {
        select('constraint', consBtn.getAttribute('data-constraint'), { scroll: true });
        return;
      }
      const conBtn = ev.target.closest('[data-concept]');
      if (conBtn) {
        const sourceLine = conBtn.closest('.ed-line');
        select('concept', conBtn.getAttribute('data-concept'), { anchor: sourceLine ? Number(sourceLine.dataset.no) : null, scroll: true });
        return;
      }
      const go = ev.target.closest('[data-goto]');
      if (go) { gotoLine(Number(go.getAttribute('data-goto'))); return; }

      if (ev.target.closest('#resetAll')) { reset(); return; }
      if (ev.target.closest('#backOverview')) { select('overview', null); return; }

      const line = ev.target.closest('#editorBody .ed-line');
      if (line && line.getAttribute('data-focus')) {
        select('concept', line.getAttribute('data-focus'), { anchor: Number(line.dataset.no), scroll: true });
      }
    });

    document.addEventListener('keydown', function (ev) {
      if (ev.key !== 'Enter' && ev.key !== ' ') return;
      const link = ev.target.closest && ev.target.closest('.tk-link');
      if (link) { ev.preventDefault(); const line = link.closest('.ed-line'); select('concept', link.getAttribute('data-concept'), { anchor: line ? Number(line.dataset.no) : null, scroll: true }); }
    });

    document.addEventListener('change', function (ev) {
      const knob = ev.target.closest('[data-knob]');
      if (!knob) return;
      const id = knob.getAttribute('data-knob');
      state.cfg[id] = Number(knob.value);
      renderAll();
      $('[data-knob="' + id + '"]').focus();
      pulse();
    });
  }

  function pulse() {
    const el = $('#statusStrip');
    el.classList.add('is-pulse');
    setTimeout(function () { el.classList.remove('is-pulse'); }, 600);
  }

  function reset() {
    state.cfg = Object.assign({}, M.DEFAULTS);
    state.lens = { tensor: true, scope: true, parallel: true, layout: true };
    state.sel = { type: 'overview', id: null };
    state.anchor = null;
    state.filter = 'all';
    applyLensAttrs();
    renderAll();
    $('#rightBody').scrollTop = 0;
  }

  // ── 启动 ───────────────────────────────────────────────────────────────────
  function init() {
    SOURCE_ROWS = M.SOURCE.map(function (e) { return { entry: e }; });
    applyLensAttrs();
    wire();
    renderAll();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
