/* Pass Transform Explorer
 *
 * The timeline and the evidence cards come precomputed from build.mjs, so the
 * app is interactive immediately. Diffs and graphs are computed live in the
 * browser from the IR snapshots themselves (lib/bundle.js is the same parser
 * the build uses), which is what lets any function, any lens and any pass be
 * inspected at full fidelity without shipping a giant precomputed blob.
 */
(function () {
  'use strict';

  var LIB = window.PTXLib;
  var INDEX = window.PTX_INDEX;

  // ── snapshot loading ──────────────────────────────────────────────────
  var SRC = Object.create(null);
  var WAITING = Object.create(null);

  window.PTX = {
    src: function (runId, idx, text) {
      var key = runId + ':' + idx;
      SRC[key] = text;
      (WAITING[key] || []).forEach(function (fn) { fn(text); });
      delete WAITING[key];
    },
  };

  /**
   * Snapshots arrive one of two ways: as `data/<run>/NN.js` next to the page,
   * or - in the single-file demo - gzip+base64 inside the document itself.
   * Either way they are fetched one at a time, only when a view needs them.
   */
  function loadSnapshot(runId, idx) {
    var key = runId + ':' + idx;
    if (SRC[key]) return Promise.resolve(SRC[key]);

    if (window.PTX_EMBEDDED) {
      var packed = window.PTX_EMBEDDED[key];
      if (!packed) return Promise.reject(new Error('这份 demo 未内嵌快照 ' + key));
      return gunzipBase64(packed).then(function (text) {
        SRC[key] = text;
        return text;
      });
    }

    return new Promise(function (resolve, reject) {
      if (WAITING[key]) { WAITING[key].push(resolve); return; }
      WAITING[key] = [resolve];
      var s = document.createElement('script');
      s.src = 'data/' + runId + '/' + String(idx).padStart(2, '0') + '.js';
      s.onerror = function () {
        delete WAITING[key];
        reject(new Error('无法加载快照 ' + s.src + '（若通过 file:// 打开，请改用本地静态服务器）'));
      };
      document.head.appendChild(s);
    });
  }

  function gunzipBase64(b64) {
    if (typeof DecompressionStream === 'undefined') {
      return Promise.reject(new Error('这个浏览器不支持 DecompressionStream，无法解压内嵌快照。请改用 Chrome 80+ / Firefox 113+ / Safari 16.4+。'));
    }
    var bin = atob(b64);
    var bytes = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    var stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    return new Response(stream).text();
  }

  // Parsing a snapshot costs ~50ms, so a handful are kept around; stepping
  // back and forth along the timeline then stays instant.
  var ANALYSIS = new Map();
  var ANALYSIS_CAP = 8;

  function analyze(runId, idx) {
    var key = runId + ':' + idx;
    if (ANALYSIS.has(key)) {
      var hit = ANALYSIS.get(key);
      ANALYSIS.delete(key);
      ANALYSIS.set(key, hit);
      return Promise.resolve(hit);
    }
    return loadSnapshot(runId, idx).then(function (text) {
      var lines = text.split(/\r?\n/);
      var an = LIB.analyzeProgram(LIB.parseDump(text, String(idx)), lines);
      an.lines = lines;
      an.text = text;
      ANALYSIS.set(key, an);
      while (ANALYSIS.size > ANALYSIS_CAP) ANALYSIS.delete(ANALYSIS.keys().next().value);
      return an;
    });
  }

  // ── tiny DOM helpers ──────────────────────────────────────────────────
  function $(id) { return document.getElementById(id); }
  var esc = LIB.escapeHtml;
  var mdInline = LIB.mdInline;
  var md = LIB.md;
  function fmt(n) { return typeof n === 'number' ? n.toLocaleString('en-US') : n; }
  function bytes(n) { return LIB.fmtBytes(n); }

  function toast(msg) {
    var t = $('toast');
    t.textContent = msg;
    t.hidden = false;
    clearTimeout(toast._t);
    toast._t = setTimeout(function () { t.hidden = true; }, 4200);
  }

  // ── state ─────────────────────────────────────────────────────────────
  var state = {
    runId: INDEX.runs[0].id,
    mode: 'pass',          // 'pass' = per-Pass timeline, 'callable' = per-callable lineage
    passIdx: 1,
    tab: 'overview',
    fn: null,
    lens: null,
    diffMode: 'split',
    onlyChanged: false,
    jumpLine: null,
    filter: '',
    docOpen: true,
    lensAuto: true,
    callable: null,
    cFilter: '',
    onlyKernels: false,
  };

  function run() { return INDEX.runs.find(function (r) { return r.id === state.runId; }); }
  function pass() { return run().passes.find(function (p) { return p.idx === state.passIdx; }); }
  function phase(id) { return INDEX.phases.find(function (p) { return p.id === id; }) || { label: id, hint: '' }; }

  // ══════════════════════════════════════════════════════════════════════
  // Timeline
  // ══════════════════════════════════════════════════════════════════════

  function renderRail() {
    var r = run();
    var changedCount = r.passes.filter(function (p) { return p.idx > 0 && p.changed; }).length;
    var total = r.passes.length - 1;
    $('railSummary').innerHTML = '<strong>' + changedCount + '</strong> / ' + total
      + ' 个 Pass 改动了 IR<span class="ptx-rail__sub">' + (total - changedCount) + ' 个对本算子为空操作</span>';

    var maxChurn = 1;
    r.passes.forEach(function (p) { maxChurn = Math.max(maxChurn, p.diff.add + p.diff.del); });

    var filter = state.filter.toLowerCase();
    var html = '';
    var lastPhase = null;

    r.passes.forEach(function (p) {
      if (state.onlyChanged && p.idx > 0 && !p.changed) return;
      if (filter && p.name.toLowerCase().indexOf(filter) < 0) return;

      if (p.phase !== lastPhase) {
        var ph = phase(p.phase);
        html += '<div class="ptx-phasehead" title="' + esc(ph.hint) + '">' + esc(ph.label) + '</div>';
        lastPhase = p.phase;
      }

      var churn = p.diff.add + p.diff.del;
      var w = churn ? Math.max(3, Math.round((churn / maxChurn) * 100)) : 0;
      var addW = churn ? Math.round((p.diff.add / churn) * w) : 0;

      html += '<button class="ptx-pass' + (p.idx === state.passIdx ? ' is-active' : '')
        + (p.idx > 0 && !p.changed ? ' is-noop' : '') + '" data-idx="' + p.idx + '">'
        + '<span class="ptx-pass__idx">' + String(p.idx).padStart(2, '0') + '</span>'
        + '<span class="ptx-pass__body">'
        + '<span class="ptx-pass__name">' + esc(p.name === 'frontend' ? '前端 IR' : p.name) + '</span>'
        + '<span class="ptx-pass__bar">'
        + '<i class="ptx-pass__bar-add" style="width:' + addW + '%"></i>'
        + '<i class="ptx-pass__bar-del" style="width:' + (w - addW) + '%"></i>'
        + '</span></span>'
        + '<span class="ptx-pass__churn">' + (churn ? fmt(churn) : '—') + '</span>'
        + '</button>';
    });

    $('passList').innerHTML = html || '<p class="ptx-empty">没有匹配的 Pass。</p>';
    var active = $('passList').querySelector('.is-active');
    if (active) active.scrollIntoView({ block: 'nearest' });
  }

  // ══════════════════════════════════════════════════════════════════════
  // Pass header
  // ══════════════════════════════════════════════════════════════════════

  function renderHeader() {
    var p = pass();
    var ph = phase(p.phase);
    $('passPhase').textContent = ph.label;
    $('passPhase').title = ph.hint;
    $('passName').textContent = p.name === 'frontend' ? '前端 IR（Pass 流水线输入）' : p.name;
    $('passHeadline').textContent = p.headline;

    var d = p.diff;
    $('passDelta').innerHTML = p.idx === 0 ? '<span class="ptx-muted">流水线起点</span>'
      : (d.add || d.del)
        ? '<b class="ptx-add">+' + fmt(d.add) + '</b> <b class="ptx-del">−' + fmt(d.del) + '</b> 行'
        : '<span class="ptx-noop">未改动 IR</span>';

    // In the single-file demo the repo is not alongside the page, so the path
    // is shown but not offered as a link that would 404.
    var link = $('passSource');
    if (p.source) {
      link.hidden = false;
      link.textContent = p.source.split('/').pop();
      if (window.PTX_STANDALONE) {
        link.removeAttribute('href');
        link.title = '仓库内路径：' + p.source;
        link.classList.add('is-inert');
      } else {
        link.href = '../../' + p.source;
        link.title = p.source;
      }
    } else {
      link.hidden = true;
    }

    document.querySelectorAll('.ptx-tab').forEach(function (b) {
      b.classList.toggle('is-active', b.dataset.tab === state.tab);
    });
    ['overview', 'diff', 'graph'].forEach(function (t) {
      $('view' + t[0].toUpperCase() + t.slice(1)).hidden = state.tab !== t;
    });
  }

  // ══════════════════════════════════════════════════════════════════════
  // Overview: evidence cards + function change table
  // ══════════════════════════════════════════════════════════════════════

  var TONE_CLASS = { add: 'is-add', remove: 'is-del', change: 'is-chg', neutral: '' };

  function renderOverview() {
    var p = pass();
    var m = p.metrics;
    var prev = run().passes.find(function (x) { return x.idx === p.idx - 1; });
    var pm = prev ? prev.metrics : null;

    var html = '<div class="ptx-metrics">' + [
      ['语句', m.stmts], ['函数', m.functions], ['循环', m.loops],
      ['Tile 值', m.tiles], ['Tensor 值', m.tensors],
      ['tile.alloc', m.allocs], ['片上内存', m.allocBytes, 'bytes'],
      ['任务', m.tasks], ['最大嵌套', m.maxNest],
    ].map(function (row) {
      var v = row[2] === 'bytes' ? bytes(row[1]) : fmt(row[1]);
      var d = pm ? row[1] - pm[metricKey(row[0])] : 0;
      return '<div class="ptx-metric"><span class="ptx-metric__label">' + row[0] + '</span>'
        + '<span class="ptx-metric__value">' + v + '</span>'
        + (d ? '<span class="ptx-metric__delta ' + (d > 0 ? 'ptx-add' : 'ptx-del') + '">'
          + (d > 0 ? '+' : '−') + (row[2] === 'bytes' ? bytes(Math.abs(d)) : fmt(Math.abs(d))) + '</span>' : '')
        + '</div>';
    }).join('') + '</div>';

    if (!p.evidence.length) {
      html += p.idx === 0
        ? '<p class="ptx-note">这是 Pass 流水线的输入快照。切到「结构图」可以先看清算子本身的结构，再沿时间线逐个 Pass 往下走。</p>'
        : '<p class="ptx-note ptx-note--noop">本 Pass 在这个算子上是空操作：逐行比对前后快照完全一致。这本身是有用的结论——它说明该 Pass 的触发条件没有在这段 IR 上命中。</p>';
    }

    p.evidence.forEach(function (card) {
      html += '<section class="ptx-card ' + (TONE_CLASS[card.tone] || '') + '">'
        + '<h3>' + esc(card.title) + '</h3>'
        + '<p class="ptx-card__headline">' + mdInline(card.headline) + '</p>';

      if (card.rows && card.rows.length) {
        html += '<table class="ptx-table"><thead><tr><th></th><th>之前</th><th>之后</th><th>Δ</th></tr></thead><tbody>';
        card.rows.forEach(function (r) {
          var d = r.after - r.before;
          var f = r.fmt === 'bytes' ? bytes : fmt;
          html += '<tr><td>' + esc(r.label) + '</td><td>' + f(r.before) + '</td><td>' + f(r.after) + '</td>'
            + '<td class="' + (d > 0 ? 'ptx-add' : d < 0 ? 'ptx-del' : 'ptx-muted') + '">'
            + (d > 0 ? '+' : d < 0 ? '−' : '') + (d ? f(Math.abs(d)) : '0') + '</td></tr>';
        });
        html += '</tbody></table>';
      }

      if (card.items && card.items.length) {
        html += '<ul class="ptx-items">' + card.items.map(function (it) {
          return '<li class="' + (TONE_CLASS[it.tone] || '') + '"><code>' + esc(it.label) + '</code>'
            + '<span>' + esc(it.note) + '</span>'
            + (it.delta ? '<b>' + (it.delta > 0 ? '+' : '−') + Math.abs(it.delta) + '</b>' : '') + '</li>';
        }).join('') + '</ul>';
        if (card.more) html += '<p class="ptx-more">另有 ' + card.more + " 项，见「代码 Diff」</p>";
      }

      if (card.subs && card.subs.length) {
        html += '<table class="ptx-table ptx-table--subs"><thead><tr><th>次数</th><th>改写前</th><th>改写后</th><th>示例</th></tr></thead><tbody>';
        card.subs.forEach(function (s) {
          html += '<tr><td class="ptx-num">' + s.count + '</td>'
            + '<td><code class="ptx-del-code">' + esc(trunc(s.from, 46)) + '</code></td>'
            + '<td><code class="ptx-add-code">' + esc(trunc(s.to, 46)) + '</code></td>'
            + '<td class="ptx-ex" title="' + esc(s.example.after) + '"><code>' + esc(trunc(s.example.after, 70)) + '</code></td></tr>';
        });
        html += '</tbody></table>'
          + '<p class="ptx-more">' + card.stats.paired + ' 行成对改写 · '
          + card.stats.pureAdd + ' 行纯新增 · ' + card.stats.pureDel + ' 行纯删除 · '
          + card.stats.distinct + ' 种不同改写</p>';
      }
      html += '</section>';
    });

    if (p.changedFunctions && p.changedFunctions.length) {
      html += '<section class="ptx-card"><h3>受影响的函数</h3>'
        + '<p class="ptx-card__headline">' + p.changedFunctions.length + ' 个函数被改动，'
        + p.sameFunctions + ' 个完全未变</p>'
        + '<table class="ptx-table ptx-table--fns"><thead><tr><th>函数</th><th>状态</th><th>语句</th><th>行数</th><th></th></tr></thead><tbody>';
      p.changedFunctions.slice(0, 60).forEach(function (f) {
        html += '<tr><td><code>' + esc(f.name) + '</code>'
          + (f.kind ? '<em>' + esc(f.kind) + '</em>' : '') + '</td>'
          + '<td><span class="ptx-status ptx-status--' + f.status + '">' + statusLabel(f.status) + '</span></td>'
          + '<td>' + deltaCell(f.stmtsBefore, f.stmtsAfter) + '</td>'
          + '<td>' + deltaCell(f.linesBefore, f.linesAfter) + '</td>'
          + '<td><button class="ptx-linkbtn" data-openfn="' + esc(f.name) + '">查看 Diff</button></td></tr>';
      });
      html += '</tbody></table>';
      if (p.changedFunctions.length > 60) html += '<p class="ptx-more">仅列出前 60 个</p>';
      html += '</section>';
    }

    $('viewOverview').innerHTML = html;
  }

  function metricKey(label) {
    return { 语句: 'stmts', 函数: 'functions', 循环: 'loops', 'Tile 值': 'tiles', 'Tensor 值': 'tensors', 'tile.alloc': 'allocs', 片上内存: 'allocBytes', 任务: 'tasks', 最大嵌套: 'maxNest' }[label];
  }
  function statusLabel(s) { return { added: '新增', removed: '移除', changed: '改写', same: '未变' }[s] || s; }
  function deltaCell(before, after) {
    if (before === after) return fmt(after);
    return '<span class="ptx-muted">' + fmt(before) + '</span> → <b>' + fmt(after) + '</b>';
  }
  function trunc(s, n) { s = String(s); return s.length > n ? s.slice(0, n - 1) + '…' : s; }

  // ══════════════════════════════════════════════════════════════════════
  // Code diff
  // ══════════════════════════════════════════════════════════════════════

  function renderDiff() {
    var p = pass();
    if (p.idx === 0) {
      $('diffFnChips').innerHTML = '';
      $('diffBody').innerHTML = '<p class="ptx-empty">流水线起点没有可比较的上一版本。</p>';
      return;
    }
    if (!p.changed) {
      $('diffFnChips').innerHTML = '';
      $('diffBody').innerHTML = '<p class="ptx-empty">本 Pass 未改动任何一行 IR。</p>';
      return;
    }

    var fns = p.changedFunctions;
    // In callable mode the chips are the family, not every function this Pass
    // happened to touch — otherwise one callable's story is buried in sixty.
    if (state.mode === 'callable') {
      var cc = currentCallable();
      if (cc) {
        var fam = Object.create(null);
        cc.members.forEach(function (m) { fam[m] = 1; });
        var sub = fns.filter(function (f) { return fam[f.name]; });
        if (sub.length) fns = sub;
      }
    }
    if (!state.fn || !fns.some(function (f) { return f.name === state.fn; })) state.fn = fns[0].name;

    $('diffFnChips').innerHTML = fns.map(function (f) {
      var churn = (f.add || 0) + (f.del || 0);
      return '<button class="ptx-chip ptx-chip--' + f.status + (f.name === state.fn ? ' is-active' : '')
        + '" data-fn="' + esc(f.name) + '" title="' + esc(f.name) + ' · ' + statusLabel(f.status) + '">'
        + esc(f.name)
        + (churn ? '<b>' + fmt(churn) + '</b>' : '<b>' + statusLabel(f.status) + '</b>')
        + '</button>';
    }).join('');

    $('diffBody').innerHTML = '<p class="ptx-loading">正在解析前后快照…</p>';
    var token = ++renderDiff._token;

    Promise.all([analyze(state.runId, p.idx - 1), analyze(state.runId, p.idx)])
      .then(function (pair) {
        if (token !== renderDiff._token) return;
        paintDiff(pair[0], pair[1]);
      })
      .catch(function (err) {
        $('diffBody').innerHTML = '<p class="ptx-empty">' + esc(err.message) + '</p>';
        toast(err.message);
      });
  }
  renderDiff._token = 0;

  function paintDiff(before, after) {
    var name = state.fn;
    var fa = before.byName.get(name);
    var fb = after.byName.get(name);
    var srcA = fa ? fa.src : [];
    var srcB = fb ? fb.src : [];
    var baseA = fa ? fa.decoLine : 1;
    var baseB = fb ? fb.decoLine : 1;

    var rows = LIB.diffLines(srcA, srcB);
    var hunks = LIB.toHunks(rows, 3, baseA, baseB);

    if (!hunks.length) {
      $('diffBody').innerHTML = '<p class="ptx-empty">该函数在本 Pass 中未发生变化。</p>';
      return;
    }

    var total = LIB.countChanges(rows);
    var head = '<div class="ptx-diffhead"><code>' + esc(name) + '</code>'
      + '<span class="ptx-add">+' + total.add + '</span><span class="ptx-del">−' + total.del + '</span>'
      + '<span class="ptx-muted">' + hunks.length + ' 处变更 · 前 ' + srcA.length + ' 行 / 后 ' + srcB.length + ' 行</span></div>';

    $('diffBody').innerHTML = head
      + (state.diffMode === 'split' ? splitView(hunks) : unifiedView(hunks));

    if (state.jumpLine) {
      var target = null;
      var cells = $('diffBody').querySelectorAll('.ptx-ln');
      for (var i = 0; i < cells.length; i++) {
        if (Number(cells[i].textContent) === state.jumpLine) { target = cells[i].parentNode; break; }
      }
      if (target) {
        target.classList.add('is-jump');
        target.scrollIntoView({ block: 'center' });
      } else {
        toast('源行 ' + state.jumpLine + ' 不在本 Pass 的变更范围内');
      }
      state.jumpLine = null;
    }
  }

  /** Pair adjacent -/+ runs so replaced lines line up and can be word-diffed. */
  function pairHunkRows(rows) {
    var out = [];
    var i = 0;
    while (i < rows.length) {
      var r = rows[i];
      if (r[0] === '=') { out.push({ kind: '=', a: r, b: r }); i++; continue; }
      var dels = [];
      var adds = [];
      while (i < rows.length && rows[i][0] === '-') { dels.push(rows[i]); i++; }
      while (i < rows.length && rows[i][0] === '+') { adds.push(rows[i]); i++; }
      var n = Math.max(dels.length, adds.length);
      for (var k = 0; k < n; k++) {
        out.push({ kind: dels[k] && adds[k] ? '~' : dels[k] ? '-' : '+', a: dels[k] || null, b: adds[k] || null });
      }
    }
    return out;
  }

  function inlineMarked(runs) {
    return runs.map(function (r) {
      return r[0] ? '<mark>' + esc(r[1]) + '</mark>' : esc(r[1]);
    }).join('');
  }

  function splitView(hunks) {
    return hunks.map(function (h) {
      var body = pairHunkRows(h.rows).map(function (pr) {
        var la = '';
        var lb = '';
        var ca = '';
        var cb = '';
        if (pr.kind === '=') {
          la = pr.a[1]; lb = pr.a[2]; ca = esc(pr.a[3]); cb = esc(pr.a[3]);
        } else if (pr.kind === '~') {
          var w = LIB.wordDiff(pr.a[3], pr.b[3]);
          la = pr.a[1]; lb = pr.b[2];
          ca = inlineMarked(w.left); cb = inlineMarked(w.right);
        } else if (pr.kind === '-') {
          la = pr.a[1]; ca = esc(pr.a[3]);
        } else {
          lb = pr.b[2]; cb = esc(pr.b[3]);
        }
        var cls = pr.kind === '=' ? '' : pr.kind === '~' ? 'is-chg' : pr.kind === '-' ? 'is-del' : 'is-add';
        return '<tr class="' + cls + '">'
          + '<td class="ptx-ln">' + (la || '') + '</td>'
          + '<td class="ptx-code ' + (pr.kind === '+' ? 'is-blank' : '') + '"><pre>' + ca + '</pre></td>'
          + '<td class="ptx-ln">' + (lb || '') + '</td>'
          + '<td class="ptx-code ' + (pr.kind === '-' ? 'is-blank' : '') + '"><pre>' + cb + '</pre></td></tr>';
      }).join('');
      return '<div class="ptx-hunk"><div class="ptx-hunk__head">@@ 前 ' + h.aStart + ' · 后 ' + h.bStart
        + ' @@ <span class="ptx-add">+' + h.add + '</span> <span class="ptx-del">−' + h.del + '</span></div>'
        + '<table class="ptx-difftable ptx-difftable--split"><tbody>' + body + '</tbody></table></div>';
    }).join('');
  }

  function unifiedView(hunks) {
    return hunks.map(function (h) {
      var body = h.rows.map(function (r) {
        var cls = r[0] === '+' ? 'is-add' : r[0] === '-' ? 'is-del' : '';
        var sign = r[0] === '=' ? ' ' : r[0];
        return '<tr class="' + cls + '">'
          + '<td class="ptx-ln">' + (r[1] || '') + '</td>'
          + '<td class="ptx-ln">' + (r[2] || '') + '</td>'
          + '<td class="ptx-sign">' + sign + '</td>'
          + '<td class="ptx-code"><pre>' + esc(r[3]) + '</pre></td></tr>';
      }).join('');
      return '<div class="ptx-hunk"><div class="ptx-hunk__head">@@ 前 ' + h.aStart + ' · 后 ' + h.bStart
        + ' @@ <span class="ptx-add">+' + h.add + '</span> <span class="ptx-del">−' + h.del + '</span></div>'
        + '<table class="ptx-difftable"><tbody>' + body + '</tbody></table></div>';
    }).join('');
  }

  // ══════════════════════════════════════════════════════════════════════
  // Structural graphs
  // ══════════════════════════════════════════════════════════════════════

  var LENSES = [
    { id: 'pass', label: 'Pass 专属', scope: 'fn', hint: '为这一类 Pass 画的专属图：它到底对这个 callable 做了什么。' },
    { id: 'opshift', label: '算子迁移', scope: 'fn', hint: '这一步退掉了哪些算子、换上了哪些。下降类 Pass 的真正动作在这里。' },
    { id: 'spacetime', label: '缓冲生命期', scope: 'fn', hint: '每个缓冲的存活区间。复用把多条短命缓冲并成少数长命缓冲。' },
    { id: 'call', label: '调用 / 作用域', scope: 'program', hint: '函数与调用关系。内联、外提、核拆分在这里最直观。' },
    { id: 'control', label: '控制流', scope: 'fn', hint: '循环 / 分支 / 作用域的嵌套骨架。展开、流水线下降在这里最直观。' },
    { id: 'dataflow', label: '数据流', scope: 'fn', hint: 'SSA def-use 图。算子替换、类型与内存空间变化在这里最直观。' },
    { id: 'task', label: '任务 DAG', scope: 'fn', hint: 'submit / pl.at 任务及其依赖边。依赖推导与通信在这里最直观。' },
    { id: 'memory', label: '内存布局', scope: 'fn', hint: '缓冲区大小、空间归属与地址。复用与分配在这里最直观。' },
  ];

  /**
   * Which view explains this Pass best for a single callable. The index's own
   * `lens` is program-wide advice; in callable mode `call` says nothing about
   * one function, and memory passes are better told as lifetimes than as a
   * buffer table, so both are redirected.
   */
  function suggestedLens(p) {
    if (state.mode !== 'callable') return p.lens;
    if (PASS_VIEWS[p.name]) return 'pass';
    if (p.phase === 'memory') return 'spacetime';
    // `call` says nothing about one function; `dataflow` draws a 180-node SSA
    // graph where the answer is simply "these operators became those".
    if (p.lens === 'call' || p.lens === 'dataflow') return 'opshift';
    return p.lens;
  }

  function renderGraph() {
    var p = pass();
    if (!state.lens) state.lens = suggestedLens(p);

    // Callable mode hides the program-scoped lens: it draws the whole call
    // graph, which is byte-identical no matter which callable is selected.
    var pv = state.mode === 'callable' ? PASS_VIEWS[p.name] : null;
    var avail = (state.mode === 'callable'
      ? LENSES.filter(function (l) { return l.scope !== 'program'; })
      : LENSES
    ).filter(function (l) { return l.id !== 'pass' || pv; })
      .map(function (l) {
        return l.id === 'pass' ? { id: l.id, label: pv.label, scope: l.scope, hint: l.hint } : l;
      });
    var suggest = suggestedLens(p);
    if (!avail.some(function (l) { return l.id === state.lens; })) state.lens = suggest;

    $('lensPicker').innerHTML = avail.map(function (l) {
      return '<button data-lens="' + l.id + '" class="' + (l.id === state.lens ? 'is-active' : '')
        + (l.id === suggest ? ' is-suggested' : '') + '" title="' + esc(l.hint) + '">' + l.label
        + (l.id === suggest ? '<i>推荐</i>' : '') + '</button>';
    }).join('');

    $('graphBody').innerHTML = '<p class="ptx-loading">正在解析前后快照…</p>';
    var token = ++renderGraph._token;
    var needPrev = p.idx > 0;

    Promise.all([needPrev ? analyze(state.runId, p.idx - 1) : Promise.resolve(null), analyze(state.runId, p.idx)])
      .then(function (pair) {
        if (token !== renderGraph._token) return;
        paintGraph(pair[0], pair[1]);
      })
      .catch(function (err) {
        $('graphBody').innerHTML = '<p class="ptx-empty">' + esc(err.message) + '</p>';
        toast(err.message);
      });
  }
  renderGraph._token = 0;

  function paintGraph(before, after) {
    var lens = LENSES.find(function (l) { return l.id === state.lens; });
    var sel = $('graphFn');

    if (lens.scope === 'program') {
      sel.hidden = true;
      var ga = before ? LIB.callGraph(before) : { nodes: [], edges: [] };
      var gb = LIB.callGraph(after);
      drawGraph(merge(ga, gb, function (n) { return n.kind + '|' + n.level + '|' + n.role + '|' + n.stmts; }),
        { hint: lens.hint, kind: 'call' });
      return;
    }

    // Function-scoped lenses. Prefer a function this pass actually changed,
    // but never land on one that is empty under the current lens - opening the
    // recommended lens onto a blank canvas is the worst possible default.
    sel.hidden = false;
    var p = pass();
    var changed = {};
    (p.changedFunctions || []).forEach(function (f) { changed[f.name] = f.status; });

    var cand = after.functions.map(function (f) {
      return { name: f.name, weight: lensWeight(f, state.lens), status: changed[f.name] || 'same' };
    });
    // Callable mode stays inside the family, including when the auto-pick below
    // looks for a function with something to show under the current lens.
    if (state.mode === 'callable') {
      var famC = currentCallable();
      if (famC) {
        var famSet = Object.create(null);
        famC.members.forEach(function (m) { famSet[m] = 1; });
        var famCand = cand.filter(function (x) { return famSet[x.name]; });
        if (famCand.length) cand = famCand;
      }
    }
    var best = cand.slice().sort(function (a, b) {
      var ca = a.status !== 'same' ? 1 : 0;
      var cb = b.status !== 'same' ? 1 : 0;
      if ((a.weight > 0) !== (b.weight > 0)) return a.weight > 0 ? -1 : 1;
      if (ca !== cb) return cb - ca;
      return b.weight - a.weight;
    })[0];

    var current = cand.find(function (c) { return c.name === state.fn; });
    if (!current || (current.weight === 0 && best && best.weight > 0)) {
      state.fn = best ? best.name : (cand[0] && cand[0].name);
      current = cand.find(function (c) { return c.name === state.fn; });
    }

    sel.innerHTML = cand.map(function (c) {
      return '<option value="' + esc(c.name) + '"' + (c.name === state.fn ? ' selected' : '') + '>'
        + esc(c.name)
        + (c.status !== 'same' ? ' · ' + statusLabel(c.status) : '')
        + (c.weight ? ' · ' + c.weight + lensUnit(state.lens) : '')
        + '</option>';
    }).join('');

    var fb = after.byName.get(state.fn);
    var fa = before ? before.byName.get(state.fn) : null;

    // A recommended lens can land on nothing — memory lenses before InitMemRef,
    // the task DAG in a compute kernel. When the lens was picked for the reader
    // rather than by them, fall back to op migration, which always has either a
    // migration to show or a definite "nothing moved" to state.
    if (state.lensAuto && !lensHasContent(state.lens, fa, fb)) {
      state.lens = 'opshift';
      lens = LENSES.find(function (l) { return l.id === state.lens; });
      $('lensPicker').querySelectorAll('button').forEach(function (b) {
        b.classList.toggle('is-active', b.dataset.lens === state.lens);
      });
    }

    if (state.lens === 'pass') {
      var pview = PASS_VIEWS[p.name];
      if (pview && pview.draw === 'outline') { drawOutline(before, after, state.fn, p.name); return; }
      if (pview && pview.draw === 'lowering') { drawLowering(fa, fb, p.name); return; }
    }

    if (state.lens === 'opshift') { drawOpShift(fa, fb, lens.hint); return; }
    if (state.lens === 'spacetime') { drawSpacetime(fa, fb, lens.hint); return; }
    if (state.lens === 'memory') { drawMemory(fa, fb, lens.hint); return; }

    var build = state.lens === 'control' ? LIB.controlTree
      : state.lens === 'dataflow' ? LIB.dataflowGraph
        : LIB.taskGraph;
    var sig = state.lens === 'control'
      ? function (n) { return n.label + '|' + n.detail; }
      : state.lens === 'dataflow'
        ? function (n) { return n.op + '|' + (n.shape || []).join('x') + '|' + n.dtype + '|' + n.space + '|' + n.buffer; }
        : function (n) { return n.label + '|' + n.level + '|' + n.type; };

    var GA = fa ? build(fa) : { nodes: [], edges: [] };
    var GB = fb ? build(fb) : { nodes: [], edges: [] };
    drawGraph(merge(GA, GB, sig), { hint: lens.hint, kind: state.lens, truncated: GA.truncated || GB.truncated });
  }

  /**
   * Union the before and after graphs and tag every node/edge with its status,
   * then lay the union out once. Laying out each side separately would move
   * every node whenever one is inserted, which buries the real change in
   * layout noise.
   */
  function merge(ga, gb, sig) {
    var A = new Map(ga.nodes.map(function (n) { return [n.id, n]; }));
    var B = new Map(gb.nodes.map(function (n) { return [n.id, n]; }));
    var nodes = [];
    var ids = new Set([].concat(ga.nodes.map(function (n) { return n.id; }), gb.nodes.map(function (n) { return n.id; })));
    ids.forEach(function (id) {
      var a = A.get(id);
      var b = B.get(id);
      var status = !a ? 'add' : !b ? 'del' : (sig(a) !== sig(b) ? 'chg' : 'same');
      var node = Object.assign({}, b || a, { status: status, before: a || null, after: b || null });
      nodes.push(node);
    });

    var ekey = function (e) { return JSON.stringify([e.from, e.to]); };
    var EA = new Set(ga.edges.map(ekey));
    var EB = new Set(gb.edges.map(ekey));
    var edges = [];
    var seen = new Set();
    [].concat(ga.edges, gb.edges).forEach(function (e) {
      var k = ekey(e);
      if (seen.has(k)) return;
      seen.add(k);
      edges.push(Object.assign({}, e, { status: !EA.has(k) ? 'add' : !EB.has(k) ? 'del' : 'same' }));
    });

    return { nodes: nodes, edges: edges, counts: countStatus(nodes, edges) };
  }

  function countStatus(nodes, edges) {
    var c = { add: 0, del: 0, chg: 0, same: 0, edgeAdd: 0, edgeDel: 0 };
    nodes.forEach(function (n) { c[n.status]++; });
    edges.forEach(function (e) { if (e.status === 'add') c.edgeAdd++; else if (e.status === 'del') c.edgeDel++; });
    return c;
  }

  // ── layered DAG layout ────────────────────────────────────────────────
  function layout(nodes, edges) {
    var byId = new Map(nodes.map(function (n) { return [n.id, n]; }));
    var out = new Map();
    var indeg = new Map();
    nodes.forEach(function (n) { out.set(n.id, []); indeg.set(n.id, 0); });
    edges.forEach(function (e) {
      if (!byId.has(e.from) || !byId.has(e.to) || e.from === e.to) return;
      out.get(e.from).push(e.to);
      indeg.set(e.to, indeg.get(e.to) + 1);
    });

    // Longest-path ranking over a topological order; nodes left over by a cycle
    // keep rank 0 rather than blocking the layout.
    var rank = new Map();
    nodes.forEach(function (n) { rank.set(n.id, 0); });
    var queue = nodes.filter(function (n) { return indeg.get(n.id) === 0; }).map(function (n) { return n.id; });
    var deg = new Map(indeg);
    var seen = 0;
    while (queue.length) {
      var id = queue.shift();
      seen++;
      out.get(id).forEach(function (to) {
        rank.set(to, Math.max(rank.get(to), rank.get(id) + 1));
        deg.set(to, deg.get(to) - 1);
        if (deg.get(to) === 0) queue.push(to);
      });
    }

    var layers = [];
    nodes.forEach(function (n) {
      var r = rank.get(n.id);
      (layers[r] = layers[r] || []).push(n);
    });

    // Barycenter ordering, a few sweeps, to cut edge crossings.
    var pos = new Map();
    layers.forEach(function (layer) { layer.forEach(function (n, i) { pos.set(n.id, i); }); });
    var preds = new Map(nodes.map(function (n) { return [n.id, []]; }));
    edges.forEach(function (e) { if (preds.has(e.to) && byId.has(e.from)) preds.get(e.to).push(e.from); });
    for (var sweep = 0; sweep < 4; sweep++) {
      for (var r2 = 1; r2 < layers.length; r2++) {
        layers[r2].sort(function (a, b) { return bary(a) - bary(b); });
        layers[r2].forEach(function (n, i) { pos.set(n.id, i); });
      }
    }
    function bary(n) {
      var ps = preds.get(n.id).filter(function (p) { return pos.has(p); });
      if (!ps.length) return pos.get(n.id);
      return ps.reduce(function (s, p) { return s + pos.get(p); }, 0) / ps.length;
    }

    var NW = 176;
    var NH = 40;
    var GX = 74;
    var GY = 16;
    var maxRows = Math.max.apply(null, layers.map(function (l) { return l.length; }).concat([1]));
    layers.forEach(function (layer, r) {
      var h = layer.length * (NH + GY) - GY;
      var top = (maxRows * (NH + GY) - GY - h) / 2;
      layer.forEach(function (n, i) {
        n.x = 24 + r * (NW + GX);
        n.y = 24 + top + i * (NH + GY);
        n.w = NW;
        n.h = NH;
      });
    });

    return {
      width: 48 + layers.length * (NW + GX),
      height: 48 + maxRows * (NH + GY),
      cyclic: seen < nodes.length,
    };
  }

  var STATUS_LABEL = { add: '新增', del: '删除', chg: '属性改变', same: '未变' };

  function drawGraph(g, opts) {
    if (!g.nodes.length) {
      $('graphBody').innerHTML = '<p class="ptx-empty">这个视角下没有可显示的节点。</p>';
      return;
    }
    var box = layout(g.nodes, g.edges);
    var byId = new Map(g.nodes.map(function (n) { return [n.id, n]; }));

    var edgeSvg = g.edges.map(function (e) {
      var a = byId.get(e.from);
      var b = byId.get(e.to);
      if (!a || !b) return '';
      var x1 = a.x + a.w;
      var y1 = a.y + a.h / 2;
      var x2 = b.x;
      var y2 = b.y + b.h / 2;
      var mx = (x1 + x2) / 2;
      return '<path class="ptx-edge ptx-edge--' + e.status + '" d="M' + x1 + ',' + y1
        + ' C' + mx + ',' + y1 + ' ' + mx + ',' + y2 + ' ' + x2 + ',' + y2 + '"/>';
    }).join('');

    var nodeSvg = g.nodes.map(function (n) {
      var sub = nodeSubtitle(n, opts.kind);
      return '<g class="ptx-node ptx-node--' + n.status + '" transform="translate(' + n.x + ',' + n.y + ')"'
        + ' data-line="' + (n.line || '') + '" data-id="' + esc(n.id) + '" tabindex="0">'
        + '<rect width="' + n.w + '" height="' + n.h + '" rx="7"/>'
        + '<text class="ptx-node__label" x="10" y="17">' + esc(trunc(n.label, 22)) + '</text>'
        + '<text class="ptx-node__sub" x="10" y="31">' + esc(trunc(sub, 26)) + '</text>'
        + '<title>' + esc(n.label + '\n' + sub + '\n' + STATUS_LABEL[n.status]
          + (n.line ? '\n源行 ' + n.line : '')) + '</title>'
        + '</g>';
    }).join('');

    var c = g.counts;
    var summary = '<div class="ptx-graphsummary">' + esc(opts.hint)
      + '<span class="ptx-graphstats">节点 ' + g.nodes.length
      + ' · <b class="ptx-add">+' + c.add + '</b> <b class="ptx-del">−' + c.del + '</b>'
      + ' · 属性改变 ' + c.chg + ' · 未变 ' + c.same
      + ' · 边 <b class="ptx-add">+' + c.edgeAdd + '</b> <b class="ptx-del">−' + c.edgeDel + '</b>'
      + (box.cyclic ? ' · 图中存在环' : '')
      + (opts.truncated ? ' · <b class="ptx-warn">节点过多，已截断</b>' : '')
      + '</span></div>';

    $('graphBody').innerHTML = summary
      + '<div class="ptx-canvas" id="canvas"><svg width="' + box.width + '" height="' + box.height + '">'
      + '<g id="viewport">' + edgeSvg + nodeSvg + '</g></svg></div>';

    enablePanZoom($('canvas'));
  }

  /** How much a function has to show under a given lens. */
  /** Would this lens draw anything for this function, before or after? */
  function lensHasContent(lens, fa, fb) {
    function any(pick) {
      return (fa && pick(fa)) || (fb && pick(fb));
    }
    if (lens === 'spacetime') return !!any(function (f) { return f.buffers.length; });
    if (lens === 'memory') return !!any(function (f) { return f.allocs.length + f.buffers.length; });
    if (lens === 'task') return !!any(function (f) { return f.tasks.length; });
    if (lens === 'control') return !!any(function (f) { return f.loops.length; });
    return true;
  }

  function lensWeight(f, lens) {
    if (lens === 'task') return f.tasks.length;
    if (lens === 'memory' || lens === 'spacetime') return f.allocs.length + f.buffers.length;
    if (lens === 'control') return f.loops.length;
    if (lens === 'opshift') return Object.keys(f.opHist).length;
    return f.stmtCount;
  }
  function lensUnit(lens) {
    return {
      task: ' 任务', memory: ' 缓冲', spacetime: ' 缓冲',
      control: ' 循环', opshift: ' 种算子',
    }[lens] || ' 语句';
  }

  function nodeSubtitle(n, kind) {
    if (kind === 'call') return [n.kind, n.level, n.stmts != null ? n.stmts + ' 语句' : ''].filter(Boolean).join(' · ');
    if (kind === 'control') return n.detail || (n.weight != null ? n.weight + ' 语句' : '');
    if (kind === 'dataflow') {
      return [n.op, n.shape ? '[' + n.shape.join('×') + ']' : '', n.dtype, n.space].filter(Boolean).join(' ');
    }
    return [n.type, n.level, n.weight ? n.weight + ' 语句' : ''].filter(Boolean).join(' · ');
  }

  function enablePanZoom(host) {
    var svg = host.querySelector('svg');
    var vp = host.querySelector('#viewport');
    var scale = 1;
    var tx = 0;
    var ty = 0;
    var dragging = false;
    var sx = 0;
    var sy = 0;

    function apply() { vp.setAttribute('transform', 'translate(' + tx + ',' + ty + ') scale(' + scale + ')'); }

    host.addEventListener('wheel', function (e) {
      e.preventDefault();
      var rect = svg.getBoundingClientRect();
      var mx = e.clientX - rect.left;
      var my = e.clientY - rect.top;
      var k = e.deltaY < 0 ? 1.12 : 1 / 1.12;
      var next = Math.min(2.6, Math.max(0.18, scale * k));
      tx = mx - (mx - tx) * (next / scale);
      ty = my - (my - ty) * (next / scale);
      scale = next;
      apply();
    }, { passive: false });

    host.addEventListener('pointerdown', function (e) {
      if (e.target.closest('.ptx-node')) return;
      dragging = true;
      sx = e.clientX - tx;
      sy = e.clientY - ty;
      host.setPointerCapture(e.pointerId);
      host.classList.add('is-dragging');
    });
    host.addEventListener('pointermove', function (e) {
      if (!dragging) return;
      tx = e.clientX - sx;
      ty = e.clientY - sy;
      apply();
    });
    host.addEventListener('pointerup', function () { dragging = false; host.classList.remove('is-dragging'); });

    host.addEventListener('click', function (e) {
      var node = e.target.closest('.ptx-node');
      if (!node) return;
      // In the call lens a node *is* a function, so clicking one opens that
      // function's diff; in the others it maps to a line inside the function
      // already on screen.
      if (state.lens === 'call') {
        var fnName = node.getAttribute('data-id');
        var changed = (pass().changedFunctions || []).some(function (f) { return f.name === fnName; });
        if (!changed) { toast(fnName + ' 在本 Pass 中没有变化'); return; }
        state.fn = fnName;
      }
      state.jumpLine = Number(node.getAttribute('data-line')) || null;
      state.tab = 'diff';
      render();
    });
  }

  // ── memory lens ───────────────────────────────────────────────────────
  // ══════════════════════════════════════════════════════════════════════
  // Pass-specific views
  //
  // A generic lens can only say "the IR changed". What a reader wants is
  // what THIS KIND of Pass does: outlining lifts a region into a function
  // and has to decide what crosses the boundary; lowering makes the data
  // movement that tensor semantics kept implicit into explicit load/store.
  // Those are different pictures, so each Pass kind draws its own.
  // ══════════════════════════════════════════════════════════════════════

  var MOVE_OPS = ["load", "store", "gather_row", "scatter_row", "copy", "assemble",
    "write", "read", "create", "create_l1", "full", "slice", "extract", "move"];
  var VIEW_OPS = ["transpose_view", "reshape", "view", "partition_view", "broadcast_view"];

  function opCategory(op) {
    var t = opTail(op) || String(op).replace(/^pl\./, "");
    if (t === "get_block_idx" || t === "get_block_num") return "meta";
    if (VIEW_OPS.indexOf(t) >= 0) return "view";
    if (MOVE_OPS.indexOf(t) >= 0) return "move";
    return "compute";
  }

  var CAT_LABEL = { move: "搬运", compute: "计算", view: "视图", meta: "" };

  /** Ordered op list from a function body — opHist only carries counts. */
  function opSequence(f) {
    var out = [];
    (f && f.src ? f.src : []).forEach(function (line) {
      var m = line.match(/pl\.(?:tensor|tile)\.[a-z_0-9]+\(/g);
      if (!m) return;
      m.forEach(function (x) { out.push(x.slice(0, -1)); });
    });
    return out;
  }

  /** Collapse the op stream into runs of one category — the pipeline stages. */
  function segments(ops) {
    var segs = [];
    ops.forEach(function (op) {
      var cat = opCategory(op);
      if (cat === "meta") return;
      var last = segs[segs.length - 1];
      if (last && last.cat === cat) { last.ops.push(op); return; }
      segs.push({ cat: cat, ops: [op] });
    });
    return segs.map(function (sg) {
      var counts = {};
      sg.ops.forEach(function (o) { var t = opTail(o) || o; counts[t] = (counts[t] || 0) + 1; });
      var top = Object.keys(counts).sort(function (a, b) { return counts[b] - counts[a]; });
      sg.label = top[0] + (counts[top[0]] > 1 ? " ×" + counts[top[0]] : "");
      sg.extra = top.length - 1;
      sg.n = sg.ops.length;
      sg.kinds = top;
      return sg;
    });
  }

  // ── outline: a region becomes a function ──────────────────────────────

  /** Names returned by the trailing return statement of a function body. */
  function returnNames(f) {
    var src = (f && f.src) || [];
    for (var i = src.length - 1; i >= 0; i--) {
      var m = src[i].match(/^\s*return\s+(.+?)\s*$/);
      if (m) return m[1].split(",").map(function (x) { return x.trim(); }).filter(Boolean);
    }
    return [];
  }

  function shortName(n) {
    return String(n).replace(/_inline\d+/, "").replace(/__(ssa|rv|phi|iter)_v\d+$/, "");
  }

  function drawOutline(before, after, fnName, passName) {
    var f = after.byName.get(fnName);
    if (!f) {
      $("graphBody").innerHTML = "<p class=\"ptx-empty\">本 Pass 之后没有这个函数。</p>";
      return;
    }
    var params = f.params || [];
    var rets = returnNames(f);
    var pnames = {};
    params.forEach(function (p) { pnames[p.name] = 1; });
    var carried = rets.filter(function (r) { return pnames[r]; });

    var hosts = after.functions.filter(function (x) {
      return (x.calls || []).some(function (c) { return c.callee === fnName; });
    });
    var host = hosts[0];
    var callLine = host && (host.calls.find(function (c) { return c.callee === fnName; }) || {}).line;
    var existedBefore = !!(before && before.byName.get(fnName));

    var W = 640, H = 300;
    var svg = "<svg class=\"ptx-passview\" viewBox=\"0 0 " + W + " " + H + "\" role=\"img\" "
      + "aria-label=\"" + esc(fnName) + " 被外提为独立函数\">"
      + "<defs><marker id=\"pvArrow\" viewBox=\"0 0 10 10\" refX=\"8\" refY=\"5\" markerWidth=\"6\" "
      + "markerHeight=\"6\" orient=\"auto-start-reverse\"><path d=\"M2 1L8 5L2 9\" fill=\"none\" "
      + "stroke=\"context-stroke\" stroke-width=\"1.5\" stroke-linecap=\"round\"/></marker></defs>";

    // before
    svg += "<text class=\"pv-h\" x=\"12\" y=\"16\">之前 · 宿主函数体内的一段区域</text>";
    svg += "<rect class=\"pv-host\" x=\"12\" y=\"24\" width=\"" + (W - 24) + "\" height=\"78\" rx=\"6\"/>";
    svg += "<text class=\"pv-t\" x=\"24\" y=\"43\">" + esc(host ? shortName(host.name) : "宿主函数") + "</text>";
    svg += "<rect class=\"pv-region\" x=\"26\" y=\"52\" width=\"" + (W - 52) + "\" height=\"40\" rx=\"5\"/>";
    svg += "<text class=\"pv-t\" x=\"38\" y=\"70\">" + esc(passName.indexOf("Incore") >= 0 ? "InCore 计算区域" : "作用域区域")
      + "</text>";
    svg += "<text class=\"pv-s\" x=\"38\" y=\"85\">" + f.stmtCount + " 条语句 · "
      + (f.loops || []).length + " 个循环</text>";

    svg += "<path d=\"M" + (W / 2) + " 104 V126\" fill=\"none\" stroke=\"var(--foreground-muted)\" "
      + "stroke-width=\"1\" marker-end=\"url(#pvArrow)\"/>";
    svg += "<rect class=\"pv-op\" x=\"" + (W / 2 - 84) + "\" y=\"128\" width=\"168\" height=\"22\" rx=\"4\"/>";
    svg += "<text class=\"pv-t pv-mid\" x=\"" + (W / 2) + "\" y=\"143\" text-anchor=\"middle\">"
      + esc(passName) + "</text>";

    // after
    svg += "<text class=\"pv-h\" x=\"12\" y=\"172\">之后 · 独立函数 + 宿主处一次调用</text>";
    svg += "<rect class=\"pv-host\" x=\"12\" y=\"180\" width=\"250\" height=\"56\" rx=\"6\"/>";
    svg += "<text class=\"pv-t\" x=\"24\" y=\"199\">" + esc(host ? shortName(host.name) : "宿主") + "</text>";
    svg += "<text class=\"pv-s\" x=\"24\" y=\"215\">" + esc(shortName(fnName)) + "(" + params.length
      + " 个实参)" + (callLine ? " · 第 " + callLine + " 行" : "") + "</text>";
    svg += "<path d=\"M266 208 H292\" fill=\"none\" stroke=\"var(--foreground-muted)\" stroke-width=\"1\" "
      + "marker-end=\"url(#pvArrow)\"/>";
    svg += "<rect class=\"pv-fn\" x=\"298\" y=\"176\" width=\"" + (W - 310) + "\" height=\"64\" rx=\"6\"/>";
    svg += "<text class=\"pv-t\" x=\"310\" y=\"196\">def " + esc(shortName(fnName)) + "(…)</text>";
    svg += "<text class=\"pv-s\" x=\"310\" y=\"212\">" + params.length + " 个参数 → " + rets.length
      + " 个返回值</text>";
    svg += "<text class=\"pv-s\" x=\"310\" y=\"228\">" + esc(f.kind || "") + " · " + f.srcLineCount + " 行</text>";

    // boundary detail
    var y = 256;
    svg += "<rect class=\"pv-note\" x=\"12\" y=\"" + y + "\" width=\"" + (W - 24) + "\" height=\"34\" rx=\"5\"/>";
    svg += "<text class=\"pv-s\" x=\"24\" y=\"" + (y + 21) + "\">"
      + (carried.length
        ? esc(carried.length + " 个值既是参数又是返回值：" + carried.map(shortName).join("、")
            + " —— 跨循环携带的累加状态")
        : "参数即 scope 内引用的外部变量；返回值即 scope 外仍要使用的结果")
      + "</text>";

    svg += "</svg>";

    var head = "<div class=\"ptx-graphsummary\">把一段核内区域抬成独立函数，编译器必须定下边界："
      + "什么传进去、什么传出来。<span class=\"ptx-graphstats\">"
      + params.length + " 参数 · " + rets.length + " 返回"
      + (carried.length ? " · <b>" + carried.length + "</b> 个循环携带" : "")
      + "</span></div>";

    var tbl = "";
    if (params.length) {
      tbl = "<table class=\"ptx-table ptx-table--fns\"><thead><tr><th>跨边界的值</th><th>类型</th>"
        + "<th>方向</th></tr></thead><tbody>";
      params.slice(0, 40).forEach(function (p) {
        var isCarried = rets.indexOf(p.name) >= 0;
        tbl += "<tr><td><code>" + esc(shortName(p.name)) + "</code></td>"
          + "<td><span class=\"ptx-muted\">" + esc(p.ctor + (p.shape ? "[" + p.shape.join("×") + "]" : "")
          + (p.dtype ? " " + p.dtype : "")) + "</span></td>"
          + "<td>" + (isCarried
            ? "<span class=\"ptx-status ptx-status--changed\">进 + 出</span>"
            : "<span class=\"ptx-status ptx-status--same\">只进</span>") + "</td></tr>";
      });
      tbl += "</tbody></table>";
    }

    $("graphBody").innerHTML = head + "<div class=\"ptx-passview__wrap\">" + svg + "</div>"
      + (existedBefore ? "" : "") + tbl;
  }

  // ── lowering: implicit movement becomes explicit ──────────────────────

  /** `pl.tensor.exp` and `pl.tile.exp` share the key `exp`, so the diff can
   *  align them as one operator that changed domain rather than as an
   *  unrelated delete plus insert. */
  function normOp(op) { return opTail(op) || String(op).replace(/^pl\./, ""); }

  /**
   * Sequence alignment of the two operator streams. Lowering is not a set
   * difference — it is the same computation re-expressed, with movement that
   * tensor semantics implied now spelled out. Aligning the streams shows both
   * at once: which operators merely changed domain, and where new load/store
   * had to be inserted between them.
   */
  function drawLowering(fa, fb, passName) {
    var opsA = opSequence(fa);
    var opsB = opSequence(fb);
    if (!opsA.length && !opsB.length) {
      $("graphBody").innerHTML = "<p class=\"ptx-empty\">这一步没有可比对的算子序列。</p>";
      return;
    }
    var rows = LIB.diffLines(opsA.map(normOp), opsB.map(normOp));

    var W = 640, TOP = 26, BAR = 22, GAP = 14;
    var H = TOP + BAR + GAP + BAR + 18;
    var colW = W / Math.max(1, rows.length);

    var moved = 0, born = 0, gone = 0;
    rows.forEach(function (r) {
      if (r.tag === "+") born++;
      else if (r.tag === "-") gone++;
      else if (opDomain(opsA[r.a]) !== opDomain(opsB[r.b])) moved++;
    });

    // Clusters of consecutive inserts / deletes, so the picture has anchors.
    var clusters = [];
    var cur = null;
    rows.forEach(function (r, i) {
      if (r.tag === "=") { cur = null; return; }
      var op = r.tag === "+" ? opsB[r.b] : opsA[r.a];
      var key = r.tag + normOp(op);
      if (cur && cur.key === key && i === cur.end + 1) { cur.n++; cur.end = i; return; }
      cur = { key: key, tag: r.tag, op: normOp(op), n: 1, start: i, end: i };
      clusters.push(cur);
    });
    var topClusters = clusters.slice().sort(function (a, b) { return b.n - a.n; }).slice(0, 5);

    var svg = "<svg class=\"ptx-passview ptx-align\" viewBox=\"0 0 " + W + " " + H + "\" role=\"img\" "
      + "aria-label=\"tensor 域与 tile 域的算子序列比对\">";

    topClusters.forEach(function (c) {
      var cx = (c.start + (c.end - c.start) / 2 + 0.5) * colW;
      var label = (c.tag === "+" ? "+" : "−") + c.op + (c.n > 1 ? " ×" + c.n : "");
      var wd = label.length * 6.2 + 8;
      var lx = Math.max(0, Math.min(W - wd, cx - wd / 2));
      svg += "<rect class=\"pv-cl " + (c.tag === "+" ? "is-born" : "is-gone") + "\" x=\"" + lx
        + "\" y=\"2\" width=\"" + wd + "\" height=\"16\" rx=\"3\"/>"
        + "<text class=\"pv-s pv-cltext\" x=\"" + (lx + 4) + "\" y=\"14\">" + esc(label) + "</text>"
        + "<line class=\"pv-clline\" x1=\"" + cx + "\" y1=\"18\" x2=\"" + cx + "\" y2=\"" + TOP + "\"/>";
    });

    rows.forEach(function (r, i) {
      var x = i * colW;
      var wd = Math.max(1.5, colW - 0.6);
      var same = r.tag === "=";
      var dom = same && opDomain(opsA[r.a]) !== opDomain(opsB[r.b]);

      if (same || r.tag === "-") {
        var oa = opsA[r.a];
        svg += "<g class=\"pv-cell\"><title>" + esc(oa) + "</title>"
          + "<rect class=\"pv-op-cell " + (r.tag === "-" ? "is-gone" : dom ? "is-moved" : "is-keep")
          + " cat-" + opCategory(oa) + "\" x=\"" + x + "\" y=\"" + TOP + "\" width=\"" + wd
          + "\" height=\"" + BAR + "\" rx=\"1.5\"/></g>";
      }
      if (same || r.tag === "+") {
        var ob = opsB[r.b];
        svg += "<g class=\"pv-cell\"><title>" + esc(ob) + "</title>"
          + "<rect class=\"pv-op-cell " + (r.tag === "+" ? "is-born" : dom ? "is-moved" : "is-keep")
          + " cat-" + opCategory(ob) + "\" x=\"" + x + "\" y=\"" + (TOP + BAR + GAP) + "\" width=\"" + wd
          + "\" height=\"" + BAR + "\" rx=\"1.5\"/></g>";
      }
      if (same && dom) {
        svg += "<line class=\"pv-link\" x1=\"" + (x + wd / 2) + "\" y1=\"" + (TOP + BAR)
          + "\" x2=\"" + (x + wd / 2) + "\" y2=\"" + (TOP + BAR + GAP) + "\"/>";
      }
    });

    svg += "<text class=\"pv-s\" x=\"0\" y=\"" + (TOP - 4) + "\">tensor 域 · " + opsA.length + " 个算子</text>";
    svg += "<text class=\"pv-s\" x=\"0\" y=\"" + (H - 4) + "\">tile 域 · " + opsB.length + " 个算子</text>";
    svg += "</svg>";

    var head = "<div class=\"ptx-graphsummary\">同一段计算换一种语义表达。上下对齐的是同一个算子，"
      + "只是换了域；断口处是被新插入的显式搬运。<span class=\"ptx-graphstats\">"
      + "<b>" + moved + "</b> 个换域 · <b class=\"ptx-add\">+" + born + "</b> 新增 · "
      + "<b class=\"ptx-del\">−" + gone + "</b> 退场</span></div>";

    var legend = "<div class=\"ptx-lifeline__legend\">"
      + "<span class=\"ptx-dot pv-d-moved\"></span>换域（tensor→tile）"
      + "<span class=\"ptx-dot pv-d-born\"></span>新增"
      + "<span class=\"ptx-dot pv-d-gone\"></span>退场"
      + "<span class=\"ptx-dot pv-d-keep\"></span>原样保留"
      + "<span class=\"ptx-lifeline__legendsep\"></span><span>深浅 = 搬运 / 计算 / 视图</span></div>";

    var hb = (fa && fa.opHist) || {};
    var ha = (fb && fb.opHist) || {};
    // "New" means the operator did not exist under either domain before — a
    // keyed-by-domain test would call `tile.gather_row` new when only
    // `tensor.gather_row` existed, which is a domain change, not new movement.
    var beforeNorm = {};
    Object.keys(hb).forEach(function (k) {
      beforeNorm[normOp(k)] = (beforeNorm[normOp(k)] || 0) + hb[k];
    });
    var newMove = [];
    Object.keys(ha).forEach(function (k) {
      var c = opCategory(k);
      if (c !== "move" && c !== "view") return;
      if (!beforeNorm[normOp(k)]) newMove.push({ op: normOp(k), n: ha[k] });
    });
    newMove.sort(function (a, b) { return b.n - a.n; });

    var note = newMove.length
      ? "<p class=\"ptx-life__note\">tensor 域把数据搬运藏在语义里，tile 域必须写明——"
        + "新出现的显式搬运："
        + newMove.slice(0, 6).map(function (m) {
            return "<code>" + esc(m.op) + "</code>×" + m.n;
          }).join("、") + "</p>"
      : "";

    $("graphBody").innerHTML = head + "<div class=\"ptx-passview__wrap\">" + svg + legend
      + "</div>" + note;
  }
  var PASS_VIEWS = {
    OutlineIncoreScopes: { label: "外提", draw: "outline" },
    OutlineHierarchyScopes: { label: "外提", draw: "outline" },
    OutlineClusterScopes: { label: "外提", draw: "outline" },
    OutlineGraphScopes: { label: "外提", draw: "outline" },
    ConvertTensorToTileOps: { label: "语义下降", draw: "lowering" },
  };

  // ══════════════════════════════════════════════════════════════════════
  // Change views
  //
  // The five structural lenses answer "what does this IR look like". Two of
  // them answer it program-wide, which in callable mode means every callable
  // renders the identical picture. These two are differential and function-
  // scoped instead: they show what THIS Pass did to THIS callable.
  // ══════════════════════════════════════════════════════════════════════

  /** `pl.tensor.gather_row` -> `gather_row`; the part that survives lowering. */
  function opTail(op) {
    var m = /^pl\.(?:tensor|tile)\.(.+)$/.exec(op);
    return m ? m[1] : null;
  }
  function opDomain(op) {
    if (/^pl\.tensor\./.test(op)) return "tensor";
    if (/^pl\.tile\./.test(op)) return "tile";
    return "other";
  }

  /**
   * Op migration. A lowering pass does not "add 6 lines" — it retires
   * `pl.tensor.exp` and lights up `pl.tile.exp`. Pairing the two by their
   * shared tail turns a histogram delta into the actual sentence: this
   * operator moved from the tensor domain to the tile domain.
   */
  function drawOpShift(fa, fb, hint) {
    var hb = (fa && fa.opHist) || {};
    var ha = (fb && fb.opHist) || {};
    var names = {};
    Object.keys(hb).forEach(function (k) { names[k] = 1; });
    Object.keys(ha).forEach(function (k) { names[k] = 1; });

    var moved = [];
    Object.keys(names).forEach(function (op) {
      var d = (ha[op] || 0) - (hb[op] || 0);
      if (d) moved.push({ op: op, before: hb[op] || 0, after: ha[op] || 0, d: d });
    });

    if (!moved.length) {
      $("graphBody").innerHTML = "<div class=\"ptx-graphsummary\">" + esc(hint) + "</div>"
        + "<p class=\"ptx-empty\">算子构成没有变化。这一步改写的是语句内部（地址、名字、顺序），"
        + "右边的 Diff 是唯一的事实来源。</p>";
      return;
    }

    var gone = moved.filter(function (m) { return m.d < 0; }).sort(function (x, y) { return x.d - y.d; });
    var born = moved.filter(function (m) { return m.d > 0; }).sort(function (x, y) { return y.d - x.d; });

    var used = {};
    var pairs = [];
    gone.forEach(function (g) {
      var tail = opTail(g.op);
      if (!tail) return;
      var hit = born.find(function (b) {
        return !used[b.op] && opTail(b.op) === tail && opDomain(b.op) !== opDomain(g.op);
      });
      if (hit) { used[hit.op] = 1; used[g.op] = 1; pairs.push({ from: g, to: hit }); }
    });

    var restGone = gone.filter(function (g) { return !used[g.op]; });
    var restBorn = born.filter(function (b) { return !used[b.op]; });

    var domains = {};
    pairs.forEach(function (p) { domains[opDomain(p.from.op) + "→" + opDomain(p.to.op)] = 1; });
    var domainLine = Object.keys(domains).map(function (k) {
      return k.replace("tensor", "Tensor 域").replace("tile", "Tile 域");
    }).join("、");

    var html = "<div class=\"ptx-graphsummary\">" + esc(hint)
      + "<span class=\"ptx-graphstats\">"
      + (pairs.length ? "<b>" + pairs.length + "</b> 个算子迁移" : "")
      + (restBorn.length ? (pairs.length ? " · " : "") + "<b class=\"ptx-add\">+" + restBorn.length + "</b> 种新算子" : "")
      + (restGone.length ? " · <b class=\"ptx-del\">−" + restGone.length + "</b> 种退场" : "")
      + "</span></div>";

    html += "<div class=\"ptx-shift\">";

    if (pairs.length) {
      html += "<div class=\"ptx-shift__head\">算子迁移"
        + (domainLine ? "<small>" + esc(domainLine) + "</small>" : "") + "</div>";
      pairs.forEach(function (p) {
        html += "<div class=\"ptx-shift__row\">"
          + "<span class=\"ptx-shift__from\"><code>" + esc(p.from.op) + "</code><b>" + p.from.before + "</b></span>"
          + "<span class=\"ptx-shift__arrow\" aria-hidden=\"true\">⟶</span>"
          + "<span class=\"ptx-shift__to\"><code>" + esc(p.to.op) + "</code><b>" + p.to.after + "</b></span>"
          + "</div>";
      });
    }

    if (restBorn.length || restGone.length) {
      html += "<div class=\"ptx-shift__head\">未配对</div><div class=\"ptx-shift__cols\">";
      html += "<div class=\"ptx-shift__col\"><h5 class=\"ptx-add\">新增 / 变多</h5>";
      html += restBorn.length ? restBorn.map(function (b) {
        return "<div class=\"ptx-shift__item\"><code>" + esc(b.op) + "</code>"
          + "<span>" + b.before + " → <b>" + b.after + "</b></span></div>";
      }).join("") : "<p class=\"ptx-empty\">—</p>";
      html += "</div><div class=\"ptx-shift__col\"><h5 class=\"ptx-del\">移除 / 变少</h5>";
      html += restGone.length ? restGone.map(function (g) {
        return "<div class=\"ptx-shift__item\"><code>" + esc(g.op) + "</code>"
          + "<span>" + g.before + " → <b>" + g.after + "</b></span></div>";
      }).join("") : "<p class=\"ptx-empty\">—</p>";
      html += "</div></div>";
    }

    $("graphBody").innerHTML = html + "</div>";
  }

  /**
   * Buffer lifetimes. Reuse is not "fewer buffers" — it is one buffer kept
   * alive across a longer span to serve several uses. Drawing each buffer as a
   * bar over statement index makes that literal: many short bars collapse into
   * a few long ones, and the reused bars are the ones that got longer.
   */
  function drawSpacetime(fa, fb, hint) {
    function lanes(f) {
      if (!f || !f.buffers) return [];
      return f.buffers
        .filter(function (b) { return typeof b.first === "number" && typeof b.last === "number"; })
        .map(function (b) {
          return {
            name: b.name, space: b.space || "unspecified", size: b.size || 0,
            first: b.first, last: b.last, uses: b.uses || 0,
          };
        })
        .sort(function (x, y) { return x.first - y.first || y.size - x.size; });
    }

    var A = lanes(fa);
    var B = lanes(fb);
    if (!A.length && !B.length) {
      $("graphBody").innerHTML = "<div class=\"ptx-graphsummary\">" + esc(hint) + "</div>"
        + "<p class=\"ptx-empty\">该函数在此阶段还没有缓冲区生命周期信息。"
        + "缓冲要到 <code>InitMemRef</code> 之后才存在。</p>";
      return;
    }

    var maxStmt = 1;
    [].concat(A, B).forEach(function (l) { maxStmt = Math.max(maxStmt, l.last); });

    var byName = {};
    A.forEach(function (l) { byName[l.name] = { before: l }; });
    B.forEach(function (l) { (byName[l.name] = byName[l.name] || {}).after = l; });

    function panel(list, label, other) {
      var W = 560, rowH = 13, top = 16;
      var H = top + Math.max(1, list.length) * rowH + 8;
      var total = list.reduce(function (s, x) { return s + x.size; }, 0);
      var out = "<div class=\"ptx-life__panel\"><h5>" + esc(label)
        + "<span>" + list.length + " 个缓冲 · " + bytes(total) + "</span></h5>"
        + "<svg class=\"ptx-life\" viewBox=\"0 0 " + W + " " + H + "\" role=\"img\" "
        + "aria-label=\"" + esc(label) + "缓冲区生命周期\">";

      [0, 0.5, 1].forEach(function (f) {
        var x = 60 + (W - 76) * f;
        out += "<line class=\"ptx-life__grid\" x1=\"" + x + "\" y1=\"" + top + "\" x2=\"" + x + "\" y2=\"" + (H - 8) + "\"/>"
          + "<text class=\"ptx-life__tick\" x=\"" + x + "\" y=\"" + (top - 5) + "\" text-anchor=\"middle\">"
          + Math.round(maxStmt * f) + "</text>";
      });

      list.forEach(function (l, i) {
        var y = top + i * rowH;
        var x0 = 60 + (W - 76) * (l.first / maxStmt);
        var x1 = 60 + (W - 76) * (l.last / maxStmt);
        var w = Math.max(2, x1 - x0);
        var twin = byName[l.name] && byName[l.name][other];
        var grew = twin && (l.last - l.first) > (twin.last - twin.first);
        var cls = "ptx-life__bar" + (!twin ? " is-solo" : grew ? " is-grew" : "");
        out += "<g class=\"ptx-life__row\">"
          + "<title>" + esc(l.name + " · " + l.space + " · " + bytes(l.size)
            + " · 语句 " + l.first + "–" + l.last + " · " + l.uses + " 次使用") + "</title>"
          + "<text class=\"ptx-life__name\" x=\"56\" y=\"" + (y + 9) + "\" text-anchor=\"end\">"
          + esc(l.name.replace(/^mem_/, "")) + "</text>"
          + "<rect class=\"" + cls + "\" x=\"" + x0 + "\" y=\"" + (y + 2) + "\" width=\"" + w
          + "\" height=\"8\" rx=\"2\"/>"
          + "</g>";
      });

      return out + "</svg></div>";
    }

    var tb = A.reduce(function (s, x) { return s + x.size; }, 0);
    var ta = B.reduce(function (s, x) { return s + x.size; }, 0);
    var reused = B.filter(function (l) {
      var t = byName[l.name] && byName[l.name].before;
      return t && (l.last - l.first) > (t.last - t.first);
    });

    var head = "<div class=\"ptx-graphsummary\">" + esc(hint)
      + "<span class=\"ptx-graphstats\">" + A.length + " → <b>" + B.length + "</b> 个缓冲 · "
      + bytes(tb) + " → <b>" + bytes(ta) + "</b>"
      + (ta !== tb ? " <b class=\"" + (ta < tb ? "ptx-add" : "ptx-del") + "\">"
        + (ta < tb ? "−" : "+") + bytes(Math.abs(ta - tb)) + "</b>" : "")
      + "</span></div>";

    var note = reused.length
      ? "<p class=\"ptx-life__note\"><b>" + reused.length + "</b> 个缓冲的存活区间被拉长"
        + "——它们现在跨越更多语句，替原来多个短命缓冲干活。横轴是语句序号。</p>"
      : "<p class=\"ptx-life__note\">横轴是语句序号，每条是一个缓冲的存活区间。</p>";

    $("graphBody").innerHTML = head + note
      + "<div class=\"ptx-life__wrap\">" + panel(A, "之前", "after") + panel(B, "之后", "before") + "</div>";
  }
  function drawMemory(fa, fb, hint) {
    var before = fa ? LIB.memoryView(fa) : [];
    var after = fb ? LIB.memoryView(fb) : [];
    var spaces = [];
    [].concat(before, after).forEach(function (s) { if (spaces.indexOf(s.space) < 0) spaces.push(s.space); });

    if (!spaces.length) {
      $('graphBody').innerHTML = '<p class="ptx-empty">该函数在此阶段还没有具体的内存分配信息。'
        + '内存要到 <code>InitMemRef</code> / <code>AllocateMemoryAddr</code> 之后才会落实。</p>';
      return;
    }

    var totalB = before.reduce(function (s, x) { return s + x.bytes; }, 0);
    var totalA = after.reduce(function (s, x) { return s + x.bytes; }, 0);
    var max = 1;
    spaces.forEach(function (sp) {
      var b = before.find(function (x) { return x.space === sp; });
      var a = after.find(function (x) { return x.space === sp; });
      max = Math.max(max, b ? b.bytes : 0, a ? a.bytes : 0);
    });

    var html = '<div class="ptx-graphsummary">' + esc(hint)
      + '<span class="ptx-graphstats">合计 ' + bytes(totalB) + ' → <b>' + bytes(totalA) + '</b>'
      + (totalA !== totalB ? ' <b class="' + (totalA < totalB ? 'ptx-add' : 'ptx-del') + '">'
        + (totalA < totalB ? '−' : '+') + bytes(Math.abs(totalA - totalB)) + '</b>' : '')
      + '</span></div><div class="ptx-mem">';

    spaces.forEach(function (sp) {
      var b = before.find(function (x) { return x.space === sp; }) || { bytes: 0, items: [] };
      var a = after.find(function (x) { return x.space === sp; }) || { bytes: 0, items: [] };
      var names = new Set();
      b.items.forEach(function (i) { names.add(i.name); });
      a.items.forEach(function (i) { names.add(i.name); });

      html += '<section class="ptx-memspace"><h4>' + esc(sp)
        + '<span>' + bytes(b.bytes) + ' → <b>' + bytes(a.bytes) + '</b>'
        + ' · ' + b.items.length + ' → ' + a.items.length + ' 个缓冲'
        + (a.dynamicCount ? ' · 其中 ' + a.dynamicCount + ' 个为动态 shape，字节数编译期未定' : '')
        + '</span></h4>'
        + '<div class="ptx-membars">'
        + memBar('之前', b, max) + memBar('之后', a, max)
        + '</div><table class="ptx-table ptx-table--mem"><thead><tr><th>缓冲</th><th>之前</th><th>之后</th><th>偏移</th><th></th></tr></thead><tbody>';

      var rows = [];
      names.forEach(function (n) {
        var bi = b.items.find(function (i) { return i.name === n; });
        var ai = a.items.find(function (i) { return i.name === n; });
        rows.push({ name: n, b: bi, a: ai });
      });
      rows.sort(function (x, y) { return ((y.a && y.a.size) || (y.b && y.b.size) || 0) - ((x.a && x.a.size) || (x.b && x.b.size) || 0); });
      rows.slice(0, 40).forEach(function (r) {
        var status = !r.b ? 'add' : !r.a ? 'del' : (r.b.size !== r.a.size ? 'chg' : 'same');
        var cell = function (it) { return !it ? '—' : it.dynamic ? '<i class="ptx-dyn">动态</i>' : bytes(it.size); };
        html += '<tr class="ptx-mem--' + status + '"><td><code>' + esc(r.name) + '</code></td>'
          + '<td>' + cell(r.b) + '</td>'
          + '<td>' + cell(r.a) + '</td>'
          + '<td class="ptx-offsets">' + (r.a && r.a.offsets && r.a.offsets.length
            ? esc(r.a.offsets.slice(0, 6).join(', ')) + (r.a.offsets.length > 6 ? ' …' : '') : '—') + '</td>'
          + '<td><span class="ptx-status ptx-status--' + ({ add: 'added', del: 'removed', chg: 'changed', same: 'same' })[status] + '">'
          + STATUS_LABEL[status] + '</span></td></tr>';
      });
      html += '</tbody></table>';
      if (rows.length > 40) html += '<p class="ptx-more">共 ' + rows.length + ' 个缓冲，仅列出最大的 40 个</p>';
      html += '</section>';
    });

    $('graphBody').innerHTML = html + '</div>';
  }

  function memBar(label, group, max) {
    var segs = group.items.slice(0, 60).map(function (it) {
      var w = (it.size || 0) / max * 100;
      return '<i style="width:' + w.toFixed(3) + '%" title="' + esc(it.name + ' · ' + bytes(it.size)) + '"></i>';
    }).join('');
    return '<div class="ptx-membar"><span>' + label + '</span><div class="ptx-membar__track">' + segs + '</div>'
      + '<b>' + bytes(group.bytes) + '</b></div>';
  }

  // ══════════════════════════════════════════════════════════════════════
  // Pass documentation
  // ══════════════════════════════════════════════════════════════════════

  var docsLoading = false;

  function renderDoc() {
    var pane = $('docPane');
    pane.hidden = !state.docOpen;
    if (!state.docOpen) return;

    var p = pass();
    if (!p.doc) {
      $('docOrigin').textContent = '';
      $('docBody').innerHTML = '<p class="ptx-empty">本仓库的 <code>repo/pto</code> 镜像里没有这个 Pass 的文档。'
        + '左侧的实测证据仍然完全可用——它直接来自两份快照的比对。</p>';
      return;
    }
    if (!window.PTX_DOCS) {
      if (!docsLoading) {
        docsLoading = true;
        var s = document.createElement('script');
        s.src = 'data/docs.js';
        s.onload = function () { docsLoading = false; renderDoc(); };
        s.onerror = function () { docsLoading = false; $('docBody').innerHTML = '<p class="ptx-empty">文档数据加载失败。</p>'; };
        document.head.appendChild(s);
      }
      $('docBody').innerHTML = '<p class="ptx-loading">正在加载 Pass 文档…</p>';
      return;
    }

    var doc = window.PTX_DOCS[p.doc];
    if (!doc) { $('docBody').innerHTML = '<p class="ptx-empty">未找到该 Pass 的文档。</p>'; return; }

    $('docOrigin').innerHTML = window.PTX_STANDALONE
      ? '来自 <span class="is-inert" title="仓库内路径：' + esc(doc.file) + '">' + esc(doc.file.split('/').pop()) + '</span>'
      : '来自 <a href="../../' + esc(doc.file) + '" target="_blank" rel="noreferrer">' + esc(doc.file.split('/').pop()) + '</a>';

    var html = '<h2>' + esc(doc.title) + '</h2>';
    if (doc.tagline) html += '<p class="ptx-doc__tagline">' + mdInline(doc.tagline) + '</p>';
    if (doc.timing) html += '<p class="ptx-doc__timing"><b>使用时机</b>' + mdInline(doc.timing) + '</p>';
    (doc.blocks || []).forEach(function (b, i) {
      html += '<details' + (i === 0 ? ' open' : '') + '><summary>' + esc(b.heading) + '</summary>'
        + md(b.body) + '</details>';
    });
    $('docBody').innerHTML = html;
  }

  // ══════════════════════════════════════════════════════════════════════
  // Callable lineage
  //
  // The index is built per Pass; this transposes it per callable so a single
  // computation can be followed down the whole pipeline. Everything here is a
  // regrouping of `pass.functions` / `pass.changedFunctions` — no new facts.
  // ══════════════════════════════════════════════════════════════════════

  var CALLABLES = Object.create(null);

  /**
   * ExpandMixedKernel turns a mixed InCore function into an AIC kernel, an AIV
   * kernel and a Group shell that coordinates them. Detecting that precisely
   * (kind flips to Group in the same Pass that adds `<name>_aic` / `_aiv`)
   * matters: plain name-prefix matching would also swallow `foo` / `foo_0`,
   * which are distinct callables the compiler merely disambiguated.
   */
  function detectSplits(r) {
    var splits = Object.create(null);
    var childOf = Object.create(null);
    r.passes.forEach(function (p) {
      var added = Object.create(null);
      var toGroup = [];
      (p.changedFunctions || []).forEach(function (cf) {
        if (cf.status === 'added') added[cf.name] = cf;
        if (cf.status === 'changed' && cf.kind === 'Group' && cf.kindBefore && cf.kindBefore !== 'Group') {
          toGroup.push(cf.name);
        }
      });
      toGroup.forEach(function (root) {
        var kids = ['_aic', '_aiv'].map(function (s) { return root + s; })
          .filter(function (n) { return added[n]; });
        if (!kids.length) return;
        splits[root] = { passIdx: p.idx, passName: p.name, kids: kids };
        kids.forEach(function (k) { childOf[k] = root; });
      });
    });
    return { splits: splits, childOf: childOf };
  }

  function buildCallables(r) {
    var lineage = detectSplits(r);
    var rows = Object.create(null);   // name -> {passIdx -> fn record}
    var marks = Object.create(null);  // name -> {passIdx -> changedFunction}
    var order = [];

    r.passes.forEach(function (p) {
      (p.functions || []).forEach(function (f) {
        if (!rows[f.name]) { rows[f.name] = Object.create(null); order.push(f.name); }
        rows[f.name][p.idx] = f;
      });
      (p.changedFunctions || []).forEach(function (cf) {
        if (!marks[cf.name]) marks[cf.name] = Object.create(null);
        marks[cf.name][p.idx] = cf;
      });
    });

    var list = order.filter(function (n) { return !lineage.childOf[n]; }).map(function (name) {
      var kids = (lineage.splits[name] || {}).kids || [];
      var members = [name].concat(kids);

      var timeline = r.passes.map(function (p) {
        var parts = members.map(function (m) {
          var rec = rows[m] && rows[m][p.idx];
          return rec ? {
            name: m, lines: rec.lines, stmts: rec.stmts, kind: rec.kind,
            loops: rec.loops || 0, allocs: rec.allocs || 0, tasks: rec.tasks || 0,
          } : null;
        }).filter(Boolean);
        var touched = members.filter(function (m) { return marks[m] && marks[m][p.idx]; });
        function sum(k) { return parts.reduce(function (a, b) { return a + b[k]; }, 0); }
        return {
          idx: p.idx,
          name: p.name,
          phase: p.phase,
          lens: p.lens,
          parts: parts,
          total: sum('lines'),
          stmts: sum('stmts'),
          loops: sum('loops'),
          allocs: sum('allocs'),
          tasks: sum('tasks'),
          kinds: parts.map(function (x) { return x.kind; }).sort().join('+'),
          marks: touched.map(function (m) { return marks[m][p.idx]; }),
        };
      });

      var alive = timeline.filter(function (t) { return t.parts.length; });
      var birth = alive.length ? alive[0] : null;
      var last = alive.length ? alive[alive.length - 1] : null;
      var peak = null;
      var grow = null;
      var shrink = null;
      timeline.forEach(function (t, i) {
        if (!t.parts.length) return;
        if (!peak || t.total > peak.total) peak = t;
        var prev = i > 0 ? timeline[i - 1] : null;
        if (!prev || !prev.parts.length) return;
        var d = t.total - prev.total;
        if (d > 0 && (!grow || d > grow.delta)) grow = { t: t, delta: d };
        if (d < 0 && (!shrink || d < shrink.delta)) shrink = { t: t, delta: d };
      });

      var finalKinds = last ? last.parts.map(function (p) { return p.kind; }) : [];
      return {
        name: name,
        members: members,
        split: lineage.splits[name] || null,
        timeline: timeline,
        birth: birth,
        last: last,
        peak: peak,
        grow: grow,
        shrink: shrink,
        finalLines: last ? last.total : 0,
        kinds: finalKinds,
        isKernel: finalKinds.some(function (k) { return k === 'AIC' || k === 'AIV'; }),
        touchCount: timeline.filter(function (t) { return t.marks.length; }).length,
      };
    });

    list.sort(function (a, b) { return b.finalLines - a.finalLines || a.name.localeCompare(b.name); });
    return list;
  }

  function callables() {
    if (!CALLABLES[state.runId]) CALLABLES[state.runId] = buildCallables(run());
    return CALLABLES[state.runId];
  }

  function currentCallable() {
    var all = callables();
    return all.find(function (c) { return c.name === state.callable; }) || all[0] || null;
  }

  // ── rail ──────────────────────────────────────────────────────────────

  function renderCallableRail() {
    var all = callables();
    var kernels = all.filter(function (c) { return c.isKernel; }).length;
    $('callableSummary').innerHTML = '<strong>' + all.length + '</strong> 个 callable'
      + '<span class="ptx-rail__sub">' + kernels + ' 个最终落到 AIC / AIV 核上</span>';

    var filter = state.cFilter.toLowerCase();
    var cur = currentCallable();
    var html = '';
    var shown = 0;

    all.forEach(function (c) {
      if (state.onlyKernels && !c.isKernel) return;
      if (filter && c.name.toLowerCase().indexOf(filter) < 0) return;
      shown++;
      var kindLabel = c.split ? 'AIC+AIV' : (c.kinds[0] || '—');
      html += '<button class="ptx-pass ptx-callable' + (cur && c.name === cur.name ? ' is-active' : '')
        + '" data-callable="' + esc(c.name) + '">'
        + '<span class="ptx-pass__idx">' + (c.birth ? String(c.birth.idx).padStart(2, '0') : '--') + '</span>'
        + '<span class="ptx-pass__body">'
        + '<span class="ptx-pass__name">' + esc(c.name) + '</span>'
        + '<span class="ptx-callable__meta">' + esc(kindLabel) + ' · ' + c.touchCount + ' 次改动</span>'
        + '</span>'
        + '<span class="ptx-pass__churn">' + fmt(c.finalLines) + '</span>'
        + '</button>';
    });

    $('callableList').innerHTML = shown ? html : '<p class="ptx-empty">没有匹配的 callable。</p>';
    var active = $('callableList').querySelector('.is-active');
    if (active) active.scrollIntoView({ block: 'nearest' });
  }

  // ── journey: what each Pass actually did to this callable ─────────────
  //
  // Line count is volume, not meaning. The structural counters the index
  // already carries per function — tile allocations, loops, tasks — say what a
  // Pass *did*: AutoTileMatmulL0 adds loops because it blocks the matmul,
  // MemoryReuse drops allocations because it shares buffers. Statements are
  // kept, but demoted: they track size, which is the least informative signal.

  var MEMBER_TONE = { AIC: 'aic', AIV: 'aiv', Group: 'group', InCore: 'incore' };

  function memberTone(part, rootName) {
    if (part.name !== rootName) return part.name.slice(-4) === '_aic' ? 'aic' : 'aiv';
    // Anything that is not a device-side compute region (Orchestration, Graph,
    // Spmd, the Group shell) reads as scaffolding, not as the kernel itself.
    return MEMBER_TONE[part.kind] || 'group';
  }

  var DIM = [
    { key: 'allocs', label: 'tile 分配', rank: 3 },
    { key: 'loops', label: '循环', rank: 2 },
    { key: 'tasks', label: '任务', rank: 2 },
    { key: 'stmts', label: '语句', rank: 1 },
  ];

  function deltasOf(prev, t) {
    return DIM.map(function (d) {
      var v = t[d.key] - (prev ? prev[d.key] : 0);
      return v ? { key: d.key, label: d.label, v: v, rank: d.rank } : null;
    }).filter(Boolean).sort(function (a, b) {
      return b.rank - a.rank || Math.abs(b.v) - Math.abs(a.v);
    });
  }

  /** Only the Passes that actually touched this callable, in pipeline order. */
  function journeySteps(c) {
    var steps = [];
    var prev = null;
    c.timeline.forEach(function (t) {
      if (t.marks.length && (t.parts.length || prev)) {
        steps.push({
          t: t,
          deltas: deltasOf(prev, t),
          kindChange: prev && prev.kinds && prev.kinds !== t.kinds
            ? { from: prev.kinds, to: t.kinds } : null,
        });
      }
      if (t.parts.length) prev = t;
    });
    return steps;
  }

  function signed(v) { return (v > 0 ? '+' : '') + v; }

  function journeyBar(c, steps) {
    if (!steps.length) return '<p class="ptx-empty">没有任何 Pass 改动过这个 callable。</p>';

    var html = '<div class="ptx-journey" id="journeyBar">';
    var lastPhase = null;

    steps.forEach(function (s) {
      if (s.t.phase !== lastPhase) {
        var ph = phase(s.t.phase);
        html += '<div class="ptx-journey__phase" title="' + esc(ph.hint) + '">'
          + '<span>' + esc(ph.label) + '</span></div>';
        lastPhase = s.t.phase;
      }

      var primary = s.deltas[0];
      // Once a structural counter explains the Pass, the statement count adds
      // nothing — it is the volume signal, not the optimisation.
      var rest = s.deltas.slice(1, 3).filter(function (d) {
        return !(d.key === 'stmts' && primary && primary.key !== 'stmts');
      });
      var churn = s.t.marks.reduce(function (a, m) {
        return a + (m.add || 0) + (m.del || 0)
          + (m.status === 'added' ? (m.linesAfter || 0) : 0)
          + (m.status === 'removed' ? (m.linesBefore || 0) : 0);
      }, 0);

      html += '<button class="ptx-step' + (s.t.idx === state.passIdx ? ' is-active' : '')
        + '" data-step="' + s.t.idx + '" title="' + esc(s.t.name) + '">'
        + '<span class="ptx-step__idx">' + String(s.t.idx).padStart(2, '0') + '</span>'
        + '<span class="ptx-step__name">' + esc(s.t.name) + '</span>';

      if (s.kindChange) {
        html += '<span class="ptx-step__kind">' + esc(s.kindChange.from)
          + ' → ' + esc(s.kindChange.to) + '</span>';
      }

      html += '<span class="ptx-step__badges">';
      if (primary) {
        html += '<b class="' + (primary.v > 0 ? 'is-up' : 'is-down') + '">'
          + esc(primary.label) + ' ' + signed(primary.v) + '</b>';
      }
      rest.forEach(function (d) {
        html += '<i>' + esc(d.label) + ' ' + signed(d.v) + '</i>';
      });
      // No counter moved, but the Pass still rewrote lines in place (filling
      // addresses, renaming, reordering). Say how much and let the panes below
      // show what.
      if (!primary && !s.kindChange) {
        html += '<i>就地改写' + (churn ? ' ' + fmt(churn) + ' 行' : '') + '</i>';
      }
      html += '</span></button>';
    });

    return html + '</div>';
  }

  /** Start / peak / end of the structural counters — the optimisation outcome. */
  function journeySummary(c, steps) {
    if (!c.birth) return '';
    var alive = c.timeline.filter(function (t) { return t.parts.length; });
    var last = alive[alive.length - 1];
    var cells = DIM.slice().sort(function (a, b) { return b.rank - a.rank; }).map(function (d) {
      var start = c.birth[d.key];
      var end = last[d.key];
      var peak = alive.reduce(function (m, t) { return Math.max(m, t[d.key]); }, 0);
      if (!peak) return '';
      var saved = peak - end;
      return '<div class="ptx-outcome">'
        + '<span class="ptx-outcome__label">' + esc(d.label) + '</span>'
        + '<span class="ptx-outcome__value">' + start + ' → <b>' + end + '</b></span>'
        + (saved > 0 ? '<span class="ptx-outcome__note">峰值 ' + peak
            + '，最终省下 ' + saved + '</span>' : '<span class="ptx-outcome__note">峰值即终值</span>')
        + '</div>';
    }).join('');
    return cells ? '<div class="ptx-outcomes">' + cells + '</div>' : '';
  }

  function renderCallableView() {
    var c = currentCallable();
    if (!c) {
      $('viewCallable').innerHTML = '<p class="ptx-empty">这份 run 没有可追踪的 callable。</p>';
      return;
    }
    state.callable = c.name;  // resolve the fallback so the URL names it too

    var steps = journeySteps(c);
    // Keep the selected Pass on one that actually touched this callable.
    if (!steps.some(function (s) { return s.t.idx === state.passIdx; })) {
      state.passIdx = steps.length ? steps[0].t.idx : state.passIdx;
    }

    $('cKind').textContent = c.split ? 'AIC + AIV' : (c.kinds[0] || '—');
    $('cName').textContent = c.name;
    $('cDelta').innerHTML = c.birth
      ? '<span class="ptx-muted">' + steps.length + ' 个 Pass 改动过它</span>'
      : '<span class="ptx-muted">未出现</span>';
    $('cHeadline').textContent = c.birth
      ? (c.birth.idx === 0
          ? '在前端 IR 中已存在'
          : 'Pass ' + String(c.birth.idx).padStart(2, '0') + ' 被外提')
        + '，穿过 ' + (c.timeline.length - 1 - c.birth.idx) + ' 个后续 Pass。选一步看它做了什么。'
      : '这个 callable 在本次编译中没有留下函数体。';
    $('cSource').textContent = c.members.length > 1 ? c.members.join(' · ') : '';

    $('viewCallable').innerHTML = journeySummary(c, steps) + journeyBar(c, steps);

    var active = $('viewCallable').querySelector('.ptx-step.is-active');
    if (active) active.scrollIntoView({ block: 'nearest', inline: 'center' });
  }

  /**
   * When a Pass touches several members of a family, focusing the one that
   * moved most is what the reader came for — the Group shell usually changes
   * by a line or two and would otherwise win just by sorting first.
   */
  function dominantMark(t, fallback) {
    if (!t || !t.marks.length) return fallback;
    var best = t.marks[0];
    var bestD = -1;
    t.marks.forEach(function (m) {
      var d = Math.abs((m.linesAfter || 0) - (m.linesBefore || 0));
      if (d > bestD) { bestD = d; best = m; }
    });
    return best.name;
  }

  /** Pick the family member this Pass moved most, so the panes open on it. */
  function focusMemberFor(c, idx) {
    var t = c.timeline.find(function (x) { return x.idx === idx; });
    return dominantMark(t, c.name);
  }

  function selectStep(idx) {
    var c = currentCallable();
    state.passIdx = idx;
    state.fn = focusMemberFor(c, idx);
    var p = run().passes[idx];
    if (p) { state.lens = suggestedLens(p); state.lensAuto = true; }
    render();
  }


  function jumpToPassDiff(idx, fn) {
    state.mode = 'pass';
    state.passIdx = idx;
    state.tab = 'diff';
    state.fn = fn || null;
    state.lens = run().passes[idx] ? run().passes[idx].lens : state.lens;
    render();
  }

  function selectCallable(name) {
    if (name === state.callable) return;
    state.callable = name;
    render();
  }

  function setMode(mode) {
    if (mode === state.mode) return;
    state.mode = mode;
    if (mode === 'callable' && !state.callable) {
      var all = callables();
      if (all.length) state.callable = all[0].name;
    }
    render();
  }

  // ══════════════════════════════════════════════════════════════════════
  // Wiring
  // ══════════════════════════════════════════════════════════════════════

  function render() {
    var byCallable = state.mode === 'callable';

    document.querySelectorAll('#railMode button').forEach(function (b) {
      b.classList.toggle('is-active', b.dataset.mode === state.mode);
    });
    $('railHeadPass').hidden = byCallable;
    $('railHeadCallable').hidden = !byCallable;
    $('passList').hidden = byCallable;
    $('callableList').hidden = !byCallable;
    $('passHead').hidden = byCallable;
    $('callableHead').hidden = !byCallable;
    $('viewCallable').hidden = !byCallable;

    document.querySelector('.ptx-main').classList.toggle('is-callable', byCallable);

    if (byCallable) {
      // Three regions: the journey on top, "what this Pass did" (structure
      // graph) bottom-left, "what changed" (diff) bottom-right. Both panes are
      // the existing views, re-laid-out by CSS rather than reimplemented.
      $('viewOverview').hidden = true;
      $('viewGraph').hidden = false;
      $('viewDiff').hidden = false;
      document.body.classList.add('ptx--nodoc');
      renderCallableRail();
      renderCallableView();
      renderGraph();
      renderDiff();
      writeHash();
      return;
    }
    document.body.classList.toggle('ptx--nodoc', !state.docOpen);

    renderRail();
    renderHeader();
    if (state.tab === 'overview') renderOverview();
    if (state.tab === 'diff') renderDiff();
    if (state.tab === 'graph') renderGraph();
    renderDoc();
    writeHash();
  }

  function selectPass(idx) {
    if (state.mode === 'callable') return;
    var r = run();
    idx = Math.max(0, Math.min(r.passes.length - 1, idx));
    if (idx === state.passIdx) return;
    state.passIdx = idx;
    state.fn = null;
    state.lens = r.passes[idx].lens;
    render();
  }

  function writeHash() {
    var h = state.mode === 'callable'
      ? '#' + state.runId + '/c/' + encodeURIComponent(state.callable || '') + '/' + state.passIdx
      : '#' + state.runId + '/' + state.passIdx + '/' + state.tab
        + (state.fn ? '/' + encodeURIComponent(state.fn) : '');
    if (location.hash !== h) history.replaceState(null, '', h);
  }

  function readHash() {
    var parts = location.hash.replace(/^#/, '').split('/');
    if (!parts[0]) return;
    if (INDEX.runs.some(function (r) { return r.id === parts[0]; })) state.runId = parts[0];
    if (parts[1] === 'c') {
      state.mode = 'callable';
      if (parts[2]) state.callable = decodeURIComponent(parts[2]);
      var pi = Number(parts[3]);
      if (!Number.isNaN(pi) && parts[3] !== '') state.passIdx = pi;
      return;
    }
    var idx = Number(parts[1]);
    if (!Number.isNaN(idx)) state.passIdx = idx;
    if (['overview', 'diff', 'graph'].indexOf(parts[2]) >= 0) state.tab = parts[2];
    if (parts[3]) state.fn = decodeURIComponent(parts[3]);
  }

  function boot() {
    $('runSelect').innerHTML = INDEX.runs.map(function (r) {
      return '<option value="' + r.id + '">' + esc(r.title) + '</option>';
    }).join('');

    readHash();
    $('runSelect').value = state.runId;
    state.lens = state.lens || suggestedLens(pass());
    updateRunMeta();

    $('runSelect').addEventListener('change', function (e) {
      state.runId = e.target.value;
      state.passIdx = Math.min(state.passIdx, run().passes.length - 1);
      state.fn = null;
      state.callable = null;
      updateRunMeta();
      render();
    });

    $('passList').addEventListener('click', function (e) {
      var b = e.target.closest('.ptx-pass');
      if (b) selectPass(Number(b.dataset.idx));
    });

    $('railMode').addEventListener('click', function (e) {
      var b = e.target.closest('button');
      if (b) setMode(b.dataset.mode);
    });

    $('callableList').addEventListener('click', function (e) {
      var b = e.target.closest('.ptx-callable');
      if (b) selectCallable(b.dataset.callable);
    });

    $('onlyKernels').addEventListener('change', function (e) {
      state.onlyKernels = e.target.checked;
      renderCallableRail();
    });
    $('callableFilter').addEventListener('input', function (e) {
      state.cFilter = e.target.value.trim();
      renderCallableRail();
    });

    $('viewCallable').addEventListener('click', function (e) {
      var s = e.target.closest('.ptx-step');
      if (s) selectStep(Number(s.dataset.step));
    });

    $('onlyChanged').addEventListener('change', function (e) {
      state.onlyChanged = e.target.checked;
      renderRail();
    });
    $('passFilter').addEventListener('input', function (e) {
      state.filter = e.target.value.trim();
      renderRail();
    });

    document.querySelector('.ptx-tabs').addEventListener('click', function (e) {
      var b = e.target.closest('.ptx-tab');
      if (!b) return;
      state.tab = b.dataset.tab;
      render();
    });

    $('viewOverview').addEventListener('click', function (e) {
      var b = e.target.closest('[data-openfn]');
      if (!b) return;
      state.fn = b.dataset.openfn;
      state.tab = 'diff';
      render();
    });

    $('diffFnChips').addEventListener('click', function (e) {
      var b = e.target.closest('.ptx-chip');
      if (!b) return;
      state.fn = b.dataset.fn;
      renderDiff();
      writeHash();
    });

    $('diffMode').addEventListener('click', function (e) {
      var b = e.target.closest('button');
      if (!b) return;
      state.diffMode = b.dataset.mode;
      $('diffMode').querySelectorAll('button').forEach(function (x) { x.classList.toggle('is-active', x === b); });
      renderDiff();
    });

    $('lensPicker').addEventListener('click', function (e) {
      var b = e.target.closest('button');
      if (!b) return;
      state.lens = b.dataset.lens;
      state.lensAuto = false;
      renderGraph();
      writeHash();
    });

    $('graphFn').addEventListener('change', function (e) {
      state.fn = e.target.value;
      renderGraph();
      writeHash();
    });

    $('prevPass').addEventListener('click', function () { selectPass(state.passIdx - 1); });
    $('nextPass').addEventListener('click', function () { selectPass(state.passIdx + 1); });

    $('docToggle').addEventListener('click', function () {
      if (state.mode === 'callable') { toast('Pass 说明只在「按 Pass」视图下可用。'); return; }
      state.docOpen = !state.docOpen;
      document.body.classList.toggle('ptx--nodoc', !state.docOpen);
      renderDoc();
    });

    $('themeToggle').addEventListener('click', function () {
      var root = document.documentElement;
      root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
    });

    document.addEventListener('keydown', function (e) {
      if (/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
      if (e.key === 'c' || e.key === 'C') { setMode(state.mode === 'callable' ? 'pass' : 'callable'); return; }
      if (state.mode === 'callable') return;
      if (e.key === 'ArrowLeft') { selectPass(state.passIdx - 1); e.preventDefault(); }
      if (e.key === 'ArrowRight') { selectPass(state.passIdx + 1); e.preventDefault(); }
      if (e.key === '1') { state.tab = 'overview'; render(); }
      if (e.key === '2') { state.tab = 'diff'; render(); }
      if (e.key === '3') { state.tab = 'graph'; render(); }
    });

    render();
  }

  function updateRunMeta() {
    var r = run();
    var last = r.passes[r.passes.length - 1];
    $('runMeta').innerHTML = esc(r.subtitle) + ' · <code>' + esc(r.dir) + '</code> · '
      + (r.passes.length - 1) + ' 个 Pass · 终态 ' + last.functions.length + ' 个函数';
  }

  boot();
})();
