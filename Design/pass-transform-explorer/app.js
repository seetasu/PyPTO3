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

  /** Keep a rendered item visible without letting the browser scroll the page. */
  function revealIn(container, target, center) {
    if (!container || !target) return;
    var outer = container.getBoundingClientRect();
    var inner = target.getBoundingClientRect();
    var y = inner.top < outer.top
      ? inner.top - outer.top
      : inner.bottom > outer.bottom
        ? inner.bottom - outer.bottom
        : 0;
    var x = inner.left < outer.left
      ? inner.left - outer.left
      : inner.right > outer.right
        ? inner.right - outer.right
        : 0;
    if (center && y) y -= (outer.height - inner.height) / 2;
    if (center && x) x -= (outer.width - inner.width) / 2;
    if (y) container.scrollTop += y;
    if (x) container.scrollLeft += x;
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
    jumpSide: 'after',   // which dump file jumpLine numbers - the gutters differ
    filter: '',
    docOpen: true,
    lensAuto: true,
    lensWanted: null,
    perfFilter: null,    // a tier id isolates that tier in the rail
    callable: null,
    cFilter: '',
    onlyKernels: false,
  };

  // ── performance tiers ─────────────────────────────────────────────────
  //
  // The "decide" tier is a fact about the compiler: those three Passes own the
  // only perf-hint codes in PyPTO. "shape" and "traffic" are a reading of the
  // pass docs. "form" is the residual and deliberately carries no badge, so an
  // unmarked Pass reads as "nothing to report", not "verified neutral".
  var PERF_TIERS = INDEX.perfTiers || [];
  var PERF_BY_ID = {};
  PERF_TIERS.forEach(function (t) { PERF_BY_ID[t.id] = t; });

  function perfTier(id) { return PERF_BY_ID[id] || PERF_BY_ID.form || { id: 'form', rank: 0, label: '', short: '' }; }
  /** Every pass has a perf block after a rebuild; older data degrades to "form". */
  function perfOf(p) { return (p && p.perf) || { tier: 'form' }; }

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
    var rp = r.perf;
    var ownedHints = r.passes.reduce(function (n, p) {
      return n + (p.perf && p.perf.hints ? p.perf.hints.lines : 0);
    }, 0);

    $('railSummary').innerHTML = '<strong>' + changedCount + '</strong> / ' + total
      + ' 个 Pass 改动了 IR<span class="ptx-rail__sub">' + (total - changedCount) + ' 个对本算子为空操作</span>'
      + (rp
        ? '<span class="ptx-rail__sub">本次编译 <b>' + rp.total + '</b> 条性能提示：'
          + (ownedHints
            ? '<b class="ptx-perfnum">' + ownedHints + '</b> 条来自 Pass'
            : '<span class="ptx-muted">没有一条来自 Pass</span>')
          + (rp.total - ownedHints ? '，' + (rp.total - ownedHints) + ' 条另有来源（见第 00 步）' : '')
          + '</span>'
        : '');

    renderPerfKey(r);

    var maxChurn = 1;
    r.passes.forEach(function (p) { maxChurn = Math.max(maxChurn, p.diff.add + p.diff.del); });

    var filter = state.filter.toLowerCase();
    var html = '';
    var lastPhase = null;

    r.passes.forEach(function (p) {
      if (state.onlyChanged && p.idx > 0 && !p.changed) return;
      if (state.perfFilter && perfOf(p).tier !== state.perfFilter) return;
      if (filter && p.name.toLowerCase().indexOf(filter) < 0) return;

      if (p.phase !== lastPhase) {
        var ph = phase(p.phase);
        html += '<div class="ptx-phasehead" title="' + esc(ph.hint) + '">' + esc(ph.label) + '</div>';
        lastPhase = p.phase;
      }

      var churn = p.diff.add + p.diff.del;
      var w = churn ? Math.max(3, Math.round((churn / maxChurn) * 100)) : 0;
      var addW = churn ? Math.round((p.diff.add / churn) * w) : 0;

      var pf = perfOf(p);
      var tier = perfTier(pf.tier);
      var hints = pf.hints;

      // Two marks, kept separate on purpose. The tier is the standing claim and
      // rides the row's left edge, so it costs the name no width and reads as a
      // column. The hint count is what THIS compilation reported, so it gets a
      // badge — a high tier with no badge is a Pass that decided without
      // complaint, which is a different thing and should look different.
      var marks = hints
        ? '<span class="ptx-pass__hints" title="' + esc('本次编译在这一步发出 ' + hints.lines
            + ' 条性能提示（' + Object.keys(hints.codes).join(' / ') + '）') + '">' + hints.lines + '</span>'
        : '';

      html += '<button class="ptx-pass pt-' + pf.tier + (tier.short ? ' has-tier' : '')
        + (p.idx === state.passIdx ? ' is-active' : '')
        + (p.idx > 0 && !p.changed ? ' is-noop' : '')
        + (hints ? ' has-hints' : '') + '" data-idx="' + p.idx + '"'
        + (tier.short ? ' title="' + esc(tier.label + '（依据：' + tier.basis + '）\n' + tier.hint) + '"' : '')
        + '>'
        + '<span class="ptx-pass__idx">' + String(p.idx).padStart(2, '0') + '</span>'
        + '<span class="ptx-pass__body">'
        + '<span class="ptx-pass__name">' + esc(p.name === 'frontend' ? '前端 IR' : p.name)
        + marks + '</span>'
        + '<span class="ptx-pass__bar">'
        + '<i class="ptx-pass__bar-add" style="width:' + addW + '%"></i>'
        + '<i class="ptx-pass__bar-del" style="width:' + (w - addW) + '%"></i>'
        + '</span></span>'
        + '<span class="ptx-pass__churn">' + (churn ? fmt(churn) : '—') + '</span>'
        + '</button>';
    });

    $('passList').innerHTML = html || '<p class="ptx-empty">没有匹配的 Pass。</p>';
    var active = $('passList').querySelector('.is-active');
    revealIn($('passList'), active);
  }

  /**
   * Tier legend, doubling as a filter. Counts are per run, so a run whose
   * pipeline lacks a Pass does not advertise an empty tier.
   */
  function renderPerfKey(r) {
    var host = $('perfKey');
    if (!host) return;
    var counts = {};
    r.passes.forEach(function (p) {
      var t = perfOf(p).tier;
      counts[t] = (counts[t] || 0) + 1;
    });

    var html = '';
    PERF_TIERS.slice().sort(function (a, b) { return b.rank - a.rank; }).forEach(function (t) {
      if (!counts[t.id]) return;
      var on = state.perfFilter === t.id;
      html += '<button class="ptx-perfkey__item pt-' + t.id + (on ? ' is-on' : '') + '"'
        + ' data-tier="' + t.id + '" aria-pressed="' + (on ? 'true' : 'false')
        + '" title="' + esc(t.label + '（依据：' + t.basis + '）\n' + t.hint) + '">'
        + '<i></i><span>' + esc(t.label) + '</span><b>' + counts[t.id] + '</b></button>';
    });
    host.innerHTML = html;
  }

  // ══════════════════════════════════════════════════════════════════════
  // Pass header
  // ══════════════════════════════════════════════════════════════════════

  function renderHeader() {
    var p = pass();
    var ph = phase(p.phase);
    $('passPhase').textContent = ph.label;
    $('passPhase').title = ph.hint;

    var pf = perfOf(p);
    var tier = perfTier(pf.tier);
    var chip = $('passPerfTier');
    if (chip) {
      chip.hidden = !tier.short;
      chip.className = 'ptx-perftier pt-' + pf.tier;
      chip.textContent = tier.label;
      chip.title = tier.label + '（依据：' + tier.basis + '）\n' + tier.hint;
    }
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

  // ═══════════════════════════════════════════════════════════════════════
  // Function body migration
  // ═══════════════════════════════════════════════════════════════════════

  var MOVE_DIR = {
    dissolve: {
      verb: '溶解',
      lead: '这些函数被拆掉，函体直接写进了调用方',
      fromLab: 'Pass 前：被溶解的函数',
      toLab: 'Pass 后：接收方',
    },
    extract: {
      verb: '外提',
      lead: '这些代码段被识别出来，外提成了独立函数',
      fromLab: 'Pass 后：新函数',
      toLab: 'Pass 前：它们原来所在的函数',
    },
  };

  /**
   * The headline figure for a Pass that moves whole function bodies.
   *
   * Left column is one row per body, height proportional to its statement
   * count, so the reader sees at a glance that `hc_pre` is 193 statements and
   * `rms_norm` is 10. Ribbons carry that weight to the destination. A body
   * copied into several places fans out and is labelled with the copy count,
   * because duplication is the whole cost of inlining and a single line would
   * hide it.
   *
   * The destination box reports the attribution AND the measured growth side by
   * side. They differ by a few percent - inlining drops the call statement and
   * folds some constants - and showing one number would be passing an estimate
   * off as a measurement.
   */
  function drawMigration(mg, fnHint) {
    if (mg.direction === 'wrap') return drawWrapFigure(mg);

    var D = MOVE_DIR[mg.direction];
    var C = mg.counts;
    var hosts = mg.hosts.slice();
    var orphans = mg.moves.filter(function (m) { return !m.host; });
    if (!hosts.length) return '';

    // One row per BODY. Drawing a row per (body, host) pair counted a body
    // copied into two callers twice, which reported 21 where the function
    // census says 20 - and made the two cards disagree.
    var bodies = mg.bodies.filter(function (b) { return b.to.length; });
    var CAP = 14;
    var shown = bodies.slice(0, CAP);
    var rest = bodies.slice(CAP);
    if (rest.length) {
      var restW = rest.reduce(function (n, b) { return n + b.stmts * Math.max(b.copies, 1); }, 0);
      shown.push({
        body: '其余 ' + rest.length + ' 个',
        stmts: restW, copies: 1, lines: 0, folded: rest.length,
        to: [{ host: rest[0].to[0].host, copies: 1 }],
      });
    }

    var padL = 8, colW = 232, gap = 140, rowH = 26, rowGap = 4, top = 34;
    var NAME_W = 140, BAR_MAX = 54;
    var W = padL + colW + gap + 230;
    var leftH = shown.length * (rowH + rowGap);

    // Host boxes are sized by attributed weight, but every box has a floor so
    // a small destination stays readable. Because of that floor the heights
    // can add up to more than the space they were given, so they are laid out
    // FIRST and the canvas is sized to whatever they actually need - the
    // previous version sized the canvas first and silently clipped the last
    // two boxes off the bottom, which read as "21 functions became one".
    var BOX_MIN = 46, BOX_GAP = 10;
    var totalClaim = hosts.reduce(function (n, h) { return n + Math.max(h.claimed, 1); }, 0);
    var budget = Math.max(leftH, hosts.length * (BOX_MIN + BOX_GAP));
    var hostBox = {}, hy = top;
    hosts.forEach(function (h) {
      var hh = Math.max(BOX_MIN, (Math.max(h.claimed, 1) / totalClaim) * budget - BOX_GAP);
      hostBox[h.name] = { y: hy, h: hh };
      hy += hh + BOX_GAP;
    });
    var rightH = hy - top - BOX_GAP;
    var H = top + Math.max(leftH, rightH) + 16;

    var maxW = Math.max.apply(null, shown.map(function (b) { return b.stmts * Math.max(b.copies, 1); }));
    var svg = '<svg class="mig" viewBox="0 0 ' + W + ' ' + H + '" width="' + W + '" height="' + H
      + '" style="min-width:' + W + 'px" role="img" aria-label="' + esc(D.verb + '迁移图') + '">';

    svg += '<text class="mig-col" x="' + padL + '" y="18">' + esc(D.fromLab) + '</text>'
      + '<text class="mig-col" x="' + (padL + colW + gap) + '" y="18">' + esc(D.toLab) + '</text>';

    // ribbons first, so the boxes sit on top of them
    var x1 = padL + colW, x2 = padL + colW + gap, mx = (x1 + x2) / 2;
    var fill = {};
    shown.forEach(function (b, i) {
      var y = top + i * (rowH + rowGap) + rowH / 2;
      // A body with several destinations draws several ribbons from its ONE
      // row, so the fan-out is visible without inventing extra rows.
      b.to.forEach(function (t) {
        var hb = hostBox[t.host];
        if (!hb) return;
        var w = b.stmts * t.copies;
        var th = Math.max(1.5, (w / maxW) * 13);
        fill[t.host] = (fill[t.host] || 0) + th + 1.5;
        var ty = hb.y + Math.min(hb.h - 4, fill[t.host] - th / 2);
        svg += '<path class="mig-ribbon' + (b.folded ? ' is-folded' : '') + '" d="M' + x1 + ' ' + y
          + ' C' + mx + ' ' + y + ',' + mx + ' ' + ty + ',' + x2 + ' ' + ty
          + '" style="stroke-width:' + th.toFixed(1) + '"/>';
        if (t.copies > 1) {
          svg += '<text class="mig-copies" x="' + (mx + 2) + '" y="' + ((y + ty) / 2 - 3)
            + '" text-anchor="middle">×' + t.copies + '</text>';
        }
      });
    });

    // source rows
    shown.forEach(function (b, i) {
      var y = top + i * (rowH + rowGap);
      var w = b.stmts * Math.max(b.copies, 1);
      var barW = Math.max(3, (w / maxW) * BAR_MAX);
      var hot = fnHint && b.body === fnHint;
      var where = b.to.length > 1
        ? ' · 进了 ' + b.to.length + ' 个调用方'
        : b.copies > 1 ? ' · 复制了 ' + b.copies + ' 份' : '';
      svg += '<g class="mig-src' + (b.folded ? ' is-folded' : '') + (hot ? ' is-hot' : '')
        + '"' + (b.folded ? '' : ' data-fn="' + esc(b.body) + '" data-line="' + (b.line || 0)
          + '" data-side="' + (b.side || 'after') + '"') + '>'
        + '<title>' + esc(b.body + ' · ' + b.stmts + ' 语句'
          + (b.lines ? ' / ' + b.lines + ' 行' : '') + where
          + (b.folded ? '' : ' · 点击看代码')) + '</title>'
        + '<rect class="mig-row" x="' + padL + '" y="' + y + '" width="' + colW + '" height="' + rowH + '" rx="5"/>'
        + '<text class="mig-name" x="' + (padL + 8) + '" y="' + (y + 17) + '">'
        + esc(fitText(b.body, 11, NAME_W)) + '</text>'
        + '<rect class="mig-bar" x="' + (padL + colW - 8 - barW) + '" y="' + (y + 9)
        + '" width="' + barW + '" height="' + (rowH - 18) + '" rx="2"/>'
        + '<text class="mig-w" x="' + (padL + colW - 10 - BAR_MAX) + '" y="' + (y + 17)
        + '" text-anchor="end">' + w + '</text>'
        + '</g>';
    });

    // destination boxes
    hosts.forEach(function (h) {
      var bx = hostBox[h.name];
      var gapPct = h.measured ? Math.round((h.claimed / h.measured - 1) * 100) : null;
      svg += '<g class="mig-host" data-fn="' + esc(h.name) + '" data-line="' + (h.line || 0)
        + '" data-side="' + (h.side || 'after') + '">'
        + '<title>' + esc(h.name + ' · 语句 ' + h.before + ' → ' + h.after
          + ' · 收下 ' + h.bodies + ' 个函体 · 点击看代码') + '</title>'
        + '<rect class="mig-hostbox" x="' + x2 + '" y="' + bx.y + '" width="222" height="' + bx.h + '" rx="7"/>'
        + '<text class="mig-hostname" x="' + (x2 + 10) + '" y="' + (bx.y + 17) + '">'
        + esc(fitText(h.name, 13, 202)) + '</text>'
        + '<text class="mig-hostnum" x="' + (x2 + 10) + '" y="' + (bx.y + 32) + '">'
        + '收下 ' + h.bodies + ' 个 · 语句 ' + fmt(h.before) + ' → ' + fmt(h.after) + '</text>';
      if (bx.h >= 58) {
        svg += '<text class="mig-hostsub" x="' + (x2 + 10) + '" y="' + (bx.y + 47) + '">'
          + '归因 ' + fmt(h.claimed) + ' · 实测 ' + fmt(Math.abs(h.measured))
          + (gapPct !== null && Math.abs(gapPct) >= 1 ? '（差 ' + (gapPct > 0 ? '+' : '') + gapPct + '%）' : '')
          + '</text>';
      }
      svg += '</g>';
    });

    svg += '</svg>';

    // The lead line has to reconcile with the 函数结构 card sitting next to it,
    // so it counts functions the same way that card does and says where the
    // survivors that received nothing went.
    var idle = C.functionsAfter - C.hosts;
    var note = '<p class="mig-note">' + esc(D.lead) + '：<b>' + C.bodies + '</b> 个函数'
      + '（＝「函数结构」里的' + (mg.direction === 'dissolve' ? '移除数' : '新增数') + '）'
      + '，去向是 <b>' + C.hosts + '</b> 个' + (mg.direction === 'dissolve' ? '调用方' : '原宿主')
      + (idle > 0 ? '；另外 ' + idle + ' 个函数没有参与' : '')
      + '。</p>';

    // A body that reached two hosts is counted by both, so the 收下 numbers on
    // the right sum to more than the body count on the left. That is exactly
    // the arithmetic a careful reader does first, so say it rather than leave
    // two true numbers looking like a contradiction.
    var hostSum = hosts.reduce(function (n, h) { return n + h.bodies; }, 0);
    note += '<p class="mig-note">搬动了 <b>' + fmt(C.stmts) + '</b> 条语句'
      + (C.stmtsWithCopies !== C.stmts
        ? '；其中 <b>' + C.duplicated + '</b> 个函体进了多个去处，按份数算实际写入 <b>'
          + fmt(C.stmtsWithCopies) + '</b> 条'
        : '')
      + '。'
      + (hostSum !== C.bodies
        ? '<span class="mig-caveat">右侧「收下」相加是 ' + hostSum + '，比 ' + C.bodies
          + ' 多，因为跨去处的函体被两边各记了一次。</span>'
        : '')
      + '</p>';

    if (orphans.length) {
      note += '<p class="mig-note mig-note--warn">' + orphans.length
        + ' 个函数没有幸存的调用方，无法归因：<code>'
        + orphans.slice(0, 6).map(function (m) { return esc(m.body); }).join('</code> <code>') + '</code></p>';
    }
    note += '<p class="mig-legend">左列一行一个函数，条长和条带粗细 = 语句数（进了多个去处的按份数计）。'
      + '归因来自调用图，实测来自两份快照的语句差；两者不完全相等是因为调用语句本身会消失。'
      + '点击任一方块跳到代码。</p>';

    return '<div class="mig-wrap">' + svg + '</div>' + note;
  }
  /**
   * `OutlineClusterScopes` creates shells, it does not move bodies. Drawing a
   * flow here would invent one, so this says what happened instead.
   */
  function drawWrapFigure(mg) {
    var ws = mg.wraps.slice().sort(function (a, b) { return b.stmts - a.stmts; });
    var CAP = 12;
    var rows = ws.slice(0, CAP).map(function (w) {
      return '<tr data-fn="' + esc(w.body) + '" data-line="' + (w.to || 0) + '" data-side="after">'
        + '<td><code>' + esc(w.body) + '</code></td>'
        + '<td class="ptx-num">' + w.stmts + '</td>'
        + '<td class="mig-around">包住 <code>' + esc(w.wrapped) + '</code>'
        + (w.via && w.via !== 'direct' ? ' <span>' + esc(w.via) + '</span>' : '') + '</td></tr>';
    }).join('');
    var med = ws.length ? ws[Math.floor(ws.length / 2)].stmts : 0;
    return '<p class="mig-note">这一步<b>没有搬动任何函体</b>。它新建了 <b>'
      + ws.length + '</b> 个壳函数（中位数 ' + med
      + ' 条语句），每个只是把一个已有函数包一层，'
      + '原函体原地不动。</p>'
      + '<table class="ptx-table ptx-table--wrap"><thead><tr><th>新壳函数</th><th>语句</th>'
      + '<th>包的是谁</th></tr></thead><tbody>' + rows + '</tbody></table>'
      + (ws.length > CAP ? '<p class="ptx-more">另有 ' + (ws.length - CAP) + ' 个</p>' : '');
  }
  /**
   * A before/after bar pair, for the cards whose whole content was a table of
   * `before  after` numbers. `range 85 → 88` is four numbers the reader has to
   * subtract in their head and still cannot see the shape of; two bars on a
   * shared scale show the composition, the direction and the size at once.
   *
   * The scale is shared across every row so the bars are comparable to each
   * other, not each normalized to itself - a row that barely moved should look
   * like it barely moved.
   */
  function compareBars(rows, opts) {
    opts = opts || {};
    var max = rows.reduce(function (m, r) { return Math.max(m, r.before, r.after); }, 1);
    var body = rows.map(function (r) {
      var d = r.after - r.before;
      var cls = d > 0 ? 'is-up' : d < 0 ? 'is-down' : 'is-flat';
      return '<div class="cmp-row ' + cls + '">'
        + '<span class="cmp-label" title="' + esc(String(r.label)) + '">'
        + esc(opts.labelOf ? opts.labelOf(r.label) : r.label) + '</span>'
        + '<span class="cmp-track">'
        + '<i class="cmp-bar cmp-bar--a" style="width:' + (r.before / max * 100).toFixed(2) + '%"></i>'
        + '<i class="cmp-bar cmp-bar--b" style="width:' + (r.after / max * 100).toFixed(2) + '%"></i>'
        + '</span>'
        + '<span class="cmp-nums"><b>' + fmt(r.before) + '</b> → <b>' + fmt(r.after) + '</b></span>'
        + '<span class="cmp-delta">' + (d > 0 ? '+' : d < 0 ? '−' : '') + (d ? fmt(Math.abs(d)) : '—') + '</span>'
        + '</div>';
    }).join('');
    return '<div class="cmp">'
      + '<div class="cmp-key"><i class="cmp-bar--a"></i>Pass 前<i class="cmp-bar--b"></i>Pass 后</div>'
      + body + '</div>';
  }

  var LOOP_LABEL = {
    range: 'range 普通循环',
    pipeline: 'pipeline 流水循环',
    spmd: 'spmd 多核并行',
    parallel: 'parallel 并行',
    unroll: 'unroll 展开',
    while: 'while 循环',
  };

  var SPACE_LABEL = {
    Vec: 'Vec 向量计算区',
    Mat: 'Mat 矩阵计算区',
    Acc: 'Acc 累加器',
    Left: 'Left L0A 左操作数',
    Right: 'Right L0B 右操作数',
    Bias: 'Bias 偏置表',
    GM: 'GM 片外内存',
  };

  /** Card bodies that deserve a picture instead of a table, keyed by card id. */
  function cardVisual(card) {
    if (!card.rows || !card.rows.length) return null;
    if (card.id === 'control') {
      var nest = card.rows.filter(function (r) { return r.label === '最大嵌套深度'; });
      var kinds = card.rows.filter(function (r) { return r.label !== '最大嵌套深度'; });
      var html = '';
      if (kinds.length) html += compareBars(kinds, { labelOf: function (k) { return LOOP_LABEL[k] || k; } });
      if (nest.length) {
        var n = nest[0];
        html += '<div class="nest">' + '<span class="nest-cap">最大嵌套深度</span>'
          + nestLadder(n.before, n.after) + '</div>';
      }
      return html;
    }
    if (card.id === 'memspace') {
      return compareBars(card.rows, { labelOf: function (k) { return SPACE_LABEL[k] || k; } });
    }
    return null;
  }

  /** Nesting depth as stacked rungs - a count that is really a shape. */
  function nestLadder(a, b) {
    var n = Math.max(a, b);
    var rungs = '';
    for (var i = 1; i <= n; i++) {
      var inA = i <= a, inB = i <= b;
      rungs += '<i class="nest-rung' + (inA && inB ? ' is-both' : inB ? ' is-new' : ' is-gone')
        + '" style="margin-left:' + ((i - 1) * 7) + 'px"></i>';
    }
    return '<span class="nest-ladder">' + rungs + '</span>'
      + '<span class="nest-num">' + a + ' → <b>' + b + '</b> 层</span>';
  }
  var TONE_CLASS = { add: 'is-add', remove: 'is-del', change: 'is-chg', neutral: '' };

  /**
   * What this Pass can do to performance, and what it actually reported.
   *
   * Two separate claims, never merged:
   *
   *   the tier   — a standing property of the Pass. "decide" is a fact (those
   *                three own PyPTO's only perf-hint codes); "shape" and
   *                "traffic" are a reading of the pass docs, and the card says
   *                which, so nothing here passes itself off as measured.
   *   the hints  — measured output from THIS compilation, read out of the run's
   *                own perf_hints.log. A high tier with no hints is a Pass that
   *                made its decisions without complaint; the card says that in
   *                as many words rather than leaving an empty table.
   */
  function perfCard(p) {
    var pf = perfOf(p);
    var tier = perfTier(pf.tier);
    var hints = pf.hints;
    if (!tier.short && !hints) return '';

    var html = '<section class="ptx-card ptx-perf' + (hints ? ' has-hints' : '') + '">'
      + '<h3>性能影响'
      + '<span class="ptx-perftier pt-' + pf.tier + '">' + esc(tier.label) + '</span>'
      + '<span class="ptx-perfbasis">依据：' + esc(tier.basis) + '</span></h3>';

    if (pf.why) html += '<p class="ptx-card__headline">' + mdInline(pf.why) + '</p>';
    if (pf.lever) {
      html += '<p class="ptx-perflever"><span>可调的地方</span>' + mdInline(pf.lever) + '</p>';
    }

    if (hints) {
      var codes = Object.keys(hints.codes).map(function (c) {
        return '<code>' + esc(c) + '</code>×' + hints.codes[c];
      }).join(' ');
      html += '<p class="ptx-perfhit"><b>本次编译在这一步发出了 ' + hints.lines + ' 条性能提示</b>'
        + (hints.occurrences > hints.lines ? '（合计 ' + hints.occurrences + ' 处）' : '')
        + ' ' + codes + '</p>'
        + '<table class="ptx-table ptx-table--hints"><thead><tr>'
        + '<th>源码位置</th><th>处</th><th>提示</th></tr></thead><tbody>';
      hints.sites.forEach(function (h) {
        html += '<tr><td class="ptx-hintat"><code title="' + esc(h.at) + '">'
          + esc(tailPath(h.at)) + '</code></td>'
          + '<td class="ptx-num">' + h.occurrences + '</td>'
          + '<td class="ptx-hintmsg">' + esc(h.message) + '</td></tr>';
      });
      html += '</tbody></table>'
        + '<p class="ptx-more">提示原文来自本次编译的 <code>report/perf_hints.log</code>，'
        + '源码路径已去掉编译机的绝对前缀。</p>';
    } else if (tier.rank >= 2) {
      html += '<p class="ptx-note ptx-note--quiet">本次编译这一步<b>没有</b>发出性能提示。'
        + '这不等于它没做取舍——'
        + (pf.tier === 'decide'
          ? '它有专属提示码，没报就是每个判定都按请求满足了。'
          : '这一档本来就没有自检，成形得好不好不会有任何信号。')
        + '</p>';
    }

    return html + '</section>';
  }

  /** Show the tail of a source path; the full one stays in the title. */
  function tailPath(at) {
    var parts = String(at).split('/');
    return parts.length <= 2 ? at : parts.slice(-2).join('/');
  }

  /**
   * Hints that no Pass produced — in these runs the post-pipeline
   * TileInnermostDimGranularity check, which is 197 of 230 lines. It is filed on
   * the frontend snapshot because it is a fact about the run, not about any one
   * Pass, and hanging it off a Pass page would misattribute it.
   */
  function runPerfCard() {
    var rp = run().perf;
    if (!rp || !rp.other || !rp.other.length) return '';
    var html = '';
    rp.other.forEach(function (g) {
      var codes = Object.keys(g.codes || {}).map(function (c) {
        return '<code>' + esc(c) + '</code>×' + g.codes[c];
      }).join(' ');
      html += '<section class="ptx-card ptx-perf ptx-perf--other">'
        + '<h3>本次编译的性能提示：' + esc(g.emitter)
        + '<span class="ptx-perfbasis">不是 Pass</span></h3>'
        + '<p class="ptx-card__headline">' + g.lines + ' 条提示'
        + (g.occurrences > g.lines ? '（合计 ' + g.occurrences + ' 处）' : '') + ' ' + codes
        + '。它来自流水线跑完之后的校验器，不是任何 Pass 的改写结果——'
        + '沿时间线找不到"是哪一步引起的"，因为答案是<b>源码里的 tile 形状</b>。</p>'
        + '<table class="ptx-table ptx-table--hints"><thead><tr>'
        + '<th>源码位置</th><th>处</th><th>提示</th></tr></thead><tbody>';
      g.sites.forEach(function (h) {
        html += '<tr><td class="ptx-hintat"><code title="' + esc(h.at) + '">'
          + esc(tailPath(h.at)) + '</code></td>'
          + '<td class="ptx-num">' + h.occurrences + '</td>'
          + '<td class="ptx-hintmsg">' + esc(h.message) + '</td></tr>';
      });
      html += '</tbody></table>';
      if (g.moreSites) {
        html += '<p class="ptx-more">另有 ' + g.moreSites + ' 个源码位置，见 <code>' + esc(rp.log) + '</code></p>';
      }
      html += '</section>';
    });
    return html;
  }

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

    html += perfCard(p);

    // Deriving where each body went needs both snapshots parsed, which is
    // async, so the slot is reserved now and filled when the parse lands.
    var wantMig = p.idx > 0 && p.changedFunctions && p.changedFunctions.some(function (f) {
      return f.status === 'added' || f.status === 'removed';
    });
    var migSlot = '<section class="ptx-card ptx-mig" id="migCard">'
      + '<h3>函数体去了哪里</h3><p class="ptx-loading">正在解析前后快照…</p></section>';
    // It goes directly after the 函数结构 card: the figure elaborates on that
    // census and the reader compares the two counts against each other, so
    // they must be adjacent rather than a scroll apart.
    var migPending = wantMig;

    if (!p.evidence.length) {
      html += p.idx === 0
        ? '<p class="ptx-note">这是 Pass 流水线的输入快照。切到「结构图」可以先看清算子本身的结构，再沿时间线逐个 Pass 往下走。</p>'
        : '<p class="ptx-note ptx-note--noop">本 Pass 在这个算子上是空操作：逐行比对前后快照完全一致。这本身是有用的结论——它说明该 Pass 的触发条件没有在这段 IR 上命中。</p>';
    }

    p.evidence.forEach(function (card) {
      html += '<section class="ptx-card ' + (TONE_CLASS[card.tone] || '') + '">'
        + '<h3>' + esc(card.title) + '</h3>'
        + '<p class="ptx-card__headline">' + mdInline(card.headline) + '</p>';

      var visual = cardVisual(card);
      if (visual) {
        html += visual;
      } else if (card.rows && card.rows.length) {
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
      if (migPending && card.id === 'functions') { html += migSlot; migPending = false; }
    });
    // No 函数结构 card on this step (a Pass can add a function without the card
    // firing); fall back to placing it after the evidence rather than dropping it.
    if (migPending) { html += migSlot; migPending = false; }

    if (p.idx === 0) html += runPerfCard();

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

    if ($('migCard')) {
      var token = ++renderOverview._token;
      Promise.all([analyze(state.runId, p.idx - 1), analyze(state.runId, p.idx)])
        .then(function (pair) {
          if (token !== renderOverview._token || !$('migCard')) return;
          var mg = LIB.moveGraph(pair[0], pair[1]);
          if (!mg) { $('migCard').remove(); return; }
          var D = MOVE_DIR[mg.direction];
          $('migCard').innerHTML = '<h3>函数体去了哪里'
            + '<span class="mig-verb mv-' + mg.direction + '">' + esc(D ? D.verb : '套壳') + '</span></h3>'
            + drawMigration(mg, state.fn);
        })
        .catch(function (err) {
          if (token !== renderOverview._token || !$('migCard')) return;
          $('migCard').innerHTML = '<h3>函数体去了哪里</h3><p class="ptx-empty">' + esc(err.message) + '</p>';
        });
    }
  }

  // Must start at a number: `++undefined` is NaN, and `NaN !== NaN` would make
  // every async fill think it had been superseded.
  renderOverview._token = 0;

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
      state.jumpLine = null;
      $('diffBody').innerHTML = '<p class="ptx-empty">该函数在本 Pass 中未发生变化。</p>';
      return;
    }

    var total = LIB.countChanges(rows);
    var head = '<div class="ptx-diffhead"><code>' + esc(name) + '</code>'
      + '<span class="ptx-add">+' + total.add + '</span><span class="ptx-del">−' + total.del + '</span>'
      + '<span class="ptx-muted">' + hunks.length + ' 处变更 · 前 ' + srcA.length + ' 行 / 后 ' + srcB.length + ' 行</span></div>';

    $('diffBody').innerHTML = head
      + (state.diffMode === 'split' ? splitView(hunks) : unifiedView(hunks));
    mountSplitScroller();

    if (state.jumpLine) {
      // Both gutters are rendered per row, before first. Matching on the number
      // alone let an after-file target land on an unrelated before-file row
      // carrying the same number, which is how a host click at step 09 jumped
      // 4434 cells away from anything relevant.
      var col = state.jumpSide === 'before' ? 0 : 1;
      var target = null;
      var trs = $('diffBody').querySelectorAll('tbody tr');
      for (var i = 0; i < trs.length; i++) {
        var ln = trs[i].querySelectorAll('.ptx-ln')[col];
        if (ln && Number(ln.textContent) === state.jumpLine) { target = trs[i]; break; }
      }
      if (target) {
        target.classList.add('is-jump');
        revealIn($('diffBody'), target, true);
      } else {
        toast((state.jumpSide === 'before' ? 'Pass 前' : 'Pass 后') + '第 ' + state.jumpLine
          + ' 行不在本 Pass 的变更范围内');
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
      var pfn = after.byName.has(state.fn) ? state.fn
        : (after.functions[0] && after.functions[0].name);
      gateLenses(before ? before.byName.get(pfn) : null, after.byName.get(pfn), pfn);
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

    gateLenses(fa, fb, state.fn);

    // Stepping along the timeline with one lens held is the normal way to read
    // this, so a lens the reader chose is remembered while it is unavailable
    // and restored the moment it can say something again.
    if (state.lensWanted && state.lensWanted !== state.lens
      && lensGate(state.lensWanted, fa, fb, state.fn).ok
      && $('lensPicker').querySelector('[data-lens="' + state.lensWanted + '"]')) {
      state.lens = state.lensWanted;
      lens = LENSES.find(function (l) { return l.id === state.lens; });
    }

    // A lens can land on nothing — memory lenses before InitMemRef, the task
    // DAG in a compute kernel. Fall back to op migration, which always has
    // either a migration to show or a definite "nothing moved" to state.
    var gate = lensGate(state.lens, fa, fb, state.fn);
    if (!gate.ok) {
      if (!state.lensAuto) toast('「' + lens.label + '」' + gate.why + '，已切到「算子迁移」');
      state.lens = 'opshift';
      lens = LENSES.find(function (l) { return l.id === state.lens; });
    }
    $('lensPicker').querySelectorAll('button').forEach(function (b) {
      b.classList.toggle('is-active', b.dataset.lens === state.lens);
    });

    // any change of lens, pass or function ends playback
    stopFlow();

    if (state.lens === 'pass') {
      var pview = PASS_VIEWS[p.name];
      if (pview && pview.draw === 'outline') { drawOutline(before, after, state.fn, p.name); return; }
      if (pview && pview.draw === 'lowering') { drawLowering(fa, fb, p.name); return; }
    }

    if (state.lens === 'opshift') { drawOpShift(fa, fb, lens.hint); return; }
    if (state.lens === 'spacetime') { drawSpacetime(fa, fb, lens.hint); return; }
    if (state.lens === 'memory') { drawMemory(fa, fb, lens.hint); return; }

    if (state.lens === 'control') { drawControl(fa, fb, lens.hint); return; }

    var build = state.lens === 'dataflow' ? LIB.dataflowGraph : LIB.taskGraph;
    var sig = state.lens === 'dataflow'
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
        n.rank = r;
      });
    });

    return {
      width: 48 + layers.length * (NW + GX),
      height: 48 + maxRows * (NH + GY),
      cyclic: seen < nodes.length,
    };
  }

  // ── before / after playback ───────────────────────────────────────────

  /**
   * The union graph is laid out once, so a node never moves between the two
   * sides — which means the change can be played back in place. Each stage is
   * one category of the diff, and everything else dims, so "what this Pass did
   * to the dataflow" arrives as a sequence rather than four colours at once.
   */
  var FLOW_STAGES = [
    { key: 'before', label: '之前' },
    { key: 'drop', label: '退场' },
    { key: 'change', label: '改写' },
    { key: 'born', label: '新增' },
    { key: 'after', label: '之后' },
  ];

  function flowNote(i, c) {
    if (i === 0) return 'Pass 执行前 · ' + (c.same + c.del + c.chg) + ' 个值';
    if (i === 1) return c.del + ' 个值不再产生，连同 ' + c.edgeDel + ' 条依赖一起退场';
    if (i === 2) return c.chg + ' 个值换了算子 / 类型 / 内存空间';
    if (i === 3) return c.add + ' 个值是这一步新引入的，带来 ' + c.edgeAdd + ' 条依赖';
    return 'Pass 执行后 · ' + (c.same + c.add + c.chg) + ' 个值';
  }

  var flowTimer = null;

  function stopFlow() {
    if (flowTimer) { clearInterval(flowTimer); flowTimer = null; }
    var b = $('flowPlay');
    if (b) { b.textContent = '▶'; b.classList.remove('is-on'); }
  }

  function mountFlowPlayer(counts) {
    var bar = $('flowPlayer');
    var canvas = $('canvas');
    if (!bar || !canvas) return;
    var scrub = $('flowScrub');
    var stage = $('flowStage');

    function show(i) {
      i = Math.max(0, Math.min(FLOW_STAGES.length - 1, i));
      canvas.dataset.phase = String(i);
      scrub.value = String(i);
      stage.innerHTML = '<b>' + esc(FLOW_STAGES[i].label) + '</b>'
        + '<span class="ptx-flow__note">' + esc(flowNote(i, counts)) + '</span>';
    }

    stopFlow();
    show(FLOW_STAGES.length - 1);

    $('flowPlay').addEventListener('click', function () {
      if (flowTimer) { stopFlow(); return; }
      var i = Number(canvas.dataset.phase) >= FLOW_STAGES.length - 1 ? -1 : Number(canvas.dataset.phase);
      $('flowPlay').textContent = '❚❚';
      $('flowPlay').classList.add('is-on');
      flowTimer = setInterval(function () {
        i++;
        show(i);
        if (i >= FLOW_STAGES.length - 1) stopFlow();
      }, 1100);
      show(Math.max(0, i));
    });

    scrub.addEventListener('input', function () {
      stopFlow();
      show(Number(scrub.value));
    });
  }

  // ── orthogonal edge routing ───────────────────────────────────────────

  /**
   * A straight curve between two layers is fine; one that spans three or more
   * cuts through whatever sits in between. Layered layout leaves empty
   * corridors between the layers, so route there: out of the source, along a
   * corridor, across a horizontal lane chosen to miss every box it passes, and
   * into the target.
   */
  function freeLane(nodes, ranks, y0, y1, pad) {
    var spans = nodes
      .filter(function (n) { return ranks.indexOf(n.rank) >= 0; })
      .map(function (n) { return [n.y - pad, n.y + n.h + pad]; })
      .sort(function (a, b) { return a[0] - b[0]; });
    if (!spans.length) return (y0 + y1) / 2;

    var merged = [spans[0].slice()];
    spans.slice(1).forEach(function (s) {
      var last = merged[merged.length - 1];
      if (s[0] <= last[1]) last[1] = Math.max(last[1], s[1]);
      else merged.push(s.slice());
    });

    var mid = (y0 + y1) / 2;
    var inside = merged.some(function (m) { return mid >= m[0] && mid <= m[1]; });
    if (!inside) return mid;

    // nearest gap between two occupied bands, else just above or below them
    var best = null;
    for (var i = 0; i < merged.length - 1; i++) {
      var lane = (merged[i][1] + merged[i + 1][0]) / 2;
      if (merged[i + 1][0] - merged[i][1] < 8) continue;
      if (best === null || Math.abs(lane - mid) < Math.abs(best - mid)) best = lane;
    }
    if (best !== null) return best;
    var above = merged[0][0] - 14;
    var below = merged[merged.length - 1][1] + 14;
    return Math.abs(above - mid) < Math.abs(below - mid) ? above : below;
  }

  /** Corner-rounded polyline. Zero-length segments are dropped first so a
   *  degenerate corner cannot produce a NaN arc. */
  function roundedPath(pts, r) {
    var p = [];
    pts.forEach(function (q) {
      var last = p[p.length - 1];
      if (!last || Math.abs(last[0] - q[0]) > 0.5 || Math.abs(last[1] - q[1]) > 0.5) p.push(q);
    });
    if (p.length < 2) return '';
    var d = 'M' + p[0][0].toFixed(1) + ',' + p[0][1].toFixed(1);
    for (var i = 1; i < p.length - 1; i++) {
      var a = p[i - 1], b = p[i], c = p[i + 1];
      var d0 = Math.hypot(b[0] - a[0], b[1] - a[1]);
      var d1 = Math.hypot(c[0] - b[0], c[1] - b[1]);
      var rr = Math.min(r, d0 / 2, d1 / 2);
      var s = [b[0] + (a[0] - b[0]) / d0 * rr, b[1] + (a[1] - b[1]) / d0 * rr];
      var e = [b[0] + (c[0] - b[0]) / d1 * rr, b[1] + (c[1] - b[1]) / d1 * rr];
      d += ' L' + s[0].toFixed(1) + ',' + s[1].toFixed(1)
        + ' Q' + b[0].toFixed(1) + ',' + b[1].toFixed(1)
        + ' ' + e[0].toFixed(1) + ',' + e[1].toFixed(1);
    }
    var z = p[p.length - 1];
    return d + ' L' + z[0].toFixed(1) + ',' + z[1].toFixed(1);
  }

  /**
   * Points for one edge. Parallel edges in the same corridor get a small
   * deterministic offset so they stay countable instead of merging into one
   * thick line.
   */
  function routeEdge(a, b, nodes, slot) {
    var x0 = a.x + a.w, y0 = a.y + a.h / 2;
    var x1 = b.x, y1 = b.y + b.h / 2;
    var jitter = (slot % 5 - 2) * 5;

    // backward edge: leave right, travel in a lane clear of every box, re-enter left
    if (x1 <= x0) {
      var top = Math.min.apply(null, nodes.map(function (n) { return n.y; }));
      var lane = Math.max(6, top - 18 - (slot % 4) * 7);
      return [[x0, y0], [x0 + 18, y0], [x0 + 18, lane], [x1 - 18, lane], [x1 - 18, y1], [x1, y1]];
    }

    var gap = x1 - x0;
    if (b.rank - a.rank <= 1 || gap < 40) {
      var cx = x0 + gap / 2 + jitter;
      return [[x0, y0], [cx, y0], [cx, y1], [x1, y1]];
    }

    // spans intermediate layers — cross them on a lane that misses their boxes
    var mids = [];
    for (var r = a.rank + 1; r < b.rank; r++) mids.push(r);
    var lane2 = freeLane(nodes, mids, y0, y1, 7) + jitter;
    var cxA = x0 + Math.min(26, gap * 0.18) + jitter;
    var cxB = x1 - Math.min(26, gap * 0.18) + jitter;
    return [[x0, y0], [cxA, y0], [cxA, lane2], [cxB, lane2], [cxB, y1], [x1, y1]];
  }

  var STATUS_LABEL = { add: '新增', del: '删除', chg: '属性改变', same: '未变' };

  // ── control flow: a nesting tree, not a dependency graph ──────────────

  var CTL = {
    fn: { glyph: "ƒ", label: "函数" },
    "for": { glyph: "⟳", label: "循环" },
    "if": { glyph: "◆", label: "分支" },
    "else": { glyph: "◇", label: "否则" },
    "with": { glyph: "⌷", label: "作用域" },
    block: { glyph: "▪", label: "语句块" },
  };

  /** `range(128)` / `pipeline(2)` / `range(t_dyn)` -> kind and trip count. */
  function loopParts(label) {
    var m = /^([A-Za-z_]\w*)\((.*)\)$/.exec(String(label));
    if (!m) return { kind: String(label), trip: null, dynamic: true };
    var raw = m[2];
    var n = /^\d+$/.test(raw) ? Number(raw) : null;
    return { kind: m[1], trip: n, dynamic: n === null, rawTrip: raw };
  }

  /**
   * Control flow is containment: a loop holds its body, a branch holds two
   * alternatives, and how deep a statement sits — and how many times its
   * enclosing loops run — is the whole point. A left-to-right dependency
   * layout says none of that, which is why this used to be indistinguishable
   * from the dataflow view. An indented tree says it directly, and the
   * cumulative trip count on each row is something no DAG can show.
   */
  function drawControl(fa, fb, hint) {
    var GA = fa ? LIB.controlTree(fa) : { nodes: [], edges: [] };
    var GB = fb ? LIB.controlTree(fb) : { nodes: [], edges: [] };
    if (!GA.nodes.length && !GB.nodes.length) {
      $("graphBody").innerHTML = "<p class=\"ptx-empty\">这个函数没有控制结构。</p>";
      return;
    }

    var sig = function (n) { return n.label + "|" + n.detail; };
    var A = new Map(GA.nodes.map(function (n) { return [n.id, n]; }));
    var B = new Map(GB.nodes.map(function (n) { return [n.id, n]; }));
    var info = new Map();
    [].concat(GA.nodes, GB.nodes).forEach(function (n) {
      if (info.has(n.id)) return;
      var a = A.get(n.id), b = B.get(n.id);
      info.set(n.id, Object.assign({}, b || a, {
        status: !a ? "add" : !b ? "del" : (sig(a) !== sig(b) ? "chg" : "same"),
        before: a || null,
      }));
    });

    // children in the order the "after" tree lists them, then anything only
    // the "before" tree had, so deleted branches still appear in place
    var kids = new Map();
    function addKid(p, c) {
      if (!kids.has(p)) kids.set(p, []);
      if (kids.get(p).indexOf(c) < 0) kids.get(p).push(c);
    }
    GB.edges.forEach(function (e) { addKid(e.from, e.to); });
    GA.edges.forEach(function (e) { addKid(e.from, e.to); });

    var root = GB.root || GA.root;
    var rows = [];
    var counts = { add: 0, del: 0, chg: 0, same: 0 };
    var maxStmt = 1;

    (function walk(id, depth, mult, dyn) {
      var n = info.get(id);
      if (!n || rows.length > 400) return;
      var m = mult, d = dyn;
      if (n.type === "for") {
        var lp = loopParts(n.label);
        n.loop = lp;
        if (lp.dynamic) d = true; else m = mult * lp.trip;
      }
      n.depth = depth;
      n.mult = mult;
      n.dynMult = dyn;
      if (n.type === "block" && n.weight > maxStmt) maxStmt = n.weight;
      counts[n.status]++;
      rows.push(n);
      (kids.get(id) || []).forEach(function (c) { walk(c, depth + 1, m, d); });
    })(root, 0, 1, false);

    var maxDepth = rows.reduce(function (s, r) { return Math.max(s, r.depth); }, 0);

    var body = rows.map(function (n) {
      var meta = CTL[n.type] || CTL.block;
      var guides = "";
      for (var i = 0; i < n.depth; i++) guides += "<span class=\"ctl-guide\"></span>";

      var head, badge = "", bar = "";
      if (n.type === "for") {
        var lp = n.loop;
        head = "<b class=\"ctl-kind\">" + esc(lp.kind) + "</b>"
          + "<span class=\"ctl-trip\">" + (lp.dynamic ? "×动态" : "×" + fmt(lp.trip)) + "</span>";
        if (lp.dynamic) badge = "<span class=\"ctl-dyn\">" + esc(clip(shortName(lp.rawTrip), 22)) + "</span>";
      } else if (n.type === "if" || n.type === "else") {
        head = "<b class=\"ctl-kind\">" + (n.type === "if" ? "if" : "else") + "</b>"
          + (n.detail ? "<code class=\"ctl-cond\">" + esc(clip(n.detail, 46)) + "</code>" : "");
      } else if (n.type === "fn") {
        head = "<b class=\"ctl-kind\">" + esc(shortName(n.label)) + "</b>"
          + "<span class=\"ctl-meta\">" + esc(n.detail || "") + "</span>";
      } else {
        head = "<span class=\"ctl-stmts\">" + n.weight + " 语句</span>"
          + (n.detail ? "<span class=\"ctl-ops\">" + esc(clip(n.detail, 40)) + "</span>" : "");
        bar = "<span class=\"ctl-bar\" style=\"width:" + Math.max(2, n.weight / maxStmt * 52) + "px\"></span>";
      }

      // what the enclosing loops multiply this row by — the number a
      // dependency graph can never show
      var run = n.type === "block" && n.mult > 1
        ? "<span class=\"ctl-run\" title=\"外层静态循环的累计倍数\">×" + fmt(n.mult)
          + (n.dynMult ? "<span class=\"ctl-dyn\">·动态</span>" : "") + "</span>"
        : "";

      return "<div class=\"ctl-row is-" + n.status + " ct-" + n.type + "\""
        + (n.line ? " data-line=\"" + n.line + "\"" : "") + ">"
        + guides
        + "<span class=\"ctl-glyph\">" + meta.glyph + "</span>"
        + "<span class=\"ctl-head\">" + head + badge + "</span>"
        + bar + run
        + "<span class=\"ctl-stat\">" + (n.status === "same" ? "" : STATUS_LABEL[n.status]) + "</span>"
        + "</div>";
    }).join("");

    var loops = rows.filter(function (n) { return n.type === "for"; });
    var deepest = rows.filter(function (n) { return n.type === "block"; })
      .reduce(function (s, n) { return Math.max(s, n.mult); }, 1);

    var summary = "<div class=\"ptx-graphsummary\">" + esc(hint)
      + "<span class=\"ptx-graphstats\">"
      + loops.length + " 个循环 · 最深嵌套 " + maxDepth
      + (deepest > 1 ? " · 最内层执行 <b>×" + fmt(deepest) + "</b>" : "")
      + " · <b class=\"ptx-add\">+" + counts.add + "</b> <b class=\"ptx-del\">−" + counts.del + "</b>"
      + " · 属性改变 " + counts.chg
      + "</span></div>";

    $("graphBody").innerHTML = summary + "<div class=\"ctl-tree\">" + body + "</div>";
  }

  function drawGraph(g, opts) {
    if (!g.nodes.length) {
      $('graphBody').innerHTML = '<p class="ptx-empty">这个视角下没有可显示的节点。</p>';
      return;
    }
    var box = layout(g.nodes, g.edges);
    var byId = new Map(g.nodes.map(function (n) { return [n.id, n]; }));

    // In a def-use graph the edges are the data, so the fat ones are the
    // expensive dependencies. Scale by the bytes the source value carries.
    var weigh = opts.kind === 'dataflow';
    var maxBytes = 0;
    if (weigh) {
      g.nodes.forEach(function (n) {
        n.bytes = valueBytes(n);
        if (n.bytes > maxBytes) maxBytes = n.bytes;
      });
    }
    var slots = {};
    var edgeSvg = g.edges.map(function (e) {
      var a = byId.get(e.from);
      var b = byId.get(e.to);
      if (!a || !b) return '';
      var key = a.rank + '>' + b.rank;
      slots[key] = (slots[key] || 0) + 1;
      var w = '';
      if (weigh && a.bytes && maxBytes > 0) {
        w = ' style="stroke-width:' + (0.8 + 2.6 * Math.log1p(a.bytes) / Math.log1p(maxBytes)).toFixed(2) + 'px"';
      }
      var d = roundedPath(routeEdge(a, b, g.nodes, slots[key]), 9);
      if (!d) return '';
      return '<path class="ptx-edge ptx-edge--' + e.status + '"' + w + ' d="' + d + '"/>';
    }).join('');

    var flow = opts.kind === 'dataflow';
    var nodeSvg = g.nodes.map(function (n) {
      var sub = nodeSubtitle(n, opts.kind);
      var head = flow ? nodeHeadline(n) : n.label;
      var src = flow && n.type === 'param';
      var cls = 'ptx-node ptx-node--' + n.status + ' nt-' + (n.type || 'x')
        + (flow && n.space ? ' sp-' + n.space : '')
        + (flow && n.type === 'value' ? ' oc-' + opCategory(n.rawOp || '') : '');
      var tx = src ? 16 : 12;
      var was = flow && n.status === 'chg' && n.before
        ? { head: nodeHeadline(n.before), sub: nodeSubtitle(n.before, opts.kind) }
        : null;
      var text = function (extra, h, u) {
        return '<text class="ptx-node__label' + extra + '" x="' + tx + '" y="17">' + esc(trunc(h, 22))
          + '</text><text class="ptx-node__sub' + extra + '" x="' + tx + '" y="31">'
          + esc(trunc(u, 26)) + '</text>';
      };
      return '<g class="' + cls + '" transform="translate(' + n.x + ',' + n.y + ')"'
        + ' data-line="' + (n.line || '') + '" data-id="' + esc(n.id) + '" tabindex="0">'
        + '<rect width="' + n.w + '" height="' + n.h + '" rx="' + (src ? n.h / 2 : 5) + '"/>'
        + (src ? '' : '<rect class="ptx-node__spine" width="3" height="' + n.h + '"/>')
        + (was ? text(' is-was', was.head, was.sub) : '')
        + text(was ? ' is-now' : '', head, sub)
        + '<title>' + esc((flow ? head + '\n' + shortName(n.label) : n.label) + '\n' + sub
          + (was ? '\n之前：' + was.head + ' ' + was.sub : '') + '\n'
          + STATUS_LABEL[n.status] + (n.line ? '\n源行 ' + n.line : '')) + '</title>'
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

    var player = flow && (c.add || c.del || c.chg)
      ? '<div class="ptx-flow" id="flowPlayer">'
        + '<button class="ptx-flow__play" id="flowPlay" title="播放这一步对数据流的改动">▶</button>'
        + '<input type="range" class="ptx-flow__scrub" id="flowScrub" min="0" max="'
        + (FLOW_STAGES.length - 1) + '" step="1" value="' + (FLOW_STAGES.length - 1)
        + '" aria-label="Pass 前后阶段">'
        + '<span class="ptx-flow__stage" id="flowStage"></span></div>'
      : '';

    $('graphBody').innerHTML = summary + player
      + '<div class="ptx-canvas" id="canvas">'
      + '<svg width="100%" height="100%"><g id="viewport">' + edgeSvg + nodeSvg + '</g></svg>'
      + '<div class="ptx-canvas__tools">'
      + '<button id="graphFit" title="缩放到能看见整张图">适应</button>'
      + '<button id="graphReset" title="回到 1:1">1:1</button>'
      + '<span class="ptx-canvas__zoom" id="graphZoom"></span></div></div>';

    enablePanZoom($('canvas'), box);
    if (player) mountFlowPlayer(c);
  }

  // ── lens availability ─────────────────────────────────────────────────

  /**
   * What each gated lens is waiting for. The index carries per-pass, per-
   * function counts, so the tooltip can name the step that brings the thing
   * into existence instead of just saying there is nothing to show.
   */
  var LENS_NEEDS = {
    control: { field: 'loops', noun: '循环', exact: true },
    task: { field: 'tasks', noun: '任务', exact: true },
    memory: { field: 'allocs', noun: '缓冲', exact: false },
    spacetime: { field: 'allocs', noun: '缓冲', exact: false },
  };

  /** First pass where any function has the thing — a fact about the pipeline
   *  rather than about one function. */
  function firstPassAnyWith(field) {
    var ps = run().passes;
    for (var i = 0; i < ps.length; i++) {
      var fns = ps[i].functions || [];
      for (var j = 0; j < fns.length; j++) {
        if (fns[j][field] > 0) return ps[i];
      }
    }
    return null;
  }

  /** First pass at or after `from` where this function has the thing. */
  function firstPassWith(fnName, field, from) {
    var ps = run().passes;
    for (var i = 0; i < ps.length; i++) {
      var p = ps[i];
      if (p.idx < from) continue;
      var fns = p.functions || [];
      for (var j = 0; j < fns.length; j++) {
        if (fns[j].name === fnName && fns[j][field] > 0) return p;
      }
    }
    return null;
  }

  /**
   * Whether a lens can say anything at this step, and if not, why. Availability
   * is decided on the loaded snapshots; the "comes back at step N" part is
   * advisory and read from the index.
   */
  function lensGate(id, fa, fb, fnName) {
    if (lensHasContent(id, fa, fb)) return { ok: true };
    var need = LENS_NEEDS[id];
    if (!need) return { ok: true };

    var why = '这一步还没有' + need.noun;
    var later = fnName ? firstPassWith(fnName, need.field, state.passIdx + 1) : null;
    if (later) {
      why += '：要到第 ' + String(later.idx).padStart(2, '0') + ' 步 ' + later.name + ' 之后才有';
    } else if (need.exact && fnName) {
      why += '：' + shortName(fnName) + ' 在整条流水线上都没有';
    } else {
      var born = firstPassAnyWith(need.field);
      if (born) {
        why += '：' + need.noun + '由第 ' + String(born.idx).padStart(2, '0') + ' 步 '
          + born.name + ' 引入';
      }
    }
    return { ok: false, why: why };
  }

  /** Reflect availability on the lens strip once the snapshots are in. */
  function gateLenses(fa, fb, fnName) {
    var strip = $('lensPicker');
    if (!strip) return;
    if (!fa && !fb) {
      strip.querySelectorAll('button').forEach(function (b) {
        b.setAttribute('aria-disabled', 'false');
        b.classList.remove('is-off');
        if (b.dataset.hint) b.title = b.dataset.hint;
      });
      return;
    }
    strip.querySelectorAll('button').forEach(function (b) {
      var g = lensGate(b.dataset.lens, fa, fb, fnName);
      b.setAttribute('aria-disabled', g.ok ? 'false' : 'true');
      b.classList.toggle('is-off', !g.ok);
      if (!g.ok) {
        b.dataset.hint = b.dataset.hint || b.title;
        b.title = b.textContent.replace('推荐', '').trim() + ' · ' + g.why;
      } else if (b.dataset.hint) {
        b.title = b.dataset.hint;
      }
    });
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

  /** Static byte volume a value carries, 0 when any dimension is dynamic. */
  function valueBytes(n) {
    if (!n || !n.shape || !n.dtype) return 0;
    var e = LIB.shapeElems(n.shape);
    return e ? e * LIB.dtypeBytes(n.dtype) : 0;
  }

  /** What a dataflow node is: a parameter is a name, a value is its operator. */
  function nodeHeadline(n) {
    if (n.type === 'param') return shortName(n.label);
    return n.op && n.op !== 'expr' ? n.op : '=';
  }

  function nodeSubtitle(n, kind) {
    if (kind === 'call') return [n.kind, n.level, n.stmts != null ? n.stmts + ' 语句' : ''].filter(Boolean).join(' · ');
    if (kind === 'control') return n.detail || (n.weight != null ? n.weight + ' 语句' : '');
    if (kind === 'dataflow') {
      var shape = n.shape ? '[' + n.shape.map(function (d) {
        return typeof d === 'number' ? d : shortName(String(d));
      }).join('×') + ']' : '';
      if (n.type === 'param') return [n.op, shape, n.dtype].filter(Boolean).join(' ');
      return [shortName(n.label), shape, n.dtype].filter(Boolean).join(' ');
    }
    return [n.type, n.level, n.weight ? n.weight + ' 语句' : ''].filter(Boolean).join(' · ');
  }

  function enablePanZoom(host, box) {
    var svg = host.querySelector('svg');
    var vp = host.querySelector('#viewport');
    var scale = 1;
    var tx = 0;
    var ty = 0;
    var dragging = false;
    var sx = 0;
    var sy = 0;

    var PAD = 18;
    // Wheel zoom stops at MIN so a scroll cannot shrink the graph to dust, but
    // an explicit "fit" may go further: the dataflow graph is ~19000px wide and
    // at MIN it still overflows, which makes a button labelled 适应 a lie.
    var MIN = 0.12, MAX = 2.6, FIT_MIN = 0.03;
    // The wheel floor follows the fit: after fitting a 19000px graph to 4%, a
    // floor of 12% would make the very next wheel tick jump the view 3x.
    var minScale = MIN;

    // Clamp against where the ink actually is, not the nominal canvas: the
    // layout leaves margin inside `box`, so keeping 80px of the BOX on screen
    // could leave only 43px of real content.
    var ink = null;
    function inkBox() {
      if (ink) return ink;
      try {
        var b = vp.getBBox();
        if (b.width && b.height) ink = { x: b.x, y: b.y, width: b.width, height: b.height };
      } catch (e) { /* not laid out yet */ }
      return ink || (box ? { x: 0, y: 0, width: box.width, height: box.height } : null);
    }

    function apply() {
      // Keep a slab of content on screen whatever the drag did. Without this a
      // flick can leave an empty canvas and no way back except the buttons.
      var ib = inkBox();
      if (ib) {
        var cw = host.clientWidth, ch = host.clientHeight;
        var keep = 80;
        // Screen position of the ink is tx + ib.x * scale, so solve for tx.
        tx = Math.min(cw - keep - ib.x * scale, Math.max(keep - (ib.x + ib.width) * scale, tx));
        ty = Math.min(ch - keep - ib.y * scale, Math.max(keep - (ib.y + ib.height) * scale, ty));
      }
      vp.setAttribute('transform', 'translate(' + tx + ',' + ty + ') scale(' + scale + ')');
      var z = $('graphZoom');
      if (z) z.textContent = Math.round(scale * 100) + '%';
    }

    /** Scale so the whole graph is visible, and centre it. */
    function fit() {
      if (!box || !box.width || !box.height) return;
      var cw = host.clientWidth, ch = host.clientHeight;
      if (!cw || !ch) return;
      var fb = inkBox() || { width: box.width, height: box.height };
      scale = Math.min(MAX, Math.max(FIT_MIN, Math.min((cw - PAD * 2) / fb.width, (ch - PAD * 2) / fb.height)));
      minScale = Math.min(MIN, scale);
      // Never blow a small graph up past 1:1 - that just makes it blurry-looking
      // and loses the sense of how small it is.
      scale = Math.min(scale, 1);
      var ib = inkBox() || { x: 0, y: 0, width: box.width, height: box.height };
      // Centre the ink, not the padded canvas.
      tx = (cw - ib.width * scale) / 2 - ib.x * scale;
      ty = (ch - ib.height * scale) / 2 - ib.y * scale;
      apply();
    }

    var fitBtn = $('graphFit');
    if (fitBtn) fitBtn.addEventListener('click', fit);
    var resetBtn = $('graphReset');
    if (resetBtn) {
      resetBtn.addEventListener('click', function () {
        scale = 1; tx = PAD; ty = PAD; apply();
      });
    }
    // Open fitted: a graph wider than the pane otherwise shows only its corner.
    //
    // The container can still be 0x0 here - the pane may be collapsed, or layout
    // may not have settled - and fitting against no size does nothing silently.
    // So watch it and fit on the first real size, until the reader takes over.
    var touched = false;
    ['wheel', 'pointerdown'].forEach(function (ev) {
      host.addEventListener(ev, function () { touched = true; }, { passive: true });
    });
    fit();
    if (typeof ResizeObserver !== 'undefined') {
      var ro = new ResizeObserver(function () {
        if (touched || !host.isConnected) return;
        if (host.clientWidth && host.clientHeight) fit();
      });
      ro.observe(host);
    }

    host.addEventListener('wheel', function (e) {
      e.preventDefault();
      var rect = svg.getBoundingClientRect();
      var mx = e.clientX - rect.left;
      var my = e.clientY - rect.top;
      var k = e.deltaY < 0 ? 1.12 : 1 / 1.12;
      var next = Math.min(MAX, Math.max(minScale, scale * k));
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
      state.jumpSide = 'after';   // the lenses are drawn from the after snapshot
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

  /** Byte volume of a value, or null when any dimension is still symbolic. */
  function staticBytes(w) {
    if (!w.shape || !w.shape.length) return null;
    var n = 1;
    for (var i = 0; i < w.shape.length; i++) {
      if (typeof w.shape[i] !== "number") return null;
      n *= w.shape[i];
    }
    var e = LIB.dtypeBytes(w.dtype);
    return e ? n * e : null;
  }

  var CROSS = {
    carry: { label: "进 + 出", cls: "carry", why: "循环把新值带回下一轮迭代" },
    inout: { label: "原地改写", cls: "inout", why: "直接写回调用方的缓冲" },
    out: { label: "只出", cls: "out", why: "函数新产出的结果" },
    shadow: { label: "改了没出去", cls: "shadow", why: "区域内更新了它，但签名上没有出口" },
    "in": { label: "只进", cls: "in", why: "只读输入" },
  };
  var CROSS_ORDER = ["carry", "inout", "out", "shadow", "in"];

  /**
   * Base names the body re-defines. The IR is still functional here, so an
   * update shows up as a new SSA version of the same base name rather than a
   * mutation — which is how a parameter can be declared `in`, be updated
   * inside, and still have nowhere to go at the boundary.
   */
  function bodyRedefs(f) {
    var set = {};
    (f.src || []).forEach(function (line) {
      var m = line.match(/^\s+([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*(?::|=)/);
      if (!m) return;
      m[1].split(",").forEach(function (t) { set[shortName(t.trim())] = 1; });
    });
    return set;
  }

  /**
   * What crosses the cut, and in which direction. Two sources have to be
   * merged: the pl.Out / pl.InOut wrappers on the parameters, and the trailing
   * return. A loop-carried accumulator arrives as a plain `in` parameter and
   * leaves through the return — flash attention's l / m / o are exactly this —
   * so neither source on its own tells the truth.
   */
  function crossings(f) {
    var isRet = {};
    returnNames(f).forEach(function (r) { isRet[r] = 1; });
    var redef = bodyRedefs(f);
    var seen = {};
    var ws = (f.params || []).map(function (p) {
      seen[p.name] = 1;
      var dir = p.dir === "out" || p.dir === "inout" ? p.dir
        : isRet[p.name] ? "carry"
          : p.ctor && p.ctor !== "Scalar" && redef[shortName(p.name)] ? "shadow"
            : "in";
      var w = { name: p.name, dir: dir, ctor: p.ctor, shape: p.shape, dtype: p.dtype };
      w.bytes = staticBytes(w);
      return w;
    });
    Object.keys(isRet).forEach(function (r) {
      if (!seen[r]) ws.push({ name: r, dir: "out", ctor: null, shape: null, dtype: null, bytes: null });
    });
    ws.sort(function (x, y) {
      var d = CROSS_ORDER.indexOf(x.dir) - CROSS_ORDER.indexOf(y.dir);
      return d || (y.bytes || 0) - (x.bytes || 0);
    });
    return ws;
  }

  function symDim(d) {
    if (typeof d === "number") return String(d);
    return d ? shortName(String(d)) : "?";
  }

  function typeStr(w) {
    if (!w.ctor) return "—";
    var sh = (w.shape || []).map(symDim);
    return w.ctor.replace(/^pld\./, "") + (sh.length ? "[" + sh.join("×") + "]" : "")
      + (w.dtype ? " " + w.dtype : "");
  }

  function clip(t, n) { return t.length > n ? t.slice(0, n - 1) + "…" : t; }

  var CJK = /[\u2e80-\u9fff\u3000-\u303f\uff00-\uffef]/;

  /** Rough rendered width of a label: CJK takes about one em, Latin ~0.55. */
  function estWidth(t, px) {
    var w = 0;
    for (var i = 0; i < t.length; i++) w += CJK.test(t[i]) ? px : px * 0.55;
    return w;
  }

  /** Truncate to what actually fits in `max` px at `px` font size. */
  function fitText(t, px, max) {
    if (estWidth(t, px) <= max) return t;
    var out = "";
    for (var i = 0; i < t.length; i++) {
      if (estWidth(out + t[i] + "…", px) > max) break;
      out += t[i];
    }
    return out + "…";
  }

  /**
   * Outlining is a cut. The picture that explains it is the cut itself: every
   * value the region used to reference freely now has to cross a named
   * boundary, in a definite direction. How many wires there are, how wide they
   * are, and which ones come back — that is what differs between callables. A
   * generic "before box / after box" pair never does.
   */
  /**
   * Split view puts two code columns side by side, and code must not wrap, so
   * neither column can simply scroll the pane — that would push the right half
   * off screen. Both halves share one offset instead, driven by a single
   * scrollbar under the diff, which also keeps the two sides on the same
   * column as you read across.
   */
  function mountSplitScroller() {
    var bar = $('diffHScroll');
    var tbl = $('diffBody').querySelector('.ptx-difftable--split');
    if (!tbl) {
      bar.hidden = true;
      return;
    }
    var widest = 0;
    tbl.querySelectorAll('td.ptx-code pre').forEach(function (p) {
      if (p.scrollWidth > widest) widest = p.scrollWidth;
    });
    var cell = tbl.querySelector('td.ptx-code');
    var visible = cell ? cell.clientWidth : 0;
    tbl.style.setProperty('--ptx-hoff', '0px');
    if (!visible || widest <= visible) {
      bar.hidden = true;
      return;
    }
    bar.hidden = false;
    bar.firstElementChild.style.width = widest + 'px';
    bar.scrollLeft = 0;
  }

  function drawOutline(before, after, fnName, passName) {
    var f = after.byName.get(fnName);
    if (!f) {
      $("graphBody").innerHTML = "<p class=\"ptx-empty\">本 Pass 之后没有这个函数。</p>";
      return;
    }

    var ws = crossings(f);
    var cnt = { carry: 0, inout: 0, out: 0, shadow: 0, "in": 0 };
    var inBytes = 0, dynCount = 0;
    ws.forEach(function (w) {
      cnt[w.dir]++;
      if (w.bytes) { if (w.dir !== "out") inBytes += w.bytes; }
      else if (w.ctor && w.ctor !== "Scalar") dynCount++;
    });

    var host = after.functions.filter(function (x) {
      return (x.calls || []).some(function (c) { return c.callee === fnName; });
    })[0];
    var callLine = host && (host.calls.find(function (c) { return c.callee === fnName; }) || {}).line;
    var born = !(before && before.byName.get(fnName));

    // what actually got lifted out
    var mix = { move: 0, compute: 0, view: 0 };
    var tally = {};
    opSequence(f).forEach(function (op) {
      var c = opCategory(op);
      if (c === "meta") return;
      mix[c]++;
      var t = opTail(op) || op;
      tally[t] = (tally[t] || 0) + 1;
    });
    var mixN = mix.move + mix.compute + mix.view;
    var topOps = Object.keys(tally).sort(function (x, y) { return tally[y] - tally[x]; }).slice(0, 4);

    // The pane this lands in is a ~370px column, so the drawing is laid out at
    // the width it actually gets. Drawing wide and letting the browser scale
    // the SVG down is what made the earlier version unreadable: at 0.44x, 12px
    // labels render at 5px.
    var host0 = $("graphBody");
    var W = Math.max(320, Math.min(900, (host0 ? host0.clientWidth : 380) - 32));
    var pad = 0, spineX = W - 10, wireX = Math.max(W - 96, W * 0.68);
    var textW = wireX - 22;
    var nameMax = Math.floor(textW / 6.8), typeMax = Math.floor(textW / 5.9);

    var MAX = 14;
    var shown = ws.slice(0, ws.length > MAX ? MAX - 1 : MAX);
    var hidden = ws.length - shown.length;

    var hostY = 18, hostH = 48;
    var y0 = hostY + hostH + 38, rowH = 34;
    var rowsEnd = y0 + Math.max(shown.length - 1, 0) * rowH + (hidden ? 24 : 0);
    var fnY = rowsEnd + 44, fnH = 82;
    var H = fnY + fnH + 6;

    var svg = "<svg class=\"ptx-passview\" viewBox=\"0 0 " + W + " " + H + "\" width=\"" + W
      + "\" height=\"" + H + "\" role=\"img\" "
      + "aria-label=\"" + esc(fnName) + " 的外提边界\">"
      + "<defs><marker id=\"pvArrow\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" markerWidth=\"4.5\" "
      + "markerHeight=\"4.5\" orient=\"auto-start-reverse\"><path d=\"M2 2L8 5L2 8Z\" "
      + "fill=\"context-stroke\" stroke=\"none\"/></marker></defs>";

    // before — the host still holds everything
    svg += "<text class=\"pv-h\" x=\"" + pad + "\" y=\"11\">之前 · 这段区域在宿主函数体内，随手引用上下文</text>"
      + "<rect class=\"pv-host\" x=\"" + pad + "\" y=\"" + hostY + "\" width=\"" + (W - pad * 2)
      + "\" height=\"" + hostH + "\" rx=\"7\"/>"
      + "<text class=\"pv-t\" x=\"" + (pad + 12) + "\" y=\"" + (hostY + 19) + "\">"
      + esc(fitText(shortName(host ? host.name : "宿主函数"), 12, W - pad * 2 - 24)) + "</text>"
      + "<text class=\"pv-s\" x=\"" + (pad + 12) + "\" y=\"" + (hostY + 36) + "\">"
      + (callLine ? "第 " + callLine + " 行：如今只剩一次调用" : "调用方") + "</text>";

    // the cut
    svg += "<text class=\"pv-h\" x=\"" + pad + "\" y=\"" + (y0 - 16) + "\">外提后必须点名的值 "
      + ws.length + " 个</text>"
      + "<text class=\"pv-cut\" x=\"" + (spineX - 4) + "\" y=\"" + (y0 - 16)
      + "\" text-anchor=\"end\">切口</text>"
      + "<line class=\"pv-cutline\" x1=\"" + spineX + "\" y1=\"" + (y0 - 12) + "\" x2=\"" + spineX
      + "\" y2=\"" + (fnY + 2) + "\"/>";

    // one row per value that has to cross
    shown.forEach(function (w, i) {
      var y = y0 + i * rowH;
      var meta = CROSS[w.dir];
      var ty = typeStr(w) + (w.bytes ? " · " + bytes(w.bytes) : "");
      var back = w.dir === "carry" || w.dir === "inout";
      svg += "<g class=\"pv-wire is-" + meta.cls + "\">"
        + "<title>" + esc(shortName(w.name) + " · " + meta.label + "：" + meta.why + "\n" + ty) + "</title>"
        + "<rect class=\"pv-wmark\" x=\"" + pad + "\" y=\"" + (y - 6) + "\" width=\"6\" height=\"12\" rx=\"2\"/>"
        + "<text class=\"pv-t pv-wname\" x=\"" + (pad + 14) + "\" y=\"" + (y - 3) + "\">"
        + esc(fitText(shortName(w.name), 12, textW)) + "</text>"
        + "<text class=\"pv-s pv-wtype\" x=\"" + (pad + 14) + "\" y=\"" + (y + 12) + "\">"
        + esc(fitText(ty, 11, textW)) + "</text>"
        + "<line class=\"pv-w\" x1=\"" + wireX + "\" y1=\"" + (y + (back ? -3 : 2)) + "\" x2=\""
        + (spineX - 3) + "\" y2=\"" + (y + (back ? -3 : 2)) + "\" marker-end=\"url(#pvArrow)\"/>"
        + (back
          ? "<line class=\"pv-w\" x1=\"" + (spineX - 3) + "\" y1=\"" + (y + 7) + "\" x2=\"" + wireX
            + "\" y2=\"" + (y + 7) + "\" marker-end=\"url(#pvArrow)\"/>"
          : "")
        + (w.dir === "shadow"
          ? "<line class=\"pv-wback\" x1=\"" + (spineX - 3) + "\" y1=\"" + (y + 9) + "\" x2=\""
            + ((wireX + spineX) / 2) + "\" y2=\"" + (y + 9) + "\" marker-end=\"url(#pvArrow)\"/>"
          : "")
        + "</g>";
    });
    if (hidden) {
      svg += "<text class=\"pv-s\" x=\"" + (pad + 14) + "\" y=\"" + (rowsEnd + 6)
        + "\">…还有 " + hidden + " 个，见下表</text>";
    }

    // after — the standalone function everything now flows into
    svg += "<text class=\"pv-h\" x=\"" + pad + "\" y=\"" + (fnY - 8) + "\">之后 · 独立函数，边界上什么都得写明白</text>"
      + "<rect class=\"pv-fn\" x=\"" + pad + "\" y=\"" + fnY + "\" width=\"" + (W - pad * 2)
      + "\" height=\"" + fnH + "\" rx=\"7\"/>"
      + "<text class=\"pv-t\" x=\"" + (pad + 12) + "\" y=\"" + (fnY + 20) + "\">def "
      + esc(fitText("def " + shortName(fnName) + "(…)", 12, W - pad * 2 - 24).replace(/^def /, "")) + "</text>"
      + "<text class=\"pv-s\" x=\"" + (pad + 12) + "\" y=\"" + (fnY + 37) + "\">"
      + esc([f.kind, f.level, f.role].filter(Boolean).join(" · "))
      + (born ? " · 本 Pass 新建" : "") + "</text>"
      + "<text class=\"pv-s\" x=\"" + (pad + 12) + "\" y=\"" + (fnY + 53) + "\">"
      + esc(fitText(f.stmtCount + " 条语句 · " + (f.loops || []).length + " 个循环 · 最深嵌套 "
        + (f.maxNest || 0)
        + (mixN ? "　｜　" + ["move", "compute", "view"].filter(function (c) { return mix[c]; })
          .map(function (c) { return CAT_LABEL[c] + " " + mix[c]; }).join(" · ") : ""),
        11, W - pad * 2 - 24)) + "</text>";

    if (mixN) {
      var barY = fnY + 60, barW = W - pad * 2 - 24, bx = pad + 12;
      ["move", "compute", "view"].forEach(function (c) {
        if (!mix[c]) return;
        var bw = mix[c] / mixN * barW;
        svg += "<rect class=\"pv-seg is-" + c + "\" x=\"" + bx + "\" y=\"" + barY + "\" width=\"" + bw
          + "\" height=\"8\" rx=\"2\"><title>" + esc(CAT_LABEL[c] + " " + mix[c] + " 个算子")
          + (topOps.length ? "\n" + topOps.map(function (t) { return t + "×" + tally[t]; }).join(" ") : "")
          + "</title></rect>";
        bx += bw;
      });
    }

    svg += "</svg>";

    // the one sentence that differs between callables
    var verdict;
    if (cnt.carry) {
      verdict = "<b>" + cnt.carry + " 个值既进又出</b>：循环每轮把新值带回下一轮，外提时它们必须原样穿过边界，"
        + "否则累加状态会在调用处断掉。";
    } else if (cnt.inout && !cnt.out) {
      verdict = "<b>没有返回值</b>：" + cnt.inout + " 个缓冲被原地改写，调用方拿到的是副作用而不是结果。";
    } else if (cnt.out || cnt.inout) {
      verdict = "产出 <b>" + (cnt.out + cnt.inout) + " 个结果</b>，其余 " + (ws.length - cnt.out - cnt.inout)
        + " 个入参在区域内不回传。";
    } else if (cnt.shadow) {
      verdict = "<b>签名上没有出口</b>：区域内更新了 " + cnt.shadow
        + " 个入参的新版本，但参数没有 Out / InOut 标记、调用处也没有接收返回值 ——"
        + "这一步的效果只能靠内存副作用传出去。";
    } else {
      verdict = "<b>全部只读</b>：这段区域不改动调用方的任何数据，边界只需单向传入。";
    }

    var chips = CROSS_ORDER.filter(function (d) { return cnt[d]; }).map(function (d) {
      return "<span class=\"pv-chip is-" + CROSS[d].cls + "\">" + esc(CROSS[d].label) + " " + cnt[d]
        + "</span>";
    }).join("");

    var head = "<div class=\"ptx-graphsummary\">" + verdict + "<span class=\"ptx-graphstats\">" + chips
      + (inBytes ? " · 入口静态体积 <b>" + bytes(inBytes) + "</b>" : "")
      + (dynCount ? " · " + dynCount + " 个动态形状" : "") + "</span></div>";

    var tbl = "<table class=\"ptx-table ptx-table--fns\"><thead><tr><th>跨边界的值</th><th>类型</th>"
      + "<th>体积</th><th>方向</th></tr></thead><tbody>";
    ws.slice(0, 60).forEach(function (w) {
      var meta = CROSS[w.dir];
      tbl += "<tr><td><code>" + esc(shortName(w.name)) + "</code></td>"
        + "<td><span class=\"ptx-muted\">" + esc(typeStr(w)) + "</span></td>"
        + "<td>" + (w.bytes ? bytes(w.bytes) : "<span class=\"ptx-muted\">动态</span>") + "</td>"
        + "<td><span class=\"pv-chip is-" + meta.cls + "\">" + esc(meta.label) + "</span> "
        + "<span class=\"ptx-muted\">" + esc(meta.why) + "</span></td></tr>";
    });
    tbl += "</tbody></table>";

    $("graphBody").innerHTML = head + "<div class=\"ptx-passview__wrap\">" + svg + "</div>" + tbl;
  }


  // ── lowering: implicit movement becomes explicit ──────────────────────

  /** `pl.tensor.exp` and `pl.tile.exp` share the key `exp`, so the diff can
   *  align them as one operator that changed domain rather than as an
   *  unrelated delete plus insert. */
  function normOp(op) { return opTail(op) || String(op).replace(/^pl\./, ""); }

  var MEM_LABEL = {
    Vec: "Vec 向量计算区", Mat: "Mat 矩阵计算区", Acc: "Acc 累加器",
    L1: "L1 片上缓存", L0A: "L0A", L0B: "L0B", L0C: "L0C", GM: "GM 全局内存",
  };
  function memLabel(m) { return MEM_LABEL[m] || m; }

  /**
   * Explicit traffic in the tile-domain body. `tile.load` carries its
   * destination in `target_memory=`, and `tile.store` always lands back in the
   * tensor domain, so its destination is GM. This is the movement that tensor
   * semantics used to leave implicit.
   */
  function tileTraffic(f) {
    var into = {}, store = 0, alloc = {};
    (f && f.src ? f.src : []).forEach(function (line) {
      var hits = line.match(/pl\.tile\.([a-z_0-9]+)\(/g);
      if (!hits) return;
      var tm = line.match(/target_memory=pl\.Mem\.(\w+)/);
      var lhs = line.match(/:\s*pl\.Tile\[[^\]]*\]\s*,\s*pl\.\w+\s*,\s*pl\.Mem\.(\w+)/)
        || line.match(/pl\.Mem\.(\w+)/);
      var dest = (tm && tm[1]) || (lhs && lhs[1]) || null;
      var usedTm = false;
      hits.forEach(function (x) {
        var op = x.slice(8, -1);
        if (op === "store") { store++; return; }
        if (op === "load" || op === "copy") {
          var d = !usedTm && dest ? dest : "未标注";
          usedTm = true;
          into[d] = (into[d] || 0) + 1;
          return;
        }
        if (op === "create" || op === "create_l1" || op === "full") {
          var a = dest || "未标注";
          alloc[a] = (alloc[a] || 0) + 1;
        }
      });
    });
    return { into: into, store: store, alloc: alloc };
  }

  /**
   * Lowering places values. In the tensor domain a value has a shape and a
   * dtype and nothing else; in the tile domain it must live somewhere
   * specific, and the movement that gets it there has to be written out. The
   * placement is the pass's real output — and it also says what kind of kernel
   * this is: all-Vec is a vector op, Mat plus Acc is a matmul, all three is
   * mixed. An aligned barcode of the two operator streams showed none of that,
   * and at this pane's width its cells were 5px wide.
   */
  function drawLowering(fa, fb, passName) {
    if (!fb) {
      $("graphBody").innerHTML = "<p class=\"ptx-empty\">这一步之后没有这个函数。</p>";
      return;
    }
    var before = (fa && fa.memSpaces) || {};
    var after = fb.memSpaces || {};
    var kindsA = (fa && fa.valueKinds) || {};
    var kindsB = fb.valueKinds || {};
    var tiers = Object.keys(after).sort(function (x, y) { return after[y] - after[x]; });

    if (!tiers.length) {
      $("graphBody").innerHTML = "<p class=\"ptx-empty\">这一步之后它还没有任何片上内存落位，"
        + "下降发生在别的 Pass。</p>";
      return;
    }

    var traffic = tileTraffic(fb);
    var placedB = tiers.reduce(function (s, k) { return s + after[k]; }, 0);
    var placedA = Object.keys(before).reduce(function (s, k) { return s + before[k]; }, 0);
    var stillTensor = kindsB.Tensor || 0;
    var moved = placedB - placedA;

    var host0 = $("graphBody");
    var W = Math.max(320, Math.min(900, (host0 ? host0.clientWidth : 380) - 32));
    var pad = 0, retX = pad + 6, busX = pad + 30, tierX = pad + 56;
    var tierW = W - tierX - pad;
    var maxN = Math.max.apply(null, tiers.map(function (t) { return after[t]; }));

    var gmY = 18, gmH = 34, tierH = 42, step = tierH + 10;
    var tY0 = gmY + gmH + 30;
    var bottom = tY0 + (tiers.length - 1) * step + tierH;
    var H = bottom + 28;

    var svg = "<svg class=\"ptx-passview ptx-lower\" viewBox=\"0 0 " + W + " " + H + "\" width=\"" + W
      + "\" height=\"" + H + "\" role=\"img\" "
      + "aria-label=\"值被放进哪块片上内存\">"
      + "<defs><marker id=\"lwArrow\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" markerWidth=\"5\" "
      + "markerHeight=\"5\" orient=\"auto-start-reverse\"><path d=\"M2 2L8 5L2 8Z\" "
      + "fill=\"context-stroke\" stroke=\"none\"/></marker></defs>";

    // what stays outside: parameters and globals keep living in the tensor domain
    svg += "<text class=\"pv-h\" x=\"" + pad + "\" y=\"11\">函数外面 · 还是 tensor，没有片上位置</text>"
      + "<rect class=\"lw-gm\" x=\"" + pad + "\" y=\"" + gmY + "\" width=\"" + (W - pad * 2)
      + "\" height=\"" + gmH + "\" rx=\"6\"/>"
      + "<text class=\"pv-t\" x=\"" + (pad + 12) + "\" y=\"" + (gmY + 22) + "\">"
      + esc(fitText("GM · 参数与全局 " + stillTensor + " 个值", 12, W - pad * 2 - 24)) + "</text>";

    // the bus every load and store rides
    svg += "<line class=\"lw-bus\" x1=\"" + busX + "\" y1=\"" + (gmY + gmH) + "\" x2=\"" + busX
      + "\" y2=\"" + (bottom - 8) + "\"/>";

    if (traffic.store) {
      svg += "<path class=\"lw-store\" d=\"M" + retX + " " + (bottom + 12) + " V" + (gmY + gmH + 4)
        + "\" fill=\"none\" marker-end=\"url(#lwArrow)\"/>"
        + "<line class=\"lw-store\" x1=\"" + retX + "\" y1=\"" + (bottom + 12) + "\" x2=\"" + tierX
        + "\" y2=\"" + (bottom + 12) + "\"/>"
        + "<text class=\"pv-s lw-storelab\" x=\"" + (tierX + 6) + "\" y=\"" + (bottom + 16)
        + "\">store ×" + traffic.store + " 写回 GM</text>";
    }

    tiers.forEach(function (t, i) {
      var y = tY0 + i * step;
      var mid = y + tierH / 2;
      var inN = traffic.into[t] || 0;
      var alN = traffic.alloc[t] || 0;
      var barMax = Math.max(24, tierW * 0.34);
      var bw = Math.max(3, after[t] / maxN * barMax);
      var barX = tierX + tierW - 12 - bw;

      svg += "<line class=\"lw-branch\" x1=\"" + busX + "\" y1=\"" + mid + "\" x2=\"" + (tierX - 4)
        + "\" y2=\"" + mid + "\" marker-end=\"url(#lwArrow)\"/>";
      svg += "<g class=\"lw-tier is-" + esc(t) + "\">"
        + "<title>" + esc(memLabel(t) + "：" + after[t] + " 个值"
          + (inN ? "\nload ×" + inN : "") + (alN ? "\n就地开辟 ×" + alN : "")) + "</title>"
        + "<rect class=\"lw-box\" x=\"" + tierX + "\" y=\"" + y + "\" width=\"" + tierW
        + "\" height=\"" + tierH + "\" rx=\"6\"/>"
        + "<text class=\"pv-t lw-name\" x=\"" + (tierX + 12) + "\" y=\"" + (y + 17) + "\">"
        + esc(fitText(memLabel(t), 12, barX - tierX - 24)) + "</text>"
        + "<text class=\"pv-s lw-count\" x=\"" + (tierX + 12) + "\" y=\"" + (y + 33) + "\">"
        + esc(fitText(after[t] + " 个值" + (alN ? " · 就地开辟 " + alN : ""), 11,
          barX - tierX - (inN ? 76 : 24))) + "</text>"
        + "<rect class=\"lw-bar\" x=\"" + barX + "\" y=\"" + (y + 9)
        + "\" width=\"" + bw + "\" height=\"9\" rx=\"3\"/>"
        + (inN
          ? "<text class=\"pv-s lw-loadlab\" x=\"" + (tierX + tierW - 12) + "\" y=\"" + (y + 33)
            + "\" text-anchor=\"end\">load ×" + inN + "</text>"
          : "")
        + "</g>";
    });

    svg += "</svg>";

    // the one sentence that differs between callables
    var hasVec = after.Vec > 0, hasMat = after.Mat > 0, hasAcc = after.Acc > 0;
    var shape = hasMat && hasAcc && hasVec ? "三块内存都用上了 —— 矩阵和向量混在一个核里"
      : hasMat && hasAcc ? "操作数进矩阵区、结果落累加器 —— 典型的 matmul"
        : hasVec && !hasMat ? "全部落在向量区 —— 这是个纯向量 kernel"
          : "落位分布见下";
    var verdict;
    if (!moved && placedA) {
      verdict = "<b>这一步没有改变它的落位</b>：它在更早的 Pass 就已经下降过，"
        + placedB + " 个值的位置原样保留。";
    } else {
      verdict = "<b>" + moved + " 个值拿到了片上位置</b>（落位 " + placedA + " → " + placedB
        + "）：tensor 值从 " + (kindsA.Tensor || 0) + " 个降到 " + (kindsB.Tensor || 0) + " 个。" + shape + "。";
    }

    var chips = tiers.map(function (t) {
      return "<span class=\"lw-chip is-" + esc(t) + "\">" + esc(t) + " " + after[t] + "</span>";
    }).join("");

    var head = "<div class=\"ptx-graphsummary\">" + verdict + "<span class=\"ptx-graphstats\">" + chips
      + " · 显式搬运 <b>" + (traffic.store
        + Object.keys(traffic.into).reduce(function (s, k) { return s + traffic.into[k]; }, 0))
      + "</b> 条</span></div>";

    // which movement instructions are genuinely new, rather than re-domained
    var hb = (fa && fa.opHist) || {};
    var ha = fb.opHist || {};
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
      ? "<p class=\"ptx-life__note\">tensor 域把搬运藏在语义里，tile 域必须写明——本步新出现的指令："
        + newMove.slice(0, 8).map(function (m) {
          return "<code>" + esc(m.op) + "</code>×" + m.n;
        }).join("、") + "</p>"
      : "<p class=\"ptx-life__note\">没有新的搬运指令：这一步只改了值的域与落位。</p>";

    var tbl = "<table class=\"ptx-table ptx-table--fns\"><thead><tr><th>片上内存</th><th>值</th>"
      + "<th>搬进来</th><th>就地开辟</th></tr></thead><tbody>";
    tiers.forEach(function (t) {
      tbl += "<tr><td><code>" + esc(t) + "</code> <span class=\"ptx-muted\">"
        + esc(memLabel(t).replace(/^\S+\s*/, "")) + "</span></td>"
        + "<td>" + after[t] + "</td>"
        + "<td>" + (traffic.into[t] ? "load ×" + traffic.into[t] : "<span class=\"ptx-muted\">—</span>") + "</td>"
        + "<td>" + (traffic.alloc[t] ? "×" + traffic.alloc[t] : "<span class=\"ptx-muted\">—</span>") + "</td></tr>";
    });
    tbl += "<tr><td><code>GM</code> <span class=\"ptx-muted\">全局</span></td><td>" + stillTensor
      + "</td><td><span class=\"ptx-muted\">—</span></td><td>"
      + (traffic.store ? "store ×" + traffic.store + " 写回" : "<span class=\"ptx-muted\">—</span>")
      + "</td></tr>";
    tbl += "</tbody></table>";

    $("graphBody").innerHTML = head + "<div class=\"ptx-passview__wrap\">" + svg + "</div>" + note + tbl;
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

  // Grouped by where the callable ends up on the device: the mixed kernels are
  // the interesting ones, pure AIC/AIV next, scaffolding last.
  var CALLABLE_GROUPS = ['混合核 AIC + AIV', 'Cube 核 AIC', 'Vector 核 AIV', '编排 / 其他'];
  var GROUP_HINT = {
    '混合核 AIC + AIV': '被 ExpandMixedKernel 拆成 Group 壳 + AIC + AIV 三个函数',
    'Cube 核 AIC': '纯矩阵计算，落在 Cube 核上',
    'Vector 核 AIV': '纯向量计算，落在 Vector 核上',
    '编排 / 其他': 'Orchestration、Graph、Inline 等非设备侧计算函数',
  };

  function groupOf(c) {
    if (c.split) return CALLABLE_GROUPS[0];
    var k = c.kinds[0];
    if (k === 'AIC') return CALLABLE_GROUPS[1];
    if (k === 'AIV') return CALLABLE_GROUPS[2];
    return CALLABLE_GROUPS[3];
  }

  function renderCallableRail() {
    var all = callables();
    var kernels = all.filter(function (c) { return c.isKernel; }).length;
    $('callableSummary').innerHTML = '<strong>' + all.length + '</strong> 个 callable'
      + '<span class="ptx-rail__sub">' + kernels + ' 个最终落到 AIC / AIV 核上</span>';

    var filter = state.cFilter.toLowerCase();
    var cur = currentCallable();
    var html = '';
    var shown = 0;
    var lastGroup = null;

    all.slice().sort(function (a, b) {
      var ga = CALLABLE_GROUPS.indexOf(groupOf(a));
      var gb = CALLABLE_GROUPS.indexOf(groupOf(b));
      return ga - gb || b.finalLines - a.finalLines || a.name.localeCompare(b.name);
    }).forEach(function (c) {
      if (state.onlyKernels && !c.isKernel) return;
      if (filter && c.name.toLowerCase().indexOf(filter) < 0) return;
      shown++;
      var g = groupOf(c);
      if (g !== lastGroup) {
        html += '<div class="ptx-phasehead" title="' + esc(GROUP_HINT[g] || '') + '">'
          + esc(g) + '</div>';
        lastGroup = g;
      }
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
    revealIn($('callableList'), active);
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

  /**
   * The journey as a river, after `Design/pass-atlas`: the whole pipeline is
   * drawn, not just the Passes that touched this callable. Seeing the ones
   * that did nothing — and the stretch before it even existed — is what puts
   * the ones that did in context. Cards could not show that.
   */
  function journeyBar(c, steps) {
    var tl = c.timeline;
    var n = tl.length;
    if (!n) return "<p class=\"ptx-empty\">没有 Pass 数据。</p>";

    var touched = {};
    steps.forEach(function (sg) { touched[sg.t.idx] = sg; });

    var padL = 104, padR = 26;
    var holder = $("viewCallable");
    var avail = holder ? holder.clientWidth - 28 : 0;
    var W = Math.max(720, n * 24 + padL + padR, avail);
    var bandY = 22, bandH = 56, spineY = bandY + 40, H = spineY + 46;
    function X(i) { return padL + (i + 0.5) / n * (W - padL - padR); }

    var birthIdx = c.birth ? tl.findIndex(function (t) { return t.idx === c.birth.idx; }) : -1;

    var svg = "<svg class=\"ptx-river\" viewBox=\"0 0 " + W + " " + H + "\" width=\"" + W
      + "\" height=\"" + H + "\" style=\"min-width:" + W + "px\" role=\"img\" aria-label=\"" + esc(c.name) + " 经历的 Pass 流水线\">";

    // Phase bands, by contiguous run rather than by first/last occurrence:
    // Simplify is filed under `lowering` but runs again at step 46, so a
    // min/max band would stretch across — and overlap — everything between.
    var runs = [];
    tl.forEach(function (t, i) {
      var last = runs[runs.length - 1];
      if (last && last.phase === t.phase) { last.end = i; return; }
      runs.push({ phase: t.phase, start: i, end: i });
    });
    runs.forEach(function (r) {
      var a = X(r.start) - 10;
      var b = X(r.end) + 10;
      svg += "<rect class=\"riv-band\" x=\"" + a + "\" y=\"" + bandY + "\" width=\"" + (b - a)
        + "\" height=\"" + bandH + "\" rx=\"9\"/>";
      var lab = phase(r.phase).label;
      if (b - a > lab.length * 9 + 12) {
        svg += "<text class=\"riv-bandlab\" x=\"" + ((a + b) / 2) + "\" y=\"" + (bandY + 14)
          + "\" text-anchor=\"middle\">" + esc(lab) + "</text>";
      }
    });
    // spine — dashed before the callable exists, solid once it does
    if (birthIdx > 0) {
      svg += "<line class=\"riv-spine is-void\" x1=\"" + padL + "\" y1=\"" + spineY
        + "\" x2=\"" + X(birthIdx) + "\" y2=\"" + spineY + "\"/>";
    }
    svg += "<line class=\"riv-spine\" x1=\"" + (birthIdx > 0 ? X(birthIdx) : padL) + "\" y1=\"" + spineY
      + "\" x2=\"" + (W - padR) + "\" y2=\"" + spineY + "\"/>";
    svg += "<text class=\"riv-axis\" x=\"" + (padL - 12) + "\" y=\"" + (spineY + 4)
      + "\" text-anchor=\"end\">执行序 →</text>";
    svg += "<text class=\"riv-axis\" x=\"" + (padL - 12) + "\" y=\"" + (bandY + 14)
      + "\" text-anchor=\"end\">阶段</text>";

    // milestones of THIS callable
    var miles = [];
    if (c.birth) miles.push({ idx: c.birth.idx, t: "诞生" });
    if (c.split) miles.push({ idx: c.split.passIdx, t: "拆为 AIC+AIV" });
    var firstMem = tl.find(function (t) { return t.allocs > 0; });
    if (firstMem) miles.push({ idx: firstMem.idx, t: "有 MemRef" });
    var mseen = {};
    miles.forEach(function (m) {
      if (mseen[m.idx]) return;
      mseen[m.idx] = 1;
      var i = tl.findIndex(function (t) { return t.idx === m.idx; });
      if (i < 0) return;
      var wd = m.t.length * 7 + 14;
      var lx = Math.max(padL, Math.min(W - padR - wd, X(i) - wd / 2));
      svg += "<rect class=\"riv-mile\" x=\"" + lx + "\" y=\"2\" width=\"" + wd
        + "\" height=\"16\" rx=\"8\"/>"
        + "<text class=\"riv-miletext\" x=\"" + (lx + wd / 2) + "\" y=\"14\" text-anchor=\"middle\">"
        + esc(m.t) + "</text>";
    });

    // nodes
    tl.forEach(function (t, i) {
      var sg = touched[t.idx];
      var alive = t.parts.length > 0;
      var cx = X(i);
      var active = t.idx === state.passIdx;

      if (!sg) {
        svg += "<g class=\"riv-node is-quiet\" data-step=\"" + t.idx + "\">"
          + "<title>" + esc(String(t.idx).padStart(2, "0") + " " + t.name + " · "
          + (alive ? "未改动它" : "它还不存在")) + "</title>"
          + "<line class=\"riv-tick" + (alive ? "" : " is-void") + "\" x1=\"" + cx + "\" y1=\"" + (spineY - 4)
          + "\" x2=\"" + cx + "\" y2=\"" + (spineY + 4) + "\"/>"
          + "<rect class=\"riv-hit\" x=\"" + (cx - 9) + "\" y=\"" + (spineY - 16)
          + "\" width=\"18\" height=\"32\"/></g>";
        return;
      }

      var primary = sg.deltas[0];
      var tone = sg.kindChange ? "split"
        : primary && primary.key === "allocs" ? (primary.v < 0 ? "save" : "mem")
        : primary && primary.key === "loops" ? "loop"
        : primary ? "stmt" : "inplace";
      var label = sg.kindChange ? "拆核"
        : primary ? primary.label + " " + signed(primary.v) : "就地改写";

      svg += "<g class=\"riv-node tone-" + tone + (active ? " is-active" : "") + "\" data-step=\"" + t.idx + "\">"
        + "<title>" + esc(String(t.idx).padStart(2, "0") + " " + t.name + " · " + label) + "</title>"
        + "<circle class=\"riv-glow\" cx=\"" + cx + "\" cy=\"" + spineY + "\" r=\"13\"/>"
        + (active ? "<circle class=\"riv-ring\" cx=\"" + cx + "\" cy=\"" + spineY + "\" r=\"10\"/>" : "")
        + "<circle class=\"riv-dot\" cx=\"" + cx + "\" cy=\"" + spineY + "\" r=\"6\"/>"
        + "<text class=\"riv-num\" x=\"" + cx + "\" y=\"" + (spineY + 24) + "\" text-anchor=\"middle\">"
        + String(t.idx).padStart(2, "0") + "</text>"
        + "<rect class=\"riv-hit\" x=\"" + (cx - 11) + "\" y=\"" + (spineY - 18)
        + "\" width=\"22\" height=\"44\"/></g>";
    });

    svg += "</svg>";

    var cur = touched[state.passIdx];
    var curLine = cur
      ? "<span class=\"riv-cur__idx\">" + String(cur.t.idx).padStart(2, "0") + "</span>"
        + "<b>" + esc(cur.t.name) + "</b>"
        + (cur.kindChange ? "<span class=\"riv-cur__kind\">" + esc(cur.kindChange.from) + " → "
            + esc(cur.kindChange.to) + "</span>" : "")
        + cur.deltas.slice(0, 3).map(function (d, i) {
            return "<span class=\"riv-cur__d " + (i === 0 ? "is-primary " : "")
              + (d.v > 0 ? "is-up" : "is-down") + "\">" + esc(d.label) + " " + signed(d.v) + "</span>";
          }).join("")
        + (cur.deltas.length ? "" : "<span class=\"riv-cur__d\">就地改写</span>")
      : "<span class=\"ptx-muted\">这一步没有改动它</span>";

    return "<div class=\"ptx-riverwrap\">" + svg + "</div>"
      + "<div class=\"riv-cur\">" + curLine + "</div>";
  }
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
    revealIn($('viewCallable'), active, true);
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
      var s = e.target.closest('.ptx-step, .riv-node');
      if (!s) return;
      var idx = Number(s.dataset.step);
      // Quiet nodes are Passes that did not touch this callable; selecting one
      // would snap back to the nearest Pass that did, which reads as a no-op.
      if (s.classList.contains('is-quiet')) { toast('这个 Pass 没有改动 ' + state.callable); return; }
      selectStep(idx);
    });

    $('onlyChanged').addEventListener('change', function (e) {
      state.onlyChanged = e.target.checked;
      renderRail();
    });
    $('passFilter').addEventListener('input', function (e) {
      state.filter = e.target.value.trim();
      renderRail();
    });
    // The legend is also the filter: clicking a tier isolates it, clicking the
    // active one clears. It never changes which Pass is selected, so the main
    // pane keeps showing the Pass you were reading even when the rail hides it.
    $('perfKey').addEventListener('click', function (e) {
      var b = e.target.closest('.ptx-perfkey__item');
      if (!b) return;
      state.perfFilter = state.perfFilter === b.dataset.tier ? null : b.dataset.tier;
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
      if (b) {
        state.fn = b.dataset.openfn;
        state.tab = 'diff';
        render();
        return;
      }
      // The migration figure is the answer to "which lines moved", so every
      // block in it opens that function's code rather than being a picture the
      // reader then has to go look up by hand.
      var m = e.target.closest('.mig-src[data-fn], .mig-host[data-fn], .ptx-table--wrap tr[data-fn]');
      if (!m) return;
      state.fn = m.dataset.fn;
      state.tab = 'diff';
      var line = Number(m.dataset.line || 0);
      state.jumpLine = line > 0 ? line : null;
      state.jumpSide = m.dataset.side === 'before' ? 'before' : 'after';
      render();
    });

    $('diffFnChips').addEventListener('click', function (e) {
      var b = e.target.closest('.ptx-chip');
      if (!b) return;
      state.fn = b.dataset.fn;
      renderDiff();
      writeHash();
    });

    function setDiffFullscreen(on) {
      var v = $('viewDiff');
      v.classList.toggle('is-fullscreen', on);
      var b = $('diffExpand');
      b.classList.toggle('is-on', on);
      b.textContent = on ? '退出全屏' : '全屏';
      b.title = on ? '返回分栏视图 (Esc)' : '全屏查看代码 Diff (F，Esc 退出)';
      mountSplitScroller();
    }

    $('diffHScroll').addEventListener('scroll', function () {
      var tbl = $('diffBody').querySelector('.ptx-difftable--split');
      if (tbl) tbl.style.setProperty('--ptx-hoff', $('diffHScroll').scrollLeft + 'px');
    });

    $('diffExpand').addEventListener('click', function () {
      setDiffFullscreen(!$('viewDiff').classList.contains('is-fullscreen'));
    });

    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && $('viewDiff').classList.contains('is-fullscreen')) {
        setDiffFullscreen(false);
        return;
      }
      // not while typing into the filter boxes
      var t = e.target;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA')) return;
      if ((e.key === 'f' || e.key === 'F') && !e.ctrlKey && !e.metaKey && !e.altKey
        && !$('viewDiff').hidden) {
        e.preventDefault();
        setDiffFullscreen(!$('viewDiff').classList.contains('is-fullscreen'));
      }
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
      if (b.getAttribute('aria-disabled') === 'true') {
        toast(b.title);
        return;
      }
      state.lens = b.dataset.lens;
      state.lensWanted = b.dataset.lens;
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
