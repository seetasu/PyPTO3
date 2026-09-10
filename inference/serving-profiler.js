/**
 * Observe profiler bridge
 *
 * 将 Toolkit Studio 的推理 profiling 数据与组件迁移到 Serving Observe。
 * 数据仍由 inference-profile-data.js 提供；这里负责目标页的浅色容器、
 * Service / Scheduler / Hardware Counter 摘要，以及算子分析页内交互。
 */
(function registerServingProfiler() {
  'use strict';

  var ctx = null;
  var GROUP_COLORS = {
    mlp: '#7f6bb3',
    attn: '#b06c3a',
    proj: '#356fae',
    norm: '#2f7a5a',
    boundary: '#8b929a',
    idle: '#a34d4d'
  };

  function esc(value) {
    return String(value == null ? '' : value).replace(/[&<>"']/g, function (c) {
      return ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c];
    });
  }

  function fmt(value, digits) {
    return value == null ? '—' : Number(value).toFixed(digits == null ? 2 : digits);
  }

  function int(value) {
    return value == null ? '—' : Number(value).toLocaleString('en-US');
  }

  function profile() {
    return window.PtoInferenceProfile && window.PtoInferenceProfile.get
      ? window.PtoInferenceProfile.get()
      : null;
  }

  var RUN_VARIANTS = {
    'RUN-042': { latency: .908, traffic: .96, tps: 1168, ttft: 166, batch: 14.9, sol: [4.0, 18.8, 67.2, 5.5], kv: .94, queue: .82 },
    'RUN-043': { latency: .836, traffic: .93, tps: 1238, ttft: 154, batch: 15.1, sol: [4.2, 19.7, 64.8, 5.3], kv: .91, queue: .74 },
    'RUN-044': { latency: .783, traffic: .89, tps: 1325, ttft: 149, batch: 15.3, sol: [4.6, 21.1, 61.3, 5.0], kv: .88, queue: .66 },
    'RUN-045': { latency: 1.013, traffic: 1.02, tps: 1053, ttft: 184, batch: 14.5, sol: [3.7, 17.9, 70.4, 5.1], kv: 1, queue: 1 },
    'RUN-046': { latency: 1.059, traffic: 1.04, tps: 998, ttft: 197, batch: 14.1, sol: [3.5, 16.8, 73.2, 5.5], kv: 1.05, queue: 1.18 }
  };

  function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
  function profileForRun(runId) {
    var base = profile();
    if (!base) return null;
    var v = RUN_VARIANTS[runId] || RUN_VARIANTS['RUN-045'];
    var p = JSON.parse(JSON.stringify(base));
    p.id = runId;
    p.title = runId + ' · Observe Profile';
    p.meta.capturedAt = runId;
    p.summary.tpot.p50 = Number((base.summary.tpot.p50 * v.latency).toFixed(2));
    p.summary.tpot.p90 = Number((base.summary.tpot.p90 * v.latency).toFixed(2));
    p.summary.tpot.p99 = Number((base.summary.tpot.p99 * v.latency).toFixed(2));
    p.summary.tps = v.tps;
    p.summary.ttft = v.ttft;
    p.summary.batchAvg = v.batch;
    p.summary.traffic.weights = Number((base.summary.traffic.weights * v.traffic).toFixed(3));
    p.summary.traffic.kv = Number((base.summary.traffic.kv * v.traffic * v.kv).toFixed(3));
    p.summary.traffic.act = Number((base.summary.traffic.act * v.traffic).toFixed(3));
    p.summary.traffic.total = Number((p.summary.traffic.weights + p.summary.traffic.kv + p.summary.traffic.act).toFixed(3));
    p.summary.sol.forEach(function (unit, index) { unit.pct = v.sol[index]; });
    p.summary.lowerBoundMs = Number((p.summary.traffic.total / p.meta.peakBw).toFixed(2));
    p.summary.efficiency = Number((p.summary.lowerBoundMs / p.summary.tpot.p50 * 100).toFixed(1));
    p.groups.forEach(function (group) { group.ms = Number((group.ms * v.latency).toFixed(4)); });
    p.ops.forEach(function (op) {
      op.totalMs = Number((op.totalMs * v.latency).toFixed(4));
      if (op.perLayerUs != null) op.perLayerUs = Number((op.perLayerUs * v.latency).toFixed(3));
      if (op.perLayer) op.perLayer = op.perLayer.map(function (value) { return Number((value * v.latency).toFixed(3)); });
      if (op.achievedBw) op.achievedBw = Number((op.achievedBw * v.traffic / v.latency).toFixed(3));
      if (op.efficiency != null) op.efficiency = Math.round(clamp(op.efficiency / v.latency, 1, 99));
      if (op.units && op.units.mte2 != null) op.units.mte2 = Number(clamp(op.units.mte2 * v.sol[2] / 70.4, 0, 100).toFixed(1));
    });
    p.memory.hbm.items.forEach(function (item) { if (item[0] === 'kv') item[2] = Number((item[2] * v.kv).toFixed(3)); if (item[0] === 'workspace') item[2] = Number((item[2] * v.traffic).toFixed(3)); });
    p.memory.onchip = p.memory.onchip.map(function (item, index) { var copy = item.slice(); copy[1] = Number(clamp(copy[1] * (1 + (v.sol[2] - 70.4) / 250 + index * .002), 1, 98).toFixed(1)); return copy; });
    var kv = p.memory.kv;
    kv.pagesUsed = Math.round(kv.pagesUsed * v.kv);
    kv.tokensLive = Math.round(kv.tokensLive * v.kv);
    kv.tokensAllocated = kv.pagesUsed * kv.pageTokens;
    kv.bytesAllocated = Number((kv.pagesUsed * kv.pageBytesMb / 1000).toFixed(3));
    kv.utilization = Number((kv.pagesUsed / kv.pagesTotal * 100).toFixed(1));
    kv.fragmentation = Number(clamp((kv.tokensAllocated - kv.tokensLive) / Math.max(1, kv.tokensAllocated) * 100, 0, 100).toFixed(2));
    kv.hitRate = Number(clamp(kv.hitRate + (1 - v.kv) * 7, 80, 99.9).toFixed(1));
    kv.blocksReal = Math.min(kv.blocksPadded, kv.pagesUsed);
    kv.density = Number((kv.blocksReal / Math.max(1, kv.blocksPadded) * 100).toFixed(1));
    p.summary.kvUsed = kv.bytesAllocated;
    p.summary.kvPct = kv.utilization;
    p.serving.batchAvg = v.batch;
    p.serving.batchOverTime = p.serving.batchOverTime.map(function (n, index) { return clamp(Math.round(n + (v.batch - base.serving.batchAvg) + ((index % 5) - 2) * .12), 1, p.meta.batch); });
    p.serving.queue.waitP50 = Math.round(p.serving.queue.waitP50 * v.queue);
    p.serving.queue.waitP99 = Math.round(p.serving.queue.waitP99 * v.queue);
    p.serving.queue.waiting = Math.max(0, Math.round(p.serving.queue.waiting * v.queue));
    p.serving.lanes.forEach(function (lane) { lane.items.forEach(function (item) { item.wait = Math.round(item.wait * v.queue); item.prefill = Math.round(item.prefill * v.latency); item.decode = Math.round(item.decode * v.latency); }); });
    p.serving.sweep.forEach(function (row) { row.tpot = Number((row.tpot * v.latency).toFixed(2)); row.tps = Math.round(row.tps / v.latency); row.traffic = Number((row.traffic * v.traffic).toFixed(2)); row.bw = Number((row.bw * v.traffic / v.latency).toFixed(2)); row.mte2 = Number(clamp(row.mte2 * v.sol[2] / 70.4, 0, 100).toFixed(1)); });
    return p;
  }

  function deltaInfo(current, baseline, direction) {
    if (current == null || baseline == null) return { status: 'changed', label: '已变化', delta: '—' };
    current = Number(current); baseline = Number(baseline);
    if (!Number.isFinite(current) || !Number.isFinite(baseline)) return { status: 'changed', label: '已变化', delta: '—' };
    var absolute = current - baseline;
    var relative = baseline === 0 ? null : absolute / Math.abs(baseline) * 100;
    if (Math.abs(absolute) < .000001) return { status: 'unchanged', label: '持平', delta: '+0' };
    if (relative !== null && Math.abs(relative) < 1) return { status: 'unchanged', label: '持平', delta: (absolute >= 0 ? '+' : '') + fmt(absolute, 2) };
    if (direction === 'neutral') return { status: Math.abs(absolute) < .000001 ? 'unchanged' : 'changed', label: Math.abs(absolute) < .000001 ? '持平' : '变化', delta: relative == null ? (absolute >= 0 ? '+' : '') + fmt(absolute, 2) : (relative >= 0 ? '+' : '') + fmt(relative, 1) + '%' };
    var better = direction === 'lower' ? absolute < 0 : absolute > 0;
    return { status: better ? 'improved' : 'regressed', label: better ? '改善' : '退化', delta: relative == null ? (absolute >= 0 ? '+' : '') + fmt(absolute, 2) : (relative >= 0 ? '+' : '') + fmt(relative, 1) + '%' };
  }

  function compareLine(current, baseline, formatted, direction, id) {
    if (baseline == null) return '<span class="so-profile-baseline">基线无数据</span>';
    var d = deltaInfo(current, baseline, direction || 'neutral');
    return '<span class="so-profile-baseline">基线 · ' + formatted + ' <em class="so-compare-state ' + d.status + '">' + d.label + ' ' + d.delta + '</em></span>';
  }

  function currentState() {
    return ctx && ctx.state && ctx.state.profiler ? ctx.state.profiler : {};
  }

  function rerender() {
    if (ctx && ctx.render) ctx.render();
  }

  function metric(label, value, detail, tone, comparison) {
    return '<article class="so-profile-kpi"><span>' + esc(label) + '</span><strong>' + value + '</strong><small class="' + (tone || '') + '">' + esc(detail) + '</small>' + (comparison || '') + '</article>';
  }

  function renderLayerSummary(p, b, baselineId) {
    var s = p.summary;
    var q = p.serving.queue;
    var kv = p.memory.kv;
    var bs = b && b.summary;
    var bq = b && b.serving.queue;
    var bkv = b && b.memory.kv;
    return '<div class="so-section-title"><h2>上层运行与硬件计数</h2><p>Service / Scheduler / Hardware runtime counter</p></div>'
      + '<div class="so-profile-layer-grid">'
      + '<section class="so-card so-profile-layer-card"><div class="so-card-head"><div><h3>Service</h3><div class="so-finding-summary">请求入口与 decode 服务状态</div></div><span class="so-status success">Completed</span></div><div class="so-card-body"><dl class="so-kv"><dt>吞吐</dt><dd>' + int(s.tps) + ' tok/s' + (bs ? compareLine(s.tps, bs.tps, int(bs.tps) + ' tok/s', 'higher', baselineId) : '') + '</dd><dt>TPOT p50 / p99</dt><dd>' + fmt(s.tpot.p50, 1) + ' / ' + fmt(s.tpot.p99, 1) + ' ms' + (bs ? compareLine(s.tpot.p50, bs.tpot.p50, fmt(bs.tpot.p50, 1) + ' / ' + fmt(bs.tpot.p99, 1) + ' ms', 'lower', baselineId) : '') + '</dd><dt>TTFT p50</dt><dd>' + int(s.ttft) + ' ms' + (bs ? compareLine(s.ttft, bs.ttft, int(bs.ttft) + ' ms', 'lower', baselineId) : '') + '</dd><dt>窗口</dt><dd>' + int(p.meta.steps) + ' steps · ' + fmt(p.meta.duration ? parseFloat(p.meta.duration) : 7.8, 1) + ' s</dd></dl></div></section>'
      + '<section class="so-card so-profile-layer-card"><div class="so-card-head"><div><h3>Scheduler</h3><div class="so-finding-summary">连续批处理与槽位复用</div></div><span class="so-pill accent">运行稳定</span></div><div class="so-card-body"><dl class="so-kv"><dt>运行中 / 等待中</dt><dd>' + q.running + ' / ' + q.waiting + (bq ? compareLine(q.waiting, bq.waiting, bq.running + ' / ' + bq.waiting, 'lower', baselineId) : '') + '</dd><dt>等待 p50 / p99</dt><dd>' + q.waitP50 + ' / ' + q.waitP99 + ' ms' + (bq ? compareLine(q.waitP50, bq.waitP50, bq.waitP50 + ' / ' + bq.waitP99 + ' ms', 'lower', baselineId) : '') + '</dd><dt>抢占 / 重计算</dt><dd>' + q.preempt + ' / ' + q.recompute + (bq ? compareLine(q.preempt, bq.preempt, bq.preempt + ' / ' + bq.recompute, 'lower', baselineId) : '') + '</dd><dt>Chunked Prefill</dt><dd>' + q.chunkedPrefill + ' 个窗口' + (bq ? compareLine(q.chunkedPrefill, bq.chunkedPrefill, bq.chunkedPrefill + ' 个窗口', 'neutral', baselineId) : '') + '</dd></dl></div></section>'
      + '<section class="so-card so-profile-layer-card"><div class="so-card-head"><div><h3>Hardware Runtime Counter</h3><div class="so-finding-summary">PMU 与带宽采集状态</div></div><span class="so-status success">已采集</span></div><div class="so-card-body"><dl class="so-kv"><dt>达成带宽</dt><dd>' + fmt(s.traffic.total / s.tpot.p50, 2) + ' TB/s · ' + fmt(s.traffic.total / s.tpot.p50 / p.meta.peakBw * 100, 1) + '% 峰值' + (bs ? compareLine(s.traffic.total / s.tpot.p50, bs.traffic.total / bs.tpot.p50, fmt(bs.traffic.total / bs.tpot.p50, 2) + ' TB/s', 'higher', baselineId) : '') + '</dd><dt>MTE2 / Vector / Cube</dt><dd>' + fmt(s.sol[2].pct, 1) + '% / ' + fmt(s.sol[1].pct, 1) + '% / ' + fmt(s.sol[0].pct, 1) + '%' + (bs ? compareLine(s.sol[2].pct, bs.sol[2].pct, fmt(bs.sol[2].pct, 1) + '% / ' + fmt(bs.sol[1].pct, 1) + '% / ' + fmt(bs.sol[0].pct, 1) + '%', 'higher', baselineId) : '') + '</dd><dt>HBM 流量 / step</dt><dd>' + fmt(s.traffic.total, 2) + ' GB' + (bs ? compareLine(s.traffic.total, bs.traffic.total, fmt(bs.traffic.total, 2) + ' GB', 'lower', baselineId) : '') + '</dd><dt>KV Cache</dt><dd>' + fmt(kv.utilization, 1) + '% · 命中 ' + fmt(kv.hitRate, 1) + '%' + (bkv ? compareLine(kv.hitRate, bkv.hitRate, fmt(bkv.utilization, 1) + '% · 命中 ' + fmt(bkv.hitRate, 1) + '%', 'higher', baselineId) : '') + '</dd></dl></div></section>'
      + '</div>';
  }

  function renderKpis(p, b, baselineId) {
    var s = p.summary;
    var bs = b && b.summary;
    return '<div class="so-profile-kpis">'
      + metric('TPOT · p50', fmt(s.tpot.p50, 1) + '<i> ms</i>', 'p90 ' + fmt(s.tpot.p90, 1) + ' · p99 ' + fmt(s.tpot.p99, 1) + ' ms', '', bs ? compareLine(s.tpot.p50, bs.tpot.p50, fmt(bs.tpot.p50, 1) + ' ms', 'lower', baselineId) : '')
      + metric('吞吐', int(s.tps) + '<i> tok/s</i>', 'batch ' + p.meta.batch + ' · 平均 ' + fmt(s.batchAvg, 1), '', bs ? compareLine(s.tps, bs.tps, int(bs.tps) + ' tok/s', 'higher', baselineId) : '')
      + metric('达成带宽', fmt(s.traffic.total / s.tpot.p50, 2) + '<i> TB/s</i>', fmt(s.traffic.total / s.tpot.p50 / p.meta.peakBw * 100, 1) + '% 峰值', 'warn', bs ? compareLine(s.traffic.total / s.tpot.p50, bs.traffic.total / bs.tpot.p50, fmt(bs.traffic.total / bs.tpot.p50, 2) + ' TB/s', 'higher', baselineId) : '')
      + metric('KV Cache', fmt(s.kvUsed, 2) + '<i> GB</i>', int(p.memory.kv.pagesUsed) + ' / ' + int(p.memory.kv.pagesTotal) + ' 页 · 碎片 ' + fmt(p.memory.kv.fragmentation, 1) + '%', '', bs ? compareLine(s.kvUsed, bs.kvUsed, fmt(bs.kvUsed, 2) + ' GB', 'lower', baselineId) : '')
      + metric('执行效率', fmt(s.efficiency, 1) + '<i> %</i>', '理论下界 ' + fmt(s.lowerBoundMs, 2) + ' ms', '', bs ? compareLine(s.efficiency, bs.efficiency, fmt(bs.efficiency, 1) + '%', 'higher', baselineId) : '')
      + '</div>';
  }

  function renderSol(p, b, baselineId) {
    var rows = p.summary.sol.map(function (unit) {
      var bottleneck = unit.pct === Math.max.apply(null, p.summary.sol.map(function (item) { return item.pct; }));
      var baseUnit = b && b.summary.sol.find(function (item) { return item.id === unit.id; });
      return '<div class="so-profile-solrow ' + (bottleneck ? 'is-bottleneck' : '') + '"><span>' + esc(unit.label) + '</span><div class="so-profile-soltrack' + (baseUnit ? ' is-grouped' : '') + '"><span data-unit="' + esc(unit.id) + '" style="width:' + unit.pct + '%"></span>' + (baseUnit ? '<span class="is-baseline" style="width:' + baseUnit.pct + '%"></span>' : '') + '</div><b>' + fmt(unit.pct, 1) + '%' + (baseUnit ? '<small>' + fmt(baseUnit.pct, 1) + '%</small>' : '') + '</b><small>' + esc(unit.detail) + (baseUnit ? compareLine(unit.pct, baseUnit.pct, fmt(baseUnit.pct, 1) + '%', 'higher', baselineId) : '') + '</small></div>';
    }).join('');
    return '<section class="so-card so-profile-card"><div class="so-card-head"><div><h3>Hardware unit utilization</h3><div class="so-finding-summary">每个 decode step 的时间去向</div></div><div class="so-profile-series-legend"><span><i></i>当前 Run</span>' + (b ? '<span class="is-baseline"><i></i>基线</span>' : '') + '</div></div><div class="so-card-body"><div class="so-profile-sol">' + rows + '</div><div class="so-profile-verdict"><strong>内存搬运是当前长 pole</strong><p>每 step 从 HBM 读取 <code>' + fmt(p.summary.traffic.total, 2) + ' GB</code>，当前达成带宽为 <code>' + fmt(p.summary.traffic.total / p.summary.tpot.p50, 2) + ' TB/s</code>；Cube 仅占 <code>' + fmt(p.summary.sol[0].pct, 1) + '%</code>，优先检查 MTE2 重叠与权重复用。</p></div></div></section>';
  }

  function renderMix(p, b, baselineId) {
    var total = p.summary.tpot.p50;
    return '<section class="so-card so-profile-card"><div class="so-card-head"><div><h3>耗时构成</h3><div class="so-finding-summary">按 Scope 进入算子分析</div></div><span class="so-pill">合计 ' + fmt(total, 1) + ' ms</span></div><div class="so-card-body"><div class="so-profile-mix">' + p.groups.map(function (g) {
      var bg = b && b.groups.find(function (item) { return item.id === g.id; });
      return '<button type="button" class="so-profile-mixrow" data-prof-mix="' + esc(g.id) + '"><i style="background:' + (GROUP_COLORS[g.id] || '#8b929a') + '"></i><b>' + esc(g.label) + '</b><strong>' + fmt(g.share, 1) + '%' + (bg ? '<span>' + fmt(bg.share, 1) + '%</span>' : '') + '</strong><small>' + fmt(g.ms, 3) + ' ms · ' + esc(g.detail) + (bg ? compareLine(g.ms, bg.ms, fmt(bg.ms, 3) + ' ms', 'lower', baselineId) : '') + '</small></button>';
    }).join('') + '</div></div></section>';
  }

  function renderOverview(p, b, baselineId) {
    return '<div class="so-profile-pane so-profile-overview">'
      + renderLayerSummary(p, b, baselineId)
      + '<div class="so-section-title"><h2>全链路性能</h2><p>Service 指标与 Hardware counter 的同一份采集上下文</p></div>'
      + renderKpis(p, b, baselineId)
      + renderSol(p, b, baselineId)
      + '<div class="so-profile-grid2">' + renderMix(p, b, baselineId) + renderBatchHistogram(p, b, baselineId) + '</div>'
      + '</div>';
  }

  function renderBatchHistogram(p, b, baselineId) {
    var values = (p.serving && p.serving.batchOverTime) || [];
    var baseValues = b && b.serving.batchOverTime || [];
    var max = Math.max.apply(null, values.concat(baseValues, [1]));
    var bars = values.slice(0, 32).map(function (value, index) {
      var baseValue = baseValues[index];
      return '<span><i title="当前 · step ' + (index + 1) + ' · batch ' + value + '" style="height:' + Math.max(12, value / max * 100) + '%"></i>' + (baseValue != null ? '<i class="is-baseline" title="基线 · step ' + (index + 1) + ' · batch ' + baseValue + '" style="height:' + Math.max(12, baseValue / max * 100) + '%"></i>' : '') + '</span>';
    }).join('');
    return '<section class="so-card so-profile-card"><div class="so-card-head"><div><h3>Batch 随时间</h3><div class="so-finding-summary">连续批处理窗口的实时 batch</div></div><span class="so-pill">平均 ' + fmt(p.serving && p.serving.batchAvg, 1) + (b ? ' · 基线 ' + fmt(b.serving.batchAvg, 1) : '') + '</span></div><div class="so-card-body"><div class="so-profile-hist' + (b ? ' is-grouped' : '') + '">' + bars + '</div><div class="so-profile-hist-meta"><span>step 1</span><b>' + (b ? '当前 / 基线成组展示' : '当前 Run') + '</b><span>step ' + p.meta.steps + '</span></div></div></section>';
  }

  function visibleOps(p) {
    var s = currentState();
    var query = String(s.query || '').trim().toLowerCase();
    var list = p.ops.filter(function (op) {
      return (!s.groupFilter || op.group === s.groupFilter) && (!query || op.name.toLowerCase().indexOf(query) >= 0 || op.scope.toLowerCase().indexOf(query) >= 0);
    });
    var key = s.sortKey || 'totalMs';
    var dir = s.sortDir === 'asc' ? 1 : -1;
    return list.slice().sort(function (a, b) {
      var av = key === 'mte2' ? a.units.mte2 : key === 'efficiency' ? a.efficiency : a[key];
      var bv = key === 'mte2' ? b.units.mte2 : key === 'efficiency' ? b.efficiency : b[key];
      if (av == null) return 1;
      if (bv == null) return -1;
      return (typeof av === 'string' ? av.localeCompare(bv) : av - bv) * dir;
    });
  }

  function opCell(current, baseline, formatted, baseFormatted, direction, baselineId, hasBase) {
    if (!baselineId) return formatted;
    if (!hasBase) return formatted + '<span class="so-profile-baseline">基线无对应算子</span>';
    if (current == null || baseline == null) return formatted + '<span class="so-profile-baseline">基线 · ' + baseFormatted + '</span>';
    return formatted + compareLine(current, baseline, baseFormatted, direction, baselineId);
  }

  function opRow(op, baseOp, maxShare, baselineId) {
    var s = currentState();
    var selected = op.id === s.selectedOp ? ' is-selected' : '';
    return '<tr class="so-prof-op-row' + selected + '" data-prof-op="' + esc(op.id) + '"><td><strong>' + esc(op.name) + '</strong><small>' + esc(op.scope) + '</small>' + (baselineId && !baseOp ? '<span class="so-profile-baseline">基线无对应算子</span>' : '') + '</td><td>' + opCell(op.calls, baseOp && baseOp.calls, int(op.calls), baseOp ? int(baseOp.calls) : '', 'neutral', baselineId, !!baseOp) + '</td><td>' + opCell(op.totalMs, baseOp && baseOp.totalMs, '<b>' + fmt(op.totalMs, 3) + '</b> ms', baseOp ? fmt(baseOp.totalMs, 3) + ' ms' : '', 'lower', baselineId, !!baseOp) + '</td><td><span class="so-prof-share"><i style="width:' + (op.share / maxShare * 100) + '%;background:' + (GROUP_COLORS[op.group] || '#356fae') + '"></i>' + fmt(op.share, 1) + '%</span>' + (baseOp ? compareLine(op.share, baseOp.share, fmt(baseOp.share, 1) + '%', 'neutral', baselineId) : '') + '</td><td>' + opCell(op.perLayerUs, baseOp && baseOp.perLayerUs, op.perLayerUs == null ? '—' : fmt(op.perLayerUs, 1) + ' μs', baseOp && baseOp.perLayerUs != null ? fmt(baseOp.perLayerUs, 1) + ' μs' : '—', 'lower', baselineId, !!baseOp) + '</td><td>' + opCell(op.units.mte2, baseOp && baseOp.units.mte2, fmt(op.units.mte2, 1) + '%', baseOp ? fmt(baseOp.units.mte2, 1) + '%' : '', 'higher', baselineId, !!baseOp) + '</td><td>' + opCell(op.achievedBw, baseOp && baseOp.achievedBw, op.achievedBw ? fmt(op.achievedBw, 2) + ' TB/s' : '—', baseOp && baseOp.achievedBw ? fmt(baseOp.achievedBw, 2) + ' TB/s' : '—', 'higher', baselineId, !!baseOp) + '</td><td><span class="so-prof-bound ' + esc(op.bound) + '">' + esc(op.boundLabel) + '</span>' + (baseOp ? '<span class="so-profile-baseline">基线 · ' + esc(baseOp.boundLabel) + '</span>' : '') + '</td><td>' + opCell(op.efficiency, baseOp && baseOp.efficiency, op.efficiency == null ? '—' : op.efficiency + '%', baseOp && baseOp.efficiency != null ? baseOp.efficiency + '%' : '—', 'higher', baselineId, !!baseOp) + '</td></tr>';
  }

  function renderOpDetail(p, b, baselineId) {
    var s = currentState();
    var op = p.ops.find(function (item) { return item.id === s.selectedOp; }) || p.ops[0];
    if (!op) return '';
    var baseOp = b && b.ops.find(function (item) { return item.id === op.id; });
    var units = ['cube', 'vector', 'mte2', 'mte3', 'sync'].filter(function (key) { return op.units[key] > 0; });
    var stack = units.map(function (key) { return '<i class="' + key + '" style="width:' + op.units[key] + '%"></i>'; }).join('');
    var baselineMissing = b && !baseOp ? '<span class="so-profile-baseline">基线 · 无对应算子</span>' : '';
    var bandwidthComparison = baseOp
      ? (op.achievedBw == null || baseOp.achievedBw == null
        ? '<span class="so-profile-baseline">基线 · ' + (baseOp.achievedBw == null ? '—' : fmt(baseOp.achievedBw, 2) + ' TB/s') + '</span>'
        : compareLine(op.achievedBw, baseOp.achievedBw, fmt(baseOp.achievedBw, 2) + ' TB/s', 'higher', baselineId))
      : '';
    var sparkMax = Math.max.apply(null, (op.perLayer || []).concat(baseOp && baseOp.perLayer || [], [1]));
    var spark = op.perLayer ? '<div class="so-prof-spark' + (baseOp && baseOp.perLayer ? ' is-grouped' : '') + '">' + op.perLayer.map(function (value, index) { var bv = baseOp && baseOp.perLayer && baseOp.perLayer[index]; return '<span><i style="height:' + (value / sparkMax * 100) + '%" title="当前 L' + index + ' · ' + fmt(value, 2) + ' μs"></i>' + (bv != null ? '<i class="is-baseline" style="height:' + (bv / sparkMax * 100) + '%" title="基线 L' + index + ' · ' + fmt(bv, 2) + ' μs"></i>' : '') + '</span>'; }).join('') + '</div>' : '<div class="so-empty">该任务没有逐层分布。</div>';
    return '<section class="so-card so-profile-card so-prof-detail"><div class="so-card-head"><div><h3>' + esc(op.name) + '</h3><div class="so-finding-summary">' + esc(op.scope) + ' · ' + esc(op.boundLabel) + ' bound</div></div><span class="so-pill accent">' + (op.efficiency == null ? '—' : op.efficiency + '% efficiency') + '</span></div><div class="so-card-body"><p class="so-prof-note">' + esc(op.note || '当前算子已关联硬件计数与逐层执行分布。') + baselineMissing + '</p><div class="so-prof-detail-grid"><section><h4>Hardware counter</h4><div class="so-prof-stack">' + stack + '</div><div class="so-prof-legend"><span>MTE2 ' + fmt(op.units.mte2, 1) + '%' + (baseOp ? ' / 基线 ' + fmt(baseOp.units.mte2, 1) + '%' : '') + '</span><span>Vector ' + fmt(op.units.vector, 1) + '%' + (baseOp ? ' / ' + fmt(baseOp.units.vector, 1) + '%' : '') + '</span><span>Cube ' + fmt(op.units.cube, 1) + '%' + (baseOp ? ' / ' + fmt(baseOp.units.cube, 1) + '%' : '') + '</span></div></section><section><h4>Per-layer latency</h4>' + spark + '</section><section><h4>Runtime facts</h4><dl class="so-kv"><dt>总耗时</dt><dd>' + fmt(op.totalMs, 3) + ' ms' + (baseOp ? compareLine(op.totalMs, baseOp.totalMs, fmt(baseOp.totalMs, 3) + ' ms', 'lower', baselineId) : '') + '</dd><dt>调用次数</dt><dd>' + int(op.calls) + (baseOp ? compareLine(op.calls, baseOp.calls, int(baseOp.calls), 'neutral', baselineId) : '') + '</dd><dt>达成带宽</dt><dd>' + (op.achievedBw ? fmt(op.achievedBw, 2) + ' TB/s' : '—') + bandwidthComparison + '</dd><dt>Arithmetic intensity</dt><dd>' + fmt(op.ai, 1) + ' FLOP/B' + (baseOp ? compareLine(op.ai, baseOp.ai, fmt(baseOp.ai, 1) + ' FLOP/B', 'neutral', baselineId) : '') + '</dd></dl></section></div></div></section>';
  }

  function renderOps(p, b, baselineId) {
    var s = currentState();
    var list = visibleOps(p);
    var maxShare = Math.max.apply(null, p.ops.map(function (op) { return op.share; }));
    var columns = [['name', '任务'], ['calls', '调用'], ['totalMs', '总耗时'], ['share', '占比'], ['perLayerUs', '每层'], ['mte2', 'MTE2'], ['achievedBw', '带宽'], ['bound', 'Bound'], ['efficiency', '效率']];
    var heads = columns.map(function (item) { var sorted = s.sortKey === item[0] ? ' is-sorted ' + s.sortDir : ''; return '<th class="' + sorted + '" data-prof-sort="' + item[0] + '">' + item[1] + '</th>'; }).join('');
    var group = s.groupFilter ? p.groups.find(function (item) { return item.id === s.groupFilter; }) : null;
    return '<div class="so-profile-pane so-profile-ops"><div class="so-prof-toolbar"><div class="so-prof-segment">' + [['flat', '按任务'], ['scope', '按 Scope'], ['bound', '按硬件单元']].map(function (item) { return '<button type="button" class="' + ((s.groupBy || 'flat') === item[0] ? 'is-active' : '') + '" data-prof-groupby="' + item[0] + '">' + item[1] + '</button>'; }).join('') + '</div><input class="so-input so-prof-search" type="search" data-prof-search placeholder="筛选任务或 Scope…" value="' + esc(s.query || '') + '">' + (group ? '<button type="button" class="so-pill accent so-prof-filter" data-prof-clear-filter>' + esc(group.label) + ' ×</button>' : '') + '<span class="so-prof-toolbar-spacer"></span><span class="so-finding-summary">' + list.length + ' / ' + p.ops.length + ' 项 · 合计 ' + fmt(list.reduce(function (sum, op) { return sum + op.totalMs; }, 0), 3) + ' ms' + (baselineId ? ' · 基线' : '') + '</span></div><div class="so-table-wrap so-prof-table-wrap"><table class="so-table so-prof-table"><thead><tr>' + heads + '</tr></thead><tbody>' + list.map(function (op) { var baseOp = b && b.ops.find(function (item) { return item.id === op.id; }); return opRow(op, baseOp, maxShare, baselineId); }).join('') + '</tbody></table></div>' + renderOpDetail(p, b, baselineId) + '</div>';
  }

  function render(tab, comparison) {
    var p = comparison && comparison.current && comparison.current.profile || profile();
    var b = comparison && comparison.baseline && comparison.baseline.profile || null;
    var baselineId = comparison && comparison.baseline && comparison.baseline.run.id || '';
    if (!p) return '<div class="so-empty">推理性能数据尚未加载。</div>';
    if (tab === 'ops') return renderOps(p, b, baselineId);
    if (tab === 'memory') return '<div class="so-profile-pane so-profile-source">' + (window.PtoInferenceMemory ? window.PtoInferenceMemory.render(p, b, { baselineId: baselineId, deltaInfo: deltaInfo }) : '<div class="so-empty">访存数据模块尚未加载。</div>') + '</div>';
    if (tab === 'serving') return '<div class="so-profile-pane so-profile-source">' + (window.PtoInferenceServing ? window.PtoInferenceServing.render(p, b, { baselineId: baselineId, deltaInfo: deltaInfo }) : '<div class="so-empty">批处理数据模块尚未加载。</div>') + '</div>';
    return renderOverview(p, b, baselineId);
  }

  function bind(nextCtx) {
    if (ctx) return;
    ctx = nextCtx;
    document.addEventListener('click', function (event) {
      var target = event.target;
      if (!target.closest('.so-profile-pane')) return;
      var sort = target.closest('[data-prof-sort]');
      if (sort) {
        var s = currentState();
        var key = sort.getAttribute('data-prof-sort');
        if (s.sortKey === key) s.sortDir = s.sortDir === 'desc' ? 'asc' : 'desc';
        else { s.sortKey = key; s.sortDir = key === 'name' || key === 'bound' ? 'asc' : 'desc'; }
        rerender();
        return;
      }
      var groupBy = target.closest('[data-prof-groupby]');
      if (groupBy) { currentState().groupBy = groupBy.getAttribute('data-prof-groupby'); rerender(); return; }
      var row = target.closest('[data-prof-op]');
      if (row) { currentState().selectedOp = row.getAttribute('data-prof-op'); rerender(); return; }
      var mix = target.closest('[data-prof-mix]');
      if (mix) { currentState().groupFilter = mix.getAttribute('data-prof-mix'); currentState().query = ''; ctx.state.observeTab = 'ops'; rerender(); return; }
      if (target.closest('[data-prof-clear-filter]')) { currentState().groupFilter = null; rerender(); }
    });
    document.addEventListener('input', function (event) {
      if (!event.target.matches('[data-prof-search]')) return;
      var caret = event.target.selectionStart;
      currentState().query = event.target.value;
      rerender();
      var next = document.querySelector('[data-prof-search]');
      if (next) { next.focus(); next.setSelectionRange(caret, caret); }
    });
  }

  window.PtoServingProfiler = { bind: bind, render: render, profileForRun: profileForRun, deltaInfo: deltaInfo };
})();
