/* Execution Triage — 一次上板执行的 Serving / Host / Device 定界。
 * 数据由 build-data.cjs 从真实上板产物生成（window.TRIAGE_DATA）；本文件只做派生与渲染。 */
(function () {
  'use strict';

  const DATA = window.TRIAGE_DATA;
  const LAYERS = ['serving', 'host', 'device'];
  const LAYER_NAME = { serving: 'Serving', host: 'Host', device: 'Device' };
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const $ = (sel) => document.querySelector(sel);
  const esc = (v) => String(v).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
  const sum = (xs, f) => xs.reduce((a, x) => a + f(x), 0);

  function fmtMs(ms, opts) {
    const v = Math.abs(ms);
    if (v >= 1000) return `${(ms / 1000).toFixed(2)} s`;
    if (v >= 100 || (opts && opts.int)) return `${Math.round(ms)} ms`;
    if (v >= 10) return `${ms.toFixed(1)} ms`;
    return `${ms.toFixed(2)} ms`;
  }
  const pct = (part, whole) => (whole > 0 ? (part / whole) * 100 : 0);
  const fmtPct = (p) => (p < 0.1 && p > 0 ? `${p.toFixed(2)}%` : `${p.toFixed(1)}%`);

  /* ================================================================ views
   * 每个 run 派生成同一形状的 view：layers / verdict / rhythm / waterfall / detail。 */

  function servingView(run, stepId) {
    const t = run.totals;
    const decode = run.steps.filter((s) => s.decodeStep);
    const laneCount = {};
    decode.forEach((s) => { laneCount[s.critLane] = (laneCount[s.critLane] || 0) + 1; });
    const hotLane = Number(Object.keys(laneCount).sort((a, b) => laneCount[b] - laneCount[a])[0]);
    const hotHits = laneCount[hotLane];
    const hotStat = run.laneStats.find((l) => l.dev === hotLane);
    const others = run.laneStats.filter((l) => l.dev !== hotLane).map((l) => l.bindAvgMs).sort((a, b) => a - b);
    const othersMed = others[Math.floor(others.length / 2)];
    // bind.args 在热点 lane 的 bind 中的占比
    let argsSum = 0; let bindSum = 0;
    decode.forEach((s) => {
      const l = run.waterfalls[s.id].lanes.find((x) => x.dev === hotLane);
      argsSum += l.bindArgsMs; bindSum += l.bindEnd - l.bindStart;
    });
    const rr = decode.map((s) => s.deviceMs);
    const newPeriod = run.decode.periodAvgMs - run.decode.whatIfSaveMs / decode.length;

    const layers = {
      serving: {
        ms: t.servingMs,
        status: { text: '可忽略', cls: 'ok' },
        top: `调度 ${fmtMs(run.serving.schedule.avgMs + run.serving.processOutput.avgMs + run.serving.queueStep.avgMs)}/step，已被流水覆盖`,
        parts: [{ label: '请求入口与收尾', ms: t.servingMs, layer: 'serving' }],
      },
      host: {
        ms: t.hostMs,
        status: { text: '首要瓶颈', cls: 'critical' },
        top: `lane bind ${fmtMs(t.hostBindMs)} · device ${hotLane} 关键 ${hotHits}/${decode.length} step`,
        parts: [
          { label: 'lane bind（关键 lane）', ms: t.hostBindMs, layer: 'host' },
          { label: 'lane 启动延迟', ms: t.hostLaunchMs, layer: 'host', shade: 0.6 },
          { label: 'Worker 暴露（未被覆盖）', ms: t.hostWorkerMs, layer: 'host', shade: 0.35 },
        ],
      },
      device: {
        ms: t.deviceMs,
        upper: true,
        status: { text: '上界 · 待实测', cls: 'warn' },
        top: `runner_run ${fmtMs(run.decode.deviceAvgMs)}/step，未采 device_wall`,
        parts: [{ label: 'runner_run（关键 lane，含同步等待）', ms: t.deviceMs, layer: 'device', upper: true }],
      },
    };

    const step = run.steps.find((s) => s.id === stepId) || decode[9];
    return {
      run,
      e2eMs: t.e2eMs,
      boundTitle: `端到端 ${fmtMs(t.e2eMs)} · 关键路径按层拆分`,
      boundNote: `请求 ${run.request.id} · ${run.stepCount.prefill} 段 prefill + ${run.stepCount.decode} 个 decode step`,
      layers,
      verdict: {
        layer: 'host',
        title: `瓶颈在 Host`,
        titleNote: `device ${hotLane} 的 bind 拖住每一步`,
        text: `Host 占关键路径 <b>${fmtPct(pct(t.hostMs, t.e2eMs))}</b>，其中 ${fmtPct(pct(t.hostBindMs, t.hostMs))} 是 lane bind。` +
          `16 条 lane 在 runner_run 末尾同步结束，最晚进入 runner_run 的 lane 决定每一步时长：` +
          `<b>${hotHits}/${decode.length}</b> 个 decode step 都是 <b>device ${hotLane}</b>（bind 均值 ${fmtMs(hotStat.bindAvgMs)}，其余 lane 中位 ${fmtMs(othersMed)}）。` +
          `Device ${fmtPct(pct(t.deviceMs, t.e2eMs))} 是上界，含同步等待。`,
        whatIf: { value: `−${fmtMs(run.decode.whatIfSaveMs)}`, text: `device ${hotLane} 与次慢 lane 同时进入 runner_run 时，E2E 预计缩短 ${fmtPct(pct(run.decode.whatIfSaveMs, t.e2eMs))}；step 周期 ${fmtMs(run.decode.periodAvgMs)} → ${fmtMs(newPeriod)}（假设 device 段不变）` },
        coverage: [
          { label: 'Serving / Worker 调度', state: 'measured' },
          { label: 'Host lane bind / runner_run', state: 'measured' },
          { label: 'Device 段（取自 runner_run 上界）', state: 'derived' },
          { label: 'device_wall · AICore', state: 'missing' },
        ],
      },
      rhythm: {
        title: 'Step 节奏',
        sub: `每根柱 = 一个 step 的关键路径（上一步结束 → 本步结束），按层堆叠。Device 段为 runner_run 上界。点击柱子查看该 step 的关键路径。`,
        unit: 'ms',
        avg: { value: run.decode.periodAvgMs, label: `decode 均值 ${fmtMs(run.decode.periodAvgMs)}` },
        items: run.steps.map((s) => ({
          id: s.id,
          serving: s.gapServingMs,
          host: s.gapWorkerMs + s.hostMs,
          device: s.deviceMs,
          tip: [
            ['Serving', s.gapServingMs, 'serving'],
            ['Host · Worker 暴露', s.gapWorkerMs, 'host'],
            ['Host · lane 启动 + bind', s.hostMs, 'host'],
            ['Device ≤ runner_run', s.deviceMs, 'device'],
          ],
          foot: `关键 lane：device ${s.critLane}（bind ${fmtMs(s.critBindMs)}，lane 中位 ${fmtMs(s.medBindMs)}）`,
          flag: s.decodeStep && s.critLane !== hotLane,
        })),
        deviceUpper: true,
      },
      step,
      waterfall: servingWaterfall(run, step, hotLane),
      detail: { serving: () => servingDetailServing(run), host: () => servingDetailHost(run, hotLane, hotHits, decode.length, argsSum / bindSum, newPeriod), device: () => servingDetailDevice(run, rr) },
      sources: [
        ['数据', run.source],
        ['时钟', run.clock],
        ['负载', run.workload],
        ['采集', run.capturedAt],
      ],
      meta: `${run.subtitle} · ${run.capturedAt}`,
    };
  }

  function servingWaterfall(run, step, hotLane) {
    const w = run.waterfalls[step.id];
    const rows = [];
    rows.push({
      label: 'Scheduler',
      group: 'serving',
      segs: w.serving.map((s) => (s.n === 'wait_worker_output'
        ? { s: s.s, d: s.d, kind: 'idle', name: 'wait_worker_output', tip: '等待 Worker 输出（不计入 Serving）' }
        : { s: s.s, d: s.d, kind: 'serving', name: s.n })),
    });
    rows.push({
      label: 'Worker',
      group: 'host',
      segs: w.worker.map((s) => ({
        s: s.s, d: s.d,
        kind: /reclaim/.test(s.n) ? 'wait' : 'host',
        name: `${s.n}${s.step ? ` · step ${s.step}` : ''}`,
        tip: /reclaim/.test(s.n) ? '等待 device 结果（reclaim）' : '与上一 step 的 device 执行重叠时不在关键路径',
      })),
    });
    const crit = step.critLane;
    w.lanes.forEach((l) => {
      const segs = [];
      if (l.bindStart - l.start > 0.3) segs.push({ s: l.start, d: l.bindStart - l.start, kind: 'host', name: 'pre-bind' });
      segs.push({ s: l.bindStart, d: l.bindEnd - l.bindStart, kind: 'host', name: 'bind', tip: `bind.args ${fmtMs(l.bindArgsMs)}` });
      if (l.rrStart - l.bindEnd > 0.3) segs.push({ s: l.bindEnd, d: l.rrStart - l.bindEnd, kind: 'wait', name: 'bind → runner_run 间隙', tip: '未被任何 span 覆盖的 host 时间' });
      segs.push({ s: l.rrStart, d: l.end - l.rrStart, kind: 'device', upper: true, name: 'runner_run', tip: 'host 观测；包含 device 执行与跨 lane 同步等待' });
      rows.push({ label: `device ${l.dev}`, crit: l.dev === crit, hot: l.dev === hotLane, group: 'lane', segs });
    });
    return {
      title: `关键路径 · ${step.id}`,
      sub: `Serving 调度、Worker 与 16 条 Host STRACE lane 在同一 host 时钟上对齐。关键 lane 决定本步时长：其 runner_run 起点之前算 Host，之后算 Device（上界）。`,
      spanMs: w.spanMs,
      rows,
      markers: [
        { x: w.prevEnd, label: '上一步结束' },
        { x: w.critStart, label: `device ${crit} 进入 runner_run`, crit: true },
        { x: w.end, label: '本步结束' },
      ],
      brackets: [
        { x0: Math.max(w.prevEnd, 0), x1: w.critStart, layer: 'host', label: `Host ${fmtMs(step.gapWorkerMs + step.hostMs)}` },
        { x0: w.critStart, x1: w.end, layer: 'device', label: `Device ≤ ${fmtMs(step.deviceMs)}` },
      ],
      legend: [['host', 'Host（bind / Worker）'], ['wait', 'Host 等待 / 未覆盖'], ['device', 'runner_run（Device 上界）'], ['serving', 'Serving 调度']],
    };
  }

  /* ================================================================ detail */

  function head(layer, view, sub) {
    const L = view.layers[layer];
    const share = L.na ? '' : `<span class="tagline">${L.upper ? '≤ ' : ''}${fmtMs(L.ms)} · ${fmtPct(pct(L.ms, view.e2eMs))}</span>`;
    const badge = L.status ? `<span class="badge ${L.status.cls}">${esc(L.status.text)}</span>` : '';
    return `<h3><span class="dot ${layer}"></span>${LAYER_NAME[layer]} ${share} ${badge}</h3><p class="d-sub">${sub}</p>`;
  }
  const kv = (rows) => `<dl class="kv">${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('')}</dl>`;

  function servingDetailHost(run, hotLane, hotHits, n, argsShare, newPeriod) {
    const t = run.totals;
    const max = Math.max(...run.laneStats.map((l) => l.bindAvgMs));
    const bars = run.laneStats.map((l) => `
      <div class="hbar ${l.dev === hotLane ? 'hot' : ''}">
        <span class="name">device ${l.dev}</span>
        <span class="track"><span class="fill" style="width:${(l.bindAvgMs / max) * 100}%"></span></span>
        <span class="val">${l.bindAvgMs.toFixed(1)}</span>
      </div>`).join('');
    return `
      ${head('host', currentView, 'Host 段 = 关键 lane 进入 runner_run 之前的时间，以及未被 device 执行覆盖的 Worker 时间。')}
      <section>
        <h4>构成</h4>
        ${kv([
          ['lane bind（关键 lane）', fmtMs(t.hostBindMs)],
          ['lane 启动延迟', fmtMs(t.hostLaunchMs)],
          ['Worker 暴露', fmtMs(t.hostWorkerMs)],
        ])}
      </section>
      <section>
        <h4>各 lane 平均 bind（ms，${n} 个 decode step）</h4>
        <div class="hbars">${bars}</div>
        <p class="d-sub">device ${hotLane} 在 ${hotHits}/${n} 个 step 中最晚进入 runner_run；其 bind 时间 ${(argsShare * 100).toFixed(2)}% 落在 bind.args（参数绑定）。</p>
      </section>
      <section>
        <h4>已被流水覆盖（不在关键路径）</h4>
        ${kv([
          ['Worker prepare_decode', `${fmtMs(run.worker.prepare.avgMs)} <small>/step</small>`],
          ['Worker dispatch_decode', `${fmtMs(run.worker.dispatch.avgMs)} <small>/step</small>`],
        ])}
        <p class="d-sub">双 buffer 流水：下一步的 prepare / dispatch 与当前步的 device 执行重叠。</p>
      </section>
      <section>
        <h4>下一步</h4>
        <ol class="steps-list">
          <li>对比 device ${hotLane} 与其它 lane 的 <b>bind.args</b> 输入规模与调用栈，确认为何稳定慢 2 倍。</li>
          <li>检查 device ${hotLane} 绑定线程的 CPU 亲和 / NUMA 位置 <span class="tagline">推断，待验证</span></li>
          <li>修复后复跑同一负载，验证 decode step 周期 ${fmtMs(run.decode.periodAvgMs)} → 约 ${fmtMs(newPeriod)}。</li>
        </ol>
      </section>`;
  }

  function servingDetailDevice(run, rr) {
    const sorted = rr.slice().sort((a, b) => a - b);
    return `
      ${head('device', currentView, 'Device 段取关键 lane 的 runner_run：host 观测值，包含 device 执行与 16 lane 的同步等待，因此只是上界。')}
      <section>
        <h4>runner_run（关键 lane，decode step）</h4>
        ${kv([
          ['均值', fmtMs(run.decode.deviceAvgMs)],
          ['P50', fmtMs(sorted[Math.floor(sorted.length / 2)])],
          ['最小 / 最大', `${fmtMs(sorted[0])} / ${fmtMs(sorted[sorted.length - 1])}`],
        ])}
      </section>
      <section>
        <h4>证据覆盖</h4>
        <div class="coverage" style="flex-direction:column">
          <span class="cov measured"><i></i>runner_run 起止（host STRACE）</span>
          <span class="cov missing"><i></i>device_wall（设备侧墙钟）</span>
          <span class="cov missing"><i></i>AICore / AIV swimlane、算子耗时</span>
          <span class="cov missing"><i></i>可归因的 H2D / D2H</span>
        </div>
      </section>
      <section>
        <h4>下一步</h4>
        <ol class="steps-list">
          <li>开启 <b>device_wall</b> 与 L2 swimlane 采集，把 Device 段从上界变成实测。</li>
          <li>有设备侧数据后，Device 段可继续下钻到算子与核占用。</li>
        </ol>
      </section>`;
  }

  function servingDetailServing(run) {
    const s = run.serving;
    const pre = run.steps.filter((x) => !x.decodeStep);
    const o = run.overrun[0];
    return `
      ${head('serving', currentView, 'Serving 段 = 关键路径上只有调度器在推进、Worker 与 device 都未工作的时间。')}
      <section>
        <h4>调度开销（每 step）</h4>
        ${kv([
          ['schedule', fmtMs(s.schedule.avgMs)],
          ['process_step_output', fmtMs(s.processOutput.avgMs)],
          ['queue_worker_step', fmtMs(s.queueStep.avgMs)],
          ['add_request（一次）', fmtMs(s.addRequest.avgMs)],
        ])}
        <p class="d-sub">wait_worker_output 平均 ${fmtMs(s.waitWorker.avgMs)}/step，是在等 Worker，归入 Host / Device。</p>
      </section>
      <section>
        <h4>首个 decode 之前</h4>
        ${kv(pre.map((x) => [x.id, fmtMs(x.periodMs)]).concat([['合计', fmtMs(run.ttftMs)]]))}
        <p class="d-sub">这段时间主要是 Host 与 Device，不是 Serving。</p>
      </section>
      ${o ? `<section>
        <h4>请求结束后的空跑</h4>
        <p class="d-sub" style="margin-top:0">流水预下发的 <b>${o.id}</b> 在响应返回后仍执行了 ${fmtMs(o.durMs)}，不计入本次 E2E，但会占用设备、推迟下一个请求。</p>
      </section>` : ''}`;
  }

  /* ================================================================ render */

  const RUN = DATA.runs[0];
  const state = { step: null, layer: null };
  let currentView = null;

  function buildView() {
    const v = servingView(RUN, state.step);
    state.step = v.step.id;
    if (!state.layer) state.layer = v.verdict.layer;
    return v;
  }

  function render() {
    currentView = buildView();
    const v = currentView;
    $('#runMeta').textContent = v.meta;
    renderVerdict(v);
    renderBound(v);
    renderTiles(v);
    renderRhythm(v);
    renderWaterfall(v);
    $('#detail').innerHTML = v.detail[state.layer]();
    $('#sources').innerHTML = v.sources.map(([k, val]) => `<span>${k}：<code>${esc(val)}</code></span>`).join('');
    writeHash();
  }

  function renderVerdict(v) {
    const L = v.verdict.layer;
    $('#verdictTitle').innerHTML = `<span class="layer-name"><span class="dot ${L}"></span>${esc(v.verdict.title)}</span><span class="muted" style="font:var(--type-body)">· ${esc(v.verdict.titleNote)}</span>`;
    $('#verdictText').innerHTML = v.verdict.text;
    const w = v.verdict.whatIf;
    $('#verdictFoot').innerHTML = `
      <div class="whatif"><strong>${esc(w.value)}</strong><span>${esc(w.text)}</span></div>
      <div class="coverage" aria-label="证据覆盖">${v.verdict.coverage.map((c) => `<span class="cov ${c.state}" title="${{ measured: '实测', derived: '推导', missing: '缺失' }[c.state]}"><i></i>${esc(c.label)}</span>`).join('')}</div>`;
  }

  function renderBound(v) {
    $('#boundTitle').textContent = v.boundTitle;
    $('#boundNote').textContent = v.boundNote;
    const bar = $('#boundBar');
    bar.innerHTML = '';
    LAYERS.forEach((layer) => {
      const L = v.layers[layer];
      if (L.na || L.ms <= 0) return;
      const share = pct(L.ms, v.e2eMs);
      const b = document.createElement('button');
      b.type = 'button';
      b.className = `seg ${layer}${L.upper ? ' upper' : ''}${state.layer === layer ? ' on' : (v.layers[state.layer].na ? '' : ' dim')}`;
      b.style.flex = `${share} 1 0`;
      b.dataset.layer = layer;
      b.setAttribute('aria-label', `${LAYER_NAME[layer]} ${fmtMs(L.ms)}，${fmtPct(share)}`);
      b.innerHTML = share > 9 ? `<span class="seg-label">${LAYER_NAME[layer]} · ${L.upper ? '≤ ' : ''}${fmtMs(L.ms)} · ${fmtPct(share)}</span>` : '';
      b.addEventListener('mouseenter', (e) => showTip(e, `<div class="t-title">${LAYER_NAME[layer]}</div>${tipRow(L.upper ? '时长（上界）' : '时长', L.ms, layer)}<div class="t-row"><span>占关键路径</span><b>${fmtPct(share)}</b></div>`));
      b.addEventListener('mousemove', moveTip);
      b.addEventListener('mouseleave', hideTip);
      bar.appendChild(b);
    });
    // drill-down of the focused layer
    const L = v.layers[state.layer];
    const parts = (L.parts || []).filter((p) => p.ms > 0);
    const total = sum(parts, (p) => p.ms) || 1;
    $('#boundSub').innerHTML = parts.length ? `
      <div class="sub-bar">${parts.map((p) => `<span style="flex:${Math.max(pct(p.ms, total), 0.8)} 1 0;background:var(--layer-${p.layer});opacity:${p.shade || 1}"></span>`).join('')}</div>
      <div class="sub-legend">${parts.map((p) => `<span><i class="sw" style="background:var(--layer-${p.layer});opacity:${p.shade || 1}"></i>${esc(p.label)} <b>${fmtMs(p.ms)}</b></span>`).join('')}</div>` : '';
  }

  function renderTiles(v) {
    $('#layerTiles').innerHTML = LAYERS.map((layer) => {
      const L = v.layers[layer];
      const share = pct(L.ms, v.e2eMs);
      return `
        <button class="tile${L.na ? ' na' : ''}" type="button" role="tab" data-layer="${layer}" aria-selected="${state.layer === layer}">
          <span class="tile-head"><span class="dot ${layer}"></span>${LAYER_NAME[layer]}<span class="badge ${L.status.cls}">${esc(L.status.text)}</span></span>
          <span class="tile-value">${L.na ? '<strong>—</strong>' : `<strong>${L.upper ? '≤' : ''}${fmtMs(L.ms)}</strong><span>${fmtPct(share)}</span>`}</span>
          <span class="tile-top">${esc(L.top)}</span>
        </button>`;
    }).join('');
  }

  /* ---------------------------------------------------------------- SVG helpers */
  function el(name, attrs, parent) {
    const n = document.createElementNS(SVG_NS, name);
    Object.entries(attrs || {}).forEach(([k, v]) => n.setAttribute(k, v));
    if (parent) parent.appendChild(n);
    return n;
  }
  function niceStep(max, target) {
    const raw = max / target;
    const p = 10 ** Math.floor(Math.log10(raw));
    return [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= raw);
  }
  function hatchDefs(svg) {
    const defs = el('defs', {}, svg);
    const pat = el('pattern', { id: 'hatch', width: 6, height: 6, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(135)' }, defs);
    el('rect', { width: 2, height: 6, fill: 'rgba(255,255,255,0.28)' }, pat);
    return defs;
  }
  const fillOf = (layer) => `var(--layer-${layer})`;
  // 聚焦某层时其余层降亮；聚焦“不适用”的层时不降亮
  const opacityOf = (layer) => (state.layer === layer || currentView.layers[state.layer].na ? 1 : 0.5);

  /* ---------------------------------------------------------------- rhythm */
  function renderRhythm(v) {
    const R = v.rhythm;
    $('#rhythmTitle').textContent = R.title;
    $('#rhythmSub').textContent = R.sub;
    $('#rhythmLegend').innerHTML = [
      v.layers.serving.na ? '' : `<span><i class="sw" style="background:var(--layer-serving)"></i>Serving</span>`,
      `<span><i class="sw" style="background:var(--layer-host)"></i>Host</span>`,
      `<span><i class="sw ${R.deviceUpper ? 'hatch' : ''}" style="background-color:var(--layer-device)"></i>Device${R.deviceUpper ? '（上界）' : ''}</span>`,
      R.avg ? `<span><i class="sw" style="height:0;border-top:1.5px dashed var(--foreground-secondary);border-radius:0"></i>${esc(R.avg.label)}</span>` : '',
    ].join('');

    const host = $('#rhythmChart');
    host.innerHTML = '';
    const W = Math.max(320, host.clientWidth - 40);
    const H = 200;
    const m = { l: 44, r: 8, t: 10, b: 26 };
    const items = R.items;
    const maxV = Math.max(...items.map((i) => i.serving + i.host + i.device)) * 1.08;
    const step = niceStep(maxV, 4);
    const yMax = Math.ceil(maxV / step) * step;
    const iw = W - m.l - m.r;
    const ih = H - m.t - m.b;
    const slot = iw / items.length;
    const bw = Math.min(56, Math.max(3, slot - (slot > 12 ? 3 : 1.5)));
    const y = (val) => m.t + ih - (val / yMax) * ih;

    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, height: H, role: 'img', 'aria-label': `${R.title}：${items.length} 个柱，按层堆叠` }, host);
    hatchDefs(svg);
    const grid = el('g', { class: 'grid' }, svg);
    for (let t = 0; t <= yMax + 1e-6; t += step) {
      el('line', { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t) }, grid);
      const tx = el('text', { x: m.l - 6, y: y(t) + 3, 'text-anchor': 'end' }, grid);
      const top = t + step > yMax + 1e-6;
      tx.textContent = `${t >= 1000 ? `${t / 1000}s` : t}${top && t < 1000 ? ` ${R.unit}` : ''}`;
    }

    const selId = v.step.id;
    const labelEvery = items.length > 20 ? Math.ceil(36 / slot / 5) * 5 : 1;
    const selIdx = items.findIndex((it) => it.id === selId);
    items.forEach((it, i) => {
      const x = m.l + i * slot + (slot - bw) / 2;
      const g = el('g', { class: 'bar-group' }, svg);
      let acc = 0;
      [['serving', it.serving], ['host', it.host], ['device', it.device]].forEach(([layer, val]) => {
        if (val <= 0) return;
        const y0 = y(acc);
        const y1 = y(acc + val);
        const hgt = Math.max(0.8, y0 - y1 - (acc > 0 ? 1.5 : 0));
        const r = el('rect', { x, y: y1, width: bw, height: hgt, rx: Math.min(2, bw / 4), fill: fillOf(layer), 'fill-opacity': opacityOf(layer) }, g);
        r.style.transition = 'fill-opacity .15s';
        if (layer === 'device' && R.deviceUpper) el('rect', { x, y: y1, width: bw, height: hgt, rx: Math.min(2, bw / 4), fill: 'url(#hatch)' }, g);
        acc += val;
      });
      if (it.flag) el('circle', { cx: x + bw / 2, cy: y(acc) - 5, r: 2, fill: 'var(--foreground)' }, g);
      if (it.id === selId) el('rect', { class: 'sel-ring', x: x - 2, y: y(acc) - 2, width: bw + 4, height: y(0) - y(acc) + 4, rx: 3 }, svg);
      const nearSel = items.length > 20 && it.id !== selId && Math.abs(i - selIdx) < Math.max(3, Math.ceil(30 / slot));
      if ((i % labelEvery === 0 && !nearSel) || it.id === selId) {
        const tx = el('text', { x: x + bw / 2, y: H - 8, 'text-anchor': 'middle', class: it.id === selId ? 'tick-strong' : '' }, svg);
        tx.textContent = items.length > 20 ? it.id.replace('Draft init', 'Draft').replace('Prefill', 'P') : it.id;
      }
      const hit = el('rect', { class: 'hit', x: m.l + i * slot, y: m.t, width: slot, height: ih, tabindex: 0, role: 'button', 'aria-label': `${it.id}：${fmtMs(it.serving + it.host + it.device)}` }, svg);
      const tipHtml = `<div class="t-title">${esc(it.id)} · ${fmtMs(it.serving + it.host + it.device)}</div>${it.tip.filter((r) => r[1] > 0.005).map((r) => tipRow(r[0], r[1], r[2])).join('')}<div class="t-row" style="margin-top:4px"><span>${esc(it.foot)}</span></div>`;
      hit.addEventListener('mouseenter', (e) => showTip(e, tipHtml));
      hit.addEventListener('mousemove', moveTip);
      hit.addEventListener('mouseleave', hideTip);
      const pick = () => { state.step = it.id; hideTip(); render(); };
      hit.addEventListener('click', pick);
      hit.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } });
    });
    if (R.avg) {
      el('line', { class: 'avg', x1: m.l, x2: W - m.r, y1: y(R.avg.value), y2: y(R.avg.value) }, svg);
    }
    el('line', { x1: m.l, x2: W - m.r, y1: y(0), y2: y(0), stroke: 'var(--border-strong)' }, svg);
  }

  /* ---------------------------------------------------------------- waterfall */
  function renderWaterfall(v) {
    const F = v.waterfall;
    $('#wfTitle').textContent = F.title;
    $('#wfSub').textContent = F.sub;
    const swatch = { host: 'background:var(--layer-host)', wait: '', device: 'background-color:var(--layer-device)', serving: 'background:var(--layer-serving)' };
    $('#wfLegend').innerHTML = F.legend.map(([k, label]) => `<span><i class="sw ${k === 'wait' ? 'wait' : ''} ${k === 'device' && v.rhythm.deviceUpper ? 'hatch' : ''}" style="${swatch[k]}"></i>${esc(label)}</span>`).join('') +
      (F.rows.some((r) => r.crit) ? `<span><i class="sw outline"></i>关键行</span>` : '');

    const host = $('#wfChart');
    host.innerHTML = '';
    const W = Math.max(340, host.clientWidth - 40);
    const rowH = 18;
    const gap = 4;
    const labelW = 92;
    const top = F.brackets.length ? 40 : 8;
    const H = top + F.rows.length * (rowH + gap) + 26;
    const m = { l: labelW, r: 12 };
    const iw = W - m.l - m.r;
    const x = (ms) => m.l + (ms / F.spanMs) * iw;

    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, height: H, role: 'img', 'aria-label': F.title }, host);
    hatchDefs(svg);
    const step = niceStep(F.spanMs, 6);
    const grid = el('g', { class: 'grid' }, svg);
    for (let t = 0; t <= F.spanMs + 1e-6; t += step) {
      el('line', { x1: x(t), x2: x(t), y1: top - 4, y2: H - 22 }, grid);
      const tx = el('text', { x: x(t), y: H - 8, 'text-anchor': 'middle' }, grid);
      tx.textContent = `${Math.round(t)}`;
    }
    const unit = el('text', { x: W - m.r, y: H - 8, 'text-anchor': 'end' }, svg);
    unit.textContent = 'ms';

    // brackets: Host / Device partition of this step
    F.brackets.forEach((b) => {
      const x0 = x(b.x0); const x1 = x(b.x1);
      const yb = 26;
      el('path', { class: 'bracket', d: `M${x0 + 1},${yb + 6} V${yb} H${x1 - 1} V${yb + 6}`, stroke: fillOf(b.layer), 'stroke-opacity': opacityOf(b.layer) === 1 ? 1 : 0.7 }, svg);
      const tx = el('text', { x: (x0 + x1) / 2, y: yb - 6, 'text-anchor': 'middle', class: 'lbl' }, svg);
      tx.textContent = b.label;
    });

    F.rows.forEach((row, i) => {
      const yy = top + i * (rowH + gap);
      if (row.crit) el('rect', { class: 'row-crit', x: 0, y: yy - 2, width: W, height: rowH + 4, rx: 4 }, svg);
      const lb = el('text', { x: 6, y: yy + rowH / 2 + 4, class: `lbl${row.crit ? ' crit' : ''}` }, svg);
      lb.textContent = row.crit && row.group === 'lane' ? `${row.label} ★` : row.label;
      row.segs.forEach((s) => {
        if (s.d <= 0) return;
        const sx = x(s.s);
        const sw = Math.max(1.5, x(s.s + s.d) - sx);
        let node;
        if (s.kind === 'idle') {
          node = el('line', { x1: sx, x2: sx + sw, y1: yy + rowH / 2, y2: yy + rowH / 2, stroke: 'var(--layer-serving)', 'stroke-opacity': 0.45, 'stroke-dasharray': '2 3' }, svg);
          node = el('rect', { x: sx, y: yy, width: sw, height: rowH, fill: 'transparent' }, svg);
        } else if (s.kind === 'span') {
          node = el('rect', { x: sx, y: yy + 3, width: sw, height: rowH - 6, rx: 3, fill: 'none', stroke: 'var(--border-strong)' }, svg);
        } else if (s.kind === 'wait') {
          node = el('rect', { x: sx, y: yy + 2, width: sw, height: rowH - 4, rx: 2, fill: 'var(--layer-host)', 'fill-opacity': state.layer === 'host' ? 0.28 : 0.16 }, svg);
        } else {
          const layer = s.kind;
          node = el('rect', { x: sx, y: yy, width: sw, height: rowH, rx: 2, fill: fillOf(layer), 'fill-opacity': opacityOf(layer) * (s.shade || 1) }, svg);
          if (s.upper) el('rect', { x: sx, y: yy, width: sw, height: rowH, rx: 2, fill: 'url(#hatch)', 'pointer-events': 'none' }, svg);
          if (row.crit && row.group === 'lane') el('rect', { x: sx + 0.75, y: yy + 0.75, width: Math.max(0, sw - 1.5), height: rowH - 1.5, rx: 2, fill: 'none', stroke: 'var(--foreground)', 'stroke-width': 1.5, 'pointer-events': 'none' }, svg);
          const label = `${s.name} ${fmtMs(s.d)}`;
          if (sw > label.length * 6.2 + 12) {
            const t = el('text', { x: sx + 6, y: yy + rowH / 2 + 4, class: 'in-bar', 'pointer-events': 'none' }, svg);
            t.textContent = label;
          }
        }
        const html = `<div class="t-title">${esc(row.label)} · ${esc(s.name)}</div>${tipRow('时长', s.d, s.kind === 'device' ? 'device' : s.kind === 'serving' || s.kind === 'idle' ? 'serving' : 'host')}<div class="t-row"><span>起点</span><b>${s.s.toFixed(1)} ms</b></div>${s.tip ? `<div class="t-row" style="margin-top:4px"><span>${esc(s.tip)}</span></div>` : ''}`;
        node.addEventListener('mouseenter', (e) => showTip(e, html));
        node.addEventListener('mousemove', moveTip);
        node.addEventListener('mouseleave', hideTip);
      });
    });

    F.markers.forEach((mk) => {
      el('line', { class: mk.crit ? 'marker-crit' : 'marker', x1: x(mk.x), x2: x(mk.x), y1: top - 4, y2: H - 22, 'pointer-events': 'none' }, svg);
      const hit = el('rect', { x: x(mk.x) - 4, y: top - 4, width: 8, height: H - top - 18, fill: 'transparent' }, svg);
      const html = `<div class="t-title">${esc(mk.label)}</div><div class="t-row"><span>时间</span><b>${mk.x.toFixed(1)} ms</b></div>`;
      hit.addEventListener('mouseenter', (e) => showTip(e, html));
      hit.addEventListener('mousemove', moveTip);
      hit.addEventListener('mouseleave', hideTip);
    });
  }

  /* ---------------------------------------------------------------- tooltip */
  const tip = $('#tip');
  function tipRow(label, ms, layer) {
    return `<div class="t-row"><span><i class="sw" style="background:var(--layer-${layer})"></i>${esc(label)}</span><b>${fmtMs(ms)}</b></div>`;
  }
  function showTip(e, html) { tip.innerHTML = html; tip.hidden = false; moveTip(e); }
  function moveTip(e) {
    const pad = 14;
    const r = tip.getBoundingClientRect();
    let left = e.clientX + pad;
    let topY = e.clientY + pad;
    if (left + r.width > window.innerWidth - 8) left = e.clientX - r.width - pad;
    if (topY + r.height > window.innerHeight - 8) topY = e.clientY - r.height - pad;
    tip.style.left = `${Math.max(8, left)}px`;
    tip.style.top = `${Math.max(8, topY)}px`;
  }
  function hideTip() { tip.hidden = true; }

  /* ---------------------------------------------------------------- state & events */
  function writeHash() {
    const h = `step=${encodeURIComponent(state.step || '')}&layer=${state.layer}`;
    if (location.hash.slice(1) !== h) history.replaceState(null, '', `#${h}`);
  }
  function readHash() {
    const p = new URLSearchParams(location.hash.slice(1));
    if (p.get('step')) state.step = p.get('step');
    if (LAYERS.includes(p.get('layer'))) state.layer = p.get('layer');
  }

  document.addEventListener('click', (e) => {
    const layerBtn = e.target.closest('[data-layer]');
    if (layerBtn) { state.layer = layerBtn.dataset.layer; render(); }
  });

  const themeBtn = $('#themeToggle');
  function applyTheme(t) { document.documentElement.setAttribute('data-theme', t); }
  try { const saved = localStorage.getItem('execution-triage-theme'); if (saved) applyTheme(saved); } catch (err) { /* storage unavailable */ }
  themeBtn.addEventListener('click', () => {
    const next = document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
    applyTheme(next);
    try { localStorage.setItem('execution-triage-theme', next); } catch (err) { /* storage unavailable */ }
    render();
  });

  let resizeTimer = 0;
  window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(render, 120); });

  readHash();
  render();
}());
