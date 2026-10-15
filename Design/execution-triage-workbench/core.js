/* Core Swimlane · 一张卡 60 个核的核级泳道
 * 数据：PyPTO 自己的 L2 swimlane 导出（Qwen3 TP1 的一次 prefill），见 build-core-data.cjs。
 * y = 核（上面 20 个 AIC，下面 40 个 AIV），x = 时间（ms，相对这次调用的起点）。
 * 一个 SPMD 任务同时落在多个核上，所以“选中一个算子”会在很多行上同时亮。 */
(function () {
  'use strict';

  const D = window.TRIAGE_CORE;
  const NS = 'http://www.w3.org/2000/svg';
  const $ = (s) => document.querySelector(s);
  const esc = (v) => String(v).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

  function fmtMs(ms) {
    const v = Math.abs(ms);
    if (v >= 100) return `${Math.round(ms)} ms`;
    if (v >= 10) return `${ms.toFixed(1)} ms`;
    if (v >= 1) return `${ms.toFixed(2)} ms`;
    return `${ms.toFixed(3)} ms`;
  }
  const fmtUs = (ms) => `${(ms * 1000).toFixed(1)} µs`;
  function el(name, attrs, parent) {
    const n = document.createElementNS(NS, name);
    Object.entries(attrs || {}).forEach(([k, v]) => n.setAttribute(k, v));
    if (parent) parent.appendChild(n);
    return n;
  }

  const tip = $('#tip');
  function showTip(e, text) { tip.textContent = text; tip.hidden = false; moveTip(e); }
  function moveTip(e) {
    const pad = 14; const r = tip.getBoundingClientRect();
    let x = e.clientX + pad; let y = e.clientY + pad;
    if (x + r.width > window.innerWidth - 8) x = e.clientX - r.width - pad;
    if (y + r.height > window.innerHeight - 8) y = e.clientY - r.height - pad;
    tip.style.left = `${Math.max(8, x)}px`; tip.style.top = `${Math.max(8, y)}px`;
  }
  const hideTip = () => { tip.hidden = true; };

  /* ------------------------------------------------------------ state */
  const state = { op: null, task: null, t0: 0, t1: D.spanMs, drag: null };
  const group = (key) => D.groups.find((g) => g.key === key) || { key, label: key };

  function readHash() {
    const q = new URLSearchParams(location.hash.slice(1));
    const o = q.get('op'); const t = q.get('task');
    if (o != null && D.ops[Number(o)]) state.op = Number(o);
    if (t != null && D.tasks[Number(t)]) state.task = Number(t);
    const r = q.get('r');
    if (r && /^[\d.]+,[\d.]+$/.test(r)) {
      const [a, b] = r.split(',').map(Number);
      if (b > a) { state.t0 = Math.max(0, a); state.t1 = Math.min(D.spanMs, b); }
    }
  }
  function writeHash() {
    const q = new URLSearchParams();
    if (state.op != null) q.set('op', state.op);
    if (state.task != null) q.set('task', state.task);
    if (state.t0 > 0.001 || state.t1 < D.spanMs - 0.001) q.set('r', `${state.t0.toFixed(3)},${state.t1.toFixed(3)}`);
    const h = q.toString() ? `#${q}` : '';
    if (h !== location.hash) history.replaceState(null, '', h || location.pathname);
  }

  /* ------------------------------------------------------------ 头部 */
  function renderHead() {
    const S = D.source;
    $('#title').textContent = `核级泳道 · ${S.model}`;
    $('#prov').innerHTML = `<b>第三份数据，和前两页都不是一回事</b>：${esc(S.exporter)} 导出的 ${esc(S.model)} ${esc(S.phase)}；`
      + `${esc(S.notSame)}。时间是 ${esc(S.clock)}，<b>不能与前两页相加或对齐</b>。`
      + `它回答的是另一个问题：<b>一次调用在一张卡的 ${D.nCore} 个核上是怎么铺开的</b>。`;
    const kpi = (k, v, sub, hot) => `<div class="dv-kpi${hot ? ' is-hot' : ''}"><span>${esc(k)}</span><b>${esc(v)}</b>${sub ? `<i>${esc(sub)}</i>` : ''}</div>`;
    $('#kpis').innerHTML = kpi('核', `${D.nCore}`, `${D.nAic} AIC + ${D.nAiv} AIV`)
      + kpi('窗口', fmtMs(D.spanMs), `${D.nTask} 个 SPMD 任务`)
      + kpi('AIC 占用', `${D.occupancy.aic}%`, `kernel ${fmtMs(D.kernelMs.aic)}`, true)
      + kpi('AIV 占用', `${D.occupancy.aiv}%`, `kernel ${fmtMs(D.kernelMs.aiv)}`)
      + kpi('核上任务', `${D.nRun}`, `${D.ops.length} 种算子`)
      + kpi('setup 合计', fmtMs(D.setupMs), '下发开销，不含 kernel');
  }

  /* ------------------------------------------------------------ 核泳道
   * 一行一个核；AIC 组在上、AIV 组在下，中间留一条缝。
   * 任务画两段：实色是 kernel，末尾一小截暗的是 local_setup。 */
  const L = { padL: 56, padR: 14, padT: 16, padB: 20, row: 7, gap: 1, split: 10 };
  function laneY(ci) {
    const i = ci;
    const extra = D.cores[ci].type === 'aiv' ? L.split : 0;
    return L.padT + i * (L.row + L.gap) + extra;
  }
  function chartH() { return laneY(D.nCore - 1) + L.row + L.padB; }

  function renderLanes() {
    const svg = $('#lanes');
    const box = svg.getBoundingClientRect();
    const w = Math.max(760, Math.round(box.width || 1000));
    const h = chartH();
    svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
    svg.setAttribute('preserveAspectRatio', 'none');
    svg.style.height = `${h}px`;
    svg.innerHTML = '';
    const x0 = L.padL; const x1 = w - L.padR;
    const span = state.t1 - state.t0;
    const X = (ms) => x0 + ((ms - state.t0) / span) * (x1 - x0);

    // 行底色 + 行名（每 5 行标一次，组首必标）
    D.cores.forEach((c, i) => {
      const y = laneY(i);
      el('rect', { class: `cr-rowbg is-${c.type}`, x: x0, y, width: x1 - x0, height: L.row }, svg);
      if (i % 5 === 0 || i === D.nAic) {
        el('text', { class: 'cr-rowname', x: x0 - 6, y: y + L.row - 0.5, 'text-anchor': 'end' }, svg).textContent = c.name;
      }
    });

    // 时间刻度
    const stepMs = niceStep(span);
    for (let t = Math.ceil(state.t0 / stepMs) * stepMs; t <= state.t1 + 1e-9; t += stepMs) {
      el('line', { class: 'dv-grid', x1: X(t), x2: X(t), y1: L.padT - 5, y2: h - L.padB + 2 }, svg);
      el('text', { class: 'dv-axis', x: X(t), y: h - L.padB + 13, 'text-anchor': 'middle' }, svg)
        .textContent = span < 2 ? t.toFixed(2) : t.toFixed(span < 12 ? 1 : 0);
    }
    el('text', { class: 'dv-axis', x: x1, y: h - L.padB + 13, 'text-anchor': 'end' }, svg).textContent = 'ms';

    // 任务：只画落在当前时间窗里的
    const g = el('g', { class: `cr-tasks${state.op != null || state.task != null ? ' is-focus' : ''}` }, svg);
    const dep = new Set();
    if (state.task != null) {
      const t = D.tasks[state.task];
      t.in.forEach((i) => dep.add(i)); t.out.forEach((i) => dep.add(i));
    }
    let drawn = 0;
    D.tasks.forEach((t, ti) => {
      const on = (state.op != null && t.o === state.op) || state.task === ti;
      const isDep = dep.has(ti);
      const gk = D.ops[t.o].group;
      t.r.forEach(([ci, ts, kms, sms]) => {
        const end = ts + kms + sms;
        if (end < state.t0 || ts > state.t1) return;
        drawn += 1;
        const y = laneY(ci);
        const kw = Math.max(on || isDep ? 1.6 : 0.5, X(ts + kms) - X(ts));
        el('rect', {
          class: `cr-task g-${gk}${on ? ' is-on' : ''}${isDep ? ' is-dep' : ''}`,
          x: X(ts), y: y + 0.5, width: kw, height: L.row - 1,
          'data-task': ti,
        }, g);
        const sw = X(ts + kms + sms) - X(ts + kms);
        if (sw > 0.6) el('rect', { class: 'cr-setup', x: X(ts + kms), y: y + 0.5, width: sw, height: L.row - 1, 'data-task': ti }, g);
      });
    });

    // 组标签
    el('text', { class: 'cr-group-label', x: x0 - 6, y: laneY(0) - 4, 'text-anchor': 'end' }, svg).textContent = `AIC ×${D.nAic}`;
    el('text', { class: 'cr-group-label', x: x0 - 6, y: laneY(D.nAic) - 4, 'text-anchor': 'end' }, svg).textContent = `AIV ×${D.nAiv}`;

    $('#laneTitle').innerHTML = `${D.nCore} 个核 <span class="dv-sub">${state.t0 < 0.001 && state.t1 > D.spanMs - 0.001
      ? `全窗口 ${fmtMs(D.spanMs)}` : `${state.t0.toFixed(2)}–${state.t1.toFixed(2)} ms（滚轮缩放 · 拖动平移）`} · 画出 ${drawn} 个任务实例</span>`;
    $('#zoomReset').disabled = state.t0 < 0.001 && state.t1 > D.spanMs - 0.001;
    $('#legend').innerHTML = D.groups.map((gg) => `<span><i class="cr-task g-${gg.key}"></i>${esc(gg.label)}</span>`).join('')
      + '<span><i class="cr-setup"></i>setup</span>';
    $('#laneNote').innerHTML = `实色是 kernel，紧跟着的暗色一小截是 <code>local_setup</code>（下发开销）。`
      + `一个 SPMD 任务同时落在多个核上，所以点一个算子会在很多行上同时亮。`
      + `AIC ${D.nAic} 个核平均占用 <b>${D.occupancy.aic}%</b>，AIV ${D.nAiv} 个核只有 <b>${D.occupancy.aiv}%</b>——`
      + `这次 prefill 的算力几乎全压在立方核上，向量核大部分时间空着。`;
  }
  function niceStep(span) {
    const raw = span / 10;
    const pow = 10 ** Math.floor(Math.log10(raw));
    const n = raw / pow;
    return (n > 5 ? 10 : n > 2 ? 5 : n > 1 ? 2 : 1) * pow;
  }

  /* ------------------------------------------------------------ 算子表 */
  function renderOps() {
    const max = D.ops[0] ? D.ops[0].kernel : 1;
    $('#opsSub').textContent = `${D.ops.length} 种 · 按 kernel 合计`;
    $('#opTable').innerHTML = `
      <thead><tr><th scope="col">算子</th><th scope="col">实例</th><th scope="col">kernel</th><th scope="col">setup</th><th scope="col">核</th></tr></thead>
      <tbody>${D.ops.map((o, i) => `
        <tr class="${state.op === i ? 'is-on' : ''}" data-op="${i}" tabindex="0"
          title="${esc(`${o.name} · ${o.nTask} 个任务 / ${o.n} 个核上实例 · kernel ${fmtMs(o.kernel)}（AIC ${fmtMs(o.aic)} · AIV ${fmtMs(o.aiv)}）· setup ${fmtMs(o.setup)} · 用了 ${o.cores} 个核`)}">
          <th scope="row"><i class="cr-task g-${o.group}"></i><b>${esc(o.name)}</b><em>${esc(group(o.group).label)}</em>
            <span class="cr-obar"><span style="width:${Math.max(1, (o.kernel / max) * 100).toFixed(1)}%"></span></span></th>
          <td>${o.n}</td>
          <td>${fmtMs(o.kernel)}</td>
          <td class="${o.setup / (o.kernel + o.setup) > 0.03 ? 'is-warn' : ''}">${fmtMs(o.setup)}</td>
          <td>${o.cores}</td>
        </tr>`).join('')}</tbody>`;
    $('#opNote').textContent = '点一行：泳道上只亮这个算子，右边换成它的每核分布。setup 占比超过 3% 的标出来——那是下发开销，不是算力。';
  }

  /* ------------------------------------------------------------ 右下：核分布 / 选中任务 */
  function renderDetail() {
    const sel = state.op;
    const cores = D.cores.map((c, i) => ({ c, i, ms: 0, n: 0 }));
    if (sel == null) {
      D.cores.forEach((c, i) => { cores[i].ms = c.kernel; cores[i].n = c.n; });
    } else {
      D.tasks.forEach((t) => { if (t.o !== sel) return; t.r.forEach(([ci, , k]) => { cores[ci].ms += k; cores[ci].n += 1; }); });
    }
    const max = Math.max(...cores.map((c) => c.ms), 0.0001);
    $('#detailTitle').textContent = sel == null ? '每个核的忙闲' : '选中算子的每核分布';
    $('#detailSub').innerHTML = sel == null
      ? `kernel 合计 ${fmtMs(D.kernelMs.aic + D.kernelMs.aiv)}`
      : `<b>${esc(D.ops[sel].name)}</b> · ${D.ops[sel].n} 个实例 / ${D.ops[sel].cores} 个核`;
    $('#coreList').innerHTML = cores.map(({ c, i, ms, n }) => `
      <div class="cr-core${ms > 0 ? '' : ' is-idle'}" title="${esc(`${c.name} · ${n} 个实例 · kernel ${fmtMs(ms)}`)}">
        <span class="cr-cn">${esc(c.name)}</span>
        <span class="cr-cbar"><span class="is-${c.type}" style="width:${((ms / max) * 100).toFixed(1)}%"></span></span>
        <span class="cr-cv">${ms > 0 ? fmtMs(ms) : '—'}</span>
      </div>`).join('');

    const box = $('#taskBox');
    if (state.task == null) {
      box.innerHTML = `<p class="dv-note">点泳道里的一个任务：这里会给它的 fanin / fanout 依赖——PyPTO 的 L2 记录里每个任务都带着上下游。</p>`;
      return;
    }
    const t = D.tasks[state.task];
    const o = D.ops[t.o];
    const span = t.r.reduce((a, [, ts, k, s]) => Math.max(a, ts + k + s), 0) - Math.min(...t.r.map((r) => r[1]));
    const line = (label, idxs) => `
      <div class="cr-dep"><span>${label}</span><div>${idxs.length
    ? idxs.map((i) => `<button type="button" class="dv-chip" data-task="${i}">${esc(D.ops[D.tasks[i].o].name)}<em>${esc(D.tasks[i].k)}</em></button>`).join('')
    : '<em class="cr-none">无</em>'}</div></div>`;
    box.innerHTML = `
      <h3 class="dv-h3">选中任务 ${esc(t.k)}</h3>
      <p class="dv-note"><b>${esc(o.name)}</b> · 在 ${t.r.length} 个核上并行 · 墙上时间 ${fmtMs(span)} ·
        kernel 合计 ${fmtMs(t.r.reduce((a, r) => a + r[2], 0))} · setup 合计 ${fmtUs(t.r.reduce((a, r) => a + r[3], 0))}</p>
      ${line('上游 fanin', t.in)}
      ${line('下游 fanout', t.out)}`;
  }

  /* ------------------------------------------------------------ 口径 */
  function renderSpec() {
    const S = D.source;
    $('#specMeta').textContent = `${S.model} · ${S.exporter}`;
    $('#specTable').innerHTML = `<tbody>
      <tr><th scope="row"><b>来源</b></th><td>${esc(S.file)}</td></tr>
      <tr><th scope="row"><b>导出格式</b></th><td>${esc(S.exporter)}（Worker View 下一核一行，${D.nAic} 个 AIC + ${D.nAiv} 个 AIV）</td></tr>
      <tr><th scope="row"><b>模型 / 阶段</b></th><td>${esc(S.model)} · ${esc(S.phase)}</td></tr>
      <tr><th scope="row"><b>与其它页的关系</b></th><td>${esc(S.notSame)}</td></tr>
      <tr><th scope="row"><b>时间</b></th><td>${esc(S.clock)}</td></tr>
      <tr><th scope="row"><b>为什么不是 decode</b></th><td>${esc(S.decodeNote)}</td></tr>
      <tr><th scope="row"><b>依赖</b></th><td>${esc(S.deps)}</td></tr>
      <tr><th scope="row"><b>kernel / setup</b></th><td>每个任务的 <code>kernel-duration-us</code> 与 <code>local_setup_us</code>，都是导出里的原值</td></tr>
    </tbody>`;
  }

  /* ------------------------------------------------------------ actions */
  function render() { renderLanes(); renderOps(); renderDetail(); writeHash(); }
  function selectOp(i) { state.op = i; state.task = null; render(); }
  function selectTask(i) {
    state.task = i;
    state.op = i == null ? state.op : D.tasks[i].o;
    render();
    const row = state.op == null ? null : $(`.dv-table tbody tr[data-op="${state.op}"]`);
    if (row) row.scrollIntoView({ block: 'nearest' });
  }

  $('#opTable').addEventListener('click', (e) => {
    const r = e.target.closest('[data-op]');
    if (!r) return;
    const i = Number(r.dataset.op);
    selectOp(state.op === i ? null : i);
  });
  $('#opTable').addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    const r = e.target.closest('[data-op]');
    if (!r) return;
    e.preventDefault();
    const i = Number(r.dataset.op);
    selectOp(state.op === i ? null : i);
  });
  $('#taskBox').addEventListener('click', (e) => {
    const b = e.target.closest('[data-task]');
    if (b) selectTask(Number(b.dataset.task));
  });
  $('#zoomReset').addEventListener('click', () => { state.t0 = 0; state.t1 = D.spanMs; render(); });

  // 泳道：悬停给任务详情，点击选中，滚轮缩放，拖动平移
  const lanes = $('#lanes');
  const tAt = (e) => {
    const r = lanes.getBoundingClientRect();
    const w = lanes.viewBox.baseVal.width || r.width;
    const px = ((e.clientX - r.left) / r.width) * w;
    const x0 = L.padL; const x1 = w - L.padR;
    return state.t0 + ((px - x0) / (x1 - x0)) * (state.t1 - state.t0);
  };
  lanes.addEventListener('mousemove', (e) => {
    if (state.drag) {
      const dt = (state.drag.t - tAt(e));
      const span = state.t1 - state.t0;
      let a = state.drag.t0 + dt; // 以按下那一刻的时间点为锚
      a = Math.max(0, Math.min(D.spanMs - span, a));
      if (Math.abs(a - state.t0) > 1e-6) { state.t0 = a; state.t1 = a + span; renderLanes(); }
      return;
    }
    const n = e.target.closest('[data-task]');
    if (!n) { hideTip(); return; }
    const t = D.tasks[Number(n.dataset.task)];
    const o = D.ops[t.o];
    const inWin = t.r.filter(([, ts, k, s]) => ts + k + s >= state.t0 && ts <= state.t1);
    const first = inWin[0] || t.r[0];
    showTip(e, `${o.name}（${t.k}）\n${D.cores[first[0]].name} · t+${first[1].toFixed(3)} ms\nkernel ${fmtUs(first[2])} · setup ${fmtUs(first[3])}\n这个任务在 ${t.r.length} 个核上并行（点击选中，看依赖）`);
  });
  lanes.addEventListener('mouseleave', hideTip);
  lanes.addEventListener('mousedown', (e) => {
    if (e.button !== 0) return;
    state.drag = { t: tAt(e), t0: state.t0, moved: false };
    lanes.classList.add('is-drag');
  });
  window.addEventListener('mouseup', (e) => {
    if (!state.drag) return;
    const moved = Math.abs(state.drag.t0 - state.t0) > 1e-6;
    state.drag = null;
    lanes.classList.remove('is-drag');
    if (moved) { writeHash(); return; }
    const n = e.target && e.target.closest ? e.target.closest('[data-task]') : null;
    if (n) selectTask(state.task === Number(n.dataset.task) ? null : Number(n.dataset.task));
    else if (e.target === lanes || (e.target.closest && e.target.closest('#lanes'))) { state.task = null; render(); }
  });
  lanes.addEventListener('wheel', (e) => {
    e.preventDefault();
    const anchor = tAt(e);
    const span = state.t1 - state.t0;
    const next = Math.max(0.05, Math.min(D.spanMs, span * (e.deltaY > 0 ? 1.25 : 0.8)));
    const ratio = (anchor - state.t0) / span;
    let a = anchor - next * ratio;
    a = Math.max(0, Math.min(D.spanMs - next, a));
    state.t0 = a; state.t1 = a + next;
    renderLanes(); writeHash();
  }, { passive: false });

  let rt;
  window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(renderLanes, 120); });
  window.addEventListener('hashchange', () => { readHash(); render(); });

  readHash();
  renderHead(); renderSpec(); render();
}());
