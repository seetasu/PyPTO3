/* Device Detail · 一张卡的设备侧详细分析
 * 数据：GBS256 压测包里的 Ascend profiler trace（只采了 1 张卡），见 build-ops-data.cjs。
 * 四块内容：66 步总览 → 选中步的多流甘特 → 算子排行 → 选中算子的逐步趋势。
 * x 一律是时间（ms，相对这一步的起点），y 只表示流与算子分组；时间是 trace 原值，不缩放。 */
(function () {
  'use strict';

  const D = window.TRIAGE_DEVICE;
  const NS = 'http://www.w3.org/2000/svg';
  const $ = (s) => document.querySelector(s);
  const esc = (v) => String(v).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
  const r2 = (v) => Math.round(v * 100) / 100;

  function fmtMs(ms) {
    const v = Math.abs(ms);
    if (v >= 1000) return `${(ms / 1000).toFixed(2)} s`;
    if (v >= 100) return `${Math.round(ms)} ms`;
    if (v >= 10) return `${ms.toFixed(1)} ms`;
    if (v >= 1) return `${ms.toFixed(2)} ms`;
    return `${ms.toFixed(3)} ms`;
  }
  function el(name, attrs, parent) {
    const n = document.createElementNS(NS, name);
    Object.entries(attrs || {}).forEach(([k, v]) => n.setAttribute(k, v));
    if (parent) parent.appendChild(n);
    return n;
  }

  /* ------------------------------------------------------------ tooltip */
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
  function withTip(node, text) {
    node.addEventListener('mouseenter', (e) => showTip(e, text));
    node.addEventListener('mousemove', moveTip);
    node.addEventListener('mouseleave', hideTip);
    return node;
  }

  /* ------------------------------------------------------------ state */
  const idx = Object.fromEntries(D.steps.map((s, i) => [s.id, i]));
  const state = { step: D.defaultStep, op: null, sort: 'ms', dev: 6 };

  function readHash() {
    const q = new URLSearchParams(location.hash.slice(1));
    if (q.get('step') && idx[q.get('step')] != null) state.step = q.get('step');
    if (q.get('op') != null && D.catalog[Number(q.get('op'))]) state.op = Number(q.get('op'));
    if (q.get('sort')) state.sort = q.get('sort');
    if (q.get('lane') != null && /^\d+$/.test(q.get('lane'))) state.dev = Number(q.get('lane'));
  }
  function writeHash() {
    const q = new URLSearchParams();
    q.set('lane', state.dev); q.set('step', state.step); q.set('sort', state.sort);
    if (state.op != null) q.set('op', state.op);
    const h = `#${q}`;
    if (h !== location.hash) history.replaceState(null, '', h);
  }
  const step = () => D.steps[idx[state.step]];
  const group = (gi) => D.groups[gi];
  const groupOfOp = (ci) => D.gIdx[D.catalog[ci].group];

  /* ------------------------------------------------------------ 头部 */
  function renderHead() {
    const S = D.source;
    $('#devTitle').textContent = `dev ${state.dev} · 设备详情`;
    $('#backLink').href = `./index.html#view=e2e&layer=device&tab=dist&lane=${state.dev}`;
    $('#prov').innerHTML = `<b>不是本请求的数据</b>：来自 ${esc(S.run)} 的 Ascend profiler trace，${esc(S.rank)}；`
      + `与单请求页的 dev ${state.dev} 既不同次采集也不同负载，时间是 trace 原值，<b>没有缩放</b>到 <code>runner_run</code> 的窗口。`
      + `这一页回答的是「一个 decode step 在一张卡上长什么样」。`;
    $('#prov').title = `固定负载 ${S.workload}\n采集 ${S.capture}`;
    const waitMs = D.steps.reduce((a, s) => a + s.wait, 0);
    const kpi = (k, v, sub, hot) => `<div class="dv-kpi${hot ? ' is-hot' : ''}"><span>${esc(k)}</span><b>${esc(v)}</b>${sub ? `<i>${esc(sub)}</i>` : ''}</div>`;
    $('#kpis').innerHTML = kpi('采到的 step', String(D.nStep), `${fmtMs(D.steps.reduce((a, s) => a + s.ms, 0))} 合计`)
      + kpi('步长 P50', fmtMs(D.period.p50), `P90 ${fmtMs(D.period.p90)}`)
      + kpi('等待占比', `${D.waitShare}%`, `${fmtMs(waitMs)} / ${D.nStep} 步`, true)
      + kpi('算子执行 P50', fmtMs(D.computeMs.p50), `最多 ${fmtMs(D.computeMs.max)}`)
      + kpi('算子种类', String(D.catalog.length), '主执行流上');
  }

  /* ------------------------------------------------------------ ① 66 步总览
   * 一根柱 = 一步，柱高按步长，自下而上堆「算子执行 / 等待 / 空档」。
   * 点任一根下钻；当前这一步整列高亮。 */
  const S1 = { w: 0, h: 150, padL: 44, padR: 12, padT: 10, padB: 18 };
  function renderSteps() {
    const svg = $('#stepsChart');
    const box = svg.getBoundingClientRect();
    S1.w = Math.max(640, Math.round(box.width || 900));
    svg.setAttribute('viewBox', `0 0 ${S1.w} ${S1.h}`);
    svg.setAttribute('preserveAspectRatio', 'none');
    svg.innerHTML = '';
    const maxMs = Math.max(...D.steps.map((s) => s.ms));
    const x0 = S1.padL; const x1 = S1.w - S1.padR;
    const bw = (x1 - x0) / D.steps.length;
    const Y = (ms) => S1.padT + (1 - ms / maxMs) * (S1.h - S1.padT - S1.padB);
    const base = S1.h - S1.padB;

    // 轴：最大值与中位两条参考线
    [[maxMs, `${Math.round(maxMs)} ms`], [D.period.p50, `P50 ${Math.round(D.period.p50)}`]].forEach(([v, label]) => {
      el('line', { class: 'dv-grid', x1: x0, x2: x1, y1: Y(v), y2: Y(v) }, svg);
      el('text', { class: 'dv-axis', x: x0 - 6, y: Y(v) + 3, 'text-anchor': 'end' }, svg).textContent = label;
    });

    D.steps.forEach((s, i) => {
      const x = x0 + i * bw;
      const w = Math.max(1, bw - 1.2);
      const g = el('g', { class: `dv-bar${s.id === state.step ? ' is-sel' : ''}`, 'data-step': s.id, tabindex: 0, role: 'button' }, svg);
      let y = base;
      [['calc', s.compute], ['wait', s.wait], ['idle', s.idle]].forEach(([cls, ms]) => {
        const hh = (ms / maxMs) * (S1.h - S1.padT - S1.padB);
        if (hh <= 0) return;
        y -= hh;
        el('rect', { class: `dv-seg ${cls}`, x, y, width: w, height: hh }, g);
      });
      el('rect', { class: 'dv-hit', x, y: S1.padT, width: Math.max(1, bw), height: base - S1.padT }, g);
      withTip(g, `${s.id} · ${fmtMs(s.ms)}\n算子执行 ${fmtMs(s.compute)} · 等待 ${fmtMs(s.wait)} · 空档 ${fmtMs(s.idle)}\n${s.nTask} 个任务（点击看这一步）`);
      g.addEventListener('click', () => selectStep(s.id));
      g.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectStep(s.id); } });
    });
    el('text', { class: 'dv-axis', x: x0, y: S1.h - 5 }, svg).textContent = D.steps[0].id;
    el('text', { class: 'dv-axis', x: x1, y: S1.h - 5, 'text-anchor': 'end' }, svg).textContent = D.steps[D.steps.length - 1].id;

    $('#stepsTitle').textContent = `${D.nStep} 个 decode step`;
    $('#stepsLegend').innerHTML = ['calc 算子执行', 'wait 等待同步', 'idle 流上空档']
      .map((s) => { const [k, l] = s.split(' '); return `<span><i class="dv-seg ${k}"></i>${l}</span>`; }).join('');
    $('#stepsNote').textContent = `步的边界取设备侧 MODEL_EXECUTE 标记。${D.nStep} 步里等待占 ${D.waitShare}%，`
      + `算子执行 P50 只有 ${fmtMs(D.computeMs.p50)}——这张卡大部分时间在等别人。点任一根柱看那一步。`;
  }

  /* ------------------------------------------------------------ ② 多流甘特
   * 第一行是主执行流的每个任务（按算子分组着色），下面几行是旁流的忙碌区间。
   * 选中某个算子时，它的任务保持高亮，其余压暗。 */
  const S2 = { h: 0, padL: 104, padR: 16, padT: 14, padB: 22, row: 15, gap: 4 };
  function renderGantt() {
    const s = step();
    const svg = $('#gantt');
    const rows = 1 + D.side.length;
    S2.h = S2.padT + rows * (S2.row + S2.gap) + S2.padB;
    const box = svg.getBoundingClientRect();
    const w = Math.max(680, Math.round(box.width || 900));
    svg.setAttribute('viewBox', `0 0 ${w} ${S2.h}`);
    svg.setAttribute('preserveAspectRatio', 'none');
    svg.innerHTML = '';
    const x0 = S2.padL; const x1 = w - S2.padR;
    const X = (ms) => x0 + (ms / s.ms) * (x1 - x0);
    const rowY = (i) => S2.padT + i * (S2.row + S2.gap);

    // 刻度：每 20 ms 一条
    const tick = s.ms > 80 ? 20 : 10;
    for (let t = 0; t <= s.ms + 0.001; t += tick) {
      el('line', { class: 'dv-grid', x1: X(t), x2: X(t), y1: S2.padT - 4, y2: S2.h - S2.padB + 2 }, svg);
      el('text', { class: 'dv-axis', x: X(t), y: S2.h - S2.padB + 14, 'text-anchor': 'middle' }, svg).textContent = `${Math.round(t)}`;
    }
    el('text', { class: 'dv-axis', x: x1, y: S2.h - S2.padB + 14, 'text-anchor': 'end' }, svg).textContent = 'ms';

    // 主执行流
    el('text', { class: 'dv-rowname is-main', x: x0 - 8, y: rowY(0) + S2.row - 4, 'text-anchor': 'end' }, svg).textContent = '主执行流 46';
    const gm = el('g', { class: `dv-main-row${state.op != null ? ' is-focus' : ''}` }, svg);
    el('rect', { class: 'dv-rowbg', x: x0, y: rowY(0), width: x1 - x0, height: S2.row }, gm);
    s.tasks.forEach(([t, d, ci]) => {
      const gi = groupOfOp(ci);
      const on = state.op === ci;
      // 选中的算子单次可能只有零点几毫秒，按真实宽度画会找不到——给它一个最小可见宽度
      const w = X(t + d) - X(t);
      const node = el('rect', {
        class: `dv-task g-${group(gi).key}${on ? ' is-on' : ''}`,
        x: X(t), y: rowY(0) + 1, width: Math.max(on ? 2.5 : 0.7, w), height: S2.row - 2,
      }, gm);
      const c = D.catalog[ci];
      withTip(node, `${c.name}\n${c.type || '—'} · ${group(gi).label}\nt+${t.toFixed(3)} ms · 时长 ${fmtMs(d)}（点击只看这个算子）`);
      node.addEventListener('click', () => selectOp(state.op === ci ? null : ci));
    });

    // 旁流：只有忙碌区间，没有任务名
    D.side.forEach((sd, i) => {
      const y = rowY(i + 1);
      const per = sd.perStep[idx[state.step]];
      el('text', { class: `dv-rowname role-${sd.role}`, x: x0 - 8, y: y + S2.row - 4, 'text-anchor': 'end' }, svg).textContent = sd.label;
      const g = el('g', {}, svg);
      el('rect', { class: 'dv-rowbg', x: x0, y, width: x1 - x0, height: S2.row }, g);
      per.spans.forEach(([t, d]) => {
        if (t > s.ms) return;
        el('rect', { class: `dv-busy role-${sd.role}`, x: X(t), y: y + 3, width: Math.max(0.7, X(Math.min(s.ms, t + d)) - X(t)), height: S2.row - 6 }, g);
      });
      withTip(g, `${sd.label} · 这一步忙 ${fmtMs(per.ms)}（占 ${Math.round((per.ms / s.ms) * 100)}%）\n旁流只画忙碌区间，不画任务名`);
    });

    const busiest = D.side.map((sd) => ({ sd, ms: sd.perStep[idx[state.step]].ms })).sort((a, b) => b.ms - a.ms)[0];
    $('#ganttTitle').innerHTML = `${esc(s.id)} <span class="dv-sub">${fmtMs(s.ms)} · ${s.nTask} 个任务 · ${s.ops.length} 种算子</span>`;
    $('#ganttLegend').innerHTML = D.groups.filter((g) => g.key !== 'marker')
      .map((g) => `<span><i class="dv-task g-${g.key}"></i>${esc(g.label)}</span>`).join('');
    $('#ganttNote').innerHTML = `主执行流上 ${fmtMs(s.wait)} 在等（${Math.round((s.wait / s.ms) * 100)}%），算子真正在算 ${fmtMs(s.compute)}。`
      + `同一时间 <b>${esc(busiest.sd.label)}</b> 忙 ${fmtMs(busiest.ms)}——主流在等的时候，活在别的流上。`
      + ` 小于 ${D.drawMin * 1000} µs 的任务没有画（${D.dropped.n} 个，合计 ${fmtMs(D.dropped.ms)}，占忙时 0.7%），但算子排行里按全部任务统计。`;
  }

  /* ------------------------------------------------------------ ③ 算子排行 */
  const SORTS = [['ms', '合计耗时'], ['n', '次数'], ['avg', '单次均值']];
  function renderOps() {
    const s = step();
    const rows = s.ops.map(([ci, n, ms]) => ({ ci, n, ms, avg: ms / n, c: D.catalog[ci] }));
    const key = state.sort;
    rows.sort((a, b) => (key === 'n' ? b.n - a.n : key === 'avg' ? b.avg - a.avg : b.ms - a.ms));
    const top = rows[0] ? rows[0][key] : 1;
    $('#opSort').innerHTML = SORTS.map(([k, l]) => `<button type="button" class="dv-chip${state.sort === k ? ' is-on' : ''}" data-sort="${k}">${l}</button>`).join('');
    $('#opTable').innerHTML = `
      <thead><tr><th scope="col">算子</th><th scope="col">次数</th><th scope="col">合计</th><th scope="col">占这一步</th></tr></thead>
      <tbody>${rows.map((r) => {
    const g = group(groupOfOp(r.ci));
    const bar = Math.max(1, ((key === 'n' ? r.n : key === 'avg' ? r.avg : r.ms) / top) * 100);
    return `<tr class="${state.op === r.ci ? 'is-on' : ''}" data-op="${r.ci}" tabindex="0"
        title="${esc(`${r.c.name} · ${r.c.type || '—'} · ${r.n} 次 · 合计 ${fmtMs(r.ms)} · 单次均值 ${fmtMs(r.avg)}`)}">
        <th scope="row"><i class="dv-task g-${g.key}"></i><b>${esc(r.c.name)}</b><em>${esc(r.c.type || '—')}</em></th>
        <td>${r.n}</td>
        <td>${fmtMs(r.ms)}</td>
        <td class="dv-share"><span style="width:${bar.toFixed(1)}%"></span><em>${((r.ms / s.ms) * 100).toFixed(1)}%</em></td>
      </tr>`;
  }).join('')}</tbody>`;
    $('#opNote').textContent = `按全部 ${s.nTask} 个任务统计（含画不出来的短任务）。点一行：甘特上只亮这个算子，右侧换成它的逐步趋势。`;
    // 选中的算子可能排在很后面（或是从甘特点进来的），让它自己滚进视野
    const sel = state.op == null ? null : $(`.dv-table tbody tr[data-op="${state.op}"]`);
    if (sel) sel.scrollIntoView({ block: 'nearest' });
  }

  /* ------------------------------------------------------------ ④ 逐步趋势
   * 没选算子时画「每步的算子执行总时长」，选了就画这个算子每步的合计。 */
  const S4 = { h: 120, padL: 40, padR: 10, padT: 12, padB: 18 };
  function renderTrend() {
    const svg = $('#trend');
    const box = svg.getBoundingClientRect();
    const w = Math.max(320, Math.round(box.width || 420));
    svg.setAttribute('viewBox', `0 0 ${w} ${S4.h}`);
    svg.setAttribute('preserveAspectRatio', 'none');
    svg.innerHTML = '';
    const sel = state.op;
    const series = D.steps.map((s) => {
      if (sel == null) return s.compute;
      const hit = s.ops.find((o) => o[0] === sel);
      return hit ? hit[2] : 0;
    });
    const counts = sel == null ? null : D.steps.map((s) => { const h = s.ops.find((o) => o[0] === sel); return h ? h[1] : 0; });
    const max = Math.max(...series, 0.0001);
    const x0 = S4.padL; const x1 = w - S4.padR;
    const bw = (x1 - x0) / D.steps.length;
    const base = S4.h - S4.padB;
    const Y = (v) => S4.padT + (1 - v / max) * (base - S4.padT);
    el('line', { class: 'dv-grid', x1: x0, x2: x1, y1: Y(max), y2: Y(max) }, svg);
    el('text', { class: 'dv-axis', x: x0 - 6, y: Y(max) + 3, 'text-anchor': 'end' }, svg).textContent = fmtMs(max).replace(' ', '');
    el('line', { class: 'dv-grid', x1: x0, x2: x1, y1: base, y2: base }, svg);
    el('text', { class: 'dv-axis', x: x0 - 6, y: base + 3, 'text-anchor': 'end' }, svg).textContent = '0';
    series.forEach((v, i) => {
      const x = x0 + i * bw;
      const h = Math.max(v > 0 ? 0.8 : 0, (v / max) * (base - S4.padT));
      const g = el('g', { class: `dv-tbar${D.steps[i].id === state.step ? ' is-sel' : ''}`, 'data-step': D.steps[i].id }, svg);
      el('rect', { class: 'dv-seg calc', x, y: base - h, width: Math.max(1, bw - 1), height: h }, g);
      el('rect', { class: 'dv-hit', x, y: S4.padT, width: Math.max(1, bw), height: base - S4.padT }, g);
      withTip(g, sel == null
        ? `${D.steps[i].id} · 算子执行 ${fmtMs(v)}`
        : `${D.steps[i].id} · ${D.catalog[sel].name}\n${counts[i]} 次 · 合计 ${fmtMs(v)}`);
      g.addEventListener('click', () => selectStep(D.steps[i].id));
    });
    el('text', { class: 'dv-axis', x: x0, y: S4.h - 5 }, svg).textContent = D.steps[0].id;
    el('text', { class: 'dv-axis', x: x1, y: S4.h - 5, 'text-anchor': 'end' }, svg).textContent = D.steps[D.steps.length - 1].id;
    const sum = series.reduce((a, b) => a + b, 0);
    const nz = series.filter((v) => v > 0).length;
    $('#trendTitle').textContent = sel == null ? '逐步趋势 · 算子执行合计' : '逐步趋势 · 选中算子';
    $('#trendSub').innerHTML = sel == null
      ? `${D.nStep} 步 · 合计 ${fmtMs(sum)}`
      : `<b>${esc(D.catalog[sel].name)}</b> · ${D.catalog[sel].type || '—'}`;
    $('#trendNote').textContent = sel == null
      ? '每根柱是那一步所有算子的执行时间合计（不含等待）。点算子排行里的某一行，这里就换成那个算子。'
      : `${D.nStep} 步里有 ${nz} 步出现过它，合计 ${fmtMs(sum)}，平均每步 ${fmtMs(sum / D.nStep)}。再点一次排行里的那一行取消选择。`;
    renderGroups();
  }

  function renderGroups() {
    const s = step();
    const rows = D.groups.map((g, i) => ({ g, ms: s.groups[i] || 0 }))
      .filter((r) => r.ms > 0).sort((a, b) => b.ms - a.ms);
    const max = rows[0] ? rows[0].ms : 1;
    $('#groups').innerHTML = rows.map((r) => `
      <div class="dv-grow" title="${esc(`${r.g.label} · ${fmtMs(r.ms)} · 占这一步 ${((r.ms / s.ms) * 100).toFixed(1)}%`)}">
        <span class="dv-gk"><i class="dv-task g-${r.g.key}"></i>${esc(r.g.label)}</span>
        <span class="dv-gbar"><span style="width:${Math.max(0.8, (r.ms / max) * 100)}%"></span></span>
        <span class="dv-gv">${fmtMs(r.ms)}</span>
      </div>`).join('');
  }

  /* ------------------------------------------------------------ 口径 */
  function renderSpec() {
    const S = D.source;
    $('#specMeta').textContent = S.stream;
    $('#specTable').innerHTML = `<tbody>
      <tr><th scope="row"><b>来源</b></th><td>${esc(S.zip)} › ${esc(S.member)}</td></tr>
      <tr><th scope="row"><b>固定负载</b></th><td><code>${esc(S.workload)}</code>（压缩包名自带：GBS256 · 输入 256 / 输出 64）</td></tr>
      <tr><th scope="row"><b>采集方式</b></th><td>${esc(S.capture)}；${esc(S.captureNote)}</td></tr>
      <tr><th scope="row"><b>同一次采集的是</b></th><td>${esc(S.sameRunAs)}；<b>不是</b> ${esc(S.notSameRunAs)}</td></tr>
      <tr><th scope="row"><b>卡数</b></th><td>${esc(S.rank)}，所以没有 16 张卡的横向对比</td></tr>
      <tr><th scope="row"><b>画了哪些流</b></th><td>${esc(S.streamNote)}</td></tr>
      <tr><th scope="row"><b>时间</b></th><td>${esc(S.clock)}；${esc(S.profiler)}</td></tr>
      <tr><th scope="row"><b>step 边界</b></th><td>设备侧 <code>MODEL_EXECUTE</code> 标记之间，共 ${D.nStep} 步（P50 ${fmtMs(D.period.p50)} · P90 ${fmtMs(D.period.p90)} · 最长 ${fmtMs(D.period.max)}）</td></tr>
      <tr><th scope="row"><b>甘特的取舍</b></th><td>小于 ${D.drawMin * 1000} µs 的任务不画（${D.dropped.n} 个 / 合计 ${fmtMs(D.dropped.ms)}），算子排行仍按全部任务统计</td></tr>
    </tbody>`;
  }

  /* ------------------------------------------------------------ actions */
  function selectStep(id) {
    if (idx[id] == null || state.step === id) return;
    state.step = id;
    renderSteps(); renderGantt(); renderOps(); renderTrend(); writeHash();
  }
  function selectOp(ci) {
    state.op = ci;
    renderGantt(); renderOps(); renderTrend(); writeHash();
  }

  $('#opSort').addEventListener('click', (e) => {
    const b = e.target.closest('[data-sort]');
    if (!b) return;
    state.sort = b.dataset.sort; renderOps(); writeHash();
  });
  $('#opTable').addEventListener('click', (e) => {
    const r = e.target.closest('[data-op]');
    if (!r) return;
    const ci = Number(r.dataset.op);
    selectOp(state.op === ci ? null : ci);
  });
  $('#opTable').addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    const r = e.target.closest('[data-op]');
    if (!r) return;
    e.preventDefault();
    const ci = Number(r.dataset.op);
    selectOp(state.op === ci ? null : ci);
  });
  // ← / → 翻 step
  document.addEventListener('keydown', (e) => {
    if (e.target && e.target.closest && e.target.closest('input, textarea')) return;
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    const i = idx[state.step] + (e.key === 'ArrowRight' ? 1 : -1);
    if (i < 0 || i >= D.steps.length) return;
    e.preventDefault();
    selectStep(D.steps[i].id);
  });
  let rt;
  window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { renderSteps(); renderGantt(); renderTrend(); }, 120); });
  window.addEventListener('hashchange', () => { readHash(); renderAll(); });

  function renderAll() {
    renderHead(); renderSteps(); renderGantt(); renderOps(); renderTrend(); renderSpec();
  }
  readHash();
  renderAll();
  writeHash();
}());
