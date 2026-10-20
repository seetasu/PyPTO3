/* Presentation of static dependencies on the existing Pipeline tick axis.
 * Task, Stack, Stage and Tick are owned exclusively by PtoPagedAttentionContext. */
(function (global) {
  'use strict';
  const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch]);
  const categories = { state: '数值状态', reuse: 'Buffer 复用', sync: 'READY 同步' };
  const labels = { qk: 'QK', softmax: 'Softmax', pv: 'PV', accumulate: 'Accumulate' };
  function project(model, selection, { category = 'state', edgeId = null } = {}) {
    const edges = model.edges.filter((edge) => edge.kind === category);
    const node = model.nodes.find((n) => n.stack === selection.stack && n.stage === selection.stage);
    const selectedEdge = edges.find((edge) => edge.id === edgeId)
      || edges.find((edge) => edge.to === node?.id) || edges.find((edge) => edge.from === node?.id)
      || edges.find((edge) => [edge.from, edge.to].some((id) => model.nodes.find((n) => n.id === id)?.stack === selection.stack))
      || edges[0] || null;
    const anchors = Object.fromEntries(model.nodes.map((n) => [n.id, {
      x: 257 + n.tick * 62, y: ['qk', 'softmax'].includes(n.stage) ? 227 : 289,
      tick: n.tick, row: ['qk', 'softmax'].includes(n.stage) ? 'produce' : 'consume',
    }]));
    return { edges, selectedEdge, anchors };
  }
  function overlay(model, projection) {
    const { selectedEdge, anchors } = projection;
    if (!selectedEdge) return '';
    const a = anchors[selectedEdge.from], b = anchors[selectedEdge.to];
    // Only the active cause is drawn. The category list exposes the remaining relations.
    // Same-cell READY events use a loop, not a second operation or data-flow edge.
    const same = a.x === b.x && a.y === b.y;
    const lane = 254;
    const ay = a.y + (a.row === 'produce' ? 19 : -19);
    const by = b.y + (b.row === 'produce' ? 19 : -19);
    const path = same
      ? `M${a.x - 15} ${ay} C${a.x - 24} ${lane},${a.x + 24} ${lane},${a.x + 15} ${ay}`
      : `M${a.x} ${ay} V${lane} H${b.x} V${by}`;
    const label = selectedEdge.kind === 'sync' ? selectedEdge.synchronization.name
      : selectedEdge.kind === 'reuse' ? 'Slot 复用' : selectedEdge.label;
    const x = same ? a.x : (a.x + b.x) / 2;
    const y = 265;
    return `<g class="kf-pa-analysis-edge is-${selectedEdge.kind}" data-pa-dependency-edge="${esc(selectedEdge.id)}" tabindex="0" role="button" aria-label="${esc(label)} · 点击解释依赖"><title>${esc(selectedEdge.explanation)}</title><defs><marker id="pa-analysis-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10Z"/></marker></defs><path class="kf-pa-analysis-edge__hit" d="${path}"/><path class="kf-pa-analysis-edge__line" d="${path}" marker-end="url(#pa-analysis-arrow)"/><text x="${x}" y="${y}" text-anchor="middle">${esc(label)}</text></g>`;
  }
  function mount(host, { context, pipeline, synchronizations, onNavigate, onChange }) {
    const lifetime = new AbortController(), { signal } = lifetime;
    let model, projection, category = 'state', edgeId = null, signature = '';
    const node = (id) => model.nodes.find((n) => n.id === id);
    const name = (id) => { const n = node(id); return `S${n.stack} ${labels[n.stage]}`; };
    const sources = (source, prefix = '') => (source?.ranges || []).map((range) => `<button class="btn btn-ghost btn-sm" data-pa-analysis-line="${range.start}">${prefix}${esc(source.file)}:${range.start}–${range.end}</button>`).join('');
    function render() {
      projection = project(model, context.selection, { category, edgeId });
      const edge = projection.selectedEdge;
      const constraint = edge?.kind === 'state' ? '消费者必须读取正确版本的状态。该约束只连接状态的生产和消费，不要求所有 Stack 的 QK、Softmax、PV 串行执行。'
        : edge?.kind === 'reuse' ? '同一 Slot 的旧数据完成最后读取后，新 Stack 才可覆盖对应 Buffer；这不是额外的 READY Event，也不表示已经发生冲突。'
        : '生产端写入 Transfer Buffer 后 SET，消费端读取前 WAIT；WAIT 限制消费端的读取先后；Transfer Buffer 承载数据，READY Event 协调跨核执行。';
      host.innerHTML = `<div class="kf-pa-analysis">
        <header><div role="group" aria-label="依赖类型">${Object.entries(categories).map(([key, label]) => `<button class="btn btn-ghost btn-sm" data-pa-analysis-category="${key}" aria-pressed="${category === key}">${label}</button>`).join('')}</div><span data-pa-analysis-summary>Task ${context.selection.task} · ${model.seqLen} tokens · ${model.stackCount} Stacks · ${model.stepCount} Steps · ${global.PtoPagedAttentionIntro.model.constants.transferSlots} Slots</span></header>
        <nav class="kf-pa-analysis__relations" aria-label="${categories[category]}关系">${projection.edges.map((e) => `<button class="btn btn-ghost btn-sm" data-pa-analysis-edge="${esc(e.id)}" aria-pressed="${e.id === edge?.id}">${esc(name(e.from))} → ${esc(name(e.to))} · ${esc(e.label)}</button>`).join('') || '<span>当前 Task 的各 Stack 使用不同 Slot，没有 Buffer 复用依赖。</span>'}</nav>
        ${edge ? `<section class="kf-pa-analysis__explanation" aria-live="polite"><header><b>${esc(name(edge.from))} → ${esc(name(edge.to))}</b><div><button class="btn btn-ghost btn-sm" data-pa-analysis-navigate="compute">定位计算流程 ↗</button><button class="btn btn-ghost btn-sm" data-pa-analysis-navigate="source">定位 PyPTO 源码 ↗</button></div></header><dl><dt>依赖谁</dt><dd>${esc(name(edge.to))} 依赖 ${esc(name(edge.from))}${edge.buffer ? ` · ${esc(edge.buffer)}` : ''}</dd><dt>为什么</dt><dd>${esc(edge.explanation)}</dd><dt>执行约束</dt><dd>${constraint}</dd></dl>${edge.versions?.length ? `<p class="kf-pa-analysis__versions">${edge.versions.map(esc).join(' → ')}</p>` : ''}
        ${edge.synchronization ? `<p>${esc(edge.synchronization.name)} · #${edge.synchronization.eventId} · ${esc(edge.synchronization.buffer)} · ${edge.synchronization.dtype}<br>SET ${edge.synchronization.producer.coreType} / ${edge.synchronization.producer.pipe} → WAIT ${edge.synchronization.consumer.coreType} / ${edge.synchronization.consumer.pipe} · set ffts_mode=${edge.synchronization.producer.fftsMode}；wait 未传入</p><nav aria-label="READY 源码">${sources(edge.synchronization.producer.source, 'SET · ')}${sources(edge.consumerSource || edge.synchronization.consumer.source, 'WAIT · ')}</nav>` : `<details><summary>源码依据</summary>${sources(edge.source)}</details>`}</section>` : ''}
        <p class="kf-pa-analysis__note">源码静态依赖与调度推导，不是运行 Trace。Tick 不是执行时长；连接只表示当前原因，其余关系可在上方选择。</p>
      </div>`;
    }
    function update(nextPipeline = pipeline) {
      pipeline = nextPipeline;
      const next = JSON.stringify([pipeline.seqLen, context.selection]);
      if (next === signature) return;
      edgeId = null; signature = next;
      model = global.PtoPagedAttentionStackDependencyModel.build(pipeline, synchronizations);
      render();
    }
    function selectEdge(id) {
      const edge = model.edges.find((e) => e.id === id && e.kind !== 'data');
      if (!edge) return;
      const target = node(edge.to);
      context.selectInstance({ stack: target.stack, stage: target.stage });
      category = edge.kind; edgeId = id;
      render(); onChange();
    }
    host.addEventListener('click', (event) => {
      const choice = event.target.closest('[data-pa-analysis-category]');
      if (choice) { category = choice.dataset.paAnalysisCategory; edgeId = null; render(); onChange(); return; }
      const edge = event.target.closest('[data-pa-analysis-edge]');
      if (edge) { selectEdge(edge.dataset.paAnalysisEdge); return; }
      const nav = event.target.closest('[data-pa-analysis-navigate]');
      const line = event.target.closest('[data-pa-analysis-line]');
      if (nav || line) {
        const target = projection.selectedEdge && node(projection.selectedEdge.to);
        if (target) context.selectInstance({ stack: target.stack, stage: target.stage });
        onNavigate(nav?.dataset.paAnalysisNavigate || 'source', line ? Number(line.dataset.paAnalysisLine) : undefined);
      }
    }, { signal });
    update();
    return { update, selectEdge, overlay: () => overlay(model, projection),
      get focusedStacks() { const e = projection.selectedEdge; return e ? [node(e.from).stack, node(e.to).stack] : []; },
      abort: () => lifetime.abort() };
  }
  global.PtoPagedAttentionStackDependency = { mount, project, overlay };
})(typeof window === 'undefined' ? globalThis : window);
