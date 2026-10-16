/* Task-local static execution dependencies. Partitioning and scheduling belong to Intro. */
(function (global) {
  'use strict';
  const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch]);
  const kinds = { data: '数据', sync: 'READY 同步', state: '数值状态 / FIFO', reuse: 'Buffer 复用' };
  const labels = { qk: 'QK', softmax: 'Softmax', pv: 'PV', accumulate: 'Accumulate' };
  function mount(host, { context, pipeline, synchronizations, onClose, onNavigate }) {
    const lifetime = new AbortController(), { signal } = lifetime;
    let model, selection, selectedEdge = null, zoom = 1, all = false, signature = '';
    host.innerHTML = `<div class="kf-pa-dependency">
      <header class="kf-pa-dependency__header"><button class="btn btn-ghost btn-sm" data-dep-back>← 返回 Pipeline</button><b>Stack Execution Dependency</b><span data-dep-summary></span></header>
      <div class="kf-pa-dependency__toolbar"><label>当前 Stack <select class="input" data-dep-stack aria-label="依赖图当前 Stack"></select></label><button class="btn btn-ghost btn-sm" data-dep-all aria-pressed="false">显示全部依赖</button><div role="group" aria-label="依赖矩阵缩放"><button class="btn btn-ghost btn-sm" data-dep-zoom="out" aria-label="缩小依赖矩阵">−</button><output data-dep-scale>100%</output><button class="btn btn-ghost btn-sm" data-dep-zoom="in" aria-label="放大依赖矩阵">＋</button><button class="btn btn-ghost btn-sm" data-dep-zoom="fit">适应宽度</button></div></div>
      <div class="kf-pa-dependency__legend">${Object.entries(kinds).map(([key, label]) => `<span class="is-${key}">${key === 'data' ? '—' : key === 'sync' ? '┄' : key === 'state' ? '↳' : '┈'} ${label}</span>`).join('')}<small>仅显示所选阶段相关的状态与复用边</small></div>
      <div class="kf-pa-dependency__viewport" tabindex="0" aria-label="Stack 与计算阶段依赖矩阵，可缩放及滚动"><svg data-dep-svg role="group" aria-label="Stack 实例依赖图"></svg></div>
      <section class="kf-pa-dependency__detail" data-dep-detail aria-live="polite"></section>
      <p class="kf-pa-dependency__note">节点中的 tick 来自源码循环：Produce(tick)，Consume(tick − PRE_LAUNCH)。横向位置表示计算阶段，纵向表示 Stack；均不表示硬件时长。</p>
    </div>`;
    const query = (s) => host.querySelector(s);
    const stages = global.PtoPagedAttentionStackDependencyModel.stages;
    const width = 1240, rowHeight = 210;
    const box = (node) => ({ x: 170 + stages.indexOf(node.stage) * 260, y: 80 + node.stack * rowHeight, w: 152, h: 76 });
    const related = (edge) => {
      const node = model.nodes.find((n) => n.stack === selection.stack && n.stage === selection.stage);
      return !!node && (edge.from === node.id || edge.to === node.id);
    };
    const nodeById = (id) => model.nodes.find((n) => n.id === id);
    const name = (id) => { const node = nodeById(id); return `S${node.stack} ${labels[node.stage]}`; };
    function route(edge) {
      const a = box(nodeById(edge.from)), b = box(nodeById(edge.to));
      if (edge.kind === 'data' || edge.kind === 'sync') {
        const y = a.y + (edge.kind === 'sync' ? 24 : 52);
        return { path: `M${a.x + a.w} ${y} H${b.x}`, x: (a.x + a.w + b.x) / 2, y: a.y + (edge.kind === 'data' ? 98 : -12) };
      }
      if (edge.kind === 'state' && a.y === b.y) {
        const y = a.y + 126;
        return { path: `M${a.x + a.w / 2} ${a.y + a.h} V${y} H${b.x + b.w / 2} V${b.y + b.h}`, x: (a.x + b.x + a.w) / 2, y: y + 18 };
      }
      if (edge.kind === 'reuse') {
        const lane = width - 70 + ['score_transfer', 'probability_transfer', 'pv_transfer'].indexOf(edge.buffer) * 20;
        return { path: `M${a.x + a.w - 14} ${a.y + a.h} V${a.y + a.h + 18} H${lane} V${b.y - 28} H${b.x + 14} V${b.y}`, x: lane - 8, y: b.y - 35 };
      }
      const x = a.x + (nodeById(edge.from).stage === 'softmax' ? 28 : 124);
      return { path: `M${x} ${a.y + a.h} V${b.y}`, x: x + 10, y: (a.y + a.h + b.y) / 2 };
    }
    function paint() {
      const svg = query('[data-dep-svg]'), height = model.stackCount * rowHeight + 90;
      svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
      svg.style.width = `${width * zoom}px`; svg.style.height = `${height * zoom}px`;
      const visibleEdges = model.edges.filter((edge) => ['data', 'sync'].includes(edge.kind) || all || related(edge) || edge.id === selectedEdge);
      svg.innerHTML = `<defs>${Object.keys(kinds).map((kind) => `<marker id="pa-dep-arrow-${kind}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="kf-pa-dependency__arrow is-${kind}" d="M0 0 L10 5 L0 10Z"/></marker>`).join('')}</defs>
        ${stages.map((stage, i) => `<text class="kf-pa-dependency__column" x="${246 + i * 260}" y="30" text-anchor="middle">${labels[stage]} · ${['qk', 'pv'].includes(stage) ? 'AIC' : 'AIV'}</text>`).join('')}
        ${model.stacks.map((stack) => `<g class="kf-pa-dependency__row${selection.stack === stack.id ? ' is-selected' : ''}"><rect x="0" y="${60 + stack.id * rowHeight}" width="${width - 100}" height="${rowHeight - 12}" rx="4"/><text x="12" y="${104 + stack.id * rowHeight}">S${stack.id} · Slot ${stack.slot}</text><text x="12" y="${130 + stack.id * rowHeight}">${stack.pages.reduce((n, p) => n + p.validTokens, 0)} tokens</text><text x="12" y="${154 + stack.id * rowHeight}">${stack.tokenStart}–${stack.tokenEnd}</text></g>`).join('')}
        ${visibleEdges.map((edge) => {
          const r = route(edge), active = related(edge) || edge.id === selectedEdge;
          return `<g class="kf-pa-dependency__edge is-${edge.kind}${active ? ' is-related' : ''}${edge.id === selectedEdge ? ' is-selected' : ''}" data-dep-edge="${esc(edge.id)}" role="button" tabindex="0" aria-label="${esc(`${kinds[edge.kind]}：${name(edge.from)} → ${name(edge.to)}，${edge.label}`)}"><title>${esc(edge.label)}</title><path class="kf-pa-dependency__hit" d="${r.path}"/><path class="kf-pa-dependency__line" d="${r.path}" marker-end="url(#pa-dep-arrow-${edge.kind})"/>${edge.kind !== 'data' || active ? `<text x="${r.x}" y="${r.y}" text-anchor="${['sync', 'data'].includes(edge.kind) ? 'middle' : 'start'}">${esc(edge.label)}</text>${edge.kind === 'sync' ? `<text x="${r.x}" y="${r.y + 17}" text-anchor="middle">SET → WAIT</text>` : ''}` : ''}</g>`;
        }).join('')}
        ${model.nodes.map((node) => {
          const r = box(node), active = node.stack === selection.stack && node.stage === selection.stage;
          return `<g class="kf-pa-dependency__node${active ? ' is-selected' : ''}${node.tick === selection.pipelineStep ? ' is-current-tick' : ''}" data-dep-node="${esc(node.id)}" role="button" tabindex="0" aria-pressed="${active}" aria-label="S${node.stack} ${labels[node.stage]} · ${node.core} · tick ${node.tick} · Slot ${node.slot}"><rect x="${r.x}" y="${r.y}" width="${r.w}" height="${r.h}" rx="4"/><text x="${r.x + 12}" y="${r.y + 25}">${labels[node.stage]} · ${node.core}</text><text x="${r.x + 12}" y="${r.y + 49}">${['qk', 'softmax'].includes(node.stage) ? 'Produce' : 'Consume'} · tick ${node.tick}</text><text x="${r.x + 12}" y="${r.y + 66}">S${node.stack} / Slot ${node.slot}</text></g>`;
        }).join('')}`;
      query('[data-dep-summary]').textContent = `Task ${selection.task} · ${model.seqLen} tokens · ${model.stackCount} Stacks · ${model.stepCount} Steps · ${global.PtoPagedAttentionIntro.model.constants.transferSlots} Slots`;
      query('[data-dep-stack]').innerHTML = (selection.stack === null ? '<option value="" selected disabled>选择 Stack</option>' : '') + model.stacks.map((s) => `<option value="${s.id}" ${s.id === selection.stack ? 'selected' : ''}>S${s.id} · Slot ${s.slot}</option>`).join('');
      query('[data-dep-scale]').textContent = `${Math.round(zoom * 100)}%`;
      query('[data-dep-all]').setAttribute('aria-pressed', String(all));
      query('.kf-pa-dependency__legend small').textContent = all ? '显示全部静态依赖；点击阶段可突出相关关系' : '仅显示所选阶段相关的状态与复用边';
      explain();
    }
    function explain() {
      const node = model.nodes.find((n) => n.stack === selection.stack && n.stage === selection.stage);
      const edge = model.edges.find((e) => e.id === selectedEdge);
      const edges = model.edges.filter((e) => node && (e.from === node.id || e.to === node.id));
      const sourceLinks = (source) => (source?.ranges || []).map((range) => `<button class="btn btn-ghost btn-sm" data-dep-source-line="${range.start}">${esc(source.file)}:${range.start}–${range.end}</button>`).join('');
      query('[data-dep-detail]').innerHTML = edge
        ? `<header><b>${esc(kinds[edge.kind])} · ${esc(name(edge.from))} → ${esc(name(edge.to))}</b><button class="btn btn-ghost btn-sm" data-dep-clear>返回阶段详情</button></header><p>${esc(edge.explanation)}</p>${edge.versions?.length ? `<p class="kf-pa-dependency__versions">${edge.versions.map(esc).join(' → ')}</p>` : ''}${edge.synchronization ? `<p>${esc(edge.synchronization.name)} #${edge.synchronization.eventId} · ${esc(edge.synchronization.buffer)} · ${esc(edge.synchronization.dtype)}<br>SET ${edge.synchronization.producer.coreType}/${edge.synchronization.producer.pipe} → WAIT ${edge.synchronization.consumer.coreType}/${edge.synchronization.consumer.pipe}</p>${sourceLinks(edge.synchronization.producer.source)}${sourceLinks(edge.consumerSource || edge.synchronization.consumer.source)}` : sourceLinks(edge.source)}`
        : node ? `<header><b>S${node.stack} · ${labels[node.stage]} · tick ${node.tick}</b><div><button class="btn btn-ghost btn-sm" data-dep-navigate="compute">定位计算流程 ↗</button><button class="btn btn-ghost btn-sm" data-dep-navigate="source">定位 PyPTO 源码 ↗</button></div></header><p>${node.stage === 'qk' ? '读取当前 Q 与分页 K；不同 Stack 的 QK 不依赖彼此的数值状态。' : node.stage === 'softmax' ? (node.stack === 0 ? '等待本 Stack 的 Scores；初始状态 m = −3.0e38、l = 0。概率写回后通知 PV。' : `等待本 Stack 的 Scores，读取 S${node.stack - 1} 更新后的 m/l；概率写回后通知 PV。`) : node.stage === 'pv' ? '等待本 Stack 的概率 Buffer；读取 V 生成 partial，不依赖前一 Stack 的 Context 数值。' : (node.stack === 0 ? '等待 S0 的 PV partial；Context 从 o = 0 开始，并读取经 FIFO 对齐的 r0。' : `等待本 Stack 的 PV partial，读取 S${node.stack - 1} 更新后的 o 与经 FIFO 对齐的 r${node.stack}。`)}</p><nav aria-label="当前阶段的依赖">${edges.map((e) => `<button class="btn btn-ghost btn-sm is-${e.kind}" data-dep-edge="${esc(e.id)}">${e.to === node.id ? '← 等待 / 读取' : '→ 供后续使用'} · ${esc(e.label)} · ${esc(name(e.to === node.id ? e.from : e.to))}</button>`).join('')}</nav><p>${model.edges.some((e) => e.kind === 'reuse') ? '复用边表示同一 Slot 的旧数据读完后才可覆盖；不会把所有计算阶段串行化。' : '当前各 Stack 使用不同 Slot，没有 Buffer 复用依赖。'} AIC/AIV 可处理不同 Stack；图中未连接的阶段不因此被强制串行。</p>` : '<p>选择一个阶段，查看它等待的数据与状态；或点击依赖边查看版本和源码。</p>';
    }
    function update(nextPipeline = pipeline) {
      pipeline = nextPipeline;
      const next = context.selection;
      const key = JSON.stringify([pipeline.seqLen, next.task, next.stack, next.stage, next.pipelineStep]);
      if (key === signature) return;
      signature = key;
      model = global.PtoPagedAttentionStackDependencyModel.build(pipeline, synchronizations);
      selection = next;
      selectedEdge = null;
      paint();
    }
    host.addEventListener('click', (event) => {
      if (event.target.closest('[data-dep-back]')) { onClose(); return; }
      const cell = event.target.closest('[data-dep-node]');
      if (cell) { const node = nodeById(cell.dataset.depNode); selectedEdge = null; context.selectInstance({ stack: node.stack, stage: node.stage }); update(); return; }
      const edge = event.target.closest('[data-dep-edge]');
      if (edge) { selectedEdge = edge.dataset.depEdge; paint(); return; }
      if (event.target.closest('[data-dep-clear]')) { selectedEdge = null; paint(); return; }
      const source = event.target.closest('[data-dep-source-line]');
      if (source) { onNavigate('source', Number(source.dataset.depSourceLine)); return; }
      const nav = event.target.closest('[data-dep-navigate]');
      if (nav) { onNavigate(nav.dataset.depNavigate); return; }
      if (event.target.closest('[data-dep-all]')) { all = !all; paint(); return; }
      const scale = event.target.closest('[data-dep-zoom]');
      if (scale) { zoom = scale.dataset.depZoom === 'fit' ? Math.max(.35, Math.min(1, query('.kf-pa-dependency__viewport').clientWidth / width)) : Math.max(.35, Math.min(1.8, zoom * (scale.dataset.depZoom === 'in' ? 1.2 : 1 / 1.2))); paint(); }
    }, { signal });
    host.addEventListener('change', (event) => {
      if (event.target.matches('[data-dep-stack]')) { selectedEdge = null; context.selectInstance({ stack: Number(event.target.value), stage: selection.stage }); update(); }
    }, { signal });
    host.addEventListener('keydown', (event) => {
      const item = event.target.closest('g[data-dep-node], g[data-dep-edge]');
      if (item && ['Enter', ' '].includes(event.key)) { event.preventDefault(); item.dispatchEvent(new MouseEvent('click', { bubbles: true })); }
    }, { signal });
    query('.kf-pa-dependency__viewport').addEventListener('wheel', (event) => {
      if (!event.ctrlKey && !event.metaKey) return;
      event.preventDefault(); zoom = Math.max(.35, Math.min(1.8, zoom * (event.deltaY < 0 ? 1.1 : 1 / 1.1))); paint();
    }, { signal, passive: false });
    update();
    return { update, abort: () => lifetime.abort() };
  }
  global.PtoPagedAttentionStackDependency = { mount };
})(window);
