/* Persistent Task identities in a shared KV-column space. Morph changes Y only. */
(function (global) {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const clamp = value => Math.max(0, Math.min(1, value));
  const ease = t => t * t * (3 - 2 * t);
  const svg = (tag, attrs, parent) => {
    const node = document.createElementNS(NS, tag);
    for (const [key, value] of Object.entries(attrs || {})) node.setAttribute(key, value);
    parent.append(node);
    return node;
  };

  function createLayout({ batch, kvHeads, blocks }) {
    const width = 1250, height = 600, top = 58, bottom = 580;
    const columnPitch = 1048 / kvHeads, frameWidth = columnPitch - 10;
    const columns = Array.from({ length: kvHeads }, (_, kv) => ({ kv, x: 172 + kv * columnPitch, width: frameWidth }));
    const assignments = Array.from({ length: blocks }, () => []);
    for (let id = 0; id < batch * kvHeads; id++) assignments[id % blocks].push(id);
    // Whole KV cycles produce one Block per KV column in each Batch-residue group.
    // For other configurations, each Block gets a full-width band, while Task X
    // still stays in its KV column and modulo ownership remains exact.
    const aligned = blocks % kvHeads === 0;
    const groupCount = aligned ? blocks / kvHeads : blocks;
    const capacities = Array.from({ length: groupCount }, (_, index) => aligned
      ? Math.max(1, Math.ceil((batch - index) / groupCount))
      : Math.max(1, ...columns.map(({ kv }) => assignments[index].filter(id => id % kvHeads === kv).length)));
    const groupGap = 12, header = 24, padding = 7;
    const pitch = Math.min(25, (bottom - top - groupGap * (groupCount - 1) - (header + padding) * groupCount) / capacities.reduce((sum, n) => sum + n, 0));
    const taskHeight = Math.min(22, (bottom - top - 16) / batch - 5, pitch - 3);
    let y = top;
    const groups = capacities.map((capacity, index) => {
      const group = { index, y, height: header + capacity * pitch + padding, capacity,
        label: aligned ? `B%${groupCount}=${index}` : `Task%${blocks}=${index}` };
      y += group.height + groupGap;
      return group;
    });
    const sources = [], targets = [];
    for (let id = 0; id < batch * kvHeads; id++) {
      const batchId = Math.floor(id / kvHeads), kv = id % kvHeads, block = id % blocks;
      const group = groups[aligned ? Math.floor(block / kvHeads) : block];
      const index = aligned ? Math.floor(batchId / groupCount) : assignments[block].filter(task => task % kvHeads === kv).indexOf(id);
      const geometry = { x: columns[kv].x + 7, w: frameWidth - 14, h: taskHeight };
      sources.push({ ...geometry, y: top + 8 + batchId * ((bottom - top - 16) / batch) });
      targets.push({ ...geometry, y: group.y + header + index * pitch });
    }
    const blockRects = assignments.map((tasks, block) => {
      const group = groups[aligned ? Math.floor(block / kvHeads) : block];
      return { x: aligned ? columns[block % kvHeads].x : columns[0].x, y: group.y,
        w: aligned ? frameWidth : columns.at(-1).x + frameWidth - columns[0].x, h: group.height };
    });
    return { width, height, columns, groups, sources, targets, assignments, blockRects, aligned };
  }

  function mount(host, { config, seqLens, onPause, onEnter }) {
    const { batch, kvHeads, blocks, qPerKv } = config;
    const count = batch * kvHeads, layout = createLayout(config);
    host.innerHTML = `<div class="viz2-am-head"><div><h2 data-am-title>Task Generation · ${count} Tasks</h2><p data-am-explain>${batch} Batches × ${kvHeads} KV Heads → ${count} Tasks</p></div><div class="viz2-am-controls"><button type="button" class="btn btn-ghost btn-sm" data-am-play>暂停</button><button type="button" class="btn btn-ghost btn-sm" data-am-replay>重播</button><div class="segmented-control segmented-control-muted"><button type="button" class="btn btn-sm" data-am-state="generation" aria-pressed="true">Task Generation</button><button type="button" class="btn btn-sm" data-am-state="assignment" aria-pressed="false">Block Assignment</button></div><label><input type="checkbox" data-am-loop checked> 循环播放</label><button type="button" class="btn btn-ghost btn-sm" data-am-fit>适应</button></div></div><div class="kf-pa-map__viewport viz2-am-viewport" tabindex="0" role="group" aria-label="固定 KV 列的 Task 重排画布"><svg class="kf-pa-map__svg viz2-am-svg" viewBox="0 0 ${layout.width} ${layout.height}" preserveAspectRatio="xMidYMid meet" aria-label="Batch 行与 KV Head 列的任务关系"><g class="kf-pa-map__scene"><g data-am-columns></g><g data-am-generation></g><g data-am-blocks></g><g data-am-group-labels></g><g data-am-tasks></g><g data-am-block-hits></g></g></svg><div class="kf-pa-map__tooltip" role="tooltip" hidden></div></div><footer class="viz2-am-footer"><p data-viz2-selection aria-live="polite">每个 Task 对应 ${qPerKv} 个 Q Head</p><span>双击 Task 查看内部 · 静态映射，不表示执行时序</span><button type="button" class="btn btn-ghost btn-sm" data-am-enter hidden>查看内部</button></footer>`;
    const viewport = host.querySelector('.viz2-am-viewport'), scene = host.querySelector('.kf-pa-map__scene');
    const generation = host.querySelector('[data-am-generation]'), blockLayer = host.querySelector('[data-am-blocks]');
    const groupLabels = host.querySelector('[data-am-group-labels]'), taskLayer = host.querySelector('[data-am-tasks]');
    const columnLayer = host.querySelector('[data-am-columns]'), hitLayer = host.querySelector('[data-am-block-hits]');
    layout.columns.forEach(({ kv, x, width }) => {
      svg('text', { x: x + width / 2, y: 30, class: 'viz2-am-kv', 'data-am-kv': kv }, columnLayer).textContent = `KV${kv}`;
    });
    svg('line', { x1: layout.columns[0].x, x2: layout.columns.at(-1).x + layout.columns.at(-1).width, y1: 44, y2: 44, class: 'viz2-am-column-rule' }, columnLayer);
    svg('text', { x: 8, y: 30, class: 'viz2-am-axis' }, generation).textContent = 'BATCH';
    svg('text', { x: 80, y: 30, class: 'viz2-am-axis' }, generation).textContent = 'seq_lens';
    const segments = Math.max(1, Math.ceil(Math.max(...seqLens) / 512));
    for (let b = 0; b < batch; b++) {
      const row = svg('g', { 'data-pa-column': b, 'data-am-batch': b }, generation), y = layout.sources[b * kvHeads].y, len = seqLens[b] || 0;
      svg('text', { x: 8, y: y + layout.sources[0].h / 2, class: 'viz2-am-batch' }, row).textContent = `B${b}`;
      svg('text', { x: 75, y: y + layout.sources[0].h / 2, class: 'viz2-am-seq' }, row).textContent = len;
      // Keep the sequence snapshot contract used by Anatomy; metadata now follows Batch rows.
      const seq = svg('g', { class: 'kf-pa-map__sequence', 'aria-label': `seq_lens ${len}` }, row);
      for (let i = 0; i < segments; i++) {
        const width = 68 / segments, x = 87 + i * width;
        svg('rect', { x, y: y + layout.sources[0].h / 2 - 3, width: width - 2, height: 6, class: 'kf-pa-map__seq-track' }, seq);
        svg('rect', { x, y: y + layout.sources[0].h / 2 - 3, width: (width - 2) * clamp((len - i * 512) / 512), height: 6, class: 'kf-pa-map__seq-fill' }, seq);
      }
    }
    layout.groups.forEach(group => {
      svg('text', { x: 8, y: group.y + group.height / 2, class: 'viz2-am-group-label', 'data-am-group': group.index }, groupLabels).textContent = group.label;
    });
    const blockNodes = layout.blockRects.map((r, block) => {
      const group = svg('g', { 'data-pa-block': block, class: 'viz2-am-block', role: 'button', tabindex: '-1', 'aria-label': `SPMD Block ${block}: Tasks ${layout.assignments[block].join(', ')}` }, blockLayer);
      svg('rect', { x: r.x, y: r.y, width: r.w, height: r.h, rx: 2, class: 'viz2-am-block-frame' }, group);
      svg('text', { x: r.x + 7, y: r.y + 15, class: 'viz2-am-block-label' }, group).textContent = `BLK ${String(block).padStart(2, '0')}`;
      return group;
    });
    const nodes = layout.sources.map((r, id) => {
      const group = svg('g', { 'data-pa-task': id, 'data-am-kv-column': id % kvHeads, class: 'kf-pa-map__task viz2-am-task', role: 'button', tabindex: id === 0 ? '0' : '-1', 'aria-label': `Task ${id}, Batch ${Math.floor(id / kvHeads)}, KV Head ${id % kvHeads}, Block ${id % blocks}` }, taskLayer);
      svg('rect', { class: 'kf-pa-map__cell', width: r.w, height: r.h, rx: 3 }, group);
      for (let q = 0; q < qPerKv; q++) svg('rect', { class: 'kf-pa-map__q', x: 4 + q * 8, y: 3, width: 5, height: Math.max(1, r.h - 6), rx: 1 }, group);
      svg('text', { x: r.w / 2, y: r.h / 2, class: 'kf-pa-map__task-id' }, group).textContent = id;
      return group;
    });
    const hits = layout.blockRects.map((r, block) => svg('rect', { x: r.x, y: r.y, width: r.w, height: 22, fill: 'transparent', class: 'viz2-am-block-hit', 'data-am-block-hit': block, role: 'button', tabindex: '-1', 'aria-label': `选择 Block ${block}` }, hitLayer));
    const media = matchMedia('(prefers-reduced-motion: reduce)');
    let progress = 0, target = 0, running = false, frame = 0, timer = 0, active = false, loop = true, selection = null, held = false;
    const play = host.querySelector('[data-am-play]');
    function describe(p) {
      const assignment = p >= .5;
      host.querySelector('[data-am-title]').textContent = assignment ? `Block Assignment · ${blocks} SPMD Blocks` : `Task Generation · ${count} Tasks`;
      host.querySelector('[data-am-explain]').textContent = assignment ? `Block ID = Task ID % ${blocks} · KV 列固定，按 Batch 余数聚合` : `${batch} Batches × ${kvHeads} KV Heads → ${count} Tasks`;
      host.querySelectorAll('[data-am-state]').forEach(button => {
        const selected = button.dataset.amState === (assignment ? 'assignment' : 'generation');
        button.setAttribute('aria-pressed', String(selected)); button.classList.toggle('is-selected', selected);
      });
    }
    function draw(p) {
      progress = clamp(p);
      host.dataset.amProgress = String(progress);
      generation.style.opacity = String(clamp(1 - progress / .2));
      const alpha = clamp((progress - .82) / .18);
      blockLayer.style.opacity = groupLabels.style.opacity = String(alpha);
      blockLayer.style.pointerEvents = alpha > .8 ? '' : 'none';
      hitLayer.style.pointerEvents = alpha > .8 ? '' : 'none';
      hits.forEach(hit => hit.setAttribute('tabindex', alpha > .8 ? '0' : '-1'));
      nodes.forEach((node, id) => {
        const from = layout.sources[id], to = layout.targets[id];
        const y = from.y + (to.y - from.y) * progress;
        // X, size, stripes and label identity are invariant for the whole morph.
        node.setAttribute('transform', `translate(${from.x} ${y})`);
      });
      describe(progress);
    }
    function clear() { clearTimeout(timer); cancelAnimationFrame(frame); timer = frame = 0; running = false; }
    function pause() { clear(); held = true; play.textContent = '播放'; onPause?.(); }
    function animateTo(next, duration, done) {
      clear(); target = next;
      if (media.matches || duration === 0 || progress === next) { draw(next); done?.(); return; }
      running = true;
      const start = performance.now(), from = progress;
      const tick = now => {
        const t = clamp((now - start) / duration);
        draw(from + (next - from) * ease(t));
        if (t < 1) frame = requestAnimationFrame(tick);
        else { running = false; frame = 0; done?.(); }
      };
      frame = requestAnimationFrame(tick);
    }
    function schedule() {
      if (!active || held || !loop || document.hidden) { if (!loop) { held = true; play.textContent = '播放'; } return; }
      clearTimeout(timer);
      timer = setTimeout(() => {
        if (!active || held || document.hidden) return;
        target = progress < .5 ? 1 : 0;
        animateTo(target, target ? 2800 : 2400, schedule);
      }, progress < .5 ? 2000 : 3600);
    }
    function manual(next) { pause(); animateTo(next, Math.max(300, Math.abs(next - progress) * 1400)); }
    play.addEventListener('click', () => {
      if (running || (!held && timer)) { pause(); return; }
      held = false; play.textContent = '暂停';
      const next = progress > 0 && progress < 1 ? target : progress === 0 ? 1 : 0;
      animateTo(next, Math.max(300, Math.abs(next - progress) * 2500), schedule);
    });
    host.querySelector('[data-am-replay]').addEventListener('click', () => { clear(); target = 0; draw(0); held = false; play.textContent = '暂停'; schedule(); });
    host.querySelectorAll('[data-am-state]').forEach(button => button.addEventListener('click', () => manual(button.dataset.amState === 'assignment' ? 1 : 0)));
    host.querySelector('[data-am-loop]').addEventListener('change', event => {
      loop = event.target.checked;
      if (!loop) { clearTimeout(timer); timer = 0; if (!running) { held = true; play.textContent = '播放'; } }
      else if (active && !held && !running) schedule();
    });
    viewport.addEventListener('pointerover', event => { if (event.target.closest('[data-pa-task],[data-pa-block],[data-am-block-hit]')) pause(); });
    let zoom = 1, panX = 0, panY = 0, drag = null, suppressClick = false;
    const camera = () => scene.setAttribute('transform', `translate(${panX} ${panY}) translate(625 300) scale(${zoom}) translate(-625 -300)`);
    function fit() { pause(); zoom = 1; panX = panY = 0; camera(); }
    host.querySelector('[data-am-fit]').addEventListener('click', fit);
    viewport.addEventListener('wheel', event => {
      event.preventDefault(); pause();
      const next = Math.max(.8, Math.min(3, zoom * Math.exp(-event.deltaY * .001)));
      const task = event.target.closest('[data-pa-task]');
      if (zoom < 2.8 && next >= 2.8 && task && onEnter) { onEnter(Number(task.dataset.paTask)); return; }
      zoom = next; camera();
    }, { passive: false });
    viewport.addEventListener('pointerdown', event => { pause(); if (event.button === 0 && !event.target.closest('[data-pa-task],[data-am-block-hit]')) drag = { x: event.clientX, y: event.clientY, id: event.pointerId, moved: false }; });
    viewport.addEventListener('pointermove', event => {
      if (!drag) return;
      const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 2) {
        drag.moved = true; viewport.setPointerCapture(drag.id);
        const bounds = viewport.getBoundingClientRect(), unit = Math.min(bounds.width / layout.width, bounds.height / layout.height);
        panX += dx / unit; panY += dy / unit; camera(); drag.x = event.clientX; drag.y = event.clientY;
      }
    });
    viewport.addEventListener('pointerup', () => { suppressClick = !!drag?.moved; drag = null; });
    viewport.addEventListener('pointercancel', () => { drag = null; });
    viewport.addEventListener('click', event => { if (suppressClick) { event.stopPropagation(); suppressClick = false; } }, true);
    viewport.addEventListener('keydown', pause);
    document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
    media.addEventListener('change', () => { if (media.matches) { clear(); draw(progress < .5 ? 0 : 1); } });
    new ResizeObserver(() => {
      const r = viewport.getBoundingClientRect(), unit = Math.min(r.width / layout.width, r.height / layout.height);
      if (unit > 0) scene.style.setProperty('--am-label-size', `${Math.max(12, 11 / unit)}px`);
    }).observe(viewport);
    function update(next, visible) {
      selection = next;
      nodes.forEach((node, id) => {
        node.classList.toggle('is-selected', id === selection?.task);
        node.classList.toggle('is-assigned', selection?.block != null && id % blocks === selection.block);
        node.setAttribute('aria-pressed', String(id === selection?.task));
      });
      blockNodes.forEach((node, id) => { node.classList.toggle('is-selected', id === selection?.block); node.setAttribute('aria-pressed', String(id === selection?.block)); });
      host.querySelector('[data-am-enter]').hidden = selection?.kind !== 'task';
      if (!visible) { clear(); active = false; }
      else if (!active) { active = true; draw(progress < .5 ? 0 : 1); if (!held) schedule(); }
    }
    draw(0);
    return { update, pause, manual, fit, settle() { clear(); draw(progress < .5 ? 0 : 1); }, get progress() { return progress; }, get viewport() { return viewport; }, get scene() { return scene; }, get tooltip() { return host.querySelector('[role="tooltip"]'); } };
  }
  global.PtoVisualization2AssignmentMorph = Object.freeze({ mount, createLayout });
})(window);
