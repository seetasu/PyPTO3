(() => {
  const frame = document.querySelector('[data-ide-frame]');
  const traceCanvas = document.querySelector('#trace-canvas');
  const distributionCanvas = document.querySelector('#distribution-canvas');
  const status = document.querySelector('#status-message');
  const filterButtons = [...document.querySelectorAll('[data-lane-filter]')];
  let activeFilter = 'all';
  let selectedEvent = null;
  let hitAreas = [];

  const events = [
    ...[182, 994, 894, 749, 673, 707, 676, 681, 661, 662, 689, 699, 649, 682, 661, 649, 649, 702, 640, 662, 646, 673, 690, 666, 645, 676, 647, 657, 624].map((duration, index) => ({
      id: `graph-${index}`, lane: 'graph', name: 'npu_fx_compiler inference', start: 19.13 + index * .77, duration: duration / 1000, thread: 'Python · 2669105',
      title: 'Host-side graph path is serial', summary: '29 calls occupy roughly 19.5 s of the active window. The trace name alone cannot prove repeated compilation, but it is a rank-local critical-path candidate.'
    })),
    { id: 'compute-a', lane: 'compute', name: 'vllm::moe_forward_shared', start: 19.13, duration: 8.43, thread: 'NPU 0 · compute', title: 'MoE shared forward is a major compute carrier', summary: 'The trace reports 1,535 calls and 14.32 s of inclusive time. Inclusive CPU-op time must not be summed with communication to estimate wall time.' },
    { id: 'compute-b', lane: 'compute', name: 'vllm::dsa_forward', start: 30.7, duration: 4.11, thread: 'NPU 0 · compute', title: 'Attention path remains active through the decode window', summary: '1,535 DSA calls accumulate 6.30 s. Compare this rank against peers before identifying a cluster-wide compute imbalance.' },
    { id: 'comm-a', lane: 'communication', name: 'Communication (Not Overlapped)', start: 23.2, duration: 12.31, thread: 'NPU 0 · communication', title: 'Communication is classified as mostly non-overlapped', summary: 'The profiler classifies 12.31 s as non-overlapped communication, nearly the full 12.41 s communication total. This warrants a TP/EP overlap experiment.' },
    { id: 'comm-b', lane: 'communication', name: 'gloo:all_reduce', start: 37.1, duration: 3.2, thread: 'NPU 0 · communication', title: 'Collective calls add a visible dependency', summary: '192 all-reduce calls contribute 1.26 s inclusively; the 250 ms maximum is more useful as a long-tail lead than a direct wall-time total.' },
    { id: 'wait-a', lane: 'wait', name: 'EVENT_WAIT', start: 20.0, duration: 8.8, thread: 'NPU 0 · stream 38', title: 'Device dependency wait is visible', summary: 'EVENT_WAIT spans multiple streams. Treat its inclusive total as dependency evidence, not as a directly additive idle-time percentage.' },
    { id: 'wait-b', lane: 'wait', name: 'NOTIFY_WAIT_SQE', start: 34.1, duration: 9.4, thread: 'NPU 0 · stream 182', title: 'Stream notification waits are concentrated', summary: 'This wait activity follows the communication-heavy section. Compare event dependencies across ranks before pinning it on a local operator.' }
  ];

  const colorFor = (lane) => {
    const map = window.PtoSwimlaneTaskPattern?.createTaskColormap?.();
    return map?.colorForTask({ colorKey: lane }, 'semantic') || getComputedStyle(document.documentElement).getPropertyValue('--primary');
  };

  function textColor(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function resizeCanvas(canvas, height) {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(height * dpr));
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width: rect.width, height };
  }

  function renderTrace() {
    const height = 232;
    const { ctx, width } = resizeCanvas(traceCanvas, height);
    const muted = textColor('--foreground-muted');
    const border = textColor('--border-subtle');
    const foreground = textColor('--foreground-secondary');
    const labelHeight = 26;
    const laneHeight = 46;
    const start = 19;
    const end = 49;
    const pad = 2;
    hitAreas = [];
    ctx.clearRect(0, 0, width, height);
    ctx.font = '11px var(--font-mono)';
    ctx.fillStyle = muted;
    for (let second = 20; second < 50; second += 5) {
      const x = ((second - start) / (end - start)) * width;
      ctx.fillText(`${second}s`, x + 3, 12);
      ctx.strokeStyle = border;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x + .5, labelHeight); ctx.lineTo(x + .5, height); ctx.stroke();
    }
    ['graph', 'compute', 'communication', 'wait'].forEach((lane, index) => {
      const y = labelHeight + index * laneHeight;
      ctx.strokeStyle = border;
      ctx.beginPath(); ctx.moveTo(0, y + laneHeight - 4 + .5); ctx.lineTo(width, y + laneHeight - 4 + .5); ctx.stroke();
    });
    const visible = events.filter((event) => activeFilter === 'all' || event.lane === activeFilter);
    visible.forEach((event) => {
      const laneIndex = ['graph', 'compute', 'communication', 'wait'].indexOf(event.lane);
      const x = Math.max(0, ((event.start - start) / (end - start)) * width);
      const eventWidth = Math.max(4, event.duration / (end - start) * width);
      const y = labelHeight + laneIndex * laneHeight + pad;
      const h = laneHeight - 11;
      const selected = selectedEvent?.id === event.id;
      window.PtoSwimlaneTaskPattern.drawTaskBar(ctx, {
        x, y, width: eventWidth, height: h, baseColor: colorFor(event.lane), task: { displayName: event.name, laneKind: event.lane, duration: event.duration * 1000, status: event.lane === 'wait' ? 'wait' : '' }, isSelected: selected, isEmphasized: selected, fontFamily: 'var(--font-mono)'
      });
      hitAreas.push({ event, x, y, width: eventWidth, height: h });
    });
    if (!visible.length) {
      ctx.fillStyle = foreground;
      ctx.font = '14px var(--font-sans)';
      ctx.fillText('该视图没有匹配事件。', 16, height / 2);
    }
  }

  function renderDistribution() {
    const height = Math.max(82, distributionCanvas.getBoundingClientRect().height || 82);
    const { ctx, width } = resizeCanvas(distributionCanvas, height);
    const muted = textColor('--foreground-muted');
    const border = textColor('--border-subtle');
    const primary = textColor('--primary');
    const warning = textColor('--warning');
    const bins = [
      { label: 'TTFT\n0–.75', value: 1, color: primary }, { label: '.75–1.5', value: 0, color: primary }, { label: '1.5–2.25', value: 102, color: primary }, { label: '2.25–3', value: 123, color: primary }, { label: '3–3.5 s', value: 30, color: primary },
      { label: 'E2E\n<22', value: 2, color: warning }, { label: '22–25', value: 24, color: warning }, { label: '25–26.5', value: 66, color: warning }, { label: '26.5–27', value: 78, color: warning }, { label: '27–28 s', value: 86, color: warning }
    ];
    ctx.clearRect(0, 0, width, height);
    const chartHeight = height - 34;
    const gap = 8;
    const barWidth = (width - gap * (bins.length - 1)) / bins.length;
    const max = 128;
    ctx.strokeStyle = border; ctx.beginPath(); ctx.moveTo(0, chartHeight + .5); ctx.lineTo(width, chartHeight + .5); ctx.stroke();
    bins.forEach((bin, index) => {
      const x = index * (barWidth + gap);
      const h = Math.max(bin.value ? 3 : 1, bin.value / max * (chartHeight - 12));
      ctx.fillStyle = bin.color;
      ctx.globalAlpha = index < 5 ? .75 : .82;
      ctx.fillRect(x, chartHeight - h, barWidth, h);
      ctx.globalAlpha = 1;
      ctx.fillStyle = muted; ctx.font = '11px var(--font-mono)'; ctx.textAlign = 'center';
      bin.label.split('\n').forEach((line, lineIndex) => ctx.fillText(line, x + barWidth / 2, chartHeight + 13 + lineIndex * 11));
    });
    ctx.textAlign = 'left';
  }

  function formatTime(start, duration) { return `${start.toFixed(2)}–${(start + duration).toFixed(2)} s`; }
  function selectEvent(event) {
    selectedEvent = event;
    document.querySelector('#selected-name').textContent = event.name;
    document.querySelector('#selected-time').textContent = formatTime(event.start, event.duration);
    document.querySelector('#selected-duration').textContent = `${(event.duration * 1000).toFixed(1)} ms`;
    document.querySelector('#selected-thread').textContent = event.thread;
    document.querySelector('#selected-title').textContent = event.title;
    document.querySelector('#selected-summary').textContent = event.summary;
    status.textContent = `Selected: ${event.name} · local evidence only`;
    renderTrace();
  }

  const problems = {
    ttft: { filter: 'all', title: '区分 burst admission 与计算延迟', body: '请求只在约 39 ms 内提交，但后半 cohort 的 TTFT 更高。用受控 request rate 重跑，并比较 admission queue。' },
    decode: { filter: 'compute', title: '拆分慢 decode 的两个阶段', body: '前 24 个记录间隔均值 783 ms，后续降至 121 ms。以服务端 token usage 核对 chunk 与 token 语义。' },
    communication: { filter: 'communication', title: '验证 TP / EP 通信与计算重叠', body: '采集所有 rank 的对齐 Trace 与 collective 指标；当前 NPU 0 只能说明本 Rank 的通信依赖明显。' },
    speculation: { filter: 'all', title: '对比 speculative decoding on / off', body: '草稿 token 99,680、接受 1,990，重算接受率 1.996%。控制 draft 长度和 warmup 后再比较 TPOT。' }
  };
  function applyProblem(problem) {
    const item = problems[problem];
    if (!item) return;
    document.querySelector('#hypothesis-title').textContent = item.title;
    document.querySelector('#hypothesis-body').textContent = item.body;
    filterButtons.find((button) => button.dataset.laneFilter === item.filter)?.click();
    status.textContent = `Focused investigation: ${item.title}`;
  }

  traceCanvas.addEventListener('click', (event) => {
    const rect = traceCanvas.getBoundingClientRect();
    const x = event.clientX - rect.left; const y = event.clientY - rect.top;
    const hit = hitAreas.find((area) => x >= area.x && x <= area.x + area.width && y >= area.y && y <= area.y + area.height);
    if (hit) selectEvent(hit.event);
  });
  filterButtons.forEach((button) => button.addEventListener('click', () => {
    activeFilter = button.dataset.laneFilter;
    filterButtons.forEach((candidate) => candidate.classList.toggle('is-selected', candidate === button));
    document.querySelector('#trace-summary').textContent = activeFilter === 'all' ? '29 graph calls · 4 active lanes' : `${activeFilter} events · rank-local slice`;
    renderTrace();
  }));
  document.querySelectorAll('[data-problem]').forEach((button) => button.addEventListener('click', () => applyProblem(button.dataset.problem)));
  document.querySelectorAll('[data-request-group]').forEach((button) => button.addEventListener('click', () => {
    status.textContent = `Cohort selected: ${button.textContent.trim().replace(/\s+/g, ' ')}`;
    document.querySelector('#hypothesis-title').textContent = '将 cohort 作为可比较样本';
    document.querySelector('#hypothesis-body').textContent = '保持相同输入/输出长度，导出这组请求与匹配时间窗，避免把 cohort 差异误判为全局性能变化。';
  }));
  document.querySelectorAll('[data-tab]').forEach((button) => button.addEventListener('click', () => {
    document.querySelectorAll('[data-tab]').forEach((candidate) => { candidate.classList.toggle('is-selected', candidate === button); candidate.setAttribute('aria-selected', String(candidate === button)); });
    status.textContent = `${button.textContent} view retained in the same rank scope`;
  }));
  document.querySelector('#validate-button').addEventListener('click', (event) => {
    event.currentTarget.textContent = '已加入验证清单';
    event.currentTarget.classList.add('is-selected');
    status.textContent = 'Validation intent recorded locally in this demo';
  });
  document.querySelectorAll('[data-search-trigger]').forEach((button) => button.addEventListener('click', () => {
    status.textContent = 'Search target: npu_fx_compiler inference · 29 matches in this trace';
    selectEvent(events.find((event) => event.id === 'graph-1'));
  }));
  document.querySelector('[data-evidence-trigger]').addEventListener('click', () => { status.textContent = 'Evidence chain: result3.json → Chrome Trace → selected rank-local event'; });
  window.addEventListener('resize', () => { renderTrace(); renderDistribution(); });
  frame.addEventListener('pointermove', (event) => {
    const rect = frame.getBoundingClientRect();
    frame.style.setProperty('--ide-cursor-x', `${event.clientX - rect.left}px`);
    frame.style.setProperty('--ide-cursor-y', `${event.clientY - rect.top}px`);
    frame.style.setProperty('--ide-cursor-alpha', '1');
    frame.style.setProperty('--ide-dot-opacity', '.45');
  });
  frame.addEventListener('pointerleave', () => { frame.style.setProperty('--ide-cursor-alpha', '0'); frame.style.setProperty('--ide-dot-opacity', '0'); });

  window.PtoIdeFrame?.init(frame);
  selectEvent(events.find((event) => event.id === 'graph-1'));
  renderDistribution();
})();
