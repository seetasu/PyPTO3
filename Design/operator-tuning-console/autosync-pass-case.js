(function () {
  'use strict';

  const workloads = {
    gemm: {
      name: 'GEMM', issue: '#643', issueUrl: 'https://github.com/hw-native-sys/PTOAS/issues/643',
      meta: '6144³ · A2/A3 · 双缓冲 K-loop', title: 'AutoSync 同步密度实验',
      subtitle: '改变同步插入与控制流结构，观察 load / Cube 阶段是否重新重叠。',
      base: 120.94, reference: 329.52, unit: 'TFLOPS', baseLatency: 3.836, refLatency: 1.408,
      passes: [
        { id: 'autosync', name: 'AutoSync 插入', detail: '自动插入 wait_flag / pipe_barrier', default: true },
        { id: 'merge', name: '合并 ping/pong 分支', detail: '减少分支边界与同步外提阻碍', default: false },
        { id: 'hoist', name: '同步外提', detail: '将可共享的 wait 移到分支前', default: false },
        { id: 'unroll', name: '展开 K-loop', detail: '消除循环控制流依赖', default: false },
      ],
      issueData: [['自动生成', '120.94 TFLOPS'], ['torch_npu', '329.52 TFLOPS'], ['手写 A2/A3', '约 1.506 ms']],
      code: ['TLOAD tile[k+1]', 'wait_flag(MTE2 → Cube)', 'pipe_barrier(PIPE_ALL)', 'TMATMUL tile[k]', 'wait_flag(Cube → MTE3)', 'TSTORE tile[k-1]'],
    },
    flash: {
      name: 'FlashAttention', issue: '#646', issueUrl: 'https://github.com/hw-native-sys/PTOAS/issues/646',
      meta: 'S1=131072 · vector normalize · PIPE_V', title: 'Vector barrier 依赖实验',
      subtitle: '切换 vector barrier 处理方式，观察 normalize 阶段的等待和吞吐。',
      base: 143.5, reference: 164.18, unit: 'TFLOPS', baseLatency: 5.72, refLatency: 5.00,
      passes: [
        { id: 'autosync', name: 'AutoSync 插入', detail: 'PTOAS 自动分析并插入同步', default: true },
        { id: 'vector', name: '保留 PIPE_V barrier', detail: 'normalize 前等待 vector 生产者', default: true },
        { id: 'merge', name: '合并 normalize 同步', detail: '合并连续 vector 消费前的同步点', default: false },
        { id: 'hoist', name: '移动 barrier 到循环外', detail: '仅模拟依赖可证明时的外提', default: false },
      ],
      issueData: [['生成版 S1=131072', '约 143.5 TFLOPS'], ['去 barrier 手工版', '164.18 TFLOPS'], ['手工版延迟', '约 5.00 ms']],
      code: ['TROWMAX(v60, scores)', 'TROWEXP(v61, scores)', 'pipe_barrier(PIPE_V)', 'TROWEXPANDDIV(v61, v61, v68)', 'TROWEXPANDDIV(v62, v62, v69)', 'TSTORE(output)'],
    },
  };
  const $ = (s) => document.querySelector(s);
  let current = 'gemm';
  let config = {};
  let hasRun = false;
  let runCount = 0;
  let running = false;
  const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n));
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  function resetConfig() { config = Object.fromEntries(workloads[current].passes.map((p) => [p.id, !!p.default])); hasRun = false; renderAll(); }
  function state() {
    const w = workloads[current];
    let gain = 0;
    let barriers = current === 'gemm' ? 18 : 7;
    if (!config.autosync) { gain += current === 'gemm' ? .15 : .07; barriers -= current === 'gemm' ? 9 : 4; }
    if (current === 'gemm') {
      if (config.merge) { gain += .19; barriers -= 4; }
      if (config.hoist) { gain += .10; barriers -= 3; }
      if (config.unroll) { gain += .12; barriers -= 2; }
      if (config.merge && config.hoist) gain += .09;
      if (config.unroll && config.merge) gain += .08;
    } else {
      if (!config.vector) { gain += .144; barriers -= 2; }
      if (config.merge) { gain += .035; barriers -= 1; }
      if (config.hoist) { gain += .02; barriers -= 1; }
      if (!config.vector && config.merge) gain += .018;
    }
    gain = clamp(gain, -.10, current === 'gemm' ? .78 : .28);
    const throughput = w.base * (1 + gain);
    const overlap = clamp((current === 'gemm' ? 29 : 48) + gain * 86 + (config.merge ? 7 : 0) + (config.hoist ? 4 : 0), 8, 94);
    const latency = w.baseLatency * w.base / throughput;
    return { w, gain, throughput, barriers: Math.max(0, barriers), overlap, latency };
  }
  function renderControls() {
    const w = workloads[current];
    $('#passControls').innerHTML = w.passes.map((p) => '<div class="ap-pass-control"><div><strong>' + esc(p.name) + '</strong><small>' + esc(p.detail) + '</small></div><button class="ap-toggle" type="button" role="switch" aria-label="' + esc(p.name) + '" aria-checked="' + (!!config[p.id]) + '" data-pass="' + p.id + '"></button></div>').join('');
    $('#passControls').querySelectorAll('[data-pass]').forEach((b) => b.addEventListener('click', () => { config[b.dataset.pass] = !config[b.dataset.pass]; b.setAttribute('aria-checked', String(config[b.dataset.pass])); $('#runStatus').textContent = '配置已修改 · 运行以查看模拟结果'; }));
  }
  function renderTimeline(s) {
    const tracks = current === 'gemm'
      ? [{ name: 'MTE2', title: 'Global → L1', kind: 'mte2' }, { name: 'Cube', title: 'TMATMUL', kind: 'cube' }, { name: 'MTE3', title: 'L1 → Global', kind: 'mte3' }]
      : [{ name: 'Vector', title: 'Normalize', kind: 'cube' }, { name: 'MTE2', title: 'Load / scores', kind: 'mte2' }, { name: 'MTE3', title: 'Output store', kind: 'mte3' }];
    const count = 4;
    const overlap = s.overlap / 100;
    const baseStep = 22 - overlap * 9;
    const blocks = [];
    tracks.forEach((lane, li) => {
      let html = '<div class="ap-lane"><div class="ap-lane-label"><b>' + lane.name + '</b>' + lane.title + '</div><div class="ap-lane-track">';
      for (let i = 0; i < count; i++) {
        const start = 3 + (i * baseStep) + li * (current === 'gemm' ? 7 : 6);
        const width = current === 'gemm' ? (li === 1 ? 17 : 14) : (li === 0 ? 18 : 14);
        html += '<span class="ap-block ' + lane.kind + '" style="left:' + start.toFixed(1) + '%;width:' + width + '%">' + (li === 1 && current === 'gemm' ? 'K' : 'tile') + i + '</span>';
      }
      if (s.barriers > 0) {
        const syncCount = clamp(Math.round(s.barriers / 3), 1, 6);
        for (let j = 0; j < syncCount; j++) {
          const x = 8 + j * (78 / syncCount);
          html += '<span class="ap-block sync" style="left:' + x.toFixed(1) + '%;width:' + (current === 'gemm' ? 4.5 : 3.8) + '%">wait</span>';
        }
      }
      html += '</div></div>';
      blocks.push(html);
    });
    $('#timeline').innerHTML = blocks.join('');
    $('#axisEnd').textContent = Math.round(s.latency * 1000) + ' μs';
  }
  function renderSummary(s) {
    const w = s.w;
    $('#passSummary').innerHTML = w.passes.map((p, i) => '<div class="ap-pass-row"><b>' + esc(p.name) + '</b><span>' + (config[p.id] ? (p.default ? '启用 · 基线行为' : '启用 · 本次改动') : '关闭 · 未执行') + '</span><em class="' + (config[p.id] !== p.default ? 'changed' : '') + '">' + (config[p.id] !== p.default ? 'CHANGED' : (config[p.id] ? 'ACTIVE' : 'SKIP')) + '</em></div>').join('');
    let message;
    if (current === 'gemm') {
      message = config.merge && config.hoist ? '<strong>分支合并 + 同步外提</strong>：模型预期跨分支 wait 减少，load 与 Cube 重叠增加；下一步应查看实际生成 C++ 是否把 wait 移出 if/else。'
        : config.unroll ? '<strong>K-loop 展开已开启</strong>：检查同步是否仍留在各分支；建议与 merge 单变量对照，避免把展开本身的代码膨胀误判为同步收益。'
          : '<strong>尝试合并 ping/pong 分支</strong>：Issue 的性能线索指向控制流影响同步外提。运行后比较 barrier 估算、重叠区间与吞吐。';
    } else {
      message = !config.vector ? '<strong>PIPE_V 已关闭</strong>：模拟结果显示等待减少。真实实验必须先证明 normalize 的读写依赖安全，再做多长度正确性覆盖。'
        : '<strong>检查 PIPE_V 的生产者/消费者关系</strong>：尝试关闭单个 barrier，再观察长序列 normalize 是否仍有等待；禁止把 mock 结果当作 barrier 可删除的证明。';
    }
    $('#recommendation').innerHTML = message;
    $('#insight').innerHTML = '<strong>' + (s.overlap > (current === 'gemm' ? 55 : 65) ? '阶段重叠改善' : '仍有明显等待') + '</strong> · 当前模拟 overlap ' + s.overlap.toFixed(0) + '%，同步点约 ' + s.barriers + ' 个。泳道中的等待块随启用的同步路径变化；实际因果需用编译产物和设备 trace 验证。';
  }
  function renderMetrics(s) {
    const w = s.w;
    const rows = [{ label: 'Issue 基线 / 自动生成', value: w.base, type: '' }, { label: '当前配置（模拟）', value: s.throughput, type: 'current' }, { label: current === 'gemm' ? 'torch_npu 对照' : '手工去 barrier 变体', value: w.reference, type: '' }];
    const max = Math.max(...rows.map((x) => x.value)) * 1.08;
    $('#metricChart').innerHTML = rows.map((r) => '<div class="ap-chart-row"><span class="ap-chart-label">' + esc(r.label) + '</span><div class="ap-chart-track"><div class="ap-chart-bar ' + r.type + '" style="width:' + (r.value / max * 100).toFixed(1) + '%"></div></div><span class="ap-chart-val">' + r.value.toFixed(2) + ' ' + w.unit + '</span></div>').join('');
    $('#metricNote').textContent = hasRun ? '模拟吞吐 ' + s.throughput.toFixed(2) + ' TFLOPS；估算 latency ' + s.latency.toFixed(3) + ' ms。相对 Issue 自动生成基线 ' + (s.gain >= 0 ? '+' : '') + (s.gain * 100).toFixed(1) + '%。' : '当前展示配置预估；点击「编译并运行」后生成一次带编号的模拟实验结果。';
    $('#issueReference').innerHTML = '<div class="ap-reference-grid">' + w.issueData.map((r) => '<div class="ap-reference"><span>' + esc(r[0]) + '</span><b>' + esc(r[1]) + '</b></div>').join('') + '</div>';
  }
  function renderCompiler(s) {
    const w = s.w;
    $('#passPipeline').innerHTML = w.passes.map((p, i) => '<div class="ap-stage"><span class="ap-stage-n">' + (i + 1) + '</span><div><b>' + esc(p.name) + '</b><small>' + esc(p.detail) + '</small></div><em class="' + (config[p.id] !== p.default ? 'changed' : '') + '">' + (config[p.id] ? (config[p.id] !== p.default ? '配置已变更' : '执行') : '跳过') + '</em></div>').join('');
    const code = w.code.map((line) => {
      const sync = /wait_flag|pipe_barrier/.test(line);
      const suppress = sync && ((!config.autosync) || (current === 'flash' && !config.vector && line.includes('PIPE_V')) || (current === 'gemm' && config.hoist && line.includes('wait_flag')));
      return '<span class="' + (suppress ? 'del' : '') + '">' + (suppress ? '− ' : '  ') + esc(line) + '</span>';
    });
    if (current === 'gemm' && config.merge) code.splice(2, 0, '<span class="add">+ // ping/pong control flow merged</span>');
    if (current === 'gemm' && config.unroll) code.push('<span class="add">+ // K-loop unrolled for this mock configuration</span>');
    $('#codeDiff').innerHTML = code.join('\n');
    $('#codeMode').textContent = hasRun ? 'RUN #' + String(runCount).padStart(3, '0') : '配置预览 · MOCK';
  }
  function renderAll() {
    const w = workloads[current], s = state();
    $('#workloadName').textContent = w.name;
    $('#workloadMeta').textContent = w.meta;
    $('#title').textContent = w.title;
    $('#subtitle').textContent = w.subtitle;
    $('#issueId').textContent = w.issue;
    $('#issueLink').href = w.issueUrl;
    $('#issueLink').textContent = 'PTOAS ' + w.issue + ' ↗';
    $('#throughput').textContent = s.throughput.toFixed(1);
    $('#throughputUnit').textContent = w.unit;
    $('#delta').textContent = hasRun ? ((s.gain >= 0 ? '+' : '') + (s.gain * 100).toFixed(1) + '%') : '待运行';
    $('#delta').className = hasRun ? (s.gain >= 0 ? 'positive' : 'negative') : '';
    $('#barrierCount').textContent = hasRun ? String(s.barriers) : '—';
    $('#overlap').textContent = hasRun ? s.overlap.toFixed(0) + '%' : '—';
    $('#runLabel').textContent = hasRun ? 'RUN #' + String(runCount).padStart(3, '0') : '未运行';
    renderControls(); renderTimeline(s); renderSummary(s); renderMetrics(s); renderCompiler(s);
  }
  function run() {
    if (running) return;
    running = true;
    const button = $('#runButton');
    button.disabled = true; button.innerHTML = '<span>◌</span> 正在编译与分析…';
    $('#runStatus').textContent = '正在应用 Pass 配置 · 生成模拟产物…';
    window.setTimeout(() => {
      runCount++; hasRun = true; running = false;
      button.disabled = false; button.innerHTML = '<span>▶</span> 编译并运行';
      $('#runStatus').textContent = '完成 · 模拟运行 #' + String(runCount).padStart(3, '0') + ' · 正确性通过（mock）';
      renderAll();
    }, 650);
  }
  $('#workloadSelect').addEventListener('change', (e) => { current = e.target.value; resetConfig(); $('#runStatus').textContent = '已切换 workload · 配置恢复为基线'; });
  $('#runButton').addEventListener('click', run);
  $('#resetButton').addEventListener('click', () => { resetConfig(); $('#runStatus').textContent = '已恢复 ' + workloads[current].name + ' 基线配置'; });
  document.querySelectorAll('.ap-tabs [data-view]').forEach((b) => b.addEventListener('click', () => {
    document.querySelectorAll('.ap-tabs [data-view]').forEach((x) => x.classList.toggle('is-active', x === b));
    document.querySelectorAll('.ap-view[data-panel]').forEach((x) => x.classList.toggle('is-visible', x.dataset.panel === b.dataset.view));
  }));
  $('#themeToggle').addEventListener('click', () => { document.documentElement.dataset.theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'; });
  resetConfig();
})();
