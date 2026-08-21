/**
 * 推理性能分析 · 访存与缓存（P4）
 *
 * 对标 Nsight Compute Memory Workload + vLLM KV 面板，叠加 Ascend 片上存储层级。
 * 带宽曲线直接由 ops 的 achievedBw × 逐层时长推导，不另立数字。
 */
(function registerInferenceMemory() {
  'use strict';

  const esc = (v) => String(v).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmt = (n, d = 2) => Number(n).toFixed(d);

  const HBM_COLOR = {
    weights: 'var(--primary)',
    kv: 'var(--warning)',
    workspace: 'var(--tone-blue-strong, #4a90d9)',
    act: 'var(--tone-green-strong, #4caf7d)',
  };

  /* HBM 占用 */
  function hbm(p) {
    const m = p.memory.hbm;
    const used = m.items.reduce((a, i) => a + i[2], 0);
    const free = m.capacity - used;
    const bar = m.items.map((i) => `<i style="width:${i[2] / m.capacity * 100}%;background:${HBM_COLOR[i[0]]}" title="${esc(i[1])} ${fmt(i[2], 2)} GB"></i>`).join('')
      + `<i style="width:${free / m.capacity * 100}%;background:var(--surface-4)" title="空闲 ${fmt(free, 2)} GB"></i>`;
    const rows = m.items.map((i) => `<tr>
        <td><span class="kf-prof-swatch" style="background:${HBM_COLOR[i[0]]}"></span>${esc(i[1])}</td>
        <td><b>${fmt(i[2], 2)}</b> GB</td>
        <td>${fmt(i[2] / m.capacity * 100, 1)}%</td>
        <td class="kf-prof-cmpnote">${esc(i[3])}</td>
      </tr>`).join('');
    return `<section class="kf-prof-card">
      <header><h3>HBM 占用构成</h3><span>${fmt(used, 2)} / ${m.capacity} GB · ${fmt(used / m.capacity * 100, 1)}%</span></header>
      <div class="kf-prof-card__body">
        <div class="kf-prof-stack" style="height:22px">${bar}</div>
        <table class="kf-prof-cmp" style="margin-top:12px">
          <thead><tr><th>项</th><th>占用</th><th>占容量</th><th style="text-align:left">说明</th></tr></thead>
          <tbody>${rows}<tr><td><span class="kf-prof-swatch" style="background:var(--surface-4)"></span>空闲</td><td><b>${fmt(free, 2)}</b> GB</td><td>${fmt(free / m.capacity * 100, 1)}%</td><td class="kf-prof-cmpnote">可再容纳 ${Math.floor(free / (p.memory.kv.pageBytesMb / 1000))} 个 KV 页</td></tr></tbody>
        </table>
      </div>
    </section>`;
  }

  /* 每 step 的 HBM 流量 */
  function traffic(p) {
    const t = p.summary.traffic;
    const tpot = p.summary.tpot.p50;
    const rows = [
      ['权重读入', t.weights, '每层 660.7 MB × 40 + LM Head 1.56 GB · 每 step 全量读一遍'],
      ['KV Cache 读入', t.kv, `${p.memory.kv.tokensLive.toLocaleString('en-US')} token × 160 KiB · paged 非连续`],
      ['激活与中间量', t.act, '层内 tile、累加器回写、fa 分块 partials'],
    ].map(([label, gb, note]) => `<tr>
        <td>${esc(label)}</td><td><b>${fmt(gb, 2)}</b> GB</td>
        <td>${fmt(gb / t.total * 100, 1)}%</td>
        <td>${fmt(gb / tpot, 2)} TB/s</td>
        <td class="kf-prof-cmpnote">${esc(note)}</td>
      </tr>`).join('');
    const achieved = t.total / tpot;
    return `<section class="kf-prof-card">
      <header><h3>每 step HBM 流量</h3><span>${fmt(t.total, 2)} GB ÷ ${fmt(tpot, 1)} ms = ${fmt(achieved, 2)} TB/s</span></header>
      <div class="kf-prof-card__body">
        <table class="kf-prof-cmp">
          <thead><tr><th>来源</th><th>字节</th><th>占比</th><th>等效带宽</th><th style="text-align:left">说明</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
        <div class="kf-prof-sol" style="margin-top:14px">
          <div class="kf-prof-solrow is-bottleneck" data-unit="mte2">
            <span>达成带宽</span>
            <div class="kf-prof-soltrack"><div class="kf-prof-solfill" style="width:${achieved / p.meta.peakBw * 100}%"></div></div>
            <div class="kf-prof-solval">${fmt(achieved / p.meta.peakBw * 100, 1)}%</div>
            <div class="kf-prof-soldetail">${fmt(achieved, 2)} / ${fmt(p.meta.peakBw, 1)} TB/s 峰值 · MTE2 占空比 ${fmt(p.summary.sol[2].pct, 1)}%</div>
          </div>
        </div>
      </div>
    </section>`;
  }

  /* 片上存储层级 */
  function onchip(p) {
    const rows = p.memory.onchip.map(([label, pct, budget, note]) => {
      const over = budget !== null && pct > budget;
      return `<div class="kf-prof-solrow${over ? ' is-bottleneck' : ''}" data-unit="${over ? 'mte2' : 'cube'}">
        <span>${esc(label)}</span>
        <div class="kf-prof-soltrack">
          <div class="kf-prof-solfill" style="width:${pct}%"></div>
          ${budget !== null ? `<i class="kf-prof-budget" style="left:${budget}%" title="编译期预算 ${budget}%"></i>` : ''}
        </div>
        <div class="kf-prof-solval">${pct}%</div>
        <div class="kf-prof-soldetail">${esc(note)}</div>
      </div>`;
    }).join('');
    return `<section class="kf-prof-card">
      <header><h3>片上存储层级</h3><span>竖线 = 编译期预算</span></header>
      <div class="kf-prof-card__body"><div class="kf-prof-sol">${rows}</div></div>
    </section>`;
  }

  /* 带宽曲线：由每个任务的 achievedBw × 该层实际时长铺出来 */
  function bandwidth(p) {
    const layer = 12;
    const chain = window.PtoInferenceTimeline?.LAYER_CHAIN || [];
    const byId = Object.fromEntries(p.ops.map((o) => [o.id, o]));
    const segs = chain.map((id) => {
      const op = byId[id];
      return { name: op.name, dur: op.perLayer ? op.perLayer[layer] : op.perLayerUs, bw: op.achievedBw, group: op.group };
    });
    const total = segs.reduce((a, s) => a + s.dur, 0);
    const peak = p.meta.peakBw;
    const H = 120;
    let x = 0;
    const bars = segs.map((s) => {
      const w = s.dur / total * 100;
      const h = Math.max(s.bw / peak * 100, 0.6);
      const el = `<i style="left:${x}%;width:${w}%;height:${h}%" class="${s.bw / peak > 0.65 ? 'is-high' : s.bw / peak > 0.3 ? 'is-mid' : 'is-low'}" title="${esc(s.name)} · ${fmt(s.bw, 2)} TB/s · ${fmt(s.dur, 1)} μs"></i>`;
      x += w;
      return el;
    }).join('');
    return `<section class="kf-prof-card">
      <header><h3>层内 MTE2 带宽曲线</h3><span>L${layer} · 每段高度 = 该任务达成带宽 / ${fmt(peak, 1)} TB/s 峰值</span></header>
      <div class="kf-prof-card__body">
        <div class="kf-prof-bwchart" style="height:${H}px">
          <i class="kf-prof-bwroof" style="bottom:100%"><b>峰值 ${fmt(peak, 1)} TB/s</b></i>
          <i class="kf-prof-bwavg" style="bottom:${p.summary.traffic.total / p.summary.tpot.p50 / peak * 100}%"><b>整步均值 ${fmt(p.summary.traffic.total / p.summary.tpot.p50, 2)}</b></i>
          ${bars}
        </div>
        <div class="kf-prof-histaxis">
          <span>投影 / MLP 段贴近 <b>70–76%</b></span>
          <span>fa_fused 段掉到 <b>51.7%</b></span>
          <span>Vector 段带宽低但耗时短</span>
        </div>
      </div>
    </section>`;
  }

  /* Paged KV */
  function paged(p) {
    const kv = p.memory.kv;
    const palette = ['var(--primary)', 'var(--warning)', 'var(--tone-blue-strong, #4a90d9)', 'var(--tone-green-strong, #4caf7d)'];
    let cells = '';
    kv.perRequest.forEach((r, i) => {
      for (let k = 0; k < r.pages; k += 1) {
        cells += `<i style="background:${palette[i % palette.length]};opacity:${0.55 + (i % 4) * 0.15}" title="${esc(r.req)} · 第 ${k + 1}/${r.pages} 页 · seq ${r.seq}"></i>`;
      }
    });
    for (let k = kv.pagesUsed; k < kv.pagesTotal; k += 1) cells += '<i class="is-free" title="空闲页"></i>';

    const maxSeq = Math.max(...kv.perRequest.map((r) => r.seq));
    const seqRows = kv.perRequest.map((r) => `<div class="kf-prof-seqrow">
        <span>${esc(r.req)}</span>
        <div class="kf-prof-seqtrack"><i style="width:${r.seq / maxSeq * 100}%"></i><u style="width:${(r.pages * kv.pageTokens - r.seq) / maxSeq * 100}%"></u></div>
        <b>${r.seq.toLocaleString('en-US')}</b><em>${r.pages} 页</em>
      </div>`).join('');

    return `<div class="kf-prof-grid2">
      <section class="kf-prof-card">
        <header><h3>页池分配位图</h3><span>${kv.pagesUsed} / ${kv.pagesTotal} 页 · 每页 ${kv.pageTokens} token / ${fmt(kv.pageBytesMb, 2)} MB</span></header>
        <div class="kf-prof-card__body">
          <div class="kf-prof-pagegrid">${cells}</div>
          <dl class="kf-prof-kv" style="margin-top:12px">
            <div><dt>已分配</dt><dd>${fmt(kv.bytesAllocated, 2)} / ${fmt(kv.bytesPool, 2)} GB · ${fmt(kv.utilization, 1)}%</dd></div>
            <div><dt>活跃 token</dt><dd>${kv.tokensLive.toLocaleString('en-US')} / ${kv.tokensAllocated.toLocaleString('en-US')}</dd></div>
            <div><dt>内部碎片</dt><dd class="kf-prof-eff good">${fmt(kv.fragmentation, 2)}%</dd></div>
            <div><dt>命中率</dt><dd>${fmt(kv.hitRate, 1)}%</dd></div>
            <div><dt>抢占 / 换出</dt><dd>${kv.preempt} / ${kv.swap}</dd></div>
          </dl>
        </div>
      </section>
      <section class="kf-prof-card">
        <header><h3>每请求 seq_len 与页占用</h3><span>实线 = 活跃 token · 空心 = 页内填充</span></header>
        <div class="kf-prof-card__body"><div class="kf-prof-seqlist">${seqRows}</div></div>
      </section>
    </div>`;
  }

  /* work table 稠密率 —— fa_work_build 的价值证明 */
  function workTable(p) {
    const kv = p.memory.kv;
    const cells = Array.from({ length: kv.blocksPadded }, (_, i) => `<i class="${i < kv.blocksReal ? 'is-real' : 'is-pad'}"></i>`).join('');
    return `<section class="kf-prof-card">
      <header><h3>fa_work_table 稠密率</h3><span>MCB 静态上界 ${kv.mcb} × ${p.meta.batch} 请求 = ${kv.blocksPadded} 块</span></header>
      <div class="kf-prof-card__body">
        <div class="kf-prof-blockgrid">${cells}</div>
        <div class="kf-prof-verdict" style="border-color:color-mix(in srgb,var(--success) 36%,var(--border-subtle));background:color-mix(in srgb,var(--success) 8%,transparent)">
          <i style="color:var(--success)">✓</i>
          <b>稠密率 ${fmt(kv.density, 1)}% · 压掉 ${kv.blocksPadded - kv.blocksReal} 个空块</b>
          <p>编译期只能按 <code>MCB = ${kv.mcb}</code> 的静态上界分配 ${kv.blocksPadded} 块；<code>fa_work_build</code> 花 2.2 μs/层 读 seq_lens 把 ragged 请求压紧成 ${kv.blocksReal} 个真实块，
          为 <code>fa_fused</code> 省掉 ${fmt((1 - kv.density / 100) * 100, 1)}% 的空块迭代。这是静态推断拿不到、只有实测才能记账的收益。</p>
        </div>
      </div>
    </section>`;
  }

  /* 精度边界 */
  function precision(p) {
    const rows = p.memory.precision.map(([label, where, count, tone]) => `<tr>
        <td>${esc(label)}</td><td>${esc(where)}</td>
        <td>${count === null ? '—' : `<b>${count}</b> 次`}</td>
        <td class="delta ${tone}">${count === 0 ? '已消除' : '符合预期'}</td>
      </tr>`).join('');
    return `<section class="kf-prof-card">
      <header><h3>精度边界记账</h3><span>层内零转换是 FP32 carry 策略的直接收益</span></header>
      <div class="kf-prof-card__body"><table class="kf-prof-cmp">
        <thead><tr><th>转换点</th><th>位置</th><th>次数</th><th>结论</th></tr></thead>
        <tbody>${rows}</tbody>
      </table></div>
    </section>`;
  }

  function render(p) {
    return `<div class="kf-prof-grid2">${hbm(p)}${onchip(p)}</div>`
      + traffic(p)
      + bandwidth(p)
      + paged(p)
      + `<div class="kf-prof-grid2">${workTable(p)}${precision(p)}</div>`;
  }

  window.PtoInferenceMemory = { render };
})();
