(() => {
  const r = window.RUN_RESULT, trace = window.TRACE_SUMMARY;
  if (!r || !trace) return;
  const $ = (id) => document.getElementById(id);
  const fmt = (n, d=0) => Number(n).toLocaleString('en-US', {minimumFractionDigits:d, maximumFractionDigits:d});
  $('completed').textContent = fmt(r.completed);
  $('failed').textContent = `${fmt(r.failed)} failed · ${fmt(r.num_prompts)} total requests`;
  $('req-rate').textContent = fmt(r.request_throughput, 2);
  $('duration').textContent = `${fmt(r.duration,1)} s`;
  $('out-rate').textContent = fmt(r.output_throughput, 1);
  $('total-rate').textContent = fmt(r.total_token_throughput, 1);
  $('ttft-p50').textContent = fmt(r.median_ttft_ms);
  $('ttft-mean').textContent = fmt(r.mean_ttft_ms);
  $('ttft-p99').textContent = fmt(r.p99_ttft_ms);
  $('tpot-p50').textContent = fmt(r.median_tpot_ms);
  $('tpot-mean').textContent = fmt(r.mean_tpot_ms);
  $('tpot-p99').textContent = fmt(r.p99_tpot_ms);
  $('concurrency').textContent = fmt(r.max_concurrency);
  $('input-tokens').textContent = fmt(r.total_input_tokens);
  $('output-tokens').textContent = fmt(r.total_output_tokens);
  $('event-count').textContent = fmt(trace.eventCount);
  $('trace-span').textContent = `${fmt(trace.traceSpanMs/1000,1)} s`;

  const series = {
    ttft: {name:'TTFT', values:r.ttfts.map(x=>x*1000), unit:'ms', color:'mint'},
    tpot: {name:'TPOT', values:r.itls.map(xs=>xs.filter(Number.isFinite).reduce((a,b)=>a+b,0)/xs.filter(Number.isFinite).length*1000).filter(Number.isFinite), unit:'ms', color:'blue'},
    itl: {name:'ITL', values:r.itls.flat().filter(Number.isFinite).map(x=>x*1000), unit:'ms', color:'purple'}
  };
  function renderChart(kind) {
    const s=series[kind], max=Math.max(...s.values), bins=24, step=max/bins || 1, counts=Array(bins).fill(0);
    s.values.forEach(v=>counts[Math.min(bins-1,Math.floor(v/step))]++);
    const peak=Math.max(...counts)||1;
    $('histogram').innerHTML=counts.map((n,i)=>`<div class="bar" style="--height:${Math.max(n?3:0,n/peak*100)}%;--bar-color:var(--${s.color})" data-tip="${fmt(i*step)}–${fmt((i+1)*step)} ${s.unit} · ${n} samples"></div>`).join('');
    $('histogram').querySelectorAll('.bar').forEach(el=>el.style.background=`linear-gradient(0deg,color-mix(in srgb,var(--${s.color}) 50%,transparent),var(--${s.color}))`);
    $('chart-name').textContent=s.name; $('chart-range').textContent=`${fmt(s.values.length)} samples · 0–${fmt(max)} ${s.unit}`;
    $('axis-top').textContent=fmt(peak); $('axis-mid').textContent=fmt(peak/2);
  }
  document.querySelectorAll('.segmented button').forEach(btn=>btn.addEventListener('click',()=>{document.querySelectorAll('.segmented button').forEach(b=>{b.classList.toggle('is-active',b===btn);b.setAttribute('aria-selected',b===btn?'true':'false')});renderChart(btn.dataset.kind)}));
  renderChart('ttft');

  const rows=trace.topEvents.slice(0,14), maxDur=rows[0]?.durMs||1;
  $('profile-list').innerHTML=rows.map(item=>{const host=item.key.startsWith('cpu_op ·');const [group,...parts]=item.key.split(' · ');const name=parts.join(' · ')||group;return `<div class="profile-row ${host?'host':'device'}"><div><div class="profile-name" title="${name.replaceAll('"','&quot;')}">${name}</div><div class="profile-sub">${group} · ${fmt(item.count)} events</div></div><div class="profile-bar"><i style="--width:${Math.max(2,item.durMs/maxDur*100)}%"></i></div><div class="profile-total">${fmt(item.durMs/1000,1)} s</div></div>`}).join('');
})();
