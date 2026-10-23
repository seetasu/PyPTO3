/* Palette values from PyPTO3-main/vendor/pto-design-system/tokens/foundation.css.
   Task categories are name-derived, not profiler-supplied semantic metadata. */
window.CSAAnalysis=(()=>{
 // PTO model-graphviz SEMANTIC_COLOR_DEFAULTS; runtime-only roles stay neutral.
 const colors={'Normalization':'#0EA5E9','Linear / projection':'#4F46E5','Attention':'#3B82F6','RoPE / position':'#A855F7','Routing / Top-k':'#F59E0B','Communication':'#06B6D4','Cache / layout':'#999999','Other / fused':'#888888'};
 const severity=['#888888','#FBDC59','#ffaa3b','#ff4b7b'];
 function classify(name){
  const n=name.split('(')[0];
  // Communication waits retain the semantic color of their collective.
  if(/allgather|a2a|^tp_o_(rs|b_publish)/.test(n))return 'Communication';
  if(/topk|sinkhorn/.test(n))return 'Routing / Top-k';
  // Fused norm/RoPE/cache kernels use normalization as their primary role.
  if(/rms|norm/.test(n))return 'Normalization';
  if(/rope/.test(n))return 'RoPE / position';
  if(/proj|linear|hadamard|^tp_o_[ab]_spmd/.test(n))return 'Linear / projection';
  if(/qk_pv|softmax|indexer_score|csa_merge/.test(n))return 'Attention';
  if(/cache|writeback|state_commit|slots|kv_touch|split_pre_post/.test(n))return 'Cache / layout';
  return 'Other / fused';
 }
 function build(tasks){const groups=new Map();tasks.forEach((t,i)=>{const k=t.rank+':'+t.name+':'+(t.family||(t.lane%72<24?'AIC':'AIV'));if(!groups.has(k))groups.set(k,[]);groups.get(k).push(i)});const result=[];for(const peers of groups.values()){const sorted=peers.map(i=>tasks[i].end-tasks[i].start).sort((a,b)=>a-b),n=sorted.length,median=(sorted[Math.floor((n-1)/2)]+sorted[Math.floor(n/2)])/2;for(const i of peers){const duration=tasks[i].end-tasks[i].start,ratio=median>0?duration/median:0,eligible=n>=4&&median>0&&duration-median>=5;result[i]={type:classify(tasks[i].name),duration,median,ratio,level:eligible?(ratio>=3?3:ratio>=2?2:ratio>=1.5?1:0):0,peers,task:tasks[i]}}}return result}
 function detail(a,i){const s=a[i],max=Math.max(...s.peers.map(j=>a[j].duration),1),scale=v=>94-v/max*78;const bars=s.peers.map((j,k)=>`<path d="M${8+k*264/s.peers.length} 94V${scale(a[j].duration)}" stroke="${j===i?'#fff':severity[a[j].level]}" stroke-width="${Math.max(1,Math.min(7,220/s.peers.length))}"/>`).join('');return `<h3 class="evidence-heading">${s.type} · Peer duration</h3><p class="finding" style="--finding:${severity[s.level]}">${s.level?'Duration deviation':'No duration flag'} · ${s.ratio.toFixed(2)}× median</p><svg class="peer-chart" role="img" aria-label="Peer duration bars, selected core white, dashed line median" viewBox="0 0 280 108">${bars}<path d="M0 ${scale(s.median)}H280" stroke="#aaa" stroke-dasharray="3 4"/></svg><dl><dt>Selected</dt><dd>${s.duration.toFixed(2)} µs</dd><dt>Peer median</dt><dd>${s.median.toFixed(2)} µs</dd><dt>Difference</dt><dd>${(s.duration-s.median).toFixed(2)} µs</dd><dt>Peers</dt><dd>${s.peers.length} cores</dd></dl>`}
 return {colors,severity,build,detail,classify};
})();
