export function concurrency(rank) {
  const events = [[0,0],[rank.windowUs,0]];
  for (const task of rank.launches) for (const b of task.blocks.items) { events.push([b.startUs,1],[b.endUs,-1]); }
  events.sort((a,b) => a[0]-b[0] || a[1]-b[1]);
  const segments=[]; let count=0, previous=events[0][0];
  for (let i=0;i<events.length;) { const time=events[i][0]; if (time>previous) segments.push({start:previous,end:time,count}); while(i<events.length && events[i][0]===time) count+=events[i++][1]; previous=time; }
  return segments;
}
export function pack(rank) {
  const lanes=new Map(), groups=[];
  for (const type of [...new Set(rank.cores.map(c=>c.ct))]) {
    const ends=[]; const tasks=rank.launches.filter(t=>t.ct===type).slice().sort((a,b)=>a.startUs-b.startUs || b.spanUs-a.spanUs);
    for (const t of tasks) { let i=ends.findIndex(end=>end<=t.startUs); if(i<0) i=ends.length; ends[i]=t.endUs; lanes.set(t.id,i); }
    groups.push({type, count:Math.max(1,ends.length)});
  }
  return {lanes,groups};
}
const quantile=(arr,q)=>{ const a=arr.slice().sort((x,y)=>x-y); return a.length ? a[Math.floor((a.length-1)*q)] : 0; };
export function feature(rank, mode) {
  const tasks=rank.launches, hits=new Set(), gaps=[];
  const p50=quantile(tasks.map(t=>t.spanUs),.5), p25=quantile(tasks.map(t=>t.spanUs),.25), n25=quantile(tasks.map(t=>t.blocks.n),.25), end90=quantile(tasks.map(t=>t.endUs),.9);
  const byTask=new Map(); for(const t of tasks) { const key=String(t.taskId); if(!byTask.has(key)) byTask.set(key,[]); byTask.get(key).push(t); }
  for(const t of tasks) {
    const cores=rank.cores.filter(c=>c.ct===t.ct).length, b=t.blocks, share=t.setupUs/(t.computeUs+t.setupUs || 1), last=cores ? b.n%cores : 0;
    let yes=false;
    if(mode==='tail') yes=t.endUs>=end90 && t.spanUs>=p50;
    if(mode==='parallel') yes=b.n>1 && t.spanUs>=p50 && cores>0 && t.computeUs/Math.max(1,t.spanUs)<cores*.25 && share<.5;
    if(mode==='imbalance') yes=b.n>=4 && b.spread>=1.5 && b.p90/Math.max(1,b.p50)>=1.4;
    if(mode==='underfill') yes=cores>0 && b.n>cores && last>0 && last/cores<.5;
    if(mode==='fragmented') yes=b.n<=Math.max(4,n25) && t.spanUs<=p25;
    if(mode==='setup') yes=share>=.3 && t.setupUs>1;
    if(mode==='bubble' && Array.isArray(rank.edges)) {
      const ids=[...new Set(rank.edges.filter(e=>String(e.succ)===String(t.taskId)).map(e=>String(e.pred)))];
      if(ids.length && ids.every(id=>byTask.has(id))) { const ready=Math.max(...ids.flatMap(id=>byTask.get(id).map(p=>p.endUs))); if(t.startUs-ready>Math.max(3,b.p50*2)) {yes=true;gaps.push({task:t,start:ready,end:t.startUs});} }
    }
    if(yes) hits.add(t.id);
  }
  return {hits,gaps};
}
export function recommend(rank) {
  const modes=['tail','parallel','imbalance','underfill','fragmented','setup','bubble'];
  const path=new Set((rank.criticalPath||[]).flatMap(p=>[String(p.taskId),...(p.ids||[])]));
  const groups=new Map();
  for(const mode of modes){
    const result=feature(rank,mode);
    for(const task of rank.launches.filter(t=>result.hits.has(t.id))){
      const kernel=task.leaf.replace(/_\d+$/,''),key=`${task.ct} · ${kernel}`;
      if(!groups.has(key))groups.set(key,{key,kernel,ct:task.ct,tasks:[],modes:[],evidence:[],score:0,onPath:false});
      const g=groups.get(key); if(!g.tasks.some(t=>t.id===task.id))g.tasks.push(task);if(!g.modes.includes(mode))g.modes.push(mode);
    }
    for(const g of groups.values()){
      const tasks=g.tasks.filter(t=>result.hits.has(t.id));if(!tasks.length)continue;
      const span=Math.max(...tasks.map(t=>t.spanUs)),cores=rank.cores.filter(c=>c.ct===g.ct).length||1;
      const setup=tasks.reduce((n,t)=>n+t.setupUs,0),compute=tasks.reduce((n,t)=>n+t.computeUs,0);let score=span/rank.windowUs, reason='最长跨度 '+span.toFixed(1)+' μs';
      if(mode==='parallel'){const width=Math.max(...tasks.map(t=>t.computeUs/Math.max(1,t.spanUs)));score*=Math.max(0,1-width/cores);reason=`平均计算并行宽度最高 ${width.toFixed(1)} / ${cores} 核`;}
      if(mode==='imbalance'){const spread=Math.max(...tasks.map(t=>t.blocks.spread));score*=Math.min(2,spread-1);reason=`最慢块 / 平均块最高 ${spread.toFixed(1)}×`;}
      if(mode==='underfill'){const fill=Math.min(...tasks.map(t=>(t.blocks.n%cores)/cores));score*=(1-fill)*.6;reason=`末波核填充最低 ${Math.round(fill*100)}%`;}
      if(mode==='fragmented'){score=tasks.length*.006+setup/rank.windowUs*.1;reason=`${tasks.length} 次短提交`;}
      if(mode==='setup'){const share=setup/Math.max(1,setup+compute);score*=share;reason=`Setup 占比 ${Math.round(share*100)}%`;}
      if(mode==='bubble'){const gap=Math.max(0,...result.gaps.filter(x=>tasks.some(t=>t.id===x.task.id)).map(x=>x.end-x.start));score=gap/rank.windowUs;reason=`最长观测间隔 ${gap.toFixed(1)} μs`;}
      g.evidence.push({mode,score,reason});g.score=Math.max(g.score,score);g.onPath ||= tasks.some(t=>path.has(t.id)||path.has(String(t.taskId)));
    }
  }
  const pool=[...groups.values()],selected=[],covered=new Set();while(pool.length&&selected.length<5){pool.sort((a,b)=>Number(b.onPath)-Number(a.onPath)||(b.score+(b.modes.some(m=>!covered.has(m))?.025:0))-(a.score+(a.modes.some(m=>!covered.has(m))?.025:0))||a.key.localeCompare(b.key));const g=pool.shift();g.evidence.sort((a,b)=>b.score-a.score);g.primaryMode=g.evidence[0].mode;g.modes.forEach(m=>covered.add(m));selected.push(g);}return selected;
}
export function topCandidate(rank,lens='wait') {
 const tasks=rank.launches, path=rank.criticalPath||[];
 if(lens==='wait'){const ids=new Map(tasks.map(t=>[t.id,t]));return path.filter(p=>p.blockNum===1&&p.spanUs>0).map(p=>(p.ids?.length?ids.get(p.ids[0]):tasks.find(t=>String(t.taskId)===String(p.taskId)))).filter(Boolean).sort((a,b)=>b.spanUs-a.spanUs)[0]||null;}
 if(lens==='tail')return tasks.filter(t=>t.tailWasteUs>0).slice().sort((a,b)=>b.tailWasteUs-a.tailWasteUs)[0]||null;
 if(lens==='setup')return tasks.slice().sort((a,b)=>(b.setupUs/(b.setupUs+b.computeUs||1))-(a.setupUs/(a.setupUs+a.computeUs||1)))[0]||null;
 if(lens==='waves')return tasks.slice().sort((a,b)=>Math.ceil(b.blocks.n/(rank.cores.filter(c=>c.ct===b.ct).length||1))-Math.ceil(a.blocks.n/(rank.cores.filter(c=>c.ct===a.ct).length||1))||b.spanUs-a.spanUs)[0]||null;
 return null;
}
export function landmarks(rank,lens='wait'){
 const longest=rank.launches.reduce((a,b)=>!a||b.spanUs>a.spanUs?b:a,null),segments=concurrency(rank),runs=[];let current=null;
 for(const s of segments){if(s.count>0&&s.count<=2){if(current&&Math.abs(current.end-s.start)<1e-8)current.end=s.end;else{current={start:s.start,end:s.end};runs.push(current);}}else current=null;}
 const min=Math.max(8,rank.windowUs/500)*2;runs.sort((a,b)=>(b.end-b.start)-(a.end-a.start));const marks=[];const top=topCandidate(rank,lens),same=top&&longest&&top.id===longest.id&&lens!=='tail';if(longest)marks.push({start:longest.startUs,end:longest.endUs,label:same?`最长 Task，也是当前视角 Top1 · ${longest.leaf}`:`最长 Task · ${longest.leaf}`});if(runs.length&&runs[0].end-runs[0].start>=min)marks.push({...runs[0],label:'最长串行段 · ≤2 核'});if(top&&!same){if(lens==='tail'){const ends=top.blocks.items.map(b=>b.endUs).sort((a,b)=>a-b);if(ends.length>1)marks.push({start:ends[Math.floor(ends.length/2)],end:ends.at(-1),label:`尾块拖尾 · ${top.leaf}`});}else marks.push({start:top.startUs,end:top.endUs,label:`当前视角 Top1 · ${top.leaf}`});}return marks;
}
