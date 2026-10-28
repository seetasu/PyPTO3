import {concurrency, pack, feature, recommend, landmarks} from './analysis.js';
const ns='http://www.w3.org/2000/svg';
function node(tag,attrs={},text) { const e=document.createElementNS(ns,tag); for(const [k,v] of Object.entries(attrs)) e.setAttribute(k,v); if(text!=null)e.textContent=text; return e; }
function add(parent,tag,attrs,text) { const e=node(tag,attrs,text); parent.append(e); return e; }
const fmt=n=>n>=1000 ? `${(n/1000).toFixed(2)} ms` : `${n.toFixed(1)} μs`;
export function renderLane(svg, rank, state, onSelect) {
  svg.replaceChildren(); const width=Math.max(1000,svg.parentElement.clientWidth), left=100, right=16, row=state.view==='core'?(state.expanded?28:Math.max(8,Math.min(18,(svg.parentElement.clientHeight-150)/Math.max(1,rank.cores.length)))):28;
  const packed=pack(rank), cores=rank.cores.slice().sort((a,b)=>a.id-b.id), groups=state.view==='core' ? [...new Set(cores.map(c=>c.ct))].map(type=>({type,count:cores.filter(c=>c.ct===type).length})) : packed.groups;
  const offsets=new Map(); let y=state.ribbon?95:42; for(const g of groups){offsets.set(g.type,y+20);y+=20+g.count*row+8;} const height=y+10;
  svg.setAttribute('viewBox',`0 0 ${width} ${height}`); svg.setAttribute('width',width); svg.setAttribute('height',height);
  const [start,end]=state.range || [0,rank.windowUs], x=t=>left+(t-start)/(end-start)*(width-left-right), clip=(a,b)=>[Math.max(start,a),Math.min(end,b)];
  const segs=concurrency(rank), analyzed=state.feature==='none'?{hits:new Set(),gaps:[]}:state.feature==='recommended'?{hits:new Set(recommend(rank).flatMap(g=>g.tasks.map(t=>t.id))),gaps:[]}:feature(rank,state.feature);
  const palette=['#e5b762','#8bc4ee','#c3a5e9','#e69b91','#92c8ac','#d2b88a','#a7b5ed','#c7b6a2']; const kernels=[...new Set(rank.launches.filter(t=>analyzed.hits.has(t.id)).map(t=>t.ct+' · '+t.leaf.replace(/_\d+$/, '')))]; const colors=new Map(kernels.map((k,i)=>[k,palette[i%palette.length]]));
  if(state.bands) for(const s of segs) if(s.count<=2) {const [a,b]=clip(s.start,s.end);if(b>a)add(svg,'rect',{x:x(a),y:state.ribbon?95:42,width:Math.max(1,x(b)-x(a)),height:height-(state.ribbon?95:42),fill:s.count===0?'#69513d':'#777',opacity:s.count===0?.2:.08});}
  if(state.marks)for(const m of landmarks(rank,state.lens)){const [a,b]=clip(m.start,m.end);if(b>a){const g=add(svg,'g',{'pointer-events':'none'});add(g,'rect',{x:x(a),y:state.ribbon?95:42,width:Math.max(1,x(b)-x(a)),height:height-(state.ribbon?95:42),fill:'#ccc',opacity:.04});for(const xx of [x(a),x(b)])add(g,'line',{x1:xx,x2:xx,y1:state.ribbon?95:42,y2:height,stroke:'#aaa','stroke-opacity':.5});add(g,'title',{},m.label);add(svg,'text',{x:x(a)+4,y:(state.ribbon?95:42)+13,fill:'#aaa','font-size':11,'pointer-events':'none'},m.label);}}
  const axis=document.getElementById('lane-axis');axis.replaceChildren();axis.setAttribute('width',width);axis.setAttribute('height',28);axis.setAttribute('viewBox',`0 0 ${width} 28`);for(let i=0;i<=8;i++){const xx=left+(width-left-right)*i/8;add(svg,'line',{x1:xx,y1:0,x2:xx,y2:height,stroke:'#303030'});add(axis,'text',{x:xx,y:17,fill:'#888','font-size':11,'text-anchor':i===0?'start':i===8?'end':'middle'},fmt(start+(end-start)*i/8));}
  if(state.ribbon){const base=87, total=cores.length||1; for(const s of segs){const [a,b]=clip(s.start,s.end);if(b>a)add(svg,'rect',{x:x(a),y:base-Math.min(1,s.count/total)*42,width:Math.max(.5,x(b)-x(a)),height:Math.min(1,s.count/total)*42,fill:'#727272'});}add(svg,'text',{x:5,y:57,fill:'#999','font-size':11},'并行宽度');}
  for(const g of groups) add(svg,'text',{x:8,y:offsets.get(g.type)-7,fill:'#aaa','font-size':12},g.type);
  const pos=new Map(); if(state.view==='core') for(const g of groups) cores.filter(c=>c.ct===g.type).forEach((c,i)=>{const yy=offsets.get(g.type)+i*row;pos.set(String(c.id),yy);if(i%4===0)add(svg,'text',{x:left-8,y:yy+10,'text-anchor':'end',fill:'#777','font-size':10},String(c.id));});
  const select=(g,t,core)=>{g.setAttribute('tabindex','0');g.setAttribute('role','button');g.setAttribute('aria-label',`${t.leaf}, ${fmt(t.spanUs)}`);g.addEventListener('click',()=>onSelect(t,core));g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();onSelect(t,core);}});};
  for(const t of rank.launches) {
    const records=state.view==='core'?t.blocks.items:[{startUs:t.startUs,endUs:t.endUs}];
    for(const b of records){const [a,z]=clip(b.startUs,b.endUs);if(z<=a)continue; const yy=state.view==='core'?pos.get(String(b.core)):offsets.get(t.ct)+packed.lanes.get(t.id)*row; if(!Number.isFinite(yy))continue;
      const selected=state.selected===t.id, hit=analyzed.hits.has(t.id), color=selected?'#f0f0f0':hit?colors.get(t.ct+' · '+t.leaf.replace(/_\d+$/, '')):'#565656', opacity=state.focus?(state.focus.includes(t.id)?1:.12):(state.feature==='none'||hit||selected?1:.23);
      const g=add(svg,'g',{class:'event',opacity}); select(g,t,b.core); const h=state.view==='core'?row-2:Math.max(3,(row-4)*Math.min(1,t.coresUsed/(cores.filter(c=>c.ct===t.ct).length||1))); const top=yy+row-h-1;
      add(g,'rect',{x:x(a),y:top,width:Math.max(1,x(z)-x(a)),height:h,fill:color,rx:1});
      if(state.composition && state.view==='core' && Number.isFinite(b.slotUs) && b.slotUs>0 && !selected) {const w=x(z)-x(a), setup=Math.max(0,Math.min(1,b.setupUs/b.slotUs));add(g,'rect',{x:x(a),y:top,width:Math.max(0,w*setup),height:h,fill:'#b48b49'});}
      if((state.view==='task'||state.expanded) && x(z)-x(a)>65 && h>=15) add(g,'text',{x:x(a)+4,y:top+h/2+4,fill:selected?'#111':'#e8e8e8','font-size':11,'pointer-events':'none'},t.leaf.length>(x(z)-x(a)-8)/7?t.leaf.slice(0,Math.max(1,Math.floor((x(z)-x(a)-15)/7)))+'…':t.leaf);
      add(g,'title',{},`${t.leaf}\n${fmt(b.startUs)} → ${fmt(b.endUs)}${b.core==null?'':` · Core ${b.core}`}`);
    }
  }
  for(const gap of analyzed.gaps){const [a,b]=clip(gap.start,gap.end);if(b>a)add(svg,'line',{x1:x(a),x2:x(b),y1:height-6,y2:height-6,stroke:'#e7b95e','stroke-width':3});}
  return {hits:analyzed.hits.size, groups:kernels.map(key=>({key,color:colors.get(key),count:rank.launches.filter(t=>analyzed.hits.has(t.id)&&t.ct+' · '+t.leaf.replace(/_\d+$/, '')===key).length}))};
}
export function renderCritical(svg, rank, selected, onSelect) {
  svg.replaceChildren();const path=rank.criticalPath||[], width=Math.max(1000,svg.parentElement.clientWidth), height=66;svg.setAttribute('viewBox',`0 0 ${width} ${height}`);svg.setAttribute('width',width);svg.setAttribute('height',height);
  if(!path.length){add(svg,'text',{x:0,y:25,fill:'#999'},'数据未提供关键路径');return;}
  const total=path.reduce((s,p)=>s+(p.pathContributionUs??p.spanUs??0),0)||1, gap=2, usable=Math.max(1,width-path.length*gap);let x=0;
  for(const p of path){const w=Math.max(0,(p.pathContributionUs??p.spanUs??0)/total*usable);const g=add(svg,'g',{class:'event',tabindex:0,role:'button','aria-label':p.name||p.taskId}); const active=(p.ids||[]).includes(selected);add(g,'rect',{x,y:4,width:w,height:28,fill:active?'#f0f0f0':p.blockNum===1?'#d17160':'#777'}); if(w>50)add(g,'text',{x:x+4,y:23,fill:'#161616','font-size':11},fmt(p.spanUs||0));add(g,'title',{},`${p.name} · 实测跨度 ${fmt(p.spanUs||0)} · 路径新增 ${fmt(p.pathContributionUs??p.spanUs??0)}`);const choose=()=>onSelect(p);g.addEventListener('click',choose);g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose();}});x+=w+gap;}
}

