import {blockPhases} from './single-dependency.js';
import {pathLaunches} from './view-mode.js';
import {concurrency, concurrencyBuckets, concurrencyRegions, pack, feature, recommend, landmarks, featureGroups} from './analysis.js';
import {drawRecommendationOverlay} from './recommendation-overlay.js';
const ns='http://www.w3.org/2000/svg';
function node(tag,attrs={},text) { const e=document.createElementNS(ns,tag); for(const [k,v] of Object.entries(attrs)) e.setAttribute(k,v); if(text!=null)e.textContent=text; return e; }
function add(parent,tag,attrs,text) { const e=node(tag,attrs,text); parent.append(e); return e; }
const fmt=n=>n>=1000 ? `${(n/1000).toFixed(2)} ms` : `${n.toFixed(1)} μs`;
export function renderLane(svg, rank, state, onSelect, onFocus = () => {}) {
  if(state.pathFilter)rank={...rank,launches:pathLaunches(rank)};
  svg.replaceChildren(); const width=Math.max(600,svg.parentElement.clientWidth), left=76, right=16, row=state.view==='core'?(state.expanded?28:Math.max(1,Math.min(18,(svg.parentElement.clientHeight+(svg.parentElement.clientWidth<600?80:0)-28-(state.ribbon?78:10)-6-72)/Math.max(1,rank.cores.length)))):38;
  const packed=pack(rank), cores=rank.cores.slice().sort((a,b)=>a.id-b.id), groups=state.view==='core' ? [...new Set(cores.map(c=>c.ct))].map(type=>({type,count:cores.filter(c=>c.ct===type).length})) : packed.groups;
  const offsets=new Map(); let y=state.ribbon?78:10; for(const g of groups){offsets.set(g.type,y+36);y+=36+g.count*row;} const height=y+4;
  svg.setAttribute('viewBox',`0 0 ${width} ${height}`); svg.setAttribute('width',width); svg.setAttribute('height',height);
  const [start,end]=state.range || [0,rank.windowUs], x=t=>left+(t-start)/(end-start)*(width-left-right), clip=(a,b)=>[Math.max(start,a),Math.min(end,b)];
  const segs=concurrency(rank), analyzed=state.feature==='none'?{hits:new Set(),gaps:[]}:state.feature==='recommended'?{hits:new Set(recommend(rank).flatMap(g=>g.tasks.map(t=>t.id))),gaps:[]}:feature(rank,state.feature);
  const palette=['#e5b762','#8bc4ee','#c3a5e9','#e69b91','#92c8ac','#d2b88a','#a7b5ed','#c7b6a2']; const kernels=[...new Set(rank.launches.filter(t=>analyzed.hits.has(t.id)).map(t=>t.ct+' · '+t.leaf.replace(/_\d+$/, '')))]; const colors=new Map(kernels.map((k,i)=>[k,palette[i%palette.length]]));
  const laneTop=state.ribbon?78:10;
  if(state.bands && !state.chain){const regions=concurrencyRegions(segs,rank.windowUs);for(const [items,fill,stroke] of [[regions.serial,'rgb(185 138 69 / 9%)','#b98a45'],[regions.idle,'rgb(253 86 68 / 18%)','#fd5644']])for(const region of items){const [a,b]=clip(region.start,region.end);if(b<=a)continue;const x0=x(a),x1=x(b);add(svg,'rect',{x:x0,y:laneTop,width:Math.max(1,x1-x0),height:height-laneTop,fill});for(const xx of [x0,x1])add(svg,'line',{x1:xx,x2:xx,y1:laneTop,y2:height,stroke,'stroke-opacity':.72,'stroke-width':1.2});}}
  if(state.marks){let markIndex=0;for(const m of landmarks(rank,state.lens)){const [a,b]=clip(m.start,m.end);if(b>a){const g=add(svg,'g',{'pointer-events':'none'}),x0=x(a),x1=x(b),top=laneTop;add(g,'rect',{x:x0,y:top,width:Math.max(1,x1-x0),height:height-top,fill:'#ccc',opacity:.04});for(const xx of [x0,x1])add(g,'line',{x1:xx,x2:xx,y1:top,y2:height,stroke:'#aaa','stroke-opacity':.5});add(g,'title',{},m.label);const label=m.label.length>30?m.label.slice(0,29)+'…':m.label,labelWidth=Math.min(240,Math.max(72,label.length*7+14)),labelX=Math.max(left,Math.min(width-right-labelWidth,x0+4)),labelY=top+4+(markIndex++%3)*20;add(g,'rect',{x:labelX,y:labelY,width:labelWidth,height:17,rx:5,fill:'#383838','fill-opacity':.96});add(g,'text',{x:labelX+6,y:labelY+12,fill:'#ddd','font-size':11},label);}}}

  const axis=document.getElementById('lane-axis');axis.replaceChildren();axis.setAttribute('width',width);axis.setAttribute('height',28);axis.setAttribute('viewBox',`0 0 ${width} 28`);const ticks=width<900?4:8;for(let i=0;i<=ticks;i++){const xx=left+(width-left-right)*i/ticks;add(svg,'line',{x1:xx,y1:0,x2:xx,y2:height,stroke:'#303030'});add(axis,'text',{x:xx,y:17,fill:'#888','font-size':11,'text-anchor':i===0?'start':i===ticks?'end':'middle'},end-start<1000?`${Math.round(start+(end-start)*i/ticks)} μs`:`${((start+(end-start)*i/ticks)/1000).toFixed(3)} ms`);}
  if(state.ribbon){
    const base=70,rib=60,total=rank.kpi?.totalCores||cores.length||1, inner=width-left-right, values=concurrencyBuckets(segs,start,end), round=v=>Math.round(v*10)/10;
    add(svg,'rect',{x:left,y:base-rib*.25,width:inner,height:rib*.25,fill:'rgb(255 255 255 / 4%)'});
    let area=`M ${left} ${base}`,topLine='',lowLine='';
    values.forEach((value,i)=>{const yy=round(base-Math.min(1,value/total)*rib),x0=round(left+inner*i/values.length),x1=round(left+inner*(i+1)/values.length);area+=` L ${x0} ${yy} L ${x1} ${yy}`;topLine+=`M ${x0} ${yy} H ${x1}`;if(value/total<=.25)lowLine+=`M ${x0} ${yy} H ${x1}`;});
    area+=` L ${left+inner} ${base} Z`;add(svg,'path',{d:area,fill:'#383838'});add(svg,'path',{d:topLine,fill:'none',stroke:'#858585','stroke-width':1.1});add(svg,'path',{d:lowLine,fill:'none',stroke:'#c4c4c4','stroke-width':2,'stroke-linecap':'round'});add(svg,'line',{x1:left,x2:width-right,y1:base-rib*.25,y2:base-rib*.25,stroke:'#707070','stroke-width':1,'stroke-dasharray':'4 4'});add(svg,'line',{x1:left,x2:width-right,y1:base,y2:base,stroke:'#303030'});
    add(svg,'circle',{cx:11,cy:20,r:11,fill:'#262626'});[4,10,7].forEach((h,i)=>add(svg,'rect',{x:6+i*4,y:24-h,width:2,height:h,rx:.5,fill:'#aaa'}));add(svg,'text',{x:29,y:25,fill:'#bbb','font-family':'var(--font)','font-size':14},'并行宽度');add(svg,'text',{x:0,y:base-rib*.25+5,fill:'#777','font-family':'var(--mono)','font-size':11},`25% · ${Math.round(total*.25)} 核`);
  }
  for(const g of groups){const groupCores=cores.filter(c=>c.ct===g.type),compute=rank.kpi?.byType?.[g.type]?.computeShare,headerY=offsets.get(g.type)-36,cy=headerY+18;add(svg,'rect',{x:0,y:headerY,width,height:36,fill:'#111'});add(svg,'circle',{cx:11,cy,r:11,fill:'#262626'});if(g.type==='AIC'){for(let r=0;r<2;r++)for(let c=0;c<2;c++)add(svg,'rect',{x:6+c*6,y:cy-5+r*6,width:4,height:4,fill:'none',stroke:'#aaa','stroke-width':1});}else for(let r=0;r<3;r++)add(svg,'line',{x1:6,x2:16,y1:cy-4+r*4,y2:cy-4+r*4,stroke:'#aaa','stroke-width':1});add(svg,'text',{x:29,y:cy+5,fill:'#bbb','font-family':'var(--font)','font-size':14},`${g.type} · ${groupCores.length} 核${Number.isFinite(compute)?' · 计算 '+Math.round(compute*100)+'%':''}`);}
  const chainCores=new Set(state.chain?rank.launches.filter(t=>state.chain.includes(String(t.taskId))).flatMap(t=>t.blocks.items.map(b=>String(b.core))):[]);
  const pos=new Map(); if(state.view==='core') for(const g of groups) cores.filter(c=>c.ct===g.type).forEach((c,i)=>{const yy=offsets.get(g.type)+i*row;pos.set(String(c.id),yy);if(state.expanded||chainCores.has(String(c.id))||i%4===0)add(svg,'text',{x:left-8,y:yy+row/2+4,'text-anchor':'end',fill:'#777','font-size':10},String(c.id));});
  let featureCohorts=state.feature==='recommended'?recommend(rank):state.feature==='none'?[]:featureGroups(rank,state.feature,analyzed);if(state.pathOnly)featureCohorts=featureCohorts.flatMap(group=>(rank.criticalPath||[]).flatMap((p,i)=>{const tasks=group.tasks.filter(t=>String(t.taskId)===String(p.taskId)||(p.ids||[]).includes(t.id));return tasks.length?[{...group,key:group.key+':path:'+p.taskId,kernel:p.name,rank:i+1,tasks}]:[];}));
  const overlay=state.view==='core'&&state.feature!=='none'?drawRecommendationOverlay(svg,featureCohorts,{x,clip,pos,row,width,left,right,height,cores,windowUs:rank.windowUs,gaps:state.feature==='recommended'?feature(rank,'bubble').gaps:analyzed.gaps,focus:state.focusKey,showNames:state.analysisOpen&&state.feature==='recommended',interactive:state.feature==='recommended',onFocus}):null;
  const select=(g,t,core)=>{g.setAttribute('tabindex','0');g.setAttribute('role','button');g.setAttribute('aria-label',`${t.leaf}, ${fmt(t.spanUs)}`);g.addEventListener('click',()=>onSelect(t,core));g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();onSelect(t,core);}});};
  for(const t of rank.launches) {
    const records=state.view==='core'?t.blocks.items:[{startUs:t.startUs,endUs:t.endUs}];
    for(const b of records){const [a,z]=clip(b.startUs,b.endUs);if(z<=a)continue; const yy=state.view==='core'?pos.get(String(b.core)):offsets.get(t.ct)+packed.lanes.get(t.id)*row; if(!Number.isFinite(yy))continue;
      const chain=state.view==='core' ? state.chain : null, chainHit=chain?.includes(String(t.taskId));
      const selected=state.selected===t.id, hit=analyzed.hits.has(t.id), color=chainHit?'#63e6b3':selected?'#f0f0f0':'#565656', opacity=chain?(chainHit?1:.15):selected?1:.85;
      const g=add(svg,'g',{class:chainHit?'event single-hit':'event',opacity}); select(g,t,b.core); const h=state.view==='core'?(chainHit?Math.max(3,Math.min(8,row-2)):state.expanded?Math.min(16,row-2):Math.min(2.6,row-1)):Math.max(3,(row-4)*Math.min(1,t.coresUsed/(cores.filter(c=>c.ct===t.ct).length||1))); const top=yy+row-h-1;
      add(g,'rect',{x:x(a),y:top,width:Math.max(1,x(z)-x(a)),height:h,fill:color,rx:1});
      if(chainHit){for(const phase of blockPhases(b,start,end))add(g,'rect',{x:x(phase.start),y:top,width:Math.max(0,x(phase.end)-x(phase.start)),height:h,fill:{setup:'#d6a75a',compute:'#63e6b3',other:'#777'}[phase.kind],'pointer-events':'none'});}
      if(!chainHit && state.composition && state.view==='core' && Number.isFinite(b.slotUs) && b.slotUs>0 && !selected) {const w=x(z)-x(a), setup=Math.max(0,Math.min(1,b.setupUs/b.slotUs));add(g,'rect',{x:x(a),y:top,width:Math.max(0,w*setup),height:h,fill:'#b48b49'});}
      if((state.view==='task'||state.expanded) && x(z)-x(a)>65 && h>=15) add(g,'text',{x:x(a)+4,y:top+h/2+4,fill:selected?'#111':'#e8e8e8','font-size':11,'pointer-events':'none'},t.leaf.length>(x(z)-x(a)-8)/7?t.leaf.slice(0,Math.max(1,Math.floor((x(z)-x(a)-15)/7)))+'…':t.leaf);
      add(g,'title',{},`${t.leaf}\n${fmt(b.startUs)} → ${fmt(b.endUs)}${b.core==null?'':` · Core ${b.core}`}`);
    }
  }
  overlay?.drawLabels();
  return {hits:analyzed.hits.size, groups:overlay?overlay.groups:kernels.map(key=>({key,color:colors.get(key),count:rank.launches.filter(t=>analyzed.hits.has(t.id)&&t.ct+' · '+t.leaf.replace(/_\d+$/, '')===key).length}))};
}
export function renderCritical(svg, rank, selected, onSelect) {
  svg.replaceChildren();const path=rank.criticalPath||[], width=Math.max(524,svg.parentElement.clientWidth-76), height=30;svg.setAttribute('viewBox',`0 0 ${width} ${height}`);svg.setAttribute('width',width);svg.setAttribute('height',height);
  if(!path.length){add(svg,'text',{x:0,y:25,fill:'#999'},'数据未提供关键路径');return;}
  const total=path.reduce((s,p)=>s+(p.pathContributionUs??p.spanUs??0),0)||1, gap=2, usable=Math.max(1,width-path.length*gap);let x=0;
  for(const p of path){const w=Math.max(0,(p.pathContributionUs??p.spanUs??0)/total*usable);const g=add(svg,'g',{class:'event',tabindex:0,role:'button','aria-label':p.name||p.taskId}); const active=(p.ids||[]).includes(selected);add(g,'rect',{x,y:4,width:w,height:22,fill:active?'#f0f0f0':p.blockNum===1?'#fd5644':'#555'}); if(w>50){const name=p.name||String(p.taskId),n=Math.max(1,Math.floor((w-8)/7));add(g,'text',{x:x+4,y:19,fill:p.blockNum===1?'#111':'#eee','font-size':10},name.length>n?name.slice(0,Math.max(1,n-1))+'…':name);}add(g,'title',{},`${p.name} · 实测跨度 ${fmt(p.spanUs||0)} · 路径新增 ${fmt(p.pathContributionUs??p.spanUs??0)}`);const choose=()=>onSelect(p);g.addEventListener('click',choose);g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose();}});x+=w+gap;}
}

