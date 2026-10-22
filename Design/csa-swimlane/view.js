(()=>{
const $=s=>document.querySelector(s),c=$("#chart"),ctx=c.getContext("2d"),mini=$("#mini"),mc=mini.getContext("2d"),scroll=$("#scroll"),panel=$("#panel"),tip=$("#tip");
const all={rank:'All',lanes:[],tasks:[],edges:[],end:Math.max(...CSA_DATA.map(d=>d.end)),unresolved:0,source:'Per-rank trace clocks'};
CSA_DATA.forEach(d=>{const offset=all.tasks.length, laneOffset=all.lanes.length;all.lanes.push(...d.lanes);all.tasks.push(...d.tasks.map(t=>({...t,lane:t.lane+laneOffset,rank:d.rank,family:d.lanes[t.lane].split("_")[0]})));all.edges.push(...d.edges.map(([a,b])=>[a+offset,b+offset]));all.unresolved+=d.unresolved});
const analysis=CSAAnalysis.build(all.tasks);let colorMode="diagnostic";
const logicalNeighbors=new Map();
let offset=0;
CSA_DATA.forEach(d=>{for(const [a,b] of d.logical||[]){for(const [from,to,direction] of [[a,b,1],[b,a,-1]])for(const i of d.groups[from]){const list=logicalNeighbors.get(i+offset)||[];list.push({indices:d.groups[to].map(j=>j+offset),direction});logicalNeighbors.set(i+offset,list)}}offset+=d.tasks.length});
function linksFor(focus){if(focus==null)return [];const direct=data.edges.filter(([a,b])=>a===focus||b===focus).map(e=>[...e,false]);const t=data.tasks[focus];for(const g of logicalNeighbors.get(focus)||[]){if(direct.some(([a,b])=>g.indices.includes(a===focus?b:a)))continue;const j=g.indices.reduce((best,j)=>Math.abs(data.tasks[j].lane-t.lane)<Math.abs(data.tasks[best].lane-t.lane)?j:best,g.indices[0]);direct.push(g.direction===1?[focus,j,true]:[j,focus,true])}return direct}
let rankFocus=null;let data=all,lo=0,span=data.end,selected=null,hover=null,w=0,h=0,drag=null;const left=140,top=110;let row=12,expanded=false;
const blocks=[];let laneOffset=0;
CSA_DATA.forEach(d=>{const aic=d.lanes.filter(n=>n.startsWith('AIC_')).length;blocks.push({rank:d.rank,first:laneOffset,last:laneOffset+d.lanes.length-1,aic});laneOffset+=d.lanes.length});
const y=i=>{const b=blocks.find(b=>i<=b.last)||blocks.at(-1),bi=blocks.indexOf(b);return top+i*row+bi*164+(i-b.first>=b.aic?48:0)},x=t=>left+(t-lo)/span*(w-left-24),time=v=>lo+(v-left)/(w-left-24)*span;
function size(){w=scroll.clientWidth;row=expanded?28:Math.max(3,(scroll.clientHeight-top-78-(blocks.length-1)*164)/Math.max(1,data.lanes.length-1));if(rankFocus!==null&&!expanded)row=Math.max(7,(scroll.clientHeight-top-80)/Math.max(1,blocks.find(b=>b.rank===rankFocus).last-blocks.find(b=>b.rank===rankFocus).first));h=y(data.lanes.length-1)+30;const d=devicePixelRatio||1;c.width=w*d;c.height=h*d;c.style.width=w+"px";c.style.height=h+"px";ctx.setTransform(d,0,0,d,0,0);mini.width=mini.clientWidth*d;mini.height=30*d;mc.setTransform(d,0,0,d,0,0);draw()}
function line(a,b,color,width=1){ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}
function draw(){window.refreshMenus?.();ctx.clearRect(0,0,w,h);
// Each Rank is one quiet surface, including its labels and both core families.
ctx.fillStyle="#111";
for(const {first,last} of blocks){
 const bottomPadding=Math.max(16,row/2);
 ctx.beginPath();
 ctx.roundRect(24,y(first)-84,Math.max(0,w-48),y(last)-y(first)+84+bottomPadding,6);
 ctx.fill();
}
ctx.font="12px monospace";ctx.fillStyle="#aaa";ctx.textAlign="left";function group(label,cy){ctx.save();ctx.translate(24,0);ctx.fillStyle="#262626";ctx.beginPath();ctx.arc(32,cy-5,11,0,Math.PI*2);ctx.fill();ctx.strokeStyle="#aaa";ctx.lineWidth=1;
if(label==='AIC'){for(let r=0;r<2;r++)for(let col=0;col<2;col++)ctx.strokeRect(27+col*6,cy-10+r*6,4,4)}else{for(let r=0;r<3;r++){ctx.beginPath();ctx.moveTo(27,cy-9+r*4);ctx.lineTo(37,cy-9+r*4);ctx.stroke()}}
ctx.font="14px system-ui";ctx.fillStyle="#bbb";ctx.textAlign='left';ctx.fillText(label,49,cy);ctx.restore()}
blocks.forEach(({rank:r,first,aic})=>{ctx.save();ctx.textAlign='left';ctx.fillStyle='#eee';ctx.font='500 16px system-ui';ctx.fillText('Rank '+r,44,y(first)-62);group('AIC',y(first)-26);group('AIV',y(first+aic)-26);ctx.restore()});const desired=span/Math.max(1,Math.floor((w-left-24)/100)),base=10**Math.floor(Math.log10(desired)),tick=([1,2,5,10].find(n=>n*base>=desired)||10)*base;for(let t=Math.ceil(lo/tick)*tick;t<lo+span;t+=tick){if(x(t)>w-70)continue;line([x(t),8],[x(t),20],"#666");ctx.fillText(t.toFixed(tick<1?2:0),x(t)+8,18)}
ctx.textAlign="right";ctx.fillText("µs",w-24,18);const focus=hover??selected,related=new Set(focus==null?[]:[focus]);if(focus!=null)linksFor(focus).forEach(([a,b])=>{if(a===focus||b===focus){related.add(a);related.add(b)}});
const focusLane=focus==null?null:data.tasks[focus].lane;
data.lanes.forEach((name,i)=>{
 const active=i===focusLane;
 const overlapsFocus=focusLane!==null&&!active&&Math.abs(y(i)-y(focusLane))<16;
 if(overlapsFocus)return;
 ctx.fillStyle=active?"#eee":"#999";ctx.textAlign="right";
 if(active||row>=18||i%4===0)ctx.fillText(name.split("_")[1],left-18,y(i)+4);
});
ctx.save();ctx.beginPath();ctx.rect(left,36,w-left-24,h-36);ctx.clip();
data.tasks.forEach((t,i)=>{if(t.end<lo||t.start>lo+span)return;const a=x(t.start),b=x(t.end);const color=taskColor(i,focus!=null&&!related.has(i));if(expanded){const height=i===focus?18:16;ctx.fillStyle=color;ctx.beginPath();ctx.roundRect(a,y(t.lane)-height/2,Math.max(1,b-a),height,2);ctx.fill()}else line([a,y(t.lane)],[Math.max(a+1,b),y(t.lane)],color,i===focus?4:Math.min(3,row*.28));if(expanded&&b-a>48){ctx.save();const start=Math.max(left,a),available=Math.min(w-24,b)-start-8;ctx.beginPath();ctx.rect(start,y(t.lane)-8,Math.max(0,available),16);ctx.clip();ctx.textAlign="left";ctx.fillStyle="#fff";ctx.shadowColor="#0009";ctx.shadowBlur=2;let label=t.name;while(label.length&&ctx.measureText(label+"…").width>available)label=label.slice(0,-1);ctx.fillText(label===t.name?label:label+"…",start+3,y(t.lane)+4);ctx.restore()}});
if(focus!=null)linksFor(focus).forEach(([a,b,logical])=>{if(a!==focus&&b!==focus)return;const s=data.tasks[a],t=data.tasks[b],sx=x(s.end),tx=x(t.start),sy=y(s.lane),ty=y(t.lane);ctx.strokeStyle="#ddd";ctx.lineWidth=1;ctx.setLineDash(logical?[2,5]:[6,4]);ctx.beginPath();ctx.moveTo(sx,sy);ctx.lineTo((sx+tx)/2,sy);ctx.lineTo((sx+tx)/2,ty);ctx.lineTo(tx,ty);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle="#eee";ctx.beginPath();ctx.moveTo(tx,ty);ctx.lineTo(tx-4,ty-3);ctx.lineTo(tx-4,ty+3);ctx.fill()});ctx.restore();
const mw=mini.clientWidth;mc.clearRect(0,0,mw,30);const bins=new Array(Math.max(1,Math.floor(mw/3))).fill(0);data.tasks.forEach(t=>{for(let i=Math.floor(t.start/data.end*bins.length);i<=Math.min(bins.length-1,Math.floor(t.end/data.end*bins.length));i++)bins[i]++});const peak=Math.max(1,...bins);mc.fillStyle="#666";bins.forEach((n,i)=>{const bh=3+n/peak*21;mc.fillRect(i*mw/bins.length,27-bh,2,bh)});nav.update({range:[lo,lo+span]});
}
function range(start,length){span=Math.max(data.end/300,Math.min(data.end,length));lo=Math.max(0,Math.min(data.end-span,start));draw()}
function zoom(f,anchor=.5){const t=lo+span*anchor;range(t-span*f*anchor,span*f)}
function hit(e){const b=c.getBoundingClientRect(),px=e.clientX-b.left,py=e.clientY-b.top;if(px<left)return null;let best=null;data.tasks.forEach((t,i)=>{if(Math.abs(y(t.lane)-py)<Math.min(9,row/2)&&px>=x(t.start)-2&&px<=x(t.end)+2)best=i});return best}
function inspect(i){const t=data.tasks[i],ins=data.edges.filter(e=>e[1]===i).length,outs=data.edges.filter(e=>e[0]===i).length;panel.hidden=false;panel.replaceChildren();const close=document.createElement("button");close.textContent="×";close.onclick=()=>{panel.hidden=true;selected=null;draw()};panel.append(close);const title=document.createElement("h2");title.textContent=t.name;panel.append(title);const content=document.createElement("div");content.innerHTML=`<p>Rank ${t.rank} · ${data.lanes[t.lane]}</p><dl><dt>Start</dt><dd>${t.start.toFixed(2)} µs</dd><dt>Duration</dt><dd>${(t.end-t.start).toFixed(2)} µs</dd><dt>Task ID</dt><dd>${t.task}</dd><dt>Incoming / outgoing</dt><dd>${ins} / ${outs}</dd></dl><p>Execution within visible window</p><svg width="280" height="36" viewBox="0 0 280 36"><path d="M0 22H280" stroke="#555"/><path d="M${Math.max(0,(t.start-lo)/span*280)} 22H${Math.min(280,(t.end-lo)/span*280)}" stroke="#eee" stroke-width="6"/></svg>`;panel.append(content);const evidence=document.createElement("section");evidence.innerHTML=CSAAnalysis.detail(analysis,i);panel.append(evidence)}
c.onpointermove=e=>{if(drag){range(drag.lo-(e.clientX-drag.x)/(w-left-24)*span,span);return}hover=hit(e);draw();tip.hidden=hover==null;if(hover!=null){const t=data.tasks[hover];tip.textContent=`${t.name}\n${analysis[hover].type} · ${analysis[hover].level ? "Duration deviation " + analysis[hover].ratio.toFixed(2)+"× median" : "No duration flag"}\nRank ${t.rank} · ${data.lanes[t.lane]} · ${(t.end-t.start).toFixed(2)} µs\n${t.start.toFixed(2)} → ${t.end.toFixed(2)} µs\n${linksFor(hover).filter(e=>!e[2]).length} trace flows · ${linksFor(hover).filter(e=>e[2]).length} logical task links`;tip.style.left=Math.max(8,Math.min(innerWidth-360,e.clientX+16))+"px";tip.style.top=Math.min(innerHeight-110,e.clientY+16)+"px"}};
c.onpointerleave=()=>{hover=null;tip.hidden=true;draw()};c.onpointerdown=e=>{drag={x:e.clientX,lo};c.setPointerCapture(e.pointerId)};c.onpointerup=e=>{if(drag&&Math.abs(e.clientX-drag.x)<4){selected=hit(e);if(selected!=null)inspect(selected)}drag=null;draw()};c.onpointercancel=()=>drag=null;
c.addEventListener("wheel",e=>{if(e.ctrlKey||e.metaKey){e.preventDefault();zoom(Math.exp(e.deltaY*.005),Math.max(0,Math.min(1,(e.offsetX-left)/(w-left-24))))}else if(e.shiftKey){e.preventDefault();range(lo+e.deltaY/(w-left-24)*span,span)}},{passive:false});
c.onkeydown=e=>{if(e.key==="Escape"){panel.hidden=true;selected=null;draw()}if(e.key==="ArrowRight"||e.key==="ArrowLeft"){e.preventDefault();range(lo+span*.1*(e.key==="ArrowRight"?1:-1),span)}};
function syncRowsButton(){const b=$("#rows"),label=expanded?'Compact rows':'Expand rows';b.setAttribute('aria-pressed',String(expanded));b.setAttribute('aria-label',label);b.removeAttribute("title")}
$("#plus").onclick=()=>zoom(.65);$("#minus").onclick=()=>zoom(1/.65);$("#fit").onclick=()=>{expanded=false;rankFocus=null;$("#rank").value="all";syncRowsButton();scroll.scrollTop=0;range(0,data.end);size()};$("#rows").onclick=()=>{
 const lane=selected==null?null:data.tasks[selected].lane;
 const viewportY=lane==null?null:y(lane)-scroll.scrollTop;
 expanded=!expanded;syncRowsButton();size();
 if(lane!==null)scroll.scrollTop=y(lane)-viewportY;
 else if(rankFocus!==null)scroll.scrollTop=y(blocks.find(b=>b.rank===rankFocus).first)-50;
};const nav=SequenceNavigation.mount($("#navigator"),{items:[],variant:'time',full:[0,data.end],onRange:([a,b])=>range(a,b-a)});
mini.onclick=e=>range(e.offsetX/mini.clientWidth*data.end-span/2,span);
function focusRank(value){rankFocus=value==='all'?null:Number(value);$("#rank").value=value;size();scroll.scrollTop=rankFocus===null?0:y(blocks.find(b=>b.rank===rankFocus).first)-50}
$("#rank").onchange=e=>focusRank(e.target.value);
c.ondblclick=e=>{const b=c.getBoundingClientRect(),py=e.clientY-b.top;focusRank(String((blocks.find(b=>py<y(b.last)+30)||blocks.at(-1)).rank));if(e.offsetX>=left)zoom(.5,Math.max(0,Math.min(1,(e.offsetX-left)/(w-left-24))))};
$("#info").onclick=()=>{panel.hidden=false;panel.innerHTML=`<button onclick="this.parentElement.hidden=true">×</button><h2>读图指南</h2><p>查看当前文件的核心执行：哪些核心耗时更长，task 之间如何衔接。</p><dl><dt>Rank</dt><dd>一张计算卡</dd><dt>AIC / AIV</dt><dd>矩阵 / 向量计算核心</dd><dt>横轴 / 条长</dt><dd>时间 / 耗时（µs）</dd></dl><h3 class="evidence-heading">颜色</h3><div id="legend" aria-live="polite">${legend()}</div><h3 class="evidence-heading">检查任务</h3><p>悬停看依赖连线；点击看耗时与跨核对比。展开行高可读 task 名称。</p><h3 class="evidence-heading">浏览画布</h3><p>拖动平移 · ⌘ / Ctrl + 滚轮缩放<br>滚动查看核心 · Fit 返回全景</p>`};
// Compact strokes need more luminance than expanded area fills.
const compactColors=Object.fromEntries(Object.entries(CSAAnalysis.colors).map(([key,hex])=>[key,brightStroke(hex)]));
function brightStroke(hex){
 const rgb=hex.slice(1).match(/../g).map(v=>parseInt(v,16)/255),max=Math.max(...rgb),min=Math.min(...rgb),d=max-min;
 if(!d)return hex;
 const l=(max+min)/2,s=d/(1-Math.abs(2*l-1));
 let hue=max===rgb[0]?((rgb[1]-rgb[2])/d+6)%6:max===rgb[1]?(rgb[2]-rgb[0])/d+2:(rgb[0]-rgb[1])/d+4;
 const light=Math.min(.76,l+.13),sat=Math.min(1,s+.10),chroma=(1-Math.abs(2*light-1))*sat,second=chroma*(1-Math.abs(hue%2-1)),m=light-chroma/2;
 const channels=hue<1?[chroma,second,0]:hue<2?[second,chroma,0]:hue<3?[0,chroma,second]:hue<4?[0,second,chroma]:hue<5?[second,0,chroma]:[chroma,0,second];
 return '#'+channels.map(v=>Math.round((v+m)*255).toString(16).padStart(2,'0')).join('');
}
function taskColor(i,dim){const a=analysis[i],base=colorMode==='type'?(expanded?CSAAnalysis.colors:compactColors)[a.type]:CSAAnalysis.severity[a.level];return dim?base+'66':base}
function legend(){const entries=colorMode==='type'?Object.entries(CSAAnalysis.colors):[['Baseline','#888888'],['≥1.5× median','#FBDC59'],['≥2× median','#ffaa3b'],['≥3× median','#ff4b7b']];return entries.map(([name,color])=>`<span><i style="background:${color}"></i>${name}</span>`).join('')+(colorMode==='diagnostic'?'<small>颜色越暖，耗时相对同类核心越长；点击查看对比。</small>':'<small>相同颜色表示相同的算子主要语义。</small>')}
$("#color-mode").onchange=e=>{colorMode=e.target.value;if($("#legend"))$("#legend").innerHTML=legend();draw()};window.addEventListener("resize",size);syncRowsButton();size();
})();
