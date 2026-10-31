/* Presentation geometry only. Dead-end / open branches never merge into Final.
   Widths encode narrative hierarchy, not counts, performance or likelihood. */
window.ExperimentRiver={positions(view){return [view.river.finding,view.river.failed,...view.river.opportunities]},render({svg,view,selected,related,choose}){
 const NS='http://www.w3.org/2000/svg',r=view.river;
 const el=(tag,attrs={},text)=>{const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text)n.textContent=text;return n};
 const defs=el('defs'),fade=el('linearGradient',{id:'river-open-fade',gradientUnits:'userSpaceOnUse',x1:0,y1:468,x2:0,y2:486});fade.append(el('stop',{offset:'0%','stop-color':'var(--foreground-muted)','stop-opacity':'.5'}),el('stop',{offset:'100%','stop-color':'var(--foreground-muted)','stop-opacity':'0'}));defs.append(fade);svg.append(defs);
 const path=(d,kind)=>{const p=el('path',{d,class:'experiment-river '+kind});svg.append(p);return p};
 const node=(p,kind)=>{const g=el('g',{class:'node river-node '+kind+(selected===p.id?' selected':''),'data-node-id':p.id,tabindex:0,role:'button','aria-label':p.label+' '+p.sub});g.append(el('rect',{x:p.x,y:p.y,width:p.w,height:p.h,rx:9}),el('text',{x:p.x+9,y:p.y+20,class:'river-label'},p.label),el('text',{x:p.x+9,y:p.y+37,class:'river-sub'},p.sub));if(selected&&!related.has(p.id))g.style.opacity='.4';const activate=e=>{e.stopPropagation();choose(p.id)};g.onclick=activate;g.onkeydown=e=>{if(['Enter',' '].includes(e.key)){e.preventDefault();activate(e)}};svg.append(g)};
 // Finding annotates the measured-baseline segment; light stem has no extra experiment.
 path(`M1155 245 C1155 225 1155 218 1155 212`,'finding-stem');node(r.finding,'finding');
 // Failed alternative leaves the candidate junction before implementation; closed red cap.
 const fx=r.failed.x,fy=r.failed.y+r.failed.h/2;
 path(`M1196 301 C1215 301 1206 ${fy} ${fx} ${fy}`,'failed');
 path(`M${fx+r.failed.w+3} ${fy-16} L${fx+r.failed.w+3} ${fy+16}`,'rejected-cap');node(r.failed,'failed');
 // Stop opens a review of known opportunities, not executed successors. Parallel fanout.
 const sx=1600,sy=357;
 svg.append(el('text',{x:1460,y:402,class:'river-opportunity-label'},'OPEN OPPORTUNITIES · 未执行，非失败'));
 for(const p of r.opportunities){const ex=p.x+p.w/2;path(`M${sx} ${sy} C${sx} 390 ${ex} 379 ${ex} ${p.y}`,'opportunity');node(p,'opportunity');path(`M${ex} ${p.y+p.h} L${ex} ${p.y+p.h+14}`,'open-end');}
 svg.append(el('text',{x:1060,y:122,class:'river-key'},'主河=实际路径　封口=尝试失败　淡出=未探索 · 宽度不编码数量'));
}};
