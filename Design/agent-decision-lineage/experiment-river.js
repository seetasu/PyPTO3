/* Geometry comes from layout anchors; widths encode hierarchy only. */
window.ExperimentRiver={positions(view){return [view.river.finding,view.river.failed,...view.river.opportunities]},render({svg,view,selected,related,choose,layer='all'}){
 const NS='http://www.w3.org/2000/svg',r=view.river,G=window.FlowGeometry;
 const el=(tag,attrs={},text)=>{const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text)n.textContent=text;return n};
 const path=(d,kind,ids)=>{const p=el('path',{d,class:'experiment-river '+kind,'data-region':'optimize','data-related':ids.join(' ')});svg.append(p);return p};
 const node=(p,kind)=>{const g=el('g',{class:'node river-node '+kind+(selected===p.id?' selected':''),'data-node-id':p.id,'data-region':'optimize',tabindex:0,role:'button','aria-label':p.label+' '+p.sub});g.append(el('rect',{x:p.x,y:p.y,width:p.w,height:p.h,rx:9}));if(kind==='opportunity')g.append(el('text',{x:p.x+8,y:p.y+13,class:'river-label'},'○ '+p.label+' · 未尝试'));else g.append(el('text',{x:p.x+9,y:p.y+21,class:'river-label'},p.label),el('text',{x:p.x+9,y:p.y+40,class:'river-sub'},p.sub));if(selected&&!related.has(p.id))g.style.opacity='.4';const activate=e=>{e.stopPropagation();choose(p.id)};g.onclick=activate;g.onkeydown=e=>{if(['Enter',' '].includes(e.key)){e.preventDefault();activate(e)}};svg.append(g);for(const t of g.querySelectorAll('text'))G.fitText(t,p.w-18)};
 const base=view.groups.find(g=>g.id==='baseline'),stop=view.groups.find(g=>g.id==='stop'),f=r.finding,p=r.failed;
 if(layer==='all'||layer==='connections'){
 path(G.curve({x:base.x+base.w/2,y:f.y+f.h},{x:base.x+base.w/2,y:base.y}),'finding-stem',['f-mask','x-base']);
 const start=G.junction(view),end={x:p.x,y:p.y+p.h/2};path(G.curve(start,end),'failed',['x-if','x-mask']);
 const opp=r.opportunities,groupX=opp[0].x,groupY=opp[0].y-24;
 // Short Stop Review bracket, not a reverse execution river. Lanes are parallel and independent.
 path(`M${stop.x+stop.w/2} ${stop.y+stop.h} V${groupY-20} H${groupX+120} V${groupY-10}`,'review-bracket',['d-stop',...opp.map(p=>p.id)]);
 const title=el('text',{x:groupX,y:groupY,class:'river-opportunity-label','data-region':'optimize','data-related':'d-stop '+opp.map(p=>p.id).join(' ')},'STOP REVIEW · OPEN OPPORTUNITIES · 未执行');svg.append(title);G.fitText(title,opp[0].w);
 for(const lane of opp){const y=lane.y+26;path(`M${lane.x} ${y} H${lane.x+lane.w-22}`,'opportunity',['d-stop',lane.id]);const defs=el('defs'),fade=el('linearGradient',{id:'fade-'+lane.id,gradientUnits:'userSpaceOnUse',x1:lane.x+lane.w-22,y1:y,x2:lane.x+lane.w,y2:y});fade.append(el('stop',{offset:'0%','stop-color':'var(--foreground-muted)','stop-opacity':'.45'}),el('stop',{offset:'100%','stop-color':'var(--foreground-muted)','stop-opacity':'0'}));defs.append(fade);svg.append(defs);const tail=path(`M${lane.x+lane.w-22} ${y} H${lane.x+lane.w}`,'open-end',['d-stop',lane.id]);tail.style.stroke='url(#fade-'+lane.id+')';}
 }
 if(layer==='all'||layer==='nodes'){node(f,'finding');node(p,'failed');for(const lane of r.opportunities)node(lane,'opportunity');path(`M${p.x+p.w} ${p.y+8} V${p.y+p.h-8}`,'rejected-cap',['x-if']);}
}};
