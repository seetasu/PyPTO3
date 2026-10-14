const colors={value:'#888',state:'#888',control:'#888'};
export default function ContextDiagram({graph,width,bounds,focusRef,selected,onSelect,onToggle}) {
  const byId=new Map(graph.nodes.map(n=>[n.id,n]));
  return <svg width={width} height={graph.height*width/graph.width} viewBox={`0 0 ${graph.width} ${graph.height}`} aria-label="CSA 计算数据流">
    <defs>{Object.entries(colors).map(([name,color])=><marker key={name} id={`context-arrow-${name}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0L10 5L0 10Z" fill={color}/></marker>)}<pattern id="context-grid" width="13" height="13" patternUnits="userSpaceOnUse"><rect width="10" height="10" fill="#B1EF6B" opacity=".24"/></pattern></defs>
    {graph.clusters.map(c=><g key={c.id}><rect x={c.x} y={c.y} width={c.width} height={c.height} rx="12" fill="none" stroke="#494949" strokeWidth="1.3"/><text x={c.x+14} y={c.y+21} fill="#bbb" fontSize="13">{c.label}</text>{c.foldable&&<g className="context-node" role="button" tabIndex="0" aria-label={`折叠 ${c.label}`} onClick={()=>onToggle(c.id)} onKeyDown={e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();onToggle(c.id);}}}><rect x={c.x+c.width-30} y={c.y+3} width="26" height="26" rx="4" fill="#202020"/><text x={c.x+c.width-17} y={c.y+22} textAnchor="middle" fill="#eee" fontSize="20">−</text></g>}</g>)}
    <g ref={focusRef}><rect {...bounds} rx="9" fill="#6954d918" stroke="#eee" strokeWidth="2"/><text x={bounds.x+10} y={bounds.y+13} fill="#fff" fontSize="11">qk_pv · QK / Softmax / PV</text></g>
    {graph.edges.map((e,i)=>{
      const a=byId.get(e.source),b=byId.get(e.target);if(!a||!b)return null;
      const start={x:a.x,y:a.y+a.height/2},end={x:b.x,y:b.y-b.height/2};
      const blocked=graph.nodes.some(n=>n.id!==a.id&&n.id!==b.id&&Math.abs(n.x-a.x)<n.width/2+8&&n.y>start.y&&n.y<end.y);
      const bypass=end.y<start.y||(a.x===b.x&&blocked);
      const lane=Math.min(a.x-a.width/2,b.x-b.width/2)-24;
      const middle=bypass?[{x:start.x,y:start.y+12},{x:lane,y:start.y+12},{x:lane,y:end.y-12},{x:end.x,y:end.y-12}]:[{x:start.x,y:(start.y+end.y)/2},{x:end.x,y:(start.y+end.y)/2}];
      const ps=[start,...middle,end],kind=e.dashed?'control':a.kind==='tensor'?'state':'value';
      return <path key={i} d={ps.map((p,j)=>`${j?'L':'M'}${p.x},${p.y}`).join(' ')} fill="none" stroke={colors[kind]} opacity=".65" strokeWidth="1.4" strokeDasharray={e.dashed?'4 4':undefined} markerEnd={`url(#context-arrow-${kind})`}><title>{e.label||`${a.label} → ${b.label}`}</title></path>;
    })}
    {graph.nodes.map(n=>{
      const tensor=n.kind==='tensor',comm=n.kind==='communication';
      return <g key={n.id} transform={`translate(${n.x-n.width/2} ${n.y-n.height/2})`} role="button" tabIndex="0" aria-label={n.label} onClick={()=>onSelect(n)} onKeyDown={e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();onSelect(n);}}} className="context-node"><title>{n.detail}</title>
        {comm&&<rect className="context-outline" x="-5" y="-5" width={n.width+10} height={n.height+10} rx={n.height/2+5} fill="none" stroke="#aaa" strokeDasharray="5 4"/>}
        <rect className="context-outline" width={n.width} height={n.height} rx={tensor?5:n.height/2} fill={tensor?'#141414':n.focus?'#6954d9':comm?'#292929':'#656565'} stroke={selected?.id===n.id?'#fff':tensor?'#B1EF6B':'#777'}/>
        {tensor?<><rect x="6" y="6" width={n.width-12} height={n.height-12} fill="url(#context-grid)"/><text x={n.width/2} y={n.height/2-2} textAnchor="middle" fill="#B1EF6B" fontSize="13" paintOrder="stroke" stroke="#141414" strokeWidth="3">{n.label}</text><text x={n.width/2} y={n.height/2+14} textAnchor="middle" fill="#ccc" fontSize="10" paintOrder="stroke" stroke="#141414" strokeWidth="3">{n.subtitle}</text></>:<><circle cx="23" cy={n.height/2} r="13" fill="#111"/><text x="23" y={n.height/2+5} textAnchor="middle" fill="#eee" fontFamily="serif" fontStyle="italic" fontSize="16">{comm?'⇄':'ƒₓ'}</text><text x={n.width/2+14} y={n.height/2+4} textAnchor="middle" fill="#fff" fontSize={n.label.length>30?10.5:12} fontWeight="600">{n.label}</text></>}
        {n.foldable&&<text x={n.width-13} y={n.height/2+5} fill="#fff" fontSize="18" textAnchor="middle">+</text>}
      </g>;
    })}
  </svg>;
}
