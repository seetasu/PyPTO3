// Static rank projection of the existing context graph; not runtime instances.
export function expandContextRanks(graph) {
  const gap=260, stride=graph.width+gap;
  const nodes=[],edges=[],clusters=[];
  for(let rank=0;rank<2;rank++) {
    const prefix=`rank${rank}::`, dx=rank*stride;
    nodes.push(...graph.nodes.map(n=>({...n,id:prefix+n.id,sourceId:n.id,rankId:`rank${rank}`,x:n.x+dx,y:n.y+60})));
    edges.push(...graph.edges.map(e=>({...e,source:prefix+e.source,target:prefix+e.target})));
    clusters.push({id:prefix+'frame',label:`Rank ${rank} · 本地输入 / 本地输出 · 静态映射 Mock`,x:dx+4,y:4,width:graph.width-8,height:graph.height+90});
    clusters.push(...graph.clusters.map(c=>({...c,id:prefix+c.id,sourceId:c.id,x:c.x+dx,y:c.y+60})));
  }
  for(const id of ['tp_allgather','attention_alltoall','output_reduce_scatter']) {
    if(!graph.nodes.some(n=>n.id===id))continue;
    for(const [a,b] of [[0,1],[1,0]])edges.push({source:`rank${a}::${id}`,target:`rank${b}::${id}`,crossRank:true,dashed:true,label:'跨 rank 数据交换 · 信号协议未展开'});
  }
  return {...graph,width:graph.width*2+gap,height:graph.height+100,nodes,edges,clusters};
}

// Route through free horizontal space, with separate directions above/below.
export function routeRankEdge(a,b,nodes) {
  const forward=a.x<b.x, clearance=12;
  const obstacles=nodes.filter(n=>n.id!==a.id&&n.id!==b.id);
  const clear=(p,q)=>!obstacles.some(n=>{
    const l=n.x-n.width/2-clearance,r=n.x+n.width/2+clearance;
    const t=n.y-n.height/2-clearance,d=n.y+n.height/2+clearance;
    return p.x===q.x ? p.x>l&&p.x<r&&Math.max(p.y,q.y)>t&&Math.min(p.y,q.y)<d
      : p.y>t&&p.y<d&&Math.max(p.x,q.x)>l&&Math.min(p.x,q.x)<r;
  });
  const sign=forward?-1:1;
  const start={x:a.x+(forward?8:-8),y:a.y+sign*(a.height/2+5)};
  const end={x:b.x+(forward?8:-8),y:b.y+sign*(b.height/2+5)};
  const candidates=[start.y+sign*22,...nodes.flatMap(n=>[n.y-n.height/2-clearance-1,n.y+n.height/2+clearance+1])]
    .filter(y=>sign*(y-start.y)>0&&sign*(y-end.y)>0).sort((x,y)=>Math.abs(x-start.y)-Math.abs(y-start.y));
  for(const y of candidates){const ps=[start,{x:start.x,y},{x:end.x,y},end];if(ps.slice(1).every((p,i)=>clear(ps[i],p)))return ps;}
  throw new Error(`No clear rank route: ${a.id} -> ${b.id}`);
}
