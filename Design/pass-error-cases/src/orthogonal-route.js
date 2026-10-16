// Orthogonal visibility-grid routing around the actual rendered node bounds.
export function routeOrthogonal(source,target,from,to,obstacles){
 const vectors={left:[-1,0],right:[1,0],top:[0,-1],bottom:[0,1]};
 const stub=(point,side)=>({x:point.x+vectors[side][0]*36,y:point.y+vectors[side][1]*36});
 const start=stub(source,from),end=stub(target,to),gap=12;
 const boxes=obstacles.map(r=>({l:r.x-gap,r:r.x+r.width+gap,t:r.y-gap,b:r.y+r.height+gap}));
 const xs=[...new Set([start.x,end.x,...boxes.flatMap(r=>[r.l,r.r])])].sort((a,b)=>a-b);
 const ys=[...new Set([start.y,end.y,...boxes.flatMap(r=>[r.t,r.b])])].sort((a,b)=>a-b);
 const clear=(a,b)=>!boxes.some(r=>a.x===b.x?a.x>r.l+.01&&a.x<r.r-.01&&Math.max(a.y,b.y)>r.t+.01&&Math.min(a.y,b.y)<r.b-.01:a.y>r.t+.01&&a.y<r.b-.01&&Math.max(a.x,b.x)>r.l+.01&&Math.min(a.x,b.x)<r.r-.01);
 const key=(x,y,d)=>`${x}:${y}:${d}`;
 const queue=[{x:xs.indexOf(start.x),y:ys.indexOf(start.y),d:'',cost:0,path:[start]}],best=new Map();
 while(queue.length){
  queue.sort((a,b)=>a.score-b.score);const current=queue.shift();
  const point={x:xs[current.x],y:ys[current.y]};
  if(point.x===end.x&&point.y===end.y){const points=[source,...current.path,target];return points.filter((p,i)=>!i||i===points.length-1||!((points[i-1].x===p.x&&p.x===points[i+1].x)||(points[i-1].y===p.y&&p.y===points[i+1].y)));}
  for(const [dx,dy,d] of [[1,0,'h'],[-1,0,'h'],[0,1,'v'],[0,-1,'v']]){
   const x=current.x+dx,y=current.y+dy;if(x<0||x>=xs.length||y<0||y>=ys.length)continue;
   const next={x:xs[x],y:ys[y]};if(!clear(point,next))continue;
   const cost=current.cost+Math.abs(point.x-next.x)+Math.abs(point.y-next.y)+(current.d&&current.d!==d?24:0),id=key(x,y,d);
   if(cost>=(best.get(id)??Infinity))continue;best.set(id,cost);
   queue.push({x,y,d,cost,score:cost+Math.abs(next.x-end.x)+Math.abs(next.y-end.y),path:[...current.path,next]});
  }
 }
 return null;
}
