import assert from 'node:assert/strict';
import {routeOrthogonal} from '../src/orthogonal-route.js';
const obstacles=[{x:0,y:0,width:100,height:50},{x:180,y:0,width:120,height:50},{x:400,y:0,width:100,height:50}];
const source={x:100,y:25},target={x:400,y:25};
const path=routeOrthogonal(source,target,'right','left',obstacles);
assert(path&&path.length>2);assert.deepEqual(path[0],source);assert.deepEqual(path.at(-1),target);
for(let i=1;i<path.length;i++){
 const a=path[i-1],b=path[i];assert(a.x===b.x||a.y===b.y,'orthogonal segments');
 for(const r of obstacles.slice(1,2))assert(!(a.x===b.x?a.x>r.x&&a.x<r.x+r.width&&Math.max(a.y,b.y)>r.y&&Math.min(a.y,b.y)<r.y+r.height:a.y>r.y&&a.y<r.y+r.height&&Math.max(a.x,b.x)>r.x&&Math.min(a.x,b.x)<r.x+r.width),'route avoids unrelated glyph');
}
console.log('Routing: endpoint identity, orthogonal segments and obstacle avoidance passed');
