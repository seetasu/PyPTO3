import test from 'node:test';
import assert from 'node:assert/strict';
import { deploymentMock } from '../src/deployment-mock.js';
test('deployment is Pass independent and cross-rank edges use communication endpoints',()=>{
 const ranks=[0,1].map(i=>({rankId:`rank${i}`,cores:{aic:24,aiv:48}}));
 const result=deploymentMock({nodes:[],edges:[]},ranks);
 assert.equal(result.nodes.length,38);
 assert.equal(new Set(result.nodes.map(n=>n.id)).size,38);
 assert.deepEqual(result,deploymentMock({nodes:[{id:'arbitrary-pass-object'}]},ranks));
 const ids=new Set(result.nodes.map(n=>n.id));
 for(const edge of result.edges) { assert.ok(ids.has(edge.source)&&ids.has(edge.target)); }
 const cross=result.edges.filter(e=>e.source.split('::')[0]!==e.target.split('::')[0]);
 assert.equal(cross.length,6);
 assert.ok(cross.every(e=>[e.source,e.target].every(id=>result.nodes.find(n=>n.id===id).data.communication)));
 assert.equal(result.nodes.filter(n=>n.type==='scope').length,14);
 assert.ok(cross.every(e=>!e.source.includes('qkpv')&&!e.target.includes('qkpv')));
 assert.ok(result.nodes.every(n=>Number.isFinite(n.position.x)&&Number.isFinite(n.position.y)));
});
