import fs from 'node:fs';
import assert from 'node:assert/strict';
const read=name=>JSON.parse(fs.readFileSync(new URL('../data/callable/'+name,import.meta.url)));
const snapshots=read('snapshots.json'),isa=read('isa.json');
assert.equal(snapshots.length,52);
for(const s of snapshots){
 const ids=new Set(s.nodes.map(n=>n.id));
 for(const e of s.edges){assert(ids.has(e.source));assert(ids.has(e.target));}
 assert(!s.nodes.some(n=>n.id==='kernel.aic'||n.id==='op.qk'));
 for(const row of s.rows)for(const owner of row.owners)assert(ids.has(owner));
}
assert.equal(snapshots[33].rows.length,53);assert.equal(snapshots[35].rows.length,12);
assert.equal(snapshots[35].rows.reduce((sum,r)=>sum+r.size,0),165120);
const occurrences=isa.instructions.flatMap(i=>i.occurrences);
assert.equal(occurrences.length,71);
assert.equal(occurrences.filter(o=>o.text.includes('pto.comm.tput(')).length,2);
assert(occurrences.every(o=>snapshots[51].nodes.some(n=>n.id===o.owner)));
console.log('52 callable states, graph endpoints, UB ownership and 71 ISA occurrences validated');
