import assert from 'node:assert/strict';
import fs from 'node:fs';
const data=JSON.parse(fs.readFileSync(new URL('../data/memory.json',import.meta.url)));
const rows=i=>data[i].rows.filter(r=>r.kernel==='qk_pv_aiv');
assert.equal(rows(33).length,30);assert.equal(rows(35).length,5);
assert.equal(rows(33).reduce((n,r)=>n+r.size,0),280576);
assert.equal(rows(35).reduce((n,r)=>n+r.size,0),49920);
const exp=i=>rows(i).find(r=>r.values.some(v=>v.name==='qk_exp_inline2324__tile'));
assert.notEqual(exp(33).base,exp(35).base);
assert.equal(exp(35).base,'mem_vec_27');assert.equal(exp(36).offset,49920);
assert.equal(data[36].reserves.find(r=>r.kernel==='qk_pv_aiv').size,32768);
for(const r of data[36].rows){assert.ok(r.offset>=0,`${r.id} missing address`);assert.equal(r.offset%32,0);}
console.log('Memory snapshots: counts, byte totals, value tracking, shared bindings, addresses and reservation passed');
