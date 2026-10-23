import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {visualPassDiff} from '../src/visual-diff.js';
const read = name => JSON.parse(fs.readFileSync(new URL(`../data/csa-qk-pv/${name}.json`,import.meta.url)));
test('pass diff excludes hidden source inventories and separately presented ISA result',()=>{
 const passes=read('passes').passes;
 const changed=visualPassDiff(read('graph'),passes).filter(p=>p.changed).map(p=>p.index);
 assert.deepEqual(changed,[1,2,4,5,7,9,10,11,17,19,23,28,33,35,36]);
 for(const pass of passes.filter(p=>p.noOp))assert.ok(!changed.includes(pass.index));
});
test('source provenance changes do not create a visible change',()=>{
 const graph=read('graph');
 assert.equal(visualPassDiff(graph,[{index:1,delta:[{op:'updateNode',id:'value.q',after:{sourceRefs:[{file:'other.py',line:999}]}}]}])[0].changed,false);
});
