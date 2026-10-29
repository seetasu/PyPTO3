import {test} from 'node:test';
import assert from 'node:assert/strict';
import {findChains,focusRange,blockPhases} from '../src/single-dependency.js';
test('单依赖链去重、排除分支与环',()=>{const edges=[['a','b'],['a','b'],['b','c'],['c','d'],['c','e'],['x','y'],['y','x']].map(([pred,succ])=>({pred,succ}));assert.deepEqual(findChains(edges,['a','b','c','d','e','x','y']),[['a','b','c']]);});
test('聚焦完整任务链并裁剪计时分段',()=>{assert.deepEqual(focusRange(['a','b'],[{taskId:'a',startUs:100,endUs:120},{taskId:'b',startUs:130,endUs:150}],1000),[94,156]);assert.deepEqual(blockPhases({startUs:100,endUs:200,setupUs:70,computeUs:20},120,190),[{kind:'setup',start:120,end:170},{kind:'other',start:170,end:180},{kind:'compute',start:180,end:190}]);assert.deepEqual(blockPhases({startUs:0,endUs:10},0,10),[]);});
