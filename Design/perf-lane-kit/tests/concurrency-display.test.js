import {test} from 'node:test';
import assert from 'node:assert/strict';
import {concurrencyBuckets,concurrencyRegions} from '../src/analysis.js';
test('并行宽度按时间加权分桶，并正确裁剪窗口',()=>{const segs=[{start:0,end:5,count:2},{start:5,end:15,count:4},{start:15,end:20,count:0}];assert.deepEqual([...concurrencyBuckets(segs,0,20,2)],[3,2]);assert.deepEqual([...concurrencyBuckets(segs,5,15,2)],[4,4]);});
test('低并行区合并相邻区间并按原版时长阈值过滤',()=>{const segs=[{start:0,end:10,count:1},{start:10,end:20,count:2},{start:20,end:29,count:0},{start:29,end:35,count:3},{start:35,end:40,count:0}];assert.deepEqual(concurrencyRegions(segs,1000),{idle:[{start:20,end:29}],serial:[{start:0,end:20}]});});
