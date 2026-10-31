import test from 'node:test';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {capabilities} from '../src/data.js';
import {recommend} from '../src/analysis.js';
import {importRun} from '../scripts/import-run.js';

test('缺少 Setup 时仅禁用依赖该数据的特征，推荐仍可运行',async t=>{
  let data;try{data=await importRun(fileURLToPath(new URL('../../../Data/_jit_decode_fwd_layers_20260625_184941/',import.meta.url)));}catch(e){if(e.code==='ENOENT'){t.skip('示例运行目录不可用');return;}throw e;}
  const rank=data.ranks[0], blocks=rank.launches.flatMap(t=>t.blocks.items);
  assert.equal(blocks.length,579);
  assert.ok(blocks.every(b=>Number.isFinite(b.computeUs)&&Number.isFinite(b.setupUs)&&Math.abs(b.slotUs-b.computeUs-b.setupUs)<1e-6));
  assert.ok(Object.values(capabilities(rank).featureModes).every(Boolean));
  const missing={...rank,launches:rank.launches.map(({setupUs,...task})=>({...task,blocks:{...task.blocks,items:task.blocks.items.map(({setupUs,...b})=>b)}}))};
  const c=capabilities(missing);assert.equal(c.featureModes.setup,false);assert.equal(c.featureModes.parallel,false);assert.equal(c.composition,false);for(const mode of ['tail','imbalance','underfill','fragmented','bubble','recommended','none'])assert.equal(c.featureModes[mode],true,mode);
  assert.ok(recommend(missing).every(g=>Number.isFinite(g.score)&&g.evidence.every(e=>Number.isFinite(e.score)&&e.mode!=='setup'&&e.mode!=='parallel')));
});
