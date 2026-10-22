const assert=require('node:assert/strict'),fs=require('node:fs');global.window={};require('./data.js');const {metrics,fitRange,boundedRange}=require('./pattern.js');
const data=window.SWIMLANE_DATA;assert.equal(data.steps.length,54);
for(const s of data.steps){assert.equal(s.lanes.length,16);assert.equal(new Set(s.lanes.map(l=>l.dev)).size,16);assert.equal(metrics(s).critical.dev,s.crit);const range=fitRange(s);assert.ok(range[0]<=Math.min(0,...s.lanes.map(l=>l.bindStart)));assert.ok(range[1]>=s.end);for(const l of s.lanes){assert.ok(l.bindStart<=l.bindEnd&&l.bindEnd<=l.rrStart&&l.rrStart<=l.rrEnd+.01&&l.rrEnd<=l.end+.01);assert.ok(!('deviceExec'in l)&&!('waitEstimate'in l));}}
const d=data.steps.find(s=>s.id==='D09'),m=metrics(d);assert.equal(d.periodMs,231.52);assert.equal(m.critical.dev,6);assert.equal(m.critical.rrStart,109.44);assert.ok(Math.abs(m.bindSum-766)<1);assert.equal(Number(m.critical.bindEnd-m.critical.bindStart).toFixed(2),'108.40');
assert.deepEqual(boundedRange(-100,0,[0,200]),[0,100]);assert.deepEqual(boundedRange(190,290,[0,200]),[100,200]);assert.deepEqual(boundedRange(-1000,1000,[0,200]),[0,200]);
const provenance=JSON.parse(fs.readFileSync(__dirname+'/provenance.json'));assert.equal(provenance.validation.rawLaneComparisons,864);
console.log('PASS: 54 steps, 864 lanes, D09 metrics, raw-event provenance, pan/zoom boundaries');
const groups=data.run.workerGroups.filter(g=>g.stepId!=null);
assert.equal(groups.length,53);
assert.equal(new Set(groups.map(g=>g.invocation)).size,53);
for(const g of groups){
  const es=g.eventIds.map(id=>data.run.workerEvents.find(e=>e.id===id));
  assert.equal(es.length,3);assert.ok(es.every(e=>e.stepId===g.stepId));
  assert.equal(g.basis,'temporal-candidate');assert.ok(g.completionGap>=0);
  const step=data.steps.find(s=>s.id===g.step);
  if(step)assert.equal(step.invocation,g.invocation);
}
assert.equal(groups.find(g=>g.step==='D01').stepId,57);
assert.equal(groups.find(g=>g.step==='D08').stepId,64);
assert.equal(groups.find(g=>g.step==='D52').stepId,108);
assert.equal(groups.filter(g=>g.step==null).length,1);
console.log('PASS: 53 unique temporal candidates, Worker phase identities, D01/D08/D52 and excluded D53 boundary');
