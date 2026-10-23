const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
function fixture(index=2){
  const context=vm.createContext({window:{}});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../js/visualization2-anatomy.js'),'utf8'),context);
  let moving=false,current=index,target=index,commits=[];
  const paging=context.window.PtoVisualization2Anatomy.createPagingInput({isAnimating:()=>moving,onPage:direction=>{
    const next=Math.max(0,Math.min(5,target+direction));if(next!==target){target=next;moving=true;}
  }});
  return {paging,send(delta,time,discrete=false){paging.input({delta,time,discrete});},
    finish(){if(moving){moving=false;current=target;commits.push(current);paging.flush();}},
    get target(){return target;},get commits(){return commits;}};
}
test('one trackpad burst and its long decaying momentum issue one command',()=>{
  const f=fixture();[[40,0],[70,16],[60,32],[40,48],[20,64],[8,80],[5,100],[3,150],[2,210]].forEach(([d,t])=>f.send(d,t));
  f.finish();f.send(3,250);f.send(2,300);f.finish();assert.deepEqual(f.commits,[3]);
});
test('renewed acceleration during animation queues one next page without silence',()=>{
  const f=fixture();f.send(45,0);f.send(60,16);f.send(20,32);f.send(6,48);
  f.send(50,100);f.send(65,116);f.send(25,132);f.send(5,148);
  assert.equal(f.target,3);assert.equal(f.paging.getState().pending,1);
  f.finish();assert.equal(f.target,4);f.finish();assert.deepEqual(f.commits,[3,4]);
});
test('two quick mouse notches queue two pages, never an unbounded queue',()=>{
  const f=fixture();f.send(48,0,true);f.send(48,60,true);f.send(48,120,true);
  f.finish();f.finish();f.finish();assert.deepEqual(f.commits,[3,4]);assert.equal(f.paging.getState().pending,0);
});
test('meaningful reversal is accepted but tiny opposite noise is ignored',()=>{
  const f=fixture();f.send(50,0);f.send(-1,20);assert.equal(f.paging.getState().pending,0);
  f.send(-35,40);f.finish();f.finish();assert.deepEqual(f.commits,[3,2]);
});
test('separated gestures rearm without clicks; reset cancels pending command',()=>{
  const f=fixture();f.send(40,0);f.finish();f.send(40,190);assert.equal(f.target,4);
  f.send(-40,210);f.paging.reset();f.finish();f.finish();assert.deepEqual(f.commits,[3,4]);
});
test('boundary input does not lock future reverse paging',()=>{
  const f=fixture(5);f.send(50,0);assert.equal(f.target,5);f.send(-40,40);f.finish();assert.deepEqual(f.commits,[4]);
});

test('a strong new stroke is recognized with no quiet interval between events',()=>{
  const f=fixture();[[45,0],[65,16],[20,32],[6,48],[48,64],[70,80],[18,96],[4,112]].forEach(([d,t])=>f.send(d,t));
  f.finish();f.finish();assert.deepEqual(f.commits,[3,4]);
});
