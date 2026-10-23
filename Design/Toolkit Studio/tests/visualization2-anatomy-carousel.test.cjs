const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
function fixture(){
  const context=vm.createContext({window:{}});
  for(const name of ['visualization2-task-encoding','visualization2-anatomy'])
    vm.runInContext(fs.readFileSync(path.join(__dirname,'../js',name+'.js'),'utf8'),context);
  const api=context.window.PtoVisualization2Anatomy;
  const model=api.createModel({seqLens:[1025,512,2048,128,1536,3584,3072,256,769,1024,2049,640,2560,129,1792,3328],blocks:24,numHeads:40,kvHeads:8,headDim:128,rowTile:16,pageTokens:128,pagesPerStack:4,transferSlots:3});
  return {api,model};
}
test('Block 03 carousel uses the six real Tasks and the shared Q structure',()=>{
  const {model}=fixture(),tasks=model.tasks.filter(t=>t.block===3);
  assert.deepEqual(Array.from(tasks,t=>t.id),[3,27,51,75,99,123]);
  for(const t of tasks){assert.deepEqual(Array.from(t.qHeads),[15,16,17,18,19]);assert.deepEqual(Array.from(t.q.shape),[16,128]);assert.deepEqual(Array.from(t.q.validShape),[5,128]);}
  for(const [id,tokens,stacks,pages] of [[51,3072,6,24],[27,128,1,1]]){
    const t=model.tasks[id];assert.equal(t.seqLen,tokens);assert.equal(t.stacks.length,stacks);assert.equal(t.pages.length,pages);
  }
});
test('all carousel stacks preserve logical Page capacity and fractional final pages',()=>{
  const {model}=fixture();
  for(const t of model.tasks){
    assert.equal(t.stacks.reduce((sum,s)=>sum+s.tokens,0),t.seqLen);
    assert.equal(t.pages.reduce((sum,p)=>sum+p.tokens,0),t.seqLen);
    for(const s of t.stacks){assert.equal(s.capacityPages.length,4);assert.equal(s.capacityPages.reduce((sum,n)=>sum+n,0),s.tokens);}
    for(const p of t.pages){assert.equal(p.stack,Math.floor(p.index/4));assert.equal(p.physicalId,null);}
  }
  assert.equal(model.tasks[3].pages.at(-1).tokens,1);
});
test('Page focus keeps its Stack relation; nonvisual fields remain available to other views',()=>{
  const {api,model}=fixture(),task=model.tasks[51],before=JSON.stringify(model);
  const r=api.related(task,{kind:'page',index:17});assert.deepEqual(Array.from(r.stacks),[4]);assert.ok(r.pages.includes(17));
  assert.equal(task.slots.length,3);assert.deepEqual(Array.from(task.output.shape),[5,128]);
  assert.equal(JSON.stringify(model),before);
});
