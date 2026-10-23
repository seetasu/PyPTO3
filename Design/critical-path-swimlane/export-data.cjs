// Usage: node export-data.cjs /path/to/PyPTO3-checkout
// Read-only source adapter; output is restricted to this pattern directory.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),cp=require('node:child_process'),assert=require('node:assert/strict');
const source=path.resolve(process.argv[2]);global.window={};
const modulePath='Design/execution-triage-workbench/';
require(path.join(source,modulePath,'data.js'));require(path.join(source,modulePath,'home-data.js'));
const r=window.TRIAGE_HOME.runs[0],raw=JSON.parse(fs.readFileSync(path.join(source,'Data/serving-strace-swimlane(2).json')));
const events=raw.traceEvents.filter(e=>e.ph==='X');
const lanePid=raw.traceEvents.find(e=>e.ph==='M'&&e.name==='process_name'&&/STRACE/.test(e.args.name)).pid;
const end=e=>e.ts+e.dur,round=n=>Math.round(n*100)/100;
let origin=events.find(e=>e.name==='http.completions').ts,verified=0;
const steps=window.TRIAGE_DATA.runs[0].steps.map(s=>{
  const id=s.id==='Prefill'?'P':s.id==='Draft init'?'DI':s.id;
  const f=r.flows[id],spans=events.filter(e=>e.pid===lanePid&&e.args?.invocation===s.inv);
  const phase=e=>e.name.split(' |')[1]?.trim();
  const lanes=f.lanes.map(({deviceExec,waitEstimate,...l})=>{
    const own=spans.filter(e=>e.tid===l.dev),bind=own.find(e=>phase(e)==='bind'),rr=own.find(e=>phase(e)==='runner_run'),run=own.find(e=>phase(e)==='simpler_run');
    for(const [key,t] of Object.entries({bindStart:bind.ts,bindEnd:end(bind),rrStart:rr.ts,end:end(run)}))assert.ok(Math.abs(l[key]-round((t-origin)/1000))<.021,`${s.id}/${l.dev}/${key}`);
    verified++;return {...l,rrEnd:round((end(rr)-origin)/1000)};
  });
  origin=Math.max(...spans.filter(e=>phase(e)==='simpler_run').map(end));
  return {id,...f,invocation:s.inv,lanes};
});
const request=events.find(e=>e.name==='http.completions');
const outputs=events.filter(e=>e.name==='scheduler.process_step_output'&&e.ts>=request.ts&&e.ts<=end(request)).sort((a,b)=>a.ts-b.ts).map((e,i,all)=>({at:round((e.ts-request.ts)/1000),duration:round(e.dur/1000),gap:i?round((e.ts-all[i-1].ts)/1000):null}));
assert.ok(outputs.length>0,'Request must contain server output events');
// Worker IDs are not STRACE invocation IDs. Preserve events and expose temporal
// candidates explicitly; never reuse the source demo's nearest-ended dispatch rule.
const workerEvents=events.filter(e=>/^WorkerProcess\.(prepare_decode|dispatch_decode|reclaim_decode|batch_prefill)$/.test(e.name)).map((e,i)=>({id:'worker-'+i,name:e.name.split('.')[1],role:e.name.endsWith('prepare_decode')?'prepare':e.name.endsWith('reclaim_decode')?'reclaim':'dispatch',stepId:e.args.step_id??null,bufferSlot:e.args.buffer_slot??null,start:round((e.ts-request.ts)/1000),end:round((end(e)-request.ts)/1000)}));
const invocations=[...new Set(events.filter(e=>e.pid===lanePid&&phaseName(e)==='simpler_run').map(e=>e.args.invocation))].map(inv=>{const own=events.filter(e=>e.pid===lanePid&&e.args?.invocation===inv&&phaseName(e)==='simpler_run');return {inv,start:Math.min(...own.map(e=>e.ts)),end:Math.max(...own.map(end))};});
function phaseName(e){return e.name.split(' |')[1]?.trim();}
const workerGroups=workerEvents.filter(e=>e.role==='reclaim').map(reclaim=>{
  const members=workerEvents.filter(e=>e.stepId===reclaim.stepId),dispatch=members.find(e=>e.role==='dispatch');
  const candidates=invocations.filter(v=>v.start>=request.ts+dispatch.start*1000&&v.end<=request.ts+reclaim.end*1000).sort((a,b)=>b.end-a.end);
  const candidate=candidates[0],step=steps.find(s=>s.invocation===candidate?.inv);
  return {id:'worker-step-'+reclaim.stepId,stepId:reclaim.stepId,eventIds:members.map(e=>e.id),step:step?.id??null,invocation:candidate?.inv??null,basis:'temporal-candidate',completionGap:candidate?round(reclaim.end-(candidate.end-request.ts)/1000):null};
});
assert.equal(new Set(workerGroups.map(g=>g.invocation)).size,workerGroups.length,'Temporal candidates must be one-to-one');
const prefill=workerEvents.find(e=>e.name==='batch_prefill');
if(prefill)workerGroups.push({id:'worker-prefill',stepId:null,eventIds:[prefill.id],steps:steps.filter(s=>s.id==='P'||s.id==='DI').map(s=>s.id),basis:'temporal-containment'});
const out={version:2,run:{title:r.title,sub:r.sub,workerRows:r.workerRows,workerEvents,workerGroups,request:{start:0,end:round(request.dur/1000),outputs}},steps};
fs.writeFileSync(path.join(__dirname,'data.js'),'window.SWIMLANE_DATA = '+JSON.stringify(out)+';\n');
const revision=cp.execFileSync('git',['-C',source,'rev-parse','HEAD'],{encoding:'utf8'}).trim();
const sources=['Data/serving-strace-swimlane(2).json',...['build-data.cjs','build-home-data.cjs','data.js','home-data.js','home.js','home.css'].map(p=>modulePath+p)].map(p=>({path:p,sha256:crypto.createHash('sha256').update(fs.readFileSync(path.join(source,p))).digest('hex'),url:`https://github.com/yinyucheng0601/PyPTO3/blob/${revision}/${encodeURI(p)}`}));
fs.writeFileSync(path.join(__dirname,'provenance.json'),JSON.stringify({repository:'https://github.com/yinyucheng0601/PyPTO3',revision,sources,excluded:['deviceExec','waitEstimate'],adaptations:['rrEnd read directly from raw runner_run event; end remains outer simpler_run endpoint','Worker phases joined by step_id; STRACE candidates use latest completion before reclaim end and start after dispatch start, validated one-to-one; temporal association is not causal proof; Prefill uses batch containment; unshown D53 retained as boundary'],validation:{rawLaneComparisons:verified,browser:'not run; visual acceptance reserved for user'}},null,2)+'\n');
console.log(`${steps.length} steps / ${verified} lanes checked against raw STRACE`);
