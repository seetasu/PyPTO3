import {calculateCriticalPath} from './critical-path.js';
const hint=/Task:(\S+?),\s*FuncId:\s*(\d+),\s*CoreId:\s*(\d+)/;
const q=(a,f)=>{const b=a.slice().sort((x,y)=>x-y);if(!b.length)return 0;const i=(b.length-1)*f,l=Math.floor(i);return b[l]+(b[Math.ceil(i)]-b[l])*(i-l);};
export function buildCapture({rank,source,trace,map,raw,depFile}){
  const types=raw.metadata?.core_types;if(!Array.isArray(types)||!Array.isArray(raw.aicore_tasks)||!Array.isArray(trace.traceEvents))throw Error(`${rank}: 缺少核元数据或事件`);
  const candidates=trace.traceEvents.filter(e=>e.cat==='event'&&e.ph==='X'&&hint.test(e.args?.['event-hint']||'')&&Number.isFinite(e.ts)&&Number.isFinite(e.dur));
  const detailed=candidates.filter(e=>Number.isFinite(e.args['kernel-duration-us']));const timed=candidates.filter(e=>Number.isFinite(e.args['duration-us']));const chosen=detailed.length?detailed:timed;
  if(chosen.length!==raw.aicore_tasks.length)throw Error(`${rank}: 可用事件 ${chosen.length} 条，与核记录 ${raw.aicore_tasks.length} 条不一致，无法安全去重`);
  if(!chosen.length)throw Error(`${rank}: 没有核事件`);
  // Match older separate setup events against the trace and raw cycle records.
  const setups=new Map(),paired=new Map(),usedSetups=new Set(),hz=raw.metadata?.clock_freq_hz;
  if(!detailed.length && Number.isFinite(hz) && hz>0){
    for(const e of trace.traceEvents){const m=/CoreId:\s*(\d+)/.exec(e.args?.['event-hint']||'');if(e.name!=='setup'||!m||e.args?.taskId==null||!Number.isFinite(e.ts)||!Number.isFinite(e.dur)||!Number.isFinite(e.args.local_setup_us))continue;const key=e.args.taskId+'|'+m[1];if(!setups.has(key))setups.set(key,[]);setups.get(key).push(e);}
    const records=new Map();for(const row of raw.aicore_tasks){const key=row[1]+'|'+row[0];if(!records.has(key))records.set(key,[]);records.get(key).push(row);}
    for(const e of chosen){const m=hint.exec(e.args['event-hint']),key=e.args.taskId+'|'+m[3],matches=(setups.get(key)||[]).filter(x=>Math.abs(x.ts+x.dur-e.ts)<0.001&&Math.abs(x.dur-x.args.local_setup_us)<0.001),rows=records.get(key)||[];if(matches.length!==1||usedSetups.has(matches[0]))continue;const setup=matches[0],valid=rows.some(x=>Number.isFinite(x[3])&&Number.isFinite(x[4])&&Number.isFinite(x[5])&&Math.abs((x[4]-x[3])*1e6/hz-e.dur)<0.001&&Math.abs(x[5]*1e6/hz-setup.dur)<0.001);if(valid){paired.set(e,setup);usedSetups.add(setup);}}
  }
  const t0=Math.min(...chosen.map(e=>paired.get(e)?.ts??e.ts)),end=Math.max(...chosen.map(e=>e.ts+e.dur)),group=new Map();
  for(const e of chosen){const m=hint.exec(e.args['event-hint']),core=Number(m[3]),type=types[core]?.toUpperCase(),leaf=map.callable_id_to_name?.[m[2]];if(!type||!leaf||e.args.taskId==null)throw Error(`${rank}: 事件缺少核类型、函数名或 taskId`);const id=`${e.args.taskId}|${leaf}`;if(!group.has(id))group.set(id,{id,taskId:String(e.args.taskId),leaf,ct:type,items:[]});const setup=paired.get(e),item={core,startUs:(setup?.ts??e.ts)-t0,endUs:e.ts+e.dur-t0};if(detailed.length){item.slotUs=e.dur;item.computeUs=e.args['kernel-duration-us'];if(Number.isFinite(e.args.local_setup_us))item.setupUs=e.args.local_setup_us;}else{item.computeUs=e.args['duration-us'];if(setup){item.setupUs=setup.dur;item.slotUs=e.ts+e.dur-setup.ts;}}group.get(id).items.push(item);}
  const launches=[...group.values()].map(t=>{const items=t.items,start=Math.min(...items.map(x=>x.startUs)),finish=Math.max(...items.map(x=>x.endUs)),durs=items.map(x=>x.computeUs).filter(Number.isFinite),mean=durs.reduce((a,b)=>a+b,0)/(durs.length||1),b={n:items.length,items};const result={id:t.id,taskId:t.taskId,leaf:t.leaf,ct:t.ct,coresUsed:new Set(items.map(x=>x.core)).size,startUs:start,endUs:finish,spanUs:finish-start,blocks:b};if(durs.length===items.length){Object.assign(b,{p50:q(durs,.5),p90:q(durs,.9),spread:mean?Math.max(...durs)/mean:0});result.computeUs=durs.reduce((a,b)=>a+b,0);}if(items.every(x=>Number.isFinite(x.setupUs)))result.setupUs=items.reduce((s,x)=>s+x.setupUs,0);return result;});
  const critical=calculateCriticalPath(launches,depFile,map.callable_id_to_name);
  return {rank,criticalPath:critical.path??[],criticalPathInfo:{method:'observed-envelope-v1',...critical,path:undefined},source,windowUs:end-t0,cores:types.map((ct,id)=>({id,ct:ct.toUpperCase()})),launches,...(Array.isArray(depFile?.edges)?{edges:depFile.edges.map(e=>({pred:String(e.pred),succ:String(e.succ),flags:e.flags||[]}))}:{})};
}
