import {readFile, readdir, writeFile, stat} from 'node:fs/promises';
import {resolve,join,basename,relative} from 'node:path';
import {fileURLToPath} from 'node:url';
import {adapt} from '../src/data.js';
import {calculateCriticalPath} from '../src/critical-path.js';
const hint=/Task:(\S+?),\s*FuncId:\s*(\d+),\s*CoreId:\s*(\d+)/;
const read=async p=>JSON.parse(await readFile(p,'utf8'));
const q=(a,f)=>{const b=a.slice().sort((x,y)=>x-y);if(!b.length)return 0;const i=(b.length-1)*f,l=Math.floor(i);return b[l]+(b[Math.ceil(i)]-b[l])*(i-l);};
export async function importRun(runPath){
 const root=resolve(runPath), dfx=basename(root)==='dfx_outputs'?root:join(root,'dfx_outputs');
 const entries=await readdir(dfx,{withFileTypes:true});const dirs=[];
 if(entries.some(e=>e.isFile()&&e.name.startsWith('merged_swimlane_'))) dirs.push({rank:'rank0',dir:dfx});
 for(const e of entries.filter(e=>e.isDirectory()&&/^rank\d+$/i.test(e.name)).sort((a,b)=>a.name.localeCompare(b.name,undefined,{numeric:true}))){const rd=join(dfx,e.name);for(const dev of (await readdir(rd,{withFileTypes:true})).filter(x=>x.isDirectory()&&/^d\d+$/i.test(x.name)).sort((a,b)=>a.name.localeCompare(b.name,undefined,{numeric:true})))dirs.push({rank:`${e.name}/${dev.name}`,dir:join(rd,dev.name)});}
 if(!dirs.length)throw Error('未找到包含 merged_swimlane 的 DFX 目录');
 const ranks=[];
 for(const {rank,dir} of dirs){
  const files=await readdir(dir), traces=files.filter(x=>/^merged_swimlane_.*\.json$/i.test(x)),maps=files.filter(x=>/^name_map.*\.json$/i.test(x)),record=files.find(x=>x==='chip_swimlane_records.json')||files.find(x=>x==='l2_swimlane_records.json');
  if(traces.length!==1||maps.length!==1||!record)throw Error(`${rank}: 需要唯一 merged_swimlane、name_map 和 chip/l2_swimlane_records；发现 ${traces.length} 个 trace、${maps.length} 个 name_map`);
  const [trace,map,raw]=await Promise.all([read(join(dir,traces[0])),read(join(dir,maps[0])),read(join(dir,record))]);
  const types=raw.metadata?.core_types;if(!Array.isArray(types)||!Array.isArray(raw.aicore_tasks)||!Array.isArray(trace.traceEvents))throw Error(`${rank}: 缺少核元数据或事件`);
  const candidates=trace.traceEvents.filter(e=>e.cat==='event'&&e.ph==='X'&&hint.test(e.args?.['event-hint']||'')&&Number.isFinite(e.ts)&&Number.isFinite(e.dur));
  const detailed=candidates.filter(e=>Number.isFinite(e.args['kernel-duration-us']));const timed=candidates.filter(e=>Number.isFinite(e.args['duration-us']));const chosen=detailed.length?detailed:timed;
  if(chosen.length!==raw.aicore_tasks.length)throw Error(`${rank}: 可用事件 ${chosen.length} 条，与核记录 ${raw.aicore_tasks.length} 条不一致，无法安全去重`);
  if(!chosen.length)throw Error(`${rank}: 没有核事件`);
  const t0=Math.min(...chosen.map(e=>e.ts)),end=Math.max(...chosen.map(e=>e.ts+e.dur)),group=new Map();
  for(const e of chosen){const m=hint.exec(e.args['event-hint']),core=Number(m[3]),type=types[core]?.toUpperCase(),leaf=map.callable_id_to_name?.[m[2]];if(!type||!leaf||e.args.taskId==null)throw Error(`${rank}: 事件缺少核类型、函数名或 taskId`);const id=`${e.args.taskId}|${leaf}`;if(!group.has(id))group.set(id,{id,taskId:String(e.args.taskId),leaf,ct:type,items:[]});const item={core,startUs:e.ts-t0,endUs:e.ts+e.dur-t0,slotUs:e.dur};if(detailed.length){item.computeUs=e.args['kernel-duration-us'];if(Number.isFinite(e.args.local_setup_us))item.setupUs=e.args.local_setup_us;}group.get(id).items.push(item);}
  const launches=[...group.values()].map(t=>{const items=t.items,start=Math.min(...items.map(x=>x.startUs)),finish=Math.max(...items.map(x=>x.endUs)),durs=items.map(x=>x.computeUs).filter(Number.isFinite),mean=durs.reduce((a,b)=>a+b,0)/(durs.length||1),b={n:items.length,items};const result={id:t.id,taskId:t.taskId,leaf:t.leaf,ct:t.ct,coresUsed:new Set(items.map(x=>x.core)).size,startUs:start,endUs:finish,spanUs:finish-start,blocks:b};if(durs.length===items.length){Object.assign(b,{p50:q(durs,.5),p90:q(durs,.9),spread:mean?Math.max(...durs)/mean:0});result.computeUs=durs.reduce((a,b)=>a+b,0);}if(items.every(x=>Number.isFinite(x.setupUs)))result.setupUs=items.reduce((s,x)=>s+x.setupUs,0);return result;});
  const depFile=files.includes('deps.json')?await read(join(dir,'deps.json')):null;
  const critical=calculateCriticalPath(launches,depFile,map.callable_id_to_name);
  ranks.push({rank,criticalPath:critical.path??[],criticalPathInfo:{method:'observed-envelope-v1',...critical,path:undefined},source:relative(root,join(dir,traces[0])),windowUs:end-t0,cores:types.map((ct,id)=>({id,ct:ct.toUpperCase()})),launches,...(Array.isArray(depFile?.edges)?{edges:depFile.edges.map(e=>({pred:String(e.pred),succ:String(e.succ),flags:e.flags||[]}))}:{})});
 }
 return adapt({schemaVersion:'perf-lane.v1',run:basename(root),sourceDirectory:root,ranks,caveats:['时间以各 rank 最早 AICore 事件为原点；关键路径是依赖图与核上时间包络的分析线索；未匹配的依赖端点不纳入路径。']});
}
if(process.argv[1] && resolve(process.argv[1])===resolve(fileURLToPath(import.meta.url))){
 const [input,output]=process.argv.slice(2);if(!input||!output){console.error('用法: node scripts/import-run.js <运行目录或 dfx_outputs> <输出 JSON 路径>');process.exitCode=1;}else try{const doc=await importRun(input);await writeFile(output,JSON.stringify(doc),{flag:'wx'});console.log(`已写入 ${output}: ${doc.ranks.map(r=>`${r.rank} ${r.launches.length} Task / ${r.launches.reduce((n,t)=>n+t.blocks.n,0)} Block`).join('；')}`);}catch(e){console.error(`导入失败：${e.message}`);process.exitCode=1;}
}



