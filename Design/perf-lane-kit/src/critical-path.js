// Same observed-envelope heuristic as the original Demo, not a proof of causality.
export function calculateCriticalPath(launches, deps, nameMap = {}) {
  if (!Array.isArray(deps?.tasks) || !deps.tasks.length || !Array.isArray(deps.edges) || !deps.edges.length) return {path:null, reason:'缺少任务依赖图'};
  const tasks=new Map();for(const task of deps.tasks){if(task.task_id==null)return {path:null,reason:'依赖图存在缺少 task_id 的任务'};const id=String(task.task_id);if(tasks.has(id))return {path:null,reason:'依赖图存在重复 task_id'};tasks.set(id,task);}
  const preds=new Map([...tasks.keys()].map(id=>[id,[]]));let missingPred=0,missingSucc=0;
  for(const e of deps.edges){if(e.pred==null||e.succ==null)return {path:null,reason:'依赖边缺少端点'};const a=String(e.pred),b=String(e.succ);if(!preds.has(b)){missingSucc++;continue;}if(!preds.has(a)){missingPred++;continue;}preds.get(b).push(a);}
  const rows=new Map(),intervals=new Map();for(const l of launches){const id=String(l.taskId);if(!tasks.has(id))continue;if(!rows.has(id))rows.set(id,[]);rows.get(id).push(l);const v=intervals.get(id)||{startUs:Infinity,endUs:-Infinity};v.startUs=Math.min(v.startUs,l.startUs);v.endUs=Math.max(v.endUs,l.endUs);intervals.set(id,v);}
  if(!intervals.size)return {path:null,reason:'依赖图中的任务没有可匹配的核上时间'};
  const span=x=>x?Math.max(0,x.endUs-x.startUs):0;
  const union=(a,b)=>!a?(b?{...b}:null):!b?{...a}:{startUs:Math.min(a.startUs,b.startUs),endUs:Math.max(a.endUs,b.endUs)};
  const memo=new Map(),parent=new Map(),visiting=new Set();
  function visit(id){if(memo.has(id))return memo.get(id);if(visiting.has(id))throw Error('依赖图存在环');visiting.add(id);const own=intervals.get(id)||null;let best=own?{...own}:null,prev=null;for(const p of preds.get(id)){const prior=visit(p),candidate=union(prior,own);if(span(candidate)>span(best)||(candidate&&prior&&span(candidate)===span(best)&&prev===null)){best=candidate;prev=p;}}visiting.delete(id);memo.set(id,best);parent.set(id,prev);return best;}
  try{for(const id of tasks.keys())visit(id);}catch(e){return {path:null,reason:e.message,missingPred,missingSucc};}
  let last=null;for(const id of tasks.keys())if(last===null||span(memo.get(id))>span(memo.get(last)))last=id;
  if(!memo.get(last))return {path:null,reason:'没有可计算的时间包络',missingPred,missingSucc};
  const ids=[];for(let id=last;id;id=parent.get(id))ids.unshift(id);
  let envelope=null;const round=n=>Math.round(n*10)/10;
  const path=ids.map(id=>{const task=tasks.get(id),entries=rows.get(id)||[],interval=intervals.get(id),next=union(envelope,interval),contribution=round(Math.max(0,span(next)-span(envelope)));envelope=next;const names=(task.kernel_ids||[]).filter(x=>x>=0).map(x=>nameMap[String(x)]).filter(Boolean);return {taskId:id,name:names.length?[...new Set(names)].join(' + '):'(编排)',spanUs:round(span(interval)),startUs:interval?round(interval.startUs):null,endUs:interval?round(interval.endUs):null,pathContributionUs:contribution,computeUs:entries.length&&entries.every(t=>Number.isFinite(t.computeUs))?round(entries.reduce((n,t)=>n+t.computeUs,0)):null,blockNum:task.block_num??null,onCore:entries.length>0,ids:entries.map(t=>t.id)};});
  return {path,spanUs:round(span(envelope)),missingPred,missingSucc,matchedTasks:intervals.size,totalTasks:tasks.size};
}
