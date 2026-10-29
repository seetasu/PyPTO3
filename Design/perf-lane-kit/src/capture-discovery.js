// Paths are normalized relative to the chosen directory; no rank hierarchy is required.
export function discoverCaptures(paths){
 const normalized=paths.map(p=>p.replace(/\\/g,'/').replace(/^\/+/,''));
 const tracePaths=normalized.filter(p=>/^merged_swimlane_.*\.json$/i.test(p.split('/').at(-1)));
 if(!tracePaths.length)throw Error('未找到 merged_swimlane 文件；请选择完整运行目录、dfx_outputs 或包含该文件的采集目录。');
 const dfxTraces=tracePaths.filter(p=>p.split('/').some(x=>x.toLowerCase()==='dfx_outputs'));
 const selected=dfxTraces.length?dfxTraces:tracePaths;
 const dirs=[...new Set(selected.map(p=>p.split('/').slice(0,-1).join('/')))].sort((a,b)=>a.localeCompare(b,undefined,{numeric:true}));
 const captures=dirs.map(dir=>{const files=normalized.filter(p=>p.split('/').slice(0,-1).join('/')===dir);const traces=files.filter(p=>/^merged_swimlane_.*\.json$/i.test(p.split('/').at(-1))),maps=files.filter(p=>/^name_map.*\.json$/i.test(p.split('/').at(-1))),records=files.filter(p=>/^(chip|l2)_swimlane_records\.json$/i.test(p.split('/').at(-1))),deps=files.find(p=>p.split('/').at(-1)==='deps.json');if(traces.length!==1||maps.length!==1||!records.length)throw Error(`${dir||'采集目录'}：需要一个 merged_swimlane、一个 name_map 和 chip 或 l2 核记录（找到 ${traces.length}、${maps.length}、${records.length}）。`);const record=records.find(p=>p.endsWith('/chip_swimlane_records.json')||p==='chip_swimlane_records.json')||records[0];const parts=dir.split('/').filter(Boolean),dfx=parts.findIndex(p=>p.toLowerCase()==='dfx_outputs'),relative=dfx>=0?parts.slice(dfx+1):parts.slice(1),label=relative.join('/')||(dirs.length===1?'单次采集':dir||'单次采集');return {rank:label,trace:traces[0],map:maps[0],record,deps,dir};});
 const counts=new Map();for(const c of captures)counts.set(c.rank,(counts.get(c.rank)||0)+1);for(const c of captures){if(counts.get(c.rank)>1)c.rank=c.dir||'单次采集';delete c.dir;}
 return captures;
}
