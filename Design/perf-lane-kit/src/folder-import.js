import {discoverCaptures} from './capture-discovery.js';
import {buildCapture} from './dfx-adapter.js';
import {adapt} from './data.js';
export async function importCaptureFiles(files){
 const selected=[...files];if(!selected.length)throw Error('未选择文件。');
 const byPath=new Map();for(const file of selected){const path=(file.webkitRelativePath||file.name).replace(/\\/g,'/');if(byPath.has(path))throw Error(`文件路径重复：${path}`);byPath.set(path,file);}
 const specs=discoverCaptures([...byPath.keys()]);const ranks=[];
 async function read(path){try{return JSON.parse(await byPath.get(path).text());}catch(e){throw Error(`${path} 不是有效 JSON：${e.message}`);}}
 for(const spec of specs){const [trace,map,raw,depFile]=await Promise.all([read(spec.trace),read(spec.map),read(spec.record),spec.deps?read(spec.deps):null]);ranks.push(buildCapture({rank:spec.rank,source:spec.trace,trace,map,raw,depFile}));}
 const root=selected[0].webkitRelativePath?.split('/')[0]||'本地采集';return adapt({schemaVersion:'perf-lane.v1',run:root,ranks,caveats:['各采集以最早 AICore 事件为原点；路径为依赖和实测时间的启发式分析线索。']});
}
