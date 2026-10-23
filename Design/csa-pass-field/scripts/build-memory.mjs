import fs from 'node:fs';
import path from 'node:path';
const root=path.resolve(import.meta.dirname,'../../..');
const dir='Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/passes_dump';
const snapshots={};
for(const index of [33,34,35,36]) {
 const file=fs.readdirSync(path.join(root,dir)).find(f=>f.startsWith(`${index}_`));
 const lines=fs.readFileSync(path.join(root,dir,file),'utf8').split('\n');
 let kernel=null; const rows=[]; const reserves=[];
 lines.forEach((text,i)=>{
  const fn=text.match(/^    def (\w+)\(/); if(fn) kernel=['qk_pv_aic','qk_pv_aiv'].includes(fn[1])?fn[1]:null;
  if(!kernel)return;
  const alloc=text.match(/^\s+(mem_\w+):.*pl\.tile\.alloc\(pl\.Mem\.(\w+),\s*(\d+)\)/);
  if(alloc)rows.push({id:`${kernel}:${alloc[1]}`,base:alloc[1],kernel,space:alloc[2],size:+alloc[3],line:i+1,values:[]});
  const reserved=text.match(/reserve_buffer\(name='([^']+)', size=(\d+), base=([^\)]+)/);
  if(reserved)reserves.push({kernel,name:reserved[1],size:+reserved[2],offset:/^\d+$/.test(reserved[3])?+reserved[3]:null});
  const val=text.match(/^\s+(\w+): pl\.Tile\[.*?pl\.MemRef\((mem_\w+), (?:pl\.const\()?(-?\d+)(?:, pl\.INT64\))?, (\d+)\).*?= (.*)/);
  if(val){const row=rows.find(r=>r.kernel===kernel&&r.base===val[2]);if(row){row.values.push({name:val[1],offset:+val[3],size:+val[4],expression:val[5],line:i+1});if(index===36)row.offset=Math.min(row.offset??Infinity,+val[3]);}}
 });
 snapshots[index]={file:`${dir}/${file}`,rows,reserves};
}
fs.writeFileSync(new URL('../data/memory.json',import.meta.url),JSON.stringify(snapshots,null,2));
