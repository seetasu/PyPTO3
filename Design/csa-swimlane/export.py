import json, hashlib, re
from pathlib import Path
root=Path(__file__).resolve().parent
source=Path("/Users/yin/PyPTO3-main/Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/dfx_outputs")
ranks=[]
for rank in range(2):
 p=next((source/f"rank{rank}/d0").glob("merged_swimlane*.json")); events=json.loads(p.read_text())["traceEvents"]
 names={e["tid"]:e["args"]["name"] for e in events if e["ph"]=="M" and e["name"]=="thread_name" and e["pid"]==4}
 lanes=sorted([k for k,v in names.items() if v.startswith(("AIC_","AIV_"))])
 tasks=[]; ids={}
 for e in events:
  if e["ph"]=="X" and e["pid"]==4 and e["tid"] in lanes:
   ids[e["id"]]=len(tasks);tasks.append(dict(id=e["id"],lane=lanes.index(e["tid"]),name=e["name"],start=e["ts"],end=e["ts"]+e["dur"],task=str(e.get("args",{}).get("taskId",""))))
 groups={};logical=set()
 for t in tasks:
  match=re.search(r'\((r\dt\d+)\)',t['name'])
  if match:groups.setdefault(match[1],[]).append(ids[t['id']])
 for e in events:
  if e.get('id') not in ids or e['ph']!='X' or e['pid']!=4:continue
  match=re.search(r'\((r\dt\d+)\)',e['name'])
  if match:
   for target in re.findall(r'r\dt\d+',e.get('args',{}).get('fanout-hint','')):
    if target in groups:logical.add((match[1],target))
 starts={};edges=[];unresolved=0
 for e in events:
  if e["pid"]!=4:continue
  if e["ph"]=="s":starts[e["id"]]=e
  if e["ph"]=="f" and e["id"] in starts:
   s=starts[e["id"]];a=ids.get(s.get("bind_id"));b=ids.get(e.get("bind_id"))
   if a is not None and b is not None:edges.append([a,b])
   else:unresolved+=1
 edges=list(map(list,sorted(set(map(tuple,edges)))))
 ranks.append(dict(rank=rank,lanes=[names[k] for k in lanes],tasks=tasks,edges=edges,groups=groups,logical=sorted(logical),end=max(t["end"] for t in tasks),source=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),unresolved=unresolved))
 print(rank,len(lanes),len(tasks),len(edges),unresolved)
(root/"data.js").write_text("window.CSA_DATA="+json.dumps(ranks,separators=(",",":"))+";\n")
