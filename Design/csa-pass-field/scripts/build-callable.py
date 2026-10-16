"""Build the merge callable scene from its inspected snapshots and explicit region config."""
import ast
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import textwrap

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
RUN = ROOT / 'Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617'
PROBE = json.loads((HERE / 'data/callable-probe/analysis.json').read_text())
BASE = json.loads((HERE / 'data/csa-qk-pv/passes.json').read_text())
OUT = HERE / 'data/callable'
OUT.mkdir(exist_ok=True)
REGIONS = [('merge','分块合并'),('normalize','归一化'),('rope','RoPE'),('pack','Pack'),('publish','TPUT'),('notify','Notify')]


def dotted(n):
    return n.id if isinstance(n,ast.Name) else dotted(n.value)+'.'+n.attr if isinstance(n,ast.Attribute) else ''


def region(name, expression=''):
    if 'notify' in expression: return 'notify'
    if name.startswith('tput_stage'): return 'publish'
    if name.startswith('gather_'): return 'rope'
    if re.search(r'pld\.(?:tile|tensor)\.put',expression): return 'publish'
    if re.search(r'm_rope|m_cos|m_sin|m_swapped|m_rot|n_rope',name): return 'rope'
    if re.search(r'n_full_bf16|attention_grouped|source_row|source_col|n_head',name): return 'pack'
    if re.search(r'n_sink|n_denom|n_full|n_bf16',name): return 'normalize'
    if name.startswith('m_'): return 'merge'
    return None


def node(id,label,kind='op',**extra):
    return dict(id=id,label=label,semanticKind={'op':'compute_region','handoff':'communication','tensor':'tensor_value','kernel':'kernel','task':'task'}.get(kind,kind),renderKind=kind,attributes={},sourceRefs=[],**extra)


snapshots=[]
for source in PROBE['snapshots']:
    index=source['index']
    sections=source['sections']
    text=(RUN/source['file']).read_text()
    tree=ast.parse(text)
    fn=next((n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='csa_merge_pack_publish'),None)
    if fn is None:
        fn=next(n for n in ast.walk(tree) if isinstance(n,ast.With) and any(isinstance(i.context_expr,ast.Call) and any(k.arg=='name_hint' and isinstance(k.value,ast.Constant) and k.value.value=='csa_merge_pack_publish_spmd' for k in i.context_expr.keywords) for i in n.items))
    grouped=defaultdict(list)
    value_regions={}
    statements=[]
    for statement in ast.walk(fn):
        name=ast.unparse(statement.target) if isinstance(statement,ast.AnnAssign) else ''
        expr=statement.value if isinstance(statement,(ast.AnnAssign,ast.Expr)) else None
        if expr is None: continue
        expression=ast.unparse(expr)
        group=region(name,expression)
        if group and name: value_regions[name]=group
        calls=[dotted(c.func) for c in ast.walk(expr) if isinstance(c,ast.Call)]
        operations=[c for c in calls if c.startswith(('pl.tensor.','pl.tile.','pld.tile.','pld.tensor.','pld.system.','pl.system.')) and c not in ['pl.tile.alloc','pl.tile.get_block_idx','pl.tile.reserve_buffer']]
        statements.append(dict(value=name,expression=expression,operations=operations,line=statement.lineno,annotation=ast.unparse(statement.annotation) if isinstance(statement,ast.AnnAssign) else None,reads=[n.id for n in ast.walk(expr) if isinstance(n,ast.Name)],region=group))
    defined={s['value'] for s in statements if s['value']}
    for iteration in range(30):
        changed=False
        consumer_groups=defaultdict(set)
        for statement in statements:
            group=statement['region'] or value_regions.get(statement['value'])
            read_groups={value_regions.get(name) or region(name) for name in statement['reads']} - {None}
            if not group and len(read_groups)==1: group=next(iter(read_groups))
            if group:
                if statement['value'] and statement['value'] not in value_regions:
                    value_regions[statement['value']]=group;changed=True
                statement['region']=group
                for name in statement['reads']:
                    if name in defined:consumer_groups[name].add(group)
        for name,groups in consumer_groups.items():
            if name not in value_regions and len(groups)==1:value_regions[name]=next(iter(groups));changed=True
        if not changed:break
    for statement in statements:
        group=statement['region'] or value_regions.get(statement['value'])
        if 'pl.system.fence' in statement['operations']:
            preceding=[item for item in statements if item['line']<statement['line'] and any(op in ('pld.tile.put','pld.tensor.put') for op in item['operations'])]
            if preceding:group='publish'
        if group and statement['operations']:grouped[group].append({k:v for k,v in statement.items() if k != 'region'})
    file='Data/DeepseekV4/'+RUN.name+'/'+source['file']
    refs=[dict(file=file,line=sections[0]['line'])]
    nodes=[node('kernel.main','csa_merge_pack_publish' if index>=9 else 'csa_merge_pack_publish_spmd', 'kernel',role='合并、归一化、RoPE、打包与发布')]
    nodes[0]['attributes']={'target':'AIV' if index>=23 else 'InCore' if index>=9 else 'SPMD','objectKind':'Kernel' if index>=23 else 'Callable' if index>=9 else 'Scope','detail':'AIV · merge → pack → publish' if index>=23 else 'scope → callable'}
    nodes[0]['sourceRefs']=refs
    nodes[0]['sourceFacts']={'loops':sections[0]['loops'],'conditions':sections[0]['conditions']}
    nodes.append(node('task.main','csa_merge_pack_publish_spmd','task',role='deps=[qk_tid, attn_rope_tid]；48 workers'))
    nodes[-1]['attributes']={'target':'AIV','workers':48,'dependencies':'qk_tid, attn_rope_tid','form':'spmd_submit' if index>=10 else 'SPMD scope'}
    edges=[]
    for key,label in REGIONS:
        id=('handoff.' if key in ['publish','notify'] else 'op.')+key
        statements=grouped[key]
        n=node(id,label,'handoff' if key in ['publish','notify'] else 'op',role={'merge':'重标定 α/β，合并 mᵢ、ℓᵢ、Oᵢ','normalize':'加入 attention sink，除以归一化分母','rope':'尾部 64 维 gather、旋转与 concat','pack':'按 head/group 地址将结果写入 attention_grouped','publish':'向 attention_window 的目标 rank 发布数据','notify':'条件满足后更新 attention_signal'}[key])
        n['attributes']={'domain':'Tile' if index>=11 else 'Tensor','operations':sum(len(s['operations']) for s in statements),'statements':len(statements)}
        n['sourceRefs']=[dict(file=file,line=s['line']) for s in statements[:4]] or refs
        n['regionOperations']=statements
        nodes.append(n)
    def edge(a,b,kind='data'):
        edges.append(dict(id=a+'->'+b,source=a,target=b,semanticKind=kind))
    ids=[n['id'] for n in nodes[2:]]
    for a,b in zip(ids,ids[1:]):edge(a,b,'control' if b=='handoff.notify' else 'data')
    edge('task.main','kernel.main','lowering')
    inputs=[('mi','mᵢ',['t_blk',1],'FP32','op.merge'),('li','ℓᵢ',['t_blk',1],'FP32','op.merge'),('oi','Oᵢ',['t_blk',512],'FP32','op.merge'),('sink','attn_sink',[64],'FP32','op.normalize'),('cos','cos / sin',[256,64],'FP32','op.rope'),('swap','swap_idx',[16,64],'INT32','op.rope')]
    for key,label,shape,dtype,target in inputs:
        n=node('value.'+key,label,'tensor',role='callable 输入');n['attributes']={'shape':shape,'dtype':dtype};nodes.append(n);edge(n['id'],target)
    for key,label,target,shape,dtype in [('grouped','attention_grouped','op.pack',[2048,4096],'BF16'),('window','attention_window','handoff.publish',[2048,4096],'BF16'),('signal','attention_signal','handoff.notify',[2,1],'INT32')]:
        n=node('value.'+key,label,'tensor',role='callable 输出 / InOut');n['attributes']={'shape':shape,'dtype':dtype};nodes.append(n);edge(target,n['id'])
    if isinstance(fn,ast.FunctionDef):
        for argument in fn.args.args:
            if argument.annotation is None or dotted(argument.annotation)!='pld.CommCtx':continue
            name=argument.arg
            target='handoff.publish' if name=='attention_window_ctx' else 'handoff.notify' if name=='attention_signal_ctx' else None
            if target is None:continue
            context=node('context.'+name,name,'scope',role='callable 通信上下文参数')
            context['attributes']={'level':'CommCtx','type':'pld.CommCtx'}
            context['sourceRefs']=[dict(file=file,line=argument.lineno)]
            nodes.append(context);edge(context['id'],target,'control')
    rows=[]
    for alloc in sections[0]['allocations']:
        name=alloc['name'];values=[]
        for binding in sections[0]['bindings']:
            if binding['buffer']!=name:continue
            owner=value_regions.get(binding['value']) or region(binding['value'],binding['expression'])
            values.append(dict(name=binding['value'],offset=binding['offset'],size=binding['bytes'],expression=binding['expression'],line=binding['line'],owner=(('handoff.' if owner in ['publish','notify'] else 'op.')+owner) if owner else None))
        offsets=[v['offset'] for v in values if v['offset'] is not None]
        rows.append(dict(id='csa_merge_pack_publish:'+name,base=name,kernel='csa_merge_pack_publish',space='Vec',size=alloc['bytes'],offset=min(offsets) if index>=36 and offsets else None,line=alloc['line'],values=values,owners=sorted(set(v['owner'] for v in values if v['owner']))))
    snapshots.append(dict(index=index,nodes=nodes,edges=edges,rows=rows,file=file,source=source))

# Explicit source-location regions from this callable's original decode_csa.py.
# Ambiguous/missing locations remain owned by the callable, never guessed by opcode.
def isa_owner(line):
    if line is None:return 'kernel.main'
    if 416<=line<=429:return 'op.merge'
    if 432<=line<=436:return 'op.normalize'
    if 438<=line<=444:return 'op.rope'
    if line==450:return 'op.pack'
    if line==458:return 'handoff.publish'
    if line==471:return 'handoff.notify'
    return 'kernel.main'

instructions=[]
by_op=defaultdict(list)
for occurrence in PROBE['isa']['occurrences']:
    loc=re.search(r'loc\("([^"]+)":(\d+):(\d+)\)',occurrence['source'])
    by_op[occurrence['op']].append(dict(line=occurrence['line'],text=occurrence['source'],kernel='csa_merge_pack_publish',owner=isa_owner(int(loc[2]) if loc else None),sourceLine=int(loc[2]) if loc else None))
for i,(op,occurrences) in enumerate(by_op.items()):
    instructions.append(dict(id='isa.'+str(i),mlirName=op,apiName=op.split('.')[-1].upper(),occurrenceCount=len(occurrences),occurrences=occurrences))
isa=dict(source={'pto':'Data/DeepseekV4/'+RUN.name+'/'+PROBE['isa']['file']},instructions=instructions)
(OUT/'isa.json').write_text(json.dumps(isa,ensure_ascii=False,indent=2)+'\n')

# Keep the original control shell contract; replace every scenario observation.
passes=[]
for i,snapshot in enumerate(snapshots[1:],1):
    source=snapshot['source'];previous=snapshots[i-1]
    metrics=lambda s:dict(callables=sum(x['kind']=='FunctionDef' for x in s['source']['sections']),statements=sum(n.get('attributes',{}).get('statements',0) for n in s['nodes']),loops=len(s['source']['sections'][0]['loops']),allocations=len(s['rows']),tasks=sum(x['name'].endswith('_spmd') for x in s['source']['sections']))
    local=source['familyChanged'];global_change=source['programChanged']
    summary='目标 callable/scope 的源码发生变化。' if local else '完整程序发生变化；当前 callable 保持不变。' if global_change else '完整程序和当前 callable 均保持不变。'
    if i==23:summary='当前 callable 转为单 AIV；未生成 AIC 或跨核 handoff。'
    if i in [33,34,35,36]:summary=f'Vec：{len(snapshot["rows"])} 个显式分配，共 {source["allocationBytes"]:,} B。'+('地址已实体化。' if i==36 else '')
    changes=[]
    before_by_id={n['id']:n for n in previous['nodes']}
    for after in snapshot['nodes']:
        before=before_by_id.get(after['id'])
        if before is None:changes.append(dict(op='addNode',id=after['id'],after=after))
        elif before!=after:changes.append(dict(op='updateNode',id=after['id'],after=after))
    base=BASE['passes'][i-1]
    passes.append(dict(index=i,name=source['name'],phase=base['phase'],lens=base['lens'],noOp=not global_change,familySourceChanged=local,visualImpact='dag-delta' if local else 'program-source-only' if global_change else 'none',delta=changes,affectedObjectIds=[c['id'] for c in changes] if local else [],summary=summary,contract={'intent':summary,'precondition':'前一阶段的目标 scope/callable IR','produces':'当前阶段静态结构、类型和内存信息','invalidates':'按所选源码的实际变化核对','validation':'完整程序与目标片段分开比较，包含装饰器和签名'},metrics={'before':metrics(previous),'after':metrics(snapshot)},sourceRefs=[{'file':snapshot['file'],'line':source['sections'][0]['line']}]))
graph=dict(nodes=snapshots[0]['nodes'],edges=snapshots[0]['edges'],initialState={'nodeIds':[n['id'] for n in snapshots[0]['nodes']],'edgeIds':[e['id'] for e in snapshots[0]['edges']]},projections={'isa':{'availableAfterPass':51}})
(OUT/'graph.json').write_text(json.dumps(graph,ensure_ascii=False,indent=2)+'\n')
(OUT/'passes.json').write_text(json.dumps({'phases':BASE['phases'],'passes':passes},ensure_ascii=False,indent=2)+'\n')
for s in snapshots:s.pop('source')
(OUT/'snapshots.json').write_text(json.dumps(snapshots,ensure_ascii=False,indent=2)+'\n')
manifest=json.loads((HERE/'data/csa-qk-pv/manifest.json').read_text());manifest['deployment']['ranks']=[]
manifest['buildIdentity']={'runId':RUN.name,'generatorSha256':__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest(),'inputHashes':{'selectedSnapshots':__import__('hashlib').sha256((HERE/'data/callable-probe/analysis.json').read_bytes()).hexdigest()}}
manifest.update({'title':'csa_merge_pack_publish','scenarioId':'l3_decode_csa.csa_merge_pack_publish','passCount':51});(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
(OUT/'views.json').write_text(json.dumps({'views':{'front':{}}})+'\n')
print('Built',len(snapshots),'snapshots;',len(snapshots[36]['rows']),'UB allocations;',sum(i['occurrenceCount'] for i in instructions),'ISA occurrences')
