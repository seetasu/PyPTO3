"""Read-only cross-callable audit; parse IR syntax without executing input code."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from datetime import date


def dotted(n):
    return n.id if isinstance(n,ast.Name) else dotted(n.value)+'.'+n.attr if isinstance(n,ast.Attribute) else ''


def number(n):
    if isinstance(n,ast.Constant) and isinstance(n.value,int):return n.value
    if isinstance(n,ast.Call) and dotted(n.func)=='pl.const' and n.args:return number(n.args[0])
    return None


def audit(run,names):
    records={name:[] for name in names}
    for file in sorted((run/'passes_dump').glob('[0-9][0-9]_*.py')):
        text=file.read_text();tree=ast.parse(text,filename=str(file));all_nodes=list(ast.walk(tree))
        functions=[n for n in all_nodes if isinstance(n,ast.FunctionDef)]
        scopes=[n for n in all_nodes if isinstance(n,(ast.With,ast.For))]
        for name in names:
            family=[n for n in functions if n.name==name or n.name.startswith(name+'_')]
            if not family:
                family=[n for n in scopes if any(isinstance(call,ast.Call) and any(k.arg=='name_hint' and isinstance(k.value,ast.Constant) and k.value.value in (name,name+'_spmd') for k in call.keywords) for call in ([i.context_expr for i in n.items] if isinstance(n,ast.With) else [n.iter]))]
            sections=[]
            for n in family:
                nodes=list(ast.walk(n));ops=Counter(dotted(c.func) for c in nodes if isinstance(c,ast.Call) and dotted(c.func).startswith(('pl.','pld.')))
                alloc=[];refs=[]
                for statement in nodes:
                    if not isinstance(statement,ast.AnnAssign):continue
                    call=statement.value
                    if isinstance(call,ast.Call) and dotted(call.func)=='pl.tile.alloc':
                        alloc.append(dict(name=ast.unparse(statement.target),space=ast.unparse(call.args[0]),bytes=number(call.args[1]),line=statement.lineno))
                    for ref in ast.walk(statement.annotation):
                        if isinstance(ref,ast.Call) and dotted(ref.func)=='pl.MemRef':refs.append(dict(value=ast.unparse(statement.target),base=ast.unparse(ref.args[0]),offset=number(ref.args[1]),bytes=number(ref.args[2])))
                sections.append(dict(name=getattr(n,'name','scope:'+name),kind=type(n).__name__,line=n.lineno,endLine=n.end_lineno,decorators=[ast.unparse(d) for d in getattr(n,'decorator_list',[])],parameters=[dict(name=a.arg,type=ast.unparse(a.annotation) if a.annotation else None) for a in n.args.args] if isinstance(n,ast.FunctionDef) else [],operations=dict(sorted(ops.items())),allocations=alloc,memrefs=refs,loops=sum(isinstance(x,ast.For) for x in nodes),conditions=sum(isinstance(x,ast.If) for x in nodes)))
            fingerprint=hashlib.sha256(''.join(ast.dump(n,include_attributes=False) for n in family).encode()).hexdigest()
            records[name].append(dict(index=int(file.name[:2]),file=str(file.relative_to(run)),fileSha256=hashlib.sha256(text.encode()).hexdigest(),familyFingerprint=fingerprint,sections=sections))
    report=[]
    for name,stages in records.items():
        assert len(stages)==52,(name,len(stages))
        missing=[s['index'] for s in stages if not s['sections']]
        # Missing discovery is a reported coverage gap, never a no-op.

        changes=[b['index'] for a,b in zip(stages,stages[1:]) if a['sections'] and b['sections'] and a['familyFingerprint']!=b['familyFingerprint']]
        summaries=[]
        for stage in stages:
            sections=stage['sections'];alloc=[a for s in sections for a in s['allocations']];ops=Counter()
            for s in sections:ops.update(s['operations'])
            summaries.append(dict(index=stage['index'],names=[s['name'] for s in sections],decorators=[d for s in sections for d in s['decorators']],allocationCount=len(alloc),knownAllocationBytes=sum(a['bytes'] for a in alloc if a['bytes'] is not None),unknownAllocationSizes=sum(a['bytes'] is None for a in alloc),spaces=dict(Counter(a['space'] for a in alloc)),fences=ops['pl.system.fence'],commContexts=[p['name'] for s in sections for p in s['parameters'] if p['type']=='pld.CommCtx'],loops=sum(s['loops'] for s in sections),conditions=sum(s['conditions'] for s in sections)))
        pto=run/'next_levels/decode_csa_test/ptoas'/f'{name}.pto'
        inventory=Counter(m.group(1) for line in pto.read_text().splitlines() for m in re.finditer(r'^\s*(?:%[^=]+?=\s*)?(pto\.[\w.]+)(?=\s|\()',line)) if pto.exists() else Counter()
        report.append(dict(callable=name,missingStages=missing,firstObservedStage=next((s['index'] for s in stages if s['sections']),None),changedPasses=changes,stages=summaries,finalOperationInventory=stages[-1]['sections'],pto=dict(file=str(pto.relative_to(run)),exists=pto.exists(),operations=dict(sorted(inventory.items()))),evidence=stages))
    return dict(schemaVersion='cross-callable-audit.v1',run=str(run),date=str(date.today()),limits=['Function-prefix matching gives candidate families, not proven lineage.','Source changes are not main-graph changes and must not drive navigation.','Static occurrences and allocation sums are not runtime counts or peak memory.','No new callable was connected to the React Flow page in this audit.'],results=report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    names=['qk_pv','csa_merge_pack_publish','csa_cache_writeback','idx_qr_proj_matmul','cp_token_allgather_payload_wait','comb_sinkhorn','csa_rope_interleave']
    report=audit(args.run.resolve(),names)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for r in report['results']:
        print(r['callable'],'changed=',r['changedPasses'])
        for i in [0,9,23,33,35,36,45,50,51]:
            s=r['stages'][i];print(i,s['names'],'alloc',s['allocationCount'],s['knownAllocationBytes'],'spaces',s['spaces'],'fences',s['fences'],'ctx',len(s['commContexts']),'loops',s['loops'])
