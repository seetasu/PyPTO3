"""Regression gates for discovery and the recorded seven-callable audit."""
import importlib.util
import json
from pathlib import Path
import tempfile

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('probe',HERE/'probe-callable.py');probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
with tempfile.TemporaryDirectory(prefix='callable-probe-') as directory:
    run=Path(directory);(run/'passes_dump').mkdir();(run/'next_levels/decode_csa_test/ptoas').mkdir(parents=True)
    (run/'next_levels/decode_csa_test/ptoas/example.pto').write_text('  pto.comm.twait %x\n')
    (run/'passes_dump/00_frontend.py').write_text('for i in pl.spmd(8, name_hint="example_spmd"):\n    pl.system.fence()\n')
    (run/'passes_dump/01_after_Outline.py').write_text('@pl.function(type=pl.FunctionType.AIV)\ndef example():\n    pl.system.fence()\n')
    result=probe.inspect(run,'example',run/'output')
    assert result['snapshots'][0]['sections'][0]['kind']=='For'
    assert result['snapshots'][0]['sections'][0]['operations']['pl.system.fence']==1
    assert result['snapshots'][1]['sections'][0]['kind']=='FunctionDef'
    assert result['isa']['communicationCount']==1
report=json.loads((HERE.parent/'docs/validation/cross-callable-audit.json').read_text())
results={r['callable']:r for r in report['results']}
assert len(results)==7
for name,r in results.items():
    assert len(r['stages'])==52 and not r['missingStages'],name
    assert r['pto']['exists'],name
assert results['idx_qr_proj_matmul']['stages'][0]['names']
assert results['comb_sinkhorn']['stages'][0]['names']
assert results['idx_qr_proj_matmul']['stages'][35]['spaces']=={'pl.Mem.Acc':1,'pl.Mem.Mat':4,'pl.Mem.Left':4,'pl.Mem.Right':2}
assert results['cp_token_allgather_payload_wait']['stages'][35]['allocationCount']==0
assert results['cp_token_allgather_payload_wait']['pto']['operations']['pto.comm.twait']==1
assert results['csa_merge_pack_publish']['stages'][49]['fences']==0
assert results['csa_merge_pack_publish']['stages'][50]['fences']==2
assert len(results['csa_merge_pack_publish']['stages'][45]['commContexts'])==4 # core + wrapper
assert results['comb_sinkhorn']['stages'][33]['allocationCount']==132
assert results['comb_sinkhorn']['stages'][35]['allocationCount']==18
print('PASS: For scope discovery, outline transition, full opcode retention; 7 × 52 audit states and structural diversity gates')
