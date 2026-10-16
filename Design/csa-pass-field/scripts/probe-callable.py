"""Inspect callable/scope IR snapshots as syntax only; never execute uploaded Python."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def dotted(node):
    if isinstance(node, ast.Name): return node.id
    if isinstance(node, ast.Attribute): return dotted(node.value) + '.' + node.attr
    return ''


def numeric(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)): return node.value
    if isinstance(node, ast.Call) and dotted(node.func) == 'pl.const' and node.args: return numeric(node.args[0])
    return None


def inspect(run, callable_name, output):
    snapshots = []
    previous = None
    previous_program = None
    output.mkdir(parents=True, exist_ok=True)
    (output / 'snapshots').mkdir(exist_ok=True)
    for file in sorted((run / 'passes_dump').glob('[0-9][0-9]_*.py')):
        source = file.read_text()
        tree = ast.parse(source, filename=str(file))
        lines = source.splitlines()
        family = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and (n.name == callable_name or n.name.startswith(callable_name + '_'))]
        scopes = []
        if not family:
            for n in ast.walk(tree):
                if not isinstance(n, (ast.With, ast.For)): continue
                for call in ([item.context_expr for item in n.items] if isinstance(n, ast.With) else [n.iter]):
                    if isinstance(call, ast.Call) and any(k.arg == 'name_hint' and isinstance(k.value, ast.Constant) and (k.value.value == callable_name or k.value.value == callable_name + '_spmd') for k in call.keywords):
                        scopes.append(n)
        selected = family or scopes
        chunks = ['\n'.join(lines[min([n.lineno] + [d.lineno for d in getattr(n, 'decorator_list', [])]) - 1:n.end_lineno]) for n in selected]
        text = '\n\n'.join(chunks)
        sections = []
        for n in selected:
            calls = [c for c in ast.walk(n) if isinstance(c, ast.Call)]
            allocs = []
            bindings = []
            for statement in ast.walk(n):
                if not isinstance(statement, ast.AnnAssign): continue
                call = statement.value
                if isinstance(call, ast.Call) and dotted(call.func) == 'pl.tile.alloc':
                    allocs.append({'name': ast.unparse(statement.target), 'space': ast.unparse(call.args[0]), 'bytes': numeric(call.args[1]), 'line': statement.lineno})
                for ref in ast.walk(statement.annotation):
                    if isinstance(ref, ast.Call) and dotted(ref.func) == 'pl.MemRef':
                        bindings.append({'value': ast.unparse(statement.target), 'buffer': ast.unparse(ref.args[0]), 'offset': numeric(ref.args[1]), 'bytes': numeric(ref.args[2]), 'line': statement.lineno, 'expression': ast.unparse(statement.value)})
            reserves = [{'name': next((ast.unparse(k.value) for k in c.keywords if k.arg == 'name'), None), 'bytes': next((numeric(k.value) for k in c.keywords if k.arg == 'size'), None), 'line': c.lineno} for c in calls if dotted(c.func).endswith('.reserve_buffer')]
            sections.append({'name': getattr(n, 'name', 'scope:' + callable_name + '_spmd'), 'kind': type(n).__name__, 'line': n.lineno, 'endLine': n.end_lineno, 'parameters': [{'name': a.arg, 'annotation': ast.unparse(a.annotation) if a.annotation else None} for a in n.args.args] if isinstance(n, ast.FunctionDef) else [], 'decorators': [ast.unparse(d) for d in n.decorator_list] if isinstance(n, ast.FunctionDef) else [], 'operations': dict(sorted(Counter(dotted(c.func) for c in calls if dotted(c.func)).items())), 'loops': [{'line': c.lineno, 'target': ast.unparse(c.target), 'iterator': ast.unparse(c.iter)} for c in ast.walk(n) if isinstance(c, ast.For)], 'conditions': [{'line': c.lineno, 'test': ast.unparse(c.test)} for c in ast.walk(n) if isinstance(c, ast.If)], 'allocations': allocs, 'bindings': bindings, 'reserves': reserves})
        row = {'index': int(file.name[:2]), 'name': file.stem[3:].removeprefix('after_'), 'file': str(file.relative_to(run)), 'sections': sections, 'sourceSha256': hashlib.sha256(text.encode()).hexdigest(), 'familyChanged': previous is not None and text != previous, 'programChanged': previous_program is not None and source != previous_program, 'sourceLines': sum(len(c.splitlines()) for c in chunks), 'allocationCount': sum(len(s['allocations']) for s in sections), 'allocationBytes': sum(a['bytes'] or 0 for s in sections for a in s['allocations'])}
        (output / 'snapshots' / file.name).write_text(text + '\n')
        snapshots.append(row)
        previous, previous_program = text, source
    pto = run / 'next_levels/decode_csa_test/ptoas' / (callable_name + '.pto')
    pto_text = pto.read_text()
    occurrences = [{'op': m.group(1), 'line': i, 'source': line.strip()} for i, line in enumerate(pto_text.splitlines(), 1) for m in re.finditer(r'^\s*(?:%[^=]+?=\s*)?(pto\.(?:t\w+|comm\.\w+))(?=\s|\()', line)]
    inventory = Counter(m.group(1) for line in pto_text.splitlines() for m in re.finditer(r'^\s*(?:%[^=]+?=\s*)?(pto\.[\w.]+)(?=\s|\()', line))
    result = {'schemaVersion': 'callable-probe.v1', 'callable': callable_name, 'run': run.name, 'snapshotCount': len(snapshots), 'snapshots': snapshots, 'isa': {'file': str(pto.relative_to(run)), 'sha256': hashlib.sha256(pto.read_bytes()).hexdigest(), 'operations': dict(sorted(Counter(o['op'] for o in occurrences).items())), 'occurrences': occurrences, 'irOperationCounts': dict(sorted(inventory.items())), 'tileInstructionCount': sum(1 for o in occurrences if not o['op'].startswith('pto.comm.')), 'communicationCount': sum(1 for o in occurrences if o['op'].startswith('pto.comm.'))}, 'limits': ['AST calls are static syntax occurrences, not runtime counts.', 'Scope-to-function matching uses explicit name_hint and function names.', 'Cross-pass value lineage, CFG and operation-to-ISA mapping have not been resolved.']}
    (output / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--callable', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.run, args.callable.replace('-', '_'), args.output)
    print('Snapshots:', result['snapshotCount'])
    print('Changed callable/scope passes:', [s['index'] for s in result['snapshots'] if s['familyChanged']])
    for row in result['snapshots']:
        if row['index'] in [0, 8, 9, 10, 11, 23, 33, 34, 35, 36, 51]:
            print(row['index'], [s['name'] for s in row['sections']], row['sourceLines'], row['allocationCount'], row['allocationBytes'])
    print('ISA static occurrences:', sum(result['isa']['operations'].values()), result['isa']['operations'])
