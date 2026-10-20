"""Resolve syntax-level def/use through scalar aliases and loop-carried arguments.
No execution or runtime completion ordering is inferred.
"""
import ast
from collections import defaultdict


def names(node):
    return [n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)]


def dependency_facts(fn, statements):
    definitions = defaultdict(list)
    for statement in statements:
        if statement['value']:
            definitions[statement['value']].append(statement)
    for values in definitions.values():
        values.sort(key=lambda s: s['line'])
    carried = {}
    for loop in ast.walk(fn):
        if not isinstance(loop, ast.For) or not isinstance(loop.iter, ast.Call):
            continue
        initial = next((k.value for k in loop.iter.keywords if k.arg == 'init_values'), None)
        if initial is None or not isinstance(loop.target, ast.Tuple) or len(loop.target.elts) < 2:
            continue
        targets = [n.id for n in ast.walk(loop.target.elts[1]) if isinstance(n, ast.Name)]
        values = initial.elts if isinstance(initial, (ast.Tuple, ast.List)) else [initial]
        feedback = []
        for child in loop.body:
            expr = child.value if isinstance(child, (ast.AnnAssign, ast.Expr)) else None
            if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and expr.func.attr == 'yield_':
                feedback = expr.args
        for i, (target, value) in enumerate(zip(targets, values)):
            carried[target] = [(name, loop.lineno, 'loop-init') for name in names(value)]
            if i < len(feedback):
                carried[target] += [(name, loop.end_lineno + 1, 'loop-carried') for name in names(feedback[i])]

    def resolve(name, line, seen, kind='data'):
        key = (name, line)
        if key in seen:
            return set()
        seen = seen | {key}
        if name in carried:
            return set().union(*(resolve(n, at, seen, relation) for n, at, relation in carried[name]))
        previous = [s for s in definitions.get(name, []) if s['line'] < line]
        if not previous:
            # Parameters and outer captured tensor names are retained as source names.
            return {(None, name, kind)}
        definition = previous[-1]
        if definition['operations']:
            return {(definition['line'], name, kind)}
        return set().union(*(resolve(n, definition['line'], seen, kind) for n in definition['reads']))

    for statement in statements:
        if not statement['operations']:
            continue
        dependencies = set().union(*(resolve(name, statement['line'], set()) for name in statement['reads']))
        statement['dependencies'] = [dict(producerLine=line, value=name, kind=kind) for line, name, kind in sorted(dependencies, key=str)]
