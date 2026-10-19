"""Extract one callable's memory-pass evidence from a PyPTO pass-dump run."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/passes_dump"
OUT = Path(__file__).resolve().parents[1] / "data/memory-flow.json"
CALLABLE = "csa_merge_pack_publish"
PASS_FILES = [
    (33, "InitMemRef", "33_after_InitMemRef.py"),
    (34, "MaterializeSemanticAliases", "34_after_MaterializeSemanticAliases.py"),
    (35, "MemoryReuse", "35_after_MemoryReuse.py"),
    (36, "AllocateMemoryAddr", "36_after_AllocateMemoryAddr.py"),
]


def get_int(node: ast.AST | None) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "const" and node.args:
        return get_int(node.args[0])
    return None


def callable_node(tree: ast.Module) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == CALLABLE
    )


def extract_stage(number: int, pass_name: str, filename: str) -> dict:
    path = RUN / filename
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    function = callable_node(ast.parse(source))
    function_lines = lines[function.lineno - 1 : function.end_lineno]

    allocations = []
    for node in ast.walk(function):
        if not (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "attr", "") == "alloc"
        ):
            continue
        allocations.append(
            {
                "base": node.target.id,
                "space": ast.unparse(node.value.args[0]).replace("pl.Mem.", ""),
                "bytes": get_int(node.value.args[1]),
                "line": node.lineno,
            }
        )
    allocations.sort(key=lambda item: item["line"])

    tiles = []
    for node in ast.walk(function):
        if not (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and isinstance(node.annotation, ast.Subscript)
            and ast.unparse(node.annotation.value) == "pl.Tile"
            and isinstance(node.annotation.slice, ast.Tuple)
            and len(node.annotation.slice.elts) >= 4
        ):
            continue
        shape_node, dtype_node, memref, space_node = node.annotation.slice.elts[:4]
        if not isinstance(memref, ast.Call) or getattr(memref.func, "attr", "") != "MemRef":
            continue
        base = ast.unparse(memref.args[0])
        offset_node = memref.args[1]
        if isinstance(offset_node, ast.Call) and getattr(offset_node.func, "attr", "") == "const":
            offset = get_int(offset_node)
        else:
            offset = get_int(offset_node)
        size = get_int(memref.args[2])
        uses = [
            ref.lineno
            for ref in ast.walk(function)
            if isinstance(ref, ast.Name)
            and ref.id == node.target.id
            and isinstance(ref.ctx, ast.Load)
            and ref.lineno >= node.lineno
        ]
        value_call = node.value
        op = ast.unparse(value_call.func) if isinstance(value_call, ast.Call) else "alias"
        tiles.append(
            {
                "name": node.target.id,
                "base": base,
                "space": ast.unparse(space_node).replace("pl.Mem.", ""),
                "shape": ast.unparse(shape_node),
                "dtype": ast.unparse(dtype_node).replace("pl.", ""),
                "bytes": size,
                "offset": offset,
                "op": op,
                "line": node.lineno,
                "lastUseLine": max(uses, default=node.lineno),
                "code": lines[node.lineno - 1].strip(),
            }
        )
    tiles.sort(key=lambda item: item["line"])

    lines_of_interest = set()
    for line_no, line in enumerate(function_lines, start=function.lineno):
        if "pl.range(1, 5, 2," in line or "pl.pipeline(" in line:
            lines_of_interest.update(range(max(function.lineno, line_no - 1), min(function.end_lineno, line_no + 2) + 1))

    return {
        "number": number,
        "pass": pass_name,
        "file": filename,
        "functionStart": function.lineno,
        "functionEnd": function.end_lineno,
        "allocations": allocations,
        "allocationCount": len(allocations),
        "allocationBytes": sum(item["bytes"] or 0 for item in allocations),
        "tiles": tiles,
        "functionCode": function_lines,
        "pipelineLines": [
            {"line": line_no, "code": lines[line_no - 1].strip()}
            for line_no in sorted(lines_of_interest)
        ],
    }


stages = [extract_stage(*entry) for entry in PASS_FILES]
init_tiles = {item["name"]: item for item in stages[0]["tiles"]}
alias_tiles = {item["name"]: item for item in stages[1]["tiles"]}
semantic_changes = [
    {"name": name, "before": init_tiles[name], "after": alias_tiles[name]}
    for name in init_tiles.keys() & alias_tiles.keys()
    if any(
        init_tiles[name][field] != alias_tiles[name][field]
        for field in ("base", "offset", "bytes")
    )
]
before = {item["name"]: item for item in stages[1]["tiles"]}
after_reuse = {item["name"]: item for item in stages[2]["tiles"]}
after_addr = {item["name"]: item for item in stages[3]["tiles"]}
reuse_pair_names = ["m_oi_inline1220__tile", "n_full_inline1211__tile"]
reuse_pair = []
for name in reuse_pair_names:
    reuse_pair.append(
        {
            "name": name,
            "stageValues": [
                next((tile for tile in stage["tiles"] if tile["name"] == name), None)
                for stage in stages
            ],
            "before": before.get(name),
            "afterReuse": after_reuse.get(name),
            "afterAddress": after_addr.get(name),
        }
    )

for allocation in stages[3]["allocations"]:
    matching = [tile for tile in stages[3]["tiles"] if tile["base"] == allocation["base"]]
    starts = [tile["offset"] for tile in matching if tile["offset"] is not None]
    allocation["offset"] = min(starts) if starts else 0

data = {
    "run": "_jit_l3_decode_csa_20260903_010617",
    "callable": CALLABLE,
    "functionType": "AIV",
    "sourceDirectory": "Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/passes_dump",
    "addressSpace": "Vec",
    "reusePair": reuse_pair,
    "semanticChanges": semantic_changes,
    "stages": stages,
    "metrics": {
        "beforeReuseCount": stages[1]["allocationCount"],
        "afterReuseCount": stages[2]["allocationCount"],
        "beforeReuseBytes": stages[1]["allocationBytes"],
        "afterReuseBytes": stages[2]["allocationBytes"],
        "finalHighWater": max(
            (item["offset"] or 0) + (item["bytes"] or 0)
            for item in stages[3]["allocations"]
        ),
    },
    "notes": [
        "Pass dump 中的源码行号用于定位静态 IR，不是设备 cycle。",
        "生命周期用名称的最后一次静态读取近似；函数含循环和流水 lowering，不能代替运行时并发时序。",
        "分配字节数是该 callable 的 tile.alloc 静态总量，不代表性能提升比例。",
    ],
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {OUT.relative_to(ROOT)}")
print("allocations:", [(stage["pass"], stage["allocationCount"], stage["allocationBytes"]) for stage in stages])
print("reuse pair:", [(item["name"], item["before"]["base"], item["afterReuse"]["base"], item["afterAddress"]["offset"]) for item in reuse_pair])
