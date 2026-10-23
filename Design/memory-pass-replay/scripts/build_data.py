"""Extract one callable's memory-pass evidence from a PyPTO pass-dump run."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/passes_dump"
OUT = Path(__file__).resolve().parents[1] / "data/memory-flow.json"
CALLABLES = [
    ("csa_merge_pack_publish", "AIV"),
    ("tp_o_a", "AIC"),
]
PASS_FILES = [
    (33, "InitMemRef", "33_after_InitMemRef.py"),
    (34, "MaterializeSemanticAliases", "34_after_MaterializeSemanticAliases.py"),
    (35, "MemoryReuse", "35_after_MemoryReuse.py"),
    (36, "AllocateMemoryAddr", "36_after_AllocateMemoryAddr.py"),
]
TILE_MILESTONES = [
    (11, "ConvertTensorToTileOps", "11_after_ConvertTensorToTileOps.py"),
    (18, "CanonicalizeTileSlice", "18_after_CanonicalizeTileSlice.py"),
    (21, "ResolveBackendOpLayouts", "21_after_ResolveBackendOpLayouts.py"),
    (30, "LowerPipelineLoops", "30_after_LowerPipelineLoops.py"),
    (33, "InitMemRef", "33_after_InitMemRef.py"),
    (34, "MaterializeSemanticAliases", "34_after_MaterializeSemanticAliases.py"),
    (35, "MemoryReuse", "35_after_MemoryReuse.py"),
    (36, "AllocateMemoryAddr", "36_after_AllocateMemoryAddr.py"),
]
TILE_GROUPS = {
    "load": "输入读取", "slice": "输入读取", "extract": "输入读取", "gather": "输入读取",
    "reshape": "布局变换", "cast": "布局变换", "concat": "布局变换", "alias": "布局变换", "yield_": "布局变换",
    "sub": "数值计算", "add": "数值计算", "exp": "数值计算", "mul": "数值计算",
    "row_expand_mul": "数值计算", "maximum": "数值计算", "col_expand_mul": "数值计算", "row_expand_div": "数值计算",
    "create": "结果构造", "assemble": "结果构造", "move": "内存复用迁移",
}


def get_int(node: ast.AST | None) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "const" and node.args:
        return get_int(node.args[0])
    return None


def callable_node(tree: ast.Module, callable_name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == callable_name
    )


def extract_stage(number: int, pass_name: str, filename: str, callable_name: str) -> dict:
    path = RUN / filename
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    function = callable_node(ast.parse(source), callable_name)
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
    tile_defs = [node for node in ast.walk(function) if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and isinstance(node.annotation, ast.Subscript) and ast.unparse(node.annotation.value) == "pl.Tile"]
    tile_names = {node.target.id for node in tile_defs}
    for node in tile_defs:
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
        op_leaf = op.rsplit(".", 1)[-1]
        inputs = sorted({ref.id for ref in ast.walk(value_call) if isinstance(ref, ast.Name) and isinstance(ref.ctx, ast.Load) and ref.id in tile_names}) if value_call else []
        operands = sorted({ref.id for ref in ast.walk(value_call) if isinstance(ref, ast.Name) and isinstance(ref.ctx, ast.Load) and ref.id != node.target.id}) if value_call else []
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
                "opKind": op_leaf,
                "group": TILE_GROUPS.get(op_leaf, "其他"),
                "inputs": inputs,
                "operands": operands,
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


def build_callable(callable_name: str, function_type: str) -> dict:
    stages = [extract_stage(*entry, callable_name) for entry in PASS_FILES]
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
    reuse_pair_names = ["m_oi_inline1220__tile", "n_full_inline1211__tile"] if function_type == "AIV" else []
    reuse_pair = [
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
        for name in reuse_pair_names
    ]

    for allocation in stages[3]["allocations"]:
        matching = [tile for tile in stages[3]["tiles"] if tile["base"] == allocation["base"]]
        starts = [tile["offset"] for tile in matching if tile["offset"] is not None]
        allocation["offset"] = min(starts) if starts else 0

    high_water_by_space = {}
    for allocation in stages[3]["allocations"]:
        end = (allocation["offset"] or 0) + (allocation["bytes"] or 0)
        high_water_by_space[allocation["space"]] = max(
            high_water_by_space.get(allocation["space"], 0), end
        )
    address_space = " · ".join(high_water_by_space) or "无片上空间"

    tile_timeline = []
    for number, pass_name, filename in TILE_MILESTONES:
        path = RUN / filename
        try:
            function = callable_node(ast.parse(path.read_text(encoding="utf-8")), callable_name)
            count = sum(1 for node in ast.walk(function) if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and isinstance(node.annotation, ast.Subscript) and ast.unparse(node.annotation.value) == "pl.Tile")
        except (OSError, StopIteration, SyntaxError):
            continue
        tile_timeline.append({"number": number, "pass": pass_name, "count": count, "file": filename})

    # The dump includes the exact frontend snapshot used for this compilation. It
    # may differ from the checked-in source, so source links point to the snapshot.
    source_snapshot = None
    if function_type == "AIV":
        frontend = RUN / "00_frontend.py"
        if frontend.exists():
            source_lines = frontend.read_text(encoding="utf-8").splitlines()
            start = next((i for i, line in enumerate(source_lines) if f'name_hint="{callable_name}_spmd"' in line), None)
            if start is not None:
                end = next((i for i in range(start + 1, len(source_lines)) if source_lines[i].startswith("            _tuple_tmp_")), min(start + 90, len(source_lines)))
                source_snapshot = {"file": "00_frontend.py", "startLine": start + 1, "lines": [{"line": i + 1, "code": source_lines[i]} for i in range(start, end)]}

    return {
        "run": "_jit_l3_decode_csa_20260903_010617",
        "callable": callable_name,
        "functionType": function_type,
        "sourceDirectory": "Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/passes_dump",
        "addressSpace": address_space,
        "reusePair": reuse_pair,
        "semanticChanges": semantic_changes,
        "stages": stages,
        "tileTimeline": tile_timeline,
        "sourceSnapshot": source_snapshot,
        "metrics": {
            "beforeReuseCount": stages[1]["allocationCount"],
            "afterReuseCount": stages[2]["allocationCount"],
            "beforeReuseBytes": stages[1]["allocationBytes"],
            "afterReuseBytes": stages[2]["allocationBytes"],
            "finalHighWater": max(high_water_by_space.values(), default=0),
            "finalHighWaterBySpace": high_water_by_space,
        },
        "notes": [
            "Pass dump 中的源码行号用于定位静态 IR，不是设备 cycle。",
            "生命周期用名称的最后一次静态读取近似；函数含循环和流水 lowering，不能代替运行时并发时序。",
            "分配字节数是该 callable 的 tile.alloc 静态总量，不代表性能提升比例。",
        ],
    }


examples = [build_callable(name, kind) for name, kind in CALLABLES]
data = {
    "defaultCallable": CALLABLES[0][0],
    "callables": examples,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {OUT.relative_to(ROOT)}")
for example in examples:
    print(example["callable"], example["functionType"], "allocations:", [(stage["pass"], stage["allocationCount"], stage["allocationBytes"]) for stage in example["stages"]])
