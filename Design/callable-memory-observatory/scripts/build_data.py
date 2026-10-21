"""Build the callable overview and per-callable memory detail from pass dumps."""

from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN_ID = "_jit_l3_decode_csa_20260903_010617"
RUN = ROOT / "Data/DeepseekV4" / RUN_ID / "passes_dump"
OUT = Path(__file__).resolve().parents[1] / "data/memory-observatory.json"
PASS_FILES = [
    (33, "InitMemRef", "33_after_InitMemRef.py"),
    (34, "MaterializeSemanticAliases", "34_after_MaterializeSemanticAliases.py"),
    (35, "MemoryReuse", "35_after_MemoryReuse.py"),
    (36, "AllocateMemoryAddr", "36_after_AllocateMemoryAddr.py"),
]


def integer(node: ast.AST | None) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = integer(node.operand)
        return -value if value is not None else None
    if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "const" and node.args:
        return integer(node.args[0])
    return None


def function_type(node: ast.FunctionDef) -> str:
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call) or ast.unparse(decorator.func) != "pl.function":
            continue
        for keyword in decorator.keywords:
            if keyword.arg == "type":
                return ast.unparse(keyword.value).split(".")[-1]
    return "Unknown"


def is_callable(node: ast.FunctionDef) -> bool:
    return any(
        isinstance(decorator, ast.Call) and ast.unparse(decorator.func) == "pl.function"
        for decorator in node.decorator_list
    )


def ref_name(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ast.unparse(node)


def extract_function(
    node: ast.FunctionDef, lines: list[str], number: int, pass_name: str, filename: str
) -> dict:
    allocations = []
    for stmt in ast.walk(node):
        if not (
            isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and isinstance(stmt.value, ast.Call)
            and getattr(stmt.value.func, "attr", "") == "alloc"
            and len(stmt.value.args) >= 2
        ):
            continue
        allocations.append(
            {
                "base": stmt.target.id,
                "space": ast.unparse(stmt.value.args[0]).replace("pl.Mem.", ""),
                "bytes": integer(stmt.value.args[1]) or 0,
                "line": stmt.lineno,
                "code": lines[stmt.lineno - 1].strip(),
                "offset": None,
                "owners": [],
            }
        )

    tiles = []
    for stmt in ast.walk(node):
        if not (
            isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and isinstance(stmt.annotation, ast.Subscript)
            and ast.unparse(stmt.annotation.value) == "pl.Tile"
            and isinstance(stmt.annotation.slice, ast.Tuple)
            and len(stmt.annotation.slice.elts) >= 4
        ):
            continue
        shape, dtype, memref, space = stmt.annotation.slice.elts[:4]
        if not isinstance(memref, ast.Call) or getattr(memref.func, "attr", "") != "MemRef":
            continue
        base = ref_name(memref.args[0]) if memref.args else ""
        offset = integer(memref.args[1]) if len(memref.args) > 1 else None
        size = integer(memref.args[2]) if len(memref.args) > 2 else None
        uses = [
            ref.lineno
            for ref in ast.walk(node)
            if isinstance(ref, ast.Name)
            and ref.id == stmt.target.id
            and isinstance(ref.ctx, ast.Load)
            and ref.lineno >= stmt.lineno
        ]
        value_call = stmt.value
        tile = {
            "name": stmt.target.id,
            "base": base,
            "space": ast.unparse(space).replace("pl.Mem.", ""),
            "shape": ast.unparse(shape),
            "dtype": ast.unparse(dtype).replace("pl.", ""),
            "bytes": size or 0,
            "offset": offset,
            "op": ast.unparse(value_call.func) if isinstance(value_call, ast.Call) else "alias",
            "line": stmt.lineno,
            "lastUseLine": max(uses, default=stmt.lineno),
            "code": lines[stmt.lineno - 1].strip(),
        }
        tiles.append(tile)

    tiles.sort(key=lambda item: item["line"])
    tiles_by_base: dict[str, list[dict]] = {}
    for tile in tiles:
        tiles_by_base.setdefault(tile["base"], []).append(tile)
    for alloc in allocations:
        owners = tiles_by_base.get(alloc["base"], [])
        alloc["owners"] = [tile["name"] for tile in owners]
        starts = [tile["offset"] for tile in owners if tile["offset"] is not None]
        alloc["offset"] = min(starts) if starts else None
    allocations.sort(key=lambda item: item["line"])

    space_bytes = Counter()
    for alloc in allocations:
        space_bytes[alloc["space"]] += alloc["bytes"]
    high_water = {}
    for space_name in space_bytes:
        ends = [
            (alloc["offset"] or 0) + alloc["bytes"]
            for alloc in allocations
            if alloc["space"] == space_name and alloc["offset"] is not None
        ]
        high_water[space_name] = max(ends, default=0)

    return {
        "number": number,
        "pass": pass_name,
        "file": filename,
        "functionStart": node.lineno,
        "functionEnd": node.end_lineno,
        "tileCount": len(tiles),
        "allocationCount": len(allocations),
        "allocationBytes": sum(alloc["bytes"] for alloc in allocations),
        "spaceBytes": dict(space_bytes),
        "highWaterBySpace": high_water,
        "allocations": allocations,
        "tiles": tiles,
    }


def main() -> None:
    parsed = []
    for number, pass_name, filename in PASS_FILES:
        source = (RUN / filename).read_text(encoding="utf-8")
        lines = source.splitlines()
        tree = ast.parse(source)
        callables = {
            node.name: (node, function_type(node))
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and is_callable(node)
        }
        parsed.append((number, pass_name, filename, lines, callables))

    names = list(parsed[0][4])
    index = []
    details = {}
    for name in names:
        stages = []
        kind = "Unknown"
        for number, pass_name, filename, lines, callables in parsed:
            node, kind = callables[name]
            stages.append(extract_function(node, lines, number, pass_name, filename))
        index.append(
            {
                "name": name,
                "type": kind,
                "stages": [
                    {
                        "number": stage["number"],
                        "pass": stage["pass"],
                        "tileCount": stage["tileCount"],
                        "slotCount": stage["allocationCount"],
                        "bytes": stage["allocationBytes"],
                        "spaceBytes": stage["spaceBytes"],
                        "highWaterBySpace": stage["highWaterBySpace"],
                    }
                    for stage in stages
                ],
            }
        )
        if any(stage["allocationCount"] for stage in stages):
            details[name] = {"name": name, "type": kind, "stages": stages}

    index.sort(key=lambda item: (-item["stages"][-1]["bytes"], item["name"]))
    result = {
        "schemaVersion": "callable-memory-observatory.v1",
        "run": RUN_ID,
        "sourceDirectory": f"Data/DeepseekV4/{RUN_ID}/passes_dump",
        "passChain": [
            {"number": number, "pass": pass_name, "file": filename}
            for number, pass_name, filename in PASS_FILES
        ],
        "callables": index,
        "details": details,
        "notes": [
            "容量是各 callable 中 tile.alloc 的静态字节数求和，不代表同时运行峰值。",
            "不同 callable 可能在调用图中串行、并行或按条件执行；跨 callable 汇总不能视为硬件总占用。",
            "偏移只在 AllocateMemoryAddr 快照中代表已分配地址；静态源码行号不表示运行时存活区间。",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    with_alloc = sum(any(stage["slotCount"] for stage in item["stages"]) for item in index)
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"callables: {len(index)} ({with_alloc} with on-chip allocations); detail views: {len(details)}")
    for item in index[:5]:
        print(item["name"], item["type"], item["stages"][0]["bytes"], "->", item["stages"][-1]["bytes"])


if __name__ == "__main__":
    main()
