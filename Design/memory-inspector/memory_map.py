#!/usr/bin/env python3
"""Build the Memory Inspector address x static-lifetime model from a pass dump.

The implementation follows the repository's documented ``memory_map.py``
contract in ``memory-map-abnormality-guide.md``: parse Tile annotations with
Python AST, extend SSA lifetimes to their last read, merge consecutive aliases
sharing one physical slot, then classify same-base overlaps as views and
different-base overlaps as conflicts.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path


SPACE_LIMITS = {
    "Vec": 184 * 1024,
    "Mat": 512 * 1024,
    "Left": 64 * 1024,
    "Right": 64 * 1024,
    "Acc": 128 * 1024,
}


def _qualified_name(node: ast.AST | None) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _qualified_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _last_name(node: ast.AST | None) -> str:
    return _qualified_name(node).rsplit(".", 1)[-1]


def _int_value(node: ast.AST | None) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _int_value(node.operand)
        return -value if value is not None else None
    if isinstance(node, ast.Call) and _last_name(node.func) == "const" and node.args:
        return _int_value(node.args[0])
    if isinstance(node, ast.BinOp):
        left, right = _int_value(node.left), _int_value(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
    return None


def _shape(node: ast.AST | None) -> list[int]:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return []
    result = [_int_value(item) for item in node.elts]
    return [value for value in result if value is not None]


def _slice_parts(annotation: ast.AST | None) -> list[ast.AST] | None:
    if not isinstance(annotation, ast.Subscript) or _last_name(annotation.value) != "Tile":
        return None
    value = annotation.slice
    return list(value.elts) if isinstance(value, ast.Tuple) else [value]


def _tile_from_annotation(name: str, annotation: ast.AST, value: ast.AST | None, line: int) -> dict | None:
    parts = _slice_parts(annotation)
    if not parts or len(parts) < 4:
        return None
    memref = parts[2]
    if not isinstance(memref, ast.Call) or _last_name(memref.func) != "MemRef" or len(memref.args) < 3:
        return None
    space = _last_name(parts[3])
    if space not in SPACE_LIMITS:
        return None
    base_node = memref.args[0]
    if isinstance(base_node, ast.Constant):
        base = str(base_node.value)
    else:
        base = _qualified_name(base_node)
    offset, size = _int_value(memref.args[1]), _int_value(memref.args[2])
    if offset is None or size is None:
        return None
    op = _qualified_name(value.func) if isinstance(value, ast.Call) else ""
    return {
        "name": name,
        "space": space,
        "base": base,
        "offset": offset,
        "size": size,
        "shape": _shape(parts[0]),
        "dtype": _last_name(parts[1]),
        "op": op.removeprefix("pl."),
        "start": line,
        "end": line,
        "aliases": [],
        "view": False,
        "conflict": False,
    }


def _function_type(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call) or _last_name(decorator.func) != "function":
            continue
        for keyword in decorator.keywords:
            if keyword.arg == "type":
                kind = _last_name(keyword.value)
                if kind in {"AIC", "AIV", "InCore"}:
                    return kind
    return None


def _read_lines(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, int]:
    reads: dict[str, int] = {}
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            reads[child.id] = max(reads.get(child.id, 0), child.lineno)
    return reads


def _merge_aliases(tiles: list[dict]) -> list[dict]:
    slots: dict[tuple, list[dict]] = defaultdict(list)
    for tile in tiles:
        slots[(tile["space"], tile["base"], tile["offset"], tile["size"])].append(tile)
    boxes: list[dict] = []
    for group in slots.values():
        group.sort(key=lambda item: (item["start"], item["end"], item["name"]))
        current = deepcopy(group[0])
        for tile in group[1:]:
            if tile["start"] <= current["end"]:
                current["end"] = max(current["end"], tile["end"])
                for alias in [tile["name"], *tile["aliases"]]:
                    if alias != current["name"] and alias not in current["aliases"]:
                        current["aliases"].append(alias)
            else:
                boxes.append(current)
                current = deepcopy(tile)
        boxes.append(current)
    order = {name: index for index, name in enumerate(SPACE_LIMITS)}
    boxes.sort(key=lambda item: (order[item["space"]], item["offset"], item["base"], item["start"], item["name"]))
    return boxes


def _overlap(a: dict, b: dict) -> bool:
    address = a["offset"] < b["offset"] + b["size"] and b["offset"] < a["offset"] + a["size"]
    lifetime = a["start"] <= b["end"] and b["start"] <= a["end"]
    return address and lifetime


def _classify_overlaps(boxes: list[dict]) -> None:
    for index, left in enumerate(boxes):
        for right in boxes[index + 1 :]:
            if left["space"] != right["space"] or not _overlap(left, right):
                continue
            if left["base"] != right["base"]:
                left["conflict"] = True
                right["conflict"] = True
                continue
            if (left["offset"], left["size"]) == (right["offset"], right["size"]):
                continue
            left_range = (left["offset"], left["offset"] + left["size"])
            right_range = (right["offset"], right["offset"] + right["size"])
            if left_range[0] >= right_range[0] and left_range[1] <= right_range[1]:
                left["view"] = True
            if right_range[0] >= left_range[0] and right_range[1] <= left_range[1]:
                right["view"] = True


def _parse_function(node: ast.FunctionDef | ast.AsyncFunctionDef, lines: list[str], ftype: str) -> dict:
    reads = _read_lines(node)
    tiles: list[dict] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.AnnAssign) or not isinstance(child.target, ast.Name):
            continue
        tile = _tile_from_annotation(child.target.id, child.annotation, child.value, child.lineno)
        if tile:
            tile["end"] = max(tile["start"], reads.get(tile["name"], tile["start"]))
            tiles.append(tile)
    boxes = _merge_aliases(tiles)
    _classify_overlaps(boxes)
    start = min([node.lineno, *[item.lineno for item in node.decorator_list]])
    end = node.end_lineno or node.lineno
    spaces = []
    for space in SPACE_LIMITS:
        selected = [box for box in boxes if box["space"] == space]
        if not selected:
            continue
        spaces.append({
            "space": space,
            "hwm": max(box["offset"] + box["size"] for box in selected),
            "limit": SPACE_LIMITS[space],
            "tiles": len(selected),
            "bases": len({box["base"] for box in selected}),
        })
    return {
        "name": node.name,
        "ftype": ftype,
        "src_start": start,
        "src_end": end,
        "source": lines[start - 1 : end],
        "spaces": spaces,
        "boxes": boxes,
    }


def parse_dump(path: str | Path) -> dict:
    dump_path = Path(path)
    source = dump_path.read_text(encoding="utf-8")
    source_lines = source.splitlines()
    tree = ast.parse(source, filename=str(dump_path))
    functions = []
    for node in sorted(
        (item for item in ast.walk(tree) if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))),
        key=lambda item: item.lineno,
    ):
        ftype = _function_type(node)
        if ftype:
            parsed = _parse_function(node, source_lines, ftype)
            if parsed["boxes"]:
                functions.append(parsed)
    match = re.search(r"\d+_(?:after|before)_([^/]+)\.py$", dump_path.name)
    return {
        "dump": dump_path.name,
        "pass_name": match.group(1) if match else dump_path.stem,
        "backend": {"name": "Ascend910B", "arch": "a2a3", "detected": False},
        "functions": functions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    args = parser.parse_args()
    result = parse_dump(args.dump)
    payload = json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
