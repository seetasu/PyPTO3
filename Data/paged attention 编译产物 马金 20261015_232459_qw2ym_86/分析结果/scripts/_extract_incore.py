#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract intra-core (核内) static structure from a PyPTO compile artifact folder.

Sources (all *static* — this artifact set contains no runtime intra-core profiling):
  kernels/{aic,aiv}/*.cpp   -> generated per-core program (each file carries BOTH aic & aiv funcs)
  ptoas/*.pto               -> PTO MLIR (loop nest / tile alloc / sync)
  kernels/*/*.o             -> ELF: .text size (== cache/incore_*.bin), ascend meta, stack size
  cache/incore_*.bin        -> raw core image blob
  dfx_outputs/*.json        -> task-granularity swimlane (kept for the "what is missing" section)

CLI:  python3 _extract_incore.py [SRC] [--out FILE]
"""
import re, os, sys, json, glob, struct
from collections import Counter, defaultdict

DEFAULT_SRC = "/Users/songchenfei/Documents/pypto项目/PyPTO3/Data/paged attention 编译产物 马金 20261015_232459_qw2ym_86"


def parse_cpp(path):
    """Per-core program: op histogram, tile types/shapes, pipe sync pairs, loop nest."""
    lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
    starts = [i for i, l in enumerate(lines)
              if re.match(r"^static\s+__aicore__\s+void\s+attn_swpipe_a(ic|iv)\(", l)]
    starts.append(len(lines))
    out = {}
    for a, b in zip(starts, starts[1:]):
        nm = re.search(r"void (\w+)\(", lines[a]).group(1)
        J = "\n".join(lines[a:b])
        d = {
            "src_lines": b - a,
            "src_start": a + 1,
            "src_end": b,
            "ops": dict(Counter(m.group(1) for m in
                                re.finditer(r"\b([A-Z][A-Z0-9_]{2,})\s*\(", J)).most_common(30)),
            "tile_types": dict(Counter(m.group(1) for m in
                                       re.finditer(r"Tile<TileType::(\w+)", J))),
            "tile_shapes": dict(Counter(m.group(0) for m in
                                        re.finditer(r"Tile<TileType::\w+, \w+, \d+, \d+", J)).most_common(24)),
            "wait_flag": dict(Counter("%s->%s" % (m.group(1), m.group(2)) for m in
                                      re.finditer(r"wait_flag\((PIPE_\w+), (PIPE_\w+), EVENT_ID\d\)", J)).most_common()),
            "set_flag": dict(Counter("%s->%s" % (m.group(1), m.group(2)) for m in
                                     re.finditer(r"set_flag\((PIPE_\w+), (PIPE_\w+), EVENT_ID\d\)", J)).most_common()),
            "event_ids": dict(Counter(m.group(1) for m in
                                      re.finditer(r"(?:wait|set)_flag\(PIPE_\w+, PIPE_\w+, (EVENT_ID\d)\)", J))),
            "loops": {"for": len(re.findall(r"\bfor\s*\(", J)),
                      "if": len(re.findall(r"\bif\s*\(", J)),
                      "while": len(re.findall(r"\bwhile\s*\(", J))},
            "tassign": len(re.findall(r"TASSIGN\(", J)),
            "gm_ptrs": len(re.findall(r"PTOAS__GLOBAL_TENSOR_DATA\(", J)),
        }
        d["wait_total"] = sum(d["wait_flag"].values())
        d["set_total"] = sum(d["set_flag"].values())
        # pipeline pairs are signed: set(A->B) must be waited on B waiting for A
        d["pipes"] = sorted(set(k.split("->")[1] for k in d["wait_flag"]) |
                            set(k.split("->")[0] for k in d["set_flag"]))
        out[nm] = d
    return out


_SZ = {"f32": 4, "bf16": 2, "f16": 2, "i32": 4, "i64": 8, "int8": 1, "u8": 1, "i8": 1, "i16": 2, "f8": 1}


def parse_pto(path):
    """PTO MLIR: op histogram, sync ops, tile allocs + per-memory-space tile footprint."""
    txt = open(path, encoding="utf-8", errors="replace").read().split("\n")
    idx = [i for i, l in enumerate(txt) if l.startswith("  func.func")] + [len(txt)]
    out = {}
    alloc_pat = re.compile(
        r"pto\.alloc_tile addr = %c(\d+)_i64.*?loc=(\w+), dtype=(\w+), rows=(\d+), cols=(\d+)", re.S)
    for a, b in zip(idx, idx[1:]):
        nm = re.search(r"@(\w+)", txt[a]).group(1)
        body = txt[a:b]
        J = "\n".join(body)
        ops = Counter(m.group(0) for l in body
                      for m in re.finditer(r"\b(?:pto|scf|arith|memref|cf|tensor)\.[a-zA-Z_.]+", l))
        sync = Counter(m.group(0) for m in re.finditer(r"pto\.sync\.[a-z_]+", J))
        loc_bytes = defaultdict(int)      # 单个 tile 最大
        loc_peak = defaultdict(int)       # 地址簿记推出的累计峰值 max(addr+size)
        loc_count = defaultdict(int)
        loc_shapes = defaultdict(lambda: defaultdict(int))
        n_alloc = 0
        for m in alloc_pat.finditer(J):
            n_alloc += 1
            addr = int(m.group(1))
            loc, dt, rows, cols = m.group(2), m.group(3), int(m.group(4)), int(m.group(5))
            nb = rows * cols * _SZ.get(dt, 2)
            loc_bytes[loc] = max(loc_bytes[loc], nb)
            loc_peak[loc] = max(loc_peak[loc], addr + nb)
            loc_count[loc] += 1
            loc_shapes[loc][(dt, rows, cols, nb)] += 1
        out[nm] = {
            "lines": len(body),
            "ops": dict(ops.most_common(26)),
            "sync": dict(sync),
            "n_alloc_tile": n_alloc,
            "scf_for": J.count("scf.for"),
            "scf_if": J.count("scf.if"),
            "loc_tile_max_bytes": dict(loc_bytes),
            "loc_peak_bytes": dict(loc_peak),
            "loc_alloc_count": dict(loc_count),
            "loc_shapes": {
                loc: [{"dtype": dt, "rows": r, "cols": c, "bytes": nb, "count": n}
                      for (dt, r, c, nb), n in sorted(v.items(), key=lambda x: -x[0][3])]
                for loc, v in loc_shapes.items()},
        }
    return out


def elf_sections(path):
    """Minimal ELF64 section-header parser (works without binutils)."""
    data = open(path, "rb").read()
    if data[:4] != b"\x7fELF" or data[4] != 2:
        return None, data
    e_shoff = struct.unpack_from("<Q", data, 0x28)[0]
    e_shentsize = struct.unpack_from("<H", data, 0x3A)[0]
    e_shnum = struct.unpack_from("<H", data, 0x3C)[0]
    e_shstrndx = struct.unpack_from("<H", data, 0x3E)[0]
    secs = []
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        name, typ, flags, addr, offset, size = struct.unpack_from("<IIQQQQ", data, off)
        secs.append({"name_off": name, "type": typ, "flags": flags,
                     "addr": addr, "offset": offset, "size": size})
    stro = secs[e_shstrndx]["offset"] if e_shstrndx < len(secs) else 0
    for s in secs:
        end = data.find(b"\0", stro + s["name_off"])
        s["name"] = data[stro + s["name_off"]:end].decode("utf-8", "replace")
    return secs, data


def parse_elf(path):
    secs, data = elf_sections(path)
    if secs is None:
        return None
    sm = {s["name"]: s for s in secs}
    d = {
        "text_bytes": sm[".text"]["size"] if ".text" in sm else None,
        "debug_line_bytes": sm[".debug_line"]["size"] if ".debug_line" in sm else None,
        "debug_info_bytes": sm[".debug_info"]["size"] if ".debug_info" in sm else None,
        "symtab_bytes": sm[".symtab"]["size"] if ".symtab" in sm else None,
        "strtab_bytes": sm[".strtab"]["size"] if ".strtab" in sm else None,
        "sections": len(secs),
    }
    for key, name, fmt in (("stack_bytes", ".ascend.stack.size.record", "<Q"),
                           ("ascend_meta_hex", ".ascend.meta", None)):
        s = sm.get(name)
        if not s:
            continue
        raw = data[s["offset"]:s["offset"] + s["size"]]
        d[key] = raw.hex() if fmt is None else struct.unpack(fmt, raw[:8])[0]
    return d


def extract(src):
    res = {"src": src, "cpp": {}, "pto": {}, "elf": {}, "cache": {}, "dfx": {}}

    for k in ("aic", "aiv"):
        for f in sorted(glob.glob(os.path.join(src, "kernels", k, "*.cpp"))):
            res["cpp"][k] = parse_cpp(f)
            res["cpp"][k]["_file"] = os.path.relpath(f, src)
        for f in sorted(glob.glob(os.path.join(src, "kernels", k, "*.o"))):
            e = parse_elf(f)
            if e:
                e["_file"] = os.path.relpath(f, src)
                res["elf"][k] = e

    for f in sorted(glob.glob(os.path.join(src, "ptoas", "*.pto"))):
        res["pto"] = parse_pto(f)
        res["pto"]["_file"] = os.path.relpath(f, src)

    for b in sorted(glob.glob(os.path.join(src, "cache", "*.bin"))):
        res["cache"][os.path.basename(b)] = {"bytes": os.path.getsize(b)}
    bc = os.path.join(src, "cache", "binary_context.json")
    if os.path.exists(bc):
        c = json.load(open(bc, encoding="utf-8")).get("context", {})
        res["cache"]["_context"] = c

    # dfx_outputs: what the runtime trace does / does not carry
    dfx = os.path.join(src, "dfx_outputs")
    msw = glob.glob(os.path.join(dfx, "merged_swimlane_*.json"))
    if msw:
        d = json.load(open(msw[0], encoding="utf-8"))
        ev = d["traceEvents"]
        names = {(e["pid"], e.get("tid")): e["args"]["name"]
                 for e in ev if e.get("ph") == "M" and e["name"] == "thread_name"}
        lanes = defaultdict(int)
        for e in ev:
            if e.get("ph") == "X":
                lanes[(e["pid"], e.get("tid"))] += 1
        core_lanes = {k: v for k, v in lanes.items() if k[0] in (3, 4)}
        res["dfx"]["swimlane"] = {
            "file": os.path.basename(msw[0]),
            "n_events": len(ev),
            "n_lanes": len(lanes),
            "core_lanes": len(core_lanes),
            "events_per_core_lane": sorted(set(core_lanes.values())),
            "core_event_names": sorted({e["name"] for e in ev
                                        if e.get("ph") == "X" and e["pid"] in (3, 4)}),
            "lane_event_counts": {names.get(k, str(k)): v for k, v in sorted(lanes.items())},
        }
    cs = os.path.join(dfx, "chip_swimlane_records.json")
    if os.path.exists(cs):
        d = json.load(open(cs, encoding="utf-8"))
        res["dfx"]["chip"] = {
            "file": "chip_swimlane_records.json",
            "metadata": d["metadata"],
            "n_aicore_tasks": len(d["aicore_tasks"]),
            "record_arity": sorted({len(r) for r in d["aicore_tasks"]}),
            "has_incore_fields": False,
        }
    return res


def main():
    argv = sys.argv[1:]
    out = None
    if "--out" in argv:
        i = argv.index("--out")
        out = argv[i + 1]
        del argv[i:i + 2]
    src = argv[0] if argv else DEFAULT_SRC
    r = extract(src)
    s = json.dumps(r, ensure_ascii=False, indent=1)
    print(s)
    if out:
        open(out, "w", encoding="utf-8").write(s)
        sys.stderr.write("wrote %s\n" % out)


if __name__ == "__main__":
    main()
