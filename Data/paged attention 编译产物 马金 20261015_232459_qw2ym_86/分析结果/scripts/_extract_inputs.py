"""抽取一次编译产物的「输入定义」，供查看器「输入」页渲染。

五路来源，各管一段，互不猜测：

  1. compiled_meta.json      → 对外 API 的 15 个参数（shape / dtype / direction / layout）
  2. dfx_outputs/deps.json   → kernel 实际收到的 19 个实参（含内部 workspace、摊平视图）
  3. passes_dump/00_frontend.py
                             → 前端函数签名（workspace 声明）、派生量表达式、spmd 块数
  4. debug/run.py            → _inline_inputs() 里 torch 的构造方式（重放默认值）
  5. profile bundle README   → 运行期 workload 命令行（如果有）

设计原则：**表达式原样保留**，能算的值才算（只算和静态 shape 相关的），
不确定的一律留 None，由 UI 标成「未记录」，不猜。
"""

import json
import os
import re


def _read(p):
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


# ---------------------------------------------------------------- 1. 对外参数
def parse_meta(root):
    d = json.loads(_read(os.path.join(root, "compiled_meta.json")) or "{}")
    out = []
    for p in d.get("params", []):
        out.append({
            "name": p.get("name"),
            "shape": p.get("shape"),
            "dtype": p.get("dtype"),
            "direction": p.get("direction"),
            "layout": p.get("layout"),
        })
    return {
        "platform": d.get("platform"),
        "backend": d.get("backend_type"),
        "modes": d.get("supported_execution_modes") or [],
        "params": out,
    }


# ---------------------------------------------------------- 2. kernel 实参
def parse_deps(root):
    d = json.loads(_read(os.path.join(root, "dfx_outputs", "deps.json")) or "{}")
    tasks = d.get("tasks") or []
    # 真 kernel 的那个 task：kernel_ids 非 [-1,-1,-1]
    real = None
    for t in tasks:
        kids = t.get("kernel_ids") or []
        if any(k is not None and k >= 0 for k in kids):
            real = t
    if real is None:
        return {"n_tasks": len(tasks), "block_num": None, "args": []}
    args = []
    for a in real.get("args", []):
        args.append({
            "idx": a.get("idx"),
            "type": a.get("type"),
            "dtype": a.get("dtype"),
            "shape": a.get("shape"),
            "strides": a.get("strides"),
        })
    return {
        "n_tasks": len(tasks),
        "block_num": real.get("block_num"),
        "kernel_ids": real.get("kernel_ids"),
        "args": args,
    }


# ------------------------------------------------------- 3. 前端签名与派生量
_SIG_HEAD = re.compile(r"^static\s+__aicore__\s+void|def\s+paged_attention\w*\(", re.M)
_PARAM = re.compile(r"^\s{4,}(\w+)\s*:\s*pl\.(Tensor|Scalar)\[(.*?)\],?\s*$")


def parse_frontend(root):
    txt = _read(os.path.join(root, "passes_dump", "00_frontend.py"))
    params, in_sig, done = [], False, False
    for line in txt.split("\n"):
        if done:
            break
        if not in_sig:
            if re.match(r"^\s+def \w+\(", line):
                in_sig = True
            continue
        if re.match(r"^\s*\)\s*->", line):
            in_sig, done = False, True
            continue
        m = _PARAM.match(line)
        if not m:
            continue
        name, kind, inner = m.group(1), m.group(2), m.group(3)
        item = {"name": name, "kind": kind.lower()}
        if kind == "Tensor":
            # inner 形如  [80, 128], pl.BF16   或   [cache_token_rows, 1024], pl.BF16
            parts = inner.rsplit(",", 1)
            shp = parts[0].strip()
            item["shape_expr"] = shp
            item["dtype"] = parts[1].strip().replace("pl.", "") if len(parts) > 1 else None
            item["shape"] = _shape_literal(shp)
        else:
            item["scalar_type"] = inner.strip().replace("pl.", "")
            item["shape"] = None
        params.append(item)

    # 派生量：抓「名字 = 表达式」整行，只对静态可算的求值
    want = {
        "active_batch": None,
        "num_tasks": None,
        "query_rows": None,
        "cache_token_rows": None,
        "max_blocks_per_seq": None,
        "page_count": None,
        "stack_count": None,
        "transfer_base": None,
        "qp_row": None,
        "cache_base": None,
    }
    for name in want:
        m = re.search(r"^\s*" + name + r"\s*:\s*pl\.Scalar\[[^\]]*\]\s*=\s*(.+?)\s*$", txt, re.M)
        if m:
            want[name] = m.group(1)
    derived = [{"name": k, "expr": v} for k, v in want.items() if v]

    spmd = re.search(r"pl\.spmd\(\s*(\d+)", txt)
    split = re.search(r"pl\.split_aiv\(\s*(\d+)", txt)
    # 页大小 / 分栈大小：从 (x + N - 1) // N 里取 N
    page = re.search(r"page_count.*?\+\s*(\d+)\s*-\s*1\)\s*//\s*(\d+)", txt)
    stack = re.search(r"stack_count.*?\+\s*(\d+)\s*-\s*1\)\s*//\s*(\d+)", txt)
    loop = re.search(r"for\s+(\w+)\s+in\s+pl\.range\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\d+)\s*\)", txt)

    return {
        "params": params,
        "n_params": len(params),
        "derived": derived,
        "blocks": int(spmd.group(1)) if spmd else None,
        "aiv_split": int(split.group(1)) if split else None,
        "page_size": int(page.group(2)) if page else None,
        "stack_size": int(stack.group(2)) if stack else None,
        "task_loop": ({"var": loop.group(1), "start": loop.group(2),
                       "stop": loop.group(3), "step": int(loop.group(4))} if loop else None),
    }


def _shape_literal(expr):
    """`[80, 128]` → [80,128]；含符号的 `[cache_token_rows, 1024]` → None（不猜）。"""
    expr = expr.strip()
    if not (expr.startswith("[") and expr.endswith("]")):
        return None
    body = expr[1:-1].strip()
    if not body:
        return []
    out = []
    for tok in body.split(","):
        tok = tok.strip()
        if re.fullmatch(r"-?\d+", tok):
            out.append(int(tok))
        else:
            return None
    return out


# ------------------------------------------------ 4. debug/run.py 的构造方式
_CTOR = re.compile(r"^\s{4}(\w+)\s*=\s*(torch\.\w+\(.*|None\b.*)$")


def parse_ctor(root):
    txt = _read(os.path.join(root, "debug", "run.py"))
    out = []
    for line in txt.split("\n"):
        m = _CTOR.match(line)
        if m:
            out.append({"name": m.group(1), "expr": m.group(2).strip()})
    return out


# ---------------------------------------------------- 5. 运行期 workload 记录
def parse_runs(root):
    """从 profile bundle 的 README 里抓命令行；抓不到的 run 明确标 None。"""
    runs = []
    parent = os.path.dirname(root.rstrip("/"))
    roots = [root.rstrip("/"), parent]
    seen = set()
    for base in roots:
        if not os.path.isdir(base):
            continue
        for sub in sorted(os.listdir(base)):
            d = os.path.join(base, sub)
            if not os.path.isdir(d) or d in seen:
                continue
            seen.add(d)
            for f in os.listdir(d):
                if not f.endswith("_README.txt"):
                    continue
                txt = _read(os.path.join(d, f))
                m = re.search(r"command:\s*(.+)", txt)
                sl = re.search(r"--seq-lens\s+([\d,]+)", txt or "")
                dm = re.search(r"(\d{8})", sub)
                runs.append({
                    "bundle": sub,
                    "cmd": m.group(1).strip() if m else None,
                    "seq_lens": sl.group(1) if sl else None,
                    "date": dm.group(1) if dm else None,
                    "readme": f,
                })
    runs.sort(key=lambda r: r.get("date") or "")
    return runs


def _values(meta, front, deps):
    """由静态 shape 推出的 workload 配置。算不出来的一律 None，不猜。"""
    p = {x["name"]: x for x in meta["params"]}

    def shp(n):
        s = p.get(n, {}).get("shape")
        return s if isinstance(s, list) else None

    v = {}
    sl, q, kp, bt, kc, out = (shp("seq_lens"), shp("q_proj"), shp("k_proj"),
                              shp("block_table"), shp("key_cache"), shp("out"))
    v["batch"] = sl[0] if sl else None
    v["query_tokens"] = q[0] if q else None
    v["hidden"] = q[1] if q and len(q) > 1 else None
    v["heads"] = out[1] if out and len(out) > 1 else None
    v["head_dim"] = out[2] if out and len(out) > 2 else None
    v["kv_dim"] = kp[1] if kp and len(kp) > 1 else None
    v["kv_heads"] = (v["kv_dim"] // v["head_dim"]) if v["kv_dim"] and v["head_dim"] else None
    v["group"] = (v["heads"] // v["kv_heads"]) if v["heads"] and v["kv_heads"] else None
    v["blocks"] = front["blocks"]
    v["aiv_split"] = front["aiv_split"]
    v["aiv_total"] = (front["blocks"] * front["aiv_split"]) if front["blocks"] else None
    v["num_tasks"] = (v["batch"] * v["kv_heads"]) if v["batch"] and v["kv_heads"] else None
    v["max_blocks_per_seq"] = (bt[0] // v["batch"]) if bt and v["batch"] else None
    v["cache_token_rows"] = (kc[0] * kc[1] // kc[1]) if kc else None
    v["page_size"] = front["page_size"]
    v["stack_size"] = front["stack_size"]
    v["n_api"] = len(meta["params"])
    v["n_kernel"] = len(deps["args"])
    v["n_workspace"] = max(0, v["n_kernel"] - v["n_api"])
    return v


_DTN = {"bfloat16": "bf16", "bf16": "bf16", "f16": "f16", "half": "f16",
        "fp32": "f32", "float32": "f32", "f32": "f32",
        "int32": "i32", "int64": "i64", "int8": "i8", "uint8": "u8",
        "index": "idx"}


def _norm_dt(x):
    return _DTN.get(str(x).lower(), str(x).lower()) if x else None


def _numel(shape):
    if not shape:
        return None
    n = 1
    for x in shape:
        n *= x
    return n


def map_kernel_names(front, deps):
    """kernel 实参没有名字（cpp 里就叫 v1..v19），只能按 (dtype, numel) 贪心匹配回前端签名。

    顺序敏感：同名同形的 k_proj / v_proj、key_cache / value_cache 靠「先到先得」区分，
    所以必须按 kernel 实参下标顺序推进，不能反查。
    """
    cands = []
    for p in front["params"]:
        if p["kind"] == "tensor" and p.get("shape"):
            cands.append({"name": p["name"], "dt": _norm_dt(p.get("dtype")),
                          "numel": _numel(p["shape"])})
    used, out = set(), []
    for a in deps["args"]:
        dt, ne = _norm_dt(a.get("dtype")), _numel(a.get("shape"))
        hit = None
        for i, c in enumerate(cands):
            if i in used:
                continue
            # block_table 会被 reshape（[64] → [2,32]），所以比 numel 不比 shape
            if c["dt"] == dt and c["numel"] == ne:
                hit = i
                break
        if hit is None:
            out.append(None)
        else:
            used.add(hit)
            out.append(cands[hit]["name"])
    return out


def extract(root):
    root = os.path.abspath(root)
    meta = parse_meta(root)
    deps = parse_deps(root)
    front = parse_frontend(root)
    ctor = parse_ctor(root)
    runs = parse_runs(root)

    api_names = [p["name"] for p in meta["params"]]
    kern_names = map_kernel_names(front, deps)

    return {
        "root": root,
        "meta": meta,
        "deps": deps,
        "front": front,
        "ctor": ctor,
        "runs": runs,
        "api_names": api_names,
        "kernel_names": kern_names,
        "values": _values(meta, front, deps),
    }


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "."
    d = extract(src)
    print(json.dumps(d, ensure_ascii=False, indent=1))
