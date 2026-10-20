#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract the 20261016 profiling bundle (PMU + msprof op simulator) for the viewer.

Three very different trust levels — never mix them:
  HARDWARE  dfx_outputs/pmu.csv             real PMU counters, 72 cores, group 7 (MEMORY_UB/MTE bw)
  HARDWARE  dfx_outputs/merged_swimlane_*   real timeline in us (the only usable time base)
  MODELED   .../kernel_insight_*/simulator  msprof op simulator, single core, synthetic seq_lens 128/129
  FAILED    costmodel_attn_swpipe           did not compile — not a measurement

CLI: python3 _extract_profile.py [BUNDLE_DIR] [--out FILE]
"""
import csv, json, os, re, sys, glob
from collections import Counter, defaultdict

DEFAULT_BUNDLE = os.path.join(
    "/Users/songchenfei/Documents/pypto项目/PyPTO3/Data/paged attention 编译产物 马金 20261015_232459_qw2ym_86",
    "qwen3_14b_paged_attention_profile_20261016")

BW_COLS = ["ub_read_bw_mte", "l2_write_bw", "main_mem_write_bw", "ub_write_bw_mte",
           "ub_read_bw_vector", "ub_write_bw_vector", "ub_read_bw_scalar", "ub_write_bw_scalar"]


def parse_pmu(path):
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        d = {"core": int(r["core_id"]), "thread": int(r["thread_id"]),
             "type": "aic" if r["core_type"] == "0" else "aiv",
             "cycles": int(r["pmu_total_cycles"]), "epoch": int(r["run_epoch"])}
        for c in BW_COLS:
            d[c] = int(r[c])
        rows.append(d)
    rows.sort(key=lambda x: x["core"])

    def agg(sel):
        g = [r for r in rows if sel(r)]
        if not g:
            return None
        cyc = [r["cycles"] for r in g]
        o = {"n": len(g), "cycles_min": min(cyc), "cycles_mean": round(sum(cyc) / len(cyc), 1),
             "cycles_max": max(cyc), "cycles_sum": sum(cyc)}
        for c in BW_COLS:
            v = [r[c] for r in g]
            o[c] = round(sum(v) / len(v), 1)
        return o

    threads = {}
    for t in sorted({r["thread"] for r in rows}):
        threads[str(t)] = {
            "aic": agg(lambda r, t=t: r["thread"] == t and r["type"] == "aic"),
            "aiv": agg(lambda r, t=t: r["thread"] == t and r["type"] == "aiv"),
            "cores": sorted(r["core"] for r in rows if r["thread"] == t),
        }
    return {
        "file": "dfx_outputs/pmu.csv",
        "pmu_group": 7, "pmu_group_label": "MEMORY_UB / MTE bandwidth",
        "n_rows": len(rows), "n_cores": len({r["core"] for r in rows}),
        "cores": rows, "threads": threads,
        "aic": agg(lambda r: r["type"] == "aic"),
        "aiv": agg(lambda r: r["type"] == "aiv"),
    }


def parse_trace(path, tag):
    d = json.load(open(path, encoding="utf-8"))
    ev = d["traceEvents"]
    names = {(e["pid"], e.get("tid")): e["args"]["name"]
             for e in ev if e.get("ph") == "M" and e["name"] == "thread_name"}
    pnames = {e["pid"]: e["args"]["name"]
              for e in ev if e.get("ph") == "M" and e["name"] == "process_name"}
    x = [e for e in ev if e.get("ph") == "X"]
    t0 = min(e["ts"] for e in x)
    t1 = max(e["ts"] + e.get("dur", 0) for e in x)

    # 每个核的 slice（取 Scheduler View pid=3，避免 pid4 重复计数）
    cores = []
    for e in x:
        if e["pid"] == 3:
            nm = names.get((3, e.get("tid")), "")
            cores.append({"name": nm, "start": round(e["ts"], 2),
                          "dur": round(e.get("dur", 0), 2)})
    cores.sort(key=lambda c: c["start"])

    # 阶段：按 lane 名字归类
    lanes = defaultdict(lambda: {"n": 0, "dur": 0.0})
    for e in x:
        nm = names.get((e["pid"], e.get("tid")), "?"
                       )
        lanes[nm]["n"] += 1
        lanes[nm]["dur"] += e.get("dur", 0)

    phases = []
    for e in x:
        if e["pid"] in (1, 2):
            phases.append({"pid": e["pid"], "lane": names.get((e["pid"], e.get("tid")), ""),
                           "name": e["name"], "ts": round(e["ts"], 2), "dur": round(e.get("dur", 0), 2)})
    phases.sort(key=lambda p: p["ts"])

    cstarts = [c["start"] for c in cores]
    return {
        "tag": tag, "file": os.path.basename(path),
        "n_events": len(ev), "n_x": len(x),
        "n_flows": sum(1 for e in ev if e.get("ph") in ("s", "f")),
        "n_lanes": len(names),
        "procs": pnames,
        "window_us": round(t1 - t0, 2),
        "cores": cores,
        "core_start_min": min(cstarts) if cstarts else None,
        "core_start_max": max(cstarts) if cstarts else None,
        "core_end_max": round(max(c["start"] + c["dur"] for c in cores), 2) if cores else None,
        "kernel_window_us": round(max(c["start"] + c["dur"] for c in cores) - min(cstarts), 2) if cores else None,
        "core_dur_mean": round(sum(c["dur"] for c in cores) / len(cores), 2) if cores else None,
        "core_dur_max": max((c["dur"] for c in cores), default=None),
        "lanes": {k: {"n": v["n"], "dur": round(v["dur"], 2)} for k, v in lanes.items()},
        "phases": phases,
    }


def parse_log_funcs(path):
    """The swimlane log prints the runtime's own per-func statistics — authoritative for exec/latency."""
    txt = open(path, encoding="utf-8", errors="replace").read()
    funcs, per, disp = [], None, None
    m = re.search(r"Func_ID\s+Func_Name.*?\n-+\n(.*?)\n*TOTAL", txt, re.S)
    if m:
        for line in m.group(1).strip().split("\n"):
            p = line.split()
            if len(p) < 10 or not p[0].isdigit():
                continue
            funcs.append({"func_id": int(p[0]), "name": p[1], "count": int(p[2]),
                          "exec_us": float(p[3]), "latency_us": float(p[4]),
                          "exec_pct": float(p[5].rstrip("%")), "head_oh_us": float(p[6]),
                          "tail_oh_us": float(p[7]), "prop_us": float(p[8]), "local_us": float(p[9])})
    m = re.search(r"Total Test Time:\s*([\d.]+)\s*us", txt)
    total = float(m.group(1)) if m else None
    m = re.search(r"Avg Exec = ([\d.]+) us,\s*Avg Latency \(dispatch->finish\) = ([\d.]+) us,\s*"
                  r"Exec/Latency = ([\d.]+)%", txt)
    if m:
        per = {"avg_exec_us": float(m.group(1)), "avg_latency_us": float(m.group(2)),
               "exec_latency_pct": float(m.group(3))}
    m = re.search(r"Dispatch→kernel start \(Host-computed\): Total = ([\d.]+) us, Max = ([\d.]+) us", txt)
    if m:
        disp = {"total_us": float(m.group(1)), "max_us": float(m.group(2))}
    # 构建/校验信息
    extra = {}
    m = re.search(r"code_bytes=(\d+) code_fnv1a64=(\S+)", txt)
    if m:
        extra["aic_code_bytes"] = int(m.group(1))
    for m in re.finditer(r"func_id=(\d+) code_begin=\S+ code_end_exclusive=\S+ code_bytes=(\d+)", txt):
        extra["code_bytes_%s" % m.group(1)] = int(m.group(2))
    v = re.findall(r"\[RUN\]\s+'(key_cache|value_cache|out)'\s+(PASS|FAIL)", txt)
    extra["validated"] = [{"tensor": a, "status": b} for a, b in v]
    return {"file": os.path.basename(path), "funcs": funcs, "per_task": per,
            "dispatch": disp, "total_test_us": total, "extra": extra}


def parse_sim(simdir):
    """msprof op simulator: per-pipe + per-instruction cycle breakdown (MODELED)."""
    out = {"dir": simdir, "cores": {}, "pipes": {}, "top_instr": [], "trace": None}
    for f in sorted(glob.glob(os.path.join(simdir, "core0.*", "*_instr_exe.csv"))):
        core = os.path.basename(os.path.dirname(f))
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        if not rows:
            continue
        tot = sum(int(r["cycles"]) for r in rows)
        byp = defaultdict(lambda: [0, 0, 0])
        for r in rows:
            b = byp[r["pipe"]]
            b[0] += int(r["cycles"]); b[1] += 1; b[2] += int(r["call_count"])
        agg = defaultdict(lambda: [0, 0, 0])
        for r in rows:
            a = agg[r["instr"]]
            a[0] += int(r["cycles"]); a[1] += int(r["call_count"]); a[2] += 1
        out["cores"][core] = {
            "file": os.path.basename(f), "rows": len(rows),
            "cycles": tot, "us": round(sum(float(r["running_time(us)"]) for r in rows), 3),
            "calls": sum(int(r["call_count"]) for r in rows),
            "pipes": [{"pipe": p, "cycles": v[0], "rows": v[1], "calls": v[2],
                       "pct": round(100.0 * v[0] / tot, 1)}
                      for p, v in sorted(byp.items(), key=lambda x: -x[1][0])],
            "instr": [{"instr": k, "cycles": v[0], "calls": v[1], "rows": v[2],
                       "pct": round(100.0 * v[0] / tot, 1)}
                      for k, v in sorted(agg.items(), key=lambda x: -x[1][0])[:14]],
        }
    tp = os.path.join(simdir, "trace.json")
    if os.path.exists(tp):
        d = json.load(open(tp, encoding="utf-8"))
        ev = d["traceEvents"]
        x = [e for e in ev if e.get("ph") == "X"]
        lanes = defaultdict(lambda: {"n": 0, "dur": 0.0})
        for e in x:
            lanes[e.get("tid", "?")]["n"] += 1
            lanes[e.get("tid", "?")]["dur"] += e.get("dur", 0)
        t0 = min(e["ts"] for e in x)
        out["trace"] = {
            "file": "simulator/trace.json", "n_events": len(ev), "n_x": len(x),
            "span_us": round(max(e["ts"] + e.get("dur", 0) for e in x) - t0, 2),
            "cores": sorted({e.get("pid") for e in x}),
            "lanes": [{"pipe": k, "n": v["n"], "dur": round(v["dur"], 2)}
                      for k, v in sorted(lanes.items(), key=lambda x: -x[1]["dur"])],
        }
    return out


def parse_costmodel(d):
    st = os.path.join(d, "STATUS.txt")
    lg = os.path.join(d, "build.log")
    o = {"dir": d, "status": "failed", "status_txt": None, "log_tail": []}
    if os.path.exists(st):
        o["status_txt"] = open(st, encoding="utf-8", errors="replace").read().strip()
    if os.path.exists(lg):
        o["log_tail"] = open(lg, encoding="utf-8", errors="replace").read().strip().split("\n")[-12:]
    return o


def extract(bundle):
    res = {"bundle": bundle, "meta": {}, "pmu": None, "runs": {}, "sim": None,
           "log": None, "costmodel": None}
    rd = os.path.join(bundle, "qwen3_14b_paged_attention_profile_20261016_README.txt")
    if os.path.exists(rd):
        res["meta"]["readme"] = open(rd, encoding="utf-8", errors="replace").read()

    builds = sorted(glob.glob(os.path.join(bundle, "build_output", "_jit_*")))
    for b in builds:
        pmu = os.path.join(b, "dfx_outputs", "pmu.csv")
        if os.path.exists(pmu):
            res["meta"]["hw_build"] = os.path.basename(b)
            res["pmu"] = parse_pmu(pmu)
            sw = glob.glob(os.path.join(b, "dfx_outputs", "merged_swimlane_*.json"))
            if sw:
                res["runs"]["new"] = parse_trace(sw[0], "new(20261016 · b2 · seq 512/513)")
        cm = os.path.join(b, "costmodel_attn_swpipe")
        if os.path.isdir(cm):
            res["costmodel"] = parse_costmodel(cm)
        ki = glob.glob(os.path.join(b, "kernel_insight_all_funcs_*"))
        for k in ki:
            sim = os.path.join(k, "funcs", "attn_swpipe", "collect", "out")
            for o in glob.glob(os.path.join(sim, "OPPROF_*")):
                res["sim"] = parse_sim(os.path.join(o, "simulator"))
                res["sim"]["root"] = os.path.relpath(o, bundle)
                res["meta"]["sim_build"] = os.path.basename(b)
            wl = os.path.join(k, "cases", "ptoas", "attn_swpipe_msprof", "WORKLOAD.txt")
            if os.path.exists(wl):
                res["meta"]["workload"] = open(wl, encoding="utf-8", errors="replace").read()
            man = os.path.join(k, "MANUAL_COLLECT_STATUS.txt")
            if os.path.exists(man):
                res["meta"]["manual_status"] = open(man, encoding="utf-8", errors="replace").read()

    # 旧产物（同一 kernel 的另一次运行）用于对比
    old = os.path.join(os.path.dirname(bundle.rstrip("/")), "dfx_outputs",
                       "merged_swimlane_20261015_232514.json")
    if os.path.exists(old):
        res["runs"]["old"] = parse_trace(old, "old(20261015)")

    for f in glob.glob(os.path.join(bundle, "*swimlane*.log")):
        res["log"] = parse_log_funcs(f)
    return res


def main():
    argv = sys.argv[1:]
    out = None
    if "--out" in argv:
        i = argv.index("--out")
        out = argv[i + 1]
        del argv[i:i + 2]
    bundle = argv[0] if argv else DEFAULT_BUNDLE
    r = extract(bundle)
    s = json.dumps(r, ensure_ascii=False, indent=1)
    print(s)
    if out:
        open(out, "w", encoding="utf-8").write(s)
        sys.stderr.write("wrote %s\n" % out)


if __name__ == "__main__":
    main()
