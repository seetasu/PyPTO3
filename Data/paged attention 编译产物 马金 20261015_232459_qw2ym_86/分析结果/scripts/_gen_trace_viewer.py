#!/usr/bin/env python3
# 生成 dfx_outputs 的 chrome-tracing 风格 HTML 查看器（自包含、内联数据）
import json, os, re, html, sys

SRC = "/Users/songchenfei/Documents/pypto项目/PyPTO3/Data/paged attention 编译产物 马金 20261015_232459_qw2ym_86/dfx_outputs"
OUT = "/Users/songchenfei/WorkBuddy/2026-10-16-15-24-07/dfx_outputs_chrome_tracing.html"

merged = json.load(open(os.path.join(SRC, "merged_swimlane_20261015_232514.json")))
chip   = json.load(open(os.path.join(SRC, "chip_swimlane_records.json")))
deps   = json.load(open(os.path.join(SRC, "deps.json")))
namemap= json.load(open(os.path.join(SRC, "name_map__jit_paged_attention_pypto_dynamic_20261015_232459_qw2ym_86.json")))

# ---------- 预处理 trace ----------
events = merged["traceEvents"]
proc_names, thread_names = {}, {}
for e in events:
    if e.get("ph") == "M":
        if e["name"] == "process_name":
            proc_names[e["pid"]] = e["args"]["name"]
        elif e["name"] == "thread_name":
            thread_names[(e["pid"], e["tid"])] = e["args"]["name"]

lanes, order = {}, []
for e in events:
    if e.get("ph") == "X":
        k = (e["pid"], e["tid"])
        if k not in lanes:
            lanes[k] = []
            order.append(k)
        lanes[k].append(e)

# 稳定排序：按 pid, 再按 tid
order.sort(key=lambda k: (k[0], k[1]))

lane_list = []
for pid, tid in order:
    lane_list.append({
        "pid": pid, "tid": tid,
        "proc": proc_names.get(pid, "pid%s" % pid),
        "name": thread_names.get((pid, tid), "tid%s" % tid),
        "ev": [[round(e["ts"], 3), round(e.get("dur", 0), 3), e["name"], e.get("cat", "")] for e in lanes[(pid, tid)]],
    })

xev = [e for e in events if e.get("ph") == "X"]
tmin = min(e["ts"] for e in xev)
tmax = max(e["ts"] + e.get("dur", 0) for e in xev)

# flow 配对
flows, tmp = [], {}
for e in events:
    if e.get("ph") in ("s", "f"):
        tmp.setdefault(e["id"], {})[e["ph"]] = e
for fid, pair in tmp.items():
    s, f = pair.get("s"), pair.get("f")
    if not s or not f:
        continue
    flows.append([round(s["ts"], 3), s["pid"], s["tid"], round(f["ts"], 3), f["pid"], f["tid"],
                  s.get("name", ""), s.get("bind_id")])

counters = [[round(e["ts"], 3), e["pid"], e["tid"], e["name"], e["args"]] for e in events if e.get("ph") == "C"]

# 统计
from collections import defaultdict, Counter
agg = defaultdict(lambda: [0.0, 0, 0.0])
for e in xev:
    a = agg[e["name"]]
    a[0] += e.get("dur", 0); a[1] += 1; a[2] = max(a[2], e.get("dur", 0))
stats = sorted(([k, round(v[0], 2), v[1], round(v[2], 2)] for k, v in agg.items()), key=lambda r: -r[1])

TRACE = {
    "file": "merged_swimlane_20261015_232514.json",
    "meta": merged.get("metadata", {}),
    "tmin": round(tmin, 3), "tmax": round(tmax, 3),
    "procs": proc_names,
    "lanes": lane_list,
    "flows": flows,
    "counters": counters,
    "stats": stats[:40],
    "nEvents": len(events),
    "phCount": dict(Counter(e.get("ph") for e in events)),
}

# ---------- chip_swimlane_records ----------
cores, ct = [], chip["metadata"]["core_types"]
for r in chip["aicore_tasks"]:
    cores.append({"core": r[0], "task": r[1], "t0": r[3], "t1": r[4], "dur": r[5], "type": ct[r[0]] if r[0] < len(ct) else "?"})

CHIP = {
    "file": "chip_swimlane_records.json",
    "level": chip.get("chip_swimlane_level"),
    "meta": chip["metadata"],
    "cores": cores,
    "sched_records": chip["scheduler_records"]["streams"][0]["records"],
    "sched_metrics": chip["scheduler_records"]["streams"][0].get("metrics", []),
    "sched_tasks": chip["scheduler_tasks"].get('producer "aicpu"', []) if isinstance(chip["scheduler_tasks"], dict) else [],
    "orch": chip["aicpu_orchestrator_phases"][0],
}

# ---------- deps ----------
DEPS = {"file": "deps.json", "runtime": deps.get("runtime"), "tasks": deps.get("tasks", [])}
NAMEMAP = {"file": "name_map__*.json", "data": namemap}

# ---------- 核内（incore）静态结构 ----------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _extract_incore import extract as extract_incore
from _extract_profile import extract as extract_profile
from _extract_inputs import extract as extract_inputs
INCORE = extract_incore(os.path.dirname(SRC.rstrip("/")))
PROFILE = extract_profile(os.path.join(os.path.dirname(SRC.rstrip("/")),
                                       "qwen3_14b_paged_attention_profile_20261016"))
INPUTS = extract_inputs(os.path.dirname(SRC.rstrip("/")))

data_js = (
    "const TRACE=" + json.dumps(TRACE, ensure_ascii=False, separators=(",", ":")) + ";\n"
    "const CHIP=" + json.dumps(CHIP, ensure_ascii=False, separators=(",", ":")) + ";\n"
    "const DEPS=" + json.dumps(DEPS, ensure_ascii=False, separators=(",", ":")) + ";\n"
    "const NAMEMAP=" + json.dumps(NAMEMAP, ensure_ascii=False, separators=(",", ":")) + ";\n"
    "const INCORE=" + json.dumps(INCORE, ensure_ascii=False, separators=(",", ":")) + ";\n"
    "const PROFILE=" + json.dumps(PROFILE, ensure_ascii=False, separators=(",", ":")) + ";\n"
    "const INPUTS=" + json.dumps(INPUTS, ensure_ascii=False, separators=(",", ":")) + ";\n"
    "const RAW_MERGED=" + json.dumps(merged, ensure_ascii=False, separators=(",", ":")) + ";\n"
)

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DFX Trace Viewer · paged attention 编译产物</title>
<style>
:root{
  --bg:#1b1d21; --bg2:#232629; --bg3:#2b2f34; --panel:#202327;
  --line:#34383e; --line2:#2a2e33;
  --ink:#e6e8ea; --ink2:#a8aeb6; --ink3:#767d87;
  --sel:#ffd24a;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:13px;overflow:hidden}
#app{display:flex;flex-direction:column;height:100vh}

/* ---- top bar ---- */
.top{background:var(--bg2);border-bottom:1px solid var(--line);flex:0 0 auto}
.top .r1{display:flex;align-items:center;gap:14px;padding:9px 14px;flex-wrap:wrap}
.brand{font-weight:650;font-size:13.5px;letter-spacing:.01em;white-space:nowrap}
.brand span{color:var(--ink3);font-weight:400;font-size:11.5px;font-family:var(--mono)}
.tabs{display:flex;gap:2px;margin-left:6px}
.tab{background:transparent;border:1px solid transparent;color:var(--ink2);padding:5px 12px;border-radius:6px;cursor:pointer;font-size:12.5px;font-family:inherit}
.tab:hover{background:var(--bg3);color:var(--ink)}
.tab.on{background:#3a4149;border-color:#4a525c;color:#fff;font-weight:600}
.spacer{flex:1}
.stat{font-family:var(--mono);font-size:11.5px;color:var(--ink2);white-space:nowrap}
.stat b{color:var(--ink);font-weight:600}
.grow{display:flex;align-items:center;gap:8px;flex:0 0 auto}
.r2{display:flex;align-items:center;gap:10px;padding:7px 14px;border-top:1px solid var(--line2);flex-wrap:wrap}
button.btn,.chk{background:var(--bg3);border:1px solid #41474f;color:var(--ink2);border-radius:5px;padding:4px 9px;cursor:pointer;font-size:11.5px;font-family:var(--mono)}
button.btn:hover,.chk:hover{background:#3a4048;color:var(--ink)}
button.btn.on{background:#4a5563;color:#fff;border-color:#5d6878}
.chk{display:inline-flex;align-items:center;gap:5px;user-select:none}
.chk input{margin:0;accent-color:#5b8dd6}
input.search{background:#191b1e;border:1px solid #41474f;color:var(--ink);border-radius:5px;padding:4px 9px;font-size:11.5px;font-family:var(--mono);width:170px}
input.search::placeholder{color:#5d646d}
input[type=range]{width:130px;accent-color:#5b8dd6}
.sep{width:1px;height:16px;background:var(--line);margin:0 2px}
.hint{font-family:var(--mono);font-size:11px;color:var(--ink3)}

/* ---- panels ---- */
.pane{flex:1;min-height:0;display:none;flex-direction:column}
.pane.on{display:flex}

/* ---- trace viewer ---- */
.tl{flex:1;min-height:0;display:flex;flex-direction:column;position:relative}
.tlwrap{flex:1;min-height:0;overflow:auto;position:relative;background:#191b1e}
.tlinner{position:relative}
.ruler{position:sticky;top:0;z-index:20;height:26px;background:var(--bg2);border-bottom:1px solid var(--line);display:block}
.ruler .tick{position:absolute;top:0;height:26px;border-left:1px solid #3b4046;color:var(--ink3);font-family:var(--mono);font-size:10px;padding-left:3px;line-height:26px;white-space:nowrap}
.ruler .tick.min{border-left-color:#2a2e33}
.grp{position:sticky;top:26px;z-index:18;background:#2a3038;border-bottom:1px solid var(--line);font-family:var(--mono);height:21px;line-height:21px}
.glab{position:sticky;left:0;display:inline-block;width:224px;box-sizing:border-box;padding:0 8px;height:21px;
      background:#2a3038;color:#c6ccd4;font-size:11px;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
      border-right:1px solid var(--line)}
.row{display:flex;height:19px;border-bottom:1px solid #212429;position:relative}
.row:hover .lab{background:#2e343b}
.lab{position:sticky;left:0;z-index:16;flex:0 0 224px;width:224px;background:#202327;border-right:1px solid var(--line);
     font-family:var(--mono);font-size:10.5px;color:var(--ink2);line-height:19px;padding:0 4px 0 8px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.lab .n{color:#545b64;float:right;font-size:9.5px}
.lab.aic{color:#8fb4ee}.lab.aiv{color:#c1a2f0}
.track{position:relative;flex:0 0 auto;height:19px}
.sl{position:absolute;top:1px;height:17px;border-radius:2px;overflow:hidden;cursor:pointer;
    font-family:var(--mono);font-size:10px;color:#0d0f11;line-height:17px;padding:0 3px;white-space:nowrap;
    border:1px solid rgba(0,0,0,.35)}
.sl:hover{outline:1px solid #fff;outline-offset:-1px;z-index:8}
.sl.sel{outline:2px solid var(--sel);outline-offset:-2px;z-index:9}
.sl.dim{opacity:.22}
.flowsvg{position:absolute;left:0;top:0;pointer-events:none;z-index:12;overflow:visible}
.flowsvg path{fill:none;stroke-width:1;opacity:.55}
.flowsvg path.hl{stroke-width:1.8;opacity:1}
.cursor{position:absolute;top:0;width:1px;background:#ff5c5c;z-index:14;pointer-events:none;display:none}

/* ---- detail ---- */
.det{flex:0 0 auto;max-height:190px;overflow:auto;background:var(--bg2);border-top:1px solid var(--line);padding:9px 14px;font-size:12px}
.det .ttl{font-family:var(--mono);font-size:12.5px;font-weight:600;margin-bottom:5px}
.det .kv{display:grid;grid-template-columns:auto 1fr;gap:1px 12px;font-family:var(--mono);font-size:11.5px;color:var(--ink2)}
.det .kv b{color:var(--ink);font-weight:600}
.det pre{margin:5px 0 0;background:#191b1e;border:1px solid var(--line2);border-radius:5px;padding:7px 9px;
         font-family:var(--mono);font-size:11px;color:#9fc08c;max-height:80px;overflow:auto}
.det .empty{color:var(--ink3);font-style:italic;font-family:var(--mono);font-size:11.5px}

/* ---- generic content panes ---- */
.scroll{flex:1;min-height:0;overflow:auto;padding:16px 18px 40px}
.h2{font-size:14.5px;font-weight:650;margin:0 0 3px}
.p{color:var(--ink2);font-size:12.5px;margin:0 0 12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:13px 15px;margin:0 0 14px}
table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:11.5px}
th{text-align:left;color:var(--ink3);font-weight:600;font-size:10px;letter-spacing:.05em;text-transform:uppercase;
   padding:5px 7px;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:4px 7px;border-bottom:1px solid var(--line2);color:var(--ink2);vertical-align:top}
tr:last-child td{border-bottom:none}
tbody tr:hover td{background:#262a2f}
.n{color:var(--ink)}
.pill{display:inline-block;font-size:10px;font-weight:600;padding:1px 6px;border-radius:9px;font-family:var(--mono)}
.pi{background:#1d2f4d;color:#8fb4ee}.pv{background:#302348;color:#c1a2f0}
.pg{background:#2b3a2c;color:#95c99a}.pa{background:#3d3218;color:#e0bd76}.pr{background:#42242a;color:#e79a9a}
.px{background:#2c3037;color:#8b939d}
.bar{height:9px;background:#2b2f34;border-radius:2px;overflow:hidden;min-width:56px}
.bar>i{display:block;height:100%;border-radius:2px}
.h3{font-size:12px;font-weight:650;color:var(--ink);margin:12px 0 8px;letter-spacing:.01em}
.ib{display:grid;grid-template-columns:152px 1fr 44px;gap:8px;align-items:center;margin:0 0 5px;
    font-family:var(--mono);font-size:10.5px}
.ibn{color:var(--ink2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ibb{height:9px;background:#2b2f34;border-radius:2px;overflow:hidden}
.ibb>i{display:block;height:100%;border-radius:2px}
.ibv{color:var(--ink3);text-align:right}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:900px){.grid2{grid-template-columns:1fr}}
.note{background:#3a331c;border-left:3px solid #d1a53c;padding:8px 12px;border-radius:0 6px 6px 0;font-size:12px;color:#e8dcb8;margin:10px 0}
.note.bad{background:#3a2323;border-left-color:#c05656;color:#f0cccc}
.legend{display:flex;gap:13px;flex-wrap:wrap;font-family:var(--mono);font-size:10.5px;color:var(--ink2);margin-top:8px}
.legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px;vertical-align:-1px}
code{font-family:var(--mono);background:#2b2f34;border:1px solid var(--line2);border-radius:3px;padding:0 4px;color:#c8ced6}
</style>
</head>
<body>
<div id="app">
  <div class="top">
    <div class="r1">
      <div class="brand">DFX Trace Viewer <span>· paged attention 20261015_232459_qw2ym_86</span></div>
      <div class="tabs">
        <button class="tab on" data-p="trace">时间线 (Chrome Tracing)</button>
        <button class="tab" data-p="core">核负载</button>
        <button class="tab" data-p="incore">核内</button>
        <button class="tab" data-p="pmu">PMU 与仿真</button>
        <button class="tab" data-p="task">任务与依赖</button>
        <button class="tab" data-p="inputs">输入定义</button>
      </div>
      <div class="spacer"></div>
      <div class="stat" id="topstat"></div>
    </div>
    <div class="r2" id="toolbar">
      <button class="btn" id="zfit">适配宽度</button>
      <button class="btn" id="zin">＋</button>
      <button class="btn" id="zout">－</button>
      <input type="range" id="zoom" min="0" max="100" value="0">
      <span class="hint" id="zlabel"></span>
      <div class="sep"></div>
      <input class="search" id="q" placeholder="过滤事件名 / 过滤 slice…">
      <div class="sep"></div>
      <label class="chk"><input type="checkbox" id="showflow" checked>flow 箭头</label>
      <label class="chk"><input type="checkbox" id="showgrp" checked>只显示非空 lane</label>
      <div class="sep"></div>
      <span class="hint">滚轮=纵向滚动 · Ctrl+滚轮=缩放 · 拖拽=平移 · 单击 slice=详情</span>
      <div class="spacer"></div>
      <button class="btn" id="dl">下载原始 trace JSON</button>
    </div>
  </div>

  <!-- ================= TRACE ================= -->
  <div class="pane on" id="pane-trace">
    <div class="tl">
      <div class="tlwrap" id="tlwrap">
        <div class="tlinner" id="tlinner"></div>
        <div class="cursor" id="cursor"></div>
      </div>
      <div class="det" id="detail">
        <div class="ttl" style="color:#a8aeb6;font-weight:600">未选中事件 · 点击任意 slice 查看详情</div>
        <div class="legend">
          <span><i style="background:#4aa3d8"></i>flow: dispatch (144)</span>
          <span><i style="background:#54b57a"></i>flow: submit→dispatch (72)</span>
          <span><i style="background:#c9a13a"></i>flow: dependency (18)</span>
          <span><i style="background:#c06070"></i>flow: complete (4)</span>
          <span style="color:#767d87">|</span>
          <span><i style="background:#4a7ebb"></i>attn_swpipe_aic_spmd</span>
          <span><i style="background:#8a63c9"></i>attn_swpipe_aiv_spmd</span>
          <span><i style="background:#5a6068"></i>dummy</span>
          <span><i style="background:#3f8f5a"></i>submit</span>
          <span><i style="background:#8f5f1f"></i>drain</span>
          <span><i style="background:#2e7fa8"></i>dispatch</span>
          <span><i style="background:#a8424e"></i>complete</span>
        </div>
      </div>    </div>
  </div>

  <!-- ================= CORE ================= -->
  <div class="pane" id="pane-core"><div class="scroll" id="corebody"></div></div>

  <!-- ================= INCORE ================= -->
  <div class="pane" id="pane-incore"><div class="scroll" id="incorebody"></div></div>
  <div class="pane" id="pane-pmu"><div class="scroll" id="pmubody"></div></div>

  <!-- ================= TASK ================= -->
  <div class="pane" id="pane-task"><div class="scroll" id="taskbody"></div></div>

  <!-- ================= INPUTS ================= -->
  <div class="pane" id="pane-inputs"><div class="scroll" id="inputsbody"></div></div>
</div>

<script>
__DATA__
</script>
<script>
/* ============ 工具 ============ */
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmt=(v,n=2)=>Number(v).toFixed(n);

/* 颜色：语义优先，其余按名字哈希 */
const SEM=[
  [/^attn_swpipe_aic_spmd/,      '#4a7ebb', '#cfe0f5'],
  [/^attn_swpipe_aiv_spmd/,      '#8a63c9', '#e4d9f7'],
  [/^dummy/,                     '#5a6068', '#d2d6db'],
  [/^alloc/,                     '#2f8f8f', '#c9e9e9'],
  [/^submit/,                    '#3f8f5a', '#cfe8d8'],
  [/^drain_prepare/,             '#b0762a', '#f0dcbb'],
  [/^drain_publish/,             '#c08a30', '#f2e2c4'],
  [/^drain/,                     '#8f5f1f', '#e8d3b0'],
  [/^dispatch/,                  '#2e7fa8', '#c8e2ef'],
  [/^complete/,                  '#a8424e', '#f0cdd2'],
  [/^release/,                   '#7a5560', '#e0ced4'],
  [/^resolve/,                   '#6d6fa8', '#d6d7ee'],
];
const PAL=['#4a7ebb','#8a63c9','#3f8f5a','#b0762a','#2e7fa8','#a8424e','#5a9e8f','#8f6b2f','#6d6fa8','#3d8b7d','#a06a3a','#557ba8'];
function colorOf(name){
  for(const [re,c] of SEM) if(re.test(name)) return c;
  let h=0; for(let i=0;i<name.length;i++) h=(h*31+name.charCodeAt(i))>>>0;
  return PAL[h%PAL.length];
}
const LANE_H=19, LAB_W=224, RULER_H=26, GRP_H=21;

/* ============ 状态 ============ */
let pxPerUs = 6, tMax = TRACE.tmax, tMin = 0;
let visibleLanes = [], laneIndex = new Map();
let selected = null;
const ZMIN=0.6, ZMAX=400;

/* ============ 布局 ============ */
function buildLanes(){
  const q = ($('#q').value||'').trim().toLowerCase();
  const onlyNonEmpty = $('#showgrp').checked;
  let out = TRACE.lanes.filter(l => l.ev.length>0 || !onlyNonEmpty);
  if(q){
    out = out.filter(l => l.name.toLowerCase().includes(q) ||
        l.ev.some(e=>e[2].toLowerCase().includes(q)));
  }
  visibleLanes = out;
  laneIndex = new Map();
  out.forEach((l,i)=>laneIndex.set(l.pid+':'+l.tid, i));
  /* 同进程内重名 lane 用 tid 消歧（如两个 Sched_2） */
  const cnt={};
  out.forEach(l=>{ const k=l.pid+'|'+l.name; cnt[k]=(cnt[k]||0)+1; });
  out.forEach(l=>{ l.label = l.name + (cnt[l.pid+'|'+l.name]>1 ? ' @'+l.tid : ''); });
}

function render(){
  const wrap=$('#tlwrap');
  const prevScroll=wrap.scrollLeft, prevW=wrap.clientWidth;
  const anchorT = (prevScroll + prevW/2)/pxPerUs;

  buildLanes();
  tMin = TRACE.tmin; tMax = TRACE.tmax;

  const W = Math.max(200, (tMax - tMin) * pxPerUs);
  const inner=$('#tlinner');

  const q=($('#q').value||'').trim().toLowerCase();
  const showFlow=$('#showflow').checked;

  /* --- 分组 + 行 --- */
  let h = '<div class="ruler" style="width:'+(LAB_W+W)+'px">';
  const step = tickStep(pxPerUs);
  for(let t=0;t<=tMax+1e-9;t+=step){
    const x = LAB_W + (t-tMin)*pxPerUs;
    if(x>LAB_W+W) break;
    h += '<div class="tick'+(t===0?' min':'')+'" style="left:'+x.toFixed(1)+'px">'+trimNum(t)+'</div>';
  }
  h += '</div>';

  let curProc=null;
  visibleLanes.forEach((l,i)=>{
    if(l.proc!==curProc){
      curProc=l.proc;
      const n=visibleLanes.filter(x=>x.proc===curProc).length;
      h += '<div class="grp" style="width:'+(LAB_W+W)+'px"><span class="glab">'+esc(curProc)+' · '+n+' lane</span></div>';
    }
    const cls = l.name.startsWith('AIC')?'aic' : l.name.startsWith('AIV')?'aiv' : '';
    h += '<div class="row" data-i="'+i+'" style="width:'+(LAB_W+W)+'px">';
    h += '<div class="lab '+cls+'" title="'+esc(l.label)+'  (pid '+l.pid+', tid '+l.tid+')">'+esc(l.label)+'<span class="n">'+l.ev.length+'</span></div>';
    h += '<div class="track" style="width:'+W+'px" data-lane="'+i+'">';
    l.ev.forEach((e,ei)=>{
      const name=e[2], x=(e[0]-tMin)*pxPerUs, w=Math.max(e[1]*pxPerUs,1.2);
      const dim = q && !name.toLowerCase().includes(q) && !l.name.toLowerCase().includes(q);
      const sel = selected && selected.pid===l.pid && selected.tid===l.tid && selected.idx===ei;
      h += '<div class="sl'+(sel?' sel':'')+(dim?' dim':'')+'" style="left:'+x.toFixed(2)+'px;width:'+w.toFixed(2)+
           'px;background:'+colorOf(name)+'" data-i="'+i+'" data-e="'+ei+'" title="'+esc(name)+'  '+fmt(e[1],2)+'µs">'+
           esc(name)+'</div>';
    });
    h += '</div></div>';
  });

  /* --- flow arrows --- */
  const H = 0;
  if(showFlow){
    /* 选中 slice 的时间区间 —— flow 端点落在区间内即视为关联 */
    const sA = selected ? selected.ts - 0.05 : null;
    const sB = selected ? selected.ts + selected.dur + 0.05 : null;
    const onSel = (pid,tid)=> selected && selected.pid===pid && selected.tid===tid;
    let paths='';
    TRACE.flows.forEach((f,fi)=>{
      const i0=laneIndex.get(f[1]+':'+f[2]); if(i0===undefined) return;
      const i1=laneIndex.get(f[4]+':'+f[5]); if(i1===undefined) return;
      const y0 = laneY(i0)+9.5, y1 = laneY(i1)+9.5;
      const x0 = LAB_W+(f[0]-tMin)*pxPerUs, x1 = LAB_W+(f[3]-tMin)*pxPerUs;
      const touching = selected && (
        (onSel(f[1],f[2]) && f[0]>=sA && f[0]<=sB) ||
        (onSel(f[4],f[5]) && f[3]>=sA && f[3]<=sB));
      const other = selected && !touching;
      const col = COLOR_FLOW(f[6]);
      const mx=(x0+x1)/2;
      paths += '<path class="'+(touching?'hl':'')+'" d="M'+x0.toFixed(1)+','+y0.toFixed(1)+
               ' C'+mx.toFixed(1)+','+y0.toFixed(1)+' '+mx.toFixed(1)+','+y1.toFixed(1)+' '+x1.toFixed(1)+','+y1.toFixed(1)+'" stroke="'+col+'"'+
               (other?' style="opacity:.07"':'')+' data-f="'+fi+'"/>';
    });
    h += '<svg class="flowsvg" width="'+(LAB_W+W)+'" height="'+Math.max(1,totalH())+'">'+paths+'</svg>';
  }

  inner.innerHTML = h;

  /* 恢复滚动锚点 */
  const newScroll = anchorT*pxPerUs - prevW/2;
  wrap.scrollLeft = Math.max(0, newScroll);
  $('#zlabel').textContent = fmt(pxPerUs,1)+' px/µs   ·   可视 '+fmt(wrap.clientWidth/pxPerUs,1)+' µs';
  $('#zoom').value = Math.round(100*(Math.log(pxPerUs)-Math.log(ZMIN))/(Math.log(ZMAX)-Math.log(ZMIN)));
  bindSlices();
  updateTopStat();
}
function laneY(i){ return RULER_H + GRP_H*(countGroupsBefore(i)+1) + LANE_H*i + i*0; }
function countGroupsBefore(i){
  let g=0, prev=null;
  for(let k=0;k<=i;k++){ if(visibleLanes[k].proc!==prev){g++; prev=visibleLanes[k].proc;} }
  return g;
}
function totalH(){
  let g=0, prev=null;
  visibleLanes.forEach(l=>{ if(l.proc!==prev){g++; prev=l.proc;} });
  return RULER_H + GRP_H*g + LANE_H*visibleLanes.length;
}
function COLOR_FLOW(n){ return n==='dispatch'?'#4aa3d8' : n==='dependency'?'#c9a13a' : n==='submit→dispatch'?'#54b57a' : '#c06070'; }

function tickStep(px){
  const want=70/px;
  const s=[0.1,0.2,0.5,1,2,5,10,20,50,100];
  for(const v of s) if(v>=want) return v;
  return 200;
}
const trimNum=v=>Number.isInteger(v)?String(v):v.toFixed(1);

/* ============ 交互 ============ */
function bindSlices(){
  $$('.sl').forEach(el=>{
    el.onclick=ev=>{
      ev.stopPropagation();
      const li=+el.dataset.i, ei=+el.dataset.e, l=visibleLanes[li], e=l.ev[ei];
      selected={pid:l.pid,tid:l.tid,idx:ei,ts:e[0],dur:e[1]};
      showDetail(l,e);
      render();          /* 重渲染以应用 .sel 与 flow 高亮 */
    };
  });
}
function showDetail(l,e){
  const [ts,dur,name,cat]=e;
  const A=ts-0.05, B=ts+dur+0.05;
  let extra=[];
  TRACE.flows.forEach((f,i)=>{
    if(f[1]===l.pid&&f[2]===l.tid&&f[0]>=A&&f[0]<=B)
      extra.push({dir:'out',id:i,kind:f[6],to:f[4]+':'+f[5],t:f[3]});
    if(f[4]===l.pid&&f[5]===l.tid&&f[3]>=A&&f[3]<=B)
      extra.push({dir:'in',id:i,kind:f[6],from:f[1]+':'+f[2],t:f[0]});
  });
  const laneName=l.label||l.name, proc=l.proc;
  let h='<div class="ttl">'+esc(name)+'</div><div class="kv">';
  h+='<span>lane</span><b>'+esc(proc)+' / '+esc(laneName)+'  (pid '+l.pid+', tid '+l.tid+')</b>';
  h+='<span>ts</span><b>'+fmt(ts,3)+' µs</b>';
  h+='<span>dur</span><b>'+fmt(dur,3)+' µs</b>';
  h+='<span>end</span><b>'+fmt(ts+dur,3)+' µs</b>';
  h+='<span>cat</span><b>'+esc(cat||'-')+'</b>';
  h+='<span>ph</span><b>X (complete event)</b>';
  if(extra.length){
    h+='</div><pre>flow 关联 ('+extra.length+')：\n'+extra.slice(0,14).map(x=>
      '  '+x.dir+'  ['+x.kind+'] #'+x.id+'  '+(x.dir==='out'?('→ '+(x.to)):('← '+(x.from)))+'  t='+fmt(x.t,3)+'µs'
    ).join('\n')+(extra.length>14?'\n  …共 '+extra.length+' 条':'')+'</pre>';
  } else { h+='</div>'; }
  $('#detail').innerHTML=h;
}

/* 缩放 */
function setZoom(np, anchorRatio){
  const wrap=$('#tlwrap');
  const r = anchorRatio===undefined ? .5 : anchorRatio;
  const at = (wrap.scrollLeft + wrap.clientWidth*r)/pxPerUs;
  pxPerUs=Math.min(ZMAX,Math.max(ZMIN,np));
  render();
  wrap.scrollLeft = Math.max(0, at*pxPerUs - wrap.clientWidth*r);
}
$('#zfit').onclick=()=>{ const w=$('#tlwrap').clientWidth-LAB_W-6; setZoom(w/(tMax-tMin)); };
$('#zin').onclick=()=>setZoom(pxPerUs*1.6);
$('#zout').onclick=()=>setZoom(pxPerUs/1.6);
$('#zoom').oninput=e=>{
  const f=+e.target.value/100;
  setZoom(Math.exp(Math.log(ZMIN)+f*(Math.log(ZMAX)-Math.log(ZMIN))));
};
$('#tlwrap').addEventListener('wheel',e=>{
  if(e.ctrlKey||e.metaKey){
    e.preventDefault();
    const r=Math.min(1,Math.max(0,(e.clientX-$('#tlwrap').getBoundingClientRect().left)/$('#tlwrap').clientWidth));
    setZoom(pxPerUs*(e.deltaY<0?1.15:1/1.15), r);
  }
},{passive:false});
/* 拖拽平移 */
let drag=null;
$('#tlwrap').addEventListener('mousedown',e=>{
  if(e.target.closest('.lab')) return;
  drag={x:e.clientX,sl:$('#tlwrap').scrollLeft};
  $('#tlwrap').style.cursor='grabbing';
});
window.addEventListener('mousemove',e=>{
  if(drag){ $('#tlwrap').scrollLeft=drag.sl-(e.clientX-drag.x); return; }
  const w=$('#tlwrap'), r=w.getBoundingClientRect();
  if(e.clientY>r.top && e.clientY<r.bottom){
    const x=e.clientX-r.left+w.scrollLeft;
    if(x>LAB_W){ const t=(x-LAB_W)/pxPerUs; $('#cursor').style.display='block';
      $('#cursor').style.left=x+'px'; $('#cursor').style.height=totalH()+'px';
      $('#cursor').title=fmt(t,3)+' µs'; }
    else $('#cursor').style.display='none';
  }
});
window.addEventListener('mouseup',()=>{ drag=null; $('#tlwrap').style.cursor=''; });
$('#q').oninput=()=>render();
$('#showgrp').onchange=()=>render();
$('#showflow').onchange=()=>render();
$('#dl').onclick=()=>{
  const b=new Blob([JSON.stringify(RAW_MERGED)],{type:'application/json'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(b);
  a.download='merged_swimlane_20261015_232514.json'; a.click();
};
document.addEventListener('keydown',e=>{ if(e.key==='f'&&(e.metaKey||e.ctrlKey)){e.preventDefault();$('#zfit').click();} });

/* ============ 顶部统计 ============ */
function updateTopStat(){
  const ph=TRACE.phCount, parts=[];
  if(ph.X) parts.push(ph.X+' slice');
  if(ph.M) parts.push(ph.M+' metadata');
  if(TRACE.flows.length) parts.push(TRACE.flows.length+' flow');
  if(ph.C) parts.push(ph.C+' counter');
  $('#topstat').innerHTML='<b>'+TRACE.nEvents+'</b> 事件 (' + parts.join(' · ') + ') &nbsp;·&nbsp; '+
    '<b>'+TRACE.lanes.length+'</b> lane &nbsp;·&nbsp; '+
    '<b>'+Object.keys(TRACE.procs).length+'</b> 进程 &nbsp;·&nbsp; span <b>'+fmt(TRACE.tmax-TRACE.tmin,2)+
    '</b> µs <span style="color:#767d87;font-size:10.5px">(单位按 µs 解释)</span>';
}

/* ============ Tab 切换 ============ */
$$('.tab').forEach(t=>t.onclick=()=>{
  $$('.tab').forEach(x=>x.classList.remove('on')); t.classList.add('on');
  $$('.pane').forEach(p=>p.classList.remove('on'));
  $('#pane-'+t.dataset.p).classList.add('on');
  $('#toolbar').style.display = t.dataset.p==='trace' ? 'flex' : 'none';
  if(t.dataset.p==='trace') render();
});

/* ==================================================================
   核负载 pane
================================================================== */
function buildCore(){
  const cs=CHIP.cores, meta=CHIP.meta;
  const aic=cs.filter(c=>c.type==='aic'), aiv=cs.filter(c=>c.type==='aiv');
  const w0=Math.min(...cs.map(c=>c.t0)), w1=Math.max(...cs.map(c=>c.t1));
  const span=w1-w0;
  const dmin=Math.min(...cs.map(c=>c.dur)), dmax=Math.max(...cs.map(c=>c.dur));
  const avg=a=>a.reduce((s,x)=>s+x,0)/a.length;
  const ns=20; // 20ns/tick @50MHz

  let h='';
  h+='<div class="h2">核负载 · chip_swimlane_records.json</div>';
  h+='<p class="p">'+esc(CHIP.file)+' · swimlane level '+CHIP.level+' · runtime <code>'+esc(meta.runtime)+
     '</code> · clock_freq_hz <code>'+meta.clock_freq_hz+'</code>（1 tick = 20 ns）· num_cores <code>'+meta.num_cores+
     '</code>（24 AIC + 48 AIV）</p>';

  h+='<div class="grid2">';
  h+='<div class="card"><div class="h2">规模</div><table><tbody>';
  h+='<tr><td>aicore_tasks 记录数</td><td class="n">'+cs.length+'（= 核数，1 条/核）</td></tr>';
  h+='<tr><td>唯一 task_id</td><td class="n">'+[...new Set(cs.map(c=>c.task))].join(', ')+'</td></tr>';
  h+='<tr><td>窗口 [t0, t1]</td><td class="n">'+w0+' … '+w1+'　（'+span+' tick = '+fmt(span*ns/1000,2)+' µs）</td></tr>';
  h+='<tr><td>idx5「dur」范围</td><td class="n">'+dmin+' … '+dmax+' tick　（'+fmt(dmin*ns/1000,2)+' … '+fmt(dmax*ns/1000,2)+' µs）</td></tr>';
  h+='<tr><td>idx5 均值</td><td class="n">AIC '+fmt(avg(aic.map(c=>c.dur)),1)+' · AIV '+fmt(avg(aiv.map(c=>c.dur)),1)+' tick</td></tr>';
  h+='</tbody></table></div>';

  h+='<div class="card"><div class="h2">字段结构（7 元组）</div><table><tbody>';
  h+='<tr><td>[0] core_id</td><td class="n">0 … 71（0–23 AIC，24–71 AIV）</td></tr>';
  h+='<tr><td>[1] task_id</td><td class="n">'+cs[0].task+'</td></tr>';
  h+='<tr><td>[2] 未标注常量</td><td class="n">恒为 1</td></tr>';
  h+='<tr><td>[3] start</td><td class="n">核 cycles</td></tr>';
  h+='<tr><td>[4] end</td><td class="n">核 cycles</td></tr>';
  h+='<tr><td>[5] dur</td><td class="n">核 cycles，<b>≠ end − start</b></td></tr>';
  h+='<tr><td>[6] 未标注常量</td><td class="n">恒为 1</td></tr>';
  h+='</tbody></table></div>';
  h+='</div>';

  /* 每核占用条：用 [t0,t1] 归一化 */
  h+='<div class="card"><div class="h2">每核占用窗口（按 [3]→[4] 归一化）</div>';
  h+='<div class="legend"><span><i style="background:#4a7ebb"></i>AIC (24)</span><span><i style="background:#8a63c9"></i>AIV (48)</span>'+
     '<span>横轴：'+w0+' → '+w1+' cycles</span></div>';
  h+='<div style="margin-top:9px">';
  cs.forEach(c=>{
    const l=100*(c.t0-w0)/span, w=Math.max(0.8,100*(c.t1-c.t0)/span);
    const col=c.type==='aic'?'#4a7ebb':'#8a63c9';
    h+='<div style="display:flex;align-items:center;gap:7px;height:14px">'+
       '<span style="flex:0 0 46px;font-family:var(--mono);font-size:10px;color:#8b939d;text-align:right">'+
       (c.type==='aic'?'AIC_':'AIV_')+c.core+'</span>'+
       '<span style="flex:1;height:9px;background:#2b2f34;border-radius:2px;overflow:hidden;position:relative">'+
       '<i style="position:absolute;left:'+l.toFixed(2)+'%;width:'+w.toFixed(2)+'%;height:100%;background:'+col+';border-radius:2px"></i></span>'+
       '<span style="flex:0 0 92px;font-family:var(--mono);font-size:10px;color:#8b939d">'+
       (c.t1-c.t0)+'c / '+c.dur+'c</span></div>';
  });
  h+='</div></div>';

  /* dur 排序柱状 */
  const srt=[...cs].sort((a,b)=>b.dur-a.dur);
  h+='<div class="card"><div class="h2">每核 dur([5]) 排行</div><p class="p">这一列才是真正的「核上工作量」；可见 AIC/AIV 内部差异明显。</p>';
  h+='<div style="margin-top:6px">';
  srt.forEach(c=>{
    const w=100*c.dur/dmax;
    const col=c.type==='aic'?'#4a7ebb':'#8a63c9';
    h+='<div style="display:flex;align-items:center;gap:7px;height:13px">'+
       '<span style="flex:0 0 46px;font-family:var(--mono);font-size:10px;color:#8b939d;text-align:right">'+
       (c.type==='aic'?'AIC_':'AIV_')+c.core+'</span>'+
       '<span style="flex:1;height:8px;background:#2b2f34;border-radius:2px;overflow:hidden">'+
       '<i style="display:block;height:100%;width:'+w.toFixed(2)+'%;background:'+col+'"></i></span>'+
       '<span style="flex:0 0 84px;font-family:var(--mono);font-size:10px;color:#8b939d">'+c.dur+'c = '+fmt(c.dur*ns/1000,2)+'µs</span></div>';
  });
  h+='</div></div>';

  /* scheduler records */
  h+='<div class="card"><div class="h2">Scheduler 记录（scheduler_records）</div>';
  h+='<table><thead><tr><th>kind</th><th>loop_iter</th><th>start</th><th>end</th><th>跨度 (cycles)</th><th>≈ µs</th><th>tasks_processed</th></tr></thead><tbody>';
  CHIP.sched_records.forEach(r=>{
    h+='<tr><td class="n">'+esc(r.kind)+'</td><td>'+r.loop_iter+'</td><td>'+r.start_cycles+'</td><td>'+r.end_cycles+
       '</td><td>'+ (r.end_cycles-r.start_cycles) +'</td><td>'+fmt((r.end_cycles-r.start_cycles)*ns/1000,2)+
       '</td><td>'+r.tasks_processed+'</td></tr>';
  });
  h+='</tbody></table></div>';

  /* orchestrator phases */
  h+='<div class="card"><div class="h2">Orchestrator 提交阶段（aicpu_orchestrator_phases）</div>';
  h+='<table><thead><tr><th>submit_idx</th><th>task_id</th><th>start_cycles</th><th>end_cycles</th><th>跨度 (cycles)</th><th>≈ µs</th></tr></thead><tbody>';
  const o0=CHIP.orch[0].start_cycles;
  CHIP.orch.forEach(p=>{
    h+='<tr><td class="n">'+p.submit_idx+'</td><td>'+p.task_id+'</td><td>'+p.start_cycles+'</td><td>'+p.end_cycles+
       '</td><td>'+(p.end_cycles-p.start_cycles)+'</td><td>'+fmt((p.end_cycles-p.start_cycles)*ns/1000,2)+'</td></tr>';
  });
  h+='</tbody></table>';
  const relLast=CHIP.orch[CHIP.orch.length-1].start_cycles-o0;
  h+='<p class="p" style="margin-top:8px">首个 submit → 真实 kernel submit(#9) 的相对偏移 = <b>'+relLast+
     ' cycles = '+fmt(relLast*ns/1000,2)+' µs</b>，与时间线中「核首次开跑 t≈119.5 µs」量级吻合。</p>';
  h+='</div>';

  h+='<div class="note"><b>口径提醒：</b>本 pane 的 cycles→µs 换算是按文件的 <code>clock_freq_hz=50000000</code>（20 ns/tick）。'+
     '但 72 核的 [3]→[4] 并集仅 <b>'+span+' cycles = '+fmt(span*ns/1000,2)+' µs</b>，'+
     '而 merged_swimlane 里同一条 kernel 的 slice 跨度是 <b>52.98 µs</b>。两者相差约 '+
     fmt(52.98/(span*ns/1000),2)+'×：说明 aicore_tasks 的时基与 AICPU 侧不同（或 [3]/[4] 只覆盖了核上「执行段」而非全部驻留）。'+
     '这一点建议找 runtime 侧确认，本查看器不对其做归一。</div>';

  $('#corebody').innerHTML=h;
}

/* ==================================================================
   任务与依赖 pane
================================================================== */
function buildTask(){
  let h='';
  h+='<div class="h2">任务与依赖 · deps.json</div>';
  h+='<p class="p">'+esc(DEPS.file)+' · runtime <code>'+esc(DEPS.runtime)+'</code> · 共 <b>'+DEPS.tasks.length+
     '</b> 个 task（7 个 dummy phase-fence barrier + 1 个真实 MixedKernels）</p>';

  h+='<div class="card"><table><thead><tr><th>#</th><th>task_id</th><th>kernel_ids</th><th>block_num</th><th>early_dispatch</th><th>scope</th><th>args 数</th><th>方向摘要</th></tr></thead><tbody>';
  DEPS.tasks.forEach((t,i)=>{
    const kids=t.kernel_ids.join(',');
    const dummy=kids==='-1,-1,-1';
    const dirs={}; (t.args||[]).forEach(a=>dirs[a.type]=(dirs[a.type]||0)+1);
    const dirStr=Object.entries(dirs).map(([k,v])=>k+'×'+v).join(' ')||'—';
    h+='<tr><td class="n">'+i+'</td><td>'+t.task_id+'</td><td><span class="pill '+(dummy?'px':(kids==='0,1,1'?'pg':'pa'))+'">'+kids+'</span></td>'+
       '<td>'+t.block_num+'</td><td>'+(t.early_dispatch?'<span class="pill pa">true</span>':'false')+'</td>'+
       '<td>'+esc(t.scope)+'</td><td>'+((t.args||[]).length)+'</td><td>'+dirStr+'</td></tr>';
  });
  h+='</tbody></table></div>';

  /* 真实 kernel 的 args 表 */
  const real=DEPS.tasks.find(t=>t.kernel_ids.join(',')==='0,1,1');
  if(real){
    h+='<div class="card"><div class="h2">真实 kernel task '+real.task_id+' 的参数表（'+real.args.length+' 个）</div>';
    h+='<table><thead><tr><th>idx</th><th>方向</th><th>dtype</th><th>shape</th><th>strides</th><th>start_offset</th><th>tensor_id</th></tr></thead><tbody>';
    real.args.forEach(a=>{
      h+='<tr><td class="n">'+a.idx+'</td><td><span class="pill '+
         (a.type==='INPUT'?'pi':a.type==='INOUT'?'pa':'pg')+'">'+a.type+'</span></td><td>'+esc(a.dtype)+
         '</td><td>['+(a.shape||[]).join(', ')+']</td><td>['+(a.strides||[]).join(', ')+']</td>'+
         '<td>'+esc(a.start_offset)+'</td><td style="font-size:10px">'+esc(a.tensor_id)+'</td></tr>';
    });
    h+='</tbody></table></div>';
  }

  h+='<div class="card"><div class="h2">callable_id → kernel 名（name_map__*.json）</div>';
  h+='<table><thead><tr><th>callable_id</th><th>kernel 名</th><th>role</th></tr></thead><tbody>';
  Object.entries(NAMEMAP.data.callable_id_to_name).forEach(([k,v])=>{
    h+='<tr><td class="n">'+k+'</td><td>'+esc(v)+'</td><td>'+(v.endsWith('_aic')?'AIC (cube)':'AIV (vector)')+'</td></tr>';
  });
  h+='</tbody></table>';
  h+='<p class="p" style="margin-top:8px">level = '+NAMEMAP.data.level+' · orchestrator_name = '+esc(String(NAMEMAP.data.orchestrator_name))+'</p>';
  h+='</div>';

  /* 事件名时长排行 */
  h+='<div class="card"><div class="h2">事件名总时长排行（来自 merged_swimlane）</div>';
  h+='<p class="p">注意：同一条 kernel slice 在 pid 3 / pid 4 两个视图各记一份，所以 kernel 类名字的 total 是 2×；'+
     '这也解释了为什么 AIV 的 total 是 AIC 的 2 倍（48 核 × 2 视图 vs 24 核 × 2 视图）。</p>';
  h+='<table><thead><tr><th>name</th><th>total dur (µs)</th><th>count</th><th>max (µs)</th><th>占比</th></tr></thead><tbody>';
  const tot=TRACE.stats.reduce((s,r)=>s+r[1],0);
  TRACE.stats.forEach(r=>{
    h+='<tr><td class="n">'+esc(r[0])+'</td><td>'+fmt(r[1],2)+'</td><td>'+r[2]+'</td><td>'+fmt(r[3],2)+
       '</td><td><div class="bar"><i style="width:'+Math.min(100,100*r[1]/tot).toFixed(1)+'%;background:'+colorOf(r[0])+'"></i></div></td></tr>';
  });
  h+='</tbody></table></div>';

  $('#taskbody').innerHTML=h;
}

/* ==================================================================
   核内 pane —— trace 无核内数据，改从静态产物重建
================================================================== */
function pipeSvg(pairs, order, color, title){
  const W=660, x0=26, bw=118, gap=(W-2*x0-4*bw)/3, y=38, bh=34;
  const idx={}; order.forEach((n,i)=>idx[n]=i);
  let s='<svg viewBox="0 0 '+W+' 106" width="100%" style="max-width:'+W+'px;height:auto">';
  s+='<text x="0" y="14" font-family="ui-monospace,monospace" font-size="11" fill="#a8aeb6">'+esc(title)+'</text>';
  order.forEach((n,i)=>{
    const x=x0+i*(bw+gap);
    s+='<rect x="'+x+'" y="'+y+'" width="'+bw+'" height="'+bh+'" rx="5" fill="#232a33" stroke="'+color+'"/>';
    s+='<text x="'+(x+bw/2)+'" y="'+(y+21)+'" text-anchor="middle" font-family="ui-monospace,monospace" font-size="12" fill="#e6e8ea">'+esc(n)+'</text>';
  });
  const ar=(up,label,src,tgt)=>{
    const i=idx[src], j=idx[tgt]; if(i===undefined||j===undefined) return;
    const ya=up? y-9 : y+bh+13;
    const xa=x0+i*(bw+gap)+bw/2, xb=x0+j*(bw+gap)+bw/2;
    const x1=up? xb-3 : xa+3, x2=up? xa+3 : xb-3;
    s+='<line x1="'+xa+'" y1="'+ya+'" x2="'+xb+'" y2="'+ya+'" stroke="'+color+'" stroke-width="1" opacity="0.75"/>';
    s+='<polygon points="'+x1+','+ya+' '+x2+','+(ya-3)+' '+x2+','+(ya+3)+'" fill="'+color+'"/>';
    s+='<text x="'+((xa+xb)/2)+'" y="'+(up? ya-4 : ya+11)+'" text-anchor="middle" font-family="ui-monospace,monospace" font-size="10" fill="'+(up?'#9fc08c':'#8b939d')+'">'+esc(label)+'</text>';
  };
  for(let i=0;i<order.length-1;i++){
    const f='PIPE_'+order[i]+'->PIPE_'+order[i+1], b='PIPE_'+order[i+1]+'->PIPE_'+order[i];
    if(pairs[f]!==undefined) ar(true,  'set '+pairs[f]+'（数据就绪）', order[i], order[i+1]);
    if(pairs[b]!==undefined) ar(false, 'set '+pairs[b]+'（缓冲释放）', order[i], order[i+1]);
  }
  s+='</svg>';
  return s;
}

/* ---------- 小工具：SVG 文本 / 箭头 / 折线 ---------- */
function T(x,y,s,size,fill,w,anchor){
  return '<text x="'+x+'" y="'+y+'" font-family="ui-monospace,monospace" font-size="'+size+'" fill="'+fill+'"'+
         (w?' font-weight="'+w+'"':'')+(anchor?' text-anchor="'+anchor+'"':'')+'>'+esc(s)+'</text>';
}
function AR(x1,y1,x2,y2,col,label,dash){
  const a=Math.atan2(y2-y1,x2-x1), hl=7.5, hw=3.4;
  const bx=x2-hl*Math.cos(a), by=y2-hl*Math.sin(a);
  let s='<line x1="'+x1+'" y1="'+y1+'" x2="'+bx.toFixed(1)+'" y2="'+by.toFixed(1)+'" stroke="'+col+
        '" stroke-width="1.3"'+(dash?' stroke-dasharray="4 3"':'')+'/>';
  s+='<polygon points="'+x2+','+y2+' '+(bx-hw*Math.sin(a)).toFixed(1)+','+(by+hw*Math.cos(a)).toFixed(1)+
     ' '+(bx+hw*Math.sin(a)).toFixed(1)+','+(by-hw*Math.cos(a)).toFixed(1)+'" fill="'+col+'"/>';
  if(label) s+=T((x1+x2)/2,(y1+y2)/2-7,label,9,'#9aa1aa',null,'middle');
  return s;
}
function PL(pts,col,label,dash){   /* 折线 + 末段箭头 */
  let d=pts.map((p,i)=>(i?'L':'M')+p[0]+' '+p[1]).join(' ');
  let s='<path d="'+d+'" fill="none" stroke="'+col+'" stroke-width="1.3"'+(dash?' stroke-dasharray="4 3"':'')+'/>';
  const n=pts.length, [x1,y1]=pts[n-2], [x2,y2]=pts[n-1];
  const a=Math.atan2(y2-y1,x2-x1), hl=7.5, hw=3.4;
  const bx=x2-hl*Math.cos(a), by=y2-hl*Math.sin(a);
  s+='<polygon points="'+x2+','+y2+' '+(bx-hw*Math.sin(a)).toFixed(1)+','+(by+hw*Math.cos(a)).toFixed(1)+
     ' '+(bx+hw*Math.sin(a)).toFixed(1)+','+(by-hw*Math.cos(a)).toFixed(1)+'" fill="'+col+'"/>';
  if(label){ const mx=pts[1][0]+(pts[2][0]-pts[1][0])*0.5; s+=T(mx,pts[1][1]-6,label,9,'#9aa1aa',null,'middle'); }
  return s;
}
/* ---------- 内存空间方框（分层布局用）---------- */
const LOCCOL={mat:'#5b8dd6',left:'#8f7ae0',right:'#c05656',acc:'#4fae72',vec:'#a06ad0'};
const LOCLBL={mat:'L1 · mat',left:'L0A · left',right:'L0B · right',acc:'L0C · acc',vec:'UB · vec'};
const MEMCAP={mat:512*1024,left:64*1024,right:64*1024,acc:128*1024,vec:192*1024};
const MEMROLE={mat:'GM ↔ 核 之间的暂存层',left:'MMAD 左操作数',right:'MMAD 右操作数',acc:'MMAD 累加器',vec:'矢量统一缓冲'};

function MEMBOX(x,y,w,h,o){
  const pct=o.used/o.cap, iw=w-24;
  const ps=pct>=0.9995?'100':(pct>=0.1?(pct*100).toFixed(1):(pct*100).toFixed(2));
  let s='<rect x="'+x+'" y="'+y+'" width="'+w+'" height="'+h+'" rx="7" fill="#1f2328" stroke="'+o.color+
        '" stroke-width="'+(o.hot?2:1.3)+'"/>';
  s+=T(x+12,y+19,o.label,12,'#e6e8ea','650');
  s+=T(x+w-12,y+19,o.role,9.5,'#767d87',null,'end');
  s+=T(x+12,y+37,fmtB(o.used)+' / '+fmtB(o.cap),10.5,'#c8ced6');
  s+=T(x+w-12,y+37,ps+'% 已用',10.5,o.color,'600','end');
  s+='<rect x="'+(x+12)+'" y="'+(y+45)+'" width="'+iw+'" height="11" rx="3" fill="#2b2f34"/>';
  s+='<rect x="'+(x+12)+'" y="'+(y+45)+'" width="'+(iw*Math.min(1,pct)).toFixed(1)+'" height="11" rx="3" fill="'+o.color+'"/>';
  s+=T(x+12,y+72,o.detail,9.5,'#8b939d');
  return s;
}
function GMSTRIP(x,y,w,h,txt,col){
  return '<rect x="'+x+'" y="'+y+'" width="'+w+'" height="'+h+'" rx="6" fill="none" stroke="'+col+
         '" stroke-width="1.2" stroke-dasharray="5 3"/>'+T(x+12,y+h/2+3.5,txt,10,col,null);
}
function ARV(x,y1,y2,col,label,side){
  const hl=7, hw=3.2, bx=x, by=y2-hl;
  let s='<line x1="'+x+'" y1="'+y1+'" x2="'+bx+'" y2="'+by+'" stroke="'+col+'" stroke-width="1.3"/>';
  s+='<polygon points="'+x+','+y2+' '+(bx-hw)+','+by+' '+(bx+hw)+','+by+'" fill="'+col+'"/>';
  if(label) s+=T(x+(side==='l'?-7:7),(y1+y2)/2+3.5,label,9,'#9aa1aa',null,side==='l'?'end':'start');
  return s;
}
function ARH(x1,x2,y,col,label){
  const hl=7, hw=3.2, bx=x2-hl;
  let s='<line x1="'+x1+'" y1="'+y+'" x2="'+bx+'" y2="'+y+'" stroke="'+col+'" stroke-width="1.3"/>';
  s+='<polygon points="'+x2+','+y+' '+bx+','+(y-hw)+' '+bx+','+(y+hw)+'" fill="'+col+'"/>';
  if(label) s+=T((x1+x2)/2,y-7,label,9,'#9aa1aa',null,'middle');
  return s;
}

function memFlowSvg(pA,pV){
  const H=656;
  let s='<svg viewBox="0 0 680 '+H+'" width="100%" style="height:auto;max-width:680px" role="img" '+
        'aria-label="AIC / AIV 核内数据通路与缓冲占用">';
  const used=(p,l)=>p.loc_peak_bytes[l]||0, mx=(p,l)=>p.loc_tile_max_bytes[l]||0,
        cnt=(p,l)=>p.loc_alloc_count[l]||0;
  const BOX=(x,y,w,h,loc,p,hot)=>MEMBOX(x,y,w,h,{
      label:LOCLBL[loc], role:MEMROLE[loc], color:LOCCOL[loc], used:used(p,loc), cap:MEMCAP[loc],
      detail:'最大单块 '+fmtB(mx(p,loc))+'  ·  分配 '+cnt(p,loc)+' 次  ·  同一时刻 ≈'+
             (used(p,loc)/mx(p,loc)).toFixed(1)+' 块这么大',
      hot:!!hot});

  /* ================= AIC：三层 ================= */
  s+=T(10,15,'AIC · Cube 核 × 24 —— 矩阵乘通路',12.5,'#e6e8ea','650');
  s+=T(10,30,'数据流：GM →(MTE2)→ L1 →(MTE1)→ L0A / L0B →(M · MMAD)→ L0C →(FIX)→ GM',10,'#8b939d');
  s+=GMSTRIP(24,42,632,22,'GM / DDR + L2    ——    K/V page · Q · O','#78828e');
  s+=ARV(150,64,80,'#5b8dd6','MTE2 · 26 次 TLOAD');
  s+=BOX(24,80,632,78,'mat',pA);
  s+=ARV(178,158,180,'#7f8b9a','MTE1','l');
  s+=ARV(502,158,180,'#7f8b9a','MTE1','l');
  s+=BOX(24,180,308,78,'left',pA);
  s+=BOX(348,180,308,78,'right',pA,true);
  s+=ARV(178,258,284,'#4fae72');
  s+=ARV(502,258,284,'#4fae72');
  s+=T(340,276,'M · MMAD —— 15 次 TMATMUL / TMATMUL_ACC',9.5,'#4fae72',null,'middle');
  s+=BOX(168,284,344,78,'acc',pA);
  s+=ARV(340,362,390,'#c05656','FIX + MTE3 · 写回 GM');
  s+=GMSTRIP(24,390,632,22,'GM / DDR + L2    ——    写回 attention 输出','#a07575');
  s+=T(10,434,'同一层 L1 喂两个操作数缓冲：L0B 打满 100%，L0A 只用 12.5% —— 左右操作数切分粒度差 8 倍。',9.5,'#e0bd76');

  /* ================= AIV：单缓冲 ================= */
  s+=T(10,468,'AIV · Vector 核 × 48 —— RMSNorm / RoPE / softmax 通路',12.5,'#e6e8ea','650');
  s+=T(10,483,'数据流：GM →(MTE2)→ UB →(V)→ UB →(MTE3)→ GM　·　单一 192 KB UB 承载全部中间量',10,'#8b939d');
  s+=GMSTRIP(24,496,632,22,'GM / DDR + L2    ——    Q / K / V · RoPE 表 · 输出','#78828e');
  s+=ARV(150,518,536,'#a06ad0','MTE2 ×20');
  s+=ARV(240,536,518,'#c05656','MTE3 · 写回');
  s+=BOX(24,536,352,78,'vec',pV);
  s+='<rect x="440" y="536" width="216" height="78" rx="7" fill="#2b2318" stroke="#e0bd76" stroke-width="1.2" stroke-dasharray="5 3"/>';
  s+=T(452,556,'Vector 单元',10.5,'#e0bd76','650');
  s+=T(452,572,'纯计算 · 不占缓冲',9.5,'#c8a86a');
  s+=T(452,590,'24 类向量算子',9.5,'#8b939d');
  s+=T(452,605,'TCOLEXPANDMUL ×20',9.5,'#8b939d');
  s+=ARH(378,438,556,'#a06ad0','V 读');
  s+=ARH(438,378,596,'#8b939d','写回 UB');
  s+=T(10,634,'212 次 alloc_tile 全挤在同一个 UB 里；20 种 tile 形状中 7 种 ≤ 512 B，却占了 67% 的分配次数。',9.5,'#e0bd76');
  s+='</svg>';
  return s;
}

function fmtB(b){
  if(b==null) return '—';
  if(b>=1024*1024) return (b/1048576).toFixed(2)+' MB';
  if(b>=1024) return (b/1024).toFixed(b>=10240?0:1)+' KB';
  return b+' B';
}

/* ---------- 单个 tile 尺寸谱（边长 ∝ √字节）---------- */
function mosaicSvg(){
  const pA=INCORE.pto.attn_swpipe_aic, pV=INCORE.pto.attn_swpipe_aiv;
  const flat=(p)=>{const o=[];Object.entries(p.loc_shapes).forEach(([loc,lst])=>lst.forEach(s=>o.push(Object.assign({loc:loc},s))));
                   return o.sort((a,b)=>b.bytes-a.bytes);};
  const a=flat(pA), v=flat(pV);
  const MAXS=62, LH=11, BH=30;
  /* 全局统一比例：两组共用同一把尺子（最大 32 KB → MAXS 像素） */
  const GLOBAL_TOP=Math.max.apply(null,a.concat(v).map(z=>z.bytes));
  const group=(items,color,perRow,colW,y0)=>{
    let s='', y=y0;
    const top=GLOBAL_TOP;
    for(let r=0;r<Math.ceil(items.length/perRow);r++){
      const row=items.slice(r*perRow,(r+1)*perRow);
      const x0=(680-row.length*colW)/2;
      const base=y+LH+MAXS;
      /* 基线：让 2 px 的小方块也能看到落在轴上 */
      s+='<line x1="'+(x0+6)+'" y1="'+base+'" x2="'+(x0+row.length*colW-6)+'" y2="'+base+
         '" stroke="#3a4048" stroke-width="1"/>';
      row.forEach((it,i)=>{
        const side=Math.max(1.6,MAXS*Math.sqrt(it.bytes/top));
        const cx=x0+i*colW+colW/2;
        s+='<rect x="'+(cx-side/2).toFixed(1)+'" y="'+(base-side).toFixed(1)+'" width="'+side.toFixed(1)+
           '" height="'+side.toFixed(1)+'" rx="'+(side>10?2:(side>4?1:0))+'" fill="'+color+'" fill-opacity="0.82" stroke="'+color+'"/>';
        s+=T(cx,base-side-5,fmtB(it.bytes),9,'#c8ced6',null,'middle');
        s+=T(cx,base+13,it.rows+'×'+it.cols+' '+it.dtype,8.5,'#8b939d',null,'middle');
        s+=T(cx,base+25,'×'+it.count,8.5,color,'600','middle');
      });
      y+=LH+MAXS+BH;
    }
    return s;
  };
  let out='<svg viewBox="0 0 680 386" width="100%" style="height:auto;max-width:680px" role="img" aria-label="tile 尺寸谱">';
  out+=T(8,13,'AIC（24 核）· 5 种形状 / 68 次分配 · 单块最大 32 KB',11,'#e6e8ea','650');
  out+=group(a,'#5b8dd6',5,112,24);
  out+=T(8,148,'AIV（48 核）· 20 种形状 / 212 次分配 · 单块最大 16 KB',11,'#e6e8ea','650');
  out+=group(v,'#a06ad0',10,60,159);
  out+='</svg>';
  return out;
}

function buildIncore(){
  const I=INCORE;
  const A=I.cpp.aic.attn_swpipe_aic, V=I.cpp.aiv.attn_swpipe_aiv;
  const pA=I.pto.attn_swpipe_aic, pV=I.pto.attn_swpipe_aiv;
  const eA=I.elf.aic||{}, eV=I.elf.aiv||{};
  const sw=I.dfx.swimlane, ch=I.dfx.chip;
  const kb=b=>b==null?'—':(b/1024).toFixed(1)+' KB ('+b+' B)';
  const PP=k=>String(k).replace(/PIPE_/g,'').replace('->',' → ');
  /* 地址簿记类不算「计算算子」 */
  const BOOK=/^(TASSIGN|PTOAS__GLOBAL_TENSOR_DATA)$/;
  const opsOf=(o,n)=>esc(Object.entries(o).filter(([k])=>!BOOK.test(k)).slice(0,n).map(([k,v])=>k+' ×'+v).join(' · '));
  const tshape=o=>esc(Object.entries(o).slice(0,4)
      .map(([k,v])=>String(k).replace(/^Tile</,'').replace(/^TileType::/,'')+' ×'+v).join(' · '));

  let h='';
  h+='<div class="h2">核内 · trace 里没有，静态产物里全有</div>';
  h+='<p class="p">这份 DFX 只采到 <b>task 粒度</b>：'+sw.n_lanes+' 个 lane / '+sw.n_events+
     ' 条事件里，<b>72 个 AIC/AIV 核 lane 每个只有 1 条 slice</b>（名字就是 '+esc(sw.core_event_names.filter(n=>/_(aic|aiv)_spmd/.test(n)).join(' / '))+
     '），既没有子事件，也没有 pipe / 指令级打点。核内的东西要从编译产物本身读 —— 下面全部是从 <code>kernels/</code>、<code>ptoas/</code>、<code>cache/</code> 实测抽出来的。</p>';

  /* ---- 1. 有 / 没有 ---- */
  h+='<div class="card"><div class="h2">核内信息分布：哪一层有什么</div>';
  h+='<table><thead><tr><th>来源</th><th>粒度</th><th>核内信息</th><th>说明</th></tr></thead><tbody>';
  h+='<tr><td class="n">dfx_outputs/merged_swimlane</td><td>核粒度</td><td><span class="pill px">无</span></td>'+
     '<td>'+ch.metadata.num_cores+' 个 AIC/AIV 核 lane 各 1 条 slice，只有起止时刻</td></tr>';
  h+='<tr><td class="n">dfx_outputs/chip_swimlane_records</td><td>核粒度</td><td><span class="pill px">无</span></td>'+
     '<td>aicore_tasks '+ch.n_aicore_tasks+' 条 × '+ch.record_arity.join('/')+' 元组（[3]起 [4]止 [5]dur），无子事件</td></tr>';
  h+='<tr><td class="n">kernels/*/*.cpp</td><td>源码语句</td><td><span class="pill pg">有</span></td>'+
     '<td>核内程序本体：AIC '+A.src_lines+' 行 · AIV '+V.src_lines+' 行（每个 .cpp 同时含两个函数）</td></tr>';
  h+='<tr><td class="n">ptoas/*.pto</td><td>PTO 算子</td><td><span class="pill pg">有</span></td>'+
     '<td>算子 / 循环 / Tile 分配 / 同步对，AIC '+pA.lines+' 行 · AIV '+pV.lines+' 行</td></tr>';
  h+='<tr><td class="n">kernels/*/*.o</td><td>机器码</td><td><span class="pill pa">部分</span></td>'+
     '<td>.text '+eA.text_bytes+' / '+eV.text_bytes+' B + DWARF 行表；本机无 DaVinci 反汇编器，指令不可读</td></tr>';
  h+='<tr><td class="n">cache/incore_*.bin</td><td>核镜像</td><td><span class="pill pa">部分</span></td>'+
     '<td>与 .o 的 .text 等长，二进制不可读</td></tr>';
  h+='</tbody></table></div>';

  /* ---- 2. AIC vs AIV ---- */
  const rr=(a,b)=>'<tr><td class="n">'+a+'</td><td>'+b[0]+'</td><td>'+b[1]+'</td></tr>';
  h+='<div class="card"><div class="h2">两份核内程序：AIC（cube）vs AIV（vector）</div>';
  h+='<p class="p">24 个 AIC 核跑同一份 <code>attn_swpipe_aic</code>，48 个 AIV 核跑同一份 <code>attn_swpipe_aiv</code>，'+
     '靠 <code>kernel_entry</code> 里的 <code>get_block_idx/get_block_num</code> 取各自那份数据。所以「核内」全局只有 2 份程序，核间差异全部来自 block_idx。</p>';
  h+='<table><thead><tr><th>项</th><th>AIC</th><th>AIV</th></tr></thead><tbody>';
  h+=rr('核内源码行数',[A.src_lines+' 行 <span style="color:#767d87">(L'+A.src_start+'–'+A.src_end+')</span>',V.src_lines+' 行 <span style="color:#767d87">(L'+V.src_start+'–'+V.src_end+')</span>']);
  h+=rr('PTO IR 行数',[pA.lines, pV.lines]);
  h+=rr('scf.for / scf.if',[A.loops['for']+' / '+A.loops['if'], V.loops['for']+' / '+V.loops['if']]);
  h+=rr('Tile 分配数（PTO alloc_tile / 生成 TASSIGN）',[pA.n_alloc_tile+' / '+A.tassign, pV.n_alloc_tile+' / '+V.tassign]);
  h+=rr('Tile 类型分布',[Object.entries(A.tile_types).map(([k,v])=>k+' '+v).join(' · '), Object.entries(V.tile_types).map(([k,v])=>k+' '+v).join(' · ')]);
  h+=rr('Tile 尺寸 TOP',[tshape(A.tile_shapes), tshape(V.tile_shapes)]);
  h+=rr('核内算子（去掉地址簿记 TASSIGN / GLOBAL_TENSOR_DATA）',[opsOf(A.ops,10), opsOf(V.ops,30)]);
  h+=rr('set_flag / wait_flag',[A.set_total+' / '+A.wait_total, V.set_total+' / '+V.wait_total]);
  h+=rr('用到的 PIPE',[A.pipes.map(PP).join(' · '), V.pipes.map(PP).join(' · ')]);
  h+='</tbody></table>';
  h+='<div class="note" style="margin-top:10px"><b>读法：</b>AIC 是纯 cube 路径 —— 15 次矩阵乘（<code>TMATMUL</code> ×9 + <code>TMATMUL_ACC</code> ×6）'+
     '配 26 次 <code>TLOAD</code> / 28 次 <code>TMOV</code>；AIV 是纯 vector 路径 —— 24 类向量算子，最重的三个是 '+
     '<code>TCOLEXPANDMUL</code> ×'+V.ops.TCOLEXPANDMUL+'、<code>TMULS</code> ×'+V.ops.TMULS+'、<code>TLOAD</code> ×'+V.ops.TLOAD+'。'+
     '这就是「核内」在做什么：AIC 只算 QK / PV 两个 matmul，其余 RMSNorm、RoPE、softmax、reshape 全在 AIV。</div>';
  h+='</div>';

  /* ---- 3. 片上缓冲：数据通路图 + 尺寸谱 ---- */
  h+='<div class="card"><div class="h2">片上缓冲：硬件通路 × 容量占用</div>';
  h+='<p class="p">方块里的进度条 = <b>累计峰值 ÷ 硬件容量</b>（容量取 a2a3 / 910B 规格）。'+
     '「累计峰值」不是单块尺寸，而是 <code>pto.alloc_tile</code> 里 <code>addr+size</code> 的上界，'+
     '即这块内存同一时刻最多被占掉多少。</p>';
  h+='<div class="note"><b>这张图怎么读：</b><br>'+
     '① <b>虚线框 = 芯片外的内存</b>（GM / DDR，大但慢）；<b>实线框 = 核内的片上缓冲</b>，只有 5 块，小但快。'+
     '数据必须从虚线框一格一格往里倒，计算单元不能直接读 GM。<br>'+
     '② <b>箭头 = 搬运方向</b>，箭头旁的标签是「用哪条硬件流水线搬的 · 搬了几次」。上下顺序就是数据的流入顺序。<br>'+
     '③ <b>框里四个数</b>：<code>已用 / 能装</code> → 进度条 → 占用百分比 → 最下面一行'+
     '<code>最大单块 · 分配次数 · 同一时刻相当于几块</code>。<b>红框 = 打满</b>。</div>';
  h+=memFlowSvg(pA,pV);
  h+='<table style="margin-top:2px"><thead><tr><th>缓冲</th><th>好比</th><th>实际干什么</th><th>谁搬进来 / 谁搬出去</th></tr></thead><tbody>';
  h+='<tr><td class="n">L1 <code>mat</code> · 512 KB</td><td>操作台</td><td>GM 与计算单元之间的<b>中转站</b>，所有数据先到这</td><td>MTE2 搬入 ← GM ； MTE1 搬出 → L0A / L0B</td></tr>';
  h+='<tr><td class="n">L0A <code>left</code> · 64 KB</td><td>左手边的料盒</td><td>矩阵乘的<b>左操作数</b>（激活值），单块只有 4 KB</td><td>MTE1 搬入 ； CUBE 直接读</td></tr>';
  h+='<tr><td class="n">L0B <code>right</code> · 64 KB</td><td>右手边的料盒</td><td>矩阵乘的<b>右操作数</b>（权重），单块 32 KB</td><td>MTE1 搬入 ； CUBE 直接读</td></tr>';
  h+='<tr><td class="n">L0C <code>acc</code> · 128 KB</td><td>出锅盘</td><td>矩阵乘的<b>累加结果</b>，算完先放这</td><td>CUBE 写入 ； FIX 搬出 → GM</td></tr>';
  h+='<tr><td class="n">UB <code>vec</code> · 192 KB</td><td>备料碗</td><td>AIV 侧<b>唯一的</b>缓冲，所有中间量都在这周转</td><td>MTE2 搬入 ； VECTOR 读写 ； MTE3 搬出</td></tr>';
  h+='</tbody></table>';
  h+='<div class="note" style="margin-top:10px"><b>「同一时刻 ≈ 几块这么大」是什么意思：</b>'+
     '代码里一共调了 31 次 <code>alloc_tile</code> 要 L1 空间，每次最大 32 KB。如果每块都单独占一个位置，要 31 × 32 KB = <b>992 KB</b>，'+
     '但 L1 只有 512 KB —— 装不下。所以编译器让它们<b>轮流借用同一段空间</b>：真正同时活着的最多只有 204 KB ÷ 32 KB ≈ <b>6.4 块</b>。'+
     'AIV 的 UB 更极端：212 次分配累加要 <b>3.3 MB</b>，实际只有 192 KB，复用倍数约 36×。<br>'+
     '这个数还有个特殊含义：<b>L0A / L0B 正好 2.0</b> 说明它们是标准的<b>双缓冲</b>（一块在算、一块在搬）；'+
     '<b>L0C = 1.0</b> 说明它只有一份，没有双缓冲，算和搬会互相等。</div>';
  h+='</div>';

  h+='<div class="card"><div class="h2">单个 tile 尺寸谱：AIC 少而大 · AIV 多而碎</div>';
  h+='<p class="p">按面积等比缩放（<b>边长 ∝ √字节数</b>），全部画到同一把尺子上。AIC 只有 5 种形状、单块最大 32 KB；'+
     'AIV 有 20 种形状、单块最大 16 KB，最小只有 32 B —— 全局最大与最小相差 <b>1024×</b>。'+
     '方块下方标 <code>×n</code> 的是该形状的分配次数。</p>';
  h+=mosaicSvg();
  h+='<div class="note" style="margin-bottom:0"><b>读出来的结论：</b>'+
     '① AIC 的 L0B 累计峰值 64.0 KB = 容量 64 KB，<b>打满 100%</b>（两个 128×128 bf16 操作数各 32 KB，正好 2.00 份槽位），'+
     '而 L0C 只用 8 KB / 128 KB = 6.25% —— <b>L0B 是这套 kernel 的硬资源边界</b>；'+
     '② L0A 只用 8 KB / 64 KB = 12.5%，13 次分配全是 16×128 小片，与 L0B 形成极端不对称 —— 这正是「左操作数按 16 行切片、右操作数整块常驻」的算法结构；'+
     '③ AIV 的 212 次 <code>alloc_tile</code> 被 <code>MemoryReuse</code> 折叠进 94.1 KB（5.88 份槽位），'+
     '但 20 种形状里有 <b>7 种 ≤ 512 B</b>，却吃掉 <b>142 次分配（占 67%）</b>；'+
     '这些 <code>[1,8]</code> / <code>[8,1]</code> / <code>[1,16]</code> 的标量级小片就是 RMSNorm / RoPE 的位置参数，搬运效率极低。</div>';
  h+='<p class="p" style="margin:10px 0 0">数据来源：<code>ptoas/attn_swpipe.pto</code> 的 '+pA.n_alloc_tile+' 次（AIC）+ '+pV.n_alloc_tile+
     ' 次（AIV）<code>pto.alloc_tile</code>，逐条按 <code>loc / dtype / rows / cols / addr</code> 复算；'+
     '不是从 <code>MemoryReuse</code> 之后的 dump 抄的，所以能和地址分配结果对上。</p>';
  h+='</div>';

  /* ---- 4. 流水同步 ---- */
  h+='<div class="card"><div class="h2">核内流水：set_flag / wait_flag 配对</div>';
  h+='<p class="p">核内几条硬件流水线（MTE2 / MTE1 / M / FIX）是<b>各自独立跑</b>的，谁也不知道谁走到哪了。'+
     '它们之间唯一的同步手段就是 <code>set_flag(src, dst, EVENT_ID)</code> 和 <code>wait_flag(src, dst, EVENT_ID)</code> —— '+
     '<b>set 是我这边好了，wait 是我在等你那边</b>，两者共用同一个 (src, dst, event) 三元组。</p>';
  h+='<p class="p">每个相邻两块之间都有<b>两条方向相反的边</b>，这是双缓冲的必然结果：'+
     '<b>上排 →（数据就绪）</b>= 上游搬完了，下游可以取；<b>下排 ←（缓冲释放）</b>= 下游读空了，上游可以覆盖。'+
     '只有上排没有下排，上游就会把下游还没读完的那块写坏。</p>';
  h+='<p class="p">两类核的 set / wait 总数完全对称（AIC '+A.set_total+'↔'+A.wait_total+'，AIV '+V.set_total+'↔'+V.wait_total+
     '），而且<b>逐条边都严格相等</b> —— 每一根 set 都有且只有一个 wait 接住，没有漏等（会卡死）、没有多 set（会假就绪读到脏数据）。</p>';
  h+=pipeSvg(A.set_flag,['MTE2','MTE1','M','FIX'],'#5b8dd6','AIC 流水（GM→L1→L0A/B→MAC→L0C→GM）');
  h+='<div style="height:12px"></div>';
  h+=pipeSvg(V.set_flag,['MTE2','V','MTE3'],'#a06ad0','AIV 流水（GM→UB→Vector→GM）');
  h+='<table style="margin-top:12px"><thead><tr><th>核</th><th>握手边 src → dst</th><th>set</th><th>wait</th><th>配对</th></tr></thead><tbody>';
  const pp=k=>k.replace(/PIPE_/g,'').replace('->',' → ');
  const rowF=(tag,src,setO,waitO)=>{
    Object.entries(setO).forEach(([k,n])=>{
      const w=waitO[k];
      const ok=(w===n);
      h+='<tr><td class="n">'+tag+'</td><td>'+esc(pp(k))+'</td><td>'+n+'</td><td>'+
         (w!==undefined?w:'—')+'</td><td>'+
         (ok?'<span class="pill pg">✓ 配对</span>':'<span class="pill pr">差 '+Math.abs(n-(w||0))+'</span>')+
         '</td></tr>';
    });
  };
  rowF('AIC',null,A.set_flag,A.wait_flag);
  rowF('AIV',null,V.set_flag,V.wait_flag);
  h+='</tbody></table>';
  h+='<div class="note"><b>这些数字本身就是流水分块的证据：</b>'+
     '<code>MTE2→MTE1</code> 16 次 = GM→L1 搬了 16 轮；<code>MTE1→M</code> 15 次 = L0 级算了 15 轮；'+
     '<code>M→FIX</code> 只有 9 次 = 只产出 9 个结果块。<b>搬入 16 轮 → 输出 9 块</b>，'+
     '说明存在<b>多块累加成一输出</b>（9 个输出块吃掉了 15 轮矩阵乘），这正是 paged attention 把多个 K/V page 累加到一个 attention 输出的形状。'+
     'AIC 三级轮数 16 / 15 / 9 逐级递减，就是分块粒度逐级变粗。</div>';
  h+='<p class="p" style="margin:9px 0 0">AIV 里还有一条跨级边 <code>MTE3→MTE2</code> ×'+V.set_flag['PIPE_MTE3->PIPE_MTE2']+'（写回完成后再让下一轮 load 复用同一段 UB），所以 AIV 的同步图不是纯链式的。'+
     ' EVENT_ID 用了 0–7 八个事件位（AIC 共 '+Object.values(A.event_ids).reduce((a,b)=>a+b,0)+' 次 set+wait、AIV 共 '+Object.values(V.event_ids).reduce((a,b)=>a+b,0)+' 次），'+
     '说明编译器把不同 buffer slot 拆到了不同 event id 上 —— 否则同一个 id 上的 set 会把前面还没被 wait 的信号覆盖掉。</p>';
  h+='</div>';

  /* ---- 5. 二进制 ---- */
  h+='<div class="card"><div class="h2">核内二进制（text/stack 段实测）</div>';
  h+='<table><thead><tr><th>项</th><th>AIC</th><th>AIV</th></tr></thead><tbody>';
  h+=rr('.o 文件',[esc(eA._file||'—'), esc(eV._file||'—')]);
  h+=rr('.text 段',[eA.text_bytes+' B', eV.text_bytes+' B']);
  h+=rr('cache/incore_*.bin',[I.cache['incore_0_aic_attn_swpipe_aic_a2a3.bin'].bytes+' B（与 .text 等长）', I.cache['incore_1_aiv_attn_swpipe_aiv_a2a3.bin'].bytes+' B（与 .text 等长）']);
  h+=rr('.debug_line',[eA.debug_line_bytes+' B', eV.debug_line_bytes+' B']);
  h+=rr('.debug_info',[eA.debug_info_bytes+' B', eV.debug_info_bytes+' B']);
  h+=rr('.symtab',[eA.symtab_bytes+' B', eV.symtab_bytes+' B']);
  h+=rr('.ascend.stack.size.record',[kb(eA.stack_bytes), kb(eV.stack_bytes)]);
  h+=rr('.ascend.meta',['<code>'+esc(eA.ascend_meta_hex||'')+'</code>','<code>'+esc(eV.ascend_meta_hex||'')+'</code>']);
  h+='</tbody></table>';
  const ctx=I.cache._context||{};
  h+='<p class="p" style="margin:9px 0 0">目标 <code>'+esc(ctx.platform||'')+'</code> · runtime <code>'+esc(ctx.runtime_name||'')+
     '</code> · PTO ISA <code>'+esc(String(ctx.pto_isa_revision||'').slice(0,12))+'</code>。'+
     '<code>.o</code> 里的 <code>.text</code> 与 <code>cache/incore_*.bin</code> 字节数完全一致，说明 incore bin 就是 <code>.text</code> 的直出；'+
     '两个核栈都是 32 KB。DWARF（.debug_line/.debug_info）是完整的，所以只要有一把能读 DaVinci ISA 的反汇编器（ascend 工具链 / llvm-ascend），'+
     '就能把 <code>.text</code> 反汇编并逐行映射回 <code>attn_swpipe_*.cpp</code> 的源码行，做出真正的「核内指令视图」。</p></div>';

  /* ---- 6. 缺什么 ---- */
  h+='<div class="card"><div class="h2">还缺什么 · 怎么补</div>';
  h+='<table><thead><tr><th>缺的数据</th><th>这份产物里</th><th>补法</th></tr></thead><tbody>';
  h+='<tr><td class="n">核内 pipe 占用率<br><span style="color:#767d87">MTE1/MTE2/M/FIX/V 各自忙闲</span></td><td><span class="pill pr">无</span></td>'+
     '<td>Ascend Profiling（msprof / aic-metric / pipe utilization）采集，或在 kernel 里加核侧打点</td></tr>';
  h+='<tr><td class="n">核内指令级时序 / stall</td><td><span class="pill pr">无</span></td><td>需 instruction trace 或 cost-model（本产物的 <code>__COSTMODEL</code> 分支只是编译开关）</td></tr>';
  h+='<tr><td class="n">核内 MTE 带宽 / L1 bank 冲突</td><td><span class="pill pr">无</span></td><td>PMU 计数器</td></tr>';
  h+='<tr><td class="n">核内访存延迟分布</td><td><span class="pill px">间接</span></td>'+
     '<td><code>report/perf_hints.log</code> 的 PH001（33 条：内维 256 B &lt; 512 B cache line）能间接反映</td></tr>';
  h+='<tr><td class="n">核内算子级耗时</td><td><span class="pill px">可估</span></td>'+
     '<td>本 pane 的算子计数 + 每核 slice 跨度（52.98 µs / 72 核）可做粗估，但不是实测</td></tr>';
  h+='</tbody></table>';
  h+='<div class="note bad" style="margin-top:10px"><b>一句话：</b>核内「结构」是静态可读的（本 pane 全部内容），核内「时序」需要另一次 profiling 采集 —— 本次产物里不存在，'+
     '不要把它当成核内性能数据来用。</div></div>';

  $('#incorebody').innerHTML=h;
}


/* ==================================================================
   PMU 与仿真 pane —— 20261016 profiling bundle
   三级可信度：硬件 PMU / 硬件 trace / 仿真模型 / 失败的 costmodel
================================================================== */
const PIPECOL={VECTOR:'#a06ad0',CUBE:'#5b8dd6',SCALAR:'#8b939d',MTE1:'#4fae72',MTE2:'#2f8f6a',
                MTE3:'#79c4a4',FIXP:'#e0bd76',FLOWCTRL:'#c05656',ALL:'#6b7280',CACHEMISS:'#4a525c'};
const pfk=v=>v>=1000?(v/1000).toFixed(v>=10000?0:1)+'k':String(v);

/* ---------- 72 核 PMU 条带图 ---------- */
function pfPmu(){
  const P=PROFILE; if(!P||!P.pmu) return '';
  const byCore={}; P.pmu.cores.forEach(c=>byCore[c.core]=c);
  const groups=[];
  ['0','1','2'].forEach(t=>{
    const cs=P.pmu.threads[t].cores;
    ['aic','aiv'].forEach(ty=>{
      const g=cs.filter(c=>byCore[c].type===ty);
      if(g.length) groups.push({t:t,type:ty,cores:g});
    });
  });
  const W=680,padL=46,padR=8,GAP=13;
  const n=P.pmu.cores.length, cw=(W-padL-padR-GAP*2)/n;
  const CY=58000;
  const y0=40,h1=110, y2=y0+h1+42, h2=72, H=y2+h2+26;
  let s='<svg viewBox="0 0 680 '+H+'" width="100%" style="height:auto;max-width:680px" role="img" aria-label="PMU 72 核">';
  /* group 背景 */
  let x=padL, gx=[];
  groups.forEach((g,gi)=>{
    const gw=g.cores.length*cw;
    s+='<rect x="'+x.toFixed(1)+'" y="'+(y0-16)+'" width="'+gw.toFixed(1)+'" height="'+(h1+h2+62)+'" fill="'+(gi%2?'#23272c':'#1d2125')+'"/>';
    s+='<text x="'+(x+gw/2).toFixed(1)+'" y="'+(y0-20)+'" text-anchor="middle" font-family="ui-monospace,monospace" font-size="9.5" fill="'+(g.type==='aic'?'#8fb4ee':'#c1a2f0')+'">t'+g.t+' · '+(g.type==='aic'?'AIC':'AIV')+' ×'+g.cores.length+'</text>';
    gx.push(x); x+=gw+GAP;
  });
  /* cycles 轴 */
  [0,20000,40000,60000].forEach(v=>{
    if(v>CY) return;
    const yy=y0+h1-(v/CY)*h1;
    s+='<line x1="'+padL+'" y1="'+yy.toFixed(1)+'" x2="'+(W-padR)+'" y2="'+yy.toFixed(1)+'" stroke="#2b2f34"/>';
    s+='<text x="'+(padL-5)+'" y="'+(yy+3).toFixed(1)+'" text-anchor="end" font-family="ui-monospace,monospace" font-size="8.5" fill="#767d87">'+(v?v/1000+'k':'0')+'</text>';
  });
  s+=T(padL-5,y0-4,'cycles',8.5,'#767d87',null,'end');
  /* cycles bars */
  x=padL;
  groups.forEach(g=>{
    g.cores.forEach(c=>{
      const d=byCore[c], bh=(d.cycles/CY)*h1;
      s+='<rect x="'+(x+0.7).toFixed(1)+'" y="'+(y0+h1-bh).toFixed(1)+'" width="'+(cw-1.4).toFixed(1)+
         '" height="'+bh.toFixed(1)+'" fill="'+(g.type==='aic'?'#5b8dd6':'#a06ad0')+'" fill-opacity="'+(g.t==='1'?0.85:1)+'"/>';
      x+=cw;
    });
    x+=GAP;
  });
  /* 组均值 */
  groups.forEach((g,gi)=>{
    const v=PROFILE.pmu.threads[g.t][g.type].cycles_mean;
    s+=T(gx[gi]+g.cores.length*cw/2, y0+h1+14, pfk(v), 9.5, g.type==='aic'?'#8fb4ee':'#c1a2f0', '650','middle');
  });
  /* 第二面板：Σ 带宽计数器（log）—— 必须按核 id 查表，不能用分组序号索引 */
  const bwOf=c=>{const d=byCore[c];
    return d.ub_read_bw_mte+d.l2_write_bw+d.main_mem_write_bw+d.ub_write_bw_mte+
           d.ub_read_bw_vector+d.ub_write_bw_vector+d.ub_read_bw_scalar+d.ub_write_bw_scalar;};
  const LOGMAX=Math.log10(4001);
  [1,10,100,1000].forEach(v=>{
    const yy=y2+h2-(Math.log10(v)/LOGMAX)*h2;
    s+='<line x1="'+padL+'" y1="'+yy.toFixed(1)+'" x2="'+(W-padR)+'" y2="'+yy.toFixed(1)+'" stroke="#2b2f34"/>';
    s+=T(padL-5,yy+3,v,8.5,'#767d87',null,'end');
  });
  s+='<line x1="'+padL+'" y1="'+y2+'" x2="'+(W-padR)+'" y2="'+y2+'" stroke="#2b2f34"/>';
  s+=T(padL-5,y2-5,'Σbw',8.5,'#767d87',null,'end');
  x=padL;
  groups.forEach(g=>{
    g.cores.forEach(c=>{
      const v=bwOf(c);
      const bh=v?Math.max(1.2,(Math.log10(v+1)/LOGMAX)*h2):0;
      s+='<rect x="'+(x+0.7).toFixed(1)+'" y="'+(y2+h2-bh).toFixed(1)+'" width="'+(cw-1.4).toFixed(1)+
         '" height="'+bh.toFixed(1)+'" fill="'+(v?((g.type==='aic'?'#5b8dd6':'#a06ad0')):'#3a3f45')+'"/>';
      x+=cw;
    });
    x+=GAP;
  });
  /* t1 零搬运标注 —— 指向 t1·AIV ×16 那组，不是 t1·AIC */
  const gi1=groups.findIndex(g=>g.t==='1'&&g.type==='aiv');
  const g1=gx[gi1]+groups[gi1].cores.length*cw/2;
  s+='<text x="'+g1.toFixed(1)+'" y="'+(y2+h2+16)+'" text-anchor="middle" font-family="ui-monospace,monospace" font-size="9.5" fill="#e0bd76">← 这一组 16 个 AIV 核心带宽计数全为 0</text>';
  s+='</svg>';
  return s;
}

/* ---------- 管线占用堆叠条 ---------- */
function pfPipes(){
  const S=PROFILE&&PROFILE.sim; if(!S||!S.cores) return '';
  const order=['core0.cubecore0','core0.veccore0','core0.veccore1'];
  const label={'core0.cubecore0':'AIC（cube）','core0.veccore0':'AIV subcore 0','core0.veccore1':'AIV subcore 1'};
  const W=680,x0=104,bw=W-x0-84, rows=order.filter(k=>S.cores[k]);
  let H=24+rows.length*48+6;
  let s='<svg viewBox="0 0 680 '+H+'" width="100%" style="height:auto;max-width:680px" role="img" aria-label="pipe 占用">';
  rows.forEach((k,i)=>{
    const c=S.cores[k], y=16+i*48;
    s+=T(0,y+13,label[k],10.5,'#e6e8ea','650');
    s+=T(0,y+27,c.cycles.toLocaleString()+' cyc · '+c.us+' µs',9,'#767d87');
    let x=x0;
    c.pipes.forEach(p=>{
      const w=bw*p.cycles/c.cycles;
      s+='<rect x="'+x.toFixed(1)+'" y="'+y+'" width="'+Math.max(w,0).toFixed(1)+'" height="21" fill="'+(PIPECOL[p.pipe]||'#666')+'"/>';
      if(w>24) s+=T((x+w/2).toFixed(1),y+14,p.pct+'%',8.5,'#10130f',null,'middle');
      x+=w;
    });
    s+=T(x0+bw+8,y+10,c.pipes[0].pipe,9,PIPECOL[c.pipes[0].pipe]||'#e6e8ea','650');
    s+=T(x0+bw+8,y+21,c.pipes[0].pct+'%',8.5,PIPECOL[c.pipes[0].pipe]||'#8b939d');
  });
  s+='</svg>';
  /* legend */
  const used={}; S.cores['core0.cubecore0'].pipes.forEach(p=>used[p.pipe]=1);
  S.cores['core0.veccore0'].pipes.forEach(p=>used[p.pipe]=1);
  return s+'|'+Object.keys(used).map(k=>'<span><i style="background:'+(PIPECOL[k]||'#666')+'"></i>'+k+'</span>').join('');
}

/* ---------- top 指令 ---------- */
function pfInstr(){
  const S=PROFILE&&PROFILE.sim; if(!S||!S.cores) return '';
  const draw=(k,title,col)=>{
    const c=S.cores[k]; if(!c) return '';
    const top=c.instr.slice(0,8), mx=top[0].cycles;
    let h='<div><div class="h3">'+esc(title)+'</div>';
    top.forEach(it=>{
      h+='<div class="ib"><span class="ibn">'+esc(it.instr)+'</span>'+
         '<span class="ibb"><i style="width:'+(100*it.cycles/mx).toFixed(1)+'%;background:'+col+'"></i></span>'+
         '<span class="ibv">'+it.pct+'%</span></div>';
    });
    return h+'</div>';
  };
  return '<div class="grid2">'+draw('core0.cubecore0','AIC · 单指令累计 cycles 占比','#5b8dd6')+
         draw('core0.veccore0','AIV · 单指令累计 cycles 占比','#a06ad0')+'</div>';
}

/* ---------- 两次运行对比 ---------- */
function pfRuns(){
  const R=PROFILE&&PROFILE.runs; if(!R||!R.new) return '';
  const o=R.old,n=R.new;
  const W=680,padL=118,bw=W-padL-64;
  const total=Math.max(o?o.window_us:0,n.window_us);
  const sc=v=>bw*v/total;
  const RH=62, TOP=20;
  let s='<svg viewBox="0 0 680 '+(TOP+2*RH+4)+'" width="100%" style="height:auto;max-width:680px" role="img" aria-label="两次运行对比">';
  const row=(tag,r,y,col)=>{
    s+=T(0,y+11,tag,10.5,'#e6e8ea','650');
    s+=T(0,y+23,r.file.replace('merged_swimlane_','').replace('.json',''),8.5,'#767d87');
    /* 全窗口 */
    s+='<rect x="'+padL+'" y="'+y+'" width="'+sc(r.window_us).toFixed(1)+'" height="8" rx="2" fill="#33383e"/>';
    s+=T(padL+sc(r.window_us)+6,y+7,r.window_us+' µs',9,'#8b939d');
    /* 首核开跑刻度 */
    const cx=padL+sc(r.core_start_min);
    s+='<line x1="'+cx.toFixed(1)+'" y1="'+(y-5)+'" x2="'+cx.toFixed(1)+'" y2="'+(y+31)+'" stroke="#e0bd76" stroke-dasharray="3 2"/>';
    s+=T(cx-5,y-4,'首核开跑 '+r.core_start_min+' µs',8.5,'#e0bd76',null,'end');
    /* 核窗口 */
    s+='<rect x="'+cx.toFixed(1)+'" y="'+(y+13)+'" width="'+sc(r.kernel_window_us).toFixed(1)+'" height="11" rx="2" fill="'+col+'"/>';
    s+=T((cx+sc(r.kernel_window_us)/2).toFixed(1),y+21.5,'核 '+r.kernel_window_us+' µs',8.5,'#10130f',null,'middle');
  };
  if(o) row('20261015',o,TOP,'#5b8dd6');
  row('20261016',n,TOP+RH,'#c05656');
  s+='</svg>';
  return s;
}

function buildProfile(){
  const P=PROFILE;
  if(!P||!P.pmu){ $('#pmubody').innerHTML='<div class="card">未找到 20261016 profiling bundle</div>'; return; }
  const pm=P.pmu, lg=P.log, S=P.sim, R=P.runs;
  let h='';

  h+='<div class="h2">20261016 profiling bundle：第一次有核内数据</div>';
  h+='<p class="p">新增目录 <code>qwen3_14b_paged_attention_profile_20261016/</code> 补上了上次说「缺」的两件事：'+
     '<b>72 核真实 PMU 计数器</b>和<b>核内指令级仿真</b>。但四份数据的可信度完全不同，必须先分清再读。</p>';

  /* ---- 1. 可信度 ---- */
  h+='<div class="card"><div class="h2">四份数据，三种可信度</div>';
  h+='<table><thead><tr><th>来源</th><th>性质</th><th>能回答什么</th><th>不能回答什么</th></tr></thead><tbody>';
  h+='<tr><td class="n">dfx_outputs/pmu.csv</td><td><span class="pill pg">硬件实测</span></td>'+
     '<td>72 核各自的 cycles 与 8 个带宽计数（PMU group '+pm.pmu_group+'：'+esc(pm.pmu_group_label)+'）</td>'+
     '<td>没有 pipe 占用率、没有 stall 原因</td></tr>';
  h+='<tr><td class="n">dfx_outputs/merged_swimlane</td><td><span class="pill pg">硬件实测</span></td>'+
     '<td>dispatch→finish 时间轴（µs），与日志 <code>Total Test Time</code> 对得上</td>'+
     '<td>核内无子事件；<code>chip_swimlane</code> 的 [3]/[4] 不是线性时间（见下）</td></tr>';
  h+='<tr><td class="n">kernel_insight_*/simulator</td><td><span class="pill pa">仿真模型</span></td>'+
     '<td>核内 <b>pipe 级 / 指令级</b> cycles 分布、同步开销占比</td>'+
     '<td>不是这次硬件运行的输入：单核、合成数据、<code>seq_lens=[128,129]</code></td></tr>';
  h+='<tr><td class="n">costmodel_attn_swpipe</td><td><span class="pill pr">无结果</span></td>'+
     '<td>—</td><td>PTO ISA 头文件在宿主工具链下编不过，<code>build.log</code> 只有报错，<b>不是性能数据</b></td></tr>';
  h+='</tbody></table>';
  h+='<div class="note" style="margin-bottom:0"><b>仿真 workload 与硬件不同，只可比「比例」不可比「绝对值」：</b>'+
     '仿真用 <code>seq_lens=[128,129]</code>（硬件是 512/513）、单核回放、输入由 <code>golden.py</code> 生成。'+
     '所以下面 AIC 仿真 44.8k cycles 和硬件 PMU 均值 41.3k 接近只是巧合，不要当成校准。</div></div>';

  /* ---- 2. PMU ---- */
  h+='<div class="card"><div class="h2">PMU：72 核真实负载 —— 三分之一的 block 在空转</div>';
  h+='<p class="p">上：每核 <code>pmu_total_cycles</code>。下：每核 8 个带宽计数器之和（对数轴，0 直接不画）。'+
     '按 AICPU 的 3 个调度线程分组排列 —— 结构立刻显现：<b>t1 那 24 个核（8 block）几乎没有数据搬运</b>。</p>';
  h+=pfPmu();
  h+='<table style="margin-top:14px"><thead><tr><th>调度线程</th><th>block</th><th>AIC cycles 均值</th><th>AIC main_mem_write_bw</th>'+
     '<th>AIV cycles 均值</th><th>AIV ub_read_bw_vector</th></tr></thead><tbody>';
  ['0','1','2'].forEach(t=>{
    const th=pm.threads[t], a=th.aic, v=th.aiv;
    const hot=(a.main_mem_write_bw<5);
    h+='<tr'+(hot?' style="background:#332a1c"':'')+'><td class="n">t'+t+'</td><td>'+th.cores.filter(c=>pm.cores[c].type==='aic').length+' AIC + '+
       th.cores.filter(c=>pm.cores[c].type==='aiv').length+' AIV</td>'+
       '<td>'+a.cycles_mean+'</td><td'+(hot?' style="color:#e0bd76;font-weight:600"':'')+'>'+a.main_mem_write_bw+'</td>'+
       '<td>'+v.cycles_mean+'</td><td'+(hot?' style="color:#e0bd76;font-weight:600"':'')+'>'+v.ub_read_bw_vector+'</td></tr>';
  });
  h+='</tbody></table>';
  h+='<div class="note"><b>这不是采集缺失，是真的空转 —— 两条独立证据链：</b>'+
     '① PMU 里 t1 的 cycles 只有其他组的 0.52–0.57×，但带宽计数是 0（AIV）/ 2（AIC 写）；若计数器本身没工作，'+
     '<code>main_mem_write_bw</code> 应也为 0，而它是 1–2。'+
     '② 不看 PMU，单看 trace：t1 的核 slice 明显更短（如 core 1 = 19.68 µs vs core 0 = 41.36 µs，比值 0.48，与 cycles 的 0.48 一致）。'+
     '两条链路互相印证 → <b>中间 8 个 block 拿到了几乎为空的 KV 分片，仍在花 26–29k cycles 等 flag</b>。</div></div>';

  /* ---- 3. 仿真 ---- */
  h+='<div class="card"><div class="h2">核内仿真：真正的瓶颈不是 matmul</div>';
  h+='<p class="p">按 pipe 累计 cycles（仿真模型值，只看比例）。左 AIC、右 AIV：</p>';
  const pipes=pfPipes().split('|');
  h+=pipes[0];
  h+='<div class="legend">'+pipes[1]+'</div>';
  h+=pfInstr();
  h+='<div class="note bad"><b>两个反直觉的数字：</b>'+
     '① AIC 上 <code>CUBE</code> 只占 <b>7.5%</b>（3374 cycles），而 <code>SCALAR</code> 37.7% + <code>WAIT_FLAG</code> 26.5% + '+
     '<code>LD/ST_XD_XN_IMM</code> 27.8%（标量地址簿记）—— 矩阵乘引擎大部分时间在等；'+
     '② AIV 上 <code>BAR</code>（barrier）一项就吃掉 <b>39.3%</b>（37955 cycles / 93 次，平均 408 cycles/次），'+
     '比向量计算本身（51.5%）里任何单个算子都重。注意这是仿真值，且 workload 更小。</div></div>';

  /* ---- 4. 两次对比 ---- */
  h+='<div class="card"><div class="h2">与 20261015 那次对比：同一个 kernel，慢了一倍</div>';
  h+='<p class="p">两次产物里 <code>attn_swpipe_aic/aiv</code> 的 <code>.text</code> 都是 4664 / 11884 B，'+
     '<code>compiled_meta.json</code> 与 <code>kernel_config.py</code> 完全一致，deps 也同样是 8 个 task（7 dummy + 1 真）。'+
     '<b>同一个二进制</b>，差别只在输入（<code>--batch 2 --seq-lens 512,513 --page-mapping reverse</code>）。</p>';
  h+=pfRuns();
  h+='<div class="legend" style="font-size:9.5px;color:#767d87">'+
     '<span><i style="background:#33383e"></i>全窗口（首个事件→最后事件）</span>'+
     '<span><i style="background:#5b8dd6"></i>核窗口 20261015</span>'+
     '<span><i style="background:#c05656"></i>核窗口 20261016</span>'+
     '<span><i style="background:#e0bd76"></i>首核开跑时刻</span></div>';
  h+='<table style="margin-top:12px"><thead><tr><th>指标</th><th>20261015</th><th>20261016</th><th>变化</th></tr></thead><tbody>';
  const rr=(k,a,b,d)=>'<tr><td class="n">'+k+'</td><td>'+a+'</td><td>'+b+'</td><td>'+(d||'')+'</td></tr>';
  if(R.old){
    h+=rr('总窗口 (µs)',R.old.window_us,R.new.window_us,'<span style="color:#e79a9a">+'+((R.new.window_us/R.old.window_us-1)*100).toFixed(0)+'%</span>');
    h+=rr('核执行窗口 (µs)',R.old.kernel_window_us,R.new.kernel_window_us,'<span style="color:#e79a9a">+'+((R.new.kernel_window_us/R.old.kernel_window_us-1)*100).toFixed(0)+'%</span>');
    h+=rr('首核开跑 (µs)',R.old.core_start_min,R.new.core_start_min,'<span style="color:#e79a9a">+'+((R.new.core_start_min/R.old.core_start_min-1)*100).toFixed(0)+'%</span>');
    h+=rr('核 slice 均值 (µs)',R.old.core_dur_mean,R.new.core_dur_mean,'—');
    h+=rr('核 slice 最大 (µs)',R.old.core_dur_max,R.new.core_dur_max,'—');
  }
  if(lg&&lg.funcs.length){
    lg.funcs.forEach(f=>{
      h+=rr(f.name+' exec (µs)','—',f.exec_us,'<span style="color:#767d87">latency '+f.latency_us+'</span>');
    });
    h+=rr('Head OH / Tail OH (µs)','—',lg.funcs[0].head_oh_us+' / '+lg.funcs[0].tail_oh_us,
          '<span style="color:#e0bd76">Tail 比 exec 还大</span>');
    h+=rr('每核 exec / window','—',lg.per_task.avg_exec_us+' / '+R.new.kernel_window_us,
          '<span style="color:#e0bd76">'+(100*lg.per_task.avg_exec_us/R.new.kernel_window_us).toFixed(1)+'%</span>');
    h+=rr('正确性校验','<span class="pill pr">无</span>',
          (lg.extra.validated||[]).map(v=>v.tensor+' '+v.status).join(' · '),'<span class="pill pg">已补</span>');
  }
  h+='</tbody></table>';
  h+='<div class="note"><b>Tail OH 23.81 µs（AIC）/ 25.76 µs（AIV）比 exec 22.33 / 22.66 µs 还大</b> —— '+
     '收尾开销约等于整个计算时长。日志给的 <code>Exec/Latency = 42.07%</code> 就是这么来的。'+
     '这条以前只能从 trace 的 slice 跨度猜，现在有 runtime 自己算的 head/tail 拆分。</div></div>';

  /* ---- 5. 修正 ---- */
  h+='<div class="card"><div class="h2">对之前结论的修正</div>';
  h+='<table><thead><tr><th>之前的说法</th><th>现在</th></tr></thead><tbody>';
  h+='<tr><td>「核内时序采不到，需要另采一次 profiling」</td><td><span class="pill pg">已补</span> PMU 72 核 + 指令级仿真都有了；仍缺 pipe 占用率与 stall 归因</td></tr>';
  h+='<tr><td>「24 block 只摊了 16 个 task，8 个 block 空转」</td>'+
     '<td><span class="pill pg">已证实</span> 8 个 block 的带宽计数为 0，cycles 只有其他组的 0.52–0.57×</td></tr>';
  h+='<tr><td>「trace 的核执行窗口 52.98 / 79.6 µs」</td>'+
     '<td><span class="pill pa">修正</span> 那是 <b>dispatch→finish 的 latency 窗口</b>，不是核上执行时间；真正的 exec 只有 22.3 / 22.7 µs/核</td></tr>';
  h+='<tr><td>「本次产物没有正确性校验」</td>'+
     '<td><span class="pill pg">已补</span> key_cache / value_cache / out 三个张量全部 PASS（<code>out</code> 还做了数值比较）</td></tr>';
  h+='<tr><td>「chip_swimlane 时基与 trace 差 1.55×」</td>'+
     '<td><span class="pill pa">升级为确定问题</span> 逐核拟合 ns/unit 在 <b>28.6–67.9</b> 之间飘，不是常数 → 该字段不是线性时间，不能用</td></tr>';
  h+='</tbody></table></div>';

  $('#pmubody').innerHTML=h;
}


/* ============================================================
   输入定义：shape 从哪来、值有没有落盘
============================================================ */
const GLOSS={
  key_cache:'KV cache 的 K 部分',value_cache:'KV cache 的 V 部分',
  block_table:'分页表：逻辑页 → 物理块',seq_lens:'每序列已有的 KV 长度',
  inv_rms_states:'每 token 的 1/rms',slot_mapping:'新 token 写入 cache 的行号',
  rope_cos:'RoPE cos 表（最大位置 4096）',rope_sin:'RoPE sin 表',
  q_proj:'本步 Q 投影（2 个新 token）',k_proj:'本步 K 投影',v_proj:'本步 V 投影',
  q_norm_w:'Q RMSNorm 权重',k_norm_w:'K RMSNorm 权重',
  layer_cache_base_token_rows:'标量：cache 行基址',
  out:'注意力输出',q_tnd_flat:'q_proj 展平（token × head）',
  score_transfer:'AIC↔AIV 传 score',probability_transfer:'AIC↔AIV 传 softmax 结果',
  pv_transfer:'AIC↔AIV 传 P·V 结果',ffts_workspace:'FFTS 同步工作区'
};
const DT={bfloat16:'bf16',fp32:'fp32',float32:'fp32',int32:'int32',int64:'int64',
          index:'index',BF16:'bf16',FP32:'fp32',INT32:'int32',INT64:'int64'};
const dt=x=>DT[x]||x||'—';
const shp=s=>Array.isArray(s)?'['+s.join(', ')+']':'—';

function buildInputs(){
  const I=INPUTS; if(!I||!I.meta){ $('#inputsbody').innerHTML='<div class="card">未找到输入定义</div>'; return; }
  const V=I.values, M=I.meta, F=I.front;
  let h='';

  h+='<div class="h2">输入定义：shape 是编译期固化的，值不一定有落盘</div>';
  h+='<p class="p">同一个 kernel 的 tile 形状、<code>alloc_tile</code> 次数、set/wait 次数<b>只由 shape 决定</b>，'+
     '与 <code>seq_lens</code> 的取值无关；而页数、分栈数、执行时间才依赖运行期的值。'+
     '所以先分清「哪些是形状、哪些是值」。</p>';

  /* ---- 1. 配置推断 ---- */
  h+='<div class="card"><div class="h2">这次跑的是什么（由 shape 反推）</div>';
  h+='<table><tbody>';
  const kv=(k,v,u)=>'<tr><td class="n">'+k+'</td><td>'+(v==null?'<span style="color:#8b939d">—</span>':v+(u||''))+'</td></tr>';
  h+=kv('平台 / 后端',esc(M.platform)+' / '+esc(M.backend||''));
  h+=kv('执行模式',(M.modes||[]).join(', ')||'—');
  h+=kv('batch（序列数）',V.batch,' 条序列');
  h+=kv('query token 数',V.query_tokens,' 个（每序列 1 个 → decode）');
  h+=kv('head / head_dim',(V.heads||'—')+' × '+(V.head_dim||'—'));
  h+=kv('kv_head / 每组',(V.kv_heads||'—')+' 个 / 组大小 '+(V.group||'—')+'（GQA）');
  h+=kv('hidden',V.hidden);
  h+=kv('SPMD 块',V.blocks+' AIC + '+V.aiv_total+' AIV','（split_aiv='+V.aiv_split+'）');
  h+=kv('任务数 = batch × kv_head',V.num_tasks,' 个任务摊到 '+V.blocks+' 块');
  h+=kv('页大小 / 分栈',V.page_size+' token/页 · '+V.stack_size+' token/栈');
  h+=kv('每序列最大页数',V.max_blocks_per_seq);
  h+=kv('cache 可容纳 token 行',V.cache_token_rows);
  h+='</tbody></table>';
  h+='<div class="note">只从 shape 就能确定这是 <b>decode</b>：<code>q_proj</code> 只有 '+V.query_tokens+
     ' 行、<code>out</code> 是 '+shp((M.params.find(p=>p.name==='out')||{}).shape)+
     '。所以核利用率的瓶颈在<b>搬 KV</b>而不是算 Q。</div></div>';

  /* ---- 2. 对外参数 ---- */
  h+='<div class="card"><div class="h2">对外 API 参数（'+V.n_api+' 个 · compiled_meta.json）</div>';
  h+='<table><thead><tr><th>#</th><th>参数</th><th>shape</th><th>dtype</th><th>方向</th><th>说明</th></tr></thead><tbody>';
  M.params.forEach((p,i)=>{
    h+='<tr><td class="n">'+(i+1)+'</td><td class="n">'+esc(p.name)+'</td><td class="n">'+shp(p.shape)+
       '</td><td>'+dt(p.dtype)+'</td><td>'+esc(p.direction||'—')+'</td><td>'+
       (GLOSS[p.name]?'<span style="color:#c8ced6">'+esc(GLOSS[p.name])+'</span>':'<span style="color:#8b939d">—</span>')+
       '</td></tr>';
  });
  h+='</tbody></table></div>';

  /* ---- 3. 内部 workspace ---- */
  const extra=[];
  I.kernel_names.forEach((n,i)=>{ if(n && I.api_names.indexOf(n)<0) extra.push({name:n,idx:i}); });
  h+='<div class="card"><div class="h2">kernel 独有实参（'+extra.length+' 个 · 用户不传）</div>';
  h+='<p class="p">对外 API 是 <b>'+V.n_api+'</b> 个参数，kernel 实际收到 <b>'+V.n_kernel+'</b> 个实参，差 '+V.n_workspace+' 个来自 runtime。'+
     '逐项拆：<code>layer_cache_base_token_rows</code> 是标量（不占指针实参，−1）、'+
     '<code>q_proj</code> 以展平后的 <code>q_tnd_flat</code> 传入（+1）、'+
     '再加 '+V.n_workspace+' 个 workspace = '+V.n_api+' − 1 + 1 + '+V.n_workspace+' = '+V.n_kernel+'。</p>';
  h+='<table><thead><tr><th>实参 idx</th><th>名称</th><th>shape</th><th>dtype</th><th>作用</th></tr></thead><tbody>';
  extra.forEach(e=>{
    const a=(I.deps.args||[]).find(x=>x.idx===e.idx)||{};
    h+='<tr><td class="n">'+e.idx+'</td><td class="n">'+esc(e.name)+'</td><td class="n">'+shp(a.shape)+
       '</td><td>'+dt(a.dtype)+'</td><td>'+(GLOSS[e.name]||'—')+'</td></tr>';
  });
  h+='</tbody></table></div>';

  /* ---- 4. 派生量 ---- */
  h+='<div class="card"><div class="h2">派生量：这些常数从哪来（'+F.derived.length+' 条）</div>';
  h+='<p class="p">全部原样摘自 <code>passes_dump/00_frontend.py</code>，所以能和后面的分块数对上。</p>';
  h+='<table><thead><tr><th>名字</th><th>源码表达式</th></tr></thead><tbody>';
  F.derived.forEach(d=>{
    h+='<tr><td class="n">'+esc(d.name)+'</td><td class="n" style="font-size:11px">'+esc(d.expr)+'</td></tr>';
  });
  h+='</tbody></table>';
  h+='<div class="note"><b>两条最关键的：</b>'+
     '<code>page_count = ceil(seq_len/'+V.page_size+')</code> → 页大小 '+V.page_size+'；'+
     '<code>stack_count = ceil(seq_len/'+V.stack_size+')</code> → KV 按 '+V.stack_size+' 分栈。'+
     '所以 <code>seq_lens=512,513</code> 会让两条序列分别落在 1 栈和 2 栈上 —— 这是负载不均的直接来源。</div></div>';

  /* ---- 5. 运行期 workload ---- */
  h+='<div class="card"><div class="h2">运行期 workload：谁有记录、谁没有</div>';
  if((I.runs||[]).length){
    I.runs.forEach(r=>{
      h+='<div class="h3">'+esc(r.date||r.bundle)+'</div>';
      h+='<div class="ib" style="grid-template-columns:84px 1fr"><span class="ibn">命令行</span>'+
         '<span style="font-family:var(--mono);font-size:10.5px;color:#c8ced6;word-break:break-all">'+esc(r.cmd||'未记录')+'</span></div>';
      h+='<div class="ib" style="grid-template-columns:84px 1fr"><span class="ibn">seq_lens</span>'+
         '<span style="font-family:var(--mono);font-size:10.5px;color:#c8ced6">'+esc(r.seq_lens||'未记录')+'</span></div>';
    });
  }
  h+='<div class="note bad"><b>20261015 那次没有命令行记录</b> —— 目录里既没有 log 也没有 README，'+
     '<code>debug/run.py</code> 里的 <code>seq_lens = randint(0,8,2)</code> 只是<b>重放时的随机默认值</b>，'+
     '不是那次 DFX 跑的真实输入。真正的值由 <code>test_paged_attention_pypto.py</code> 生成，那个脚本不在这个目录里。'+
     '要补齐得带 <code>--dump-args 2</code> 重跑一次。</div></div>';

  /* ---- 6. 静态 vs 运行期 ---- */
  h+='<div class="card"><div class="h2">哪些数字由输入决定</div>';
  h+='<table><thead><tr><th>数字</th><th>由什么决定</th><th>换输入会变吗</th></tr></thead><tbody>';
  const rw=(a,b,c)=>'<tr><td class="n">'+a+'</td><td>'+b+'</td><td>'+c+'</td></tr>';
  h+=rw('tile 形状 / <code>alloc_tile</code> 次数 / set·wait 次数','编译期静态，只跟 shape 有关','<span class="pill pa">不变</span>');
  h+=rw('片上缓冲峰值 / 复用槽位（6.4 / 2.0 / 1.0）','编译期地址分配结果','<span class="pill pa">不变</span>');
  h+=rw('page_count / stack_count','<code>seq_lens</code> 的值','<span class="pill pg">变</span>');
  h+=rw('每个 block 的循环轮数','<code>seq_lens</code> 的值','<span class="pill pg">变</span>');
  h+=rw('cycles / 带宽 / 核利用率 / trace 时长','运行期实测','<span class="pill pg">变</span>');
  h+='</tbody></table>';
  h+='<p class="p" style="margin:9px 0 0">换句话说：前面几页的 <b>tile 谱</b>和 <b>flag 配对表</b>换输入也不会变，'+
     '属于 kernel 的固有属性；而 <b>PMU 条带图</b>和 <b>trace 时间线</b>只对当次输入成立。</p></div>';

  $('#inputsbody').innerHTML=h;
}


/* ============ 启动 ============ */
$('#showgrp').checked=true;
render();
buildCore();
buildIncore();
buildProfile();
buildTask();
buildInputs();
</script>
</body>
</html>
"""

open(OUT, "w", encoding="utf-8").write(HTML.replace("__DATA__", data_js))
print("wrote", OUT, os.path.getsize(OUT), "bytes")
print("lanes:", len(lane_list), "flows:", len(flows), "cores:", len(cores), "tasks:", len(DEPS["tasks"]))
