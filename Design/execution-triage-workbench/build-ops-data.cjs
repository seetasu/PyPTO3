#!/usr/bin/env node
/* 算子级下钻的数据：从 GBS256 压测包里的 Ascend profiler trace 里，取设备主执行流
 * （pid 2733163936 / tid 46）上的任务，按 MODEL_EXECUTE 切成一个个 decode step。
 *
 * 口径与出处见 README.md「算子级下钻」。三件事必须说清楚：
 *   1. 这份 trace 来自 GBS256 压测，**与单请求页的 GBS1 STRACE 不是同一次采集**；
 *   2. trace 只采了 **1 张卡**（设备侧只有一个 device 进程、通信任务的 src rank 全为 0），
 *      所以它是「一张卡上一个 decode step 长什么样」的实例，不是本请求 dev 6 的实测；
 *   3. 时间全部按 trace 原值给出，没有缩放到 dev 6 的 runner_run 窗口。
 *
 * 用法：node Design/execution-triage-workbench/build-ops-data.cjs
 * 只读压缩包里的 profiler trace 条目，流式解压，不落盘、不整包解压。 */
'use strict';

const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

const ROOT = path.resolve(__dirname, '..', '..');
const ZIP = path.join(ROOT, 'Data', 'dsv4-flash-dspark-a3-dp4tp4ep16-gbs256-in256o64.zip');
const MEMBER = 'dsv4-flash-dspark-a3-dp4tp4ep16-gbs256-in256o64/dsv4-flash-dspark-a3-dp4tp4ep16-gbs256-in256o64.json';
const OUT = path.join(__dirname, 'ops-data.js');

const DEV_PID = '2733163936';  // 设备进程（Ascend Hardware）
const HCCL_PID = '2733164000'; // 通信进程（HCCL）
const MAIN_TID = '46';         // 主执行流：模型的算子按序在这条流上跑
const N_TOP = 18;              // 侧栏算子榜长度
const OUT_DEV = path.join(__dirname, 'device-data.js');
/* 详情页的多流对齐里另外画这几条：两条次执行流、三条同步队列、两条通信流。
 * 它们只保留「什么时候忙」，不保留任务名——详情页用它们回答「主流在等的时候别人在干嘛」。 */
const SIDE = [
  { key: '38', label: '执行流 38', role: 'exec' },
  { key: '37', label: '执行流 37', role: 'exec' },
  { key: '142', label: '同步队列 142', role: 'sync' },
  { key: '143', label: '同步队列 143', role: 'sync' },
  { key: '182', label: '同步队列 182', role: 'sync' },
  { key: 'hccl/11', label: '通信 alltoallv', role: 'comm' },
  { key: 'hccl/1', label: '通信 allgather', role: 'comm' },
];
const SIDE_TIDS = new Set(SIDE.map((s) => s.key));

const r2 = (v) => Math.round(v * 100) / 100;
const r3 = (v) => Math.round(v * 1000) / 1000;

/* ------------------------------------------------------------ 读 zip 条目 */
/* 顺着末尾的中央目录找到条目的本地头，再流式 inflate——这份 trace 解压后 1.38 GB，
 * 不能整个读进内存。 */
function entryInfo(zipPath, name) {
  const fd = fs.openSync(zipPath, 'r');
  try {
    const size = fs.fstatSync(fd).size;
    const tailLen = Math.min(size, 66560);
    const tail = Buffer.alloc(tailLen);
    fs.readSync(fd, tail, 0, tailLen, size - tailLen);
    const eocdOff = tail.lastIndexOf(Buffer.from([0x50, 0x4b, 0x05, 0x06]));
    if (eocdOff < 0) throw new Error('找不到 ZIP 中央目录（EOCD）');
    let cdSize = tail.readUInt32LE(eocdOff + 12);
    let cdOff = tail.readUInt32LE(eocdOff + 16);
    if (cdOff === 0xffffffff || cdSize === 0xffffffff) {
      const locOff = tail.lastIndexOf(Buffer.from([0x50, 0x4b, 0x06, 0x07]));
      const z64 = Buffer.alloc(56);
      fs.readSync(fd, z64, 0, 56, Number(tail.readBigUInt64LE(locOff + 8)));
      cdSize = Number(z64.readBigUInt64LE(40));
      cdOff = Number(z64.readBigUInt64LE(48));
    }
    const cd = Buffer.alloc(cdSize);
    fs.readSync(fd, cd, 0, cdSize, cdOff);
    for (let p = 0; p + 46 <= cd.length && cd.readUInt32LE(p) === 0x02014b50;) {
      const nLen = cd.readUInt16LE(p + 28);
      const eLen = cd.readUInt16LE(p + 30);
      const cLen = cd.readUInt16LE(p + 32);
      if (cd.toString('utf8', p + 46, p + 46 + nLen) === name) {
        const method = cd.readUInt16LE(p + 10);
        let comp = cd.readUInt32LE(p + 20);
        let unc = cd.readUInt32LE(p + 24);
        let localOff = cd.readUInt32LE(p + 42);
        if (comp === 0xffffffff || unc === 0xffffffff || localOff === 0xffffffff) {
          const ex = cd.subarray(p + 46 + nLen, p + 46 + nLen + eLen);
          for (let q = 0; q + 4 <= ex.length;) {
            const id = ex.readUInt16LE(q); const len = ex.readUInt16LE(q + 2);
            if (id === 0x0001) {
              let o = q + 4;
              if (unc === 0xffffffff) { unc = Number(ex.readBigUInt64LE(o)); o += 8; }
              if (comp === 0xffffffff) { comp = Number(ex.readBigUInt64LE(o)); o += 8; }
              if (localOff === 0xffffffff) localOff = Number(ex.readBigUInt64LE(o));
            }
            q += 4 + len;
          }
        }
        const lh = Buffer.alloc(30);
        fs.readSync(fd, lh, 0, 30, localOff);
        return { method, comp, unc, dataOff: localOff + 30 + lh.readUInt16LE(26) + lh.readUInt16LE(28) };
      }
      p += 46 + nLen + eLen + cLen;
    }
    throw new Error(`压缩包里没有 ${name}`);
  } finally {
    fs.closeSync(fd);
  }
}

/* 设备侧任务事件的形状是固定的，直接用正则扫原文，比逐个 JSON.parse 快两个数量级：
 * {"name": …, "pid": …, "tid": …, "ts": "…", "dur": …, "ph": "X", "args": {…}} */
const RE_TASK = /\{"name": "([^"]*)", "pid": (\d+), "tid": (\d+), "ts": "([\d.]+)", "dur": ([\d.eE+-]+), "ph": "X", "args": \{([^{}]*)\}\}/g;
const RE_PROC = /\{"ph": "M", "name": "process_name", "pid": (\d+), "tid": \d+, "args": \{"name": "([^"]*)"\}\}/g;

function scanTrace() {
  return new Promise((resolve, reject) => {
    const info = entryInfo(ZIP, MEMBER);
    const rs = fs.createReadStream(ZIP, { start: info.dataOff, end: info.dataOff + info.comp - 1, highWaterMark: 1 << 22 });
    const inf = zlib.createInflateRaw({ chunkSize: 1 << 22 });
    const names = []; const nameIdx = new Map(); const tasks = [];
    const procName = {}; const streams = {};
    let tail = '';
    inf.on('error', reject);
    inf.on('data', (c) => {
      const s = tail + c.toString('utf8');
      let m; let last = 0;
      RE_PROC.lastIndex = 0;
      while ((m = RE_PROC.exec(s))) procName[m[1]] = m[2];
      RE_TASK.lastIndex = 0;
      while ((m = RE_TASK.exec(s))) {
        last = m.index + m[0].length;
        const inDev = m[2] === DEV_PID;
        const inHccl = m[2] === HCCL_PID;
        if (!inDev && !inHccl) continue;
        const dur = Number(m[5]);
        // 旁流只留下「什么时候忙」，用于详情页的多流对齐；主执行流留完整任务
        const key = inHccl ? `hccl/${m[3]}` : m[3];
        streams[key] = streams[key] || { n: 0, ms: 0, busy: [] };
        streams[key].n += 1; streams[key].ms += dur / 1000;
        if (SIDE_TIDS.has(key)) streams[key].busy.push([Number(m[4]), dur / 1000]);
        if (!inDev || m[3] !== MAIN_TID) continue;
        const type = (/"Task Type": "([^"]*)"/.exec(m[6]) || [, ''])[1];
        const nk = `${m[1]}|${type}`;
        let i = nameIdx.get(nk);
        if (i === undefined) { i = names.length; names.push({ name: m[1], type }); nameIdx.set(nk, i); }
        tasks.push([Number(m[4]), dur / 1000, i]);
      }
      tail = s.slice(Math.max(0, last));
      if (tail.length > (1 << 20)) tail = tail.slice(-(1 << 16));
    });
    inf.on('end', () => { tasks.sort((a, b) => a[0] - b[0]); resolve({ names, tasks, procName, streams }); });
    rs.pipe(inf);
  });
}

/* ------------------------------------------------------------ 算子分组
 * 按名字归到几类可读的语义分组；顺序即优先级，先匹配到的赢。
 * 分组只影响配色与小结，表里仍然给 trace 里的原始算子名。 */
const GROUPS = [
  { key: 'marker', label: '执行标记', re: /^(MODEL_EXECUTE|EVENT_RECORD|PLACE_HOLDER)/ },
  { key: 'wait', label: '等待同步', re: /WAIT/ },
  { key: 'ep', label: 'EP 通信', re: /MoeDistribute|Hc(Pre|Post)|Compressor|hcom|alltoall|allgather|reduce_scatter|allReduce/i },
  { key: 'moe', label: 'MoE 专家', re: /GroupedMatmul|Swiglu|MoeToken|MoeGating|MoeInit|MoeFinalize/i },
  { key: 'attn', label: '注意力 · KV', re: /Attn|Attention|Indexer|slot_mapping|Rotary/i },
  { key: 'mm', label: '矩阵乘 · 量化', re: /Matmul|MatMul|Quant|RmsNorm|Softmax/i },
  { key: 'sample', label: '采样 · 投机', re: /ArgMax|sample_|Uniform|Random|dflash|dspark/i },
  { key: 'mem', label: '访存 · 搬运', re: /MEMCPY|Copy|Transpose|TensorMove|ViewCopy|Pad|Fill|MemSet|Slice|Concat|ConcatD|Stack|Pack|Gather|Scatter|Embedding|Index/i },
  { key: 'misc', label: '其他', re: /./ },
];
const groupOf = (name) => GROUPS.find((g) => g.re.test(name)).key;

function quant(sorted, p) {
  const i = (sorted.length - 1) * p;
  const lo = Math.floor(i); const hi = Math.ceil(i);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
}
const stat = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  return { mean: r2(s.reduce((a, b) => a + b, 0) / s.length), p50: r2(quant(s, 0.5)), p90: r2(quant(s, 0.9)), max: r2(s[s.length - 1]) };
};

(async () => {
  const t0 = Date.now();
  const { names, tasks, procName, streams } = await scanTrace();
  if (!tasks.length) throw new Error('主执行流上没有任务，检查 pid / tid');

  // MODEL_EXECUTE 是一次模型执行的起点，正好用来切 step
  const marks = [];
  tasks.forEach((t, i) => { if (names[t[2]].type === 'MODEL_EXECUTE') marks.push(i); });
  if (marks.length < 3) throw new Error('没找到 MODEL_EXECUTE 标记');

  const segs = [];
  for (let k = 0; k < marks.length - 1; k += 1) {
    const i0 = marks[k]; const i1 = marks[k + 1];
    const t0ms = tasks[i0][0] / 1000;
    const ms = tasks[i1][0] / 1000 - t0ms;
    // 执行标记（MODEL_EXECUTE 等）不是算子，不计入“计算”
    let wait = 0; let compute = 0; let busy = 0; let marker = 0;
    for (let i = i0; i < i1; i += 1) {
      const d = tasks[i][1]; const g = groupOf(names[tasks[i][2]].name);
      busy += d;
      if (g === 'wait') wait += d;
      else if (g === 'marker') marker += d;
      else compute += d;
    }
    segs.push({ k, i0, i1, t0: t0ms, ms: r2(ms), wait: r2(wait), compute: r2(compute), marker: r2(marker), idle: r2(Math.max(0, ms - busy)), n: i1 - i0 });
  }

  // 详细给出三步：最快 / 中位 / 最慢
  const byMs = segs.slice().sort((a, b) => a.ms - b.ms);
  const pick = [byMs[0], byMs[Math.floor(byMs.length / 2)], byMs[byMs.length - 1]];
  const LABEL = ['最快', '中位', '最慢'];

  const detail = pick.map((s, pi) => {
    const slice = tasks.slice(s.i0, s.i1);
    const byOp = new Map();
    const byGroup = new Map();
    slice.forEach(([, d, ni]) => {
      const { name, type } = names[ni];
      const g = groupOf(name);
      const o = byOp.get(name) || { name, type, group: g, n: 0, ms: 0 };
      o.n += 1; o.ms += d; byOp.set(name, o);
      const gg = byGroup.get(g) || { key: g, n: 0, ms: 0 };
      gg.n += 1; gg.ms += d; byGroup.set(g, gg);
    });
    const ops = [...byOp.values()].sort((a, b) => b.ms - a.ms).map((o) => ({ ...o, ms: r3(o.ms) }));
    const groups = GROUPS.map((g) => byGroup.get(g.key)).filter(Boolean)
      .map((g) => ({ key: g.key, n: g.n, ms: r2(g.ms) })).sort((a, b) => b.ms - a.ms);
    return {
      id: `S${s.k + 1}`,
      label: LABEL[pi],
      ms: s.ms,
      wait: s.wait,
      compute: s.compute,
      // 标记本身不是算子，并到“流上空档”里，免得把 0.8 ms 算进计算
      idle: r2(s.idle + s.marker),
      nTask: s.n,
      nOp: ops.length,
      groups,
      ops: ops.slice(0, N_TOP),
      // 画时间带用：[起点 ms, 时长 ms, 分组下标]，起点相对这一步
      tasks: slice.map(([t, d, ni]) => [r3(t / 1000 - s.t0), r3(d), GROUPS.findIndex((g) => g.key === groupOf(names[ni].name))]),
    };
  });

  const topStreams = Object.entries(streams).sort((a, b) => b[1].ms - a[1].ms)
    .slice(0, 6).map(([tid, v]) => ({ tid, n: v.n, ms: Math.round(v.ms) }));

  const out = {
    source: {
      zip: 'Data/dsv4-flash-dspark-a3-dp4tp4ep16-gbs256-in256o64.zip',
      member: MEMBER.split('/').pop(),
      run: 'GBS256 压测（256 并发 · 输入 256 / 输出 64）',
      sameRunAs: 'overview.html（压测总览）',
      notSameRunAs: 'index.html（单请求 GBS1 STRACE）',
      rank: '只采了 1 张卡',
      stream: `${procName[DEV_PID] || '设备进程'} · pid ${DEV_PID} / stream ${MAIN_TID}`,
      streamNote: `设备进程共 ${Object.keys(streams).length} 条流，这里只画主执行流；通信与 SQE 流不在其中`,
      clock: 'trace 原始时间，未缩放到 runner_run 窗口',
      profiler: '采集时 profiler 开着，绝对耗时偏大',
      // 采集口径：负载串取自压缩包自己的名字，不另写死
      workload: path.basename(ZIP, '.zip'),
      capture: 'Ascend profiler 导出的 trace（与 result3.json 同一个压缩包）',
      captureNote: 'pypto-serving 的 .agents/skills 里没有这份 Ascend 导出的采集与解析流程（它们吃的是自家 SA_PROFILE 的 Chrome trace 与 [STRACE] 日志），所以这一页按导出原值读。',
      // 算子级分析本身是有 skill 口径的，只是作用在另一种 trace 上
      analysis: 'qwen3-14b-online-perf-test §6–7 给的就是算子级口径：ph="X" 按事件名聚合，kernel 行取 args.device_wall_us（才是设备真实耗时，span dur 含 host 下发开销），并排除一次性启动 span。换到 pypto-serving 自己的 profiler 上复现这一页时按它来；端到端性能数另用 vllm-bench-perf。',
    },
    nStep: segs.length,
    period: stat(segs.map((s) => s.ms)),
    waitShare: r2((segs.reduce((a, s) => a + s.wait, 0) / segs.reduce((a, s) => a + s.ms, 0)) * 100),
    computeMs: stat(segs.map((s) => s.compute)),
    groups: GROUPS.map((g) => ({ key: g.key, label: g.label })),
    steps: detail,
    streams: topStreams,
  };

  fs.writeFileSync(OUT, `/* Generated by build-ops-data.cjs — do not edit by hand. */\nwindow.TRIAGE_OPS = ${JSON.stringify(out)};\n`);
  console.log(`主执行流任务 ${tasks.length}，切出 ${segs.length} 个 step；步长 P50 ${out.period.p50} ms，等待占 ${out.waitShare}%`);
  detail.forEach((d) => console.log(`  ${d.label} ${d.id}：${d.ms} ms（等待 ${d.wait} · 计算 ${d.compute} · 空档 ${d.idle}）${d.nTask} 个任务 / ${d.nOp} 种算子`));
  console.log(`ops-data.js ${fs.statSync(OUT).size} bytes`);

  /* ---------------------------------------------------------- 设备详情页的数据
   * 侧栏只给三步的小结；详情页要能逐步翻、能排算子、能看某个算子在 66 步里的变化，
   * 所以这里给全部 66 步：每步的任务序列（去掉零时长的 EVENT_RECORD 标记）、
   * 旁流的忙碌区间、以及算子目录。时间一律是 trace 原值。 */
  const ZERO_MARK = new Set(['EVENT_RECORD', 'PLACE_HOLDER_SQE']);
  const catalog = names.map((n) => ({ name: n.name, type: n.type, group: groupOf(n.name) }));
  const gIdx = Object.fromEntries(GROUPS.map((g, i) => [g.key, i]));

  // 旁流：把忙碌区间按 step 切开，详情页用它回答「主流在等的时候别人在干嘛」
  const sideByStep = SIDE.map((s) => {
    const busy = (streams[s.key] && streams[s.key].busy ? streams[s.key].busy : []).sort((a, b) => a[0] - b[0]);
    let p = 0;
    const perStep = segs.map((seg) => {
      const t0us = tasks[seg.i0][0]; const t1us = t0us + seg.ms * 1000;
      while (p < busy.length && busy[p][0] < t0us) p += 1;
      const spans = [];
      let ms = 0;
      for (let q = p; q < busy.length && busy[q][0] < t1us; q += 1) {
        const rel = (busy[q][0] - t0us) / 1000;
        ms += busy[q][1];
        if (busy[q][1] >= 0.01) spans.push([r3(rel), r3(busy[q][1])]);
      }
      return { ms: r2(ms), spans };
    });
    return { ...s, n: streams[s.key] ? streams[s.key].n : 0, perStep };
  });

  /* 任务分两份存：
   *   ops   —— 按算子汇总，用**全部**任务算，所以次数和合计是准的；
   *   tasks —— 只留 ≥ DRAW_MIN 的，用来画甘特。26,672 个 < 4 µs 的任务加起来只有
   *            38 ms（忙时的 0.68%），画出来不到一个像素，留着只会让文件翻倍。 */
  const DRAW_MIN = 0.004;
  let nDropped = 0; let msDropped = 0;
  const devSteps = segs.map((s) => {
    const slice = tasks.slice(s.i0, s.i1);
    const byGroup = new Map();
    const byOp = new Map();
    slice.forEach(([, d, ni]) => {
      const g = groupOf(names[ni].name);
      byGroup.set(g, (byGroup.get(g) || 0) + d);
      const o = byOp.get(ni) || [ni, 0, 0];
      o[1] += 1; o[2] += d; byOp.set(ni, o);
    });
    const drawn = slice.filter(([, d, ni]) => {
      const keep = d >= DRAW_MIN && !(d === 0 && ZERO_MARK.has(names[ni].type));
      if (!keep) { nDropped += 1; msDropped += d; }
      return keep;
    });
    return {
      id: `S${s.k + 1}`,
      t0: r2(s.t0 - segs[0].t0),            // 相对第一步的起点，用于 66 步总览
      ms: s.ms,
      wait: s.wait,
      compute: s.compute,
      idle: r2(s.idle + s.marker),
      nTask: s.n,
      groups: GROUPS.map((g) => r3(byGroup.get(g.key) || 0)),
      // [算子目录下标, 次数, 合计 ms]——按全部任务算
      ops: [...byOp.values()].map((o) => [o[0], o[1], r3(o[2])]).sort((a, b) => b[2] - a[2]),
      // [起点 ms, 时长 ms, 算子目录下标]——只含画得出来的那些
      tasks: drawn.map(([t, d, ni]) => [r3(t / 1000 - s.t0), r3(d), ni]),
    };
  });

  const dev = {
    source: { ...out.source, streamNote: `设备进程共 ${Object.keys(streams).filter((k) => !k.startsWith('hccl/')).length} 条流；详情页画主执行流 + ${SIDE.length} 条旁流（只画忙碌区间）` },
    nStep: segs.length,
    period: out.period,
    waitShare: out.waitShare,
    computeMs: out.computeMs,
    groups: GROUPS.map((g, i) => ({ key: g.key, label: g.label, i })),
    gIdx,
    catalog,
    steps: devSteps,
    side: sideByStep,
    defaultStep: detail[1].id,
    drawMin: DRAW_MIN,
    dropped: { n: nDropped, ms: r2(msDropped) },
  };
  fs.writeFileSync(OUT_DEV, `/* Generated by build-ops-data.cjs — do not edit by hand. */\nwindow.TRIAGE_DEVICE = ${JSON.stringify(dev)};\n`);
  console.log(`device-data.js ${fs.statSync(OUT_DEV).size} bytes（${devSteps.length} 步 · ${devSteps.reduce((a, s) => a + s.tasks.length, 0)} 个任务 · ${catalog.length} 种算子 · ${SIDE.length} 条旁流）`);
  console.log(`用时 ${((Date.now() - t0) / 1000).toFixed(1)} s`);
})().catch((e) => { console.error(e); process.exit(1); });
