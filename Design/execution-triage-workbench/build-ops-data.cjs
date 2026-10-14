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

const DEV_PID = '2733163936'; // 设备进程（Ascend Hardware）
const MAIN_TID = '46';        // 主执行流：模型的算子按序在这条流上跑
const N_DETAIL = 3;           // 详细给出任务序列的 step 数（最快 / 中位 / 最慢）
const N_TOP = 18;             // 算子榜长度

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
        if (m[2] !== DEV_PID) continue;
        const dur = Number(m[5]);
        streams[m[3]] = streams[m[3]] || { n: 0, ms: 0 };
        streams[m[3]].n += 1; streams[m[3]].ms += dur / 1000;
        if (m[3] !== MAIN_TID) continue;
        const type = (/"Task Type": "([^"]*)"/.exec(m[6]) || [, ''])[1];
        const key = `${m[1]}|${type}`;
        let i = nameIdx.get(key);
        if (i === undefined) { i = names.length; names.push({ name: m[1], type }); nameIdx.set(key, i); }
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
  console.log(`bytes ${fs.statSync(OUT).size}，用时 ${((Date.now() - t0) / 1000).toFixed(1)} s`);
})().catch((e) => { console.error(e); process.exit(1); });
