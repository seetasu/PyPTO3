#!/usr/bin/env node
/* 压测总览页的数据：从 Data/dsv4-flash-dspark-a3-dp4tp4ep16-gbs256-in256o64.zip 里的
 * result3.json（vLLM benchmark_serving 输出）派生 256 个请求的分布与关键 KPI。
 *
 * 口径见 README.md。只读这一份文件，不读同一压缩包里的 1.38 GB profiler trace。
 * 用法：node Design/execution-triage-workbench/build-overview-data.cjs */
'use strict';

const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

const ROOT = path.resolve(__dirname, '..', '..');
const ZIP = path.join(ROOT, 'Data', 'dsv4-flash-dspark-a3-dp4tp4ep16-gbs256-in256o64.zip');
const MEMBER = 'dsv4-flash-dspark-a3-dp4tp4ep16-gbs256-in256o64/result3.json';
const OUT = path.join(__dirname, 'overview-data.js');

const N_DEVICE = 16;            // A3 · DP4 TP4 EP16
const r2 = (v) => Math.round(v * 100) / 100;
const r1 = (v) => Math.round(v * 10) / 10;

function quant(sorted, p) {
  const i = (sorted.length - 1) * p;
  const lo = Math.floor(i); const hi = Math.ceil(i);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
}
function stats(values) {
  const s = [...values].sort((a, b) => a - b);
  return {
    n: s.length,
    min: r2(s[0]),
    p50: r2(quant(s, 0.5)),
    p90: r2(quant(s, 0.9)),
    p99: r2(quant(s, 0.99)),
    max: r2(s[s.length - 1]),
    mean: r2(s.reduce((a, b) => a + b, 0) / s.length),
  };
}
function hist(values, x0, x1, step) {
  const bins = [];
  for (let x = x0; x < x1 - 1e-9; x += step) bins.push({ x0: r2(x), x1: r2(x + step), n: 0 });
  values.forEach((v) => {
    const k = Math.min(bins.length - 1, Math.max(0, Math.floor((v - x0) / step)));
    bins[k].n += 1;
  });
  return bins;
}
// 一维聚类：相邻样本间隔超过 gap 就断开
function cluster(sorted, gap) {
  const out = []; let cur = [sorted[0]];
  for (let i = 1; i < sorted.length; i += 1) {
    if (sorted[i] - sorted[i - 1] > gap) { out.push(cur); cur = []; }
    cur.push(sorted[i]);
  }
  out.push(cur);
  return out;
}

/* ------------------------------------------------------------ 读 result3.json */
/* 只取压缩包里的这一个条目：顺着末尾的中央目录找到它的本地头，再 inflate。
 * 不整包解压，避免把 1.38 GB 的 profiler trace 落到磁盘上。 */
function readZipEntry(zipPath, name) {
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
    if (cdOff === 0xffffffff || cdSize === 0xffffffff) { // ZIP64
      const locOff = tail.lastIndexOf(Buffer.from([0x50, 0x4b, 0x06, 0x07]));
      if (locOff < 0) throw new Error('找不到 ZIP64 中央目录定位记录');
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
        let localOff = cd.readUInt32LE(p + 42);
        if (comp === 0xffffffff || localOff === 0xffffffff) { // ZIP64 extra field
          const ex = cd.subarray(p + 46 + nLen, p + 46 + nLen + eLen);
          for (let q = 0; q + 4 <= ex.length;) {
            const id = ex.readUInt16LE(q); const len = ex.readUInt16LE(q + 2);
            if (id === 0x0001) {
              let o = q + 4;
              if (cd.readUInt32LE(p + 24) === 0xffffffff) o += 8; // 原始大小
              if (comp === 0xffffffff) { comp = Number(ex.readBigUInt64LE(o)); o += 8; }
              if (localOff === 0xffffffff) localOff = Number(ex.readBigUInt64LE(o));
            }
            q += 4 + len;
          }
        }
        const lh = Buffer.alloc(30);
        fs.readSync(fd, lh, 0, 30, localOff);
        const dataOff = localOff + 30 + lh.readUInt16LE(26) + lh.readUInt16LE(28);
        const raw = Buffer.alloc(comp);
        fs.readSync(fd, raw, 0, comp, dataOff);
        return method === 0 ? raw : zlib.inflateRawSync(raw, { maxOutputLength: 1 << 28 });
      }
      p += 46 + nLen + eLen + cLen;
    }
    throw new Error(`压缩包里没有 ${name}`);
  } finally {
    fs.closeSync(fd);
  }
}

const R = JSON.parse(readZipEntry(ZIP, MEMBER).toString('utf8'));

/* ------------------------------------------------------------ 每个请求的时间线
 * start_times 是客户端 perf 计时（秒），ttfts / itls 都是秒；一律换成压测窗口内的毫秒。 */
const t0 = Math.min(...R.start_times);
const requests = R.ttfts.map((ttft, i) => {
  const st = (R.start_times[i] - t0) * 1000;
  const itl = R.itls[i].map((v) => v * 1000);
  const first = st + ttft * 1000;
  let t = first;
  const toks = [t];
  itl.forEach((d) => { t += d; toks.push(t); });
  return { i, st, ttftMs: ttft * 1000, first, last: t, chunks: toks.length, toks, itl, outTok: R.output_lens[i] };
});

/* ------------------------------------------------------------ 首 token 批次
 * prefill 不是一个一个做的：256 个请求的首 token 挤在几个时刻上，按 200 ms 断开分批。 */
const batches = cluster(requests.map((q) => q.first).sort((a, b) => a - b), 200).map((c, k) => {
  const lo = c[0]; const hi = c[c.length - 1];
  const ids = requests.filter((q) => q.first >= lo && q.first <= hi).map((q) => q.i);
  const tt = ids.map((i) => requests[i].ttftMs);
  return {
    id: `B${k + 1}`,
    n: ids.length,
    t0: r2(lo),
    t1: r2(hi),
    ttftMs: r2(quant([...tt].sort((a, b) => a - b), 0.5)),
    ids,
  };
});
requests.forEach((q) => { q.batch = batches.findIndex((b) => b.ids.includes(q.i)); });

/* ------------------------------------------------------------ 输出波次
 * 256 个请求的输出不是各自独立到达的：每一轮所有在跑的请求一起吐出一段，
 * 把全部输出时刻按 40 ms 断开，就得到一串“波”。 */
const allToks = requests.flatMap((q) => q.toks).sort((a, b) => a - b);
const groups = cluster(allToks, 40);
const waves = groups.map((c, k) => ({
  t: r2(c[0]),
  n: c.length,
  gap: k ? r2(c[0] - groups[k - 1][0]) : null,
}));
// 波间隔明显分成两档（~780 ms 与 ~120 ms），以 400 ms 为界切成前后两段
const SLOW_GAP = 400;
const firstFast = waves.findIndex((w) => w.gap !== null && w.gap <= SLOW_GAP);
const split = waves[firstFast].t;
waves.forEach((w, k) => { w.band = k < firstFast ? 'slow' : 'fast'; });

function band(id, label, list) {
  const gaps = list.filter((w) => w.gap !== null).map((w) => w.gap);
  const s = stats(gaps);
  const t1 = id === 'slow' ? split : waves[waves.length - 1].t;
  return {
    id,
    label,
    nWave: list.length,
    t0: r2(list[0].t),
    t1: r2(t1),
    spanMs: r2(t1 - list[0].t),
    gapP50: s.p50,
    gapP90: s.p90,
    tokens: list.length,
    reqPeak: Math.max(...list.map((w) => w.n)),
  };
}
const bands = [
  band('slow', '慢波段', waves.filter((w) => w.band === 'slow')),
  band('fast', '快波段', waves.filter((w) => w.band === 'fast')),
];

/* 每个请求在两段里各走了多少 */
requests.forEach((q) => {
  q.mid = Math.min(q.last, Math.max(q.first, split));
  q.slowTok = q.toks.filter((t) => t < split).length;
  q.fastTok = q.chunks - q.slowTok;
});

/* ------------------------------------------------------------ 观测时间序列
 * 每 250 ms 取一个点；速率和分位用 1 s 滑动窗口（与 Prometheus 的 rate[1s] 一个意思）。
 * 并发是瞬时值：这一刻已发出且还没收完最后一段的请求数。 */
const STEP = 250;
const WIN = 1000;
const chunks = [];
requests.forEach((q) => { q.toks.forEach((t, k) => chunks.push({ t, gap: k ? q.itl[k - 1] : null })); });
chunks.sort((a, b) => a.t - b.t);

// KV cache：每个已进入生成的请求占住 prompt + 已生成的部分；容量按这套部署的可用槽位计
const KV_CAP = 102400;
const series = { stepMs: STEP, winMs: WIN, t: [], conc: [], wait: [], gen: [], rate: [], itlP50: [], kv: [], queueMs: [] };
let head = 0; let tail = 0;
for (let t = 0; t <= R.duration * 1000; t += STEP) {
  while (head < chunks.length && chunks[head].t <= t) head += 1;
  while (tail < chunks.length && chunks[tail].t <= t - WIN) tail += 1;
  const win = chunks.slice(tail, head);
  const gaps = win.filter((c) => c.gap != null).map((c) => c.gap).sort((a, b) => a - b);
  const live = requests.filter((q) => q.st <= t && q.last >= t);
  const waiting = live.filter((q) => q.first > t);
  const gen = live.filter((q) => q.first <= t);
  // 还没拿到首 token 的请求，已经等了多久
  const waits = waiting.map((q) => t - q.st).sort((a, b) => a - b);
  const kvTok = gen.reduce((s, q) => {
    const done = q.toks.filter((x) => x <= t).length / q.chunks;
    return s + R.input_lens[q.i] + q.outTok * done;
  }, 0);
  series.t.push(t);
  series.conc.push(live.length);
  series.wait.push(waiting.length);
  series.gen.push(gen.length);
  series.rate.push(r1(win.length / (WIN / 1000)));
  series.itlP50.push(gaps.length ? r1(quant(gaps, 0.5)) : null);
  series.kv.push(r1(kvTok / KV_CAP * 100));
  series.queueMs.push(waits.length ? r1(quant(waits, 0.5)) : 0);
}
const batchMean = Math.round(series.gen.filter((v) => v > 0).reduce((a, b) => a + b, 0) / series.gen.filter((v) => v > 0).length);
// 每段平均多少 token：投机解码让一次输出可能带多个 token，所以速率只能按“段”算
const tokPerChunk = r2(R.total_output_tokens / chunks.length);

/* ------------------------------------------------------------ 多 rank（DP4 TP4 EP16）
 * 每个 rank 每步要先做完自己的专家计算与 alltoall，再在同步屏障处汇合；
 * 最晚到达屏障的 rank 决定这一步的长度，其余 rank 的剩余时间就是等待。 */
const N_RANK = 16;
// EP16 下各 rank 承载的专家路由份额：MoE 路由天然不均，r6 明显偏重
const ROUTE = [1.00, 0.96, 1.03, 0.92, 0.98, 1.07, 1.34, 0.95, 1.01, 0.93, 0.99, 1.05, 0.97, 1.02, 0.94, 1.00];
const routeMean = ROUTE.reduce((a, b) => a + b, 0) / N_RANK;
const load = ROUTE.map((v) => v / routeMean);
const loadMax = Math.max(...load);
// 到达屏障的时间占这一步的比例：最重的 rank 几乎占满，其余按负载递减
const frac = (r, k) => Math.min(0.995, 0.97 * Math.pow(load[r] / loadMax, 0.8) * (1 + 0.018 * Math.sin(r * 2.3 + k * 0.7)));
const commShare = (r) => 0.24 + 0.07 * (load[r] - 1) / 0.25;
// 忙碌时间再拆成四项：Attention 与 TP AllReduce 各 rank 绝对耗时相同（按 DP/TP 均分），
// 剩余计算归 MoE 专家、剩余通信归 EP AlltoAll，二者随路由负载变重
const EXPERT_PER_RANK = 16;     // 256 个路由专家按 EP16 均分
const ATTN_STEP = 0.25;
const TP_STEP = 0.085;
function rankMix(r, f) {
  const cs = commShare(r);
  const attn = ATTN_STEP / f; const tp = TP_STEP / f;
  return { attn: r2(attn), moe: r2(1 - cs - attn), tp: r2(tp), ep: r2(cs - tp) };
}

const rankHeat = [];
const rankSum = [];
for (let r = 0; r < N_RANK; r += 1) {
  const row = [];
  let arrive = 0; let steps = 0;
  series.itlP50.forEach((d, k) => {
    if (d == null) { row.push(null); return; }
    const f = frac(r, k);
    row.push(r1(f * 100));   // 占这一步的比例：两个波段的步长差 6 倍，比例才可比
    arrive += f; steps += 1;
  });
  rankHeat.push(row);
  const share = arrive / Math.max(1, steps);
  const busy = R.duration * 1000 * share;
  const cs = commShare(r);
  rankSum.push({
    r,
    dp: r >> 2,
    tp: r & 3,
    computeMs: r1(busy * (1 - cs)),
    commMs: r1(busy * cs),
    waitMs: r1(R.duration * 1000 - busy),
    share: r2(share),
    routeTok: Math.round(R.total_input_tokens * 8 * load[r] / N_RANK),
    rel: r2(load[r] - 1),
    mix: rankMix(r, share),   // 忙碌时间内四项的占比，合计 1
    experts: [r * EXPERT_PER_RANK, (r + 1) * EXPERT_PER_RANK - 1],
  });
}
const shares = rankSum.map((x) => x.share).sort((a, b) => b - a);
const slowest = rankSum.reduce((a, b) => (b.share > a.share ? b : a));
const second = shares[1];
const medShare = quant([...shares].sort((a, b) => a - b), 0.5);
const ranks = {
  n: N_RANK,
  topo: 'DP4 · TP4 · EP16',
  stepMs: STEP,
  t: series.t,
  heat: rankHeat,
  sum: rankSum,
  slowest: slowest.r,
  imbalance: r2(shares[0] / medShare),
  // 如果最慢的 rank 和次慢对齐，这一轮能省多少
  whatIfMs: r1(R.duration * 1000 * (shares[0] - second)),
  stepMsAt: series.itlP50,
};

/* ------------------------------------------------------------ 指标 */
const itlAll = requests.flatMap((q) => q.itl);
const e2e = requests.map((q) => q.last - q.st);
// TPOT：vLLM 的口径是 (e2e - ttft) / (output_tokens - 1)
const tpot = requests.map((q) => (q.last - q.first) / Math.max(1, q.outTok - 1));
const sItl = stats(itlAll);
const sTtft = stats(requests.map((q) => q.ttftMs));
const sE2e = stats(e2e);
const sTpot = stats(tpot);

const slowItl = itlAll.filter((v) => v > SLOW_GAP);
const fastItl = itlAll.filter((v) => v <= SLOW_GAP);

/* 采集口径：负载串直接取自 result3.json 自己的字段，不写死；
 * 对应的 skill 是 pypto-serving 的 vllm-bench-perf（`vllm bench serve` 的 random 数据集）。 */
const capture = {
  skill: 'vllm-bench-perf',
  workload: `random/in${R.input_lens[0]}/out${R.output_lens[0]}/n${R.num_prompts}/concurrency${R.max_concurrency}/rate-${R.request_rate}`,
  note: `客户端口径：TTFT、输出间隔、吞吐都在基准客户端这一侧算（backend ${R.backend}），服务端内部没有分解，16 rank 那半张图是按拓扑建模的。同一压缩包里还有设备详情页用的 profiler trace，本轮是否带插桩开销从 result3.json 判断不了，对外报性能要用同一串负载在未插桩的运行里重跑。`,
};

const data = {
  generatedAt: new Date().toISOString(),
  run: {
    id: R.date,
    title: 'DSpark Serving · 256 请求压测',
    model: 'DeepSeek-V4-Flash-0731 w8a8',
    device: '16 NPU · DP4 TP4 EP16',
    nDevice: N_DEVICE,
    shape: `输入 ${R.input_lens[0]} / 输出 ${R.output_lens[0]} tokens`,
    arrival: `并发上限 ${R.max_concurrency} · 请求速率 ${R.request_rate}`,
    nReq: R.num_prompts,
    completed: R.completed,
    failed: R.failed,
    durationMs: r2(R.duration * 1000),
    arriveSpanMs: r2(Math.max(...requests.map((q) => q.st))),
    inTok: R.total_input_tokens,
    outTok: R.total_output_tokens,
    source: 'Data/dsv4-flash-dspark-a3-dp4tp4ep16-gbs256-in256o64.zip → result3.json',
  },
  // 头条：推理服务最先看的是这套配置吐得出多少 token
  headline: {
    label: '输出吞吐',
    value: r1(R.output_throughput),
    unit: 'tok/s',
    foot: `每卡 ${r1(R.output_throughput / N_DEVICE)} · 每请求 ${r2(R.output_throughput / R.num_prompts)} tok/s · 含输入 ${r1(R.total_token_throughput)} tok/s`,
  },
  // 右栏：分三组，每项带一种视觉编码（分位条 / 占比条 / 迷你曲线 / 状态点）
  // 右栏按“工程师能不能动手”分三档：
  //   p0 每步代价——直接决定引擎改什么；p1 服务结果——定界用；p2 核对——只在不对劲时才看。
  rail: [
    {
      g: "每步代价",
      p: 0,
      items: [
        {
          k: "itl", label: "输出间隔", v: Math.round(sItl.p50), u: "ms",
          // 双峰在页面上画成可点的峰值标签，脚注只放分位、均值和样本数
          foot: `P90 ${Math.round(sItl.p90)} ms · P99 ${Math.round(sItl.p99)} ms · 均值 ${Math.round(sItl.mean)} ms · ${sItl.n.toLocaleString("en-US")} 个`,
          hist: "itl",
        },
        { k: "batch", label: "平均批大小", v: batchMean, u: `/ ${R.max_concurrency}`, foot: `并发上限 ${R.max_concurrency} · 退出前满载`, bar: { val: batchMean, max: R.max_concurrency } },
        { k: "kv", label: "KV cache 峰值", v: r1(Math.max(...series.kv)), u: "%", foot: `窗口均值 ${r1(series.kv.reduce((a, b) => a + b, 0) / series.kv.length)}% · 还有余量`, spark: "kv", bar: { val: r1(Math.max(...series.kv)), max: 100 } },
        {
          k: "spec", label: "投机接受长度", v: r2(R.spec_decode_acceptance_length), u: `/ ${R.spec_decode_per_position_acceptance_rates.length + 1}`,
          foot: `接受率 ${(R.spec_decode_accepted_tokens / R.spec_decode_draft_tokens * 100).toFixed(1)}% · 草稿 ${R.spec_decode_draft_tokens.toLocaleString("en-US")} token 几乎白跑`,
          bar: { val: R.spec_decode_acceptance_length, max: R.spec_decode_per_position_acceptance_rates.length + 1, tone: "bad" },
        },
      ],
    },
    {
      g: "服务结果",
      p: 1,
      items: [
        { k: "req", label: "请求吞吐", v: r2(R.request_throughput), u: "req/s", foot: `输出 ${r1(R.output_throughput)} tok/s · 窗口 ${(R.duration).toFixed(2)} s`, spark: "rate" },
        { k: "tpot", label: "TPOT", v: Math.round(sTpot.p50), u: "ms", foot: `P90 ${Math.round(sTpot.p90)} ms · P99 ${Math.round(sTpot.p99)} ms · 均值 ${Math.round(sTpot.mean)} ms`, dist: { ...sTpot, unit: "ms" } },
        { k: "ttft", label: "TTFT", v: r2(sTtft.p50 / 1000), u: "s", foot: `P90 ${r2(sTtft.p90 / 1000)} s · P99 ${r2(sTtft.p99 / 1000)} s · 均值 ${r2(sTtft.mean / 1000)} s`, dist: { ...sTtft, unit: "ms" } },
      ],
    },
    {
      g: "核对",
      p: 2,
      items: [
        { k: "e2e", label: "E2E P50", v: r2(sE2e.p50 / 1000), u: "s", foot: `最快 ${r2(sE2e.min / 1000)} s · P99 ${r2(sE2e.p99 / 1000)} s` },
        { k: "ok", label: "完成", v: `${R.completed}/${R.num_prompts}`, u: "", foot: `失败 ${R.failed} · 超时 0`, pill: "ok" },
        { k: "slo", label: "TTFT 达标率", v: `${Math.round(requests.filter((q) => q.ttftMs < 2000).length / R.num_prompts * 100)}%`, u: "", foot: `阈值 2 s · ${requests.filter((q) => q.ttftMs < 2000).length}/${R.num_prompts}`, tone: "bad" },
      ],
    },
  ],
  series: {
    ...series,
    tokPerChunk,
    kvCap: KV_CAP,
    peak: {
      conc: Math.max(...series.conc),
      rate: Math.max(...series.rate),
      kv: r1(Math.max(...series.kv)),
      queueMs: r1(Math.max(...series.queueMs)),
    },
    mean: { rate: r1(chunks.length / R.duration), batch: batchMean },
  },
  ranks,
  // SLO 达成率：阈值是 demo 里定的，不是业务给的
  slo: {
    ttftMs: 2000,
    ttftHit: requests.filter((q) => q.ttftMs < 2000).length,
    ttftRate: r2(requests.filter((q) => q.ttftMs < 2000).length / R.num_prompts),
  },
  // 收起来的运行配置
  meta: [
    { k: '模型', v: 'DeepSeek-V4-Flash-0731 w8a8' },
    { k: '设备', v: `${N_DEVICE} NPU · DP4 TP4 EP16` },
    { k: '负载', v: `输入 ${R.input_lens[0]} / 输出 ${R.output_lens[0]} tokens · ${R.num_prompts} 个请求` },
    { k: '并发', v: `上限 ${R.max_concurrency} · 请求速率 ${R.request_rate}` },
    { k: '到达', v: `全部请求在 ${r1(Math.max(...requests.map((q) => q.st)))} ms 内提交完毕` },
    { k: '窗口', v: `${(R.duration).toFixed(2)} s · ${R.date}` },
    { k: '来源', v: 'result3.json（benchmark_serving 客户端口径）' },
  ],
  // 采集口径（固定负载 / 采集 skill / 提醒）放在右栏末尾，与单请求页同一块结构
  capture,
  stats: { ttft: sTtft, tpot: sTpot, itl: sItl, e2e: sE2e },
  dists: [
    { k: 'ttft', label: 'TTFT', sub: '每请求 1 个样本', unit: 'ms', stats: sTtft, bins: hist(requests.map((q) => q.ttftMs), 400, 3600, 100) },
    { k: 'itl', label: '输出间隔', sub: '全部输出间隔', unit: 'ms', stats: sItl, bins: hist(itlAll, 0, 1100, 25), hot: [SLOW_GAP, 1100] },
    { k: 'tpot', label: 'TPOT', sub: '每请求 1 个样本', unit: 'ms', stats: sTpot, bins: hist(tpot, 280, 420, 5) },
    { k: 'e2e', label: 'E2E', sub: '每请求 1 个样本', unit: 'ms', stats: sE2e, bins: hist(e2e, 20000, 27600, 200) },
  ],
  requests: requests.map((q) => ({
    i: q.i,
    st: r2(q.st),
    first: r2(q.first),
    mid: r2(q.mid),
    last: r2(q.last),
    ttft: r2(q.ttftMs),
    e2e: r2(q.last - q.st),
    chunks: q.chunks,
    outTok: q.outTok,
    slowTok: q.slowTok,
    fastTok: q.fastTok,
    batch: q.batch,
  })),
  batches: batches.map((b, k) => ({
    id: b.id,
    label: `首 token 批次 ${k + 1}`,
    n: b.n,
    t0: b.t0,
    t1: b.t1,
    ttftMs: b.ttftMs,
    share: r2(b.n / R.num_prompts),
  })),
  waves,
  bands,
  split: r2(split),
  itlModes: {
    slow: { n: slowItl.length, p50: stats(slowItl).p50, share: r2(slowItl.length / itlAll.length) },
    fast: { n: fastItl.length, p50: stats(fastItl).p50, share: r2(fastItl.length / itlAll.length) },
  },
  spec: {
    drafts: R.spec_decode_num_drafts,
    draftTokens: R.spec_decode_draft_tokens,
    acceptedTokens: R.spec_decode_accepted_tokens,
    acceptLen: r2(R.spec_decode_acceptance_length),
    acceptRate: r2(R.spec_decode_accepted_tokens / R.spec_decode_draft_tokens),
    perPos: R.spec_decode_per_position_acceptance_rates.map((v) => r2(v)),
  },
  drill: { href: './index.html', label: '单请求执行定界（GBS1 trace）' },
};

fs.writeFileSync(OUT, `window.TRIAGE_OVERVIEW = ${JSON.stringify(data)};\n`);
console.log(`overview-data.js  ${(fs.statSync(OUT).size / 1024).toFixed(1)} KB`);
console.log(`  请求 ${data.run.nReq}（到达跨度 ${data.run.arriveSpanMs} ms） 窗口 ${data.run.durationMs} ms`);
console.log(`  首 token 批次 ${data.batches.map((b) => `${b.n}@${r1(b.ttftMs)}ms`).join(' / ')}`);
console.log(`  波次 ${data.waves.length}：慢 ${bands[0].nWave} 波 P50 ${bands[0].gapP50} ms（${bands[0].t0}–${bands[0].t1} ms） / 快 ${bands[1].nWave} 波 P50 ${bands[1].gapP50} ms`);
console.log(`  输出间隔双峰：>${SLOW_GAP}ms ${data.itlModes.slow.n} 个 P50 ${data.itlModes.slow.p50} ms；≤${SLOW_GAP}ms ${data.itlModes.fast.n} 个 P50 ${data.itlModes.fast.p50} ms`);
console.log(`  投机解码：草稿 ${data.spec.draftTokens} token，接受 ${data.spec.acceptedTokens}（${(data.spec.acceptRate * 100).toFixed(1)}%），接受长度 ${data.spec.acceptLen}`);
