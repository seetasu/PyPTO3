/* Summary of dsv4-flash-dspark-a3-dp4tp4ep16-gbs256-in256o64.zip/result3.json.
 * The 1.38 GB Chrome/PyTorch trace remains an external raw artifact; this
 * compact case intentionally exposes only request-level benchmark evidence. */
(function () {
  'use strict';
  const id = 'serving_gbs256';
  window.TUNING_RUNS[id] = {
    kind: 'serving-benchmark',
    case: {
      id: id,
      label: 'serving_gbs256',
      sub: 'DeepSeek V4 Flash · GBS256 · DP4/TP4/EP16',
      program: 'dsv4-flash-dspark · GBS256 benchmark',
      model: 'DeepSeek-V4-Flash-0731-w8a8',
      backend: 'OpenAI serving',
      runDir: 'dsv4-flash-dspark-a3-dp4tp4ep16-gbs256-in256o64.zip',
      capturedAt: '2026-09-28',
    },
    benchmark: {
      requests: 256,
      completed: 256,
      failed: 0,
      concurrency: 256,
      inputTokens: 65536,
      outputTokens: 16384,
      durationS: 27.4187982790172,
      requestThroughput: 9.33666010431643,
      outputThroughput: 597.545658419336,
      totalTokenThroughput: 2987.72829209668,
      latency: [
        { id: 'ttft', label: 'TTFT', mean: 2317.43384473521, median: 2587.81796367839, p99: 3427.94064551126, unit: 'ms', hint: '首 token 时间' },
        { id: 'tpot', label: 'TPOT', mean: 382.258111968743, median: 386.494812415174, p99: 404.77913321432, unit: 'ms', hint: '每输出 token 时间' },
        { id: 'itl', label: 'ITL', mean: 432.939527322717, median: 134.56238922663, p99: 874.827653169632, unit: 'ms', hint: 'token 间隔' },
      ],
      trace: {
        raw: 'dsv4-flash-dspark-a3-dp4tp4ep16-gbs256-in256o64.json',
        bytes: 1377413488,
        observed: ['PyTorch CPU op', 'aten::copy_', 'Event::synchronize', 'gloo:all_reduce'],
        missing: ['host_wall_us / device_wall_us 同 scope 拆分', 'WorkerProcess 统计', '可归因的 H2D / D2H'],
      },
    },
  };
  window.TUNING_CASES.push({
    id: id,
    label: 'serving_gbs256',
    sub: 'DeepSeek V4 Flash · 256 并发 benchmark',
  });
}());
