/* Aggregated from C:\Users\cyf12\Downloads\serving-strace-swimlane(2).json.
 * The source is a Chrome trace: one Serving WorkerProcess, 16 Host STRACE
 * device lanes, 55 decode invocations (56–110), and one complete request.
 * It does not expose device_wall, H2D, or an explicit result-copy event. */
(function () {
  'use strict';
  const run = window.TUNING_RUNS && window.TUNING_RUNS.decode_csa;
  if (!run) return;

  run.e2eTriage = {
    benchmark: {
      e2e_wall_us: 12574109,
      host_wall_us: 302605.1,
      source: 'Serving Strace · 完整请求 / decode step 跨 scope',
      scope: 'E2E = http.completions 完整请求；Host = 每 decode step 取 16 lane 中最长 chip.run 后求均值；device_wall 未采集',
    },
    workers: [
      { id: 'serving-worker (pid 2531562)', count: 1, avg_us: 0, min_us: 0, max_us: 0, source: 'Serving Strace · 单 WorkerProcess，不能判断 worker 间不均衡' },
    ],
    servingWait: { count: 54, avg_us: 234710.6, min_us: 142510, max_us: 419528, source: 'scheduler.wait_worker_output · 实测' },
    workerTasks: [
      { id: 'WorkerProcess.prepare_decode', count: 53, avg_us: 90173.6, min_us: 75236, max_us: 156732, source: 'Serving Strace · 实测' },
      { id: 'WorkerProcess.dispatch_decode', count: 53, avg_us: 33088.1, min_us: 31223, max_us: 43641, source: 'Serving Strace · 实测' },
      { id: 'WorkerProcess.reclaim_decode', count: 53, avg_us: 229282.4, min_us: 142917, max_us: 344831, source: 'Serving Strace · 实测' },
      { id: 'WorkerProcess.execute_step', count: 2, avg_us: 209359, min_us: 3, max_us: 418715, source: 'Serving Strace · 实测；含 prefill / decode，不能与 decode-only 直接比较' },
    ],
    lanes: [
      { id: 'device 0', count: 55, avg_us: 302601.9, min_us: 65897.8, max_us: 375423.7 },
      { id: 'device 1', count: 55, avg_us: 224944, min_us: 64013, max_us: 277735.3 },
      { id: 'device 2', count: 55, avg_us: 225184.1, min_us: 62828.1, max_us: 277181.2 },
      { id: 'device 3', count: 55, avg_us: 225025.7, min_us: 61109.6, max_us: 277181.5 },
      { id: 'device 4', count: 55, avg_us: 224739.6, min_us: 59851.5, max_us: 277020.8 },
      { id: 'device 5', count: 55, avg_us: 224860.7, min_us: 58160.9, max_us: 276826.3 },
      { id: 'device 6', count: 55, avg_us: 224522, min_us: 55867.7, max_us: 276937.4 },
      { id: 'device 7', count: 55, avg_us: 224735, min_us: 55165.9, max_us: 277127.9 },
      { id: 'device 8', count: 55, avg_us: 223586, min_us: 53064.2, max_us: 277202.6 },
      { id: 'device 9', count: 55, avg_us: 224940.6, min_us: 52013.2, max_us: 277124 },
      { id: 'device 10', count: 55, avg_us: 224329.9, min_us: 50777.8, max_us: 277163.7 },
      { id: 'device 11', count: 55, avg_us: 223717.8, min_us: 49420.2, max_us: 276625.3 },
      { id: 'device 12', count: 55, avg_us: 224170.2, min_us: 48055.3, max_us: 276785.6 },
      { id: 'device 13', count: 55, avg_us: 224020.6, min_us: 46198.4, max_us: 277071.3 },
      { id: 'device 14', count: 55, avg_us: 223813.3, min_us: 44744.5, max_us: 277029.2 },
      { id: 'device 15', count: 55, avg_us: 224026.5, min_us: 43271.8, max_us: 276848.8 },
    ],
    rounds: [
      { id: 'D0 · inv 56', bind_h2d: { state: 'repeat', us: 33130.61, source: 'bind 实测；H2D 未采集' }, compile_register: { state: 'repeat', us: 5202.43, source: 'bind.prebuilt 实测；register proxy' }, result_copy: { state: 'unknown', source: '结果拷回未采集' } },
      { id: 'D0 · inv 57', bind_h2d: { state: 'repeat', us: 19028.01, source: 'bind 实测；H2D 未采集' }, compile_register: { state: 'repeat', us: 5249.98, source: 'bind.prebuilt 实测；register proxy' }, result_copy: { state: 'unknown', source: '结果拷回未采集' } },
      { id: 'D0 · inv 58', bind_h2d: { state: 'repeat', us: 28053.52, source: 'bind 实测；H2D 未采集' }, compile_register: { state: 'repeat', us: 4.77, source: 'bind.prebuilt 实测；register proxy' }, result_copy: { state: 'unknown', source: '结果拷回未采集' } },
      { id: 'D0 · inv 110', bind_h2d: { state: 'repeat', us: 68996.78, source: 'bind 实测；H2D 未采集' }, compile_register: { state: 'repeat', us: 4.07, source: 'bind.prebuilt 实测；register proxy' }, result_copy: { state: 'unknown', source: '结果拷回未采集' } },
    ],
  };
}());
