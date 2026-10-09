/*
 * Teaching fixture: a complete L2 -> L1 -> compiler diagnosis chain.
 * The scheduling trace and PH-MR-001 values are borrowed from decode_csa;
 * the L1 bridge and causal interpretation are fabricated for teaching.
 * Keep the case label explicit so this can never be mistaken for a measured
 * compiler regression.
 */
(function () {
  'use strict';

  const clone = (value) => JSON.parse(JSON.stringify(value));
  const base = window.TUNING_RUNS.decode_csa;
  const run = clone(base);
  const taskId = 'r2t28';
  const passName = 'MemoryReuse';
  const hintSite = 'decode_compressor_ratio4.py:110';
  const mockId = 'MOCK-C1';
  const evidence = [
    { artifact: '依赖关键路径（decode_csa 实测）', locator: 'kv_score_proj_0', value: 'slack 0 · 256 块 / 24 AIC · span 395.64 us' },
    { artifact: 'PH-MR-001（decode_csa 实测）', locator: hintSite, value: 'Right · depth 2→1 · 5 组 · 32 KB/stage · 64 KB free' },
    { artifact: 'L1 PMU（教学模拟）', locator: 'kv_score_proj_0', value: 'mte2 0.42 · mac 0.06；提示访存等待可能未被流水隐藏' },
  ];
  const subjects = (view, tasks, tab) => ({
    view, tab: tab || null, overlay: null, tasks: tasks || [], lanes: [], sites: [],
    files: [], ranks: [], schedPhases: [], absent: false,
  });

  run.case.id = 'mock_l2_pass_chain';
  run.case.label = '模拟因果链';
  run.case.program = 'MOCK · PH-MR-001 反向案例';
  run.case.sub = '真实 L2 / PH hint + 模拟 L1 因果桥接';
  run.case.model = '混合教学案例（实测证据与模拟桥接分开标注）';
  run.case.compileSource = null;
  run.defaultRank = 'rank0';

  const finding = {
    id: mockId,
    kind: 'chain',
    level: 'l2',
    severity: 'high',
    axis: 'dependency',
    title: '模拟归因：MemoryReuse 深度回退可能放大关键任务耗时',
    metric: 'kv_score_proj_0 · 256 块 / 24 AIC · span 395.64 us · slack 0',
    cost: { us: 395.64, share: 8.11, basis: '实测关键任务 span 上界，不是该 Pass 的可回收收益' },
    claim: '这是基于现有性能提示反向搭建的教学案例，不是已验证根因。L2 真实轨迹显示 kv_score_proj_0 位于零 slack 关键链且与非关键任务共享 AIC；同一源码点真实 PH-MR-001 显示 MemoryReuse 将流水深度从 2 降为 1。中间的 L1 PMU 桥接数据为模拟，用来演示怎样验证「流水回退是否导致搬运等待、进而拉长块耗时」。',
    chain: [
      {
        level: 'l2', role: 'observe',
        headline: '零 slack 的 kv_score_proj_0 与非关键工作竞争 AIC',
        detail: '真实 trace：256 块分布在 24 个 AIC，span 395.64 us；AIC_0 / AIC_18 可见其他任务穿插在其 block 间隙。L2 确认关键任务被共享资源上的工作打断，但单看泳道不能区分块内变慢与调度等待。',
        evidence: [evidence[0]],
        subjects: subjects('l2', [taskId, 'r2t23', 'r3t11']),
        chips: [
          { kind: 'task', id: taskId, label: 'kv_score_proj_0', value: '395.64 us · slack 0' },
          { kind: 'lane', id: 'AIC_0', label: 'AIC_0', value: '竞争窗口' },
          { kind: 'lane', id: 'AIC_18', label: 'AIC_18', value: '竞争窗口' },
        ],
      },
      {
        level: 'l1', role: 'descend',
        headline: 'PMU 模拟读数显示搬运繁忙、Cube 利用偏低',
        detail: '教学模拟读数：mte2 0.42、mac 0.06，作为「搬运等待可能未被计算覆盖」的线索。该数据不是本 run 实测；若真实重采不能复现，就不能把 PH-MR-001 当作当前 L2 现象的解释。',
        evidence: [
          evidence[2],
          { artifact: 'kernel block timing（decode_csa 实测）', locator: 'kv_score_proj_0', value: 'durMed 6.24 us · durP90 6.80 us' },
        ],
        subjects: subjects('l1', [taskId]),
        chips: [{ kind: 'task', id: taskId, label: 'kv_score_proj_0', value: '块内流水正常' }],
      },
      {
        level: 'compiler', role: 'root',
        headline: 'PH-MR-001 将源码点落到 MemoryReuse 深度回退',
        detail: '真实 hint：decode_compressor_ratio4.py:110 的 Right 侧每级 stage 需要 32 KB，5 组请求 depth 2；预算 64 KB 时实际拟合为 depth 1。该位置与 kv_score_proj / kv_score_proj_0 的任务源码行对应。它证明发生了回退，但不能单独证明回退造成 L2 span 增长；仍需按提示做单变量重编译与复测。',
        evidence: [evidence[1]],
        subjects: Object.assign(subjects('compiler', [taskId], 'depth'), { sites: [hintSite] }),
        chips: [{ kind: 'site', id: hintSite, label: hintSite, value: 'PH-MR-001 · 2→1' }],
      },
    ],
    terminus: { level: 'compiler', reason: '已定位到 MemoryReuse 的 PH-MR-001；因果仍待单变量实验验证' },
    evidence,
    focus: { view: 'l2', task: taskId },
    lever: '先对该源码点做一个实验：只降低 Right tile 的每级占用或减少同驻 group，观察 MemoryReuse 是否恢复 depth 2；不同时改 AIC 优先级和调度策略。',
    guardrail: '缩小 tile 会增加循环 / 搬运次数，减少 group 可能降低并行度；需要正确性校验，并确认竞争任务 slack 与 makespan 没有恶化。',
    verify: '重编译检查 PH-MR-001 是否从 2→1 恢复到 2；再重测对应任务的 block 中位时长、span、间隙和 makespan。hint 消失但 L2 不变，说明这条提示不是主因。',
    subjects: subjects('l2', [taskId, 'r2t23', 'r3t11']),
    chips: [
      { kind: 'task', id: taskId, label: 'kv_score_proj_0', value: '395.64 us' },
      { kind: 'lane', id: 'AIC_0', label: 'AIC_0', value: '竞争窗口' },
    ],
    levels: ['l2', 'l1', 'compiler'],
    queueLevels: ['l2', 'l1', 'compiler'],
    rootPass: passName,
  };

  run.findings = [finding];
  run.chainCount = 1;
  run.hygieneCount = 0;
  run.investigations = [];

  window.TUNING_RUNS[mockId] = run;
  window.TUNING_CASES.push({
    id: mockId,
    label: '模拟 PH-MR-001 · MemoryReuse',
    sub: '复用真实 L2 / PH hint；L1 因果桥接为模拟',
  });
})();
