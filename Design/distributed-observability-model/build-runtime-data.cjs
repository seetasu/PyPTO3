/* Build compact, source-backed detail data for the distributed run viewer. */
const fs = require('fs');
const path = require('path');

const repoRoot = path.resolve(__dirname, '..', '..');
const sourceRoot = path.join(repoRoot, 'Data', 'DeepseekV4', '_jit_l3_decode_csa_20260903_010617');
const outputPath = path.join(repoRoot, 'Design', 'distributed-observability-prototype', 'src', 'run-data.json');
const sourceLabel = 'Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617';
const readJson = (filePath) => JSON.parse(fs.readFileSync(filePath, 'utf8'));
const flatten = (value) => Array.isArray(value)
  ? value.flatMap((item) => Array.isArray(item) ? item : [item])
  : [];
const usFromCycles = (cycles, hz) => Number.isFinite(cycles) && hz ? cycles * 1e6 / hz : null;
const round = (value, digits = 2) => value == null ? null : Number(value.toFixed(digits));

const meta = readJson(path.join(sourceRoot, 'distributed_meta.json'));
const runId = 'run:decode-csa:20260903-010617';
const ranks = [];
const rankDirs = fs.readdirSync(path.join(sourceRoot, 'dfx_outputs'))
  .filter((entry) => /^rank\d+$/.test(entry))
  .sort((a, b) => Number(a.slice(4)) - Number(b.slice(4)));

for (const rankDir of rankDirs) {
  const rankIndex = Number(rankDir.slice(4));
  const deviceId = meta.distributed_config.device_ids[rankIndex] ?? null;
  const dfxRelative = `dfx_outputs/${rankDir}/d0`;
  const dfxDir = path.join(sourceRoot, 'dfx_outputs', rankDir, 'd0');
  const deps = readJson(path.join(dfxDir, 'deps.json'));
  const names = readJson(path.join(dfxDir, 'name_map.json')).callable_id_to_name;
  const chip = readJson(path.join(dfxDir, 'chip_swimlane_records.json'));
  const traceFile = fs.readdirSync(dfxDir).find((file) => file.startsWith('merged_swimlane_') && file.endsWith('.json'));
  const trace = readJson(path.join(dfxDir, traceFile)).traceEvents;
  const hz = chip.metadata.clock_freq_hz;

  const taskById = new Map(deps.tasks.map((task) => [String(task.task_id), {
    id: String(task.task_id),
    scope: task.scope,
    earlyDispatch: task.early_dispatch,
    blockNum: task.block_num,
    kernelIds: task.kernel_ids.filter((id) => id >= 0),
    kernelNames: task.kernel_ids.filter((id) => id >= 0).map((id) => names[id]).filter(Boolean),
    args: task.args.map((arg) => ({
      index: arg.idx,
      role: arg.type,
      tensorId: String(arg.tensor_id),
      dtype: arg.dtype,
      shape: arg.shape,
      startOffset: arg.start_offset,
      strides: arg.strides,
    })),
    observed: { deviceEventCount: 0, kernelCoreTimeUs: 0, setupTimeUs: 0, deviceSpanUs: null, deviceStartOffsetUs: null, coreIds: [], submitCount: 0, schedulerTimeUs: 0, schedulerStartOffsetUs: null },
  }]));

  const traceEvents = trace.filter((event) => event.ph === 'X' && event.args?.taskId != null);
  const traceOrigin = traceEvents.reduce((min, event) => Math.min(min, event.ts), Infinity);
  const traceByTask = new Map();
  for (const event of traceEvents) {
    const taskId = String(event.args.taskId);
    if (!traceByTask.has(taskId)) traceByTask.set(taskId, { count: 0, start: Infinity, end: -Infinity, kernelCoreTimeUs: 0, setupTimeUs: 0, coreIds: new Set(), kernelNames: new Map() });
    const aggregate = traceByTask.get(taskId);
    aggregate.count += 1;
    aggregate.start = Math.min(aggregate.start, event.ts);
    aggregate.end = Math.max(aggregate.end, event.ts + event.dur);
    aggregate.kernelCoreTimeUs += Number(event.args['kernel-duration-us'] ?? 0);
    aggregate.setupTimeUs += Number(event.args.local_setup_us ?? 0);
    if (event.args['event-hint']) {
      const coreMatch = String(event.args['event-hint']).match(/CoreId:(\d+)/);
      if (coreMatch) aggregate.coreIds.add(Number(coreMatch[1]));
    }
    const cleanName = String(event.name || '').replace(/\(r\d+t\d+\)$/, '');
    if (cleanName) aggregate.kernelNames.set(cleanName, (aggregate.kernelNames.get(cleanName) || 0) + 1);
  }
  for (const [taskId, aggregate] of traceByTask) {
    const task = taskById.get(taskId);
    if (!task) continue;
    task.observed.deviceEventCount = aggregate.count;
    task.observed.kernelCoreTimeUs = round(aggregate.kernelCoreTimeUs);
    task.observed.setupTimeUs = round(aggregate.setupTimeUs);
    task.observed.deviceSpanUs = round(aggregate.end - aggregate.start);
    task.observed.deviceStartOffsetUs = round(aggregate.start - traceOrigin);
    task.observed.coreIds = [...aggregate.coreIds].sort((a, b) => a - b);
    task.observed.traceKernelNames = [...aggregate.kernelNames.entries()].sort((a, b) => b[1] - a[1]).map(([name, count]) => ({ name, eventCount: count }));
  }

  const submitEvents = flatten(chip.aicpu_orchestrator_phases);
  const submitOrigin = submitEvents.reduce((min, event) => Math.min(min, event.start_cycles), Infinity);
  for (const event of submitEvents) {
    const task = taskById.get(String(event.task_id));
    if (!task) continue;
    task.observed.submitCount += 1;
    task.observed.schedulerTimeUs = round(task.observed.schedulerTimeUs + usFromCycles(event.end_cycles - event.start_cycles, hz));
    const offset = usFromCycles(event.start_cycles - submitOrigin, hz);
    task.observed.schedulerStartOffsetUs = task.observed.schedulerStartOffsetUs == null ? round(offset) : Math.min(task.observed.schedulerStartOffsetUs, round(offset));
  }

  const schedulerPhases = flatten(chip.aicpu_scheduler_phases);
  const schedulerSummary = Object.entries(schedulerPhases.reduce((byKind, event) => {
    const kind = event.kind || 'unknown';
    byKind[kind] ||= { kind, count: 0, totalTimeUs: 0, tasksProcessed: 0, firstOffsetUs: null };
    const summary = byKind[kind];
    summary.count += 1;
    summary.totalTimeUs += usFromCycles(event.end_cycles - event.start_cycles, hz);
    summary.tasksProcessed += Number(event.tasks_processed || 0);
    const offset = usFromCycles(event.start_cycles - schedulerPhases.reduce((min, item) => Math.min(min, item.start_cycles), Infinity), hz);
    summary.firstOffsetUs = summary.firstOffsetUs == null ? offset : Math.min(summary.firstOffsetUs, offset);
    return byKind;
  }, {})).map(([, entry]) => ({
    ...entry,
    totalTimeUs: round(entry.totalTimeUs),
    firstOffsetUs: round(entry.firstOffsetUs),
  }));

  const taskIds = new Set(deps.tasks.map((task) => String(task.task_id)));
  const boundaryIds = new Set(deps.edges.flatMap((edge) => [String(edge.pred), String(edge.succ)]).filter((id) => !taskIds.has(id)));
  const artifactsRoot = path.join(sourceRoot, 'next_levels', 'decode_csa_test');
  const artifactGroups = ['orchestration', 'ptoas', 'kernels'].map((group) => {
    const folder = path.join(artifactsRoot, group);
    if (!fs.existsSync(folder)) return { group, files: [] };
    const files = [];
    const walk = (current) => {
      for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
        const entryPath = path.join(current, entry.name);
        if (entry.isDirectory()) walk(entryPath);
        else files.push(path.relative(sourceRoot, entryPath).replace(/\\/g, '/'));
      }
    };
    walk(folder);
    return { group, files: files.sort() };
  });

  ranks.push({
    id: `${runId}:rank:${rankIndex}`,
    rankIndex,
    deviceId,
    label: `Rank ${rankIndex} · Device ${deviceId}`,
    source: `${sourceLabel}/${dfxRelative}`,
    device: {
      coreCount: chip.metadata.num_cores,
      aicCount: chip.metadata.core_types.filter((type) => type === 'aic').length,
      aivCount: chip.metadata.core_types.filter((type) => type === 'aiv').length,
      clockHz: hz,
    },
    counts: {
      taskInvocations: deps.tasks.length,
      dependencyEdges: deps.edges.length,
      dependencyOnlyTaskRefs: boundaryIds.size,
      deviceTraceEvents: traceEvents.length,
      schedulerPhaseEvents: schedulerPhases.length,
      submitEvents: submitEvents.length,
      taskEventsWithTiming: [...traceByTask.keys()].filter((id) => taskIds.has(id)).length,
    },
    scheduler: {
      source: `${sourceLabel}/${dfxRelative}/chip_swimlane_records.json`,
      phases: schedulerSummary,
      submissions: submitEvents.map((event) => ({
        submitIndex: event.submit_idx,
        taskId: String(event.task_id),
        startOffsetUs: round(usFromCycles(event.start_cycles - submitOrigin, hz)),
        durationUs: round(usFromCycles(event.end_cycles - event.start_cycles, hz)),
      })),
    },
    tasks: [...taskById.values()].map((task) => ({
      ...task,
      dependencies: deps.edges.filter((edge) => String(edge.pred) === task.id || String(edge.succ) === task.id).map((edge) => ({
        direction: String(edge.pred) === task.id ? 'out' : 'in',
        otherTaskId: String(edge.pred) === task.id ? String(edge.succ) : String(edge.pred),
        tensorId: edge.tensor_id == null ? null : String(edge.tensor_id),
        source: edge.source,
        flags: edge.flags,
      })),
    })),
    dependencies: deps.edges.map((edge) => ({
      pred: String(edge.pred),
      succ: String(edge.succ),
      tensorId: edge.tensor_id == null ? null : String(edge.tensor_id),
      source: edge.source,
      flags: edge.flags,
    })),
    dependencyOnlyTaskIds: [...boundaryIds].sort(),
    artifacts: artifactGroups,
    trace: {
      file: `${sourceLabel}/${dfxRelative}/${traceFile}`,
      sourceRunRelativeStart: 'Rank-local merged trace origin; do not compare offsets across ranks.',
      eventCount: traceEvents.length,
    },
  });
}

const graph = readJson(path.join(repoRoot, 'Design', 'distributed-observability-model', 'subjects.json'));
const data = {
  schemaVersion: 1,
  run: {
    id: runId,
    name: 'decode_csa_test',
    workload: 'DeepSeek-V4 · decode_csa',
    status: 'captured',
    rankCount: ranks.length,
    deviceIds: meta.distributed_config.device_ids,
    platform: meta.platform,
    backend: meta.backend_type,
    runtime: meta.distributed_config.runtime,
    source: `${sourceLabel}/distributed_meta.json`,
  },
  communication: graph.subjects.filter((subject) => subject.type === 'communication-sequence').map((subject) => ({
    id: subject.id.split(':').at(-1),
    name: subject.displayName.split(' · ')[1],
    family: subject.attributes.family,
    observedTaskCount: subject.attributes.participantTaskCount,
    correlation: subject.attributes.correlation,
    crossRankTimingCorrelation: subject.attributes.crossRankTimingCorrelation,
    members: graph.relations.filter((relation) => relation.type === 'participates-in' && relation.to === subject.id).map((relation) => relation.from),
  })),
  ranks,
  coverage: graph.coverage,
};

fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.writeFileSync(outputPath, `${JSON.stringify(data, null, 2)}\n`);
console.log(`Wrote ${path.relative(repoRoot, outputPath)} (${(fs.statSync(outputPath).size / 1024).toFixed(0)} KiB)`);
console.log(JSON.stringify({ ranks: ranks.length, tasks: ranks.map((rank) => rank.counts.taskInvocations), deps: ranks.map((rank) => rank.counts.dependencyEdges), traceEvents: ranks.map((rank) => rank.counts.deviceTraceEvents), communications: data.communication.map((item) => [item.id, item.observedTaskCount]) }));
