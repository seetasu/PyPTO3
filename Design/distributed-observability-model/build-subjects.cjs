/*
 * Build a fact-preserving cross-rank observability subject graph.
 *
 * It deliberately models entities and declared relations only.  It does not
 * infer a critical path, communication latency, or a root cause from a name.
 */
const fs = require('fs');
const path = require('path');

const repoRoot = path.resolve(__dirname, '..', '..');
const sourceRoot = path.join(
  repoRoot,
  'Data',
  'DeepseekV4',
  '_jit_l3_decode_csa_20260903_010617',
);
const outputPath = path.join(__dirname, 'subjects.json');

const readJson = (file) => JSON.parse(fs.readFileSync(file, 'utf8'));
const sourcePath = (...parts) => path.posix.join(
  'Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617',
  ...parts,
);
const flatRecordCount = (value) => Array.isArray(value)
  ? value.reduce((count, item) => count + (Array.isArray(item) ? item.length : 1), 0)
  : 0;

function communicationSequence(kernelNames) {
  for (const name of kernelNames) {
    const match = name.match(/^(cp_token_allgather|o_group_a2a|tp_o_rs)_/);
    if (!match) continue;
    const family = {
      cp_token_allgather: 'allgather',
      o_group_a2a: 'all-to-all',
      tp_o_rs: 'reduce-scatter',
    }[match[1]];
    return { id: match[1], family };
  }
  return null;
}

const meta = readJson(path.join(sourceRoot, 'distributed_meta.json'));
const runId = 'run:decode-csa:20260903-010617';
const deviceGroupId = `${runId}:device-group`;
const subjects = [];
const relations = [];
const subjectIds = new Set();

function addSubject(subject) {
  if (subjectIds.has(subject.id)) throw new Error(`duplicate subject: ${subject.id}`);
  subjectIds.add(subject.id);
  subjects.push(subject);
}

function addRelation(relation) {
  relations.push(relation);
}

addSubject({
  id: runId,
  type: 'distributed-run',
  displayName: 'decode_csa_test · distributed run',
  attributes: {
    program: 'decode_csa_test',
    platform: meta.platform,
    backendType: meta.backend_type,
    runtime: meta.distributed_config.runtime,
    worldSize: meta.distributed_config.device_ids.length,
  },
  observations: [{ source: sourcePath('distributed_meta.json') }],
});

addSubject({
  id: deviceGroupId,
  type: 'device-group',
  displayName: `device group · ${meta.distributed_config.device_ids.join(', ')}`,
  attributes: {
    deviceIds: meta.distributed_config.device_ids,
    worldSize: meta.distributed_config.device_ids.length,
  },
  observations: [{ source: sourcePath('distributed_meta.json'), field: 'distributed_config.device_ids' }],
});
addRelation({ type: 'contains', from: runId, to: deviceGroupId });

const communicationMembers = new Map();
const rankDirs = fs.readdirSync(path.join(sourceRoot, 'dfx_outputs'))
  .filter((entry) => /^rank\d+$/.test(entry))
  .sort((left, right) => Number(left.slice(4)) - Number(right.slice(4)));

for (const rankDir of rankDirs) {
  const rankIndex = Number(rankDir.slice(4));
  const rankId = `${runId}:rank:${rankIndex}`;
  const deviceId = meta.distributed_config.device_ids[rankIndex] ?? null;
  const dfxDir = path.join(sourceRoot, 'dfx_outputs', rankDir, 'd0');
  const deps = readJson(path.join(dfxDir, 'deps.json'));
  const nameMap = readJson(path.join(dfxDir, 'name_map.json'));
  const chip = readJson(path.join(dfxDir, 'chip_swimlane_records.json'));
  const deviceSubjectId = `${runId}:device:${deviceId ?? `unmapped-rank-${rankIndex}`}`;
  const schedulerId = `${rankId}:scheduler`;

  addSubject({
    id: deviceSubjectId,
    type: 'device',
    displayName: deviceId === null ? `device unknown · rank ${rankIndex}` : `device ${deviceId}`,
    attributes: { deviceId, rankIndex },
    observations: [{ source: sourcePath('distributed_meta.json'), field: 'distributed_config.device_ids' }],
  });
  addRelation({ type: 'contains', from: deviceGroupId, to: deviceSubjectId });

  addSubject({
    id: rankId,
    type: 'rank',
    displayName: `rank ${rankIndex}`,
    attributes: {
      rankIndex,
      deviceId,
      program: 'decode_csa_test',
      taskCount: deps.tasks.length,
      dependencyEdgeCount: deps.edges.length,
    },
    observations: [
      { source: sourcePath('dfx_outputs', rankDir, 'd0/deps.json') },
      { source: sourcePath('dfx_outputs', rankDir, 'd0/chip_swimlane_records.json') },
    ],
  });
  addRelation({ type: 'contains', from: runId, to: rankId });
  addRelation({ type: 'executes-on', from: rankId, to: deviceSubjectId });

  addSubject({
    id: schedulerId,
    type: 'scheduler',
    displayName: `AICPU scheduler · rank ${rankIndex}`,
    attributes: {
      schedulerPhaseRecordCount: flatRecordCount(chip.aicpu_scheduler_phases),
      orchestratorSubmitRecordCount: flatRecordCount(chip.aicpu_orchestrator_phases),
    },
    observations: [{
      source: sourcePath('dfx_outputs', rankDir, 'd0/chip_swimlane_records.json'),
      fields: ['aicpu_scheduler_phases', 'aicpu_orchestrator_phases'],
    }],
  });
  addRelation({ type: 'contains', from: rankId, to: schedulerId });

  const kernelById = nameMap.callable_id_to_name;
  for (const task of deps.tasks) {
    const taskId = String(task.task_id);
    const taskSubjectId = `${rankId}:task:${taskId}`;
    const kernelIds = task.kernel_ids.filter((kernelId) => kernelId >= 0);
    const kernelNames = kernelIds.map((kernelId) => kernelById[kernelId]).filter(Boolean);
    const sequence = communicationSequence(kernelNames);

    addSubject({
      id: taskSubjectId,
      type: 'task-invocation',
      displayName: kernelNames.join(' + ') || `task ${taskId}`,
      attributes: {
        taskId,
        scope: task.scope,
        earlyDispatch: task.early_dispatch,
        blockNum: task.block_num,
        kernelIds,
        kernelNames,
        communicationSequence: sequence?.id ?? null,
      },
      observations: [{ source: sourcePath('dfx_outputs', rankDir, 'd0/deps.json'), record: `tasks[task_id=${taskId}]` }],
    });
    addRelation({ type: 'contains', from: rankId, to: taskSubjectId });

    for (const [position, kernelId] of kernelIds.entries()) {
      const kernelName = kernelById[kernelId];
      if (!kernelName) continue;
      const kernelSubjectId = `${runId}:kernel:${kernelId}`;
      if (!subjectIds.has(kernelSubjectId)) {
        addSubject({
          id: kernelSubjectId,
          type: 'kernel-definition',
          displayName: kernelName,
          attributes: { kernelId, kernelName },
          observations: [{ source: sourcePath('dfx_outputs', rankDir, 'd0/name_map.json'), field: `callable_id_to_name.${kernelId}` }],
        });
      }
      addRelation({ type: 'invokes', from: taskSubjectId, to: kernelSubjectId, position });
    }

    if (sequence) {
      if (!communicationMembers.has(sequence.id)) communicationMembers.set(sequence.id, { family: sequence.family, members: [] });
      communicationMembers.get(sequence.id).members.push(taskSubjectId);
    }
  }

  // deps.json can declare predecessors/successors that are absent from its
  // detailed tasks table. Keep them visible as boundary objects instead of
  // dropping their edges or pretending that they are normal invocations.
  const listedTaskIds = new Set(deps.tasks.map((task) => String(task.task_id)));
  const referencedTaskIds = new Set(deps.edges.flatMap((edge) => [String(edge.pred), String(edge.succ)]));
  for (const taskId of referencedTaskIds) {
    if (listedTaskIds.has(taskId)) continue;
    const taskSubjectId = `${rankId}:task:${taskId}`;
    addSubject({
      id: taskSubjectId,
      type: 'unexpanded-task-reference',
      displayName: `task ${taskId} · dependency-only`,
      attributes: {
        taskId,
        rankIndex,
        detailAvailability: 'referenced by deps.edges but absent from deps.tasks',
      },
      observations: [{ source: sourcePath('dfx_outputs', rankDir, 'd0/deps.json'), record: `edges[*] task_id=${taskId}` }],
    });
    addRelation({ type: 'contains', from: rankId, to: taskSubjectId });
  }

  for (const edge of deps.edges) {
    addRelation({
      type: 'task-dependency',
      from: `${rankId}:task:${edge.pred}`,
      to: `${rankId}:task:${edge.succ}`,
      attributes: {
        source: edge.source,
        flags: edge.flags,
        tensorId: edge.tensor_id,
      },
      observations: [{ source: sourcePath('dfx_outputs', rankDir, 'd0/deps.json'), record: `edges[pred=${edge.pred},succ=${edge.succ}]` }],
    });
  }
}

for (const [sequenceId, sequence] of communicationMembers) {
  const collectiveId = `${runId}:communication:${sequenceId}`;
  addSubject({
    id: collectiveId,
    type: 'communication-sequence',
    displayName: `${sequence.family} · ${sequenceId}`,
    attributes: {
      family: sequence.family,
      participantTaskCount: sequence.members.length,
      correlation: 'same named sequence observed independently on each rank',
      crossRankTimingCorrelation: 'unavailable',
    },
    observations: [{
      source: 'dfx_outputs/rank*/d0/name_map.json + deps.json',
      basis: 'kernel names with the same sequence prefix',
    }],
  });
  addRelation({ type: 'contains', from: runId, to: collectiveId });
  for (const taskSubjectId of sequence.members) addRelation({ type: 'participates-in', from: taskSubjectId, to: collectiveId });
}

const subjectCounts = subjects.reduce((counts, subject) => {
  counts[subject.type] = (counts[subject.type] || 0) + 1;
  return counts;
}, {});
const relationCounts = relations.reduce((counts, relation) => {
  counts[relation.type] = (counts[relation.type] || 0) + 1;
  return counts;
}, {});

const graph = {
  schemaVersion: 1,
  title: 'decode_csa_test 跨 rank 性能观测主体图',
  scope: '主体对象与原始声明关系；不含时延归因、关键路径或根因推断。',
  sourceRun: {
    id: runId,
    root: sourcePath(),
    distributedMeta: sourcePath('distributed_meta.json'),
  },
  subjects,
  relations,
  coverage: {
    observed: [
      'distributed run, device group, device, rank, AICPU scheduler, task invocation, kernel definition',
      'same-name cross-rank communication sequences: allgather, all-to-all, reduce-scatter',
      'intra-rank task dependency edges declared in deps.json',
    ],
    unavailable: [
      'collective ID and communicator/group ID',
      'message byte size and transport topology',
      'explicit cross-rank dependency edges',
      'a clock-aligned cross-rank timing correlation',
    ],
  },
  summary: { subjectCounts, relationCounts },
};

fs.writeFileSync(outputPath, `${JSON.stringify(graph, null, 2)}\n`);
console.log(`Wrote ${path.relative(repoRoot, outputPath)}`);
console.log(JSON.stringify(graph.summary));
