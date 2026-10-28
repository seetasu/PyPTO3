// Adapter for the original perf-lane.v1 export. Never invent unavailable measurements.
export function adapt(input) {
  if (input?.schemaVersion !== 'perf-lane.v1' || !Array.isArray(input.ranks) || !input.ranks.length) throw new Error('需要包含 ranks 的 perf-lane.v1 数据');
  for (const rank of input.ranks) {
    if (!Array.isArray(rank.launches) || !Array.isArray(rank.cores) || !(rank.windowUs > 0)) throw new Error(`Rank ${rank.rank ?? ''} 缺少 launches、cores 或有效 windowUs`);
    for (const task of rank.launches) {
      if (task.id == null || task.taskId == null || task.ct == null || task.leaf == null || !Number.isFinite(task.spanUs) || !Number.isFinite(task.coresUsed) || !rank.cores.some(c => c.ct === task.ct) || !Number.isFinite(task.startUs) || !Number.isFinite(task.endUs) || task.endUs < task.startUs || !Array.isArray(task.blocks?.items)) throw new Error(`Task ${task.id ?? '?'} 的时间或 block 数据无效`);
      for (const b of task.blocks.items) if (!Number.isFinite(b.startUs) || !Number.isFinite(b.endUs) || b.endUs < b.startUs || !rank.cores.some(c => String(c.id) === String(b.core))) throw new Error(`Task ${task.id} 的 block 数据无效`);
    }
  }
  return input;
}
export function capabilities(rank) {
  return { critical: Array.isArray(rank.criticalPath) && rank.criticalPath.length > 0,
    features: rank.launches.every(t => ['n','spread','p90','p50'].every(k => Number.isFinite(t.blocks[k])) && Number.isFinite(t.computeUs) && Number.isFinite(t.setupUs)),
    composition: rank.launches.every(t => t.blocks.items.every(b => Number.isFinite(b.setupUs) && Number.isFinite(b.computeUs) && Number.isFinite(b.slotUs))),
    dependencies: Array.isArray(rank.edges) };
}

