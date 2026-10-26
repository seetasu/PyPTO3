import { readFileSync } from "node:fs";
import { resolve } from "node:path";

export function buildSetupAnalysis(root, run, dump) {
  const sourceAt = (path, needles, label) => {
    const source = readFileSync(resolve(root, path), "utf8").split(/\r?\n/);
    return { path, label, lines: needles.map(needle => {
      const line = source.findIndex(text => text.includes(needle));
      if (line < 0) throw new Error(`Missing timing evidence: ${needle}`);
      return { line: line + 1, text: source[line] };
    }) };
  };
  const timingSources = [
    sourceAt("repo/simpler/simpler_setup/tools/swimlane_converter.py", ["receive_us = _to_us(start_cycles - r2s_cycles)", "local_setup_us = start_us - receive_us"], "转换脚本 · 字段计算公式"),
    sourceAt("repo/simpler/src/a2a3/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp", ["uint64_t receive_time =", "dcci(exec_payload, ENTIRE_DATA_CACHE)", "if (exec_payload->src_payload != 0)", "if (read_dmb_high32() == task_id)", "uint64_t start_time =", "execute_task(exec_payload)"], "运行时参考实现 · 领取、门控与执行计时"),
  ];
  const ranks = {};
  for (const rank of ["rank0", "rank1"]) {
    const base = `${run}/dfx_outputs/${rank}/d0`;
    const raw = JSON.parse(readFileSync(resolve(root, base, "chip_swimlane_records.json"), "utf8"));
    const deps = JSON.parse(readFileSync(resolve(root, base, "deps.json"), "utf8"));
    const rows = raw.aicore_tasks, cpu = raw.aicpu_tasks;
    const frequency = raw.metadata.clock_freq_hz / 1e6;
    const origin = Math.min(...rows.map(row => row[3] - row[5]));
    const time = cycles => (cycles - origin) / frequency;
    const targetId = 8589934647, predId = 8589934645;
    const target = rows.filter(row => row[1] === targetId), pred = rows.filter(row => row[1] === predId);
    if (target.length !== 48 || pred.length !== 72) throw new Error(`${rank}: unexpected task/block identity`);
    if (!deps.edges.some(edge => edge.pred === String(predId) && edge.succ === String(targetId))) throw new Error("Missing qk_pv dependency");
    const predEnd = time(Math.max(...pred.map(row => row[4])));
    const predFinish = time(Math.max(...cpu.filter(row => pred.some(p => p[0] === row[0] && p[2] === row[1])).map(row => row[3])));
    const blocks = target.map(row => {
      const previous = rows.filter(p => p !== row && p[0] === row[0] && p[4] <= row[3]).sort((a, b) => b[4] - a[4])[0];
      const dispatch = cpu.find(p => p[0] === row[0] && p[1] === row[2]);
      return { core: row[0], receive: time(row[3] - row[5]), start: time(row[3]), end: time(row[4]),
        setup: row[5] / frequency, kernel: (row[4] - row[3]) / frequency,
        previousEnd: time(previous[4]), previousTask: String(previous[1]), dispatch: time(dispatch[2]),
        recordIndex: rows.indexOf(row) };
    }).sort((a, b) => a.core - b.core);
    const setup = blocks.reduce((sum, block) => sum + block.setup, 0);
    const beforeEnd = blocks.reduce((sum, block) => sum + Math.max(0, predEnd - block.receive), 0);
    ranks[rank] = { blocks, predEnd, predFinish, setup, beforeEnd, overlapShare: beforeEnd / setup,
      residualMean: blocks.reduce((sum, block) => sum + block.start - Math.max(predFinish, block.receive), 0) / blocks.length,
      rawPath: `${base}/chip_swimlane_records.json`, depsPath: `${base}/deps.json`,
      pickupGapMin: Math.min(...blocks.map(b => b.receive - b.previousEnd)), pickupGapMax: Math.max(...blocks.map(b => b.receive - b.previousEnd)) };
  }
  return {
    ranks,
    boundary: "归因边界：约 99% 是 Setup 区间与前驱执行区间的重叠比例，结合运行时门控机制支持依赖等待解释；归档没有独立 doorbell 时间戳。Pass 可定位依赖来源，不能独立确定运行时等待耗时。",
    steps: [
      { pass: "实测", name: "receive → kernel start → end", title: "对齐各核执行时序", summary: "泳道左端是领取时间，真实 kernel 开始时间在 Setup 区间之后。选择任意 block，对照同核前驱结束和整体前驱完成。", finding: "两个 rank 的 48 个核都在自己的 qk_pv block 结束后很快领取当前任务，随后集中在 qk_pv 整体完成后进入 kernel。提前领取的核，记录到更长的 Setup。", visual: "blocks", next: "继续量化 Setup 中有多少区间发生在前驱完成之前。", sources: [] },
      { pass: "归因", name: "时段重叠 ≠ 精确等待计数", title: "区分等待与本地准备", summary: "local_setup_us = kernel_start − receive；提前派发路径中，领取后等待依赖的区间也会进入该字段。", finding: "本次约 99% 的 Setup 核时位于 qk_pv 最后一个 block 执行结束之前。剩余时段包含通知、轮询与本地准备，不能全部认作初始化。流水缓冲和通信 Fence 位于 kernel 执行窗口内。", visual: "totals", next: "下一步追溯：当前任务为什么需要等待整个 qk_pv？运行时源码为本仓库参考镜像，归档尚未绑定采集二进制的确切版本，需与原始时序交叉验证。", sources: timingSources },
      { pass: "P00–P40", name: "frontend → Outline → Dependencies", title: "追溯 Pass 中的依赖", summary: "前端已显式写入 deps=[qk_tid, attn_rope_tid]；外提与提交降低后仍保留整任务依赖。", finding: "已确认：P09 外提 kernel，P10 形成 spmd_submit；P39 / P40 对照仍保留显式 deps。不能把既有的全任务等待归因于 P40 新增。creator / tensormap 等实际依赖还需与 deps.json 联合解释。", next: "优化问题：是否存在安全的分片依赖空间？这需要校验数据范围、通知语义和消费者一致性，并重新运行验证。", sources: [dump("00_frontend", [1707, 1710], "前端 · 原有显式依赖"), dump("09_after_OutlineIncoreScopes", [2739, 2758, 2759, 2762, 2763], "P09 · 前驱与外提调用"), dump("10_after_OutlineClusterScopes", [2973], "P10 · 降为提交"), dump("39_after_DeriveCallDirections", [5030], "P39 · 依赖推导前"), dump("40_after_AutoDeriveTaskDependencies", [5030], "P40 · 显式依赖保留")] },
      { pass: "联动", name: "前驱收尾 / 依赖粒度", title: "返回泳道追查前驱", summary: "继续追查 qk_pv 收尾与依赖粒度。", finding: "当前时序支持前驱依赖等待；调整依赖粒度需要校验数据范围与消费者一致性，并以运行对照验证。", visual: "actions", next: "在泳道中查看 qk_pv 的执行与当前任务的开始时间。", sources: [] },
    ],
  };
}
