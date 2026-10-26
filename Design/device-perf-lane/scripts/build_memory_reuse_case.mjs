import { readFileSync } from "node:fs";
import { resolve } from "node:path";

// Match stable SSA names across P34/P35, retaining their pre-reuse stage provenance.
export function buildMemoryReuseCase(root, run, dump) {
  const beforeFile = "34_after_MaterializeSemanticAliases", afterFile = "35_after_MemoryReuse";
  function body(file) {
    const lines = readFileSync(resolve(root, run, `passes_dump/${file}.py`), "utf8").split(/\r?\n/);
    const start = lines.findIndex(line => line.startsWith("    def csa_merge_pack_publish("));
    if (start < 0) throw new Error(`Missing kernel: ${file}`);
    let end = start + 1;
    while (end < lines.length && !/^    (def |@pl.function)/.test(lines[end])) end++;
    return lines.slice(start, end).map((text, i) => ({ line: start + i + 1, text }));
  }
  const before = body(beforeFile), after = body(afterFile), mappings = [];
  const describe = {
    m_cur_oi_inline1216__tile: "载入 oi 分片", t__tile_5: "新分片加权",
    m_oi_inline1220__tile_1: "更新累积结果", t__tile_10: "上轮结果加权",
  };
  for (const row of before) {
    const stage = row.text.match(/pipeline_membership": "0:(\d+)"/);
    const variable = row.text.trim().match(/^(\w+):/);
    const mem = row.text.match(/pl.MemRef\((mem_\w+),.*?, (\d+)\), pl.Mem.Vec/);
    const op = row.text.match(/= pl.tile.(\w+)\(/);
    if (!stage || !variable || !mem || !op) continue;
    const target = after.find(line => line.text.trim().startsWith(variable[1] + ":"));
    const result = target?.text.match(/pl.MemRef\((mem_\w+),.*?, (\d+)\), pl.Mem.Vec/);
    if (!result) continue;
    mappings.push({ id: variable[1], stage: Number(stage[1]), op: op[1], label: describe[variable[1]] || op[1],
      bytes: Number(mem[2]), beforeBuffer: mem[1], afterBuffer: result[1],
      before: dump(beforeFile, [row.line], "P34 · 复用前"), after: dump(afterFile, [target.line], "P35 · 复用后") });
  }
  const crossStage = [...new Set(mappings.map(value => value.afterBuffer))].filter(buffer =>
    new Set(mappings.filter(value => value.afterBuffer === buffer).map(value => value.stage)).size > 1);
  const buffers = crossStage.map(buffer => {
    const values = mappings.filter(value => value.afterBuffer === buffer && !["reshape", "slice", "extract"].includes(value.op));
    const declaration = after.find(row => row.text.trim().startsWith(buffer + ":") && row.text.includes("pl.tile.alloc"));
    const bytes = Number(declaration?.text.match(/pl.tile.alloc\(pl.Mem.Vec, (\d+)\)/)?.[1]);
    if (!bytes || !values.length || new Set(values.map(v => v.stage)).size < 2) throw new Error(`Incomplete buffer evidence: ${buffer}`);
    return { id: buffer, bytes, values, totalMappedValues: mappings.filter(value => value.afterBuffer === buffer).length,
      allocation: dump(afterFile, [declaration.line], "P35 · 物理缓冲声明") };
  }).sort((a, b) => Number(b.id === "mem_vec_15") - Number(a.id === "mem_vec_15") || b.bytes - a.bytes || a.id.localeCompare(b.id));
  const primary = buffers.find(buffer => buffer.id === "mem_vec_15");
  if (!primary || primary.values.length !== 4 || primary.values.find(value => value.id === "t__tile_10")?.stage !== 1) throw new Error("Primary cross-stage mapping changed");
  const path = `${run}/report/perf_hints.log`, warningText = readFileSync(resolve(root, path), "utf8").split(/\r?\n/)[0];
  if (!warningText.includes("MemoryReuse") || !warningText.includes("only 1 of 2 buffers fit") || !warningText.includes("co-resident")) throw new Error("Diagnostic changed");
  const warning = { path, label: "PH-MR-001 · 原始编译诊断", lines: [{ line: 1, text: warningText }] };
  const mappingPath = `${run}/next_levels/decode_csa_test/ptoas/csa_merge_pack_publish.pto`;
  const pto = readFileSync(resolve(root, mappingPath), "utf8").split(/\r?\n/);
  const location = pto.findIndex(text => text.includes('decode_csa.py":397:9'));
  if (location < 0) throw new Error("Cannot bind diagnostic source location to kernel");
  return { id: "memoryreuse-depth", title: "P35 MemoryReuse · 流水隔离深度退化", kernel: "csa_merge_pack_publish", requested: 2, achieved: 1,
    group: 0, space: "Vec", stageBytes: 32768, availableBytes: 188416, buffers, warning,
    mapping: { path: mappingPath, label: "最终 PTO · 诊断源码位置对应", lines: [{ line: location + 1, text: pto[location] }] },
    boundary: "这是资源约束下的编译退化案例，尚未认定为编译器缺陷。图中结构与映射来自快照；没有独立的指令级停顿计时，也未测得恢复深度 2 的收益。",
    steps: [
      { pass: "请求", name: "stage=2 → PH-MR-001", title: "确认 2 → 1 的诊断", visual: "memory-request", sources: [dump("29_after_LowerPipelineToSlots", [297, 301], "P29 · 请求两阶段流水")], finding: "已确认：同一 kernel 的 group 0 在 Vec 请求深度 2，MemoryReuse 报告只满足 1。effective depth 是阶段存储隔离能力，不是 kernel 的总缓冲数。", next: "追踪跨阶段的具体存储映射。" },
      { pass: "P34 → P35", name: "SSA value → MemRef", title: "哪些缓冲被跨阶段复用", visual: "memory-map", sources: [], finding: "按 SSA 名称匹配 P34/P35，阶段来源取自 P34 的 pipeline_membership。只展示计算值，reshape/view 别名不重复列出；同址复用事实不等于每个复用都是可去除的假依赖。", next: "查看同驻空间约束为什么使隔离深度下降。" },
      { pass: "P35", name: "Global Vec capacity pressure", title: "为什么不能维持深度 2", visual: "memory-capacity", sources: [], finding: "PH-MR-001 明确区分：单组本身可以容纳两阶段，失败来自同驻缓冲 / 其他流水组的整体空间竞争。184 KiB 是扣保留区后的可规划容量，不是剩余空闲量；终态缓冲声明不能代替峰值存活分析。", next: "核对已经确认的结构与尚未验证的性能影响。" },
      { pass: "边界", name: "Confirmed / needs verification", title: "证据与影响分开看", visual: "memory-verification", sources: [], finding: "已确认的是流水请求、诊断、阶段来源与存储复用映射。尚未确认容量临界的具体存活集合、哪条等待限制执行、以及任何加速比例。", next: "需要内存规划的峰值 / 冲突输出、指令级轨迹和相同输入的编译运行对照。" },
    ] };
}
