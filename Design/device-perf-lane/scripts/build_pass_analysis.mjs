// Curated evidence, copied from this run's original artifacts; no browser fetch required.
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { resolve, dirname } from "node:path";
import { buildSetupAnalysis } from "./build_setup_analysis.mjs";
import { buildMemoryReuseCase } from "./build_memory_reuse_case.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "../../..");
const run = "Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617";
function excerpt(file, lines, label) {
  const path = `${run}/${file}`;
  const source = readFileSync(resolve(root, path), "utf8").split(/\r?\n/);
  return { path, label, lines: lines.map(line => {
    if (!source[line - 1]) throw new Error(`Missing source: ${path}:${line}`);
    return { line, text: source[line - 1] };
  }) };
}
const dump = (file, lines, label) => excerpt(`passes_dump/${file}.py`, lines, label);
const evidence = {
  kernel: "csa_merge_pack_publish", run: "_jit_l3_decode_csa_20260903_010617",
  warning: excerpt("report/perf_hints.log", [1], "PH-MR-001 · 编译器诊断"),
  mapping: excerpt("next_levels/decode_csa_test/ptoas/csa_merge_pack_publish.pto", [59], "告警与 kernel 的源码位置对应"),
  steps: [
    { pass: "P29", name: "LowerPipelineToSlots", title: "确认流水请求", summary: "循环请求 stage=2；每份 [16, 512] FP32 tile 为 32 KiB。", finding: "已确认：在循环降低前仍保留两阶段流水请求。这是预期结构，不代表最终一定有两份可重叠使用的缓冲。", next: "下一步查看 P30 如何把循环展开为两个迭代阶段。", sources: [dump("29_after_LowerPipelineToSlots", [297, 301], "P29 · 流水请求与 tile 大小")] },
    { pass: "P30", name: "LowerPipelineLoops", title: "追踪两个迭代阶段", summary: "步长变为 2，生成 0:0 / 0:1 两组 pipeline_membership。", finding: "已确认：循环体包含两个迭代阶段。这里的 stage=1 是降低后的表达，不能单凭这一字段认定流水退化；实际缓冲约束要继续看 P35 与编译告警。", next: "下一步检查内存复用后的实际存储关系。", sources: [dump("29_after_LowerPipelineToSlots", [297, 301], "降低前 · P29"), dump("30_after_LowerPipelineLoops", [358, 362, 400], "降低后 · P30")] },
    { pass: "P35", name: "MemoryReuse", title: "定位缓冲复用约束", summary: "编译器报告：请求深度 2，但相关缓冲仅能满足 1 / 2。", finding: "已确认：PH-MR-001 报告同驻缓冲造成空间竞争，阶段共享存储并串行化。IR 同时显示 mem_vec_15 从加载结果复用为中间计算结果，两个加载分别使用 mem_vec_15 与 mem_vec_18。告警未指明具体变量；此处展示可见的复用实例，尚不能认定它就是告警对象。", next: "分析方向：检查同驻 tile 的生命周期与占用。缩小 tile 或改变流水深度的收益，需要重新编译和运行对照。", sources: [dump("34_after_MaterializeSemanticAliases", [561, 564, 596, 597, 598], "复用前 · P34"), dump("35_after_MemoryReuse", [400, 403, 435, 436, 437, 469, 470, 471], "复用后 · P35")] },
    { pass: "P50", name: "InsertCommFence", title: "确认通信 Fence 插入", summary: "两处 Put 后各新增一条 fence；两份 staging tile 使用同一缓冲区间。", finding: "已确认：P49 尚无这两条 Fence，P50 在两个 Put 后插入。两处 staging 都使用 mem_vec_66、偏移 99584、大小 65536B。缓冲复用与通信同步是需要联合检查的线索；快照不能单独证明 Fence 冗余或确定其全部插入原因。", next: "分析方向：结合通信完成语义检查同步要求，再用事件计时确认成本；不能直接删除 Fence。此分支与流水缓冲约束都作用于同一 kernel，不表示前者必然导致后者。", sources: [dump("49_after_ClassifyIterArgCarry", [601, 602, 605, 606], "插入前 · P49"), dump("50_after_InsertCommFence", [605, 606, 607, 610, 611, 612], "插入后 · P50")] },
  ],
};
evidence.boundary = "归因边界：以下为同一 kernel 的编译证据，尚未证明流水串行化或 Fence 造成了高 Setup；需核实 local_setup_us 计时边界并做运行对照。";
evidence.steps[2].diagnostic = {
  title: "PH-MR-001 · 已报告的编译约束",
  text: "请求 2 阶段 → 相关缓冲仅满足 1 / 2；每阶段 32 KiB。告警指出同驻缓冲竞争导致阶段共享存储并串行化。",
  note: "告警与该 kernel 的 PTO 源码位置均指向 decode_csa.py:397:9。",
  sources: [evidence.warning, evidence.mapping],
};
delete evidence.warning;
delete evidence.mapping;
const backend = (kernel, ext, lines, label) => excerpt(`next_levels/decode_csa_test/ptoas/${kernel}.${ext}`, lines, label);
const matmul = "qr_hadamard_matmul", quant = "qr_hadamard_quant";
const cases = {
  [evidence.kernel]: evidence,
  [matmul]: {
    kernel: matmul,
    boundary: "归因边界：tile、布局和搬运粒度为静态编译证据；PH001 是后端建议，不代表实测带宽或缓存缺失。慢块差异和低并行度还需结合启动间隔、同核任务与运行计数器分析。",
    steps: [
      { pass: "P00", name: "frontend → OutlineIncoreScopes", title: "追溯 64 行切分", summary: "每个 block 计算 [64,128] × [128,128]，输出 [64,128] FP32。", finding: "已确认：bs_heads // 64 与每块 64 行在前端就已写定，P09 外提后保持这一规模。当前快照未显示不同 block 有不同大小的矩阵计算。", next: "下一步检查张量计算怎样变成 tile 加载、矩阵乘和写回。", facts: [["A 输入", "64 × 128 · BF16", "16 KiB / block"], ["B 输入", "128 × 128 · BF16", "32 KiB / block"], ["C 输出", "64 × 128 · FP32", "32 KiB / block"]], sources: [dump("00_frontend", [690, 691, 692, 693], "前端 · SPMD 与切片"), dump("09_after_OutlineIncoreScopes", [1619, 1620, 1621, 1622, 1623], "P09 · 外提后的 kernel")] },
      { pass: "P11", name: "ConvertTensorToTileOps → AutoTileMatmulL0", title: "核对 tile 粒度", summary: "生成两次 Mat 加载、一次 tile.matmul 和一次写回；P17 后该函数仍保持同一 tile 规模。", finding: "已确认：P11 将张量切片与 assemble 降为显式 tile.load / tile.store。P17 快照没有改变这个 kernel 的 [64,128] 与 [128,128] 形状，不能把固定切分归因于 P17 新增，也不能由此认定缺少某项优化。", next: "下一步沿 Mat → Left / Right → Acc 追踪内存空间和后端布局。", sources: [dump("11_after_ConvertTensorToTileOps", [1873, 1874, 1875, 1876], "P11 · 显式 Tile 操作"), dump("17_after_AutoTileMatmulL0", [1943, 1944, 1945, 1946], "P17 · tile 规模保持一致")] },
      { pass: "P19", name: "InferTileMemorySpace → PTOAS", title: "追踪布局与搬运路径", summary: "输入经 Mat 搬到 Left / Right，乘积位于 Acc；最终 PTO 给出具体块布局。", finding: "已确认：P19 新增两次 tile.move；最终 A 的 Left tile 使用 col_major / row_major，B 的 Right tile 使用 row_major / col_major。它们是块布局与子布局，不能直接等同于 GM 连续行宽，也不能仅凭布局不同认定发生了多余转置。", next: "下一步对照 PH001，定位被提示的输入加载及最终 TLOAD / TMOV 指令。", facts: [["A 路径", "GM → Mat → Left", "col_major / row_major"], ["B 路径", "GM → Mat → Right", "row_major / col_major"], ["输出路径", "Acc → GM", "FP32 · 32 KiB"]], sources: [dump("18_after_CanonicalizeTileSlice", [1944, 1945, 1946, 1947], "P18 · 空间推断前"), dump("19_after_InferTileMemorySpace", [1960, 1961, 1962, 1963, 1964, 1965], "P19 · 增加 Left / Right 搬运"), backend(matmul, "pto", [17, 20, 22, 24, 26, 29], "最终 PTO · 内存空间与布局")] },
      { pass: "PTOAS", name: "Generated code · PH001", title: "定位输入搬运告警", summary: "两个 BF16 输入的内层跨度均为 128 × 2 = 256B；后端建议至少 512B。", finding: "已确认：告警映射到 decode_indexer.py:727:9，最终代码包含两次 TLOAD、两次 TMOV、一次 TMATMUL 和一次 TSTORE。总 tile 大小不等于单行搬运跨度；256B / 512B 的比值不能当作实测搬运效率。", next: "分析方向：比较满足矩阵语义与片上容量的 tile / 数据布局方案，验证搬运计数器与端到端耗时；扩大 M 不会改变固定 K=128 对应的 256B 内层跨度。", diagnostic: { title: "PH001 · 输入搬运粒度", text: "A：64 × 256B = 16 KiB；B：128 × 256B = 32 KiB。两条告警均建议内层跨度至少 512B。", note: "通过 PTO 的源码位置映射到同一 kernel；这是静态建议，不是实际 GM 事务数。", sources: [excerpt("report/perf_hints.log", [83, 84], "PH001 · 两个输入加载"), backend(matmul, "pto", [17, 20], "源码位置 · decode_indexer.py:727:9")] }, facts: [["A 加载", "BF16[128] · 256B", "建议 ≥ 512B"], ["B 加载", "BF16[128] · 256B", "建议 ≥ 512B"]], sources: [backend(matmul, "cpp", [58, 71, 79, 86, 94, 103], "最终 C++ · 加载、搬运、计算与写回")] },
    ],
  },
  [quant]: {
    kernel: quant,
    boundary: "归因边界：两轮读取、布局和转换链可从编译产物确认；静态字节数不等于实际 GM 流量。缓冲复用不代表输入只读取一次；尾波填充与慢块差异仍需运行时证据。",
    steps: [
      { pass: "P00", name: "frontend", title: "确认两轮输入访问", summary: "每个 block 处理 64 行；第一轮求幅值最大值，第二轮用量化 scale 再次读取并转换。", finding: "已确认：两轮 h0 / h1 循环在前端已存在，各读取两个 [64,64] FP32 分片。每块按代码可计 64 KiB 输入加载，唯一输入数据为 32 KiB；这不是硬件实测流量，缓存可能改变实际 GM 访问。", next: "下一步检查切片、归约和转换如何降低为 Tile 操作。", facts: [["第一轮", "2 × [64,64] FP32", "32 KiB · 求最大值"], ["第二轮", "2 × [64,64] FP32", "32 KiB · 量化"], ["输出", "[64,1] FP32 + [64,128] INT8", "256B + 8 KiB"]], sources: [dump("00_frontend", [696, 698, 700, 701, 703, 705, 707, 709, 710, 712, 713, 714, 715], "前端 · 两轮切片及转换")] },
      { pass: "P11", name: "ConvertTensorToTileOps", title: "定位加载与转换链", summary: "两个切片循环变成两组 tile.load，scale 和 INT8 结果分别写回。", finding: "已确认：降低后保留 FP32 → INT32（rint）→ FP16（round）→ INT8（trunc）。这条链来自前端，并带有不同舍入语义；不能直接合并成一次转换而假设数值结果相同。", next: "下一步比较内存复用前后的缓冲归属，区分复用存储与保留输入数据。", sources: [dump("09_after_OutlineIncoreScopes", [1630, 1631, 1642, 1644, 1645, 1647, 1648, 1649], "P09 · 张量级切片与转换"), dump("11_after_ConvertTensorToTileOps", [1887, 1888, 1900, 1902, 1903, 1905, 1906, 1907, 1908], "P11 · Tile 加载与多级转换")] },
      { pass: "P35", name: "MemoryReuse", title: "区分缓冲复用与少读一次", summary: "输入加载和量化中间结果复用 mem_vec_7；两个循环中的加载仍保留。", finding: "已确认：P34 为输入和转换结果指定不同缓冲，P35 将其中多项结果放入 mem_vec_7。转换逐步覆盖该缓冲；复用了地址并不表示第一轮的全部输入被保存到第二轮。是否能保留完整输入，需要重新评估同时存活的 tile、容量和依赖。", next: "下一步检查最终布局及 PH001 的具体加载、scale 写回和 INT8 写回位置。", sources: [dump("34_after_MaterializeSemanticAliases", [4413, 4428, 4430, 4431, 4432], "P34 · 复用前的缓冲归属"), dump("35_after_MemoryReuse", [3417, 3418, 3419, 3420, 3425, 3440, 3442, 3443, 3444], "P35 · 共享缓冲与保留的加载")] },
      { pass: "PTOAS", name: "Generated layout · PH001", title: "定位读写粒度告警", summary: "FP32 输入分片内层 256B，scale 写回 4B，INT8 分片写回 64B。", finding: "已确认：最终输入 tile 为 row_major，scale tile 为 col_major，INT8 tile 为 row_major；这些片上布局应与 GM 视图共同分析。PH001 提示三个方向的搬运粒度，不能把 scale tile 的 col_major 直接当作 GM 连续布局，也不能由告警计算实际带宽。", next: "分析方向：评估输入保留、scale 输出组织与量化分片方案。改变 tile 或布局时同时校验数值、容量与搬运计数器；当前 256 blocks / 48 AIV 核的尾波只是块数推算，不能归因于该告警。", facts: [["输入加载 · 两轮", "FP32[64] · 256B", "建议 ≥ 512B"], ["scale 写回", "FP32[1] · 4B", "总计 64 × 4B"], ["INT8 分片写回", "INT8[64] · 64B", "每片 64 × 64B"]], diagnostic: { title: "PH001 · 加载与写回粒度", text: "两次输入加载、scale 写回和 INT8 写回均有静态粒度提示；建议值是后端规则，不是实测效率。", note: "四条告警通过 PTO loc 分别对应 decode_indexer.py:745、751、754、759。", sources: [excerpt("report/perf_hints.log", [85, 86, 87, 88], "PH001 · 四条读写告警"), backend(quant, "pto", [26, 47, 54, 66], "最终 PTO · 布局与源码位置")] }, sources: [backend(quant, "cpp", [64, 77, 154, 162, 177, 215], "最终 C++ · 两轮加载和两类写回")] },
    ],
  },
};
evidence.setupAnalysis = buildSetupAnalysis(root, run, dump);
evidence.steps[0].sources = [dump("28_after_SkewCrossCorePipeline", [297, 301, 333, 334], "P28 · 输入循环"), dump("29_after_LowerPipelineToSlots", [297, 301, 333, 334], "P29 · 输出循环")];
evidence.boundary = "核内分析：流水缓冲与通信 Fence 均发生在 kernel 执行窗口内，不能用它们解释本次主要由执行前等待构成的高 Setup。优化效果需重编译并做运行对照。";
evidence.steps[3].finding = "已确认：P50 在两个 Put 后分别插入 GM Fence。两处 staging 复用同一区间，但 Fence 的用途是满足通信发布的数据先可见、再通知契约；不能将插入原因归为缓冲复用。最终 PTO 与 C++ 可见 GM Fence / DDR 屏障。";
const payload = { run: evidence.run, cases, independentCases: { memoryReuse: buildMemoryReuseCase(root, run, dump) } };
writeFileSync(resolve(here, "../data/pass-analysis.js"), `// Generated by scripts/build_pass_analysis.mjs.\nwindow.PERF_PASS_ANALYSIS = ${JSON.stringify(payload, null, 2)};\n`, "utf8");
console.log("Generated data/pass-analysis.js from original pass snapshots.");
