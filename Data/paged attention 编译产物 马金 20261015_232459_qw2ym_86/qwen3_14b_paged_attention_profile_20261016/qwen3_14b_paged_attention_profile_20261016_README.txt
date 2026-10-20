Qwen3-14B Page Attention profiling bundle, 2026-10-16

Hardware run (allocated task-submit device 15):
  task ID: task_20261016_035019_202196021438
  command: python models/qwen3_14b/test_paged_attention_pypto.py -p a2a3 -d 15 --batch 2 --seq-lens 512,513 --page-mapping reverse --enable-chip-swimlane 4 --enable-pmu 7
  status: PASS; key_cache, value_cache, out validated
  raw build: build_output/_jit_paged_attention_pypto_dynamic_20261016_035027_48baglaw/
  DFX: dfx_outputs/chip_swimlane_records.json, deps.json, merged_swimlane_20261016_035043.json, pmu.csv
  PMU group: 7 (MEMORY_UB / MTE bandwidth); pmu.csv has 72 data rows
  log: pypto-lib-pa-swimlane-pmu7-card15-20261016.log

In-core msprof op simulator (allocated task-submit device 15):
  task ID: task_20261016_035044_243564522975
  kernel source: build_output/_jit_paged_attention_pypto_dynamic_20261016_024040_yhb_got8/ptoas/attn_swpipe.cpp
  kernel SHA256: 50735cac9c9e0f8d0caa41ad55fe83c51106c12891df7ec41109549ecd838ccb
  This SHA256 equals the hardware run's generated attn_swpipe.cpp SHA256.
  raw output: build_output/_jit_paged_attention_pypto_dynamic_20261016_024040_yhb_got8/kernel_insight_all_funcs_20261016_024701/funcs/attn_swpipe/collect/out/OPPROF_20261016035052_VMLWAAXWMNZVNZVI/
  status: msprof success; simulator/trace.json contains 5749 events and there are nonempty AIC/AIV instruction CSV files
  simulated workload: standalone single-core generated case with synthetic inputs, seq_lens 128,129; not the hardware run's tensor inputs or schedule
  see kernel_insight_all_funcs_20261016_024701/cases/ptoas/attn_swpipe_msprof/WORKLOAD.txt and MANUAL_COLLECT_STATUS.txt
  log: pypto-lib-pa-msprof-card15-20261016.log

PTO ISA Perf-Sim cost-model attempt:
  build_output/_jit_paged_attention_pypto_dynamic_20261016_024040_yhb_got8/costmodel_attn_swpipe/
  status: compile failed with the available host toolchain and PTO ISA headers; no cost-model estimate or trace was generated
  build.log and STATUS.txt document the failure. Do not treat this directory as a measured result.

The hardware DFX data, msprof simulator data, and failed cost-model attempt are kept in their original directory structure. The msprof simulator cycle counts are modeled values, not PMU hardware measurements.
