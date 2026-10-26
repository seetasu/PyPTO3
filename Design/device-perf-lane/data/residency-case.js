// Curated public evidence, verified against the GitHub API on 2026-10-24.
// Event-level traces are synthesized by residency-case.js, not collected on hardware.
window.RESIDENCY_CASE_DATA = {
  id: 'pr2080', title: '#2080 · 固定矩阵为什么被搬了 32 次？',
  issue: 'https://github.com/hw-native-sys/pypto/issues/2077',
  pr: 'https://github.com/hw-native-sys/pypto/pull/2080',
  files: 'https://github.com/hw-native-sys/pypto/pull/2080/files',
  commit: '3a01119f9c65e0772998f0c945762d3c15abcfd3', merged: '2026-07-24',
  implementation: 'https://github.com/hw-native-sys/pypto/blob/3a01119f9c65e0772998f0c945762d3c15abcfd3/src/ir/transforms/loop_invariant_mat_residency.cpp',
  integration: 'https://github.com/hw-native-sys/pypto/blob/3a01119f9c65e0772998f0c945762d3c15abcfd3/src/ir/transforms/infer_tile_memory_space_pass.cpp',
  cases: [
    { id: 'k5t32', label: 'K5 · T=32 · 128 KiB', panel: 131072, trips: 32, before: 118.03, after: 99.49, reported: 15.7, ci: [0.837, 0.848] },
    { id: 'k5t8', label: 'K5 · T=8 · 128 KiB', panel: 131072, trips: 8, before: 31.34, after: 27.27, reported: 13.0, ci: [0.857, 0.892] },
    { id: 'k2t32', label: 'K2 · T=32 · 64 KiB', panel: 65536, trips: 32, before: 99.15, after: 90.64, reported: 8.6, ci: [0.910, 0.921] },
    { id: 'k1t32', label: 'K1 · T=32 · 4 KiB', panel: 4096, trips: 32, before: 28.46, after: 26.73, reported: 6.1, ci: [0.930, 0.944] },
  ],
  validation: { inference: 89, transforms: 2285, codegen: 582, devices: 2, device: 'Ascend 910B2', ptoas: 'v0.50', input: 'BF16', output: 'FP32' },
  // The T=8 traffic arithmetic is derived; the PR's exact simulator byte table is T=32 only.
  traffic: { k5t32: 4063232, k2t32: 2031616, k1t32: 126976 },
};
