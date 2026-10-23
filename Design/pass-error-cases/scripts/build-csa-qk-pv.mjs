import fs from 'node:fs';
import crypto from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseDump, resetParserErrors, parserErrors } from '../../pass-transform-explorer/lib/pyir.mjs';
import { analyzeProgram } from '../../pass-transform-explorer/lib/analyze.mjs';
import { lensOf, PHASES } from '../../pass-transform-explorer/lib/passinfo.mjs';
import { csaLines, controlFacts, structuralFacts } from './visual-facts.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '../../..');
const OUT = path.join(HERE, '../data/csa-qk-pv');
const PASS_DIR_REL = 'Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/passes_dump';
const PTO_REL = 'Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/next_levels/decode_csa_test/ptoas/qk_pv.pto';
const CPP_REL = 'Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/next_levels/decode_csa_test/ptoas/qk_pv.cpp';
const PASS_DIR = path.join(ROOT, PASS_DIR_REL);
const PTO_FILE = path.join(ROOT, PTO_REL);
const CPP_FILE = path.join(ROOT, CPP_REL);

const ref = (file, line = null, evidence = 'observed') => ({ file, ...(line ? { line } : {}), evidence });
const phaseForIndex = (index) => {
  if (index <= 7) return 'frontend';
  if (index <= 10) return 'outline';
  if (index <= 22) return 'lowering';
  if (index <= 31) return 'pipeline';
  if (index <= 38) return 'memory';
  if (index <= 41) return 'task';
  if (index <= 46) return 'dist';
  return 'runtime';
};
const node = (id, semanticKind, renderKind, label, role, attributes = {}, provenance = 'curated', sourceRefs = []) => ({
  id, semanticKind, renderKind, label, role, attributes, identity: { lineageRoot: id, aliases: [] }, provenance, sourceRefs,
});
const edge = (id, semanticKind, source, target, sourcePort = 'out', targetPort = 'in', attributes = {}, provenance = 'curated', sourceRefs = []) => ({
  id, semanticKind, source, target, sourcePort, targetPort, attributes, provenance, sourceRefs,
});

// Keep source line numbers while embedding only changed lines and their immediate context.
function changedExcerpt(text, other) {
  const otherLines = new Set(other.split('\n'));
  const lines = text.split('\n');
  const selected = new Set();
  lines.forEach((line, index) => { if (!otherLines.has(line)) for (let i = Math.max(0, index - 1); i <= Math.min(lines.length - 1, index + 1); i++) selected.add(i); });
  return [...selected].sort((a,b) => a-b).map(i => `${i + 1}: ${lines[i]}`).join('\n') || '本侧没有独有行；参见另一侧的新增/删除内容。';
}

function readPassSnapshots() {
  const files = fs.readdirSync(PASS_DIR).filter((file) => /^\d+_.*\.py$/.test(file)).sort();
  const snapshots = [];

  for (const file of files) {
    const match = file.match(/^(\d+)_(.*)\.py$/);
    const index = Number(match[1]);
    const name = match[2] === 'frontend' ? 'frontend' : match[2].replace(/^after_/, '');
    const fileRel = `${PASS_DIR_REL}/${file}`;
    const text = fs.readFileSync(path.join(PASS_DIR, file), 'utf8');
    resetParserErrors();
    const program = parseDump(text, file);
    const analysis = analyzeProgram(program, text.split(/\r?\n/));
    const failures = parserErrors();
    if (failures.length) console.warn(`! ${file}: ${failures.length} expression parse failures`);

    const family = analysis.functions
      .filter((fn) => /^qk_pv(?:_|$)/.test(fn.name))
      .sort((a, b) => a.name.localeCompare(b.name));
    const familyText = family.map((fn) => `${fn.name}\n${(fn.src || []).join('\n')}`).join('\n\n');
    snapshots.push({
      index,
      name,
      file,
      fileRel,
      family,
      familyText,
      sourceText: text,
      metrics: {
        callables: family.length,
        statements: family.reduce((sum, fn) => sum + fn.stmtCount, 0),
        loops: family.reduce((sum, fn) => sum + fn.loops.length, 0),
        allocations: family.reduce((sum, fn) => sum + fn.allocs.length, 0),
        tasks: family.reduce((sum, fn) => sum + fn.tasks.length, 0),
      },
      sourceRefs: family.map((fn) => ref(fileRel, fn.decoLine + 1)).slice(0, 6),
    });
  }

  if (snapshots.length !== 52 || snapshots[0]?.name !== 'frontend') {
    throw new Error(`Expected frontend + 51 Pass snapshots, found ${snapshots.length}`);
  }
  return snapshots;
}

function extractIsa() {
  const source = fs.readFileSync(PTO_FILE, 'utf8');
  const cpp = fs.readFileSync(CPP_FILE, 'utf8');
  const selected = [
    'tload', 'tstore', 'tmatmul', 'tmatmul.acc', 'texp', 'trowmax', 'trowsum',
    'tpush_to_aiv', 'tpush_to_aic', 'tpop_from_aiv', 'tpop_from_aic', 'tfree_from_aiv', 'tfree_from_aic',
  ];
  const apiMap = {
    tload: 'TLOAD', tstore: 'TSTORE', tmatmul: 'TMATMUL', 'tmatmul.acc': 'TMATMUL',
    texp: 'TEXP', trowmax: 'TROWMAX', trowsum: 'TROWSUM',
    tpush_to_aiv: 'TPUSH', tpush_to_aic: 'TPUSH', tpop_from_aiv: 'TPOP', tpop_from_aic: 'TPOP',
    tfree_from_aiv: 'TFREE', tfree_from_aic: 'TFREE',
  };
  const unitMap = {
    tload: 'mixed', tstore: 'mixed', tmatmul: 'AIC', 'tmatmul.acc': 'AIC',
    texp: 'AIV', trowmax: 'AIV', trowsum: 'AIV',
    tpush_to_aiv: 'AIC', tpop_from_aic: 'AIV', tfree_from_aic: 'AIV',
    tpush_to_aic: 'AIV', tpop_from_aiv: 'AIC', tfree_from_aiv: 'AIC',
  };
  const occurrences = new Map(selected.map((name) => [name, []]));
  source.split(/\r?\n/).forEach((line, index) => {
    for (const match of line.matchAll(/\bpto\.(tmatmul\.acc|[a-z0-9_]+)/g)) {
      if (occurrences.has(match[1])) occurrences.get(match[1]).push(index + 1);
    }
  });

  return selected.map((name) => {
    const apiName = apiMap[name];
    const cppVerified = cpp.includes(`${apiName}(`);
    return {
      id: `isa.${name.replace(/\./g, '-')}`,
      mlirName: `pto.${name}`,
      apiName: cppVerified ? apiName : null,
      apiVerifiedInCpp: cppVerified,
      family: name.startsWith('tmatmul') ? 'matrix' : name.startsWith('trow') || name === 'texp' ? 'vector' : name.startsWith('tpush') || name.startsWith('tpop') || name.startsWith('tfree') ? 'cross-core' : 'memory',
      executionUnit: unitMap[name],
      occurrenceCount: occurrences.get(name).length,
      countMeaning: 'static source occurrence in qk_pv.pto; not runtime execution count',
      sourceRefs: occurrences.get(name).map((line) => ref(PTO_REL, line)),
      occurrences: occurrences.get(name).map((line) => {
        const lines = source.split(/\r?\n/);
        const prefix = lines.slice(0, line).join('\n');
        const functions = [...prefix.matchAll(/func\.func @([\w]+)/g)];
        return { line, kernel: functions.at(-1)?.[1], text: lines[line - 1].trim() };
      }),
      apiSourceRefs: cppVerified ? [ref(CPP_REL, null, 'observed-api-name')] : [],
    };
  });
}

const allNodes = [
  node('scope.qk-pv', 'scope', 'scope', 'qk_pv · mixed InCore scope', 'CSA callable scope', { level: 'InCore', collapsedCount: 8 }, 'derived', [ref(`${PASS_DIR_REL}/00_frontend.py`)]),
  node('value.q', 'tensor_value', 'tensor', 'Q', 'query activation', { shape: ['T_HEADS', 512], dtype: 'BF16', tensorRole: 'i' }),
  node('value.kv', 'tensor_value', 'tensor', 'KV blocks', 'key/value cache blocks', { shape: [32, 512], dtype: 'BF16', tensorRole: 'i' }),
  node('op.qk', 'op', 'op', 'tensor.matmul · QKᵀ', 'QK score projection', { domain: 'Tensor' }),
  node('value.scores', 'tensor_value', 'tensor', 'QK scores', 'score tile', { shape: [32, 128], dtype: 'FP32', tensorRole: 'i' }),
  node('op.rowmax', 'op', 'op', 'tensor.row_max', 'online softmax row maximum', { domain: 'Tensor' }),
  node('value.max', 'tensor_value', 'tensor', 'mᵢ', 'row maximum accumulator', { shape: [32, 1], dtype: 'FP32', tensorRole: 'a' }),
  node('op.exp', 'op', 'op', 'tensor.exp', 'shift and exponentiate', { domain: 'Tensor' }),
  node('value.prob', 'tensor_value', 'tensor', 'exp(score − mᵢ)', 'softmax numerator', { shape: [32, 128], dtype: 'BF16', tensorRole: 'i' }),
  node('op.rowsum', 'op', 'op', 'tensor.row_sum', 'online softmax row sum', { domain: 'Tensor' }),
  node('value.sum', 'tensor_value', 'tensor', 'ℓᵢ', 'row sum accumulator', { shape: [32, 1], dtype: 'FP32', tensorRole: 'a' }),
  node('op.pv', 'op', 'op', 'tensor.matmul · PV', 'probability-value accumulation', { domain: 'Tensor' }),
  node('value.out', 'tensor_value', 'tensor', 'Oᵢ', 'partial attention output', { shape: [16, 512], dtype: 'FP32', tensorRole: 'i' }),
  node('kernel.aic', 'kernel', 'kernel', 'qk_pv_aic', 'Cube kernel', { target: 'AIC', detail: 'QK + PV matmul' }, 'observed', [ref(`${PASS_DIR_REL}/23_after_ExpandMixedKernel.py`, 1714)]),
  node('kernel.aiv', 'kernel', 'kernel', 'qk_pv_aiv', 'Vector kernel', { target: 'AIV', detail: 'online softmax' }, 'observed', [ref(`${PASS_DIR_REL}/23_after_ExpandMixedKernel.py`, 1851)]),
  node('handoff.c2v-push', 'op', 'handoff', 'C2V · push', 'Cube to Vector transfer', { direction: 'AIC → AIV', slots: 2 }),
  node('handoff.c2v-pop', 'op', 'handoff', 'C2V · pop', 'Vector receives scores', { direction: 'AIC → AIV', slots: 2 }),
  node('handoff.v2c-push', 'op', 'handoff', 'V2C · push', 'Vector to Cube transfer', { direction: 'AIV → AIC', slots: 2 }),
  node('handoff.v2c-pop', 'op', 'handoff', 'V2C · pop', 'Cube receives probabilities', { direction: 'AIV → AIC', slots: 2 }),
  node('memory.qk-pv', 'memref', 'memory', 'qk_pv buffers · 分空间展开', 'pre-reuse local buffers', { space: 'Mat / Left / Right / Acc / Vec', buffers: 53, bytes: 944128 }),
  node('memory.pipe', 'memref', 'memory', 'GM pipe buffer · symbolic', 'cross-core pipeline backing', { space: 'GM', offset: null }),
  node('task.plan', 'scope', 'task', 'valid-qk-plan scope', 'prepares sparse QK plan', { target: 'CORE_GROUP', form: 'scope', dependencyOrder: 1 }, 'observed', [ref(`${PASS_DIR_REL}/00_frontend.py`, 1464)]),
  node('task.qk-pv', 'scope', 'task', 'qk_pv · SPMD scope', 'SPMD scope with task dependencies', { target: 'SPMD', form: 'scope', dependencyOrder: 2 }, 'observed', [ref(`${PASS_DIR_REL}/00_frontend.py`, 1510)]),
  node('isa.load', 'isa_group', 'isa', 'TLOAD ×12', 'tile loads', { executionUnit: 'mixed', api: 'TLOAD', detailLevel: 'isa' }, 'observed', [ref(PTO_REL)]),
  node('isa.matmul', 'isa_group', 'isa', 'TMATMUL ×8 · ACC ×8', 'Cube matrix instructions', { executionUnit: 'AIC', api: 'TMATMUL', detailLevel: 'isa' }, 'observed', [ref(PTO_REL)]),
  node('isa.c2v', 'isa_group', 'isa', 'TPUSH→AIV ×2', 'Cube to Vector ISA protocol', { executionUnit: 'cross-core', api: 'TPUSH / TPOP / TFREE', detailLevel: 'isa' }, 'observed', [ref(PTO_REL)]),
  node('isa.softmax', 'isa_group', 'isa', 'TROWMAX · TEXP · TROWSUM ×2', 'Vector softmax instructions', { executionUnit: 'AIV', api: 'TROWMAX / TEXP / TROWSUM', detailLevel: 'isa' }, 'observed', [ref(PTO_REL)]),
  node('isa.v2c', 'isa_group', 'isa', 'TPUSH→AIC ×2', 'Vector to Cube ISA protocol', { executionUnit: 'cross-core', api: 'TPUSH / TPOP / TFREE', detailLevel: 'isa' }, 'observed', [ref(PTO_REL)]),
  node('isa.store', 'isa_group', 'isa', 'TSTORE ×11', 'tile stores', { executionUnit: 'mixed', api: 'TSTORE', detailLevel: 'isa' }, 'observed', [ref(PTO_REL)]),
  node('control.qk', 'control_context', 'context', '控制流与作用域', '从相邻快照提取的 CSA 控制结构；计算主干是提炼视图', { rows: [] }, 'derived'),
];

const allEdges = [
  edge('data.q-qk', 'data', 'value.q', 'op.qk', 'out', 'lhs'),
  edge('data.kv-qk', 'data', 'value.kv', 'op.qk', 'out', 'rhs'),
  edge('data.qk-scores', 'data', 'op.qk', 'value.scores'),
  edge('data.scores-rowmax', 'data', 'value.scores', 'op.rowmax'),
  edge('data.rowmax-max', 'data', 'op.rowmax', 'value.max'),
  edge('data.scores-exp', 'data', 'value.scores', 'op.exp'),
  edge('data.max-exp', 'data', 'value.max', 'op.exp'),
  edge('data.exp-prob', 'data', 'op.exp', 'value.prob'),
  edge('data.prob-rowsum', 'data', 'value.prob', 'op.rowsum'),
  edge('data.rowsum-sum', 'data', 'op.rowsum', 'value.sum'),
  edge('data.prob-pv', 'data', 'value.prob', 'op.pv'),
  edge('data.kv-pv', 'data', 'value.kv', 'op.pv'),
  edge('data.pv-out', 'data', 'op.pv', 'value.out'),
  edge('data.qk-c2v-push', 'data', 'op.qk', 'handoff.c2v-push'),
  edge('control.c2v', 'control', 'handoff.c2v-push', 'handoff.c2v-pop', 'out', 'in', { label: 'slot 0/1' }),
  edge('data.c2v-scores', 'data', 'handoff.c2v-pop', 'value.scores'),
  edge('data.prob-v2c-push', 'data', 'value.prob', 'handoff.v2c-push'),
  edge('control.v2c', 'control', 'handoff.v2c-push', 'handoff.v2c-pop', 'out', 'in', { label: 'slot 0/1' }),
  edge('data.v2c-pv', 'data', 'handoff.v2c-pop', 'op.pv'),
  edge('binding.qk-memory', 'memoryBinding', 'op.qk', 'memory.qk-pv', 'memory', 'in'),
  edge('binding.softmax-memory', 'memoryBinding', 'op.exp', 'memory.qk-pv', 'memory', 'in'),
  edge('binding.pipe', 'memoryBinding', 'handoff.c2v-push', 'memory.pipe', 'memory', 'in'),
  edge('task.qk-aic', 'lowering', 'task.qk-pv', 'kernel.aic', 'out', 'in', { label: 'kernel family' }),
  edge('task.qk-aiv', 'lowering', 'task.qk-pv', 'kernel.aiv', 'out', 'in', { label: 'kernel family' }),
  edge('task.plan-qk', 'taskDependency', 'task.plan', 'task.qk-pv', 'out', 'in', { label: '依赖 QK 计划', detail: 'deps=[qk_plan_tid, cache_ready_dep]；缓存就绪依赖未绘制为节点' }),
  edge('lower.qk-load', 'lowering', 'op.qk', 'isa.load'),
  edge('lower.qk-matmul', 'lowering', 'op.qk', 'isa.matmul'),
  edge('lower.c2v', 'lowering', 'handoff.c2v-push', 'isa.c2v'),
  edge('lower.softmax', 'lowering', 'op.exp', 'isa.softmax'),
  edge('lower.v2c', 'lowering', 'handoff.v2c-push', 'isa.v2c'),
  edge('lower.store', 'lowering', 'op.pv', 'isa.store'),
];

const initialNodeIds = [...allNodes.slice(0, 13).map((item) => item.id), 'task.plan', 'task.qk-pv', 'control.qk'];
const initialEdgeIds = [...allEdges.slice(0, 13).map((item) => item.id), 'task.plan-qk'];

// These are real control regions and gather operations, not a code card.
for (const [id, label] of [['region.iter', 'for qk_it'], ['region.guard', 'if qk_block_valid > 0'], ['region.runs', 'for qk_run · unroll(5)']]) {
  allNodes.push(node(id, 'control_scope', 'loop', label, 'CSA control region', {}, 'derived'));
  initialNodeIds.push(id);
}
for (let i = 0; i < 5; i++) {
  allNodes.push(node(`gather.${i}`, 'op', 'op', `gather_row · ${i}`, 'unrolled original-KV gather', { auxiliary: true }, 'derived'));
  allEdges.push(edge(`gather.next.${i}`, 'data', `gather.${i}`, i === 4 ? 'value.kv' : `gather.${i + 1}`, i === 4 ? 'memory' : 'out', i === 4 ? 'memory' : 'in'));
}
for (const core of ['aic', 'aiv']) {
  allNodes.push(node(`region.${core}.iter`, 'control_scope', 'loop', 'for qk_it · qk_lane_iters', 'kernel-local loop', { kernel: `kernel.${core}`, controlKind: 'loop' }, 'derived'));
  allNodes.push(node(`region.${core}.guard`, 'control_scope', 'loop', 'if 0 < qk_block_valid', 'kernel-local condition', { kernel: `kernel.${core}`, controlKind: 'condition' }, 'derived'));
}
allEdges.push(edge('gather.folded-kv', 'data', 'gather.0', 'value.kv', 'memory', 'memory'));
allEdges.push(edge('control.valid-gather', 'control', 'region.guard', 'gather.0', 'out', 'in', { label: '有效块' }));
initialNodeIds.push('gather.0');
initialEdgeIds.push('gather.folded-kv', 'control.valid-gather');

const graph = {
  schemaVersion: 'csa-pass-field.graph.v1',
  scenarioId: 'l3_decode_csa.qk_pv',
  entryPassId: 'frontend',
  nodes: allNodes,
  edges: allEdges,
  initialState: { nodeIds: initialNodeIds, edgeIds: initialEdgeIds },
  projections: {
    isa: {
      nodeIds: ['isa.load', 'isa.matmul', 'isa.c2v', 'isa.softmax', 'isa.v2c', 'isa.store'],
      edgeIds: ['lower.qk-load', 'lower.qk-matmul', 'lower.c2v', 'lower.softmax', 'lower.v2c', 'lower.store'],
      availableAfterPass: 51,
      provenance: 'post-Pass PTOAS projection; not a MaterializeValidShapeSymbols delta',
    },
  },
  notes: {
    topology: 'contains relations are expressed by scope/kernel glyph boundaries; overlays do not participate in the primary DAG',
    provenance: 'the compact compute spine is curated from qk_pv; Pass order, changed/no-op status, memory counts and ISA names are source-backed',
  },
};

const dimensions = {
  'region.iter': [0, 0, 1200, 600], 'region.guard': [0, 0, 1200, 500], 'region.runs': [0, 0, 1200, 130],
  'region.aic.iter': [0,0,1200,400], 'region.aic.guard': [0,0,1100,300],
  'region.aiv.iter': [0,0,1200,400], 'region.aiv.guard': [0,0,1100,300],
  ...Object.fromEntries(Array.from({ length: 5 }, (_, i) => [`gather.${i}`, [0, 0, 220, 48]])),
  'control.qk': [0, 0, 960, 270],
  'scope.qk-pv': [0, 0, 1200, 620],
  'value.q': [0, 0, 160, 64], 'value.kv': [0, 0, 160, 64],
  'op.qk': [0, 0, 260, 48], 'value.scores': [0, 0, 160, 64],
  'op.rowmax': [0, 0, 140, 48], 'value.max': [0, 0, 160, 64],
  'op.exp': [0, 0, 140, 48], 'value.prob': [0, 0, 180, 64],
  'op.rowsum': [0, 0, 140, 48], 'value.sum': [0, 0, 160, 64],
  'op.pv': [0, 0, 260, 48], 'value.out': [0, 0, 160, 64],
  'kernel.aic': [0, 0, 1200, 170], 'kernel.aiv': [0, 0, 1200, 240],
  'handoff.c2v-push': [0, 0, 160, 48], 'handoff.c2v-pop': [0, 0, 160, 48],
  'handoff.v2c-push': [0, 0, 160, 48], 'handoff.v2c-pop': [0, 0, 160, 48],
  'memory.qk-pv': [0, 0, 320, 56], 'memory.pipe': [0, 0, 290, 56],
  'task.plan': [0, 0, 260, 60], 'task.qk-pv': [0, 0, 260, 60],
  'isa.load': [0, 0, 260, 60], 'isa.matmul': [0, 0, 320, 60], 'isa.c2v': [0, 0, 270, 60],
  'isa.softmax': [0, 0, 420, 60], 'isa.v2c': [0, 0, 270, 60], 'isa.store': [0, 0, 260, 60],
};

const views = {
  schemaVersion: 'csa-pass-field.views.v1',
  scenarioId: 'l3_decode_csa.qk_pv',
  views: {
    front: {
      viewport: { width: 1880, height: 960, padding: 0.08 },
      nodes: Object.fromEntries(Object.entries(dimensions).map(([id, [x, y, width, height]]) => [id, { x, y, width, height, locked: false }])),
      layout: {
        engine: 'dagre@0.8.5',
        reference: 'https://github.com/kc-ml2/llvm-flow/blob/main/llvm-flow-frontend/src/components/modules/llvmcfg/BaseLayoutFlow.tsx',
        direction: 'LR',
        rankSep: 48,
        nodeSep: 52,
        edgeSep: 18,
        containerMembers: {
          'kernel.aic': ['op.qk', 'handoff.c2v-push', 'handoff.v2c-pop', 'op.pv'],
          'kernel.aiv': ['handoff.c2v-pop', 'value.scores', 'op.rowmax', 'value.max', 'op.exp', 'value.prob', 'op.rowsum', 'value.sum', 'handoff.v2c-push'],
        },
        externalSlots: {
          'value.q': 'aic',
          'value.kv': 'between',
          'value.out': 'aic',
        },
      },
      layoutPolicy: 'Dagre computes leaf-node geometry from the active DAG; scope and kernel bounds are derived from explicit JSON membership; user drag overrides remain local until reset',
    },
  },
};

const updateNode = (id, before, after) => ({ op: 'updateNode', id, before, after });
const addNode = (id) => ({ op: 'addNode', id, before: null, after: { active: true } });
const addEdge = (id) => ({ op: 'addEdge', id, before: null, after: { active: true } });
const removeEdge = (id) => ({ op: 'removeEdge', id, before: { active: true }, after: null });

const GRAPH_DELTAS = {
  OutlineIncoreScopes: [
    updateNode('task.plan', { label: 'valid-qk-plan scope', semanticKind: 'scope', attributes: { target: 'CORE_GROUP', form: 'scope', dependencyOrder: 1 } }, { label: 'valid-qk-plan task', semanticKind: 'task', attributes: { target: 'AIV', form: 'submit', dependencyOrder: 1 } }),
    updateNode('scope.qk-pv', { label: 'qk_pv · mixed InCore scope', attributes: { level: 'InCore', collapsedCount: 8 } }, { label: 'qk_pv · outlined InCore callable', attributes: { level: 'InCore', collapsedCount: 8 } }),
  ],
  OutlineClusterScopes: [
    updateNode('task.qk-pv', { label: 'qk_pv · SPMD scope', semanticKind: 'scope', attributes: { target: 'SPMD', form: 'scope', dependencyOrder: 2 } }, { label: 'qk_pv_spmd task', semanticKind: 'task', attributes: { target: 'SPMD', form: 'spmd_submit', dependencyOrder: 2 } }),
  ],
  ConvertTensorToTileOps: [
    { op: 'removeNode', id: 'control.qk', before: { active: true }, after: null },
    ...['value.q', 'value.kv', 'value.scores', 'value.max', 'value.prob', 'value.sum', 'value.out'].map((id) => updateNode(id, { semanticKind: 'tensor_value' }, { semanticKind: 'tile_value' })),
    updateNode('op.qk', { label: 'tensor.matmul · QKᵀ', attributes: { domain: 'Tensor' } }, { label: 'tile.matmul · QKᵀ', attributes: { domain: 'Tile' } }),
    updateNode('op.rowmax', { label: 'tensor.row_max', attributes: { domain: 'Tensor' } }, { label: 'tile.row_max', attributes: { domain: 'Tile' } }),
    updateNode('op.exp', { label: 'tensor.exp', attributes: { domain: 'Tensor' } }, { label: 'tile.exp', attributes: { domain: 'Tile' } }),
    updateNode('op.rowsum', { label: 'tensor.row_sum', attributes: { domain: 'Tensor' } }, { label: 'tile.row_sum', attributes: { domain: 'Tile' } }),
    updateNode('op.pv', { label: 'tensor.matmul · PV', attributes: { domain: 'Tensor' } }, { label: 'tile.matmul · PV', attributes: { domain: 'Tile' } }),
  ],
  AutoTileMatmulL0: [
    updateNode('op.qk', { label: 'tile.matmul · QKᵀ' }, { label: 'tile.matmul · QKᵀ · L0' }),
    updateNode('op.pv', { label: 'tile.matmul · PV' }, { label: 'tile.matmul · PV · L0' }),
  ],
  ExpandMixedKernel: [
    addNode('kernel.aic'), addNode('kernel.aiv'),
    addNode('handoff.c2v-push'), addNode('handoff.c2v-pop'), addNode('handoff.v2c-push'), addNode('handoff.v2c-pop'),
    removeEdge('data.qk-scores'), removeEdge('data.prob-pv'),
    addEdge('data.qk-c2v-push'), addEdge('control.c2v'), addEdge('data.c2v-scores'),
    addEdge('data.prob-v2c-push'), addEdge('control.v2c'), addEdge('data.v2c-pv'),
    addEdge('task.qk-aic'), addEdge('task.qk-aiv'),
  ],
  SkewCrossCorePipeline: [
    updateNode('handoff.c2v-push', { label: 'C2V · push' }, { label: 'C2V · push · slot 0/1' }),
    updateNode('handoff.v2c-push', { label: 'V2C · push' }, { label: 'V2C · push · slot 0/1' }),
  ],
  InitMemRef: [
    addNode('memory.qk-pv'), addNode('memory.pipe'),
    addEdge('binding.qk-memory'), addEdge('binding.softmax-memory'), addEdge('binding.pipe'),
  ],
  MemoryReuse: [
    updateNode('memory.qk-pv', { label: 'qk_pv buffers · 分空间展开', attributes: { space: 'Mat / Left / Right / Acc / Vec', buffers: 53, bytes: 944128 } }, { label: 'qk_pv buffers · 复用关系已更新', attributes: { space: 'Mat / Left / Right / Acc / Vec', buffers: 12, bytes: 361216, reused: true } }),
  ],
  AllocateMemoryAddr: [
    updateNode('memory.qk-pv', { label: 'qk_pv buffers · 复用关系已更新' }, { label: 'qk_pv buffers · 空间内地址' }),
    updateNode('memory.pipe', { label: 'GM pipe buffer · symbolic', attributes: { space: 'GM', offset: null } }, { label: 'GM pipe buffer · codegen workspace', attributes: { space: 'GM', offset: null } }),
  ],
};

const CONTRACTS = {
  OutlineIncoreScopes: ['层级作用域仍嵌在主函数中', '独立 InCore callable', '原嵌套作用域身份'],
  OutlineClusterScopes: ['SPMD 作用域及其已有显式依赖', '抽取 qk_pv_spmd 并以 spmd_submit 提交，保留依赖', '内联 SPMD 作用域'],
  ConvertTensorToTileOps: ['Tensor 域算子已规范化', 'Tile 值、显式 load/store 与 Tile 算子', 'Tensor 域局部数据流'],
  AutoTileMatmulL0: ['Tile matmul 尚未落到 L0 分块结构', 'QK/PV 的 L0 matmul 循环结构', '未分块的 matmul 形态'],
  ExpandMixedKernel: ['mixed InCore 同时包含 Cube 与 Vector 运算', 'AIC/AIV kernel family 与双向交接', '单一 mixed kernel 身份'],
  SkewCrossCorePipeline: ['跨核生产与消费已显式', 'slot 化错位流水依赖', '未错位的生产消费顺序'],
  InitMemRef: ['Tile 已有 memory space/stride', 'MemRef 与底层 buffer 身份', '纯值语义的存储未决状态'],
  MemoryReuse: ['MemRef 生命周期与空间约束可分析', '满足约束的 buffer 共享', '被合并的独立 buffer 身份'],
  AllocateMemoryAddr: ['复用后的 MemRef 集合稳定', '每个 MemRef 的地址/偏移', '符号地址未决状态'],
  AutoDeriveTaskDependencies: ['Task 已具有可推导的读写方向', '若触发则补全 Task 依赖；本场景未命中', '无'],
  MaterializeValidShapeSymbols: ['运行时边界和动态 shape 已收尾', 'valid-shape 符号；可追溯至 PTOAS 输出', '未实体化的 shape 符号'],
};

function buildPasses(snapshots) {
  let previousContext = { rows: [] };
  const model = new Map(allNodes.map((item) => [item.id, structuredClone(item)]));
  return snapshots.slice(1).map((snapshot, passIndex) => {
    const previous = snapshots[passIndex];
    const familyChanged = snapshot.familyText !== previous.familyText;
    const programChanged = snapshot.sourceText !== previous.sourceText;
    const delta = [...(GRAPH_DELTAS[snapshot.name] || [])];
    if (snapshot.index <= 10 && programChanged) {
      const contextOf = (snap) => {
        const lines = snap.sourceText.split(/\r?\n/);
        const scopeStart = lines.findIndex((line) => /name_hint="qk_pv_spmd"/.test(line));
        const start = scopeStart >= 0 ? scopeStart : lines.findIndex((line) => /def qk_pv\(/.test(line));
        const block = start >= 0 ? lines.slice(start, start + 300) : lines;
        const patterns = [/name_hint="qk_pv_spmd"|pl.spmd_submit\(self.qk_pv_spmd/, /for qk_it/, /if .*qk_block_valid/, /for qk_run.*pl.unroll|qk_run_lo.*=/, /tensor.matmul/];
        const rows = patterns.map((pattern) => {
          const index = block.findIndex((line) => pattern.test(line));
          return index < 0 ? null : { line: (start >= 0 ? start : 0) + index + 1, text: block[index].trim() };
        }).filter(Boolean);
        if (snap.index >= 9) {
          for (const pattern of [/pl.spmd_submit\(self.qk_pv_spmd/, /def qk_pv\(/, /pl.submit\(self.csa_slots_build_valid_qk_plan/, /self.qk_pv\(/]) {
            const index = lines.findIndex((line) => pattern.test(line));
            if (index >= 0) rows.push({ line: index + 1, text: lines[index].trim() });
          }
        }
        return rows.slice(0, 6);
      };
      const descriptions = {
        1: '函数内联：CSA 计算进入调用者，保留 SPMD 与条件分支',
        2: '循环展开：qk_run 的 unroll 循环变成展开语句',
        4: 'SSA 转换：变量定义与使用显式版本化',
        5: '表达式简化：保留控制结构，简化标量表达式',
        7: '调用表达式展开：嵌套调用拆为临时值',
        9: 'InCore 抽取：计算体成为独立 qk_pv callable；计划作用域变为 submit',
        10: 'Cluster 抽取：SPMD 作用域变为 qk_pv_spmd 的 spmd_submit',
      };
      const nextContext = { rows: contextOf(snapshot), beforeRows: contextOf(previous), description: descriptions[snapshot.index], source: snapshot.fileRel, beforeSource: previous.fileRel };
      delta.push(updateNode('control.qk', { attributes: previousContext }, { attributes: nextContext }));
      previousContext = nextContext;
    }
    // Apply curated patches first; derived facts preserve all other fields.
    for (const patch of delta) if (patch.op === 'updateNode') Object.assign(model.get(patch.id), structuredClone(patch.after));
    const patchNode = (id, after) => {
      const target = model.get(id);
      const before = Object.fromEntries(Object.keys(after).map((key) => [key, target[key]]).filter(([, value]) => value !== undefined));
      if (JSON.stringify(before) !== JSON.stringify(after)) {
        delta.push(updateNode(id, before, after));
        Object.assign(target, structuredClone(after));
      }
    };
    if (snapshot.index <= 10 && programChanged) {
      const facts = controlFacts(snapshot);
      patchNode('region.iter', { label: facts.ssa ? 'for qk_it · SSA iter_args (ℓ, m, O)' : 'for qk_it · qk_lane_iters', sourceFacts: { ssa: facts.ssa, inline: /inline/.test(facts.loop?.text || '') } });
      patchNode('region.guard', { label: snapshot.index >= 5 ? 'if 0 < qk_block_valid' : 'if qk_block_valid > 0', sourceFacts: { flattened: facts.flattened } });
      patchNode('region.runs', { label: facts.runCount ? `for qk_run · unroll(${facts.runCount})` : 'qk_run · 5 处条件 gather', sourceFacts: { expanded: !facts.runCount, lowerBound: facts.lowerBound } });
      facts.gathers.slice(0, 5).forEach((item, i) => patchNode(`gather.${i}`, { label: item.label, sourceFacts: { version: item.version }, sourceRefs: [item.sourceRef] }));
      if (snapshot.index === 2) {
        delta.push(removeEdge('gather.folded-kv'), ...[1, 2, 3, 4].map((i) => addNode(`gather.${i}`)), ...[0, 1, 2, 3, 4].map((i) => addEdge(`gather.next.${i}`)));
      }
    }
    if (snapshot.index === 11) {
      for (const id of ['region.runs', ...[0, 1, 2, 3, 4].map((i) => `gather.${i}`)]) delta.push({ op: 'removeNode', id, before: { active: true }, after: null });
      delta.push(removeEdge('control.valid-gather'), ...[0, 1, 2, 3, 4].map((i) => removeEdge(`gather.next.${i}`)));
    }
    if (snapshot.index === 23) {
      for (const id of ['region.iter', 'region.guard']) delta.push({ op: 'removeNode', id, before: { active: true }, after: null });
      for (const core of ['aic', 'aiv']) for (const kind of ['iter', 'guard']) delta.push({op:'addNode',id:`region.${core}.${kind}`,before:null,after:{active:true}});
    }
    if (snapshot.index >= 9 && familyChanged) {
      const rows = csaLines(snapshot);
      patchNode('scope.qk-pv', { sourceFacts: structuralFacts(rows) });
      for (const id of ['kernel.aic', 'kernel.aiv']) {
        const name = id.endsWith('aic') ? 'qk_pv_aic' : 'qk_pv_aiv';
        let active = false;
        const body = rows.filter((row) => {
          if (/^    def /.test(row.text)) active = row.text.includes(`def ${name}(`);
          return active;
        });
        if (body.length) {
          patchNode(id, { sourceFacts: structuralFacts(body) });
          const core = id.split('.')[1];
          for (const [kind, pattern] of [['iter', /for qk_it\w*.* in pl\.(range|pipeline)\(/], ['guard', /if .*qk_block_valid/]]) {
            const row = body.find(row => pattern.test(row.text));
            patchNode(`region.${core}.${kind}`, {
              attributes: {kernel:id,controlKind:kind === 'iter' ? 'loop' : 'condition',present:Boolean(row)},
              sourceRefs: row ? [{file:snapshot.fileRel,line:row.line,evidence:'observed'}] : [],
            });
          }
        }
      }
      // Persist source-observed memory-space assignments on the value glyphs.
      for (const [id, pattern] of [['value.q', /qk_q_tile.*: pl.Tile/], ['value.kv', /qk_kv.*: pl.Tile/], ['value.scores', /^\s*qk_raw\w*__tile(?:_Vec)?\s*: pl.Tile/], ['value.prob', /qk_p_bf16.*: pl.Tile/]]) {
        const spaces = [...new Set(rows.filter((row) => pattern.test(row.text)).flatMap((row) => [...row.text.matchAll(/pl.Mem\.(\w+)/g)].map((m) => m[1])))].sort();
        if (spaces.length) patchNode(id, { sourceFacts: { spaces } });
      }
    }
    const changed = programChanged;
    const contract = CONTRACTS[snapshot.name] || [
      '上一 Pass 输出满足当前 Pass 的触发条件',
      familyChanged ? 'qk_pv 源 IR 发生可观测改写' : programChanged ? '整份程序 IR 有变化；qk_pv family 保持不变' : '整份程序 IR 保持不变',
      familyChanged ? '被替换的局部 IR 形态' : programChanged ? '完整程序中被替换的 IR 形态' : '无',
    ];
    return {
      id: `pass.${String(snapshot.index).padStart(2, '0')}.${snapshot.name}`,
      index: snapshot.index,
      name: snapshot.name,
      phase: phaseForIndex(snapshot.index),
      lens: lensOf(snapshot.name),
      changed,
      noOp: !changed,
      familySourceChanged: familyChanged,
      programSourceChanged: programChanged,
      visualImpact: delta.length ? 'dag-delta' : familyChanged ? 'source-only' : programChanged ? 'program-source-only' : 'none',
      summary: delta.length
        ? contract[1]
        : familyChanged
          ? 'qk_pv family 源 IR 有变化；提炼 DAG 保持当前语义结构，详情以 source evidence 为准。'
          : programChanged
            ? '整份程序 IR 有变化；qk_pv family 与提炼 DAG 保持不变。'
            : '整份 Pass 快照前后逐行一致；该 Pass 是真实 no-op。',
      observations: { programChanged, familyChanged },
      programEvidence: !familyChanged && programChanged && snapshot.index > 10 ? {
        before: { file: previous.fileRel, text: changedExcerpt(previous.sourceText, snapshot.sourceText) },
        after: { file: snapshot.fileRel, text: changedExcerpt(snapshot.sourceText, previous.sourceText) },
      } : null,
      contract: {
        intent: !familyChanged && programChanged && snapshot.index > 10 ? '整份程序 IR 有变化；qk_pv family 保持不变' : contract[1],
        precondition: contract[0],
        produces: !changed ? '整份 IR 快照保持不变' : !familyChanged && snapshot.index > 10 ? '变化在 qk_pv 场景之外；当前图保持不变' : contract[1],
        invalidates: changed ? contract[2] : '无',
        validation: familyChanged ? '前后 qk_pv family 源码比较' : programChanged ? '前后整份 Pass 快照比较' : 'exact whole-program equality',
      },
      metrics: { before: previous.metrics, after: snapshot.metrics },
      affectedObjectIds: [...new Set(delta.flatMap((operation) => [operation.id]))],
      delta,
      sourceRefs: snapshot.sourceRefs.length ? snapshot.sourceRefs : [ref(snapshot.fileRel)],
    };
  });
}

const snapshots = readPassSnapshots();
const passRows = buildPasses(snapshots);
const instructions = extractIsa();
const passData = {
  schemaVersion: 'csa-pass-field.passes.v1',
  scenarioId: 'l3_decode_csa.qk_pv',
  phases: PHASES,
  baseline: { id: 'frontend', index: 0, sourceRefs: snapshots[0].sourceRefs },
  passes: passRows,
  summary: {
    count: passRows.length,
    sourceChangedForQkPvFamily: passRows.filter((pass) => pass.familySourceChanged).length,
    changedForProgram: passRows.filter((pass) => pass.changed).length,
    noOpForProgram: passRows.filter((pass) => pass.noOp).length,
    dagDeltaPasses: passRows.filter((pass) => pass.delta.length).length,
    comparison: 'whole-program exact snapshot comparison, with qk_pv family comparison recorded separately',
  },
};
const isa = {
  schemaVersion: 'csa-pass-field.isa.v1',
  scenarioId: 'l3_decode_csa.qk_pv',
  source: { pto: PTO_REL, cpp: CPP_REL },
  instructions,
};

const runId = '_jit_l3_decode_csa_20260903_010617';
const runRoot = `Data/DeepseekV4/${runId}`;
const rankRoot = path.join(ROOT, runRoot, 'dfx_outputs');
const ranks = fs.readdirSync(rankRoot).filter(name => /^rank\d+$/.test(name)).sort().map(rankId => {
  const file = `${runRoot}/dfx_outputs/${rankId}/d0/chip_swimlane_records.json`;
  const { metadata, aicore_tasks } = JSON.parse(fs.readFileSync(path.join(ROOT, file), 'utf8'));
  return { rankId, dispatchId: 'd0', cores: Object.fromEntries(['aic', 'aiv'].map(type => [type, metadata.core_types.filter(value => value === type).length])), recordCount: aicore_tasks.length, sourceRefs: [ref(file)] };
});
const hash = text => crypto.createHash('sha256').update(text).digest('hex');
const inputHashes = Object.fromEntries([...snapshots.map(s => s.fileRel), PTO_REL, CPP_REL, ...ranks.flatMap(r => r.sourceRefs.map(s => s.file))].map(file => [file, hash(fs.readFileSync(path.join(ROOT, file)))]));
const objectLinks = {
  schemaVersion: 'csa-pass-field.object-links.v1', runId,
  taskKey: ['runId', 'rankId', 'dispatchId', 'taskId'],
  kernelInstanceKey: ['runId', 'rankId', 'dispatchId', 'taskId', 'funcId'],
  blockKey: ['runId', 'rankId', 'dispatchId', 'taskId', 'funcId', 'coreId', 'recordId'],
  taskIdEncoding: 'string',
  links: allNodes.map(n => ({ staticObjectId: n.id, status: 'unresolved', runtimeIds: [], reason: '尚未接入 dispatch 名称表与任务记录的精确身份关联；当前仅展示编译结构', sourceRefs: n.sourceRefs })),
};

const manifest = {
  schemaVersion: 'csa-pass-field.manifest.v1',
  scenarioId: 'l3_decode_csa.qk_pv',
  title: 'CSA qk_pv mixed Cube/Vector handoff',
  generatedAt: new Date().toISOString(),
  buildIdentity: { runId, inputHashes, generatorSha256: hash(fs.readFileSync(fileURLToPath(import.meta.url))) },
  deployment: { view: 'shared-compile-structure', ranks },
  sourceRoots: { passDumps: PASS_DIR_REL, pto: PTO_REL, cpp: CPP_REL },
  passCount: passRows.length,
  artifacts: { graph: 'graph.json', passes: 'passes.json', isa: 'isa.json', views: 'views.json', objectLinks: 'object-links.json' },
  evidenceSummary: { sourceChangedForQkPvFamily: passData.summary.sourceChangedForQkPvFamily, changedForProgram: passData.summary.changedForProgram, noOpForProgram: passData.summary.noOpForProgram, isaStaticOccurrences: instructions.reduce((sum, item) => sum + item.occurrenceCount, 0) },
};

fs.mkdirSync(OUT, { recursive: true });
for (const [name, data] of Object.entries({ manifest, graph, passes: passData, isa, views, 'object-links': objectLinks })) {
  fs.writeFileSync(path.join(OUT, `${name}.json`), `${JSON.stringify(data, null, 2)}\n`, 'utf8');
}

console.log(`CSA qk_pv: ${passRows.length} Passes, ${passData.summary.sourceChangedForQkPvFamily} family-source changes, ${passData.summary.changedForProgram} program changes, ${passData.summary.noOpForProgram} no-op`);
console.log(`ISA: ${instructions.reduce((sum, item) => sum + item.occurrenceCount, 0)} selected static occurrences`);
