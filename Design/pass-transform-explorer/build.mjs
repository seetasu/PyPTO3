// Build the Pass Transform Explorer data set.
//
//   node build.mjs [--runs a,b] [--no-src]
//
// Produces:
//   data/index.js       run + pass timeline, measured deltas, evidence cards,
//                       per-Pass performance tier and that run's real perf hints
//   data/docs.js        pass prose pulled from repo/pto/docs/zh-cn/dev/passes
//   data/<run>/NN.js    one IR snapshot's raw text, loaded on demand
//   lib/bundle.js       the parser/analyzer as a classic script for the viewer
//
// The viewer re-parses snapshots in the browser with the same code that runs
// here, so diffs and graphs are computed live at full fidelity instead of being
// frozen into a precomputed blob.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseDump, resetParserErrors, parserErrors } from './lib/pyir.mjs';
import { analyzeProgram } from './lib/analyze.mjs';
import { buildEvidence, compareFunctions, metricSnapshot, rewritePatterns } from './lib/evidence.mjs';
import { diffLines, countChanges } from './lib/diff.mjs';
import { extractDoc, normalizeName, phaseOf, lensOf, PHASES } from './lib/passinfo.mjs';
import { PERF_TIERS, perfOf, parsePerfHints, groupPerfHints } from './lib/perfinfo.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '../..');

const RUNS = [
  {
    id: 'l3_decode_csa',
    title: 'DeepSeek V4 · L3 Decode CSA',
    subtitle: '多卡分布式 decode，含通信与 CSA 注意力',
    dir: 'Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/passes_dump',
  },
  {
    id: 'decode_fwd_layers',
    title: 'Decode Forward Layers',
    subtitle: '单卡 decode layer 前向，含 RMSNorm / QKV / FA / MLP',
    dir: 'Data/_jit_decode_fwd_layers_20260625_184941/passes_dump',
  },
];

const DOC_DIR = path.join(REPO, 'repo/pto/docs/zh-cn/dev/passes');
const SRC_DIR = path.join(REPO, 'repo/pto/src/ir/transforms');

const args = process.argv.slice(2);
const emitSrc = !args.includes('--no-src');
const onlyRuns = (() => {
  const i = args.indexOf('--runs');
  return i >= 0 ? new Set(args[i + 1].split(',')) : null;
})();

// ---------------------------------------------------------------------------
// Pass documentation
// ---------------------------------------------------------------------------

function loadDocs() {
  const docs = new Map();
  if (!fs.existsSync(DOC_DIR)) return docs;
  for (const file of fs.readdirSync(DOC_DIR)) {
    if (!file.endsWith('.md')) continue;
    const stem = file.replace(/\.md$/, '').replace(/^\d+-/, '');
    const md = fs.readFileSync(path.join(DOC_DIR, file), 'utf8');
    docs.set(normalizeName(stem), { ...extractDoc(md), file: 'repo/pto/docs/zh-cn/dev/passes/' + file });
  }
  return docs;
}

function loadSources() {
  const srcs = new Map();
  if (!fs.existsSync(SRC_DIR)) return srcs;
  for (const file of fs.readdirSync(SRC_DIR)) {
    if (!file.endsWith('.cpp')) continue;
    const stem = file.replace(/\.cpp$/, '').replace(/_pass$/, '');
    srcs.set(normalizeName(stem), 'repo/pto/src/ir/transforms/' + file);
  }
  return srcs;
}

/**
 * The real performance hints the compiler emitted for this run.
 *
 * This is measured output, not a model of the pipeline: a Pass that shows hints
 * here degraded something on THIS operator. A Pass with a high tier and no
 * hints made its decisions without complaint, which is a different statement
 * and the UI keeps them apart.
 */
function loadPerfHints(runDir) {
  const log = path.join(runDir, '../report/perf_hints.log');
  if (!fs.existsSync(log)) return null;
  const parsed = parsePerfHints(fs.readFileSync(log, 'utf8'));
  if (parsed.unparsed.length) {
    console.warn(`! ${path.relative(REPO, log)}: ${parsed.unparsed.length} unparsed hint lines`);
  }
  return { ...parsed, log: path.relative(REPO, log).split(path.sep).join('/') };
}

/**
 * A Pass's performance profile plus whatever it actually reported on this run.
 *
 * `tier` is a standing claim about the Pass; `hints` is measured output from
 * one compilation. Keeping both on the same object lets the UI show the gap —
 * a `decide` Pass with no hints made its calls without complaint, which is very
 * different from a `decide` Pass that shed 33 buffers.
 */
function passPerf(name, group) {
  const prof = perfOf(name);
  const out = { tier: prof.tier };
  if (prof.why) out.why = prof.why;
  if (prof.lever) out.lever = prof.lever;
  if (prof.doc) out.doc = prof.doc;
  if (group) {
    out.hints = {
      lines: group.lines,
      occurrences: group.occurrences,
      codes: group.codes,
      // One entry per distinct source site, busiest first. The compiler already
      // folded repeats at a site, so `occurrences` is its count, not ours.
      sites: group.sites
        .slice()
        .sort((a, b) => b.occurrences - a.occurrences)
        .map((s) => ({ code: s.code, at: s.at, occurrences: s.occurrences, message: s.message })),
    };
  }
  return out;
}

// A non-Pass emitter can fire at hundreds of sites (197 for the innermost-dim
// check). The full list would dominate index.js without telling the reader
// anything the first few rows don't, so it is capped and the remainder counted.
const OTHER_SITE_CAP = 24;

/** Merge sites that carry the identical message, heaviest first. */
function rollupSites(sites) {
  const by = new Map();
  for (const s of sites) {
    const key = s.code + '\u0000' + s.at + '\u0000' + s.message;
    if (!by.has(key)) by.set(key, { code: s.code, at: s.at, message: s.message, occurrences: 0 });
    by.get(key).occurrences += s.occurrences;
  }
  return [...by.values()].sort((a, b) => b.occurrences - a.occurrences || a.at.localeCompare(b.at));
}

// ---------------------------------------------------------------------------
// Diff accounting for the timeline magnitude bars
// ---------------------------------------------------------------------------

// Token-substitution sampling is capped so one enormous pass (InitMemRef
// rewrites thousands of lines) cannot dominate the build time.
const REWRITE_ROW_BUDGET = 12000;

function diffStats(prev, next, funcRows) {
  let add = 0;
  let del = 0;
  let firstChange = null;
  const sampled = [];

  for (const row of funcRows) {
    if (row.status === 'same') continue;
    if (row.status === 'added') {
      const f = next.byName.get(row.name);
      add += f.src ? f.src.length : 0;
      if (!firstChange) firstChange = { fn: row.name, line: f.decoLine, side: 'after' };
      continue;
    }
    if (row.status === 'removed') {
      const f = prev.byName.get(row.name);
      del += f.src ? f.src.length : 0;
      if (!firstChange) firstChange = { fn: row.name, line: f.decoLine, side: 'before' };
      continue;
    }
    const p = prev.byName.get(row.name);
    const n = next.byName.get(row.name);
    const rows = diffLines(p.src || [], n.src || []);
    const c = countChanges(rows);
    add += c.add;
    del += c.del;
    row.add = c.add;
    row.del = c.del;
    if (sampled.length < REWRITE_ROW_BUDGET) sampled.push(...rows);
    if (!firstChange) {
      const hit = rows.find((r) => r.tag !== '=');
      if (hit) {
        firstChange = {
          fn: row.name,
          line: hit.tag === '-' ? p.decoLine + hit.a : n.decoLine + hit.b,
          side: hit.tag === '-' ? 'before' : 'after',
        };
      }
    }
  }

  // Module-level symbol declarations live outside any function.
  const gp = prev.globals.map((g) => g.name + ' = ' + g.src);
  const gn = next.globals.map((g) => g.name + ' = ' + g.src);
  const gc = countChanges(diffLines(gp, gn));
  add += gc.add;
  del += gc.del;

  return { add, del, firstChange, rewrite: rewritePatterns(sampled) };
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

const docs = loadDocs();
const cppSources = loadSources();
const t0 = Date.now();
const runsOut = [];
const docsOut = {};

for (const run of RUNS) {
  if (onlyRuns && !onlyRuns.has(run.id)) continue;
  const dir = path.join(REPO, run.dir);
  if (!fs.existsSync(dir)) {
    console.warn(`! skipping ${run.id}: ${dir} not found`);
    continue;
  }
  const files = fs.readdirSync(dir).filter((f) => /^\d+_.*\.py$/.test(f)).sort();
  const outDir = path.join(HERE, 'data', run.id);
  fs.mkdirSync(outDir, { recursive: true });

  // Pass names come from the snapshot filenames, so the emitter split below is
  // decided by this run rather than by a hardcoded list of verifier names.
  const passNames = new Set(files.map((f) => {
    const n = f.match(/^\d+_(.*)\.py$/)[1];
    return n === 'frontend' ? 'frontend' : n.replace(/^after_/, '');
  }));
  const perf = loadPerfHints(dir);
  const hintGroups = perf ? groupPerfHints(perf.records, passNames) : [];
  const hintsByPass = new Map(hintGroups.filter((g) => g.isPass).map((g) => [g.emitter, g]));

  const passes = [];
  let prev = null;
  let prevName = null;

  for (const file of files) {
    const m = file.match(/^(\d+)_(.*)\.py$/);
    const idx = Number(m[1]);
    const name = m[2] === 'frontend' ? 'frontend' : m[2].replace(/^after_/, '');
    const text = fs.readFileSync(path.join(dir, file), 'utf8');
    const lines = text.split(/\r?\n/);

    resetParserErrors();
    const prog = parseDump(text, file);
    const an = analyzeProgram(prog, lines);
    if (parserErrors().length) {
      console.warn(`! ${run.id}/${file}: ${parserErrors().length} expression parse failures`);
    }

    const entry = {
      idx,
      name,
      file,
      title: name === 'frontend' ? '前端 IR' : name,
      phase: phaseOf(name),
      lens: lensOf(name),
      metrics: metricSnapshot(an),
      bytes: Buffer.byteLength(text),
      doc: docs.has(normalizeName(name)) ? normalizeName(name) : null,
      source: cppSources.get(normalizeName(name)) || null,
      perf: passPerf(name, hintsByPass.get(name)),
      functions: an.functions.map((f) => ({
        name: f.name,
        kind: f.kind,
        level: f.level,
        role: f.role,
        stmts: f.stmtCount,
        lines: f.srcLineCount,
        line: f.decoLine,
        loops: f.loops.length,
        allocs: f.allocs.length,
        tasks: f.tasks.length,
      })),
      topOps: Object.entries(an.totals.ops).sort((a, b) => b[1] - a[1]).slice(0, 10).map(([k, v]) => [k, v]),
    };

    if (prev) {
      const funcRows = compareFunctions(prev, an);
      const stats = diffStats(prev, an, funcRows);
      const ev = buildEvidence(prev, an, funcRows, stats.rewrite);
      entry.from = prevName;
      entry.diff = { add: stats.add, del: stats.del, firstChange: stats.firstChange };
      entry.changedFunctions = funcRows.filter((r) => r.status !== 'same');
      entry.sameFunctions = funcRows.filter((r) => r.status === 'same').length;
      entry.evidence = ev.cards;
      entry.headline = ev.headline;
      entry.changed = stats.add + stats.del > 0;
    } else {
      entry.diff = { add: 0, del: 0, firstChange: null };
      entry.changedFunctions = [];
      entry.sameFunctions = an.functions.length;
      entry.evidence = [];
      entry.headline = `前端 IR：${an.totals.functions} 个函数，${an.totals.stmts} 条语句`;
      entry.changed = false;
    }

    if (emitSrc) {
      fs.writeFileSync(
        path.join(outDir, String(idx).padStart(2, '0') + '.js'),
        'PTX.src(' + JSON.stringify(run.id) + ',' + idx + ',' + JSON.stringify(text) + ');\n',
        'utf8',
      );
    }

    passes.push(entry);
    if (entry.doc && !docsOut[entry.doc]) docsOut[entry.doc] = docs.get(entry.doc);
    prev = an;
    prevName = name;
    process.stdout.write(`\r  ${run.id}  ${file.padEnd(46)}`);
  }

  process.stdout.write('\r' + ' '.repeat(70) + '\r');
  console.log(`  ${run.id}: ${passes.length} snapshots, program=${passes[0] ? passes[0].functions.length : 0} fn -> ${passes[passes.length - 1].functions.length} fn`);

  if (perf) {
    const owned = hintGroups.filter((g) => g.isPass).reduce((n, g) => n + g.lines, 0);
    console.log(`  ${run.id}: ${perf.records.length} perf hints — ${owned} from Passes, `
      + `${perf.records.length - owned} from elsewhere (${hintGroups.map((g) => g.emitter).join(', ')})`);
  } else {
    console.log(`  ${run.id}: no perf_hints.log alongside the dump`);
  }

  runsOut.push({
    id: run.id,
    title: run.title,
    subtitle: run.subtitle,
    dir: run.dir,
    program: passes[0] ? (passes[0].programName || null) : null,
    passes,
    // Run-level view of the same log: the totals, and the emitters that are not
    // Passes at all. Without this the 197 `PH001`s would look like nobody's
    // problem just because no Pass owns them.
    perf: perf ? {
      log: perf.log,
      total: perf.records.length,
      emitters: hintGroups.map((g) => ({
        emitter: g.emitter,
        isPass: g.isPass,
        lines: g.lines,
        occurrences: g.occurrences,
        codes: g.codes,
      })),
      // Non-Pass emitters keep their sites here, since no Pass page will show them.
      other: hintGroups.filter((g) => !g.isPass).map((g) => {
        const rows = rollupSites(g.sites);
        return {
          emitter: g.emitter,
          lines: g.lines,
          occurrences: g.occurrences,
          codes: g.codes,
          sites: rows.slice(0, OTHER_SITE_CAP),
          moreSites: Math.max(0, rows.length - OTHER_SITE_CAP),
        };
      }),
    } : null,
  });
}

// ---------------------------------------------------------------------------
// Emit
// ---------------------------------------------------------------------------

fs.mkdirSync(path.join(HERE, 'data'), { recursive: true });

const index = {
  generated: new Date().toISOString(),
  phases: PHASES,
  perfTiers: PERF_TIERS,
  runs: runsOut,
};
fs.writeFileSync(
  path.join(HERE, 'data/index.js'),
  '// Generated by build.mjs - do not hand-edit.\nwindow.PTX_INDEX = ' + JSON.stringify(index) + ';\n',
  'utf8',
);

fs.writeFileSync(
  path.join(HERE, 'data/docs.js'),
  '// Generated by build.mjs - do not hand-edit.\nwindow.PTX_DOCS = ' + JSON.stringify(docsOut) + ';\n',
  'utf8',
);

// Bundle the ES modules as one classic script so the viewer also works from
// file:// (where `<script type="module" src=...>` is blocked by CORS).
const BUNDLE_ORDER = ['pyir.mjs', 'diff.mjs', 'analyze.mjs', 'evidence.mjs', 'movegraph.mjs', 'memmap.mjs', 'passinfo.mjs', 'markdown.mjs'];
const bundled = BUNDLE_ORDER.map((f) => {
  const src = fs.readFileSync(path.join(HERE, 'lib', f), 'utf8');
  return src
    .split(/\r?\n/)
    .filter((l) => !/^\s*import\s.*from\s.*;\s*$/.test(l))
    .map((l) => l.replace(/^export\s+(const|function|class|let)\s/, '$1 '))
    .join('\n');
}).join('\n\n');

const exportedNames = [];
for (const f of BUNDLE_ORDER) {
  const src = fs.readFileSync(path.join(HERE, 'lib', f), 'utf8');
  for (const m of src.matchAll(/^export\s+(?:const|function|class|let)\s+([A-Za-z0-9_$]+)/gm)) exportedNames.push(m[1]);
}

fs.writeFileSync(
  path.join(HERE, 'lib/bundle.js'),
  '// Generated by build.mjs from lib/*.mjs - do not hand-edit.\n'
  + 'window.PTXLib = (function () {\n'
  + bundled + '\n'
  + 'return {' + [...new Set(exportedNames)].join(',') + '};\n'
  + '})();\n',
  'utf8',
);

// Stamp the page's asset URLs with the build id so a rebuild is never masked
// by a cached app.js / styles.css / bundle.js.
const stamp = String(Date.now().toString(36));
const indexHtmlPath = path.join(HERE, 'index.html');
fs.writeFileSync(
  indexHtmlPath,
  fs.readFileSync(indexHtmlPath, 'utf8')
    .replace(/(href|src)="((?!\.\.\/)[^"]*?)\?v=[A-Za-z0-9]+"/g, `$1="$2?v=${stamp}"`),
  'utf8',
);

const bytes = (p) => fs.statSync(path.join(HERE, p)).size;
console.log(`\nindex.js  ${(bytes('data/index.js') / 1024).toFixed(0)} KiB`);
console.log(`docs.js   ${(bytes('data/docs.js') / 1024).toFixed(0)} KiB`);
console.log(`bundle.js ${(bytes('lib/bundle.js') / 1024).toFixed(0)} KiB`);
console.log(`done in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
