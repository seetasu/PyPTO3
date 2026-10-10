// Unit checks for the pieces that are easy to get subtly wrong.
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';
import { parseDump, parseExpression, dotted, literal, splitTop, findTop } from './pyir.mjs';
import { parseType, analyzeProgram, controlTree, dataflowGraph } from './analyze.mjs';
import { diffLines, toHunks, countChanges, wordDiff, tokenEdits } from './diff.mjs';
import { rewritePatterns } from './evidence.mjs';
import { md, mdInline, escapeHtml } from './markdown.mjs';
import { extractDoc, normalizeName } from './passinfo.mjs';
import { moveGraph } from './movegraph.mjs';
import { programMemoryMap, memoryMapStats } from './memmap.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '../../..');

let pass = 0;
let fail = 0;
function ok(name, cond, detail) {
  if (cond) { pass++; return; }
  fail++;
  console.log(`  FAIL ${name}${detail ? ' :: ' + detail : ''}`);
}
function eq(name, got, want) {
  ok(name, JSON.stringify(got) === JSON.stringify(want), `got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);
}

// ── expression parsing ────────────────────────────────────────────────
{
  const e = parseExpression('pl.tensor.matmul(a, pl.tensor.slice(w, [16, 256], [0, k]), a_trans=False, out_dtype=pl.FP32)');
  eq('call name', dotted(e.fn), 'pl.tensor.matmul');
  eq('arg count', e.args.length, 2);
  eq('kwarg a_trans', literal(e.kwargs.a_trans), false);
  eq('kwarg out_dtype', dotted(e.kwargs.out_dtype), 'pl.FP32');
  eq('nested slice shape', literal(e.args[1].args[1]), [16, 256]);
}
{
  const e = parseExpression('pl.cast(group_base__ssa_v0, pl.INDEX) + peer_tp_inline2472__idx_v0');
  eq('binop', e.k, 'bin');
  eq('binop op', e.op, '+');
}
{
  // Strings containing brackets and commas must not confuse the splitter.
  eq('splitTop respects strings', splitTop('a, "x,y", [1, 2]', ',').length, 3);
  eq('findTop skips brackets', findTop('f(a: b) : c', ':'), 8);
}

// ── type decoding ─────────────────────────────────────────────────────
{
  const t = parseType(parseExpression('pl.Tile[[16, 256], pl.BF16, pl.MemRef(mem_vec_2, pl.const(8192, pl.INT64), 16384), pl.Mem.Vec]'));
  eq('tile ctor', t.ctor, 'Tile');
  eq('tile shape', t.shape, [16, 256]);
  eq('tile dtype', t.dtype, 'BF16');
  eq('tile space', t.space, 'Vec');
  eq('tile memref', t.memref, { buffer: 'mem_vec_2', offset: 8192, size: 16384 });
}
{
  const t = parseType(parseExpression('pl.Out[pl.Tensor[[16, 5120], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 327680)]]'));
  eq('out direction', t.dir, 'out');
  eq('out ctor', t.ctor, 'Tensor');
  eq('out buffer', t.memref.buffer, 'mem_ddr_0');
}
{
  const t = parseType(parseExpression('pl.Tensor[[t_dim__ssa_v0, 16384], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]'));
  eq('dynamic dim kept symbolic', t.shape[0], 't_dim__ssa_v0');
  eq('dynamic memref size', t.memref.size, 0);
}

// ── statement structure ───────────────────────────────────────────────
{
  const src = [
    '# pypto.program: demo',
    'import pypto.language as pl',
    'T_DYN = pl.dynamic("T_DYN")',
    '@pl.program',
    'class demo:',
    '    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)',
    '    def f(self, x: pl.Tensor[[4, 4], pl.FP32]) -> pl.Tensor[[4, 4], pl.FP32]:',
    '        p: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)',
    '        for i, (acc,) in pl.pipeline(8, stage=2, init_values=(x,)):',
    '            if i > 0:',
    '                acc: pl.Tensor[[4, 4], pl.FP32] = pl.tile.add(acc, x)',
    '            else:',
    '                acc: pl.Tensor[[4, 4], pl.FP32] = pl.tile.mul(acc, x)',
    '        return x',
  ].join('\n');
  const prog = parseDump(src, 'demo');
  eq('program name', prog.name, 'demo');
  eq('globals', prog.globals.map((g) => g.name), ['T_DYN']);
  eq('function count', prog.functions.length, 1);

  const fn = prog.functions[0];
  eq('deco type', fn.deco.type, 'pl.FunctionType.AIV');
  eq('params', fn.params.map((p) => p.name), ['self', 'x']);
  eq('top-level stmts', fn.body.map((s) => s.kind), ['assign', 'for', 'return']);

  const loop = fn.body[1];
  eq('loop kind', loop.loopKind, 'pipeline');
  eq('loop trip', loop.tripCount, 8);
  eq('loop stage', loop.stage, 2);
  eq('loop carries', loop.initValues, ['x']);
  eq('if/else captured', [loop.body[0].body.length, loop.body[0].orelse.length], [1, 1]);

  const an = analyzeProgram(prog, src.split('\n'));
  const f = an.functions[0];
  eq('alloc space', f.allocs[0].space, 'Vec');
  eq('alloc size', f.allocs[0].size, 1024);
  eq('role', f.role, 'SubWorker');
  eq('stmt count', f.stmtCount, 6);

  const ct = controlTree(f);
  ok('control tree has the pipeline node', ct.nodes.some((n) => n.label === 'pipeline(8)'));
  ok('control tree has an else branch', ct.nodes.some((n) => n.type === 'else'));

  const df = dataflowGraph(f);
  ok('dataflow links x into tile.add', df.edges.some((e) => e.from === 'p:x' && e.to.startsWith('v:acc')));
}

// ── diff ──────────────────────────────────────────────────────────────
{
  const a = ['a', 'b', 'c', 'd'];
  const b = ['a', 'x', 'c', 'd'];
  const rows = diffLines(a, b);
  eq('diff counts', countChanges(rows), { add: 1, del: 1 });
  eq('diff keeps order', rows.map((r) => r.tag).join(''), '=-+==');

  const hunks = toHunks(rows, 1, 10, 20);
  eq('hunk count', hunks.length, 1);
  eq('hunk start lines', [hunks[0].aStart, hunks[0].bStart], [10, 20]);
}
{
  // Repeated near-identical lines: the unique anchor must win over a naive LCS.
  const a = ['x = load(0)', 'y = load(0)', 'z = load(0)', 'UNIQUE_TAIL'];
  const b = ['x = load(0)', 'y = load(0)', 'NEW = load(9)', 'z = load(0)', 'UNIQUE_TAIL'];
  const rows = diffLines(a, b);
  eq('anchored insert', countChanges(rows), { add: 1, del: 0 });
}
{
  const w = wordDiff('t: pl.Mem.Vec = load(a)', 't: pl.Mem.Mat = load(a)');
  ok('word diff marks only the changed token', w.left.filter((r) => r[0]).map((r) => r[1]).join('') === 'Vec');
}
{
  const e = tokenEdits('x = f(a, 0)', 'x = f(a, 4096)');
  eq('token edit', e.map((r) => [r.old, r.new]), [['0', '4096']]);
}
{
  // Unrelated deleted/added lines must not be reported as a substitution.
  const rows = diffLines(['alpha = one(1)', 'beta = two(2)'], ['gamma = three(3)', 'delta = four(4)']);
  const rw = rewritePatterns(rows);
  eq('no bogus pairing', rw.paired, 0);
  eq('counted as pure add/del', [rw.pureAdd, rw.pureDel], [2, 2]);
}
{
  const rows = diffLines(['a = f(x, 0)', 'b = f(y, 0)'], ['a = f(x, 512)', 'b = f(y, 512)']);
  const rw = rewritePatterns(rows);
  eq('substitution found', [rw.top[0].from, rw.top[0].to, rw.top[0].count], ['0', '512', 2]);
}

// ── markdown ──────────────────────────────────────────────────────────
{
  eq('escape', escapeHtml('<a & "b">'), '&lt;a &amp; &quot;b&quot;&gt;');
  ok('inline code', mdInline('use `pl.tile.load`').includes('<code>pl.tile.load</code>'));
  ok('inline bold', mdInline('**核心**').includes('<strong>核心</strong>'));
  ok('unsafe link neutralised', mdInline('[x](javascript:alert(1))').includes('href="#"'));

  // The horizontal rule used to stall the block loop forever.
  ok('horizontal rule terminates', md('para\n\n---\n\nmore').includes('<hr>'));
  ok('bare numeric line terminates', md('1.5x faster than before').includes('<p>'));
  ok('table', md('| a | b |\n| --- | --- |\n| 1 | 2 |').includes('<td>1</td>'));
  ok('fenced code', md('```cpp\nint x;\n```').includes('int x;'));
  ok('ordered list', md('1. first\n2. second').startsWith('<ol>'));
}
{
  // Every real pass doc must render, and must terminate.
  const dir = path.join(REPO, 'repo/pto/docs/zh-cn/dev/passes');
  if (fs.existsSync(dir)) {
    let rendered = 0;
    for (const file of fs.readdirSync(dir).filter((f) => f.endsWith('.md'))) {
      const raw = fs.readFileSync(path.join(dir, file), 'utf8');
      const doc = extractDoc(raw);
      const html = doc.blocks.map((b) => md(b.body)).join('');
      ok(`doc has sections: ${file}`, doc.blocks.length > 0);
      ok(`doc has a title: ${file}`, Boolean(doc.title));
      ok(`doc renders: ${file}`, typeof html === 'string');
      rendered++;
    }
    ok('rendered every pass doc', rendered > 30, `${rendered} docs`);
  }
}

// ── name mapping ──────────────────────────────────────────────────────
{
  eq('normalize FlattenTileNdTo2D', normalizeName('FlattenTileNdTo2D'), normalizeName('flatten_tile_nd_to_2d'));
  eq('normalize InitMemRef', normalizeName('InitMemRef'), normalizeName('init_memref'));
  eq('normalize SynthesizeAllReduceSignals', normalizeName('SynthesizeAllReduceSignals'), normalizeName('synthesize_allreduce_signals'));
}

// ── the browser bundle really exposes what app.js calls ───────────────
{
  const bundlePath = path.join(HERE, 'bundle.js');
  if (fs.existsSync(bundlePath)) {
    const sandbox = { window: {} };
    vm.createContext(sandbox);
    vm.runInContext(fs.readFileSync(bundlePath, 'utf8'), sandbox);
    const lib = sandbox.window.PTXLib;
    const needed = ['parseDump', 'analyzeProgram', 'diffLines', 'toHunks', 'countChanges', 'wordDiff',
      'callGraph', 'controlTree', 'dataflowGraph', 'taskGraph', 'memoryView', 'fmtBytes', 'moveGraph',
      'md', 'mdInline', 'escapeHtml'];
    for (const n of needed) ok(`bundle exports ${n}`, typeof lib[n] === 'function');
    ok('bundle has no NUL bytes', !fs.readFileSync(bundlePath, 'utf8').includes('\0'));
  } else {
    ok('bundle exists', false, 'run `node build.mjs` first');
  }
}


// ── a tail call is a call ────────────────────────────────────
//
// `return self.foo(...)` used to parse into raw strings with no `expr`, so the
// call graph silently dropped the edge. In the real dump that was one edge out
// of hundreds - and it carried 1,599 of the 1,647 statements InlineFunctions
// moves, so the migration figure attributed nothing to its biggest host.
{
  const src = [
    'class M:',
    '    @pl.function(level=pl.Level.HOST)',
    '    def outer(self, x: pl.Tensor[[4], pl.FP32]):',
    '        return self.inner(x)',
    '    @pl.function(type=pl.FunctionType.Inline)',
    '    def inner(self, x: pl.Tensor[[4], pl.FP32]):',
    '        y: pl.Tensor[[4], pl.FP32] = pl.tensor.cast(x, pl.FP32)',
    '        pl.tensor.write(x, y)',
  ].join('\n');
  const an = analyzeProgram(parseDump(src, 't'), src.split('\n'));
  const outer = an.byName.get('outer');
  ok('tail call is recorded', outer && outer.calls.length === 1,
    JSON.stringify(outer && outer.calls));
  // Read through a placeholder rather than indexing blind: when this check
  // fails it must report a FAIL, not throw and skip every test after it.
  const tc = (outer && outer.calls[0]) || {};
  eq('tail call names its callee', tc.callee, 'inner');
  eq('tail call is marked as one', tc.via, 'tail');
}

// ── where a function body went ───────────────────────────────
{
  const dumps = path.join(REPO, 'Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/passes_dump');
  if (!fs.existsSync(dumps)) {
    ok('move-graph fixtures present', true, 'skipped - dumps not checked out');
  } else {
    const load = (f) => {
      const t = fs.readFileSync(path.join(dumps, f), 'utf8');
      return analyzeProgram(parseDump(t, f), t.split(/\r?\n/));
    };

    // dissolve: every removed body must find a surviving host, and the
    // attribution must land near the measured growth rather than anywhere.
    const inl = moveGraph(load('00_frontend.py'), load('01_after_InlineFunctions.py'));
    ok('InlineFunctions reads as a dissolve', inl && inl.direction === 'dissolve');
    ok('no body is left unattributed', inl.moves.every((m) => m.host),
      inl.moves.filter((m) => !m.host).map((m) => m.body).join(','));
    const big = inl.hosts.find((h) => h.name === 'decode_csa_test');
    ok('the biggest host is credited', big && big.claimed > 1000, JSON.stringify(big && big.claimed));
    ok('attribution tracks the measured growth',
      inl.hosts.every((h) => h.measured > 0 && h.claimed / h.measured > 0.8 && h.claimed / h.measured < 1.3),
      inl.hosts.map((h) => h.name + ' ' + (h.claimed / h.measured).toFixed(2)).join(', '));
    ok('a duplicated body reports its copies',
      inl.moves.some((m) => m.copies > 1));

    // The figure's headline count must agree with the function census shown
    // beside it. `moves` has one row per (body, host) pair, so counting rows
    // there reported 21 bodies where the census says 20 removed.
    const beforeAn = load('00_frontend.py');
    const afterAn = load('01_after_InlineFunctions.py');
    const removedCount = beforeAn.functions.length - afterAn.functions.length;
    eq('body count matches the function census', inl.counts.bodies, removedCount);
    ok('pair rows really do exceed body count here', inl.moves.length > inl.counts.bodies,
      inl.moves.length + ' rows vs ' + inl.counts.bodies + ' bodies');
    eq('one row per body in the drawing list', inl.bodies.length, inl.counts.bodies);
    ok('statements are reported both ways',
      inl.counts.stmts > 0 && inl.counts.stmtsWithCopies >= inl.counts.stmts,
      inl.counts.stmts + ' / ' + inl.counts.stmtsWithCopies);
    eq('host count matches the reconciled hosts', inl.counts.hosts, inl.hosts.length);
    ok('every destination a body names has a host box',
      inl.bodies.every((b) => b.to.every((t) => inl.hosts.some((h) => h.name === t.host))));

    // extract: the same shape with the arrows reversed.
    const out = moveGraph(load('08_after_OutlineHierarchyScopes.py'), load('09_after_OutlineIncoreScopes.py'));
    ok('OutlineIncoreScopes reads as an extract', out && out.direction === 'extract');
    ok('extracted bodies are attributed', out.moves.every((m) => m.host));
    ok('the parent shrank', out.hosts.every((h) => h.after < h.before),
      out.hosts.map((h) => h.name + ' ' + h.before + '->' + h.after).join(', '));

    // wrap: nothing moved. Reporting this as an extract would claim 109
    // statements left a function that actually grew by 3.
    const wrap = moveGraph(load('09_after_OutlineIncoreScopes.py'), load('10_after_OutlineClusterScopes.py'));
    ok('OutlineClusterScopes reads as shells, not a move', wrap && wrap.direction === 'wrap',
      wrap && wrap.direction);
    ok('shells claim no moved statements', wrap.moves.length === 0 && wrap.hosts.length === 0);
    ok('every shell names what it wraps', wrap.wraps.length > 0 && wrap.wraps.every((w) => w.wrapped));

    // Every clickable target must say WHICH dump file its line number is in.
    // The two files number the same function very differently, and the diff
    // renders both gutters per row, so a bare number can match the wrong one.
    const sides = ['before', 'after'];
    ok('every body target declares a side',
      inl.bodies.every((b) => sides.includes(b.side)),
      JSON.stringify(inl.bodies.map((b) => b.side).filter((x) => !sides.includes(x))));
    ok('every host target declares a side', inl.hosts.every((h) => sides.includes(h.side)));
    ok('a dissolved body is numbered in the before file',
      inl.bodies.every((b) => b.side === 'before'));
    ok('a surviving host is numbered in the after file',
      inl.hosts.every((h) => h.side === 'after'));
    ok('an extracted body is numbered in the after file',
      out.bodies.every((b) => b.side === 'after'));

    // The collision this guards against is real in the shipped data: at step 09
    // the host's after-file line also exists in the before gutter, and only the
    // before one is rendered.
    const ext = moveGraph(load('08_after_OutlineHierarchyScopes.py'), load('09_after_OutlineIncoreScopes.py'));
    const host09 = ext.hosts[0];
    const fa09 = load('08_after_OutlineHierarchyScopes.py').byName.get(host09.name);
    ok('the wrong-gutter collision really is reachable here',
      host09.line >= fa09.decoLine && host09.line <= fa09.decoLine + fa09.src.length - 1,
      host09.name + ' after-line ' + host09.line + ' vs before range ' + fa09.decoLine + '..' + (fa09.decoLine + fa09.src.length - 1));

    // a Pass that touches no function boundary must produce no figure at all.
    const none = moveGraph(load('03_after_CtrlFlowTransform.py'), load('04_after_ConvertToSSA.py'));
    ok('no figure when no boundary moved', none === null);
  }
}
// ── address × lifetime memory map ─────────────────────────────────────
// Design/memory-inspector/Memory_V2.html ships the Python reference output for
// one AllocateMemoryAddr dump. The browser-side port must reproduce it box for
// box, or the two tools would disagree about the same snapshot.
{
  const dump = path.join(REPO, 'Design/assets/32_after_AllocateMemoryAddr.py');
  const ref = path.join(REPO, 'Design/assets/32_after_AllocateMemoryAddr.memory_map.json');
  if (fs.existsSync(dump) && fs.existsSync(ref)) {
    const src = fs.readFileSync(dump, 'utf8');
    const want = JSON.parse(fs.readFileSync(ref, 'utf8'));
    const got = programMemoryMap(analyzeProgram(parseDump(src, '32'), src.split(/\r?\n/)));
    const key = (b) => [b.name, b.space, b.base, b.offset, b.size, b.start, b.end, b.aliases.length,
      b.view, b.conflict, b.op, b.dtype, b.shape.join('x')].join('|');
    eq('memmap function set', got.functions.map((f) => f.name), want.functions.map((f) => f.name));
    let mismatch = 0;
    for (const wf of want.functions) {
      const gf = got.functions.find((f) => f.name === wf.name);
      if (!gf) continue;
      const G = new Set(gf.boxes.map(key));
      mismatch += wf.boxes.filter((b) => !G.has(key(b))).length + Math.abs(gf.boxes.length - wf.boxes.length);
      ok('memmap spaces ' + wf.name, JSON.stringify(gf.spaces) === JSON.stringify(wf.spaces));
      ok('memmap range ' + wf.name, gf.src_start === wf.src_start && gf.src_end === wf.src_end);
    }
    eq('memmap boxes match the Python reference', mismatch, 0);
    const st = memoryMapStats(got.functions);
    eq('memmap headline counts', [st.boxes, st.views, st.conflicts, st.overflow], [697, 108, 0, 0]);
  }

  // Before AllocateMemoryAddr every base sits at offset 0: the overlaps are
  // placeholders, and must not be reported as conflicts.
  const pre = 'Data/_jit_decode_fwd_layers_20260625_184941/passes_dump/31_after_MemoryReuse.py';
  if (fs.existsSync(path.join(REPO, pre))) {
    const src = fs.readFileSync(path.join(REPO, pre), 'utf8');
    const st = memoryMapStats(programMemoryMap(analyzeProgram(parseDump(src, '31'), src.split(/\r?\n/))).functions);
    ok('unplaced bases are detected', st.unplaced > 0, JSON.stringify(st));
    eq('unplaced overlaps are pending, not conflicts', st.conflicts, 0);
    ok('pending overlaps are counted', st.pending > 0);
  }
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
