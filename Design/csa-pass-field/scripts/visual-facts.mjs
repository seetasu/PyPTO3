// Read structural facts from the selected CSA body, never from Pass names.
export function csaLines(snapshot) {
  const lines = snapshot.sourceText.split(/\r?\n/);
  if (snapshot.index >= 9) {
    let active = false;
    return lines.flatMap((text, index) => {
      if (/^    def /.test(text)) active = /^    def qk_pv(?:_|\()/.test(text);
      return active ? [{ text, line: index + 1 }] : [];
    });
  }
  const start = lines.findIndex((line) => /name_hint="qk_pv_spmd"/.test(line));
  if (start < 0) throw new Error('Missing CSA SPMD scope');
  const indent = lines[start].search(/\S/);
  let end = start + 1;
  while (end < lines.length && (!lines[end].trim() || lines[end].search(/\S/) > indent)) end++;
  return lines.slice(start, end).map((text, index) => ({ text, line: start + index + 1 }));
}

export function controlFacts(snapshot) {
  const rows = csaLines(snapshot);
  const find = (pattern) => rows.find(({ text }) => pattern.test(text));
  const gatherRows = rows.filter(({ text }) => /pl.tensor.gather_row\(.*ori_kv_flat.*qk_run_lo/.test(text));
  const compact = (text) => text.replace(/_inline\d+/g, '').replace(/pl\.const\((\d+), pl.INDEX\)/g, '$1').replace(/pl\./g, '');
  const run = find(/for qk_run.*unroll/);
  return {
    loop: find(/for qk_it/),
    guard: find(/if .*qk_block_valid/),
    gathers: gatherRows.map((row, index) => ({
      label: run ? 'gather_row · qk_run' : `gather_row · ${index}`,
      version: row.text.match(/qk_kv\w*__(ssa|phi)_v\d+/)?.[0]?.replace(/_inline\d+/g, '') || 'qk_kv',
      sourceRef: { file: snapshot.fileRel, line: row.line, evidence: 'observed' },
    })),
    runCount: run ? Number(run.text.match(/unroll\((\d+)\)/)?.[1]) : null,
    lowerBound: compact(find(/qk_run_lo.*=/)?.text.split(' = ')[1] || ''),
    ssa: Boolean(find(/for qk_it.*init_values=/)),
    flattened: Boolean(find(/__tmp_v/)),
  };
}

export function structuralFacts(rows) {
  const text = rows.map((row) => row.text).join('\n');
  const loops = rows.filter((row) => /^\s*for /.test(row.text));
  const spaces = [...new Set([...text.matchAll(/pl.Mem\.(\w+)/g)].map((m) => m[1]))].sort();
  const splits = [...new Set([...text.matchAll(/(?:split|split_num)\s*[=:]\s*(\d+)/g)].map((m) => Number(m[1])))];
  return {
    loops: loops.map((row) => ({ name: row.text.trim().match(/^for ([\w]+)/)?.[1].replace(/_inline\d+/g, ''), kind: /pl.pipeline\(/.test(row.text) ? 'pipeline' : 'range', stage: Number(row.text.match(/stage=(\d+)/)?.[1]) || null, bounds: row.text.split(/ in pl\.(?:range|pipeline)\(/)[1]?.split(/, (?:stage|init_values)=/)[0].replace(/_inline\d+/g, '') })),
    spaces,
    split: splits,
    pipelineLowered: /pipeline_membership/.test(text),
    directed: /arg_directions/.test(text),
    gmPipe: /__gm_pipe_buffer/.test(text),
    freeSplits: rows.filter((row) => /\.tfree_to_/.test(row.text)).map((row) => row.text.match(/split=(\d+)/)?.[1] ?? null),
  };
}
