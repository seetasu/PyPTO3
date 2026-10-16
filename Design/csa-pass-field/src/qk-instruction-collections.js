// Source occurrences, not runtime execution counts. Never attribute a whole
// program aggregate to one representative OP.
function operandTypes(text) {
  return [...text.matchAll(/!pto\.(tile_buf|partition_tensor_view)<([^>]+)>/g)].map(match => {
    if (match[1] === 'partition_tensor_view') {
      const parts = match[2].split('x');
      return { label: 'Tensor view', dtype: parts.pop().toUpperCase().replace('F32', 'FP32'), shape: parts };
    }
    const fields = Object.fromEntries(match[2].split(',').map(field => field.trim().split('=')));
    return { label: `${fields.loc} Tile`, dtype: fields.dtype?.toUpperCase().replace('F32', 'FP32'), shape: [fields.rows, fields.cols] };
  });
}
export function instructionCollections(instructions) {
  const buckets = new Map();
  const locations = new Map();
  const location = text => text.match(/loc\((.*)\)$/)?.[1];
  for (const instruction of instructions.filter(item => item.mlirName.startsWith('pto.tmatmul'))) for (const occurrence of instruction.occurrences) {
    const output = occurrence.text.match(/outs\((%[^\s:]+)/)?.[1] || '';
    const owner = output.includes('qk_raw') ? 'op.qk' : output.includes('qk_oi') ? 'op.pv' : null;
    if (owner && location(occurrence.text)) locations.set(location(occurrence.text), owner);
  }
  for (const instruction of instructions) {
    for (const occurrence of instruction.occurrences) {
      const kernelId = occurrence.kernel.endsWith('_aic') ? 'kernel.aic' : 'kernel.aiv';
      const fixed = { 'pto.texp': 'op.exp', 'pto.trowmax': 'op.rowmax', 'pto.trowsum': 'op.rowsum', 'pto.tpush_to_aiv': 'handoff.c2v-push', 'pto.tpush_to_aic': 'handoff.v2c-push', 'pto.tpop_from_aiv': 'handoff.v2c-pop', 'pto.tpop_from_aic': 'handoff.c2v-pop' };
      let owner = fixed[instruction.mlirName] || kernelId;
      if (instruction.mlirName.startsWith('pto.tmatmul')) {
        const output = occurrence.text.match(/outs\((%[^\s:]+)/)?.[1] || '';
        if (output.includes('qk_raw')) owner = 'op.qk';
        else if (output.includes('qk_oi')) owner = 'op.pv';
        else owner = locations.get(location(occurrence.text)) || kernelId;
      }
      const id = `instructions.${instruction.mlirName}.${owner}`;
      const label = instruction.mlirName === 'pto.tmatmul.acc' ? 'TMATMUL.ACC' : instruction.apiName || instruction.mlirName.replace('pto.', '').toUpperCase().replace('_TO_', '→').replace('_FROM_', '←');
      if (!buckets.has(id)) buckets.set(id, { id, label, owner, kernel: occurrence.kernel, kernelId, semanticKind: 'instruction_collection', renderKind: 'isa', instances: [], attributes: { detailLevel: 'isa' } });
      const inputSection = occurrence.text.match(/ins\(([\s\S]*?)\)\s*outs/ )?.[1] || occurrence.text.split(' loc(')[0];
      const outputSection = occurrence.text.match(/outs\(([\s\S]*?)\)\s*loc/)?.[1] || occurrence.text.split(' -> ')[1] || '';
      buckets.get(id).instances.push({ ...occurrence, id: `${id}.L${occurrence.line}`, label, collectionId: id,
        inputs: occurrence.text.includes(' -> ') ? [] : operandTypes(inputSection), outputs: operandTypes(outputSection),
        input: occurrence.text.match(/ins\(([^:]+)\s*:/)?.[1]?.trim() || occurrence.text.match(/pto\.\w+\((%[^:]+)\s*:/)?.[1]?.trim(),
        output: occurrence.text.match(/outs\(([^:]+)\s*:/)?.[1]?.trim() || occurrence.text.match(/^\s*(%\S+)\s*=/)?.[1] });
    }
  }
  return [...buckets.values()];
}
