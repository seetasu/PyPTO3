// View models derived exclusively from the serialized PTO source occurrences.
export function loadInstances(instruction) {
  return instruction.occurrences.map((occurrence) => {
    const input = occurrence.text.match(/ins\((%[^\s:]+)\s*:\s*!pto.partition_tensor_view<([^>]+)>/);
    const output = occurrence.text.match(/outs\((%[^\s:]+)\s*:\s*!pto.tile_buf<([^>]+)>/);
    if (!input || !output) throw new Error(`Unrecognized TLOAD at L${occurrence.line}`);
    const fields = Object.fromEntries(output[2].split(',').map((field) => field.trim().split('=')));
    const inputParts = input[2].split('x');
    const dtype = inputParts.pop().toUpperCase().replace('F32', 'FP32');
    return { ...occurrence, id: `isa.load.L${occurrence.line}`, input: input[1], output: output[1],
      inputShape: inputParts, shape: [fields.rows, fields.cols], dtype,
      space: fields.loc, validShape: [fields.v_row, fields.v_col],
      group: `${occurrence.kernel}:${fields.loc}:${fields.rows}x${fields.cols}:${dtype}` };
  });
}

export function groupLoadInstances(instances) {
  const kernels = new Map();
  for (const instance of instances) {
    if (!kernels.has(instance.kernel)) kernels.set(instance.kernel, new Map());
    const groups = kernels.get(instance.kernel);
    if (!groups.has(instance.group)) groups.set(instance.group, { ...instance, instances: [] });
    groups.get(instance.group).instances.push(instance);
  }
  return [...kernels].map(([kernel, groups]) => ({ kernel, groups: [...groups.values()] }));
}
