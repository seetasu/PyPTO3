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
 const buckets=new Map();
 for(const instruction of instructions)for(const occurrence of instruction.occurrences){
  const owner=occurrence.owner||'kernel.main',id=`instructions.${instruction.mlirName}.${owner}`;
  if(!buckets.has(id))buckets.set(id,{id,label:instruction.apiName,owner,kernel:occurrence.kernel,kernelId:'kernel.main',semanticKind:'instruction_collection',renderKind:'isa',instances:[],attributes:{detailLevel:'isa'}});
  const parts=occurrence.text.split('outs(');
  buckets.get(id).instances.push({...occurrence,id:`${id}.L${occurrence.line}`,label:instruction.apiName,collectionId:id,inputs:operandTypes(parts[0]),outputs:parts[1]?operandTypes(parts[1]):[],input:parts[0],output:parts[1]});
 }
 return [...buckets.values()];
}
