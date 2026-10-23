// Source intervals preserve nesting and branch ownership; never infer control from opcode.
export const cleanControl = text => text.replace(/_inline\d+__(?:idx|iter|ssa|tile)_?v?\d*/g, '').replace(/__(?:idx|iter|ssa)_v\d+/g, '');
export function controlScopes(tree, parent = null, depth = 0, path = 'root') {
 const result = [];
 for (const [index,item] of (tree || []).entries()) {
  const id = 'control.' + path + '.' + index;
  const end = item.elseLine ? item.elseLine - 1 : item.endLine;
  result.push({...item, id, parent, depth, start: item.line, end});
  result.push(...controlScopes(item.body, id, depth + 1, path + '.' + index));
  if (item.elseLine) {
   const branch = id + '.else';
   result.push({id: branch, parent, depth, kind: 'else', start: item.elseLine, end: item.endLine, line: item.elseLine});
   result.push(...controlScopes(item.otherwise, branch, depth + 1, path + '.' + index + '.else'));
  }
 }
 return result;
}
export function scopeForLine(scopes, line) {
 return scopes.filter(s => line > s.start && line <= s.end || s.kind === 'else' && line === s.start).sort((a,b) => b.depth-a.depth)[0]?.id || null;
}
export function layoutControl(scopes, buckets, widths, vertical) {
 const positions = new Map(), frames = [];
 const columns = vertical ? 3 : 6, cell = Math.max(230, ...widths) + 28;
 function place(parent, x, y) {
  const items = [
   ...scopes.filter(s => s.parent === parent).map(scope => ({line:scope.line, scope})),
   ...buckets.filter(b => b.scope === parent).map(bucket => ({line:Math.min(...bucket.statements.map(s=>s.line)),bucket}))
  ].sort((a,b)=>a.line-b.line);
  let cursor=y, width=0, pending=[];
  function flush(){
   pending.forEach((b,i)=>positions.set(b.id,{x:x+i%columns*cell,y:cursor+Math.floor(i/columns)*76}));
   if(pending.length){width=Math.max(width,Math.min(columns,pending.length)*cell-28);cursor+=Math.ceil(pending.length/columns)*76;pending=[];}
  }
  for(const item of items){
   if(item.bucket){pending.push(item.bucket);continue;}
   flush();
   const child=place(item.scope.id,x+24,cursor+64);
   const w=Math.max(380,child.width+48),h=Math.max(124,child.height+80);
   frames.push({...item.scope,position:{x,y:cursor},width:w,height:h});
   width=Math.max(width,w);cursor+=h+28;
  }
  flush();
  return {width,height:cursor-y};
 }
 const bounds=place(null,160,350);
 // Sibling scopes share the same right edge; nesting contributes a fixed inset.
 const width=Math.max(bounds.width,vertical?840:1800);
 for(const frame of frames)frame.width=width-frame.depth*48;
 return {positions,frames,bottom:350+bounds.height,width};
}
