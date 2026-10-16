import dump from '../data/inspector-memory.json';

export { dump };
export const bytes = n => `${Number((n / 1024).toFixed(2))} KiB`;
export const memorySpaces={
 Vec:{label:'向量',hardware:'UB',meaning:'向量计算的通用片上缓冲',color:'var(--gradient)'},
 Mat:{label:'矩阵',hardware:'L1',meaning:'矩阵计算进入 L0 前的输入缓冲',color:'var(--op)'},
 Left:{label:'左矩阵',hardware:'L0A',meaning:'Matmul 左操作数缓冲',color:'var(--input)'},
 Right:{label:'右矩阵',hardware:'L0B',meaning:'Matmul 右操作数缓冲',color:'var(--weight)'},
 Acc:{label:'累加',hardware:'L0C',meaning:'矩阵乘 FP32 累加结果缓冲',color:'var(--aggregate)'},
};
export const memorySpaceLabel=name=>memorySpaces[name]?.label||name;
export const memorySpaceDescription=name=>{const item=memorySpaces[name];return item?`${item.label} · ${item.hardware} · ${item.meaning}`:name;};
export const memorySpaceColor=name=>memorySpaces[name]?.color||'var(--line)';
export const memorySemanticColor=box=>{
 if(box.view)return 'var(--weight)';
 const op=box.op.toLowerCase();
 if(/sum|max|min|acc|reduce/.test(op))return 'var(--aggregate)';
 if(/load|extract|create|full/.test(op))return 'var(--input)';
 return 'var(--op)';
};
export const memoryBlockColor=(box,mode)=>mode==='semantic'?memorySemanticColor(box):memorySpaceColor(box.space);
export const memoryBlockTextColor=(box,mode,active)=>active||memoryBlockColor(box,mode)==='var(--op)'?'var(--text)':'var(--bg)';
export function stats(fn) {
  const overflow = box => box.offset + box.size > fn.spaces.find(s => s.space === box.space).limit;
  return { boxes: fn.boxes.length, views: fn.boxes.filter(b => b.view).length,
    conflicts: fn.boxes.filter(b => b.conflict).length, overflow: fn.boxes.filter(overflow).length };
}

// Memory_V2 coordinate mapping: one source line = 17 px, one byte = scale px.
export function mapGeometry(fn,space,box,fitUsed,scale){
 const extent=(fitUsed?space.hwm:space.limit)||1;
 return {x:box.offset*scale,width:box.size*scale,y:(box.start-fn.src_start)*17,height:Math.max(16,(box.end-box.start+1)*17-1),extent};
}
