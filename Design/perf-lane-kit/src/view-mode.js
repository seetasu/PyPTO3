export const FEATURES=['tail','parallel','imbalance','underfill','fragmented','setup','bubble'];
export function parseViewParams(search){
 const q=new URLSearchParams(search),view=q.get('view')||'combined',feature=q.get('feature')||'none';
 if(!['combined','critical','lane'].includes(view))return {error:`未知视图模式：${view}`};
 if(!['none',...FEATURES].includes(feature))return {error:`未知 Task 特征：${feature}`};
 if(view==='critical'&&feature!=='none')return {error:'关键路径独立视图无法同时显示 Task 特征图层'};
 return {view,feature,analysis:q.get('analysis')==='1'};
}
