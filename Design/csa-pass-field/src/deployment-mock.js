// Static deployment design projection, independent of compiler Pass progression.
export const deploymentStages = [
  ['gather', 'Token all-gather', '跨 rank token 汇集', 285],
  ['prepare', 'Q / KV 准备', 'CSA 局部计算与缓存准备', 290],
  ['qkpv', 'qk_pv', 'AIC · QK / PV  ↔  AIV · Softmax', 390],
  ['publish', 'Merge / Pack / Publish', '分块归并 · 按目标 rank 写入', 397],
  ['receive', 'O-group 接收', 'attention_window / attention_signal', 479],
  ['output', 'O-proj · Reduce-scatter', '输出投影与分片归并', 489],
  ['post', 'HC post', '本 rank 输出', 498],
];
// Source-level calls and representative operations, not a complete lowered DAG.
const contents = {
  gather: [['allgather', 'Token all-gather', true]],
  prepare: [['q', 'q_proj_rope'], ['kv', 'kv_proj_rope'], ['cache', 'Cache writeback']],
  qkpv: [['qkpv', 'qk_pv']],
  publish: [['merge', 'Merge / Pack'], ['put', 'tensor.put', true]],
  receive: [['a2a', 'o_group_a2a', true]],
  output: [['proj', 'O-proj'], ['scatter', 'Reduce-scatter', true]],
  post: [['post', 'hc_post']],
};
const endpoint = {gather:'allgather',publish:'put',receive:'a2a',output:'scatter'};
export function deploymentMock(_flow, ranks, selectedRank = null) {
  const nodes = [], edges = [];
  ranks.forEach((rank, index) => {
    const x = index * 1000;
    nodes.push({id:`${rank.rankId}::container`,type:'rankFrame',position:{x,y:0},data:{rank,selected:selectedRank===rank.rankId},style:{width:740,height:1430,zIndex:-1},draggable:false,selectable:false});
    deploymentStages.forEach(([id,label,detail,line], i) => {
      const groupId = `${rank.rankId}::${id}`;
      nodes.push({id:groupId,type:'scope',position:{x:x+70,y:180+i*175},style:{width:600,height:135},draggable:false,
        data:{rankId:rank.rankId,stageId:id,label,detail,line,attributes:{hideScopeLabel:true}}});
      contents[id].forEach(([key,name,communication],j)=> {
        nodes.push({id:`${groupId}.${key}`,type:'deploymentGlyph',parentNode:groupId,extent:'parent',position:{x:20+j*190,y:65},style:{width:180,height:44},draggable:false,
          data:{rankId:rank.rankId,stageId:id,label:name,detail,line,communication:!!communication}});
      });
      if(i) {
        const prev=deploymentStages[i-1][0];
        edges.push({id:`${rank.rankId}::local-${i}`,source:`${rank.rankId}::${prev}.${contents[prev].at(-1)[0]}`,target:`${groupId}.${contents[id][0][0]}`,sourceHandle:'out',targetHandle:'in',type:'smoothstep',markerEnd:{type:'arrowclosed',color:'#999'},style:{stroke:'#999',strokeWidth:2}});
      }
    });
  });
  if(ranks.length===2) {
    const [a,b]=ranks.map(r=>r.rankId);
    for(const [from,to,label] of [['gather','gather','token 分片汇集'],['publish','receive','attention payload · put'],['output','output','输出分片归并']]) {
      for(const [sourceRank,targetRank,reverse] of [[a,b,false],[b,a,true]]) edges.push({
        id:`${sourceRank}-${targetRank}-${from}`,source:`${sourceRank}::${from}.${endpoint[from]}`,target:`${targetRank}::${to}.${endpoint[to]}`,
        sourceHandle:reverse?'left-out':'right-out',targetHandle:reverse?'right-in':'left-in',type:'smoothstep',
        label:reverse?'':label,labelStyle:{fill:'#b8d9d4',fontSize:15},labelBgStyle:{fill:'#151515'},
        markerEnd:{type:'arrowclosed',color:'#76b5ac'},style:{stroke:'#76b5ac',strokeWidth:2,strokeDasharray:'7 5'},
      });
    }
  }
  return {nodes,edges};
}
