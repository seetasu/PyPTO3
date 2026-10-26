/* CSA semantic graph: source-backed module dependencies; rendering belongs to the shared Graph. */
(function (global) {
  'use strict';
  const facts=global.PtoCsaModel.objects;
  const source=(key)=>{const [file,start,end]=facts[key].source.find(([name])=>name==='decode_csa.py')||facts[key].source[0];return {file,ranges:[{start,end}],symbols:[]};};
  const ports=()=>({inputs:{default:{}},outputs:{default:{}}});
  const base=(id,type,name,key,description)=>({id,type,name,description,source:source(key),ports:ports(),provenance:[{kind:'fact',origin:'source',fields:[],note:'decode_csa.py 及调用子函数的静态依赖。'}]});
  const op=(id,name,key,description,api=[])=>({...base(id,'operation',name,key,description),operation:{semantics:description,category:'compute',api}});
  const data=(id,name,key,description,shape,dtype)=>{const value={role:'intermediate',granularity:'tensor',entity:id,...(shape?{shape:{allocated:shape}}:{}),...(dtype?{dtype}:{})};const n=base(id,'data',name,key,description);n.data=value;n.provenance=[{kind:'fact',origin:'source',fields:Object.keys(value).map(k=>'data.'+k)}];return n;};
  const state=(id,name,key,description)=>{const value={initial:'已有请求状态或空 Cache',update:description};const n=base(id,'state',name,key,description);n.state=value;n.provenance=[{kind:'fact',origin:'source',fields:Object.keys(value).map(k=>'state.'+k)}];return n;};
  const edge=(s,t,label,type='data')=>({id:s+'>'+t,s,t,label,type,sourcePort:'default',targetPort:'default'});
  function module(id,name,key,children,links,description){
    const subgraph={schemaVersion:2,id:id+'-detail',nodes:children,edges:links.map(x=>edge(...x)),groups:[]};
    const first=children[0].id,last=children[children.length-1].id;
    return {...base(id,'composite',name,key,description),composite:{semantics:description,category:'module',api:[],subgraph,portMappings:{inputs:{default:{node:first,port:'default',relation:'view'}},outputs:{default:{node:last,port:'default',relation:'view'}}}}};
  }
  const simple=(id,name,key,description)=>module(id,name,key,[op('compute',name,key,description)],[],description);
  const nodes=[
    simple('hc-pre','HC Pre','pre','四路 HC 输入混合为 Attention 输入。'),
    simple('rmsnorm','RMSNorm','pre','归一化本地 token 行。'),
    data('local','Local normalized input','local','Q 和 Indexer query 读取本地 token 行。',['T_DYN','D'],'BF16'),
    simple('allgather','AllGather','gather','跨 Rank 收齐 token 行，形成每 Rank 的完整输入。'),
    data('full','Gathered input','full','KV 和两路 Compressor 读取完整 token 行。',['KV_T_DYN','D'],'BF16'),
    simple('q','Q Projection','q',facts.q.what),simple('kv','KV Projection','kv',facts.kv.what),
    simple('write','Original KV Writeback','write',facts.write.what),state('original','Original KV Cache','original',facts.original.what),
    simple('main','Main Compressor','main',facts.main.what),state('compressed','Compressed KV Cache','compressed',facts.compressed.what),
    simple('idxcomp','Indexer Compressor','idxcomp',facts.idxcomp.what),state('indexcache','Indexer KV Cache','indexcache',facts.indexcache.what),
    simple('indexer','Indexer / Top-k','indexer',facts.indexer.what),
    state('ready','cache_ready_dep','ready','只汇合 Original KV Writeback 与 Main Compressor 写入的 Task ID。'),
    module('attention','Sparse Attention','attention',[
      op('window','Window KV Read','attention','读取 Original KV Cache 中的最近窗口。'),
      op('selected','Compressed KV Read','attention','按 Top-k 位置读取 Compressed KV Cache。'),
      op('qk','QK','attention','Q 与两路 KV 计算注意力分数。'),
      op('softmax','Softmax','attention','归一化注意力分数。'),
      op('pv','PV','attention','用概率权重汇聚 V。')
    ],[['window','qk','recent KV'],['selected','qk','selected KV'],['qk','softmax','scores'],['softmax','pv','probability']],facts.attention.what),
    module('publish','Merge / Pack / Publish','publish',[
      op('merge','Merge','publish','合并稀疏块输出与统计量。'),
      op('pack','Pack','publish','按目标 O Group 打包输出。'),
      op('put','Put / Notify','publish','写入目标 Rank 窗口并发送就绪通知。')
    ],[['merge','pack','merged heads'],['pack','put','group payload']],facts.publish.what),
    simple('a2a','All-to-All receive','a2a','等待、读取并回收 Publish 的跨 Rank 信号；不重复发送 payload。'),
    simple('oproj','Grouped O Projection','oproj',facts.oproj.what),
    module('rs','ReduceScatter','rs',[
      op('publish','Publish partial','rs','将各 Rank 的部分投影结果发送给 token owner。'),
      op('reduce','FP32 accumulate','rs','目标 Rank 以 FP32 累加跨 Rank 部分结果。'),
      op('finish','Complete','rs','完成通信信号回收并写出本地结果。')
    ],[['publish','reduce','partial','control'],['reduce','finish','reduced output']],facts.rs.what),
    simple('post','HC Post','post',facts.post.what)
  ];
  const attention=nodes.find(n=>n.id==='attention');
  attention.ports.inputs={q:{},original:{},compressed:{},positions:{},ready:{}};
  attention.composite.portMappings.inputs={q:{node:'qk',port:'default',relation:'view'},original:{node:'window',port:'default',relation:'view'},compressed:{node:'selected',port:'default',relation:'view'},positions:{node:'selected',port:'default',relation:'view'},ready:{node:'qk',port:'default',relation:'control'}};
  const links=[
    ['hc-pre','rmsnorm','HC input'],['rmsnorm','local','normalized local'],['local','allgather','local rows'],['allgather','full','rank-major rows'],
    ['local','q','local Q'],['allgather','q','call dependency','control'],['full','kv','gathered KV'],['full','main','gathered compressor input'],['full','idxcomp','gathered index input'],
    ['kv','write','KV values'],['write','original','cache write'],['main','compressed','compressed write'],['idxcomp','indexcache','index write'],
    ['write','ready','ori_cache_write_tid','control'],['main','ready','cmp_cache_write_tid','control'],
    ['q','indexer','local qr'],['indexcache','indexer','index cache'],['idxcomp','indexer','idx_cache_write_tid','control'],
    ['q','attention','Q'],['original','attention','window KV'],['compressed','attention','selected KV'],['indexer','attention','top-k positions'],['ready','attention','cache_ready_dep','control'],
    ['attention','publish','attention output'],['publish','a2a','Put / Notify → receive','control'],['a2a','oproj','owned groups'],['oproj','rs','partial output'],['rs','post','reduced local output']
  ];
  const edges=links.map(x=>edge(...x));
  const attentionPorts={'Q':'q','window KV':'original','selected KV':'compressed','top-k positions':'positions','cache_ready_dep':'ready'};
  edges.filter(e=>e.t==='attention').forEach(e=>{e.targetPort=attentionPorts[e.label];});
  const stageDefs=[
    ['input','Input Preparation',['hc-pre','rmsnorm','local','allgather','full','q','kv'],'pre'],
    ['cache','KV & Index Preparation',['write','original','main','compressed','idxcomp','indexcache','indexer','ready'],'write'],
    ['sparse','Sparse Attention',['attention'],'attention'],
    ['redistribute','Attention Output Redistribution',['publish','a2a'],'publish'],
    ['output','Output Projection',['oproj','rs','post'],'oproj']
  ];
  const owner=Object.fromEntries(stageDefs.flatMap(([id,,members])=>members.map(member=>[member,id])));
  const stageNodes=stageDefs.map(([id,name,members,key])=>{
    const memberNodes=members.map(member=>nodes.find(n=>n.id===member));
    const inner=edges.filter(e=>owner[e.s]===id&&owner[e.t]===id);
    const externalIn=edges.filter(e=>owner[e.t]===id&&owner[e.s]!==id);
    const externalOut=edges.filter(e=>owner[e.s]===id&&owner[e.t]!==id);
    const portSet=(items)=>Object.fromEntries(items.map((e,i)=>['p'+i,{}]));
    const mapSet=(items,side)=>Object.fromEntries(items.map((e,i)=>['p'+i,{node:e[side],port:e[side==='s'?'sourcePort':'targetPort'],relation:e.type==='control'?'control':'view'}]));
    const n=base(id,'composite',name,key,'计算语义分组；不表示严格串行或单个执行 scope。');
    n.ports={inputs:portSet(externalIn),outputs:portSet(externalOut)};
    n.composite={semantics:n.description,category:'stage',api:[],subgraph:{schemaVersion:2,id:id+'-modules',nodes:memberNodes,edges:inner,groups:[]},portMappings:{inputs:mapSet(externalIn,'t'),outputs:mapSet(externalOut,'s')}};
    return n;
  });
  const crossEdges=edges.filter(e=>owner[e.s]!==owner[e.t]).map(e=>{
    const from=stageNodes.find(n=>n.id===owner[e.s]),to=stageNodes.find(n=>n.id===owner[e.t]);
    const sourceEntries=Object.entries(from.composite.portMappings.outputs),targetEntries=Object.entries(to.composite.portMappings.inputs);
    const sourcePort=sourceEntries.find(([,m])=>m.node===e.s&&m.port===e.sourcePort)?.[0];
    const targetPort=targetEntries.find(([,m])=>m.node===e.t&&m.port===e.targetPort)?.[0];
    return {...e,id:'stage:'+e.id,s:from.id,t:to.id,sourcePort,targetPort};
  });
  global.PTO_CSA_SEMANTIC_GRAPH={schemaVersion:2,id:'csa-compute',sourceFiles:{'decode_csa.py':{lineCount:global.PTO_CSA_SOURCE.text.split('\n').length}},constants:{},runtimeDims:['T_DYN','KV_T_DYN'],nodes:stageNodes,edges:crossEdges,groups:[]};
})(window);
