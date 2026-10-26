/* CSA compute graph. Each child range is checked against the local implementation. */
(function (global) {
  'use strict';
  const facts=global.PtoCsaModel.objects;
  const root='decode_csa.py';
  const ranges={
    'decode_csa.py':2079,'hc_pre.py':467,'hc_post.py':239,'rmsnorm.py':233,
    'decode_cp_token_allgather.py':331,'qkv_proj_rope.py':1176,
    'decode_compressor_ratio4.py':608,'decode_indexer_compressor.py':689,
    'decode_indexer.py':1336,'decode_sparse_attn_csa.py':819,'decode_o_proj.py':775
  };
  const src=(file,start,end)=>({file,ranges:[{start,end}],symbols:[]});
  const factSrc=key=>{const [file,start,end]=facts[key].source.find(([f])=>f===root)||facts[key].source[0];return src(file,start,end);};
  const P=(inputs=['default'],outputs=['default'])=>({inputs:Object.fromEntries(inputs.map(x=>[x,{}])),outputs:Object.fromEntries(outputs.map(x=>[x,{}]))});
  const base=(id,type,name,description,source,ports=P())=>({id,type,name,description,source,ports,provenance:[{kind:'fact',origin:'source',fields:[]}]});
  const op=(id,name,description,source,ins=['default'],outs=['default'])=>({...base(id,'operation',name,description,source,P(ins,outs)),operation:{semantics:description,category:'compute',api:[]}});
  const data=(id,name,description,source,shape,dtype,options={})=>{
    const n=base(id,'data',name,description,source);
    n.data={role:options.role||'intermediate',granularity:'tensor',entity:id,...(shape?{shape:{allocated:shape}}:{}),...(dtype?{dtype}:{}),...(options.symbol?{symbol:options.symbol}:{}),...(options.storage?{storage:{space:options.storage}}:{})};
    if(options.producer)n.data.producer=options.producer;
    if(options.consumers)n.data.consumers=options.consumers;
    const facts=['shape','dtype','symbol'].filter(k=>k in n.data).map(k=>'data.'+k);
    const resolved=['role','granularity','entity','storage'].filter(k=>k in n.data).map(k=>'data.'+k);
    n.provenance=[...(facts.length?[{kind:'fact',origin:`${source.file}:${source.ranges[0].start}–${source.ranges[0].end}`,fields:facts}]:[]),{kind:'resolved',origin:'CSA 静态数据流',fields:resolved,note:'节点角色、存储归类及 Producer / Consumer 由源码调用和读写关系整理。'}];
    return n;
  };
  const state=(id,name,description,source,options={})=>{
    const n=base(id,'state',name,description,source);
    n.state={initial:options.initial||'跨 Decode Step 保留的已有内容',update:description,...(options.dtype?{dtype:options.dtype}:{}),...(options.shape?{shape:{allocated:options.shape}}:{}),...(options.storage?{storage:{space:options.storage}}:{}),...(options.feedback?{feedback:options.feedback}:{}),granularity:options.granularity||'request'};
    const facts=['dtype','shape'].filter(k=>k in n.state).map(k=>'state.'+k);
    const resolved=Object.keys(n.state).filter(k=>!['dtype','shape'].includes(k)).map(k=>'state.'+k);
    n.provenance=[...(facts.length?[{kind:'fact',origin:`${source.file}:${source.ranges[0].start}–${source.ranges[0].end}`,fields:facts}]:[]),{kind:'resolved',origin:'CSA Cache / State 生命周期',fields:resolved,note:'当前调用的写入与跨 Decode Step 留存分别表达。'}];
    return n;
  };
  const E=(s,t,label,type='data',sp='default',tp='default')=>({id:`${s}>${t}:${label}`,s,t,label,type,sourcePort:sp,targetPort:tp});
  const m=(node,port='default',relation='view')=>({node,port,relation});
  function mod(id,name,description,source,children,links,inputs,outputs){
    const n=base(id,'composite',name,description,source,P(Object.keys(inputs),Object.keys(outputs)));
    n.composite={semantics:description,category:'module',api:[],subgraph:{schemaVersion:2,id:id+'-detail',nodes:children,edges:links.map(x=>E(...x)),groups:[]},portMappings:{inputs,outputs}};
    return n;
  }
  const S=(a,b,c)=>src(a,b,c);
  const hc=mod('hc-pre','HC Pre','RMS 分支和线性分支在门控处汇合，混合得到 Attention 输入；Sinkhorn 权重供 HC Post 使用。',factSrc('pre'),[
    data('input','HC input','本地 HC 输入与后续残差。',S('hc_pre.py',49,49),['T_DYN','HC_MULT','D'],'FP32',{symbol:'x',storage:'GM',role:'input',consumers:['rms','linear','mix']}),
    op('rms','HC RMS', '归一化 HC 输入。',S('hc_pre.py',74,92)),
    op('linear','HC Linear','计算 HC 线性分支。',S('hc_pre.py',97,123)),
    op('gate','Pre/Post gate','从 RMS 与线性结果计算门控。',S('hc_pre.py',132,155)),
    op('sinkhorn','Sinkhorn combination','计算 HC 路由与组合权重。',S('hc_pre.py',159,255)),
    op('mix','Input mix','根据门控和组合权重形成 Attention 输入。',S('hc_pre.py',258,285))
  ],[['input','rms','HC input'],['input','linear','HC input'],['input','mix','HC input'],['rms','gate','RMS branch'],['linear','gate','linear branch'],['gate','mix','pre gate'],['rms','sinkhorn','RMS branch'],['linear','sinkhorn','linear branch']],{input:m('input')},{mixed:m('mix'),post:m('gate'),comb:m('sinkhorn')});
  const rms=mod('rmsnorm','RMSNorm','本地 token 行的平方和、逆 RMS 与权重缩放。',factSrc('pre'),[
    op('sum','Row square sum','按行归约平方和。',S('rmsnorm.py',34,48)),
    op('inv','Inverse RMS','计算归一化系数。',S('rmsnorm.py',49,55)),
    op('scale','Scale / cast','乘权重并输出 BF16 行。',S('rmsnorm.py',56,60))
  ],[['sum','inv','row sum'],['inv','scale','inverse RMS']],{input:m('sum')},{normalized:m('scale')});
  const gather=mod('allgather','AllGather','Put / Notify、等待、读回和 Complete 形成完整 Rank 输入。',factSrc('gather'),[
    op('put','Put / Notify','将本地 token 行放入各 Rank 接收窗并通知。',S('decode_cp_token_allgather.py',94,118)),
    op('wait','Wait payload','等待远端 payload 通知。',S('decode_cp_token_allgather.py',121,131)),
    op('readback','Readback / Notify','读回各 Rank token 行并通知完成。',S('decode_cp_token_allgather.py',136,153)),
    op('complete','Complete','等待读回确认并回收信号。',S('decode_cp_token_allgather.py',155,182))
  ],[['put','wait','remote payload','control'],['wait','readback','payload ready','control'],['readback','complete','readback notified','control']],{local:m('put')},{full:m('readback'),done:m('complete','default','control')});
  const q=mod('q','Q Projection','本地输入生成 Q 和 Indexer query；RoPE 使用本地位置。',factSrc('q'),[
    op('seed','Q low-rank seed','计算 Q 低秩投影。',S('qkv_proj_rope.py',271,300)),
    op('norm','Q RMS / quant','归一化并量化 Q 中间结果。',S('qkv_proj_rope.py',309,368)),
    op('matmul','Q projection','投影至 Q heads。',S('qkv_proj_rope.py',375,420)),
    op('rope','Q dequant / RoPE','反量化、head 归一化并应用 RoPE。',S('qkv_proj_rope.py',421,553))
  ],[['seed','norm','low-rank Q'],['norm','matmul','quantized Q'],['matmul','rope','Q heads']],{local:m('seed'),position:m('rope')},{q:m('rope'),qr:m('norm'),qr_scale:m('norm')});
  const kv=mod('kv','KV Projection','从 AllGather 完整输入投影 KV，归一化并应用 RoPE。',factSrc('kv'),[
    op('project','KV projection','计算 KV 低秩投影。',S('qkv_proj_rope.py',577,608)),
    op('norm-rope','KV norm / RoPE','归一化 KV 并应用位置旋转。',S('qkv_proj_rope.py',616,760))
  ],[['project','norm-rope','KV projection']],{full:m('project'),position:m('norm-rope')},{kv:m('norm-rope')});
  const write=mod('write','Original KV Writeback','将原始 KV 写入 Original Cache 并返回写入 Task ID。',factSrc('write'),[
    op('address','Cache slot','按请求和位置定位 Original Cache 槽位。',S(root,321,327)),
    op('commit','KV cache write','写入原始 KV，产生 ori_cache_write_tid。',S(root,328,334))
  ],[['address','commit','slot']],{kv:m('commit'),slot:m('address')},{cache:m('commit'),tid:m('commit','default','control')});
  const main=mod('main','Main Compressor','投影、池化和状态提交分支；归一化、RoPE 后写入 Compressed Cache。',factSrc('main'),[
    op('projection','Compressor projection','压缩输入投影。',S('decode_compressor_ratio4.py',110,136)),
    op('pool','Temporal pool','与已有状态合并并池化。',S('decode_compressor_ratio4.py',145,215)),
    op('commit','State commit','提交池化状态与长度。',S('decode_compressor_ratio4.py',221,233)),
    op('norm','Compressed norm','归一化压缩 KV。',S('decode_compressor_ratio4.py',235,256)),
    op('rope','Compressed RoPE','对压缩 K 应用 RoPE。',S('decode_compressor_ratio4.py',258,277)),
    op('write','Compressed write','写入 Main Compressed KV Cache。',S('decode_compressor_ratio4.py',279,287))
  ],[['projection','pool','projected KV'],['projection','commit','projection state'],['pool','commit','pooled state','control'],['pool','norm','pooled KV'],['norm','rope','normalized KV'],['rope','write','compressed KV']],{full:m('projection'),state:m('pool'),slot:m('write')},{cache:m('write'),tid:m('write','default','control'),state:m('commit')});
  const idxcomp=mod('idxcomp','Indexer Compressor','独立的 Indexer KV 压缩、状态提交和 Cache 写入。',factSrc('idxcomp'),[
    op('projection','Indexer projection','Indexer KV 投影。',S('decode_indexer_compressor.py',115,142)),
    op('pool','Indexer pool','与已有状态合并并池化。',S('decode_indexer_compressor.py',151,225)),
    op('commit','Indexer state commit','提交 Indexer 压缩状态。',S('decode_indexer_compressor.py',231,244)),
    op('norm-rope','Indexer norm / RoPE','归一化并旋转 Indexer K。',S('decode_indexer_compressor.py',246,294)),
    op('rotate','Indexer layout rotate','转换缓存布局。',S('decode_indexer_compressor.py',296,304)),
    op('write','Indexer cache write','写入独立 Indexer KV Cache 与 Scale。',S('decode_indexer_compressor.py',306,335),['default'],['cache','scale','tid'])
  ],[['projection','pool','projected KV'],['projection','commit','projection state'],['pool','commit','pooled state','control'],['pool','norm-rope','pooled KV'],['norm-rope','rotate','rotated K'],['rotate','write','index KV']],{full:m('projection'),state:m('pool'),slot:m('write')},{cache:m('write','cache'),scale:m('write','scale'),tid:m('write','tid','control'),state:m('commit')});
  const topk=mod('topk','Hierarchical Top-k','逐叶得分、组内归约和 query 归并，选出候选位置。',S('decode_indexer.py',429,622),[
    op('score','Cache score / weighting','读取 Indexer Cache 与 Scale，计算 query 分数并应用权重。',S('decode_indexer.py',480,597),['query','weight','cache','scale','ready']),
    op('leaf','Leaf Top-k','对已计算分数执行叶级 Top-k。',S('decode_indexer.py',176,224)),
    op('group','Group wave Top-k','合并各叶候选。',S('decode_indexer.py',228,312)),
    op('query-merge','Query merge Top-k','跨组归并 query 候选。',S('decode_indexer.py',316,426))
  ],[['score','leaf','weighted scores'],['leaf','group','leaf candidates'],['group','query-merge','group candidates']],{query:m('score','query'),weight:m('score','weight'),cache:m('score','cache'),scale:m('score','scale'),ready:m('score','ready','control')},{positions:m('query-merge')});
  const indexer=mod('indexer','Indexer / Top-k','本地 query 与权重分支汇合后，读取 Indexer Cache 并分层筛选。',factSrc('indexer'),[
    op('q-proj','Indexer Q projection','从本地 qr 与 qr_scale 生成 Indexer query。',S('decode_indexer.py',651,681),['qr','scale']),
    op('q-rope','Indexer Q RoPE','对 query 应用 RoPE。',S('decode_indexer.py',682,721)),
    op('q-rotate','Indexer Q rotate','变换 query 布局。',S('decode_indexer.py',722,759)),
    op('weight','Weight projection','由本地输入生成 Indexer 权重。',S('decode_indexer.py',761,797)),
    topk
  ],[['q-proj','q-rope','query'],['q-rope','q-rotate','rotated query'],['q-rotate','topk','quantized query','data','default','query'],['weight','topk','weights','data','default','weight']],{qr:m('q-proj','qr'),qr_scale:m('q-proj','scale'),local:m('weight'),cache:m('topk','cache'),scale:m('topk','scale'),cache_tid:m('topk','ready','control')},{positions:m('topk','positions')});
  const attention=mod('attention','Sparse Attention','准备候选，分别读取 Original 窗口和 Compressed KV；QK、Softmax、PV 得到 partial result。',factSrc('attention'),[
    op('plan','Candidate index preparation','整理 Top-k 候选和窗口索引。',S('decode_sparse_attn_csa.py',143,194),['positions']),
    op('window','Window KV Read','从 Original Cache 读取最近窗口 KV。',S('decode_sparse_attn_csa.py',226,265),['cache','plan']),
    op('selected','Compressed KV Read','从 Compressed Cache 读取候选 KV。',S('decode_sparse_attn_csa.py',266,281),['cache','plan']),
    op('qk','QK','Q 与两路 KV 计算分数并应用 mask。',S('decode_sparse_attn_csa.py',280,292),['q','window','selected','plan','ready']),
    op('softmax','Softmax','归一化注意力分数。',S('decode_sparse_attn_csa.py',291,297)),
    op('pv','PV / partial','加权聚合 V 并写出 partial result。',S('decode_sparse_attn_csa.py',298,313),['default'],['mi','li','oi','tid']),
    op('rope','Inverse RoPE metadata','生成后续合并使用的 RoPE 元数据。',S('decode_sparse_attn_csa.py',315,345))
  ],[['plan','window','window positions','data','default','plan'],['plan','selected','selected positions','data','default','plan'],['window','qk','window KV','data','default','window'],['selected','qk','compressed KV','data','default','selected'],['plan','qk','attention mask','data','default','plan'],['qk','softmax','scores'],['softmax','pv','probabilities']],{q:m('qk','q'),original:m('window','cache'),compressed:m('selected','cache'),positions:m('plan','positions'),ready:m('qk','ready','control')},{mi:m('pv','mi'),li:m('pv','li'),oi:m('pv','oi'),rope:m('rope'),qk_tid:m('pv','tid','control'),rope_tid:m('rope','default','control')});
  const publish=mod('publish','Merge / Pack / Publish','同一 Publish 任务中合并 partial、处理 Sink 与 inverse RoPE、按目标 Rank 打包并 Put / Notify。',factSrc('publish'),[
    op('merge','Merge statistics','合并稀疏块结果与统计量。',S(root,411,427),['mi','li','oi','tid']),
    op('sink','Sink contribution','加入 Sink 分支。',S(root,429,433)),
    op('inverse-rope','Inverse RoPE','对合并输出执行 inverse RoPE。',S(root,435,441),['context','rope']),
    op('cast','Output cast','转换合并结果精度。',S(root,442,444)),
    op('pack','Pack groups','按目标 O Group 打包。',S(root,445,457)),
    op('put','Put / Notify','向目标 Rank Put，并发送就绪通知。',S(root,458,477))
  ],[['merge','sink','merged partial'],['sink','inverse-rope','sink adjusted','data','default','context'],['inverse-rope','cast','restored result'],['cast','pack','cast heads'],['pack','put','group payload']],{mi:m('merge','mi'),li:m('merge','li'),oi:m('merge','oi'),rope:m('inverse-rope','rope'),qk_tid:m('merge','tid','control'),rope_tid:m('inverse-rope','rope','control')},{groups:m('pack'),notified:m('put','default','control')});
  const a2a=mod('a2a','All-to-All receive','等待 Publish 通知、读回目标 Group，并回收通信信号；不重复 Put。',factSrc('a2a'),[
    op('wait','Wait notification','等待目标 Group 已发布。',S('decode_o_proj.py',269,285)),
    op('readback','Group readback','读取本 Rank 拥有的 Group 数据。',S('decode_o_proj.py',287,300)),
    op('complete','Signal complete','回收完成信号。',S('decode_o_proj.py',301,335))
  ],[['wait','readback','publish ready','control'],['readback','complete','read complete','control']],{groups:m('readback'),notified:m('wait','default','control')},{owned:m('readback'),done:m('complete','default','control')});
  const rs=mod('rs','ReduceScatter','同一 o_proj_reduce_scatter 调用中的跨 Rank partial 发布、等待、FP32 累加与完成。',factSrc('rs'),[
    op('publish','Publish partial','发布部分 O 投影结果。',S('decode_o_proj.py',606,640)),
    op('wait','Wait partials','等待其他 Rank 的部分结果。',S('decode_o_proj.py',642,647)),
    op('reduce','FP32 accumulate','将跨 Rank 部分结果累加到目标 token。',S('decode_o_proj.py',648,663)),
    op('finish','Complete / write','回收信号并写出本地结果。',S('decode_o_proj.py',664,697))
  ],[['publish','wait','partials sent','control'],['wait','reduce','remote partials','control'],['reduce','finish','reduced local output']],{partial:m('publish')},{reduced:m('finish')});
  const oproj=mod('oproj','Grouped O Projection / ReduceScatter','单次 o_proj_reduce_scatter 调用：分组 O 投影后发布 partial，并跨 Rank 归约。',factSrc('oproj'),[
    op('view','Group view','整理接收的 O Group。',S('decode_o_proj.py',474,503)),
    op('low-rank','Low-rank O','计算低秩中间结果。',S('decode_o_proj.py',505,522)),
    op('quant','O quantization','量化 O 投影输入。',S('decode_o_proj.py',524,556)),
    op('projection','Grouped O matmul','按 O Group 投影。',S('decode_o_proj.py',558,574)),
    op('dequant','O dequantization','反量化部分投影结果。',S('decode_o_proj.py',576,609)),rs
  ],[['view','low-rank','owned groups'],['low-rank','quant','low-rank result'],['quant','projection','quantized input'],['projection','dequant','projected partial'],['dequant','rs','partial O','data','default','partial']],{owned:m('view'),receive_done:m('view','default','control')},{reduced:m('rs','reduced')});
  const post=mod('post','HC Post','把归约后的 Attention 输出与 HC 残差、组合分支汇合。',factSrc('post'),[
    op('weight','Post weight','计算 HC post 权重贡献。',S('hc_post.py',56,59)),
    op('accumulate','Residual / combination','累加残差与组合分支。',S('hc_post.py',60,66),['reduced','comb','post','residual']),
    op('store','Output store','写出 HC Post 结果。',S('hc_post.py',68,69))
  ],[['weight','accumulate','post contribution','data','default','post'],['accumulate','store','final result']],{reduced:m('accumulate','reduced'),post:m('weight'),comb:m('accumulate','comb'),residual:m('accumulate','residual')},{output:m('store')});
  const nodes=[
    data('hc-input','HC input / residual','本地 HC 输入，既供 HC Pre 也供 HC Post 残差分支。',S(root,153,153),['T_DYN','HC_MULT','D'],'FP32',{symbol:'x_hc',storage:'GM',role:'input',consumers:['hc-pre']}),
    hc,rms,
    data('local','x_normed_t','本地归一化 token 行；Q 和 Indexer query 读取本地行。',S(root,273,275),['T_DYN','D'],'BF16',{symbol:'x_normed_t',storage:'GM',producer:'rmsnorm',consumers:['allgather','q']}),
    gather,
    data('full','x_normed_full','AllGather 后的完整 Rank token 行；KV 与两路 Compressor 读取。',S(root,279,289),['KV_T_DYN','D'],'BF16',{symbol:'x_normed_full',storage:'GM',producer:'allgather',consumers:['kv']}),
    q,
    data('q-tensor','Q heads','本地 Q heads，供 Sparse Attention 使用。',S(root,221,221),['T_DYN','H','HEAD_DIM'],'BF16',{symbol:'q',storage:'GM',producer:'q'}),
    data('qr-tensor','Indexer qr','本地 Q 低秩量化结果，供 Indexer query 投影。',S(root,292,292),['T_DYN','Q_LORA'],'INT8',{symbol:'qr',storage:'GM',producer:'q'}),
    data('qr-scale','Indexer qr_scale','qr 对应的反量化 scale。',S(root,293,293),['T_DYN',1],'FP32',{symbol:'qr_scale',storage:'GM',producer:'q'}),
    kv,
    data('kv-full','kv_full','完整输入投影的 KV，供 Original KV Writeback。',S(root,291,291),['KV_T_DYN','HEAD_DIM'],'BF16',{symbol:'kv_full',storage:'GM',producer:'kv'}),
    write,
    state('original','Original KV Cache','当前调用写入 Original KV；Sparse Attention 读取窗口，内容跨 Decode Step 保留。',S(root,186,186),{dtype:'BF16',shape:['ORI_BLOCK_NUM_DYN','BLOCK_SIZE',1,'HEAD_DIM'],storage:'GM',feedback:'跨 Decode Step 保留历史 KV，当前调用写入新槽位。'}),
    main,
    state('main-state','Main compressor state','当前调用提交池化状态，下次 Decode Step 继续读取。',S(root,174,175),{dtype:'FP32',shape:['MAIN_STATE_BLOCK_NUM_DYN','MAIN_STATE_BLOCK_SIZE','MAIN_STATE_DIM'],storage:'GM',feedback:'下一 Decode Step 的 Main Compressor 读取更新后的状态。'}),
    state('compressed','Compressed KV Cache','当前调用写入压缩 KV；Sparse Attention 读取候选，内容跨 Decode Step 保留。',S(root,187,187),{dtype:'BF16',shape:['CMP_BLOCK_NUM_DYN','BLOCK_SIZE',1,'HEAD_DIM'],storage:'GM',feedback:'跨 Decode Step 保留历史压缩 KV，当前调用写入新槽位。'}),
    idxcomp,
    state('idx-state','Indexer compressor state','当前调用提交 Indexer 压缩状态，下次 Decode Step 继续读取。',S(root,184,185),{dtype:'FP32',shape:['INNER_STATE_BLOCK_NUM_DYN','INNER_STATE_BLOCK_SIZE','INNER_STATE_DIM'],storage:'GM',feedback:'下一 Decode Step 的 Indexer Compressor 读取更新后的状态。'}),
    state('indexcache','Indexer KV Cache','当前调用写入 Indexer INT8 KV，Indexer 读取候选，内容跨 Decode Step 保留。',S(root,189,189),{dtype:'INT8',shape:['IDX_CACHE_BLOCK_NUM_DYN','BLOCK_SIZE',1,'IDX_HEAD_DIM'],storage:'GM',feedback:'跨 Decode Step 保留历史 Indexer KV。'}),
    state('indexscale','Indexer KV Scale Cache','与 Indexer INT8 KV 配套的 FP32 Scale，当前调用写入并跨 Step 保留。',S(root,190,190),{dtype:'FP32',shape:['IDX_CACHE_BLOCK_NUM_DYN','BLOCK_SIZE',1,1],storage:'GM',feedback:'与 Indexer KV 同步保留和读取。'}),
    indexer,
    data('topk-positions','Top-k positions','Indexer 归并后的压缩候选位置，供 Sparse Attention 规划。',S(root,223,223),['T_DYN','IDX_TOPK'],'INT32',{symbol:'idx_topk',storage:'GM',producer:'indexer'}),
    state('ready','cache_ready_dep','当前调用仅汇合 Original KV Writeback 与 Main Compressor 写入 Task ID。',S(root,361,361),{initial:'当前调用尚未汇合写入 Task ID',granularity:'task'}),
    attention,
    data('partial-mi','Attention partial max','稀疏块 softmax 最大值统计。',S('decode_sparse_attn_csa.py',204,204),['T_DYN*(H//H_TILE)*SPARSE_BLOCKS*H_TILE',1],'FP32',{symbol:'sparse_blk_mi',storage:'GM',producer:'attention'}),
    data('partial-li','Attention partial sum','稀疏块 softmax 归一化和。',S('decode_sparse_attn_csa.py',205,205),['T_DYN*(H//H_TILE)*SPARSE_BLOCKS*H_TILE',1],'FP32',{symbol:'sparse_blk_li',storage:'GM',producer:'attention'}),
    data('partial-oi','Attention partial output','稀疏块 PV 的未合并 FP32 输出。',S('decode_sparse_attn_csa.py',206,206),['T_DYN*(H//H_TILE)*SPARSE_BLOCKS*H_TILE','HEAD_DIM'],'FP32',{symbol:'sparse_blk_oi',storage:'GM',producer:'attention'}),
    publish,
    data('groups','Packed O Groups','按目标 O Group 打包的本地发送缓冲区；Publish 已执行 Put / Notify。',S(root,395,395),['O_GROUPS*LOCAL_T_PAD','O_GROUP_IN'],'BF16',{symbol:'attention_grouped',storage:'GM',producer:'publish',consumers:['a2a']}),
    a2a,
    data('owned-groups','Owned O Groups','目标 Rank 完成 Wait / Readback 后拥有的 O Groups。',S(root,295,295),['ATTENTION_WINDOW_ROWS','O_GROUP_IN'],'BF16',{symbol:'attention_local_flat',storage:'GM',producer:'a2a'}),
    oproj,
    data('reduced','Reduced attention output','O Projection 的跨 Rank partial 已累加到 token owner。',S(root,296,296),['T_DYN','D'],'BF16',{symbol:'attn_out',storage:'GM',producer:'oproj',consumers:['post']}),
    post,
    data('final-output','Final HC output','HC Post 写出的本地输出。',S(root,206,206),['T_DYN','HC_MULT','D'],'FP32',{symbol:'x_out',storage:'GM',role:'output',producer:'post'})
  ];
  const L=[
    ['hc-input','hc-pre','HC input','data','default','input'],['hc-input','post','HC residual','data','default','residual'],
    ['hc-pre','rmsnorm','HC mixed input','data','mixed','input'],['rmsnorm','local','x_normed_t','data','normalized'],
    ['local','allgather','local rows','data','default','local'],['allgather','full','x_normed_full','data','full'],
    ['local','q','local Q','data','default','local'],['full','kv','gathered KV','data','default','full'],['full','main','gathered compressor input','data','default','full'],['full','idxcomp','gathered index input','data','default','full'],
    ['q','q-tensor','Q heads','data','q'],['q','qr-tensor','qr','data','qr'],['q','qr-scale','qr_scale','data','qr_scale'],
    ['kv','kv-full','KV values','data','kv'],['kv-full','write','Original KV write input','data','default','kv'],
    ['write','original','Original cache current write','state','cache'],['main','compressed','Compressed cache current write','state','cache'],['idxcomp','indexcache','Indexer KV current write','state','cache'],['idxcomp','indexscale','Indexer Scale current write','state','scale'],
    ['main','main-state','Main state current commit','state','state'],['main-state','main','Main state next Decode Step','feedback','default','state'],
    ['idxcomp','idx-state','Indexer state current commit','state','state'],['idx-state','idxcomp','Indexer state next Decode Step','feedback','default','state'],
    ['write','ready','ori_cache_write_tid','control','tid'],['main','ready','cmp_cache_write_tid','control','tid'],
    ['qr-tensor','indexer','local qr','data','default','qr'],['qr-scale','indexer','qr_scale','data','default','qr_scale'],['local','indexer','local weights','data','default','local'],
    ['indexcache','indexer','Indexer KV read','state','default','cache'],['indexscale','indexer','Indexer Scale read','state','default','scale'],['idxcomp','indexer','idx_cache_write_tid','control','tid','cache_tid'],
    ['indexer','topk-positions','Top-k positions','data','positions'],
    ['q-tensor','attention','Q read','data','default','q'],['original','attention','Original window read','state','default','original'],['compressed','attention','Compressed candidate read','state','default','compressed'],['topk-positions','attention','Top-k positions','data','default','positions'],['ready','attention','cache_ready_dep','control','default','ready'],
    ['attention','partial-mi','partial mi','data','mi'],['attention','partial-li','partial li','data','li'],['attention','partial-oi','partial oi','data','oi'],
    ['partial-mi','publish','partial max read','data','default','mi'],['partial-li','publish','partial sum read','data','default','li'],['partial-oi','publish','partial output read','data','default','oi'],
    ['attention','publish','RoPE metadata','data','rope','rope'],['attention','publish','qk_tid','control','qk_tid','qk_tid'],['attention','publish','attn_rope_tid','control','rope_tid','rope_tid'],
    ['publish','groups','Packed O Groups','data','groups'],['groups','a2a','跨 Rank Put → Readback','data','default','groups'],['publish','a2a','跨 Rank Notify → Wait','control','notified','notified'],
    ['a2a','owned-groups','Owned O Groups','data','owned'],['owned-groups','oproj','O Group input','data','default','owned'],['a2a','oproj','receive complete','control','done','receive_done'],
    ['oproj','reduced','ReduceScatter result','data','reduced'],['reduced','post','reduced local output','data','default','reduced'],
    ['hc-pre','post','HC post weight','data','post','post'],['hc-pre','post','HC combination','data','comb','comb'],['post','final-output','x_out','data','output']
  ];
  const edges=L.map(x=>E(...x));
  edges.filter(e=>e.s==='groups'&&e.t==='a2a').forEach(e=>{e.communication='cross-rank Put / Readback';});
  edges.filter(e=>e.s==='publish'&&e.t==='a2a').forEach(e=>{e.communication='cross-rank Notify / Wait';});
  const stages=[
    ['input','Input Preparation',['hc-input','hc-pre','rmsnorm','local','allgather','full','q','q-tensor','qr-tensor','qr-scale','kv','kv-full'],'pre'],
    ['cache','KV & Index Preparation',['write','original','main','main-state','compressed','idxcomp','idx-state','indexcache','indexscale','indexer','topk-positions','ready'],'write'],
    ['sparse','Sparse Attention',['attention','partial-mi','partial-li','partial-oi'],'attention'],
    ['redistribute','Attention Output Redistribution',['publish','groups','a2a','owned-groups'],'publish'],
    ['output','Output Projection',['oproj','reduced','post','final-output'],'oproj']
  ];
  const owner=Object.fromEntries(stages.flatMap(([id,,members])=>members.map(member=>[member,id])));
  const stageNodes=stages.map(([id,name,members,key])=>{
    const inner=edges.filter(e=>owner[e.s]===id&&owner[e.t]===id), inbound=edges.filter(e=>owner[e.t]===id&&owner[e.s]!==id),outbound=edges.filter(e=>owner[e.s]===id&&owner[e.t]!==id);
    const pin=(e,side)=>`${side}:${e.id}`;
    const ins=Object.fromEntries(inbound.map(e=>[pin(e,'t'),m(e.t,e.targetPort,e.type==='control'?'control':'view')]));
    const outs=Object.fromEntries(outbound.map(e=>[pin(e,'s'),m(e.s,e.sourcePort,e.type==='control'?'control':'view')]));
    const n=mod(id,name,'计算语义分组；不表示严格串行或单个执行 scope。',factSrc(key),members.map(member=>nodes.find(n=>n.id===member)),[],ins,outs);
    n.w=260;n.h=76;
    n.composite.category='stage';n.composite.subgraph={schemaVersion:2,id:id+'-modules',nodes:members.map(member=>nodes.find(n=>n.id===member)),edges:inner,groups:[]};
    return n;
  });
  const cross=edges.filter(e=>owner[e.s]!==owner[e.t]).map(e=>({...e,id:'stage:'+e.id,s:owner[e.s],t:owner[e.t],sourcePort:`s:${e.id}`,targetPort:`t:${e.id}`}));
  global.PTO_CSA_SEMANTIC_GRAPH={schemaVersion:2,id:'csa-compute',sourceFiles:Object.fromEntries(Object.entries(ranges).map(([file,lineCount])=>[file,{lineCount}])),constants:{},runtimeDims:['T_DYN','KV_T_DYN'],nodes:stageNodes,edges:cross,groups:[]};
  global.PTO_SEMANTIC_COMPUTATION_GRAPH.validate(global.PTO_CSA_SEMANTIC_GRAPH);
})(window);
