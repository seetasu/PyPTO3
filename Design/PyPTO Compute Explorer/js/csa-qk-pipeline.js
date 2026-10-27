/* qk_pv Pipeline projection. Scenario and qk_order belong to Assignment. */
(function(global){
  'use strict';
  const work=global.PtoCsaQkWork,subSource='decode_sparse_attn_csa.py',mainSource='decode_csa.py';
  const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const evidence={
    mask:{title:'valid_block_mask',fact:'每个 (Query, Sparse Block) 读取有效性；有效分支执行 Gather 与矩阵计算，无效分支写中性 Partial。',file:subSource,range:[223,226]},
    query:{title:'Query',fact:'同一 Query 的 64 Heads；Head Batch 0 读取 Heads 0–31，Batch 1 读取 Heads 32–63。',file:subSource,range:[278,283]},
    gather:{title:'KV Gather',fact:'SB0 从 Original KV 的 Page 连续区间 Gather；SB1–4 依据压缩索引逐行 Gather。无效槽位用有限值填充并由 sparse_bias 屏蔽。',file:subSource,range:[225,276]},
    kv:{title:'L1 KV Tile',fact:'一个 BF16 [128,512] KV Tile 供两个 Head Batch 共用；QK 使用 Kᵀ，PV 使用 V。',file:subSource,range:[225,296]},
    qk:{title:'QK Matmul',fact:'Q [32,512] × Kᵀ [512,128] → Scores FP32 [32,128]。32 个 Head 分别对应 128 个 KV 位置。',file:subSource,range:[280,284]},
    stats:{title:'Scale / Bias / Softmax Statistics',fact:'Scores → Scale → Bias → Row Max(m [32,1]) → Exp(exp [32,128]) → Row Sum(l [32,1])。当前仅是单个 Sparse Block 的局部统计。',file:subSource,range:[284,294]},
    pv:{title:'PV Matmul',fact:'exp 转为 BF16 [32,128] 后与 V [128,512] 相乘，产生 Partial O FP32 [32,512]；m、l 不进入 PV。',file:subSource,range:[294,296]},
    write:{title:'Partial Write',fact:'每批 32 Heads 的 m [32,1]、l [32,1]、o [32,512] 分成两个 H_TILE=16 的存储分段；两批共四段。',file:subSource,range:[297,304]},
    neutral:{title:'Neutral Partial Write',fact:'Invalid 分支跳过 Gather、QK、Softmax、PV，逐 Head 写 mi=-3.0e38、li=0、oi=0。',file:subSource,range:[305,313]},
    partial:{title:'Partial Results',fact:'该 Work Item 写出当前 Sparse Block 的 m/l/o。两批共四个 16 Head 分段；同一 Query 的其他 Sparse Blocks 也写各自 Partial。',file:subSource,range:[204,206]},
    merge:{title:'csa_merge_pack_publish',fact:'TP=2 主流程中，该后续 Task 依赖 qk_tid 与 attn_rope_tid；合并同一 Query 多个 Sparse Blocks 的 Partial m/l/o，随后 Normalize、Attention Sink、Inverse RoPE、Pack、Publish。它不在 qk_pv SPMD 循环内。',file:mainSource,range:[395,477]}
  };
  function create(host,options){
    let focus=null,context=null,scenario=null,lane=0,level='lane',selection=null,scale=1;
    const moduleId=options.moduleId;
    const scenarioId=()=>scenario?.items[0]?.scenarioId;
    const detail=item=>work.getWorkItemDetails(item.id,scenario.queue,scenario.blockCount);
    function itemFocus(item){const d=detail(item);return {kind:'work-item',rank:context.rank,scenarioId:scenarioId(),moduleId,operationId:moduleId+'/dispatch',workItemId:item.id,requestIndex:item.requestIndex,tokenIndex:item.token,positionId:item.positionId,sparseBlockIndex:item.sparseBlock,valid:item.valid,validKvCount:item.validKvCount,queuePosition:d.queuePosition,laneId:d.block,laneIteration:d.iteration};}
    const laneFocus=index=>({kind:'lane',rank:context.rank,scenarioId:scenarioId(),moduleId,operationId:moduleId+'/dispatch',blockId:index,laneId:index});
    const currentItem=()=>focus?.kind==='work-item'&&focus.scenarioId===scenarioId()&&focus.rank===context.rank?scenario.byId.get(focus.workItemId):null;
    function pickItem(id){const item=scenario.byId.get(id);if(!item)return;focus=itemFocus(item);lane=focus.laneId;level='item';selection=null;options.onFocus(focus);render();}
    function pickLane(index){if(index<0||index>=scenario.blockCount)return;lane=index;level='lane';selection=null;focus=laneFocus(index);options.onFocus(focus);render();}
    function select(batch,operation){selection={workItemId:focus.workItemId,batchIndex:batch,operation};render();}
    function inspector(){if(!selection)return '';const info=evidence[selection.operation]||evidence.query,item=currentItem(),batch=selection.batchIndex;
      return '<aside class="csa-qp-inspector" aria-label="计算详情"><header><b>'+esc(info.title)+'</b><button class="btn btn-ghost btn-sm" data-qp-close aria-label="关闭详情">×</button></header>'+(batch!=null?'<span class="csa-qp-inspector-scope">Work Item '+item.id+' / Head Batch '+batch+' · Heads '+(batch*32)+'–'+(batch*32+31)+'</span>':'<span class="csa-qp-inspector-scope">Work Item '+item.id+'</span>')+'<p>'+esc(info.fact)+'</p><button class="btn btn-ghost btn-sm" data-qp-source="'+info.file+':'+info.range.join(':')+'">'+esc(info.file)+' : '+info.range.join('–')+' ↗</button></aside>';}
    function laneView(){const block=scenario.blocks[lane],cards=block.entries.map(entry=>{const item=scenario.byId.get(entry.id);return '<button class="csa-qp-work '+(item.valid?'is-valid':'is-invalid')+'" data-qp-item="'+item.id+'" title="qk_order['+entry.queuePosition+'] · Lane '+lane+' / iteration '+entry.iteration+'"><span>it '+entry.iteration+' · qk_order['+entry.queuePosition+']</span><strong>Work Item '+item.id+'</strong><small>Q'+item.token+' / SB'+item.sparseBlock+'</small><em>'+(item.valid?'Valid':'Invalid')+'</em></button>';}).join('');return '<div class="csa-qp-lane-canvas"><div class="csa-qp-lane-list">'+scenario.blocks.map(b=>'<button class="btn btn-sm '+(b.index===lane?'is-selected':'')+'" data-qp-lane="'+b.index+'" aria-pressed="'+(b.index===lane)+'">Lane '+String(b.index).padStart(2,'0')+' <small>'+b.entries.length+'</small></button>').join('')+'</div><div class="csa-qp-lane-sequence"><div class="csa-qp-sequence-title"><b>Lane '+lane+' · Work Item Sequence</b><span>qk_order['+lane+'] → ['+(lane+scenario.blockCount)+'] → ['+(lane+2*scenario.blockCount)+'] … · 步长 '+scenario.blockCount+'</span></div><div class="csa-qp-work-list">'+cards+'</div><div class="csa-qp-sequence-foot">'+block.validCount+' Valid · '+block.invalidCount+' Invalid · 逻辑遍历顺序</div></div></div>';}
    const path=(d,kind='flow')=>'<path class="csa-qp-'+kind+'" d="'+d+'" marker-end="url(#csa-qp-arrow)"/>';
    const label=(x,y,text,cls='label')=>'<text class="csa-qp-'+cls+'" x="'+x+'" y="'+y+'">'+esc(text)+'</text>';
    function node(op,batch,x,y,w,h,title,sub,extra=''){const active=selection?.operation===op&&selection?.batchIndex===batch,dim=selection?.batchIndex!=null&&batch!=null&&selection.batchIndex!==batch;return '<g class="csa-qp-node'+(active?' is-selected':'')+(dim?' is-dimmed':'')+' '+extra+'" data-qp-operation="'+op+'"'+(batch!=null?' data-qp-batch="'+batch+'"':'')+' role="button" tabindex="0" aria-label="'+esc((batch!=null?'Batch '+batch+' ':'')+title)+'"><rect x="'+x+'" y="'+y+'" width="'+w+'" height="'+h+'" rx="8"/>'+label(x+12,y+23,title,'title')+label(x+12,y+44,sub,'sub')+'</g>';}
    function matrix(x,y,w,h,rows,cols,kind){return '<g class="csa-qp-matrix '+kind+'"><rect x="'+x+'" y="'+y+'" width="'+w+'" height="'+h+'" rx="2"/>'+Array.from({length:3},(_,i)=>'<path d="M'+(x+(i+1)*w/4)+' '+y+'v'+h+' M'+x+' '+(y+(i+1)*h/4)+'h'+w+'"/>').join('')+label(x+w/2,y+h+16,rows+' × '+cols,'matrix-caption')+'</g>';}
    function batchRow(i){const y=240+i*245,dim=selection?.batchIndex!=null&&selection.batchIndex!==i;return '<g class="csa-qp-batch'+(dim?' is-dimmed':'')+'" data-qp-batch-row="'+i+'">'+
      '<g class="csa-qp-batch-header" data-qp-pick-batch="'+i+'" role="button" tabindex="0"><rect x="24" y="'+(y-29)+'" width="165" height="26" rx="5"/>'+label(34,y-11,'Batch '+i+' · Heads '+(i*32)+'–'+(i*32+31),'batch-title')+'</g>'+
      node('query',i,24,y,140,64,'Q Tile','[32,512]')+
      node('qk',i,232,y,195,121,'QK Matmul','FP32 Scores')+matrix(244,y+55,34,32,32,512,'q')+label(287,y+78,'×','symbol')+matrix(305,y+55,34,32,512,128,'k')+label(348,y+78,'→','symbol')+matrix(375,y+55,34,32,32,128,'score')+
      node('stats',i,467,y,251,121,'Softmax Statistics','')+label(480,y+65,'Scores → Scale → Bias','step')+label(480,y+85,'Row Max → Exp → Row Sum','step')+label(480,y+106,'m [32,1] · exp [32,128] · l [32,1]','step')+
      node('pv',i,758,y,195,121,'PV Matmul','FP32 Partial O')+matrix(770,y+55,34,32,32,128,'exp')+label(814,y+78,'×','symbol')+matrix(830,y+55,34,32,128,512,'v')+label(873,y+78,'→','symbol')+matrix(901,y+55,34,32,32,512,'output')+
      node('write',i,1001,y,168,121,'Partial Write','m / l / o → H_TILE 16')+label(1014,y+82,'Heads '+(i*32)+'–'+(i*32+15),'segment')+label(1014,y+103,'Heads '+(i*32+16)+'–'+(i*32+31),'segment')+
      path('M164 '+(y+32)+' H232')+path('M427 '+(y+60)+' H467')+path('M718 '+(y+60)+' H758')+path('M953 '+(y+60)+' H1001')+
      path('M718 '+(y+110)+' V'+(y+145)+' H980 V'+(y+112)+' H1001','stat-flow')+
      path('M1169 '+(y+61)+' H'+(1192+i*18)+' V'+(786+i*12)+' H1170','output-flow')+
      label(744,y+134,'m / l → write','edge-note')+label(1178,y+144,'2 segments','edge-note')+'</g>';}
    function validGraph(item){return node('query',null,24,24,240,76,'Query Q'+item.token,'64 Heads × 512')+node('gather',null,386,24,245,76,item.sparseBlock===0?'Original KV / Window':'Compressed KV / Indices','SB'+item.sparseBlock+' · '+item.validKvCount+'/128 valid rows')+node('kv',null,690,24,240,76,'L1 KV Tile','BF16 [128,512]')+
      label(24,164,'2 Head Batch Iterations · QK_M_TILE = 32','section')+label(950,145,'KV Tile 复用','section')+
      path('M144 100 V211 H94 V240','input-flow')+path('M144 100 V456 H94 V485','input-flow')+path('M631 62 H690','input-flow')+
      path('M814 100 V182 H328 V240','reuse-flow')+path('M880 100 V194 H851 V240','reuse-flow')+path('M930 62 H1225 V427 H328 V485','reuse-flow')+path('M930 82 H1233 V439 H851 V485','reuse-flow')+
      label(350,204,'Kᵀ [512,128]','edge-note')+label(900,204,'V [128,512]','edge-note')+
      batchRow(0)+batchRow(1)+
      node('partial',null,790,765,380,70,'Partial m / l / o','4 × 16-Head segments')+
      node('merge',null,790,873,380,70,'csa_merge_pack_publish','Merge · Normalize · Pack · Publish')+
      path('M980 835 V873','output-flow');}
    function invalidGraph(item){return node('mask',null,24,24,252,76,'valid_block_mask','Q'+item.token+' / SB'+item.sparseBlock+' = Invalid')+
      label(24,158,'Valid Path · 跳过','section')+
      '<g class="csa-qp-skipped">'+node('gather',null,24,194,200,65,'KV Gather','Skipped')+node('qk',null,250,194,160,65,'QK','Skipped')+node('stats',null,436,194,215,65,'Softmax Statistics','Skipped')+node('pv',null,677,194,160,65,'PV','Skipped')+'</g>'+
      node('neutral',null,470,370,360,100,'Neutral Partial Write','mi = −3.0e38 · li = 0 · oi = 0')+
      path('M276 62 H355 V420 H470','branch-flow')+label(310,341,'Invalid Path','section')+
      node('partial',null,470,615,360,85,'Partial m / l / o','4 × 16-Head neutral segments')+
      node('merge',null,470,805,360,85,'csa_merge_pack_publish','后续合并多个 Sparse Blocks')+
      path('M650 470 V615','output-flow')+path('M650 700 V805','output-flow');}
    function itemView(){const item=currentItem();if(!item)return laneView();const d=detail(item),graph=(item.valid?validGraph(item):invalidGraph(item)),selectedBatch=selection?.batchIndex;
      return '<div class="csa-qp-item-top"><button class="btn btn-ghost btn-sm" data-qp-back>← Lane '+lane+'</button><b>Work Item '+item.id+' · Q'+item.token+' / SB'+item.sparseBlock+'</b><span class="csa-qp-valid '+(item.valid?'':'is-invalid')+'">'+(item.valid?'Valid':'Invalid')+'</span><span>qk_order['+d.queuePosition+'] · it '+d.iteration+'</span><button class="btn btn-ghost btn-sm" data-qp-prev '+(d.iteration===0?'disabled':'')+'>←</button><button class="btn btn-ghost btn-sm" data-qp-next '+(d.iteration>=scenario.blocks[lane].entries.length-1?'disabled':'')+'>→</button></div>'+
      (item.valid?'<div class="csa-qp-batch-nav"><button class="btn btn-sm '+(selectedBatch==null?'is-selected':'')+'" data-qp-batch-nav="all">整个 Work Item</button><button class="btn btn-sm '+(selectedBatch===0?'is-selected':'')+'" data-qp-batch-nav="0">Iteration 0 · Heads 0–31</button><button class="btn btn-sm '+(selectedBatch===1?'is-selected':'')+'" data-qp-batch-nav="1">Iteration 1 · Heads 32–63</button></div>':'')+
      '<div class="csa-qp-stage"><div class="csa-qp-stage-inner"><svg class="csa-qp-svg" viewBox="0 0 1240 970" style="transform:scale('+scale+')" data-selected-operation="'+esc(selection?.operation||'')+'" data-selected-batch="'+(selectedBatch??'')+'" aria-label="Work Item 内部计算数据流"><defs><marker id="csa-qp-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z"/></marker></defs>'+graph+'</svg></div>'+inspector()+'</div><div class="csa-qp-bottom"><span>QK_M_TILE=32 · H_TILE=16 · ATTN_K_TILE=128</span><span><code>pl.pipeline(H // QK_M_TILE, stage=2)</code>：2 次逻辑迭代；stage=2 是软件流水声明，执行重叠未验证</span><div><button class="btn btn-ghost btn-sm" data-qp-zoom="0.8">−</button><button class="btn btn-ghost btn-sm" data-qp-fit>适应</button><button class="btn btn-ghost btn-sm" data-qp-zoom="1.25">＋</button></div></div>';}
    function render(){if(!scenario)return;const oldStage=host.querySelector('.csa-qp-stage-inner'),left=oldStage?.scrollLeft||0,top=oldStage?.scrollTop||0;host.innerHTML='<div class="csa-qp-shell"><div class="csa-qp-context"><b>qk_pv Pipeline</b><span>Rank '+context.rank+' · '+(context.mode==='source'?'Source-scale':'Teaching')+' · '+(context.workload==='mixed'?'Mixed Context':'Position '+context.position)+' · '+scenario.items.length+' Work Items / '+scenario.blockCount+' Lanes</span></div>'+(level==='item'&&currentItem()?itemView():laneView())+'</div>';const stage=host.querySelector('.csa-qp-stage-inner');if(stage){stage.scrollLeft=left;stage.scrollTop=top;}}
    function update(nextFocus,nextContext,sharedScenario){context={mode:nextContext.mode||'source',workload:nextContext.workload||'mixed',position:nextContext.position??8192,rank:nextContext.rank??nextFocus?.rank??0};scenario=sharedScenario||work.deriveScenario(context);const same=nextFocus?.kind==='work-item'&&nextFocus.scenarioId===scenarioId()&&nextFocus.rank===context.rank&&scenario.byId.has(nextFocus.workItemId);focus=same?itemFocus(scenario.byId.get(nextFocus.workItemId)):nextFocus;if(same){lane=focus.laneId;level='item';}else if(nextFocus?.kind==='lane'&&nextFocus.scenarioId===scenarioId()&&nextFocus.rank===context.rank){lane=nextFocus.laneId;level='lane';}else{lane=Math.min(lane,scenario.blockCount-1);level='lane';}selection=null;render();}
    host.addEventListener('click',event=>{const target=event.target.closest('button,[data-qp-operation],[data-qp-pick-batch]');if(!target)return;
      if(target.dataset.qpLane!=null)return pickLane(Number(target.dataset.qpLane));
      if(target.dataset.qpItem!=null)return pickItem(Number(target.dataset.qpItem));
      if(target.hasAttribute('data-qp-back'))return pickLane(lane);
      if(target.hasAttribute('data-qp-prev')||target.hasAttribute('data-qp-next')){const index=focus.laneIteration+(target.hasAttribute('data-qp-next')?1:-1),entry=scenario.blocks[lane].entries[index];if(entry)pickItem(entry.id);return;}
      if(target.dataset.qpBatchNav!=null){const value=target.dataset.qpBatchNav;selection=value==='all'?null:{workItemId:focus.workItemId,batchIndex:Number(value),operation:null};render();return;}
      if(target.dataset.qpPickBatch!=null)return select(Number(target.dataset.qpPickBatch),null);
      if(target.dataset.qpOperation){const batch=target.dataset.qpBatch==null?null:Number(target.dataset.qpBatch);select(batch,target.dataset.qpOperation);return;}
      if(target.hasAttribute('data-qp-close')){selection=null;render();return;}
      if(target.dataset.qpSource){const [file,a,z]=target.dataset.qpSource.split(':');options.onSource(file,Number(a),Number(z));return;}
      if(target.dataset.qpZoom){scale=Math.max(.7,Math.min(1.8,scale*Number(target.dataset.qpZoom)));render();return;}
      if(target.hasAttribute('data-qp-fit')){scale=1;render();}
    });
    host.addEventListener('keydown',event=>{if(event.key!=='Enter'&&event.key!==' ')return;const target=event.target;if(target.dataset.qpOperation){event.preventDefault();select(target.dataset.qpBatch==null?null:Number(target.dataset.qpBatch),target.dataset.qpOperation);}else if(target.dataset.qpPickBatch!=null){event.preventDefault();select(Number(target.dataset.qpPickBatch),null);}});
    return {update,getLocalSelection:()=>selection?{...selection}:null};
  }
  global.PtoCsaQkPipeline=Object.freeze({create});
})(window);
