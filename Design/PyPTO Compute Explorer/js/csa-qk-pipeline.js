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
    let focus=null,context=null,scenario=null,lane=0,level='lane',selection=null,camera=null,drag=null,mapMode='path',lastItemId=null,mapCamera=null,mapDrag=null;
    const bounds={x:0,y:0,width:1240,height:970},padding=24;
    const moduleId=options.moduleId;
    const scenarioId=()=>scenario?.items[0]?.scenarioId;
    const detail=item=>work.getWorkItemDetails(item.id,scenario.queue,scenario.blockCount);
    function itemFocus(item){const d=detail(item);return {kind:'work-item',rank:context.rank,scenarioId:scenarioId(),moduleId,operationId:moduleId+'/dispatch',workItemId:item.id,requestIndex:item.requestIndex,tokenIndex:item.token,positionId:item.positionId,sparseBlockIndex:item.sparseBlock,valid:item.valid,validKvCount:item.validKvCount,queuePosition:d.queuePosition,laneId:d.block,laneIteration:d.iteration};}
    const laneFocus=index=>({kind:'lane',rank:context.rank,scenarioId:scenarioId(),moduleId,operationId:moduleId+'/dispatch',blockId:index,laneId:index});
    const currentItem=()=>focus?.kind==='work-item'&&focus.scenarioId===scenarioId()&&focus.rank===context.rank?scenario.byId.get(focus.workItemId):null;
    function pickItem(id){const item=scenario.byId.get(id);if(!item)return;focus=itemFocus(item);lane=focus.laneId;lastItemId=id;level='item';selection=null;options.onFocus(focus);render();}
    function pickLane(index){if(index<0||index>=scenario.blockCount)return;if(level!=='item')lastItemId=null;lane=index;level='lane';selection=null;focus=laneFocus(index);options.onFocus(focus);render();}
    function pickRequest(index){lastItemId=null;focus=index<0?{kind:'module',rank:context.rank,semanticId:moduleId,moduleId}:{kind:'request',rank:context.rank,scenarioId:scenarioId(),moduleId,requestIndex:index};level='lane';options.onFocus(focus);render();}
    function pickQuery(index){lastItemId=null;focus={kind:'query',rank:context.rank,scenarioId:scenarioId(),moduleId,requestIndex:Math.floor(index/8),tokenIndex:index};level='lane';options.onFocus(focus);render();}
    function mapSvg(){return host.querySelector('.csa-qp-map-svg');}
    function applyMapCamera(){const svg=mapSvg();if(!svg)return;const fullWidth=Number(svg.dataset.mapWidth),fullHeight=Number(svg.dataset.mapHeight);if(!mapCamera)mapCamera={x:0,y:0,width:fullWidth,height:fullHeight};mapCamera.width=Math.min(fullWidth,mapCamera.width);mapCamera.height=Math.min(fullHeight,mapCamera.height);mapCamera.x=Math.max(0,Math.min(fullWidth-mapCamera.width,mapCamera.x));mapCamera.y=Math.max(0,Math.min(fullHeight-mapCamera.height,mapCamera.y));svg.setAttribute('viewBox',[mapCamera.x,mapCamera.y,mapCamera.width,mapCamera.height].join(' '));}
    function zoomMap(factor){const svg=mapSvg();if(!svg)return;applyMapCamera();const fullWidth=Number(svg.dataset.mapWidth),fullHeight=Number(svg.dataset.mapHeight),scale=Math.max(1,Math.min(6,fullWidth/mapCamera.width*factor)),cx=mapCamera.x+mapCamera.width/2,cy=mapCamera.y+mapCamera.height/2;mapCamera.width=fullWidth/scale;mapCamera.height=fullHeight/scale;mapCamera.x=cx-mapCamera.width/2;mapCamera.y=cy-mapCamera.height/2;applyMapCamera();}
    function viewport(){return host.querySelector('.csa-qp-stage-inner');}
    function cameraSize(){const view=viewport();return {width:view?.clientWidth||0,height:view?.clientHeight||0};}
    function clampCamera(){if(!camera)return;const visible=80;for(const [axis,size] of [['x','width'],['y','height']]){const low=bounds[axis]-camera[size]+visible,high=bounds[axis]+bounds[size]-visible;camera[axis]=Math.max(low,Math.min(high,camera[axis]));}}
    function applyCamera(){const svg=host.querySelector('.csa-qp-svg');if(!svg||!camera)return;const size=cameraSize();camera.viewportWidth=size.width;camera.viewportHeight=size.height;svg.setAttribute('viewBox',[camera.x,camera.y,camera.width,camera.height].join(' '));}
    function fitCamera(){const size=cameraSize();if(size.width<1||size.height<1)return;const scale=Math.min((size.width-2*padding)/bounds.width,(size.height-2*padding)/bounds.height);if(!(scale>0))return;camera={x:bounds.x+(bounds.width-size.width/scale)/2,y:bounds.y+(bounds.height-size.height/scale)/2,width:size.width/scale,height:size.height/scale,mode:'fit'};applyCamera();}
    function zoomCamera(factor,clientX,clientY){const view=viewport();if(!view||!camera)return;const rect=view.getBoundingClientRect(),px=clientX==null?.5:Math.max(0,Math.min(1,(clientX-rect.left)/rect.width)),py=clientY==null?.5:Math.max(0,Math.min(1,(clientY-rect.top)/rect.height));const fitScale=Math.min((rect.width-2*padding)/bounds.width,(rect.height-2*padding)/bounds.height),currentScale=rect.width/camera.width,nextScale=Math.max(fitScale*.75,Math.min(fitScale*6,currentScale*factor)),width=rect.width/nextScale,height=rect.height/nextScale;camera={x:camera.x+camera.width*px-width*px,y:camera.y+camera.height*py-height*py,width,height,mode:'manual'};clampCamera();applyCamera();}
    function focusNode(operation,batch){const svg=host.querySelector('.csa-qp-svg');if(!svg||!camera||!operation)return;const target=svg.querySelector('[data-qp-operation="'+operation+'"]'+(batch==null?'':'[data-qp-batch="'+batch+'"]'));if(!target)return;const box=target.getBBox();camera.x=box.x+box.width/2-camera.width/2;camera.y=box.y+box.height/2-camera.height/2;camera.mode='manual';clampCamera();applyCamera();}
    function resizeCamera(){if(!camera||!viewport())return;if(camera.mode==='fit')return fitCamera();const size=cameraSize(),scale=camera.viewportWidth/camera.width;if(!(size.width>0&&size.height>0&&scale>0))return;const cx=camera.x+camera.width/2,cy=camera.y+camera.height/2;camera.width=size.width/scale;camera.height=size.height/scale;camera.x=cx-camera.width/2;camera.y=cy-camera.height/2;clampCamera();applyCamera();}
    function syncSelection(){const svg=host.querySelector('.csa-qp-svg');if(!svg)return;svg.dataset.selectedOperation=selection?.operation||'';svg.dataset.selectedBatch=selection?.batchIndex??'';svg.querySelectorAll('[data-qp-batch-row]').forEach(el=>el.classList.toggle('is-dimmed',selection?.batchIndex!=null&&Number(el.dataset.qpBatchRow)!==selection.batchIndex));svg.querySelectorAll('[data-qp-operation]').forEach(el=>{const batch=el.dataset.qpBatch==null?null:Number(el.dataset.qpBatch);el.classList.toggle('is-selected',selection?.operation===el.dataset.qpOperation&&selection?.batchIndex===batch);el.classList.toggle('is-dimmed',selection?.batchIndex!=null&&batch!=null&&batch!==selection.batchIndex);});host.querySelectorAll('[data-qp-batch-nav]').forEach(el=>el.classList.toggle('is-selected',el.dataset.qpBatchNav===(selection?.batchIndex==null?'all':String(selection.batchIndex))));const aside=host.querySelector('.csa-qp-inspector');if(aside){aside.innerHTML=inspector();aside.hidden=!selection?.operation;}requestAnimationFrame(()=>{if(camera?.mode==='fit')fitCamera();else{resizeCamera();if(selection?.operation)focusNode(selection.operation,selection.batchIndex);}});}
    function select(batch,operation){selection={workItemId:focus.workItemId,batchIndex:batch,operation};syncSelection();}
    function navActions(){const item=currentItem(),operation=selection?.operation;if(!item||!operation)return '';
      const nav=(id,label,title)=>'<button class="btn btn-ghost btn-sm csa-qp-nav" data-qp-nav="'+id+'" title="'+esc(title||label)+'">'+esc(label)+'</button>';
      let body='';
      // P5：只发出跳转请求，Tab 与 Focus 由 csa-explorer 统一管理。
      if(operation==='query')body=nav('input/q-tensor','查看 Q 来源 · Q heads','Q heads 由 Q Projection 生成');
      else if(operation==='gather')body=item.sparseBlock===0
        ? nav('cache/original','查看上游 · Original KV Cache','SB0 从 Original KV 的 Page 连续区间 Gather')
        : nav('cache/compressed','查看上游 · Compressed KV Cache','QK/PV 实际读取的压缩 KV')+nav('cache/topk-positions','位置来源 · Top-K Positions','这些压缩位置由 Indexer 的相关性筛选产生');
      else if(operation==='partial')body=nav('redistribute/publish','查看下游 · Merge / Pack / Publish','csa_merge_pack_publish 合并多个 Sparse Block 的 partial')+'<span class="csa-qp-nav-label">Partial Tensor · sparse_blk_mi / li / oi</span>'+nav('sparse/partial-mi','mi','sparse_blk_mi')+nav('sparse/partial-li','li','sparse_blk_li')+nav('sparse/partial-oi','oi','sparse_blk_oi');
      else if(operation==='mask')body=nav('sparse/attention/valid-block-mask','查看上游 · Valid Block Mask','每个 (Query, Sparse Block) 的有效性');
      return body?'<div class="csa-qp-nav-group">'+body+'</div>':'';
    }
    function inspector(){if(!selection)return '';const info=evidence[selection.operation]||evidence.query,item=currentItem(),batch=selection.batchIndex;
      return '<header><b>'+esc(info.title)+'</b><button class="btn btn-ghost btn-sm" data-qp-close aria-label="关闭详情">×</button></header>'+(batch!=null?'<span class="csa-qp-inspector-scope">Work Item '+item.id+' / Head Batch '+batch+' · Heads '+(batch*32)+'–'+(batch*32+31)+'</span>':'<span class="csa-qp-inspector-scope">Work Item '+item.id+'</span>')+'<p>'+esc(info.fact)+'</p><button class="btn btn-ghost btn-sm" data-qp-source="'+info.file+':'+info.range.join(':')+'">'+esc(info.file)+' : '+info.range.join('–')+' ↗</button>'+navActions();}
    function mapToolbar(){if(!scenario)return '';const columns=Math.max(...scenario.blocks.map(block=>block.entries.length));
      const active=focus?.scenarioId===scenarioId()&&focus?.rank===context.rank?focus:null;
      const request=active?.kind==='request'?active.requestIndex:null,query=active?.kind==='query'?active.tokenIndex:null,requestValue=request??(query!=null?Math.floor(query/8):'');
      const requestOptions=Array.from({length:Math.ceil(scenario.items.length/40)},(_,i)=>'<option value="'+i+'"'+(requestValue===i?' selected':'')+'>Request '+i+'</option>').join('');
      const queryOptions=Array.from({length:scenario.items.length/5},(_,i)=>'<option value="'+i+'"'+(query===i?' selected':'')+'>Q'+i+'</option>').join('');
      return '<b>Lane Workload Map</b><span>'+scenario.blockCount+' Lanes × '+columns+' Iterations</span><label>Request <select data-qp-map-request><option value="">全部</option>'+requestOptions+'</select></label><label>Query <select data-qp-map-query><option value="">全部</option>'+queryOptions+'</select></label><label class="csa-qp-map-mode'+(mapMode==='rows'?' is-on':'')+'"><input type="checkbox" data-qp-map-rows'+(mapMode==='rows'?' checked':'')+'><span>Valid KV Rows</span></label>';
    }
    function laneView(){
      const columns=Math.max(...scenario.blocks.map(block=>block.entries.length)),cell=19,gap=3,row=25,left=73,top=43,summaryX=left+columns*(cell+gap)+9,width=summaryX+150,height=top+scenario.blockCount*row+27;
      const active=focus?.scenarioId===scenarioId()&&focus?.rank===context.rank?focus:null;
      const request=active?.kind==='request'?active.requestIndex:null,query=active?.kind==='query'?active.tokenIndex:null,selectedLane=active?.kind==='lane'?active.laneId:null;
      const ticks=Array.from({length:columns},(_,i)=>i%5===0||i===columns-1?'<text x="'+(left+i*(cell+gap)+cell/2)+'" y="27" text-anchor="middle">'+i+'</text>':'').join('');
      const rows=scenario.blocks.map(block=>{
        let original=0,compressed=0,invalid=0;
        const cells=block.entries.map(entry=>{
          const item=scenario.byId.get(entry.id);if(!item)return '';
          const type=!item.valid?'invalid':item.sparseBlock===0?'original':'compressed';
          if(type==='invalid')invalid++;else if(type==='original')original++;else compressed++;
          const related=request!=null?item.requestIndex===request:query!=null?item.token===query:selectedLane!=null?block.index===selectedLane:true;
          const note='Work Item '+item.id+' · Q'+item.token+' / SB'+item.sparseBlock+' · qk_order['+entry.queuePosition+'] · Lane '+block.index+' / iteration '+entry.iteration+' · '+(item.valid?type==='original'?'Original KV':'Compressed KV':'Invalid Skip')+' · '+item.validKvCount+' / 128 valid rows';
          const intensity=item.valid?Math.max(.18,Math.min(1,item.validKvCount/128)):0;
          return '<rect class="csa-qp-map-cell is-'+type+(related?'':' is-dimmed')+(item.id===lastItemId?' is-last':'')+'" x="'+(left+entry.iteration*(cell+gap))+'" y="'+(top+block.index*row)+'" width="'+cell+'" height="18" rx="2" data-qp-item="'+item.id+'" role="button" tabindex="0" aria-label="'+esc(note)+'" style="--kv-intensity:'+intensity+'"><title>'+esc(note)+'</title></rect>';
        }).join('');
        const y=top+block.index*row;
        return '<g class="csa-qp-map-row'+(selectedLane!=null&&selectedLane!==block.index?' is-dimmed':'')+'"><text class="csa-qp-map-lane" x="5" y="'+(y+14)+'" data-qp-lane="'+block.index+'" role="button" tabindex="0">Lane '+String(block.index).padStart(2,'0')+'</text>'+cells+'<text class="csa-qp-map-count" x="'+summaryX+'" y="'+(y+14)+'">'+original+' O · '+compressed+' C · '+invalid+' I</text></g>';
      }).join('');
      const block=selectedLane==null?null:scenario.blocks[selectedLane];
      const summary=block?'Lane '+selectedLane+' · '+block.entries.length+' Work Items · '+block.validCount+' Valid / '+block.invalidCount+' Invalid · qk_order['+selectedLane+'] + iteration × '+scenario.blockCount:request!=null?'Request '+request+' · '+scenario.items.filter(item=>item.requestIndex===request).length+' Work Items':query!=null?'Q'+query+' · 5 Sparse Block Work Items':'按 qk_order 的 Lane 迭代位置排列；列不表示时间或跨 Lane 等待';
      return '<div class="csa-qp-map"><div class="csa-qp-map-viewport" tabindex="0" role="group" aria-label="Lane Workload Map，拖拽平移，选择单元格查看 Work Item"><svg class="csa-qp-map-svg mode-'+mapMode+'" viewBox="0 0 '+width+' '+height+'" data-map-width="'+width+'" data-map-height="'+height+'" aria-label="SPMD Lane 与 Work Item 迭代矩阵"><text class="csa-qp-map-axis" x="'+left+'" y="14">Iteration →</text>'+ticks+rows+'</svg></div><div class="csa-qp-map-footer"><span>'+esc(summary)+'</span><span class="csa-qp-map-legend"><i class="is-original"></i>Original KV <i class="is-compressed"></i>Compressed KV <i class="is-invalid"></i>Invalid Skip'+(mapMode==='rows'?' · 颜色强度 = 有效 KV 行数 / 128':'')+'</span><div><button class="btn btn-ghost btn-sm" data-qp-map-zoom="0.8">−</button><button class="btn btn-ghost btn-sm" data-qp-map-fit>适应</button><button class="btn btn-ghost btn-sm" data-qp-map-zoom="1.25">＋</button></div></div></div>';
    }
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
      return '<div class="csa-qp-item-top"><button class="btn btn-ghost btn-sm" data-qp-back>← Lane '+lane+'</button><b>Work Item '+item.id+' · Q'+item.token+' / SB'+item.sparseBlock+'</b><span class="csa-qp-valid '+(item.valid?'':'is-invalid')+'">'+(item.valid?'Valid':'Invalid')+'</span><button class="btn btn-ghost btn-sm csa-qp-orderlink" data-qp-nav="sparse/attention/qk-order" title="查看 Work Ordering 写出的 qk_order">qk_order['+d.queuePosition+'] · it '+d.iteration+' ↗</button><button class="btn btn-ghost btn-sm csa-qp-orderlink" data-qp-nav="sparse/attention/valid-block-mask" title="查看 valid_block_mask">valid_block_mask ↗</button><button class="btn btn-ghost btn-sm" data-qp-prev '+(d.iteration===0?'disabled':'')+'>←</button><button class="btn btn-ghost btn-sm" data-qp-next '+(d.iteration>=scenario.blocks[lane].entries.length-1?'disabled':'')+'>→</button></div>'+
      (item.valid?'<div class="csa-qp-batch-nav"><button class="btn btn-sm '+(selectedBatch==null?'is-selected':'')+'" data-qp-batch-nav="all">整个 Work Item</button><button class="btn btn-sm '+(selectedBatch===0?'is-selected':'')+'" data-qp-batch-nav="0">Iteration 0 · Heads 0–31</button><button class="btn btn-sm '+(selectedBatch===1?'is-selected':'')+'" data-qp-batch-nav="1">Iteration 1 · Heads 32–63</button></div>':'')+
      '<div class="csa-qp-stage"><div class="csa-qp-stage-inner" tabindex="0" role="group" aria-label="计算画布，拖拽或触控板平移，方向键移动，Ctrl 或 Command 滚轮缩放"><svg class="csa-qp-svg" viewBox="0 0 1240 970" data-selected-operation="'+esc(selection?.operation||'')+'" data-selected-batch="'+(selectedBatch??'')+'" aria-label="Work Item 内部计算数据流"><defs><marker id="csa-qp-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z"/></marker></defs>'+graph+'</svg></div><aside class="csa-qp-inspector" aria-label="计算详情" hidden></aside></div><div class="csa-qp-bottom"><span>QK_M_TILE=32 · H_TILE=16 · ATTN_K_TILE=128</span><span><code>pl.pipeline(H // QK_M_TILE, stage=2)</code>：2 次逻辑迭代；stage=2 是软件流水声明，执行重叠未验证</span><div><button class="btn btn-ghost btn-sm" data-qp-zoom="0.8">−</button><button class="btn btn-ghost btn-sm" data-qp-fit>适应</button><button class="btn btn-ghost btn-sm" data-qp-zoom="1.25">＋</button></div></div>';}
    function render(){if(!scenario)return;const itemMode=level==='item'&&currentItem();host.innerHTML='<div class="csa-qp-shell"><div class="csa-qp-context"><span class="csa-qp-context-scope">Rank '+context.rank+' · qk_pv</span>'+(itemMode?'':'<div class="csa-qp-context-tools">'+mapToolbar()+'</div>')+'</div>'+(itemMode?itemView():laneView())+'</div>';if(itemMode)requestAnimationFrame(()=>{if(camera?.mode==='manual')resizeCamera();else fitCamera();syncSelection();});else applyMapCamera();}
    function update(nextFocus,nextContext,sharedScenario){const previous=focus,previousScenario=scenarioId();context={mode:nextContext.mode||'source',workload:nextContext.workload||'mixed',position:nextContext.position??8192,rank:nextContext.rank??nextFocus?.rank??0};scenario=sharedScenario||work.deriveScenario(context);if(previousScenario!==scenarioId()){mapCamera=null;lastItemId=null;}const same=nextFocus?.kind==='work-item'&&nextFocus.scenarioId===scenarioId()&&nextFocus.rank===context.rank&&scenario.byId.has(nextFocus.workItemId);focus=same?itemFocus(scenario.byId.get(nextFocus.workItemId)):nextFocus;if(same){lane=focus.laneId;lastItemId=focus.workItemId;level='item';}else if(nextFocus?.kind==='lane'&&nextFocus.scenarioId===scenarioId()&&nextFocus.rank===context.rank){lane=nextFocus.laneId;level='lane';}else{lane=Math.min(lane,scenario.blockCount-1);level='lane';}if(previous?.workItemId!==focus?.workItemId||previous?.scenarioId!==focus?.scenarioId){selection=null;camera=null;}render();}
    host.addEventListener('click',event=>{const target=event.target.closest('button,[data-qp-operation],[data-qp-pick-batch],[data-qp-item],[data-qp-lane]');if(!target)return;
      if(target.dataset.qpLane!=null)return pickLane(Number(target.dataset.qpLane));
      if(target.dataset.qpItem!=null)return pickItem(Number(target.dataset.qpItem));
      if(target.dataset.qpMapZoom){zoomMap(Number(target.dataset.qpMapZoom));return;}
      if(target.hasAttribute('data-qp-map-fit')){mapCamera=null;applyMapCamera();return;}
      if(target.dataset.qpNav)return options.onNavigate?.(target.dataset.qpNav);
      if(target.hasAttribute('data-qp-back'))return pickLane(lane);
      if(target.hasAttribute('data-qp-prev')||target.hasAttribute('data-qp-next')){const index=focus.laneIteration+(target.hasAttribute('data-qp-next')?1:-1),entry=scenario.blocks[lane].entries[index];if(entry)pickItem(entry.id);return;}
      if(target.dataset.qpBatchNav!=null){const value=target.dataset.qpBatchNav;selection=value==='all'?null:{workItemId:focus.workItemId,batchIndex:Number(value),operation:null};syncSelection();return;}
      if(target.dataset.qpPickBatch!=null)return select(Number(target.dataset.qpPickBatch),null);
      if(target.dataset.qpOperation){const batch=target.dataset.qpBatch==null?null:Number(target.dataset.qpBatch);select(batch,target.dataset.qpOperation);return;}
      if(target.hasAttribute('data-qp-close')){selection=null;syncSelection();return;}
      if(target.dataset.qpSource){const [file,a,z]=target.dataset.qpSource.split(':');options.onSource(file,Number(a),Number(z));return;}
      if(target.dataset.qpZoom){zoomCamera(Number(target.dataset.qpZoom));return;}
      if(target.hasAttribute('data-qp-fit'))fitCamera();
    });
    host.addEventListener('change',event=>{if(event.target.matches('[data-qp-map-request]')){const value=event.target.value;value===''?pickRequest(-1):pickRequest(Number(value));}else if(event.target.matches('[data-qp-map-query]')){const value=event.target.value;value===''?pickRequest(-1):pickQuery(Number(value));}else if(event.target.matches('[data-qp-map-rows]')){mapMode=event.target.checked?'rows':'path';render();}});
    host.addEventListener('keydown',event=>{const target=event.target;if(target.classList.contains('csa-qp-stage-inner')&&camera&&['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key)){event.preventDefault();camera.x+=(event.key==='ArrowRight'?1:event.key==='ArrowLeft'?-1:0)*camera.width*.12;camera.y+=(event.key==='ArrowDown'?1:event.key==='ArrowUp'?-1:0)*camera.height*.12;camera.mode='manual';clampCamera();applyCamera();return;}if(event.key!=='Enter'&&event.key!==' ')return;if(target.dataset.qpItem!=null||target.dataset.qpLane!=null){event.preventDefault();target.click();}else if(target.dataset.qpOperation){event.preventDefault();select(target.dataset.qpBatch==null?null:Number(target.dataset.qpBatch),target.dataset.qpOperation);}else if(target.dataset.qpPickBatch!=null){event.preventDefault();select(Number(target.dataset.qpPickBatch),null);}});
    host.addEventListener('pointerdown',event=>{const view=event.target.closest('.csa-qp-map-viewport');if(event.button!==0||!view||event.target.closest('[data-qp-item],[data-qp-lane]'))return;applyMapCamera();mapDrag={x:event.clientX,y:event.clientY,camera:{...mapCamera},pointerId:event.pointerId};view.setPointerCapture(event.pointerId);view.classList.add('is-panning');});
    host.addEventListener('pointermove',event=>{if(!mapDrag||!mapCamera)return;const view=host.querySelector('.csa-qp-map-viewport'),box=view.getBoundingClientRect();if(!box.width||!box.height)return;mapCamera.x=mapDrag.camera.x-(event.clientX-mapDrag.x)*mapDrag.camera.width/box.width;mapCamera.y=mapDrag.camera.y-(event.clientY-mapDrag.y)*mapDrag.camera.height/box.height;applyMapCamera();});
    const endMapPan=()=>{mapDrag=null;host.querySelector('.csa-qp-map-viewport')?.classList.remove('is-panning');};host.addEventListener('pointerup',endMapPan);host.addEventListener('pointercancel',endMapPan);
    host.addEventListener('pointerdown',event=>{const view=event.target.closest('.csa-qp-stage-inner');if(event.button!==0||!view||event.target.closest('[data-qp-operation],[data-qp-pick-batch]')||!camera)return;drag={x:event.clientX,y:event.clientY,camera:{...camera},pointerId:event.pointerId};view.setPointerCapture(event.pointerId);view.classList.add('is-panning');});
    host.addEventListener('pointermove',event=>{if(!drag||!camera)return;const size=cameraSize();if(!size.width||!size.height)return;camera.x=drag.camera.x-(event.clientX-drag.x)*drag.camera.width/size.width;camera.y=drag.camera.y-(event.clientY-drag.y)*drag.camera.height/size.height;camera.mode='manual';clampCamera();applyCamera();});
    const endPan=()=>{drag=null;viewport()?.classList.remove('is-panning');};host.addEventListener('pointerup',endPan);host.addEventListener('pointercancel',endPan);
    host.addEventListener('wheel',event=>{if(!event.target.closest('.csa-qp-stage-inner')||!camera)return;event.preventDefault();if(event.ctrlKey||event.metaKey){zoomCamera(Math.exp(-event.deltaY*.003),event.clientX,event.clientY);}else{const size=cameraSize();camera.x+=event.deltaX*camera.width/size.width;camera.y+=event.deltaY*camera.height/size.height;camera.mode='manual';clampCamera();applyCamera();}},{passive:false});
    if(global.ResizeObserver)new global.ResizeObserver(()=>resizeCamera()).observe(host);
    return {update,getLocalSelection:()=>selection?{...selection}:null};
  }
  global.PtoCsaQkPipeline=Object.freeze({create});
})(window);
