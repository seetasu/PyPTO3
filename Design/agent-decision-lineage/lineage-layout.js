/* Semantic presentation projection. Coordinates are stable across selection/focus.
   Ribbons have fixed visual width: they encode relationships, never counts. */
window.LINEAGE_VIEW = {
 width:1688,height:500,
 groups:[
 {id:'requirements',stage:1,x:24,y:210,w:152,h:112,label:'Softmax SPEC',sub:'fp16 · [B,N] → [B,N]',items:[['e1','数值稳定行归约','需求合同']]},
 {id:'constraints',stage:2,x:218,y:175,w:178,h:205,label:'知识 / API / 硬件约束',sub:'设计依据',items:[['d-kb','知识选择','归约 · 尾块 · 生命周期'],['d-api','VF API 适配','ZERO / ONE · fp32 sum'],['d-constraint','A5 / 950 · UB 248KB','初始设备为默认假设']]},
 {id:'design',stage:3,x:438,y:140,w:205,h:295,label:'采用的设计',sub:'Design Decision Lineage',items:[['d-vf','Vector / VF','纯 Vector · 无 Cube'],['d-precision','Mixed Precision','fp16 → fp32 → fp16'],['d-tiling','Single Tile 3×8192','整行归约 · 无跨 tile 状态'],['d-buffer','UB 192KB / 248KB','Double Buffer · in/out 双槽'],['d-schedule','Scheduling · 行块分核','单 kernel launch']]},
 {id:'kernel',stage:4,x:687,y:210,w:153,h:112,label:'PyPTO Kernel',sub:'生成的实现',items:[['i-kernel','三趟 max / sum / div','softmax_rows_vf']]},
 {id:'correctness',stage:4,x:875,y:210,w:146,h:112,label:'Correctness PASS',sub:'报告记录 · 非本机复测',items:[['x-verify','✓ 8/8 PASS','与 Baseline 精度一致']]},
 {id:'baseline',stage:5,x:1057,y:245,w:120,h:112,label:'Baseline',sub:'[32,4096] · kernel',items:[['x-base','5.926 μs','round_001']]},
 {id:'optimization',stage:5,x:1220,y:225,w:130,h:160,label:'Mask Lifting',sub:'✓ 采用 · full mask 提升',items:[['x-mask','提升并复用 mask','三趟计算保持'],['x-tail','Empty-tail-loop','组成部分 · 非独立收益']]},
 {id:'final',stage:5,x:1388,y:245,w:110,h:112,label:'Final',sub:'[32,4096] · kernel',items:[['x-final','4.946 μs','1.198× vs baseline']]},
 {id:'stop',stage:5,x:1535,y:245,w:130,h:112,label:'Target Met / Stop',sub:'✓ 目标与门槛通过',items:[['d-stop','为什么停止','保留未探索机会']]}
 ],
 branches:[
 {id:'d-online',parent:'d-tiling',x:447,y:502,w:192,h:72,title:'跨 tile 流式归约',subtitle:'− 设计未采用 · 整行已容纳',mode:'design',relation:'selected over'}
 ],
 // Fixed-width hierarchy, never population/throughput. Junctions are presentation only.
 river:{
   junction:{x:1196,y:300,label:'optimization alternatives · 展示分岔'},
   finding:{id:'f-mask',x:1057,y:160,w:137,h:52,label:'Finding · mask 重复',sub:'update_mask 开销线索'},
   failed:{id:'x-if',x:1255,y:131,w:165,h:61,label:'Runtime if',sub:'× tried & failed · rejected end'},
   opportunities:[
     {id:'x-rows',x:1100,y:419,w:165,h:49,label:'TILE_ROWS Adjustment',sub:'○ not attempted'},
     {id:'x-reuse',x:1300,y:419,w:165,h:49,label:'Exp Result Reuse',sub:'○ not attempted'},
     {id:'x-fusion',x:1500,y:419,w:165,h:49,label:'Pass Fusion',sub:'○ not attempted'}
   ]
 },
 /* The story connections below are sourced presentation relations, not new runs. */
 links:[
 ['requirements','constraints','workflow handoff','EXPLORE_REPORT.md','§1.1 输入摘要'],
 ['constraints','design','constrained by','DESIGN.md','Knowledge Bindings / §1–3'],
 ['design','kernel','produces','test_softmax.py','softmax_rows_vf / softmax_tile_group_kernel'],
 ['kernel','correctness','validated by','PERFORMANCE_REPORT.md','§8'],
 ['correctness','baseline','workflow handoff','PERFORMANCE_REPORT.md','§3 Stage 4 implementation'],
 ['baseline','optimization','improved by','PERFORMANCE_REPORT.md','§3–4'],
 ['optimization','final','measured as','PERFORMANCE_REPORT.md','§5'],
 ['final','stop','supports stop','PERFORMANCE_REPORT.md','§6、§11']
 ]
};
