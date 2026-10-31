/* Semantic presentation projection. Coordinates are stable across selection/focus.
   Ribbons have fixed visual width: they encode relationships, never counts. */
window.LINEAGE_VIEW = {
 width:1540,height:690,
 groups:[
 {id:'requirements',stage:1,x:24,y:210,w:152,h:112,label:'Softmax SPEC',sub:'fp16 · [B,N] → [B,N]',items:[['e1','数值稳定行归约','需求合同']]},
 {id:'constraints',stage:2,x:218,y:175,w:178,h:205,label:'知识 / API / 硬件约束',sub:'设计依据',items:[['d-kb','知识选择','归约 · 尾块 · 生命周期'],['d-api','VF API 适配','ZERO / ONE · fp32 sum'],['d-constraint','A5 / 950 · UB 248KB','初始设备为默认假设']]},
 {id:'design',stage:3,x:438,y:140,w:205,h:295,label:'采用的设计',sub:'Design Decision Lineage',items:[['d-vf','Vector / VF','纯 Vector · 无 Cube'],['d-precision','Mixed Precision','fp16 → fp32 → fp16'],['d-tiling','Single Tile 3×8192','整行归约 · 无跨 tile 状态'],['d-buffer','UB 192KB / 248KB','Double Buffer · in/out 双槽'],['d-schedule','行块跨核 striding','单 kernel launch']]},
 {id:'kernel',stage:4,x:687,y:210,w:153,h:112,label:'PyPTO Kernel',sub:'生成的实现',items:[['i-kernel','三趟 max / sum / div','softmax_rows_vf']]},
 {id:'correctness',stage:4,x:875,y:210,w:146,h:112,label:'Correctness PASS',sub:'报告记录 · 非本机复测',items:[['x-verify','✓ 8/8 PASS','与 Baseline 精度一致']]},
 {id:'baseline',stage:5,x:1057,y:210,w:132,h:112,label:'Baseline',sub:'[32,4096] · kernel',items:[['x-base','5.926 μs','round_001']]},
 {id:'optimization',stage:5,x:1225,y:175,w:135,h:165,label:'Mask Lifting',sub:'减少循环内 mask 开销',items:[['x-mask','✓ 采用','full mask 提升'],['x-tail','Empty tail loop','替代 runtime if']]},
 {id:'final',stage:5,x:1396,y:210,w:128,h:112,label:'Final',sub:'[32,4096] · kernel',items:[['x-final','4.946 μs','round_002 · 1.198×']]}
 ],
 branches:[
 {id:'d-stop',parent:'x-final',x:1396,y:375,w:128,h:64,title:'Target Met / Stop',subtitle:'✓ 交付门槛通过',relation:'supports stop'},
 {id:'d-online',parent:'d-tiling',x:447,y:502,w:192,h:72,title:'跨 tile 流式归约',subtitle:'− 设计未采用 · 整行已容纳',mode:'design',relation:'selected over'},
 {id:'x-if',parent:'x-mask',x:1070,y:428,w:208,h:78,title:'Runtime if',subtitle:'× 实际尝试失败 · C++ 作用域',relation:'failed attempt'},
 {id:'x-rows',parent:'x-mask',x:1070,y:565,w:208,h:70,title:'TILE_ROWS Adjustment',subtitle:'○ 未尝试 · 默认目标已满足',relation:'not explored'},
 {id:'x-reuse',parent:'x-mask',x:1310,y:565,w:208,h:70,title:'Exp Reuse / Pass Fusion',subtitle:'○ 未尝试 · 超出冻结设计',relation:'not explored'}
 ],
 /* The story connections below are sourced presentation relations, not new runs. */
 links:[
 ['requirements','constraints','workflow handoff','EXPLORE_REPORT.md','§1.1 输入摘要'],
 ['constraints','design','constrained by','DESIGN.md','Knowledge Bindings / §1–3'],
 ['design','kernel','produces','test_softmax.py','softmax_rows_vf / softmax_tile_group_kernel'],
 ['kernel','correctness','validated by','PERFORMANCE_REPORT.md','§8'],
 ['correctness','baseline','workflow handoff','PERFORMANCE_REPORT.md','§3 Stage 4 implementation'],
 ['baseline','optimization','improved by','PERFORMANCE_REPORT.md','§3–4'],
 ['optimization','final','measured as','PERFORMANCE_REPORT.md','§5']
 ]
};
