export const variants = [
 {id:'separate',label:'独立 scope',addresses:[32768,32832,32896,32960],result:'PASS',detail:'相同 mix_x 计算放在独立 scope 中，精度通过。'},
 {id:'fused',label:'融合 scope',addresses:[74304,74304,74304,74304],result:'FAIL · 97.94%',detail:'post、comb 通过；x_mixed 失败。四个 per-head scale 被报告为同址。'},
 {id:'materialized',label:'融合 · 独立 buffer',addresses:[512,576,640,704],result:'FAIL · 约 97.9%',detail:'通过 mul(..., 1.0) 实体化 scale，地址分离后仍错误；剩余根因未确认。'}
];
export const checkpoints = [
 {name:'Scope 对照',detail:'根据 Issue 源码描述重建局部计算关系；不代表完整编译 dump。'},
 {name:'InitMemref',detail:'待获取本案例 Pass 输入/输出快照；当前仅保留计算关系。'},
 {name:'MemoryReuse',detail:'待获取别名与 live range 产物；不能由最终共址倒推本 Pass 已发生错误。'},
 {name:'LegalizePtoBufferReuse',detail:'案例列出的排查阶段；对应版本产物尚未接入。'},
 {name:'AllocateMemoryAddr',detail:'展示 Issue 报告的最终 UB 起始地址；大小、精确区间与首次共址阶段待获取。'},
 {name:'数值结果',detail:'报告结果与地址对照；未重新执行硬件实验。'}
];
export const source='https://github.com/hw-native-sys/pypto/issues/1723';
