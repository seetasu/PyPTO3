export const sampleFor=l=>l?.leaf==='csa_merge_pack_publish'?'csa_merge_pack_publish':/^qk_pv_(aic|aiv)$/.test(l?.leaf||'')?'qk_pv':null;
export function kernelsFor(id,sample,collections){
 if(!id)return [];
 const matches=collections.filter(c=>c.id===id||c.owner===id||c.kernelId===id);
 if(matches.length)return [...new Set(matches.flatMap(c=>c.instances?.length?c.instances.map(i=>i.kernel):[c.kernel]).filter(Boolean))];
 if(sample==='csa_merge_pack_publish'&&id==='kernel.main')return ['csa_merge_pack_publish'];
 if(id==='kernel.aic')return ['qk_pv_aic'];
 if(id==='kernel.aiv')return ['qk_pv_aiv'];
 return [];
}
