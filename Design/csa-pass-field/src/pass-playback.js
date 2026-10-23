// Pass timing is independent of the optional UB allocation player.
export function nextPassAction(pass,steps){
 const next=steps.find(index=>index>pass);
 return next==null?{type:'stop'}:{type:'pass',pass:next,delay:1800};
}
