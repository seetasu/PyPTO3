(function(global){
  'use strict';
  const NS='http://www.w3.org/2000/svg';
  const fmt=n=>Number(n).toFixed(2)+' ms';
  function metrics(step){
    const critical=step.lanes.reduce((a,b)=>a.rrStart>b.rrStart?a:b);
    return {critical,bindSum:step.lanes.reduce((n,l)=>n+l.bindEnd-l.bindStart,0),startSpread:Math.max(...step.lanes.map(l=>l.rrStart))-Math.min(...step.lanes.map(l=>l.rrStart)),endSpread:Math.max(...step.lanes.map(l=>l.end))-Math.min(...step.lanes.map(l=>l.end))};
  }
  function fitRange(s){return [Math.min(s.t0,0,...s.lanes.map(l=>l.bindStart)),Math.max(s.t1,s.end)];}
  function boundedRange(a,b,full){const width=Math.min(full[1]-full[0],Math.max(5,b-a));const start=Math.min(full[1]-width,Math.max(full[0],a));return [start,start+width];}
  function mount(root,data,options={}){
    if(!data.steps?.length)throw new Error('Swimlane requires at least one step');
    data.steps.forEach(s=>{if(!s.lanes?.length)throw new Error('Step requires lanes');const ids=new Set();s.lanes.forEach(l=>{if(ids.has(l.dev)||![l.bindStart,l.bindEnd,l.rrStart,l.rrEnd,l.end].every(Number.isFinite)||l.bindStart>l.bindEnd||l.rrStart>l.rrEnd)throw new Error('Invalid lane windows');ids.add(l.dev);});});
    const abort=new AbortController(),on=(node,type,fn)=>node.addEventListener(type,fn,{signal:abort.signal});
    const hash=new URLSearchParams(options.syncHash?location.hash.slice(1):'');
    let index=Math.max(0,data.steps.findIndex(s=>s.id===(hash.get('step')||options.step||'D09'))),selected=null,hover=null,layer='all',showInspector=false,range=fitRange(data.steps[index]);
    if(hash.has('lane')&&data.steps[index].lanes.some(l=>String(l.dev)===hash.get('lane')))selected=Number(hash.get('lane'));
    const overview=true;
    let offset=0;const timeline=data.steps.map(step=>{const entry={step,offset};offset+=step.end;return entry;});
    const allRange=[Math.min(0,...timeline.map(t=>t.offset+t.step.t0)),Math.max(...timeline.map(t=>t.offset+t.step.t1))];
    const request=data.run.request;
    const workerEvents=data.run.workerEvents||[],workerGroups=data.run.workerGroups||[];
    let pinnedWorker=null;
    const groupForStep=step=>workerGroups.find(g=>g.step===step.id||g.steps?.includes(step.id));
    if(request){allRange[0]=Math.min(allRange[0],request.start);allRange[1]=Math.max(allRange[1],request.end);}
    const periods=data.steps.filter(s=>/^D\d+$/.test(s.id)).map(s=>s.periodMs).sort((a,b)=>a-b);
    const quantile=p=>periods[Math.floor(p*(periods.length-1))],p90=quantile(.9),median=(periods[Math.floor((periods.length-1)/2)]+periods[Math.floor(periods.length/2)])/2;
    const slow=s=>/^D\d+$/.test(s.id)&&s.periodMs>p90;
    const slowText=s=>slow(s)?'Above Decode P90 · '+fmt(s.periodMs)+' · +'+fmt(s.periodMs-median)+' vs median':'';
    const fullRange=()=>allRange;
    const initial=timeline[index];
    range=hash.get('view')==='all'?[...allRange]:fitRange(initial.step).map(t=>t+initial.offset);
    root.classList.add('cps');
    root.innerHTML=`<header><h1 class="page-title">Inference Request</h1></header>
    <div class="workspace"><div class="canvas"><svg role="group" aria-label="Serving Host Device 共享时间轴泳道" tabindex="0"></svg></div></div>
    <div class="top-tools"><button data-inspect data-liquid-glass="icon" aria-expanded="false" aria-label="打开检查器" title="检查器">☷</button><button data-info data-liquid-glass="icon" aria-expanded="false" aria-label="运行信息与数据口径" title="运行信息与数据口径">ⓘ</button></div>
    <aside class="inspect-panel" data-liquid-glass="panel" hidden><button class="close" data-close-inspect aria-label="关闭检查器">×</button><div class="inspector"></div></aside>
    <section class="info-panel" data-liquid-glass="panel" hidden>
      <button class="close" data-close-info aria-label="关闭运行信息">×</button>
      <h2>读懂这张图</h2>
      <p>一条大模型推理请求，在服务程序、Host 和 16 个 NPU 之间的执行时间关系。</p>
      <h3>推理阶段</h3>
      <p><strong>Prefill · P</strong>：处理输入提示词，建立各层 KV Cache，为后续生成做准备。</p>
      <p><strong>Decode · D01–D52</strong>：利用 KV Cache 继续生成的连续执行轮次。D23 是第 23 轮，不是模型第 23 层；这份记录不能直接把一轮等同于一个最终输出 token。</p>
      <h3>执行层级与对象</h3>
      <p><strong>Serving</strong>：组织推理请求的服务程序，展示上层执行节奏。</p>
      <p><strong>Host</strong>：CPU 侧负责准备、下发、回收和协调设备工作的程序。</p>
      <p><strong>Device</strong>：承接计算任务的 NPU。本区显示 Host 记录的 runner_run 调用窗口，包含可能的同步等待，不等于 NPU 纯计算耗时。</p>
      <p><strong>lane / dev</strong>：lane 是 Host 侧执行通道，dev 是对应设备编号；本数据有 16 组对应关系，如 lane 7 → dev 7。</p>
      <p><strong>bind</strong>：进入 runner_run 前采集的 bind 总时间窗，未拆分内存分配与 H2D 搬运。</p>
      <h3>颜色与连线</h3>
      <div class="legend"><span><i></i>灰色：采集时间窗</span><span><i class="red"></i>红色：本轮最晚进入 runner_run 的路径</span><span><i class="white"></i>白色：悬停 / 固定选择</span><span>虚线：同 lane 调用衔接</span></div>
      <p>红色帮助定位哪条通道更晚准备好，是性能排查线索，不表示程序报错、设备故障，也不能单凭它确定瓶颈根因。红色行头表示该 lane 在至少一轮中最晚进入 runner_run。</p>
      <h3>运行信息与数据口径</h3>
      <p>DSpark Serving · GBS1 · 单请求 · 16 NPU。服务端 STRACE 共用 Host CLOCK_MONOTONIC；全局时间轴保留流水重叠。lane 间时长合计不等于墙钟时间。调用末端接近是观察结果，具体同步行为需设备事件验证。</p>
      <p>Serving 菱形为首次服务端输出处理，后续短刻度为输出处理事件；不代表客户端收到 token。导航红色色块表示周期超过本次 Decode P90 的 step；点击打开分析，查看偏差与执行线索。</p><p>准备 / 下发 / 回收：悬停双向高亮关联的 lane/dev，点击固定并查看阶段时间图。Worker 内部用 step_id 关联，跨层按时间匹配。</p><h3>操作</h3>
      <p>滚轮以光标为中心缩放，放大后逐渐显示路径连线与标注；拖动平移，悬停查看调用细节。底部导航 / ← → 定位 step，Fit 返回全局，点击行查看分析，Esc 关闭面板。</p>
      <div class="source-links"><a href="README.md">Pattern 契约</a><a href="provenance.json">来源指纹</a><a href="https://github.com/yinyucheng0601/PyPTO3/blob/55472831/Data/serving-strace-swimlane(2).json">原始 STRACE</a></div>
    </section>
    <div class="preview" role="tooltip" hidden></div>
    <div class="bottom-controls"><div class="navigator" aria-label="Execution step navigation"></div></div><nav class="toolbar" data-liquid-glass="control" aria-label="Canvas zoom"><button data-zoom="out" aria-label="Zoom out">−</button><button data-zoom="in" aria-label="Zoom in">＋</button><button data-fit>Fit</button></nav>`;
    const $=s=>root.querySelector(s),svg=$('svg');
    const stepName=s=>s.id==='P'?'Prefill':s.id==='DI'?'Draft Init':'Decode '+s.id.slice(1);
    const navigation=SequenceNavigation.mount($(".navigator"),{variant:"time",full:allRange,items:timeline.map(({step,offset})=>({label:stepName(step),start:offset,end:offset+step.end,alert:slow(step),detail:slowText(step)})),onSelect:i=>setStep(i),onFocus:i=>setStep(i,true),onRange:next=>moveRange(next)});
    const text=(parent,x,y,t,cls='')=>{const e=add(parent,'text',{x,y,class:cls});e.textContent=t;return e;};
    function add(parent,tag,attrs={}){const scale=svg.clientHeight/650;attrs={...attrs};for(const key of ['y','y1','y2','height'])if(typeof attrs[key]==='number')attrs[key]*=scale;if(attrs.points)attrs.points=attrs.points.split(' ').map(p=>{const [x,y]=p.split(',');return x+','+(Number(y)*scale);}).join(' ');const e=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));parent.append(e);return e;}
    let paintPinned=()=>{},analysisMode='lane';
    function bandTitle(y,label,left){
      const title=text(svg,44,y,label,'band-title');title.setAttribute('text-anchor','start');
      const centerY=y*svg.clientHeight/650-8;const badge=add(svg,'g',{transform:`translate(16,${centerY})`,'aria-hidden':'true'});add(badge,'circle',{r:15,fill:'#292929'});const icon=add(badge,'g',{transform:'translate(-8,-8) scale(.73)',fill:'none',stroke:'#c0c0c0','stroke-width':1.4});
      const paths={Serving:'M2 2H20V9H2Z M2 13H20V20H2Z M5 5H7 M5 16H7',Host:'M1 2H21V16H1Z M11 16V21 M6 21H16',Device:'M5 5H17V17H5Z M8 1V5 M14 1V5 M8 17V21 M14 17V21 M1 8H5 M1 14H5 M17 8H21 M17 14H21'};
      add(icon,'path',{d:paths[label]});
    }
    function sync(){if(!options.syncHash)return;const p=new URLSearchParams();p.set('view','timeline');p.set('step',data.steps[index].id);if(selected!=null)p.set('lane',selected);history.replaceState(null,'','#'+p);}
    function info(l){return `lane ${l.dev} → dev ${l.dev} · bind ${fmt(l.bindEnd-l.bindStart)} · 调用前空档 ${fmt(Math.max(0,l.rrStart-l.bindEnd))} · runner_run ${fmt(l.rrEnd-l.rrStart)}`;}
    function positionTip(clientX,clientY){
      const tip=$('.preview'),box=root.getBoundingClientRect(),gap=16,pad=12;
      let x=clientX-box.left+gap,y=clientY-box.top+gap;
      if(x+tip.offsetWidth>box.width-pad)x=clientX-box.left-tip.offsetWidth-gap;
      if(y+tip.offsetHeight>box.height-pad)y=clientY-box.top-tip.offsetHeight-gap;
      tip.style.left=Math.max(pad,Math.min(x,box.width-tip.offsetWidth-pad))+'px';
      tip.style.top=Math.max(pad,Math.min(y,box.height-tip.offsetHeight-pad))+'px';
    }
    function showTip(l,x,y,step=data.steps[index]){const tip=$('.preview');tip.textContent=`${stepName(step)} · Lane ${l.dev}\nbind          ${fmt(l.bindEnd-l.bindStart)}\n调用前空档    ${fmt(Math.max(0,l.rrStart-l.bindEnd))}\nrunner_run    ${fmt(l.rrEnd-l.rrStart)}`;tip.hidden=false;positionTip(x,y);}
    function inspector(){const s=data.steps[index],m=metrics(s),l=s.lanes.find(l=>l.dev===selected);const p=$('.inspector');$('.inspect-panel').hidden=!showInspector;p.hidden=!showInspector;if(!showInspector)return;p.replaceChildren();const cap=document.createElement('div');cap.className='eyebrow';cap.textContent=l?'PINNED LANE / '+s.id:'STEP / '+s.id;p.append(cap);const h=document.createElement('h2');h.textContent=l?`lane ${l.dev} → dev ${l.dev}`:'最晚进入 · lane '+m.critical.dev;p.append(h);const dl=document.createElement('dl');const values=l?[['bind 总窗',fmt(l.bindEnd-l.bindStart)],['调用前空档',fmt(Math.max(0,l.rrStart-l.bindEnd))],['runner_run 总窗',fmt(l.rrEnd-l.rrStart)],['进入时刻',fmt(l.rrStart)],['runner_run 结束',fmt(l.rrEnd)]]:[['进入前 · 关键 lane',fmt(m.critical.rrStart)],['至本步结束',fmt(s.end-m.critical.rrStart)],['16 lane bind 合计',fmt(m.bindSum)],['进入时刻极差',fmt(m.startSpread)],['结束时刻极差',fmt(m.endSpread)]];values.forEach(([k,v])=>{const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=k;dd.textContent=v;dl.append(dt,dd);});p.append(dl);const note=document.createElement('p');note.textContent='runner_run 包含调用与可能的同步等待。当前数据未独立采集 device_wall。';p.append(note);if(l){const b=document.createElement('button');b.textContent='取消固定';on(b,'click',()=>{selected=null;sync();render();});p.append(b);}}
    function render(){
      svg.onpointermove=null;svg.onclick=null;
      navigation.update({index,range});
      renderOverview();
    }
    function laneAnalysis(dev){
      pinnedWorker=null;analysisMode='lane';selected=dev;paintPinned();showInspector=true;$('.info-panel').hidden=true;$('[data-info]').setAttribute('aria-expanded','false');sync();
      const rows=data.steps.map(s=>{const l=s.lanes.find(l=>l.dev===dev),bind=l.bindEnd-l.bindStart;const sorted=s.lanes.map(l=>l.bindEnd-l.bindStart).sort((a,b)=>a-b);return {s,l,bind,median:(sorted[7]+sorted[8])/2};});
      const avg=key=>rows.reduce((a,r)=>a+key(r),0)/rows.length;
      const ranks=data.steps[0].lanes.map(l=>({dev:l.dev,bind:data.steps.reduce((a,s)=>{const x=s.lanes.find(x=>x.dev===l.dev);return a+x.bindEnd-x.bindStart;},0)/data.steps.length})).sort((a,b)=>b.bind-a.bind);
      const latest=rows.filter(r=>metrics(r.s).critical.dev===dev).length,mean=avg(r=>r.bind),med=avg(r=>r.median),peak=Math.max(...rows.map(r=>r.bind));
      const panel=$('.inspector');panel.hidden=false;$('.inspect-panel').hidden=false;$('[data-inspect]').setAttribute('aria-expanded','true');
      panel.innerHTML=`<div class="eyebrow">ALL STEPS / LANE ${dev}</div><h2>Host lane ${dev} · Bind</h2><dl><dt>平均 bind · 排名 ${ranks.findIndex(r=>r.dev===dev)+1} / 16</dt><dd>${fmt(mean)}</dd><dt>相对逐步中位均值</dt><dd>${(mean/med).toFixed(2)}×</dd><dt>最晚进入 runner_run</dt><dd>${latest} / ${rows.length} steps</dd><dt>runner_run 平均调用窗</dt><dd>${fmt(avg(r=>r.l.rrEnd-r.l.rrStart))}</dd></dl><h3>Bind across steps</h3><svg class="analysis-spark" viewBox="0 0 280 78" role="img" aria-label="lane bind 与每步中位趋势"><polyline points="${rows.map((r,i)=>`${i/(rows.length-1)*280},${74-r.median/peak*68}`).join(' ')}" fill="none" stroke="#aaa" stroke-dasharray="3 3"/><polyline points="${rows.map((r,i)=>`${i/(rows.length-1)*280},${74-r.bind/peak*68}`).join(' ')}" fill="none" stroke="#fd5644"/></svg><p>实线：lane ${dev}；虚线：每步 16 lane 中位。横轴 P → D52，纵轴 0–${fmt(peak)}。</p><h3>Bind ranking · mean</h3>${ranks.slice(0,5).map(r=>`<div class="analysis-rank"><span>lane ${r.dev}</span><meter min="0" max="${ranks[0].bind}" value="${r.bind}"></meter><span>${r.bind.toFixed(1)} ms</span></div>`).join('')}<h3>阶段分布</h3><p>${['P','DI','Decode'].map(stage=>{const rs=rows.filter(r=>stage==='Decode'?r.s.id.startsWith('D')&&r.s.id!=='DI':r.s.id===stage);return stage+'：'+rs.filter(r=>metrics(r.s).critical.dev===dev).length+'/'+rs.length+' steps 最晚进入';}).join('<br>')}</p><p>红色行头表示该 lane 至少一次最晚进入 runner_run，属于排查线索。bind 未拆分分配与 H2D；runner_run 不是纯设备执行时间。</p>`;
    }
    function stepAnalysis(){
      const step=data.steps[index],m=metrics(step),c=m.critical;
      const decode=data.steps.filter(s=>/^D\d+$/.test(s.id));
      const med=xs=>{const a=xs.sort((a,b)=>a-b),n=a.length;return (a[Math.floor((n-1)/2)]+a[Math.floor(n/2)])/2;};
      const base=med(decode.map(s=>s.periodMs));
      const baseline=get=>med(decode.map(s=>get(s.lanes.find(l=>l.dev===c.dev))));
      const bind=c.bindEnd-c.bindStart,bindBase=baseline(l=>l.bindEnd-l.bindStart);
      const runner=c.rrEnd-c.rrStart,runnerBase=baseline(l=>l.rrEnd-l.rrStart);
      const delta=(a,b)=>(a>=b?'+':'−')+fmt(Math.abs(a-b));
      const panel=$('.inspector');panel.hidden=false;$('.inspect-panel').hidden=false;$('[data-inspect]').setAttribute('aria-expanded','true');
      const comparison=(label,value,reference)=>{
        const max=Math.max(value,reference)*1.08;
        return `<div class="metric-comparison"><div><span>${label}</span><strong>${fmt(value)}</strong><small>${delta(value,reference)}</small></div><div class="comparison-track"><i style="width:${value/max*100}%"></i><b style="left:${reference/max*100}%" title="Decode 中位数 ${fmt(reference)}"></b></div></div>`;
      };
      const maxPeriod=Math.max(...decode.map(s=>s.periodMs))*1.08;
      const points=decode.map((s,i)=>`${i/(decode.length-1)*280},${70-s.periodMs/maxPeriod*60}`).join(' ');
      const chosen=decode.indexOf(step);
      const trend=`<svg class="analysis-spark" viewBox="0 0 280 78" role="img" aria-label="Decode 周期趋势，红点为选中步骤"><line x1="0" x2="280" y1="${70-p90/maxPeriod*60}" y2="${70-p90/maxPeriod*60}" stroke="#777" stroke-dasharray="3 3"/><polyline points="${points}" fill="none" stroke="#bbb" stroke-width="1.5"/>${chosen<0?'':`<circle cx="${chosen/(decode.length-1)*280}" cy="${70-step.periodMs/maxPeriod*60}" r="4" fill="#fd5644"/>`}</svg>`;
      const ranked=[...step.lanes].sort((a,b)=>(b.bindEnd-b.bindStart)-(a.bindEnd-a.bindStart));
      const maxBind=ranked[0].bindEnd-ranked[0].bindStart;
      panel.innerHTML=`<div class="eyebrow">STEP ANALYSIS / ${step.id}</div><h2>${stepName(step)} · ${slow(step)?'周期偏长':'执行分析'}</h2><h3>Decode 周期趋势</h3>${trend}<p class="chart-caption">D01 → D52 · 虚线 P90 ${fmt(p90)} · 红点为当前步骤</p>${comparison('本轮周期',step.periodMs,base)}<h3>最晚进入 · lane ${c.dev}</h3>${comparison('bind',bind,bindBase)}${comparison('runner_run 调用窗',runner,runnerBase)}<p class="chart-caption">实条：本轮；竖标：Decode 中位数。bind / runner_run 基线来自同 lane。</p><h3>本轮 Bind 排名</h3>${ranked.slice(0,5).map(l=>`<button class="step-lane-rank" data-analyze-lane="${l.dev}" aria-label="查看 lane ${l.dev} 跨 step 趋势"><span>lane ${l.dev}</span><span class="rank-track"><i style="width:${(l.bindEnd-l.bindStart)/maxBind*100}%;background:${l.dev===c.dev?'#fd5644':'#aaa'}"></i></span><span>${(l.bindEnd-l.bindStart).toFixed(1)} ms</span></button>`).join('')}<p class="chart-caption">点击排名查看该 lane 的跨 step 趋势。进入时刻极差 ${fmt(m.startSpread)}。以上为时间差异线索，尚不足以确认根因。</p>`;
      panel.querySelectorAll('[data-analyze-lane]').forEach(button=>on(button,'click',()=>laneAnalysis(Number(button.dataset.analyzeLane))));

      const b=document.createElement('button');b.textContent='聚焦本轮与 lane '+c.dev;on(b,'click',()=>{selected=c.dev;setStep(index,true);});panel.append(b);
      const trendButton=document.createElement('button');trendButton.textContent='查看 lane '+c.dev+' 跨 step 趋势';on(trendButton,'click',()=>laneAnalysis(c.dev));panel.append(trendButton);
    }
    function workerAnalysis(group){
      pinnedWorker=group.id;analysisMode='worker';showInspector=true;
      $('.info-panel').hidden=true;$('[data-info]').setAttribute('aria-expanded','false');
      const panel=$('.inspector');panel.hidden=false;$('.inspect-panel').hidden=false;$('[data-inspect]').setAttribute('aria-expanded','true');
      const events=workerEvents.filter(e=>group.eventIds.includes(e.id));
      const lo=Math.min(...events.map(e=>e.start)),hi=Math.max(...events.map(e=>e.end));
      const names={prepare:'准备',dispatch:'下发',reclaim:'回收'};
      panel.innerHTML=`<div class="eyebrow">WORKER / ${group.stepId??'PREFILL'}</div><h2>${group.step||group.steps?.join(' + ')||'未展示的轮次'} · 阶段关联</h2><svg viewBox="0 0 280 112" role="img" aria-label="Worker 三阶段的共享时间轴">${events.map(e=>{const y=18+['prepare','dispatch','reclaim'].indexOf(e.role)*32;return `<text x="0" y="${y}" fill="#bbb" font-size="12">${names[e.role]}</text><rect x="${45+(e.start-lo)/(hi-lo)*230}" y="${y-9}" width="${Math.max(1,(e.end-e.start)/(hi-lo)*230)}" height="7" fill="#ddd"/>`;}).join('')}</svg><p class="chart-caption">共享时间轴 · ${fmt(hi-lo)}。回收窗口包含等待，可与下一轮准备重叠。</p>${events.map(e=>`<div class="metric-comparison"><div><span>${names[e.role]}</span><strong>${fmt(e.end-e.start)}</strong></div></div>`).join('')}<p>Worker 三阶段通过同一 step_id 关联。与 lane / dev 按时间匹配${group.completionGap==null?'':`，回收结束晚于调用组结束 ${fmt(group.completionGap)}`}；原始记录没有跨层关联 ID，因此不作为已证实的因果链。</p>`;
    }
    function renderOverview(){
      const width=Math.max(300,$('.canvas').clientWidth),left=148,right=24,plot=width-left-right;
      svg.setAttribute('viewBox',`0 0 ${width} ${svg.clientHeight}`);svg.replaceChildren();
      const X=t=>left+(t-range[0])/(range[1]-range[0])*plot,H=d=>187+d*12,D=d=>413+d*12;
      const defs=add(svg,'defs'),clip=add(defs,'clipPath',{id:uid});add(clip,'rect',{x:left,y:0,width:plot,height:600});
      const g=add(svg,'g',{'clip-path':`url(#${uid})`});
      timeline.forEach(({step,offset},i)=>{if(i%2===0)for(const [y,height]of [[68,22],[116,52],[181,192],[407,192]])add(g,'rect',{x:X(offset),y,width:Math.max(0,X(offset+step.end)-X(offset)),height,fill:'#171717','pointer-events':'none'});});
      const pinned=[H,D].map(Y=>add(g,'rect',{x:left,y:selected==null?0:Y(selected)-6,width:plot,height:12,fill:'#fff',opacity:.055,'pointer-events':'none',visibility:selected==null?'hidden':'visible'}));
      const row=add(g,'rect',{x:left,y:0,width:plot,height:12,fill:'#fff',opacity:.10,'pointer-events':'none',visibility:'hidden'});
      const pair=add(g,'rect',{x:left,y:0,width:plot,height:12,fill:'#fff',opacity:.06,'pointer-events':'none',visibility:'hidden'});
      const col=add(g,'rect',{x:left,y:70,width:0,height:530,fill:'#fff',opacity:.08,'pointer-events':'none',visibility:'hidden'});
      [[65,'Serving'],[104,'Host'],[392,'Device']].forEach(([y,label])=>bandTitle(y,label,left));
      if(request){
        request.outputs.forEach((output,i)=>{
          const x=X(output.at);if(x<left-6||x>width-right+6)return;
          const mark=add(g,'g',{'data-output':i,tabindex:0,role:'img','aria-label':i===0?'First server output processing':'Server output processing '+(i+1)});
          add(mark,'rect',{x:x-6,y:80,width:12,height:16,fill:'transparent'});
          if(i===0){add(mark,'polygon',{points:`${x},82 ${x+4},87 ${x},92 ${x-4},87`,fill:'#eee'});if(x<width-160)text(mark,x+9,91,'First server output');}
          else add(mark,'line',{x1:x,x2:x,y1:84,y2:90,stroke:'#ccc','stroke-width':1.5});
          const detail=(i===0?'First server output processing':'Server output processing '+(i+1))+'\nFrom request start  '+fmt(output.at)+(output.gap==null?'':'\nSince previous output  '+fmt(output.gap));
          const show=e=>{const tip=$('.preview');tip.textContent=detail;tip.hidden=false;const b=mark.getBoundingClientRect();positionTip(e?.clientX??b.left,e?.clientY??b.top);};
          on(mark,'pointerenter',show);on(mark,'focus',()=>show());on(mark,'blur',()=>$('.preview').hidden=true);
        });
      }
      const workerMarks=[];
      ['prepare','dispatch','reclaim'].forEach((role,ri)=>{
        const y=122+ri*18;text(svg,44,y+4,['准备','下发','回收'][ri]).setAttribute('text-anchor','start');
        workerEvents.filter(e=>e.role===role).forEach(event=>{
          const group=workerGroups.find(g=>g.eventIds.includes(event.id));
          const mark=add(g,'g',{'data-worker':event.id,tabindex:0,role:'button','aria-label':event.name+' '+(event.stepId??'Prefill')});
          add(mark,'rect',{x:X(event.start),y:y-7,width:Math.max(4,X(event.end)-X(event.start)),height:14,fill:'transparent'});
          const bar=add(mark,'rect',{x:X(event.start),y:y-3,width:Math.max(1,X(event.end)-X(event.start)),height:6,fill:'#888'});
          workerMarks.push({bar,group});mark.style.cursor='pointer';
          const preview=e=>{linked(group);const tip=$('.preview');tip.textContent=event.name+'\nWorker step '+(event.stepId??'Prefill')+' · '+fmt(event.end-event.start)+'\n'+(group?.step||group?.steps?.join(' + ')||'未展示的轮次')+' · 时间匹配';tip.hidden=false;const b=mark.getBoundingClientRect();positionTip(e?.clientX??b.left,e?.clientY??b.top);};
          on(mark,'pointerenter',preview);on(mark,'focus',()=>preview());on(mark,'blur',()=>{linked(workerGroups.find(g=>g.id===pinnedWorker));$('.preview').hidden=true;});
          const select=()=>{if(group){workerAnalysis(group);render();}};
          on(mark,'click',e=>{e.stopPropagation();select();});on(mark,'keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();e.stopPropagation();select();}});
        });
      });
      const counts=new Map();data.steps.forEach(s=>{const dev=metrics(s).critical.dev;counts.set(dev,(counts.get(dev)||0)+1);});const flagged=new Set(counts.keys());
      const highlight=(dev,entry,band)=>{const scale=svg.clientHeight/650;for(const r of [row,pair])r.setAttribute('visibility',dev==null?'hidden':'visible');if(dev!=null){row.setAttribute('y',((band==='runner'?D(dev):H(dev))-6)*scale);pair.setAttribute('y',((band==='runner'?H(dev):D(dev))-6)*scale);}col.setAttribute('visibility',entry?'visible':'hidden');if(entry){const x=X(entry.offset);col.setAttribute('x',x);col.setAttribute('width',Math.max(2,X(entry.offset+entry.step.end)-x));}};
      const detail=add(g,'g',{'pointer-events':'none'});
      const linkedLayer=add(g,'g',{'pointer-events':'none'});
      function linked(group,dev=null){
        linkedLayer.replaceChildren();
        // Worker events identify a group, not an entire time-column selection.
        // Clear stale crosshairs; lane hover owns its own row/column highlight.
        if(dev==null){highlight(null,null);if(group)detail.replaceChildren();}
        workerMarks.forEach(m=>m.bar.setAttribute('fill',group&&m.group?.id===group.id?'#eee':'#888'));
        if(!group)return;
        const entries=timeline.filter(t=>t.step.id===group.step||group.steps?.includes(t.step.id));
        entries.forEach(entry=>{entry.step.lanes.filter(l=>dev==null||l.dev===dev).forEach(l=>{
          for(const [a,b,y]of [[l.bindStart,l.bindEnd,H(l.dev)],[l.rrStart,l.rrEnd,D(l.dev)]])add(linkedLayer,'line',{x1:X(entry.offset+a),x2:X(entry.offset+b),y1:y,y2:y,stroke:l.dev===metrics(entry.step).critical.dev?'#fd5644':'#eee','stroke-width':2.4});
        });});
      }
      function connections(entry,dev=null){
        detail.replaceChildren();if(!entry)return;
        const {step,offset}=entry,m=metrics(step);
        const pixels=(step.end/(range[1]-range[0]))*plot;
        const opacity=Math.max(0,Math.min(1,(pixels-65)/180));
        if(!opacity)return;detail.setAttribute('opacity',opacity);
        for(const l of step.lanes){
          const hot=l.dev===m.critical.dev,focus=l.dev===(dev??selected);
          if(!hot&&!focus)continue;
          const cls=focus?'paired':'critical';
          add(detail,'polyline',{points:`${X(offset+l.bindEnd)},${H(l.dev)} ${X(offset+l.rrStart)},${D(l.dev)}`,class:cls,'stroke-dasharray':'3 4'});
          for(const [a,b,y]of [[l.bindStart,l.bindEnd,H(l.dev)],[l.rrStart,l.rrEnd,D(l.dev)]])add(detail,'line',{x1:X(offset+a),x2:X(offset+b),y1:y,y2:y,class:cls});
          if(hot&&pixels>180)text(detail,Math.max(left+6,X(offset+l.bindStart)+6),H(l.dev)-9,`Latest entry · bind ${fmt(l.bindEnd-l.bindStart)}`,'label-hot');
        }
      }
      const rowHeaders=[];
      text(svg,44,H(0)+4,"lane");text(svg,44,D(0)+4,"dev");
      for(let dev=0;dev<16;dev++)for(const [y,band]of [[H(dev),'bind'],[D(dev),'runner']]){
        const hot=flagged.has(dev);const label=text(svg,left-12,y+4,String(dev),hot?'label-hot':'');label.setAttribute('text-anchor','end');label.style.cursor='pointer';label.setAttribute('tabindex','0');label.setAttribute('role','button');label.setAttribute('aria-label',`Analyze lane ${dev}${hot?' · latest entry observed':''}`);label.dataset.lane=dev;rowHeaders.push(label);const labelTitle=add(label,'title');labelTitle.textContent='Analyze lane '+dev+(hot?' · latest entry '+counts.get(dev)+' / '+data.steps.length+' steps':'');
        on(label,'pointerenter',()=>{highlight(dev,null,band);$('.preview').hidden=true;});on(label,'focus',()=>highlight(dev,null,band));on(label,'click',()=>laneAnalysis(dev));on(label,'keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();laneAnalysis(dev);}});
      }
      paintPinned=()=>{const scale=svg.clientHeight/650;pinned.forEach((r,i)=>{r.setAttribute('visibility',selected==null?'hidden':'visible');if(selected!=null)r.setAttribute('y',((i===0?H(selected):D(selected))-6)*scale);});rowHeaders.forEach(h=>h.style.textDecoration=Number(h.dataset.lane)===selected?'underline':'none');};paintPinned();
      timeline.forEach((entry,i)=>{
        const {step,offset}=entry;if(offset+step.end<range[0]||offset+step.t0>range[1])return;
        const m=metrics(step),x=X(offset),w=X(offset+step.end)-x;
        const lines=add(g,'g',{'pointer-events':'none'});
        step.serving.filter(e=>e.kind==='own').forEach(e=>add(lines,'rect',{x:X(offset+e.s),y:77,width:Math.max(1,X(offset+e.s+e.d)-X(offset+e.s)),height:5,class:'bar'}));
        step.lanes.forEach(l=>{const hot=l.dev===m.critical.dev;[[l.bindStart,l.bindEnd,H(l.dev)],[l.rrStart,l.rrEnd,D(l.dev)]].forEach(([a,b,y])=>add(lines,'line',{x1:X(offset+a),x2:X(offset+b),y1:y,y2:y,stroke:hot?'#fd5644':'#888','stroke-width':hot?2:1.4}));});
        const stride=Math.max(1,Math.ceil(42/Math.max(1,w)));const show=i===0||i===data.steps.length-1||(i>=2&&(i-2)%stride===0);if(show){const label=text(lines,x+Math.max(2,w/2),22,step.id);label.setAttribute('text-anchor','middle');}add(lines,'line',{x1:x+w/2,x2:x+w/2,y1:29,y2:34,stroke:'#555'});
        const head=add(g,'rect',{x,y:4,width:Math.max(2,w),height:52,fill:'transparent',tabindex:0,role:'button','aria-label':'Open '+stepName(step)});
        const headerTip=e=>{highlight(null,entry);const tip=$('.preview');tip.textContent=stepName(step)+'\n'+fmt(step.periodMs)+(slow(step)?'\n'+slowText(step):'')+'\nClick to select · Double-click to focus';tip.hidden=false;positionTip(e.clientX,e.clientY);};on(head,'pointerenter',headerTip);on(head,'focus',()=>highlight(null,entry));on(head,'click',()=>setStep(i));on(head,'keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();setStep(i);}});
      });
      const locate=e=>{const b=svg.getBoundingClientRect(),x=e.clientX-b.left,y=(e.clientY-b.top)/b.height*650;if(x<left||x>width-right)return null;const t=range[0]+(x-left)/plot*(range[1]-range[0]);const band=y>=407&&y<599?'runner':y>=181&&y<373?'bind':null;const dev=band?Math.round((y-(band==='bind'?187:413))/12):null;const entry=timeline.find(ti=>{const l=ti.step.lanes.find(l=>l.dev===dev);return l&&t>=ti.offset+(band==='bind'?l.bindStart:l.rrStart)&&t<=ti.offset+(band==='bind'?l.bindEnd:l.rrEnd);})||timeline.find(ti=>t>=ti.offset&&t<ti.offset+ti.step.end);return {entry,band,dev};};
      svg.onpointermove=e=>{if(e.target.closest('[data-output], [data-worker]'))return;if(!overview)return;const at=locate(e);if(!at){if(!rowHeaders.includes(e.target))highlight(null,null);return;}if(at.dev!=null&&at.entry){highlight(at.dev,at.entry,at.band);connections(at.entry,at.dev);linked(groupForStep(at.entry.step),at.dev);const l=at.entry.step.lanes.find(l=>l.dev===at.dev);showTip(l,e.clientX,e.clientY,at.entry.step);}else if(at.band){highlight(at.dev,null,at.band);}else if((e.clientY-svg.getBoundingClientRect().top)/svg.clientHeight*650>56){highlight(null,null);linked(workerGroups.find(g=>g.id===pinnedWorker));$('.preview').hidden=true;}};
      svg.onclick=e=>{if(suppressClick){suppressClick=false;return;}if(!overview)return;const at=locate(e);if(at?.dev!=null){if(at.entry)index=timeline.indexOf(at.entry);laneAnalysis(at.dev);}};
      g.append(linkedLayer);g.append(detail);connections(timeline[index]);linked(workerGroups.find(g=>g.id===pinnedWorker));
      const span=range[1]-range[0],tick=Math.pow(10,Math.floor(Math.log10(span)))/2;
      for(let t=Math.ceil(range[0]/tick)*tick;t<=range[1];t+=tick){const x=X(t);if(x<left||x>width-right)continue;add(svg,'line',{x1:x,x2:x,y1:605,y2:612,stroke:'#858585'});text(svg,x,634,span<1800?Math.round(t)+' ms':(t/1000).toFixed(1)+' s').setAttribute('text-anchor','middle');}
      if(showInspector&&analysisMode==='worker'&&pinnedWorker)workerAnalysis(workerGroups.find(g=>g.id===pinnedWorker));else if(showInspector&&analysisMode==='step')stepAnalysis();else if(showInspector&&selected!=null)laneAnalysis(selected);else{$('.inspect-panel').hidden=true;$('[data-inspect]').setAttribute('aria-expanded','false');}
    }
    const uid='cps-clip-'+Math.random().toString(36).slice(2);
    let animation=0;
    function moveRange(target,animate=false){
      cancelAnimationFrame(animation);
      if(!animate||matchMedia('(prefers-reduced-motion: reduce)').matches){range=target;render();return;}
      const from=[...range],start=performance.now();
      const frame=now=>{const t=Math.min(1,(now-start)/360),ease=1-Math.pow(1-t,3);range=from.map((v,i)=>v+(target[i]-v)*ease);render();if(t<1)animation=requestAnimationFrame(frame);};
      animation=requestAnimationFrame(frame);
    }
    function setStep(i,focus=false){pinnedWorker=null;if(analysisMode==='worker')analysisMode='step';index=Math.max(0,Math.min(data.steps.length-1,i));hover=null;if(slow(data.steps[index])){analysisMode='step';showInspector=true;$('.info-panel').hidden=true;$('[data-info]').setAttribute('aria-expanded','false');}$('.preview').hidden=true;const entry=timeline[index],local=fitRange(entry.step).map(t=>t+entry.offset);if(focus)moveRange(boundedRange(...local,allRange),true);else if(entry.offset+entry.step.end<=range[0]||entry.offset>=range[1]){const w=range[1]-range[0],center=entry.offset+entry.step.end/2;moveRange(boundedRange(center-w/2,center+w/2,allRange),true);}else render();sync();}
    function zoom(f,anchor=.5){cancelAnimationFrame(animation);const w=range[1]-range[0],at=range[0]+w*anchor;moveRange(boundedRange(at-w*f*anchor,at+w*f*(1-anchor),allRange));}
    on($('[data-fit]'),'click',()=>moveRange([...allRange],true));
    root.querySelectorAll('[data-zoom]').forEach(b=>on(b,'click',()=>zoom(b.dataset.zoom==='in'?.7:1.4)));
    on($('[data-inspect]'),'click',()=>{showInspector=!showInspector;$('.info-panel').hidden=true;$('[data-info]').setAttribute('aria-expanded','false');render();});
    on($('[data-close-inspect]'),'click',()=>{pinnedWorker=null;showInspector=false;render();$('[data-inspect]').focus();});
    const toggleInfo=()=>{const panel=$('.info-panel');panel.hidden=!panel.hidden;$('[data-info]').setAttribute('aria-expanded',String(!panel.hidden));if(!panel.hidden){showInspector=false;render();}};
    on($('[data-info]'),'click',toggleInfo);on($('[data-close-info]'),'click',()=>{toggleInfo();$('[data-info]').focus();});
    on(svg,'pointerleave',()=>{hover=null;$('.preview').hidden=true;render();});
    on(svg,'pointermove',e=>{if(!$('.preview').hidden)positionTip(e.clientX,e.clientY);});
    svg.addEventListener('wheel',e=>{e.preventDefault();const box=svg.getBoundingClientRect();zoom(Math.exp(Math.max(-100,Math.min(100,e.deltaY))*.003),Math.max(0,Math.min(1,(e.clientX-box.left-148)/(box.width-172))));},{passive:false,signal:abort.signal});
    let drag=null,suppressClick=false;on(svg,'pointerdown',e=>{if(e.target.closest('.row-hit, [data-worker]'))return;cancelAnimationFrame(animation);suppressClick=false;drag={x:e.clientX,range:[...range]};svg.setPointerCapture(e.pointerId);});on(svg,'pointermove',e=>{if(!drag)return;if(Math.abs(drag.x-e.clientX)>4)suppressClick=true;const shift=(drag.x-e.clientX)/(svg.clientWidth-172)*(drag.range[1]-drag.range[0]);range=boundedRange(drag.range[0]+shift,drag.range[1]+shift,fullRange());render();});on(svg,'pointerup',()=>drag=null);on(svg,'pointercancel',()=>drag=null);
    on(root,'keydown',e=>{if(['ArrowLeft','ArrowRight','ArrowDown','ArrowUp'].includes(e.key)){e.preventDefault();const inNav=e.target.closest('.navigator');setStep(index+(['ArrowRight','ArrowDown'].includes(e.key)?1:-1));if(inNav)$('.navigator').children[index].focus();}if(e.key==='Escape'){pinnedWorker=null;showInspector=false;$('.info-panel').hidden=true;$('[data-info]').setAttribute('aria-expanded','false');$('.preview').hidden=true;render();}});
    const observer=new ResizeObserver(()=>render());observer.observe(root);sync();render();
    return {setStep:id=>{const i=data.steps.findIndex(s=>s.id===id);if(i<0)throw new Error('Unknown step '+id);setStep(i);},getState:()=>({step:data.steps[index].id,selected,layer,view:'timeline',range:[...range],showInspector}),destroy:()=>{cancelAnimationFrame(animation);observer.disconnect();navigation.destroy();abort.abort();root.replaceChildren();root.classList.remove('cps');}};
  }
  const api={mount,metrics,fitRange,boundedRange};if(typeof module!=='undefined')module.exports=api;global.CriticalPathSwimlane=api;
})(typeof window!=='undefined'?window:globalThis);
