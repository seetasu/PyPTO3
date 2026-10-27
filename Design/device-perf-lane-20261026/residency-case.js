window.RESIDENCY_CASE = (() => {
  const e = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const num = n => n.toLocaleString('en-US');
  const time = n => n.toFixed(2) + ' μs';
  const mib = n => (n / 1048576).toFixed(3) + ' MiB';
  const steps = ['泳道上的问题', '重复搬运', '定位到 Pass', '修改与安全条件', '验证与收益'];
  let root, sceneRoot, data, state, onTask;
  const current = () => data.cases.find(c => c.id === state.caseId);
  const tag = (text, mock = false) => '<span class="rc-tag' + (mock ? ' mock' : '') + '">' + text + '</span>';
  const action = (step, text) => '<button type="button" class="ctl rc-next" data-rc-step="' + step + '">' + text + ' →</button>';
  function metrics(c) {
    return '<div class="rc-metrics">' + [[time(c.before), '修改前 · 独立 QK'], [time(c.after), '修改后 · 独立 QK'], [c.reported.toFixed(1) + '%', '作者报告的耗时下降'], [num((c.trips - 1) * c.panel) + ' B', '固定面板减少搬运量' + (data.traffic[c.id] ? ' · 已报告' : ' · 按次数推导')]].map(([v,l]) => '<div><strong>' + v + '</strong><small>' + l + '</small></div>').join('') + '</div>';
  }
  function lanes(c) {
    const max = c.before + 20, left = n => (n / max * 100).toFixed(4);
    const row = (after) => {
      const dur = after ? c.after : c.before;
      return '<div class="rc-lane-row"><span>' + (after ? '修改后' : '修改前') + '</span><div class="rc-track"><button class="rc-context" style="left:0;width:' + left(6) + '%" data-rc-task="prepare" data-rc-arm="' + (after ? 'after' : 'before') + '" title="准备任务：模拟 6 μs">准备</button><button class="rc-qk' + (after ? ' after' : '') + (state.task === 'qk' && state.arm === (after ? 'after' : 'before') ? ' selected' : '') + '" style="left:' + left(8) + '%;width:' + left(dur) + '%" data-rc-task="qk" data-rc-arm="' + (after ? 'after' : 'before') + '" aria-pressed="' + (state.task === 'qk' && state.arm === (after ? 'after' : 'before')) + '">QK · ' + time(dur) + '</button><button class="rc-context rc-consumer" style="left:' + left(8+dur) + '%;width:' + left(8) + '%" data-rc-task="consumer" data-rc-arm="' + (after ? 'after' : 'before') + '" title="依赖消费者：模拟 8 μs">后继</button></div></div>';
    };
    return '<section class="rc-lanes"><div class="rc-section-head"><h3>同一时间尺度 · 任务泳道对照</h3>' + tag('任务上下文 / 起点：Mock', true) + '</div><div class="rc-axis"><span>0</span><span>' + time(max) + '</span></div>' + row(false) + row(true) + '<p class="rc-note">QK 条宽取 PR 公开计时；准备、后继、起点及单链依赖为模拟上下文，不属于原 DeepSeek trace，也不能用于计算整模型收益。点击 QK 进入分析。</p></section>';
  }
  function iteration(c) {
    return '<div class="rc-iteration"><label for="rc-iteration">当前观测轮次 <b>' + (state.iter + 1) + ' / ' + c.trips + '</b></label><input id="rc-iteration" data-rc-iteration type="range" min="0" max="' + (c.trips-1) + '" value="' + state.iter + '"><span>固定面板 A · GM 偏移不变</span></div>';
  }
  // An intentionally simple serial illustration calibrated to the two published endpoints.
  // It preserves streamed/compute event durations across arms, not real pipeline overlap.
  function events(c, after) {
    const stationary = (c.before - c.after) / (c.trips - 1), cycle = c.before / c.trips;
    const streamed = (cycle - stationary) * 0.4, compute = (cycle - stationary) * 0.6;
    const out = [];
    if (after) out.push({ type: 'stationary', iter: 0, start: 0, duration: stationary });
    for (let i=0;i<c.trips;i++) {
      let start = after ? stationary + i*(streamed+compute) : i*cycle;
      if (!after) { out.push({type:'stationary', iter:i, start, duration:stationary}); start += stationary; }
      out.push({type:'streamed',iter:i,start,duration:streamed}); start += streamed;
      out.push({type:'compute',iter:i,start,duration:compute});
    }
    return out;
  }
  function micro(c) {
    const width = n => (n/c.before*100).toFixed(5);
    return '<div class="rc-section-head"><h3>搬运与计算展开</h3>' + tag('逐事件时长与排布：Mock', true) + '</div><div class="rc-micro-scroll"><div class="rc-micro">' + [false,true].map(after => '<section><h4>' + (after?'修改后：先加载一次，再复用':'修改前：每轮重复加载') + '</h4>' + [['stationary','MTE2 · 固定 A'],['streamed','MTE2 · 流动 B'],['compute','CUBE · 计算']].map(([type,label]) => '<div class="rc-micro-row"><span>' + label + '</span><div>' + events(c,after).filter(v=>v.type===type).map(v=>'<button type="button" class="rc-event ' + type + (state.object===({stationary:'fixed',streamed:'streamed',compute:'compute'})[type] && (v.iter===state.iter || after&&type==='stationary')?' selected':'') + '" style="left:' + width(v.start) + '%;width:' + width(v.duration) + '%" data-rc-event="' + type + '" data-rc-iter="' + v.iter + '" title="' + (after&&type==='stationary'?'循环前唯一加载':'第 '+(v.iter+1)+' 轮 · '+label) + ' · 模拟位置" aria-label="' + (after?'修改后':'修改前') + ' ' + label + ' 第 ' + (v.iter+1) + ' 轮" aria-pressed="' + (state.object===({stationary:'fixed',streamed:'streamed',compute:'compute'})[type] && (v.iter===state.iter || after&&type==='stationary')) + '"></button>').join('') + '</div></div>').join('') + '</section>').join('') + '</div></div><p class="rc-note">为解释次数差异，将事件串行排布并校准到公开 QK 总时长；流动搬运与计算时长按示意比例拆分。图不复原实际重叠，不用于证明每条 DMA 的耗时。</p>';
  }
  function flow(c) {
    return '<div class="rc-flow"><article><h4>修改前 · 第 ' + (state.iter+1) + ' 轮</h4><div class="rc-box stationary">GM · 固定 A<br><b>' + c.panel/1024 + ' KiB · 同一内容</b></div><div class="rc-arrow">↓ 又搬一次</div><div class="rc-box stationary">Mat / L1 · A_panel</div><div class="rc-arrow">↓ 当前 K 切片</div><div class="rc-box">Left / L0 → matmul</div></article><article><h4>修改后 · 第 ' + (state.iter+1) + ' 轮</h4><div class="rc-box resident">循环前 GM → Mat / L1<br><b>只加载一次 A_panel</b></div><div class="rc-arrow">↓ 第 ' + (state.iter+1) + ' 轮复用驻留面板</div><div class="rc-box resident">Mat / L1 · 保持驻留</div><div class="rc-arrow">↓ 当前 K 切片 · 保留</div><div class="rc-box">Left / L0 → matmul</div></article></div>';
  }
  function loopCells(c, after) {
    return '<div class="rc-cells">' + Array.from({length:c.trips},(_,i)=>'<button type="button" class="' + (after?'resident':'stationary') + (i===state.iter?' selected':'') + '" data-rc-iter="' + i + '" aria-pressed="' + (i===state.iter) + '" title="第 '+(i+1)+' 轮：'+(after?'复用驻留 A':'再次从 GM 加载 A')+'">' + (i+1) + '<small>' + (after?'复用':'加载') + '</small></button>').join('') + '</div>';
  }
  function transfer(c) {
    const before=c.panel*c.trips;
    const observed = {fixed:'固定 A · GM 偏移不变，修改前每轮加载、修改后驻留复用。',streamed:'流动 B · 偏移随 n 改变，两种版本都必须逐轮搬运。',compute:'CUBE 计算 · 模拟事件时长保持一致；公开报告 Matrix / FIXPIPE 不变。'}[state.object] || '固定 A · 选择同一轮的两侧搬运观察。';
    return '<div class="rc-inspector"><small>当前选择 · 第 ' + (state.iter+1) + ' 轮</small><p>' + observed + '</p>' + action(2,'联动到这条操作的 Pass 前后位置') + '</div><div class="rc-two"><article class="rc-card"><h4>固定 A · ' + c.trips + ' 次 GM → L1</h4>' + loopCells(c,false) + '<strong>' + mib(before) + '</strong></article><article class="rc-card"><h4>固定 A · 1 次加载 + ' + (c.trips-1) + ' 次复用</h4>' + loopCells(c,true) + '<strong>' + mib(c.panel) + '</strong></article></div><div class="rc-verdict">省下 (' + c.trips + ' − 1) × ' + c.panel/1024 + ' KiB = <b>' + num(before-c.panel) + ' B</b>；流动 B 的流量不变。</div>' + action(2,'这些重复加载由哪个 Pass 产生，又在哪里消除');
  }
  const codeBefore = [
    ['loop','for n in range(T):'], ['fixed','    A_panel = tile.load(A, ..., Mat)'], ['streamed','    B_panel = tile.load(B, [n, ...], Mat)'], ['loop','    for k in pipeline(...):'], ['extract','        A_k = tile.extract(A_panel, [k, ...], Left)'], ['extract','        B_k = tile.extract(B_panel, [k, ...], Right)'], ['compute','        C = tile.matmul(A_k, B_k)'],
  ];
  const codeAfter = [
    ['fixed','A_panel = tile.load(A, ..., Mat)  # 一次'], ['loop','for n in range(T):'], ['streamed','    B_panel = tile.load(B, [n, ...], Mat)'], ['loop','    for k in pipeline(...):'], ['extract','        A_k = tile.extract(A_panel, [k, ...], Left)'], ['extract','        B_k = tile.extract(B_panel, [k, ...], Right)'], ['compute','        C = tile.matmul(A_k, B_k)'],
  ];
  function code(rows,title) {
    return '<article class="rc-card rc-code"><h4>' + title + '</h4><small>语义示意代码 · 非逐行原始 dump</small><div role="group" aria-label="'+title+'">' + rows.map(([id,text],i)=>'<button class="' + (id===state.object?'selected':'') + '" type="button" data-rc-object="' + id + '" aria-pressed="' + (id===state.object) + '"><em>' + (i+1) + '</em><code>' + e(text) + '</code></button>').join('') + '</div></article>';
  }
  function pass(c) {
    const facts = {fixed:['固定 A 的 tile.load','原本在外层循环内，执行 '+c.trips+' 次。外提至循环前后仅执行 1 次；Mat 生命周期延长。'],streamed:['流动 B 的 tile.load','依赖 n 的偏移，必须逐轮更新；位置与搬运次数保持。'],extract:['K 相关的 tile.extract','仍在 K 流水内部，Left/Right staging 与 ping-pong 保留；不会将整个 L0 生命周期一起外提。'],compute:['tile.matmul / matmul_acc','数学计算保持；PR 报告 Matrix / FIXPIPE 不变。'],loop:['外层循环与 K 流水','外层复用固定面板，内层仍按 K 切片计算；这两个层次分别处理。']};
    const f=facts[state.object];
    return '<div class="rc-pass-chain"><span>ConvertTensorToTileOps<small>生成桥接 load<br>标记编译器来源</small></span><b>→</b><span>AutoTileMatmulL0<small>处理 K 分块<br>不负责外层驻留</small></span><b>→</b><strong>InferTileMemorySpace<small>明确 Mat / Left / Right<br>接入驻留分析与外提</small></strong></div><div class="rc-two">' + code(codeBefore,'基线编译器 · 推断后仍在循环内') + code(codeAfter,'PR #2080 · 推断后已外提') + '</div><div class="rc-inspector"><small>当前观测对象 · 点击两侧代码联动</small><h4>' + f[0] + '</h4><p>' + f[1] + '</p></div>' + flow(c) + '<p class="rc-note">根因是缺少外层不变量驻留优化。PR 将分析接入 InferTileMemorySpace 的返回路径；没有把 AutoTile 的正常职责边界当作故障。</p>' + action(3,'查看外提的安全条件与修改位置');
  }
  function safety() {
    const ok=state.gate==='safe';
    const scenarios=[['safe','编译器自有、只读固定面板'],['external','外部输入，无法证明不别名'],['write','循环内可能写入源存储']];
    return '<div class="rc-section-head"><h3>为什么可以把这条 load 移出循环？</h3>' + tag('规则演示 · 非新增实测',true) + '</div><div class="rc-gate-options">' + scenarios.map(([id,label])=>'<button type="button" class="ctl '+(state.gate===id?'on':'')+'" data-rc-gate="'+id+'" aria-pressed="'+(state.gate===id)+'">'+label+'</button>').join('') + '</div><div class="rc-safety"><div class="rc-checks">' + [['偏移、形状和依赖不随循环变化',true],['顺序、静态、非空循环；无副作用边界',true],['存储来自可分析的 tensor.create',state.gate!=='external'],['只读，且不与可写实际参数别名',ok],['延长 Mat 生命周期仍满足容量',true]].map(([label,pass])=>'<div class="'+(pass?'':'fail')+'"><b>'+ (pass?'✓':'?')+'</b>'+label+'</div>').join('')+'</div><article class="rc-decision '+(ok?'':'decline')+'"><small>分析结论</small><h3>'+(ok?'允许外提':'拒绝外提')+'</h3><p>'+(ok?'循环前加载 Mat 面板一次；K 相关的 L0 提取保持原位。':'保留循环内原始 load；不能仅凭参数名不同推断 noalias。')+'</p></article></div><div class="rc-two"><article class="rc-card"><h4>修改 1 · 来源标记</h4><code>convert_tensor_to_tile_ops_pass.cpp</code><p>仅标记编译器生成的 tensor → Mat 桥接加载，显式手写 load 不自动移动。</p></article><article class="rc-card"><h4>修改 2 · 驻留分析</h4><code>InferTileMemorySpace → residency::Apply</code><p>在内存空间确定后分析使用链、别名、顺序与容量，再执行外提。</p><a href="'+data.integration+'" target="_blank" rel="noopener">查看接入代码 ↗</a> · <a href="'+data.implementation+'" target="_blank" rel="noopener">查看驻留分析实现 ↗</a></article></div><p class="rc-note">规则场景用于解释拒绝条件，不会改写上方已发布的性能结果；这些条件属于保守优化，尚非通用外部输入驻留契约。</p>' + action(4,'核对结构、流量、计时与正确性');
  }
  function validation(c) {
    return '<div class="rc-validation"><article class="rc-card"><small>结构验证</small><h4>加载外提，K 流水保留</h4><p>固定 A：'+c.trips+' → 1 次；流动 B、K 提取、matmul 留在循环内。</p></article><article class="rc-card"><small>流量验证</small><h4>'+num((c.trips-1)*c.panel)+' B</h4><p>'+(data.traffic[c.id]?'PR 模拟器确认的 MTE2 减少量':'按 (T−1) × 面板大小推导；PR 未列此项逐字节测量')+'。Matrix / FIXPIPE 保持。</p></article><article class="rc-card"><small>设备计时</small><h4>'+time(c.before)+' → '+time(c.after)+'</h4><p>两台 910B2 同向；PR/base 比值的 95% CI：['+c.ci.join(', ')+']。</p></article><article class="rc-card"><small>正确性与回归</small><h4>作者报告数值验证通过</h4><p>BF16 输入 / FP32 累积与输出；覆盖 RHS 转置控制组。89 项空间推断、2,285 项变换、582 项 codegen 测试通过。</p></article></div><div class="rc-table-wrap"><table class="rc-table"><caption>PR 公开的独立 QK 设备计时 · 点击行切换整个案例</caption><thead><tr><th>描述符</th><th>修改前</th><th>修改后</th><th>耗时下降</th></tr></thead><tbody>'+data.cases.map(v=>'<tr class="'+(v.id===c.id?'selected':'')+'"><th><button type="button" data-rc-case="'+v.id+'" aria-pressed="'+(v.id===c.id)+'">'+v.label+'</button></th><td>'+time(v.before)+'</td><td>'+time(v.after)+'</td><td>'+v.reported.toFixed(1)+'%</td></tr>').join('')+'</tbody></table></div><div class="rc-inspector"><h4>反证与边界</h4><p>K1 在 T=1、T=8 时基本持平，符合无复用 / 小面板低次数的预期。这里的收益是独立 QK，不是完整模型的 15.7% 加速；模拟泳道后继提前仅用于展示依赖传播。</p></div>'+action(0,'回到泳道看这次改动的落点');
  }
  function finding(c) {
    if(state.task!=='qk') return '<div class="rc-inspector"><h4>'+ (state.task==='prepare'?'准备任务':'后继消费者')+' · 模拟上下文</h4><p>'+ (state.task==='prepare'?'两条泳道中保持相同的 6 μs 准备段，不属于 PR 的测量。':'模拟为紧依赖 QK，随 QK 结束提前 '+time(c.before-c.after)+'；实际端到端收益还取决于其他依赖与重叠。')+'</p><button class="ctl" type="button" data-rc-task="qk">选中 QK 分析根因</button></div>';
    return '<div class="rc-finding"><div><small>选中任务 · 独立 QK</small><h3>同一块固定矩阵，被重复送进 L1</h3><p>MTE2 在基线模拟器轨迹中是暴露的关键流水。先看搬运次数，再追踪编译器为何没有保留驻留面板。</p></div><strong>−'+time(c.before-c.after)+'<small>公开 QK 计时差</small></strong></div>'+flow(c)+action(1,'展开 '+c.trips+' 轮搬运，追踪固定面板 A');
  }
  function sources() {
    return '<details class="rc-sources"><summary>来源与 Mock 范围</summary><p>公开：PR 的 QK 计时、T=32 搬运字节与验证报告。T=8 字节量按次数推导。</p><p>Mock：任务起点、准备/后继、逐事件时长与排布。代码为语义示意，未重建真实重叠；不混入 decode 原始数据。未在本机重跑 NPU。</p><a target="_blank" rel="noopener" href="'+data.issue+'">问题 #2077 ↗</a> · <a target="_blank" rel="noopener" href="'+data.pr+'">修复与验证 #2080 ↗</a> · <a target="_blank" rel="noopener" href="'+data.files+'">修改文件 ↗</a><p>核对日期 2026-10-24 · 固定源码版本 '+data.commit+'</p></details>';
  }
  function render() {
    const c=current();
    sceneRoot.innerHTML='<div class="rc-main-head"><div><small>性能调优场景 · PR #2080</small><h2>固定矩阵 · '+c.trips+' 次重复搬运</h2><p>Ascend 910B2 · 独立 QK · '+tag('公开计时')+' '+tag('上下文与指令排布 Mock',true)+'</p></div><label>测试描述符 <select class="ctl" data-rc-select aria-label="切换测试描述符">'+data.cases.map(v=>'<option value="'+v.id+'"'+(v.id===c.id?' selected':'')+'>'+v.label+'</option>').join('')+'</select></label></div>'+metrics(c)+lanes(c)+'<div class="rc-main-toolbar" role="group" aria-label="主泳道细节"><button type="button" class="ctl '+(state.laneView==='task'?'on':'')+'" data-rc-view="task" aria-pressed="'+(state.laneView==='task')+'">任务对照</button><button type="button" class="ctl '+(state.laneView==='micro'?'on':'')+'" data-rc-view="micro" aria-pressed="'+(state.laneView==='micro')+'">展开核内搬运</button><span>点击耗时 QK → 右侧证据联动</span></div>'+(state.laneView==='micro'?'<section class="rc-main-micro">'+iteration(c)+micro(c)+'</section>':'<div class="rc-lane-prompt"><span>① 在主泳道选中 QK</span><span>② 展开固定面板的重复搬运</span><span>③ 右侧追踪 Pass 与验证</span></div>');
    const selected=state.task!==null;
    root.dataset.selectedTask=state.task||'';
    root.innerHTML='<header class="rc-panel-head"><small>#2080 · 性能调优闭环</small><h2>'+(selected?(state.task==='qk'?'QK · '+(state.arm==='after'?'修改后':'修改前'):'模拟上下文任务'):'选择主泳道中的任务')+'</h2><p>'+(selected?'所选描述符：'+c.label:'点击左侧 QK，追踪它为什么反复搬运同一面板。')+'</p></header>'+(selected?'<nav class="rc-steps" aria-label="任务分析证据链">'+steps.map((label,i)=>'<button type="button" data-rc-step="'+i+'"'+(state.step===i?' aria-current="step"':'')+'><span>0'+(i+1)+'</span>'+label+'</button>').join('')+'</nav><section class="rc-evidence" aria-live="polite"><div class="rc-section-head"><h3>'+steps[state.step]+'</h3></div>'+[finding,transfer,pass,safety,validation][state.step](c)+'</section>':'<div class="rc-empty"><div class="rc-empty-glyph" aria-hidden="true">↖</div><h3>先从耗时任务出发</h3><p>修改前 '+time(c.before)+'，修改后 '+time(c.after)+'。固定 A 的重复加载是本案例要追踪的观测对象。</p><button type="button" class="pass-entry" data-rc-task="qk" data-rc-arm="before">选中修改前 QK →</button></div>')+sources();
  }
  function handleClick(event) {
    const button=event.target.closest('[data-rc-step],[data-rc-task],[data-rc-iter],[data-rc-object],[data-rc-gate],[data-rc-case],[data-rc-view]');
    if(!button)return;
    const host=sceneRoot.contains(button)?sceneRoot:root;
    const d=button.dataset, scroll=root.scrollTop, laneScroll=sceneRoot.scrollTop;
    let focus;
    if(d.rcStep!==undefined){state.step=Number(d.rcStep);if(state.step===1)state.laneView='micro';focus='[data-rc-step="'+state.step+'"]';}
    if(d.rcTask){state.task=d.rcTask;state.arm=d.rcArm||'before';state.step=state.task==='qk'?1:0;if(state.task==='qk')state.laneView='micro';focus='[data-rc-task="'+state.task+'"][data-rc-arm="'+state.arm+'"]';onTask?.();}
    if(d.rcIter!==undefined){state.iter=Number(d.rcIter);state.object=d.rcEvent?({stationary:'fixed',streamed:'streamed',compute:'compute'})[d.rcEvent]:'fixed';focus='[data-rc-iter="'+state.iter+'"]';}
    if(d.rcObject){state.object=d.rcObject;state.laneView='micro';focus='[data-rc-object="'+state.object+'"]';}
    if(d.rcGate){state.gate=d.rcGate;focus='[data-rc-gate="'+state.gate+'"]';}
    if(d.rcCase){state.caseId=d.rcCase;state.iter=Math.min(state.iter,current().trips-1);focus='[data-rc-case="'+state.caseId+'"]';}
    if(d.rcView){state.laneView=d.rcView;if(d.rcView==='micro'&&state.task!=='qk'){state.task='qk';state.arm='before';state.step=1;onTask?.();}focus='[data-rc-view="'+state.laneView+'"]';}
    render();root.scrollTop=d.rcStep!==undefined||d.rcTask?0:scroll;sceneRoot.scrollTop=laneScroll;
    if(focus)host.querySelector(focus)?.focus({preventScroll:true});
  }
  function init(mainElement,panelElement,payload,selectCallback) {
    sceneRoot=mainElement;root=panelElement;data=payload;onTask=selectCallback;
    state={caseId:'k5t32',step:0,iter:0,object:'fixed',gate:'safe',task:null,arm:'before',laneView:'task'};
    for(const host of [sceneRoot,root]) {
      host.addEventListener('click',handleClick);
      host.addEventListener('change',event=>{
        const scroll=root.scrollTop,laneScroll=sceneRoot.scrollTop;
        let focus;
        if(event.target.matches('[data-rc-select]')){state.caseId=event.target.value;state.iter=Math.min(state.iter,current().trips-1);focus='[data-rc-select]';}
        else if(event.target.matches('[data-rc-iteration]')){state.iter=Number(event.target.value);focus='[data-rc-iteration]';}
        else return;
        render();root.scrollTop=scroll;sceneRoot.scrollTop=laneScroll;host.querySelector(focus)?.focus({preventScroll:true});
      });
    }
  }
  function activate() {
    if(!data?.cases?.length){sceneRoot.innerHTML='<p role="alert">#2080 场景证据未加载，请检查 data/residency-case.js。</p>';root.textContent='暂无案例数据。';return;}
    render();
  }
  return { init, activate, events };
})();
