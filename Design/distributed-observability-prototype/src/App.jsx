import { useMemo, useState } from "react";
import {
  ArrowLeft, ArrowRight, ArrowSquareOut, ArrowsLeftRight, CaretDown,
  CaretRight, ChartLine, Clock, Cpu, Database, FileCode, GitBranch,
  MagnifyingGlass, Network, Rows, Stack, X,
} from "@phosphor-icons/react";
import runData from "./run-data.json";

const fmt = (value, digits = 2) => value == null ? "—" : `${Number(value).toFixed(digits)} µs`;
const compact = (value) => new Intl.NumberFormat("zh-CN").format(value ?? 0);
const shortId = (value) => String(value).slice(-6);
const localSpan = (task) => task?.observed?.deviceSpanUs;
function crossRankSpanDiffs(ranks) {
  if (ranks.length < 2) return [];
  const maps = ranks.map((rank) => new Map(rank.tasks.map((task) => [task.id, task])));
  return ranks[0].tasks.map((task) => {
    const matches = maps.map((map) => map.get(task.id));
    const spans = matches.map(localSpan);
    if (matches.some((match) => !match || match.kernelNames.join("|") !== task.kernelNames.join("|")) || spans.some((span) => span == null)) return null;
    const fastest = Math.min(...spans);
    const slowest = Math.max(...spans);
    return { id: task.id, name: task.kernelNames.join(" + ") || "调度 / 边界任务", matches, spans, gapUs: slowest - fastest, ratio: fastest > 0 ? slowest / fastest : null };
  }).filter((item) => item && item.gapUs > 0).sort((a, b) => b.gapUs - a.gapUs);
}
const familyLabel = (name) => ({
  cp_token_allgather: "AllGather",
  o_group_a2a: "All-to-All",
  tp_o_rs: "Reduce-Scatter",
}[name] || name);

function IconButton({ label, active, children, onClick }) {
  return <button className={`rail-button${active ? " active" : ""}`} aria-label={label} title={label} onClick={onClick}>{children}</button>;
}

function Header({ page, onBack, onNavigate }) {
  return <header className="topbar">
    <div className="brand-mark"><Stack size={21} weight="duotone" /></div>
    <div className="brand-name">PyPTO <span>Distributed Run</span></div>
    <div className="top-divider" />
    <div className="run-breadcrumb">
      {page === "rank" && <><button className="back-button" onClick={onBack}><ArrowLeft size={16} /> Runs</button><span className="crumb-slash">/</span></>}
      <span className="crumb-muted">DeepSeek-V4</span><span className="crumb-slash">/</span>
      <strong>{runData.run.name}</strong>
    </div>
    <div className="topbar-spacer" />
    <button className={`top-link${page === "overview" ? " selected" : ""}`} onClick={() => onNavigate("overview")}>运行概览</button>
    <button className={`top-link${page === "rank" ? " selected" : ""}`} onClick={() => onNavigate("rank")}>Rank 详情</button>
    <div className="top-divider small-divider" />
    <span className="run-status"><i /> 已采集</span>
    <button className="avatar" aria-label="当前工作区">P</button>
  </header>;
}

function Sidebar({ page, onNavigate }) {
  return <aside className="sidebar">
    <div className="sidebar-run"><span className="run-dot" /><div><small>当前 Run</small><strong>{runData.run.id.split(":").at(-1)}</strong></div></div>
    <div className="sidebar-label">观测</div>
    <IconButton label="运行关系" active={page === "overview"} onClick={() => onNavigate("overview")}><Network size={19} /></IconButton>
    <IconButton label="Rank 详情" active={page === "rank"} onClick={() => onNavigate("rank")}><Cpu size={19} /></IconButton>
    <IconButton label="通信序列" onClick={() => onNavigate("comms")}><ArrowsLeftRight size={19} /></IconButton>
    <IconButton label="运行产物" onClick={() => onNavigate("artifacts")}><FileCode size={19} /></IconButton>
    <div className="sidebar-bottom"><Database size={16} /><span>本地采集</span></div>
  </aside>;
}

function RunMeta() {
  return <div className="run-meta-line">
    <span><i className="meta-dot cyan" />{runData.run.rankCount} Ranks</span>
    <span><i className="meta-dot" />{runData.run.platform} · {runData.run.backend}</span>
    <span><i className="meta-dot violet" />{runData.run.runtime}</span>
    <span className="meta-source">来源 <code>{runData.run.source}</code></span>
  </div>;
}

function SchedulerLayer({ rank, onOpen }) {
  return <section className="rank-layer scheduler-layer">
    <div className="layer-head"><span className="layer-icon slate"><Rows size={15} /></span><div className="layer-title"><strong>调度事件</strong><small>AICPU Scheduler</small></div><span className="layer-count">{compact(rank.counts.schedulerPhaseEvents)} 条</span><button className="layer-open" onClick={onOpen} aria-label="查看调度事件"><ArrowSquareOut size={14} /></button></div>
    <div className="phase-list">
      {rank.scheduler.phases.map((phase) => <div className="phase-chip" key={phase.kind}><span>{phase.kind.replaceAll("_", " ")}</span><b>{phase.count}</b></div>)}
    </div>
    <div className="layer-foot"><span>{compact(rank.counts.submitEvents)} 次任务提交记录</span><span className="source-mark">原始事件</span></div>
  </section>;
}

function ExecutionLayer({ rank, onOpen, onSelectTask, selectedTaskId, focusTaskIds }) {
  const ordered = rank.tasks.filter((task) => task.observed.deviceEventCount > 0)
    .sort((a, b) => (a.observed.deviceStartOffsetUs ?? Infinity) - (b.observed.deviceStartOffsetUs ?? Infinity));
  const focusIndices = focusTaskIds.map((id) => ordered.findIndex((task) => task.id === id)).filter((index) => index >= 0);
  const visibleIndices = new Set();
  for (const index of focusIndices) for (let offset = -1; offset <= 1; offset++) if (ordered[index + offset]) visibleIndices.add(index + offset);
  const visibleTasks = (visibleIndices.size ? [...visibleIndices].sort((a, b) => a - b).map((index) => ordered[index]) : ordered.slice(0, 7)).slice(0, 9);
  return <section className="rank-process">
    <div className="rank-process-heading"><div><span className="layer-icon cyan"><Cpu size={15} /></span><span><strong>设备执行时间线</strong><small>本 Rank trace · 按起始偏移排列</small></span></div><span className="layer-count">{rank.counts.taskInvocations} tasks</span></div>
    <div className="process-rows">{visibleTasks.map((task) => {
      const incoming = task.dependencies.filter((edge) => edge.direction === "in").length;
      const outgoing = task.dependencies.filter((edge) => edge.direction === "out").length;
      const isCommunication = task.kernelNames.some((name) => /allgather|a2a|reduce.?scatter/i.test(name));
      const isFocus = focusTaskIds.includes(task.id);
      return <button key={task.id} className={`process-row${isFocus ? " process-row-focus" : ""}${selectedTaskId === task.id ? " selected" : ""}`} onClick={() => onSelectTask(task)}>
        <span className={`process-time${isFocus ? " focus-time" : ""}`}>{fmt(task.observed.deviceStartOffsetUs, 0)}</span>
        <span className={`process-marker${isCommunication ? " marker-comm" : ""}${isFocus ? " marker-focus" : ""}`} />
        <span className="process-copy"><strong title={task.kernelNames.join(" + ")}>{task.kernelNames.join(" + ") || "调度 / 边界任务"}</strong><small>Task {task.id} · {incoming} 入 / {outgoing} 出依赖</small></span>
        <span className={`process-duration${isFocus ? " focus-duration" : ""}`}>{fmt(task.observed.deviceSpanUs, 1)}</span>
      </button>;
    })}</div>
    <button className="process-open-all" onClick={onOpen}>打开完整任务时间线 <ArrowRight size={13} /></button>
  </section>;
}

function CommunicationLayer({ rank, communications, onSelect, onSelectTask, selectedId }) {
  return <section className="rank-comms">
    <div className="rank-process-heading"><div><span className="layer-icon amber"><ArrowsLeftRight size={15} /></span><span><strong>通信任务阶段</strong><small>该 Rank 内的实际任务与设备跨度</small></span></div></div>
    <div className="communication-chips">
      {communications.map((communication) => {
        const taskIds = new Set(communication.members.filter((id) => id.includes(`:rank:${rank.rankIndex}:`)).map((id) => id.split(":task:").at(-1)));
        const tasks = rank.tasks.filter((task) => taskIds.has(task.id)).sort((a, b) => (a.observed.deviceStartOffsetUs ?? Infinity) - (b.observed.deviceStartOffsetUs ?? Infinity));
        return <div key={communication.id} className={`comm-sequence${selectedId === communication.id ? " selected" : ""}`}>
          <button className="comm-sequence-title" onClick={() => onSelect(communication)}><span><strong>{familyLabel(communication.id)}</strong><small>{communication.id}</small></span><b>{tasks.length} stages</b></button>
          {tasks.map((task) => <button className="comm-stage" key={task.id} onClick={() => onSelectTask(task)} title={`Task ${task.id} · ${fmt(task.observed.deviceSpanUs, 1)}`}><span>{task.kernelNames.join(" + ")}</span><code>{fmt(task.observed.deviceSpanUs, 1)}</code></button>)}
        </div>;
      })}
    </div>
    <div className="layer-foot"><span>各阶段按本 Rank trace 顺序展示</span><span className="source-mark">未推断跨卡配对</span></div>
  </section>;
}

function RankRegion({ rank, index, communications, selectedCommunication, selectedTaskId, onOpen, onSelectCommunication, onSelectTask, onNavigate, focusTaskIds }) {
  const artifactCount = rank.artifacts.reduce((total, group) => total + group.files.length, 0);
  return <article className={`rank-region rank-tone-${index}`}>
    <div className="rank-region-title">
      <span className="rank-cube"><Cpu size={16} weight="duotone" /></span>
      <div><strong>Rank {rank.rankIndex}</strong><small>Device {rank.deviceId}</small></div>
      <span className="rank-region-number">0{rank.rankIndex + 1}</span>
      <button className="rank-open-link" onClick={() => { onOpen(rank); onNavigate("rank"); }}>打开 Rank 详情 <ArrowSquareOut size={13} /></button>
    </div>
    <div className="rank-hardware-line"><span>{rank.device.coreCount} cores</span><i /> <span>AIC {rank.device.aicCount}</span><i /> <span>AIV {rank.device.aivCount}</span></div>
    <div className="rank-evidence"><span><i className="available-dot" />{compact(rank.counts.deviceTraceEvents)} 设备 trace 事件</span><span>{compact(rank.counts.dependencyEdges)} 条本地依赖</span><button onClick={() => { onOpen(rank); onNavigate("rank", "scheduler"); }}>{rank.counts.submitEvents} 条调度提交 <ArrowSquareOut size={11} /></button><button onClick={() => { onOpen(rank); onNavigate("rank", "artifacts"); }}>{compact(artifactCount)} 个构建文件 <ArrowSquareOut size={11} /></button></div>
    <ExecutionLayer rank={rank} onOpen={() => { onOpen(rank); onNavigate("rank", "tasks"); }} onSelectTask={onSelectTask} selectedTaskId={selectedTaskId} focusTaskIds={focusTaskIds} />
    <CommunicationLayer rank={rank} communications={communications} onSelect={onSelectCommunication} onSelectTask={onSelectTask} selectedId={selectedCommunication?.id} />
  </article>;
}

function CommunicationCorridor({ communications, selectedCommunication, onSelect }) {
  return <section className="relation-corridor" aria-label="跨 Rank 通信关系">
    <div className="corridor-title"><span>通信过程</span><small>真实采集到的任务阶段</small></div>
    <div className="corridor-events">
      {communications.map((communication) => {
        const memberIds = new Set(communication.members.map((member) => member.split(":task:").at(-1)));
        const stageNames = [...new Set(runData.ranks.flatMap((rank) => rank.tasks.filter((task) => memberIds.has(task.id)).flatMap((task) => task.kernelNames.map((name) => name.replace(`${communication.id}_`, "")))))];
        return <button key={communication.id} className={`corridor-event${selectedCommunication?.id === communication.id ? " selected" : ""}`} onClick={() => onSelect(communication)}>
        <span className="bridge-line left" /><span className="bridge-node left" />
        <span className="bridge-label"><small>{familyLabel(communication.id)}</small><strong>{communication.id}</strong><em>{stageNames.join(" → ")}</em></span>
        <span className="bridge-node right" /><span className="bridge-line right" />
      </button>;
      })}
    </div>
    <div className="corridor-note"><span className="note-dot" />按任务前缀归组，不表示配对时序</div>
  </section>;
}

function CrossRankFindings({ ranks, diffs, onSelectTask }) {
  return <section className="findings-panel">
    <div className="findings-heading"><div><span className="section-kicker">SAME TASK ID · DEVICE SPAN</span><h2>这次 Run 的 Rank 耗时分歧</h2><p>只比较同一 Task ID 的本地设备跨度，不比较 Rank 起始时刻。</p></div><span className="finding-basis"><i />来自 trace</span></div>
    <div className="findings-list">{diffs.slice(0, 3).map((finding) => <div className="finding-row" key={finding.id}>
      <button className="finding-open" onClick={() => onSelectTask(finding)} aria-label={`查看 Task ${finding.id} 详情`}><ArrowSquareOut size={14} /></button>
      <div className="finding-task"><strong>{finding.name}</strong><small>Task {finding.id}</small></div>
      <div className="finding-rank-values">{ranks.map((rank, index) => <div className="finding-rank-value" key={rank.id}><span className={`finding-dot tone-${rank.rankIndex}`} />R{rank.rankIndex}<i className="finding-bar"><b style={{ width: `${100 * finding.spans[index] / Math.max(...finding.spans)}%` }} /></i><strong>{fmt(finding.spans[index], 1)}</strong></div>)}</div>
      <div className="finding-gap"><small>跨度差</small><b>{fmt(finding.gapUs, 1)}</b>{finding.ratio != null && <em>{finding.ratio.toFixed(1)}×</em>}</div>
    </div>)}</div>
  </section>;
}

function CommunicationInspector({ communication, ranks, onClose, onOpenRank, onNavigate }) {
  const participants = ranks.map((rank) => ({ rank, count: communication.members.filter((id) => id.includes(`:rank:${rank.rankIndex}:`)).length })).filter((entry) => entry.count > 0);
  return <aside className="inspector-panel">
    <div className="inspector-head"><div><small>SELECTED RELATION</small><h2>{communication.id}</h2><span className="comm-type">{familyLabel(communication.id)}</span></div><button className="icon-close" onClick={onClose} aria-label="关闭"><X size={17} /></button></div>
    <div className="inspector-visual"><span className="mini-rank cyan-mini"><Cpu size={17} /></span><span className="mini-wire"><i /></span><span className="mini-rank purple-mini"><Cpu size={17} /></span></div>
    <section className="inspect-section"><h3>涉及 Rank</h3>{participants.map(({ rank, count }) => <button className="participant-row" key={rank.id} onClick={() => { onOpenRank(rank); onNavigate("rank"); }}><span className={`participant-dot tone-${rank.rankIndex}`} /><strong>Rank {rank.rankIndex}</strong><small>Device {rank.deviceId}</small><b>{count} tasks</b><ArrowRight size={14} /></button>)}</section>
    <section className="inspect-section"><h3>采集关联方式</h3><p className="inspect-copy">在每个 Rank 的任务与 kernel 名称中观察到相同前缀。来源数据没有通信实例 ID，因此这里表示序列关联，不表示精确的跨卡事件配对。</p></section>
    <section className="inspect-section source-section"><h3>可观测字段</h3><div className="source-field"><span>任务调用 / kernel 名称</span><b><i className="available-dot" />已采集</b></div><div className="source-field"><span>跨 Rank collective ID</span><b className="unavailable"><i />未提供</b></div><div className="source-field"><span>消息大小 / 链路拓扑</span><b className="unavailable"><i />未提供</b></div><div className="source-path"><Database size={13} /> deps.json · name_map.json</div></section>
  </aside>;
}

function Overview({ selectedCommunication, selectedTask, setSelectedTask, setSelectedCommunication, setSelectedRank, navigate }) {
  const ranks = runData.ranks;
  const spanDiffs = useMemo(() => crossRankSpanDiffs(ranks), [ranks]);
  const focusTaskIds = spanDiffs.slice(0, 3).map((item) => item.id);
  return <div className="page-content overview-page">
    <div className="page-intro"><div><div className="eyebrow"><i /> DISTRIBUTED EXECUTION · OBSERVABILITY</div><h1>{runData.run.name}</h1><p>一次上板执行的跨 Rank 过程与产物</p></div><div className="page-intro-actions"><span className="run-id-tag">{runData.run.id}</span><button className="quiet-button"><Database size={15} /> 原始采集 <ArrowSquareOut size={13} /></button></div></div>
    <RunMeta />
    <CrossRankFindings ranks={ranks} diffs={spanDiffs} onSelectTask={(finding) => { setSelectedRank(ranks[0]); setSelectedTask(finding.matches[0]); navigate("rank", "tasks"); }} />
    <div className="overview-layout">
      <div className="rank-map-wrap">
        <div className="map-heading"><div><span className="section-kicker">RANK SPACES</span><h2>Rank 执行区域</h2></div><div className="map-tools"><span className="map-chip"><i />{ranks.length} regions</span><span className="map-scale">每个区域独立时钟</span></div></div>
        <div className={`rank-map${ranks.length > 2 ? " rank-map-many" : ""}`}>
          {ranks.map((rank, index) => <RankRegion key={rank.id} rank={rank} index={index} communications={runData.communication} selectedCommunication={selectedCommunication} selectedTaskId={selectedTask?.id} focusTaskIds={focusTaskIds} onOpen={setSelectedRank} onSelectCommunication={setSelectedCommunication} onSelectTask={(task) => { setSelectedRank(rank); setSelectedTask(task); navigate("rank", "tasks"); }} onNavigate={navigate} />)}
          <CommunicationCorridor communications={runData.communication} selectedCommunication={selectedCommunication} onSelect={(communication) => { setSelectedCommunication(communication); setSelectedTask(null); }} />
        </div>
        <div className="rank-scale-note"><span><i />Rank 区域</span><span><i className="legend-task" />Task invocation</span><span><i className="legend-dep" />本 Rank 依赖</span><span><i className="legend-comm" />通信序列关联</span><small>Rank 内时间戳未作跨卡对齐</small></div>
      </div>
      <CommunicationInspector communication={selectedCommunication} ranks={ranks} onClose={() => setSelectedCommunication(null)} onOpenRank={setSelectedRank} onNavigate={navigate} />
    </div>
    <CoverageFooter />
  </div>;
}

function CoverageFooter() {
  return <div className="coverage-footer"><div><span className="coverage-icon"><ChartLine size={16} /></span><span><strong>观测覆盖</strong><small>存在 Rank 本地执行证据、调度记录与声明依赖</small></span></div><div className="coverage-items"><span><i className="available-dot" />设备事件</span><span><i className="available-dot" />调度阶段</span><span><i className="available-dot" />Kernel 名称</span><span><i className="unavailable-dot" />通信字节数</span><span><i className="unavailable-dot" />跨 Rank 时钟对齐</span></div></div>;
}

function RankSubnav({ tab, setTab }) {
  const tabs = [["tasks", "任务调用", Cpu], ["scheduler", "调度事件", Rows], ["artifacts", "Kernel 与产物", FileCode], ["dependencies", "依赖边", GitBranch]];
  return <div className="rank-subnav">{tabs.map(([id, label, Icon]) => <button key={id} className={tab === id ? "active" : ""} onClick={() => setTab(id)}><Icon size={15} />{label}</button>)}</div>;
}

function TaskDetail({ task, rank }) {
  if (!task) return <div className="empty-detail"><Cpu size={22} /><strong>选择一个任务</strong><span>查看任务参数、设备执行和前后依赖。</span></div>;
  const incoming = task.dependencies.filter((edge) => edge.direction === "in");
  const outgoing = task.dependencies.filter((edge) => edge.direction === "out");
  return <div className="task-detail-body">
    <div className="task-detail-title"><small>TASK INVOCATION</small><h2>{task.kernelNames.join(" + ") || "调度 / 边界任务"}</h2><code>Task {task.id}</code></div>
    <div className="detail-stats"><div><small>Blocks</small><strong>{task.blockNum ?? "—"}</strong></div><div><small>设备事件</small><strong>{task.observed.deviceEventCount}</strong></div><div><small>设备跨度</small><strong>{fmt(task.observed.deviceSpanUs, 1)}</strong></div></div>
    <section className="detail-group"><h3>调度与执行</h3><DetailRow label="Scope" value={task.scope} /><DetailRow label="Early dispatch" value={task.earlyDispatch ? "是" : "否"} /><DetailRow label="Kernel core-time" value={fmt(task.observed.kernelCoreTimeUs)} /><DetailRow label="Local setup" value={fmt(task.observed.setupTimeUs)} /><DetailRow label="参与 Core" value={task.observed.coreIds.length ? task.observed.coreIds.join(", ") : "未关联"} /><DetailRow label="Rank 内起始偏移" value={fmt(task.observed.deviceStartOffsetUs)} /></section>
    <section className="detail-group"><h3>Tensor 参数 <span>{task.args.length}</span></h3><div className="arg-list">{task.args.map((arg) => <div className="arg-row" key={`${arg.index}-${arg.tensorId}`}><div><b>{arg.role}</b><code>arg {arg.index}</code></div><span>{arg.dtype} · {arg.shape?.join(" × ") || "shape unavailable"}</span><small>tensor {arg.tensorId}</small></div>)}</div></section>
    <section className="detail-group"><h3>依赖关系 <span>{incoming.length} 入 · {outgoing.length} 出</span></h3><div className="dependency-detail-list">{[...incoming.map((edge) => ({ ...edge, label: "前序" })), ...outgoing.map((edge) => ({ ...edge, label: "后继" }))].slice(0, 8).map((edge, index) => <div className="dependency-detail" key={`${edge.label}-${edge.otherTaskId}-${index}`}><span>{edge.label}</span><code>Task {edge.otherTaskId}</code><small>tensor {edge.tensorId || "—"}</small><small>{edge.source} · {(edge.flags || []).join(", ")}</small></div>)}{incoming.length + outgoing.length > 8 && <span className="more-evidence">另有 {incoming.length + outgoing.length - 8} 条关联，可在依赖边中查看</span>}</div></section>
    <div className="evidence-foot"><Database size={13} />{rank.source}/deps.json · merged_swimlane*.json</div>
  </div>;
}

function DetailRow({ label, value }) { return <div className="detail-row"><span>{label}</span><b>{value ?? "—"}</b></div>; }

function TaskBrowser({ rank, selectedTask, setSelectedTask }) {
  const [query, setQuery] = useState("");
  const [sortMode, setSortMode] = useState("time");
  const tasks = useMemo(() => {
    const matching = rank.tasks.filter((task) => `${task.id} ${task.kernelNames.join(" ")} ${task.scope}`.toLowerCase().includes(query.toLowerCase()));
    return matching.sort((a, b) => sortMode === "time" ? (a.observed.deviceStartOffsetUs ?? Infinity) - (b.observed.deviceStartOffsetUs ?? Infinity) : Number(a.id) - Number(b.id));
  }, [rank, query, sortMode]);
  const maxTimeline = Math.max(1, ...tasks.map((task) => (task.observed.deviceStartOffsetUs || 0) + (task.observed.deviceSpanUs || 0)));
  return <div className="task-browser-layout">
    <section className="detail-main-panel">
      <div className="detail-panel-heading"><div><span className="section-kicker">TASK GRAPH / TRACE INDEX</span><h2>Rank {rank.rankIndex} · 全部任务调用</h2><p>{rank.counts.taskInvocations} 个已展开调用，{rank.counts.dependencyEdges} 条本地依赖，{rank.counts.dependencyOnlyTaskRefs} 个依赖边引用任务未展开</p></div><div className="search-box"><MagnifyingGlass size={15} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索 Task / Kernel / scope" /></div></div>
      <div className="trace-head"><span>Task / Kernel</span><span>块数</span><span>Core-time</span><span>Span（Rank 本地）</span><span>时间轴 · rank-local</span></div>
      <div className="trace-axis"><span>本地 trace 时间范围</span><span>0</span><span>{fmt(maxTimeline, 0)}</span></div>
      <div className="task-trace-list">{tasks.map((task) => {
        const left = 100 * (task.observed.deviceStartOffsetUs || 0) / maxTimeline;
        const width = task.observed.deviceSpanUs == null ? 0 : Math.max(0.4, Math.min(100 - left, 100 * task.observed.deviceSpanUs / maxTimeline));
        return <button key={task.id} className={`trace-task-row${selectedTask?.id === task.id ? " selected" : ""}`} onClick={() => setSelectedTask(task)}>
          <span className="trace-task-name"><i className={task.observed.deviceEventCount ? "has-trace" : "no-trace"} /><span><b>{task.kernelNames.join(" + ") || "调度 / 边界任务"}</b><small>Task {task.id} · {task.scope}</small></span></span>
          <span className="trace-number">{task.blockNum ?? "—"}</span><span className="trace-number">{fmt(task.observed.kernelCoreTimeUs, 1)}</span><span className="trace-number">{fmt(task.observed.deviceSpanUs, 1)}</span>
          <span className="task-track"><i style={{ left: `${Math.min(98, left)}%`, width: `${width}%` }} /></span>
        </button>;
      })}{!tasks.length && <div className="no-results">没有匹配的任务。</div>}</div>
      <div className="rank-local-note"><Clock size={14} />跨度与起始偏移仅在本 Rank trace 内定义；Rank 间时钟未对齐，不用于跨 Rank 时间比较。</div>
    </section>
    <aside className="task-inspector"><TaskDetail task={selectedTask} rank={rank} /></aside>
  </div>;
}

function SchedulerBrowser({ rank, selectedTask, setSelectedTask }) {
  const [phasePage, setPhasePage] = useState(0);
  const [selectedKind, setSelectedKind] = useState(rank.scheduler.phases[0]?.kind || "dispatch");
  const [query, setQuery] = useState("");
  const phaseEvents = rank.scheduler.phases;
  const pageSize = 24;
  const events = rank.scheduler.submissions.filter((event) => !query || `${event.taskId} ${event.submitIndex}`.includes(query));
  return <div className="scheduler-view">
    <div className="detail-panel-heading"><div><span className="section-kicker">AICPU SCHEDULER</span><h2>Rank {rank.rankIndex} · 调度过程</h2><p>按设备时钟换算 cycles；下表展示每次 Orchestrator 任务提交记录。</p></div><div className="search-box"><MagnifyingGlass size={15} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="按 Task ID / submit index" /></div></div>
    <div className="scheduler-summary">{phaseEvents.map((phase) => <button key={phase.kind} className={`scheduler-summary-item${selectedKind === phase.kind ? " selected" : ""}`} onClick={() => setSelectedKind(phase.kind)}><span>{phase.kind}</span><strong>{compact(phase.count)}</strong><small>{fmt(phase.totalTimeUs)} 累计 · {phase.tasksProcessed} tasks processed</small></button>)}</div>
    <section className="detail-main-panel scheduler-events-panel"><div className="table-title"><h3>任务提交事件</h3><span>{compact(events.length)} records · source: {rank.source}/chip_swimlane_records.json</span></div><div className="scheduler-event-head"><span>Submit index</span><span>Task ID</span><span>Rank-local offset</span><span>记录耗时</span><span>任务</span></div>{events.slice(phasePage * pageSize, (phasePage + 1) * pageSize).map((event) => <button className={`scheduler-event-row${selectedTask?.id === event.taskId ? " selected" : ""}`} key={`${event.submitIndex}-${event.taskId}`} onClick={() => setSelectedTask(rank.tasks.find((task) => task.id === event.taskId) || null)}><code>{event.submitIndex}</code><code>{event.taskId}</code><span>{fmt(event.startOffsetUs, 1)}</span><span>{fmt(event.durationUs, 1)}</span><span>{rank.tasks.find((task) => task.id === event.taskId)?.kernelNames.join(" + ") || "引用未展开任务"}</span></button>)}<div className="pagination"><span>第 {phasePage + 1} / {Math.max(1, Math.ceil(events.length / pageSize))} 页</span><div><button disabled={phasePage === 0} onClick={() => setPhasePage(Math.max(0, phasePage - 1))}>上一页</button><button disabled={(phasePage + 1) * pageSize >= events.length} onClick={() => setPhasePage(phasePage + 1)}>下一页</button></div></div></section>
    <div className="rank-local-note"><Clock size={14} />调度阶段与任务提交记录来自独立字段；界面不将 AICPU 阶段和 AICore trace 合并成未经校准的统一时间轴。</div>
  </div>;
}

function ArtifactBrowser({ rank }) {
  const [query, setQuery] = useState("");
  const allFiles = rank.artifacts.flatMap((group) => group.files.map((file) => ({ file, group: group.group })));
  const files = allFiles.filter((item) => item.file.toLowerCase().includes(query.toLowerCase()));
  return <section className="detail-main-panel artifact-browser"><div className="detail-panel-heading"><div><span className="section-kicker">BUILD OUTPUTS</span><h2>Rank {rank.rankIndex} · Kernel 与产物索引</h2><p>展示本次采集目录中实际存在的 orchestration、PTOAS 与 kernel 文件。</p></div><div className="search-box"><MagnifyingGlass size={15} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="筛选文件名或路径" /></div></div><div className="artifact-group-list">{rank.artifacts.map((group) => <section className="artifact-group" key={group.group}><div className="artifact-group-head"><span className="artifact-type-icon"><FileCode size={15} /></span><strong>{group.group}</strong><small>{group.files.length} files</small><CaretDown size={14} /></div><div className="artifact-files">{files.filter((item) => item.group === group.group).slice(0, 60).map((item) => <div className="artifact-file" key={item.file}><FileCode size={14} /><code>{item.file}</code><span>{item.file.split(".").at(-1)}</span></div>)}{files.filter((item) => item.group === group.group).length > 60 && <div className="artifact-more">已显示 60 项；使用搜索筛选其余文件</div>}</div></section>)}</div><div className="rank-local-note"><Database size={14} />文件列表从 next_levels/decode_csa_test 扫描。生成代码与运行时 callable ID 没有显式逐项映射文件，页面不会推定一一对应。</div></section>;
}

function DependencyBrowser({ rank, selectedTask, setSelectedTask }) {
  const [query, setQuery] = useState("");
  const dependencies = rank.dependencies.filter((edge) => `${edge.pred} ${edge.succ} ${edge.tensorId || ""} ${edge.source} ${(edge.flags || []).join(" ")}`.toLowerCase().includes(query.toLowerCase()));
  return <div className="dependency-view"><section className="detail-main-panel"><div className="detail-panel-heading"><div><span className="section-kicker">DECLARED TASK DEPENDENCIES</span><h2>Rank {rank.rankIndex} · 依赖边</h2><p>{rank.counts.dependencyEdges} 条 deps.json 中声明的 rank-local 边。</p></div><div className="search-box"><MagnifyingGlass size={15} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Task ID / tensor ID / edge source" /></div></div><div className="dependency-table-head"><span>Predecessor</span><span>Successor</span><span>Tensor</span><span>Source / flags</span></div><div className="dependency-table">{dependencies.slice(0, 300).map((edge, index) => <div className="dependency-row" key={`${edge.pred}-${edge.succ}-${index}`}><button onClick={() => setSelectedTask(rank.tasks.find((task) => task.id === edge.pred) || null)}><code>{edge.pred}</code></button><ArrowRight size={13} /><button onClick={() => setSelectedTask(rank.tasks.find((task) => task.id === edge.succ) || null)}><code>{edge.succ}</code></button><code className="tensor-col">{edge.tensorId || "—"}</code><span>{edge.source} · {(edge.flags || []).join(", ") || "—"}</span></div>)}{dependencies.length > 300 && <div className="artifact-more">已显示前 300 条，继续缩小搜索范围查看剩余依赖。</div>}</div></section><aside className="task-inspector"><TaskDetail task={selectedTask} rank={rank} /></aside></div>;
}

function RankPage({ rank, tab, setTab, selectedTask, setSelectedTask, onBack }) {
  return <div className="page-content rank-page"><div className="rank-page-heading"><button className="back-button" onClick={onBack}><ArrowLeft size={15} />返回 Run 概览</button><div className="rank-heading-main"><div className="rank-heading-badge"><Cpu size={18} /></div><div><div className="eyebrow">RANK EXECUTION WORKSPACE</div><h1>Rank {rank.rankIndex} <span>· Device {rank.deviceId}</span></h1><p>{rank.counts.taskInvocations} task invocations · {rank.counts.dependencyEdges} dependency edges · {rank.counts.deviceTraceEvents} device trace events</p></div></div><div className="rank-page-actions"><span className="hardware-pill">{rank.device.coreCount} cores · AIC {rank.device.aicCount} · AIV {rank.device.aivCount}</span><button className="quiet-button"><Database size={14} />查看采集来源</button></div></div>
    <RankSubnav tab={tab} setTab={setTab} />
    {tab === "tasks" && <TaskBrowser rank={rank} selectedTask={selectedTask} setSelectedTask={setSelectedTask} />}
    {tab === "scheduler" && <SchedulerBrowser rank={rank} selectedTask={selectedTask} setSelectedTask={setSelectedTask} />}
    {tab === "artifacts" && <ArtifactBrowser rank={rank} />}
    {tab === "dependencies" && <DependencyBrowser rank={rank} selectedTask={selectedTask} setSelectedTask={setSelectedTask} />}
  </div>;
}

function CommunicationPage({ selectedCommunication, setSelectedCommunication }) {
  return <div className="page-content generic-data-page"><div className="page-intro"><div><div className="eyebrow">CROSS-RANK RELATIONS</div><h1>通信序列</h1><p>按实际 kernel 名称组织采集到的通信相关任务。</p></div></div><div className="communication-page-grid">{runData.communication.map((communication) => { const selected = communication.id === selectedCommunication?.id; return <button key={communication.id} className={`communication-record${selected ? " selected" : ""}`} onClick={() => setSelectedCommunication(communication)}><div className="communication-record-head"><span className="record-comm-icon"><ArrowsLeftRight size={17} /></span><span>{familyLabel(communication.id)}</span><b>{communication.observedTaskCount} tasks</b></div><h2>{communication.id}</h2><div className="record-participants">{runData.ranks.map((rank) => <span key={rank.id}>Rank {rank.rankIndex} · {communication.members.filter((id) => id.includes(`:rank:${rank.rankIndex}:`)).length} tasks</span>)}</div><p>同名前缀任务在多个 Rank 输出中均有观测。缺少 collective ID 与跨 Rank 对时信息。</p></button>; })}</div></div>;
}

function ArtifactPage() {
  const files = runData.ranks[0].artifacts;
  return <div className="page-content generic-data-page"><div className="page-intro"><div><div className="eyebrow">RUN OUTPUT INDEX</div><h1>模型 / 算子产物</h1><p>本次采集中的构建与运行时产物目录。</p></div></div><div className="artifact-run-summary"><div><small>Program</small><strong>{runData.run.name}</strong></div><div><small>采集根目录</small><code>{runData.run.source.replace("/distributed_meta.json", "")}</code></div></div><div className="artifact-summary-grid">{files.map((group) => <div className="artifact-summary-item" key={group.group}><span><FileCode size={16} />{group.group}</span><strong>{group.files.length}</strong><small>实际文件</small><code>{group.files.slice(0, 3).join(" · ")}</code></div>)}</div></div>;
}

export function App() {
  const [page, setPage] = useState("overview");
  const [rankTab, setRankTab] = useState("tasks");
  const [selectedRank, setSelectedRank] = useState(runData.ranks[0]);
  const [selectedCommunication, setSelectedCommunication] = useState(runData.communication[0] || null);
  const [selectedTask, setSelectedTask] = useState(null);
  const navigate = (nextPage, nextTab) => {
    if (nextTab) setRankTab(nextTab);
    if (nextPage === "rank" && !selectedRank) setSelectedRank(runData.ranks[0]);
    setPage(nextPage);
  };
  return <div className="app-shell">
    <Header page={page} onBack={() => setPage("overview")} onNavigate={navigate} />
    <Sidebar page={page} onNavigate={navigate} />
    <main className="main-shell">
      {page === "overview" && <Overview selectedCommunication={selectedCommunication} selectedTask={selectedTask} setSelectedTask={setSelectedTask} setSelectedCommunication={setSelectedCommunication} setSelectedRank={setSelectedRank} navigate={navigate} />}
      {page === "rank" && selectedRank && <RankPage rank={selectedRank} tab={rankTab} setTab={setRankTab} selectedTask={selectedTask} setSelectedTask={setSelectedTask} onBack={() => setPage("overview")} />}
      {page === "comms" && <CommunicationPage selectedCommunication={selectedCommunication} setSelectedCommunication={setSelectedCommunication} />}
      {page === "artifacts" && <ArtifactPage />}
    </main>
  </div>;
}
