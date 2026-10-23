import {nextPassAction} from './pass-playback.js';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ReactFlow, {
  Background,
  BackgroundVariant,
  Panel,
  ReactFlowProvider,
  useReactFlow,
  useNodesInitialized,
  getTransformForBounds,
} from 'reactflow';
import 'reactflow/dist/style.css';
import fxIcon from './vendor/fx.svg';
import {sample,isQk} from './sample-registry.js';
import {CallableSelector} from './CallableSelector.jsx';
import {visualPassDiff} from './visual-diff.js';
const {manifest,isa,graph,instructionCollections}=sample;
import { buildDomainState, groupPassesForRail, passes, toReactFlow } from './graph-state';
import { edgeTypes, GraphMarkers, nodeTypes, MemoryLegend, DiagramSettings } from './graph-components';
import './styles.css';

import { MemoryDetails as MergeMemoryDetails, memoryRows as mergeMemoryRows, visibleChangedPasses } from './CallableField';

import {MemoryDetails as QkMemoryDetails,memoryRows as qkMemoryRows} from './MemoryField.jsx';
const MemoryDetails=isQk?QkMemoryDetails:MergeMemoryDetails;
const memoryRows=isQk?qkMemoryRows:mergeMemoryRows;
import './vendor/liquid-glass-controls/liquid-glass.css';
import './vendor/liquid-glass-controls/liquid-glass.js';

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const phaseById = new Map(passes.phases.map((phase) => [phase.id, phase]));
const collections = instructionCollections(isa.instructions);
const allInstances = collections.flatMap(item => item.instances);
const changedPasses = isQk?visualPassDiff(graph,passes.passes).filter(p=>p.changed).map(p=>p.index):visibleChangedPasses;
const passScope = pass => changedPasses.includes(pass.index) ? 'local' : 'none';
const passScopeLabel = pass => ({ none: '主体图未变化', local: '主体图有变化', global: '全局变化 · csa_merge_pack_publish 不变' })[passScope(pass)];
const resultIndex = manifest.passCount + 1;
const playbackSteps = [...changedPasses, resultIndex];
const codegenDescription = '前面的 Pass 负责变换、降低和准备 IR；随后 PTOCodegen 将准备好的 IR 输出为 .pto 指令。';
const resultStage = { ...passes.passes.at(-1), index: resultIndex, name: 'PTO-ISA · 最终产物', lens: '代码生成', noOp: false, delta: [], affectedObjectIds: [], summary: codegenDescription, contract: { intent: codegenDescription }, sourceRefs: [{ file: isa.source.pto, evidence: 'PTOCodegen 输出的 PTO 指令产物' }] };

function fitGraph(flow, duration = 320) {
  const nodes = flow.getNodes();
  if (!nodes.length) return;
  const x = Math.min(...nodes.map(n => n.position.x));
  const y = Math.min(...nodes.map(n => n.position.y));
  const width = Math.max(...nodes.map(n => n.position.x + Number(n.style?.width || n.width))) - x;
  const height = Math.max(...nodes.map(n => n.position.y + Number(n.style?.height || n.height))) - y;
  // Keep the full graph clear of the heading and the floating Pass controls.
  const [tx, ty, zoom] = getTransformForBounds({ x, y, width, height }, window.innerWidth - 64, window.innerHeight - 250, 0.12, 1, 0.08);
  flow.setViewport({ x: tx + 32, y: ty + 140, zoom }, { duration: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : duration });
}

function initialUrlState() {
  const params = new URLSearchParams(window.location.hash.replace(/^#/, ''));
  return {
    passIndex: params.get('stage') === 'isa' || params.get('isa') === '1' ? resultIndex : clamp(Number(params.get('pass')) || 35, 1, manifest.passCount),
    selectedId: params.get('node') || null,
    showIsa: params.get('isa') === '1',
  };
}

function Icon({ name }) {
  const paths = {
    settings: <><path d="M4 7h5m4 0h7M4 17h9m4 0h3" /><circle cx="11" cy="7" r="2" /><circle cx="15" cy="17" r="2" /></>,
    fit: <><path d="M8 3H3v5M16 3h5v5M8 21H3v-5M16 21h5v-5" /><path d="M3 8l5-5M21 8l-5-5M3 16l5 5M21 16l-5 5" /></>,
    reset: <><path d="M4 12a8 8 0 1 0 2.34-5.66L4 8.68" /><path d="M4 4v4.68h4.68" /></>,
    info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v5" /><circle cx="12" cy="8" r=".7" fill="currentColor" stroke="none" /></>,
    detail: <><path d="M4 6h16M4 12h16M4 18h10" /><circle cx="18" cy="18" r="2" /></>,
    close: <path d="M6 6l12 12M18 6L6 18" />,
    play: <path d="M8 5l11 7-11 7Z" />,
    pause: <path d="M8 5v14M16 5v14" />,
    zoom: <><circle cx="10" cy="10" r="6" /><path d="m15 15 6 6M7 10h6M10 7v6" /></>,
  };
  return <svg viewBox="0 0 24 24" aria-hidden="true">{paths[name]}</svg>;
}

function CanvasTools({ onResetPositions }) {
  const flow = useReactFlow();
  return <Panel position="bottom-right" className="canvas-tools" data-liquid-glass="control">
    <button type="button" className="zoom-trigger" aria-label="展开缩放控件"><Icon name="zoom" /></button>
    <div className="zoom-options">
    <button type="button" onClick={() => flow.zoomOut({ duration: 160 })} aria-label="缩小视图" data-tooltip="缩小">−</button>
    <output data-liquid-readout aria-label="当前缩放">ZOOM</output>
    <button type="button" onClick={() => flow.zoomIn({ duration: 160 })} aria-label="放大视图" data-tooltip="放大">＋</button>
    <span className="tool-divider" />
    <button type="button" onClick={() => fitGraph(flow)} aria-label="返回整体" data-tooltip="返回整体"><Icon name="fit" /></button>
    <button type="button" onClick={() => { onResetPositions(); requestAnimationFrame(() => fitGraph(flow)); }} aria-label="恢复自动布局" data-tooltip="恢复自动布局"><Icon name="reset" /></button>
    </div>
  </Panel>;
}

function GraphCanvas({ nodes, edges, showIsa, loadExpanded, onSelect, onHover, onMoveNode, onResetPositions, onExpandRegion }) {
  const flow = useReactFlow();
  const initialized = useNodesInitialized();
  const fitted = useRef(false);
  useEffect(()=>{if(initialized)requestAnimationFrame(()=>fitGraph(flow,0));},[nodes.length,initialized]);
  useEffect(()=>{const onResize=()=>fitGraph(flow,0);window.addEventListener('resize',onResize);return ()=>window.removeEventListener('resize',onResize);},[flow]);
  const overviewViewport = useRef(null);
  useEffect(() => {
    if (loadExpanded && showIsa) {
      if (!overviewViewport.current) overviewViewport.current = flow.getViewport();
      const target = nodes.find((node) => node.id === loadExpanded);
      const frame = requestAnimationFrame(() => {
        if (target) {
          const [x,y,zoom] = getTransformForBounds({ ...target.position, width:Number(target.style.width), height:Number(target.style.height) }, window.innerWidth - 96, window.innerHeight - 280, .12, 1, .12);
          flow.setViewport({x:x+48,y:y+150,zoom}, {duration:360});
        }
      });
      return () => cancelAnimationFrame(frame);
    }
    if (overviewViewport.current) {
      flow.setViewport(overviewViewport.current, { duration: 300 });
      overviewViewport.current = null;
    }
  }, [flow, loadExpanded, showIsa]);
  useEffect(() => {
    if (fitted.current || !initialized) return;
    fitted.current = true;
    requestAnimationFrame(() => fitGraph(flow, 0));
  }, [flow, initialized]);
  useEffect(() => {
    if (!showIsa) return;
    const frame = requestAnimationFrame(() => fitGraph(flow, 360));
    return () => cancelAnimationFrame(frame);
  }, [flow, showIsa]);

  return <ReactFlow
    nodes={nodes}
    edges={edges}
    nodeTypes={nodeTypes}
    edgeTypes={edgeTypes}
    minZoom={0.12}
    maxZoom={2.2}
    fitViewOptions={{ padding: 0.1 }}
    nodesConnectable={false}
    elementsSelectable
    proOptions={{ hideAttribution: true }}
    onNodeClick={(_, node) => { if (node.type !== 'rankFrame' && !node.data.instances) onSelect(node.id); }}
    onNodeDoubleClick={(_,node)=>node.data.regionOperations && onExpandRegion?.(node.data.owner||node.id)}
    onNodeMouseEnter={(_, node) => onHover(node.id)}
    onNodeMouseLeave={() => onHover(null)}
    onNodeDragStop={(_, node) => onMoveNode(node.id, node.position)}
    onPaneClick={() => onSelect(null)}
    defaultEdgeOptions={{ focusable: true }}
  >
    <GraphMarkers />

    <CanvasTools onResetPositions={onResetPositions} />
  </ReactFlow>;
}

function InfoPanel({ open, onClose, palette }) {
  if (!open) return null;
  return <aside className="info-panel overlay-panel" data-liquid-glass="panel" role="dialog" aria-label="图例与证据说明">
    <header><span>READING KEY</span><button type="button" onClick={onClose} aria-label="关闭说明"><Icon name="close" /></button></header>
    <h2>{sample.label} · 编译过程</h2>
    <p>{isQk?'QK / PV 计算经编译拆分至 AIC 与 AIV；主图展示计算、跨核交接和各空间存储。':'所选 callable 在前端 SPMD scope 中完成分块合并、归一化、RoPE、打包与发布，随后下降为单 AIV 和 UB 存储。'}</p>
    <p>{isQk?'选择计算或存储对象查看绑定关系，最终 ISA 可展开逐条指令。':'默认按区域显示实际算子类型与次数，以及循环边界。双击区域或算子组可展开逐语句算子；选择后高亮对应 UB 范围。'}</p>
    <div className="legend-grid">
      <span><i className="legend-swatch tensor" />Tensor / Tile</span>
      {palette === 'semantic' && <span><i className="legend-swatch" style={{background:'#F2C45F'}} />归约统计量 · mᵢ / ℓᵢ</span>}
      <span><i className="legend-swatch communication" />通信 · 发布／通知</span>
      <span><i className="legend-swatch op" />计算区域 / IR OP</span>
      <span><i className="legend-swatch task" />Task · 调度任务</span>
      <span><i className="legend-swatch isa" />PTO ISA · 指令组</span>
      <span><i className="legend-swatch control-scope" />Control · 循环／条件作用域</span>
      <span><MemoryLegend />MemRef / buffer</span>
      <span><MemoryLegend highlighted />内存高亮</span>
    </div>
    <div className="relation-list">
      <span><i className="key-line data" />数据依赖</span>
      <span><i className="key-line control" />控制／交接</span>
      <span><i className="key-line binding" />存储绑定</span>
      <span><i className="key-line lowering" />跨阶段对应</span>
    </div>
    {!isQk&&<p>区域按样本配置；循环与条件见详情。</p>}
    <p data-liquid-muted>数据来自本地编译产物；指令数为静态计数，展开可查看输入、输出和源码。</p>
    <div className="info-document-links"><a href="./docs/callable-explorer-rules.html" target="_blank" rel="noopener noreferrer">通用化规则文档 ↗</a><a href="./docs/cross-callable-validation.html" target="_blank" rel="noopener noreferrer">跨 callable 验证报告 ↗</a></div>
    <a href="https://github.com/hw-native-sys/pto-isa" target="_blank" rel="noreferrer">PTO-ISA source repository ↗</a>
  </aside>;
}

function PassRail({ currentIndex, onSelect, playing, onTogglePlay, onReset }) {
  const [labelLeft, setLabelLeft] = useState(0);
  const [expandedRuns, setExpandedRuns] = useState(new Set());
  const groups = useMemo(() => {
    const runs = new Set(expandedRuns);
    groupPassesForRail(runs).forEach((group) => group.items.forEach((item) => {
      if (item.type === 'noop-run' && item.passes.some((pass) => pass.index === currentIndex)) runs.add(item.key);
    }));
    return groupPassesForRail(runs);
  }, [expandedRuns, currentIndex]);
  const railRef = useRef(null);
  useEffect(() => {
    const rail = railRef.current;
    const update = () => {
      const active = rail?.querySelector(`[data-pass-index="${currentIndex}"]`);
      if (active) setLabelLeft(active.getBoundingClientRect().left + active.offsetWidth / 2 - rail.parentElement.getBoundingClientRect().left);
    };
    const observer = new ResizeObserver(update);
    observer.observe(rail);
    rail.addEventListener('scroll', update);
    update();
    return () => { observer.disconnect(); rail.removeEventListener('scroll', update); };
  }, [currentIndex, expandedRuns]);
  useEffect(() => {
    const active = railRef.current?.querySelector(`[data-pass-index="${currentIndex}"]`);
    active?.scrollIntoView({ block: 'nearest', inline: 'center', behavior: 'instant' });
  }, [currentIndex, expandedRuns]);

  const toggleRun = (key) => setExpandedRuns((value) => {
    const next = new Set(value);
    if (next.has(key)) next.delete(key); else next.add(key);
    return next;
  });

  return <div className="bottom-navigation"><CallableSelector/><nav className="pass-rail" aria-label="51 Pass 导航">
    <span className="current-pass-label" style={{ left: labelLeft }}>{currentIndex === resultIndex ? resultStage.name : passes.passes[currentIndex - 1].name}</span>
    <div className="rail-surface" data-liquid-glass="control" aria-hidden="true" />
    <button type="button" className="rail-play" onClick={onTogglePlay} aria-label={playing ? '暂停 Pass 播放' : '播放 Pass'} aria-pressed={playing}><Icon name={playing ? 'pause' : 'play'} /></button>
    <button type="button" className="rail-play rail-reset" onClick={onReset} aria-label="重置播放，回到第一个 Pass" data-tooltip="重置播放"><Icon name="reset" /></button>
    <div className="rail-window" ref={railRef}>
      <div className="rail-scroll">
        {groups.map((group) => <div className="phase-group" key={group.segmentKey}>
          <div className="phase-name"><span>{group.label}</span></div>
          <div className="phase-passes">
            {group.items.map((item) => item.type === 'noop-run'
              ? <button key={item.key} type="button" className="noop-run" onClick={() => toggleRun(item.key)} data-tooltip={item.passes.map((pass) => pass.name).join('\n')} aria-label={`展开 ${item.passes.length} 个 no-op Pass`}>
                  <span>{item.passes.length}×</span>
                </button>
              : <button
                  key={item.pass.id}
                  type="button"
                  data-pass-index={item.pass.index}
                  className={`pass-tick scope-${passScope(item.pass)} ${currentIndex === item.pass.index ? 'is-current' : ''}`}
                  onClick={() => onSelect(item.pass.index)}
                  data-tooltip={`${String(item.pass.index).padStart(2, '0')} ${item.pass.name}\n${passScopeLabel(item.pass)}\n${item.pass.summary}`}
                  aria-label={`Pass ${item.pass.index} ${item.pass.name} · ${passScopeLabel(item.pass)}`}
                  aria-current={currentIndex === item.pass.index ? 'step' : undefined}
                ><span>{String(item.pass.index).padStart(2, '0')}</span><small>{item.pass.name}</small></button>)}
          </div>
        </div>)}
        <div className="phase-group"><div className="phase-name"><span>最终产物</span></div><div className="phase-passes"><button type="button" data-pass-index={resultIndex} className={`pass-tick ${currentIndex === resultIndex ? 'is-current' : ''}`} onClick={() => onSelect(resultIndex)} aria-label="查看最终 PTO-ISA 产物" data-tooltip="PTOCodegen · 最终指令产物"><span>ISA</span></button></div></div>
      </div>
    </div>
  </nav></div>;
}

function SourceRefs({ refs = [] }) {
  if (!refs.length) return <span className="muted">来源位置待补充</span>;
  return <ul className="source-refs">{refs.slice(0, 4).map((source, index) => <li key={`${source.file}:${source.line || index}`}><code>{source.file}{source.line ? `:${source.line}` : ''}</code><span>{source.evidence}</span></li>)}</ul>;
}

function Inspector({ open, side, currentPass, selectedObject, objectActive, showIsa, onShowIsa, onJumpFinal, onClose }) {
  if (!open) return null;
  const phase = phaseById.get(currentPass.phase);
  const instructionRows = isa.instructions.filter(item => item.occurrences.some(o=>o.owner===selectedObject?.id));
  const before = currentPass.metrics.before;
  const after = currentPass.metrics.after;
  const metricRows = [
    ['callables', 'Callable'], ['statements', '语句'], ['loops', 'Loop'], ['allocations', 'Alloc'], ['tasks', '内嵌 Task'],
  ].filter(([key]) => before[key] !== after[key]);
  return <aside className={`inspector overlay-panel inspector-${side}`} role="dialog" aria-label="Pass 与对象检查器">
    <header className="inspector-header"><span>INSPECTOR</span><button type="button" onClick={onClose} aria-label="关闭检查器"><Icon name="close" /></button></header>
    <section className="pass-card">
      <div className="pass-index"><span>{showIsa ? 'ISA' : String(currentPass.index).padStart(2, '0')}</span><i>{showIsa ? '最终产物' : '/ 51'}</i></div>
      <span className="eyebrow">{phase?.label} · {currentPass.lens}</span>
      <h2>{currentPass.name}</h2>
      <p>{currentPass.summary}</p>

    </section>

    {selectedObject && <section className="object-block">
      <header><span>所选对象</span><small>{selectedObject.semanticKind}</small></header>
      <div className="object-title"><i className={`object-icon kind-${selectedObject.renderKind}`}>{selectedObject.renderKind === 'op' && <img src={fxIcon} alt="" aria-hidden="true" />}</i><div><h3>{selectedObject.label}</h3></div></div>
      {!objectActive && <div className="tombstone">已保留跨阶段选择；此处展示对象的登记信息。</div>}
      <p>{selectedObject.role}</p>
      {selectedObject.sourceText && <details><summary>指令源码</summary><pre className="instruction-source">{selectedObject.sourceText}</pre></details>}
      {selectedObject.regionOperations && <details><summary>区域内算子 · {selectedObject.attributes.operations} 处</summary>{selectedObject.regionOperations.map((item,index)=><div key={index} className="region-source-row"><small>L{item.line} · {item.operations.join(" · ")}</small><code>{item.value}</code><pre className="instruction-source">{item.expression}</pre></div>)}</details>}
      <details><summary>对象属性</summary><dl className="object-data">{Object.entries(selectedObject.attributes || {}).slice(0, 7).flatMap(([key, value]) => [<dt key={`${key}-k`}>{key}</dt>, <dd key={`${key}-v`}>{Array.isArray(value) ? value.join(' × ') : String(value)}</dd>])}</dl></details>
      {selectedObject.sourceFacts && <details><summary>源码结构与属性</summary>
        {selectedObject.sourceFacts.loops?.length > 0 && <><p>源码包含 {selectedObject.sourceFacts.loops.length} 个静态循环结构；迭代范围见各循环标签。</p><ul>{selectedObject.sourceFacts.loops.map((loop, index) => <li key={index}><code>{loop.target} · {loop.iterator}</code>{loop.stage != null ? ` · pipeline stage ${loop.stage}` : ''}</li>)}</ul></>}
        <dl className="object-data">{Object.entries(selectedObject.sourceFacts).filter(([key]) => key !== 'loops').flatMap(([key, value]) => [<dt key={key}>{key}</dt>, <dd key={`${key}-value`}>{JSON.stringify(value)}</dd>])}</dl>
      </details>}
      {instructionRows.length > 0 && <div className="isa-occurrences">
        <p>数量统计 {sample.label}.pto 中的静态指令位置；连线表达按所属 Kernel、操作类型与源码位置建立的对应关系。</p>
        {instructionRows.map((instruction) => <details key={instruction.id}>
          <summary>{instruction.mlirName} · {instruction.occurrenceCount} 处源码</summary>
          {instruction.occurrences.map((occurrence) => <div key={occurrence.line}><strong>{occurrence.kernel} · L{occurrence.line}</strong><code>{occurrence.text}</code></div>)}
        </details>)}
      </div>}
    </section>}

    <section className="contract-block">
      <details><summary>Pass 语义与验证</summary>
      <dl>
        <dt>Pass 意图</dt><dd>{currentPass.contract.intent}</dd>
        <dt>前置性质</dt><dd>{currentPass.contract.precondition}</dd>
        <dt>产生性质</dt><dd>{currentPass.contract.produces}</dd>
        <dt>失效性质</dt><dd>{currentPass.contract.invalidates}</dd>
        <dt>验证</dt><dd>{currentPass.contract.validation}</dd>
      </dl></details>
    </section>

    {metricRows.length > 0 && <section className="metric-block">
      <header><span>发生变化的指标</span><small>before → after</small></header>
      <div className="metric-grid">{metricRows.map(([key, label]) => <div key={key}><span>{label}</span><strong>{before[key]}<i>→</i>{after[key]}</strong></div>)}</div>
    </section>}

    <section className="evidence-block">
      <details><summary>源码与运行范围</summary>
      <SourceRefs refs={currentPass.sourceRefs} />
      {!showIsa && currentPass.programEvidence && <details className="program-evidence"><summary>变化位于完整 CSA · 查看前后源码变化</summary><p>{sample.label} family 保持不变。以下摘录两侧独有行及相邻上下文，保留原始行号；并非完整逐行 diff。</p>{Object.entries(currentPass.programEvidence).map(([side, source]) => <details key={side}><summary>{side === 'before' ? '之前' : '之后'} · {source.file.split('/').at(-1)}</summary><pre className="instruction-source">{source.text}</pre></details>)}</details>}
      <details><summary>运行范围与版本</summary><p>当前查看共享编译结构；完整运行 {manifest.deployment.ranks.length} ranks。</p>{manifest.deployment.ranks.map(rank => <p key={rank.rankId}>{rank.rankId} / {rank.dispatchId} · 资源清单 AIC {rank.cores.aic} / AIV {rank.cores.aiv} · {rank.recordCount} 条执行记录</p>)}<p>资源清单不代表所选任务用核数；任务用核尚未关联。</p><code>{manifest.buildIdentity.runId}</code><p>生成器 {manifest.buildIdentity.generatorSha256.slice(0, 12)}</p></details>
      </details>
      {!showIsa && <button className="isa-toggle" type="button" onClick={onJumpFinal}>查看最终 PTO-ISA 产物</button>}
      {showIsa && <div className="isa-summary"><b>{isa.instructions.reduce((sum, item) => sum + item.occurrenceCount, 0)}</b><span>selected static occurrences</span><small>{isa.source.pto}</small></div>}
    </section>
  </aside>;
}

function CanvasTooltip() {
  const [tip, setTip] = useState(null);
  useEffect(() => {
    const show = (event) => {
      const target = event.target.closest?.('[data-tooltip]') || event.target.querySelector?.('[data-tooltip]');
      const text = target?.getAttribute('data-tooltip');
      if (!text || text.trim() === target.textContent.trim()) { setTip(null); return; }
      const rect = target.getBoundingClientRect();
      setTip({ text, x: Math.max(12, Math.min(rect.left, window.innerWidth - 352)), y: Math.max(12, Math.min(rect.bottom + 10, window.innerHeight - 180)) });
    };
    const hide = () => setTip(null);
    document.addEventListener('pointerover', show);
    document.addEventListener('focusin', show);
    document.addEventListener('pointerdown', hide);
    document.addEventListener('focusout', hide);
    document.addEventListener('pointerout', hide);
    window.addEventListener('blur', hide);
    return () => {
      document.removeEventListener('pointerover', show);
      document.removeEventListener('focusin', show);
      document.removeEventListener('pointerdown', hide);
      document.removeEventListener('focusout', hide);
      document.removeEventListener('pointerout', hide);
      window.removeEventListener('blur', hide);
    };
  }, []);
  return tip && <div className="canvas-tooltip" role="tooltip" style={{left:tip.x,top:tip.y}}>{tip.text}</div>;
}

const callableSources=Object.entries(import.meta.glob('../data/callable-probe/snapshots/*.py',{query:'?raw',import:'default',eager:true})).sort(([a],[b])=>a.localeCompare(b)).map(([,text])=>text);
function SourceChange({index}){
 const before=(callableSources[index-1]||'').split('\n'),after=(callableSources[index]||'').split('\n');
 const oldLines=new Set(before),newLines=new Set(after);
 const removed=before.filter(line=>line.trim()&&!newLines.has(line)),added=after.filter(line=>line.trim()&&!oldLines.has(line));
 return <details className="source-change"><summary>{removed.length||added.length?`目标源码变化 · +${added.length} / −${removed.length} 行`:'目标源码未变化'}</summary><pre>{removed.map(line=>'− '+line).concat(added.map(line=>'+ '+line)).join('\n')||'本步没有改变目标 callable/scope。'}</pre></details>;
}

function App() {
  const initial = useMemo(initialUrlState, []);
  useEffect(()=>{document.title=`Callable Explorer · ${sample.label}`;},[]);
  const [passIndex, setPassIndex] = useState(initial.passIndex);

  const [settingsOpen, setSettingsOpen] = useState(false);
  const [contextOpen, setContextOpen] = useState(() => new URLSearchParams(location.hash.slice(1)).get('view') === 'deployment');
  const [ranksExpanded, setRanksExpanded] = useState(() => new URLSearchParams(location.hash.slice(1)).get('view') === 'deployment' || new URLSearchParams(location.hash.slice(1)).get('ranks') === 'expanded');
  const [deployment, setDeployment] = useState(false);
  const [selectedRank, setSelectedRank] = useState(null);
  const [deploymentSelection, setDeploymentSelection] = useState(null);
  const [palette, setPalette] = useState(() => new URLSearchParams(location.hash.slice(1)).get('palette') === 'semantic' ? 'semantic' : 'identity');
  const [layout, setLayout] = useState(() => new URLSearchParams(location.hash.slice(1)).get('layout') === 'vertical' ? 'vertical' : 'horizontal');
  const [playing, setPlaying] = useState(false);
  const [selectedId, setSelectedId] = useState(initial.selectedId);
  const [hoverId, setHoverId] = useState(null);
  const showIsa = passIndex === resultIndex;
  const [infoOpen, setInfoOpen] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [positionOverrides, setPositionOverrides] = useState({});
  const [loadExpanded, setLoadExpanded] = useState(false);
  const [expandedRegion,setExpandedRegion]=useState(null);
  const [memoryCursor,setMemoryCursor]=useState({pass:null,step:-1});
  const allocationRows=memoryRows(passIndex);
  const memoryStep=memoryCursor.pass===passIndex?memoryCursor.step:-1;
  const memoryPass=allocationRows.length>0;
  const [memoryPlaying,setMemoryPlaying]=useState(false);
  useEffect(()=>{setMemoryPlaying(false);setMemoryCursor({pass:passIndex,step:-1});},[passIndex]);
  useEffect(()=>{
    if(!playing)return;
    const action=nextPassAction(passIndex,playbackSteps);
    if(action.type==='stop'){setPlaying(false);return;}
    const timer=setTimeout(()=>setPassIndex(action.pass),action.delay);
    return ()=>clearTimeout(timer);
  },[playing,passIndex]);
  useEffect(()=>{
    if(!memoryPlaying)return;
    if(memoryStep+1>=allocationRows.length){setMemoryPlaying(false);return;}
    const timer=setTimeout(()=>setMemoryCursor({pass:passIndex,step:memoryStep+1}),200);
    return ()=>clearTimeout(timer);
  },[memoryPlaying,memoryStep,passIndex,allocationRows.length]);
  const loadData = useMemo(() => ({
    expandedRegion,
    memoryReplayId:memoryPass&&memoryStep>=0?'allocation.'+allocationRows[memoryStep]?.id:null,
    memoryPlayback:memoryPass?{step:memoryStep,total:allocationRows.length,label:allocationRows[memoryStep]?.base,playing:memoryPlaying,
      toggle:()=>{setPlaying(false);if(memoryStep>=allocationRows.length-1)setMemoryCursor({pass:passIndex,step:-1});setMemoryPlaying(value=>!value);},
      reset:()=>{setMemoryPlaying(false);setMemoryCursor({pass:passIndex,step:-1});}}:null,
    onToggle: (id) => { setLoadExpanded((value) => value === id ? false : id); setInspectorOpen(false); setSelectedId(null); setHoverId(null); },
    onInstanceSelect: (instance) => { setSelectedId(instance.id); setInspectorOpen(true); setSettingsOpen(false); setInfoOpen(false); },
  }), [expandedRegion,memoryPass,memoryStep,allocationRows,memoryPlaying,passIndex]);
  const currentPass = showIsa ? resultStage : passes.passes[passIndex - 1];
  const domain = useMemo(() => buildDomainState(passIndex), [passIndex]);
  const flow = useMemo(() => toReactFlow(domain, { currentPass, selectedId, hoverId, showIsa, positionOverrides, layout, loadExpanded, loadData }), [domain, currentPass, selectedId, hoverId, showIsa, positionOverrides, layout, loadExpanded, loadData]);
  const displayFlow = flow;
  const selectedMemory = memoryRows(passIndex).find(row => `allocation.${row.id}` === selectedId);
  const selectedInstance = allInstances.find((item) => item.id === selectedId);
  const selectedObject = selectedInstance ? {
    id: selectedInstance.id, label: `${selectedInstance.label} · L${selectedInstance.line}`, semanticKind: 'instruction', renderKind: 'isa',
    role: `${selectedInstance.kernel} 内的一处静态指令`, sourceText: selectedInstance.text,
    attributes: { input: selectedInstance.input, output: selectedInstance.output, shape: selectedInstance.shape, validShape: selectedInstance.validShape, dtype: selectedInstance.dtype, space: selectedInstance.space },
  } : selectedId ? domain.registry.get(selectedId) || flow.nodes.find(n=>n.id===selectedId)?.data : null;
  const objectActive = selectedInstance ? showIsa : selectedId ? flow.nodes.some((item) => item.id === selectedId) : false;
  const phase = phaseById.get(currentPass.phase);

  const selectPass = useCallback((next) => {
    const target = clamp(next, 1, resultIndex);
    setMemoryCursor({pass:null,step:-1});
    setPassIndex(target);
  }, []);

  const selectNode = useCallback((id) => {
    setSelectedId(id);
    if (id) { setInspectorOpen(id.startsWith('allocation.')); setSettingsOpen(false); setInfoOpen(false); }
    else {
      setLoadExpanded(false);
      setInspectorOpen(false);
      setHoverId(null);
    }
  }, []);

  useEffect(() => {
    const params = new URLSearchParams({ pass: String(Math.min(passIndex, manifest.passCount)) });
    params.set('callable',sample.id);
    if (selectedId) params.set('node', selectedId);
    if (showIsa) params.set('stage', 'isa');
    params.set('palette', palette);
    params.set('layout', layout);
    params.set('view', 'compile');
    if(ranksExpanded)params.set('ranks','expanded');else params.delete('ranks');
    window.history.replaceState(null, '', `#${params}`);
  }, [passIndex, selectedId, showIsa, palette, layout, deployment, ranksExpanded]);

  useEffect(() => {
    const onKey = (event) => {
      const target = event.target;
      if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target?.tagName) || target?.isContentEditable) return;
      if (!deployment && event.key === 'ArrowRight') { event.preventDefault(); selectPass(passIndex + 1); }
      if (!deployment && event.key === 'ArrowLeft') { event.preventDefault(); selectPass(passIndex - 1); }
      if (event.key === 'Escape') { setInfoOpen(false); setInspectorOpen(false); setSettingsOpen(false); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [passIndex, selectPass, deployment]);

  const jumpFinal = () => {
    setPassIndex(resultIndex);
  };

  const selectedFlowNode = flow.nodes.find((node) => node.id === selectedId);
  const flowCenter = flow.nodes.length
    ? (Math.min(...flow.nodes.map((node) => node.position.x)) + Math.max(...flow.nodes.map((node) => node.position.x + Number(node.style.width)))) / 2
    : 0;
  const inspectorSide = 'right';

  return <DiagramSettings.Provider value={{palette, layout}}><main className={`app-shell palette-${palette} layout-${layout}`}>
    <section className="graph-stage" aria-label="CSA csa_merge_pack_publish Pass DAG">
      <ReactFlowProvider>
        <GraphCanvas key={`${layout}-${deployment}-${passIndex >= 33 && !showIsa}`}
          nodes={displayFlow.nodes}
          edges={displayFlow.edges}
          showIsa={showIsa}
          loadExpanded={loadExpanded}
          onSelect={id => { const [rank, staticId] = id?.split('::') || []; if(deployment)setSelectedRank(rank); setDeploymentSelection(deployment ? displayFlow.nodes.find(n => n.id === id)?.data || null : null); if (!deployment) selectNode(id); }}
          onExpandRegion={id=>setExpandedRegion(value=>value===id?null:id)}
          onHover={id => setHoverId(deployment ? id?.split('::')[1] || null : id)}
          onMoveNode={(id, position) => setPositionOverrides((value) => ({ ...value, [id]: position }))}
          onResetPositions={() => setPositionOverrides({})}
        />
      </ReactFlowProvider>

      <header className="focus-heading">
        <span className="eyebrow">CSA PASS FIELD · {sample.label}</span>
        <div className="focus-title"><strong>{deployment ? '2R' : showIsa ? 'ISA' : String(currentPass.index).padStart(2, '0')}</strong><div><h1>{deployment ? 'CSA 双卡部署' : currentPass.name}</h1><p>{deployment ? '静态编排 · 独立于 Pass 进度' : showIsa ? 'PTOCodegen · 代码生成' : `${phase?.label} · ${currentPass.lens}`}</p></div></div>
        {!isQk&&<SourceChange index={passIndex}/>}<div className="focus-state"><p>{deployment ? '双 rank 运行配置 + CSA 源码通信关系' : currentPass.contract.intent}</p><p>{deployment ? `双卡部署 · Mock${selectedRank ? ` · ${selectedRank}` : ""}` : `${selectedRank ? selectedRank + ' / csa_merge_pack_publish · ' : ''}${isQk?'AIC + AIV · qk_pv':'单 AIV callable · 48 workers'}`}</p></div>
      </header>


      {deployment && <div className="deployment-note">虚线：跨 rank 静态通信 · 实线：阶段顺序 · Mock 不表示执行时间</div>}
      <div className="top-actions">
        <button type="button" data-liquid-glass="icon" aria-label="页面设置" data-tooltip="页面设置" aria-expanded={settingsOpen} onClick={() => {setSettingsOpen(v=>!v);setInfoOpen(false);setInspectorOpen(false);}}><Icon name="settings" /></button>
        <button type="button" data-liquid-glass="icon" aria-expanded={infoOpen} onClick={() => { setInfoOpen((value) => !value); setInspectorOpen(false); setSettingsOpen(false); }} aria-label="图例与说明" data-tooltip="图例与说明"><Icon name="info" /></button>
        <button type="button" data-liquid-glass="icon" aria-expanded={inspectorOpen} onClick={() => { setInspectorOpen((value) => !value); setInfoOpen(false); setSettingsOpen(false); }} aria-label="Pass 详情" data-tooltip="Pass 详情"><Icon name="detail" /></button>
      </div>

      <div className="relation-key" aria-label="关系图例">
        <span><i className="key-line data" />DATA</span>
        <span><i className="key-line control" />CONTROL</span>
        <span><i className="key-line binding" />BINDING</span>
      </div>

      <InfoPanel open={infoOpen} palette={palette} onClose={() => setInfoOpen(false)} />
      {settingsOpen && <aside className="info-panel overlay-panel settings-panel" role="dialog" aria-label="页面设置"><header><span>页面设置</span><button onClick={()=>setSettingsOpen(false)} aria-label="关闭设置">×</button></header><h3>染色方案</h3><div className="settings-options">{[['identity','对象类别'],['semantic','语义角色']].map(([id,label])=><button key={id} aria-pressed={palette===id} onClick={()=>setPalette(id)}>{label}</button>)}</div><p>{palette==='identity'?'浅蓝 Tensor · 紫色计算 · 青绿通信 · 黄绿 Task / ISA · 蓝色存储 · 灰色控制':'Tensor 按激活绿／统计黄区分；ISA 继承计算紫／通信青绿／访存蓝；Task 与控制采用灰色。'}</p><h3>布局方向</h3><div className="settings-options">{[['horizontal','横向'],['vertical','纵向']].map(([id,label])=><button key={id} aria-pressed={layout===id} onClick={()=>{setPositionOverrides({});setLayout(id);}}>{label}</button>)}</div></aside>}
      {deployment && deploymentSelection && <aside className="inspector overlay-panel inspector-right" role="dialog" aria-label="部署阶段"><header className="inspector-header"><span>{deploymentSelection.rankId}</span><button onClick={()=>setDeploymentSelection(null)} aria-label="关闭部署详情">×</button></header><h2>{deploymentSelection.label}</h2><p>{deploymentSelection.detail}</p><p>静态编排关系 Mock；未展开迭代、信号协议与动态任务，不作为执行 DAG。</p><p>decode_csa.py · L{deploymentSelection.line}</p>{deploymentSelection.stageId==='qkpv' && <button className="isa-toggle" onClick={()=>{setDeployment(false);setDeploymentSelection(null);setPassIndex(23);}}>查看 csa_merge_pack_publish 编译结构 · Pass 23</button>}</aside>}
      {inspectorOpen && selectedMemory && <MemoryDetails row={selectedMemory} passIndex={passIndex} onClose={()=>setInspectorOpen(false)} />}
      <Inspector
        open={inspectorOpen && !deployment && !selectedMemory}
        side={inspectorSide}
        currentPass={currentPass}
        selectedObject={selectedObject}
        objectActive={objectActive}
        showIsa={showIsa}
        onJumpFinal={jumpFinal}
        onClose={() => setInspectorOpen(false)}
      />
      {!deployment && <PassRail currentIndex={passIndex} onReset={() => { setMemoryPlaying(false); setPlaying(false); setMemoryCursor({pass:null,step:-1}); selectPass(1); }} onSelect={(index) => { setPlaying(false); selectPass(index); }} playing={playing} onTogglePlay={() => {
        setMemoryPlaying(false);
        if (!playing && !playbackSteps.includes(passIndex)) setPassIndex(playbackSteps.find(index => index > passIndex) || playbackSteps[0]);
        else if (!playing && passIndex === resultIndex) setPassIndex(playbackSteps[0]);
        setPlaying((value) => !value);
      }} />}
    </section>
  <CanvasTooltip /></main></DiagramSettings.Provider>;
}

export default App;
