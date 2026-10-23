import { Fragment, memo, useState, createContext, useContext } from 'react';
export const DiagramSettings = createContext({ palette: 'identity', layout: 'horizontal' });
import { tensorGeometry } from './tensor-glyph.js';
import fxIcon from './vendor/fx.svg';
import memrefSource from './vendor/system-memref.svg?raw';
const memrefPlate = memrefSource.match(/<g data-object="buffers"[\s\S]*?<\/g>/)[0].replace(/<text[\s\S]*?<\/text>/g, '');
import {
  BaseEdge,
  EdgeLabelRenderer,
  Handle,
  Position,
} from 'reactflow';

function Ports({ target = Position.Left, source = Position.Right, memory = true }) {
  const vertical = useContext(DiagramSettings).layout === 'vertical';
  const cardinal=<>{[['left',Position.Left],['right',Position.Right],['top',Position.Top],['bottom',Position.Bottom]].map(([name,position])=><Fragment key={name}><Handle type="source" position={position} id={`${name}-out`} className="port"/><Handle type="target" position={position} id={`${name}-in`} className="port"/></Fragment>)}</>;
  if (vertical) {
    const rotate = p => ({ left: Position.Top, right: Position.Bottom, top: Position.Left, bottom: Position.Right })[p];
    return <>{cardinal}<Handle type="source" position={Position.Right} id="side-out" className="port" /><Handle type="target" position={Position.Left} id="side-in" className="port" /><Handle type="target" position={rotate(target)} id="in" className="port" /><Handle type="target" position={rotate(target)} id="lhs" className="port" /><Handle type="target" position={rotate(target)} id="rhs" className="port" style={{left:'80%'}} /><Handle type="source" position={rotate(source)} id="out" className="port" />{memory && <><Handle type="target" position={Position.Right} id="memory" className="port" /><Handle type="source" position={Position.Left} id="memory" className="port" /></>}</>;
  }
  return <>{cardinal}
    <Handle type="target" position={target} id="in" className="port" />
    <Handle type="target" position={target} id="lhs" className="port port-lhs" />
    <Handle type="target" position={target} id="rhs" className="port port-rhs" />
    {memory && <Handle type="target" position={Position.Top} id="memory" className="port" />}
    <Handle type="source" position={source} id="out" className="port" />
    {memory && <Handle type="source" position={Position.Bottom} id="memory" className="port" />}
  </>;
}

function StateMark({ data }) {
  if (!data.touched || !data.involvement?.length) return null;
  return <span className="object-change-detail">
    <span className="change-tip" role="tooltip"><strong>当前 Pass 涉及这个对象</strong>{data.involvement.map((reason, index) => <span key={index}>{reason}</span>)}</span>
  </span>;
}

function TensorNodeBase({ data, ports = true }) {
  const role = data.attributes?.tensorRole || 'i';
  const color = useContext(DiagramSettings).palette === 'identity' ? '#8ec9f0' : ({ a: '#F2C45F', w: '#BCA0F3', g: '#79D4F2' })[role] || '#B1EF6B';
  const { width, height, detail, columns } = tensorGeometry(data);
  return <div data-tooltip={data.memoryRanges?.join('\n')} style={{ width, height }} className={`glyph tensor-glyph tensor-${role} ${data.selected ? 'is-selected' : ''} ${data.touched ? 'is-touched' : ''}`}>
    {ports && <Ports />}
    <svg className="tile-plate" viewBox={`0 0 ${width} ${height}`} aria-label={data.label}>
      <rect x=".6" y=".6" width={width - 1.2} height={height - 1.2} rx="6" fill="#131313" stroke={color} strokeWidth="1.2" />
      <g className="data-role-grid">
        {Array.from({ length: columns * 4 }, (_, i) => <rect key={i} x={(width - (columns * 14 - 3)) / 2 + i % columns * 14} y={5.5 + Math.floor(i / columns) * 14} width="11" height="11" fill={color} fillOpacity=".22" />)}
      </g>
      <g style={{ paintOrder: 'stroke', stroke: '#131313', strokeWidth: 2.5, strokeLinejoin: 'round' }}>
        <text className="graph-name" x={width / 2} y="27" fill={color} textAnchor="middle">{data.label}</text>
        <text className="graph-meta" x={width / 2} y="48" fill="#ccc" textAnchor="middle">{detail}</text>
      </g>
    </svg>
    {data.semanticKind === 'tensor_value' && <small className="value-domain">Tensor</small>}
    {data.semanticKind === 'tile_value' && <small className="value-domain">Tile{data.sourceFacts?.spaces?.length ? ` · ${data.sourceFacts.spaces.join(' / ')}` : ''}</small>}
    <StateMark data={data} />
  </div>;
}

function OpNodeBase({ data, ports = true }) {
  const displayLabel = data.label;
  return <div data-tooltip={data.memoryRanges?.join('\n')} className={`glyph op-glyph ${data.selected ? 'is-selected' : ''} ${data.touched ? 'is-touched' : ''}`}>
    {ports && <Ports />}
    <img className="op-mark" src={fxIcon} alt="" aria-hidden="true" />
    <strong>{displayLabel}</strong>
    {data.sourceFacts?.version && <small className="op-version">{data.sourceFacts.version}</small>}
    <StateMark data={data} />
  </div>;
}

function HandoffNodeBase({ data, ports = true }) {
  const displayLabel = data.label.replace(/ · slot.*$/, '');
  return <div className={`glyph handoff-glyph ${data.selected ? 'is-selected' : ''} ${data.touched ? 'is-touched' : ''}`}>
    {ports && <Ports />}
    <span className="handoff-mark"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 9H21L16 3M21 15H3L8 21" fill="none" stroke="currentColor" strokeWidth="2.3" strokeLinecap="round" strokeLinejoin="round"/></svg></span>
    <strong>{displayLabel}</strong>
    <StateMark data={data} />
  </div>;
}

function ScopeNodeBase({ data }) {
  return <div className={`glyph scope-glyph ${data.touched ? 'is-touched' : ''}`}>
    <header>{!data.attributes?.hideScopeLabel && <span>{data.attributes?.level || 'IR'} SCOPE</span>}<strong>{data.label}</strong></header>
    {data.bounds && <small className="control-bounds">{data.bounds}</small>}
    <StateMark data={data} />
  </div>;
}

function KernelNodeBase({ data }) {
  return <div className={`glyph kernel-glyph ${data.selected ? 'is-selected' : ''} ${data.touched ? 'is-touched' : ''}`}>
    <Ports target={Position.Top} source={Position.Bottom} memory={false} />
    <header><strong>{data.label}</strong><em>目标 {data.attributes?.target} · {data.attributes?.objectKind || 'Kernel'}</em></header>
    <small>{data.attributes?.detail}</small>
    <StateMark data={data} />
  </div>;
}

function LoopNode({ data }) {
  const condition = data.id === 'region.guard' || data.attributes?.controlKind === 'condition';
  const description = condition ? '条件作用域：满足标签中的条件时，执行框内操作。' : '循环作用域：按标签中的迭代范围重复执行框内操作；每个框对应一个源码循环结构。';
  // Parametric form of approved system-loop.svg: solid left bracket, dashed range.
  return <div className={`glyph loop-region ${data.id === 'region.guard' || data.attributes?.controlKind === 'condition' ? 'condition-region' : ''} ${data.sourceFacts?.expanded ? 'is-unrolled' : ''}`}>
    <svg className="loop-range" width="100%" height="100%"><rect x="1" y="1" width="calc(100% - 2px)" height="calc(100% - 2px)" fill="none" stroke="#8f82d6" strokeWidth="1.5" strokeDasharray="4 5" /></svg>
    <div className="loop-bracket" />
    <header className="control-label" tabIndex={0}><span className="control-caption">{data.label}</span><span className="control-tip" role="tooltip"><strong>{data.label}</strong><span>{description}</span>{data.attributes?.kernel && <span>所属 Kernel：{data.attributes.kernel.replace('kernel.', 'qk_pv_')}</span>}{data.sourceFacts?.lowerBound && <span>首段起点：{data.sourceFacts.lowerBound}</span>}{data.sourceFacts?.flattened && <span>条件表达式已展开为临时值</span>}{data.sourceFacts?.ssa && <span>循环携带值：ℓᵢ → ℓᵢ₊₁；mᵢ → mᵢ₊₁；Oᵢ → Oᵢ₊₁</span>}</span></header>
    {data.id === 'region.guard' && <Handle type="source" position={Position.Bottom} id="out" className="port" style={{ left: 110, top: 44, bottom: 'auto' }} />}
    <StateMark data={data} />
  </div>;
}

function TaskIcon() {
  return <svg viewBox="0 0 36 36" aria-hidden="true">
    <path d="M9 10 H27 M9 18 H20 M9 26 H16 M23 22 L28 26 L23 30" />
  </svg>;
}

function TaskNodeBase({ data }) {
  return <div className={`glyph task-glyph ${data.attributes?.form === 'scope' ? 'task-as-scope' : ''} ${data.selected ? 'is-selected' : ''} ${data.touched ? 'is-touched' : ''}`}>
    <Ports memory={false} />
    <span className="solid-icon"><TaskIcon /></span>
    <strong>{data.label}</strong>
    <StateMark data={data} />
  </div>;
}

function ContextNode({ data }) {
  return <div className="control-context">
    <header>最近一次控制结构改写 · {data.attributes.description}</header>
    <div className="context-columns">{[['之前', data.attributes.beforeRows, data.attributes.beforeSource], ['之后', data.attributes.rows, data.attributes.source]].map(([label, rows, source]) => <section key={label}>
      <strong>{label}</strong><small data-tooltip={source}>{source?.split('/').at(-1)}</small>
      {(rows || []).map((row) => <code key={row.line} data-tooltip={row.text}>L{row.line} {row.text}</code>)}
    </section>)}</div>
  </div>;
}

function MemoryNodeBase({ data }) {
  if(data.spaceContainer){
   const w=data.gridWidth,h=data.gridHeight,step=12,side=9,cols=Math.floor(w/step),rows=Math.floor(h/step);
   return <div className="memory-space"><header style={{height:data.gridTop===64?40:24}}>{data.label}{data.playback&&<span className="memory-playback nodrag nopan" onPointerDown={event=>event.stopPropagation()} onDoubleClick={event=>event.stopPropagation()} onClick={event=>event.stopPropagation()}>
     <span className="memory-playback-label">{data.playback.label||''}</span>
     <button type="button" aria-label={data.playback.playing?'暂停内存分配':'播放内存分配'} aria-pressed={data.playback.playing} onClick={event=>{event.stopPropagation();data.playback.toggle();}}><svg viewBox="0 0 24 24" aria-hidden="true">{data.playback.playing?<path d="M7 5H10V19H7ZM14 5H17V19H14Z" fill="currentColor"/>:<path d="M7 4L21 12L7 20Z" fill="currentColor"/>}</svg></button>
     <span className="memory-playback-progress">{data.playback.step+1}/{data.playback.total}</span>
     <button type="button" aria-label="重置内存分配" onClick={event=>{event.stopPropagation();data.playback.reset();}}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 7A8 8 0 1 1 4 13M7 3V8H12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg></button>
    </span>}</header><svg className="memory-base-grid" style={{top:data.gridTop||36}} width={w} height={h} viewBox={`0 0 ${w} ${h}`} aria-hidden="true">{Array.from({length:cols*rows},(_,i)=><rect key={i} x={i%cols*step+1} y={Math.floor(i/cols)*step+1} width={side} height={side} rx="1.5" fill={(data.highlights||[]).some(r=>i%cols*step+1<r.x+r.width&&i%cols*step+1+side>r.x&&Math.floor(i/cols)*step+1<r.y+r.height&&Math.floor(i/cols)*step+1+side>r.y)?'#0077FF':'none'} stroke={(data.highlights||[]).some(r=>i%cols*step+1<r.x+r.width&&i%cols*step+1+side>r.x&&Math.floor(i/cols)*step+1<r.y+r.height&&Math.floor(i/cols)*step+1+side>r.y)?'#0077FF':'#545454'} strokeWidth="1"/>)}</svg></div>;
  }
  if(data.compactBlock){
   return <div className={`memory-block ${data.shared?'is-shared':''} ${data.touched?'is-active':''} ${data.selected?'is-selected':''}`} data-tooltip={data.description}>
    <Ports target={Position.Top} source={Position.Bottom} memory={false}/><span>{data.base.replace('mem_','')}</span>{data.shared&&<i aria-label="共享分配">≡</i>}
  </div>;
  }

  const label = data.allocation ? data.base : data.id.includes('pipe') ? 'GM pipe buffer' : 'qk_pv buffers';
  return <div className={`glyph memory-glyph ${data.selected ? 'is-selected' : ''}`} data-tooltip={data.label}>
    <Ports target={Position.Top} source={Position.Bottom} memory={false} />
    <MemoryNodeBase data={{spaceContainer:true,label,gridWidth:268,gridHeight:54,highlights:[]}} />
  </div>;

}

export function MemoryLegend({ highlighted = false }) {
  return <svg className={`memory-legend ${highlighted ? 'is-highlighted' : ''}`} viewBox="20 35 300 106" aria-hidden="true"><g dangerouslySetInnerHTML={{__html:memrefPlate}} /></svg>;
}

function TerminalIcon() {
  return <svg viewBox="0 0 36 36" aria-hidden="true">
    <path d="M9 9 L16 17 L9 25 M20 25 H29" />
  </svg>;
}

function InstructionCollection({ data }) {
  const [active, setActive] = useState(null);
  const instance = data.instances.find(item => item.id === active) || data.instances[0];
  return <div className={`instruction-collection isa-role-${/PUSH|POP|FREE/.test(data.label) ? 'communication' : /LOAD|STORE/.test(data.label) ? 'memory' : 'compute'} ${data.expanded ? 'is-expanded' : ''}`}>
    <Ports target={Position.Top} source={Position.Right} memory={false} />
    <button className="glyph isa-glyph collection-toggle nodrag nopan" aria-expanded={data.expanded} aria-label={`${data.expanded ? '收起' : '展开'} ${data.label} ${data.instances.length} 处静态指令`} onClick={event => { event.stopPropagation(); data.onToggle(); }}>
      <span className="solid-icon"><TerminalIcon /></span><span className="collection-heading"><strong>{data.label} ×{data.instances.length}</strong><small>{data.ownerLabel}</small></span><span className="collection-plus">{data.expanded ? '−' : '+'}</span>
    </button>
    {data.expanded && <div className="collection-body">
      <div className="instruction-grid">{data.instances.map(item => <button key={item.id} data-instance={item.id} className={`glyph isa-glyph nodrag nopan ${instance.id === item.id ? 'is-selected' : ''}`} onMouseEnter={() => setActive(item.id)} onFocus={() => setActive(item.id)} onClick={event => { event.stopPropagation(); setActive(item.id); data.onInstanceSelect(item); }}>
        <span className="solid-icon"><TerminalIcon /></span><strong>{data.label}</strong><small>L{item.line}</small>
      </button>)}</div>
      <div className="instruction-operands">
        {!!instance.inputs?.length && <><div className="operand-tiles">{instance.inputs.map((operand, index) => <div className="operand-tile" key={index} data-tooltip={instance.input}><TensorNodeBase ports={false} data={{label: operand.label, attributes: operand}} /></div>)}</div><span>→</span></>}
        <div className="glyph isa-glyph operand-instruction"><span className="solid-icon"><TerminalIcon /></span><strong>{data.label}</strong></div>
        {!!instance.outputs?.length && <><span>→</span><div className="operand-tiles">{instance.outputs.map((operand,index) => <div className="operand-tile" key={index} data-tooltip={instance.output}><TensorNodeBase ports={false} data={{label: operand.label, attributes: operand}} /></div>)}</div></>}
      </div>
      <small>每块对应一处静态指令；点击查看操作数与源码。</small>
    </div>}
  </div>;
}

export const TensorNode = memo(TensorNodeBase);
export const OpNode = memo(OpNodeBase);
export const HandoffNode = memo(HandoffNodeBase);
export const ScopeNode = memo(ScopeNodeBase);
export const KernelNode = memo(KernelNodeBase);
export const TaskNode = memo(TaskNodeBase);
export const MemoryNode = memo(MemoryNodeBase);
export const IsaNode = memo(InstructionCollection);

export function GraphMarkers() {
  return <svg className="graph-markers" aria-hidden="true">
    <defs>
      <marker id="csa-edge-data" viewBox="0 0 14 14" refX="12" refY="7" markerWidth="10" markerHeight="10" orient="auto">
        <path d="M8 7 H12" stroke="#131313" strokeWidth="2.5" />
        <path d="M3 2 L8 7 L3 12 M7 2 L12 7 L7 12" stroke="#E8E8E8" strokeWidth="1.4" strokeLinejoin="miter" fill="none" />
      </marker>
      <marker id="csa-edge-control" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="10" markerHeight="10" markerUnits="userSpaceOnUse" orient="auto">
        <path d="M1 1 L9 5 L1 9" stroke="#c8c8c8" strokeWidth="1.5" fill="none" />
      </marker>
      <marker id="csa-edge-task" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
        <path d="M1 1 L9 5 L1 9" stroke="#c8c8c8" strokeWidth="1.5" fill="none" />
      </marker>
      <marker id="csa-edge-memory" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="10" markerHeight="10" markerUnits="userSpaceOnUse" orient="auto">
        <circle cx="5" cy="5" r="3" fill="#131313" stroke="#777" />
      </marker>
      <marker id="csa-edge-lowering" viewBox="0 0 12 12" refX="11" refY="6" markerWidth="9" markerHeight="9" orient="auto">
        <path d="M1 6 L6 1 L11 6 L6 11Z" fill="#131313" stroke="#aaa" />
      </marker>
      <marker id="csa-edge-lineage" viewBox="0 0 12 12" refX="11" refY="6" markerWidth="9" markerHeight="9" orient="auto">
        <path d="M1 6 L6 1 L11 6 L6 11Z" fill="#131313" stroke="#777" />
      </marker>
    </defs>
  </svg>;
}

function EdgeBase({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, data, kind }) {
  const [hovered, setHovered] = useState(false);
  const emphasized = (data?.instructionLink || data?.hoverOnly) ? data.hoverFocused : hovered || data?.focused;
  const labelX = (sourceX + targetX) / 2;
  const labelY = (sourceY + targetY) / 2;
  const verticalSource = sourcePosition === Position.Top || sourcePosition === Position.Bottom;
  const verticalTarget = targetPosition === Position.Top || targetPosition === Position.Bottom;
  // Aligned ports need no bend; mixed axes need one; parallel axes need two.
  // Do not insert the extra endpoint stubs used by smooth-step routing.
  const path = data?.routePoints?.length ? data.routePoints.map((p,i)=>`${i?'L':'M'} ${p.x} ${p.y}`).join(' ') : Math.abs(sourceY - targetY) < 0.5 || Math.abs(sourceX - targetX) < 0.5
    ? `M ${sourceX} ${sourceY} L ${targetX} ${targetY}`
    : verticalSource !== verticalTarget
      ? verticalSource
        ? `M ${sourceX} ${sourceY} V ${targetY} H ${targetX}`
        : `M ${sourceX} ${sourceY} H ${targetX} V ${targetY}`
      : verticalSource
        ? `M ${sourceX} ${sourceY} V ${labelY} H ${targetX} V ${targetY}`
        : `M ${sourceX} ${sourceY} H ${labelX} V ${targetY} H ${targetX}`;
  const label = data?.attributes?.label;
  const showLabel = label && emphasized;
  return <g data-edge-highlighted={emphasized ? "true" : undefined} onMouseEnter={() => setHovered(true)} onMouseLeave={() => setHovered(false)}>
    <BaseEdge id={id} path={path} markerEnd={`url(#csa-edge-${kind})`} style={{
      stroke: kind === 'memory' ? emphasized ? '#0077FF' : data?.compact ? '#687c91' : '#353535' : kind === 'control' ? '#b8b8b8' : '#ccc',
      strokeWidth: kind === 'memory' ? (emphasized ? 2.5 : 1.2) : kind === 'control' ? 2 : emphasized ? 2 : 1.3,
      opacity: (data?.instructionLink || data?.hoverOnly) ? emphasized ? 1 : 0 : emphasized ? 1 : kind === 'memory' ? .18 : .10,
      strokeLinejoin: 'round',
      strokeDasharray: kind === 'control' ? '10 6' : kind === 'lowering' ? '3 7' : kind === 'lineage' ? '2 7' : undefined,
    }} />
    {showLabel && <EdgeLabelRenderer>
      <span data-tooltip={data?.attributes?.detail} className={`edge-label edge-label-${kind}`} style={{ transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)` }}>{label}</span>
    </EdgeLabelRenderer>}
  </g>;
}

const makeEdge = (kind) => memo((props) => <EdgeBase {...props} kind={kind} />);
export const DataEdge = makeEdge('data');
export const ControlEdge = makeEdge('control');
export const TaskDependencyEdge = makeEdge('task');
export const MemoryBindingEdge = makeEdge('memory');
export const LoweringEdge = makeEdge('lowering');
export const LineageEdge = makeEdge('lineage');

function RankFrame({ data }) {
  const { rank } = data;
  return <div className={`rank-frame ${data.selected ? 'rank-selected' : ''}`}>
    <header><strong>{rank.rankId.replace('rank', 'Rank ')}</strong><span>完整 CSA · 静态部署骨架</span></header>
    <div className="rank-resources">{['aic', 'aiv'].map(type => <div key={type}><b>{type.toUpperCase()} × {rank.cores[type]}</b><div className="core-inventory">{Array.from({length: rank.cores[type]}, (_, i) => <i key={i} data-tooltip={`${type.toUpperCase()} 资源条目 ${i} · 未指定任务`} />)}</div></div>)}</div>
  </div>;
}

function DeploymentGlyph({data}) {
  const Glyph = data.communication ? HandoffNodeBase : OpNodeBase;
  return <><Glyph data={data} ports={false} />
    <Handle className="port" type="target" position={Position.Top} id="in" /><Handle className="port" type="source" position={Position.Bottom} id="out" />
    {['Left','Right'].map(side => <span key={side}><Handle className="port" type="source" position={Position[side]} id={`${side.toLowerCase()}-out`} style={{top:'35%'}} /><Handle className="port" type="target" position={Position[side]} id={`${side.toLowerCase()}-in`} style={{top:'65%'}} /></span>)}
  </>;
}
export const nodeTypes = {
  deploymentGlyph: memo(DeploymentGlyph),
  rankFrame: memo(RankFrame),
  loop: memo(LoopNode),
  context: memo(ContextNode),
  tensor: TensorNode,
  op: OpNode,
  handoff: HandoffNode,
  scope: ScopeNode,
  kernel: KernelNode,
  task: TaskNode,
  memory: MemoryNode,
  isa: IsaNode,
};

export const edgeTypes = {
  data: DataEdge,
  control: ControlEdge,
  taskDependency: TaskDependencyEdge,
  memoryBinding: MemoryBindingEdge,
  lowering: LoweringEdge,
  lineage: LineageEdge,
};
