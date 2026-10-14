import { memo, useState, createContext, useContext } from 'react';
export const DiagramSettings = createContext({ palette: 'identity', layout: 'horizontal' });
import { tensorGeometry } from './tensor-glyph.js';
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
  if (vertical) {
    const rotate = p => ({ left: Position.Top, right: Position.Bottom, top: Position.Left, bottom: Position.Right })[p];
    return <><Handle type="source" position={Position.Right} id="side-out" className="port" /><Handle type="target" position={Position.Left} id="side-in" className="port" /><Handle type="target" position={rotate(target)} id="in" className="port" /><Handle type="target" position={rotate(target)} id="lhs" className="port" /><Handle type="target" position={rotate(target)} id="rhs" className="port" style={{left:'80%'}} /><Handle type="source" position={rotate(source)} id="out" className="port" />{memory && <><Handle type="target" position={Position.Right} id="memory" className="port" /><Handle type="source" position={Position.Left} id="memory" className="port" /></>}</>;
  }
  return <>
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
  return <div style={{ width, height }} className={`glyph tensor-glyph tensor-${role} ${data.selected ? 'is-selected' : ''} ${data.touched ? 'is-touched' : ''}`}>
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

function OpNodeBase({ data }) {
  const displayLabel = data.label;
  return <div className={`glyph op-glyph ${data.selected ? 'is-selected' : ''} ${data.touched ? 'is-touched' : ''}`}>
    <Ports />
    <span className="op-mark">ƒₓ</span>
    <strong>{displayLabel}</strong>
    {data.sourceFacts?.version && <small className="op-version">{data.sourceFacts.version}</small>}
    <StateMark data={data} />
  </div>;
}

function HandoffNodeBase({ data }) {
  const displayLabel = data.label.replace(/ · slot.*$/, '');
  return <div className={`glyph handoff-glyph ${data.selected ? 'is-selected' : ''} ${data.touched ? 'is-touched' : ''}`}>
    <Ports />
    <span className="handoff-mark">⇄</span>
    <strong>{displayLabel}</strong>
    <StateMark data={data} />
  </div>;
}

function ScopeNodeBase({ data }) {
  return <div className={`glyph scope-glyph ${data.touched ? 'is-touched' : ''}`}>
    <header><span>{data.attributes?.level || 'IR'} SCOPE</span><strong>{data.label}</strong></header>
    <StateMark data={data} />
  </div>;
}

function KernelNodeBase({ data }) {
  return <div className={`glyph kernel-glyph ${data.selected ? 'is-selected' : ''} ${data.touched ? 'is-touched' : ''}`}>
    <Ports target={Position.Top} source={Position.Bottom} memory={false} />
    <header><strong>{data.label}</strong><em>{data.attributes?.target}</em></header>
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
      <strong>{label}</strong><small title={source}>{source?.split('/').at(-1)}</small>
      {(rows || []).map((row) => <code key={row.line} title={row.text}>L{row.line} {row.text}</code>)}
    </section>)}</div>
  </div>;
}

function MemoryNodeBase({ data }) {
  return <div className={`glyph memory-glyph ${data.selected ? 'is-selected' : ''} ${data.touched ? 'is-touched' : ''}`}>
    <Ports target={Position.Top} source={Position.Bottom} memory={false} />
    <svg className="memref-plate" viewBox="20 35 300 106" aria-label={data.label}>
      <g dangerouslySetInnerHTML={{ __html: memrefPlate }} />
      <text className="graph-name" x="44" y="73.16" fill="#fff" dominantBaseline="middle">{data.id.includes('pipe') ? 'GM pipe buffer' : 'qk_pv buffers'}</text>
      <text className="graph-meta" x="44" y="113.44" fill="#eee" dominantBaseline="middle">{data.label.replace(/^qk_pv buffers\s*·?\s*|^GM pipe buffer\s*·?\s*/g, '')}</text>
    </svg>
    <StateMark data={data} />
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
    {data.columnFirst && <span className="instruction-origin">↓ {data.ownerLabel}</span>}
    <Ports target={Position.Top} source={Position.Right} memory={false} />
    <button className="glyph isa-glyph collection-toggle nodrag nopan" aria-expanded={data.expanded} aria-label={`${data.expanded ? '收起' : '展开'} ${data.label} ${data.instances.length} 处静态指令`} onClick={event => { event.stopPropagation(); data.onToggle(); }}>
      <span className="solid-icon"><TerminalIcon /></span><span className="collection-heading"><strong>{data.label} ×{data.instances.length}</strong><small title={data.ownerLabel}>{data.ownerLabel}</small></span><span className="collection-plus">{data.expanded ? '−' : '+'}</span>
    </button>
    {data.expanded && <div className="collection-body">
      <div className="instruction-grid">{data.instances.map(item => <button key={item.id} data-instance={item.id} className={`glyph isa-glyph nodrag nopan ${instance.id === item.id ? 'is-selected' : ''}`} onMouseEnter={() => setActive(item.id)} onFocus={() => setActive(item.id)} onClick={event => { event.stopPropagation(); setActive(item.id); data.onInstanceSelect(item); }}>
        <span className="solid-icon"><TerminalIcon /></span><strong>{data.label}</strong><small>L{item.line}</small>
      </button>)}</div>
      <div className="instruction-operands">
        {!!instance.inputs?.length && <><div className="operand-tiles">{instance.inputs.map((operand, index) => <div className="operand-tile" key={index} title={instance.input}><TensorNodeBase ports={false} data={{label: operand.label, attributes: operand}} /></div>)}</div><span>→</span></>}
        <div className="glyph isa-glyph operand-instruction"><span className="solid-icon"><TerminalIcon /></span><strong>{data.label}</strong></div>
        {!!instance.outputs?.length && <><span>→</span><div className="operand-tiles">{instance.outputs.map((operand,index) => <div className="operand-tile" key={index} title={instance.output}><TensorNodeBase ports={false} data={{label: operand.label, attributes: operand}} /></div>)}</div></>}
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
  const emphasized = hovered || data?.focused;
  const labelX = (sourceX + targetX) / 2;
  const labelY = (sourceY + targetY) / 2;
  const verticalSource = sourcePosition === Position.Top || sourcePosition === Position.Bottom;
  const verticalTarget = targetPosition === Position.Top || targetPosition === Position.Bottom;
  // Aligned ports need no bend; mixed axes need one; parallel axes need two.
  // Do not insert the extra endpoint stubs used by smooth-step routing.
  const path = Math.abs(sourceY - targetY) < 0.5 || Math.abs(sourceX - targetX) < 0.5
    ? `M ${sourceX} ${sourceY} L ${targetX} ${targetY}`
    : verticalSource !== verticalTarget
      ? verticalSource
        ? `M ${sourceX} ${sourceY} V ${targetY} H ${targetX}`
        : `M ${sourceX} ${sourceY} H ${targetX} V ${targetY}`
      : verticalSource
        ? `M ${sourceX} ${sourceY} V ${labelY} H ${targetX} V ${targetY}`
        : `M ${sourceX} ${sourceY} H ${labelX} V ${targetY} H ${targetX}`;
  const label = data?.attributes?.label;
  const showLabel = label && (kind === 'task' || emphasized);
  return <g onMouseEnter={() => setHovered(true)} onMouseLeave={() => setHovered(false)}>
    <BaseEdge id={id} path={path} markerEnd={`url(#csa-edge-${kind})`} style={{
      stroke: kind === 'memory' ? emphasized ? 'var(--memory)' : '#353535' : kind === 'control' ? '#b8b8b8' : '#ccc',
      strokeWidth: kind === 'memory' ? 12 : kind === 'control' ? 2 : emphasized ? 2 : 1.3,
      opacity: kind === 'memory' ? 1 : emphasized ? 1 : kind === 'control' ? .5 : .28,
      strokeLinejoin: 'round',
      strokeDasharray: kind === 'control' ? '10 6' : kind === 'lowering' ? '3 7' : kind === 'lineage' ? '2 7' : undefined,
    }} />
    {showLabel && <EdgeLabelRenderer>
      <span title={data?.attributes?.detail} className={`edge-label edge-label-${kind}`} style={{ transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)` }}>{label}</span>
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

export const nodeTypes = {
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
