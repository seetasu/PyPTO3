import { useMemo, useState } from 'react';

const evidence = [
  { id: 'rope', kind: 'fact', title: '两套 RoPE profile 已从 43 层构建产物中提取', meta: 'route-map.json · 14:28' },
  { id: 'repeat', kind: 'fact', title: '固定序 reduction 后，9/9 请求逐字节一致', meta: 'repeatability.csv · 14:36' },
  { id: 'tail', kind: 'fact', title: '48 / 96 token 正常；128 token 尾部劣化', meta: 'quality-boundary.md · 14:21' },
  { id: 'route', kind: 'hypothesis', title: 'CSA / HCA 层被错误路由到基础 RoPE', meta: '等待 profile A/B 验证' },
  { id: 'atomic', kind: 'hypothesis', title: '并发 AtomicAdd 放大了请求间数值差异', meta: '等待 fixed-order 对照' },
  { id: 'scope', kind: 'unknown', title: '是否存在特定 head 或 layer 的额外放大效应', meta: '尚未采集 head-level 证据' },
];

const branches = {
  rope: {
    label: 'RoPE 路由',
    title: '逐层 profile 路由可能解释长尾质量劣化',
    summary: 'SWA 使用基础 profile；CSA / HCA 应使用压缩 profile。将一张基础表传给所有主层，会让位置相位误差随 decode 位置累积。',
    controls: ['分别打包基础 / 压缩两套 profile', '按层组路由：SWA、CSA/HCA、MTP', '在 48 / 96 / 128 token 阶梯回归中验证'],
    evidence: '43 层 profile route map',
  },
  deterministic: {
    label: '确定性归约',
    title: 'split-K 的完成顺序可能解释请求 / rank 间波动',
    summary: '并发 FP32 AtomicAdd 的完成顺序不保证一致。很小的舍入差异可跨越后续 greedy argmax 边界，并在自回归 decode 中逐步放大。',
    controls: ['输出 disjoint partial buffer', '按 split 索引升序完成归约', '以 9 次重复 × 3 个长度检查字节一致性'],
    evidence: '9 次 repeated-run hash 对照',
  },
};

const routeRows = [
  ['0–1', 'SWA', '基础', '基础', '匹配'],
  ['2–12', 'CSA', '基础', '压缩', '需要修复'],
  ['13–24', 'HCA', '基础', '压缩', '需要修复'],
  ['25–42', 'CSA / HCA', '基础', '压缩', '需要修复'],
  ['43', 'MTP draft', '基础', '基础', '匹配'],
];

const navItems = ['因果地图', '证据', '实验', '运行'];

function Dot({ kind }) {
  return <span className={`dot dot-${kind}`} aria-hidden="true" />;
}

function BranchCard({ id, active, onSelect }) {
  const branch = branches[id];
  return (
    <button type="button" className={`branch ${active ? 'is-active' : ''}`} onClick={() => onSelect(id)} aria-pressed={active}>
      <span className="branch-kicker"><Dot kind="cyan" /> 诊断路径</span>
      <strong>{branch.label}</strong>
      <span>{id === 'rope' ? '逐层 profile 与模型语义对齐' : '跨重复运行的归约一致性'}</span>
      <span className="branch-tags"><b>{id === 'rope' ? 'H1' : 'H2'}</b><i>{id === 'rope' ? 'E1' : 'E2'}</i></span>
    </button>
  );
}

export function App() {
  const [activeNav, setActiveNav] = useState('因果地图');
  const [activeBranch, setActiveBranch] = useState('rope');
  const [evidenceFilter, setEvidenceFilter] = useState('all');
  const [selectedLength, setSelectedLength] = useState('128');
  const [experimentOpen, setExperimentOpen] = useState(false);
  const [planCreated, setPlanCreated] = useState(false);
  const branch = branches[activeBranch];
  const visibleEvidence = useMemo(() => evidence.filter((item) => evidenceFilter === 'all' || item.kind === evidenceFilter), [evidenceFilter]);

  const createPlan = (event) => { event.preventDefault(); setPlanCreated(true); setExperimentOpen(false); };

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="product-mark">PTO3 <span>· Diagnose</span></div>
        <div className="case-crumb">案例库 <b>/</b> 长 decode 质量回溯 <b>/</b> 调查 #951</div>
        <div className="run-meta"><span>DeepSeek-V4-Flash</span><span>8 ranks · a2a3</span><span className="status-chip">调查中</span></div>
      </header>
      <div className="workspace">
        <aside className="sidebar" aria-label="工作台导航">
          <div className="case-label">INVESTIGATION</div><h1>长 decode<br />质量劣化</h1><p>48 / 96 token 正常；128 token 后段开始偏离参考输出。</p>
          <nav>{navItems.map((item) => <button key={item} type="button" className={activeNav === item ? 'nav-item is-active' : 'nav-item'} onClick={() => setActiveNav(item)}>{item}</button>)}</nav>
          <div className="sidebar-foot"><span>当前比较</span><b>main ↔ regression</b><small>commit 384553a · 9 个重复运行</small></div>
        </aside>
        <section className="main-pane">
          <div className="page-heading"><div><p className="eyebrow">因果地图</p><h2>同一症状，分开验证两个根因</h2><p>先确定 profile 路由是否正确，再检查并发归约是否让结果随运行漂移。</p></div><div className="legend"><span><Dot kind="magenta" />观察到的症状</span><span><Dot kind="cyan" />已验证证据</span><span><Dot kind="amber" />待验证假设</span></div></div>
          <section className="causal-map" aria-label="长 decode 质量劣化因果地图">
            <div className="symptom-node"><span>OBSERVED</span><strong>长 decode 尾部质量劣化</strong><small>48 / 96 token 稳定，128 token 后段开始重复或失真</small></div>
            <div className="branch-row"><BranchCard id="rope" active={activeBranch === 'rope'} onSelect={setActiveBranch} /><BranchCard id="deterministic" active={activeBranch === 'deterministic'} onSelect={setActiveBranch} /></div>
            <div className="consequence-row"><div><span className="node-code">M1</span><strong>位置相位偏差逐层累积</strong><small>影响远距离 token 对齐</small></div><div><span className="node-code">M2</span><strong>微小数值差异跨过 argmax 边界</strong><small>放大为请求 / rank 间差异</small></div></div>
            <div className="map-conclusion"><span>✓</span><div><strong>两个根因并不互斥</strong><small>修复确定性可消除“每次不同”；修复 profile 才能恢复长序列语义。</small></div></div>
          </section>
          <section className="detail-panel" aria-live="polite"><div className="detail-copy"><p className="eyebrow">选中路径 · {branch.label}</p><h3>{branch.title}</h3><p>{branch.summary}</p><div className="evidence-link"><Dot kind="cyan" /> {branch.evidence}</div></div><div className="control-list"><span>建议的控制变量</span>{branch.controls.map((control, index) => <div key={control}><i>0{index + 1}</i>{control}</div>)}</div></section>
          <section className="lower-grid">
            <article className="matrix-panel"><header><div><p className="eyebrow">验证矩阵</p><h3>长度 × 重复运行</h3></div><span>点击一行查看对应证据</span></header><div className="matrix-head"><span>生成长度</span>{[1,2,3,4,5,6,7,8,9].map((run) => <span key={run}>R{run}</span>)}</div>{['48','96','128'].map((length) => <button type="button" className={`matrix-row ${selectedLength === length ? 'is-selected' : ''}`} key={length} onClick={() => setSelectedLength(length)}><strong>{length} token</strong>{Array.from({ length: 9 }, (_, index) => <span className={length === '128' ? 'matrix-dot is-failing' : 'matrix-dot'} key={index} />)}</button>)}<div className="matrix-note"><Dot kind={selectedLength === '128' ? 'magenta' : 'cyan'} /> {selectedLength === '128' ? '128 token：质量边界已复现；用两条单变量实验拆分根因。' : `${selectedLength} token：作为稳定对照组，保留相同输入与运行契约。`}</div></article>
            <article className="route-panel"><header><div><p className="eyebrow">层级路由</p><h3>基础 / 压缩 profile 对照</h3></div><button type="button" className="text-button" onClick={() => setActiveBranch('rope')}>聚焦路径</button></header><div className="route-table"><div className="route-head"><span>层</span><span>模块</span><span>当前</span><span>应为</span></div>{routeRows.map(([layer,module,current,expected,state]) => <div className={`route-row ${state === '需要修复' ? 'needs-fix' : ''}`} key={layer}><span>{layer}</span><span>{module}</span><span>{current}</span><span>{expected}</span></div>)}</div></article>
          </section>
        </section>
        <aside className="evidence-rail"><header><div><p className="eyebrow">证据</p><h2>调查笔记</h2></div><span className="evidence-count">{visibleEvidence.length}</span></header><div className="filter-row">{[['all','全部'],['fact','事实'],['hypothesis','假设']].map(([id,label]) => <button type="button" key={id} className={evidenceFilter === id ? 'is-active' : ''} onClick={() => setEvidenceFilter(id)}>{label}</button>)}</div><div className="evidence-list">{visibleEvidence.map((item) => <button type="button" className="evidence-item" key={item.id} onClick={() => item.id === 'rope' && setActiveBranch('rope')}><Dot kind={item.kind === 'fact' ? 'cyan' : item.kind === 'hypothesis' ? 'amber' : 'muted'} /><span><b>{item.kind === 'fact' ? '事实' : item.kind === 'hypothesis' ? '假设' : '未知'}</b><strong>{item.title}</strong><small>{item.meta}</small></span></button>)}</div><div className={`next-step ${planCreated ? 'is-complete' : ''}`}><p className="eyebrow">下一步</p><strong>{planCreated ? '验证计划已创建' : '建立可控实验'}</strong><p>{planCreated ? '两条单变量分支已排入队列：先 profile 路由，后固定序归约。' : '固定输入、长度与 profile；每次只切换一个根因。'}</p><button type="button" className="primary-button" onClick={() => setExperimentOpen(true)}>{planCreated ? '查看计划' : '创建可控实验'}</button></div></aside>
      </div>
      {experimentOpen && <div className="modal-backdrop" role="presentation"><form className="experiment-modal" onSubmit={createPlan}><div className="modal-top"><div><p className="eyebrow">新建实验</p><h2>拆分长 decode 的两个根因</h2></div><button type="button" className="close-button" onClick={() => setExperimentOpen(false)}>关闭</button></div><p>基线使用同一 checkpoint、输入和 48 / 96 / 128 token 阶梯。每项实验只改变一个控制变量。</p><label><input type="checkbox" defaultChecked /> 仅修复 RoPE profile 路由；保留现有 reduction</label><label><input type="checkbox" defaultChecked /> 仅使用固定序 split-K reduction；保留现有 profile</label><label><input type="checkbox" defaultChecked /> 每项运行 9 次，并记录逐字节 hash 与长尾质量</label><div className="modal-actions"><button type="button" className="secondary-button" onClick={() => setExperimentOpen(false)}>取消</button><button type="submit" className="primary-button">创建验证计划</button></div></form></div>}
    </main>
  );
}
