/* Read-only UI projection of the semantic graph. No geometry or semantic defaults. */
(function (root) {
  'use strict';
  const own = (value, key) => Object.prototype.hasOwnProperty.call(value, key);
  const human = (value) => {
    if (Array.isArray(value)) return value.map(human).join(' · ');
    if (value && typeof value === 'object') return Object.entries(value).map(([key, item]) => `${key}: ${human(item)}`).join(' · ');
    return String(value);
  };
  function sourceRanges(node) {
    if (node.source?.ranges) return node.source.ranges.map(({ start, end }) => [start, end]);
    if (Array.isArray(node.sourceRanges)) return node.sourceRanges.map((range) => Array.isArray(range) ? range : [range.start, range.end]);
    const lines = String(node.lines || '').match(/\d+(?:\s*[-–]\s*\d+)?/g) || [];
    return lines.map((range) => range.split(/\s*[-–]\s*/).map(Number)).map((pair) => pair.length === 1 ? [pair[0], pair[0]] : pair);
  }
  const sourceLine = (node) => sourceRanges(node)[node.source?.primaryRange || 0]?.[0];
  function formatSource(source) {
    if (!source) return '';
    const ranges = source.ranges || [];
    return `${source.file}${ranges.length ? ':' + ranges.map(({ start, end }) => start === end ? start : `${start}–${end}`).join(', ') : ''}`;
  }
  function node(semantic, context = {}) {
    const p = semantic[semantic.type] || {}, type = semantic.type;
    const sub = type === 'data' ? [p.dtype, p.granularity].filter(Boolean).slice(0, 2).join(' · ') :
      type === 'state' ? [p.dtype, p.granularity && `跨 ${p.granularity}`].filter(Boolean).slice(0, 2).join(' · ') :
      type === 'composite' ? [p.category, p.subgraph && `${p.subgraph.nodes.length} 个节点`].filter(Boolean).join(' · ') :
      [p.category, p.api?.[0]].filter(Boolean).slice(0, 2).join(' · ');
    const id = context.id || semantic.id;
    return { ...semantic, semantic, id, parentId: context.parentId,
      semanticType: type, type, kind: type === 'data' || type === 'state' ? 'data' : 'operation',
      badge: type.toUpperCase(), title: semantic.name, sub, detail: semantic.description,
      api: Array.isArray(p.api) ? p.api.join(' · ') : '', lines: (semantic.source?.ranges || []).map(({ start, end }) => start === end ? String(start) : `${start}–${end}`).join('、'),
      sourceRanges: sourceRanges(semantic), sourceLine: sourceLine(semantic) };
  }
  function edge(semantic, context = {}) {
    return { ...semantic, semantic, id: context.id || semantic.id,
      s: context.sourceId || semantic.s, t: context.targetId || semantic.t,
      internal: !!context.internal,
      route: semantic.route || (semantic.type === 'feedback' ? 'feedback' : semantic.type) };
  }
  function locate(model, qualifiedId) {
    if (!model || !qualifiedId) return null;
    const parts = qualifiedId.split('/'); let graph = model, found;
    for (let i = 0; i < parts.length; i++) {
      found = graph.nodes?.find((item) => item.id === parts[i]);
      if (!found) return null;
      if (i < parts.length - 1) graph = found.composite?.subgraph;
    }
    return { node: found, graph };
  }
  function inspector(item, { model, qualifiedId } = {}) {
    const semantic = item.semantic || item;
    const p = semantic[semantic.type] || {};
    const sections = [];
    const add = (title, entries) => {
      const rows = entries.filter(([, value]) => value !== undefined && value !== null && value !== '' &&
        !(Array.isArray(value) && !value.length) && !(typeof value === 'object' && !Object.keys(value).length)).map(([label, value]) => ({ label, value: human(value) }));
      if (rows.length) sections.push({ title, rows });
    };
    add('语义', [['类型', semantic.type], ['名称', semantic.name], ['描述', semantic.description]]);
    // Render every populated payload key; new engineering attributes remain visible.
    add('属性', Object.entries(p).filter(([key]) => !['subgraph', 'portMappings', 'producer', 'consumers', 'inputs', 'outputs'].includes(key)));
    if (semantic.ports) add('端口', ['inputs', 'outputs'].flatMap((side) => Object.entries(semantic.ports[side] || {}).map(([name, value]) => [`${side}.${name}`, Object.keys(value).length ? value : name])));
    if (p.portMappings) add('端口映射', ['inputs', 'outputs'].flatMap((side) => Object.entries(p.portMappings[side] || {}).map(([name, value]) => [`${side}.${name}`, value])));
    if (semantic.source) add('源码', [['位置', formatSource(semantic.source)], ['符号', semantic.source.symbols?.length ? semantic.source.symbols : undefined]]);
    if (semantic.provenance?.length) add('来源依据', semantic.provenance.map((entry, i) => [`${i + 1}. ${entry.kind}`, { origin: entry.origin, fields: entry.fields, ...(own(entry, 'note') ? { note: entry.note } : {}) }]));
    const found = locate(model, qualifiedId || item.id);
    if (found) {
      const id = found.node.id, edges = found.graph.edges || [];
      const incoming = edges.filter((edge) => edge.t === id);
      const outgoing = edges.filter((edge) => edge.s === id);
      add('依赖', [
        ['生产者', incoming.filter((edge) => edge.type === 'data').map((edge) => edge.s)],
        ['消费者', outgoing.filter((edge) => edge.type === 'data').map((edge) => edge.t)],
        ['输入', incoming.map((edge) => `${edge.s}.${edge.sourcePort} → ${edge.targetPort} (${edge.type})`)],
        ['输出', outgoing.map((edge) => `${edge.sourcePort} → ${edge.t}.${edge.targetPort} (${edge.type})`)]
      ]);
    }
    return sections;
  }
  root.PTO_COMPUTATION_PRESENTATION = { node, edge, sourceRanges, sourceLine, formatSource, inspector };
})(typeof window === 'undefined' ? globalThis : window);
