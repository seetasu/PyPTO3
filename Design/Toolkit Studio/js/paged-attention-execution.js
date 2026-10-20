/* Logical execution counts for one Task. No device instruction or timing estimates. */
(function (root) {
  'use strict';
  // Fixed linear scale: 1 → 1.5px, 6 → 5px, 24 → 17.6px, 28 → 20.4px.
  const width = (count) => 1.5 + 0.7 * (count - 1);
  function derive(model, taskModel, scenario) {
    const { task, seqLen } = scenario || {};
    const value = (key) => model.constants[key].value;
    if (!Number.isInteger(task) || task < 0 || task >= value('NUM_TASKS')
      || !Number.isInteger(seqLen) || seqLen < 1 || seqLen > value('SEQ_LEN')) return null;
    const data = taskModel.pipelineFor(seqLen);
    const pages = data.pageCount, stacks = data.stackCount;
    const count = (n, unit, ranges) => ({ count: n, unit, label: `×${n} ${unit}`, source: { file: 'paged_attention_pypto.py', ranges } });
    const page = (...ranges) => count(pages, 'Pages', ranges);
    const stack = (...ranges) => count(stacks, 'Stacks', ranges);
    const nodes = {
      qk: page([279, 356]), pv: page([364, 718]),
      'scale-mask': stack([762, 788]), softmax: stack([762, 825]),
      'context-acc': stack([827, 861]), 'output-norm': count(1, '/ Task', [[864, 873]]),
      'qk/matmul': page([284, 356]), 'pv/matmul': page([369, 718]),
    };
    const edges = {};
    // Match declared semantic IDs, including internal edges, rather than routed endpoints.
    const pairs = {
      'k-cache>qk': page([284, 356]), 'qk>scores': page([284, 356]),
      'scores>scale-mask': stack([762, 776]), 'scale-mask>softmax': stack([777, 788]),
      'softmax>probability': stack([802, 816]), 'probability>pv': stack([364, 718]),
      'v-cache>pv': page([369, 718]), 'pv>context-acc': stack([719, 724], [827, 840]),
      'output-norm>output': count(1, '/ Task', [[864, 873]]),
      'qk/k-page>matmul': page([284, 356]), 'qk/matmul>score': page([284, 356]),
      'pv/v-page>matmul': page([369, 718]), 'pv/probability>matmul': page([369, 718]),
      'pv/matmul>partial': stack([503, 718]),
    };
    function visit(graph, prefix = '') {
      for (const edge of graph.edges) {
        if (edge.type !== 'data') continue;
        const entry = pairs[`${prefix}${edge.s}>${edge.t}`];
        if (entry) edges[prefix + edge.id] = { ...entry, width: width(entry.count) };
      }
      for (const node of graph.nodes) if (node.composite) visit(node.composite.subgraph, `${prefix}${node.id}/`);
    }
    visit(model);
    return { task, seqLen, pages, stacks, ticks: data.stepCount, stacksDetail: data.stacks.map((stack) => ({ ...stack,
        branch: stack.pages.length === value('STACK_PAGES') ? '完整 Stack 分支' : `tail_pages == ${stack.pages.length}` })),
      pageTokens: value('BLOCK_SIZE'), pagesPerStack: value('STACK_PAGES'), preLaunch: value('PRE_LAUNCH'),
      pvMatmul: stacks, pvMatmulAcc: pages - stacks, nodes, edges };
  }
  root.PtoPagedAttentionExecution = { derive, width };
})(typeof window === 'undefined' ? globalThis : window);
