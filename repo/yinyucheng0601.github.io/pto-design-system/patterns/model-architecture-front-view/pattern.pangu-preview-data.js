(function registerOpenPanguFrontPreviewData(global) {
  'use strict';

  const MODEL = Object.freeze({
    id: 'openpangu-2.0-flash',
    label: 'openPangu 2.0 Flash',
    layerCount: 46,
    denseLayers: [0, 1],
    expertCount: 256,
    residualStreams: 4,
  });

  const sourceRef = 'openPangu-2.0-Infer/components/omni-npu/src/omni_npu/v1/models/pangu/pangu_v2_moe.py';

  function expertId(index) {
    return `expert_${index}`;
  }

  function build(layerIndex = 38) {
    const layer = Math.max(0, Math.min(MODEL.layerCount - 1, Number(layerIndex) || 0));
    const dense = MODEL.denseLayers.includes(layer);
    const experts = Array.from({ length: MODEL.expertCount }, (_, index) => expertId(index));
    const nodes = {
      input_tokens: { label: 'Token IDs', kind: 'tensor', typeLabel: 'Input Tensor', colorKey: 'io:input', sourceRef },
      positions: { label: 'Position IDs', kind: 'tensor', typeLabel: 'Input Tensor', colorKey: 'io:input', sourceRef },
      embedding_weight: { label: 'Embedding Weight', kind: 'tensor', typeLabel: 'Parameter', colorKey: 'io:parameter', sourceRef },
      token_embedding: { label: 'Vocab Parallel Embedding', kind: 'op', colorKey: 'front:embedding', sourceRef },
      previous_layers: { label: layer ? `Previous Decoder Layers · L0–L${layer - 1}` : 'Model Input', kind: 'module', colorKey: 'front:embedding', sourceRef },
      decoder_layer: { label: 'Decoder Layer', kind: 'module', colorKey: 'module:decoder', sourceRef },
      layer_input: { label: `H${layer} · 4 × residual streams`, kind: 'state', colorKey: 'front:residual', hideTypeLabel: true, overlayKind: 'residual-state', sourceRef },
      mhc_attention: { label: 'mHC Attention State Flow', kind: 'module', colorKey: 'front:attention', sourceRef },
      mhc_attention_pre: { label: 'mHC Attention Pre', kind: 'op', colorKey: 'front:attention', sourceRef },
      mhc_attention_residual_state: { label: 'Attention Residual', kind: 'state', colorKey: 'front:residual', hideTypeLabel: true, overlayKind: 'residual-state', sourceRef },
      input_layernorm: { label: 'Input RMSNorm', kind: 'op', colorKey: 'front:norm', sourceRef },
      sparse_mla_attention: { label: 'Sparse MLA Attention', kind: 'module', colorKey: 'front:attention', sourceRef },
      attention_weights: { label: 'Q / KV / O Weights', kind: 'tensor', typeLabel: 'Parameter', colorKey: 'io:parameter', sourceRef },
      q_latent: { label: 'Q Projection', kind: 'op', colorKey: 'front:attention', sourceRef },
      kv_latent: { label: 'KV Projection', kind: 'op', colorKey: 'front:attention', sourceRef },
      rope_cache: { label: 'RoPE / KV Cache', kind: 'tensor', typeLabel: 'State Tensor', colorKey: 'io:state', sourceRef },
      attention_core: { label: 'Sparse FlashAttention', kind: 'op', colorKey: 'front:attention', sourceRef },
      output_projection: { label: 'Output Projection', kind: 'op', colorKey: 'front:attention', sourceRef },
      post_attention_norm: { label: 'Post Attention RMSNorm', kind: 'op', colorKey: 'front:norm', sourceRef },
      attention_residual_merge: { label: 'Attention Residual Merge', kind: 'op', colorKey: 'front:residual', overlayKind: 'residual-op', sourceRef },
      middle_state: { label: `H${layer}′ · 4 × residual streams`, kind: 'state', colorKey: 'front:residual', hideTypeLabel: true, overlayKind: 'residual-state', sourceRef },
      mhc_mlp: { label: 'mHC FFN State Flow', kind: 'module', colorKey: 'front:ffn', sourceRef },
      mhc_mlp_pre: { label: 'mHC FFN Pre', kind: 'op', colorKey: 'front:ffn', sourceRef },
      mhc_mlp_residual_state: { label: 'FFN Residual', kind: 'state', colorKey: 'front:residual', hideTypeLabel: true, overlayKind: 'residual-state', sourceRef },
      pre_mlp_norm: { label: 'Pre MLP RMSNorm', kind: 'op', colorKey: 'front:norm', sourceRef },
      dense_mlp: { label: 'Dense MLP', kind: 'module', colorKey: 'front:ffn', sourceRef },
      dense_weights: { label: 'Dense MLP Weights', kind: 'tensor', typeLabel: 'Parameter', colorKey: 'io:parameter', sourceRef },
      dense_gate_up: { label: 'Gate + Up Projection', kind: 'op', colorKey: 'front:ffn', sourceRef },
      dense_silu: { label: 'SiLU Multiply', kind: 'op', colorKey: 'front:ffn', sourceRef },
      dense_down: { label: 'Down Projection', kind: 'op', colorKey: 'front:ffn', sourceRef },
      moe_ffn: { label: 'MoE FFN', kind: 'module', colorKey: 'front:moe', sourceRef },
      router_gate: { label: 'Router Gate', kind: 'op', colorKey: 'front:moe', sourceRef },
      route_topk: { label: 'TopK Router', kind: 'op', colorKey: 'front:moe', sourceRef },
      routed_expert_bank: { label: 'Routed Experts', kind: 'module', colorKey: 'front:moe', sourceRef },
      shared_expert: { label: 'Shared Expert', kind: 'op', colorKey: 'front:expert', sourceRef },
      expert_combine: { label: 'Expert Combine', kind: 'op', colorKey: 'front:moe', sourceRef },
      post_mlp_norm: { label: 'Post MLP RMSNorm', kind: 'op', colorKey: 'front:norm', sourceRef },
      ffn_residual_merge: { label: 'FFN Residual Merge', kind: 'op', colorKey: 'front:residual', overlayKind: 'residual-op', sourceRef },
      layer_output: { label: `H${layer + 1} · 4 × residual streams`, kind: 'state', colorKey: 'front:residual', hideTypeLabel: true, overlayKind: 'residual-state', sourceRef },
      remaining_layers: { label: layer < MODEL.layerCount - 1 ? `Remaining Decoder Layers · L${layer + 1}–L45` : 'Decoder Output', kind: 'module', colorKey: 'front:embedding', sourceRef },
      final_norm: { label: 'Final RMSNorm', kind: 'op', colorKey: 'front:norm', sourceRef },
      lm_head_weight: { label: 'LM Head Weight', kind: 'tensor', typeLabel: 'Parameter', colorKey: 'io:parameter', sourceRef },
      lm_head: { label: 'LM Head', kind: 'op', colorKey: 'front:head', sourceRef },
      logits: { label: 'Logits', kind: 'tensor', typeLabel: 'Output Tensor', colorKey: 'io:output', sourceRef },
    };
    experts.forEach((id, index) => {
      nodes[id] = { label: `E${String(index).padStart(3, '0')}`, kind: 'module', colorKey: 'front:expert', sourceRef };
    });

    const attention = {
      type: 'cluster', id: 'sparse_mla_attention', label: 'Sparse MLA Attention', width: 980, children: [
        { type: 'row', items: [{ id: 'attention_weights', lane: 'left' }] },
        { type: 'row', items: ['q_latent', 'kv_latent'] },
        { type: 'row', items: [{ id: 'positions', lane: 'left' }, 'attention_core', { id: 'rope_cache', lane: 'right' }] },
        { type: 'row', items: ['output_projection'] },
      ],
    };

    const feedForward = dense
      ? {
          type: 'cluster', id: 'dense_mlp', label: 'Dense MLP', width: 980, children: [
            { type: 'row', items: [{ id: 'dense_weights', lane: 'left' }, 'dense_gate_up'] },
            { type: 'row', items: ['dense_silu'] },
            { type: 'row', items: ['dense_down'] },
          ],
        }
      : {
          type: 'cluster', id: 'moe_ffn', label: 'MoE FFN', width: 1080, children: [
            { type: 'row', items: ['router_gate'] },
            { type: 'row', items: ['route_topk'] },
            { type: 'expert-bank', id: 'routed_expert_bank', label: 'Routed Experts', expertIds: experts, columns: 16, width: 760, aside: ['shared_expert'] },
            { type: 'row', items: ['expert_combine'] },
          ],
        };

    const structure = [
      { type: 'row', items: ['input_tokens'] },
      { type: 'row', items: [{ id: 'embedding_weight', lane: 'left' }, 'token_embedding'] },
    ];
    if (layer > 0) structure.push({ type: 'row', items: ['previous_layers'] });
    structure.push({
      type: 'cluster', id: 'decoder_layer', label: 'Decoder Layer', width: 1360,
      repeatCount: MODEL.layerCount, selectedInstanceIndex: layer, collapsible: false,
      children: [
        { type: 'row', items: ['layer_input'] },
        {
          type: 'cluster', id: 'mhc_attention', label: 'mHC Attention State Flow', width: 1240, children: [
            { type: 'row', items: ['mhc_attention_pre', { id: 'mhc_attention_residual_state', lane: 'right', width: 210 }] },
            { type: 'row', items: ['input_layernorm'] },
            attention,
            { type: 'row', items: ['post_attention_norm'] },
            { type: 'row', items: ['attention_residual_merge'] },
          ],
        },
        { type: 'row', items: ['middle_state'] },
        {
          type: 'cluster', id: 'mhc_mlp', label: 'mHC FFN State Flow', width: 1240, children: [
            { type: 'row', items: ['mhc_mlp_pre', { id: 'mhc_mlp_residual_state', lane: 'right', width: 210 }] },
            { type: 'row', items: ['pre_mlp_norm'] },
            feedForward,
            { type: 'row', items: ['post_mlp_norm'] },
            { type: 'row', items: ['ffn_residual_merge'] },
          ],
        },
        { type: 'row', items: ['layer_output'] },
      ],
    });
    if (layer < MODEL.layerCount - 1) structure.push({ type: 'row', items: ['remaining_layers'] });
    structure.push(
      { type: 'row', items: ['final_norm'] },
      { type: 'row', items: [{ id: 'lm_head_weight', lane: 'left' }, 'lm_head'] },
      { type: 'row', items: ['logits'] },
    );

    const edges = [
      { id: 'input-embedding', source: 'input_tokens', target: 'token_embedding', semanticEdgeType: 'activation' },
      { id: 'embedding-weight', source: 'embedding_weight', target: 'token_embedding', semanticEdgeType: 'parameter', sourceAnchor: 'right', targetAnchor: 'left', curve: 'horizontal' },
      { id: 'position-attention', source: 'positions', target: 'attention_core', semanticEdgeType: 'state', sourceAnchor: 'right', targetAnchor: 'left', curve: 'horizontal' },
      { id: 'embedding-context', source: 'token_embedding', target: layer ? 'previous_layers' : 'layer_input', semanticEdgeType: 'activation' },
      ...(layer ? [{ id: 'context-layer', source: 'previous_layers', target: 'layer_input', semanticEdgeType: 'activation' }] : []),
      { id: 'layer-attention-pre', source: 'layer_input', target: 'mhc_attention_pre', semanticEdgeType: 'activation' },
      { id: 'attention-state-pre', source: 'mhc_attention_residual_state', target: 'mhc_attention_pre', semanticEdgeType: 'state', sourceAnchor: 'left', targetAnchor: 'right', curve: 'horizontal' },
      { id: 'pre-input-norm', source: 'mhc_attention_pre', target: 'input_layernorm', semanticEdgeType: 'activation' },
      { id: 'norm-q', source: 'input_layernorm', target: 'q_latent', semanticEdgeType: 'activation' },
      { id: 'norm-kv', source: 'input_layernorm', target: 'kv_latent', semanticEdgeType: 'activation' },
      { id: 'attention-param', source: 'attention_weights', target: 'q_latent', semanticEdgeType: 'parameter', sourceAnchor: 'right', targetAnchor: 'left', curve: 'horizontal' },
      { id: 'q-core', source: 'q_latent', target: 'attention_core', semanticEdgeType: 'activation' },
      { id: 'kv-core', source: 'kv_latent', target: 'attention_core', semanticEdgeType: 'activation' },
      { id: 'cache-core', source: 'rope_cache', target: 'attention_core', semanticEdgeType: 'state', sourceAnchor: 'left', targetAnchor: 'right', curve: 'horizontal' },
      { id: 'core-output', source: 'attention_core', target: 'output_projection', semanticEdgeType: 'activation' },
      { id: 'output-post-norm', source: 'output_projection', target: 'post_attention_norm', semanticEdgeType: 'activation' },
      { id: 'post-norm-attn-merge', source: 'post_attention_norm', target: 'attention_residual_merge', semanticEdgeType: 'activation' },
      { id: 'attn-merge-middle', source: 'attention_residual_merge', target: 'middle_state', semanticEdgeType: 'activation' },
      { id: 'middle-ffn-pre', source: 'middle_state', target: 'mhc_mlp_pre', semanticEdgeType: 'activation' },
      { id: 'ffn-state-pre', source: 'mhc_mlp_residual_state', target: 'mhc_mlp_pre', semanticEdgeType: 'state', sourceAnchor: 'left', targetAnchor: 'right', curve: 'horizontal' },
      { id: 'ffn-pre-norm', source: 'mhc_mlp_pre', target: 'pre_mlp_norm', semanticEdgeType: 'activation' },
      ...(dense ? [
        { id: 'norm-dense', source: 'pre_mlp_norm', target: 'dense_gate_up', semanticEdgeType: 'activation' },
        { id: 'dense-param', source: 'dense_weights', target: 'dense_gate_up', semanticEdgeType: 'parameter', sourceAnchor: 'right', targetAnchor: 'left', curve: 'horizontal' },
        { id: 'gate-silu', source: 'dense_gate_up', target: 'dense_silu', semanticEdgeType: 'activation' },
        { id: 'silu-down', source: 'dense_silu', target: 'dense_down', semanticEdgeType: 'activation' },
        { id: 'dense-post', source: 'dense_down', target: 'post_mlp_norm', semanticEdgeType: 'activation' },
      ] : [
        { id: 'norm-router', source: 'pre_mlp_norm', target: 'router_gate', semanticEdgeType: 'activation' },
        { id: 'router-topk', source: 'router_gate', target: 'route_topk', semanticEdgeType: 'activation' },
        { id: 'topk-experts', source: 'route_topk', target: 'routed_expert_bank', semanticEdgeType: 'communication' },
        { id: 'experts-combine', source: 'routed_expert_bank', target: 'expert_combine', semanticEdgeType: 'communication' },
        { id: 'shared-combine', source: 'shared_expert', target: 'expert_combine', semanticEdgeType: 'activation', sourceAnchor: 'bottom', targetAnchor: 'right' },
        { id: 'combine-post', source: 'expert_combine', target: 'post_mlp_norm', semanticEdgeType: 'activation' },
      ]),
      { id: 'post-ffn-merge', source: 'post_mlp_norm', target: 'ffn_residual_merge', semanticEdgeType: 'activation' },
      { id: 'ffn-merge-output', source: 'ffn_residual_merge', target: 'layer_output', semanticEdgeType: 'activation' },
      ...(layer < MODEL.layerCount - 1 ? [
        { id: 'layer-remaining', source: 'layer_output', target: 'remaining_layers', semanticEdgeType: 'activation' },
        { id: 'remaining-final', source: 'remaining_layers', target: 'final_norm', semanticEdgeType: 'activation' },
      ] : [{ id: 'layer-final', source: 'layer_output', target: 'final_norm', semanticEdgeType: 'activation' }]),
      { id: 'final-head', source: 'final_norm', target: 'lm_head', semanticEdgeType: 'activation' },
      { id: 'head-weight', source: 'lm_head_weight', target: 'lm_head', semanticEdgeType: 'parameter', sourceAnchor: 'right', targetAnchor: 'left', curve: 'horizontal' },
      { id: 'head-logits', source: 'lm_head', target: 'logits', semanticEdgeType: 'activation' },
    ];

    return {
      schemaVersion: 'model_architecture_front_view.v1',
      model: MODEL,
      metadata: { selectedLayer: layer, branch: dense ? 'dense' : 'moe', sourceChecked: true },
      provenance: sourceRef,
      nodes,
      structure,
      edges,
      residualBundles: [
        { id: 'attention-residual-x4', source: 'layer_input', target: 'attention_residual_merge', count: MODEL.residualStreams, side: 'left', containerId: 'mhc_attention', containerInset: 82, label: 'mHC residual ×4' },
        { id: 'ffn-residual-x4', source: 'middle_state', target: 'ffn_residual_merge', count: MODEL.residualStreams, side: 'left', containerId: 'mhc_mlp', containerInset: 82, label: 'mHC residual ×4' },
      ],
      selectedNodeId: dense ? 'dense_mlp' : 'expert_193',
      initialCollapsedIds: [],
    };
  }

  const api = Object.freeze({ model: MODEL, build });
  global.PtoOpenPanguFrontViewPreviewData = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
