'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const pattern = require('./pattern.js');
const previewData = require('./pattern.pangu-preview-data.js');

const moeData = previewData.build(38);
assert.equal(pattern.validate(moeData), moeData);
assert.equal(moeData.schemaVersion, 'model_architecture_front_view.v1');
assert.equal(Object.values(moeData.nodes).some((node) => Number.isFinite(node.x) || Number.isFinite(node.y)), false);

const moeLayout = pattern.createLayout(moeData, new Set());
assert.equal(moeLayout.nodes.filter((node) => node.overlayKind === 'expert-cell').length, 256);
assert.ok(moeLayout.clusters.some((cluster) => cluster.id === 'decoder_layer'));
assert.ok(moeLayout.clusters.some((cluster) => cluster.id === 'sparse_mla_attention'));
assert.ok(moeLayout.clusters.some((cluster) => cluster.id === 'moe_ffn'));
assert.ok(moeLayout.clusters.some((cluster) => cluster.id === 'routed_expert_bank'));
const decoderCluster = moeLayout.clusters.find((cluster) => cluster.id === 'decoder_layer');
const layerInput = moeLayout.nodes.find((node) => node.id === 'layer_input');
assert.ok(layerInput.y - layerInput.height / 2 >= decoderCluster.y + 108);
assert.deepEqual(moeLayout.bundles.map((bundle) => bundle.count), [4, 4]);
assert.deepEqual(moeLayout.bundles.map((bundle) => bundle.containerId), ['mhc_attention', 'mhc_mlp']);
assert.equal(moeLayout.nodes.find((node) => node.id === 'embedding_weight')?.kind, 'tensor');
assert.equal(moeLayout.nodes.find((node) => node.id === 'embedding_weight')?.width, 210);
assert.equal(moeLayout.nodes.find((node) => node.id === 'input_tokens')?.x, moeLayout.width / 2);
assert.equal(
  moeLayout.nodes.find((node) => node.id === 'positions')?.y,
  moeLayout.nodes.find((node) => node.id === 'attention_core')?.y,
);
assert.equal(
  moeLayout.nodes.find((node) => node.id === 'rope_cache')?.y,
  moeLayout.nodes.find((node) => node.id === 'attention_core')?.y,
);
assert.ok(
  moeLayout.nodes.find((node) => node.id === 'positions')?.x
    < moeLayout.nodes.find((node) => node.id === 'attention_core')?.x,
);
assert.ok(
  moeLayout.nodes.find((node) => node.id === 'rope_cache')?.x
    > moeLayout.nodes.find((node) => node.id === 'attention_core')?.x,
);
assert.equal(moeLayout.nodes.find((node) => node.id === 'router_gate')?.width, 300);
assert.equal(moeLayout.nodes.find((node) => node.id === 'route_topk')?.width, 300);
assert.equal(moeLayout.nodes.find((node) => node.id === 'attention_residual_merge')?.colorKey, 'front:residual');
assert.equal(moeLayout.nodes.find((node) => node.id === 'router_gate')?.colorKey, 'front:moe');
assert.equal(moeLayout.nodes.find((node) => node.id === 'expert_0')?.colorKey, 'front:expert');

const collapsedLayout = pattern.createLayout(moeData, new Set(['routed_expert_bank']));
assert.equal(collapsedLayout.nodes.filter((node) => node.overlayKind === 'expert-cell').length, 0);
assert.equal(collapsedLayout.nodes.find((node) => node.id === 'routed_expert_bank')?.collapsed, true);

const denseLayout = pattern.createLayout(previewData.build(0), new Set());
assert.ok(denseLayout.clusters.some((cluster) => cluster.id === 'dense_mlp'));
assert.equal(denseLayout.clusters.some((cluster) => cluster.id === 'moe_ffn'), false);
assert.equal(denseLayout.nodes.filter((node) => node.overlayKind === 'expert-cell').length, 0);

const implementation = fs.readFileSync(path.join(__dirname, 'pattern.js'), 'utf8');
assert.equal(/openPangu|expert_193|\b256\b|pangu_v2_moe/.test(implementation), false);
assert.match(implementation, /onRepeatInstanceChange/);
assert.match(implementation, /options\.onLayerChange/);
const styles = fs.readFileSync(path.join(__dirname, 'pattern.css'), 'utf8');
assert.match(styles, /vector-effect:\s*non-scaling-stroke/);
assert.match(styles, /\.pto-model-architecture-front-view \.pto-model-graphviz-toggle \{[\s\S]*display:\s*inline/);
const graphvizImplementation = fs.readFileSync(path.join(__dirname, '..', 'model-graphviz', 'pattern.js'), 'utf8');
const graphvizStyles = fs.readFileSync(path.join(__dirname, '..', 'model-graphviz', 'pattern.css'), 'utf8');
assert.match(graphvizImplementation, /pto-model-graphviz-toggle-control/);
assert.match(graphvizImplementation, /pto-model-graphviz-toggle-hit/);
assert.match(graphvizStyles, /\.pto-model-graphviz-toggle-hit[\s\S]*pointer-events:\s*all/);

const contract = JSON.parse(fs.readFileSync(path.join(__dirname, 'pattern.json'), 'utf8'));
assert.equal(contract.id, 'model-architecture-front-view');
assert.equal(contract.schemaVersion, 'model_architecture_front_view.v1');

console.log('model-architecture-front-view: 34 assertions passed');
