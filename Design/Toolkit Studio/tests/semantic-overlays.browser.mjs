/* Browser acceptance runner for the documented Codex tab.playwright interface.
 * Open index_v2-slz.html, choose paged_attention_pypto.py → 计算流程, expand QK,
 * select its Matmul, then call run(tab). No browser runtime or dependency is installed.
 * Evaluations only read DOM state; all actions use locator.click/press. */
import assert from 'node:assert/strict';
async function capture(tab) {
  return tab.playwright.locator('#pypaTaskScene').evaluate((scene) => ({
    nodes: [...scene.querySelectorAll('[data-pypa-node]')].map((n) => [n.dataset.pypaNode, n.style.left, n.style.top, n.style.width, n.style.minHeight, n.getAttribute('aria-expanded')]),
    edges: [...scene.querySelectorAll('[data-pypa-edge]')].map((e) => [e.dataset.pypaEdge, e.getAttribute('d')]),
    selection: [...scene.querySelectorAll('.is-selected[data-pypa-node]')].map((n) => n.dataset.pypaNode),
    transform: scene.style.transform,
    inspector: scene.closest('.kf-pypa-graph').querySelector('#pypaNodeInspector').innerHTML,
    inspectorHidden: scene.closest('.kf-pypa-graph').querySelector('#pypaNodeInspector').hidden,
    overflow: [...scene.querySelectorAll('[data-pypa-node]:not(.is-expanded-head)')].filter((n) => n.scrollHeight > n.clientHeight + 1).map((n) => [n.dataset.pypaNode,n.scrollHeight,n.clientHeight]),
  }));
}
async function run(tab) {
  const keys = ['shape','precision','memory','control'];
  const initial = await capture(tab);
  assert.ok(initial.nodes.some((n) => n[0] === 'qk/q-tile'), 'QK is expanded');
  assert.ok(initial.selection.length && !initial.inspectorHidden, 'selected Inspector is open');
  const states = [];
  for (let mask = 0; mask < 16; mask++) {
    for (let i = 0; i < keys.length; i++) {
      const button = tab.playwright.locator(`[data-pypa-overlay="${keys[i]}"]`);
      if ((await button.getAttribute('aria-pressed') === 'true') !== !!(mask & (1 << i))) await button.click();
    }
    const after = await capture(tab);
    assert.deepEqual(after.nodes, initial.nodes, `nodes/coordinates/expansion: mask ${mask}`);
    assert.deepEqual(after.edges, initial.edges, `edge identity/routes: mask ${mask}`);
    assert.deepEqual(after.selection, initial.selection, `selection: mask ${mask}`);
    assert.equal(after.transform, initial.transform, `pan/zoom: mask ${mask}`);
    assert.equal(after.inspector, initial.inspector, `Inspector content: mask ${mask}`);
    assert.equal(after.inspectorHidden, initial.inspectorHidden, `Inspector visibility: mask ${mask}`);
    assert.equal(JSON.stringify(after.overflow), '[]', `card overflow: mask ${mask}`);
    const counts = await tab.playwright.locator('#pypaTaskScene').evaluate((scene) => ({
      shape: scene.querySelectorAll('.kf-pypa-graph__node-shape').length,
      precision: scene.querySelectorAll('[data-overlay-field="precision"]').length,
      memory: scene.querySelectorAll('[data-overlay-field="memory"]').length,
      control: [...scene.querySelectorAll('[data-pypa-edge][data-edge-type="control"]')].map((e) => [getComputedStyle(e).strokeDasharray, getComputedStyle(e).opacity]),
      state: scene.querySelectorAll('[data-pypa-edge][data-edge-type="state"]').length,
      feedback: [...scene.querySelectorAll('[data-pypa-edge][data-edge-type="feedback"]')].map((e) => getComputedStyle(e).strokeDasharray),
    }));
    for (const [key,bit] of [['shape',1],['precision',2],['memory',4]]) assert.equal(counts[key] > 0, !!(mask & bit), `${key}: mask ${mask}`);
    assert.ok(counts.control.length && (counts.state || counts.feedback.length), 'real control and state dependencies survive');
    for (const [dash, opacity] of counts.control) assert.equal(dash !== 'none' && opacity === '1', !!(mask & 8));
    for (const dash of counts.feedback) assert.equal(dash !== 'none', !!(mask & 8));
    states.push({mask, nodeCount:after.nodes.length, edgeCount:after.edges.length, fields:counts});
  }
  // Native buttons support Space and Enter without custom keyboard handlers.
  const shape = tab.playwright.locator('[data-pypa-overlay="shape"]');
  await shape.press('Space'); assert.equal(await shape.getAttribute('aria-pressed'), 'false');
  await shape.press('Enter'); assert.equal(await shape.getAttribute('aria-pressed'), 'true');
  const beforeClose = await capture(tab);
  await tab.playwright.locator('[data-pypa-inspector-close]').click();
  const closed = await capture(tab);
  assert.equal(closed.inspectorHidden,true);
  assert.equal(closed.transform,beforeClose.transform);
  return {passed:true, combinations:states.length, keyboard:true, closePreservesTransform:true, states};
}
export {run,capture};
