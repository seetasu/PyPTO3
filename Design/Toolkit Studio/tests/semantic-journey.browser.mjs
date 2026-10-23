/* Run with the Codex browser tab API after opening PA's compute view at desktop
 * width, collapsed composites, and Q visible. Reads DOM only; actions use locators. */
import assert from 'node:assert/strict';
export async function snapshot(tab) {
  return tab.playwright.locator('#pypaTaskScene').evaluate(scene => {
    const pane = scene.closest('.kf-pypa-graph'), bar = pane.querySelector('[data-pypa-journey]');
    return {anchor:bar.dataset.anchorId,direction:bar.dataset.direction,active:!bar.hidden,
      transform:scene.style.transform,
      nodes:[...scene.querySelectorAll('[data-pypa-node]')].map(n=>[n.dataset.pypaNode,n.style.left,n.style.top,n.dataset.journeyRole,n.classList.contains('is-selected')]),
      edges:[...scene.querySelectorAll('[data-pypa-edge]')].map(e=>[e.dataset.pypaEdge,e.classList.contains('is-journey-path')]),
      inspectorHidden:pane.querySelector('#pypaNodeInspector').hidden,
      inspectorText:pane.querySelector('#pypaNodeInspector').textContent,
      viewport:{w:pane.querySelector('#pypaTaskViewport').clientWidth,h:pane.querySelector('#pypaTaskViewport').clientHeight},
      mapRect:[...pane.querySelector('[data-pypa-mini-viewport]').attributes].filter(a=>['x','y','width','height'].includes(a.name)).map(a=>[a.name,Number(a.value)]),
      sceneSize:{w:Number(scene.style.width.replace('px','')),h:Number(scene.style.height.replace('px',''))},
      pathList:[...pane.querySelectorAll('[data-pypa-journey-node]')].map(n=>n.dataset.pypaJourneyNode),
    };
  });
}
export async function pathStep(tab,id) {
  await tab.playwright.locator('.kf-pypa-graph__journey-list > summary').click();
  await tab.playwright.locator(`[data-pypa-journey-node="${id}"]`).click();
}
export async function run(tab) {
  if (await tab.playwright.locator('[data-pypa-journey-exit]').isVisible()) await tab.playwright.locator('[data-pypa-journey-exit]').click();
  if (await tab.playwright.locator('[data-pypa-inspector-close]').isVisible()) await tab.playwright.locator('[data-pypa-inspector-close]').click();
  await tab.playwright.locator('[data-pypa-node="q"]').click();
  await tab.playwright.locator('[data-pypa-journey-start="q"]').click();
  assert.equal((await snapshot(tab)).direction,'both');
  await tab.playwright.locator('[data-pypa-journey-direction="downstream"]').click();
  const q=await snapshot(tab);
  assert.equal(q.anchor,'q');assert.ok(q.nodes.some(n=>n[0]==='output'&&n[3]==='path'));
  for (const id of ['k-cache','v-cache','k-proj','v-proj','aux']) assert.ok(!q.nodes.some(n=>n[0]===id&&['path','anchor'].includes(n[3])),id);
  // Overlay changes preserve directed path sets, geometry and the exploration origin.
  for (const key of ['shape','precision','memory','control']) {
    await tab.playwright.locator(`[data-pypa-overlay="${key}"]`).click();
    const after=await snapshot(tab);
    assert.equal(after.anchor,q.anchor);assert.equal(after.direction,q.direction);
    assert.equal(after.transform,q.transform);assert.deepEqual(after.nodes,q.nodes);assert.deepEqual(after.edges,q.edges);
    await tab.playwright.locator(`[data-pypa-overlay="${key}"]`).click();
  }
  // Inspect a path step without redefining Q as the original anchor.
  await pathStep(tab,'qk');
  let open=await snapshot(tab);assert.equal(open.anchor,'q');assert.match(open.inspectorText,/Shape Transformation/);
  await tab.playwright.locator('#pypaNodeInspector [data-pypa-source-range="0"]').click();
  assert.equal((await snapshot(tab)).anchor,'q');
  assert.ok(await tab.playwright.locator('#dslEditor .is-pypa-source-evidence').count());
  await tab.playwright.locator('[data-pypa-inspector-close]').click();
  await tab.playwright.locator('[data-pypa-node="qk"]').click(); // explicit Composite expansion
  open=await snapshot(tab);assert.equal(open.anchor,'q');
  assert.ok(open.nodes.some(n=>n[0]==='qk/q-tile'&&n[3]==='path'));
  assert.ok(!open.nodes.some(n=>n[0]==='qk/k-page'&&n[3]==='path'));
  assert.ok(open.edges.some(e=>e[0]==='qk/matmul>score:Scores'&&e[1]));
  await tab.playwright.locator('[data-pypa-inspector-close]').click();
  await pathStep(tab,'qk/q-tile');
  await tab.playwright.locator('[data-pypa-journey-start="qk/q-tile"]').click();
  assert.equal((await snapshot(tab)).anchor,'qk/q-tile');
  await tab.playwright.locator('[data-pypa-inspector-close]').click();
  await pathStep(tab,'qk');
  await tab.playwright.locator('[data-pypa-inspector-close]').click();
  await tab.playwright.locator('[data-pypa-node="qk"]').click(); // fold origin's parent
  const folded=await snapshot(tab);
  assert.equal(folded.anchor,'qk/q-tile');
  assert.ok(folded.nodes.some(n=>n[0]==='qk'&&n[3]==='anchor'));
  assert.equal(folded.nodes.some(n=>n[0]==='qk/q-tile'),false);
  await tab.playwright.locator('[data-pypa-inspector-close]').click();
  assert.equal((await snapshot(tab)).active,true);
  await tab.playwright.locator('[data-pypa-journey-exit]').click();
  const exited=await snapshot(tab);assert.equal(exited.active,false);assert.ok(exited.nodes.some(n=>n[4]));
  // Minimap affine viewport agrees with the main canvas after each controller action.
  const checkMap = async () => {
    const shot=await snapshot(tab),matches=shot.transform.match(/translate\(([-\d.]+)px, ([-\d.]+)px\) scale\(([-\d.]+)\)/);
    const [,tx,ty,scale]=matches.map(Number), miniScale=Math.min(150/shot.sceneSize.w,180/shot.sceneSize.h);
    const rect=Object.fromEntries(shot.mapRect),ox=(150-shot.sceneSize.w*miniScale)/2,oy=(180-shot.sceneSize.h*miniScale)/2;
    for (const [actual,expected] of [[rect.x,ox-tx/scale*miniScale],[rect.y,oy-ty/scale*miniScale],[rect.width,shot.viewport.w/scale*miniScale],[rect.height,shot.viewport.h/scale*miniScale]]) // CSSOM rounds serialized transforms; allow 0.005 minimap pixels, not a layout mismatch.
      assert.ok(Math.abs(actual-expected)<.005, `Minimap ${actual} != ${expected}`);
    return shot;
  };
  for(const action of ['in','out','fit','reset','actual']) {await tab.playwright.locator(`[data-pypa-op="${action}"]`).click();await checkMap();}
  let before=await snapshot(tab);
  await tab.playwright.locator('[data-pypa-minimap-svg]').click();
  let after=await checkMap();
  assert.equal(after.transform.split('scale(')[1],before.transform.split('scale(')[1]);
  await tab.playwright.locator('[data-pypa-minimap-svg]').press('ArrowDown');await checkMap();
  for(const stage of await tab.playwright.locator('[data-pypa-stage]').all()) {
    before=await snapshot(tab);await stage.click();after=await checkMap();
    assert.equal(after.transform.split('scale(')[1],before.transform.split('scale(')[1]);
  }
  await tab.playwright.locator('[data-pypa-minimap-toggle]').click();
  assert.equal(await tab.playwright.locator('[data-pypa-minimap]').isVisible(),false);
  await tab.playwright.locator('[data-pypa-minimap-toggle]').click();
  assert.equal(await tab.playwright.locator('[data-pypa-minimap]').isVisible(),true);
  return {passed:true,qReachesOutput:true,otherInputsExcluded:true,overlays:4,sourceLinked:true,compositeContinuity:true,foldedOrigin:true,minimapActions:7,stageNavigation:true};
}
