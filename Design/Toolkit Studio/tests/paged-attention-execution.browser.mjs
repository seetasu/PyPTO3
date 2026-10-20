/* Run with the documented Codex tab API after opening Task 0 → 计算流程. */
import assert from 'node:assert/strict';
async function capture(tab) {
  return tab.playwright.locator('#pypaTaskScene').evaluate(scene => ({
    transform: scene.style.transform,
    nodes: [...scene.querySelectorAll('[data-pypa-node]')].map(n => [n.dataset.pypaNode,n.style.left,n.style.top,n.style.width,n.style.minHeight,n.getAttribute('aria-expanded')]),
    edges: [...scene.querySelectorAll('[data-pypa-edge]')].map(e => [e.dataset.pypaEdge,e.getAttribute('d')]),
    weighted: [...scene.querySelectorAll('[data-pypa-edge]')].filter(e => !!e.style.strokeWidth).map(e => [e.dataset.pypaEdge,e.style.strokeWidth]),
    selection: [...scene.querySelectorAll('[data-pypa-node].is-selected')].map(n => n.dataset.pypaNode),
    counts: [...scene.querySelectorAll('[data-pypa-execution-count]')].map(n => [n.dataset.pypaExecutionCount,n.textContent]),
    detachedCounts: [...scene.querySelectorAll('[data-pypa-execution-count]')].filter(n => !n.closest('[data-pypa-node]')).length,
    stacks: [...scene.querySelectorAll('[data-pypa-execution-stack]')].map(n => ({ id: n.dataset.pypaExecutionStack,
      layers: n.children.length, pointerEvents: getComputedStyle(n).pointerEvents, hidden: n.getAttribute('aria-hidden'),
      instances: [...n.children].map(layer => Number(layer.dataset.pypaExecutionInstance)),
      footprint: Math.max(...[...n.children].map(layer => Number(layer.style.getPropertyValue('--execution-offset').replace('px','')))) })),
  }));
}
export async function run(tab) {
  const input = () => tab.playwright.locator('[data-pypa-execution-length]');
  await input().fill('1025');
  const original = await capture(tab);
  assert.ok(original.counts.some(([id,text]) => id === 'qk' && text === '×9 Pages'));
  assert.equal(original.detachedCounts,0,'counts belong to their node');
  assert.equal(original.stacks.length,0,'collapsed stages have badges only');
  for (const [len, summary] of [['513','5 Pages · 2 Stacks · 4 Pipeline ticks'],['129','2 Pages · 1 Stacks · 3 Pipeline ticks'],['3584','28 Pages · 7 Stacks · 9 Pipeline ticks']]) {
    await input().fill(len);
    assert.equal(await tab.playwright.locator('[data-pypa-execution-summary] strong').innerText(),summary);
    const after = await capture(tab);
    for (const key of ['transform','nodes','edges','selection']) assert.deepEqual(after[key],original[key],key);
  }
  await input().fill('0');
  assert.match(await tab.playwright.locator('[data-pypa-execution-summary] strong').innerText(),/28 Pages/);
  await input().fill('1025');
  const toggle = tab.playwright.locator('[data-pypa-overlay="execution"]');
  await toggle.click();
  const off = await capture(tab);
  assert.equal(off.counts.length,0);
  assert.equal(off.stacks.length,0);
  assert.equal(off.weighted.length,0);
  assert.equal(await input().count(),0);
  await toggle.click();
  const on = await capture(tab);
  assert.deepEqual(on,original);
  // QK child has grounded Page counts; no blanket inheritance to every child.
  await tab.playwright.locator('[data-pypa-node="qk"]').press('Enter');
  const expanded = await capture(tab);
  assert.ok(expanded.counts.some(([id,text]) => id === 'qk/matmul' && text === '×9 Pages'));
  assert.ok(!expanded.counts.some(([id]) => id === 'qk/q-tile'));
  for (const [len, pages] of [['1',1],['2048',16],['3584',28]]) {
    await input().fill(len);
    const after = await capture(tab);
    for (const key of ['transform','nodes','edges','selection']) assert.deepEqual(after[key],expanded[key],key);
    assert.equal(after.detachedCounts,0);
    if (pages === 1) assert.equal(after.stacks.length,0,'one instance needs no underlays');
    else {
      const deck = after.stacks.find(stack => stack.id === 'qk/matmul');
      assert.equal(deck.layers,pages-1);
      assert.equal(deck.instances.join(','),Array.from({length:pages-1},(_,i) => pages-i).join(','));
      assert.ok(deck.footprint <= 18,'dense decks stay within composite padding');
      assert.equal(deck.pointerEvents,'none');
      assert.equal(deck.hidden,'true');
    }
  }
  const deckOn = await capture(tab);
  await toggle.click();
  const deckOff = await capture(tab);
  assert.equal(deckOff.stacks.length,0);
  assert.equal(deckOff.counts.length,0);
  for (const key of ['transform','nodes','edges','selection']) assert.deepEqual(deckOff[key],deckOn[key],key);
  await toggle.click();
  assert.deepEqual(await capture(tab),deckOn,'restore the same repeated cards');
  await input().fill('512');
  const expandedAfter = await capture(tab);
  for (const key of ['transform','nodes','edges']) assert.deepEqual(expandedAfter[key],expanded[key]);
  await tab.playwright.getByRole('button', {name:'任务分配',exact:true}).click();
  assert.equal(await tab.playwright.locator('[data-pa-length]').evaluate(el => el.value),'512');
  await tab.playwright.locator('[data-pypa-view="compute"]').click();
  assert.match(await tab.playwright.locator('[data-pypa-execution-summary]').innerText(),/4 Pages · 1 Stacks · 3 Pipeline ticks/);
  await input().fill('1025');
  return { passed: true, scenarios: ['input boundaries','embedded counts','1/16/28 instance decks','overlay restoration','geometry preservation','expansion','shared scenario after remount'] };
}
