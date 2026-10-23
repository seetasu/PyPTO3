/* Run from Toolkit Studio with the page served at BASE_URL (default :5500).
 * PLAYWRIGHT_MODULE may point to a bundled Playwright installation.
 * Uses an isolated Chrome profile; never attaches to the user's browser.
 */
const assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.BASE_URL||'http://127.0.0.1:5500/Design/Toolkit%20Studio/index_v2-slz.html';
async function run(){
  const browser=await chromium.launch({channel:'chrome',headless:true});
  const page=await browser.newPage({viewport:{width:1600,height:1000}}),errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  const state=()=>page.evaluate(()=>PtoVisualization2.getState());
  const settled=async id=>{await page.waitForFunction(id=>PtoVisualization2.getState().selection?.task===id,id);await page.waitForTimeout(80);};
  const wait=ms=>page.waitForTimeout(ms);
  const current=()=>page.locator('#viz2-pipeline-intra .viz2-intra-card.is-current');
  const nav=async id=>{await page.locator(`[data-intra-task-go="${id}"]`).click();await settled(id);};
  const wheel=async(deltaX,deltaY=0,extra={})=>page.locator('#viz2-intra-carousel').dispatchEvent('wheel',{deltaX,deltaY,...extra});
  try{
    await page.goto(base);await page.waitForFunction(()=>!!window.PtoVisualization2);
    const opener=page.locator('[data-visualization2-open]');
    console.log('openers',await opener.count());
    // The entry lives in the paged_attention source tab's visualization actions.
    await opener.first().evaluate(el=>el.click());
    await page.locator('[data-viz2-tab="pipeline"]').click();
    await page.locator('[data-bem-task="3"]').dblclick();await settled(3);await wait(900);
    assert.equal((await state()).views.pipeline.subview,'block');
    assert.equal(await page.locator('.viz2-intra-card').count(),6);
    assert.equal(await page.locator('.viz2-intra-reference').count(),0);
    assert.ok(await current().locator('h3').isVisible());
    const geometry=await page.locator('#viz2-intra-carousel').evaluate(viewport=>{
      const box=viewport.getBoundingClientRect(),cards=[...viewport.querySelectorAll('[data-intra-task]')],current=cards.find(c=>c.classList.contains('is-current')),r=current.getBoundingClientRect();
      return {center:Math.abs(r.x+r.width/2-box.x-box.width/2),width:r.width,visible:cards.filter(c=>{const r=c.getBoundingClientRect();return r.left<box.right&&r.right>box.left;}).length};
    });assert.ok(geometry.center<2);assert.equal(geometry.visible,2);console.log('entry/card geometry',geometry);
    for(const id of [27,51,75,99,123]){await wheel(48);await settled(id);await wait(120);}
    assert.equal((await state()).selection.task,123);console.log('five consecutive gestures passed');
    await wheel(60);await wait(450);assert.equal((await state()).selection.task,123);
    await wheel(-48);await settled(99);
    await nav(3);
    for(const delta of [45,65,40,18,6,3,2]){await wheel(delta);await wait(18);}
    await settled(27);await wait(220);assert.equal((await state()).selection.task,27);console.log('momentum tail passed');
    await nav(3);
    for(const [delta,ms] of [[45,16],[65,16],[20,16],[6,52],[50,16],[65,16],[18,16],[4,16]]){await wheel(delta);await wait(ms);}
    await settled(51);console.log('animation pending/flush passed');
    // Real vertical wheel input must scroll the Task without changing selection.
    const scroll=current().locator('.viz2-intra-tick-scroll');await scroll.hover();
    await page.mouse.wheel(0,340);await wait(200);assert.equal((await state()).selection.task,51);
    const saved=await scroll.evaluate(el=>el.scrollTop);assert.ok(saved>0);console.log('vertical scrolling',saved);
    await nav(75);await nav(51);assert.ok(await current().locator('.viz2-intra-tick-scroll').evaluate(el=>el.scrollTop)>0);
    // Arrow controls, keyboard, Shift+wheel and dragging use the same bounded navigation.
    await page.locator('[data-intra-next]').click();await settled(75);
    await page.locator('[data-intra-prev]').click();await settled(51);
    await page.locator('#viz2-intra-carousel').focus();await page.keyboard.press('ArrowLeft');await settled(27);
    await wheel(0,48,{shiftKey:true});await settled(51);
    const box=await current().boundingBox();await page.mouse.move(box.x+box.width*.7,box.y+75);await page.mouse.down();await page.mouse.move(box.x+box.width*.2,box.y+75,{steps:12});await page.mouse.up();await settled(75);console.log('all navigation controls passed');
    // Every card uses its own operations, marker IDs, and original model values.
    const semantics=await page.locator('#viz2-pipeline-intra').evaluate(host=>{
      const cards=[...host.querySelectorAll('[data-intra-task]')],markers=[...host.querySelectorAll('marker')].map(el=>el.id);
      return {counts:cards.map(card=>({task:Number(card.dataset.intraTask),operations:card.querySelectorAll('[data-intra-kind="stage"]').length,events:card.querySelectorAll('[data-intra-kind="event"]').length})),markersUnique:markers.length===new Set(markers).size};
    });assert.ok(semantics.markersUnique);assert.equal(semantics.counts.find(t=>t.task===27).operations,4);assert.equal(semantics.counts.find(t=>t.task===51).operations,24);console.log('card-local semantics',semantics);
    await nav(51);
    if(process.env.SCREENSHOT)await page.screenshot({path:process.env.SCREENSHOT});
    for(const selector of ['[data-intra-kind="tick"][data-intra-index="0"]','[data-intra-kind="stack"][data-intra-index="0"]','[data-intra-kind="stage"][data-intra-index="0"][data-intra-stage="qk"]','[data-intra-kind="slot"][data-intra-buffer="score_transfer"][data-intra-from="0"]','[data-intra-kind="output"]']){
      const object=current().locator(selector);await object.evaluate(el=>el.scrollIntoView({block:'center',inline:'nearest'}));
      // SVG path reuse targets have no box center on the stroke. Keyboard activation tests its accessible action.
      await object.focus();await page.keyboard.press('Enter');
      assert.ok(await page.locator('#viz2-pipeline-intra aside').isVisible());
      assert.ok((await page.locator('[data-intra-detail]').textContent()).length>80);
      await page.locator('[data-intra-close]').click();
    }
    console.log('Tick/Stack/Stage/Slot/Output Inspectors passed');
    await current().locator('.viz2-intra-tick-scroll').evaluate(el=>el.scrollTop=0);
    await current().locator('[data-intra-kind="event"][data-intra-index="0"][data-intra-event="QK_READY_EVENT"]').click();
    await page.locator('[data-intra-sync-source="producer"]').click();assert.ok((await page.locator('[data-intra-source-preview]').textContent()).includes('set'));
    await page.locator('[data-intra-sync-source="consumer"]').click();assert.ok(await page.locator('[data-intra-source-preview]').isVisible());
    if(process.env.SCREENSHOT)await page.screenshot({path:process.env.SCREENSHOT.replace('.png','-ready.png')});
    await page.locator('[data-intra-graph-ready]').click();await wait(850);
    assert.equal((await state()).activeTab,'graph');assert.equal((await state()).selection.task,51);assert.equal((await state()).selection.subobject.kind,'event');
    await page.locator('[data-viz2-tab="pipeline"]').click();await wait(850);assert.equal((await state()).selection.subobject.event,'QK_READY_EVENT');
    await page.locator('[data-viz2-tab="assignment"]').click();await wait(850);assert.equal((await state()).selection.task,51);
    await page.locator('[data-viz2-tab="pipeline"]').click();await wait(850);assert.equal((await state()).selection.task,51);console.log('READY evidence and cross-tab identity passed');
    // Select a deep row via cross-view mapping and verify its Task-local node is revealed.
    await page.evaluate(()=>PtoVisualization2.dispatch({type:'pipeline-intra-focus',focus:{kind:'stage',index:5,stage:'accumulate'}}));await wait(100);
    const deep=await current().locator('[data-intra-kind="stage"][data-intra-index="5"][data-intra-stage="accumulate"]').evaluate(el=>{const r=el.getBoundingClientRect(),v=el.closest('.viz2-intra-tick-scroll').getBoundingClientRect();return r.top>=v.top-1&&r.bottom<=v.bottom+1;});assert.ok(deep);
    await page.locator('[data-intra-close]').click();await nav(75);assert.equal((await state()).selection.subobject,null);await nav(51);assert.equal((await state()).selection.subobject.index,5);console.log('Task-local focus memory/reveal passed');
    await page.locator('[data-intra-close]').click();
    for(const width of [1100,760]){await page.setViewportSize({width,height:900});await wait(100);const overflow=await page.locator('#viz2-pipeline-intra').evaluate(el=>el.scrollWidth>el.clientWidth+1);assert.equal(overflow,false);if(width===760&&process.env.SCREENSHOT)await page.screenshot({path:process.env.SCREENSHOT.replace('.png','-narrow.png')});}
    await page.setViewportSize({width:1600,height:1000});await wait(100);
    await wheel(0,-100,{ctrlKey:true});await wait(80);assert.ok((await state()).views.pipeline.intraCameras[51].scale>1);assert.equal((await state()).selection.task,51);
    await page.locator('[data-intra-fit]').click();await wait(80);assert.equal((await state()).views.pipeline.intraCameras[51].scale,1);
    await page.locator('[data-intra-back]').click();await wait(850);assert.equal((await state()).views.pipeline.subview,null);
    await page.locator('[data-bem-task="27"]').dblclick();await settled(27);await wait(850);
    assert.equal(await page.locator('.viz2-semantic-overlay,.viz2-transition-overlay,.viz2-intra-masked').count(),0);console.log('enter/exit morph and resize passed');
    // Interrupt a semantic morph and a carousel animation; neither may leave masks or pending work.
    await page.locator('[data-intra-back]').click();await wait(40);await page.locator('[data-viz2-tab="assignment"]').click();await wait(850);await page.locator('[data-viz2-tab="pipeline"]').click();await wait(850);
    await page.locator('[data-bem-task="3"]').dblclick();await wait(50);await page.setViewportSize({width:1500,height:950});await wait(850);
    await wheel(48);await wait(60);await page.locator('[data-viz2-tab="assignment"]').click();await wait(850);await page.locator('[data-viz2-tab="pipeline"]').click();await wait(850);
    assert.equal(await page.locator('.viz2-semantic-overlay,.viz2-transition-overlay,.viz2-intra-masked').count(),0);
    await page.emulateMedia({reducedMotion:'reduce'});await nav(3);await wheel(48);await settled(27);await wheel(-48);await settled(3);
    assert.equal(await page.locator('[data-intra-prev]').isDisabled(),true);console.log('interruptions and reduced-motion passed');
    await page.emulateMedia({reducedMotion:'no-preference'});await nav(51);
    assert.deepEqual(errors,[]);console.log('PASS: Pipeline carousel browser checks; no page errors');
  }finally{await browser.close();}
}
run().catch(error=>{console.error(error);process.exitCode=1;});
