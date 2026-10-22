/* Browser geometry regression for the Graph view. Open visualization2 → Graph,
 * then call inspect(tab) after default render and after expanding a Composite. */
import assert from 'node:assert/strict';

async function inspect(tab) {
  const result = await tab.playwright.locator('#viz2-graph-inter .viz2-graph-viewport').evaluate((viewport) => {
    const scene = viewport.querySelector('[data-viz2-graph-scene]');
    const rect = (element) => {
      const r = element.getBoundingClientRect();
      return { left:r.left, right:r.right, top:r.top, bottom:r.bottom, width:r.width, height:r.height };
    };
    const overlaps = (a,b,tolerance=1) => Math.min(a.right,b.right)-Math.max(a.left,b.left)>tolerance &&
      Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top)>tolerance;
    const scale = Number(scene.style.transform.match(/scale\(([^)]+)\)/)?.[1] || 1);
    const sceneBox = rect(scene);
    const nodes = [...scene.querySelectorAll('[data-pypa-node]')].map((element) =>
      ({ id:element.dataset.pypaNode, element, box:rect(element) }));
    const regions = [...scene.querySelectorAll('[data-pypa-region]')].map((element) =>
      ({ id:element.dataset.pypaRegion, box:rect(element) }));
    const issues = [];
    for (const node of nodes) {
      const expectedWidth = Number(node.element.style.width.replace('px',''))*scale;
      const expectedHeight = Number(node.element.style.height.replace('px',''))*scale;
      if (Math.abs(node.box.width-expectedWidth)>1 || Math.abs(node.box.height-expectedHeight)>1)
        issues.push(`size ${node.id}: DOM ${node.box.width}×${node.box.height}, layout ${expectedWidth}×${expectedHeight}`);
    }
    for (let i=0;i<nodes.length;i++) for (let j=i+1;j<nodes.length;j++) {
      const a=nodes[i],b=nodes[j];
      if (a.id.startsWith(b.id+'/') || b.id.startsWith(a.id+'/')) continue;
      if (overlaps(a.box,b.box)) issues.push(`nodes ${a.id} / ${b.id}`);
    }
    for (let i=0;i<regions.length;i++) for (let j=i+1;j<regions.length;j++) {
      const a=regions[i],b=regions[j];
      if (a.id.includes('/') || b.id.includes('/')) continue;
      if (overlaps(a.box,b.box)) issues.push(`regions ${a.id} / ${b.id}`);
    }
    for (const title of viewport.querySelectorAll('.viz2-graph-stage-title')) {
      if (getComputedStyle(title).visibility==='hidden') continue;
      for (const node of nodes) if (overlaps(rect(title),node.box))
        issues.push(`region title ${title.dataset.viz2Stage} / ${node.id}`);
    }
    const byId = new Map(nodes.map((node) => [node.id,node]));
    const svg = scene.querySelector('svg.kf-pypa-graph__edges');
    for (const path of svg.querySelectorAll('[data-pypa-edge]')) {
      const s=byId.get(path.dataset.edgeSource),t=byId.get(path.dataset.edgeTarget);
      if (!s||!t) { issues.push(`missing endpoint ${path.dataset.pypaEdge}`); continue; }
      const values=(path.getAttribute('d').match(/-?\d+(?:\.\d+)?/g)||[]).map(Number);
      if (values.length<4) { issues.push(`empty route ${path.dataset.pypaEdge}`); continue; }
      const points=[[values[0],values[1]],[values[values.length-2],values[values.length-1]]]
        .map(([x,y])=>({ x:sceneBox.left+x*scale,y:sceneBox.top+y*scale }));
      for (const [point,node] of [[points[0],s],[points[1],t]]) {
        const box=node.box;
        const near=Math.min(Math.abs(point.x-box.left),Math.abs(point.x-box.right),
          Math.abs(point.y-box.top),Math.abs(point.y-box.bottom));
        if (near>2) issues.push(`edge boundary ${path.dataset.pypaEdge} / ${node.id}: ${near.toFixed(1)}px`);
      }
    }
    for (const label of viewport.querySelectorAll('[data-viz2-graph-label]')) {
      if (getComputedStyle(label).visibility==='hidden') continue;
      const node=byId.get(label.dataset.viz2GraphLabel),name=label.querySelector('[data-viz2-graph-name]');
      if (node && name?.textContent!==node.element.title) issues.push(`incomplete title ${node.id}`);
      if (label.scrollWidth>label.clientWidth+1) issues.push(`clipped title ${node?.id}`);
    }
    return { issues, nodeCount:nodes.length, regionCount:regions.length, edgeCount:svg.querySelectorAll('[data-pypa-edge]').length, scale };
  });
  assert.equal(result.issues.length, 0, result.issues.join('\n'));
  return result;
}

export { inspect };
