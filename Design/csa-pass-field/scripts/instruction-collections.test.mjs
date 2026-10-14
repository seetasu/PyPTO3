import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { instructionCollections } from '../src/instruction-collections.js';
import { layoutVisibleGraph } from '../src/layout.js';
const read = name => JSON.parse(fs.readFileSync(new URL(`../data/csa-qk-pv/${name}.json`, import.meta.url)));
const isa = read('isa');
const groups = instructionCollections(isa.instructions);
test('every static ISA occurrence appears once in a kernel-scoped collection', () => {
  const instances = groups.flatMap(group => group.instances);
  assert.equal(instances.length, isa.instructions.reduce((n, instruction) => n + instruction.occurrences.length, 0));
  assert.equal(new Set(instances.map(item => item.line)).size, instances.length);
  for (const group of groups) assert.ok(group.instances.every(item => item.kernel === group.kernel));
  assert.deepEqual(groups.filter(group => group.label === 'TLOAD').map(group => group.instances.length), [11, 1]);
});
test('matmul source locations distinguish QK and PV including anonymous operands', () => {
  for (const label of ['TMATMUL', 'TMATMUL.ACC']) {
    const selected = groups.filter(group => group.label === label);
    assert.deepEqual(selected.map(group => [group.owner, group.instances.length]), [['op.qk', 4], ['op.pv', 4]]);
  }
});
test('every collection has a vertical owner edge and disjoint layout, folded or expanded', () => {
  const graph = read('graph'), views = read('views');
  const nodes = [...graph.nodes.filter(node => !['isa', 'context'].includes(node.renderKind)), ...groups];
  const edges = [...graph.edges.filter(edge => !edge.target.startsWith('isa.')), ...groups.map(group => ({ source: group.owner, target: group.id, semanticKind: group.owner.startsWith('kernel.') ? 'lineage' : 'lowering' }))];
  for (const loadExpanded of [false, ...groups.map(group => group.id)]) {
    const { positions, sizes } = layoutVisibleGraph(nodes, edges, views.views.front, {loadExpanded,referenceNodes:graph.nodes,referenceEdges:graph.edges});
    for (const group of groups) {
      const p = positions.get(group.id), s = sizes.get(group.id), owner = positions.get(group.owner), os = sizes.get(group.owner);
      assert.ok(Number.isFinite(p?.x) && Number.isFinite(p?.y));
      assert.ok(Math.abs(p.x+s.width/2-owner.x-os.width/2)<.001);
    }
    for (const owner of new Set(groups.map(group=>group.owner))) {
      const column=groups.filter(group=>group.owner===owner);
      for(let i=1;i<column.length;i++) {
        const previous=column[i-1], current=column[i];
        assert.equal(positions.get(current.id).y - positions.get(previous.id).y - sizes.get(previous.id).height,16);
      }
    }
    for (let i=0;i<groups.length;i++) for(let j=i+1;j<groups.length;j++) {
      const a=positions.get(groups[i].id),as=sizes.get(groups[i].id),b=positions.get(groups[j].id),bs=sizes.get(groups[j].id);
      assert.ok(a.x+as.width<=b.x || b.x+bs.width<=a.x || a.y+as.height<=b.y || b.y+bs.height<=a.y);
    }
  }
});
