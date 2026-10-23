import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { layoutVisibleGraph } from '../src/layout.js';
import { instructionCollections } from '../src/instruction-collections.js';
const read=n=>JSON.parse(fs.readFileSync(new URL(`../data/csa-qk-pv/${n}.json`,import.meta.url)));
const graph=read('graph'), passes=read('passes').passes, view=read('views').views.front;
const groups=instructionCollections(read('isa').instructions);
test('unobstructed operator-to-ISA side ports share the same horizontal axis',()=>{
  const nodes=[...graph.nodes.filter(n=>!['isa','context'].includes(n.renderKind)),...groups];
  const {positions,sizes}=layoutVisibleGraph(nodes,graph.edges,view,{layout:'vertical'});
  for(const owner of ['op.exp','op.rowsum','op.pv']) {
    const first=groups.find(n=>n.owner===owner);
    assert.equal(positions.get(first.id).y+sizes.get(first.id).height/2,
      positions.get(owner).y+sizes.get(owner).height/2,owner);
  }
});
test('vertical layout keeps every pass and expanded ISA group disjoint',()=>{
  const active=new Set(graph.initialState.nodeIds);
  const registry=new Map(graph.nodes.map(n=>[n.id,structuredClone(n)]));
  for(const pass of passes) {
    for(const op of pass.delta) {
      if(op.op==='addNode')active.add(op.id);
      if(op.op==='removeNode')active.delete(op.id);
      if(op.op==='updateNode')Object.assign(registry.get(op.id),structuredClone(op.after));
    }
    const base=[...active].map(id=>registry.get(id)).filter(n=>n.renderKind!=='context'&&n.renderKind!=='isa'&&n.attributes?.present!==false);
    for(const expanded of pass.index===51?[false,...groups.map(n=>n.id)]:[false]) {
      const nodes=pass.index===51?[...base,...groups]:base;
      const {positions,sizes}=layoutVisibleGraph(nodes,graph.edges,view,{layout:'vertical',loadExpanded:expanded});
      const leaves=nodes.filter(n=>!['scope','kernel','loop'].includes(n.renderKind));
      for(const n of nodes)assert.ok(Number.isFinite(positions.get(n.id)?.x),n.id);
      for(let i=0;i<leaves.length;i++)for(let j=i+1;j<leaves.length;j++) {
        const a=leaves[i].id,b=leaves[j].id,p=positions.get(a),q=positions.get(b),s=sizes.get(a),t=sizes.get(b);
        assert.ok(p.x+s.width<=q.x||q.x+t.width<=p.x||p.y+s.height<=q.y||q.y+t.height<=p.y,`pass ${pass.index}: ${a} overlaps ${b}`);
      }
      for(const core of ['aic','aiv']) {
        const id=`kernel.${core}`;if(!active.has(id))continue;
        const p=positions.get(id),s=sizes.get(id);
        for(const member of view.layout.containerMembers[id])if(positions.has(member)) {
          const q=positions.get(member),t=sizes.get(member);
          assert.ok(q.x>=p.x&&q.y>=p.y+120&&q.x+t.width<=p.x+s.width&&q.y+t.height<=p.y+s.height,member);
        }
      }
    }
  }
});
