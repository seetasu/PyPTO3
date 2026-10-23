import mergeManifest from '../data/callable/manifest.json';
import mergeIsa from '../data/callable/isa.json';
import mergeGraph from '../data/callable/graph.json';
import qkManifest from '../data/csa-qk-pv/manifest.json';
import qkIsa from '../data/csa-qk-pv/isa.json';
import qkGraph from '../data/csa-qk-pv/graph.json';
import {instructionCollections as mergeCollections} from './instruction-collections.js';
import {instructionCollections as qkCollections} from './qk-instruction-collections.js';

export const samples=[
 {id:'qk_pv',label:'qk_pv',description:'AIC + AIV',manifest:qkManifest,isa:qkIsa,graph:qkGraph,instructionCollections:qkCollections},
 {id:'csa_merge_pack_publish',label:'csa_merge_pack_publish',description:'AIV',manifest:mergeManifest,isa:mergeIsa,graph:mergeGraph,instructionCollections:mergeCollections},
];
export const resolveSample=id=>samples.find(sample=>sample.id===id)||samples[1];
export const sample=resolveSample(typeof location==='undefined'?null:new URLSearchParams(location.hash.slice(1)).get('callable'));
export const isQk=sample.id==='qk_pv';
export function sampleHash(id,hash){
 const params=new URLSearchParams(hash.replace(/^#/,''));params.set('callable',resolveSample(id).id);params.delete('node');params.delete('ranks');params.set('view','compile');
 return '#'+params;
}
