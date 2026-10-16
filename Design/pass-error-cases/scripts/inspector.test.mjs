import {mkdtemp,readFile,rm} from 'node:fs/promises';
import assert from 'node:assert/strict';
import {build} from 'esbuild';
import {execFileSync} from 'node:child_process';
import {createRequire} from 'node:module';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
const temporary=await mkdtemp(join(tmpdir(),'mi-map-'));
const generated=join(temporary,'memory-map.json');
execFileSync('python3',[
 fileURLToPath(new URL('../../memory-inspector/memory_map.py',import.meta.url)),
 fileURLToPath(new URL('../../assets/32_after_AllocateMemoryAddr.py',import.meta.url)),
 '--output',generated,
]);
const source=await readFile(new URL('../src/inspector-data.js',import.meta.url),'utf8');
const appSource=await readFile(new URL('../src/MemoryInspectorApp.jsx',import.meta.url),'utf8');
const chartSource=await readFile(new URL('../src/FunctionMemoryMap.jsx',import.meta.url),'utf8');
const json=await readFile(new URL('../data/inspector-memory.json',import.meta.url),'utf8');
const canonical=await readFile(new URL('../../assets/32_after_AllocateMemoryAddr.memory_map.json',import.meta.url),'utf8');
const regenerated=await readFile(generated,'utf8');
assert.deepEqual(JSON.parse(regenerated),JSON.parse(canonical));
assert.deepEqual(JSON.parse(json),JSON.parse(regenerated));
const {dump,mapGeometry,memoryBlockTextColor,memorySemanticColor,memorySpaceColor,stats}=await import('data:text/javascript;base64,'+Buffer.from(source.replace("import dump from '../data/inspector-memory.json';",`const dump=${json};`)).toString('base64'));
assert.equal(dump.functions.length,38);
assert.deepEqual(['Vec','Mat','Left','Right','Acc'].map(memorySpaceColor),['var(--gradient)','var(--op)','var(--input)','var(--weight)','var(--aggregate)']);
assert.equal(memorySemanticColor({view:true,op:'tile.reshape'}),'var(--weight)');
assert.equal(memorySemanticColor({view:false,op:'tile.row_sum'}),'var(--aggregate)');
assert.equal(memorySemanticColor({view:false,op:'tile.load'}),'var(--input)');
assert.equal(memorySemanticColor({view:false,op:'tile.muls'}),'var(--op)');
assert.equal(memoryBlockTextColor({space:'Left',view:false,op:'tile.load'},'space',false),'var(--bg)');
assert.equal(memoryBlockTextColor({space:'Mat',view:false,op:'tile.muls'},'space',false),'var(--text)');
const totals=dump.functions.map(stats).reduce((a,s)=>Object.fromEntries(Object.keys(s).map(k=>[k,(a[k]||0)+s[k]])),{});
assert.deepEqual(totals,{boxes:697,views:108,conflicts:0,overflow:0});
for(const fn of dump.functions)for(const box of fn.boxes){
 const space=fn.spaces.find(s=>s.space===box.space),g=mapGeometry(fn,space,box,false,.01);
 assert.equal(g.x,box.offset*.01);assert.equal(g.width,box.size*.01);
 assert.equal(g.y,(box.start-fn.src_start)*17);
 assert.equal(g.height,Math.max(16,(box.end-box.start+1)*17-1));
 assert.equal(g.extent,space.limit);assert.equal(mapGeometry(fn,space,box,true,.01).extent,space.hwm||1);
}
// Exercise the actual chart renderer rather than the discarded storage projection.
const result=await build({stdin:{contents:`import React from 'react';import {renderToStaticMarkup} from 'react-dom/server';import Chart,{verticalProjection,zoomViewport} from './src/FunctionMemoryMap.jsx';import {dump} from './src/inspector-data.js';
const fn=dump.functions[0];export const html=renderToStaticMarkup(React.createElement(Chart,{fn,spaceName:'all',fitUsed:false,selected:0,line:12,onSelect:()=>{}}));
const matEntries=dump.functions.flatMap(f=>f.boxes.map(box=>({box:{...box,functionName:f.name}}))).filter(({box})=>box.space==='Mat');
const projection=verticalProjection({name:'全部函数'},matEntries,17);export const matProjection={height:projection.height,lineScale:projection.lineScale,count:projection.positions.size};export const zoomed=zoomViewport({x:10,y:20,scaleX:1,scaleY:1},.5,{x:100,y:80},.2);`,resolveDir:process.cwd(),loader:'jsx'},bundle:true,platform:'node',format:'cjs',jsx:'automatic',write:false,loader:{'.svg':'text','.css':'empty'}});
const module={exports:{}};new Function('require','module','exports',result.outputFiles[0].text)(createRequire(import.meta.url),module,module.exports);const {html}=module.exports;
assert.match(html,/Function Memory Map/);assert.match(html,/t__tile/);assert.match(html,/184 KiB/);assert.match(html,/L12/);assert.doesNotMatch(html,/memory-space|memory-base-grid/);
assert.match(html,/fill="var\(--gradient\)"/);assert.doesNotMatch(html,/fill-opacity/);assert.doesNotMatch(html,/fill="var\(--line\)"/);assert.match(html,/stroke="var\(--stage\)" stroke-width="2" paint-order="stroke fill"/);assert.doesNotMatch(html,/stroke-dasharray/);
assert.match(html,/function \/ line/);assert.doesNotMatch(html,/NaN KiB/);
assert.match(html,/向量/);assert.match(html,/fill="var\(--text\)"/);assert.match(html,/fill="var\(--bg\)"/);assert.match(html,/overflow:hidden/);assert.match(html,/data-tooltip=/);
assert.equal(module.exports.matProjection.count,64);assert.ok(module.exports.matProjection.height>1100);assert.equal(module.exports.matProjection.lineScale,17);
assert.deepEqual(module.exports.zoomed,{x:10,y:50,scaleX:1,scaleY:.5});
assert.match(appSource,/内存图筛选/);assert.match(appSource,/className="case-workflow"/);assert.match(appSource,/<details><summary>源码/);
assert.match(appSource,/染色方式/);assert.match(appSource,/CanvasTooltip/);assert.doesNotMatch(appSource,/<h3>内存空间<\/h3>/);
assert.match(appSource,/className="inspector overlay-panel inspector-right"/);assert.doesNotMatch(appSource,/data-liquid-glass="panel"/);
assert.match(chartSource,/addEventListener\('wheel',wheel,\{passive:false\}\)/);assert.match(chartSource,/setPointerCapture/);assert.match(chartSource,/固定顶部地址轴/);assert.match(chartSource,/固定左侧函数与源码行轴/);assert.doesNotMatch(chartSource,/scale\(\$\{|<title>/);
assert.match(chartSource,/inset:'128px 28px 0'/);assert.match(chartSource,/当前横纵比例/);assert.match(chartSource,/scaleX:1,scaleY:minScaleY/);
assert.doesNotMatch(appSource,/Function Memory Map 控件|case-pipeline-note|flexWrap|pass-rail|canvas-tools|rail-play|MemoryZoom|MemoryRail|<h3>源码|CAP|USED/);
await rm(temporary,{recursive:true,force:true});
console.log('Function Memory Map: Python AST output exactly matches the 38-function source dataset; 697 coordinates render with the CP-01 palette and a one-pixel outer background separation.');
