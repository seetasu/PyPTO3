import {createServer} from 'node:http';
import {readFile} from 'node:fs/promises';
import {resolve,extname,sep} from 'node:path';
import {fileURLToPath} from 'node:url';
import {importRun} from './scripts/import-run.js';
const root=resolve(fileURLToPath(new URL('.',import.meta.url))),referenceRoot=resolve(root,'../device-perf-lane-20261026');
const args=process.argv.slice(2);let dataset;
try{if(args.length&&(args.length!==2||args[0]!=='--run'))throw Error('用法: node server.js [--run 运行目录]');dataset=args.length?await importRun(args[1]):JSON.parse(await readFile(resolve(root,'data/example.json'),'utf8'));}catch(e){console.error(`启动失败：${e.message}`);process.exit(1);}
const payload=JSON.stringify(dataset),mime={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.json':'application/json; charset=utf-8'};
const bridge=`window.addEventListener('message', e=>{if(e.origin!==location.origin||e.source!==window.parent||e.data?.type!=='perf-lane-drilldown')return;const m=e.data;if(m.scene!=='deepseek')return;if(D.ranks.some(r=>r.rank===m.rank)){$('rank').value=m.rank;loadRank(m.rank);}const t=R.launches.find(x=>x.id===m.task&&window.QK_RESIDENCY.isTarget(x));if(t)select(t.id,Number.isFinite(m.core)?m.core:null,false,true);});\n`;
createServer(async(req,res)=>{try{const pathname=decodeURIComponent(new URL(req.url,'http://localhost').pathname);
 if(pathname==='/drilldown'){let html=await readFile(resolve(referenceRoot,'index.html'),'utf8');const end='  loadRank(D.ranks[0].rank);';if(!html.includes(end))throw Error('reference bridge marker missing');const at=html.lastIndexOf(end);html=(html.slice(0,at)+end+'\n'+bridge+html.slice(at+end.length)).replace('<head>','<head><base href="/reference/">');res.writeHead(200,{'Content-Type':mime['.html']});res.end(html);return;}
 if(pathname.startsWith('/reference/')){const ref=resolve(referenceRoot,'.'+pathname.slice('/reference'.length));if(ref!==referenceRoot&&!ref.startsWith(referenceRoot+sep))throw Error('invalid path');const file=ref===referenceRoot?resolve(referenceRoot,'index.html'):ref;const contents=await readFile(file);res.writeHead(200,{'Content-Type':mime[extname(file)]||'application/octet-stream'});res.end(contents);return;}
 if(pathname==='/api/data'){res.writeHead(200,{'Content-Type':mime['.json'],'Cache-Control':'no-store'});res.end(payload);return;}
 const path=resolve(root,'.'+pathname);if(path!==root&&!path.startsWith(root+sep))throw Error('invalid path');const file=path===root?resolve(root,'index.html'):path;const contents=await readFile(file);res.writeHead(200,{'Content-Type':mime[extname(file)]||'application/octet-stream'});res.end(contents);
 }catch{res.writeHead(404);res.end('Not found');}}).listen(0,'127.0.0.1',function(){console.log(`http://127.0.0.1:${this.address().port}/`);});
