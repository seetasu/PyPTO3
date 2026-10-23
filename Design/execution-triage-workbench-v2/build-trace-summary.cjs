const { spawn } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const zip = process.argv[2];
const member = 'dsv4-flash-dspark-a3-dp4tp4ep16-gbs256-in256o64/dsv4-flash-dspark-a3-dp4tp4ep16-gbs256-in256o64.json';
const child = spawn('tar', ['-xOf', zip, member], { stdio: ['ignore', 'pipe', 'inherit'] });
const cats = new Map(), names = new Map(), phases = new Map();
let depth=0, inString=false, escaped=false, buf='', events=0, firstTs=null, lastTs=null, minTs=Infinity, maxTs=-Infinity;
function add(map,key,dur){ const x=map.get(key)||{count:0,durUs:0}; x.count++; x.durUs+=dur||0; map.set(key,x); }
function accept(text){let e; try{e=JSON.parse(text)}catch{return} events++; const cat=e.cat||'(none)', name=e.name||'(unnamed)', dur=Number(e.dur)||0; add(cats,cat,dur); add(phases,e.ph||'(none)',dur); if(dur) add(names,`${cat} · ${name}`,dur); const ts=Number(e.ts); if(Number.isFinite(ts)){minTs=Math.min(minTs,ts);maxTs=Math.max(maxTs,ts)} }
child.stdout.setEncoding('utf8');
child.stdout.on('data', chunk=>{for(let i=0;i<chunk.length;i++){const c=chunk[i];if(depth===0){if(c==='{'){depth=1;buf='{';inString=false;escaped=false}continue}buf+=c;if(inString){if(escaped)escaped=false;else if(c==='\\')escaped=true;else if(c==='"')inString=false;continue}if(c==='"'){inString=true;continue}if(c==='{')depth++;else if(c==='}')depth--;if(depth===0){accept(buf);buf=''}}});
child.on('close', code=>{if(code!==0){process.exitCode=code||1;return}const top=map=>[...map.entries()].sort((a,b)=>b[1].durUs-a[1].durUs).slice(0,40).map(([key,v])=>({key,count:v.count,durMs:+(v.durUs/1000).toFixed(3)}));const out={schemaVersion:1,source:'PyTorch profiler Chrome trace from archive',eventCount:events,traceSpanMs:+((maxTs-minTs)/1000).toFixed(3),timestampRangeUs:[minTs,maxTs],categories:top(cats),phases:top(phases),topEvents:top(names)};fs.writeFileSync(path.join(__dirname,'trace-summary.json'),JSON.stringify(out,null,2)+'\n');console.log(JSON.stringify({events,traceSpanMs:out.traceSpanMs,categories:out.categories.slice(0,20),phases:out.phases.slice(0,12),topEvents:out.topEvents.slice(0,15)},null,2));});
