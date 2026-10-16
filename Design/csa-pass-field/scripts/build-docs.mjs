// Render the repository docs' supported Markdown subset to static Pages assets.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const output=path.join(root,'public/docs');
fs.mkdirSync(output,{recursive:true});
const escape=text=>text.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
function inline(text){return escape(text).replace(/`([^`]+)`/g,'<code>$1</code>').replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>').replace(/\[([^\]]+)\]\(([^)]+)\)/g,(_,label,url)=>`<a href="${url.replace(/\.md$/,'\.html')}">${label}</a>`);}
function render(markdown){
 const lines=markdown.split('\n'),out=[];let i=0;
 while(i<lines.length){
  const line=lines[i];if(!line.trim()){i++;continue;}
  if(line.startsWith('```')){const code=[];i++;while(i<lines.length&&!lines[i].startsWith('```'))code.push(lines[i++]);i++;out.push('<pre><code>'+escape(code.join('\n'))+'</code></pre>');continue;}
  const heading=line.match(/^(#{1,6}) (.+)$/);if(heading){out.push(`<h${heading[1].length}>${inline(heading[2])}</h${heading[1].length}>`);i++;continue;}
  if(line.startsWith('|')&&/^\|[\s:|\-]+$/.test(lines[i+1]||'')){
   const cells=row=>row.replace(/^\||\|$/g,'').split('|').map(x=>x.trim());
   out.push('<div class="table-scroll"><table><thead><tr>'+cells(line).map(x=>'<th>'+inline(x)+'</th>').join('')+'</tr></thead><tbody>');i+=2;
   while(i<lines.length&&lines[i].startsWith('|'))out.push('<tr>'+cells(lines[i++]).map(x=>'<td>'+inline(x)+'</td>').join('')+'</tr>');out.push('</tbody></table></div>');continue;
  }
  if(/^(?:- |\d+\. )/.test(line)){const ordered=/^\d/.test(line),tag=ordered?'ol':'ul',pattern=ordered?/^\d+\. /:/^- /;out.push('<'+tag+'>');while(i<lines.length&&pattern.test(lines[i]))out.push('<li>'+inline(lines[i++].replace(pattern,''))+'</li>');out.push('</'+tag+'>');continue;}
  const paragraph=[line];i++;while(i<lines.length&&lines[i].trim()&&!/^(?:#|```|\||- |\d+\. )/.test(lines[i]))paragraph.push(lines[i++]);out.push('<p>'+inline(paragraph.join(' '))+'</p>');
 }
 return out.join('\n');
}
const style=`:root{color-scheme:dark;font-family:Arial,"PingFang SC",sans-serif;background:#151515;color:#ddd}body{margin:0}main{max-width:920px;margin:auto;padding:40px 28px 80px}nav{display:flex;gap:24px;margin-bottom:32px;font-size:13px}a{color:#9ac7ed;text-underline-offset:4px}h1{font-size:30px;line-height:1.4;color:#fff}h2{font-size:21px;margin-top:36px;color:#eee}p,li{font-size:15px;line-height:1.9}li{margin:8px 0}code{font:13px ui-monospace,monospace;color:#c7d8e5}pre{padding:20px;border-radius:10px;background:#242424;overflow:auto;line-height:1.7}.table-scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.7}th,td{text-align:left;vertical-align:top;padding:12px;border-bottom:1px solid #383838}th{color:#fff;background:#222}strong{color:#fff}@media(max-width:600px){main{padding:24px 18px}h1{font-size:25px}}`;
for(const name of ['callable-explorer-rules','cross-callable-validation']){
 const markdown=fs.readFileSync(path.join(root,'docs',name+'.md'),'utf8');
 const title=markdown.split('\n')[0].replace(/^# /,'');
 const html=`<!doctype html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${escape(title)}</title><style>${style}</style></head><body><main><nav><a href="../index.html">返回 Callable Explorer</a><a href="./${name}.md" download>下载 Markdown</a></nav>${render(markdown)}</main></body></html>`;
 fs.writeFileSync(path.join(output,name+'.html'),html);fs.writeFileSync(path.join(output,name+'.md'),markdown);
}
fs.mkdirSync(path.join(output,'validation'),{recursive:true});
fs.copyFileSync(path.join(root,'docs/validation/cross-callable-audit.json'),path.join(output,'validation/cross-callable-audit.json'));
console.log('Built two static documentation pages with relative cross-links and audit data');
