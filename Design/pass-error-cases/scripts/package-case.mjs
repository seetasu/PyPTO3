import {readFile,writeFile} from 'node:fs/promises';
let html=await readFile('dist/entry.html','utf8');
const script=html.match(/<script[^>]*src="([^"]+)"[^>]*><\/script>/),style=html.match(/<link[^>]*href="([^"]+\.css)"[^>]*>/);
if(!script||!style)throw Error('Missing bundle');
const js=(await readFile('dist/'+script[1],'utf8')).replaceAll('</script','<\\/script');
const css=await readFile('dist/'+style[1],'utf8');
html=html.replace(script[0],'').replace(style[0],()=>'<style>'+css+'</style>').replace('</body>',()=>'<script>'+js+'</script></body>').replace('CSA Pass Field · qk_pv','Pass Error Cases · CP-01');
await writeFile('index.html',html);
console.log('Standalone CP-01 packaged');
