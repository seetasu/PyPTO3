(() => {
  const papers = window.PYPTO_WHITEPAPERS || [];
  if (!papers.length) return;
  const $ = (s, root = document) => root.querySelector(s);
  const tabs = $('#paper-tabs'), toc = $('#toc'), article = $('#article-start');
  const home = $('#home-guide'), readerView = $('#reader-view'), main = $('#top'), rail = $('.rail');
  let current = 0, visualIndex = 0, headings = [], activeObserver;
  const esc = s => s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const slug = s => s.toLowerCase().replace(/[^\p{L}\p{N}]+/gu,'-').replace(/^-|-$/g,'');
  function inline(s) {
    let x = esc(s);
    x = x.replace(/!\[([^\]]*)\]\(([^)]+)\)/g,(_,alt,url) => `<figure class="diagram"><img src="${url}" alt="${alt}" loading="lazy"><figcaption>${alt}</figcaption></figure>`);
    x = x.replace(/`([^`]+)`/g,'<code>$1</code>');
    x = x.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>').replace(/\*(.+?)\*/g,'<em>$1</em>');
    x = x.replace(/\[([^\]]+)\]\(([^)]+)\)/g,(_,label,url) => `<a href="${esc(url)}">${label}</a>`);
    return x;
  }
  function renderMarkdown(md) {
    const lines = md.replace(/\r/g,'').split('\n'); let out='', i=0;
    while (i<lines.length) {
      const line=lines[i];
      if (!line.trim()) { i++; continue; }
      if (line.startsWith('```')) { const lang=line.slice(3).trim(); i++; let code=''; while(i<lines.length&&!lines[i].startsWith('```')) code+=lines[i++]+'\n'; i++; out+=`<pre><code class="language-${esc(lang)}">${esc(code.trimEnd())}</code></pre>`; continue; }
      if (/^\|.*\|$/.test(line.trim()) && i+1<lines.length && /^\|?\s*:?-{3,}/.test(lines[i+1].trim())) {
        const cells=s=>s.trim().replace(/^\||\|$/g,'').split('|').map(v=>v.trim()); const heads=cells(line); i+=2; let rows='';
        while(i<lines.length&&/^\|.*\|$/.test(lines[i].trim())) { rows+='<tr>'+cells(lines[i]).map(c=>`<td>${inline(c)}</td>`).join('')+'</tr>'; i++; }
        out+=`<div class="table-wrap"><table><thead><tr>${heads.map(c=>`<th>${inline(c)}</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table></div>`; continue;
      }
      if (/^!\[[^\]]*\]\([^)]+\)$/.test(line.trim())) { out+=inline(line.trim()); i++; continue; }
      const h=line.match(/^(#{1,4})\s+(.+)$/); if(h){const level=h[1].length,title=h[2].replace(/\s+#+\s*$/,'');const id=slug(title);out+=`<h${level} id="${id}">${inline(title)}</h${level}>`;i++;continue;}
      if (/^---+$/.test(line.trim())) {out+='<hr>';i++;continue;}
      if (/^>\s?/.test(line)) {let q=[];while(i<lines.length&&/^>\s?/.test(lines[i]))q.push(lines[i++].replace(/^>\s?/,''));out+=`<blockquote>${q.map(x=>`<p>${inline(x)}</p>`).join('')}</blockquote>`;continue;}
      if (/^\s*[-*+]\s+/.test(line) || /^\s*\d+\.\s+/.test(line)) {const ordered=/^\s*\d+\./.test(line),tag=ordered?'ol':'ul';let items='';while(i<lines.length&&(/^\s*[-*+]\s+/.test(lines[i])||/^\s*\d+\.\s+/.test(lines[i]))){items+=`<li>${inline(lines[i].replace(/^\s*(?:[-*+]\s+|\d+\.\s+)/,''))}</li>`;i++;}out+=`<${tag}>${items}</${tag}>`;continue;}
      let p=[line.trim()];i++;while(i<lines.length&&lines[i].trim()&&!/^(#{1,4}\s|```|>|\s*[-*+]\s+|\s*\d+\.\s+|---+$)/.test(lines[i])&&!/^\|.*\|$/.test(lines[i].trim())&&!/^!\[[^\]]*\]\([^)]+\)$/.test(lines[i].trim()))p.push(lines[i++].trim());out+=`<p>${inline(p.join(' '))}</p>`;
    }
    return out;
  }
  function setPaper(index, hash) {
    current=(index+papers.length)%papers.length; const p=papers[current]; visualIndex=0;
    home.hidden=true; readerView.hidden=false; main.classList.remove('is-home'); rail.hidden=false; $('.search-box').hidden=false;
    [...tabs.children].forEach(b=>b.setAttribute('aria-current',Number(b.dataset.paperIndex)===current?'true':'false'));
    $('#cover-number').textContent=`0${current+1} / 0${papers.length}`; $('#cover-title').textContent=p.title; $('#cover-lead').textContent=p.lead;
    $('#cover-meta').innerHTML=p.tags.map(t=>`<span>${esc(t)}</span>`).join(''); $('#rail-title').textContent=p.shortTitle;
    article.innerHTML=renderMarkdown(p.markdown);
    headings=[...article.querySelectorAll('h2,h3')].filter(h=>h.textContent.trim());
    toc.innerHTML=headings.map((h,i)=>`<a href="#${h.id}" data-heading="${h.id}" class="${h.tagName==='H3'?'sub':''}"><span>${String(i+1).padStart(2,'0')}</span>${esc(h.textContent)}</a>`).join('');
    $('#read-count').textContent=`0 / ${headings.length}`;
    const guideMap=$('#guide-map'); guideMap.className='guide-links';
    guideMap.innerHTML=headings.slice(0,8).map((h,i)=>`<a href="#${h.id}"><b>${String(i+1).padStart(2,'0')}</b> ${esc(h.textContent)}</a>`).join('');
    drawVisual(); setupObserver(); $('#search').value=''; clearSearch();
    if(hash&&location.hash) requestAnimationFrame(()=>document.querySelector(location.hash)?.scrollIntoView()); else { if(location.hash) history.replaceState(null,'',location.pathname+location.search); window.scrollTo({top:0,behavior:'smooth'}); }
  }
  function showHome(updateHash=true){
    home.hidden=false; readerView.hidden=true; main.classList.add('is-home'); rail.hidden=true; $('.search-box').hidden=true;
    [...tabs.children].forEach(b=>b.setAttribute('aria-current',b.dataset.home?'true':'false'));
    $('#search').value=''; clearSearch();
    if(updateHash) history.replaceState(null,'',`${location.pathname}${location.search}#home-guide`);
    window.scrollTo({top:0,behavior:'smooth'});
  }
  function drawVisual(){const p=papers[current],state=p.visuals[visualIndex];$('#visual-step').textContent=`0${visualIndex+1} — 0${p.visuals.length}`;$('#concept-map').innerHTML=state.rows.map(row=>`<div class="map-row">${row.map((n,i)=>`${i?'<span class="map-arrow" aria-hidden="true">→</span>':''}<span class="map-node" data-tone="${esc(n.tone||'')}">${esc(n.label)}</span>`).join('')}</div>`).join('')+`<p class="map-caption">${esc(state.caption)}</p>`;$('#visual-note').textContent=state.note;$('#visual-dots').innerHTML=p.visuals.map((_,i)=>`<button class="visual-dot ${i===visualIndex?'active':''}" aria-label="查看阶段 ${i+1}" data-visual="${i}"></button>`).join('');$('#visual-dots').querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{visualIndex=+b.dataset.visual;drawVisual()}));}
  function setupObserver(){activeObserver?.disconnect();activeObserver=new IntersectionObserver(entries=>{const visible=entries.filter(e=>e.isIntersecting).sort((a,b)=>a.boundingClientRect.top-b.boundingClientRect.top);if(visible.length){const id=visible[0].target.id;toc.querySelectorAll('a').forEach(a=>a.classList.toggle('active',a.dataset.heading===id));const n=headings.findIndex(h=>h.id===id);$('#read-count').textContent=`${Math.max(0,n+1)} / ${headings.length}`;}},{rootMargin:'-18% 0px -72% 0px'});headings.forEach(h=>activeObserver.observe(h));}
  function clearSearch(){article.querySelectorAll('mark').forEach(m=>m.replaceWith(document.createTextNode(m.textContent)));article.querySelectorAll('.search-dim,.search-hidden').forEach(e=>e.classList.remove('search-dim','search-hidden'));$('#search-status').classList.remove('visible');}
  function search(q){clearSearch();q=q.trim();if(!q)return;const lower=q.toLowerCase();let count=0;const blocks=[...article.querySelectorAll('p,li,h2,h3,h4,td,blockquote')];blocks.forEach(el=>{if(el.textContent.toLowerCase().includes(lower))count++;});
    const children=[...article.children];let group=[];const groups=[];for(const el of children){if(el.tagName==='H2'){if(group.length)groups.push(group);group=[el];}else if(group.length)group.push(el);}if(group.length)groups.push(group);
    for(const items of groups){const hit=items.some(el=>el.textContent.toLowerCase().includes(lower));if(!hit)items.forEach(el=>el.classList.add('search-hidden'));}
    const status=$('#search-status');status.textContent=count?`找到 ${count} 处 · 已收起无匹配章节`:'没有匹配内容';status.classList.add('visible');if(count){const first=blocks.find(el=>el.textContent.toLowerCase().includes(lower));first?.scrollIntoView({behavior:'smooth',block:'center'});}}
  const homeTab=document.createElement('button');homeTab.className='paper-tab';homeTab.textContent='阅读导览';homeTab.dataset.home='true';homeTab.setAttribute('aria-current','true');homeTab.addEventListener('click',()=>showHome());tabs.append(homeTab);
  papers.forEach((p,i)=>{const b=document.createElement('button');b.className='paper-tab';b.textContent=p.shortTitle;b.dataset.paperIndex=String(i);b.addEventListener('click',()=>setPaper(i));tabs.append(b);});
  home.querySelectorAll('[data-paper]').forEach(button=>button.addEventListener('click',()=>setPaper(Number(button.dataset.paper))));
  $('#brand-home').addEventListener('click',event=>{event.preventDefault();showHome();});
  $('#visual-prev').addEventListener('click',()=>{visualIndex=(visualIndex+papers[current].visuals.length-1)%papers[current].visuals.length;drawVisual()});$('#visual-next').addEventListener('click',()=>{visualIndex=(visualIndex+1)%papers[current].visuals.length;drawVisual()});
  $('#search').addEventListener('input',e=>search(e.target.value));document.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='k'){e.preventDefault();$('#search').focus();}if(e.key==='Escape'){$('#search').value='';clearSearch();$('#search').blur();}});
  const progress=$('#reading-progress'),back=$('#back-top');window.addEventListener('scroll',()=>{const max=document.documentElement.scrollHeight-innerHeight;progress.style.width=`${max>0?scrollY/max*100:0}%`;back.classList.toggle('visible',scrollY>500);},{passive:true});back.addEventListener('click',()=>window.scrollTo({top:0,behavior:'smooth'}));
  if(location.hash&&location.hash!=='#home-guide'&&location.hash!=='#top') setPaper(0,true); else showHome(false);
})();
