(function(){'use strict';
const source=window.PTO_PAGED_ATTENTION_PYPTO_SOURCE;
const editor=document.getElementById('dslEditor'), bar=document.getElementById('sourceRegionBar'), scroller=document.querySelector('.kf-main-body');
  const pagedAttentionPyptoFocusMeta = {
    constants: { label: '模块常量与静态断言', lines: '9–65', detail: 'BATCH / HEAD_DIM / BLOCK_SIZE / STACK_* 等编译期常量与 8 条 assert 自检' },
    signature: { label: '函数签名与张量契约', lines: '67–108', detail: '27 个 Tensor / TaskID 入参，导出 out，并把分页缓存 reshape 成 BSND 视图' },
    spmd: { label: 'SPMD 入口与生产者依赖', lines: '109–126', detail: '24 个 SPMD block，显式声明 7 个 Phase-0 生产者的 TaskId 依赖' },
    kvphase: { label: 'Phase 0 · K/V 归一化与 Cache 写回', lines: '127–213', detail: '前 32 个 AIV lane：K 做 RMSNorm + RoPE 写 key cache，V 乘 inv_rms 写 value cache' },
    qphase: { label: 'Phase 0 · Q 归一化与 RoPE', lines: '215–251', detail: '每组 GROUP 个 query head 做 RMSNorm + RoPE，按 BSND 行写回 q2d' },
    barrier: { label: 'Phase-0 发布与全核 Barrier', lines: '253–259', detail: 'cacheinvalid → fence → syncall(MIX) → cacheinvalid，保证 PA 读到已发布的 Q/K/V' },
    taskloop: { label: 'Attention 任务循环与 Q 装载', lines: '261–277', detail: '每个 (batch, kv_head) 一个 task，推导 page / stack 数并把 Q tile 装进 L1' },
    qk: { label: 'QK 生产 · Cube', lines: '278–362', detail: '按 STACK_PAGES 逐页 load K、matmul 出 score 写入 transfer，含 1/2/3 页末栈 tail 分支' },
    pv: { label: 'PV 生产 · Cube', lines: '364–724', detail: '一次预取整栈 V 并 ping-pong P，按页数分 1/2/3/4 页 tail 分支做 matmul_acc' },
    softmax: { label: '在线 Softmax 与 FIFO 更新 · Vector', lines: '726–862', detail: '双 AIV lane 上 scale / mask / row_max / exp / row_sum，用 pending FIFO 做三态 rescale' },
    out: { label: '输出归一化与写回', lines: '864–875', detail: 'o / l_sum 归一化并转 BF16 写回 out2d，返回本 kernel 的 attn_tid' },
  };

  const PY_KEYWORDS = new Set([
    'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def',
    'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if',
    'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise',
    'return', 'try', 'while', 'with', 'yield', 'match', 'case'
  ]);

  // 内置函数 / 类型 / 异常。单独给一个颜色，和普通调用区分开。
  const PY_BUILTINS = new Set([
    'abs', 'aiter', 'all', 'anext', 'any', 'ascii', 'bin', 'bool', 'breakpoint',
    'bytearray', 'bytes', 'callable', 'chr', 'classmethod', 'compile', 'complex',
    'delattr', 'dict', 'dir', 'divmod', 'enumerate', 'eval', 'exec', 'filter',
    'float', 'format', 'frozenset', 'getattr', 'globals', 'hasattr', 'hash', 'help',
    'hex', 'id', 'input', 'int', 'isinstance', 'issubclass', 'iter', 'len', 'list',
    'locals', 'map', 'max', 'memoryview', 'min', 'next', 'object', 'oct', 'open',
    'ord', 'pow', 'print', 'property', 'range', 'repr', 'reversed', 'round', 'set',
    'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum', 'super', 'tuple',
    'type', 'vars', 'zip',
    'ArithmeticError', 'AssertionError', 'AttributeError', 'BaseException',
    'Exception', 'ImportError', 'IndexError', 'KeyError', 'KeyboardInterrupt',
    'MemoryError', 'NameError', 'NotImplementedError', 'OSError', 'OverflowError',
    'RuntimeError', 'StopIteration', 'TypeError', 'ValueError', 'ZeroDivisionError',
  ]);

  function highlightPythonLines(source) {
    const esc = (s) => s.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
    const span = (cls, s) => `<span class="${cls}">${esc(s)}</span>`;
    const lines = source.split('\n');
    const out = [];
    let triple = null; // { quote, cls } — 跨行中的三引号字符串

    for (let li = 0; li < lines.length; li++) {
      const line = lines[li];
      let html = '';
      let i = 0;

      // 同类相邻 token 合并成一个 span。decode_layer.py 有 1768 行，逐字符各出一个
      // span 会把 DOM 体积翻好几倍；plain 与 token 两个缓冲互斥，谁在用另一个必为空。
      let plain = '';
      let tokCls = null;
      let tokBuf = '';
      const flushPlain = () => { if (plain) { html += esc(plain); plain = ''; } };
      const flushTok = () => { if (tokBuf) { html += span(tokCls, tokBuf); tokCls = null; tokBuf = ''; } };
      const pushTok = (cls, s) => {
        if (tokCls !== cls) { flushPlain(); flushTok(); tokCls = cls; }
        tokBuf += s;
      };
      const pushPlain = (s) => { flushTok(); plain += s; };

      if (triple) {
        const close = line.indexOf(triple.quote);
        if (close === -1) { out.push(span(triple.cls, line)); continue; }
        html += span(triple.cls, line.slice(0, close + 3));
        i = close + 3;
        triple = null;
      }

      let expectName = null; // 'func' | 'class' — next identifier is a definition name
      let prev = '';         // 上一个有意义的字符，用于判断 x.y 是不是属性访问

      while (i < line.length) {
        const rest = line.slice(i);
        const ch = line[i];

        // comment
        if (ch === '#') { pushTok('tok-com', line.slice(i)); i = line.length; break; }

        // 字符串（可带 r/b/u/f 前缀）。三引号且顶在视觉行首 → 当 docstring 单独上色。
        const sm = /^([rRbBuUfF]{0,2})("""|'''|"|')/.exec(rest);
        if (sm) {
          const prefixLen = sm[1].length;
          const q = sm[2];
          const startQuote = i + prefixLen;
          const tokenStart = i;
          const cls = q.length === 3 && line.slice(0, tokenStart).trim() === '' ? 'tok-doc' : 'tok-str';
          let text;
          if (q.length === 3) {
            const after = line.indexOf(q, startQuote + 3);
            if (after === -1) {
              pushTok(cls, line.slice(tokenStart));
              triple = { quote: q, cls };
              i = line.length;
              break;
            }
            text = line.slice(tokenStart, after + 3);
            i = after + 3;
          } else {
            let j = startQuote + 1;
            while (j < line.length) {
              if (line[j] === '\\') { j += 2; continue; }
              if (line[j] === q) { j++; break; }
              j++;
            }
            text = line.slice(tokenStart, j);
            i = j;
          }
          // f-string：把 {...} 里的内容单独上色（只做一层，不解析格式串与嵌套花括号）
          const bodyFrom = prefixLen + q.length;
          if (/[fF]/.test(sm[1]) && text.length >= bodyFrom + q.length && text.endsWith(q)) {
            pushTok(cls, text.slice(0, bodyFrom));
            const body = text.slice(bodyFrom, text.length - q.length);
            const re = /\{[^{}]*\}/g;
            let at = 0;
            let brace;
            while ((brace = re.exec(body))) {
              if (brace.index > at) pushTok(cls, body.slice(at, brace.index));
              pushTok('tok-interp', brace[0]);
              at = brace.index + brace[0].length;
            }
            if (at < body.length) pushTok(cls, body.slice(at));
            pushTok(cls, text.slice(text.length - q.length));
          } else {
            pushTok(cls, text);
          }
          prev = text.slice(-1);
          continue;
        }

        // number
        const nm = /^(0[xXoObB][0-9a-fA-F_]+|(?:\d[\d_]*\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[jJ]?)/.exec(rest);
        if (nm && /\d/.test(nm[0])) { pushTok('tok-num', nm[0]); prev = nm[0].slice(-1); i += nm[0].length; continue; }

        // decorator (only at the visual start of a line)
        if (ch === '@') {
          const dm = /^@[A-Za-z_][\w.]*/.exec(rest);
          if (dm && line.slice(0, i).trim() === '') { pushTok('tok-dec', dm[0]); prev = dm[0].slice(-1); i += dm[0].length; continue; }
        }

        // identifier / keyword / 属性
        const im = /^[A-Za-z_]\w*/.exec(rest);
        if (im) {
          const word = im[0];
          const isCall = /^\s*\(/.test(line.slice(i + word.length));
          const isAttr = prev === '.';
          let cls = null;
          if (expectName) { cls = expectName === 'class' ? 'tok-cls' : 'tok-fn'; expectName = null; }
          else if (word === 'def') { cls = 'tok-kw'; expectName = 'func'; }
          else if (word === 'class') { cls = 'tok-kw'; expectName = 'class'; }
          else if (isAttr) cls = isCall ? 'tok-fn' : 'tok-prop';
          else if (PY_KEYWORDS.has(word)) cls = 'tok-kw';
          else if (word === 'True' || word === 'False' || word === 'None') cls = 'tok-const';
          else if (word === 'self' || word === 'cls') cls = 'tok-self';
          else if (PY_BUILTINS.has(word)) cls = 'tok-builtin';
          else if (isCall) cls = 'tok-fn';
          if (cls) pushTok(cls, word); else pushPlain(word);
          prev = word.slice(-1);
          i += word.length;
          continue;
        }

        // 运算符（相邻同类会合并成一个 span，所以不用穷举 **= / >>= 这类组合）
        if (/^(?::=|[-+*/%@&|^~<>=!])/.test(rest)) { pushTok('tok-op', ch); prev = ch; i++; continue; }
        // 括号与分隔符（. 不在此列，属性链保持干净）
        if (/[()[\]{},:;]/.test(ch)) { pushTok('tok-punc', ch); prev = ch; i++; continue; }

        pushPlain(ch);
        prev = ch;
        i++;
      }
      flushTok();
      flushPlain();
      out.push(html);
    }
    return out;
  }


const entries=Object.entries(pagedAttentionPyptoFocusMeta), lines=source.split('\n'), colored=highlightPythonLines(source);
const escape=s=>s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const focusForLine=n=>entries.find(([,v])=>{const [a,b]=v.lines.split('–').map(Number);return n>=a&&n<=b;})?.[0]||null;
editor.innerHTML=lines.map((line,i)=>'<div data-paged-attention-line="'+(i+1)+'" data-paged-attention-focus="'+(focusForLine(i+1)||'')+'"><i>'+(i+1)+'</i><code style="--kf-code-indent:'+(line.match(/^\s*/)||[''])[0].length+'ch">'+(colored[i]||' ')+'</code></div>').join('');
let selected=null, expanded=false, programmatic=false;
function current(){const rows=editor.children,top=scroller.getBoundingClientRect().top+8;let lo=0,hi=rows.length-1,at=0;while(lo<=hi){const mid=(lo+hi)>>1;if(rows[mid].getBoundingClientRect().bottom>top){at=mid;hi=mid-1;}else lo=mid+1;}return rows[at]?.dataset.pagedAttentionFocus||null;}
function render(){const active=selected||current(),index=entries.findIndex(([k])=>k===active),item=index>=0?entries[index][1]:{label:'文件头',lines:'1–8',detail:'许可证声明与模块说明'};
bar.hidden=false;
bar.innerHTML='<div class="kf-source-region__bar"><button type="button" class="kf-source-region__head" data-source-region-toggle aria-expanded="'+expanded+'"><span class="kf-source-region__pos">'+(index>=0?index+1:'—')+'/'+entries.length+'</span><b>'+item.label+'</b><i>'+item.lines+'</i><small>'+item.detail+'</small><em>'+(expanded?'收起':'全部')+'</em></button><button type="button" class="btn btn-sm" data-visualization2-open aria-controls="visualization2">Compute Explorer</button></div><div class="kf-source-region__list" '+(expanded?'':'hidden')+'>'+entries.map(([key,value],n)=>'<button type="button" class="'+(key===active?'is-active':'')+'" data-source-region="'+key+'"><span>'+String(n+1).padStart(2,'0')+'</span><b>'+value.label+'</b><i>'+value.lines+'</i><small>'+value.detail+'</small></button>').join('')+'</div>';}
bar.addEventListener('click',e=>{if(e.target.closest('[data-source-region-toggle]')){expanded=!expanded;render();return;}const button=e.target.closest('[data-source-region]');if(!button)return;selected=button.dataset.sourceRegion;expanded=false;programmatic=true;render();const [start]=pagedAttentionPyptoFocusMeta[selected].lines.split('–').map(Number);const target=editor.querySelector('[data-paged-attention-line="'+start+'"]');if(target){scroller.style.scrollBehavior='auto';scroller.scrollTop+=target.getBoundingClientRect().top-scroller.getBoundingClientRect().top-bar.offsetHeight;requestAnimationFrame(()=>{scroller.style.scrollBehavior='';});}editor.querySelectorAll('[data-paged-attention-focus]').forEach(el=>el.classList.toggle('is-paged-attention-line-active',el.dataset.pagedAttentionFocus===selected));render();setTimeout(()=>{programmatic=false;},300);});
scroller.addEventListener('scroll',()=>{if(programmatic)return;selected=null;render();},{passive:true});
editor.addEventListener('click',e=>{const row=e.target.closest('[data-paged-attention-line]');if(!row)return;selected=row.dataset.pagedAttentionFocus;editor.querySelectorAll('[data-paged-attention-focus]').forEach(el=>el.classList.toggle('is-paged-attention-line-active',el.dataset.pagedAttentionFocus===selected));render();});
document.getElementById('projectTree').addEventListener('click',e=>{const folder=e.target.closest('[data-tree-toggle]');if(folder){const group=document.querySelector('[data-tree-group="'+folder.dataset.treeToggle+'"]');if(group){group.hidden=!group.hidden;folder.setAttribute('aria-expanded',String(!group.hidden));}return;}if(e.target.closest('[data-file="paged_attention_pypto.py"]')){document.querySelector('.kf-main-body').scrollTop=0;editor.focus({preventScroll:true});}});
render();
})();
