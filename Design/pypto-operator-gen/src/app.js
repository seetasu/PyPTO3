/** PyPTO 算子生成 Demo · 交互 */

(() => {
  const D = window.PyptoGenData;
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const setText = (selector, value, root = document) => {
    const node = $(selector, root);
    if (node) node.textContent = value ?? '';
  };

  const KB = (bytes) => `${Math.round(bytes / 1024)} KB`;

  /* ── 代码高亮（Python / PTO MLIR / log） ─────────────────────────── */
  const PY_KEYWORDS = new Set([
    'def', 'return', 'import', 'from', 'with', 'for', 'in', 'range', 'if', 'else', 'elif',
    'class', 'as', 'None', 'True', 'False', 'and', 'or', 'not', 'assert', 'lambda', 'print'
  ]);
  const PY_TYPES = new Set([
    'pl', 'torch', 'RunConfig', 'Tensor', 'Out', 'BF16', 'FP32', 'Level', 'CORE_GROUP',
    'MemorySpace', 'Mat', 'Left', 'Right', 'Acc', 'Vec', 'DDR'
  ]);

  const escapeHtml = (value) => String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  const TOKEN = /("""[\s\S]*?"""|#[^\n]*|\/\/[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b\d+(?:\.\d+)?\b|\b[A-Za-z_][\w]*\b)/g;

  const highlight = (line) => {
    let html = '';
    let cursor = 0;
    for (const match of String(line).matchAll(TOKEN)) {
      const token = match[0];
      const index = match.index ?? 0;
      html += escapeHtml(line.slice(cursor, index));
      cursor = index + token.length;
      let cls = '';
      if (token.startsWith('#') || token.startsWith('//') || token.startsWith('"""')) cls = 'syntax-generated';
      else if (token.startsWith('"') || token.startsWith("'")) cls = 'syntax-user';
      else if (/^\d/.test(token)) cls = 'syntax-number';
      else if (PY_KEYWORDS.has(token)) cls = 'syntax-keyword';
      else if (PY_TYPES.has(token) || /^pto$/.test(token)) cls = 'syntax-type';
      else if (line[cursor] === '(') cls = 'syntax-call';
      html += cls ? `<span class="${cls}">${escapeHtml(token)}</span>` : escapeHtml(token);
    }
    html += escapeHtml(line.slice(cursor));
    return html || '&nbsp;';
  };

  /** 逐行高亮，同时跟踪跨行的 """docstring""" 状态。 */
  const highlightLines = (code) => {
    let inDoc = false;
    return String(code).replace(/\n$/, '').split('\n').map((line) => {
      const fences = (line.match(/"""/g) || []).length;
      if (inDoc) {
        const html = `<span class="syntax-generated">${escapeHtml(line) || '&nbsp;'}</span>`;
        if (fences % 2 === 1) inDoc = false;
        return html;
      }
      if (fences % 2 === 1) {
        inDoc = true;
        return `<span class="syntax-generated">${escapeHtml(line)}</span>`;
      }
      return highlight(line);
    });
  };

  const renderCodeSurface = (root, code) => {
    root.innerHTML = highlightLines(code).map((html, index) => (
      `<div class="pto-ide-frame__code-line">` +
      `<span class="pto-ide-frame__code-line-number">${index + 1}</span>` +
      `<span class="pto-ide-frame__code-line-code">${html}</span>` +
      `</div>`
    )).join('');
  };

  /* ── 左栏：API 说明 ─────────────────────────────────────────────── */
  const bindApiPills = () => {
    $$('[data-api]').forEach((pill) => {
      pill.addEventListener('click', () => {
        const note = D.apiNotes[pill.dataset.api];
        $$('[data-api]').forEach((other) => {
          const active = other === pill;
          other.classList.toggle('is-selected', active);
          other.setAttribute('aria-pressed', String(active));
        });
        setText('[data-api-title]', note.title);
        setText('[data-api-copy]', note.copy);
      });
    });
  };

  /* ── 候选对比表 ─────────────────────────────────────────────────── */
  const renderCandidateTable = () => {
    const body = $('[data-candidate-rows]');
    body.replaceChildren(...D.candidateRows.map((row) => {
      const tr = document.createElement('tr');
      const head = document.createElement('td');
      head.textContent = row.label;
      tr.append(head);
      row.values.forEach((value, index) => {
        const td = document.createElement('td');
        td.textContent = value;
        const classes = [];
        if (index === 1) classes.push('is-recommended');
        const tone = row.tones?.[index];
        if (tone === 'positive') classes.push('cell-positive');
        if (tone === 'warning') classes.push('cell-warning');
        if (classes.length) td.className = classes.join(' ');
        tr.append(td);
      });
      return tr;
    }));
  };

  /* ── 流水示意 ───────────────────────────────────────────────────── */
  const showPipeline = (key) => {
    const panel = $('[data-pipeline-panel]');
    const plan = D.pipelines[key];
    const candidate = D.candidates[key];
    panel.hidden = false;
    setText('[data-pipeline-title]', `${candidate.name} · 流水示意`);
    setText('[data-pipeline-meta]', plan.meta);
    setText('[data-pipeline-note]', plan.note);

    const strip = $('[data-pipeline-strip]');
    strip.replaceChildren(...plan.lanes.map((lane) => {
      const row = document.createElement('div');
      row.className = 'pipeline-lane';
      const label = document.createElement('strong');
      label.textContent = lane.label;
      const track = document.createElement('div');
      track.className = 'pipeline-track';
      lane.blocks.forEach((block) => {
        const [kind, text] = block.split(':');
        const cell = document.createElement('span');
        cell.className = `pipeline-block is-${kind}`;
        cell.textContent = text || '';
        track.append(cell);
      });
      row.append(label, track);
      return row;
    }));
    panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  };

  /* ── 决策点矩阵 ─────────────────────────────────────────────────── */
  let matrixRendered = false;

  /** 一个取值用了多少预算，以及它越没越界。 */
  const usage = (row, option) => {
    if (row.capacity === 0) return { used: option.count ?? 0, cap: 0, ratio: option.defect ? 1 : 0.2, over: Boolean(option.defect) };
    const used = row.countMode ? option.count : option.bytes;
    return { used, cap: row.capacity, ratio: used / row.capacity, over: used > row.capacity };
  };

  const amount = (row, option) => (row.countMode ? `${option.count}` : KB(option.bytes));

  const renderDecisionMatrix = () => {
    if (matrixRendered) return;
    matrixRendered = true;
    const matrix = $('[data-atlas-matrix]');
    const readout = $('[data-atlas-readout]');
    const maxOptions = Math.max(...D.decisionMatrix.map((row) => row.options.length));
    matrix.style.setProperty('--atlas-columns', String(maxOptions));

    const select = (row, option, button) => {
      $$('.atlas-cell', matrix).forEach((other) => other.classList.toggle('is-selected', other === button));
      setText('[data-atlas-summary]', `${row.label} · ${option.label}${option.chosen ? '（源码取值）' : ''}`);
      readout.hidden = false;
      readout.replaceChildren(...[
        ['取值', option.label],
        ['占用', row.capacity === 0 ? '不适用' : `${amount(row, option)} / ${row.countMode ? row.capacity : KB(row.capacity)}（${row.budget}）`],
        ['结论', option.verdict],
        ['是否被采用', option.chosen ? '是，这就是源码里的数字' : '否'],
        ['出处', row.source]
      ].map(([label, value]) => {
        const line = document.createElement('div');
        line.className = 'trace-detail-row';
        const dt = document.createElement('dt');
        const dd = document.createElement('dd');
        dt.textContent = label;
        dd.textContent = value;
        line.append(dt, dd);
        return line;
      }));
    };

    matrix.replaceChildren(...D.decisionMatrix.map((row) => {
      const line = document.createElement('div');
      line.className = 'atlas-row';

      const label = document.createElement('div');
      label.className = 'atlas-row-label';
      const strong = document.createElement('strong');
      const hint = document.createElement('span');
      const budget = document.createElement('span');
      strong.textContent = row.label;
      hint.textContent = row.hint;
      budget.textContent = row.budget;
      label.append(strong, hint, budget);
      line.append(label);

      row.options.forEach((option) => {
        const u = usage(row, option);
        const button = document.createElement('button');
        button.type = 'button';
        button.className = `atlas-cell${u.over ? ' is-over' : ''}`;

        const title = document.createElement('strong');
        title.textContent = `${option.chosen ? '★ ' : ''}${option.label}`;
        const meta = document.createElement('span');
        meta.textContent = row.capacity === 0
          ? (option.defect ? '已知缺陷' : '安全')
          : `${amount(row, option)}${u.over ? ' · 越界' : ''}`;

        const bars = document.createElement('div');
        bars.className = 'atlas-bars';
        const bar = document.createElement('div');
        bar.className = `atlas-bar${u.over ? ' is-over' : ''}`;
        const fill = document.createElement('i');
        fill.style.width = `${Math.min(100, u.ratio * 100)}%`;
        bar.append(fill);
        bars.append(bar);

        button.append(title, meta, bars);
        button.addEventListener('click', () => select(row, option, button));
        line.append(button);
      });

      // 补齐空格子，保持列对齐
      for (let i = row.options.length; i < maxOptions; i += 1) {
        line.append(document.createElement('span'));
      }
      return line;
    }));
  };

  /* ── 拒绝浏览器 ─────────────────────────────────────────────────── */
  const rejectionState = {};

  const renderRejectionBrowser = (key) => {
    const config = D.rejectionBrowsers[key];
    const state = rejectionState[key] ?? (rejectionState[key] = {
      category: config.defaultCategory,
      case: config.cases[config.defaultCategory][0].id
    });

    setText('[data-rejection-browser-title]', config.title);
    setText('[data-rejection-browser-copy]', config.copy);
    setText('[data-rejection-browser-status]', config.status);

    $('[data-rejection-categories]').replaceChildren(...config.categories.map((category) => {
      const button = document.createElement('button');
      const copy = document.createElement('span');
      const title = document.createElement('strong');
      const hint = document.createElement('span');
      const count = document.createElement('b');
      button.type = 'button';
      button.className = `btn btn-ghost rejection-category${category.id === state.category ? ' is-selected' : ''}`;
      button.setAttribute('aria-pressed', String(category.id === state.category));
      title.textContent = category.label;
      hint.textContent = category.hint;
      count.textContent = String(category.count);
      copy.className = 'rejection-category-copy';
      copy.append(title, hint);
      button.append(copy, count);
      button.addEventListener('click', () => {
        state.category = category.id;
        state.case = config.cases[category.id][0].id;
        renderRejectionBrowser(key);
      });
      return button;
    }));

    const cases = config.cases[state.category];
    setText('[data-rejection-case-list-title]', '被拒候选');
    setText('[data-rejection-case-count]', `${cases.length} 个展开案例 · 共 ${config.categories.find((c) => c.id === state.category).count} 个`);

    $('[data-rejection-cases]').replaceChildren(...cases.map((item) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = `rejection-case${item.id === state.case ? ' is-selected' : ''}`;
      const id = document.createElement('span');
      id.className = 'rejection-case-id';
      id.textContent = item.id;
      const copy = document.createElement('span');
      copy.className = 'rejection-case-copy';
      const strong = document.createElement('strong');
      const hint = document.createElement('span');
      strong.textContent = item.title;
      hint.textContent = item.rule;
      copy.append(strong, hint);
      button.append(id, copy);
      button.addEventListener('click', () => {
        state.case = item.id;
        renderRejectionBrowser(key);
      });
      return button;
    }));

    const selected = cases.find((item) => item.id === state.case) ?? cases[0];
    const detail = $('[data-rejection-case-detail]');
    detail.hidden = false;
    setText('[data-rejection-case-rule]', `${selected.id} · ${selected.rule}`);
    setText('[data-rejection-case-copy]', selected.copy);
    // .reason-list li 是 18px 序号 + 正文的两列网格，两个 span 都要给。
    $('[data-rejection-case-reasons]').replaceChildren(...selected.reasons.map((reason, index) => {
      const li = document.createElement('li');
      const marker = document.createElement('span');
      const copy = document.createElement('span');
      marker.className = 'reason-index';
      marker.textContent = String(index + 1);
      copy.textContent = reason;
      li.append(marker, copy);
      return li;
    }));
    setText('[data-status-stage]', `Generation trace · ${selected.id} · ${config.status}`);
  };

  /* ── 阶段 / 转换详情 ────────────────────────────────────────────── */
  const fillList = (root, rows) => {
    root.replaceChildren(...(rows ?? []).map(([label, value]) => {
      const row = document.createElement('div');
      row.className = 'trace-detail-row';
      const dt = document.createElement('dt');
      const dd = document.createElement('dd');
      dt.textContent = label;
      dd.textContent = value;
      row.append(dt, dd);
      return row;
    }));
  };

  const showCandidates = () => {
    $('[data-view-panel="candidates"]').hidden = false;
    $('[data-view-panel="trace-detail"]').hidden = true;
    setText('[data-center-meta]', 'Candidates · TP 世界');
    setText('[data-status-stage]', 'Reverse trace · decode_csa.py · _TP_CHOICES = (1, 2, 4)');
  };

  const showTraceDetail = (view) => {
    $('[data-view-panel="candidates"]').hidden = true;
    $('[data-view-panel="trace-detail"]').hidden = false;
    setText('#traceTitle', view.title);
    setText('[data-center-meta]', view.paneMeta);
    setText('[data-status-stage]', view.status);

    const hasBrowser = Boolean(view.browser);
    const hasMatrix = Boolean(view.matrix);
    $('[data-rejection-browser]').hidden = !hasBrowser;
    $('[data-generic-trace-detail]').hidden = hasBrowser;
    $('[data-tile-atlas]').hidden = !hasMatrix;
    $('[data-generic-trace-footer]').hidden = hasBrowser;

    if (hasBrowser) {
      renderRejectionBrowser(view.browser);
      return;
    }

    setText('[data-detail-list-title]', view.listTitle);
    setText('[data-detail-output-title]', view.outputTitle);
    setText('[data-detail-output-strong]', view.outputStrong);
    setText('[data-detail-output-copy]', view.outputCopy);
    setText('[data-detail-footer-strong]', view.footerStrong);
    setText('[data-detail-footer-copy]', view.footerCopy);
    fillList($('[data-detail-list]'), view.list);
    fillList($('[data-detail-output-list]'), view.outputList);
    if (hasMatrix) renderDecisionMatrix();
  };

  const applySelection = (selection) => {
    if (!selection) {
      showCandidates();
      return;
    }
    const view = D.traceViews[selection.id];
    if (!view) {
      showCandidates();
      return;
    }
    if (view.candidates) {
      showCandidates();
      return;
    }
    showTraceDetail(view);
  };

  /* ── 代码预览弹窗 ───────────────────────────────────────────────── */
  let previewCandidate = 'tp2';
  let previewFile = 'orchestration';

  const renderCodePreview = () => {
    const candidate = D.candidates[previewCandidate];
    const files = D.projectFiles(previewCandidate);
    const file = files.find((item) => item.id === previewFile) ?? files[0];

    setText('[data-code-preview-meta]', `${candidate.name} · 入口 ${candidate.entry}`);
    setText('[data-code-preview-summary]', candidate.note);
    setText('[data-code-preview-filename]', file.path);

    const nav = $('[data-code-preview-files]');
    nav.replaceChildren(nav.firstElementChild);
    files.forEach((item) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = `candidate-code-preview__file${item.id === file.id ? ' is-active' : ''}`;
      button.textContent = item.path.split('/').pop();
      button.addEventListener('click', () => {
        previewFile = item.id;
        renderCodePreview();
      });
      nav.append(button);
    });

    const code = $('[data-code-preview-code]');
    code.innerHTML = highlightLines(file.code).map((html) => `<div>${html}</div>`).join('');
  };

  const openCodePreview = (key) => {
    previewCandidate = key;
    previewFile = 'orchestration';
    renderCodePreview();
    $('[data-code-preview-modal]').hidden = false;
  };

  const closeCodePreview = () => { $('[data-code-preview-modal]').hidden = true; };

  /* ── 生成算子工程 ───────────────────────────────────────────────── */
  const FOLDER_ICON = '<svg class="generated-project-tree__icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9l-.81-1.2A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"></path></svg>';
  const FILE_ICON = '<svg class="generated-project-tree__icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"></path><path d="M14 2v6h6"></path></svg>';
  const OPEN_CHEVRON = '<svg class="generated-project-tree__chevron" viewBox="0 0 24 24" aria-hidden="true"><path d="m6 9 6 6 6-6"></path></svg>';

  let projectCandidate = null;
  let projectFile = 'orchestration';

  const renderProjectCode = () => {
    const files = D.projectFiles(projectCandidate);
    const file = files.find((item) => item.id === projectFile) ?? files[0];

    const crumbs = $('[data-project-breadcrumbs]');
    const trail = document.createElement('span');
    trail.innerHTML =
      `<strong>${file.path.split('/').slice(0, -1).join(' / ') || 'fused_linear_gelu'}</strong> / ${file.path.split('/').pop()}`;
    const back = document.createElement('button');
    back.type = 'button';
    back.className = 'btn btn-sm btn-ghost';
    back.textContent = '← 返回候选比较';
    back.addEventListener('click', backToTrace);
    crumbs.replaceChildren(trail, back);

    renderCodeSurface($('[data-project-code]'), file.code);

    $$('[data-project-tabs] .tab-control-item').forEach((tab) => {
      const active = tab.dataset.projectTab === file.id;
      tab.classList.toggle('is-selected', active);
      tab.setAttribute('aria-selected', String(active));
    });
    $$('[data-explorer-view="project"] .generated-project-tree__row').forEach((row) => {
      row.classList.toggle('is-active', row.dataset.projectFile === file.id);
    });
  };

  const generateProject = (key) => {
    projectCandidate = key;
    projectFile = 'orchestration';
    const candidate = D.candidates[key];
    const files = D.projectFiles(key);

    $('[data-explorer-title]').textContent = '工程视图';
    $('[data-explorer-meta]').textContent = `${candidate.name} · ${files.length} 个文件`;
    $('[data-explorer-view="operator"]').hidden = true;
    $('[data-explorer-view="project"]').hidden = false;
    $('#generationTracePanel').hidden = true;
    $('[data-project-view]').hidden = false;
    $('[data-project-tabstrip]').hidden = false;
    $('[data-center-title]').textContent = 'decode_csa';
    $('[data-center-meta]').textContent = `${candidate.name} · 入口 ${candidate.entry}`;
    setText('[data-project-status]', `工程视图已打开：${files.length} 个文件（节选）。`);

    // 行是 4 列网格：chevron · icon · name · status（见 base.css），
    // 缩进由 --tree-depth 控制，不要用 padding 覆盖。
    const tree = $('[data-explorer-view="project"]');
    tree.replaceChildren(...D.projectTree.map((node) => {
      const isFile = node.type === 'file';
      const row = document.createElement(isFile ? 'button' : 'div');
      row.className = isFile
        ? 'generated-project-tree__row'
        : 'generated-project-tree__row generated-project-tree__group';
      row.style.setProperty('--tree-depth', String(node.depth));
      row.innerHTML =
        `${isFile ? '<span aria-hidden="true"></span>' : OPEN_CHEVRON}` +
        `${isFile ? FILE_ICON : FOLDER_ICON}` +
        `<span class="generated-project-tree__name">${node.name}</span>` +
        `<span class="generated-project-tree__status${isFile ? ' is-modified' : ''}">${isFile ? 'A' : ''}</span>`;
      if (isFile) {
        row.type = 'button';
        row.dataset.projectFile = node.id;
        row.addEventListener('click', () => {
          projectFile = node.id;
          renderProjectCode();
        });
      }
      return row;
    }));

    const tabs = $('[data-project-tabs]');
    tabs.replaceChildren(...files.map((file) => {
      const tab = document.createElement('button');
      tab.type = 'button';
      tab.className = 'tab-control-item';
      tab.dataset.projectTab = file.id;
      tab.setAttribute('role', 'tab');
      tab.textContent = file.path.split('/').pop();
      tab.addEventListener('click', () => {
        projectFile = file.id;
        renderProjectCode();
      });
      return tab;
    }));

    renderProjectCode();
  };

  const backToTrace = () => {
    $('[data-explorer-title]').textContent = 'Kernel Intent';
    $('[data-explorer-meta]').textContent = 'DeepSeek-V4 CSA · TP 2';
    $('[data-explorer-view="operator"]').hidden = false;
    $('[data-explorer-view="project"]').hidden = true;
    $('#generationTracePanel').hidden = false;
    $('[data-project-view]').hidden = true;
    $('[data-project-tabstrip]').hidden = true;
    $('[data-center-title]').textContent = 'Generation Trace';
    $('[data-center-meta]').textContent = 'Candidates · TP 世界';
  };

  /* ── 启动 ───────────────────────────────────────────────────────── */
  const init = () => {
    bindApiPills();
    renderCandidateTable();

    // 谱系组件挂载。它对每一步做数量守恒校验，数据不守恒会抛错；
    // 裸调用一旦抛错 init() 就静默中断，所以把失败显式画到页面上。
    const host = $('#lineageMapSurface');
    let controller = null;
    if (!window.PtoCandidatePopulationLineage) {
      host.innerHTML = '<div class="disclaimer-note"><strong>谱系组件未加载</strong>'
        + '<span>components/candidate-population-lineage/pattern.js 没有执行，检查该文件是否存在。</span></div>';
    } else {
      try {
        controller = window.PtoCandidatePopulationLineage.render(host, {
          data: D.lineage,
          initialSelection: 'shortlist',
          onSelect: applySelection
        });
      } catch (error) {
        host.innerHTML = '<div class="disclaimer-note"><strong>谱系渲染失败</strong><span></span></div>';
        $('span', host).textContent = String(error && error.message ? error.message : error);
      }
    }

    if (controller) {
      // 分支色带与拒绝扇的填充只来自 pattern.css；它没加载的话，
      // 画面上会只剩那条内联渐变的「通过带」。这里显式查一下。
      const svg = $('.pto-candidate-lineage__svg', host);
      if (svg && !getComputedStyle(svg).minWidth.startsWith('720')) {
        const note = document.createElement('div');
        note.className = 'disclaimer-note';
        note.innerHTML = '<strong>组件样式未生效</strong><span>'
          + 'components/candidate-population-lineage/pattern.css 没有加载，'
          + '分支色带与拒绝扇会不可见，只剩一条渐变通过带。</span>';
        host.prepend(note);
      }
      window.addEventListener('resize', () => controller.resize());
      applySelection(controller.getSelection());
      $('[data-show-trace]').addEventListener('click', () => {
        controller.select('tile-space');
        host.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
    } else {
      // 组件起不来时，按钮退回到直接打开第一个阶段的详情。
      showCandidates();
      $('[data-show-trace]').addEventListener('click', () => {
        applySelection({ id: 'tile-space', kind: 'stage' });
      });
    }

    $$('[data-show-pipeline]').forEach((button) => {
      button.addEventListener('click', () => showPipeline(button.dataset.showPipeline));
    });
    $('[data-close-pipeline]').addEventListener('click', () => { $('[data-pipeline-panel]').hidden = true; });

    $$('[data-code-preview]').forEach((button) => {
      button.addEventListener('click', () => openCodePreview(button.dataset.codePreview));
    });
    $$('[data-close-code-preview]').forEach((button) => {
      button.addEventListener('click', closeCodePreview);
    });
    $('[data-generate-project-from-preview]').addEventListener('click', () => {
      closeCodePreview();
      generateProject(previewCandidate);
    });
    $$('[data-generate-project]').forEach((button) => {
      button.addEventListener('click', () => generateProject(button.dataset.generateProject));
    });

    document.addEventListener('keydown', (event) => {
      if (event.key !== 'Escape') return;
      if (!$('[data-code-preview-modal]').hidden) closeCodePreview();
      else if (projectCandidate && !$('[data-project-view]').hidden) backToTrace();
    });
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
