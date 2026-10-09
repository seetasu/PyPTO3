import { operatorModel as model } from './operator-model.js';

const $ = (selector) => document.querySelector(selector);

const state = {
  workUnit: model.workUnits.find((unit) => unit.id === model.selectedWorkUnit),
  stepIndex: 3,
  playing: false,
  timer: null,
  matrices: {},
  playback: null,
  sourceOverride: null,
  sourceTab: 'matmul.py',
};

const fmt = (range) => `${range[0]}:${range[1]}`;

const esc = (value) =>
  String(value).replace(/[&<>"']/g, (char) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  })[char]);

function syntax(line) {
  let text = esc(line);
  text = text.replace(/(&quot;[^&]*?&quot;)/g, '<span class="syntax-string">$1</span>');
  text = text.replace(/\b(256|64)\b/g, '<span class="syntax-number">$1</span>');
  text = text.replace(/\b(import|as|def|for|in|with|return)\b/g, '<span class="syntax-keyword">$1</span>');
  text = text.replace(/\b(pl|Tensor|Out|FP32|parallel|matmul)\b/g, '<span class="syntax-type">$1</span>');
  return text;
}

function sourceLinesForStep() {
  if (state.sourceTab === 'all') return [];
  return state.sourceOverride || model.steps[state.stepIndex].sourceRefs;
}

function renderSource() {
  const selected = sourceLinesForStep();
  const scrollTarget = () => $('#sourceCode .is-active');

  $('#sourceCode').innerHTML = model.source.lines
    .map((line, index) => {
      const isActive = selected.includes(index + 1) ? ' is-active' : '';
      return `<div class="explorer-code-line${isActive}" data-line="${index + 1}">
        <span class="explorer-code-num">${index + 1}</span>
        <code class="explorer-code-text">${syntax(line) || '&nbsp;'}</code>
      </div>`;
    })
    .join('');

  requestAnimationFrame(() => scrollTarget()?.scrollIntoView({ block: 'center' }));
}

function matrixScene(tensor) {
  const workUnit = state.workUnit;
  const region = tensor === 'A'
    ? workUnit.inputs.A
    : tensor === 'B'
      ? workUnit.inputs.B
      : workUnit.output;
  const cells = [];

  for (let y = 0; y < 16; y += 1) {
    for (let x = 0; x < 16; x += 1) {
      cells.push({
        row: y * 16,
        column: x * 16,
        rowSpan: 16,
        columnSpan: 16,
        style: 'value',
        tone: 'neutral',
        states: [],
      });
    }
  }

  return {
    extent: { rows: 256, columns: 256 },
    axes: {
      rows: tensor === 'B' ? 'K' : 'M',
      columns: tensor === 'A' ? 'K' : 'N',
    },
    cells,
  };
}

function renderMatrices() {
  for (const tensor of ['A', 'B', 'C']) {
    const scene = matrixScene(tensor);
    const canvas = $(`#matrix${tensor}`);
    if (state.matrices[tensor]) {
      state.matrices[tensor].update(scene);
    } else {
      state.matrices[tensor] = window.PtoMatrixCanvas.render(canvas, scene, {
        ariaLabel: `${tensor} selected region`,
        showAxes: false,
        showTooltip: false,
        interactive: false,
        autoFit: true,
        padding: { top: 0, right: 0, bottom: 0, left: 0 },
      });
    }
  }

  const workUnit = state.workUnit;
  $('#captionA').textContent = fmt(workUnit.inputs.A.rows);
  $('#captionB').textContent = fmt(workUnit.inputs.B.cols);
  $('#captionC').textContent = fmt(workUnit.output.cols);

  for (const [tensor, region] of [['A', workUnit.inputs.A], ['B', workUnit.inputs.B], ['C', workUnit.output]]) {
    const focus = $(`#focus${tensor}`);
    focus.style.left = `${(region.cols[0] / 256) * 100}%`;
    focus.style.top = `${(region.rows[0] / 256) * 100}%`;
    focus.style.width = `${((region.cols[1] - region.cols[0]) / 256) * 100}%`;
    focus.style.height = `${((region.rows[1] - region.rows[0]) / 256) * 100}%`;
  }

  document.querySelectorAll('.tensor-block').forEach((block) => {
    block.dataset.m = workUnit.indices.m;
    block.dataset.n = workUnit.indices.n;
  });
}

function renderWorkUnits() {
  const workUnit = state.workUnit;

  $('#workUnitList').innerHTML = model.workUnits
    .map(
      (unit) => `<button class="work-unit${unit.id === workUnit.id ? ' is-selected' : ''}" type="button" data-unit="${unit.id}" role="option" aria-selected="${unit.id === workUnit.id}">
        <span>${unit.id}</span><span>→ ${unit.indices.m},${unit.indices.n}</span>
      </button>`
    )
    .join('');

  $('#instructionContext').innerHTML =
    `4 × 4 logical work units&nbsp; · &nbsp;selected ${workUnit.id}&nbsp; · &nbsp;no physical core mapping`;

  $('#outputCard').innerHTML =
    `<span>Select<br>Output Tile</span><code>C[M${workUnit.indices.m},N${workUnit.indices.n}]<br>64 × 64</code>`;

  $('#mOptions').innerHTML = Array.from({ length: 4 }, (_, index) =>
    `<button type="button" class="loop-option${index === workUnit.indices.m ? ' is-selected' : ''}" data-m="${index}">M${index}</button>`
  ).join('');

  $('#nOptions').innerHTML = Array.from({ length: 4 }, (_, index) =>
    `<button type="button" class="loop-option${index === workUnit.indices.n ? ' is-selected' : ''}" data-n="${index}">N${index}</button>`
  ).join('');
}

function renderSteps() {
  const workUnit = state.workUnit;
  const currentStep = model.steps[state.stepIndex];

  const items = [
    { id: 'enter', title: 'Enter CORE_GROUP', detail: 'matmul_tile' },
    { id: 'sliceA', title: 'Slice A', detail: `A[${fmt(workUnit.inputs.A.rows)}, :] → [64,256]` },
    { id: 'sliceB', title: 'Slice B', detail: `B[:, ${fmt(workUnit.inputs.B.cols)}] → [256,64]` },
    { id: 'matmul', title: 'MatMul', detail: '[64,256] × [256,64] → [64,64]' },
    { id: 'write', title: 'Write C', detail: `result → C[${fmt(workUnit.output.rows)}, ${fmt(workUnit.output.cols)}]` },
  ];

  $('#journeyList').innerHTML = items
    .map((item, index) => {
      const stepIndex = model.steps.findIndex((step) => step.id === item.id);
      const selected = currentStep.id === item.id ? ' is-selected' : '';
      const arrow = index < items.length - 1 ? '<span class="step-arrow">→</span>' : '';
      return `<button type="button" class="instruction-card${selected}" data-step="${stepIndex}">
        <strong>${item.title}</strong><code>${esc(item.detail)}</code>
      </button>${arrow}`;
    })
    .join('');

  syncPlayback();
  renderSource();
}

function render() {
  renderWorkUnits();
  renderMatrices();
  renderSteps();
}

function selectUnit(id) {
  state.workUnit = model.workUnits.find((unit) => unit.id === id) || state.workUnit;
  state.sourceOverride = null;
  render();
}

function selectStep(index) {
  state.stepIndex = Math.max(0, Math.min(model.steps.length - 1, index));
  state.sourceOverride = null;
  renderSteps();
}

function setPlaying(playing) {
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
  state.playing = playing;

  if (playing) {
    if (state.stepIndex >= model.steps.length - 1) selectStep(0);
    state.timer = setInterval(() => {
      if (state.stepIndex >= model.steps.length - 1) {
        setPlaying(false);
        return;
      }
      selectStep(state.stepIndex + 1);
    }, 1100);
  }

  syncPlayback();
}

function syncPlayback() {
  const helper = window.PtoFloatingPlaybackControl;
  const byId = (id) => document.getElementById(id);

  if (byId('explorer-scrubber')) {
    byId('explorer-scrubber').max = String(model.steps.length - 1);
    byId('explorer-scrubber').value = String(state.stepIndex);
  }
  if (byId('explorer-scrubber-label')) {
    byId('explorer-scrubber-label').textContent =
      `${state.stepIndex + 1} / ${model.steps.length} · ${state.workUnit.id} · ${model.steps[state.stepIndex].label}`;
  }
  if (byId('explorer-scrubber-opname')) byId('explorer-scrubber-opname').textContent = '';
  if (byId('explorer-play')) {
    byId('explorer-play').innerHTML = helper.iconLabel(
      state.playing ? 'pause' : 'play',
      state.playing ? 'Pause' : 'Play'
    );
  }

  state.playback?.sync({ playing: state.playing });
}

function initPlayback() {
  const mount = $('#playbackMount');
  if (!mount) return;

  const helper = window.PtoFloatingPlaybackControl;
  const ids = {
    shell: 'explorer-shell',
    toggle: 'explorer-toggle',
    collapsedButton: 'explorer-collapsed',
    collapsedIcon: 'explorer-collapsed-icon',
    controls: 'explorer-controls',
    stepBack: 'explorer-back',
    play: 'explorer-play',
    stepForward: 'explorer-forward',
    replay: 'explorer-replay',
    scrubber: 'explorer-scrubber',
    scrubberLabel: 'explorer-scrubber-label',
    scrubberOpname: 'explorer-scrubber-opname',
    scrubberHover: 'explorer-scrubber-hover',
  };

  const control = helper.createControl({ ids, className: 'pto-floating-playback--preview' });
  mount.appendChild(control);
  control.querySelector('.pto-floating-playback__label').textContent = 'Logical order';

  state.playback = helper.init({ root: control, isPlaying: () => state.playing });
  helper.initScrubberHover({
    root: control,
    getTotalSteps: () => model.steps.length,
    getLabelForStep: (index) => model.steps[index]?.label || '',
  });

  $('#explorer-back').onclick = () => {
    setPlaying(false);
    selectStep(state.stepIndex - 1);
  };
  $('#explorer-forward').onclick = () => {
    setPlaying(false);
    selectStep(state.stepIndex + 1);
  };
  $('#explorer-play').onclick = () => setPlaying(!state.playing);
  $('#explorer-replay').onclick = () => {
    setPlaying(false);
    selectStep(0);
  };
  $('#explorer-scrubber').oninput = (event) => {
    setPlaying(false);
    selectStep(Number(event.target.value));
  };
}

function setDrawer(open) {
  $('#sourceDrawer').classList.toggle('is-open', open);
  $('#sourceDrawer').setAttribute('aria-hidden', String(!open));
  $('#evidenceToggle').setAttribute('aria-expanded', String(open));
}

function showDetails(kind) {
  let title;
  let html;

  if (kind === 'inside') {
    title = 'Inside pl.matmul';
    html =
      '<p>K = 256 is consumed in one source-level MatMul. Internal K tiling, memory movement and instruction sequence require compiler evidence.</p><p>Not available from source.</p>';
  } else {
    title = 'Source Evidence';
    html = `<p>Verified excerpt from <a href="${model.source.url}" target="_blank" rel="noopener noreferrer">examples/beginner/matmul.py ↗</a></p>`;
  }

  $('#drawerTitle').textContent = title;
  $('#drawerContent').innerHTML = html;
  setDrawer(true);
}

function focusSource(kind) {
  state.sourceOverride = model.sourceRefs[kind] || null;
  state.sourceTab = 'matmul.py';
  $('#sourceFileTab').classList.add('is-selected');
  $('#sourceAllTab').classList.remove('is-selected');
  $('#sourceFileTab').setAttribute('aria-selected', 'true');
  $('#sourceAllTab').setAttribute('aria-selected', 'false');
  renderSource();
  $('#sourceCode .is-active')?.scrollIntoView({ block: 'center' });
}

document.addEventListener('click', (event) => {
  const unit = event.target.closest('[data-unit]');
  if (unit) {
    selectUnit(unit.dataset.unit);
    return;
  }

  const axis = event.target.closest('[data-m],[data-n]');
  if (axis) {
    const m = axis.dataset.m ?? state.workUnit.indices.m;
    const n = axis.dataset.n ?? state.workUnit.indices.n;
    selectUnit(`M${m}N${n}`);
    return;
  }

  const step = event.target.closest('[data-step]');
  if (step) {
    setPlaying(false);
    selectStep(Number(step.dataset.step));
    return;
  }

  const source = event.target.closest('[data-source]');
  if (source) {
    focusSource(source.dataset.source);
    return;
  }

  if (event.target.closest('#insideMatmul')) showDetails('inside');
  if (event.target.closest('#evidenceToggle')) setDrawer(!$('#sourceDrawer').classList.contains('is-open'));
  if (event.target.closest('#closeDrawer')) setDrawer(false);
});

$('#sourceAllTab').onclick = () => {
  state.sourceTab = 'all';
  $('#sourceAllTab').classList.add('is-selected');
  $('#sourceFileTab').classList.remove('is-selected');
  $('#sourceAllTab').setAttribute('aria-selected', 'true');
  $('#sourceFileTab').setAttribute('aria-selected', 'false');
  renderSource();
};
$('#sourceFileTab').onclick = () => {
  state.sourceTab = 'matmul.py';
  $('#sourceFileTab').classList.add('is-selected');
  $('#sourceAllTab').classList.remove('is-selected');
  $('#sourceFileTab').setAttribute('aria-selected', 'true');
  $('#sourceAllTab').setAttribute('aria-selected', 'false');
  renderSource();
};

document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') setDrawer(false);
});

initPlayback();
render();
