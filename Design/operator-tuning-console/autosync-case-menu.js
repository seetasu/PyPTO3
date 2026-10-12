/* Add the PTOAS autosync investigation to the existing case menu without
 * changing the tuning console's measured run datasets. */
(function () {
  'use strict';

  const menu = document.getElementById('caseMenu');
  if (!menu) return;

  function addCaseLink() {
    if (menu.querySelector('[data-autosync-case]')) return;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'tc-case-item';
    button.dataset.autosyncCase = 'true';
    button.innerHTML = '<div class="hd"><span class="nm">PTOAS · AutoSync</span>'
      + '<span class="sub">编译性能案例</span></div>'
      + '<div class="ly"><span class="on">性能</span><span class="on">同步</span>'
      + '<span class="off">L2 task</span></div>'
      + '<div class="mt">#643 GEMM · #646 FlashAttention</div>';
    button.addEventListener('click', () => {
      window.location.href = 'autosync-pass-case.html';
    });
    menu.appendChild(button);
  }

  addCaseLink();
  new MutationObserver(addCaseLink).observe(menu, { childList: true });
})();
