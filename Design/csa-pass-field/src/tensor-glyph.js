// Current legend 01.4 / training tensorNode: 11px cells, 3px gaps.
export function tensorGeometry(data) {
  const detail = `${(data.attributes?.shape || []).join(' × ')}  ${data.attributes?.dtype || ''}`.trim();
  const measure = (text, size) => [...text].reduce((sum, c) => sum + size * (/[\u2e80-\uffff]/.test(c) ? 1 : /[ilI .·]/.test(c) ? .3 : /[MWQO]/.test(c) ? .85 : .6), 0);
  const width = Math.ceil(Math.max(104, measure(data.label || '', 16) + 28, measure(detail, 12) + 28));
  return { width, height: 64, detail, columns: Math.floor((width - 7) / 14) };
}
