import { readFile, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const moduleDir = path.dirname(fileURLToPath(import.meta.url));
const projectDir = path.resolve(moduleDir, '..');
const distDir = path.join(projectDir, 'dist');
const indexPath = path.join(distDir, 'index.html');

let html = await readFile(indexPath, 'utf8');

const scriptPattern = /<script\b[^>]*\bsrc=["']([^"']+)["'][^>]*><\/script>/i;
const stylesheetPattern = /<link\b[^>]*\brel=["']stylesheet["'][^>]*\bhref=["']([^"']+)["'][^>]*>/i;
const scriptMatch = html.match(scriptPattern);
const stylesheetMatch = html.match(stylesheetPattern);

if (!scriptMatch || !stylesheetMatch) {
  throw new Error('Expected one Vite script and stylesheet reference in dist/index.html');
}

function assetPath(relativePath) {
  return path.join(distDir, relativePath.replace(/^\.\//, ''));
}

const [javascript, stylesheet] = await Promise.all([
  readFile(assetPath(scriptMatch[1]), 'utf8'),
  readFile(assetPath(stylesheetMatch[1]), 'utf8'),
]);

const inlineScript = `<script data-csa-pass-field-bundle>\n${javascript.replaceAll('</script', '<\\/script')}\n</script>`;

html = html
  .replace(
    stylesheetPattern,
    () => `<style data-csa-pass-field-bundle>\n${stylesheet.replaceAll('</style', '<\\/style')}\n</style>`,
  )
  .replace(scriptPattern, '')
  .replace('</body>', () => `${inlineScript}\n  </body>`);

if (/<script\b[^>]*\bsrc=/i.test(html) || /<link\b[^>]*\brel=["']stylesheet["']/i.test(html)) {
  throw new Error('Standalone output still contains external script or stylesheet references');
}

await writeFile(indexPath, html, 'utf8');

const { size } = await stat(indexPath);
console.log(`inlined production page: dist/index.html (${Math.round(size / 1024)} KiB)`);

const sha256 = value => crypto.createHash('sha256').update(value).digest('hex');
const sourceManifest = JSON.parse(await readFile(path.join(projectDir, 'data/callable/manifest.json'), 'utf8'));
await writeFile(path.join(distDir, 'artifact-manifest.json'), JSON.stringify({
  schemaVersion: 'csa-pass-field.artifact.v1',
  runId: sourceManifest.buildIdentity.runId,
  htmlSha256: sha256(html),
  generatorSha256: sourceManifest.buildIdentity.generatorSha256,
  inputHashes: sourceManifest.buildIdentity.inputHashes,
}, null, 2) + '\n');
