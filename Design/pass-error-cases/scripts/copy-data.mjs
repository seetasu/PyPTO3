import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const source = path.join(here, '../data/csa-qk-pv');
const target = path.join(here, '../dist/data/csa-qk-pv');

fs.mkdirSync(path.dirname(target), { recursive: true });
fs.cpSync(source, target, { recursive: true, force: true });
console.log('copied scenario JSON to dist/data/csa-qk-pv');
