import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const source = path.join(here, '../data/callable');
const target = path.join(here, '../dist/data/callable');

fs.mkdirSync(path.dirname(target), { recursive: true });
fs.cpSync(source, target, { recursive: true, force: true });
console.log('copied scenario JSON to dist/data/callable');

fs.cpSync(path.join(here,'../data/csa-qk-pv'),path.join(here,'../dist/data/csa-qk-pv'),{recursive:true,force:true});
fs.copyFileSync(path.join(here,'../data/memory.json'),path.join(here,'../dist/data/memory.json'));
console.log('copied qk_pv scenario and memory data');
