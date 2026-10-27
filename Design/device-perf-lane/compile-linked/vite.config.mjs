import {defineConfig} from '../../csa-pass-field/node_modules/vite/dist/node/index.js';

import {fileURLToPath} from 'node:url';
const here=fileURLToPath(new URL('.',import.meta.url));
const modules=fileURLToPath(new URL('../../csa-pass-field/node_modules/',import.meta.url));
export default defineConfig({define:{"process.env.NODE_ENV":JSON.stringify("production")},esbuild:{jsx:"automatic"},resolve:{alias:{react:modules+'react', 'react-dom':modules+'react-dom',reactflow:modules+'reactflow',dagre:modules+'dagre'}},build:{outDir:here+'dist',emptyOutDir:true,lib:{entry:here+'main.jsx',name:'CompileLinked',formats:['iife'],fileName:()=> 'compile-linked.js',cssFileName:'compile-linked'},assetsInlineLimit:100000}});
