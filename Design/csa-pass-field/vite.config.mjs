import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  base: './',
  plugins: [react()],
  server: { host: '127.0.0.1', port: 4197 },
  preview: { host: '127.0.0.1', port: 4198 },
  build: { outDir: 'dist', emptyOutDir: true },
});
