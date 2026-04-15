import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  base: '/webai.js/',
  resolve: {
    alias: {
      '@webai/core': resolve(__dirname, '../core/src/index.ts'),
      'webai': resolve(__dirname, '../cli/src/assembler.ts'),
    },
  },
  build: {
    outDir: 'dist',
  },
});
