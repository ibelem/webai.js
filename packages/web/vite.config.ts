import { defineConfig } from 'vite';
import { resolve } from 'path';
import basicSsl from '@vitejs/plugin-basic-ssl';

export default defineConfig({
  base: '/webai.js/',
  plugins: [basicSsl()],
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
