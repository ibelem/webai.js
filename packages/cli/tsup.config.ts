import { defineConfig } from 'tsup';

export default defineConfig([
  {
    entry: ['src/cli.ts'],
    format: ['esm'],
    tsconfig: 'tsconfig.build.json',
    dts: false,
    clean: true,
    sourcemap: true,
    banner: {
      js: '#!/usr/bin/env node',
    },
  },
  {
    entry: ['src/index.ts'],
    format: ['esm'],
    tsconfig: 'tsconfig.build.json',
    dts: true,
    clean: false,
    sourcemap: true,
  },
]);
