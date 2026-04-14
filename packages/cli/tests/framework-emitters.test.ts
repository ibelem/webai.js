/**
 * Framework emitter tests (Layer 2).
 *
 * Tests file structure, CodeBlock inclusion, CSS presence,
 * ARIA accessibility attributes, package.json dependencies, README content.
 */

import { describe, it, expect } from 'vitest';
import type { ResolvedConfig, ModelMetadata } from '@webai/core';
import type { GeneratedFile } from '../src/types.js';
import { emitLayer1 } from '../src/emitters/index.js';
import { emitLayer2 } from '../src/frameworks/index.js';
import { emitHtml } from '../src/frameworks/html.js';
import { emitReactVite } from '../src/frameworks/react-vite.js';
import { emitVanillaVite } from '../src/frameworks/vanilla-vite.js';
import { emitNextjs } from '../src/frameworks/nextjs.js';
import { emitSvelteKit } from '../src/frameworks/sveltekit.js';

const classificationMeta: ModelMetadata = {
  format: 'onnx',
  inputs: [{ name: 'input', dataType: 'float32', shape: [1, 3, 224, 224] }],
  outputs: [{ name: 'output', dataType: 'float32', shape: [1, 1000] }],
};

/** Safe file lookup — fails test if file not found */
function getFile(files: GeneratedFile[], path: string): string {
  const file = files.find((f) => f.path === path);
  expect(file, `Expected file "${path}" to exist`).toBeDefined();
  return file?.content ?? '';
}

function makeConfig(overrides: Partial<ResolvedConfig> = {}): ResolvedConfig {
  return {
    task: 'image-classification',
    engine: 'ort',
    backend: 'auto',
    framework: 'html',
    input: 'file',
    mode: 'raw',
    lang: 'js',
    outputDir: './output/',
    offline: false,
    theme: 'dark',
    verbose: false,
    force: false,
    preprocess: {
      imageSize: 224,
      mean: [0.485, 0.456, 0.406],
      std: [0.229, 0.224, 0.225],
      layout: 'nchw',
    },
    preprocessIsDefault: true,
    modelMeta: classificationMeta,
    modelPath: './mobilenet.onnx',
    modelName: 'mobilenet',
    modelSource: 'local-path',
    ...overrides,
  };
}

function generateHtml(overrides: Partial<ResolvedConfig> = {}) {
  const config = makeConfig({ framework: 'html', ...overrides });
  const blocks = emitLayer1(config);
  return { files: emitHtml(config, blocks), config, blocks };
}

function generateReactVite(overrides: Partial<ResolvedConfig> = {}) {
  const config = makeConfig({ framework: 'react-vite', ...overrides });
  const blocks = emitLayer1(config);
  return { files: emitReactVite(config, blocks), config, blocks };
}

function generateVanillaVite(overrides: Partial<ResolvedConfig> = {}) {
  const config = makeConfig({ framework: 'vanilla-vite', ...overrides });
  const blocks = emitLayer1(config);
  return { files: emitVanillaVite(config, blocks), config, blocks };
}

function generateNextjs(overrides: Partial<ResolvedConfig> = {}) {
  const config = makeConfig({ framework: 'nextjs', ...overrides });
  const blocks = emitLayer1(config);
  return { files: emitNextjs(config, blocks), config, blocks };
}

function generateSvelteKit(overrides: Partial<ResolvedConfig> = {}) {
  const config = makeConfig({ framework: 'sveltekit', ...overrides });
  const blocks = emitLayer1(config);
  return { files: emitSvelteKit(config, blocks), config, blocks };
}

// ---- emitLayer2 dispatcher ----

describe('emitLayer2', () => {
  it('dispatches to html emitter', () => {
    const config = makeConfig({ framework: 'html' });
    const blocks = emitLayer1(config);
    const files = emitLayer2(config, blocks);
    expect(files.some((f) => f.path === 'index.html')).toBe(true);
  });

  it('dispatches to react-vite emitter', () => {
    const config = makeConfig({ framework: 'react-vite' });
    const blocks = emitLayer1(config);
    const files = emitLayer2(config, blocks);
    expect(files.some((f) => f.path === 'package.json')).toBe(true);
  });

  it('dispatches to nextjs framework', () => {
    const config = makeConfig({ framework: 'nextjs' });
    const blocks = emitLayer1(config);
    const files = emitLayer2(config, blocks);
    expect(files.some((f) => f.path === 'package.json')).toBe(true);
    expect(files.some((f) => f.path.startsWith('app/'))).toBe(true);
  });

  it('dispatches to vanilla-vite framework', () => {
    const config = makeConfig({ framework: 'vanilla-vite' });
    const blocks = emitLayer1(config);
    const files = emitLayer2(config, blocks);
    expect(files.some((f) => f.path === 'package.json')).toBe(true);
    expect(files.some((f) => f.path === 'src/style.css')).toBe(true);
  });

  it('dispatches to sveltekit framework', () => {
    const config = makeConfig({ framework: 'sveltekit' });
    const blocks = emitLayer1(config);
    const files = emitLayer2(config, blocks);
    expect(files.some((f) => f.path === 'package.json')).toBe(true);
    expect(files.some((f) => f.path === 'src/routes/+page.svelte')).toBe(true);
  });

  it('throws for unsupported framework', () => {
    const config = makeConfig({ framework: 'unknown' as ResolvedConfig['framework'] });
    const blocks = emitLayer1(config);
    expect(() => emitLayer2(config, blocks)).toThrow(/Unsupported framework/);
  });
});

// ---- HTML framework ----

describe('emitHtml', () => {
  describe('file structure', () => {
    it('produces index.html and README.md', () => {
      const { files } = generateHtml();
      const paths = files.map((f) => f.path);
      expect(paths).toContain('index.html');
      expect(paths).toContain('README.md');
      expect(files).toHaveLength(2);
    });
  });

  describe('index.html content', () => {
    it('includes DOCTYPE and html lang', () => {
      const { files } = generateHtml();
      const html = getFile(files, 'index.html');
      expect(html).toContain('<!DOCTYPE html>');
      expect(html).toContain('lang="en"');
    });

    it('includes CSS design system variables', () => {
      const { files } = generateHtml();
      const html = getFile(files, 'index.html');
      expect(html).toContain('--webai-bg');
      expect(html).toContain('--webai-accent');
      expect(html).toContain('--webai-font-mono');
    });

    it('includes light theme override', () => {
      const { files } = generateHtml();
      const html = getFile(files, 'index.html');
      expect(html).toContain('[data-theme="light"]');
    });

    it('sets data-theme from config', () => {
      const { files } = generateHtml({ theme: 'light' });
      const html = getFile(files, 'index.html');
      expect(html).toContain('data-theme="light"');
    });

    it('includes model name in title', () => {
      const { files } = generateHtml();
      const html = getFile(files, 'index.html');
      expect(html).toContain('<title>mobilenet — Image Classification</title>');
    });

    it('includes preprocessing code', () => {
      const { files } = generateHtml();
      const html = getFile(files, 'index.html');
      expect(html).toContain('function resizeImage');
      expect(html).toContain('function normalize');
      expect(html).toContain('function toNCHW');
      expect(html).toContain('function preprocessImage');
    });

    it('includes inference code with CDN import', () => {
      const { files } = generateHtml();
      const html = getFile(files, 'index.html');
      expect(html).toContain('cdn.jsdelivr.net/npm/onnxruntime-web');
      expect(html).toContain('function createSession');
      expect(html).toContain('function runInference');
    });

    it('includes postprocessing code', () => {
      const { files } = generateHtml();
      const html = getFile(files, 'index.html');
      expect(html).toContain('function softmax');
      expect(html).toContain('function topK');
      expect(html).toContain('function postprocessResults');
    });

    it('includes file input UI (drop zone)', () => {
      const { files } = generateHtml();
      const html = getFile(files, 'index.html');
      expect(html).toContain('drop-zone');
      expect(html).toContain('Drop an image here');
      expect(html).toContain('type="file"');
      expect(html).toContain('accept="image/*"');
    });

    it('includes status bar', () => {
      const { files } = generateHtml();
      const html = getFile(files, 'index.html');
      expect(html).toContain('status-bar');
      expect(html).toContain('mobilenet');
    });

    it('includes footer with attribution', () => {
      const { files } = generateHtml();
      const html = getFile(files, 'index.html');
      expect(html).toContain('Generated by webai.js');
      expect(html).toContain('ORT Web');
    });
  });

  describe('accessibility', () => {
    it('includes ARIA landmarks', () => {
      const { files } = generateHtml();
      const html = getFile(files, 'index.html');
      expect(html).toContain('<main>');
      expect(html).toContain('<aside');
    });

    it('includes live region for results', () => {
      const { files } = generateHtml();
      const html = getFile(files, 'index.html');
      expect(html).toContain('role="status"');
      expect(html).toContain('aria-live="polite"');
      expect(html).toContain('aria-atomic="true"');
    });

    it('includes skip link', () => {
      const { files } = generateHtml();
      const html = getFile(files, 'index.html');
      expect(html).toContain('skip-link');
      expect(html).toContain('Skip to results');
    });

    it('drop zone has role and keyboard support', () => {
      const { files } = generateHtml();
      const html = getFile(files, 'index.html');
      expect(html).toContain('role="button"');
      expect(html).toContain('tabindex="0"');
      expect(html).toContain('aria-label');
    });

    it('has reduced-motion media query', () => {
      const { files } = generateHtml();
      const html = getFile(files, 'index.html');
      expect(html).toContain('prefers-reduced-motion');
    });
  });

  describe('README', () => {
    it('includes model name and task', () => {
      const { files } = generateHtml();
      const readme = getFile(files, 'README.md');
      expect(readme).toContain('mobilenet');
      expect(readme).toContain('Image Classification');
    });

    it('includes quick start instructions', () => {
      const { files } = generateHtml();
      const readme = getFile(files, 'README.md');
      expect(readme).toContain('local server');
    });

    it('includes how it works section', () => {
      const { files } = generateHtml();
      const readme = getFile(files, 'README.md');
      expect(readme).toContain('Preprocessing');
      expect(readme).toContain('224x224');
      expect(readme).toContain('softmax');
    });
  });
});

// ---- React-Vite framework ----

describe('emitReactVite', () => {
  describe('file structure (JS)', () => {
    it('produces correct file set', () => {
      const { files } = generateReactVite({ lang: 'js' });
      const paths = files.map((f) => f.path);
      expect(paths).toContain('package.json');
      expect(paths).toContain('vite.config.js');
      expect(paths).toContain('index.html');
      expect(paths).toContain('src/main.jsx');
      expect(paths).toContain('src/App.jsx');
      expect(paths).toContain('src/App.css');
      expect(paths).toContain('src/lib/preprocess.js');
      expect(paths).toContain('src/lib/inference.js');
      expect(paths).toContain('src/lib/postprocess.js');
      expect(paths).toContain('README.md');
      expect(files).toHaveLength(10);
    });
  });

  describe('file structure (TS)', () => {
    it('uses .tsx and .ts extensions', () => {
      const { files } = generateReactVite({ lang: 'ts' });
      const paths = files.map((f) => f.path);
      expect(paths).toContain('src/main.tsx');
      expect(paths).toContain('src/App.tsx');
      expect(paths).toContain('src/lib/preprocess.ts');
      expect(paths).toContain('src/lib/inference.ts');
      expect(paths).toContain('src/lib/postprocess.ts');
    });
  });

  describe('package.json', () => {
    it('includes react and react-dom dependencies', () => {
      const { files } = generateReactVite();
      const pkg = JSON.parse(getFile(files, 'package.json'));
      expect(pkg.dependencies.react).toBeDefined();
      expect(pkg.dependencies['react-dom']).toBeDefined();
    });

    it('includes onnxruntime-web dependency', () => {
      const { files } = generateReactVite();
      const pkg = JSON.parse(getFile(files, 'package.json'));
      expect(pkg.dependencies['onnxruntime-web']).toBeDefined();
    });

    it('includes vite and react plugin in devDependencies', () => {
      const { files } = generateReactVite();
      const pkg = JSON.parse(getFile(files, 'package.json'));
      expect(pkg.devDependencies.vite).toBeDefined();
      expect(pkg.devDependencies['@vitejs/plugin-react']).toBeDefined();
    });

    it('includes TypeScript devDependencies when lang=ts', () => {
      const { files } = generateReactVite({ lang: 'ts' });
      const pkg = JSON.parse(getFile(files, 'package.json'));
      expect(pkg.devDependencies.typescript).toBeDefined();
      expect(pkg.devDependencies['@types/react']).toBeDefined();
    });

    it('has dev, build, preview scripts', () => {
      const { files } = generateReactVite();
      const pkg = JSON.parse(getFile(files, 'package.json'));
      expect(pkg.scripts.dev).toBe('vite');
      expect(pkg.scripts.build).toBe('vite build');
      expect(pkg.scripts.preview).toBe('vite preview');
    });
  });

  describe('lib modules', () => {
    it('preprocess module exports functions', () => {
      const { files } = generateReactVite();
      const code = getFile(files, 'src/lib/preprocess.js');
      expect(code).toContain('export function resizeImage');
      expect(code).toContain('export function normalize');
      expect(code).toContain('export function toNCHW');
      expect(code).toContain('export function preprocessImage');
    });

    it('inference module exports functions', () => {
      const { files } = generateReactVite();
      const code = getFile(files, 'src/lib/inference.js');
      expect(code).toContain("import * as ort from 'onnxruntime-web'");
      expect(code).toContain('export async function createSession');
      expect(code).toContain('export async function runInference');
    });

    it('postprocess module exports functions', () => {
      const { files } = generateReactVite();
      const code = getFile(files, 'src/lib/postprocess.js');
      expect(code).toContain('export function softmax');
      expect(code).toContain('export function topK');
      expect(code).toContain('export function postprocessResults');
    });
  });

  describe('App component', () => {
    it('imports from lib modules', () => {
      const { files } = generateReactVite();
      const app = getFile(files, 'src/App.jsx');
      expect(app).toContain("from './lib/inference.js'");
      expect(app).toContain("from './lib/preprocess.js'");
      expect(app).toContain("from './lib/postprocess.js'");
    });

    it('uses React hooks', () => {
      const { files } = generateReactVite();
      const app = getFile(files, 'src/App.jsx');
      expect(app).toContain('useState');
      expect(app).toContain('useEffect');
      expect(app).toContain('useRef');
      expect(app).toContain('useCallback');
    });

    it('includes model path reference', () => {
      const { files } = generateReactVite();
      const app = getFile(files, 'src/App.jsx');
      expect(app).toContain('/mobilenet.onnx');
    });
  });

  describe('accessibility', () => {
    it('App has ARIA live region for results', () => {
      const { files } = generateReactVite();
      const app = getFile(files, 'src/App.jsx');
      expect(app).toContain('role="status"');
      expect(app).toContain('aria-live="polite"');
      expect(app).toContain('aria-atomic="true"');
    });

    it('App has skip link', () => {
      const { files } = generateReactVite();
      const app = getFile(files, 'src/App.jsx');
      expect(app).toContain('skip-link');
      expect(app).toContain('Skip to results');
    });

    it('drop zone has role and aria-label', () => {
      const { files } = generateReactVite();
      const app = getFile(files, 'src/App.jsx');
      expect(app).toContain('role="button"');
      expect(app).toContain('aria-label');
    });

    it('CSS includes reduced-motion query', () => {
      const { files } = generateReactVite();
      const css = getFile(files, 'src/App.css');
      expect(css).toContain('prefers-reduced-motion');
    });
  });

  describe('CSS', () => {
    it('includes design system variables', () => {
      const { files } = generateReactVite();
      const css = getFile(files, 'src/App.css');
      expect(css).toContain('--webai-bg');
      expect(css).toContain('--webai-accent');
    });

    it('includes component styles', () => {
      const { files } = generateReactVite();
      const css = getFile(files, 'src/App.css');
      expect(css).toContain('.drop-zone');
      expect(css).toContain('.result-row');
      expect(css).toContain('.status-bar');
    });
  });

  describe('README', () => {
    it('includes npm install instructions', () => {
      const { files } = generateReactVite();
      const readme = getFile(files, 'README.md');
      expect(readme).toContain('npm install');
      expect(readme).toContain('npm run dev');
    });

    it('lists all files', () => {
      const { files } = generateReactVite();
      const readme = getFile(files, 'README.md');
      expect(readme).toContain('package.json');
      expect(readme).toContain('src/App.jsx');
      expect(readme).toContain('src/lib/inference.js');
    });
  });
});

// ---- Vanilla-Vite framework ----

describe('emitVanillaVite', () => {
  describe('file structure (JS)', () => {
    it('produces correct file set', () => {
      const { files } = generateVanillaVite({ lang: 'js' });
      const paths = files.map((f) => f.path);
      expect(paths).toContain('package.json');
      expect(paths).toContain('vite.config.js');
      expect(paths).toContain('index.html');
      expect(paths).toContain('src/main.js');
      expect(paths).toContain('src/style.css');
      expect(paths).toContain('src/lib/preprocess.js');
      expect(paths).toContain('src/lib/inference.js');
      expect(paths).toContain('src/lib/postprocess.js');
      expect(paths).toContain('README.md');
    });
  });

  describe('file structure (TS)', () => {
    it('uses .ts extensions', () => {
      const { files } = generateVanillaVite({ lang: 'ts' });
      const paths = files.map((f) => f.path);
      expect(paths).toContain('src/main.ts');
      expect(paths).toContain('src/lib/preprocess.ts');
      expect(paths).toContain('src/lib/inference.ts');
      expect(paths).toContain('src/lib/postprocess.ts');
    });
  });

  describe('package.json', () => {
    it('includes onnxruntime-web dependency', () => {
      const { files } = generateVanillaVite();
      const pkg = JSON.parse(getFile(files, 'package.json'));
      expect(pkg.dependencies['onnxruntime-web']).toBeDefined();
    });

    it('does not include React dependencies', () => {
      const { files } = generateVanillaVite();
      const pkg = JSON.parse(getFile(files, 'package.json'));
      expect(pkg.dependencies.react).toBeUndefined();
    });

    it('has dev, build, preview scripts', () => {
      const { files } = generateVanillaVite();
      const pkg = JSON.parse(getFile(files, 'package.json'));
      expect(pkg.scripts.dev).toBe('vite');
      expect(pkg.scripts.build).toBe('vite build');
      expect(pkg.scripts.preview).toBe('vite preview');
    });
  });

  describe('lib modules', () => {
    it('preprocess module exports functions', () => {
      const { files } = generateVanillaVite();
      const code = getFile(files, 'src/lib/preprocess.js');
      expect(code).toContain('export function resizeImage');
      expect(code).toContain('export function preprocessImage');
    });

    it('inference module exports functions', () => {
      const { files } = generateVanillaVite();
      const code = getFile(files, 'src/lib/inference.js');
      expect(code).toContain("import * as ort from 'onnxruntime-web'");
      expect(code).toContain('export async function createSession');
      expect(code).toContain('export async function runInference');
    });

    it('postprocess module exports functions', () => {
      const { files } = generateVanillaVite();
      const code = getFile(files, 'src/lib/postprocess.js');
      expect(code).toContain('export function softmax');
      expect(code).toContain('export function topK');
    });
  });

  describe('main module', () => {
    it('imports from lib modules', () => {
      const { files } = generateVanillaVite();
      const main = getFile(files, 'src/main.js');
      expect(main).toContain("from './lib/inference.js'");
      expect(main).toContain("from './lib/preprocess.js'");
      expect(main).toContain("from './lib/postprocess.js'");
    });

    it('references model path', () => {
      const { files } = generateVanillaVite();
      const main = getFile(files, 'src/main.js');
      expect(main).toContain('mobilenet');
    });
  });
});

// ---- Next.js framework ----

describe('emitNextjs', () => {
  describe('file structure (JS)', () => {
    it('produces correct file set', () => {
      const { files } = generateNextjs({ lang: 'js' });
      const paths = files.map((f) => f.path);
      expect(paths).toContain('package.json');
      expect(paths).toContain('next.config.mjs');
      expect(paths).toContain('app/layout.jsx');
      expect(paths).toContain('app/page.jsx');
      expect(paths).toContain('app/globals.css');
      expect(paths).toContain('README.md');
    });
  });

  describe('file structure (TS)', () => {
    it('uses .tsx extensions and includes tsconfig', () => {
      const { files } = generateNextjs({ lang: 'ts' });
      const paths = files.map((f) => f.path);
      expect(paths).toContain('app/layout.tsx');
      expect(paths).toContain('app/page.tsx');
      expect(paths).toContain('tsconfig.json');
    });
  });

  describe('package.json', () => {
    it('includes next, react, and react-dom', () => {
      const { files } = generateNextjs();
      const pkg = JSON.parse(getFile(files, 'package.json'));
      expect(pkg.dependencies.next).toBeDefined();
      expect(pkg.dependencies.react).toBeDefined();
      expect(pkg.dependencies['react-dom']).toBeDefined();
    });

    it('has dev, build, start scripts', () => {
      const { files } = generateNextjs();
      const pkg = JSON.parse(getFile(files, 'package.json'));
      expect(pkg.scripts.dev).toContain('next dev');
      expect(pkg.scripts.build).toContain('next build');
      expect(pkg.scripts.start).toContain('next start');
    });
  });

  describe('app files', () => {
    it('page includes use client directive', () => {
      const { files } = generateNextjs();
      const page = getFile(files, 'app/page.jsx');
      expect(page).toContain("'use client'");
    });

    it('layout includes metadata export', () => {
      const { files } = generateNextjs();
      const layout = getFile(files, 'app/layout.jsx');
      expect(layout).toContain('metadata');
    });
  });

  describe('lib modules', () => {
    it('has inference lib with ort import', () => {
      const { files } = generateNextjs();
      const code = getFile(files, 'lib/inference.js');
      expect(code).toContain("import * as ort from 'onnxruntime-web'");
      expect(code).toContain('export async function createSession');
    });
  });
});

// ---- SvelteKit framework ----

describe('emitSvelteKit', () => {
  describe('file structure', () => {
    it('produces correct file set', () => {
      const { files } = generateSvelteKit({ lang: 'js' });
      const paths = files.map((f) => f.path);
      expect(paths).toContain('package.json');
      expect(paths).toContain('svelte.config.js');
      expect(paths).toContain('vite.config.js');
      expect(paths).toContain('src/app.html');
      expect(paths).toContain('src/app.css');
      expect(paths).toContain('src/routes/+layout.svelte');
      expect(paths).toContain('src/routes/+page.svelte');
      expect(paths).toContain('README.md');
    });
  });

  describe('file structure (TS)', () => {
    it('uses .ts extensions for lib modules', () => {
      const { files } = generateSvelteKit({ lang: 'ts' });
      const paths = files.map((f) => f.path);
      expect(paths).toContain('src/lib/preprocess.ts');
      expect(paths).toContain('src/lib/inference.ts');
      expect(paths).toContain('src/lib/postprocess.ts');
    });
  });

  describe('package.json', () => {
    it('includes svelte and @sveltejs/kit', () => {
      const { files } = generateSvelteKit();
      const pkg = JSON.parse(getFile(files, 'package.json'));
      expect(pkg.devDependencies.svelte || pkg.dependencies.svelte).toBeDefined();
      expect(pkg.devDependencies['@sveltejs/kit'] || pkg.dependencies['@sveltejs/kit']).toBeDefined();
    });

    it('has dev and build scripts', () => {
      const { files } = generateSvelteKit();
      const pkg = JSON.parse(getFile(files, 'package.json'));
      expect(pkg.scripts.dev).toContain('vite dev');
      expect(pkg.scripts.build).toContain('vite build');
    });
  });

  describe('page component', () => {
    it('imports from $lib modules', () => {
      const { files } = generateSvelteKit();
      const page = getFile(files, 'src/routes/+page.svelte');
      expect(page).toContain('$lib/');
    });

    it('includes script and markup sections', () => {
      const { files } = generateSvelteKit();
      const page = getFile(files, 'src/routes/+page.svelte');
      expect(page).toContain('<script');
      expect(page).toContain('</script>');
    });
  });

  describe('lib modules', () => {
    it('inference lib has ort import', () => {
      const { files } = generateSvelteKit();
      const code = getFile(files, 'src/lib/inference.js');
      expect(code).toContain("import * as ort from 'onnxruntime-web'");
      expect(code).toContain('export async function createSession');
    });
  });
});

// ---- Online model support (MODEL_PATH = URL) ----

describe('online model support', () => {
  const urlOverrides = {
    modelSource: 'url' as const,
    modelUrl: 'https://huggingface.co/user/repo/resolve/main/mobilenet.onnx',
    modelName: 'mobilenet',
  };

  describe('html framework uses URL as MODEL_PATH', () => {
    it('sets MODEL_PATH to URL instead of local path', () => {
      const { files } = generateHtml(urlOverrides);
      const html = getFile(files, 'index.html');
      expect(html).toContain("const MODEL_PATH = 'https://huggingface.co/user/repo/resolve/main/mobilenet.onnx'");
    });

    it('does NOT contain local .onnx path reference', () => {
      const { files } = generateHtml(urlOverrides);
      const html = getFile(files, 'index.html');
      expect(html).not.toContain("'./mobilenet.onnx'");
    });
  });

  describe('react-vite framework uses URL as MODEL_PATH', () => {
    it('sets MODEL_PATH to URL in App component', () => {
      const { files } = generateReactVite(urlOverrides);
      const app = getFile(files, 'src/App.jsx');
      expect(app).toContain("const MODEL_PATH = 'https://huggingface.co/user/repo/resolve/main/mobilenet.onnx'");
    });
  });

  describe('vanilla-vite framework uses URL as MODEL_PATH', () => {
    it('sets MODEL_PATH to URL in main module', () => {
      const { files } = generateVanillaVite(urlOverrides);
      const main = getFile(files, 'src/main.js');
      expect(main).toContain("const MODEL_PATH = 'https://huggingface.co/user/repo/resolve/main/mobilenet.onnx'");
    });
  });

  describe('nextjs framework uses URL as MODEL_PATH', () => {
    it('sets MODEL_PATH to URL in page component', () => {
      const { files } = generateNextjs(urlOverrides);
      const page = getFile(files, 'app/page.jsx');
      expect(page).toContain("const MODEL_PATH = 'https://huggingface.co/user/repo/resolve/main/mobilenet.onnx'");
    });
  });

  describe('sveltekit framework uses URL as MODEL_PATH', () => {
    it('sets MODEL_PATH to URL in page.svelte', () => {
      const { files } = generateSvelteKit(urlOverrides);
      const page = getFile(files, 'src/routes/+page.svelte');
      expect(page).toContain("const MODEL_PATH = 'https://huggingface.co/user/repo/resolve/main/mobilenet.onnx'");
    });
  });

  describe('local model still uses relative path', () => {
    it('html uses relative path for local model', () => {
      const { files } = generateHtml({ modelSource: 'local-path' });
      const html = getFile(files, 'index.html');
      expect(html).toContain("const MODEL_PATH = './mobilenet.onnx'");
    });

    it('react-vite uses relative path for local model', () => {
      const { files } = generateReactVite({ modelSource: 'local-path' });
      const app = getFile(files, 'src/App.jsx');
      expect(app).toContain("const MODEL_PATH = '/mobilenet.onnx'");
    });
  });

  describe('README adapts to online model', () => {
    it('README mentions URL loads automatically for url source', () => {
      const { files } = generateHtml(urlOverrides);
      const readme = getFile(files, 'README.md');
      expect(readme).toContain('loads automatically from the URL');
    });

    it('README does NOT say "copy your model" for url source', () => {
      const { files } = generateHtml(urlOverrides);
      const readme = getFile(files, 'README.md');
      expect(readme).not.toContain('Copy your model file');
    });

    it('README says "copy your model" for local source', () => {
      const { files } = generateHtml({ modelSource: 'local-path' });
      const readme = getFile(files, 'README.md');
      expect(readme).toContain('Copy your model file');
    });
  });

  describe('hf-model-id source also uses URL', () => {
    it('hf-model-id with modelUrl uses URL as MODEL_PATH', () => {
      const { files } = generateHtml({
        modelSource: 'hf-model-id',
        modelUrl: 'https://huggingface.co/microsoft/resnet-50/resolve/main/model.onnx',
        modelName: 'model',
      });
      const html = getFile(files, 'index.html');
      expect(html).toContain("const MODEL_PATH = 'https://huggingface.co/microsoft/resnet-50/resolve/main/model.onnx'");
    });
  });
});
