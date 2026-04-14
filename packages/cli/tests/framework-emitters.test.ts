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

  it('throws for unsupported framework', () => {
    const config = makeConfig({ framework: 'nextjs' as ResolvedConfig['framework'] });
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
