/**
 * Assembler integration tests (T29-T31).
 *
 * T29: Assembled HTML+JS output parses as valid JS
 * T30: Assembled React-Vite+TS output parses as valid TS
 * T31: Assembled output produces correct file structure
 */

import { describe, it, expect } from 'vitest';
import ts from 'typescript';
import type { ResolvedConfig, ModelMetadata } from '@webai/core';
import type { GeneratedFile } from '../src/types.js';
import { assemble } from '../src/assembler.js';

/** Safe file lookup — fails test if file not found */
function getFile(files: GeneratedFile[], path: string): string {
  const file = files.find((f) => f.path === path);
  expect(file, `Expected file "${path}" to exist`).toBeDefined();
  return file?.content ?? '';
}

const classificationMeta: ModelMetadata = {
  format: 'onnx',
  inputs: [{ name: 'input', dataType: 'float32', shape: [1, 3, 224, 224] }],
  outputs: [{ name: 'output', dataType: 'float32', shape: [1, 1000] }],
};

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

/**
 * Parse source code with TypeScript compiler API.
 * Returns diagnostics (errors) if any.
 */
function parseSource(
  code: string,
  fileName: string,
  isTypeScript: boolean,
): ts.Diagnostic[] {
  const sourceFile = ts.createSourceFile(
    fileName,
    code,
    ts.ScriptTarget.ES2022,
    true,
    isTypeScript ? ts.ScriptKind.TSX : ts.ScriptKind.JSX,
  );

  // Create a minimal program to get syntax diagnostics
  const host = ts.createCompilerHost({});
  const originalGetSourceFile = host.getSourceFile;
  host.getSourceFile = (name, languageVersion) => {
    if (name === fileName) return sourceFile;
    return originalGetSourceFile.call(host, name, languageVersion);
  };

  const program = ts.createProgram(
    [fileName],
    {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.ES2022,
      jsx: ts.JsxEmit.ReactJSX,
      noEmit: true,
      allowJs: true,
      skipLibCheck: true,
      // Only check syntax, not type errors (we don't have dependency types)
      noResolve: true,
    },
    host,
  );

  // Get syntax-only diagnostics (not semantic, since we can't resolve imports)
  return [...program.getSyntacticDiagnostics(sourceFile)];
}

function formatDiagnostics(diagnostics: ts.Diagnostic[]): string {
  return diagnostics
    .map((d) => {
      const msg = ts.flattenDiagnosticMessageText(d.messageText, '\n');
      const line = d.file?.getLineAndCharacterOfPosition(d.start ?? 0);
      return `Line ${(line?.line ?? 0) + 1}: ${msg}`;
    })
    .join('\n');
}

// ---- T29: HTML+JS parses as valid JS ----

describe('T29: assembled HTML+JS is valid JS', () => {
  it('index.html inline script parses without syntax errors', () => {
    const files = assemble(makeConfig({ framework: 'html', lang: 'js' }));
    const content = getFile(files, 'index.html');

    // Extract the inline <script type="module"> content
    const scriptMatch = content.match(/<script type="module">([\s\S]*?)<\/script>/);
    expect(scriptMatch, 'Expected <script type="module"> block in index.html').toBeTruthy();

    const jsCode = scriptMatch?.[1] ?? '';
    const diagnostics = parseSource(jsCode, 'inline.js', false);
    expect(
      diagnostics.length,
      `JS syntax errors in index.html:\n${formatDiagnostics(diagnostics)}`,
    ).toBe(0);
  });
});

// ---- T30: React-Vite+TS parses as valid TS ----

describe('T30: assembled React-Vite+TS is valid TS', () => {
  it('App.tsx parses without syntax errors', () => {
    const files = assemble(makeConfig({ framework: 'react-vite', lang: 'ts' }));
    const content = getFile(files, 'src/App.tsx');

    const diagnostics = parseSource(content, 'App.tsx', true);
    expect(
      diagnostics.length,
      `TS syntax errors in App.tsx:\n${formatDiagnostics(diagnostics)}`,
    ).toBe(0);
  });

  it('preprocess.ts parses without syntax errors', () => {
    const files = assemble(makeConfig({ framework: 'react-vite', lang: 'ts' }));
    const content = getFile(files, 'src/lib/preprocess.ts');

    const diagnostics = parseSource(content, 'preprocess.ts', true);
    expect(
      diagnostics.length,
      `TS syntax errors in preprocess.ts:\n${formatDiagnostics(diagnostics)}`,
    ).toBe(0);
  });

  it('inference.ts parses without syntax errors', () => {
    const files = assemble(makeConfig({ framework: 'react-vite', lang: 'ts' }));
    const content = getFile(files, 'src/lib/inference.ts');

    const diagnostics = parseSource(content, 'inference.ts', true);
    expect(
      diagnostics.length,
      `TS syntax errors in inference.ts:\n${formatDiagnostics(diagnostics)}`,
    ).toBe(0);
  });

  it('postprocess.ts parses without syntax errors', () => {
    const files = assemble(makeConfig({ framework: 'react-vite', lang: 'ts' }));
    const content = getFile(files, 'src/lib/postprocess.ts');

    const diagnostics = parseSource(content, 'postprocess.ts', true);
    expect(
      diagnostics.length,
      `TS syntax errors in postprocess.ts:\n${formatDiagnostics(diagnostics)}`,
    ).toBe(0);
  });

  it('main.tsx parses without syntax errors', () => {
    const files = assemble(makeConfig({ framework: 'react-vite', lang: 'ts' }));
    const content = getFile(files, 'src/main.tsx');

    const diagnostics = parseSource(content, 'main.tsx', true);
    expect(
      diagnostics.length,
      `TS syntax errors in main.tsx:\n${formatDiagnostics(diagnostics)}`,
    ).toBe(0);
  });
});

// ---- T31: correct file structure ----

describe('T31: assembled output has correct file structure', () => {
  it('html framework produces index.html + README.md', () => {
    const files = assemble(makeConfig({ framework: 'html' }));
    const paths = files.map((f) => f.path).sort();
    expect(paths).toEqual(['README.md', 'index.html']);
  });

  it('react-vite JS framework produces 10 files', () => {
    const files = assemble(makeConfig({ framework: 'react-vite', lang: 'js' }));
    const paths = files.map((f) => f.path).sort();
    expect(paths).toEqual([
      'README.md',
      'index.html',
      'package.json',
      'src/App.css',
      'src/App.jsx',
      'src/lib/inference.js',
      'src/lib/postprocess.js',
      'src/lib/preprocess.js',
      'src/main.jsx',
      'vite.config.js',
    ]);
  });

  it('react-vite TS framework produces 10 files with .ts/.tsx extensions', () => {
    const files = assemble(makeConfig({ framework: 'react-vite', lang: 'ts' }));
    const paths = files.map((f) => f.path).sort();
    expect(paths).toEqual([
      'README.md',
      'index.html',
      'package.json',
      'src/App.css',
      'src/App.tsx',
      'src/lib/inference.ts',
      'src/lib/postprocess.ts',
      'src/lib/preprocess.ts',
      'src/main.tsx',
      'vite.config.js',
    ]);
  });

  it('all generated files have non-empty content', () => {
    const files = assemble(makeConfig({ framework: 'react-vite' }));
    for (const file of files) {
      expect(file.content.length, `${file.path} should not be empty`).toBeGreaterThan(0);
    }
  });

  it('package.json is valid JSON', () => {
    const files = assemble(makeConfig({ framework: 'react-vite' }));
    const content = getFile(files, 'package.json');
    expect(() => JSON.parse(content)).not.toThrow();
  });
});
