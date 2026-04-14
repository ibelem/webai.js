/**
 * Snapshot tests for generated code (T24-T28).
 *
 * Each test captures the full generated output for a specific config combo.
 * Snapshots detect unintended changes to emitted code.
 *
 * T24: image-classification + ORT Web + html + file input + raw mode (JS)
 * T25: image-classification + ORT Web + react-vite + file input + raw mode (JS)
 * T26: image-classification + ORT Web + html + file input + compact mode (JS)
 * T27: image-classification + ORT Web + html + file input + raw mode (TS)
 * T28: image-classification + ORT Web + html + --backend webnn-npu
 */

import { describe, it, expect } from 'vitest';
import type { ResolvedConfig, ModelMetadata } from '@webai/core';
import { assemble } from '../src/assembler.js';

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
 * Serialize generated files into a stable, diffable format.
 * Each file: === path === followed by content.
 */
function serializeFiles(config: ResolvedConfig): string {
  const files = assemble(config);
  return files
    .map((f) => `=== ${f.path} ===\n${f.content}`)
    .join('\n\n');
}

// ---- T24: classification + ORT + html + file + raw + JS ----

describe('T24: snapshot — html + JS + raw mode', () => {
  it('matches snapshot', () => {
    const output = serializeFiles(makeConfig({
      framework: 'html',
      lang: 'js',
      mode: 'raw',
    }));
    expect(output).toMatchSnapshot();
  });
});

// ---- T25: classification + ORT + react-vite + file + raw + JS ----

describe('T25: snapshot — react-vite + JS + raw mode', () => {
  it('matches snapshot', () => {
    const output = serializeFiles(makeConfig({
      framework: 'react-vite',
      lang: 'js',
      mode: 'raw',
    }));
    expect(output).toMatchSnapshot();
  });
});

// ---- T26: classification + ORT + html + file + compact + JS ----

describe('T26: snapshot — html + JS + compact mode', () => {
  it('matches snapshot', () => {
    const output = serializeFiles(makeConfig({
      framework: 'html',
      lang: 'js',
      mode: 'compact',
    }));
    expect(output).toMatchSnapshot();
  });
});

// ---- T27: classification + ORT + html + file + raw + TS ----

describe('T27: snapshot — html + TS + raw mode', () => {
  it('matches snapshot', () => {
    const output = serializeFiles(makeConfig({
      framework: 'html',
      lang: 'ts',
      mode: 'raw',
    }));
    expect(output).toMatchSnapshot();
  });
});

// ---- T28: classification + ORT + html + backend webnn-npu ----

describe('T28: snapshot — html + forced webnn-npu backend', () => {
  it('matches snapshot', () => {
    const output = serializeFiles(makeConfig({
      framework: 'html',
      lang: 'js',
      mode: 'raw',
      backend: 'webnn-npu',
    }));
    expect(output).toMatchSnapshot();
  });
});
