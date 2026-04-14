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
    modelSource: 'local-path',
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

// ---- LiteRT engine snapshot ----

const liteRTMeta: ModelMetadata = {
  format: 'tflite',
  inputs: [{ name: 'input', dataType: 'float32', shape: [1, 224, 224, 3] }],
  outputs: [{ name: 'output', dataType: 'float32', shape: [1, 1000] }],
};

describe('snapshot — classification + LiteRT + html', () => {
  it('matches snapshot', () => {
    const output = serializeFiles(makeConfig({
      engine: 'litert',
      framework: 'html',
      lang: 'js',
      modelMeta: liteRTMeta,
      modelPath: './mobilenet.tflite',
      modelName: 'mobilenet-tflite',
    }));
    expect(output).toMatchSnapshot();
  });
});

// ---- WebNN engine snapshot ----

describe('snapshot — classification + WebNN + html', () => {
  it('matches snapshot', () => {
    const output = serializeFiles(makeConfig({
      engine: 'webnn',
      backend: 'webnn-npu',
      framework: 'html',
      lang: 'js',
    }));
    expect(output).toMatchSnapshot();
  });
});

// ---- Offline mode snapshot ----

describe('snapshot — classification + ORT + html + offline', () => {
  it('matches snapshot', () => {
    const output = serializeFiles(makeConfig({
      framework: 'html',
      lang: 'js',
      offline: true,
    }));
    expect(output).toMatchSnapshot();
  });
});

// ---- Object Detection task snapshot ----

const detectionMeta: ModelMetadata = {
  format: 'onnx',
  inputs: [{ name: 'images', dataType: 'float32', shape: [1, 3, 640, 640] }],
  outputs: [{ name: 'output0', dataType: 'float32', shape: [1, 84, 8400] }],
};

describe('snapshot — object-detection + ORT + html', () => {
  it('matches snapshot', () => {
    const output = serializeFiles(makeConfig({
      task: 'object-detection',
      framework: 'html',
      lang: 'js',
      modelMeta: detectionMeta,
      modelPath: './yolov8n.onnx',
      modelName: 'yolov8n',
      preprocess: { imageSize: 640, mean: [0, 0, 0], std: [1, 1, 1], layout: 'nchw' },
    }));
    expect(output).toMatchSnapshot();
  });
});

// ---- Image Segmentation task snapshot ----

const segmentationMeta: ModelMetadata = {
  format: 'onnx',
  inputs: [{ name: 'input', dataType: 'float32', shape: [1, 3, 512, 512] }],
  outputs: [{ name: 'output', dataType: 'float32', shape: [1, 21, 512, 512] }],
};

describe('snapshot — image-segmentation + ORT + html', () => {
  it('matches snapshot', () => {
    const output = serializeFiles(makeConfig({
      task: 'image-segmentation',
      framework: 'html',
      lang: 'js',
      modelMeta: segmentationMeta,
      modelPath: './deeplabv3.onnx',
      modelName: 'deeplabv3',
      preprocess: { imageSize: 512, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], layout: 'nchw' },
    }));
    expect(output).toMatchSnapshot();
  });
});

// ---- Feature Extraction task snapshot ----

const extractionMeta: ModelMetadata = {
  format: 'onnx',
  inputs: [{ name: 'input', dataType: 'float32', shape: [1, 3, 224, 224] }],
  outputs: [{ name: 'output', dataType: 'float32', shape: [1, 768] }],
};

describe('snapshot — feature-extraction + ORT + html', () => {
  it('matches snapshot', () => {
    const output = serializeFiles(makeConfig({
      task: 'feature-extraction',
      framework: 'html',
      lang: 'js',
      modelMeta: extractionMeta,
      modelPath: './clip-vit.onnx',
      modelName: 'clip-vit',
    }));
    expect(output).toMatchSnapshot();
  });
});

// ---- Offline + React-Vite snapshot (multi-file framework) ----

describe('snapshot — classification + ORT + react-vite + offline', () => {
  it('matches snapshot', () => {
    const output = serializeFiles(makeConfig({
      framework: 'react-vite',
      lang: 'js',
      offline: true,
    }));
    expect(output).toMatchSnapshot();
  });
});

// ---- LiteRT + React-Vite snapshot (multi-file framework + alt engine) ----

describe('snapshot — classification + LiteRT + react-vite', () => {
  it('matches snapshot', () => {
    const output = serializeFiles(makeConfig({
      engine: 'litert',
      framework: 'react-vite',
      lang: 'js',
      modelMeta: liteRTMeta,
      modelPath: './mobilenet.tflite',
      modelName: 'mobilenet-tflite',
    }));
    expect(output).toMatchSnapshot();
  });
});
