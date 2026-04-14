/**
 * Emitter structure tests: verify CodeBlock shape, exports, imports, and content.
 */

import { describe, it, expect } from 'vitest';
import type { ResolvedConfig, ModelMetadata } from '@webai/core';
import { emitLayer1 } from '../src/emitters/index.js';
import { emitPreprocessBlock } from '../src/emitters/preprocess.js';
import { emitPostprocessBlock } from '../src/emitters/postprocess.js';
import { emitOrtInferenceBlock } from '../src/emitters/inference-ort.js';

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

describe('emitLayer1', () => {
  it('returns 3 blocks for image-classification + ort', () => {
    const blocks = emitLayer1(makeConfig());
    expect(blocks).toHaveLength(3);
    expect(blocks.map((b) => b.id)).toEqual(['preprocess', 'inference', 'postprocess']);
  });

  it('inference block has onnxruntime-web import', () => {
    const blocks = emitLayer1(makeConfig());
    const inference = blocks.find((b) => b.id === 'inference');
    expect(inference?.imports).toContain('onnxruntime-web');
  });

  it('preprocess and postprocess blocks have no imports', () => {
    const blocks = emitLayer1(makeConfig());
    const pre = blocks.find((b) => b.id === 'preprocess');
    const post = blocks.find((b) => b.id === 'postprocess');
    expect(pre?.imports).toHaveLength(0);
    expect(post?.imports).toHaveLength(0);
  });
});

describe('emitPreprocessBlock', () => {
  it('exports resizeImage, normalize, toNCHW, preprocessImage', () => {
    const block = emitPreprocessBlock(makeConfig());
    expect(block.exports).toEqual(['resizeImage', 'normalize', 'toNCHW', 'preprocessImage']);
  });

  it('includes preprocessing warning when preprocessIsDefault', () => {
    const block = emitPreprocessBlock(makeConfig({ preprocessIsDefault: true }));
    expect(block.code).toContain('WARNING: Preprocessing uses task defaults');
  });

  it('omits warning when preprocessIsDefault is false', () => {
    const block = emitPreprocessBlock(makeConfig({ preprocessIsDefault: false }));
    expect(block.code).not.toContain('WARNING');
  });

  it('uses correct imageSize from config', () => {
    const block = emitPreprocessBlock(makeConfig({
      preprocess: { imageSize: 640, mean: [0, 0, 0], std: [1, 1, 1], layout: 'nchw' },
    }));
    expect(block.code).toContain('640');
  });

  it('emits TypeScript annotations when lang=ts', () => {
    const block = emitPreprocessBlock(makeConfig({ lang: 'ts' }));
    expect(block.code).toContain(': Float32Array');
    expect(block.code).toContain(': number');
  });

  it('omits TypeScript annotations when lang=js', () => {
    const block = emitPreprocessBlock(makeConfig({ lang: 'js' }));
    // Check for return type annotations (function signatures, not docstrings)
    expect(block.code).not.toContain('): Float32Array');
    expect(block.code).not.toContain('): Uint8ClampedArray');
    expect(block.code).not.toMatch(/\w+: number/);
  });
});

describe('emitPostprocessBlock', () => {
  it('exports softmax, topK, postprocessResults for classification', () => {
    const block = emitPostprocessBlock(makeConfig());
    expect(block.exports).toEqual(['softmax', 'topK', 'postprocessResults']);
  });

  it('emits TypeScript annotations when lang=ts', () => {
    const block = emitPostprocessBlock(makeConfig({ lang: 'ts' }));
    expect(block.code).toContain(': Float32Array');
    expect(block.code).toContain(': ArrayLike<number>');
  });
});

describe('emitOrtInferenceBlock', () => {
  it('exports createSession, runInference, getBackendLabel', () => {
    const block = emitOrtInferenceBlock(makeConfig());
    expect(block.exports).toEqual(['createSession', 'runInference', 'getBackendLabel']);
  });

  it('uses onnxruntime-web import', () => {
    const block = emitOrtInferenceBlock(makeConfig());
    expect(block.code).toContain("import * as ort from 'onnxruntime-web'");
  });

  it('emits auto-selection EP chain when backend=auto', () => {
    const block = emitOrtInferenceBlock(makeConfig({ backend: 'auto' }));
    expect(block.code).toContain("'ml' in navigator");
    expect(block.code).toContain("deviceType: 'npu'");
    expect(block.code).toContain("deviceType: 'gpu'");
    expect(block.code).toContain('navigator.gpu');
    expect(block.code).toContain("'wasm'");
  });

  it('emits only wasm provider when backend=wasm', () => {
    const block = emitOrtInferenceBlock(makeConfig({ backend: 'wasm' }));
    expect(block.code).toContain("['wasm']");
    expect(block.code).not.toContain("'ml' in navigator");
  });

  it('emits only webgpu provider when backend=webgpu', () => {
    const block = emitOrtInferenceBlock(makeConfig({ backend: 'webgpu' }));
    expect(block.code).toContain("['webgpu']");
  });

  it('emits webnn-npu provider when backend=webnn-npu', () => {
    const block = emitOrtInferenceBlock(makeConfig({ backend: 'webnn-npu' }));
    expect(block.code).toContain("deviceType: 'npu'");
    expect(block.code).not.toContain("'ml' in navigator");
  });

  it('uses model input name and shape from metadata', () => {
    const block = emitOrtInferenceBlock(makeConfig());
    expect(block.code).toContain("'input'");
    expect(block.code).toContain('[1, 3, 224, 224]');
  });

  it('uses custom input name from metadata', () => {
    const customMeta: ModelMetadata = {
      format: 'onnx',
      inputs: [{ name: 'images', dataType: 'float32', shape: [1, 3, 640, 640] }],
      outputs: [{ name: 'output0', dataType: 'float32', shape: [1, 84, 8400] }],
    };
    const block = emitOrtInferenceBlock(makeConfig({ modelMeta: customMeta }));
    expect(block.code).toContain("'images'");
    expect(block.code).toContain('[1, 3, 640, 640]');
  });

  it('emits TypeScript annotations when lang=ts', () => {
    const block = emitOrtInferenceBlock(makeConfig({ lang: 'ts' }));
    expect(block.code).toContain(': ort.InferenceSession');
    expect(block.code).toContain(': Promise<Float32Array>');
  });
});
