/**
 * Cross-verification tests (T20-T23):
 * Eval emitted code in a sandbox and compare output to real @webai/core functions.
 *
 * Uses vm.runInNewContext() per Decision #35 for isolated execution.
 */

import { describe, it, expect } from 'vitest';
import * as vm from 'node:vm';
import {
  resizeImage,
  normalize,
  toNCHW,
  softmax,
  topK,
  type ResolvedConfig,
  type ModelMetadata,
} from '@webai/core';
import { emitPreprocessBlock } from '../src/emitters/preprocess.js';
import { emitPostprocessBlock } from '../src/emitters/postprocess.js';

/** Run JS code in a fresh vm context and return the exported function */
function evalInSandbox(code: string, functionName: string): (...args: unknown[]) => unknown {
  const sandbox = {
    Float32Array,
    Uint8ClampedArray,
    Uint8Array,
    Array,
    Math,
    Infinity,
    console,
  };

  // Wrap code to expose the function we want
  const wrappedCode = `${code}\n\n__result__ = ${functionName};`;
  const context = vm.createContext({ ...sandbox, __result__: undefined });
  vm.runInNewContext(wrappedCode, context);
  return context.__result__ as (...args: unknown[]) => unknown;
}

// Shared config for tests
const classificationMeta: ModelMetadata = {
  format: 'onnx',
  inputs: [{ name: 'input', dataType: 'float32', shape: [1, 3, 224, 224] }],
  outputs: [{ name: 'output', dataType: 'float32', shape: [1, 1000] }],
};

const baseConfig: ResolvedConfig = {
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
};

describe('T20: emitted resizeImage matches real resizeImage', () => {
  const block = emitPreprocessBlock(baseConfig);
  const emittedResize = evalInSandbox(block.code, 'resizeImage');

  const testInputs = [
    // 4x4 RGBA → 2x2
    { src: new Uint8ClampedArray(4 * 4 * 4).fill(128), srcW: 4, srcH: 4, dstW: 2, dstH: 2, ch: 4 },
    // 1x1 → 1x1
    { src: new Uint8ClampedArray([255, 0, 128, 255]), srcW: 1, srcH: 1, dstW: 1, dstH: 1, ch: 4 },
    // 3x2 RGB → 2x2
    { src: new Uint8ClampedArray([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]), srcW: 3, srcH: 2, dstW: 2, dstH: 2, ch: 3 },
    // 2x2 RGBA → 4x4 (upscale)
    { src: new Uint8ClampedArray([0, 0, 0, 255, 255, 255, 255, 255, 128, 128, 128, 255, 64, 64, 64, 255]), srcW: 2, srcH: 2, dstW: 4, dstH: 4, ch: 4 },
    // Non-square 4x2 → 3x3
    { src: new Uint8ClampedArray(4 * 2 * 4).map((_, i) => i % 256), srcW: 4, srcH: 2, dstW: 3, dstH: 3, ch: 4 },
  ];

  for (let i = 0; i < testInputs.length; i++) {
    it(`test input ${i + 1}`, () => {
      const { src, srcW, srcH, dstW, dstH, ch } = testInputs[i];

      const real = resizeImage(src, { srcWidth: srcW, srcHeight: srcH, dstWidth: dstW, dstHeight: dstH, channels: ch });
      const emitted = emittedResize(src, srcW, srcH, dstW, dstH, ch) as Uint8ClampedArray;

      expect(emitted.length).toBe(real.length);
      for (let j = 0; j < real.length; j++) {
        expect(emitted[j]).toBe(real[j]);
      }
    });
  }
});

describe('T21: emitted softmax matches real softmax', () => {
  const block = emitPostprocessBlock(baseConfig);
  const emittedSoftmax = evalInSandbox(block.code, 'softmax');

  const testInputs = [
    new Float32Array([1.0, 2.0, 3.0]),
    new Float32Array([0, 0, 0, 0]),
    new Float32Array([100, 100, 100]),
    new Float32Array([-1, -2, -3, -4, -5]),
    new Float32Array([10, 0, -10]),
  ];

  for (let i = 0; i < testInputs.length; i++) {
    it(`test input ${i + 1}`, () => {
      const input = testInputs[i];
      const real = softmax(input);
      const emitted = emittedSoftmax(input) as Float32Array;

      expect(emitted.length).toBe(real.length);
      for (let j = 0; j < real.length; j++) {
        expect(Math.abs(emitted[j] - real[j])).toBeLessThan(1e-6);
      }
    });
  }
});

describe('T22: emitted topK matches real topK', () => {
  const block = emitPostprocessBlock(baseConfig);
  const emittedTopK = evalInSandbox(block.code, 'topK');

  const testInputs: { arr: Float32Array; k: number }[] = [
    { arr: new Float32Array([0.1, 0.7, 0.05, 0.15]), k: 2 },
    { arr: new Float32Array([0.5, 0.5, 0.5]), k: 2 },
    { arr: new Float32Array([0.3, 0.1, 0.6]), k: 5 },
    { arr: new Float32Array([0.9, 0.05, 0.02, 0.01, 0.01, 0.01]), k: 3 },
    { arr: new Float32Array([0.25, 0.25, 0.25, 0.25]), k: 1 },
  ];

  for (let i = 0; i < testInputs.length; i++) {
    it(`test input ${i + 1}`, () => {
      const { arr, k } = testInputs[i];
      const real = topK(arr, k);
      const emitted = emittedTopK(arr, k) as { indices: number[]; values: number[] };

      expect(emitted.indices).toEqual(real.indices);
      for (let j = 0; j < real.values.length; j++) {
        expect(Math.abs(emitted.values[j] - real.values[j])).toBeLessThan(1e-6);
      }
    });
  }
});

describe('T23: full emitted preprocessing chain matches real functions', () => {
  const block = emitPreprocessBlock(baseConfig);
  // Load all preprocessing functions from the emitted code
  const emittedPreprocess = evalInSandbox(block.code, 'preprocessImage');

  it('4x4 RGBA input produces identical output', () => {
    // Create 4x4 RGBA test image
    const w = 4;
    const h = 4;
    const src = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < src.length; i++) {
      src[i] = (i * 37 + 13) % 256; // Deterministic pseudo-random pixels
    }

    // Real chain: resize → normalize → toNCHW
    const targetSize = 224;
    const resized = resizeImage(src, { srcWidth: w, srcHeight: h, dstWidth: targetSize, dstHeight: targetSize, channels: 4 });
    const normalized = normalize(resized, targetSize * targetSize, {
      mean: [0.485, 0.456, 0.406],
      std: [0.229, 0.224, 0.225],
      channels: 3,
      srcChannels: 4,
    });
    const real = toNCHW(normalized, targetSize, targetSize, 3);

    // Emitted chain: preprocessImage(imageData, srcWidth, srcHeight)
    const emitted = emittedPreprocess(src, w, h) as Float32Array;

    expect(emitted.length).toBe(real.length);
    for (let j = 0; j < real.length; j++) {
      expect(Math.abs(emitted[j] - real[j])).toBeLessThan(1e-6);
    }
  });

  it('1x1 RGBA input produces identical output', () => {
    const src = new Uint8ClampedArray([200, 100, 50, 255]);

    const targetSize = 224;
    const resized = resizeImage(src, { srcWidth: 1, srcHeight: 1, dstWidth: targetSize, dstHeight: targetSize, channels: 4 });
    const normalized = normalize(resized, targetSize * targetSize, {
      mean: [0.485, 0.456, 0.406],
      std: [0.229, 0.224, 0.225],
      channels: 3,
      srcChannels: 4,
    });
    const real = toNCHW(normalized, targetSize, targetSize, 3);

    const emitted = emittedPreprocess(src, 1, 1) as Float32Array;

    expect(emitted.length).toBe(real.length);
    for (let j = 0; j < real.length; j++) {
      expect(Math.abs(emitted[j] - real[j])).toBeLessThan(1e-6);
    }
  });

  it('non-square 6x3 RGBA input produces identical output', () => {
    const w = 6;
    const h = 3;
    const src = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < src.length; i++) {
      src[i] = (i * 53 + 7) % 256;
    }

    const targetSize = 224;
    const resized = resizeImage(src, { srcWidth: w, srcHeight: h, dstWidth: targetSize, dstHeight: targetSize, channels: 4 });
    const normalized = normalize(resized, targetSize * targetSize, {
      mean: [0.485, 0.456, 0.406],
      std: [0.229, 0.224, 0.225],
      channels: 3,
      srcChannels: 4,
    });
    const real = toNCHW(normalized, targetSize, targetSize, 3);

    const emitted = emittedPreprocess(src, w, h) as Float32Array;

    expect(emitted.length).toBe(real.length);
    for (let j = 0; j < real.length; j++) {
      expect(Math.abs(emitted[j] - real[j])).toBeLessThan(1e-6);
    }
  });
});
