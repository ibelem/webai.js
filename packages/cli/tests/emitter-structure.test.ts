/**
 * Emitter structure tests: verify CodeBlock shape, exports, imports, and content.
 */

import { describe, it, expect } from 'vitest';
import type { ResolvedConfig, ModelMetadata } from '@webai/core';
import { emitLayer1 } from '../src/emitters/index.js';
import { emitInputBlock } from '../src/emitters/input.js';
import { emitPreprocessBlock } from '../src/emitters/preprocess.js';
import { emitPostprocessBlock } from '../src/emitters/postprocess.js';
import { emitOrtInferenceBlock } from '../src/emitters/inference-ort.js';
import { emitLiteRTInferenceBlock } from '../src/emitters/inference-litert.js';
import { emitWebNNInferenceBlock } from '../src/emitters/inference-webnn.js';
import { emitOpfsCacheBlock } from '../src/emitters/opfs-cache.js';

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

  it('exports nms, decodeDetections, postprocessDetections for object-detection', () => {
    const block = emitPostprocessBlock(makeConfig({ task: 'object-detection' }));
    expect(block.exports).toEqual(['nms', 'decodeDetections', 'postprocessDetections']);
  });

  it('emits BoundingBox interface for detection in TS mode', () => {
    const block = emitPostprocessBlock(makeConfig({ task: 'object-detection', lang: 'ts' }));
    expect(block.code).toContain('interface BoundingBox');
    expect(block.code).toContain('classIndex: number');
  });

  it('omits BoundingBox interface for detection in JS mode', () => {
    const block = emitPostprocessBlock(makeConfig({ task: 'object-detection', lang: 'js' }));
    expect(block.code).not.toContain('interface BoundingBox');
  });

  it('detection code contains iou, nms, decodeDetections functions', () => {
    const block = emitPostprocessBlock(makeConfig({ task: 'object-detection' }));
    expect(block.code).toContain('function iou(');
    expect(block.code).toContain('function nms(');
    expect(block.code).toContain('function decodeDetections(');
    expect(block.code).toContain('function postprocessDetections(');
  });

  it('exports argmaxMask, postprocessSegmentation for image-segmentation', () => {
    const block = emitPostprocessBlock(makeConfig({ task: 'image-segmentation' }));
    expect(block.exports).toEqual(['argmaxMask', 'postprocessSegmentation']);
  });

  it('segmentation code contains argmaxMask function', () => {
    const block = emitPostprocessBlock(makeConfig({ task: 'image-segmentation' }));
    expect(block.code).toContain('function argmaxMask(');
    expect(block.code).toContain('Uint8Array');
  });

  it('exports postprocessEmbeddings for feature-extraction', () => {
    const block = emitPostprocessBlock(makeConfig({ task: 'feature-extraction' }));
    expect(block.exports).toEqual(['postprocessEmbeddings']);
  });

  it('feature-extraction code returns Float32Array passthrough', () => {
    const block = emitPostprocessBlock(makeConfig({ task: 'feature-extraction' }));
    expect(block.code).toContain('Float32Array');
    expect(block.code).toContain('function postprocessEmbeddings(');
  });

  it('exports sampleNextToken, postprocessGeneration for text-generation', () => {
    const block = emitPostprocessBlock(makeConfig({ task: 'text-generation' }));
    expect(block.exports).toEqual(['sampleNextToken', 'postprocessGeneration']);
  });

  it('text-generation code contains sampleNextToken and postprocessGeneration functions', () => {
    const block = emitPostprocessBlock(makeConfig({ task: 'text-generation' }));
    expect(block.code).toContain('function sampleNextToken(');
    expect(block.code).toContain('function postprocessGeneration(');
    expect(block.code).toContain('greedy decoding (argmax)');
  });

  it('exports postprocessZeroShot for zero-shot-classification', () => {
    const block = emitPostprocessBlock(makeConfig({ task: 'zero-shot-classification' }));
    expect(block.exports).toEqual(['postprocessZeroShot']);
  });

  it('zero-shot code contains postprocessZeroShot function with softmax', () => {
    const block = emitPostprocessBlock(makeConfig({ task: 'zero-shot-classification' }));
    expect(block.code).toContain('function postprocessZeroShot(');
    expect(block.code).toContain('Softmax over scores');
    expect(block.code).toContain('entailment scores');
  });

  it('uses same exports for audio-classification as image-classification', () => {
    const block = emitPostprocessBlock(makeConfig({ task: 'audio-classification' }));
    expect(block.exports).toEqual(['softmax', 'topK', 'postprocessResults']);
  });

  it('uses same exports for text-classification as image-classification', () => {
    const block = emitPostprocessBlock(makeConfig({ task: 'text-classification' }));
    expect(block.exports).toEqual(['softmax', 'topK', 'postprocessResults']);
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

describe('emitInputBlock', () => {
  it('returns empty block for file input', () => {
    const block = emitInputBlock(makeConfig({ input: 'file' }));
    expect(block.id).toBe('input');
    expect(block.code).toBe('');
    expect(block.exports).toEqual([]);
  });

  it('exports captureFrame, startCamera, stopStream, createInferenceLoop for camera', () => {
    const block = emitInputBlock(makeConfig({ input: 'camera' }));
    expect(block.exports).toEqual(['captureFrame', 'startCamera', 'stopStream', 'createInferenceLoop']);
  });

  it('camera code contains getUserMedia', () => {
    const block = emitInputBlock(makeConfig({ input: 'camera' }));
    expect(block.code).toContain('getUserMedia');
    expect(block.code).toContain('facingMode');
  });

  it('exports captureFrame, stopStream, createInferenceLoop for video', () => {
    const block = emitInputBlock(makeConfig({ input: 'video' }));
    expect(block.exports).toEqual(['captureFrame', 'stopStream', 'createInferenceLoop']);
  });

  it('video does not include startCamera or startScreenCapture', () => {
    const block = emitInputBlock(makeConfig({ input: 'video' }));
    expect(block.code).not.toContain('startCamera');
    expect(block.code).not.toContain('startScreenCapture');
  });

  it('exports captureFrame, startScreenCapture, stopStream, createInferenceLoop for screen', () => {
    const block = emitInputBlock(makeConfig({ input: 'screen' }));
    expect(block.exports).toEqual(['captureFrame', 'startScreenCapture', 'stopStream', 'createInferenceLoop']);
  });

  it('screen code contains getDisplayMedia', () => {
    const block = emitInputBlock(makeConfig({ input: 'screen' }));
    expect(block.code).toContain('getDisplayMedia');
  });

  it('exports startMicrophone, captureAudio, stopStream for mic', () => {
    const block = emitInputBlock(makeConfig({ input: 'mic' }));
    expect(block.exports).toEqual(['startMicrophone', 'captureAudio', 'stopStream']);
  });

  it('mic code contains AudioContext and AnalyserNode', () => {
    const block = emitInputBlock(makeConfig({ input: 'mic' }));
    expect(block.code).toContain('AudioContext');
    expect(block.code).toContain('AnalyserNode');
  });

  it('inference loop contains frame skipping logic', () => {
    const block = emitInputBlock(makeConfig({ input: 'camera' }));
    expect(block.code).toContain('frameSkip');
    expect(block.code).toContain('requestAnimationFrame');
    expect(block.code).toContain('cancelAnimationFrame');
  });

  it('emits TypeScript annotations for camera when lang=ts', () => {
    const block = emitInputBlock(makeConfig({ input: 'camera', lang: 'ts' }));
    expect(block.code).toContain(': HTMLVideoElement');
    expect(block.code).toContain(': HTMLCanvasElement');
    expect(block.code).toContain(': ImageData');
    expect(block.code).toContain(': MediaStream');
  });

  it('omits TypeScript annotations for camera when lang=js', () => {
    const block = emitInputBlock(makeConfig({ input: 'camera', lang: 'js' }));
    expect(block.code).not.toContain(': HTMLVideoElement');
    expect(block.code).not.toContain(': HTMLCanvasElement');
  });

  it('has no imports for any input mode', () => {
    for (const input of ['file', 'camera', 'video', 'screen', 'mic'] as const) {
      const block = emitInputBlock(makeConfig({ input }));
      expect(block.imports).toEqual([]);
    }
  });
});

describe('emitLayer1 with input block', () => {
  it('includes input block for camera input (4 blocks total)', () => {
    const blocks = emitLayer1(makeConfig({ input: 'camera' }));
    expect(blocks).toHaveLength(4);
    expect(blocks.map((b) => b.id)).toEqual(['input', 'preprocess', 'inference', 'postprocess']);
  });

  it('omits input block for file input (3 blocks total)', () => {
    const blocks = emitLayer1(makeConfig({ input: 'file' }));
    expect(blocks).toHaveLength(3);
    expect(blocks.map((b) => b.id)).toEqual(['preprocess', 'inference', 'postprocess']);
  });

  it('includes input block for screen input', () => {
    const blocks = emitLayer1(makeConfig({ input: 'screen' }));
    const inputBlock = blocks.find((b) => b.id === 'input');
    expect(inputBlock).toBeDefined();
    expect(inputBlock?.exports).toContain('startScreenCapture');
  });
});

describe('emitLiteRTInferenceBlock', () => {
  it('exports createSession, runInference, getBackendLabel', () => {
    const block = emitLiteRTInferenceBlock(makeConfig({ engine: 'litert' }));
    expect(block.exports).toEqual(['createSession', 'runInference', 'getBackendLabel']);
  });

  it('uses @anthropic-ai/litert-web import', () => {
    const block = emitLiteRTInferenceBlock(makeConfig({ engine: 'litert' }));
    expect(block.imports).toContain('@anthropic-ai/litert-web');
    expect(block.code).toContain("import * as litert from '@anthropic-ai/litert-web'");
  });

  it('includes GPU delegate when backend=webgpu', () => {
    const block = emitLiteRTInferenceBlock(makeConfig({ engine: 'litert', backend: 'webgpu' }));
    expect(block.code).toContain('createGpuDelegate');
    expect(block.code).toContain('delegates');
  });

  it('omits GPU delegate when backend is not webgpu', () => {
    const block = emitLiteRTInferenceBlock(makeConfig({ engine: 'litert', backend: 'wasm' }));
    expect(block.code).not.toContain('createGpuDelegate');
  });

  it('uses model input shape from metadata', () => {
    const block = emitLiteRTInferenceBlock(makeConfig({ engine: 'litert' }));
    expect(block.code).toContain('[1, 3, 224, 224]');
  });

  it('emits TypeScript annotations when lang=ts', () => {
    const block = emitLiteRTInferenceBlock(makeConfig({ engine: 'litert', lang: 'ts' }));
    expect(block.code).toContain(': litert.TFLiteModel');
    expect(block.code).toContain(': Promise<Float32Array>');
  });

  it('has id=inference', () => {
    const block = emitLiteRTInferenceBlock(makeConfig({ engine: 'litert' }));
    expect(block.id).toBe('inference');
  });
});

describe('emitWebNNInferenceBlock', () => {
  it('exports createSession, runInference, getBackendLabel', () => {
    const block = emitWebNNInferenceBlock(makeConfig({ engine: 'webnn' }));
    expect(block.exports).toEqual(['createSession', 'runInference', 'getBackendLabel']);
  });

  it('has no npm imports (browser API)', () => {
    const block = emitWebNNInferenceBlock(makeConfig({ engine: 'webnn' }));
    expect(block.imports).toEqual([]);
  });

  it('checks for WebNN support', () => {
    const block = emitWebNNInferenceBlock(makeConfig({ engine: 'webnn' }));
    expect(block.code).toContain("'ml' in navigator");
  });

  it('includes device fallback chain', () => {
    const block = emitWebNNInferenceBlock(makeConfig({ engine: 'webnn' }));
    expect(block.code).toContain('devicePrefs');
    expect(block.code).toContain('createContext');
  });

  it('uses npu device for webnn-npu backend', () => {
    const block = emitWebNNInferenceBlock(makeConfig({ engine: 'webnn', backend: 'webnn-npu' }));
    expect(block.code).toContain("'npu'");
  });

  it('uses gpu device for webnn-gpu backend', () => {
    const block = emitWebNNInferenceBlock(makeConfig({ engine: 'webnn', backend: 'webnn-gpu' }));
    expect(block.code).toContain("'gpu'");
  });

  it('uses cpu device for webnn-cpu backend', () => {
    const block = emitWebNNInferenceBlock(makeConfig({ engine: 'webnn', backend: 'webnn-cpu' }));
    expect(block.code).toContain("'cpu'");
  });

  it('includes MLTensor-based inference', () => {
    const block = emitWebNNInferenceBlock(makeConfig({ engine: 'webnn' }));
    expect(block.code).toContain('createTensor');
    expect(block.code).toContain('dispatch');
  });

  it('emits TypeScript annotations when lang=ts', () => {
    const block = emitWebNNInferenceBlock(makeConfig({ engine: 'webnn', lang: 'ts' }));
    expect(block.code).toContain(': MLContext');
    expect(block.code).toContain(': MLGraph');
    expect(block.code).toContain(': Promise<Float32Array>');
  });

  it('getBackendLabel returns WebNN device label', () => {
    const block = emitWebNNInferenceBlock(makeConfig({ engine: 'webnn' }));
    expect(block.code).toContain('WebNN');
    expect(block.code).toContain('toUpperCase');
  });

  it('has id=inference', () => {
    const block = emitWebNNInferenceBlock(makeConfig({ engine: 'webnn' }));
    expect(block.id).toBe('inference');
  });
});

describe('emitOpfsCacheBlock', () => {
  it('returns empty block when offline=false', () => {
    const block = emitOpfsCacheBlock(makeConfig({ offline: false }));
    expect(block.id).toBe('opfs-cache');
    expect(block.code).toBe('');
    expect(block.exports).toEqual([]);
    expect(block.imports).toEqual([]);
  });

  it('exports cachedFetch and clearModelCache when offline=true', () => {
    const block = emitOpfsCacheBlock(makeConfig({ offline: true }));
    expect(block.exports).toEqual(['cachedFetch', 'clearModelCache']);
  });

  it('includes OPFS storage API calls when offline=true', () => {
    const block = emitOpfsCacheBlock(makeConfig({ offline: true }));
    expect(block.code).toContain('navigator.storage.getDirectory');
    expect(block.code).toContain('getDirectoryHandle');
    expect(block.code).toContain('webai-cache');
  });

  it('includes cache hit/miss logging', () => {
    const block = emitOpfsCacheBlock(makeConfig({ offline: true }));
    expect(block.code).toContain('Cache hit');
    expect(block.code).toContain('Cache miss');
  });

  it('includes clearModelCache function', () => {
    const block = emitOpfsCacheBlock(makeConfig({ offline: true }));
    expect(block.code).toContain('function clearModelCache');
    expect(block.code).toContain('removeEntry');
  });

  it('has no npm imports', () => {
    const block = emitOpfsCacheBlock(makeConfig({ offline: true }));
    expect(block.imports).toEqual([]);
  });

  it('emits TypeScript annotations when lang=ts', () => {
    const block = emitOpfsCacheBlock(makeConfig({ offline: true, lang: 'ts' }));
    expect(block.code).toContain(': string');
    expect(block.code).toContain(': Promise<ArrayBuffer>');
    expect(block.code).toContain(': Promise<void>');
  });

  it('omits TypeScript annotations when lang=js', () => {
    const block = emitOpfsCacheBlock(makeConfig({ offline: true, lang: 'js' }));
    expect(block.code).not.toContain(': Promise<ArrayBuffer>');
    expect(block.code).not.toContain(': Promise<void>');
  });
});

describe('emitLayer1 with engine variations', () => {
  it('uses litert inference block when engine=litert', () => {
    const blocks = emitLayer1(makeConfig({ engine: 'litert' }));
    const inference = blocks.find((b) => b.id === 'inference');
    expect(inference).toBeDefined();
    expect(inference?.imports).toContain('@anthropic-ai/litert-web');
  });

  it('uses webnn inference block when engine=webnn', () => {
    const blocks = emitLayer1(makeConfig({ engine: 'webnn' }));
    const inference = blocks.find((b) => b.id === 'inference');
    expect(inference).toBeDefined();
    expect(inference?.imports).toEqual([]);
    expect(inference?.code).toContain("'ml' in navigator");
  });

  it('includes opfs-cache block when offline=true', () => {
    const blocks = emitLayer1(makeConfig({ offline: true }));
    const opfs = blocks.find((b) => b.id === 'opfs-cache');
    expect(opfs).toBeDefined();
    expect(opfs?.exports).toContain('cachedFetch');
  });

  it('omits opfs-cache block when offline=false', () => {
    const blocks = emitLayer1(makeConfig({ offline: false }));
    const opfs = blocks.find((b) => b.id === 'opfs-cache');
    expect(opfs).toBeUndefined();
  });

  it('produces 4 blocks for ort + offline (preprocess, inference, postprocess, opfs)', () => {
    const blocks = emitLayer1(makeConfig({ offline: true }));
    expect(blocks).toHaveLength(4);
    expect(blocks.map((b) => b.id)).toEqual(['preprocess', 'inference', 'postprocess', 'opfs-cache']);
  });
});
