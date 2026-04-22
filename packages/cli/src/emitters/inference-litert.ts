/**
 * LiteRT.js inference emitter (Layer 1).
 *
 * Generates code that:
 * 1. Initializes LiteRT.js Wasm runtime
 * 2. Loads and compiles a .tflite model (with WebGPU, WebNN, or Wasm acceleration)
 * 3. Runs inference with Tensor creation
 * 4. Returns raw output data for postprocessing
 *
 * LiteRT.js is Google's on-device ML runtime for web.
 * Package: @litertjs/core
 * API: loadLiteRt() → loadAndCompile() → model.run()
 *
 * The generated code exports the same interface as ORT:
 *   createSession(modelPath) → model
 *   runInference(model, inputData) → Float32Array
 *   getBackendLabel(model) → string
 */

import type { ResolvedConfig } from '@webai/core';
import type { CodeBlock } from '../types.js';

const LITERT_PKG = '@litertjs/core';

/** CDN base URL for loading LiteRT Wasm files in HTML framework */
export const LITERT_CDN = 'https://cdn.jsdelivr.net/npm/@litertjs/core/wasm/';

function emitCreateSession(config: ResolvedConfig, ts: boolean): string {
  const t = ts;

  // When offline, load model via OPFS cache first
  const modelSource = config.offline
    ? `  const cacheKey = modelPath.split('/').pop() || 'model.tflite';
  const modelData = await cachedFetch(modelPath, cacheKey);`
    : '';

  const modelArg = config.offline ? 'new Uint8Array(modelData)' : 'modelPath';

  return `/**
 * Create a LiteRT inference session.
 * Loads and compiles a .tflite model for browser-based inference.
 * Backend is read from the runtime <select id="backend"> element.
 */
async function createSession(modelPath${t ? ': string' : ''})${t ? ': Promise<any>' : ''} {
${modelSource}${modelSource ? '\n' : ''}  // Initialize LiteRT.js Wasm runtime
  await loadLiteRt(LITERT_WASM_PATH);

  // Read backend from the runtime <select> element
  const backendSelect = document.getElementById('backend')${t ? ' as HTMLSelectElement' : ''};
  const backend = backendSelect ? backendSelect.value : 'webgpu';

  // Map backend selector value to LiteRT compile options
  const options${t ? ': Record<string, unknown>' : ''} = {};
  if (backend === 'webgpu') {
    options.accelerator = 'webgpu';
  } else if (backend.startsWith('webnn')) {
    // WebNN with device preference (webnn-cpu, webnn-gpu, webnn-npu)
    const device = backend.split('-')[1] || 'gpu';
    options.accelerator = 'webnn';
    options.webNNOptions = { devicePreference: device };
  }
  // else: no accelerator → defaults to 'wasm'

  const model = await loadAndCompile(${modelArg}, options);

  const label = backend.startsWith('webnn') ? 'WebNN ' + (backend.split('-')[1] || 'gpu').toUpperCase()
    : backend === 'webgpu' ? 'WebGPU' : 'Wasm';
  console.log('LiteRT session created (' + label + ')');
  return model;
}`;
}

function emitRunInference(config: ResolvedConfig, ts: boolean): string {
  const t = ts;
  const inputShape = config.modelMeta.inputs[0]?.shape ?? [1, 3, 224, 224];
  const shapeStr = `[${inputShape.map((d) => (typeof d === 'string' ? d : String(d))).join(', ')}]`;

  return `/**
 * Run inference on preprocessed input data.
 * Input: Float32Array from preprocessImage()
 * Output: Float32Array of raw model output
 */
async function runInference(
  model${t ? ': any' : ''},
  inputData${t ? ': Float32Array' : ''}
)${t ? ': Promise<Float32Array>' : ''} {
  const inputTensor = new Tensor(inputData, ${shapeStr});

  // Read backend to decide if we need GPU tensors
  const backendSelect = document.getElementById('backend')${t ? ' as HTMLSelectElement' : ''};
  const backend = backendSelect ? backendSelect.value : 'webgpu';

  let runInput = inputTensor;
  if (backend === 'webgpu') {
    // WebGPU requires GPU tensor buffers
    runInput = await inputTensor.moveTo('webgpu');
    inputTensor.delete();
  }
  // WebNN and Wasm use host memory tensors (no move needed)

  const results = await model.run(runInput);
  runInput.delete();

  // Move result to CPU to read the data
  const result = results[0];
  const output = new Float32Array(await result.data());
  result.delete();

  return output;
}`;
}

function emitBackendStatus(ts: boolean): string {
  const t = ts;
  return `/**
 * Get a display string for the active backend.
 */
function getBackendLabel(model${t ? ': any' : ''})${t ? ': string' : ''} {
  void model;
  const backendSelect = document.getElementById('backend')${t ? ' as HTMLSelectElement | null' : ''};
  const backend = backendSelect ? backendSelect.value : 'webgpu';
  const label = backend.startsWith('webnn') ? 'WebNN ' + (backend.split('-')[1] || 'gpu').toUpperCase()
    : backend === 'webgpu' ? 'WebGPU' : 'Wasm';
  return 'LiteRT (' + label + ')';
}`;
}

/**
 * Emit the LiteRT.js inference CodeBlock.
 */
export function emitLiteRTInferenceBlock(config: ResolvedConfig): CodeBlock {
  const ts = config.lang === 'ts';

  const importLine = `import { loadLiteRt, loadAndCompile, Tensor } from '${LITERT_PKG}';`;
  const wasmPath = `const LITERT_WASM_PATH = '/node_modules/@litertjs/core/wasm/';`;

  const parts = [
    importLine,
    wasmPath,
    '',
    emitCreateSession(config, ts),
    emitRunInference(config, ts),
    emitBackendStatus(ts),
  ];

  return {
    id: 'inference',
    code: parts.join('\n\n'),
    imports: [LITERT_PKG],
    exports: ['createSession', 'runInference', 'getBackendLabel'],
  };
}
