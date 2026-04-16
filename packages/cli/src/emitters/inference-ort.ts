/**
 * ONNX Runtime Web inference emitter (Layer 1).
 *
 * Generates code that:
 * 1. Creates an ORT InferenceSession with backend selected from a runtime <select>
 * 2. Runs inference with proper tensor creation
 * 3. Returns raw output data for postprocessing
 *
 * Available backends: WebNN NPU, WebNN GPU (default), WebNN CPU, WebGPU, Wasm
 * The user switches backends via a <select id="backend"> in the generated page.
 */

import type { ResolvedConfig } from '@webai/core';
import type { CodeBlock } from '../types.js';

/** Generate the execution provider list based on runtime backend selection */
function emitProviders(ts: boolean): string {
  const providerType = ts ? ': (string | { name: string; deviceType?: string })[]' : '';

  return `  // Read backend from the runtime <select> element
  const backendSelect = document.getElementById('backend')${ts ? ' as HTMLSelectElement' : ''};
  const backend = backendSelect ? backendSelect.value : 'webnn-gpu';

  const providers${providerType} = [];
  switch (backend) {
    case 'webnn-npu':
      providers.push({ name: 'webnn', deviceType: 'npu' });
      break;
    case 'webnn-gpu':
      providers.push({ name: 'webnn', deviceType: 'gpu' });
      break;
    case 'webnn-cpu':
      providers.push({ name: 'webnn', deviceType: 'cpu' });
      break;
    case 'webgpu':
      providers.push('webgpu');
      break;
    case 'wasm':
    default:
      providers.push('wasm');
      break;
  }`;
}

/** Emit the createSession function */
function emitCreateSession(config: ResolvedConfig, ts: boolean): string {
  const t = ts;
  const providers = emitProviders(ts);

  // When offline, load model via OPFS cache (cachedFetch is in scope from opfs-cache block)
  const modelLoad = config.offline
    ? `  const cacheKey = modelPath.split('/').pop() || 'model.onnx';
  const modelBuffer = await cachedFetch(modelPath, cacheKey);
  const session = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: providers,
  });`
    : `  const session = await ort.InferenceSession.create(modelPath, {
    executionProviders: providers,
  });`;

  return `/**
 * Create an ONNX Runtime Web inference session with backend selection.
 * The execution provider list determines hardware acceleration priority.
 */
async function createSession(modelPath${t ? ': string' : ''})${t ? ': Promise<ort.InferenceSession>' : ''} {
${providers}

${modelLoad}

  // Log which backend was selected
  console.log('Inference session created');

  return session;
}`;
}

/** Emit the runInference function */
function emitRunInference(config: ResolvedConfig, ts: boolean): string {
  const t = ts;
  const inputShape = config.modelMeta.inputs[0]?.shape ?? [1, 3, 224, 224];
  const shapeStr = `[${inputShape.map((d) => (typeof d === 'string' ? d : String(d))).join(', ')}]`;

  return `/**
 * Run inference on preprocessed input data.
 * Input: Float32Array from preprocessImage()
 * Output: Float32Array of raw model output (logits/scores)
 */
async function runInference(
  session${t ? ': ort.InferenceSession' : ''},
  inputData${t ? ': Float32Array' : ''}
)${t ? ': Promise<Float32Array>' : ''} {
  const tensor = new ort.Tensor('float32', inputData, ${shapeStr});
  const inputName = session.inputNames[0];
  const feeds${t ? ': Record<string, ort.Tensor>' : ''} = { [inputName]: tensor };

  const results = await session.run(feeds);
  const outputName = session.outputNames[0];
  const output = results[outputName];

  return output.data${t ? ' as Float32Array' : ''};
}`;
}

/** Emit the backend status helper for UI display */
function emitBackendStatus(ts: boolean): string {
  const t = ts;
  return `/**
 * Get a display string for the active backend.
 * Used by the status bar: "yolov8n · 8ms · ORT (WebNN GPU)"
 */
function getBackendLabel(session${t ? ': ort.InferenceSession' : ''})${t ? ': string' : ''} {
  void session;
  const backendSelect = document.getElementById('backend')${t ? ' as HTMLSelectElement | null' : ''};
  const backend = backendSelect ? backendSelect.value : 'webnn-gpu';
  const labels = {
    'webnn-npu': 'WebNN NPU',
    'webnn-gpu': 'WebNN GPU',
    'webnn-cpu': 'WebNN CPU',
    'webgpu': 'WebGPU',
    'wasm': 'Wasm',
  };
  return 'ORT (' + (labels[backend] || backend) + ')';
}`;
}

/**
 * Emit the ONNX Runtime Web inference CodeBlock.
 */
export function emitOrtInferenceBlock(config: ResolvedConfig): CodeBlock {
  const ts = config.lang === 'ts';

  const importLine = ts
    ? `import * as ort from 'onnxruntime-web';`
    : `import * as ort from 'onnxruntime-web';`;

  const parts = [
    importLine,
    '',
    emitCreateSession(config, ts),
    emitRunInference(config, ts),
    emitBackendStatus(ts),
  ];

  return {
    id: 'inference',
    code: parts.join('\n\n'),
    imports: ['onnxruntime-web'],
    exports: ['createSession', 'runInference', 'getBackendLabel'],
  };
}
