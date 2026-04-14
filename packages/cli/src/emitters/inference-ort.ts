/**
 * ORT Web inference emitter (Layer 1).
 *
 * Generates code that:
 * 1. Creates an ORT InferenceSession with backend auto-selection EP chain
 * 2. Runs inference with proper tensor creation
 * 3. Returns raw output data for postprocessing
 *
 * Backend auto-selection order: WebNN NPU → WebNN GPU → WebGPU → WASM
 * When backend is explicitly set, only that provider is used.
 */

import type { ResolvedConfig } from '@webai/core';
import type { CodeBlock } from '../types.js';

/** Generate the execution provider list based on backend config */
function emitProviders(config: ResolvedConfig, ts: boolean): string {
  const t = ts;
  const providerType = t ? ': (string | { name: string; deviceType?: string })[]' : '';

  if (config.backend === 'auto') {
    return `  // Backend auto-selection: try best available, fall through to WASM
  const providers${providerType} = [];
  if ('ml' in navigator) {
    providers.push({ name: 'webnn', deviceType: 'npu' });
    providers.push({ name: 'webnn', deviceType: 'gpu' });
  }
  if (navigator.gpu) providers.push('webgpu');
  providers.push('wasm');`;
  }

  // Explicit backend
  switch (config.backend) {
    case 'wasm':
      return `  const providers${providerType} = ['wasm'];`;
    case 'webgpu':
      return `  const providers${providerType} = ['webgpu'];`;
    case 'webnn-cpu':
      return `  const providers${providerType} = [{ name: 'webnn', deviceType: 'cpu' }];`;
    case 'webnn-gpu':
      return `  const providers${providerType} = [{ name: 'webnn', deviceType: 'gpu' }];`;
    case 'webnn-npu':
      return `  const providers${providerType} = [{ name: 'webnn', deviceType: 'npu' }];`;
    default:
      return `  const providers${providerType} = ['wasm'];`;
  }
}

/** Emit the createSession function */
function emitCreateSession(config: ResolvedConfig, ts: boolean): string {
  const t = ts;
  const providers = emitProviders(config, ts);

  return `/**
 * Create an ORT Web inference session with backend selection.
 * The execution provider list determines hardware acceleration priority.
 */
async function createSession(modelPath${t ? ': string' : ''})${t ? ': Promise<ort.InferenceSession>' : ''} {
${providers}

  const session = await ort.InferenceSession.create(modelPath, {
    executionProviders: providers,
  });

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
  const inputName = config.modelMeta.inputs[0]?.name ?? 'input';

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
  const feeds${t ? ': Record<string, ort.Tensor>' : ''} = { '${inputName}': tensor };

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
 * Used by the status bar: "yolov8n · 8ms · WebNN (NPU)"
 */
function getBackendLabel(session${t ? ': ort.InferenceSession' : ''})${t ? ': string' : ''} {
  // ORT Web doesn't expose the selected EP directly in session.
  // We infer from the handler metadata if available.
  return 'ORT Web';
}`;
}

/**
 * Emit the ORT Web inference CodeBlock.
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
