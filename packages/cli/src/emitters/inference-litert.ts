/**
 * LiteRT.js inference emitter (Layer 1).
 *
 * Generates code that:
 * 1. Creates a LiteRT (TFLite) inference session
 * 2. Runs inference with proper tensor creation
 * 3. Returns raw output data for postprocessing
 *
 * LiteRT.js is Google's on-device ML runtime for web.
 * Uses @anthropic/anthropic-litert-web npm package (placeholder).
 *
 * The generated code exports the same interface as ORT:
 *   createSession(modelPath) → session
 *   runInference(session, inputData) → Float32Array
 *   getBackendLabel(session) → string
 */

import type { ResolvedConfig } from '@webai/core';
import type { CodeBlock } from '../types.js';

const LITERT_PKG = '@anthropic-ai/litert-web';

function emitCreateSession(config: ResolvedConfig, ts: boolean): string {
  const t = ts;

  // When offline, load model via OPFS cache first
  const modelSource = config.offline
    ? `  const cacheKey = modelPath.split('/').pop() || 'model.tflite';
  const modelData = await cachedFetch(modelPath, cacheKey);`
    : '';

  const modelArg = config.offline ? 'modelData' : 'modelPath';

  const delegateCode = config.backend === 'webgpu'
    ? `  const delegate = await litert.createGpuDelegate();
  const session = await litert.TFLiteModel.load(${modelArg}, { delegates: [delegate] });`
    : `  const session = await litert.TFLiteModel.load(${modelArg});`;

  return `/**
 * Create a LiteRT inference session.
 * Loads a .tflite model for browser-based inference.
 */
async function createSession(modelPath${t ? ': string' : ''})${t ? ': Promise<litert.TFLiteModel>' : ''} {
${modelSource}${modelSource ? '\n' : ''}${delegateCode}

  console.log('LiteRT session created');
  return session;
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
  session${t ? ': litert.TFLiteModel' : ''},
  inputData${t ? ': Float32Array' : ''}
)${t ? ': Promise<Float32Array>' : ''} {
  const inputTensor = {
    data: inputData,
    shape: ${shapeStr},
    dtype: 'float32',
  };

  const results = await session.predict([inputTensor]);
  const output = results[0];

  return output.data${t ? ' as Float32Array' : ''};
}`;
}

function emitBackendStatus(ts: boolean): string {
  const t = ts;
  return `/**
 * Get a display string for the active backend.
 */
function getBackendLabel(session${t ? ': litert.TFLiteModel' : ''})${t ? ': string' : ''} {
  void session;
  return 'LiteRT';
}`;
}

/**
 * Emit the LiteRT.js inference CodeBlock.
 */
export function emitLiteRTInferenceBlock(config: ResolvedConfig): CodeBlock {
  const ts = config.lang === 'ts';

  const importLine = `import * as litert from '${LITERT_PKG}';`;

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
    imports: [LITERT_PKG],
    exports: ['createSession', 'runInference', 'getBackendLabel'],
  };
}
