/**
 * WebNN inference emitter (Layer 1).
 *
 * Generates code that uses the WebNN API directly (MLGraphBuilder)
 * for lowest-level hardware acceleration without ORT or LiteRT wrappers.
 *
 * WebNN provides native OS-level ML acceleration:
 *   - NPU (neural processing unit) via DirectML, CoreML, etc.
 *   - GPU via DirectML, Metal, etc.
 *   - CPU fallback
 *
 * The generated code exports the same interface as ORT/LiteRT:
 *   createSession(modelPath) → session object
 *   runInference(session, inputData) → Float32Array
 *   getBackendLabel(session) → string
 *
 * Note: WebNN graph construction from model files requires a conversion
 * step (model2webnn or similar). This emitter generates the inference
 * runner; the model graph code would be in auxiliaryFiles.
 */

import type { ResolvedConfig } from '@webai/core';
import type { CodeBlock } from '../types.js';

function emitDevicePreference(ts: boolean): string {
  return `  // Read device type from the runtime <select> element
  const backendSelect = document.getElementById('backend')${ts ? ' as HTMLSelectElement' : ''};
  const backendValue = backendSelect ? backendSelect.value : 'webnn-gpu';
  const deviceMap = { 'webnn-npu': 'npu', 'webnn-gpu': 'gpu', 'webnn-cpu': 'cpu' };
  const devicePref = deviceMap[backendValue] || 'gpu';`;
}

function emitCreateSession(config: ResolvedConfig, ts: boolean): string {
  const t = ts;
  const devicePrefCode = emitDevicePreference(ts);
  const inputShape = config.modelMeta.inputs[0]?.shape ?? [1, 3, 224, 224];
  const shapeStr = `[${inputShape.map((d) => (typeof d === 'string' ? d : String(d))).join(', ')}]`;

  const sessionType = t
    ? `: Promise<{ context: MLContext; graph: MLGraph; inputShape: number[]; deviceType: string }>`
    : '';

  return `/**
 * Create a WebNN inference session.
 * Uses MLContext and MLGraphBuilder for native hardware acceleration.
 *
 * The device type is read from the runtime <select id="backend"> element.
 * Available: WebNN GPU (default), WebNN NPU, WebNN CPU.
 */
async function createSession(modelPath${t ? ': string' : ''})${sessionType} {
  if (!('ml' in navigator)) {
    throw new Error('WebNN is not supported in this browser.');
  }

${devicePrefCode}

  // Try preferred device, fall back gracefully
  const devicePrefs = [devicePref, 'gpu', 'cpu'];
  let context${t ? ': MLContext | null' : ''} = null;
  let usedDevice = 'cpu';

  for (const device of devicePrefs) {
    try {
      context = await navigator.ml.createContext({ deviceType: device }${t ? ' as MLContextOptions' : ''});
      usedDevice = device${t ? ' as string' : ''};
      break;
    } catch {
      continue;
    }
  }

  if (!context) {
    throw new Error('Could not create WebNN context on any device.');
  }

  // Fetch and build the model graph
${config.offline
    ? `  const cacheKey = modelPath.split('/').pop() || 'model.bin';
  const modelBuffer = await cachedFetch(modelPath, cacheKey);`
    : `  const response = await fetch(modelPath);
  const modelBuffer = await response.arrayBuffer();`}

  const builder = new MLGraphBuilder(context);

  // Build input descriptor
  const inputShape = ${shapeStr};
  const input = builder.input('input', {
    dataType: 'float32',
    shape: inputShape,
  });

  // Note: In production, the graph operations would be generated from
  // the model structure (via model2webnn or equivalent).
  // This is a minimal placeholder that loads pre-compiled graph data.
  void input;
  void modelBuffer;

  const graph = await builder.build({ output: input });

  console.log('WebNN session created on', usedDevice);
  return { context, graph, inputShape, deviceType: usedDevice };
}`;
}

function emitRunInference(_config: ResolvedConfig, ts: boolean): string {
  const t = ts;
  const sessionType = t ? ': { context: MLContext; graph: MLGraph; inputShape: number[] }' : '';

  return `/**
 * Run inference on preprocessed input data using WebNN.
 * Input: Float32Array from preprocessImage()
 * Output: Float32Array of raw model output
 */
async function runInference(
  session${sessionType},
  inputData${t ? ': Float32Array' : ''}
)${t ? ': Promise<Float32Array>' : ''} {
  const inputTensor = await session.context.createTensor(
    'float32',
    session.inputShape,
  );
  await inputTensor.write(inputData);

  const outputTensor = await session.context.createTensor(
    'float32',
    session.inputShape, // Output shape depends on model; simplified here
  );

  const inputs = { input: inputTensor };
  const outputs = { output: outputTensor };

  await session.context.dispatch(session.graph, inputs, outputs);

  const outputData = new Float32Array(outputTensor.size);
  await outputTensor.read(outputData);

  return outputData;
}`;
}

function emitBackendStatus(ts: boolean): string {
  const t = ts;
  const sessionType = t ? ': { deviceType: string }' : '';
  return `/**
 * Get a display string for the active backend.
 */
function getBackendLabel(session${sessionType})${t ? ': string' : ''} {
  const device = session.deviceType || 'unknown';
  return 'WebNN (' + device.toUpperCase() + ')';
}`;
}

/**
 * Emit the WebNN inference CodeBlock.
 */
export function emitWebNNInferenceBlock(config: ResolvedConfig): CodeBlock {
  const ts = config.lang === 'ts';

  const parts = [
    emitCreateSession(config, ts),
    emitRunInference(config, ts),
    emitBackendStatus(ts),
  ];

  return {
    id: 'inference',
    code: parts.join('\n\n'),
    imports: [], // WebNN is a browser API, no npm package needed
    exports: ['createSession', 'runInference', 'getBackendLabel'],
  };
}
