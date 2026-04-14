/**
 * Layer 1: Inference emitters.
 *
 * Takes a ResolvedConfig, produces CodeBlock[] containing:
 * - Preprocessing code (resize, normalize, layout transpose)
 * - Inference code (engine-specific session + run)
 * - Postprocessing code (task-specific: softmax, topK, etc.)
 *
 * The assembler calls emitLayer1(config) and passes the blocks to Layer 2.
 */

import type { ResolvedConfig } from '@webai/core';
import type { CodeBlock } from '../types.js';
import { emitPreprocessBlock } from './preprocess.js';
import { emitPostprocessBlock } from './postprocess.js';
import { emitOrtInferenceBlock } from './inference-ort.js';
import { emitLiteRTInferenceBlock } from './inference-litert.js';
import { emitWebNNInferenceBlock } from './inference-webnn.js';
import { emitInputBlock } from './input.js';
import { emitOpfsCacheBlock } from './opfs-cache.js';

/**
 * Emit all Layer 1 code blocks for the given config.
 *
 * @param config - Fully resolved config from the resolver
 * @returns Array of CodeBlocks: input, preprocess, inference, postprocess
 */
export function emitLayer1(config: ResolvedConfig): CodeBlock[] {
  const blocks: CodeBlock[] = [];

  // Input capture (mode-specific: camera/video/screen/mic utilities)
  const inputBlock = emitInputBlock(config);
  if (inputBlock.code) {
    blocks.push(inputBlock);
  }

  // Preprocessing (task-specific: image resize + normalize + layout)
  blocks.push(emitPreprocessBlock(config));

  // Inference (engine-specific: session creation + run)
  if (config.engine === 'ort') {
    blocks.push(emitOrtInferenceBlock(config));
  } else if (config.engine === 'litert') {
    blocks.push(emitLiteRTInferenceBlock(config));
  } else if (config.engine === 'webnn') {
    blocks.push(emitWebNNInferenceBlock(config));
  }

  // Postprocessing (task-specific: softmax + topK for classification, nms for detection, etc.)
  blocks.push(emitPostprocessBlock(config));

  // OPFS offline caching (when config.offline is true)
  const opfsBlock = emitOpfsCacheBlock(config);
  if (opfsBlock.code) {
    blocks.push(opfsBlock);
  }

  return blocks;
}

export { emitInputBlock } from './input.js';
export { emitPreprocessBlock } from './preprocess.js';
export { emitPostprocessBlock } from './postprocess.js';
export { emitOrtInferenceBlock } from './inference-ort.js';
export { emitLiteRTInferenceBlock } from './inference-litert.js';
export { emitWebNNInferenceBlock } from './inference-webnn.js';
export { emitOpfsCacheBlock } from './opfs-cache.js';
export { emitAudioPreprocessBlock } from './audio-preprocess.js';
export { emitTextPreprocessBlock } from './text-preprocess.js';
export type { CodeBlock, GeneratedFile } from '../types.js';
