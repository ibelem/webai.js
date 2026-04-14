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

/**
 * Emit all Layer 1 code blocks for the given config.
 *
 * @param config - Fully resolved config from the resolver
 * @returns Array of CodeBlocks: preprocess, inference, postprocess
 */
export function emitLayer1(config: ResolvedConfig): CodeBlock[] {
  const blocks: CodeBlock[] = [];

  // Preprocessing (task-specific: image resize + normalize + layout)
  blocks.push(emitPreprocessBlock(config));

  // Inference (engine-specific: session creation + run)
  if (config.engine === 'ort') {
    blocks.push(emitOrtInferenceBlock(config));
  }
  // Phase 1b: litert, webnn emitters

  // Postprocessing (task-specific: softmax + topK for classification, nms for detection, etc.)
  blocks.push(emitPostprocessBlock(config));

  return blocks;
}

export { emitPreprocessBlock } from './preprocess.js';
export { emitPostprocessBlock } from './postprocess.js';
export { emitOrtInferenceBlock } from './inference-ort.js';
export type { CodeBlock, GeneratedFile } from '../types.js';
