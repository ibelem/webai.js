/**
 * Postprocess emitter: generates standalone postprocessing source code.
 *
 * Mirrors the logic in @webai/core postprocessing functions exactly.
 * Cross-verified by tests T21 and T22: eval(emitted code) === real function output.
 */

import type { ResolvedConfig } from '@webai/core';
import type { CodeBlock } from '../types.js';

/** Emit the softmax function as standalone JS/TS code */
function emitSoftmax(ts: boolean): string {
  const t = ts;
  return `/**
 * Softmax: convert raw logits to probabilities.
 * Uses max-subtraction for numerical stability.
 */
function softmax(logits${t ? ': ArrayLike<number>' : ''})${t ? ': Float32Array' : ''} {
  const len = logits.length;
  const out = new Float32Array(len);

  let max = -Infinity;
  for (let i = 0; i < len; i++) {
    if (logits[i] > max) max = logits[i];
  }

  let sum = 0;
  for (let i = 0; i < len; i++) {
    out[i] = Math.exp(logits[i] - max);
    sum += out[i];
  }

  for (let i = 0; i < len; i++) {
    out[i] /= sum;
  }

  return out;
}`;
}

/** Emit the topK function as standalone JS/TS code */
function emitTopK(ts: boolean): string {
  const t = ts;
  const resultType = t ? ': { indices: number[]; values: number[] }' : '';
  return `/**
 * Return the top K elements by value, descending.
 * Ties are broken by lower index first.
 */
function topK(arr${t ? ': ArrayLike<number>' : ''}, k${t ? ': number' : ''})${resultType} {
  const len = arr.length;
  const n = Math.min(k, len);

  const indices = Array.from({ length: len }, (_, i) => i);
  indices.sort((a, b) => {
    const diff = arr[b] - arr[a];
    return diff !== 0 ? diff : a - b;
  });

  const topIndices = indices.slice(0, n);
  const topValues = topIndices.map((i) => arr[i]);

  return { indices: topIndices, values: topValues };
}`;
}

/** Emit a postprocessResults convenience function for image-classification */
function emitPostprocessClassification(ts: boolean): string {
  const t = ts;
  const resultType = t ? ': { indices: number[]; values: number[] }' : '';
  return `/**
 * Postprocess model output: softmax → top-5 results.
 */
function postprocessResults(output${t ? ': ArrayLike<number>' : ''})${resultType} {
  const probs = softmax(output);
  return topK(probs, 5);
}`;
}

/**
 * Emit the postprocess CodeBlock for a given config.
 * Currently supports image-classification (softmax + topK).
 */
export function emitPostprocessBlock(config: ResolvedConfig): CodeBlock {
  const ts = config.lang === 'ts';

  const parts: string[] = [];
  const exports: string[] = [];

  if (config.task === 'image-classification') {
    parts.push(emitSoftmax(ts));
    parts.push(emitTopK(ts));
    parts.push(emitPostprocessClassification(ts));
    exports.push('softmax', 'topK', 'postprocessResults');
  }

  return {
    id: 'postprocess',
    code: parts.join('\n\n'),
    imports: [],
    exports,
  };
}
