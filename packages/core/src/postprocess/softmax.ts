/**
 * Softmax: convert raw logits to probabilities.
 */

/**
 * Compute softmax over an array of logits.
 * Uses the max-subtraction trick for numerical stability.
 *
 * @param logits - Raw model output scores
 * @returns Float32Array of probabilities summing to 1.0
 */
export function softmax(logits: ArrayLike<number>): Float32Array {
  const len = logits.length;
  const out = new Float32Array(len);

  // Find max for numerical stability
  let max = -Infinity;
  for (let i = 0; i < len; i++) {
    if (logits[i] > max) max = logits[i];
  }

  // exp(x - max) and sum
  let sum = 0;
  for (let i = 0; i < len; i++) {
    out[i] = Math.exp(logits[i] - max);
    sum += out[i];
  }

  // Normalize
  for (let i = 0; i < len; i++) {
    out[i] /= sum;
  }

  return out;
}
