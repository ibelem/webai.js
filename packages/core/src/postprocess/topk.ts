/**
 * Top-K selection: return the K highest-scoring indices and their values.
 */

export interface TopKResult {
  indices: number[];
  values: number[];
}

/**
 * Return the top K elements by value, in descending order.
 * When values tie, the lower index comes first (deterministic).
 * If k > array length, returns all elements.
 *
 * @param arr - Scores or probabilities
 * @param k - Number of top elements to return
 * @returns Indices and values of the top K elements
 */
export function topK(arr: ArrayLike<number>, k: number): TopKResult {
  const len = arr.length;
  const n = Math.min(k, len);

  // Build index array and sort descending by value, then ascending by index for ties
  const indices = Array.from({ length: len }, (_, i) => i);
  indices.sort((a, b) => {
    const diff = arr[b] - arr[a];
    return diff !== 0 ? diff : a - b;
  });

  const topIndices = indices.slice(0, n);
  const topValues = topIndices.map((i) => arr[i]);

  return { indices: topIndices, values: topValues };
}
