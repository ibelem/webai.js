/**
 * Argmax: find the index of the maximum value.
 */

/**
 * Return the index of the largest element.
 * For ties, returns the first (lowest) index.
 *
 * @param arr - Array of numeric values
 * @returns Index of the maximum value, or -1 for empty arrays
 */
export function argmax(arr: ArrayLike<number>): number {
  if (arr.length === 0) return -1;

  let maxIdx = 0;
  let maxVal = arr[0];

  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > maxVal) {
      maxVal = arr[i];
      maxIdx = i;
    }
  }

  return maxIdx;
}
