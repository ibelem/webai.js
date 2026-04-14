import { describe, it, expect } from 'vitest';
import { topK } from './topk.js';

describe('topK', () => {
  it('returns top 3 from a simple array', () => {
    const result = topK([0.1, 0.9, 0.3, 0.7, 0.5], 3);
    expect(result.indices).toEqual([1, 3, 4]);
    expect(result.values).toEqual([0.9, 0.7, 0.5]);
  });

  // T16: ties → deterministic ordering (lower index first)
  it('T16: breaks ties deterministically by lower index first', () => {
    const result = topK([0.5, 0.8, 0.8, 0.5, 0.8], 3);
    // Three values of 0.8 at indices 1, 2, 4 — should come in index order
    expect(result.indices).toEqual([1, 2, 4]);
    expect(result.values).toEqual([0.8, 0.8, 0.8]);
  });

  it('T16: all-equal values returns indices in order', () => {
    const result = topK([0.5, 0.5, 0.5, 0.5], 2);
    expect(result.indices).toEqual([0, 1]);
  });

  // T17: k > array length → returns all elements
  it('T17: k larger than array returns all elements', () => {
    const result = topK([3, 1, 2], 10);
    expect(result.indices).toEqual([0, 2, 1]);
    expect(result.values).toEqual([3, 2, 1]);
    expect(result.indices.length).toBe(3);
  });

  it('T17: k equals array length returns all', () => {
    const result = topK([10, 20], 2);
    expect(result.indices).toEqual([1, 0]);
    expect(result.values).toEqual([20, 10]);
  });

  it('works with negative values', () => {
    const result = topK([-1, -5, -2, -0.5], 2);
    expect(result.indices).toEqual([3, 0]);
    expect(result.values).toEqual([-0.5, -1]);
  });

  it('handles empty array', () => {
    const result = topK([], 5);
    expect(result.indices).toEqual([]);
    expect(result.values).toEqual([]);
  });
});
