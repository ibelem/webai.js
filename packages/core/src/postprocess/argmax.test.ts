import { describe, it, expect } from 'vitest';
import { argmax } from './argmax.js';

describe('argmax', () => {
  // T18: known array → correct index
  it('T18: returns correct index for simple array', () => {
    expect(argmax([0.1, 0.3, 0.9, 0.2])).toBe(2);
  });

  it('T18: returns first index for ties', () => {
    expect(argmax([5, 5, 5])).toBe(0);
  });

  it('T18: works with negative values', () => {
    expect(argmax([-10, -5, -20, -1])).toBe(3);
  });

  it('returns 0 for single-element array', () => {
    expect(argmax([42])).toBe(0);
  });

  it('returns -1 for empty array', () => {
    expect(argmax([])).toBe(-1);
  });

  it('works with Float32Array', () => {
    const f32 = new Float32Array([0.1, 0.7, 0.3]);
    expect(argmax(f32)).toBe(1);
  });
});
