import { describe, it, expect } from 'vitest';
import { softmax } from './softmax.js';

describe('softmax', () => {
  // T15: known logits → expected probabilities (sum to 1.0 within 1e-6)
  it('T15: produces correct probabilities from known logits', () => {
    const logits = [2.0, 1.0, 0.1];
    const probs = softmax(logits);

    expect(probs).toBeInstanceOf(Float32Array);
    expect(probs.length).toBe(3);

    // Sum should be 1.0
    const sum = probs[0] + probs[1] + probs[2];
    expect(Math.abs(sum - 1.0)).toBeLessThan(1e-6);

    // Highest logit → highest probability
    expect(probs[0]).toBeGreaterThan(probs[1]);
    expect(probs[1]).toBeGreaterThan(probs[2]);

    // Check exact values: softmax([2, 1, 0.1])
    const e2 = Math.exp(2.0 - 2.0); // 1
    const e1 = Math.exp(1.0 - 2.0); // e^-1
    const e01 = Math.exp(0.1 - 2.0); // e^-1.9
    const total = e2 + e1 + e01;
    expect(probs[0]).toBeCloseTo(e2 / total, 5);
    expect(probs[1]).toBeCloseTo(e1 / total, 5);
    expect(probs[2]).toBeCloseTo(e01 / total, 5);
  });

  it('T15: handles uniform logits', () => {
    const probs = softmax([1.0, 1.0, 1.0, 1.0]);
    const sum = Array.from(probs).reduce((a, b) => a + b, 0);
    expect(Math.abs(sum - 1.0)).toBeLessThan(1e-6);

    // All should be equal (~0.25)
    for (const p of probs) {
      expect(p).toBeCloseTo(0.25, 5);
    }
  });

  it('handles large logits without overflow', () => {
    const probs = softmax([1000, 1001, 999]);
    const sum = Array.from(probs).reduce((a, b) => a + b, 0);
    expect(Math.abs(sum - 1.0)).toBeLessThan(1e-6);
    expect(Number.isFinite(probs[0])).toBe(true);
    expect(Number.isFinite(probs[1])).toBe(true);
    expect(Number.isFinite(probs[2])).toBe(true);
  });

  it('handles single element', () => {
    const probs = softmax([42]);
    expect(probs.length).toBe(1);
    expect(probs[0]).toBeCloseTo(1.0, 6);
  });
});
