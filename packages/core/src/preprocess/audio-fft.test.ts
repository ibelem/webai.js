import { describe, it, expect } from 'vitest';
import { fft, hannWindow } from './audio-fft.js';

describe('hannWindow', () => {
  it('produces correct length', () => {
    const w = hannWindow(8);
    expect(w.length).toBe(8);
  });

  it('is zero at endpoints', () => {
    const w = hannWindow(8);
    expect(w[0]).toBeCloseTo(0, 10);
    expect(w[7]).toBeCloseTo(0, 10);
  });

  it('peaks at center', () => {
    const w = hannWindow(8);
    // For even length N=8, center indices are 3 and 4
    // Both should have the max value
    const maxVal = Math.max(...Array.from(w));
    expect(w[3]).toBe(maxVal);
    expect(w[4]).toBe(maxVal);
    // Max should be close to 1 but not exactly 1 for even lengths
    expect(maxVal).toBeGreaterThan(0.9);
    expect(maxVal).toBeLessThan(1.0);
  });

  it('is symmetric', () => {
    const w = hannWindow(16);
    for (let i = 0; i < 8; i++) {
      expect(w[i]).toBeCloseTo(w[15 - i], 10);
    }
  });
});

describe('fft', () => {
  it('DC signal: all energy in bin 0', () => {
    const re = new Float64Array([1, 1, 1, 1]);
    const im = new Float64Array(4);
    fft(re, im);
    expect(re[0]).toBeCloseTo(4, 10);
    expect(im[0]).toBeCloseTo(0, 10);
    for (let i = 1; i < 4; i++) {
      expect(Math.abs(re[i])).toBeLessThan(1e-10);
      expect(Math.abs(im[i])).toBeLessThan(1e-10);
    }
  });

  it('pure cosine at bin 1', () => {
    const n = 8;
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      re[i] = Math.cos((2 * Math.PI * i) / n);
    }
    fft(re, im);
    const mag1 = Math.sqrt(re[1] * re[1] + im[1] * im[1]);
    expect(mag1).toBeCloseTo(4, 8);
    const mag7 = Math.sqrt(re[7] * re[7] + im[7] * im[7]);
    expect(mag7).toBeCloseTo(4, 8);
  });

  it('Parseval theorem: energy preserved', () => {
    const n = 16;
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      re[i] = Math.sin(i * 0.7) + Math.cos(i * 1.3);
    }
    const timeDomainEnergy = re.reduce((s, v) => s + v * v, 0);
    fft(re, im);
    const freqDomainEnergy = re.reduce((s, v, i) => s + v * v + im[i] * im[i], 0) / n;
    expect(freqDomainEnergy).toBeCloseTo(timeDomainEnergy, 6);
  });

  it('handles length 1', () => {
    const re = new Float64Array([5]);
    const im = new Float64Array([0]);
    fft(re, im);
    expect(re[0]).toBe(5);
    expect(im[0]).toBe(0);
  });
});
