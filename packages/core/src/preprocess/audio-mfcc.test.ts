import { describe, it, expect } from 'vitest';
import { mfcc } from './audio-mfcc.js';

describe('mfcc', () => {
  it('output shape is [numFrames, numCoeffs]', () => {
    const numFrames = 10;
    const numMelBands = 40;
    const numCoeffs = 13;
    const melData = new Float32Array(numFrames * numMelBands);
    for (let i = 0; i < melData.length; i++) {
      melData[i] = Math.random() * -5;
    }
    const result = mfcc(melData, numFrames, numMelBands, numCoeffs);
    expect(result.length).toBe(numFrames * numCoeffs);
  });

  it('first coefficient (c0) is proportional to total energy', () => {
    const numFrames = 1;
    const numMelBands = 8;
    const numCoeffs = 4;
    const uniform = new Float32Array(numMelBands).fill(-1);
    const result = mfcc(uniform, numFrames, numMelBands, numCoeffs);
    expect(result[0]).toBeCloseTo(-8, 5);
  });

  it('produces finite values', () => {
    const numFrames = 5;
    const numMelBands = 40;
    const numCoeffs = 13;
    const melData = new Float32Array(numFrames * numMelBands);
    for (let i = 0; i < melData.length; i++) {
      melData[i] = -3 + Math.random();
    }
    const result = mfcc(melData, numFrames, numMelBands, numCoeffs);
    for (let i = 0; i < result.length; i++) {
      expect(Number.isFinite(result[i])).toBe(true);
    }
  });

  it('different inputs produce different coefficients', () => {
    const numMelBands = 16;
    const numCoeffs = 8;
    const a = new Float32Array(numMelBands).fill(-2);
    const b = new Float32Array(numMelBands);
    for (let i = 0; i < numMelBands; i++) {
      b[i] = -2 + (i / numMelBands);
    }
    const ra = mfcc(a, 1, numMelBands, numCoeffs);
    const rb = mfcc(b, 1, numMelBands, numCoeffs);
    let differ = false;
    for (let c = 1; c < numCoeffs; c++) {
      if (Math.abs(ra[c] - rb[c]) > 1e-6) differ = true;
    }
    expect(differ).toBe(true);
  });
});
