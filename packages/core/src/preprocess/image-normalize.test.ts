import { describe, it, expect } from 'vitest';
import { normalize } from './image-normalize.js';
import { toNCHW } from './image-to-nchw.js';

describe('normalize', () => {
  // T13: known pixel values → expected float32 tensor
  it('T13: normalizes known RGB values with ImageNet mean/std', () => {
    // Single pixel: RGB = (128, 64, 192)
    const src = new Uint8Array([128, 64, 192]);
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    const result = normalize(src, 1, { mean, std });

    expect(result).toBeInstanceOf(Float32Array);
    expect(result.length).toBe(3);

    // R: (128/255 - 0.485) / 0.229
    const expectedR = (128 / 255 - 0.485) / 0.229;
    // G: (64/255 - 0.456) / 0.224
    const expectedG = (64 / 255 - 0.456) / 0.224;
    // B: (192/255 - 0.406) / 0.225
    const expectedB = (192 / 255 - 0.406) / 0.225;

    expect(result[0]).toBeCloseTo(expectedR, 5);
    expect(result[1]).toBeCloseTo(expectedG, 5);
    expect(result[2]).toBeCloseTo(expectedB, 5);
  });

  // T14: all-zero image → normalized correctly
  it('T14: normalizes all-zero image without NaN or Infinity', () => {
    // 2x2 black image, RGB
    const src = new Uint8Array(2 * 2 * 3); // all zeros
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    const result = normalize(src, 4, { mean, std });

    expect(result.length).toBe(12);
    // All values should be finite (negative, since 0/255 - mean < 0)
    for (let i = 0; i < result.length; i++) {
      expect(Number.isFinite(result[i])).toBe(true);
    }

    // Check first pixel R channel: (0 - 0.485) / 0.229
    expect(result[0]).toBeCloseTo(-0.485 / 0.229, 5);
  });

  it('handles RGBA source with 3-channel output', () => {
    // RGBA pixel: (100, 150, 200, 255)
    const src = new Uint8Array([100, 150, 200, 255]);
    const result = normalize(src, 1, {
      mean: [0.5, 0.5, 0.5],
      std: [0.5, 0.5, 0.5],
      channels: 3,
      srcChannels: 4,
    });

    expect(result.length).toBe(3);
    // R: (100/255 - 0.5) / 0.5
    expect(result[0]).toBeCloseTo((100 / 255 - 0.5) / 0.5, 5);
    // Alpha should be skipped
  });
});

describe('normalizeToNCHW (combined)', () => {
  // T13 extended: full pipeline normalize → toNCHW with exact values
  it('T13: normalize + toNCHW produces correct NCHW tensor', () => {
    // 2x2 RGB image
    const src = new Uint8Array([
      255, 0, 0, // red
      0, 255, 0, // green
      0, 0, 255, // blue
      255, 255, 255, // white
    ]);

    const mean = [0.0, 0.0, 0.0];
    const std = [1.0, 1.0, 1.0];

    const normalized = normalize(src, 4, { mean, std });
    const nchw = toNCHW(normalized, 2, 2, 3);

    expect(nchw.length).toBe(3 * 2 * 2); // C*H*W

    // Channel 0 (R): [255/255, 0, 0, 255/255] = [1, 0, 0, 1]
    expect(nchw[0]).toBeCloseTo(1.0, 5);   // (0,0) R
    expect(nchw[1]).toBeCloseTo(0.0, 5);   // (1,0) R
    expect(nchw[2]).toBeCloseTo(0.0, 5);   // (0,1) R
    expect(nchw[3]).toBeCloseTo(1.0, 5);   // (1,1) R

    // Channel 1 (G): [0, 255/255, 0, 255/255] = [0, 1, 0, 1]
    expect(nchw[4]).toBeCloseTo(0.0, 5);
    expect(nchw[5]).toBeCloseTo(1.0, 5);
    expect(nchw[6]).toBeCloseTo(0.0, 5);
    expect(nchw[7]).toBeCloseTo(1.0, 5);

    // Channel 2 (B): [0, 0, 255/255, 255/255] = [0, 0, 1, 1]
    expect(nchw[8]).toBeCloseTo(0.0, 5);
    expect(nchw[9]).toBeCloseTo(0.0, 5);
    expect(nchw[10]).toBeCloseTo(1.0, 5);
    expect(nchw[11]).toBeCloseTo(1.0, 5);
  });
});
