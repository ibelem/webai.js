import { describe, it, expect } from 'vitest';
import { resizeImage } from './image-resize.js';

describe('resizeImage', () => {
  // T10: known 4x4 input → expected 2x2 output (exact pixel values)
  it('T10: resizes 4x4 RGB to 2x2 with correct bilinear values', () => {
    // 4x4 image, 3 channels (RGB), values form a gradient
    // Row 0: (0,0,0) (85,85,85) (170,170,170) (255,255,255)
    // Row 1: (64,64,64) (128,128,128) (192,192,192) (255,255,255)
    // Row 2: (128,128,128) (170,170,170) (212,212,212) (255,255,255)
    // Row 3: (192,192,192) (212,212,212) (234,234,234) (255,255,255)
    const src = new Uint8Array([
      0, 0, 0, 85, 85, 85, 170, 170, 170, 255, 255, 255,
      64, 64, 64, 128, 128, 128, 192, 192, 192, 255, 255, 255,
      128, 128, 128, 170, 170, 170, 212, 212, 212, 255, 255, 255,
      192, 192, 192, 212, 212, 212, 234, 234, 234, 255, 255, 255,
    ]);

    const result = resizeImage(src, {
      srcWidth: 4,
      srcHeight: 4,
      dstWidth: 2,
      dstHeight: 2,
      channels: 3,
    });

    expect(result).toBeInstanceOf(Uint8ClampedArray);
    expect(result.length).toBe(2 * 2 * 3);

    // Corner pixels: bilinear with ratio mapping (srcW-1)/(dstW-1) = 3/1 = 3
    // dst(0,0) maps to src(0,0) = (0,0,0)
    expect(result[0]).toBe(0);
    expect(result[1]).toBe(0);
    expect(result[2]).toBe(0);

    // dst(1,0) maps to src(3,0) = (255,255,255)
    expect(result[3]).toBe(255);
    expect(result[4]).toBe(255);
    expect(result[5]).toBe(255);

    // dst(0,1) maps to src(0,3) = (192,192,192)
    expect(result[6]).toBe(192);
    expect(result[7]).toBe(192);
    expect(result[8]).toBe(192);

    // dst(1,1) maps to src(3,3) = (255,255,255)
    expect(result[9]).toBe(255);
    expect(result[10]).toBe(255);
    expect(result[11]).toBe(255);
  });

  // T11: 1x1 image → still works
  it('T11: handles 1x1 input resized to 1x1', () => {
    const src = new Uint8Array([100, 150, 200]);
    const result = resizeImage(src, {
      srcWidth: 1,
      srcHeight: 1,
      dstWidth: 1,
      dstHeight: 1,
      channels: 3,
    });

    expect(result.length).toBe(3);
    expect(result[0]).toBe(100);
    expect(result[1]).toBe(150);
    expect(result[2]).toBe(200);
  });

  it('T11: handles 1x1 input upscaled to 3x3', () => {
    const src = new Uint8Array([42, 84, 126]);
    const result = resizeImage(src, {
      srcWidth: 1,
      srcHeight: 1,
      dstWidth: 3,
      dstHeight: 3,
      channels: 3,
    });

    // All pixels should be the same as the single source pixel
    expect(result.length).toBe(3 * 3 * 3);
    for (let i = 0; i < result.length; i += 3) {
      expect(result[i]).toBe(42);
      expect(result[i + 1]).toBe(84);
      expect(result[i + 2]).toBe(126);
    }
  });

  // T12: non-square input to square output
  it('T12: resizes 6x2 non-square to 3x3 square', () => {
    // 6 wide x 2 tall, RGB
    const src = new Uint8Array(6 * 2 * 3);
    // Fill with a horizontal gradient: column index * 40
    for (let y = 0; y < 2; y++) {
      for (let x = 0; x < 6; x++) {
        const idx = (y * 6 + x) * 3;
        const v = x * 40; // 0, 40, 80, 120, 160, 200
        src[idx] = v;
        src[idx + 1] = v;
        src[idx + 2] = v;
      }
    }

    const result = resizeImage(src, {
      srcWidth: 6,
      srcHeight: 2,
      dstWidth: 3,
      dstHeight: 3,
      channels: 3,
    });

    expect(result.length).toBe(3 * 3 * 3);
    // First pixel (0,0) maps to src (0,0) = 0
    expect(result[0]).toBe(0);
    // Last column (2,*) maps to src x = 5 * (5/2) = but ratio = 5/2 = 2.5
    // dst x=2 → src x = 2.5 * 2 = 5 → pixel value 200
    expect(result[6]).toBe(200);
  });

  it('defaults to 4 channels (RGBA)', () => {
    const src = new Uint8Array([10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255, 100, 110, 120, 255]);
    const result = resizeImage(src, {
      srcWidth: 2,
      srcHeight: 2,
      dstWidth: 1,
      dstHeight: 1,
    });

    expect(result.length).toBe(4);
    // 1x1 maps to src(0,0) (ratio is 0 for 1-pixel output from >1 src)
    // With dstWidth=1: ratio = (2-1)/(1-1||1) = 1/1 = 1... wait
    // xRatio = (srcW-1) / (dstW-1 || 1) = 1/1 = 1 when dstW=1 uses || 1
    // Actually dstW-1 = 0, so (dstW-1 || 1) = 1, ratio = 1/1 = 1
    // srcX = 1 * 0 = 0, so maps to src(0,0)
    expect(result[0]).toBe(10);
    expect(result[3]).toBe(255);
  });

  it('throws on zero target dimensions', () => {
    const src = new Uint8Array([1, 2, 3]);
    expect(() =>
      resizeImage(src, { srcWidth: 1, srcHeight: 1, dstWidth: 0, dstHeight: 1, channels: 3 }),
    ).toThrow(RangeError);
  });
});
