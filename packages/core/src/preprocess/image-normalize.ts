/**
 * Per-channel mean/std normalization for image pixel data.
 * Operates on flat Uint8 pixel arrays, outputs Float32Array.
 */

export interface NormalizeOptions {
  /** Per-channel means, e.g. [0.485, 0.456, 0.406] for ImageNet */
  mean: readonly number[];
  /** Per-channel standard deviations, e.g. [0.229, 0.224, 0.225] for ImageNet */
  std: readonly number[];
  /** Number of color channels (default: 3) */
  channels?: number;
  /** Total channels in source data including alpha (default: same as channels) */
  srcChannels?: number;
}

/**
 * Normalize pixel values from [0, 255] to ((value/255) - mean) / std per channel.
 *
 * @param src - Flat pixel data in HWC layout (e.g. RGBRGBRGB or RGBARGBA)
 * @param pixelCount - Total number of pixels (width * height)
 * @param opts - Normalization parameters
 * @returns Float32Array of normalized values in HWC layout, length = pixelCount * channels
 */
export function normalize(
  src: Uint8Array | Uint8ClampedArray,
  pixelCount: number,
  opts: NormalizeOptions,
): Float32Array {
  const channels = opts.channels ?? 3;
  const srcChannels = opts.srcChannels ?? channels;
  const { mean, std } = opts;

  const out = new Float32Array(pixelCount * channels);

  for (let i = 0; i < pixelCount; i++) {
    const srcBase = i * srcChannels;
    const dstBase = i * channels;
    for (let c = 0; c < channels; c++) {
      out[dstBase + c] = (src[srcBase + c] / 255 - mean[c]) / std[c];
    }
  }

  return out;
}
