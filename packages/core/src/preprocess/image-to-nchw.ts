/**
 * Transpose image data from HWC (height, width, channels) to NCHW layout.
 * Most vision models expect NCHW input tensors.
 */

/**
 * Transpose a flat HWC float array to NCHW layout with batch dimension = 1.
 *
 * Input layout:  [h0w0c0, h0w0c1, h0w0c2, h0w1c0, ...] (HWC)
 * Output layout: [c0h0w0, c0h0w1, ..., c1h0w0, ...]     (NCHW, N=1)
 *
 * @param src - Normalized float data in HWC layout, length = height * width * channels
 * @param width - Image width
 * @param height - Image height
 * @param channels - Number of channels (default: 3)
 * @returns Float32Array in NCHW layout, length = 1 * channels * height * width
 */
export function toNCHW(
  src: Float32Array,
  width: number,
  height: number,
  channels = 3,
): Float32Array {
  const out = new Float32Array(channels * height * width);
  const hw = height * width;

  for (let c = 0; c < channels; c++) {
    const channelOffset = c * hw;
    for (let h = 0; h < height; h++) {
      const rowOffset = h * width;
      for (let w = 0; w < width; w++) {
        out[channelOffset + rowOffset + w] = src[(h * width + w) * channels + c];
      }
    }
  }

  return out;
}
