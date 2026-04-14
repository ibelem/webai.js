/**
 * Bilinear interpolation image resize on flat RGBA/RGB typed arrays.
 * No Canvas or DOM dependency — works in Node, Workers, and browsers.
 */

export interface ResizeOptions {
  /** Source image width in pixels */
  srcWidth: number;
  /** Source image height in pixels */
  srcHeight: number;
  /** Target width in pixels */
  dstWidth: number;
  /** Target height in pixels */
  dstHeight: number;
  /** Channels per pixel (3 = RGB, 4 = RGBA). Default: 4 */
  channels?: number;
}

/**
 * Resize an image using bilinear interpolation.
 *
 * @param src - Flat pixel data in row-major order (HWC layout)
 * @param opts - Source/target dimensions and channel count
 * @returns New Uint8ClampedArray with resized pixel data
 */
export function resizeImage(
  src: Uint8Array | Uint8ClampedArray,
  opts: ResizeOptions,
): Uint8ClampedArray {
  const { srcWidth, srcHeight, dstWidth, dstHeight } = opts;
  const channels = opts.channels ?? 4;

  if (dstWidth <= 0 || dstHeight <= 0) {
    throw new RangeError(`Target dimensions must be positive: ${dstWidth}x${dstHeight}`);
  }

  const dst = new Uint8ClampedArray(dstWidth * dstHeight * channels);

  const xRatio = srcWidth > 1 ? (srcWidth - 1) / (dstWidth - 1 || 1) : 0;
  const yRatio = srcHeight > 1 ? (srcHeight - 1) / (dstHeight - 1 || 1) : 0;

  for (let y = 0; y < dstHeight; y++) {
    const srcY = yRatio * y;
    const yFloor = Math.floor(srcY);
    const yCeil = Math.min(yFloor + 1, srcHeight - 1);
    const yLerp = srcY - yFloor;

    for (let x = 0; x < dstWidth; x++) {
      const srcX = xRatio * x;
      const xFloor = Math.floor(srcX);
      const xCeil = Math.min(xFloor + 1, srcWidth - 1);
      const xLerp = srcX - xFloor;

      const dstIdx = (y * dstWidth + x) * channels;

      // Four neighbors
      const tlIdx = (yFloor * srcWidth + xFloor) * channels;
      const trIdx = (yFloor * srcWidth + xCeil) * channels;
      const blIdx = (yCeil * srcWidth + xFloor) * channels;
      const brIdx = (yCeil * srcWidth + xCeil) * channels;

      for (let c = 0; c < channels; c++) {
        const top = src[tlIdx + c] * (1 - xLerp) + src[trIdx + c] * xLerp;
        const bottom = src[blIdx + c] * (1 - xLerp) + src[brIdx + c] * xLerp;
        dst[dstIdx + c] = Math.round(top * (1 - yLerp) + bottom * yLerp);
      }
    }
  }

  return dst;
}
