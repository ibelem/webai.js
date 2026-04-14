/**
 * Preprocess emitter: generates standalone preprocessing source code.
 *
 * Mirrors the logic in @webai/core preprocessing functions exactly.
 * Cross-verified by tests T20 and T23: eval(emitted code) === real function output.
 */

import type { ResolvedConfig } from '@webai/core';
import type { CodeBlock } from '../types.js';

/** Emit the resizeImage function as standalone JS/TS code */
function emitResizeImage(ts: boolean): string {
  const typeAnnotations = ts;
  return `/**
 * Resize image using bilinear interpolation.
 * No Canvas or DOM dependency — works in Workers and browsers.
 */
function resizeImage(
  src${typeAnnotations ? ': Uint8Array | Uint8ClampedArray' : ''},
  srcWidth${typeAnnotations ? ': number' : ''},
  srcHeight${typeAnnotations ? ': number' : ''},
  dstWidth${typeAnnotations ? ': number' : ''},
  dstHeight${typeAnnotations ? ': number' : ''},
  channels${typeAnnotations ? ': number' : ''} = 4
)${typeAnnotations ? ': Uint8ClampedArray' : ''} {
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
}`;
}

/** Emit the normalize function as standalone JS/TS code */
function emitNormalize(ts: boolean): string {
  const t = ts;
  return `/**
 * Normalize pixel values: (value/255 - mean) / std per channel.
 */
function normalize(
  src${t ? ': Uint8Array | Uint8ClampedArray' : ''},
  pixelCount${t ? ': number' : ''},
  mean${t ? ': readonly number[]' : ''},
  std${t ? ': readonly number[]' : ''},
  channels${t ? ': number' : ''} = 3,
  srcChannels${t ? ': number' : ''} = channels
)${t ? ': Float32Array' : ''} {
  const out = new Float32Array(pixelCount * channels);

  for (let i = 0; i < pixelCount; i++) {
    const srcBase = i * srcChannels;
    const dstBase = i * channels;
    for (let c = 0; c < channels; c++) {
      out[dstBase + c] = (src[srcBase + c] / 255 - mean[c]) / std[c];
    }
  }

  return out;
}`;
}

/** Emit the toNCHW function as standalone JS/TS code */
function emitToNCHW(ts: boolean): string {
  const t = ts;
  return `/**
 * Transpose image data from HWC to NCHW layout (batch=1).
 */
function toNCHW(
  src${t ? ': Float32Array' : ''},
  width${t ? ': number' : ''},
  height${t ? ': number' : ''},
  channels${t ? ': number' : ''} = 3
)${t ? ': Float32Array' : ''} {
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
}`;
}

/** Emit a convenience preprocessImage function that chains resize + normalize + toNCHW */
function emitPreprocessImage(config: ResolvedConfig, ts: boolean): string {
  const { imageSize, mean, std, layout } = config.preprocess;
  const t = ts;
  const meanStr = `[${mean.join(', ')}]`;
  const stdStr = `[${std.join(', ')}]`;

  const warning = config.preprocessIsDefault
    ? `\n// WARNING: Preprocessing uses task defaults (mean=${meanStr}, std=${stdStr}).
// If results look wrong, check these values against your model's training config.\n`
    : '';

  const layoutStep = layout === 'nchw'
    ? `  return toNCHW(normalized, ${imageSize}, ${imageSize}, 3);`
    : `  return normalized;`;

  return `${warning}/**
 * Full preprocessing pipeline: resize → normalize → ${layout.toUpperCase()} layout.
 * Input: RGBA pixel data from canvas getImageData().
 * Output: Float32Array ready for model input tensor.
 */
function preprocessImage(
  imageData${t ? ': Uint8ClampedArray' : ''},
  srcWidth${t ? ': number' : ''},
  srcHeight${t ? ': number' : ''}
)${t ? ': Float32Array' : ''} {
  const resized = resizeImage(imageData, srcWidth, srcHeight, ${imageSize}, ${imageSize}, 4);

  // Strip alpha channel (RGBA → RGB) during normalization
  const normalized = normalize(resized, ${imageSize} * ${imageSize}, ${meanStr}, ${stdStr}, 3, 4);

${layoutStep}
}`;
}

/**
 * Emit the preprocess CodeBlock for a given config.
 */
export function emitPreprocessBlock(config: ResolvedConfig): CodeBlock {
  const ts = config.lang === 'ts';

  const parts = [
    emitResizeImage(ts),
    emitNormalize(ts),
    emitToNCHW(ts),
    emitPreprocessImage(config, ts),
  ];

  return {
    id: 'preprocess',
    code: parts.join('\n\n'),
    imports: [],
    exports: ['resizeImage', 'normalize', 'toNCHW', 'preprocessImage'],
  };
}
