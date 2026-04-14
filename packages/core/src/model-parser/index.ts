export type { ModelMetadata, TensorInfo, DataType, ModelFormat } from './types.js';
export { detectFormat } from './format-detector.js';
export { parseOnnxMetadata } from './onnx-parser.js';
export { parseTfliteMetadata } from './tflite-parser.js';

import type { ModelMetadata } from './types.js';
import { detectFormat } from './format-detector.js';
import { parseOnnxMetadata } from './onnx-parser.js';
import { parseTfliteMetadata } from './tflite-parser.js';

/**
 * Parse model metadata, auto-detecting format from magic bytes.
 *
 * @param buffer - Raw model file bytes
 * @returns Parsed metadata with format, inputs, and outputs
 * @throws Error if format is unknown or parsing fails
 */
export function parseModelMetadata(buffer: Uint8Array): ModelMetadata {
  const format = detectFormat(buffer);

  switch (format) {
    case 'onnx':
      return parseOnnxMetadata(buffer);
    case 'tflite':
      return parseTfliteMetadata(buffer);
    default:
      throw new Error(
        'Could not parse model: unrecognized format. Expected ONNX (.onnx) or TFLite (.tflite)',
      );
  }
}
