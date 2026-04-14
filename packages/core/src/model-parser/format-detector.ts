/**
 * Detect model file format from magic bytes.
 *
 * TFLite: bytes 4-7 are "TFL3" (FlatBuffers file identifier).
 * ONNX: first byte is a protobuf field tag (0x08 = ir_version, 0x12 or 0x3A = graph).
 *       Protobuf is lenient, so ONNX is the fallback for non-TFLite binary files.
 *
 * TFLite is checked first because ONNX protobuf's first byte can collide
 * with FlatBuffers' root table offset value.
 */

import type { ModelFormat } from './types.js';

// "TFL3" as bytes — FlatBuffers file identifier for TFLite
const TFLITE_MAGIC = [0x54, 0x46, 0x4c, 0x33]; // T, F, L, 3

// Common ONNX protobuf first-byte tags
const ONNX_TAGS = new Set([
  0x08, // field 1 (ir_version), wire type 0 (varint)
  0x12, // field 2 (opset_import), wire type 2 (length-delimited)
  0x3a, // field 7 (graph), wire type 2 (length-delimited)
]);

export function detectFormat(buffer: Uint8Array): ModelFormat {
  if (buffer.length < 8) return 'unknown';

  // Check TFLite first: bytes 4-7 must be "TFL3"
  if (
    buffer[4] === TFLITE_MAGIC[0] &&
    buffer[5] === TFLITE_MAGIC[1] &&
    buffer[6] === TFLITE_MAGIC[2] &&
    buffer[7] === TFLITE_MAGIC[3]
  ) {
    return 'tflite';
  }

  // Check for common ONNX protobuf first-byte tags
  if (ONNX_TAGS.has(buffer[0])) {
    return 'onnx';
  }

  return 'unknown';
}
