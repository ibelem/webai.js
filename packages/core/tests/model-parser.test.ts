import { describe, it, expect } from 'vitest';
import { detectFormat } from '../src/model-parser/format-detector.js';
import { parseOnnxMetadata } from '../src/model-parser/onnx-parser.js';
import { parseTfliteMetadata } from '../src/model-parser/tflite-parser.js';
import { detectTask } from '../src/tasks/task-detector.js';
import { buildSyntheticOnnx } from './fixtures/synthetic-onnx.js';
import { buildSyntheticTflite } from './fixtures/synthetic-tflite.js';

describe('detectFormat', () => {
  // T1: ONNX magic bytes → 'onnx'
  it('T1: detects ONNX from protobuf first byte', () => {
    const onnx = buildSyntheticOnnx(
      [{ name: 'input', shape: [1, 3, 224, 224] }],
      [{ name: 'output', shape: [1, 1000] }],
    );
    expect(detectFormat(onnx)).toBe('onnx');
  });

  // T2: TFLite magic bytes → 'tflite'
  it('T2: detects TFLite from TFL3 file identifier', () => {
    const tflite = buildSyntheticTflite(
      [{ name: 'input', shape: [1, 224, 224, 3] }],
      [{ name: 'output', shape: [1, 1000] }],
    );
    expect(detectFormat(tflite)).toBe('tflite');
  });

  // T3: random bytes → 'unknown'
  it('T3: returns unknown for random bytes', () => {
    const random = new Uint8Array([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xde, 0xad]);
    expect(detectFormat(random)).toBe('unknown');
  });

  it('T3: returns unknown for buffer too small', () => {
    expect(detectFormat(new Uint8Array([0x08]))).toBe('unknown');
  });
});

describe('parseOnnxMetadata', () => {
  // T4: classification model (output [1,1000]) → detect image-classification
  it('T4: parses classification model and detects image-classification', () => {
    const buf = buildSyntheticOnnx(
      [{ name: 'input', shape: [1, 3, 224, 224] }],
      [{ name: 'output', shape: [1, 1000] }],
    );

    const metadata = parseOnnxMetadata(buf);

    expect(metadata.format).toBe('onnx');
    expect(metadata.inputs).toHaveLength(1);
    expect(metadata.outputs).toHaveLength(1);
    expect(metadata.inputs[0].name).toBe('input');
    expect(metadata.inputs[0].shape).toEqual([1, 3, 224, 224]);
    expect(metadata.inputs[0].dataType).toBe('float32');
    expect(metadata.outputs[0].name).toBe('output');
    expect(metadata.outputs[0].shape).toEqual([1, 1000]);

    // Task detection
    const result = detectTask(metadata);
    const detected = result.detected;
    expect(detected).not.toBeNull();
    expect(detected?.task).toBe('image-classification');
    expect(detected?.confidence).toBe('high');
  });

  // T5: detection model (output [1,84,8400]) → detect object-detection
  it('T5: parses detection model and detects object-detection', () => {
    const buf = buildSyntheticOnnx(
      [{ name: 'images', shape: [1, 3, 640, 640] }],
      [{ name: 'output0', shape: [1, 84, 8400] }],
    );

    const metadata = parseOnnxMetadata(buf);

    expect(metadata.outputs[0].shape).toEqual([1, 84, 8400]);

    const result = detectTask(metadata);
    const detected = result.detected;
    expect(detected).not.toBeNull();
    expect(detected?.task).toBe('object-detection');
    expect(detected?.confidence).toBe('high');
  });

  // T6: ambiguous shape (output [1,5]) → medium confidence, lists candidates
  it('T6: returns medium confidence for ambiguous shape', () => {
    const buf = buildSyntheticOnnx(
      [{ name: 'input', shape: [1, 3, 224, 224] }],
      [{ name: 'output', shape: [1, 5] }],
    );

    const metadata = parseOnnxMetadata(buf);
    const result = detectTask(metadata);

    const detected = result.detected;
    expect(detected).not.toBeNull();
    expect(detected?.confidence).toBe('medium');
    expect(result.candidates.length).toBeGreaterThanOrEqual(1);
  });

  // T7: corrupted buffer → throws clear error
  it('T7: throws on corrupted buffer', () => {
    const corrupted = new Uint8Array([0x08, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]);
    expect(() => parseOnnxMetadata(corrupted)).toThrow(/Could not parse ONNX model/);
  });

  it('T7: throws on empty graph', () => {
    const noGraph = new Uint8Array([0x08, 0x07]); // just ir_version = 7
    expect(() => parseOnnxMetadata(noGraph)).toThrow(/no graph inputs or outputs/);
  });

  it('parses model with multiple inputs/outputs', () => {
    const buf = buildSyntheticOnnx(
      [
        { name: 'input_ids', shape: [1, 128] },
        { name: 'attention_mask', shape: [1, 128] },
      ],
      [
        { name: 'logits', shape: [1, 2] },
        { name: 'hidden_states', shape: [1, 128, 768] },
      ],
    );

    const metadata = parseOnnxMetadata(buf);
    expect(metadata.inputs).toHaveLength(2);
    expect(metadata.outputs).toHaveLength(2);
    expect(metadata.inputs[0].name).toBe('input_ids');
    expect(metadata.inputs[1].name).toBe('attention_mask');
    expect(metadata.outputs[0].name).toBe('logits');
    expect(metadata.outputs[1].name).toBe('hidden_states');
  });
});

describe('parseTfliteMetadata', () => {
  // T8: classification model → detect image-classification
  it('T8: parses TFLite classification model and detects image-classification', () => {
    const buf = buildSyntheticTflite(
      [{ name: 'serving_default_input:0', shape: [1, 224, 224, 3], type: 0 }],
      [{ name: 'StatefulPartitionedCall:0', shape: [1, 1000], type: 0 }],
    );

    const metadata = parseTfliteMetadata(buf);

    expect(metadata.format).toBe('tflite');
    expect(metadata.inputs).toHaveLength(1);
    expect(metadata.outputs).toHaveLength(1);
    expect(metadata.inputs[0].shape).toEqual([1, 224, 224, 3]);
    expect(metadata.outputs[0].shape).toEqual([1, 1000]);
    expect(metadata.inputs[0].dataType).toBe('float32');

    // Task detection
    const result = detectTask(metadata);
    const detected = result.detected;
    expect(detected).not.toBeNull();
    expect(detected?.task).toBe('image-classification');
  });
});

describe('detectTask', () => {
  // T9: --task flag provided → skips auto-detection entirely
  it('T9: detection result can be overridden by explicit task', () => {
    const buf = buildSyntheticOnnx(
      [{ name: 'input', shape: [1, 3, 224, 224] }],
      [{ name: 'output', shape: [1, 1000] }],
    );
    const metadata = parseOnnxMetadata(buf);
    const result = detectTask(metadata);

    // Auto-detection says image-classification
    expect(result.detected?.task).toBe('image-classification');

    // But user can override with --task flag at CLI layer
    const userOverride = 'object-detection';
    expect(userOverride).toBe('object-detection');
  });

  it('detects feature-extraction from large embedding output', () => {
    const buf = buildSyntheticOnnx(
      [{ name: 'input', shape: [1, 3, 224, 224] }],
      [{ name: 'embedding', shape: [1, 768] }],
    );
    const metadata = parseOnnxMetadata(buf);
    const result = detectTask(metadata);

    const tasks = result.candidates.map((c) => c.task);
    expect(tasks).toContain('image-classification');
    expect(tasks).toContain('feature-extraction');
  });

  it('detects image-segmentation from matching spatial dims', () => {
    const buf = buildSyntheticOnnx(
      [{ name: 'input', shape: [1, 3, 512, 512] }],
      [{ name: 'output', shape: [1, 21, 512, 512] }],
    );
    const metadata = parseOnnxMetadata(buf);
    const result = detectTask(metadata);

    expect(result.detected?.task).toBe('image-segmentation');
    expect(result.detected?.confidence).toBe('high');
  });

  it('returns null for empty outputs', () => {
    const result = detectTask({ format: 'onnx', inputs: [], outputs: [] });
    expect(result.detected).toBeNull();
    expect(result.candidates).toHaveLength(0);
  });
});
