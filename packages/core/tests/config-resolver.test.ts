import { describe, it, expect } from 'vitest';
import { resolveConfig, ConfigValidationError } from '../src/config/resolver.js';
import type { CliFlags } from '../src/config/types.js';
import type { ModelMetadata } from '../src/model-parser/types.js';

// Helper: classification model metadata
const classificationMeta: ModelMetadata = {
  format: 'onnx',
  inputs: [{ name: 'input', dataType: 'float32', shape: [1, 3, 224, 224] }],
  outputs: [{ name: 'output', dataType: 'float32', shape: [1, 1000] }],
};

// Helper: detection model metadata
const detectionMeta: ModelMetadata = {
  format: 'onnx',
  inputs: [{ name: 'images', dataType: 'float32', shape: [1, 3, 640, 640] }],
  outputs: [{ name: 'output0', dataType: 'float32', shape: [1, 84, 8400] }],
};

// Helper: ambiguous model metadata (5D input → 'unknown' input type → low confidence)
const ambiguousMeta: ModelMetadata = {
  format: 'onnx',
  inputs: [{ name: 'input', dataType: 'float32', shape: [1, 2, 3, 4, 5] }],
  outputs: [{ name: 'output', dataType: 'float32', shape: [1, 15] }],
};

describe('resolveConfig', () => {
  describe('defaults', () => {
    it('resolves all defaults for classification model with just --model', () => {
      const flags: CliFlags = { model: './mobilenet.onnx' };
      const { config, steps } = resolveConfig(flags, classificationMeta);

      expect(config.task).toBe('image-classification');
      expect(config.engine).toBe('ort');
      expect(config.backend).toBe('auto');
      expect(config.framework).toBe('html');
      expect(config.input).toBe('file');
      expect(config.mode).toBe('raw');
      expect(config.lang).toBe('js');
      expect(config.outputDir).toBe('./output/');
      expect(config.offline).toBe(false);
      expect(config.theme).toBe('dark');
      expect(config.verbose).toBe(false);
      expect(config.force).toBe(false);
      expect(config.modelPath).toBe('./mobilenet.onnx');
      expect(config.modelName).toBe('mobilenet');
      expect(config.modelMeta).toBe(classificationMeta);

      // Check resolver trace
      expect(steps.length).toBeGreaterThan(0);
      const taskStep = steps.find((s) => s.field === 'task');
      expect(taskStep?.source).toBe('auto-detect');
    });

    it('uses task default input for detection models (camera)', () => {
      const flags: CliFlags = { model: './yolov8n.onnx' };
      const { config } = resolveConfig(flags, detectionMeta);

      expect(config.task).toBe('object-detection');
      expect(config.input).toBe('camera');
    });

    it('resolves preprocessing from task profile', () => {
      const flags: CliFlags = { model: './mobilenet.onnx' };
      const { config } = resolveConfig(flags, classificationMeta);

      // ImageNet defaults for image-classification
      expect(config.preprocess.imageSize).toBe(224);
      expect(config.preprocess.mean).toEqual([0.485, 0.456, 0.406]);
      expect(config.preprocess.std).toEqual([0.229, 0.224, 0.225]);
      expect(config.preprocess.layout).toBe('nchw');
      expect(config.preprocessIsDefault).toBe(true);
    });

    it('resolves detection preprocessing (non-ImageNet)', () => {
      const flags: CliFlags = { model: './yolo.onnx' };
      const { config } = resolveConfig(flags, detectionMeta);

      expect(config.preprocess.imageSize).toBe(640);
      expect(config.preprocess.mean).toEqual([0, 0, 0]);
      expect(config.preprocess.std).toEqual([1, 1, 1]);
    });
  });

  describe('explicit flags override defaults', () => {
    it('uses explicit --task instead of auto-detection', () => {
      const flags: CliFlags = {
        model: './model.onnx',
        task: 'feature-extraction',
      };
      const { config, steps } = resolveConfig(flags, classificationMeta);

      expect(config.task).toBe('feature-extraction');
      const taskStep = steps.find((s) => s.field === 'task');
      expect(taskStep?.source).toBe('cli');
    });

    it('uses all explicit flags', () => {
      const flags: CliFlags = {
        model: './model.onnx',
        task: 'image-classification',
        engine: 'webnn',
        backend: 'webnn-gpu',
        framework: 'react-vite',
        input: 'camera',
        mode: 'compact',
        lang: 'ts',
        output: './dist/',
        offline: true,
        theme: 'light',
        verbose: true,
        force: true,
      };
      const { config } = resolveConfig(flags, classificationMeta);

      expect(config.engine).toBe('webnn');
      expect(config.backend).toBe('webnn-gpu');
      expect(config.framework).toBe('react-vite');
      expect(config.input).toBe('camera');
      expect(config.mode).toBe('compact');
      expect(config.lang).toBe('ts');
      expect(config.outputDir).toBe('./dist/');
      expect(config.offline).toBe(true);
      expect(config.theme).toBe('light');
      expect(config.verbose).toBe(true);
      expect(config.force).toBe(true);
    });
  });

  describe('backend shorthand for webnn engine', () => {
    it('expands -b npu to webnn-npu when engine is webnn', () => {
      const flags: CliFlags = {
        model: './model.onnx',
        task: 'image-classification',
        engine: 'webnn',
        backend: 'npu',
      };
      const { config } = resolveConfig(flags, classificationMeta);
      expect(config.backend).toBe('webnn-npu');
    });

    it('does not expand shorthand for non-webnn engine', () => {
      const flags: CliFlags = {
        model: './model.onnx',
        task: 'image-classification',
        engine: 'ort',
        backend: 'wasm',
      };
      const { config } = resolveConfig(flags, classificationMeta);
      expect(config.backend).toBe('wasm');
    });
  });

  describe('model name extraction', () => {
    it('extracts name from path with extension', () => {
      const { config } = resolveConfig(
        { model: '/path/to/yolov8n.onnx' },
        detectionMeta,
      );
      expect(config.modelName).toBe('yolov8n');
    });

    it('handles path without extension', () => {
      const { config } = resolveConfig(
        { model: './model', task: 'image-classification' },
        classificationMeta,
      );
      expect(config.modelName).toBe('model');
    });

    it('handles Windows-style paths', () => {
      const { config } = resolveConfig(
        { model: 'C:\\Users\\models\\resnet50.onnx', task: 'image-classification' },
        classificationMeta,
      );
      expect(config.modelName).toBe('resnet50');
    });
  });

  describe('validation errors', () => {
    it('rejects invalid engine value', () => {
      const flags: CliFlags = { model: './m.onnx', task: 'image-classification', engine: 'tensorflow' };
      expect(() => resolveConfig(flags, classificationMeta)).toThrow(ConfigValidationError);
      expect(() => resolveConfig(flags, classificationMeta)).toThrow(/Invalid engine/);
    });

    it('rejects invalid framework value', () => {
      const flags: CliFlags = { model: './m.onnx', task: 'image-classification', framework: 'angular' };
      expect(() => resolveConfig(flags, classificationMeta)).toThrow(/Invalid framework/);
    });

    it('rejects invalid task value', () => {
      const flags: CliFlags = { model: './m.onnx', task: 'video-generation' };
      expect(() => resolveConfig(flags, classificationMeta)).toThrow(/Invalid task/);
    });

    it('rejects mic input for image-classification', () => {
      const flags: CliFlags = { model: './m.onnx', task: 'image-classification', input: 'mic' };
      try {
        resolveConfig(flags, classificationMeta);
        expect.fail('should have thrown');
      } catch (e) {
        expect(e).toBeInstanceOf(ConfigValidationError);
        expect((e as ConfigValidationError).message).toContain('mic input is not supported');
        expect((e as ConfigValidationError).suggestion).toBeDefined();
      }
    });

    it('rejects litert engine for speech-to-text', () => {
      const audioMeta: ModelMetadata = {
        format: 'onnx',
        inputs: [{ name: 'audio', dataType: 'float32', shape: [1, 16000] }],
        outputs: [{ name: 'text', dataType: 'float32', shape: [1, 500] }],
      };
      const flags: CliFlags = { model: './whisper.onnx', task: 'speech-to-text', engine: 'litert' };
      expect(() => resolveConfig(flags, audioMeta)).toThrow(/litert engine is not supported/);
    });
  });

  describe('auto-detection failures', () => {
    it('throws when task cannot be detected and no --task flag', () => {
      const emptyMeta: ModelMetadata = {
        format: 'onnx',
        inputs: [],
        outputs: [],
      };
      const flags: CliFlags = { model: './mystery.onnx' };
      expect(() => resolveConfig(flags, emptyMeta)).toThrow(/Could not detect task/);
    });

    it('throws for low-confidence detection without --task', () => {
      const flags: CliFlags = { model: './tiny.onnx' };
      expect(() => resolveConfig(flags, ambiguousMeta)).toThrow(ConfigValidationError);
    });
  });

  describe('resolver trace', () => {
    it('records source for each resolved field', () => {
      const flags: CliFlags = {
        model: './model.onnx',
        engine: 'webnn',
      };
      const { steps } = resolveConfig(flags, classificationMeta);

      const engineStep = steps.find((s) => s.field === 'engine');
      expect(engineStep?.source).toBe('cli');
      expect(engineStep?.value).toBe('webnn');

      const frameworkStep = steps.find((s) => s.field === 'framework');
      expect(frameworkStep?.source).toBe('global-default');
      expect(frameworkStep?.value).toBe('html');

      const inputStep = steps.find((s) => s.field === 'input');
      expect(inputStep?.source).toBe('task-default');
    });
  });
});
