import { describe, it, expect } from 'vitest';
import { generateCompareHtml, generateCompareJson } from '../src/compare.js';
import type { ModelMetadata } from '@webai/core';

describe('compare command', () => {
  const modelMeta: ModelMetadata = {
    format: 'onnx',
    inputs: [{ name: 'input', dataType: 'float32', shape: [1, 3, 224, 224] }],
    outputs: [{ name: 'output', dataType: 'float32', shape: [1, 1000] }],
  };

  describe('generateCompareHtml', () => {
    it('produces valid HTML with benchmark script', () => {
      const html = generateCompareHtml('model.onnx', modelMeta);
      expect(html).toContain('<!DOCTYPE html>');
      expect(html).toContain('onnxruntime-web');
      expect(html).toContain('async function benchmark(');
      expect(html).toContain('wasm');
      expect(html).toContain('webgpu');
      expect(html).toContain('webnn');
    });

    it('includes all metric categories', () => {
      const html = generateCompareHtml('model.onnx', modelMeta);
      expect(html).toContain('Cold Start');
      expect(html).toContain('Warm');
      expect(html).toContain('Throughput');
    });

    it('uses correct model input shape', () => {
      const html = generateCompareHtml('model.onnx', modelMeta);
      expect(html).toContain('[1, 3, 224, 224]');
      // Also should contain the input name
      expect(html).toContain('input');
    });

    it('handles model URL', () => {
      const html = generateCompareHtml('https://hf.co/model.onnx', modelMeta);
      expect(html).toContain('https://hf.co/model.onnx');
    });
  });

  describe('generateCompareJson', () => {
    it('produces JSON template structure', () => {
      const json = generateCompareJson('model.onnx', modelMeta);
      const parsed = JSON.parse(json);
      expect(parsed.model).toBe('model.onnx');
      expect(parsed.backends).toEqual(['wasm', 'webgpu', 'webnn']);
      expect(parsed.metrics).toContain('cold_latency_ms');
      expect(parsed.metrics).toContain('warm_latency_ms');
      expect(parsed.metrics).toContain('throughput_ips');
    });
  });
});
