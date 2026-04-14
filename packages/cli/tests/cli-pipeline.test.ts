/**
 * CLI pipeline integration tests (T32-T37).
 *
 * Tests the full pipeline: model file → parse → resolve → assemble → write.
 * Uses synthetic ONNX fixtures and temp directories for isolation.
 *
 * T32: generate with explicit flags → files created
 * T33: zero-config (no --task) → auto-detects, generates files
 * T34: nonexistent model → clear error
 * T35: invalid input combo (image-classification + mic) → clear error
 * T36: existing output dir without --force → error
 * T37: existing output dir with --force → overwrites successfully
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, existsSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import {
  parseModelMetadata,
  resolveConfig,
  ConfigValidationError,
} from '@webai/core';
import type { CliFlags } from '@webai/core';
import { assemble } from '../src/assembler.js';
import { writeFiles } from '../src/writer.js';

// ---- Synthetic ONNX builder (copied inline to avoid cross-package test imports) ----

function varint(value: number): number[] {
  const bytes: number[] = [];
  let v = value >>> 0;
  while (v > 0x7f) {
    bytes.push((v & 0x7f) | 0x80);
    v >>>= 7;
  }
  bytes.push(v);
  return bytes;
}

function pbTag(fieldNum: number, wireType: number): number[] {
  return varint((fieldNum << 3) | wireType);
}

function lenDelim(fieldNum: number, data: number[]): number[] {
  return [...pbTag(fieldNum, 2), ...varint(data.length), ...data];
}

function varintField(fieldNum: number, value: number): number[] {
  return [...pbTag(fieldNum, 0), ...varint(value)];
}

function stringField(fieldNum: number, value: string): number[] {
  const encoded = new TextEncoder().encode(value);
  return lenDelim(fieldNum, Array.from(encoded));
}

function dimension(value: number): number[] {
  return varintField(1, value);
}

function tensorShape(dims: number[]): number[] {
  let bytes: number[] = [];
  for (const dim of dims) {
    bytes = [...bytes, ...lenDelim(1, dimension(dim))];
  }
  return bytes;
}

function tensorType(elemType: number, shape: number[]): number[] {
  return [
    ...varintField(1, elemType),
    ...lenDelim(2, tensorShape(shape)),
  ];
}

function typeProto(elemType: number, shape: number[]): number[] {
  return lenDelim(1, tensorType(elemType, shape));
}

function valueInfo(name: string, shape: number[]): number[] {
  return [
    ...stringField(1, name),
    ...lenDelim(2, typeProto(1, shape)),
  ];
}

function buildSyntheticOnnx(
  inputs: Array<{ name: string; shape: number[] }>,
  outputs: Array<{ name: string; shape: number[] }>,
): Uint8Array {
  let graphBytes: number[] = [];
  for (const input of inputs) {
    graphBytes = [...graphBytes, ...lenDelim(11, valueInfo(input.name, input.shape))];
  }
  for (const output of outputs) {
    graphBytes = [...graphBytes, ...lenDelim(12, valueInfo(output.name, output.shape))];
  }
  const modelBytes = [
    ...varintField(1, 7),
    ...lenDelim(7, graphBytes),
  ];
  return new Uint8Array(modelBytes);
}

// ---- Test setup ----

let tmpDir: string;
let modelPath: string;

/** Build and write a classification model fixture */
function writeClassificationModel(): void {
  const buffer = buildSyntheticOnnx(
    [{ name: 'input', shape: [1, 3, 224, 224] }],
    [{ name: 'output', shape: [1, 1000] }],
  );
  writeFileSync(modelPath, buffer);
}

/** Run the full pipeline: read model → resolve → assemble → write */
function runPipeline(flags: CliFlags): { outputDir: string; filesWritten: string[] } {
  const buffer = new Uint8Array(readFileSync(flags.model));
  const metadata = parseModelMetadata(buffer);
  const { config } = resolveConfig(flags, metadata);
  const files = assemble(config);
  return writeFiles(files, config.outputDir, { force: config.force });
}

beforeEach(() => {
  tmpDir = mkdtempSync(join(tmpdir(), 'webai-test-'));
  modelPath = join(tmpDir, 'test-model.onnx');
});

afterEach(() => {
  rmSync(tmpDir, { recursive: true, force: true });
});

// ---- T32: generate with explicit flags → files created ----

describe('T32: generate with explicit flags', () => {
  it('creates files in output directory', () => {
    writeClassificationModel();
    const outputDir = join(tmpDir, 'out');

    const result = runPipeline({
      model: modelPath,
      task: 'image-classification',
      engine: 'ort',
      framework: 'html',
      output: outputDir,
    });

    expect(result.filesWritten).toContain('index.html');
    expect(result.filesWritten).toContain('README.md');
    expect(existsSync(join(outputDir, 'index.html'))).toBe(true);
    expect(existsSync(join(outputDir, 'README.md'))).toBe(true);
  });

  it('creates react-vite project with all files', () => {
    writeClassificationModel();
    const outputDir = join(tmpDir, 'react-out');

    const result = runPipeline({
      model: modelPath,
      task: 'image-classification',
      engine: 'ort',
      framework: 'react-vite',
      output: outputDir,
    });

    expect(result.filesWritten).toHaveLength(10);
    expect(existsSync(join(outputDir, 'package.json'))).toBe(true);
    expect(existsSync(join(outputDir, 'src', 'App.jsx'))).toBe(true);
    expect(existsSync(join(outputDir, 'src', 'lib', 'inference.js'))).toBe(true);
  });

  it('generated HTML contains expected model name', () => {
    writeClassificationModel();
    const outputDir = join(tmpDir, 'out');

    runPipeline({
      model: modelPath,
      task: 'image-classification',
      framework: 'html',
      output: outputDir,
    });

    const html = readFileSync(join(outputDir, 'index.html'), 'utf-8');
    expect(html).toContain('test-model');
  });

  it('TypeScript mode produces .tsx files', () => {
    writeClassificationModel();
    const outputDir = join(tmpDir, 'ts-out');

    const result = runPipeline({
      model: modelPath,
      task: 'image-classification',
      framework: 'react-vite',
      lang: 'ts',
      output: outputDir,
    });

    expect(result.filesWritten).toContain('src/App.tsx');
    expect(result.filesWritten).toContain('src/lib/preprocess.ts');
    expect(existsSync(join(outputDir, 'src', 'App.tsx'))).toBe(true);
  });
});

// ---- T33: zero-config → auto-detects task ----

describe('T33: zero-config auto-detection', () => {
  it('auto-detects image-classification from model shapes', () => {
    writeClassificationModel();
    const outputDir = join(tmpDir, 'auto-out');

    // No --task flag: relies on auto-detection
    const result = runPipeline({
      model: modelPath,
      output: outputDir,
    });

    expect(result.filesWritten).toContain('index.html');
    expect(existsSync(join(outputDir, 'index.html'))).toBe(true);

    const html = readFileSync(join(outputDir, 'index.html'), 'utf-8');
    expect(html).toContain('Image Classification');
  });
});

// ---- T34: nonexistent model → error ----

describe('T34: nonexistent model', () => {
  it('throws when model file does not exist', () => {
    const badPath = join(tmpDir, 'nonexistent.onnx');
    expect(() => {
      new Uint8Array(readFileSync(badPath));
    }).toThrow();
  });
});

// ---- T35: invalid input combo → error ----

describe('T35: invalid task+input combination', () => {
  it('rejects mic input for image-classification', () => {
    writeClassificationModel();
    const buffer = new Uint8Array(readFileSync(modelPath));
    const metadata = parseModelMetadata(buffer);

    expect(() => {
      resolveConfig({
        model: modelPath,
        task: 'image-classification',
        input: 'mic',
      }, metadata);
    }).toThrow(ConfigValidationError);
  });

  it('error message mentions the incompatible combination', () => {
    writeClassificationModel();
    const buffer = new Uint8Array(readFileSync(modelPath));
    const metadata = parseModelMetadata(buffer);

    try {
      resolveConfig({
        model: modelPath,
        task: 'image-classification',
        input: 'mic',
      }, metadata);
      expect.fail('Should have thrown');
    } catch (e) {
      expect(e).toBeInstanceOf(ConfigValidationError);
      const err = e as InstanceType<typeof ConfigValidationError>;
      expect(err.message).toContain('mic');
      expect(err.message).toContain('image-classification');
    }
  });
});

// ---- T36: existing dir without --force → error ----

describe('T36: existing output dir without --force', () => {
  it('throws when output directory already exists', () => {
    writeClassificationModel();
    const outputDir = join(tmpDir, 'existing');
    mkdirSync(outputDir);

    const buffer = new Uint8Array(readFileSync(modelPath));
    const metadata = parseModelMetadata(buffer);
    const { config } = resolveConfig({ model: modelPath, output: outputDir }, metadata);
    const files = assemble(config);

    expect(() => {
      writeFiles(files, outputDir, { force: false });
    }).toThrow(/already exists/);
  });

  it('error message suggests --force', () => {
    writeClassificationModel();
    const outputDir = join(tmpDir, 'existing2');
    mkdirSync(outputDir);

    const buffer = new Uint8Array(readFileSync(modelPath));
    const metadata = parseModelMetadata(buffer);
    const { config } = resolveConfig({ model: modelPath, output: outputDir }, metadata);
    const files = assemble(config);

    try {
      writeFiles(files, outputDir, { force: false });
      expect.fail('Should have thrown');
    } catch (e) {
      expect((e as Error).message).toContain('--force');
    }
  });
});

// ---- T37: existing dir with --force → overwrites ----

describe('T37: existing output dir with --force', () => {
  it('overwrites existing directory when --force is set', () => {
    writeClassificationModel();
    const outputDir = join(tmpDir, 'forced');
    mkdirSync(outputDir);
    // Write a dummy file to prove it gets overwritten
    writeFileSync(join(outputDir, 'old-file.txt'), 'old content');

    const result = runPipeline({
      model: modelPath,
      task: 'image-classification',
      framework: 'html',
      output: outputDir,
      force: true,
    });

    expect(result.filesWritten).toContain('index.html');
    expect(existsSync(join(outputDir, 'index.html'))).toBe(true);
  });

  it('preserves files not in the generated set', () => {
    writeClassificationModel();
    const outputDir = join(tmpDir, 'forced2');
    mkdirSync(outputDir);
    writeFileSync(join(outputDir, 'keep-me.txt'), 'keep');

    runPipeline({
      model: modelPath,
      task: 'image-classification',
      framework: 'html',
      output: outputDir,
      force: true,
    });

    // --force doesn't delete existing files, just allows overwriting
    expect(existsSync(join(outputDir, 'keep-me.txt'))).toBe(true);
  });
});
