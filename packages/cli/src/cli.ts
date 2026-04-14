import { readFileSync } from 'node:fs';
import { Command } from 'commander';
import {
  VERSION,
  parseModelMetadata,
  resolveConfig,
  ConfigValidationError,
} from '@webai/core';
import type { CliFlags } from '@webai/core';
import { assemble } from './assembler.js';
import { writeFiles, formatSummary } from './writer.js';

const program = new Command();

program
  .name('webai')
  .description('Generate standalone JS/TS code for browser-based AI inference')
  .version(VERSION);

/**
 * Core generate logic shared between `webai generate` and the zero-config shorthand.
 */
function runGenerate(flags: CliFlags): void {
  // 1. Read and parse model file
  let buffer: Uint8Array;
  try {
    buffer = new Uint8Array(readFileSync(flags.model));
  } catch {
    console.error(`✗ Model not found: ${flags.model}`);
    process.exitCode = 1;
    return;
  }

  let metadata;
  try {
    metadata = parseModelMetadata(buffer);
  } catch (e) {
    console.error(`✗ Could not parse model: ${flags.model}`);
    if (e instanceof Error) console.error(`  ${e.message}`);
    process.exitCode = 1;
    return;
  }

  // 2. Resolve config
  let result;
  try {
    result = resolveConfig(flags, metadata);
  } catch (e) {
    if (e instanceof ConfigValidationError) {
      console.error(`✗ ${e.message}`);
      if (e.suggestion) console.error(`  ${e.suggestion}`);
    } else if (e instanceof Error) {
      console.error(`✗ Configuration error: ${e.message}`);
    }
    process.exitCode = 1;
    return;
  }

  const { config, steps } = result;

  // 3. Assemble (Layer 1 → Layer 2)
  const files = assemble(config);

  // 4. Write files (Layer 3)
  let writeResult;
  try {
    writeResult = writeFiles(files, config.outputDir, { force: config.force });
  } catch (e) {
    if (e instanceof Error) {
      console.error(`✗ ${e.message}`);
    }
    process.exitCode = 1;
    return;
  }

  // 5. Print summary
  console.log(formatSummary(config, writeResult, steps));
}

program
  .command('generate')
  .description('Generate inference code from a model file')
  .requiredOption('-m, --model <path>', 'Path to model file (.onnx or .tflite)')
  .option('-t, --task <task>', 'Task type (auto-detected if omitted)')
  .option('-e, --engine <engine>', 'Inference engine: ort, litert, webnn', 'ort')
  .option('-b, --backend <backend>', 'Hardware backend: wasm, webgpu, webnn-cpu, webnn-gpu, webnn-npu')
  .option('-f, --framework <framework>', 'Output framework: html, react-vite, vanilla-vite, nextjs, sveltekit', 'html')
  .option('-i, --input <input>', 'Input mode: file, camera, video, mic, screen')
  .option('--mode <mode>', 'Code style: raw, compact', 'raw')
  .option('-l, --lang <lang>', 'Output language: js, ts', 'js')
  .option('-o, --output <dir>', 'Output directory', './output/')
  .option('--offline', 'Enable offline-first with OPFS model caching')
  .option('--theme <theme>', 'UI theme: dark, light', 'dark')
  .option('-v, --verbose', 'Print resolver trace and debug info')
  .option('--force', 'Overwrite existing output directory')
  .action((options) => {
    runGenerate(options as CliFlags);
  });

program
  .command('compare')
  .description('Benchmark model across available backends')
  .argument('<model>', 'Path to model file')
  .option('--json', 'Output machine-readable JSON instead of HTML')
  .action((_model, _options) => {
    console.error('✗ compare command is not yet implemented (Phase 2)');
    process.exitCode = 1;
  });

// Zero-config shorthand: webai ./model.onnx
program
  .argument('[model]', 'Model file path (shorthand for webai generate -m <model>)')
  .action((model) => {
    if (model && !program.args.includes('generate') && !program.args.includes('compare')) {
      runGenerate({ model });
    }
  });

program.parse();
