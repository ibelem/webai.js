import { Command } from 'commander';
import { VERSION } from '@webai/core';

const program = new Command();

program
  .name('webai')
  .description('Generate standalone JS/TS code for browser-based AI inference')
  .version(VERSION);

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
    console.log('generate command called with:', options);
    // TODO: wire up assembler pipeline
  });

program
  .command('compare')
  .description('Benchmark model across available backends')
  .argument('<model>', 'Path to model file')
  .option('--json', 'Output machine-readable JSON instead of HTML')
  .action((model, options) => {
    console.log('compare command called for:', model, options);
    // TODO: implement in Phase 2
  });

// Zero-config shorthand: webai ./model.onnx
program
  .argument('[model]', 'Model file path (shorthand for webai generate -m <model>)')
  .action((model) => {
    if (model && !program.args.includes('generate') && !program.args.includes('compare')) {
      console.log('zero-config mode for:', model);
      // TODO: delegate to generate with auto-detection
    }
  });

program.parse();
