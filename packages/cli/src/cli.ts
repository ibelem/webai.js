import { readFileSync } from 'node:fs';
import { Command } from 'commander';
import {
  VERSION,
  parseModelMetadata,
  resolveConfig,
  ConfigValidationError,
  classifyModelInput,
  transformHuggingFaceUrl,
  getHuggingFaceMirrorUrl,
  isHuggingFaceUrl,
  buildHfApiUrl,
  buildHfFileUrl,
  pickBestModelFile,
} from '@webai/core';
import type { CliFlags, ModelSourceType } from '@webai/core';
import { assemble } from './assembler.js';
import { writeFiles, formatSummary } from './writer.js';
import { generateCompareHtml, generateCompareJson } from './compare.js';

const program = new Command();

program
  .name('webai')
  .description('Generate standalone JS/TS code for browser-based AI inference')
  .version(VERSION);

/**
 * Fetch a URL with timeout support.
 */
async function fetchWithTimeout(url: string, timeoutMs: number): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { signal: controller.signal });
    return response;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Fetch model bytes from a URL.
 * Tries the primary URL first, falls back to HuggingFace mirror if available.
 */
async function fetchModelFromUrl(url: string, verbose: boolean): Promise<{ buffer: Uint8Array; finalUrl: string }> {
  const normalizedUrl = isHuggingFaceUrl(url) ? transformHuggingFaceUrl(url) : url;
  const mirrorUrl = getHuggingFaceMirrorUrl(normalizedUrl);

  // Try primary URL first
  try {
    if (verbose && mirrorUrl) {
      console.log('  Trying primary URL...');
    }
    const response = await fetchWithTimeout(normalizedUrl, 30_000);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    const contentType = response.headers.get('content-type') || '';
    if (contentType.includes('text/html')) {
      throw new Error(
        `URL returned HTML, not a model file. For HuggingFace, use the /resolve/ URL (not /blob/).`,
      );
    }
    const arrayBuffer = await response.arrayBuffer();
    return { buffer: new Uint8Array(arrayBuffer), finalUrl: normalizedUrl };
  } catch (primaryError) {
    // Try mirror if available
    if (mirrorUrl) {
      console.log('  Primary HuggingFace URL slow or unavailable, trying mirror...');
      try {
        const response = await fetchWithTimeout(mirrorUrl, 30_000);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const arrayBuffer = await response.arrayBuffer();
        // Return the original (non-mirror) URL for generated code
        return { buffer: new Uint8Array(arrayBuffer), finalUrl: normalizedUrl };
      } catch {
        // Mirror also failed, throw original error
      }
    }
    const msg = primaryError instanceof Error ? primaryError.message : String(primaryError);
    throw new Error(`Could not fetch model from ${normalizedUrl}\n  ${msg}`);
  }
}

/**
 * Resolve a HuggingFace model ID to a direct download URL.
 * Fetches the model's file list via HF API and picks the best model file.
 */
async function resolveHfModelId(
  modelId: string,
  preferTflite: boolean,
  verbose: boolean,
): Promise<{ url: string; filename: string }> {
  const apiUrl = buildHfApiUrl(modelId);
  if (verbose) {
    console.log(`  Fetching model info: ${apiUrl}`);
  }

  let response;
  try {
    response = await fetchWithTimeout(apiUrl, 15_000);
  } catch {
    throw new Error(
      `Could not reach HuggingFace API for "${modelId}". Check your network connection.`,
    );
  }

  if (!response.ok) {
    if (response.status === 404) {
      throw new Error(`Model not found on HuggingFace: "${modelId}". Check the model ID.`);
    }
    throw new Error(`HuggingFace API error for "${modelId}": HTTP ${response.status}`);
  }

  let data;
  try {
    data = await response.json() as { siblings?: Array<{ rfilename: string }> };
  } catch {
    throw new Error(`Invalid response from HuggingFace API for "${modelId}".`);
  }

  if (!data.siblings || !Array.isArray(data.siblings)) {
    throw new Error(`No file listing found for "${modelId}".`);
  }

  const filename = pickBestModelFile(data.siblings, preferTflite);
  if (!filename) {
    const ext = preferTflite ? '.tflite' : '.onnx';
    throw new Error(
      `No ${ext} files found in "${modelId}". This model may not have compatible weights.\n` +
      `  Try downloading and converting manually, or use a different model.`,
    );
  }

  const url = buildHfFileUrl(modelId, filename);
  return { url, filename };
}

/**
 * Core generate logic shared between `webai generate` and the zero-config shorthand.
 */
async function runGenerate(flags: CliFlags): Promise<void> {
  const sourceType: ModelSourceType = classifyModelInput(flags.model);
  const verbose = flags.verbose ?? false;

  let buffer: Uint8Array;
  let modelUrl: string | undefined;

  if (sourceType === 'local-path') {
    // Local file: read from disk
    try {
      buffer = new Uint8Array(readFileSync(flags.model));
    } catch {
      console.error(`✗ Model not found: ${flags.model}`);
      process.exitCode = 1;
      return;
    }
  } else if (sourceType === 'hf-model-id') {
    // HuggingFace model ID: resolve to URL, then fetch
    const preferTflite = flags.engine === 'litert';
    try {
      const resolved = await resolveHfModelId(flags.model, preferTflite, verbose);
      console.log(`✓ Resolved model ID: ${flags.model} → ${resolved.filename}`);
      modelUrl = resolved.url;

      const fetched = await fetchModelFromUrl(resolved.url, verbose);
      buffer = fetched.buffer;
      modelUrl = fetched.finalUrl;
      console.log(`✓ Fetched model (${(buffer.byteLength / 1024 / 1024).toFixed(1)} MB)`);
    } catch (e) {
      console.error(`✗ ${e instanceof Error ? e.message : String(e)}`);
      process.exitCode = 1;
      return;
    }
  } else {
    // Direct URL: fetch
    try {
      const fetched = await fetchModelFromUrl(flags.model, verbose);
      buffer = fetched.buffer;
      modelUrl = fetched.finalUrl;
      const host = new URL(modelUrl).hostname;
      console.log(`✓ Fetched model (${(buffer.byteLength / 1024 / 1024).toFixed(1)} MB, from ${host})`);
    } catch (e) {
      console.error(`✗ ${e instanceof Error ? e.message : String(e)}`);
      process.exitCode = 1;
      return;
    }
  }

  // Parse model metadata from the buffer
  let metadata;
  try {
    metadata = parseModelMetadata(buffer);
  } catch (e) {
    console.error(`✗ Could not parse model: ${flags.model}`);
    if (e instanceof Error) console.error(`  ${e.message}`);
    process.exitCode = 1;
    return;
  }

  // Resolve config (pass model source info through flags)
  let result;
  try {
    result = resolveConfig(
      { ...flags, modelSource: sourceType, modelUrl },
      metadata,
    );
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

  // Assemble (Layer 1 → Layer 2)
  const files = assemble(config);

  // Write files (Layer 3)
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

  // Print summary
  console.log(formatSummary(config, writeResult, steps));
}

program
  .command('generate')
  .description('Generate inference code from a model file')
  .requiredOption('-m, --model <path>', 'Path, URL, or HuggingFace model ID (.onnx or .tflite)')
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
  .argument('<model>', 'Path, URL, or HuggingFace model ID (.onnx)')
  .option('--json', 'Output machine-readable JSON instead of HTML')
  .option('-o, --output <dir>', 'Output directory', './compare-output/')
  .option('-v, --verbose', 'Print debug info')
  .option('--force', 'Overwrite existing output directory')
  .action(async (model: string, options: { json?: boolean; output?: string; verbose?: boolean; force?: boolean }) => {
    const verbose = options.verbose ?? false;
    const sourceType = classifyModelInput(model);

    let buffer: Uint8Array;
    let modelUrl: string | undefined;

    if (sourceType === 'local-path') {
      try {
        buffer = new Uint8Array(readFileSync(model));
      } catch {
        console.error(`✗ Model not found: ${model}`);
        process.exitCode = 1;
        return;
      }
    } else if (sourceType === 'hf-model-id') {
      try {
        const resolved = await resolveHfModelId(model, false, verbose);
        console.log(`✓ Resolved model ID: ${model} → ${resolved.filename}`);
        modelUrl = resolved.url;

        const fetched = await fetchModelFromUrl(resolved.url, verbose);
        buffer = fetched.buffer;
        modelUrl = fetched.finalUrl;
        console.log(`✓ Fetched model (${(buffer.byteLength / 1024 / 1024).toFixed(1)} MB)`);
      } catch (e) {
        console.error(`✗ ${e instanceof Error ? e.message : String(e)}`);
        process.exitCode = 1;
        return;
      }
    } else {
      try {
        const fetched = await fetchModelFromUrl(model, verbose);
        buffer = fetched.buffer;
        modelUrl = fetched.finalUrl;
        const host = new URL(modelUrl).hostname;
        console.log(`✓ Fetched model (${(buffer.byteLength / 1024 / 1024).toFixed(1)} MB, from ${host})`);
      } catch (e) {
        console.error(`✗ ${e instanceof Error ? e.message : String(e)}`);
        process.exitCode = 1;
        return;
      }
    }

    // Parse model metadata
    let metadata;
    try {
      metadata = parseModelMetadata(buffer);
    } catch (e) {
      console.error(`✗ Could not parse model: ${model}`);
      if (e instanceof Error) console.error(`  ${e.message}`);
      process.exitCode = 1;
      return;
    }

    // Use the URL for remote models so the HTML page can fetch it
    const displayPath = modelUrl ?? model;

    if (options.json) {
      const json = generateCompareJson(displayPath, metadata);
      const outputDir = options.output ?? './compare-output/';
      try {
        writeFiles(
          [{ path: 'compare.json', content: json }],
          outputDir,
          { force: options.force },
        );
        console.log(`✓ Generated compare.json in ${outputDir}`);
      } catch (e) {
        if (e instanceof Error) console.error(`✗ ${e.message}`);
        process.exitCode = 1;
        return;
      }
    } else {
      const html = generateCompareHtml(displayPath, metadata);
      const outputDir = options.output ?? './compare-output/';
      try {
        writeFiles(
          [{ path: 'index.html', content: html }],
          outputDir,
          { force: options.force },
        );
        console.log(`✓ Generated benchmark page in ${outputDir}`);
        console.log(`▸ Next: cd ${outputDir} && npx serve .`);
      } catch (e) {
        if (e instanceof Error) console.error(`✗ ${e.message}`);
        process.exitCode = 1;
        return;
      }
    }
  });

// Zero-config shorthand: webai ./model.onnx or webai https://hf.co/.../model.onnx or webai user/repo
program
  .argument('[model]', 'Model file path, URL, or HuggingFace model ID')
  .action((model) => {
    if (model && !program.args.includes('generate') && !program.args.includes('compare')) {
      runGenerate({ model });
    }
  });

program.parse();
