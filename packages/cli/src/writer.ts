/**
 * Layer 3: File writer + CLI output.
 *
 * Takes GeneratedFile[] from the assembler and writes them to disk.
 * Handles --force, directory creation, and structured CLI output.
 */

import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import type { ResolvedConfig, ResolverStep } from '@webai/core';
import type { GeneratedFile } from './types.js';

export interface WriteResult {
  outputDir: string;
  filesWritten: string[];
}

/**
 * Write generated files to disk.
 *
 * @param files - Generated files from the assembler
 * @param outputDir - Target output directory
 * @param options - Writer options
 * @returns Write result with list of written file paths
 * @throws Error if output directory exists without --force
 */
export function writeFiles(
  files: GeneratedFile[],
  outputDir: string,
  options: { force?: boolean } = {},
): WriteResult {
  // Check if output directory exists
  if (existsSync(outputDir) && !options.force) {
    throw new Error(
      `Output directory already exists: ${outputDir}\n` +
      `Use --force to overwrite existing files.`,
    );
  }

  // Create output directory
  mkdirSync(outputDir, { recursive: true });

  const filesWritten: string[] = [];

  for (const file of files) {
    const fullPath = join(outputDir, file.path);
    const dir = dirname(fullPath);

    // Create subdirectories as needed
    if (dir !== outputDir) {
      mkdirSync(dir, { recursive: true });
    }

    writeFileSync(fullPath, file.content, 'utf-8');
    filesWritten.push(file.path);
  }

  return { outputDir, filesWritten };
}

/**
 * Format the resolver steps for verbose output.
 */
function formatSteps(steps: ResolverStep[]): string {
  const lines: string[] = [];
  for (const s of steps) {
    const sourceLabel =
      s.source === 'cli' ? '' :
      s.source === 'auto-detect' ? ' (auto-detected)' :
      s.source === 'task-default' ? ' (task default)' :
      ' (default)';
    lines.push(`  ${s.field}: ${s.value}${sourceLabel}`);
  }
  return lines.join('\n');
}

/**
 * Format the task label for display.
 */
function taskLabel(task: string): string {
  return task
    .split('-')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

/**
 * Format the engine label for display.
 */
function engineLabel(engine: string, backend: string): string {
  const engineNames: Record<string, string> = {
    ort: 'ONNX Runtime Web',
    litert: 'LiteRT.js',
    webnn: 'WebNN API',
  };
  const name = engineNames[engine] ?? engine;
  if (backend === 'auto') {
    return `${name} (auto-select: WebNN NPU → WebNN GPU → WebGPU → WASM)`;
  }
  return `${name} (${backend})`;
}

/**
 * Format the full CLI summary output.
 *
 * @param config - Resolved config
 * @param result - Write result
 * @param steps - Resolver trace steps (for verbose mode)
 * @returns Formatted summary string
 */
export function formatSummary(
  config: ResolvedConfig,
  result: WriteResult,
  steps?: ResolverStep[],
): string {
  const lines: string[] = [];

  // Resolver checkmarks
  lines.push(`✓ Parsed model: ${config.modelPath}`);
  lines.push(`✓ Detected task: ${taskLabel(config.task)}`);
  lines.push(`✓ Engine: ${engineLabel(config.engine, config.backend)}`);
  lines.push(`✓ Input: ${config.input}`);
  lines.push(`✓ Framework: ${config.framework}`);

  if (config.preprocessIsDefault) {
    lines.push(`⚠ Using task defaults for preprocessing — verify mean/std match your model.`);
  }

  lines.push('');

  // Verbose resolver trace
  if (config.verbose && steps) {
    lines.push('Resolver trace:');
    lines.push(formatSteps(steps));
    lines.push('');
  }

  // File list
  lines.push(`Generated ${result.filesWritten.length} files in ${result.outputDir}`);
  for (const f of result.filesWritten) {
    lines.push(`  ${f}`);
  }

  lines.push('');

  // Next steps
  if (config.framework === 'html') {
    lines.push(`▸ Next: cd ${result.outputDir} && npx serve .`);
  } else {
    lines.push(`▸ Next: cd ${result.outputDir} && npm install && npm run dev`);
  }

  return lines.join('\n');
}
