/**
 * Assembler: orchestrates the code generation pipeline.
 *
 * ResolvedConfig → Layer 1 (CodeBlock[]) → Layer 2 (GeneratedFile[])
 *
 * The assembler is the single entry point for code generation.
 * It never writes files — that's Layer 3 (writer).
 */

import type { ResolvedConfig } from '@webai/core';
import type { GeneratedFile } from './types.js';
import { emitLayer1 } from './emitters/index.js';
import { emitLayer2 } from './frameworks/index.js';

export type { GeneratedFile } from './types.js';

/**
 * Generate all files for the given config.
 *
 * @param config - Fully resolved config from the resolver
 * @returns Array of generated files ready for the writer
 */
export function assemble(config: ResolvedConfig): GeneratedFile[] {
  const blocks = emitLayer1(config);
  const files = emitLayer2(config, blocks);
  return files;
}
