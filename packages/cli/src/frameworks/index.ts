/**
 * Layer 2: Framework IO emitters.
 *
 * Takes ResolvedConfig + CodeBlock[] from Layer 1,
 * produces GeneratedFile[] ready for Layer 3 (writer).
 *
 * Each framework emitter wraps the code blocks in framework-specific
 * scaffold: project files, component structure, CSS, package.json, README.
 */

import type { ResolvedConfig } from '@webai/core';
import type { CodeBlock, GeneratedFile } from '../types.js';
import { emitHtml } from './html.js';
import { emitVanillaVite } from './vanilla-vite.js';
import { emitReactVite } from './react-vite.js';
import { emitNextjs } from './nextjs.js';
import { emitSvelteKit } from './sveltekit.js';

/**
 * Emit all Layer 2 files for the given config and code blocks.
 *
 * @param config - Fully resolved config
 * @param blocks - Code blocks from Layer 1 emitters
 * @returns Array of GeneratedFiles ready for the writer
 */
export function emitLayer2(config: ResolvedConfig, blocks: CodeBlock[]): GeneratedFile[] {
  switch (config.framework) {
    case 'html':
      return emitHtml(config, blocks);
    case 'vanilla-vite':
      return emitVanillaVite(config, blocks);
    case 'react-vite':
      return emitReactVite(config, blocks);
    case 'nextjs':
      return emitNextjs(config, blocks);
    case 'sveltekit':
      return emitSvelteKit(config, blocks);
    default:
      throw new Error(`Unsupported framework: ${config.framework}`);
  }
}

export { emitHtml } from './html.js';
export { emitVanillaVite } from './vanilla-vite.js';
export { emitReactVite } from './react-vite.js';
export { emitNextjs } from './nextjs.js';
export { emitSvelteKit } from './sveltekit.js';
