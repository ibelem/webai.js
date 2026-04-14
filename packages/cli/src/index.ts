// webai CLI — programmatic API
// Re-exports for use as a library (e.g., from Web UI in Phase 2)

export { VERSION } from '@webai/core';

// Types
export type { CodeBlock, GeneratedFile } from './types.js';

// Layer 1: Inference emitters
export {
  emitLayer1,
  emitPreprocessBlock,
  emitPostprocessBlock,
  emitOrtInferenceBlock,
} from './emitters/index.js';

// Layer 2: Framework IO emitters
export {
  emitLayer2,
  emitHtml,
  emitReactVite,
} from './frameworks/index.js';
