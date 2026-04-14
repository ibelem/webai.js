/**
 * Shared types for the code generation pipeline.
 *
 * CodeBlock = output of Layer 1 (inference emitters)
 * GeneratedFile = output of Layer 2 (framework IO emitters) + input to Layer 3 (writer)
 */

/**
 * A generated file with its relative path and content.
 * Used by Layer 2 (framework emitters) and Layer 3 (writer).
 */
export interface GeneratedFile {
  /** Relative path within output directory, e.g. 'app/page.tsx' */
  path: string;
  /** Full file content */
  content: string;
}

/**
 * A code block produced by a Layer 1 emitter.
 * Each block is a self-contained chunk of generated code
 * (preprocessing, inference, postprocessing, etc.).
 */
export interface CodeBlock {
  /** Block identifier: 'preprocess' | 'inference' | 'postprocess' | 'types' */
  id: string;
  /** Generated source code */
  code: string;
  /** npm packages this block depends on, e.g. ['onnxruntime-web'] */
  imports: string[];
  /** Function/variable names exported by this block */
  exports: string[];
  /** Additional binary/config files (e.g. WGWT weights for WebNN engine) */
  auxiliaryFiles?: GeneratedFile[];
}
