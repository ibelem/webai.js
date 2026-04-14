/**
 * Configuration types for the webai.js code generation pipeline.
 *
 * CliFlags → resolveConfig() → ResolvedConfig
 *   CLI parses user flags into CliFlags.
 *   The resolver fills defaults, runs task detection, validates combos.
 *   Emitters consume the fully-resolved ResolvedConfig.
 */

import type { TaskType, InputMode, PreprocessDefaults } from '../tasks/types.js';
import type { ModelMetadata } from '../model-parser/types.js';
import type { ModelSourceType } from '../model-source/types.js';

/** Inference engine (which JS library) */
export type Engine = 'ort' | 'litert' | 'webnn';

/** Hardware backend */
export type Backend = 'auto' | 'wasm' | 'webgpu' | 'webnn-cpu' | 'webnn-gpu' | 'webnn-npu';

/** Output framework template */
export type Framework = 'html' | 'vanilla-vite' | 'react-vite' | 'nextjs' | 'sveltekit';

/** Code style */
export type CodeMode = 'raw' | 'compact';

/** Output language */
export type OutputLang = 'js' | 'ts';

/** UI theme */
export type Theme = 'dark' | 'light';

/**
 * Raw CLI flags as parsed by commander.
 * Optional fields = not provided by user (resolver fills defaults).
 */
export interface CliFlags {
  model: string;
  task?: string;
  engine?: string;
  backend?: string;
  framework?: string;
  input?: string;
  mode?: string;
  lang?: string;
  output?: string;
  offline?: boolean;
  theme?: string;
  verbose?: boolean;
  force?: boolean;
  /** Set by CLI after classifying model input (not a CLI flag) */
  modelSource?: ModelSourceType;
  /** Set by CLI after resolving model URL (not a CLI flag) */
  modelUrl?: string;
}

/**
 * Fully resolved configuration. Every field has a concrete value.
 * This is the single input to the assembler pipeline.
 */
export interface ResolvedConfig {
  /** Resolved task type */
  task: TaskType;
  /** Inference engine */
  engine: Engine;
  /** Hardware backend */
  backend: Backend;
  /** Output framework */
  framework: Framework;
  /** Input mode */
  input: InputMode;
  /** Code style */
  mode: CodeMode;
  /** Output language */
  lang: OutputLang;
  /** Output directory path */
  outputDir: string;
  /** Enable offline-first (OPFS model caching) */
  offline: boolean;
  /** UI theme */
  theme: Theme;
  /** Print resolver trace */
  verbose: boolean;
  /** Overwrite existing output */
  force: boolean;

  /** Preprocessing config for the resolved task */
  preprocess: PreprocessDefaults;
  /** Whether preprocessing used task defaults (not model-specific config) */
  preprocessIsDefault: boolean;

  /** Parsed model metadata */
  modelMeta: ModelMetadata;
  /** Original model path/identifier */
  modelPath: string;
  /** Model file name without extension */
  modelName: string;
  /** How the model was provided: local file, URL, or HuggingFace model ID */
  modelSource: ModelSourceType;
  /** Direct download URL (when modelSource is 'url' or 'hf-model-id') */
  modelUrl?: string;
}

/** A single step in the resolver trace (for --verbose output) */
export interface ResolverStep {
  field: string;
  value: string;
  source: 'cli' | 'auto-detect' | 'task-default' | 'global-default';
}
