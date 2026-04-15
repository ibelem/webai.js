/**
 * Config resolver: CliFlags + ModelMetadata → ResolvedConfig.
 *
 * Resolution order:
 * 1. Task: explicit --task flag, or auto-detect from model shapes
 * 2. Engine: explicit --engine flag, or default 'ort'
 * 3. Backend: explicit --backend flag, or 'auto'
 * 4. Framework: explicit --framework flag, or 'html'
 * 5. Input: explicit --input flag, or task default
 * 6. Preprocessing: from task profile (ImageNet defaults for image tasks)
 * 7. Validate all combinations against compatibility matrix
 */

import type { TaskType, InputMode } from '../tasks/types.js';
import type { ModelMetadata } from '../model-parser/types.js';
import { detectTask } from '../tasks/task-detector.js';
import { TASK_PROFILES } from '../tasks/task-profiles.js';
import type {
  CliFlags,
  ResolvedConfig,
  ResolverStep,
  Engine,
  Backend,
  Framework,
  CodeMode,
  OutputLang,
  Theme,
} from './types.js';
import { ConfigValidationError, validateTaskInput, validateTaskEngine } from './compatibility.js';

const VALID_ENGINES = new Set<Engine>(['ort', 'litert', 'webnn']);
const VALID_BACKENDS = new Set<Backend>(['auto', 'wasm', 'webgpu', 'webnn-cpu', 'webnn-gpu', 'webnn-npu']);
const VALID_FRAMEWORKS = new Set<Framework>(['html', 'vanilla-vite', 'react-vite', 'nextjs', 'svelte-vite', 'sveltekit', 'vue-vite', 'nuxt', 'astro']);
const VALID_MODES = new Set<CodeMode>(['raw', 'compact']);
const VALID_LANGS = new Set<OutputLang>(['js', 'ts']);
const VALID_THEMES = new Set<Theme>(['dark', 'light']);
const VALID_TASKS = new Set<string>([
  'image-classification', 'object-detection', 'image-segmentation',
  'feature-extraction', 'speech-to-text', 'audio-classification',
  'text-to-speech', 'text-classification', 'text-generation',
  'zero-shot-classification', 'fill-mask', 'sentence-similarity',
  'depth-estimation', 'token-classification', 'question-answering',
  'summarization', 'translation',
  'image-to-text', 'audio-to-audio',
  'speaker-diarization', 'voice-activity-detection',
  'text2text-generation', 'conversational', 'table-question-answering',
  'visual-question-answering', 'document-question-answering', 'image-text-to-text',
]);

function assertValid<T extends string>(value: string, validSet: Set<T>, label: string): T {
  if (!validSet.has(value as T)) {
    const valid = Array.from(validSet).join(', ');
    throw new ConfigValidationError(
      `Invalid ${label}: "${value}"`,
      `Valid options: ${valid}`,
    );
  }
  return value as T;
}

/** Normalize backend shorthand for -e webnn: "npu" → "webnn-npu" */
function normalizeBackend(backend: string, engine: Engine): string {
  if (engine === 'webnn' && !backend.startsWith('webnn-') && backend !== 'auto') {
    const expanded = `webnn-${backend}`;
    if (VALID_BACKENDS.has(expanded as Backend)) {
      return expanded;
    }
  }
  return backend;
}

/**
 * Extract model name from a path or URL.
 *
 * "/path/to/yolov8n.onnx" → "yolov8n"
 * "https://huggingface.co/user/repo/resolve/main/model.onnx" → "model"
 * "https://hf.co/user/repo/resolve/main/onnx/model_q4.onnx?download=true" → "model_q4"
 * "user/repo" → "repo" (HuggingFace model ID)
 */
export function extractModelName(modelPath: string): string {
  // Strip query params and hash
  const cleaned = modelPath.split('?')[0].split('#')[0];

  // Handle both Unix and Windows separators
  const lastSlash = Math.max(cleaned.lastIndexOf('/'), cleaned.lastIndexOf('\\'));
  const base = lastSlash >= 0 ? cleaned.slice(lastSlash + 1) : cleaned;
  const dotIdx = base.lastIndexOf('.');
  return dotIdx > 0 ? base.slice(0, dotIdx) : base;
}

export interface ResolveResult {
  config: ResolvedConfig;
  steps: ResolverStep[];
}

/**
 * Resolve CLI flags + model metadata into a fully-resolved config.
 *
 * @param flags - Raw CLI flags from commander
 * @param metadata - Parsed model metadata (from parseModelMetadata)
 * @returns Resolved config + trace steps
 * @throws ConfigValidationError for invalid flag values or incompatible combinations
 */
export function resolveConfig(flags: CliFlags, metadata: ModelMetadata): ResolveResult {
  const steps: ResolverStep[] = [];

  function step(field: string, value: string, source: ResolverStep['source']): void {
    steps.push({ field, value, source });
  }

  // 1. Task resolution
  let task: TaskType;
  if (flags.task) {
    task = assertValid(flags.task, VALID_TASKS as Set<string>, 'task') as TaskType;
    step('task', task, 'cli');
  } else {
    const detection = detectTask(metadata);
    if (!detection.detected) {
      throw new ConfigValidationError(
        'Could not detect task from model shape. Use --task to specify.',
      );
    }
    if (detection.detected.confidence === 'low') {
      const candidateList = detection.candidates.map((c) => c.task).join(', ');
      throw new ConfigValidationError(
        `Could not confidently detect task from model shape`,
        `Possible tasks: ${candidateList}. Use --task to specify.`,
      );
    }
    task = detection.detected.task;
    step('task', `${task} (${detection.detected.reason})`, 'auto-detect');
  }

  // 2. Engine
  const engine: Engine = flags.engine
    ? assertValid(flags.engine, VALID_ENGINES, 'engine')
    : 'ort';
  step('engine', engine, flags.engine ? 'cli' : 'global-default');

  // 3. Backend (normalize shorthand for webnn)
  const rawBackend = flags.backend ? normalizeBackend(flags.backend, engine) : 'auto';
  const backend: Backend = assertValid(rawBackend, VALID_BACKENDS, 'backend');
  step('backend', backend, flags.backend ? 'cli' : 'global-default');

  // 4. Framework
  const framework: Framework = flags.framework
    ? assertValid(flags.framework, VALID_FRAMEWORKS, 'framework')
    : 'html';
  step('framework', framework, flags.framework ? 'cli' : 'global-default');

  // 5. Input mode (default from task profile)
  const taskProfile = TASK_PROFILES[task];
  let input: InputMode;
  if (flags.input) {
    input = flags.input as InputMode;
    step('input', input, 'cli');
  } else {
    input = taskProfile.defaultInput;
    step('input', input, 'task-default');
  }

  // 6. Other flags
  const mode: CodeMode = flags.mode
    ? assertValid(flags.mode, VALID_MODES, 'mode')
    : 'raw';
  const lang: OutputLang = flags.lang
    ? assertValid(flags.lang, VALID_LANGS, 'lang')
    : 'js';
  const theme: Theme = flags.theme
    ? assertValid(flags.theme, VALID_THEMES, 'theme')
    : 'dark';

  const outputDir = flags.output ?? './output/';
  const offline = flags.offline ?? false;
  const verbose = flags.verbose ?? false;
  const force = flags.force ?? false;

  // 7. Preprocessing (from task profile, Decision #29)
  const preprocess = taskProfile.preprocess ?? {
    imageSize: 224,
    mean: [0, 0, 0],
    std: [1, 1, 1],
    layout: 'nchw' as const,
  };
  const preprocessIsDefault = true; // Always task defaults in Phase 1a (no HF config yet)
  step('preprocess', `${preprocess.imageSize}px, mean=[${preprocess.mean}]`, 'task-default');

  // 8. Validate combinations
  validateTaskInput(task, input);
  validateTaskEngine(task, engine);

  const modelName = extractModelName(flags.model);
  const modelSource = flags.modelSource ?? 'local-path';
  const modelUrl = flags.modelUrl;

  return {
    config: {
      task,
      engine,
      backend,
      framework,
      input,
      mode,
      lang,
      outputDir,
      offline,
      theme,
      verbose,
      force,
      preprocess,
      preprocessIsDefault,
      modelMeta: metadata,
      modelPath: flags.model,
      modelName,
      modelSource,
      modelUrl,
    },
    steps,
  };
}

// Re-export for convenience
export { ConfigValidationError } from './compatibility.js';
