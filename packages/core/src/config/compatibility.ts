/**
 * Compatibility matrices: task × input mode, task × engine.
 * Used by the resolver to validate flag combinations and reject
 * invalid ones with helpful error messages.
 */

import type { TaskType, InputMode } from '../tasks/types.js';
import type { Engine } from './types.js';

/**
 * Task × Input Mode compatibility.
 * true = supported, absent = not supported.
 */
const TASK_INPUT_MATRIX: Record<TaskType, Set<InputMode>> = {
  'image-classification': new Set(['file', 'camera', 'video', 'screen']),
  'object-detection': new Set(['file', 'camera', 'video', 'screen']),
  'image-segmentation': new Set(['file', 'camera', 'video', 'screen']),
  'feature-extraction': new Set(['file', 'camera', 'video']),
  'speech-to-text': new Set(['file', 'mic']),
  'audio-classification': new Set(['file', 'mic']),
  'text-to-speech': new Set(['file']),
  'text-classification': new Set(['file']),
  'text-generation': new Set(['file']),
  'zero-shot-classification': new Set(['file']),
  'fill-mask': new Set(['file']),
  'sentence-similarity': new Set(['file']),
  'depth-estimation': new Set(['file', 'camera', 'video', 'screen']),
  'token-classification': new Set(['file']),
  'question-answering': new Set(['file']),
  'summarization': new Set(['file']),
  'translation': new Set(['file']),
};

/**
 * Task × Engine compatibility.
 */
const TASK_ENGINE_MATRIX: Record<TaskType, Set<Engine>> = {
  'image-classification': new Set(['ort', 'litert', 'webnn']),
  'object-detection': new Set(['ort', 'litert', 'webnn']),
  'image-segmentation': new Set(['ort', 'litert', 'webnn']),
  'feature-extraction': new Set(['ort', 'litert', 'webnn']),
  'speech-to-text': new Set(['ort']),
  'audio-classification': new Set(['ort']),
  'text-to-speech': new Set(['ort']),
  'text-classification': new Set(['ort', 'litert', 'webnn']),
  'text-generation': new Set(['ort', 'litert', 'webnn']),
  'zero-shot-classification': new Set(['ort', 'litert', 'webnn']),
  'fill-mask': new Set(['ort', 'litert', 'webnn']),
  'sentence-similarity': new Set(['ort', 'litert', 'webnn']),
  'depth-estimation': new Set(['ort', 'litert', 'webnn']),
  'token-classification': new Set(['ort', 'litert', 'webnn']),
  'question-answering': new Set(['ort', 'litert', 'webnn']),
  'summarization': new Set(['ort', 'litert', 'webnn']),
  'translation': new Set(['ort', 'litert', 'webnn']),
};

export class ConfigValidationError extends Error {
  constructor(
    message: string,
    public readonly suggestion?: string,
  ) {
    super(message);
    this.name = 'ConfigValidationError';
  }
}

/**
 * Validate that a task + input mode combination is supported.
 * @throws ConfigValidationError with a suggestion for valid inputs
 */
export function validateTaskInput(task: TaskType, input: InputMode): void {
  const supported = TASK_INPUT_MATRIX[task];
  if (!supported?.has(input)) {
    const validInputs = Array.from(supported ?? []).join(', ');
    throw new ConfigValidationError(
      `${input} input is not supported for ${task}`,
      `Try --input ${validInputs ? validInputs.split(', ')[0] : 'file'}`,
    );
  }
}

/**
 * Validate that a task + engine combination is supported.
 * @throws ConfigValidationError with a suggestion for valid engines
 */
export function validateTaskEngine(task: TaskType, engine: Engine): void {
  const supported = TASK_ENGINE_MATRIX[task];
  if (!supported?.has(engine)) {
    const validEngines = Array.from(supported ?? []).join(', ');
    throw new ConfigValidationError(
      `${engine} engine is not supported for ${task}`,
      `Try --engine ${validEngines ? validEngines.split(', ')[0] : 'ort'}`,
    );
  }
}

/**
 * Get supported input modes for a task.
 */
export function getSupportedInputs(task: TaskType): InputMode[] {
  return Array.from(TASK_INPUT_MATRIX[task] ?? []);
}
