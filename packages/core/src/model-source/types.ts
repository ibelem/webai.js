/**
 * Model source types.
 *
 * A model input can be:
 * - A local file path (./model.onnx, /abs/path/model.tflite)
 * - A direct URL (https://huggingface.co/.../model.onnx)
 * - A HuggingFace model ID (user/repo)
 */

/** Classified model input type */
export type ModelSourceType = 'local-path' | 'url' | 'hf-model-id';

/** Result of classifying a model input string */
export interface ModelSourceInfo {
  type: ModelSourceType;
  /** Original input string */
  original: string;
  /** Normalized URL (for url and hf-model-id types) */
  url?: string;
  /** Model filename extracted from input */
  filename?: string;
}
