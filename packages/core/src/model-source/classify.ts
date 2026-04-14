/**
 * Classify a model input string as a local path, URL, or HuggingFace model ID.
 *
 * Classification rules:
 * - Starts with http:// or https:// → url
 * - Contains exactly one "/" and no ".", looks like "user/repo" → hf-model-id
 * - Everything else → local-path
 */

import type { ModelSourceType } from './types.js';

/**
 * Classify a model input string.
 *
 * @param input - Raw model string from user (CLI arg or Web UI field)
 * @returns The classified source type
 */
export function classifyModelInput(input: string): ModelSourceType {
  const trimmed = input.trim();

  // URL: starts with http:// or https://
  if (/^https?:\/\//i.test(trimmed)) {
    return 'url';
  }

  // HuggingFace model ID: "owner/repo" pattern
  // Must have exactly one slash, no protocol, no file extension, no path separators
  // Examples: "user/model", "org/model-name"
  // NOT: "./model.onnx", "path/to/model.onnx", "user/repo/file.onnx"
  if (/^[a-zA-Z0-9_-]+\/[a-zA-Z0-9._-]+$/.test(trimmed) && !trimmed.includes('.')) {
    return 'hf-model-id';
  }

  return 'local-path';
}
