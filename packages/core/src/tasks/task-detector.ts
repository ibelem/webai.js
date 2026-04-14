/**
 * Task auto-detection from model tensor shapes.
 *
 * Priority order (per plan):
 * 1. Model metadata (custom fields) — highest confidence
 * 2. Output tensor shape patterns
 * 3. Input tensor shape context (narrows candidates)
 *
 * Shape heuristics:
 *   [1, N] where N=1000 or N<5000          → image-classification (high)
 *   [1, N] where N<10                       → text-classification or small vocab (medium)
 *   [1, C, H, W] where output≈input shape  → image-segmentation (high)
 *   [1, K, D] where K>100 and D>4          → object-detection/YOLO (high)
 *   [1, D] where D>256                     → feature-extraction/embeddings (medium)
 *
 * Input context:
 *   4D [1,3,H,W] → image task
 *   2D [1,N]     → text/sequence task
 *   3D [1,T,F]   → audio task
 */

import type { ModelMetadata } from '../model-parser/types.js';
import type { TaskCandidate, TaskDetectionResult, Confidence } from './types.js';

/** Get numeric values from shape, treating dynamic dims as 0 */
function numericShape(shape: (number | string)[]): number[] {
  return shape.map((d) => (typeof d === 'number' ? d : 0));
}

function candidate(task: TaskCandidate['task'], confidence: Confidence, reason: string): TaskCandidate {
  return { task, confidence, reason };
}

/** Detect input modality from first input tensor shape */
function detectInputType(metadata: ModelMetadata): 'image' | 'text' | 'audio' | 'unknown' {
  if (metadata.inputs.length === 0) return 'unknown';

  const shape = numericShape(metadata.inputs[0].shape);

  // 4D: [batch, channels, height, width] or [batch, height, width, channels]
  if (shape.length === 4) {
    const [, c, h, w] = shape;
    // NCHW: channels in {1, 3, 4}
    if (c >= 1 && c <= 4 && h > 4 && w > 4) return 'image';
    // NHWC: last dim in {1, 3, 4}
    const lastDim = shape[3];
    if (lastDim >= 1 && lastDim <= 4 && shape[1] > 4 && shape[2] > 4) return 'image';
  }

  // 3D: [batch, time, features] — audio
  if (shape.length === 3 && shape[1] > 100) return 'audio';

  // 2D: [batch, sequence] — text
  if (shape.length === 2) return 'text';

  return 'unknown';
}

/**
 * Detect task type from model metadata using shape heuristics.
 *
 * @param metadata - Parsed model metadata with input/output tensor info
 * @returns Detection result with best match and all candidates
 */
export function detectTask(metadata: ModelMetadata): TaskDetectionResult {
  const candidates: TaskCandidate[] = [];

  if (metadata.outputs.length === 0) {
    return { detected: null, candidates: [] };
  }

  const inputType = detectInputType(metadata);
  const outShape = numericShape(metadata.outputs[0].shape);
  const inShape = metadata.inputs.length > 0 ? numericShape(metadata.inputs[0].shape) : [];

  // Pattern: [1, N] — 2D output
  if (outShape.length === 2) {
    const [, n] = outShape;

    // Classification: N around 1000 or large vocab < 5000
    if (n >= 100 && n < 5000) {
      if (inputType === 'image') {
        candidates.push(
          candidate('image-classification', 'high', `output shape [1,${n}] with image input`),
        );
      } else if (inputType === 'text') {
        candidates.push(
          candidate('text-classification', 'high', `output shape [1,${n}] with text input`),
        );
      } else if (inputType === 'audio') {
        candidates.push(
          candidate('audio-classification', 'high', `output shape [1,${n}] with audio input`),
        );
      } else {
        // No input context, default to image-classification for large N
        candidates.push(
          candidate(
            'image-classification',
            n >= 500 ? 'high' : 'medium',
            `output shape [1,${n}]`,
          ),
        );
      }
    }

    // Small output: text-classification or ambiguous
    if (n > 0 && n < 100) {
      if (inputType === 'text') {
        candidates.push(
          candidate('text-classification', 'medium', `output shape [1,${n}] with text input`),
        );
      } else if (inputType === 'image' && n < 10) {
        candidates.push(
          candidate('image-classification', 'medium', `output shape [1,${n}] — small class count`),
        );
      } else {
        candidates.push(
          candidate('text-classification', 'low', `output shape [1,${n}] — ambiguous`),
        );
        candidates.push(
          candidate('image-classification', 'low', `output shape [1,${n}] — ambiguous`),
        );
      }
    }

    // Feature extraction: large embedding dimension
    if (n > 256) {
      candidates.push(
        candidate('feature-extraction', 'medium', `output shape [1,${n}] — large embedding dim`),
      );
    }
  }

  // Pattern: [1, K, D] — 3D output (detection / YOLO)
  if (outShape.length === 3) {
    const [, k, d] = outShape;

    if (k > 100 && d > 4) {
      candidates.push(
        candidate('object-detection', 'high', `output shape [1,${k},${d}] — detection grid`),
      );
    }

    // Also could be reversed: [1, D, K] (transposed YOLO output)
    if (d > 100 && k > 4) {
      candidates.push(
        candidate('object-detection', 'high', `output shape [1,${k},${d}] — transposed detection`),
      );
    }
  }

  // Pattern: [1, C, H, W] or [1, H, W, C] — 4D output (segmentation)
  if (outShape.length === 4 && inShape.length === 4) {
    // Check if output spatial dims match input spatial dims
    const inH = inShape.length === 4 ? inShape[2] : 0;
    const inW = inShape.length === 4 ? inShape[3] : 0;
    const outH = outShape[2];
    const outW = outShape[3];

    if (inH > 0 && inW > 0 && outH === inH && outW === inW) {
      candidates.push(
        candidate('image-segmentation', 'high', `output matches input spatial dims [${outH},${outW}]`),
      );
    } else if (outShape[1] > 1 && outShape[2] > 16 && outShape[3] > 16) {
      candidates.push(
        candidate('image-segmentation', 'medium', `4D output shape [1,${outShape[1]},${outShape[2]},${outShape[3]}]`),
      );
    }
  }

  // Sort: high > medium > low
  const order: Record<string, number> = { high: 0, medium: 1, low: 2 };
  candidates.sort((a, b) => order[a.confidence] - order[b.confidence]);

  // Deduplicate: if same task appears multiple times, keep highest confidence
  const seen = new Set<string>();
  const deduped: TaskCandidate[] = [];
  for (const c of candidates) {
    if (!seen.has(c.task)) {
      seen.add(c.task);
      deduped.push(c);
    }
  }

  return {
    detected: deduped.length > 0 ? deduped[0] : null,
    candidates: deduped,
  };
}
