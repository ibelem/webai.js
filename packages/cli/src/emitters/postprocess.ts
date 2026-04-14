/**
 * Postprocess emitter: generates standalone postprocessing source code.
 *
 * Mirrors the logic in @webai/core postprocessing functions exactly.
 * Cross-verified by tests T21 and T22: eval(emitted code) === real function output.
 *
 * Task dispatch:
 *   image-classification  → softmax + topK
 *   object-detection      → iou + nms + decodeDetections
 *   image-segmentation    → argmaxMask
 *   feature-extraction    → postprocessEmbeddings (passthrough)
 */

import type { ResolvedConfig } from '@webai/core';
import type { CodeBlock } from '../types.js';

// ---- Shared: softmax, topK (used by classification + audio-classification + text-classification) ----

/** Emit the softmax function as standalone JS/TS code */
function emitSoftmax(ts: boolean): string {
  const t = ts;
  return `/**
 * Softmax: convert raw logits to probabilities.
 * Uses max-subtraction for numerical stability.
 */
function softmax(logits${t ? ': ArrayLike<number>' : ''})${t ? ': Float32Array' : ''} {
  const len = logits.length;
  const out = new Float32Array(len);

  let max = -Infinity;
  for (let i = 0; i < len; i++) {
    if (logits[i] > max) max = logits[i];
  }

  let sum = 0;
  for (let i = 0; i < len; i++) {
    out[i] = Math.exp(logits[i] - max);
    sum += out[i];
  }

  for (let i = 0; i < len; i++) {
    out[i] /= sum;
  }

  return out;
}`;
}

/** Emit the topK function as standalone JS/TS code */
function emitTopK(ts: boolean): string {
  const t = ts;
  const resultType = t ? ': { indices: number[]; values: number[] }' : '';
  return `/**
 * Return the top K elements by value, descending.
 * Ties are broken by lower index first.
 */
function topK(arr${t ? ': ArrayLike<number>' : ''}, k${t ? ': number' : ''})${resultType} {
  const len = arr.length;
  const n = Math.min(k, len);

  const indices = Array.from({ length: len }, (_, i) => i);
  indices.sort((a, b) => {
    const diff = arr[b] - arr[a];
    return diff !== 0 ? diff : a - b;
  });

  const topIndices = indices.slice(0, n);
  const topValues = topIndices.map((i) => arr[i]);

  return { indices: topIndices, values: topValues };
}`;
}

/** Emit a postprocessResults convenience function for image-classification */
function emitPostprocessClassification(ts: boolean): string {
  const t = ts;
  const resultType = t ? ': { indices: number[]; values: number[] }' : '';
  return `/**
 * Postprocess model output: softmax → top-5 results.
 */
function postprocessResults(output${t ? ': ArrayLike<number>' : ''})${resultType} {
  const probs = softmax(output);
  return topK(probs, 5);
}`;
}

// ---- Object Detection: iou + nms + decode ----

function emitIou(ts: boolean): string {
  const t = ts;
  const boxType = t ? 'BoundingBox' : '';
  const paramA = t ? `a: ${boxType}` : 'a';
  const paramB = t ? `b: ${boxType}` : 'b';
  return `/**
 * Compute Intersection over Union (IoU) between two bounding boxes.
 */
function iou(${paramA}, ${paramB})${t ? ': number' : ''} {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);

  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = a.width * a.height;
  const areaB = b.width * b.height;
  const union = areaA + areaB - intersection;

  return union > 0 ? intersection / union : 0;
}`;
}

function emitNms(ts: boolean): string {
  const t = ts;
  return `/**
 * Non-Maximum Suppression: filter overlapping bounding boxes.
 * Keeps the highest-scoring box and suppresses boxes with IoU >= threshold.
 */
function nms(boxes${t ? ': BoundingBox[]' : ''}, iouThreshold${t ? ': number' : ''})${t ? ': number[]' : ''} {
  if (boxes.length === 0) return [];

  const indices = Array.from({ length: boxes.length }, (_, i) => i);
  indices.sort((a, b) => boxes[b].score - boxes[a].score);

  const kept${t ? ': number[]' : ''} = [];
  const suppressed = new Set(${t ? '<number>' : ''});

  for (const idx of indices) {
    if (suppressed.has(idx)) continue;
    kept.push(idx);

    for (const other of indices) {
      if (other === idx || suppressed.has(other)) continue;
      if (iou(boxes[idx], boxes[other]) >= iouThreshold) {
        suppressed.add(other);
      }
    }
  }

  return kept;
}`;
}

function emitBoundingBoxType(ts: boolean): string {
  if (!ts) return '';
  return `interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
  classIndex: number;
  score: number;
}`;
}

function emitDecodeDetections(ts: boolean): string {
  const t = ts;
  return `/**
 * Decode YOLO-style detection output.
 *
 * Input tensor shape: [1, numAttributes, numAnchors]
 *   - numAttributes = 4 (cx, cy, w, h) + numClasses
 *   - Values are in model input pixel space (e.g., 640x640)
 *
 * Output: BoundingBox[] with x, y, width, height, classIndex, score.
 * Boxes with confidence below scoreThreshold are discarded.
 */
function decodeDetections(
  output${t ? ': Float32Array' : ''},
  numAnchors${t ? ': number' : ''},
  numAttributes${t ? ': number' : ''},
  scoreThreshold${t ? ': number' : ''} = 0.25
)${t ? ': BoundingBox[]' : ''} {
  const numClasses = numAttributes - 4;
  const boxes${t ? ': BoundingBox[]' : ''} = [];

  for (let a = 0; a < numAnchors; a++) {
    // Transpose: data is stored as [attr][anchor], read column-major
    const cx = output[0 * numAnchors + a];
    const cy = output[1 * numAnchors + a];
    const w = output[2 * numAnchors + a];
    const h = output[3 * numAnchors + a];

    // Find best class
    let bestClass = 0;
    let bestScore = -Infinity;
    for (let c = 0; c < numClasses; c++) {
      const score = output[(4 + c) * numAnchors + a];
      if (score > bestScore) {
        bestScore = score;
        bestClass = c;
      }
    }

    if (bestScore < scoreThreshold) continue;

    boxes.push({
      x: cx - w / 2,
      y: cy - h / 2,
      width: w,
      height: h,
      classIndex: bestClass,
      score: bestScore,
    });
  }

  return boxes;
}`;
}

function emitPostprocessDetection(ts: boolean): string {
  const t = ts;
  return `/**
 * Postprocess detection model output: decode boxes → NMS → return filtered boxes.
 *
 * @param output - Raw model output (Float32Array)
 * @param numAnchors - Number of anchor boxes (e.g., 8400 for YOLOv8)
 * @param numAttributes - Number of attributes per anchor (4 + numClasses)
 * @param scoreThreshold - Minimum confidence to keep a detection (default: 0.25)
 * @param iouThreshold - IoU threshold for NMS (default: 0.45)
 */
function postprocessDetections(
  output${t ? ': Float32Array' : ''},
  numAnchors${t ? ': number' : ''},
  numAttributes${t ? ': number' : ''},
  scoreThreshold${t ? ': number' : ''} = 0.25,
  iouThreshold${t ? ': number' : ''} = 0.45
)${t ? ': BoundingBox[]' : ''} {
  const allBoxes = decodeDetections(output, numAnchors, numAttributes, scoreThreshold);
  const keptIndices = nms(allBoxes, iouThreshold);
  return keptIndices.map((i) => allBoxes[i]);
}`;
}

// ---- Image Segmentation: argmax mask ----

function emitArgmaxMask(ts: boolean): string {
  const t = ts;
  return `/**
 * Compute argmax segmentation mask from model output.
 *
 * Input: Float32Array of shape [numClasses, height, width] (CHW layout).
 * Output: Uint8Array of shape [height * width], each value is the class index
 * with the highest score at that pixel.
 */
function argmaxMask(
  output${t ? ': Float32Array' : ''},
  numClasses${t ? ': number' : ''},
  height${t ? ': number' : ''},
  width${t ? ': number' : ''}
)${t ? ': Uint8Array' : ''} {
  const numPixels = height * width;
  const mask = new Uint8Array(numPixels);

  for (let p = 0; p < numPixels; p++) {
    let bestClass = 0;
    let bestVal = output[p]; // class 0 at offset p
    for (let c = 1; c < numClasses; c++) {
      const val = output[c * numPixels + p];
      if (val > bestVal) {
        bestVal = val;
        bestClass = c;
      }
    }
    mask[p] = bestClass;
  }

  return mask;
}`;
}

function emitPostprocessSegmentation(ts: boolean): string {
  const t = ts;
  return `/**
 * Postprocess segmentation model output: extract class mask.
 *
 * @param output - Raw model output (Float32Array), shape [1, C, H, W]
 * @param numClasses - Number of segmentation classes
 * @param height - Output mask height
 * @param width - Output mask width
 * @returns Uint8Array mask where each pixel is the class index
 */
function postprocessSegmentation(
  output${t ? ': Float32Array' : ''},
  numClasses${t ? ': number' : ''},
  height${t ? ': number' : ''},
  width${t ? ': number' : ''}
)${t ? ': Uint8Array' : ''} {
  return argmaxMask(output, numClasses, height, width);
}`;
}

// ---- Feature Extraction: passthrough ----

function emitPostprocessEmbeddings(ts: boolean): string {
  const t = ts;
  return `/**
 * Postprocess feature extraction output.
 * Returns the raw embedding vector (no transformation needed).
 *
 * @param output - Raw model output (Float32Array), shape [1, D]
 * @returns Float32Array embedding vector
 */
function postprocessEmbeddings(output${t ? ': Float32Array' : ''})${t ? ': Float32Array' : ''} {
  return output instanceof Float32Array ? output : new Float32Array(output);
}`;
}

// ---- Block emitter dispatch ----

/**
 * Emit the postprocess CodeBlock for a given config.
 *
 * Dispatches by task:
 *   image-classification  → softmax + topK + postprocessResults
 *   object-detection      → BoundingBox type + iou + nms + decodeDetections + postprocessDetections
 *   image-segmentation    → argmaxMask + postprocessSegmentation
 *   feature-extraction    → postprocessEmbeddings
 */
export function emitPostprocessBlock(config: ResolvedConfig): CodeBlock {
  const ts = config.lang === 'ts';

  const parts: string[] = [];
  const exports: string[] = [];

  switch (config.task) {
    case 'image-classification':
    case 'audio-classification':
    case 'text-classification':
      parts.push(emitSoftmax(ts));
      parts.push(emitTopK(ts));
      parts.push(emitPostprocessClassification(ts));
      exports.push('softmax', 'topK', 'postprocessResults');
      break;

    case 'object-detection': {
      const typeDecl = emitBoundingBoxType(ts);
      if (typeDecl) parts.push(typeDecl);
      parts.push(emitIou(ts));
      parts.push(emitNms(ts));
      parts.push(emitDecodeDetections(ts));
      parts.push(emitPostprocessDetection(ts));
      exports.push('nms', 'decodeDetections', 'postprocessDetections');
      break;
    }

    case 'image-segmentation':
      parts.push(emitArgmaxMask(ts));
      parts.push(emitPostprocessSegmentation(ts));
      exports.push('argmaxMask', 'postprocessSegmentation');
      break;

    case 'feature-extraction':
      parts.push(emitPostprocessEmbeddings(ts));
      exports.push('postprocessEmbeddings');
      break;

    default:
      // Tasks without postprocessing (speech-to-text, text-generation, text-to-speech)
      break;
  }

  return {
    id: 'postprocess',
    code: parts.join('\n\n'),
    imports: [],
    exports,
  };
}
