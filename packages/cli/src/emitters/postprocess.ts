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
 *   fill-mask             → softmax + topK + postprocessFillMask
 *   sentence-similarity   → cosineSimilarity + postprocessSimilarity
 *   depth-estimation      → depthNormalize + depthToColormap + postprocessDepth
 *   token-classification  → tokenArgmax + extractSpans
 *   question-answering    → postprocessQA (start/end span extraction)
 *   summarization         → seq2seqGreedyDecode + postprocessSummarization
 *   translation           → seq2seqGreedyDecode + postprocessTranslation
 *   image-to-text         → seq2seqGreedyDecode + postprocessImageToText
 *   audio-to-audio        → normalizeWaveform + playAudio + postprocessAudioToAudio
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

// ---- Speech-to-Text: greedy CTC decoder ----

function emitGreedyDecode(ts: boolean): string {
  const t = ts;
  return `/**
 * Greedy CTC decoder: decode speech recognition logits.
 * Performs argmax at each timestep, then collapses repeated tokens and blanks.
 *
 * @param logits - Raw model logits (Float32Array)
 * @param numTimesteps - Number of time steps
 * @param vocabSize - Vocabulary size
 * @param blankIndex - CTC blank token index (default: 0)
 * @returns Array of token indices
 */
function greedyDecode(
  logits${t ? ': Float32Array' : ''},
  numTimesteps${t ? ': number' : ''},
  vocabSize${t ? ': number' : ''},
  blankIndex${t ? ': number' : ''} = 0
)${t ? ': number[]' : ''} {
  const tokens${t ? ': number[]' : ''} = [];
  let prev = -1;

  for (let t = 0; t < numTimesteps; t++) {
    let bestIdx = 0;
    let bestVal = logits[t * vocabSize];
    for (let v = 1; v < vocabSize; v++) {
      const val = logits[t * vocabSize + v];
      if (val > bestVal) {
        bestVal = val;
        bestIdx = v;
      }
    }

    // CTC collapsing: skip blank and repeated tokens
    if (bestIdx !== blankIndex && bestIdx !== prev) {
      tokens.push(bestIdx);
    }
    prev = bestIdx;
  }

  return tokens;
}`;
}

function emitPostprocessTranscript(ts: boolean): string {
  const t = ts;
  return `/**
 * Postprocess speech-to-text model output: decode logits → transcript.
 *
 * @param logits - Raw model logits (Float32Array)
 * @param numTimesteps - Number of time steps
 * @param vocabSize - Vocabulary size
 * @param vocab - Vocabulary array mapping indices to characters
 * @returns Transcribed text string
 */
function postprocessTranscript(
  logits${t ? ': Float32Array' : ''},
  numTimesteps${t ? ': number' : ''},
  vocabSize${t ? ': number' : ''},
  vocab${t ? ': string[]' : ''}
)${t ? ': string' : ''} {
  const indices = greedyDecode(logits, numTimesteps, vocabSize);
  return indices.map((i) => vocab[i]).join('');
}`;
}

// ---- Text-to-Speech: audio output ----

function emitPlayAudio(ts: boolean): string {
  const t = ts;
  return `/**
 * Play audio samples in the browser.
 * Creates an AudioContext and plays the samples as a buffer.
 *
 * @param samples - Audio samples (Float32Array, values in [-1, 1])
 * @param sampleRate - Sample rate in Hz (default: 22050)
 * @returns Promise that resolves when playback ends
 */
function playAudio(
  samples${t ? ': Float32Array' : ''},
  sampleRate${t ? ': number' : ''} = 22050
)${t ? ': Promise<void>' : ''} {
  return new Promise((resolve) => {
    const ctx = new AudioContext({ sampleRate });
    const buffer = ctx.createBuffer(1, samples.length, sampleRate);
    buffer.copyToChannel(samples, 0);

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    source.onended = () => resolve();
    source.start();
  });
}`;
}

function emitPostprocessAudio(ts: boolean): string {
  const t = ts;
  return `/**
 * Postprocess text-to-speech model output: clamp samples for safe playback.
 *
 * @param output - Raw model output (Float32Array)
 * @returns Clamped audio samples ready for playback
 */
function postprocessAudio(output${t ? ': Float32Array' : ''})${t ? ': Float32Array' : ''} {
  const samples = new Float32Array(output.length);
  for (let i = 0; i < output.length; i++) {
    samples[i] = Math.max(-1, Math.min(1, output[i]));
  }
  return samples;
}`;
}

// ---- Zero-Shot Classification: entailment scoring ----

function emitPostprocessZeroShot(ts: boolean): string {
  const t = ts;
  return `/**
 * Postprocess zero-shot classification results.
 * Takes entailment scores for each candidate label and normalizes with softmax.
 *
 * @param scores - Array of entailment scores, one per candidate label
 * @param labels - Array of candidate label strings
 * @returns Sorted results: { label, score } pairs, descending by score
 */
function postprocessZeroShot(
  scores${t ? ': number[]' : ''},
  labels${t ? ': string[]' : ''}
)${t ? ': Array<{ label: string; score: number }>' : ''} {
  // Softmax over scores
  const maxScore = Math.max(...scores);
  const expScores = scores.map((s) => Math.exp(s - maxScore));
  const sumExp = expScores.reduce((a, b) => a + b, 0);
  const probs = expScores.map((e) => e / sumExp);

  // Pair labels with probabilities and sort descending
  const results = labels.map((label, i) => ({ label, score: probs[i] }));
  results.sort((a, b) => b.score - a.score);
  return results;
}`;
}

// ---- Text Generation: greedy token sampling ----

function emitSampleNextToken(ts: boolean): string {
  const t = ts;
  return `/**
 * Sample the next token from logits using greedy decoding (argmax).
 *
 * @param logits - Raw model output logits for the last position
 * @returns Token ID with the highest probability
 */
function sampleNextToken(logits${t ? ': Float32Array | number[]' : ''})${t ? ': number' : ''} {
  let bestIdx = 0;
  let bestVal = logits[0];
  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > bestVal) {
      bestVal = logits[i];
      bestIdx = i;
    }
  }
  return bestIdx;
}`;
}

function emitPostprocessGeneration(ts: boolean): string {
  const t = ts;
  return `/**
 * Postprocess text generation: extract next-token logits from model output.
 * For autoregressive models, the last position's logits predict the next token.
 *
 * @param output - Raw model output (Float32Array), shape [1, seqLen, vocabSize]
 * @param seqLen - Sequence length
 * @param vocabSize - Vocabulary size
 * @returns Logits for the last position (Float32Array of size vocabSize)
 */
function postprocessGeneration(
  output${t ? ': Float32Array' : ''},
  seqLen${t ? ': number' : ''},
  vocabSize${t ? ': number' : ''}
)${t ? ': Float32Array' : ''} {
  const lastOffset = (seqLen - 1) * vocabSize;
  return output.slice(lastOffset, lastOffset + vocabSize);
}`;
}

// ---- Fill-Mask: softmax + topK over masked position ----

function emitPostprocessFillMask(ts: boolean): string {
  const t = ts;
  return `/**
 * Postprocess fill-mask output: get top predictions for the masked position.
 * Applies softmax over the logits at the masked token index, then returns
 * the top-k token indices with their probabilities.
 *
 * @param logits - Model output logits for the masked position (Float32Array)
 * @param k - Number of top predictions to return (default: 5)
 * @returns Top-k token indices and their probabilities
 */
function postprocessFillMask(
  logits${t ? ': Float32Array' : ''},
  k${t ? ': number' : ''} = 5
)${t ? ': { indices: number[]; values: number[] }' : ''} {
  const probs = softmax(logits);
  return topK(probs, k);
}`;
}

// ---- Sentence Similarity: cosine similarity ----

function emitCosineSimilarity(ts: boolean): string {
  const t = ts;
  return `/**
 * Compute cosine similarity between two embedding vectors.
 * Returns a value in [-1, 1] where 1 = identical direction.
 */
function cosineSimilarity(
  a${t ? ': Float32Array' : ''},
  b${t ? ': Float32Array' : ''}
)${t ? ': number' : ''} {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom > 0 ? dot / denom : 0;
}`;
}

function emitPostprocessSimilarity(ts: boolean): string {
  const t = ts;
  return `/**
 * Postprocess sentence-similarity: compute similarity between two embeddings.
 * Returns cosine similarity in [-1, 1].
 *
 * @param embeddingA - First sentence embedding (Float32Array)
 * @param embeddingB - Second sentence embedding (Float32Array)
 * @returns Cosine similarity score
 */
function postprocessSimilarity(
  embeddingA${t ? ': Float32Array' : ''},
  embeddingB${t ? ': Float32Array' : ''}
)${t ? ': number' : ''} {
  return cosineSimilarity(embeddingA, embeddingB);
}`;
}

// ---- Depth Estimation: normalize + colormap visualization ----

function emitDepthNormalize(ts: boolean): string {
  const t = ts;
  return `/**
 * Normalize raw depth map to [0, 1] using min-max scaling.
 *
 * @param depthMap - Raw depth predictions (Float32Array)
 * @returns Normalized depth values in [0, 1]
 */
function depthNormalize(depthMap${t ? ': Float32Array' : ''})${t ? ': Float32Array' : ''} {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < depthMap.length; i++) {
    if (depthMap[i] < min) min = depthMap[i];
    if (depthMap[i] > max) max = depthMap[i];
  }

  const range = max - min;
  const out = new Float32Array(depthMap.length);
  for (let i = 0; i < depthMap.length; i++) {
    out[i] = range > 0 ? (depthMap[i] - min) / range : 0;
  }
  return out;
}`;
}

function emitDepthToColormap(ts: boolean): string {
  const t = ts;
  return `/**
 * Convert normalized depth map [0,1] to RGBA colormap for visualization.
 * Uses a jet-like colormap: blue (near) → green → red (far).
 *
 * @param depth - Normalized depth values in [0, 1]
 * @param width - Image width
 * @param height - Image height
 * @returns RGBA pixel data (Uint8ClampedArray) for canvas rendering
 */
function depthToColormap(
  depth${t ? ': Float32Array' : ''},
  width${t ? ': number' : ''},
  height${t ? ': number' : ''}
)${t ? ': Uint8ClampedArray' : ''} {
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < depth.length; i++) {
    const d = depth[i];
    const idx = i * 4;
    let r, g, b;
    if (d < 0.25) {
      r = 0; g = Math.round(d * 4 * 255); b = 255;
    } else if (d < 0.5) {
      r = 0; g = 255; b = Math.round((1 - (d - 0.25) * 4) * 255);
    } else if (d < 0.75) {
      r = Math.round((d - 0.5) * 4 * 255); g = 255; b = 0;
    } else {
      r = 255; g = Math.round((1 - (d - 0.75) * 4) * 255); b = 0;
    }
    rgba[idx] = r;
    rgba[idx + 1] = g;
    rgba[idx + 2] = b;
    rgba[idx + 3] = 255;
  }
  return rgba;
}`;
}

function emitPostprocessDepth(ts: boolean): string {
  const t = ts;
  return `/**
 * Postprocess depth estimation model output.
 * Normalizes the raw depth map for use and visualization.
 *
 * @param output - Raw model output (Float32Array)
 * @returns Normalized depth map in [0, 1]
 */
function postprocessDepth(output${t ? ': Float32Array' : ''})${t ? ': Float32Array' : ''} {
  return depthNormalize(output);
}`;
}

// ---- Token Classification (NER): per-token argmax + IOB span extraction ----

function emitTokenArgmax(ts: boolean): string {
  const t = ts;
  return `/**
 * Argmax over per-token logits to get predicted label index per token.
 *
 * @param logits - Model output (Float32Array), shape [1, seqLen, numLabels]
 * @param seqLen - Number of tokens
 * @param numLabels - Number of label classes
 * @returns Array of label indices, one per token
 */
function tokenArgmax(
  logits${t ? ': Float32Array' : ''},
  seqLen${t ? ': number' : ''},
  numLabels${t ? ': number' : ''}
)${t ? ': number[]' : ''} {
  const labels${t ? ': number[]' : ''} = [];
  for (let t = 0; t < seqLen; t++) {
    let bestIdx = 0;
    let bestVal = logits[t * numLabels];
    for (let l = 1; l < numLabels; l++) {
      const val = logits[t * numLabels + l];
      if (val > bestVal) {
        bestVal = val;
        bestIdx = l;
      }
    }
    labels.push(bestIdx);
  }
  return labels;
}`;
}

function emitExtractSpans(ts: boolean): string {
  const t = ts;
  const spanType = t ? ': Array<{ label: number; start: number; end: number }>' : '';
  return `/**
 * Extract contiguous entity spans from IOB-style label indices.
 * Groups consecutive tokens with the same non-O label into spans.
 * Label index 0 is treated as "O" (outside any entity).
 *
 * @param labels - Per-token label indices
 * @returns Array of spans with label index, start token, end token (exclusive)
 */
function extractSpans(labels${t ? ': number[]' : ''})${spanType} {
  const spans${t ? ': Array<{ label: number; start: number; end: number }>' : ''} = [];
  let current${t ? ': { label: number; start: number; end: number } | null' : ''} = null;

  for (let i = 0; i < labels.length; i++) {
    const label = labels[i];
    if (label === 0) {
      if (current) { spans.push(current); current = null; }
    } else if (current && current.label === label) {
      current.end = i + 1;
    } else {
      if (current) spans.push(current);
      current = { label, start: i, end: i + 1 };
    }
  }
  if (current) spans.push(current);
  return spans;
}`;
}

function emitPostprocessTokenClassification(ts: boolean): string {
  const t = ts;
  return `/**
 * Postprocess token classification (NER) output.
 * Performs argmax per token, then extracts entity spans.
 *
 * @param logits - Raw model output (Float32Array), shape [1, seqLen, numLabels]
 * @param seqLen - Number of tokens
 * @param numLabels - Number of label classes
 * @returns Entity spans with label index and token positions
 */
function postprocessTokenClassification(
  logits${t ? ': Float32Array' : ''},
  seqLen${t ? ': number' : ''},
  numLabels${t ? ': number' : ''}
)${t ? ': Array<{ label: number; start: number; end: number }>' : ''} {
  const labels = tokenArgmax(logits, seqLen, numLabels);
  return extractSpans(labels);
}`;
}

// ---- Question Answering: start/end logit span extraction ----

function emitPostprocessQA(ts: boolean): string {
  const t = ts;
  return `/**
 * Postprocess extractive question answering output.
 * Finds the best answer span from start and end logits.
 *
 * @param startLogits - Start position logits (Float32Array), one per token
 * @param endLogits - End position logits (Float32Array), one per token
 * @param maxAnswerLen - Maximum answer length in tokens (default: 15)
 * @returns Best span: { startIndex, endIndex, score }
 */
function postprocessQA(
  startLogits${t ? ': Float32Array' : ''},
  endLogits${t ? ': Float32Array' : ''},
  maxAnswerLen${t ? ': number' : ''} = 15
)${t ? ': { startIndex: number; endIndex: number; score: number }' : ''} {
  const len = startLogits.length;
  let bestScore = -Infinity;
  let bestStart = 0;
  let bestEnd = 0;

  for (let s = 0; s < len; s++) {
    for (let e = s; e < Math.min(s + maxAnswerLen, len); e++) {
      const score = startLogits[s] + endLogits[e];
      if (score > bestScore) {
        bestScore = score;
        bestStart = s;
        bestEnd = e;
      }
    }
  }

  return { startIndex: bestStart, endIndex: bestEnd, score: bestScore };
}`;
}

// ---- Summarization / Translation: shared seq2seq greedy decode ----

function emitSeq2SeqGreedyDecode(ts: boolean): string {
  const t = ts;
  return `/**
 * Greedy decode for seq2seq models (summarization, translation).
 * Extracts the most likely token at each output position via argmax.
 *
 * @param logits - Decoder output logits (Float32Array), shape [1, seqLen, vocabSize]
 * @param seqLen - Output sequence length
 * @param vocabSize - Vocabulary size
 * @param eosTokenId - End-of-sequence token ID (stops decoding, default: 1)
 * @returns Array of decoded token IDs
 */
function seq2seqGreedyDecode(
  logits${t ? ': Float32Array' : ''},
  seqLen${t ? ': number' : ''},
  vocabSize${t ? ': number' : ''},
  eosTokenId${t ? ': number' : ''} = 1
)${t ? ': number[]' : ''} {
  const tokens${t ? ': number[]' : ''} = [];

  for (let t = 0; t < seqLen; t++) {
    const offset = t * vocabSize;
    let bestIdx = 0;
    let bestVal = logits[offset];
    for (let v = 1; v < vocabSize; v++) {
      const val = logits[offset + v];
      if (val > bestVal) {
        bestVal = val;
        bestIdx = v;
      }
    }
    if (bestIdx === eosTokenId) break;
    tokens.push(bestIdx);
  }

  return tokens;
}`;
}

function emitPostprocessSummarization(ts: boolean): string {
  const t = ts;
  return `/**
 * Postprocess summarization model output.
 * Greedy-decodes the output sequence from decoder logits.
 *
 * @param logits - Decoder output logits (Float32Array), shape [1, seqLen, vocabSize]
 * @param seqLen - Output sequence length
 * @param vocabSize - Vocabulary size
 * @returns Decoded token IDs for the summary
 */
function postprocessSummarization(
  logits${t ? ': Float32Array' : ''},
  seqLen${t ? ': number' : ''},
  vocabSize${t ? ': number' : ''}
)${t ? ': number[]' : ''} {
  return seq2seqGreedyDecode(logits, seqLen, vocabSize);
}`;
}

function emitPostprocessTranslation(ts: boolean): string {
  const t = ts;
  return `/**
 * Postprocess translation model output.
 * Greedy-decodes the output sequence from decoder logits.
 *
 * @param logits - Decoder output logits (Float32Array), shape [1, seqLen, vocabSize]
 * @param seqLen - Output sequence length
 * @param vocabSize - Vocabulary size
 * @returns Decoded token IDs for the translation
 */
function postprocessTranslation(
  logits${t ? ': Float32Array' : ''},
  seqLen${t ? ': number' : ''},
  vocabSize${t ? ': number' : ''}
)${t ? ': number[]' : ''} {
  return seq2seqGreedyDecode(logits, seqLen, vocabSize);
}`;
}

// ---- Image-to-Text: encoder-decoder captioning ----

function emitPostprocessImageToText(ts: boolean): string {
  const t = ts;
  return `/**
 * Postprocess image-to-text (captioning) model output.
 * Greedy-decodes the output sequence from decoder logits.
 *
 * @param logits - Decoder output logits (Float32Array), shape [1, seqLen, vocabSize]
 * @param seqLen - Output sequence length
 * @param vocabSize - Vocabulary size
 * @returns Decoded token IDs for the caption
 */
function postprocessImageToText(
  logits${t ? ': Float32Array' : ''},
  seqLen${t ? ': number' : ''},
  vocabSize${t ? ': number' : ''}
)${t ? ': number[]' : ''} {
  return seq2seqGreedyDecode(logits, seqLen, vocabSize);
}`;
}

// ---- Audio-to-Audio: waveform normalization + playback ----

function emitNormalizeWaveform(ts: boolean): string {
  const t = ts;
  return `/**
 * Normalize audio waveform to [-1, 1] range using peak normalization.
 * Prevents clipping while preserving dynamics.
 *
 * @param samples - Raw audio samples (Float32Array)
 * @returns Normalized samples in [-1, 1]
 */
function normalizeWaveform(samples${t ? ': Float32Array' : ''})${t ? ': Float32Array' : ''} {
  let peak = 0;
  for (let i = 0; i < samples.length; i++) {
    const abs = Math.abs(samples[i]);
    if (abs > peak) peak = abs;
  }

  if (peak === 0) return samples;

  const out = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    out[i] = samples[i] / peak;
  }
  return out;
}`;
}

function emitPostprocessAudioToAudio(ts: boolean): string {
  const t = ts;
  return `/**
 * Postprocess audio-to-audio model output.
 * Normalizes the waveform for safe playback.
 *
 * @param output - Raw model output waveform (Float32Array)
 * @returns Normalized audio samples ready for playback
 */
function postprocessAudioToAudio(output${t ? ': Float32Array' : ''})${t ? ': Float32Array' : ''} {
  return normalizeWaveform(output);
}`;
}

// ---- Block emitter dispatch ----

/**
 * Emit the postprocess CodeBlock for a given config.
 *
 * Dispatches by task:
 *   image-classification     → softmax + topK + postprocessResults
 *   object-detection         → BoundingBox type + iou + nms + decodeDetections + postprocessDetections
 *   image-segmentation       → argmaxMask + postprocessSegmentation
 *   feature-extraction       → postprocessEmbeddings
 *   speech-to-text           → greedyDecode + postprocessTranscript
 *   text-to-speech           → playAudio + postprocessAudio
 *   zero-shot-classification → postprocessZeroShot
 *   text-generation          → sampleNextToken + postprocessGeneration
 *   fill-mask                → softmax + topK + postprocessFillMask
 *   sentence-similarity      → cosineSimilarity + postprocessSimilarity
 *   depth-estimation         → depthNormalize + depthToColormap + postprocessDepth
 *   token-classification     → tokenArgmax + extractSpans + postprocessTokenClassification
 *   question-answering       → postprocessQA
 *   summarization            → seq2seqGreedyDecode + postprocessSummarization
 *   translation              → seq2seqGreedyDecode + postprocessTranslation
 *   image-to-text            → seq2seqGreedyDecode + postprocessImageToText
 *   audio-to-audio           → normalizeWaveform + playAudio + postprocessAudioToAudio
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

    case 'zero-shot-classification':
      parts.push(emitPostprocessZeroShot(ts));
      exports.push('postprocessZeroShot');
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

    case 'speech-to-text':
      parts.push(emitGreedyDecode(ts));
      parts.push(emitPostprocessTranscript(ts));
      exports.push('greedyDecode', 'postprocessTranscript');
      break;

    case 'text-to-speech':
      parts.push(emitPlayAudio(ts));
      parts.push(emitPostprocessAudio(ts));
      exports.push('postprocessAudio', 'playAudio');
      break;

    case 'text-generation':
      parts.push(emitSampleNextToken(ts));
      parts.push(emitPostprocessGeneration(ts));
      exports.push('sampleNextToken', 'postprocessGeneration');
      break;

    case 'fill-mask':
      parts.push(emitSoftmax(ts));
      parts.push(emitTopK(ts));
      parts.push(emitPostprocessFillMask(ts));
      exports.push('softmax', 'topK', 'postprocessFillMask');
      break;

    case 'sentence-similarity':
      parts.push(emitCosineSimilarity(ts));
      parts.push(emitPostprocessSimilarity(ts));
      exports.push('cosineSimilarity', 'postprocessSimilarity');
      break;

    case 'depth-estimation':
      parts.push(emitDepthNormalize(ts));
      parts.push(emitDepthToColormap(ts));
      parts.push(emitPostprocessDepth(ts));
      exports.push('depthNormalize', 'depthToColormap', 'postprocessDepth');
      break;

    case 'token-classification':
      parts.push(emitTokenArgmax(ts));
      parts.push(emitExtractSpans(ts));
      parts.push(emitPostprocessTokenClassification(ts));
      exports.push('tokenArgmax', 'extractSpans', 'postprocessTokenClassification');
      break;

    case 'question-answering':
      parts.push(emitPostprocessQA(ts));
      exports.push('postprocessQA');
      break;

    case 'summarization':
      parts.push(emitSeq2SeqGreedyDecode(ts));
      parts.push(emitPostprocessSummarization(ts));
      exports.push('seq2seqGreedyDecode', 'postprocessSummarization');
      break;

    case 'translation':
      parts.push(emitSeq2SeqGreedyDecode(ts));
      parts.push(emitPostprocessTranslation(ts));
      exports.push('seq2seqGreedyDecode', 'postprocessTranslation');
      break;

    case 'image-to-text':
      parts.push(emitSeq2SeqGreedyDecode(ts));
      parts.push(emitPostprocessImageToText(ts));
      exports.push('seq2seqGreedyDecode', 'postprocessImageToText');
      break;

    case 'audio-to-audio':
      parts.push(emitNormalizeWaveform(ts));
      parts.push(emitPlayAudio(ts));
      parts.push(emitPostprocessAudioToAudio(ts));
      exports.push('normalizeWaveform', 'playAudio', 'postprocessAudioToAudio');
      break;

    default:
      // Tasks without postprocessing
      break;
  }

  return {
    id: 'postprocess',
    code: parts.join('\n\n'),
    imports: [],
    exports,
  };
}
