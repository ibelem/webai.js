/**
 * HTML framework emitter (Layer 2).
 *
 * Produces a single index.html with inline CSS + JS.
 * Uses CDN import for onnxruntime-web. No build step required.
 * Just open in browser with a local server (for module/CORS).
 *
 * Dispatches on config.input × config.task to generate the right UI:
 *   file + classification → drop zone + bar chart
 *   file + detection      → drop zone + canvas bounding boxes
 *   file + segmentation   → drop zone + canvas mask overlay
 *   file + extraction     → drop zone + embedding info
 *   camera/screen         → video feed + canvas overlay + inference loop
 *   video                 → video file + canvas overlay + inference loop
 *   mic                   → audio capture + results
 */

import type { ResolvedConfig } from '@webai/core';
import type { CodeBlock, GeneratedFile } from '../types.js';
import {
  emitDesignSystemCSS,
  emitAppCSS,
  stripImports,
  findBlock,
  emitReadme,
  getTaskLabel,
  getEngineLabel,
  getModelPath,
} from './shared.js';

const ORT_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/esm/ort.min.mjs';

// ---- Helpers ----

function isClassificationTask(task: string): boolean {
  return task === 'image-classification' || task === 'audio-classification' || task === 'text-classification';
}

/** Concatenate Layer 1 block code into a script preamble */
function emitBlockCode(config: ResolvedConfig, blocks: CodeBlock[]): string {
  const inputBlock = findBlock(blocks, 'input');
  const preprocessBlock = findBlock(blocks, 'preprocess');
  const inferenceBlock = findBlock(blocks, 'inference');
  const postprocessBlock = findBlock(blocks, 'postprocess');
  const opfsBlock = findBlock(blocks, 'opfs-cache');

  const sections: string[] = [];

  // Engine-specific import (only ORT needs a CDN import for HTML)
  if (config.engine === 'ort') {
    sections.push(`import * as ort from '${ORT_CDN}';`);
  }

  // OPFS caching utilities (when offline mode enabled)
  if (opfsBlock?.code) {
    sections.push(`// --- OPFS Cache ---\n${opfsBlock.code}`);
  }

  if (inputBlock?.code) {
    sections.push(`// --- Input Capture ---\n${inputBlock.code}`);
  }
  sections.push(`// --- Preprocessing ---\n${preprocessBlock?.code ?? ''}`);
  if (inferenceBlock) {
    sections.push(`// --- Inference ---\n${stripImports(inferenceBlock.code)}`);
  }
  sections.push(`// --- Postprocessing ---\n${postprocessBlock?.code ?? ''}`);

  return sections.join('\n\n');
}

// ---- Audio task helpers ----

function isAudioTask(task: string): boolean {
  return task === 'audio-classification' || task === 'speech-to-text' || task === 'text-to-speech';
}

// ---- Color palette for detection/segmentation ----

function emitColorPalette(): string {
  return `const COLORS = [
  [56, 189, 248],  // sky
  [249, 115, 22],  // orange
  [34, 197, 94],   // green
  [168, 85, 247],  // purple
  [251, 191, 36],  // amber
  [239, 68, 68],   // red
  [20, 184, 166],  // teal
  [236, 72, 153],  // pink
  [99, 102, 241],  // indigo
  [163, 230, 53],  // lime
];`;
}

// ---- File + Classification script ----

function emitFileClassificationScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  return `${emitBlockCode(config, blocks)}

// --- Application ---
const MODEL_PATH = '${getModelPath(config, '.')}';
let session = null;

async function init() {
  updateStatus('Loading model...');
  try {
    session = await createSession(MODEL_PATH);
    updateStatus('${config.modelName} \\u00b7 Ready');
  } catch (e) {
    updateStatus('Failed to load model');
    console.error('Model load error:', e);
  }
}

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const previewImage = document.getElementById('previewImage');
const resultsDiv = document.getElementById('results');
const changeBtn = document.getElementById('changeBtn');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fileInput.click();
  }
});

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (file) handleFile(file);
});

changeBtn.addEventListener('click', () => {
  preview.hidden = true;
  dropZone.hidden = false;
  resultsDiv.innerHTML = '';
  fileInput.value = '';
});

async function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    resultsDiv.textContent = 'Unsupported file type. Try JPG, PNG, or WebP.';
    return;
  }

  const url = URL.createObjectURL(file);
  previewImage.src = url;
  preview.hidden = false;
  dropZone.hidden = true;

  await new Promise((resolve) => { previewImage.onload = resolve; });

  const canvas = document.createElement('canvas');
  canvas.width = previewImage.naturalWidth;
  canvas.height = previewImage.naturalHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(previewImage, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  if (!session) {
    resultsDiv.textContent = 'Model not loaded yet. Please wait.';
    return;
  }

  updateStatus('${config.modelName} \\u00b7 Processing...');
  const start = performance.now();

  const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
  const output = await runInference(session, inputTensor);
  const results = postprocessResults(output);

  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus('${config.modelName} \\u00b7 ' + elapsed + 'ms \\u00b7 ' + getBackendLabel(session));

  renderResults(results);
  URL.revokeObjectURL(url);
}

function renderResults(results) {
  resultsDiv.innerHTML = '';
  const maxValue = results.values[0] || 1;

  for (let i = 0; i < results.indices.length; i++) {
    const pct = (results.values[i] * 100).toFixed(1);
    if (results.values[i] < 0.01) continue;

    const row = document.createElement('div');
    row.className = 'result-row' + (i === 0 ? ' top-result' : '');
    row.setAttribute('tabindex', '0');
    row.setAttribute('aria-label', 'Class ' + results.indices[i] + ': ' + pct + ' percent');

    row.innerHTML =
      '<span class="result-label">Class ' + results.indices[i] + '</span>' +
      '<div class="result-bar-container"><div class="result-bar" style="width:' +
      ((results.values[i] / maxValue) * 100) + '%"></div></div>' +
      '<span class="result-pct">' + pct + '%</span>';

    resultsDiv.appendChild(row);
  }
}

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

init();`;
}

// ---- File + Detection script ----

function emitFileDetectionScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  const outputShape = config.modelMeta.outputs[0]?.shape ?? [1, 84, 8400];
  const numAttributes = outputShape[1] ?? 84;
  const numAnchors = outputShape[2] ?? 8400;

  return `${emitBlockCode(config, blocks)}

// --- Application ---
${emitColorPalette()}

const MODEL_PATH = '${getModelPath(config, '.')}';
const NUM_ATTRIBUTES = ${numAttributes};
const NUM_ANCHORS = ${numAnchors};
let session = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

async function init() {
  updateStatus('Loading model...');
  try {
    session = await createSession(MODEL_PATH);
    updateStatus('${config.modelName} \\u00b7 Ready');
  } catch (e) {
    updateStatus('Failed to load model');
    console.error('Model load error:', e);
  }
}

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const previewImage = document.getElementById('previewImage');
const overlay = document.getElementById('overlay');
const resultsDiv = document.getElementById('results');
const changeBtn = document.getElementById('changeBtn');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => { dropZone.classList.remove('drag-over'); });
dropZone.addEventListener('drop', (e) => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
fileInput.addEventListener('change', () => { const file = fileInput.files[0]; if (file) handleFile(file); });
changeBtn.addEventListener('click', () => {
  preview.hidden = true; dropZone.hidden = false; resultsDiv.innerHTML = '';
  fileInput.value = '';
  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, overlay.width, overlay.height);
});

async function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    resultsDiv.textContent = 'Unsupported file type. Try JPG, PNG, or WebP.';
    return;
  }

  const url = URL.createObjectURL(file);
  previewImage.src = url;
  preview.hidden = false;
  dropZone.hidden = true;

  await new Promise((resolve) => { previewImage.onload = resolve; });

  overlay.width = previewImage.naturalWidth;
  overlay.height = previewImage.naturalHeight;

  const canvas = document.createElement('canvas');
  canvas.width = previewImage.naturalWidth;
  canvas.height = previewImage.naturalHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(previewImage, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  if (!session) { resultsDiv.textContent = 'Model not loaded yet.'; return; }

  updateStatus('${config.modelName} \\u00b7 Processing...');
  const start = performance.now();

  const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
  const output = await runInference(session, inputTensor);
  const boxes = postprocessDetections(output, NUM_ANCHORS, NUM_ATTRIBUTES);

  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus('${config.modelName} \\u00b7 ' + elapsed + 'ms \\u00b7 ' + getBackendLabel(session));

  renderDetections(boxes, previewImage.naturalWidth, previewImage.naturalHeight);
  URL.revokeObjectURL(url);
}

function renderDetections(boxes, imgW, imgH) {
  const modelSize = ${config.preprocess.imageSize};
  const scaleX = imgW / modelSize;
  const scaleY = imgH / modelSize;

  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  resultsDiv.innerHTML = '';

  for (const box of boxes) {
    const c = COLORS[box.classIndex % COLORS.length];
    const color = 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')';

    const x = box.x * scaleX;
    const y = box.y * scaleY;
    const w = box.width * scaleX;
    const h = box.height * scaleY;

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);

    const label = 'Class ' + box.classIndex + ' (' + (box.score * 100).toFixed(0) + '%)';
    ctx.font = '14px system-ui, sans-serif';
    ctx.fillStyle = color;
    const textW = ctx.measureText(label).width;
    ctx.fillRect(x, y - 20, textW + 8, 20);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, x + 4, y - 5);

    const row = document.createElement('div');
    row.className = 'result-row';
    row.setAttribute('tabindex', '0');
    row.setAttribute('aria-label', label);
    row.innerHTML = '<span class="result-label">' + label + '</span>';
    resultsDiv.appendChild(row);
  }

  if (boxes.length === 0) {
    resultsDiv.textContent = 'No detections found.';
  }
}

init();`;
}

// ---- File + Segmentation script ----

function emitFileSegmentationScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  const outputShape = config.modelMeta.outputs[0]?.shape ?? [1, 21, 512, 512];
  const numClasses = outputShape[1] ?? 21;
  const maskH = outputShape[2] ?? 512;
  const maskW = outputShape[3] ?? 512;

  return `${emitBlockCode(config, blocks)}

// --- Application ---
${emitColorPalette()}

const MODEL_PATH = '${getModelPath(config, '.')}';
const NUM_CLASSES = ${numClasses};
const MASK_H = ${maskH};
const MASK_W = ${maskW};
let session = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

async function init() {
  updateStatus('Loading model...');
  try {
    session = await createSession(MODEL_PATH);
    updateStatus('${config.modelName} \\u00b7 Ready');
  } catch (e) {
    updateStatus('Failed to load model');
    console.error('Model load error:', e);
  }
}

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const previewImage = document.getElementById('previewImage');
const overlay = document.getElementById('overlay');
const resultsDiv = document.getElementById('results');
const changeBtn = document.getElementById('changeBtn');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => { dropZone.classList.remove('drag-over'); });
dropZone.addEventListener('drop', (e) => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
fileInput.addEventListener('change', () => { const file = fileInput.files[0]; if (file) handleFile(file); });
changeBtn.addEventListener('click', () => {
  preview.hidden = true; dropZone.hidden = false; resultsDiv.innerHTML = '';
  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  fileInput.value = '';
});

async function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    resultsDiv.textContent = 'Unsupported file type. Try JPG, PNG, or WebP.';
    return;
  }

  const url = URL.createObjectURL(file);
  previewImage.src = url;
  preview.hidden = false;
  dropZone.hidden = true;

  await new Promise((resolve) => { previewImage.onload = resolve; });

  overlay.width = previewImage.naturalWidth;
  overlay.height = previewImage.naturalHeight;

  const canvas = document.createElement('canvas');
  canvas.width = previewImage.naturalWidth;
  canvas.height = previewImage.naturalHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(previewImage, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  if (!session) { resultsDiv.textContent = 'Model not loaded yet.'; return; }

  updateStatus('${config.modelName} \\u00b7 Processing...');
  const start = performance.now();

  const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
  const output = await runInference(session, inputTensor);
  const mask = postprocessSegmentation(output, NUM_CLASSES, MASK_H, MASK_W);

  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus('${config.modelName} \\u00b7 ' + elapsed + 'ms \\u00b7 ' + getBackendLabel(session));

  renderMask(mask, previewImage.naturalWidth, previewImage.naturalHeight);
  URL.revokeObjectURL(url);
}

function renderMask(mask, displayW, displayH) {
  // Draw mask at native resolution to offscreen canvas, then scale
  const maskCanvas = document.createElement('canvas');
  maskCanvas.width = MASK_W;
  maskCanvas.height = MASK_H;
  const maskCtx = maskCanvas.getContext('2d');
  const maskImage = maskCtx.createImageData(MASK_W, MASK_H);

  const classesFound = new Set();
  for (let i = 0; i < mask.length; i++) {
    const cls = mask[i];
    classesFound.add(cls);
    const c = COLORS[cls % COLORS.length];
    maskImage.data[i * 4] = c[0];
    maskImage.data[i * 4 + 1] = c[1];
    maskImage.data[i * 4 + 2] = c[2];
    maskImage.data[i * 4 + 3] = 128; // semi-transparent
  }
  maskCtx.putImageData(maskImage, 0, 0);

  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  ctx.drawImage(maskCanvas, 0, 0, displayW, displayH);

  resultsDiv.innerHTML = '';
  for (const cls of classesFound) {
    const c = COLORS[cls % COLORS.length];
    const row = document.createElement('div');
    row.className = 'result-row';
    row.innerHTML =
      '<span class="color-swatch" style="background:rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')"></span>' +
      '<span class="result-label">Class ' + cls + '</span>';
    resultsDiv.appendChild(row);
  }
}

init();`;
}

// ---- File + Feature Extraction script ----

function emitFileFeatureExtractionScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  return `${emitBlockCode(config, blocks)}

// --- Application ---
const MODEL_PATH = '${getModelPath(config, '.')}';
let session = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

async function init() {
  updateStatus('Loading model...');
  try {
    session = await createSession(MODEL_PATH);
    updateStatus('${config.modelName} \\u00b7 Ready');
  } catch (e) {
    updateStatus('Failed to load model');
    console.error('Model load error:', e);
  }
}

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const previewImage = document.getElementById('previewImage');
const resultsDiv = document.getElementById('results');
const changeBtn = document.getElementById('changeBtn');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => { dropZone.classList.remove('drag-over'); });
dropZone.addEventListener('drop', (e) => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
fileInput.addEventListener('change', () => { const file = fileInput.files[0]; if (file) handleFile(file); });
changeBtn.addEventListener('click', () => {
  preview.hidden = true; dropZone.hidden = false; resultsDiv.innerHTML = '';
  fileInput.value = '';
});

async function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    resultsDiv.textContent = 'Unsupported file type. Try JPG, PNG, or WebP.';
    return;
  }

  const url = URL.createObjectURL(file);
  previewImage.src = url;
  preview.hidden = false;
  dropZone.hidden = true;

  await new Promise((resolve) => { previewImage.onload = resolve; });

  const canvas = document.createElement('canvas');
  canvas.width = previewImage.naturalWidth;
  canvas.height = previewImage.naturalHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(previewImage, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  if (!session) { resultsDiv.textContent = 'Model not loaded yet.'; return; }

  updateStatus('${config.modelName} \\u00b7 Processing...');
  const start = performance.now();

  const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
  const output = await runInference(session, inputTensor);
  const embedding = postprocessEmbeddings(output);

  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus('${config.modelName} \\u00b7 ' + elapsed + 'ms \\u00b7 ' + getBackendLabel(session));

  renderEmbedding(embedding);
  URL.revokeObjectURL(url);
}

function renderEmbedding(embedding) {
  let norm = 0;
  for (let i = 0; i < embedding.length; i++) {
    norm += embedding[i] * embedding[i];
  }
  norm = Math.sqrt(norm);

  const first5 = Array.from(embedding.slice(0, 5)).map(v => v.toFixed(4)).join(', ');

  resultsDiv.innerHTML =
    '<div class="embedding-info">' +
    '<p><strong>Dimensions:</strong> ' + embedding.length + '</p>' +
    '<p><strong>L2 Norm:</strong> ' + norm.toFixed(4) + '</p>' +
    '<p><strong>First 5 values:</strong> [' + first5 + ', ...]</p>' +
    '</div>';
}

init();`;
}

// ---- Camera / Screen realtime script ----

function emitRealtimeScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  const isScreen = config.input === 'screen';
  const startFn = isScreen ? 'startScreenCapture' : 'startCamera';
  const label = isScreen ? 'Screen Capture' : 'Camera';

  // Determine postprocess call based on task
  let processOutput: string;
  let renderCall: string;
  let extraCode = '';

  switch (config.task) {
    case 'object-detection': {
      const outputShape = config.modelMeta.outputs[0]?.shape ?? [1, 84, 8400];
      const numAttributes = outputShape[1] ?? 84;
      const numAnchors = outputShape[2] ?? 8400;
      processOutput = `const boxes = postprocessDetections(output, ${numAnchors}, ${numAttributes});`;
      renderCall = 'renderDetections(overlayCtx, boxes, video.videoWidth, video.videoHeight);';
      extraCode = `
${emitColorPalette()}

function renderDetections(ctx, boxes, videoW, videoH) {
  const modelSize = ${config.preprocess.imageSize};
  const scaleX = videoW / modelSize;
  const scaleY = videoH / modelSize;
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  for (const box of boxes) {
    const c = COLORS[box.classIndex % COLORS.length];
    const color = 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')';
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(box.x * scaleX, box.y * scaleY, box.width * scaleX, box.height * scaleY);
    ctx.font = '14px system-ui, sans-serif';
    ctx.fillStyle = color;
    const label = 'Class ' + box.classIndex + ' (' + (box.score * 100).toFixed(0) + '%)';
    const tw = ctx.measureText(label).width;
    ctx.fillRect(box.x * scaleX, box.y * scaleY - 20, tw + 8, 20);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, box.x * scaleX + 4, box.y * scaleY - 5);
  }
}`;
      break;
    }

    case 'image-segmentation': {
      const outputShape = config.modelMeta.outputs[0]?.shape ?? [1, 21, 512, 512];
      const numClasses = outputShape[1] ?? 21;
      const maskH = outputShape[2] ?? 512;
      const maskW = outputShape[3] ?? 512;
      processOutput = `const mask = postprocessSegmentation(output, ${numClasses}, ${maskH}, ${maskW});`;
      renderCall = 'renderMask(overlayCtx, mask, video.videoWidth, video.videoHeight);';
      extraCode = `
${emitColorPalette()}

function renderMask(ctx, mask, displayW, displayH) {
  const maskCanvas = document.createElement('canvas');
  maskCanvas.width = ${maskW};
  maskCanvas.height = ${maskH};
  const maskCtx = maskCanvas.getContext('2d');
  const maskImage = maskCtx.createImageData(${maskW}, ${maskH});
  for (let i = 0; i < mask.length; i++) {
    const c = COLORS[mask[i] % COLORS.length];
    maskImage.data[i * 4] = c[0];
    maskImage.data[i * 4 + 1] = c[1];
    maskImage.data[i * 4 + 2] = c[2];
    maskImage.data[i * 4 + 3] = 128;
  }
  maskCtx.putImageData(maskImage, 0, 0);
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.drawImage(maskCanvas, 0, 0, displayW, displayH);
}`;
      break;
    }

    default: {
      // Classification or other
      processOutput = 'const results = postprocessResults(output);';
      renderCall = `
    const label = 'Class ' + results.indices[0] + ' (' + (results.values[0] * 100).toFixed(1) + '%)';
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
    overlayCtx.font = 'bold 24px system-ui, sans-serif';
    overlayCtx.fillStyle = 'rgba(59, 130, 246, 0.85)';
    const tw = overlayCtx.measureText(label).width;
    overlayCtx.fillRect(8, 8, tw + 16, 36);
    overlayCtx.fillStyle = '#fff';
    overlayCtx.fillText(label, 16, 34);`;
      break;
    }
  }

  return `${emitBlockCode(config, blocks)}

// --- Application ---
${extraCode}

const MODEL_PATH = '${getModelPath(config, '.')}';
let session = null;
let currentStream = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

async function init() {
  updateStatus('Loading model...');
  try {
    session = await createSession(MODEL_PATH);
    updateStatus('${config.modelName} \\u00b7 Ready \\u00b7 Tap Start');
  } catch (e) {
    updateStatus('Failed to load model');
    console.error('Model load error:', e);
  }
}

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const overlayCtx = overlay.getContext('2d');
const startBtn = document.getElementById('startBtn');
const pauseBtn = document.getElementById('pauseBtn');
const permissionPrompt = document.getElementById('permissionPrompt');
const videoContainer = document.getElementById('videoContainer');

let loop = null;

startBtn.addEventListener('click', async () => {
  try {
    currentStream = await ${startFn}(video);
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;

    permissionPrompt.hidden = true;
    videoContainer.hidden = false;

    loop = createInferenceLoop({
      video,
      canvas: overlay,
      async onFrame(imageData) {
        const start = performance.now();
        const inputTensor = preprocessImage(imageData.data, imageData.width, imageData.height);
        const output = await runInference(session, inputTensor);
        ${processOutput}
        const elapsed = performance.now() - start;
        ${renderCall}
        return { result: null, elapsed };
      },
      onStatus(elapsed) {
        updateStatus('${config.modelName} \\u00b7 ' + elapsed.toFixed(1) + 'ms \\u00b7 ' + getBackendLabel(session));
      },
    });
    loop.start();
  } catch (e) {
    updateStatus('${label} access denied');
    console.error('${label} error:', e);
  }
});

pauseBtn.addEventListener('click', () => {
  if (loop) {
    loop.stop();
    loop = null;
    pauseBtn.textContent = '\\u25b6 Resume';
    pauseBtn.addEventListener('click', function resume() {
      pauseBtn.removeEventListener('click', resume);
      loop = createInferenceLoop({
        video, canvas: overlay,
        async onFrame(imageData) {
          const start = performance.now();
          const inputTensor = preprocessImage(imageData.data, imageData.width, imageData.height);
          const output = await runInference(session, inputTensor);
          ${processOutput}
          const elapsed = performance.now() - start;
          ${renderCall}
          return { result: null, elapsed };
        },
        onStatus(elapsed) {
          updateStatus('${config.modelName} \\u00b7 ' + elapsed.toFixed(1) + 'ms \\u00b7 ' + getBackendLabel(session));
        },
      });
      loop.start();
      pauseBtn.textContent = '\\u23f8 Pause';
    }, { once: true });
  }
});

init();`;
}

// ---- File + Audio Classification script ----

function emitFileAudioClassificationScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  return `${emitBlockCode(config, blocks)}

// --- Application ---
const MODEL_PATH = '${getModelPath(config, '.')}';
let session = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

async function init() {
  updateStatus('Loading model...');
  try {
    session = await createSession(MODEL_PATH);
    updateStatus('${config.modelName} \\u00b7 Ready');
  } catch (e) {
    updateStatus('Failed to load model');
    console.error('Model load error:', e);
  }
}

const fileInput = document.getElementById('fileInput');
const resultsDiv = document.getElementById('results');

fileInput.addEventListener('change', async () => {
  const file = fileInput.files[0];
  if (!file) return;

  if (!session) {
    resultsDiv.textContent = 'Model not loaded yet. Please wait.';
    return;
  }

  updateStatus('${config.modelName} \\u00b7 Decoding audio...');

  const arrayBuffer = await file.arrayBuffer();
  const audioCtx = new OfflineAudioContext(1, 1, 16000);
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

  // Resample to 16kHz mono
  const offlineCtx = new OfflineAudioContext(1, Math.ceil(audioBuffer.duration * 16000), 16000);
  const source = offlineCtx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(offlineCtx.destination);
  source.start();
  const resampled = await offlineCtx.startRendering();
  const samples = resampled.getChannelData(0);

  updateStatus('${config.modelName} \\u00b7 Processing...');
  const start = performance.now();

  const mel = melSpectrogram(samples, 16000, 512, 160, 40);
  const features = mfcc(mel.data, mel.numFrames, mel.numMelBands, 13);
  const output = await runInference(session, features);
  const results = postprocessResults(output);

  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus('${config.modelName} \\u00b7 ' + elapsed + 'ms \\u00b7 ' + getBackendLabel(session));

  renderResults(results);
});

function renderResults(results) {
  resultsDiv.innerHTML = '';
  const maxValue = results.values[0] || 1;

  for (let i = 0; i < results.indices.length; i++) {
    const pct = (results.values[i] * 100).toFixed(1);
    if (results.values[i] < 0.01) continue;

    const row = document.createElement('div');
    row.className = 'result-row' + (i === 0 ? ' top-result' : '');
    row.setAttribute('tabindex', '0');
    row.setAttribute('aria-label', 'Class ' + results.indices[i] + ': ' + pct + ' percent');

    row.innerHTML =
      '<span class="result-label">Class ' + results.indices[i] + '</span>' +
      '<div class="result-bar-container"><div class="result-bar" style="width:' +
      ((results.values[i] / maxValue) * 100) + '%"></div></div>' +
      '<span class="result-pct">' + pct + '%</span>';

    resultsDiv.appendChild(row);
  }
}

init();`;
}

// ---- File + Speech-to-Text script ----

function emitFileSpeechToTextScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  return `${emitBlockCode(config, blocks)}

// --- Application ---
const MODEL_PATH = '${getModelPath(config, '.')}';
const VOCAB = [' ', ...'abcdefghijklmnopqrstuvwxyz'.split(''), "'"];
let session = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

async function init() {
  updateStatus('Loading model...');
  try {
    session = await createSession(MODEL_PATH);
    updateStatus('${config.modelName} \\u00b7 Ready');
  } catch (e) {
    updateStatus('Failed to load model');
    console.error('Model load error:', e);
  }
}

const fileInput = document.getElementById('fileInput');
const transcript = document.getElementById('transcript');

fileInput.addEventListener('change', async () => {
  const file = fileInput.files[0];
  if (!file) return;

  if (!session) {
    transcript.textContent = 'Model not loaded yet. Please wait.';
    return;
  }

  updateStatus('${config.modelName} \\u00b7 Decoding audio...');

  const arrayBuffer = await file.arrayBuffer();
  const audioCtx = new OfflineAudioContext(1, 1, 16000);
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

  // Resample to 16kHz mono
  const offlineCtx = new OfflineAudioContext(1, Math.ceil(audioBuffer.duration * 16000), 16000);
  const source = offlineCtx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(offlineCtx.destination);
  source.start();
  const resampled = await offlineCtx.startRendering();
  const samples = resampled.getChannelData(0);

  updateStatus('${config.modelName} \\u00b7 Processing...');
  const start = performance.now();

  const mel = melSpectrogram(samples, 16000, 512, 160, 80);
  const output = await runInference(session, mel.data);
  const vocabSize = VOCAB.length + 1;
  const numTimesteps = Math.floor(output.length / vocabSize);
  const text = postprocessTranscript(output, numTimesteps, vocabSize, VOCAB);

  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus('${config.modelName} \\u00b7 ' + elapsed + 'ms \\u00b7 ' + getBackendLabel(session));

  transcript.textContent = text || '(no speech detected)';
});

init();`;
}

// ---- Realtime (mic) + Speech-to-Text script ----

function emitRealtimeSpeechToTextScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  return `${emitBlockCode(config, blocks)}

// --- Application ---
const MODEL_PATH = '${getModelPath(config, '.')}';
const VOCAB = [' ', ...'abcdefghijklmnopqrstuvwxyz'.split(''), "'"];
let session = null;
let capture = null;
let loop = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

async function init() {
  updateStatus('Loading model...');
  try {
    session = await createSession(MODEL_PATH);
    updateStatus('${config.modelName} \\u00b7 Ready');
  } catch (e) {
    updateStatus('Failed to load model');
    console.error('Model load error:', e);
  }
}

async function processAudio(samples) {
  const mel = melSpectrogram(samples, 16000, 512, 160, 80);
  const output = await runInference(session, mel.data);
  const vocabSize = VOCAB.length + 1;
  const numTimesteps = Math.floor(output.length / vocabSize);
  return postprocessTranscript(output, numTimesteps, vocabSize, VOCAB);
}

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const transcript = document.getElementById('transcript');

startBtn.addEventListener('click', async () => {
  if (!session) {
    updateStatus('Model not loaded yet.');
    return;
  }

  try {
    capture = await startAudioCapture(16000);
    startBtn.disabled = true;
    stopBtn.disabled = false;
    updateStatus('${config.modelName} \\u00b7 Listening...');

    loop = createAudioInferenceLoop({
      getSamples: capture.getSamples,
      onResult(text) {
        transcript.textContent = text || '(listening...)';
      },
      intervalMs: 2000,
    });
    loop.start();
  } catch (e) {
    updateStatus('Microphone access denied');
    console.error('Mic error:', e);
  }
});

stopBtn.addEventListener('click', () => {
  if (loop) { loop.stop(); loop = null; }
  if (capture) {
    stopStream(capture.stream);
    capture.audioContext.close();
    capture = null;
  }
  startBtn.disabled = false;
  stopBtn.disabled = true;
  updateStatus('${config.modelName} \\u00b7 Stopped');
});

init();`;
}

// ---- Realtime (mic) + Audio Classification script ----

function emitRealtimeAudioClassificationScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  return `${emitBlockCode(config, blocks)}

// --- Application ---
const MODEL_PATH = '${getModelPath(config, '.')}';
let session = null;
let capture = null;
let loop = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

async function init() {
  updateStatus('Loading model...');
  try {
    session = await createSession(MODEL_PATH);
    updateStatus('${config.modelName} \\u00b7 Ready');
  } catch (e) {
    updateStatus('Failed to load model');
    console.error('Model load error:', e);
  }
}

async function processAudio(samples) {
  const mel = melSpectrogram(samples, 16000, 512, 160, 40);
  const features = mfcc(mel.data, mel.numFrames, mel.numMelBands, 13);
  const output = await runInference(session, features);
  return postprocessResults(output);
}

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const resultsDiv = document.getElementById('results');

startBtn.addEventListener('click', async () => {
  if (!session) {
    updateStatus('Model not loaded yet.');
    return;
  }

  try {
    capture = await startAudioCapture(16000);
    startBtn.disabled = true;
    stopBtn.disabled = false;
    updateStatus('${config.modelName} \\u00b7 Listening...');

    loop = createAudioInferenceLoop({
      getSamples: capture.getSamples,
      onResult(results) {
        renderResults(results);
      },
      intervalMs: 2000,
    });
    loop.start();
  } catch (e) {
    updateStatus('Microphone access denied');
    console.error('Mic error:', e);
  }
});

stopBtn.addEventListener('click', () => {
  if (loop) { loop.stop(); loop = null; }
  if (capture) {
    stopStream(capture.stream);
    capture.audioContext.close();
    capture = null;
  }
  startBtn.disabled = false;
  stopBtn.disabled = true;
  updateStatus('${config.modelName} \\u00b7 Stopped');
});

function renderResults(results) {
  resultsDiv.innerHTML = '';
  const maxValue = results.values[0] || 1;

  for (let i = 0; i < results.indices.length; i++) {
    const pct = (results.values[i] * 100).toFixed(1);
    if (results.values[i] < 0.01) continue;

    const row = document.createElement('div');
    row.className = 'result-row' + (i === 0 ? ' top-result' : '');
    row.setAttribute('tabindex', '0');
    row.setAttribute('aria-label', 'Class ' + results.indices[i] + ': ' + pct + ' percent');

    row.innerHTML =
      '<span class="result-label">Class ' + results.indices[i] + '</span>' +
      '<div class="result-bar-container"><div class="result-bar" style="width:' +
      ((results.values[i] / maxValue) * 100) + '%"></div></div>' +
      '<span class="result-pct">' + pct + '%</span>';

    resultsDiv.appendChild(row);
  }
}

init();`;
}

// ---- Text-to-Speech script ----

function emitTextToSpeechScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  return `${emitBlockCode(config, blocks)}

// --- Application ---
const MODEL_PATH = '${getModelPath(config, '.')}';
let session = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

async function init() {
  updateStatus('Loading model...');
  try {
    session = await createSession(MODEL_PATH);
    updateStatus('${config.modelName} \\u00b7 Ready');
  } catch (e) {
    updateStatus('Failed to load model');
    console.error('Model load error:', e);
  }
}

const textInput = document.getElementById('textInput');
const synthesizeBtn = document.getElementById('synthesizeBtn');

synthesizeBtn.addEventListener('click', async () => {
  const text = textInput.value.trim();
  if (!text) return;

  if (!session) {
    updateStatus('Model not loaded yet. Please wait.');
    return;
  }

  updateStatus('${config.modelName} \\u00b7 Synthesizing...');
  const start = performance.now();

  // Simple char-to-charCode tokenization
  const tokens = new Float32Array(text.length);
  for (let i = 0; i < text.length; i++) {
    tokens[i] = text.charCodeAt(i);
  }

  const output = await runInference(session, tokens);
  const samples = postprocessAudio(output);
  await playAudio(samples);

  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus('${config.modelName} \\u00b7 ' + elapsed + 'ms \\u00b7 ' + getBackendLabel(session));
});

init();`;
}

// ---- Text Classification script ----

function emitTextClassificationScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  return `${emitBlockCode(config, blocks)}

// --- Application ---
const MODEL_PATH = '${getModelPath(config, '.')}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';
let session = null;
let tokenizer = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

async function init() {
  updateStatus('Loading model and tokenizer...');
  try {
    [session, tokenizer] = await Promise.all([
      createSession(MODEL_PATH),
      loadTokenizer(TOKENIZER_PATH),
    ]);
    updateStatus('${config.modelName} \\u00b7 Ready');
  } catch (e) {
    updateStatus('Failed to load model');
    console.error('Model load error:', e);
  }
}

const textInput = document.getElementById('textInput');
const classifyBtn = document.getElementById('classifyBtn');
const resultsDiv = document.getElementById('results');

classifyBtn.addEventListener('click', async () => {
  const text = textInput.value.trim();
  if (!text) return;

  if (!session || !tokenizer) {
    resultsDiv.textContent = 'Model not loaded yet. Please wait.';
    return;
  }

  updateStatus('${config.modelName} \\u00b7 Processing...');
  const start = performance.now();

  const { inputIds, attentionMask } = tokenizeText(tokenizer, text);
  const output = await runInference(session, inputIds);
  const results = postprocessResults(output);

  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus('${config.modelName} \\u00b7 ' + elapsed + 'ms \\u00b7 ' + getBackendLabel(session));

  renderResults(results);
});

function renderResults(results) {
  resultsDiv.innerHTML = '';
  const maxValue = results.values[0] || 1;

  for (let i = 0; i < results.indices.length; i++) {
    const pct = (results.values[i] * 100).toFixed(1);
    if (results.values[i] < 0.01) continue;

    const row = document.createElement('div');
    row.className = 'result-row' + (i === 0 ? ' top-result' : '');
    row.setAttribute('tabindex', '0');
    row.setAttribute('aria-label', 'Class ' + results.indices[i] + ': ' + pct + ' percent');

    row.innerHTML =
      '<span class="result-label">Class ' + results.indices[i] + '</span>' +
      '<div class="result-bar-container"><div class="result-bar" style="width:' +
      ((results.values[i] / maxValue) * 100) + '%"></div></div>' +
      '<span class="result-pct">' + pct + '%</span>';

    resultsDiv.appendChild(row);
  }
}

init();`;
}

// ---- Zero-Shot Classification script ----

function emitZeroShotScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  return `${emitBlockCode(config, blocks)}

// --- Application ---
const MODEL_PATH = '${getModelPath(config, '.')}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';
let session = null;
let tokenizer = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

async function init() {
  updateStatus('Loading model and tokenizer...');
  try {
    [session, tokenizer] = await Promise.all([
      createSession(MODEL_PATH),
      loadTokenizer(TOKENIZER_PATH),
    ]);
    updateStatus('${config.modelName} \\u00b7 Ready');
  } catch (e) {
    updateStatus('Failed to load model');
    console.error('Model load error:', e);
  }
}

const textInput = document.getElementById('textInput');
const labelsInput = document.getElementById('labelsInput');
const classifyBtn = document.getElementById('classifyBtn');
const resultsDiv = document.getElementById('results');

classifyBtn.addEventListener('click', async () => {
  const text = textInput.value.trim();
  const labelsRaw = labelsInput.value.trim();
  if (!text || !labelsRaw) return;

  if (!session || !tokenizer) {
    resultsDiv.textContent = 'Model not loaded yet. Please wait.';
    return;
  }

  const labels = labelsRaw.split(',').map((l) => l.trim()).filter(Boolean);
  if (labels.length === 0) return;

  updateStatus('${config.modelName} \\u00b7 Processing...');
  const start = performance.now();

  const scores = [];
  for (const label of labels) {
    const hypothesis = text + ' </s></s> ' + 'This is about ' + label + '.';
    const { inputIds } = tokenizeText(tokenizer, hypothesis);
    const output = await runInference(session, inputIds);
    const score = output[output.length > 2 ? 2 : output.length - 1];
    scores.push(score);
  }

  const results = postprocessZeroShot(scores, labels);

  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus('${config.modelName} \\u00b7 ' + elapsed + 'ms \\u00b7 ' + getBackendLabel(session));

  renderResults(results);
});

function renderResults(results) {
  resultsDiv.innerHTML = '';
  const maxValue = results[0]?.score || 1;

  for (let i = 0; i < results.length; i++) {
    const pct = (results[i].score * 100).toFixed(1);

    const row = document.createElement('div');
    row.className = 'result-row' + (i === 0 ? ' top-result' : '');
    row.setAttribute('tabindex', '0');
    row.setAttribute('aria-label', results[i].label + ': ' + pct + ' percent');

    row.innerHTML =
      '<span class="result-label">' + results[i].label + '</span>' +
      '<div class="result-bar-container"><div class="result-bar" style="width:' +
      ((results[i].score / maxValue) * 100) + '%"></div></div>' +
      '<span class="result-pct">' + pct + '%</span>';

    resultsDiv.appendChild(row);
  }
}

init();`;
}

// ---- Text Generation script ----

function emitTextGenerationScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  return `${emitBlockCode(config, blocks)}

// --- Application ---
const MODEL_PATH = '${getModelPath(config, '.')}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';
const MAX_NEW_TOKENS = 50;
const EOS_TOKEN_ID = 2;
let session = null;
let tokenizer = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

async function init() {
  updateStatus('Loading model and tokenizer...');
  try {
    [session, tokenizer] = await Promise.all([
      createSession(MODEL_PATH),
      loadTokenizer(TOKENIZER_PATH),
    ]);
    updateStatus('${config.modelName} \\u00b7 Ready');
  } catch (e) {
    updateStatus('Failed to load model');
    console.error('Model load error:', e);
  }
}

const textInput = document.getElementById('textInput');
const generateBtn = document.getElementById('generateBtn');
const outputDiv = document.getElementById('output');

generateBtn.addEventListener('click', async () => {
  const prompt = textInput.value.trim();
  if (!prompt) return;

  if (!session || !tokenizer) {
    outputDiv.textContent = 'Model not loaded yet. Please wait.';
    return;
  }

  generateBtn.disabled = true;
  updateStatus('${config.modelName} \\u00b7 Generating...');
  const start = performance.now();

  const encoded = tokenizer.encode(prompt);
  let inputIds = encoded.inputIds;
  outputDiv.textContent = prompt;

  for (let i = 0; i < MAX_NEW_TOKENS; i++) {
    const inputBigInt = new BigInt64Array(inputIds.map((id) => BigInt(id)));
    const output = await runInference(session, inputBigInt);

    const vocabSize = tokenizer.getVocabSize();
    const seqLen = inputIds.length;
    const logits = postprocessGeneration(output, seqLen, vocabSize);
    const nextToken = sampleNextToken(logits);

    if (nextToken === EOS_TOKEN_ID) break;

    inputIds = [...inputIds, nextToken];
    const decoded = tokenizer.decode(inputIds);
    outputDiv.textContent = decoded;
  }

  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus('${config.modelName} \\u00b7 ' + elapsed + 'ms \\u00b7 ' + getBackendLabel(session));
  generateBtn.disabled = false;
});

init();`;
}

// ---- Script dispatcher ----

function emitAppScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  // Audio tasks
  if (config.task === 'audio-classification') {
    if (config.input === 'mic') {
      return emitRealtimeAudioClassificationScript(config, blocks);
    }
    return emitFileAudioClassificationScript(config, blocks);
  }
  if (config.task === 'speech-to-text') {
    if (config.input === 'mic') {
      return emitRealtimeSpeechToTextScript(config, blocks);
    }
    return emitFileSpeechToTextScript(config, blocks);
  }
  if (config.task === 'text-to-speech') {
    return emitTextToSpeechScript(config, blocks);
  }

  // Text tasks
  if (config.task === 'text-classification') return emitTextClassificationScript(config, blocks);
  if (config.task === 'zero-shot-classification') return emitZeroShotScript(config, blocks);
  if (config.task === 'text-generation') return emitTextGenerationScript(config, blocks);

  // Realtime input modes (visual)
  if (config.input === 'camera' || config.input === 'screen') {
    return emitRealtimeScript(config, blocks);
  }

  // File input: dispatch by task
  if (config.input === 'file') {
    if (isClassificationTask(config.task)) {
      return emitFileClassificationScript(config, blocks);
    }
    switch (config.task) {
      case 'object-detection':
        return emitFileDetectionScript(config, blocks);
      case 'image-segmentation':
        return emitFileSegmentationScript(config, blocks);
      case 'feature-extraction':
        return emitFileFeatureExtractionScript(config, blocks);
      default:
        return emitFileClassificationScript(config, blocks);
    }
  }

  // Fallback: file classification for any unhandled combo
  return emitFileClassificationScript(config, blocks);
}

// ---- HTML body content ----

/** File input body (classification tasks: unchanged for snapshot compat) */
function emitFileClassificationBody(config: ResolvedConfig): string {
  const taskLabel = getTaskLabel(config.task);
  return `    <div class="container">
      <div>
        <div class="drop-zone" id="dropZone" role="button" tabindex="0"
             aria-label="Drop an image here or click to browse for ${taskLabel.toLowerCase()}">
          <p>Drop an image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input type="file" id="fileInput" accept="image/*" hidden>
        </div>

        <div id="preview" class="preview" hidden>
          <img id="previewImage" alt="Selected image for classification">
          <button id="changeBtn" class="change-btn">Choose another image</button>
        </div>
      </div>

      <div id="results" class="results" role="status" aria-live="polite" aria-atomic="true">
      </div>
    </div>`;
}

/** File input body with canvas overlay (detection/segmentation) */
function emitFileOverlayBody(config: ResolvedConfig): string {
  const taskLabel = getTaskLabel(config.task);
  return `    <div class="container">
      <div>
        <div class="drop-zone" id="dropZone" role="button" tabindex="0"
             aria-label="Drop an image here or click to browse for ${taskLabel.toLowerCase()}">
          <p>Drop an image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input type="file" id="fileInput" accept="image/*" hidden>
        </div>

        <div id="preview" class="preview" hidden>
          <div class="preview-wrapper">
            <img id="previewImage" alt="Selected image for ${taskLabel.toLowerCase()}">
            <canvas id="overlay"></canvas>
          </div>
          <button id="changeBtn" class="change-btn">Choose another image</button>
        </div>
      </div>

      <div id="results" class="results" role="status" aria-live="polite" aria-atomic="true">
      </div>
    </div>`;
}

/** Camera / screen capture body */
function emitRealtimeBody(config: ResolvedConfig): string {
  const isScreen = config.input === 'screen';
  const actionLabel = isScreen ? 'capture your screen' : 'use your camera';
  const btnLabel = isScreen ? 'Start Screen Capture' : 'Enable Camera';
  const taskLabel = getTaskLabel(config.task);
  return `    <div id="permissionPrompt" class="permission-prompt">
      <p>This app ${actionLabel} to run ${taskLabel.toLowerCase()} in real time.</p>
      <p class="hint">No video is recorded or sent anywhere.</p>
      <button id="startBtn" class="primary-btn">${btnLabel}</button>
    </div>

    <div id="videoContainer" hidden>
      <div class="video-wrapper">
        <video id="video" autoplay playsinline muted></video>
        <canvas id="overlay"></canvas>
      </div>
      <div class="controls">
        <button id="pauseBtn" class="control-btn">\u23f8 Pause</button>
      </div>
    </div>`;
}

/** Audio file input body (classification or STT) */
function emitAudioFileBody(config: ResolvedConfig): string {
  const taskLabel = getTaskLabel(config.task);
  const isSTT = config.task === 'speech-to-text';
  const resultsContent = isSTT
    ? `      <pre id="transcript" class="transcript" role="status" aria-live="polite" aria-atomic="true"></pre>`
    : `      <div id="results" class="results" role="status" aria-live="polite" aria-atomic="true">\n      </div>`;

  return `    <h2>${taskLabel}</h2>

    <a href="#${isSTT ? 'transcript' : 'results'}" class="skip-link">Skip to results</a>

    <div>
      <label for="fileInput">Choose an audio file &#9835;</label>
      <input type="file" id="fileInput" accept="audio/*" aria-label="Select audio file for ${taskLabel.toLowerCase()}">
    </div>

${resultsContent}`;
}

/** Mic-based audio body (classification or STT) */
function emitAudioMicBody(config: ResolvedConfig): string {
  const taskLabel = getTaskLabel(config.task);
  const isSTT = config.task === 'speech-to-text';
  const resultsContent = isSTT
    ? `    <pre id="transcript" class="transcript" role="status" aria-live="polite" aria-atomic="true">(listening...)</pre>`
    : `    <div id="results" class="results" role="status" aria-live="polite" aria-atomic="true">\n    </div>`;

  return `    <h2>${taskLabel}</h2>

    <div class="controls" role="group" aria-label="Recording controls">
      <button id="startBtn" class="controls-btn" aria-label="Start recording">Start Recording</button>
      <button id="stopBtn" class="controls-btn" disabled aria-label="Stop recording">Stop Recording</button>
    </div>

${resultsContent}`;
}

/** Text-to-Speech body */
function emitTtsBody(): string {
  return `    <h2>Text to Speech</h2>

    <div class="tts-input">
      <label for="textInput">Enter text to synthesize</label>
      <textarea id="textInput" rows="4" aria-label="Text to synthesize">Hello, this is a test of text to speech.</textarea>
      <button id="synthesizeBtn" class="primary-btn" aria-label="Synthesize speech">Synthesize</button>
    </div>`;
}

/** Text Classification body */
function emitTextClassificationBody(config: ResolvedConfig): string {
  const taskLabel = getTaskLabel(config.task);
  return `    <h2>${taskLabel}</h2>

    <div class="text-input">
      <label for="textInput">Enter text to classify</label>
      <textarea id="textInput" rows="4" aria-label="Text to classify">This movie was absolutely wonderful.</textarea>
      <button id="classifyBtn" class="run-btn" aria-label="Classify text">Classify</button>
    </div>

    <div id="results" class="results" role="status" aria-live="polite" aria-atomic="true">
    </div>`;
}

/** Zero-Shot Classification body */
function emitZeroShotBody(config: ResolvedConfig): string {
  const taskLabel = getTaskLabel(config.task);
  return `    <h2>${taskLabel}</h2>

    <div class="text-input">
      <label for="textInput">Enter text to classify</label>
      <textarea id="textInput" rows="4" aria-label="Text to classify">The stock market surged today after the Fed announcement.</textarea>
      <label for="labelsInput">Candidate labels (comma-separated)</label>
      <input type="text" id="labelsInput" class="labels-input" value="politics, finance, sports, technology" aria-label="Comma-separated candidate labels">
      <button id="classifyBtn" class="run-btn" aria-label="Classify text">Classify</button>
    </div>

    <div id="results" class="results" role="status" aria-live="polite" aria-atomic="true">
    </div>`;
}

/** Text Generation body */
function emitTextGenerationBody(config: ResolvedConfig): string {
  const taskLabel = getTaskLabel(config.task);
  return `    <h2>${taskLabel}</h2>

    <div class="text-input">
      <label for="textInput">Enter a prompt</label>
      <textarea id="textInput" rows="4" aria-label="Prompt for text generation">Once upon a time</textarea>
      <button id="generateBtn" class="run-btn" aria-label="Generate text">Generate</button>
    </div>

    <div id="output" class="generation-output" role="status" aria-live="polite" aria-atomic="true">
    </div>`;
}

function emitBodyContent(config: ResolvedConfig): string {
  // Audio tasks
  if (config.task === 'text-to-speech') {
    return emitTtsBody();
  }
  if (config.task === 'speech-to-text' || config.task === 'audio-classification') {
    if (config.input === 'mic') {
      return emitAudioMicBody(config);
    }
    return emitAudioFileBody(config);
  }

  // Text tasks
  if (config.task === 'text-classification') return emitTextClassificationBody(config);
  if (config.task === 'zero-shot-classification') return emitZeroShotBody(config);
  if (config.task === 'text-generation') return emitTextGenerationBody(config);

  if (config.input === 'camera' || config.input === 'screen') {
    return emitRealtimeBody(config);
  }
  if (config.input === 'file') {
    if (isClassificationTask(config.task) || config.task === 'feature-extraction') {
      return emitFileClassificationBody(config);
    }
    return emitFileOverlayBody(config);
  }
  return emitFileClassificationBody(config);
}

/** Extra CSS for overlay/camera modes */
function emitExtendedCSS(): string {
  return `
/* Canvas overlay */
.preview-wrapper {
  position: relative;
  display: inline-block;
}

.preview-wrapper #overlay,
.video-wrapper #overlay {
  position: absolute;
  top: 0;
  left: 0;
  pointer-events: none;
}

/* Video / Camera */
.video-wrapper {
  position: relative;
}

.video-wrapper video {
  display: block;
  max-width: 100%;
  border-radius: var(--webai-radius);
}

/* Permission prompt */
.permission-prompt {
  text-align: center;
  padding: var(--webai-space-8);
}

.primary-btn {
  background: var(--webai-accent);
  color: white;
  border: none;
  padding: var(--webai-space-3) var(--webai-space-6);
  border-radius: var(--webai-radius);
  cursor: pointer;
  font-size: var(--webai-font-size-lg);
  margin-top: var(--webai-space-4);
}

.primary-btn:hover {
  opacity: 0.9;
}

.controls {
  display: flex;
  gap: var(--webai-space-2);
  margin-top: var(--webai-space-2);
}

.control-btn {
  background: var(--webai-surface);
  border: 1px solid var(--webai-border);
  color: var(--webai-text);
  padding: var(--webai-space-1) var(--webai-space-3);
  border-radius: var(--webai-radius);
  cursor: pointer;
  font-size: var(--webai-font-size-sm);
}

/* Segmentation legend */
.color-swatch {
  display: inline-block;
  width: 14px;
  height: 14px;
  border-radius: 2px;
  vertical-align: middle;
  margin-right: var(--webai-space-1);
}

/* Embedding info */
.embedding-info p {
  margin-bottom: var(--webai-space-2);
  font-family: var(--webai-font-mono);
  font-size: var(--webai-font-size-sm);
}`;
}

/** Audio-specific CSS */
function emitAudioCSS(): string {
  return `
/* Transcript display */
.transcript {
  background: var(--webai-surface);
  border: 1px solid var(--webai-border);
  border-radius: 8px;
  padding: 1rem;
  min-height: 100px;
  white-space: pre-wrap;
  font-family: inherit;
}

/* Audio controls */
.controls {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1rem;
}

.controls button {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 1rem;
  background: var(--webai-accent);
  color: white;
}

.controls button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* TTS input */
.tts-input {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.tts-input textarea {
  background: var(--webai-surface);
  border: 1px solid var(--webai-border);
  border-radius: 6px;
  color: var(--webai-text);
  padding: 0.75rem;
  font-size: 1rem;
  resize: vertical;
}`;
}

function needsExtendedCSS(config: ResolvedConfig): boolean {
  return config.input === 'camera' || config.input === 'screen' ||
    config.task === 'object-detection' || config.task === 'image-segmentation' ||
    config.task === 'feature-extraction';
}

function needsAudioCSS(config: ResolvedConfig): boolean {
  return isAudioTask(config.task);
}

/**
 * Emit HTML framework files.
 */
export function emitHtml(config: ResolvedConfig, blocks: CodeBlock[]): GeneratedFile[] {
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const theme = config.theme;

  const designCSS = emitDesignSystemCSS(config);
  const appCSS = emitAppCSS();
  const extraCSS = needsExtendedCSS(config) ? emitExtendedCSS() : '';
  const audioCSS = needsAudioCSS(config) ? emitAudioCSS() : '';
  const appScript = emitAppScript(config, blocks);
  const bodyContent = emitBodyContent(config);

  const html = `<!DOCTYPE html>
<html lang="en" data-theme="${theme}">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${config.modelName} — ${taskLabel}</title>
  <style>
${designCSS}
${appCSS}${extraCSS}${audioCSS}
  </style>
</head>
<body>
  <a href="#results" class="skip-link">Skip to results</a>

  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>

${bodyContent}
  </main>

  <aside class="status-bar">
    <span id="status">${config.modelName} · Loading...</span>
  </aside>

  <div class="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>

  <script type="module">
${appScript}
  </script>
</body>
</html>`;

  const files: GeneratedFile[] = [
    { path: 'index.html', content: html },
    { path: 'README.md', content: emitReadme(config, ['index.html', 'README.md']) },
  ];

  // Include auxiliary files from Layer 1 blocks (e.g. AudioWorklet processor)
  for (const block of blocks) {
    if (block.auxiliaryFiles) {
      for (const aux of block.auxiliaryFiles) {
        files.push(aux);
      }
    }
  }

  return files;
}
