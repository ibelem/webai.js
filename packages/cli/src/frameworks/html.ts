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
  buildPageHeading,
} from './shared.js';
import { LITERT_CDN } from '../emitters/inference-litert.js';

const ORT_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.mjs';

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
  } else if (config.engine === 'litert') {
    sections.push(`import { loadLiteRt, loadAndCompile, Tensor } from '${LITERT_CDN.replace('/wasm/', '/dist/litertjs-core.min.mjs')}';`);
    sections.push(`const LITERT_WASM_PATH = '${LITERT_CDN}';`);
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
  return task === 'audio-classification' || task === 'speech-to-text' || task === 'text-to-speech' ||
    task === 'audio-to-audio' || task === 'speaker-diarization' || task === 'voice-activity-detection';
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

// ---- Source-aware helpers ----

/** True when the model will be fetched from a URL (HF or generic) */
function isRemoteModel(config: ResolvedConfig): boolean {
  return config.modelSource !== 'local-path' && !!config.modelUrl;
}

/** Progress bar + filename + stats HTML */
function emitProgressBarHtml(): string {
  return `<div id="progressBar" class="progress-container" hidden>
        <div class="progress-label">
          <span id="progressFilename" class="progress-filename"></span>
          <span id="progressStats" class="progress-stats">0%</span>
        </div>
        <div class="progress-track">
          <div id="progressFill" class="progress-fill" style="width: 0%"></div>
        </div>
      </div>`;
}

/** Reusable JS: formatBytes + fetchWithProgress + showProgress/updateProgress/completeProgress */
function emitProgressHelpers(): string {
  return `
function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function showProgress(filename) {
  const el = document.getElementById('progressBar');
  el.hidden = false;
  document.getElementById('progressFilename').textContent = filename;
  document.getElementById('progressStats').textContent = '0%';
  const fill = document.getElementById('progressFill');
  fill.style.width = '0%';
  fill.className = 'progress-fill';
}

function updateProgress(loaded, total) {
  const fill = document.getElementById('progressFill');
  const stats = document.getElementById('progressStats');
  if (total && total > 0) {
    fill.classList.remove('indeterminate');
    const pct = Math.min(100, (loaded / total) * 100);
    fill.style.width = pct.toFixed(1) + '%';
    stats.textContent = formatBytes(loaded) + ' / ' + formatBytes(total) + '  \\u00b7  ' + pct.toFixed(0) + '%';
  } else {
    if (!fill.classList.contains('indeterminate')) fill.classList.add('indeterminate');
    stats.textContent = formatBytes(loaded);
  }
}

function completeProgress(finalSize) {
  const fill = document.getElementById('progressFill');
  fill.classList.remove('indeterminate');
  fill.classList.add('done');
  fill.style.width = '100%';
  document.getElementById('progressStats').textContent = formatBytes(finalSize);
}

async function fetchWithProgress(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error('HTTP ' + res.status + ': ' + res.statusText);
  const contentLength = res.headers.get('Content-Length');
  const total = contentLength ? parseInt(contentLength, 10) : null;
  if (!res.body) {
    const buf = await res.arrayBuffer();
    updateProgress(buf.byteLength, buf.byteLength);
    return buf;
  }
  const reader = res.body.getReader();
  if (total && total > 0) {
    const result = new Uint8Array(total);
    let offset = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      result.set(value, offset);
      offset += value.byteLength;
      updateProgress(offset, total);
    }
    return result.buffer;
  }
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    updateProgress(loaded, null);
  }
  const result = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) { result.set(c, off); off += c.byteLength; }
  return result.buffer;
}`;
}

/** Emit remote model init (fetchWithProgress → createSession → enable Run btn) */
function emitRemoteInit(modelName: string, opts?: { tokenizer?: boolean; readyMsg?: string; enableRunBtn?: boolean }): string {
  const { tokenizer = false, readyMsg = `${modelName} \\u00b7 Ready`, enableRunBtn = true } = opts ?? {};
  const tokenizerLoad = tokenizer ? `\n    tokenizer = await loadTokenizer(TOKENIZER_PATH);` : '';
  const enableBtn = enableRunBtn ? `\n    document.getElementById('runBtn').disabled = false;` : '';
  return `async function init() {
  const filename = MODEL_PATH.split('/').pop() || '${modelName}';
  showProgress(filename);
  updateStatus('Downloading model...');
  try {
    const buf = await fetchWithProgress(MODEL_PATH);
    completeProgress(buf.byteLength);
    updateStatus('Creating session...');
    session = await createSession(buf);${tokenizerLoad}
    updateStatus('${readyMsg}');${enableBtn}
  } catch (e) {
    updateStatus('Failed to load model');
    console.error('Model load error:', e);
  }
}`;
}

/** Emit local model init (createSession from path) */
function emitLocalInit(modelName: string, opts?: { tokenizer?: boolean; readyMsg?: string }): string {
  const { tokenizer = false, readyMsg = `${modelName} \\u00b7 Ready` } = opts ?? {};
  const loadMsg = tokenizer ? 'Loading model and tokenizer...' : 'Loading model...';
  const sessionLoad = tokenizer
    ? `[session, tokenizer] = await Promise.all([
      createSession(MODEL_PATH),
      loadTokenizer(TOKENIZER_PATH),
    ]);`
    : `session = await createSession(MODEL_PATH);`;
  return `async function init() {
  updateStatus('${loadMsg}');
  try {
    ${sessionLoad}
    updateStatus('${readyMsg}');
  } catch (e) {
    updateStatus('Failed to load model');
    console.error('Model load error:', e);
  }
}`;
}

/** Emit drop zone event listeners for local model */
function emitDropZoneListeners(extraReset = ''): string {
  return `const dropZone = document.getElementById('dropZone');
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
  fileInput.value = '';${extraReset}
});`;
}

/** Emit remote Run button + file input listeners */
function emitRunButtonListeners(): string {
  return `const runBtn = document.getElementById('runBtn');
runBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => { const file = fileInput.files[0]; if (file) handleFile(file); });`;
}

// ---- File + Classification script ----

function emitFileClassificationScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  const remote = isRemoteModel(config);
  const modelPath = getModelPath(config, '.');
  const modelName = config.modelName;

  const sharedInfer = `
async function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    resultsDiv.textContent = 'Unsupported file type. Try JPG, PNG, or WebP.';
    return;
  }
  const url = URL.createObjectURL(file);
  previewImage.src = url;
  preview.hidden = false;${!remote ? '\n  dropZone.hidden = true;' : ''}
  await new Promise((resolve) => { previewImage.onload = resolve; });
  const canvas = document.createElement('canvas');
  canvas.width = previewImage.naturalWidth;
  canvas.height = previewImage.naturalHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(previewImage, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  if (!session) { resultsDiv.textContent = 'Model not loaded yet.'; return; }
  updateStatus('${modelName} \\u00b7 Processing...');
  const start = performance.now();
  const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
  const output = await runInference(session, inputTensor);
  const results = postprocessResults(output);
  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus('${modelName} \\u00b7 ' + elapsed + 'ms \\u00b7 ' + getBackendLabel(session));
  renderResults(results);
  URL.revokeObjectURL(url);
}`;

  const renderFn = `
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
}`;

  return `${emitBlockCode(config, blocks)}
${remote ? emitProgressHelpers() : ''}

// --- Application ---
const MODEL_PATH = '${modelPath}';
let session = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const previewImage = document.getElementById('previewImage');
const resultsDiv = document.getElementById('results');

${remote ? emitRunButtonListeners() : emitDropZoneListeners()}

${sharedInfer}
${renderFn}

${remote ? emitRemoteInit(modelName) : emitLocalInit(modelName)}

init();`;
}

// ---- File + Detection script ----

function emitFileDetectionScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  const remote = isRemoteModel(config);
  const modelPath = getModelPath(config, '.');
  const modelName = config.modelName;
  const outputShape = config.modelMeta.outputs[0]?.shape ?? [1, 84, 8400];
  const numAttributes = outputShape[1] ?? 84;
  const numAnchors = outputShape[2] ?? 8400;
  const overlayReset = `\n  const ctx = overlay.getContext('2d');\n  ctx.clearRect(0, 0, overlay.width, overlay.height);`;

  const handleFile = `
async function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    resultsDiv.textContent = 'Unsupported file type. Try JPG, PNG, or WebP.';
    return;
  }
  const url = URL.createObjectURL(file);
  previewImage.src = url;
  preview.hidden = false;${!remote ? '\n  dropZone.hidden = true;' : ''}
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
  updateStatus('${modelName} \\u00b7 Processing...');
  const start = performance.now();
  const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
  const output = await runInference(session, inputTensor);
  const boxes = postprocessDetections(output, NUM_ANCHORS, NUM_ATTRIBUTES);
  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus('${modelName} \\u00b7 ' + elapsed + 'ms \\u00b7 ' + getBackendLabel(session));
  renderDetections(boxes, previewImage.naturalWidth, previewImage.naturalHeight);
  URL.revokeObjectURL(url);
}`;

  return `${emitBlockCode(config, blocks)}
${remote ? emitProgressHelpers() : ''}

// --- Application ---
${emitColorPalette()}

const MODEL_PATH = '${modelPath}';
const NUM_ATTRIBUTES = ${numAttributes};
const NUM_ANCHORS = ${numAnchors};
let session = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const previewImage = document.getElementById('previewImage');
const overlay = document.getElementById('overlay');
const resultsDiv = document.getElementById('results');

${remote ? emitRunButtonListeners() : emitDropZoneListeners(overlayReset)}

${handleFile}

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

${remote ? emitRemoteInit(modelName) : emitLocalInit(modelName)}

init();`;
}

// ---- File + Segmentation script ----

function emitFileSegmentationScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  const remote = isRemoteModel(config);
  const modelPath = getModelPath(config, '.');
  const modelName = config.modelName;
  const outputShape = config.modelMeta.outputs[0]?.shape ?? [1, 21, 512, 512];
  const numClasses = outputShape[1] ?? 21;
  const maskH = outputShape[2] ?? 512;
  const maskW = outputShape[3] ?? 512;
  const overlayReset = `\n  const ctx = overlay.getContext('2d');\n  ctx.clearRect(0, 0, overlay.width, overlay.height);`;

  const handleFile = `
async function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    resultsDiv.textContent = 'Unsupported file type. Try JPG, PNG, or WebP.';
    return;
  }
  const url = URL.createObjectURL(file);
  previewImage.src = url;
  preview.hidden = false;${!remote ? '\n  dropZone.hidden = true;' : ''}
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
  updateStatus('${modelName} \\u00b7 Processing...');
  const start = performance.now();
  const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
  const output = await runInference(session, inputTensor);
  const mask = postprocessSegmentation(output, NUM_CLASSES, MASK_H, MASK_W);
  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus('${modelName} \\u00b7 ' + elapsed + 'ms \\u00b7 ' + getBackendLabel(session));
  renderMask(mask, previewImage.naturalWidth, previewImage.naturalHeight);
  URL.revokeObjectURL(url);
}`;

  return `${emitBlockCode(config, blocks)}
${remote ? emitProgressHelpers() : ''}

// --- Application ---
${emitColorPalette()}

const MODEL_PATH = '${modelPath}';
const NUM_CLASSES = ${numClasses};
const MASK_H = ${maskH};
const MASK_W = ${maskW};
let session = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const previewImage = document.getElementById('previewImage');
const overlay = document.getElementById('overlay');
const resultsDiv = document.getElementById('results');

${remote ? emitRunButtonListeners() : emitDropZoneListeners(overlayReset)}

${handleFile}

function renderMask(mask, displayW, displayH) {
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
    maskImage.data[i * 4 + 3] = 128;
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

${remote ? emitRemoteInit(modelName) : emitLocalInit(modelName)}

init();`;
}

// ---- File + Feature Extraction script ----

function emitFileFeatureExtractionScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  const remote = isRemoteModel(config);
  const modelPath = getModelPath(config, '.');
  const modelName = config.modelName;

  const handleFile = `
async function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    resultsDiv.textContent = 'Unsupported file type. Try JPG, PNG, or WebP.';
    return;
  }
  const url = URL.createObjectURL(file);
  previewImage.src = url;
  preview.hidden = false;${!remote ? '\n  dropZone.hidden = true;' : ''}
  await new Promise((resolve) => { previewImage.onload = resolve; });
  const canvas = document.createElement('canvas');
  canvas.width = previewImage.naturalWidth;
  canvas.height = previewImage.naturalHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(previewImage, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  if (!session) { resultsDiv.textContent = 'Model not loaded yet.'; return; }
  updateStatus('${modelName} \\u00b7 Processing...');
  const start = performance.now();
  const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
  const output = await runInference(session, inputTensor);
  const embedding = postprocessEmbeddings(output);
  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus('${modelName} \\u00b7 ' + elapsed + 'ms \\u00b7 ' + getBackendLabel(session));
  renderEmbedding(embedding);
  URL.revokeObjectURL(url);
}`;

  return `${emitBlockCode(config, blocks)}
${remote ? emitProgressHelpers() : ''}

// --- Application ---
const MODEL_PATH = '${modelPath}';
let session = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const previewImage = document.getElementById('previewImage');
const resultsDiv = document.getElementById('results');

${remote ? emitRunButtonListeners() : emitDropZoneListeners()}

${handleFile}

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

${remote ? emitRemoteInit(modelName) : emitLocalInit(modelName)}

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

  const remote = isRemoteModel(config);
  const modelPath = getModelPath(config, '.');
  const modelName = config.modelName;

  return `${emitBlockCode(config, blocks)}
${remote ? emitProgressHelpers() : ''}

// --- Application ---
${extraCode}

const MODEL_PATH = '${modelPath}';
let session = null;
let currentStream = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

${remote ? emitRemoteInit(modelName, { readyMsg: `${modelName} \\u00b7 Ready \\u00b7 Tap Start`, enableRunBtn: false }) : emitLocalInit(modelName, { readyMsg: `${modelName} \\u00b7 Ready \\u00b7 Tap Start` })}

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
  const remote = isRemoteModel(config);
  const modelPath = getModelPath(config, '.');
  const modelName = config.modelName;

  return `${emitBlockCode(config, blocks)}
${remote ? emitProgressHelpers() : ''}

// --- Application ---
const MODEL_PATH = '${modelPath}';
let session = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

${remote ? emitRemoteInit(modelName, { enableRunBtn: false }) : emitLocalInit(modelName)}

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
  const remote = isRemoteModel(config);
  const modelPath = getModelPath(config, '.');
  const modelName = config.modelName;

  return `${emitBlockCode(config, blocks)}
${remote ? emitProgressHelpers() : ''}

// --- Application ---
const MODEL_PATH = '${modelPath}';
const VOCAB = [' ', ...'abcdefghijklmnopqrstuvwxyz'.split(''), "'"];
let session = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

${remote ? emitRemoteInit(modelName, { enableRunBtn: false }) : emitLocalInit(modelName)}

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
  const remote = isRemoteModel(config);
  const modelPath = getModelPath(config, '.');
  const modelName = config.modelName;

  return `${emitBlockCode(config, blocks)}
${remote ? emitProgressHelpers() : ''}

// --- Application ---
const MODEL_PATH = '${modelPath}';
const VOCAB = [' ', ...'abcdefghijklmnopqrstuvwxyz'.split(''), "'"];
let session = null;
let capture = null;
let loop = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

${remote ? emitRemoteInit(modelName, { enableRunBtn: false }) : emitLocalInit(modelName)}

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
  const remote = isRemoteModel(config);
  const modelPath = getModelPath(config, '.');
  const modelName = config.modelName;

  return `${emitBlockCode(config, blocks)}
${remote ? emitProgressHelpers() : ''}

// --- Application ---
const MODEL_PATH = '${modelPath}';
let session = null;
let capture = null;
let loop = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

${remote ? emitRemoteInit(modelName, { enableRunBtn: false }) : emitLocalInit(modelName)}

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
  const remote = isRemoteModel(config);
  const modelPath = getModelPath(config, '.');
  const modelName = config.modelName;

  return `${emitBlockCode(config, blocks)}
${remote ? emitProgressHelpers() : ''}

// --- Application ---
const MODEL_PATH = '${modelPath}';
let session = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

${remote ? emitRemoteInit(modelName, { enableRunBtn: false }) : emitLocalInit(modelName)}

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
  const remote = isRemoteModel(config);
  const modelPath = getModelPath(config, '.');
  const modelName = config.modelName;

  return `${emitBlockCode(config, blocks)}
${remote ? emitProgressHelpers() : ''}

// --- Application ---
const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';
let session = null;
let tokenizer = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

${remote ? emitRemoteInit(modelName, { tokenizer: true, enableRunBtn: false }) : emitLocalInit(modelName, { tokenizer: true })}

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
  const remote = isRemoteModel(config);
  const modelPath = getModelPath(config, '.');
  const modelName = config.modelName;

  return `${emitBlockCode(config, blocks)}
${remote ? emitProgressHelpers() : ''}

// --- Application ---
const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';
let session = null;
let tokenizer = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

${remote ? emitRemoteInit(modelName, { tokenizer: true, enableRunBtn: false }) : emitLocalInit(modelName, { tokenizer: true })}

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
  const remote = isRemoteModel(config);
  const modelPath = getModelPath(config, '.');
  const modelName = config.modelName;

  return `${emitBlockCode(config, blocks)}
${remote ? emitProgressHelpers() : ''}

// --- Application ---
const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';
const MAX_NEW_TOKENS = 50;
const EOS_TOKEN_ID = 2;
let session = null;
let tokenizer = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

${remote ? emitRemoteInit(modelName, { tokenizer: true, enableRunBtn: false }) : emitLocalInit(modelName, { tokenizer: true })}

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

// ---- Fill-Mask script ----

function emitFillMaskScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  const remote = isRemoteModel(config);
  const modelPath = getModelPath(config, '.');
  const modelName = config.modelName;

  return `${emitBlockCode(config, blocks)}
${remote ? emitProgressHelpers() : ''}

// --- Application ---
const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';
let session = null;
let tokenizer = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

${remote ? emitRemoteInit(modelName, { tokenizer: true, enableRunBtn: false }) : emitLocalInit(modelName, { tokenizer: true })}

const textInput = document.getElementById('textInput');
const predictBtn = document.getElementById('predictBtn');
const resultsDiv = document.getElementById('results');

predictBtn.addEventListener('click', async () => {
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
  const results = postprocessFillMask(output, inputIds, tokenizer);

  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus('${config.modelName} \\u00b7 ' + elapsed + 'ms \\u00b7 ' + getBackendLabel(session));

  resultsDiv.innerHTML = '';
  for (const pred of results) {
    const row = document.createElement('div');
    row.className = 'mask-prediction';
    row.innerHTML = '<span class="token">' + pred.token + '</span><span class="prob">' + (pred.score * 100).toFixed(1) + '%</span>';
    resultsDiv.appendChild(row);
  }
});

init();`;
}

// ---- Sentence Similarity script ----

function emitSentenceSimilarityScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  const remote = isRemoteModel(config);
  const modelPath = getModelPath(config, '.');
  const modelName = config.modelName;

  return `${emitBlockCode(config, blocks)}
${remote ? emitProgressHelpers() : ''}

// --- Application ---
const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';
let session = null;
let tokenizer = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

${remote ? emitRemoteInit(modelName, { tokenizer: true, enableRunBtn: false }) : emitLocalInit(modelName, { tokenizer: true })}

const sourceInput = document.getElementById('sourceInput');
const compareInput = document.getElementById('compareInput');
const compareBtn = document.getElementById('compareBtn');
const resultsDiv = document.getElementById('results');

compareBtn.addEventListener('click', async () => {
  const source = sourceInput.value.trim();
  const comparisons = compareInput.value.trim().split('\\n').filter(Boolean);
  if (!source || comparisons.length === 0) return;

  if (!session || !tokenizer) {
    resultsDiv.textContent = 'Model not loaded yet. Please wait.';
    return;
  }

  updateStatus('${config.modelName} \\u00b7 Computing embeddings...');
  const start = performance.now();

  const { inputIds: srcIds } = tokenizeText(tokenizer, source);
  const srcEmb = await runInference(session, srcIds);

  resultsDiv.innerHTML = '';
  for (const sentence of comparisons) {
    const { inputIds: cmpIds } = tokenizeText(tokenizer, sentence);
    const cmpEmb = await runInference(session, cmpIds);
    const score = cosineSimilarity(srcEmb, cmpEmb);

    const row = document.createElement('div');
    row.className = 'similarity-score';
    row.innerHTML = '<span>' + sentence + '</span><span class="value">' + score.toFixed(4) + '</span>';
    resultsDiv.appendChild(row);
  }

  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus('${config.modelName} \\u00b7 ' + elapsed + 'ms \\u00b7 ' + getBackendLabel(session));
});

init();`;
}

// ---- Depth Estimation script ----

function emitDepthEstimationScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  const remote = isRemoteModel(config);
  const modelPath = getModelPath(config, '.');
  const modelName = config.modelName;

  const handleFile = `
function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  const url = URL.createObjectURL(file);
  previewImage.src = url;
  previewImage.onload = () => processImage(previewImage);
  preview.hidden = false;${!remote ? '\n  dropZone.hidden = true;' : ''}
}`;

  return `${emitBlockCode(config, blocks)}
${remote ? emitProgressHelpers() : ''}

// --- Application ---
const MODEL_PATH = '${modelPath}';
let session = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const previewImage = document.getElementById('previewImage');
const depthCanvas = document.getElementById('depthCanvas');

${remote ? emitRunButtonListeners() : emitDropZoneListeners()}

${handleFile}

async function processImage(img) {
  if (!session) return;
  updateStatus('${modelName} \\u00b7 Processing...');
  const start = performance.now();

  const input = preprocessImage(img);
  const output = await runInference(session, input);
  const depthMap = postprocessDepth(output);

  depthCanvas.width = img.naturalWidth;
  depthCanvas.height = img.naturalHeight;
  const ctx = depthCanvas.getContext('2d');
  const imgData = ctx.createImageData(depthCanvas.width, depthCanvas.height);
  for (let i = 0; i < depthMap.length; i++) {
    const v = depthMap[i];
    imgData.data[i * 4] = v;
    imgData.data[i * 4 + 1] = v;
    imgData.data[i * 4 + 2] = v;
    imgData.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);

  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus('${modelName} \\u00b7 ' + elapsed + 'ms \\u00b7 ' + getBackendLabel(session));
}

${remote ? emitRemoteInit(modelName) : emitLocalInit(modelName)}

init();`;
}

// ---- Token Classification (NER) script ----

function emitTokenClassificationScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  const remote = isRemoteModel(config);
  const modelPath = getModelPath(config, '.');
  const modelName = config.modelName;

  return `${emitBlockCode(config, blocks)}
${remote ? emitProgressHelpers() : ''}

// --- Application ---
const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';
let session = null;
let tokenizer = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

${remote ? emitRemoteInit(modelName, { tokenizer: true, enableRunBtn: false }) : emitLocalInit(modelName, { tokenizer: true })}

const textInput = document.getElementById('textInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsDiv = document.getElementById('results');

analyzeBtn.addEventListener('click', async () => {
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
  const entities = postprocessTokenClassification(output, inputIds, tokenizer);

  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus('${config.modelName} \\u00b7 ' + elapsed + 'ms \\u00b7 ' + getBackendLabel(session));

  let html = text;
  for (let i = entities.length - 1; i >= 0; i--) {
    const e = entities[i];
    const before = html.slice(0, e.start);
    const word = html.slice(e.start, e.end);
    const after = html.slice(e.end);
    html = before + '<span class="ner-entity" data-type="' + e.type + '" title="' + e.type + ' (' + (e.score * 100).toFixed(1) + '%)">' + word + '</span>' + after;
  }
  resultsDiv.innerHTML = html;
});

init();`;
}

// ---- Question Answering script ----

function emitQuestionAnsweringScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  const remote = isRemoteModel(config);
  const modelPath = getModelPath(config, '.');
  const modelName = config.modelName;

  return `${emitBlockCode(config, blocks)}
${remote ? emitProgressHelpers() : ''}

// --- Application ---
const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';
let session = null;
let tokenizer = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

${remote ? emitRemoteInit(modelName, { tokenizer: true, enableRunBtn: false }) : emitLocalInit(modelName, { tokenizer: true })}

const contextInput = document.getElementById('contextInput');
const questionInput = document.getElementById('questionInput');
const answerBtn = document.getElementById('answerBtn');
const answerDiv = document.getElementById('answer');

answerBtn.addEventListener('click', async () => {
  const context = contextInput.value.trim();
  const question = questionInput.value.trim();
  if (!context || !question) return;

  if (!session || !tokenizer) {
    answerDiv.textContent = 'Model not loaded yet. Please wait.';
    return;
  }

  updateStatus('${config.modelName} \\u00b7 Processing...');
  const start = performance.now();

  const combined = question + ' [SEP] ' + context;
  const { inputIds, attentionMask } = tokenizeText(tokenizer, combined);
  const output = await runInference(session, inputIds);
  const result = postprocessQA(output, inputIds, tokenizer);

  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus('${config.modelName} \\u00b7 ' + elapsed + 'ms \\u00b7 ' + getBackendLabel(session));

  answerDiv.innerHTML = '<div>' + result.answer + '</div><div class="score">Confidence: ' + (result.score * 100).toFixed(1) + '%</div>';
});

init();`;
}

// ---- Summarization script ----

function emitSummarizationScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  const remote = isRemoteModel(config);
  const modelPath = getModelPath(config, '.');
  const modelName = config.modelName;

  return `${emitBlockCode(config, blocks)}
${remote ? emitProgressHelpers() : ''}

// --- Application ---
const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';
const MAX_NEW_TOKENS = 128;
const EOS_TOKEN_ID = 1;
let session = null;
let tokenizer = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

${remote ? emitRemoteInit(modelName, { tokenizer: true, enableRunBtn: false }) : emitLocalInit(modelName, { tokenizer: true })}

const textInput = document.getElementById('textInput');
const summarizeBtn = document.getElementById('summarizeBtn');
const outputDiv = document.getElementById('output');

summarizeBtn.addEventListener('click', async () => {
  const text = textInput.value.trim();
  if (!text) return;

  if (!session || !tokenizer) {
    outputDiv.textContent = 'Model not loaded yet. Please wait.';
    return;
  }

  summarizeBtn.disabled = true;
  updateStatus('${config.modelName} \\u00b7 Summarizing...');
  const start = performance.now();

  const { inputIds } = tokenizeText(tokenizer, text);
  const output = await runInference(session, inputIds);
  const summary = postprocessSummarization(output, tokenizer, MAX_NEW_TOKENS, EOS_TOKEN_ID);

  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus('${config.modelName} \\u00b7 ' + elapsed + 'ms \\u00b7 ' + getBackendLabel(session));

  outputDiv.textContent = summary;
  summarizeBtn.disabled = false;
});

init();`;
}

// ---- Translation script ----

function emitTranslationScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  const remote = isRemoteModel(config);
  const modelPath = getModelPath(config, '.');
  const modelName = config.modelName;

  return `${emitBlockCode(config, blocks)}
${remote ? emitProgressHelpers() : ''}

// --- Application ---
const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';
const MAX_NEW_TOKENS = 128;
const EOS_TOKEN_ID = 1;
let session = null;
let tokenizer = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

${remote ? emitRemoteInit(modelName, { tokenizer: true, enableRunBtn: false }) : emitLocalInit(modelName, { tokenizer: true })}

const textInput = document.getElementById('textInput');
const translateBtn = document.getElementById('translateBtn');
const outputDiv = document.getElementById('output');

translateBtn.addEventListener('click', async () => {
  const text = textInput.value.trim();
  if (!text) return;

  if (!session || !tokenizer) {
    outputDiv.textContent = 'Model not loaded yet. Please wait.';
    return;
  }

  translateBtn.disabled = true;
  updateStatus('${config.modelName} \\u00b7 Translating...');
  const start = performance.now();

  const { inputIds } = tokenizeText(tokenizer, text);
  const output = await runInference(session, inputIds);
  const translation = postprocessTranslation(output, tokenizer, MAX_NEW_TOKENS, EOS_TOKEN_ID);

  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus('${config.modelName} \\u00b7 ' + elapsed + 'ms \\u00b7 ' + getBackendLabel(session));

  outputDiv.textContent = translation;
  translateBtn.disabled = false;
});

init();`;
}

// ---- Text2Text Generation script ----

function emitText2TextScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  const remote = isRemoteModel(config);
  const modelPath = getModelPath(config, '.');
  const modelName = config.modelName;

  return `${emitBlockCode(config, blocks)}
${remote ? emitProgressHelpers() : ''}

// --- Application ---
const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';
const MAX_NEW_TOKENS = 128;
const EOS_TOKEN_ID = 1;
let session = null;
let tokenizer = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

${remote ? emitRemoteInit(modelName, { tokenizer: true, enableRunBtn: false }) : emitLocalInit(modelName, { tokenizer: true })}

const textInput = document.getElementById('textInput');
const runBtn = document.getElementById('runBtn');
const outputDiv = document.getElementById('output');

runBtn.addEventListener('click', async () => {
  const text = textInput.value.trim();
  if (!text) return;

  if (!session || !tokenizer) {
    outputDiv.textContent = 'Model not loaded yet. Please wait.';
    return;
  }

  runBtn.disabled = true;
  updateStatus('${config.modelName} \\u00b7 Processing...');
  const start = performance.now();

  const { inputIds } = tokenizeText(tokenizer, text);
  const output = await runInference(session, inputIds);
  const result = postprocessText2Text(output, tokenizer, MAX_NEW_TOKENS, EOS_TOKEN_ID);

  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus('${config.modelName} \\u00b7 ' + elapsed + 'ms \\u00b7 ' + getBackendLabel(session));

  outputDiv.textContent = result;
  runBtn.disabled = false;
});

init();`;
}

// ---- Conversational script ----

function emitConversationalScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  const remote = isRemoteModel(config);
  const modelPath = getModelPath(config, '.');
  const modelName = config.modelName;

  return `${emitBlockCode(config, blocks)}
${remote ? emitProgressHelpers() : ''}

// --- Application ---
const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';
const MAX_NEW_TOKENS = 50;
const EOS_TOKEN_ID = 2;
let session = null;
let tokenizer = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

${remote ? emitRemoteInit(modelName, { tokenizer: true, enableRunBtn: false }) : emitLocalInit(modelName, { tokenizer: true })}

const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const history = [];

function addMessage(role, text) {
  const div = document.createElement('div');
  div.className = 'chat-msg ' + role;
  div.textContent = text;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

sendBtn.addEventListener('click', async () => {
  const text = chatInput.value.trim();
  if (!text) return;

  if (!session || !tokenizer) return;

  addMessage('user', text);
  chatInput.value = '';
  history.push(text);

  sendBtn.disabled = true;
  updateStatus('${config.modelName} \\u00b7 Generating...');
  const start = performance.now();

  const prompt = history.join(' ');
  const encoded = tokenizer.encode(prompt);
  let inputIds = encoded.inputIds;

  for (let i = 0; i < MAX_NEW_TOKENS; i++) {
    const inputBigInt = new BigInt64Array(inputIds.map((id) => BigInt(id)));
    const output = await runInference(session, inputBigInt);

    const vocabSize = tokenizer.getVocabSize();
    const seqLen = inputIds.length;
    const logits = postprocessConversational(output, seqLen, vocabSize);
    const nextToken = sampleNextToken(logits);

    if (nextToken === EOS_TOKEN_ID) break;
    inputIds = [...inputIds, nextToken];
  }

  const decoded = tokenizer.decode(inputIds);
  const reply = decoded.slice(prompt.length).trim();
  addMessage('bot', reply || '(no response)');
  history.push(reply);

  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus('${config.modelName} \\u00b7 ' + elapsed + 'ms \\u00b7 ' + getBackendLabel(session));
  sendBtn.disabled = false;
});

chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendBtn.click(); }
});

init();`;
}

// ---- Table Question Answering script ----

function emitTableQAScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  const remote = isRemoteModel(config);
  const modelPath = getModelPath(config, '.');
  const modelName = config.modelName;

  return `${emitBlockCode(config, blocks)}
${remote ? emitProgressHelpers() : ''}

// --- Application ---
const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';
let session = null;
let tokenizer = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

${remote ? emitRemoteInit(modelName, { tokenizer: true, enableRunBtn: false }) : emitLocalInit(modelName, { tokenizer: true })}

const tableInput = document.getElementById('tableInput');
const questionInput = document.getElementById('questionInput');
const answerBtn = document.getElementById('answerBtn');
const answerDiv = document.getElementById('answer');

answerBtn.addEventListener('click', async () => {
  const table = tableInput.value.trim();
  const question = questionInput.value.trim();
  if (!table || !question) return;

  if (!session || !tokenizer) {
    answerDiv.textContent = 'Model not loaded yet. Please wait.';
    return;
  }

  updateStatus('${config.modelName} \\u00b7 Processing...');
  const start = performance.now();

  const combined = table + ' [SEP] ' + question;
  const { inputIds, attentionMask } = tokenizeText(tokenizer, combined);
  const output = await runInference(session, inputIds);
  const result = postprocessTableQA(output, inputIds, tokenizer);

  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus('${config.modelName} \\u00b7 ' + elapsed + 'ms \\u00b7 ' + getBackendLabel(session));

  answerDiv.innerHTML = '<div>' + result.answer + '</div><div class="score">Confidence: ' + (result.score * 100).toFixed(1) + '%</div>';
});

init();`;
}

// ---- Image-to-Text script ----

function emitImageToTextScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  const remote = isRemoteModel(config);
  const modelPath = getModelPath(config, '.');
  const modelName = config.modelName;

  const handleFile = `
function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  const url = URL.createObjectURL(file);
  previewImage.src = url;
  previewImage.onload = () => processImage(previewImage);
  preview.hidden = false;${!remote ? '\n  dropZone.hidden = true;' : ''}
}`;

  return `${emitBlockCode(config, blocks)}
${remote ? emitProgressHelpers() : ''}

// --- Application ---
const MODEL_PATH = '${modelPath}';
let session = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const previewImage = document.getElementById('previewImage');
const resultsDiv = document.getElementById('output');

${remote ? emitRunButtonListeners() : emitDropZoneListeners()}

${handleFile}

async function processImage(img) {
  if (!session) return;
  updateStatus('${modelName} \\u00b7 Generating caption...');
  const start = performance.now();

  const input = preprocessImage(img);
  const output = await runInference(session, input);
  const caption = postprocessImageToText(output);

  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus('${modelName} \\u00b7 ' + elapsed + 'ms \\u00b7 ' + getBackendLabel(session));

  resultsDiv.textContent = caption;
}

${remote ? emitRemoteInit(modelName) : emitLocalInit(modelName)}

init();`;
}

// ---- Visual Question Answering script ----

function emitVQAScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  const remote = isRemoteModel(config);
  const modelPath = getModelPath(config, '.');
  const modelName = config.modelName;

  const handleFile = `
function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  const url = URL.createObjectURL(file);
  previewImage.src = url;
  previewImage.onload = () => { currentImage = previewImage; };
  preview.hidden = false;${!remote ? '\n  dropZone.hidden = true;' : ''}
}`;

  return `${emitBlockCode(config, blocks)}
${remote ? emitProgressHelpers() : ''}

// --- Application ---
const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';
let session = null;
let tokenizer = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const previewImage = document.getElementById('previewImage');
const questionInput = document.getElementById('questionInput');
const askBtn = document.getElementById('askBtn');
const resultsDiv = document.getElementById('answer');
let currentImage = null;

${remote ? emitRunButtonListeners() : emitDropZoneListeners('\n  currentImage = null;')}

${handleFile}

askBtn.addEventListener('click', async () => {
  const question = questionInput.value.trim();
  if (!question || !currentImage) return;

  if (!session || !tokenizer) {
    resultsDiv.textContent = 'Model not loaded yet. Please wait.';
    return;
  }

  updateStatus('${modelName} \\u00b7 Processing...');
  const start = performance.now();

  const imageInput = preprocessImage(currentImage);
  const output = await runInference(session, imageInput);
  const answer = postprocessVQA(output, tokenizer);

  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus('${modelName} \\u00b7 ' + elapsed + 'ms \\u00b7 ' + getBackendLabel(session));

  resultsDiv.textContent = answer;
});

${remote ? emitRemoteInit(modelName, { tokenizer: true }) : emitLocalInit(modelName, { tokenizer: true })}

init();`;
}

// ---- Document Question Answering script ----

function emitDocQAScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  const remote = isRemoteModel(config);
  const modelPath = getModelPath(config, '.');
  const modelName = config.modelName;

  const handleFile = `
function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  const url = URL.createObjectURL(file);
  previewImage.src = url;
  previewImage.onload = () => { currentImage = previewImage; };
  preview.hidden = false;${!remote ? '\n  dropZone.hidden = true;' : ''}
}`;

  return `${emitBlockCode(config, blocks)}
${remote ? emitProgressHelpers() : ''}

// --- Application ---
const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';
let session = null;
let tokenizer = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const previewImage = document.getElementById('previewImage');
const questionInput = document.getElementById('questionInput');
const askBtn = document.getElementById('askBtn');
const resultsDiv = document.getElementById('answer');
let currentImage = null;

${remote ? emitRunButtonListeners() : emitDropZoneListeners('\n  currentImage = null;')}

${handleFile}

askBtn.addEventListener('click', async () => {
  const question = questionInput.value.trim();
  if (!question || !currentImage) return;

  if (!session || !tokenizer) {
    resultsDiv.textContent = 'Model not loaded yet. Please wait.';
    return;
  }

  updateStatus('${modelName} \\u00b7 Processing...');
  const start = performance.now();

  const imageInput = preprocessImage(currentImage);
  const output = await runInference(session, imageInput);
  const answer = postprocessDocQA(output, tokenizer);

  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus('${modelName} \\u00b7 ' + elapsed + 'ms \\u00b7 ' + getBackendLabel(session));

  resultsDiv.textContent = answer;
});

${remote ? emitRemoteInit(modelName, { tokenizer: true }) : emitLocalInit(modelName, { tokenizer: true })}

init();`;
}

// ---- Image-Text-to-Text script ----

function emitImageTextToTextScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  const remote = isRemoteModel(config);
  const modelPath = getModelPath(config, '.');
  const modelName = config.modelName;

  const handleFile = `
function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  const url = URL.createObjectURL(file);
  previewImage.src = url;
  previewImage.onload = () => { currentImage = previewImage; };
  preview.hidden = false;${!remote ? '\n  dropZone.hidden = true;' : ''}
}`;

  return `${emitBlockCode(config, blocks)}
${remote ? emitProgressHelpers() : ''}

// --- Application ---
const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';
let session = null;
let tokenizer = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const previewImage = document.getElementById('previewImage');
const promptInput = document.getElementById('promptInput');
const generateBtn = document.getElementById('generateBtn');
const resultsDiv = document.getElementById('output');
let currentImage = null;

${remote ? emitRunButtonListeners() : emitDropZoneListeners('\n  currentImage = null;')}

${handleFile}

generateBtn.addEventListener('click', async () => {
  const prompt = promptInput.value.trim();
  if (!currentImage) return;

  if (!session || !tokenizer) {
    resultsDiv.textContent = 'Model not loaded yet. Please wait.';
    return;
  }

  generateBtn.disabled = true;
  updateStatus('${modelName} \\u00b7 Generating...');
  const start = performance.now();

  const imageInput = preprocessImage(currentImage);
  const output = await runInference(session, imageInput);
  const result = postprocessImageTextToText(output, tokenizer);

  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus('${modelName} \\u00b7 ' + elapsed + 'ms \\u00b7 ' + getBackendLabel(session));

  resultsDiv.textContent = result;
  generateBtn.disabled = false;
});

${remote ? emitRemoteInit(modelName, { tokenizer: true }) : emitLocalInit(modelName, { tokenizer: true })}

init();`;
}

// ---- Audio-to-Audio script ----

function emitAudioToAudioScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  const remote = isRemoteModel(config);
  const modelPath = getModelPath(config, '.');
  const modelName = config.modelName;

  return `${emitBlockCode(config, blocks)}
${remote ? emitProgressHelpers() : ''}

// --- Application ---
const MODEL_PATH = '${modelPath}';
let session = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

${remote ? emitRemoteInit(modelName, { enableRunBtn: false }) : emitLocalInit(modelName)}

const fileInput = document.getElementById('fileInput');
const outputDiv = document.getElementById('output');

fileInput.addEventListener('change', async () => {
  const file = fileInput.files[0];
  if (!file || !session) return;

  updateStatus('${config.modelName} \\u00b7 Processing...');
  const start = performance.now();

  const audioCtx = new AudioContext({ sampleRate: 16000 });
  const arrayBuffer = await file.arrayBuffer();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  const samples = audioBuffer.getChannelData(0);

  const input = preprocessAudio(samples);
  const output = await runInference(session, input);
  const processedSamples = postprocessAudioToAudio(output);
  await playAudio(processedSamples);

  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus('${config.modelName} \\u00b7 ' + elapsed + 'ms \\u00b7 ' + getBackendLabel(session));
  outputDiv.textContent = 'Processed ' + (samples.length / 16000).toFixed(1) + 's of audio. Playing output...';
});

init();`;
}

// ---- Speaker Diarization script ----

function emitSpeakerDiarizationScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  const remote = isRemoteModel(config);
  const modelPath = getModelPath(config, '.');
  const modelName = config.modelName;

  return `${emitBlockCode(config, blocks)}
${remote ? emitProgressHelpers() : ''}

// --- Application ---
const MODEL_PATH = '${modelPath}';
let session = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

${remote ? emitRemoteInit(modelName, { enableRunBtn: false }) : emitLocalInit(modelName)}

const fileInput = document.getElementById('fileInput');
const resultsDiv = document.getElementById('results');

fileInput.addEventListener('change', async () => {
  const file = fileInput.files[0];
  if (!file || !session) return;

  updateStatus('${config.modelName} \\u00b7 Processing...');
  const start = performance.now();

  const audioCtx = new AudioContext({ sampleRate: 16000 });
  const arrayBuffer = await file.arrayBuffer();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  const samples = audioBuffer.getChannelData(0);

  const input = preprocessAudio(samples);
  const output = await runInference(session, input);
  const segments = postprocessSpeakerDiarization(output);

  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus('${config.modelName} \\u00b7 ' + elapsed + 'ms \\u00b7 ' + getBackendLabel(session));

  resultsDiv.innerHTML = '';
  for (const seg of segments) {
    const row = document.createElement('div');
    row.className = 'diarization-segment';
    row.innerHTML = '<span class="speaker">Speaker ' + seg.speaker + '</span><span>' + seg.text + '</span><span class="time">' + seg.start.toFixed(1) + 's - ' + seg.end.toFixed(1) + 's</span>';
    resultsDiv.appendChild(row);
  }
});

init();`;
}

// ---- Voice Activity Detection script ----

function emitVADScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  const remote = isRemoteModel(config);
  const modelPath = getModelPath(config, '.');
  const modelName = config.modelName;

  return `${emitBlockCode(config, blocks)}
${remote ? emitProgressHelpers() : ''}

// --- Application ---
const MODEL_PATH = '${modelPath}';
let session = null;

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

${remote ? emitRemoteInit(modelName, { enableRunBtn: false }) : emitLocalInit(modelName)}

const fileInput = document.getElementById('fileInput');
const resultsDiv = document.getElementById('results');

fileInput.addEventListener('change', async () => {
  const file = fileInput.files[0];
  if (!file || !session) return;

  updateStatus('${config.modelName} \\u00b7 Processing...');
  const start = performance.now();

  const audioCtx = new AudioContext({ sampleRate: 16000 });
  const arrayBuffer = await file.arrayBuffer();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  const samples = audioBuffer.getChannelData(0);

  const input = preprocessAudio(samples);
  const output = await runInference(session, input);
  const segments = postprocessVAD(output);

  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus('${config.modelName} \\u00b7 ' + elapsed + 'ms \\u00b7 ' + getBackendLabel(session));

  resultsDiv.innerHTML = '';
  for (const seg of segments) {
    const row = document.createElement('div');
    row.className = 'vad-segment';
    row.innerHTML = '<span class="label">' + seg.label + '</span><span class="time">' + seg.start.toFixed(1) + 's - ' + seg.end.toFixed(1) + 's</span>';
    resultsDiv.appendChild(row);
  }
});

init();`;
}

// ---- Script dispatcher ----

function emitAppScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  const task = config.task;
  switch (task) {
    // Audio tasks
    case 'audio-classification':
      return config.input === 'mic' ? emitRealtimeAudioClassificationScript(config, blocks) : emitFileAudioClassificationScript(config, blocks);
    case 'speech-to-text':
      return config.input === 'mic' ? emitRealtimeSpeechToTextScript(config, blocks) : emitFileSpeechToTextScript(config, blocks);
    case 'text-to-speech':
      return emitTextToSpeechScript(config, blocks);
    case 'audio-to-audio':
      return emitAudioToAudioScript(config, blocks);
    case 'speaker-diarization':
      return emitSpeakerDiarizationScript(config, blocks);
    case 'voice-activity-detection':
      return emitVADScript(config, blocks);
    // Text tasks
    case 'text-classification':
      return emitTextClassificationScript(config, blocks);
    case 'zero-shot-classification':
      return emitZeroShotScript(config, blocks);
    case 'text-generation':
      return emitTextGenerationScript(config, blocks);
    case 'fill-mask':
      return emitFillMaskScript(config, blocks);
    case 'sentence-similarity':
      return emitSentenceSimilarityScript(config, blocks);
    case 'token-classification':
      return emitTokenClassificationScript(config, blocks);
    case 'question-answering':
      return emitQuestionAnsweringScript(config, blocks);
    case 'summarization':
      return emitSummarizationScript(config, blocks);
    case 'translation':
      return emitTranslationScript(config, blocks);
    case 'text2text-generation':
      return emitText2TextScript(config, blocks);
    case 'conversational':
      return emitConversationalScript(config, blocks);
    case 'table-question-answering':
      return emitTableQAScript(config, blocks);
    // Multimodal tasks
    case 'image-to-text':
      return emitImageToTextScript(config, blocks);
    case 'visual-question-answering':
      return emitVQAScript(config, blocks);
    case 'document-question-answering':
      return emitDocQAScript(config, blocks);
    case 'image-text-to-text':
      return emitImageTextToTextScript(config, blocks);
    // Image tasks: dispatch by input mode
    case 'depth-estimation':
      return emitDepthEstimationScript(config, blocks);
    default: {
      if (config.input === 'camera' || config.input === 'screen') {
        return emitRealtimeScript(config, blocks);
      }
      if (config.input === 'file') {
        switch (task) {
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
      return emitFileClassificationScript(config, blocks);
    }
  }
}

// ---- HTML body content ----

/** File input body (classification tasks) — source-aware */
function emitFileClassificationBody(config: ResolvedConfig): string {
  const taskLabel = getTaskLabel(config.task);
  const remote = isRemoteModel(config);

  const inputArea = remote
    ? `${emitProgressBarHtml()}

        <div id="preview" class="preview" hidden>
          <img id="previewImage" alt="Selected image for classification">
        </div>

        <button id="runBtn" class="run-model-btn" disabled>&#x25b6; Run Inference</button>
        <input type="file" id="fileInput" accept="image/*" style="position:absolute;left:-9999px;width:1px;height:1px;opacity:0;" aria-hidden="true" tabindex="-1">`
    : `<div class="drop-zone" id="dropZone" role="button" tabindex="0"
             aria-label="Drop an image here or click to browse for ${taskLabel.toLowerCase()}">
          <p>Drop an image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input type="file" id="fileInput" accept="image/*" style="position:absolute;left:-9999px;width:1px;height:1px;opacity:0;" aria-hidden="true" tabindex="-1">
        </div>

        <div id="preview" class="preview" hidden>
          <img id="previewImage" alt="Selected image for classification">
          <button id="changeBtn" class="change-btn">Choose another image</button>
        </div>`;

  return `    <div class="container">
      <div>
        ${inputArea}
      </div>

      <div id="results" class="results" role="status" aria-live="polite" aria-atomic="true">
      </div>
    </div>`;
}

/** File input body with canvas overlay (detection/segmentation) */
function emitFileOverlayBody(config: ResolvedConfig): string {
  const taskLabel = getTaskLabel(config.task);
  const remote = isRemoteModel(config);

  const inputArea = remote
    ? `${emitProgressBarHtml()}

        <div id="preview" class="preview" hidden>
          <div class="preview-wrapper">
            <img id="previewImage" alt="Selected image for ${taskLabel.toLowerCase()}">
            <canvas id="overlay"></canvas>
          </div>
        </div>

        <button id="runBtn" class="run-model-btn" disabled>&#x25b6; Run Inference</button>
        <input type="file" id="fileInput" accept="image/*" style="position:absolute;left:-9999px;width:1px;height:1px;opacity:0;" aria-hidden="true" tabindex="-1">`
    : `<div class="drop-zone" id="dropZone" role="button" tabindex="0"
             aria-label="Drop an image here or click to browse for ${taskLabel.toLowerCase()}">
          <p>Drop an image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input type="file" id="fileInput" accept="image/*" style="position:absolute;left:-9999px;width:1px;height:1px;opacity:0;" aria-hidden="true" tabindex="-1">
        </div>

        <div id="preview" class="preview" hidden>
          <div class="preview-wrapper">
            <img id="previewImage" alt="Selected image for ${taskLabel.toLowerCase()}">
            <canvas id="overlay"></canvas>
          </div>
          <button id="changeBtn" class="change-btn">Choose another image</button>
        </div>`;

  return `    <div class="container">
      <div>
        ${inputArea}
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

/** Fill-Mask body */
function emitFillMaskBody(config: ResolvedConfig): string {
  const taskLabel = getTaskLabel(config.task);
  return `    <h2>${taskLabel}</h2>

    <div class="text-input">
      <label for="textInput">Enter text with [MASK] token</label>
      <textarea id="textInput" rows="4" aria-label="Text with mask token">The capital of France is [MASK].</textarea>
      <button id="predictBtn" class="run-btn" aria-label="Predict masked token">Predict</button>
    </div>

    <div id="results" class="mask-predictions" role="status" aria-live="polite" aria-atomic="true">
    </div>`;
}

/** Sentence Similarity body */
function emitSentenceSimilarityBody(config: ResolvedConfig): string {
  const taskLabel = getTaskLabel(config.task);
  return `    <h2>${taskLabel}</h2>

    <div class="text-input">
      <label for="sourceInput">Source sentence</label>
      <textarea id="sourceInput" rows="2" aria-label="Source sentence">The weather is lovely today.</textarea>
      <label for="compareInput">Sentences to compare (one per line)</label>
      <textarea id="compareInput" rows="4" aria-label="Comparison sentences">It is a beautiful day.
The sun is shining bright.
I need to buy groceries.</textarea>
      <button id="compareBtn" class="run-btn" aria-label="Compare similarity">Compare</button>
    </div>

    <div id="results" class="similarity-pairs" role="status" aria-live="polite" aria-atomic="true">
    </div>`;
}

/** Depth Estimation body */
function emitDepthEstimationBody(config: ResolvedConfig): string {
  const taskLabel = getTaskLabel(config.task);
  const remote = isRemoteModel(config);

  const inputArea = remote
    ? `${emitProgressBarHtml()}

        <div id="preview" class="preview" hidden>
          <img id="previewImage" alt="Selected image for depth estimation">
        </div>

        <button id="runBtn" class="run-model-btn" disabled>&#x25b6; Run Inference</button>
        <input type="file" id="fileInput" accept="image/*" style="position:absolute;left:-9999px;width:1px;height:1px;opacity:0;" aria-hidden="true" tabindex="-1">`
    : `<div class="drop-zone" id="dropZone" role="button" tabindex="0"
             aria-label="Drop an image here or click to browse for ${taskLabel.toLowerCase()}">
          <p>Drop an image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input type="file" id="fileInput" accept="image/*" style="position:absolute;left:-9999px;width:1px;height:1px;opacity:0;" aria-hidden="true" tabindex="-1">
        </div>

        <div id="preview" class="preview" hidden>
          <img id="previewImage" alt="Selected image for depth estimation">
          <button id="changeBtn" class="change-btn">Choose another image</button>
        </div>`;

  return `    <div class="container">
      <div>
        ${inputArea}
      </div>

      <div>
        <canvas id="depthCanvas" class="depth-canvas" role="img" aria-label="Depth estimation output"></canvas>
      </div>
    </div>`;
}

/** Token Classification (NER) body */
function emitTokenClassificationBody(config: ResolvedConfig): string {
  const taskLabel = getTaskLabel(config.task);
  return `    <h2>${taskLabel}</h2>

    <div class="text-input">
      <label for="textInput">Enter text to analyze</label>
      <textarea id="textInput" rows="4" aria-label="Text for named entity recognition">John Smith works at Google in Mountain View, California.</textarea>
      <button id="analyzeBtn" class="run-btn" aria-label="Analyze entities">Analyze</button>
    </div>

    <div id="results" class="ner-output" role="status" aria-live="polite" aria-atomic="true">
    </div>`;
}

/** Question Answering body */
function emitQuestionAnsweringBody(config: ResolvedConfig): string {
  const taskLabel = getTaskLabel(config.task);
  return `    <h2>${taskLabel}</h2>

    <div class="qa-input">
      <label for="contextInput">Context</label>
      <textarea id="contextInput" rows="4" class="text-input" aria-label="Context passage">The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair.</textarea>
      <label for="questionInput">Question</label>
      <input type="text" id="questionInput" class="labels-input" value="When was the Eiffel Tower built?" aria-label="Question about the context">
      <button id="answerBtn" class="run-btn" aria-label="Find answer">Answer</button>
    </div>

    <div id="answer" class="qa-answer" role="status" aria-live="polite" aria-atomic="true">
    </div>`;
}

/** Summarization body */
function emitSummarizationBody(config: ResolvedConfig): string {
  const taskLabel = getTaskLabel(config.task);
  return `    <h2>${taskLabel}</h2>

    <div class="text-input">
      <label for="textInput">Enter text to summarize</label>
      <textarea id="textInput" rows="6" aria-label="Text to summarize">Artificial intelligence has transformed many industries. Machine learning models can now process natural language, recognize images, and generate creative content. These advances have led to applications in healthcare, finance, education, and entertainment.</textarea>
      <button id="summarizeBtn" class="run-btn" aria-label="Summarize text">Summarize</button>
    </div>

    <div id="output" class="generation-output" role="status" aria-live="polite" aria-atomic="true">
    </div>`;
}

/** Translation body */
function emitTranslationBody(config: ResolvedConfig): string {
  const taskLabel = getTaskLabel(config.task);
  return `    <h2>${taskLabel}</h2>

    <div class="text-input">
      <label for="textInput">Enter text to translate</label>
      <textarea id="textInput" rows="4" aria-label="Text to translate">Hello, how are you today?</textarea>
      <button id="translateBtn" class="run-btn" aria-label="Translate text">Translate</button>
    </div>

    <div id="output" class="generation-output" role="status" aria-live="polite" aria-atomic="true">
    </div>`;
}

/** Text2Text Generation body */
function emitText2TextBody(config: ResolvedConfig): string {
  const taskLabel = getTaskLabel(config.task);
  return `    <h2>${taskLabel}</h2>

    <div class="text-input">
      <label for="textInput">Enter input text</label>
      <textarea id="textInput" rows="4" aria-label="Input text">Paraphrase: The house is big and beautiful.</textarea>
      <button id="runBtn" class="run-btn" aria-label="Process text">Run</button>
    </div>

    <div id="output" class="generation-output" role="status" aria-live="polite" aria-atomic="true">
    </div>`;
}

/** Conversational body */
function emitConversationalBody(config: ResolvedConfig): string {
  const taskLabel = getTaskLabel(config.task);
  return `    <h2>${taskLabel}</h2>

    <div id="chatMessages" class="chat-messages" role="log" aria-live="polite" aria-atomic="false">
    </div>

    <div class="chat-input-row">
      <input type="text" id="chatInput" placeholder="Type a message..." aria-label="Chat message input">
      <button id="sendBtn" class="run-btn" aria-label="Send message">Send</button>
    </div>`;
}

/** Table QA body */
function emitTableQABody(config: ResolvedConfig): string {
  const taskLabel = getTaskLabel(config.task);
  return `    <h2>${taskLabel}</h2>

    <div class="qa-input">
      <label for="tableInput">Table data (CSV format)</label>
      <div class="table-input">
        <textarea id="tableInput" rows="4" aria-label="Table data in CSV format">Name, Age, City
Alice, 30, New York
Bob, 25, San Francisco
Charlie, 35, Chicago</textarea>
      </div>
      <label for="questionInput">Question</label>
      <input type="text" id="questionInput" class="labels-input" value="Who lives in San Francisco?" aria-label="Question about the table">
      <button id="answerBtn" class="run-btn" aria-label="Find answer">Answer</button>
    </div>

    <div id="answer" class="qa-answer" role="status" aria-live="polite" aria-atomic="true">
    </div>`;
}

/** Image-to-Text body */
function emitImageToTextBody(config: ResolvedConfig): string {
  const taskLabel = getTaskLabel(config.task);
  return `    <div class="container">
      <div>
        <div class="drop-zone" id="dropZone" role="button" tabindex="0"
             aria-label="Drop an image here or click to browse for ${taskLabel.toLowerCase()}">
          <p>Drop an image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input type="file" id="fileInput" accept="image/*" style="position:absolute;left:-9999px;width:1px;height:1px;opacity:0;" aria-hidden="true" tabindex="-1">
        </div>

        <div id="preview" class="preview" hidden>
          <img id="previewImage" alt="Selected image for captioning">
          <button id="changeBtn" class="change-btn">Choose another image</button>
        </div>
      </div>

      <div id="output" class="generation-output" role="status" aria-live="polite" aria-atomic="true">
      </div>
    </div>`;
}

/** Visual Question Answering body */
function emitVQABody(config: ResolvedConfig): string {
  const taskLabel = getTaskLabel(config.task);
  return `    <h2>${taskLabel}</h2>

    <div class="container">
      <div class="multimodal-input">
        <div class="drop-zone" id="dropZone" role="button" tabindex="0"
             aria-label="Drop an image here or click to browse">
          <p>Drop an image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input type="file" id="fileInput" accept="image/*" style="position:absolute;left:-9999px;width:1px;height:1px;opacity:0;" aria-hidden="true" tabindex="-1">
        </div>

        <div id="preview" class="preview" hidden>
          <img id="previewImage" alt="Selected image for VQA">
          <button id="changeBtn" class="change-btn">Choose another image</button>
        </div>

        <input type="text" id="questionInput" class="question-input" placeholder="Ask a question about the image..." value="What is in this image?" aria-label="Question about the image">
        <button id="askBtn" class="run-btn" aria-label="Ask question">Ask</button>
      </div>

      <div id="answer" class="qa-answer" role="status" aria-live="polite" aria-atomic="true">
      </div>
    </div>`;
}

/** Document Question Answering body */
function emitDocQABody(config: ResolvedConfig): string {
  const taskLabel = getTaskLabel(config.task);
  return `    <h2>${taskLabel}</h2>

    <div class="container">
      <div class="multimodal-input">
        <div class="drop-zone" id="dropZone" role="button" tabindex="0"
             aria-label="Drop a document image here or click to browse">
          <p>Drop a document image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input type="file" id="fileInput" accept="image/*" style="position:absolute;left:-9999px;width:1px;height:1px;opacity:0;" aria-hidden="true" tabindex="-1">
        </div>

        <div id="preview" class="preview" hidden>
          <img id="previewImage" alt="Selected document image">
          <button id="changeBtn" class="change-btn">Choose another image</button>
        </div>

        <input type="text" id="questionInput" class="question-input" placeholder="Ask a question about the document..." value="What is the total amount?" aria-label="Question about the document">
        <button id="askBtn" class="run-btn" aria-label="Ask question">Ask</button>
      </div>

      <div id="answer" class="qa-answer" role="status" aria-live="polite" aria-atomic="true">
      </div>
    </div>`;
}

/** Image-Text-to-Text body */
function emitImageTextToTextBody(config: ResolvedConfig): string {
  const taskLabel = getTaskLabel(config.task);
  return `    <h2>${taskLabel}</h2>

    <div class="container">
      <div class="multimodal-input">
        <div class="drop-zone" id="dropZone" role="button" tabindex="0"
             aria-label="Drop an image here or click to browse">
          <p>Drop an image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input type="file" id="fileInput" accept="image/*" style="position:absolute;left:-9999px;width:1px;height:1px;opacity:0;" aria-hidden="true" tabindex="-1">
        </div>

        <div id="preview" class="preview" hidden>
          <img id="previewImage" alt="Selected image">
          <button id="changeBtn" class="change-btn">Choose another image</button>
        </div>

        <input type="text" id="promptInput" class="question-input" placeholder="Enter a prompt..." value="Describe this image in detail." aria-label="Text prompt for the image">
        <button id="generateBtn" class="run-btn" aria-label="Generate text">Generate</button>
      </div>

      <div id="output" class="generation-output" role="status" aria-live="polite" aria-atomic="true">
      </div>
    </div>`;
}

/** Audio-to-Audio body */
function emitAudioToAudioBody(config: ResolvedConfig): string {
  const taskLabel = getTaskLabel(config.task);
  return `    <h2>${taskLabel}</h2>

    <div>
      <label for="fileInput">Choose an audio file &#9835;</label>
      <input type="file" id="fileInput" accept="audio/*" aria-label="Select audio file for ${taskLabel.toLowerCase()}">
    </div>

    <div id="output" class="generation-output" role="status" aria-live="polite" aria-atomic="true">
    </div>`;
}

/** Speaker Diarization body */
function emitSpeakerDiarizationBody(config: ResolvedConfig): string {
  const taskLabel = getTaskLabel(config.task);
  return `    <h2>${taskLabel}</h2>

    <div>
      <label for="fileInput">Choose an audio file &#9835;</label>
      <input type="file" id="fileInput" accept="audio/*" aria-label="Select audio file for ${taskLabel.toLowerCase()}">
    </div>

    <div id="results" class="diarization-timeline" role="status" aria-live="polite" aria-atomic="true">
    </div>`;
}

/** Voice Activity Detection body */
function emitVADBody(config: ResolvedConfig): string {
  const taskLabel = getTaskLabel(config.task);
  return `    <h2>${taskLabel}</h2>

    <div>
      <label for="fileInput">Choose an audio file &#9835;</label>
      <input type="file" id="fileInput" accept="audio/*" aria-label="Select audio file for ${taskLabel.toLowerCase()}">
    </div>

    <div id="results" class="vad-segments" role="status" aria-live="polite" aria-atomic="true">
    </div>`;
}

function emitBodyContent(config: ResolvedConfig): string {
  switch (config.task) {
    // Audio tasks
    case 'text-to-speech':
      return emitTtsBody();
    case 'speech-to-text':
    case 'audio-classification':
      return config.input === 'mic' ? emitAudioMicBody(config) : emitAudioFileBody(config);
    case 'audio-to-audio':
      return emitAudioToAudioBody(config);
    case 'speaker-diarization':
      return emitSpeakerDiarizationBody(config);
    case 'voice-activity-detection':
      return emitVADBody(config);
    // Text tasks
    case 'text-classification':
      return emitTextClassificationBody(config);
    case 'zero-shot-classification':
      return emitZeroShotBody(config);
    case 'text-generation':
      return emitTextGenerationBody(config);
    case 'fill-mask':
      return emitFillMaskBody(config);
    case 'sentence-similarity':
      return emitSentenceSimilarityBody(config);
    case 'token-classification':
      return emitTokenClassificationBody(config);
    case 'question-answering':
      return emitQuestionAnsweringBody(config);
    case 'summarization':
      return emitSummarizationBody(config);
    case 'translation':
      return emitTranslationBody(config);
    case 'text2text-generation':
      return emitText2TextBody(config);
    case 'conversational':
      return emitConversationalBody(config);
    case 'table-question-answering':
      return emitTableQABody(config);
    // Multimodal tasks
    case 'image-to-text':
      return emitImageToTextBody(config);
    case 'visual-question-answering':
      return emitVQABody(config);
    case 'document-question-answering':
      return emitDocQABody(config);
    case 'image-text-to-text':
      return emitImageTextToTextBody(config);
    // Depth estimation
    case 'depth-estimation':
      return emitDepthEstimationBody(config);
    default:
      break;
  }

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
    config.task === 'feature-extraction' || config.task === 'depth-estimation' ||
    config.task === 'image-to-text' || config.task === 'visual-question-answering' ||
    config.task === 'document-question-answering' || config.task === 'image-text-to-text';
}

function needsAudioCSS(config: ResolvedConfig): boolean {
  return isAudioTask(config.task);
}

/**
 * Generate the backend <select> HTML for the generated page.
 * Options and defaults depend on the engine:
 * - ORT: WebNN NPU, WebNN GPU (default), WebNN CPU, WebGPU, Wasm
 * - LiteRT: WebGPU (default), Wasm; WebNN options disabled (coming later)
 * - WebNN: WebNN NPU, WebNN GPU (default), WebNN CPU
 */
function emitBackendSelect(engine: string): string {
  if (engine === 'litert') {
    return `<select id="backend" class="backend-select" aria-label="Backend">
      <option value="webnn-npu" disabled>WebNN NPU (coming soon)</option>
      <option value="webnn-gpu" disabled>WebNN GPU (coming soon)</option>
      <option value="webnn-cpu" disabled>WebNN CPU (coming soon)</option>
      <option value="webgpu" selected>WebGPU</option>
      <option value="wasm">Wasm</option>
    </select>`;
  }
  if (engine === 'webnn') {
    return `<select id="backend" class="backend-select" aria-label="Backend">
      <option value="webnn-npu">WebNN NPU</option>
      <option value="webnn-gpu" selected>WebNN GPU</option>
      <option value="webnn-cpu">WebNN CPU</option>
    </select>`;
  }
  // ORT (default)
  return `<select id="backend" class="backend-select" aria-label="Backend">
      <option value="webnn-npu">WebNN NPU</option>
      <option value="webnn-gpu" selected>WebNN GPU</option>
      <option value="webnn-cpu">WebNN CPU</option>
      <option value="webgpu">WebGPU</option>
      <option value="wasm">Wasm</option>
    </select>`;
}

/**
 * Emit HTML framework files.
 */
export function emitHtml(config: ResolvedConfig, blocks: CodeBlock[]): GeneratedFile[] {
  const engineLabel = getEngineLabel(config.engine);
  const theme = config.theme;
  const { titleText, headingHtml } = buildPageHeading(config);

  const designCSS = emitDesignSystemCSS(config);
  const appCSS = emitAppCSS();
  const extraCSS = needsExtendedCSS(config) ? emitExtendedCSS() : '';
  const audioCSS = needsAudioCSS(config) ? emitAudioCSS() : '';
  const appScript = emitAppScript(config, blocks);
  const bodyContent = emitBodyContent(config);
  const backendSelect = emitBackendSelect(config.engine);

  const html = `<!DOCTYPE html>
<html lang="en" data-theme="${theme}">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${titleText}</title>
  <style>
${designCSS}
${appCSS}${extraCSS}${audioCSS}
  </style>
</head>
<body>
  <a href="#results" class="skip-link">Skip to results</a>

  <main>
    ${headingHtml}

${bodyContent}
  </main>

  <aside class="status-bar">
    <span id="status">${config.modelName} · Loading...</span>
    ${backendSelect}
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
