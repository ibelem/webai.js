/**
 * Vanilla-Vite framework emitter (Layer 2).
 *
 * Produces a Vite project with plain JS/TS (no framework):
 *   package.json, vite.config.js, index.html,
 *   src/main.{js|ts}, src/style.css,
 *   src/lib/input.{js|ts} (if non-file input),
 *   src/lib/preprocess.{js|ts}, src/lib/inference.{js|ts},
 *   src/lib/postprocess.{js|ts}, README.md
 *
 * Uses ES modules with Vite dev server. No framework dependency.
 */

import type { ResolvedConfig } from '@webai/core';
import type { CodeBlock, GeneratedFile } from '../types.js';
import {
  emitDesignSystemCSS,
  emitAppCSS,
  addExports,
  findBlock,
  collectImports,
  collectAuxiliaryFiles,
  emitReadme,
  getTaskLabel,
  getEngineLabel,
  getModelPath,
} from './shared.js';

const libExt = (config: ResolvedConfig) => (config.lang === 'ts' ? 'ts' : 'js');

function emitPackageJson(config: ResolvedConfig, blocks: CodeBlock[]): string {
  const deps: Record<string, string> = {};
  for (const imp of collectImports(blocks)) {
    if (imp === 'onnxruntime-web') deps[imp] = '^1.21.0';
    else deps[imp] = 'latest';
  }

  const devDeps: Record<string, string> = {
    vite: '^6.0.0',
  };
  if (config.lang === 'ts') {
    devDeps['typescript'] = '^5.7.0';
  }

  return JSON.stringify(
    {
      name: config.modelName,
      private: true,
      version: '0.0.0',
      type: 'module',
      scripts: {
        dev: 'vite',
        build: 'vite build',
        preview: 'vite preview',
      },
      dependencies: deps,
      devDependencies: devDeps,
    },
    null,
    2,
  );
}

function emitViteConfig(): string {
  return `import { defineConfig } from 'vite';

export default defineConfig({});
`;
}

function emitFileClassificationBody(config: ResolvedConfig): string {
  const taskLabel = getTaskLabel(config.task);
  return `      <div class="container">
        <div>
          <div class="drop-zone" id="dropZone" role="button" tabindex="0"
               aria-label="Drop an image here or click to browse for ${taskLabel.toLowerCase()}">
            <p>Drop an image here or click to browse</p>
            <p class="hint">Supports JPG, PNG, WebP</p>
            <input type="file" id="fileInput" accept="image/*" hidden>
          </div>
          <div id="preview" class="preview" hidden>
            <img id="previewImage" alt="Selected image for ${taskLabel.toLowerCase()}">
            <button id="changeBtn" class="change-btn">Choose another image</button>
          </div>
        </div>
        <div id="results" class="results" role="status" aria-live="polite" aria-atomic="true">
        </div>
      </div>`;
}

function emitFileOverlayBody(config: ResolvedConfig): string {
  const taskLabel = getTaskLabel(config.task);
  return `      <div class="container">
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

function emitRealtimeBody(config: ResolvedConfig): string {
  const isScreen = config.input === 'screen';
  const actionLabel = isScreen ? 'capture your screen' : 'use your camera';
  const btnLabel = isScreen ? 'Start Screen Capture' : 'Enable Camera';
  const taskLabel = getTaskLabel(config.task);
  return `      <div id="permissionPrompt" class="permission-prompt">
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

function emitAudioFileBody(config: ResolvedConfig): string {
  const taskLabel = getTaskLabel(config.task);
  const isSTT = config.task === 'speech-to-text';
  const resultsContent = isSTT
    ? `      <pre id="transcript" class="transcript" role="status" aria-live="polite" aria-atomic="true"></pre>`
    : `      <div id="results" class="results" role="status" aria-live="polite" aria-atomic="true">\n      </div>`;
  return `      <h2>${taskLabel}</h2>
      <div>
        <label for="fileInput">Choose an audio file</label>
        <input type="file" id="fileInput" accept="audio/*" aria-label="Select audio file for ${taskLabel.toLowerCase()}">
      </div>
${resultsContent}`;
}

function emitAudioMicBody(config: ResolvedConfig): string {
  const taskLabel = getTaskLabel(config.task);
  const isSTT = config.task === 'speech-to-text';
  const resultsContent = isSTT
    ? `      <pre id="transcript" class="transcript" role="status" aria-live="polite" aria-atomic="true">(listening...)</pre>`
    : `      <div id="results" class="results" role="status" aria-live="polite" aria-atomic="true">\n      </div>`;
  return `      <h2>${taskLabel}</h2>
      <div class="controls" role="group" aria-label="Recording controls">
        <button id="startBtn" aria-label="Start recording">Start Recording</button>
        <button id="stopBtn" disabled aria-label="Stop recording">Stop Recording</button>
      </div>
${resultsContent}`;
}

function emitTtsBody(): string {
  return `      <h2>Text to Speech</h2>
      <div class="tts-input">
        <label for="textInput">Enter text to synthesize</label>
        <textarea id="textInput" rows="4" aria-label="Text to synthesize">Hello, this is a test of text to speech.</textarea>
        <button id="synthesizeBtn" class="primary-btn" aria-label="Synthesize speech">Synthesize</button>
      </div>`;
}

function emitBodyContent(config: ResolvedConfig): string {
  // Audio tasks
  if (config.task === 'text-to-speech') return emitTtsBody();
  if (config.task === 'speech-to-text' || config.task === 'audio-classification') {
    if (config.input === 'mic') return emitAudioMicBody(config);
    return emitAudioFileBody(config);
  }

  if (config.input === 'camera' || config.input === 'screen') return emitRealtimeBody(config);
  if (config.task === 'object-detection' || config.task === 'image-segmentation') return emitFileOverlayBody(config);
  return emitFileClassificationBody(config);
}

function emitIndexHtml(config: ResolvedConfig): string {
  const taskLabel = getTaskLabel(config.task);
  const le = libExt(config);
  const bodyContent = emitBodyContent(config);
  return `<!DOCTYPE html>
<html lang="en" data-theme="${config.theme}">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>${config.modelName} — ${taskLabel}</title>
    <link rel="stylesheet" href="/src/style.css" />
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
    <div class="footer">Generated by webai.js · ${config.modelName} · ${getEngineLabel(config.engine)}</div>
    <script type="module" src="/src/main.${le}"></script>
  </body>
</html>
`;
}

// ---- File + Classification main ----

function emitFileClassificationMain(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';

  return `import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { preprocessImage } from './lib/preprocess.${le}';
import { postprocessResults } from './lib/postprocess.${le}';
import './style.css';

const MODEL_PATH = '${getModelPath(config, '')}';
let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = null;

function updateStatus(text${t ? ': string' : ''}) {
  document.getElementById('status')${t ? '!' : ''}.textContent = text;
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

const dropZone = document.getElementById('dropZone')${t ? '!' : ''};
const fileInput = document.getElementById('fileInput') as ${t ? 'HTMLInputElement' : 'any'};
const preview = document.getElementById('preview')${t ? '!' : ''};
const previewImage = document.getElementById('previewImage') as ${t ? 'HTMLImageElement' : 'any'};
const resultsDiv = document.getElementById('results')${t ? '!' : ''};
const changeBtn = document.getElementById('changeBtn')${t ? '!' : ''};

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', (e${t ? ': KeyboardEvent' : ''}) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});
dropZone.addEventListener('dragover', (e${t ? ': DragEvent' : ''}) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => { dropZone.classList.remove('drag-over'); });
dropZone.addEventListener('drop', (e${t ? ': DragEvent' : ''}) => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  const file = e.dataTransfer${t ? '!' : ''}.files[0];
  if (file) handleFile(file);
});
fileInput.addEventListener('change', () => {
  const file = fileInput.files${t ? '!' : ''}[0];
  if (file) handleFile(file);
});
changeBtn.addEventListener('click', () => {
  preview.hidden = true; dropZone.hidden = false;
  resultsDiv.innerHTML = ''; fileInput.value = '';
});

async function handleFile(file${t ? ': File' : ''}) {
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
  const ctx = canvas.getContext('2d')${t ? '!' : ''};
  ctx.drawImage(previewImage, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  if (!session) { resultsDiv.textContent = 'Model not loaded yet.'; return; }

  updateStatus('${config.modelName} \\u00b7 Processing...');
  const start = performance.now();

  const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
  const output = await runInference(session, inputTensor);
  const results = postprocessResults(output);

  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`);

  renderResults(results);
  URL.revokeObjectURL(url);
}

function renderResults(results${t ? ': { indices: number[]; values: number[] }' : ''}) {
  resultsDiv.innerHTML = '';
  const maxValue = results.values[0] || 1;

  for (let i = 0; i < results.indices.length; i++) {
    const pct = (results.values[i] * 100).toFixed(1);
    if (results.values[i] < 0.01) continue;

    const row = document.createElement('div');
    row.className = 'result-row' + (i === 0 ? ' top-result' : '');
    row.setAttribute('tabindex', '0');
    row.setAttribute('aria-label', \`Class \${results.indices[i]}: \${pct} percent\`);

    row.innerHTML =
      '<span class="result-label">Class ' + results.indices[i] + '</span>' +
      '<div class="result-bar-container"><div class="result-bar" style="width:' +
      ((results.values[i] / maxValue) * 100) + '%"></div></div>' +
      '<span class="result-pct">' + pct + '%</span>';

    resultsDiv.appendChild(row);
  }
}

init();
`;
}

// ---- File + Detection main ----

function emitFileDetectionMain(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const outputShape = config.modelMeta.outputs[0]?.shape ?? [1, 84, 8400];
  const numAttributes = outputShape[1] ?? 84;
  const numAnchors = outputShape[2] ?? 8400;

  return `import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { preprocessImage } from './lib/preprocess.${le}';
import { postprocessDetections } from './lib/postprocess.${le}';
import './style.css';

const MODEL_PATH = '${getModelPath(config, '')}';
const NUM_ATTRIBUTES = ${numAttributes};
const NUM_ANCHORS = ${numAnchors};
const COLORS = [[56,189,248],[249,115,22],[34,197,94],[168,85,247],[251,191,36],[239,68,68],[20,184,166],[236,72,153],[99,102,241],[163,230,53]];
let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = null;

function updateStatus(text${t ? ': string' : ''}) {
  document.getElementById('status')${t ? '!' : ''}.textContent = text;
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

const dropZone = document.getElementById('dropZone')${t ? '!' : ''};
const fileInput = document.getElementById('fileInput') as ${t ? 'HTMLInputElement' : 'any'};
const preview = document.getElementById('preview')${t ? '!' : ''};
const previewImage = document.getElementById('previewImage') as ${t ? 'HTMLImageElement' : 'any'};
const overlay = document.getElementById('overlay') as ${t ? 'HTMLCanvasElement' : 'any'};
const resultsDiv = document.getElementById('results')${t ? '!' : ''};
const changeBtn = document.getElementById('changeBtn')${t ? '!' : ''};

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', (e${t ? ': KeyboardEvent' : ''}) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});
dropZone.addEventListener('dragover', (e${t ? ': DragEvent' : ''}) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => { dropZone.classList.remove('drag-over'); });
dropZone.addEventListener('drop', (e${t ? ': DragEvent' : ''}) => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  const file = e.dataTransfer${t ? '!' : ''}.files[0];
  if (file) handleFile(file);
});
fileInput.addEventListener('change', () => { const file = fileInput.files${t ? '!' : ''}[0]; if (file) handleFile(file); });
changeBtn.addEventListener('click', () => {
  preview.hidden = true; dropZone.hidden = false; resultsDiv.innerHTML = '';
  fileInput.value = '';
  const ctx = overlay.getContext('2d')${t ? '!' : ''};
  ctx.clearRect(0, 0, overlay.width, overlay.height);
});

async function handleFile(file${t ? ': File' : ''}) {
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
  const ctx = canvas.getContext('2d')${t ? '!' : ''};
  ctx.drawImage(previewImage, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  if (!session) { resultsDiv.textContent = 'Model not loaded yet.'; return; }

  updateStatus('${config.modelName} \\u00b7 Processing...');
  const start = performance.now();

  const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
  const output = await runInference(session, inputTensor);
  const boxes = postprocessDetections(output, NUM_ANCHORS, NUM_ATTRIBUTES);

  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`);

  renderDetections(boxes, previewImage.naturalWidth, previewImage.naturalHeight);
  URL.revokeObjectURL(url);
}

function renderDetections(boxes${t ? ': { x: number; y: number; width: number; height: number; classIndex: number; score: number }[]' : ''}, imgW${t ? ': number' : ''}, imgH${t ? ': number' : ''}) {
  const modelSize = ${config.preprocess.imageSize};
  const scaleX = imgW / modelSize;
  const scaleY = imgH / modelSize;

  const ctx = overlay.getContext('2d')${t ? '!' : ''};
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  resultsDiv.innerHTML = '';

  for (const box of boxes) {
    const c = COLORS[box.classIndex % COLORS.length];
    const color = 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')';

    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.strokeRect(box.x * scaleX, box.y * scaleY, box.width * scaleX, box.height * scaleY);

    const label = 'Class ' + box.classIndex + ' (' + (box.score * 100).toFixed(0) + '%)';
    ctx.font = '14px system-ui, sans-serif'; ctx.fillStyle = color;
    const tw = ctx.measureText(label).width;
    ctx.fillRect(box.x * scaleX, box.y * scaleY - 20, tw + 8, 20);
    ctx.fillStyle = '#fff'; ctx.fillText(label, box.x * scaleX + 4, box.y * scaleY - 5);

    const row = document.createElement('div');
    row.className = 'result-row';
    row.setAttribute('tabindex', '0');
    row.setAttribute('aria-label', label);
    row.innerHTML = '<span class="result-label">' + label + '</span>';
    resultsDiv.appendChild(row);
  }

  if (boxes.length === 0) { resultsDiv.textContent = 'No detections found.'; }
}

init();
`;
}

// ---- File + Segmentation main ----

function emitFileSegmentationMain(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const outputShape = config.modelMeta.outputs[0]?.shape ?? [1, 21, 512, 512];
  const numClasses = outputShape[1] ?? 21;
  const maskH = outputShape[2] ?? 512;
  const maskW = outputShape[3] ?? 512;

  return `import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { preprocessImage } from './lib/preprocess.${le}';
import { postprocessSegmentation } from './lib/postprocess.${le}';
import './style.css';

const MODEL_PATH = '${getModelPath(config, '')}';
const NUM_CLASSES = ${numClasses};
const MASK_H = ${maskH};
const MASK_W = ${maskW};
const COLORS = [[56,189,248],[249,115,22],[34,197,94],[168,85,247],[251,191,36],[239,68,68],[20,184,166],[236,72,153],[99,102,241],[163,230,53]];
let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = null;

function updateStatus(text${t ? ': string' : ''}) {
  document.getElementById('status')${t ? '!' : ''}.textContent = text;
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

const dropZone = document.getElementById('dropZone')${t ? '!' : ''};
const fileInput = document.getElementById('fileInput') as ${t ? 'HTMLInputElement' : 'any'};
const preview = document.getElementById('preview')${t ? '!' : ''};
const previewImage = document.getElementById('previewImage') as ${t ? 'HTMLImageElement' : 'any'};
const overlay = document.getElementById('overlay') as ${t ? 'HTMLCanvasElement' : 'any'};
const resultsDiv = document.getElementById('results')${t ? '!' : ''};
const changeBtn = document.getElementById('changeBtn')${t ? '!' : ''};

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', (e${t ? ': KeyboardEvent' : ''}) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});
dropZone.addEventListener('dragover', (e${t ? ': DragEvent' : ''}) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => { dropZone.classList.remove('drag-over'); });
dropZone.addEventListener('drop', (e${t ? ': DragEvent' : ''}) => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  const file = e.dataTransfer${t ? '!' : ''}.files[0];
  if (file) handleFile(file);
});
fileInput.addEventListener('change', () => { const file = fileInput.files${t ? '!' : ''}[0]; if (file) handleFile(file); });
changeBtn.addEventListener('click', () => {
  preview.hidden = true; dropZone.hidden = false; resultsDiv.innerHTML = '';
  const ctx = overlay.getContext('2d')${t ? '!' : ''};
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  fileInput.value = '';
});

async function handleFile(file${t ? ': File' : ''}) {
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
  const ctx = canvas.getContext('2d')${t ? '!' : ''};
  ctx.drawImage(previewImage, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  if (!session) { resultsDiv.textContent = 'Model not loaded yet.'; return; }

  updateStatus('${config.modelName} \\u00b7 Processing...');
  const start = performance.now();

  const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
  const output = await runInference(session, inputTensor);
  const mask = postprocessSegmentation(output, NUM_CLASSES, MASK_H, MASK_W);

  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`);

  renderMask(mask, previewImage.naturalWidth, previewImage.naturalHeight);
  URL.revokeObjectURL(url);
}

function renderMask(mask${t ? ': Uint8Array' : ''}, displayW${t ? ': number' : ''}, displayH${t ? ': number' : ''}) {
  const maskCanvas = document.createElement('canvas');
  maskCanvas.width = MASK_W; maskCanvas.height = MASK_H;
  const maskCtx = maskCanvas.getContext('2d')${t ? '!' : ''};
  const maskImage = maskCtx.createImageData(MASK_W, MASK_H);

  const classesFound = new Set${t ? '<number>' : ''}();
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

  const ctx = overlay.getContext('2d')${t ? '!' : ''};
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

init();
`;
}

// ---- File + Feature Extraction main ----

function emitFileFeatureExtractionMain(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';

  return `import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { preprocessImage } from './lib/preprocess.${le}';
import { postprocessEmbeddings } from './lib/postprocess.${le}';
import './style.css';

const MODEL_PATH = '${getModelPath(config, '')}';
let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = null;

function updateStatus(text${t ? ': string' : ''}) {
  document.getElementById('status')${t ? '!' : ''}.textContent = text;
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

const dropZone = document.getElementById('dropZone')${t ? '!' : ''};
const fileInput = document.getElementById('fileInput') as ${t ? 'HTMLInputElement' : 'any'};
const preview = document.getElementById('preview')${t ? '!' : ''};
const previewImage = document.getElementById('previewImage') as ${t ? 'HTMLImageElement' : 'any'};
const resultsDiv = document.getElementById('results')${t ? '!' : ''};
const changeBtn = document.getElementById('changeBtn')${t ? '!' : ''};

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', (e${t ? ': KeyboardEvent' : ''}) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});
dropZone.addEventListener('dragover', (e${t ? ': DragEvent' : ''}) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => { dropZone.classList.remove('drag-over'); });
dropZone.addEventListener('drop', (e${t ? ': DragEvent' : ''}) => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  const file = e.dataTransfer${t ? '!' : ''}.files[0];
  if (file) handleFile(file);
});
fileInput.addEventListener('change', () => { const file = fileInput.files${t ? '!' : ''}[0]; if (file) handleFile(file); });
changeBtn.addEventListener('click', () => {
  preview.hidden = true; dropZone.hidden = false; resultsDiv.innerHTML = '';
  fileInput.value = '';
});

async function handleFile(file${t ? ': File' : ''}) {
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
  const ctx = canvas.getContext('2d')${t ? '!' : ''};
  ctx.drawImage(previewImage, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  if (!session) { resultsDiv.textContent = 'Model not loaded yet.'; return; }

  updateStatus('${config.modelName} \\u00b7 Processing...');
  const start = performance.now();

  const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
  const output = await runInference(session, inputTensor);
  const embedding = postprocessEmbeddings(output);

  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`);

  renderEmbedding(embedding);
  URL.revokeObjectURL(url);
}

function renderEmbedding(embedding${t ? ': Float32Array' : ''}) {
  let norm = 0;
  for (let i = 0; i < embedding.length; i++) { norm += embedding[i] * embedding[i]; }
  norm = Math.sqrt(norm);

  const first5 = Array.from(embedding.slice(0, 5)).map((v${t ? ': number' : ''}) => v.toFixed(4)).join(', ');

  resultsDiv.innerHTML =
    '<div class="embedding-info">' +
    '<p><strong>Dimensions:</strong> ' + embedding.length + '</p>' +
    '<p><strong>L2 Norm:</strong> ' + norm.toFixed(4) + '</p>' +
    '<p><strong>First 5 values:</strong> [' + first5 + ', ...]</p>' +
    '</div>';
}

init();
`;
}

// ---- Camera / Screen realtime main ----

function emitRealtimeMain(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const isScreen = config.input === 'screen';
  const startFn = isScreen ? 'startScreenCapture' : 'startCamera';
  const label = isScreen ? 'Screen Capture' : 'Camera';

  let processOutput: string;
  let renderCall: string;
  let extraConst = '';
  let extraFn = '';

  switch (config.task) {
    case 'object-detection': {
      const outputShape = config.modelMeta.outputs[0]?.shape ?? [1, 84, 8400];
      const numAttributes = outputShape[1] ?? 84;
      const numAnchors = outputShape[2] ?? 8400;
      extraConst = `const NUM_ATTRIBUTES = ${numAttributes};\nconst NUM_ANCHORS = ${numAnchors};\nconst COLORS = [[56,189,248],[249,115,22],[34,197,94],[168,85,247],[251,191,36],[239,68,68],[20,184,166],[236,72,153],[99,102,241],[163,230,53]];`;
      processOutput = `const boxes = postprocessDetections(output, NUM_ANCHORS, NUM_ATTRIBUTES);`;
      renderCall = `renderDetections(overlayCtx, boxes, video.videoWidth, video.videoHeight);`;
      extraFn = `
function renderDetections(ctx${t ? ': CanvasRenderingContext2D' : ''}, boxes${t ? ': { x: number; y: number; width: number; height: number; classIndex: number; score: number }[]' : ''}, videoW${t ? ': number' : ''}, videoH${t ? ': number' : ''}) {
  const modelSize = ${config.preprocess.imageSize};
  const scaleX = videoW / modelSize;
  const scaleY = videoH / modelSize;
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  for (const box of boxes) {
    const c = COLORS[box.classIndex % COLORS.length];
    const color = 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')';
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.strokeRect(box.x * scaleX, box.y * scaleY, box.width * scaleX, box.height * scaleY);
    const lbl = 'Class ' + box.classIndex + ' (' + (box.score * 100).toFixed(0) + '%)';
    ctx.font = '14px system-ui, sans-serif'; ctx.fillStyle = color;
    const tw = ctx.measureText(lbl).width;
    ctx.fillRect(box.x * scaleX, box.y * scaleY - 20, tw + 8, 20);
    ctx.fillStyle = '#fff'; ctx.fillText(lbl, box.x * scaleX + 4, box.y * scaleY - 5);
  }
}`;
      break;
    }
    case 'image-segmentation': {
      const outputShape = config.modelMeta.outputs[0]?.shape ?? [1, 21, 512, 512];
      const numClasses = outputShape[1] ?? 21;
      const maskH = outputShape[2] ?? 512;
      const maskW = outputShape[3] ?? 512;
      extraConst = `const NUM_CLASSES = ${numClasses};\nconst MASK_H = ${maskH};\nconst MASK_W = ${maskW};\nconst COLORS = [[56,189,248],[249,115,22],[34,197,94],[168,85,247],[251,191,36],[239,68,68],[20,184,166],[236,72,153],[99,102,241],[163,230,53]];`;
      processOutput = `const mask = postprocessSegmentation(output, NUM_CLASSES, MASK_H, MASK_W);`;
      renderCall = `renderMask(overlayCtx, mask, video.videoWidth, video.videoHeight);`;
      extraFn = `
function renderMask(ctx${t ? ': CanvasRenderingContext2D' : ''}, mask${t ? ': Uint8Array' : ''}, displayW${t ? ': number' : ''}, displayH${t ? ': number' : ''}) {
  const maskCanvas = document.createElement('canvas');
  maskCanvas.width = MASK_W; maskCanvas.height = MASK_H;
  const maskCtx = maskCanvas.getContext('2d')${t ? '!' : ''};
  const maskImage = maskCtx.createImageData(MASK_W, MASK_H);
  for (let i = 0; i < mask.length; i++) {
    const c = COLORS[mask[i] % COLORS.length];
    maskImage.data[i * 4] = c[0]; maskImage.data[i * 4 + 1] = c[1];
    maskImage.data[i * 4 + 2] = c[2]; maskImage.data[i * 4 + 3] = 128;
  }
  maskCtx.putImageData(maskImage, 0, 0);
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.drawImage(maskCanvas, 0, 0, displayW, displayH);
}`;
      break;
    }
    default: {
      processOutput = `const results = postprocessResults(output);`;
      renderCall = `
    const lbl = 'Class ' + results.indices[0] + ' (' + (results.values[0] * 100).toFixed(1) + '%)';
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
    overlayCtx.font = 'bold 24px system-ui, sans-serif';
    overlayCtx.fillStyle = 'rgba(59, 130, 246, 0.85)';
    const tw = overlayCtx.measureText(lbl).width;
    overlayCtx.fillRect(8, 8, tw + 16, 36);
    overlayCtx.fillStyle = '#fff';
    overlayCtx.fillText(lbl, 16, 34);`;
      break;
    }
  }

  // Determine postprocess import
  let postImport: string;
  if (config.task === 'object-detection') postImport = `import { postprocessDetections } from './lib/postprocess.${le}';`;
  else if (config.task === 'image-segmentation') postImport = `import { postprocessSegmentation } from './lib/postprocess.${le}';`;
  else postImport = `import { postprocessResults } from './lib/postprocess.${le}';`;

  return `import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { preprocessImage } from './lib/preprocess.${le}';
${postImport}
import { ${startFn}, captureFrame, createInferenceLoop } from './lib/input.${le}';
import './style.css';

const MODEL_PATH = '${getModelPath(config, '')}';
${extraConst}
let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = null;
let currentStream${t ? ': MediaStream | null' : ''} = null;

function updateStatus(text${t ? ': string' : ''}) {
  document.getElementById('status')${t ? '!' : ''}.textContent = text;
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
${extraFn}

const video = document.getElementById('video') as ${t ? 'HTMLVideoElement' : 'any'};
const overlay = document.getElementById('overlay') as ${t ? 'HTMLCanvasElement' : 'any'};
const overlayCtx = overlay.getContext('2d')${t ? '!' : ''};
const startBtn = document.getElementById('startBtn')${t ? '!' : ''};
const pauseBtn = document.getElementById('pauseBtn')${t ? '!' : ''};
const permissionPrompt = document.getElementById('permissionPrompt')${t ? '!' : ''};
const videoContainer = document.getElementById('videoContainer')${t ? '!' : ''};

let loop${t ? ': ReturnType<typeof createInferenceLoop> | null' : ''} = null;

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
      async onFrame(imageData${t ? ': ImageData' : ''}) {
        const start = performance.now();
        const inputTensor = preprocessImage(imageData.data, imageData.width, imageData.height);
        const output = await runInference(session${t ? '!' : ''}, inputTensor);
        ${processOutput}
        const elapsed = performance.now() - start;
        ${renderCall}
        return { result: null, elapsed };
      },
      onStatus(elapsed${t ? ': number' : ''}) {
        updateStatus('${config.modelName} \\u00b7 ' + elapsed.toFixed(1) + 'ms \\u00b7 ' + getBackendLabel(session${t ? '!' : ''}));
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
        async onFrame(imageData${t ? ': ImageData' : ''}) {
          const start = performance.now();
          const inputTensor = preprocessImage(imageData.data, imageData.width, imageData.height);
          const output = await runInference(session${t ? '!' : ''}, inputTensor);
          ${processOutput}
          const elapsed = performance.now() - start;
          ${renderCall}
          return { result: null, elapsed };
        },
        onStatus(elapsed${t ? ': number' : ''}) {
          updateStatus('${config.modelName} \\u00b7 ' + elapsed.toFixed(1) + 'ms \\u00b7 ' + getBackendLabel(session${t ? '!' : ''}));
        },
      });
      loop.start();
      pauseBtn.textContent = '\\u23f8 Pause';
    }, { once: true });
  }
});

init();
`;
}

// ---- File + Audio Classification main ----

function emitFileAudioClassificationMain(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';

  return `import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { melSpectrogram, mfcc } from './lib/preprocess.${le}';
import { postprocessResults } from './lib/postprocess.${le}';
import './style.css';

const MODEL_PATH = '${getModelPath(config, '')}';
let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = null;

function updateStatus(text${t ? ': string' : ''}) {
  document.getElementById('status')${t ? '!' : ''}.textContent = text;
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

const fileInput = document.getElementById('fileInput') as ${t ? 'HTMLInputElement' : 'any'};
const resultsDiv = document.getElementById('results')${t ? '!' : ''};

fileInput.addEventListener('change', async () => {
  const file = fileInput.files${t ? '!' : ''}[0];
  if (!file || !session) return;

  updateStatus('${config.modelName} \\u00b7 Decoding audio...');

  const arrayBuffer = await file.arrayBuffer();
  const audioCtx = new OfflineAudioContext(1, 1, 16000);
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

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
  updateStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`);

  renderResults(results);
});

function renderResults(results${t ? ': { indices: number[]; values: number[] }' : ''}) {
  resultsDiv.innerHTML = '';
  const maxValue = results.values[0] || 1;
  for (let i = 0; i < results.indices.length; i++) {
    const pct = (results.values[i] * 100).toFixed(1);
    if (results.values[i] < 0.01) continue;
    const row = document.createElement('div');
    row.className = 'result-row' + (i === 0 ? ' top-result' : '');
    row.setAttribute('tabindex', '0');
    row.setAttribute('aria-label', \`Class \${results.indices[i]}: \${pct} percent\`);
    row.innerHTML =
      '<span class="result-label">Class ' + results.indices[i] + '</span>' +
      '<div class="result-bar-container"><div class="result-bar" style="width:' +
      ((results.values[i] / maxValue) * 100) + '%"></div></div>' +
      '<span class="result-pct">' + pct + '%</span>';
    resultsDiv.appendChild(row);
  }
}

init();
`;
}

// ---- File + Speech-to-Text main ----

function emitFileSpeechToTextMain(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';

  return `import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { melSpectrogram } from './lib/preprocess.${le}';
import { postprocessTranscript } from './lib/postprocess.${le}';
import './style.css';

const MODEL_PATH = '${getModelPath(config, '')}';
const VOCAB = [' ', ...'abcdefghijklmnopqrstuvwxyz'.split(''), "'"];
let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = null;

function updateStatus(text${t ? ': string' : ''}) {
  document.getElementById('status')${t ? '!' : ''}.textContent = text;
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

const fileInput = document.getElementById('fileInput') as ${t ? 'HTMLInputElement' : 'any'};
const transcript = document.getElementById('transcript')${t ? '!' : ''};

fileInput.addEventListener('change', async () => {
  const file = fileInput.files${t ? '!' : ''}[0];
  if (!file || !session) return;

  updateStatus('${config.modelName} \\u00b7 Decoding audio...');

  const arrayBuffer = await file.arrayBuffer();
  const audioCtx = new OfflineAudioContext(1, 1, 16000);
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

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
  updateStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`);

  transcript.textContent = text || '(no speech detected)';
});

init();
`;
}

// ---- Mic + Speech-to-Text main ----

function emitMicSpeechToTextMain(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';

  return `import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { melSpectrogram } from './lib/preprocess.${le}';
import { postprocessTranscript } from './lib/postprocess.${le}';
import { startAudioCapture, stopStream, createAudioInferenceLoop } from './lib/input.${le}';
import './style.css';

const MODEL_PATH = '${getModelPath(config, '')}';
const VOCAB = [' ', ...'abcdefghijklmnopqrstuvwxyz'.split(''), "'"];
let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = null;
let capture${t ? ': any' : ''} = null;
let loop${t ? ': any' : ''} = null;

function updateStatus(text${t ? ': string' : ''}) {
  document.getElementById('status')${t ? '!' : ''}.textContent = text;
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

async function processAudio(samples${t ? ': Float32Array' : ''}) {
  const mel = melSpectrogram(samples, 16000, 512, 160, 80);
  const output = await runInference(session${t ? '!' : ''}, mel.data);
  const vocabSize = VOCAB.length + 1;
  const numTimesteps = Math.floor(output.length / vocabSize);
  return postprocessTranscript(output, numTimesteps, vocabSize, VOCAB);
}

const startBtn = document.getElementById('startBtn')${t ? '!' : ''};
const stopBtn = document.getElementById('stopBtn') as ${t ? 'HTMLButtonElement' : 'any'};
const transcript = document.getElementById('transcript')${t ? '!' : ''};

startBtn.addEventListener('click', async () => {
  if (!session) { updateStatus('Model not loaded yet.'); return; }
  try {
    capture = await startAudioCapture(16000);
    (startBtn as ${t ? 'HTMLButtonElement' : 'any'}).disabled = true;
    stopBtn.disabled = false;
    updateStatus('${config.modelName} \\u00b7 Listening...');
    loop = createAudioInferenceLoop({
      getSamples: capture.getSamples,
      onResult(text${t ? ': string' : ''}) { transcript.textContent = text || '(listening...)'; },
      intervalMs: 2000,
    });
    loop.start();
  } catch (e) { updateStatus('Microphone access denied'); console.error('Mic error:', e); }
});

stopBtn.addEventListener('click', () => {
  if (loop) { loop.stop(); loop = null; }
  if (capture) { stopStream(capture.stream); capture.audioContext.close(); capture = null; }
  (startBtn as ${t ? 'HTMLButtonElement' : 'any'}).disabled = false;
  stopBtn.disabled = true;
  updateStatus('${config.modelName} \\u00b7 Stopped');
});

init();
`;
}

// ---- Mic + Audio Classification main ----

function emitMicAudioClassificationMain(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';

  return `import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { melSpectrogram, mfcc } from './lib/preprocess.${le}';
import { postprocessResults } from './lib/postprocess.${le}';
import { startAudioCapture, stopStream, createAudioInferenceLoop } from './lib/input.${le}';
import './style.css';

const MODEL_PATH = '${getModelPath(config, '')}';
let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = null;
let capture${t ? ': any' : ''} = null;
let loop${t ? ': any' : ''} = null;

function updateStatus(text${t ? ': string' : ''}) {
  document.getElementById('status')${t ? '!' : ''}.textContent = text;
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

async function processAudio(samples${t ? ': Float32Array' : ''}) {
  const mel = melSpectrogram(samples, 16000, 512, 160, 40);
  const features = mfcc(mel.data, mel.numFrames, mel.numMelBands, 13);
  const output = await runInference(session${t ? '!' : ''}, features);
  return postprocessResults(output);
}

const startBtn = document.getElementById('startBtn')${t ? '!' : ''};
const stopBtn = document.getElementById('stopBtn') as ${t ? 'HTMLButtonElement' : 'any'};
const resultsDiv = document.getElementById('results')${t ? '!' : ''};

startBtn.addEventListener('click', async () => {
  if (!session) { updateStatus('Model not loaded yet.'); return; }
  try {
    capture = await startAudioCapture(16000);
    (startBtn as ${t ? 'HTMLButtonElement' : 'any'}).disabled = true;
    stopBtn.disabled = false;
    updateStatus('${config.modelName} \\u00b7 Listening...');
    loop = createAudioInferenceLoop({
      getSamples: capture.getSamples,
      onResult(results${t ? ': { indices: number[]; values: number[] }' : ''}) { renderResults(results); },
      intervalMs: 2000,
    });
    loop.start();
  } catch (e) { updateStatus('Microphone access denied'); console.error('Mic error:', e); }
});

stopBtn.addEventListener('click', () => {
  if (loop) { loop.stop(); loop = null; }
  if (capture) { stopStream(capture.stream); capture.audioContext.close(); capture = null; }
  (startBtn as ${t ? 'HTMLButtonElement' : 'any'}).disabled = false;
  stopBtn.disabled = true;
  updateStatus('${config.modelName} \\u00b7 Stopped');
});

function renderResults(results${t ? ': { indices: number[]; values: number[] }' : ''}) {
  resultsDiv.innerHTML = '';
  const maxValue = results.values[0] || 1;
  for (let i = 0; i < results.indices.length; i++) {
    const pct = (results.values[i] * 100).toFixed(1);
    if (results.values[i] < 0.01) continue;
    const row = document.createElement('div');
    row.className = 'result-row' + (i === 0 ? ' top-result' : '');
    row.setAttribute('tabindex', '0');
    row.setAttribute('aria-label', \`Class \${results.indices[i]}: \${pct} percent\`);
    row.innerHTML =
      '<span class="result-label">Class ' + results.indices[i] + '</span>' +
      '<div class="result-bar-container"><div class="result-bar" style="width:' +
      ((results.values[i] / maxValue) * 100) + '%"></div></div>' +
      '<span class="result-pct">' + pct + '%</span>';
    resultsDiv.appendChild(row);
  }
}

init();
`;
}

// ---- Text-to-Speech main ----

function emitTextToSpeechMain(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';

  return `import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { postprocessAudio, playAudio } from './lib/postprocess.${le}';
import './style.css';

const MODEL_PATH = '${getModelPath(config, '')}';
let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = null;

function updateStatus(text${t ? ': string' : ''}) {
  document.getElementById('status')${t ? '!' : ''}.textContent = text;
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

const textInput = document.getElementById('textInput') as ${t ? 'HTMLTextAreaElement' : 'any'};
const synthesizeBtn = document.getElementById('synthesizeBtn')${t ? '!' : ''};

synthesizeBtn.addEventListener('click', async () => {
  const text = textInput.value.trim();
  if (!text || !session) return;

  updateStatus('${config.modelName} \\u00b7 Synthesizing...');
  const start = performance.now();

  const tokens = new Float32Array(text.length);
  for (let i = 0; i < text.length; i++) { tokens[i] = text.charCodeAt(i); }

  const output = await runInference(session, tokens);
  const samples = postprocessAudio(output);
  await playAudio(samples);

  const elapsed = (performance.now() - start).toFixed(1);
  updateStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`);
});

init();
`;
}

// ---- Main dispatcher ----

function emitMain(config: ResolvedConfig): string {
  // Audio tasks
  if (config.task === 'text-to-speech') return emitTextToSpeechMain(config);
  if (config.task === 'audio-classification') {
    if (config.input === 'mic') return emitMicAudioClassificationMain(config);
    return emitFileAudioClassificationMain(config);
  }
  if (config.task === 'speech-to-text') {
    if (config.input === 'mic') return emitMicSpeechToTextMain(config);
    return emitFileSpeechToTextMain(config);
  }

  // Visual tasks
  if (config.input === 'camera' || config.input === 'screen') return emitRealtimeMain(config);
  if (config.task === 'object-detection') return emitFileDetectionMain(config);
  if (config.task === 'image-segmentation') return emitFileSegmentationMain(config);
  if (config.task === 'feature-extraction') return emitFileFeatureExtractionMain(config);
  return emitFileClassificationMain(config);
}

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

.primary-btn:hover { opacity: 0.9; }

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

function needsExtendedCSS(config: ResolvedConfig): boolean {
  return config.input === 'camera' || config.input === 'screen' ||
    config.task === 'object-detection' || config.task === 'image-segmentation' ||
    config.task === 'feature-extraction';
}

function emitStyleCss(config: ResolvedConfig): string {
  const extra = needsExtendedCSS(config) ? emitExtendedCSS() : '';
  return `${emitDesignSystemCSS(config)}\n\n${emitAppCSS()}${extra}`;
}

/** Wrap a CodeBlock's code with exports for use as a lib module */
function toLibModule(block: CodeBlock | undefined): string {
  if (!block) return '';
  return addExports(block.code, block.exports);
}

/**
 * Emit Vanilla-Vite framework files.
 */
export function emitVanillaVite(config: ResolvedConfig, blocks: CodeBlock[]): GeneratedFile[] {
  const le = libExt(config);

  const inputBlock = findBlock(blocks, 'input');
  const preprocessBlock = findBlock(blocks, 'preprocess');
  const inferenceBlock = findBlock(blocks, 'inference');
  const postprocessBlock = findBlock(blocks, 'postprocess');
  const opfsBlock = findBlock(blocks, 'opfs-cache');

  const filePaths: string[] = [
    'package.json',
    'vite.config.js',
    'index.html',
    `src/main.${le}`,
    'src/style.css',
  ];

  if (inputBlock?.code) {
    filePaths.push(`src/lib/input.${le}`);
  }

  filePaths.push(
    `src/lib/preprocess.${le}`,
    `src/lib/inference.${le}`,
    `src/lib/postprocess.${le}`,
    'README.md',
  );

  // Prepend OPFS cache utilities to inference module when offline mode is enabled
  const inferenceContent = opfsBlock?.code
    ? `${opfsBlock.code}\n\n${toLibModule(inferenceBlock)}`
    : toLibModule(inferenceBlock);

  const files: GeneratedFile[] = [
    { path: 'package.json', content: emitPackageJson(config, blocks) },
    { path: 'vite.config.js', content: emitViteConfig() },
    { path: 'index.html', content: emitIndexHtml(config) },
    { path: `src/main.${le}`, content: emitMain(config) },
    { path: 'src/style.css', content: emitStyleCss(config) },
  ];

  if (inputBlock?.code) {
    files.push({ path: `src/lib/input.${le}`, content: toLibModule(inputBlock) });
  }

  files.push(
    { path: `src/lib/preprocess.${le}`, content: toLibModule(preprocessBlock) },
    { path: `src/lib/inference.${le}`, content: inferenceContent },
    { path: `src/lib/postprocess.${le}`, content: toLibModule(postprocessBlock) },
    { path: 'README.md', content: emitReadme(config, filePaths) },
  );

  // Include auxiliary files from Layer 1 blocks (e.g. AudioWorklet processor)
  files.push(...collectAuxiliaryFiles(blocks));

  return files;
}
