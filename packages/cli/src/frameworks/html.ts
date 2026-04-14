/**
 * HTML framework emitter (Layer 2).
 *
 * Produces a single index.html with inline CSS + JS.
 * Uses CDN import for onnxruntime-web. No build step required.
 * Just open in browser with a local server (for module/CORS).
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
} from './shared.js';

const ORT_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/esm/ort.min.mjs';

/** Generate the file input + classification application JS */
function emitAppScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  const preprocessBlock = findBlock(blocks, 'preprocess');
  const inferenceBlock = findBlock(blocks, 'inference');
  const postprocessBlock = findBlock(blocks, 'postprocess');

  const preprocessCode = preprocessBlock?.code ?? '';
  const inferenceCode = inferenceBlock ? stripImports(inferenceBlock.code) : '';
  const postprocessCode = postprocessBlock?.code ?? '';

  return `import * as ort from '${ORT_CDN}';

// --- Preprocessing ---
${preprocessCode}

// --- Inference ---
${inferenceCode}

// --- Postprocessing ---
${postprocessCode}

// --- Application ---
const MODEL_PATH = './${config.modelName}.onnx';
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

/**
 * Emit HTML framework files.
 */
export function emitHtml(config: ResolvedConfig, blocks: CodeBlock[]): GeneratedFile[] {
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const theme = config.theme;

  const designCSS = emitDesignSystemCSS(config);
  const appCSS = emitAppCSS();
  const appScript = emitAppScript(config, blocks);

  const html = `<!DOCTYPE html>
<html lang="en" data-theme="${theme}">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${config.modelName} — ${taskLabel}</title>
  <style>
${designCSS}
${appCSS}
  </style>
</head>
<body>
  <a href="#results" class="skip-link">Skip to results</a>

  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>

    <div class="container">
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
    </div>
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

  return files;
}
