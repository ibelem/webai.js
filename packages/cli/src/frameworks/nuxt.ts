/**
 * Nuxt 3 framework emitter (Layer 2).
 *
 * Produces a Nuxt 3 project:
 *   package.json, nuxt.config.ts,
 *   assets/app.css, app.vue,
 *   pages/index.vue,
 *   lib/input.{ts|js} (if non-file input),
 *   lib/preprocess.{ts|js}, lib/inference.{ts|js},
 *   lib/postprocess.{ts|js}, README.md
 *
 * Uses Vue 3 SFC with <script setup>. All ML inference runs client-side (ssr: false).
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
    nuxt: '^3.15.0',
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
        dev: 'nuxt dev',
        build: 'nuxt build',
        preview: 'nuxt preview',
      },
      dependencies: deps,
      devDependencies: devDeps,
    },
    null,
    2,
  );
}

function emitNuxtConfig(): string {
  return `export default defineNuxtConfig({
  ssr: false,
  compatibilityDate: '2025-01-01',
  css: ['~/assets/app.css'],
});
`;
}

function emitAppVue(): string {
  return `<template>
  <NuxtPage />
</template>
`;
}

function emitExtendedCSS(): string {
  return `
/* Canvas overlay */
.preview-wrapper {
  position: relative;
  display: inline-block;
}

.preview-wrapper .overlay-canvas,
.video-wrapper canvas {
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
    config.task === 'feature-extraction' || config.task === 'depth-estimation' ||
    config.task === 'image-to-text' || config.task === 'visual-question-answering' ||
    config.task === 'document-question-answering' || config.task === 'image-text-to-text';
}

function emitAssetsCss(config: ResolvedConfig): string {
  const extra = needsExtendedCSS(config) ? emitExtendedCSS() : '';
  return `${emitDesignSystemCSS(config)}\n\n${emitAppCSS()}${extra}`;
}

// ---- Vue page emitters by task/input ----

function emitFileClassificationPage(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script setup${tsLang}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from '~/lib/inference.${le}';
import { preprocessImage } from '~/lib/preprocess.${le}';
import { postprocessResults } from '~/lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';

const session = shallowRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
const results = shallowRef${t ? '<{ indices: number[]; values: number[] } | null>' : ''}(null);
const imageUrl = ref${t ? '<string | null>' : ''}(null);
const status = ref('Loading model...');
const dragOver = ref(false);
const fileInput = ref${t ? '<HTMLInputElement | null>' : ''}(null);

onMounted(() => {
  createSession(MODEL_PATH).then((s) => {
    session.value = s;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => { status.value = 'Failed to load model'; console.error('Model load error:', e); });
});

async function processImage(file${t ? ': File' : ''}) {
  if (!file.type.startsWith('image/')) { status.value = 'Unsupported file type.'; return; }
  const url = URL.createObjectURL(file);
  imageUrl.value = url; results.value = null;
  const img = new Image(); img.src = url;
  await new Promise((resolve) => { img.onload = resolve; });
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d')${t ? '!' : ''}; ctx.drawImage(img, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  if (!session.value) { status.value = 'Model not loaded yet.'; return; }
  status.value = '${config.modelName} \\u00b7 Processing...';
  const start = performance.now();
  const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
  const output = await runInference(session.value, inputTensor);
  const r = postprocessResults(output);
  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  results.value = r;
  URL.revokeObjectURL(url);
}

function handleDrop(e${t ? ': DragEvent' : ''}) {
  e.preventDefault(); dragOver.value = false;
  const f = e.dataTransfer?.files[0]; if (f) processImage(f);
}

function handleFileChange(e${t ? ': Event' : ''}) {
  const f = (e.target as HTMLInputElement).files?.[0]; if (f) processImage(f);
}

function reset() { imageUrl.value = null; results.value = null; if (fileInput.value) fileInput.value.value = ''; }
</script>

<template>
  <a href="#results" class="skip-link">Skip to results</a>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="container">
      <div>
        <div v-if="!imageUrl" :class="['drop-zone', { 'drag-over': dragOver }]" role="button" tabindex="0"
          aria-label="Drop an image here or click to browse for ${taskLabel.toLowerCase()}"
          @click="fileInput?.click()"
          @keydown.enter.space.prevent="fileInput?.click()"
          @dragover.prevent="dragOver = true"
          @dragleave="dragOver = false"
          @drop="handleDrop">
          <p>Drop an image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input ref="fileInput" type="file" accept="image/*" style="position:absolute;left:-9999px;width:1px;height:1px;opacity:0;" aria-hidden="true" tabindex="-1" @change="handleFileChange" />
        </div>
        <div v-else class="preview">
          <img :src="imageUrl" alt="Selected image for classification" />
          <button class="change-btn" @click="reset">Choose another image</button>
        </div>
      </div>
      <div id="results" class="results" role="status" aria-live="polite" aria-atomic="true">
        <template v-if="results">
          <template v-for="(idx, i) in results.indices" :key="idx">
            <div v-if="results.values[i] >= 0.01" :class="['result-row', { 'top-result': i === 0 }]" tabindex="0"
              :aria-label="\`Class \${idx}: \${(results.values[i] * 100).toFixed(1)} percent\`">
              <span class="result-label">Class {{ idx }}</span>
              <div class="result-bar-container">
                <div class="result-bar" :style="{ width: (results.values[i] / (results.values[0] || 1)) * 100 + '%' }"></div>
              </div>
              <span class="result-pct">{{ (results.values[i] * 100).toFixed(1) }}%</span>
            </div>
          </template>
        </template>
      </div>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
</template>
`;
}

function emitFileDetectionPage(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';
  const outputShape = config.modelMeta.outputs[0]?.shape ?? [1, 84, 8400];
  const numAttributes = outputShape[1] ?? 84;
  const numAnchors = outputShape[2] ?? 8400;

  const boxInterface = t ? '\ninterface Box { x: number; y: number; width: number; height: number; classIndex: number; score: number; }\n' : '';

  return `<script setup${tsLang}>
import { ref, shallowRef, onMounted, watch } from 'vue';
import { createSession, runInference, getBackendLabel } from '~/lib/inference.${le}';
import { preprocessImage } from '~/lib/preprocess.${le}';
import { postprocessDetections } from '~/lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
const NUM_ATTRIBUTES = ${numAttributes};
const NUM_ANCHORS = ${numAnchors};
const COLORS = [[56,189,248],[249,115,22],[34,197,94],[168,85,247],[251,191,36],[239,68,68],[20,184,166],[236,72,153],[99,102,241],[163,230,53]];
${boxInterface}
const session = shallowRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
const boxes = shallowRef${t ? '<Box[] | null>' : ''}(null);
const imageUrl = ref${t ? '<string | null>' : ''}(null);
const status = ref('Loading model...');
const dragOver = ref(false);
const imgSize = ref({ w: 0, h: 0 });
const fileInput = ref${t ? '<HTMLInputElement | null>' : ''}(null);
const overlay = ref${t ? '<HTMLCanvasElement | null>' : ''}(null);

onMounted(() => {
  createSession(MODEL_PATH).then((s) => {
    session.value = s;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => { status.value = 'Failed to load model'; console.error('Model load error:', e); });
});

watch([boxes, imgSize], () => {
  if (!boxes.value || !overlay.value) return;
  const canvas = overlay.value;
  canvas.width = imgSize.value.w; canvas.height = imgSize.value.h;
  const ctx = canvas.getContext('2d')${t ? '!' : ''};
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const scale = imgSize.value.w / ${config.preprocess.imageSize};
  for (const box of boxes.value) {
    const c = COLORS[box.classIndex % COLORS.length];
    const color = \`rgb(\${c[0]},\${c[1]},\${c[2]})\`;
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.strokeRect(box.x * scale, box.y * scale, box.width * scale, box.height * scale);
    const label = \`Class \${box.classIndex} (\${(box.score * 100).toFixed(0)}%)\`;
    ctx.font = '14px system-ui, sans-serif'; ctx.fillStyle = color;
    const tw = ctx.measureText(label).width;
    ctx.fillRect(box.x * scale, box.y * scale - 20, tw + 8, 20);
    ctx.fillStyle = '#fff'; ctx.fillText(label, box.x * scale + 4, box.y * scale - 5);
  }
});

async function processImage(file${t ? ': File' : ''}) {
  if (!file.type.startsWith('image/')) { status.value = 'Unsupported file type.'; return; }
  const url = URL.createObjectURL(file);
  imageUrl.value = url; boxes.value = null;
  const img = new Image(); img.src = url;
  await new Promise((resolve) => { img.onload = resolve; });
  imgSize.value = { w: img.naturalWidth, h: img.naturalHeight };
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d')${t ? '!' : ''}; ctx.drawImage(img, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  if (!session.value) { status.value = 'Model not loaded yet.'; return; }
  status.value = '${config.modelName} \\u00b7 Processing...';
  const start = performance.now();
  const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
  const output = await runInference(session.value, inputTensor);
  const detected = postprocessDetections(output, NUM_ANCHORS, NUM_ATTRIBUTES);
  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  boxes.value = detected;
  URL.revokeObjectURL(url);
}

function handleDrop(e${t ? ': DragEvent' : ''}) { e.preventDefault(); dragOver.value = false; const f = e.dataTransfer?.files[0]; if (f) processImage(f); }
function handleFileChange(e${t ? ': Event' : ''}) { const f = (e.target as HTMLInputElement).files?.[0]; if (f) processImage(f); }
function reset() { imageUrl.value = null; boxes.value = null; if (fileInput.value) fileInput.value.value = ''; }
</script>

<template>
  <a href="#results" class="skip-link">Skip to results</a>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="container">
      <div>
        <div v-if="!imageUrl" :class="['drop-zone', { 'drag-over': dragOver }]" role="button" tabindex="0"
          aria-label="Drop an image here or click to browse for ${taskLabel.toLowerCase()}"
          @click="fileInput?.click()" @keydown.enter.space.prevent="fileInput?.click()"
          @dragover.prevent="dragOver = true" @dragleave="dragOver = false" @drop="handleDrop">
          <p>Drop an image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input ref="fileInput" type="file" accept="image/*" style="position:absolute;left:-9999px;width:1px;height:1px;opacity:0;" aria-hidden="true" tabindex="-1" @change="handleFileChange" />
        </div>
        <div v-else class="preview">
          <div class="preview-wrapper">
            <img :src="imageUrl" alt="Selected image for detection" />
            <canvas ref="overlay" class="overlay-canvas"></canvas>
          </div>
          <button class="change-btn" @click="reset">Choose another image</button>
        </div>
      </div>
      <div id="results" class="results" role="status" aria-live="polite" aria-atomic="true">
        <p v-if="boxes && boxes.length === 0">No detections found.</p>
        <template v-if="boxes">
          <div v-for="(box, i) in boxes" :key="i" class="result-row" tabindex="0"
            :aria-label="\`Class \${box.classIndex}: \${(box.score * 100).toFixed(0)} percent\`">
            <span class="result-label">Class {{ box.classIndex }} ({{ (box.score * 100).toFixed(0) }}%)</span>
          </div>
        </template>
      </div>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
</template>
`;
}

function emitFileSegmentationPage(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';
  const outputShape = config.modelMeta.outputs[0]?.shape ?? [1, 21, 512, 512];
  const numClasses = outputShape[1] ?? 21;
  const maskH = outputShape[2] ?? 512;
  const maskW = outputShape[3] ?? 512;

  return `<script setup${tsLang}>
import { ref, shallowRef, onMounted, watch } from 'vue';
import { createSession, runInference, getBackendLabel } from '~/lib/inference.${le}';
import { preprocessImage } from '~/lib/preprocess.${le}';
import { postprocessSegmentation } from '~/lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
const NUM_CLASSES = ${numClasses};
const MASK_H = ${maskH};
const MASK_W = ${maskW};
const COLORS = [[56,189,248],[249,115,22],[34,197,94],[168,85,247],[251,191,36],[239,68,68],[20,184,166],[236,72,153],[99,102,241],[163,230,53]];

const session = shallowRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
const mask = shallowRef${t ? '<Uint8Array | null>' : ''}(null);
const imageUrl = ref${t ? '<string | null>' : ''}(null);
const status = ref('Loading model...');
const dragOver = ref(false);
const imgSize = ref({ w: 0, h: 0 });
const fileInput = ref${t ? '<HTMLInputElement | null>' : ''}(null);
const overlay = ref${t ? '<HTMLCanvasElement | null>' : ''}(null);

onMounted(() => {
  createSession(MODEL_PATH).then((s) => {
    session.value = s;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => { status.value = 'Failed to load model'; console.error('Model load error:', e); });
});

watch([mask, imgSize], () => {
  if (!mask.value || !overlay.value) return;
  const canvas = overlay.value;
  canvas.width = imgSize.value.w; canvas.height = imgSize.value.h;
  const maskCanvas = document.createElement('canvas');
  maskCanvas.width = MASK_W; maskCanvas.height = MASK_H;
  const maskCtx = maskCanvas.getContext('2d')${t ? '!' : ''};
  const maskImage = maskCtx.createImageData(MASK_W, MASK_H);
  for (let i = 0; i < mask.value.length; i++) {
    const c = COLORS[mask.value[i] % COLORS.length];
    maskImage.data[i * 4] = c[0]; maskImage.data[i * 4 + 1] = c[1];
    maskImage.data[i * 4 + 2] = c[2]; maskImage.data[i * 4 + 3] = 128;
  }
  maskCtx.putImageData(maskImage, 0, 0);
  const ctx = canvas.getContext('2d')${t ? '!' : ''};
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(maskCanvas, 0, 0, imgSize.value.w, imgSize.value.h);
});

async function processImage(file${t ? ': File' : ''}) {
  if (!file.type.startsWith('image/')) { status.value = 'Unsupported file type.'; return; }
  const url = URL.createObjectURL(file);
  imageUrl.value = url; mask.value = null;
  const img = new Image(); img.src = url;
  await new Promise((resolve) => { img.onload = resolve; });
  imgSize.value = { w: img.naturalWidth, h: img.naturalHeight };
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d')${t ? '!' : ''}; ctx.drawImage(img, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  if (!session.value) { status.value = 'Model not loaded yet.'; return; }
  status.value = '${config.modelName} \\u00b7 Processing...';
  const start = performance.now();
  const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
  const output = await runInference(session.value, inputTensor);
  const m = postprocessSegmentation(output, NUM_CLASSES, MASK_H, MASK_W);
  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  mask.value = m;
  URL.revokeObjectURL(url);
}

function handleDrop(e${t ? ': DragEvent' : ''}) { e.preventDefault(); dragOver.value = false; const f = e.dataTransfer?.files[0]; if (f) processImage(f); }
function handleFileChange(e${t ? ': Event' : ''}) { const f = (e.target as HTMLInputElement).files?.[0]; if (f) processImage(f); }
function reset() { imageUrl.value = null; mask.value = null; if (fileInput.value) fileInput.value.value = ''; }
</script>

<template>
  <a href="#results" class="skip-link">Skip to results</a>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="container">
      <div>
        <div v-if="!imageUrl" :class="['drop-zone', { 'drag-over': dragOver }]" role="button" tabindex="0"
          aria-label="Drop an image here or click to browse for ${taskLabel.toLowerCase()}"
          @click="fileInput?.click()" @keydown.enter.space.prevent="fileInput?.click()"
          @dragover.prevent="dragOver = true" @dragleave="dragOver = false" @drop="handleDrop">
          <p>Drop an image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input ref="fileInput" type="file" accept="image/*" style="position:absolute;left:-9999px;width:1px;height:1px;opacity:0;" aria-hidden="true" tabindex="-1" @change="handleFileChange" />
        </div>
        <div v-else class="preview">
          <div class="preview-wrapper">
            <img :src="imageUrl" alt="Selected image for segmentation" />
            <canvas ref="overlay" class="overlay-canvas"></canvas>
          </div>
          <button class="change-btn" @click="reset">Choose another image</button>
        </div>
      </div>
      <div id="results" class="results" role="status" aria-live="polite" aria-atomic="true">
      </div>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
</template>
`;
}

function emitFileFeatureExtractionPage(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script setup${tsLang}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from '~/lib/inference.${le}';
import { preprocessImage } from '~/lib/preprocess.${le}';
import { postprocessEmbeddings } from '~/lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';

const session = shallowRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
const embedding = shallowRef${t ? '<{ dims: number; norm: string; first5: string } | null>' : ''}(null);
const imageUrl = ref${t ? '<string | null>' : ''}(null);
const status = ref('Loading model...');
const dragOver = ref(false);
const fileInput = ref${t ? '<HTMLInputElement | null>' : ''}(null);

onMounted(() => {
  createSession(MODEL_PATH).then((s) => {
    session.value = s;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => { status.value = 'Failed to load model'; console.error('Model load error:', e); });
});

async function processImage(file${t ? ': File' : ''}) {
  if (!file.type.startsWith('image/')) { status.value = 'Unsupported file type.'; return; }
  const url = URL.createObjectURL(file);
  imageUrl.value = url; embedding.value = null;
  const img = new Image(); img.src = url;
  await new Promise((resolve) => { img.onload = resolve; });
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d')${t ? '!' : ''}; ctx.drawImage(img, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  if (!session.value) { status.value = 'Model not loaded yet.'; return; }
  status.value = '${config.modelName} \\u00b7 Processing...';
  const start = performance.now();
  const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
  const output = await runInference(session.value, inputTensor);
  const emb = postprocessEmbeddings(output);
  let norm = 0;
  for (let i = 0; i < emb.length; i++) { norm += emb[i] * emb[i]; }
  norm = Math.sqrt(norm);
  const first5 = Array.from(emb.slice(0, 5)).map((v${t ? ': number' : ''}) => v.toFixed(4)).join(', ');
  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  embedding.value = { dims: emb.length, norm: norm.toFixed(4), first5 };
  URL.revokeObjectURL(url);
}

function handleDrop(e${t ? ': DragEvent' : ''}) { e.preventDefault(); dragOver.value = false; const f = e.dataTransfer?.files[0]; if (f) processImage(f); }
function handleFileChange(e${t ? ': Event' : ''}) { const f = (e.target as HTMLInputElement).files?.[0]; if (f) processImage(f); }
function reset() { imageUrl.value = null; embedding.value = null; if (fileInput.value) fileInput.value.value = ''; }
</script>

<template>
  <a href="#results" class="skip-link">Skip to results</a>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="container">
      <div>
        <div v-if="!imageUrl" :class="['drop-zone', { 'drag-over': dragOver }]" role="button" tabindex="0"
          aria-label="Drop an image here or click to browse for ${taskLabel.toLowerCase()}"
          @click="fileInput?.click()" @keydown.enter.space.prevent="fileInput?.click()"
          @dragover.prevent="dragOver = true" @dragleave="dragOver = false" @drop="handleDrop">
          <p>Drop an image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input ref="fileInput" type="file" accept="image/*" style="position:absolute;left:-9999px;width:1px;height:1px;opacity:0;" aria-hidden="true" tabindex="-1" @change="handleFileChange" />
        </div>
        <div v-else class="preview">
          <img :src="imageUrl" alt="Selected image for feature extraction" />
          <button class="change-btn" @click="reset">Choose another image</button>
        </div>
      </div>
      <div id="results" class="results embedding-info" role="status" aria-live="polite" aria-atomic="true">
        <template v-if="embedding">
          <p><strong>Dimensions:</strong> {{ embedding.dims }}</p>
          <p><strong>L2 Norm:</strong> {{ embedding.norm }}</p>
          <p><strong>First 5 values:</strong> [{{ embedding.first5 }}, ...]</p>
        </template>
      </div>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
</template>
`;
}

// ---- Depth Estimation Page ----

function emitDepthEstimationPage(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script setup${tsLang}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from '~/lib/inference.${le}';
import { preprocessImage } from '~/lib/preprocess.${le}';
import { postprocessDepth } from '~/lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';

const session = shallowRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
const imageUrl = ref${t ? '<string | null>' : ''}(null);
const status = ref('Loading model...');
const dragOver = ref(false);
const fileInput = ref${t ? '<HTMLInputElement | null>' : ''}(null);
const depthCanvas = ref${t ? '<HTMLCanvasElement | null>' : ''}(null);

onMounted(() => {
  createSession(MODEL_PATH).then((s) => {
    session.value = s;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => { status.value = 'Failed to load model'; console.error('Model load error:', e); });
});

async function processImage(file${t ? ': File' : ''}) {
  if (!file.type.startsWith('image/')) { status.value = 'Unsupported file type.'; return; }
  const url = URL.createObjectURL(file);
  imageUrl.value = url;
  const img = new Image(); img.src = url;
  await new Promise((resolve) => { img.onload = resolve; });
  if (!session.value) { status.value = 'Model not loaded yet.'; return; }
  status.value = '${config.modelName} \\u00b7 Processing...';
  const start = performance.now();
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d')${t ? '!' : ''}; ctx.drawImage(img, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
  const output = await runInference(session.value, inputTensor);
  const depthMap = postprocessDepth(output);
  if (depthCanvas.value) {
    depthCanvas.value.width = img.naturalWidth; depthCanvas.value.height = img.naturalHeight;
    const dCtx = depthCanvas.value.getContext('2d')${t ? '!' : ''};
    const imgData = dCtx.createImageData(depthCanvas.value.width, depthCanvas.value.height);
    for (let i = 0; i < depthMap.length; i++) {
      const v = depthMap[i];
      imgData.data[i * 4] = v; imgData.data[i * 4 + 1] = v; imgData.data[i * 4 + 2] = v; imgData.data[i * 4 + 3] = 255;
    }
    dCtx.putImageData(imgData, 0, 0);
  }
  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  URL.revokeObjectURL(url);
}

function handleDrop(e${t ? ': DragEvent' : ''}) { e.preventDefault(); dragOver.value = false; const f = e.dataTransfer?.files[0]; if (f) processImage(f); }
function handleFileChange(e${t ? ': Event' : ''}) { const f = (e.target as HTMLInputElement).files?.[0]; if (f) processImage(f); }
function reset() { imageUrl.value = null; if (fileInput.value) fileInput.value.value = ''; }
</script>

<template>
  <a href="#results" class="skip-link">Skip to results</a>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="container">
      <div>
        <div v-if="!imageUrl" :class="['drop-zone', { 'drag-over': dragOver }]" role="button" tabindex="0"
          aria-label="Drop an image here or click to browse for ${taskLabel.toLowerCase()}"
          @click="fileInput?.click()" @keydown.enter.space.prevent="fileInput?.click()"
          @dragover.prevent="dragOver = true" @dragleave="dragOver = false" @drop="handleDrop">
          <p>Drop an image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input ref="fileInput" type="file" accept="image/*" style="position:absolute;left:-9999px;width:1px;height:1px;opacity:0;" aria-hidden="true" tabindex="-1" @change="handleFileChange" />
        </div>
        <div v-else class="preview">
          <img :src="imageUrl" alt="Source image for depth estimation" />
          <button class="change-btn" @click="reset">Choose another image</button>
        </div>
      </div>
      <div id="results"><canvas ref="depthCanvas" class="depth-canvas"></canvas></div>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
</template>
`;
}

// ---- Audio: File + Classification Page ----

function emitFileAudioClassificationPage(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script setup${tsLang}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from '~/lib/inference.${le}';
import { melSpectrogram, mfcc } from '~/lib/preprocess.${le}';
import { postprocessResults } from '~/lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';

const session = shallowRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
const results = shallowRef${t ? '<{ indices: number[]; values: number[] } | null>' : ''}(null);
const status = ref('Loading model...');

onMounted(() => {
  createSession(MODEL_PATH).then((s) => {
    session.value = s;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => { status.value = 'Failed to load model'; console.error('Model load error:', e); });
});

async function handleFileChange(e${t ? ': Event' : ''}) {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file || !session.value) return;

  status.value = '${config.modelName} \\u00b7 Decoding audio...';

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

  status.value = '${config.modelName} \\u00b7 Processing...';
  const start = performance.now();

  const mel = melSpectrogram(samples, 16000, 512, 160, 40);
  const features = mfcc(mel.data, mel.numFrames, mel.numMelBands, 13);
  const output = await runInference(session.value, features);
  const r = postprocessResults(output);

  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  results.value = r;
}
</script>

<template>
  <a href="#results" class="skip-link">Skip to results</a>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div>
      <label for="fileInput">Choose an audio file</label>
      <input id="fileInput" type="file" accept="audio/*" @change="handleFileChange" aria-label="Select audio file for ${taskLabel.toLowerCase()}" />
    </div>
    <div id="results" class="results" role="status" aria-live="polite" aria-atomic="true">
      <template v-if="results">
        <template v-for="(idx, i) in results.indices" :key="idx">
          <div v-if="results.values[i] >= 0.01" :class="['result-row', { 'top-result': i === 0 }]" tabindex="0"
            :aria-label="\`Class \${idx}: \${(results.values[i] * 100).toFixed(1)} percent\`">
            <span class="result-label">Class {{ idx }}</span>
            <div class="result-bar-container">
              <div class="result-bar" :style="{ width: (results.values[i] / (results.values[0] || 1)) * 100 + '%' }"></div>
            </div>
            <span class="result-pct">{{ (results.values[i] * 100).toFixed(1) }}%</span>
          </div>
        </template>
      </template>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
</template>
`;
}

// ---- Audio: File + Speech-to-Text Page ----

function emitFileSpeechToTextPage(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script setup${tsLang}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from '~/lib/inference.${le}';
import { melSpectrogram } from '~/lib/preprocess.${le}';
import { postprocessTranscript } from '~/lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
const VOCAB = [' ', ...'abcdefghijklmnopqrstuvwxyz'.split(''), "'"];

const session = shallowRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
const transcript = ref('');
const status = ref('Loading model...');

onMounted(() => {
  createSession(MODEL_PATH).then((s) => {
    session.value = s;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => { status.value = 'Failed to load model'; console.error('Model load error:', e); });
});

async function handleFileChange(e${t ? ': Event' : ''}) {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file || !session.value) return;

  status.value = '${config.modelName} \\u00b7 Decoding audio...';

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

  status.value = '${config.modelName} \\u00b7 Processing...';
  const start = performance.now();

  const mel = melSpectrogram(samples, 16000, 512, 160, 80);
  const output = await runInference(session.value, mel.data);
  const vocabSize = VOCAB.length + 1;
  const numTimesteps = Math.floor(output.length / vocabSize);
  const text = postprocessTranscript(output, numTimesteps, vocabSize, VOCAB);

  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  transcript.value = text || '(no speech detected)';
}
</script>

<template>
  <a href="#transcript" class="skip-link">Skip to results</a>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div>
      <label for="fileInput">Choose an audio file</label>
      <input id="fileInput" type="file" accept="audio/*" @change="handleFileChange" aria-label="Select audio file for ${taskLabel.toLowerCase()}" />
    </div>
    <pre id="transcript" class="transcript" role="status" aria-live="polite" aria-atomic="true">{{ transcript }}</pre>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
</template>
`;
}

// ---- Audio: Mic + Speech-to-Text Page ----

function emitMicSpeechToTextPage(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script setup${tsLang}>
import { ref, shallowRef, onMounted, onUnmounted } from 'vue';
import { createSession, runInference, getBackendLabel } from '~/lib/inference.${le}';
import { melSpectrogram } from '~/lib/preprocess.${le}';
import { postprocessTranscript } from '~/lib/postprocess.${le}';
import { startAudioCapture, stopStream, createAudioInferenceLoop } from '~/lib/input.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
const VOCAB = [' ', ...'abcdefghijklmnopqrstuvwxyz'.split(''), "'"];

const session = shallowRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
const transcript = ref('(listening...)');
const status = ref('Loading model...');
const recording = ref(false);
let capture${t ? ': any' : ''} = null;
let loop${t ? ': any' : ''} = null;

onMounted(() => {
  createSession(MODEL_PATH).then((s) => {
    session.value = s;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => { status.value = 'Failed to load model'; console.error('Model load error:', e); });
});

onUnmounted(() => {
  if (loop) loop.stop();
  if (capture) { stopStream(capture.stream); capture.audioContext.close(); }
});

async function handleStart() {
  if (!session.value) { status.value = 'Model not loaded yet.'; return; }
  try {
    capture = await startAudioCapture(16000);
    recording.value = true;
    status.value = '${config.modelName} \\u00b7 Listening...';
    loop = createAudioInferenceLoop({
      getSamples: capture.getSamples,
      onResult(text${t ? ': string' : ''}) { transcript.value = text || '(listening...)'; },
      intervalMs: 2000,
    });
    loop.start();
  } catch (e) { status.value = 'Microphone access denied'; console.error('Mic error:', e); }
}

function handleStop() {
  if (loop) { loop.stop(); loop = null; }
  if (capture) { stopStream(capture.stream); capture.audioContext.close(); capture = null; }
  recording.value = false;
  status.value = '${config.modelName} \\u00b7 Stopped';
}
</script>

<template>
  <a href="#transcript" class="skip-link">Skip to results</a>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="controls" role="group" aria-label="Recording controls">
      <button @click="handleStart" :disabled="recording" aria-label="Start recording">Start Recording</button>
      <button @click="handleStop" :disabled="!recording" aria-label="Stop recording">Stop Recording</button>
    </div>
    <pre id="transcript" class="transcript" role="status" aria-live="polite" aria-atomic="true">{{ transcript }}</pre>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
</template>
`;
}

// ---- Audio: Mic + Classification Page ----

function emitMicAudioClassificationPage(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script setup${tsLang}>
import { ref, shallowRef, onMounted, onUnmounted } from 'vue';
import { createSession, runInference, getBackendLabel } from '~/lib/inference.${le}';
import { melSpectrogram, mfcc } from '~/lib/preprocess.${le}';
import { postprocessResults } from '~/lib/postprocess.${le}';
import { startAudioCapture, stopStream, createAudioInferenceLoop } from '~/lib/input.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';

const session = shallowRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
const results = shallowRef${t ? '<{ indices: number[]; values: number[] } | null>' : ''}(null);
const status = ref('Loading model...');
const recording = ref(false);
let capture${t ? ': any' : ''} = null;
let loop${t ? ': any' : ''} = null;

onMounted(() => {
  createSession(MODEL_PATH).then((s) => {
    session.value = s;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => { status.value = 'Failed to load model'; console.error('Model load error:', e); });
});

onUnmounted(() => {
  if (loop) loop.stop();
  if (capture) { stopStream(capture.stream); capture.audioContext.close(); }
});

async function handleStart() {
  if (!session.value) { status.value = 'Model not loaded yet.'; return; }
  try {
    capture = await startAudioCapture(16000);
    recording.value = true;
    status.value = '${config.modelName} \\u00b7 Listening...';
    loop = createAudioInferenceLoop({
      getSamples: capture.getSamples,
      onResult(r${t ? ': { indices: number[]; values: number[] }' : ''}) { results.value = r; },
      intervalMs: 2000,
    });
    loop.start();
  } catch (e) { status.value = 'Microphone access denied'; console.error('Mic error:', e); }
}

function handleStop() {
  if (loop) { loop.stop(); loop = null; }
  if (capture) { stopStream(capture.stream); capture.audioContext.close(); capture = null; }
  recording.value = false;
  status.value = '${config.modelName} \\u00b7 Stopped';
}
</script>

<template>
  <a href="#results" class="skip-link">Skip to results</a>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="controls" role="group" aria-label="Recording controls">
      <button @click="handleStart" :disabled="recording" aria-label="Start recording">Start Recording</button>
      <button @click="handleStop" :disabled="!recording" aria-label="Stop recording">Stop Recording</button>
    </div>
    <div id="results" class="results" role="status" aria-live="polite" aria-atomic="true">
      <template v-if="results">
        <template v-for="(idx, i) in results.indices" :key="idx">
          <div v-if="results.values[i] >= 0.01" :class="['result-row', { 'top-result': i === 0 }]" tabindex="0"
            :aria-label="\`Class \${idx}: \${(results.values[i] * 100).toFixed(1)} percent\`">
            <span class="result-label">Class {{ idx }}</span>
            <div class="result-bar-container">
              <div class="result-bar" :style="{ width: (results.values[i] / (results.values[0] || 1)) * 100 + '%' }"></div>
            </div>
            <span class="result-pct">{{ (results.values[i] * 100).toFixed(1) }}%</span>
          </div>
        </template>
      </template>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
</template>
`;
}

// ---- Audio: Text-to-Speech Page ----

function emitTextToSpeechPage(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script setup${tsLang}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from '~/lib/inference.${le}';
import { postprocessAudio, playAudio } from '~/lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';

const session = shallowRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
const status = ref('Loading model...');
const text = ref('Hello, this is a test of text to speech.');

onMounted(() => {
  createSession(MODEL_PATH).then((s) => {
    session.value = s;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => { status.value = 'Failed to load model'; console.error('Model load error:', e); });
});

async function handleSynthesize() {
  const trimmed = text.value.trim();
  if (!trimmed || !session.value) return;

  status.value = '${config.modelName} \\u00b7 Synthesizing...';
  const start = performance.now();

  const tokens = new Float32Array(trimmed.length);
  for (let i = 0; i < trimmed.length; i++) { tokens[i] = trimmed.charCodeAt(i); }

  const output = await runInference(session.value, tokens);
  const samples = postprocessAudio(output);
  await playAudio(samples);

  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
}
</script>

<template>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="tts-input">
      <label for="textInput">Enter text to synthesize</label>
      <textarea id="textInput" rows="4" v-model="text" aria-label="Text to synthesize"></textarea>
      <button class="primary-btn" @click="handleSynthesize" aria-label="Synthesize speech">Synthesize</button>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
</template>
`;
}

// ---- Audio: Audio-to-Audio Page ----

function emitAudioToAudioPage(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script setup${tsLang}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from '~/lib/inference.${le}';
import { postprocessAudioToAudio, playAudio } from '~/lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';

const session = shallowRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
const status = ref('Loading model...');
const output = ref('');

onMounted(() => {
  createSession(MODEL_PATH).then((s) => {
    session.value = s;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => { status.value = 'Failed to load model'; console.error('Model load error:', e); });
});

async function handleFileChange(e${t ? ': Event' : ''}) {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file || !session.value) return;

  status.value = '${config.modelName} \\u00b7 Processing...';
  const start = performance.now();

  const audioCtx = new AudioContext({ sampleRate: 16000 });
  const arrayBuffer = await file.arrayBuffer();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  const samples = audioBuffer.getChannelData(0);
  const input = new Float32Array(samples);
  const raw = await runInference(session.value, input);
  const processedSamples = postprocessAudioToAudio(raw);
  await playAudio(processedSamples);

  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  output.value = 'Processed ' + (samples.length / 16000).toFixed(1) + 's of audio. Playing output...';
}
</script>

<template>
  <a href="#results" class="skip-link">Skip to results</a>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div>
      <label for="fileInput">Choose an audio file</label>
      <input id="fileInput" type="file" accept="audio/*" @change="handleFileChange" aria-label="Select audio file for ${taskLabel.toLowerCase()}" />
    </div>
    <div id="results" class="generation-output" role="status" aria-live="polite" aria-atomic="true">{{ output }}</div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
</template>
`;
}

// ---- Speaker Diarization Page ----

function emitSpeakerDiarizationPage(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script setup${tsLang}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from '~/lib/inference.${le}';
import { postprocessSpeakerDiarization } from '~/lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';

const session = shallowRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
const segments = shallowRef${t ? '<Array<{ speaker: number; start: number; end: number; text: string }> | null>' : ''}(null);
const status = ref('Loading model...');

onMounted(() => {
  createSession(MODEL_PATH).then((s) => {
    session.value = s;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => { status.value = 'Failed to load model'; console.error('Model load error:', e); });
});

async function handleFileChange(e${t ? ': Event' : ''}) {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file || !session.value) return;

  status.value = '${config.modelName} \\u00b7 Processing...';
  const start = performance.now();

  const audioCtx = new AudioContext({ sampleRate: 16000 });
  const arrayBuffer = await file.arrayBuffer();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  const samples = audioBuffer.getChannelData(0);
  const input = new Float32Array(samples);
  const raw = await runInference(session.value, input);
  const results = postprocessSpeakerDiarization(raw);

  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  segments.value = results;
}
</script>

<template>
  <a href="#results" class="skip-link">Skip to results</a>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div>
      <label for="fileInput">Choose an audio file</label>
      <input id="fileInput" type="file" accept="audio/*" @change="handleFileChange" aria-label="Select audio file for ${taskLabel.toLowerCase()}" />
    </div>
    <div id="results" class="diarization-timeline" role="status" aria-live="polite" aria-atomic="true">
      <template v-if="segments">
        <div v-for="(seg, i) in segments" :key="i" class="diarization-segment" tabindex="0"
          :aria-label="\`Speaker \${seg.speaker}: \${seg.start.toFixed(1)}s to \${seg.end.toFixed(1)}s\`">
          <span class="speaker">Speaker {{ seg.speaker }}</span>
          <span>{{ seg.text }}</span>
          <span class="time">{{ seg.start.toFixed(1) }}s - {{ seg.end.toFixed(1) }}s</span>
        </div>
      </template>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
</template>
`;
}

// ---- Voice Activity Detection Page ----

function emitVADPage(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script setup${tsLang}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from '~/lib/inference.${le}';
import { postprocessVAD } from '~/lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';

const session = shallowRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
const segments = shallowRef${t ? '<Array<{ label: string; start: number; end: number }> | null>' : ''}(null);
const status = ref('Loading model...');

onMounted(() => {
  createSession(MODEL_PATH).then((s) => {
    session.value = s;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => { status.value = 'Failed to load model'; console.error('Model load error:', e); });
});

async function handleFileChange(e${t ? ': Event' : ''}) {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file || !session.value) return;

  status.value = '${config.modelName} \\u00b7 Processing...';
  const start = performance.now();

  const audioCtx = new AudioContext({ sampleRate: 16000 });
  const arrayBuffer = await file.arrayBuffer();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  const samples = audioBuffer.getChannelData(0);
  const input = new Float32Array(samples);
  const raw = await runInference(session.value, input);
  const results = postprocessVAD(raw);

  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  segments.value = results;
}
</script>

<template>
  <a href="#results" class="skip-link">Skip to results</a>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div>
      <label for="fileInput">Choose an audio file</label>
      <input id="fileInput" type="file" accept="audio/*" @change="handleFileChange" aria-label="Select audio file for ${taskLabel.toLowerCase()}" />
    </div>
    <div id="results" class="vad-segments" role="status" aria-live="polite" aria-atomic="true">
      <template v-if="segments">
        <div v-for="(seg, i) in segments" :key="i" class="vad-segment" tabindex="0"
          :aria-label="\`\${seg.label}: \${seg.start.toFixed(1)}s to \${seg.end.toFixed(1)}s\`">
          <span class="label">{{ seg.label }}</span>
          <span class="time">{{ seg.start.toFixed(1) }}s - {{ seg.end.toFixed(1) }}s</span>
        </div>
      </template>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
</template>
`;
}

// ---- Text: Classification Page ----

function emitTextClassificationPage(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script setup${tsLang}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from '~/lib/inference.${le}';
import { loadTokenizer, tokenizeText } from '~/lib/preprocess.${le}';
import { postprocessResults } from '~/lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

const session = shallowRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
const tokenizer = shallowRef${t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : ''}(null);
const results = shallowRef${t ? '<{ indices: number[]; values: number[] } | null>' : ''}(null);
const text = ref('This is an amazing product!');
const status = ref('Loading model and tokenizer...');

onMounted(() => {
  Promise.all([
    createSession(MODEL_PATH),
    loadTokenizer(TOKENIZER_PATH),
  ]).then(([s, tok]) => {
    session.value = s;
    tokenizer.value = tok;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => { status.value = 'Failed to load model or tokenizer'; console.error('Load error:', e); });
});

async function handleClassify() {
  const trimmed = text.value.trim();
  if (!trimmed || !session.value || !tokenizer.value) return;

  status.value = '${config.modelName} \\u00b7 Processing...';
  const start = performance.now();

  const { inputIds, attentionMask } = tokenizeText(tokenizer.value, trimmed, 128);
  const output = await runInference(session.value, inputIds, attentionMask);
  const r = postprocessResults(output);

  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  results.value = r;
}
</script>

<template>
  <a href="#results" class="skip-link">Skip to results</a>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="text-input">
      <label for="textInput">Enter text to classify</label>
      <textarea id="textInput" rows="4" v-model="text" aria-label="Text to classify"></textarea>
      <button class="run-btn" @click="handleClassify" :disabled="!session || !tokenizer" aria-label="Classify text">Classify</button>
    </div>
    <div id="results" class="results" role="status" aria-live="polite" aria-atomic="true">
      <template v-if="results">
        <template v-for="(idx, i) in results.indices" :key="idx">
          <div v-if="results.values[i] >= 0.01" :class="['result-row', { 'top-result': i === 0 }]" tabindex="0"
            :aria-label="\`Class \${idx}: \${(results.values[i] * 100).toFixed(1)} percent\`">
            <span class="result-label">Class {{ idx }}</span>
            <div class="result-bar-container">
              <div class="result-bar" :style="{ width: (results.values[i] / (results.values[0] || 1)) * 100 + '%' }"></div>
            </div>
            <span class="result-pct">{{ (results.values[i] * 100).toFixed(1) }}%</span>
          </div>
        </template>
      </template>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
</template>
`;
}

// ---- Text: Zero-Shot Classification Page ----

function emitZeroShotClassificationPage(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script setup${tsLang}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from '~/lib/inference.${le}';
import { loadTokenizer, tokenizeText } from '~/lib/preprocess.${le}';
import { postprocessZeroShot } from '~/lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

const session = shallowRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
const tokenizer = shallowRef${t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : ''}(null);
const results = shallowRef${t ? '<Array<{ label: string; score: number }> | null>' : ''}(null);
const text = ref('This is a thrilling adventure story.');
const labels = ref('travel, cooking, politics, sports, technology');
const status = ref('Loading model and tokenizer...');

onMounted(() => {
  Promise.all([
    createSession(MODEL_PATH),
    loadTokenizer(TOKENIZER_PATH),
  ]).then(([s, tok]) => {
    session.value = s;
    tokenizer.value = tok;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => { status.value = 'Failed to load model or tokenizer'; console.error('Load error:', e); });
});

async function handleClassify() {
  if (!session.value || !tokenizer.value || !text.value.trim() || !labels.value.trim()) return;

  status.value = '${config.modelName} \\u00b7 Processing...';
  const start = performance.now();

  const candidateLabels = labels.value.split(',').map((l) => l.trim()).filter(Boolean);
  const scores${t ? ': number[]' : ''} = [];

  for (const label of candidateLabels) {
    const hypothesis = \`This text is about \${label}.\`;
    const { inputIds, attentionMask } = tokenizeText(tokenizer.value, text.value + ' ' + hypothesis, 128);
    const output = await runInference(session.value, inputIds, attentionMask);
    const entailmentScore = output[2] ?? output[output.length - 1];
    scores.push(entailmentScore);
  }

  const r = postprocessZeroShot(scores, candidateLabels);

  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  results.value = r;
}
</script>

<template>
  <a href="#results" class="skip-link">Skip to results</a>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="text-input">
      <label for="textInput">Enter text to classify</label>
      <textarea id="textInput" rows="4" v-model="text" aria-label="Text to classify"></textarea>
      <label for="labelsInput">Candidate labels (comma-separated)</label>
      <input id="labelsInput" type="text" v-model="labels" aria-label="Candidate labels" class="labels-input" />
      <button class="run-btn" @click="handleClassify" :disabled="!session || !tokenizer" aria-label="Classify text">Classify</button>
    </div>
    <div id="results" class="results" role="status" aria-live="polite" aria-atomic="true">
      <template v-if="results">
        <div v-for="(r, i) in results" :key="r.label" :class="['result-row', { 'top-result': i === 0 }]" tabindex="0"
          :aria-label="\`\${r.label}: \${(r.score * 100).toFixed(1)} percent\`">
          <span class="result-label">{{ r.label }}</span>
          <div class="result-bar-container">
            <div class="result-bar" :style="{ width: (r.score / (results[0]?.score || 1)) * 100 + '%' }"></div>
          </div>
          <span class="result-pct">{{ (r.score * 100).toFixed(1) }}%</span>
        </div>
      </template>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
</template>
`;
}

// ---- Text: Generation Page ----

function emitTextGenerationPage(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script setup${tsLang}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from '~/lib/inference.${le}';
import { loadTokenizer } from '~/lib/preprocess.${le}';
import { postprocessGeneration, sampleNextToken } from '~/lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

const session = shallowRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
const tokenizer = shallowRef${t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : ''}(null);
const text = ref('Once upon a time');
const output = ref('');
const status = ref('Loading model and tokenizer...');
const generating = ref(false);

onMounted(() => {
  Promise.all([
    createSession(MODEL_PATH),
    loadTokenizer(TOKENIZER_PATH),
  ]).then(([s, tok]) => {
    session.value = s;
    tokenizer.value = tok;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => { status.value = 'Failed to load model or tokenizer'; console.error('Load error:', e); });
});

async function handleGenerate() {
  if (!session.value || !tokenizer.value || !text.value.trim() || generating.value) return;

  generating.value = true;
  status.value = '${config.modelName} \\u00b7 Generating...';
  const start = performance.now();

  const encoded = tokenizer.value.encode(text.value);
  let inputIds = encoded.inputIds;
  let generatedText = text.value;

  const maxTokens = 50;
  const eosTokenId = 2;

  for (let i = 0; i < maxTokens; i++) {
    const seqLen = inputIds.length;
    const vocabSize = tokenizer.value.getVocabSize();

    const inputTensor = new BigInt64Array(inputIds.map((id${t ? ': number' : ''}) => BigInt(id)));
    const modelOutput = await runInference(session.value, inputTensor);

    const logits = postprocessGeneration(modelOutput, seqLen, vocabSize);
    const nextToken = sampleNextToken(logits);

    if (nextToken === eosTokenId) break;

    inputIds.push(nextToken);
    const decoded = tokenizer.value.decode([nextToken]);
    generatedText += decoded;
    output.value = generatedText;
  }

  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  generating.value = false;
}
</script>

<template>
  <a href="#output" class="skip-link">Skip to output</a>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="text-input">
      <label for="textInput">Enter prompt text</label>
      <textarea id="textInput" rows="4" v-model="text" aria-label="Prompt text" :disabled="generating"></textarea>
      <button class="run-btn" @click="handleGenerate" :disabled="generating" aria-label="Generate text">
        {{ generating ? 'Generating...' : 'Generate' }}
      </button>
    </div>
    <div id="output" class="transcript" role="status" aria-live="polite" aria-atomic="true">
      {{ output || '(generated text will appear here)' }}
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
</template>
`;
}

// ---- Fill-Mask Page ----

function emitFillMaskPage(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script setup${tsLang}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from '~/lib/inference.${le}';
import { loadTokenizer, tokenizeText } from '~/lib/preprocess.${le}';
import { postprocessFillMask } from '~/lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

const session = shallowRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
const tokenizer = shallowRef${t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : ''}(null);
const predictions = shallowRef${t ? '<Array<{ token: string; score: number }> | null>' : ''}(null);
const text = ref('The capital of France is [MASK].');
const status = ref('Loading model and tokenizer...');

onMounted(() => {
  Promise.all([
    createSession(MODEL_PATH),
    loadTokenizer(TOKENIZER_PATH),
  ]).then(([s, tok]) => {
    session.value = s;
    tokenizer.value = tok;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => { status.value = 'Failed to load model or tokenizer'; console.error('Load error:', e); });
});

async function handlePredict() {
  const trimmed = text.value.trim();
  if (!trimmed || !session.value || !tokenizer.value) return;

  status.value = '${config.modelName} \\u00b7 Processing...';
  const start = performance.now();

  const { inputIds } = tokenizeText(tokenizer.value, trimmed);
  const output = await runInference(session.value, inputIds);
  const results = postprocessFillMask(output, inputIds, tokenizer.value);

  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  predictions.value = results;
}
</script>

<template>
  <a href="#results" class="skip-link">Skip to results</a>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="text-input">
      <label for="textInput">Enter text with [MASK] token</label>
      <textarea id="textInput" rows="4" v-model="text" aria-label="Text with MASK token"></textarea>
      <button class="run-btn" @click="handlePredict" :disabled="!session || !tokenizer" aria-label="Predict masked token">Predict</button>
    </div>
    <div id="results" class="mask-predictions" role="status" aria-live="polite" aria-atomic="true">
      <template v-if="predictions">
        <div v-for="(p, i) in predictions" :key="i" class="mask-prediction" tabindex="0"
          :aria-label="\`\${p.token}: \${(p.score * 100).toFixed(1)} percent\`">
          <span class="token">{{ p.token }}</span>
          <span class="prob">{{ (p.score * 100).toFixed(1) }}%</span>
        </div>
      </template>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
</template>
`;
}

// ---- Sentence Similarity Page ----

function emitSentenceSimilarityPage(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script setup${tsLang}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from '~/lib/inference.${le}';
import { loadTokenizer, tokenizeText } from '~/lib/preprocess.${le}';
import { cosineSimilarity } from '~/lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

const session = shallowRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
const tokenizer = shallowRef${t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : ''}(null);
const scores = shallowRef${t ? '<Array<{ sentence: string; score: number }> | null>' : ''}(null);
const source = ref('The weather is lovely today.');
const comparisons = ref('It is a beautiful day.\\nThe sun is shining bright.\\nI need to buy groceries.');
const status = ref('Loading model and tokenizer...');

onMounted(() => {
  Promise.all([
    createSession(MODEL_PATH),
    loadTokenizer(TOKENIZER_PATH),
  ]).then(([s, tok]) => {
    session.value = s;
    tokenizer.value = tok;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => { status.value = 'Failed to load model or tokenizer'; console.error('Load error:', e); });
});

async function handleCompare() {
  if (!session.value || !tokenizer.value || !source.value.trim()) return;

  status.value = '${config.modelName} \\u00b7 Computing...';
  const start = performance.now();

  const { inputIds: srcIds } = tokenizeText(tokenizer.value, source.value);
  const srcEmb = await runInference(session.value, srcIds);
  const lines = comparisons.value.split('\\n').filter(Boolean);
  const results${t ? ': Array<{ sentence: string; score: number }>' : ''} = [];
  for (const line of lines) {
    const { inputIds: cmpIds } = tokenizeText(tokenizer.value, line);
    const cmpEmb = await runInference(session.value, cmpIds);
    results.push({ sentence: line, score: cosineSimilarity(srcEmb, cmpEmb) });
  }

  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  scores.value = results;
}
</script>

<template>
  <a href="#results" class="skip-link">Skip to results</a>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="text-input">
      <label for="source">Source sentence</label>
      <textarea id="source" rows="2" v-model="source" aria-label="Source sentence"></textarea>
      <label for="compare">Sentences to compare (one per line)</label>
      <textarea id="compare" rows="4" v-model="comparisons" aria-label="Comparison sentences"></textarea>
      <button class="run-btn" @click="handleCompare" :disabled="!session || !tokenizer" aria-label="Compare sentences">Compare</button>
    </div>
    <div id="results" class="similarity-pairs" role="status" aria-live="polite" aria-atomic="true">
      <template v-if="scores">
        <div v-for="(s, i) in scores" :key="i" class="similarity-score" tabindex="0"
          :aria-label="\`\${s.sentence}: \${s.score.toFixed(4)}\`">
          <span>{{ s.sentence }}</span>
          <span class="value">{{ s.score.toFixed(4) }}</span>
        </div>
      </template>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
</template>
`;
}

// ---- Token Classification (NER) Page ----

function emitTokenClassificationPage(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script setup${tsLang}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from '~/lib/inference.${le}';
import { loadTokenizer, tokenizeText } from '~/lib/preprocess.${le}';
import { postprocessTokenClassification } from '~/lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

const session = shallowRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
const tokenizer = shallowRef${t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : ''}(null);
const nerHtml = ref('');
const text = ref('John Smith works at Google in Mountain View, California.');
const status = ref('Loading model and tokenizer...');

onMounted(() => {
  Promise.all([
    createSession(MODEL_PATH),
    loadTokenizer(TOKENIZER_PATH),
  ]).then(([s, tok]) => {
    session.value = s;
    tokenizer.value = tok;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => { status.value = 'Failed to load model or tokenizer'; console.error('Load error:', e); });
});

async function handleAnalyze() {
  const trimmed = text.value.trim();
  if (!trimmed || !session.value || !tokenizer.value) return;

  status.value = '${config.modelName} \\u00b7 Processing...';
  const start = performance.now();

  const { inputIds } = tokenizeText(tokenizer.value, trimmed);
  const output = await runInference(session.value, inputIds);
  const entities = postprocessTokenClassification(output, inputIds, tokenizer.value);
  let result = trimmed;
  for (let i = entities.length - 1; i >= 0; i--) {
    const e = entities[i];
    result = result.slice(0, e.start) + '<span class="ner-entity" data-type="' + e.type + '">' + result.slice(e.start, e.end) + '</span>' + result.slice(e.end);
  }
  nerHtml.value = result;

  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
}
</script>

<template>
  <a href="#results" class="skip-link">Skip to results</a>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="text-input">
      <label for="textInput">Enter text to analyze</label>
      <textarea id="textInput" rows="4" v-model="text" aria-label="Text to analyze for named entities"></textarea>
      <button class="run-btn" @click="handleAnalyze" :disabled="!session || !tokenizer" aria-label="Analyze entities">Analyze</button>
    </div>
    <div id="results" class="ner-output" role="status" aria-live="polite" aria-atomic="true">
      <span v-if="nerHtml" v-html="nerHtml"></span>
      <span v-else>{{ text }}</span>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
</template>
`;
}

// ---- Question Answering Page ----

function emitQuestionAnsweringPage(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script setup${tsLang}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from '~/lib/inference.${le}';
import { loadTokenizer, tokenizeText } from '~/lib/preprocess.${le}';
import { postprocessQA } from '~/lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

const session = shallowRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
const tokenizer = shallowRef${t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : ''}(null);
const answer = shallowRef${t ? '<{ answer: string; score: number } | null>' : ''}(null);
const context = ref('The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was constructed from 1887 to 1889.');
const question = ref('When was the Eiffel Tower built?');
const status = ref('Loading model and tokenizer...');

onMounted(() => {
  Promise.all([
    createSession(MODEL_PATH),
    loadTokenizer(TOKENIZER_PATH),
  ]).then(([s, tok]) => {
    session.value = s;
    tokenizer.value = tok;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => { status.value = 'Failed to load model or tokenizer'; console.error('Load error:', e); });
});

async function handleAnswer() {
  if (!session.value || !tokenizer.value || !context.value.trim() || !question.value.trim()) return;

  status.value = '${config.modelName} \\u00b7 Processing...';
  const start = performance.now();

  const combined = question.value + ' [SEP] ' + context.value;
  const { inputIds } = tokenizeText(tokenizer.value, combined);
  const output = await runInference(session.value, inputIds);
  const result = postprocessQA(output, inputIds, tokenizer.value);

  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  answer.value = result;
}
</script>

<template>
  <a href="#results" class="skip-link">Skip to results</a>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="qa-input">
      <label for="context">Context</label>
      <textarea id="context" rows="4" v-model="context" aria-label="Context passage"></textarea>
      <label for="question">Question</label>
      <input id="question" type="text" v-model="question" aria-label="Question" class="labels-input" />
      <button class="run-btn" @click="handleAnswer" :disabled="!session || !tokenizer" aria-label="Get answer">Answer</button>
    </div>
    <div id="results" class="qa-answer" role="status" aria-live="polite" aria-atomic="true">
      <template v-if="answer">
        <div>{{ answer.answer }}</div>
        <div class="score">Confidence: {{ (answer.score * 100).toFixed(1) }}%</div>
      </template>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
</template>
`;
}

// ---- Summarization Page ----

function emitSummarizationPage(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script setup${tsLang}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from '~/lib/inference.${le}';
import { loadTokenizer, tokenizeText } from '~/lib/preprocess.${le}';
import { postprocessSummarization } from '~/lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

const session = shallowRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
const tokenizer = shallowRef${t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : ''}(null);
const output = ref('');
const text = ref('Artificial intelligence has transformed many industries. Machine learning models can now process natural language, recognize images, and generate creative content.');
const processing = ref(false);
const status = ref('Loading model and tokenizer...');

onMounted(() => {
  Promise.all([
    createSession(MODEL_PATH),
    loadTokenizer(TOKENIZER_PATH),
  ]).then(([s, tok]) => {
    session.value = s;
    tokenizer.value = tok;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => { status.value = 'Failed to load model or tokenizer'; console.error('Load error:', e); });
});

async function handleSummarize() {
  const trimmed = text.value.trim();
  if (!trimmed || !session.value || !tokenizer.value || processing.value) return;

  processing.value = true;
  status.value = '${config.modelName} \\u00b7 Summarizing...';
  const start = performance.now();

  const { inputIds } = tokenizeText(tokenizer.value, trimmed);
  const raw = await runInference(session.value, inputIds);
  const summary = postprocessSummarization(raw, tokenizer.value, 128, 1);

  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  output.value = summary;
  processing.value = false;
}
</script>

<template>
  <a href="#output" class="skip-link">Skip to output</a>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="text-input">
      <label for="textInput">Enter text to summarize</label>
      <textarea id="textInput" rows="6" v-model="text" :disabled="processing" aria-label="Text to summarize"></textarea>
      <button class="run-btn" @click="handleSummarize" :disabled="!session || !tokenizer || processing" aria-label="Summarize text">{{ processing ? 'Summarizing...' : 'Summarize' }}</button>
    </div>
    <div id="output" class="generation-output" role="status" aria-live="polite" aria-atomic="true">{{ output || '(summary will appear here)' }}</div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
</template>
`;
}

// ---- Translation Page ----

function emitTranslationPage(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script setup${tsLang}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from '~/lib/inference.${le}';
import { loadTokenizer, tokenizeText } from '~/lib/preprocess.${le}';
import { postprocessTranslation } from '~/lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

const session = shallowRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
const tokenizer = shallowRef${t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : ''}(null);
const output = ref('');
const text = ref('Hello, how are you today?');
const processing = ref(false);
const status = ref('Loading model and tokenizer...');

onMounted(() => {
  Promise.all([
    createSession(MODEL_PATH),
    loadTokenizer(TOKENIZER_PATH),
  ]).then(([s, tok]) => {
    session.value = s;
    tokenizer.value = tok;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => { status.value = 'Failed to load model or tokenizer'; console.error('Load error:', e); });
});

async function handleTranslate() {
  const trimmed = text.value.trim();
  if (!trimmed || !session.value || !tokenizer.value || processing.value) return;

  processing.value = true;
  status.value = '${config.modelName} \\u00b7 Translating...';
  const start = performance.now();

  const { inputIds } = tokenizeText(tokenizer.value, trimmed);
  const raw = await runInference(session.value, inputIds);
  const translation = postprocessTranslation(raw, tokenizer.value, 128, 1);

  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  output.value = translation;
  processing.value = false;
}
</script>

<template>
  <a href="#output" class="skip-link">Skip to output</a>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="text-input">
      <label for="textInput">Enter text to translate</label>
      <textarea id="textInput" rows="4" v-model="text" :disabled="processing" aria-label="Text to translate"></textarea>
      <button class="run-btn" @click="handleTranslate" :disabled="!session || !tokenizer || processing" aria-label="Translate text">{{ processing ? 'Translating...' : 'Translate' }}</button>
    </div>
    <div id="output" class="generation-output" role="status" aria-live="polite" aria-atomic="true">{{ output || '(translation will appear here)' }}</div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
</template>
`;
}

// ---- Text-to-Text Generation Page ----

function emitText2TextPage(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script setup${tsLang}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from '~/lib/inference.${le}';
import { loadTokenizer, tokenizeText } from '~/lib/preprocess.${le}';
import { postprocessText2Text } from '~/lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

const session = shallowRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
const tokenizer = shallowRef${t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : ''}(null);
const output = ref('');
const text = ref('Paraphrase: The house is big and beautiful.');
const processing = ref(false);
const status = ref('Loading model and tokenizer...');

onMounted(() => {
  Promise.all([
    createSession(MODEL_PATH),
    loadTokenizer(TOKENIZER_PATH),
  ]).then(([s, tok]) => {
    session.value = s;
    tokenizer.value = tok;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => { status.value = 'Failed to load model or tokenizer'; console.error('Load error:', e); });
});

async function handleRun() {
  const trimmed = text.value.trim();
  if (!trimmed || !session.value || !tokenizer.value || processing.value) return;

  processing.value = true;
  status.value = '${config.modelName} \\u00b7 Processing...';
  const start = performance.now();

  const { inputIds } = tokenizeText(tokenizer.value, trimmed);
  const raw = await runInference(session.value, inputIds);
  const result = postprocessText2Text(raw, tokenizer.value, 128, 1);

  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  output.value = result;
  processing.value = false;
}
</script>

<template>
  <a href="#output" class="skip-link">Skip to output</a>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="text-input">
      <label for="textInput">Enter input text</label>
      <textarea id="textInput" rows="4" v-model="text" :disabled="processing" aria-label="Input text"></textarea>
      <button class="run-btn" @click="handleRun" :disabled="!session || !tokenizer || processing" aria-label="Run text-to-text generation">{{ processing ? 'Processing...' : 'Run' }}</button>
    </div>
    <div id="output" class="generation-output" role="status" aria-live="polite" aria-atomic="true">{{ output || '(output will appear here)' }}</div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
</template>
`;
}

// ---- Conversational Page ----

function emitConversationalPage(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script setup${tsLang}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from '~/lib/inference.${le}';
import { loadTokenizer } from '~/lib/preprocess.${le}';
import { postprocessConversational, sampleNextToken } from '~/lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

const session = shallowRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
const tokenizer = shallowRef${t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : ''}(null);
const messages = ref${t ? '<Array<{ role: string; text: string }>>' : ''}([]);
const input = ref('');
const generating = ref(false);
const status = ref('Loading model and tokenizer...');
const history${t ? ': string[]' : ''} = [];

onMounted(() => {
  Promise.all([
    createSession(MODEL_PATH),
    loadTokenizer(TOKENIZER_PATH),
  ]).then(([s, tok]) => {
    session.value = s;
    tokenizer.value = tok;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => { status.value = 'Failed to load model or tokenizer'; console.error('Load error:', e); });
});

async function handleSend() {
  if (!session.value || !tokenizer.value || !input.value.trim() || generating.value) return;
  const userMsg = input.value.trim();
  input.value = '';
  messages.value = [...messages.value, { role: 'user', text: userMsg }];
  history.push(userMsg);
  generating.value = true;
  status.value = '${config.modelName} \\u00b7 Generating...';
  const start = performance.now();
  const prompt = history.join(' ');
  const encoded = tokenizer.value.encode(prompt);
  let inputIds = encoded.inputIds;
  for (let i = 0; i < 50; i++) {
    const inputBigInt = new BigInt64Array(inputIds.map((id${t ? ': number' : ''}) => BigInt(id)));
    const output = await runInference(session.value, inputBigInt);
    const vocabSize = tokenizer.value.getVocabSize();
    const logits = postprocessConversational(output, inputIds.length, vocabSize);
    const nextToken = sampleNextToken(logits);
    if (nextToken === 2) break;
    inputIds = [...inputIds, nextToken];
  }
  const decoded = tokenizer.value.decode(inputIds);
  const reply = decoded.slice(prompt.length).trim() || '(no response)';
  history.push(reply);
  messages.value = [...messages.value, { role: 'bot', text: reply }];
  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  generating.value = false;
}

function handleKeydown(e${t ? ': KeyboardEvent' : ''}) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
}
</script>

<template>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="chat-messages">
      <div v-for="(m, i) in messages" :key="i" :class="['chat-msg', m.role]">{{ m.text }}</div>
    </div>
    <div class="chat-input-row">
      <input type="text" v-model="input" @keydown="handleKeydown" placeholder="Type a message..." :disabled="generating" aria-label="Chat input" />
      <button class="run-btn" @click="handleSend" :disabled="generating" aria-label="Send message">Send</button>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
</template>
`;
}

// ---- Table Question Answering Page ----

function emitTableQAPage(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script setup${tsLang}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from '~/lib/inference.${le}';
import { loadTokenizer, tokenizeText } from '~/lib/preprocess.${le}';
import { postprocessTableQA } from '~/lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

const session = shallowRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
const tokenizer = shallowRef${t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : ''}(null);
const answer = shallowRef${t ? '<{ answer: string; score: number } | null>' : ''}(null);
const table = ref('Name, Age, City\\nAlice, 30, New York\\nBob, 25, San Francisco');
const question = ref('Who lives in San Francisco?');
const status = ref('Loading model and tokenizer...');

onMounted(() => {
  Promise.all([
    createSession(MODEL_PATH),
    loadTokenizer(TOKENIZER_PATH),
  ]).then(([s, tok]) => {
    session.value = s;
    tokenizer.value = tok;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => { status.value = 'Failed to load model or tokenizer'; console.error('Load error:', e); });
});

async function handleAnswer() {
  if (!session.value || !tokenizer.value || !table.value.trim() || !question.value.trim()) return;

  status.value = '${config.modelName} \\u00b7 Processing...';
  const start = performance.now();

  const combined = table.value + ' [SEP] ' + question.value;
  const { inputIds } = tokenizeText(tokenizer.value, combined);
  const output = await runInference(session.value, inputIds);
  const result = postprocessTableQA(output, inputIds, tokenizer.value);

  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  answer.value = result;
}
</script>

<template>
  <a href="#results" class="skip-link">Skip to results</a>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="qa-input">
      <label for="table">Table data (CSV format)</label>
      <div class="table-input"><textarea id="table" rows="4" v-model="table" aria-label="Table data in CSV format"></textarea></div>
      <label for="question">Question</label>
      <input id="question" type="text" v-model="question" aria-label="Question about the table" class="labels-input" />
      <button class="run-btn" @click="handleAnswer" :disabled="!session || !tokenizer" aria-label="Get answer">Answer</button>
    </div>
    <div id="results" class="qa-answer" role="status" aria-live="polite" aria-atomic="true">
      <template v-if="answer">
        <div>{{ answer.answer }}</div>
        <div class="score">Confidence: {{ (answer.score * 100).toFixed(1) }}%</div>
      </template>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
</template>
`;
}

// ---- Image-to-Text Page ----

function emitImageToTextPage(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script setup${tsLang}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from '~/lib/inference.${le}';
import { preprocessImage } from '~/lib/preprocess.${le}';
import { postprocessImageToText } from '~/lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';

const session = shallowRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
const caption = ref('');
const imageUrl = ref${t ? '<string | null>' : ''}(null);
const status = ref('Loading model...');
const dragOver = ref(false);
const fileInput = ref${t ? '<HTMLInputElement | null>' : ''}(null);

onMounted(() => {
  createSession(MODEL_PATH).then((s) => {
    session.value = s;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => { status.value = 'Failed to load model'; console.error('Model load error:', e); });
});

async function processImage(file${t ? ': File' : ''}) {
  if (!file.type.startsWith('image/')) { status.value = 'Unsupported file type.'; return; }
  const url = URL.createObjectURL(file);
  imageUrl.value = url; caption.value = '';
  const img = new Image(); img.src = url;
  await new Promise((resolve) => { img.onload = resolve; });
  if (!session.value) { status.value = 'Model not loaded yet.'; return; }
  status.value = '${config.modelName} \\u00b7 Generating caption...';
  const start = performance.now();
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d')${t ? '!' : ''}; ctx.drawImage(img, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
  const output = await runInference(session.value, inputTensor);
  const text = postprocessImageToText(output);
  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  caption.value = text;
  URL.revokeObjectURL(url);
}

function handleDrop(e${t ? ': DragEvent' : ''}) { e.preventDefault(); dragOver.value = false; const f = e.dataTransfer?.files[0]; if (f) processImage(f); }
function handleFileChange(e${t ? ': Event' : ''}) { const f = (e.target as HTMLInputElement).files?.[0]; if (f) processImage(f); }
function reset() { imageUrl.value = null; caption.value = ''; if (fileInput.value) fileInput.value.value = ''; }
</script>

<template>
  <a href="#results" class="skip-link">Skip to results</a>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="container">
      <div>
        <div v-if="!imageUrl" :class="['drop-zone', { 'drag-over': dragOver }]" role="button" tabindex="0"
          aria-label="Drop an image here or click to browse for ${taskLabel.toLowerCase()}"
          @click="fileInput?.click()" @keydown.enter.space.prevent="fileInput?.click()"
          @dragover.prevent="dragOver = true" @dragleave="dragOver = false" @drop="handleDrop">
          <p>Drop an image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input ref="fileInput" type="file" accept="image/*" style="position:absolute;left:-9999px;width:1px;height:1px;opacity:0;" aria-hidden="true" tabindex="-1" @change="handleFileChange" />
        </div>
        <div v-else class="preview">
          <img :src="imageUrl" alt="Selected image for captioning" />
          <button class="change-btn" @click="reset">Choose another image</button>
        </div>
      </div>
      <div id="results" class="generation-output" role="status" aria-live="polite" aria-atomic="true">{{ caption || '(caption will appear here)' }}</div>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
</template>
`;
}

// ---- Visual Question Answering Page ----

function emitVQAPage(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script setup${tsLang}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from '~/lib/inference.${le}';
import { preprocessImage } from '~/lib/preprocess.${le}';
import { postprocessVQA } from '~/lib/postprocess.${le}';
import { loadTokenizer } from '~/lib/preprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

const session = shallowRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
const tokenizer = shallowRef${t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : ''}(null);
const answer = ref('');
const imageUrl = ref${t ? '<string | null>' : ''}(null);
const question = ref('What is in this image?');
const status = ref('Loading model...');
const dragOver = ref(false);
const fileInput = ref${t ? '<HTMLInputElement | null>' : ''}(null);
const loadedImg = shallowRef${t ? '<HTMLImageElement | null>' : ''}(null);

onMounted(() => {
  Promise.all([
    createSession(MODEL_PATH),
    loadTokenizer(TOKENIZER_PATH),
  ]).then(([s, tok]) => {
    session.value = s;
    tokenizer.value = tok;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => { status.value = 'Failed to load model'; console.error('Model load error:', e); });
});

function handleFile(file${t ? ': File' : ''}) {
  if (!file.type.startsWith('image/')) return;
  const url = URL.createObjectURL(file);
  imageUrl.value = url;
  const img = new Image();
  img.src = url;
  img.onload = () => { loadedImg.value = img; };
}

async function handleAsk() {
  if (!session.value || !tokenizer.value || !loadedImg.value || !question.value.trim()) return;
  status.value = '${config.modelName} \\u00b7 Processing...';
  const start = performance.now();
  const canvas = document.createElement('canvas');
  canvas.width = loadedImg.value.naturalWidth; canvas.height = loadedImg.value.naturalHeight;
  const ctx = canvas.getContext('2d')${t ? '!' : ''}; ctx.drawImage(loadedImg.value, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
  const output = await runInference(session.value, inputTensor);
  const result = postprocessVQA(output, tokenizer.value);
  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  answer.value = result;
}

function handleDrop(e${t ? ': DragEvent' : ''}) { e.preventDefault(); dragOver.value = false; const f = e.dataTransfer?.files[0]; if (f) handleFile(f); }
function handleFileChange(e${t ? ': Event' : ''}) { const f = (e.target as HTMLInputElement).files?.[0]; if (f) handleFile(f); }
function reset() { imageUrl.value = null; answer.value = ''; loadedImg.value = null; if (fileInput.value) fileInput.value.value = ''; }
</script>

<template>
  <a href="#results" class="skip-link">Skip to results</a>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="container">
      <div class="multimodal-input">
        <div v-if="!imageUrl" :class="['drop-zone', { 'drag-over': dragOver }]" role="button" tabindex="0"
          aria-label="Drop an image here or click to browse"
          @click="fileInput?.click()" @keydown.enter.space.prevent="fileInput?.click()"
          @dragover.prevent="dragOver = true" @dragleave="dragOver = false" @drop="handleDrop">
          <p>Drop an image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input ref="fileInput" type="file" accept="image/*" style="position:absolute;left:-9999px;width:1px;height:1px;opacity:0;" aria-hidden="true" tabindex="-1" @change="handleFileChange" />
        </div>
        <div v-else class="preview">
          <img :src="imageUrl" alt="Selected image" />
          <button class="change-btn" @click="reset">Choose another image</button>
        </div>
        <input type="text" v-model="question" placeholder="Ask a question..." aria-label="Question about the image" class="question-input" />
        <button class="run-btn" @click="handleAsk" aria-label="Ask question">Ask</button>
      </div>
      <div id="results" class="qa-answer" role="status" aria-live="polite" aria-atomic="true">{{ answer }}</div>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
</template>
`;
}

// ---- Document Question Answering Page ----

function emitDocQAPage(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script setup${tsLang}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from '~/lib/inference.${le}';
import { preprocessImage } from '~/lib/preprocess.${le}';
import { postprocessDocQA } from '~/lib/postprocess.${le}';
import { loadTokenizer } from '~/lib/preprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

const session = shallowRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
const tokenizer = shallowRef${t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : ''}(null);
const answer = ref('');
const imageUrl = ref${t ? '<string | null>' : ''}(null);
const question = ref('What is the total amount?');
const status = ref('Loading model...');
const dragOver = ref(false);
const fileInput = ref${t ? '<HTMLInputElement | null>' : ''}(null);
const loadedImg = shallowRef${t ? '<HTMLImageElement | null>' : ''}(null);

onMounted(() => {
  Promise.all([
    createSession(MODEL_PATH),
    loadTokenizer(TOKENIZER_PATH),
  ]).then(([s, tok]) => {
    session.value = s;
    tokenizer.value = tok;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => { status.value = 'Failed to load model'; console.error('Model load error:', e); });
});

function handleFile(file${t ? ': File' : ''}) {
  if (!file.type.startsWith('image/')) return;
  const url = URL.createObjectURL(file);
  imageUrl.value = url;
  const img = new Image();
  img.src = url;
  img.onload = () => { loadedImg.value = img; };
}

async function handleAsk() {
  if (!session.value || !tokenizer.value || !loadedImg.value || !question.value.trim()) return;
  status.value = '${config.modelName} \\u00b7 Processing...';
  const start = performance.now();
  const canvas = document.createElement('canvas');
  canvas.width = loadedImg.value.naturalWidth; canvas.height = loadedImg.value.naturalHeight;
  const ctx = canvas.getContext('2d')${t ? '!' : ''}; ctx.drawImage(loadedImg.value, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
  const output = await runInference(session.value, inputTensor);
  const result = postprocessDocQA(output, tokenizer.value);
  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  answer.value = result;
}

function handleDrop(e${t ? ': DragEvent' : ''}) { e.preventDefault(); dragOver.value = false; const f = e.dataTransfer?.files[0]; if (f) handleFile(f); }
function handleFileChange(e${t ? ': Event' : ''}) { const f = (e.target as HTMLInputElement).files?.[0]; if (f) handleFile(f); }
function reset() { imageUrl.value = null; answer.value = ''; loadedImg.value = null; if (fileInput.value) fileInput.value.value = ''; }
</script>

<template>
  <a href="#results" class="skip-link">Skip to results</a>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="container">
      <div class="multimodal-input">
        <div v-if="!imageUrl" :class="['drop-zone', { 'drag-over': dragOver }]" role="button" tabindex="0"
          aria-label="Drop a document image here or click to browse"
          @click="fileInput?.click()" @keydown.enter.space.prevent="fileInput?.click()"
          @dragover.prevent="dragOver = true" @dragleave="dragOver = false" @drop="handleDrop">
          <p>Drop a document image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input ref="fileInput" type="file" accept="image/*" style="position:absolute;left:-9999px;width:1px;height:1px;opacity:0;" aria-hidden="true" tabindex="-1" @change="handleFileChange" />
        </div>
        <div v-else class="preview">
          <img :src="imageUrl" alt="Document image" />
          <button class="change-btn" @click="reset">Choose another image</button>
        </div>
        <input type="text" v-model="question" placeholder="Ask about the document..." aria-label="Question about the document" class="question-input" />
        <button class="run-btn" @click="handleAsk" aria-label="Ask question">Ask</button>
      </div>
      <div id="results" class="qa-answer" role="status" aria-live="polite" aria-atomic="true">{{ answer }}</div>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
</template>
`;
}

// ---- Image-Text-to-Text Page ----

function emitImageTextToTextPage(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script setup${tsLang}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from '~/lib/inference.${le}';
import { preprocessImage } from '~/lib/preprocess.${le}';
import { postprocessImageTextToText } from '~/lib/postprocess.${le}';
import { loadTokenizer } from '~/lib/preprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

const session = shallowRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
const tokenizer = shallowRef${t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : ''}(null);
const output = ref('');
const imageUrl = ref${t ? '<string | null>' : ''}(null);
const prompt = ref('Describe this image in detail.');
const status = ref('Loading model...');
const dragOver = ref(false);
const processing = ref(false);
const fileInput = ref${t ? '<HTMLInputElement | null>' : ''}(null);
const loadedImg = shallowRef${t ? '<HTMLImageElement | null>' : ''}(null);

onMounted(() => {
  Promise.all([
    createSession(MODEL_PATH),
    loadTokenizer(TOKENIZER_PATH),
  ]).then(([s, tok]) => {
    session.value = s;
    tokenizer.value = tok;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => { status.value = 'Failed to load model'; console.error('Model load error:', e); });
});

function handleFile(file${t ? ': File' : ''}) {
  if (!file.type.startsWith('image/')) return;
  const url = URL.createObjectURL(file);
  imageUrl.value = url;
  const img = new Image();
  img.src = url;
  img.onload = () => { loadedImg.value = img; };
}

async function handleGenerate() {
  if (!session.value || !tokenizer.value || !loadedImg.value || processing.value) return;
  processing.value = true;
  status.value = '${config.modelName} \\u00b7 Generating...';
  const start = performance.now();
  const canvas = document.createElement('canvas');
  canvas.width = loadedImg.value.naturalWidth; canvas.height = loadedImg.value.naturalHeight;
  const ctx = canvas.getContext('2d')${t ? '!' : ''}; ctx.drawImage(loadedImg.value, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
  const raw = await runInference(session.value, inputTensor);
  const result = postprocessImageTextToText(raw, tokenizer.value);
  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  output.value = result;
  processing.value = false;
}

function handleDrop(e${t ? ': DragEvent' : ''}) { e.preventDefault(); dragOver.value = false; const f = e.dataTransfer?.files[0]; if (f) handleFile(f); }
function handleFileChange(e${t ? ': Event' : ''}) { const f = (e.target as HTMLInputElement).files?.[0]; if (f) handleFile(f); }
function reset() { imageUrl.value = null; output.value = ''; loadedImg.value = null; if (fileInput.value) fileInput.value.value = ''; }
</script>

<template>
  <a href="#results" class="skip-link">Skip to results</a>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="container">
      <div class="multimodal-input">
        <div v-if="!imageUrl" :class="['drop-zone', { 'drag-over': dragOver }]" role="button" tabindex="0"
          aria-label="Drop an image here or click to browse"
          @click="fileInput?.click()" @keydown.enter.space.prevent="fileInput?.click()"
          @dragover.prevent="dragOver = true" @dragleave="dragOver = false" @drop="handleDrop">
          <p>Drop an image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input ref="fileInput" type="file" accept="image/*" style="position:absolute;left:-9999px;width:1px;height:1px;opacity:0;" aria-hidden="true" tabindex="-1" @change="handleFileChange" />
        </div>
        <div v-else class="preview">
          <img :src="imageUrl" alt="Selected image" />
          <button class="change-btn" @click="reset">Choose another image</button>
        </div>
        <input type="text" v-model="prompt" placeholder="Enter a prompt..." aria-label="Prompt for image description" class="question-input" />
        <button class="run-btn" @click="handleGenerate" :disabled="processing" aria-label="Generate text">{{ processing ? 'Generating...' : 'Generate' }}</button>
      </div>
      <div id="results" class="generation-output" role="status" aria-live="polite" aria-atomic="true">{{ output || '(output will appear here)' }}</div>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
</template>
`;
}

// ---- Realtime (Camera/Screen) Page ----

function emitRealtimePage(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';
  const isScreen = config.input === 'screen';
  const startFn = isScreen ? 'startScreenCapture' : 'startCamera';
  const label = isScreen ? 'Screen Capture' : 'Camera';
  const actionLabel = isScreen ? 'capture your screen' : 'use your camera';
  const btnLabel = isScreen ? 'Start Screen Capture' : 'Enable Camera';

  let postImport: string;
  let extraConst = '';
  let processAndRender: string;

  switch (config.task) {
    case 'object-detection': {
      const outputShape = config.modelMeta.outputs[0]?.shape ?? [1, 84, 8400];
      const numAttributes = outputShape[1] ?? 84;
      const numAnchors = outputShape[2] ?? 8400;
      postImport = `import { postprocessDetections } from '~/lib/postprocess.${le}';`;
      extraConst = `\nconst NUM_ATTRIBUTES = ${numAttributes};\nconst NUM_ANCHORS = ${numAnchors};\nconst COLORS = [[56,189,248],[249,115,22],[34,197,94],[168,85,247],[251,191,36],[239,68,68],[20,184,166],[236,72,153],[99,102,241],[163,230,53]];`;
      processAndRender = `const boxes = postprocessDetections(output, NUM_ANCHORS, NUM_ATTRIBUTES);
        const elapsed = performance.now() - start;
        const modelSize = ${config.preprocess.imageSize};
        const scaleX = imageData.width / modelSize;
        const scaleY = imageData.height / modelSize;
        overlayCtx.clearRect(0, 0, overlayEl.width, overlayEl.height);
        for (const box of boxes) {
          const c = COLORS[box.classIndex % COLORS.length];
          const color = \\\`rgb(\\\${c[0]},\\\${c[1]},\\\${c[2]})\\\`;
          overlayCtx.strokeStyle = color; overlayCtx.lineWidth = 2;
          overlayCtx.strokeRect(box.x * scaleX, box.y * scaleY, box.width * scaleX, box.height * scaleY);
          const lbl = 'Class ' + box.classIndex + ' (' + (box.score * 100).toFixed(0) + '%)';
          overlayCtx.font = '14px system-ui, sans-serif'; overlayCtx.fillStyle = color;
          const tw = overlayCtx.measureText(lbl).width;
          overlayCtx.fillRect(box.x * scaleX, box.y * scaleY - 20, tw + 8, 20);
          overlayCtx.fillStyle = '#fff'; overlayCtx.fillText(lbl, box.x * scaleX + 4, box.y * scaleY - 5);
        }`;
      break;
    }
    case 'image-segmentation': {
      const outputShape = config.modelMeta.outputs[0]?.shape ?? [1, 21, 512, 512];
      const numClasses = outputShape[1] ?? 21;
      const maskH = outputShape[2] ?? 512;
      const maskW = outputShape[3] ?? 512;
      postImport = `import { postprocessSegmentation } from '~/lib/postprocess.${le}';`;
      extraConst = `\nconst NUM_CLASSES = ${numClasses};\nconst MASK_H = ${maskH};\nconst MASK_W = ${maskW};\nconst COLORS = [[56,189,248],[249,115,22],[34,197,94],[168,85,247],[251,191,36],[239,68,68],[20,184,166],[236,72,153],[99,102,241],[163,230,53]];`;
      processAndRender = `const mask = postprocessSegmentation(output, NUM_CLASSES, MASK_H, MASK_W);
        const elapsed = performance.now() - start;
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
        overlayCtx.clearRect(0, 0, overlayEl.width, overlayEl.height);
        overlayCtx.drawImage(maskCanvas, 0, 0, videoEl.videoWidth, videoEl.videoHeight);`;
      break;
    }
    default: {
      postImport = `import { postprocessResults } from '~/lib/postprocess.${le}';`;
      processAndRender = `const results = postprocessResults(output);
        const elapsed = performance.now() - start;
        const lbl = 'Class ' + results.indices[0] + ' (' + (results.values[0] * 100).toFixed(1) + '%)';
        overlayCtx.clearRect(0, 0, overlayEl.width, overlayEl.height);
        overlayCtx.font = 'bold 24px system-ui, sans-serif';
        overlayCtx.fillStyle = 'rgba(59, 130, 246, 0.85)';
        const tw = overlayCtx.measureText(lbl).width;
        overlayCtx.fillRect(8, 8, tw + 16, 36);
        overlayCtx.fillStyle = '#fff';
        overlayCtx.fillText(lbl, 16, 34);`;
      break;
    }
  }

  return `<script setup${tsLang}>
import { ref, shallowRef, onMounted, onUnmounted } from 'vue';
import { createSession, runInference, getBackendLabel } from '~/lib/inference.${le}';
import { preprocessImage } from '~/lib/preprocess.${le}';
${postImport}
import { ${startFn}, captureFrame, createInferenceLoop } from '~/lib/input.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
${extraConst}

const session = shallowRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
const status = ref('Loading model...');
const started = ref(false);
const video = ref${t ? '<HTMLVideoElement | null>' : ''}(null);
const overlay = ref${t ? '<HTMLCanvasElement | null>' : ''}(null);
let loop${t ? ': ReturnType<typeof createInferenceLoop> | null' : ''} = null;

onMounted(() => {
  createSession(MODEL_PATH).then((s) => {
    session.value = s;
    status.value = '${config.modelName} \\u00b7 Ready \\u00b7 Tap Start';
  }).catch((e) => { status.value = 'Failed to load model'; console.error('Model load error:', e); });
});

onUnmounted(() => { if (loop) loop.stop(); });

async function handleStart() {
  const videoEl = video.value;
  const overlayEl = overlay.value;
  if (!videoEl || !overlayEl || !session.value) return;
  try {
    await ${startFn}(videoEl);
    overlayEl.width = videoEl.videoWidth;
    overlayEl.height = videoEl.videoHeight;
    started.value = true;

    const overlayCtx = overlayEl.getContext('2d')${t ? '!' : ''};
    loop = createInferenceLoop({
      video: videoEl,
      canvas: overlayEl,
      async onFrame(imageData${t ? ': ImageData' : ''}) {
        const start = performance.now();
        const inputTensor = preprocessImage(imageData.data, imageData.width, imageData.height);
        const output = await runInference(session.value${t ? '!' : ''}, inputTensor);
        ${processAndRender}
        return { result: null, elapsed };
      },
      onStatus(elapsed${t ? ': number' : ''}) {
        status.value = '${config.modelName} \\u00b7 ' + elapsed.toFixed(1) + 'ms \\u00b7 ' + getBackendLabel(session.value${t ? '!' : ''});
      },
    });
    loop.start();
  } catch (e) {
    status.value = '${label} access denied';
    console.error('${label} error:', e);
  }
}

function handlePause() {
  if (loop) { loop.stop(); loop = null; }
}
</script>

<template>
  <a href="#results" class="skip-link">Skip to results</a>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div v-if="!started" class="permission-prompt">
      <p>This app ${actionLabel} to run ${taskLabel.toLowerCase()} in real time.</p>
      <p class="hint">No video is recorded or sent anywhere.</p>
      <button class="primary-btn" @click="handleStart">${btnLabel}</button>
    </div>
    <div v-else>
      <div class="video-wrapper">
        <video ref="video" autoplay playsinline muted></video>
        <canvas ref="overlay"></canvas>
      </div>
      <div class="controls">
        <button class="control-btn" @click="handlePause">\\u23f8 Pause</button>
      </div>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
</template>
`;
}

// ---- Page dispatcher ----

function emitPageVue(config: ResolvedConfig): string {
  switch (config.task) {
    case 'text-to-speech': return emitTextToSpeechPage(config);
    case 'audio-classification':
      return config.input === 'mic' ? emitMicAudioClassificationPage(config) : emitFileAudioClassificationPage(config);
    case 'speech-to-text':
      return config.input === 'mic' ? emitMicSpeechToTextPage(config) : emitFileSpeechToTextPage(config);
    case 'audio-to-audio': return emitAudioToAudioPage(config);
    case 'speaker-diarization': return emitSpeakerDiarizationPage(config);
    case 'voice-activity-detection': return emitVADPage(config);
    case 'text-classification': return emitTextClassificationPage(config);
    case 'zero-shot-classification': return emitZeroShotClassificationPage(config);
    case 'text-generation': return emitTextGenerationPage(config);
    case 'fill-mask': return emitFillMaskPage(config);
    case 'sentence-similarity': return emitSentenceSimilarityPage(config);
    case 'token-classification': return emitTokenClassificationPage(config);
    case 'question-answering': return emitQuestionAnsweringPage(config);
    case 'summarization': return emitSummarizationPage(config);
    case 'translation': return emitTranslationPage(config);
    case 'text2text-generation': return emitText2TextPage(config);
    case 'conversational': return emitConversationalPage(config);
    case 'table-question-answering': return emitTableQAPage(config);
    case 'image-to-text': return emitImageToTextPage(config);
    case 'visual-question-answering': return emitVQAPage(config);
    case 'document-question-answering': return emitDocQAPage(config);
    case 'image-text-to-text': return emitImageTextToTextPage(config);
    case 'depth-estimation': return emitDepthEstimationPage(config);
    case 'object-detection':
      return (config.input === 'camera' || config.input === 'screen') ? emitRealtimePage(config) : emitFileDetectionPage(config);
    case 'image-segmentation':
      return (config.input === 'camera' || config.input === 'screen') ? emitRealtimePage(config) : emitFileSegmentationPage(config);
    case 'feature-extraction': return emitFileFeatureExtractionPage(config);
    case 'image-classification':
    default:
      return (config.input === 'camera' || config.input === 'screen') ? emitRealtimePage(config) : emitFileClassificationPage(config);
  }
}

/** Wrap a CodeBlock's code with exports for use as a lib module */
function toLibModule(block: CodeBlock | undefined): string {
  if (!block) return '';
  return addExports(block.code, block.exports);
}

/**
 * Emit Nuxt 3 framework files.
 */
export function emitNuxt(config: ResolvedConfig, blocks: CodeBlock[]): GeneratedFile[] {
  const le = libExt(config);

  const inputBlock = findBlock(blocks, 'input');
  const preprocessBlock = findBlock(blocks, 'preprocess');
  const inferenceBlock = findBlock(blocks, 'inference');
  const postprocessBlock = findBlock(blocks, 'postprocess');
  const opfsBlock = findBlock(blocks, 'opfs-cache');

  const filePaths: string[] = [
    'package.json',
    'nuxt.config.ts',
    'app.vue',
    'assets/app.css',
    'pages/index.vue',
  ];

  if (inputBlock?.code) {
    filePaths.push(`lib/input.${le}`);
  }

  filePaths.push(
    `lib/preprocess.${le}`,
    `lib/inference.${le}`,
    `lib/postprocess.${le}`,
    'README.md',
  );

  // Prepend OPFS cache utilities to inference module when offline mode is enabled
  const inferenceContent = opfsBlock?.code
    ? `${opfsBlock.code}\n\n${toLibModule(inferenceBlock)}`
    : toLibModule(inferenceBlock);

  const files: GeneratedFile[] = [
    { path: 'package.json', content: emitPackageJson(config, blocks) },
    { path: 'nuxt.config.ts', content: emitNuxtConfig() },
    { path: 'app.vue', content: emitAppVue() },
    { path: 'assets/app.css', content: emitAssetsCss(config) },
    { path: 'pages/index.vue', content: emitPageVue(config) },
  ];

  if (inputBlock?.code) {
    files.push({ path: `lib/input.${le}`, content: toLibModule(inputBlock) });
  }

  files.push(
    { path: `lib/preprocess.${le}`, content: toLibModule(preprocessBlock) },
    { path: `lib/inference.${le}`, content: inferenceContent },
    { path: `lib/postprocess.${le}`, content: toLibModule(postprocessBlock) },
    { path: 'README.md', content: emitReadme(config, filePaths) },
  );

  // Include auxiliary files from Layer 1 blocks (e.g. AudioWorklet processor)
  files.push(...collectAuxiliaryFiles(blocks));

  return files;
}
