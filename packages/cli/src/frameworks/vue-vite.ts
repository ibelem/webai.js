/**
 * Vue-Vite framework emitter (Layer 2).
 *
 * Produces a full Vue 3 + Vite project:
 *   package.json, vite.config.js, index.html, src/main, src/App.vue, src/App.css,
 *   src/lib/preprocess, src/lib/inference, src/lib/postprocess, README.md
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
const mainExt = (config: ResolvedConfig) => (config.lang === 'ts' ? 'ts' : 'js');
const scriptLang = (config: ResolvedConfig) => (config.lang === 'ts' ? ' lang="ts"' : '');

function emitPackageJson(config: ResolvedConfig, blocks: CodeBlock[]): string {
  const deps: Record<string, string> = {
    vue: '^3.5.0',
  };
  for (const imp of collectImports(blocks)) {
    if (imp === 'onnxruntime-web') deps[imp] = '^1.21.0';
    else deps[imp] = 'latest';
  }

  const devDeps: Record<string, string> = {
    '@vitejs/plugin-vue': '^5.0.0',
    vite: '^6.0.0',
  };
  if (config.lang === 'ts') {
    devDeps['typescript'] = '^5.7.0';
    devDeps['vue-tsc'] = '^2.0.0';
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
import vue from '@vitejs/plugin-vue';

export default defineConfig({
  plugins: [vue()],
});
`;
}

function emitIndexHtml(config: ResolvedConfig): string {
  const taskLabel = getTaskLabel(config.task);
  return `<!DOCTYPE html>
<html lang="en" data-theme="${config.theme}">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>${config.modelName} — ${taskLabel}</title>
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="/src/main.${mainExt(config)}"></script>
  </body>
</html>
`;
}

function emitMain(_config: ResolvedConfig): string {
  return `import { createApp } from 'vue';
import App from './App.vue';
import './App.css';

createApp(App).mount('#app');
`;
}

function emitApp(config: ResolvedConfig): string {
  switch (config.task) {
    case 'text-to-speech': return emitTextToSpeechApp(config);
    case 'audio-classification':
      return config.input === 'mic' ? emitMicAudioClassificationApp(config) : emitFileAudioClassificationApp(config);
    case 'speech-to-text':
      return config.input === 'mic' ? emitMicSpeechToTextApp(config) : emitFileSpeechToTextApp(config);
    case 'audio-to-audio': return emitAudioToAudioApp(config);
    case 'speaker-diarization': return emitSpeakerDiarizationApp(config);
    case 'voice-activity-detection': return emitVADApp(config);
    case 'text-classification': return emitTextClassificationApp(config);
    case 'zero-shot-classification': return emitZeroShotClassificationApp(config);
    case 'text-generation': return emitTextGenerationApp(config);
    case 'fill-mask': return emitFillMaskApp(config);
    case 'sentence-similarity': return emitSentenceSimilarityApp(config);
    case 'token-classification': return emitTokenClassificationApp(config);
    case 'question-answering': return emitQuestionAnsweringApp(config);
    case 'summarization': return emitSummarizationApp(config);
    case 'translation': return emitTranslationApp(config);
    case 'text2text-generation': return emitText2TextApp(config);
    case 'conversational': return emitConversationalApp(config);
    case 'table-question-answering': return emitTableQAApp(config);
    case 'image-to-text': return emitImageToTextApp(config);
    case 'visual-question-answering': return emitVQAApp(config);
    case 'document-question-answering': return emitDocQAApp(config);
    case 'image-text-to-text': return emitImageTextToTextApp(config);
    case 'depth-estimation': return emitDepthEstimationApp(config);
    default: break;
  }
  if (config.input === 'camera' || config.input === 'screen') {
    return emitRealtimeApp(config);
  }
  if (config.task === 'object-detection') return emitFileDetectionApp(config);
  if (config.task === 'image-segmentation') return emitFileSegmentationApp(config);
  if (config.task === 'feature-extraction') return emitFileFeatureExtractionApp(config);
  return emitFileClassificationApp(config);
}

// ---- Image Classification (file) ----

function emitFileClassificationApp(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const sl = scriptLang(config);

  const resultsType = t ? '<{ indices: number[]; values: number[] } | null>' : '';
  const stringNullType = t ? '<string | null>' : '';
  const sessionType = t ? '<Awaited<ReturnType<typeof createSession>> | null>' : '';
  const inputRefType = t ? '<HTMLInputElement | null>' : '';

  return `<script setup${sl}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { preprocessImage } from './lib/preprocess.${le}';
import { postprocessResults } from './lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';

const results = ref${resultsType}(null);
const status = ref('Loading model...');
const imageUrl = ref${stringNullType}(null);
const dragOver = ref(false);
const session = shallowRef${sessionType}(null);
const fileInputRef = ref${inputRefType}(null);

onMounted(() => {
  createSession(MODEL_PATH).then((s) => {
    session.value = s;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => {
    status.value = 'Failed to load model';
    console.error('Model load error:', e);
  });
});

async function processImage(file${t ? ': File' : ''}) {
  if (!file.type.startsWith('image/')) {
    status.value = 'Unsupported file type. Try JPG, PNG, or WebP.';
    return;
  }

  const url = URL.createObjectURL(file);
  imageUrl.value = url;
  results.value = null;

  const img = new Image();
  img.src = url;
  await new Promise((resolve) => { img.onload = resolve; });

  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d')${t ? '!' : ''};
  ctx.drawImage(img, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  if (!session.value) {
    status.value = 'Model not loaded yet. Please wait.';
    return;
  }

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
  e.preventDefault();
  dragOver.value = false;
  const file = e.dataTransfer${t ? '!' : ''}.files[0];
  if (file) processImage(file);
}

function handleFileChange(e${t ? ': Event' : ''}) {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (file) processImage(file);
}

function reset() {
  imageUrl.value = null;
  results.value = null;
  if (fileInputRef.value) fileInputRef.value.value = '';
}
</script>

<template>
  <a href="#results" class="skip-link">Skip to results</a>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="container">
      <div>
        <div
          v-if="!imageUrl"
          :class="['drop-zone', { 'drag-over': dragOver }]"
          role="button"
          tabindex="0"
          aria-label="Drop an image here or click to browse for ${taskLabel.toLowerCase()}"
          @click="fileInputRef?.click()"
          @keydown="(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInputRef?.click(); } }"
          @dragover.prevent="dragOver = true"
          @dragleave="dragOver = false"
          @drop="handleDrop"
        >
          <p>Drop an image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input ref="fileInputRef" type="file" accept="image/*" hidden @change="handleFileChange" />
        </div>
        <div v-else class="preview">
          <img :src="imageUrl" alt="Selected image for classification" />
          <button class="change-btn" @click="reset">Choose another image</button>
        </div>
      </div>
      <div id="results" class="results" role="status" aria-live="polite" aria-atomic="true">
        <template v-if="results">
          <template v-for="(idx, i) in results.indices" :key="idx">
            <div
              v-if="results.values[i] >= 0.01"
              :class="['result-row', { 'top-result': i === 0 }]"
              tabindex="0"
              :aria-label="\`Class \${idx}: \${(results.values[i] * 100).toFixed(1)} percent\`"
            >
              <span class="result-label">Class {{ idx }}</span>
              <div class="result-bar-container">
                <div class="result-bar" :style="{ width: \`\${(results.values[i] / (results.values[0] || 1)) * 100}%\` }" />
              </div>
              <span class="result-pct">{{ (results.values[i] * 100).toFixed(1) }}%</span>
            </div>
          </template>
        </template>
      </div>
    </div>
  </main>
  <aside class="status-bar">
    <span>{{ status }}</span>
  </aside>
  <div class="footer">Generated by webai.js \u00b7 ${config.modelName} \u00b7 ${engineLabel}</div>
</template>
`;
}

// ---- Object Detection (file) ----

function emitFileDetectionApp(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const outputShape = config.modelMeta.outputs[0]?.shape ?? [1, 84, 8400];
  const numAttributes = outputShape[1] ?? 84;
  const numAnchors = outputShape[2] ?? 8400;
  const sl = scriptLang(config);

  const boxInterface = t ? `\ninterface Box { x: number; y: number; width: number; height: number; classIndex: number; score: number; }\n` : '';
  const boxesType = t ? '<Box[] | null>' : '';
  const stringNullType = t ? '<string | null>' : '';
  const sessionType = t ? '<Awaited<ReturnType<typeof createSession>> | null>' : '';
  const inputRefType = t ? '<HTMLInputElement | null>' : '';
  const canvasRefType = t ? '<HTMLCanvasElement | null>' : '';
  const imgSizeType = t ? '<{ w: number; h: number }>' : '';

  return `<script setup${sl}>
import { ref, shallowRef, onMounted, watch } from 'vue';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { preprocessImage } from './lib/preprocess.${le}';
import { postprocessDetections } from './lib/postprocess.${le}';
${boxInterface}
const MODEL_PATH = '${getModelPath(config, '')}';
const NUM_ATTRIBUTES = ${numAttributes};
const NUM_ANCHORS = ${numAnchors};
const COLORS = [[56,189,248],[249,115,22],[34,197,94],[168,85,247],[251,191,36],[239,68,68],[20,184,166],[236,72,153],[99,102,241],[163,230,53]];

const boxes = ref${boxesType}(null);
const status = ref('Loading model...');
const imageUrl = ref${stringNullType}(null);
const dragOver = ref(false);
const imgSize = ref${imgSizeType}({ w: 0, h: 0 });
const session = shallowRef${sessionType}(null);
const fileInputRef = ref${inputRefType}(null);
const overlayRef = ref${canvasRefType}(null);

onMounted(() => {
  createSession(MODEL_PATH).then((s) => {
    session.value = s;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => { status.value = 'Failed to load model'; console.error('Model load error:', e); });
});

watch([boxes, imgSize], () => {
  if (!boxes.value || !overlayRef.value) return;
  const canvas = overlayRef.value;
  canvas.width = imgSize.value.w;
  canvas.height = imgSize.value.h;
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

function handleDrop(e${t ? ': DragEvent' : ''}) { e.preventDefault(); dragOver.value = false; const f = e.dataTransfer${t ? '!' : ''}.files[0]; if (f) processImage(f); }
function handleFileChange(e${t ? ': Event' : ''}) { const f = (e.target as HTMLInputElement).files?.[0]; if (f) processImage(f); }
function reset() { imageUrl.value = null; boxes.value = null; if (fileInputRef.value) fileInputRef.value.value = ''; }
</script>

<template>
  <a href="#results" class="skip-link">Skip to results</a>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="container">
      <div>
        <div
          v-if="!imageUrl"
          :class="['drop-zone', { 'drag-over': dragOver }]"
          role="button"
          tabindex="0"
          aria-label="Drop an image here or click to browse for ${taskLabel.toLowerCase()}"
          @click="fileInputRef?.click()"
          @keydown="(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInputRef?.click(); } }"
          @dragover.prevent="dragOver = true"
          @dragleave="dragOver = false"
          @drop="handleDrop"
        >
          <p>Drop an image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input ref="fileInputRef" type="file" accept="image/*" hidden @change="handleFileChange" />
        </div>
        <div v-else class="preview">
          <div class="preview-wrapper">
            <img :src="imageUrl" alt="Selected image for detection" />
            <canvas ref="overlayRef" class="overlay-canvas" />
          </div>
          <button class="change-btn" @click="reset">Choose another image</button>
        </div>
      </div>
      <div id="results" class="results" role="status" aria-live="polite" aria-atomic="true">
        <p v-if="boxes && boxes.length === 0">No detections found.</p>
        <div
          v-for="(box, i) in boxes"
          :key="i"
          class="result-row"
          tabindex="0"
          :aria-label="\`Class \${box.classIndex}: \${(box.score * 100).toFixed(0)} percent\`"
        >
          <span class="result-label">Class {{ box.classIndex }} ({{ (box.score * 100).toFixed(0) }}%)</span>
        </div>
      </div>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js \u00b7 ${config.modelName} \u00b7 ${engineLabel}</div>
</template>
`;
}

// ---- Image Segmentation (file) ----

function emitFileSegmentationApp(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const outputShape = config.modelMeta.outputs[0]?.shape ?? [1, 21, 512, 512];
  const numClasses = outputShape[1] ?? 21;
  const maskH = outputShape[2] ?? 512;
  const maskW = outputShape[3] ?? 512;
  const sl = scriptLang(config);

  const maskType = t ? '<Uint8Array | null>' : '';
  const stringNullType = t ? '<string | null>' : '';
  const sessionType = t ? '<Awaited<ReturnType<typeof createSession>> | null>' : '';
  const inputRefType = t ? '<HTMLInputElement | null>' : '';
  const canvasRefType = t ? '<HTMLCanvasElement | null>' : '';
  const imgSizeType = t ? '<{ w: number; h: number }>' : '';

  return `<script setup${sl}>
import { ref, shallowRef, onMounted, watch, computed } from 'vue';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { preprocessImage } from './lib/preprocess.${le}';
import { postprocessSegmentation } from './lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
const NUM_CLASSES = ${numClasses};
const MASK_H = ${maskH};
const MASK_W = ${maskW};
const COLORS = [[56,189,248],[249,115,22],[34,197,94],[168,85,247],[251,191,36],[239,68,68],[20,184,166],[236,72,153],[99,102,241],[163,230,53]];

const mask = ref${maskType}(null);
const status = ref('Loading model...');
const imageUrl = ref${stringNullType}(null);
const dragOver = ref(false);
const imgSize = ref${imgSizeType}({ w: 0, h: 0 });
const session = shallowRef${sessionType}(null);
const fileInputRef = ref${inputRefType}(null);
const overlayRef = ref${canvasRefType}(null);

const classCount = computed(() => mask.value ? new Set(Array.from(mask.value)).size : 0);

onMounted(() => {
  createSession(MODEL_PATH).then((s) => {
    session.value = s;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => { status.value = 'Failed to load model'; console.error('Model load error:', e); });
});

watch([mask, imgSize], () => {
  if (!mask.value || !overlayRef.value) return;
  const canvas = overlayRef.value;
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

function handleDrop(e${t ? ': DragEvent' : ''}) { e.preventDefault(); dragOver.value = false; const f = e.dataTransfer${t ? '!' : ''}.files[0]; if (f) processImage(f); }
function handleFileChange(e${t ? ': Event' : ''}) { const f = (e.target as HTMLInputElement).files?.[0]; if (f) processImage(f); }
function reset() { imageUrl.value = null; mask.value = null; if (fileInputRef.value) fileInputRef.value.value = ''; }
</script>

<template>
  <a href="#results" class="skip-link">Skip to results</a>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="container">
      <div>
        <div
          v-if="!imageUrl"
          :class="['drop-zone', { 'drag-over': dragOver }]"
          role="button"
          tabindex="0"
          aria-label="Drop an image here or click to browse for ${taskLabel.toLowerCase()}"
          @click="fileInputRef?.click()"
          @keydown="(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInputRef?.click(); } }"
          @dragover.prevent="dragOver = true"
          @dragleave="dragOver = false"
          @drop="handleDrop"
        >
          <p>Drop an image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input ref="fileInputRef" type="file" accept="image/*" hidden @change="handleFileChange" />
        </div>
        <div v-else class="preview">
          <div class="preview-wrapper">
            <img :src="imageUrl" alt="Selected image for segmentation" />
            <canvas ref="overlayRef" class="overlay-canvas" />
          </div>
          <button class="change-btn" @click="reset">Choose another image</button>
        </div>
      </div>
      <div id="results" class="results" role="status" aria-live="polite" aria-atomic="true">
        <p v-if="mask">{{ classCount }} classes detected</p>
      </div>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js \u00b7 ${config.modelName} \u00b7 ${engineLabel}</div>
</template>
`;
}

// ---- Feature Extraction (file) ----

function emitFileFeatureExtractionApp(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const sl = scriptLang(config);

  const embType = t ? '<Float32Array | null>' : '';
  const stringNullType = t ? '<string | null>' : '';
  const sessionType = t ? '<Awaited<ReturnType<typeof createSession>> | null>' : '';
  const inputRefType = t ? '<HTMLInputElement | null>' : '';

  return `<script setup${sl}>
import { ref, shallowRef, onMounted, computed } from 'vue';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { preprocessImage } from './lib/preprocess.${le}';
import { postprocessEmbeddings } from './lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';

const embedding = ref${embType}(null);
const status = ref('Loading model...');
const imageUrl = ref${stringNullType}(null);
const dragOver = ref(false);
const session = shallowRef${sessionType}(null);
const fileInputRef = ref${inputRefType}(null);

const l2Norm = computed(() => embedding.value ? Math.sqrt(Array.from(embedding.value).reduce((s, v) => s + v * v, 0)).toFixed(4) : '');
const first5 = computed(() => embedding.value ? Array.from(embedding.value.slice(0, 5)).map(v => v.toFixed(4)).join(', ') : '');

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
  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  embedding.value = emb;
  URL.revokeObjectURL(url);
}

function handleDrop(e${t ? ': DragEvent' : ''}) { e.preventDefault(); dragOver.value = false; const f = e.dataTransfer${t ? '!' : ''}.files[0]; if (f) processImage(f); }
function handleFileChange(e${t ? ': Event' : ''}) { const f = (e.target as HTMLInputElement).files?.[0]; if (f) processImage(f); }
function reset() { imageUrl.value = null; embedding.value = null; if (fileInputRef.value) fileInputRef.value.value = ''; }
</script>

<template>
  <a href="#results" class="skip-link">Skip to results</a>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="container">
      <div>
        <div
          v-if="!imageUrl"
          :class="['drop-zone', { 'drag-over': dragOver }]"
          role="button"
          tabindex="0"
          aria-label="Drop an image here or click to browse for ${taskLabel.toLowerCase()}"
          @click="fileInputRef?.click()"
          @keydown="(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInputRef?.click(); } }"
          @dragover.prevent="dragOver = true"
          @dragleave="dragOver = false"
          @drop="handleDrop"
        >
          <p>Drop an image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input ref="fileInputRef" type="file" accept="image/*" hidden @change="handleFileChange" />
        </div>
        <div v-else class="preview">
          <img :src="imageUrl" alt="Selected image for feature extraction" />
          <button class="change-btn" @click="reset">Choose another image</button>
        </div>
      </div>
      <div id="results" class="results" role="status" aria-live="polite" aria-atomic="true">
        <div v-if="embedding" class="embedding-info">
          <p><strong>Dimensions:</strong> {{ embedding.length }}</p>
          <p><strong>L2 Norm:</strong> {{ l2Norm }}</p>
          <p><strong>First 5:</strong> [{{ first5 }}, ...]</p>
        </div>
      </div>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js \u00b7 ${config.modelName} \u00b7 ${engineLabel}</div>
</template>
`;
}

// ---- Audio: File + Classification ----

function emitFileAudioClassificationApp(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const sl = scriptLang(config);

  const resultsType = t ? '<{ indices: number[]; values: number[] } | null>' : '';
  const sessionType = t ? '<Awaited<ReturnType<typeof createSession>> | null>' : '';

  return `<script setup${sl}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { melSpectrogram, mfcc } from './lib/preprocess.${le}';
import { postprocessResults } from './lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';

const results = ref${resultsType}(null);
const status = ref('Loading model...');
const session = shallowRef${sessionType}(null);

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
          <div
            v-if="results.values[i] >= 0.01"
            :class="['result-row', { 'top-result': i === 0 }]"
            tabindex="0"
            :aria-label="\`Class \${idx}: \${(results.values[i] * 100).toFixed(1)} percent\`"
          >
            <span class="result-label">Class {{ idx }}</span>
            <div class="result-bar-container">
              <div class="result-bar" :style="{ width: \`\${(results.values[i] / (results.values[0] || 1)) * 100}%\` }" />
            </div>
            <span class="result-pct">{{ (results.values[i] * 100).toFixed(1) }}%</span>
          </div>
        </template>
      </template>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js \u00b7 ${config.modelName} \u00b7 ${engineLabel}</div>
</template>
`;
}

// ---- Audio: File + Speech-to-Text ----

function emitFileSpeechToTextApp(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const sl = scriptLang(config);

  const sessionType = t ? '<Awaited<ReturnType<typeof createSession>> | null>' : '';

  return `<script setup${sl}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { melSpectrogram } from './lib/preprocess.${le}';
import { postprocessTranscript } from './lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
const VOCAB = [' ', ...'abcdefghijklmnopqrstuvwxyz'.split(''), "'"];

const transcript = ref('');
const status = ref('Loading model...');
const session = shallowRef${sessionType}(null);

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
  <div class="footer">Generated by webai.js \u00b7 ${config.modelName} \u00b7 ${engineLabel}</div>
</template>
`;
}

// ---- Audio: Mic + Speech-to-Text ----

function emitMicSpeechToTextApp(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const sl = scriptLang(config);

  const sessionType = t ? '<Awaited<ReturnType<typeof createSession>> | null>' : '';
  const anyType = t ? '<any>' : '';

  return `<script setup${sl}>
import { ref, shallowRef, onMounted, onUnmounted } from 'vue';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { melSpectrogram } from './lib/preprocess.${le}';
import { postprocessTranscript } from './lib/postprocess.${le}';
import { startAudioCapture, stopStream, createAudioInferenceLoop } from './lib/input.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
const VOCAB = [' ', ...'abcdefghijklmnopqrstuvwxyz'.split(''), "'"];

const transcript = ref('(listening...)');
const status = ref('Loading model...');
const recording = ref(false);
const session = shallowRef${sessionType}(null);
const captureRef = shallowRef${anyType}(null);
const loopRef = shallowRef${anyType}(null);

onMounted(() => {
  createSession(MODEL_PATH).then((s) => {
    session.value = s;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => { status.value = 'Failed to load model'; console.error('Model load error:', e); });
});

onUnmounted(() => {
  if (loopRef.value) loopRef.value.stop();
  if (captureRef.value) { stopStream(captureRef.value.stream); captureRef.value.audioContext.close(); }
});

async function processAudio(samples${t ? ': Float32Array' : ''}) {
  const mel = melSpectrogram(samples, 16000, 512, 160, 80);
  const output = await runInference(session.value${t ? '!' : ''}, mel.data);
  const vocabSize = VOCAB.length + 1;
  const numTimesteps = Math.floor(output.length / vocabSize);
  return postprocessTranscript(output, numTimesteps, vocabSize, VOCAB);
}

async function handleStart() {
  if (!session.value) { status.value = 'Model not loaded yet.'; return; }
  try {
    captureRef.value = await startAudioCapture(16000);
    recording.value = true;
    status.value = '${config.modelName} \\u00b7 Listening...';
    loopRef.value = createAudioInferenceLoop({
      getSamples: captureRef.value.getSamples,
      onResult(text${t ? ': string' : ''}) { transcript.value = text || '(listening...)'; },
      intervalMs: 2000,
    });
    loopRef.value.start();
  } catch (e) { status.value = 'Microphone access denied'; console.error('Mic error:', e); }
}

function handleStop() {
  if (loopRef.value) { loopRef.value.stop(); loopRef.value = null; }
  if (captureRef.value) { stopStream(captureRef.value.stream); captureRef.value.audioContext.close(); captureRef.value = null; }
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
  <div class="footer">Generated by webai.js \u00b7 ${config.modelName} \u00b7 ${engineLabel}</div>
</template>
`;
}

// ---- Audio: Mic + Classification ----

function emitMicAudioClassificationApp(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const sl = scriptLang(config);

  const resultsType = t ? '<{ indices: number[]; values: number[] } | null>' : '';
  const sessionType = t ? '<Awaited<ReturnType<typeof createSession>> | null>' : '';
  const anyType = t ? '<any>' : '';

  return `<script setup${sl}>
import { ref, shallowRef, onMounted, onUnmounted } from 'vue';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { melSpectrogram, mfcc } from './lib/preprocess.${le}';
import { postprocessResults } from './lib/postprocess.${le}';
import { startAudioCapture, stopStream, createAudioInferenceLoop } from './lib/input.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';

const results = ref${resultsType}(null);
const status = ref('Loading model...');
const recording = ref(false);
const session = shallowRef${sessionType}(null);
const captureRef = shallowRef${anyType}(null);
const loopRef = shallowRef${anyType}(null);

onMounted(() => {
  createSession(MODEL_PATH).then((s) => {
    session.value = s;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => { status.value = 'Failed to load model'; console.error('Model load error:', e); });
});

onUnmounted(() => {
  if (loopRef.value) loopRef.value.stop();
  if (captureRef.value) { stopStream(captureRef.value.stream); captureRef.value.audioContext.close(); }
});

async function processAudio(samples${t ? ': Float32Array' : ''}) {
  const mel = melSpectrogram(samples, 16000, 512, 160, 40);
  const features = mfcc(mel.data, mel.numFrames, mel.numMelBands, 13);
  const output = await runInference(session.value${t ? '!' : ''}, features);
  return postprocessResults(output);
}

async function handleStart() {
  if (!session.value) { status.value = 'Model not loaded yet.'; return; }
  try {
    captureRef.value = await startAudioCapture(16000);
    recording.value = true;
    status.value = '${config.modelName} \\u00b7 Listening...';
    loopRef.value = createAudioInferenceLoop({
      getSamples: captureRef.value.getSamples,
      onResult(r${t ? ': { indices: number[]; values: number[] }' : ''}) { results.value = r; },
      intervalMs: 2000,
    });
    loopRef.value.start();
  } catch (e) { status.value = 'Microphone access denied'; console.error('Mic error:', e); }
}

function handleStop() {
  if (loopRef.value) { loopRef.value.stop(); loopRef.value = null; }
  if (captureRef.value) { stopStream(captureRef.value.stream); captureRef.value.audioContext.close(); captureRef.value = null; }
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
          <div
            v-if="results.values[i] >= 0.01"
            :class="['result-row', { 'top-result': i === 0 }]"
            tabindex="0"
            :aria-label="\`Class \${idx}: \${(results.values[i] * 100).toFixed(1)} percent\`"
          >
            <span class="result-label">Class {{ idx }}</span>
            <div class="result-bar-container">
              <div class="result-bar" :style="{ width: \`\${(results.values[i] / (results.values[0] || 1)) * 100}%\` }" />
            </div>
            <span class="result-pct">{{ (results.values[i] * 100).toFixed(1) }}%</span>
          </div>
        </template>
      </template>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js \u00b7 ${config.modelName} \u00b7 ${engineLabel}</div>
</template>
`;
}

// ---- Audio: Text-to-Speech ----

function emitTextToSpeechApp(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const sl = scriptLang(config);

  const sessionType = t ? '<Awaited<ReturnType<typeof createSession>> | null>' : '';

  return `<script setup${sl}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { postprocessAudio, playAudio } from './lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';

const status = ref('Loading model...');
const text = ref('Hello, this is a test of text to speech.');
const session = shallowRef${sessionType}(null);

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
      <textarea id="textInput" rows="4" v-model="text" aria-label="Text to synthesize" />
      <button class="primary-btn" @click="handleSynthesize" aria-label="Synthesize speech">Synthesize</button>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js \u00b7 ${config.modelName} \u00b7 ${engineLabel}</div>
</template>
`;
}

// ---- Text/NLP: Text Classification ----

function emitTextClassificationApp(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');
  const sl = scriptLang(config);

  const resultsType = t ? '<{ indices: number[]; values: number[] } | null>' : '';
  const sessionType = t ? '<Awaited<ReturnType<typeof createSession>> | null>' : '';
  const tokenizerType = t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : '';

  return `<script setup${sl}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { loadTokenizer, tokenizeText } from './lib/preprocess.${le}';
import { postprocessResults } from './lib/postprocess.${le}';

const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

const results = ref${resultsType}(null);
const status = ref('Loading model...');
const text = ref('This is an amazing product!');
const session = shallowRef${sessionType}(null);
const tokenizer = shallowRef${tokenizerType}(null);

onMounted(() => {
  Promise.all([
    createSession(MODEL_PATH),
    loadTokenizer(TOKENIZER_PATH)
  ]).then(([s, tok]) => {
    session.value = s;
    tokenizer.value = tok;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => {
    status.value = 'Failed to load model';
    console.error('Model load error:', e);
  });
});

async function handleClassify() {
  if (!session.value || !tokenizer.value || !text.value.trim()) return;

  status.value = '${config.modelName} \\u00b7 Processing...';
  const start = performance.now();

  const { inputIds, attentionMask } = tokenizeText(tokenizer.value, text.value, 128);
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
      <textarea id="textInput" rows="4" v-model="text" aria-label="Text to classify" />
      <button class="primary-btn" @click="handleClassify" aria-label="Classify text">Classify</button>
    </div>
    <div id="results" class="results" role="status" aria-live="polite" aria-atomic="true">
      <template v-if="results">
        <template v-for="(idx, i) in results.indices" :key="idx">
          <div
            v-if="results.values[i] >= 0.01"
            :class="['result-row', { 'top-result': i === 0 }]"
            tabindex="0"
            :aria-label="\`Class \${idx}: \${(results.values[i] * 100).toFixed(1)} percent\`"
          >
            <span class="result-label">Class {{ idx }}</span>
            <div class="result-bar-container">
              <div class="result-bar" :style="{ width: \`\${(results.values[i] / (results.values[0] || 1)) * 100}%\` }" />
            </div>
            <span class="result-pct">{{ (results.values[i] * 100).toFixed(1) }}%</span>
          </div>
        </template>
      </template>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js \u00b7 ${config.modelName} \u00b7 ${engineLabel}</div>
</template>
`;
}

// ---- Text/NLP: Zero-Shot Classification ----

function emitZeroShotClassificationApp(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');
  const sl = scriptLang(config);

  const resultsType = t ? '<Array<{ label: string; score: number }> | null>' : '';
  const sessionType = t ? '<Awaited<ReturnType<typeof createSession>> | null>' : '';
  const tokenizerType = t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : '';

  return `<script setup${sl}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { loadTokenizer, tokenizeText } from './lib/preprocess.${le}';
import { postprocessZeroShot } from './lib/postprocess.${le}';

const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

const results = ref${resultsType}(null);
const status = ref('Loading model...');
const text = ref('This is a thrilling adventure story.');
const labels = ref('travel, cooking, politics, sports, technology');
const session = shallowRef${sessionType}(null);
const tokenizer = shallowRef${tokenizerType}(null);

onMounted(() => {
  Promise.all([
    createSession(MODEL_PATH),
    loadTokenizer(TOKENIZER_PATH)
  ]).then(([s, tok]) => {
    session.value = s;
    tokenizer.value = tok;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => {
    status.value = 'Failed to load model';
    console.error('Model load error:', e);
  });
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
    // For NLI models, entailment is usually the last class (index 2)
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
      <textarea id="textInput" rows="4" v-model="text" aria-label="Text to classify" />
      <label for="labelsInput">Candidate labels (comma-separated)</label>
      <input id="labelsInput" type="text" v-model="labels" aria-label="Candidate labels" />
      <button class="primary-btn" @click="handleClassify" aria-label="Classify text">Classify</button>
    </div>
    <div id="results" class="results" role="status" aria-live="polite" aria-atomic="true">
      <template v-if="results">
        <div
          v-for="(r, i) in results"
          :key="r.label"
          :class="['result-row', { 'top-result': i === 0 }]"
          tabindex="0"
          :aria-label="\`\${r.label}: \${(r.score * 100).toFixed(1)} percent\`"
        >
          <span class="result-label">{{ r.label }}</span>
          <div class="result-bar-container">
            <div class="result-bar" :style="{ width: \`\${(r.score / (results[0]?.score || 1)) * 100}%\` }" />
          </div>
          <span class="result-pct">{{ (r.score * 100).toFixed(1) }}%</span>
        </div>
      </template>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js \u00b7 ${config.modelName} \u00b7 ${engineLabel}</div>
</template>
`;
}

// ---- Text/NLP: Text Generation ----

function emitTextGenerationApp(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');
  const sl = scriptLang(config);

  const sessionType = t ? '<Awaited<ReturnType<typeof createSession>> | null>' : '';
  const tokenizerType = t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : '';

  return `<script setup${sl}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { loadTokenizer } from './lib/preprocess.${le}';
import { postprocessGeneration, sampleNextToken } from './lib/postprocess.${le}';

const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

const output = ref('');
const status = ref('Loading model...');
const text = ref('Once upon a time');
const generating = ref(false);
const session = shallowRef${sessionType}(null);
const tokenizer = shallowRef${tokenizerType}(null);

onMounted(() => {
  Promise.all([
    createSession(MODEL_PATH),
    loadTokenizer(TOKENIZER_PATH)
  ]).then(([s, tok]) => {
    session.value = s;
    tokenizer.value = tok;
    status.value = '${config.modelName} \\u00b7 Ready';
  }).catch((e) => {
    status.value = 'Failed to load model';
    console.error('Model load error:', e);
  });
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
      <textarea id="textInput" rows="4" v-model="text" aria-label="Prompt text" :disabled="generating" />
      <button class="primary-btn" @click="handleGenerate" :disabled="generating" aria-label="Generate text">
        {{ generating ? 'Generating...' : 'Generate' }}
      </button>
    </div>
    <div id="output" class="transcript" role="status" aria-live="polite" aria-atomic="true">
      {{ output || '(generated text will appear here)' }}
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js \u00b7 ${config.modelName} \u00b7 ${engineLabel}</div>
</template>
`;
}

// ---- Fill-Mask ----

function emitFillMaskApp(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');
  const sl = scriptLang(config);

  const predictionsType = t ? '<Array<{ token: string; score: number }> | null>' : '';
  const sessionType = t ? '<Awaited<ReturnType<typeof createSession>> | null>' : '';
  const tokenizerType = t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : '';

  return `<script setup${sl}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { loadTokenizer, tokenizeText } from './lib/preprocess.${le}';
import { postprocessFillMask } from './lib/postprocess.${le}';

const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

const predictions = ref${predictionsType}(null);
const status = ref('Loading model...');
const text = ref('The capital of France is [MASK].');
const session = shallowRef${sessionType}(null);
const tokenizer = shallowRef${tokenizerType}(null);

onMounted(() => {
  Promise.all([createSession(MODEL_PATH), loadTokenizer(TOKENIZER_PATH)])
    .then(([s, tok]) => { session.value = s; tokenizer.value = tok; status.value = '${config.modelName} \\u00b7 Ready'; })
    .catch((e) => { status.value = 'Failed to load model'; console.error(e); });
});

async function handlePredict() {
  if (!session.value || !tokenizer.value || !text.value.trim()) return;
  status.value = '${config.modelName} \\u00b7 Processing...';
  const start = performance.now();
  const { inputIds } = tokenizeText(tokenizer.value, text.value);
  const output = await runInference(session.value, inputIds);
  const results = postprocessFillMask(output, inputIds, tokenizer.value);
  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  predictions.value = results;
}
</script>

<template>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="text-input">
      <label for="textInput">Enter text with [MASK] token</label>
      <textarea id="textInput" rows="4" v-model="text" />
      <button class="run-btn" @click="handlePredict">Predict</button>
    </div>
    <div class="mask-predictions" role="status" aria-live="polite">
      <div v-for="(p, i) in predictions" :key="i" class="mask-prediction">
        <span class="token">{{ p.token }}</span>
        <span class="prob">{{ (p.score * 100).toFixed(1) }}%</span>
      </div>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js \u00b7 ${config.modelName} \u00b7 ${engineLabel}</div>
</template>
`;
}

// ---- Sentence Similarity ----

function emitSentenceSimilarityApp(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');
  const sl = scriptLang(config);

  const scoresType = t ? '<Array<{ sentence: string; score: number }> | null>' : '';
  const sessionType = t ? '<Awaited<ReturnType<typeof createSession>> | null>' : '';
  const tokenizerType = t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : '';

  return `<script setup${sl}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { loadTokenizer, tokenizeText } from './lib/preprocess.${le}';
import { cosineSimilarity } from './lib/postprocess.${le}';

const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

const scores = ref${scoresType}(null);
const status = ref('Loading model...');
const source = ref('The weather is lovely today.');
const comparisons = ref('It is a beautiful day.\\nThe sun is shining bright.\\nI need to buy groceries.');
const session = shallowRef${sessionType}(null);
const tokenizer = shallowRef${tokenizerType}(null);

onMounted(() => {
  Promise.all([createSession(MODEL_PATH), loadTokenizer(TOKENIZER_PATH)])
    .then(([s, tok]) => { session.value = s; tokenizer.value = tok; status.value = '${config.modelName} \\u00b7 Ready'; })
    .catch((e) => { status.value = 'Failed to load model'; console.error(e); });
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
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="text-input">
      <label for="source">Source sentence</label>
      <textarea id="source" rows="2" v-model="source" />
      <label for="compare">Sentences to compare (one per line)</label>
      <textarea id="compare" rows="4" v-model="comparisons" />
      <button class="run-btn" @click="handleCompare">Compare</button>
    </div>
    <div class="similarity-pairs" role="status" aria-live="polite">
      <div v-for="(s, i) in scores" :key="i" class="similarity-score">
        <span>{{ s.sentence }}</span>
        <span class="value">{{ s.score.toFixed(4) }}</span>
      </div>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js \u00b7 ${config.modelName} \u00b7 ${engineLabel}</div>
</template>
`;
}

// ---- Depth Estimation ----

function emitDepthEstimationApp(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');
  const sl = scriptLang(config);

  const stringNullType = t ? '<string | null>' : '';
  const sessionType = t ? '<Awaited<ReturnType<typeof createSession>> | null>' : '';
  const canvasRefType = t ? '<HTMLCanvasElement | null>' : '';
  const inputRefType = t ? '<HTMLInputElement | null>' : '';

  return `<script setup${sl}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { preprocessImage } from './lib/preprocess.${le}';
import { postprocessDepth } from './lib/postprocess.${le}';

const MODEL_PATH = '${modelPath}';

const status = ref('Loading model...');
const imageUrl = ref${stringNullType}(null);
const dragOver = ref(false);
const session = shallowRef${sessionType}(null);
const canvasRef = ref${canvasRefType}(null);
const fileInputRef = ref${inputRefType}(null);

onMounted(() => {
  createSession(MODEL_PATH).then((s) => { session.value = s; status.value = '${config.modelName} \\u00b7 Ready'; })
    .catch((e) => { status.value = 'Failed to load model'; console.error(e); });
});

async function processImage(file${t ? ': File' : ''}) {
  if (!file.type.startsWith('image/') || !session.value) return;
  const url = URL.createObjectURL(file);
  imageUrl.value = url;
  const img = new Image();
  img.src = url;
  await new Promise((r) => { img.onload = r; });
  status.value = '${config.modelName} \\u00b7 Processing...';
  const start = performance.now();
  const input = preprocessImage(img);
  const output = await runInference(session.value, input);
  const depthMap = postprocessDepth(output);
  const canvas = canvasRef.value;
  if (canvas) {
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d')${t ? '!' : ''};
    const imgData = ctx.createImageData(canvas.width, canvas.height);
    for (let i = 0; i < depthMap.length; i++) {
      const v = depthMap[i];
      imgData.data[i * 4] = v; imgData.data[i * 4 + 1] = v; imgData.data[i * 4 + 2] = v; imgData.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
  }
  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
}

function handleDrop(e${t ? ': DragEvent' : ''}) { e.preventDefault(); dragOver.value = false; const f = e.dataTransfer${t ? '!' : ''}.files[0]; if (f) processImage(f); }
function handleFileChange(e${t ? ': Event' : ''}) { const f = (e.target as HTMLInputElement).files?.[0]; if (f) processImage(f); }
</script>

<template>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="container">
      <div>
        <div
          v-if="!imageUrl"
          :class="['drop-zone', { 'drag-over': dragOver }]"
          @click="fileInputRef?.click()"
          @dragover.prevent="dragOver = true"
          @dragleave="dragOver = false"
          @drop="handleDrop"
        >
          <p>Drop an image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input ref="fileInputRef" type="file" accept="image/*" hidden @change="handleFileChange" />
        </div>
        <div v-else class="preview">
          <img :src="imageUrl" alt="Source" />
          <button class="change-btn" @click="imageUrl = null">Choose another image</button>
        </div>
      </div>
      <div><canvas ref="canvasRef" class="depth-canvas" /></div>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js \u00b7 ${config.modelName} \u00b7 ${engineLabel}</div>
</template>
`;
}

// ---- Token Classification (NER) ----

function emitTokenClassificationApp(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');
  const sl = scriptLang(config);

  const sessionType = t ? '<Awaited<ReturnType<typeof createSession>> | null>' : '';
  const tokenizerType = t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : '';

  return `<script setup${sl}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { loadTokenizer, tokenizeText } from './lib/preprocess.${le}';
import { postprocessTokenClassification } from './lib/postprocess.${le}';

const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

const html = ref('');
const status = ref('Loading model...');
const text = ref('John Smith works at Google in Mountain View, California.');
const session = shallowRef${sessionType}(null);
const tokenizer = shallowRef${tokenizerType}(null);

onMounted(() => {
  Promise.all([createSession(MODEL_PATH), loadTokenizer(TOKENIZER_PATH)])
    .then(([s, tok]) => { session.value = s; tokenizer.value = tok; status.value = '${config.modelName} \\u00b7 Ready'; })
    .catch((e) => { status.value = 'Failed to load model'; console.error(e); });
});

async function handleAnalyze() {
  if (!session.value || !tokenizer.value || !text.value.trim()) return;
  status.value = '${config.modelName} \\u00b7 Processing...';
  const start = performance.now();
  const { inputIds } = tokenizeText(tokenizer.value, text.value);
  const output = await runInference(session.value, inputIds);
  const entities = postprocessTokenClassification(output, inputIds, tokenizer.value);
  let result = text.value;
  for (let i = entities.length - 1; i >= 0; i--) {
    const e = entities[i];
    result = result.slice(0, e.start) + '<span class="ner-entity" data-type="' + e.type + '">' + result.slice(e.start, e.end) + '</span>' + result.slice(e.end);
  }
  html.value = result;
  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
}
</script>

<template>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="text-input">
      <label for="textInput">Enter text to analyze</label>
      <textarea id="textInput" rows="4" v-model="text" />
      <button class="run-btn" @click="handleAnalyze">Analyze</button>
    </div>
    <div class="ner-output" role="status" aria-live="polite" v-html="html || text" />
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js \u00b7 ${config.modelName} \u00b7 ${engineLabel}</div>
</template>
`;
}

// ---- Question Answering ----

function emitQuestionAnsweringApp(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');
  const sl = scriptLang(config);

  const answerType = t ? '<{ answer: string; score: number } | null>' : '';
  const sessionType = t ? '<Awaited<ReturnType<typeof createSession>> | null>' : '';
  const tokenizerType = t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : '';

  return `<script setup${sl}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { loadTokenizer, tokenizeText } from './lib/preprocess.${le}';
import { postprocessQA } from './lib/postprocess.${le}';

const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

const answer = ref${answerType}(null);
const status = ref('Loading model...');
const context = ref('The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was constructed from 1887 to 1889.');
const question = ref('When was the Eiffel Tower built?');
const session = shallowRef${sessionType}(null);
const tokenizer = shallowRef${tokenizerType}(null);

onMounted(() => {
  Promise.all([createSession(MODEL_PATH), loadTokenizer(TOKENIZER_PATH)])
    .then(([s, tok]) => { session.value = s; tokenizer.value = tok; status.value = '${config.modelName} \\u00b7 Ready'; })
    .catch((e) => { status.value = 'Failed to load model'; console.error(e); });
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
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="qa-input">
      <label for="context">Context</label>
      <textarea id="context" rows="4" v-model="context" />
      <label for="question">Question</label>
      <input id="question" type="text" v-model="question" class="labels-input" />
      <button class="run-btn" @click="handleAnswer">Answer</button>
    </div>
    <div class="qa-answer" role="status" aria-live="polite">
      <template v-if="answer">
        <div>{{ answer.answer }}</div>
        <div class="score">Confidence: {{ (answer.score * 100).toFixed(1) }}%</div>
      </template>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js \u00b7 ${config.modelName} \u00b7 ${engineLabel}</div>
</template>
`;
}

// ---- Summarization ----

function emitSummarizationApp(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');
  const sl = scriptLang(config);

  const sessionType = t ? '<Awaited<ReturnType<typeof createSession>> | null>' : '';
  const tokenizerType = t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : '';

  return `<script setup${sl}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { loadTokenizer, tokenizeText } from './lib/preprocess.${le}';
import { postprocessSummarization } from './lib/postprocess.${le}';

const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

const output = ref('');
const status = ref('Loading model...');
const text = ref('Artificial intelligence has transformed many industries. Machine learning models can now process natural language, recognize images, and generate creative content.');
const processing = ref(false);
const session = shallowRef${sessionType}(null);
const tokenizer = shallowRef${tokenizerType}(null);

onMounted(() => {
  Promise.all([createSession(MODEL_PATH), loadTokenizer(TOKENIZER_PATH)])
    .then(([s, tok]) => { session.value = s; tokenizer.value = tok; status.value = '${config.modelName} \\u00b7 Ready'; })
    .catch((e) => { status.value = 'Failed to load model'; console.error(e); });
});

async function handleSummarize() {
  if (!session.value || !tokenizer.value || !text.value.trim() || processing.value) return;
  processing.value = true;
  status.value = '${config.modelName} \\u00b7 Summarizing...';
  const start = performance.now();
  const { inputIds } = tokenizeText(tokenizer.value, text.value);
  const raw = await runInference(session.value, inputIds);
  const summary = postprocessSummarization(raw, tokenizer.value, 128, 1);
  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  output.value = summary;
  processing.value = false;
}
</script>

<template>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="text-input">
      <label for="textInput">Enter text to summarize</label>
      <textarea id="textInput" rows="6" v-model="text" :disabled="processing" />
      <button class="run-btn" @click="handleSummarize" :disabled="processing">{{ processing ? 'Summarizing...' : 'Summarize' }}</button>
    </div>
    <div class="generation-output" role="status" aria-live="polite">{{ output || '(summary will appear here)' }}</div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js \u00b7 ${config.modelName} \u00b7 ${engineLabel}</div>
</template>
`;
}

// ---- Translation ----

function emitTranslationApp(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');
  const sl = scriptLang(config);

  const sessionType = t ? '<Awaited<ReturnType<typeof createSession>> | null>' : '';
  const tokenizerType = t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : '';

  return `<script setup${sl}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { loadTokenizer, tokenizeText } from './lib/preprocess.${le}';
import { postprocessTranslation } from './lib/postprocess.${le}';

const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

const output = ref('');
const status = ref('Loading model...');
const text = ref('Hello, how are you today?');
const processing = ref(false);
const session = shallowRef${sessionType}(null);
const tokenizer = shallowRef${tokenizerType}(null);

onMounted(() => {
  Promise.all([createSession(MODEL_PATH), loadTokenizer(TOKENIZER_PATH)])
    .then(([s, tok]) => { session.value = s; tokenizer.value = tok; status.value = '${config.modelName} \\u00b7 Ready'; })
    .catch((e) => { status.value = 'Failed to load model'; console.error(e); });
});

async function handleTranslate() {
  if (!session.value || !tokenizer.value || !text.value.trim() || processing.value) return;
  processing.value = true;
  status.value = '${config.modelName} \\u00b7 Translating...';
  const start = performance.now();
  const { inputIds } = tokenizeText(tokenizer.value, text.value);
  const raw = await runInference(session.value, inputIds);
  const translation = postprocessTranslation(raw, tokenizer.value, 128, 1);
  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  output.value = translation;
  processing.value = false;
}
</script>

<template>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="text-input">
      <label for="textInput">Enter text to translate</label>
      <textarea id="textInput" rows="4" v-model="text" :disabled="processing" />
      <button class="run-btn" @click="handleTranslate" :disabled="processing">{{ processing ? 'Translating...' : 'Translate' }}</button>
    </div>
    <div class="generation-output" role="status" aria-live="polite">{{ output || '(translation will appear here)' }}</div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js \u00b7 ${config.modelName} \u00b7 ${engineLabel}</div>
</template>
`;
}

// ---- Text2Text Generation ----

function emitText2TextApp(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');
  const sl = scriptLang(config);

  const sessionType = t ? '<Awaited<ReturnType<typeof createSession>> | null>' : '';
  const tokenizerType = t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : '';

  return `<script setup${sl}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { loadTokenizer, tokenizeText } from './lib/preprocess.${le}';
import { postprocessText2Text } from './lib/postprocess.${le}';

const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

const output = ref('');
const status = ref('Loading model...');
const text = ref('Paraphrase: The house is big and beautiful.');
const processing = ref(false);
const session = shallowRef${sessionType}(null);
const tokenizer = shallowRef${tokenizerType}(null);

onMounted(() => {
  Promise.all([createSession(MODEL_PATH), loadTokenizer(TOKENIZER_PATH)])
    .then(([s, tok]) => { session.value = s; tokenizer.value = tok; status.value = '${config.modelName} \\u00b7 Ready'; })
    .catch((e) => { status.value = 'Failed to load model'; console.error(e); });
});

async function handleRun() {
  if (!session.value || !tokenizer.value || !text.value.trim() || processing.value) return;
  processing.value = true;
  status.value = '${config.modelName} \\u00b7 Processing...';
  const start = performance.now();
  const { inputIds } = tokenizeText(tokenizer.value, text.value);
  const raw = await runInference(session.value, inputIds);
  const result = postprocessText2Text(raw, tokenizer.value, 128, 1);
  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  output.value = result;
  processing.value = false;
}
</script>

<template>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="text-input">
      <label for="textInput">Enter input text</label>
      <textarea id="textInput" rows="4" v-model="text" :disabled="processing" />
      <button class="run-btn" @click="handleRun" :disabled="processing">{{ processing ? 'Processing...' : 'Run' }}</button>
    </div>
    <div class="generation-output" role="status" aria-live="polite">{{ output || '(output will appear here)' }}</div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js \u00b7 ${config.modelName} \u00b7 ${engineLabel}</div>
</template>
`;
}

// ---- Conversational ----

function emitConversationalApp(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');
  const sl = scriptLang(config);

  const messagesType = t ? '<Array<{ role: string; text: string }>>' : '';
  const sessionType = t ? '<Awaited<ReturnType<typeof createSession>> | null>' : '';
  const tokenizerType = t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : '';
  const historyType = t ? '<string[]>' : '';

  return `<script setup${sl}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { loadTokenizer } from './lib/preprocess.${le}';
import { postprocessConversational, sampleNextToken } from './lib/postprocess.${le}';

const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

const messages = ref${messagesType}([]);
const input = ref('');
const status = ref('Loading model...');
const generating = ref(false);
const session = shallowRef${sessionType}(null);
const tokenizer = shallowRef${tokenizerType}(null);
const history = ref${historyType}([]);

onMounted(() => {
  Promise.all([createSession(MODEL_PATH), loadTokenizer(TOKENIZER_PATH)])
    .then(([s, tok]) => { session.value = s; tokenizer.value = tok; status.value = '${config.modelName} \\u00b7 Ready'; })
    .catch((e) => { status.value = 'Failed to load model'; console.error(e); });
});

async function handleSend() {
  if (!session.value || !tokenizer.value || !input.value.trim() || generating.value) return;
  const userMsg = input.value.trim();
  input.value = '';
  messages.value = [...messages.value, { role: 'user', text: userMsg }];
  history.value.push(userMsg);
  generating.value = true;
  status.value = '${config.modelName} \\u00b7 Generating...';
  const start = performance.now();
  const prompt = history.value.join(' ');
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
  history.value.push(reply);
  messages.value = [...messages.value, { role: 'bot', text: reply }];
  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  generating.value = false;
}

function handleKeyDown(e${t ? ': KeyboardEvent' : ''}) {
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
      <input type="text" v-model="input" @keydown="handleKeyDown" placeholder="Type a message..." :disabled="generating" />
      <button class="run-btn" @click="handleSend" :disabled="generating">Send</button>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js \u00b7 ${config.modelName} \u00b7 ${engineLabel}</div>
</template>
`;
}

// ---- Table Question Answering ----

function emitTableQAApp(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');
  const sl = scriptLang(config);

  const answerType = t ? '<{ answer: string; score: number } | null>' : '';
  const sessionType = t ? '<Awaited<ReturnType<typeof createSession>> | null>' : '';
  const tokenizerType = t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : '';

  return `<script setup${sl}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { loadTokenizer, tokenizeText } from './lib/preprocess.${le}';
import { postprocessTableQA } from './lib/postprocess.${le}';

const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

const answer = ref${answerType}(null);
const status = ref('Loading model...');
const table = ref('Name, Age, City\\nAlice, 30, New York\\nBob, 25, San Francisco');
const question = ref('Who lives in San Francisco?');
const session = shallowRef${sessionType}(null);
const tokenizer = shallowRef${tokenizerType}(null);

onMounted(() => {
  Promise.all([createSession(MODEL_PATH), loadTokenizer(TOKENIZER_PATH)])
    .then(([s, tok]) => { session.value = s; tokenizer.value = tok; status.value = '${config.modelName} \\u00b7 Ready'; })
    .catch((e) => { status.value = 'Failed to load model'; console.error(e); });
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
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="qa-input">
      <label for="table">Table data (CSV format)</label>
      <div class="table-input"><textarea id="table" rows="4" v-model="table" /></div>
      <label for="question">Question</label>
      <input id="question" type="text" v-model="question" class="labels-input" />
      <button class="run-btn" @click="handleAnswer">Answer</button>
    </div>
    <div class="qa-answer" role="status" aria-live="polite">
      <template v-if="answer">
        <div>{{ answer.answer }}</div>
        <div class="score">Confidence: {{ (answer.score * 100).toFixed(1) }}%</div>
      </template>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js \u00b7 ${config.modelName} \u00b7 ${engineLabel}</div>
</template>
`;
}

// ---- Image-to-Text ----

function emitImageToTextApp(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');
  const sl = scriptLang(config);

  const stringNullType = t ? '<string | null>' : '';
  const sessionType = t ? '<Awaited<ReturnType<typeof createSession>> | null>' : '';
  const inputRefType = t ? '<HTMLInputElement | null>' : '';

  return `<script setup${sl}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { preprocessImage } from './lib/preprocess.${le}';
import { postprocessImageToText } from './lib/postprocess.${le}';

const MODEL_PATH = '${modelPath}';

const caption = ref('');
const status = ref('Loading model...');
const imageUrl = ref${stringNullType}(null);
const dragOver = ref(false);
const session = shallowRef${sessionType}(null);
const fileInputRef = ref${inputRefType}(null);

onMounted(() => {
  createSession(MODEL_PATH).then((s) => { session.value = s; status.value = '${config.modelName} \\u00b7 Ready'; })
    .catch((e) => { status.value = 'Failed to load model'; console.error(e); });
});

async function processImage(file${t ? ': File' : ''}) {
  if (!file.type.startsWith('image/') || !session.value) return;
  const url = URL.createObjectURL(file);
  imageUrl.value = url;
  const img = new Image();
  img.src = url;
  await new Promise((r) => { img.onload = r; });
  status.value = '${config.modelName} \\u00b7 Generating caption...';
  const start = performance.now();
  const input = preprocessImage(img);
  const output = await runInference(session.value, input);
  const text = postprocessImageToText(output);
  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  caption.value = text;
}

function handleDrop(e${t ? ': DragEvent' : ''}) { e.preventDefault(); dragOver.value = false; const f = e.dataTransfer${t ? '!' : ''}.files[0]; if (f) processImage(f); }
function handleFileChange(e${t ? ': Event' : ''}) { const f = (e.target as HTMLInputElement).files?.[0]; if (f) processImage(f); }
</script>

<template>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="container">
      <div>
        <div
          v-if="!imageUrl"
          :class="['drop-zone', { 'drag-over': dragOver }]"
          @click="fileInputRef?.click()"
          @dragover.prevent="dragOver = true"
          @dragleave="dragOver = false"
          @drop="handleDrop"
        >
          <p>Drop an image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input ref="fileInputRef" type="file" accept="image/*" hidden @change="handleFileChange" />
        </div>
        <div v-else class="preview">
          <img :src="imageUrl" alt="Source" />
          <button class="change-btn" @click="() => { imageUrl = null; caption = ''; }">Choose another image</button>
        </div>
      </div>
      <div class="generation-output" role="status" aria-live="polite">{{ caption || '(caption will appear here)' }}</div>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js \u00b7 ${config.modelName} \u00b7 ${engineLabel}</div>
</template>
`;
}

// ---- Visual Question Answering ----

function emitVQAApp(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');
  const sl = scriptLang(config);

  const stringNullType = t ? '<string | null>' : '';
  const sessionType = t ? '<Awaited<ReturnType<typeof createSession>> | null>' : '';
  const tokenizerType = t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : '';
  const imgElType = t ? '<HTMLImageElement | null>' : '';
  const inputRefType = t ? '<HTMLInputElement | null>' : '';

  return `<script setup${sl}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { preprocessImage } from './lib/preprocess.${le}';
import { postprocessVQA } from './lib/postprocess.${le}';
import { loadTokenizer } from './lib/preprocess.${le}';

const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

const answer = ref('');
const status = ref('Loading model...');
const imageUrl = ref${stringNullType}(null);
const question = ref('What is in this image?');
const dragOver = ref(false);
const session = shallowRef${sessionType}(null);
const tokenizer = shallowRef${tokenizerType}(null);
const imgEl = shallowRef${imgElType}(null);
const fileInputRef = ref${inputRefType}(null);

onMounted(() => {
  Promise.all([createSession(MODEL_PATH), loadTokenizer(TOKENIZER_PATH)])
    .then(([s, tok]) => { session.value = s; tokenizer.value = tok; status.value = '${config.modelName} \\u00b7 Ready'; })
    .catch((e) => { status.value = 'Failed to load model'; console.error(e); });
});

function handleFile(file${t ? ': File' : ''}) {
  if (!file.type.startsWith('image/')) return;
  const url = URL.createObjectURL(file);
  imageUrl.value = url;
  const img = new Image();
  img.src = url;
  img.onload = () => { imgEl.value = img; };
}

async function handleAsk() {
  if (!session.value || !tokenizer.value || !imgEl.value || !question.value.trim()) return;
  status.value = '${config.modelName} \\u00b7 Processing...';
  const start = performance.now();
  const input = preprocessImage(imgEl.value);
  const output = await runInference(session.value, input);
  const result = postprocessVQA(output, tokenizer.value);
  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  answer.value = result;
}

function handleDrop(e${t ? ': DragEvent' : ''}) { e.preventDefault(); dragOver.value = false; const f = e.dataTransfer${t ? '!' : ''}.files[0]; if (f) handleFile(f); }
function handleFileChange(e${t ? ': Event' : ''}) { const f = (e.target as HTMLInputElement).files?.[0]; if (f) handleFile(f); }
</script>

<template>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="container">
      <div class="multimodal-input">
        <div
          v-if="!imageUrl"
          :class="['drop-zone', { 'drag-over': dragOver }]"
          @click="fileInputRef?.click()"
          @dragover.prevent="dragOver = true"
          @dragleave="dragOver = false"
          @drop="handleDrop"
        >
          <p>Drop an image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input ref="fileInputRef" type="file" accept="image/*" hidden @change="handleFileChange" />
        </div>
        <div v-else class="preview">
          <img :src="imageUrl" alt="Selected" />
          <button class="change-btn" @click="() => { imageUrl = null; answer = ''; imgEl = null; }">Choose another image</button>
        </div>
        <input type="text" class="question-input" v-model="question" placeholder="Ask a question..." />
        <button class="run-btn" @click="handleAsk">Ask</button>
      </div>
      <div class="qa-answer" role="status" aria-live="polite">{{ answer }}</div>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js \u00b7 ${config.modelName} \u00b7 ${engineLabel}</div>
</template>
`;
}

// ---- Document Question Answering ----

function emitDocQAApp(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');
  const sl = scriptLang(config);

  const stringNullType = t ? '<string | null>' : '';
  const sessionType = t ? '<Awaited<ReturnType<typeof createSession>> | null>' : '';
  const tokenizerType = t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : '';
  const imgElType = t ? '<HTMLImageElement | null>' : '';
  const inputRefType = t ? '<HTMLInputElement | null>' : '';

  return `<script setup${sl}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { preprocessImage } from './lib/preprocess.${le}';
import { postprocessDocQA } from './lib/postprocess.${le}';
import { loadTokenizer } from './lib/preprocess.${le}';

const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

const answer = ref('');
const status = ref('Loading model...');
const imageUrl = ref${stringNullType}(null);
const question = ref('What is the total amount?');
const dragOver = ref(false);
const session = shallowRef${sessionType}(null);
const tokenizer = shallowRef${tokenizerType}(null);
const imgEl = shallowRef${imgElType}(null);
const fileInputRef = ref${inputRefType}(null);

onMounted(() => {
  Promise.all([createSession(MODEL_PATH), loadTokenizer(TOKENIZER_PATH)])
    .then(([s, tok]) => { session.value = s; tokenizer.value = tok; status.value = '${config.modelName} \\u00b7 Ready'; })
    .catch((e) => { status.value = 'Failed to load model'; console.error(e); });
});

function handleFile(file${t ? ': File' : ''}) {
  if (!file.type.startsWith('image/')) return;
  const url = URL.createObjectURL(file);
  imageUrl.value = url;
  const img = new Image();
  img.src = url;
  img.onload = () => { imgEl.value = img; };
}

async function handleAsk() {
  if (!session.value || !tokenizer.value || !imgEl.value || !question.value.trim()) return;
  status.value = '${config.modelName} \\u00b7 Processing...';
  const start = performance.now();
  const input = preprocessImage(imgEl.value);
  const output = await runInference(session.value, input);
  const result = postprocessDocQA(output, tokenizer.value);
  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  answer.value = result;
}

function handleDrop(e${t ? ': DragEvent' : ''}) { e.preventDefault(); dragOver.value = false; const f = e.dataTransfer${t ? '!' : ''}.files[0]; if (f) handleFile(f); }
function handleFileChange(e${t ? ': Event' : ''}) { const f = (e.target as HTMLInputElement).files?.[0]; if (f) handleFile(f); }
</script>

<template>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="container">
      <div class="multimodal-input">
        <div
          v-if="!imageUrl"
          :class="['drop-zone', { 'drag-over': dragOver }]"
          @click="fileInputRef?.click()"
          @dragover.prevent="dragOver = true"
          @dragleave="dragOver = false"
          @drop="handleDrop"
        >
          <p>Drop a document image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input ref="fileInputRef" type="file" accept="image/*" hidden @change="handleFileChange" />
        </div>
        <div v-else class="preview">
          <img :src="imageUrl" alt="Document" />
          <button class="change-btn" @click="() => { imageUrl = null; answer = ''; imgEl = null; }">Choose another image</button>
        </div>
        <input type="text" class="question-input" v-model="question" placeholder="Ask about the document..." />
        <button class="run-btn" @click="handleAsk">Ask</button>
      </div>
      <div class="qa-answer" role="status" aria-live="polite">{{ answer }}</div>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js \u00b7 ${config.modelName} \u00b7 ${engineLabel}</div>
</template>
`;
}

// ---- Image-Text-to-Text ----

function emitImageTextToTextApp(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');
  const sl = scriptLang(config);

  const stringNullType = t ? '<string | null>' : '';
  const sessionType = t ? '<Awaited<ReturnType<typeof createSession>> | null>' : '';
  const tokenizerType = t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : '';
  const imgElType = t ? '<HTMLImageElement | null>' : '';
  const inputRefType = t ? '<HTMLInputElement | null>' : '';

  return `<script setup${sl}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { preprocessImage } from './lib/preprocess.${le}';
import { postprocessImageTextToText } from './lib/postprocess.${le}';
import { loadTokenizer } from './lib/preprocess.${le}';

const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

const output = ref('');
const status = ref('Loading model...');
const imageUrl = ref${stringNullType}(null);
const prompt = ref('Describe this image in detail.');
const dragOver = ref(false);
const processing = ref(false);
const session = shallowRef${sessionType}(null);
const tokenizer = shallowRef${tokenizerType}(null);
const imgEl = shallowRef${imgElType}(null);
const fileInputRef = ref${inputRefType}(null);

onMounted(() => {
  Promise.all([createSession(MODEL_PATH), loadTokenizer(TOKENIZER_PATH)])
    .then(([s, tok]) => { session.value = s; tokenizer.value = tok; status.value = '${config.modelName} \\u00b7 Ready'; })
    .catch((e) => { status.value = 'Failed to load model'; console.error(e); });
});

function handleFile(file${t ? ': File' : ''}) {
  if (!file.type.startsWith('image/')) return;
  const url = URL.createObjectURL(file);
  imageUrl.value = url;
  const img = new Image();
  img.src = url;
  img.onload = () => { imgEl.value = img; };
}

async function handleGenerate() {
  if (!session.value || !tokenizer.value || !imgEl.value || processing.value) return;
  processing.value = true;
  status.value = '${config.modelName} \\u00b7 Generating...';
  const start = performance.now();
  const input = preprocessImage(imgEl.value);
  const raw = await runInference(session.value, input);
  const result = postprocessImageTextToText(raw, tokenizer.value);
  const elapsed = (performance.now() - start).toFixed(1);
  status.value = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session.value)}\`;
  output.value = result;
  processing.value = false;
}

function handleDrop(e${t ? ': DragEvent' : ''}) { e.preventDefault(); dragOver.value = false; const f = e.dataTransfer${t ? '!' : ''}.files[0]; if (f) handleFile(f); }
function handleFileChange(e${t ? ': Event' : ''}) { const f = (e.target as HTMLInputElement).files?.[0]; if (f) handleFile(f); }
</script>

<template>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div class="container">
      <div class="multimodal-input">
        <div
          v-if="!imageUrl"
          :class="['drop-zone', { 'drag-over': dragOver }]"
          @click="fileInputRef?.click()"
          @dragover.prevent="dragOver = true"
          @dragleave="dragOver = false"
          @drop="handleDrop"
        >
          <p>Drop an image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input ref="fileInputRef" type="file" accept="image/*" hidden @change="handleFileChange" />
        </div>
        <div v-else class="preview">
          <img :src="imageUrl" alt="Selected" />
          <button class="change-btn" @click="() => { imageUrl = null; output = ''; imgEl = null; }">Choose another image</button>
        </div>
        <input type="text" class="question-input" v-model="prompt" placeholder="Enter a prompt..." />
        <button class="run-btn" @click="handleGenerate" :disabled="processing">{{ processing ? 'Generating...' : 'Generate' }}</button>
      </div>
      <div class="generation-output" role="status" aria-live="polite">{{ output || '(output will appear here)' }}</div>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js \u00b7 ${config.modelName} \u00b7 ${engineLabel}</div>
</template>
`;
}

// ---- Audio-to-Audio ----

function emitAudioToAudioApp(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');
  const sl = scriptLang(config);

  const sessionType = t ? '<Awaited<ReturnType<typeof createSession>> | null>' : '';

  return `<script setup${sl}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { postprocessAudioToAudio, playAudio } from './lib/postprocess.${le}';

const MODEL_PATH = '${modelPath}';

const status = ref('Loading model...');
const output = ref('');
const session = shallowRef${sessionType}(null);

onMounted(() => {
  createSession(MODEL_PATH).then((s) => { session.value = s; status.value = '${config.modelName} \\u00b7 Ready'; })
    .catch((e) => { status.value = 'Failed to load model'; console.error(e); });
});

async function handleFile(e${t ? ': Event' : ''}) {
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
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div><label for="fileInput">Choose an audio file</label><input id="fileInput" type="file" accept="audio/*" @change="handleFile" /></div>
    <div class="generation-output" role="status" aria-live="polite">{{ output }}</div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js \u00b7 ${config.modelName} \u00b7 ${engineLabel}</div>
</template>
`;
}

// ---- Speaker Diarization ----

function emitSpeakerDiarizationApp(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');
  const sl = scriptLang(config);

  const segmentsType = t ? '<Array<{ speaker: number; start: number; end: number; text: string }> | null>' : '';
  const sessionType = t ? '<Awaited<ReturnType<typeof createSession>> | null>' : '';

  return `<script setup${sl}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { postprocessSpeakerDiarization } from './lib/postprocess.${le}';

const MODEL_PATH = '${modelPath}';

const segments = ref${segmentsType}(null);
const status = ref('Loading model...');
const session = shallowRef${sessionType}(null);

onMounted(() => {
  createSession(MODEL_PATH).then((s) => { session.value = s; status.value = '${config.modelName} \\u00b7 Ready'; })
    .catch((e) => { status.value = 'Failed to load model'; console.error(e); });
});

async function handleFile(e${t ? ': Event' : ''}) {
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
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div><label for="fileInput">Choose an audio file</label><input id="fileInput" type="file" accept="audio/*" @change="handleFile" /></div>
    <div class="diarization-timeline" role="status" aria-live="polite">
      <div v-for="(seg, i) in segments" :key="i" class="diarization-segment">
        <span class="speaker">Speaker {{ seg.speaker }}</span>
        <span>{{ seg.text }}</span>
        <span class="time">{{ seg.start.toFixed(1) }}s - {{ seg.end.toFixed(1) }}s</span>
      </div>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js \u00b7 ${config.modelName} \u00b7 ${engineLabel}</div>
</template>
`;
}

// ---- Voice Activity Detection ----

function emitVADApp(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');
  const sl = scriptLang(config);

  const segmentsType = t ? '<Array<{ label: string; start: number; end: number }> | null>' : '';
  const sessionType = t ? '<Awaited<ReturnType<typeof createSession>> | null>' : '';

  return `<script setup${sl}>
import { ref, shallowRef, onMounted } from 'vue';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { postprocessVAD } from './lib/postprocess.${le}';

const MODEL_PATH = '${modelPath}';

const segments = ref${segmentsType}(null);
const status = ref('Loading model...');
const session = shallowRef${sessionType}(null);

onMounted(() => {
  createSession(MODEL_PATH).then((s) => { session.value = s; status.value = '${config.modelName} \\u00b7 Ready'; })
    .catch((e) => { status.value = 'Failed to load model'; console.error(e); });
});

async function handleFile(e${t ? ': Event' : ''}) {
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
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div><label for="fileInput">Choose an audio file</label><input id="fileInput" type="file" accept="audio/*" @change="handleFile" /></div>
    <div class="vad-segments" role="status" aria-live="polite">
      <div v-for="(seg, i) in segments" :key="i" class="vad-segment">
        <span class="label">{{ seg.label }}</span>
        <span class="time">{{ seg.start.toFixed(1) }}s - {{ seg.end.toFixed(1) }}s</span>
      </div>
    </div>
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js \u00b7 ${config.modelName} \u00b7 ${engineLabel}</div>
</template>
`;
}

// ---- Realtime (Camera / Screen) ----

function emitRealtimeApp(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const isScreen = config.input === 'screen';
  const startFn = isScreen ? 'startScreenCapture' : 'startCamera';
  const btnLabel = isScreen ? 'Start Screen Capture' : 'Enable Camera';
  const actionLabel = isScreen ? 'capture your screen' : 'use your camera';
  const sl = scriptLang(config);

  const sessionType = t ? '<Awaited<ReturnType<typeof createSession>> | null>' : '';
  const videoRefType = t ? '<HTMLVideoElement | null>' : '';
  const canvasRefType = t ? '<HTMLCanvasElement | null>' : '';
  const loopRefType = t ? '<ReturnType<typeof createInferenceLoop> | null>' : '';

  // Determine postprocess import and processing code based on task
  let postImport: string;
  let processCode: string;
  let renderCode: string;
  let extraConsts = '';

  if (config.task === 'object-detection') {
    const outputShape = config.modelMeta.outputs[0]?.shape ?? [1, 84, 8400];
    const numAttributes = outputShape[1] ?? 84;
    const numAnchors = outputShape[2] ?? 8400;
    postImport = `import { postprocessDetections } from './lib/postprocess.${le}';`;
    extraConsts = `const NUM_ATTRIBUTES = ${numAttributes};\nconst NUM_ANCHORS = ${numAnchors};\nconst COLORS = [[56,189,248],[249,115,22],[34,197,94],[168,85,247],[251,191,36],[239,68,68],[20,184,166],[236,72,153],[99,102,241],[163,230,53]];`;
    processCode = `const boxes = postprocessDetections(output, NUM_ANCHORS, NUM_ATTRIBUTES);`;
    renderCode = `
        const scale = videoRef.value${t ? '!' : ''}.videoWidth / ${config.preprocess.imageSize};
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        for (const box of boxes) {
          const c = COLORS[box.classIndex % COLORS.length];
          const color = \\\`rgb(\\\${c[0]},\\\${c[1]},\\\${c[2]})\\\`;
          ctx.strokeStyle = color; ctx.lineWidth = 2;
          ctx.strokeRect(box.x * scale, box.y * scale, box.width * scale, box.height * scale);
          const label = \\\`Class \\\${box.classIndex} (\\\${(box.score * 100).toFixed(0)}%)\\\`;
          ctx.font = '14px system-ui'; ctx.fillStyle = color;
          ctx.fillRect(box.x * scale, box.y * scale - 20, ctx.measureText(label).width + 8, 20);
          ctx.fillStyle = '#fff'; ctx.fillText(label, box.x * scale + 4, box.y * scale - 5);
        }`;
  } else if (config.task === 'image-segmentation') {
    const outputShape = config.modelMeta.outputs[0]?.shape ?? [1, 21, 512, 512];
    const numClasses = outputShape[1] ?? 21;
    const maskH = outputShape[2] ?? 512;
    const maskW = outputShape[3] ?? 512;
    postImport = `import { postprocessSegmentation } from './lib/postprocess.${le}';`;
    extraConsts = `const NUM_CLASSES = ${numClasses};\nconst MASK_H = ${maskH};\nconst MASK_W = ${maskW};\nconst COLORS = [[56,189,248],[249,115,22],[34,197,94],[168,85,247],[251,191,36],[239,68,68],[20,184,166],[236,72,153],[99,102,241],[163,230,53]];`;
    processCode = `const mask = postprocessSegmentation(output, NUM_CLASSES, MASK_H, MASK_W);`;
    renderCode = `
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
        ctx.drawImage(maskCanvas, 0, 0, ctx.canvas.width, ctx.canvas.height);`;
  } else {
    postImport = `import { postprocessResults } from './lib/postprocess.${le}';`;
    processCode = `const results = postprocessResults(output);`;
    renderCode = `
        const label = \\\`Class \\\${results.indices[0]} (\\\${(results.values[0] * 100).toFixed(1)}%)\\\`;
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        ctx.font = 'bold 24px system-ui'; ctx.fillStyle = 'rgba(59,130,246,0.85)';
        ctx.fillRect(8, 8, ctx.measureText(label).width + 16, 36);
        ctx.fillStyle = '#fff'; ctx.fillText(label, 16, 34);`;
  }

  return `<script setup${sl}>
import { ref, shallowRef, onMounted, onUnmounted } from 'vue';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { preprocessImage } from './lib/preprocess.${le}';
${postImport}
import { ${startFn}, captureFrame, stopStream, createInferenceLoop } from './lib/input.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
${extraConsts}

const status = ref('Loading model...');
const started = ref(false);
const session = shallowRef${sessionType}(null);
const videoRef = ref${videoRefType}(null);
const overlayRef = ref${canvasRefType}(null);
const loopRef = shallowRef${loopRefType}(null);

onMounted(() => {
  createSession(MODEL_PATH).then((s) => {
    session.value = s;
    status.value = '${config.modelName} \\u00b7 Ready \\u00b7 Tap Start';
  }).catch((e) => { status.value = 'Failed to load model'; console.error('Model load error:', e); });
});

onUnmounted(() => { if (loopRef.value) loopRef.value.stop(); });

async function startCapture() {
  if (!videoRef.value || !overlayRef.value || !session.value) return;
  try {
    await ${startFn}(videoRef.value);
    overlayRef.value.width = videoRef.value.videoWidth;
    overlayRef.value.height = videoRef.value.videoHeight;
    started.value = true;
    loopRef.value = createInferenceLoop({
      video: videoRef.value,
      canvas: overlayRef.value,
      async onFrame(imageData${t ? ': ImageData' : ''}) {
        const t0 = performance.now();
        const inputTensor = preprocessImage(imageData.data, imageData.width, imageData.height);
        const output = await runInference(session.value${t ? '!' : ''}, inputTensor);
        ${processCode}
        const elapsed = performance.now() - t0;
        const ctx = overlayRef.value${t ? '!' : ''}.getContext('2d')${t ? '!' : ''};
        ${renderCode}
        return { result: null, elapsed };
      },
      onStatus(elapsed${t ? ': number' : ''}) {
        status.value = \`${config.modelName} \\u00b7 \${elapsed.toFixed(1)}ms \\u00b7 \${getBackendLabel(session.value${t ? '!' : ''})}\`;
      },
    });
    loopRef.value.start();
  } catch (e) {
    status.value = '${isScreen ? 'Screen capture' : 'Camera'} access denied';
    console.error(e);
  }
}
</script>

<template>
  <a href="#results" class="skip-link">Skip to results</a>
  <main>
    <h1>${config.modelName} — ${taskLabel}</h1>
    <div v-if="!started" class="permission-prompt">
      <p>This app ${actionLabel} to run ${taskLabel.toLowerCase()} in real time.</p>
      <p class="hint">No video is recorded or sent anywhere.</p>
      <button class="primary-btn" @click="startCapture">${btnLabel}</button>
    </div>
    <div v-if="started" class="video-wrapper">
      <video ref="videoRef" autoplay playsinline muted />
      <canvas ref="overlayRef" class="overlay-canvas" />
    </div>
    <video v-if="!started" ref="videoRef" hidden />
  </main>
  <aside class="status-bar"><span>{{ status }}</span></aside>
  <div class="footer">Generated by webai.js \u00b7 ${config.modelName} \u00b7 ${engineLabel}</div>
</template>
`;
}

// ---- Extended CSS ----

function emitExtendedCSS(): string {
  return `
/* Canvas overlay */
.preview-wrapper {
  position: relative;
  display: inline-block;
}

.preview-wrapper .overlay-canvas,
.video-wrapper .overlay-canvas {
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

function emitAppCssFile(config: ResolvedConfig): string {
  const extra = needsExtendedCSS(config) ? emitExtendedCSS() : '';
  return `${emitDesignSystemCSS(config)}\n\n${emitAppCSS()}${extra}`;
}

/** Wrap a CodeBlock's code with exports for use as a lib module */
function toLibModule(block: CodeBlock | undefined): string {
  if (!block) return '';
  return addExports(block.code, block.exports);
}

/**
 * Emit Vue-Vite framework files.
 */
export function emitVueVite(config: ResolvedConfig, blocks: CodeBlock[]): GeneratedFile[] {
  const le = libExt(config);
  const me = mainExt(config);

  const inputBlock = findBlock(blocks, 'input');
  const preprocessBlock = findBlock(blocks, 'preprocess');
  const inferenceBlock = findBlock(blocks, 'inference');
  const postprocessBlock = findBlock(blocks, 'postprocess');
  const opfsBlock = findBlock(blocks, 'opfs-cache');

  const filePaths: string[] = [
    'package.json',
    'vite.config.js',
    'index.html',
    `src/main.${me}`,
    'src/App.vue',
    'src/App.css',
  ];

  // Include input lib module only for non-file input modes
  if (inputBlock?.code) {
    filePaths.push(`src/lib/input.${le}`);
  }

  filePaths.push(
    `src/lib/preprocess.${le}`,
    `src/lib/inference.${le}`,
    `src/lib/postprocess.${le}`,
    'README.md',
  );

  const files: GeneratedFile[] = [
    { path: 'package.json', content: emitPackageJson(config, blocks) },
    { path: 'vite.config.js', content: emitViteConfig() },
    { path: 'index.html', content: emitIndexHtml(config) },
    { path: `src/main.${me}`, content: emitMain(config) },
    { path: 'src/App.vue', content: emitApp(config) },
    { path: 'src/App.css', content: emitAppCssFile(config) },
  ];

  if (inputBlock?.code) {
    files.push({ path: `src/lib/input.${le}`, content: toLibModule(inputBlock) });
  }

  // Prepend OPFS cache utilities to inference module when offline mode is enabled
  const inferenceContent = opfsBlock?.code
    ? `${opfsBlock.code}\n\n${toLibModule(inferenceBlock)}`
    : toLibModule(inferenceBlock);

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
