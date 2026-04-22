/**
 * SvelteKit framework emitter (Layer 2).
 *
 * Produces a SvelteKit project:
 *   package.json, svelte.config.js, vite.config.js,
 *   src/app.css, src/app.html,
 *   src/routes/+page.svelte, src/routes/+layout.svelte,
 *   src/lib/input.{ts|js} (if non-file),
 *   src/lib/preprocess.{ts|js}, src/lib/inference.{ts|js},
 *   src/lib/postprocess.{ts|js}, README.md
 *
 * Uses Svelte 5 with SvelteKit. All ML inference runs client-side.
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
  buildFooterText,
  getModelPath,
} from './shared.js';

const libExt = (config: ResolvedConfig) => (config.lang === 'ts' ? 'ts' : 'js');

function emitPackageJson(config: ResolvedConfig, blocks: CodeBlock[]): string {
  const deps: Record<string, string> = {};
  for (const imp of collectImports(blocks)) {
    if (imp === 'onnxruntime-web') deps[imp] = '^1.21.0';
    else deps[imp] = 'latest';
  }

  return JSON.stringify(
    {
      name: config.modelName,
      private: true,
      version: '0.0.0',
      type: 'module',
      scripts: {
        dev: 'vite dev',
        build: 'vite build',
        preview: 'vite preview',
      },
      dependencies: deps,
      devDependencies: {
        '@sveltejs/adapter-static': '^3.0.0',
        '@sveltejs/kit': '^2.0.0',
        '@sveltejs/vite-plugin-svelte': '^4.0.0',
        svelte: '^5.0.0',
        vite: '^6.0.0',
        ...(config.lang === 'ts' ? { typescript: '^5.7.0' } : {}),
      },
    },
    null,
    2,
  );
}

function emitSvelteConfig(): string {
  return `import adapter from '@sveltejs/adapter-static';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  kit: {
    adapter: adapter({
      fallback: 'index.html',
    }),
  },
};

export default config;
`;
}

function emitViteConfig(): string {
  return `import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [sveltekit()],
});
`;
}

function emitAppHtml(config: ResolvedConfig): string {
  const taskLabel = getTaskLabel(config.task);
  return `<!doctype html>
<html lang="en" data-theme="${config.theme}">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>${config.modelName} — ${taskLabel}</title>
    %sveltekit.head%
  </head>
  <body>
    %sveltekit.body%
  </body>
</html>
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

function emitAppCss(config: ResolvedConfig): string {
  const extra = needsExtendedCSS(config) ? emitExtendedCSS() : '';
  return `${emitDesignSystemCSS(config)}\n\n${emitAppCSS()}${extra}`;
}

function emitLayoutSvelte(): string {
  return `<script>
  import '../app.css';
  let { children } = $props();
</script>

{@render children()}
`;
}

// ---- Svelte page emitters by task/input ----

function emitFileClassificationPageSvelte(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script${tsLang}>
  import { onMount } from 'svelte';
  import { createSession, runInference, getBackendLabel } from '$lib/inference.${le}';
  import { preprocessImage } from '$lib/preprocess.${le}';
  import { postprocessResults } from '$lib/postprocess.${le}';

  const MODEL_PATH = '${getModelPath(config, '')}';

  let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = $state(null);
  let results${t ? ': { indices: number[]; values: number[] } | null' : ''} = $state(null);
  let imageUrl${t ? ': string | null' : ''} = $state(null);
  let status${t ? ': string' : ''} = $state('Loading model...');
  let dragOver = $state(false);
  let fileInput${t ? ': HTMLInputElement' : ''};

  onMount(() => {
    createSession(MODEL_PATH).then((s) => {
      session = s;
      status = '${config.modelName} \\u00b7 Ready';
    }).catch((e) => { status = 'Failed to load model'; console.error('Model load error:', e); });
  });

  async function processImage(file${t ? ': File' : ''}) {
    if (!file.type.startsWith('image/')) { status = 'Unsupported file type.'; return; }
    const url = URL.createObjectURL(file);
    imageUrl = url; results = null;
    const img = new Image(); img.src = url;
    await new Promise((resolve) => { img.onload = resolve; });
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d')${t ? '!' : ''}; ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    if (!session) { status = 'Model not loaded yet.'; return; }
    status = '${config.modelName} \\u00b7 Processing...';
    const start = performance.now();
    const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
    const output = await runInference(session, inputTensor);
    const r = postprocessResults(output);
    const elapsed = (performance.now() - start).toFixed(1);
    status = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`;
    results = r;
    URL.revokeObjectURL(url);
  }

  function reset() { imageUrl = null; results = null; if (fileInput) fileInput.value = ''; }
</script>

<a href="#results" class="skip-link">Skip to results</a>
<main>
  <h1>${config.modelName} — ${taskLabel}</h1>
  <div class="container">
    <div>
      {#if !imageUrl}
        <div class="drop-zone" class:drag-over={dragOver} role="button" tabindex="0"
          aria-label="Drop an image here or click to browse for ${taskLabel.toLowerCase()}"
          onclick={() => fileInput?.click()}
          onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput?.click(); } }}
          ondragover={(e) => { e.preventDefault(); dragOver = true; }}
          ondragleave={() => { dragOver = false; }}
          ondrop={(e) => { e.preventDefault(); dragOver = false; const f = e.dataTransfer?.files[0]; if (f) processImage(f); }}>
          <p>Drop an image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input bind:this={fileInput} type="file" accept="image/*" style="position:absolute;left:-9999px;width:1px;height:1px;opacity:0;" aria-hidden="true" tabindex="-1" onchange={(e) => { const f = e.currentTarget.files?.[0]; if (f) processImage(f); }} />
        </div>
      {:else}
        <div class="preview">
          <img src={imageUrl} alt="Selected image for classification" />
          <button class="change-btn" onclick={reset}>Choose another image</button>
        </div>
      {/if}
    </div>
    <div id="results" class="results" role="status" aria-live="polite" aria-atomic="true">
      {#if results}
        {#each results.indices as idx, i}
          {#if results.values[i] >= 0.01}
            {@const pct = (results.values[i] * 100).toFixed(1)}
            {@const maxVal = results.values[0] || 1}
            <div class="result-row" class:top-result={i === 0} tabindex="0" aria-label="Class {idx}: {pct} percent">
              <span class="result-label">Class {idx}</span>
              <div class="result-bar-container">
                <div class="result-bar" style="width: {(results.values[i] / maxVal) * 100}%"></div>
              </div>
              <span class="result-pct">{pct}%</span>
            </div>
          {/if}
        {/each}
      {/if}
    </div>
  </div>
</main>
<aside class="status-bar"><span>{status}</span></aside>
<div class="footer">${buildFooterText(config)}</div>
`;
}

function emitFileDetectionPageSvelte(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';
  const outputShape = config.modelMeta.outputs[0]?.shape ?? [1, 84, 8400];
  const numAttributes = outputShape[1] ?? 84;
  const numAnchors = outputShape[2] ?? 8400;

  const boxType = t ? '\n  interface Box { x: number; y: number; width: number; height: number; classIndex: number; score: number; }' : '';

  return `<script${tsLang}>
  import { onMount } from 'svelte';
  import { createSession, runInference, getBackendLabel } from '$lib/inference.${le}';
  import { preprocessImage } from '$lib/preprocess.${le}';
  import { postprocessDetections } from '$lib/postprocess.${le}';

  const MODEL_PATH = '${getModelPath(config, '')}';
  const NUM_ATTRIBUTES = ${numAttributes};
  const NUM_ANCHORS = ${numAnchors};
  const COLORS = [[56,189,248],[249,115,22],[34,197,94],[168,85,247],[251,191,36],[239,68,68],[20,184,166],[236,72,153],[99,102,241],[163,230,53]];
${boxType}

  let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = $state(null);
  let boxes${t ? ': Box[] | null' : ''} = $state(null);
  let imageUrl${t ? ': string | null' : ''} = $state(null);
  let status${t ? ': string' : ''} = $state('Loading model...');
  let dragOver = $state(false);
  let imgSize = $state({ w: 0, h: 0 });
  let fileInput${t ? ': HTMLInputElement' : ''};
  let overlay${t ? ': HTMLCanvasElement' : ''};

  onMount(() => {
    createSession(MODEL_PATH).then((s) => {
      session = s;
      status = '${config.modelName} \\u00b7 Ready';
    }).catch((e) => { status = 'Failed to load model'; console.error('Model load error:', e); });
  });

  $effect(() => {
    if (!boxes || !overlay) return;
    overlay.width = imgSize.w; overlay.height = imgSize.h;
    const ctx = overlay.getContext('2d')${t ? '!' : ''};
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    const scale = imgSize.w / ${config.preprocess.imageSize};
    for (const box of boxes) {
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
    if (!file.type.startsWith('image/')) { status = 'Unsupported file type.'; return; }
    const url = URL.createObjectURL(file);
    imageUrl = url; boxes = null;
    const img = new Image(); img.src = url;
    await new Promise((resolve) => { img.onload = resolve; });
    imgSize = { w: img.naturalWidth, h: img.naturalHeight };
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d')${t ? '!' : ''}; ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    if (!session) { status = 'Model not loaded yet.'; return; }
    status = '${config.modelName} \\u00b7 Processing...';
    const start = performance.now();
    const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
    const output = await runInference(session, inputTensor);
    const detected = postprocessDetections(output, NUM_ANCHORS, NUM_ATTRIBUTES);
    const elapsed = (performance.now() - start).toFixed(1);
    status = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`;
    boxes = detected;
    URL.revokeObjectURL(url);
  }

  function reset() { imageUrl = null; boxes = null; if (fileInput) fileInput.value = ''; }
</script>

<a href="#results" class="skip-link">Skip to results</a>
<main>
  <h1>${config.modelName} — ${taskLabel}</h1>
  <div class="container">
    <div>
      {#if !imageUrl}
        <div class="drop-zone" class:drag-over={dragOver} role="button" tabindex="0"
          aria-label="Drop an image here or click to browse for ${taskLabel.toLowerCase()}"
          onclick={() => fileInput?.click()}
          onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput?.click(); } }}
          ondragover={(e) => { e.preventDefault(); dragOver = true; }}
          ondragleave={() => { dragOver = false; }}
          ondrop={(e) => { e.preventDefault(); dragOver = false; const f = e.dataTransfer?.files[0]; if (f) processImage(f); }}>
          <p>Drop an image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input bind:this={fileInput} type="file" accept="image/*" style="position:absolute;left:-9999px;width:1px;height:1px;opacity:0;" aria-hidden="true" tabindex="-1" onchange={(e) => { const f = e.currentTarget.files?.[0]; if (f) processImage(f); }} />
        </div>
      {:else}
        <div class="preview">
          <div class="preview-wrapper">
            <img src={imageUrl} alt="Selected image for detection" />
            <canvas bind:this={overlay} class="overlay-canvas"></canvas>
          </div>
          <button class="change-btn" onclick={reset}>Choose another image</button>
        </div>
      {/if}
    </div>
    <div id="results" class="results" role="status" aria-live="polite" aria-atomic="true">
      {#if boxes && boxes.length === 0}
        <p>No detections found.</p>
      {/if}
      {#if boxes}
        {#each boxes as box, i}
          <div class="result-row" tabindex="0" aria-label="Class {box.classIndex}: {(box.score * 100).toFixed(0)} percent">
            <span class="result-label">Class {box.classIndex} ({(box.score * 100).toFixed(0)}%)</span>
          </div>
        {/each}
      {/if}
    </div>
  </div>
</main>
<aside class="status-bar"><span>{status}</span></aside>
<div class="footer">${buildFooterText(config)}</div>
`;
}

function emitFileSegmentationPageSvelte(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';
  const outputShape = config.modelMeta.outputs[0]?.shape ?? [1, 21, 512, 512];
  const numClasses = outputShape[1] ?? 21;
  const maskH = outputShape[2] ?? 512;
  const maskW = outputShape[3] ?? 512;

  return `<script${tsLang}>
  import { onMount } from 'svelte';
  import { createSession, runInference, getBackendLabel } from '$lib/inference.${le}';
  import { preprocessImage } from '$lib/preprocess.${le}';
  import { postprocessSegmentation } from '$lib/postprocess.${le}';

  const MODEL_PATH = '${getModelPath(config, '')}';
  const NUM_CLASSES = ${numClasses};
  const MASK_H = ${maskH};
  const MASK_W = ${maskW};
  const COLORS = [[56,189,248],[249,115,22],[34,197,94],[168,85,247],[251,191,36],[239,68,68],[20,184,166],[236,72,153],[99,102,241],[163,230,53]];

  let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = $state(null);
  let mask${t ? ': Uint8Array | null' : ''} = $state(null);
  let imageUrl${t ? ': string | null' : ''} = $state(null);
  let status${t ? ': string' : ''} = $state('Loading model...');
  let dragOver = $state(false);
  let imgSize = $state({ w: 0, h: 0 });
  let fileInput${t ? ': HTMLInputElement' : ''};
  let overlay${t ? ': HTMLCanvasElement' : ''};

  onMount(() => {
    createSession(MODEL_PATH).then((s) => {
      session = s;
      status = '${config.modelName} \\u00b7 Ready';
    }).catch((e) => { status = 'Failed to load model'; console.error('Model load error:', e); });
  });

  $effect(() => {
    if (!mask || !overlay) return;
    overlay.width = imgSize.w; overlay.height = imgSize.h;
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
    const ctx = overlay.getContext('2d')${t ? '!' : ''};
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    ctx.drawImage(maskCanvas, 0, 0, imgSize.w, imgSize.h);
  });

  async function processImage(file${t ? ': File' : ''}) {
    if (!file.type.startsWith('image/')) { status = 'Unsupported file type.'; return; }
    const url = URL.createObjectURL(file);
    imageUrl = url; mask = null;
    const img = new Image(); img.src = url;
    await new Promise((resolve) => { img.onload = resolve; });
    imgSize = { w: img.naturalWidth, h: img.naturalHeight };
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d')${t ? '!' : ''}; ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    if (!session) { status = 'Model not loaded yet.'; return; }
    status = '${config.modelName} \\u00b7 Processing...';
    const start = performance.now();
    const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
    const output = await runInference(session, inputTensor);
    const m = postprocessSegmentation(output, NUM_CLASSES, MASK_H, MASK_W);
    const elapsed = (performance.now() - start).toFixed(1);
    status = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`;
    mask = m;
    URL.revokeObjectURL(url);
  }

  function reset() { imageUrl = null; mask = null; if (fileInput) fileInput.value = ''; }
</script>

<a href="#results" class="skip-link">Skip to results</a>
<main>
  <h1>${config.modelName} — ${taskLabel}</h1>
  <div class="container">
    <div>
      {#if !imageUrl}
        <div class="drop-zone" class:drag-over={dragOver} role="button" tabindex="0"
          aria-label="Drop an image here or click to browse for ${taskLabel.toLowerCase()}"
          onclick={() => fileInput?.click()}
          onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput?.click(); } }}
          ondragover={(e) => { e.preventDefault(); dragOver = true; }}
          ondragleave={() => { dragOver = false; }}
          ondrop={(e) => { e.preventDefault(); dragOver = false; const f = e.dataTransfer?.files[0]; if (f) processImage(f); }}>
          <p>Drop an image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input bind:this={fileInput} type="file" accept="image/*" style="position:absolute;left:-9999px;width:1px;height:1px;opacity:0;" aria-hidden="true" tabindex="-1" onchange={(e) => { const f = e.currentTarget.files?.[0]; if (f) processImage(f); }} />
        </div>
      {:else}
        <div class="preview">
          <div class="preview-wrapper">
            <img src={imageUrl} alt="Selected image for segmentation" />
            <canvas bind:this={overlay} class="overlay-canvas"></canvas>
          </div>
          <button class="change-btn" onclick={reset}>Choose another image</button>
        </div>
      {/if}
    </div>
    <div id="results" class="results" role="status" aria-live="polite" aria-atomic="true">
    </div>
  </div>
</main>
<aside class="status-bar"><span>{status}</span></aside>
<div class="footer">${buildFooterText(config)}</div>
`;
}

function emitFileFeatureExtractionPageSvelte(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script${tsLang}>
  import { onMount } from 'svelte';
  import { createSession, runInference, getBackendLabel } from '$lib/inference.${le}';
  import { preprocessImage } from '$lib/preprocess.${le}';
  import { postprocessEmbeddings } from '$lib/postprocess.${le}';

  const MODEL_PATH = '${getModelPath(config, '')}';

  let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = $state(null);
  let embedding${t ? ': { dims: number; norm: string; first5: string } | null' : ''} = $state(null);
  let imageUrl${t ? ': string | null' : ''} = $state(null);
  let status${t ? ': string' : ''} = $state('Loading model...');
  let dragOver = $state(false);
  let fileInput${t ? ': HTMLInputElement' : ''};

  onMount(() => {
    createSession(MODEL_PATH).then((s) => {
      session = s;
      status = '${config.modelName} \\u00b7 Ready';
    }).catch((e) => { status = 'Failed to load model'; console.error('Model load error:', e); });
  });

  async function processImage(file${t ? ': File' : ''}) {
    if (!file.type.startsWith('image/')) { status = 'Unsupported file type.'; return; }
    const url = URL.createObjectURL(file);
    imageUrl = url; embedding = null;
    const img = new Image(); img.src = url;
    await new Promise((resolve) => { img.onload = resolve; });
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d')${t ? '!' : ''}; ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    if (!session) { status = 'Model not loaded yet.'; return; }
    status = '${config.modelName} \\u00b7 Processing...';
    const start = performance.now();
    const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
    const output = await runInference(session, inputTensor);
    const emb = postprocessEmbeddings(output);
    let norm = 0;
    for (let i = 0; i < emb.length; i++) { norm += emb[i] * emb[i]; }
    norm = Math.sqrt(norm);
    const first5 = Array.from(emb.slice(0, 5)).map((v${t ? ': number' : ''}) => v.toFixed(4)).join(', ');
    const elapsed = (performance.now() - start).toFixed(1);
    status = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`;
    embedding = { dims: emb.length, norm: norm.toFixed(4), first5 };
    URL.revokeObjectURL(url);
  }

  function reset() { imageUrl = null; embedding = null; if (fileInput) fileInput.value = ''; }
</script>

<a href="#results" class="skip-link">Skip to results</a>
<main>
  <h1>${config.modelName} — ${taskLabel}</h1>
  <div class="container">
    <div>
      {#if !imageUrl}
        <div class="drop-zone" class:drag-over={dragOver} role="button" tabindex="0"
          aria-label="Drop an image here or click to browse for ${taskLabel.toLowerCase()}"
          onclick={() => fileInput?.click()}
          onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput?.click(); } }}
          ondragover={(e) => { e.preventDefault(); dragOver = true; }}
          ondragleave={() => { dragOver = false; }}
          ondrop={(e) => { e.preventDefault(); dragOver = false; const f = e.dataTransfer?.files[0]; if (f) processImage(f); }}>
          <p>Drop an image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input bind:this={fileInput} type="file" accept="image/*" style="position:absolute;left:-9999px;width:1px;height:1px;opacity:0;" aria-hidden="true" tabindex="-1" onchange={(e) => { const f = e.currentTarget.files?.[0]; if (f) processImage(f); }} />
        </div>
      {:else}
        <div class="preview">
          <img src={imageUrl} alt="Selected image for feature extraction" />
          <button class="change-btn" onclick={reset}>Choose another image</button>
        </div>
      {/if}
    </div>
    <div id="results" class="results embedding-info" role="status" aria-live="polite" aria-atomic="true">
      {#if embedding}
        <p><strong>Dimensions:</strong> {embedding.dims}</p>
        <p><strong>L2 Norm:</strong> {embedding.norm}</p>
        <p><strong>First 5 values:</strong> [{embedding.first5}, ...]</p>
      {/if}
    </div>
  </div>
</main>
<aside class="status-bar"><span>{status}</span></aside>
<div class="footer">${buildFooterText(config)}</div>
`;
}

// ---- Fill Mask Page ----

function emitFillMaskSvelte(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script${tsLang}>
  import { onMount } from 'svelte';
  import { createSession, runInference, getBackendLabel } from '$lib/inference.${le}';
  import { loadTokenizer, tokenizeText } from '$lib/preprocess.${le}';
  import { postprocessFillMask } from '$lib/postprocess.${le}';

  const MODEL_PATH = '${getModelPath(config, '')}';
  const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

  let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = $state(null);
  let tokenizer${t ? ': Awaited<ReturnType<typeof loadTokenizer>> | null' : ''} = $state(null);
  let predictions${t ? ': Array<{ token: string; score: number }> | null' : ''} = $state(null);
  let text = $state('The capital of France is [MASK].');
  let status${t ? ': string' : ''} = $state('Loading model and tokenizer...');

  onMount(() => {
    Promise.all([
      createSession(MODEL_PATH),
      loadTokenizer(TOKENIZER_PATH),
    ]).then(([s, tok]) => {
      session = s;
      tokenizer = tok;
      status = '${config.modelName} \\u00b7 Ready';
    }).catch((e) => { status = 'Failed to load model or tokenizer'; console.error('Load error:', e); });
  });

  async function handlePredict() {
    const trimmed = text.trim();
    if (!trimmed || !session || !tokenizer) return;

    status = '${config.modelName} \\u00b7 Processing...';
    const start = performance.now();

    const { inputIds } = tokenizeText(tokenizer, trimmed);
    const output = await runInference(session, inputIds);
    const results = postprocessFillMask(output, inputIds, tokenizer);

    const elapsed = (performance.now() - start).toFixed(1);
    status = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`;
    predictions = results;
  }
</script>

<a href="#results" class="skip-link">Skip to results</a>
<main>
  <h1>${config.modelName} — ${taskLabel}</h1>
  <div>
    <label for="textInput">Enter text with [MASK] token</label>
    <textarea id="textInput" rows="4" bind:value={text} aria-label="Text with MASK token"></textarea>
    <button class="primary-btn" onclick={handlePredict} disabled={!session || !tokenizer} aria-label="Predict masked token">Predict</button>
  </div>
  <div id="results" class="mask-predictions" role="status" aria-live="polite" aria-atomic="true">
    {#if predictions}
      {#each predictions as p, i}
        <div class="mask-prediction" tabindex="0" aria-label="{p.token}: {(p.score * 100).toFixed(1)} percent">
          <span class="token">{p.token}</span>
          <span class="prob">{(p.score * 100).toFixed(1)}%</span>
        </div>
      {/each}
    {/if}
  </div>
</main>
<aside class="status-bar"><span>{status}</span></aside>
<div class="footer">${buildFooterText(config)}</div>
`;
}

// ---- Sentence Similarity Page ----

function emitSentenceSimilaritySvelte(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script${tsLang}>
  import { onMount } from 'svelte';
  import { createSession, runInference, getBackendLabel } from '$lib/inference.${le}';
  import { loadTokenizer, tokenizeText } from '$lib/preprocess.${le}';
  import { cosineSimilarity } from '$lib/postprocess.${le}';

  const MODEL_PATH = '${getModelPath(config, '')}';
  const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

  let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = $state(null);
  let tokenizer${t ? ': Awaited<ReturnType<typeof loadTokenizer>> | null' : ''} = $state(null);
  let scores${t ? ': Array<{ sentence: string; score: number }> | null' : ''} = $state(null);
  let source = $state('The weather is lovely today.');
  let comparisons = $state('It is a beautiful day.\\nThe sun is shining bright.\\nI need to buy groceries.');
  let status${t ? ': string' : ''} = $state('Loading model and tokenizer...');

  onMount(() => {
    Promise.all([
      createSession(MODEL_PATH),
      loadTokenizer(TOKENIZER_PATH),
    ]).then(([s, tok]) => {
      session = s;
      tokenizer = tok;
      status = '${config.modelName} \\u00b7 Ready';
    }).catch((e) => { status = 'Failed to load model or tokenizer'; console.error('Load error:', e); });
  });

  async function handleCompare() {
    if (!session || !tokenizer || !source.trim()) return;

    status = '${config.modelName} \\u00b7 Computing...';
    const start = performance.now();

    const { inputIds: srcIds } = tokenizeText(tokenizer, source);
    const srcEmb = await runInference(session, srcIds);
    const lines = comparisons.split('\\n').filter(Boolean);
    const results${t ? ': Array<{ sentence: string; score: number }>' : ''} = [];
    for (const line of lines) {
      const { inputIds: cmpIds } = tokenizeText(tokenizer, line);
      const cmpEmb = await runInference(session, cmpIds);
      results.push({ sentence: line, score: cosineSimilarity(srcEmb, cmpEmb) });
    }

    const elapsed = (performance.now() - start).toFixed(1);
    status = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`;
    scores = results;
  }
</script>

<a href="#results" class="skip-link">Skip to results</a>
<main>
  <h1>${config.modelName} — ${taskLabel}</h1>
  <div>
    <label for="source">Source sentence</label>
    <textarea id="source" rows="2" bind:value={source} aria-label="Source sentence"></textarea>
    <label for="compare">Sentences to compare (one per line)</label>
    <textarea id="compare" rows="4" bind:value={comparisons} aria-label="Comparison sentences"></textarea>
    <button class="primary-btn" onclick={handleCompare} disabled={!session || !tokenizer} aria-label="Compare sentences">Compare</button>
  </div>
  <div id="results" class="similarity-pairs" role="status" aria-live="polite" aria-atomic="true">
    {#if scores}
      {#each scores as s, i}
        <div class="similarity-score" tabindex="0" aria-label="{s.sentence}: {s.score.toFixed(4)}">
          <span>{s.sentence}</span>
          <span class="value">{s.score.toFixed(4)}</span>
        </div>
      {/each}
    {/if}
  </div>
</main>
<aside class="status-bar"><span>{status}</span></aside>
<div class="footer">${buildFooterText(config)}</div>
`;
}

// ---- Depth Estimation Page ----

function emitDepthEstimationSvelte(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script${tsLang}>
  import { onMount } from 'svelte';
  import { createSession, runInference, getBackendLabel } from '$lib/inference.${le}';
  import { preprocessImage } from '$lib/preprocess.${le}';
  import { postprocessDepth } from '$lib/postprocess.${le}';

  const MODEL_PATH = '${getModelPath(config, '')}';

  let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = $state(null);
  let imageUrl${t ? ': string | null' : ''} = $state(null);
  let status${t ? ': string' : ''} = $state('Loading model...');
  let dragOver = $state(false);
  let fileInput${t ? ': HTMLInputElement' : ''};
  let depthCanvas${t ? ': HTMLCanvasElement' : ''};

  onMount(() => {
    createSession(MODEL_PATH).then((s) => {
      session = s;
      status = '${config.modelName} \\u00b7 Ready';
    }).catch((e) => { status = 'Failed to load model'; console.error('Model load error:', e); });
  });

  async function processImage(file${t ? ': File' : ''}) {
    if (!file.type.startsWith('image/')) { status = 'Unsupported file type.'; return; }
    const url = URL.createObjectURL(file);
    imageUrl = url;
    const img = new Image(); img.src = url;
    await new Promise((resolve) => { img.onload = resolve; });
    if (!session) { status = 'Model not loaded yet.'; return; }
    status = '${config.modelName} \\u00b7 Processing...';
    const start = performance.now();
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d')${t ? '!' : ''}; ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
    const output = await runInference(session, inputTensor);
    const depthMap = postprocessDepth(output);
    if (depthCanvas) {
      depthCanvas.width = img.naturalWidth; depthCanvas.height = img.naturalHeight;
      const dCtx = depthCanvas.getContext('2d')${t ? '!' : ''};
      const imgData = dCtx.createImageData(depthCanvas.width, depthCanvas.height);
      for (let i = 0; i < depthMap.length; i++) {
        const v = depthMap[i];
        imgData.data[i * 4] = v; imgData.data[i * 4 + 1] = v; imgData.data[i * 4 + 2] = v; imgData.data[i * 4 + 3] = 255;
      }
      dCtx.putImageData(imgData, 0, 0);
    }
    const elapsed = (performance.now() - start).toFixed(1);
    status = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`;
    URL.revokeObjectURL(url);
  }

  function reset() { imageUrl = null; if (fileInput) fileInput.value = ''; }
</script>

<a href="#results" class="skip-link">Skip to results</a>
<main>
  <h1>${config.modelName} — ${taskLabel}</h1>
  <div class="container">
    <div>
      {#if !imageUrl}
        <div class="drop-zone" class:drag-over={dragOver} role="button" tabindex="0"
          aria-label="Drop an image here or click to browse for ${taskLabel.toLowerCase()}"
          onclick={() => fileInput?.click()}
          onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput?.click(); } }}
          ondragover={(e) => { e.preventDefault(); dragOver = true; }}
          ondragleave={() => { dragOver = false; }}
          ondrop={(e) => { e.preventDefault(); dragOver = false; const f = e.dataTransfer?.files[0]; if (f) processImage(f); }}>
          <p>Drop an image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input bind:this={fileInput} type="file" accept="image/*" style="position:absolute;left:-9999px;width:1px;height:1px;opacity:0;" aria-hidden="true" tabindex="-1" onchange={(e) => { const f = e.currentTarget.files?.[0]; if (f) processImage(f); }} />
        </div>
      {:else}
        <div class="preview">
          <img src={imageUrl} alt="Source image for depth estimation" />
          <button class="change-btn" onclick={reset}>Choose another image</button>
        </div>
      {/if}
    </div>
    <div id="results"><canvas bind:this={depthCanvas} class="depth-canvas"></canvas></div>
  </div>
</main>
<aside class="status-bar"><span>{status}</span></aside>
<div class="footer">${buildFooterText(config)}</div>
`;
}

// ---- Token Classification (NER) Page ----

function emitTokenClassificationSvelte(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script${tsLang}>
  import { onMount } from 'svelte';
  import { createSession, runInference, getBackendLabel } from '$lib/inference.${le}';
  import { loadTokenizer, tokenizeText } from '$lib/preprocess.${le}';
  import { postprocessTokenClassification } from '$lib/postprocess.${le}';

  const MODEL_PATH = '${getModelPath(config, '')}';
  const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

  let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = $state(null);
  let tokenizer${t ? ': Awaited<ReturnType<typeof loadTokenizer>> | null' : ''} = $state(null);
  let nerHtml${t ? ': string' : ''} = $state('');
  let text = $state('John Smith works at Google in Mountain View, California.');
  let status${t ? ': string' : ''} = $state('Loading model and tokenizer...');

  onMount(() => {
    Promise.all([
      createSession(MODEL_PATH),
      loadTokenizer(TOKENIZER_PATH),
    ]).then(([s, tok]) => {
      session = s;
      tokenizer = tok;
      status = '${config.modelName} \\u00b7 Ready';
    }).catch((e) => { status = 'Failed to load model or tokenizer'; console.error('Load error:', e); });
  });

  async function handleAnalyze() {
    const trimmed = text.trim();
    if (!trimmed || !session || !tokenizer) return;

    status = '${config.modelName} \\u00b7 Processing...';
    const start = performance.now();

    const { inputIds } = tokenizeText(tokenizer, trimmed);
    const output = await runInference(session, inputIds);
    const entities = postprocessTokenClassification(output, inputIds, tokenizer);
    let result = trimmed;
    for (let i = entities.length - 1; i >= 0; i--) {
      const e = entities[i];
      result = result.slice(0, e.start) + '<span class="ner-entity" data-type="' + e.type + '">' + result.slice(e.start, e.end) + '</span>' + result.slice(e.end);
    }
    nerHtml = result;

    const elapsed = (performance.now() - start).toFixed(1);
    status = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`;
  }
</script>

<a href="#results" class="skip-link">Skip to results</a>
<main>
  <h1>${config.modelName} — ${taskLabel}</h1>
  <div>
    <label for="textInput">Enter text to analyze</label>
    <textarea id="textInput" rows="4" bind:value={text} aria-label="Text to analyze for named entities"></textarea>
    <button class="primary-btn" onclick={handleAnalyze} disabled={!session || !tokenizer} aria-label="Analyze entities">Analyze</button>
  </div>
  <div id="results" class="ner-output" role="status" aria-live="polite" aria-atomic="true">
    {#if nerHtml}
      {@html nerHtml}
    {:else}
      {text}
    {/if}
  </div>
</main>
<aside class="status-bar"><span>{status}</span></aside>
<div class="footer">${buildFooterText(config)}</div>
`;
}

// ---- Question Answering Page ----

function emitQuestionAnsweringSvelte(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script${tsLang}>
  import { onMount } from 'svelte';
  import { createSession, runInference, getBackendLabel } from '$lib/inference.${le}';
  import { loadTokenizer, tokenizeText } from '$lib/preprocess.${le}';
  import { postprocessQA } from '$lib/postprocess.${le}';

  const MODEL_PATH = '${getModelPath(config, '')}';
  const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

  let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = $state(null);
  let tokenizer${t ? ': Awaited<ReturnType<typeof loadTokenizer>> | null' : ''} = $state(null);
  let answer${t ? ': { answer: string; score: number } | null' : ''} = $state(null);
  let context = $state('The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was constructed from 1887 to 1889.');
  let question = $state('When was the Eiffel Tower built?');
  let status${t ? ': string' : ''} = $state('Loading model and tokenizer...');

  onMount(() => {
    Promise.all([
      createSession(MODEL_PATH),
      loadTokenizer(TOKENIZER_PATH),
    ]).then(([s, tok]) => {
      session = s;
      tokenizer = tok;
      status = '${config.modelName} \\u00b7 Ready';
    }).catch((e) => { status = 'Failed to load model or tokenizer'; console.error('Load error:', e); });
  });

  async function handleAnswer() {
    if (!session || !tokenizer || !context.trim() || !question.trim()) return;

    status = '${config.modelName} \\u00b7 Processing...';
    const start = performance.now();

    const combined = question + ' [SEP] ' + context;
    const { inputIds } = tokenizeText(tokenizer, combined);
    const output = await runInference(session, inputIds);
    const result = postprocessQA(output, inputIds, tokenizer);

    const elapsed = (performance.now() - start).toFixed(1);
    status = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`;
    answer = result;
  }
</script>

<a href="#results" class="skip-link">Skip to results</a>
<main>
  <h1>${config.modelName} — ${taskLabel}</h1>
  <div>
    <label for="context">Context</label>
    <textarea id="context" rows="4" bind:value={context} aria-label="Context passage"></textarea>
    <label for="question">Question</label>
    <input id="question" type="text" bind:value={question} aria-label="Question" />
    <button class="primary-btn" onclick={handleAnswer} disabled={!session || !tokenizer} aria-label="Get answer">Answer</button>
  </div>
  <div id="results" class="qa-answer" role="status" aria-live="polite" aria-atomic="true">
    {#if answer}
      <div>{answer.answer}</div>
      <div class="score">Confidence: {(answer.score * 100).toFixed(1)}%</div>
    {/if}
  </div>
</main>
<aside class="status-bar"><span>{status}</span></aside>
<div class="footer">${buildFooterText(config)}</div>
`;
}

// ---- Summarization Page ----

function emitSummarizationSvelte(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script${tsLang}>
  import { onMount } from 'svelte';
  import { createSession, runInference, getBackendLabel } from '$lib/inference.${le}';
  import { loadTokenizer, tokenizeText } from '$lib/preprocess.${le}';
  import { postprocessSummarization } from '$lib/postprocess.${le}';

  const MODEL_PATH = '${getModelPath(config, '')}';
  const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

  let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = $state(null);
  let tokenizer${t ? ': Awaited<ReturnType<typeof loadTokenizer>> | null' : ''} = $state(null);
  let output = $state('');
  let text = $state('Artificial intelligence has transformed many industries. Machine learning models can now process natural language, recognize images, and generate creative content.');
  let processing = $state(false);
  let status${t ? ': string' : ''} = $state('Loading model and tokenizer...');

  onMount(() => {
    Promise.all([
      createSession(MODEL_PATH),
      loadTokenizer(TOKENIZER_PATH),
    ]).then(([s, tok]) => {
      session = s;
      tokenizer = tok;
      status = '${config.modelName} \\u00b7 Ready';
    }).catch((e) => { status = 'Failed to load model or tokenizer'; console.error('Load error:', e); });
  });

  async function handleSummarize() {
    const trimmed = text.trim();
    if (!trimmed || !session || !tokenizer || processing) return;

    processing = true;
    status = '${config.modelName} \\u00b7 Summarizing...';
    const start = performance.now();

    const { inputIds } = tokenizeText(tokenizer, trimmed);
    const raw = await runInference(session, inputIds);
    const summary = postprocessSummarization(raw, tokenizer, 128, 1);

    const elapsed = (performance.now() - start).toFixed(1);
    status = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`;
    output = summary;
    processing = false;
  }
</script>

<a href="#output" class="skip-link">Skip to output</a>
<main>
  <h1>${config.modelName} — ${taskLabel}</h1>
  <div>
    <label for="textInput">Enter text to summarize</label>
    <textarea id="textInput" rows="6" bind:value={text} disabled={processing} aria-label="Text to summarize"></textarea>
    <button class="primary-btn" onclick={handleSummarize} disabled={!session || !tokenizer || processing} aria-label="Summarize text">{processing ? 'Summarizing...' : 'Summarize'}</button>
  </div>
  <div id="output" class="generation-output" role="status" aria-live="polite" aria-atomic="true">{output || '(summary will appear here)'}</div>
</main>
<aside class="status-bar"><span>{status}</span></aside>
<div class="footer">${buildFooterText(config)}</div>
`;
}

// ---- Translation Page ----

function emitTranslationSvelte(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script${tsLang}>
  import { onMount } from 'svelte';
  import { createSession, runInference, getBackendLabel } from '$lib/inference.${le}';
  import { loadTokenizer, tokenizeText } from '$lib/preprocess.${le}';
  import { postprocessTranslation } from '$lib/postprocess.${le}';

  const MODEL_PATH = '${getModelPath(config, '')}';
  const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

  let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = $state(null);
  let tokenizer${t ? ': Awaited<ReturnType<typeof loadTokenizer>> | null' : ''} = $state(null);
  let output = $state('');
  let text = $state('Hello, how are you today?');
  let processing = $state(false);
  let status${t ? ': string' : ''} = $state('Loading model and tokenizer...');

  onMount(() => {
    Promise.all([
      createSession(MODEL_PATH),
      loadTokenizer(TOKENIZER_PATH),
    ]).then(([s, tok]) => {
      session = s;
      tokenizer = tok;
      status = '${config.modelName} \\u00b7 Ready';
    }).catch((e) => { status = 'Failed to load model or tokenizer'; console.error('Load error:', e); });
  });

  async function handleTranslate() {
    const trimmed = text.trim();
    if (!trimmed || !session || !tokenizer || processing) return;

    processing = true;
    status = '${config.modelName} \\u00b7 Translating...';
    const start = performance.now();

    const { inputIds } = tokenizeText(tokenizer, trimmed);
    const raw = await runInference(session, inputIds);
    const translation = postprocessTranslation(raw, tokenizer, 128, 1);

    const elapsed = (performance.now() - start).toFixed(1);
    status = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`;
    output = translation;
    processing = false;
  }
</script>

<a href="#output" class="skip-link">Skip to output</a>
<main>
  <h1>${config.modelName} — ${taskLabel}</h1>
  <div>
    <label for="textInput">Enter text to translate</label>
    <textarea id="textInput" rows="4" bind:value={text} disabled={processing} aria-label="Text to translate"></textarea>
    <button class="primary-btn" onclick={handleTranslate} disabled={!session || !tokenizer || processing} aria-label="Translate text">{processing ? 'Translating...' : 'Translate'}</button>
  </div>
  <div id="output" class="generation-output" role="status" aria-live="polite" aria-atomic="true">{output || '(translation will appear here)'}</div>
</main>
<aside class="status-bar"><span>{status}</span></aside>
<div class="footer">${buildFooterText(config)}</div>
`;
}

// ---- Text-to-Text Generation Page ----

function emitText2TextSvelte(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script${tsLang}>
  import { onMount } from 'svelte';
  import { createSession, runInference, getBackendLabel } from '$lib/inference.${le}';
  import { loadTokenizer, tokenizeText } from '$lib/preprocess.${le}';
  import { postprocessText2Text } from '$lib/postprocess.${le}';

  const MODEL_PATH = '${getModelPath(config, '')}';
  const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

  let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = $state(null);
  let tokenizer${t ? ': Awaited<ReturnType<typeof loadTokenizer>> | null' : ''} = $state(null);
  let output = $state('');
  let text = $state('Paraphrase: The house is big and beautiful.');
  let processing = $state(false);
  let status${t ? ': string' : ''} = $state('Loading model and tokenizer...');

  onMount(() => {
    Promise.all([
      createSession(MODEL_PATH),
      loadTokenizer(TOKENIZER_PATH),
    ]).then(([s, tok]) => {
      session = s;
      tokenizer = tok;
      status = '${config.modelName} \\u00b7 Ready';
    }).catch((e) => { status = 'Failed to load model or tokenizer'; console.error('Load error:', e); });
  });

  async function handleRun() {
    const trimmed = text.trim();
    if (!trimmed || !session || !tokenizer || processing) return;

    processing = true;
    status = '${config.modelName} \\u00b7 Processing...';
    const start = performance.now();

    const { inputIds } = tokenizeText(tokenizer, trimmed);
    const raw = await runInference(session, inputIds);
    const result = postprocessText2Text(raw, tokenizer, 128, 1);

    const elapsed = (performance.now() - start).toFixed(1);
    status = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`;
    output = result;
    processing = false;
  }
</script>

<a href="#output" class="skip-link">Skip to output</a>
<main>
  <h1>${config.modelName} — ${taskLabel}</h1>
  <div>
    <label for="textInput">Enter input text</label>
    <textarea id="textInput" rows="4" bind:value={text} disabled={processing} aria-label="Input text"></textarea>
    <button class="primary-btn" onclick={handleRun} disabled={!session || !tokenizer || processing} aria-label="Run text-to-text generation">{processing ? 'Processing...' : 'Run'}</button>
  </div>
  <div id="output" class="generation-output" role="status" aria-live="polite" aria-atomic="true">{output || '(output will appear here)'}</div>
</main>
<aside class="status-bar"><span>{status}</span></aside>
<div class="footer">${buildFooterText(config)}</div>
`;
}

// ---- Conversational Page ----

function emitConversationalSvelte(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script${tsLang}>
  import { onMount } from 'svelte';
  import { createSession, runInference, getBackendLabel } from '$lib/inference.${le}';
  import { loadTokenizer } from '$lib/preprocess.${le}';
  import { postprocessConversational, sampleNextToken } from '$lib/postprocess.${le}';

  const MODEL_PATH = '${getModelPath(config, '')}';
  const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

  let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = $state(null);
  let tokenizer${t ? ': Awaited<ReturnType<typeof loadTokenizer>> | null' : ''} = $state(null);
  let messages${t ? ': Array<{ role: string; text: string }>' : ''} = $state([]);
  let input = $state('');
  let generating = $state(false);
  let status${t ? ': string' : ''} = $state('Loading model and tokenizer...');
  let history${t ? ': string[]' : ''} = [];

  onMount(() => {
    Promise.all([
      createSession(MODEL_PATH),
      loadTokenizer(TOKENIZER_PATH),
    ]).then(([s, tok]) => {
      session = s;
      tokenizer = tok;
      status = '${config.modelName} \\u00b7 Ready';
    }).catch((e) => { status = 'Failed to load model or tokenizer'; console.error('Load error:', e); });
  });

  async function handleSend() {
    if (!session || !tokenizer || !input.trim() || generating) return;
    const userMsg = input.trim();
    input = '';
    messages = [...messages, { role: 'user', text: userMsg }];
    history.push(userMsg);
    generating = true;
    status = '${config.modelName} \\u00b7 Generating...';
    const start = performance.now();
    const prompt = history.join(' ');
    const encoded = tokenizer.encode(prompt);
    let inputIds = encoded.inputIds;
    for (let i = 0; i < 50; i++) {
      const inputBigInt = new BigInt64Array(inputIds.map((id${t ? ': number' : ''}) => BigInt(id)));
      const output = await runInference(session, inputBigInt);
      const vocabSize = tokenizer.getVocabSize();
      const logits = postprocessConversational(output, inputIds.length, vocabSize);
      const nextToken = sampleNextToken(logits);
      if (nextToken === 2) break;
      inputIds = [...inputIds, nextToken];
    }
    const decoded = tokenizer.decode(inputIds);
    const reply = decoded.slice(prompt.length).trim() || '(no response)';
    history.push(reply);
    messages = [...messages, { role: 'bot', text: reply }];
    const elapsed = (performance.now() - start).toFixed(1);
    status = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`;
    generating = false;
  }
</script>

<main>
  <h1>${config.modelName} — ${taskLabel}</h1>
  <div class="chat-messages">
    {#each messages as m, i}
      <div class="chat-msg {m.role}">{m.text}</div>
    {/each}
  </div>
  <div class="chat-input-row">
    <input type="text" bind:value={input}
      onkeydown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
      placeholder="Type a message..." disabled={generating} aria-label="Chat input" />
    <button class="primary-btn" onclick={handleSend} disabled={generating} aria-label="Send message">Send</button>
  </div>
</main>
<aside class="status-bar"><span>{status}</span></aside>
<div class="footer">${buildFooterText(config)}</div>
`;
}

// ---- Table Question Answering Page ----

function emitTableQASvelte(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script${tsLang}>
  import { onMount } from 'svelte';
  import { createSession, runInference, getBackendLabel } from '$lib/inference.${le}';
  import { loadTokenizer, tokenizeText } from '$lib/preprocess.${le}';
  import { postprocessTableQA } from '$lib/postprocess.${le}';

  const MODEL_PATH = '${getModelPath(config, '')}';
  const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

  let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = $state(null);
  let tokenizer${t ? ': Awaited<ReturnType<typeof loadTokenizer>> | null' : ''} = $state(null);
  let answer${t ? ': { answer: string; score: number } | null' : ''} = $state(null);
  let table = $state('Name, Age, City\\nAlice, 30, New York\\nBob, 25, San Francisco');
  let question = $state('Who lives in San Francisco?');
  let status${t ? ': string' : ''} = $state('Loading model and tokenizer...');

  onMount(() => {
    Promise.all([
      createSession(MODEL_PATH),
      loadTokenizer(TOKENIZER_PATH),
    ]).then(([s, tok]) => {
      session = s;
      tokenizer = tok;
      status = '${config.modelName} \\u00b7 Ready';
    }).catch((e) => { status = 'Failed to load model or tokenizer'; console.error('Load error:', e); });
  });

  async function handleAnswer() {
    if (!session || !tokenizer || !table.trim() || !question.trim()) return;

    status = '${config.modelName} \\u00b7 Processing...';
    const start = performance.now();

    const combined = table + ' [SEP] ' + question;
    const { inputIds } = tokenizeText(tokenizer, combined);
    const output = await runInference(session, inputIds);
    const result = postprocessTableQA(output, inputIds, tokenizer);

    const elapsed = (performance.now() - start).toFixed(1);
    status = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`;
    answer = result;
  }
</script>

<a href="#results" class="skip-link">Skip to results</a>
<main>
  <h1>${config.modelName} — ${taskLabel}</h1>
  <div>
    <label for="table">Table data (CSV format)</label>
    <textarea id="table" rows="4" bind:value={table} aria-label="Table data in CSV format"></textarea>
    <label for="question">Question</label>
    <input id="question" type="text" bind:value={question} aria-label="Question about the table" />
    <button class="primary-btn" onclick={handleAnswer} disabled={!session || !tokenizer} aria-label="Get answer">Answer</button>
  </div>
  <div id="results" class="qa-answer" role="status" aria-live="polite" aria-atomic="true">
    {#if answer}
      <div>{answer.answer}</div>
      <div class="score">Confidence: {(answer.score * 100).toFixed(1)}%</div>
    {/if}
  </div>
</main>
<aside class="status-bar"><span>{status}</span></aside>
<div class="footer">${buildFooterText(config)}</div>
`;
}

// ---- Image-to-Text Page ----

function emitImageToTextSvelte(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script${tsLang}>
  import { onMount } from 'svelte';
  import { createSession, runInference, getBackendLabel } from '$lib/inference.${le}';
  import { preprocessImage } from '$lib/preprocess.${le}';
  import { postprocessImageToText } from '$lib/postprocess.${le}';

  const MODEL_PATH = '${getModelPath(config, '')}';

  let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = $state(null);
  let caption = $state('');
  let imageUrl${t ? ': string | null' : ''} = $state(null);
  let status${t ? ': string' : ''} = $state('Loading model...');
  let dragOver = $state(false);
  let fileInput${t ? ': HTMLInputElement' : ''};

  onMount(() => {
    createSession(MODEL_PATH).then((s) => {
      session = s;
      status = '${config.modelName} \\u00b7 Ready';
    }).catch((e) => { status = 'Failed to load model'; console.error('Model load error:', e); });
  });

  async function processImage(file${t ? ': File' : ''}) {
    if (!file.type.startsWith('image/')) { status = 'Unsupported file type.'; return; }
    const url = URL.createObjectURL(file);
    imageUrl = url; caption = '';
    const img = new Image(); img.src = url;
    await new Promise((resolve) => { img.onload = resolve; });
    if (!session) { status = 'Model not loaded yet.'; return; }
    status = '${config.modelName} \\u00b7 Generating caption...';
    const start = performance.now();
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d')${t ? '!' : ''}; ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
    const output = await runInference(session, inputTensor);
    const text = postprocessImageToText(output);
    const elapsed = (performance.now() - start).toFixed(1);
    status = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`;
    caption = text;
    URL.revokeObjectURL(url);
  }

  function reset() { imageUrl = null; caption = ''; if (fileInput) fileInput.value = ''; }
</script>

<a href="#results" class="skip-link">Skip to results</a>
<main>
  <h1>${config.modelName} — ${taskLabel}</h1>
  <div class="container">
    <div>
      {#if !imageUrl}
        <div class="drop-zone" class:drag-over={dragOver} role="button" tabindex="0"
          aria-label="Drop an image here or click to browse for ${taskLabel.toLowerCase()}"
          onclick={() => fileInput?.click()}
          onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput?.click(); } }}
          ondragover={(e) => { e.preventDefault(); dragOver = true; }}
          ondragleave={() => { dragOver = false; }}
          ondrop={(e) => { e.preventDefault(); dragOver = false; const f = e.dataTransfer?.files[0]; if (f) processImage(f); }}>
          <p>Drop an image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input bind:this={fileInput} type="file" accept="image/*" style="position:absolute;left:-9999px;width:1px;height:1px;opacity:0;" aria-hidden="true" tabindex="-1" onchange={(e) => { const f = e.currentTarget.files?.[0]; if (f) processImage(f); }} />
        </div>
      {:else}
        <div class="preview">
          <img src={imageUrl} alt="Selected image for captioning" />
          <button class="change-btn" onclick={reset}>Choose another image</button>
        </div>
      {/if}
    </div>
    <div id="results" class="generation-output" role="status" aria-live="polite" aria-atomic="true">{caption || '(caption will appear here)'}</div>
  </div>
</main>
<aside class="status-bar"><span>{status}</span></aside>
<div class="footer">${buildFooterText(config)}</div>
`;
}

// ---- Visual Question Answering Page ----

function emitVQASvelte(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script${tsLang}>
  import { onMount } from 'svelte';
  import { createSession, runInference, getBackendLabel } from '$lib/inference.${le}';
  import { preprocessImage } from '$lib/preprocess.${le}';
  import { postprocessVQA } from '$lib/postprocess.${le}';
  import { loadTokenizer } from '$lib/preprocess.${le}';

  const MODEL_PATH = '${getModelPath(config, '')}';
  const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

  let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = $state(null);
  let tokenizer${t ? ': Awaited<ReturnType<typeof loadTokenizer>> | null' : ''} = $state(null);
  let answer = $state('');
  let imageUrl${t ? ': string | null' : ''} = $state(null);
  let question = $state('What is in this image?');
  let status${t ? ': string' : ''} = $state('Loading model...');
  let dragOver = $state(false);
  let fileInput${t ? ': HTMLInputElement' : ''};
  let loadedImg${t ? ': HTMLImageElement | null' : ''} = $state(null);

  onMount(() => {
    Promise.all([
      createSession(MODEL_PATH),
      loadTokenizer(TOKENIZER_PATH),
    ]).then(([s, tok]) => {
      session = s;
      tokenizer = tok;
      status = '${config.modelName} \\u00b7 Ready';
    }).catch((e) => { status = 'Failed to load model'; console.error('Model load error:', e); });
  });

  function handleFile(file${t ? ': File' : ''}) {
    if (!file.type.startsWith('image/')) return;
    const url = URL.createObjectURL(file);
    imageUrl = url;
    const img = new Image();
    img.src = url;
    img.onload = () => { loadedImg = img; };
  }

  async function handleAsk() {
    if (!session || !tokenizer || !loadedImg || !question.trim()) return;
    status = '${config.modelName} \\u00b7 Processing...';
    const start = performance.now();
    const canvas = document.createElement('canvas');
    canvas.width = loadedImg.naturalWidth; canvas.height = loadedImg.naturalHeight;
    const ctx = canvas.getContext('2d')${t ? '!' : ''}; ctx.drawImage(loadedImg, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
    const output = await runInference(session, inputTensor);
    const result = postprocessVQA(output, tokenizer);
    const elapsed = (performance.now() - start).toFixed(1);
    status = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`;
    answer = result;
  }

  function reset() { imageUrl = null; answer = ''; loadedImg = null; if (fileInput) fileInput.value = ''; }
</script>

<a href="#results" class="skip-link">Skip to results</a>
<main>
  <h1>${config.modelName} — ${taskLabel}</h1>
  <div class="container">
    <div class="multimodal-input">
      {#if !imageUrl}
        <div class="drop-zone" class:drag-over={dragOver} role="button" tabindex="0"
          aria-label="Drop an image here or click to browse"
          onclick={() => fileInput?.click()}
          onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput?.click(); } }}
          ondragover={(e) => { e.preventDefault(); dragOver = true; }}
          ondragleave={() => { dragOver = false; }}
          ondrop={(e) => { e.preventDefault(); dragOver = false; const f = e.dataTransfer?.files[0]; if (f) handleFile(f); }}>
          <p>Drop an image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input bind:this={fileInput} type="file" accept="image/*" style="position:absolute;left:-9999px;width:1px;height:1px;opacity:0;" aria-hidden="true" tabindex="-1" onchange={(e) => { const f = e.currentTarget.files?.[0]; if (f) handleFile(f); }} />
        </div>
      {:else}
        <div class="preview">
          <img src={imageUrl} alt="Selected image" />
          <button class="change-btn" onclick={reset}>Choose another image</button>
        </div>
      {/if}
      <input type="text" bind:value={question} placeholder="Ask a question..." aria-label="Question about the image" />
      <button class="primary-btn" onclick={handleAsk} aria-label="Ask question">Ask</button>
    </div>
    <div id="results" class="qa-answer" role="status" aria-live="polite" aria-atomic="true">{answer}</div>
  </div>
</main>
<aside class="status-bar"><span>{status}</span></aside>
<div class="footer">${buildFooterText(config)}</div>
`;
}

// ---- Document Question Answering Page ----

function emitDocQASvelte(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script${tsLang}>
  import { onMount } from 'svelte';
  import { createSession, runInference, getBackendLabel } from '$lib/inference.${le}';
  import { preprocessImage } from '$lib/preprocess.${le}';
  import { postprocessDocQA } from '$lib/postprocess.${le}';
  import { loadTokenizer } from '$lib/preprocess.${le}';

  const MODEL_PATH = '${getModelPath(config, '')}';
  const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

  let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = $state(null);
  let tokenizer${t ? ': Awaited<ReturnType<typeof loadTokenizer>> | null' : ''} = $state(null);
  let answer = $state('');
  let imageUrl${t ? ': string | null' : ''} = $state(null);
  let question = $state('What is the total amount?');
  let status${t ? ': string' : ''} = $state('Loading model...');
  let dragOver = $state(false);
  let fileInput${t ? ': HTMLInputElement' : ''};
  let loadedImg${t ? ': HTMLImageElement | null' : ''} = $state(null);

  onMount(() => {
    Promise.all([
      createSession(MODEL_PATH),
      loadTokenizer(TOKENIZER_PATH),
    ]).then(([s, tok]) => {
      session = s;
      tokenizer = tok;
      status = '${config.modelName} \\u00b7 Ready';
    }).catch((e) => { status = 'Failed to load model'; console.error('Model load error:', e); });
  });

  function handleFile(file${t ? ': File' : ''}) {
    if (!file.type.startsWith('image/')) return;
    const url = URL.createObjectURL(file);
    imageUrl = url;
    const img = new Image();
    img.src = url;
    img.onload = () => { loadedImg = img; };
  }

  async function handleAsk() {
    if (!session || !tokenizer || !loadedImg || !question.trim()) return;
    status = '${config.modelName} \\u00b7 Processing...';
    const start = performance.now();
    const canvas = document.createElement('canvas');
    canvas.width = loadedImg.naturalWidth; canvas.height = loadedImg.naturalHeight;
    const ctx = canvas.getContext('2d')${t ? '!' : ''}; ctx.drawImage(loadedImg, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
    const output = await runInference(session, inputTensor);
    const result = postprocessDocQA(output, tokenizer);
    const elapsed = (performance.now() - start).toFixed(1);
    status = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`;
    answer = result;
  }

  function reset() { imageUrl = null; answer = ''; loadedImg = null; if (fileInput) fileInput.value = ''; }
</script>

<a href="#results" class="skip-link">Skip to results</a>
<main>
  <h1>${config.modelName} — ${taskLabel}</h1>
  <div class="container">
    <div class="multimodal-input">
      {#if !imageUrl}
        <div class="drop-zone" class:drag-over={dragOver} role="button" tabindex="0"
          aria-label="Drop a document image here or click to browse"
          onclick={() => fileInput?.click()}
          onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput?.click(); } }}
          ondragover={(e) => { e.preventDefault(); dragOver = true; }}
          ondragleave={() => { dragOver = false; }}
          ondrop={(e) => { e.preventDefault(); dragOver = false; const f = e.dataTransfer?.files[0]; if (f) handleFile(f); }}>
          <p>Drop a document image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input bind:this={fileInput} type="file" accept="image/*" style="position:absolute;left:-9999px;width:1px;height:1px;opacity:0;" aria-hidden="true" tabindex="-1" onchange={(e) => { const f = e.currentTarget.files?.[0]; if (f) handleFile(f); }} />
        </div>
      {:else}
        <div class="preview">
          <img src={imageUrl} alt="Document image" />
          <button class="change-btn" onclick={reset}>Choose another image</button>
        </div>
      {/if}
      <input type="text" bind:value={question} placeholder="Ask about the document..." aria-label="Question about the document" />
      <button class="primary-btn" onclick={handleAsk} aria-label="Ask question">Ask</button>
    </div>
    <div id="results" class="qa-answer" role="status" aria-live="polite" aria-atomic="true">{answer}</div>
  </div>
</main>
<aside class="status-bar"><span>{status}</span></aside>
<div class="footer">${buildFooterText(config)}</div>
`;
}

// ---- Image-Text-to-Text Page ----

function emitImageTextToTextSvelte(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script${tsLang}>
  import { onMount } from 'svelte';
  import { createSession, runInference, getBackendLabel } from '$lib/inference.${le}';
  import { preprocessImage } from '$lib/preprocess.${le}';
  import { postprocessImageTextToText } from '$lib/postprocess.${le}';
  import { loadTokenizer } from '$lib/preprocess.${le}';

  const MODEL_PATH = '${getModelPath(config, '')}';
  const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

  let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = $state(null);
  let tokenizer${t ? ': Awaited<ReturnType<typeof loadTokenizer>> | null' : ''} = $state(null);
  let output = $state('');
  let imageUrl${t ? ': string | null' : ''} = $state(null);
  let prompt = $state('Describe this image in detail.');
  let status${t ? ': string' : ''} = $state('Loading model...');
  let dragOver = $state(false);
  let processing = $state(false);
  let fileInput${t ? ': HTMLInputElement' : ''};
  let loadedImg${t ? ': HTMLImageElement | null' : ''} = $state(null);

  onMount(() => {
    Promise.all([
      createSession(MODEL_PATH),
      loadTokenizer(TOKENIZER_PATH),
    ]).then(([s, tok]) => {
      session = s;
      tokenizer = tok;
      status = '${config.modelName} \\u00b7 Ready';
    }).catch((e) => { status = 'Failed to load model'; console.error('Model load error:', e); });
  });

  function handleFile(file${t ? ': File' : ''}) {
    if (!file.type.startsWith('image/')) return;
    const url = URL.createObjectURL(file);
    imageUrl = url;
    const img = new Image();
    img.src = url;
    img.onload = () => { loadedImg = img; };
  }

  async function handleGenerate() {
    if (!session || !tokenizer || !loadedImg || processing) return;
    processing = true;
    status = '${config.modelName} \\u00b7 Generating...';
    const start = performance.now();
    const canvas = document.createElement('canvas');
    canvas.width = loadedImg.naturalWidth; canvas.height = loadedImg.naturalHeight;
    const ctx = canvas.getContext('2d')${t ? '!' : ''}; ctx.drawImage(loadedImg, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
    const raw = await runInference(session, inputTensor);
    const result = postprocessImageTextToText(raw, tokenizer);
    const elapsed = (performance.now() - start).toFixed(1);
    status = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`;
    output = result;
    processing = false;
  }

  function reset() { imageUrl = null; output = ''; loadedImg = null; if (fileInput) fileInput.value = ''; }
</script>

<a href="#results" class="skip-link">Skip to results</a>
<main>
  <h1>${config.modelName} — ${taskLabel}</h1>
  <div class="container">
    <div class="multimodal-input">
      {#if !imageUrl}
        <div class="drop-zone" class:drag-over={dragOver} role="button" tabindex="0"
          aria-label="Drop an image here or click to browse"
          onclick={() => fileInput?.click()}
          onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput?.click(); } }}
          ondragover={(e) => { e.preventDefault(); dragOver = true; }}
          ondragleave={() => { dragOver = false; }}
          ondrop={(e) => { e.preventDefault(); dragOver = false; const f = e.dataTransfer?.files[0]; if (f) handleFile(f); }}>
          <p>Drop an image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input bind:this={fileInput} type="file" accept="image/*" style="position:absolute;left:-9999px;width:1px;height:1px;opacity:0;" aria-hidden="true" tabindex="-1" onchange={(e) => { const f = e.currentTarget.files?.[0]; if (f) handleFile(f); }} />
        </div>
      {:else}
        <div class="preview">
          <img src={imageUrl} alt="Selected image" />
          <button class="change-btn" onclick={reset}>Choose another image</button>
        </div>
      {/if}
      <input type="text" bind:value={prompt} placeholder="Enter a prompt..." aria-label="Prompt for image description" />
      <button class="primary-btn" onclick={handleGenerate} disabled={processing} aria-label="Generate text">{processing ? 'Generating...' : 'Generate'}</button>
    </div>
    <div id="results" class="generation-output" role="status" aria-live="polite" aria-atomic="true">{output || '(output will appear here)'}</div>
  </div>
</main>
<aside class="status-bar"><span>{status}</span></aside>
<div class="footer">${buildFooterText(config)}</div>
`;
}

// ---- Audio-to-Audio Page ----

function emitAudioToAudioSvelte(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script${tsLang}>
  import { onMount } from 'svelte';
  import { createSession, runInference, getBackendLabel } from '$lib/inference.${le}';
  import { postprocessAudioToAudio, playAudio } from '$lib/postprocess.${le}';

  const MODEL_PATH = '${getModelPath(config, '')}';

  let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = $state(null);
  let status${t ? ': string' : ''} = $state('Loading model...');
  let output = $state('');

  onMount(() => {
    createSession(MODEL_PATH).then((s) => {
      session = s;
      status = '${config.modelName} \\u00b7 Ready';
    }).catch((e) => { status = 'Failed to load model'; console.error('Model load error:', e); });
  });

  async function handleFileChange(e${t ? ': Event' : ''}) {
    const file = (e.currentTarget as HTMLInputElement).files?.[0];
    if (!file || !session) return;

    status = '${config.modelName} \\u00b7 Processing...';
    const start = performance.now();

    const audioCtx = new AudioContext({ sampleRate: 16000 });
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    const samples = audioBuffer.getChannelData(0);
    const input = new Float32Array(samples);
    const raw = await runInference(session, input);
    const processedSamples = postprocessAudioToAudio(raw);
    await playAudio(processedSamples);

    const elapsed = (performance.now() - start).toFixed(1);
    status = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`;
    output = 'Processed ' + (samples.length / 16000).toFixed(1) + 's of audio. Playing output...';
  }
</script>

<a href="#results" class="skip-link">Skip to results</a>
<main>
  <h1>${config.modelName} — ${taskLabel}</h1>
  <div>
    <label for="fileInput">Choose an audio file</label>
    <input id="fileInput" type="file" accept="audio/*" onchange={handleFileChange} aria-label="Select audio file for ${taskLabel.toLowerCase()}" />
  </div>
  <div id="results" class="generation-output" role="status" aria-live="polite" aria-atomic="true">{output}</div>
</main>
<aside class="status-bar"><span>{status}</span></aside>
<div class="footer">${buildFooterText(config)}</div>
`;
}

// ---- Speaker Diarization Page ----

function emitSpeakerDiarizationSvelte(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script${tsLang}>
  import { onMount } from 'svelte';
  import { createSession, runInference, getBackendLabel } from '$lib/inference.${le}';
  import { postprocessSpeakerDiarization } from '$lib/postprocess.${le}';

  const MODEL_PATH = '${getModelPath(config, '')}';

  let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = $state(null);
  let segments${t ? ': Array<{ speaker: number; start: number; end: number; text: string }> | null' : ''} = $state(null);
  let status${t ? ': string' : ''} = $state('Loading model...');

  onMount(() => {
    createSession(MODEL_PATH).then((s) => {
      session = s;
      status = '${config.modelName} \\u00b7 Ready';
    }).catch((e) => { status = 'Failed to load model'; console.error('Model load error:', e); });
  });

  async function handleFileChange(e${t ? ': Event' : ''}) {
    const file = (e.currentTarget as HTMLInputElement).files?.[0];
    if (!file || !session) return;

    status = '${config.modelName} \\u00b7 Processing...';
    const start = performance.now();

    const audioCtx = new AudioContext({ sampleRate: 16000 });
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    const samples = audioBuffer.getChannelData(0);
    const input = new Float32Array(samples);
    const raw = await runInference(session, input);
    const results = postprocessSpeakerDiarization(raw);

    const elapsed = (performance.now() - start).toFixed(1);
    status = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`;
    segments = results;
  }
</script>

<a href="#results" class="skip-link">Skip to results</a>
<main>
  <h1>${config.modelName} — ${taskLabel}</h1>
  <div>
    <label for="fileInput">Choose an audio file</label>
    <input id="fileInput" type="file" accept="audio/*" onchange={handleFileChange} aria-label="Select audio file for ${taskLabel.toLowerCase()}" />
  </div>
  <div id="results" class="diarization-timeline" role="status" aria-live="polite" aria-atomic="true">
    {#if segments}
      {#each segments as seg, i}
        <div class="diarization-segment" tabindex="0" aria-label="Speaker {seg.speaker}: {seg.start.toFixed(1)}s to {seg.end.toFixed(1)}s">
          <span class="speaker">Speaker {seg.speaker}</span>
          <span>{seg.text}</span>
          <span class="time">{seg.start.toFixed(1)}s - {seg.end.toFixed(1)}s</span>
        </div>
      {/each}
    {/if}
  </div>
</main>
<aside class="status-bar"><span>{status}</span></aside>
<div class="footer">${buildFooterText(config)}</div>
`;
}

// ---- Voice Activity Detection Page ----

function emitVADSvelte(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script${tsLang}>
  import { onMount } from 'svelte';
  import { createSession, runInference, getBackendLabel } from '$lib/inference.${le}';
  import { postprocessVAD } from '$lib/postprocess.${le}';

  const MODEL_PATH = '${getModelPath(config, '')}';

  let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = $state(null);
  let segments${t ? ': Array<{ label: string; start: number; end: number }> | null' : ''} = $state(null);
  let status${t ? ': string' : ''} = $state('Loading model...');

  onMount(() => {
    createSession(MODEL_PATH).then((s) => {
      session = s;
      status = '${config.modelName} \\u00b7 Ready';
    }).catch((e) => { status = 'Failed to load model'; console.error('Model load error:', e); });
  });

  async function handleFileChange(e${t ? ': Event' : ''}) {
    const file = (e.currentTarget as HTMLInputElement).files?.[0];
    if (!file || !session) return;

    status = '${config.modelName} \\u00b7 Processing...';
    const start = performance.now();

    const audioCtx = new AudioContext({ sampleRate: 16000 });
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    const samples = audioBuffer.getChannelData(0);
    const input = new Float32Array(samples);
    const raw = await runInference(session, input);
    const results = postprocessVAD(raw);

    const elapsed = (performance.now() - start).toFixed(1);
    status = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`;
    segments = results;
  }
</script>

<a href="#results" class="skip-link">Skip to results</a>
<main>
  <h1>${config.modelName} — ${taskLabel}</h1>
  <div>
    <label for="fileInput">Choose an audio file</label>
    <input id="fileInput" type="file" accept="audio/*" onchange={handleFileChange} aria-label="Select audio file for ${taskLabel.toLowerCase()}" />
  </div>
  <div id="results" class="vad-segments" role="status" aria-live="polite" aria-atomic="true">
    {#if segments}
      {#each segments as seg, i}
        <div class="vad-segment" tabindex="0" aria-label="{seg.label}: {seg.start.toFixed(1)}s to {seg.end.toFixed(1)}s">
          <span class="label">{seg.label}</span>
          <span class="time">{seg.start.toFixed(1)}s - {seg.end.toFixed(1)}s</span>
        </div>
      {/each}
    {/if}
  </div>
</main>
<aside class="status-bar"><span>{status}</span></aside>
<div class="footer">${buildFooterText(config)}</div>
`;
}

function emitRealtimePageSvelte(config: ResolvedConfig): string {
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
      postImport = `import { postprocessDetections } from '$lib/postprocess.${le}';`;
      extraConst = `\n  const NUM_ATTRIBUTES = ${numAttributes};\n  const NUM_ANCHORS = ${numAnchors};\n  const COLORS = [[56,189,248],[249,115,22],[34,197,94],[168,85,247],[251,191,36],[239,68,68],[20,184,166],[236,72,153],[99,102,241],[163,230,53]];`;
      processAndRender = `const boxes = postprocessDetections(output, NUM_ANCHORS, NUM_ATTRIBUTES);
        const elapsed = performance.now() - start;
        const modelSize = ${config.preprocess.imageSize};
        const scaleX = imageData.width / modelSize;
        const scaleY = imageData.height / modelSize;
        overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
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
      postImport = `import { postprocessSegmentation } from '$lib/postprocess.${le}';`;
      extraConst = `\n  const NUM_CLASSES = ${numClasses};\n  const MASK_H = ${maskH};\n  const MASK_W = ${maskW};\n  const COLORS = [[56,189,248],[249,115,22],[34,197,94],[168,85,247],[251,191,36],[239,68,68],[20,184,166],[236,72,153],[99,102,241],[163,230,53]];`;
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
        overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
        overlayCtx.drawImage(maskCanvas, 0, 0, video.videoWidth, video.videoHeight);`;
      break;
    }
    default: {
      postImport = `import { postprocessResults } from '$lib/postprocess.${le}';`;
      processAndRender = `const results = postprocessResults(output);
        const elapsed = performance.now() - start;
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

  return `<script${tsLang}>
  import { onMount } from 'svelte';
  import { createSession, runInference, getBackendLabel } from '$lib/inference.${le}';
  import { preprocessImage } from '$lib/preprocess.${le}';
  ${postImport}
  import { ${startFn}, captureFrame, createInferenceLoop } from '$lib/input.${le}';

  const MODEL_PATH = '${getModelPath(config, '')}';
${extraConst}

  let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = $state(null);
  let status${t ? ': string' : ''} = $state('Loading model...');
  let started = $state(false);
  let video${t ? ': HTMLVideoElement' : ''};
  let overlay${t ? ': HTMLCanvasElement' : ''};
  let loop${t ? ': ReturnType<typeof createInferenceLoop> | null' : ''} = $state(null);

  onMount(() => {
    createSession(MODEL_PATH).then((s) => {
      session = s;
      status = '${config.modelName} \\u00b7 Ready \\u00b7 Tap Start';
    }).catch((e) => { status = 'Failed to load model'; console.error('Model load error:', e); });
  });

  async function handleStart() {
    try {
      await ${startFn}(video);
      overlay.width = video.videoWidth;
      overlay.height = video.videoHeight;
      started = true;

      const overlayCtx = overlay.getContext('2d')${t ? '!' : ''};
      loop = createInferenceLoop({
        video,
        canvas: overlay,
        async onFrame(imageData${t ? ': ImageData' : ''}) {
          const start = performance.now();
          const inputTensor = preprocessImage(imageData.data, imageData.width, imageData.height);
          const output = await runInference(session${t ? '!' : ''}, inputTensor);
          ${processAndRender}
          return { result: null, elapsed };
        },
        onStatus(elapsed${t ? ': number' : ''}) {
          status = '${config.modelName} \\u00b7 ' + elapsed.toFixed(1) + 'ms \\u00b7 ' + getBackendLabel(session${t ? '!' : ''});
        },
      });
      loop.start();
    } catch (e) {
      status = '${label} access denied';
      console.error('${label} error:', e);
    }
  }

  function handlePause() {
    if (loop) { loop.stop(); loop = null; }
  }
</script>

<a href="#results" class="skip-link">Skip to results</a>
<main>
  <h1>${config.modelName} — ${taskLabel}</h1>
  {#if !started}
    <div class="permission-prompt">
      <p>This app ${actionLabel} to run ${taskLabel.toLowerCase()} in real time.</p>
      <p class="hint">No video is recorded or sent anywhere.</p>
      <button class="primary-btn" onclick={handleStart}>${btnLabel}</button>
    </div>
  {:else}
    <div>
      <div class="video-wrapper">
        <video bind:this={video} autoplay playsinline muted></video>
        <canvas bind:this={overlay}></canvas>
      </div>
      <div class="controls">
        <button class="control-btn" onclick={handlePause}>\\u23f8 Pause</button>
      </div>
    </div>
  {/if}
</main>
<aside class="status-bar"><span>{status}</span></aside>
<div class="footer">${buildFooterText(config)}</div>
`;
}

// ---- Audio: File + Classification Page ----

function emitFileAudioClassificationPageSvelte(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script${tsLang}>
  import { onMount } from 'svelte';
  import { createSession, runInference, getBackendLabel } from '$lib/inference.${le}';
  import { melSpectrogram, mfcc } from '$lib/preprocess.${le}';
  import { postprocessResults } from '$lib/postprocess.${le}';

  const MODEL_PATH = '${getModelPath(config, '')}';

  let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = $state(null);
  let results${t ? ': { indices: number[]; values: number[] } | null' : ''} = $state(null);
  let status${t ? ': string' : ''} = $state('Loading model...');

  onMount(() => {
    createSession(MODEL_PATH).then((s) => {
      session = s;
      status = '${config.modelName} \\u00b7 Ready';
    }).catch((e) => { status = 'Failed to load model'; console.error('Model load error:', e); });
  });

  async function handleFileChange(e${t ? ': Event' : ''}) {
    const file = (e.currentTarget as HTMLInputElement).files?.[0];
    if (!file || !session) return;

    status = '${config.modelName} \\u00b7 Decoding audio...';

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

    status = '${config.modelName} \\u00b7 Processing...';
    const start = performance.now();

    const mel = melSpectrogram(samples, 16000, 512, 160, 40);
    const features = mfcc(mel.data, mel.numFrames, mel.numMelBands, 13);
    const output = await runInference(session, features);
    const r = postprocessResults(output);

    const elapsed = (performance.now() - start).toFixed(1);
    status = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`;
    results = r;
  }
</script>

<a href="#results" class="skip-link">Skip to results</a>
<main>
  <h1>${config.modelName} — ${taskLabel}</h1>
  <div>
    <label for="fileInput">Choose an audio file</label>
    <input id="fileInput" type="file" accept="audio/*" onchange={handleFileChange} aria-label="Select audio file for ${taskLabel.toLowerCase()}" />
  </div>
  <div id="results" class="results" role="status" aria-live="polite" aria-atomic="true">
    {#if results}
      {#each results.indices as idx, i}
        {#if results.values[i] >= 0.01}
          {@const pct = (results.values[i] * 100).toFixed(1)}
          {@const maxVal = results.values[0] || 1}
          <div class="result-row" class:top-result={i === 0} tabindex="0" aria-label="Class {idx}: {pct} percent">
            <span class="result-label">Class {idx}</span>
            <div class="result-bar-container">
              <div class="result-bar" style="width: {(results.values[i] / maxVal) * 100}%"></div>
            </div>
            <span class="result-pct">{pct}%</span>
          </div>
        {/if}
      {/each}
    {/if}
  </div>
</main>
<aside class="status-bar"><span>{status}</span></aside>
<div class="footer">${buildFooterText(config)}</div>
`;
}

// ---- Audio: File + STT Page ----

function emitFileSpeechToTextPageSvelte(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script${tsLang}>
  import { onMount } from 'svelte';
  import { createSession, runInference, getBackendLabel } from '$lib/inference.${le}';
  import { melSpectrogram } from '$lib/preprocess.${le}';
  import { postprocessTranscript } from '$lib/postprocess.${le}';

  const MODEL_PATH = '${getModelPath(config, '')}';
  const VOCAB = [' ', ...'abcdefghijklmnopqrstuvwxyz'.split(''), "'"];

  let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = $state(null);
  let transcript${t ? ': string' : ''} = $state('');
  let status${t ? ': string' : ''} = $state('Loading model...');

  onMount(() => {
    createSession(MODEL_PATH).then((s) => {
      session = s;
      status = '${config.modelName} \\u00b7 Ready';
    }).catch((e) => { status = 'Failed to load model'; console.error('Model load error:', e); });
  });

  async function handleFileChange(e${t ? ': Event' : ''}) {
    const file = (e.currentTarget as HTMLInputElement).files?.[0];
    if (!file || !session) return;

    status = '${config.modelName} \\u00b7 Decoding audio...';

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

    status = '${config.modelName} \\u00b7 Processing...';
    const start = performance.now();

    const mel = melSpectrogram(samples, 16000, 512, 160, 80);
    const output = await runInference(session, mel.data);
    const vocabSize = VOCAB.length + 1;
    const numTimesteps = Math.floor(output.length / vocabSize);
    const text = postprocessTranscript(output, numTimesteps, vocabSize, VOCAB);

    const elapsed = (performance.now() - start).toFixed(1);
    status = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`;
    transcript = text || '(no speech detected)';
  }
</script>

<a href="#transcript" class="skip-link">Skip to results</a>
<main>
  <h1>${config.modelName} — ${taskLabel}</h1>
  <div>
    <label for="fileInput">Choose an audio file</label>
    <input id="fileInput" type="file" accept="audio/*" onchange={handleFileChange} aria-label="Select audio file for ${taskLabel.toLowerCase()}" />
  </div>
  <pre id="transcript" class="transcript" role="status" aria-live="polite" aria-atomic="true">{transcript}</pre>
</main>
<aside class="status-bar"><span>{status}</span></aside>
<div class="footer">${buildFooterText(config)}</div>
`;
}

// ---- Audio: Mic + STT Page ----

function emitMicSpeechToTextPageSvelte(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script${tsLang}>
  import { onMount, onDestroy } from 'svelte';
  import { createSession, runInference, getBackendLabel } from '$lib/inference.${le}';
  import { melSpectrogram } from '$lib/preprocess.${le}';
  import { postprocessTranscript } from '$lib/postprocess.${le}';
  import { startAudioCapture, stopStream, createAudioInferenceLoop } from '$lib/input.${le}';

  const MODEL_PATH = '${getModelPath(config, '')}';
  const VOCAB = [' ', ...'abcdefghijklmnopqrstuvwxyz'.split(''), "'"];

  let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = $state(null);
  let transcript${t ? ': string' : ''} = $state('(listening...)');
  let status${t ? ': string' : ''} = $state('Loading model...');
  let recording = $state(false);
  let capture${t ? ': any' : ''} = $state(null);
  let loop${t ? ': any' : ''} = $state(null);

  onMount(() => {
    createSession(MODEL_PATH).then((s) => {
      session = s;
      status = '${config.modelName} \\u00b7 Ready';
    }).catch((e) => { status = 'Failed to load model'; console.error('Model load error:', e); });
  });

  onDestroy(() => {
    if (loop) loop.stop();
    if (capture) { stopStream(capture.stream); capture.audioContext.close(); }
  });

  async function processAudio(samples${t ? ': Float32Array' : ''}) {
    const mel = melSpectrogram(samples, 16000, 512, 160, 80);
    const output = await runInference(session${t ? '!' : ''}, mel.data);
    const vocabSize = VOCAB.length + 1;
    const numTimesteps = Math.floor(output.length / vocabSize);
    return postprocessTranscript(output, numTimesteps, vocabSize, VOCAB);
  }

  async function handleStart() {
    if (!session) { status = 'Model not loaded yet.'; return; }
    try {
      capture = await startAudioCapture(16000);
      recording = true;
      status = '${config.modelName} \\u00b7 Listening...';
      loop = createAudioInferenceLoop({
        getSamples: capture.getSamples,
        onResult(text${t ? ': string' : ''}) { transcript = text || '(listening...)'; },
        intervalMs: 2000,
      });
      loop.start();
    } catch (e) { status = 'Microphone access denied'; console.error('Mic error:', e); }
  }

  function handleStop() {
    if (loop) { loop.stop(); loop = null; }
    if (capture) { stopStream(capture.stream); capture.audioContext.close(); capture = null; }
    recording = false;
    status = '${config.modelName} \\u00b7 Stopped';
  }
</script>

<a href="#transcript" class="skip-link">Skip to results</a>
<main>
  <h1>${config.modelName} — ${taskLabel}</h1>
  <div class="controls" role="group" aria-label="Recording controls">
    <button onclick={handleStart} disabled={recording} aria-label="Start recording">Start Recording</button>
    <button onclick={handleStop} disabled={!recording} aria-label="Stop recording">Stop Recording</button>
  </div>
  <pre id="transcript" class="transcript" role="status" aria-live="polite" aria-atomic="true">{transcript}</pre>
</main>
<aside class="status-bar"><span>{status}</span></aside>
<div class="footer">${buildFooterText(config)}</div>
`;
}

// ---- Audio: Mic + Classification Page ----

function emitMicAudioClassificationPageSvelte(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script${tsLang}>
  import { onMount, onDestroy } from 'svelte';
  import { createSession, runInference, getBackendLabel } from '$lib/inference.${le}';
  import { melSpectrogram, mfcc } from '$lib/preprocess.${le}';
  import { postprocessResults } from '$lib/postprocess.${le}';
  import { startAudioCapture, stopStream, createAudioInferenceLoop } from '$lib/input.${le}';

  const MODEL_PATH = '${getModelPath(config, '')}';

  let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = $state(null);
  let results${t ? ': { indices: number[]; values: number[] } | null' : ''} = $state(null);
  let status${t ? ': string' : ''} = $state('Loading model...');
  let recording = $state(false);
  let capture${t ? ': any' : ''} = $state(null);
  let loop${t ? ': any' : ''} = $state(null);

  onMount(() => {
    createSession(MODEL_PATH).then((s) => {
      session = s;
      status = '${config.modelName} \\u00b7 Ready';
    }).catch((e) => { status = 'Failed to load model'; console.error('Model load error:', e); });
  });

  onDestroy(() => {
    if (loop) loop.stop();
    if (capture) { stopStream(capture.stream); capture.audioContext.close(); }
  });

  async function processAudio(samples${t ? ': Float32Array' : ''}) {
    const mel = melSpectrogram(samples, 16000, 512, 160, 40);
    const features = mfcc(mel.data, mel.numFrames, mel.numMelBands, 13);
    const output = await runInference(session${t ? '!' : ''}, features);
    return postprocessResults(output);
  }

  async function handleStart() {
    if (!session) { status = 'Model not loaded yet.'; return; }
    try {
      capture = await startAudioCapture(16000);
      recording = true;
      status = '${config.modelName} \\u00b7 Listening...';
      loop = createAudioInferenceLoop({
        getSamples: capture.getSamples,
        onResult(r${t ? ': { indices: number[]; values: number[] }' : ''}) { results = r; },
        intervalMs: 2000,
      });
      loop.start();
    } catch (e) { status = 'Microphone access denied'; console.error('Mic error:', e); }
  }

  function handleStop() {
    if (loop) { loop.stop(); loop = null; }
    if (capture) { stopStream(capture.stream); capture.audioContext.close(); capture = null; }
    recording = false;
    status = '${config.modelName} \\u00b7 Stopped';
  }
</script>

<a href="#results" class="skip-link">Skip to results</a>
<main>
  <h1>${config.modelName} — ${taskLabel}</h1>
  <div class="controls" role="group" aria-label="Recording controls">
    <button onclick={handleStart} disabled={recording} aria-label="Start recording">Start Recording</button>
    <button onclick={handleStop} disabled={!recording} aria-label="Stop recording">Stop Recording</button>
  </div>
  <div id="results" class="results" role="status" aria-live="polite" aria-atomic="true">
    {#if results}
      {#each results.indices as idx, i}
        {#if results.values[i] >= 0.01}
          {@const pct = (results.values[i] * 100).toFixed(1)}
          {@const maxVal = results.values[0] || 1}
          <div class="result-row" class:top-result={i === 0} tabindex="0" aria-label="Class {idx}: {pct} percent">
            <span class="result-label">Class {idx}</span>
            <div class="result-bar-container">
              <div class="result-bar" style="width: {(results.values[i] / maxVal) * 100}%"></div>
            </div>
            <span class="result-pct">{pct}%</span>
          </div>
        {/if}
      {/each}
    {/if}
  </div>
</main>
<aside class="status-bar"><span>{status}</span></aside>
<div class="footer">${buildFooterText(config)}</div>
`;
}

// ---- Audio: TTS Page ----

function emitTextToSpeechPageSvelte(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script${tsLang}>
  import { onMount } from 'svelte';
  import { createSession, runInference, getBackendLabel } from '$lib/inference.${le}';
  import { postprocessAudio, playAudio } from '$lib/postprocess.${le}';

  const MODEL_PATH = '${getModelPath(config, '')}';

  let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = $state(null);
  let status${t ? ': string' : ''} = $state('Loading model...');
  let text = $state('Hello, this is a test of text to speech.');

  onMount(() => {
    createSession(MODEL_PATH).then((s) => {
      session = s;
      status = '${config.modelName} \\u00b7 Ready';
    }).catch((e) => { status = 'Failed to load model'; console.error('Model load error:', e); });
  });

  async function handleSynthesize() {
    const trimmed = text.trim();
    if (!trimmed || !session) return;

    status = '${config.modelName} \\u00b7 Synthesizing...';
    const start = performance.now();

    const tokens = new Float32Array(trimmed.length);
    for (let i = 0; i < trimmed.length; i++) { tokens[i] = trimmed.charCodeAt(i); }

    const output = await runInference(session, tokens);
    const samples = postprocessAudio(output);
    await playAudio(samples);

    const elapsed = (performance.now() - start).toFixed(1);
    status = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`;
  }
</script>

<main>
  <h1>${config.modelName} — ${taskLabel}</h1>
  <div class="tts-input">
    <label for="textInput">Enter text to synthesize</label>
    <textarea id="textInput" rows="4" bind:value={text} aria-label="Text to synthesize"></textarea>
    <button class="primary-btn" onclick={handleSynthesize} aria-label="Synthesize speech">Synthesize</button>
  </div>
</main>
<aside class="status-bar"><span>{status}</span></aside>
<div class="footer">${buildFooterText(config)}</div>
`;
}

// ---- Text: Classification Page ----

function emitTextClassificationPageSvelte(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script${tsLang}>
  import { onMount } from 'svelte';
  import { createSession, runInference, getBackendLabel } from '$lib/inference.${le}';
  import { loadTokenizer, tokenizeText } from '$lib/preprocess.${le}';
  import { postprocessResults } from '$lib/postprocess.${le}';

  const MODEL_PATH = '${getModelPath(config, '')}';
  const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

  let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = $state(null);
  let tokenizer${t ? ': Awaited<ReturnType<typeof loadTokenizer>> | null' : ''} = $state(null);
  let results${t ? ': { indices: number[]; values: number[] } | null' : ''} = $state(null);
  let text = $state('');
  let status${t ? ': string' : ''} = $state('Loading model and tokenizer...');

  onMount(() => {
    Promise.all([
      createSession(MODEL_PATH),
      loadTokenizer(TOKENIZER_PATH),
    ]).then(([s, tok]) => {
      session = s;
      tokenizer = tok;
      status = '${config.modelName} \\u00b7 Ready';
    }).catch((e) => { status = 'Failed to load model or tokenizer'; console.error('Load error:', e); });
  });

  async function handleClassify() {
    const trimmed = text.trim();
    if (!trimmed || !session || !tokenizer) return;

    status = '${config.modelName} \\u00b7 Processing...';
    const start = performance.now();

    const { inputIds, attentionMask } = tokenizeText(tokenizer, trimmed);
    const output = await runInference(session, inputIds);
    const r = postprocessResults(output);

    const elapsed = (performance.now() - start).toFixed(1);
    status = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`;
    results = r;
  }
</script>

<a href="#results" class="skip-link">Skip to results</a>
<main>
  <h1>${config.modelName} — ${taskLabel}</h1>
  <div>
    <label for="textInput">Enter text to classify</label>
    <textarea id="textInput" rows="4" bind:value={text} aria-label="Text to classify"></textarea>
    <button class="primary-btn" onclick={handleClassify} disabled={!session || !tokenizer} aria-label="Classify text">Classify</button>
  </div>
  <div id="results" class="results" role="status" aria-live="polite" aria-atomic="true">
    {#if results}
      {#each results.indices as idx, i}
        {#if results.values[i] >= 0.01}
          {@const pct = (results.values[i] * 100).toFixed(1)}
          {@const maxVal = results.values[0] || 1}
          <div class="result-row" class:top-result={i === 0} tabindex="0" aria-label="Class {idx}: {pct} percent">
            <span class="result-label">Class {idx}</span>
            <div class="result-bar-container">
              <div class="result-bar" style="width: {(results.values[i] / maxVal) * 100}%"></div>
            </div>
            <span class="result-pct">{pct}%</span>
          </div>
        {/if}
      {/each}
    {/if}
  </div>
</main>
<aside class="status-bar"><span>{status}</span></aside>
<div class="footer">${buildFooterText(config)}</div>
`;
}

// ---- Text: Zero-Shot Classification Page ----

function emitZeroShotClassificationPageSvelte(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script${tsLang}>
  import { onMount } from 'svelte';
  import { createSession, runInference, getBackendLabel } from '$lib/inference.${le}';
  import { loadTokenizer, tokenizeText } from '$lib/preprocess.${le}';
  import { postprocessZeroShot } from '$lib/postprocess.${le}';

  const MODEL_PATH = '${getModelPath(config, '')}';
  const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

  let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = $state(null);
  let tokenizer${t ? ': Awaited<ReturnType<typeof loadTokenizer>> | null' : ''} = $state(null);
  let results${t ? ': { label: string; score: number }[] | null' : ''} = $state(null);
  let text = $state('');
  let labels = $state('technology, politics, sports');
  let status${t ? ': string' : ''} = $state('Loading model and tokenizer...');

  onMount(() => {
    Promise.all([
      createSession(MODEL_PATH),
      loadTokenizer(TOKENIZER_PATH),
    ]).then(([s, tok]) => {
      session = s;
      tokenizer = tok;
      status = '${config.modelName} \\u00b7 Ready';
    }).catch((e) => { status = 'Failed to load model or tokenizer'; console.error('Load error:', e); });
  });

  async function handleClassify() {
    const trimmedText = text.trim();
    const trimmedLabels = labels.trim();
    if (!trimmedText || !trimmedLabels || !session || !tokenizer) return;

    const labelArray = trimmedLabels.split(',').map((l) => l.trim()).filter(Boolean);
    if (labelArray.length === 0) return;

    status = '${config.modelName} \\u00b7 Processing...';
    const start = performance.now();

    const scores${t ? ': number[]' : ''} = [];
    for (const label of labelArray) {
      const hypothesis = trimmedText + ' </s></s> This is about ' + label + '.';
      const { inputIds } = tokenizeText(tokenizer, hypothesis);
      const output = await runInference(session, inputIds);
      const score = output[output.length > 2 ? 2 : output.length - 1];
      scores.push(score);
    }

    const r = postprocessZeroShot(scores, labelArray);

    const elapsed = (performance.now() - start).toFixed(1);
    status = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`;
    results = r;
  }
</script>

<a href="#results" class="skip-link">Skip to results</a>
<main>
  <h1>${config.modelName} — ${taskLabel}</h1>
  <div>
    <label for="textInput">Enter text to classify</label>
    <textarea id="textInput" rows="4" bind:value={text} aria-label="Text to classify"></textarea>
    <label for="labelsInput">Candidate labels (comma-separated)</label>
    <input id="labelsInput" type="text" bind:value={labels} aria-label="Candidate labels" />
    <button class="primary-btn" onclick={handleClassify} disabled={!session || !tokenizer} aria-label="Classify text">Classify</button>
  </div>
  <div id="results" class="results" role="status" aria-live="polite" aria-atomic="true">
    {#if results}
      {#each results as result, i}
        {@const pct = (result.score * 100).toFixed(1)}
        {@const maxScore = results[0]?.score || 1}
        <div class="result-row" class:top-result={i === 0} tabindex="0" aria-label="{result.label}: {pct} percent">
          <span class="result-label">{result.label}</span>
          <div class="result-bar-container">
            <div class="result-bar" style="width: {(result.score / maxScore) * 100}%"></div>
          </div>
          <span class="result-pct">{pct}%</span>
        </div>
      {/each}
    {/if}
  </div>
</main>
<aside class="status-bar"><span>{status}</span></aside>
<div class="footer">${buildFooterText(config)}</div>
`;
}

// ---- Text: Generation Page ----

function emitTextGenerationPageSvelte(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const tsLang = t ? ' lang="ts"' : '';

  return `<script${tsLang}>
  import { onMount } from 'svelte';
  import { createSession, runInference, getBackendLabel } from '$lib/inference.${le}';
  import { loadTokenizer } from '$lib/preprocess.${le}';
  import { postprocessGeneration, sampleNextToken } from '$lib/postprocess.${le}';

  const MODEL_PATH = '${getModelPath(config, '')}';
  const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';
  const MAX_NEW_TOKENS = 50;
  const EOS_TOKEN_ID = 2;

  let session${t ? ': Awaited<ReturnType<typeof createSession>> | null' : ''} = $state(null);
  let tokenizer${t ? ': Awaited<ReturnType<typeof loadTokenizer>> | null' : ''} = $state(null);
  let text = $state('Once upon a time');
  let output = $state('');
  let status${t ? ': string' : ''} = $state('Loading model and tokenizer...');
  let generating = $state(false);

  onMount(() => {
    Promise.all([
      createSession(MODEL_PATH),
      loadTokenizer(TOKENIZER_PATH),
    ]).then(([s, tok]) => {
      session = s;
      tokenizer = tok;
      status = '${config.modelName} \\u00b7 Ready';
    }).catch((e) => { status = 'Failed to load model or tokenizer'; console.error('Load error:', e); });
  });

  async function handleGenerate() {
    const prompt = text.trim();
    if (!prompt || !session || !tokenizer) return;

    generating = true;
    status = '${config.modelName} \\u00b7 Generating...';
    const start = performance.now();

    const encoded = tokenizer.encode(prompt);
    let inputIds = encoded.inputIds;
    output = prompt;

    for (let i = 0; i < MAX_NEW_TOKENS; i++) {
      const inputBigInt = new BigInt64Array(inputIds.map((id${t ? ': number' : ''}) => BigInt(id)));
      const modelOutput = await runInference(session, inputBigInt);

      const vocabSize = tokenizer.getVocabSize();
      const seqLen = inputIds.length;
      const logits = postprocessGeneration(modelOutput, seqLen, vocabSize);
      const nextToken = sampleNextToken(logits);

      if (nextToken === EOS_TOKEN_ID) break;

      inputIds = [...inputIds, nextToken];
      const decoded = tokenizer.decode(inputIds);
      output = decoded;
    }

    const elapsed = (performance.now() - start).toFixed(1);
    status = \`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(session)}\`;
    generating = false;
  }
</script>

<a href="#output" class="skip-link">Skip to output</a>
<main>
  <h1>${config.modelName} — ${taskLabel}</h1>
  <div>
    <label for="textInput">Enter prompt</label>
    <textarea id="textInput" rows="4" bind:value={text} aria-label="Text prompt"></textarea>
    <button class="primary-btn" onclick={handleGenerate} disabled={!session || !tokenizer || generating} aria-label="Generate text">Generate</button>
  </div>
  <div id="output" class="output" role="status" aria-live="polite" aria-atomic="true">
    <pre>{output}</pre>
  </div>
</main>
<aside class="status-bar"><span>{status}</span></aside>
<div class="footer">${buildFooterText(config)}</div>
`;
}

// ---- Page dispatcher ----

function emitPageSvelte(config: ResolvedConfig): string {
  // Text/NLP tasks
  if (config.task === 'text-classification') return emitTextClassificationPageSvelte(config);
  if (config.task === 'zero-shot-classification') return emitZeroShotClassificationPageSvelte(config);
  if (config.task === 'text-generation') return emitTextGenerationPageSvelte(config);
  if (config.task === 'fill-mask') return emitFillMaskSvelte(config);
  if (config.task === 'sentence-similarity') return emitSentenceSimilaritySvelte(config);
  if (config.task === 'token-classification') return emitTokenClassificationSvelte(config);
  if (config.task === 'question-answering') return emitQuestionAnsweringSvelte(config);
  if (config.task === 'summarization') return emitSummarizationSvelte(config);
  if (config.task === 'translation') return emitTranslationSvelte(config);
  if (config.task === 'text2text-generation') return emitText2TextSvelte(config);
  if (config.task === 'conversational') return emitConversationalSvelte(config);
  if (config.task === 'table-question-answering') return emitTableQASvelte(config);

  // Audio tasks
  if (config.task === 'text-to-speech') return emitTextToSpeechPageSvelte(config);
  if (config.task === 'audio-classification') {
    if (config.input === 'mic') return emitMicAudioClassificationPageSvelte(config);
    return emitFileAudioClassificationPageSvelte(config);
  }
  if (config.task === 'speech-to-text') {
    if (config.input === 'mic') return emitMicSpeechToTextPageSvelte(config);
    return emitFileSpeechToTextPageSvelte(config);
  }
  if (config.task === 'audio-to-audio') return emitAudioToAudioSvelte(config);
  if (config.task === 'speaker-diarization') return emitSpeakerDiarizationSvelte(config);
  if (config.task === 'voice-activity-detection') return emitVADSvelte(config);

  // Multimodal tasks
  if (config.task === 'image-to-text') return emitImageToTextSvelte(config);
  if (config.task === 'visual-question-answering') return emitVQASvelte(config);
  if (config.task === 'document-question-answering') return emitDocQASvelte(config);
  if (config.task === 'image-text-to-text') return emitImageTextToTextSvelte(config);

  // Visual tasks
  if (config.input === 'camera' || config.input === 'screen') return emitRealtimePageSvelte(config);
  if (config.task === 'object-detection') return emitFileDetectionPageSvelte(config);
  if (config.task === 'image-segmentation') return emitFileSegmentationPageSvelte(config);
  if (config.task === 'depth-estimation') return emitDepthEstimationSvelte(config);
  if (config.task === 'feature-extraction') return emitFileFeatureExtractionPageSvelte(config);
  return emitFileClassificationPageSvelte(config);
}

/** Wrap a CodeBlock's code with exports for use as a lib module */
function toLibModule(block: CodeBlock | undefined): string {
  if (!block) return '';
  return addExports(block.code, block.exports);
}

/**
 * Emit SvelteKit framework files.
 */
export function emitSvelteKit(config: ResolvedConfig, blocks: CodeBlock[]): GeneratedFile[] {
  const le = libExt(config);

  const inputBlock = findBlock(blocks, 'input');
  const preprocessBlock = findBlock(blocks, 'preprocess');
  const inferenceBlock = findBlock(blocks, 'inference');
  const postprocessBlock = findBlock(blocks, 'postprocess');
  const opfsBlock = findBlock(blocks, 'opfs-cache');

  const filePaths: string[] = [
    'package.json',
    'svelte.config.js',
    'vite.config.js',
    'src/app.html',
    'src/app.css',
    'src/routes/+layout.svelte',
    'src/routes/+page.svelte',
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
    { path: 'svelte.config.js', content: emitSvelteConfig() },
    { path: 'vite.config.js', content: emitViteConfig() },
    { path: 'src/app.html', content: emitAppHtml(config) },
    { path: 'src/app.css', content: emitAppCss(config) },
    { path: 'src/routes/+layout.svelte', content: emitLayoutSvelte() },
    { path: 'src/routes/+page.svelte', content: emitPageSvelte(config) },
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
