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

function emitAppCss(config: ResolvedConfig): string {
  return `${emitDesignSystemCSS(config)}\n\n${emitAppCSS()}`;
}

function emitLayoutSvelte(): string {
  return `<script>
  import '../app.css';
  let { children } = $props();
</script>

{@render children()}
`;
}

function emitPageSvelte(config: ResolvedConfig): string {
  const le = libExt(config);
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);

  const tsLang = t ? ' lang="ts"' : '';
  const typeAnnotations = t ? {
    session: ': Awaited<ReturnType<typeof createSession>> | null',
    results: ': { indices: number[]; values: number[] } | null',
    imageUrl: ': string | null',
    status: ': string',
    file: ': File',
  } : { session: '', results: '', imageUrl: '', status: '', file: '' };

  return `<script${tsLang}>
  import { onMount } from 'svelte';
  import { createSession, runInference, getBackendLabel } from '$lib/inference.${le}';
  import { preprocessImage } from '$lib/preprocess.${le}';
  import { postprocessResults } from '$lib/postprocess.${le}';

  const MODEL_PATH = '${getModelPath(config, '')}';

  let session${typeAnnotations.session} = $state(null);
  let results${typeAnnotations.results} = $state(null);
  let imageUrl${typeAnnotations.imageUrl} = $state(null);
  let status${typeAnnotations.status} = $state('Loading model...');
  let dragOver = $state(false);
  let fileInput${t ? ': HTMLInputElement' : ''};

  onMount(() => {
    createSession(MODEL_PATH).then((s) => {
      session = s;
      status = '${config.modelName} \\u00b7 Ready';
    }).catch((e) => {
      status = 'Failed to load model';
      console.error('Model load error:', e);
    });
  });

  async function processImage(file${typeAnnotations.file}) {
    if (!file.type.startsWith('image/')) {
      status = 'Unsupported file type. Try JPG, PNG, or WebP.';
      return;
    }

    const url = URL.createObjectURL(file);
    imageUrl = url;
    results = null;

    const img = new Image();
    img.src = url;
    await new Promise((resolve) => { img.onload = resolve; });

    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d')${t ? '!' : ''};
    ctx.drawImage(img, 0, 0);
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

  function reset() {
    imageUrl = null;
    results = null;
    if (fileInput) fileInput.value = '';
  }
</script>

<a href="#results" class="skip-link">Skip to results</a>

<main>
  <h1>${config.modelName} — ${taskLabel}</h1>
  <div class="container">
    <div>
      {#if !imageUrl}
        <div
          class="drop-zone"
          class:drag-over={dragOver}
          role="button"
          tabindex="0"
          aria-label="Drop an image here or click to browse for ${taskLabel.toLowerCase()}"
          onclick={() => fileInput?.click()}
          onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput?.click(); } }}
          ondragover={(e) => { e.preventDefault(); dragOver = true; }}
          ondragleave={() => { dragOver = false; }}
          ondrop={(e) => { e.preventDefault(); dragOver = false; const f = e.dataTransfer?.files[0]; if (f) processImage(f); }}
        >
          <p>Drop an image here or click to browse</p>
          <p class="hint">Supports JPG, PNG, WebP</p>
          <input bind:this={fileInput} type="file" accept="image/*" hidden onchange={(e) => { const f = e.currentTarget.files?.[0]; if (f) processImage(f); }} />
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
            <div
              class="result-row"
              class:top-result={i === 0}
              tabindex="0"
              aria-label="Class {idx}: {pct} percent"
            >
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

<aside class="status-bar">
  <span>{status}</span>
</aside>

<div class="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
`;
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
    { path: `src/lib/inference.${le}`, content: toLibModule(inferenceBlock) },
    { path: `src/lib/postprocess.${le}`, content: toLibModule(postprocessBlock) },
    { path: 'README.md', content: emitReadme(config, filePaths) },
  );

  return files;
}
