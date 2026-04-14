/**
 * Next.js framework emitter (Layer 2).
 *
 * Produces a Next.js App Router project:
 *   package.json, next.config.mjs, tsconfig.json (TS only),
 *   app/layout.{tsx|jsx}, app/page.{tsx|jsx}, app/globals.css,
 *   lib/input.{ts|js} (if non-file), lib/preprocess.{ts|js},
 *   lib/inference.{ts|js}, lib/postprocess.{ts|js}, README.md
 *
 * Uses App Router with 'use client' for browser ML inference.
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

const ext = (config: ResolvedConfig) => (config.lang === 'ts' ? 'tsx' : 'jsx');
const libExt = (config: ResolvedConfig) => (config.lang === 'ts' ? 'ts' : 'js');

function emitPackageJson(config: ResolvedConfig, blocks: CodeBlock[]): string {
  const deps: Record<string, string> = {
    next: '^15.0.0',
    react: '^19.0.0',
    'react-dom': '^19.0.0',
  };
  for (const imp of collectImports(blocks)) {
    if (imp === 'onnxruntime-web') deps[imp] = '^1.21.0';
    else deps[imp] = 'latest';
  }

  const devDeps: Record<string, string> = {};
  if (config.lang === 'ts') {
    devDeps['typescript'] = '^5.7.0';
    devDeps['@types/react'] = '^19.0.0';
    devDeps['@types/react-dom'] = '^19.0.0';
    devDeps['@types/node'] = '^22.0.0';
  }

  const pkg: Record<string, unknown> = {
    name: config.modelName,
    private: true,
    version: '0.0.0',
    scripts: {
      dev: 'next dev',
      build: 'next build',
      start: 'next start',
    },
    dependencies: deps,
  };

  if (Object.keys(devDeps).length > 0) {
    pkg.devDependencies = devDeps;
  }

  return JSON.stringify(pkg, null, 2);
}

function emitNextConfig(): string {
  return `/** @type {import('next').NextConfig} */
const nextConfig = {
  // Configure webpack to handle ONNX WASM files
  webpack(config) {
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
      path: false,
    };
    return config;
  },
};

export default nextConfig;
`;
}

function emitTsConfig(): string {
  return JSON.stringify(
    {
      compilerOptions: {
        target: 'ES2017',
        lib: ['dom', 'dom.iterable', 'esnext'],
        allowJs: true,
        skipLibCheck: true,
        strict: true,
        noEmit: true,
        esModuleInterop: true,
        module: 'esnext',
        moduleResolution: 'bundler',
        resolveJsonModule: true,
        isolatedModules: true,
        jsx: 'preserve',
        incremental: true,
        plugins: [{ name: 'next' }],
        paths: { '@/*': ['./*'] },
      },
      include: ['next-env.d.ts', '**/*.ts', '**/*.tsx', '.next/types/**/*.ts'],
      exclude: ['node_modules'],
    },
    null,
    2,
  );
}

function emitLayout(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const taskLabel = getTaskLabel(config.task);
  const metadataExport = t
    ? `\nimport type { Metadata } from 'next';\n\nexport const metadata: Metadata = {
  title: '${config.modelName} — ${taskLabel}',
  description: '${taskLabel} powered by ${getEngineLabel(config.engine)}',
};\n`
    : `\nexport const metadata = {
  title: '${config.modelName} — ${taskLabel}',
  description: '${taskLabel} powered by ${getEngineLabel(config.engine)}',
};\n`;

  const childrenType = t ? ': { children: React.ReactNode }' : '';

  return `import './globals.css';
${metadataExport}
export default function RootLayout({ children }${childrenType}) {
  return (
    <html lang="en" data-theme="${config.theme}">
      <body>{children}</body>
    </html>
  );
}
`;
}

// ---- Page emitters by task/input ----

function emitFileClassificationPage(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);

  const stateType = t ? '<{ indices: number[]; values: number[] } | null>' : '';
  const refType = t ? '<HTMLInputElement | null>' : '';
  const eventType = t ? ': React.DragEvent' : '';
  const changeType = t ? ': React.ChangeEvent<HTMLInputElement>' : '';

  return `'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from '../lib/inference.${le}';
import { preprocessImage } from '../lib/preprocess.${le}';
import { postprocessResults } from '../lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';

export default function Page() {
  const [results, setResults] = useState${stateType}(null);
  const [status, setStatus] = useState('Loading model...');
  const [imageUrl, setImageUrl] = useState${t ? '<string | null>' : ''}(null);
  const [dragOver, setDragOver] = useState(false);
  const sessionRef = useRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
  const fileInputRef = useRef${refType}(null);

  useEffect(() => {
    createSession(MODEL_PATH).then((s) => {
      sessionRef.current = s;
      setStatus('${config.modelName} \\u00b7 Ready');
    }).catch((e) => { setStatus('Failed to load model'); console.error('Model load error:', e); });
  }, []);

  const processImage = useCallback(async (file${t ? ': File' : ''}) => {
    if (!file.type.startsWith('image/')) { setStatus('Unsupported file type.'); return; }
    const url = URL.createObjectURL(file);
    setImageUrl(url); setResults(null);
    const img = new Image(); img.src = url;
    await new Promise((resolve) => { img.onload = resolve; });
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d')${t ? '!' : ''}; ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    if (!sessionRef.current) { setStatus('Model not loaded yet.'); return; }
    setStatus('${config.modelName} \\u00b7 Processing...');
    const start = performance.now();
    const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
    const output = await runInference(sessionRef.current, inputTensor);
    const r = postprocessResults(output);
    const elapsed = (performance.now() - start).toFixed(1);
    setStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(sessionRef.current)}\`);
    setResults(r);
    URL.revokeObjectURL(url);
  }, []);

  const handleDrop = useCallback((e${eventType}) => { e.preventDefault(); setDragOver(false); const file = e.dataTransfer.files[0]; if (file) processImage(file); }, [processImage]);
  const handleFileChange = useCallback((e${changeType}) => { const file = e.target.files?.[0]; if (file) processImage(file); }, [processImage]);
  const reset = useCallback(() => { setImageUrl(null); setResults(null); if (fileInputRef.current) fileInputRef.current.value = ''; }, []);

  return (
    <>
      <a href="#results" className="skip-link">Skip to results</a>
      <main>
        <h1>${config.modelName} — ${taskLabel}</h1>
        <div className="container">
          <div>
            {!imageUrl ? (
              <div className={\`drop-zone\${dragOver ? ' drag-over' : ''}\`} role="button" tabIndex={0}
                aria-label="Drop an image here or click to browse for ${taskLabel.toLowerCase()}"
                onClick={() => fileInputRef.current?.click()}
                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInputRef.current?.click(); } }}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)} onDrop={handleDrop}>
                <p>Drop an image here or click to browse</p>
                <p className="hint">Supports JPG, PNG, WebP</p>
                <input ref={fileInputRef} type="file" accept="image/*" hidden onChange={handleFileChange} />
              </div>
            ) : (
              <div className="preview">
                <img src={imageUrl} alt="Selected image for classification" />
                <button className="change-btn" onClick={reset}>Choose another image</button>
              </div>
            )}
          </div>
          <div id="results" className="results" role="status" aria-live="polite" aria-atomic="true">
            {results && results.indices.map((idx${t ? ': number' : ''}, i${t ? ': number' : ''}) => {
              const pct = (results.values[i] * 100).toFixed(1);
              if (results.values[i] < 0.01) return null;
              const maxVal = results.values[0] || 1;
              return (
                <div key={idx} className={\`result-row\${i === 0 ? ' top-result' : ''}\`} tabIndex={0}
                     aria-label={\`Class \${idx}: \${pct} percent\`}>
                  <span className="result-label">Class {idx}</span>
                  <div className="result-bar-container">
                    <div className="result-bar" style={{ width: \`\${(results.values[i] / maxVal) * 100}%\` }} />
                  </div>
                  <span className="result-pct">{pct}%</span>
                </div>
              );
            })}
          </div>
        </div>
      </main>
      <aside className="status-bar"><span>{status}</span></aside>
      <div className="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
    </>
  );
}
`;
}

function emitFileDetectionPage(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const outputShape = config.modelMeta.outputs[0]?.shape ?? [1, 84, 8400];
  const numAttributes = outputShape[1] ?? 84;
  const numAnchors = outputShape[2] ?? 8400;

  const boxType = t ? 'interface Box { x: number; y: number; width: number; height: number; classIndex: number; score: number; }\n\n' : '';
  const stateType = t ? '<Box[] | null>' : '';
  const refType = t ? '<HTMLInputElement | null>' : '';
  const canvasRefType = t ? '<HTMLCanvasElement | null>' : '';
  const eventType = t ? ': React.DragEvent' : '';
  const changeType = t ? ': React.ChangeEvent<HTMLInputElement>' : '';

  return `'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from '../lib/inference.${le}';
import { preprocessImage } from '../lib/preprocess.${le}';
import { postprocessDetections } from '../lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
const NUM_ATTRIBUTES = ${numAttributes};
const NUM_ANCHORS = ${numAnchors};
const COLORS = [[56,189,248],[249,115,22],[34,197,94],[168,85,247],[251,191,36],[239,68,68],[20,184,166],[236,72,153],[99,102,241],[163,230,53]];

${boxType}export default function Page() {
  const [boxes, setBoxes] = useState${stateType}(null);
  const [status, setStatus] = useState('Loading model...');
  const [imageUrl, setImageUrl] = useState${t ? '<string | null>' : ''}(null);
  const [dragOver, setDragOver] = useState(false);
  const [imgSize, setImgSize] = useState${t ? '<{ w: number; h: number }>' : ''}({ w: 0, h: 0 });
  const sessionRef = useRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
  const fileInputRef = useRef${refType}(null);
  const overlayRef = useRef${canvasRefType}(null);

  useEffect(() => {
    createSession(MODEL_PATH).then((s) => {
      sessionRef.current = s;
      setStatus('${config.modelName} \\u00b7 Ready');
    }).catch((e) => { setStatus('Failed to load model'); console.error('Model load error:', e); });
  }, []);

  useEffect(() => {
    if (!boxes || !overlayRef.current) return;
    const canvas = overlayRef.current;
    canvas.width = imgSize.w; canvas.height = imgSize.h;
    const ctx = canvas.getContext('2d')${t ? '!' : ''};
    ctx.clearRect(0, 0, canvas.width, canvas.height);
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
  }, [boxes, imgSize]);

  const processImage = useCallback(async (file${t ? ': File' : ''}) => {
    if (!file.type.startsWith('image/')) { setStatus('Unsupported file type.'); return; }
    const url = URL.createObjectURL(file);
    setImageUrl(url); setBoxes(null);
    const img = new Image(); img.src = url;
    await new Promise((resolve) => { img.onload = resolve; });
    setImgSize({ w: img.naturalWidth, h: img.naturalHeight });
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d')${t ? '!' : ''}; ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    if (!sessionRef.current) { setStatus('Model not loaded yet.'); return; }
    setStatus('${config.modelName} \\u00b7 Processing...');
    const start = performance.now();
    const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
    const output = await runInference(sessionRef.current, inputTensor);
    const detected = postprocessDetections(output, NUM_ANCHORS, NUM_ATTRIBUTES);
    const elapsed = (performance.now() - start).toFixed(1);
    setStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(sessionRef.current)}\`);
    setBoxes(detected);
    URL.revokeObjectURL(url);
  }, []);

  const handleDrop = useCallback((e${eventType}) => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files[0]; if (f) processImage(f); }, [processImage]);
  const handleFileChange = useCallback((e${changeType}) => { const f = e.target.files?.[0]; if (f) processImage(f); }, [processImage]);
  const reset = useCallback(() => { setImageUrl(null); setBoxes(null); if (fileInputRef.current) fileInputRef.current.value = ''; }, []);

  return (
    <>
      <a href="#results" className="skip-link">Skip to results</a>
      <main>
        <h1>${config.modelName} — ${taskLabel}</h1>
        <div className="container">
          <div>
            {!imageUrl ? (
              <div className={\`drop-zone\${dragOver ? ' drag-over' : ''}\`} role="button" tabIndex={0}
                aria-label="Drop an image here or click to browse for ${taskLabel.toLowerCase()}"
                onClick={() => fileInputRef.current?.click()}
                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInputRef.current?.click(); } }}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)} onDrop={handleDrop}>
                <p>Drop an image here or click to browse</p>
                <p className="hint">Supports JPG, PNG, WebP</p>
                <input ref={fileInputRef} type="file" accept="image/*" hidden onChange={handleFileChange} />
              </div>
            ) : (
              <div className="preview">
                <div className="preview-wrapper">
                  <img src={imageUrl} alt="Selected image for detection" />
                  <canvas ref={overlayRef} className="overlay-canvas" />
                </div>
                <button className="change-btn" onClick={reset}>Choose another image</button>
              </div>
            )}
          </div>
          <div id="results" className="results" role="status" aria-live="polite" aria-atomic="true">
            {boxes && boxes.length === 0 && <p>No detections found.</p>}
            {boxes && boxes.map((box${t ? ': Box' : ''}, i${t ? ': number' : ''}) => (
              <div key={i} className="result-row" tabIndex={0} aria-label={\`Class \${box.classIndex}: \${(box.score * 100).toFixed(0)} percent\`}>
                <span className="result-label">Class {box.classIndex} ({(box.score * 100).toFixed(0)}%)</span>
              </div>
            ))}
          </div>
        </div>
      </main>
      <aside className="status-bar"><span>{status}</span></aside>
      <div className="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
    </>
  );
}
`;
}

function emitFileSegmentationPage(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const outputShape = config.modelMeta.outputs[0]?.shape ?? [1, 21, 512, 512];
  const numClasses = outputShape[1] ?? 21;
  const maskH = outputShape[2] ?? 512;
  const maskW = outputShape[3] ?? 512;

  const stateType = t ? '<Uint8Array | null>' : '';
  const refType = t ? '<HTMLInputElement | null>' : '';
  const canvasRefType = t ? '<HTMLCanvasElement | null>' : '';
  const eventType = t ? ': React.DragEvent' : '';
  const changeType = t ? ': React.ChangeEvent<HTMLInputElement>' : '';

  return `'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from '../lib/inference.${le}';
import { preprocessImage } from '../lib/preprocess.${le}';
import { postprocessSegmentation } from '../lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
const NUM_CLASSES = ${numClasses};
const MASK_H = ${maskH};
const MASK_W = ${maskW};
const COLORS = [[56,189,248],[249,115,22],[34,197,94],[168,85,247],[251,191,36],[239,68,68],[20,184,166],[236,72,153],[99,102,241],[163,230,53]];

export default function Page() {
  const [mask, setMask] = useState${stateType}(null);
  const [status, setStatus] = useState('Loading model...');
  const [imageUrl, setImageUrl] = useState${t ? '<string | null>' : ''}(null);
  const [dragOver, setDragOver] = useState(false);
  const [imgSize, setImgSize] = useState${t ? '<{ w: number; h: number }>' : ''}({ w: 0, h: 0 });
  const sessionRef = useRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
  const fileInputRef = useRef${refType}(null);
  const overlayRef = useRef${canvasRefType}(null);

  useEffect(() => {
    createSession(MODEL_PATH).then((s) => {
      sessionRef.current = s;
      setStatus('${config.modelName} \\u00b7 Ready');
    }).catch((e) => { setStatus('Failed to load model'); console.error('Model load error:', e); });
  }, []);

  useEffect(() => {
    if (!mask || !overlayRef.current) return;
    const canvas = overlayRef.current;
    canvas.width = imgSize.w; canvas.height = imgSize.h;
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
    const ctx = canvas.getContext('2d')${t ? '!' : ''};
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(maskCanvas, 0, 0, imgSize.w, imgSize.h);
  }, [mask, imgSize]);

  const processImage = useCallback(async (file${t ? ': File' : ''}) => {
    if (!file.type.startsWith('image/')) { setStatus('Unsupported file type.'); return; }
    const url = URL.createObjectURL(file);
    setImageUrl(url); setMask(null);
    const img = new Image(); img.src = url;
    await new Promise((resolve) => { img.onload = resolve; });
    setImgSize({ w: img.naturalWidth, h: img.naturalHeight });
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d')${t ? '!' : ''}; ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    if (!sessionRef.current) { setStatus('Model not loaded yet.'); return; }
    setStatus('${config.modelName} \\u00b7 Processing...');
    const start = performance.now();
    const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
    const output = await runInference(sessionRef.current, inputTensor);
    const m = postprocessSegmentation(output, NUM_CLASSES, MASK_H, MASK_W);
    const elapsed = (performance.now() - start).toFixed(1);
    setStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(sessionRef.current)}\`);
    setMask(m);
    URL.revokeObjectURL(url);
  }, []);

  const handleDrop = useCallback((e${eventType}) => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files[0]; if (f) processImage(f); }, [processImage]);
  const handleFileChange = useCallback((e${changeType}) => { const f = e.target.files?.[0]; if (f) processImage(f); }, [processImage]);
  const reset = useCallback(() => { setImageUrl(null); setMask(null); if (fileInputRef.current) fileInputRef.current.value = ''; }, []);

  return (
    <>
      <a href="#results" className="skip-link">Skip to results</a>
      <main>
        <h1>${config.modelName} — ${taskLabel}</h1>
        <div className="container">
          <div>
            {!imageUrl ? (
              <div className={\`drop-zone\${dragOver ? ' drag-over' : ''}\`} role="button" tabIndex={0}
                aria-label="Drop an image here or click to browse for ${taskLabel.toLowerCase()}"
                onClick={() => fileInputRef.current?.click()}
                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInputRef.current?.click(); } }}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)} onDrop={handleDrop}>
                <p>Drop an image here or click to browse</p>
                <p className="hint">Supports JPG, PNG, WebP</p>
                <input ref={fileInputRef} type="file" accept="image/*" hidden onChange={handleFileChange} />
              </div>
            ) : (
              <div className="preview">
                <div className="preview-wrapper">
                  <img src={imageUrl} alt="Selected image for segmentation" />
                  <canvas ref={overlayRef} className="overlay-canvas" />
                </div>
                <button className="change-btn" onClick={reset}>Choose another image</button>
              </div>
            )}
          </div>
          <div id="results" className="results" role="status" aria-live="polite" aria-atomic="true">
          </div>
        </div>
      </main>
      <aside className="status-bar"><span>{status}</span></aside>
      <div className="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
    </>
  );
}
`;
}

function emitFileFeatureExtractionPage(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);

  const stateType = t ? '<{ dims: number; norm: string; first5: string } | null>' : '';
  const refType = t ? '<HTMLInputElement | null>' : '';
  const eventType = t ? ': React.DragEvent' : '';
  const changeType = t ? ': React.ChangeEvent<HTMLInputElement>' : '';

  return `'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from '../lib/inference.${le}';
import { preprocessImage } from '../lib/preprocess.${le}';
import { postprocessEmbeddings } from '../lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';

export default function Page() {
  const [embedding, setEmbedding] = useState${stateType}(null);
  const [status, setStatus] = useState('Loading model...');
  const [imageUrl, setImageUrl] = useState${t ? '<string | null>' : ''}(null);
  const [dragOver, setDragOver] = useState(false);
  const sessionRef = useRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
  const fileInputRef = useRef${refType}(null);

  useEffect(() => {
    createSession(MODEL_PATH).then((s) => {
      sessionRef.current = s;
      setStatus('${config.modelName} \\u00b7 Ready');
    }).catch((e) => { setStatus('Failed to load model'); console.error('Model load error:', e); });
  }, []);

  const processImage = useCallback(async (file${t ? ': File' : ''}) => {
    if (!file.type.startsWith('image/')) { setStatus('Unsupported file type.'); return; }
    const url = URL.createObjectURL(file);
    setImageUrl(url); setEmbedding(null);
    const img = new Image(); img.src = url;
    await new Promise((resolve) => { img.onload = resolve; });
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d')${t ? '!' : ''}; ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    if (!sessionRef.current) { setStatus('Model not loaded yet.'); return; }
    setStatus('${config.modelName} \\u00b7 Processing...');
    const start = performance.now();
    const inputTensor = preprocessImage(imageData.data, canvas.width, canvas.height);
    const output = await runInference(sessionRef.current, inputTensor);
    const emb = postprocessEmbeddings(output);
    let norm = 0;
    for (let i = 0; i < emb.length; i++) { norm += emb[i] * emb[i]; }
    norm = Math.sqrt(norm);
    const first5 = Array.from(emb.slice(0, 5)).map((v${t ? ': number' : ''}) => v.toFixed(4)).join(', ');
    const elapsed = (performance.now() - start).toFixed(1);
    setStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(sessionRef.current)}\`);
    setEmbedding({ dims: emb.length, norm: norm.toFixed(4), first5 });
    URL.revokeObjectURL(url);
  }, []);

  const handleDrop = useCallback((e${eventType}) => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files[0]; if (f) processImage(f); }, [processImage]);
  const handleFileChange = useCallback((e${changeType}) => { const f = e.target.files?.[0]; if (f) processImage(f); }, [processImage]);
  const reset = useCallback(() => { setImageUrl(null); setEmbedding(null); if (fileInputRef.current) fileInputRef.current.value = ''; }, []);

  return (
    <>
      <a href="#results" className="skip-link">Skip to results</a>
      <main>
        <h1>${config.modelName} — ${taskLabel}</h1>
        <div className="container">
          <div>
            {!imageUrl ? (
              <div className={\`drop-zone\${dragOver ? ' drag-over' : ''}\`} role="button" tabIndex={0}
                aria-label="Drop an image here or click to browse for ${taskLabel.toLowerCase()}"
                onClick={() => fileInputRef.current?.click()}
                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInputRef.current?.click(); } }}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)} onDrop={handleDrop}>
                <p>Drop an image here or click to browse</p>
                <p className="hint">Supports JPG, PNG, WebP</p>
                <input ref={fileInputRef} type="file" accept="image/*" hidden onChange={handleFileChange} />
              </div>
            ) : (
              <div className="preview">
                <img src={imageUrl} alt="Selected image for feature extraction" />
                <button className="change-btn" onClick={reset}>Choose another image</button>
              </div>
            )}
          </div>
          <div id="results" className="results embedding-info" role="status" aria-live="polite" aria-atomic="true">
            {embedding && (
              <>
                <p><strong>Dimensions:</strong> {embedding.dims}</p>
                <p><strong>L2 Norm:</strong> {embedding.norm}</p>
                <p><strong>First 5 values:</strong> [{embedding.first5}, ...]</p>
              </>
            )}
          </div>
        </div>
      </main>
      <aside className="status-bar"><span>{status}</span></aside>
      <div className="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
    </>
  );
}
`;
}

function emitRealtimePage(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
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
      postImport = `import { postprocessDetections } from '../lib/postprocess.${le}';`;
      extraConst = `const NUM_ATTRIBUTES = ${numAttributes};\nconst NUM_ANCHORS = ${numAnchors};\nconst COLORS = [[56,189,248],[249,115,22],[34,197,94],[168,85,247],[251,191,36],[239,68,68],[20,184,166],[236,72,153],[99,102,241],[163,230,53]];`;
      processAndRender = `const boxes = postprocessDetections(output, NUM_ANCHORS, NUM_ATTRIBUTES);
        const elapsed = performance.now() - start;
        // Draw bounding boxes
        const modelSize = ${config.preprocess.imageSize};
        const scaleX = imageData.width / modelSize;
        const scaleY = imageData.height / modelSize;
        overlayCtx.clearRect(0, 0, overlayRef.current${t ? '!' : ''}.width, overlayRef.current${t ? '!' : ''}.height);
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
      postImport = `import { postprocessSegmentation } from '../lib/postprocess.${le}';`;
      extraConst = `const NUM_CLASSES = ${numClasses};\nconst MASK_H = ${maskH};\nconst MASK_W = ${maskW};\nconst COLORS = [[56,189,248],[249,115,22],[34,197,94],[168,85,247],[251,191,36],[239,68,68],[20,184,166],[236,72,153],[99,102,241],[163,230,53]];`;
      processAndRender = `const mask = postprocessSegmentation(output, NUM_CLASSES, MASK_H, MASK_W);
        const elapsed = performance.now() - start;
        // Draw mask overlay
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
        overlayCtx.clearRect(0, 0, overlayRef.current${t ? '!' : ''}.width, overlayRef.current${t ? '!' : ''}.height);
        overlayCtx.drawImage(maskCanvas, 0, 0, videoRef.current${t ? '!' : ''}.videoWidth, videoRef.current${t ? '!' : ''}.videoHeight);`;
      break;
    }
    default: {
      postImport = `import { postprocessResults } from '../lib/postprocess.${le}';`;
      processAndRender = `const results = postprocessResults(output);
        const elapsed = performance.now() - start;
        const lbl = 'Class ' + results.indices[0] + ' (' + (results.values[0] * 100).toFixed(1) + '%)';
        overlayCtx.clearRect(0, 0, overlayRef.current${t ? '!' : ''}.width, overlayRef.current${t ? '!' : ''}.height);
        overlayCtx.font = 'bold 24px system-ui, sans-serif';
        overlayCtx.fillStyle = 'rgba(59, 130, 246, 0.85)';
        const tw = overlayCtx.measureText(lbl).width;
        overlayCtx.fillRect(8, 8, tw + 16, 36);
        overlayCtx.fillStyle = '#fff';
        overlayCtx.fillText(lbl, 16, 34);`;
      break;
    }
  }

  return `'use client';

import { useState, useEffect, useRef } from 'react';
import { createSession, runInference, getBackendLabel } from '../lib/inference.${le}';
import { preprocessImage } from '../lib/preprocess.${le}';
${postImport}
import { ${startFn}, captureFrame, createInferenceLoop } from '../lib/input.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
${extraConst}

export default function Page() {
  const [status, setStatus] = useState('Loading model...');
  const [started, setStarted] = useState(false);
  const sessionRef = useRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
  const videoRef = useRef${t ? '<HTMLVideoElement | null>' : ''}(null);
  const overlayRef = useRef${t ? '<HTMLCanvasElement | null>' : ''}(null);
  const loopRef = useRef${t ? '<ReturnType<typeof createInferenceLoop> | null>' : ''}(null);

  useEffect(() => {
    createSession(MODEL_PATH).then((s) => {
      sessionRef.current = s;
      setStatus('${config.modelName} \\u00b7 Ready \\u00b7 Tap Start');
    }).catch((e) => { setStatus('Failed to load model'); console.error('Model load error:', e); });
  }, []);

  const handleStart = async () => {
    if (!videoRef.current || !overlayRef.current) return;
    try {
      await ${startFn}(videoRef.current);
      overlayRef.current.width = videoRef.current.videoWidth;
      overlayRef.current.height = videoRef.current.videoHeight;
      setStarted(true);

      const overlayCtx = overlayRef.current.getContext('2d')${t ? '!' : ''};
      loopRef.current = createInferenceLoop({
        video: videoRef.current,
        canvas: overlayRef.current,
        async onFrame(imageData${t ? ': ImageData' : ''}) {
          const start = performance.now();
          const inputTensor = preprocessImage(imageData.data, imageData.width, imageData.height);
          const output = await runInference(sessionRef.current${t ? '!' : ''}, inputTensor);
          ${processAndRender}
          return { result: null, elapsed };
        },
        onStatus(elapsed${t ? ': number' : ''}) {
          setStatus('${config.modelName} \\u00b7 ' + elapsed.toFixed(1) + 'ms \\u00b7 ' + getBackendLabel(sessionRef.current${t ? '!' : ''}));
        },
      });
      loopRef.current.start();
    } catch (e) {
      setStatus('${label} access denied');
      console.error('${label} error:', e);
    }
  };

  const handlePause = () => {
    if (loopRef.current) { loopRef.current.stop(); loopRef.current = null; }
  };

  return (
    <>
      <a href="#results" className="skip-link">Skip to results</a>
      <main>
        <h1>${config.modelName} — ${taskLabel}</h1>
        {!started ? (
          <div className="permission-prompt">
            <p>This app ${actionLabel} to run ${taskLabel.toLowerCase()} in real time.</p>
            <p className="hint">No video is recorded or sent anywhere.</p>
            <button className="primary-btn" onClick={handleStart}>${btnLabel}</button>
          </div>
        ) : (
          <div>
            <div className="video-wrapper">
              <video ref={videoRef} autoPlay playsInline muted />
              <canvas ref={overlayRef} />
            </div>
            <div className="controls">
              <button className="control-btn" onClick={handlePause}>\\u23f8 Pause</button>
            </div>
          </div>
        )}
      </main>
      <aside className="status-bar"><span>{status}</span></aside>
      <div className="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
    </>
  );
}
`;
}

// ---- Audio: File + Classification Page ----

function emitFileAudioClassificationPage(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);

  const stateType = t ? '<{ indices: number[]; values: number[] } | null>' : '';

  return `'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from '../lib/inference.${le}';
import { melSpectrogram, mfcc } from '../lib/preprocess.${le}';
import { postprocessResults } from '../lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';

export default function Page() {
  const [results, setResults] = useState${stateType}(null);
  const [status, setStatus] = useState('Loading model...');
  const sessionRef = useRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);

  useEffect(() => {
    createSession(MODEL_PATH).then((s) => {
      sessionRef.current = s;
      setStatus('${config.modelName} \\u00b7 Ready');
    }).catch((e) => { setStatus('Failed to load model'); console.error('Model load error:', e); });
  }, []);

  const handleFileChange = useCallback(async (e${t ? ': React.ChangeEvent<HTMLInputElement>' : ''}) => {
    const file = e.target.files?.[0];
    if (!file || !sessionRef.current) return;

    setStatus('${config.modelName} \\u00b7 Decoding audio...');

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

    setStatus('${config.modelName} \\u00b7 Processing...');
    const start = performance.now();

    const mel = melSpectrogram(samples, 16000, 512, 160, 40);
    const features = mfcc(mel.data, mel.numFrames, mel.numMelBands, 13);
    const output = await runInference(sessionRef.current, features);
    const r = postprocessResults(output);

    const elapsed = (performance.now() - start).toFixed(1);
    setStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(sessionRef.current)}\`);
    setResults(r);
  }, []);

  return (
    <>
      <a href="#results" className="skip-link">Skip to results</a>
      <main>
        <h1>${config.modelName} — ${taskLabel}</h1>
        <div>
          <label htmlFor="fileInput">Choose an audio file</label>
          <input id="fileInput" type="file" accept="audio/*" onChange={handleFileChange} aria-label="Select audio file for ${taskLabel.toLowerCase()}" />
        </div>
        <div id="results" className="results" role="status" aria-live="polite" aria-atomic="true">
          {results && results.indices.map((idx${t ? ': number' : ''}, i${t ? ': number' : ''}) => {
            const pct = (results.values[i] * 100).toFixed(1);
            if (results.values[i] < 0.01) return null;
            const maxVal = results.values[0] || 1;
            return (
              <div key={idx} className={\`result-row\${i === 0 ? ' top-result' : ''}\`} tabIndex={0}
                   aria-label={\`Class \${idx}: \${pct} percent\`}>
                <span className="result-label">Class {idx}</span>
                <div className="result-bar-container">
                  <div className="result-bar" style={{ width: \`\${(results.values[i] / maxVal) * 100}%\` }} />
                </div>
                <span className="result-pct">{pct}%</span>
              </div>
            );
          })}
        </div>
      </main>
      <aside className="status-bar"><span>{status}</span></aside>
      <div className="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
    </>
  );
}
`;
}

// ---- Audio: File + STT Page ----

function emitFileSpeechToTextPage(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);

  return `'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from '../lib/inference.${le}';
import { melSpectrogram } from '../lib/preprocess.${le}';
import { postprocessTranscript } from '../lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
const VOCAB = [' ', ...'abcdefghijklmnopqrstuvwxyz'.split(''), "'"];

export default function Page() {
  const [transcript, setTranscript] = useState('');
  const [status, setStatus] = useState('Loading model...');
  const sessionRef = useRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);

  useEffect(() => {
    createSession(MODEL_PATH).then((s) => {
      sessionRef.current = s;
      setStatus('${config.modelName} \\u00b7 Ready');
    }).catch((e) => { setStatus('Failed to load model'); console.error('Model load error:', e); });
  }, []);

  const handleFileChange = useCallback(async (e${t ? ': React.ChangeEvent<HTMLInputElement>' : ''}) => {
    const file = e.target.files?.[0];
    if (!file || !sessionRef.current) return;

    setStatus('${config.modelName} \\u00b7 Decoding audio...');

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

    setStatus('${config.modelName} \\u00b7 Processing...');
    const start = performance.now();

    const mel = melSpectrogram(samples, 16000, 512, 160, 80);
    const output = await runInference(sessionRef.current, mel.data);
    const vocabSize = VOCAB.length + 1;
    const numTimesteps = Math.floor(output.length / vocabSize);
    const text = postprocessTranscript(output, numTimesteps, vocabSize, VOCAB);

    const elapsed = (performance.now() - start).toFixed(1);
    setStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(sessionRef.current)}\`);
    setTranscript(text || '(no speech detected)');
  }, []);

  return (
    <>
      <a href="#transcript" className="skip-link">Skip to results</a>
      <main>
        <h1>${config.modelName} — ${taskLabel}</h1>
        <div>
          <label htmlFor="fileInput">Choose an audio file</label>
          <input id="fileInput" type="file" accept="audio/*" onChange={handleFileChange} aria-label="Select audio file for ${taskLabel.toLowerCase()}" />
        </div>
        <pre id="transcript" className="transcript" role="status" aria-live="polite" aria-atomic="true">{transcript}</pre>
      </main>
      <aside className="status-bar"><span>{status}</span></aside>
      <div className="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
    </>
  );
}
`;
}

// ---- Audio: Mic + STT Page ----

function emitMicSpeechToTextPage(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);

  return `'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from '../lib/inference.${le}';
import { melSpectrogram } from '../lib/preprocess.${le}';
import { postprocessTranscript } from '../lib/postprocess.${le}';
import { startAudioCapture, stopStream, createAudioInferenceLoop } from '../lib/input.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
const VOCAB = [' ', ...'abcdefghijklmnopqrstuvwxyz'.split(''), "'"];

export default function Page() {
  const [transcript, setTranscript] = useState('(listening...)');
  const [status, setStatus] = useState('Loading model...');
  const [recording, setRecording] = useState(false);
  const sessionRef = useRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
  const captureRef = useRef${t ? '<any>' : ''}(null);
  const loopRef = useRef${t ? '<any>' : ''}(null);

  useEffect(() => {
    createSession(MODEL_PATH).then((s) => {
      sessionRef.current = s;
      setStatus('${config.modelName} \\u00b7 Ready');
    }).catch((e) => { setStatus('Failed to load model'); console.error('Model load error:', e); });
    return () => {
      if (loopRef.current) loopRef.current.stop();
      if (captureRef.current) { stopStream(captureRef.current.stream); captureRef.current.audioContext.close(); }
    };
  }, []);

  async function processAudio(samples${t ? ': Float32Array' : ''}) {
    const mel = melSpectrogram(samples, 16000, 512, 160, 80);
    const output = await runInference(sessionRef.current${t ? '!' : ''}, mel.data);
    const vocabSize = VOCAB.length + 1;
    const numTimesteps = Math.floor(output.length / vocabSize);
    return postprocessTranscript(output, numTimesteps, vocabSize, VOCAB);
  }

  const handleStart = useCallback(async () => {
    if (!sessionRef.current) { setStatus('Model not loaded yet.'); return; }
    try {
      captureRef.current = await startAudioCapture(16000);
      setRecording(true);
      setStatus('${config.modelName} \\u00b7 Listening...');
      loopRef.current = createAudioInferenceLoop({
        getSamples: captureRef.current.getSamples,
        onResult(text${t ? ': string' : ''}) { setTranscript(text || '(listening...)'); },
        intervalMs: 2000,
      });
      loopRef.current.start();
    } catch (e) { setStatus('Microphone access denied'); console.error('Mic error:', e); }
  }, []);

  const handleStop = useCallback(() => {
    if (loopRef.current) { loopRef.current.stop(); loopRef.current = null; }
    if (captureRef.current) { stopStream(captureRef.current.stream); captureRef.current.audioContext.close(); captureRef.current = null; }
    setRecording(false);
    setStatus('${config.modelName} \\u00b7 Stopped');
  }, []);

  return (
    <>
      <a href="#transcript" className="skip-link">Skip to results</a>
      <main>
        <h1>${config.modelName} — ${taskLabel}</h1>
        <div className="controls" role="group" aria-label="Recording controls">
          <button onClick={handleStart} disabled={recording} aria-label="Start recording">Start Recording</button>
          <button onClick={handleStop} disabled={!recording} aria-label="Stop recording">Stop Recording</button>
        </div>
        <pre id="transcript" className="transcript" role="status" aria-live="polite" aria-atomic="true">{transcript}</pre>
      </main>
      <aside className="status-bar"><span>{status}</span></aside>
      <div className="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
    </>
  );
}
`;
}

// ---- Audio: Mic + Classification Page ----

function emitMicAudioClassificationPage(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);

  const stateType = t ? '<{ indices: number[]; values: number[] } | null>' : '';

  return `'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from '../lib/inference.${le}';
import { melSpectrogram, mfcc } from '../lib/preprocess.${le}';
import { postprocessResults } from '../lib/postprocess.${le}';
import { startAudioCapture, stopStream, createAudioInferenceLoop } from '../lib/input.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';

export default function Page() {
  const [results, setResults] = useState${stateType}(null);
  const [status, setStatus] = useState('Loading model...');
  const [recording, setRecording] = useState(false);
  const sessionRef = useRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
  const captureRef = useRef${t ? '<any>' : ''}(null);
  const loopRef = useRef${t ? '<any>' : ''}(null);

  useEffect(() => {
    createSession(MODEL_PATH).then((s) => {
      sessionRef.current = s;
      setStatus('${config.modelName} \\u00b7 Ready');
    }).catch((e) => { setStatus('Failed to load model'); console.error('Model load error:', e); });
    return () => {
      if (loopRef.current) loopRef.current.stop();
      if (captureRef.current) { stopStream(captureRef.current.stream); captureRef.current.audioContext.close(); }
    };
  }, []);

  async function processAudio(samples${t ? ': Float32Array' : ''}) {
    const mel = melSpectrogram(samples, 16000, 512, 160, 40);
    const features = mfcc(mel.data, mel.numFrames, mel.numMelBands, 13);
    const output = await runInference(sessionRef.current${t ? '!' : ''}, features);
    return postprocessResults(output);
  }

  const handleStart = useCallback(async () => {
    if (!sessionRef.current) { setStatus('Model not loaded yet.'); return; }
    try {
      captureRef.current = await startAudioCapture(16000);
      setRecording(true);
      setStatus('${config.modelName} \\u00b7 Listening...');
      loopRef.current = createAudioInferenceLoop({
        getSamples: captureRef.current.getSamples,
        onResult(r${t ? ': { indices: number[]; values: number[] }' : ''}) { setResults(r); },
        intervalMs: 2000,
      });
      loopRef.current.start();
    } catch (e) { setStatus('Microphone access denied'); console.error('Mic error:', e); }
  }, []);

  const handleStop = useCallback(() => {
    if (loopRef.current) { loopRef.current.stop(); loopRef.current = null; }
    if (captureRef.current) { stopStream(captureRef.current.stream); captureRef.current.audioContext.close(); captureRef.current = null; }
    setRecording(false);
    setStatus('${config.modelName} \\u00b7 Stopped');
  }, []);

  return (
    <>
      <a href="#results" className="skip-link">Skip to results</a>
      <main>
        <h1>${config.modelName} — ${taskLabel}</h1>
        <div className="controls" role="group" aria-label="Recording controls">
          <button onClick={handleStart} disabled={recording} aria-label="Start recording">Start Recording</button>
          <button onClick={handleStop} disabled={!recording} aria-label="Stop recording">Stop Recording</button>
        </div>
        <div id="results" className="results" role="status" aria-live="polite" aria-atomic="true">
          {results && results.indices.map((idx${t ? ': number' : ''}, i${t ? ': number' : ''}) => {
            const pct = (results.values[i] * 100).toFixed(1);
            if (results.values[i] < 0.01) return null;
            const maxVal = results.values[0] || 1;
            return (
              <div key={idx} className={\`result-row\${i === 0 ? ' top-result' : ''}\`} tabIndex={0}
                   aria-label={\`Class \${idx}: \${pct} percent\`}>
                <span className="result-label">Class {idx}</span>
                <div className="result-bar-container">
                  <div className="result-bar" style={{ width: \`\${(results.values[i] / maxVal) * 100}%\` }} />
                </div>
                <span className="result-pct">{pct}%</span>
              </div>
            );
          })}
        </div>
      </main>
      <aside className="status-bar"><span>{status}</span></aside>
      <div className="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
    </>
  );
}
`;
}

// ---- Audio: Text-to-Speech Page ----

function emitTextToSpeechPage(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);

  return `'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from '../lib/inference.${le}';
import { postprocessAudio, playAudio } from '../lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';

export default function Page() {
  const [status, setStatus] = useState('Loading model...');
  const [text, setText] = useState('Hello, this is a test of text to speech.');
  const sessionRef = useRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);

  useEffect(() => {
    createSession(MODEL_PATH).then((s) => {
      sessionRef.current = s;
      setStatus('${config.modelName} \\u00b7 Ready');
    }).catch((e) => { setStatus('Failed to load model'); console.error('Model load error:', e); });
  }, []);

  const handleSynthesize = useCallback(async () => {
    const trimmed = text.trim();
    if (!trimmed || !sessionRef.current) return;

    setStatus('${config.modelName} \\u00b7 Synthesizing...');
    const start = performance.now();

    const tokens = new Float32Array(trimmed.length);
    for (let i = 0; i < trimmed.length; i++) { tokens[i] = trimmed.charCodeAt(i); }

    const output = await runInference(sessionRef.current, tokens);
    const samples = postprocessAudio(output);
    await playAudio(samples);

    const elapsed = (performance.now() - start).toFixed(1);
    setStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(sessionRef.current)}\`);
  }, [text]);

  return (
    <>
      <main>
        <h1>${config.modelName} — ${taskLabel}</h1>
        <div className="tts-input">
          <label htmlFor="textInput">Enter text to synthesize</label>
          <textarea id="textInput" rows={4} value={text} onChange={(e) => setText(e.target.value)} aria-label="Text to synthesize" />
          <button className="primary-btn" onClick={handleSynthesize} aria-label="Synthesize speech">Synthesize</button>
        </div>
      </main>
      <aside className="status-bar"><span>{status}</span></aside>
      <div className="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
    </>
  );
}
`;
}

// ---- Page dispatcher ----

function emitPage(config: ResolvedConfig): string {
  // Audio tasks
  if (config.task === 'text-to-speech') return emitTextToSpeechPage(config);
  if (config.task === 'audio-classification') {
    if (config.input === 'mic') return emitMicAudioClassificationPage(config);
    return emitFileAudioClassificationPage(config);
  }
  if (config.task === 'speech-to-text') {
    if (config.input === 'mic') return emitMicSpeechToTextPage(config);
    return emitFileSpeechToTextPage(config);
  }

  // Visual tasks
  if (config.input === 'camera' || config.input === 'screen') return emitRealtimePage(config);
  if (config.task === 'object-detection') return emitFileDetectionPage(config);
  if (config.task === 'image-segmentation') return emitFileSegmentationPage(config);
  if (config.task === 'feature-extraction') return emitFileFeatureExtractionPage(config);
  return emitFileClassificationPage(config);
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
    config.task === 'feature-extraction';
}

function emitGlobalsCss(config: ResolvedConfig): string {
  const extra = needsExtendedCSS(config) ? emitExtendedCSS() : '';
  return `${emitDesignSystemCSS(config)}\n\n${emitAppCSS()}${extra}`;
}

/** Wrap a CodeBlock's code with exports for use as a lib module */
function toLibModule(block: CodeBlock | undefined): string {
  if (!block) return '';
  return addExports(block.code, block.exports);
}

/**
 * Emit Next.js framework files.
 */
export function emitNextjs(config: ResolvedConfig, blocks: CodeBlock[]): GeneratedFile[] {
  const le = libExt(config);
  const e = ext(config);

  const inputBlock = findBlock(blocks, 'input');
  const preprocessBlock = findBlock(blocks, 'preprocess');
  const inferenceBlock = findBlock(blocks, 'inference');
  const postprocessBlock = findBlock(blocks, 'postprocess');
  const opfsBlock = findBlock(blocks, 'opfs-cache');

  const filePaths: string[] = [
    'package.json',
    'next.config.mjs',
  ];

  if (config.lang === 'ts') {
    filePaths.push('tsconfig.json');
  }

  filePaths.push(
    `app/layout.${e}`,
    `app/page.${e}`,
    'app/globals.css',
  );

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
    { path: 'next.config.mjs', content: emitNextConfig() },
  ];

  if (config.lang === 'ts') {
    files.push({ path: 'tsconfig.json', content: emitTsConfig() });
  }

  files.push(
    { path: `app/layout.${e}`, content: emitLayout(config) },
    { path: `app/page.${e}`, content: emitPage(config) },
    { path: 'app/globals.css', content: emitGlobalsCss(config) },
  );

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
