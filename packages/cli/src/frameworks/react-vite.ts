/**
 * React-Vite framework emitter (Layer 2).
 *
 * Produces a full React + Vite project:
 *   package.json, vite.config.js, index.html, src/main, src/App, src/App.css,
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
  emitReadme,
  getTaskLabel,
  getEngineLabel,
  getModelPath,
} from './shared.js';

const ext = (config: ResolvedConfig) => (config.lang === 'ts' ? 'tsx' : 'jsx');
const libExt = (config: ResolvedConfig) => (config.lang === 'ts' ? 'ts' : 'js');

function emitPackageJson(config: ResolvedConfig, blocks: CodeBlock[]): string {
  const deps: Record<string, string> = {
    react: '^19.0.0',
    'react-dom': '^19.0.0',
  };
  for (const imp of collectImports(blocks)) {
    if (imp === 'onnxruntime-web') deps[imp] = '^1.21.0';
    else deps[imp] = 'latest';
  }

  const devDeps: Record<string, string> = {
    '@vitejs/plugin-react': '^4.3.0',
    vite: '^6.0.0',
  };
  if (config.lang === 'ts') {
    devDeps['typescript'] = '^5.7.0';
    devDeps['@types/react'] = '^19.0.0';
    devDeps['@types/react-dom'] = '^19.0.0';
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
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
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
    <div id="root"></div>
    <script type="module" src="/src/main.${ext(config)}"></script>
  </body>
</html>
`;
}

function emitMain(config: ResolvedConfig): string {
  const e = ext(config);
  const appImport = e === 'tsx' ? './App.tsx' : './App.jsx';
  return `import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from '${appImport}';
import './App.css';

createRoot(document.getElementById('root')${config.lang === 'ts' ? '!' : ''}).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
`;
}

function emitApp(config: ResolvedConfig): string {
  if (config.input === 'camera' || config.input === 'screen') {
    return emitRealtimeApp(config);
  }
  if (config.task === 'object-detection') return emitFileDetectionApp(config);
  if (config.task === 'image-segmentation') return emitFileSegmentationApp(config);
  if (config.task === 'feature-extraction') return emitFileFeatureExtractionApp(config);
  return emitFileClassificationApp(config);
}

function emitFileClassificationApp(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);

  const stateType = t ? '<{ indices: number[]; values: number[] } | null>' : '';
  const refType = t ? '<HTMLInputElement | null>' : '';
  const eventType = t ? ': React.DragEvent' : '';
  const changeType = t ? ': React.ChangeEvent<HTMLInputElement>' : '';

  return `import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { preprocessImage } from './lib/preprocess.${le}';
import { postprocessResults } from './lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';

export default function App() {
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
    }).catch((e) => {
      setStatus('Failed to load model');
      console.error('Model load error:', e);
    });
  }, []);

  const processImage = useCallback(async (file${t ? ': File' : ''}) => {
    if (!file.type.startsWith('image/')) {
      setStatus('Unsupported file type. Try JPG, PNG, or WebP.');
      return;
    }

    const url = URL.createObjectURL(file);
    setImageUrl(url);
    setResults(null);

    const img = new Image();
    img.src = url;
    await new Promise((resolve) => { img.onload = resolve; });

    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d')${t ? '!' : ''};
    ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    if (!sessionRef.current) {
      setStatus('Model not loaded yet. Please wait.');
      return;
    }

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

  const handleDrop = useCallback((e${eventType}) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) processImage(file);
  }, [processImage]);

  const handleFileChange = useCallback((e${changeType}) => {
    const file = e.target.files?.[0];
    if (file) processImage(file);
  }, [processImage]);

  const reset = useCallback(() => {
    setImageUrl(null);
    setResults(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  }, []);

  return (
    <>
      <a href="#results" className="skip-link">Skip to results</a>
      <main>
        <h1>${config.modelName} — ${taskLabel}</h1>
        <div className="container">
          <div>
            {!imageUrl ? (
              <div
                className={\`drop-zone\${dragOver ? ' drag-over' : ''}\`}
                role="button"
                tabIndex={0}
                aria-label="Drop an image here or click to browse for ${taskLabel.toLowerCase()}"
                onClick={() => fileInputRef.current?.click()}
                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInputRef.current?.click(); } }}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
              >
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
      <aside className="status-bar">
        <span>{status}</span>
      </aside>
      <div className="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
    </>
  );
}
`;
}

function emitFileDetectionApp(config: ResolvedConfig): string {
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

  return `import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { preprocessImage } from './lib/preprocess.${le}';
import { postprocessDetections } from './lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
const NUM_ATTRIBUTES = ${numAttributes};
const NUM_ANCHORS = ${numAnchors};
const COLORS = [[56,189,248],[249,115,22],[34,197,94],[168,85,247],[251,191,36],[239,68,68],[20,184,166],[236,72,153],[99,102,241],[163,230,53]];

${boxType}export default function App() {
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
    canvas.width = imgSize.w;
    canvas.height = imgSize.h;
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

function emitFileSegmentationApp(config: ResolvedConfig): string {
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

  return `import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { preprocessImage } from './lib/preprocess.${le}';
import { postprocessSegmentation } from './lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
const NUM_CLASSES = ${numClasses};
const MASK_H = ${maskH};
const MASK_W = ${maskW};
const COLORS = [[56,189,248],[249,115,22],[34,197,94],[168,85,247],[251,191,36],[239,68,68],[20,184,166],[236,72,153],[99,102,241],[163,230,53]];

export default function App() {
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
            {mask && <p>{new Set(Array.from(mask)).size} classes detected</p>}
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

function emitFileFeatureExtractionApp(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);

  const stateType = t ? '<Float32Array | null>' : '';
  const refType = t ? '<HTMLInputElement | null>' : '';
  const eventType = t ? ': React.DragEvent' : '';
  const changeType = t ? ': React.ChangeEvent<HTMLInputElement>' : '';

  return `import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { preprocessImage } from './lib/preprocess.${le}';
import { postprocessEmbeddings } from './lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';

export default function App() {
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
    const elapsed = (performance.now() - start).toFixed(1);
    setStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(sessionRef.current)}\`);
    setEmbedding(emb);
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
          <div id="results" className="results" role="status" aria-live="polite" aria-atomic="true">
            {embedding && (
              <div className="embedding-info">
                <p><strong>Dimensions:</strong> {embedding.length}</p>
                <p><strong>L2 Norm:</strong> {Math.sqrt(Array.from(embedding).reduce((s, v) => s + v * v, 0)).toFixed(4)}</p>
                <p><strong>First 5:</strong> [{Array.from(embedding.slice(0, 5)).map(v => v.toFixed(4)).join(', ')}, ...]</p>
              </div>
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

function emitRealtimeApp(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const isScreen = config.input === 'screen';
  const startFn = isScreen ? 'startScreenCapture' : 'startCamera';
  const btnLabel = isScreen ? 'Start Screen Capture' : 'Enable Camera';
  const actionLabel = isScreen ? 'capture your screen' : 'use your camera';

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
        const scale = videoRef.current${t ? '!' : ''}.videoWidth / ${config.preprocess.imageSize};
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

  return `import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from './lib/inference.${le}';
import { preprocessImage } from './lib/preprocess.${le}';
${postImport}
import { ${startFn}, captureFrame, stopStream, createInferenceLoop } from './lib/input.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
${extraConsts}

export default function App() {
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
    return () => { if (loopRef.current) loopRef.current.stop(); };
  }, []);

  const start = useCallback(async () => {
    if (!videoRef.current || !overlayRef.current || !sessionRef.current) return;
    try {
      await ${startFn}(videoRef.current);
      overlayRef.current.width = videoRef.current.videoWidth;
      overlayRef.current.height = videoRef.current.videoHeight;
      setStarted(true);
      loopRef.current = createInferenceLoop({
        video: videoRef.current,
        canvas: overlayRef.current,
        async onFrame(imageData${t ? ': ImageData' : ''}) {
          const t0 = performance.now();
          const inputTensor = preprocessImage(imageData.data, imageData.width, imageData.height);
          const output = await runInference(sessionRef.current${t ? '!' : ''}, inputTensor);
          ${processCode}
          const elapsed = performance.now() - t0;
          const ctx = overlayRef.current${t ? '!' : ''}.getContext('2d')${t ? '!' : ''};
          ${renderCode}
          return { result: null, elapsed };
        },
        onStatus(elapsed${t ? ': number' : ''}) {
          setStatus(\`${config.modelName} \\u00b7 \${elapsed.toFixed(1)}ms \\u00b7 \${getBackendLabel(sessionRef.current${t ? '!' : ''})}\`);
        },
      });
      loopRef.current.start();
    } catch (e) {
      setStatus('${isScreen ? 'Screen capture' : 'Camera'} access denied');
      console.error(e);
    }
  }, []);

  return (
    <>
      <a href="#results" className="skip-link">Skip to results</a>
      <main>
        <h1>${config.modelName} — ${taskLabel}</h1>
        {!started && (
          <div className="permission-prompt">
            <p>This app ${actionLabel} to run ${taskLabel.toLowerCase()} in real time.</p>
            <p className="hint">No video is recorded or sent anywhere.</p>
            <button className="primary-btn" onClick={start}>${btnLabel}</button>
          </div>
        )}
        {started && (
          <div className="video-wrapper">
            <video ref={videoRef} autoPlay playsInline muted />
            <canvas ref={overlayRef} className="overlay-canvas" />
          </div>
        )}
        <video ref={!started ? videoRef : undefined} hidden />
      </main>
      <aside className="status-bar"><span>{status}</span></aside>
      <div className="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
    </>
  );
}
`;
}

function emitAppCssFile(config: ResolvedConfig): string {
  return `${emitDesignSystemCSS(config)}\n\n${emitAppCSS()}`;
}

/** Wrap a CodeBlock's code with exports for use as a lib module */
function toLibModule(block: CodeBlock | undefined): string {
  if (!block) return '';
  return addExports(block.code, block.exports);
}

/**
 * Emit React-Vite framework files.
 */
export function emitReactVite(config: ResolvedConfig, blocks: CodeBlock[]): GeneratedFile[] {
  const le = libExt(config);
  const e = ext(config);

  const inputBlock = findBlock(blocks, 'input');
  const preprocessBlock = findBlock(blocks, 'preprocess');
  const inferenceBlock = findBlock(blocks, 'inference');
  const postprocessBlock = findBlock(blocks, 'postprocess');
  const opfsBlock = findBlock(blocks, 'opfs-cache');

  const filePaths: string[] = [
    'package.json',
    'vite.config.js',
    'index.html',
    `src/main.${e}`,
    `src/App.${e}`,
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
    { path: `src/main.${e}`, content: emitMain(config) },
    { path: `src/App.${e}`, content: emitApp(config) },
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

  return files;
}
