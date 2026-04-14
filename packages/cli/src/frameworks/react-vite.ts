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

const MODEL_PATH = '/${config.modelName}.onnx';

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

  const preprocessBlock = findBlock(blocks, 'preprocess');
  const inferenceBlock = findBlock(blocks, 'inference');
  const postprocessBlock = findBlock(blocks, 'postprocess');

  const filePaths = [
    'package.json',
    'vite.config.js',
    'index.html',
    `src/main.${e}`,
    `src/App.${e}`,
    'src/App.css',
    `src/lib/preprocess.${le}`,
    `src/lib/inference.${le}`,
    `src/lib/postprocess.${le}`,
    'README.md',
  ];

  const files: GeneratedFile[] = [
    { path: 'package.json', content: emitPackageJson(config, blocks) },
    { path: 'vite.config.js', content: emitViteConfig() },
    { path: 'index.html', content: emitIndexHtml(config) },
    { path: `src/main.${e}`, content: emitMain(config) },
    { path: `src/App.${e}`, content: emitApp(config) },
    { path: 'src/App.css', content: emitAppCssFile(config) },
    { path: `src/lib/preprocess.${le}`, content: toLibModule(preprocessBlock) },
    { path: `src/lib/inference.${le}`, content: toLibModule(inferenceBlock) },
    { path: `src/lib/postprocess.${le}`, content: toLibModule(postprocessBlock) },
    { path: 'README.md', content: emitReadme(config, filePaths) },
  ];

  return files;
}
