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

// ---- Fill-Mask ----

function emitFillMaskPage(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');

  return `'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from '../lib/inference.${le}';
import { loadTokenizer, tokenizeText } from '../lib/preprocess.${le}';
import { postprocessFillMask } from '../lib/postprocess.${le}';

const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

export default function Page() {
  const [predictions, setPredictions] = useState${t ? '<Array<{ token: string; score: number }> | null>' : ''}(null);
  const [status, setStatus] = useState('Loading model...');
  const [text, setText] = useState('The capital of France is [MASK].');
  const sessionRef = useRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
  const tokenizerRef = useRef${t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : ''}(null);

  useEffect(() => {
    Promise.all([createSession(MODEL_PATH), loadTokenizer(TOKENIZER_PATH)])
      .then(([s, tok]) => { sessionRef.current = s; tokenizerRef.current = tok; setStatus('${config.modelName} \\u00b7 Ready'); })
      .catch((e) => { setStatus('Failed to load model'); console.error(e); });
  }, []);

  const handlePredict = useCallback(async () => {
    if (!sessionRef.current || !tokenizerRef.current || !text.trim()) return;
    setStatus('${config.modelName} \\u00b7 Processing...');
    const start = performance.now();
    const { inputIds } = tokenizeText(tokenizerRef.current, text);
    const output = await runInference(sessionRef.current, inputIds);
    const results = postprocessFillMask(output, inputIds, tokenizerRef.current);
    const elapsed = (performance.now() - start).toFixed(1);
    setStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(sessionRef.current)}\`);
    setPredictions(results);
  }, [text]);

  return (
    <>
      <main>
        <h1>${config.modelName} — ${taskLabel}</h1>
        <div className="text-input-section">
          <label htmlFor="textInput">Enter text with [MASK] token</label>
          <textarea id="textInput" rows={4} value={text} onChange={(e) => setText(e.target.value)} />
          <button className="primary-btn" onClick={handlePredict}>Predict</button>
        </div>
        <div className="mask-predictions" role="status" aria-live="polite">
          {predictions && predictions.map((p${t ? ': { token: string; score: number }' : ''}, i${t ? ': number' : ''}) => (
            <div key={i} className="mask-prediction">
              <span className="token">{p.token}</span>
              <span className="prob">{(p.score * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </main>
      <aside className="status-bar"><span>{status}</span></aside>
      <div className="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
    </>
  );
}
`;
}

// ---- Sentence Similarity ----

function emitSentenceSimilarityPage(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');

  return `'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from '../lib/inference.${le}';
import { loadTokenizer, tokenizeText } from '../lib/preprocess.${le}';
import { cosineSimilarity } from '../lib/postprocess.${le}';

const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

export default function Page() {
  const [scores, setScores] = useState${t ? '<Array<{ sentence: string; score: number }> | null>' : ''}(null);
  const [status, setStatus] = useState('Loading model...');
  const [source, setSource] = useState('The weather is lovely today.');
  const [comparisons, setComparisons] = useState('It is a beautiful day.\\nThe sun is shining bright.\\nI need to buy groceries.');
  const sessionRef = useRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
  const tokenizerRef = useRef${t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : ''}(null);

  useEffect(() => {
    Promise.all([createSession(MODEL_PATH), loadTokenizer(TOKENIZER_PATH)])
      .then(([s, tok]) => { sessionRef.current = s; tokenizerRef.current = tok; setStatus('${config.modelName} \\u00b7 Ready'); })
      .catch((e) => { setStatus('Failed to load model'); console.error(e); });
  }, []);

  const handleCompare = useCallback(async () => {
    if (!sessionRef.current || !tokenizerRef.current || !source.trim()) return;
    setStatus('${config.modelName} \\u00b7 Computing...');
    const start = performance.now();
    const { inputIds: srcIds } = tokenizeText(tokenizerRef.current, source);
    const srcEmb = await runInference(sessionRef.current, srcIds);
    const lines = comparisons.split('\\n').filter(Boolean);
    const results${t ? ': Array<{ sentence: string; score: number }>' : ''} = [];
    for (const line of lines) {
      const { inputIds: cmpIds } = tokenizeText(tokenizerRef.current, line);
      const cmpEmb = await runInference(sessionRef.current, cmpIds);
      results.push({ sentence: line, score: cosineSimilarity(srcEmb, cmpEmb) });
    }
    const elapsed = (performance.now() - start).toFixed(1);
    setStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(sessionRef.current)}\`);
    setScores(results);
  }, [source, comparisons]);

  return (
    <>
      <main>
        <h1>${config.modelName} — ${taskLabel}</h1>
        <div className="text-input-section">
          <label htmlFor="source">Source sentence</label>
          <textarea id="source" rows={2} value={source} onChange={(e) => setSource(e.target.value)} />
          <label htmlFor="compare">Sentences to compare (one per line)</label>
          <textarea id="compare" rows={4} value={comparisons} onChange={(e) => setComparisons(e.target.value)} />
          <button className="primary-btn" onClick={handleCompare}>Compare</button>
        </div>
        <div className="similarity-pairs" role="status" aria-live="polite">
          {scores && scores.map((s${t ? ': { sentence: string; score: number }' : ''}, i${t ? ': number' : ''}) => (
            <div key={i} className="similarity-score">
              <span>{s.sentence}</span>
              <span className="value">{s.score.toFixed(4)}</span>
            </div>
          ))}
        </div>
      </main>
      <aside className="status-bar"><span>{status}</span></aside>
      <div className="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
    </>
  );
}
`;
}

// ---- Depth Estimation ----

function emitDepthEstimationPage(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');

  return `'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from '../lib/inference.${le}';
import { preprocessImage } from '../lib/preprocess.${le}';
import { postprocessDepth } from '../lib/postprocess.${le}';

const MODEL_PATH = '${modelPath}';

export default function Page() {
  const [status, setStatus] = useState('Loading model...');
  const [imageUrl, setImageUrl] = useState${t ? '<string | null>' : ''}(null);
  const [dragOver, setDragOver] = useState(false);
  const sessionRef = useRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
  const canvasRef = useRef${t ? '<HTMLCanvasElement | null>' : ''}(null);
  const fileInputRef = useRef${t ? '<HTMLInputElement | null>' : ''}(null);

  useEffect(() => {
    createSession(MODEL_PATH).then((s) => { sessionRef.current = s; setStatus('${config.modelName} \\u00b7 Ready'); })
      .catch((e) => { setStatus('Failed to load model'); console.error(e); });
  }, []);

  const processImage = useCallback(async (file${t ? ': File' : ''}) => {
    if (!file.type.startsWith('image/') || !sessionRef.current) return;
    const url = URL.createObjectURL(file);
    setImageUrl(url);
    const img = new Image();
    img.src = url;
    await new Promise((r) => { img.onload = r; });
    setStatus('${config.modelName} \\u00b7 Processing...');
    const start = performance.now();
    const input = preprocessImage(img);
    const output = await runInference(sessionRef.current, input);
    const depthMap = postprocessDepth(output);
    const canvas = canvasRef.current;
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
    setStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(sessionRef.current)}\`);
  }, []);

  return (
    <>
      <main>
        <h1>${config.modelName} — ${taskLabel}</h1>
        <div className="container">
          <div>
            {!imageUrl ? (
              <div className={\`drop-zone\${dragOver ? ' drag-over' : ''}\`} onClick={() => fileInputRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }} onDragLeave={() => setDragOver(false)}
                onDrop={(e) => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files[0]; if (f) processImage(f); }}>
                <p>Drop an image here or click to browse</p>
                <p className="hint">Supports JPG, PNG, WebP</p>
                <input ref={fileInputRef} type="file" accept="image/*" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) processImage(f); }} />
              </div>
            ) : (
              <div className="preview">
                <img src={imageUrl} alt="Source" />
                <button className="change-btn" onClick={() => setImageUrl(null)}>Choose another image</button>
              </div>
            )}
          </div>
          <div><canvas ref={canvasRef} className="depth-canvas" /></div>
        </div>
      </main>
      <aside className="status-bar"><span>{status}</span></aside>
      <div className="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
    </>
  );
}
`;
}

// ---- Token Classification (NER) ----

function emitTokenClassificationPage(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');

  return `'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from '../lib/inference.${le}';
import { loadTokenizer, tokenizeText } from '../lib/preprocess.${le}';
import { postprocessTokenClassification } from '../lib/postprocess.${le}';

const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

export default function Page() {
  const [html, setHtml] = useState('');
  const [status, setStatus] = useState('Loading model...');
  const [text, setText] = useState('John Smith works at Google in Mountain View, California.');
  const sessionRef = useRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
  const tokenizerRef = useRef${t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : ''}(null);

  useEffect(() => {
    Promise.all([createSession(MODEL_PATH), loadTokenizer(TOKENIZER_PATH)])
      .then(([s, tok]) => { sessionRef.current = s; tokenizerRef.current = tok; setStatus('${config.modelName} \\u00b7 Ready'); })
      .catch((e) => { setStatus('Failed to load model'); console.error(e); });
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!sessionRef.current || !tokenizerRef.current || !text.trim()) return;
    setStatus('${config.modelName} \\u00b7 Processing...');
    const start = performance.now();
    const { inputIds } = tokenizeText(tokenizerRef.current, text);
    const output = await runInference(sessionRef.current, inputIds);
    const entities = postprocessTokenClassification(output, inputIds, tokenizerRef.current);
    let result = text;
    for (let i = entities.length - 1; i >= 0; i--) {
      const e = entities[i];
      result = result.slice(0, e.start) + '<span class="ner-entity" data-type="' + e.type + '">' + result.slice(e.start, e.end) + '</span>' + result.slice(e.end);
    }
    setHtml(result);
    const elapsed = (performance.now() - start).toFixed(1);
    setStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(sessionRef.current)}\`);
  }, [text]);

  return (
    <>
      <main>
        <h1>${config.modelName} — ${taskLabel}</h1>
        <div className="text-input-section">
          <label htmlFor="textInput">Enter text to analyze</label>
          <textarea id="textInput" rows={4} value={text} onChange={(e) => setText(e.target.value)} />
          <button className="primary-btn" onClick={handleAnalyze}>Analyze</button>
        </div>
        <div className="ner-output" role="status" aria-live="polite" dangerouslySetInnerHTML={{ __html: html || text }} />
      </main>
      <aside className="status-bar"><span>{status}</span></aside>
      <div className="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
    </>
  );
}
`;
}

// ---- Question Answering ----

function emitQuestionAnsweringPage(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');

  return `'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from '../lib/inference.${le}';
import { loadTokenizer, tokenizeText } from '../lib/preprocess.${le}';
import { postprocessQA } from '../lib/postprocess.${le}';

const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

export default function Page() {
  const [answer, setAnswer] = useState${t ? '<{ answer: string; score: number } | null>' : ''}(null);
  const [status, setStatus] = useState('Loading model...');
  const [context, setContext] = useState('The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was constructed from 1887 to 1889.');
  const [question, setQuestion] = useState('When was the Eiffel Tower built?');
  const sessionRef = useRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
  const tokenizerRef = useRef${t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : ''}(null);

  useEffect(() => {
    Promise.all([createSession(MODEL_PATH), loadTokenizer(TOKENIZER_PATH)])
      .then(([s, tok]) => { sessionRef.current = s; tokenizerRef.current = tok; setStatus('${config.modelName} \\u00b7 Ready'); })
      .catch((e) => { setStatus('Failed to load model'); console.error(e); });
  }, []);

  const handleAnswer = useCallback(async () => {
    if (!sessionRef.current || !tokenizerRef.current || !context.trim() || !question.trim()) return;
    setStatus('${config.modelName} \\u00b7 Processing...');
    const start = performance.now();
    const combined = question + ' [SEP] ' + context;
    const { inputIds } = tokenizeText(tokenizerRef.current, combined);
    const output = await runInference(sessionRef.current, inputIds);
    const result = postprocessQA(output, inputIds, tokenizerRef.current);
    const elapsed = (performance.now() - start).toFixed(1);
    setStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(sessionRef.current)}\`);
    setAnswer(result);
  }, [context, question]);

  return (
    <>
      <main>
        <h1>${config.modelName} — ${taskLabel}</h1>
        <div className="text-input-section">
          <label htmlFor="context">Context</label>
          <textarea id="context" rows={4} value={context} onChange={(e) => setContext(e.target.value)} />
          <label htmlFor="question">Question</label>
          <input id="question" type="text" value={question} onChange={(e) => setQuestion(e.target.value)} />
          <button className="primary-btn" onClick={handleAnswer}>Answer</button>
        </div>
        <div className="qa-answer" role="status" aria-live="polite">
          {answer && <><div>{answer.answer}</div><div className="score">Confidence: {(answer.score * 100).toFixed(1)}%</div></>}
        </div>
      </main>
      <aside className="status-bar"><span>{status}</span></aside>
      <div className="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
    </>
  );
}
`;
}

// ---- Summarization ----

function emitSummarizationPage(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');

  return `'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from '../lib/inference.${le}';
import { loadTokenizer, tokenizeText } from '../lib/preprocess.${le}';
import { postprocessSummarization } from '../lib/postprocess.${le}';

const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

export default function Page() {
  const [output, setOutput] = useState('');
  const [status, setStatus] = useState('Loading model...');
  const [text, setText] = useState('Artificial intelligence has transformed many industries. Machine learning models can now process natural language, recognize images, and generate creative content.');
  const [processing, setProcessing] = useState(false);
  const sessionRef = useRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
  const tokenizerRef = useRef${t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : ''}(null);

  useEffect(() => {
    Promise.all([createSession(MODEL_PATH), loadTokenizer(TOKENIZER_PATH)])
      .then(([s, tok]) => { sessionRef.current = s; tokenizerRef.current = tok; setStatus('${config.modelName} \\u00b7 Ready'); })
      .catch((e) => { setStatus('Failed to load model'); console.error(e); });
  }, []);

  const handleSummarize = useCallback(async () => {
    if (!sessionRef.current || !tokenizerRef.current || !text.trim() || processing) return;
    setProcessing(true);
    setStatus('${config.modelName} \\u00b7 Summarizing...');
    const start = performance.now();
    const { inputIds } = tokenizeText(tokenizerRef.current, text);
    const raw = await runInference(sessionRef.current, inputIds);
    const summary = postprocessSummarization(raw, tokenizerRef.current, 128, 1);
    const elapsed = (performance.now() - start).toFixed(1);
    setStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(sessionRef.current)}\`);
    setOutput(summary);
    setProcessing(false);
  }, [text, processing]);

  return (
    <>
      <main>
        <h1>${config.modelName} — ${taskLabel}</h1>
        <div className="text-input-section">
          <label htmlFor="textInput">Enter text to summarize</label>
          <textarea id="textInput" rows={6} value={text} onChange={(e) => setText(e.target.value)} disabled={processing} />
          <button className="primary-btn" onClick={handleSummarize} disabled={processing}>{processing ? 'Summarizing...' : 'Summarize'}</button>
        </div>
        <div className="text-output" role="status" aria-live="polite">
          {output && <pre>{output}</pre>}
        </div>
      </main>
      <aside className="status-bar"><span>{status}</span></aside>
      <div className="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
    </>
  );
}
`;
}

// ---- Translation ----

function emitTranslationPage(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');

  return `'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from '../lib/inference.${le}';
import { loadTokenizer, tokenizeText } from '../lib/preprocess.${le}';
import { postprocessTranslation } from '../lib/postprocess.${le}';

const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

export default function Page() {
  const [output, setOutput] = useState('');
  const [status, setStatus] = useState('Loading model...');
  const [text, setText] = useState('Hello, how are you today?');
  const [processing, setProcessing] = useState(false);
  const sessionRef = useRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
  const tokenizerRef = useRef${t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : ''}(null);

  useEffect(() => {
    Promise.all([createSession(MODEL_PATH), loadTokenizer(TOKENIZER_PATH)])
      .then(([s, tok]) => { sessionRef.current = s; tokenizerRef.current = tok; setStatus('${config.modelName} \\u00b7 Ready'); })
      .catch((e) => { setStatus('Failed to load model'); console.error(e); });
  }, []);

  const handleTranslate = useCallback(async () => {
    if (!sessionRef.current || !tokenizerRef.current || !text.trim() || processing) return;
    setProcessing(true);
    setStatus('${config.modelName} \\u00b7 Translating...');
    const start = performance.now();
    const { inputIds } = tokenizeText(tokenizerRef.current, text);
    const raw = await runInference(sessionRef.current, inputIds);
    const translation = postprocessTranslation(raw, tokenizerRef.current, 128, 1);
    const elapsed = (performance.now() - start).toFixed(1);
    setStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(sessionRef.current)}\`);
    setOutput(translation);
    setProcessing(false);
  }, [text, processing]);

  return (
    <>
      <main>
        <h1>${config.modelName} — ${taskLabel}</h1>
        <div className="text-input-section">
          <label htmlFor="textInput">Enter text to translate</label>
          <textarea id="textInput" rows={4} value={text} onChange={(e) => setText(e.target.value)} disabled={processing} />
          <button className="primary-btn" onClick={handleTranslate} disabled={processing}>{processing ? 'Translating...' : 'Translate'}</button>
        </div>
        <div className="text-output" role="status" aria-live="polite">
          {output && <pre>{output}</pre>}
        </div>
      </main>
      <aside className="status-bar"><span>{status}</span></aside>
      <div className="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
    </>
  );
}
`;
}

// ---- Text2Text Generation ----

function emitText2TextPage(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');

  return `'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from '../lib/inference.${le}';
import { loadTokenizer, tokenizeText } from '../lib/preprocess.${le}';
import { postprocessText2Text } from '../lib/postprocess.${le}';

const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

export default function Page() {
  const [output, setOutput] = useState('');
  const [status, setStatus] = useState('Loading model...');
  const [text, setText] = useState('Paraphrase: The house is big and beautiful.');
  const [processing, setProcessing] = useState(false);
  const sessionRef = useRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
  const tokenizerRef = useRef${t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : ''}(null);

  useEffect(() => {
    Promise.all([createSession(MODEL_PATH), loadTokenizer(TOKENIZER_PATH)])
      .then(([s, tok]) => { sessionRef.current = s; tokenizerRef.current = tok; setStatus('${config.modelName} \\u00b7 Ready'); })
      .catch((e) => { setStatus('Failed to load model'); console.error(e); });
  }, []);

  const handleRun = useCallback(async () => {
    if (!sessionRef.current || !tokenizerRef.current || !text.trim() || processing) return;
    setProcessing(true);
    setStatus('${config.modelName} \\u00b7 Processing...');
    const start = performance.now();
    const { inputIds } = tokenizeText(tokenizerRef.current, text);
    const raw = await runInference(sessionRef.current, inputIds);
    const result = postprocessText2Text(raw, tokenizerRef.current, 128, 1);
    const elapsed = (performance.now() - start).toFixed(1);
    setStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(sessionRef.current)}\`);
    setOutput(result);
    setProcessing(false);
  }, [text, processing]);

  return (
    <>
      <main>
        <h1>${config.modelName} — ${taskLabel}</h1>
        <div className="text-input-section">
          <label htmlFor="textInput">Enter input text</label>
          <textarea id="textInput" rows={4} value={text} onChange={(e) => setText(e.target.value)} disabled={processing} />
          <button className="primary-btn" onClick={handleRun} disabled={processing}>{processing ? 'Processing...' : 'Run'}</button>
        </div>
        <div className="text-output" role="status" aria-live="polite">
          {output && <pre>{output}</pre>}
        </div>
      </main>
      <aside className="status-bar"><span>{status}</span></aside>
      <div className="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
    </>
  );
}
`;
}

// ---- Conversational ----

function emitConversationalPage(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');

  return `'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from '../lib/inference.${le}';
import { loadTokenizer } from '../lib/preprocess.${le}';
import { postprocessConversational, sampleNextToken } from '../lib/postprocess.${le}';

const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

export default function Page() {
  const [messages, setMessages] = useState${t ? '<Array<{ role: string; text: string }>>' : ''}([]);
  const [input, setInput] = useState('');
  const [status, setStatus] = useState('Loading model...');
  const [generating, setGenerating] = useState(false);
  const sessionRef = useRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
  const tokenizerRef = useRef${t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : ''}(null);
  const historyRef = useRef${t ? '<string[]>' : ''}([]);

  useEffect(() => {
    Promise.all([createSession(MODEL_PATH), loadTokenizer(TOKENIZER_PATH)])
      .then(([s, tok]) => { sessionRef.current = s; tokenizerRef.current = tok; setStatus('${config.modelName} \\u00b7 Ready'); })
      .catch((e) => { setStatus('Failed to load model'); console.error(e); });
  }, []);

  const handleSend = useCallback(async () => {
    if (!sessionRef.current || !tokenizerRef.current || !input.trim() || generating) return;
    const userMsg = input.trim();
    setInput('');
    setMessages((prev) => [...prev, { role: 'user', text: userMsg }]);
    historyRef.current.push(userMsg);
    setGenerating(true);
    setStatus('${config.modelName} \\u00b7 Generating...');
    const start = performance.now();
    const prompt = historyRef.current.join(' ');
    const encoded = tokenizerRef.current.encode(prompt);
    let inputIds = encoded.inputIds;
    for (let i = 0; i < 50; i++) {
      const inputBigInt = new BigInt64Array(inputIds.map((id${t ? ': number' : ''}) => BigInt(id)));
      const output = await runInference(sessionRef.current, inputBigInt);
      const vocabSize = tokenizerRef.current.getVocabSize();
      const logits = postprocessConversational(output, inputIds.length, vocabSize);
      const nextToken = sampleNextToken(logits);
      if (nextToken === 2) break;
      inputIds = [...inputIds, nextToken];
    }
    const decoded = tokenizerRef.current.decode(inputIds);
    const reply = decoded.slice(prompt.length).trim() || '(no response)';
    historyRef.current.push(reply);
    setMessages((prev) => [...prev, { role: 'bot', text: reply }]);
    const elapsed = (performance.now() - start).toFixed(1);
    setStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(sessionRef.current)}\`);
    setGenerating(false);
  }, [input, generating]);

  return (
    <>
      <main>
        <h1>${config.modelName} — ${taskLabel}</h1>
        <div className="chat-messages">
          {messages.map((m${t ? ': { role: string; text: string }' : ''}, i${t ? ': number' : ''}) => (
            <div key={i} className={\`chat-msg \${m.role}\`}>{m.text}</div>
          ))}
        </div>
        <div className="chat-input-row">
          <input type="text" value={input} onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
            placeholder="Type a message..." disabled={generating} />
          <button className="primary-btn" onClick={handleSend} disabled={generating}>Send</button>
        </div>
      </main>
      <aside className="status-bar"><span>{status}</span></aside>
      <div className="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
    </>
  );
}
`;
}

// ---- Table Question Answering ----

function emitTableQAPage(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');

  return `'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from '../lib/inference.${le}';
import { loadTokenizer, tokenizeText } from '../lib/preprocess.${le}';
import { postprocessTableQA } from '../lib/postprocess.${le}';

const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

export default function Page() {
  const [answer, setAnswer] = useState${t ? '<{ answer: string; score: number } | null>' : ''}(null);
  const [status, setStatus] = useState('Loading model...');
  const [table, setTable] = useState('Name, Age, City\\nAlice, 30, New York\\nBob, 25, San Francisco');
  const [question, setQuestion] = useState('Who lives in San Francisco?');
  const sessionRef = useRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
  const tokenizerRef = useRef${t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : ''}(null);

  useEffect(() => {
    Promise.all([createSession(MODEL_PATH), loadTokenizer(TOKENIZER_PATH)])
      .then(([s, tok]) => { sessionRef.current = s; tokenizerRef.current = tok; setStatus('${config.modelName} \\u00b7 Ready'); })
      .catch((e) => { setStatus('Failed to load model'); console.error(e); });
  }, []);

  const handleAnswer = useCallback(async () => {
    if (!sessionRef.current || !tokenizerRef.current || !table.trim() || !question.trim()) return;
    setStatus('${config.modelName} \\u00b7 Processing...');
    const start = performance.now();
    const combined = table + ' [SEP] ' + question;
    const { inputIds } = tokenizeText(tokenizerRef.current, combined);
    const output = await runInference(sessionRef.current, inputIds);
    const result = postprocessTableQA(output, inputIds, tokenizerRef.current);
    const elapsed = (performance.now() - start).toFixed(1);
    setStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(sessionRef.current)}\`);
    setAnswer(result);
  }, [table, question]);

  return (
    <>
      <main>
        <h1>${config.modelName} — ${taskLabel}</h1>
        <div className="text-input-section">
          <label htmlFor="table">Table data (CSV format)</label>
          <textarea id="table" rows={4} value={table} onChange={(e) => setTable(e.target.value)} />
          <label htmlFor="question">Question</label>
          <input id="question" type="text" value={question} onChange={(e) => setQuestion(e.target.value)} />
          <button className="primary-btn" onClick={handleAnswer}>Answer</button>
        </div>
        <div className="qa-answer" role="status" aria-live="polite">
          {answer && <><div>{answer.answer}</div><div className="score">Confidence: {(answer.score * 100).toFixed(1)}%</div></>}
        </div>
      </main>
      <aside className="status-bar"><span>{status}</span></aside>
      <div className="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
    </>
  );
}
`;
}

// ---- Image-to-Text ----

function emitImageToTextPage(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');

  return `'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from '../lib/inference.${le}';
import { preprocessImage } from '../lib/preprocess.${le}';
import { postprocessImageToText } from '../lib/postprocess.${le}';

const MODEL_PATH = '${modelPath}';

export default function Page() {
  const [caption, setCaption] = useState('');
  const [status, setStatus] = useState('Loading model...');
  const [imageUrl, setImageUrl] = useState${t ? '<string | null>' : ''}(null);
  const [dragOver, setDragOver] = useState(false);
  const sessionRef = useRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
  const fileInputRef = useRef${t ? '<HTMLInputElement | null>' : ''}(null);

  useEffect(() => {
    createSession(MODEL_PATH).then((s) => { sessionRef.current = s; setStatus('${config.modelName} \\u00b7 Ready'); })
      .catch((e) => { setStatus('Failed to load model'); console.error(e); });
  }, []);

  const processImage = useCallback(async (file${t ? ': File' : ''}) => {
    if (!file.type.startsWith('image/') || !sessionRef.current) return;
    const url = URL.createObjectURL(file);
    setImageUrl(url);
    const img = new Image();
    img.src = url;
    await new Promise((r) => { img.onload = r; });
    setStatus('${config.modelName} \\u00b7 Generating caption...');
    const start = performance.now();
    const input = preprocessImage(img);
    const output = await runInference(sessionRef.current, input);
    const text = postprocessImageToText(output);
    const elapsed = (performance.now() - start).toFixed(1);
    setStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(sessionRef.current)}\`);
    setCaption(text);
  }, []);

  return (
    <>
      <main>
        <h1>${config.modelName} — ${taskLabel}</h1>
        <div className="container">
          <div>
            {!imageUrl ? (
              <div className={\`drop-zone\${dragOver ? ' drag-over' : ''}\`} onClick={() => fileInputRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }} onDragLeave={() => setDragOver(false)}
                onDrop={(e) => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files[0]; if (f) processImage(f); }}>
                <p>Drop an image here or click to browse</p><p className="hint">Supports JPG, PNG, WebP</p>
                <input ref={fileInputRef} type="file" accept="image/*" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) processImage(f); }} />
              </div>
            ) : (
              <div className="preview">
                <img src={imageUrl} alt="Source" />
                <button className="change-btn" onClick={() => { setImageUrl(null); setCaption(''); }}>Choose another image</button>
              </div>
            )}
          </div>
          <div className="text-output" role="status" aria-live="polite">
            {caption && <pre>{caption}</pre>}
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

// ---- Visual Question Answering ----

function emitVQAPage(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');

  return `'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from '../lib/inference.${le}';
import { preprocessImage } from '../lib/preprocess.${le}';
import { postprocessVQA } from '../lib/postprocess.${le}';
import { loadTokenizer } from '../lib/preprocess.${le}';

const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

export default function Page() {
  const [answer, setAnswer] = useState('');
  const [status, setStatus] = useState('Loading model...');
  const [imageUrl, setImageUrl] = useState${t ? '<string | null>' : ''}(null);
  const [question, setQuestion] = useState('What is in this image?');
  const [dragOver, setDragOver] = useState(false);
  const sessionRef = useRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
  const tokenizerRef = useRef${t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : ''}(null);
  const imgRef = useRef${t ? '<HTMLImageElement | null>' : ''}(null);
  const fileInputRef = useRef${t ? '<HTMLInputElement | null>' : ''}(null);

  useEffect(() => {
    Promise.all([createSession(MODEL_PATH), loadTokenizer(TOKENIZER_PATH)])
      .then(([s, tok]) => { sessionRef.current = s; tokenizerRef.current = tok; setStatus('${config.modelName} \\u00b7 Ready'); })
      .catch((e) => { setStatus('Failed to load model'); console.error(e); });
  }, []);

  const handleFile = useCallback((file${t ? ': File' : ''}) => {
    if (!file.type.startsWith('image/')) return;
    const url = URL.createObjectURL(file);
    setImageUrl(url);
    const img = new Image();
    img.src = url;
    img.onload = () => { imgRef.current = img; };
  }, []);

  const handleAsk = useCallback(async () => {
    if (!sessionRef.current || !tokenizerRef.current || !imgRef.current || !question.trim()) return;
    setStatus('${config.modelName} \\u00b7 Processing...');
    const start = performance.now();
    const input = preprocessImage(imgRef.current);
    const output = await runInference(sessionRef.current, input);
    const result = postprocessVQA(output, tokenizerRef.current);
    const elapsed = (performance.now() - start).toFixed(1);
    setStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(sessionRef.current)}\`);
    setAnswer(result);
  }, [question]);

  return (
    <>
      <main>
        <h1>${config.modelName} — ${taskLabel}</h1>
        <div className="container">
          <div className="multimodal-input">
            {!imageUrl ? (
              <div className={\`drop-zone\${dragOver ? ' drag-over' : ''}\`} onClick={() => fileInputRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }} onDragLeave={() => setDragOver(false)}
                onDrop={(e) => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files[0]; if (f) handleFile(f); }}>
                <p>Drop an image here or click to browse</p><p className="hint">Supports JPG, PNG, WebP</p>
                <input ref={fileInputRef} type="file" accept="image/*" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }} />
              </div>
            ) : (
              <div className="preview">
                <img src={imageUrl} alt="Selected" />
                <button className="change-btn" onClick={() => { setImageUrl(null); setAnswer(''); imgRef.current = null; }}>Choose another image</button>
              </div>
            )}
            <input type="text" className="question-input" value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Ask a question..." />
            <button className="primary-btn" onClick={handleAsk}>Ask</button>
          </div>
          <div className="qa-answer" role="status" aria-live="polite">{answer}</div>
        </div>
      </main>
      <aside className="status-bar"><span>{status}</span></aside>
      <div className="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
    </>
  );
}
`;
}

// ---- Document Question Answering ----

function emitDocQAPage(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');

  return `'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from '../lib/inference.${le}';
import { preprocessImage } from '../lib/preprocess.${le}';
import { postprocessDocQA } from '../lib/postprocess.${le}';
import { loadTokenizer } from '../lib/preprocess.${le}';

const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

export default function Page() {
  const [answer, setAnswer] = useState('');
  const [status, setStatus] = useState('Loading model...');
  const [imageUrl, setImageUrl] = useState${t ? '<string | null>' : ''}(null);
  const [question, setQuestion] = useState('What is the total amount?');
  const [dragOver, setDragOver] = useState(false);
  const sessionRef = useRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
  const tokenizerRef = useRef${t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : ''}(null);
  const imgRef = useRef${t ? '<HTMLImageElement | null>' : ''}(null);
  const fileInputRef = useRef${t ? '<HTMLInputElement | null>' : ''}(null);

  useEffect(() => {
    Promise.all([createSession(MODEL_PATH), loadTokenizer(TOKENIZER_PATH)])
      .then(([s, tok]) => { sessionRef.current = s; tokenizerRef.current = tok; setStatus('${config.modelName} \\u00b7 Ready'); })
      .catch((e) => { setStatus('Failed to load model'); console.error(e); });
  }, []);

  const handleFile = useCallback((file${t ? ': File' : ''}) => {
    if (!file.type.startsWith('image/')) return;
    const url = URL.createObjectURL(file);
    setImageUrl(url);
    const img = new Image();
    img.src = url;
    img.onload = () => { imgRef.current = img; };
  }, []);

  const handleAsk = useCallback(async () => {
    if (!sessionRef.current || !tokenizerRef.current || !imgRef.current || !question.trim()) return;
    setStatus('${config.modelName} \\u00b7 Processing...');
    const start = performance.now();
    const input = preprocessImage(imgRef.current);
    const output = await runInference(sessionRef.current, input);
    const result = postprocessDocQA(output, tokenizerRef.current);
    const elapsed = (performance.now() - start).toFixed(1);
    setStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(sessionRef.current)}\`);
    setAnswer(result);
  }, [question]);

  return (
    <>
      <main>
        <h1>${config.modelName} — ${taskLabel}</h1>
        <div className="container">
          <div className="multimodal-input">
            {!imageUrl ? (
              <div className={\`drop-zone\${dragOver ? ' drag-over' : ''}\`} onClick={() => fileInputRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }} onDragLeave={() => setDragOver(false)}
                onDrop={(e) => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files[0]; if (f) handleFile(f); }}>
                <p>Drop a document image here or click to browse</p><p className="hint">Supports JPG, PNG, WebP</p>
                <input ref={fileInputRef} type="file" accept="image/*" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }} />
              </div>
            ) : (
              <div className="preview">
                <img src={imageUrl} alt="Document" />
                <button className="change-btn" onClick={() => { setImageUrl(null); setAnswer(''); imgRef.current = null; }}>Choose another image</button>
              </div>
            )}
            <input type="text" className="question-input" value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Ask about the document..." />
            <button className="primary-btn" onClick={handleAsk}>Ask</button>
          </div>
          <div className="qa-answer" role="status" aria-live="polite">{answer}</div>
        </div>
      </main>
      <aside className="status-bar"><span>{status}</span></aside>
      <div className="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
    </>
  );
}
`;
}

// ---- Image-Text-to-Text ----

function emitImageTextToTextPage(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');

  return `'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from '../lib/inference.${le}';
import { preprocessImage } from '../lib/preprocess.${le}';
import { postprocessImageTextToText } from '../lib/postprocess.${le}';
import { loadTokenizer } from '../lib/preprocess.${le}';

const MODEL_PATH = '${modelPath}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

export default function Page() {
  const [output, setOutput] = useState('');
  const [status, setStatus] = useState('Loading model...');
  const [imageUrl, setImageUrl] = useState${t ? '<string | null>' : ''}(null);
  const [prompt, setPrompt] = useState('Describe this image in detail.');
  const [dragOver, setDragOver] = useState(false);
  const [processing, setProcessing] = useState(false);
  const sessionRef = useRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
  const tokenizerRef = useRef${t ? '<Awaited<ReturnType<typeof loadTokenizer>> | null>' : ''}(null);
  const imgRef = useRef${t ? '<HTMLImageElement | null>' : ''}(null);
  const fileInputRef = useRef${t ? '<HTMLInputElement | null>' : ''}(null);

  useEffect(() => {
    Promise.all([createSession(MODEL_PATH), loadTokenizer(TOKENIZER_PATH)])
      .then(([s, tok]) => { sessionRef.current = s; tokenizerRef.current = tok; setStatus('${config.modelName} \\u00b7 Ready'); })
      .catch((e) => { setStatus('Failed to load model'); console.error(e); });
  }, []);

  const handleFile = useCallback((file${t ? ': File' : ''}) => {
    if (!file.type.startsWith('image/')) return;
    const url = URL.createObjectURL(file);
    setImageUrl(url);
    const img = new Image();
    img.src = url;
    img.onload = () => { imgRef.current = img; };
  }, []);

  const handleGenerate = useCallback(async () => {
    if (!sessionRef.current || !tokenizerRef.current || !imgRef.current || processing) return;
    setProcessing(true);
    setStatus('${config.modelName} \\u00b7 Generating...');
    const start = performance.now();
    const input = preprocessImage(imgRef.current);
    const raw = await runInference(sessionRef.current, input);
    const result = postprocessImageTextToText(raw, tokenizerRef.current);
    const elapsed = (performance.now() - start).toFixed(1);
    setStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(sessionRef.current)}\`);
    setOutput(result);
    setProcessing(false);
  }, [processing]);

  return (
    <>
      <main>
        <h1>${config.modelName} — ${taskLabel}</h1>
        <div className="container">
          <div className="multimodal-input">
            {!imageUrl ? (
              <div className={\`drop-zone\${dragOver ? ' drag-over' : ''}\`} onClick={() => fileInputRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }} onDragLeave={() => setDragOver(false)}
                onDrop={(e) => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files[0]; if (f) handleFile(f); }}>
                <p>Drop an image here or click to browse</p><p className="hint">Supports JPG, PNG, WebP</p>
                <input ref={fileInputRef} type="file" accept="image/*" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }} />
              </div>
            ) : (
              <div className="preview">
                <img src={imageUrl} alt="Selected" />
                <button className="change-btn" onClick={() => { setImageUrl(null); setOutput(''); imgRef.current = null; }}>Choose another image</button>
              </div>
            )}
            <input type="text" className="question-input" value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="Enter a prompt..." />
            <button className="primary-btn" onClick={handleGenerate} disabled={processing}>{processing ? 'Generating...' : 'Generate'}</button>
          </div>
          <div className="text-output" role="status" aria-live="polite">
            {output && <pre>{output}</pre>}
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

// ---- Audio-to-Audio ----

function emitAudioToAudioPage(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');

  return `'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from '../lib/inference.${le}';
import { postprocessAudioToAudio, playAudio } from '../lib/postprocess.${le}';

const MODEL_PATH = '${modelPath}';

export default function Page() {
  const [status, setStatus] = useState('Loading model...');
  const [output, setOutput] = useState('');
  const sessionRef = useRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);

  useEffect(() => {
    createSession(MODEL_PATH).then((s) => { sessionRef.current = s; setStatus('${config.modelName} \\u00b7 Ready'); })
      .catch((e) => { setStatus('Failed to load model'); console.error(e); });
  }, []);

  const handleFile = useCallback(async (e${t ? ': React.ChangeEvent<HTMLInputElement>' : ''}) => {
    const file = e.target.files?.[0];
    if (!file || !sessionRef.current) return;
    setStatus('${config.modelName} \\u00b7 Processing...');
    const start = performance.now();
    const audioCtx = new AudioContext({ sampleRate: 16000 });
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    const samples = audioBuffer.getChannelData(0);
    const input = new Float32Array(samples);
    const raw = await runInference(sessionRef.current, input);
    const processedSamples = postprocessAudioToAudio(raw);
    await playAudio(processedSamples);
    const elapsed = (performance.now() - start).toFixed(1);
    setStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(sessionRef.current)}\`);
    setOutput('Processed ' + (samples.length / 16000).toFixed(1) + 's of audio. Playing output...');
  }, []);

  return (
    <>
      <main>
        <h1>${config.modelName} — ${taskLabel}</h1>
        <div><label htmlFor="fileInput">Choose an audio file</label><input id="fileInput" type="file" accept="audio/*" onChange={handleFile} /></div>
        <div className="text-output" role="status" aria-live="polite">
          {output && <pre>{output}</pre>}
        </div>
      </main>
      <aside className="status-bar"><span>{status}</span></aside>
      <div className="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
    </>
  );
}
`;
}

// ---- Speaker Diarization ----

function emitSpeakerDiarizationPage(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');

  return `'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from '../lib/inference.${le}';
import { postprocessSpeakerDiarization } from '../lib/postprocess.${le}';

const MODEL_PATH = '${modelPath}';

export default function Page() {
  const [segments, setSegments] = useState${t ? '<Array<{ speaker: number; start: number; end: number; text: string }> | null>' : ''}(null);
  const [status, setStatus] = useState('Loading model...');
  const sessionRef = useRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);

  useEffect(() => {
    createSession(MODEL_PATH).then((s) => { sessionRef.current = s; setStatus('${config.modelName} \\u00b7 Ready'); })
      .catch((e) => { setStatus('Failed to load model'); console.error(e); });
  }, []);

  const handleFile = useCallback(async (e${t ? ': React.ChangeEvent<HTMLInputElement>' : ''}) => {
    const file = e.target.files?.[0];
    if (!file || !sessionRef.current) return;
    setStatus('${config.modelName} \\u00b7 Processing...');
    const start = performance.now();
    const audioCtx = new AudioContext({ sampleRate: 16000 });
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    const samples = audioBuffer.getChannelData(0);
    const input = new Float32Array(samples);
    const raw = await runInference(sessionRef.current, input);
    const results = postprocessSpeakerDiarization(raw);
    const elapsed = (performance.now() - start).toFixed(1);
    setStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(sessionRef.current)}\`);
    setSegments(results);
  }, []);

  return (
    <>
      <main>
        <h1>${config.modelName} — ${taskLabel}</h1>
        <div><label htmlFor="fileInput">Choose an audio file</label><input id="fileInput" type="file" accept="audio/*" onChange={handleFile} /></div>
        <div className="diarization-timeline" role="status" aria-live="polite">
          {segments && segments.map((seg${t ? ': { speaker: number; start: number; end: number; text: string }' : ''}, i${t ? ': number' : ''}) => (
            <div key={i} className="diarization-segment">
              <span className="speaker">Speaker {seg.speaker}</span>
              <span>{seg.text}</span>
              <span className="time">{seg.start.toFixed(1)}s - {seg.end.toFixed(1)}s</span>
            </div>
          ))}
        </div>
      </main>
      <aside className="status-bar"><span>{status}</span></aside>
      <div className="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
    </>
  );
}
`;
}

// ---- Voice Activity Detection ----

function emitVADPage(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const modelPath = getModelPath(config, '');

  return `'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from '../lib/inference.${le}';
import { postprocessVAD } from '../lib/postprocess.${le}';

const MODEL_PATH = '${modelPath}';

export default function Page() {
  const [segments, setSegments] = useState${t ? '<Array<{ label: string; start: number; end: number }> | null>' : ''}(null);
  const [status, setStatus] = useState('Loading model...');
  const sessionRef = useRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);

  useEffect(() => {
    createSession(MODEL_PATH).then((s) => { sessionRef.current = s; setStatus('${config.modelName} \\u00b7 Ready'); })
      .catch((e) => { setStatus('Failed to load model'); console.error(e); });
  }, []);

  const handleFile = useCallback(async (e${t ? ': React.ChangeEvent<HTMLInputElement>' : ''}) => {
    const file = e.target.files?.[0];
    if (!file || !sessionRef.current) return;
    setStatus('${config.modelName} \\u00b7 Processing...');
    const start = performance.now();
    const audioCtx = new AudioContext({ sampleRate: 16000 });
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    const samples = audioBuffer.getChannelData(0);
    const input = new Float32Array(samples);
    const raw = await runInference(sessionRef.current, input);
    const results = postprocessVAD(raw);
    const elapsed = (performance.now() - start).toFixed(1);
    setStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(sessionRef.current)}\`);
    setSegments(results);
  }, []);

  return (
    <>
      <main>
        <h1>${config.modelName} — ${taskLabel}</h1>
        <div><label htmlFor="fileInput">Choose an audio file</label><input id="fileInput" type="file" accept="audio/*" onChange={handleFile} /></div>
        <div className="vad-segments" role="status" aria-live="polite">
          {segments && segments.map((seg${t ? ': { label: string; start: number; end: number }' : ''}, i${t ? ': number' : ''}) => (
            <div key={i} className="vad-segment">
              <span className="label">{seg.label}</span>
              <span className="time">{seg.start.toFixed(1)}s - {seg.end.toFixed(1)}s</span>
            </div>
          ))}
        </div>
      </main>
      <aside className="status-bar"><span>{status}</span></aside>
      <div className="footer">Generated by webai.js · ${config.modelName} · ${engineLabel}</div>
    </>
  );
}
`;
}

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

// ---- Text Classification Page ----

function emitTextClassificationPage(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);

  const stateType = t ? '<{ indices: number[]; values: number[] } | null>' : '';
  const tokenizerType = t ? '<any>' : '';

  return `'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from '../lib/inference.${le}';
import { loadTokenizer, tokenizeText } from '../lib/preprocess.${le}';
import { postprocessResults } from '../lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

export default function Page() {
  const [results, setResults] = useState${stateType}(null);
  const [status, setStatus] = useState('Loading model...');
  const [text, setText] = useState('This is a sample text for classification.');
  const sessionRef = useRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
  const tokenizerRef = useRef${tokenizerType}(null);

  useEffect(() => {
    Promise.all([createSession(MODEL_PATH), loadTokenizer(TOKENIZER_PATH)])
      .then(([s, tok]) => {
        sessionRef.current = s;
        tokenizerRef.current = tok;
        setStatus('${config.modelName} \\u00b7 Ready');
      })
      .catch((e) => { setStatus('Failed to load model'); console.error('Model load error:', e); });
  }, []);

  const handleClassify = useCallback(async () => {
    const trimmed = text.trim();
    if (!trimmed || !sessionRef.current || !tokenizerRef.current) return;

    setStatus('${config.modelName} \\u00b7 Processing...');
    const start = performance.now();

    const { inputIds } = tokenizeText(tokenizerRef.current, trimmed);
    const output = await runInference(sessionRef.current, inputIds);
    const r = postprocessResults(output);

    const elapsed = (performance.now() - start).toFixed(1);
    setStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(sessionRef.current)}\`);
    setResults(r);
  }, [text]);

  return (
    <>
      <a href="#results" className="skip-link">Skip to results</a>
      <main>
        <h1>${config.modelName} — ${taskLabel}</h1>
        <div className="text-input-section">
          <label htmlFor="textInput">Enter text to classify</label>
          <textarea id="textInput" rows={4} value={text} onChange={(e) => setText(e.target.value)} aria-label="Text input for classification" />
          <button className="primary-btn" onClick={handleClassify} aria-label="Classify text">Classify</button>
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

// ---- Zero-Shot Classification Page ----

function emitZeroShotClassificationPage(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);

  const resultType = t ? '<Array<{ label: string; score: number }> | null>' : '';
  const tokenizerType = t ? '<any>' : '';

  return `'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from '../lib/inference.${le}';
import { loadTokenizer, tokenizeText } from '../lib/preprocess.${le}';
import { postprocessZeroShot } from '../lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';

export default function Page() {
  const [results, setResults] = useState${resultType}(null);
  const [status, setStatus] = useState('Loading model...');
  const [text, setText] = useState('This is a sample text for classification.');
  const [labels, setLabels] = useState('politics, sports, technology');
  const sessionRef = useRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
  const tokenizerRef = useRef${tokenizerType}(null);

  useEffect(() => {
    Promise.all([createSession(MODEL_PATH), loadTokenizer(TOKENIZER_PATH)])
      .then(([s, tok]) => {
        sessionRef.current = s;
        tokenizerRef.current = tok;
        setStatus('${config.modelName} \\u00b7 Ready');
      })
      .catch((e) => { setStatus('Failed to load model'); console.error('Model load error:', e); });
  }, []);

  const handleClassify = useCallback(async () => {
    const trimmed = text.trim();
    const labelsRaw = labels.trim();
    if (!trimmed || !labelsRaw || !sessionRef.current || !tokenizerRef.current) return;

    const labelList = labelsRaw.split(',').map((l) => l.trim()).filter(Boolean);
    if (labelList.length === 0) return;

    setStatus('${config.modelName} \\u00b7 Processing...');
    const start = performance.now();

    const scores${t ? ': number[]' : ''} = [];
    for (const label of labelList) {
      const hypothesis = trimmed + ' </s></s> ' + 'This is about ' + label + '.';
      const { inputIds } = tokenizeText(tokenizerRef.current, hypothesis);
      const output = await runInference(sessionRef.current, inputIds);
      const score = output[output.length > 2 ? 2 : output.length - 1];
      scores.push(score);
    }

    const r = postprocessZeroShot(scores, labelList);

    const elapsed = (performance.now() - start).toFixed(1);
    setStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(sessionRef.current)}\`);
    setResults(r);
  }, [text, labels]);

  return (
    <>
      <a href="#results" className="skip-link">Skip to results</a>
      <main>
        <h1>${config.modelName} — ${taskLabel}</h1>
        <div className="text-input-section">
          <label htmlFor="textInput">Enter text to classify</label>
          <textarea id="textInput" rows={4} value={text} onChange={(e) => setText(e.target.value)} aria-label="Text input for classification" />
          <label htmlFor="labelsInput">Candidate labels (comma-separated)</label>
          <input id="labelsInput" type="text" value={labels} onChange={(e) => setLabels(e.target.value)} aria-label="Comma-separated candidate labels" />
          <button className="primary-btn" onClick={handleClassify} aria-label="Classify text">Classify</button>
        </div>
        <div id="results" className="results" role="status" aria-live="polite" aria-atomic="true">
          {results && results.map((r${t ? ': { label: string; score: number }' : ''}, i${t ? ': number' : ''}) => {
            const pct = (r.score * 100).toFixed(1);
            const maxVal = results[0]?.score || 1;
            return (
              <div key={r.label} className={\`result-row\${i === 0 ? ' top-result' : ''}\`} tabIndex={0}
                   aria-label={\`\${r.label}: \${pct} percent\`}>
                <span className="result-label">{r.label}</span>
                <div className="result-bar-container">
                  <div className="result-bar" style={{ width: \`\${(r.score / maxVal) * 100}%\` }} />
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

// ---- Text Generation Page ----

function emitTextGenerationPage(config: ResolvedConfig): string {
  const t = config.lang === 'ts';
  const le = libExt(config);
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);

  const tokenizerType = t ? '<any>' : '';

  return `'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { createSession, runInference, getBackendLabel } from '../lib/inference.${le}';
import { loadTokenizer } from '../lib/preprocess.${le}';
import { sampleNextToken, postprocessGeneration } from '../lib/postprocess.${le}';

const MODEL_PATH = '${getModelPath(config, '')}';
const TOKENIZER_PATH = MODEL_PATH.replace(/\\.onnx$/, '') + '/tokenizer.json';
const MAX_NEW_TOKENS = 50;
const EOS_TOKEN_ID = 2;

export default function Page() {
  const [status, setStatus] = useState('Loading model...');
  const [text, setText] = useState('Once upon a time');
  const [output, setOutput] = useState('');
  const [generating, setGenerating] = useState(false);
  const sessionRef = useRef${t ? '<Awaited<ReturnType<typeof createSession>> | null>' : ''}(null);
  const tokenizerRef = useRef${tokenizerType}(null);

  useEffect(() => {
    Promise.all([createSession(MODEL_PATH), loadTokenizer(TOKENIZER_PATH)])
      .then(([s, tok]) => {
        sessionRef.current = s;
        tokenizerRef.current = tok;
        setStatus('${config.modelName} \\u00b7 Ready');
      })
      .catch((e) => { setStatus('Failed to load model'); console.error('Model load error:', e); });
  }, []);

  const handleGenerate = useCallback(async () => {
    const prompt = text.trim();
    if (!prompt || !sessionRef.current || !tokenizerRef.current) return;

    setGenerating(true);
    setStatus('${config.modelName} \\u00b7 Generating...');
    const start = performance.now();

    const encoded = tokenizerRef.current.encode(prompt);
    let inputIds = encoded.inputIds;
    setOutput(prompt);

    for (let i = 0; i < MAX_NEW_TOKENS; i++) {
      const inputBigInt = new BigInt64Array(inputIds.map((id${t ? ': number' : ''}) => BigInt(id)));
      const modelOutput = await runInference(sessionRef.current, inputBigInt);

      const vocabSize = tokenizerRef.current.getVocabSize();
      const seqLen = inputIds.length;
      const logits = postprocessGeneration(modelOutput, seqLen, vocabSize);
      const nextToken = sampleNextToken(logits);

      if (nextToken === EOS_TOKEN_ID) break;

      inputIds = [...inputIds, nextToken];
      const decoded = tokenizerRef.current.decode(inputIds);
      setOutput(decoded);
    }

    const elapsed = (performance.now() - start).toFixed(1);
    setStatus(\`${config.modelName} \\u00b7 \${elapsed}ms \\u00b7 \${getBackendLabel(sessionRef.current)}\`);
    setGenerating(false);
  }, [text]);

  return (
    <>
      <a href="#output" className="skip-link">Skip to output</a>
      <main>
        <h1>${config.modelName} — ${taskLabel}</h1>
        <div className="text-input-section">
          <label htmlFor="textInput">Enter prompt text</label>
          <textarea id="textInput" rows={3} value={text} onChange={(e) => setText(e.target.value)} aria-label="Text prompt for generation" />
          <button className="primary-btn" onClick={handleGenerate} disabled={generating} aria-label="Generate text">Generate</button>
        </div>
        <div id="output" className="text-output" role="status" aria-live="polite" aria-atomic="true">
          {output && <pre>{output}</pre>}
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

  // Text/NLP tasks
  if (config.task === 'text-classification') return emitTextClassificationPage(config);
  if (config.task === 'zero-shot-classification') return emitZeroShotClassificationPage(config);
  if (config.task === 'text-generation') return emitTextGenerationPage(config);
  if (config.task === 'fill-mask') return emitFillMaskPage(config);
  if (config.task === 'sentence-similarity') return emitSentenceSimilarityPage(config);
  if (config.task === 'token-classification') return emitTokenClassificationPage(config);
  if (config.task === 'question-answering') return emitQuestionAnsweringPage(config);
  if (config.task === 'summarization') return emitSummarizationPage(config);
  if (config.task === 'translation') return emitTranslationPage(config);
  if (config.task === 'text2text-generation') return emitText2TextPage(config);
  if (config.task === 'conversational') return emitConversationalPage(config);
  if (config.task === 'table-question-answering') return emitTableQAPage(config);

  // Audio tasks (new)
  if (config.task === 'audio-to-audio') return emitAudioToAudioPage(config);
  if (config.task === 'speaker-diarization') return emitSpeakerDiarizationPage(config);
  if (config.task === 'voice-activity-detection') return emitVADPage(config);

  // Multimodal tasks
  if (config.task === 'image-to-text') return emitImageToTextPage(config);
  if (config.task === 'visual-question-answering') return emitVQAPage(config);
  if (config.task === 'document-question-answering') return emitDocQAPage(config);
  if (config.task === 'image-text-to-text') return emitImageTextToTextPage(config);

  // Depth estimation
  if (config.task === 'depth-estimation') return emitDepthEstimationPage(config);

  // Visual realtime
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
}

/* Text input section */
.text-input-section {
  margin-bottom: var(--webai-space-6);
}

.text-input-section label {
  display: block;
  margin-bottom: var(--webai-space-2);
  font-weight: 500;
  color: var(--webai-text);
}

.text-input-section textarea,
.text-input-section input[type="text"] {
  width: 100%;
  padding: var(--webai-space-3);
  border: 1px solid var(--webai-border);
  border-radius: var(--webai-radius);
  background: var(--webai-surface);
  color: var(--webai-text);
  font-family: inherit;
  font-size: var(--webai-font-size-base);
  margin-bottom: var(--webai-space-3);
}

.text-input-section textarea {
  resize: vertical;
}

/* Text output */
.text-output {
  margin-top: var(--webai-space-4);
}

.text-output pre {
  padding: var(--webai-space-4);
  background: var(--webai-surface);
  border: 1px solid var(--webai-border);
  border-radius: var(--webai-radius);
  color: var(--webai-text);
  font-family: var(--webai-font-mono);
  font-size: var(--webai-font-size-sm);
  white-space: pre-wrap;
  word-wrap: break-word;
  overflow-x: auto;
}`;
}

function needsExtendedCSS(config: ResolvedConfig): boolean {
  return config.input === 'camera' || config.input === 'screen' ||
    config.task === 'object-detection' || config.task === 'image-segmentation' ||
    config.task === 'feature-extraction' ||
    config.task === 'text-classification' || config.task === 'zero-shot-classification' ||
    config.task === 'text-generation' ||
    config.task === 'depth-estimation' || config.task === 'image-to-text' ||
    config.task === 'visual-question-answering' || config.task === 'document-question-answering' ||
    config.task === 'image-text-to-text';
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
