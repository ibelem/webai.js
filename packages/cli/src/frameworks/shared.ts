/**
 * Shared utilities for framework emitters (Layer 2).
 *
 * CSS design system, README template, and CodeBlock processing helpers.
 */

import type { ResolvedConfig } from '@webai/core';
import type { CodeBlock, GeneratedFile } from '../types.js';

// ---- CSS Design System ----

const TASK_LABELS: Record<string, string> = {
  'image-classification': 'Image Classification',
  'object-detection': 'Object Detection',
  'image-segmentation': 'Image Segmentation',
  'feature-extraction': 'Feature Extraction',
  'depth-estimation': 'Depth Estimation',
  'speech-to-text': 'Speech to Text',
  'audio-classification': 'Audio Classification',
  'text-to-speech': 'Text to Speech',
  'audio-to-audio': 'Audio to Audio',
  'speaker-diarization': 'Speaker Diarization',
  'voice-activity-detection': 'Voice Activity Detection',
  'text-classification': 'Text Classification',
  'text-generation': 'Text Generation',
  'zero-shot-classification': 'Zero-Shot Classification',
  'fill-mask': 'Fill Mask',
  'sentence-similarity': 'Sentence Similarity',
  'token-classification': 'Token Classification (NER)',
  'question-answering': 'Question Answering',
  'summarization': 'Summarization',
  'translation': 'Translation',
  'text2text-generation': 'Text-to-Text Generation',
  'conversational': 'Conversational',
  'table-question-answering': 'Table Question Answering',
  'image-to-text': 'Image to Text',
  'visual-question-answering': 'Visual Question Answering',
  'document-question-answering': 'Document Question Answering',
  'image-text-to-text': 'Image-Text to Text',
};

const ENGINE_LABELS: Record<string, string> = {
  ort: 'ONNX Runtime Web',
  litert: 'LiteRT.js',
  webnn: 'WebNN API',
};

export function getTaskLabel(task: string): string {
  return TASK_LABELS[task] ?? task;
}

export function getEngineLabel(engine: string): string {
  return ENGINE_LABELS[engine] ?? engine;
}

/**
 * Get the MODEL_PATH value for generated code.
 *
 * When the model came from a URL or HuggingFace model ID, the generated code
 * references the URL directly (works with fetch() and ORT's InferenceSession.create()).
 * When the model is local, it uses a relative path.
 *
 * @param config - Resolved config
 * @param prefix - Path prefix for local models ('.' for html, '' for vite/next/svelte)
 * @returns The MODEL_PATH string value (without quotes)
 */
export function getModelPath(config: ResolvedConfig, prefix = '.'): string {
  if (config.modelSource !== 'local-path' && config.modelUrl) {
    return config.modelUrl;
  }
  return `${prefix}/${config.modelName}.onnx`;
}

/** Generate CSS custom properties (design system) */
export function emitDesignSystemCSS(_config: ResolvedConfig): string {
  return `:root {
  /* Colors — dark theme (default) */
  --webai-bg: #0a0a0a;
  --webai-surface: #1a1a1a;
  --webai-text: #e5e5e5;
  --webai-text-muted: #737373;
  --webai-accent: #3b82f6;
  --webai-success: #22c55e;
  --webai-warning: #eab308;
  --webai-error: #ef4444;
  --webai-border: #262626;

  /* Typography */
  --webai-font-mono: ui-monospace, 'Cascadia Code', 'Source Code Pro', monospace;
  --webai-font-sans: system-ui, -apple-system, sans-serif;
  --webai-font-size-sm: 12px;
  --webai-font-size-base: 14px;
  --webai-font-size-lg: 16px;

  /* Spacing (4px base unit) */
  --webai-space-1: 4px;
  --webai-space-2: 8px;
  --webai-space-3: 12px;
  --webai-space-4: 16px;
  --webai-space-6: 24px;
  --webai-space-8: 32px;

  /* Layout */
  --webai-status-height: 32px;
  --webai-radius: 4px;
}

/* Light theme override */
:root[data-theme="light"] {
  --webai-bg: #fafafa;
  --webai-surface: #ffffff;
  --webai-text: #171717;
  --webai-text-muted: #737373;
  --webai-border: #e5e5e5;
}`;
}

/** Generate component CSS for file input + classification results */
export function emitAppCSS(): string {
  return `* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  background: var(--webai-bg);
  color: var(--webai-text);
  font-family: var(--webai-font-sans);
  font-size: var(--webai-font-size-base);
  min-height: 100vh;
  padding-bottom: var(--webai-status-height);
}

main {
  max-width: 960px;
  margin: 0 auto;
  padding: var(--webai-space-8) var(--webai-space-4);
}

h1 {
  font-size: 20px;
  font-weight: 600;
  margin-bottom: var(--webai-space-6);
}

.container {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--webai-space-6);
  align-items: start;
}

@media (max-width: 640px) {
  .container {
    grid-template-columns: 1fr;
  }
}

/* Drop zone */
.drop-zone {
  border: 2px dashed var(--webai-border);
  border-radius: var(--webai-radius);
  padding: var(--webai-space-8);
  text-align: center;
  cursor: pointer;
  transition: border-color 0.15s;
}

.drop-zone:hover,
.drop-zone:focus-visible,
.drop-zone.drag-over {
  border-color: var(--webai-accent);
}

.drop-zone:focus-visible {
  outline: 2px solid var(--webai-accent);
  outline-offset: 2px;
}

.drop-zone .hint {
  color: var(--webai-text-muted);
  font-size: var(--webai-font-size-sm);
  margin-top: var(--webai-space-2);
}

/* Preview */
.preview img {
  max-width: 100%;
  border-radius: var(--webai-radius);
  display: block;
}

.preview .change-btn {
  margin-top: var(--webai-space-2);
  background: none;
  border: 1px solid var(--webai-border);
  color: var(--webai-text-muted);
  padding: var(--webai-space-1) var(--webai-space-3);
  border-radius: var(--webai-radius);
  cursor: pointer;
  font-size: var(--webai-font-size-sm);
}

.preview .change-btn:hover {
  border-color: var(--webai-text-muted);
}

/* Results */
.results {
  display: flex;
  flex-direction: column;
  gap: var(--webai-space-1);
}

.result-row {
  display: grid;
  grid-template-columns: 120px 1fr 60px;
  align-items: center;
  gap: var(--webai-space-2);
  height: 24px;
  opacity: 0.7;
}

.result-row:focus-visible {
  outline: 1px solid var(--webai-accent);
  outline-offset: 1px;
  border-radius: 2px;
}

.result-row.top-result {
  opacity: 1;
  font-weight: 600;
}

.result-label {
  font-family: var(--webai-font-mono);
  font-size: var(--webai-font-size-sm);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.result-bar-container {
  height: 16px;
  background: var(--webai-surface);
  border-radius: 2px;
  overflow: hidden;
}

.result-bar {
  height: 100%;
  background: var(--webai-accent);
  border-radius: 2px;
  transition: width 0.3s ease;
}

.result-pct {
  font-family: var(--webai-font-mono);
  font-size: var(--webai-font-size-sm);
  text-align: right;
  color: var(--webai-text-muted);
}

.result-row.top-result .result-pct {
  color: var(--webai-text);
}

/* Status bar */
.status-bar {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  height: var(--webai-status-height);
  background: var(--webai-surface);
  border-top: 1px solid var(--webai-border);
  display: flex;
  align-items: center;
  padding: 0 var(--webai-space-4);
  font-family: var(--webai-font-mono);
  font-size: var(--webai-font-size-sm);
  color: var(--webai-text-muted);
}

/* Footer */
.footer {
  position: fixed;
  bottom: var(--webai-status-height);
  right: var(--webai-space-4);
  font-size: var(--webai-font-size-sm);
  color: var(--webai-text-muted);
  opacity: 0.5;
  padding: var(--webai-space-1);
}

/* Skip link */
.skip-link {
  position: absolute;
  left: -9999px;
  top: var(--webai-space-2);
  background: var(--webai-accent);
  color: white;
  padding: var(--webai-space-2) var(--webai-space-4);
  border-radius: var(--webai-radius);
  z-index: 100;
  text-decoration: none;
}

.skip-link:focus {
  left: var(--webai-space-2);
}

/* Audio: transcript display */
.transcript {
  background: var(--webai-surface);
  border: 1px solid var(--webai-border);
  border-radius: 8px;
  padding: 1rem;
  min-height: 100px;
  white-space: pre-wrap;
}

/* Audio: controls */
.controls { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
.controls button {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  background: var(--webai-accent);
  color: white;
}
.controls button:disabled { opacity: 0.5; cursor: not-allowed; }

/* Audio: TTS input */
.tts-input { display: flex; flex-direction: column; gap: 0.5rem; }
.tts-input textarea {
  background: var(--webai-surface);
  border: 1px solid var(--webai-border);
  border-radius: 6px;
  color: var(--webai-text);
  padding: 0.75rem;
  resize: vertical;
}

/* Text task: input area */
.text-input {
  display: flex;
  flex-direction: column;
  gap: var(--webai-space-3);
}

.text-input textarea {
  background: var(--webai-surface);
  border: 1px solid var(--webai-border);
  border-radius: var(--webai-radius);
  color: var(--webai-text);
  font-family: var(--webai-font-sans);
  font-size: var(--webai-font-size-base);
  padding: var(--webai-space-3);
  resize: vertical;
  min-height: 100px;
}

.text-input textarea:focus {
  outline: 2px solid var(--webai-accent);
  outline-offset: -2px;
}

/* Text task: run button */
.run-btn {
  align-self: flex-start;
  background: var(--webai-accent);
  color: white;
  border: none;
  padding: var(--webai-space-2) var(--webai-space-6);
  border-radius: var(--webai-radius);
  cursor: pointer;
  font-size: var(--webai-font-size-base);
}

.run-btn:hover { opacity: 0.9; }
.run-btn:disabled { opacity: 0.5; cursor: not-allowed; }

/* Text task: generation output */
.generation-output {
  background: var(--webai-surface);
  border: 1px solid var(--webai-border);
  border-radius: var(--webai-radius);
  padding: var(--webai-space-3);
  white-space: pre-wrap;
  font-family: var(--webai-font-sans);
  min-height: 80px;
}

/* Zero-shot: label input */
.labels-input {
  background: var(--webai-surface);
  border: 1px solid var(--webai-border);
  border-radius: var(--webai-radius);
  color: var(--webai-text);
  font-family: var(--webai-font-mono);
  font-size: var(--webai-font-size-sm);
  padding: var(--webai-space-2);
}

/* Depth estimation: canvas output */
.depth-canvas {
  max-width: 100%;
  border-radius: var(--webai-radius);
  display: block;
}

/* NER: entity highlights */
.ner-output {
  background: var(--webai-surface);
  border: 1px solid var(--webai-border);
  border-radius: var(--webai-radius);
  padding: var(--webai-space-3);
  line-height: 2;
  white-space: pre-wrap;
}

.ner-entity {
  padding: 2px 6px;
  border-radius: 3px;
  font-weight: 500;
}

.ner-entity[data-type="PER"] { background: rgba(59,130,246,0.2); color: #60a5fa; }
.ner-entity[data-type="ORG"] { background: rgba(34,197,94,0.2); color: #4ade80; }
.ner-entity[data-type="LOC"] { background: rgba(234,179,8,0.2); color: #facc15; }
.ner-entity[data-type="MISC"] { background: rgba(168,85,247,0.2); color: #c084fc; }

/* QA: answer highlight */
.qa-input { display: flex; flex-direction: column; gap: var(--webai-space-3); }
.qa-input label {
  font-size: var(--webai-font-size-sm);
  color: var(--webai-text-muted);
}
.qa-answer {
  background: var(--webai-surface);
  border: 1px solid var(--webai-border);
  border-radius: var(--webai-radius);
  padding: var(--webai-space-3);
  font-size: var(--webai-font-size-lg);
}
.qa-answer .score {
  font-size: var(--webai-font-size-sm);
  color: var(--webai-text-muted);
  margin-top: var(--webai-space-1);
}

/* Similarity: score pairs */
.similarity-pairs {
  display: flex;
  flex-direction: column;
  gap: var(--webai-space-2);
}
.similarity-score {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--webai-space-2) var(--webai-space-3);
  background: var(--webai-surface);
  border-radius: var(--webai-radius);
}
.similarity-score .value {
  font-family: var(--webai-font-mono);
  font-weight: 600;
  color: var(--webai-accent);
}

/* Fill-mask: predictions */
.mask-predictions {
  display: flex;
  flex-direction: column;
  gap: var(--webai-space-1);
}
.mask-prediction {
  display: flex;
  justify-content: space-between;
  padding: var(--webai-space-2) var(--webai-space-3);
  background: var(--webai-surface);
  border-radius: var(--webai-radius);
}
.mask-prediction .token { font-weight: 600; color: var(--webai-accent); }
.mask-prediction .prob {
  font-family: var(--webai-font-mono);
  font-size: var(--webai-font-size-sm);
  color: var(--webai-text-muted);
}

/* Conversational: chat bubbles */
.chat-messages {
  display: flex;
  flex-direction: column;
  gap: var(--webai-space-2);
  max-height: 400px;
  overflow-y: auto;
  padding: var(--webai-space-3);
  background: var(--webai-surface);
  border: 1px solid var(--webai-border);
  border-radius: var(--webai-radius);
}
.chat-msg {
  padding: var(--webai-space-2) var(--webai-space-3);
  border-radius: var(--webai-radius);
  max-width: 80%;
}
.chat-msg.user { background: var(--webai-accent); color: white; align-self: flex-end; }
.chat-msg.bot { background: var(--webai-border); align-self: flex-start; }
.chat-input-row { display: flex; gap: var(--webai-space-2); }
.chat-input-row input {
  flex: 1;
  background: var(--webai-surface);
  border: 1px solid var(--webai-border);
  border-radius: var(--webai-radius);
  color: var(--webai-text);
  padding: var(--webai-space-2) var(--webai-space-3);
}

/* Table QA: table input */
.table-input textarea {
  background: var(--webai-surface);
  border: 1px solid var(--webai-border);
  border-radius: var(--webai-radius);
  color: var(--webai-text);
  font-family: var(--webai-font-mono);
  font-size: var(--webai-font-size-sm);
  padding: var(--webai-space-3);
  resize: vertical;
  width: 100%;
  min-height: 120px;
}

/* Multimodal: image + question layout */
.multimodal-input {
  display: flex;
  flex-direction: column;
  gap: var(--webai-space-3);
}
.multimodal-input .question-input {
  background: var(--webai-surface);
  border: 1px solid var(--webai-border);
  border-radius: var(--webai-radius);
  color: var(--webai-text);
  padding: var(--webai-space-2) var(--webai-space-3);
}

/* Speaker diarization: timeline */
.diarization-timeline {
  display: flex;
  flex-direction: column;
  gap: var(--webai-space-1);
}
.diarization-segment {
  display: grid;
  grid-template-columns: 80px 1fr 80px;
  align-items: center;
  gap: var(--webai-space-2);
  padding: var(--webai-space-1) var(--webai-space-2);
  background: var(--webai-surface);
  border-radius: var(--webai-radius);
  font-size: var(--webai-font-size-sm);
}
.diarization-segment .speaker {
  font-weight: 600;
  color: var(--webai-accent);
}
.diarization-segment .time {
  font-family: var(--webai-font-mono);
  color: var(--webai-text-muted);
  text-align: right;
}

/* VAD: speech segments */
.vad-segments {
  display: flex;
  flex-direction: column;
  gap: var(--webai-space-1);
}
.vad-segment {
  display: flex;
  justify-content: space-between;
  padding: var(--webai-space-2) var(--webai-space-3);
  background: var(--webai-surface);
  border-radius: var(--webai-radius);
  font-family: var(--webai-font-mono);
  font-size: var(--webai-font-size-sm);
}
.vad-segment .label { color: var(--webai-success); }
.vad-segment .time { color: var(--webai-text-muted); }

/* Reduced motion */
@media (prefers-reduced-motion: reduce) {
  .result-bar { transition: none; }
}`;
}

// ---- Audio Task Helpers ----

/** Check if a task is an audio task */
export function isAudioTask(task: string): boolean {
  return ['speech-to-text', 'audio-classification', 'text-to-speech', 'audio-to-audio', 'speaker-diarization', 'voice-activity-detection'].includes(task);
}

/** Check if a task is a text/NLP task */
export function isTextTask(task: string): boolean {
  return ['text-classification', 'text-generation', 'zero-shot-classification', 'fill-mask', 'sentence-similarity', 'token-classification', 'question-answering', 'summarization', 'translation', 'text2text-generation', 'conversational', 'table-question-answering'].includes(task);
}

/** Collect auxiliary files from all code blocks (e.g. AudioWorklet processor) */
export function collectAuxiliaryFiles(blocks: CodeBlock[]): GeneratedFile[] {
  const files: GeneratedFile[] = [];
  for (const block of blocks) {
    if (block.auxiliaryFiles) {
      files.push(...block.auxiliaryFiles);
    }
  }
  return files;
}

// ---- Code Processing ----

/** Strip import lines from a code block's code */
export function stripImports(code: string): string {
  return code
    .split('\n')
    .filter((line) => !line.startsWith('import '))
    .join('\n')
    .replace(/^\n+/, '');
}

/** Add export keyword to function declarations for named exports */
export function addExports(code: string, exportNames: string[]): string {
  let result = code;
  for (const name of exportNames) {
    result = result.replace(
      new RegExp(`^(async )?function ${name}\\(`, 'm'),
      (_match, asyncPrefix) => `export ${asyncPrefix ?? ''}function ${name}(`,
    );
  }
  return result;
}

/** Find a CodeBlock by id */
export function findBlock(blocks: CodeBlock[], id: string): CodeBlock | undefined {
  return blocks.find((b) => b.id === id);
}

/** Collect all npm dependencies from CodeBlocks */
export function collectImports(blocks: CodeBlock[]): string[] {
  const deps = new Set<string>();
  for (const block of blocks) {
    for (const imp of block.imports) {
      deps.add(imp);
    }
  }
  return Array.from(deps);
}

// ---- README ----

export function emitReadme(config: ResolvedConfig, files: string[]): string {
  const taskLabel = getTaskLabel(config.task);
  const engineLabel = getEngineLabel(config.engine);
  const { imageSize, mean, std } = config.preprocess;

  const isRemote = config.modelSource !== 'local-path' && !!config.modelUrl;

  let quickStart: string;
  if (config.framework === 'html') {
    if (isRemote) {
      quickStart = `1. Start a local server: \`npx serve .\` (or any static file server)
2. Open \`index.html\` in your browser
3. The model loads automatically from the URL on first run`;
    } else {
      quickStart = `1. Copy your model file (\`${config.modelName}.onnx\`) into this directory
2. Start a local server: \`npx serve .\` (or any static file server)
3. Open \`index.html\` in your browser`;
    }
  } else if (isRemote) {
    quickStart = `\`\`\`bash
npm install
npm run dev
\`\`\`

The model loads automatically from the URL on first run. No manual file copy needed.`;
  } else if (config.framework === 'react-vite' || config.framework === 'vue-vite' || config.framework === 'nuxt' || config.framework === 'astro') {
    quickStart = `\`\`\`bash
npm install
# Copy your model file to public/
cp /path/to/${config.modelName}.onnx public/
npm run dev
\`\`\``;
  } else {
    quickStart = `\`\`\`bash
npm install
npm run dev
\`\`\``;
  }

  return `# ${config.modelName} — ${taskLabel}

Generated by [webai.js](https://github.com/ibelem/webai.js)

## Quick Start

${quickStart}

## How It Works

1. **Preprocessing**: Image is resized to ${imageSize}x${imageSize}px, normalized with
   mean=[${mean.join(', ')}] and std=[${std.join(', ')}], then transposed to ${config.preprocess.layout.toUpperCase()} layout.
2. **Inference**: Model runs via ${engineLabel}${config.backend === 'auto' ? ' with automatic backend selection (WebNN NPU > WebNN GPU > WebGPU > WASM)' : ` with ${config.backend} backend`}.
3. **Postprocessing**: Raw logits are converted to probabilities (softmax), then top-5 results are extracted.

## Configuration

| Setting | Value |
|---------|-------|
| Task | ${taskLabel} |
| Engine | ${engineLabel} |
| Backend | ${config.backend} |
| Framework | ${config.framework} |
| Model | ${config.modelName} |
| Input | ${config.input} |

## Files

${files.map((f) => `- \`${f}\``).join('\n')}
`;
}
