/**
 * webai.js Web UI — main entry point.
 *
 * Wires up: config panel → assembler → Monaco preview → Try it iframe.
 * Also: theme toggle, copy-to-clipboard, download-as-zip.
 */

import './style.css';
import { resolveConfig } from '@webai/core';
import type { CliFlags } from '@webai/core';
import { assemble } from 'webai';
import type { GeneratedFile } from 'webai';
import { zipSync, strToU8 } from 'fflate';
import { setupConfigPanel, type ConfigValues } from './config-panel.js';
import { setupCodePreview, updateCodePreview, getActiveFileContent } from './code-preview.js';
import { setupTryIt } from './try-it.js';
import { createMockMetadata } from './mock-metadata.js';

let currentFramework = 'html';
let currentFiles: GeneratedFile[] = [];

function generateCode(values: ConfigValues): GeneratedFile[] {
  const metadata = createMockMetadata(values.task, values.engine === 'litert' ? 'tflite' : 'onnx');

  const ext = values.engine === 'litert' ? '.tflite' : '.onnx';
  const flags: CliFlags = {
    model: values.modelUrl ?? `./${values.modelName}${ext}`,
    task: values.task,
    engine: values.engine,
    backend: values.backend,
    framework: values.framework,
    input: values.input,
    lang: values.lang,
    mode: 'raw',
    output: './output/',
    offline: values.offline,
    theme: values.theme,
    modelSource: values.modelSource,
    modelUrl: values.modelUrl,
  };

  try {
    const { config } = resolveConfig(flags, metadata);
    return assemble(config);
  } catch (e) {
    console.error('Generation error:', e);
    return [{
      path: 'error.txt',
      content: `Code generation failed:\n${e instanceof Error ? e.message : String(e)}`,
    }];
  }
}

// ---- Theme toggle ----

function setupThemeToggle(btn: HTMLButtonElement): void {
  const stored = localStorage.getItem('webai-theme');
  if (stored === 'light') {
    document.documentElement.setAttribute('data-theme', 'light');
  }

  btn.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('webai-theme', next);
  });
}

// ---- Copy to clipboard ----

function setupCopyButton(btn: HTMLButtonElement): void {
  btn.addEventListener('click', async () => {
    const content = getActiveFileContent();
    if (!content) return;

    try {
      await navigator.clipboard.writeText(content);
      const original = btn.textContent;
      btn.textContent = 'Copied!';
      setTimeout(() => { btn.textContent = original; }, 1500);
    } catch {
      // Fallback for older browsers
      const textarea = document.createElement('textarea');
      textarea.value = content;
      textarea.style.position = 'fixed';
      textarea.style.opacity = '0';
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);

      const original = btn.textContent;
      btn.textContent = 'Copied!';
      setTimeout(() => { btn.textContent = original; }, 1500);
    }
  });
}

// ---- Download as ZIP ----

function setupDownloadZip(btn: HTMLButtonElement): void {
  btn.addEventListener('click', () => {
    if (currentFiles.length === 0) return;

    const files: Record<string, Uint8Array> = {};
    for (const file of currentFiles) {
      files[file.path] = strToU8(file.content);
    }

    const zipped = zipSync(files);
    const blob = new Blob([zipped.buffer as ArrayBuffer], { type: 'application/zip' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = 'webai-generated.zip';
    a.click();

    URL.revokeObjectURL(url);
  });
}

// ---- Init ----

async function init(): Promise<void> {
  const configPanel = document.getElementById('configPanel')!;
  const editorContainer = document.getElementById('editor')!;
  const tabContainer = document.getElementById('fileTabs')!;
  const tryItSection = document.getElementById('tryItSection')!;
  const tryItBtn = document.getElementById('tryItBtn') as HTMLButtonElement;
  const closeTryIt = document.getElementById('closeTryIt') as HTMLButtonElement;
  const tryItFrame = document.getElementById('tryItFrame') as HTMLIFrameElement;
  const themeToggle = document.getElementById('themeToggle') as HTMLButtonElement;
  const copyBtn = document.getElementById('copyBtn') as HTMLButtonElement;
  const downloadZipBtn = document.getElementById('downloadZipBtn') as HTMLButtonElement;

  setupThemeToggle(themeToggle);
  setupCopyButton(copyBtn);
  setupDownloadZip(downloadZipBtn);

  await setupCodePreview(editorContainer, tabContainer);

  setupTryIt(
    tryItSection,
    tryItBtn,
    closeTryIt,
    tryItFrame,
    () => currentFiles,
    () => currentFramework,
  );

  setupConfigPanel(configPanel, (values) => {
    currentFramework = values.framework;
    currentFiles = generateCode(values);
    updateCodePreview(currentFiles, tabContainer);
  });
}

init();
