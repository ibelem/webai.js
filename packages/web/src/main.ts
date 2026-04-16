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
import { setupConfigPanel, updateUrlParams, readUrlParams, type ConfigValues } from './config-panel.js';
import { setupCodePreview, updateCodePreview, getActiveFileContent, setEditorTheme, relayoutEditor } from './code-preview.js';
import { setupTryIt, canTryIt, runInIframe } from './try-it.js';
import { createMockMetadata } from './mock-metadata.js';

let currentFramework = 'html';
let currentFiles: GeneratedFile[] = [];
let latestConfigValues: ConfigValues | null = null;

function generateCode(values: ConfigValues, pageTheme: string): GeneratedFile[] {
  const metadata = createMockMetadata(values.task, values.engine === 'litert' ? 'tflite' : 'onnx');

  const ext = values.engine === 'litert' ? '.tflite' : '.onnx';
  const flags: CliFlags = {
    model: values.modelUrl ?? `./${values.modelName}${ext}`,
    task: values.task,
    engine: values.engine,
    framework: values.framework,
    input: values.input,
    lang: values.lang,
    mode: 'raw',
    output: './output/',
    offline: values.offline,
    theme: pageTheme,
    modelSource: values.modelSource,
    modelUrl: values.modelUrl,
    hfModelId: values.hfModelId,
    hfFile: values.hfFile,
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

function getPageTheme(): 'dark' | 'light' {
  return document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
}

function setupThemeToggle(btn: HTMLButtonElement, onThemeChange: (theme: string) => void): void {
  // URL param takes precedence, then localStorage
  const urlParams = readUrlParams();
  const themeSource = urlParams.theme ?? localStorage.getItem('webai-theme');
  if (themeSource === 'light') {
    document.documentElement.setAttribute('data-theme', 'light');
    setEditorTheme('light');
  }

  btn.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('webai-theme', next);
    setEditorTheme(next);
    onThemeChange(next);
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

// ---- Panel toggles ----

function setupSidebarToggle(btn: HTMLButtonElement, layout: HTMLElement): void {
  btn.addEventListener('click', () => {
    layout.classList.toggle('sidebar-collapsed');
    btn.classList.toggle('is-collapsed');
    const collapsed = layout.classList.contains('sidebar-collapsed');
    btn.setAttribute('aria-expanded', String(!collapsed));
    btn.title = collapsed ? 'Show sidebar' : 'Hide sidebar';
    // Re-measure Monaco after grid column transition
    setTimeout(() => relayoutEditor(), 300);
  });
}

function setupFullscreenToggle(btn: HTMLButtonElement, section: HTMLElement): void {
  btn.addEventListener('click', () => {
    const entering = !section.classList.contains('fullscreen');
    section.classList.toggle('fullscreen');
    btn.title = entering ? 'Exit fullscreen' : 'Enter fullscreen';
    // Double rAF ensures browser has completed reflow before Monaco re-measures
    requestAnimationFrame(() => requestAnimationFrame(() => relayoutEditor()));
  });
}

function setupEscapeKey(codeSection: HTMLElement, tryItSection: HTMLElement): void {
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (tryItSection.classList.contains('fullscreen')) {
      tryItSection.classList.remove('fullscreen');
    } else if (codeSection.classList.contains('fullscreen')) {
      codeSection.classList.remove('fullscreen');
    }
    requestAnimationFrame(() => requestAnimationFrame(() => relayoutEditor()));
  });
}

// ---- Init ----

async function init(): Promise<void> {
  // Redirect bare URL to include full default params
  if (!window.location.search) {
    const defaults = new URLSearchParams({
      model: 'webnn/mobilenet-v2',
      file: 'onnx/model_fp16.onnx',
      task: 'image-classification',
      engine: 'ort',
      framework: 'html',
      input: 'file',
      lang: 'js',
      theme: 'dark',
    });
    const url = `${window.location.pathname}?${defaults.toString()}`;
    window.history.replaceState(null, '', url);
  }

  const layout = document.querySelector('.layout') as HTMLElement;
  const configPanel = document.getElementById('configPanel') as HTMLElement;
  const codeSection = document.querySelector('.code-section') as HTMLElement;
  const editorContainer = document.getElementById('editor') as HTMLElement;
  const tabContainer = document.getElementById('fileTabs') as HTMLElement;
  const tryItSection = document.getElementById('tryItSection') as HTMLElement;
  const tryItBtn = document.getElementById('tryItBtn') as HTMLButtonElement;
  const closeTryIt = document.getElementById('closeTryIt') as HTMLButtonElement;
  const tryItFrame = document.getElementById('tryItFrame') as HTMLIFrameElement;
  const themeToggle = document.getElementById('themeToggle') as HTMLButtonElement;
  const copyBtn = document.getElementById('copyBtn') as HTMLButtonElement;
  const downloadZipBtn = document.getElementById('downloadZipBtn') as HTMLButtonElement;
  const sidebarToggle = document.getElementById('sidebarToggle') as HTMLButtonElement;
  const codeFullscreenBtn = document.getElementById('codeFullscreenBtn') as HTMLButtonElement;
  const previewFullscreenBtn = document.getElementById('previewFullscreenBtn') as HTMLButtonElement;

  setupThemeToggle(themeToggle, (next) => {
    if (latestConfigValues) {
      currentFiles = generateCode(latestConfigValues, next);
      updateCodePreview(currentFiles, tabContainer);
      updateUrlParams(latestConfigValues, next);
      if (!tryItSection.classList.contains('is-closed') && canTryIt(latestConfigValues.framework)) {
        runInIframe(tryItFrame, currentFiles);
      }
    }
  });
  setupCopyButton(copyBtn);
  setupDownloadZip(downloadZipBtn);
  setupSidebarToggle(sidebarToggle, layout);
  setupFullscreenToggle(codeFullscreenBtn, codeSection);
  setupFullscreenToggle(previewFullscreenBtn, tryItSection);
  setupEscapeKey(codeSection, tryItSection);

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
    latestConfigValues = values;
    currentFiles = generateCode(values, getPageTheme());
    updateCodePreview(currentFiles, tabContainer);
    updateUrlParams(values, getPageTheme());

    // Auto-run preview if it's open
    if (!tryItSection.classList.contains('is-closed')) {
      if (canTryIt(values.framework)) {
        runInIframe(tryItFrame, currentFiles);
      }
    }
  });
}

init();
