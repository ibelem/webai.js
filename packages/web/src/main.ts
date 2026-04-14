/**
 * webai.js Web UI — main entry point.
 *
 * Wires up: config panel → assembler → Monaco preview → Try it iframe.
 */

import './style.css';
import { resolveConfig } from '@webai/core';
import type { CliFlags } from '@webai/core';
import { assemble } from 'webai';
import type { GeneratedFile } from 'webai';
import { setupConfigPanel, type ConfigValues } from './config-panel.js';
import { setupCodePreview, updateCodePreview } from './code-preview.js';
import { setupTryIt } from './try-it.js';
import { createMockMetadata } from './mock-metadata.js';

let currentFramework = 'html';
let currentFiles: GeneratedFile[] = [];

function generateCode(values: ConfigValues): GeneratedFile[] {
  const metadata = createMockMetadata(values.task, values.engine === 'litert' ? 'tflite' : 'onnx');

  const flags: CliFlags = {
    model: `./${values.modelName}.onnx`,
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
    modelSource: 'local-path',
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

async function init(): Promise<void> {
  const configPanel = document.getElementById('configPanel')!;
  const editorContainer = document.getElementById('editor')!;
  const tabContainer = document.getElementById('fileTabs')!;
  const tryItSection = document.getElementById('tryItSection')!;
  const tryItBtn = document.getElementById('tryItBtn') as HTMLButtonElement;
  const closeTryIt = document.getElementById('closeTryIt') as HTMLButtonElement;
  const tryItFrame = document.getElementById('tryItFrame') as HTMLIFrameElement;

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
