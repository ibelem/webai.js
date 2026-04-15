/**
 * Config panel: dropdown-based configuration mirroring CLI flags.
 *
 * Renders <select> elements for: task, engine, backend, framework, input, lang, theme.
 * Also has a text input for model name.
 * Fires onChange callback whenever any value changes.
 */

import { TASK_PROFILES, type TaskType, type InputMode } from '@webai/core';
import type { Engine, Backend, Framework, OutputLang, Theme } from '@webai/core';
import { setupHfPicker, type HfPickerResult } from './hf-picker.js';

export interface ConfigValues {
  task: TaskType;
  engine: Engine;
  backend: Backend;
  framework: Framework;
  input: InputMode;
  lang: OutputLang;
  theme: Theme;
  modelName: string;
  offline: boolean;
  modelUrl?: string;
  modelSource: 'local-path' | 'hf-model-id' | 'url';
}

interface SelectOption {
  value: string;
  label: string;
}

const TASKS: SelectOption[] = Object.entries(TASK_PROFILES).map(([key, profile]) => ({
  value: key,
  label: profile.label,
}));

const ENGINES: SelectOption[] = [
  { value: 'ort', label: 'ORT Web' },
  { value: 'litert', label: 'LiteRT.js' },
  { value: 'webnn', label: 'WebNN API' },
];

const BACKENDS: SelectOption[] = [
  { value: 'auto', label: 'Auto (WebNN NPU > GPU > WebGPU > WASM)' },
  { value: 'wasm', label: 'WASM' },
  { value: 'webgpu', label: 'WebGPU' },
  { value: 'webnn-cpu', label: 'WebNN CPU' },
  { value: 'webnn-gpu', label: 'WebNN GPU' },
  { value: 'webnn-npu', label: 'WebNN NPU' },
];

const FRAMEWORKS: SelectOption[] = [
  { value: 'html', label: 'HTML (single file)' },
  { value: 'vanilla-vite', label: 'Vanilla + Vite' },
  { value: 'react-vite', label: 'React + Vite' },
  { value: 'nextjs', label: 'Next.js' },
  { value: 'svelte-vite', label: 'Svelte + Vite' },
  { value: 'sveltekit', label: 'SvelteKit' },
  { value: 'vue-vite', label: 'Vue + Vite' },
  { value: 'nuxt', label: 'Nuxt' },
  { value: 'astro', label: 'Astro' },
];

const LANGS: SelectOption[] = [
  { value: 'js', label: 'JavaScript' },
  { value: 'ts', label: 'TypeScript' },
];

const THEMES: SelectOption[] = [
  { value: 'dark', label: 'Dark' },
  { value: 'light', label: 'Light' },
];

function createSelect(
  id: string,
  label: string,
  options: SelectOption[],
  defaultValue: string,
): HTMLElement {
  const wrapper = document.createElement('div');

  const lbl = document.createElement('label');
  lbl.htmlFor = id;
  lbl.textContent = label;
  wrapper.appendChild(lbl);

  const select = document.createElement('select');
  select.id = id;
  select.name = id;
  for (const opt of options) {
    const option = document.createElement('option');
    option.value = opt.value;
    option.textContent = opt.label;
    if (opt.value === defaultValue) option.selected = true;
    select.appendChild(option);
  }
  wrapper.appendChild(select);

  return wrapper;
}

function createTextInput(id: string, label: string, defaultValue: string): HTMLElement {
  const wrapper = document.createElement('div');

  const lbl = document.createElement('label');
  lbl.htmlFor = id;
  lbl.textContent = label;
  wrapper.appendChild(lbl);

  const input = document.createElement('input');
  input.type = 'text';
  input.id = id;
  input.name = id;
  input.value = defaultValue;
  input.spellcheck = false;
  wrapper.appendChild(input);

  return wrapper;
}

function createCheckbox(id: string, label: string, defaultChecked: boolean): HTMLElement {
  const wrapper = document.createElement('div');
  wrapper.style.display = 'flex';
  wrapper.style.alignItems = 'center';
  wrapper.style.gap = '6px';
  wrapper.style.marginTop = '12px';

  const input = document.createElement('input');
  input.type = 'checkbox';
  input.id = id;
  input.name = id;
  input.checked = defaultChecked;
  wrapper.appendChild(input);

  const lbl = document.createElement('label');
  lbl.htmlFor = id;
  lbl.textContent = label;
  lbl.style.marginTop = '0';
  wrapper.appendChild(lbl);

  return wrapper;
}

export function getInputOptions(task: TaskType): SelectOption[] {
  const profile = TASK_PROFILES[task];
  if (!profile) return [{ value: 'file', label: 'File' }];
  return profile.supportedInputs.map((input) => ({
    value: input,
    label: input.charAt(0).toUpperCase() + input.slice(1),
  }));
}

export function setupConfigPanel(
  container: HTMLElement,
  onChange: (values: ConfigValues) => void,
): void {
  const defaultTask: TaskType = 'image-classification';

  container.appendChild(createTextInput('modelName', 'Model Name or HuggingFace ID', 'mobilenet'));

  // HF picker container (rendered below model name input)
  const hfPickerContainer = document.createElement('div');
  hfPickerContainer.id = 'hfPicker';
  hfPickerContainer.hidden = true;
  container.appendChild(hfPickerContainer);

  container.appendChild(createSelect('task', 'Task', TASKS, defaultTask));
  container.appendChild(createSelect('engine', 'Engine', ENGINES, 'ort'));
  container.appendChild(createSelect('backend', 'Backend', BACKENDS, 'auto'));
  container.appendChild(createSelect('framework', 'Framework', FRAMEWORKS, 'html'));
  container.appendChild(createSelect('input', 'Input Mode', getInputOptions(defaultTask), 'file'));
  container.appendChild(createSelect('lang', 'Language', LANGS, 'js'));
  container.appendChild(createSelect('theme', 'Generated UI Theme', THEMES, 'dark'));
  container.appendChild(createCheckbox('offline', 'Offline (OPFS cache)', false));

  // Track HF picker state
  let hfResult: HfPickerResult | null = null;

  const taskSelect = container.querySelector('#task') as HTMLSelectElement;
  const inputSelect = container.querySelector('#input') as HTMLSelectElement;
  const modelInput = container.querySelector('#modelName') as HTMLInputElement;

  function updateInputOptions(): void {
    const task = taskSelect.value as TaskType;
    const options = getInputOptions(task);
    inputSelect.innerHTML = '';
    for (const opt of options) {
      const option = document.createElement('option');
      option.value = opt.value;
      option.textContent = opt.label;
      inputSelect.appendChild(option);
    }
  }

  function getValues(): ConfigValues {
    const values: ConfigValues = {
      task: taskSelect.value as TaskType,
      engine: (container.querySelector('#engine') as HTMLSelectElement).value as Engine,
      backend: (container.querySelector('#backend') as HTMLSelectElement).value as Backend,
      framework: (container.querySelector('#framework') as HTMLSelectElement).value as Framework,
      input: inputSelect.value as InputMode,
      lang: (container.querySelector('#lang') as HTMLSelectElement).value as OutputLang,
      theme: (container.querySelector('#theme') as HTMLSelectElement).value as Theme,
      modelName: modelInput.value || 'model',
      offline: (container.querySelector('#offline') as HTMLInputElement).checked,
      modelSource: hfResult ? hfResult.modelSource : 'local-path',
    };
    if (hfResult) {
      values.modelUrl = hfResult.url;
    }
    return values;
  }

  // Wire up HF picker
  setupHfPicker(
    modelInput,
    hfPickerContainer,
    () => (container.querySelector('#engine') as HTMLSelectElement).value === 'litert',
    (result) => {
      hfResult = result;
      // Auto-set task from HF pipeline_tag if available
      if (result?.pipelineTag) {
        const hfTaskMap: Record<string, string> = {
          'image-classification': 'image-classification',
          'object-detection': 'object-detection',
          'image-segmentation': 'image-segmentation',
          'feature-extraction': 'feature-extraction',
          'automatic-speech-recognition': 'speech-to-text',
          'audio-classification': 'audio-classification',
          'text-to-speech': 'text-to-speech',
          'text-classification': 'text-classification',
          'text-generation': 'text-generation',
          'zero-shot-classification': 'zero-shot-classification',
          'fill-mask': 'fill-mask',
          'sentence-similarity': 'sentence-similarity',
          'depth-estimation': 'depth-estimation',
          'token-classification': 'token-classification',
          'question-answering': 'question-answering',
          'summarization': 'summarization',
          'translation': 'translation',
          'image-to-text': 'image-to-text',
          'audio-to-audio': 'audio-to-audio',
          'speaker-diarization': 'speaker-diarization',
          'voice-activity-detection': 'voice-activity-detection',
          'text2text-generation': 'text2text-generation',
          'conversational': 'conversational',
          'table-question-answering': 'table-question-answering',
          'visual-question-answering': 'visual-question-answering',
          'document-question-answering': 'document-question-answering',
          'image-text-to-text': 'image-text-to-text',
        };
        const mapped = hfTaskMap[result.pipelineTag];
        if (mapped && taskSelect.value !== mapped) {
          taskSelect.value = mapped;
          updateInputOptions();
        }
      }
      onChange(getValues());
    },
  );

  // Listen for changes on all inputs
  container.addEventListener('change', () => {
    if (document.activeElement === taskSelect) {
      updateInputOptions();
    }
    onChange(getValues());
  });

  container.addEventListener('input', (e) => {
    if ((e.target as HTMLElement).id === 'modelName') {
      onChange(getValues());
    }
  });

  // Fire initial
  onChange(getValues());
}
