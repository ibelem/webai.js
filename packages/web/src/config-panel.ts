/**
 * Config panel: dropdown-based configuration mirroring CLI flags.
 *
 * Renders <select> elements for: task, engine, backend, framework, input, lang, theme.
 * Also has a text input for model name.
 * Fires onChange callback whenever any value changes.
 * Supports URL parameter sync — reads initial values from URL, writes back on change.
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
  tooltip?: string;
}

// ---- Option lists ----

const TASKS: SelectOption[] = Object.entries(TASK_PROFILES)
  .map(([key, profile]) => ({ value: key, label: profile.label }))
  .sort((a, b) => a.label.localeCompare(b.label));

const ENGINES: SelectOption[] = [
  { value: 'ort', label: 'ORT Web' },
  { value: 'litert', label: 'LiteRT.js' },
  { value: 'webnn', label: 'WebNN API' },
];

const BACKENDS: SelectOption[] = [
  { value: 'auto', label: 'Auto', tooltip: 'Auto (WebNN NPU > WebNN GPU > WebGPU > Wasm)' },
  { value: 'webnn-npu', label: 'WebNN NPU' },
  { value: 'webnn-gpu', label: 'WebNN GPU' },
  { value: 'webnn-cpu', label: 'WebNN CPU' },
  { value: 'webgpu', label: 'WebGPU' },
  { value: 'wasm', label: 'Wasm' },
];

const FRAMEWORKS: SelectOption[] = [
  { value: 'astro', label: 'Astro' },
  { value: 'html', label: 'HTML (single file)' },
  { value: 'nextjs', label: 'Next.js' },
  { value: 'nuxt', label: 'Nuxt' },
  { value: 'react-vite', label: 'React + Vite' },
  { value: 'svelte-vite', label: 'Svelte + Vite' },
  { value: 'sveltekit', label: 'SvelteKit' },
  { value: 'vanilla-vite', label: 'Vanilla + Vite' },
  { value: 'vue-vite', label: 'Vue + Vite' },
];

const LANGS: SelectOption[] = [
  { value: 'js', label: 'JavaScript' },
  { value: 'ts', label: 'TypeScript' },
];

const THEMES: SelectOption[] = [
  { value: 'dark', label: 'Dark' },
  { value: 'light', label: 'Light' },
];

// ---- SVG Icons ----

const FRAMEWORK_ICONS: Record<string, string> = {
  'astro': '<svg viewBox="0 0 24 24" width="14" height="14" fill="none"><path d="M8.36 2.17a.5.5 0 0 1 .54.05l7.13 5.7a.5.5 0 0 1-.16.88L8.5 11.58a.5.5 0 0 1-.63-.63l.3-8.38a.5.5 0 0 1 .19-.4Z" fill="#BC52EE"/><path d="M15.64 2.17a.5.5 0 0 0-.54.05l-7.13 5.7a.5.5 0 0 0 .16.88L15.5 11.58a.5.5 0 0 0 .63-.63l-.3-8.38a.5.5 0 0 0-.19-.4Z" fill="#BC52EE"/><path d="M8 16c0 2.5 1.79 4 4 4s4-1.5 4-4c0-1.5-.5-2.5-1.5-3.5L12 10l-2.5 2.5C8.5 13.5 8 14.5 8 16Z" fill="#FF5D01"/></svg>',
  'html': '<svg viewBox="0 0 24 24" width="14" height="14"><path d="M4.14 3l1.59 17.84L12 22.36l6.27-1.52L19.86 3H4.14ZM17.3 7.73H8.55l.2 2.24h8.36l-.63 7.04-4.48 1.24-4.49-1.24-.3-3.44h2.19l.15 1.75 2.45.66 2.45-.66.26-2.84H7.83L7.2 5.5h9.54l-.44 2.23Z" fill="#E34F26"/></svg>',
  'nextjs': '<svg viewBox="0 0 24 24" width="14" height="14"><circle cx="12" cy="12" r="10" fill="currentColor"/><path d="M9.5 8v8M9.5 8l8 10.5" stroke="var(--bg)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" fill="none"/><circle cx="15.5" cy="8" r="1" fill="var(--bg)"/></svg>',
  'nuxt': '<svg viewBox="0 0 24 24" width="14" height="14"><path d="M13.5 19H21l-5.25-9.5L13.5 13.9 10.65 9l-6.88 10H13.5Z" fill="#00DC82"/><path d="M13.5 19L10.65 9l-6.88 10H13.5Z" fill="#00C16A"/></svg>',
  'react-vite': '<svg viewBox="-11 -10 22 20" width="14" height="14"><circle r="2" fill="#61DAFB"/><g stroke="#61DAFB" fill="none" stroke-width="1"><ellipse rx="10" ry="4"/><ellipse rx="10" ry="4" transform="rotate(60)"/><ellipse rx="10" ry="4" transform="rotate(120)"/></g></svg>',
  'svelte-vite': '<svg viewBox="0 0 24 24" width="14" height="14"><path d="M20.28 4.74a6.2 6.2 0 0 0-8.51-1.66L7.2 6.44a5.44 5.44 0 0 0-2.47 3.64 5.7 5.7 0 0 0 .56 3.78 5.36 5.36 0 0 0-.83 2.04 5.77 5.77 0 0 0 .99 4.36 6.2 6.2 0 0 0 8.51 1.66l4.57-3.36a5.44 5.44 0 0 0 2.47-3.64 5.7 5.7 0 0 0-.56-3.78 5.36 5.36 0 0 0 .83-2.04 5.77 5.77 0 0 0-.99-4.36Z" fill="#FF3E00"/><path d="M10.06 19.93a3.82 3.82 0 0 1-4.1-1.53 3.56 3.56 0 0 1-.61-2.69 3.34 3.34 0 0 1 .14-.63l.15-.38.34.24a6.4 6.4 0 0 0 1.92 1.03l.18.05-.02.18a1.08 1.08 0 0 0 .2.82 1.16 1.16 0 0 0 1.25.47 1.07 1.07 0 0 0 .3-.13l4.57-3.36a1 1 0 0 0 .45-.66 1.1 1.1 0 0 0-.19-.82 1.16 1.16 0 0 0-1.25-.47 1.07 1.07 0 0 0-.3.13l-1.75 1.28a3.46 3.46 0 0 1-.97.43 3.82 3.82 0 0 1-4.1-1.53 3.56 3.56 0 0 1-.61-2.69 3.32 3.32 0 0 1 1.5-2.17l4.57-3.36a3.46 3.46 0 0 1 .97-.43 3.82 3.82 0 0 1 4.1 1.53 3.56 3.56 0 0 1 .61 2.69 3.34 3.34 0 0 1-.14.63l-.15.38-.34-.24a6.4 6.4 0 0 0-1.92-1.03l-.18-.05.02-.18a1.08 1.08 0 0 0-.2-.82 1.16 1.16 0 0 0-1.25-.47 1.07 1.07 0 0 0-.3.13l-4.57 3.36a1 1 0 0 0-.45.66 1.1 1.1 0 0 0 .19.82 1.16 1.16 0 0 0 1.25.47 1.07 1.07 0 0 0 .3-.13l1.75-1.28a3.46 3.46 0 0 1 .97-.43 3.82 3.82 0 0 1 4.1 1.53 3.56 3.56 0 0 1 .61 2.69 3.32 3.32 0 0 1-1.5 2.17l-4.57 3.36a3.46 3.46 0 0 1-.97.43Z" fill="#fff"/></svg>',
  'sveltekit': '<svg viewBox="0 0 24 24" width="14" height="14"><path d="M20.28 4.74a6.2 6.2 0 0 0-8.51-1.66L7.2 6.44a5.44 5.44 0 0 0-2.47 3.64 5.7 5.7 0 0 0 .56 3.78 5.36 5.36 0 0 0-.83 2.04 5.77 5.77 0 0 0 .99 4.36 6.2 6.2 0 0 0 8.51 1.66l4.57-3.36a5.44 5.44 0 0 0 2.47-3.64 5.7 5.7 0 0 0-.56-3.78 5.36 5.36 0 0 0 .83-2.04 5.77 5.77 0 0 0-.99-4.36Z" fill="#FF3E00"/><path d="M10.06 19.93a3.82 3.82 0 0 1-4.1-1.53 3.56 3.56 0 0 1-.61-2.69 3.34 3.34 0 0 1 .14-.63l.15-.38.34.24a6.4 6.4 0 0 0 1.92 1.03l.18.05-.02.18a1.08 1.08 0 0 0 .2.82 1.16 1.16 0 0 0 1.25.47 1.07 1.07 0 0 0 .3-.13l4.57-3.36a1 1 0 0 0 .45-.66 1.1 1.1 0 0 0-.19-.82 1.16 1.16 0 0 0-1.25-.47 1.07 1.07 0 0 0-.3.13l-1.75 1.28a3.46 3.46 0 0 1-.97.43 3.82 3.82 0 0 1-4.1-1.53 3.56 3.56 0 0 1-.61-2.69 3.32 3.32 0 0 1 1.5-2.17l4.57-3.36a3.46 3.46 0 0 1 .97-.43 3.82 3.82 0 0 1 4.1 1.53 3.56 3.56 0 0 1 .61 2.69 3.34 3.34 0 0 1-.14.63l-.15.38-.34-.24a6.4 6.4 0 0 0-1.92-1.03l-.18-.05.02-.18a1.08 1.08 0 0 0-.2-.82 1.16 1.16 0 0 0-1.25-.47 1.07 1.07 0 0 0-.3.13l-4.57 3.36a1 1 0 0 0-.45.66 1.1 1.1 0 0 0 .19.82 1.16 1.16 0 0 0 1.25.47 1.07 1.07 0 0 0 .3-.13l1.75-1.28a3.46 3.46 0 0 1 .97-.43 3.82 3.82 0 0 1 4.1 1.53 3.56 3.56 0 0 1 .61 2.69 3.32 3.32 0 0 1-1.5 2.17l-4.57 3.36a3.46 3.46 0 0 1-.97.43Z" fill="#fff"/></svg>',
  'vanilla-vite': '<svg viewBox="0 0 24 24" width="14" height="14"><path d="M21.9 2.4L12.7 21.8a.5.5 0 0 1-.9 0L2.1 2.4a.5.5 0 0 1 .5-.7l8.7 1.6a.5.5 0 0 0 .2 0l8.9-1.6a.5.5 0 0 1 .5.7Z" fill="url(#vg)"/><defs><linearGradient id="vg" x1="2" y1="2" x2="14" y2="20" gradientUnits="userSpaceOnUse"><stop stop-color="#41D1FF"/><stop offset="1" stop-color="#BD34FE"/></linearGradient></defs></svg>',
  'vue-vite': '<svg viewBox="0 0 24 24" width="14" height="14"><path d="M2 3h3.63L12 14.43 18.37 3H22L12 21 2 3Z" fill="#41B883"/><path d="M7.35 3L12 11.11 16.65 3h-3.63L12 5.18 10.98 3H7.35Z" fill="#35495E"/></svg>',
};

const ENGINE_ICONS: Record<string, string> = {
  'ort': '<svg viewBox="0 0 24 24" width="14" height="14"><path d="M12 2L3 7v10l9 5 9-5V7l-9-5Zm0 2.18L18.36 7.5 12 10.82 5.64 7.5 12 4.18ZM5 9.08l6 3.32v6.52l-6-3.32V9.08Zm14 0v6.52l-6 3.32v-6.52l6-3.32Z" fill="#044DA1"/></svg>',
  'litert': '<svg viewBox="0 0 24 24" width="14" height="14"><path d="M1.3 13.4L4.4 7.3l3.1 6.1H1.3Z" fill="#FF6F00"/><path d="M6.5 13.4L12 2.5l5.5 10.9H6.5Z" fill="#FF9100"/><path d="M16.5 13.4l3.1-6.1 3.1 6.1h-6.2Z" fill="#FFC107"/><path d="M4.4 15.4h15.2v2.1H4.4v-2.1Zm2.1 3.6h11v2H6.5v-2Z" fill="#FF6F00"/></svg>',
  'webnn': '<svg viewBox="0 0 24 24" width="14" height="14"><circle cx="6" cy="6" r="2" fill="#3B82F6"/><circle cx="18" cy="6" r="2" fill="#3B82F6"/><circle cx="6" cy="18" r="2" fill="#3B82F6"/><circle cx="18" cy="18" r="2" fill="#3B82F6"/><circle cx="12" cy="12" r="2.5" fill="#3B82F6"/><line x1="6" y1="6" x2="12" y2="12" stroke="#3B82F6" stroke-width="1.2"/><line x1="18" y1="6" x2="12" y2="12" stroke="#3B82F6" stroke-width="1.2"/><line x1="6" y1="18" x2="12" y2="12" stroke="#3B82F6" stroke-width="1.2"/><line x1="18" y1="18" x2="12" y2="12" stroke="#3B82F6" stroke-width="1.2"/></svg>',
};

// ---- URL parameter helpers ----

/** URL param key → config panel element ID */
const URL_PARAM_MAP: Record<string, string> = {
  model: 'modelName',
  file: 'modelFile',
  task: 'task',
  engine: 'engine',
  backend: 'backend',
  framework: 'framework',
  input: 'input',
  lang: 'lang',
  uitheme: 'theme',
  offline: 'offline',
};

/** Config panel element ID → URL param key (reverse of above) */
const ID_TO_PARAM: Record<string, string> = {};
for (const [param, id] of Object.entries(URL_PARAM_MAP)) {
  ID_TO_PARAM[id] = param;
}

export function readUrlParams(): Record<string, string> {
  const params = new URLSearchParams(window.location.search);
  const result: Record<string, string> = {};
  for (const [key, value] of params.entries()) {
    result[key] = value;
  }
  return result;
}

export function updateUrlParams(values: ConfigValues, pageTheme: string): void {
  const params = new URLSearchParams();
  params.set('model', values.modelName);
  params.set('task', values.task);
  params.set('engine', values.engine);
  params.set('backend', values.backend);
  params.set('framework', values.framework);
  params.set('input', values.input);
  params.set('lang', values.lang);
  params.set('uitheme', values.theme);
  params.set('theme', pageTheme);
  if (values.offline) params.set('offline', 'true');
  const url = `${window.location.pathname}?${params.toString()}`;
  window.history.replaceState(null, '', url);
}

// ---- DOM helpers ----

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
    if (opt.tooltip) option.title = opt.tooltip;
    if (opt.value === defaultValue) option.selected = true;
    select.appendChild(option);
  }
  wrapper.appendChild(select);

  return wrapper;
}

function createIconSelect(
  id: string,
  label: string,
  options: SelectOption[],
  defaultValue: string,
  iconMap: Record<string, string>,
): HTMLElement {
  const wrapper = document.createElement('div');

  const lbl = document.createElement('label');
  lbl.htmlFor = id;
  lbl.textContent = label;
  wrapper.appendChild(lbl);

  const row = document.createElement('div');
  row.className = 'icon-select';

  const iconSpan = document.createElement('span');
  iconSpan.className = 'select-icon';
  iconSpan.innerHTML = iconMap[defaultValue] ?? '';
  row.appendChild(iconSpan);

  const select = document.createElement('select');
  select.id = id;
  select.name = id;
  for (const opt of options) {
    const option = document.createElement('option');
    option.value = opt.value;
    option.textContent = opt.label;
    if (opt.tooltip) option.title = opt.tooltip;
    if (opt.value === defaultValue) option.selected = true;
    select.appendChild(option);
  }
  row.appendChild(select);

  select.addEventListener('change', () => {
    iconSpan.innerHTML = iconMap[select.value] ?? '';
  });

  wrapper.appendChild(row);
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

// ---- Defaults ----

const DEFAULTS: Record<string, string> = {
  model: 'webnn/mobilenet-v2',
  task: 'image-classification',
  engine: 'ort',
  backend: 'auto',
  framework: 'html',
  input: 'file',
  lang: 'js',
  uitheme: 'dark',
  theme: 'dark',
  offline: '',
};

export function setupConfigPanel(
  container: HTMLElement,
  onChange: (values: ConfigValues) => void,
): void {
  // Read URL params and merge with defaults
  const urlParams = readUrlParams();
  const initial = { ...DEFAULTS, ...urlParams };

  const defaultTask = initial.task as TaskType;
  const defaultFramework = initial.framework;
  const defaultEngine = initial.engine;

  container.appendChild(createTextInput('modelName', 'Model Name or HuggingFace ID', initial.model));

  // HF picker container (rendered below model name input)
  const hfPickerContainer = document.createElement('div');
  hfPickerContainer.id = 'hfPicker';
  hfPickerContainer.hidden = true;
  container.appendChild(hfPickerContainer);

  container.appendChild(createSelect('task', 'Task', TASKS, defaultTask));
  container.appendChild(createIconSelect('engine', 'Engine', ENGINES, defaultEngine, ENGINE_ICONS));
  container.appendChild(createSelect('backend', 'Backend', BACKENDS, initial.backend));
  container.appendChild(createIconSelect('framework', 'Framework', FRAMEWORKS, defaultFramework, FRAMEWORK_ICONS));
  container.appendChild(createSelect('input', 'Input Mode', getInputOptions(defaultTask), initial.input));
  container.appendChild(createSelect('lang', 'Language', LANGS, initial.lang));
  container.appendChild(createSelect('theme', 'Generated UI Theme', THEMES, initial.uitheme));
  container.appendChild(createCheckbox('offline', 'Offline (OPFS cache)', initial.offline === 'true'));

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
