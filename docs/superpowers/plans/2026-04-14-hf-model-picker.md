# HuggingFace Model File Picker — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a user types a HuggingFace model ID (e.g., `Xenova/yolov8n`) into the Web UI's model name field, fetch the repo's file list from the HF API and show a dropdown so the user can pick which .onnx/.tflite file to use for code generation.

**Architecture:** New `packages/web/src/hf-picker.ts` module handles the entire flow: detect HF model IDs in the input field, fetch the HF API, render a file-select dropdown, and report the chosen file URL back to main.ts. The config panel's `ConfigValues` gains an optional `modelUrl` field so the assembler generates code that references the HF URL directly. Reuses `classifyModelInput`, `buildHfApiUrl`, `buildHfFileUrl`, and `pickBestModelFile` from `@webai/core` — no duplication.

**Tech Stack:** Vanilla TypeScript DOM, HuggingFace API (`https://huggingface.co/api/models/{id}`), existing `@webai/core` model-source utilities.

**CRITICAL:** Do NOT git commit or git push. Only make file edits and run tests. The user handles all git operations.

---

## Design Decisions

1. **Detect HF model IDs using the existing `classifyModelInput` function.** It returns `'hf-model-id'` for strings matching the `owner/repo` pattern (no dots, exactly one slash). We call it on every keystroke (debounced) in the model name input.

2. **Show the model file picker inline below the model name input.** When an HF ID is detected, a small UI appears: a loading spinner → then a `<select>` of `.onnx`/`.tflite` files with sizes. The picker disappears when the input changes to a non-HF value.

3. **Pre-select the best file using `pickBestModelFile`.** The existing heuristic (shallowest path, shortest name) picks a sensible default. The user can override via the dropdown.

4. **Pass `modelUrl` through to the assembler.** When the user picks a file, `ConfigValues` gets `modelSource: 'hf-model-id'` and `modelUrl` set to the direct download URL. The generated code then uses the URL directly (e.g., `const MODEL_PATH = 'https://huggingface.co/...'`).

5. **Also populate task auto-detection from HF API.** The API response includes `pipeline_tag` which maps to task types. When a user picks a model, we can auto-set the task dropdown.

6. **Cache API responses.** Same HF ID typed twice shouldn't re-fetch. Simple in-memory `Map<string, HfModelInfo>`.

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `packages/web/src/hf-picker.ts` | Detect HF IDs, fetch API, render file dropdown, report selection |

### Modified Files

| File | Changes |
|------|---------|
| `packages/web/src/config-panel.ts` | Add `modelUrl` and `modelSource` to `ConfigValues`, integrate HF picker below model name input |
| `packages/web/src/main.ts` | Pass `modelUrl`/`modelSource` from config values into `CliFlags` |
| `packages/web/src/style.css` | CSS for HF picker (loading state, file dropdown, file size badges) |

---

## Task 1: HF Picker Module

**Files:**
- Create: `packages/web/src/hf-picker.ts`

This module does the heavy lifting: fetches HF API, renders a file selector, fires callbacks.

- [ ] **Step 1: Create `packages/web/src/hf-picker.ts`**

```typescript
/**
 * HuggingFace model file picker.
 *
 * When the user types a HuggingFace model ID (e.g., "Xenova/yolov8n"),
 * fetches the repo's file list and shows a dropdown to pick which
 * .onnx/.tflite file to use.
 */

import {
  classifyModelInput,
  buildHfApiUrl,
  buildHfFileUrl,
  pickBestModelFile,
} from '@webai/core';

/** File info from HF API */
export interface HfFileInfo {
  filename: string;
  size: number | null;
  url: string;
}

/** Result of picking a model file */
export interface HfPickerResult {
  modelId: string;
  filename: string;
  url: string;
  pipelineTag: string | null;
}

/** Cached API responses */
const apiCache = new Map<string, { files: HfFileInfo[]; pipelineTag: string | null }>();

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Fetch model file list from HuggingFace API.
 * Returns only .onnx and .tflite files.
 */
async function fetchModelFiles(modelId: string): Promise<{ files: HfFileInfo[]; pipelineTag: string | null }> {
  const cached = apiCache.get(modelId);
  if (cached) return cached;

  const apiUrl = buildHfApiUrl(modelId);
  const response = await fetch(apiUrl);

  if (!response.ok) {
    if (response.status === 404) {
      throw new Error(`Model not found: "${modelId}"`);
    }
    throw new Error(`HuggingFace API error: HTTP ${response.status}`);
  }

  const data = await response.json() as {
    siblings?: Array<{ rfilename: string; size?: number }>;
    pipeline_tag?: string;
  };

  if (!data.siblings || !Array.isArray(data.siblings)) {
    throw new Error(`No files found for "${modelId}"`);
  }

  const modelExts = ['.onnx', '.tflite'];
  const files: HfFileInfo[] = data.siblings
    .filter((s) => modelExts.some((ext) => s.rfilename.toLowerCase().endsWith(ext)))
    .map((s) => ({
      filename: s.rfilename,
      size: s.size ?? null,
      url: buildHfFileUrl(modelId, s.rfilename),
    }));

  const result = { files, pipelineTag: data.pipeline_tag ?? null };
  apiCache.set(modelId, result);
  return result;
}

/**
 * Set up the HF picker on a model name input.
 *
 * @param input - The model name text input element
 * @param container - Container element to render the picker into (below the input)
 * @param preferTflite - Whether to prefer .tflite files (when engine is litert)
 * @param onSelect - Callback when a file is selected (or null when deselected)
 */
export function setupHfPicker(
  input: HTMLInputElement,
  container: HTMLElement,
  getPreferTflite: () => boolean,
  onSelect: (result: HfPickerResult | null) => void,
): void {
  let debounceTimer: ReturnType<typeof setTimeout> | null = null;
  let currentModelId = '';
  let abortController: AbortController | null = null;

  function render(state: 'idle' | 'loading' | 'error' | 'files', data?: {
    files?: HfFileInfo[];
    pipelineTag?: string | null;
    bestFile?: string | null;
    error?: string;
    modelId?: string;
  }): void {
    container.innerHTML = '';

    if (state === 'idle') {
      container.hidden = true;
      return;
    }

    container.hidden = false;

    if (state === 'loading') {
      const el = document.createElement('div');
      el.className = 'hf-picker-status';
      el.textContent = 'Loading files...';
      container.appendChild(el);
      return;
    }

    if (state === 'error') {
      const el = document.createElement('div');
      el.className = 'hf-picker-status hf-picker-error';
      el.textContent = data?.error ?? 'Error loading model';
      container.appendChild(el);
      return;
    }

    if (state === 'files' && data?.files) {
      if (data.files.length === 0) {
        const el = document.createElement('div');
        el.className = 'hf-picker-status hf-picker-error';
        el.textContent = 'No .onnx or .tflite files found in this repo';
        container.appendChild(el);
        onSelect(null);
        return;
      }

      const wrapper = document.createElement('div');
      wrapper.className = 'hf-picker-files';

      const lbl = document.createElement('label');
      lbl.htmlFor = 'hfFile';
      lbl.textContent = 'Model File';
      wrapper.appendChild(lbl);

      const select = document.createElement('select');
      select.id = 'hfFile';
      for (const file of data.files) {
        const option = document.createElement('option');
        option.value = file.filename;
        const sizeStr = file.size !== null ? ` (${formatFileSize(file.size)})` : '';
        option.textContent = `${file.filename}${sizeStr}`;
        if (file.filename === data.bestFile) option.selected = true;
        select.appendChild(option);
      }
      wrapper.appendChild(select);
      container.appendChild(wrapper);

      // Fire initial selection
      const fireSelect = (): void => {
        const selectedFile = data.files!.find((f) => f.filename === select.value);
        if (selectedFile && data.modelId) {
          onSelect({
            modelId: data.modelId,
            filename: selectedFile.filename,
            url: selectedFile.url,
            pipelineTag: data.pipelineTag ?? null,
          });
        }
      };

      select.addEventListener('change', fireSelect);
      fireSelect();
    }
  }

  async function checkInput(): Promise<void> {
    const value = input.value.trim();
    const sourceType = classifyModelInput(value);

    if (sourceType !== 'hf-model-id') {
      currentModelId = '';
      render('idle');
      onSelect(null);
      return;
    }

    if (value === currentModelId) return;
    currentModelId = value;

    // Cancel any in-flight request
    if (abortController) abortController.abort();
    abortController = new AbortController();

    render('loading');

    try {
      const { files, pipelineTag } = await fetchModelFiles(value);
      // Check if input changed while we were fetching
      if (input.value.trim() !== value) return;

      const siblings = files.map((f) => ({ rfilename: f.filename }));
      const bestFile = pickBestModelFile(siblings, getPreferTflite());

      render('files', { files, pipelineTag, bestFile, modelId: value });
    } catch (e) {
      if (input.value.trim() !== value) return;
      render('error', { error: e instanceof Error ? e.message : String(e) });
      onSelect(null);
    }
  }

  input.addEventListener('input', () => {
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(checkInput, 400);
  });
}
```

- [ ] **Step 2: Verify the file was created and imports resolve**

Run: `cd C:/Users/Min/github/webai.js/packages/web && npx vite build 2>&1`
Expected: Build succeeds (the new module isn't imported yet, but shouldn't break anything)

---

## Task 2: Integrate HF Picker into Config Panel

**Files:**
- Modify: `packages/web/src/config-panel.ts:1-208`

Add `modelUrl` and `modelSource` to `ConfigValues`. After rendering the model name input, create a container div for the HF picker and call `setupHfPicker`. When a file is selected, update the config values and fire the onChange callback.

- [ ] **Step 1: Update `ConfigValues` interface**

In `packages/web/src/config-panel.ts`, add two fields to the `ConfigValues` interface:

```typescript
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
  modelSource: 'local-path' | 'hf-model-id';
}
```

- [ ] **Step 2: Import setupHfPicker and ModelSourceType**

Add to imports at top of `config-panel.ts`:

```typescript
import { setupHfPicker, type HfPickerResult } from './hf-picker.js';
import type { ModelSourceType } from '@webai/core';
```

Note: `ModelSourceType` is already exported from `@webai/core` (see `packages/core/src/index.ts` line 53).

- [ ] **Step 3: Update `setupConfigPanel` to add the HF picker container and wire it up**

After the `createTextInput('modelName', ...)` call in `setupConfigPanel`, insert a container div for the picker. Track the current HF selection in a local variable. Update `getValues()` to include `modelUrl` and `modelSource`.

The full updated `setupConfigPanel` function:

```typescript
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
      modelSource: hfResult ? 'hf-model-id' : 'local-path',
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
      // Don't fire onChange here for HF IDs — the picker's onSelect handles it
      // But still fire for regular model names (non-HF)
      onChange(getValues());
    }
  });

  // Fire initial
  onChange(getValues());
}
```

- [ ] **Step 4: Verify build passes**

Run: `cd C:/Users/Min/github/webai.js/packages/web && npx vite build 2>&1`
Expected: Build succeeds

---

## Task 3: Update main.ts to Pass modelUrl/modelSource

**Files:**
- Modify: `packages/web/src/main.ts:1-77`

The `generateCode` function currently hardcodes `modelSource: 'local-path'`. Update it to use the values from the config panel.

- [ ] **Step 1: Update `generateCode` in main.ts**

Change the `generateCode` function to use `values.modelSource` and `values.modelUrl`:

```typescript
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
```

Key changes:
- `model` uses `values.modelUrl` when available (HF file URL), falls back to local path
- `modelSource` comes from config values instead of hardcoded `'local-path'`
- `modelUrl` passed through
- File extension respects engine selection (`.tflite` for litert)

- [ ] **Step 2: Verify build and existing tests pass**

Run: `cd C:/Users/Min/github/webai.js/packages/web && npx vite build 2>&1`
Expected: Build succeeds

Run: `cd C:/Users/Min/github/webai.js && npx vitest run 2>&1 | tail -5`
Expected: All 445 tests pass

---

## Task 4: CSS for HF Picker

**Files:**
- Modify: `packages/web/src/style.css`

- [ ] **Step 1: Add HF picker styles to the end of style.css (before the responsive media query)**

Insert before the `/* Responsive */` comment:

```css
/* HuggingFace picker */
#hfPicker {
  margin-top: 4px;
}

.hf-picker-status {
  font-size: 12px;
  color: var(--text-muted);
  padding: 4px 0;
}

.hf-picker-error {
  color: #ef4444;
}

.hf-picker-files label {
  display: block;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
  margin-bottom: 4px;
  margin-top: 8px;
}

.hf-picker-files select {
  width: 100%;
  padding: 6px 8px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 4px;
  color: var(--text);
  font-size: 12px;
  font-family: var(--font-mono);
}

.hf-picker-files select:focus {
  outline: 2px solid var(--accent);
  outline-offset: -1px;
}
```

- [ ] **Step 2: Verify build succeeds**

Run: `cd C:/Users/Min/github/webai.js/packages/web && npx vite build 2>&1`
Expected: Build succeeds

---

## Task 5: Integration Test

**Files:**
- No new files

- [ ] **Step 1: Run existing tests to confirm no regressions**

Run: `cd C:/Users/Min/github/webai.js && npx vitest run 2>&1 | tail -5`
Expected: All 445 tests pass

- [ ] **Step 2: Verify production build**

Run: `cd C:/Users/Min/github/webai.js/packages/web && npx vite build 2>&1`
Expected: Builds without errors

- [ ] **Step 3: Start dev server and test manually**

Run: `cd C:/Users/Min/github/webai.js/packages/web && npx vite`

Test these scenarios:
1. Type `mobilenet` in model name → no HF picker shown, code generates normally
2. Type `Xenova/yolov8n` → "Loading files..." appears, then a file dropdown with .onnx files
3. Pick a different file from the dropdown → code regenerates with new model URL
4. The generated code should show `const MODEL_PATH = 'https://huggingface.co/Xenova/yolov8n/resolve/main/...'` instead of `./model.onnx`
5. Task dropdown auto-updates based on HF `pipeline_tag`
6. Clear the input back to `mobilenet` → HF picker disappears, code reverts to local path
7. Type a non-existent model like `nonexistent/doesntexist123` → error message "Model not found"
8. Type the same HF model ID again → instant response (cached)

- [ ] **Step 4: Fix any issues found during testing**

Apply fixes as needed.

---

## Notes for Implementers

### HuggingFace API Response Shape

`GET https://huggingface.co/api/models/{owner}/{repo}` returns:

```json
{
  "pipeline_tag": "image-classification",
  "siblings": [
    { "rfilename": "model.onnx", "size": 12345678 },
    { "rfilename": "model_quantized.onnx", "size": 3456789 },
    { "rfilename": "config.json", "size": 1234 }
  ]
}
```

The `siblings` array lists all files. We filter for `.onnx`/`.tflite`. The `size` field is in bytes.

The `pipeline_tag` maps to task types. Common mappings:
- `image-classification` → `image-classification`
- `object-detection` → `object-detection`
- `automatic-speech-recognition` → `speech-to-text`
- `text-classification` → `text-classification`
- `text-generation` → `text-generation`

### CORS

The HuggingFace API at `huggingface.co/api/models/*` supports CORS from any origin. No proxy needed.

### classifyModelInput

Already exists in `@webai/core`. Returns `'hf-model-id'` for strings matching `owner/repo` pattern (no dots, exactly one slash, alphanumeric + hyphens). Examples:
- `"Xenova/yolov8n"` → `'hf-model-id'`
- `"mobilenet"` → `'local-path'` (no slash)
- `"./model.onnx"` → `'local-path'` (has dot)
- `"https://huggingface.co/..."` → `'url'` (has protocol)

### Debounce

The picker debounces API calls by 400ms to avoid hammering HF API while the user types. If the input changes before the API responds, the stale response is discarded.
