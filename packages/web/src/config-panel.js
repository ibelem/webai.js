/**
 * Config panel: dropdown-based configuration mirroring CLI flags.
 *
 * Renders <select> elements for: task, engine, backend, framework, input, lang, theme.
 * Also has a text input for model name.
 * Fires onChange callback whenever any value changes.
 */
import { TASK_PROFILES } from '@webai/core';
import { setupHfPicker } from './hf-picker.js';
const TASKS = Object.entries(TASK_PROFILES).map(([key, profile]) => ({
    value: key,
    label: profile.label,
}));
const ENGINES = [
    { value: 'ort', label: 'ORT Web' },
    { value: 'litert', label: 'LiteRT.js' },
    { value: 'webnn', label: 'WebNN API' },
];
const BACKENDS = [
    { value: 'auto', label: 'Auto (WebNN NPU > GPU > WebGPU > WASM)' },
    { value: 'wasm', label: 'WASM' },
    { value: 'webgpu', label: 'WebGPU' },
    { value: 'webnn-cpu', label: 'WebNN CPU' },
    { value: 'webnn-gpu', label: 'WebNN GPU' },
    { value: 'webnn-npu', label: 'WebNN NPU' },
];
const FRAMEWORKS = [
    { value: 'html', label: 'HTML (single file)' },
    { value: 'vanilla-vite', label: 'Vanilla + Vite' },
    { value: 'react-vite', label: 'React + Vite' },
    { value: 'nextjs', label: 'Next.js' },
    { value: 'sveltekit', label: 'SvelteKit' },
];
const LANGS = [
    { value: 'js', label: 'JavaScript' },
    { value: 'ts', label: 'TypeScript' },
];
const THEMES = [
    { value: 'dark', label: 'Dark' },
    { value: 'light', label: 'Light' },
];
function createSelect(id, label, options, defaultValue) {
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
        if (opt.value === defaultValue)
            option.selected = true;
        select.appendChild(option);
    }
    wrapper.appendChild(select);
    return wrapper;
}
function createTextInput(id, label, defaultValue) {
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
function createCheckbox(id, label, defaultChecked) {
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
export function getInputOptions(task) {
    const profile = TASK_PROFILES[task];
    if (!profile)
        return [{ value: 'file', label: 'File' }];
    return profile.supportedInputs.map((input) => ({
        value: input,
        label: input.charAt(0).toUpperCase() + input.slice(1),
    }));
}
export function setupConfigPanel(container, onChange) {
    const defaultTask = 'image-classification';
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
    let hfResult = null;
    const taskSelect = container.querySelector('#task');
    const inputSelect = container.querySelector('#input');
    const modelInput = container.querySelector('#modelName');
    function updateInputOptions() {
        const task = taskSelect.value;
        const options = getInputOptions(task);
        inputSelect.innerHTML = '';
        for (const opt of options) {
            const option = document.createElement('option');
            option.value = opt.value;
            option.textContent = opt.label;
            inputSelect.appendChild(option);
        }
    }
    function getValues() {
        const values = {
            task: taskSelect.value,
            engine: container.querySelector('#engine').value,
            backend: container.querySelector('#backend').value,
            framework: container.querySelector('#framework').value,
            input: inputSelect.value,
            lang: container.querySelector('#lang').value,
            theme: container.querySelector('#theme').value,
            modelName: modelInput.value || 'model',
            offline: container.querySelector('#offline').checked,
            modelSource: hfResult ? hfResult.modelSource : 'local-path',
        };
        if (hfResult) {
            values.modelUrl = hfResult.url;
        }
        return values;
    }
    // Wire up HF picker
    setupHfPicker(modelInput, hfPickerContainer, () => container.querySelector('#engine').value === 'litert', (result) => {
        hfResult = result;
        // Auto-set task from HF pipeline_tag if available
        if (result?.pipelineTag) {
            const hfTaskMap = {
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
    });
    // Listen for changes on all inputs
    container.addEventListener('change', () => {
        if (document.activeElement === taskSelect) {
            updateInputOptions();
        }
        onChange(getValues());
    });
    container.addEventListener('input', (e) => {
        if (e.target.id === 'modelName') {
            onChange(getValues());
        }
    });
    // Fire initial
    onChange(getValues());
}
//# sourceMappingURL=config-panel.js.map