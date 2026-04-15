/**
 * HuggingFace model file picker.
 *
 * When the user types a HuggingFace model ID (e.g., "Xenova/yolov8n")
 * or pastes a HuggingFace URL (e.g., "https://huggingface.co/Xenova/yolov8n/blob/main/model.onnx"),
 * fetches the repo's file list and shows a dropdown to pick which
 * .onnx/.tflite file to use.
 */
import { classifyModelInput, isHuggingFaceUrl, transformHuggingFaceUrl, buildHfApiUrl, buildHfFileUrl, pickBestModelFile, } from '@webai/core';
/**
 * Parse a HuggingFace URL to extract model ID and optional filename.
 *
 * Handles patterns:
 * - https://huggingface.co/owner/repo → { modelId: "owner/repo", filename: null }
 * - https://huggingface.co/owner/repo/blob/main/model.onnx → { modelId: "owner/repo", filename: "model.onnx" }
 * - https://huggingface.co/owner/repo/resolve/main/onnx/model.onnx → { modelId: "owner/repo", filename: "onnx/model.onnx" }
 * - https://hf.co/owner/repo/blob/main/model.onnx → same
 *
 * Returns null if the URL doesn't match a known HF pattern.
 */
function parseHfUrl(url) {
    try {
        const parsed = new URL(url);
        // Remove leading slash, split into segments
        const segments = parsed.pathname.replace(/^\//, '').split('/');
        if (segments.length < 2)
            return null;
        const modelId = `${segments[0]}/${segments[1]}`;
        // Just owner/repo — no file
        if (segments.length === 2) {
            return { modelId, filename: null };
        }
        // owner/repo/blob/branch/path or owner/repo/resolve/branch/path
        if ((segments[2] === 'blob' || segments[2] === 'resolve') && segments.length >= 5) {
            // segments[3] is branch, segments[4+] is the file path
            const filename = segments.slice(4).join('/');
            return { modelId, filename: filename || null };
        }
        // owner/repo/tree/... or other pages — treat as repo-level
        return { modelId, filename: null };
    }
    catch {
        return null;
    }
}
/** Cached API responses */
const apiCache = new Map();
function formatFileSize(bytes) {
    if (bytes < 1024)
        return `${bytes} B`;
    if (bytes < 1024 * 1024)
        return `${(bytes / 1024).toFixed(0)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
/**
 * Fetch model file list from HuggingFace API.
 * Returns only .onnx and .tflite files.
 */
async function fetchModelFiles(modelId) {
    const cached = apiCache.get(modelId);
    if (cached)
        return cached;
    const apiUrl = buildHfApiUrl(modelId);
    const response = await fetch(apiUrl);
    if (!response.ok) {
        if (response.status === 404) {
            throw new Error(`Model not found: "${modelId}"`);
        }
        throw new Error(`HuggingFace API error: HTTP ${response.status}`);
    }
    const data = await response.json();
    if (!data.siblings || !Array.isArray(data.siblings)) {
        throw new Error(`No files found for "${modelId}"`);
    }
    const modelExts = ['.onnx', '.tflite'];
    const files = data.siblings
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
 * @param getPreferTflite - Function returning whether to prefer .tflite files
 * @param onSelect - Callback when a file is selected (or null when deselected)
 */
export function setupHfPicker(input, container, getPreferTflite, onSelect, initialFile) {
    let debounceTimer = null;
    let currentModelId = '';
    function render(state, data) {
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
                if (file.filename === data.bestFile)
                    option.selected = true;
                select.appendChild(option);
            }
            wrapper.appendChild(select);
            container.appendChild(wrapper);
            // Fire initial selection
            const fireSelect = () => {
                const selectedFile = data.files.find((f) => f.filename === select.value);
                if (selectedFile && data.modelId) {
                    onSelect({
                        modelId: data.modelId,
                        filename: selectedFile.filename,
                        url: selectedFile.url,
                        pipelineTag: data.pipelineTag ?? null,
                        modelSource: data.modelSource ?? 'hf-model-id',
                    });
                }
            };
            select.addEventListener('change', fireSelect);
            fireSelect();
        }
    }
    async function checkInput(preferFile) {
        const value = input.value.trim();
        const sourceType = classifyModelInput(value);
        // Case 1: HuggingFace model ID (e.g., "Xenova/yolov8n")
        if (sourceType === 'hf-model-id') {
            if (value === currentModelId && !preferFile)
                return;
            currentModelId = value;
            render('loading');
            try {
                const { files, pipelineTag } = await fetchModelFiles(value);
                if (input.value.trim() !== value)
                    return;
                const siblings = files.map((f) => ({ rfilename: f.filename }));
                // Use preferFile from URL param if available, otherwise auto-pick
                const bestFile = preferFile && files.some((f) => f.filename === preferFile)
                    ? preferFile
                    : pickBestModelFile(siblings, getPreferTflite());
                render('files', { files, pipelineTag, bestFile, modelId: value, modelSource: 'hf-model-id' });
            }
            catch (e) {
                if (input.value.trim() !== value)
                    return;
                render('error', { error: e instanceof Error ? e.message : String(e) });
                onSelect(null);
            }
            return;
        }
        // Case 2: HuggingFace URL (e.g., "https://huggingface.co/Xenova/yolov8n/blob/main/model.onnx")
        if (sourceType === 'url' && isHuggingFaceUrl(value)) {
            const parsed = parseHfUrl(value);
            if (!parsed) {
                currentModelId = '';
                render('idle');
                onSelect(null);
                return;
            }
            if (value === currentModelId && !preferFile)
                return;
            currentModelId = value;
            render('loading');
            try {
                const { files, pipelineTag } = await fetchModelFiles(parsed.modelId);
                if (input.value.trim() !== value)
                    return;
                // If the URL points to a specific file, pre-select it
                const bestFile = parsed.filename
                    ? parsed.filename
                    : preferFile && files.some((f) => f.filename === preferFile)
                        ? preferFile
                        : pickBestModelFile(files.map((f) => ({ rfilename: f.filename })), getPreferTflite());
                render('files', { files, pipelineTag, bestFile, modelId: parsed.modelId, modelSource: 'url' });
            }
            catch (e) {
                if (input.value.trim() !== value)
                    return;
                // If URL points to a specific model file, use it directly even if API fails
                if (parsed.filename && /\.(onnx|tflite)$/i.test(parsed.filename)) {
                    const directUrl = transformHuggingFaceUrl(value);
                    container.innerHTML = '';
                    container.hidden = false;
                    const el = document.createElement('div');
                    el.className = 'hf-picker-status';
                    el.textContent = `Using: ${parsed.filename}`;
                    container.appendChild(el);
                    onSelect({
                        modelId: parsed.modelId,
                        filename: parsed.filename,
                        url: directUrl,
                        pipelineTag: null,
                        modelSource: 'url',
                    });
                    return;
                }
                render('error', { error: e instanceof Error ? e.message : String(e) });
                onSelect(null);
            }
            return;
        }
        // Case 3: Not an HF input — clear picker
        currentModelId = '';
        render('idle');
        onSelect(null);
    }
    input.addEventListener('input', () => {
        if (debounceTimer)
            clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => checkInput(), 400);
    });
    // Auto-trigger on setup if the input already has a value (e.g., from URL params)
    if (input.value.trim()) {
        checkInput(initialFile);
    }
}
//# sourceMappingURL=hf-picker.js.map