/**
 * webai.js Web UI — main entry point.
 *
 * Wires up: config panel → assembler → Monaco preview → Try it iframe.
 * Also: theme toggle, copy-to-clipboard, download-as-zip.
 */
import './style.css';
import { resolveConfig } from '@webai/core';
import { assemble } from 'webai';
import { zipSync, strToU8 } from 'fflate';
import { setupConfigPanel, updateUrlParams, readUrlParams } from './config-panel.js';
import { setupCodePreview, updateCodePreview, getActiveFileContent, setEditorTheme } from './code-preview.js';
import { setupTryIt } from './try-it.js';
import { createMockMetadata } from './mock-metadata.js';
let currentFramework = 'html';
let currentFiles = [];
let latestConfigValues = null;
function generateCode(values) {
    const metadata = createMockMetadata(values.task, values.engine === 'litert' ? 'tflite' : 'onnx');
    const ext = values.engine === 'litert' ? '.tflite' : '.onnx';
    const flags = {
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
    }
    catch (e) {
        console.error('Generation error:', e);
        return [{
                path: 'error.txt',
                content: `Code generation failed:\n${e instanceof Error ? e.message : String(e)}`,
            }];
    }
}
// ---- Theme toggle ----
function getPageTheme() {
    return document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
}
function setupThemeToggle(btn) {
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
        // Update URL param for page theme
        if (latestConfigValues) {
            updateUrlParams(latestConfigValues, next);
        }
    });
}
// ---- Copy to clipboard ----
function setupCopyButton(btn) {
    btn.addEventListener('click', async () => {
        const content = getActiveFileContent();
        if (!content)
            return;
        try {
            await navigator.clipboard.writeText(content);
            const original = btn.textContent;
            btn.textContent = 'Copied!';
            setTimeout(() => { btn.textContent = original; }, 1500);
        }
        catch {
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
function setupDownloadZip(btn) {
    btn.addEventListener('click', () => {
        if (currentFiles.length === 0)
            return;
        const files = {};
        for (const file of currentFiles) {
            files[file.path] = strToU8(file.content);
        }
        const zipped = zipSync(files);
        const blob = new Blob([zipped.buffer], { type: 'application/zip' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'webai-generated.zip';
        a.click();
        URL.revokeObjectURL(url);
    });
}
// ---- Init ----
async function init() {
    // Redirect bare URL to include full default params
    if (!window.location.search) {
        const defaults = new URLSearchParams({
            model: 'webnn/mobilenet-v2',
            task: 'image-classification',
            engine: 'ort',
            backend: 'auto',
            framework: 'html',
            input: 'file',
            lang: 'js',
            uitheme: 'dark',
            theme: 'dark',
        });
        const url = `${window.location.pathname}?${defaults.toString()}`;
        window.history.replaceState(null, '', url);
    }
    const configPanel = document.getElementById('configPanel');
    const editorContainer = document.getElementById('editor');
    const tabContainer = document.getElementById('fileTabs');
    const tryItSection = document.getElementById('tryItSection');
    const tryItBtn = document.getElementById('tryItBtn');
    const closeTryIt = document.getElementById('closeTryIt');
    const tryItFrame = document.getElementById('tryItFrame');
    const themeToggle = document.getElementById('themeToggle');
    const copyBtn = document.getElementById('copyBtn');
    const downloadZipBtn = document.getElementById('downloadZipBtn');
    setupThemeToggle(themeToggle);
    setupCopyButton(copyBtn);
    setupDownloadZip(downloadZipBtn);
    await setupCodePreview(editorContainer, tabContainer);
    setupTryIt(tryItSection, tryItBtn, closeTryIt, tryItFrame, () => currentFiles, () => currentFramework);
    setupConfigPanel(configPanel, (values) => {
        currentFramework = values.framework;
        latestConfigValues = values;
        currentFiles = generateCode(values);
        updateCodePreview(currentFiles, tabContainer);
        updateUrlParams(values, getPageTheme());
    });
}
init();
//# sourceMappingURL=main.js.map