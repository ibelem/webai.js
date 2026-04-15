/**
 * Code preview: Monaco editor with tabbed files.
 *
 * Loads Monaco from CDN via vs-loader.
 * Displays generated files as tabs — click a tab to switch files.
 */
let editor = null;
let monacoReady = null;
let currentFiles = [];
let activeTabIndex = 0;
const MONACO_CDN = 'https://cdn.jsdelivr.net/npm/monaco-editor@0.52.2/min';
function loadMonacoScript() {
    return new Promise((resolve, reject) => {
        if (window.require) {
            resolve();
            return;
        }
        const script = document.createElement('script');
        script.src = `${MONACO_CDN}/vs/loader.js`;
        script.onload = () => resolve();
        script.onerror = reject;
        document.head.appendChild(script);
    });
}
function initMonaco(container) {
    if (monacoReady)
        return monacoReady;
    monacoReady = loadMonacoScript().then(() => {
        return new Promise((resolve) => {
            window.require.config({ paths: { vs: `${MONACO_CDN}/vs` } });
            window.require(['vs/editor/editor.main'], () => {
                editor = window.monaco.editor.create(container, {
                    value: '// Select options to generate code',
                    language: 'javascript',
                    theme: 'vs-dark',
                    readOnly: true,
                    minimap: { enabled: false },
                    fontSize: 13,
                    fontFamily: "ui-monospace, 'Cascadia Code', 'Source Code Pro', monospace",
                    lineNumbers: 'on',
                    scrollBeyondLastLine: false,
                    automaticLayout: true,
                    wordWrap: 'on',
                    padding: { top: 8 },
                });
                resolve();
            });
        });
    });
    return monacoReady;
}
function getLanguage(path) {
    if (path.endsWith('.ts') || path.endsWith('.tsx'))
        return 'typescript';
    if (path.endsWith('.js') || path.endsWith('.jsx'))
        return 'javascript';
    if (path.endsWith('.html') || path.endsWith('.svelte'))
        return 'html';
    if (path.endsWith('.css'))
        return 'css';
    if (path.endsWith('.json'))
        return 'json';
    if (path.endsWith('.md'))
        return 'markdown';
    return 'plaintext';
}
function renderTabs(tabContainer) {
    tabContainer.innerHTML = '';
    for (let i = 0; i < currentFiles.length; i++) {
        const tab = document.createElement('button');
        tab.className = 'file-tab' + (i === activeTabIndex ? ' active' : '');
        tab.textContent = currentFiles[i].path;
        tab.role = 'tab';
        tab.setAttribute('aria-selected', String(i === activeTabIndex));
        tab.addEventListener('click', () => {
            activeTabIndex = i;
            renderTabs(tabContainer);
            showFile(i);
        });
        tabContainer.appendChild(tab);
    }
}
function showFile(index) {
    if (!editor || !currentFiles[index])
        return;
    const file = currentFiles[index];
    const lang = getLanguage(file.path);
    const model = window.monaco.editor.createModel(file.content, lang);
    editor.setModel(model);
}
export async function setupCodePreview(editorContainer, _tabContainer) {
    await initMonaco(editorContainer);
}
export function updateCodePreview(files, tabContainer) {
    currentFiles = files;
    activeTabIndex = 0;
    renderTabs(tabContainer);
    if (files.length > 0) {
        showFile(0);
    }
}
export function getGeneratedFiles() {
    return currentFiles;
}
export function getActiveFileContent() {
    if (!currentFiles[activeTabIndex])
        return null;
    return currentFiles[activeTabIndex].content;
}
//# sourceMappingURL=code-preview.js.map