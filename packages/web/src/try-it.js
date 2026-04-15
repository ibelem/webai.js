/**
 * "Try it" functionality: runs generated HTML in a sandboxed iframe.
 *
 * Only works when framework is 'html' (single-file apps).
 * Other frameworks need npm install + build — show a message instead.
 */
export function canTryIt(framework) {
    return framework === 'html';
}
export function runInIframe(iframe, files) {
    const htmlFile = files.find((f) => f.path === 'index.html');
    if (!htmlFile) {
        iframe.srcdoc = '<body style="font-family:system-ui;padding:2rem;color:#666">No index.html found in generated files.</body>';
        return;
    }
    iframe.srcdoc = htmlFile.content;
}
export function setupTryIt(section, tryItBtn, closeBtn, iframe, getFiles, getFramework) {
    tryItBtn.addEventListener('click', () => {
        const framework = getFramework();
        if (!canTryIt(framework)) {
            iframe.srcdoc = `<body style="font-family:system-ui;padding:2rem;color:#666">
        <h2>Preview not available</h2>
        <p>The <strong>${framework}</strong> template requires <code>npm install && npm run dev</code>.</p>
        <p>Copy the generated code and run it locally.</p>
      </body>`;
            section.hidden = false;
            return;
        }
        const files = getFiles();
        runInIframe(iframe, files);
        section.hidden = false;
    });
    closeBtn.addEventListener('click', () => {
        section.hidden = true;
        iframe.srcdoc = '';
    });
}
//# sourceMappingURL=try-it.js.map