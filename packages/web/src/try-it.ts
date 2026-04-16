/**
 * "Try it" functionality: runs generated HTML in a sandboxed iframe.
 *
 * Only works when framework is 'html' (single-file apps).
 * Other frameworks need npm install + build — show a message instead.
 */

import type { GeneratedFile } from 'webai';
import { relayoutEditor } from './code-preview.js';

export function canTryIt(framework: string): boolean {
  return framework === 'html';
}

export function runInIframe(
  iframe: HTMLIFrameElement,
  files: GeneratedFile[],
): void {
  const htmlFile = files.find((f) => f.path === 'index.html');
  if (!htmlFile) {
    iframe.srcdoc = '<body style="font-family:system-ui;padding:2rem;color:#666">No index.html found in generated files.</body>';
    return;
  }
  iframe.srcdoc = htmlFile.content;
}

export function setupTryIt(
  section: HTMLElement,
  tryItBtn: HTMLButtonElement,
  closeBtn: HTMLButtonElement,
  iframe: HTMLIFrameElement,
  getFiles: () => GeneratedFile[],
  getFramework: () => string,
): void {
  tryItBtn.addEventListener('click', () => {
    const isOpen = !section.classList.contains('is-closed');
    if (isOpen) {
      section.classList.remove('fullscreen');
      section.classList.add('is-closed');
      iframe.srcdoc = '';
      requestAnimationFrame(() => requestAnimationFrame(() => relayoutEditor()));
      return;
    }
    const framework = getFramework();
    if (!canTryIt(framework)) {
      iframe.srcdoc = `<body style="font-family:system-ui;padding:2rem;color:#666">
        <h2>Preview not available</h2>
        <p>The <strong>${framework}</strong> template requires <code>npm install && npm run dev</code>.</p>
        <p>Copy the generated code and run it locally.</p>
      </body>`;
      section.classList.remove('is-closed');
      requestAnimationFrame(() => requestAnimationFrame(() => relayoutEditor()));
      return;
    }
    const files = getFiles();
    runInIframe(iframe, files);
    section.classList.remove('is-closed');
    requestAnimationFrame(() => requestAnimationFrame(() => relayoutEditor()));
  });

  closeBtn.addEventListener('click', () => {
    section.classList.remove('fullscreen');
    section.classList.add('is-closed');
    iframe.srcdoc = '';
    requestAnimationFrame(() => requestAnimationFrame(() => relayoutEditor()));
  });
}
