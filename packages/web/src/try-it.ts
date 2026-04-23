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

const FRAMEWORK_REASONS: Record<string, string> = {
  'react-vite': 'React + Vite projects require JSX compilation and a dev server.',
  'vanilla-vite': 'Vanilla + Vite projects use ES module imports that need bundling.',
  'vue-vite': 'Vue + Vite projects require SFC compilation (.vue files).',
  nuxt: 'Nuxt projects require server-side rendering and a build step.',
  sveltekit: 'SvelteKit projects require Svelte compilation and a dev server.',
  'svelte-vite': 'Svelte + Vite projects require Svelte compilation (.svelte files).',
  nextjs: 'Next.js projects require React/JSX compilation and a Node.js server.',
  astro: 'Astro projects require component compilation and a build step.',
};

export function getPreviewUnavailableHtml(framework: string, theme: 'dark' | 'light' = 'dark'): string {
  const reason = FRAMEWORK_REASONS[framework] ?? 'This framework requires a build step.';
  const dark = theme === 'dark';
  const bg = dark ? 'rgb(212, 212, 212)' : '#ffffff';
  const text = dark ? '#888' : '#555';
  const heading = dark ? '#ccc' : '#222';
  const preBg = dark ? '#0d0d1a' : '#e8e8e8';
  const preText = dark ? '#7ec8e3' : '#1a6b8a';
  return `<body style="font-family:system-ui,-apple-system,sans-serif;padding:1.5rem;color:${text};background:${bg};font-size:13px;line-height:1.5">
    <h2 style="color:${heading};margin-bottom:0.25rem;font-size:14px;font-weight:600">Preview not available</h2>
    <p style="margin-top:0.25rem">${reason}</p>
    <p style="margin-top:0.75rem">Run the following to preview locally:</p>
    <pre style="background:${preBg};padding:0.75rem;border-radius:6px;color:${preText};font-size:12px;margin-top:0.25rem"><code>npm install\nnpm run dev</code></pre>
    <p style="margin-top:0.5rem">Then open the local URL shown in the terminal.</p>
  </body>`;
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
  getTheme: () => 'dark' | 'light',
): void {
  tryItBtn.addEventListener('click', () => {
    const isOpen = !section.classList.contains('is-closed');
    if (isOpen) {
      section.classList.remove('fullscreen');
      section.classList.add('is-closed');
      requestAnimationFrame(() => requestAnimationFrame(() => relayoutEditor()));
      return;
    }
    // Content is kept in sync by config onChange; just show the section
    // If iframe was cleared on close, reload content now
    if (!iframe.srcdoc) {
      const framework = getFramework();
      if (!canTryIt(framework)) {
        iframe.srcdoc = getPreviewUnavailableHtml(framework, getTheme());
      } else {
        const files = getFiles();
        runInIframe(iframe, files);
      }
    }
    section.classList.remove('is-closed');
    requestAnimationFrame(() => requestAnimationFrame(() => relayoutEditor()));
  });

  closeBtn.addEventListener('click', () => {
    section.classList.remove('fullscreen');
    section.classList.add('is-closed');
    requestAnimationFrame(() => requestAnimationFrame(() => relayoutEditor()));
  });
}
