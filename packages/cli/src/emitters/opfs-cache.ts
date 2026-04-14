/**
 * OPFS (Origin Private File System) caching emitter.
 *
 * When config.offline is true, emits a utility that:
 * 1. First checks if the model file exists in OPFS
 * 2. If cached: reads directly from OPFS (instant, no network)
 * 3. If not cached: fetches from network, stores in OPFS for next time
 *
 * This gives users offline capability after the first load.
 * The generated code wraps the model path with a cached fetch.
 */

import type { ResolvedConfig } from '@webai/core';
import type { CodeBlock } from '../types.js';

function emitOpfsCache(ts: boolean): string {
  const t = ts;
  return `/**
 * Load a file with OPFS (Origin Private File System) caching.
 *
 * First load: fetches from network, stores in OPFS.
 * Subsequent loads: reads from OPFS (works offline).
 *
 * @param url - URL to fetch the file from
 * @param cacheKey - Filename to use in OPFS cache
 * @returns ArrayBuffer of the file contents
 */
async function cachedFetch(
  url${t ? ': string' : ''},
  cacheKey${t ? ': string' : ''}
)${t ? ': Promise<ArrayBuffer>' : ''} {
  try {
    const root = await navigator.storage.getDirectory();
    const dirHandle = await root.getDirectoryHandle('webai-cache', { create: true });

    // Try to read from cache first
    try {
      const fileHandle = await dirHandle.getFileHandle(cacheKey);
      const file = await fileHandle.getFile();
      console.log('[OPFS] Cache hit:', cacheKey);
      return await file.arrayBuffer();
    } catch {
      // Not cached yet, fetch from network
    }

    console.log('[OPFS] Cache miss, fetching:', url);
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error('Failed to fetch ' + url + ': ' + response.status);
    }
    const buffer = await response.arrayBuffer();

    // Store in cache for next time
    try {
      const fileHandle = await dirHandle.getFileHandle(cacheKey, { create: true });
      const writable = await fileHandle.createWritable();
      await writable.write(buffer);
      await writable.close();
      console.log('[OPFS] Cached:', cacheKey, '(' + (buffer.byteLength / 1024 / 1024).toFixed(1) + 'MB)');
    } catch (e) {
      console.warn('[OPFS] Failed to cache:', e);
    }

    return buffer;
  } catch (e) {
    // OPFS not available (e.g., non-secure context), fall back to direct fetch
    console.warn('[OPFS] Not available, fetching directly:', e);
    const response = await fetch(url);
    return await response.arrayBuffer();
  }
}`;
}

function emitClearCache(ts: boolean): string {
  const t = ts;
  return `/**
 * Clear all cached model files from OPFS.
 * Useful for forcing a fresh download or freeing storage.
 */
async function clearModelCache()${t ? ': Promise<void>' : ''} {
  try {
    const root = await navigator.storage.getDirectory();
    await root.removeEntry('webai-cache', { recursive: true });
    console.log('[OPFS] Cache cleared');
  } catch (e) {
    console.warn('[OPFS] Failed to clear cache:', e);
  }
}`;
}

/**
 * Emit the OPFS caching CodeBlock.
 * Only emitted when config.offline is true.
 */
export function emitOpfsCacheBlock(config: ResolvedConfig): CodeBlock {
  const ts = config.lang === 'ts';

  if (!config.offline) {
    return {
      id: 'opfs-cache',
      code: '',
      imports: [],
      exports: [],
    };
  }

  const parts = [
    emitOpfsCache(ts),
    emitClearCache(ts),
  ];

  return {
    id: 'opfs-cache',
    code: parts.join('\n\n'),
    imports: [],
    exports: ['cachedFetch', 'clearModelCache'],
  };
}
