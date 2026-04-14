/**
 * Text preprocess emitter: generates BPE tokenizer as browser-runnable code.
 *
 * The generated code loads a tokenizer.json file at runtime and provides
 * an encode() function that converts text → input_ids + attention_mask.
 *
 * Same pattern as audio-preprocess.ts and preprocess.ts: each emit*
 * function returns a template literal string containing a complete function.
 */

import type { ResolvedConfig } from '@webai/core';
import type { CodeBlock } from '../types.js';

/** Emit the BPE tokenizer class as standalone browser JS/TS code */
function emitBPETokenizer(ts: boolean): string {
  const t = ts;
  return `/**
 * BPE Tokenizer: loads tokenizer.json and encodes text to token IDs.
 */
class BPETokenizer {
  constructor(tokenizerJson${t ? ': any' : ''}) {
    this.vocab = new Map(Object.entries(tokenizerJson.model.vocab));
    this.reverseVocab = new Map(${t ? '<number, string>' : ''});
    for (const [token, id] of this.vocab) {
      this.reverseVocab.set(id${t ? ' as number' : ''}, token);
    }
    this.mergeRank = new Map(${t ? '<string, number>' : ''});
    const merges = tokenizerJson.model.merges;
    for (let i = 0; i < merges.length; i++) {
      this.mergeRank.set(merges[i], i);
    }
  }

  encode(text${t ? ': string' : ''}, maxLength${t ? '?: number' : ''})${t ? ': { inputIds: number[]; attentionMask: number[] }' : ''} {
    if (!text) return { inputIds: [], attentionMask: [] };

    const chars = Array.from(text);
    const tokens = this._applyBPE(chars);
    const unkId = this.vocab.get('<unk>') ?? 0;
    let inputIds = tokens.map((t) => this.vocab.get(t) ?? unkId);
    let attentionMask = inputIds.map(() => 1);

    if (maxLength !== undefined && inputIds.length > maxLength) {
      inputIds = inputIds.slice(0, maxLength);
      attentionMask = attentionMask.slice(0, maxLength);
    }

    return { inputIds, attentionMask };
  }

  decode(ids${t ? ': number[]' : ''})${t ? ': string' : ''} {
    return ids.map((id) => this.reverseVocab.get(id) ?? '').join('');
  }

  getVocabSize()${t ? ': number' : ''} {
    return this.vocab.size;
  }

  _applyBPE(tokens${t ? ': string[]' : ''})${t ? ': string[]' : ''} {
    if (tokens.length <= 1) return tokens;
    let symbols = [...tokens];

    while (symbols.length > 1) {
      let bestRank = Infinity;
      let bestIdx = -1;
      for (let i = 0; i < symbols.length - 1; i++) {
        const key = symbols[i] + ' ' + symbols[i + 1];
        const rank = this.mergeRank.get(key);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestIdx = i;
        }
      }
      if (bestIdx === -1) break;
      const merged = symbols[bestIdx] + symbols[bestIdx + 1];
      symbols = [...symbols.slice(0, bestIdx), merged, ...symbols.slice(bestIdx + 2)];
    }
    return symbols;
  }
}`;
}

/** Emit the loadTokenizer function */
function emitLoadTokenizer(ts: boolean): string {
  const t = ts;
  return `/**
 * Load a HuggingFace tokenizer.json and return a BPETokenizer instance.
 *
 * @param url - URL or path to tokenizer.json
 * @returns BPETokenizer instance ready for encoding
 */
async function loadTokenizer(url${t ? ': string' : ''})${t ? ': Promise<BPETokenizer>' : ''} {
  const response = await fetch(url);
  if (!response.ok) throw new Error('Failed to load tokenizer: ' + response.status);
  const json = await response.json();
  return new BPETokenizer(json);
}`;
}

/** Emit the tokenizeText convenience function */
function emitTokenizeText(ts: boolean): string {
  const t = ts;
  return `/**
 * Tokenize text input for model inference.
 * Returns typed arrays suitable for creating ORT tensors.
 *
 * @param tokenizer - BPETokenizer instance
 * @param text - Input text string
 * @param maxLength - Maximum sequence length (truncates if exceeded)
 * @returns Object with inputIds and attentionMask as BigInt64Arrays
 */
function tokenizeText(
  tokenizer${t ? ': BPETokenizer' : ''},
  text${t ? ': string' : ''},
  maxLength${t ? ': number' : ''} = 128
)${t ? ': { inputIds: BigInt64Array; attentionMask: BigInt64Array }' : ''} {
  const encoded = tokenizer.encode(text, maxLength);
  const inputIds = new BigInt64Array(encoded.inputIds.map((id) => BigInt(id)));
  const attentionMask = new BigInt64Array(encoded.attentionMask.map((m) => BigInt(m)));
  return { inputIds, attentionMask };
}`;
}

/**
 * Emit the text preprocessing CodeBlock.
 * Used by all text/NLP tasks: text-classification, zero-shot-classification,
 * text-generation, and feature-extraction (text mode).
 */
export function emitTextPreprocessBlock(config: ResolvedConfig): CodeBlock {
  const ts = config.lang === 'ts';

  const parts = [
    emitBPETokenizer(ts),
    emitLoadTokenizer(ts),
    emitTokenizeText(ts),
  ];

  return {
    id: 'preprocess',
    code: parts.join('\n\n'),
    imports: [],
    exports: ['BPETokenizer', 'loadTokenizer', 'tokenizeText'],
  };
}
