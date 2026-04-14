import { describe, it, expect } from 'vitest';
import { BPETokenizer } from './text-tokenizer.js';

const MINI_TOKENIZER_JSON = {
  model: {
    type: 'BPE',
    vocab: {
      '<pad>': 0,
      '<s>': 1,
      '</s>': 2,
      '<unk>': 3,
      'h': 4,
      'e': 5,
      'l': 6,
      'o': 7,
      ' ': 8,
      'w': 9,
      'r': 10,
      'd': 11,
      'he': 12,
      'll': 13,
      'hel': 14,
      'lo': 15,
      'hell': 16,
      'hello': 17,
      'wo': 18,
      'wor': 19,
      'world': 20,
    },
    merges: [
      'h e',
      'he l',
      'hel l',
      'hell o',
      'l o',
      'l l',
      'w o',
      'wo r',
      'wor l',
      'worl d',
    ],
  },
  added_tokens: [
    { id: 0, content: '<pad>', special: true },
    { id: 1, content: '<s>', special: true },
    { id: 2, content: '</s>', special: true },
    { id: 3, content: '<unk>', special: true },
  ],
};

describe('BPETokenizer', () => {
  it('constructs from tokenizer.json data', () => {
    const tokenizer = new BPETokenizer(MINI_TOKENIZER_JSON);
    expect(tokenizer).toBeDefined();
  });

  it('encodes a single known word', () => {
    const tokenizer = new BPETokenizer(MINI_TOKENIZER_JSON);
    const result = tokenizer.encode('hello');
    expect(result.inputIds).toEqual([17]);
  });

  it('encodes multiple words with spaces', () => {
    const tokenizer = new BPETokenizer(MINI_TOKENIZER_JSON);
    const result = tokenizer.encode('hello world');
    expect(result.inputIds).toEqual([17, 8, 20]);
  });

  it('handles unknown characters with <unk> token', () => {
    const tokenizer = new BPETokenizer(MINI_TOKENIZER_JSON);
    const result = tokenizer.encode('hello!');
    expect(result.inputIds).toEqual([17, 3]);
  });

  it('returns attention_mask of all ones', () => {
    const tokenizer = new BPETokenizer(MINI_TOKENIZER_JSON);
    const result = tokenizer.encode('hello world');
    expect(result.attentionMask).toEqual([1, 1, 1]);
    expect(result.attentionMask.length).toBe(result.inputIds.length);
  });

  it('pads to maxLength when specified', () => {
    const tokenizer = new BPETokenizer(MINI_TOKENIZER_JSON);
    const result = tokenizer.encode('hello', { maxLength: 8, padTokenId: 0 });
    expect(result.inputIds.length).toBe(8);
    expect(result.attentionMask.length).toBe(8);
    expect(result.inputIds.slice(1)).toEqual([0, 0, 0, 0, 0, 0, 0]);
    expect(result.attentionMask.slice(1)).toEqual([0, 0, 0, 0, 0, 0, 0]);
  });

  it('truncates to maxLength when input exceeds it', () => {
    const tokenizer = new BPETokenizer(MINI_TOKENIZER_JSON);
    const result = tokenizer.encode('hello world', { maxLength: 2 });
    expect(result.inputIds.length).toBe(2);
    expect(result.attentionMask.length).toBe(2);
    expect(result.inputIds).toEqual([17, 8]);
  });

  it('returns empty arrays for empty input', () => {
    const tokenizer = new BPETokenizer(MINI_TOKENIZER_JSON);
    const result = tokenizer.encode('');
    expect(result.inputIds).toEqual([]);
    expect(result.attentionMask).toEqual([]);
  });

  it('pre-tokenizes by splitting on character boundaries', () => {
    const tokenizer = new BPETokenizer(MINI_TOKENIZER_JSON);
    const result = tokenizer.encode('he');
    expect(result.inputIds).toEqual([12]);
  });

  it('getVocabSize returns correct count', () => {
    const tokenizer = new BPETokenizer(MINI_TOKENIZER_JSON);
    expect(tokenizer.getVocabSize()).toBe(21);
  });

  it('decode converts token IDs back to text', () => {
    const tokenizer = new BPETokenizer(MINI_TOKENIZER_JSON);
    const text = tokenizer.decode([17, 8, 20]);
    expect(text).toBe('hello world');
  });
});
