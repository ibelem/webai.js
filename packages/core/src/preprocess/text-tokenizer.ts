/**
 * BPE (Byte-Pair Encoding) tokenizer.
 *
 * Loads a HuggingFace-format tokenizer.json and encodes text to token IDs.
 * Used by text tasks (classification, generation, zero-shot, embedding).
 *
 * This is a simplified implementation covering the common BPE case.
 * It does NOT handle all tokenizer.json variants (WordPiece, Unigram, etc.).
 */

export interface TokenizerJson {
  model: {
    type: string;
    vocab: Record<string, number>;
    merges: string[];
  };
  added_tokens?: Array<{ id: number; content: string; special: boolean }>;
}

export interface EncodeResult {
  inputIds: number[];
  attentionMask: number[];
}

export interface EncodeOptions {
  maxLength?: number;
  padTokenId?: number;
}

export class BPETokenizer {
  private vocab: Map<string, number>;
  private reverseVocab: Map<number, string>;
  private merges: Array<[string, string]>;
  private mergeRank: Map<string, number>;

  constructor(tokenizerJson: TokenizerJson) {
    this.vocab = new Map(Object.entries(tokenizerJson.model.vocab));
    this.reverseVocab = new Map<number, string>();
    for (const [token, id] of this.vocab) {
      this.reverseVocab.set(id, token);
    }

    this.merges = tokenizerJson.model.merges.map((m) => {
      const parts = m.split(' ');
      return [parts[0], parts[1]];
    });

    this.mergeRank = new Map<string, number>();
    for (let i = 0; i < this.merges.length; i++) {
      const key = `${this.merges[i][0]} ${this.merges[i][1]}`;
      this.mergeRank.set(key, i);
    }
  }

  encode(text: string, options?: EncodeOptions): EncodeResult {
    if (text.length === 0) {
      return { inputIds: [], attentionMask: [] };
    }

    const chars = Array.from(text);
    const tokens = this.applyBPE(chars);

    const unkId = this.vocab.get('<unk>') ?? 0;
    let inputIds = tokens.map((t) => this.vocab.get(t) ?? unkId);
    let attentionMask = inputIds.map(() => 1);

    if (options?.maxLength !== undefined && inputIds.length > options.maxLength) {
      inputIds = inputIds.slice(0, options.maxLength);
      attentionMask = attentionMask.slice(0, options.maxLength);
    }

    if (options?.maxLength !== undefined && inputIds.length < options.maxLength) {
      const padId = options.padTokenId ?? 0;
      const padLen = options.maxLength - inputIds.length;
      inputIds = inputIds.concat(new Array(padLen).fill(padId));
      attentionMask = attentionMask.concat(new Array(padLen).fill(0));
    }

    return { inputIds, attentionMask };
  }

  decode(ids: number[]): string {
    return ids.map((id) => this.reverseVocab.get(id) ?? '').join('');
  }

  getVocabSize(): number {
    return this.vocab.size;
  }

  private applyBPE(tokens: string[]): string[] {
    if (tokens.length <= 1) return tokens;

    let symbols = [...tokens];

    while (symbols.length > 1) {
      let bestRank = Infinity;
      let bestIdx = -1;

      for (let i = 0; i < symbols.length - 1; i++) {
        const key = `${symbols[i]} ${symbols[i + 1]}`;
        const rank = this.mergeRank.get(key);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestIdx = i;
        }
      }

      if (bestIdx === -1) break;

      const merged = symbols[bestIdx] + symbols[bestIdx + 1];
      symbols = [
        ...symbols.slice(0, bestIdx),
        merged,
        ...symbols.slice(bestIdx + 2),
      ];
    }

    return symbols;
  }
}
