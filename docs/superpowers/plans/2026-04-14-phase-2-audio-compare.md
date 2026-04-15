# Phase 2: Audio Tasks + Compare Command

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add audio inference support (ASR, audio classification, TTS) with browser-native audio preprocessing (Mel spectrogram, MFCC via vendored FFT), enhanced mic capture with AudioWorklet, and a `webai compare` benchmarking command.

**Architecture:** Audio preprocessing follows the same 3-layer pattern as image tasks. Core package gets real DSP functions (FFT, STFT, mel, MFCC) for cross-verification testing. CLI emitters generate equivalent browser code as strings. Framework emitters wrap audio tasks with mic-capture UI, waveform display, and audio playback. The compare command generates a standalone HTML benchmark page that runs ONNX Runtime Web across backends (WASM, WebGPU, WebNN) and displays results.

**Tech Stack:** Vitest, TypeScript, vendored radix-2 FFT (zero deps), AudioWorklet API, Web Audio API, ONNX Runtime Web

---

## File Map

### New files

| File | Responsibility |
|------|---------------|
| `packages/core/src/preprocess/audio-fft.ts` | Radix-2 FFT + Hann window |
| `packages/core/src/preprocess/audio-mel.ts` | STFT, mel filterbank, mel spectrogram |
| `packages/core/src/preprocess/audio-mfcc.ts` | MFCC via DCT-II on mel spectrogram |
| `packages/core/tests/audio-fft.test.ts` | FFT unit tests |
| `packages/core/tests/audio-mel.test.ts` | Mel spectrogram unit tests |
| `packages/core/tests/audio-mfcc.test.ts` | MFCC unit tests |
| `packages/cli/src/emitters/audio-preprocess.ts` | Emits audio preprocessing as browser code |
| `packages/cli/src/compare.ts` | Compare command: generates benchmark HTML |

### Modified files

| File | Changes |
|------|---------|
| `packages/core/src/preprocess/index.ts` | Export FFT, mel, MFCC |
| `packages/cli/src/emitters/preprocess.ts` | Route audio tasks to audio preprocess emitter |
| `packages/cli/src/emitters/input.ts` | Add AudioWorklet capture + audio inference loop |
| `packages/cli/src/emitters/postprocess.ts` | Add STT greedy decoder + TTS audio output |
| `packages/cli/src/emitters/index.ts` | Import audio preprocess emitter |
| `packages/cli/src/frameworks/html.ts` | Add audio task scripts + body emitters |
| `packages/cli/src/frameworks/react-vite.ts` | Add audio task components |
| `packages/cli/src/frameworks/vanilla-vite.ts` | Add audio task support |
| `packages/cli/src/frameworks/nextjs.ts` | Add audio task support |
| `packages/cli/src/frameworks/sveltekit.ts` | Add audio task support |
| `packages/cli/src/cli.ts` | Wire up compare command |
| `packages/cli/tests/snapshot.test.ts` | Add audio task snapshots |
| `packages/cli/tests/framework-emitters.test.ts` | Add audio structural tests |

---

### Task 1: Core Audio FFT

**Files:**
- Create: `packages/core/src/preprocess/audio-fft.ts`
- Test: `packages/core/tests/audio-fft.test.ts`

- [ ] **Step 1: Write the failing tests**

```typescript
// packages/core/tests/audio-fft.test.ts
import { describe, it, expect } from 'vitest';
import { fft, hannWindow } from '../src/preprocess/audio-fft.js';

describe('hannWindow', () => {
  it('produces correct length', () => {
    const w = hannWindow(8);
    expect(w.length).toBe(8);
  });

  it('is zero at endpoints', () => {
    const w = hannWindow(8);
    expect(w[0]).toBeCloseTo(0, 10);
    expect(w[7]).toBeCloseTo(0, 10);
  });

  it('peaks at center', () => {
    const w = hannWindow(8);
    expect(w[4]).toBeCloseTo(1, 10);
  });

  it('is symmetric', () => {
    const w = hannWindow(16);
    for (let i = 0; i < 8; i++) {
      expect(w[i]).toBeCloseTo(w[15 - i], 10);
    }
  });
});

describe('fft', () => {
  it('DC signal: all energy in bin 0', () => {
    const re = new Float64Array([1, 1, 1, 1]);
    const im = new Float64Array(4);
    fft(re, im);
    expect(re[0]).toBeCloseTo(4, 10);
    expect(im[0]).toBeCloseTo(0, 10);
    // Other bins should be zero
    for (let i = 1; i < 4; i++) {
      expect(Math.abs(re[i])).toBeLessThan(1e-10);
      expect(Math.abs(im[i])).toBeLessThan(1e-10);
    }
  });

  it('pure cosine at bin 1', () => {
    const n = 8;
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      re[i] = Math.cos((2 * Math.PI * i) / n);
    }
    fft(re, im);
    // Bin 1 should have magnitude n/2 = 4
    const mag1 = Math.sqrt(re[1] * re[1] + im[1] * im[1]);
    expect(mag1).toBeCloseTo(4, 8);
    // Bin n-1 (mirror) should also have magnitude 4
    const mag7 = Math.sqrt(re[7] * re[7] + im[7] * im[7]);
    expect(mag7).toBeCloseTo(4, 8);
  });

  it('Parseval theorem: energy preserved', () => {
    const n = 16;
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    // Random-ish signal
    for (let i = 0; i < n; i++) {
      re[i] = Math.sin(i * 0.7) + Math.cos(i * 1.3);
    }
    const timeDomainEnergy = re.reduce((s, v) => s + v * v, 0);

    fft(re, im);
    const freqDomainEnergy = re.reduce((s, v, i) => s + v * v + im[i] * im[i], 0) / n;

    expect(freqDomainEnergy).toBeCloseTo(timeDomainEnergy, 6);
  });

  it('handles length 1', () => {
    const re = new Float64Array([5]);
    const im = new Float64Array([0]);
    fft(re, im);
    expect(re[0]).toBe(5);
    expect(im[0]).toBe(0);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/core && npx vitest run tests/audio-fft.test.ts`
Expected: FAIL — module `../src/preprocess/audio-fft.js` not found

- [ ] **Step 3: Implement FFT and Hann window**

```typescript
// packages/core/src/preprocess/audio-fft.ts

/**
 * Hann window function.
 * w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))
 */
export function hannWindow(length: number): Float64Array {
  const w = new Float64Array(length);
  for (let i = 0; i < length; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (length - 1)));
  }
  return w;
}

/**
 * In-place radix-2 Cooley-Tukey FFT.
 * Input length must be a power of 2.
 * Transforms (re, im) arrays in place.
 */
export function fft(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  if (n <= 1) return;

  // Bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;

    if (i < j) {
      let tmp = re[i]; re[i] = re[j]; re[j] = tmp;
      tmp = im[i]; im[i] = im[j]; im[j] = tmp;
    }
  }

  // Cooley-Tukey butterfly
  for (let len = 2; len <= n; len *= 2) {
    const halfLen = len / 2;
    const angle = (-2 * Math.PI) / len;
    const wRe = Math.cos(angle);
    const wIm = Math.sin(angle);

    for (let i = 0; i < n; i += len) {
      let curRe = 1;
      let curIm = 0;

      for (let j = 0; j < halfLen; j++) {
        const uRe = re[i + j];
        const uIm = im[i + j];
        const tRe = curRe * re[i + j + halfLen] - curIm * im[i + j + halfLen];
        const tIm = curRe * im[i + j + halfLen] + curIm * re[i + j + halfLen];

        re[i + j] = uRe + tRe;
        im[i + j] = uIm + tIm;
        re[i + j + halfLen] = uRe - tRe;
        im[i + j + halfLen] = uIm - tIm;

        const newCurRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = newCurRe;
      }
    }
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/core && npx vitest run tests/audio-fft.test.ts`
Expected: PASS (all 6 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/core/src/preprocess/audio-fft.ts packages/core/tests/audio-fft.test.ts
git commit -m "feat(core): add radix-2 FFT and Hann window for audio preprocessing"
```

---

### Task 2: Core Mel Spectrogram

**Files:**
- Create: `packages/core/src/preprocess/audio-mel.ts`
- Test: `packages/core/tests/audio-mel.test.ts`

- [ ] **Step 1: Write the failing tests**

```typescript
// packages/core/tests/audio-mel.test.ts
import { describe, it, expect } from 'vitest';
import { stft, melFilterbank, melSpectrogram } from '../src/preprocess/audio-mel.js';

describe('stft', () => {
  it('returns correct number of frames', () => {
    // 1 second at 16kHz, fftSize=512, hop=160
    const samples = new Float32Array(16000);
    const { magnitudes } = stft(samples, 512, 160);
    const expectedFrames = Math.floor((16000 - 512) / 160) + 1;
    expect(magnitudes.length).toBe(expectedFrames);
  });

  it('each frame has fftSize/2+1 bins', () => {
    const samples = new Float32Array(1024);
    const { magnitudes } = stft(samples, 256, 128);
    expect(magnitudes[0].length).toBe(129); // 256/2 + 1
  });

  it('detects pure tone frequency', () => {
    const sampleRate = 16000;
    const freq = 440; // A4
    const n = 4096;
    const samples = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      samples[i] = Math.sin((2 * Math.PI * freq * i) / sampleRate);
    }
    const fftSize = 1024;
    const { magnitudes } = stft(samples, fftSize, fftSize);
    // Find peak bin in first frame
    const frame = magnitudes[0];
    let peakBin = 0;
    for (let i = 1; i < frame.length; i++) {
      if (frame[i] > frame[peakBin]) peakBin = i;
    }
    const peakFreq = (peakBin * sampleRate) / fftSize;
    expect(Math.abs(peakFreq - freq)).toBeLessThan(sampleRate / fftSize);
  });
});

describe('melFilterbank', () => {
  it('returns correct number of filters', () => {
    const filters = melFilterbank(80, 512, 16000);
    expect(filters.length).toBe(80);
  });

  it('each filter has fftSize/2+1 bins', () => {
    const filters = melFilterbank(40, 256, 16000);
    for (const f of filters) {
      expect(f.length).toBe(129); // 256/2 + 1
    }
  });

  it('filters are non-negative', () => {
    const filters = melFilterbank(80, 512, 16000);
    for (const f of filters) {
      for (let i = 0; i < f.length; i++) {
        expect(f[i]).toBeGreaterThanOrEqual(0);
      }
    }
  });

  it('filter centers increase monotonically (mel-spaced)', () => {
    const filters = melFilterbank(40, 512, 16000);
    const centers = filters.map((f) => {
      let maxVal = 0;
      let maxIdx = 0;
      for (let i = 0; i < f.length; i++) {
        if (f[i] > maxVal) { maxVal = f[i]; maxIdx = i; }
      }
      return maxIdx;
    });
    for (let i = 1; i < centers.length; i++) {
      expect(centers[i]).toBeGreaterThanOrEqual(centers[i - 1]);
    }
  });
});

describe('melSpectrogram', () => {
  it('output shape is [numFrames, numMelBands]', () => {
    const samples = new Float32Array(16000);
    const result = melSpectrogram(samples, 16000, 512, 160, 80);
    const expectedFrames = Math.floor((16000 - 512) / 160) + 1;
    expect(result.numFrames).toBe(expectedFrames);
    expect(result.numMelBands).toBe(80);
    expect(result.data.length).toBe(expectedFrames * 80);
  });

  it('log mel values are finite', () => {
    // Silence input should produce very small (negative) log values, not NaN/Inf
    const samples = new Float32Array(16000);
    const result = melSpectrogram(samples, 16000, 512, 160, 80);
    for (let i = 0; i < result.data.length; i++) {
      expect(Number.isFinite(result.data[i])).toBe(true);
    }
  });

  it('loud signal produces higher mel values than silence', () => {
    const silence = new Float32Array(16000);
    const loud = new Float32Array(16000);
    for (let i = 0; i < 16000; i++) {
      loud[i] = Math.sin((2 * Math.PI * 440 * i) / 16000);
    }
    const silentMel = melSpectrogram(silence, 16000, 512, 160, 80);
    const loudMel = melSpectrogram(loud, 16000, 512, 160, 80);
    // Average mel energy should be higher for the loud signal
    const avgSilent = silentMel.data.reduce((s, v) => s + v, 0) / silentMel.data.length;
    const avgLoud = loudMel.data.reduce((s, v) => s + v, 0) / loudMel.data.length;
    expect(avgLoud).toBeGreaterThan(avgSilent);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/core && npx vitest run tests/audio-mel.test.ts`
Expected: FAIL — module not found

- [ ] **Step 3: Implement STFT, mel filterbank, and mel spectrogram**

```typescript
// packages/core/src/preprocess/audio-mel.ts
import { fft, hannWindow } from './audio-fft.js';

/**
 * Short-Time Fourier Transform.
 * Applies a Hann window and computes magnitude spectrum per frame.
 *
 * @returns Array of magnitude spectra, each with fftSize/2+1 bins.
 */
export function stft(
  samples: Float32Array,
  fftSize: number,
  hopSize: number,
): { magnitudes: Float64Array[] } {
  const window = hannWindow(fftSize);
  const numFrames = Math.max(0, Math.floor((samples.length - fftSize) / hopSize) + 1);
  const numBins = fftSize / 2 + 1;
  const magnitudes: Float64Array[] = [];

  for (let f = 0; f < numFrames; f++) {
    const offset = f * hopSize;
    const re = new Float64Array(fftSize);
    const im = new Float64Array(fftSize);

    for (let j = 0; j < fftSize; j++) {
      re[j] = (samples[offset + j] ?? 0) * window[j];
    }

    fft(re, im);

    // Magnitude spectrum (only positive frequencies)
    const mag = new Float64Array(numBins);
    for (let k = 0; k < numBins; k++) {
      mag[k] = Math.sqrt(re[k] * re[k] + im[k] * im[k]);
    }
    magnitudes.push(mag);
  }

  return { magnitudes };
}

/**
 * Hz to mel scale (Slaney/HTK formula).
 */
function hzToMel(hz: number): number {
  return 2595 * Math.log10(1 + hz / 700);
}

/**
 * Mel to Hz scale.
 */
function melToHz(mel: number): number {
  return 700 * (10 ** (mel / 2595) - 1);
}

/**
 * Build triangular mel-scale filterbank.
 *
 * @param numMelBands - Number of mel bands (e.g. 80 for Whisper, 64 for YAMNet)
 * @param fftSize - FFT size used in STFT
 * @param sampleRate - Audio sample rate in Hz
 * @returns Array of numMelBands filters, each with fftSize/2+1 weights.
 */
export function melFilterbank(
  numMelBands: number,
  fftSize: number,
  sampleRate: number,
): Float64Array[] {
  const numBins = fftSize / 2 + 1;
  const melMin = hzToMel(0);
  const melMax = hzToMel(sampleRate / 2);

  // numMelBands + 2 equally spaced points in mel scale
  const melPoints = new Float64Array(numMelBands + 2);
  for (let i = 0; i < numMelBands + 2; i++) {
    melPoints[i] = melToHz(melMin + ((melMax - melMin) * i) / (numMelBands + 1));
  }

  const filters: Float64Array[] = [];

  for (let m = 0; m < numMelBands; m++) {
    const filter = new Float64Array(numBins);
    const low = melPoints[m];
    const center = melPoints[m + 1];
    const high = melPoints[m + 2];

    for (let k = 0; k < numBins; k++) {
      const freq = (k * sampleRate) / fftSize;
      if (freq >= low && freq <= center && center > low) {
        filter[k] = (freq - low) / (center - low);
      } else if (freq > center && freq <= high && high > center) {
        filter[k] = (high - freq) / (high - center);
      }
    }
    filters.push(filter);
  }

  return filters;
}

export interface MelSpectrogramResult {
  /** Flat Float32Array of shape [numFrames, numMelBands], row-major */
  data: Float32Array;
  numFrames: number;
  numMelBands: number;
}

/**
 * Compute log-mel spectrogram from raw audio samples.
 *
 * Pipeline: STFT → power spectrum → mel filterbank → log10
 *
 * @param samples - Raw PCM audio, mono, as Float32Array
 * @param sampleRate - Audio sample rate (e.g. 16000)
 * @param fftSize - FFT window size (e.g. 512)
 * @param hopSize - Hop between frames (e.g. 160)
 * @param numMelBands - Number of mel bands (e.g. 80)
 */
export function melSpectrogram(
  samples: Float32Array,
  sampleRate: number,
  fftSize: number,
  hopSize: number,
  numMelBands: number,
): MelSpectrogramResult {
  const { magnitudes } = stft(samples, fftSize, hopSize);
  const filters = melFilterbank(numMelBands, fftSize, sampleRate);
  const numFrames = magnitudes.length;
  const numBins = fftSize / 2 + 1;
  const data = new Float32Array(numFrames * numMelBands);

  for (let f = 0; f < numFrames; f++) {
    const mag = magnitudes[f];

    for (let m = 0; m < numMelBands; m++) {
      let sum = 0;
      const filterWeights = filters[m];
      for (let k = 0; k < numBins; k++) {
        // Power spectrum: magnitude squared
        sum += mag[k] * mag[k] * filterWeights[k];
      }
      // Log mel (floor at 1e-10 to avoid -Infinity)
      data[f * numMelBands + m] = Math.log10(Math.max(sum, 1e-10));
    }
  }

  return { data, numFrames, numMelBands };
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/core && npx vitest run tests/audio-mel.test.ts`
Expected: PASS (all 9 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/core/src/preprocess/audio-mel.ts packages/core/tests/audio-mel.test.ts
git commit -m "feat(core): add STFT, mel filterbank, and mel spectrogram"
```

---

### Task 3: Core MFCC

**Files:**
- Create: `packages/core/src/preprocess/audio-mfcc.ts`
- Test: `packages/core/tests/audio-mfcc.test.ts`
- Modify: `packages/core/src/preprocess/index.ts`

- [ ] **Step 1: Write the failing tests**

```typescript
// packages/core/tests/audio-mfcc.test.ts
import { describe, it, expect } from 'vitest';
import { mfcc } from '../src/preprocess/audio-mfcc.js';

describe('mfcc', () => {
  it('output shape is [numFrames, numCoeffs]', () => {
    const numFrames = 10;
    const numMelBands = 40;
    const numCoeffs = 13;
    // Fake mel spectrogram data
    const melData = new Float32Array(numFrames * numMelBands);
    for (let i = 0; i < melData.length; i++) {
      melData[i] = Math.random() * -5; // Log mel values are typically negative
    }
    const result = mfcc(melData, numFrames, numMelBands, numCoeffs);
    expect(result.length).toBe(numFrames * numCoeffs);
  });

  it('first coefficient (c0) is proportional to total energy', () => {
    const numFrames = 1;
    const numMelBands = 8;
    const numCoeffs = 4;

    // All mel bands equal: c0 should be large, higher coefficients small
    const uniform = new Float32Array(numMelBands).fill(-1);
    const result = mfcc(uniform, numFrames, numMelBands, numCoeffs);
    const c0 = result[0];
    // c0 = sum of all mel bands * cos(0) = sum
    expect(c0).toBeCloseTo(-8, 5);
  });

  it('produces finite values', () => {
    const numFrames = 5;
    const numMelBands = 40;
    const numCoeffs = 13;
    const melData = new Float32Array(numFrames * numMelBands);
    for (let i = 0; i < melData.length; i++) {
      melData[i] = -3 + Math.random();
    }
    const result = mfcc(melData, numFrames, numMelBands, numCoeffs);
    for (let i = 0; i < result.length; i++) {
      expect(Number.isFinite(result[i])).toBe(true);
    }
  });

  it('different inputs produce different coefficients', () => {
    const numMelBands = 16;
    const numCoeffs = 8;
    const a = new Float32Array(numMelBands);
    const b = new Float32Array(numMelBands);
    for (let i = 0; i < numMelBands; i++) {
      a[i] = -2;
      b[i] = -2 + (i / numMelBands); // Linearly varying
    }
    const ra = mfcc(a, 1, numMelBands, numCoeffs);
    const rb = mfcc(b, 1, numMelBands, numCoeffs);
    // At least one higher-order coefficient should differ
    let differ = false;
    for (let c = 1; c < numCoeffs; c++) {
      if (Math.abs(ra[c] - rb[c]) > 1e-6) differ = true;
    }
    expect(differ).toBe(true);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/core && npx vitest run tests/audio-mfcc.test.ts`
Expected: FAIL — module not found

- [ ] **Step 3: Implement MFCC**

```typescript
// packages/core/src/preprocess/audio-mfcc.ts

/**
 * Compute MFCC (Mel-Frequency Cepstral Coefficients) from log-mel spectrogram.
 * Uses DCT-II to decorrelate mel bands.
 *
 * @param melData - Flat log-mel spectrogram, shape [numFrames, numMelBands], row-major
 * @param numFrames - Number of time frames
 * @param numMelBands - Number of mel bands per frame
 * @param numCoeffs - Number of MFCC coefficients to keep (typically 13)
 * @returns Float32Array of shape [numFrames, numCoeffs], row-major
 */
export function mfcc(
  melData: Float32Array,
  numFrames: number,
  numMelBands: number,
  numCoeffs: number,
): Float32Array {
  const result = new Float32Array(numFrames * numCoeffs);

  for (let f = 0; f < numFrames; f++) {
    const frameOffset = f * numMelBands;
    const outOffset = f * numCoeffs;

    for (let c = 0; c < numCoeffs; c++) {
      let sum = 0;
      for (let m = 0; m < numMelBands; m++) {
        sum +=
          melData[frameOffset + m] *
          Math.cos((Math.PI * c * (m + 0.5)) / numMelBands);
      }
      result[outOffset + c] = sum;
    }
  }

  return result;
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/core && npx vitest run tests/audio-mfcc.test.ts`
Expected: PASS (all 4 tests)

- [ ] **Step 5: Update core preprocess index to export audio modules**

In `packages/core/src/preprocess/index.ts`, add:

```typescript
export { fft, hannWindow } from './audio-fft.js';
export { stft, melFilterbank, melSpectrogram, type MelSpectrogramResult } from './audio-mel.js';
export { mfcc } from './audio-mfcc.js';
```

- [ ] **Step 6: Run all core tests**

Run: `cd packages/core && npx vitest run`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add packages/core/src/preprocess/audio-mfcc.ts packages/core/tests/audio-mfcc.test.ts packages/core/src/preprocess/index.ts
git commit -m "feat(core): add MFCC and export all audio preprocessing"
```

---

### Task 4: Audio Preprocess Emitter

**Files:**
- Create: `packages/cli/src/emitters/audio-preprocess.ts`
- Modify: `packages/cli/src/emitters/preprocess.ts`
- Modify: `packages/cli/src/emitters/index.ts`

This emitter generates the FFT, STFT, mel spectrogram, and MFCC as browser-runnable code strings. Same pattern as the image preprocess emitter: the generated code mirrors the core functions exactly.

- [ ] **Step 1: Write the failing test**

Add to `packages/cli/tests/framework-emitters.test.ts`, inside a new `describe('audio preprocess emitter')` block:

```typescript
describe('audio preprocess emitter', () => {
  it('emits mel spectrogram functions for speech-to-text', () => {
    const config = makeConfig({ task: 'speech-to-text', input: 'file', engine: 'ort' });
    const blocks = emitLayer1(config);
    const preprocessBlock = blocks.find((b) => b.id === 'preprocess');
    expect(preprocessBlock).toBeDefined();
    expect(preprocessBlock!.code).toContain('function fft(');
    expect(preprocessBlock!.code).toContain('function hannWindow(');
    expect(preprocessBlock!.code).toContain('function stft(');
    expect(preprocessBlock!.code).toContain('function melFilterbank(');
    expect(preprocessBlock!.code).toContain('function melSpectrogram(');
    expect(preprocessBlock!.exports).toContain('melSpectrogram');
  });

  it('emits MFCC for audio-classification', () => {
    const config = makeConfig({ task: 'audio-classification', input: 'file', engine: 'ort' });
    const blocks = emitLayer1(config);
    const preprocessBlock = blocks.find((b) => b.id === 'preprocess');
    expect(preprocessBlock).toBeDefined();
    expect(preprocessBlock!.code).toContain('function mfcc(');
    expect(preprocessBlock!.exports).toContain('mfcc');
    expect(preprocessBlock!.exports).toContain('melSpectrogram');
  });

  it('does not emit image preprocessing for audio tasks', () => {
    const config = makeConfig({ task: 'speech-to-text', input: 'file', engine: 'ort' });
    const blocks = emitLayer1(config);
    const preprocessBlock = blocks.find((b) => b.id === 'preprocess');
    expect(preprocessBlock!.code).not.toContain('function resizeImage(');
    expect(preprocessBlock!.code).not.toContain('function toNCHW(');
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/cli && npx vitest run tests/framework-emitters.test.ts -t "audio preprocess emitter"`
Expected: FAIL — no audio functions in preprocess block

- [ ] **Step 3: Create the audio preprocess emitter**

```typescript
// packages/cli/src/emitters/audio-preprocess.ts

/**
 * Audio preprocess emitter: generates standalone audio preprocessing code.
 *
 * Mirrors the core FFT, STFT, mel spectrogram, and MFCC functions exactly.
 * Cross-verified by tests: eval(emitted code) === real function output.
 *
 * Task dispatch:
 *   speech-to-text       → FFT + STFT + mel spectrogram
 *   audio-classification → FFT + STFT + mel spectrogram + MFCC
 *   text-to-speech       → empty (no audio preprocessing for TTS input)
 */

import type { ResolvedConfig } from '@webai/core';
import type { CodeBlock } from '../types.js';

function emitHannWindow(ts: boolean): string {
  const t = ts;
  return `/**
 * Hann window function.
 */
function hannWindow(length${t ? ': number' : ''})${t ? ': Float64Array' : ''} {
  const w = new Float64Array(length);
  for (let i = 0; i < length; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (length - 1)));
  }
  return w;
}`;
}

function emitFft(ts: boolean): string {
  const t = ts;
  return `/**
 * In-place radix-2 Cooley-Tukey FFT.
 * Input length must be a power of 2.
 */
function fft(re${t ? ': Float64Array' : ''}, im${t ? ': Float64Array' : ''})${t ? ': void' : ''} {
  const n = re.length;
  if (n <= 1) return;

  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    while (j & bit) { j ^= bit; bit >>= 1; }
    j ^= bit;
    if (i < j) {
      let tmp = re[i]; re[i] = re[j]; re[j] = tmp;
      tmp = im[i]; im[i] = im[j]; im[j] = tmp;
    }
  }

  for (let len = 2; len <= n; len *= 2) {
    const halfLen = len / 2;
    const angle = (-2 * Math.PI) / len;
    const wRe = Math.cos(angle);
    const wIm = Math.sin(angle);

    for (let i = 0; i < n; i += len) {
      let curRe = 1, curIm = 0;
      for (let j = 0; j < halfLen; j++) {
        const uRe = re[i + j];
        const uIm = im[i + j];
        const tRe = curRe * re[i + j + halfLen] - curIm * im[i + j + halfLen];
        const tIm = curRe * im[i + j + halfLen] + curIm * re[i + j + halfLen];
        re[i + j] = uRe + tRe;
        im[i + j] = uIm + tIm;
        re[i + j + halfLen] = uRe - tRe;
        im[i + j + halfLen] = uIm - tIm;
        const newCurRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = newCurRe;
      }
    }
  }
}`;
}

function emitStft(ts: boolean): string {
  const t = ts;
  return `/**
 * Short-Time Fourier Transform with Hann window.
 * Returns power spectrum frames (magnitude squared).
 */
function stft(
  samples${t ? ': Float32Array' : ''},
  fftSize${t ? ': number' : ''},
  hopSize${t ? ': number' : ''}
)${t ? ': Float64Array[]' : ''} {
  const window = hannWindow(fftSize);
  const numFrames = Math.max(0, Math.floor((samples.length - fftSize) / hopSize) + 1);
  const numBins = fftSize / 2 + 1;
  const frames${t ? ': Float64Array[]' : ''} = [];

  for (let f = 0; f < numFrames; f++) {
    const offset = f * hopSize;
    const re = new Float64Array(fftSize);
    const im = new Float64Array(fftSize);

    for (let j = 0; j < fftSize; j++) {
      re[j] = (samples[offset + j] || 0) * window[j];
    }

    fft(re, im);

    const power = new Float64Array(numBins);
    for (let k = 0; k < numBins; k++) {
      power[k] = re[k] * re[k] + im[k] * im[k];
    }
    frames.push(power);
  }

  return frames;
}`;
}

function emitMelFilterbank(ts: boolean): string {
  const t = ts;
  return `/**
 * Build triangular mel-scale filterbank.
 */
function melFilterbank(
  numMelBands${t ? ': number' : ''},
  fftSize${t ? ': number' : ''},
  sampleRate${t ? ': number' : ''}
)${t ? ': Float64Array[]' : ''} {
  const hzToMel = (hz${t ? ': number' : ''}) => 2595 * Math.log10(1 + hz / 700);
  const melToHz = (mel${t ? ': number' : ''}) => 700 * (10 ** (mel / 2595) - 1);

  const numBins = fftSize / 2 + 1;
  const melMin = hzToMel(0);
  const melMax = hzToMel(sampleRate / 2);

  const melPoints = new Float64Array(numMelBands + 2);
  for (let i = 0; i < numMelBands + 2; i++) {
    melPoints[i] = melToHz(melMin + ((melMax - melMin) * i) / (numMelBands + 1));
  }

  const filters${t ? ': Float64Array[]' : ''} = [];
  for (let m = 0; m < numMelBands; m++) {
    const filter = new Float64Array(numBins);
    const low = melPoints[m];
    const center = melPoints[m + 1];
    const high = melPoints[m + 2];

    for (let k = 0; k < numBins; k++) {
      const freq = (k * sampleRate) / fftSize;
      if (freq >= low && freq <= center && center > low) {
        filter[k] = (freq - low) / (center - low);
      } else if (freq > center && freq <= high && high > center) {
        filter[k] = (high - freq) / (high - center);
      }
    }
    filters.push(filter);
  }

  return filters;
}`;
}

function emitMelSpectrogram(ts: boolean): string {
  const t = ts;
  return `/**
 * Compute log-mel spectrogram from raw audio samples.
 * Pipeline: STFT -> mel filterbank -> log10
 *
 * @param samples - Mono PCM audio as Float32Array
 * @param sampleRate - Sample rate in Hz (e.g. 16000)
 * @param fftSize - FFT window size (e.g. 512)
 * @param hopSize - Hop between frames (e.g. 160)
 * @param numMelBands - Number of mel bands (e.g. 80)
 * @returns Float32Array of shape [numFrames * numMelBands]
 */
function melSpectrogram(
  samples${t ? ': Float32Array' : ''},
  sampleRate${t ? ': number' : ''},
  fftSize${t ? ': number' : ''},
  hopSize${t ? ': number' : ''},
  numMelBands${t ? ': number' : ''}
)${t ? ': { data: Float32Array; numFrames: number }' : ''} {
  const frames = stft(samples, fftSize, hopSize);
  const filters = melFilterbank(numMelBands, fftSize, sampleRate);
  const numFrames = frames.length;
  const numBins = fftSize / 2 + 1;
  const data = new Float32Array(numFrames * numMelBands);

  for (let f = 0; f < numFrames; f++) {
    const power = frames[f];
    for (let m = 0; m < numMelBands; m++) {
      let sum = 0;
      const w = filters[m];
      for (let k = 0; k < numBins; k++) {
        sum += power[k] * w[k];
      }
      data[f * numMelBands + m] = Math.log10(Math.max(sum, 1e-10));
    }
  }

  return { data, numFrames };
}`;
}

function emitMfcc(ts: boolean): string {
  const t = ts;
  return `/**
 * Compute MFCC from log-mel spectrogram via DCT-II.
 *
 * @param melData - Flat log-mel data [numFrames * numMelBands]
 * @param numFrames - Number of time frames
 * @param numMelBands - Number of mel bands per frame
 * @param numCoeffs - Number of MFCC coefficients (typically 13)
 * @returns Float32Array of shape [numFrames * numCoeffs]
 */
function mfcc(
  melData${t ? ': Float32Array' : ''},
  numFrames${t ? ': number' : ''},
  numMelBands${t ? ': number' : ''},
  numCoeffs${t ? ': number' : ''}
)${t ? ': Float32Array' : ''} {
  const result = new Float32Array(numFrames * numCoeffs);

  for (let f = 0; f < numFrames; f++) {
    const frameOffset = f * numMelBands;
    const outOffset = f * numCoeffs;
    for (let c = 0; c < numCoeffs; c++) {
      let sum = 0;
      for (let m = 0; m < numMelBands; m++) {
        sum += melData[frameOffset + m] * Math.cos((Math.PI * c * (m + 0.5)) / numMelBands);
      }
      result[outOffset + c] = sum;
    }
  }

  return result;
}`;
}

/**
 * Emit the audio preprocessing CodeBlock for a given config.
 *
 * Dispatches by task:
 *   speech-to-text       -> FFT + STFT + mel spectrogram + preprocessAudio
 *   audio-classification -> FFT + STFT + mel spectrogram + MFCC + preprocessAudio
 *   text-to-speech       -> empty block (no audio preprocessing)
 */
export function emitAudioPreprocessBlock(config: ResolvedConfig): CodeBlock {
  const ts = config.lang === 'ts';

  if (config.task === 'text-to-speech') {
    return { id: 'preprocess', code: '', imports: [], exports: [] };
  }

  const parts: string[] = [
    emitHannWindow(ts),
    emitFft(ts),
    emitStft(ts),
    emitMelFilterbank(ts),
    emitMelSpectrogram(ts),
  ];
  const exports = ['hannWindow', 'fft', 'stft', 'melFilterbank', 'melSpectrogram'];

  if (config.task === 'audio-classification') {
    parts.push(emitMfcc(ts));
    exports.push('mfcc');
  }

  return {
    id: 'preprocess',
    code: parts.join('\n\n'),
    imports: [],
    exports,
  };
}
```

- [ ] **Step 4: Update preprocess.ts to dispatch audio tasks**

In `packages/cli/src/emitters/preprocess.ts`, modify `emitPreprocessBlock` to dispatch audio tasks to the audio emitter. Add at the top:

```typescript
import { emitAudioPreprocessBlock } from './audio-preprocess.js';
```

Then at the start of `emitPreprocessBlock`, before the existing image logic:

```typescript
export function emitPreprocessBlock(config: ResolvedConfig): CodeBlock {
  // Audio tasks use a different preprocessing pipeline
  const audioTasks: string[] = ['speech-to-text', 'audio-classification', 'text-to-speech'];
  if (audioTasks.includes(config.task)) {
    return emitAudioPreprocessBlock(config);
  }

  // Image preprocessing (existing code unchanged)
  const ts = config.lang === 'ts';
  // ... rest of existing function
```

- [ ] **Step 5: Update emitters/index.ts**

Add export for the audio preprocess emitter:

```typescript
export { emitAudioPreprocessBlock } from './audio-preprocess.js';
```

- [ ] **Step 6: Run tests**

Run: `cd packages/cli && npx vitest run tests/framework-emitters.test.ts -t "audio preprocess emitter"`
Expected: PASS

Run: `cd packages/cli && npx vitest run`
Expected: All existing tests still PASS (image tasks unchanged)

- [ ] **Step 7: Commit**

```bash
git add packages/cli/src/emitters/audio-preprocess.ts packages/cli/src/emitters/preprocess.ts packages/cli/src/emitters/index.ts
git commit -m "feat(cli): add audio preprocess emitter with FFT, mel spectrogram, MFCC"
```

---

### Task 5: Enhanced Mic Capture with AudioWorklet

**Files:**
- Modify: `packages/cli/src/emitters/input.ts`

The current mic capture uses `AnalyserNode.getFloatTimeDomainData()` which gives a single snapshot buffer. For audio ML tasks, we need continuous PCM capture via AudioWorklet with configurable sample windows. Per CEO plan Decision #22: separate .js file for worklet processor, ScriptProcessorNode fallback.

- [ ] **Step 1: Write the failing test**

Add to `packages/cli/tests/framework-emitters.test.ts`:

```typescript
describe('audio input emitter', () => {
  it('emits AudioWorklet processor for mic input with audio task', () => {
    const config = makeConfig({ task: 'speech-to-text', input: 'mic', engine: 'ort' });
    const blocks = emitLayer1(config);
    const inputBlock = blocks.find((b) => b.id === 'input');
    expect(inputBlock).toBeDefined();
    expect(inputBlock!.code).toContain('AudioWorkletProcessor');
    expect(inputBlock!.code).toContain('function startAudioCapture(');
    expect(inputBlock!.code).toContain('function createAudioInferenceLoop(');
    expect(inputBlock!.auxiliaryFiles).toBeDefined();
    expect(inputBlock!.auxiliaryFiles!.length).toBe(1);
    expect(inputBlock!.auxiliaryFiles![0].path).toContain('audio-processor');
  });

  it('emits stopStream for mic cleanup', () => {
    const config = makeConfig({ task: 'audio-classification', input: 'mic', engine: 'ort' });
    const blocks = emitLayer1(config);
    const inputBlock = blocks.find((b) => b.id === 'input');
    expect(inputBlock!.exports).toContain('stopStream');
    expect(inputBlock!.exports).toContain('startAudioCapture');
    expect(inputBlock!.exports).toContain('createAudioInferenceLoop');
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/cli && npx vitest run tests/framework-emitters.test.ts -t "audio input emitter"`
Expected: FAIL — no AudioWorklet code in input block

- [ ] **Step 3: Add AudioWorklet capture and audio inference loop to input.ts**

In `packages/cli/src/emitters/input.ts`, add new functions:

```typescript
/** Emit the AudioWorklet processor code (written to separate .js file) */
function emitAudioWorkletProcessor(): string {
  return `/**
 * AudioWorklet processor for continuous PCM capture.
 * Posts raw Float32Array buffers to the main thread.
 */
class AudioCaptureProcessor extends AudioWorkletProcessor {
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (input && input[0] && input[0].length > 0) {
      this.port.postMessage(new Float32Array(input[0]));
    }
    return true;
  }
}

registerProcessor('audio-capture-processor', AudioCaptureProcessor);
`;
}

/** Emit startAudioCapture: AudioWorklet-based continuous capture with ring buffer */
function emitStartAudioCapture(ts: boolean): string {
  const t = ts;
  const returnType = t
    ? `: Promise<{
  stream: MediaStream;
  buffer: Float32Array;
  getSamples: () => Float32Array;
  audioContext: AudioContext;
}>`
    : '';
  return `/**
 * Start continuous audio capture via AudioWorklet.
 * Captures raw PCM samples into a ring buffer.
 *
 * @param sampleRate - Target sample rate (default: 16000)
 * @param bufferSeconds - Ring buffer duration in seconds (default: 30)
 * @returns Object with stream, current buffer, and getSamples() to read the buffer
 */
async function startAudioCapture(
  sampleRate${t ? ': number' : ''} = 16000,
  bufferSeconds${t ? ': number' : ''} = 30
)${returnType} {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const audioContext = new AudioContext({ sampleRate });
  const source = audioContext.createMediaStreamSource(stream);

  const bufferSize = sampleRate * bufferSeconds;
  const buffer = new Float32Array(bufferSize);
  let writePos = 0;
  let filled = false;

  try {
    await audioContext.audioWorklet.addModule('audio-processor.js');
    const workletNode = new AudioWorkletNode(audioContext, 'audio-capture-processor');
    workletNode.port.onmessage = (e${t ? ': MessageEvent<Float32Array>' : ''}) => {
      const samples = e.data;
      for (let i = 0; i < samples.length; i++) {
        buffer[writePos] = samples[i];
        writePos = (writePos + 1) % bufferSize;
        if (writePos === 0) filled = true;
      }
    };
    source.connect(workletNode);
    workletNode.connect(audioContext.destination);
  } catch {
    // Fallback: ScriptProcessorNode (deprecated but widely supported)
    console.warn('AudioWorklet unavailable, using ScriptProcessorNode fallback');
    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (e${t ? ': AudioProcessingEvent' : ''}) => {
      const samples = e.inputBuffer.getChannelData(0);
      for (let i = 0; i < samples.length; i++) {
        buffer[writePos] = samples[i];
        writePos = (writePos + 1) % bufferSize;
        if (writePos === 0) filled = true;
      }
    };
    source.connect(processor);
    processor.connect(audioContext.destination);
  }

  function getSamples()${t ? ': Float32Array' : ''} {
    if (!filled) {
      return buffer.slice(0, writePos);
    }
    // Unwrap ring buffer: [writePos..end, 0..writePos]
    const result = new Float32Array(bufferSize);
    result.set(buffer.subarray(writePos), 0);
    result.set(buffer.subarray(0, writePos), bufferSize - writePos);
    return result;
  }

  return { stream, buffer, getSamples, audioContext };
}`;
}

/** Emit audio inference loop: interval-based (not RAF since there's no video) */
function emitAudioInferenceLoop(ts: boolean): string {
  const t = ts;
  const optsType = t
    ? `: {
  getSamples: () => Float32Array;
  onResult: (result: unknown, elapsed: number) => void;
  intervalMs?: number;
}`
    : '';
  return `/**
 * Create an interval-based audio inference loop.
 * Reads the latest audio samples, runs inference, and calls onResult.
 *
 * Unlike video inference loop (which uses requestAnimationFrame),
 * audio inference runs on a fixed interval since there's no frame to sync with.
 *
 * @returns Object with start() and stop() methods.
 */
function createAudioInferenceLoop(opts${optsType})${t ? ': { start: () => void; stop: () => void }' : ''} {
  let timerId${t ? ': number' : ''} = 0;
  const intervalMs = opts.intervalMs || 1000;

  async function tick() {
    const samples = opts.getSamples();
    if (samples.length === 0) return;

    const start = performance.now();
    const result = await processAudio(samples);
    const elapsed = performance.now() - start;

    opts.onResult(result, elapsed);
  }

  return {
    start() {
      tick();
      timerId = window.setInterval(tick, intervalMs);
    },
    stop() {
      window.clearInterval(timerId);
    },
  };
}`;
}
```

Then update the `mic` case in `emitInputBlock`:

```typescript
    case 'mic': {
      const isAudioTask = ['speech-to-text', 'audio-classification'].includes(config.task);

      if (isAudioTask) {
        parts.push(emitStartAudioCapture(ts));
        parts.push(emitAudioInferenceLoop(ts));
        parts.push(emitStopStream(ts));
        exports.push('startAudioCapture', 'createAudioInferenceLoop', 'stopStream');

        return {
          id: 'input',
          code: parts.join('\n\n'),
          imports: [],
          exports,
          auxiliaryFiles: [
            {
              path: 'audio-processor.js',
              content: emitAudioWorkletProcessor(),
            },
          ],
        };
      }

      // Non-audio task with mic (legacy path)
      parts.push(emitStartMicrophone(ts));
      parts.push(emitCaptureAudio(ts));
      parts.push(emitStopStream(ts));
      exports.push('startMicrophone', 'captureAudio', 'stopStream');
      break;
    }
```

- [ ] **Step 4: Run tests**

Run: `cd packages/cli && npx vitest run tests/framework-emitters.test.ts -t "audio input emitter"`
Expected: PASS

Run: `cd packages/cli && npx vitest run`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add packages/cli/src/emitters/input.ts
git commit -m "feat(cli): enhanced mic capture with AudioWorklet and audio inference loop"
```

---

### Task 6: Audio Postprocessing Emitters

**Files:**
- Modify: `packages/cli/src/emitters/postprocess.ts`

Speech-to-text needs a greedy CTC decoder. Audio-classification already works (reuses softmax + topK). TTS outputs raw audio waveform that needs conversion to playable format.

- [ ] **Step 1: Write the failing tests**

Add to `packages/cli/tests/framework-emitters.test.ts`:

```typescript
describe('audio postprocess emitter', () => {
  it('emits greedy decoder for speech-to-text', () => {
    const config = makeConfig({ task: 'speech-to-text', input: 'file', engine: 'ort' });
    const blocks = emitLayer1(config);
    const postBlock = blocks.find((b) => b.id === 'postprocess');
    expect(postBlock).toBeDefined();
    expect(postBlock!.code).toContain('function greedyDecode(');
    expect(postBlock!.code).toContain('function postprocessTranscript(');
    expect(postBlock!.exports).toContain('postprocessTranscript');
  });

  it('emits audio output for text-to-speech', () => {
    const config = makeConfig({ task: 'text-to-speech', input: 'file', engine: 'ort' });
    const blocks = emitLayer1(config);
    const postBlock = blocks.find((b) => b.id === 'postprocess');
    expect(postBlock).toBeDefined();
    expect(postBlock!.code).toContain('function playAudio(');
    expect(postBlock!.code).toContain('function postprocessAudio(');
    expect(postBlock!.exports).toContain('postprocessAudio');
    expect(postBlock!.exports).toContain('playAudio');
  });

  it('reuses classification postprocessing for audio-classification', () => {
    const config = makeConfig({ task: 'audio-classification', input: 'file', engine: 'ort' });
    const blocks = emitLayer1(config);
    const postBlock = blocks.find((b) => b.id === 'postprocess');
    expect(postBlock!.code).toContain('function softmax(');
    expect(postBlock!.code).toContain('function topK(');
    expect(postBlock!.code).toContain('function postprocessResults(');
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/cli && npx vitest run tests/framework-emitters.test.ts -t "audio postprocess emitter"`
Expected: FAIL — no greedy decoder or audio output in postprocess block

- [ ] **Step 3: Add STT and TTS postprocessing to postprocess.ts**

Add these functions in `packages/cli/src/emitters/postprocess.ts`:

```typescript
// ---- Speech-to-Text: Greedy CTC decoder ----

function emitGreedyDecode(ts: boolean): string {
  const t = ts;
  return `/**
 * Greedy CTC decoder: take argmax at each timestep and collapse repeats.
 * Removes blank tokens (index 0 by convention).
 *
 * @param logits - Model output, shape [1, T, vocab_size]
 * @param vocabSize - Size of the vocabulary
 * @param blankIndex - Index of the CTC blank token (default: 0)
 * @returns Array of token indices
 */
function greedyDecode(
  logits${t ? ': Float32Array' : ''},
  numTimesteps${t ? ': number' : ''},
  vocabSize${t ? ': number' : ''},
  blankIndex${t ? ': number' : ''} = 0
)${t ? ': number[]' : ''} {
  const tokens${t ? ': number[]' : ''} = [];
  let prevToken = -1;

  for (let t = 0; t < numTimesteps; t++) {
    const offset = t * vocabSize;
    let bestIdx = 0;
    let bestVal = logits[offset];
    for (let v = 1; v < vocabSize; v++) {
      if (logits[offset + v] > bestVal) {
        bestVal = logits[offset + v];
        bestIdx = v;
      }
    }

    // CTC: skip blanks and repeated tokens
    if (bestIdx !== blankIndex && bestIdx !== prevToken) {
      tokens.push(bestIdx);
    }
    prevToken = bestIdx;
  }

  return tokens;
}`;
}

function emitPostprocessTranscript(ts: boolean): string {
  const t = ts;
  return `/**
 * Postprocess speech-to-text model output.
 * Decodes token indices to text using a character vocabulary.
 *
 * @param logits - Raw model output (Float32Array)
 * @param numTimesteps - Number of output timesteps
 * @param vocabSize - Vocabulary size
 * @param vocab - Character array mapping index to character
 * @returns Decoded transcript string
 */
function postprocessTranscript(
  logits${t ? ': Float32Array' : ''},
  numTimesteps${t ? ': number' : ''},
  vocabSize${t ? ': number' : ''},
  vocab${t ? ': string[]' : ''}
)${t ? ': string' : ''} {
  const tokens = greedyDecode(logits, numTimesteps, vocabSize);
  return tokens.map((i) => vocab[i] || '').join('');
}`;
}

// ---- Text-to-Speech: Audio output ----

function emitPlayAudio(ts: boolean): string {
  const t = ts;
  return `/**
 * Play a Float32Array as audio through the speakers.
 * Creates an AudioBuffer and plays it via AudioContext.
 *
 * @param samples - Raw PCM audio samples (mono, -1 to 1 range)
 * @param sampleRate - Sample rate (e.g. 22050 for most TTS models)
 */
async function playAudio(
  samples${t ? ': Float32Array' : ''},
  sampleRate${t ? ': number' : ''} = 22050
)${t ? ': Promise<void>' : ''} {
  const audioCtx = new AudioContext();
  const buffer = audioCtx.createBuffer(1, samples.length, sampleRate);
  buffer.getChannelData(0).set(samples);

  const source = audioCtx.createBufferSource();
  source.buffer = buffer;
  source.connect(audioCtx.destination);
  source.start();

  return new Promise((resolve) => {
    source.onended = () => {
      audioCtx.close();
      resolve();
    };
  });
}`;
}

function emitPostprocessAudio(ts: boolean): string {
  const t = ts;
  return `/**
 * Postprocess TTS model output: extract audio waveform.
 * Most TTS models output raw Float32 PCM samples.
 *
 * @param output - Raw model output (Float32Array)
 * @returns Float32Array of audio samples ready for playback
 */
function postprocessAudio(output${t ? ': Float32Array' : ''})${t ? ': Float32Array' : ''} {
  // Clamp to [-1, 1] range for safe playback
  const samples = new Float32Array(output.length);
  for (let i = 0; i < output.length; i++) {
    samples[i] = Math.max(-1, Math.min(1, output[i]));
  }
  return samples;
}`;
}
```

Then update the switch in `emitPostprocessBlock`:

```typescript
    case 'speech-to-text':
      parts.push(emitGreedyDecode(ts));
      parts.push(emitPostprocessTranscript(ts));
      exports.push('greedyDecode', 'postprocessTranscript');
      break;

    case 'text-to-speech':
      parts.push(emitPlayAudio(ts));
      parts.push(emitPostprocessAudio(ts));
      exports.push('postprocessAudio', 'playAudio');
      break;
```

Remove `speech-to-text` and `text-to-speech` from the default case comment.

- [ ] **Step 4: Run tests**

Run: `cd packages/cli && npx vitest run tests/framework-emitters.test.ts -t "audio postprocess emitter"`
Expected: PASS

Run: `cd packages/cli && npx vitest run`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add packages/cli/src/emitters/postprocess.ts
git commit -m "feat(cli): add STT greedy decoder and TTS audio output postprocessing"
```

---

### Task 7: HTML Framework — Audio Task UI

**Files:**
- Modify: `packages/cli/src/frameworks/html.ts`

Add three new script emitters for audio tasks and update the dispatcher.

- [ ] **Step 1: Write the failing tests**

Add to `packages/cli/tests/framework-emitters.test.ts`:

```typescript
describe('HTML audio task dispatch', () => {
  it('generates audio classification file input UI', () => {
    const config = makeConfig({ task: 'audio-classification', input: 'file', engine: 'ort', framework: 'html' });
    const files = emitHtml(config, emitLayer1(config));
    const html = getFile(files, 'index.html');
    expect(html).toContain('type="file"');
    expect(html).toContain('accept="audio/*"');
    expect(html).toContain('postprocessResults');
    expect(html).toContain('melSpectrogram');
  });

  it('generates speech-to-text mic input UI', () => {
    const config = makeConfig({ task: 'speech-to-text', input: 'mic', engine: 'ort', framework: 'html' });
    const files = emitHtml(config, emitLayer1(config));
    const html = getFile(files, 'index.html');
    expect(html).toContain('startAudioCapture');
    expect(html).toContain('postprocessTranscript');
    expect(html).toContain('transcript');
  });

  it('generates TTS text input UI', () => {
    const config = makeConfig({ task: 'text-to-speech', input: 'file', engine: 'ort', framework: 'html' });
    const files = emitHtml(config, emitLayer1(config));
    const html = getFile(files, 'index.html');
    expect(html).toContain('textarea');
    expect(html).toContain('playAudio');
    expect(html).toContain('Synthesize');
  });

  it('includes AudioWorklet processor file for mic input', () => {
    const config = makeConfig({ task: 'speech-to-text', input: 'mic', engine: 'ort', framework: 'html' });
    const files = emitHtml(config, emitLayer1(config));
    const processorFile = files.find((f) => f.path.includes('audio-processor'));
    expect(processorFile).toBeDefined();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/cli && npx vitest run tests/framework-emitters.test.ts -t "HTML audio task dispatch"`
Expected: FAIL

- [ ] **Step 3: Add audio task scripts and body emitters to html.ts**

Add these functions to `packages/cli/src/frameworks/html.ts`:

```typescript
/** Emit script for audio classification with file input */
function emitFileAudioClassificationScript(
  config: ResolvedConfig,
  blockCode: string,
  modelPath: string,
): string {
  return `${blockCode}

async function main() {
  const statusEl = document.getElementById('status');
  const resultsEl = document.getElementById('results');
  const fileInput = document.getElementById('audioFile');

  statusEl.textContent = 'Loading model...';
  const session = await createSession('${modelPath}');
  statusEl.textContent = 'Ready. Select an audio file.';

  fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    statusEl.textContent = 'Processing audio...';
    const arrayBuffer = await file.arrayBuffer();
    const audioCtx = new OfflineAudioContext(1, 1, 16000);
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

    // Resample to 16kHz mono
    const offlineCtx = new OfflineAudioContext(1, Math.ceil(audioBuffer.duration * 16000), 16000);
    const source = offlineCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(offlineCtx.destination);
    source.start();
    const resampled = await offlineCtx.startRendering();
    const samples = resampled.getChannelData(0);

    const { data, numFrames } = melSpectrogram(samples, 16000, 512, 160, 64);
    const mfccData = mfcc(data, numFrames, 64, 13);

    const start = performance.now();
    const output = await runInference(session, new Float32Array(mfccData));
    const elapsed = performance.now() - start;

    const { indices, values } = postprocessResults(output);
    statusEl.textContent = \`Inference: \${elapsed.toFixed(1)}ms · \${getBackendLabel(session)}\`;

    resultsEl.innerHTML = indices.map((idx, i) =>
      \`<div class="result-row">
        <span class="result-label">Class \${idx}</span>
        <div class="result-bar" style="width: \${(values[i] * 100).toFixed(1)}%"></div>
        <span class="result-score">\${(values[i] * 100).toFixed(1)}%</span>
      </div>\`
    ).join('');
  });
}

main();`;
}

/** Emit script for speech-to-text with file input */
function emitFileSpeechToTextScript(
  config: ResolvedConfig,
  blockCode: string,
  modelPath: string,
): string {
  return `${blockCode}

async function main() {
  const statusEl = document.getElementById('status');
  const transcriptEl = document.getElementById('transcript');
  const fileInput = document.getElementById('audioFile');

  statusEl.textContent = 'Loading model...';
  const session = await createSession('${modelPath}');
  statusEl.textContent = 'Ready. Select an audio file.';

  fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    statusEl.textContent = 'Transcribing...';
    const arrayBuffer = await file.arrayBuffer();
    const audioCtx = new OfflineAudioContext(1, 1, 16000);
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

    const offlineCtx = new OfflineAudioContext(1, Math.ceil(audioBuffer.duration * 16000), 16000);
    const source = offlineCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(offlineCtx.destination);
    source.start();
    const resampled = await offlineCtx.startRendering();
    const samples = resampled.getChannelData(0);

    const { data } = melSpectrogram(samples, 16000, 512, 160, 80);

    const start = performance.now();
    const output = await runInference(session, new Float32Array(data));
    const elapsed = performance.now() - start;

    const numTimesteps = Math.floor(output.length / vocab.length);
    const text = postprocessTranscript(output, numTimesteps, vocab.length, vocab);
    statusEl.textContent = \`Inference: \${elapsed.toFixed(1)}ms · \${getBackendLabel(session)}\`;
    transcriptEl.textContent = text || '(no speech detected)';
  });
}

// Default vocabulary (override with model-specific vocab)
const vocab = [' ', ...Array.from({ length: 26 }, (_, i) => String.fromCharCode(97 + i)), "'"];

main();`;
}

/** Emit script for speech-to-text with mic input (realtime) */
function emitRealtimeSpeechToTextScript(
  config: ResolvedConfig,
  blockCode: string,
  modelPath: string,
): string {
  return `${blockCode}

async function main() {
  const statusEl = document.getElementById('status');
  const transcriptEl = document.getElementById('transcript');
  const startBtn = document.getElementById('startBtn');
  const stopBtn = document.getElementById('stopBtn');

  statusEl.textContent = 'Loading model...';
  const session = await createSession('${modelPath}');
  statusEl.textContent = 'Ready. Click Start to begin recording.';

  let audioCapture = null;
  let inferenceLoop = null;

  async function processAudio(samples) {
    const { data } = melSpectrogram(samples, 16000, 512, 160, 80);
    const output = await runInference(session, new Float32Array(data));
    const numTimesteps = Math.floor(output.length / vocab.length);
    return postprocessTranscript(output, numTimesteps, vocab.length, vocab);
  }

  startBtn.addEventListener('click', async () => {
    audioCapture = await startAudioCapture(16000, 30);
    inferenceLoop = createAudioInferenceLoop({
      getSamples: audioCapture.getSamples,
      onResult: (text, elapsed) => {
        transcriptEl.textContent = text || '(listening...)';
        statusEl.textContent = \`Inference: \${elapsed.toFixed(1)}ms · \${getBackendLabel(session)}\`;
      },
      intervalMs: 2000,
    });
    inferenceLoop.start();
    startBtn.disabled = true;
    stopBtn.disabled = false;
  });

  stopBtn.addEventListener('click', () => {
    if (inferenceLoop) inferenceLoop.stop();
    if (audioCapture) {
      stopStream(audioCapture.stream);
      audioCapture.audioContext.close();
    }
    startBtn.disabled = false;
    stopBtn.disabled = true;
  });
}

const vocab = [' ', ...Array.from({ length: 26 }, (_, i) => String.fromCharCode(97 + i)), "'"];

main();`;
}

/** Emit script for text-to-speech */
function emitTextToSpeechScript(
  config: ResolvedConfig,
  blockCode: string,
  modelPath: string,
): string {
  return `${blockCode}

async function main() {
  const statusEl = document.getElementById('status');
  const textInput = document.getElementById('textInput');
  const synthesizeBtn = document.getElementById('synthesizeBtn');

  statusEl.textContent = 'Loading model...';
  const session = await createSession('${modelPath}');
  statusEl.textContent = 'Ready. Enter text and click Synthesize.';

  synthesizeBtn.addEventListener('click', async () => {
    const text = textInput.value.trim();
    if (!text) return;

    statusEl.textContent = 'Synthesizing...';
    synthesizeBtn.disabled = true;

    // Simple character-level tokenization (model-specific tokenizer should replace this)
    const inputData = new Float32Array(text.split('').map((c) => c.charCodeAt(0)));

    const start = performance.now();
    const output = await runInference(session, inputData);
    const elapsed = performance.now() - start;

    const audioSamples = postprocessAudio(output);
    statusEl.textContent = \`Synthesized in \${elapsed.toFixed(1)}ms · \${getBackendLabel(session)}\`;
    synthesizeBtn.disabled = false;

    await playAudio(audioSamples);
  });
}

main();`;
}
```

Add body emitters for audio tasks:

```typescript
/** Body for audio file input (classification, STT) */
function emitAudioFileBody(config: ResolvedConfig): string {
  const label = TASK_LABELS[config.task] || config.task;
  return `    <header>
      <h1 aria-live="polite">${label}</h1>
    </header>
    <main>
      <a class="skip-link" href="#results">Skip to results</a>
      <div class="drop-zone" role="region" aria-label="Audio input">
        <label for="audioFile" class="drop-zone-label">
          <span class="drop-zone-icon" aria-hidden="true">&#9835;</span>
          <span>Choose an audio file</span>
          <input type="file" id="audioFile" accept="audio/*" />
        </label>
      </div>
      <div id="results" class="results" role="region" aria-label="Results" aria-live="polite"></div>
      ${config.task === 'speech-to-text' ? '<pre id="transcript" class="transcript" role="region" aria-label="Transcript" aria-live="polite"></pre>' : ''}
      <div id="status" class="status-bar" role="status" aria-live="polite">Initializing...</div>
    </main>`;
}

/** Body for mic-based audio tasks */
function emitAudioMicBody(config: ResolvedConfig): string {
  const label = TASK_LABELS[config.task] || config.task;
  return `    <header>
      <h1 aria-live="polite">${label}</h1>
    </header>
    <main>
      <a class="skip-link" href="#results">Skip to results</a>
      <div class="controls" role="region" aria-label="Recording controls">
        <button id="startBtn" type="button">Start Recording</button>
        <button id="stopBtn" type="button" disabled>Stop Recording</button>
      </div>
      <div id="results" class="results" role="region" aria-label="Results" aria-live="polite"></div>
      ${config.task === 'speech-to-text' ? '<pre id="transcript" class="transcript" role="region" aria-label="Transcript" aria-live="polite"></pre>' : ''}
      <div id="status" class="status-bar" role="status" aria-live="polite">Initializing...</div>
    </main>`;
}

/** Body for TTS */
function emitTtsBody(): string {
  return `    <header>
      <h1 aria-live="polite">Text to Speech</h1>
    </header>
    <main>
      <div class="tts-input" role="region" aria-label="Text input">
        <label for="textInput">Enter text to synthesize:</label>
        <textarea id="textInput" rows="4" placeholder="Type something..."></textarea>
        <button id="synthesizeBtn" type="button">Synthesize</button>
      </div>
      <div id="status" class="status-bar" role="status" aria-live="polite">Initializing...</div>
    </main>`;
}
```

Then update `emitAppScript` to dispatch audio tasks:

```typescript
function emitAppScript(config: ResolvedConfig, blocks: CodeBlock[]): string {
  const blockCode = emitBlockCode(config, blocks);
  const modelPath = getModelPath(config);

  // Audio tasks
  if (config.task === 'audio-classification') {
    if (config.input === 'mic') {
      return emitRealtimeAudioClassificationScript(config, blockCode, modelPath);
    }
    return emitFileAudioClassificationScript(config, blockCode, modelPath);
  }
  if (config.task === 'speech-to-text') {
    if (config.input === 'mic') {
      return emitRealtimeSpeechToTextScript(config, blockCode, modelPath);
    }
    return emitFileSpeechToTextScript(config, blockCode, modelPath);
  }
  if (config.task === 'text-to-speech') {
    return emitTextToSpeechScript(config, blockCode, modelPath);
  }

  // Image tasks (existing dispatch, unchanged)
  // ...
}
```

And update `emitAppBody` to dispatch audio tasks:

```typescript
function emitAppBody(config: ResolvedConfig): string {
  if (config.task === 'text-to-speech') return emitTtsBody();
  if (['speech-to-text', 'audio-classification'].includes(config.task)) {
    return config.input === 'mic'
      ? emitAudioMicBody(config)
      : emitAudioFileBody(config);
  }
  // Existing image dispatch...
}
```

Add CSS for audio-specific UI in the `emitAudioCSS` helper (add to the `<style>` block):

```css
.transcript {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem;
  min-height: 100px;
  white-space: pre-wrap;
  font-family: inherit;
}

.controls {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1rem;
}

.controls button {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 1rem;
  background: var(--accent);
  color: white;
}

.controls button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.tts-input {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.tts-input textarea {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text);
  padding: 0.75rem;
  font-size: 1rem;
  resize: vertical;
}
```

In `emitHtml`, handle auxiliary files from input blocks (the AudioWorklet processor file):

```typescript
export function emitHtml(config: ResolvedConfig, blocks: CodeBlock[]): GeneratedFile[] {
  // ... existing code ...
  const files: GeneratedFile[] = [
    { path: 'index.html', content: html },
    { path: 'README.md', content: readme },
  ];

  // Include auxiliary files from Layer 1 blocks (e.g. AudioWorklet processor)
  for (const block of blocks) {
    if (block.auxiliaryFiles) {
      for (const aux of block.auxiliaryFiles) {
        files.push(aux);
      }
    }
  }

  return files;
}
```

- [ ] **Step 4: Run tests**

Run: `cd packages/cli && npx vitest run tests/framework-emitters.test.ts -t "HTML audio task dispatch"`
Expected: PASS

Run: `cd packages/cli && npx vitest run`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add packages/cli/src/frameworks/html.ts
git commit -m "feat(cli): HTML framework audio task UI (STT, audio classification, TTS)"
```

---

### Task 8: Multi-Framework Audio Support

**Files:**
- Modify: `packages/cli/src/frameworks/react-vite.ts`
- Modify: `packages/cli/src/frameworks/vanilla-vite.ts`
- Modify: `packages/cli/src/frameworks/nextjs.ts`
- Modify: `packages/cli/src/frameworks/sveltekit.ts`
- Modify: `packages/cli/src/frameworks/shared.ts`

Each framework emitter needs audio task routing. The pattern is the same across all frameworks: extract audio code blocks into lib modules, add audio-specific components/pages, include auxiliary files.

- [ ] **Step 1: Write the failing tests**

Add to `packages/cli/tests/framework-emitters.test.ts`:

```typescript
describe.each([
  ['react-vite', emitReactVite],
  ['vanilla-vite', emitVanillaVite],
  ['nextjs', emitNextjs],
  ['sveltekit', emitSvelteKit],
] as const)('%s audio tasks', (framework, emitFn) => {
  it('generates audio classification file input', () => {
    const config = makeConfig({ task: 'audio-classification', input: 'file', engine: 'ort', framework });
    const files = emitFn(config, emitLayer1(config));
    const mainFile = files.find((f) => f.content.includes('postprocessResults'));
    expect(mainFile).toBeDefined();
    expect(mainFile!.content).toContain('melSpectrogram');
  });

  it('generates speech-to-text with mic input', () => {
    const config = makeConfig({ task: 'speech-to-text', input: 'mic', engine: 'ort', framework });
    const files = emitFn(config, emitLayer1(config));
    const mainFile = files.find((f) => f.content.includes('postprocessTranscript'));
    expect(mainFile).toBeDefined();
    expect(mainFile!.content).toContain('startAudioCapture');
  });

  it('generates text-to-speech UI', () => {
    const config = makeConfig({ task: 'text-to-speech', input: 'file', engine: 'ort', framework });
    const files = emitFn(config, emitLayer1(config));
    const mainFile = files.find((f) => f.content.includes('playAudio'));
    expect(mainFile).toBeDefined();
  });

  it('includes AudioWorklet processor for mic input', () => {
    const config = makeConfig({ task: 'speech-to-text', input: 'mic', engine: 'ort', framework });
    const files = emitFn(config, emitLayer1(config));
    const processorFile = files.find((f) => f.path.includes('audio-processor'));
    expect(processorFile).toBeDefined();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/cli && npx vitest run tests/framework-emitters.test.ts -t "audio tasks"`
Expected: FAIL for all 4 frameworks

- [ ] **Step 3: Update shared.ts with audio helpers**

In `packages/cli/src/frameworks/shared.ts`, add audio CSS to `emitAppCSS()`:

```typescript
// Add these rules to the CSS string returned by emitAppCSS():
.transcript {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem;
  min-height: 100px;
  white-space: pre-wrap;
}

.controls {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1rem;
}

.controls button {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  background: var(--accent);
  color: white;
}

.controls button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.tts-input {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.tts-input textarea {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text);
  padding: 0.75rem;
  resize: vertical;
}
```

Add a helper to collect auxiliary files from blocks:

```typescript
/** Collect all auxiliary files from code blocks */
export function collectAuxiliaryFiles(blocks: CodeBlock[]): GeneratedFile[] {
  const files: GeneratedFile[] = [];
  for (const block of blocks) {
    if (block.auxiliaryFiles) {
      files.push(...block.auxiliaryFiles);
    }
  }
  return files;
}

/** Check if config is for an audio task */
export function isAudioTask(task: string): boolean {
  return ['speech-to-text', 'audio-classification', 'text-to-speech'].includes(task);
}
```

- [ ] **Step 4: Update each framework emitter**

For each of the 4 framework emitters (`react-vite.ts`, `vanilla-vite.ts`, `nextjs.ts`, `sveltekit.ts`), apply the same pattern:

1. Import `collectAuxiliaryFiles` and `isAudioTask` from `./shared.js`
2. In the main `emit*` function, after generating files, append auxiliary files:
   ```typescript
   files.push(...collectAuxiliaryFiles(blocks));
   ```
3. Add audio task routing in the main component/page logic. For each framework:

**react-vite.ts**: Add audio task components (`AudioClassificationApp`, `SpeechToTextApp`, `TextToSpeechApp`) that use `useEffect` + `useRef` for audio capture, call the preprocessing/inference/postprocessing lib functions. Route in `App.tsx` based on task.

**vanilla-vite.ts**: Add audio task `main()` functions that set up event listeners on file input, mic buttons, or textarea. Route in `main.ts` based on task config constant.

**nextjs.ts**: Add `'use client'` audio components. Route in `page.tsx` based on task.

**sveltekit.ts**: Add Svelte 5 audio components using `$effect` and `$state` runes. Route in `+page.svelte` based on task.

The audio task UI logic for each framework follows the same pattern as the HTML framework (Task 7) but adapted to each framework's component model. The key code is:

- **File audio input:** `<input type="file" accept="audio/*">` → `decodeAudioData` → mel spectrogram → inference → display results
- **Mic input:** `startAudioCapture()` → `createAudioInferenceLoop()` → display streaming results
- **TTS:** `<textarea>` → tokenize → inference → `playAudio()`

- [ ] **Step 5: Run tests**

Run: `cd packages/cli && npx vitest run tests/framework-emitters.test.ts -t "audio tasks"`
Expected: PASS for all 4 frameworks

Run: `cd packages/cli && npx vitest run`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add packages/cli/src/frameworks/shared.ts packages/cli/src/frameworks/react-vite.ts packages/cli/src/frameworks/vanilla-vite.ts packages/cli/src/frameworks/nextjs.ts packages/cli/src/frameworks/sveltekit.ts
git commit -m "feat(cli): audio task support across all framework emitters"
```

---

### Task 9: Audio Snapshot Tests

**Files:**
- Modify: `packages/cli/tests/snapshot.test.ts`

Add snapshot coverage for audio task × framework × input × engine combinations.

- [ ] **Step 1: Add audio snapshot test cases**

Add these test cases to `packages/cli/tests/snapshot.test.ts`:

```typescript
// Audio Classification
it('audio-classification + html + file + ort', () => {
  const config = makeConfig({
    task: 'audio-classification',
    input: 'file',
    engine: 'ort',
    framework: 'html',
    lang: 'js',
  });
  const files = assemble(config);
  expect(serializeFiles(files)).toMatchSnapshot();
});

it('audio-classification + html + mic + ort', () => {
  const config = makeConfig({
    task: 'audio-classification',
    input: 'mic',
    engine: 'ort',
    framework: 'html',
    lang: 'js',
  });
  const files = assemble(config);
  expect(serializeFiles(files)).toMatchSnapshot();
});

it('audio-classification + react-vite + file + ort', () => {
  const config = makeConfig({
    task: 'audio-classification',
    input: 'file',
    engine: 'ort',
    framework: 'react-vite',
    lang: 'js',
  });
  const files = assemble(config);
  expect(serializeFiles(files)).toMatchSnapshot();
});

// Speech-to-Text
it('speech-to-text + html + file + ort', () => {
  const config = makeConfig({
    task: 'speech-to-text',
    input: 'file',
    engine: 'ort',
    framework: 'html',
    lang: 'js',
  });
  const files = assemble(config);
  expect(serializeFiles(files)).toMatchSnapshot();
});

it('speech-to-text + html + mic + ort', () => {
  const config = makeConfig({
    task: 'speech-to-text',
    input: 'mic',
    engine: 'ort',
    framework: 'html',
    lang: 'js',
  });
  const files = assemble(config);
  expect(serializeFiles(files)).toMatchSnapshot();
});

it('speech-to-text + vanilla-vite + mic + ort + ts', () => {
  const config = makeConfig({
    task: 'speech-to-text',
    input: 'mic',
    engine: 'ort',
    framework: 'vanilla-vite',
    lang: 'ts',
  });
  const files = assemble(config);
  expect(serializeFiles(files)).toMatchSnapshot();
});

// Text-to-Speech
it('text-to-speech + html + file + ort', () => {
  const config = makeConfig({
    task: 'text-to-speech',
    input: 'file',
    engine: 'ort',
    framework: 'html',
    lang: 'js',
  });
  const files = assemble(config);
  expect(serializeFiles(files)).toMatchSnapshot();
});

it('text-to-speech + nextjs + file + ort + ts', () => {
  const config = makeConfig({
    task: 'text-to-speech',
    input: 'file',
    engine: 'ort',
    framework: 'nextjs',
    lang: 'ts',
  });
  const files = assemble(config);
  expect(serializeFiles(files)).toMatchSnapshot();
});

// Audio + offline
it('audio-classification + html + file + ort + offline', () => {
  const config = makeConfig({
    task: 'audio-classification',
    input: 'file',
    engine: 'ort',
    framework: 'html',
    lang: 'js',
    offline: true,
  });
  const files = assemble(config);
  expect(serializeFiles(files)).toMatchSnapshot();
});
```

- [ ] **Step 2: Generate snapshots**

Run: `cd packages/cli && npx vitest run tests/snapshot.test.ts --update`
Expected: 9 new snapshots written

- [ ] **Step 3: Verify snapshots pass**

Run: `cd packages/cli && npx vitest run tests/snapshot.test.ts`
Expected: All snapshot tests PASS

- [ ] **Step 4: Run full test suite**

Run: `cd packages/cli && npx vitest run`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add packages/cli/tests/snapshot.test.ts packages/cli/tests/__snapshots__/
git commit -m "test(cli): add audio task snapshot tests (9 combos)"
```

---

### Task 10: Compare Command

**Files:**
- Create: `packages/cli/src/compare.ts`
- Modify: `packages/cli/src/cli.ts`

Per CEO plan: `webai compare ./model.onnx` generates a single HTML benchmark page. Metrics: first-inference latency (cold), steady-state latency (warm, avg of 10), throughput (inferences/sec), peak JS heap. Also `--json` for machine-readable output. Phase 2 scope: ONNX Runtime Web backends only (WASM, WebGPU, WebNN).

The compare command doesn't run benchmarks at generation time. It generates an HTML page that, when opened in a browser, runs the benchmarks client-side and displays results.

- [ ] **Step 1: Write the failing test**

Create `packages/cli/tests/compare.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { generateCompareHtml, generateCompareJson } from '../src/compare.js';

describe('compare command', () => {
  const modelMeta = {
    format: 'onnx' as const,
    inputs: [{ name: 'input', shape: [1, 3, 224, 224], dtype: 'float32' }],
    outputs: [{ name: 'output', shape: [1, 1000], dtype: 'float32' }],
    opsetVersion: 13,
  };

  describe('generateCompareHtml', () => {
    it('produces valid HTML with benchmark script', () => {
      const html = generateCompareHtml('model.onnx', modelMeta);
      expect(html).toContain('<!DOCTYPE html>');
      expect(html).toContain('onnxruntime-web');
      expect(html).toContain('async function benchmark(');
      expect(html).toContain('wasm');
      expect(html).toContain('webgpu');
      expect(html).toContain('webnn');
    });

    it('includes all metric categories', () => {
      const html = generateCompareHtml('model.onnx', modelMeta);
      expect(html).toContain('Cold Start');
      expect(html).toContain('Warm Average');
      expect(html).toContain('Throughput');
    });

    it('uses correct model input shape', () => {
      const html = generateCompareHtml('model.onnx', modelMeta);
      expect(html).toContain('[1, 3, 224, 224]');
    });

    it('handles model URL', () => {
      const html = generateCompareHtml('https://hf.co/model.onnx', modelMeta);
      expect(html).toContain('https://hf.co/model.onnx');
    });
  });

  describe('generateCompareJson', () => {
    it('produces JSON template structure', () => {
      const json = generateCompareJson('model.onnx', modelMeta);
      const parsed = JSON.parse(json);
      expect(parsed.model).toBe('model.onnx');
      expect(parsed.backends).toEqual(['wasm', 'webgpu', 'webnn']);
      expect(parsed.metrics).toContain('cold_latency_ms');
      expect(parsed.metrics).toContain('warm_latency_ms');
      expect(parsed.metrics).toContain('throughput_ips');
    });
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/cli && npx vitest run tests/compare.test.ts`
Expected: FAIL — module not found

- [ ] **Step 3: Implement compare.ts**

```typescript
// packages/cli/src/compare.ts

/**
 * Compare command: generates a benchmark page for comparing
 * ONNX Runtime Web backends (WASM, WebGPU, WebNN).
 *
 * The generated HTML runs benchmarks client-side in the browser.
 * Metrics: cold start latency, warm average latency, throughput, peak heap.
 */

import type { ModelMetadata } from '@webai/core';

const ORT_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

const BACKENDS = ['wasm', 'webgpu', 'webnn'] as const;

interface CompareJsonTemplate {
  model: string;
  backends: string[];
  metrics: string[];
  inputShape: (number | string)[];
  results: null;
}

/**
 * Generate the benchmark HTML page.
 */
export function generateCompareHtml(
  modelPath: string,
  modelMeta: ModelMetadata,
): string {
  const inputShape = modelMeta.inputs[0]?.shape ?? [1, 3, 224, 224];
  const shapeStr = JSON.stringify(inputShape);
  const inputName = modelMeta.inputs[0]?.name ?? 'input';

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>webai compare — ${modelPath}</title>
  <script src="${ORT_CDN}ort.min.js"></script>
  <style>
    :root {
      --bg: #0a0a0a; --card: #141414; --border: #2a2a2a;
      --text: #e0e0e0; --muted: #888; --accent: #3b82f6;
      --green: #22c55e; --yellow: #eab308; --red: #ef4444;
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; background: var(--bg); color: var(--text); padding: 2rem; }
    h1 { margin-bottom: 0.5rem; }
    .subtitle { color: var(--muted); margin-bottom: 2rem; }
    .status { padding: 1rem; background: var(--card); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 2rem; }
    table { width: 100%; border-collapse: collapse; background: var(--card); border-radius: 8px; overflow: hidden; }
    th, td { padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid var(--border); }
    th { color: var(--muted); font-weight: 500; font-size: 0.875rem; text-transform: uppercase; }
    .bar-cell { position: relative; }
    .bar { height: 24px; border-radius: 4px; transition: width 0.3s ease; }
    .bar-wasm { background: var(--accent); }
    .bar-webgpu { background: var(--green); }
    .bar-webnn { background: var(--yellow); }
    .skipped { color: var(--muted); font-style: italic; }
    .best { font-weight: 700; }
    .error { color: var(--red); }
    .charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; margin-top: 2rem; }
    .chart { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; }
    .chart h3 { font-size: 0.875rem; color: var(--muted); margin-bottom: 0.75rem; }
    .chart-bar { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem; }
    .chart-label { width: 70px; font-size: 0.875rem; }
    .chart-value { font-size: 0.875rem; color: var(--muted); }
  </style>
</head>
<body>
  <h1>webai compare</h1>
  <p class="subtitle">Model: ${modelPath} · Input: ${shapeStr}</p>
  <div class="status" id="status">Initializing benchmarks...</div>

  <table>
    <thead>
      <tr>
        <th>Backend</th>
        <th>Status</th>
        <th>Cold Start (ms)</th>
        <th>Warm Avg (ms)</th>
        <th>Throughput (inf/s)</th>
        <th>Peak Heap (MB)</th>
      </tr>
    </thead>
    <tbody id="results">
      <tr><td colspan="6">Running benchmarks...</td></tr>
    </tbody>
  </table>

  <div class="charts" id="charts"></div>

  <script>
    const MODEL_PATH = '${modelPath}';
    const INPUT_SHAPE = ${shapeStr};
    const INPUT_NAME = '${inputName}';
    const WARM_RUNS = 10;

    async function benchmark(providerConfig, label) {
      const providers = Array.isArray(providerConfig) ? providerConfig : [providerConfig];
      const result = { backend: label, status: 'running', cold: 0, warm: 0, throughput: 0, heap: 0 };

      try {
        // Cold start: session creation + first inference
        const heapBefore = performance.memory ? performance.memory.usedJSHeapSize : 0;
        const coldStart = performance.now();
        const session = await ort.InferenceSession.create(MODEL_PATH, { executionProviders: providers });
        const inputSize = INPUT_SHAPE.reduce((a, b) => (typeof b === 'number' ? a * b : a), 1);
        const inputData = new Float32Array(inputSize);
        const tensor = new ort.Tensor('float32', inputData, INPUT_SHAPE);
        const feeds = {};
        feeds[INPUT_NAME] = tensor;
        await session.run(feeds);
        result.cold = performance.now() - coldStart;

        // Warm runs
        const warmTimes = [];
        for (let i = 0; i < WARM_RUNS; i++) {
          const t0 = performance.now();
          await session.run(feeds);
          warmTimes.push(performance.now() - t0);
        }
        result.warm = warmTimes.reduce((a, b) => a + b) / warmTimes.length;
        result.throughput = 1000 / result.warm;

        const heapAfter = performance.memory ? performance.memory.usedJSHeapSize : 0;
        result.heap = (heapAfter - heapBefore) / 1024 / 1024;
        result.status = 'done';
      } catch (e) {
        result.status = 'error';
        result.error = e.message;
      }

      return result;
    }

    function renderResults(results) {
      const tbody = document.getElementById('results');
      const maxWarm = Math.max(...results.filter(r => r.status === 'done').map(r => r.warm), 1);

      tbody.innerHTML = results.map(r => {
        if (r.status === 'error') {
          return '<tr><td>' + r.backend + '</td><td class="skipped">Not available</td><td colspan="4" class="error">' + (r.error || 'Failed') + '</td></tr>';
        }
        if (r.status === 'running') {
          return '<tr><td>' + r.backend + '</td><td>Running...</td><td colspan="4"></td></tr>';
        }
        return '<tr>'
          + '<td>' + r.backend + '</td>'
          + '<td>Done</td>'
          + '<td>' + r.cold.toFixed(1) + '</td>'
          + '<td>' + r.warm.toFixed(1) + '</td>'
          + '<td>' + r.throughput.toFixed(1) + '</td>'
          + '<td>' + (r.heap > 0 ? r.heap.toFixed(1) : 'N/A') + '</td>'
          + '</tr>';
      }).join('');

      // Bar charts
      const charts = document.getElementById('charts');
      const metrics = [
        { key: 'cold', label: 'Cold Start (ms)', lower: true },
        { key: 'warm', label: 'Warm Average (ms)', lower: true },
        { key: 'throughput', label: 'Throughput (inf/s)', lower: false },
      ];
      charts.innerHTML = metrics.map(m => {
        const vals = results.filter(r => r.status === 'done');
        const maxVal = Math.max(...vals.map(r => r[m.key]), 1);
        return '<div class="chart"><h3>' + m.label + '</h3>'
          + vals.map(r => {
            const pct = (r[m.key] / maxVal * 100).toFixed(0);
            return '<div class="chart-bar">'
              + '<span class="chart-label">' + r.backend + '</span>'
              + '<div class="bar bar-' + r.backend + '" style="width:' + pct + '%;height:24px"></div>'
              + '<span class="chart-value">' + r[m.key].toFixed(1) + '</span>'
              + '</div>';
          }).join('')
          + '</div>';
      }).join('');
    }

    async function runAll() {
      const status = document.getElementById('status');
      const results = [];

      const configs = [
        { providers: 'wasm', label: 'wasm' },
        { providers: 'webgpu', label: 'webgpu' },
        { providers: [{ name: 'webnn', deviceType: 'gpu' }], label: 'webnn' },
      ];

      for (const cfg of configs) {
        status.textContent = 'Benchmarking ' + cfg.label + '...';
        const result = await benchmark(cfg.providers, cfg.label);
        results.push(result);
        renderResults(results);
      }

      // Mark best
      const done = results.filter(r => r.status === 'done');
      if (done.length > 0) {
        const best = done.reduce((a, b) => a.warm < b.warm ? a : b);
        status.textContent = 'Complete. Fastest: ' + best.backend + ' (' + best.warm.toFixed(1) + 'ms avg)';
      } else {
        status.textContent = 'No backends available.';
      }
    }

    runAll();
  </script>
</body>
</html>`;
}

/**
 * Generate a JSON template for machine-readable output.
 * The actual results are filled in by the browser at runtime,
 * but this provides the structure and metadata.
 */
export function generateCompareJson(
  modelPath: string,
  modelMeta: ModelMetadata,
): string {
  const template: CompareJsonTemplate = {
    model: modelPath,
    backends: [...BACKENDS],
    metrics: ['cold_latency_ms', 'warm_latency_ms', 'throughput_ips', 'peak_heap_mb'],
    inputShape: modelMeta.inputs[0]?.shape ?? [1, 3, 224, 224],
    results: null,
  };
  return JSON.stringify(template, null, 2);
}
```

- [ ] **Step 4: Wire up compare command in cli.ts**

Replace the compare command stub in `packages/cli/src/cli.ts`:

```typescript
import { generateCompareHtml, generateCompareJson } from './compare.js';
```

Replace the `compare` command action:

```typescript
program
  .command('compare')
  .description('Benchmark model across available ONNX Runtime Web backends')
  .argument('<model>', 'Path or URL to model file (.onnx)')
  .option('--json', 'Output JSON template instead of HTML')
  .option('-o, --output <dir>', 'Output directory', './compare-output/')
  .option('-v, --verbose', 'Print debug info')
  .action(async (model, options) => {
    const sourceType = classifyModelInput(model);
    const verbose = options.verbose ?? false;

    let buffer: Uint8Array;
    let modelUrl: string | undefined;

    if (sourceType === 'local-path') {
      try {
        buffer = new Uint8Array(readFileSync(model));
      } catch {
        console.error(`✗ Model not found: ${model}`);
        process.exitCode = 1;
        return;
      }
    } else if (sourceType === 'hf-model-id') {
      try {
        const resolved = await resolveHfModelId(model, false, verbose);
        modelUrl = resolved.url;
        const fetched = await fetchModelFromUrl(resolved.url, verbose);
        buffer = fetched.buffer;
        modelUrl = fetched.finalUrl;
      } catch (e) {
        console.error(`✗ ${e instanceof Error ? e.message : String(e)}`);
        process.exitCode = 1;
        return;
      }
    } else {
      try {
        const fetched = await fetchModelFromUrl(model, verbose);
        buffer = fetched.buffer;
        modelUrl = fetched.finalUrl;
      } catch (e) {
        console.error(`✗ ${e instanceof Error ? e.message : String(e)}`);
        process.exitCode = 1;
        return;
      }
    }

    let metadata;
    try {
      metadata = parseModelMetadata(buffer);
    } catch (e) {
      console.error(`✗ Could not parse model: ${model}`);
      process.exitCode = 1;
      return;
    }

    const effectivePath = modelUrl || model;

    if (options.json) {
      const json = generateCompareJson(effectivePath, metadata);
      const files = [{ path: 'compare.json', content: json }];
      writeFiles(files, options.output, { force: true });
      console.log(`✓ Compare JSON written to ${options.output}/compare.json`);
    } else {
      const html = generateCompareHtml(effectivePath, metadata);
      const files = [{ path: 'index.html', content: html }];
      writeFiles(files, options.output, { force: true });
      console.log(`✓ Compare page written to ${options.output}/index.html`);
      console.log(`  Open in a browser to run benchmarks.`);
    }
  });
```

- [ ] **Step 5: Run tests**

Run: `cd packages/cli && npx vitest run tests/compare.test.ts`
Expected: PASS

Run: `cd packages/cli && npx vitest run`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add packages/cli/src/compare.ts packages/cli/src/cli.ts packages/cli/tests/compare.test.ts
git commit -m "feat(cli): implement webai compare command with benchmark page generation"
```

---

### Task 11: Final Integration Test

- [ ] **Step 1: Run complete test suite**

Run: `cd packages/core && npx vitest run`
Expected: All core tests PASS

Run: `cd packages/cli && npx vitest run`
Expected: All CLI tests PASS

- [ ] **Step 2: Run typecheck**

Run: `cd packages/core && npx tsc --noEmit`
Expected: No errors

Run: `cd packages/cli && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Smoke test the CLI**

```bash
# Generate audio classification app
cd packages/cli
node dist/cli.js generate -m ../fixtures/test-audio-model.onnx -t audio-classification -f html -o /tmp/test-audio-class

# Generate speech-to-text app
node dist/cli.js generate -m ../fixtures/test-audio-model.onnx -t speech-to-text -f html -i mic -o /tmp/test-stt

# Generate compare page
node dist/cli.js compare ../fixtures/test-model.onnx -o /tmp/test-compare
```

Expected: All three generate output files without errors.

- [ ] **Step 4: Commit any fixes from integration testing**

If any issues found, fix and commit with descriptive message.
