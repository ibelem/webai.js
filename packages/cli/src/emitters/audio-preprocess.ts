/**
 * Audio preprocess emitter: generates standalone audio preprocessing source code.
 *
 * Mirrors the logic in @webai/core audio preprocessing functions exactly.
 * Same pattern as preprocess.ts (image): each emit* function returns a string
 * containing a complete JavaScript/TypeScript function.
 */

import type { ResolvedConfig } from '@webai/core';
import type { CodeBlock } from '../types.js';

/** Emit the hannWindow function as standalone JS/TS code */
function emitHannWindow(ts: boolean): string {
  const t = ts;
  return `/**
 * Hann window function.
 * w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))
 */
function hannWindow(length${t ? ': number' : ''})${t ? ': Float64Array' : ''} {
  const w = new Float64Array(length);
  for (let i = 0; i < length; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (length - 1)));
  }
  return w;
}`;
}

/** Emit the fft function as standalone JS/TS code */
function emitFft(ts: boolean): string {
  const t = ts;
  return `/**
 * In-place radix-2 Cooley-Tukey FFT.
 * Input length must be a power of 2.
 * Transforms (re, im) arrays in place.
 */
function fft(re${t ? ': Float64Array' : ''}, im${t ? ': Float64Array' : ''})${t ? ': void' : ''} {
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
}`;
}

/** Emit the stft function as standalone JS/TS code */
function emitStft(ts: boolean): string {
  const t = ts;
  return `/**
 * Short-Time Fourier Transform.
 * Returns magnitude spectrogram as array of Float64Array frames.
 */
function stft(
  samples${t ? ': Float32Array' : ''},
  fftSize${t ? ': number' : ''},
  hopSize${t ? ': number' : ''}
)${t ? ': { magnitudes: Float64Array[] }' : ''} {
  const window = hannWindow(fftSize);
  const numFrames = Math.max(0, Math.floor((samples.length - fftSize) / hopSize) + 1);
  const numBins = fftSize / 2 + 1;
  const magnitudes${t ? ': Float64Array[]' : ''} = [];

  for (let f = 0; f < numFrames; f++) {
    const offset = f * hopSize;
    const re = new Float64Array(fftSize);
    const im = new Float64Array(fftSize);
    for (let j = 0; j < fftSize; j++) {
      re[j] = (samples[offset + j] ?? 0) * window[j];
    }
    fft(re, im);
    const mag = new Float64Array(numBins);
    for (let k = 0; k < numBins; k++) {
      mag[k] = Math.sqrt(re[k] * re[k] + im[k] * im[k]);
    }
    magnitudes.push(mag);
  }

  return { magnitudes };
}`;
}

/** Emit the melFilterbank function as standalone JS/TS code */
function emitMelFilterbank(ts: boolean): string {
  const t = ts;
  return `/**
 * Create triangular mel-scale filterbank.
 */
function melFilterbank(
  numMelBands${t ? ': number' : ''},
  fftSize${t ? ': number' : ''},
  sampleRate${t ? ': number' : ''}
)${t ? ': Float64Array[]' : ''} {
  function hzToMel(hz${t ? ': number' : ''})${t ? ': number' : ''} {
    return 2595 * Math.log10(1 + hz / 700);
  }
  function melToHz(mel${t ? ': number' : ''})${t ? ': number' : ''} {
    return 700 * (10 ** (mel / 2595) - 1);
  }

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

/** Emit the melSpectrogram function as standalone JS/TS code */
function emitMelSpectrogram(ts: boolean): string {
  const t = ts;
  return `/**
 * Compute log-mel spectrogram from raw audio samples.
 */
function melSpectrogram(
  samples${t ? ': Float32Array' : ''},
  sampleRate${t ? ': number' : ''},
  fftSize${t ? ': number' : ''},
  hopSize${t ? ': number' : ''},
  numMelBands${t ? ': number' : ''}
)${t ? ': { data: Float32Array; numFrames: number; numMelBands: number }' : ''} {
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
        sum += mag[k] * mag[k] * filterWeights[k];
      }
      data[f * numMelBands + m] = Math.log10(Math.max(sum, 1e-10));
    }
  }

  return { data, numFrames, numMelBands };
}`;
}

/** Emit the mfcc function as standalone JS/TS code */
function emitMfcc(ts: boolean): string {
  const t = ts;
  return `/**
 * Compute MFCC (Mel-Frequency Cepstral Coefficients) from log-mel spectrogram.
 * Uses DCT-II to decorrelate mel bands.
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
        sum +=
          melData[frameOffset + m] *
          Math.cos((Math.PI * c * (m + 0.5)) / numMelBands);
      }
      result[outOffset + c] = sum;
    }
  }

  return result;
}`;
}

/**
 * Emit the audio preprocess CodeBlock for a given config.
 *
 * - text-to-speech: returns empty block (no preprocessing needed)
 * - speech-to-text: hannWindow, fft, stft, melFilterbank, melSpectrogram
 * - audio-classification: all of the above + mfcc
 */
export function emitAudioPreprocessBlock(config: ResolvedConfig): CodeBlock {
  if (config.task === 'text-to-speech') {
    return {
      id: 'preprocess',
      code: '',
      imports: [],
      exports: [],
    };
  }

  const ts = config.lang === 'ts';

  const parts = [
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
