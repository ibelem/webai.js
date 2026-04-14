// packages/core/src/preprocess/audio-mel.ts
import { fft, hannWindow } from './audio-fft.js';

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
    const mag = new Float64Array(numBins);
    for (let k = 0; k < numBins; k++) {
      mag[k] = Math.sqrt(re[k] * re[k] + im[k] * im[k]);
    }
    magnitudes.push(mag);
  }

  return { magnitudes };
}

function hzToMel(hz: number): number {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel: number): number {
  return 700 * (10 ** (mel / 2595) - 1);
}

export function melFilterbank(
  numMelBands: number,
  fftSize: number,
  sampleRate: number,
): Float64Array[] {
  const numBins = fftSize / 2 + 1;
  const melMin = hzToMel(0);
  const melMax = hzToMel(sampleRate / 2);
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
  data: Float32Array;
  numFrames: number;
  numMelBands: number;
}

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
        sum += mag[k] * mag[k] * filterWeights[k];
      }
      data[f * numMelBands + m] = Math.log10(Math.max(sum, 1e-10));
    }
  }

  return { data, numFrames, numMelBands };
}
