import { describe, it, expect } from 'vitest';
import { stft, melFilterbank, melSpectrogram } from './audio-mel.js';

describe('stft', () => {
  it('returns correct number of frames', () => {
    const samples = new Float32Array(16000);
    const { magnitudes } = stft(samples, 512, 160);
    const expectedFrames = Math.floor((16000 - 512) / 160) + 1;
    expect(magnitudes.length).toBe(expectedFrames);
  });

  it('each frame has fftSize/2+1 bins', () => {
    const samples = new Float32Array(1024);
    const { magnitudes } = stft(samples, 256, 128);
    expect(magnitudes[0].length).toBe(129);
  });

  it('detects pure tone frequency', () => {
    const sampleRate = 16000;
    const freq = 440;
    const n = 4096;
    const samples = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      samples[i] = Math.sin((2 * Math.PI * freq * i) / sampleRate);
    }
    const fftSize = 1024;
    const { magnitudes } = stft(samples, fftSize, fftSize);
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
      expect(f.length).toBe(129);
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

  it('filter centers increase monotonically', () => {
    const filters = melFilterbank(40, 512, 16000);
    const centers = filters.map((f) => {
      let maxVal = 0, maxIdx = 0;
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
    const avgSilent = silentMel.data.reduce((s, v) => s + v, 0) / silentMel.data.length;
    const avgLoud = loudMel.data.reduce((s, v) => s + v, 0) / loudMel.data.length;
    expect(avgLoud).toBeGreaterThan(avgSilent);
  });
});
