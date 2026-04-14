/**
 * Compute MFCC (Mel-Frequency Cepstral Coefficients) from log-mel spectrogram.
 * Uses DCT-II to decorrelate mel bands.
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
