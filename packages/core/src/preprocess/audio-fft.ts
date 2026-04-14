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
