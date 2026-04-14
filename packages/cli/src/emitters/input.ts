/**
 * Input emitter: generates capture utility code for different input modes.
 *
 * Dispatches by config.input:
 *   file    → empty block (file reading is framework-specific)
 *   camera  → startCamera + captureFrame + stopStream + inferenceLoop
 *   video   → captureFrame + inferenceLoop
 *   screen  → startScreenCapture + captureFrame + stopStream + inferenceLoop
 *   mic     → startMicrophone + captureAudio + stopStream
 *
 * The inference loop uses auto-calibrated frame skipping (Decision #16):
 * measures inference time and adjusts skip rate to match model throughput.
 */

import type { ResolvedConfig } from '@webai/core';
import type { CodeBlock } from '../types.js';

// ---- Visual capture: shared across camera, video, screen ----

/** Emit captureFrame: draw video element to canvas, return ImageData */
function emitCaptureFrame(ts: boolean): string {
  const t = ts;
  return `/**
 * Capture a single frame from a video element.
 * Draws the current video frame to a canvas and returns the pixel data.
 */
function captureFrame(
  video${t ? ': HTMLVideoElement' : ''},
  canvas${t ? ': HTMLCanvasElement' : ''}
)${t ? ': ImageData' : ''} {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d')${t ? '!' : ''};
  ctx.drawImage(video, 0, 0);
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}`;
}

/**
 * Emit inference loop with auto-calibrated frame skipping.
 * Per Decision #16: measures inference time and auto-adjusts skip rate.
 */
function emitInferenceLoop(ts: boolean): string {
  const t = ts;
  const callbackType = t
    ? ': (imageData: ImageData) => Promise<{ result: unknown; elapsed: number }>'
    : '';
  const statusType = t ? ': (elapsed: number) => void' : '';
  const optsType = t
    ? `: {
  video: HTMLVideoElement;
  canvas: HTMLCanvasElement;
  onFrame${callbackType};
  onStatus${statusType};
}`
    : '';
  return `/**
 * Create an auto-calibrated inference loop using requestAnimationFrame.
 * Measures inference time and skips frames to avoid queuing work
 * faster than the model can process.
 *
 * @returns Object with start() and stop() methods.
 */
function createInferenceLoop(opts${optsType})${t ? ': { start: () => void; stop: () => void }' : ''} {
  let running = false;
  let frameSkip = 0;
  let skipCount = 0;
  let rafId = 0;

  async function loop() {
    if (!running) return;

    if (skipCount < frameSkip) {
      skipCount++;
      rafId = requestAnimationFrame(loop);
      return;
    }
    skipCount = 0;

    const imageData = captureFrame(opts.video, opts.canvas);
    const { result, elapsed } = await opts.onFrame(imageData);
    void result;

    // Auto-calibrate: target ~60fps frame budget (16.67ms)
    // If inference takes 50ms, skip ~2 frames to avoid queuing
    frameSkip = Math.max(0, Math.floor(elapsed / 16.67) - 1);

    opts.onStatus(elapsed);

    if (running) {
      rafId = requestAnimationFrame(loop);
    }
  }

  return {
    start() {
      running = true;
      rafId = requestAnimationFrame(loop);
    },
    stop() {
      running = false;
      cancelAnimationFrame(rafId);
    },
  };
}`;
}

// ---- Camera-specific ----

function emitStartCamera(ts: boolean): string {
  const t = ts;
  return `/**
 * Start camera capture using getUserMedia.
 * Attaches the stream to a video element and waits for playback.
 *
 * @param video - Video element to attach the camera stream to
 * @param facingMode - Camera facing mode: 'user' (front) or 'environment' (back)
 * @returns The MediaStream (keep reference to stop later)
 */
async function startCamera(
  video${t ? ': HTMLVideoElement' : ''},
  facingMode${t ? ': string' : ''} = 'user'
)${t ? ': Promise<MediaStream>' : ''} {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode },
  });
  video.srcObject = stream;
  await video.play();
  return stream;
}`;
}

// ---- Screen capture ----

function emitStartScreenCapture(ts: boolean): string {
  const t = ts;
  return `/**
 * Start screen capture using getDisplayMedia.
 * Attaches the stream to a video element and waits for playback.
 *
 * @param video - Video element to attach the screen stream to
 * @returns The MediaStream (keep reference to stop later)
 */
async function startScreenCapture(
  video${t ? ': HTMLVideoElement' : ''}
)${t ? ': Promise<MediaStream>' : ''} {
  const stream = await navigator.mediaDevices.getDisplayMedia({
    video: true,
  });
  video.srcObject = stream;
  await video.play();
  return stream;
}`;
}

// ---- Shared stream stop ----

function emitStopStream(ts: boolean): string {
  const t = ts;
  return `/**
 * Stop all tracks on a MediaStream.
 */
function stopStream(stream${t ? ': MediaStream' : ''})${t ? ': void' : ''} {
  for (const track of stream.getTracks()) {
    track.stop();
  }
}`;
}

// ---- Microphone capture ----

function emitStartMicrophone(ts: boolean): string {
  const t = ts;
  const returnType = t
    ? ': Promise<{ stream: MediaStream; analyser: AnalyserNode; buffer: Float32Array }>'
    : '';
  return `/**
 * Start microphone capture using getUserMedia.
 * Sets up an AudioContext and AnalyserNode for reading audio data.
 *
 * @param fftSize - FFT size for the analyser (default: 2048)
 * @returns Object with stream, analyser, and a pre-allocated buffer
 */
async function startMicrophone(fftSize${t ? ': number' : ''} = 2048)${returnType} {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const audioCtx = new AudioContext();
  const source = audioCtx.createMediaStreamSource(stream);
  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = fftSize;
  source.connect(analyser);

  const buffer = new Float32Array(analyser.fftSize);
  return { stream, analyser, buffer };
}`;
}

function emitCaptureAudio(ts: boolean): string {
  const t = ts;
  return `/**
 * Capture current audio data from an AnalyserNode.
 * Writes time-domain data into the provided buffer and returns it.
 */
function captureAudio(
  analyser${t ? ': AnalyserNode' : ''},
  buffer${t ? ': Float32Array' : ''}
)${t ? ': Float32Array' : ''} {
  analyser.getFloatTimeDomainData(buffer);
  return buffer;
}`;
}

// ---- Block emitter dispatch ----

/**
 * Emit the input CodeBlock for a given config.
 *
 * Dispatches by input mode:
 *   file    → empty block
 *   camera  → captureFrame + startCamera + stopStream + inferenceLoop
 *   video   → captureFrame + stopStream + inferenceLoop
 *   screen  → captureFrame + startScreenCapture + stopStream + inferenceLoop
 *   mic     → startMicrophone + captureAudio + stopStream
 */
export function emitInputBlock(config: ResolvedConfig): CodeBlock {
  const ts = config.lang === 'ts';

  const parts: string[] = [];
  const exports: string[] = [];

  switch (config.input) {
    case 'camera':
      parts.push(emitCaptureFrame(ts));
      parts.push(emitStartCamera(ts));
      parts.push(emitStopStream(ts));
      parts.push(emitInferenceLoop(ts));
      exports.push('captureFrame', 'startCamera', 'stopStream', 'createInferenceLoop');
      break;

    case 'video':
      parts.push(emitCaptureFrame(ts));
      parts.push(emitStopStream(ts));
      parts.push(emitInferenceLoop(ts));
      exports.push('captureFrame', 'stopStream', 'createInferenceLoop');
      break;

    case 'screen':
      parts.push(emitCaptureFrame(ts));
      parts.push(emitStartScreenCapture(ts));
      parts.push(emitStopStream(ts));
      parts.push(emitInferenceLoop(ts));
      exports.push('captureFrame', 'startScreenCapture', 'stopStream', 'createInferenceLoop');
      break;

    case 'mic':
      parts.push(emitStartMicrophone(ts));
      parts.push(emitCaptureAudio(ts));
      parts.push(emitStopStream(ts));
      exports.push('startMicrophone', 'captureAudio', 'stopStream');
      break;

    case 'file':
    default:
      // File input has no capture utilities (framework handles file reading)
      break;
  }

  return {
    id: 'input',
    code: parts.join('\n\n'),
    imports: [],
    exports,
  };
}
