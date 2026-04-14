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

// ---- AudioWorklet-based mic capture for audio ML tasks ----

/** Emit AudioWorklet processor source (written to a separate .js file) */
function emitAudioWorkletProcessor(): string {
  return `class AudioCaptureProcessor extends AudioWorkletProcessor {
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (input && input[0] && input[0].length > 0) {
      this.port.postMessage(new Float32Array(input[0]));
    }
    return true;
  }
}
registerProcessor('audio-capture-processor', AudioCaptureProcessor);`;
}

/**
 * Emit startAudioCapture: continuous PCM capture via AudioWorklet with ring buffer.
 * Falls back to ScriptProcessorNode if AudioWorklet is unavailable.
 */
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
 * Start continuous audio capture via AudioWorklet with a ring buffer.
 * Falls back to ScriptProcessorNode if AudioWorklet is not available.
 *
 * @param sampleRate - Audio sample rate (default: 16000)
 * @param bufferSeconds - Ring buffer duration in seconds (default: 30)
 * @returns Object with stream, buffer, getSamples(), and audioContext
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
  let totalWritten = 0;

  function writeSamples(samples${t ? ': Float32Array' : ''})${t ? ': void' : ''} {
    for (let i = 0; i < samples.length; i++) {
      buffer[writePos] = samples[i];
      writePos = (writePos + 1) % bufferSize;
    }
    totalWritten += samples.length;
  }

  function getSamples()${t ? ': Float32Array' : ''} {
    if (totalWritten < bufferSize) {
      // Buffer not full yet — return only the valid portion
      return buffer.slice(0, Math.min(totalWritten, bufferSize));
    }
    // Buffer full — unwrap from writePos
    const result = new Float32Array(bufferSize);
    const tail = bufferSize - writePos;
    result.set(buffer.subarray(writePos, writePos + tail), 0);
    result.set(buffer.subarray(0, writePos), tail);
    return result;
  }

  try {
    await audioContext.audioWorklet.addModule('audio-processor.js');
    const workletNode = new AudioWorkletNode(audioContext, 'audio-capture-processor');
    workletNode.port.onmessage = (e${t ? ': MessageEvent<Float32Array>' : ''}) => {
      writeSamples(e.data);
    };
    source.connect(workletNode);
    workletNode.connect(audioContext.destination);
  } catch (_err) {
    console.warn('AudioWorklet not available, falling back to ScriptProcessorNode');
    const scriptNode = audioContext.createScriptProcessor(4096, 1, 1);
    scriptNode.onaudioprocess = (e${t ? ': AudioProcessingEvent' : ''}) => {
      writeSamples(e.inputBuffer.getChannelData(0));
    };
    source.connect(scriptNode);
    scriptNode.connect(audioContext.destination);
  }

  return { stream, buffer, getSamples, audioContext };
}`;
}

/**
 * Emit createAudioInferenceLoop: interval-based audio inference loop.
 * Calls processAudio(samples) which is expected to be in scope.
 */
function emitAudioInferenceLoop(ts: boolean): string {
  const t = ts;
  const optsType = t
    ? `: {
  getSamples: () => Float32Array;
  onResult: (result: unknown) => void;
  intervalMs?: number;
}`
    : '';
  return `/**
 * Create an interval-based audio inference loop.
 * Calls processAudio(samples) on each tick and passes the result to onResult.
 *
 * @param opts - Options with getSamples, onResult callback, and optional intervalMs
 * @returns Object with start() and stop() methods.
 */
function createAudioInferenceLoop(opts${optsType})${t ? ': { start: () => void; stop: () => void }' : ''} {
  let timerId${t ? ': ReturnType<typeof setInterval> | null' : ''} = null;
  const interval = opts.intervalMs ?? 1000;

  async function tick() {
    const samples = opts.getSamples();
    if (samples.length === 0) return;
    const result = await processAudio(samples);
    opts.onResult(result);
  }

  return {
    start() {
      if (timerId !== null) return;
      timerId = setInterval(tick, interval);
    },
    stop() {
      if (timerId !== null) {
        clearInterval(timerId);
        timerId = null;
      }
    },
  };
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
