/**
 * Compare command: generates a self-contained HTML benchmark page
 * that compares ORT Web backends (WASM, WebGPU, WebNN) client-side.
 *
 * The generated page loads ONNX Runtime Web from CDN and runs
 * benchmarks directly in the browser when opened.
 */

import type { ModelMetadata } from '@webai/core';

const ORT_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
const BACKENDS = ['wasm', 'webgpu', 'webnn'] as const;

/**
 * Generate a self-contained HTML benchmark page.
 *
 * The page loads ORT Web from CDN, runs cold + warm benchmarks
 * for each backend, and displays results in a table with bar charts.
 */
export function generateCompareHtml(modelPath: string, modelMeta: ModelMetadata): string {
  const inputShape = modelMeta.inputs[0]?.shape ?? [1, 3, 224, 224];
  const shapeStr = '[' + inputShape.join(', ') + ']';
  const inputName = modelMeta.inputs[0]?.name ?? 'input';
  const inputDtype = modelMeta.inputs[0]?.dataType ?? 'float32';

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>webai compare — ${escapeHtml(modelPath)}</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
    background: #0d1117; color: #c9d1d9; padding: 2rem;
    max-width: 960px; margin: 0 auto;
  }
  h1 { font-size: 1.4rem; margin-bottom: 0.25rem; color: #58a6ff; }
  .subtitle { color: #8b949e; font-size: 0.85rem; margin-bottom: 1.5rem; }
  .model-path { color: #f0883e; word-break: break-all; }
  table { width: 100%; border-collapse: collapse; margin-bottom: 2rem; }
  th, td { padding: 0.6rem 0.8rem; text-align: left; border-bottom: 1px solid #21262d; }
  th { color: #8b949e; font-weight: 600; font-size: 0.8rem; text-transform: uppercase; }
  td { font-variant-numeric: tabular-nums; }
  .status-ok { color: #3fb950; }
  .status-error { color: #f85149; }
  .status-running { color: #d29922; }
  .status-pending { color: #8b949e; }
  .bar-container { display: flex; gap: 1.5rem; flex-wrap: wrap; margin-bottom: 2rem; }
  .bar-group { flex: 1; min-width: 250px; }
  .bar-group h3 { font-size: 0.85rem; color: #8b949e; margin-bottom: 0.5rem; }
  .bar-row { display: flex; align-items: center; margin-bottom: 0.4rem; gap: 0.5rem; }
  .bar-label { width: 60px; font-size: 0.75rem; color: #8b949e; text-align: right; }
  .bar-track { flex: 1; height: 20px; background: #161b22; border-radius: 4px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 4px; transition: width 0.3s ease; }
  .bar-fill.wasm { background: #3fb950; }
  .bar-fill.webgpu { background: #58a6ff; }
  .bar-fill.webnn { background: #bc8cff; }
  .bar-value { width: 70px; font-size: 0.75rem; color: #c9d1d9; }
  .winner { background: #1f2a1f; border-radius: 6px; padding: 1rem; margin-top: 1rem; border: 1px solid #238636; }
  .winner h3 { color: #3fb950; margin-bottom: 0.25rem; }
  #log { font-size: 0.8rem; color: #8b949e; white-space: pre-wrap; margin-top: 1rem; }
</style>
</head>
<body>

<h1>webai compare</h1>
<p class="subtitle">Model: <span class="model-path">${escapeHtml(modelPath)}</span></p>
<p class="subtitle">Input: <code>${escapeHtml(inputName)}</code> ${escapeHtml(shapeStr)} (${escapeHtml(inputDtype)})</p>

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
    <tr id="row-wasm"><td>wasm</td><td class="status-pending">pending</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
    <tr id="row-webgpu"><td>webgpu</td><td class="status-pending">pending</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
    <tr id="row-webnn"><td>webnn</td><td class="status-pending">pending</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  </tbody>
</table>

<div class="bar-container">
  <div class="bar-group">
    <h3>Cold Start (ms) — lower is better</h3>
    <div id="bars-cold"></div>
  </div>
  <div class="bar-group">
    <h3>Warm Avg (ms) — lower is better</h3>
    <div id="bars-warm"></div>
  </div>
  <div class="bar-group">
    <h3>Throughput (inf/s) — higher is better</h3>
    <div id="bars-throughput"></div>
  </div>
</div>

<div id="winner-box" style="display:none" class="winner">
  <h3 id="winner-text"></h3>
  <p id="winner-detail" style="color:#8b949e;font-size:0.85rem;"></p>
</div>

<pre id="log"></pre>

<script src="${ORT_CDN}ort.all.min.js"></script>
<script>
const MODEL_PATH = ${JSON.stringify(modelPath)};
const INPUT_NAME = ${JSON.stringify(inputName)};
const INPUT_SHAPE = ${shapeStr};
const INPUT_DTYPE = ${JSON.stringify(inputDtype)};
const WARM_RUNS = 10;

function log(msg) {
  document.getElementById('log').textContent += msg + '\\n';
}

function createTensor() {
  const size = INPUT_SHAPE.reduce((a, b) => a * b, 1);
  let data;
  if (INPUT_DTYPE === 'float32') {
    data = new Float32Array(size);
    for (let i = 0; i < size; i++) data[i] = Math.random();
  } else if (INPUT_DTYPE === 'int32') {
    data = new Int32Array(size);
  } else if (INPUT_DTYPE === 'int64') {
    data = new BigInt64Array(size);
  } else if (INPUT_DTYPE === 'uint8') {
    data = new Uint8Array(size);
    for (let i = 0; i < size; i++) data[i] = Math.floor(Math.random() * 256);
  } else {
    data = new Float32Array(size);
    for (let i = 0; i < size; i++) data[i] = Math.random();
  }
  return new ort.Tensor(INPUT_DTYPE, data, INPUT_SHAPE);
}

function getHeapMB() {
  if (performance.memory) return performance.memory.usedJSHeapSize / 1024 / 1024;
  return null;
}

async function benchmark(providers, label) {
  log('--- ' + label + ' ---');
  const result = { backend: label, status: 'ok', cold: 0, warm: 0, throughput: 0, heap: null };

  try {
    const heapBefore = getHeapMB();

    // Cold start: session creation + first inference
    const t0 = performance.now();
    const session = await ort.InferenceSession.create(MODEL_PATH, { executionProviders: providers });
    const tensor = createTensor();
    const feeds = { [INPUT_NAME]: tensor };
    await session.run(feeds);
    const t1 = performance.now();
    result.cold = Math.round((t1 - t0) * 100) / 100;
    log('Cold Start: ' + result.cold + ' ms');

    // Warm runs
    const times = [];
    for (let i = 0; i < WARM_RUNS; i++) {
      const ws = performance.now();
      await session.run(feeds);
      const we = performance.now();
      times.push(we - ws);
    }
    const avg = times.reduce((a, b) => a + b, 0) / times.length;
    result.warm = Math.round(avg * 100) / 100;
    result.throughput = Math.round(1000 / avg * 100) / 100;
    log('Warm Avg: ' + result.warm + ' ms (' + result.throughput + ' inf/s)');

    const heapAfter = getHeapMB();
    if (heapBefore !== null && heapAfter !== null) {
      result.heap = Math.round((heapAfter) * 10) / 10;
      log('Peak Heap: ~' + result.heap + ' MB');
    }

    await session.release();
  } catch (e) {
    result.status = 'error';
    log('Error: ' + e.message);
  }

  return result;
}

function updateRow(r) {
  const row = document.getElementById('row-' + r.backend);
  if (!row) return;
  const cells = row.children;
  cells[1].textContent = r.status;
  cells[1].className = r.status === 'ok' ? 'status-ok' : 'status-error';
  if (r.status === 'ok') {
    cells[2].textContent = r.cold;
    cells[3].textContent = r.warm;
    cells[4].textContent = r.throughput;
    cells[5].textContent = r.heap !== null ? r.heap : '—';
  } else {
    cells[2].textContent = '—';
    cells[3].textContent = '—';
    cells[4].textContent = '—';
    cells[5].textContent = '—';
  }
}

function setRunning(backend) {
  const row = document.getElementById('row-' + backend);
  if (!row) return;
  row.children[1].textContent = 'running...';
  row.children[1].className = 'status-running';
}

function renderBars(results) {
  const okResults = results.filter(r => r.status === 'ok');
  if (okResults.length === 0) return;

  const maxCold = Math.max(...okResults.map(r => r.cold));
  const maxWarm = Math.max(...okResults.map(r => r.warm));
  const maxTput = Math.max(...okResults.map(r => r.throughput));

  function makeBars(containerId, results, getter, maxVal, unit) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    for (const r of results) {
      const val = r.status === 'ok' ? getter(r) : 0;
      const pct = maxVal > 0 ? (val / maxVal * 100) : 0;
      container.innerHTML +=
        '<div class="bar-row">' +
          '<span class="bar-label">' + r.backend + '</span>' +
          '<div class="bar-track"><div class="bar-fill ' + r.backend + '" style="width:' + pct + '%"></div></div>' +
          '<span class="bar-value">' + (r.status === 'ok' ? val + ' ' + unit : 'n/a') + '</span>' +
        '</div>';
    }
  }

  makeBars('bars-cold', results, r => r.cold, maxCold, 'ms');
  makeBars('bars-warm', results, r => r.warm, maxWarm, 'ms');
  makeBars('bars-throughput', results, r => r.throughput, maxTput, 'inf/s');
}

function showWinner(results) {
  const okResults = results.filter(r => r.status === 'ok');
  if (okResults.length === 0) return;
  const fastest = okResults.reduce((a, b) => a.warm < b.warm ? a : b);
  const box = document.getElementById('winner-box');
  box.style.display = 'block';
  document.getElementById('winner-text').textContent = 'Fastest: ' + fastest.backend;
  document.getElementById('winner-detail').textContent =
    fastest.warm + ' ms avg latency, ' + fastest.throughput + ' inf/s throughput';
}

async function runAll() {
  const results = [];
  log('Starting benchmarks for: ' + MODEL_PATH);
  log('Input shape: ${shapeStr}');
  log('');

  // WASM
  setRunning('wasm');
  const wasmResult = await benchmark(['wasm'], 'wasm');
  results.push(wasmResult);
  updateRow(wasmResult);
  renderBars(results);

  // WebGPU
  setRunning('webgpu');
  const webgpuResult = await benchmark(['webgpu'], 'webgpu');
  results.push(webgpuResult);
  updateRow(webgpuResult);
  renderBars(results);

  // WebNN
  setRunning('webnn');
  const webnnResult = await benchmark([{ name: 'webnn', deviceType: 'gpu' }], 'webnn');
  results.push(webnnResult);
  updateRow(webnnResult);
  renderBars(results);

  log('');
  log('Done.');
  showWinner(results);
}

runAll();
</script>
</body>
</html>`;
}

/**
 * Generate a JSON template for machine-readable benchmark metadata.
 */
export function generateCompareJson(modelPath: string, modelMeta: ModelMetadata): string {
  const inputShape = modelMeta.inputs[0]?.shape ?? [1, 3, 224, 224];

  return JSON.stringify(
    {
      model: modelPath,
      backends: [...BACKENDS],
      metrics: ['cold_latency_ms', 'warm_latency_ms', 'throughput_ips', 'peak_heap_mb'],
      inputShape,
      results: null,
    },
    null,
    2,
  );
}

/** Escape HTML special characters */
function escapeHtml(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
