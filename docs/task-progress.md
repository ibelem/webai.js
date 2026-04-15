# webai.js — Task Implementation Progress

## Implemented Tasks (27/27)

### Image Tasks (5/5) ✓ DONE

| Task | Profile | Preprocess | Postprocess | Emitter | HTML | React | Vanilla | Next.js | Svelte | Web UI Mock | Status |
|------|---------|------------|-------------|---------|------|-------|---------|---------|--------|-------------|--------|
| image-classification | Y | Y | softmax+topK | Y | Y | Y | Y | Y | Y | Y | Done |
| object-detection | Y | Y | nms+bbox | Y | Y | Y | Y | Y | Y | Y | Done |
| image-segmentation | Y | Y | argmax mask | Y | Y | Y | Y | Y | Y | Y | Done |
| feature-extraction | Y | Y | passthrough | Y | Y | Y | Y | Y | Y | Y | Done |
| depth-estimation | Y | Y | depthNormalize+colormap | Y | — | — | — | — | — | Y | Phase 3A |

### Audio Tasks (6/6) ✓ DONE

| Task | Profile | Preprocess | Postprocess | Emitter | HTML | React | Vanilla | Next.js | Svelte | Web UI Mock | Status |
|------|---------|------------|-------------|---------|------|-------|---------|---------|--------|-------------|--------|
| audio-classification | Y | MFCC | softmax+topK | Y | Y | Y | Y | Y | Y | Y | Done |
| speech-to-text | Y | MFCC | CTC decode | Y | Y | Y | Y | Y | Y | Y | Done |
| text-to-speech | Y | BPE tokenizer | audio playback | Y | Y | Y | Y | Y | Y | Y | Done |
| audio-to-audio | Y | MFCC | normalizeWaveform+playAudio | Y | — | — | — | — | — | Y | Phase 3C |
| speaker-diarization | Y | MFCC | segmentSpeakers | Y | — | — | — | — | — | Y | Phase 4 |
| voice-activity-detection | Y | MFCC | binaryThreshold | Y | — | — | — | — | — | Y | Phase 4 |

### Text Tasks (12/12) ✓ DONE

| Task | Profile | Preprocess | Postprocess | Emitter | HTML | React | Vanilla | Next.js | Svelte | Web UI Mock | Status |
|------|---------|------------|-------------|---------|------|-------|---------|---------|--------|-------------|--------|
| text-classification | Y | BPE tokenizer | softmax+topK | Y | Y | Y | Y | Y | Y | Y | Done |
| text-generation | Y | BPE tokenizer | argmax loop | Y | Y | Y | Y | Y | Y | Y | Done |
| zero-shot-classification | Y | BPE tokenizer | entailment softmax | Y | Y | Y | Y | Y | Y | Y | Done |
| fill-mask | Y | BPE tokenizer | softmax+topK | Y | — | — | — | — | — | Y | Phase 3A |
| sentence-similarity | Y | BPE tokenizer | cosineSimilarity | Y | — | — | — | — | — | Y | Phase 3A |
| token-classification (NER) | Y | BPE tokenizer | tokenArgmax+extractSpans | Y | — | — | — | — | — | Y | Phase 3B |
| question-answering | Y | BPE tokenizer | start/end span extraction | Y | — | — | — | — | — | Y | Phase 3B |
| summarization | Y | BPE tokenizer | seq2seqGreedyDecode | Y | — | — | — | — | — | Y | Phase 3B |
| translation | Y | BPE tokenizer | seq2seqGreedyDecode | Y | — | — | — | — | — | Y | Phase 3B |
| text2text-generation | Y | BPE tokenizer | seq2seqGreedyDecode | Y | — | — | — | — | — | Y | Phase 4 |
| conversational | Y | BPE tokenizer | sampleNextToken | Y | — | — | — | — | — | Y | Phase 4 |
| table-question-answering | Y | BPE tokenizer | span extraction | Y | — | — | — | — | — | Y | Phase 4 |

### Multimodal Tasks (4/4) ✓ DONE

| Task | Profile | Preprocess | Postprocess | Emitter | HTML | React | Vanilla | Next.js | Svelte | Web UI Mock | Status |
|------|---------|------------|-------------|---------|------|-------|---------|---------|--------|-------------|--------|
| image-to-text | Y | Y (image) | seq2seqGreedyDecode | Y | — | — | — | — | — | Y | Phase 3C |
| visual-question-answering | Y | Y (image) | seq2seqGreedyDecode | Y | — | — | — | — | — | Y | Phase 4 |
| document-question-answering | Y | Y (image) | seq2seqGreedyDecode | Y | — | — | — | — | — | Y | Phase 4 |
| image-text-to-text | Y | Y (image) | seq2seqGreedyDecode | Y | — | — | — | — | — | Y | Phase 4 |

---

## Video / WebCodecs Capabilities

Added in the video streams session (prior to Phase 3):

| Feature | Input Modes | Description |
|---------|-------------|-------------|
| `captureFrameZeroCopy` | camera, video, screen | WebCodecs `VideoFrame` zero-copy capture, falls back to canvas |
| `processVideoFrames` | video | Seek-based batch frame extraction (faster than real-time) |
| `createFrameAccumulator` | camera, video, screen | Ring buffer for clip-based temporal inference with configurable stride |
| `createClipInferenceLoop` | camera, video, screen | Non-blocking rAF loop for temporal/clip inference |

---

## Implementation Priority

### Phase 3A — Low-Hanging Fruit ✓ DONE

| Task | Postprocessing | Status |
|------|---------------|--------|
| **fill-mask** | softmax + topK + postprocessFillMask | Done |
| **sentence-similarity** | cosineSimilarity + postprocessSimilarity | Done |
| **depth-estimation** | depthNormalize + depthToColormap + postprocessDepth | Done |

### Phase 3B — Medium Effort (Text) ✓ DONE

| Task | Postprocessing | Status |
|------|---------------|--------|
| **token-classification (NER)** | tokenArgmax + extractSpans + postprocessTokenClassification | Done |
| **question-answering** | postprocessQA (start/end logit span extraction) | Done |
| **summarization** | seq2seqGreedyDecode + postprocessSummarization | Done |
| **translation** | seq2seqGreedyDecode + postprocessTranslation | Done |

### Phase 3C — Medium Effort (Vision + Audio) ✓ DONE

| Task | Postprocessing | Status |
|------|---------------|--------|
| **image-to-text** | seq2seqGreedyDecode + postprocessImageToText | Done |
| **audio-to-audio** | normalizeWaveform + playAudio + postprocessAudioToAudio | Done |

### Phase 3D — High Effort (Multimodal) ✓ DONE (see Phase 4)

Moved to Phase 4 below along with remaining audio/text tasks.

### Phase 4 — Remaining Tasks ✓ DONE

| Task | Postprocessing | Status |
|------|---------------|--------|
| **speaker-diarization** | segmentSpeakers (per-frame speaker argmax + segment merging) | Done |
| **voice-activity-detection** | binaryThreshold (speech probability thresholding) | Done |
| **text2text-generation** | seq2seqGreedyDecode + postprocessText2Text | Done |
| **conversational** | sampleNextToken + postprocessConversational | Done |
| **table-question-answering** | postprocessTableQA (start/end span extraction) | Done |
| **visual-question-answering** | seq2seqGreedyDecode + postprocessVQA | Done |
| **document-question-answering** | seq2seqGreedyDecode + postprocessDocQA | Done |
| **image-text-to-text** | seq2seqGreedyDecode + postprocessImageTextToText | Done |

---

## Remaining Work

- **Framework UI for Phase 3/4 tasks** — 17 tasks have emitters but not framework-specific UI templates (HTML/React/Vanilla/Next.js/SvelteKit). Currently they generate code using the default/fallback UI.

---

## Per-Task Checklist

Each new task requires:

- [ ] **Task profile** — `packages/core/src/tasks/task-profiles.ts` (label, default input, supported inputs, preprocess defaults, postprocess type)
- [ ] **TaskType union** — `packages/core/src/tasks/types.ts`
- [ ] **Compatibility matrices** — `packages/core/src/config/compatibility.ts` (task x input, task x engine)
- [ ] **Postprocessing emitter** — `packages/cli/src/emitters/postprocess.ts` (new case in switch)
- [ ] **Preprocessing emitter** — `packages/cli/src/emitters/preprocess.ts` (if new preprocessing needed)
- [ ] **Framework UI x5** — `packages/cli/src/frameworks/{html,react-vite,vanilla-vite,nextjs,sveltekit}.ts`
- [ ] **Shared CSS** — `packages/cli/src/frameworks/shared.ts` (if new UI components needed)
- [ ] **Web UI mock metadata** — `packages/web/src/mock-metadata.ts` (mock tensor shapes)
- [ ] **Snapshot tests** — `packages/cli/tests/snapshot.test.ts` (new snapshot case)
- [ ] **Emitter structure tests** — `packages/cli/tests/emitter-structure.test.ts` (if new emitter functions)

---

## Architecture Reference

```
packages/core/src/                    ← Knowledge layer
├── tasks/
│   ├── types.ts                      ← TaskType union (27 tasks)
│   ├── task-profiles.ts              ← TASK_PROFILES[task]
│   └── task-detector.ts              ← Auto-detect from model shape
├── config/
│   ├── compatibility.ts              ← Task x Input, Task x Engine
│   └── resolver.ts                   ← CliFlags → ResolvedConfig
├── preprocess/
│   ├── image-resize.ts               ← Bilinear resize
│   ├── image-normalize.ts            ← Channel-wise normalization
│   ├── image-to-nchw.ts              ← HWC → NCHW transpose
│   ├── audio-{fft,mel,mfcc}.ts       ← Audio feature extraction
│   └── text-tokenizer.ts             ← BPE tokenizer
└── postprocess/
    ├── softmax.ts, topk.ts, argmax.ts, nms.ts

packages/cli/src/                     ← Code generation layer
├── emitters/                         ← Layer 1: CodeBlock[]
│   ├── preprocess.ts                 ← Task dispatch → image/audio/text
│   ├── audio-preprocess.ts           ← Audio-specific code gen
│   ├── text-preprocess.ts            ← BPE tokenizer code gen
│   ├── postprocess.ts                ← Task-specific postprocess code gen
│   ├── input.ts                      ← File/camera/video/screen/mic
│   ├── inference-{ort,litert,webnn}.ts
│   └── opfs-cache.ts
├── frameworks/                       ← Layer 2: GeneratedFile[]
│   ├── shared.ts                     ← CSS, README, helpers
│   ├── html.ts                       ← Single-file HTML
│   ├── react-vite.ts                 ← React + Vite
│   ├── vanilla-vite.ts               ← Vanilla JS + Vite
│   ├── nextjs.ts                     ← Next.js
│   └── sveltekit.ts                  ← SvelteKit
└── assembler.ts                      ← Orchestrates Layer 1 → Layer 2

packages/web/src/                     ← Web UI
├── main.ts                           ← Entry point, wires everything
├── config-panel.ts                   ← Dropdown config
├── code-preview.ts                   ← Monaco editor
├── mock-metadata.ts                  ← Mock tensor shapes per task
├── hf-picker.ts                      ← HuggingFace model picker
├── try-it.ts                         ← iframe sandbox
└── style.css
```

---

## Engines

| Engine | Package | Supported Tasks |
|--------|---------|----------------|
| **ORT Web** (onnxruntime-web) | ort | All 19 tasks |
| **LiteRT.js** (TFLite) | litert | Image + Text + Multimodal tasks (not audio) |
| **WebNN API** | webnn | Image + Text + Multimodal tasks (not audio) |

## Frameworks

| Framework | Output | "Try it" in Web UI |
|-----------|--------|-------------------|
| HTML (single file) | index.html | Yes (iframe) |
| Vanilla + Vite | Multi-file JS/TS | No |
| React + Vite | React components | No |
| Next.js | Pages/components | No |
| SvelteKit | Svelte components | No |

## Input Modes

| Mode | Used By |
|------|---------|
| file | All tasks |
| camera | Image tasks (classification, detection, segmentation, depth-estimation, image-to-text) |
| video | Image tasks + feature-extraction |
| screen | Image tasks (not feature-extraction) |
| mic | Audio tasks (speech-to-text, audio-classification, audio-to-audio) |
