# webai.js — Task Implementation Progress

## Implemented Tasks (19/25)

### Image Tasks (5/5) ✓

| Task | Profile | Preprocess | Postprocess | Emitter | HTML | React | Vanilla | Next.js | Svelte | Web UI Mock | Status |
|------|---------|------------|-------------|---------|------|-------|---------|---------|--------|-------------|--------|
| image-classification | Y | Y | softmax+topK | Y | Y | Y | Y | Y | Y | Y | Done |
| object-detection | Y | Y | nms+bbox | Y | Y | Y | Y | Y | Y | Y | Done |
| image-segmentation | Y | Y | argmax mask | Y | Y | Y | Y | Y | Y | Y | Done |
| feature-extraction | Y | Y | passthrough | Y | Y | Y | Y | Y | Y | Y | Done |
| depth-estimation | Y | Y | depthNormalize+colormap | Y | — | — | — | — | — | Y | Phase 3A |

### Audio Tasks (4/6)

| Task | Profile | Preprocess | Postprocess | Emitter | HTML | React | Vanilla | Next.js | Svelte | Web UI Mock | Status |
|------|---------|------------|-------------|---------|------|-------|---------|---------|--------|-------------|--------|
| audio-classification | Y | MFCC | softmax+topK | Y | Y | Y | Y | Y | Y | Y | Done |
| speech-to-text | Y | MFCC | CTC decode | Y | Y | Y | Y | Y | Y | Y | Done |
| text-to-speech | Y | BPE tokenizer | audio playback | Y | Y | Y | Y | Y | Y | Y | Done |
| audio-to-audio | Y | MFCC | normalizeWaveform+playAudio | Y | — | — | — | — | — | Y | Phase 3C |
| speaker-diarization | | | | | | | | | | | — |
| voice-activity-detection | | | | | | | | | | | — |

### Text Tasks (9/12)

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
| text2text-generation | | | | | | | | | | | — |
| conversational | | | | | | | | | | | — |
| table-question-answering | | | | | | | | | | | — |

### Multimodal Tasks (1/4)

| Task | Profile | Preprocess | Postprocess | Emitter | HTML | React | Vanilla | Next.js | Svelte | Web UI Mock | Status |
|------|---------|------------|-------------|---------|------|-------|---------|---------|--------|-------------|--------|
| image-to-text | Y | Y (image) | seq2seqGreedyDecode | Y | — | — | — | — | — | Y | Phase 3C |
| visual-question-answering | | | | | | | | | | | — |
| document-question-answering | | | | | | | | | | | — |
| image-text-to-text | | | | | | | | | | | — |

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

### Phase 3D — High Effort (Multimodal) — Not Started

| Task | What's Needed | Effort |
|------|--------------|--------|
| **visual-question-answering** | Image encoder + text encoder + decoder, dual input UI | High |
| **document-question-answering** | Similar to VQA but with document/page images | High |
| **image-text-to-text** | Vision-language model, combined pipeline | High |

---

## Remaining Work

- **Framework UI for Phase 3 tasks** — 9 new tasks have emitters but not framework-specific UI templates (HTML/React/Vanilla/Next.js/SvelteKit). Currently they generate code using the default/fallback UI.
- **Phase 3D (multimodal)** — visual-question-answering, document-question-answering, image-text-to-text. Requires dual-input UI and more complex encoder-decoder pipelines.
- **speaker-diarization, voice-activity-detection** — Not planned for near term.

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
│   ├── types.ts                      ← TaskType union (19 tasks)
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
