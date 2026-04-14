# webai.js — Task Implementation Progress

## Implemented Tasks (10/25)

### Image Tasks (4/5)

| Task | Profile | Preprocess | Postprocess | Emitter | HTML | React | Vanilla | Next.js | Svelte | Web UI Mock | Status |
|------|---------|------------|-------------|---------|------|-------|---------|---------|--------|-------------|--------|
| image-classification | Y | Y | softmax+topK | Y | Y | Y | Y | Y | Y | Y | Done |
| object-detection | Y | Y | nms+bbox | Y | Y | Y | Y | Y | Y | Y | Done |
| image-segmentation | Y | Y | argmax mask | Y | Y | Y | Y | Y | Y | Y | Done |
| feature-extraction | Y | Y | passthrough | Y | Y | Y | Y | Y | Y | Y | Done |
| depth-estimation | | | | | | | | | | | Planned |

### Audio Tasks (3/6)

| Task | Profile | Preprocess | Postprocess | Emitter | HTML | React | Vanilla | Next.js | Svelte | Web UI Mock | Status |
|------|---------|------------|-------------|---------|------|-------|---------|---------|--------|-------------|--------|
| audio-classification | Y | MFCC | softmax+topK | Y | Y | Y | Y | Y | Y | Y | Done |
| speech-to-text | Y | MFCC | CTC decode | Y | Y | Y | Y | Y | Y | Y | Done |
| text-to-speech | Y | BPE tokenizer | audio playback | Y | Y | Y | Y | Y | Y | Y | Done |
| audio-to-audio | | | | | | | | | | | Planned |
| speaker-diarization | | | | | | | | | | | — |
| voice-activity-detection | | | | | | | | | | | — |

### Text Tasks (3/12)

| Task | Profile | Preprocess | Postprocess | Emitter | HTML | React | Vanilla | Next.js | Svelte | Web UI Mock | Status |
|------|---------|------------|-------------|---------|------|-------|---------|---------|--------|-------------|--------|
| text-classification | Y | BPE tokenizer | softmax+topK | Y | Y | Y | Y | Y | Y | Y | Done |
| text-generation | Y | BPE tokenizer | argmax loop | Y | Y | Y | Y | Y | Y | Y | Done |
| zero-shot-classification | Y | BPE tokenizer | entailment softmax | Y | Y | Y | Y | Y | Y | Y | Done |
| fill-mask | | | | | | | | | | | Planned |
| token-classification (NER) | | | | | | | | | | | Planned |
| question-answering | | | | | | | | | | | Planned |
| summarization | | | | | | | | | | | Planned |
| translation | | | | | | | | | | | Planned |
| sentence-similarity | | | | | | | | | | | Planned |
| text2text-generation | | | | | | | | | | | — |
| conversational | | | | | | | | | | | — |
| table-question-answering | | | | | | | | | | | — |

### Multimodal Tasks (0/4)

| Task | Profile | Preprocess | Postprocess | Emitter | HTML | React | Vanilla | Next.js | Svelte | Web UI Mock | Status |
|------|---------|------------|-------------|---------|------|-------|---------|---------|--------|-------------|--------|
| image-to-text | | | | | | | | | | | Planned |
| visual-question-answering | | | | | | | | | | | — |
| document-question-answering | | | | | | | | | | | — |
| image-text-to-text | | | | | | | | | | | — |

---

## Implementation Priority

### Phase 3A — Low-Hanging Fruit

Reuse existing preprocessing/postprocessing with minor variations.

| Task | What's Needed | Effort |
|------|--------------|--------|
| **fill-mask** | BPE tokenizer (exists), single forward pass, replace `[MASK]` token with top predictions | Low |
| **sentence-similarity** | Feature-extraction (exists) x2 inputs, cosine similarity | Low |
| **depth-estimation** | Image preprocess (exists), output is HxW float map (render as grayscale/colormap) | Low |

### Phase 3B — Medium Effort (Text)

Need new postprocessing logic or decoder loops.

| Task | What's Needed | Effort |
|------|--------------|--------|
| **token-classification (NER)** | BPE tokenizer (exists), per-token label output, entity span merging | Medium |
| **question-answering** | BPE tokenizer (exists), encode question+context, extract start/end span logits | Medium |
| **summarization** | Seq2seq encoder-decoder loop (new), similar to text-generation but with encoder | Medium |
| **translation** | Same architecture as summarization (seq2seq) | Medium |

### Phase 3C — Medium Effort (Vision + Audio)

| Task | What's Needed | Effort |
|------|--------------|--------|
| **image-to-text** | Image encoder (exists) + text decoder loop (new) | Medium |
| **audio-to-audio** | Audio preprocess (exists), output is waveform (render + playback) | Low-Medium |

### Phase 3D — High Effort (Multimodal)

| Task | What's Needed | Effort |
|------|--------------|--------|
| **visual-question-answering** | Image encoder + text encoder + decoder, dual input UI | High |
| **document-question-answering** | Similar to VQA but with document/page images | High |
| **image-text-to-text** | Vision-language model, combined pipeline | High |

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
│   ├── types.ts                      ← TaskType union
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
| **ORT Web** (onnxruntime-web) | ort | All 10 tasks |
| **LiteRT.js** (TFLite) | litert | Image + Text tasks |
| **WebNN API** | webnn | Image + Text tasks |

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
| camera | Image tasks |
| video | Image tasks + feature-extraction |
| screen | Image tasks (not feature-extraction) |
| mic | Audio tasks (speech-to-text, audio-classification) |
