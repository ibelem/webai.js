/**
 * Generate mock ModelMetadata for the Web UI.
 *
 * In the CLI, metadata comes from parsing an actual .onnx/.tflite file.
 * In the Web UI, we create plausible shapes based on the selected task
 * so the assembler can generate correct code.
 */
const TASK_SHAPES = {
    'image-classification': {
        inputs: [{ name: 'input', dataType: 'float32', shape: [1, 3, 224, 224] }],
        outputs: [{ name: 'output', dataType: 'float32', shape: [1, 1000] }],
    },
    'object-detection': {
        inputs: [{ name: 'images', dataType: 'float32', shape: [1, 3, 640, 640] }],
        outputs: [{ name: 'output0', dataType: 'float32', shape: [1, 84, 8400] }],
    },
    'image-segmentation': {
        inputs: [{ name: 'input', dataType: 'float32', shape: [1, 3, 512, 512] }],
        outputs: [{ name: 'output', dataType: 'float32', shape: [1, 21, 512, 512] }],
    },
    'feature-extraction': {
        inputs: [{ name: 'input', dataType: 'float32', shape: [1, 3, 224, 224] }],
        outputs: [{ name: 'output', dataType: 'float32', shape: [1, 768] }],
    },
    'speech-to-text': {
        inputs: [{ name: 'input', dataType: 'float32', shape: [1, 80, 100] }],
        outputs: [{ name: 'output', dataType: 'float32', shape: [1, 100, 28] }],
    },
    'audio-classification': {
        inputs: [{ name: 'input', dataType: 'float32', shape: [1, 13, 100] }],
        outputs: [{ name: 'output', dataType: 'float32', shape: [1, 527] }],
    },
    'text-to-speech': {
        inputs: [{ name: 'input', dataType: 'float32', shape: [1, 100] }],
        outputs: [{ name: 'output', dataType: 'float32', shape: [1, 22050] }],
    },
    'text-classification': {
        inputs: [
            { name: 'input_ids', dataType: 'int64', shape: [1, 128] },
            { name: 'attention_mask', dataType: 'int64', shape: [1, 128] },
        ],
        outputs: [{ name: 'logits', dataType: 'float32', shape: [1, 2] }],
    },
    'text-generation': {
        inputs: [
            { name: 'input_ids', dataType: 'int64', shape: [1, 128] },
            { name: 'attention_mask', dataType: 'int64', shape: [1, 128] },
        ],
        outputs: [{ name: 'logits', dataType: 'float32', shape: [1, 128, 50257] }],
    },
    'zero-shot-classification': {
        inputs: [
            { name: 'input_ids', dataType: 'int64', shape: [1, 128] },
            { name: 'attention_mask', dataType: 'int64', shape: [1, 128] },
        ],
        outputs: [{ name: 'logits', dataType: 'float32', shape: [1, 3] }],
    },
    'fill-mask': {
        inputs: [
            { name: 'input_ids', dataType: 'int64', shape: [1, 128] },
            { name: 'attention_mask', dataType: 'int64', shape: [1, 128] },
        ],
        outputs: [{ name: 'logits', dataType: 'float32', shape: [1, 128, 30522] }],
    },
    'sentence-similarity': {
        inputs: [
            { name: 'input_ids', dataType: 'int64', shape: [1, 128] },
            { name: 'attention_mask', dataType: 'int64', shape: [1, 128] },
        ],
        outputs: [{ name: 'output', dataType: 'float32', shape: [1, 384] }],
    },
    'depth-estimation': {
        inputs: [{ name: 'input', dataType: 'float32', shape: [1, 3, 384, 384] }],
        outputs: [{ name: 'output', dataType: 'float32', shape: [1, 1, 384, 384] }],
    },
    'token-classification': {
        inputs: [
            { name: 'input_ids', dataType: 'int64', shape: [1, 128] },
            { name: 'attention_mask', dataType: 'int64', shape: [1, 128] },
        ],
        outputs: [{ name: 'logits', dataType: 'float32', shape: [1, 128, 9] }],
    },
    'question-answering': {
        inputs: [
            { name: 'input_ids', dataType: 'int64', shape: [1, 384] },
            { name: 'attention_mask', dataType: 'int64', shape: [1, 384] },
        ],
        outputs: [
            { name: 'start_logits', dataType: 'float32', shape: [1, 384] },
            { name: 'end_logits', dataType: 'float32', shape: [1, 384] },
        ],
    },
    'summarization': {
        inputs: [
            { name: 'input_ids', dataType: 'int64', shape: [1, 512] },
            { name: 'attention_mask', dataType: 'int64', shape: [1, 512] },
        ],
        outputs: [{ name: 'logits', dataType: 'float32', shape: [1, 128, 32128] }],
    },
    'translation': {
        inputs: [
            { name: 'input_ids', dataType: 'int64', shape: [1, 512] },
            { name: 'attention_mask', dataType: 'int64', shape: [1, 512] },
        ],
        outputs: [{ name: 'logits', dataType: 'float32', shape: [1, 128, 32128] }],
    },
    'image-to-text': {
        inputs: [{ name: 'pixel_values', dataType: 'float32', shape: [1, 3, 224, 224] }],
        outputs: [{ name: 'logits', dataType: 'float32', shape: [1, 64, 50257] }],
    },
    'audio-to-audio': {
        inputs: [{ name: 'input', dataType: 'float32', shape: [1, 1, 16000] }],
        outputs: [{ name: 'output', dataType: 'float32', shape: [1, 1, 16000] }],
    },
    'speaker-diarization': {
        inputs: [{ name: 'input', dataType: 'float32', shape: [1, 1, 16000] }],
        outputs: [{ name: 'output', dataType: 'float32', shape: [1, 500, 4] }],
    },
    'voice-activity-detection': {
        inputs: [{ name: 'input', dataType: 'float32', shape: [1, 1, 16000] }],
        outputs: [{ name: 'output', dataType: 'float32', shape: [1, 500] }],
    },
    'text2text-generation': {
        inputs: [
            { name: 'input_ids', dataType: 'int64', shape: [1, 512] },
            { name: 'attention_mask', dataType: 'int64', shape: [1, 512] },
        ],
        outputs: [{ name: 'logits', dataType: 'float32', shape: [1, 128, 32128] }],
    },
    'conversational': {
        inputs: [
            { name: 'input_ids', dataType: 'int64', shape: [1, 512] },
            { name: 'attention_mask', dataType: 'int64', shape: [1, 512] },
        ],
        outputs: [{ name: 'logits', dataType: 'float32', shape: [1, 512, 50257] }],
    },
    'table-question-answering': {
        inputs: [
            { name: 'input_ids', dataType: 'int64', shape: [1, 512] },
            { name: 'attention_mask', dataType: 'int64', shape: [1, 512] },
        ],
        outputs: [
            { name: 'start_logits', dataType: 'float32', shape: [1, 512] },
            { name: 'end_logits', dataType: 'float32', shape: [1, 512] },
        ],
    },
    'visual-question-answering': {
        inputs: [{ name: 'pixel_values', dataType: 'float32', shape: [1, 3, 224, 224] }],
        outputs: [{ name: 'logits', dataType: 'float32', shape: [1, 64, 50257] }],
    },
    'document-question-answering': {
        inputs: [{ name: 'pixel_values', dataType: 'float32', shape: [1, 3, 224, 224] }],
        outputs: [{ name: 'logits', dataType: 'float32', shape: [1, 64, 50257] }],
    },
    'image-text-to-text': {
        inputs: [{ name: 'pixel_values', dataType: 'float32', shape: [1, 3, 224, 224] }],
        outputs: [{ name: 'logits', dataType: 'float32', shape: [1, 128, 50257] }],
    },
};
export function createMockMetadata(task, format = 'onnx') {
    const shapes = TASK_SHAPES[task] ?? TASK_SHAPES['image-classification'];
    return {
        format,
        inputs: shapes.inputs,
        outputs: shapes.outputs,
    };
}
//# sourceMappingURL=mock-metadata.js.map