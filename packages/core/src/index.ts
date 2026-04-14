// @webai/core — knowledge layer
// Preprocessing, postprocessing, task profiles, model parsing, config resolution

export const VERSION = '0.1.0';

// Preprocessing
export { resizeImage, type ResizeOptions } from './preprocess/index.js';
export { normalize, type NormalizeOptions } from './preprocess/index.js';
export { toNCHW } from './preprocess/index.js';

// Postprocessing
export { softmax } from './postprocess/index.js';
export { topK, type TopKResult } from './postprocess/index.js';
export { argmax } from './postprocess/index.js';
export { nms, type BoundingBox } from './postprocess/index.js';

// Model parsing
export {
  detectFormat,
  parseOnnxMetadata,
  parseTfliteMetadata,
  parseModelMetadata,
  type ModelMetadata,
  type TensorInfo,
  type DataType,
  type ModelFormat,
} from './model-parser/index.js';

// Tasks
export {
  detectTask,
  TASK_PROFILES,
  type TaskType,
  type Confidence,
  type TaskCandidate,
  type TaskDetectionResult,
  type InputMode,
  type PreprocessDefaults,
  type TaskProfile,
} from './tasks/index.js';
