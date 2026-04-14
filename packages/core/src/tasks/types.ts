/**
 * Task types and profiles for web AI inference.
 */

export type TaskType =
  | 'image-classification'
  | 'object-detection'
  | 'image-segmentation'
  | 'feature-extraction'
  | 'speech-to-text'
  | 'audio-classification'
  | 'text-to-speech'
  | 'text-classification'
  | 'text-generation';

export type Confidence = 'high' | 'medium' | 'low';

export interface TaskCandidate {
  task: TaskType;
  confidence: Confidence;
  reason: string;
}

export interface TaskDetectionResult {
  /** Best-matching task, or null if no match */
  detected: TaskCandidate | null;
  /** All candidates considered, sorted by confidence descending */
  candidates: TaskCandidate[];
}

export type InputMode = 'file' | 'camera' | 'video' | 'mic' | 'screen';

/** Preprocessing chain defaults for a task */
export interface PreprocessDefaults {
  imageSize: number;
  mean: readonly number[];
  std: readonly number[];
  layout: 'nchw' | 'nhwc';
}

/** Per-task configuration profile */
export interface TaskProfile {
  task: TaskType;
  /** Human-readable label */
  label: string;
  /** Default input mode */
  defaultInput: InputMode;
  /** Supported input modes */
  supportedInputs: readonly InputMode[];
  /** Default preprocessing for image tasks, null for non-image tasks */
  preprocess: PreprocessDefaults | null;
  /** Postprocessing steps needed */
  postprocess: readonly string[];
}
