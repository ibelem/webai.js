/**
 * Types for parsed model metadata.
 * Aligned with model2webnn's TensorInfo/ModelMetadata contract
 * so we can swap in model2webnn as a dependency later.
 */

/** WebNN-aligned data type names */
export type DataType =
  | 'float32'
  | 'float16'
  | 'int64'
  | 'uint64'
  | 'int32'
  | 'uint32'
  | 'int8'
  | 'uint8'
  | 'int4'
  | 'uint4';

/** Description of a single input or output tensor */
export interface TensorInfo {
  /** Tensor name from the model graph */
  name: string;
  /** Element data type */
  dataType: DataType;
  /** Shape dimensions. Numbers for static dims, strings for dynamic/symbolic dims. */
  shape: (number | string)[];
}

/** Lightweight model metadata — graph I/O only, no weights */
export interface ModelMetadata {
  format: 'onnx' | 'tflite';
  inputs: TensorInfo[];
  outputs: TensorInfo[];
}

/** Model file format */
export type ModelFormat = 'onnx' | 'tflite' | 'unknown';
