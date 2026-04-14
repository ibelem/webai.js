/**
 * Build synthetic ONNX protobuf buffers for testing.
 *
 * Creates minimal valid ONNX ModelProto with only graph I/O fields.
 * No weights, no nodes — just enough for metadata parsing and task detection.
 */

interface SyntheticTensor {
  name: string;
  /** ONNX elem_type enum: 1=float32, 2=uint8, 3=int8, 6=int32, 7=int64 */
  elemType?: number;
  shape: number[];
}

/** Encode a varint */
function varint(value: number): number[] {
  const bytes: number[] = [];
  let v = value >>> 0;
  while (v > 0x7f) {
    bytes.push((v & 0x7f) | 0x80);
    v >>>= 7;
  }
  bytes.push(v);
  return bytes;
}

/** Encode a protobuf tag (field number + wire type) */
function tag(fieldNum: number, wireType: number): number[] {
  return varint((fieldNum << 3) | wireType);
}

/** Encode a length-delimited field */
function lenDelim(fieldNum: number, data: number[]): number[] {
  return [...tag(fieldNum, 2), ...varint(data.length), ...data];
}

/** Encode a varint field */
function varintField(fieldNum: number, value: number): number[] {
  return [...tag(fieldNum, 0), ...varint(value)];
}

/** Encode a string field */
function stringField(fieldNum: number, value: string): number[] {
  const encoded = new TextEncoder().encode(value);
  return lenDelim(fieldNum, Array.from(encoded));
}

/** Encode a TensorShapeProto.Dimension */
function dimension(value: number): number[] {
  // DimensionProto with dim_value (field 1, varint)
  return varintField(1, value);
}

/** Encode a TensorShapeProto */
function tensorShape(dims: number[]): number[] {
  let bytes: number[] = [];
  for (const dim of dims) {
    bytes = [...bytes, ...lenDelim(1, dimension(dim))];
  }
  return bytes;
}

/** Encode a TensorTypeProto */
function tensorType(elemType: number, shape: number[]): number[] {
  return [
    ...varintField(1, elemType), // elem_type
    ...lenDelim(2, tensorShape(shape)), // shape
  ];
}

/** Encode a TypeProto (field 1 = tensor_type) */
function typeProto(elemType: number, shape: number[]): number[] {
  return lenDelim(1, tensorType(elemType, shape));
}

/** Encode a ValueInfoProto (field 1 = name, field 2 = type) */
function valueInfo(tensor: SyntheticTensor): number[] {
  return [
    ...stringField(1, tensor.name),
    ...lenDelim(2, typeProto(tensor.elemType ?? 1, tensor.shape)),
  ];
}

/**
 * Build a synthetic ONNX buffer with given inputs and outputs.
 * Creates a minimal valid ModelProto.
 */
export function buildSyntheticOnnx(
  inputs: SyntheticTensor[],
  outputs: SyntheticTensor[],
): Uint8Array {
  // Build GraphProto with input (field 11) and output (field 12)
  let graphBytes: number[] = [];

  for (const input of inputs) {
    graphBytes = [...graphBytes, ...lenDelim(11, valueInfo(input))];
  }
  for (const output of outputs) {
    graphBytes = [...graphBytes, ...lenDelim(12, valueInfo(output))];
  }

  // ModelProto: ir_version (field 1) + graph (field 7)
  const modelBytes = [
    ...varintField(1, 7), // ir_version = 7
    ...lenDelim(7, graphBytes), // graph
  ];

  return new Uint8Array(modelBytes);
}
