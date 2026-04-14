/**
 * Minimal ONNX protobuf metadata parser.
 *
 * Reads only graph inputs and outputs (fields 11, 12 of GraphProto).
 * Skips weight data entirely via length-varint jumps.
 * Similar zero-copy approach to model2webnn (commits 1a0a6dc + 5dff577).
 *
 * When model2webnn is available as an npm dependency, this can be replaced
 * with a direct import. This implementation covers Phase 1a testing needs.
 */

import type { ModelMetadata, TensorInfo, DataType } from './types.js';

// ONNX protobuf element type enum → DataType string
const ONNX_ELEM_TYPE: Record<number, DataType> = {
  1: 'float32',
  2: 'uint8',
  3: 'int8',
  5: 'float16',
  6: 'int32',
  7: 'int64',
  10: 'float16',
  11: 'float32', // double → float32 (approximate)
  12: 'uint32',
  13: 'uint64',
};

/** Read a protobuf varint, return [value, bytesConsumed] */
function readVarint(buf: Uint8Array, offset: number): [number, number] {
  let result = 0;
  let shift = 0;
  let pos = offset;

  while (pos < buf.length) {
    const byte = buf[pos];
    result |= (byte & 0x7f) << shift;
    pos++;
    if ((byte & 0x80) === 0) break;
    shift += 7;
    if (shift > 35) throw new Error('Varint too long');
  }

  return [result >>> 0, pos - offset];
}

/** Skip a protobuf field based on wire type */
function skipField(buf: Uint8Array, offset: number, wireType: number): number {
  switch (wireType) {
    case 0: { // varint
      const [, len] = readVarint(buf, offset);
      return offset + len;
    }
    case 1: // 64-bit
      return offset + 8;
    case 2: { // length-delimited
      const [length, lenBytes] = readVarint(buf, offset);
      return offset + lenBytes + length;
    }
    case 5: // 32-bit
      return offset + 4;
    default:
      throw new Error(`Unknown wire type: ${wireType}`);
  }
}

/**
 * Parse a TensorShapeProto.Dimension repeated field.
 * Each dim has field 1 (dim_value, varint) or field 2 (dim_param, string).
 */
function parseShape(buf: Uint8Array, start: number, end: number): (number | string)[] {
  const dims: (number | string)[] = [];
  let pos = start;

  while (pos < end) {
    const [tag, tagLen] = readVarint(buf, pos);
    pos += tagLen;
    const fieldNum = tag >> 3;
    const wireType = tag & 0x7;

    if (fieldNum === 1 && wireType === 2) {
      // dim is a DimensionProto, length-delimited
      const [dimLen, dimLenBytes] = readVarint(buf, pos);
      pos += dimLenBytes;
      const dimEnd = pos + dimLen;

      // Parse DimensionProto fields
      let dimValue: number | string | null = null;
      let dimPos = pos;
      while (dimPos < dimEnd) {
        const [dimTag, dimTagLen] = readVarint(buf, dimPos);
        dimPos += dimTagLen;
        const dimField = dimTag >> 3;
        const dimWire = dimTag & 0x7;

        if (dimField === 1 && dimWire === 0) {
          // dim_value (int64 as varint)
          const [val, valLen] = readVarint(buf, dimPos);
          dimPos += valLen;
          dimValue = val;
        } else if (dimField === 2 && dimWire === 2) {
          // dim_param (string)
          const [strLen, strLenBytes] = readVarint(buf, dimPos);
          dimPos += strLenBytes;
          dimValue = new TextDecoder().decode(buf.slice(dimPos, dimPos + strLen));
          dimPos += strLen;
        } else {
          dimPos = skipField(buf, dimPos, dimWire);
        }
      }

      dims.push(dimValue ?? 0);
      pos = dimEnd;
    } else {
      pos = skipField(buf, pos, wireType);
    }
  }

  return dims;
}

/**
 * Parse a TensorTypeProto: elem_type (field 1) + shape (field 2).
 */
function parseTensorType(
  buf: Uint8Array,
  start: number,
  end: number,
): { dataType: DataType; shape: (number | string)[] } {
  let dataType: DataType = 'float32';
  let shape: (number | string)[] = [];
  let pos = start;

  while (pos < end) {
    const [tag, tagLen] = readVarint(buf, pos);
    pos += tagLen;
    const fieldNum = tag >> 3;
    const wireType = tag & 0x7;

    if (fieldNum === 1 && wireType === 0) {
      // elem_type (int32)
      const [val, valLen] = readVarint(buf, pos);
      pos += valLen;
      dataType = ONNX_ELEM_TYPE[val] ?? 'float32';
    } else if (fieldNum === 2 && wireType === 2) {
      // shape (TensorShapeProto)
      const [shapeLen, shapeLenBytes] = readVarint(buf, pos);
      pos += shapeLenBytes;
      shape = parseShape(buf, pos, pos + shapeLen);
      pos += shapeLen;
    } else {
      pos = skipField(buf, pos, wireType);
    }
  }

  return { dataType, shape };
}

/**
 * Parse a ValueInfoProto: name (field 1) + type (field 2).
 */
function parseValueInfo(buf: Uint8Array, start: number, end: number): TensorInfo {
  let name = '';
  let dataType: DataType = 'float32';
  let shape: (number | string)[] = [];
  let pos = start;

  while (pos < end) {
    const [tag, tagLen] = readVarint(buf, pos);
    pos += tagLen;
    const fieldNum = tag >> 3;
    const wireType = tag & 0x7;

    if (fieldNum === 1 && wireType === 2) {
      // name (string)
      const [strLen, strLenBytes] = readVarint(buf, pos);
      pos += strLenBytes;
      name = new TextDecoder().decode(buf.slice(pos, pos + strLen));
      pos += strLen;
    } else if (fieldNum === 2 && wireType === 2) {
      // type (TypeProto)
      const [typeLen, typeLenBytes] = readVarint(buf, pos);
      pos += typeLenBytes;
      const typeEnd = pos + typeLen;

      // TypeProto: field 1 is tensor_type (TensorTypeProto)
      let typePos = pos;
      while (typePos < typeEnd) {
        const [typeTag, typeTagLen] = readVarint(buf, typePos);
        typePos += typeTagLen;
        const typeField = typeTag >> 3;
        const typeWire = typeTag & 0x7;

        if (typeField === 1 && typeWire === 2) {
          const [ttLen, ttLenBytes] = readVarint(buf, typePos);
          typePos += ttLenBytes;
          const tt = parseTensorType(buf, typePos, typePos + ttLen);
          dataType = tt.dataType;
          shape = tt.shape;
          typePos += ttLen;
        } else {
          typePos = skipField(buf, typePos, typeWire);
        }
      }
      pos = typeEnd;
    } else {
      pos = skipField(buf, pos, wireType);
    }
  }

  return { name, dataType, shape };
}

/**
 * Parse GraphProto: reads only input (field 11) and output (field 12).
 * Skips all other fields (nodes, initializers, etc.) in O(1) per field.
 */
function parseGraph(
  buf: Uint8Array,
  start: number,
  end: number,
): { inputs: TensorInfo[]; outputs: TensorInfo[] } {
  const inputs: TensorInfo[] = [];
  const outputs: TensorInfo[] = [];
  let pos = start;

  while (pos < end) {
    const [tag, tagLen] = readVarint(buf, pos);
    pos += tagLen;
    const fieldNum = tag >> 3;
    const wireType = tag & 0x7;

    if (fieldNum === 11 && wireType === 2) {
      // input (ValueInfoProto)
      const [len, lenBytes] = readVarint(buf, pos);
      pos += lenBytes;
      inputs.push(parseValueInfo(buf, pos, pos + len));
      pos += len;
    } else if (fieldNum === 12 && wireType === 2) {
      // output (ValueInfoProto)
      const [len, lenBytes] = readVarint(buf, pos);
      pos += lenBytes;
      outputs.push(parseValueInfo(buf, pos, pos + len));
      pos += len;
    } else {
      pos = skipField(buf, pos, wireType);
    }
  }

  return { inputs, outputs };
}

/**
 * Parse ONNX model metadata from a raw buffer.
 * Reads only graph inputs/outputs. Skips weights and nodes entirely.
 *
 * @param buffer - Raw ONNX model file bytes
 * @returns Model metadata with input/output tensor descriptions
 * @throws Error if the buffer cannot be parsed
 */
export function parseOnnxMetadata(buffer: Uint8Array): ModelMetadata {
  let pos = 0;
  let inputs: TensorInfo[] = [];
  let outputs: TensorInfo[] = [];

  try {
    while (pos < buffer.length) {
      const [tag, tagLen] = readVarint(buffer, pos);
      pos += tagLen;
      const fieldNum = tag >> 3;
      const wireType = tag & 0x7;

      if (fieldNum === 7 && wireType === 2) {
        // graph (GraphProto) — the field we care about
        const [graphLen, graphLenBytes] = readVarint(buffer, pos);
        pos += graphLenBytes;
        const graph = parseGraph(buffer, pos, pos + graphLen);
        inputs = graph.inputs;
        outputs = graph.outputs;
        pos += graphLen;
      } else {
        pos = skipField(buffer, pos, wireType);
      }
    }
  } catch (e) {
    throw new Error(
      `Could not parse ONNX model: ${e instanceof Error ? e.message : 'invalid protobuf'}`,
      { cause: e },
    );
  }

  if (inputs.length === 0 && outputs.length === 0) {
    throw new Error('Could not parse ONNX model: no graph inputs or outputs found');
  }

  return { format: 'onnx', inputs, outputs };
}
