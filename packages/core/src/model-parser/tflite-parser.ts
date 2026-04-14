/**
 * Minimal TFLite FlatBuffers metadata parser.
 *
 * Reads only subgraph inputs/outputs tensor descriptions.
 * Never touches the buffers vector (weight data).
 *
 * TFLite FlatBuffers schema (simplified for what we read):
 *   Model (root)
 *     subgraphs: [SubGraph]      // field offset 4 (vtable idx 4)
 *     SubGraph
 *       tensors: [Tensor]        // field offset 0 (vtable idx 4)
 *       inputs: [int]            // field offset 1 (vtable idx 6)
 *       outputs: [int]           // field offset 2 (vtable idx 8)
 *     Tensor
 *       name: string             // field offset 0 (vtable idx 4)
 *       shape: [int]             // field offset 1 (vtable idx 6)
 *       type: TensorType(byte)   // field offset 2 (vtable idx 8)
 */

import type { ModelMetadata, TensorInfo, DataType } from './types.js';

// TFLite TensorType enum → DataType
const TFLITE_TYPES: Record<number, DataType> = {
  0: 'float32',
  1: 'float16',
  2: 'int32',
  3: 'uint8',
  4: 'int64',
  // 5: string (not a numeric type, skip)
  6: 'int8', // bool → int8
  7: 'int8', // int16 → int8 (approximate)
  // 8: complex64 (not supported)
  9: 'int8',
  10: 'float32', // float64 → float32 (approximate)
  11: 'uint64',
  // 12: resource (not supported)
  // 13: variant (not supported)
  14: 'uint32',
  // 15: uint16
  16: 'int4',
};

class FlatBufferReader {
  private view: DataView;
  private buf: Uint8Array;

  constructor(buffer: Uint8Array) {
    this.buf = buffer;
    this.view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  }

  int32(offset: number): number {
    return this.view.getInt32(offset, true);
  }

  uint16(offset: number): number {
    return this.view.getUint16(offset, true);
  }

  uint8(offset: number): number {
    return this.buf[offset];
  }

  /** Get the root table position */
  rootTable(): number {
    return this.int32(0);
  }

  /** Read a field offset from a table's vtable. Returns 0 if field not present. */
  fieldOffset(tablePos: number, fieldIndex: number): number {
    const vtableOffset = this.int32(tablePos);
    const vtablePos = tablePos - vtableOffset;
    const vtableSize = this.uint16(vtablePos);
    const slotOffset = 4 + fieldIndex * 2; // 4 bytes for vtable size + object size
    if (slotOffset >= vtableSize) return 0;
    return this.uint16(vtablePos + slotOffset);
  }

  /** Read an offset field (indirect reference) */
  indirect(tablePos: number, fieldIndex: number): number {
    const off = this.fieldOffset(tablePos, fieldIndex);
    if (off === 0) return 0;
    const fieldPos = tablePos + off;
    return fieldPos + this.int32(fieldPos);
  }

  /** Read a vector length */
  vectorLen(vectorPos: number): number {
    return this.int32(vectorPos);
  }

  /** Read an int32 from a vector at index */
  vectorInt32(vectorPos: number, index: number): number {
    return this.int32(vectorPos + 4 + index * 4);
  }

  /** Read a table from a vector of tables */
  vectorTable(vectorPos: number, index: number): number {
    const elemPos = vectorPos + 4 + index * 4;
    return elemPos + this.int32(elemPos);
  }

  /** Read a string */
  string(tablePos: number, fieldIndex: number): string {
    const strPos = this.indirect(tablePos, fieldIndex);
    if (strPos === 0) return '';
    const len = this.int32(strPos);
    return new TextDecoder().decode(this.buf.slice(strPos + 4, strPos + 4 + len));
  }

  /** Read a scalar uint8 field */
  scalarUint8(tablePos: number, fieldIndex: number, defaultVal = 0): number {
    const off = this.fieldOffset(tablePos, fieldIndex);
    if (off === 0) return defaultVal;
    return this.uint8(tablePos + off);
  }
}

/**
 * Parse TFLite model metadata from a raw buffer.
 * Reads only the first subgraph's tensor descriptions.
 *
 * @param buffer - Raw TFLite model file bytes
 * @returns Model metadata with input/output tensor descriptions
 * @throws Error if the buffer cannot be parsed
 */
export function parseTfliteMetadata(buffer: Uint8Array): ModelMetadata {
  try {
    const fb = new FlatBufferReader(buffer);
    const rootPos = fb.rootTable();

    // Model.subgraphs (field index 0)
    const subgraphsPos = fb.indirect(rootPos, 0);
    if (subgraphsPos === 0) {
      throw new Error('no subgraphs found');
    }

    const numSubgraphs = fb.vectorLen(subgraphsPos);
    if (numSubgraphs === 0) {
      throw new Error('empty subgraphs vector');
    }

    // Read first subgraph
    const sgPos = fb.vectorTable(subgraphsPos, 0);

    // SubGraph.tensors (field index 0)
    const tensorsPos = fb.indirect(sgPos, 0);
    const numTensors = tensorsPos !== 0 ? fb.vectorLen(tensorsPos) : 0;

    // Parse all tensor descriptions
    const tensors: TensorInfo[] = [];
    for (let i = 0; i < numTensors; i++) {
      const tPos = fb.vectorTable(tensorsPos, i);

      // Tensor.name (field 0)
      const name = fb.string(tPos, 0);

      // Tensor.shape (field 1) — vector of int32
      const shapePos = fb.indirect(tPos, 1);
      const shape: (number | string)[] = [];
      if (shapePos !== 0) {
        const shapeLen = fb.vectorLen(shapePos);
        for (let s = 0; s < shapeLen; s++) {
          const dim = fb.vectorInt32(shapePos, s);
          shape.push(dim === -1 ? `dim_${s}` : dim);
        }
      }

      // Tensor.type (field 2) — uint8 enum
      const typeVal = fb.scalarUint8(tPos, 2);
      const dataType: DataType = TFLITE_TYPES[typeVal] ?? 'float32';

      tensors.push({ name, dataType, shape });
    }

    // SubGraph.inputs (field 1) — vector of int32 indices
    const inputsPos = fb.indirect(sgPos, 1);
    const inputs: TensorInfo[] = [];
    if (inputsPos !== 0) {
      const numInputs = fb.vectorLen(inputsPos);
      for (let i = 0; i < numInputs; i++) {
        const idx = fb.vectorInt32(inputsPos, i);
        if (idx >= 0 && idx < tensors.length) {
          inputs.push(tensors[idx]);
        }
      }
    }

    // SubGraph.outputs (field 2) — vector of int32 indices
    const outputsPos = fb.indirect(sgPos, 2);
    const outputTensors: TensorInfo[] = [];
    if (outputsPos !== 0) {
      const numOutputs = fb.vectorLen(outputsPos);
      for (let i = 0; i < numOutputs; i++) {
        const idx = fb.vectorInt32(outputsPos, i);
        if (idx >= 0 && idx < tensors.length) {
          outputTensors.push(tensors[idx]);
        }
      }
    }

    return { format: 'tflite', inputs, outputs: outputTensors };
  } catch (e) {
    throw new Error(
      `Could not parse TFLite model: ${e instanceof Error ? e.message : 'invalid FlatBuffer'}`,
      { cause: e },
    );
  }
}
