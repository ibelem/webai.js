/**
 * Build synthetic TFLite FlatBuffers buffers for testing.
 *
 * Creates minimal valid TFLite Model with one subgraph containing
 * only tensor descriptions and input/output index arrays.
 * No operators, no buffers (weight data).
 *
 * FlatBuffers layout (little-endian):
 *   [root_offset:i32] [file_id:4bytes "TFL3"] [... tables and vectors ...]
 */

interface SyntheticTensor {
  name: string;
  /** TFLite TensorType: 0=FLOAT32, 1=FLOAT16, 2=INT32, 3=UINT8, 4=INT64 */
  type?: number;
  shape: number[];
}

/**
 * Build a synthetic TFLite buffer with given inputs and outputs.
 *
 * Manually lays out FlatBuffers binary with correct relative offsets.
 */
export function buildSyntheticTflite(
  inputs: SyntheticTensor[],
  outputs: SyntheticTensor[],
): Uint8Array {
  const allTensors = [...inputs, ...outputs];
  const inputIndices = inputs.map((_, i) => i);
  const outputIndices = outputs.map((_, i) => inputs.length + i);

  // Layout:
  //   0: root_offset (int32) → points to Model table
  //   4: "TFL3" file identifier
  //   8+: data (strings, vectors, tables, vtables)

  const data: number[] = [];

  function pushI32(v: number): number {
    const off = data.length;
    const n = v | 0;
    data.push(n & 0xff, (n >> 8) & 0xff, (n >> 16) & 0xff, (n >> 24) & 0xff);
    return off;
  }

  function pushU16(v: number): void {
    data.push(v & 0xff, (v >> 8) & 0xff);
  }

  function pushU8(v: number): void {
    data.push(v & 0xff);
  }

  function align4(): void {
    while (data.length % 4 !== 0) data.push(0);
  }

  function writeString(s: string): number {
    align4();
    const off = data.length;
    const enc = new TextEncoder().encode(s);
    pushI32(enc.length);
    for (const b of enc) data.push(b);
    data.push(0);
    return off;
  }

  function writeI32Vec(vals: number[]): number {
    align4();
    const off = data.length;
    pushI32(vals.length);
    for (const v of vals) pushI32(v);
    return off;
  }

  // Step 1: Write all string data and shape vectors
  const stringOffsets: number[] = [];
  const shapeOffsets: number[] = [];

  for (const t of allTensors) {
    stringOffsets.push(writeString(t.name));
    shapeOffsets.push(writeI32Vec(t.shape));
  }

  // Step 2: Write input/output index vectors
  const inputVecOff = writeI32Vec(inputIndices);
  const outputVecOff = writeI32Vec(outputIndices);

  // Step 3: Write tensor tables
  const tensorTableOffsets: number[] = [];

  for (let i = 0; i < allTensors.length; i++) {
    align4();

    // VTable for Tensor: [size:u16=10] [objSize:u16] [name:u16] [shape:u16] [type:u16]
    const vtableOff = data.length;
    pushU16(10);
    pushU16(16);
    pushU16(4);
    pushU16(8);
    pushU16(12);

    align4();

    const tableOff = data.length;
    tensorTableOffsets.push(tableOff);

    pushI32(tableOff - vtableOff); // soffset back to vtable
    pushI32(stringOffsets[i] - data.length); // name offset
    pushI32(shapeOffsets[i] - data.length); // shape offset
    pushU8(allTensors[i].type ?? 0); // type
    while (data.length % 4 !== 0) data.push(0);
  }

  // Step 4: Write tensor offset vector
  align4();
  const tensorVecOff = data.length;
  pushI32(allTensors.length);
  for (const tOff of tensorTableOffsets) {
    pushI32(tOff - data.length);
  }

  // Step 5: SubGraph table
  align4();
  const sgVtableOff = data.length;
  pushU16(10);
  pushU16(16);
  pushU16(4);
  pushU16(8);
  pushU16(12);

  align4();
  const sgTableOff = data.length;

  pushI32(sgTableOff - sgVtableOff);
  pushI32(tensorVecOff - data.length);
  pushI32(inputVecOff - data.length);
  pushI32(outputVecOff - data.length);

  // Step 6: Subgraphs vector (1 subgraph)
  align4();
  const subgraphsVecOff = data.length;
  pushI32(1);
  pushI32(sgTableOff - data.length);

  // Step 7: Model table
  align4();
  const modelVtableOff = data.length;
  pushU16(6);
  pushU16(8);
  pushU16(4);

  align4();
  const modelTableOff = data.length;

  pushI32(modelTableOff - modelVtableOff);
  pushI32(subgraphsVecOff - data.length);

  // Build final buffer: [root_offset:i32] ["TFL3"] [data...]
  const header: number[] = [];
  const rootOffset = 8 + modelTableOff;
  header.push(rootOffset & 0xff, (rootOffset >> 8) & 0xff, (rootOffset >> 16) & 0xff, (rootOffset >> 24) & 0xff);
  header.push(0x54, 0x46, 0x4c, 0x33); // "TFL3"

  return new Uint8Array([...header, ...data]);
}
