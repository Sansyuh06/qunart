// Op compatibility matrix for export format pre-check

export const opCompatibility = [
  { op: 'Linear',              onnx: 'green', tflite: 'green',  coreml: 'green',  tensorrt: 'green' },
  { op: 'Embedding',           onnx: 'green', tflite: 'green',  coreml: 'green',  tensorrt: 'green' },
  { op: 'LayerNorm',           onnx: 'green', tflite: 'yellow', coreml: 'green',  tensorrt: 'green',
    caveats: { tflite: 'Fused LayerNorm may be decomposed into primitive ops. Slight performance overhead.' } },
  { op: 'MultiHeadAttention',  onnx: 'green', tflite: 'yellow', coreml: 'yellow', tensorrt: 'green',
    caveats: { tflite: 'Must be unrolled into individual matmul + softmax ops. No native fused attention.', coreml: 'Supported via decomposition. May require coremltools >= 7.0.' } },
  { op: 'GeLU',                onnx: 'green', tflite: 'yellow', coreml: 'green',  tensorrt: 'green',
    caveats: { tflite: 'Approximated as sigmoid(1.702 * x) * x. Small numerical difference.' } },
  { op: 'Softmax',             onnx: 'green', tflite: 'green',  coreml: 'green',  tensorrt: 'green' },
  { op: 'Conv1D',              onnx: 'green', tflite: 'green',  coreml: 'green',  tensorrt: 'green' },
  { op: 'MatMul',              onnx: 'green', tflite: 'green',  coreml: 'green',  tensorrt: 'green' },
  { op: 'Dropout',             onnx: 'green', tflite: 'green',  coreml: 'green',  tensorrt: 'green' },
  { op: 'Reshape',             onnx: 'green', tflite: 'green',  coreml: 'green',  tensorrt: 'green' },
  { op: 'Transpose',           onnx: 'green', tflite: 'green',  coreml: 'green',  tensorrt: 'yellow',
    caveats: { tensorrt: 'Dynamic shape transpose may require explicit profile configuration.' } },
  { op: 'Gather',              onnx: 'green', tflite: 'yellow', coreml: 'green',  tensorrt: 'green',
    caveats: { tflite: 'Limited to specific axis configurations. Negative indices not supported.' } },
  { op: 'RotaryEmbedding',     onnx: 'yellow', tflite: 'red',   coreml: 'yellow', tensorrt: 'yellow',
    caveats: { onnx: 'Custom op — requires onnxruntime >= 1.16 or manual decomposition.', tflite: 'Not supported. Must decompose into sin/cos + multiply manually.', coreml: 'Requires custom layer registration.', tensorrt: 'Plugin required. Not available in stock TensorRT.' } },
  { op: 'FlashAttention',      onnx: 'red',    tflite: 'red',   coreml: 'red',    tensorrt: 'yellow',
    caveats: { onnx: 'No ONNX op mapping. Must fall back to standard attention.', tflite: 'Not supported.', coreml: 'Not supported.', tensorrt: 'Supported via TensorRT-LLM plugin only.' } },
  { op: 'KVCache',             onnx: 'yellow', tflite: 'red',   coreml: 'red',    tensorrt: 'green',
    caveats: { onnx: 'Requires dynamic shapes and careful IO binding.', tflite: 'Stateful ops not supported in standard TFLite.', coreml: 'No native KV cache support. Must handle externally.' } },
];

export function getCompatibilityForModel(modelOps) {
  // modelOps is an array of op names present in the model
  return opCompatibility.filter(row => modelOps.includes(row.op));
}

export function hasIncompatibleOps(format) {
  return opCompatibility.some(row => row[format] === 'red');
}

export function getFormatSummary(format) {
  const red = opCompatibility.filter(r => r[format] === 'red').length;
  const yellow = opCompatibility.filter(r => r[format] === 'yellow').length;
  const green = opCompatibility.filter(r => r[format] === 'green').length;
  return { red, yellow, green };
}
