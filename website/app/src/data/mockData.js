// Realistic mock datasets — model specs, sample projects, runs

export const modelDatabase = {
  'microsoft/DialoGPT-medium': {
    name: 'DialoGPT-medium',
    fullName: 'microsoft/DialoGPT-medium',
    task: 'Causal LM (Conversational)',
    params: 354823168,
    paramsStr: '354M',
    sizeFP32: 1.34,
    sizeBF16: 0.67,
    arch: 'GPT2LMHeadModel',
    license: 'MIT',
    layers: 24,
    hiddenSize: 1024,
    attentionHeads: 16,
    vocabSize: 50257,
    ops: ['Linear', 'Embedding', 'LayerNorm', 'MultiHeadAttention', 'GeLU', 'Softmax', 'Conv1D', 'Dropout', 'Reshape'],
  },
  'microsoft/phi-2': {
    name: 'Phi-2',
    fullName: 'microsoft/phi-2',
    task: 'Causal LM',
    params: 2700000000,
    paramsStr: '2.7B',
    sizeFP32: 5.4,
    sizeBF16: 2.7,
    arch: 'PhiForCausalLM',
    license: 'MIT',
    layers: 32,
    hiddenSize: 2560,
    attentionHeads: 32,
    vocabSize: 51200,
    ops: ['Linear', 'Embedding', 'LayerNorm', 'MultiHeadAttention', 'GeLU', 'Softmax', 'Dropout', 'Reshape', 'RotaryEmbedding'],
    warning: 'Models over 7B are tough to compress for most edge devices without significant accuracy loss. At 2.7B, it\'s doable with aggressive pruning.',
  },
  'TinyLlama/TinyLlama-1.1B-Chat-v1.0': {
    name: 'TinyLlama-1.1B',
    fullName: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task: 'Causal LM (Chat)',
    params: 1100000000,
    paramsStr: '1.1B',
    sizeFP32: 4.4,
    sizeBF16: 2.2,
    arch: 'LlamaForCausalLM',
    license: 'Apache 2.0',
    layers: 22,
    hiddenSize: 2048,
    attentionHeads: 32,
    vocabSize: 32000,
    ops: ['Linear', 'Embedding', 'LayerNorm', 'MultiHeadAttention', 'GeLU', 'Softmax', 'Dropout', 'Reshape', 'RotaryEmbedding'],
  },
  'distilbert-base-uncased': {
    name: 'DistilBERT-base',
    fullName: 'distilbert-base-uncased',
    task: 'Masked LM / Classification',
    params: 66000000,
    paramsStr: '66M',
    sizeFP32: 0.25,
    sizeBF16: 0.13,
    arch: 'DistilBertModel',
    license: 'Apache 2.0',
    layers: 6,
    hiddenSize: 768,
    attentionHeads: 12,
    vocabSize: 30522,
    ops: ['Linear', 'Embedding', 'LayerNorm', 'MultiHeadAttention', 'GeLU', 'Softmax', 'Dropout', 'Reshape'],
  },
  'openai/whisper-tiny': {
    name: 'Whisper-tiny',
    fullName: 'openai/whisper-tiny',
    task: 'Speech Recognition',
    params: 39000000,
    paramsStr: '39M',
    sizeFP32: 0.15,
    sizeBF16: 0.08,
    arch: 'WhisperForConditionalGeneration',
    license: 'MIT',
    layers: 4,
    hiddenSize: 384,
    attentionHeads: 6,
    vocabSize: 51865,
    ops: ['Linear', 'Embedding', 'LayerNorm', 'MultiHeadAttention', 'GeLU', 'Softmax', 'Conv1D', 'Dropout', 'Reshape'],
  },
};

// Generate realistic heatmap sensitivity scores for a transformer model
export function generateSensitivityScores(numLayers) {
  const scores = [];
  for (let i = 0; i < numLayers; i++) {
    let base;
    // Early attention layers tend to be more sensitive
    if (i < 3) {
      base = 0.55 + Math.random() * 0.35; // 0.55 - 0.90
    }
    // Middle layers are generally less sensitive
    else if (i < numLayers - 3) {
      base = 0.12 + Math.random() * 0.35; // 0.12 - 0.47
    }
    // Final projection layers are sensitive
    else {
      base = 0.45 + Math.random() * 0.40; // 0.45 - 0.85
    }
    // Add some random spikes
    if (Math.random() < 0.12) {
      base = Math.min(1, base + 0.25);
    }
    scores.push({
      index: i,
      name: i < numLayers / 2
        ? `transformer.h.${i}.mlp.c_fc`
        : `transformer.h.${i}.mlp.c_proj`,
      type: 'Linear',
      params: Math.floor(800000 + Math.random() * 600000),
      sensitivity: parseFloat(base.toFixed(3)),
      locked: false,
      weightDistribution: generateWeightDistribution(),
    });
  }
  return scores;
}

function generateWeightDistribution() {
  // Generate a rough bell curve for the weight histogram
  const bins = 30;
  const data = [];
  for (let i = 0; i < bins; i++) {
    const x = (i - bins / 2) / (bins / 4);
    const y = Math.exp(-0.5 * x * x) + Math.random() * 0.1;
    data.push(parseFloat(y.toFixed(3)));
  }
  return data;
}

// Sample projects for seeding
export const sampleProjects = [
  {
    id: 'proj-1',
    name: 'DialoGPT Edge Bot',
    modelId: 'microsoft/DialoGPT-medium',
    hardware: 'rpi4',
    method: 'Quantum-QUBO Pruning + INT8',
    pruneRatio: 20,
    quantization: 'INT8',
    budgetSize: 512,
    budgetLatency: 100,
    budgetAccuracy: 5,
    createdAt: '2026-05-28T14:30:00Z',
    updatedAt: '2026-05-30T09:15:00Z',
    status: 'completed',
    runs: [
      {
        id: 'run-1-1',
        number: 1,
        method: 'Quantum-QUBO Pruning + INT8',
        pruneRatio: 20,
        quantization: 'INT8',
        hardware: 'rpi4',
        status: 'completed',
        startedAt: '2026-05-28T14:32:00Z',
        completedAt: '2026-05-28T14:47:00Z',
        duration: 912,
        originalParams: 354823168,
        compressedParams: 283858534,
        originalSize: 1340,
        compressedSize: 358,
        compressionRatio: 3.74,
        accuracyRetained: 94.2,
        accuracyTarget: 95,
        stages: [
          { name: 'Load', status: 'done', duration: 12 },
          { name: 'Calibrate', status: 'done', duration: 8 },
          { name: 'Analyze', status: 'done', duration: 45 },
          { name: 'QUBO Prune', status: 'done', duration: 380 },
          { name: 'Fine-tune', status: 'done', duration: 340 },
          { name: 'Quantize', status: 'done', duration: 95 },
          { name: 'Export', status: 'done', duration: 32 },
        ],
      },
      {
        id: 'run-1-2',
        number: 2,
        method: 'Quantum-QUBO Pruning + INT8',
        pruneRatio: 15,
        quantization: 'INT8',
        hardware: 'rpi4',
        status: 'completed',
        startedAt: '2026-05-29T10:00:00Z',
        completedAt: '2026-05-29T10:14:00Z',
        duration: 840,
        originalParams: 354823168,
        compressedParams: 301599593,
        originalSize: 1340,
        compressedSize: 402,
        compressionRatio: 3.33,
        accuracyRetained: 96.8,
        accuracyTarget: 95,
        stages: [
          { name: 'Load', status: 'done', duration: 11 },
          { name: 'Calibrate', status: 'done', duration: 8 },
          { name: 'Analyze', status: 'done', duration: 42 },
          { name: 'QUBO Prune', status: 'done', duration: 350 },
          { name: 'Fine-tune', status: 'done', duration: 310 },
          { name: 'Quantize', status: 'done', duration: 88 },
          { name: 'Export', status: 'done', duration: 31 },
        ],
      },
    ],
  },
  {
    id: 'proj-2',
    name: 'Phi-2 Mobile Assistant',
    modelId: 'microsoft/phi-2',
    hardware: 'android-high',
    method: 'Hybrid Pruning+INT8',
    pruneRatio: 25,
    quantization: 'INT8',
    budgetSize: 2000,
    budgetLatency: 200,
    budgetAccuracy: 3,
    createdAt: '2026-05-30T11:00:00Z',
    updatedAt: '2026-05-30T16:45:00Z',
    status: 'completed',
    runs: [
      {
        id: 'run-2-1',
        number: 1,
        method: 'Hybrid Pruning+INT8',
        pruneRatio: 25,
        quantization: 'INT8',
        hardware: 'android-high',
        status: 'completed',
        startedAt: '2026-05-30T11:05:00Z',
        completedAt: '2026-05-30T11:28:00Z',
        duration: 1380,
        originalParams: 2700000000,
        compressedParams: 2025000000,
        originalSize: 5400,
        compressedSize: 1420,
        compressionRatio: 3.80,
        accuracyRetained: 87.3,
        accuracyTarget: 97,
        stages: [
          { name: 'Load', status: 'done', duration: 45 },
          { name: 'Calibrate', status: 'done', duration: 22 },
          { name: 'Analyze', status: 'done', duration: 120 },
          { name: 'QUBO Prune', status: 'done', duration: 520 },
          { name: 'Fine-tune', status: 'done', duration: 480 },
          { name: 'Quantize', status: 'done', duration: 145 },
          { name: 'Export', status: 'done', duration: 48 },
        ],
      },
    ],
  },
  {
    id: 'proj-3',
    name: 'TinyLlama Jetson Deploy',
    modelId: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    hardware: 'jetson-nano',
    method: 'Quantum-QUBO Pruning + INT8',
    pruneRatio: 18,
    quantization: 'INT8',
    budgetSize: 1200,
    budgetLatency: 50,
    budgetAccuracy: 4,
    createdAt: '2026-05-31T08:00:00Z',
    updatedAt: '2026-05-31T08:00:00Z',
    status: 'new',
    runs: [],
  },
];

// Compression method specs
export const compressionMethods = [
  {
    id: 'qubo-int8',
    name: 'Quantum-QUBO Pruning + INT8 Quantization',
    shortName: 'QUBO + INT8',
    ratioRange: [3.5, 4.5],
    accuracyRange: [92, 97],
    description: 'Best overall — recommended for most configurations. Uses QUBO-guided structured pruning followed by INT8 quantization.',
  },
  {
    id: 'hybrid-int8',
    name: 'Hybrid Pruning + INT8 Quantization',
    shortName: 'Hybrid + INT8',
    ratioRange: [3.0, 4.0],
    accuracyRange: [90, 96],
    description: 'Classical greedy pruning with INT8. Faster than QUBO, slightly less optimal pruning decisions.',
  },
  {
    id: 'int8-only',
    name: 'INT8 Quantization Only',
    shortName: 'INT8 Only',
    ratioRange: [2.5, 3.5],
    accuracyRange: [95, 99],
    description: 'No pruning, quantization only. Safest accuracy but lower compression ratio.',
  },
  {
    id: 'qubo-int4',
    name: 'Quantum-QUBO Pruning + INT4 Quantization',
    shortName: 'QUBO + INT4',
    ratioRange: [5.0, 7.0],
    accuracyRange: [85, 93],
    description: 'Aggressive compression. Higher risk of accuracy loss, especially for complex models.',
  },
  {
    id: 'prune-only',
    name: 'Structured Pruning Only (FP32)',
    shortName: 'Prune Only',
    ratioRange: [1.2, 1.5],
    accuracyRange: [94, 98],
    description: 'Pruning without quantization. Minimal compression but preserves precision.',
  },
];

// Generate mock terminal log lines for a compression run
export function generateLogLines() {
  return [
    { text: '> Initializing compression pipeline...', type: 'info' },
    { text: '> Loading model: microsoft/DialoGPT-medium', type: 'info' },
    { text: '  Tokenizer loaded (vocab_size=50257)', type: 'info' },
    { text: '  Model loaded: 354,823,168 parameters', type: 'info' },
    { text: '  Device: CPU (CUDA not available)', type: 'info' },
    { text: '✓ Model loaded successfully [12.3s]', type: 'success' },
    { text: '> Preparing calibration dataset...', type: 'info' },
    { text: '  Task samples: 18 | Base pool: 52 | Total: 70', type: 'info' },
    { text: '  Effective task weight: 26%', type: 'info' },
    { text: '✓ Calibration dataset prepared [8.1s]', type: 'success' },
    { text: '> Computing layer sensitivity scores...', type: 'info' },
    { text: '  Analyzing layer 1/24: transformer.h.0.mlp.c_fc', type: 'info' },
    { text: '  Analyzing layer 4/24: transformer.h.3.mlp.c_fc', type: 'info' },
    { text: '  Analyzing layer 8/24: transformer.h.7.mlp.c_fc', type: 'info' },
    { text: '  ⚠ Layer 7 sensitivity: 0.823 (high — consider locking)', type: 'warn' },
    { text: '  Analyzing layer 12/24: transformer.h.11.mlp.c_proj', type: 'info' },
    { text: '  Analyzing layer 16/24: transformer.h.15.mlp.c_proj', type: 'info' },
    { text: '  Analyzing layer 20/24: transformer.h.19.mlp.c_proj', type: 'info' },
    { text: '  Analyzing layer 24/24: transformer.h.23.mlp.c_proj', type: 'info' },
    { text: '✓ Importance scores computed for 24 layers [45.2s]', type: 'success' },
    { text: '> Starting QUBO-guided pruning (target: 20%)...', type: 'info' },
    { text: '  Simulated Annealing: T=10.0, cooling=0.95, max_iter=400', type: 'info' },
    { text: '  Pruning transformer.h.0.mlp.c_fc: 4096 → 3276 neurons', type: 'info' },
    { text: '  Pruning transformer.h.3.mlp.c_fc: 4096 → 3276 neurons', type: 'info' },
    { text: '  Skipping transformer.h.7.mlp.c_fc (locked — high sensitivity)', type: 'warn' },
    { text: '  Pruning transformer.h.11.mlp.c_proj: 1024 → 819 neurons', type: 'info' },
    { text: '  Pruning transformer.h.19.mlp.c_proj: 1024 → 819 neurons', type: 'info' },
    { text: '  Pruning transformer.h.23.mlp.c_proj: 1024 → 819 neurons', type: 'info' },
    { text: '  Total neurons pruned: 14,336', type: 'info' },
    { text: '  Parameter reduction: 20.3%', type: 'info' },
    { text: '✓ QUBO pruning complete [380s]', type: 'success' },
    { text: '> Fine-tuning for performance recovery...', type: 'info' },
    { text: '  Optimizer: AdamW (lr=1e-5, weight_decay=0.01)', type: 'info' },
    { text: '  Step 20/100, Loss: 3.4521', type: 'info' },
    { text: '  Step 40/100, Loss: 2.8934', type: 'info' },
    { text: '  Step 60/100, Loss: 2.5127', type: 'info' },
    { text: '  Step 80/100, Loss: 2.3401', type: 'info' },
    { text: '  Step 100/100, Loss: 2.2156', type: 'info' },
    { text: '✓ Fine-tuning complete [340s]', type: 'success' },
    { text: '> Applying INT8 dynamic quantization...', type: 'info' },
    { text: '  Quantizing Linear layers to qint8', type: 'info' },
    { text: '  Original: 1.340 GB → Quantized: 0.358 GB', type: 'info' },
    { text: '  Compression ratio: 3.74x', type: 'info' },
    { text: '✓ Quantization complete [95s]', type: 'success' },
    { text: '> Exporting compressed model...', type: 'info' },
    { text: '  Saving quantized_model.pth', type: 'info' },
    { text: '  Saving tokenizer files', type: 'info' },
    { text: '  Generating compression_metrics.json', type: 'info' },
    { text: '  Generating DEPLOY_README.md', type: 'info' },
    { text: '✓ Export complete [32s]', type: 'success' },
    { text: '', type: 'info' },
    { text: '═══════════════════════════════════════════', type: 'success' },
    { text: '  COMPRESSION COMPLETE', type: 'success' },
    { text: '  Ratio: 3.74x | Size: 1.34GB → 358MB', type: 'success' },
    { text: '  Accuracy retained: 94.2%', type: 'success' },
    { text: '═══════════════════════════════════════════', type: 'success' },
  ];
}
