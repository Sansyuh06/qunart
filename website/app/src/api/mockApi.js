// Mock API layer — simulates all backend calls with realistic delays

import { modelDatabase, generateSensitivityScores, generateLogLines, sampleProjects } from '../data/mockData.js';
import { estimateBenchmark } from '../data/hardwareProfiles.js';

// Simulate network delay
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Fetch model metadata from "HuggingFace"
export async function fetchModelMetadata(modelId) {
  await delay(800);
  const normalized = modelId.trim();

  // Direct match
  if (modelDatabase[normalized]) {
    return { success: true, data: modelDatabase[normalized] };
  }

  // Fuzzy match — check if any key contains the input
  for (const [key, value] of Object.entries(modelDatabase)) {
    if (key.toLowerCase().includes(normalized.toLowerCase()) ||
        value.name.toLowerCase().includes(normalized.toLowerCase())) {
      return { success: true, data: value };
    }
  }

  // Unknown model — generate plausible metadata
  return {
    success: true,
    data: {
      name: normalized.split('/').pop(),
      fullName: normalized,
      task: 'Causal LM',
      params: 125000000,
      paramsStr: '125M',
      sizeFP32: 0.47,
      sizeBF16: 0.24,
      arch: 'TransformerLM',
      license: 'Unknown',
      layers: 12,
      hiddenSize: 768,
      attentionHeads: 12,
      vocabSize: 30000,
      ops: ['Linear', 'Embedding', 'LayerNorm', 'MultiHeadAttention', 'GeLU', 'Softmax', 'Dropout', 'Reshape'],
    },
  };
}

// Simulate a compression run with progressive stage updates
export function runCompression(projectConfig, onStageUpdate, onLogLine, onComplete) {
  const stages = [
    { name: 'Load', duration: 2000 },
    { name: 'Calibrate', duration: 1500 },
    { name: 'Analyze', duration: 3000 },
    { name: 'QUBO Prune', duration: 4000 },
    { name: 'Fine-tune', duration: 3500 },
    { name: 'Quantize', duration: 2000 },
    { name: 'Export', duration: 1500 },
  ];

  const logLines = generateLogLines();
  let logIndex = 0;
  let stageIndex = 0;
  let cancelled = false;

  const stageResults = stages.map(s => ({
    name: s.name,
    status: 'pending',
    duration: null,
  }));

  // Log line emitter
  const logInterval = setInterval(() => {
    if (cancelled || logIndex >= logLines.length) {
      clearInterval(logInterval);
      return;
    }
    onLogLine(logLines[logIndex]);
    logIndex++;
  }, 350);

  // Stage progression
  function advanceStage() {
    if (cancelled || stageIndex >= stages.length) {
      clearInterval(logInterval);
      if (!cancelled) {
        // Generate final metrics
        const model = modelDatabase[projectConfig.modelId] || modelDatabase['microsoft/DialoGPT-medium'];
        const pruneRatio = projectConfig.pruneRatio || 20;
        const compressedParams = Math.floor(model.params * (1 - pruneRatio / 100));
        const compressedSize = Math.floor(model.sizeFP32 * 1000 * (1 - pruneRatio / 100) * 0.27);
        const accuracyRetained = 88 + Math.random() * 10;

        const metrics = {
          originalParams: model.params,
          compressedParams,
          originalSize: Math.floor(model.sizeFP32 * 1000),
          compressedSize,
          compressionRatio: parseFloat((model.sizeFP32 * 1000 / compressedSize).toFixed(2)),
          accuracyRetained: parseFloat(accuracyRetained.toFixed(1)),
          accuracyTarget: 100 - (projectConfig.budgetAccuracy || 5),
        };
        onComplete(metrics, stageResults);
      }
      return;
    }

    stageResults[stageIndex].status = 'running';
    onStageUpdate([...stageResults]);

    const stageDuration = stages[stageIndex].duration;
    setTimeout(() => {
      if (cancelled) return;
      stageResults[stageIndex].status = 'done';
      stageResults[stageIndex].duration = parseFloat((stageDuration / 1000).toFixed(1));
      stageIndex++;
      onStageUpdate([...stageResults]);
      advanceStage();
    }, stageDuration);
  }

  advanceStage();

  // Return cancel function
  return () => {
    cancelled = true;
    clearInterval(logInterval);
    if (stageIndex < stages.length) {
      stageResults[stageIndex].status = 'error';
      onStageUpdate([...stageResults]);
    }
  };
}

// Get heatmap sensitivity data
export async function getHeatmapData(modelId) {
  await delay(2000);
  const model = modelDatabase[modelId] || modelDatabase['microsoft/DialoGPT-medium'];
  return generateSensitivityScores(model.layers);
}

// Get hardware benchmark estimates
export async function getBenchmarkEstimate(hardwareId, compressedSizeMB, quantType) {
  await delay(500);
  return estimateBenchmark(hardwareId, compressedSizeMB, quantType || 'INT8');
}

// Get recovery suggestions based on run metrics
export async function getRecoverySuggestions(runMetrics, config) {
  await delay(300);
  const gap = (100 - (config.budgetAccuracy || 5)) - runMetrics.accuracyRetained;
  if (gap <= 0) return [];

  return [
    {
      id: 'lock-layers',
      title: 'Lock high-sensitivity layers from pruning',
      description: `Layers 7, 8, and 11 have sensitivity scores above 0.7. Skipping these from pruning typically recovers 1.5–2.5% accuracy.`,
      estimatedImpact: '+2.1%',
      action: 'apply-lock',
    },
    {
      id: 'add-calibration',
      title: 'Add domain-specific calibration samples',
      description: `Your current task weight is 26%. Adding 30+ more task-specific examples and raising weight to 40% typically improves accuracy in specialized domains by 0.8–1.2%.`,
      estimatedImpact: '+0.9%',
      action: 'update-calibration',
    },
    {
      id: 'reduce-pruning',
      title: `Reduce pruning ratio from ${config.pruneRatio || 20}% to ${Math.max(10, (config.pruneRatio || 20) - 6)}%`,
      description: `At ${config.pruneRatio || 20}% pruning this model loses more accuracy than average for its architecture. ${Math.max(10, (config.pruneRatio || 20) - 6)}% is the estimated inflection point for stable accuracy.`,
      estimatedImpact: '+1.3%',
      tradeoff: '−0.8× compression tradeoff',
      action: 'apply-prune-reduction',
    },
  ];
}

// Simulate export
export async function exportModel(runId, format) {
  await delay(1500);

  const readmContent = `# Deployment Guide — ${format.toUpperCase()} Export

## Loading the Model
\`\`\`python
# ${format === 'onnx' ? 'import onnxruntime as ort\nsess = ort.InferenceSession("model.onnx")' :
  format === 'tflite' ? 'import tensorflow as tf\ninterpreter = tf.lite.Interpreter(model_path="model.tflite")\ninterpreter.allocate_tensors()' :
  format === 'coreml' ? 'import coremltools as ct\nmodel = ct.models.MLModel("model.mlpackage")' :
  'import tensorrt as trt\nruntime = trt.Runtime(trt.Logger(trt.Logger.WARNING))'}
\`\`\`

## Hardware-specific flags
- num_threads=4 for RPi4
- Use GPU delegate on Android for better performance
- Set NNAPI delegate for Android Neural Networks API

## Known Limitations
- INT8 models may show slight numerical differences vs FP32
- Batch size must be 1 for edge deployment
- Max sequence length is limited by available RAM
`;

  // Trigger a dummy download
  const blob = new Blob([readmContent], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `DEPLOY_README_${format}.md`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);

  return { success: true, filename: `compressed_model.${format === 'coreml' ? 'mlpackage' : format}` };
}

// Get all sample projects (for seeding)
export function getSampleProjects() {
  return JSON.parse(JSON.stringify(sampleProjects));
}
