import { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useProjects } from '../context/ProjectContext.jsx';
import { useToast } from '../context/ToastContext.jsx';
import { runCompression as mockRunCompression, getBenchmarkEstimate, getRecoverySuggestions, exportModel } from '../api/mockApi.js';
import { modelDatabase } from '../data/mockData.js';
import { getHardwareProfile } from '../data/hardwareProfiles.js';
import { opCompatibility } from '../data/opCompatibility.js';

export default function RunPage() {
  const { id, runId } = useParams();
  const navigate = useNavigate();
  const { getProject, updateRun } = useProjects();
  const { addToast } = useToast();
  const project = getProject(id);
  const run = project?.runs.find(r => r.id === runId);

  const [stages, setStages] = useState(run?.stages || [
    { name: 'Load', status: 'pending', duration: null },
    { name: 'Calibrate', status: 'pending', duration: null },
    { name: 'Analyze', status: 'pending', duration: null },
    { name: 'QUBO Prune', status: 'pending', duration: null },
    { name: 'Fine-tune', status: 'pending', duration: null },
    { name: 'Quantize', status: 'pending', duration: null },
    { name: 'Export', status: 'pending', duration: null },
  ]);
  const [logLines, setLogLines] = useState([]);
  const [metrics, setMetrics] = useState(run?.status === 'completed' ? {
    originalParams: run.originalParams,
    compressedParams: run.compressedParams,
    originalSize: run.originalSize,
    compressedSize: run.compressedSize,
    compressionRatio: run.compressionRatio,
    accuracyRetained: run.accuracyRetained,
    accuracyTarget: run.accuracyTarget,
  } : null);
  const [completed, setCompleted] = useState(run?.status === 'completed');
  const [benchmark, setBenchmark] = useState(null);
  const [suggestions, setSuggestions] = useState([]);
  const [showConfirmCancel, setShowConfirmCancel] = useState(false);
  const [exporting, setExporting] = useState('');

  const cancelRef = useRef(null);
  const logRef = useRef(null);

  const model = modelDatabase[project?.modelId];
  const hw = getHardwareProfile(project?.hardware);
  const stageIcons = ['📦', '📚', '🔍', '✂️', '🔧', '⚡', '📤'];

  // Start the compression simulation if run is new (running status)
  useEffect(() => {
    if (!project || !run || run.status === 'completed' || run.status === 'cancelled') return;

    const cancel = mockRunCompression(
      project,
      (newStages) => setStages(newStages),
      (line) => setLogLines(prev => [...prev, line]),
      (finalMetrics, finalStages) => {
        setMetrics(finalMetrics);
        setStages(finalStages);
        setCompleted(true);
        updateRun(id, runId, {
          ...finalMetrics,
          stages: finalStages,
          status: 'completed',
          completedAt: new Date().toISOString(),
          duration: Math.floor((Date.now() - new Date(run.startedAt).getTime()) / 1000),
        });
        addToast('Compression complete.', 'success');
      },
    );
    cancelRef.current = cancel;

    return () => {
      if (cancelRef.current) cancelRef.current();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Auto-scroll terminal
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logLines]);

  // Fetch benchmark + suggestions after completion
  useEffect(() => {
    if (!completed || !metrics || !hw) return;
    getBenchmarkEstimate(project.hardware, metrics.compressedSize, project.quantization).then(setBenchmark);
    getRecoverySuggestions(metrics, project).then(setSuggestions);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [completed]);

  const handleCancel = () => {
    if (cancelRef.current) cancelRef.current();
    updateRun(id, runId, { status: 'cancelled' });
    setShowConfirmCancel(false);
    addToast('Run cancelled.', 'warning');
  };

  const handleExport = async (format) => {
    setExporting(format);
    await exportModel(runId, format);
    setExporting('');
    addToast(`Export complete: ${format}`, 'success');
  };

  if (!project || !run) {
    return <div className="page"><p className="text-muted">Run not found.</p></div>;
  }

  const formatParams = (n) => {
    if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(0)}M`;
    return n.toLocaleString();
  };

  const formatSize = (mb) => {
    if (mb >= 1000) return `${(mb / 1000).toFixed(2)}GB`;
    return `${mb}MB`;
  };

  const modelOps = model?.ops || ['Linear', 'Embedding', 'LayerNorm', 'MultiHeadAttention', 'GeLU', 'Softmax', 'Dropout', 'Reshape'];
  const relevantOps = opCompatibility.filter(row => modelOps.includes(row.op));

  return (
    <div className="page">
      {/* Header */}
      <div className="flex justify-between items-center" style={{ marginBottom: 20 }}>
        <div>
          <h2>Run #{run.number} — {run.method}</h2>
          <div className="flex items-center gap-8" style={{ marginTop: 4 }}>
            {hw && <span className="text-sm text-muted">{hw.icon} {hw.name}</span>}
            <div className="flex items-center gap-4">
              <span className={`status-dot ${completed ? 'status-dot-green' : run.status === 'cancelled' ? 'status-dot-red' : 'status-dot-yellow'}`} />
              <span className="text-sm" style={{ textTransform: 'capitalize' }}>{completed ? 'Completed' : run.status}</span>
            </div>
          </div>
        </div>
        {!completed && run.status !== 'cancelled' && (
          <button className="btn btn-danger-outline" onClick={() => setShowConfirmCancel(true)}>
            Cancel run
          </button>
        )}
      </div>

      {/* Stage Pipeline */}
      <div className="pipeline">
        {stages.map((stage, i) => (
          <div key={stage.name} style={{ display: 'flex', alignItems: 'center' }}>
            <div className={`pipeline-stage ${stage.status}`}>
              <div className="pipeline-icon">
                {stage.status === 'done' ? '✓' : stageIcons[i]}
              </div>
              <span className="pipeline-label">{stage.name}</span>
              {stage.duration != null && (
                <span className="pipeline-duration">{stage.duration}s</span>
              )}
            </div>
            {i < stages.length - 1 && (
              <div className={`pipeline-connector ${stage.status === 'done' ? 'done' : ''}`} />
            )}
          </div>
        ))}
      </div>

      {/* Live Metrics */}
      <div className="grid-2" style={{ gridTemplateColumns: 'repeat(4, 1fr)', marginTop: 20, marginBottom: 20 }}>
        <div className="stat-tile">
          <span className="stat-label">Parameters</span>
          <span className={`stat-value ${completed ? 'metric-flash' : ''}`} style={{ fontSize: '1.2rem' }}>
            {metrics ? `${formatParams(metrics.originalParams)} → ${formatParams(metrics.compressedParams)}` : '—'}
          </span>
          {metrics && <span className="stat-change text-green">−{((1 - metrics.compressedParams / metrics.originalParams) * 100).toFixed(1)}%</span>}
        </div>
        <div className="stat-tile">
          <span className="stat-label">Size</span>
          <span className={`stat-value ${completed ? 'metric-flash' : ''}`} style={{ fontSize: '1.2rem' }}>
            {metrics ? `${formatSize(metrics.originalSize)} → ${formatSize(metrics.compressedSize)}` : '—'}
          </span>
          {metrics && <span className="stat-change text-green">−{((1 - metrics.compressedSize / metrics.originalSize) * 100).toFixed(1)}%</span>}
        </div>
        <div className="stat-tile">
          <span className="stat-label">Accuracy Retained</span>
          <span className={`stat-value ${completed ? 'metric-flash' : ''}`} style={{ fontSize: '1.2rem' }}>
            {metrics ? `${metrics.accuracyRetained}%` : '—'}
          </span>
        </div>
        <div className="stat-tile">
          <span className="stat-label">Compression</span>
          <span className={`stat-value stat-value-accent ${completed ? 'metric-flash' : ''}`} style={{ fontSize: '1.5rem' }}>
            {metrics ? `${metrics.compressionRatio}×` : '—'}
          </span>
        </div>
      </div>

      {/* Terminal Log */}
      {!completed && (
        <div className="terminal" ref={logRef}>
          {logLines.map((line, i) => (
            <div key={i} className={`log-line log-${line.type}`}>{line.text}</div>
          ))}
          {logLines.length === 0 && <div className="log-line log-info">Waiting for pipeline to start...</div>}
        </div>
      )}

      {/* Results (post-completion) */}
      {completed && metrics && (
        <>
          {/* Benchmark Predictor */}
          {benchmark && (
            <div className="card" style={{ marginTop: 24 }}>
              <h3 style={{ marginBottom: 12 }}>Estimated on-device performance — {hw?.name}</h3>
              <div className="flex flex-col gap-12">
                {benchmark.ram && (
                  <div className="flex justify-between items-center">
                    <span className="text-sm">RAM usage</span>
                    <div className="flex items-center gap-8">
                      <span className="font-mono text-sm">~{benchmark.ram[0]}–{benchmark.ram[1]}MB</span>
                      <span className={`confidence-${benchmark.confidence}`}>
                        {benchmark.confidence === 'high' ? 'High confidence' : benchmark.confidence === 'medium' ? 'Medium confidence' : 'Low confidence'}
                      </span>
                    </div>
                  </div>
                )}
                {benchmark.latency && (
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Latency</span>
                    <div className="flex items-center gap-8">
                      <span className="font-mono text-sm">{benchmark.latency[0]}–{benchmark.latency[1]}ms/token</span>
                      <span className={`confidence-${benchmark.confidence}`}>
                        {benchmark.confidence} confidence
                      </span>
                    </div>
                  </div>
                )}
                {benchmark.throughput && (
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Throughput</span>
                    <span className="font-mono text-sm">~{benchmark.throughput[0]}–{benchmark.throughput[1]} tokens/sec</span>
                  </div>
                )}
              </div>
              <p className="text-xs text-dim" style={{ marginTop: 12 }}>{benchmark.note}</p>
            </div>
          )}

          {/* Recovery Suggestions */}
          {suggestions.length > 0 && metrics.accuracyRetained < metrics.accuracyTarget && (
            <div style={{ marginTop: 24 }}>
              <h3 style={{ marginBottom: 4 }}>Accuracy fell short — here's why and what to try</h3>
              <p className="text-sm text-muted" style={{ marginBottom: 16 }}>
                Target was ≥{metrics.accuracyTarget}% retention. Got {metrics.accuracyRetained}%.
                Gap: −{(metrics.accuracyTarget - metrics.accuracyRetained).toFixed(1)}%
              </p>
              <div className="flex flex-col gap-12">
                {suggestions.map(s => (
                  <div key={s.id} className="suggestion-card">
                    <h4>{s.title}</h4>
                    <p className="text-sm text-muted">{s.description}</p>
                    <div className="flex justify-between items-center">
                      <span className="suggestion-impact">Estimated: {s.estimatedImpact}</span>
                      <button className="btn btn-sm btn-primary" onClick={() => {
                        if (s.action === 'update-calibration') navigate(`/project/${id}`);
                        else {
                          addToast('Config applied. Re-run to test.', 'success');
                          navigate(`/project/${id}`);
                        }
                      }}>
                        {s.action === 'update-calibration' ? 'Update calibration data' : 'Apply and re-run'}
                      </button>
                    </div>
                    {s.tradeoff && <span className="text-xs text-muted">{s.tradeoff}</span>}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Export Panel */}
          <div style={{ marginTop: 24 }}>
            <h3 style={{ marginBottom: 12 }}>Export Compressed Model</h3>

            {/* Op Compatibility Table */}
            <div className="table-container" style={{ marginBottom: 16 }}>
              <table>
                <thead>
                  <tr>
                    <th>Operation</th>
                    <th>ONNX</th>
                    <th>TFLite</th>
                    <th>CoreML</th>
                    <th>TensorRT</th>
                  </tr>
                </thead>
                <tbody>
                  {relevantOps.map(row => (
                    <tr key={row.op}>
                      <td className="font-mono text-sm">{row.op}</td>
                      {['onnx', 'tflite', 'coreml', 'tensorrt'].map(fmt => (
                        <td key={fmt}>
                          <span
                            className={`op-${row[fmt]}`}
                            title={row.caveats?.[fmt] || ''}
                          >
                            {row[fmt] === 'green' ? '✓' : row[fmt] === 'yellow' ? '⚠' : '✗'}
                          </span>
                          {row[fmt] === 'yellow' && row.caveats?.[fmt] && (
                            <span className="text-xs text-dim" style={{ marginLeft: 4 }}>*</span>
                          )}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Export buttons */}
            <div className="flex gap-8">
              {[
                { format: 'onnx', label: 'ONNX (.onnx)', disabled: false },
                { format: 'tflite', label: 'TFLite (.tflite)', disabled: relevantOps.some(r => r.tflite === 'red') },
                { format: 'coreml', label: 'CoreML (.mlpackage)', disabled: false },
                { format: 'tensorrt', label: 'TensorRT (.engine)', disabled: false },
              ].map(exp => (
                <button
                  key={exp.format}
                  className="btn"
                  disabled={exp.disabled || exporting === exp.format}
                  onClick={() => handleExport(exp.format)}
                  title={exp.disabled ? 'Incompatible ops detected' : ''}
                  style={{ opacity: exp.disabled ? 0.4 : 1 }}
                >
                  {exporting === exp.format ? 'Generating...' : exp.label}
                </button>
              ))}
            </div>
          </div>
        </>
      )}

      {/* Confirm Cancel Dialog */}
      {showConfirmCancel && (
        <div className="confirm-overlay" onClick={() => setShowConfirmCancel(false)}>
          <div className="confirm-dialog" onClick={e => e.stopPropagation()}>
            <h3>Cancel this run?</h3>
            <p>Partial results will be saved.</p>
            <div className="confirm-actions">
              <button className="btn" onClick={() => setShowConfirmCancel(false)}>Keep running</button>
              <button className="btn btn-primary" onClick={handleCancel}>Cancel run</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
