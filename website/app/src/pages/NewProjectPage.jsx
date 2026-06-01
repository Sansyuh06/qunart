import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useProjects } from '../context/ProjectContext.jsx';
import { useToast } from '../context/ToastContext.jsx';
import { fetchModelMetadata } from '../api/mockApi.js';
import { modelDatabase, compressionMethods } from '../data/mockData.js';
import { hardwareProfiles } from '../data/hardwareProfiles.js';

const WIZARD_STORAGE = 'qunart_wizard';

export default function NewProjectPage() {
  const navigate = useNavigate();
  const { createProject } = useProjects();
  const { addToast } = useToast();
  const [step, setStep] = useState(1);

  // Restore wizard state from localStorage via lazy initializers
  const savedWizard = (() => {
    try {
      const saved = localStorage.getItem(WIZARD_STORAGE);
      return saved ? JSON.parse(saved) : {};
    } catch { return {}; }
  })();

  // Step 1 state
  const [modelSource, setModelSource] = useState(savedWizard.modelSource || 'hub');
  const [modelId, setModelId] = useState(savedWizard.modelId || '');
  const [modelMeta, setModelMeta] = useState(null);
  const [loadingMeta, setLoadingMeta] = useState(false);

  // Step 2 state
  const [selectedHardware, setSelectedHardware] = useState(savedWizard.selectedHardware || '');

  // Step 3 state
  const [budgetSize, setBudgetSize] = useState(savedWizard.budgetSize || 512);
  const [budgetLatency, setBudgetLatency] = useState(savedWizard.budgetLatency || 100);
  const [budgetAccuracy, setBudgetAccuracy] = useState(savedWizard.budgetAccuracy ?? 5);
  const [selectedMethod, setSelectedMethod] = useState(savedWizard.selectedMethod || 'qubo-int8');

  useEffect(() => {
    localStorage.setItem(WIZARD_STORAGE, JSON.stringify({
      modelId, selectedHardware, budgetSize, budgetLatency, budgetAccuracy, selectedMethod, modelSource,
    }));
  }, [modelId, selectedHardware, budgetSize, budgetLatency, budgetAccuracy, selectedMethod, modelSource]);

  const handleModelLookup = async () => {
    if (!modelId.trim()) return;
    setLoadingMeta(true);
    const result = await fetchModelMetadata(modelId);
    setModelMeta(result.data);
    setLoadingMeta(false);
  };

  const handleDemoModel = () => {
    setModelSource('demo');
    setModelId('microsoft/DialoGPT-medium');
    setModelMeta(modelDatabase['microsoft/DialoGPT-medium']);
  };

  const handleCreate = () => {
    const method = compressionMethods.find(m => m.id === selectedMethod);
    const id = createProject({
      modelId: modelId || 'microsoft/DialoGPT-medium',
      hardware: selectedHardware || 'rpi4',
      method: method?.name || 'Quantum-QUBO Pruning + INT8 Quantization',
      pruneRatio: 20,
      quantization: selectedMethod.includes('int4') ? 'INT4' : 'INT8',
      budgetSize,
      budgetLatency,
      budgetAccuracy,
    });
    localStorage.removeItem(WIZARD_STORAGE);
    addToast('Project created.', 'success');
    navigate(`/project/${id}`);
  };

  const hw = hardwareProfiles.find(h => h.id === selectedHardware);

  const accuracyLabel = budgetAccuracy < 2 ? { text: 'Aggressive constraint', cls: 'badge-green' }
    : budgetAccuracy <= 5 ? { text: 'Standard', cls: 'badge-yellow' }
    : { text: 'Lenient', cls: 'badge-red' };

  return (
    <div className="page" style={{ maxWidth: 720, margin: '0 auto' }}>
      {/* Step Indicator */}
      <div className="step-indicator">
        {[1, 2, 3].map((s, i) => (
          <div key={s} style={{ display: 'flex', alignItems: 'center' }}>
            <div className={`step-dot ${step === s ? 'active' : step > s ? 'completed' : ''}`}>
              {step > s ? '✓' : s}
            </div>
            {i < 2 && <div className={`step-line ${step > s ? 'completed' : ''}`} />}
          </div>
        ))}
      </div>

      {/* Step 1: Model Source */}
      {step === 1 && (
        <div className="fade-in">
          <h2 style={{ marginBottom: 24 }}>Where's your model?</h2>

          <div className="flex gap-12" style={{ marginBottom: 20 }}>
            {[
              { key: 'hub', label: 'HuggingFace Hub' },
              { key: 'upload', label: 'Upload' },
              { key: 'demo', label: 'Demo Model' },
            ].map(opt => (
              <button
                key={opt.key}
                className={`btn ${modelSource === opt.key ? 'btn-primary' : ''}`}
                onClick={() => {
                  setModelSource(opt.key);
                  if (opt.key === 'demo') handleDemoModel();
                }}
              >
                {opt.label}
              </button>
            ))}
          </div>

          {modelSource === 'hub' && (
            <div className="section">
              <label className="form-label">Model ID</label>
              <div className="flex gap-8">
                <input
                  className="input input-mono"
                  placeholder="microsoft/phi-2"
                  value={modelId}
                  onChange={e => setModelId(e.target.value)}
                  onBlur={handleModelLookup}
                  onKeyDown={e => e.key === 'Enter' && handleModelLookup()}
                />
                <button className="btn" onClick={handleModelLookup} disabled={loadingMeta}>
                  {loadingMeta ? '...' : 'Lookup'}
                </button>
              </div>
            </div>
          )}

          {modelSource === 'upload' && (
            <div className="drop-zone" style={{ marginTop: 16 }}>
              <p>Drop .pt, .safetensors, .gguf, or .bin file here</p>
              <p className="text-xs text-muted" style={{ marginTop: 8 }}>Or click to browse</p>
            </div>
          )}

          {modelMeta && (
            <div className="card" style={{ marginTop: 20 }}>
              <div className="flex justify-between items-start">
                <h3>{modelMeta.name}</h3>
                <span className="badge badge-gray">{modelMeta.license}</span>
              </div>
              <div className="flex flex-col gap-8" style={{ marginTop: 12 }}>
                <div className="flex justify-between">
                  <span className="text-muted text-sm">Task</span>
                  <span className="text-sm">{modelMeta.task}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted text-sm">Parameters</span>
                  <span className="text-sm font-mono">{modelMeta.paramsStr}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted text-sm">Size</span>
                  <span className="text-sm font-mono">~{modelMeta.sizeFP32}GB FP32 / ~{modelMeta.sizeBF16}GB BF16</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted text-sm">Architecture</span>
                  <span className="text-sm font-mono">{modelMeta.arch}</span>
                </div>
              </div>
              {modelMeta.warning && (
                <div className="alert alert-yellow" style={{ marginTop: 12 }}>
                  ⚠ {modelMeta.warning}
                </div>
              )}
            </div>
          )}

          <div className="flex justify-end" style={{ marginTop: 24 }}>
            <button className="btn btn-primary" onClick={() => setStep(2)} disabled={!modelMeta && modelSource !== 'upload'}>
              Next →
            </button>
          </div>
        </div>
      )}

      {/* Step 2: Target Hardware */}
      {step === 2 && (
        <div className="fade-in">
          <h2 style={{ marginBottom: 24 }}>What device is this running on?</h2>

          <div className="grid-3">
            {hardwareProfiles.filter(h => h.id !== 'custom').map(h => (
              <div
                key={h.id}
                className={`hw-card ${selectedHardware === h.id ? 'selected' : ''}`}
                onClick={() => setSelectedHardware(h.id)}
              >
                <div className="flex items-center gap-8">
                  <span style={{ fontSize: '1.2rem' }}>{h.icon}</span>
                  <span className="hw-card-name">{h.name}</span>
                </div>
                <div className="hw-card-specs">
                  {h.arch} · {h.ram} RAM{h.gpu ? ` · ${h.gpu}` : ''}
                </div>
                {h.warning && (
                  <div className="text-xs text-yellow" style={{ marginTop: 4 }}>⚠ {h.warning}</div>
                )}
              </div>
            ))}
          </div>

          {hw && (
            <div className="card" style={{ marginTop: 20 }}>
              <div className="text-sm">
                Maximum viable compressed model size: <strong className="font-mono text-accent">~{hw.maxModelSize}{hw.maxModelSizeUnit}</strong>
              </div>
              <div className="text-sm text-muted" style={{ marginTop: 4 }}>
                Compatible methods: {hw.compatibleMethods.join(', ')}
              </div>
            </div>
          )}

          <div className="flex justify-between" style={{ marginTop: 24 }}>
            <button className="btn" onClick={() => setStep(1)}>← Back</button>
            <button className="btn btn-primary" onClick={() => setStep(3)} disabled={!selectedHardware}>
              Next →
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Compression Budget */}
      {step === 3 && (
        <div className="fade-in">
          <h2 style={{ marginBottom: 24 }}>What are your constraints?</h2>

          <div className="section">
            <div className="slider-container">
              <div className="slider-header">
                <label className="form-label">Model must fit in</label>
                <span className="slider-value">{budgetSize >= 1000 ? `${(budgetSize/1000).toFixed(1)}GB` : `${budgetSize}MB`}</span>
              </div>
              <input type="range" min={50} max={4000} step={50} value={budgetSize}
                onChange={e => setBudgetSize(Number(e.target.value))} />
            </div>
          </div>

          <div className="section">
            <div className="slider-container">
              <div className="slider-header">
                <label className="form-label">Max latency (ms/token)</label>
                <span className="slider-value">{budgetLatency >= 2000 ? 'Not critical' : `${budgetLatency}ms`}</span>
              </div>
              <input type="range" min={10} max={2000} step={10} value={budgetLatency}
                onChange={e => setBudgetLatency(Number(e.target.value))} />
            </div>
          </div>

          <div className="section">
            <div className="slider-container">
              <div className="slider-header">
                <label className="form-label">Max accuracy drop</label>
                <div className="flex items-center gap-8">
                  <span className="slider-value">{budgetAccuracy}%</span>
                  <span className={`badge ${accuracyLabel.cls}`}>{accuracyLabel.text}</span>
                </div>
              </div>
              <input type="range" min={0} max={10} step={0.5} value={budgetAccuracy}
                onChange={e => setBudgetAccuracy(Number(e.target.value))} />
            </div>
          </div>

          <div style={{ marginTop: 20 }}>
            <h3 style={{ marginBottom: 12 }}>Viable compression methods</h3>
            {compressionMethods.map(m => (
              <div
                key={m.id}
                className={`card card-interactive ${selectedMethod === m.id ? '' : ''}`}
                style={{
                  marginBottom: 8,
                  borderColor: selectedMethod === m.id ? 'var(--accent)' : undefined,
                  background: selectedMethod === m.id ? 'var(--accent-dim)' : undefined,
                }}
                onClick={() => setSelectedMethod(m.id)}
              >
                <div className="flex justify-between items-center">
                  <div>
                    <strong className="text-sm">{m.name}</strong>
                    <div className="text-xs text-muted" style={{ marginTop: 2 }}>{m.description}</div>
                  </div>
                  <div style={{ textAlign: 'right', flexShrink: 0, marginLeft: 12 }}>
                    <div className="font-mono text-sm">~{m.ratioRange[0]}–{m.ratioRange[1]}×</div>
                    <div className="font-mono text-xs text-muted">{m.accuracyRange[0]}–{m.accuracyRange[1]}% acc</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="flex justify-between" style={{ marginTop: 24 }}>
            <button className="btn" onClick={() => setStep(2)}>← Back</button>
            <button className="btn btn-primary btn-lg" onClick={handleCreate}>
              Create Project
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
