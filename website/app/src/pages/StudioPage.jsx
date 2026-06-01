import { useState, useEffect, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useProjects } from '../context/ProjectContext.jsx';
import { useToast } from '../context/ToastContext.jsx';
import { getHeatmapData } from '../api/mockApi.js';
import { compressionMethods } from '../data/mockData.js';
import { getHardwareProfile } from '../data/hardwareProfiles.js';
import { getPoolCounts, getTotalBaseCount } from '../data/calibrationPool.js';

export default function StudioPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { getProject, updateProject, addRun } = useProjects();
  const { addToast } = useToast();
  const project = getProject(id);

  // Config state
  const [pruneRatio, setPruneRatio] = useState(project?.pruneRatio || 20);
  const [quantization, setQuantization] = useState(project?.quantization || 'INT8');
  const [quboEnabled, setQuboEnabled] = useState(true);
  const [fineTune, setFineTune] = useState(true);
  const [fineTuneSteps, setFineTuneSteps] = useState(100);
  const [learningRate, setLearningRate] = useState('1e-5');
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Heatmap state — loading derived from version mismatch
  const [heatmapData, setHeatmapData] = useState(null);
  const [selectedLayer, setSelectedLayer] = useState(null);
  const [lockedLayers, setLockedLayers] = useState(new Set(project?.lockedLayers || []));

  // Calibration state
  const [calibSamples, setCalibSamples] = useState(project?.calibrationSamples || '');
  const [calibRatio, setCalibRatio] = useState(project?.calibrationRatio || 35);

  // Mobile tab
  const [activeTab, setActiveTab] = useState('config');

  // Heatmap fetch tracking
  const [heatmapVersion, setHeatmapVersion] = useState(0);
  const [loadedVersion, setLoadedVersion] = useState(-1);
  const heatmapLoading = loadedVersion < heatmapVersion;

  useEffect(() => {
    const currentModelId = project?.modelId;
    if (!currentModelId) return;
    let ignore = false;
    const thisVersion = heatmapVersion;
    getHeatmapData(currentModelId).then(data => {
      if (!ignore) {
        setHeatmapData(data);
        setLoadedVersion(thisVersion);
      }
    });
    return () => { ignore = true; };
  }, [project?.modelId, heatmapVersion]);

  // Save config to project on change
  useEffect(() => {
    if (!project) return;
    updateProject(id, {
      pruneRatio,
      quantization,
      lockedLayers: Array.from(lockedLayers),
      calibrationSamples: calibSamples,
      calibrationRatio: calibRatio,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pruneRatio, quantization, lockedLayers, calibSamples, calibRatio]);

  const hw = getHardwareProfile(project?.hardware);

  const taskSampleCount = useMemo(() => {
    return calibSamples.split('\n').filter(l => l.trim()).length;
  }, [calibSamples]);

  const baseCount = getTotalBaseCount();
  const totalSamples = taskSampleCount + baseCount;
  const effectiveWeight = totalSamples > 0 ? Math.round((taskSampleCount / totalSamples) * 100) : 0;
  const poolCounts = getPoolCounts();

  const riskLevel = pruneRatio < 15 ? 'low' : pruneRatio < 25 ? 'moderate' : 'high';
  const riskText = {
    low: '0–15%: low risk',
    moderate: '15–25%: moderate risk — check heatmap for sensitive layers',
    high: '25%+: high risk — check heatmap for sensitive layers',
  };

  const lockedCount = lockedLayers.size;
  const totalLayers = heatmapData?.length || 0;

  const toggleLock = (idx) => {
    setLockedLayers(prev => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  };

  const handleRunCompression = () => {
    const method = compressionMethods.find(m => m.id === 'qubo-int8');
    addRun(id, {
      method: method?.name || 'Quantum-QUBO Pruning + INT8',
      pruneRatio,
      quantization,
      hardware: project.hardware,
      status: 'running',
    });
    addToast('Run started.', 'success');
    // Get the latest project to find the new run
    const updated = getProject(id);
    const latestRun = updated?.runs[updated.runs.length - 1];
    if (latestRun) {
      navigate(`/project/${id}/run/${latestRun.id}`);
    }
  };

  if (!project) {
    return <div className="page"><p className="text-muted">Project not found.</p></div>;
  }

  const renderHeatmapBars = () => {
    if (heatmapLoading || !heatmapData) {
      return (
        <div className="flex gap-4 items-end" style={{ height: 200 }}>
          {Array.from({ length: 24 }).map((_, i) => (
            <div
              key={i}
              className="skeleton skeleton-bar"
              style={{
                width: 16,
                height: `${30 + Math.random() * 70}%`,
                animationDelay: `${i * 50}ms`,
              }}
            />
          ))}
        </div>
      );
    }

    const maxSens = Math.max(...heatmapData.map(d => d.sensitivity));
    return (
      <div style={{ position: 'relative' }}>
        <div className="flex items-center gap-12" style={{ marginBottom: 8 }}>
          <span className="text-xs"><span style={{ color: 'var(--green)' }}>■</span> Low</span>
          <span className="text-xs"><span style={{ color: 'var(--yellow)' }}>■</span> Moderate</span>
          <span className="text-xs"><span style={{ color: 'var(--red)' }}>■</span> High</span>
        </div>
        <div className="flex gap-4 items-end" style={{ height: 200 }}>
          {heatmapData.map((layer, i) => {
            const pct = (layer.sensitivity / Math.max(maxSens, 1)) * 100;
            const isLocked = lockedLayers.has(i);
            const color = layer.sensitivity < 0.3 ? 'var(--green)'
              : layer.sensitivity < 0.6 ? 'var(--yellow)' : 'var(--red)';
            return (
              <div
                key={i}
                className="heatmap-bar"
                onClick={() => setSelectedLayer(selectedLayer === i ? null : i)}
                style={{
                  width: 16,
                  height: `${pct}%`,
                  minHeight: 4,
                  background: isLocked
                    ? 'repeating-linear-gradient(45deg, var(--surface-3), var(--surface-3) 2px, var(--text-dim) 2px, var(--text-dim) 4px)'
                    : color,
                  opacity: isLocked ? 0.5 : (selectedLayer === i ? 1 : 0.85),
                  borderRadius: '2px 2px 0 0',
                  transition: 'height 300ms ease-out',
                  animationDelay: `${i * 15}ms`,
                  position: 'relative',
                  cursor: 'pointer',
                  border: selectedLayer === i ? '1px solid var(--text)' : 'none',
                  flex: '1 1 0',
                }}
                title={`Layer ${i}: ${layer.name} (${layer.sensitivity.toFixed(3)})`}
              >
                {isLocked && (
                  <span style={{ position: 'absolute', top: -14, left: '50%', transform: 'translateX(-50%)', fontSize: '0.6rem' }}>
                    🔒
                  </span>
                )}
              </div>
            );
          })}
        </div>
        <div className="flex gap-4" style={{ marginTop: 2 }}>
          {heatmapData.map((_, i) => (
            <div key={i} className="text-xs text-dim" style={{ flex: '1 1 0', textAlign: 'center', fontSize: '0.55rem' }}>
              {i}
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderLayerDetail = () => {
    if (selectedLayer === null || !heatmapData || !heatmapData[selectedLayer]) return null;
    const layer = heatmapData[selectedLayer];
    const isLocked = lockedLayers.has(selectedLayer);
    const maxPrune = layer.sensitivity < 0.3 ? '35%' : layer.sensitivity < 0.6 ? '20%' : '8%';

    return (
      <div className="card" style={{ marginTop: 12 }}>
        <div className="flex justify-between items-center" style={{ marginBottom: 8 }}>
          <h4 className="font-mono text-sm">{layer.name}</h4>
          <span className="badge badge-gray">{layer.type}</span>
        </div>
        <div className="text-sm text-muted" style={{ marginBottom: 4 }}>
          {layer.params.toLocaleString()} params
        </div>
        <div className="text-sm" style={{ marginBottom: 8 }}>
          Recommended max pruning: <strong className="font-mono">{maxPrune}</strong>
        </div>
        {/* Mini weight distribution */}
        <div className="flex items-end gap-1" style={{ height: 60, marginBottom: 8 }}>
          {layer.weightDistribution.map((v, i) => (
            <div key={i} style={{
              flex: '1 1 0',
              height: `${v * 100}%`,
              background: 'var(--accent)',
              opacity: 0.5,
              borderRadius: '1px 1px 0 0',
            }} />
          ))}
        </div>
        <div className="toggle-wrap" onClick={() => toggleLock(selectedLayer)}>
          <div className={`toggle-track ${isLocked ? 'active' : ''}`}>
            <div className="toggle-knob" />
          </div>
          <span className="text-sm">Lock this layer (skip pruning)</span>
        </div>
      </div>
    );
  };

  // Calibration ratio badge
  const ratioBadge = calibRatio === 35 ? { text: 'Recommended (research-validated)', cls: 'badge-green' }
    : calibRatio > 45 ? { text: 'High — may reduce generalization', cls: 'badge-yellow' }
    : calibRatio < 25 ? { text: 'Low — less task-specific optimization', cls: 'badge-yellow' }
    : { text: 'Custom', cls: 'badge-gray' };

  // Config panel content
  const configPanel = (
    <div style={{ width: '100%' }}>
      <div className="section">
        <div className="section-title">Compression Method</div>
        <select className="select" value={quboEnabled ? 'qubo-int8' : 'hybrid-int8'}
          onChange={e => setQuboEnabled(e.target.value === 'qubo-int8')}>
          {compressionMethods.map(m => (
            <option key={m.id} value={m.id}>{m.shortName}</option>
          ))}
        </select>
      </div>

      <div className="section">
        <div className="section-title">Pruning</div>
        <div className="slider-container">
          <div className="slider-header">
            <span className="text-sm">Pruning ratio</span>
            <span className="slider-value">{pruneRatio}%</span>
          </div>
          <input type="range" min={0} max={40} value={pruneRatio}
            onChange={e => setPruneRatio(Number(e.target.value))} />
          <span className={`risk-indicator risk-${riskLevel}`}>{riskText[riskLevel]}</span>
        </div>
      </div>

      <div className="section">
        <div className="section-title">Quantization</div>
        <div className="toggle-group">
          {['FP32', 'FP8', 'INT8', 'INT4'].map(q => (
            <button key={q} className={`toggle-group-btn ${quantization === q ? 'active' : ''}`}
              onClick={() => setQuantization(q)}>
              {q}
            </button>
          ))}
        </div>
        {quantization === 'INT4' && (
          <div className="text-xs text-muted" style={{ marginTop: 4 }}>
            Higher compression, more accuracy risk on complex models
          </div>
        )}
      </div>

      <div className="section">
        <div className="section-title">Optimizer</div>
        <div className="toggle-wrap" onClick={() => setQuboEnabled(!quboEnabled)}>
          <div className={`toggle-track ${quboEnabled ? 'active' : ''}`}>
            <div className="toggle-knob" />
          </div>
          <span className="text-sm">Quantum QUBO optimizer</span>
        </div>
        {quboEnabled && (
          <div
            className="collapsible-header"
            onClick={() => setShowAdvanced(!showAdvanced)}
            style={{ marginTop: 8 }}
          >
            <span className="text-xs text-muted">Advanced</span>
            <span className={`collapsible-arrow ${showAdvanced ? 'open' : ''}`}>▼</span>
          </div>
        )}
        {quboEnabled && showAdvanced && (
          <div className="flex flex-col gap-8" style={{ marginTop: 8 }}>
            <div className="flex justify-between items-center">
              <span className="text-xs text-muted">Temperature</span>
              <input className="number-input-sm" type="number" step="0.1" defaultValue={10.0} />
            </div>
            <div className="flex justify-between items-center">
              <span className="text-xs text-muted">Cooling rate</span>
              <input className="number-input-sm" type="number" step="0.01" defaultValue={0.95} />
            </div>
            <div className="flex justify-between items-center">
              <span className="text-xs text-muted">Max iterations</span>
              <input className="number-input-sm" type="number" step="10" defaultValue={400} />
            </div>
          </div>
        )}
      </div>

      <div className="section">
        <div className="section-title">Fine-tuning</div>
        <div className="toggle-wrap" onClick={() => setFineTune(!fineTune)}>
          <div className={`toggle-track ${fineTune ? 'active' : ''}`}>
            <div className="toggle-knob" />
          </div>
          <span className="text-sm">Fine-tune after pruning</span>
        </div>
        {fineTune && (
          <div className="flex gap-16" style={{ marginTop: 8 }}>
            <div>
              <span className="text-xs text-muted">Steps</span>
              <input className="number-input-sm" type="number" value={fineTuneSteps}
                onChange={e => setFineTuneSteps(Number(e.target.value))} />
            </div>
            <div>
              <span className="text-xs text-muted">Learning rate</span>
              <input className="number-input-sm" type="text" value={learningRate}
                onChange={e => setLearningRate(e.target.value)} />
            </div>
          </div>
        )}
      </div>

      <div className="text-xs text-dim" style={{ marginTop: 12 }}>
        ~4m 20s on CPU · ~1m 10s with CUDA
      </div>
    </div>
  );

  // Heatmap panel content
  const heatmapPanel = (
    <div style={{ width: '100%' }}>
      <h3 style={{ marginBottom: 4 }}>Layer Compression Safety</h3>
      <div className="text-xs text-muted" style={{ marginBottom: 16 }}>
        Computed from: gradient variance × activation distribution entropy
      </div>
      {renderHeatmapBars()}
      {renderLayerDetail()}
      <div className="text-xs text-muted" style={{ marginTop: 12 }}>
        {lockedCount} of {totalLayers} layers locked from pruning.
      </div>
      <div className="text-xs text-dim" style={{ marginTop: 8 }}>
        Heatmap runs a lightweight calibration pass on your data. Takes ~15 seconds. Recalculate if you change calibration data.
      </div>
      <button className="btn btn-sm" style={{ marginTop: 8 }} onClick={() => {
        setHeatmapVersion(v => v + 1);
        addToast('Heatmap recalculating...', 'info');
      }}>
        Recalculate
      </button>
    </div>
  );

  // Calibration panel content
  const calibPanel = (
    <div style={{ width: '100%' }}>
      <h3 style={{ marginBottom: 4 }}>Calibration Data</h3>

      <div className="section" style={{ marginTop: 12 }}>
        <label className="form-label">Your inference examples</label>
        <div className="text-xs text-muted" style={{ marginBottom: 6 }}>
          Paste real inputs your model will see — 20 to 50 samples, one per line.
          This is the most impactful setting in this entire panel.
        </div>
        <textarea
          className="textarea"
          rows={8}
          value={calibSamples}
          onChange={e => setCalibSamples(e.target.value)}
          placeholder={"How do I configure a VPN on Ubuntu?\nExplain backpropagation to a 10-year-old\nWhat are the symptoms of appendicitis"}
        />
        <div className="text-xs text-muted" style={{ marginTop: 4 }}>
          {taskSampleCount} samples added
        </div>
      </div>

      <div className="section">
        <div className="slider-container">
          <div className="slider-header">
            <span className="form-label">Task data weight</span>
            <div className="flex items-center gap-8">
              <span className="slider-value">{calibRatio}%</span>
              <span className={`badge ${ratioBadge.cls}`}>{ratioBadge.text}</span>
            </div>
          </div>
          <input type="range" min={20} max={50} value={calibRatio}
            onChange={e => setCalibRatio(Number(e.target.value))} />
        </div>
      </div>

      <div className="section">
        <div className="collapsible-header" onClick={() => {}}>
          <span className="text-sm text-muted">Base diversity pool</span>
          <span className="collapsible-arrow">▼</span>
        </div>
        <div className="flex flex-wrap gap-12" style={{ marginTop: 8 }}>
          {Object.entries(poolCounts).map(([cat, count]) => (
            <span key={cat} className="text-xs text-muted" style={{ textTransform: 'capitalize' }}>
              {cat}: {count}
            </span>
          ))}
        </div>
      </div>

      <div className="card" style={{ marginTop: 12, padding: 12 }}>
        <span className="font-mono text-sm">
          Using {taskSampleCount} task + {baseCount} base = {totalSamples} total calibration samples
        </span>
        <br />
        <span className="font-mono text-xs text-muted">
          Effective task weight: {effectiveWeight}%
        </span>
      </div>
    </div>
  );

  return (
    <div className="page-full">
      {/* Top bar */}
      <div className="flex justify-between items-center" style={{ marginBottom: 20, padding: '0 24px' }}>
        <div className="flex items-center gap-16">
          <h2>{project.name}</h2>
          {hw && <span className="badge badge-gray">{hw.icon} {hw.name}</span>}
        </div>
        <button className="btn btn-primary btn-lg" onClick={handleRunCompression} id="run-compression-btn">
          Run Compression
        </button>
      </div>

      {/* Mobile tabs */}
      <div className="flex gap-4" style={{ padding: '0 24px', marginBottom: 16, display: 'none' }}>
        {['config', 'heatmap', 'calibration'].map(tab => (
          <button key={tab} className={`btn btn-sm ${activeTab === tab ? 'btn-primary' : ''}`}
            onClick={() => setActiveTab(tab)} style={{ textTransform: 'capitalize' }}>
            {tab}
          </button>
        ))}
      </div>

      {/* Three-panel layout */}
      <div className="flex gap-20" style={{ padding: '0 24px' }}>
        <div style={{ width: 280, flexShrink: 0 }}>{configPanel}</div>
        <div style={{ flex: 1, minWidth: 0 }}>{heatmapPanel}</div>
        <div style={{ width: 300, flexShrink: 0 }}>{calibPanel}</div>
      </div>
    </div>
  );
}
