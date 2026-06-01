import { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { useProjects } from '../context/ProjectContext.jsx';
import { getHardwareProfile } from '../data/hardwareProfiles.js';

export default function HistoryPage() {
  const navigate = useNavigate();
  const { getAllRuns } = useProjects();
  const allRuns = getAllRuns();

  const [sortCol, setSortCol] = useState('startedAt');
  const [sortDir, setSortDir] = useState('desc');
  const [filterHardware, setFilterHardware] = useState('');
  const [filterStatus, setFilterStatus] = useState('');
  const [selected, setSelected] = useState(new Set());
  const [showCompare, setShowCompare] = useState(false);

  const handleSort = (col) => {
    if (sortCol === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortCol(col); setSortDir('desc'); }
  };

  const filteredRuns = useMemo(() => {
    let runs = [...allRuns];
    if (filterHardware) runs = runs.filter(r => r.hardware === filterHardware);
    if (filterStatus) runs = runs.filter(r => r.status === filterStatus);
    runs.sort((a, b) => {
      let va = a[sortCol], vb = b[sortCol];
      if (typeof va === 'string') va = va.toLowerCase();
      if (typeof vb === 'string') vb = vb.toLowerCase();
      if (va < vb) return sortDir === 'asc' ? -1 : 1;
      if (va > vb) return sortDir === 'asc' ? 1 : -1;
      return 0;
    });
    return runs;
  }, [allRuns, filterHardware, filterStatus, sortCol, sortDir]);

  // Find best run (highest ratio that met accuracy target)
  const bestRunId = useMemo(() => {
    const candidates = filteredRuns.filter(r =>
      r.status === 'completed' && r.accuracyRetained >= (r.accuracyTarget || 0)
    );
    if (!candidates.length) return null;
    candidates.sort((a, b) => (b.compressionRatio || 0) - (a.compressionRatio || 0));
    return candidates[0]?.id;
  }, [filteredRuns]);

  const toggleSelect = (runId) => {
    setSelected(prev => {
      const next = new Set(prev);
      if (next.has(runId)) next.delete(runId);
      else if (next.size < 3) next.add(runId);
      return next;
    });
  };

  const selectedRuns = useMemo(() => {
    return filteredRuns.filter(r => selected.has(r.id));
  }, [filteredRuns, selected]);

  const formatDate = (iso) => {
    if (!iso) return '—';
    return new Date(iso).toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  };

  const formatSize = (mb) => {
    if (!mb) return '—';
    if (mb >= 1000) return `${(mb / 1000).toFixed(1)}GB`;
    return `${mb}MB`;
  };

  const uniqueHardware = [...new Set(allRuns.map(r => r.hardware).filter(Boolean))];
  const uniqueStatuses = [...new Set(allRuns.map(r => r.status).filter(Boolean))];

  if (allRuns.length === 0) {
    return (
      <div className="page">
        <div className="empty-state">
          <h2>No runs yet</h2>
          <p>Complete a compression run from a project to see it here.</p>
          <button className="btn btn-primary" onClick={() => navigate('/dashboard')}>Go to Dashboard</button>
        </div>
      </div>
    );
  }

  return (
    <div className="page">
      <div className="flex justify-between items-center" style={{ marginBottom: 20 }}>
        <h1>Experiment History</h1>
        <div className="flex gap-8">
          <select className="select" style={{ width: 'auto' }} value={filterHardware}
            onChange={e => setFilterHardware(e.target.value)}>
            <option value="">All hardware</option>
            {uniqueHardware.map(h => {
              const hw = getHardwareProfile(h);
              return <option key={h} value={h}>{hw?.name || h}</option>;
            })}
          </select>
          <select className="select" style={{ width: 'auto' }} value={filterStatus}
            onChange={e => setFilterStatus(e.target.value)}>
            <option value="">All statuses</option>
            {uniqueStatuses.map(s => (
              <option key={s} value={s} style={{ textTransform: 'capitalize' }}>{s}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="table-container">
        <table>
          <thead>
            <tr>
              <th style={{ width: 32 }}></th>
              <th onClick={() => handleSort('projectName')} className={sortCol === 'projectName' ? 'sorted' : ''}>Project</th>
              <th onClick={() => handleSort('number')} className={sortCol === 'number' ? 'sorted' : ''}>Run #</th>
              <th onClick={() => handleSort('startedAt')} className={sortCol === 'startedAt' ? 'sorted' : ''}>Date</th>
              <th onClick={() => handleSort('method')} className={sortCol === 'method' ? 'sorted' : ''}>Method</th>
              <th onClick={() => handleSort('pruneRatio')} className={sortCol === 'pruneRatio' ? 'sorted' : ''}>Prune %</th>
              <th>Hardware</th>
              <th onClick={() => handleSort('originalSize')}>Original</th>
              <th onClick={() => handleSort('compressedSize')}>Compressed</th>
              <th onClick={() => handleSort('compressionRatio')} className={sortCol === 'compressionRatio' ? 'sorted' : ''}>Ratio</th>
              <th onClick={() => handleSort('accuracyRetained')} className={sortCol === 'accuracyRetained' ? 'sorted' : ''}>Accuracy</th>
              <th>Status</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {filteredRuns.map(run => {
              const hw = getHardwareProfile(run.hardwareTarget || run.hardware);
              const isBest = run.id === bestRunId;
              return (
                <tr key={run.id} className={selected.has(run.id) ? 'selected' : ''}>
                  <td>
                    <div className={`checkbox ${selected.has(run.id) ? 'checked' : ''}`}
                      onClick={() => toggleSelect(run.id)} />
                  </td>
                  <td>{run.projectName}</td>
                  <td className="font-mono">{run.number}</td>
                  <td className="text-sm">{formatDate(run.startedAt)}</td>
                  <td className="text-sm">{run.method}</td>
                  <td className="font-mono">{run.pruneRatio}%</td>
                  <td className="text-sm">{hw?.icon} {hw?.name || '—'}</td>
                  <td className="font-mono">{formatSize(run.originalSize)}</td>
                  <td className="font-mono">{formatSize(run.compressedSize)}</td>
                  <td className="font-mono text-accent">{run.compressionRatio ? `${run.compressionRatio}×` : '—'}</td>
                  <td className="font-mono">{run.accuracyRetained ? `${run.accuracyRetained}%` : '—'}</td>
                  <td>
                    <span className={`badge badge-${run.status === 'completed' ? 'green' : run.status === 'running' ? 'yellow' : 'red'}`}>
                      {run.status}
                    </span>
                  </td>
                  <td>
                    {isBest && <span className="star-icon" title="Best run">★</span>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Compare Bar */}
      {selected.size >= 2 && (
        <div className="sticky-bottom-bar">
          <span className="text-sm">{selected.size} runs selected</span>
          <button className="btn btn-primary" onClick={() => setShowCompare(true)}>
            Compare {selected.size} runs
          </button>
        </div>
      )}

      {/* Compare Modal */}
      {showCompare && selectedRuns.length >= 2 && (
        <div className="modal-overlay" onClick={() => setShowCompare(false)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Run Comparison</h3>
              <button className="btn btn-ghost" onClick={() => setShowCompare(false)}>✕</button>
            </div>
            <div className="modal-body">
              <table style={{ width: '100%' }}>
                <thead>
                  <tr>
                    <th>Metric</th>
                    {selectedRuns.map(r => (
                      <th key={r.id}>{r.projectName} #{r.number}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[
                    { label: 'Compression Ratio', key: 'compressionRatio', suffix: '×', higher: true },
                    { label: 'Accuracy Retained', key: 'accuracyRetained', suffix: '%', higher: true },
                    { label: 'Compressed Size', key: 'compressedSize', suffix: 'MB', higher: false },
                    { label: 'Prune Ratio', key: 'pruneRatio', suffix: '%', higher: false },
                  ].map(metric => {
                    const values = selectedRuns.map(r => r[metric.key] || 0);
                    const best = metric.higher ? Math.max(...values) : Math.min(...values);
                    const maxVal = Math.max(...values);
                    return (
                      <tr key={metric.key}>
                        <td className="text-sm">{metric.label}</td>
                        {selectedRuns.map((r, i) => {
                          const val = values[i];
                          const isBest = val === best;
                          return (
                            <td key={r.id}>
                              <div className="font-mono text-sm" style={{ marginBottom: 4, color: isBest ? 'var(--green)' : 'var(--text)' }}>
                                {val}{metric.suffix}
                              </div>
                              <div className="compare-bar">
                                <div
                                  className={`compare-bar-fill ${isBest ? 'best' : 'other'}`}
                                  style={{ width: `${maxVal > 0 ? (val / maxVal * 100) : 0}%` }}
                                />
                              </div>
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
