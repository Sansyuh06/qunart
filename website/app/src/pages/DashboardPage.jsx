import { useNavigate } from 'react-router-dom';
import { useProjects } from '../context/ProjectContext.jsx';
import { modelDatabase } from '../data/mockData.js';
import { getHardwareProfile } from '../data/hardwareProfiles.js';

export default function DashboardPage() {
  const { projects } = useProjects();
  const navigate = useNavigate();

  const formatSize = (mb) => {
    if (mb >= 1000) return `${(mb / 1000).toFixed(1)}GB`;
    return `${mb}MB`;
  };

  const formatDate = (iso) => {
    if (!iso) return '';
    const d = new Date(iso);
    const now = new Date();
    const diff = now - d;
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  const getStatusBadge = (status) => {
    const map = {
      completed: { className: 'badge-green', label: 'Completed' },
      running: { className: 'badge-yellow', label: 'Running' },
      failed: { className: 'badge-red', label: 'Failed' },
      new: { className: 'badge-gray', label: 'New' },
    };
    const s = map[status] || map.new;
    return <span className={`badge ${s.className}`}>{s.label}</span>;
  };

  if (projects.length === 0) {
    return (
      <div className="page">
        <div className="empty-state">
          <h2>No projects yet</h2>
          <p>
            Qunart compresses AI models for edge hardware — Raspberry Pi, Jetson, Android.
            Quantum-QUBO pruning + INT8 quantization. Start a project to compress your first model.
          </p>
          <button className="btn btn-primary btn-lg" onClick={() => navigate('/project/new')}>
            Start your first project
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="page">
      <div style={{ marginBottom: 24 }}>
        <h1>Projects</h1>
      </div>

      <div className="grid-2">
        {projects.map(project => {
          const model = modelDatabase[project.modelId];
          const hw = getHardwareProfile(project.hardware);
          const lastRun = project.runs[project.runs.length - 1];
          const originalSize = model ? Math.floor(model.sizeFP32 * 1000) : 0;
          const compressedSize = lastRun?.compressedSize;
          const ratio = lastRun?.compressionRatio;

          return (
            <div
              key={project.id}
              className="card card-interactive"
              onClick={() => navigate(`/project/${project.id}`)}
              id={`project-card-${project.id}`}
            >
              <div className="flex justify-between items-start" style={{ marginBottom: 12 }}>
                <div>
                  <h3 style={{ marginBottom: 4 }}>{project.name}</h3>
                  <span className="text-sm text-muted font-mono">{project.modelId}</span>
                </div>
                {getStatusBadge(project.status)}
              </div>

              {hw && (
                <div className="flex items-center gap-8" style={{ marginBottom: 8 }}>
                  <span style={{ fontSize: '1rem' }}>{hw.icon}</span>
                  <span className="text-sm text-muted">{hw.name}</span>
                </div>
              )}

              {lastRun && compressedSize ? (
                <div className="flex justify-between items-end" style={{ marginTop: 12 }}>
                  <div>
                    <span className="text-sm text-muted">
                      {formatSize(originalSize)} → {formatSize(compressedSize)}
                    </span>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div className="font-mono text-accent" style={{ fontSize: '1.5rem', fontWeight: 600, lineHeight: 1 }}>
                      {ratio}×
                    </div>
                  </div>
                </div>
              ) : (
                <div style={{ marginTop: 12 }}>
                  <span className="text-sm text-muted">No runs yet</span>
                </div>
              )}

              <div className="text-xs text-dim" style={{ marginTop: 10 }}>
                {formatDate(project.updatedAt)}
              </div>
            </div>
          );
        })}

        <div className="card card-dashed" onClick={() => navigate('/project/new')} id="new-project-card">
          <span className="plus-icon">+</span>
          <span className="text-sm text-muted">New Project</span>
        </div>
      </div>
    </div>
  );
}
