import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { getSampleProjects } from '../api/mockApi.js';

const ProjectContext = createContext(null);
const STORAGE_KEY = 'qunart_projects';

export function ProjectProvider({ children }) {
  const [projects, setProjects] = useState(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed) && parsed.length > 0) return parsed;
      }
    } catch { /* ignore */ }
    // Seed with sample projects on first load
    const samples = getSampleProjects();
    localStorage.setItem(STORAGE_KEY, JSON.stringify(samples));
    return samples;
  });

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(projects));
  }, [projects]);

  const createProject = useCallback((config) => {
    const id = 'proj-' + Date.now();
    const newProject = {
      id,
      name: config.name || config.modelId?.split('/').pop() || 'New Project',
      modelId: config.modelId,
      hardware: config.hardware,
      method: config.method,
      pruneRatio: config.pruneRatio || 20,
      quantization: config.quantization || 'INT8',
      budgetSize: config.budgetSize || 512,
      budgetLatency: config.budgetLatency || 100,
      budgetAccuracy: config.budgetAccuracy || 5,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      status: 'new',
      runs: [],
      lockedLayers: [],
      calibrationSamples: '',
      calibrationRatio: 35,
    };
    setProjects(prev => [...prev, newProject]);
    return id;
  }, []);

  const getProject = useCallback((id) => {
    return projects.find(p => p.id === id) || null;
  }, [projects]);

  const updateProject = useCallback((id, data) => {
    setProjects(prev => prev.map(p =>
      p.id === id ? { ...p, ...data, updatedAt: new Date().toISOString() } : p
    ));
  }, []);

  const deleteProject = useCallback((id) => {
    setProjects(prev => prev.filter(p => p.id !== id));
  }, []);

  const addRun = useCallback((projectId, runData) => {
    setProjects(prev => prev.map(p => {
      if (p.id !== projectId) return p;
      const runNumber = p.runs.length + 1;
      const run = {
        ...runData,
        id: `run-${projectId}-${runNumber}`,
        number: runNumber,
        startedAt: new Date().toISOString(),
      };
      return {
        ...p,
        runs: [...p.runs, run],
        status: 'running',
        updatedAt: new Date().toISOString(),
      };
    }));
  }, []);

  const updateRun = useCallback((projectId, runId, runData) => {
    setProjects(prev => prev.map(p => {
      if (p.id !== projectId) return p;
      return {
        ...p,
        runs: p.runs.map(r => r.id === runId ? { ...r, ...runData } : r),
        status: runData.status === 'completed' ? 'completed' : p.status,
        updatedAt: new Date().toISOString(),
      };
    }));
  }, []);

  const getRuns = useCallback((projectId) => {
    const project = projects.find(p => p.id === projectId);
    return project ? project.runs : [];
  }, [projects]);

  const getAllRuns = useCallback(() => {
    return projects.flatMap(p =>
      p.runs.map(r => ({ ...r, projectName: p.name, projectId: p.id, hardwareTarget: p.hardware }))
    );
  }, [projects]);

  return (
    <ProjectContext.Provider value={{
      projects,
      createProject,
      getProject,
      updateProject,
      deleteProject,
      addRun,
      updateRun,
      getRuns,
      getAllRuns,
    }}>
      {children}
    </ProjectContext.Provider>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export function useProjects() {
  const ctx = useContext(ProjectContext);
  if (!ctx) throw new Error('useProjects must be used within ProjectProvider');
  return ctx;
}
