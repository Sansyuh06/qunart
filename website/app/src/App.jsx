import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext.jsx';
import { ProjectProvider } from './context/ProjectContext.jsx';
import { ToastProvider } from './context/ToastContext.jsx';
import GeometryCanvas from './components/layout/GeometryCanvas.jsx';
import Navbar from './components/layout/Navbar.jsx';
import LoginPage from './pages/LoginPage.jsx';
import SignupPage from './pages/SignupPage.jsx';
import DashboardPage from './pages/DashboardPage.jsx';
import NewProjectPage from './pages/NewProjectPage.jsx';
import StudioPage from './pages/StudioPage.jsx';
import RunPage from './pages/RunPage.jsx';
import HistoryPage from './pages/HistoryPage.jsx';

function ProtectedRoute({ children }) {
  const { isAuthenticated } = useAuth();
  if (!isAuthenticated) return <Navigate to="/login" replace />;
  return children;
}

function AppLayout({ children }) {
  return (
    <>
      <GeometryCanvas />
      <Navbar />
      <main style={{ position: 'relative', zIndex: 1, flex: 1 }}>
        {children}
      </main>
    </>
  );
}

function AppRoutes() {
  const { isAuthenticated } = useAuth();

  return (
    <Routes>
      <Route path="/login" element={isAuthenticated ? <Navigate to="/dashboard" replace /> : <LoginPage />} />
      <Route path="/signup" element={isAuthenticated ? <Navigate to="/dashboard" replace /> : <SignupPage />} />
      <Route path="/dashboard" element={
        <ProtectedRoute><AppLayout><DashboardPage /></AppLayout></ProtectedRoute>
      } />
      <Route path="/project/new" element={
        <ProtectedRoute><AppLayout><NewProjectPage /></AppLayout></ProtectedRoute>
      } />
      <Route path="/project/:id" element={
        <ProtectedRoute><AppLayout><StudioPage /></AppLayout></ProtectedRoute>
      } />
      <Route path="/project/:id/run/:runId" element={
        <ProtectedRoute><AppLayout><RunPage /></AppLayout></ProtectedRoute>
      } />
      <Route path="/history" element={
        <ProtectedRoute><AppLayout><HistoryPage /></AppLayout></ProtectedRoute>
      } />
      <Route path="*" element={<Navigate to={isAuthenticated ? "/dashboard" : "/login"} replace />} />
    </Routes>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <ProjectProvider>
          <ToastProvider>
            <AppRoutes />
          </ToastProvider>
        </ProjectProvider>
      </AuthProvider>
    </BrowserRouter>
  );
}
