import { NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext.jsx';

export default function Navbar() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <nav className="navbar">
      <div className="nav-brand">
        <h1>QUNART</h1>
        <span>by SANEICS</span>
      </div>

      <div className="nav-links">
        <NavLink to="/dashboard" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          Dashboard
        </NavLink>
        <NavLink to="/history" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          History
        </NavLink>
      </div>

      <div className="nav-user">
        <span className="text-sm text-muted">{user?.email}</span>
        <div className="nav-avatar" onClick={handleLogout} title="Logout">
          {user?.initial || '?'}
        </div>
      </div>
    </nav>
  );
}
