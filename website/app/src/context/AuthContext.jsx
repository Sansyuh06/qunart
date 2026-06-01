import { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext(null);

const STORAGE_KEY = 'qunart_auth';

export function AuthProvider({ children }) {
  const [user, setUser] = useState(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      return stored ? JSON.parse(stored) : null;
    } catch {
      return null;
    }
  });

  useEffect(() => {
    if (user) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(user));
    } else {
      localStorage.removeItem(STORAGE_KEY);
    }
  }, [user]);

  const login = (email, password) => {
    if (!email || !password) {
      return { success: false, error: 'Email and password are required.' };
    }
    if (password.length < 4) {
      return { success: false, error: 'Invalid credentials.' };
    }
    const userData = {
      id: 'user-1',
      email,
      name: email.split('@')[0],
      initial: email[0].toUpperCase(),
    };
    setUser(userData);
    return { success: true };
  };

  const signup = (email, password, name) => {
    if (!email || !password || !name) {
      return { success: false, error: 'All fields are required.' };
    }
    if (password.length < 6) {
      return { success: false, error: 'Password must be at least 6 characters.' };
    }
    const userData = {
      id: 'user-' + Date.now(),
      email,
      name,
      initial: name[0].toUpperCase(),
    };
    setUser(userData);
    return { success: true };
  };

  const logout = () => {
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, login, signup, logout, isAuthenticated: !!user }}>
      {children}
    </AuthContext.Provider>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
