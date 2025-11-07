/**
 * Application Context
 * Global state management using React Context
 * 
 * Author: A-P-U-R-B-O
 * Created: 2025-11-07 15:32:27 UTC
 */

import { createContext, useContext, useState, useEffect } from 'react';

const AppContext = createContext(null);

export const AppProvider = ({ children }) => {
  const [user] = useState({
    name: 'A-P-U-R-B-O',
    role: 'Admin',
    email: 'admin@construction-safety.ai',
  });

  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem('theme');
    return saved || 'light';
  });

  const [notifications, setNotifications] = useState([]);
  const [stats, setStats] = useState({
    totalDetections: 0,
    totalAlerts: 0,
    activeMonitoring: false,
  });

  // Apply theme
  useEffect(() => {
    const root = window.document.documentElement;
    root.classList.remove('light', 'dark');
    root.classList.add(theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === 'light' ? 'dark' : 'light'));
  };

  const addNotification = (notification) => {
    setNotifications((prev) => [
      ...prev,
      { ...notification, id: Date.now(), timestamp: new Date() },
    ]);
  };

  const clearNotifications = () => {
    setNotifications([]);
  };

  const updateStats = (newStats) => {
    setStats((prev) => ({ ...prev, ...newStats }));
  };

  const value = {
    user,
    theme,
    toggleTheme,
    notifications,
    addNotification,
    clearNotifications,
    stats,
    updateStats,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
};

export const useApp = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within AppProvider');
  }
  return context;
};
