/**
 * Main Application Component
 * Construction Safety AI Detection System - Frontend
 * 
 * Author: A-P-U-R-B-O
 * Created: 2025-11-07 15:32:27 UTC
 * Version: 1.0.0
 * 
 * This is the root component that handles routing, layout, and global state.
 */

import { useState, useEffect, lazy, Suspense } from 'react';
import { Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import toast from 'react-hot-toast';

// Contexts
import { AppProvider, useApp } from '@/contexts/AppContext';

// Layout Components
import Header from '@/components/Layout/Header';
import Sidebar from '@/components/Layout/Sidebar';
import Footer from '@/components/Layout/Footer';
import LoadingSpinner from '@/components/UI/LoadingSpinner';
import ErrorBoundary from '@/components/ErrorBoundary';

// Lazy load pages for code splitting
const Dashboard = lazy(() => import('@/pages/Dashboard'));
const LiveMonitoring = lazy(() => import('@/pages/LiveMonitoring'));
const ImageDetection = lazy(() => import('@/pages/ImageDetection'));
const VideoProcessing = lazy(() => import('@/pages/VideoProcessing'));
const Analytics = lazy(() => import('@/pages/Analytics'));
const Alerts = lazy(() => import('@/pages/Alerts'));
const Settings = lazy(() => import('@/pages/Settings'));
const About = lazy(() => import('@/pages/About'));
const NotFound = lazy(() => import('@/pages/NotFound'));

// ============================================================================
// Constants
// ============================================================================

const APP_METADATA = {
  name: 'Construction Safety AI',
  version: '1.0.0',
  author: 'A-P-U-R-B-O',
  buildTime: '2025-11-07 15:32:27 UTC',
};

// Page transition variants
const pageVariants = {
  initial: {
    opacity: 0,
    x: -20,
  },
  animate: {
    opacity: 1,
    x: 0,
    transition: {
      duration: 0.3,
      ease: 'easeOut',
    },
  },
  exit: {
    opacity: 0,
    x: 20,
    transition: {
      duration: 0.2,
      ease: 'easeIn',
    },
  },
};

// ============================================================================
// Loading Fallback Component
// ============================================================================

const PageLoadingFallback = () => (
  <div className="flex items-center justify-center min-h-screen bg-gray-50 dark:bg-gray-900">
    <div className="text-center">
      <LoadingSpinner size="large" />
      <p className="mt-4 text-gray-600 dark:text-gray-400 font-medium">
        Loading page...
      </p>
    </div>
  </div>
);

// ============================================================================
// Main Layout Component
// ============================================================================

const AppLayout = ({ children }) => {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [isMobile, setIsMobile] = useState(false);
  const location = useLocation();

  // Handle responsive sidebar
  useEffect(() => {
    const checkMobile = () => {
      const mobile = window.innerWidth < 1024;
      setIsMobile(mobile);
      if (mobile) {
        setSidebarOpen(false);
      }
    };

    checkMobile();
    window.addEventListener('resize', checkMobile);

    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  // Close sidebar on mobile when route changes
  useEffect(() => {
    if (isMobile) {
      setSidebarOpen(false);
    }
  }, [location.pathname, isMobile]);

  const toggleSidebar = () => setSidebarOpen(!sidebarOpen);

  return (
    <div className="flex h-screen overflow-hidden bg-gray-50 dark:bg-gray-900">
      {/* Sidebar */}
      <Sidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        isMobile={isMobile}
      />

      {/* Mobile sidebar overlay */}
      {isMobile && sidebarOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-40 bg-gray-900 bg-opacity-50 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main Content */}
      <div className="flex flex-col flex-1 overflow-hidden">
        {/* Header */}
        <Header
          onMenuClick={toggleSidebar}
          sidebarOpen={sidebarOpen}
        />

        {/* Page Content */}
        <main className="flex-1 overflow-x-hidden overflow-y-auto bg-gray-50 dark:bg-gray-900">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <ErrorBoundary>
              <AnimatePresence mode="wait">
                <motion.div
                  key={location.pathname}
                  variants={pageVariants}
                  initial="initial"
                  animate="animate"
                  exit="exit"
                >
                  {children}
                </motion.div>
              </AnimatePresence>
            </ErrorBoundary>
          </div>
        </main>

        {/* Footer */}
        <Footer />
      </div>
    </div>
  );
};

// ============================================================================
// App Routes Component
// ============================================================================

const AppRoutes = () => {
  return (
    <Suspense fallback={<PageLoadingFallback />}>
      <Routes>
        {/* Dashboard - Home */}
        <Route path="/" element={<Dashboard />} />
        <Route path="/dashboard" element={<Navigate to="/" replace />} />

        {/* Live Monitoring */}
        <Route path="/live" element={<LiveMonitoring />} />
        <Route path="/monitoring" element={<Navigate to="/live" replace />} />

        {/* Image Detection */}
        <Route path="/image" element={<ImageDetection />} />
        <Route path="/detect" element={<Navigate to="/image" replace />} />

        {/* Video Processing */}
        <Route path="/video" element={<VideoProcessing />} />
        <Route path="/process" element={<Navigate to="/video" replace />} />

        {/* Analytics */}
        <Route path="/analytics" element={<Analytics />} />
        <Route path="/stats" element={<Navigate to="/analytics" replace />} />

        {/* Alerts */}
        <Route path="/alerts" element={<Alerts />} />
        <Route path="/notifications" element={<Navigate to="/alerts" replace />} />

        {/* Settings */}
        <Route path="/settings" element={<Settings />} />
        <Route path="/config" element={<Navigate to="/settings" replace />} />

        {/* About */}
        <Route path="/about" element={<About />} />
        <Route path="/info" element={<Navigate to="/about" replace />} />

        {/* 404 Not Found */}
        <Route path="*" element={<NotFound />} />
      </Routes>
    </Suspense>
  );
};

// ============================================================================
// Main App Component
// ============================================================================

const App = () => {
  const [isInitialized, setIsInitialized] = useState(false);
  const [initError, setInitError] = useState(null);

  // Initialize application
  useEffect(() => {
    const initializeApp = async () => {
      try {
        console.log('%c🚀 Initializing App Component...', 'color: #3b82f6; font-weight: bold');

        // Check if backend is reachable (optional health check)
        try {
          const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
          const response = await fetch(`${apiUrl}/health`, {
            method: 'GET',
            headers: {
              'Content-Type': 'application/json',
            },
          });

          if (response.ok) {
            const data = await response.json();
            console.log('%c✅ Backend API is reachable', 'color: #22c55e; font-weight: bold');
            console.log('Status:', data.status);
          } else {
            console.warn('%c⚠️  Backend API returned non-OK status', 'color: #f59e0b; font-weight: bold');
          }
        } catch (error) {
          console.warn('%c⚠️  Backend API is not reachable', 'color: #f59e0b; font-weight: bold');
          console.warn('The app will still work, but live features may be unavailable');
        }

        // Set up global event listeners
        setupGlobalEventListeners();

        // Mark as initialized
        setIsInitialized(true);

        console.log('%c✅ App Component initialized successfully', 'color: #22c55e; font-weight: bold');
        console.log('%c━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', 'color: #ef4444');
        console.log(`%c👤 Current User: ${APP_METADATA.author}`, 'color: #8b5cf6; font-weight: bold');
        console.log(`%c🕐 Session Started: ${new Date().toISOString()}`, 'color: #8b5cf6; font-weight: bold');
        console.log('%c━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', 'color: #ef4444');

      } catch (error) {
        console.error('%c❌ App initialization failed:', 'color: #ef4444; font-weight: bold', error);
        setInitError(error);
      }
    };

    initializeApp();

    // Cleanup
    return () => {
      cleanupGlobalEventListeners();
    };
  }, []);

  // Set up global event listeners
  const setupGlobalEventListeners = () => {
    // Online/Offline detection
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    // Visibility change (tab switching)
    document.addEventListener('visibilitychange', handleVisibilityChange);

    // Before unload (prevent accidental close if processing)
    window.addEventListener('beforeunload', handleBeforeUnload);
  };

  // Cleanup global event listeners
  const cleanupGlobalEventListeners = () => {
    window.removeEventListener('online', handleOnline);
    window.removeEventListener('offline', handleOffline);
    document.removeEventListener('visibilitychange', handleVisibilityChange);
    window.removeEventListener('beforeunload', handleBeforeUnload);
  };

  // Event handlers
  const handleOnline = () => {
    console.log('%c🌐 Connection restored', 'color: #22c55e; font-weight: bold');
    toast.success('Connection restored! Back online.', {
      icon: '🌐',
      duration: 3000,
    });
  };

  const handleOffline = () => {
    console.log('%c📡 Connection lost', 'color: #f59e0b; font-weight: bold');
    toast.error('Connection lost! Check your internet.', {
      icon: '📡',
      duration: 5000,
    });
  };

  const handleVisibilityChange = () => {
    if (document.hidden) {
      console.log('%c👁️ Tab hidden', 'color: #6b7280');
    } else {
      console.log('%c👁️ Tab visible', 'color: #6b7280');
    }
  };

  const handleBeforeUnload = (e) => {
    // Only show warning if there's active processing
    // You can add logic here to check if processing is active
    const isProcessing = false; // Replace with actual check

    if (isProcessing) {
      e.preventDefault();
      e.returnValue = 'Processing is in progress. Are you sure you want to leave?';
      return e.returnValue;
    }
  };

  // Show initialization error
  if (initError) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-red-500 to-red-600 text-white">
        <div className="text-center p-8 max-w-lg">
          <div className="text-6xl mb-4">⚠️</div>
          <h1 className="text-3xl font-bold mb-4">Initialization Error</h1>
          <p className="text-lg mb-6 opacity-90">
            Failed to initialize the application. Please refresh the page or contact support.
          </p>
          <div className="bg-white bg-opacity-10 rounded-lg p-4 mb-6 text-left">
            <p className="font-mono text-sm break-all">
              {initError.message || 'Unknown error'}
            </p>
          </div>
          <button
            onClick={() => window.location.reload()}
            className="bg-white text-red-600 px-6 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
          >
            🔄 Refresh Page
          </button>
        </div>
      </div>
    );
  }

  // Show loading state
  if (!isInitialized) {
    return <PageLoadingFallback />;
  }

  // Render main application
  return (
    <AppProvider>
      <div className="app-container">
        <AppLayout>
          <AppRoutes />
        </AppLayout>

        {/* Development indicator */}
        {import.meta.env.DEV && (
          <div className="fixed bottom-4 left-4 z-50 bg-purple-600 text-white px-3 py-1 rounded-full text-xs font-semibold shadow-lg">
            🛠️ DEV MODE
          </div>
        )}

        {/* User indicator */}
        <div className="fixed bottom-4 right-4 z-50 bg-gray-800 text-white px-3 py-1 rounded-full text-xs font-medium shadow-lg flex items-center gap-2">
          <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
          <span>A-P-U-R-B-O</span>
        </div>

        {/* Connection status indicator */}
        <ConnectionStatus />
      </div>
    </AppProvider>
  );
};

// ============================================================================
// Connection Status Component
// ============================================================================

const ConnectionStatus = () => {
  const [isOnline, setIsOnline] = useState(navigator.onLine);

  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  if (isOnline) return null;

  return (
    <motion.div
      initial={{ y: 50, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      exit={{ y: 50, opacity: 0 }}
      className="fixed bottom-20 left-1/2 transform -translate-x-1/2 z-50"
    >
      <div className="bg-red-600 text-white px-6 py-3 rounded-full shadow-lg flex items-center gap-3">
        <div className="w-3 h-3 bg-white rounded-full animate-pulse"></div>
        <span className="font-medium">You are offline</span>
      </div>
    </motion.div>
  );
};

// ============================================================================
// Export
// ============================================================================

export default App;
export { APP_METADATA };
