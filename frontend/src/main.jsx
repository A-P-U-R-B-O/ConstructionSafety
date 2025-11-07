/**
 * Main Entry Point - React Application
 * Construction Safety AI Detection System - Frontend
 * 
 * Author: A-P-U-R-B-O
 * Created: 2025-11-07 15:29:28 UTC
 * Version: 1.0.0
 * 
 * This file initializes the React application, sets up providers,
 * and mounts the app to the DOM.
 */

import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import App from './App';
import './index.css';

// ============================================================================
// Constants & Configuration
// ============================================================================

const APP_CONFIG = {
  name: 'Construction Safety AI',
  version: '1.0.0',
  author: 'A-P-U-R-B-O',
  buildTime: __BUILD_TIME__,
  apiUrl: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  wsUrl: import.meta.env.VITE_WS_URL || 'ws://localhost:8000',
};

// ============================================================================
// Logger Utility
// ============================================================================

const Logger = {
  info: (message, ...args) => {
    console.log(`%c[INFO]%c ${message}`, 'color: #3b82f6; font-weight: bold', 'color: inherit', ...args);
  },
  success: (message, ...args) => {
    console.log(`%c[SUCCESS]%c ${message}`, 'color: #22c55e; font-weight: bold', 'color: inherit', ...args);
  },
  warn: (message, ...args) => {
    console.warn(`%c[WARN]%c ${message}`, 'color: #f59e0b; font-weight: bold', 'color: inherit', ...args);
  },
  error: (message, ...args) => {
    console.error(`%c[ERROR]%c ${message}`, 'color: #ef4444; font-weight: bold', 'color: inherit', ...args);
  },
  debug: (message, ...args) => {
    if (import.meta.env.DEV) {
      console.debug(`%c[DEBUG]%c ${message}`, 'color: #8b5cf6; font-weight: bold', 'color: inherit', ...args);
    }
  },
};

// ============================================================================
// Environment Validation
// ============================================================================

const validateEnvironment = () => {
  Logger.info('🔍 Validating environment...');

  const requiredEnvVars = ['VITE_API_URL', 'VITE_WS_URL'];
  const missingVars = requiredEnvVars.filter(
    (varName) => !import.meta.env[varName]
  );

  if (missingVars.length > 0) {
    Logger.warn(
      `⚠️  Missing environment variables: ${missingVars.join(', ')}`
    );
    Logger.warn('Using default values instead');
  }

  // Log configuration
  Logger.info('📋 Application Configuration:');
  console.table({
    Name: APP_CONFIG.name,
    Version: APP_CONFIG.version,
    Author: APP_CONFIG.author,
    'Build Time': APP_CONFIG.buildTime,
    'API URL': APP_CONFIG.apiUrl,
    'WebSocket URL': APP_CONFIG.wsUrl,
    Mode: import.meta.env.MODE,
    Dev: import.meta.env.DEV,
  });

  Logger.success('✅ Environment validation complete');
};

// ============================================================================
// Browser Compatibility Check
// ============================================================================

const checkBrowserCompatibility = () => {
  Logger.info('🌐 Checking browser compatibility...');

  const features = {
    WebSocket: typeof WebSocket !== 'undefined',
    'Web Workers': typeof Worker !== 'undefined',
    'Local Storage': typeof Storage !== 'undefined',
    'Canvas API': !!document.createElement('canvas').getContext,
    'Fetch API': typeof fetch !== 'undefined',
    'ES6 Support': typeof Promise !== 'undefined',
  };

  const unsupportedFeatures = Object.entries(features)
    .filter(([_, supported]) => !supported)
    .map(([feature]) => feature);

  if (unsupportedFeatures.length > 0) {
    Logger.error(
      '❌ Browser compatibility issues detected:',
      unsupportedFeatures
    );
    Logger.error(
      'Please upgrade to a modern browser (Chrome 90+, Firefox 88+, Safari 14+, Edge 90+)'
    );
  } else {
    Logger.success('✅ Browser is compatible with all required features');
  }

  // Check for specific APIs we need
  if (!('mediaDevices' in navigator) || !('getUserMedia' in navigator.mediaDevices)) {
    Logger.warn('⚠️  Camera API may not be available');
  }

  return unsupportedFeatures.length === 0;
};

// ============================================================================
// Performance Monitoring
// ============================================================================

const setupPerformanceMonitoring = () => {
  if (import.meta.env.PROD) {
    // Monitor Core Web Vitals in production
    if ('web-vitals' in window) {
      import('web-vitals').then(({ getCLS, getFID, getFCP, getLCP, getTTFB }) => {
        getCLS(console.log);
        getFID(console.log);
        getFCP(console.log);
        getLCP(console.log);
        getTTFB(console.log);
      }).catch(() => {
        Logger.debug('Web Vitals not available');
      });
    }
  }

  // Log initial page load performance
  window.addEventListener('load', () => {
    if (window.performance && window.performance.timing) {
      const perfData = window.performance.timing;
      const pageLoadTime = perfData.loadEventEnd - perfData.navigationStart;

      Logger.info('📊 Page Performance Metrics:');
      console.table({
        'Total Load Time': `${pageLoadTime}ms`,
        'DNS Lookup': `${perfData.domainLookupEnd - perfData.domainLookupStart}ms`,
        'TCP Connection': `${perfData.connectEnd - perfData.connectStart}ms`,
        'Server Response': `${perfData.responseEnd - perfData.requestStart}ms`,
        'DOM Processing': `${perfData.domComplete - perfData.domLoading}ms`,
        'Page Render': `${perfData.loadEventEnd - perfData.loadEventStart}ms`,
      });
    }
  });
};

// ============================================================================
// Error Boundary Setup
// ============================================================================

const setupErrorHandling = () => {
  // Global error handler
  window.addEventListener('error', (event) => {
    Logger.error('❌ Global Error:', event.error);
    
    // You can send errors to a logging service here
    if (import.meta.env.PROD) {
      // Example: Send to error tracking service
      // sendErrorToService(event.error);
    }
  });

  // Unhandled promise rejection handler
  window.addEventListener('unhandledrejection', (event) => {
    Logger.error('❌ Unhandled Promise Rejection:', event.reason);
    
    if (import.meta.env.PROD) {
      // Example: Send to error tracking service
      // sendErrorToService(event.reason);
    }
  });
};

// ============================================================================
// Service Worker Registration (PWA)
// ============================================================================

const registerServiceWorker = async () => {
  if ('serviceWorker' in navigator && import.meta.env.PROD) {
    try {
      Logger.info('📡 Registering Service Worker...');
      
      const registration = await navigator.serviceWorker.register('/sw.js', {
        scope: '/',
      });

      Logger.success('✅ Service Worker registered:', registration.scope);

      // Listen for updates
      registration.addEventListener('updatefound', () => {
        const newWorker = registration.installing;
        
        newWorker?.addEventListener('statechange', () => {
          if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
            Logger.info('🔄 New version available! Please refresh.');
            
            // Show update notification
            // You can trigger a toast notification here
          }
        });
      });
    } catch (error) {
      Logger.error('❌ Service Worker registration failed:', error);
    }
  }
};

// ============================================================================
// Development Tools
// ============================================================================

const setupDevelopmentTools = () => {
  if (import.meta.env.DEV) {
    Logger.debug('🛠️  Development mode enabled');

    // Add helpful globals for debugging
    window.__APP_CONFIG__ = APP_CONFIG;
    window.__APP_VERSION__ = APP_CONFIG.version;
    window.__REACT_VERSION__ = React.version;

    // React DevTools detection
    if (typeof window.__REACT_DEVTOOLS_GLOBAL_HOOK__ !== 'undefined') {
      Logger.success('🔧 React DevTools detected');
    }

    Logger.debug('💡 Debugging helpers available:');
    Logger.debug('  - window.__APP_CONFIG__');
    Logger.debug('  - window.__APP_VERSION__');
    Logger.debug('  - window.__REACT_VERSION__');
  }
};

// ============================================================================
// App Initialization Banner
// ============================================================================

const printInitializationBanner = () => {
  const banner = `
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║              🏗️  CONSTRUCTION SAFETY AI                       ║
║                                                                ║
║  Real-time AI-Powered Construction Site Safety Monitoring     ║
║                                                                ║
║  Version: ${APP_CONFIG.version.padEnd(50)}║
║  Author:  ${APP_CONFIG.author.padEnd(50)}║
║  Date:    2025-11-07 15:29:28 UTC${' '.repeat(28)}║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
  `;

  console.log('%c' + banner, 'color: #ef4444; font-family: monospace;');
};

// ============================================================================
// Main Initialization
// ============================================================================

const initialize = async () => {
  try {
    // Print banner
    printInitializationBanner();

    // Run initialization checks
    Logger.info('🚀 Initializing application...');
    
    validateEnvironment();
    const isCompatible = checkBrowserCompatibility();
    
    if (!isCompatible) {
      Logger.warn('⚠️  Some features may not work correctly');
    }

    setupErrorHandling();
    setupPerformanceMonitoring();
    setupDevelopmentTools();

    // Register service worker (PWA)
    await registerServiceWorker();

    Logger.success('✅ Application initialization complete!');
    Logger.info('🎯 Rendering React application...');
  } catch (error) {
    Logger.error('❌ Initialization failed:', error);
    throw error;
  }
};

// ============================================================================
// React Root Render
// ============================================================================

const renderApp = () => {
  const rootElement = document.getElementById('root');

  if (!rootElement) {
    Logger.error('❌ Root element not found!');
    throw new Error('Failed to find root element');
  }

  // Create React root
  const root = ReactDOM.createRoot(rootElement);

  // Render application
  root.render(
    <React.StrictMode>
      <BrowserRouter>
        <App />
        
        {/* Toast Notifications */}
        <Toaster
          position="top-right"
          reverseOrder={false}
          gutter={8}
          containerClassName=""
          containerStyle={{}}
          toastOptions={{
            // Default options
            duration: 4000,
            style: {
              background: '#363636',
              color: '#fff',
              fontSize: '14px',
              fontWeight: '500',
              borderRadius: '8px',
              padding: '12px 16px',
            },
            // Success style
            success: {
              duration: 3000,
              iconTheme: {
                primary: '#22c55e',
                secondary: '#fff',
              },
              style: {
                background: '#22c55e',
                color: '#fff',
              },
            },
            // Error style
            error: {
              duration: 5000,
              iconTheme: {
                primary: '#ef4444',
                secondary: '#fff',
              },
              style: {
                background: '#ef4444',
                color: '#fff',
              },
            },
            // Warning style
            loading: {
              iconTheme: {
                primary: '#f59e0b',
                secondary: '#fff',
              },
            },
          }}
        />
      </BrowserRouter>
    </React.StrictMode>
  );

  Logger.success('✅ React application rendered successfully!');
  Logger.info('🎨 User Interface ready');
  Logger.info('📡 Waiting for user interaction...');
};

// ============================================================================
// Bootstrap Application
// ============================================================================

(async () => {
  try {
    await initialize();
    renderApp();
  } catch (error) {
    Logger.error('❌ Fatal error during application bootstrap:', error);
    
    // Show error UI
    const rootElement = document.getElementById('root');
    if (rootElement) {
      rootElement.innerHTML = `
        <div style="
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 100vh;
          padding: 2rem;
          text-align: center;
          background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
          color: white;
          font-family: Inter, system-ui, sans-serif;
        ">
          <div style="font-size: 4rem; margin-bottom: 1rem;">⚠️</div>
          <h1 style="font-size: 2rem; font-weight: 700; margin-bottom: 1rem;">
            Application Failed to Load
          </h1>
          <p style="font-size: 1.125rem; margin-bottom: 2rem; max-width: 600px;">
            We encountered an error while initializing the Construction Safety AI system.
            Please try refreshing the page or contact support if the problem persists.
          </p>
          <button
            onclick="window.location.reload()"
            style="
              background: white;
              color: #ef4444;
              font-size: 1rem;
              font-weight: 600;
              padding: 0.75rem 2rem;
              border: none;
              border-radius: 8px;
              cursor: pointer;
              transition: all 0.2s;
            "
            onmouseover="this.style.transform='scale(1.05)'"
            onmouseout="this.style.transform='scale(1)'"
          >
            🔄 Refresh Page
          </button>
          <div style="
            margin-top: 2rem;
            padding: 1rem;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            font-family: 'Fira Code', monospace;
            font-size: 0.875rem;
            max-width: 600px;
            word-break: break-all;
          ">
            <strong>Error Details:</strong><br>
            ${error.message || 'Unknown error'}
          </div>
        </div>
      `;
    }
  }
})();

// ============================================================================
// Hot Module Replacement (HMR) - Development Only
// ============================================================================

if (import.meta.hot) {
  import.meta.hot.accept();
  Logger.debug('🔥 Hot Module Replacement enabled');
}

// ============================================================================
// Export for Testing
// ============================================================================

export { APP_CONFIG, Logger };
