/**
 * API Service Client
 * Centralized API communication layer for Construction Safety AI
 * 
 * Author: A-P-U-R-B-O
 * Created: 2025-11-07 16:18:02 UTC
 * Version: 1.0.0
 * 
 * This module provides a complete API client with:
 * - RESTful endpoints
 * - WebSocket connections
 * - File upload handling
 * - Error handling
 * - Request/response interceptors
 * - Authentication (future)
 */

import axios from 'axios';
import toast from 'react-hot-toast';

// ============================================================================
// Configuration
// ============================================================================

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const WS_BASE_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';
const API_TIMEOUT = 30000; // 30 seconds
const UPLOAD_TIMEOUT = 120000; // 2 minutes for file uploads

// Current user info
const CURRENT_USER = 'A-P-U-R-B-O';
const CURRENT_TIMESTAMP = '2025-11-07 16:18:02';

// ============================================================================
// Axios Instance Configuration
// ============================================================================

/**
 * Create axios instance with default configuration
 */
const axiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: API_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
});

/**
 * Create axios instance for file uploads
 */
const axiosUploadInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: UPLOAD_TIMEOUT,
  headers: {
    'Content-Type': 'multipart/form-data',
  },
});

// ============================================================================
// Request Interceptor
// ============================================================================

axiosInstance.interceptors.request.use(
  (config) => {
    // Add timestamp to all requests
    config.headers['X-Request-Time'] = new Date().toISOString();
    
    // Add user info
    config.headers['X-User'] = CURRENT_USER;
    
    // Add request ID for tracking
    config.headers['X-Request-ID'] = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // Log request in development
    if (import.meta.env.DEV) {
      console.log(`%c[API Request] ${config.method?.toUpperCase()} ${config.url}`, 'color: #3b82f6; font-weight: bold');
      if (config.data) {
        console.log('Request Data:', config.data);
      }
    }
    
    return config;
  },
  (error) => {
    console.error('[API Request Error]:', error);
    return Promise.reject(error);
  }
);

// Apply same interceptor to upload instance
axiosUploadInstance.interceptors.request.use(
  axiosInstance.interceptors.request.handlers[0].fulfilled,
  axiosInstance.interceptors.request.handlers[0].rejected
);

// ============================================================================
// Response Interceptor
// ============================================================================

axiosInstance.interceptors.response.use(
  (response) => {
    // Log response in development
    if (import.meta.env.DEV) {
      const processingTime = response.headers['x-processing-time'];
      console.log(
        `%c[API Response] ${response.config.method?.toUpperCase()} ${response.config.url}`,
        'color: #22c55e; font-weight: bold'
      );
      if (processingTime) {
        console.log(`Processing Time: ${processingTime}`);
      }
      console.log('Response Data:', response.data);
    }
    
    return response;
  },
  (error) => {
    // Handle errors
    const errorResponse = handleApiError(error);
    
    // Log error in development
    if (import.meta.env.DEV) {
      console.error(
        `%c[API Error] ${error.config?.method?.toUpperCase()} ${error.config?.url}`,
        'color: #ef4444; font-weight: bold',
        errorResponse
      );
    }
    
    return Promise.reject(errorResponse);
  }
);

// Apply same interceptor to upload instance
axiosUploadInstance.interceptors.response.use(
  axiosInstance.interceptors.response.handlers[0].fulfilled,
  axiosInstance.interceptors.response.handlers[0].rejected
);

// ============================================================================
// Error Handler
// ============================================================================

/**
 * Handle API errors and format them consistently
 * @param {Error} error - Axios error object
 * @returns {Object} Formatted error object
 */
function handleApiError(error) {
  if (error.response) {
    // Server responded with error status
    const { status, data } = error.response;
    
    const errorObj = {
      success: false,
      status,
      message: data?.error || data?.message || 'An error occurred',
      detail: data?.detail || null,
      errors: data?.errors || null,
      timestamp: new Date().toISOString(),
    };
    
    // Show toast notification for errors
    if (status >= 500) {
      toast.error(`Server Error: ${errorObj.message}`);
    } else if (status === 404) {
      toast.error('Resource not found');
    } else if (status === 401) {
      toast.error('Unauthorized access');
    } else if (status === 403) {
      toast.error('Access forbidden');
    }
    
    return errorObj;
  } else if (error.request) {
    // Request made but no response
    const errorObj = {
      success: false,
      status: 0,
      message: 'No response from server. Please check your connection.',
      detail: 'Network error or server is down',
      timestamp: new Date().toISOString(),
    };
    
    toast.error('Connection failed. Check your network.');
    
    return errorObj;
  } else {
    // Error setting up request
    const errorObj = {
      success: false,
      status: 0,
      message: error.message || 'Request failed',
      detail: null,
      timestamp: new Date().toISOString(),
    };
    
    toast.error('Request failed');
    
    return errorObj;
  }
}

// ============================================================================
// API Service Object
// ============================================================================

const api = {
  
  // ==========================================================================
  // Health & System
  // ==========================================================================
  
  /**
   * Check API health
   * @returns {Promise<Object>} Health status
   */
  health: {
    /**
     * Basic health check
     */
    check: async () => {
      const response = await axiosInstance.get('/health');
      return response.data;
    },
    
    /**
     * Liveness probe
     */
    live: async () => {
      const response = await axiosInstance.get('/health/live');
      return response.data;
    },
    
    /**
     * Readiness probe
     */
    ready: async () => {
      const response = await axiosInstance.get('/health/ready');
      return response.data;
    },
    
    /**
     * Detailed health check
     */
    detailed: async () => {
      const response = await axiosInstance.get('/health/detailed');
      return response.data;
    },
    
    /**
     * System metrics
     */
    metrics: async () => {
      const response = await axiosInstance.get('/health/metrics');
      return response.data;
    },
  },
  
  // ==========================================================================
  // Detection
  // ==========================================================================
  
  detection: {
    /**
     * Detect objects from base64 image
     * @param {string} imageBase64 - Base64 encoded image
     * @param {Object} options - Detection options
     * @returns {Promise<Object>} Detection results
     */
    detectBase64: async (imageBase64, options = {}) => {
      const response = await axiosInstance.post('/api/v1/detection/detect', {
        image_base64: imageBase64,
        confidence_threshold: options.confidence || 0.5,
        resize: options.resize !== false,
        include_landmarks: options.landmarks || false,
        alert_severity: options.alertSeverity || null,
      });
      return response.data;
    },
    
    /**
     * Detect objects from image file upload
     * @param {File} imageFile - Image file object
     * @param {Object} options - Detection options
     * @returns {Promise<Object>} Detection results
     */
    detectFile: async (imageFile, options = {}) => {
      const formData = new FormData();
      formData.append('file', imageFile);
      
      // Add options as query params
      const params = new URLSearchParams();
      if (options.confidence) params.append('confidence_threshold', options.confidence);
      if (options.annotate !== false) params.append('annotate', 'true');
      
      const response = await axiosUploadInstance.post(
        `/api/v1/detection/upload?${params.toString()}`,
        formData,
        {
          onUploadProgress: (progressEvent) => {
            if (options.onProgress) {
              const percentCompleted = Math.round(
                (progressEvent.loaded * 100) / progressEvent.total
              );
              options.onProgress(percentCompleted);
            }
          },
        }
      );
      return response.data;
    },
    
    /**
     * Batch detection from multiple images
     * @param {Array<string>} imageBase64Array - Array of base64 images
     * @param {Object} options - Detection options
     * @returns {Promise<Object>} Batch detection results
     */
    detectBatch: async (imageBase64Array, options = {}) => {
      const response = await axiosInstance.post('/api/v1/detection/batch', {
        images: imageBase64Array,
        confidence_threshold: options.confidence || 0.5,
      });
      return response.data;
    },
    
    /**
     * Get detection statistics
     * @returns {Promise<Object>} Detection statistics
     */
    getStats: async () => {
      const response = await axiosInstance.get('/api/v1/detection/stats');
      return response.data;
    },
    
    /**
     * Get supported classes
     * @returns {Promise<Object>} List of supported detection classes
     */
    getClasses: async () => {
      const response = await axiosInstance.get('/api/v1/detection/classes');
      return response.data;
    },
  },
  
  // ==========================================================================
  // Video Processing
  // ==========================================================================
  
  video: {
    /**
     * Process video file
     * @param {File} videoFile - Video file object
     * @param {Object} options - Processing options
     * @returns {Promise<Object>} Processing results
     */
    process: async (videoFile, options = {}) => {
      const formData = new FormData();
      formData.append('file', videoFile);
      
      const params = new URLSearchParams();
      if (options.fps) params.append('fps', options.fps);
      if (options.confidence) params.append('confidence_threshold', options.confidence);
      if (options.output !== false) params.append('save_output', 'true');
      
      const response = await axiosUploadInstance.post(
        `/api/v1/video/process?${params.toString()}`,
        formData,
        {
          onUploadProgress: (progressEvent) => {
            if (options.onProgress) {
              const percentCompleted = Math.round(
                (progressEvent.loaded * 100) / progressEvent.total
              );
              options.onProgress(percentCompleted);
            }
          },
        }
      );
      return response.data;
    },
    
    /**
     * Get video processing status
     * @param {string} processId - Processing ID
     * @returns {Promise<Object>} Processing status
     */
    getStatus: async (processId) => {
      const response = await axiosInstance.get(`/api/v1/video/status/${processId}`);
      return response.data;
    },
  },
  
  // ==========================================================================
  // Alerts
  // ==========================================================================
  
  alerts: {
    /**
     * Get all alerts with optional filters
     * @param {Object} filters - Filter options
     * @returns {Promise<Object>} Alerts list
     */
    getAll: async (filters = {}) => {
      const params = new URLSearchParams();
      if (filters.severity) params.append('severity', filters.severity);
      if (filters.status) params.append('status', filters.status);
      if (filters.from) params.append('from_timestamp', filters.from);
      if (filters.to) params.append('to_timestamp', filters.to);
      if (filters.limit) params.append('limit', filters.limit);
      if (filters.offset) params.append('offset', filters.offset);
      
      const response = await axiosInstance.get(`/api/v1/alerts?${params.toString()}`);
      return response.data;
    },
    
    /**
     * Get alert by ID
     * @param {string} alertId - Alert ID
     * @returns {Promise<Object>} Alert details
     */
    getById: async (alertId) => {
      const response = await axiosInstance.get(`/api/v1/alerts/${alertId}`);
      return response.data;
    },
    
    /**
     * Acknowledge alert
     * @param {string} alertId - Alert ID
     * @param {string} notes - Optional notes
     * @returns {Promise<Object>} Updated alert
     */
    acknowledge: async (alertId, notes = '') => {
      const response = await axiosInstance.post(`/api/v1/alerts/${alertId}/acknowledge`, {
        acknowledged_by: CURRENT_USER,
        notes,
      });
      return response.data;
    },
    
    /**
     * Resolve alert
     * @param {string} alertId - Alert ID
     * @param {string} notes - Resolution notes
     * @param {Array<string>} actions - Actions taken
     * @returns {Promise<Object>} Updated alert
     */
    resolve: async (alertId, notes, actions = []) => {
      const response = await axiosInstance.post(`/api/v1/alerts/${alertId}/resolve`, {
        resolved_by: CURRENT_USER,
        resolution_notes: notes,
        actions_taken: actions,
      });
      return response.data;
    },
    
    /**
     * Get alert statistics
     * @returns {Promise<Object>} Alert statistics
     */
    getStats: async () => {
      const response = await axiosInstance.get('/api/v1/alerts/stats');
      return response.data;
    },
  },
  
  // ==========================================================================
  // Analytics
  // ==========================================================================
  
  analytics: {
    /**
     * Get dashboard analytics
     * @param {Object} params - Query parameters
     * @returns {Promise<Object>} Analytics data
     */
    getDashboard: async (params = {}) => {
      const response = await axiosInstance.get('/api/v1/analytics/dashboard', { params });
      return response.data;
    },
    
    /**
     * Get detection trends
     * @param {string} period - Time period (day, week, month)
     * @returns {Promise<Object>} Trend data
     */
    getTrends: async (period = 'week') => {
      const response = await axiosInstance.get(`/api/v1/analytics/trends?period=${period}`);
      return response.data;
    },
    
    /**
     * Get danger assessment history
     * @param {Object} params - Query parameters
     * @returns {Promise<Object>} Assessment history
     */
    getDangerHistory: async (params = {}) => {
      const response = await axiosInstance.get('/api/v1/analytics/danger-history', { params });
      return response.data;
    },
    
    /**
     * Export analytics report
     * @param {string} format - Export format (pdf, csv, json)
     * @returns {Promise<Blob>} Report file
     */
    exportReport: async (format = 'pdf') => {
      const response = await axiosInstance.get(`/api/v1/analytics/export?format=${format}`, {
        responseType: 'blob',
      });
      return response.data;
    },
  },
  
  // ==========================================================================
  // System Info
  // ==========================================================================
  
  system: {
    /**
     * Get system information
     * @returns {Promise<Object>} System info
     */
    getInfo: async () => {
      const response = await axiosInstance.get('/info');
      return response.data;
    },
    
    /**
     * Get system status
     * @returns {Promise<Object>} System status
     */
    getStatus: async () => {
      const response = await axiosInstance.get('/status');
      return response.data;
    },
    
    /**
     * Reset system (admin only)
     * @returns {Promise<Object>} Reset confirmation
     */
    reset: async () => {
      const response = await axiosInstance.post('/reset');
      return response.data;
    },
  },
};

// ============================================================================
// WebSocket Manager
// ============================================================================

/**
 * WebSocket connection manager for real-time communication
 */
export class WebSocketManager {
  constructor(endpoint = '/ws/detect') {
    this.endpoint = endpoint;
    this.ws = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 1000;
    this.listeners = {
      open: [],
      message: [],
      error: [],
      close: [],
    };
  }
  
  /**
   * Connect to WebSocket
   */
  connect() {
    const wsUrl = `${WS_BASE_URL}${this.endpoint}`;
    
    console.log(`%c[WebSocket] Connecting to ${wsUrl}`, 'color: #8b5cf6; font-weight: bold');
    
    this.ws = new WebSocket(wsUrl);
    
    this.ws.onopen = (event) => {
      console.log('%c[WebSocket] Connected', 'color: #22c55e; font-weight: bold');
      this.reconnectAttempts = 0;
      this.listeners.open.forEach(callback => callback(event));
      toast.success('Connected to live monitoring', { icon: '📡' });
    };
    
    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (import.meta.env.DEV) {
          console.log('%c[WebSocket] Message received', 'color: #3b82f6', data);
        }
        this.listeners.message.forEach(callback => callback(data));
      } catch (error) {
        console.error('[WebSocket] Error parsing message:', error);
      }
    };
    
    this.ws.onerror = (event) => {
      console.error('%c[WebSocket] Error', 'color: #ef4444; font-weight: bold', event);
      this.listeners.error.forEach(callback => callback(event));
      toast.error('WebSocket connection error');
    };
    
    this.ws.onclose = (event) => {
      console.log('%c[WebSocket] Disconnected', 'color: #f59e0b; font-weight: bold');
      this.listeners.close.forEach(callback => callback(event));
      
      // Attempt reconnection
      if (this.reconnectAttempts < this.maxReconnectAttempts) {
        this.reconnectAttempts++;
        const delay = this.reconnectDelay * this.reconnectAttempts;
        console.log(`[WebSocket] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        setTimeout(() => this.connect(), delay);
      } else {
        toast.error('WebSocket connection lost. Please refresh.', { duration: 5000 });
      }
    };
  }
  
  /**
   * Send data through WebSocket
   * @param {Object} data - Data to send
   */
  send(data) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
      if (import.meta.env.DEV) {
        console.log('%c[WebSocket] Sent', 'color: #3b82f6', data);
      }
    } else {
      console.error('[WebSocket] Not connected. Cannot send data.');
      toast.error('WebSocket not connected');
    }
  }
  
  /**
   * Register event listener
   * @param {string} event - Event name (open, message, error, close)
   * @param {Function} callback - Callback function
   */
  on(event, callback) {
    if (this.listeners[event]) {
      this.listeners[event].push(callback);
    }
  }
  
  /**
   * Remove event listener
   * @param {string} event - Event name
   * @param {Function} callback - Callback function
   */
  off(event, callback) {
    if (this.listeners[event]) {
      this.listeners[event] = this.listeners[event].filter(cb => cb !== callback);
    }
  }
  
  /**
   * Close WebSocket connection
   */
  close() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
      console.log('%c[WebSocket] Connection closed', 'color: #6b7280');
    }
  }
  
  /**
   * Check if WebSocket is connected
   * @returns {boolean} Connection status
   */
  isConnected() {
    return this.ws && this.ws.readyState === WebSocket.OPEN;
  }
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Convert file to base64
 * @param {File} file - File object
 * @returns {Promise<string>} Base64 string
 */
export const fileToBase64 = (file) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => {
      // Remove data URI prefix
      const base64 = reader.result.split(',')[1];
      resolve(base64);
    };
    reader.onerror = (error) => reject(error);
  });
};

/**
 * Download blob as file
 * @param {Blob} blob - Blob data
 * @param {string} filename - File name
 */
export const downloadBlob = (blob, filename) => {
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
};

/**
 * Format file size
 * @param {number} bytes - File size in bytes
 * @returns {string} Formatted size
 */
export const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
};

// ============================================================================
// Export
// ============================================================================

export default api;

// Export configuration for external use
export const config = {
  API_BASE_URL,
  WS_BASE_URL,
  CURRENT_USER,
  CURRENT_TIMESTAMP,
};

// Log initialization
console.log('%c[API Service] Initialized', 'color: #22c55e; font-weight: bold');
console.log('API Base URL:', API_BASE_URL);
console.log('WebSocket URL:', WS_BASE_URL);
console.log('Current User:', CURRENT_USER);
console.log('Timestamp:', CURRENT_TIMESTAMP);