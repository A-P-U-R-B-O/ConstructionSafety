/**
 * useDetection Custom Hook
 * React hook for object detection operations
 * 
 * Author: A-P-U-R-B-O
 * Created: 2025-11-07 16:30:24 UTC
 * Version: 1.0.0
 */

import { useState, useCallback, useRef } from 'react';
import api, { fileToBase64 } from '@/services/api';
import toast from 'react-hot-toast';

/**
 * Custom hook for detection operations
 * 
 * @param {Object} options - Detection options
 * @param {number} options.confidenceThreshold - Confidence threshold (0-1)
 * @param {boolean} options.autoAlert - Auto-show alerts on detection
 * @param {Function} options.onDetection - Callback for detection results
 * @param {Function} options.onError - Error callback
 * @returns {Object} Detection state and functions
 */
const useDetection = (options = {}) => {
  const {
    confidenceThreshold = 0.5,
    autoAlert = true,
    onDetection = null,
    onError = null,
  } = options;

  const [isDetecting, setIsDetecting] = useState(false);
  const [detectionResult, setDetectionResult] = useState(null);
  const [error, setError] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [processingTime, setProcessingTime] = useState(0);
  const [history, setHistory] = useState([]);

  const abortControllerRef = useRef(null);
  const historyMaxSize = useRef(20);

  /**
   * Detect objects from base64 image
   * @param {string} base64Image - Base64 encoded image
   * @param {Object} detectionOptions - Additional options
   * @returns {Promise<Object>} Detection result
   */
  const detectBase64 = useCallback(async (base64Image, detectionOptions = {}) => {
    try {
      console.log('%c[useDetection] Starting detection (base64)...', 'color: #3b82f6; font-weight: bold');
      
      setIsDetecting(true);
      setError(null);
      setUploadProgress(0);

      const startTime = Date.now();

      const result = await api.detection.detectBase64(base64Image, {
        confidence: detectionOptions.confidence || confidenceThreshold,
        resize: detectionOptions.resize !== false,
        landmarks: detectionOptions.landmarks || false,
        alertSeverity: detectionOptions.alertSeverity || null,
      });

      const endTime = Date.now();
      const processingTimeMs = endTime - startTime;

      console.log('%c[useDetection] Detection completed', 'color: #22c55e; font-weight: bold');
      console.log('Results:', result);
      console.log(`Processing time: ${processingTimeMs}ms`);

      setDetectionResult(result);
      setProcessingTime(processingTimeMs);
      setIsDetecting(false);

      // Add to history
      addToHistory({
        type: 'base64',
        result,
        timestamp: new Date().toISOString(),
        processingTime: processingTimeMs,
      });

      // Show alerts if enabled
      if (autoAlert && result.alerts && result.alerts.length > 0) {
        result.alerts.forEach((alert) => {
          if (alert.severity === 'critical' || alert.severity === 'high') {
            toast.error(alert.message, {
              icon: '⚠️',
              duration: 5000,
            });
          }
        });
      }

      // Call callback if provided
      if (onDetection) {
        onDetection(result);
      }

      toast.success(`Detected ${result.total_objects} objects in ${processingTimeMs}ms`, {
        icon: '✅',
      });

      return result;
    } catch (err) {
      console.error('%c[useDetection] Detection failed:', 'color: #ef4444; font-weight: bold', err);
      
      const errorMessage = err.message || 'Detection failed';
      setError(errorMessage);
      setIsDetecting(false);

      if (onError) {
        onError(err);
      }

      toast.error(errorMessage);
      throw err;
    }
  }, [confidenceThreshold, autoAlert, onDetection, onError]);

  /**
   * Detect objects from image file
   * @param {File} imageFile - Image file object
   * @param {Object} detectionOptions - Additional options
   * @returns {Promise<Object>} Detection result
   */
  const detectFile = useCallback(async (imageFile, detectionOptions = {}) => {
    try {
      console.log('%c[useDetection] Starting detection (file)...', 'color: #3b82f6; font-weight: bold');
      console.log('File:', imageFile.name, `(${(imageFile.size / 1024).toFixed(2)} KB)`);
      
      setIsDetecting(true);
      setError(null);
      setUploadProgress(0);

      const startTime = Date.now();

      const result = await api.detection.detectFile(imageFile, {
        confidence: detectionOptions.confidence || confidenceThreshold,
        annotate: detectionOptions.annotate !== false,
        onProgress: (percent) => {
          setUploadProgress(percent);
        },
      });

      const endTime = Date.now();
      const processingTimeMs = endTime - startTime;

      console.log('%c[useDetection] Detection completed', 'color: #22c55e; font-weight: bold');
      console.log('Results:', result);
      console.log(`Total time: ${processingTimeMs}ms`);

      setDetectionResult(result);
      setProcessingTime(processingTimeMs);
      setIsDetecting(false);
      setUploadProgress(100);

      // Add to history
      addToHistory({
        type: 'file',
        filename: imageFile.name,
        result,
        timestamp: new Date().toISOString(),
        processingTime: processingTimeMs,
      });

      // Show alerts if enabled
      if (autoAlert && result.alerts && result.alerts.length > 0) {
        result.alerts.forEach((alert) => {
          if (alert.severity === 'critical' || alert.severity === 'high') {
            toast.error(alert.message, {
              icon: '⚠️',
              duration: 5000,
            });
          }
        });
      }

      // Call callback if provided
      if (onDetection) {
        onDetection(result);
      }

      toast.success(`Detected ${result.total_objects} objects`, {
        icon: '✅',
      });

      return result;
    } catch (err) {
      console.error('%c[useDetection] Detection failed:', 'color: #ef4444; font-weight: bold', err);
      
      const errorMessage = err.message || 'Detection failed';
      setError(errorMessage);
      setIsDetecting(false);
      setUploadProgress(0);

      if (onError) {
        onError(err);
      }

      toast.error(errorMessage);
      throw err;
    }
  }, [confidenceThreshold, autoAlert, onDetection, onError]);

  /**
   * Detect from multiple images (batch)
   * @param {Array<File>} imageFiles - Array of image files
   * @param {Object} detectionOptions - Additional options
   * @returns {Promise<Object>} Batch detection result
   */
  const detectBatch = useCallback(async (imageFiles, detectionOptions = {}) => {
    try {
      console.log('%c[useDetection] Starting batch detection...', 'color: #3b82f6; font-weight: bold');
      console.log(`Processing ${imageFiles.length} images`);
      
      setIsDetecting(true);
      setError(null);

      const startTime = Date.now();

      // Convert files to base64
      const base64Images = await Promise.all(
        imageFiles.map((file) => fileToBase64(file))
      );

      const result = await api.detection.detectBatch(base64Images, {
        confidence: detectionOptions.confidence || confidenceThreshold,
      });

      const endTime = Date.now();
      const processingTimeMs = endTime - startTime;

      console.log('%c[useDetection] Batch detection completed', 'color: #22c55e; font-weight: bold');
      console.log('Results:', result);

      setDetectionResult(result);
      setProcessingTime(processingTimeMs);
      setIsDetecting(false);

      // Add to history
      addToHistory({
        type: 'batch',
        count: imageFiles.length,
        result,
        timestamp: new Date().toISOString(),
        processingTime: processingTimeMs,
      });

      toast.success(
        `Processed ${result.successful_detections}/${result.total_images} images`,
        { icon: '✅' }
      );

      return result;
    } catch (err) {
      console.error('%c[useDetection] Batch detection failed:', 'color: #ef4444; font-weight: bold', err);
      
      const errorMessage = err.message || 'Batch detection failed';
      setError(errorMessage);
      setIsDetecting(false);

      if (onError) {
        onError(err);
      }

      toast.error(errorMessage);
      throw err;
    }
  }, [confidenceThreshold, onError]);

  /**
   * Cancel ongoing detection
   */
  const cancelDetection = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    setIsDetecting(false);
    setUploadProgress(0);
    setError('Detection cancelled');
    
    toast.error('Detection cancelled', { icon: '⏹️' });
  }, []);

  /**
   * Clear detection result
   */
  const clearResult = useCallback(() => {
    setDetectionResult(null);
    setError(null);
    setUploadProgress(0);
    setProcessingTime(0);
  }, []);

  /**
   * Add detection to history
   */
  const addToHistory = useCallback((entry) => {
    setHistory((prev) => {
      const newHistory = [entry, ...prev].slice(0, historyMaxSize.current);
      return newHistory;
    });
  }, []);

  /**
   * Clear detection history
   */
  const clearHistory = useCallback(() => {
    setHistory([]);
  }, []);

  /**
   * Get detection statistics
   */
  const getStats = useCallback(async () => {
    try {
      const stats = await api.detection.getStats();
      return stats;
    } catch (err) {
      console.error('[useDetection] Failed to get stats:', err);
      return null;
    }
  }, []);

  /**
   * Get supported classes
   */
  const getClasses = useCallback(async () => {
    try {
      const classes = await api.detection.getClasses();
      return classes;
    } catch (err) {
      console.error('[useDetection] Failed to get classes:', err);
      return null;
    }
  }, []);

  return {
    // State
    isDetecting,
    detectionResult,
    error,
    uploadProgress,
    processingTime,
    history,

    // Detection functions
    detectBase64,
    detectFile,
    detectBatch,

    // Controls
    cancelDetection,
    clearResult,
    clearHistory,

    // Utilities
    getStats,
    getClasses,
  };
};

export default useDetection;
