/**
 * useCamera Custom Hook
 * React hook for camera/webcam access and management
 * 
 * Author: A-P-U-R-B-O
 * Created: 2025-11-07 16:30:24 UTC
 * Version: 1.0.0
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import toast from 'react-hot-toast';

/**
 * Custom hook for camera access and video capture
 * 
 * @param {Object} options - Camera options
 * @param {Object} options.constraints - Media constraints
 * @param {boolean} options.autoStart - Auto-start camera on mount
 * @param {Function} options.onFrame - Callback for each frame
 * @param {number} options.frameRate - Frame capture rate (fps)
 * @returns {Object} Camera state and controls
 */
const useCamera = (options = {}) => {
  const {
    constraints = {
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: 'user',
      },
      audio: false,
    },
    autoStart = false,
    onFrame = null,
    frameRate = 10, // Default 10 FPS
  } = options;

  const [isActive, setIsActive] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [devices, setDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState(null);
  const [stream, setStream] = useState(null);
  const [videoSize, setVideoSize] = useState({ width: 0, height: 0 });

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const frameIntervalRef = useRef(null);

  /**
   * Get available camera devices
   */
  const getDevices = useCallback(async () => {
    try {
      const deviceList = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = deviceList.filter((device) => device.kind === 'videoinput');
      
      console.log('[useCamera] Available devices:', videoDevices);
      setDevices(videoDevices);
      
      if (videoDevices.length > 0 && !selectedDeviceId) {
        setSelectedDeviceId(videoDevices[0].deviceId);
      }
      
      return videoDevices;
    } catch (error) {
      console.error('[useCamera] Error getting devices:', error);
      setError('Failed to get camera devices');
      return [];
    }
  }, [selectedDeviceId]);

  /**
   * Start camera stream
   */
  const startCamera = useCallback(async () => {
    try {
      console.log('%c[useCamera] Starting camera...', 'color: #3b82f6; font-weight: bold');
      setIsLoading(true);
      setError(null);

      // Build constraints with selected device
      const mediaConstraints = {
        ...constraints,
        video: {
          ...constraints.video,
          ...(selectedDeviceId && { deviceId: { exact: selectedDeviceId } }),
        },
      };

      // Request camera access
      const mediaStream = await navigator.mediaDevices.getUserMedia(mediaConstraints);

      console.log('%c[useCamera] Camera access granted', 'color: #22c55e; font-weight: bold');
      
      streamRef.current = mediaStream;
      setStream(mediaStream);

      // Attach to video element
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        
        // Wait for video to load
        await new Promise((resolve) => {
          videoRef.current.onloadedmetadata = () => {
            const { videoWidth, videoHeight } = videoRef.current;
            setVideoSize({ width: videoWidth, height: videoHeight });
            resolve();
          };
        });
      }

      setIsActive(true);
      setIsLoading(false);
      
      toast.success('Camera started successfully', { icon: '📹' });

      // Start frame capture if callback provided
      if (onFrame) {
        startFrameCapture();
      }
    } catch (error) {
      console.error('%c[useCamera] Camera start failed:', 'color: #ef4444; font-weight: bold', error);
      
      let errorMessage = 'Failed to access camera';
      
      if (error.name === 'NotAllowedError') {
        errorMessage = 'Camera access denied. Please grant permission.';
      } else if (error.name === 'NotFoundError') {
        errorMessage = 'No camera found on this device';
      } else if (error.name === 'NotReadableError') {
        errorMessage = 'Camera is already in use';
      }
      
      setError(errorMessage);
      setIsLoading(false);
      toast.error(errorMessage);
      throw error;
    }
  }, [constraints, selectedDeviceId, onFrame]);

  /**
   * Stop camera stream
   */
  const stopCamera = useCallback(() => {
    console.log('%c[useCamera] Stopping camera...', 'color: #6b7280');

    // Stop frame capture
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }

    // Stop all tracks
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        track.stop();
      });
    }

    // Clear video element
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    streamRef.current = null;
    setStream(null);
    setIsActive(false);
    setError(null);
    
    toast.success('Camera stopped', { icon: '⏹️' });
  }, []);

  /**
   * Start capturing frames at specified rate
   */
  const startFrameCapture = useCallback(() => {
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
    }

    const intervalMs = 1000 / frameRate;

    frameIntervalRef.current = setInterval(() => {
      if (videoRef.current && canvasRef.current && onFrame) {
        captureFrame();
      }
    }, intervalMs);

    console.log(`[useCamera] Frame capture started at ${frameRate} FPS`);
  }, [frameRate, onFrame]);

  /**
   * Capture a single frame from video
   */
  const captureFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) {
      return null;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Get image data
    const imageData = canvas.toDataURL('image/jpeg', 0.8);
    
    // Get base64 without prefix
    const base64 = imageData.split(',')[1];

    // Call frame callback if provided
    if (onFrame) {
      onFrame({
        base64,
        imageData,
        timestamp: new Date().toISOString(),
        width: canvas.width,
        height: canvas.height,
      });
    }

    return { base64, imageData };
  }, [onFrame]);

  /**
   * Take a snapshot
   */
  const takeSnapshot = useCallback(() => {
    if (!isActive) {
      toast.error('Camera is not active');
      return null;
    }

    const frameData = captureFrame();
    
    if (frameData) {
      toast.success('Snapshot captured!', { icon: '📸' });
    }
    
    return frameData;
  }, [isActive, captureFrame]);

  /**
   * Switch to different camera device
   */
  const switchCamera = useCallback(async (deviceId) => {
    console.log(`[useCamera] Switching to device: ${deviceId}`);
    
    const wasActive = isActive;
    
    if (wasActive) {
      stopCamera();
    }
    
    setSelectedDeviceId(deviceId);
    
    if (wasActive) {
      // Small delay before restarting
      setTimeout(() => {
        startCamera();
      }, 500);
    }
  }, [isActive, stopCamera, startCamera]);

  /**
   * Toggle camera on/off
   */
  const toggleCamera = useCallback(() => {
    if (isActive) {
      stopCamera();
    } else {
      startCamera();
    }
  }, [isActive, startCamera, stopCamera]);

  // Get devices on mount
  useEffect(() => {
    getDevices();
  }, [getDevices]);

  // Auto-start camera if enabled
  useEffect(() => {
    if (autoStart && devices.length > 0) {
      startCamera();
    }

    // Cleanup on unmount
    return () => {
      stopCamera();
    };
  }, [autoStart, devices]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current);
      }
    };
  }, []);

  return {
    // State
    isActive,
    isLoading,
    error,
    stream,
    videoSize,
    devices,
    selectedDeviceId,

    // Refs
    videoRef,
    canvasRef,

    // Controls
    startCamera,
    stopCamera,
    toggleCamera,
    switchCamera,
    captureFrame,
    takeSnapshot,
    getDevices,
  };
};

export default useCamera;
