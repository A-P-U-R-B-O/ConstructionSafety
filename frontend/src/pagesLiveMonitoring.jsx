/**
 * Live Monitoring Page Component
 * Real-time camera feed with live detection and alerts
 * 
 * Author: A-P-U-R-B-O
 * Created: 2025-11-07 16:13:32 UTC
 * Version: 1.0.0
 */

import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Camera,
  Video,
  Square,
  Play,
  Pause,
  RefreshCw,
  Settings,
  Maximize,
  Minimize,
  AlertTriangle,
  CheckCircle,
  Eye,
  Activity,
  Zap,
  Users,
  Shield,
  TrendingUp,
  Download,
  Radio
} from 'lucide-react';
import toast from 'react-hot-toast';
import { useApp } from '@/contexts/AppContext';

const LiveMonitoring = () => {
  const { updateStats, addNotification } = useApp();
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [selectedCamera, setSelectedCamera] = useState('camera-1');
  const [fps, setFps] = useState(0);
  const [detectionCount, setDetectionCount] = useState(0);
  const [alertCount, setAlertCount] = useState(0);
  
  const [liveDetections, setLiveDetections] = useState([]);
  const [liveAlerts, setLiveAlerts] = useState([]);
  
  const [stats, setStats] = useState({
    totalObjects: 0,
    dangerousObjects: 0,
    peopleCount: 0,
    vehicleCount: 0,
    processingTime: 0,
  });

  const [settings, setSettings] = useState({
    confidenceThreshold: 0.5,
    showBoundingBoxes: true,
    showLabels: true,
    alertSound: true,
    autoSave: false,
  });

  const cameras = [
    { id: 'camera-1', name: 'Main Entrance', status: 'online' },
    { id: 'camera-2', name: 'Zone A - North', status: 'online' },
    { id: 'camera-3', name: 'Zone B - East', status: 'online' },
    { id: 'camera-4', name: 'Zone C - South', status: 'offline' },
  ];

  // Simulate live monitoring
  useEffect(() => {
    if (isMonitoring && !isPaused) {
      const interval = setInterval(() => {
        // Simulate detections
        const newDetection = {
          id: Date.now(),
          class: ['person', 'truck', 'car', 'cone'][Math.floor(Math.random() * 4)],
          confidence: (0.7 + Math.random() * 0.3).toFixed(2),
          dangerLevel: ['safe', 'low', 'medium', 'high', 'critical'][Math.floor(Math.random() * 5)],
          timestamp: new Date().toLocaleTimeString(),
        };

        setLiveDetections(prev => [newDetection, ...prev].slice(0, 10));
        setDetectionCount(prev => prev + 1);
        setFps(10 + Math.floor(Math.random() * 5));

        // Simulate alerts (20% chance)
        if (Math.random() < 0.2) {
          const newAlert = {
            id: Date.now(),
            severity: ['low', 'medium', 'high', 'critical'][Math.floor(Math.random() * 4)],
            message: [
              'Worker detected near machinery',
              'PPE verification required',
              'Vehicle approaching restricted area',
              'Multiple hazards detected'
            ][Math.floor(Math.random() * 4)],
            timestamp: new Date().toLocaleTimeString(),
          };

          setLiveAlerts(prev => [newAlert, ...prev].slice(0, 5));
          setAlertCount(prev => prev + 1);

          // Show toast for critical alerts
          if (newAlert.severity === 'critical') {
            toast.error(newAlert.message, {
              icon: '🚨',
              duration: 5000,
            });
          }

          addNotification({
            title: `${newAlert.severity.toUpperCase()} Alert`,
            message: newAlert.message,
          });
        }

        // Update stats
        setStats({
          totalObjects: Math.floor(Math.random() * 10),
          dangerousObjects: Math.floor(Math.random() * 3),
          peopleCount: Math.floor(Math.random() * 5),
          vehicleCount: Math.floor(Math.random() * 3),
          processingTime: 35 + Math.floor(Math.random() * 20),
        });
      }, 1000);

      return () => clearInterval(interval);
    }
  }, [isMonitoring, isPaused]);

  const handleStartMonitoring = async () => {
    try {
      // Request camera access
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 1280, height: 720 } 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      setIsMonitoring(true);
      updateStats({ activeMonitoring: true });
      
      toast.success('Live monitoring started!', {
        icon: '📹',
      });
    } catch (error) {
      console.error('Camera access error:', error);
      toast.error('Could not access camera. Using demo mode.', {
        icon: '⚠️',
      });
      
      // Start in demo mode anyway
      setIsMonitoring(true);
      updateStats({ activeMonitoring: true });
    }
  };

  const handleStopMonitoring = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
    }

    setIsMonitoring(false);
    setIsPaused(false);
    updateStats({ activeMonitoring: false });
    
    toast.success('Live monitoring stopped', {
      icon: '⏹️',
    });
  };

  const handleTogglePause = () => {
    setIsPaused(!isPaused);
    toast.success(isPaused ? 'Monitoring resumed' : 'Monitoring paused', {
      icon: isPaused ? '▶️' : '⏸️',
    });
  };

  const handleToggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  const handleCameraChange = (cameraId) => {
    setSelectedCamera(cameraId);
    toast.success(`Switched to ${cameras.find(c => c.id === cameraId)?.name}`, {
      icon: '📹',
    });
  };

  const handleSnapshot = () => {
    toast.success('Snapshot saved!', {
      icon: '📸',
    });
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
              Live Monitoring
            </h1>
            {isMonitoring && (
              <span className="flex items-center gap-2 px-3 py-1 bg-red-100 dark:bg-red-900 rounded-full">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500"></span>
                </span>
                <span className="text-sm font-semibold text-red-800 dark:text-red-200">
                  LIVE
                </span>
              </span>
            )}
          </div>
          <p className="mt-1 text-gray-600 dark:text-gray-400">
            Real-time safety monitoring and detection
          </p>
        </div>

        <div className="flex items-center gap-3">
          {!isMonitoring ? (
            <button
              onClick={handleStartMonitoring}
              className="btn btn-primary flex items-center gap-2"
            >
              <Play className="w-4 h-4" />
              <span>Start Monitoring</span>
            </button>
          ) : (
            <>
              <button
                onClick={handleTogglePause}
                className="btn btn-secondary flex items-center gap-2"
              >
                {isPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
                <span>{isPaused ? 'Resume' : 'Pause'}</span>
              </button>
              <button
                onClick={handleStopMonitoring}
                className="btn btn-danger flex items-center gap-2"
              >
                <Square className="w-4 h-4" />
                <span>Stop</span>
              </button>
            </>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Video Feed */}
        <div className="lg:col-span-2 space-y-4">
          
          {/* Video Container */}
          <div className="card overflow-hidden">
            <div className="relative bg-gray-900 aspect-video">
              {/* Video Element */}
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
              />
              
              {/* Canvas Overlay for Detections */}
              <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full pointer-events-none"
              />

              {/* Demo Placeholder */}
              {!isMonitoring && (
                <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
                  <div className="text-center text-white">
                    <Camera className="w-16 h-16 mx-auto mb-4 opacity-50" />
                    <p className="text-lg font-medium mb-2">No Active Monitoring</p>
                    <p className="text-sm text-gray-400">Click "Start Monitoring" to begin</p>
                  </div>
                </div>
              )}

              {/* Status Overlay */}
              {isMonitoring && (
                <div className="absolute top-4 left-4 right-4 flex items-start justify-between">
                  {/* FPS Counter */}
                  <div className="glass px-3 py-2 rounded-lg">
                    <div className="flex items-center gap-2 text-white text-sm font-medium">
                      <Activity className="w-4 h-4" />
                      <span>{fps} FPS</span>
                    </div>
                  </div>

                  {/* Recording Indicator */}
                  <div className="glass px-3 py-2 rounded-lg">
                    <div className="flex items-center gap-2 text-white text-sm font-medium">
                      <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                      <span>Recording</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Bottom Controls */}
              {isMonitoring && (
                <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between">
                  {/* Camera Selector */}
                  <select
                    value={selectedCamera}
                    onChange={(e) => handleCameraChange(e.target.value)}
                    className="glass px-3 py-2 rounded-lg text-white text-sm font-medium bg-opacity-50 border-0 focus:ring-2 focus:ring-white"
                  >
                    {cameras.map(camera => (
                      <option key={camera.id} value={camera.id} className="bg-gray-800">
                        {camera.name} {camera.status === 'offline' ? '(Offline)' : ''}
                      </option>
                    ))}
                  </select>

                  {/* Action Buttons */}
                  <div className="flex items-center gap-2">
                    <button
                      onClick={handleSnapshot}
                      className="glass p-2 rounded-lg hover:bg-white hover:bg-opacity-20 transition-colors"
                      title="Take Snapshot"
                    >
                      <Download className="w-5 h-5 text-white" />
                    </button>
                    <button
                      onClick={handleToggleFullscreen}
                      className="glass p-2 rounded-lg hover:bg-white hover:bg-opacity-20 transition-colors"
                      title="Toggle Fullscreen"
                    >
                      {isFullscreen ? (
                        <Minimize className="w-5 h-5 text-white" />
                      ) : (
                        <Maximize className="w-5 h-5 text-white" />
                      )}
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Real-time Stats */}
          {isMonitoring && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <LiveStatCard
                icon={Eye}
                label="Total Objects"
                value={stats.totalObjects}
                color="blue"
              />
              <LiveStatCard
                icon={AlertTriangle}
                label="Dangerous"
                value={stats.dangerousObjects}
                color="red"
              />
              <LiveStatCard
                icon={Users}
                label="People"
                value={stats.peopleCount}
                color="green"
              />
              <LiveStatCard
                icon={Shield}
                label="Vehicles"
                value={stats.vehicleCount}
                color="purple"
              />
            </div>
          )}
        </div>

        {/* Sidebar - Detections & Alerts */}
        <div className="space-y-6">
          
          {/* Live Detections */}
          <div className="card">
            <div className="card-header">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                Live Detections
              </h3>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Total: {detectionCount}
              </p>
            </div>
            <div className="card-body p-0">
              <div className="max-h-64 overflow-y-auto scrollbar-thin">
                {liveDetections.length === 0 ? (
                  <div className="px-6 py-8 text-center text-gray-500 dark:text-gray-400">
                    <Eye className="w-12 h-12 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">No detections yet</p>
                  </div>
                ) : (
                  <div className="divide-y divide-gray-200 dark:divide-gray-700">
                    {liveDetections.map((detection) => (
                      <DetectionItem key={detection.id} detection={detection} />
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Live Alerts */}
          <div className="card">
            <div className="card-header">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                Live Alerts
              </h3>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Total: {alertCount}
              </p>
            </div>
            <div className="card-body p-0">
              <div className="max-h-64 overflow-y-auto scrollbar-thin">
                {liveAlerts.length === 0 ? (
                  <div className="px-6 py-8 text-center text-gray-500 dark:text-gray-400">
                    <AlertTriangle className="w-12 h-12 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">No alerts</p>
                  </div>
                ) : (
                  <div className="divide-y divide-gray-200 dark:divide-gray-700">
                    {liveAlerts.map((alert) => (
                      <AlertItem key={alert.id} alert={alert} />
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Processing Info */}
          {isMonitoring && (
            <div className="card bg-gradient-to-br from-blue-500 to-blue-600 text-white">
              <div className="card-body">
                <div className="flex items-center gap-3 mb-4">
                  <Zap className="w-6 h-6" />
                  <h3 className="text-lg font-semibold">Processing</h3>
                </div>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-blue-100">Model</span>
                    <span className="font-semibold">YOLOv8n</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-blue-100">Device</span>
                    <span className="font-semibold">CPU</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-blue-100">Latency</span>
                    <span className="font-semibold">{stats.processingTime}ms</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-blue-100">Status</span>
                    <span className="flex items-center gap-1.5">
                      <CheckCircle className="w-4 h-4" />
                      <span className="font-semibold">Active</span>
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// Live Stat Card Component
// ============================================================================

const LiveStatCard = ({ icon: Icon, label, value, color }) => {
  const colorClasses = {
    blue: 'from-blue-500 to-blue-600',
    red: 'from-red-500 to-red-600',
    green: 'from-green-500 to-green-600',
    purple: 'from-purple-500 to-purple-600',
  };

  return (
    <div className="card">
      <div className="card-body">
        <div className="flex items-center gap-3">
          <div className={`p-2 bg-gradient-to-br ${colorClasses[color]} rounded-lg`}>
            <Icon className="w-5 h-5 text-white" />
          </div>
          <div>
            <p className="text-xs text-gray-600 dark:text-gray-400">{label}</p>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">{value}</p>
          </div>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// Detection Item Component
// ============================================================================

const DetectionItem = ({ detection }) => {
  const dangerColors = {
    safe: 'text-gray-600 bg-gray-100',
    low: 'text-blue-600 bg-blue-100',
    medium: 'text-yellow-600 bg-yellow-100',
    high: 'text-orange-600 bg-orange-100',
    critical: 'text-red-600 bg-red-100',
  };

  return (
    <div className="px-4 py-3 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors">
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-900 dark:text-white capitalize">
            {detection.class}
          </p>
          <p className="text-xs text-gray-500 dark:text-gray-400">
            {detection.confidence} confidence • {detection.timestamp}
          </p>
        </div>
        <span className={`px-2 py-1 rounded text-xs font-medium ${dangerColors[detection.dangerLevel]}`}>
          {detection.dangerLevel}
        </span>
      </div>
    </div>
  );
};

// ============================================================================
// Alert Item Component
// ============================================================================

const AlertItem = ({ alert }) => {
  const severityColors = {
    low: 'border-l-blue-500 bg-blue-50 dark:bg-blue-900 dark:bg-opacity-20',
    medium: 'border-l-yellow-500 bg-yellow-50 dark:bg-yellow-900 dark:bg-opacity-20',
    high: 'border-l-orange-500 bg-orange-50 dark:bg-orange-900 dark:bg-opacity-20',
    critical: 'border-l-red-500 bg-red-50 dark:bg-red-900 dark:bg-opacity-20',
  };

  return (
    <div className={`px-4 py-3 bor
