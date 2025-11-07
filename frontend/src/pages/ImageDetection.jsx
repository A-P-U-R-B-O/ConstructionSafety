/**
 * Image Detection Page Component
 * Upload and detect objects from images
 * 
 * Author: A-P-U-R-B-O
 * Created: 2025-11-07 16:38:03 UTC
 * Version: 1.0.0
 */

import { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import {
  Upload,
  Image as ImageIcon,
  X,
  Download,
  RefreshCw,
  AlertTriangle,
  CheckCircle,
  Eye,
  Zap,
} from 'lucide-react';
import toast from 'react-hot-toast';
import useDetection from '@/hooks/useDetection';

const ImageDetection = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const fileInputRef = useRef(null);

  const {
    isDetecting,
    detectionResult,
    uploadProgress,
    processingTime,
    detectFile,
    clearResult,
  } = useDetection({
    confidenceThreshold: 0.5,
    autoAlert: true,
  });

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        toast.error('Please select an image file');
        return;
      }

      setSelectedImage(file);
      
      // Create preview URL
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      
      toast.success(`Image loaded: ${file.name}`);
    }
  };

  const handleDetect = async () => {
    if (!selectedImage) {
      toast.error('Please select an image first');
      return;
    }

    try {
      await detectFile(selectedImage);
    } catch (error) {
      console.error('Detection failed:', error);
    }
  };

  const handleClear = () => {
    setSelectedImage(null);
    setPreviewUrl(null);
    clearResult();
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      toast.success(`Image loaded: ${file.name}`);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Image Detection
        </h1>
        <p className="mt-1 text-gray-600 dark:text-gray-400">
          Upload an image to detect objects and safety hazards
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Upload Section */}
        <div className="space-y-6">
          <div className="card">
            <div className="card-header">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                Upload Image
              </h2>
            </div>
            <div className="card-body">
              {!previewUrl ? (
                <div
                  onDragOver={handleDragOver}
                  onDrop={handleDrop}
                  className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-8 text-center hover:border-red-500 dark:hover:border-red-500 transition-colors cursor-pointer"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                  <p className="text-gray-600 dark:text-gray-400 mb-2">
                    Drag and drop an image here, or click to select
                  </p>
                  <p className="text-sm text-gray-500 dark:text-gray-500">
                    Supports: JPG, PNG, WEBP (Max 10MB)
                  </p>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="relative group">
                    <img
                      src={previewUrl}
                      alt="Preview"
                      className="w-full rounded-lg"
                    />
                    <button
                      onClick={handleClear}
                      className="absolute top-2 right-2 p-2 bg-red-600 text-white rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                  
                  {selectedImage && (
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      <p>File: {selectedImage.name}</p>
                      <p>Size: {(selectedImage.size / 1024).toFixed(2)} KB</p>
                    </div>
                  )}
                </div>
              )}
            </div>
            <div className="card-footer">
              <div className="flex gap-3">
                <button
                  onClick={handleDetect}
                  disabled={!selectedImage || isDetecting}
                  className="btn btn-primary flex-1"
                >
                  {isDetecting ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      <span>Detecting... {uploadProgress}%</span>
                    </>
                  ) : (
                    <>
                      <Eye className="w-4 h-4" />
                      <span>Detect Objects</span>
                    </>
                  )}
                </button>
                <button
                  onClick={handleClear}
                  disabled={!selectedImage}
                  className="btn btn-secondary"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Results Section */}
        <div className="space-y-6">
          {detectionResult ? (
            <>
              <div className="card">
                <div className="card-header">
                  <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                    Detection Results
                  </h2>
                </div>
                <div className="card-body space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center p-4 bg-blue-50 dark:bg-blue-900 dark:bg-opacity-20 rounded-lg">
                      <p className="text-sm text-gray-600 dark:text-gray-400">Total Objects</p>
                      <p className="text-2xl font-bold text-gray-900 dark:text-white">
                        {detectionResult.total_objects}
                      </p>
                    </div>
                    <div className="text-center p-4 bg-red-50 dark:bg-red-900 dark:bg-opacity-20 rounded-lg">
                      <p className="text-sm text-gray-600 dark:text-gray-400">Dangerous</p>
                      <p className="text-2xl font-bold text-red-600 dark:text-red-400">
                        {detectionResult.dangerous_objects}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">Processing Time</span>
                    <span className="font-semibold text-gray-900 dark:text-white">
                      {processingTime}ms
                    </span>
                  </div>

                  {detectionResult.annotated_image_base64 && (
                    <div>
                      <img
                        src={`data:image/jpeg;base64,${detectionResult.annotated_image_base64}`}
                        alt="Annotated"
                        className="w-full rounded-lg border border-gray-200 dark:border-gray-700"
                      />
                    </div>
                  )}
                </div>
              </div>

              {detectionResult.detections && detectionResult.detections.length > 0 && (
                <div className="card">
                  <div className="card-header">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                      Detected Objects
                    </h3>
                  </div>
                  <div className="card-body p-0">
                    <div className="max-h-64 overflow-y-auto scrollbar-thin">
                      {detectionResult.detections.map((detection, idx) => (
                        <div
                          key={idx}
                          className="px-6 py-3 border-b border-gray-200 dark:border-gray-700 last:border-0 hover:bg-gray-50 dark:hover:bg-gray-700"
                        >
                          <div className="flex items-center justify-between">
                            <div>
                              <p className="font-medium text-gray-900 dark:text-white capitalize">
                                {detection.class}
                              </p>
                              <p className="text-xs text-gray-500 dark:text-gray-400">
                                Confidence: {(detection.confidence * 100).toFixed(1)}%
                              </p>
                            </div>
                            <span className={`px-2 py-1 rounded text-xs font-medium ${
                              detection.is_dangerous
                                ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                                : 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                            }`}>
                              {detection.is_dangerous ? 'Dangerous' : 'Safe'}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="card">
              <div className="card-body">
                <div className="text-center py-12">
                  <ImageIcon className="w-16 h-16 mx-auto mb-4 text-gray-300 dark:text-gray-600" />
                  <p className="text-gray-500 dark:text-gray-400">
                    Upload an image and click "Detect Objects" to see results
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ImageDetection;
