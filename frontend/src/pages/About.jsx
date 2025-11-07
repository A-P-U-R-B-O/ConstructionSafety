/**
 * About Page Component
 * Information about the application
 * 
 * Author: A-P-U-R-B-O
 * Created: 2025-11-07 16:38:03 UTC
 * Version: 1.0.0
 */

import { Shield, Github, Heart, Zap, Eye, Camera } from 'lucide-react';

const About = () => {
  return (
    <div className="space-y-6">
      <div className="card">
        <div className="card-body text-center py-12">
          <div className="flex items-center justify-center w-20 h-20 bg-gradient-to-br from-red-500 to-red-600 rounded-2xl mx-auto mb-6 shadow-xl">
            <Shield className="w-12 h-12 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
            Construction Safety AI
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-400 mb-4">
            Real-time AI-powered safety monitoring system
          </p>
          <div className="flex items-center justify-center gap-2 text-gray-500 dark:text-gray-400">
            <span>Version 1.0.0</span>
            <span>•</span>
            <span>Built with ❤️ by A-P-U-R-B-O</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card">
          <div className="card-body text-center">
            <div className="flex items-center justify-center w-12 h-12 bg-blue-100 dark:bg-blue-900 rounded-lg mx-auto mb-4">
              <Eye className="w-6 h-6 text-blue-600 dark:text-blue-400" />
            </div>
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
              YOLOv8 Detection
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              State-of-the-art object detection using Ultralytics YOLOv8
            </p>
          </div>
        </div>

        <div className="card">
          <div className="card-body text-center">
            <div className="flex items-center justify-center w-12 h-12 bg-red-100 dark:bg-red-900 rounded-lg mx-auto mb-4">
              <Camera className="w-6 h-6 text-red-600 dark:text-red-400" />
            </div>
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
              Real-time Monitoring
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Live camera feed processing with WebSocket support
            </p>
          </div>
        </div>

        <div className="card">
          <div className="card-body text-center">
            <div className="flex items-center justify-center w-12 h-12 bg-green-100 dark:bg-green-900 rounded-lg mx-auto mb-4">
              <Zap className="w-6 h-6 text-green-600 dark:text-green-400" />
            </div>
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
              Fast Processing
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Optimized for real-time performance with sub-50ms latency
            </p>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            Technology Stack
          </h2>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { name: 'React', color: 'bg-blue-500' },
              { name: 'FastAPI', color: 'bg-green-500' },
              { name: 'YOLOv8', color: 'bg-red-500' },
              { name: 'Tailwind CSS', color: 'bg-cyan-500' },
              { name: 'WebSocket', color: 'bg-purple-500' },
              { name: 'OpenCV', color: 'bg-yellow-500' },
              { name: 'Docker', color: 'bg-blue-600' },
              { name: 'Vite', color: 'bg-purple-600' },
            ].map((tech, idx) => (
              <div
                key={idx}
                className="flex items-center gap-2 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg"
              >
                <div className={`w-3 h-3 rounded-full ${tech.color}`}></div>
                <span className="text-sm font-medium text-gray-900 dark:text-white">
                  {tech.name}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            Features
          </h2>
        </div>
        <div className="card-body">
          <ul className="space-y-2 text-gray-600 dark:text-gray-400">
            <li className="flex items-center gap-2">
              <span className="text-green-500">✓</span>
              <span>Real-time object detection from camera feeds</span>
            </li>
            <li className="flex items-center gap-2">
              <span className="text-green-500">✓</span>
              <span>Image upload and batch processing</span>
            </li>
            <li className="flex items-center gap-2">
              <span className="text-green-500">✓</span>
              <span>Video file processing with annotation</span>
            </li>
            <li className="flex items-center gap-2">
              <span className="text-green-500">✓</span>
              <span>Intelligent alert system with severity levels</span>
            </li>
            <li className="flex items-center gap-2">
              <span className="text-green-500">✓</span>
              <span>Comprehensive analytics and reporting</span>
            </li>
            <li className="flex items-center gap-2">
              <span className="text-green-500">✓</span>
              <span>Dark mode support</span>
            </li>
            <li className="flex items-center gap-2">
              <span className="text-green-500">✓</span>
              <span>PWA support for mobile installation</span>
            </li>
          </ul>
        </div>
      </div>

      <div className="card bg-gradient-to-br from-red-500 to-red-600 text-white">
        <div className="card-body text-center">
          <Heart className="w-12 h-12 mx-auto mb-4" />
          <h3 className="text-xl font-bold mb-2">Open Source</h3>
          <p className="mb-4 opacity-90">
            This project is open source and available on GitHub
          </p>
          <a
            href="https://github.com/A-P-U-R-B-O/construction-safety-ai"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 bg-white text-red-600 px-6 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
          >
            <Github className="w-5 h-5" />
            <span>View on GitHub</span>
          </a>
        </div>
      </div>

      <div className="text-center text-sm text-gray-500 dark:text-gray-400">
        <p>© 2025 Construction Safety AI. All rights reserved.</p>
        <p className="mt-1">
          Developed by <span className="font-semibold text-red-600 dark:text-red-400">A-P-U-R-B-O</span>
        </p>
        <p className="mt-1">Created: 2025-11-07 16:38:03 UTC</p>
      </div>
    </div>
  );
};

export default About;
