/**
 * Footer Component
 * Application footer with metadata and links
 * 
 * Author: A-P-U-R-B-O
 * Created: 2025-11-07 16:03:18 UTC
 * Version: 1.0.0
 */

import { Link } from 'react-router-dom';
import { Heart, Github, Shield } from 'lucide-react';

const Footer = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="py-6">
          {/* Main Footer Content */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-6">
            
            {/* Brand Section */}
            <div className="flex flex-col space-y-3">
              <div className="flex items-center gap-2">
                <div className="flex items-center justify-center w-8 h-8 bg-gradient-to-br from-red-500 to-red-600 rounded-lg">
                  <Shield className="w-5 h-5 text-white" />
                </div>
                <span className="font-bold text-gray-900 dark:text-white">
                  Construction Safety AI
                </span>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                AI-powered construction site safety monitoring using YOLOv8 computer vision.
              </p>
              <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                <span>Made with</span>
                <Heart className="w-4 h-4 text-red-500 fill-current" />
                <span>by A-P-U-R-B-O</span>
              </div>
            </div>

            {/* Quick Links */}
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
                Quick Links
              </h3>
              <ul className="space-y-2 text-sm">
                <li>
                  <Link 
                    to="/" 
                    className="text-gray-600 dark:text-gray-400 hover:text-red-600 dark:hover:text-red-400 transition-colors"
                  >
                    Dashboard
                  </Link>
                </li>
                <li>
                  <Link 
                    to="/live" 
                    className="text-gray-600 dark:text-gray-400 hover:text-red-600 dark:hover:text-red-400 transition-colors"
                  >
                    Live Monitoring
                  </Link>
                </li>
                <li>
                  <Link 
                    to="/analytics" 
                    className="text-gray-600 dark:text-gray-400 hover:text-red-600 dark:hover:text-red-400 transition-colors"
                  >
                    Analytics
                  </Link>
                </li>
                <li>
                  <Link 
                    to="/about" 
                    className="text-gray-600 dark:text-gray-400 hover:text-red-600 dark:hover:text-red-400 transition-colors"
                  >
                    About
                  </Link>
                </li>
              </ul>
            </div>

            {/* System Info */}
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
                System Info
              </h3>
              <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                <li className="flex items-center justify-between">
                  <span>Version:</span>
                  <span className="font-mono font-medium">1.0.0</span>
                </li>
                <li className="flex items-center justify-between">
                  <span>Build Date:</span>
                  <span className="font-mono font-medium">2025-11-07</span>
                </li>
                <li className="flex items-center justify-between">
                  <span>Status:</span>
                  <span className="flex items-center gap-1.5">
                    <span className="w-2 h-2 bg-green-500 rounded-full"></span>
                    <span className="font-medium text-green-600 dark:text-green-400">
                      Operational
                    </span>
                  </span>
                </li>
                <li className="flex items-center justify-between">
                  <span>Session:</span>
                  <span className="font-mono font-medium text-xs">
                    {new Date().toLocaleDateString()}
                  </span>
                </li>
              </ul>
            </div>
          </div>

          {/* Bottom Bar */}
          <div className="pt-6 border-t border-gray-200 dark:border-gray-700">
            <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
              
              {/* Copyright */}
              <div className="text-sm text-gray-600 dark:text-gray-400 text-center sm:text-left">
                <p>
                  © {currentYear} Construction Safety AI. All rights reserved.
                </p>
                <p className="mt-1">
                  Developed by{' '}
                  <span className="font-semibold text-red-600 dark:text-red-400">
                    A-P-U-R-B-O
                  </span>
                </p>
              </div>

              {/* Social Links */}
              <div className="flex items-center gap-4">
                <a
                  href="https://github.com/A-P-U-R-B-O"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-gray-600 dark:text-gray-400 hover:text-red-600 dark:hover:text-red-400 transition-colors"
                  aria-label="GitHub"
                >
                  <Github className="w-5 h-5" />
                </a>
                
                {/* Additional links can be added here */}
              </div>
            </div>
          </div>

          {/* Tech Stack Badge */}
          <div className="mt-4 pt-4 border-t border-gray-100 dark:border-gray-800">
            <div className="flex flex-wrap items-center justify-center gap-2 text-xs text-gray-500 dark:text-gray-500">
              <span>Built with:</span>
              <span className="px-2 py-0.5 bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 rounded">
                React
              </span>
              <span className="px-2 py-0.5 bg-purple-100 dark:bg-purple-900 text-purple-700 dark:text-purple-300 rounded">
                Vite
              </span>
              <span className="px-2 py-0.5 bg-cyan-100 dark:bg-cyan-900 text-cyan-700 dark:text-cyan-300 rounded">
                Tailwind CSS
              </span>
              <span className="px-2 py-0.5 bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300 rounded">
                FastAPI
              </span>
              <span className="px-2 py-0.5 bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300 rounded">
                YOLOv8
              </span>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
