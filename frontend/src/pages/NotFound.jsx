/**
 * 404 Not Found Page Component
 * 
 * Author: A-P-U-R-B-O
 * Created: 2025-11-07 16:38:03 UTC
 * Version: 1.0.0
 */

import { Link } from 'react-router-dom';
import { Home, Search, ArrowLeft } from 'lucide-react';

const NotFound = () => {
  return (
    <div className="min-h-[70vh] flex items-center justify-center">
      <div className="text-center">
        <div className="mb-8">
          <h1 className="text-9xl font-bold text-red-600 dark:text-red-400">
            404
          </h1>
          <div className="text-6xl mb-4">🚧</div>
        </div>
        
        <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
          Page Not Found
        </h2>
        
        <p className="text-lg text-gray-600 dark:text-gray-400 mb-8 max-w-md mx-auto">
          Oops! The page you're looking for doesn't exist. It might have been moved or deleted.
        </p>
        
        <div className="flex items-center justify-center gap-4">
          <Link to="/" className="btn btn-primary flex items-center gap-2">
            <Home className="w-4 h-4" />
            <span>Go Home</span>
          </Link>
          
          <button
            onClick={() => window.history.back()}
            className="btn btn-secondary flex items-center gap-2"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Go Back</span>
          </button>
        </div>
        
        <div className="mt-12 text-sm text-gray-500 dark:text-gray-400">
          <p>If you believe this is an error, please contact support.</p>
          <p className="mt-2">Current User: <span className="font-semibold text-red-600 dark:text-red-400">A-P-U-R-B-O</span></p>
        </div>
      </div>
    </div>
  );
};

export default NotFound;
