/**
 * Analytics Page Component
 * View detection statistics and analytics
 * 
 * Author: A-P-U-R-B-O
 * Created: 2025-11-07 16:38:03 UTC
 * Version: 1.0.0
 */

import { useState } from 'react';
import { BarChart3, TrendingUp, Eye, AlertTriangle, Clock } from 'lucide-react';

const Analytics = () => {
  const [timeRange, setTimeRange] = useState('week');

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Analytics
          </h1>
          <p className="mt-1 text-gray-600 dark:text-gray-400">
            Detection statistics and insights
          </p>
        </div>
        <select
          value={timeRange}
          onChange={(e) => setTimeRange(e.target.value)}
          className="px-4 py-2"
        >
          <option value="day">Last 24 Hours</option>
          <option value="week">Last 7 Days</option>
          <option value="month">Last 30 Days</option>
          <option value="year">Last Year</option>
        </select>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="card">
          <div className="card-body">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Total Detections</p>
                <p className="text-3xl font-bold text-gray-900 dark:text-white">1,247</p>
                <p className="text-xs text-green-600 dark:text-green-400 mt-1">
                  +12.5% from last period
                </p>
              </div>
              <Eye className="w-12 h-12 text-blue-500" />
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-body">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Total Alerts</p>
                <p className="text-3xl font-bold text-gray-900 dark:text-white">234</p>
                <p className="text-xs text-red-600 dark:text-red-400 mt-1">
                  +8 new today
                </p>
              </div>
              <AlertTriangle className="w-12 h-12 text-red-500" />
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-body">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Avg. Processing</p>
                <p className="text-3xl font-bold text-gray-900 dark:text-white">45ms</p>
                <p className="text-xs text-green-600 dark:text-green-400 mt-1">
                  -5ms faster
                </p>
              </div>
              <Clock className="w-12 h-12 text-purple-500" />
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-body">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Success Rate</p>
                <p className="text-3xl font-bold text-gray-900 dark:text-white">98.5%</p>
                <p className="text-xs text-green-600 dark:text-green-400 mt-1">
                  Excellent
                </p>
              </div>
              <TrendingUp className="w-12 h-12 text-green-500" />
            </div>
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
              Detection Trends
            </h2>
          </div>
          <div className="card-body">
            <div className="h-64 flex items-center justify-center text-gray-400">
              <BarChart3 className="w-16 h-16 mb-2" />
              <p>Chart visualization coming soon</p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
              Object Distribution
            </h2>
          </div>
          <div className="card-body">
            <div className="space-y-3">
              {['Person', 'Vehicle', 'Cone', 'Equipment'].map((item, idx) => (
                <div key={idx}>
                  <div className="flex items-center justify-between text-sm mb-1">
                    <span className="text-gray-700 dark:text-gray-300">{item}</span>
                    <span className="font-semibold text-gray-900 dark:text-white">
                      {Math.floor(Math.random() * 100)}
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div
                      className="bg-blue-600 h-2 rounded-full"
                      style={{ width: `${Math.floor(Math.random() * 100)}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analytics;
