/**
 * Dashboard Page Component
 * Main overview dashboard with statistics, recent activity, and system status
 * 
 * Author: A-P-U-R-B-O
 * Created: 2025-11-07 16:13:32 UTC
 * Version: 1.0.0
 */

import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Activity,
  AlertTriangle,
  Camera,
  TrendingUp,
  Users,
  Shield,
  Clock,
  CheckCircle,
  XCircle,
  Video,
  Image as ImageIcon,
  BarChart3,
  ChevronRight,
  RefreshCw,
  Download,
  Eye,
  Zap
} from 'lucide-react';
import { useApp } from '@/contexts/AppContext';
import toast from 'react-hot-toast';

const Dashboard = () => {
  const { stats, updateStats } = useApp();
  const [loading, setLoading] = useState(true);
  const [dashboardData, setDashboardData] = useState({
    totalDetections: 1247,
    totalAlerts: 23,
    criticalAlerts: 5,
    activeMonitoring: 3,
    systemUptime: '99.8%',
    avgResponseTime: '45ms',
    detectionsToday: 156,
    alertsToday: 8,
  });

  const [recentActivity, setRecentActivity] = useState([
    {
      id: 1,
      type: 'detection',
      severity: 'high',
      message: 'Worker detected near heavy machinery',
      location: 'Zone A - North',
      timestamp: '2025-11-07 16:10:15',
      icon: AlertTriangle,
      color: 'text-red-600',
      bg: 'bg-red-50',
    },
    {
      id: 2,
      type: 'detection',
      severity: 'medium',
      message: 'PPE verification required',
      location: 'Zone B - East',
      timestamp: '2025-11-07 16:08:42',
      icon: Shield,
      color: 'text-yellow-600',
      bg: 'bg-yellow-50',
    },
    {
      id: 3,
      type: 'success',
      severity: 'low',
      message: 'Safety protocol compliance verified',
      location: 'Zone C - South',
      timestamp: '2025-11-07 16:05:20',
      icon: CheckCircle,
      color: 'text-green-600',
      bg: 'bg-green-50',
    },
    {
      id: 4,
      type: 'detection',
      severity: 'high',
      message: 'Vehicle approaching restricted area',
      location: 'Zone D - West',
      timestamp: '2025-11-07 16:02:38',
      icon: AlertTriangle,
      color: 'text-red-600',
      bg: 'bg-red-50',
    },
    {
      id: 5,
      type: 'info',
      severity: 'low',
      message: 'Scheduled maintenance completed',
      location: 'System',
      timestamp: '2025-11-07 15:58:11',
      icon: Activity,
      color: 'text-blue-600',
      bg: 'bg-blue-50',
    },
  ]);

  useEffect(() => {
    // Simulate data loading
    const timer = setTimeout(() => {
      setLoading(false);
      updateStats({
        totalDetections: dashboardData.totalDetections,
        totalAlerts: dashboardData.totalAlerts,
      });
    }, 800);

    return () => clearTimeout(timer);
  }, []);

  const handleRefresh = async () => {
    toast.promise(
      new Promise((resolve) => setTimeout(resolve, 1000)),
      {
        loading: 'Refreshing dashboard data...',
        success: 'Dashboard updated successfully!',
        error: 'Failed to refresh data',
      }
    );
  };

  const handleExport = () => {
    toast.success('Exporting dashboard report...', {
      icon: '📊',
    });
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-red-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4"
      >
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Dashboard
          </h1>
          <p className="mt-1 text-gray-600 dark:text-gray-400">
            Welcome back, <span className="font-semibold text-red-600 dark:text-red-400">A-P-U-R-B-O</span>
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-500">
            Last updated: {new Date().toLocaleString()}
          </p>
        </div>
        
        <div className="flex items-center gap-3">
          <button
            onClick={handleRefresh}
            className="btn btn-secondary flex items-center gap-2"
          >
            <RefreshCw className="w-4 h-4" />
            <span>Refresh</span>
          </button>
          <button
            onClick={handleExport}
            className="btn btn-primary flex items-center gap-2"
          >
            <Download className="w-4 h-4" />
            <span>Export</span>
          </button>
        </div>
      </motion.div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          title="Total Detections"
          value={dashboardData.totalDetections}
          icon={Eye}
          color="blue"
          trend="+12.5%"
          trendUp={true}
        />
        <StatsCard
          title="Active Alerts"
          value={dashboardData.totalAlerts}
          icon={AlertTriangle}
          color="red"
          trend="+3"
          trendUp={true}
          critical={dashboardData.criticalAlerts}
        />
        <StatsCard
          title="Live Monitoring"
          value={dashboardData.activeMonitoring}
          icon={Video}
          color="green"
          trend="3 active"
          badge="LIVE"
        />
        <StatsCard
          title="System Uptime"
          value={dashboardData.systemUptime}
          icon={Zap}
          color="purple"
          trend="Last 30 days"
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Recent Activity */}
        <div className="lg:col-span-2">
          <div className="card">
            <div className="card-header flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Recent Activity
                </h2>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Latest detections and alerts
                </p>
              </div>
              <Link
                to="/alerts"
                className="text-sm text-red-600 dark:text-red-400 hover:underline flex items-center gap-1"
              >
                View all
                <ChevronRight className="w-4 h-4" />
              </Link>
            </div>
            <div className="card-body p-0">
              <div className="divide-y divide-gray-200 dark:divide-gray-700">
                {recentActivity.map((activity, index) => (
                  <ActivityItem key={activity.id} activity={activity} index={index} />
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Quick Actions & System Status */}
        <div className="space-y-6">
          
          {/* Quick Actions */}
          <div className="card">
            <div className="card-header">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                Quick Actions
              </h2>
            </div>
            <div className="card-body space-y-3">
              <Link
                to="/live"
                className="flex items-center gap-3 p-3 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors group"
              >
                <div className="flex items-center justify-center w-10 h-10 bg-red-100 dark:bg-red-900 rounded-lg group-hover:bg-red-200 dark:group-hover:bg-red-800 transition-colors">
                  <Camera className="w-5 h-5 text-red-600 dark:text-red-400" />
                </div>
                <div className="flex-1">
                  <p className="font-medium text-gray-900 dark:text-white">
                    Start Live Monitoring
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    Real-time camera feed
                  </p>
                </div>
                <ChevronRight className="w-5 h-5 text-gray-400" />
              </Link>

              <Link
                to="/image"
                className="flex items-center gap-3 p-3 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors group"
              >
                <div className="flex items-center justify-center w-10 h-10 bg-blue-100 dark:bg-blue-900 rounded-lg group-hover:bg-blue-200 dark:group-hover:bg-blue-800 transition-colors">
                  <ImageIcon className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                </div>
                <div className="flex-1">
                  <p className="font-medium text-gray-900 dark:text-white">
                    Upload Image
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    Detect from image
                  </p>
                </div>
                <ChevronRight className="w-5 h-5 text-gray-400" />
              </Link>

              <Link
                to="/video"
                className="flex items-center gap-3 p-3 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors group"
              >
                <div className="flex items-center justify-center w-10 h-10 bg-purple-100 dark:bg-purple-900 rounded-lg group-hover:bg-purple-200 dark:group-hover:bg-purple-800 transition-colors">
                  <Video className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                </div>
                <div className="flex-1">
                  <p className="font-medium text-gray-900 dark:text-white">
                    Process Video
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    Batch processing
                  </p>
                </div>
                <ChevronRight className="w-5 h-5 text-gray-400" />
              </Link>

              <Link
                to="/analytics"
                className="flex items-center gap-3 p-3 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors group"
              >
                <div className="flex items-center justify-center w-10 h-10 bg-green-100 dark:bg-green-900 rounded-lg group-hover:bg-green-200 dark:group-hover:bg-green-800 transition-colors">
                  <BarChart3 className="w-5 h-5 text-green-600 dark:text-green-400" />
                </div>
                <div className="flex-1">
                  <p className="font-medium text-gray-900 dark:text-white">
                    View Analytics
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    Insights & reports
                  </p>
                </div>
                <ChevronRight className="w-5 h-5 text-gray-400" />
              </Link>
            </div>
          </div>

          {/* System Status */}
          <div className="card">
            <div className="card-header">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                System Status
              </h2>
            </div>
            <div className="card-body space-y-4">
              <StatusItem
                label="API Server"
                status="operational"
                value="Online"
              />
              <StatusItem
                label="Detection Model"
                status="operational"
                value="YOLOv8n"
              />
              <StatusItem
                label="WebSocket"
                status="operational"
                value="Connected"
              />
              <StatusItem
                label="Response Time"
                status="operational"
                value={dashboardData.avgResponseTime}
              />
            </div>
          </div>

          {/* Today's Summary */}
          <div className="card bg-gradient-to-br from-red-500 to-red-600 text-white">
            <div className="card-body">
              <h3 className="text-lg font-semibold mb-4">Today's Summary</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-red-100">Detections</span>
                  <span className="text-2xl font-bold">{dashboardData.detectionsToday}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-red-100">Alerts</span>
                  <span className="text-2xl font-bold">{dashboardData.alertsToday}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-red-100">Critical</span>
                  <span className="text-2xl font-bold">{dashboardData.criticalAlerts}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// Stats Card Component
// ============================================================================

const StatsCard = ({ title, value, icon: Icon, color, trend, trendUp, critical, badge }) => {
  const colorClasses = {
    blue: 'from-blue-500 to-blue-600',
    red: 'from-red-500 to-red-600',
    green: 'from-green-500 to-green-600',
    purple: 'from-purple-500 to-purple-600',
    yellow: 'from-yellow-500 to-yellow-600',
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="card overflow-hidden"
    >
      <div className="card-body">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">
              {title}
            </p>
            <div className="flex items-baseline gap-2">
              <h3 className="text-3xl font-bold text-gray-900 dark:text-white">
                {value}
              </h3>
              {critical && (
                <span className="text-sm text-red-600 dark:text-red-400 font-medium">
                  ({critical} critical)
                </span>
              )}
            </div>
            {trend && (
              <div className="flex items-center gap-1 mt-2">
                <TrendingUp className={`w-4 h-4 ${trendUp ? 'text-green-600' : 'text-red-600'}`} />
                <span className={`text-sm font-medium ${trendUp ? 'text-green-600' : 'text-red-600'}`}>
                  {trend}
                </span>
              </div>
            )}
          </div>
          <div className={`p-3 bg-gradient-to-br ${colorClasses[color]} rounded-lg`}>
            <Icon className="w-6 h-6 text-white" />
          </div>
        </div>
        {badge && (
          <div className="mt-3">
            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-semibold bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 animate-pulse">
              {badge}
            </span>
          </div>
        )}
      </div>
    </motion.div>
  );
};

// ============================================================================
// Activity Item Component
// ============================================================================

const ActivityItem = ({ activity, index }) => {
  const Icon = activity.icon;

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.05 }}
      className="px-6 py-4 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
    >
      <div className="flex items-start gap-4">
        <div className={`flex-shrink-0 w-10 h-10 ${activity.bg} dark:bg-opacity-20 rounded-lg flex items-center justify-center`}>
          <Icon className={`w-5 h-5 ${activity.color}`} />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-gray-900 dark:text-white">
            {activity.message}
          </p>
          <div className="flex items-center gap-3 mt-1">
            <span className="text-xs text-gray-500 dark:text-gray-400">
              {activity.location}
            </span>
            <span className="text-xs text-gray-400 dark:text-gray-500">
              •
            </span>
            <span className="text-xs text-gray-500 dark:text-gray-400 flex items-center gap-1">
              <Clock className="w-3 h-3" />
              {activity.timestamp}
            </span>
          </div>
        </div>
        <span className={`
          flex-shrink-0 px-2 py-1 rounded-full text-xs font-medium
          ${activity.severity === 'high' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' : ''}
          ${activity.severity === 'medium' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' : ''}
          ${activity.severity === 'low' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' : ''}
        `}>
          {activity.severity}
        </span>
      </div>
    </motion.div>
  );
};

// ============================================================================
// Status Item Component
// ============================================================================

const StatusItem = ({ label, status, value }) => {
  const statusColors = {
    operational: 'bg-green-500',
    warning: 'bg-yellow-500',
    error: 'bg-red-500',
  };

  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-3">
        <div className={`w-2 h-2 ${statusColors[status]} rounded-full`}></div>
        <span className="text-sm text-gray-600 dark:text-gray-400">{label}</span>
      </div>
      <span className="text-sm font-medium text-gray-900 dark:text-white">
        {value}
      </span>
    </div>
  );
};

export default Dashboard;
