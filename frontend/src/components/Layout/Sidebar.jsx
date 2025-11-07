/**
 * Sidebar Component
 * Navigation sidebar with menu items and status indicators
 * 
 * Author: A-P-U-R-B-O
 * Created: 2025-11-07 16:03:18 UTC
 * Version: 1.0.0
 */

import { Link, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Home,
  Video,
  Image,
  PlayCircle,
  BarChart3,
  Bell,
  Settings,
  Info,
  Shield,
  Activity,
  X
} from 'lucide-react';
import { useApp } from '@/contexts/AppContext';

const Sidebar = ({ isOpen, onClose, isMobile }) => {
  const location = useLocation();
  const { stats } = useApp();

  const navigation = [
    {
      name: 'Dashboard',
      href: '/',
      icon: Home,
      badge: null,
    },
    {
      name: 'Live Monitoring',
      href: '/live',
      icon: Video,
      badge: stats.activeMonitoring ? 'LIVE' : null,
      badgeColor: 'bg-red-500 animate-pulse',
    },
    {
      name: 'Image Detection',
      href: '/image',
      icon: Image,
      badge: null,
    },
    {
      name: 'Video Processing',
      href: '/video',
      icon: PlayCircle,
      badge: null,
    },
    {
      name: 'Analytics',
      href: '/analytics',
      icon: BarChart3,
      badge: null,
    },
    {
      name: 'Alerts',
      href: '/alerts',
      icon: Bell,
      badge: stats.totalAlerts > 0 ? stats.totalAlerts : null,
      badgeColor: 'bg-red-600',
    },
  ];

  const bottomNavigation = [
    {
      name: 'Settings',
      href: '/settings',
      icon: Settings,
    },
    {
      name: 'About',
      href: '/about',
      icon: Info,
    },
  ];

  const isActive = (path) => {
    if (path === '/') {
      return location.pathname === '/';
    }
    return location.pathname.startsWith(path);
  };

  return (
    <>
      {/* Sidebar Container */}
      <motion.aside
        initial={false}
        animate={{
          x: isOpen ? 0 : -320,
        }}
        transition={{
          type: 'spring',
          stiffness: 300,
          damping: 30,
        }}
        className={`
          fixed lg:relative inset-y-0 left-0 z-50
          w-64 bg-white dark:bg-gray-800 
          border-r border-gray-200 dark:border-gray-700
          flex flex-col shadow-xl lg:shadow-none
        `}
      >
        {/* Sidebar Header */}
        <div className="flex items-center justify-between h-16 px-4 border-b border-gray-200 dark:border-gray-700">
          <Link to="/" className="flex items-center gap-3">
            <div className="flex items-center justify-center w-10 h-10 bg-gradient-to-br from-red-500 to-red-600 rounded-lg shadow-lg">
              <Shield className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-gray-900 dark:text-white">
                Safety AI
              </h2>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                v1.0.0
              </p>
            </div>
          </Link>
          
          {/* Close button (mobile only) */}
          {isMobile && (
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors lg:hidden"
            >
              <X className="w-5 h-5 text-gray-600 dark:text-gray-300" />
            </button>
          )}
        </div>

        {/* System Status */}
        <div className="px-4 py-3 bg-gray-50 dark:bg-gray-900">
          <div className="flex items-center gap-3 text-sm">
            <div className="relative">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <div className="absolute inset-0 w-3 h-3 bg-green-500 rounded-full animate-ping opacity-75"></div>
            </div>
            <div>
              <p className="font-medium text-gray-900 dark:text-white">
                System Online
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                All services operational
              </p>
            </div>
          </div>
        </div>

        {/* Navigation Links */}
        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto scrollbar-thin">
          {/* Main Navigation */}
          <div className="space-y-1">
            {navigation.map((item) => {
              const active = isActive(item.href);
              const Icon = item.icon;

              return (
                <Link
                  key={item.name}
                  to={item.href}
                  onClick={isMobile ? onClose : undefined}
                  className={`
                    group flex items-center justify-between px-3 py-2.5 rounded-lg
                    transition-all duration-200
                    ${active
                      ? 'bg-red-50 dark:bg-red-900 dark:bg-opacity-20 text-red-600 dark:text-red-400'
                      : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                    }
                  `}
                >
                  <div className="flex items-center gap-3">
                    <Icon className={`w-5 h-5 ${active ? 'text-red-600 dark:text-red-400' : ''}`} />
                    <span className="font-medium text-sm">{item.name}</span>
                  </div>
                  
                  {/* Badge */}
                  {item.badge && (
                    <span className={`
                      px-2 py-0.5 text-xs font-semibold text-white rounded-full
                      ${item.badgeColor || 'bg-gray-600'}
                    `}>
                      {item.badge}
                    </span>
                  )}
                </Link>
              );
            })}
          </div>

          {/* Divider */}
          <div className="my-4 border-t border-gray-200 dark:border-gray-700"></div>

          {/* Bottom Navigation */}
          <div className="space-y-1">
            {bottomNavigation.map((item) => {
              const active = isActive(item.href);
              const Icon = item.icon;

              return (
                <Link
                  key={item.name}
                  to={item.href}
                  onClick={isMobile ? onClose : undefined}
                  className={`
                    group flex items-center gap-3 px-3 py-2.5 rounded-lg
                    transition-all duration-200
                    ${active
                      ? 'bg-red-50 dark:bg-red-900 dark:bg-opacity-20 text-red-600 dark:text-red-400'
                      : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                    }
                  `}
                >
                  <Icon className={`w-5 h-5 ${active ? 'text-red-600 dark:text-red-400' : ''}`} />
                  <span className="font-medium text-sm">{item.name}</span>
                </Link>
              );
            })}
          </div>
        </nav>

        {/* Stats Section */}
        <div className="px-4 py-3 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-gray-600 dark:text-gray-400">Detections Today</span>
              <span className="font-semibold text-gray-900 dark:text-white">
                {stats.totalDetections}
              </span>
            </div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-gray-600 dark:text-gray-400">Active Alerts</span>
              <span className="font-semibold text-red-600 dark:text-red-400">
                {stats.totalAlerts}
              </span>
            </div>
          </div>
        </div>

        {/* User Info Footer */}
        <div className="px-4 py-3 border-t border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-to-br from-red-500 to-red-600 rounded-full flex items-center justify-center text-white font-semibold text-sm">
              A
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                A-P-U-R-B-O
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Admin
              </p>
            </div>
          </div>
        </div>
      </motion.aside>
    </>
  );
};

export default Sidebar;
