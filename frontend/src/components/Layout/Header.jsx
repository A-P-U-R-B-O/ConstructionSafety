/**
 * Header Component
 * Top navigation bar with menu toggle, search, and user actions
 * 
 * Author: A-P-U-R-B-O
 * Created: 2025-11-07 16:03:18 UTC
 * Version: 1.0.0
 */

import { useState } from 'react';
import { Link } from 'react-router-dom';
import { 
  Menu, 
  X, 
  Search, 
  Bell, 
  Settings, 
  Moon, 
  Sun,
  User,
  LogOut,
  Activity,
  Shield
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useApp } from '@/contexts/AppContext';

const Header = ({ onMenuClick, sidebarOpen }) => {
  const { user, theme, toggleTheme, notifications } = useApp();
  const [showSearch, setShowSearch] = useState(false);
  const [showNotifications, setShowNotifications] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const unreadNotifications = notifications.filter(n => !n.read).length;

  return (
    <header className="sticky top-0 z-30 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 shadow-sm">
      <div className="flex items-center justify-between h-16 px-4 sm:px-6 lg:px-8">
        
        {/* Left Section - Logo & Menu Toggle */}
        <div className="flex items-center gap-4">
          {/* Menu Toggle */}
          <button
            onClick={onMenuClick}
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors lg:hidden"
            aria-label="Toggle menu"
          >
            {sidebarOpen ? (
              <X className="w-6 h-6 text-gray-600 dark:text-gray-300" />
            ) : (
              <Menu className="w-6 h-6 text-gray-600 dark:text-gray-300" />
            )}
          </button>

          {/* Logo */}
          <Link 
            to="/" 
            className="flex items-center gap-3 group"
          >
            <div className="flex items-center justify-center w-10 h-10 bg-gradient-to-br from-red-500 to-red-600 rounded-lg shadow-lg group-hover:shadow-xl transition-shadow">
              <Shield className="w-6 h-6 text-white" />
            </div>
            <div className="hidden sm:block">
              <h1 className="text-lg font-bold text-gray-900 dark:text-white">
                Safety AI
              </h1>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Construction Monitor
              </p>
            </div>
          </Link>
        </div>

        {/* Center Section - Search (Desktop) */}
        <div className="hidden md:flex flex-1 max-w-md mx-8">
          <div className="relative w-full">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search detections, alerts, analytics..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-gray-100 dark:bg-gray-700 border-0 rounded-lg focus:ring-2 focus:ring-red-500 transition-shadow"
            />
          </div>
        </div>

        {/* Right Section - Actions */}
        <div className="flex items-center gap-2">
          
          {/* Search Toggle (Mobile) */}
          <button
            onClick={() => setShowSearch(!showSearch)}
            className="md:hidden p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            aria-label="Toggle search"
          >
            <Search className="w-5 h-5 text-gray-600 dark:text-gray-300" />
          </button>

          {/* System Status Indicator */}
          <div className="hidden lg:flex items-center gap-2 px-3 py-1.5 bg-green-50 dark:bg-green-900 dark:bg-opacity-20 rounded-full">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
            </span>
            <span className="text-xs font-medium text-green-700 dark:text-green-300">
              System Active
            </span>
          </div>

          {/* Theme Toggle */}
          <button
            onClick={toggleTheme}
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            aria-label="Toggle theme"
          >
            {theme === 'dark' ? (
              <Sun className="w-5 h-5 text-yellow-500" />
            ) : (
              <Moon className="w-5 h-5 text-gray-600" />
            )}
          </button>

          {/* Notifications */}
          <div className="relative">
            <button
              onClick={() => setShowNotifications(!showNotifications)}
              className="relative p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              aria-label="Notifications"
            >
              <Bell className="w-5 h-5 text-gray-600 dark:text-gray-300" />
              {unreadNotifications > 0 && (
                <span className="absolute top-1 right-1 flex items-center justify-center w-4 h-4 text-xs font-bold text-white bg-red-600 rounded-full">
                  {unreadNotifications}
                </span>
              )}
            </button>

            {/* Notifications Dropdown */}
            <AnimatePresence>
              {showNotifications && (
                <NotificationsDropdown
                  notifications={notifications}
                  onClose={() => setShowNotifications(false)}
                />
              )}
            </AnimatePresence>
          </div>

          {/* Settings */}
          <Link
            to="/settings"
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            aria-label="Settings"
          >
            <Settings className="w-5 h-5 text-gray-600 dark:text-gray-300" />
          </Link>

          {/* User Menu */}
          <div className="relative">
            <button
              onClick={() => setShowUserMenu(!showUserMenu)}
              className="flex items-center gap-2 p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            >
              <div className="w-8 h-8 bg-gradient-to-br from-red-500 to-red-600 rounded-full flex items-center justify-center text-white font-semibold text-sm">
                {user.name.charAt(0)}
              </div>
              <span className="hidden lg:block text-sm font-medium text-gray-700 dark:text-gray-300">
                {user.name}
              </span>
            </button>

            {/* User Dropdown */}
            <AnimatePresence>
              {showUserMenu && (
                <UserDropdown
                  user={user}
                  onClose={() => setShowUserMenu(false)}
                />
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>

      {/* Mobile Search Bar */}
      <AnimatePresence>
        {showSearch && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="md:hidden border-t border-gray-200 dark:border-gray-700 px-4 py-3"
          >
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 bg-gray-100 dark:bg-gray-700 border-0 rounded-lg focus:ring-2 focus:ring-red-500"
                autoFocus
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
};

// ============================================================================
// Notifications Dropdown Component
// ============================================================================

const NotificationsDropdown = ({ notifications, onClose }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className="absolute right-0 mt-2 w-80 bg-white dark:bg-gray-800 rounded-lg shadow-xl border border-gray-200 dark:border-gray-700 overflow-hidden z-50"
    >
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
        <h3 className="font-semibold text-gray-900 dark:text-white">
          Notifications
        </h3>
        <button
          onClick={onClose}
          className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Notifications List */}
      <div className="max-h-96 overflow-y-auto scrollbar-thin">
        {notifications.length === 0 ? (
          <div className="px-4 py-8 text-center text-gray-500 dark:text-gray-400">
            <Bell className="w-12 h-12 mx-auto mb-2 opacity-50" />
            <p>No notifications</p>
          </div>
        ) : (
          notifications.map((notification) => (
            <div
              key={notification.id}
              className="px-4 py-3 hover:bg-gray-50 dark:hover:bg-gray-700 border-b border-gray-100 dark:border-gray-700 last:border-0 cursor-pointer"
            >
              <div className="flex items-start gap-3">
                <div className={`flex-shrink-0 w-2 h-2 mt-2 rounded-full ${
                  notification.read ? 'bg-gray-300' : 'bg-red-500'
                }`}></div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 dark:text-white">
                    {notification.title}
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    {notification.message}
                  </p>
                  <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
                    {new Date(notification.timestamp).toLocaleTimeString()}
                  </p>
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Footer */}
      {notifications.length > 0 && (
        <div className="px-4 py-3 border-t border-gray-200 dark:border-gray-700">
          <Link
            to="/alerts"
            className="text-sm text-red-600 dark:text-red-400 hover:underline"
            onClick={onClose}
          >
            View all alerts →
          </Link>
        </div>
      )}
    </motion.div>
  );
};

// ============================================================================
// User Dropdown Component
// ============================================================================

const UserDropdown = ({ user, onClose }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className="absolute right-0 mt-2 w-64 bg-white dark:bg-gray-800 rounded-lg shadow-xl border border-gray-200 dark:border-gray-700 overflow-hidden z-50"
    >
      {/* User Info */}
      <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
        <p className="text-sm font-semibold text-gray-900 dark:text-white">
          {user.name}
        </p>
        <p className="text-xs text-gray-500 dark:text-gray-400">
          {user.email}
        </p>
        <div className="mt-2">
          <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200">
            {user.role}
          </span>
        </div>
      </div>

      {/* Menu Items */}
      <div className="py-2">
        <Link
          to="/settings"
          className="flex items-center gap-3 px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
          onClick={onClose}
        >
          <User className="w-4 h-4" />
          <span>Profile</span>
        </Link>
        
        <Link
          to="/analytics"
          className="flex items-center gap-3 px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
          onClick={onClose}
        >
          <Activity className="w-4 h-4" />
          <span>Activity</span>
        </Link>
        
        <Link
          to="/settings"
          className="flex items-center gap-3 px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
          onClick={onClose}
        >
          <Settings className="w-4 h-4" />
          <span>Settings</span>
        </Link>
      </div>

      {/* Logout */}
      <div className="border-t border-gray-200 dark:border-gray-700 py-2">
        <button
          className="flex items-center gap-3 px-4 py-2 text-sm text-red-600 dark:text-red-400 hover:bg-gray-100 dark:hover:bg-gray-700 w-full"
          onClick={() => {
            console.log('Logout clicked');
            onClose();
          }}
        >
          <LogOut className="w-4 h-4" />
          <span>Logout</span>
        </button>
      </div>

      {/* Footer */}
      <div className="px-4 py-2 bg-gray-50 dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700">
        <p className="text-xs text-gray-500 dark:text-gray-400">
          Session: {new Date().toLocaleTimeString()}
        </p>
      </div>
    </motion.div>
  );
};

export default Header;
