/**
 * Settings Page Component
 * Application settings and configuration
 * 
 * Author: A-P-U-R-B-O
 * Created: 2025-11-07 16:38:03 UTC
 * Version: 1.0.0
 */

import { useState } from 'react';
import { Settings as SettingsIcon, User, Bell, Shield, Palette } from 'lucide-react';
import { useApp } from '@/contexts/AppContext';
import toast from 'react-hot-toast';

const Settings = () => {
  const { theme, toggleTheme, user } = useApp();
  const [settings, setSettings] = useState({
    confidenceThreshold: 0.5,
    alertSound: true,
    autoSave: false,
    notifications: true,
  });

  const handleSave = () => {
    toast.success('Settings saved successfully!');
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Settings
        </h1>
        <p className="mt-1 text-gray-600 dark:text-gray-400">
          Manage your application preferences
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Sidebar */}
        <div className="lg:col-span-1">
          <div className="card">
            <div className="card-body p-4">
              <nav className="space-y-1">
                {[
                  { icon: User, label: 'Profile', active: true },
                  { icon: Palette, label: 'Appearance', active: false },
                  { icon: Bell, label: 'Notifications', active: false },
                  { icon: Shield, label: 'Security', active: false },
                ].map((item, idx) => (
                  <button
                    key={idx}
                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-left transition-colors ${
                      item.active
                        ? 'bg-red-50 dark:bg-red-900 dark:bg-opacity-20 text-red-600 dark:text-red-400'
                        : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                    }`}
                  >
                    <item.icon className="w-5 h-5" />
                    <span className="font-medium">{item.label}</span>
                  </button>
                ))}
              </nav>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="lg:col-span-2 space-y-6">
          {/* Profile */}
          <div className="card">
            <div className="card-header">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                Profile Information
              </h2>
            </div>
            <div className="card-body space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Name
                </label>
                <input
                  type="text"
                  defaultValue={user.name}
                  className="w-full"
                  readOnly
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Email
                </label>
                <input
                  type="email"
                  defaultValue={user.email}
                  className="w-full"
                  readOnly
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Role
                </label>
                <input
                  type="text"
                  defaultValue={user.role}
                  className="w-full"
                  readOnly
                />
              </div>
            </div>
          </div>

          {/* Appearance */}
          <div className="card">
            <div className="card-header">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                Appearance
              </h2>
            </div>
            <div className="card-body">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium text-gray-900 dark:text-white">
                    Dark Mode
                  </p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Current theme: {theme === 'dark' ? 'Dark' : 'Light'}
                  </p>
                </div>
                <button
                  onClick={toggleTheme}
                  className="btn btn-secondary"
                >
                  Toggle Theme
                </button>
              </div>
            </div>
          </div>

          {/* Detection Settings */}
          <div className="card">
            <div className="card-header">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                Detection Settings
              </h2>
            </div>
            <div className="card-body space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Confidence Threshold: {settings.confidenceThreshold}
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={settings.confidenceThreshold}
                  onChange={(e) =>
                    setSettings({ ...settings, confidenceThreshold: parseFloat(e.target.value) })
                  }
                  className="w-full"
                />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium text-gray-900 dark:text-white">Alert Sound</p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Play sound for critical alerts
                  </p>
                </div>
                <input
                  type="checkbox"
                  checked={settings.alertSound}
                  onChange={(e) => setSettings({ ...settings, alertSound: e.target.checked })}
                />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium text-gray-900 dark:text-white">Auto Save</p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Automatically save detections
                  </p>
                </div>
                <input
                  type="checkbox"
                  checked={settings.autoSave}
                  onChange={(e) => setSettings({ ...settings, autoSave: e.target.checked })}
                />
              </div>
            </div>
            <div className="card-footer">
              <button onClick={handleSave} className="btn btn-primary">
                Save Settings
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Settings;
