/**
 * useWebSocket Custom Hook
 * React hook for WebSocket connections with automatic reconnection
 * 
 * Author: A-P-U-R-B-O
 * Created: 2025-11-07 16:30:24 UTC
 * Version: 1.0.0
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { WebSocketManager } from '@/services/api';
import toast from 'react-hot-toast';

/**
 * Custom hook for WebSocket connections
 * 
 * @param {string} endpoint - WebSocket endpoint (default: '/ws/detect')
 * @param {Object} options - Configuration options
 * @param {boolean} options.autoConnect - Auto-connect on mount (default: false)
 * @param {Function} options.onMessage - Message handler callback
 * @param {Function} options.onOpen - Connection open callback
 * @param {Function} options.onClose - Connection close callback
 * @param {Function} options.onError - Error callback
 * @returns {Object} WebSocket state and controls
 */
const useWebSocket = (endpoint = '/ws/detect', options = {}) => {
  const {
    autoConnect = false,
    onMessage,
    onOpen,
    onClose,
    onError,
  } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [lastMessage, setLastMessage] = useState(null);
  const [messageHistory, setMessageHistory] = useState([]);
  const [error, setError] = useState(null);
  const [reconnectAttempt, setReconnectAttempt] = useState(0);

  const wsRef = useRef(null);
  const messageHistoryMaxSize = useRef(50);

  /**
   * Initialize WebSocket connection
   */
  const connect = useCallback(() => {
    if (wsRef.current?.isConnected()) {
      console.warn('[useWebSocket] Already connected');
      return;
    }

    console.log('%c[useWebSocket] Connecting...', 'color: #8b5cf6; font-weight: bold');
    setIsConnecting(true);
    setError(null);

    // Create WebSocket manager
    wsRef.current = new WebSocketManager(endpoint);

    // Handle connection open
    wsRef.current.on('open', (event) => {
      console.log('%c[useWebSocket] Connected successfully', 'color: #22c55e; font-weight: bold');
      setIsConnected(true);
      setIsConnecting(false);
      setReconnectAttempt(0);
      setError(null);

      if (onOpen) {
        onOpen(event);
      }
    });

    // Handle incoming messages
    wsRef.current.on('message', (data) => {
      setLastMessage(data);
      
      // Update message history
      setMessageHistory((prev) => {
        const newHistory = [data, ...prev].slice(0, messageHistoryMaxSize.current);
        return newHistory;
      });

      if (onMessage) {
        onMessage(data);
      }
    });

    // Handle errors
    wsRef.current.on('error', (event) => {
      console.error('%c[useWebSocket] Error', 'color: #ef4444; font-weight: bold', event);
      const errorMsg = 'WebSocket connection error';
      setError(errorMsg);
      setIsConnecting(false);

      if (onError) {
        onError(event);
      }
    });

    // Handle connection close
    wsRef.current.on('close', (event) => {
      console.log('%c[useWebSocket] Connection closed', 'color: #f59e0b; font-weight: bold');
      setIsConnected(false);
      setIsConnecting(false);

      if (onClose) {
        onClose(event);
      }

      // Track reconnect attempts
      setReconnectAttempt((prev) => prev + 1);
    });

    // Start connection
    wsRef.current.connect();
  }, [endpoint, onMessage, onOpen, onClose, onError]);

  /**
   * Disconnect WebSocket
   */
  const disconnect = useCallback(() => {
    if (wsRef.current) {
      console.log('%c[useWebSocket] Disconnecting...', 'color: #6b7280');
      wsRef.current.close();
      wsRef.current = null;
      setIsConnected(false);
      setIsConnecting(false);
      setError(null);
    }
  }, []);

  /**
   * Send data through WebSocket
   * @param {Object} data - Data to send
   */
  const sendMessage = useCallback((data) => {
    if (!wsRef.current || !isConnected) {
      console.warn('[useWebSocket] Cannot send message - not connected');
      toast.error('Not connected to server');
      return false;
    }

    try {
      wsRef.current.send(data);
      return true;
    } catch (error) {
      console.error('[useWebSocket] Error sending message:', error);
      setError(error.message);
      return false;
    }
  }, [isConnected]);

  /**
   * Clear message history
   */
  const clearHistory = useCallback(() => {
    setMessageHistory([]);
    setLastMessage(null);
  }, []);

  /**
   * Reconnect WebSocket
   */
  const reconnect = useCallback(() => {
    disconnect();
    setTimeout(() => {
      connect();
    }, 500);
  }, [connect, disconnect]);

  // Auto-connect on mount if enabled
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    // Cleanup on unmount
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [autoConnect, connect]);

  return {
    // Connection state
    isConnected,
    isConnecting,
    error,
    reconnectAttempt,

    // Message data
    lastMessage,
    messageHistory,

    // Controls
    connect,
    disconnect,
    reconnect,
    sendMessage,
    clearHistory,

    // WebSocket instance (for advanced usage)
    ws: wsRef.current,
  };
};

export default useWebSocket;
