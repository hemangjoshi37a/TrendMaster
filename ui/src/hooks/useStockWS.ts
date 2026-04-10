import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * Custom Hook to manage WebSocket life-cycle for live stock ticks
 */
export const useStockWS = (symbol: string | null, onPriceUpdate?: (price: number) => void) => {
  const [wsStatus, setWsStatus] = useState<'connected' | 'disconnected' | 'reconnecting'>('disconnected');
  const ws = useRef<WebSocket | null>(null);
  const reconnectTimeout = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef<number>(0);

  const connect = useCallback((targetSymbol: string) => {
    if (ws.current) ws.current.close();
    if (reconnectTimeout.current) clearTimeout(reconnectTimeout.current);
    reconnectAttempts.current = 0;

    const createConnection = () => {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      let host = window.location.host;
      
      // Development port redirection for local dev
      if (host === 'localhost:3000') {
        host = 'localhost:8000';
      }
      
      const socket = new WebSocket(`${protocol}//${host}/ws/ticks/${targetSymbol.toUpperCase()}`);

      socket.onopen = () => {
        setWsStatus('connected');
        reconnectAttempts.current = 0;
      };

      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.price && onPriceUpdate) {
          onPriceUpdate(data.price);
        }
      };

      socket.onclose = () => {
        setWsStatus('reconnecting');
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);
        reconnectAttempts.current += 1;
        reconnectTimeout.current = setTimeout(createConnection, delay);
      };

      socket.onerror = () => {
        socket.close();
      };

      ws.current = socket;
    };

    createConnection();
  }, [onPriceUpdate]);

  useEffect(() => {
    if (symbol) {
      connect(symbol);
    }
    return () => {
      if (ws.current) ws.current.close();
      if (reconnectTimeout.current) clearTimeout(reconnectTimeout.current);
    };
  }, [symbol, connect]);

  return { wsStatus };
};
