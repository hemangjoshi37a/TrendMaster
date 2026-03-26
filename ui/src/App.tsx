import React, { useState, useEffect, useRef, useCallback } from 'react';
import { BrowserRouter, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import LandingPage from './LandingPage';
import './App.css';
import LineChart from './LineChart';
import ErrorBoundary from './ErrorBoundary';

interface Company {
  symbol: string;
  name: string;
}

interface PredictionData {
  symbol: string;
  company_name: string;
  dates: string[];
  prices: number[];
  prediction_start_index: number;
  confidence_score?: number;
  warning?: string;
}

interface RecentStock {
  symbol: string;
  price: number | null;
}

interface MarketIndex {
  name: string;
  price: number;
  change_pct: number;
}

type TimeframePeriod = '1mo' | '3mo' | '6mo' | '1y' | '2y' | '5y' | 'max';

const TIMEFRAME_MAP: { label: string; period: TimeframePeriod }[] = [
  { label: '1M', period: '1mo' },
  { label: '3M', period: '3mo' },
  { label: '6M', period: '6mo' },
  { label: '1Y', period: '1y' },
  { label: '2Y', period: '2y' },
  { label: '5Y', period: '5y' },
  { label: 'MAX', period: 'max' },
];

function Dashboard() {
  const [query, setQuery] = useState<string>('');
  const [suggestions, setSuggestions] = useState<Company[]>([]);
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [livePrice, setLivePrice] = useState<number | null>(null);
  const [prevPrice, setPrevPrice] = useState<number | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [recentStocks, setRecentStocks] = useState<RecentStock[]>([]);
  const [marketOpen, setMarketOpen] = useState<boolean>(true);
  const [showSuggestions, setShowSuggestions] = useState<boolean>(false);
  const [selectedTimeframe, setSelectedTimeframe] = useState<TimeframePeriod>('1y');
  const [wsStatus, setWsStatus] = useState<'connected' | 'disconnected' | 'reconnecting'>('disconnected');
  const [marketIndices, setMarketIndices] = useState<MarketIndex[]>([]);
  const location = useLocation();
  const [currentSymbol, setCurrentSymbol] = useState<string>('');
  const [isPro, setIsPro] = useState<boolean>(location.state?.isPro || false);
  const [showPricing, setShowPricing] = useState<boolean>(false);
  const [alerts, setAlerts] = useState<{symbol: string, target: number, type: 'above' | 'below', active: boolean}[]>([]);
  const [triggeredAlerts, setTriggeredAlerts] = useState<string[]>([]); // Array of alert IDs or symbols that flared
  const [showAddAlert, setShowAddAlert] = useState<boolean>(false);
  const [alertForm, setAlertForm] = useState<{target: string, type: 'above' | 'below'}>({target: '', type: 'above'});

  // --- Trial / Subscription Expiry Logic ---
  const TRIAL_DAYS = 10;
  const PRO_DAYS = 30;

  // Persist and read account metadata from localStorage
  const getAccountMeta = () => {
    const raw = localStorage.getItem('tm_account');
    return raw ? JSON.parse(raw) : null;
  };

  const saveAccountMeta = (meta: { isPro: boolean; startDate: string }) => {
    localStorage.setItem('tm_account', JSON.stringify(meta));
  };

  // On mount: if navigating from landing with state, record the sign-in date
  useEffect(() => {
    if (location.state?.isPro !== undefined) {
      const existing = getAccountMeta();
      // Only reset the start date if this is a fresh login (no existing record or plan changed)
      if (!existing || existing.isPro !== location.state.isPro) {
        saveAccountMeta({ isPro: location.state.isPro, startDate: new Date().toISOString() });
      }
    }
  }, []);

  const accountMeta = getAccountMeta();
  const startDate = accountMeta ? new Date(accountMeta.startDate) : new Date();
  const daysSinceStart = Math.floor((Date.now() - startDate.getTime()) / (1000 * 60 * 60 * 24));
  const daysAllowed = isPro ? PRO_DAYS : TRIAL_DAYS;
  const daysRemaining = Math.max(0, daysAllowed - daysSinceStart);
  const isExpired = daysRemaining === 0;
  // ---- end expiry logic ----

  const ws = useRef<WebSocket | null>(null);
  const isSelecting = useRef<boolean>(false);
  const reconnectTimeout = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef<number>(0);

  useEffect(() => {
    const saved = localStorage.getItem('recentStocks');
    if (saved) setRecentStocks(JSON.parse(saved));

    const savedAlerts = localStorage.getItem('tm_alerts');
    if (savedAlerts) setAlerts(JSON.parse(savedAlerts));

    const checkMarket = () => {
      const now = new Date();
      const istTime = new Date(now.toLocaleString('en-US', { timeZone: 'Asia/Kolkata' }));
      const day = istTime.getDay();
      const hour = istTime.getHours();
      const min = istTime.getMinutes();
      const isWeekend = day === 0 || day === 6;
      const isWorkingHours = (hour === 9 && min >= 15) || (hour > 9 && hour < 15) || (hour === 15 && min <= 30);
      setMarketOpen(!isWeekend && isWorkingHours);
    };
    checkMarket();
    const interval = setInterval(checkMarket, 60000);

    fetchMarketOverview();
    const marketInterval = setInterval(fetchMarketOverview, 60000);

    return () => {
      clearInterval(interval);
      clearInterval(marketInterval);
    };
  }, []);

  const fetchMarketOverview = async () => {
    try {
      const res = await fetch('/api/market-overview');
      if (res.ok) {
        const data: MarketIndex[] = await res.json();
        setMarketIndices(data);
      }
    } catch (e) {
      console.error('Failed to fetch market overview:', e);
    }
  };

  useEffect(() => {
    if (isSelecting.current) {
      isSelecting.current = false;
      return;
    }

    if (query.length > 1) {
      const fetchSuggestions = async () => {
        try {
          const res = await fetch(`/api/search?query=${encodeURIComponent(query)}`);
          if (res.ok) {
            const data = await res.json();
            setSuggestions(data);
            setShowSuggestions(data.length > 0);
          }
        } catch (e) {
          console.error("Search error", e);
        }
      };
      const delayDebounceFn = setTimeout(fetchSuggestions, 300);
      return () => clearTimeout(delayDebounceFn);
    } else {
      setSuggestions([]);
      setShowSuggestions(false);
    }
  }, [query]);

  const removeAlert = (symbol: string, target: number) => {
    const updated = alerts.filter(a => !(a.symbol === symbol && a.target === target));
    setAlerts(updated);
    localStorage.setItem('tm_alerts', JSON.stringify(updated));
  };

  const addAlert = (symbol: string, target: number, type: 'above' | 'below') => {
    if (!isPro) return;
    const newAlert = { symbol: symbol.toUpperCase(), target, type, active: true };
    const updated = [...alerts, newAlert];
    setAlerts(updated);
    localStorage.setItem('tm_alerts', JSON.stringify(updated));
  };

  const dismissAlert = (id: string) => {
    setTriggeredAlerts(prev => prev.filter(a => a !== id));
  };

  const checkAlerts = useCallback((symbol: string, price: number) => {
    if (!isPro) return;
    
    alerts.forEach(alert => {
      if (alert.active && alert.symbol === symbol) {
        const isTriggered = alert.type === 'above' ? price >= alert.target : price <= alert.target;
        if (isTriggered) {
          // Trigger alert!
          const alertId = `${symbol}-${alert.target}`;
          setTriggeredAlerts(prev => Array.from(new Set([...prev, alertId])));
          
          // Deactivate so it doesn't spam
          const updated = alerts.map(a => 
            a.symbol === symbol && a.target === alert.target ? { ...a, active: false } : a
          );
          setAlerts(updated);
          localStorage.setItem('tm_alerts', JSON.stringify(updated));
        }
      }
    });
  }, [alerts, isPro]);

  const connectWebSocket = useCallback((stockSymbol: string) => {
    if (ws.current) ws.current.close();
    if (reconnectTimeout.current) clearTimeout(reconnectTimeout.current);
    reconnectAttempts.current = 0;

    const createConnection = () => {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.host;
      const socket = new WebSocket(`${protocol}//${host}/ws/ticks/${stockSymbol.toUpperCase()}`);

      socket.onopen = () => {
        setWsStatus('connected');
        reconnectAttempts.current = 0;
      };

      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.price) {
          setLivePrice(prev => {
            setPrevPrice(prev);
            checkAlerts(stockSymbol.toUpperCase(), data.price);
            return data.price;
          });
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
  }, []);

  useEffect(() => {
    return () => {
      if (ws.current) ws.current.close();
      if (reconnectTimeout.current) clearTimeout(reconnectTimeout.current);
    };
  }, []);

  const fetchPrediction = useCallback(async (symbol: string, period: TimeframePeriod) => {
    setLoading(true);
    setError(null);
    setPrediction(null);
    setLivePrice(null);
    setPrevPrice(null);

    try {
      const response = await fetch(`/api/predict?stock_symbol=${encodeURIComponent(symbol)}&period=${period}`);
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || "Failed to fetch prediction for " + symbol);
      }
      const data: PredictionData = await response.json();
      setPrediction(data);
      connectWebSocket(symbol);

      setRecentStocks(prev => {
        const filtered = prev.filter(s => s.symbol !== symbol);
        const updated = [{ symbol: symbol, price: null }, ...filtered].slice(0, 10);
        localStorage.setItem('recentStocks', JSON.stringify(updated));
        return updated;
      });
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [connectWebSocket]);

  const handleManualSearch = () => {
    if (query.trim()) {
      const sym = query.trim().toUpperCase();
      setCurrentSymbol(sym);
      isSelecting.current = true;
      setQuery(sym);
      setSuggestions([]);
      setShowSuggestions(false);
      fetchPrediction(sym, selectedTimeframe);
    }
  };

  const handleSelectCompany = (company: Company | RecentStock) => {
    isSelecting.current = true;
    setQuery(company.symbol);
    setSuggestions([]);
    setShowSuggestions(false);
    setCurrentSymbol(company.symbol);
    fetchPrediction(company.symbol, selectedTimeframe);
  };

  const handleTimeframeChange = (period: TimeframePeriod) => {
    if (loading) return;
    setSelectedTimeframe(period);
    if (currentSymbol) {
      fetchPrediction(currentSymbol, period);
    }
  };

  const priceChange = livePrice && prevPrice ? livePrice - prevPrice : 0;
  const priceChangePct = prevPrice ? (priceChange / prevPrice) * 100 : 0;

  return (
    <div className="App dark-theme">
      {/* Top Navigation */}
      <nav className="navbar">
        <div className="brand">
          <div className="logo">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
            </svg>
            TrendMaster <span>{isPro ? 'PRO' : 'FREE TRIAL'}</span>
          </div>
        </div>

        <div className="search-box">
          <div className="search-input-wrapper">
            <div className="search-icon">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="11" cy="11" r="8"></circle>
                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
              </svg>
            </div>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onFocus={() => suggestions.length > 0 && setShowSuggestions(true)}
              onKeyDown={(e) => e.key === 'Enter' && handleManualSearch()}
              placeholder="Search markets, symbols (e.g. RELIANCE, TCS)..."
              autoComplete="off"
            />
          </div>
          {showSuggestions && (
            <ul className="suggestions">
              {suggestions.map((c) => (
                <li key={c.symbol} onClick={() => handleSelectCompany(c)}>
                  <span className="sym">{c.symbol}</span>
                  <span className="nam">{c.name}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
        
        <div className="brand">
          <div className={`market-status ${marketOpen ? 'open' : 'closed'}`}>
            <span className="status-dot"></span>
            {marketOpen ? 'NSE OPEN' : 'NSE CLOSED'}
          </div>
        </div>
      </nav>

      {/* Trial / Expiry Banner */}
      {!isExpired && (
        <div style={{
          background: isPro
            ? 'linear-gradient(90deg, rgba(8,153,129,0.15), transparent)'
            : daysRemaining <= 3
              ? 'linear-gradient(90deg, rgba(242,54,69,0.15), transparent)'
              : 'linear-gradient(90deg, rgba(41,98,255,0.12), transparent)',
          borderBottom: '1px solid',
          borderColor: isPro ? '#089981' : daysRemaining <= 3 ? '#f23645' : '#2962FF',
          padding: '8px 24px',
          fontSize: '0.85rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <span style={{ color: '#d1d4dc' }}>
            {isPro
              ? `✅ Pro Terminal — ${daysRemaining} day${daysRemaining !== 1 ? 's' : ''} remaining in your subscription`
              : `⏳ Free Trial — ${daysRemaining} day${daysRemaining !== 1 ? 's' : ''} remaining. Upgrade to keep full access.`}
          </span>
          {!isPro && (
            <button
              onClick={() => setShowPricing(true)}
              style={{ fontSize: '0.8rem', padding: '4px 14px', background: '#2962FF', color: '#fff', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 600 }}
            >
              Upgrade to Pro
            </button>
          )}
        </div>
      )}

      {/* Triggered Alerts Notification Bar */}
      {isPro && triggeredAlerts.length > 0 && (
        <div className="alert-notifications-bar">
          {triggeredAlerts.map(id => {
            const [sym, target] = id.split('-');
            return (
              <div key={id} className="alert-banner-item">
                <span className="ab-icon">🔔</span>
                <span className="ab-text">Target Hit: <b>{sym}</b> reached <b>₹{Number(target).toLocaleString('en-IN')}</b></span>
                <button className="ab-close" onClick={() => dismissAlert(id)}>×</button>
              </div>
            );
          })}
        </div>
      )}

      {/* Expired Full Paywall */}
      {isExpired && (
        <div style={{
          position: 'fixed', inset: 0, background: 'rgba(15,18,26,0.97)',
          backdropFilter: 'blur(16px)', display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center', zIndex: 9999, textAlign: 'center', padding: '40px'
        }}>
          <div style={{ fontSize: '3rem', marginBottom: '16px' }}>{isPro ? '🔄' : '🔒'}</div>
          <h2 style={{ fontSize: '2.5rem', fontWeight: 800, color: '#fff', marginBottom: '16px' }}>
            {isPro ? 'Your Pro Subscription Has Expired' : 'Your 10-Day Free Trial Has Ended'}
          </h2>
          <p style={{ color: '#8c9bad', fontSize: '1.1rem', maxWidth: '500px', lineHeight: 1.6, marginBottom: '40px' }}>
            {isPro
              ? 'Renew your Pro plan to continue accessing real-time forecasts, confidence scores, and the full 10-day prediction horizon.'
              : 'You have used your free 10-day trial. Subscribe to Pro to continue making AI-powered predictions on NSE stocks.'}
          </p>
          <button
            onClick={() => setShowPricing(true)}
            style={{ padding: '18px 48px', background: 'linear-gradient(90deg, #2962FF, #1E53E5)', color: '#fff', border: 'none', borderRadius: '10px', fontWeight: 800, fontSize: '1.2rem', cursor: 'pointer', boxShadow: '0 8px 32px rgba(41,98,255,0.4)' }}
          >
            {isPro ? 'Renew Pro — $49/mo' : 'Subscribe to Pro — $49/mo'}
          </button>
        </div>
      )}

      {/* Main Dashboard Layout */}
      <main className="dashboard" onClick={() => setShowSuggestions(false)}>
        
        {/* Left/Center Column - Chart & Table */}
        <div className="main-column">
          <ErrorBoundary>
            <div className="chart-panel">
              {prediction ? (
                <>
                  <div className="chart-header">
                    <div className="stock-info">
                      <div className="stock-symbol">
                        {prediction.symbol}
                        {isPro && (
                          <button 
                            className={`alert-bell-btn ${alerts.some(a => a.symbol === prediction.symbol && a.active) ? 'has-active' : ''}`}
                            onClick={() => {
                              setAlertForm({ target: livePrice?.toString() || '', type: 'above' });
                              setShowAddAlert(true);
                            }}
                            title="Set Price Alert"
                          >
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path>
                              <path d="M13.73 21a2 2 0 0 1-3.46 0"></path>
                            </svg>
                          </button>
                        )}
                        {wsStatus === 'connected' && (
                          <span className="ws-status live"><span className="pulse-dot"></span> LIVE</span>
                        )}
                        {wsStatus === 'reconnecting' && (
                          <span className="ws-status reconnecting">RECONNECTING...</span>
                        )}
                      </div>
                      <div className="stock-name">{prediction.company_name}</div>
                    </div>
                    <div className="stock-price-container">
                      <div className="current-price">
                        {livePrice ? livePrice.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : prediction.prices[prediction.prediction_start_index - 1]?.toFixed(2) || "---"}
                      </div>
                      {priceChange !== 0 && (
                        <div className={`price-change ${priceChange >= 0 ? 'up' : 'down'}`}>
                          {priceChange >= 0 ? '▲' : '▼'} {Math.abs(priceChange).toFixed(2)} ({Math.abs(priceChangePct).toFixed(2)}%)
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="chart-toolbar">
                    <div className="timeframe-selector">
                      {TIMEFRAME_MAP.map(tf => (
                        <button
                          key={tf.period}
                          className={`tf-btn ${selectedTimeframe === tf.period ? 'active' : ''}`}
                          onClick={() => handleTimeframeChange(tf.period)}
                          disabled={loading}
                        >
                          {tf.label}
                        </button>
                      ))}
                    </div>
                    <div className="chart-legend">
                      <div className="legend-item">
                        <div className="legend-color" style={{ background: '#2962FF' }}></div>
                        Historical Data
                      </div>
                      <div className="legend-item">
                        <div className="legend-color" style={{ background: '#F23645' }}></div>
                        Transformer Forecast
                      </div>
                    </div>
                  </div>

                  {prediction.warning && (
                    <div style={{
                      margin: '0 20px 8px',
                      padding: '8px 12px',
                      background: 'rgba(255, 152, 0, 0.12)',
                      border: '1px solid rgba(255, 152, 0, 0.35)',
                      borderRadius: 'var(--radius-sm)',
                      fontSize: '0.78rem',
                      color: 'var(--warning)',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px'
                    }}>
                      <span>⚠</span>
                      <span>{prediction.warning} — Showing historical data only.</span>
                    </div>
                  )}

                  <div className="chart-container-wrapper animate-fade-in">
                    <LineChart data={prediction} isPro={isPro} />
                  </div>
                </>
              ) : (
                <div className="empty-state">
                  {loading ? (
                    <div className="loader">
                      <div className="loader-spinner"></div>
                      <p>Running Transformer Model on {query}...</p>
                    </div>
                  ) : error ? (
                    <>
                      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--error)" strokeWidth="1.5">
                        <circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line>
                      </svg>
                      <h3>Analysis Failed</h3>
                      <p>{error}</p>
                      {currentSymbol && (
                        <button
                          onClick={() => fetchPrediction(currentSymbol, selectedTimeframe)}
                          style={{
                            marginTop: '16px',
                            padding: '8px 20px',
                            background: 'var(--accent)',
                            color: 'white',
                            border: 'none',
                            borderRadius: 'var(--radius-md)',
                            cursor: 'pointer',
                            fontWeight: 600,
                            fontSize: '0.85rem'
                          }}
                        >
                          ↺ Retry
                        </button>
                      )}
                    </>
                  ) : (
                    <div className={`welcome-screen${isPro ? ' pro' : ''}`}>
                      {/* Animated background orbs — gold for Pro, blue for Free */}
                      <div className="welcome-orb welcome-orb-1" />
                      <div className="welcome-orb welcome-orb-2" />
                      <div className="welcome-orb welcome-orb-3" />

                      <div className="welcome-inner">
                        {isPro ? (
                          /* ── PRO TERMINAL WELCOME ── */
                          <>
                            {/* Pro Header */}
                            <div className="welcome-header">
                              <div className="welcome-logo-mark pro-mark">
                                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                  <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
                                </svg>
                              </div>
                              <div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                  <h2 className="welcome-title pro-title">
                                    {(() => {
                                      const h = new Date(new Date().toLocaleString('en-US', { timeZone: 'Asia/Kolkata' })).getHours();
                                      return h < 12 ? 'Good Morning' : h < 17 ? 'Good Afternoon' : 'Good Evening';
                                    })()}, Pro Member
                                  </h2>
                                  <span className="pro-elite-badge">⚡ PRO</span>
                                </div>
                                <p className="welcome-sub">All features unlocked · AI Confidence Scores · Full 10-Day Forecast</p>
                              </div>
                              <div className={`welcome-market-badge ${marketOpen ? 'open' : 'closed'}`}>
                                <span className="status-dot" />
                                {marketOpen ? 'NSE LIVE' : 'NSE CLOSED'}
                              </div>
                            </div>

                            {/* Pro Capabilities Bar */}
                            <div className="pro-caps-bar">
                              {[
                                { icon: '🔮', label: 'Full AI Forecast' },
                                { icon: '⚡', label: 'Real-time WebSockets' },
                                { icon: '📊', label: 'AI Confidence Scores' },
                                { icon: '📈', label: 'Full Historical Data' },
                              ].map((cap, i) => (
                                <div key={i} className="pro-cap-pill" style={{ animationDelay: `${i * 0.06}s` }}>
                                  <span>{cap.icon}</span>
                                  <span>{cap.label}</span>
                                </div>
                              ))}
                            </div>

                            {/* Live Index Cards — gold accent */}
                            <div className="index-cards-grid">
                              {marketIndices.length > 0 ? marketIndices.map((idx, i) => (
                                <div
                                  key={idx.name}
                                  className={`index-card pro-card ${idx.change_pct >= 0 ? 'bull' : 'bear'}`}
                                  style={{ animationDelay: `${i * 0.08}s` }}
                                  onClick={() => handleSelectCompany({ symbol: idx.name.replace(' ', ''), price: null })}
                                >
                                  <div className="ic-glow" />
                                  <div className="ic-top">
                                    <span className="ic-name">{idx.name}</span>
                                    <span className={`ic-badge ${idx.change_pct >= 0 ? 'bull' : 'bear'}`}>
                                      {idx.change_pct >= 0 ? '▲' : '▼'} {Math.abs(idx.change_pct).toFixed(2)}%
                                    </span>
                                  </div>
                                  <div className="ic-price">{idx.price.toLocaleString('en-IN')}</div>
                                  <div className="ic-bar">
                                    <div className="ic-bar-fill" style={{ width: `${Math.min(Math.abs(idx.change_pct) * 10, 100)}%` }} />
                                  </div>
                                </div>
                              )) : [1, 2, 3].map(i => (
                                <div key={i} className="index-card skeleton" style={{ animationDelay: `${i * 0.1}s` }}>
                                  <div className="skeleton-line" style={{ width: '60%', height: '12px', marginBottom: '12px' }} />
                                  <div className="skeleton-line" style={{ width: '80%', height: '24px' }} />
                                </div>
                              ))}
                            </div>

                            {/* AI Signal Feed — Pro exclusive */}
                            <div className="welcome-divider">
                              <div className="divider-line" />
                              <span className="divider-text pro-divider-text">🤖 AI Signal Feed</span>
                              <div className="divider-line" />
                            </div>

                            <div className="ai-signals-row">
                              {[
                                { symbol: 'NIFTY50', label: 'Nifty 50', signal: 'BULLISH', confidence: 87, reason: 'Strong momentum breakout above 200 EMA with volume confirmation' },
                                { symbol: 'RELIANCE', label: 'Reliance', signal: 'BULLISH', confidence: 74, reason: 'Consolidation breakout with institutional volume surge detected' },
                                { symbol: 'INFY', label: 'Infosys', signal: 'BEARISH', confidence: 68, reason: 'MACD bearish crossover with RSI overbought divergence' },
                              ].map((sig, i) => (
                                <div
                                  key={sig.symbol}
                                  className={`ai-signal-card ${sig.signal === 'BULLISH' ? 'bull' : 'bear'}`}
                                  style={{ animationDelay: `${0.35 + i * 0.1}s` }}
                                  onClick={() => handleSelectCompany({ symbol: sig.symbol, price: null })}
                                >
                                  <div className="asc-glow" />
                                  <div className="asc-top">
                                    <div>
                                      <div className="asc-symbol">{sig.symbol}</div>
                                      <div className="asc-label">{sig.label}</div>
                                    </div>
                                    <span className={`asc-badge ${sig.signal === 'BULLISH' ? 'bull' : 'bear'}`}>
                                      {sig.signal === 'BULLISH' ? '▲' : '▼'} {sig.signal}
                                    </span>
                                  </div>
                                  <div className="asc-reason">{sig.reason}</div>
                                  <div className="asc-conf-row">
                                    <span className="asc-conf-label">AI Confidence</span>
                                    <span className="asc-conf-val">{sig.confidence}%</span>
                                  </div>
                                  <div className="asc-conf-bar">
                                    <div
                                      className={`asc-conf-fill ${sig.signal === 'BULLISH' ? 'bull' : 'bear'}`}
                                      style={{ width: `${sig.confidence}%` }}
                                    />
                                  </div>
                                </div>
                              ))}
                            </div>

                            {/* Quick Launch — gold hover */}
                            <div className="welcome-divider" style={{ marginTop: '20px' }}>
                              <div className="divider-line" />
                              <span className="divider-text">⚡ Quick Launch</span>
                              <div className="divider-line" />
                            </div>
                            <div className="quick-stocks-grid">
                              {[
                                { symbol: 'RELIANCE', label: 'Reliance' },
                                { symbol: 'TCS', label: 'TCS' },
                                { symbol: 'HDFCBANK', label: 'HDFC Bank' },
                                { symbol: 'INFY', label: 'Infosys' },
                                { symbol: 'ICICIBANK', label: 'ICICI Bank' },
                                { symbol: 'WIPRO', label: 'Wipro' },
                                { symbol: 'SBIN', label: 'SBI' },
                                { symbol: 'TATAMOTORS', label: 'Tata Motors' },
                                { symbol: 'ADANIENT', label: 'Adani Ent.' },
                                { symbol: 'BAJFINANCE', label: 'Bajaj Finance' },
                              ].map((stock, i) => (
                                <button
                                  key={stock.symbol}
                                  className="quick-chip pro-chip"
                                  style={{ animationDelay: `${0.5 + i * 0.04}s` }}
                                  onClick={() => handleSelectCompany({ symbol: stock.symbol, price: null })}
                                >
                                  <span className="qc-symbol pro-sym">{stock.symbol}</span>
                                  <span className="qc-label">{stock.label}</span>
                                </button>
                              ))}
                            </div>

                            {recentStocks.length > 0 && (
                              <div className="welcome-recent" style={{ marginTop: '16px' }}>
                                <span className="welcome-recent-label">Recent</span>
                                {recentStocks.slice(0, 5).map(s => (
                                  <div key={s.symbol} className="recent-chip" onClick={() => handleSelectCompany(s)}>
                                    {s.symbol}
                                  </div>
                                ))}
                              </div>
                            )}
                          </>
                        ) : (
                          /* ── FREE TRIAL WELCOME ── */
                          <>
                            <div className="welcome-header">
                              <div className="welcome-logo-mark">
                                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                  <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
                                </svg>
                              </div>
                              <div>
                                <h2 className="welcome-title">Market Pulse</h2>
                                <p className="welcome-sub">Live indices · Search any NSE symbol to begin AI forecasting</p>
                              </div>
                              <div className={`welcome-market-badge ${marketOpen ? 'open' : 'closed'}`}>
                                <span className="status-dot" />
                                {marketOpen ? 'NSE LIVE' : 'NSE CLOSED'}
                              </div>
                            </div>

                            <div className="index-cards-grid">
                              {marketIndices.length > 0 ? marketIndices.map((idx, i) => (
                                <div
                                  key={idx.name}
                                  className={`index-card ${idx.change_pct >= 0 ? 'bull' : 'bear'}`}
                                  style={{ animationDelay: `${i * 0.08}s` }}
                                  onClick={() => handleSelectCompany({ symbol: idx.name.replace(' ', ''), price: null })}
                                >
                                  <div className="ic-glow" />
                                  <div className="ic-top">
                                    <span className="ic-name">{idx.name}</span>
                                    <span className={`ic-badge ${idx.change_pct >= 0 ? 'bull' : 'bear'}`}>
                                      {idx.change_pct >= 0 ? '▲' : '▼'} {Math.abs(idx.change_pct).toFixed(2)}%
                                    </span>
                                  </div>
                                  <div className="ic-price">{idx.price.toLocaleString('en-IN')}</div>
                                  <div className="ic-bar">
                                    <div className="ic-bar-fill" style={{ width: `${Math.min(Math.abs(idx.change_pct) * 10, 100)}%` }} />
                                  </div>
                                </div>
                              )) : [1, 2, 3].map(i => (
                                <div key={i} className="index-card skeleton" style={{ animationDelay: `${i * 0.1}s` }}>
                                  <div className="skeleton-line" style={{ width: '60%', height: '12px', marginBottom: '12px' }} />
                                  <div className="skeleton-line" style={{ width: '80%', height: '24px' }} />
                                </div>
                              ))}
                            </div>

                            <div className="welcome-divider">
                              <div className="divider-line" />
                              <span className="divider-text">⚡ Quick Launch</span>
                              <div className="divider-line" />
                            </div>

                            <div className="quick-stocks-grid">
                              {[
                                { symbol: 'RELIANCE', label: 'Reliance' },
                                { symbol: 'TCS', label: 'TCS' },
                                { symbol: 'HDFCBANK', label: 'HDFC Bank' },
                                { symbol: 'INFY', label: 'Infosys' },
                                { symbol: 'ICICIBANK', label: 'ICICI Bank' },
                                { symbol: 'WIPRO', label: 'Wipro' },
                                { symbol: 'SBIN', label: 'SBI' },
                                { symbol: 'TATAMOTORS', label: 'Tata Motors' },
                                { symbol: 'ADANIENT', label: 'Adani Ent.' },
                                { symbol: 'BAJFINANCE', label: 'Bajaj Finance' },
                              ].map((stock, i) => (
                                <button
                                  key={stock.symbol}
                                  className="quick-chip"
                                  style={{ animationDelay: `${0.2 + i * 0.04}s` }}
                                  onClick={() => handleSelectCompany({ symbol: stock.symbol, price: null })}
                                >
                                  <span className="qc-symbol">{stock.symbol}</span>
                                  <span className="qc-label">{stock.label}</span>
                                </button>
                              ))}
                            </div>

                            {recentStocks.length > 0 && (
                              <div className="welcome-recent">
                                <span className="welcome-recent-label">Recent</span>
                                {recentStocks.slice(0, 5).map(s => (
                                  <div key={s.symbol} className="recent-chip" onClick={() => handleSelectCompany(s)}>
                                    {s.symbol}
                                  </div>
                                ))}
                              </div>
                            )}
                          </>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
            
            {/* Price Table Panel */}
            {prediction && (
              <div className="table-panel">
                <div className="panel-title">Forecast Data</div>
                <div className="table-wrapper">
                  <table className="prediction-grid">
                    <thead>
                      <tr>
                        <th>Date</th>
                        <th>Target Price</th>
                        <th>Change</th>
                        <th>AI Signal</th>
                      </tr>
                    </thead>
                    <tbody>
                      {prediction.dates.slice(prediction.prediction_start_index, prediction.prediction_start_index + 10).map((date, i) => {
                        const idx = prediction.prediction_start_index + i;
                        const price = prediction.prices[idx];
                        const prevPriceVal = prediction.prices[idx - 1];
                        if (price === undefined || prevPriceVal === undefined) return null;
                        const change = price - prevPriceVal;
                        const changePct = (change / prevPriceVal) * 100;

                        const isLockedRow = !isPro && i > 0;

                        return (
                          <tr key={date} className={`animate-fade-in-delayed ${isLockedRow ? 'locked-blur' : ''}`} style={{ animationDelay: `${i * 0.05}s` }}>
                            <td>{new Date(date).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })}</td>
                            <td style={{ color: 'var(--text-bright)' }}>{isLockedRow ? '₹₹,₹₹₹.₹₹' : price.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
                            <td style={{ color: change >= 0 && !isLockedRow ? 'var(--success)' : change < 0 && !isLockedRow ? 'var(--error)' : 'var(--text-muted)' }}>
                              {!isLockedRow ? `${change >= 0 ? '+' : ''}${changePct.toFixed(2)}%` : '---'}
                            </td>
                            <td>
                              {!isLockedRow ? (
                                <span className={`trend-badge ${change >= 0 ? 'bullish' : 'bearish'}`}>
                                  {change >= 0 ? 'BULL' : 'BEAR'}
                                </span>
                              ) : (
                                <span className="trend-badge" style={{background: 'var(--border)', color: 'var(--text-muted)'}}>LOCK</span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                      {!isPro && (
                        <tr className="unlock-row-cta">
                          <td colSpan={4}>
                            <button className="tv-btn-get-started" onClick={() => setShowPricing(true)}>
                              Unlock 10-Day Forecast
                            </button>
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </ErrorBoundary>
        </div>

        {/* Right Sidebar */}
        <div className="sidebar">
          {prediction && (
            <div className="widget">
              <div className="widget-title">Model Specifications</div>
              {prediction.warning ? (
                <p className="ai-summary" style={{ color: 'var(--warning)', fontWeight: 500 }}>
                  Forecast unavailable. Showing historical data only.
                </p>
              ) : (
                <p className="ai-summary">
                  TransAm architecture analyzing attention interactions across <b>{selectedTimeframe.toUpperCase()}</b> historical patterns.
                </p>
              )}
              <div className="stat-row">
                <span className="stat-label">Model Type</span>
                <span className="stat-value">Transformer (Multi-Head)</span>
              </div>
              <div className="stat-row">
                <span className="stat-label">Features Used</span>
                <span className="stat-value">Price, Volume, Tech Ind.</span>
              </div>
              <div className="stat-row">
                <span className="stat-label">Prediction Horizon</span>
                <span className="stat-value">10 Trading Days</span>
              </div>
              <div className="stat-row" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '8px', position: 'relative' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%' }} className={!isPro ? 'locked-blur' : ''}>
                  <span className="stat-label">Confidence Score</span>
                  <span className="stat-value" style={{
                    color: (prediction.confidence_score ?? 0) >= 65 ? 'var(--success)'
                         : (prediction.confidence_score ?? 0) >= 40 ? 'var(--warning)'
                         : 'var(--error)'
                  }}>
                    {prediction.confidence_score !== undefined ? `${prediction.confidence_score}%` : 'N/A'}
                  </span>
                </div>
                <div style={{ width: '100%', height: '6px', background: 'var(--border)', borderRadius: '3px', overflow: 'hidden' }} className={!isPro ? 'locked-blur' : ''}>
                  <div style={{
                    width: `${prediction.confidence_score ?? 0}%`,
                    height: '100%',
                    background: (prediction.confidence_score ?? 0) >= 65 ? 'var(--success)'
                              : (prediction.confidence_score ?? 0) >= 40 ? 'var(--warning)'
                              : 'var(--error)',
                    transition: 'width 0.6s ease'
                  }}></div>
                </div>
                {!isPro && (
                  <div className="pro-overlay-lock">
                    <button className="tv-btn-login" onClick={() => setShowPricing(true)} style={{fontSize: '0.8rem', padding: '4px 8px', background: 'rgba(41, 98, 255, 0.2)', borderRadius: '4px', border: '1px solid #2962FF', color: '#fff'}}>
                      🔒 Upgrade to Pro
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}

          {isPro && alerts.length > 0 && (
            <div className="widget">
              <div className="widget-title">Active Alerts</div>
              <div className="alerts-list">
                {alerts.map((a, i) => (
                  <div key={`${a.symbol}-${a.target}`} className={`alert-item ${a.active ? 'active' : 'triggered'}`}>
                    <div className="ai-info">
                      <span className="ai-sym">{a.symbol}</span>
                      <span className="ai-target">{a.type === 'above' ? '≥' : '≤'} ₹{a.target.toLocaleString('en-IN')}</span>
                    </div>
                    <div className="ai-actions">
                      {!a.active && <span className="ai-status">HIT</span>}
                      <button className="ai-del" onClick={() => removeAlert(a.symbol, a.target)}>×</button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {isPro && (
            <div className="widget">
              <div className="widget-title">Sector Heatmap</div>
              <div className="heatmap-grid">
                {[
                  { name: 'IT', change: 1.2 },
                  { name: 'BANK', change: -0.8 },
                  { name: 'AUTO', change: 2.1 },
                  { name: 'PHARMA', change: 0.5 },
                  { name: 'FMCG', change: -0.3 },
                  { name: 'METAL', change: 1.7 },
                  { name: 'MEDIA', change: -1.2 },
                  { name: 'REALTY', change: 0.9 },
                ].map((s, i) => (
                  <div 
                    key={s.name} 
                    className={`heatmap-cell ${s.change >= 0 ? 'bull' : 'bear'}`}
                    style={{ animationDelay: `${i * 0.05}s` }}
                  >
                    <span className="hm-name">{s.name}</span>
                    <span className="hm-val">{s.change >= 0 ? '+' : ''}{s.change}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="widget">
            <div className="widget-title">Market Overview</div>
            <div className="indices-list">
              {marketIndices.length > 0 ? marketIndices.map(idx => (
                <div className="index-row" key={idx.name}>
                  <span className="index-name">{idx.name}</span>
                  <div className="index-values">
                    <span className="index-price">{idx.price.toLocaleString('en-IN')}</span>
                    <span className={`index-change ${idx.change_pct >= 0 ? 'up' : 'down'}`}>
                      {idx.change_pct >= 0 ? '+' : ''}{idx.change_pct.toFixed(2)}%
                    </span>
                  </div>
                </div>
              )) : (
                <div className="index-row"><span className="index-name" style={{ color: 'var(--text-muted)' }}>Loading market data...</span></div>
              )}
            </div>
          </div>
          
          <div className="widget">
            <div className="widget-title">Recent History</div>
            <div className="recent-list">
              {recentStocks.length > 0 ? recentStocks.map(s => (
                <div key={s.symbol} className="recent-chip" onClick={() => handleSelectCompany(s)}>
                  {s.symbol}
                </div>
              )) : (
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>No recent searches</span>
              )}
            </div>
          </div>

        </div>
      </main>

      {/* Pricing Modal */}
      {showPricing && (
        <div className="modal-backdrop">
          <div className="auth-card modal-card" style={{maxWidth: '800px'}}>
            <button className="close-btn" onClick={() => setShowPricing(false)}>×</button>
            <h2 className="auth-heading" style={{marginBottom: '40px'}}>Upgrade to TrendMaster Pro</h2>
            
            <div style={{display: 'flex', gap: '20px'}}>
              <div style={{flex: 1, padding: '30px', background: 'var(--surface-bg)', border: '1px solid var(--border)', borderRadius: '12px'}}>
                <h3 style={{color: '#8c9bad', fontSize: '1.2rem'}}>Basic</h3>
                <div style={{fontSize: '2.5rem', color: '#fff', fontWeight: 'bold', margin: '20px 0'}}>$0<span style={{fontSize: '1rem', color: '#8c9bad'}}>/10 days</span></div>
                <ul className="feature-list" style={{marginBottom: '30px'}}>
                  <li style={{fontSize: '0.9rem'}}>10-Day Forecast Chart (Day 1 Unlocked)</li>
                  <li style={{fontSize: '0.9rem'}}>Historical Data Access</li>
                  <li style={{fontSize: '0.9rem'}}>Basic Charting</li>
                </ul>
                <button className="tv-btn-secondary-large" style={{width: '100%'}} disabled>Current Plan</button>
              </div>

              <div style={{flex: 1, padding: '30px', background: 'linear-gradient(145deg, #1e222d, #131722)', border: '2px solid #2962FF', borderRadius: '12px', position: 'relative'}}>
                <div style={{position: 'absolute', top: '-12px', right: '20px', background: '#2962FF', color: '#fff', padding: '4px 12px', borderRadius: '12px', fontSize: '0.8rem', fontWeight: 'bold'}}>MOST POPULAR</div>
                <h3 style={{color: '#fff', fontSize: '1.2rem'}}>Pro Terminal</h3>
                <div style={{fontSize: '2.5rem', color: '#fff', fontWeight: 'bold', margin: '20px 0'}}>$49<span style={{fontSize: '1rem', color: '#8c9bad'}}>/mo</span></div>
                <ul className="feature-list" style={{marginBottom: '30px'}}>
                  <li style={{fontSize: '0.9rem'}}>Full AI Forecast Horizon</li>
                  <li style={{fontSize: '0.9rem'}}>Real-time WebSockets</li>
                  <li style={{fontSize: '0.9rem'}}>AI Confidence Scores</li>
                  <li style={{fontSize: '0.9rem'}}>No Latency Limits</li>
                </ul>
                <button className="tv-btn-primary-large" style={{width: '100%'}} onClick={() => {
                  setIsPro(true);
                  setShowPricing(false);
                }}>Subscribe Now</button>
              </div>
            </div>
          </div>
        </div>
      )}
      {/* Add Alert Modal */}
      {showAddAlert && (
        <div className="custom-modal-overlay">
          <div className="custom-modal">
            <div className="modal-header">
              <h3>Set Price Alert for {prediction?.symbol}</h3>
              <button onClick={() => setShowAddAlert(false)}>×</button>
            </div>
            <div className="modal-body">
              <div className="form-group">
                <label>Target Price (₹)</label>
                <input 
                  type="number" 
                  value={alertForm.target} 
                  onChange={e => setAlertForm({...alertForm, target: e.target.value})}
                  placeholder="Enter price..."
                  autoFocus
                />
              </div>
              <div className="form-group">
                <label>Alert when price is:</label>
                <div className="toggle-group">
                  <button 
                    className={alertForm.type === 'above' ? 'active' : ''} 
                    onClick={() => setAlertForm({...alertForm, type: 'above'})}
                  >
                    Above or Equal
                  </button>
                  <button 
                    className={alertForm.type === 'below' ? 'active' : ''} 
                    onClick={() => setAlertForm({...alertForm, type: 'below'})}
                  >
                    Below or Equal
                  </button>
                </div>
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn-cancel" onClick={() => setShowAddAlert(false)}>Cancel</button>
              <button className="btn-save" onClick={() => {
                const target = parseFloat(alertForm.target);
                if (target > 0) {
                  addAlert(prediction!.symbol, target, alertForm.type);
                  setShowAddAlert(false);
                }
              }}>Set Alert</button>
            </div>
          </div>
        </div>
      )}
      {/* Footer */}
      <Footer isPro={isPro} wsStatus={wsStatus} />
    </div>
  );
}

// --- Sub-components ---

function Footer({ isPro, wsStatus }: { isPro: boolean, wsStatus: string }) {
  return (
    <footer className={`footer ${isPro ? 'pro-footer' : ''}`}>
      <div className="footer-content">
        <div className="footer-grid">
          <div className="footer-section brand-section">
            <div className="footer-logo">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
              </svg>
              TrendMaster <span>{isPro ? 'PRO' : 'FREE'}</span>
            </div>
            <p className="footer-desc">
              Empowering NSE traders with next-gen Transformer AI. Real-time patterns, 10-day forecasts, and high-confidence signals.
            </p>
            <div className="footer-socials">
              <span className="social-icon">𝕏</span>
              <span className="social-icon">in</span>
              <span className="social-icon">✉</span>
            </div>
          </div>

          <div className="footer-section">
            <h4>Platform</h4>
            <ul>
              <li><a href="#markets">Markets</a></li>
              <li><a href="#signals">AI Signals</a></li>
              <li><a href="#heatmap">Sector Heatmap</a></li>
              <li><a href="#alerts">Price Alerts</a></li>
            </ul>
          </div>

          <div className="footer-section">
            <h4>Resources</h4>
            <ul>
              <li><a href="#help">Help Center</a></li>
              <li><a href="#api">API Documentation</a></li>
              <li><a href="#blog">Market Insights</a></li>
              <li><a href="#status">System Status</a></li>
            </ul>
          </div>

          <div className="footer-section status-section">
            <h4>System Status</h4>
            <div className="status-pills">
              <div className="status-pill">
                <span className={`dot ${wsStatus === 'connected' ? 'online' : 'reconnecting'}`}></span>
                Live Feed: {wsStatus === 'connected' ? 'Stable' : 'Connecting...'}
              </div>
              <div className="status-pill">
                <span className="dot online"></span>
                AI Core: Operational
              </div>
              <div className="status-pill">
                <span className="dot online"></span>
                API: 12ms
              </div>
            </div>
          </div>
        </div>

        <div className="footer-bottom">
          <div className="footer-legal">
            <span>© 2026 TrendMaster AI. All rights reserved.</span>
            <div className="legal-links">
              <a href="#privacy">Privacy</a>
              <a href="#terms">Terms</a>
              <a href="#disclaimer">Disclaimer</a>
            </div>
          </div>
          <div className="footer-disclaimer">
            <b>Disclaimer:</b> Trading involves significant risk. AI predictions are based on historical patterns and are for educational purposes only. Always consult a financial advisor.
          </div>
        </div>
      </div>
    </footer>
  );
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
