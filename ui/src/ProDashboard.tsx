import React, { useState, useEffect, useRef, useCallback } from 'react';
import LineChart from './LineChart';
import ErrorBoundary from './ErrorBoundary';
import './App.css';

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

function ProDashboard() {
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
  const [currentSymbol, setCurrentSymbol] = useState<string>('');

  const ws = useRef<WebSocket | null>(null);
  const isSelecting = useRef<boolean>(false);
  const reconnectTimeout = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef<number>(0);

  useEffect(() => {
    const saved = localStorage.getItem('recentStocks');
    if (saved) setRecentStocks(JSON.parse(saved));

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
          <div className="logo" onClick={() => window.location.href='/'} style={{cursor: 'pointer'}}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
            </svg>
            TrendMaster <span style={{marginLeft: '8px', background: 'rgba(41,98,255,0.2)', padding: '2px 8px', borderRadius: '4px', fontSize: '0.8rem', color: '#2962FF'}}>PRO</span>
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

      {/* Main Dashboard Layout */}
      <main className="dashboard" onClick={() => setShowSuggestions(false)}>
        
        {/* Left/Center Column - Chart & Table */}
        <div className="main-column">
          <ErrorBoundary>
            <div className="chart-panel pro-glow-panel">
              {prediction ? (
                <>
                  <div className="chart-header">
                    <div className="stock-info">
                      <div className="stock-symbol">
                        {prediction.symbol}
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
                    <LineChart data={prediction} />
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
                    <>
                      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="9" y1="21" x2="9" y2="9"></line>
                      </svg>
                      <h3>Terminal Ready</h3>
                      <p>Search for an NSE symbol to view AI-powered forecasts.</p>
                      
                      <div className="recent-list" style={{ marginTop: '24px', justifyContent: 'center' }}>
                        {recentStocks.slice(0, 5).map(s => (
                          <div key={s.symbol} className="recent-chip" onClick={() => handleSelectCompany(s)}>
                            {s.symbol}
                          </div>
                        ))}
                      </div>
                    </>
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

                        return (
                          <tr key={date} className="animate-fade-in-delayed" style={{ animationDelay: `${i * 0.05}s` }}>
                            <td>{new Date(date).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })}</td>
                            <td style={{ color: 'var(--text-bright)' }}>{price.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
                            <td style={{ color: change >= 0 ? 'var(--success)' : 'var(--error)' }}>
                              {change >= 0 ? '+' : ''}{changePct.toFixed(2)}%
                            </td>
                            <td>
                              <span className={`trend-badge ${change >= 0 ? 'bullish' : 'bearish'}`}>
                                {change >= 0 ? 'BULL' : 'BEAR'}
                              </span>
                            </td>
                          </tr>
                        );
                      })}
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
              <div className="stat-row" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '8px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
                  <span className="stat-label">Confidence Score</span>
                  <span className="stat-value" style={{
                    color: (prediction.confidence_score ?? 0) >= 65 ? 'var(--success)'
                         : (prediction.confidence_score ?? 0) >= 40 ? 'var(--warning)'
                         : 'var(--error)'
                  }}>
                    {prediction.confidence_score !== undefined ? `${prediction.confidence_score}%` : 'N/A'}
                  </span>
                </div>
                <div style={{ width: '100%', height: '6px', background: 'var(--border)', borderRadius: '3px', overflow: 'hidden' }}>
                  <div style={{
                    width: `${prediction.confidence_score ?? 0}%`,
                    height: '100%',
                    background: (prediction.confidence_score ?? 0) >= 65 ? 'var(--success)'
                              : (prediction.confidence_score ?? 0) >= 40 ? 'var(--warning)'
                              : 'var(--error)',
                    transition: 'width 0.6s ease'
                  }}></div>
                </div>
              </div>
            </div>
          )}

          {prediction && !prediction.warning && (
            <div className="widget animate-fade-in" style={{animationDelay: '0.2s'}}>
              <div className="widget-title">Deep Learning Signals <span style={{fontSize: '0.6rem', background: '#089981', color: '#fff', padding: '2px 4px', borderRadius: '4px'}}>LIVE</span></div>
              <div className="pro-widget-grid">
                <div className="pro-mini-card">
                  <span className="label">Social Sentiment</span>
                  <span className="value bull">82% Bullish</span>
                </div>
                <div className="pro-mini-card">
                  <span className="label">Inst. Flow</span>
                  <span className="value bull">+4.2B INR</span>
                </div>
                <div className="pro-mini-card">
                  <span className="label">Volatility (HV)</span>
                  <span className="value">14.6%</span>
                </div>
                <div className="pro-mini-card">
                  <span className="label">Delta Skew</span>
                  <span className="value bear">-0.15</span>
                </div>
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
    </div>
  );
}

export default ProDashboard;
