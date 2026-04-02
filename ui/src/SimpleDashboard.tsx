import React, { useState, useEffect, useRef, useCallback } from 'react';
import LineChart from './LineChart';
import ErrorBoundary from './ErrorBoundary';
import TopNav from './TopNav';
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

function SimpleDashboard() {
  const [query, setQuery] = useState<string>('');
  const [suggestions, setSuggestions] = useState<Company[]>([]);
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [recentStocks, setRecentStocks] = useState<RecentStock[]>([]);
  const [marketOpen, setMarketOpen] = useState<boolean>(true);
  const [showSuggestions, setShowSuggestions] = useState<boolean>(false);
  const [selectedTimeframe, setSelectedTimeframe] = useState<TimeframePeriod>('1y');
  const [marketIndices, setMarketIndices] = useState<MarketIndex[]>([]);
  const [currentSymbol, setCurrentSymbol] = useState<string>('');

  const isSelecting = useRef<boolean>(false);

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

  const fetchPrediction = useCallback(async (symbol: string, period: TimeframePeriod) => {
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await fetch(`/api/predict?stock_symbol=${encodeURIComponent(symbol)}&period=${period}`);
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || "Failed to fetch top data for " + symbol);
      }
      const data: PredictionData = await response.json();
      
      // Free users will just see history, not forecasts.
      data.warning = "Free Account";
      
      setPrediction(data);

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
  }, []);

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

  return (
    <div className="App dark-theme">
      {/* Top Navigation */}
      <TopNav 
        activePage="markets" 
        isPro={false} 
        searchElement={
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
        }
        rightActions={
          <div className="brand" style={{marginLeft: '16px'}}>
            <div className={`market-status ${marketOpen ? 'open' : 'closed'}`}>
              <span className="status-dot"></span>
              {marketOpen ? 'NSE OPEN' : 'NSE CLOSED'}
            </div>
          </div>
        }
      />

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
                        <span className="ws-status reconnecting" style={{background: 'rgba(140, 155, 173, 0.2)', color: '#8c9bad'}}>EOD DATA ONLY</span>
                      </div>
                      <div className="stock-name">{prediction.company_name}</div>
                    </div>
                    <div className="stock-price-container">
                      <div className="current-price">
                        {prediction.prices[prediction.prediction_start_index - 1]?.toFixed(2) || "---"}
                      </div>
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
                    </div>
                  </div>

                  {prediction.warning && (
                    <div style={{
                      margin: '0 20px 8px',
                      padding: '8px 12px',
                      background: 'rgba(41, 98, 255, 0.12)',
                      border: '1px solid rgba(41, 98, 255, 0.35)',
                      borderRadius: 'var(--radius-sm)',
                      fontSize: '0.78rem',
                      color: '#2962FF',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px'
                    }}>
                      <span>ℹ️</span>
                      <span>You are viewing basic End-Of-Day data. <a href="/#pricing" style={{color: '#2962FF', fontWeight: 'bold'}}>Upgrade to Pro</a> for Live WebSockets and AI Forecasts.</span>
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
                      <p>Loading market data for {query}...</p>
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
                      <h3>Basic Dashboard Ready</h3>
                      <p>Search for an NSE symbol to view historical action.</p>
                      
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
                <div className="panel-title">Historical Data (Last 5 Days)</div>
                <div className="table-wrapper">
                  <table className="prediction-grid">
                    <thead>
                      <tr>
                        <th>Date</th>
                        <th>Close Price</th>
                        <th>Change</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {prediction.dates.slice(Math.max(0, prediction.prediction_start_index - 6), prediction.prediction_start_index - 1).reverse().map((date, i) => {
                        const idx = prediction.dates.indexOf(date);
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
                              <span style={{ color: '#8c9bad' }}>Settled</span>
                            </td>
                          </tr>
                        );
                      })}
                      <tr>
                         <td colSpan={4} style={{textAlign: 'center', padding: '20px', color: '#8c9bad', borderBottom: 'none'}}>
                            🔒 <a href="/#pricing" style={{color: '#2962FF', textDecoration: 'none', fontWeight: 'bold'}}>Upgrade to Pro</a> to unlock 10-day AI forecasts and real-time signals.
                         </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </ErrorBoundary>
        </div>

        {/* Right Sidebar */}
        <div className="sidebar">
          
          <div className="widget">
            <div className="upsell-banner">
              <h3>Unlock AI Forecasts</h3>
              <p>Get access to our Deep Learning Transformer Models, exact entry/exit targets, and Live WebSocket feeds.</p>
              <button className="upgrade-btn" onClick={() => window.location.href='/#pricing'}>
                Upgrade to Pro
              </button>
            </div>
          </div>

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

export default SimpleDashboard;
