import React, { useState, useEffect, useRef, useCallback } from 'react';
import LineChart from './LineChart';
import ErrorBoundary from './ErrorBoundary';
import TopNav from './TopNav';
import Footer from './Footer';
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
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [recentStocks, setRecentStocks] = useState<RecentStock[]>([]);
  const [showSuggestions, setShowSuggestions] = useState<boolean>(false);
  const [selectedTimeframe, setSelectedTimeframe] = useState<TimeframePeriod>('1y');
  const [marketIndices, setMarketIndices] = useState<MarketIndex[]>([]);
  const [currentSymbol, setCurrentSymbol] = useState<string>('');


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
    const saved = localStorage.getItem('recentStocks');
    if (saved) setRecentStocks(JSON.parse(saved));

    fetchMarketOverview();
    const marketInterval = setInterval(fetchMarketOverview, 60000);

    return () => {
      clearInterval(marketInterval);
    };
  }, []);


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
                      <div className="stock-symbol" style={{ fontFamily: 'Outfit', fontWeight: 800 }}>
                        {prediction.symbol}
                        <span className="ws-status reconnecting" style={{background: 'var(--bg-pannel)', color: 'var(--text-dim)', border: '1px solid var(--glass-border)'}}>EOD DATA ONLY</span>
                      </div>
                      <div className="stock-name">{prediction.company_name}</div>
                    </div>
                    <div className="stock-price-container">
                      <div className="current-price" style={{ fontFamily: 'JetBrains Mono', fontWeight: 800 }}>
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
                        <div className="legend-color" style={{ background: 'var(--brand-primary)' }}></div>
                        Historical Data
                      </div>
                    </div>
                  </div>

                  {prediction.warning && (
                    <div style={{
                      margin: '0 20px 8px',
                      padding: '12px 16px',
                      background: 'var(--brand-primary-glow)',
                      border: '1px solid var(--brand-primary-glow)',
                      borderRadius: 'var(--radius-md)',
                      fontSize: '0.85rem',
                      color: 'var(--text-bright)',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '12px',
                      backdropFilter: 'var(--glass-blur)'
                    }}>
                      <span style={{ fontSize: '1.2rem' }}>💎</span>
                      <span style={{ flex: 1 }}>You are viewing basic End-Of-Day data. <a href="/#pricing" style={{color: 'var(--brand-primary)', fontWeight: 'bold', textDecoration: 'none'}}>Upgrade to Pro</a> for Live WebSockets and AI Forecasts.</span>
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
                      <div className="loader-spinner" style={{ borderColor: 'var(--brand-primary-glow)', borderTopColor: 'var(--brand-primary)' }}></div>
                      <p style={{ color: 'var(--text-muted)' }}>Loading market data for {currentSymbol}...</p>
                    </div>
                  ) : error ? (
                    <>
                      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--error)" strokeWidth="1.5">
                        <circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line>
                      </svg>
                      <h3 style={{ color: 'var(--text-bright)' }}>Analysis Failed</h3>
                      <p style={{ color: 'var(--text-muted)' }}>{error}</p>
                      {currentSymbol && (
                        <button
                          onClick={() => fetchPrediction(currentSymbol, selectedTimeframe)}
                          style={{
                            marginTop: '16px',
                            padding: '10px 24px',
                            background: 'var(--brand-primary)',
                            color: 'white',
                            border: 'none',
                            borderRadius: 'var(--radius-md)',
                            cursor: 'pointer',
                            fontWeight: 700,
                            fontSize: '0.9rem',
                            boxShadow: '0 4px 12px var(--brand-primary-glow)'
                          }}
                        >
                          ↺ Retry Analysis
                        </button>
                      )}
                    </>
                  ) : (
                    <>
                      <div className="empty-state-icon" style={{ marginBottom: '24px', opacity: 0.5, color: 'var(--brand-primary)' }}>
                        <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
                           <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="9" y1="21" x2="9" y2="9"></line>
                        </svg>
                      </div>
                      <h3 style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--text-bright)', fontFamily: 'Outfit' }}>Market Terminal Ready</h3>
                      <p style={{ color: 'var(--text-muted)', maxWidth: '400px' }}>Search for an NSE symbol or select a recent asset to begin your analysis.</p>
                      
                      <div className="recent-list" style={{ marginTop: '32px', justifyContent: 'center', gap: '12px' }}>
                        {recentStocks.slice(0, 5).map(s => (
                          <div key={s.symbol} className="recent-chip" onClick={() => {
                            setCurrentSymbol(s.symbol);
                            fetchPrediction(s.symbol, selectedTimeframe);
                          }} style={{ padding: '8px 16px', background: 'var(--bg-pannel)', border: '1px solid var(--glass-border)' }}>
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
                <div className="panel-title" style={{ fontFamily: 'Outfit', letterSpacing: '1px', fontWeight: 700 }}>HISTORICAL PERFORMANCE (5D)</div>
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
                            <td style={{ fontWeight: 600 }}>{new Date(date).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })}</td>
                            <td style={{ color: 'var(--text-bright)', fontFamily: 'JetBrains Mono', fontWeight: 700 }}>{price.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
                            <td style={{ color: change >= 0 ? 'var(--success)' : 'var(--error)', fontWeight: 800 }}>
                              {change >= 0 ? '▲' : '▼'} {Math.abs(changePct).toFixed(2)}%
                            </td>
                            <td>
                              <span style={{ color: 'var(--text-dim)', fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase' }}>Settled</span>
                            </td>
                          </tr>
                        );
                      })}
                      <tr>
                         <td colSpan={4} style={{textAlign: 'center', padding: '32px', color: 'var(--text-dim)', borderBottom: 'none', background: 'var(--bg-pannel-lighter)'}}>
                            <span style={{ fontSize: '1.2rem', marginRight: '8px' }}>🚀</span>
                            <a href="/#pricing" style={{color: 'var(--brand-primary)', textDecoration: 'none', fontWeight: 'bold'}}>Unlock 10-day AI forecasts</a> and real-time signals with TrendMaster Pro.
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
          
          <div className="widget" style={{ padding: 0, overflow: 'hidden' }}>
            <div className="upsell-banner" style={{ background: 'linear-gradient(135deg, var(--brand-primary) 0%, var(--brand-accent) 100%)', padding: '24px' }}>
              <h3 style={{ color: '#fff', fontSize: '1.2rem', fontWeight: 800 }}>Master the Markets</h3>
              <p style={{ color: 'rgba(255,255,255,0.8)', fontSize: '0.85rem', marginBottom: '20px' }}>Get access to our Deep Learning Transformer Models and exact entry/exit targets.</p>
              <button className="upgrade-btn" onClick={() => window.location.href='/#pricing'} style={{ width: '100%', padding: '12px', background: '#fff', color: 'var(--brand-primary)', border: 'none', borderRadius: 'var(--radius-sm)', fontWeight: 800, cursor: 'pointer' }}>
                Upgrade to Pro
              </button>
            </div>
          </div>

          <div className="widget">
            <div className="widget-title" style={{ fontFamily: 'Outfit', letterSpacing: '1px' }}>Market Overview</div>
            <div className="indices-list">
              {marketIndices.length > 0 ? marketIndices.map(idx => (
                <div className="index-row" key={idx.name}>
                  <span className="index-name" style={{ fontWeight: 600 }}>{idx.name}</span>
                  <div className="index-values">
                    <span className="index-price" style={{ fontFamily: 'JetBrains Mono' }}>{idx.price.toLocaleString('en-IN')}</span>
                    <span className={`index-change ${idx.change_pct >= 0 ? 'up' : 'down'}`}>
                      {idx.change_pct >= 0 ? '▲' : '▼'}{Math.abs(idx.change_pct).toFixed(2)}%
                    </span>
                  </div>
                </div>
              )) : (
                <div className="index-row"><span className="index-name" style={{ color: 'var(--text-muted)' }}>Loading market data...</span></div>
              )}
            </div>
          </div>
          
          <div className="widget">
            <div className="widget-title" style={{ fontFamily: 'Outfit', letterSpacing: '1px' }}>Recent History</div>
            <div className="recent-list">
              {recentStocks.length > 0 ? recentStocks.map(s => (
                <div key={s.symbol} className="recent-chip" onClick={() => {
                  setCurrentSymbol(s.symbol);
                  fetchPrediction(s.symbol, selectedTimeframe);
                }} style={{ background: 'var(--bg-pannel)', border: '1px solid var(--glass-border)' }}>
                  {s.symbol}
                </div>
              )) : (
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>No recent searches</span>
              )}
            </div>
          </div>

        </div>
      </main>
      <Footer isPro={false} wsStatus="disconnected" />
    </div>
  );
}

export default SimpleDashboard;
