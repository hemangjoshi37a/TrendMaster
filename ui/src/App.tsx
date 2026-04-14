import React, { useState, useEffect, useCallback } from 'react';
import { BrowserRouter, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import LandingPage from './LandingPage';
import BacktestLab from './BacktestLab';
import ProDashboard from './ProDashboard';
import MultiversePage from './MultiversePage';
import NewsTerminal from './NewsTerminal';
import ChaosSandbox from './ChaosSandbox';
import Footer from './Footer';
import PaperTrading from './PaperTrading';
import Portfolio from './Portfolio';
import TopNav from './TopNav';
import WealthArchitect from './WealthArchitect';
import ErrorBoundary from './ErrorBoundary';

// Modular Services & Hooks
import { marketApi } from './services/marketApi';
import { useStockWS } from './hooks/useStockWS';
import { useMarketData } from './hooks/useMarketData';
import { useAccount } from './hooks/useAccount';
import { PredictionData, TimeframePeriod, Alert, RecentStock } from './types/market';

// Modular Components
import { WelcomeScreen } from './components/Dashboard/WelcomeScreen';
import { ChartPanel } from './components/Dashboard/ChartPanel';
import { ForecastTable } from './components/Dashboard/ForecastTable';
import { SidebarWidgets } from './components/Dashboard/SidebarWidgets';
import { ExpiryBanner, ExpiryOverlay } from './components/Dashboard/ExpiryBanner';

import './App.css';

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
  const location = useLocation();
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [livePrice, setLivePrice] = useState<number | null>(null);
  const [prevPrice, setPrevPrice] = useState<number | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [recentStocks, setRecentStocks] = useState<RecentStock[]>([]);
  const [selectedTimeframe, setSelectedTimeframe] = useState<TimeframePeriod>('1y');
  const [currentSymbol, setCurrentSymbol] = useState<string>('');
  
  // Modals & UI State
  const [showPricing, setShowPricing] = useState<boolean>(false);
  const [showAddAlert, setShowAddAlert] = useState<boolean>(false);
  const [alertForm, setAlertForm] = useState<{target: string, type: 'above' | 'below'}>({target: '', type: 'above'});
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [triggeredAlerts, setTriggeredAlerts] = useState<string[]>([]);
  const [showSuggestions, setShowSuggestions] = useState<boolean>(false);

  // Custom Hooks
  const { isPro, daysRemaining, isExpired } = useAccount();
  const { marketIndices, marketOpen } = useMarketData();
  
  const checkAlerts = useCallback((symbol: string, price: number) => {
    if (!isPro) return;
    alerts.forEach(alert => {
      if (alert.active && alert.symbol === symbol) {
        const isTriggered = alert.type === 'above' ? price >= alert.target : price <= alert.target;
        if (isTriggered) {
          const alertId = `${symbol}-${alert.target}`;
          setTriggeredAlerts(prev => Array.from(new Set([...prev, alertId])));
          const updated = alerts.map(a => 
            a.symbol === symbol && a.target === alert.target ? { ...a, active: false } : a
          );
          setAlerts(updated);
          localStorage.setItem('tm_alerts', JSON.stringify(updated));
        }
      }
    });
  }, [alerts, isPro]);

  const { wsStatus } = useStockWS(currentSymbol, (price) => {
    setLivePrice(prev => {
      setPrevPrice(prev);
      checkAlerts(currentSymbol.toUpperCase(), price);
      return price;
    });
  });

  const fetchPrediction = useCallback(async (symbol: string, period: TimeframePeriod) => {
    setLoading(true);
    setError(null);
    setPrediction(null);
    setLivePrice(null);
    setPrevPrice(null);
    try {
      const data = await marketApi.getPrediction(symbol, period);
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

  useEffect(() => {
    const saved = localStorage.getItem('recentStocks');
    if (saved) setRecentStocks(JSON.parse(saved));
    const savedAlerts = localStorage.getItem('tm_alerts');
    if (savedAlerts) setAlerts(JSON.parse(savedAlerts));
  }, []);

  useEffect(() => {
    const symbol = location.state?.searchSymbol || 'RELIANCE';
    setCurrentSymbol(symbol);
    fetchPrediction(symbol, selectedTimeframe);
  }, [location.state?.searchSymbol, selectedTimeframe, fetchPrediction]);

  const handleTimeframeChange = (period: TimeframePeriod) => {
    if (loading) return;
    setSelectedTimeframe(period);
  };

  const removeAlert = (symbol: string, target: number) => {
    const updated = alerts.filter(a => !(a.symbol === symbol && a.target === target));
    setAlerts(updated);
    localStorage.setItem('tm_alerts', JSON.stringify(updated));
  };

  const addAlert = (symbol: string, target: number, type: 'above' | 'below') => {
    if (!isPro) return;
    const newAlert: Alert = { symbol: symbol.toUpperCase(), target, type, active: true };
    const updated = [...alerts, newAlert];
    setAlerts(updated);
    localStorage.setItem('tm_alerts', JSON.stringify(updated));
  };

  return (
    <div className="App dark-theme">
      <TopNav activePage="markets" isPro={isPro} />
      
      <ExpiryBanner 
        isPro={isPro} 
        daysRemaining={daysRemaining} 
        isExpired={isExpired} 
        onUpgradeClick={() => setShowPricing(true)} 
      />

      {isPro && triggeredAlerts.length > 0 && (
        <div className="alert-notifications-bar">
          {triggeredAlerts.map(id => {
            const [sym, target] = id.split('-');
            return (
              <div key={id} className="alert-banner-item">
                <span className="ab-icon">🔔</span>
                <span className="ab-text">Target Hit: <b>{sym}</b> reached <b>₹{Number(target).toLocaleString('en-IN')}</b></span>
                <button className="ab-close" onClick={() => setTriggeredAlerts(prev => prev.filter(a => a !== id))}>×</button>
              </div>
            );
          })}
        </div>
      )}

      <ExpiryOverlay isPro={isPro} isExpired={isExpired} daysRemaining={daysRemaining} onUpgradeClick={() => setShowPricing(true)} />

      <main className="dashboard" onClick={() => setShowSuggestions(false)}>
        <div className="main-column">
          <ErrorBoundary>
            {prediction ? (
              <>
                <ChartPanel 
                  prediction={prediction}
                  isPro={isPro}
                  livePrice={livePrice}
                  prevPrice={prevPrice}
                  wsStatus={wsStatus}
                  alerts={alerts}
                  loading={loading}
                  selectedTimeframe={selectedTimeframe}
                  timeframeMap={TIMEFRAME_MAP}
                  onTimeframeChange={handleTimeframeChange}
                  onAlertClick={() => {
                    setAlertForm({ target: livePrice?.toString() || '', type: 'above' });
                    setShowAddAlert(true);
                  }}
                />
                <ForecastTable 
                    prediction={prediction} 
                    isPro={isPro} 
                    onUpgradeClick={() => setShowPricing(true)} 
                />
              </>
            ) : (
                <div className="empty-state">
                  {loading ? (
                    <div className="loader">
                      <div className="loader-spinner"></div>
                      <p>Running Transformer Model on {currentSymbol}...</p>
                    </div>
                  ) : error ? (
                    <>
                      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--error)" strokeWidth="1.5">
                        <circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line>
                      </svg>
                      <h3>Analysis Failed</h3>
                      <p>{error}</p>
                      <button onClick={() => fetchPrediction(currentSymbol, selectedTimeframe)} className="glass-pannel" style={{marginTop:'16px', padding:'10px 24px', background:'var(--brand-primary)', color:'white', border:'none', cursor:'pointer', fontWeight:700}}>↺ Retry Analysis</button>
                    </>
                  ) : (
                    <WelcomeScreen 
                      isPro={isPro}
                      marketOpen={marketOpen}
                      marketIndices={marketIndices}
                      recentStocks={recentStocks}
                      selectedTimeframe={selectedTimeframe}
                      onSymbolSelect={(sym) => {
                        setCurrentSymbol(sym);
                        fetchPrediction(sym, selectedTimeframe);
                      }}
                    />
                  )}
                </div>
            )}
          </ErrorBoundary>
        </div>

        <SidebarWidgets 
          prediction={prediction}
          selectedTimeframe={selectedTimeframe}
          isPro={isPro}
          marketIndices={marketIndices}
          recentStocks={recentStocks}
          alerts={alerts}
          onStockSelect={(sym) => {
              setCurrentSymbol(sym);
              fetchPrediction(sym, selectedTimeframe);
          }}
          onRemoveAlert={removeAlert}
          onUpgradeClick={() => setShowPricing(true)}
        />
      </main>

      {/* Pricing Modal */}
      {showPricing && (
        <div className="custom-modal-overlay" onClick={() => setShowPricing(false)}>
          <div className="custom-modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Upgrade to Pro</h3>
              <button onClick={() => setShowPricing(false)}>&times;</button>
            </div>
            <div className="modal-body">
              <p>Get the edge with our Premium Terminal:</p>
              <ul style={{ color: 'var(--text-muted)', fontSize: '0.9rem', paddingLeft: '20px' }}>
                <li>10-Day Full AI Forecast Horizon</li>
                <li>Real-time WebSocket Data Feed</li>
                <li>AI Confidence & Probability Scores</li>
                <li>Advanced Sector Heatmaps</li>
              </ul>
              <button className="btn-save" style={{ width: '100%' }} onClick={() => {
                marketApi.saveAccountMeta({ isPro: true, startDate: new Date().toISOString() });
                window.location.reload();
              }}>Subscribe Now — $49/mo</button>
            </div>
          </div>
        </div>
      )}

      {/* Add Alert Modal */}
      {showAddAlert && (
        <div className="custom-modal-overlay" onClick={() => setShowAddAlert(false)}>
          <div className="custom-modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Set Price Alert for {prediction?.symbol}</h3>
              <button onClick={() => setShowAddAlert(false)}>&times;</button>
            </div>
            <div className="modal-body">
              <div className="form-group">
                <label>Target Price (₹)</label>
                <input type="number" value={alertForm.target} onChange={e => setAlertForm({...alertForm, target: e.target.value})} />
              </div>
              <div className="toggle-group">
                <button className={alertForm.type === 'above' ? 'active' : ''} onClick={() => setAlertForm({...alertForm, type: 'above'})}>Above</button>
                <button className={alertForm.type === 'below' ? 'active' : ''} onClick={() => setAlertForm({...alertForm, type: 'below'})}>Below</button>
              </div>
              <button className="btn-save" onClick={() => {
                const target = parseFloat(alertForm.target);
                if (target > 0 && prediction) {
                  addAlert(prediction.symbol, target, alertForm.type);
                  setShowAddAlert(false);
                }
              }}>Set Alert</button>
            </div>
          </div>
        </div>
      )}

      <Footer isPro={isPro} wsStatus={wsStatus} />
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/wealth-architect" element={<WealthArchitect />} />
        <Route path="/backtest" element={<BacktestLab />} />
        <Route path="/pro" element={<ProDashboard />} />
        <Route path="/multiverse" element={<MultiversePage />} />
        <Route path="/sandbox" element={<ChaosSandbox />} />
        <Route path="/paper-trading" element={<PaperTrading />} />
        <Route path="/news" element={<NewsTerminal />} />
        <Route path="/portfolio" element={<Portfolio />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
