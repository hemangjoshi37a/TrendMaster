import React, { useState, useEffect } from 'react';
import { useLocation, Link } from 'react-router-dom';
import ChaosChart from './ChaosChart';
import Footer from './Footer';
import TopNav from './TopNav';
import './ChaosSandbox.css';

interface PredictionData {
  symbol: string;
  company_name: string;
  dates: string[];
  prices: number[];
  prediction_start_index: number;
  confidence_score?: number;
}

const ChaosSandbox: React.FC = () => {
  const [symbol, setSymbol] = useState('RELIANCE');
  const [shockPct, setShockPct] = useState(0); // Slider value
  const [loadingBase, setLoadingBase] = useState(false);
  const [loadingShock, setLoadingShock] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [baseData, setBaseData] = useState<PredictionData | null>(null);
  const [shockData, setShockData] = useState<PredictionData | null>(null);
  const location = useLocation();
  const isPro = location.state?.isPro || false;

  // Track if a generic shock slider fetch is queued to debounce
  const [typingTimeout, setTypingTimeout] = useState<NodeJS.Timeout | null>(null);

  const fetchBasePrediction = async (sym: string) => {
    setLoadingBase(true);
    setError(null);
    try {
      const resp = await fetch(`/api/predict?stock_symbol=${sym}&period=1y`);
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || "Base prediction failed");
      }
      const data = await resp.json();
      setBaseData(data);
      // Wait to fetch shock data until base is done if neutral shock, just copy
      if (shockPct === 0) {
        setShockData(data);
      } else {
        fetchShockPrediction(sym, shockPct);
      }
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoadingBase(false);
    }
  };

  const fetchShockPrediction = async (sym: string, pct: number) => {
    setLoadingShock(true);
    try {
      const resp = await fetch(`/api/simulate?stock_symbol=${sym}&period=1y&shock_pct=${pct}&no_cache=true`);
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        if (!error) setError(err.detail || "Shock prediction failed");
        return;
      }
      const data = await resp.json();
      setShockData(data);
    } catch (e: any) {
      if (!error) setError(e.message);
    } finally {
      setLoadingShock(false);
    }
  };

  // Debounced shock updates
  const handleShockChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setShockPct(val);

    if (typingTimeout) clearTimeout(typingTimeout);

    setTypingTimeout(setTimeout(() => {
      if (val === 0 && baseData) {
         setShockData(baseData);
      } else {
         fetchShockPrediction(symbol, val);
      }
    }, 400));
  };

  const handleRunSimulation = () => {
    fetchBasePrediction(symbol);
  };

  useEffect(() => {
      // Clean up timeout
      return () => { if (typingTimeout) clearTimeout(typingTimeout); };
  }, [typingTimeout]);

  return (
    <div className="chaos-lab-wrapper dark-theme">
      <TopNav activePage="sandbox" isPro={isPro} />
      <div className="backtest-lab"> {/* Reusing some structural classes from backtest */}
        <div className="backtest-header chaos-header">
          <div className="backtest-title">
            <h1 style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span style={{ fontSize: '2rem' }}>🌀</span> Chaos Sandbox
            </h1>
            <p>Inject macroeconomic shocks and watch the AI instantly re-calibrate its 10-day forecast.</p>
          </div>
          
          <div className="backtest-controls chaos-controls">
            <div className="control-group">
              <label>Symbol</label>
              <input 
                type="text" 
                className="backtest-input" 
                value={symbol} 
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                placeholder="e.g. RELIANCE"
              />
            </div>
            <button 
              className="run-backtest-btn chaos-btn" 
              onClick={handleRunSimulation}
              disabled={loadingBase}
            >
              {loadingBase ? (
                <>
                  <span className="bt-spinner" style={{ width: '16px', height: '16px', borderWidth: '2px', borderColor: '#fff', borderTopColor: 'transparent' }}></span>
                  Initializing...
                </>
              ) : (
                <>
                   Load Baseline
                </>
              )}
            </button>
          </div>
        </div>

        <div className="chaos-grid">
          <div className="chaos-chart-section">
            {baseData && shockData ? (
              <div className="chart-container-bt">
                 <div className="chaos-chart-legend">
                    <span className="legend-item"><span className="dot" style={{background: '#787B86'}}></span> Baseline AI Forecast</span>
                    <span className="legend-item"><span className="dot" style={{background: shockPct >= 0 ? '#089981' : '#F23645'}}></span> Shocked Response Forecast</span>
                    {(loadingBase || loadingShock) && <span className="chaos-inferring">Inferring...</span>}
                 </div>
                 <ChaosChart baseData={baseData} shockData={shockData} shockPct={shockPct} />
              </div>
            ) : (
              <div className="backtest-empty">
                {loadingBase ? (
                  <div className="bt-loader">
                    <div className="bt-spinner"></div>
                    <p>Loading Deep Learning Matrix...</p>
                  </div>
                ) : error ? (
                  <div className="error-msg">
                     <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#f23645" strokeWidth="1.5">
                      <circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line>
                    </svg>
                    <h3>Simulation Failed</h3>
                    <p>{error}</p>
                  </div>
                ) : (
                  <>
                    <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
                      <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>
                    </svg>
                    <h3>System Ready</h3>
                    <p>Load a stock's baseline forecast to begin injecting market anomalies.</p>
                  </>
                )}
              </div>
            )}
          </div>

          <div className="chaos-sidebar">
             <div className="chaos-widget">
                <div className="widget-title">Market Anomaly Injector</div>
                <p style={{fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '24px', lineHeight: 1.5}}>
                    Force the AI to react to sudden price movement combinations by artificially moving the most recent closing price.
                </p>

                <div className="slider-group">
                   <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px'}}>
                       <label>Instant Price Shock</label>
                       <span className={`shock-badge ${shockPct > 0 ? 'bull' : shockPct < 0 ? 'bear' : ''}`}>
                          {shockPct > 0 ? '+' : ''}{shockPct}%
                       </span>
                   </div>
                   <input
                      type="range"
                      min="-30"
                      max="30"
                      step="0.5"
                      disabled={!baseData || (!isPro && shockPct !== 0 && loadingShock)}
                      value={shockPct}
                      onChange={handleShockChange}
                      className={`chaos-slider ${shockPct >= 0 ? 'chaos-slider-bull' : 'chaos-slider-bear'}`}
                   />
                   <div className="slider-labels">
                      <span>Crash (-30%)</span>
                      <span>0%</span>
                      <span>Euphoria (+30%)</span>
                   </div>
                </div>

                {!isPro && (
                   <div className="pro-lock-overlay">
                       <div className="lock-icon">🔒</div>
                       <p>Pro feature</p>
                       <button className="tv-btn-login" style={{fontSize: '0.7rem', padding: '4px 8px'}}>Upgrade</button>
                   </div>
                )}
             </div>

             <div className="chaos-widget analysis-widget">
                 <div className="widget-title">AI Reaction</div>
                 {baseData && shockData ? (
                     <p className="chaos-narrative">
                         {shockPct === 0 ? "The model is currently projecting its standard 10-day baseline based on unmanipulated recent indicators." :
                          shockPct < -5 ? `A heavy ${shockPct}% crash forces indicators like RSI to plunge into oversold territory. The Transformer model interprets this panic as ${shockData.prices[shockData.prices.length-1] > shockData.prices[shockData.prediction_start_index-1] ? 'a massive buying opportunity leading to a sharp V-shaped recovery.' : 'a confirmation of long-term structural breakdown, projecting continued bearishness.'}` :
                          shockPct > 5 ? `In response to the sudden ${shockPct}% euphoria, momentum indicators redline. The AI anticipates ${shockData.prices[shockData.prices.length-1] > shockData.prices[shockData.prediction_start_index-1] ? 'the breakout has sustained legs and projects further vertical movement.' : 'a classic bull trap and projects heavy sideways correction over the next 10 days.'}` :
                          `A mild ${shockPct}% oscillation causes the AI to adjust micro-momentum, slightly shifting the trajectory without altering the long-term trend.`
                         }
                     </p>
                 ) : (
                     <p style={{fontSize: '0.85rem', color: 'var(--text-muted)'}}>No anomalies injected yet.</p>
                 )}
             </div>
          </div>
        </div>
      </div>
      <Footer isPro={isPro} wsStatus="connected" />
    </div>
  );
};

export default ChaosSandbox;
