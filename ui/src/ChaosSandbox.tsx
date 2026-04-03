import React, { useState, useEffect, useCallback } from 'react';
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

interface Scenario {
  name: string;
  label: string;
  price: number;
  vix: number;
}

const SCENARIOS: Scenario[] = [
  { name: 'Baseline', label: 'Normal Market', price: 0, vix: 15 },
  { name: 'Flash Crash', label: 'Panic Selling', price: -15, vix: 45 },
  { name: 'Short Squeeze', label: 'Vertical Melt', price: 20, vix: 35 },
  { name: 'Bull Trap', label: 'Fading Rally', price: 5, vix: 25 },
  { name: 'Black Swan', label: 'Tail Risk', price: -25, vix: 65 },
];

const ChaosSandbox: React.FC = () => {
  const [symbol, setSymbol] = useState('RELIANCE');
  const [shockPct, setShockPct] = useState(0); 
  const [vixShock, setVixShock] = useState(15); 
  const [activeScenario, setActiveScenario] = useState('Baseline');
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
      if (shockPct === 0 && vixShock === 15) {
        setShockData(data);
      } else {
        fetchShockPrediction(sym, shockPct, vixShock);
      }
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoadingBase(false);
    }
  };

  const fetchShockPrediction = useCallback(async (sym: string, pct: number, vix: number) => {
    setLoadingShock(true);
    try {
      const resp = await fetch(`/api/simulate?stock_symbol=${sym}&period=1y&shock_pct=${pct}&vix=${vix}&no_cache=true`);
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
  }, [error]);

  const debouncedUpdate = useCallback((pct: number, vix: number) => {
    if (typingTimeout) clearTimeout(typingTimeout);
    setTypingTimeout(setTimeout(() => {
      if (pct === 0 && vix === 15 && baseData) {
        setShockData(baseData);
      } else {
        fetchShockPrediction(symbol, pct, vix);
      }
    }, 450));
  }, [baseData, fetchShockPrediction, symbol, typingTimeout]);

  const handleScenarioSelect = (scenario: Scenario) => {
    setActiveScenario(scenario.name);
    setShockPct(scenario.price);
    setVixShock(scenario.vix);
    if (baseData) {
       fetchShockPrediction(symbol, scenario.price, scenario.vix);
    }
  };

  const handleManualPriceChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setShockPct(val);
    setActiveScenario('Custom');
    debouncedUpdate(val, vixShock);
  };

  const handleManualVixChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setVixShock(val);
    setActiveScenario('Custom');
    debouncedUpdate(shockPct, val);
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
                    {(loadingBase || loadingShock) && <span className="chaos-inferring">Inferring Chaos...</span>}
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
                <div className="widget-title">Scenario Presets</div>
                <div className="scenario-presets">
                  {SCENARIOS.map(s => (
                    <button 
                      key={s.name} 
                      className={`preset-btn ${activeScenario === s.name ? 'active' : ''}`}
                      onClick={() => handleScenarioSelect(s)}
                      disabled={!baseData}
                    >
                      <span>{s.name}</span>
                      <span className="preset-label">{s.label}</span>
                    </button>
                  ))}
                </div>

                <div className="widget-title" style={{marginTop: '10px'}}>Manual Anomalies</div>
                <div className="slider-group">
                   <div className="shock-label-row">
                       <label>Price Shock</label>
                       <span className={`shock-badge ${shockPct > 0 ? 'bull' : shockPct < 0 ? 'bear' : ''}`}>
                          {shockPct > 0 ? '+' : ''}{shockPct}%
                       </span>
                   </div>
                   <input
                      type="range"
                      min="-40"
                      max="40"
                      step="1"
                      disabled={!baseData}
                      value={shockPct}
                      onChange={handleManualPriceChange}
                      className={`chaos-slider ${shockPct >= 0 ? 'chaos-slider-bull' : 'chaos-slider-bear'}`}
                   />

                   <div className="shock-label-row" style={{marginTop: '20px'}}>
                       <label>Volatility (VIX)</label>
                       <span className="shock-badge" style={{color: vixShock > 40 ? 'var(--error)' : vixShock > 25 ? 'var(--warning)' : 'var(--success)'}}>
                          {vixShock}
                       </span>
                   </div>
                   <input
                      type="range"
                      min="10"
                      max="80"
                      step="1"
                      disabled={!baseData}
                      value={vixShock}
                      onChange={handleManualVixChange}
                      className="chaos-slider chaos-slider-vix"
                   />
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
                     <div className="chaos-narrative">
                         <p style={{marginBottom: '10px'}}>
                           {shockPct === 0 && vixShock === 15 ? 
                              "The model is projecting its standard 10-day baseline. Volatility is within historical norms." :
                              vixShock > 50 ? 
                              `Extreme fear identified (VIX: ${vixShock}). The Transformer's attention mechanism is heavily weighted towards recent price volatility, leading to a highly erratic forecast.` :
                              `A ${shockPct}% shock with VIX at ${vixShock} forces the AI to recalibrate.`
                           }
                         </p>
                         <p>
                           {shockPct < -10 ? 
                            "The model interprets this plunge as structural weakness. Unless a V-shaped recovery is immediate, further decay is anticipated." :
                            shockPct > 10 ?
                            "The AI flags a potential short-squeeze scenario. Recent volume clusters suggest strong institutional buying at these 'shock' levels." :
                            "Scenario results indicate a return to mean after initial volatility oscillation."
                           }
                         </p>
                     </div>
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
