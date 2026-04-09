import React, { useState, useEffect, useCallback } from 'react';
import MultiverseChart from './MultiverseChart';
import ErrorBoundary from './ErrorBoundary';
import TopNav from './TopNav';
import RiskLab from './components/RiskLab';
import DeepScan from './components/DeepScan';
import './App.css'; 

interface RiskStats {
  var_95: number;
  es_95: number;
  kelly_fraction: number;
  pop: number;
}

interface MatrixItem {
  day: number;
  date: string;
  mean: number;
  upper: number;
  lower: number;
}

interface MultiverseData {
  symbol: string;
  dates: string[];
  prices: number[];
  cloud_upper?: number[];
  cloud_lower?: number[];
  prediction_start_index: number;
  is_stochastic: boolean;
  warning?: string;
  distribution?: {
     bins: number[];
     counts: number[];
     chaos_score: number;
  };
  risk_stats?: RiskStats;
  matrix?: MatrixItem[];
}

const MultiversePage: React.FC = () => {
  const [query, setQuery] = useState<string>('');
  const [data, setData] = useState<MultiverseData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  // Deep Scan State
  const [scanResults, setScanResults] = useState<any[] | null>(null);
  const [isScanning, setIsScanning] = useState<boolean>(false);

  const QUICK_SCANS = [
    { symbol: 'RELIANCE', name: 'Reliance Ind.' },
    { symbol: 'TCS', name: 'TCS' },
    { symbol: 'HDFCBANK', name: 'HDFC Bank' },
    { symbol: 'INFY', name: 'Infosys' },
    { symbol: 'SBIN', name: 'SBI' }
  ];

  const fetchMultiverse = useCallback(async (symbol: string) => {
    setLoading(true);
    setError(null);
    setScanResults(null); // Reset scan on new symbol
    try {
      const resp = await fetch(`/api/multiverse?stock_symbol=${encodeURIComponent(symbol)}`);
      if (!resp.ok) {
        let errorMsg = "Multiverse simulation failed";
        try {
          const err = await resp.json();
          errorMsg = err.detail || errorMsg;
        } catch (e) {
          errorMsg = `Server Error: ${resp.status} ${resp.statusText}`;
        }
        throw new Error(errorMsg);
      }
      const resData = await resp.json();
      setData(resData);
    } catch (e: any) {
      setError(e.message === "Failed to fetch" ? "Network Error: Could not connect to the simulation server." : e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const runDeepScan = async () => {
    if (!data?.symbol) return;
    setIsScanning(true);
    try {
      const resp = await fetch(`/api/multiverse/deep-scan?stock_symbol=${encodeURIComponent(data.symbol)}`);
      if (!resp.ok) throw new Error("Deep scan failed");
      const scanData = await resp.json();
      setScanResults(scanData.scan_results);
    } catch (e: any) {
      console.error(e);
    } finally {
      setIsScanning(false);
    }
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) fetchMultiverse(query.trim().toUpperCase());
  };

  return (
    <div className="App dark-theme">
      <TopNav activePage="multiverse" isPro={true} />
      
      <main className="dashboard">
        <div className="main-column" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {/* Hero / Header Section */}
          <section className="chart-panel pro-glow-outer">
            <div className="chart-header">
              <div className="stock-info">
                <div className="stock-symbol" style={{ letterSpacing: '2px' }}>
                  <span className="quantum-icon" style={{ color: 'var(--accent)', marginRight: '8px' }}>⚛</span>
                  {data?.symbol || (query || "Quantum Hub")}
                  <span className="ws-status live" style={{ marginLeft: '12px', fontSize: '0.65rem', border: '1px solid var(--accent)', padding: '2px 8px', borderRadius: '4px' }}>
                    <span className="pulse-dot"></span> PROBABILITY ENGINE ACTIVE
                  </span>
                </div>
                <div className="stock-name" style={{ opacity: 0.7 }}>Multiverse Risk Intelligence Suite</div>
              </div>

              <div className="search-box">
                <form onSubmit={handleSearch} className="search-input-wrapper" style={{ boxShadow: '0 0 20px rgba(0, 240, 255, 0.1)' }}>
                  <div className="search-icon">🔍</div>
                  <input 
                    type="text" 
                    placeholder="Analyze New Reality..." 
                    value={query}
                    onChange={(e) => setQuery(e.target.value.toUpperCase())}
                  />
                  <button 
                    type="submit" 
                    disabled={loading}
                    className="scan-btn"
                  >
                    {loading ? '...' : 'SCAN'}
                  </button>
                </form>
              </div>
            </div>

            <div className="chart-container-wrapper" style={{ flex: 1, position: 'relative', minHeight: '520px' }}>
              <ErrorBoundary>
                {data ? (
                  <MultiverseChart data={data} />
                ) : (
                  <div className="empty-state-quantum">
                    {loading ? (
                       <div className="loader-quantum">
                          <div className="spinner-beams"></div>
                          <p style={{ marginTop: '20px', letterSpacing: '4px', fontSize: '0.8rem' }}>SAMPLING PARALLEL FUTURES</p>
                       </div>
                    ) : (
                       <div className="discovery-dashboard" style={{ padding: '60px', width: '100%', textAlign: 'center' }}>
                          <div className="discovery-hero">
                             <div className="quantum-logo-large">⚛</div>
                             <h2 style={{ fontSize: '2.5rem', fontWeight: 900, marginBottom: '15px', background: 'linear-gradient(to bottom, #fff, #777)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>Quantum Risk Hub</h2>
                             <p style={{ color: 'var(--text-muted)', fontSize: '1.1rem', maxWidth: '700px', margin: '0 auto 50px', lineHeight: '1.8' }}>
                                Unlock professional-grade stochastic modeling. Beyond simple predictions, we quantify the unknown using Monte Carlo Dropout sampling across 128 parallel market realities.
                             </p>
                          </div>

                          <div className="quick-scan-grid-quantum">
                             {QUICK_SCANS.map(stock => (
                                <button key={stock.symbol} onClick={() => fetchMultiverse(stock.symbol)} className="quantum-chip">
                                   <div className="chip-symbol">{stock.symbol}</div>
                                   <div className="chip-name">{stock.name}</div>
                                   <div className="chip-action">INITIATE SCAN</div>
                                </button>
                             ))}
                          </div>

                          {error && <div className="error-glow">{error}</div>}
                       </div>
                    )}
                  </div>
                )}
              </ErrorBoundary>
            </div>
          </section>

          {/* Deep Intelligence Section */}
          {data && (
            <div className="intelligence-layer animate-slide-up">
              <RiskLab 
                stats={data.risk_stats || { var_95: 0, es_95: 0, kelly_fraction: 0, pop: 0 }} 
                matrix={data.matrix || []}
                symbol={data.symbol} 
              />
              
              <div className="backtest-trigger-panel" style={{ marginTop: '30px', textAlign: 'center' }}>
                 {!scanResults ? (
                    <button 
                      onClick={runDeepScan} 
                      disabled={isScanning}
                      className="deep-scan-trigger"
                    >
                       {isScanning ? 'CONDUCTING HISTORICAL SCAN...' : '⚡ CONDUCT HISTORICAL STOCHASTIC SCAN'}
                    </button>
                 ) : (
                    <DeepScan results={scanResults} symbol={data.symbol} />
                 )}
              </div>

              <div className="quantum-footer-disclaimer">
                The Multiverse Explorer utilizes MC Dropout layers (p=0.2) to simulate the model's epistemic uncertainty. 
                Average convergence achieved across 128 iterations. All risk metrics are theoretical projections.
              </div>
            </div>
          )}
        </div>

        <div className="sidebar">
          {data && data.distribution && (
             <div className="widget-quantum animate-fade-in">
                <div className="widget-title-quantum">Intelligence Gauge</div>
                <div className="chaos-meter-wrapper">
                   <div className="chaos-header">
                      <span className="chaos-label">Regime Entropy</span>
                      <span className="chaos-value" style={{ 
                         color: data.distribution.chaos_score > 5 ? '#F23645' : data.distribution.chaos_score > 2 ? '#FF9800' : '#08BB81' 
                      }}>
                         {data.distribution.chaos_score.toFixed(2)}
                      </span>
                   </div>
                   <div className="chaos-bar-bg">
                      <div className="chaos-bar-fill" style={{ 
                         width: `${Math.min(data.distribution.chaos_score * 10, 100)}%`, 
                         background: data.distribution.chaos_score > 5 ? '#F23645' : data.distribution.chaos_score > 2 ? '#FF9800' : '#08BB81' 
                      }}></div>
                   </div>
                   <p className="chaos-desc">
                      {data.distribution.chaos_score > 5 
                        ? 'High-entropy regime detected. AI attention is heavily divided between conflicting futures.' 
                        : 'Low-entropy state. High model conviction in the current trajectory.'}
                   </p>
                </div>
             </div>
          )}

          <div className="widget-quantum">
            <div className="widget-title-quantum">Engine Specs</div>
            <div className="spec-grid">
              <div className="spec-item"><span className="label">Algorithm</span><span className="val">MC Dropout</span></div>
              <div className="spec-item"><span className="label">Passes</span><span className="val">128</span></div>
              <div className="spec-item"><span className="label">Window</span><span className="val">30 Days</span></div>
              <div className="spec-item"><span className="label">Horizon</span><span className="val">10 Days</span></div>
            </div>
          </div>

          <div className="widget-quantum animate-fade-in">
             <div className="widget-title-quantum">Scenario Legend</div>
             <div className="legend-list-quantum">
                <div className="leg-item"><div className="dot" style={{background: '#00F0FF'}}></div><span>Most Likely</span></div>
                <div className="leg-item"><div className="dot" style={{background: '#08BB81'}}></div><span>Optimistic Bound</span></div>
                <div className="leg-item"><div className="dot" style={{background: '#F23645'}}></div><span>Pessimistic Bound</span></div>
                <div className="leg-item"><div className="cloud-box"></div><span>95% Confidence Cloud</span></div>
             </div>
          </div>

          <div className="ad-widget-quantum">
            <div className="shield-icon">🛡️</div>
            <h3>Capital Protection</h3>
            <p>Use the Worst Case line as your hard-stop boundary during High Entropy regimes.</p>
          </div>
        </div>
      </main>
      
      <style>{`
        .empty-state-quantum { min-height: 500px; display: flex; align-items: center; justify-content: center; }
        .quantum-logo-large { font-size: 4rem; color: var(--accent); margin-bottom: 20px; animation: rotate 10s linear infinite; }
        @keyframes rotate { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        
        .quick-scan-grid-quantum { display: grid; gridTemplateColumns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; max-width: 1000px; margin: 0 auto; }
        .quantum-chip { background: rgba(255,255,255,0.03); border: 1px solid var(--border); padding: 20px; border-radius: 12px; cursor: pointer; transition: all 0.3s; text-align: left; }
        .quantum-chip:hover { border-color: var(--accent); background: rgba(0, 240, 255, 0.05); transform: translateY(-3px); }
        .chip-symbol { font-weight: 800; color: var(--accent); font-size: 1.1rem; }
        .chip-name { font-size: 0.75rem; color: var(--text-muted); margin-top: 4px; }
        .chip-action { font-size: 0.6rem; color: var(--text-dark); margin-top: 12px; font-weight: 700; letter-spacing: 1px; }
        
        .deep-scan-trigger { width: 100%; padding: 20px; background: transparent; border: 1px dashed var(--accent); color: var(--accent); font-weight: 800; letter-spacing: 2px; border-radius: 12px; cursor: pointer; transition: all 0.3s; }
        .deep-scan-trigger:hover { background: rgba(0, 240, 255, 0.05); border-style: solid; box-shadow: 0 0 30px rgba(0, 240, 255, 0.1); }
        
        .widget-quantum { background: var(--bg-elevated); padding: 20px; border-radius: 16px; border: 1px solid var(--border); margin-bottom: 20px; }
        .widget-title-quantum { font-size: 0.75rem; font-weight: 800; color: var(--text-dark); textTransform: uppercase; margin-bottom: 15px; letter-spacing: 1px; }
        .spec-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .spec-item { display: flex; flex-direction: column; }
        .spec-item .label { font-size: 0.65rem; color: var(--text-muted); }
        .spec-item .val { font-size: 0.85rem; font-weight: 700; }
        
        .legend-list-quantum { display: flex; flex-direction: column; gap: 12px; }
        .leg-item { display: flex; align-items: center; gap: 10px; font-size: 0.8rem; }
        .leg-item .dot { width: 12px; height: 12px; border-radius: 3px; }
        .leg-item .cloud-box { width: 12px; height: 12px; border-radius: 3px; background: rgba(41, 98, 255, 0.2); }
        
        .ad-widget-quantum { background: linear-gradient(135deg, rgba(41, 98, 255, 0.1) 0%, transparent 100%); padding: 24px; border-radius: 16px; border: 1px solid rgba(41, 98, 255, 0.2); text-align: center; }
        .shield-icon { font-size: 2rem; margin-bottom: 10px; }
        .ad-widget-quantum h3 { font-size: 1rem; margin-bottom: 8px; color: var(--text-main); }
        .ad-widget-quantum p { font-size: 0.75rem; color: var(--text-muted); line-height: 1.5; }
        
        .error-glow { color: #F23645; padding: 20px; background: rgba(242, 54, 69, 0.1); border-radius: 8px; margin-top: 30px; border: 1px solid rgba(242, 54, 69, 0.2); text-shadow: 0 0 10px rgba(242, 54, 69, 0.3); }
        .quantum-footer-disclaimer { margin-top: 40px; padding: 20px; text-align: center; color: var(--text-dark); font-size: 0.7rem; line-height: 1.6; border-top: 1px solid var(--border); }
      `}</style>
    </div>
  );
};

export default MultiversePage;
