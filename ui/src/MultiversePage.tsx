import React, { useState, useEffect, useCallback } from 'react';
import MultiverseChart from './MultiverseChart';
import ErrorBoundary from './ErrorBoundary';
import TopNav from './TopNav';
import './App.css'; 

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
}

const MultiversePage: React.FC = () => {
  const [query, setQuery] = useState<string>('');
  const [data, setData] = useState<MultiverseData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

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

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) fetchMultiverse(query.trim().toUpperCase());
  };

  return (
    <div className="App dark-theme">
      <TopNav activePage="multiverse" isPro={true} />
      
      <main className="dashboard">
        <div className="main-column">
          <div className="chart-panel">
            <div className="chart-header">
              <div className="stock-info">
                <div className="stock-symbol">
                  <span className="quantum-icon" style={{ color: 'var(--accent)', marginRight: '8px' }}>⚛</span>
                  {data?.symbol || (query || "Quantum Hub")}
                  <span className="ws-status live" style={{ marginLeft: '12px', fontSize: '0.65rem' }}>
                    <span className="pulse-dot"></span> STOCHASTIC ENGINE
                  </span>
                </div>
                <div className="stock-name">Multiverse Probability Explorer</div>
              </div>

              <div className="search-box">
                <form onSubmit={handleSearch} className="search-input-wrapper">
                  <div className="search-icon">🔍</div>
                  <input 
                    type="text" 
                    placeholder="Search NSE Symbol..." 
                    value={query}
                    onChange={(e) => setQuery(e.target.value.toUpperCase())}
                  />
                  <button 
                    type="submit" 
                    disabled={loading}
                    style={{
                      background: 'var(--accent)',
                      border: 'none',
                      color: 'white',
                      padding: '0 15px',
                      height: '100%',
                      borderRadius: '0 var(--radius-md) var(--radius-md) 0',
                      cursor: 'pointer',
                      fontSize: '0.8rem',
                      fontWeight: 600
                    }}
                  >
                    {loading ? '...' : 'SCAN'}
                  </button>
                </form>
              </div>
            </div>

            <div className="chart-container-wrapper" style={{ flex: 1, position: 'relative', minHeight: '500px' }}>
              <ErrorBoundary>
                {data ? (
                  <MultiverseChart data={data} />
                ) : (
                  <div className="empty-state">
                    {loading ? (
                       <div className="loader">
                          <div className="loader-spinner"></div>
                          <p>Sampling 128 Parallel Realities...</p>
                       </div>
                    ) : (
                       <div className="discovery-dashboard" style={{ padding: '40px', width: '100%' }}>
                          <div className="discovery-hero" style={{ textAlign: 'center', marginBottom: '40px' }}>
                             <h2 style={{ fontSize: '2rem', marginBottom: '10px' }}>Quantum Hub</h2>
                             <p style={{ color: 'var(--text-muted)', fontSize: '1rem', maxWidth: '600px', margin: '0 auto' }}>
                                Visualize market uncertainty using Monte Carlo sampling. Beyond deterministic forecasts, explore the probability of every possible outcome.
                             </p>
                          </div>

                          <div className="quick-scan-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px', maxWidth: '1000px', margin: '0 auto' }}>
                             {QUICK_SCANS.map(stock => (
                                <button 
                                  key={stock.symbol}
                                  onClick={() => fetchMultiverse(stock.symbol)}
                                  className="pro-glow-panel animate-fade-in"
                                  style={{
                                    background: 'var(--bg-elevated)',
                                    border: '1px solid var(--border)',
                                    padding: '24px',
                                    borderRadius: 'var(--radius-lg)',
                                    textAlign: 'left',
                                    cursor: 'pointer',
                                    transition: 'all 0.3s ease'
                                  }}
                                >
                                   <div style={{ color: 'var(--accent)', fontWeight: 800, fontSize: '1.2rem' }}>{stock.symbol}</div>
                                   <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginTop: '4px' }}>{stock.name}</div>
                                   <div style={{ marginTop: '16px', fontSize: '0.7rem', color: 'var(--text-dark)', textTransform: 'uppercase', letterSpacing: '1px' }}>Quick Scan →</div>
                                </button>
                             ))}
                          </div>

                          {error && (
                             <div className="error-box" style={{ textAlign: 'center', marginTop: '40px', color: 'var(--error)' }}>
                                <p>{error}</p>
                             </div>
                          )}
                       </div>
                    )}
                  </div>
                )}
              </ErrorBoundary>
            </div>
          </div>

          {data && (
            <div className="table-panel" style={{ borderTop: '1px solid var(--border)' }}>
              <div className="panel-title">Probability Insights</div>
              <div style={{ padding: '20px', color: 'var(--text-muted)', fontSize: '0.9rem', lineHeight: '1.6' }}>
                The <b>Most Likely</b> path (Cyan) represents the statistical mean of 128 Monte Carlo Dropout simulations. 
                The <b>Best Case</b> (Green) and <b>Worst Case</b> (Red) lines define the 95% confidence interval boundaries. 
                Wider clouds indicate periods of higher regime uncertainty where the Transformer attention is divided across multiple significant futures.
              </div>
            </div>
          )}
        </div>

        <div className="sidebar">
          {data && data.distribution && (
             <div className="widget animate-fade-in">
                <div className="widget-title">Intelligence Gauge</div>
                <div className="stat-row" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '12px' }}>
                   <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
                      <span className="stat-label">Regime Chaos Score</span>
                      <span className="stat-value" style={{ 
                         color: data.distribution.chaos_score > 5 ? 'var(--error)' : data.distribution.chaos_score > 2 ? 'var(--warning)' : 'var(--success)' 
                      }}>
                         {data.distribution.chaos_score.toFixed(2)} pts
                      </span>
                   </div>
                   <div style={{ width: '100%', height: '8px', background: 'var(--border)', borderRadius: '4px', overflow: 'hidden' }}>
                      <div style={{ 
                         width: `${Math.min(data.distribution.chaos_score * 10, 100)}%`, 
                         height: '100%', 
                         background: data.distribution.chaos_score > 5 ? 'var(--error)' : data.distribution.chaos_score > 2 ? 'var(--warning)' : 'var(--success)' 
                      }}></div>
                   </div>
                   <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                      Higher scores indicate extreme divergence in AI viewpoints.
                   </p>
                </div>
             </div>
          )}

          <div className="widget">
            <div className="widget-title">Engine Specifications</div>
            <p className="ai-summary">
              Stochastic inference engine utilizing <b>MC Dropout</b> layers to sample the model's epistemic uncertainty.
            </p>
            <div className="stat-row">
              <span className="stat-label">Sampling Method</span>
              <span className="stat-value">MC Dropout</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">Iteration Count</span>
              <span className="stat-value">128 Passes</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">Confidence Interval</span>
              <span className="stat-value">95% (2σ)</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">Device</span>
              <span className="stat-value">CPU / CUDA</span>
            </div>
          </div>

          <div className="widget animate-fade-in">
             <div className="widget-title">Strategic Scenarios</div>
             <div className="indices-list">
                <div className="legend-row" style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
                   <div style={{background: '#00F0FF', height: '4px', width: '24px', borderRadius: '2px'}}></div>
                   <span style={{ fontSize: '0.85rem', color: 'var(--text-main)' }}>Most Likely (Mean)</span>
                </div>
                <div className="legend-row" style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
                   <div style={{background: '#08BB81', height: '4px', width: '24px', borderRadius: '2px'}}></div>
                   <span style={{ fontSize: '0.85rem', color: 'var(--text-main)' }}>Best Case (Upper Bound)</span>
                </div>
                <div className="legend-row" style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
                   <div style={{background: '#F23645', height: '4px', width: '24px', borderRadius: '2px'}}></div>
                   <span style={{ fontSize: '0.85rem', color: 'var(--text-main)' }}>Worst Case (Lower Bound)</span>
                </div>
                <div className="legend-row" style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                   <div style={{background: 'rgba(41, 98, 255, 0.25)', height: '12px', width: '24px', borderRadius: '2px'}}></div>
                   <span style={{ fontSize: '0.85rem', color: 'var(--text-main)' }}>Probability Cloud</span>
                </div>
             </div>
          </div>

          <div className="widget animate-fade-in" style={{ animationDelay: '0.2s' }}>
            <div className="widget-title">Usage Tip</div>
            <p className="ai-summary" style={{ fontSize: '0.8rem' }}>
              Unlike a standard forecast, the Multiverse Explorer shows you the <b>risk bandwidth</b>. Use the Worst Case line as a guide for Stop Loss placement in volatile regimes.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
};

export default MultiversePage;
