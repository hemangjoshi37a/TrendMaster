import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import TopNav from './TopNav';
import Footer from './Footer';
import './WealthArchitect.css';

interface StockRec {
  symbol: string;
  name: string;
  price: number;
  predicted_return: number;
  confidence: number;
  wealth_score: number;
  recommended_qty: number;
  total_cost: number;
  rationale: string;
  suggested_stop_loss?: number;
}

interface AdvisorResult {
  budget: number;
  total_allocated: number;
  recommendations: StockRec[];
}

const SECTORS = ['All', 'Banking', 'IT', 'Auto', 'Pharma', 'Energy', 'FMCG', 'Metal'];
const TYPES = [
  { id: 'Penny', label: 'Rebel (Small)', desc: 'Under ₹250' },
  { id: 'Mid', label: 'Warrior (Mid)', desc: '₹250 - ₹1500' },
  { id: 'Large', label: 'Guardian (Large)', desc: 'Above ₹1500' }
];

const HORIZONS = [
  { id: 10, label: 'Short-Term', desc: '10-Day Alpha Burst' },
  { id: 22, label: 'Mid-Term', desc: '1-Month Swing' },
  { id: 60, label: 'Long-Term', desc: '1-Quarter Trend' }
];

const WealthArchitect: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const isPro = location.state?.isPro || false;

  const [budget, setBudget] = useState<string>('50000');
  const [sector, setSector] = useState<string>('All');
  const [stockType, setStockType] = useState<string>('Mid');
  const [horizon, setHorizon] = useState<number>(10);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<AdvisorResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runAdvisor = async () => {
    if (!budget || isNaN(Number(budget))) {
      setError("Please enter a valid investment amount.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const resp = await fetch(`/api/wealth-advisor?budget=${budget}&sector=${sector}&stock_type=${stockType}&horizon=${horizon}`);
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || "Engine failed to locate suitable alpha. Try different filters.");
      }
      const data = await resp.json();
      setResult(data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  if (!isPro) {
    return (
      <div className="wealth-advisor-wrapper">
        <TopNav activePage="markets" isPro={false} />
        <div className="advisor-pro-locked">
           <div className="lock-icon-lg">🔐</div>
           <h1>AI Wealth Architect</h1>
           <p style={{color: 'var(--text-muted)', maxWidth: '500px', margin: '20px auto 40px'}}>
             This premium intelligence engine is reserved for Pro members only. 
             Upgrade for personalized stock picking and portfolio allocation.
           </p>
           <button 
             className="run-analysis-btn" 
             style={{margin: '0 auto'}}
             onClick={() => navigate('/dashboard', { state: { isPro: false } })}
           >
             Back to Dashboard
           </button>
        </div>
        <Footer isPro={false} wsStatus="disconnected" />
      </div>
    );
  }

  return (
    <div className="wealth-advisor-wrapper">
      <TopNav activePage="markets" isPro={true} />
      
      <div className="advisor-container">
        <div className="advisor-header">
           <div className="alpha-badge">Operational Alpha</div>
           <h1>AI Wealth Architect</h1>
           <p>Deploy capital with Transformer-powered precision. Enter your blueprints below.</p>
        </div>

        <div className="advisor-form-grid">
           {/* Budget Card */}
           <div className="advisor-card">
              <div className="card-title">
                 <span style={{fontSize: '1.2rem'}}>💰</span>
                 Investment Budget
              </div>
              <div className="budget-input-wrapper">
                 <span>₹</span>
                 <input 
                    type="text" 
                    className="budget-input" 
                    value={budget}
                    onChange={(e) => setBudget(e.target.value)}
                    placeholder="50,000"
                 />
              </div>
              <p style={{fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '16px'}}>
                 Total amount you wish to deploy across 3 recommendations.
              </p>
           </div>

           {/* Sector Card */}
           <div className="advisor-card">
              <div className="card-title">
                 <span style={{fontSize: '1.2rem'}}>🏗️</span>
                 Preferred Sector
              </div>
              <div className="options-list">
                 {SECTORS.map(s => (
                    <button 
                       key={s} 
                       className={`option-btn ${sector === s ? 'active' : ''}`}
                       onClick={() => setSector(s)}
                    >
                       {s}
                    </button>
                 ))}
              </div>
           </div>

           {/* Cap Card */}
           <div className="advisor-card">
              <div className="card-title">
                 <span style={{fontSize: '1.2rem'}}>🛡️</span>
                 Capitalization Persona
              </div>
              <div style={{display: 'flex', flexDirection: 'column', gap: '10px'}}>
                 {TYPES.map(t => (
                    <button 
                       key={t.id} 
                       className={`option-btn ${stockType === t.id ? 'active' : ''}`}
                       style={{textAlign: 'left', padding: '15px'}}
                       onClick={() => setStockType(t.id)}
                    >
                       <div style={{fontWeight: 700}}>{t.label}</div>
                       <div style={{fontSize: '0.7rem', opacity: 0.7}}>{t.desc}</div>
                    </button>
                 ))}
              </div>
           </div>

           {/* Horizon Card */}
           <div className="advisor-card">
              <div className="card-title">
                 <span style={{fontSize: '1.2rem'}}>⏳</span>
                 Investment Horizon
              </div>
              <div style={{display: 'flex', flexDirection: 'column', gap: '10px'}}>
                 {HORIZONS.map(h => (
                    <button 
                       key={h.id} 
                       className={`option-btn ${horizon === h.id ? 'active' : ''}`}
                       style={{textAlign: 'left', padding: '15px'}}
                       onClick={() => setHorizon(h.id)}
                    >
                       <div style={{fontWeight: 700}}>{h.label}</div>
                       <div style={{fontSize: '0.7rem', opacity: 0.7}}>{h.desc}</div>
                    </button>
                 ))}
              </div>
           </div>
        </div>

        <button 
           className="run-analysis-btn" 
           onClick={runAdvisor}
           disabled={loading}
        >
           {loading ? 'Synthesizing Alpha...' : 'Construct Portfolio'}
        </button>

        {loading && (
           <div className="scanning-alpha">
              <div className="scan-radar"></div>
              <h3>Analyzing Market Matrix...</h3>
              <p style={{color: 'var(--text-muted)'}}>The Transformer is scoring potential candidates based on predicted 10-day velocity.</p>
           </div>
        )}

        {error && (
           <div className="error-msg" style={{textAlign: 'center', marginTop: '40px'}}>
              <p style={{color: 'var(--error)'}}>⚠️ {error}</p>
           </div>
        )}

        {result && (
           <div className="results-reveal">
              <div className="results-header">
                 <h2>The Trinity Protocol</h2>
                 <div className="divider"></div>
                 <div style={{fontSize: '0.9rem', color: 'var(--text-muted)'}}>
                    Allocated: <b style={{color: '#fff'}}>₹{result.total_allocated.toLocaleString('en-IN')}</b>
                 </div>
              </div>

              <div className="recommendations-grid">
                 {result.recommendations.map((rec, i) => (
                    <div key={rec.symbol} className={`rec-card ${i === 0 ? 'top' : ''}`}>
                       {i === 0 && <div className="top-pick-badge">Primary Pick</div>}
                       <div className="rec-card-header">
                          <div className="rec-symbol">{rec.symbol}</div>
                          <div className="rec-name">{rec.name}</div>
                       </div>
                       
                       <div className="rec-body">
                          <div className="rec-stats">
                             <div className="stat-item">
                                <label>Target Gain</label>
                                <div className="val up">+{rec.predicted_return}%</div>
                             </div>
                             <div className="stat-item">
                                <label>AI Confidence</label>
                                <div className="val">{rec.confidence}%</div>
                             </div>
                             <div className="stat-item" style={{gridColumn: 'span 2'}}>
                                <label>Entry Price</label>
                                <div className="val">₹{rec.price.toLocaleString('en-IN')}</div>
                             </div>
                             {rec.suggested_stop_loss && (
                                <div className="stat-item" style={{gridColumn: 'span 2', borderTop: '1px solid rgba(246, 70, 93, 0.1)', paddingTop: '10px'}}>
                                   <label style={{color: '#F6465D'}}>🛡️ Guardian Stop-Loss</label>
                                   <div className="val" style={{color: '#F6465D'}}>₹{rec.suggested_stop_loss.toLocaleString('en-IN')}</div>
                                </div>
                             )}
                          </div>

                          <div className="rec-rationale">
                             {rec.rationale}
                          </div>
                       </div>

                       <div className="rec-footer">
                          <div>
                             <div className="qty-label">推奨数量</div>
                             <div className="qty-val">{rec.recommended_qty} Shares</div>
                          </div>
                          <div style={{textAlign: 'right'}}>
                             <div className="qty-label">Est. Cost</div>
                             <div style={{fontWeight: 700}}>₹{rec.total_cost.toLocaleString('en-IN')}</div>
                          </div>
                       </div>
                    </div>
                 ))}
              </div>
           </div>
        )}
      </div>

      <Footer isPro={isPro} wsStatus="disconnected" />
    </div>
  );
};

export default WealthArchitect;
