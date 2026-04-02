import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import TopNav from './TopNav';
import Footer from './Footer';
import PortfolioAnalytics from './PortfolioAnalytics';
import './Portfolio.css';

interface Position {
  symbol: string;
  qty: number;
  avgPrice: number;
  takeProfit?: number;
  stopLoss?: number;
}

interface Transaction {
  id: string;
  symbol: string;
  type: 'BUY' | 'SELL';
  qty: number;
  price: number;
  date: string;
}

interface PaperTradingState {
  cash: number;
  positions: Position[];
  history: Transaction[];
}

const DEFAULT_CASH = 1000000;

const Portfolio: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const isPro = location.state?.isPro || false;

  const [state, setState] = useState<PaperTradingState>(() => {
    const saved = localStorage.getItem('tm_paper_trading');
    if (saved) return JSON.parse(saved);
    return { cash: DEFAULT_CASH, positions: [], history: [] };
  });

  const [livePrices, setLivePrices] = useState<{ [symbol: string]: number }>({});
  const [toast, setToast] = useState<{msg: string, type: 'success'|'error'} | null>(null);
  const [showAnalytics, setShowAnalytics] = useState(true);
  const [equityHistory, setEquityHistory] = useState<{ date: string; value: number }[]>(() => {
    const saved = localStorage.getItem('tm_equity_history');
    return saved ? JSON.parse(saved) : [];
  });

  useEffect(() => {
    localStorage.setItem('tm_paper_trading', JSON.stringify(state));
  }, [state]);

  useEffect(() => {
    localStorage.setItem('tm_equity_history', JSON.stringify(equityHistory));
  }, [equityHistory]);

  // Calculations
  let totalEquity = state.cash;
  let totalUnrealized = 0;
  state.positions.forEach(pos => {
    const currentPrice = livePrices[pos.symbol] || pos.avgPrice;
    totalEquity += pos.qty * currentPrice;
    totalUnrealized += (currentPrice - pos.avgPrice) * pos.qty;
  });

  const trackEquity = () => {
    const now = new Date();
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    setEquityHistory(prev => {
      const last = prev[prev.length - 1];
      if (last && last.value === totalEquity) return prev; // No change
      return [...prev, { date: timeStr, value: totalEquity }].slice(-50); // Keep last 50
    });
  };

  useEffect(() => {
    const timer = setTimeout(trackEquity, 2000); // Record shortly after load
    return () => clearTimeout(timer);
  }, [livePrices, totalEquity]); // Track on price updates or initial load

  const showToast = (msg: string, type: 'success' | 'error') => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3000);
  };

  // Poll live prices for all open positions
  useEffect(() => {
    if (state.positions.length === 0) return;
    const updatePrices = async () => {
      for (const pos of state.positions) {
        try {
          const resp = await fetch(`/api/quote?symbol=${encodeURIComponent(pos.symbol)}`);
          if (resp.ok) {
            const data = await resp.json();
            setLivePrices(prev => ({...prev, [pos.symbol]: data.price}));
          }
        } catch(e) {}
      }
    };
    updatePrices(); // initial fetch
    const interval = setInterval(updatePrices, 15000); // 15s polling
    return () => clearInterval(interval);
  }, [state.positions]);

  // Auto-Executor Loop: Evaluate bounds whenever live prices change
  useEffect(() => {
    state.positions.forEach(pos => {
      const currentPrice = livePrices[pos.symbol];
      if (!currentPrice) return;

      let triggered = false;
      let reason = '';

      if (pos.takeProfit && currentPrice >= pos.takeProfit) {
        triggered = true;
        reason = `Take Profit Triggered (≥ ₹${pos.takeProfit})`;
      } else if (pos.stopLoss && currentPrice <= pos.stopLoss) {
        triggered = true;
        reason = `Stop Loss Triggered (≤ ₹${pos.stopLoss})`;
      }

      if (triggered) {
        setState(prev => {
          const pIndex = prev.positions.findIndex(p => p.symbol === pos.symbol);
          if (pIndex === -1) return prev;
          
          const p = prev.positions[pIndex];
          const newCash = prev.cash + (p.qty * currentPrice);
          const newPositions = [...prev.positions];
          newPositions.splice(pIndex, 1);
          
          const tx: Transaction = {
            id: Math.random().toString(36).substr(2, 9),
            symbol: p.symbol,
            type: 'SELL',
            qty: p.qty,
            price: currentPrice,
            date: new Date().toISOString()
          };

          setTimeout(() => showToast(`${reason}: Auto-sold ${p.qty} ${p.symbol} at ₹${currentPrice.toFixed(2)}`, 'success'), 100);
          return { cash: newCash, positions: newPositions, history: [tx, ...prev.history].slice(0, 100) };
        });
      }
    });
  }, [livePrices, state.positions]);

  const handleSellAll = (symbol: string, currentPrice: number) => {
    setState(prev => {
      let newCash = prev.cash;
      let newPositions = [...prev.positions];
      
      const posIndex = newPositions.findIndex(p => p.symbol === symbol);
      if (posIndex === -1) return prev;
      
      const pos = newPositions[posIndex];
      const cost = pos.qty * currentPrice;
      
      newCash += cost;
      newPositions.splice(posIndex, 1);
      
      const tx: Transaction = {
        id: Math.random().toString(36).substr(2, 9),
        symbol: symbol,
        type: 'SELL',
        qty: pos.qty,
        price: currentPrice,
        date: new Date().toISOString()
      };

      showToast(`Instant Market SELL: ${pos.qty} ${symbol} Executed`, 'success');
      return { cash: newCash, positions: newPositions, history: [tx, ...prev.history].slice(0, 100) };
    });
  };

  return (
    <div className="portfolio-wrapper">
      <TopNav 
        activePage="portfolio" 
        isPro={isPro} 
        totalEquityOverride={totalEquity}
        totalUnrealizedOverride={totalUnrealized}
      />
      
      <div className="portfolio-container">
        <div className="portfolio-header">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '24px' }}>
            <div>
              <h1>My Portfolio</h1>
              <p>Real-time performance tracking of all your simulated holdings.</p>
            </div>
            <button 
              className={`analytics-toggle ${showAnalytics ? 'active' : ''}`}
              onClick={() => setShowAnalytics(!showAnalytics)}
            >
              {showAnalytics ? '🙈 Hide Analytics' : '📊 Show Analytics'}
            </button>
          </div>
        </div>

        {showAnalytics && (
          <PortfolioAnalytics 
            positions={state.positions} 
            livePrices={livePrices} 
            equityHistory={equityHistory} 
          />
        )}

        <div className="portfolio-metrics">
          <div className="metric-card">
            <div className="metric-card-glow"></div>
            <div className="metric-label">Total Equity</div>
            <div className="metric-value">₹{totalEquity.toLocaleString('en-IN', {maximumFractionDigits: 2})}</div>
          </div>
          <div className={`metric-card ${totalUnrealized >= 0 ? 'positive' : 'negative'}`}>
            <div className="metric-card-glow"></div>
            <div className="metric-label">Unrealized P&L</div>
            <div className="metric-value">
              {totalUnrealized >= 0 ? '+' : ''}₹{totalUnrealized.toLocaleString('en-IN', {maximumFractionDigits: 2})}
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-card-glow"></div>
            <div className="metric-label">Available Cash</div>
            <div className="metric-value">₹{state.cash.toLocaleString('en-IN', {maximumFractionDigits: 2})}</div>
          </div>
        </div>

        <div className="portfolio-table-container">
          <table className="portfolio-table">
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Shares</th>
                <th>Avg Cost</th>
                <th>Live Price</th>
                <th>Unrealized P&L</th>
                <th>Limits (TP/SL)</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {state.positions.length > 0 ? (
                state.positions.map(pos => {
                  const currentP = livePrices[pos.symbol] || pos.avgPrice;
                  const pnl = (currentP - pos.avgPrice) * pos.qty;
                  const pnlPct = ((currentP - pos.avgPrice) / pos.avgPrice) * 100;
                  
                  return (
                    <tr key={pos.symbol}>
                      <td>
                        <span className="portfolio-sym-badge" onClick={() => navigate('/paper-trading', { state: { isPro, symbol: pos.symbol }})}>
                          {pos.symbol}
                        </span>
                      </td>
                      <td>{pos.qty}</td>
                      <td>₹{pos.avgPrice.toLocaleString('en-IN', {maximumFractionDigits: 2})}</td>
                      <td>₹{currentP.toLocaleString('en-IN', {maximumFractionDigits: 2})}</td>
                      <td className={`portfolio-pnl ${pnl >= 0 ? 'bull' : 'bear'}`}>
                        {pnl >= 0 ? '+' : ''}{pnl.toLocaleString('en-IN', {maximumFractionDigits: 2})} ({pnlPct.toFixed(2)}%)
                      </td>
                      <td style={{fontSize: '0.8rem', color: '#848E9C'}}>
                         {pos.takeProfit ? `TP: ₹${pos.takeProfit.toLocaleString('en-IN')}` : ''}
                         {pos.takeProfit && pos.stopLoss ? <br/> : ''}
                         {pos.stopLoss ? `SL: ₹${pos.stopLoss.toLocaleString('en-IN')}` : ''}
                         {!pos.takeProfit && !pos.stopLoss ? '---' : ''}
                      </td>
                      <td>
                        <div style={{display: 'flex', gap: '8px'}}>
                          <button 
                            className="portfolio-sell-btn"
                            onClick={() => handleSellAll(pos.symbol, currentP)}
                          >
                            Instant Sell
                          </button>
                          <button 
                            style={{background: 'rgba(252, 213, 53, 0.1)', color: '#FCD535', border: '1px solid rgba(252, 213, 53, 0.3)', padding: '6px 12px', borderRadius: '4px', fontSize: '0.8rem', fontWeight: 700, cursor: 'pointer'}}
                            onClick={() => navigate('/paper-trading', { state: { isPro, symbol: pos.symbol }})}
                          >
                            Trade
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan={7}>
                    <div className="portfolio-empty">
                      <h3>No open positions</h3>
                      <p>Use the Sandbox or Live Terminal to find setups and begin Paper Trading.</p>
                      <button 
                        onClick={() => navigate('/paper-trading', { state: { isPro }})} 
                        style={{marginTop: '16px', background: 'var(--accent)', color: 'white', border: 'none', padding: '10px 20px', borderRadius: '8px', cursor: 'pointer', fontWeight: 600}}
                      >
                         Open Trading Terminal
                      </button>
                    </div>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
      
      {toast && (
        <div style={{
          position: 'fixed', bottom: '24px', right: '24px', zIndex: 9999,
          background: toast.type === 'success' ? '#0ECB81' : '#F6465D',
          color: 'white', padding: '12px 24px', borderRadius: '8px',
          fontWeight: 600, boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
          animation: 'fadeIn 0.3s ease'
        }}>
          {toast.msg}
        </div>
      )}
      
      <Footer isPro={isPro} wsStatus="connected" />
    </div>
  );
};

export default Portfolio;
