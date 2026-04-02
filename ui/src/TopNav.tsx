import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import './PaperTrading.css';

interface Position { symbol: string; qty: number; avgPrice: number; takeProfit?: number; stopLoss?: number; }
interface PaperTradingState { cash: number; positions: Position[]; history: any[]; }

interface TopNavProps {
  activePage: 'markets' | 'sandbox' | 'paper-trading' | 'backtest' | 'news' | 'portfolio';
  isPro: boolean;
  searchElement?: React.ReactNode;
  hidePnl?: boolean;
  totalEquityOverride?: number;
  totalUnrealizedOverride?: number;
  rightActions?: React.ReactNode;
}

const TopNav: React.FC<TopNavProps> = ({ 
  activePage, 
  isPro, 
  searchElement, 
  hidePnl = false, 
  totalEquityOverride, 
  totalUnrealizedOverride,
  rightActions
}) => {
  const [state, setState] = useState<PaperTradingState>(() => {
    const saved = localStorage.getItem('tm_paper_trading');
    if (saved) return JSON.parse(saved);
    return { cash: 1000000, positions: [], history: [] };
  });
  const [livePrices, setLivePrices] = useState<{ [symbol: string]: number }>({});

  useEffect(() => {
    // Only poll if we aren't overriding the PNL and there are positions
    if (state.positions.length === 0 || hidePnl || totalEquityOverride !== undefined) return;
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
    updatePrices();
    const interval = setInterval(updatePrices, 15000);
    return () => clearInterval(interval);
  }, [state.positions, hidePnl, totalEquityOverride]);

  let totalEquity = state.cash;
  let totalUnrealized = 0;

  if (totalEquityOverride !== undefined && totalUnrealizedOverride !== undefined) {
      totalEquity = totalEquityOverride;
      totalUnrealized = totalUnrealizedOverride;
  } else if (!hidePnl) {
      state.positions.forEach(pos => {
        const currentPrice = livePrices[pos.symbol] || pos.avgPrice;
        const posValue = pos.qty * currentPrice;
        totalEquity += posValue;
        totalUnrealized += (currentPrice - pos.avgPrice) * pos.qty;
      });
  }

  return (
    <nav className="pt-pro-nav" style={{ position: 'sticky', top: 0, zIndex: 1000 }}>
      <div className="pt-nav-left">
        <Link to="/" className="pt-brand">TrendMaster</Link>
        <div className="pt-nav-links">
          <Link to="/dashboard" className={activePage === 'markets' ? 'active' : ''} state={{ isPro }}>Markets</Link>
          <Link to="/news" className={activePage === 'news' ? 'active' : ''} state={{ isPro }}>News</Link>
          <Link to="/sandbox" className={activePage === 'sandbox' ? 'active' : ''} state={{ isPro }}>Sandbox</Link>
          <Link to="/paper-trading" className={activePage === 'paper-trading' ? 'active' : ''} state={{ isPro }}>Paper Trading</Link>
          <Link to="/portfolio" className={activePage === 'portfolio' ? 'active' : ''} state={{ isPro }}>Portfolio</Link>
          {isPro && <Link to="/backtest" className={activePage === 'backtest' ? 'active' : ''} state={{ isPro }}>Backtest Lab</Link>}
        </div>
      </div>
      
      <div className="pt-nav-right">
         {searchElement}
         {!hidePnl && (
             <div className="pt-account-chip" style={{ background: 'var(--surface-bg)', border: '1px solid var(--border)', borderRadius: '6px', padding: '6px 12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ color: 'var(--text-muted)', fontSize: '0.7rem', fontWeight: 600, letterSpacing: '0.5px' }}>EQUITY</span>
                <span style={{ fontWeight: 700, color: '#EAECEF', fontSize: '0.9rem' }}>₹{totalEquity.toLocaleString('en-IN', {maximumFractionDigits: 0})}</span>
                <div style={{ width: '1px', height: '14px', background: 'var(--border)', margin: '0 4px' }} />
                <span style={{ color: 'var(--text-muted)', fontSize: '0.7rem', fontWeight: 600, letterSpacing: '0.5px' }}>PNL</span>
                <span style={{ color: totalUnrealized >= 0 ? '#0ECB81' : '#F6465D', fontWeight: 700, fontSize: '0.9rem' }}>
                  {totalUnrealized >= 0 ? '+' : ''}₹{totalUnrealized.toLocaleString('en-IN', {maximumFractionDigits: 0})}
                </span>
             </div>
         )}
         {rightActions}
      </div>
    </nav>
  );
};

export default TopNav;
