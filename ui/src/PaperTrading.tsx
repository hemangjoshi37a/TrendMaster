import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useLocation, Link, useNavigate } from 'react-router-dom';
import LineChart from './LineChart';
import './PaperTrading.css';

interface Position {
  symbol: string;
  qty: number;
  avgPrice: number;
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

interface PredictionData {
  symbol: string;
  company_name: string;
  dates: string[];
  prices: number[];
  prediction_start_index: number;
  confidence_score?: number;
  warning?: string;
}

const DEFAULT_CASH = 1000000;

const PaperTrading: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const isPro = location.state?.isPro || false;

  const [state, setState] = useState<PaperTradingState>(() => {
    const saved = localStorage.getItem('tm_paper_trading');
    if (saved) return JSON.parse(saved);
    return { cash: DEFAULT_CASH, positions: [], history: [] };
  });

  const [livePrices, setLivePrices] = useState<{ [symbol: string]: number }>({});
  
  // Terminal State
  const [searchInput, setSearchInput] = useState('RELIANCE');
  const [activeSymbol, setActiveSymbol] = useState('RELIANCE');
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [loadingPred, setLoadingPred] = useState(false);
  
  // Order State
  const [orderSide, setOrderSide] = useState<'BUY' | 'SELL'>('BUY');
  const [orderQty, setOrderQty] = useState<number | ''>('');
  
  // UI State
  const [activeTab, setActiveTab] = useState<'positions' | 'history'>('positions');
  const [toast, setToast] = useState<{msg: string, type: 'success'|'error'} | null>(null);

  const ws = useRef<WebSocket | null>(null);
  const [liveQuote, setLiveQuote] = useState<number | null>(null);
  const [prevQuote, setPrevQuote] = useState<number | null>(null);

  useEffect(() => {
    localStorage.setItem('tm_paper_trading', JSON.stringify(state));
  }, [state]);

  const showToast = (msg: string, type: 'success' | 'error') => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3000);
  };

  const connectWebSocket = useCallback((symbol: string) => {
    if (ws.current) ws.current.close();

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const socket = new WebSocket(`${protocol}//${host}/ws/ticks/${symbol.toUpperCase()}`);

    socket.onopen = () => console.log('WS Connected:', symbol);
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.price) {
        setLiveQuote(prev => {
          setPrevQuote(prev);
          // Auto update the background dictionary too for P&L tracking
          setLivePrices(lp => ({...lp, [symbol]: data.price}));
          return data.price;
        });
      }
    };

    socket.onclose = () => console.log('WS Closed');
    ws.current = socket;
  }, []);

  const loadSymbolData = useCallback(async (symbol: string) => {
    const symUpper = symbol.trim().toUpperCase();
    if (!symUpper) return;
    
    setSearchInput(symUpper);
    setActiveSymbol(symUpper);
    setLoadingPred(true);
    setLiveQuote(null);
    setPrevQuote(null);
    setPrediction(null);
    setOrderQty('');

    try {
      // Fetch 10-day AI prediction to power the chart
      const resp = await fetch(`/api/predict?stock_symbol=${encodeURIComponent(symUpper)}&period=6mo`);
      if (!resp.ok) throw new Error("Symbol not found or data missing.");
      const data = await resp.json();
      setPrediction(data);
      
      // Connect WS for live price for order entry
      connectWebSocket(symUpper);
    } catch (e: any) {
      showToast(e.message, 'error');
    } finally {
      setLoadingPred(false);
    }
  }, [connectWebSocket]);

  // Initial load
  useEffect(() => {
    loadSymbolData('RELIANCE');
    return () => { if (ws.current) ws.current.close(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Background polling for all OTHER positions
  useEffect(() => {
    if (state.positions.length === 0) return;
    const updatePrices = async () => {
      for (const pos of state.positions) {
        // Skip current active symbol to preserve fast socket updates
        if (pos.symbol === activeSymbol) continue; 
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
  }, [state.positions, activeSymbol]);

  const executeTrade = () => {
    const currentPrice = liveQuote || (prediction ? prediction.prices[prediction.prediction_start_index - 1] : 0);
    if (!currentPrice || !orderQty || orderQty <= 0) return;
    
    const qty = Number(orderQty);
    const cost = qty * currentPrice;

    setState(prev => {
      let newCash = prev.cash;
      let newPositions = [...prev.positions];
      
      const posIndex = newPositions.findIndex(p => p.symbol === activeSymbol);
      const pos = posIndex >= 0 ? newPositions[posIndex] : null;

      if (orderSide === 'BUY') {
        if (cost > newCash) {
          showToast("Insufficient Buying Power!", 'error');
          return prev;
        }
        newCash -= cost;
        if (pos) {
          const totalCost = (pos.qty * pos.avgPrice) + cost;
          pos.qty += qty;
          pos.avgPrice = totalCost / pos.qty;
        } else {
          newPositions.push({ symbol: activeSymbol, qty, avgPrice: currentPrice });
        }
        showToast(`Market BUY ${qty} ${activeSymbol} Executed`, 'success');
      } else {
        if (!pos || pos.qty < qty) {
          showToast("Insufficient Shares to Sell!", 'error');
          return prev;
        }
        newCash += cost;
        pos.qty -= qty;
        if (pos.qty === 0) {
          newPositions.splice(posIndex, 1);
        }
        showToast(`Market SELL ${qty} ${activeSymbol} Executed`, 'success');
      }

      const tx: Transaction = {
        id: Math.random().toString(36).substr(2, 9),
        symbol: activeSymbol,
        type: orderSide,
        qty,
        price: currentPrice,
        date: new Date().toISOString()
      };

      setOrderQty(''); 
      return { cash: newCash, positions: newPositions, history: [tx, ...prev.history].slice(0, 100) };
    });
  };

  // Calculations
  let totalEquity = state.cash;
  let totalUnrealized = 0;

  state.positions.forEach(pos => {
    // If it's the active symbol, liveQuote is most accurate, otherwise use polled livePrices
    const currentPrice = (pos.symbol === activeSymbol && liveQuote) ? liveQuote : (livePrices[pos.symbol] || pos.avgPrice);
    const posValue = pos.qty * currentPrice;
    totalEquity += posValue;
    totalUnrealized += (currentPrice - pos.avgPrice) * pos.qty;
  });

  const activePrice = liveQuote || (prediction ? prediction.prices[prediction.prediction_start_index - 1] : 0);
  const activeChange = liveQuote && prevQuote ? liveQuote - prevQuote : 0;
  const activeChangePct = prevQuote && liveQuote ? (activeChange / prevQuote) * 100 : 0;
  
  // 24h simulation fallback
  let dayChange = 0;
  let dayChangePct = 0;
  if (prediction && prediction.prediction_start_index >= 2) {
      const yesterday = prediction.prices[prediction.prediction_start_index - 2];
      const today = activePrice || prediction.prices[prediction.prediction_start_index - 1];
      dayChange = today - yesterday;
      dayChangePct = (dayChange / yesterday) * 100;
  }

  const activePos = state.positions.find(p => p.symbol === activeSymbol);
  const maxBuyShares = activePrice > 0 ? Math.floor(state.cash / activePrice) : 0;

  return (
    <div className="pt-pro-wrapper">
      {/* Top Navbar */}
      <nav className="pt-pro-nav">
        <div className="pt-nav-left">
          <Link to="/" className="pt-brand">TrendMaster</Link>
          <div className="pt-nav-links">
            <Link to="/dashboard">Markets</Link>
            <Link to="/sandbox">Sandbox</Link>
            <Link to="/paper-trading" className="active">Paper Trading</Link>
            {isPro && <Link to="/backtest">Backtest Lab</Link>}
          </div>
        </div>
        
        <div className="pt-nav-right">
           <input 
              type="text"
              className="pt-search-input"
              value={searchInput}
              onChange={e => setSearchInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && loadSymbolData(searchInput)}
              placeholder="Search ticker..."
           />
           <div className="pt-account-chip">
              💰 ₹{totalEquity.toLocaleString('en-IN', {maximumFractionDigits: 0})} PNL: <span style={{color: totalUnrealized >= 0 ? '#0ECB81' : '#F6465D'}}>{totalUnrealized >= 0 ? '+' : ''}₹{totalUnrealized.toLocaleString('en-IN', {maximumFractionDigits: 0})}</span>
           </div>
           <button onClick={() => navigate('/dashboard')} style={{background: 'transparent', color: '#848E9C', border:'none', cursor:'pointer', fontSize: '1rem', fontWeight: 700}}>×</button>
        </div>
      </nav>

      {/* Main Trading Workspace */}
      <div className="pt-pro-workspace">
        
        {/* Left Column (Chart + Positions) */}
        <div className="pt-col-main">
          
          {/* Active Ticker Info Header */}
          <div className="pt-ticker-header">
             <div className="pt-ticker-symbol">
                {activeSymbol}
                <div className="pt-ticker-name">{prediction?.company_name || 'Loading...'}</div>
             </div>
             
             <div className="pt-ticker-price-block">
                <span className={`pt-ticker-price-num ${dayChange < 0 ? 'bear' : ''}`}>
                    ₹{activePrice > 0 ? activePrice.toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2}) : '0.00'}
                </span>
             </div>

             <div className="pt-ticker-stat">
                <span className="pt-stat-label">24h Change</span>
                <span className={`pt-stat-val ${dayChange >= 0 ? 'bull' : 'bear'}`}>
                   {dayChange >= 0 ? '+' : ''}{dayChangePct.toFixed(2)}%
                </span>
             </div>

             <div className="pt-ticker-stat" style={{marginLeft: 'auto', textAlign: 'right'}}>
                <span className="pt-stat-label">AI Confidence</span>
                <span className="pt-stat-val" style={{color: '#FCD535'}}>{prediction?.confidence_score || '--'}%</span>
             </div>
          </div>

          {/* Chart Area */}
          <div className="pt-chart-container">
             <div className="pt-chart-wrapper">
                 {loadingPred && (
                     <div className="pt-chart-overlay">
                         <div className="bt-spinner" style={{borderColor: '#FCD535'}}></div>
                         <div>Deep Learning Inference in progress...</div>
                     </div>
                 )}
                 {!loadingPred && prediction && (
                     <LineChart data={prediction} isPro={isPro} />
                 )}
             </div>
          </div>

          {/* Bottom Pane */}
          <div className="pt-bottom-pane">
             <div className="pt-tabs-header">
                <div className={`pt-tab ${activeTab === 'positions' ? 'active' : ''}`} onClick={() => setActiveTab('positions')}>
                  Positions ({state.positions.length})
                </div>
                <div className={`pt-tab ${activeTab === 'history' ? 'active' : ''}`} onClick={() => setActiveTab('history')}>
                  Order History
                </div>
             </div>
             <div className="pt-tab-content">
                {activeTab === 'positions' && (
                   <table className="pt-dense-table">
                     <thead>
                       <tr>
                         <th>Symbol</th>
                         <th>Size</th>
                         <th>Entry Price</th>
                         <th>Mark Price</th>
                         <th>Unrealized PNL</th>
                         <th>Action</th>
                       </tr>
                     </thead>
                     <tbody>
                       {state.positions.length > 0 ? state.positions.map(pos => {
                         const currentP = (pos.symbol === activeSymbol && liveQuote) ? liveQuote : (livePrices[pos.symbol] || pos.avgPrice);
                         const pnl = (currentP - pos.avgPrice) * pos.qty;
                         const pnlPct = ((currentP - pos.avgPrice) / pos.avgPrice) * 100;
                         return (
                           <tr key={pos.symbol}>
                             <td className="pt-sym-badge" style={{cursor: 'pointer'}} onClick={() => loadSymbolData(pos.symbol)}>{pos.symbol}</td>
                             <td>{pos.qty}</td>
                             <td>{pos.avgPrice.toLocaleString('en-IN', {maximumFractionDigits: 2})}</td>
                             <td>{currentP.toLocaleString('en-IN', {maximumFractionDigits: 2})}</td>
                             <td className={pnl >= 0 ? 'bull' : 'bear'} style={{color: pnl>=0 ? '#0ECB81' : '#F6465D'}}>
                                {pnl >= 0 ? '+' : ''}{pnl.toLocaleString('en-IN', {maximumFractionDigits: 2})} ({pnlPct.toFixed(2)}%)
                             </td>
                             <td>
                               <button 
                                 onClick={() => {
                                     loadSymbolData(pos.symbol);
                                     setOrderSide('SELL');
                                     setOrderQty(pos.qty);
                                 }}
                                 style={{background:'transparent', border:'1px solid #2B3139', color:'#848E9C', borderRadius:'4px', cursor:'pointer', padding:'2px 8px'}}
                               >Close</button>
                             </td>
                           </tr>
                         );
                       }) : (
                         <tr><td colSpan={6} style={{textAlign:'center', color:'#848E9C', padding:'24px'}}>No open positions.</td></tr>
                       )}
                     </tbody>
                   </table>
                )}
                {activeTab === 'history' && (
                   <table className="pt-dense-table">
                     <thead>
                       <tr>
                         <th>Time</th>
                         <th>Symbol</th>
                         <th>Type</th>
                         <th>Price</th>
                         <th>Amount</th>
                         <th>Total</th>
                       </tr>
                     </thead>
                     <tbody>
                       {state.history.length > 0 ? state.history.map(tx => (
                         <tr key={tx.id}>
                           <td style={{color: '#848E9C'}}>{new Date(tx.date).toLocaleTimeString()}</td>
                           <td className="pt-sym-badge">{tx.symbol}</td>
                           <td className={tx.type === 'BUY' ? 'pt-tx-buy' : 'pt-tx-sell'}>{tx.type}</td>
                           <td>{tx.price.toLocaleString('en-IN', {maximumFractionDigits: 2})}</td>
                           <td>{tx.qty}</td>
                           <td>{(tx.price * tx.qty).toLocaleString('en-IN', {maximumFractionDigits: 2})}</td>
                         </tr>
                       )) : (
                         <tr><td colSpan={6} style={{textAlign:'center', color:'#848E9C', padding:'24px'}}>No order history.</td></tr>
                       )}
                     </tbody>
                   </table>
                )}
             </div>
          </div>
        </div>

        {/* Right Column (Order Entry) */}
        <div className="pt-col-side">
           <div className="pt-order-header">
              <button 
                 className={`pt-btn-side ${orderSide === 'BUY' ? 'active-buy' : ''}`} 
                 onClick={() => setOrderSide('BUY')}
              >BUY</button>
              <button 
                 className={`pt-btn-side ${orderSide === 'SELL' ? 'active-sell' : ''}`} 
                 onClick={() => setOrderSide('SELL')}
              >SELL</button>
           </div>
           
           <div className="pt-order-body">
              <div style={{display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#848E9C', marginBottom: '16px'}}>
                 <span>Avail</span>
                 <span style={{color: '#EAECEF', fontWeight: 600}}>
                   {orderSide === 'BUY' ? `₹${state.cash.toLocaleString('en-IN', {maximumFractionDigits: 2})}` : `${activePos?.qty || 0} ${activeSymbol}`}
                 </span>
              </div>

              <div className="pt-input-group">
                 <span className="pt-input-label">Price</span>
                 <input className="pt-input-raw" value="Market" disabled />
                 <span className="pt-input-suffix">INR</span>
              </div>

              <div className="pt-input-group">
                 <span className="pt-input-label">Amount</span>
                 <input 
                    className="pt-input-raw" 
                    type="number" 
                    min="1"
                    value={orderQty}
                    onChange={e => setOrderQty(e.target.valueAsNumber || '')}
                    placeholder="0.00"
                 />
                 <span className="pt-input-suffix">Shares</span>
              </div>

              <div className="pt-slider-container">
                 <input 
                    type="range" 
                    className="pt-slider" 
                    min="0" 
                    max="100" 
                    step="1"
                    value={
                      orderQty 
                      ? orderSide === 'BUY' 
                          ? Math.min(100, (Number(orderQty) / maxBuyShares) * 100) || 0
                          : Math.min(100, (Number(orderQty) / (activePos?.qty || 1)) * 100) || 0
                      : 0
                    }
                    onChange={(e) => {
                       const pct = Number(e.target.value) / 100;
                       if (orderSide === 'BUY') {
                          setOrderQty(Math.floor(maxBuyShares * pct) || '');
                       } else {
                          setOrderQty(Math.floor((activePos?.qty || 0) * pct) || '');
                       }
                    }}
                 />
              </div>

              <div className="pt-order-summary">
                 <div className="pt-summary-row">
                    <span className="label">Total Est. Cost</span>
                    <span className="value">₹{(Number(orderQty || 0) * activePrice).toLocaleString('en-IN', {maximumFractionDigits: 2})}</span>
                 </div>
                 <button 
                    className={`pt-submit-btn ${orderSide === 'BUY' ? 'buy' : 'sell'}`}
                    disabled={
                       loadingPred || 
                       !orderQty || 
                       Number(orderQty) <= 0 || 
                       (orderSide === 'BUY' && (Number(orderQty) * activePrice > state.cash)) ||
                       (orderSide === 'SELL' && (!activePos || activePos.qty < Number(orderQty)))
                    }
                    onClick={executeTrade}
                 >
                    {orderSide} {activeSymbol}
                 </button>
              </div>

           </div>
        </div>

      </div>
      
      {toast && (
        <div className={`toast ${toast.type}`}>
          {toast.msg}
        </div>
      )}
    </div>
  );
};

export default PaperTrading;
