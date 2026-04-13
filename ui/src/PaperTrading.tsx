import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useLocation, Link, useNavigate } from 'react-router-dom';
import LineChart from './LineChart';
import EquityCurve from './EquityCurve';
import TopNav from './TopNav';
import './PaperTrading.css';

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
  equityHistory: { time: string; value: number }[];
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

  const [state, setState] = useState<PaperTradingState>({ cash: DEFAULT_CASH, positions: [], history: [], equityHistory: [] });

  const fetchPortfolioState = async () => {
    try {
      const sessionStr = localStorage.getItem('tm_session');
      if (!sessionStr) return;
      const session = JSON.parse(sessionStr);

      const res = await fetch('/api/user', {
        headers: { 'X-User-Id': session.userId.toString() }
      });
      if (res.ok) {
        const user = await res.json();
        setState(prev => ({
          ...prev, // preserve equity history
          cash: user.cash_balance,
          positions: user.positions.map((p: any) => ({
            symbol: p.symbol,
            qty: p.quantity,
            avgPrice: p.average_price,
            takeProfit: p.take_profit,
            stopLoss: p.stop_loss
          })),
          history: user.transactions.map((t: any) => ({
            id: t.id,
            symbol: t.symbol,
            type: t.type,
            qty: t.quantity,
            price: t.price,
            date: t.timestamp
          }))
        }));
      }
    } catch(e) { console.error("Failed to load DB state", e); }
  };

  useEffect(() => {
    fetchPortfolioState();
  }, []);

  const [livePrices, setLivePrices] = useState<{ [symbol: string]: number }>({});
  
  // Terminal State
  const [searchInput, setSearchInput] = useState('RELIANCE');
  const [activeSymbol, setActiveSymbol] = useState('RELIANCE');
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [loadingPred, setLoadingPred] = useState(false);
  
  // Order State
  const [orderSide, setOrderSide] = useState<'BUY' | 'SELL'>('BUY');
  const [orderQty, setOrderQty] = useState<number | ''>('');
  const [takeProfit, setTakeProfit] = useState<number | ''>('');
  const [stopLoss, setStopLoss] = useState<number | ''>('');
  
  // UI State
  const [activeTab, setActiveTab] = useState<'positions' | 'history'>('positions');
  const [toast, setToast] = useState<{msg: string, type: 'success'|'error'} | null>(null);

  const ws = useRef<WebSocket | null>(null);
  const [liveQuote, setLiveQuote] = useState<number | null>(null);
  const [prevQuote, setPrevQuote] = useState<number | null>(null);

  // Removed localStorage saves since we are using Database

  const showToast = (msg: string, type: 'success' | 'error') => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3000);
  };

  const connectWebSocket = useCallback((symbol: string) => {
    if (ws.current) ws.current.close();

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    let host = window.location.host;
    if (host === 'localhost:3000') {
      host = 'localhost:8000';
    }
    const socket = new WebSocket(`${protocol}//${host}/ws/ticks/${symbol.toUpperCase()}`);

    socket.onopen = () => console.log('WS Connected:', symbol);
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.price) {
        setLiveQuote(prev => {
          setPrevQuote(prev);
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
    setTakeProfit('');
    setStopLoss('');

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
  const initialSearch = location.state?.searchSymbol || 'RELIANCE';

  useEffect(() => {
    loadSymbolData(initialSearch);
    return () => { if (ws.current) ws.current.close(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialSearch]);

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

  // Equity History Tracking
  useEffect(() => {
    const updateEquity = () => {
       setState(prev => {
         let currentTotal = prev.cash;
         prev.positions.forEach(pos => {
           const p = (pos.symbol === activeSymbol && liveQuote) ? liveQuote : (livePrices[pos.symbol] || pos.avgPrice);
           currentTotal += p * pos.qty;
         });

         const nowStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
         // Keep last 50 points
         const newHistory = [...prev.equityHistory, { time: nowStr, value: currentTotal }].slice(-50);
         return { ...prev, equityHistory: newHistory };
       });
    };

    const interval = setInterval(updateEquity, 60000); // 1 min refresh
    return () => clearInterval(interval);
  }, [liveQuote, livePrices, activeSymbol]);

  // Auto-Executor Loop: Evaluate bounds whenever live prices change
  useEffect(() => {
    state.positions.forEach(pos => {
      const currentPrice = (pos.symbol === activeSymbol && liveQuote) ? liveQuote : livePrices[pos.symbol];
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
        const triggerAutoTrade = async () => {
          try {
            const sessionStr = localStorage.getItem('tm_session');
            const session = sessionStr ? JSON.parse(sessionStr) : null;
            if (!session) return;
            
            const resp = await fetch('/api/portfolio/trade', {
              method: 'POST',
              headers: { 
                 'Content-Type': 'application/json',
                 'X-User-Id': session.userId.toString()
              },
              body: JSON.stringify({ symbol: pos.symbol, price: currentPrice, quantity: pos.qty, type: 'SELL' })
            });
            if (resp.ok) {
              showToast(`${reason}: Auto-sold ${pos.qty} ${pos.symbol} at ₹${currentPrice.toFixed(2)}`, 'success');
              fetchPortfolioState();
            }
          } catch(e) { console.error(e); }
        };
        triggerAutoTrade();
      }
    });
  }, [liveQuote, livePrices, activeSymbol, state.positions]);

  const executeTrade = async () => {
    const currentPrice = liveQuote || (prediction ? prediction.prices[prediction.prediction_start_index - 1] : 0);
    if (!currentPrice || !orderQty || orderQty <= 0) return;
    
    const qty = Number(orderQty);
    let tpVal = takeProfit ? Number(takeProfit) : undefined;
    let slVal = stopLoss ? Number(stopLoss) : undefined;

    if (orderSide === 'BUY') {
      if (tpVal !== undefined && tpVal <= currentPrice) {
        showToast("Take Profit must be > entry price!", 'error');
        return;
      }
      if (slVal !== undefined && slVal >= currentPrice) {
        showToast("Stop Loss must be < entry price!", 'error');
        return;
      }
      if (qty * currentPrice > state.cash) {
        showToast("Insufficient Buying Power!", 'error');
        return;
      }
    } else {
      const pos = state.positions.find(p => p.symbol === activeSymbol);
      if (!pos || pos.qty < qty) {
        showToast("Insufficient Shares to Sell!", 'error');
        return;
      }
    }

    try {
      const sessionStr = localStorage.getItem('tm_session');
      if (!sessionStr) return;
      const session = JSON.parse(sessionStr);

      const resp = await fetch('/api/portfolio/trade', {
        method: 'POST',
        headers: { 
           'Content-Type': 'application/json',
           'X-User-Id': session.userId.toString()
        },
        body: JSON.stringify({ 
          symbol: activeSymbol, 
          price: currentPrice, 
          quantity: qty, 
          type: orderSide,
          take_profit: tpVal,
          stop_loss: slVal
        })
      });
      
      if (resp.ok) {
        showToast(`Market ${orderSide} ${qty} ${activeSymbol} Executed`, 'success');
        setOrderQty('');
        setTakeProfit('');
        setStopLoss('');
        fetchPortfolioState();
      } else {
        showToast(`Trade Failed`, 'error');
      }
    } catch(e) {
      showToast(`Network Error`, 'error');
    }
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

  // Trade Lines
  const tradeLines = [];
  if (activePos) {
    tradeLines.push({ price: activePos.avgPrice, type: 'entry' as const, label: 'ENTRY' });
    if (activePos.takeProfit) tradeLines.push({ price: activePos.takeProfit, type: 'tp' as const, label: 'TP' });
    if (activePos.stopLoss) tradeLines.push({ price: activePos.stopLoss, type: 'sl' as const, label: 'SL' });
  }

  return (
    <div className="pt-pro-wrapper">
      {/* Top Navbar */}
      <TopNav 
        activePage="paper-trading" 
        isPro={isPro} 
      />

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
                     <LineChart data={prediction} isPro={isPro} tradeLines={tradeLines} />
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
                         <th>Limits (TP/SL)</th>
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
                             <td>₹{pos.avgPrice.toLocaleString('en-IN', {maximumFractionDigits: 2})}</td>
                             <td>₹{currentP.toLocaleString('en-IN', {maximumFractionDigits: 2})}</td>
                             <td className={pnl >= 0 ? 'bull' : 'bear'} style={{color: pnl>=0 ? '#0ECB81' : '#F6465D'}}>
                                {pnl >= 0 ? '+' : ''}{pnl.toLocaleString('en-IN', {maximumFractionDigits: 2})} ({pnlPct.toFixed(2)}%)
                             </td>
                             <td style={{fontSize: '0.75rem', color: '#848E9C'}}>
                                {pos.takeProfit ? `TP: ₹${pos.takeProfit.toLocaleString('en-IN')}` : ''}
                                {pos.takeProfit && pos.stopLoss ? <br/> : ''}
                                {pos.stopLoss ? `SL: ₹${pos.stopLoss.toLocaleString('en-IN')}` : ''}
                                {!pos.takeProfit && !pos.stopLoss ? '---' : ''}
                             </td>
                             <td>
                               <div style={{display: 'flex', gap: '4px', justifyContent: 'flex-end'}}>
                                 <button 
                                   onClick={() => {
                                       loadSymbolData(pos.symbol);
                                       setOrderSide('SELL');
                                       setOrderQty(pos.qty);
                                   }}
                                   style={{background:'transparent', border:'1px solid #2B3139', color:'#848E9C', borderRadius:'4px', cursor:'pointer', padding:'2px 8px'}}
                                 >Close</button>
                                 <button
                                   onClick={() => loadSymbolData(pos.symbol)}
                                   style={{background:'transparent', border:'1px solid #2B3139', color:'#FCD535', borderRadius:'4px', cursor:'pointer', padding:'2px 8px'}}
                                 >Trade</button>
                               </div>
                             </td>
                           </tr>
                         );
                       }) : (
                         <tr><td colSpan={7} style={{textAlign:'center', color:'#848E9C', padding:'24px'}}>No open positions.</td></tr>
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

              {/* Advanced Order Types (TP/SL) */}
              {orderSide === 'BUY' && (
                 <div className="pt-advanced-orders" style={{marginTop: '8px', borderTop: '1px solid #2B3139', paddingTop: '16px'}}>
                   <div style={{fontSize: '0.7rem', color: '#FCD535', fontWeight: 700, marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '6px', letterSpacing: '0.5px'}}>
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                         <circle cx="12" cy="12" r="10"></circle>
                         <line x1="12" y1="8" x2="12" y2="12"></line>
                         <line x1="12" y1="16" x2="12.01" y2="16"></line>
                      </svg>
                      ADVANCED CONTROLS
                   </div>
                   
                   <div className="pt-input-group" style={{marginBottom: '10px'}}>
                      <span className="pt-input-label" style={{width: '90px'}}>Take Profit</span>
                      <input 
                         className="pt-input-raw" 
                         type="number" 
                         value={takeProfit}
                         onChange={e => setTakeProfit(e.target.valueAsNumber || '')}
                         placeholder="Target ₹"
                      />
                   </div>

                   <div className="pt-input-group" style={{marginBottom: '16px'}}>
                      <span className="pt-input-label" style={{width: '90px'}}>Stop Loss</span>
                      <input 
                         className="pt-input-raw" 
                         type="number" 
                         value={stopLoss}
                         onChange={e => setStopLoss(e.target.valueAsNumber || '')}
                         placeholder="Exit ₹"
                      />
                   </div>
                 </div>
              )}

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

              <div className="pt-equity-widget">
                <div className="pt-equity-label">
                  <span>Account Performance</span>
                  <span style={{color: '#FCD535'}}>7D Snapshot</span>
                </div>
                <div className="pt-equity-chart">
                  <EquityCurve data={state.equityHistory} />
                </div>
              </div>

           </div>
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
    </div>
  );
};

export default PaperTrading;
