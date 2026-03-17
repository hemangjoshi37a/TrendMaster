import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import LineChart from './LineChart';

interface Company {
  symbol: string;
  name: string;
}

interface PredictionData {
  symbol: string;
  company_name: string;
  dates: string[];
  prices: number[];
  prediction_start_index: number;
}

interface RecentStock {
  symbol: string;
  price: number | null;
}

function App() {
  const [query, setQuery] = useState<string>('');
  const [suggestions, setSuggestions] = useState<Company[]>([]);
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [livePrice, setLivePrice] = useState<number | null>(null);
  const [prevPrice, setPrevPrice] = useState<number | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [recentStocks, setRecentStocks] = useState<RecentStock[]>([]);
  const [marketOpen, setMarketOpen] = useState<boolean>(true);
  const [showSuggestions, setShowSuggestions] = useState<boolean>(false);
  
  const ws = useRef<WebSocket | null>(null);
  const isSelecting = useRef<boolean>(false);

  // Initialize recent stocks from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('recentStocks');
    if (saved) setRecentStocks(JSON.parse(saved));
    
    const checkMarket = () => {
      const now = new Date();
      const istTime = new Date(now.toLocaleString('en-US', { timeZone: 'Asia/Kolkata' }));
      const day = istTime.getDay();
      const hour = istTime.getHours();
      const min = istTime.getMinutes();
      const isWeekend = day === 0 || day === 6;
      const isWorkingHours = (hour === 9 && min >= 15) || (hour > 9 && hour < 15) || (hour === 15 && min <= 30);
      setMarketOpen(!isWeekend && isWorkingHours);
    };
    checkMarket();
    const interval = setInterval(checkMarket, 60000);
    return () => clearInterval(interval);
  }, []);

  // Autocomplete search
  useEffect(() => {
    if (isSelecting.current) {
      isSelecting.current = false;
      return;
    }

    if (query.length > 1) {
      const fetchSuggestions = async () => {
        try {
          const res = await fetch(`http://localhost:8000/api/search?query=${query}`);
          if (res.ok) {
            const data = await res.json();
            setSuggestions(data);
            setShowSuggestions(data.length > 0);
          }
        } catch (e) {
          console.error("Search error", e);
        }
      };
      const delayDebounceFn = setTimeout(fetchSuggestions, 300);
      return () => clearTimeout(delayDebounceFn);
    } else {
      setSuggestions([]);
      setShowSuggestions(false);
    }
  }, [query]);

  const connectWebSocket = (stockSymbol: string) => {
    if (ws.current) ws.current.close();
    const socket = new WebSocket(`ws://localhost:8000/ws/ticks/${stockSymbol.toUpperCase()}`);
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.price) {
        setLivePrice(prev => {
          setPrevPrice(prev);
          return data.price;
        });
      }
    };
    ws.current = socket;
  };

  const handleManualSearch = () => {
    if (query.trim()) {
      handleSelectCompany({ symbol: query.trim().toUpperCase(), name: "" });
    }
  };

  const handleSelectCompany = async (company: Company) => {
    isSelecting.current = true;
    setQuery(company.symbol);
    setSuggestions([]);
    setShowSuggestions(false);
    setLoading(true);
    setError(null);
    setPrediction(null);
    setLivePrice(null);
    setPrevPrice(null);

    try {
      const response = await fetch(`http://localhost:8000/api/predict?stock_symbol=${company.symbol}`);
      if (!response.ok) {
        throw new Error("Failed to fetch prediction. Check if model is trained for " + company.symbol);
      }
      const data: PredictionData = await response.json();
      setPrediction(data);
      connectWebSocket(company.symbol);
      
      setRecentStocks(prev => {
        const filtered = prev.filter(s => s.symbol !== company.symbol);
        const updated = [{ symbol: company.symbol, price: null }, ...filtered].slice(0, 5);
        localStorage.setItem('recentStocks', JSON.stringify(updated));
        return updated;
      });
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const priceChange = livePrice && prevPrice ? livePrice - prevPrice : 0;
  const priceChangePct = prevPrice ? (priceChange / prevPrice) * 100 : 0;

  return (
    <div className="App dark-theme">
      <nav className="navbar">
        <div className="brand">
          <div className="logo">TrendMaster <span>PRO</span></div>
          <div className="market-status">
            <div className={`status-dot ${marketOpen ? '' : 'closed'}`}></div>
            {marketOpen ? 'NSE OPEN' : 'NSE CLOSED'}
          </div>
        </div>

        <div className="search-box">
          <div className="search-input-wrapper">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onFocus={() => suggestions.length > 0 && setShowSuggestions(true)}
              onKeyDown={(e) => e.key === 'Enter' && handleManualSearch()}
              placeholder="Search Symbol (e.g. RELIANCE, TCS...)"
              autoComplete="off"
            />
            <button className="search-btn" onClick={handleManualSearch}>
              <svg viewBox="0 0 24 24" width="20" height="20"><path fill="currentColor" d="M15.5 14h-.79l-.28-.27A6.471 6.471 0 0 0 16 9.5 6.5 6.5 0 1 0 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/></svg>
            </button>
          </div>
          {showSuggestions && (
            <ul className="suggestions">
              {suggestions.map((c) => (
                <li key={c.symbol} onClick={() => handleSelectCompany(c)}>
                  <span className="sym">{c.symbol}</span>
                  <span className="nam">{c.name}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
        
        <div className="nav-actions">
          <div className="user-icon" style={{ width: '32px', height: '32px', borderRadius: '50%', background: 'var(--accent)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '14px', fontWeight: 'bold', color: 'white' }}>
            A
          </div>
        </div>
      </nav>

      <main className="dashboard" onClick={() => setShowSuggestions(false)}>
        <div className="main-content">
          {error && <div className="notification error">Error: {error}</div>}
          
          <div className="card">
            {prediction ? (
              <>
                <div className="stock-header">
                  <div className="stock-title">
                    <h1>{prediction.company_name}</h1>
                    <div className="stock-meta">
                      <span className="ticker">{prediction.symbol}</span>
                      <span className="label">NSE INDIA</span>
                    </div>
                  </div>
                  <div className="stock-price">
                    <div className="price-row">
                      <span className="current-price">₹{livePrice?.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || "---"}</span>
                      {priceChange !== 0 && (
                        <span className={`price-change ${priceChange >= 0 ? 'up' : 'down'}`}>
                          {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)} ({priceChangePct.toFixed(2)}%)
                        </span>
                      )}
                    </div>
                    <span className="label">Real-time Data</span>
                  </div>
                </div>

                <div className="chart-controls">
                  <div className="timeframes">
                    <button className="tf-btn active">1D</button>
                    <button className="tf-btn">5D</button>
                    <button className="tf-btn">1M</button>
                    <button className="tf-btn">1Y</button>
                    <button className="tf-btn">MAX</button>
                  </div>
                  <div className="chart-legend-simple">
                    <span style={{ color: '#2962ff', marginRight: '15px' }}>● Historical</span>
                    <span style={{ color: '#f23645' }}>--- AI Prediction</span>
                  </div>
                </div>

                <div className="chart-wrapper">
                  <LineChart data={prediction} />
                </div>
              </>
            ) : (
              <div className="empty-state">
                {loading ? (
                  <div className="loader">
                    <svg width="64" height="64" viewBox="0 0 24 24"><path fill="var(--accent)" d="M12,4V2A10,10 0 0,0 2,12H4A8,8 0 0,1 12,4Z"/></svg>
                    <p>Deep Learning Model Analyzing {query}...</p>
                  </div>
                ) : (
                  <>
                    <svg width="120" height="120" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                    <h2>Welcome to TrendMaster PRO</h2>
                    <p>Search for any NSE stock to see Transformer-based future predictions</p>
                  </>
                )}
              </div>
            )}
          </div>

          {prediction && (
            <div className="card analysis-section">
              <h2>Transformer Prediction Analysis</h2>
              <table className="prediction-table">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Predicted Price</th>
                    <th>Expected Change</th>
                    <th>Trend</th>
                  </tr>
                </thead>
                <tbody>
                  {prediction.dates.slice(prediction.prediction_start_index, prediction.prediction_start_index + 10).map((date, i) => {
                    const idx = prediction.prediction_start_index + i;
                    const price = prediction.prices[idx];
                    const prevPriceVal = prediction.prices[idx - 1];
                    const change = price - prevPriceVal;
                    const changePct = (change / prevPriceVal) * 100;
                    
                    return (
                      <tr key={date}>
                        <td>{new Date(date).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })}</td>
                        <td style={{ fontWeight: 700 }}>₹{price.toFixed(2)}</td>
                        <td className={`change-cell ${change >= 0 ? 'up' : 'down'}`}>
                          {change >= 0 ? '▲' : '▼'} {Math.abs(changePct).toFixed(2)}%
                        </td>
                        <td>
                          <span style={{ 
                            padding: '4px 10px', 
                            borderRadius: '4px', 
                            fontSize: '0.8rem',
                            background: change >= 0 ? 'rgba(8, 153, 129, 0.1)' : 'rgba(242, 54, 69, 0.1)',
                            color: change >= 0 ? 'var(--success)' : 'var(--error)'
                          }}>
                            {change >= 0 ? 'BULLISH' : 'BEARISH'}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div className="sidebar">
          <div className="card sidebar-section">
            <h3>AI Insight</h3>
            <div style={{ padding: '10px 0' }}>
              <p style={{ fontSize: '0.9rem', lineHeight: '1.6', color: 'var(--text-muted)' }}>
                Our Transformer model uses multi-head attention to analyze historical price patterns and technical indicators.
              </p>
              <div className="stat-item">
                <span>Model Confidence</span>
                <span style={{ color: 'var(--success)', fontWeight: 700 }}>HIGH</span>
              </div>
              <div className="stat-item">
                <span>Prediction Steps</span>
                <span>10 Days</span>
              </div>
            </div>
          </div>

          <div className="card sidebar-section">
            <h3>Recent Searches</h3>
            <div className="recent-list">
              {recentStocks.length > 0 ? recentStocks.map(s => (
                <div key={s.symbol} className="recent-item" onClick={() => handleSelectCompany({ symbol: s.symbol, name: "" })}>
                  <span className="sym">{s.symbol}</span>
                  <span className="price">VIEW</span>
                </div>
              )) : (
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', textAlign: 'center' }}>No recent stocks</p>
              )}
            </div>
          </div>

          <div className="card sidebar-section" style={{ flex: 1 }}>
            <h3>Market Overview</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '15px', marginTop: '10px' }}>
              <div className="stat-item">
                <span>NIFTY 50</span>
                <span style={{ color: 'var(--success)' }}>+0.42%</span>
              </div>
              <div className="stat-item">
                <span>SENSEX</span>
                <span style={{ color: 'var(--success)' }}>+0.38%</span>
              </div>
              <div className="stat-item">
                <span>Bank Nifty</span>
                <span style={{ color: 'var(--error)' }}>-0.15%</span>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
