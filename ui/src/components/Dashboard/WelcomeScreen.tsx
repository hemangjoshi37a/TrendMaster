import React from 'react';
import { MarketIndex, RecentStock } from '../../types/market';

interface WelcomeScreenProps {
  isPro: boolean;
  marketOpen: boolean;
  marketIndices: MarketIndex[];
  recentStocks: RecentStock[];
  selectedTimeframe: string;
  onSymbolSelect: (symbol: string) => void;
}

export const WelcomeScreen: React.FC<WelcomeScreenProps> = ({
  isPro,
  marketOpen,
  marketIndices,
  recentStocks,
  onSymbolSelect
}) => {
  return (
    <div className={`welcome-screen${isPro ? ' pro' : ''}`}>
      <div className="welcome-orb welcome-orb-1" />
      <div className="welcome-orb welcome-orb-2" />
      <div className="welcome-orb welcome-orb-3" />

      <div className="welcome-inner">
        {isPro ? (
          /* ── PRO TERMINAL WELCOME ── */
          <>
            <div className="welcome-header">
              <div className="welcome-logo-mark pro-mark">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
                </svg>
              </div>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <h2 className="welcome-title pro-title">
                    {(() => {
                      const h = new Date(new Date().toLocaleString('en-US', { timeZone: 'Asia/Kolkata' })).getHours();
                      return h < 12 ? 'Good Morning' : h < 17 ? 'Good Afternoon' : 'Good Evening';
                    })()}, Pro Member
                  </h2>
                  <span className="pro-elite-badge">⚡ PRO</span>
                </div>
                <p className="welcome-sub">All features unlocked · AI Confidence Scores · Full 10-Day Forecast</p>
              </div>
              <div className={`welcome-market-badge ${marketOpen ? 'open' : 'closed'}`}>
                <span className="status-dot" />
                {marketOpen ? 'NSE LIVE' : 'NSE CLOSED'}
              </div>
            </div>

            <div className="pro-caps-bar">
              {[
                { icon: '🔮', label: 'Full AI Forecast' },
                { icon: '⚡', label: 'Real-time WebSockets' },
                { icon: '📊', label: 'AI Confidence Scores' },
                { icon: '📈', label: 'Full Historical Data' },
              ].map((cap, i) => (
                <div key={i} className="pro-cap-pill" style={{ animationDelay: `${i * 0.06}s` }}>
                  <span>{cap.icon}</span>
                  <span>{cap.label}</span>
                </div>
              ))}
            </div>

            <div className="index-cards-grid">
              {marketIndices.length > 0 ? marketIndices.map((idx, i) => (
                <div
                  key={idx.name}
                  className={`index-card pro-card ${idx.change_pct >= 0 ? 'bull' : 'bear'}`}
                  style={{ animationDelay: `${i * 0.08}s` }}
                  onClick={() => onSymbolSelect(idx.name.replace(' ', ''))}
                >
                  <div className="ic-glow" />
                  <div className="ic-top">
                    <span className="ic-name">{idx.name}</span>
                    <span className={`ic-badge ${idx.change_pct >= 0 ? 'bull' : 'bear'}`}>
                      {idx.change_pct >= 0 ? '▲' : '▼'} {Math.abs(idx.change_pct).toFixed(2)}%
                    </span>
                  </div>
                  <div className="ic-price">{idx.price.toLocaleString('en-IN')}</div>
                  <div className="ic-bar">
                    <div className="ic-bar-fill" style={{ width: `${Math.min(Math.abs(idx.change_pct) * 10, 100)}%` }} />
                  </div>
                </div>
              )) : [1, 2, 3].map(i => (
                <div key={i} className="index-card skeleton" style={{ animationDelay: `${i * 0.1}s` }}>
                  <div className="skeleton-line" style={{ width: '60%', height: '12px', marginBottom: '12px' }} />
                  <div className="skeleton-line" style={{ width: '80%', height: '24px' }} />
                </div>
              ))}
            </div>

            <div className="welcome-divider">
              <div className="divider-line" />
              <span className="divider-text pro-divider-text">🤖 AI Signal Feed</span>
              <div className="divider-line" />
            </div>

            <div className="ai-signals-row">
              {[
                { symbol: 'NIFTY50', label: 'Nifty 50', signal: 'BULLISH', confidence: 87, reason: 'Strong momentum breakout above 200 EMA with volume confirmation' },
                { symbol: 'RELIANCE', label: 'Reliance', signal: 'BULLISH', confidence: 74, reason: 'Consolidation breakout with institutional volume surge detected' },
                { symbol: 'INFY', label: 'Infosys', signal: 'BEARISH', confidence: 68, reason: 'MACD bearish crossover with RSI overbought divergence' },
              ].map((sig, i) => (
                <div
                  key={sig.symbol}
                  className={`ai-signal-card ${sig.signal === 'BULLISH' ? 'bull' : 'bear'}`}
                  style={{ animationDelay: `${0.35 + i * 0.1}s` }}
                  onClick={() => onSymbolSelect(sig.symbol)}
                >
                  <div className="asc-glow" />
                  <div className="asc-top">
                    <div>
                      <div className="asc-symbol">{sig.symbol}</div>
                      <div className="asc-label">{sig.label}</div>
                    </div>
                    <span className={`asc-badge ${sig.signal === 'BULLISH' ? 'bull' : 'bear'}`}>
                      {sig.signal === 'BULLISH' ? '▲' : '▼'} {sig.signal}
                    </span>
                  </div>
                  <div className="asc-reason">{sig.reason}</div>
                  <div className="asc-conf-row">
                    <span className="asc-conf-label">AI Confidence</span>
                    <span className="asc-conf-val">{sig.confidence}%</span>
                  </div>
                  <div className="asc-conf-bar">
                    <div className={`asc-conf-fill ${sig.signal === 'BULLISH' ? 'bull' : 'bear'}`} style={{ width: `${sig.confidence}%` }} />
                  </div>
                </div>
              ))}
            </div>

            <div className="welcome-divider" style={{ marginTop: '20px' }}>
              <div className="divider-line" />
              <span className="divider-text">⚡ Quick Launch</span>
              <div className="divider-line" />
            </div>
            <div className="quick-stocks-grid">
              {[
                { symbol: 'RELIANCE', label: 'Reliance' },
                { symbol: 'TCS', label: 'TCS' },
                { symbol: 'HDFCBANK', label: 'HDFC Bank' },
                { symbol: 'INFY', label: 'Infosys' },
                { symbol: 'ICICIBANK', label: 'ICICI Bank' },
                { symbol: 'WIPRO', label: 'Wipro' },
                { symbol: 'SBIN', label: 'SBI' },
                { symbol: 'TATAMOTORS', label: 'Tata Motors' },
                { symbol: 'ADANIENT', label: 'Adani Ent.' },
                { symbol: 'BAJFINANCE', label: 'Bajaj Finance' },
              ].map((stock, i) => (
                <button
                  key={stock.symbol}
                  className="quick-chip pro-chip"
                  style={{ animationDelay: `${0.5 + i * 0.04}s` }}
                  onClick={() => onSymbolSelect(stock.symbol)}
                >
                  <span className="qc-symbol pro-sym">{stock.symbol}</span>
                  <span className="qc-label">{stock.label}</span>
                </button>
              ))}
            </div>
          </>
        ) : (
          /* ── FREE TRIAL WELCOME ── */
          <>
            <div className="welcome-header">
              <div className="welcome-logo-mark">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
                </svg>
              </div>
              <div>
                <h2 className="welcome-title">Market Pulse</h2>
                <p className="welcome-sub">Live indices · Search any NSE symbol to begin AI forecasting</p>
              </div>
              <div className={`welcome-market-badge ${marketOpen ? 'open' : 'closed'}`}>
                <span className="status-dot" />
                {marketOpen ? 'NSE LIVE' : 'NSE CLOSED'}
              </div>
            </div>

            <div className="index-cards-grid">
              {marketIndices.length > 0 ? marketIndices.map((idx, i) => (
                <div
                  key={idx.name}
                  className={`index-card ${idx.change_pct >= 0 ? 'bull' : 'bear'}`}
                  style={{ animationDelay: `${i * 0.08}s` }}
                  onClick={() => onSymbolSelect(idx.name.replace(' ', ''))}
                >
                  <div className="ic-glow" />
                  <div className="ic-top">
                    <span className="ic-name">{idx.name}</span>
                    <span className={`ic-badge ${idx.change_pct >= 0 ? 'bull' : 'bear'}`}>
                      {idx.change_pct >= 0 ? '▲' : '▼'} {Math.abs(idx.change_pct).toFixed(2)}%
                    </span>
                  </div>
                  <div className="ic-price">{idx.price.toLocaleString('en-IN')}</div>
                  <div className="ic-bar">
                    <div className="ic-bar-fill" style={{ width: `${Math.min(Math.abs(idx.change_pct) * 10, 100)}%` }} />
                  </div>
                </div>
              )) : [1, 2, 3].map(i => (
                <div key={i} className="index-card skeleton" style={{ animationDelay: `${i * 0.1}s` }}>
                  <div className="skeleton-line" style={{ width: '60%', height: '12px', marginBottom: '12px' }} />
                  <div className="skeleton-line" style={{ width: '80%', height: '24px' }} />
                </div>
              ))}
            </div>

            <div className="welcome-divider">
              <div className="divider-line" />
              <span className="divider-text">⚡ Quick Launch</span>
              <div className="divider-line" />
            </div>

            <div className="quick-stocks-grid">
              {[
                { symbol: 'RELIANCE', label: 'Reliance' },
                { symbol: 'TCS', label: 'TCS' },
                { symbol: 'HDFCBANK', label: 'HDFC Bank' },
                { symbol: 'INFY', label: 'Infosys' },
                { symbol: 'ICICIBANK', label: 'ICICI Bank' },
                { symbol: 'WIPRO', label: 'Wipro' },
                { symbol: 'SBIN', label: 'SBI' },
                { symbol: 'TATAMOTORS', label: 'Tata Motors' },
                { symbol: 'ADANIENT', label: 'Adani Ent.' },
                { symbol: 'BAJFINANCE', label: 'Bajaj Finance' },
              ].map((stock, i) => (
                <button
                  key={stock.symbol}
                  className="quick-chip"
                  style={{ animationDelay: `${0.2 + i * 0.04}s` }}
                  onClick={() => onSymbolSelect(stock.symbol)}
                >
                  <span className="qc-symbol">{stock.symbol}</span>
                  <span className="qc-label">{stock.label}</span>
                </button>
              ))}
            </div>
          </>
        )}

        {recentStocks.length > 0 && (
          <div className="welcome-recent" style={{ marginTop: '16px' }}>
            <span className="welcome-recent-label">Recent</span>
            {recentStocks.slice(0, 5).map(s => (
              <div key={s.symbol} className="recent-chip" onClick={() => onSymbolSelect(s.symbol)}>
                {s.symbol}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
