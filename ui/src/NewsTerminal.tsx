import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useLocation, Link, useNavigate } from 'react-router-dom';
import Footer from './Footer';
import TopNav from './TopNav';
import './NewsTerminal.css';

interface NewsItem {
  id: string;
  title: string;
  source: string;
  url: string;
  impact: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  score: number;
  category: string;
  timestamp: string;
}

const MarketTicker: React.FC = () => {
  const [tickers] = useState([
    { symbol: 'NIFTY 50', value: '23,456.20', change: '+142.10', trend: 'bull' },
    { symbol: 'RELIANCE', value: '2,921.45', change: '-12.30', trend: 'bear' },
    { symbol: 'TCS', value: '3,842.10', change: '+45.60', trend: 'bull' },
    { symbol: 'HDFC BANK', value: '1,562.90', change: '+22.45', trend: 'bull' },
    { symbol: 'INFY', value: '1,634.30', change: '-5.20', trend: 'bear' },
    { symbol: 'SBI', value: '824.15', change: '+18.70', trend: 'bull' },
  ]);

  return (
    <div className="market-ticker-container">
      <div className="market-ticker">
        {[...tickers, ...tickers].map((t, i) => (
          <div className="ticker-item" key={i}>
            <span className="ticker-label">{t.symbol}</span>
            <span className="ticker-value">{t.value}</span>
            <span className={`ticker-trend ${t.trend}`}>{t.change}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

const NewsTerminal: React.FC = () => {
  const [news, setNews] = useState<NewsItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const location = useLocation();
  const navigate = useNavigate();
  const isPro = location.state?.isPro || false;

  const fetchNews = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch('/api/news');
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || "Failed to fetch market news");
      }
      const data = await resp.json();
      setNews(data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchNews();
    const interval = setInterval(fetchNews, 300000);
    return () => clearInterval(interval);
  }, [fetchNews]);

  const bullishCount = news.filter(n => n.impact === 'BULLISH').length;
  const bearishCount = news.filter(n => n.impact === 'BEARISH').length;
  const overallSentiment = bullishCount > bearishCount ? 'BULLISH' : bullishCount < bearishCount ? 'BEARISH' : 'NEUTRAL';

  const extractTickers = (title: string) => {
    const commonWords = ['NYSE', 'NASDAQ', 'BSE', 'NSE', 'REIT', 'FED', 'GDP', 'CPI'];
    const matches = title.match(/\b[A-Z]{3,10}\b/g) || [];
    return Array.from(new Set(matches)).filter(t => !commonWords.includes(t));
  };

  const handleTrade = (e: React.MouseEvent, symbol: string) => {
    e.preventDefault();
    e.stopPropagation();
    navigate('/paper-trading', { state: { isPro, searchSymbol: symbol } });
  };

  const newsItems = useMemo(() => {
    if (news.length === 0) return { featured: null, rest: [] };
    const sorted = [...news].sort((a, b) => b.score - a.score);
    return { featured: sorted[0], rest: sorted.slice(1) };
  }, [news]);

  return (
    <div className="news-terminal-wrapper dark-theme">
      <TopNav activePage="news" isPro={isPro} />

      <MarketTicker />

      <div className="news-terminal">
        <div className="news-header">
          <div className="news-title">
            <h1>Market Terminal</h1>
            <p>Real-time AI-powered financial intelligence and macro analysis.</p>
          </div>

          {!loading && !error && news.length > 0 && (
            <div className="sentiment-overview">
              <div className="sentiment-gauge-v2">
                <div className="gauge-bg"></div>
                <div className={`gauge-fill ${overallSentiment.toLowerCase()}`}></div>
                <div className="gauge-text">{overallSentiment}</div>
              </div>

              <div className="sentiment-stats">
                <div style={{ fontSize: '0.9rem', color: '#fff', fontWeight: 700 }}>
                  Global Pulse
                </div>
                <div style={{ display: 'flex', height: '6px', width: '160px', background: '#2a2e39', borderRadius: '3px', marginTop: '8px', overflow: 'hidden' }}>
                  <div style={{ width: `${(bullishCount / news.length) * 100}%`, background: 'var(--sentiment-emerald)' }}></div>
                  <div style={{ width: `${(bearishCount / news.length) * 100}%`, background: 'var(--sentiment-ruby)' }}></div>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="news-content">
          {loading ? (
            <div className="news-loading">
              <div className="bt-spinner"></div>
              <h3>Synchronizing feeds...</h3>
            </div>
          ) : error ? (
            <div className="news-error">
              <h3>System Offline</h3>
              <p>{error}</p>
              <button className="news-trade-btn" style={{ marginTop: '20px' }} onClick={fetchNews}>Retry Connection</button>
            </div>
          ) : (
            <div className="news-grid">
              {newsItems.featured && (
                <a
                  className="news-card news-card-featured"
                  href={newsItems.featured.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ animationDelay: '0s' }}
                >
                  <div className="featured-tag">BREAKING</div>
                  <div className="news-card-meta">
                    <span className="news-source">{newsItems.featured.source}</span>
                    <span style={{ display: 'flex', alignItems: 'center' }}>
                      <span className={`sentiment-dot ${newsItems.featured.impact.toLowerCase()}`}></span>
                      <span style={{ color: newsItems.featured.impact === 'BULLISH' ? 'var(--sentiment-emerald)' : 'var(--sentiment-ruby)', fontWeight: 800 }}>{newsItems.featured.impact}</span>
                    </span>
                  </div>
                  <h3>{newsItems.featured.title}</h3>

                  <div className="news-card-footer">
                    <span className="news-time">Institutional Grade Feed</span>
                    <div style={{ display: 'flex', gap: '8px' }}>
                      {extractTickers(newsItems.featured.title).map(sym => (
                        <button key={sym} className="news-trade-btn" onClick={(e) => handleTrade(e, sym)}>Trade {sym}</button>
                      ))}
                    </div>
                  </div>
                </a>
              )}

              {newsItems.rest.map((item, i) => {
                const tickers = extractTickers(item.title);
                return (
                  <a
                    key={item.id || i}
                    className="news-card"
                    href={item.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{ animationDelay: `${(i + 1) * 0.05}s` }}
                  >
                    <div className="news-card-meta">
                      <span className="news-source">{item.source}</span>
                      <span style={{ display: 'flex', alignItems: 'center' }}>
                        <span className={`sentiment-dot ${item.impact.toLowerCase()}`}></span>
                        <span style={{ fontSize: '0.7rem', fontWeight: 800 }}>{item.impact}</span>
                      </span>
                    </div>
                    <h3>{item.title}</h3>

                    {tickers.length > 0 && (
                      <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
                        {tickers.map(sym => (
                          <span key={sym} className="news-time" style={{ color: '#2962ff', fontWeight: 700, cursor: 'pointer' }} onClick={(e) => handleTrade(e, sym)}>
                            {sym} ↗
                          </span>
                        ))}
                      </div>
                    )}

                    <div className="news-card-footer">
                      <span className="news-time">{item.category}</span>
                      <span className="news-time">{new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                    </div>
                  </a>
                );
              })}
            </div>
          )}
        </div>
      </div>
      <Footer isPro={isPro} wsStatus="connected" />
    </div>
  );
};

export default NewsTerminal;
