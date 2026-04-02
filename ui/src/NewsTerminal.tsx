import React, { useState, useEffect } from 'react';
import { useLocation, Link } from 'react-router-dom';
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

const NewsTerminal: React.FC = () => {
  const [news, setNews] = useState<NewsItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const location = useLocation();
  const isPro = location.state?.isPro || false;

  const fetchNews = async () => {
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
  };

  useEffect(() => {
    fetchNews();
    const interval = setInterval(fetchNews, 300000); // Refresh every 5 mins
    return () => clearInterval(interval);
  }, []);

  const bullishCount = news.filter(n => n.impact === 'BULLISH').length;
  const bearishCount = news.filter(n => n.impact === 'BEARISH').length;
  const overallSentiment = bullishCount > bearishCount ? 'BULLISH' : bullishCount < bearishCount ? 'BEARISH' : 'NEUTRAL';

  return (
    <div className="news-terminal-wrapper dark-theme">
      <TopNav activePage="news" isPro={isPro} />
      <div className="news-terminal">
        <div className="news-header">
          <div className="news-title">
            <h1>Global News terminal</h1>
            <p>Real-time AI-powered market sentiment and macro analysis.</p>
          </div>

          {!loading && !error && news.length > 0 && (
            <div className="sentiment-overview">
              <div className="sentiment-gauge">
                <div className="sentiment-label">Global Pulse</div>
                <div className={`sentiment-value ${overallSentiment.toLowerCase()}`}>
                  {overallSentiment}
                </div>
              </div>
              <div style={{ width: '1px', height: '40px', background: 'var(--border)' }}></div>
              <div className="sentiment-stats">
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  {bullishCount} Bullish / {bearishCount} Bearish signals
                </div>
                <div style={{ display: 'flex', height: '4px', background: 'var(--border)', borderRadius: '2px', marginTop: '6px', overflow: 'hidden' }}>
                  <div style={{ width: `${(bullishCount / news.length) * 100}%`, background: 'var(--success)' }}></div>
                  <div style={{ width: `${(bearishCount / news.length) * 100}%`, background: 'var(--error)' }}></div>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="news-content">
          {loading ? (
            <div className="news-loading">
              <div className="bt-spinner"></div>
              <h3>Scanning Global Feeds...</h3>
              <p>Our AI is analyzing the latest market headlines for macro impact.</p>
            </div>
          ) : error ? (
            <div className="news-error">
               <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="var(--error)" strokeWidth="1">
                <circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line>
              </svg>
              <h3>Feed Interrupted</h3>
              <p>{error}</p>
              <button className="run-backtest-btn" style={{ marginTop: '20px' }} onClick={fetchNews}>Retry Connection</button>
            </div>
          ) : (
            <div className="news-grid">
              {news.map((item, i) => (
                <a 
                  key={item.id || i} 
                  className="news-card animate-fade-in" 
                  href={item.url} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  style={{ animationDelay: `${i * 0.05}s` }}
                >
                  <div className="news-card-meta">
                    <span className="news-source">{item.source}</span>
                    <span className={`impact-badge ${item.impact}`}>{item.impact}</span>
                  </div>
                  <h3>{item.title}</h3>
                  <div className="news-card-footer">
                    <span className="news-category">{item.category}</span>
                    <span className="news-time">
                      {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  </div>
                </a>
              ))}
            </div>
          )}
        </div>
      </div>
      <Footer isPro={isPro} wsStatus="connected" />
    </div>
  );
};

export default NewsTerminal;
