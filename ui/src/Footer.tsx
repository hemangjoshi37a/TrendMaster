import React from 'react';

interface FooterProps {
  isPro: boolean;
  wsStatus: 'connected' | 'disconnected' | 'reconnecting';
}

const Footer: React.FC<FooterProps> = ({ isPro, wsStatus }) => {
  return (
    <footer className={`footer ${isPro ? 'pro-footer' : ''}`}>
      <div className="footer-content">
        <div className="footer-grid">
          <div className="footer-section brand-section">
            <div className="footer-logo">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
              </svg>
              TrendMaster <span>{isPro ? 'PRO' : 'FREE'}</span>
            </div>
            <p className="footer-desc">
              Empowering NSE traders with next-gen Transformer AI. Real-time patterns, 10-day forecasts, and high-confidence signals.
            </p>
            <div className="footer-socials">
              <span className="social-icon">𝕏</span>
              <span className="social-icon">in</span>
              <span className="social-icon">✉</span>
            </div>
          </div>

          <div className="footer-section">
            <h4>Platform</h4>
            <ul>
              <li><a href="#markets">Markets</a></li>
              <li><a href="#signals">AI Signals</a></li>
              <li><a href="#heatmap">Sector Heatmap</a></li>
              <li><a href="#alerts">Price Alerts</a></li>
            </ul>
          </div>

          <div className="footer-section">
            <h4>Resources</h4>
            <ul>
              <li><a href="#help">Help Center</a></li>
              <li><a href="#api">API Documentation</a></li>
              <li><a href="#blog">Market Insights</a></li>
              <li><a href="#status">System Status</a></li>
            </ul>
          </div>

          <div className="footer-section status-section">
            <h4>System Status</h4>
            <div className="status-pills">
              <div className="status-pill">
                <span className={`dot ${wsStatus === 'connected' ? 'online' : 'reconnecting'}`}></span>
                Live Feed: {wsStatus === 'connected' ? 'Stable' : 'Connecting...'}
              </div>
              <div className="status-pill">
                <span className="dot online"></span>
                AI Core: Operational
              </div>
              <div className="status-pill">
                <span className="dot online"></span>
                API: 12ms
              </div>
            </div>
          </div>
        </div>

        <div className="footer-bottom">
          <div className="footer-legal">
            <span>© 2026 TrendMaster AI. All rights reserved.</span>
            <div className="legal-links">
              <a href="#privacy">Privacy</a>
              <a href="#terms">Terms</a>
              <a href="#disclaimer">Disclaimer</a>
            </div>
          </div>
          <div className="footer-disclaimer">
            <b>Disclaimer:</b> Trading involves significant risk. AI predictions are based on historical patterns and are for educational purposes only. Always consult a financial advisor.
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
