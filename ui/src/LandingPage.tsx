import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './LandingPage.css';

const LandingPage: React.FC = () => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [authMode, setAuthMode] = useState<'signin' | 'signup'>('signin');
  const [selectedPlan, setSelectedPlan] = useState<'basic' | 'pro'>('basic');
  const [scrollProgress, setScrollProgress] = useState(0);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [authError, setAuthError] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);

      const totalScroll = document.documentElement.scrollTop;
      const windowHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
      const scroll = `${totalScroll / windowHeight}`;
      setScrollProgress(Number(scroll));
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const openAuth = (mode: 'signin' | 'signup', plan: 'basic' | 'pro' = 'basic') => {
    setAuthMode(mode);
    setSelectedPlan(plan);
    setAuthError('');
    setShowAuthModal(true);
  };

  // Static demo credentials
  const DEMO_ACCOUNTS = [
    { email: 'free@trendmaster.ai', password: 'free123', isPro: false },
    { email: 'pro@trendmaster.ai', password: 'pro123', isPro: true },
  ];

  const handleAuthSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password) {
      setAuthError('Please fill in all required fields.');
      return;
    }

    if (authMode === 'signin') {
      const match = DEMO_ACCOUNTS.find(a => a.email === email && a.password === password);
      if (match) {
        navigate('/dashboard', { state: { isPro: match.isPro } });
      } else {
        setAuthError('Invalid credentials. Use free@trendmaster.ai / free123 or pro@trendmaster.ai / pro123');
      }
    } else {
      // Sign-up: trust the plan the user chose on the page
      navigate('/dashboard', { state: { isPro: selectedPlan === 'pro' } });
    }
  };

  const scrollTo = (id: string) => {
    const el = document.getElementById(id);
    if (el) {
      const topOffset = el.getBoundingClientRect().top + window.scrollY - 80; // Account for fixed navbar
      window.scrollTo({ top: topOffset, behavior: 'smooth' });
    }
  };

  const tickers = [
    { symbol: "NIFTY 50", price: "22,419.55", change: "+0.85%" },
    { symbol: "SENSEX", price: "73,651.35", change: "+0.72%" },
    { symbol: "BANKNIFTY", price: "47,560.00", change: "+1.20%" },
    { symbol: "RELIANCE", price: "2,950.20", change: "-0.45%" },
    { symbol: "HDFCBANK", price: "1,450.15", change: "+1.30%" },
    { symbol: "TCS", price: "4,120.60", change: "+0.25%" },
    { symbol: "INFY", price: "1,610.45", change: "-1.10%" },
    { symbol: "TATAMOTORS", price: "980.30", change: "+2.50%" },
  ];

  return (
    <div className="tv-landing">
      {/* Animated AI Background Orbs */}
      <div className="ai-background">
        <div className="ai-orb ai-orb-1"></div>
        <div className="ai-orb ai-orb-2"></div>
        <div className="ai-orb ai-orb-3"></div>
      </div>

      {/* Dynamic Scroll Progress Bar */}
      <div className="scroll-progress-bar" style={{ transform: `scaleX(${scrollProgress})` }}></div>

      {/* Navbar Minimalist */}
      <nav className={`tv-navbar ${isScrolled ? 'scrolled' : ''}`}>
        <div className="tv-nav-container">
          <div className="tv-nav-logo" onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}>
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 3v18h18" />
              <path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3" />
            </svg>
            <span className="logo-text">TrendMaster</span>
          </div>
          <div className="tv-nav-links">
            <span className="nav-item" onClick={() => scrollTo('features')} style={{ cursor: 'pointer' }}>Features</span>
            <span className="nav-item" onClick={() => scrollTo('how-it-works')} style={{ cursor: 'pointer' }}>Pipeline</span>
            <span className="nav-item" onClick={() => scrollTo('pricing')} style={{ cursor: 'pointer' }}>Pricing</span>
            <span className="nav-item" onClick={() => scrollTo('faq')} style={{ cursor: 'pointer' }}>FAQ</span>
          </div>
          <div className="tv-nav-actions">
            <button className="tv-btn-login" onClick={() => openAuth('signin')}>Log In</button>
            <button className="tv-btn-get-started" onClick={() => openAuth('signup', 'pro')}>Subscribe to Pro</button>
          </div>
        </div>
      </nav>

      {/* Ticker Tape */}
      <div className="tv-ticker-wrap">
        <div className="tv-ticker">
          {[...tickers, ...tickers, ...tickers].map((t, i) => (
            <div className="tv-ticker-item" key={i}>
              <span className="ticker-symbol">{t.symbol}</span>
              <span className="ticker-price">{t.price}</span>
              <span className={`ticker-change ${t.change.startsWith('+') ? 'positive' : 'negative'}`}>
                {t.change}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Hero Section */}
      <header className="tv-hero-section">
        <div className="hero-grid-bg"></div>
        <div className="tv-hero-content">
          <h1 className="hero-headline">
            Predict the trend /<br />
            <span className="hero-highlight">Then trade.</span>
          </h1>
          <p className="hero-subline">
            TrendMaster leverages state-of-the-art Transformer Networks to anticipate NSE market volatility with institutional-grade computational precision.
          </p>
          <div className="hero-cta">
            <button className="tv-btn-primary-large" onClick={() => openAuth('signup')}>Start Forecasting</button>
            <button className="tv-btn-secondary-large" onClick={() => scrollTo('accuracy')}>View Model Proofs</button>
          </div>
        </div>

        {/* Mock Chart App Window */}
        <div className="tv-mock-app">
          <div className="target-glow"></div>
          <div className="mock-toolbar">
            <div className="mock-dots">
              <span></span><span></span><span></span>
            </div>
            <div className="mock-title">NIFTY 50 • 1D • TrendMaster TransAm Prediction Engine</div>
          </div>
          <div className="mock-chart-area">
            <div className="mock-line-chart">
              <svg viewBox="0 0 1000 300" preserveAspectRatio="none">
                <path d="M0,250 C100,240 200,180 300,200 C400,220 500,100 600,150 C700,200 800,50 900,80 L1000,40 L1000,300 L0,300 Z" fill="url(#chart-gradient)" />
                <path d="M0,250 C100,240 200,180 300,200 C400,220 500,100 600,150 C700,200 800,50 900,80 L1000,40" fill="none" stroke="#2962FF" strokeWidth="4" />

                {/* AI Projection Line */}
                <path d="M700,200 C800,50 900,80 1000,40" fill="none" stroke="#089981" strokeWidth="4" strokeDasharray="10, 10" className="animated-dash" />
                <circle cx="700" cy="200" r="6" fill="#089981">
                  <animate attributeName="r" values="6;10;6" dur="2s" repeatCount="indefinite" />
                </circle>
                <text x="720" y="190" fill="#089981" fontSize="16" fontWeight="bold">Forecast Horizon</text>

                <defs>
                  <linearGradient id="chart-gradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="rgba(41, 98, 255, 0.4)" />
                    <stop offset="100%" stopColor="rgba(41, 98, 255, 0.0)" />
                  </linearGradient>
                </defs>
              </svg>
            </div>
          </div>
        </div>
      </header>

      {/* Massive Scrollable Sections */}
      <main className="tv-main-content" style={{ background: 'transparent', position: 'relative', zIndex: 10 }}>

        {/* Features: AI Forecasts */}
        <section id="features" className="tv-feature-block scroll-reveal">
          <div className="tv-feature-text">
            <h2>Features: Deep Learning Infrastructure.</h2>
            <p className="feature-desc">We process terabytes of raw tick data through specialized Transformer networks, capturing non-linear relationships that traditional lagging indicators completely miss.</p>
            <ul className="feature-list">
              <li>10-day forward-looking predictive windows</li>
              <li>Neural networks trained on decades of NSE pricing data</li>
              <li>Dynamically adaptive to macroeconomic volatility</li>
            </ul>
          </div>
          <div className="tv-feature-visual mock-panel-1">
            <div className="mock-chart-overlay"></div>
            {/* Simple Glassmorphism overlay describing a feature */}
            <div style={{ position: 'absolute', bottom: '20px', left: '20px', background: 'rgba(255,255,255,0.1)', backdropFilter: 'blur(10px)', padding: '16px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.2)', maxWidth: '300px' }}>
              <h4 style={{ margin: '0 0 8px 0', color: '#fff' }}>Pattern Recognition</h4>
              <p style={{ margin: 0, color: '#d1d4dc', fontSize: '0.85rem' }}>Detecting hidden fractal structures invisible to human charting analysis.</p>
            </div>
          </div>
        </section>

        {/* AI Models: High Confidence Signals */}
        <section id="models" className="tv-feature-block reverse scroll-reveal">
          <div className="tv-feature-visual mock-panel-2">
            <div className="mock-ai-nodes">
              <div className="node n1"></div>
              <div className="node n2"></div>
              <div className="node n3"></div>
              <svg><line x1="20" y1="20" x2="100" y2="100" stroke="#fff" strokeWidth="2" opacity="0.3" /></svg>
            </div>
            {/* Context metric */}
            <div style={{ position: 'absolute', top: '20px', right: '20px', background: 'rgba(8,153,129,0.2)', border: '1px solid #089981', padding: '12px 20px', borderRadius: '8px', color: '#089981', fontWeight: 'bold' }}>
              BULLISH SIGNAL: 94.2% CONFIDENCE
            </div>
          </div>
          <div className="tv-feature-text">
            <h2>Models: Institutional-Grade Signals.</h2>
            <p className="feature-desc">We democratize Wall Street technology. Our multi-head attention models assign real-time probabilistic scoring to impending price action, so you know strictly when momentum is structurally sound.</p>
            <ul className="feature-list">
              <li>Real-time confidence scoring prevents false signals</li>
              <li>Aggregating volume, momentum, and MACD divergence</li>
              <li>Calculated entry and trailing stop loss suggestions</li>
            </ul>
          </div>
        </section>

        {/* Accuracy: Live Integration */}
        <section id="accuracy" className="tv-feature-block scroll-reveal" style={{ marginBottom: 0 }}>
          <div className="tv-feature-text">
            <h2>Accuracy: Zero-Latency Execution.</h2>
            <p className="feature-desc">Millisecond tick streaming guarantees you act on our predictions exactly when the signal flashes. Our models recalculate trajectory forecasts simultaneously as fresh price action unfolds.</p>
            <ul className="feature-list">
              <li>WebSockets push ticks globally within ~25ms</li>
              <li>Immediate adjustments to the 'Forecast Horizon'</li>
              <li>Seamless, distraction-free analytical environment</li>
            </ul>
          </div>
          <div className="tv-feature-visual mock-panel-3">
            <div className="community-bubbles">
              <div className="bubble b1" style={{ borderLeftColor: '#089981' }}>"BankNifty resistance breakout verified: 85% Confidence"</div>
              <div className="bubble b2" style={{ borderRightColor: '#2962FF' }}>"RELIANCE historical fractal match detected."</div>
              <div className="bubble b3" style={{ borderLeftColor: '#f23645' }}>"Bearish divergence identified across Nifty IT sector."</div>
            </div>
          </div>
        </section>

        {/* About Us Section */}
        <section id="about" className="tv-feature-block reverse scroll-reveal" style={{ marginBottom: '120px' }}>
          <div className="tv-feature-visual" style={{ background: '#181c27', display: 'flex', flexDirection: 'column', padding: '40px', justifyContent: 'center', position: 'relative', overflow: 'hidden' }}>
            <div style={{ position: 'absolute', top: '-50%', left: '-50%', width: '200%', height: '200%', background: 'radial-gradient(circle, rgba(123,31,162,0.15) 0%, transparent 60%)', animation: 'spin 20s linear infinite' }}></div>
            <h3 style={{ color: '#fff', fontSize: '2rem', marginBottom: '16px', zIndex: 1 }}>Our Mission</h3>
            <p style={{ color: '#d1d4dc', fontSize: '1.1rem', lineHeight: '1.6', zIndex: 1 }}>
              "To democratize quantitative dominance. We built TrendMaster because retail traders deserve the identical computational edge that institutional funds have fiercely guarded for decades."
            </p>
            <div style={{ marginTop: '32px', zIndex: 1, display: 'flex', gap: '20px' }}>
              <div>
                <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#2962FF' }}>10+</div>
                <div style={{ fontSize: '0.85rem', color: '#8c9bad', textTransform: 'uppercase' }}>AI Researchers</div>
              </div>
              <div>
                <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#089981' }}>100M+</div>
                <div style={{ fontSize: '0.85rem', color: '#8c9bad', textTransform: 'uppercase' }}>Data Points Mapped</div>
              </div>
            </div>
          </div>
          <div className="tv-feature-text">
            <h2>About Us: The Quants Behind the AI.</h2>
            <p className="feature-desc">TrendMaster was forged by a collective of machine learning engineers, data scientists, and veteran quantitative analysts who saw a fundamental imbalance in the market ecosystem.</p>
            <ul className="feature-list">
              <li>Pioneers in applying Transformer networks to NSE feeds</li>
              <li>Committed to radical algorithmic transparency</li>
              <li>Relentlessly optimizing for predictive statistical accuracy</li>
            </ul>
          </div>
        </section>

        {/* Testimonials */}
        <section id="testimonials" className="tv-testimonials-section scroll-reveal">
          <h2>Engineered for Traders. Validated by Results.</h2>
          <div className="testimonials-track">
            <div className="testimonial-card">
              <div className="t-stars">★★★★★</div>
              <p className="t-quote">"The 10-day forward looking projection on BankNifty saved me from a massive fake-out. Absolutely unprecedented UI and latency."</p>
              <div className="t-author">— Rohan S., Options Trader</div>
            </div>
            <div className="testimonial-card">
              <div className="t-stars">★★★★★</div>
              <p className="t-quote">"TrendMaster feels like holding a Bloomberg terminal injected with next-level deep learning. The confidence score dictates my entire sizing strategy now."</p>
              <div className="t-author">— Anil K., Quantitative Analyst</div>
            </div>
            <div className="testimonial-card">
              <div className="t-stars">★★★★★</div>
              <p className="t-quote">"Smooth, flawless execution. It seamlessly bridges the gap between complex PyTorch inference and an incredibly intuitive visual dashboard."</p>
              <div className="t-author">— Priya M., Retail Investor</div>
            </div>
          </div>
        </section>

        {/* Advanced Tooling Layout */}
        <section id="advanced-tools" className="tv-tools-section scroll-reveal">
          <div className="tools-header">
            <h2>A Terminal Without Compromise</h2>
            <p>Every metric, every indicator, optimized for sheer computational speed.</p>
          </div>
          <div className="tv-bento-grid">
            <div className="bento-card col-span-2">
              <div className="bento-content">
                <h3>Deep Pattern Recognition</h3>
                <p>Detect hidden structural shifts using multi-layer transformer modules that learn dynamically from your favorite index feeds.</p>
              </div>
              <div className="bento-bg pattern-bg-1"></div>
            </div>
            <div className="bento-card">
              <div className="bento-content">
                <h3>Websocket Streaming</h3>
                <p>0.2ms latency to the dashboard.</p>
              </div>
              <div className="bento-bg pattern-bg-2"></div>
            </div>
            <div className="bento-card">
              <div className="bento-content">
                <h3>Risk Parameterization</h3>
                <p>Proprietary standard deviation bands to map out localized drawdowns.</p>
              </div>
              <div className="bento-bg pattern-bg-3"></div>
            </div>
            <div className="bento-card col-span-2">
              <div className="bento-content">
                <h3>Confidence Scoring Matrix</h3>
                <p>Trade only when the probabilities align perfectly. Mathematical confirmation before capital allocation.</p>
              </div>
              <div className="bento-bg pattern-bg-4"></div>
            </div>
          </div>
        </section>

        {/* FAQ Section */}
        <section id="faq" className="tv-faq-section scroll-reveal">
          <div className="faq-container">
            <h2>Frequently Asked Questions</h2>
            <div className="faq-item">
              <details>
                <summary>How does the AI model actually predict prices?</summary>
                <div className="faq-answer">We utilize a proprietary PyTorch-based Transformer architecture (TransAm) combined with internal positional embeddings. It ingests the last 30 intervals of OHLC data alongside technical indicators (RSI, EMA, MACD) to emit a 10-step forward looking probabilistic array.</div>
              </details>
            </div>
            <div className="faq-item">
              <details>
                <summary>Is the market data real-time?</summary>
                <div className="faq-answer">Yes. We connect to NSE exchange feeds via dual-redundant WebSockets, ensuring live tick streaming directly to your client instance with less than 25ms of latency.</div>
              </details>
            </div>
            <div className="faq-item">
              <details>
                <summary>What markets does TrendMaster support?</summary>
                <div className="faq-answer">Currently, TrendMaster Pro is optimized specifically for National Stock Exchange (NSE) indices and standard equity options. Global equities and FX commodities are planned for v2.0.</div>
              </details>
            </div>
            <div className="faq-item">
              <details>
                <summary>Do I need to know how to code to use the platform?</summary>
                <div className="faq-answer">Not at all. We have condensed complex neural network evaluation processes into a beautiful, point-and-click UI environment. Just search your ticker, and the mathematics happen instantly in the background.</div>
              </details>
            </div>
          </div>
        </section>

        {/* Pricing Section */}
        <section id="pricing" className="tv-pricing-section scroll-reveal">
          <h2 className="tv-section-heading">Simple, Transparent Pricing.</h2>
          <p className="tv-section-subheading">Choose the plan that best fits your trading style and ambition.</p>
          <div className="pricing-cards-container">
            {/* Basic Plan */}
            <div className="pricing-card">
              <h3 className="plan-title">Basic</h3>
              <p className="plan-description">10-day free trial. No credit card required.</p>
              <div className="price">$0<span className="price-suffix">/10 days</span></div>
              <ul className="plan-features">
                <li className="feature-item included">10-Day Forecast Chart (Day 1 Unlocked)</li>
                <li className="feature-item included">Historical Data Access</li>
                <li className="feature-item included">Basic Charting Tools</li>
                <li className="feature-item included">Email Support</li>
                <li className="feature-item excluded">Full 10-Day Forecast Table</li>
                <li className="feature-item excluded">AI Confidence Scores</li>
                <li className="feature-item excluded">Real-time WebSockets</li>
                <li className="feature-item excluded">Priority Support</li>
              </ul>
              <button className="tv-btn-secondary-large" onClick={() => openAuth('signup', 'basic')}>Start 10-Day Free Trial</button>
            </div>

            {/* Pro Plan */}
            <div className="pricing-card pro-plan">
              <div className="badge">MOST POPULAR</div>
              <h3 className="plan-title">Pro Terminal</h3>
              <p className="plan-description">Full access for 30 days. Cancel anytime.</p>
              <div className="price">$49<span className="price-suffix">/30 days</span></div>
              <ul className="plan-features">
                <li className="feature-item included">Full AI Forecast Horizon</li>
                <li className="feature-item included">Real-time WebSockets (25ms latency)</li>
                <li className="feature-item included">Advanced AI Confidence Scores</li>
                <li className="feature-item included">No Latency Limits</li>
                <li className="feature-item included">Full Historical Data Access</li>
                <li className="feature-item included">Premium Charting & Indicators</li>
                <li className="feature-item included">Calculated Entry/Exit Suggestions</li>
                <li className="feature-item included">Priority Email & Chat Support</li>
              </ul>
              <button className="tv-btn-primary-large" onClick={() => openAuth('signup', 'pro')}>Subscribe to Pro</button>
            </div>
          </div>
        </section>

        {/* Huge CTA Bottom */}
        <section className="tv-bottom-cta">
          <h2>Ready to trade with a true quantitative edge?</h2>
          <p>Join the future of retail trading. Predict earlier, execute flawlessly.</p>
          <button className="tv-btn-primary-mega" onClick={() => openAuth('signup')}>
            Initialize Forecasting Dashboard
          </button>
        </section>

      </main>

      {/* Footer */}
      <footer className="tv-footer" style={{ position: 'relative', zIndex: 10 }}>
        <div className="tv-footer-grid">
          <div className="footer-col brand-col">
            <div className="tv-nav-logo">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M3 3v18h18" />
                <path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3" />
              </svg>
              <span className="logo-text">TrendMaster</span>
            </div>
            <p style={{ marginTop: '20px', lineHeight: '1.7', color: '#d1d4dc', fontSize: '0.95rem' }}>
              Empowering global retail traders with Deep Learning and Institutional-Grade AI.
            </p>
          </div>
          <div className="footer-col">
            <h4>Technology</h4>
            <span>Transformer Architecture</span>
            <span>Feature Engineering</span>
            <span>Model Validation</span>
          </div>
          <div className="footer-col">
            <h4>Platform</h4>
            <span onClick={() => scrollTo('features')} style={{ cursor: 'pointer' }}>Features</span>
            <span onClick={() => scrollTo('accuracy')} style={{ cursor: 'pointer' }}>Accuracy Metrics</span>
            <span onClick={() => scrollTo('pricing')} style={{ cursor: 'pointer' }}>Pricing</span>
          </div>
          <div className="footer-col">
            <h4>Company</h4>
            <span onClick={() => scrollTo('about')} style={{ cursor: 'pointer' }}>About Us</span>
            <span onClick={() => scrollTo('faq')} style={{ cursor: 'pointer' }}>FAQ</span>
            <span onClick={() => openAuth('signin')} style={{ cursor: 'pointer' }}>Log In</span>
          </div>
        </div>
        <div className="footer-bottom">
          <p>© {new Date().getFullYear()} TrendMaster AI. All rights reserved.</p>
        </div>
      </footer>

      {/* Auth Modal */}
      {showAuthModal && (
        <div className="modal-backdrop">
          <div className="auth-card modal-card">
            <button className="close-btn" onClick={() => setShowAuthModal(false)}>×</button>

            <div className="logo-container">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="logo-icon">
                <path d="M3 3v18h18" />
                <path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3" />
              </svg>
            </div>

            <h2 className="auth-heading">{authMode === 'signin' ? 'Welcome Back' : 'Create Account'}</h2>
            {selectedPlan === 'pro' && (
              <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '12px' }}>
                <span style={{ background: 'linear-gradient(90deg,#2962FF,#089981)', color: '#fff', padding: '4px 14px', borderRadius: '20px', fontSize: '0.8rem', fontWeight: 'bold' }}>⚡ Pro Plan Selected</span>
              </div>
            )}
            <p className="auth-subheading" style={{ marginBottom: '24px' }}>
              {authMode === 'signin'
                ? 'Enter your details to access the AI dashboard.'
                : 'Sign up to start AI-powered trading predictions.'}
            </p>

            <form onSubmit={handleAuthSubmit} className="auth-form">
              {authMode === 'signup' && (
                <div className="input-group">
                  <label>Full Name</label>
                  <input type="text" placeholder="John Doe" required />
                </div>
              )}
              {authError && <div style={{ color: '#f23645', fontSize: '0.85rem', marginBottom: '8px', textAlign: 'center', background: 'rgba(242, 54, 69, 0.1)', padding: '8px', borderRadius: '4px' }}>{authError}</div>}
              <div className="input-group">
                <label>Email Address</label>
                <input type="email" placeholder="trade@trendmaster.ai" value={email} onChange={e => setEmail(e.target.value)} required />
              </div>
              <div className="input-group">
                <label>Password</label>
                <input type="password" placeholder="••••••••" value={password} onChange={e => setPassword(e.target.value)} required />
              </div>
              <button type="submit" className="primary-btn" style={selectedPlan === 'pro' ? { background: 'linear-gradient(90deg,#2962FF,#1E53E5)', boxShadow: '0 4px 16px rgba(41,98,255,0.4)' } : {}}>
                {authMode === 'signin' ? 'Sign In' : (selectedPlan === 'pro' ? 'Subscribe & Start Pro' : 'Sign Up Free')}
              </button>
            </form>
            <div className="auth-footer">
              <p>
                {authMode === 'signin' ? "Don't have an account?" : "Already have an account?"}
                <button className="toggle-btn" onClick={() => setAuthMode(authMode === 'signin' ? 'signup' : 'signin')}>
                  {authMode === 'signin' ? 'Sign up' : 'Sign in'}
                </button>
              </p>
            </div>
          </div>
        </div>
      )}

    </div>
  );
};

export default LandingPage;
