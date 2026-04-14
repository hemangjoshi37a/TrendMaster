import React, { useState, useEffect, useRef } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import './TopNav.css';

interface Position { symbol: string; qty: number; avgPrice: number; takeProfit?: number; stopLoss?: number; }
interface PaperTradingState { cash: number; positions: Position[]; history: any[]; }
interface Company { symbol: string; name: string; }

interface TopNavProps {
  activePage: 'markets' | 'sandbox' | 'paper-trading' | 'backtest' | 'news' | 'portfolio' | 'multiverse' | 'wealth-architect';
  isPro: boolean;
  hidePnl?: boolean;
  totalEquityOverride?: number;
  totalUnrealizedOverride?: number;
}

const NAV_LINKS = [
  { key: 'markets',        label: 'Markets',    href: '/dashboard'     },
  { key: 'news',           label: 'News',        href: '/news'          },
  { key: 'paper-trading',  label: 'Trading',     href: '/paper-trading' },
  { key: 'portfolio',      label: 'Portfolio',   href: '/portfolio'     },
] as const;

const PRO_LINKS = [
  { key: 'multiverse',       label: 'Multiverse',  href: '/multiverse'       },
  { key: 'wealth-architect', label: 'AI Wealth',   href: '/wealth-architect' },
  { key: 'backtest',         label: 'Backtest',    href: '/backtest'         },
] as const;

const TopNav: React.FC<TopNavProps> = ({
  activePage,
  isPro: isProProp,
  hidePnl = false,
  totalEquityOverride,
  totalUnrealizedOverride,
}) => {
  const navigate = useNavigate();
  const [isPro, setIsPro] = useState(isProProp);
  const [state, setState] = useState<PaperTradingState>(() => {
    const saved = localStorage.getItem('tm_paper_trading');
    if (saved) return JSON.parse(saved);
    return { cash: 1000000, positions: [], history: [] };
  });
  const [livePrices, setLivePrices] = useState<{ [symbol: string]: number }>({});
  const [query, setQuery] = useState('');
  const [suggestions, setSuggestions] = useState<Company[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [marketOpen, setMarketOpen] = useState(true);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const searchRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const raw = localStorage.getItem('tm_account');
    if (raw) {
      const meta = JSON.parse(raw);
      if (meta.isPro) setIsPro(true);
    }
  }, []);

  useEffect(() => {
    const checkMarket = () => {
      const ist = new Date(new Date().toLocaleString('en-US', { timeZone: 'Asia/Kolkata' }));
      const day = ist.getDay(), h = ist.getHours(), m = ist.getMinutes();
      setMarketOpen(day !== 0 && day !== 6 && ((h === 9 && m >= 15) || (h > 9 && h < 15) || (h === 15 && m <= 30)));
    };
    checkMarket();
    const id = setInterval(checkMarket, 60000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (query.length > 1) {
      const fetchSuggestions = async () => {
        try {
          const res = await fetch(`/api/search?query=${encodeURIComponent(query)}`);
          if (res.ok) {
            const data = await res.json();
            setSuggestions(data);
            setShowSuggestions(data.length > 0);
          }
        } catch (e) {}
      };
      const t = setTimeout(fetchSuggestions, 300);
      return () => clearTimeout(t);
    } else {
      setSuggestions([]);
      setShowSuggestions(false);
    }
  }, [query]);

  useEffect(() => {
    const handleOutside = (e: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(e.target as Node)) {
        setShowSuggestions(false);
      }
    };
    document.addEventListener('mousedown', handleOutside);
    return () => document.removeEventListener('mousedown', handleOutside);
  }, []);

  useEffect(() => {
    if (state.positions.length === 0 || hidePnl || totalEquityOverride !== undefined) return;
    const updatePrices = async () => {
      for (const pos of state.positions) {
        try {
          const resp = await fetch(`/api/quote?symbol=${encodeURIComponent(pos.symbol)}`);
          if (resp.ok) {
            const data = await resp.json();
            setLivePrices(prev => ({ ...prev, [pos.symbol]: data.price }));
          }
        } catch (e) {}
      }
    };
    updatePrices();
    const id = setInterval(updatePrices, 15000);
    return () => clearInterval(id);
  }, [state.positions, hidePnl, totalEquityOverride]);

  let totalEquity = state.cash;
  let totalUnrealized = 0;
  if (totalEquityOverride !== undefined && totalUnrealizedOverride !== undefined) {
    totalEquity = totalEquityOverride;
    totalUnrealized = totalUnrealizedOverride;
  } else if (!hidePnl) {
    state.positions.forEach(pos => {
      const cur = livePrices[pos.symbol] || pos.avgPrice;
      totalEquity += pos.qty * cur;
      totalUnrealized += (cur - pos.avgPrice) * pos.qty;
    });
  }

  const handleSelect = (symbol: string) => {
    setShowSuggestions(false);
    setQuery('');
    navigate('/dashboard', { state: { isPro, searchSymbol: symbol } });
    if (activePage === 'markets') window.location.reload();
  };

  const handleLogout = () => {
    localStorage.removeItem('tm_account');
    navigate('/');
  };

  const formatINR = (n: number) => n.toLocaleString('en-IN', { maximumFractionDigits: 0 });

  return (
    <nav className="tn-nav">
      <div className="tn-container">

        {/* ── LEFT: Logo + Nav Links ── */}
        <div className="tn-left">
          <Link to="/" className="tn-brand">
            {/* Chart icon */}
            <svg className="tn-brand-icon" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 3v18h18" />
              <path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3" />
            </svg>
            <span className="tn-brand-name">TrendMaster</span>
            {isPro && <span className="tn-pro-badge">PRO</span>}
          </Link>

          <div className="tn-divider" />

          <nav className="tn-links" aria-label="Main navigation">
            {NAV_LINKS.map(({ key, label, href }) => (
              <Link
                key={key}
                to={href}
                state={{ isPro }}
                className={`tn-link${activePage === key ? ' tn-link--active' : ''}`}
              >
                {label}
              </Link>
            ))}
            {isPro && PRO_LINKS.map(({ key, label, href }) => (
              <Link
                key={key}
                to={href}
                state={{ isPro }}
                className={`tn-link${activePage === key ? ' tn-link--active' : ''}`}
              >
                {label}
              </Link>
            ))}
          </nav>
        </div>

        {/* ── CENTER: Search ── */}
        <div className="tn-center" ref={searchRef}>
          <div className="tn-search">
            <svg className="tn-search-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
            <input
              type="text"
              placeholder="Search markets…"
              value={query}
              onChange={e => setQuery(e.target.value)}
              onFocus={() => suggestions.length > 0 && setShowSuggestions(true)}
              aria-label="Search stocks"
              autoComplete="off"
            />
            {query && (
              <button className="tn-search-clear" onClick={() => { setQuery(''); setShowSuggestions(false); }} aria-label="Clear search">
                ×
              </button>
            )}
          </div>
          {showSuggestions && (
            <ul className="tn-suggestions" role="listbox">
              {suggestions.map(c => (
                <li key={c.symbol} role="option" onClick={() => handleSelect(c.symbol)}>
                  <span className="tn-sym">{c.symbol}</span>
                  <span className="tn-name">{c.name}</span>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* ── RIGHT: Equity + Status + Logout ── */}
        <div className="tn-right">
          {!hidePnl && (
            <div className="tn-equity-chip">
              <div className="tn-chip-col">
                <span className="tn-chip-label">EQUITY</span>
                <span className="tn-chip-val">₹{formatINR(totalEquity)}</span>
              </div>
              <span className="tn-chip-sep" />
              <div className="tn-chip-col">
                <span className="tn-chip-label">P&amp;L</span>
                <span className={`tn-chip-val ${totalUnrealized >= 0 ? 'pos' : 'neg'}`}>
                  {totalUnrealized >= 0 ? '+' : ''}₹{formatINR(totalUnrealized)}
                </span>
              </div>
            </div>
          )}

          <div className={`tn-market-status ${marketOpen ? 'open' : 'closed'}`}>
            <span className="tn-status-dot" />
            {marketOpen ? 'LIVE' : 'CLOSED'}
          </div>

          <button className="tn-logout" onClick={handleLogout} aria-label="Log out">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
              <polyline points="16 17 21 12 16 7" />
              <line x1="21" y1="12" x2="9" y2="12" />
            </svg>
            <span>Logout</span>
          </button>
        </div>

      </div>
    </nav>
  );
};

export default TopNav;
