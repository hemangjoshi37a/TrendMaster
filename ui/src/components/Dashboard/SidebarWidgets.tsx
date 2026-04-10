import React from 'react';
import SectorHeatmap from '../../SectorHeatmap';
import { PredictionData, MarketIndex, RecentStock, Alert, TimeframePeriod } from '../../types/market';

interface SidebarWidgetsProps {
  prediction: PredictionData | null;
  selectedTimeframe: TimeframePeriod;
  isPro: boolean;
  marketIndices: MarketIndex[];
  recentStocks: RecentStock[];
  alerts: Alert[];
  onStockSelect: (symbol: string) => void;
  onRemoveAlert: (symbol: string, target: number) => void;
  onUpgradeClick: () => void;
}

export const SidebarWidgets: React.FC<SidebarWidgetsProps> = ({
  prediction,
  selectedTimeframe,
  isPro,
  marketIndices,
  recentStocks,
  alerts,
  onStockSelect,
  onRemoveAlert,
  onUpgradeClick
}) => {
  return (
    <div className="sidebar">
      {prediction && (
        <div className="widget">
          <div className="widget-title">Model Specifications</div>
          {prediction.warning ? (
            <p className="ai-summary" style={{ color: 'var(--warning)', fontWeight: 500 }}>
              Forecast unavailable. Showing historical data only.
            </p>
          ) : (
            <p className="ai-summary">
              TransAm architecture analyzing attention interactions across <b>{selectedTimeframe.toUpperCase()}</b> historical patterns.
            </p>
          )}
          <div className="stat-row">
            <span className="stat-label">Model Type</span>
            <span className="stat-value">Transformer (Multi-Head)</span>
          </div>
          <div className="stat-row">
            <span className="stat-label">Features Used</span>
            <span className="stat-value">Price, Volume, Tech Ind.</span>
          </div>
          <div className="stat-row">
            <span className="stat-label">Prediction Horizon</span>
            <span className="stat-value">10 Trading Days</span>
          </div>
          <div className="stat-row" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '8px', position: 'relative' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%' }} className={!isPro ? 'locked-blur' : ''}>
              <span className="stat-label">Confidence Score</span>
              <span className="stat-value" style={{
                color: (prediction.confidence_score ?? 0) >= 65 ? 'var(--success)'
                     : (prediction.confidence_score ?? 0) >= 40 ? 'var(--warning)'
                     : 'var(--error)'
              }}>
                {prediction.confidence_score !== undefined ? `${prediction.confidence_score}%` : 'N/A'}
              </span>
            </div>
            <div style={{ width: '100%', height: '6px', background: 'var(--border)', borderRadius: '3px', overflow: 'hidden' }} className={!isPro ? 'locked-blur' : ''}>
              <div style={{
                width: `${prediction.confidence_score ?? 0}%`,
                height: '100%',
                background: (prediction.confidence_score ?? 0) >= 65 ? 'var(--success)'
                          : (prediction.confidence_score ?? 0) >= 40 ? 'var(--warning)'
                          : 'var(--error)',
                transition: 'width 0.6s ease'
              }}></div>
            </div>
            {!isPro && (
              <div className="pro-overlay-lock">
                <button className="tv-btn-login" onClick={onUpgradeClick} style={{fontSize: '0.8rem', padding: '4px 8px', background: 'rgba(41, 98, 255, 0.2)', borderRadius: '4px', border: '1px solid #2962FF', color: '#fff'}}>
                  🔒 Upgrade to Pro
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {isPro && alerts.length > 0 && (
        <div className="widget">
          <div className="widget-title">Active Alerts</div>
          <div className="alerts-list">
            {alerts.map((a, i) => (
              <div key={`${a.symbol}-${a.target}`} className={`alert-item ${a.active ? 'active' : 'triggered'}`}>
                <div className="ai-info">
                  <span className="ai-sym">{a.symbol}</span>
                  <span className="ai-target">{a.type === 'above' ? '≥' : '≤'} ₹{a.target.toLocaleString('en-IN')}</span>
                </div>
                <div className="ai-actions">
                  {!a.active && <span className="ai-status">HIT</span>}
                  <button className="ai-del" onClick={() => onRemoveAlert(a.symbol, a.target)}>×</button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {isPro && (
        <SectorHeatmap isPro={isPro} />
      )}

      <div className="widget">
        <div className="widget-title">Market Overview</div>
        <div className="indices-list">
          {marketIndices.length > 0 ? marketIndices.map(idx => (
            <div className="index-row" key={idx.name}>
              <span className="index-name">{idx.name}</span>
              <div className="index-values">
                <span className="index-price">{idx.price.toLocaleString('en-IN')}</span>
                <span className={`index-change ${idx.change_pct >= 0 ? 'up' : 'down'}`}>
                  {idx.change_pct >= 0 ? '+' : ''}{idx.change_pct.toFixed(2)}%
                </span>
              </div>
            </div>
          )) : (
            <div className="index-row"><span className="index-name" style={{ color: 'var(--text-muted)' }}>Loading market data...</span></div>
          )}
        </div>
      </div>
      
      <div className="widget">
        <div className="widget-title">Recent History</div>
        <div className="recent-list">
          {recentStocks.length > 0 ? recentStocks.map(s => (
            <div key={s.symbol} className="recent-chip" onClick={() => onStockSelect(s.symbol)}>
              {s.symbol}
            </div>
          )) : (
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>No recent searches</span>
          )}
        </div>
      </div>
    </div>
  );
};
