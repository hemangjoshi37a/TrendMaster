import React from 'react';
import LineChart from '../../LineChart';
import { PredictionData, TimeframePeriod, Alert } from '../../types/market';

interface ChartPanelProps {
  prediction: PredictionData;
  isPro: boolean;
  livePrice: number | null;
  prevPrice: number | null;
  wsStatus: string;
  alerts: Alert[];
  loading: boolean;
  selectedTimeframe: TimeframePeriod;
  timeframeMap: { label: string; period: TimeframePeriod }[];
  onTimeframeChange: (period: TimeframePeriod) => void;
  onAlertClick: () => void;
}

export const ChartPanel: React.FC<ChartPanelProps> = ({
  prediction,
  isPro,
  livePrice,
  prevPrice,
  wsStatus,
  alerts,
  loading,
  selectedTimeframe,
  timeframeMap,
  onTimeframeChange,
  onAlertClick
}) => {
  const priceChange = livePrice && prevPrice ? livePrice - prevPrice : 0;
  const priceChangePct = prevPrice ? (priceChange / prevPrice) * 100 : 0;

  return (
    <div className="chart-panel">
      <div className="chart-header">
        <div className="stock-info">
          <div className="stock-symbol">
            {prediction.symbol}
            {isPro && (
              <button 
                className={`alert-bell-btn ${alerts.some(a => a.symbol === prediction.symbol && a.active) ? 'has-active' : ''}`}
                onClick={onAlertClick}
                title="Set Price Alert"
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path>
                  <path d="M13.73 21a2 2 0 0 1-3.46 0"></path>
                </svg>
              </button>
            )}
            {wsStatus === 'connected' && (
              <span className="ws-status live"><span className="pulse-dot"></span> LIVE</span>
            )}
            {wsStatus === 'reconnecting' && (
              <span className="ws-status reconnecting">RECONNECTING...</span>
            )}
          </div>
          <div className="stock-name">{prediction.company_name}</div>
        </div>
        <div className="stock-price-container">
          <div className="current-price">
            {livePrice ? livePrice.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : prediction.prices[prediction.prediction_start_index - 1]?.toFixed(2) || "---"}
          </div>
          {priceChange !== 0 && (
            <div className={`price-change ${priceChange >= 0 ? 'up' : 'down'}`}>
              {priceChange >= 0 ? '▲' : '▼'} {Math.abs(priceChange).toFixed(2)} ({Math.abs(priceChangePct).toFixed(2)}%)
            </div>
          )}
        </div>
      </div>

      <div className="chart-toolbar">
        <div className="timeframe-selector">
          {timeframeMap.map(tf => (
            <button
              key={tf.period}
              className={`tf-btn ${selectedTimeframe === tf.period ? 'active' : ''}`}
              onClick={() => onTimeframeChange(tf.period)}
              disabled={loading}
            >
              {tf.label}
            </button>
          ))}
        </div>
        <div className="chart-legend">
          <div className="legend-item">
            <div className="legend-color" style={{ background: '#2962FF' }}></div>
            Historical Data
          </div>
          <div className="legend-item">
            <div className="legend-color" style={{ background: '#F23645' }}></div>
            Transformer Forecast
          </div>
        </div>
      </div>

      {prediction.warning && (
        <div style={{
          margin: '0 20px 8px',
          padding: '8px 12px',
          background: 'rgba(255, 152, 0, 0.12)',
          border: '1px solid rgba(255, 152, 0, 0.35)',
          borderRadius: 'var(--radius-sm)',
          fontSize: '0.78rem',
          color: 'var(--warning)',
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}>
          <span>⚠</span>
          <span>{prediction.warning} — Showing historical data only.</span>
        </div>
      )}

      <div className="chart-container-wrapper animate-fade-in">
        <LineChart data={prediction} isPro={isPro} />
      </div>
    </div>
  );
};
