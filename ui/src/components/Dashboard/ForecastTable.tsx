import React from 'react';
import { PredictionData } from '../../types/market';

interface ForecastTableProps {
  prediction: PredictionData;
  isPro: boolean;
  onUpgradeClick: () => void;
}

export const ForecastTable: React.FC<ForecastTableProps> = ({
  prediction,
  isPro,
  onUpgradeClick
}) => {
  return (
    <div className="table-panel">
      <div className="panel-title">Forecast Data</div>
      <div className="table-wrapper">
        <table className="prediction-grid">
          <thead>
            <tr>
              <th>Date</th>
              <th>Target Price</th>
              <th>Change</th>
              <th>AI Signal</th>
            </tr>
          </thead>
          <tbody>
            {prediction.dates.slice(prediction.prediction_start_index, prediction.prediction_start_index + 10).map((date, i) => {
              const idx = prediction.prediction_start_index + i;
              const price = prediction.prices[idx];
              const prevPriceVal = prediction.prices[idx - 1];
              if (price === undefined || prevPriceVal === undefined) return null;
              const change = price - prevPriceVal;
              const changePct = (change / prevPriceVal) * 100;

              const isLockedRow = !isPro && i > 0;

              return (
                <tr key={date} className={`animate-fade-in-delayed ${isLockedRow ? 'locked-blur' : ''}`} style={{ animationDelay: `${i * 0.05}s` }}>
                  <td>{new Date(date).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })}</td>
                  <td style={{ color: 'var(--text-bright)' }}>{isLockedRow ? '₹₹,₹₹₹.₹₹' : price.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
                  <td style={{ color: change >= 0 && !isLockedRow ? 'var(--success)' : change < 0 && !isLockedRow ? 'var(--error)' : 'var(--text-muted)' }}>
                    {!isLockedRow ? `${change >= 0 ? '+' : ''}${changePct.toFixed(2)}%` : '---'}
                  </td>
                  <td>
                    {!isLockedRow ? (
                      <span className={`trend-badge ${change >= 0 ? 'bullish' : 'bearish'}`}>
                        {change >= 0 ? 'BULL' : 'BEAR'}
                      </span>
                    ) : (
                      <span className="trend-badge" style={{background: 'var(--border)', color: 'var(--text-muted)'}}>LOCK</span>
                    )}
                  </td>
                </tr>
              );
            })}
            {!isPro && (
              <tr className="unlock-row-cta">
                <td colSpan={4}>
                  <button className="tv-btn-get-started" onClick={onUpgradeClick}>
                    Unlock 10-Day Forecast
                  </button>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
