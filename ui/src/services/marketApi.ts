import { PredictionData, MarketIndex, TimeframePeriod } from '../types/market';

/**
 * Service to handle all NSE Market API calls
 */
export const marketApi = {
  /**
   * Fetches the 10-day Transformer prediction for a specific symbol
   */
  async getPrediction(symbol: string, period: TimeframePeriod): Promise<PredictionData> {
    const response = await fetch(`/api/predict?stock_symbol=${encodeURIComponent(symbol)}&period=${period}`);
    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || `Failed to fetch prediction for ${symbol}`);
    }
    return response.json();
  },

  /**
   * Fetches the market overview (NIFTY 50, SENSEX, etc.)
   */
  async getMarketOverview(): Promise<MarketIndex[]> {
    const res = await fetch('/api/market-overview');
    if (!res.ok) {
      throw new Error('Failed to fetch market overview');
    }
    return res.json();
  },

  /**
   * Authenticates or initializes account metadata
   */
  getAccountMeta() {
    const raw = localStorage.getItem('tm_account');
    return raw ? JSON.parse(raw) : null;
  },

  /**
   * Saves account metadata to persistence
   */
  saveAccountMeta(meta: { isPro: boolean; startDate: string }) {
    localStorage.setItem('tm_account', JSON.stringify(meta));
  }
};
