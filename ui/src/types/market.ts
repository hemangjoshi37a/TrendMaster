export interface Company {
  symbol: string;
  name: string;
}

export interface PredictionData {
  symbol: string;
  company_name: string;
  dates: string[];
  prices: number[];
  prediction_start_index: number;
  confidence_score?: number;
  warning?: string;
}

export interface RecentStock {
  symbol: string;
  price: number | null;
}

export interface MarketIndex {
  name: string;
  price: number;
  change_pct: number;
}

export type TimeframePeriod = '1mo' | '3mo' | '6mo' | '1y' | '2y' | '5y' | 'max';

export interface Alert {
  symbol: string;
  target: number;
  type: 'above' | 'below';
  active: boolean;
}

export interface AccountMeta {
  isPro: boolean;
  startDate: string;
}
