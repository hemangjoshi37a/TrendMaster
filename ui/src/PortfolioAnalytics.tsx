import React from 'react';
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Filler
} from 'chart.js';
import { Doughnut, Line } from 'react-chartjs-2';

ChartJS.register(
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Filler
);

// Top NSE Stock Sector Mapping
const SECTOR_MAP: { [key: string]: string } = {
  'RELIANCE': 'Energy',
  'TCS': 'IT',
  'INFY': 'IT',
  'HDFCBANK': 'Finance',
  'ICICIBANK': 'Finance',
  'SBIN': 'Finance',
  'BHARTIARTL': 'Telecom',
  'ITC': 'FMCG',
  'HINDUNILVR': 'FMCG',
  'LT': 'Engineering',
  'AXISBANK': 'Finance',
  'KOTAKBANK': 'Finance',
  'WIPRO': 'IT',
  'HCLTECH': 'IT',
  'TITAN': 'Consumer Durables',
  'ASIANPAINT': 'Consumer Durables',
  'ULTRACEMCO': 'Construction',
  'BAJFINANCE': 'Finance',
  'BAJAJFINSV': 'Finance',
  'MARUTI': 'Auto',
  'TATAMOTORS': 'Auto',
  'M&M': 'Auto',
  'JSWSTEEL': 'Metals',
  'TATASTEEL': 'Metals',
  'SUNPHARMA': 'Pharma',
  'DRREDDY': 'Pharma',
  'CIPLA:': 'Pharma',
  'ADANIENT': 'Energy',
  'ADANIPORTS': 'Infrastructure',
  'COALINDIA': 'Energy',
  'NTPC': 'Energy',
  'POWERGRID': 'Energy',
  'ONGC': 'Energy',
  'GRASIM': 'Materials',
  'HINDALCO': 'Metals',
  'APOLLOHOSP': 'Healthcare',
  'BAJAJ-AUTO': 'Auto',
  'EICHERMOT': 'Auto',
  'HEROMOTOCO': 'Auto',
  'BPCL': 'Energy',
  'IOC': 'Energy'
};

interface Position {
  symbol: string;
  qty: number;
  avgPrice: number;
}

interface PortfolioAnalyticsProps {
  positions: Position[];
  livePrices: { [symbol: string]: number };
  equityHistory: { date: string; value: number }[];
}

const PortfolioAnalytics: React.FC<PortfolioAnalyticsProps> = ({ 
  positions, 
  livePrices, 
  equityHistory 
}) => {
  
  // 1. Calculate Sector Distribution
  const sectorWeights: { [sector: string]: number } = {};
  let totalCap = 0;

  positions.forEach(pos => {
    const sector = SECTOR_MAP[pos.symbol] || 'Others';
    const currentPrice = livePrices[pos.symbol] || pos.avgPrice;
    const value = pos.qty * currentPrice;
    
    sectorWeights[sector] = (sectorWeights[sector] || 0) + value;
    totalCap += value;
  });

  const doughnutData = {
    labels: Object.keys(sectorWeights),
    datasets: [{
      label: 'Allocation (₹)',
      data: Object.values(sectorWeights),
      backgroundColor: [
        '#2962FF', '#0ECB81', '#F6465D', '#FCD535', '#9C27B0', 
        '#FF5722', '#00BCD4', '#795548', '#607D8B'
      ],
      borderColor: 'rgba(20, 24, 34, 0.8)',
      borderWidth: 2,
    }]
  };

  const doughnutOptions: any = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right',
        labels: { color: '#848E9C', usePointStyle: true, padding: 20, font: { size: 12 } }
      },
      tooltip: {
        callbacks: {
          label: (ctx: any) => {
            const val = ctx.raw;
            const pct = ((val / totalCap) * 100).toFixed(1);
            return ` ₹${val.toLocaleString('en-IN')} (${pct}%)`;
          }
        }
      }
    },
    cutout: '70%'
  };

  // 2. Format Equity History
  const lineData = {
    labels: equityHistory.map(h => h.date),
    datasets: [{
      label: 'Total Equity',
      data: equityHistory.map(h => h.value),
      fill: true,
      borderColor: '#2962FF',
      backgroundColor: 'rgba(41, 98, 255, 0.1)',
      tension: 0.4,
      pointRadius: 0,
      pointHoverRadius: 5,
    }]
  };

  const lineOptions: any = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#848E9C' } },
      x: { grid: { display: false }, ticks: { display: false } }
    },
    plugins: {
      legend: { display: false },
      tooltip: { mode: 'index', intersect: false }
    }
  };

  return (
    <div className="analytics-grid">
      <div className="analytics-card">
        <h3>Sector Diversification</h3>
        <div className="chart-wrapper">
          {positions.length > 0 ? (
            <Doughnut data={doughnutData} options={doughnutOptions} />
          ) : (
            <div className="chart-placeholder">No active positions.</div>
          )}
        </div>
      </div>

      <div className="analytics-card">
        <h3>Equity Growth (Performance)</h3>
        <div className="chart-wrapper">
          {equityHistory.length > 1 ? (
             <Line data={lineData} options={lineOptions} />
          ) : (
             <div className="chart-placeholder">Tracking your account growth...</div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PortfolioAnalytics;
