import { useState, useEffect } from 'react';
import { marketApi } from '../services/marketApi';
import { MarketIndex } from '../types/market';

/**
 * Hook to manage market status (Open/Closed) and Index data polling
 */
export const useMarketData = () => {
  const [marketIndices, setMarketIndices] = useState<MarketIndex[]>([]);
  const [marketOpen, setMarketOpen] = useState<boolean>(true);

  const fetchOverview = async () => {
    try {
      const data = await marketApi.getMarketOverview();
      setMarketIndices(data);
    } catch (e) {
      console.error('Market overview fetch failed:', e);
    }
  };

  useEffect(() => {
    // Check Market Status logic
    const checkMarket = () => {
      const now = new Date();
      // IST conversion
      const istTime = new Date(now.toLocaleString('en-US', { timeZone: 'Asia/Kolkata' }));
      const day = istTime.getDay();
      const hour = istTime.getHours();
      const min = istTime.getMinutes();
      
      const isWeekend = day === 0 || day === 6;
      const isWorkingHours = (hour === 9 && min >= 15) || (hour > 9 && hour < 15) || (hour === 15 && min <= 30);
      
      setMarketOpen(!isWeekend && isWorkingHours);
    };

    checkMarket();
    fetchOverview();

    const marketInterval = setInterval(fetchOverview, 60000);
    const statusInterval = setInterval(checkMarket, 60000);

    return () => {
      clearInterval(marketInterval);
      clearInterval(statusInterval);
    };
  }, []);

  return { marketIndices, marketOpen };
};
