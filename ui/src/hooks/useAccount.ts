import { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { marketApi } from '../services/marketApi';

/**
 * Hook to manage user account state, subscription level, and trial period
 */
export const useAccount = () => {
  const location = useLocation();
  const [isPro, setIsPro] = useState<boolean>(() => {
    const meta = marketApi.getAccountMeta();
    if (meta) return meta.isPro;
    return location.state?.isPro || false;
  });

  const TRIAL_DAYS = 10;
  const PRO_DAYS = 30;

  useEffect(() => {
    if (location.state?.isPro !== undefined) {
      const existing = marketApi.getAccountMeta();
      if (!existing || existing.isPro !== location.state.isPro) {
        marketApi.saveAccountMeta({ 
            isPro: location.state.isPro, 
            startDate: new Date().toISOString() 
        });
      }
    }
  }, [location.state?.isPro]);

  const meta = marketApi.getAccountMeta();
  const startDate = meta ? new Date(meta.startDate) : new Date();
  const daysSinceStart = Math.floor((Date.now() - startDate.getTime()) / (1000 * 60 * 60 * 24));
  const daysAllowed = isPro ? PRO_DAYS : TRIAL_DAYS;
  const daysRemaining = Math.max(0, daysAllowed - daysSinceStart);
  const isExpired = daysRemaining === 0;

  return { isPro, setIsPro, daysRemaining, isExpired };
};
