import React from 'react';

interface ExpiryBannerProps {
  isPro: boolean;
  daysRemaining: number;
  isExpired: boolean;
  onUpgradeClick: () => void;
}

export const ExpiryBanner: React.FC<ExpiryBannerProps> = ({
  isPro,
  daysRemaining,
  isExpired,
  onUpgradeClick
}) => {
  if (isExpired) return null;

  return (
    <div style={{
      background: isPro
        ? 'linear-gradient(90deg, var(--brand-accent-glow), transparent)'
        : daysRemaining <= 3
          ? 'linear-gradient(90deg, hsla(355, 78%, 56%, 0.15), transparent)'
          : 'linear-gradient(90deg, var(--brand-primary-glow), transparent)',
      borderBottom: '1px solid',
      borderColor: isPro ? 'var(--brand-accent)' : daysRemaining <= 3 ? 'var(--error)' : 'var(--brand-primary)',
      padding: '8px 24px',
      fontSize: '0.85rem',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      backdropFilter: 'var(--glass-blur)',
      zIndex: 100
    }}>
      <span style={{ color: 'var(--text-main)', fontWeight: 500 }}>
        {isPro
          ? `✅ Pro Terminal — ${daysRemaining} day${daysRemaining !== 1 ? 's' : ''} remaining in your subscription`
          : `⏳ Free Trial — ${daysRemaining} day${daysRemaining !== 1 ? 's' : ''} remaining. Upgrade to keep full access.`}
      </span>
      {!isPro && (
        <button
          onClick={onUpgradeClick}
          className="glow-button"
          style={{ fontSize: '0.8rem', padding: '6px 16px', background: 'var(--brand-primary)', color: 'white', border: 'none', borderRadius: 'var(--radius-sm)', cursor: 'pointer', fontWeight: 700 }}
        >
          Upgrade to Pro
        </button>
      )}
    </div>
  );
};

export const ExpiryOverlay: React.FC<ExpiryBannerProps> = ({
    isPro,
    isExpired,
    onUpgradeClick
}) => {
    if (!isExpired) return null;

    return (
        <div style={{
            position: 'fixed', inset: 0, background: 'var(--bg-obsidian)',
            backdropFilter: 'blur(16px)', display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center', zIndex: 9999, textAlign: 'center', padding: 'var(--spacing-xl)'
        }}>
            <div style={{ fontSize: '4rem', marginBottom: 'var(--spacing-md)', animation: 'glowPulse 2s infinite' }}>{isPro ? '🔄' : '🔒'}</div>
            <h2 style={{ fontSize: '2.5rem', fontWeight: 800, color: 'var(--text-bright)', marginBottom: 'var(--spacing-md)', fontFamily: 'Outfit' }}>
                {isPro ? 'Your Pro Subscription Has Expired' : 'Your 10-Day Free Trial Has Ended'}
            </h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '1.1rem', maxWidth: '500px', lineHeight: 1.6, marginBottom: 'var(--spacing-xl)' }}>
                {isPro
                    ? 'Renew your Pro plan to continue accessing real-time forecasts, confidence scores, and the full 10-day prediction horizon.'
                    : 'You have used your free 10-day trial. Subscribe to Pro to continue making AI-powered predictions on NSE stocks.'}
            </p>
            <button
                onClick={onUpgradeClick}
                style={{ 
                    padding: '18px 48px', 
                    background: 'linear-gradient(90deg, var(--brand-primary), var(--brand-accent))', 
                    color: '#fff', 
                    border: 'none', 
                    borderRadius: 'var(--radius-lg)', 
                    fontWeight: 800, 
                    fontSize: '1.2rem', 
                    cursor: 'pointer', 
                    boxShadow: 'var(--shadow-lg)' 
                }}
            >
                {isPro ? 'Renew Pro — $49/mo' : 'Subscribe to Pro — $49/mo'}
            </button>
        </div>
    );
};
