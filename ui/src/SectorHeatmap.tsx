import React, { useState, useEffect } from 'react';
import './SectorHeatmap.css';

interface Sector {
  name: string;
  ticker: string;
  change: number;
  weight: number;
}

interface SectorHeatmapProps {
  isPro: boolean;
}

const SectorHeatmap: React.FC<SectorHeatmapProps> = ({ isPro }) => {
  const [sectors, setSectors] = useState<Sector[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchSectors = async () => {
      if (!isPro) return; // Only fetch if Pro
      try {
        const resp = await fetch('/api/sectors');
        if (resp.ok) {
          const data = await resp.json();
          setSectors(data);
        }
      } catch (e) {
        console.error("Failed to fetch sector data", e);
      } finally {
        setLoading(false);
      }
    };
    
    fetchSectors();
    const interval = setInterval(fetchSectors, 60000); // 1 min refresh
    return () => clearInterval(interval);
  }, [isPro]);

  if (!isPro) {
    return (
      <div className="widget">
        <div className="widget-title">
          Sector Heatmap
          <span className="pro-badge">PRO</span>
        </div>
        <div className="locked-feature" style={{ height: '180px' }}>
          <div className="locked-blur" style={{ backgroundImage: `url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><rect x="0" y="0" width="50" height="50" fill="%232A2E39"/><rect x="52" y="0" width="48" height="50" fill="%231E222D"/><rect x="0" y="52" width="100" height="48" fill="%232A2E39"/></svg>')` }}></div>
          <div className="locked-overlay">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
            </svg>
            <span>Pro Access Required</span>
          </div>
        </div>
      </div>
    );
  }

  // Simplified Treemap Layout logic for a 1-row or 2-row grid.
  // Real algorithms algorithms (like squarified treemap) are complex to write from scratch without D3.
  // Here we use a CSS Grid approach with proportional flex-basis.
  
  const totalWeight = sectors.reduce((sum, s) => sum + s.weight, 0);

  const getColor = (change: number) => {
    if (change > 2) return '#089981'; // Strong Buy
    if (change > 0) return 'rgba(8, 153, 129, 0.6)'; // Buy
    if (change < -2) return '#F23645'; // Strong Sell
    if (change < 0) return 'rgba(242, 54, 69, 0.6)'; // Sell
    return '#2A2E39'; // Neutral
  };

  return (
    <div className="widget">
      <div className="widget-title">
        Sector Heatmap
        <span className="live-badge">LIVE</span>
      </div>
      
      {loading ? (
        <div className="hm-loader">
          <div className="bt-spinner" style={{ width: '20px', height: '20px', borderWidth: '2px' }}></div>
        </div>
      ) : sectors.length > 0 ? (
        <div className="treemap-container">
           {sectors.map((sector) => {
              const heightPercentage = Math.max((sector.weight / totalWeight) * 100 * 2, 10); // scale up for visibility
              return (
                  <div 
                    key={sector.ticker} 
                    className="treemap-cell"
                    style={{ 
                      flexGrow: sector.weight,
                      flexBasis: `${heightPercentage}%`,
                      backgroundColor: getColor(sector.change)
                    }}
                    title={`${sector.name}: ${sector.change}%`}
                  >
                    <div className="tm-name">{sector.name.replace('Nifty ', '')}</div>
                    <div className="tm-change">{sector.change > 0 ? '+' : ''}{sector.change}%</div>
                  </div>
              )
           })}
        </div>
      ) : (
        <div className="hm-empty">No data</div>
      )}
    </div>
  );
};

export default SectorHeatmap;
