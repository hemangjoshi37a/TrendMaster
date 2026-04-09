import React from 'react';

interface RiskStats {
  var_95: number;
  es_95: number;
  kelly_fraction: number;
  pop: number;
}

interface MatrixItem {
  day: number;
  date: string;
  mean: number;
  upper: number;
  lower: number;
}

interface RiskLabProps {
  stats: RiskStats;
  matrix: MatrixItem[];
  symbol: string;
}

const RiskLab: React.FC<RiskLabProps> = ({ stats, matrix, symbol }) => {
  return (
    <div className="risk-lab-container animate-fade-in" style={{ marginTop: '24px' }}>
      <div className="panel-title" style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <span style={{ fontSize: '1.2rem' }}>🛡️</span> Quantum Risk Intelligence
      </div>
      
      <div className="risk-grid" style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
        gap: '20px',
        padding: '20px'
      }}>
        {/* Kelly Optimizer */}
        <div className="pro-glow-panel" style={{ background: 'rgba(255,255,255,0.03)', padding: '20px', borderRadius: '12px', border: '1px solid var(--border)' }}>
          <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginBottom: '8px', fontWeight: 600 }}>KELLY CRITERION OPTIMIZER</div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '10px' }}>
            <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--accent)' }}>{(stats.kelly_fraction * 100).toFixed(1)}%</div>
            <div style={{ fontSize: '0.9rem', color: 'var(--success)' }}>Optimal Size</div>
          </div>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-dark)', marginTop: '12px', lineHeight: '1.4' }}>
            Suggested capital allocation to maximize logarithmic growth while minimizing drawdown risk for {symbol}.
          </p>
        </div>

        {/* Probability of Profit */}
        <div className="pro-glow-panel" style={{ background: 'rgba(255,255,255,0.03)', padding: '20px', borderRadius: '12px', border: '1px solid var(--border)' }}>
          <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginBottom: '8px', fontWeight: 600 }}>PROBABILITY OF PROFIT (PoP)</div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '10px' }}>
            <div style={{ fontSize: '2rem', fontWeight: 800, color: stats.pop > 50 ? 'var(--success)' : 'var(--warning)' }}>{stats.pop}%</div>
            <div style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>Confidence</div>
          </div>
          <div style={{ width: '100%', height: '6px', background: 'var(--border)', borderRadius: '3px', marginTop: '15px', overflow: 'hidden' }}>
            <div style={{ width: `${stats.pop}%`, height: '100%', background: stats.pop > 50 ? 'var(--success)' : 'var(--warning)', transition: 'width 1s ease-out' }}></div>
          </div>
        </div>

        {/* Tail Risk (VaR) */}
        <div className="pro-glow-panel" style={{ background: 'rgba(255,255,255,0.03)', padding: '20px', borderRadius: '12px', border: '1px solid var(--border)' }}>
          <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginBottom: '8px', fontWeight: 600 }}>95% VALUE AT RISK (VaR)</div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '10px' }}>
            <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--error)' }}>{stats.var_95}%</div>
            <div style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>Potential Leak</div>
          </div>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-dark)', marginTop: '12px' }}>
            Worst expected loss under normal market conditions with 95% certainty.
          </p>
        </div>
      </div>

      {/* Scenario Matrix Table */}
      <div className="scenario-matrix-panel" style={{ padding: '0 20px 20px' }}>
        <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginBottom: '15px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '1px' }}>
          Parallel Future Matrix
        </div>
        <div style={{ overflowX: 'auto', borderRadius: '8px', border: '1px solid var(--border)' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
            <thead>
              <tr style={{ background: 'rgba(255,255,255,0.02)', borderBottom: '1px solid var(--border)' }}>
                <th style={{ padding: '12px', textAlign: 'left', color: 'var(--text-muted)' }}>Horizon</th>
                <th style={{ padding: '12px', textAlign: 'left', color: 'var(--text-muted)' }}>Date</th>
                <th style={{ padding: '12px', textAlign: 'right', color: '#F23645' }}>Worst Case (Red)</th>
                <th style={{ padding: '12px', textAlign: 'right', color: '#00F0FF' }}>Most Likely (Cyan)</th>
                <th style={{ padding: '12px', textAlign: 'right', color: '#08BB81' }}>Best Case (Green)</th>
                <th style={{ padding: '12px', textAlign: 'right', color: 'var(--text-muted)' }}>Range %</th>
              </tr>
            </thead>
            <tbody>
              {matrix.map((row, i) => {
                const rangePct = ((row.upper - row.lower) / row.lower * 100).toFixed(2);
                return (
                  <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', transition: 'background 0.2s' }} className="table-row-hover">
                    <td style={{ padding: '12px', color: 'var(--accent)', fontWeight: 600 }}>T + {row.day} Days</td>
                    <td style={{ padding: '12px' }}>{row.date}</td>
                    <td style={{ padding: '12px', textAlign: 'right', fontWeight: 500 }}>₹{row.lower.toLocaleString()}</td>
                    <td style={{ padding: '12px', textAlign: 'right', fontWeight: 500, background: 'rgba(0, 240, 255, 0.03)' }}>₹{row.mean.toLocaleString()}</td>
                    <td style={{ padding: '12px', textAlign: 'right', fontWeight: 500 }}>₹{row.upper.toLocaleString()}</td>
                    <td style={{ padding: '12px', textAlign: 'right', color: 'var(--text-dark)' }}>{rangePct}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default RiskLab;
