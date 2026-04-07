import React, { useState } from 'react';
import { createChart, ColorType } from 'lightweight-charts';

interface ScanResult {
  date: string;
  expected: number[];
  upper: number[];
  lower: number[];
  actual: number[];
}

interface DeepScanProps {
  results: ScanResult[];
  symbol: string;
}

const DeepScan: React.FC<DeepScanProps> = ({ results, symbol }) => {
  const [selectedIdx, setSelectedIdx] = useState(0);
  const current = results[selectedIdx];
  
  // Calculate accuracy for the selected snapshot
  const finalActual = current.actual[current.actual.length - 1];
  const finalMean = current.expected[current.expected.length - 1];
  const finalUpper = current.upper[current.upper.length - 1];
  const finalLower = current.lower[current.lower.length - 1];
  
  const wasAccurate = finalActual <= finalUpper && finalActual >= finalLower;
  const errorPct = Math.abs((finalActual - finalMean) / finalActual * 100).toFixed(2);

  return (
    <div className="deep-scan-module animate-fade-in" style={{ 
      background: 'var(--bg-elevated)', 
      borderRadius: '16px', 
      border: '1px solid var(--accent)',
      padding: '24px',
      marginTop: '30px',
      boxShadow: '0 0 30px rgba(0, 240, 255, 0.05)'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <div>
          <h3 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ color: 'var(--accent)' }}>⚛️</span> Historical Multiverse Scan
          </h3>
          <p style={{ color: 'var(--text-dark)', fontSize: '0.8rem', marginTop: '4px' }}>
            Retrospective stochastic review of {symbol} across 20 historical checkpoints.
          </p>
        </div>
        
        <div style={{ display: 'flex', gap: '15px' }}>
          <div className="pro-stat-mini">
            <span style={{ color: 'var(--text-dark)', fontSize: '0.7rem' }}>CONVICTION SCORE</span>
            <div style={{ fontSize: '1rem', fontWeight: 800 }}>{wasAccurate ? 'HIGH' : 'LOW'}</div>
          </div>
          <div className="pro-stat-mini">
            <span style={{ color: 'var(--text-dark)', fontSize: '0.7rem' }}>HISTORICAL ERROR</span>
            <div style={{ fontSize: '1rem', fontWeight: 800, color: 'var(--accent)' }}>{errorPct}%</div>
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', gap: '30px' }}>
        {/* Checkpoint Navigator */}
        <div style={{ maxHeight: '400px', overflowY: 'auto', paddingRight: '10px' }} className="custom-scrollbar">
          {results.map((res, idx) => (
            <button 
              key={idx}
              onClick={() => setSelectedIdx(idx)}
              style={{
                width: '100%',
                padding: '12px 16px',
                marginBottom: '8px',
                background: idx === selectedIdx ? 'rgba(0, 240, 255, 0.1)' : 'transparent',
                border: `1px solid ${idx === selectedIdx ? 'var(--accent)' : 'var(--border)'}`,
                borderRadius: '8px',
                textAlign: 'left',
                cursor: 'pointer',
                transition: 'all 0.2s',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
              }}
            >
              <span style={{ color: idx === selectedIdx ? 'var(--text-main)' : 'var(--text-muted)', fontSize: '0.85rem' }}>
                {res.date}
              </span>
              {idx === selectedIdx && <span style={{ color: 'var(--accent)' }}>●</span>}
            </button>
          ))}
        </div>

        {/* Snapshot Detail View */}
        <div style={{ background: 'rgba(0,0,0,0.2)', borderRadius: '12px', padding: '20px', border: '1px solid var(--border)' }}>
          <div style={{ marginBottom: '20px', display: 'flex', gap: '20px' }}>
            <div style={{ flex: 1 }}>
              <div style={{ color: 'var(--text-dark)', fontSize: '0.75rem', marginBottom: '8px' }}>WAS WITHIN PROBABILITY CLOUD?</div>
              <div style={{ 
                padding: '10px', 
                borderRadius: '6px', 
                textAlign: 'center', 
                fontWeight: 800,
                background: wasAccurate ? 'rgba(8, 187, 129, 0.1)' : 'rgba(242, 54, 69, 0.1)',
                color: wasAccurate ? 'var(--success)' : 'var(--error)',
                border: `1px solid ${wasAccurate ? 'var(--success)' : 'var(--error)'}`
              }}>
                {wasAccurate ? 'CAPTURED WITHIN 95% CI' : 'OUTLIER EVENT DETECTED'}
              </div>
            </div>
            <div style={{ flex: 1 }}>
               <div style={{ color: 'var(--text-dark)', fontSize: '0.75rem', marginBottom: '8px' }}>REALITY VS MEAN</div>
               <div style={{ padding: '10px', textAlign: 'center', fontSize: '1.2rem', fontWeight: 800 }}>
                  ₹{finalActual.toFixed(0)} <span style={{ color: 'var(--text-dark)', fontSize: '0.8rem' }}>/</span> ₹{finalMean.toFixed(0)}
               </div>
            </div>
          </div>

          <div style={{ padding: '15px', background: 'rgba(255,255,255,0.02)', borderRadius: '10px', border: '1px dotted var(--border)' }}>
             <p style={{ margin: 0, fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: '1.5' }}>
                <strong style={{ color: 'var(--text-main)' }}>Retrospective Analysis:</strong> This simulation was anchored on {current.date}. 
                {wasAccurate 
                  ? " The model correctly anticipated the risk volatility, with the actual price landing comfortably within the predicted Multiverse boundaries."
                  : " The actual price action exceeded the model's 95% confidence boundaries, indicating an extremely rare or unexpected market shock occurred during this window."
                }
             </p>
          </div>
          
          <div style={{ marginTop: '20px', fontSize: '0.7rem', color: 'var(--text-dark)', textAlign: 'center', textTransform: 'uppercase', letterSpacing: '2px' }}>
             Deep Scan Matrix Component v0.4.1
          </div>
        </div>
      </div>
    </div>
  );
};

export default DeepScan;
