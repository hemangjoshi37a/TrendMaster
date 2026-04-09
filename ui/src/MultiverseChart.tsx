import React, { useEffect, useRef } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi } from 'lightweight-charts';

interface MultiverseData {
  dates: string[];
  prices: number[];
  cloud_upper?: number[];
  cloud_lower?: number[];
  prediction_start_index: number;
  distribution?: {
     bins: number[];
     counts: number[];
     chaos_score: number;
  };
}

interface MultiverseChartProps {
  data: MultiverseData;
}

const MultiverseChart: React.FC<MultiverseChartProps> = ({ data }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  
  const histSeriesRef = useRef<ISeriesApi<"Area"> | null>(null);
  const predSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const upperCloudRef = useRef<ISeriesApi<"Area"> | null>(null);
  const lowerCloudRef = useRef<ISeriesApi<"Area"> | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const container = chartContainerRef.current;
    
    const chart = createChart(container, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#d1d4dc',
        fontSize: 12,
        fontFamily: "'JetBrains Mono', monospace",
      },
      grid: {
        vertLines: { color: 'rgba(42, 46, 57, 0.2)' },
        horzLines: { color: 'rgba(42, 46, 57, 0.2)' },
      },
      crosshair: {
        mode: 1,
        vertLine: { color: '#00d2ff', width: 1, style: 3, labelBackgroundColor: '#00d2ff' },
        horzLine: { color: '#00d2ff', width: 1, style: 3, labelBackgroundColor: '#00d2ff' },
      },
      rightPriceScale: { 
        borderColor: 'rgba(42, 46, 57, 0.3)', 
        autoScale: true,
        scaleMargins: {
          top: 0.2,
          bottom: 0.2,
        },
      },
      timeScale: { 
        borderColor: 'rgba(42, 46, 57, 0.3)', 
        timeVisible: true,
        fixLeftEdge: true,
        fixRightEdge: true,
      },
      width: container.clientWidth,
      height: container.clientHeight || 680,
    });

    // Stochastic Cloud - Upper Part
    const upperCloud = chart.addAreaSeries({
      topColor: 'rgba(41, 98, 255, 0.25)',
      bottomColor: 'rgba(41, 98, 255, 0.25)',
      lineVisible: false,
      priceLineVisible: false,
      lastValueVisible: false,
    });

    // 2. Stochastic Cloud - Lower Part (The "Eraser" layer synchronized with Bg-Obsidian)
    const lowerCloud = chart.addAreaSeries({
      topColor: '#0B0E14', 
      bottomColor: '#0B0E14',
      lineVisible: false,
      priceLineVisible: false,
      lastValueVisible: false,
    });

    // 3. Historical Data
    const histSeries = chart.addAreaSeries({
      lineColor: '#2962FF',
      topColor: 'rgba(41, 98, 255, 0.15)',
      bottomColor: 'rgba(41, 98, 255, 0.0)',
      lineWidth: 2,
      lastValueVisible: false,
    });

    // 4. Mean Prediction Line (Most Likely)
    const predSeries = chart.addLineSeries({
      color: '#00F0FF', 
      lineWidth: 4,
      lineStyle: 0,
      crosshairMarkerVisible: true,
      lastValueVisible: true,
      title: 'MOST LIKELY',
    });

    // 5. Best Case Line (Upper Bound)
    const upperLine = chart.addLineSeries({
      color: '#08BB81',
      lineWidth: 3,
      lineStyle: 0, 
      lastValueVisible: true,
      title: 'BEST CASE',
    });

    // 6. Worst Case Line (Lower Bound)
    const lowerLine = chart.addLineSeries({
      color: '#F23645',
      lineWidth: 3,
      lineStyle: 0, 
      lastValueVisible: true,
      title: 'WORST CASE',
    });

    chartRef.current = chart;
    histSeriesRef.current = histSeries;
    predSeriesRef.current = predSeries;
    upperCloudRef.current = upperCloud;
    lowerCloudRef.current = lowerCloud;
    (chartRef.current as any).upperLine = upperLine;
    (chartRef.current as any).lowerLine = lowerLine;

    const resizeObserver = new ResizeObserver(entries => {
      if (entries.length > 0 && chartRef.current) {
        const { width, height } = entries[0].contentRect;
        chartRef.current.applyOptions({ width, height });
      }
    });
    resizeObserver.observe(container);

    if (data) updateChartData(data);

    return () => {
      resizeObserver.disconnect();
      chart.remove();
    };
  }, []);

  const updateChartData = (mdata: MultiverseData) => {
    if (!chartRef.current || !mdata) return;

    try {
        const { dates, prices, cloud_upper, cloud_lower, prediction_start_index: psi } = mdata;

        // Hist Data
        const histPoints = dates.slice(0, psi).map((date, i) => ({ time: date, value: prices[i] }));
        histSeriesRef.current?.setData(histPoints);

        // Pred Data (Mean)
        const meanPoints = dates.slice(psi - 1).map((date, i) => ({
            time: date,
            value: prices[psi - 1 + i]
        }));
        predSeriesRef.current?.setData(meanPoints);

        // Cloud & Lines Data
        if (cloud_upper && cloud_lower) {
            const anchorPoint = { time: dates[psi - 1], value: prices[psi - 1] };
            const futureDates = dates.slice(psi);
            
            const upperPoints = [
                anchorPoint,
                ...futureDates.map((date, i) => ({ time: date, value: cloud_upper[i] }))
            ];
            const lowerPoints = [
                anchorPoint,
                ...futureDates.map((date, i) => ({ time: date, value: cloud_lower[i] }))
            ];
            
            upperCloudRef.current?.setData(futureDates.map((date, i) => ({ time: date, value: cloud_upper[i] })));
            lowerCloudRef.current?.setData(futureDates.map((date, i) => ({ time: date, value: cloud_lower[i] })));

            (chartRef.current as any).upperLine?.setData(upperPoints);
            (chartRef.current as any).lowerLine?.setData(lowerPoints);
        }

        setTimeout(() => {
            chartRef.current?.timeScale().fitContent();
        }, 50);
    } catch (e) {
        console.error("Multiverse chart update error", e);
    }
  };

  useEffect(() => {
    if (data) updateChartData(data);
  }, [data]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', minHeight: '680px', display: 'flex', borderRadius: 'var(--radius-lg)', overflow: 'hidden' }}>
      <div ref={chartContainerRef} style={{ flex: 1, position: 'relative' }} />
      
      {/* Probability Density Heatmap Sidebar (Refined) */}
      {data.distribution && (
         <div className="prob-heatmap" style={{ 
            width: '100px', 
            background: 'rgba(15, 18, 26, 0.4)', 
            borderLeft: '1px solid var(--glass-border)',
            display: 'flex',
            flexDirection: 'column-reverse',
            padding: '40px 0 30px',
            position: 'relative'
         }}>
            {data.distribution.counts.map((count, i) => {
               const maxCount = Math.max(...data.distribution!.counts);
               const intensity = (count / maxCount);
               return (
                  <div key={i} style={{ 
                     flex: 1, 
                     width: `${intensity * 100}%`, 
                     background: intensity > 0.5 
                        ? `linear-gradient(90deg, transparent 0%, var(--brand-primary) 100%)` 
                        : `rgba(0, 240, 255, ${0.05 + intensity * 0.4})`,
                     margin: '1px 0',
                     borderRadius: '0 4px 4px 0',
                     minHeight: '2px',
                     boxShadow: intensity > 0.9 ? '0 0 15px var(--brand-primary-glow)' : 'none',
                     transition: 'all 0.5s ease'
                  }} title={`Density: ${count} realities`} />
               );
            })}
            <div style={{ 
               position: 'absolute', 
               top: '15px', 
               left: '50%',
               transform: 'translateX(-50%)',
               fontSize: '0.65rem', 
               color: 'var(--text-dim)', 
               textTransform: 'uppercase',
               fontWeight: 800,
               letterSpacing: '1px',
               width: '100%',
               textAlign: 'center'
            }}>
               Density Area
            </div>
         </div>
      )}
    </div>
  );
};

export default MultiverseChart;
