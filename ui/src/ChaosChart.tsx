import React, { useEffect, useRef } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi } from 'lightweight-charts';

interface PredictionData {
  dates: string[];
  prices: number[];
  prediction_start_index: number;
}

interface ChaosChartProps {
  baseData: PredictionData | null;
  shockData: PredictionData | null;
  shockPct: number;
}

const ChaosChart: React.FC<ChaosChartProps> = ({ baseData, shockData, shockPct }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const histSeriesRef = useRef<ISeriesApi<"Area"> | null>(null);
  const basePredSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const shockPredSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const container = chartContainerRef.current;
    
    // Create chart
    const chart = createChart(container, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#787B86',
        fontSize: 12,
        fontFamily: 'Inter, -apple-system, system-ui, sans-serif',
      },
      grid: {
        vertLines: { color: 'rgba(42, 46, 57, 0.4)' },
        horzLines: { color: 'rgba(42, 46, 57, 0.4)' },
      },
      crosshair: {
        mode: 1, // Magnet
      },
      rightPriceScale: {
        borderColor: 'rgba(42, 46, 57, 0.8)',
        autoScale: true,
      },
      timeScale: {
        borderColor: 'rgba(42, 46, 57, 0.8)',
        timeVisible: true,
      },
      width: container.clientWidth || 600,
      height: container.clientHeight || 400,
    });

    const areaSeries = chart.addAreaSeries({
      lineColor: '#2962FF',
      topColor: 'rgba(41, 98, 255, 0.3)',
      bottomColor: 'rgba(41, 98, 255, 0.0)',
      lineWidth: 2,
    });

    const baseLineSeries = chart.addLineSeries({
      color: '#787B86', // Base pred gets grey
      lineWidth: 2,
      lineStyle: 1, // Dotted
      crosshairMarkerVisible: true,
    });

    const shockLineSeries = chart.addLineSeries({
      color: shockPct >= 0 ? '#089981' : '#F23645', // Green or Red based on shock
      lineWidth: 3,
      lineStyle: 0, // Solid
      crosshairMarkerVisible: true,
    });

    chartRef.current = chart;
    histSeriesRef.current = areaSeries;
    basePredSeriesRef.current = baseLineSeries;
    shockPredSeriesRef.current = shockLineSeries;

    const resizeObserver = new ResizeObserver(entries => {
      if (entries.length > 0 && chartRef.current) {
        const { width, height } = entries[0].contentRect;
        chartRef.current.applyOptions({ width, height });
      }
    });
    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
      chart.remove();
    };
  }, []);

  const updateChartData = () => {
    if (!histSeriesRef.current || !basePredSeriesRef.current || !shockPredSeriesRef.current) return;
    if (!baseData || !shockData) return;

    try {
        const { dates, prices, prediction_start_index: psi } = shockData; // History is same

        // Historical series (from shockData)
        const histData = dates
          .slice(0, psi)
          .map((date, i) => ({ time: date, value: prices[i] }));

        // Base series
        const baseAnchorPoint = { time: baseData.dates[psi - 1], value: baseData.prices[psi - 1] };
        const baseForecastPoints = baseData.dates
          .slice(psi)
          .map((date, i) => ({ time: date, value: baseData.prices[psi + i] }));
        
        // Shock series
        // Note: the shock is applied to the final historical tick, so the anchor point is actually different for the shock series.
        const shockAnchorPoint = { time: dates[psi - 1], value: prices[psi - 1] };
        const shockForecastPoints = dates
          .slice(psi)
          .map((date, i) => ({ time: date, value: prices[psi + i] }));

        histSeriesRef.current.setData(histData);
        basePredSeriesRef.current.setData([baseAnchorPoint, ...baseForecastPoints]);
        
        // Color update based on shock
        shockPredSeriesRef.current.applyOptions({
            color: shockPct >= 0 ? '#089981' : '#F23645'
        });
        shockPredSeriesRef.current.setData([baseAnchorPoint, shockAnchorPoint, ...shockForecastPoints]);
        
        setTimeout(() => {
            chartRef.current?.timeScale().fitContent();
        }, 50);
    } catch (e) {
        console.error("Error setting chart data:", e);
    }
  };

  useEffect(() => {
    updateChartData();
  }, [baseData, shockData, shockPct]);

  return (
    <div ref={chartContainerRef} style={{ width: '100%', height: '100%' }} />
  );
};

export default ChaosChart;
