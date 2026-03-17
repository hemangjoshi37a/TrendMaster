import React, { useEffect, useRef } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi } from 'lightweight-charts';

interface PredictionData {
  dates: string[];
  prices: number[];
  prediction_start_index: number;
}

interface LineChartProps {
  data: PredictionData;
}

const LineChart: React.FC<LineChartProps> = ({ data }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const histSeriesRef = useRef<ISeriesApi<"Area"> | null>(null);
  const predSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    console.log("Initializing chart...");
    const container = chartContainerRef.current;
    
    // Create chart
    const chart = createChart(container, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#787b86',
        fontSize: 12,
        fontFamily: 'Inter, sans-serif',
      },
      grid: {
        vertLines: { color: 'rgba(42, 46, 57, 0.1)' },
        horzLines: { color: 'rgba(42, 46, 57, 0.1)' },
      },
      rightPriceScale: {
        borderColor: 'rgba(197, 203, 206, 0.1)',
        autoScale: true,
      },
      timeScale: {
        borderColor: 'rgba(197, 203, 206, 0.1)',
        timeVisible: true,
      },
      width: container.clientWidth || 600,
      height: container.clientHeight || 400,
    });

    const areaSeries = chart.addAreaSeries({
      lineColor: '#2962ff',
      topColor: 'rgba(41, 98, 255, 0.4)',
      bottomColor: 'rgba(41, 98, 255, 0.0)',
      lineWidth: 3,
    });

    const lineSeries = chart.addLineSeries({
      color: '#f23645',
      lineWidth: 3,
      lineStyle: 2,
    });

    chartRef.current = chart;
    histSeriesRef.current = areaSeries;
    predSeriesRef.current = lineSeries;

    const resizeObserver = new ResizeObserver(entries => {
      if (entries.length > 0 && chartRef.current) {
        const { width, height } = entries[0].contentRect;
        chartRef.current.applyOptions({ width, height });
      }
    });
    resizeObserver.observe(container);

    // Initial data load if available
    if (data && data.dates && data.dates.length > 0) {
        updateChartData(data);
    }

    return () => {
      resizeObserver.disconnect();
      chart.remove();
    };
  }, []);

  const updateChartData = (predictionData: PredictionData) => {
    if (!histSeriesRef.current || !predSeriesRef.current || !predictionData) return;

    try {
        const histData = predictionData.dates
          .slice(0, predictionData.prediction_start_index)
          .map((date, i) => ({
            time: new Date(date).toISOString().split('T')[0],
            value: predictionData.prices[i],
          }));

        const futureData = predictionData.dates
          .slice(predictionData.prediction_start_index - 1)
          .map((date, i) => ({
            time: new Date(date).toISOString().split('T')[0],
            value: predictionData.prices[i + predictionData.prediction_start_index - 1],
          }));

        histSeriesRef.current.setData(histData);
        predSeriesRef.current.setData(futureData);
        
        setTimeout(() => {
            chartRef.current?.timeScale().fitContent();
        }, 50);
    } catch (e) {
        console.error("Error setting chart data:", e);
    }
  };

  useEffect(() => {
    if (data) {
      updateChartData(data);
    }
  }, [data]);

  return (
    <div ref={chartContainerRef} style={{ width: '100%', height: '100%', minHeight: '400px' }} />
  );
};

export default LineChart;
