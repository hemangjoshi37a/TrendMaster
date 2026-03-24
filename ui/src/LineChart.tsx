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
        vertLine: {
          color: '#787B86',
          width: 1,
          style: 3,
          labelBackgroundColor: '#2962FF',
        },
        horzLine: {
          color: '#787B86',
          width: 1,
          style: 3,
          labelBackgroundColor: '#2962FF',
        },
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

    const lineSeries = chart.addLineSeries({
      color: '#F23645',
      lineWidth: 2,
      lineStyle: 2, // Dashed
      crosshairMarkerVisible: true,
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
<<<<<<< HEAD
        const { dates, prices, prediction_start_index: psi } = predictionData;

        // Historical series: all data up to and including the last known price
        const histData = dates
          .slice(0, psi)
          .map((date, i) => ({ time: date, value: prices[i] }));

        // Prediction series: starts from the last historical point (as anchor)
        // so the two series connect seamlessly with no visual gap.
        const anchorPoint = { time: dates[psi - 1], value: prices[psi - 1] };
        const forecastPoints = dates
          .slice(psi)
          .map((date, i) => ({ time: date, value: prices[psi + i] }));
        const futureData = [anchorPoint, ...forecastPoints];
=======
        const histData = predictionData.dates
          .slice(0, predictionData.prediction_start_index)
          .map((date, i) => ({
            time: date,
            value: predictionData.prices[i],
          }));

        const futureData = predictionData.dates
          .slice(predictionData.prediction_start_index - 1)
          .map((date, i) => ({
            time: date,
            value: predictionData.prices[i + predictionData.prediction_start_index - 1],
          }));
>>>>>>> 908e367a2824764ef0b35736e589e8fbcc6ffd45

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
    <div ref={chartContainerRef} style={{ width: '100%', height: '100%' }} />
  );
};

export default LineChart;
