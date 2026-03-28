import React, { useEffect, useRef } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi } from 'lightweight-charts';

interface BacktestBurst {
  start_index: number;
  dates: string[];
  prices: number[];
}

interface BacktestData {
  actual: {
    dates: string[];
    prices: number[];
  };
  bursts: BacktestBurst[];
}

interface BacktestChartProps {
  data: BacktestData | null;
}

const BacktestChart: React.FC<BacktestChartProps> = ({ data }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const actualSeriesRef = useRef<ISeriesApi<"Area"> | null>(null);
  const burstSeriesRefs = useRef<ISeriesApi<"Line">[]>([]);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const container = chartContainerRef.current;
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
      rightPriceScale: {
        borderColor: 'rgba(42, 46, 57, 0.8)',
        autoScale: true,
      },
      timeScale: {
        borderColor: 'rgba(42, 46, 57, 0.8)',
        timeVisible: true,
      },
      width: container.clientWidth || 800,
      height: container.clientHeight || 500,
    });

    const areaSeries = chart.addAreaSeries({
      lineColor: '#2962FF',
      topColor: 'rgba(41, 98, 255, 0.2)',
      bottomColor: 'rgba(41, 98, 255, 0.0)',
      lineWidth: 2,
    });

    chartRef.current = chart;
    actualSeriesRef.current = areaSeries;

    const handleResize = () => {
      if (chartRef.current && chartContainerRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
          height: chartContainerRef.current.clientHeight,
        });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, []);

  useEffect(() => {
    if (!chartRef.current || !actualSeriesRef.current || !data) return;

    const { actual, bursts } = data;
    const chart = chartRef.current;
    if (!chart || !actualSeriesRef.current) return;

    // Clear previous bursts safely
    burstSeriesRefs.current.forEach(s => {
      try {
        chart.removeSeries(s);
      } catch (e) {
        console.warn("Could not remove series:", e);
      }
    });
    burstSeriesRefs.current = [];

    // Set actual data
    const actualPoints = actual.dates.map((d, i) => ({
      time: d,
      value: actual.prices[i]
    }));
    actualSeriesRef.current.setData(actualPoints);

    // Add each burst as a separate line
    bursts.forEach((burst) => {
      const burstLine = chartRef.current!.addLineSeries({
        color: 'rgba(242, 54, 69, 0.6)',
        lineWidth: 1,
        lineStyle: 0,
        lastValueVisible: false,
        priceLineVisible: false,
      });

      // Anchor it to the actual price at the start index
      const anchor = {
        time: actual.dates[burst.start_index],
        value: actual.prices[burst.start_index]
      };

      const burstPoints = [
        anchor,
        ...burst.dates.map((d, i) => ({
          time: d,
          value: burst.prices[i]
        }))
      ];

      burstLine.setData(burstPoints);
      burstSeriesRefs.current.push(burstLine);
    });

    setTimeout(() => {
      chartRef.current?.timeScale().fitContent();
    }, 100);

  }, [data]);

  return <div ref={chartContainerRef} style={{ width: '100%', height: '100%' }} />;
};

export default BacktestChart;
