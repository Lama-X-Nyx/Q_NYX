'use client';

import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ReferenceLine, ReferenceArea, Cell,
} from 'recharts';

const COLORS = {
  bg: '#0a0a0b',
  grid: '#1f2937',
  text: '#9ca3af',
  green: '#22c55e',
  red: '#ef4444',
  yellow: '#eab308',
  blue: '#3b82f6',
  purple: '#a855f7',
  cyan: '#06b6d4',
};

const ASSET_COLORS: Record<string, string> = {
  BTCUSDT: '#f59e0b',
  ETHUSDT: '#3b82f6',
  SOLUSDT: '#a855f7',
};

const tooltipStyle = {
  backgroundColor: '#111827',
  border: '1px solid #1f2937',
  borderRadius: '4px',
  fontSize: '11px',
};

// =========================================================================
// 1. Health + Multiplier time-series line chart with state bands
// =========================================================================

export function HealthTimeSeries({ data, height = 140 }: {
  data: Array<{ ts: string | number; health: number; smoothed: number; multiplier: number }>;
  height?: number;
}) {
  if (!data || data.length === 0) {
    return <EmptyChart height={height} label="No health data yet" />;
  }
  const series = data.map((d, i) => ({ idx: i, ...d }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={series} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
        <CartesianGrid strokeDasharray="2 2" stroke={COLORS.grid} />
        <XAxis dataKey="idx" tick={{ fontSize: 9, fill: COLORS.text }} axisLine={false} />
        <YAxis domain={[0, 1]} tick={{ fontSize: 9, fill: COLORS.text }} axisLine={false} />
        <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: '#f3f4f6' }} />
        <ReferenceArea y1={0} y2={0.35} fill={COLORS.red} fillOpacity={0.05} />
        <ReferenceArea y1={0.35} y2={0.7} fill={COLORS.yellow} fillOpacity={0.05} />
        <ReferenceArea y1={0.7} y2={1} fill={COLORS.green} fillOpacity={0.05} />
        <ReferenceLine y={0.35} stroke={COLORS.red} strokeDasharray="3 3" strokeOpacity={0.5} />
        <ReferenceLine y={0.7} stroke={COLORS.green} strokeDasharray="3 3" strokeOpacity={0.5} />
        <Line type="monotone" dataKey="health" stroke={COLORS.cyan} strokeWidth={1} dot={false} isAnimationActive={false} name="Raw" />
        <Line type="monotone" dataKey="smoothed" stroke={COLORS.blue} strokeWidth={2} dot={false} isAnimationActive={false} name="Smoothed" />
        <Line type="monotone" dataKey="multiplier" stroke={COLORS.green} strokeWidth={1.5} dot={false} isAnimationActive={false} name="Multiplier" strokeDasharray="4 2" />
      </LineChart>
    </ResponsiveContainer>
  );
}

// =========================================================================
// 2. Equity + Drawdown area chart
// =========================================================================

export function EquityCurve({ data, height = 160 }: {
  data: Array<{ ts: string | number; equity: number; balance?: number; exposure?: number }>;
  height?: number;
}) {
  if (!data || data.length === 0) {
    return <EmptyChart height={height} label="No equity data yet" />;
  }
  const series = data.map((d, i) => ({ idx: i, ...d }));
  const min = Math.min(...series.map(s => s.equity));
  const max = Math.max(...series.map(s => s.equity));
  const pad = (max - min) * 0.05 || 100;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={series} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={COLORS.blue} stopOpacity={0.4} />
            <stop offset="100%" stopColor={COLORS.blue} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="2 2" stroke={COLORS.grid} />
        <XAxis dataKey="idx" tick={{ fontSize: 9, fill: COLORS.text }} axisLine={false} />
        <YAxis domain={[min - pad, max + pad]} tick={{ fontSize: 9, fill: COLORS.text }} axisLine={false} tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} />
        <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: '#f3f4f6' }} formatter={(v) => `$${Number(v).toLocaleString()}`} />
        <Area type="monotone" dataKey="equity" stroke={COLORS.blue} strokeWidth={2} fill="url(#eqGrad)" isAnimationActive={false} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

// =========================================================================
// 3. Drawdown curve (computed from equity)
// =========================================================================

export function DrawdownCurve({ data, height = 100 }: {
  data: Array<{ ts: string | number; equity: number }>;
  height?: number;
}) {
  if (!data || data.length === 0) {
    return <EmptyChart height={height} label="No drawdown data yet" />;
  }
  let peak = data[0].equity;
  const series = data.map((d, i) => {
    peak = Math.max(peak, d.equity);
    const dd = peak > 0 ? -((peak - d.equity) / peak) * 100 : 0;
    return { idx: i, drawdown: dd };
  });
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={series} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="2 2" stroke={COLORS.grid} />
        <XAxis dataKey="idx" tick={{ fontSize: 9, fill: COLORS.text }} axisLine={false} />
        <YAxis tick={{ fontSize: 9, fill: COLORS.text }} axisLine={false} tickFormatter={(v) => `${v.toFixed(1)}%`} />
        <Tooltip contentStyle={tooltipStyle} formatter={(v) => `${Number(v).toFixed(2)}%`} />
        <Area type="monotone" dataKey="drawdown" stroke={COLORS.red} strokeWidth={1.5} fill={COLORS.red} fillOpacity={0.2} isAnimationActive={false} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

// =========================================================================
// 4. Fill / Miss / Timeout grouped bar chart
// =========================================================================

export function FillMissBars({ data, height = 140 }: {
  data: Array<{ asset: string; placed: number; filled: number; timed_out: number }>;
  height?: number;
}) {
  if (!data || data.length === 0) {
    return <EmptyChart height={height} label="No execution data" />;
  }
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
        <CartesianGrid strokeDasharray="2 2" stroke={COLORS.grid} />
        <XAxis dataKey="asset" tick={{ fontSize: 10, fill: COLORS.text }} axisLine={false} tickFormatter={(v) => v.replace('USDT', '')} />
        <YAxis tick={{ fontSize: 9, fill: COLORS.text }} axisLine={false} />
        <Tooltip contentStyle={tooltipStyle} />
        <Legend wrapperStyle={{ fontSize: '10px' }} />
        <Bar dataKey="placed" fill={COLORS.blue} name="Placed" />
        <Bar dataKey="filled" fill={COLORS.green} name="Filled" />
        <Bar dataKey="timed_out" fill={COLORS.red} name="Timed out" />
      </BarChart>
    </ResponsiveContainer>
  );
}

// =========================================================================
// 5. Gauge / score indicator
// =========================================================================

export function ScoreGauge({ value, label, max = 1, thresholds = [0.35, 0.7] }: {
  value: number; label: string; max?: number; thresholds?: [number, number];
}) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100));
  const color = value >= thresholds[1] ? COLORS.green : value >= thresholds[0] ? COLORS.yellow : COLORS.red;
  const circumference = 2 * Math.PI * 36;
  const offset = circumference * (1 - pct / 100);
  return (
    <div className="flex flex-col items-center">
      <svg width="80" height="80" viewBox="0 0 80 80" className="-rotate-90">
        <circle cx="40" cy="40" r="36" fill="none" stroke={COLORS.grid} strokeWidth="6" />
        <circle cx="40" cy="40" r="36" fill="none" stroke={color} strokeWidth="6" strokeDasharray={circumference} strokeDashoffset={offset} strokeLinecap="round" className="transition-all duration-500" />
      </svg>
      <div className="-mt-12 text-center">
        <div className="text-lg font-bold" style={{ color }}>{value.toFixed(2)}</div>
      </div>
      <div className="text-xs text-gray-500 mt-3">{label}</div>
    </div>
  );
}

// =========================================================================
// 6. Heatmap (e.g. stability segments, execution stress)
// =========================================================================

export function Heatmap({ cells, xKey, yKey, valueKey, xLabels, yLabels, height = 180, title, colorScale = 'green-red' }: {
  cells: Array<Record<string, unknown>>;
  xKey: string;
  yKey: string;
  valueKey: string;
  xLabels?: string[];
  yLabels?: string[];
  height?: number;
  title?: string;
  colorScale?: 'green-red' | 'red-green' | 'blue';
}) {
  if (!cells || cells.length === 0) {
    return <EmptyChart height={height} label={`No ${title || 'heatmap'} data`} />;
  }
  const xs = xLabels || Array.from(new Set(cells.map(c => String(c[xKey]))));
  const ys = yLabels || Array.from(new Set(cells.map(c => String(c[yKey]))));
  const values = cells.map(c => Number(c[valueKey]));
  const vmin = Math.min(...values);
  const vmax = Math.max(...values);
  const range = vmax - vmin || 1;

  const getColor = (v: number) => {
    const t = (v - vmin) / range;
    if (colorScale === 'blue') {
      const a = 0.2 + t * 0.7;
      return `rgba(59, 130, 246, ${a})`;
    }
    if (colorScale === 'red-green') {
      const r = Math.round(239 * (1 - t) + 34 * t);
      const g = Math.round(68 * (1 - t) + 197 * t);
      const b = Math.round(68 * (1 - t) + 94 * t);
      return `rgba(${r}, ${g}, ${b}, 0.75)`;
    }
    const r = Math.round(34 * (1 - t) + 239 * t);
    const g = Math.round(197 * (1 - t) + 68 * t);
    const b = Math.round(94 * (1 - t) + 68 * t);
    return `rgba(${r}, ${g}, ${b}, 0.75)`;
  };

  const lookup = new Map<string, number>();
  cells.forEach(c => {
    lookup.set(`${c[xKey]}__${c[yKey]}`, Number(c[valueKey]));
  });

  return (
    <div className="overflow-auto" style={{ maxHeight: height }}>
      <table className="text-xs">
        <thead>
          <tr>
            <th className="p-1 text-left text-gray-500"></th>
            {xs.map(x => (
              <th key={x} className="p-1 text-gray-400 font-normal text-xs">{x.replace('USDT', '')}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {ys.map(y => (
            <tr key={y}>
              <td className="p-1 text-gray-400 text-right pr-2">{y}</td>
              {xs.map(x => {
                const v = lookup.get(`${x}__${y}`);
                const bg = v !== undefined ? getColor(v) : 'transparent';
                return (
                  <td
                    key={x}
                    className="p-1 text-center border border-gray-800"
                    style={{ backgroundColor: bg, minWidth: '50px' }}
                    title={`${x} × ${y} = ${v?.toFixed(3) ?? '-'}`}
                  >
                    <span className="text-gray-100 font-mono text-xs">
                      {v !== undefined ? v.toFixed(2) : '-'}
                    </span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// =========================================================================
// 7. Capital ladder (Sharpe/retention vs capital)
// =========================================================================

export function CapacityLadder({ data, metric = 'sharpe', height = 180 }: {
  data: Array<{ asset: string; capital: number; sharpe: number; retention: number; pnl: number }>;
  metric?: 'sharpe' | 'retention' | 'pnl';
  height?: number;
}) {
  if (!data || data.length === 0) {
    return <EmptyChart height={height} label="No capacity data" />;
  }
  const assets = Array.from(new Set(data.map(d => d.asset)));
  const caps = Array.from(new Set(data.map(d => d.capital))).sort((a, b) => a - b);
  const series = caps.map(cap => {
    const row: Record<string, number | string> = { capital: cap };
    assets.forEach(a => {
      const d = data.find(x => x.asset === a && x.capital === cap);
      row[a] = d ? d[metric] : 0;
    });
    return row;
  });
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={series} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
        <CartesianGrid strokeDasharray="2 2" stroke={COLORS.grid} />
        <XAxis dataKey="capital" scale="log" domain={['dataMin', 'dataMax']} tick={{ fontSize: 9, fill: COLORS.text }} axisLine={false} tickFormatter={(v) => { const n = Number(v); return n >= 1e6 ? `$${(n/1e6).toFixed(0)}M` : n >= 1e3 ? `$${(n/1e3).toFixed(0)}k` : `$${n}`; }} />
        <YAxis tick={{ fontSize: 9, fill: COLORS.text }} axisLine={false} />
        <Tooltip contentStyle={tooltipStyle} />
        <Legend wrapperStyle={{ fontSize: '10px' }} />
        {assets.map(a => (
          <Line key={a} type="monotone" dataKey={a} stroke={ASSET_COLORS[a] || COLORS.blue} strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} name={a.replace('USDT', '')} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

// =========================================================================
// 8. Stability score bar chart by asset
// =========================================================================

export function StabilityScoreBars({ data, height = 140 }: {
  data: Array<{ asset: string; score_v1: number; score_v2: number; classification: string }>;
  height?: number;
}) {
  if (!data || data.length === 0) {
    return <EmptyChart height={height} label="No stability data" />;
  }
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 5, right: 10, left: -20, bottom: 0 }} layout="vertical">
        <CartesianGrid strokeDasharray="2 2" stroke={COLORS.grid} horizontal={false} />
        <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 9, fill: COLORS.text }} axisLine={false} />
        <YAxis type="category" dataKey="asset" tick={{ fontSize: 10, fill: COLORS.text }} axisLine={false} tickFormatter={(v) => v.replace('USDT', '')} width={60} />
        <Tooltip contentStyle={tooltipStyle} />
        <Legend wrapperStyle={{ fontSize: '10px' }} />
        <Bar dataKey="score_v1" fill={COLORS.cyan} name="v1" />
        <Bar dataKey="score_v2" fill={COLORS.blue} name="v2" />
      </BarChart>
    </ResponsiveContainer>
  );
}

// =========================================================================
// 9. Sensitivity bars (offset / fill / fee)
// =========================================================================

export function SensitivityBars({ data, height = 140 }: {
  data: Array<{ asset: string; offset: number; fill: number; fee: number }>;
  height?: number;
}) {
  if (!data || data.length === 0) {
    return <EmptyChart height={height} label="No sensitivity data" />;
  }
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
        <CartesianGrid strokeDasharray="2 2" stroke={COLORS.grid} />
        <XAxis dataKey="asset" tick={{ fontSize: 10, fill: COLORS.text }} axisLine={false} tickFormatter={(v) => v.replace('USDT', '')} />
        <YAxis domain={[0, 1]} tick={{ fontSize: 9, fill: COLORS.text }} axisLine={false} />
        <Tooltip contentStyle={tooltipStyle} />
        <Legend wrapperStyle={{ fontSize: '10px' }} />
        <Bar dataKey="offset" fill={COLORS.cyan} name="Offset" />
        <Bar dataKey="fill" fill={COLORS.red} name="Fill" />
        <Bar dataKey="fee" fill={COLORS.yellow} name="Fee" />
      </BarChart>
    </ResponsiveContainer>
  );
}

// =========================================================================
// 10. Decision waterfall (pipeline degradation visual)
// =========================================================================

export function DecisionWaterfall({ steps, height = 140 }: {
  steps: Array<{ name: string; value: number; blocked?: boolean }>;
  height?: number;
}) {
  if (!steps || steps.length === 0) {
    return <EmptyChart height={height} label="No decision data" />;
  }
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={steps} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
        <CartesianGrid strokeDasharray="2 2" stroke={COLORS.grid} />
        <XAxis dataKey="name" tick={{ fontSize: 9, fill: COLORS.text }} axisLine={false} />
        <YAxis domain={[0, 1]} tick={{ fontSize: 9, fill: COLORS.text }} axisLine={false} />
        <Tooltip contentStyle={tooltipStyle} />
        <Bar dataKey="value" name="Size Factor">
          {steps.map((s, i) => (
            <Cell key={i} fill={s.blocked ? COLORS.red : s.value >= 0.7 ? COLORS.green : s.value >= 0.3 ? COLORS.yellow : COLORS.red} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// =========================================================================
// 11. Allocation donut (portfolio per-asset)
// =========================================================================

export function AllocationBars({ data, height = 100 }: {
  data: Array<{ asset: string; value: number }>;
  height?: number;
}) {
  if (!data || data.length === 0 || data.every(d => d.value === 0)) {
    return <EmptyChart height={height} label="No allocation" />;
  }
  const total = data.reduce((s, d) => s + d.value, 0) || 1;
  return (
    <div className="space-y-1.5">
      {data.map(d => {
        const pct = (d.value / total) * 100;
        return (
          <div key={d.asset}>
            <div className="flex justify-between text-xs mb-0.5">
              <span className="text-gray-400">{d.asset.replace('USDT', '')}</span>
              <span className="text-gray-300 font-mono">${(d.value / 1000).toFixed(1)}k ({pct.toFixed(0)}%)</span>
            </div>
            <div className="w-full h-2 bg-gray-800 rounded overflow-hidden">
              <div className="h-full rounded transition-all" style={{ width: `${pct}%`, backgroundColor: ASSET_COLORS[d.asset] || COLORS.blue }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

// =========================================================================
// 12. Live vs Research comparison
// =========================================================================

export function LiveVsResearch({ rows, height = 140 }: {
  rows: Array<{ metric: string; expected: number; actual: number; unit?: string }>;
  height?: number;
}) {
  if (!rows || rows.length === 0) {
    return <EmptyChart height={height} label="No comparison data" />;
  }
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={rows} margin={{ top: 5, right: 10, left: -20, bottom: 0 }} layout="vertical">
        <CartesianGrid strokeDasharray="2 2" stroke={COLORS.grid} horizontal={false} />
        <XAxis type="number" tick={{ fontSize: 9, fill: COLORS.text }} axisLine={false} />
        <YAxis type="category" dataKey="metric" tick={{ fontSize: 10, fill: COLORS.text }} axisLine={false} width={90} />
        <Tooltip contentStyle={tooltipStyle} />
        <Legend wrapperStyle={{ fontSize: '10px' }} />
        <Bar dataKey="expected" fill={COLORS.blue} name="Expected (OOS)" />
        <Bar dataKey="actual" fill={COLORS.yellow} name="Actual (Live)" />
      </BarChart>
    </ResponsiveContainer>
  );
}

// =========================================================================
// Helpers
// =========================================================================

function EmptyChart({ height, label }: { height: number; label: string }) {
  return (
    <div style={{ height }} className="flex items-center justify-center text-xs text-gray-600 border border-dashed border-gray-800 rounded">
      {label}
    </div>
  );
}

export { ASSET_COLORS };
