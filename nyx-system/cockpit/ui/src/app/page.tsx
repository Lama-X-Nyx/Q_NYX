'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import { fetchApi, connectWs } from '@/lib/api';
import type { SystemState, PortfolioState, HealthSnapshot, Order, DecisionResult, WsMessage } from '@/lib/api';

const ASSETS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'];

function StatusBadge({ state }: { state: string }) {
  const colors: Record<string, string> = {
    running: 'bg-green-500', normal: 'bg-green-500',
    paused: 'bg-yellow-500', degraded: 'bg-yellow-500',
    stopped: 'bg-red-500', critical: 'bg-red-500',
  };
  return (
    <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-xs font-medium ${colors[state] || 'bg-gray-500'} text-white`}>
      <span className="w-1.5 h-1.5 rounded-full bg-white/60 animate-pulse" />
      {state.toUpperCase()}
    </span>
  );
}

function Panel({ title, children, className = '' }: { title: string; children: React.ReactNode; className?: string }) {
  return (
    <div className={`bg-gray-900 border border-gray-800 rounded-lg overflow-hidden ${className}`}>
      <div className="px-3 py-2 border-b border-gray-800 bg-gray-900/50">
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">{title}</h3>
      </div>
      <div className="p-3">{children}</div>
    </div>
  );
}

function MetricRow({ label, value, unit = '', color = '' }: { label: string; value: string | number; unit?: string; color?: string }) {
  return (
    <div className="flex justify-between items-center py-1">
      <span className="text-xs text-gray-500">{label}</span>
      <span className={`text-sm font-mono ${color || 'text-gray-200'}`}>
        {value}{unit}
      </span>
    </div>
  );
}

function HealthBar({ value, label }: { value: number; label: string }) {
  const pct = Math.max(0, Math.min(100, value * 100));
  const color = pct >= 70 ? 'bg-green-500' : pct >= 35 ? 'bg-yellow-500' : 'bg-red-500';
  return (
    <div className="mb-2">
      <div className="flex justify-between text-xs mb-1">
        <span className="text-gray-500">{label}</span>
        <span className="text-gray-300">{pct.toFixed(0)}%</span>
      </div>
      <div className="w-full h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full transition-all duration-500`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function MultiplierGauge({ value }: { value: number }) {
  const color = value >= 0.8 ? 'text-green-400' : value >= 0.3 ? 'text-yellow-400' : 'text-red-400';
  return (
    <div className="text-center">
      <div className={`text-2xl font-bold font-mono ${color}`}>{value.toFixed(2)}</div>
      <div className="text-xs text-gray-500">Risk Multiplier</div>
    </div>
  );
}

function PipelineStep({ name, status, detail }: { name: string; status: 'pass' | 'block' | 'skip' | 'none'; detail?: string }) {
  const colors = { pass: 'border-green-500 text-green-400', block: 'border-red-500 text-red-400', skip: 'border-yellow-500 text-yellow-400', none: 'border-gray-700 text-gray-600' };
  const icons = { pass: '\u2713', block: '\u2715', skip: '\u2298', none: '\u00B7' };
  return (
    <div className={`flex items-center gap-2 px-2 py-1.5 border-l-2 ${colors[status]}`}>
      <span className="w-4 text-center font-bold text-xs">{icons[status]}</span>
      <span className="text-xs flex-1">{name}</span>
      {detail && <span className="text-xs text-gray-500 font-mono">{detail}</span>}
    </div>
  );
}

export default function Dashboard() {
  const [state, setState] = useState<SystemState | null>(null);
  const [portfolio, setPortfolio] = useState<PortfolioState | null>(null);
  const [health, setHealth] = useState<HealthSnapshot | null>(null);
  const [orders, setOrders] = useState<Order[]>([]);
  const [decisions, setDecisions] = useState<Record<string, DecisionResult>>({});
  const [selectedAsset, setSelectedAsset] = useState('BTCUSDT');
  const [wsConnected, setWsConnected] = useState(false);
  const [equityCurve, setEquityCurve] = useState<number[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  const handleWsMessage = useCallback((msg: WsMessage) => {
    if (msg.channel === 'runtime') {
      const d = msg.data as { symbol: string; result: DecisionResult };
      setDecisions(prev => ({ ...prev, [d.symbol as string]: d.result }));
    } else if (msg.channel === 'health') {
      setHealth(msg.data as HealthSnapshot);
    } else if (msg.channel === 'portfolio') {
      const pd = msg.data as { equity: number };
      setEquityCurve(prev => [...prev.slice(-200), pd.equity]);
    }
  }, []);

  useEffect(() => {
    const poll = async () => {
      try {
        const [s, p, h, o] = await Promise.all([
          fetchApi<SystemState>('/api/state'),
          fetchApi<PortfolioState>('/api/portfolio'),
          fetchApi<HealthSnapshot>('/api/health'),
          fetchApi<Order[]>('/api/orders?limit=20'),
        ]);
        setState(s);
        setPortfolio(p);
        setHealth(h);
        setOrders(o);
        for (const sym of ASSETS) {
          try {
            const d = await fetchApi<DecisionResult>(`/api/decision/${sym}`);
            setDecisions(prev => ({ ...prev, [sym]: d }));
          } catch {}
        }
      } catch {}
    };
    poll();
    const interval = setInterval(poll, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const ws = connectWs(handleWsMessage);
    wsRef.current = ws;
    ws.onopen = () => setWsConnected(true);
    ws.onclose = () => setWsConnected(false);
    return () => ws.close();
  }, [handleWsMessage]);

  const currentDecision = decisions[selectedAsset];
  const layers = (currentDecision?.layers || {}) as Record<string, Record<string, unknown>>;
  const signal = layers.signal as { direction?: number; conviction?: number; p_trade?: number } | undefined;
  const fq = layers.fractal_quality as { quality_bucket?: string; size_multiplier?: number } | undefined;
  const em = layers.execution_monitor as { risk_multiplier?: number; health_score?: number; flags?: string[] } | undefined;
  const risk = layers.risk as { allowed?: boolean; reason?: string } | undefined;
  const omsLayer = layers.oms as { order_id?: number; side?: string; quantity?: number; limit_price?: number } | undefined;

  const pipelineStatus = (key: string): 'pass' | 'block' | 'skip' | 'none' => {
    if (!currentDecision) return 'none';
    const action = currentDecision.action;
    if (action === 'FLAT') return key === 'gbm' ? 'skip' : 'none';
    if (action === 'SKIP_QUALITY' && key === 'fractal') return 'block';
    if (action === 'SKIP_DEPENDENCY' && key === 'dependency') return 'block';
    if (action === 'BLOCKED_ALLOCATOR' && key === 'allocator') return 'block';
    if (action === 'BLOCKED_EXECUTION_CRITICAL' && key === 'execution') return 'block';
    if (action === 'BLOCKED_RISK' && key === 'risk') return 'block';
    if (action === 'ORDER_SUBMITTED') return 'pass';
    if (layers[key === 'gbm' ? 'signal' : key]) return 'pass';
    return 'none';
  };

  const formatUsd = (v: number | undefined) => v != null ? `$${v.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}` : '-';
  const formatPct = (v: number | undefined) => v != null ? `${(v * 100).toFixed(1)}%` : '-';
  const pnlColor = (v: number | undefined) => !v ? '' : v > 0 ? 'text-green-400' : 'text-red-400';

  return (
    <div className="min-h-screen flex flex-col">
      <header className="bg-gray-900 border-b border-gray-800 px-4 py-2 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h1 className="text-lg font-bold tracking-tight">NYX<span className="text-blue-400">.</span>cockpit</h1>
          <StatusBadge state={state?.system_state || 'stopped'} />
          <span className="text-xs text-gray-500 uppercase">{state?.system_mode || 'paper'}</span>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <span className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-green-400' : 'bg-red-400'}`} />
            <span className="text-xs text-gray-500">{wsConnected ? 'WS connected' : 'WS disconnected'}</span>
          </div>
          <StatusBadge state={state?.global_execution_state || 'normal'} />
          <span className="text-xs text-gray-400 font-mono">{state?.timestamp || ''}</span>
        </div>
      </header>

      <div className="flex-1 grid grid-cols-12 gap-2 p-2">
        {/* Left — Assets + Portfolio */}
        <div className="col-span-2 space-y-2">
          <Panel title="Assets">
            <div className="space-y-1">
              {ASSETS.map(sym => {
                const pos = portfolio?.assets?.[sym];
                const assetHealth = health?.per_asset?.[sym];
                return (
                  <button
                    key={sym}
                    onClick={() => setSelectedAsset(sym)}
                    className={`w-full text-left px-2 py-2 rounded text-sm transition ${selectedAsset === sym ? 'bg-blue-500/20 border border-blue-500/50' : 'hover:bg-gray-800 border border-transparent'}`}
                  >
                    <div className="flex justify-between items-center">
                      <span className="font-medium">{sym.replace('USDT', '')}</span>
                      {assetHealth && <StatusBadge state={assetHealth.health_score >= 0.7 ? 'normal' : assetHealth.health_score >= 0.35 ? 'degraded' : 'critical'} />}
                    </div>
                    <div className="flex justify-between mt-1 text-xs text-gray-500">
                      <span>{pos?.side || 'flat'}</span>
                      <span className={pnlColor(pos?.unrealized_pnl)}>{pos?.unrealized_pnl ? formatUsd(pos.unrealized_pnl) : '-'}</span>
                    </div>
                  </button>
                );
              })}
            </div>
          </Panel>

          <Panel title="Portfolio">
            <MetricRow label="Equity" value={formatUsd(portfolio?.total_equity)} />
            <MetricRow label="Balance" value={formatUsd(portfolio?.available_balance)} />
            <MetricRow label="Exposure" value={formatUsd(portfolio?.total_exposure)} />
            <MetricRow label="Realized PnL" value={formatUsd(portfolio?.realized_pnl_total)} color={pnlColor(portfolio?.realized_pnl_total)} />
          </Panel>
        </div>

        {/* Center — Pipeline + Orders + Equity */}
        <div className="col-span-6 space-y-2">
          <Panel title={`Decision Pipeline \u2014 ${selectedAsset}`}>
            <div className="grid grid-cols-2 gap-2">
              <div className="space-y-0.5">
                <PipelineStep name="1. GBM Signal" status={pipelineStatus('gbm')} detail={signal ? `p=${(signal.p_trade ?? 0).toFixed(2)} dir=${signal.direction}` : undefined} />
                <PipelineStep name="2. Jesse Fractal" status={pipelineStatus('fractal_quality')} detail={fq ? `${fq.quality_bucket} \u00D7${(fq.size_multiplier ?? 0).toFixed(2)}` : undefined} />
                <PipelineStep name="3. Dependency" status={pipelineStatus('dependency')} />
                <PipelineStep name="4. Allocator" status={pipelineStatus('allocator')} />
              </div>
              <div className="space-y-0.5">
                <PipelineStep name="5. Exec Monitor" status={pipelineStatus('execution')} detail={em ? `mult=${(em.risk_multiplier ?? 0).toFixed(2)}` : undefined} />
                <PipelineStep name="6. Risk Engine" status={pipelineStatus('risk')} detail={risk && !risk.allowed ? (risk.reason ?? '').slice(0, 30) : undefined} />
                <PipelineStep name="7. OMS" status={pipelineStatus('oms')} detail={omsLayer ? `${omsLayer.side} ${(omsLayer.quantity ?? 0).toFixed(4)} @ ${(omsLayer.limit_price ?? 0).toFixed(0)}` : undefined} />
                <div className="px-2 py-1.5 mt-2">
                  <span className="text-xs text-gray-500">Action: </span>
                  <span className={`text-xs font-bold ${currentDecision?.action === 'ORDER_SUBMITTED' ? 'text-green-400' : currentDecision?.action === 'FLAT' ? 'text-gray-500' : 'text-yellow-400'}`}>
                    {currentDecision?.action || 'NO DATA'}
                  </span>
                </div>
              </div>
            </div>
          </Panel>

          <Panel title="Orders & Fills">
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-gray-500 border-b border-gray-800">
                    <th className="text-left py-1 px-1">Time</th>
                    <th className="text-left py-1 px-1">Asset</th>
                    <th className="text-left py-1 px-1">Side</th>
                    <th className="text-right py-1 px-1">Qty</th>
                    <th className="text-right py-1 px-1">Price</th>
                    <th className="text-left py-1 px-1">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {orders.slice(0, 15).map(o => (
                    <tr key={o.order_id} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                      <td className="py-1 px-1 text-gray-500">{new Date(o.created_at * 1000).toLocaleTimeString()}</td>
                      <td className="py-1 px-1">{o.symbol.replace('USDT', '')}</td>
                      <td className={`py-1 px-1 ${o.side === 'buy' ? 'text-green-400' : 'text-red-400'}`}>{o.side.toUpperCase()}</td>
                      <td className="py-1 px-1 text-right font-mono">{o.quantity.toFixed(4)}</td>
                      <td className="py-1 px-1 text-right font-mono">{o.price.toFixed(2)}</td>
                      <td className="py-1 px-1">
                        <span className={`px-1.5 py-0.5 rounded text-xs ${o.status === 'FILLED' ? 'bg-green-500/20 text-green-400' : o.status === 'CANCELLED' ? 'bg-yellow-500/20 text-yellow-400' : o.status === 'REJECTED' ? 'bg-red-500/20 text-red-400' : 'bg-gray-700 text-gray-300'}`}>
                          {o.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                  {orders.length === 0 && (
                    <tr><td colSpan={6} className="text-center py-4 text-gray-600">No orders yet</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </Panel>

          <Panel title="Equity Curve">
            <div className="h-24 flex items-end gap-px">
              {equityCurve.length > 0 ? (
                (() => {
                  const min = Math.min(...equityCurve);
                  const max = Math.max(...equityCurve);
                  const range = max - min || 1;
                  return equityCurve.slice(-100).map((v, i) => (
                    <div key={i} className="flex-1 bg-blue-500/60 rounded-t" style={{ height: `${Math.max(4, ((v - min) / range) * 100)}%` }} />
                  ));
                })()
              ) : (
                <div className="w-full text-center text-xs text-gray-600 self-center">Waiting for data...</div>
              )}
            </div>
          </Panel>
        </div>

        {/* Right — Health + Comparison + Metrics */}
        <div className="col-span-4 space-y-2">
          <Panel title="Execution Health (T46.1)">
            {health?.per_asset ? (
              Object.entries(health.per_asset).map(([sym, h]) => (
                <div key={sym} className="mb-3 pb-3 border-b border-gray-800 last:border-0">
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-xs font-medium">{sym.replace('USDT', '')}</span>
                    <MultiplierGauge value={h.multiplier} />
                  </div>
                  <HealthBar value={h.health_score} label="Health" />
                  <div className="grid grid-cols-3 gap-2 text-xs mt-1">
                    <div><span className="text-gray-500">Fill </span><span className="text-gray-300">{formatPct(h.fill_rate)}</span></div>
                    <div><span className="text-gray-500">Miss </span><span className="text-gray-300">{formatPct(h.miss_rate)}</span></div>
                    <div><span className="text-gray-500">Fee </span><span className="text-gray-300">{h.avg_fee.toFixed(2)}</span></div>
                  </div>
                  {h.flags.length > 0 && (
                    <div className="flex flex-wrap gap-1 mt-1">
                      {h.flags.map(f => (
                        <span key={f} className={`text-xs px-1.5 py-0.5 rounded ${f.includes('critical') || f.includes('collapse') ? 'bg-red-500/20 text-red-400' : f.includes('degraded') || f.includes('spike') ? 'bg-yellow-500/20 text-yellow-400' : 'bg-green-500/20 text-green-400'}`}>
                          {f}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))
            ) : (
              <div className="text-xs text-gray-600">No execution data yet</div>
            )}
            {health?.systemic_degradation && (
              <div className="bg-red-500/20 border border-red-500/50 rounded px-2 py-1 text-xs text-red-400 font-bold mt-2">
                SYSTEMIC DEGRADATION DETECTED
              </div>
            )}
          </Panel>

          <Panel title="Live vs Research">
            <div className="text-xs text-gray-500 space-y-1">
              <div className="flex justify-between"><span>Expected fill rate (OOS)</span><span className="text-gray-300">66%</span></div>
              <div className="flex justify-between"><span>Actual fill rate</span><span className="text-gray-300">{health?.per_asset?.[selectedAsset]?.fill_rate ? formatPct(health.per_asset[selectedAsset].fill_rate) : '-'}</span></div>
              <div className="flex justify-between"><span>Expected miss rate</span><span className="text-gray-300">34%</span></div>
              <div className="flex justify-between"><span>Actual miss rate</span><span className="text-gray-300">{health?.per_asset?.[selectedAsset]?.miss_rate ? formatPct(health.per_asset[selectedAsset].miss_rate) : '-'}</span></div>
            </div>
          </Panel>

          <Panel title="System Metrics">
            <MetricRow label="Global Multiplier" value={(state?.global_risk_multiplier ?? 1).toFixed(2)} />
            <MetricRow label="Connected Assets" value={state?.connected_assets?.length || 0} />
            <MetricRow label="Open Orders" value={orders.filter(o => o.status === 'SUBMITTED').length} />
            <MetricRow label="Total Fills" value={orders.filter(o => o.status === 'FILLED').length} />
          </Panel>
        </div>
      </div>
    </div>
  );
}
