'use client';

import { useEffect, useState, useCallback } from 'react';
import {
  fetchApi, fetchApiAuth, connectWs, postControlAuth,
  login as apiLogin, setAuthToken, getAuthToken, clearAuth,
} from '@/lib/api';
import type {
  SystemState, PortfolioState, HealthSnapshot, Order,
  DecisionResult, WsMessage, HealthPoint, EquityPoint, FillMissBar,
  StabilityAnalytics, CapacityAnalytics, ExecutionStressAnalytics,
  PaperControlStatus,
} from '@/lib/api';
import {
  HealthTimeSeries, EquityCurve, DrawdownCurve, FillMissBars,
  ScoreGauge, Heatmap, CapacityLadder, StabilityScoreBars,
  SensitivityBars, DecisionWaterfall, AllocationBars, LiveVsResearch,
} from '@/components/charts';

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

function Panel({ title, subtitle, children, className = '' }: {
  title: string; subtitle?: string; children: React.ReactNode; className?: string;
}) {
  return (
    <div className={`bg-gray-900 border border-gray-800 rounded-lg overflow-hidden ${className}`}>
      <div className="px-3 py-2 border-b border-gray-800 bg-gray-900/50 flex justify-between items-baseline">
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">{title}</h3>
        {subtitle && <span className="text-xs text-gray-600">{subtitle}</span>}
      </div>
      <div className="p-3">{children}</div>
    </div>
  );
}

function ClassBadge({ cls }: { cls: string }) {
  const colors: Record<string, string> = {
    core: 'bg-green-500/20 text-green-400 border-green-500/50',
    core_scalable: 'bg-green-500/20 text-green-400 border-green-500/50',
    core_scalable_under_realistic_execution: 'bg-green-500/20 text-green-400 border-green-500/50',
    robust: 'bg-green-500/20 text-green-400 border-green-500/50',
    satellite: 'bg-blue-500/20 text-blue-400 border-blue-500/50',
    satellite_only: 'bg-blue-500/20 text-blue-400 border-blue-500/50',
    scalable_with_caution: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/50',
    scalable_with_execution_caution: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/50',
    opportunistic: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50',
    opportunistic_small_only: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50',
    regime_sensitive: 'bg-orange-500/20 text-orange-400 border-orange-500/50',
    unstable: 'bg-red-500/20 text-red-400 border-red-500/50',
    execution_fragile: 'bg-red-500/20 text-red-400 border-red-500/50',
    avoid: 'bg-red-500/20 text-red-400 border-red-500/50',
    not_deployable: 'bg-red-500/20 text-red-400 border-red-500/50',
  };
  return (
    <span className={`text-xs px-2 py-0.5 rounded border ${colors[cls] || 'bg-gray-800 text-gray-400 border-gray-700'}`}>
      {cls.replace(/_/g, ' ')}
    </span>
  );
}

function ConfirmModal({ title, message, onConfirm, onCancel, destructive = false }: {
  title: string; message: string; onConfirm: () => void; onCancel: () => void; destructive?: boolean;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-6 max-w-md w-full shadow-2xl">
        <h3 className="text-lg font-bold mb-2">{title}</h3>
        <p className="text-sm text-gray-400 mb-6">{message}</p>
        <div className="flex justify-end gap-3">
          <button onClick={onCancel} className="px-4 py-2 text-sm rounded bg-gray-800 hover:bg-gray-700 text-gray-300">Cancel</button>
          <button onClick={onConfirm} className={`px-4 py-2 text-sm rounded font-medium ${destructive ? 'bg-red-600 hover:bg-red-500' : 'bg-blue-600 hover:bg-blue-500'} text-white`}>
            Confirm
          </button>
        </div>
      </div>
    </div>
  );
}

const PAPER_STATE_COLORS: Record<string, string> = {
  paper_enabled: 'bg-green-500',
  paper_paused: 'bg-yellow-500',
  paper_disabled: 'bg-gray-500',
  paper_critical_blocked: 'bg-red-500',
};

const PAPER_STATE_LABELS: Record<string, string> = {
  paper_enabled: 'PAPER ON',
  paper_paused: 'PAUSED',
  paper_disabled: 'PAPER OFF',
  paper_critical_blocked: 'CRITICAL BLOCK',
};

function LoginScreen({ onLogin }: { onLogin: (token: string, role: string, env: string) => void }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      const r = await apiLogin(username, password);
      setAuthToken(r.token);
      onLogin(r.token, r.role, r.environment);
    } catch {
      setError('Invalid credentials');
    } finally {
      setLoading(false);
    }
  };
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-950">
      <form onSubmit={handleSubmit} className="bg-gray-900 border border-gray-800 rounded-lg p-8 w-96 space-y-4">
        <h1 className="text-2xl font-bold text-center">NYX<span className="text-blue-400">.</span>cockpit</h1>
        <p className="text-xs text-center text-gray-500">Operator Login</p>
        {error && <div className="text-sm text-red-400 text-center">{error}</div>}
        <input type="text" placeholder="Username" value={username} onChange={e => setUsername(e.target.value)} className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm" autoFocus />
        <input type="password" placeholder="Password" value={password} onChange={e => setPassword(e.target.value)} className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm" />
        <button type="submit" disabled={loading} className="w-full py-2 bg-blue-600 hover:bg-blue-500 rounded text-sm font-medium disabled:opacity-50">
          {loading ? 'Connecting...' : 'Login'}
        </button>
      </form>
    </div>
  );
}

export default function App() {
  const [authed, setAuthed] = useState(false);
  const [role, setRole] = useState('');
  const [env, setEnv] = useState('paper');
  useEffect(() => {
    const t = getAuthToken();
    if (t) { setAuthed(true); }
  }, []);
  if (!authed) {
    return <LoginScreen onLogin={(token, r, e) => { setAuthed(true); setRole(r); setEnv(e); }} />;
  }
  return <Dashboard userRole={role} environment={env} onLogout={() => { clearAuth(); setAuthed(false); }} />;
}

function Dashboard({ userRole, environment, onLogout }: { userRole: string; environment: string; onLogout: () => void }) {
  const [state, setState] = useState<SystemState | null>(null);
  const [portfolio, setPortfolio] = useState<PortfolioState | null>(null);
  const [health, setHealth] = useState<HealthSnapshot | null>(null);
  const [orders, setOrders] = useState<Order[]>([]);
  const [decisions, setDecisions] = useState<Record<string, DecisionResult>>({});
  const [selectedAsset, setSelectedAsset] = useState('BTCUSDT');
  const [wsConnected, setWsConnected] = useState(false);
  const [healthSeries, setHealthSeries] = useState<Record<string, HealthPoint[]>>({});
  const [equitySeries, setEquitySeries] = useState<EquityPoint[]>([]);
  const [fillMissBars, setFillMissBars] = useState<FillMissBar[]>([]);
  const [runs, setRuns] = useState<string[]>([]);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [paperStatus, setPaperStatus] = useState<PaperControlStatus | null>(null);
  const [confirmAction, setConfirmAction] = useState<{ action: string; title: string; message: string; destructive: boolean } | null>(null);
  const [stability, setStability] = useState<StabilityAnalytics | null>(null);
  const [capacity, setCapacity] = useState<CapacityAnalytics | null>(null);
  const [execStress, setExecStress] = useState<ExecutionStressAnalytics | null>(null);

  const handleWsMessage = useCallback((msg: WsMessage) => {
    if (msg.channel === 'runtime') {
      const d = msg.data as { symbol: string; result: DecisionResult };
      setDecisions(prev => ({ ...prev, [d.symbol]: d.result }));
    } else if (msg.channel === 'health') {
      setHealth(msg.data as HealthSnapshot);
    }
  }, []);

  useEffect(() => {
    const poll = async () => {
      try {
        const [s, p, h, o, fm, rl, ps] = await Promise.all([
          fetchApi<SystemState>('/api/state'),
          fetchApi<PortfolioState>('/api/portfolio'),
          fetchApi<HealthSnapshot>('/api/health'),
          fetchApi<Order[]>('/api/orders?limit=30'),
          fetchApi<FillMissBar[]>('/api/analytics/fill_miss_bars'),
          fetchApi<string[]>('/api/runs'),
          fetchApi<PaperControlStatus>('/api/control/paper/status'),
        ]);
        setState(s);
        setPortfolio(p);
        setHealth(h);
        setOrders(o);
        setFillMissBars(fm);
        setRuns(rl);
        setPaperStatus(ps);
        for (const sym of ASSETS) {
          try {
            const [d, hs] = await Promise.all([
              fetchApi<DecisionResult>(`/api/decision/${sym}`),
              fetchApi<HealthPoint[]>(`/api/series/health/${sym}`),
            ]);
            setDecisions(prev => ({ ...prev, [sym]: d }));
            setHealthSeries(prev => ({ ...prev, [sym]: hs }));
          } catch {}
        }
        const eq = await fetchApi<EquityPoint[]>('/api/series/equity');
        setEquitySeries(eq);
      } catch {}
    };
    poll();
    const interval = setInterval(poll, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const ws = connectWs(handleWsMessage);
    ws.onopen = () => setWsConnected(true);
    ws.onclose = () => setWsConnected(false);
    return () => ws.close();
  }, [handleWsMessage]);

  useEffect(() => {
    if (!selectedRun) return;
    (async () => {
      try {
        const [st, cp, es] = await Promise.all([
          fetchApi<StabilityAnalytics>(`/api/analytics/stability/${selectedRun}`),
          fetchApi<CapacityAnalytics>(`/api/analytics/capacity/${selectedRun}`),
          fetchApi<ExecutionStressAnalytics>(`/api/analytics/execution_stress/${selectedRun}`),
        ]);
        setStability(st);
        setCapacity(cp);
        setExecStress(es);
      } catch {}
    })();
  }, [selectedRun]);

  const currentDecision = decisions[selectedAsset];
  const layers = (currentDecision?.layers || {}) as Record<string, Record<string, unknown>>;
  const signal = layers.signal as { direction?: number; conviction?: number; p_trade?: number } | undefined;
  const fq = layers.fractal_quality as { quality_bucket?: string; size_multiplier?: number } | undefined;
  const dep = layers.dependency as { adjusted_size_multiplier?: number } | undefined;
  const emLayer = layers.execution_monitor as { risk_multiplier?: number; health_score?: number; flags?: string[] } | undefined;
  const risk = layers.risk as { allowed?: boolean; reason?: string } | undefined;

  const assetHealth = health?.per_asset?.[selectedAsset];
  const formatUsd = (v: number | undefined) => v != null ? `$${v.toLocaleString('en-US', { maximumFractionDigits: 0 })}` : '-';
  const formatPct = (v: number | undefined) => v != null ? `${(v * 100).toFixed(1)}%` : '-';
  const pnlColor = (v: number | undefined) => !v ? '' : v > 0 ? 'text-green-400' : 'text-red-400';

  const waterfallSteps = [
    { name: 'GBM', value: signal?.p_trade ?? 0, blocked: !signal?.direction },
    { name: 'Fractal', value: fq?.size_multiplier ?? 0, blocked: currentDecision?.action === 'SKIP_QUALITY' },
    { name: 'Dependency', value: dep?.adjusted_size_multiplier ?? fq?.size_multiplier ?? 0, blocked: currentDecision?.action === 'SKIP_DEPENDENCY' },
    { name: 'ExecMon', value: emLayer?.risk_multiplier ?? 1.0, blocked: currentDecision?.action === 'BLOCKED_EXECUTION_CRITICAL' },
    { name: 'Risk', value: risk?.allowed === false ? 0 : 1, blocked: currentDecision?.action === 'BLOCKED_RISK' },
  ];

  const allocationData = ASSETS.map(sym => {
    const pos = portfolio?.assets?.[sym];
    const mark = 0;
    const val = pos ? Math.abs(pos.quantity * (pos.avg_entry || mark)) : 0;
    return { asset: sym, value: val };
  });

  const liveVsResearchRows = [
    { metric: 'Fill rate %', expected: 66, actual: (assetHealth?.fill_rate ?? 0) * 100 },
    { metric: 'Miss rate %', expected: 34, actual: (assetHealth?.miss_rate ?? 0) * 100 },
    { metric: 'Health score', expected: 100, actual: (assetHealth?.health_score ?? 0) * 100 },
  ];

  const paperState = paperStatus?.current_state || 'paper_disabled';

  const handlePaperAction = async (action: string) => {
    const r = await postControlAuth(action);
    if (r.state) {
      setPaperStatus(prev => prev ? { ...prev, current_state: r.state! } : prev);
    }
    setConfirmAction(null);
  };

  const askConfirm = (action: string, title: string, message: string, destructive = false) => {
    setConfirmAction({ action, title, message, destructive });
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Confirmation modal */}
      {confirmAction && (
        <ConfirmModal
          title={confirmAction.title}
          message={confirmAction.message}
          destructive={confirmAction.destructive}
          onConfirm={() => handlePaperAction(confirmAction.action)}
          onCancel={() => setConfirmAction(null)}
        />
      )}

      {/* Top bar */}
      <header className="bg-gray-900 border-b border-gray-800 px-4 py-2 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h1 className="text-lg font-bold tracking-tight">NYX<span className="text-blue-400">.</span>cockpit</h1>

          {/* Paper status badge */}
          <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-bold text-white ${PAPER_STATE_COLORS[paperState] || 'bg-gray-500'}`}>
            <span className="w-2 h-2 rounded-full bg-white/50 animate-pulse" />
            {PAPER_STATE_LABELS[paperState] || paperState}
          </span>

          {/* Primary controls */}
          <div className="flex items-center gap-1 border-l border-gray-700 pl-3">
            {paperState === 'paper_disabled' ? (
              <button onClick={() => handlePaperAction('on')} className="px-3 py-1 rounded text-xs font-medium bg-green-600 hover:bg-green-500 text-white">
                START PAPER
              </button>
            ) : paperState === 'paper_enabled' ? (
              <>
                <button onClick={() => handlePaperAction('pause')} className="px-3 py-1 rounded text-xs font-medium bg-yellow-600 hover:bg-yellow-500 text-white">
                  PAUSE
                </button>
                <button onClick={() => askConfirm('off', 'Stop Paper Trading', 'This will stop all paper trading. No new orders will be placed. Existing positions remain.', true)} className="px-3 py-1 rounded text-xs font-medium bg-red-700 hover:bg-red-600 text-white">
                  STOP
                </button>
              </>
            ) : paperState === 'paper_paused' ? (
              <>
                <button onClick={() => handlePaperAction('resume')} className="px-3 py-1 rounded text-xs font-medium bg-green-600 hover:bg-green-500 text-white">
                  RESUME
                </button>
                <button onClick={() => askConfirm('off', 'Stop Paper Trading', 'This will stop paper trading completely.', true)} className="px-3 py-1 rounded text-xs font-medium bg-red-700 hover:bg-red-600 text-white">
                  STOP
                </button>
              </>
            ) : paperState === 'paper_critical_blocked' ? (
              <span className="text-xs text-red-400 font-bold animate-pulse">SYSTEM BLOCKED — {paperStatus?.reason || 'unknown'}</span>
            ) : null}
          </div>

          {/* Secondary controls */}
          <div className="flex items-center gap-1 border-l border-gray-700 pl-3">
            <button onClick={() => askConfirm('cancel_all', 'Cancel All Orders', 'Cancel all open paper orders. Filled positions remain.', false)} className="px-2 py-1 rounded text-xs bg-gray-700 hover:bg-gray-600 text-gray-300">
              Cancel All
            </button>
            <button onClick={() => askConfirm('flatten', 'Flatten Portfolio', 'Close ALL open positions. This is a destructive action.', true)} className="px-2 py-1 rounded text-xs bg-gray-700 hover:bg-red-700 text-gray-300 hover:text-white">
              Flatten
            </button>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <span className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-green-400' : 'bg-red-400'}`} />
            <span className="text-xs text-gray-500">{wsConnected ? 'WS' : 'OFFLINE'}</span>
          </div>
          <StatusBadge state={state?.global_execution_state || 'normal'} />
          <span className={`text-xs px-2 py-0.5 rounded font-bold ${environment === 'live' ? 'bg-red-500/20 text-red-400 border border-red-500/50' : environment === 'testnet' ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/50' : 'bg-green-500/20 text-green-400 border border-green-500/50'}`}>
            {environment?.toUpperCase() || 'PAPER'}
          </span>
          {userRole && <span className="text-xs text-gray-500">{userRole}</span>}
          <button onClick={onLogout} className="text-xs text-gray-500 hover:text-white px-2 py-0.5 rounded hover:bg-gray-800">Logout</button>
          <span className="text-xs text-gray-400 font-mono">{state?.timestamp || ''}</span>
        </div>
      </header>

      <div className="flex-1 grid grid-cols-12 gap-2 p-2">
        {/* LEFT — assets + allocation + portfolio */}
        <div className="col-span-2 space-y-2">
          <Panel title="Assets">
            <div className="space-y-1">
              {ASSETS.map(sym => {
                const pos = portfolio?.assets?.[sym];
                const ah = health?.per_asset?.[sym];
                return (
                  <button
                    key={sym}
                    onClick={() => setSelectedAsset(sym)}
                    className={`w-full text-left px-2 py-2 rounded text-sm transition ${selectedAsset === sym ? 'bg-blue-500/20 border border-blue-500/50' : 'hover:bg-gray-800 border border-transparent'}`}
                  >
                    <div className="flex justify-between items-center">
                      <span className="font-medium">{sym.replace('USDT', '')}</span>
                      {ah && <div className={`w-2 h-2 rounded-full ${ah.health_score >= 0.7 ? 'bg-green-400' : ah.health_score >= 0.35 ? 'bg-yellow-400' : 'bg-red-400'}`} />}
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

          <Panel title="Allocation">
            <AllocationBars data={allocationData} />
          </Panel>

          <Panel title="Portfolio" subtitle={formatUsd(portfolio?.total_equity)}>
            <div className="space-y-1 text-xs">
              <div className="flex justify-between"><span className="text-gray-500">Balance</span><span className="font-mono">{formatUsd(portfolio?.available_balance)}</span></div>
              <div className="flex justify-between"><span className="text-gray-500">Exposure</span><span className="font-mono">{formatUsd(portfolio?.total_exposure)}</span></div>
              <div className="flex justify-between"><span className="text-gray-500">Realized PnL</span><span className={`font-mono ${pnlColor(portfolio?.realized_pnl_total)}`}>{formatUsd(portfolio?.realized_pnl_total)}</span></div>
            </div>
          </Panel>
        </div>

        {/* CENTER — pipeline + portfolio charts + fill/miss */}
        <div className="col-span-6 space-y-2">
          <div className="grid grid-cols-2 gap-2">
            <Panel title={`Decision Waterfall — ${selectedAsset.replace('USDT', '')}`} subtitle={currentDecision?.action || 'NO DATA'}>
              <DecisionWaterfall steps={waterfallSteps} height={140} />
            </Panel>
            <Panel title="Signal Confidence" subtitle={signal ? `direction=${signal.direction}` : '-'}>
              <div className="flex items-center justify-around h-[140px]">
                <ScoreGauge value={signal?.p_trade ?? 0} label="p_trade" />
                <ScoreGauge value={fq?.size_multiplier ?? 0} label="size_mult" max={1.25} thresholds={[0.3, 0.7]} />
                <ScoreGauge value={emLayer?.risk_multiplier ?? 1} label="risk_mult" />
              </div>
            </Panel>
          </div>

          <Panel title="Equity Curve" subtitle={`${equitySeries.length} bars`}>
            <EquityCurve data={equitySeries} height={160} />
          </Panel>

          <div className="grid grid-cols-2 gap-2">
            <Panel title="Drawdown" subtitle="%">
              <DrawdownCurve data={equitySeries} height={120} />
            </Panel>
            <Panel title="Fill / Miss / Timeout" subtitle="per asset">
              <FillMissBars data={fillMissBars} height={120} />
            </Panel>
          </div>

          <Panel title="Recent Orders">
            <div className="overflow-x-auto max-h-40">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-gray-500 border-b border-gray-800 sticky top-0 bg-gray-900">
                    <th className="text-left py-1 px-1">Time</th>
                    <th className="text-left py-1 px-1">Asset</th>
                    <th className="text-left py-1 px-1">Side</th>
                    <th className="text-right py-1 px-1">Qty</th>
                    <th className="text-right py-1 px-1">Price</th>
                    <th className="text-left py-1 px-1">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {orders.slice(0, 10).map(o => (
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
                    <tr><td colSpan={6} className="text-center py-3 text-gray-600">No orders yet</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </Panel>
        </div>

        {/* RIGHT — execution health + live vs research */}
        <div className="col-span-4 space-y-2">
          <Panel title={`Execution Health — ${selectedAsset.replace('USDT', '')}`} subtitle={assetHealth ? `health=${assetHealth.health_score.toFixed(2)}` : '-'}>
            <HealthTimeSeries data={healthSeries[selectedAsset] || []} height={140} />
            {assetHealth?.flags && assetHealth.flags.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-2">
                {assetHealth.flags.map(f => (
                  <span key={f} className={`text-xs px-1.5 py-0.5 rounded ${f.includes('critical') || f.includes('collapse') ? 'bg-red-500/20 text-red-400' : f.includes('degraded') || f.includes('spike') ? 'bg-yellow-500/20 text-yellow-400' : 'bg-green-500/20 text-green-400'}`}>{f}</span>
                ))}
              </div>
            )}
            {health?.systemic_degradation && (
              <div className="bg-red-500/20 border border-red-500/50 rounded px-2 py-1 text-xs text-red-400 font-bold mt-2">
                SYSTEMIC DEGRADATION DETECTED
              </div>
            )}
          </Panel>

          <Panel title="Live vs Research" subtitle="expected vs actual">
            <LiveVsResearch rows={liveVsResearchRows} height={140} />
          </Panel>

          <Panel title="Research — OOS Runs" subtitle={`${runs.length} saved`}>
            <select
              value={selectedRun || ''}
              onChange={(e) => setSelectedRun(e.target.value || null)}
              className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-200 mb-2"
            >
              <option value="">— Select a run —</option>
              {runs.map(r => <option key={r} value={r}>{r}</option>)}
            </select>

            {stability && stability.stability_scores && (
              <>
                <div className="text-xs text-gray-500 mb-1 mt-2">Stability Score (T43.1)</div>
                <StabilityScoreBars data={stability.stability_scores} height={100} />
                <div className="flex flex-wrap gap-1 mt-1">
                  {stability.stability_scores.map(s => (
                    <div key={s.asset} className="text-xs">
                      <span className="text-gray-400">{s.asset.replace('USDT', '')}:</span>
                      <ClassBadge cls={s.classification} />
                    </div>
                  ))}
                </div>
              </>
            )}

            {stability && stability.regime_matrix && stability.regime_matrix.length > 0 && (
              <>
                <div className="text-xs text-gray-500 mb-1 mt-3">Regime Matrix (Sharpe)</div>
                <Heatmap
                  cells={stability.regime_matrix}
                  xKey="asset"
                  yKey="regime"
                  valueKey="sharpe"
                  height={120}
                  colorScale="green-red"
                />
              </>
            )}

            {capacity && capacity.ladder_series && capacity.ladder_series.length > 0 && (
              <>
                <div className="text-xs text-gray-500 mb-1 mt-3">Capacity Ladder (Sharpe vs Capital)</div>
                <CapacityLadder data={capacity.ladder_series} metric="sharpe" height={140} />
                <div className="flex flex-wrap gap-1 mt-1">
                  {capacity.deployability.map(d => (
                    <div key={d.asset} className="text-xs flex items-center gap-1">
                      <span className="text-gray-400">{d.asset.replace('USDT', '')}:</span>
                      <ClassBadge cls={d.class} />
                    </div>
                  ))}
                </div>
              </>
            )}

            {execStress && execStress.sensitivities && execStress.sensitivities.length > 0 && (
              <>
                <div className="text-xs text-gray-500 mb-1 mt-3">Execution Sensitivities (T44.1)</div>
                <SensitivityBars data={execStress.sensitivities} height={120} />
              </>
            )}
          </Panel>
        </div>
      </div>
    </div>
  );
}
