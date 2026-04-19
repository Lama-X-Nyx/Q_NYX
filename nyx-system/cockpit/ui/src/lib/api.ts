const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8100';
const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8100';

export async function fetchApi<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export function connectWs(onMessage: (data: WsMessage) => void): WebSocket {
  const ws = new WebSocket(`${WS_BASE}/ws/live`);
  ws.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data) as WsMessage;
      onMessage(msg);
    } catch {}
  };
  ws.onclose = () => {
    setTimeout(() => connectWs(onMessage), 3000);
  };
  return ws;
}

export interface WsMessage {
  channel: string;
  data: Record<string, unknown>;
  ts: number;
}

export interface SystemState {
  system_state: string;
  system_mode: string;
  timestamp: string;
  connected_assets: string[];
  global_execution_state: string;
  global_risk_multiplier: number;
}

export interface PortfolioState {
  total_equity?: number;
  available_balance?: number;
  total_exposure?: number;
  realized_pnl_total?: number;
  assets: Record<string, {
    side: string;
    quantity: number;
    avg_entry: number;
    unrealized_pnl: number;
    realized_pnl: number;
  }>;
}

export interface Order {
  order_id: number;
  client_order_id: string;
  symbol: string;
  side: string;
  quantity: number;
  price: number;
  status: string;
  filled_qty: number;
  avg_fill_price: number;
  created_at: number;
  reject_reason: string | null;
  cancel_reason: string | null;
}

export interface HealthSnapshot {
  available?: boolean;
  per_asset?: Record<string, {
    placed: number;
    filled: number;
    missed: number;
    fill_rate: number;
    miss_rate: number;
    avg_fee: number;
    health_score: number;
    multiplier: number;
    flags: string[];
  }>;
  systemic_degradation?: boolean;
}

export interface DecisionResult {
  timestamp?: string;
  action: string;
  layers: Record<string, unknown>;
}

export interface HealthPoint {
  ts: string | number;
  health: number;
  smoothed: number;
  multiplier: number;
}

export interface EquityPoint {
  ts: string | number;
  equity: number;
  balance: number;
  exposure: number;
  pnl: number;
}

export interface FillMissBar {
  asset: string;
  placed: number;
  filled: number;
  timed_out: number;
}

export interface StabilityAnalytics {
  run_id: string;
  heatmap_segments: Array<{ asset: string; segment: string; sharpe: number; pnl: number; dd: number }>;
  stability_scores: Array<{ asset: string; score_v1: number; score_v2: number; classification: string; flags: string[] }>;
  regime_matrix: Array<{ asset: string; regime: string; sharpe: number; dd: number; wr: number }>;
  tail_distribution: Array<{ asset: string; worst_trade: number; tail_95: number; tail_99: number; skewness: number; kurtosis: number }>;
}

export interface CapacityAnalytics {
  run_id: string;
  ladder_series: Array<{ asset: string; capital: number; sharpe: number; pnl: number; retention: number }>;
  deployability: Array<{ asset: string; class: string; max_capital: number; flags: string[] }>;
  breakpoints: Array<{ asset: string; capital: number; reason: string }>;
}

export interface ExecutionStressAnalytics {
  run_id: string;
  heatmap_cells: Array<{ asset: string; capital: number; offset_bps: number; fill_degradation: number; fee_multiplier: number; composite_retention: number; miss_rate: number }>;
  sensitivities: Array<{ asset: string; offset: number; fill: number; fee: number }>;
  deployability: Array<{ asset: string; class: string; flags: string[] }>;
}
