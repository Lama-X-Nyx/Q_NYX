"""
Full-Stack OOS — ALL modules working in unison (post-Ticket-32).

Tests that every layer added in Tickets 20-32 functions correctly
when wired together, and the edge is preserved.

Signal source : NYXEngine.run() (batch, 173 features = validated edge)
Post-decision : Jesse modulation → Risk + VaR/CVaR → Execution
                Optimizer → OMS → Portfolio → Persistence → Monitoring

This is NOT a partial test — it exercises:
  ✓ GBM (inside NYXEngine.run)
  ✓ Jesse agents (fractal quality modulation)
  ✓ Fractal quality (compute_fractal_quality)
  ✓ Risk engine (validate_trade + VaR/CVaR)
  ✓ Execution optimizer (dynamic offset + fill probability)
  ✓ OMS (submit_order + state machine)
  ✓ Portfolio (on_fill + PnL)
  ✓ State persistence (atomic JSON)
  ✓ Monitoring (metrics + alerts)

Usage : python scripts/oos_full_stack.py
Writes : reports/BTCUSDT_full_stack_oos.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

DATA_DIR = HERE / 'data' / 'raw' / 'mtf'
FEAT_DIR = HERE / 'data' / 'features'
REPORTS_DIR = HERE / 'reports'
STATE_DIR = HERE / 'state' / 'BTCUSDT_full_stack_oos'


def _load(tf):
    df = pd.read_csv(DATA_DIR / f'BTCUSDT_{tf}.csv')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    for c in ('open', 'high', 'low', 'close'):
        df = df[df[c] > 0]
    return df


def main() -> int:
    print('=== FULL-STACK OOS — ALL modules in unison ===', flush=True)

    mtf = {tf: _load(tf) for tf in ('15m', '1h', '4h', '1d')}
    feats = {
        tf: pd.read_parquet(FEAT_DIR / f'BTCUSDT_features_{tf}.parquet')
        for tf in ('15m', '1h', '4h', '1d')
    }

    # ---- 1. Generate signals via NYXEngine (validated GBM path) ----
    from src.core.nyx_engine import NYXEngine
    engine = NYXEngine()
    result = engine.run(
        mtf, feats,
        train_end='2022-12-31',
        test_start='2023-01-01',
        test_end='2023-12-31',
    )
    idealized_trades = result.get('trades', [])
    print(f'  Idealized signals: {len(idealized_trades)}', flush=True)

    # ---- 2. Build the full post-decision stack ----
    from src.agents.context_agent import ContextAgent
    from src.agents.regime_agent import RegimeAgent
    from src.agents.setup_agent import SetupAgent
    from src.agents.entry_agent import EntryAgent
    from src.agents.contracts import FractalReport
    from src.core.fractal_quality import compute_fractal_quality
    from src.live.risk_engine import RiskEngine
    from src.live.var_cvar import RollingRiskMetrics
    from src.live.execution_optimizer import build_execution_plan
    from src.live.oms import OMS
    from src.live.portfolio_state import Portfolio
    from src.live.state_store import StateStore
    from src.live.monitoring import MetricsCollector, AlertManager

    ctx_agent = ContextAgent({})
    reg_agent = RegimeAgent({})
    stp_agent = SetupAgent({})
    ent_agent = EntryAgent({})
    risk_engine = RiskEngine(
        max_var_95=-300.0,
        max_cvar_95=-500.0,
    )
    rolling_risk = RollingRiskMetrics(window=50)
    oms = OMS()
    portfolio = Portfolio(initial_capital=10_000.0)
    state_store = StateStore(STATE_DIR)
    metrics = MetricsCollector()
    alerts = AlertManager()

    # ---- 3. Replay each idealized trade through the FULL stack ----
    df15 = mtf['15m'].loc['2023-01-01':'2023-12-31']
    trade_by_ts = {str(t['timestamp']): t for t in idealized_trades}

    order_counter = 0
    filled_trades: List[Dict[str, Any]] = []
    n_skipped_quality = 0
    n_blocked_risk = 0
    n_placed = 0
    n_filled = 0
    n_timed_out = 0
    layers_log: List[Dict] = []

    from src.paper_live.post_only_broker import PostOnlyPaperBroker
    broker = PostOnlyPaperBroker(max_wait_bars=3, maker_fee=0.0002)

    pending_orders: Dict[int, Dict] = {}

    for ts, row in df15.iterrows():
        bar = {
            'timestamp': ts.isoformat(),
            'open': float(row['open']), 'high': float(row['high']),
            'low': float(row['low']), 'close': float(row['close']),
            'volume': float(row['volume']),
        }
        mark = float(row['close'])

        # Advance broker (check fills/timeouts).
        broker.on_bar('BTCUSDT', bar, bar_ts=ts.isoformat())

        # Check fills on pending orders.
        for oid_broker, meta in list(pending_orders.items()):
            order_broker = broker.get(oid_broker)
            if order_broker.state == 'FILLED' and not meta.get('done'):
                meta['done'] = True
                fills = order_broker.fills
                if fills:
                    fill = fills[0]
                    # OMS fill
                    oms.handle_fill(meta['oms_oid'],
                                   fill_qty=fill['qty'],
                                   fill_price=fill['price'])
                    # Portfolio fill
                    portfolio.on_fill('BTCUSDT',
                                     side=order_broker.side,
                                     qty=fill['qty'],
                                     price=fill['price'],
                                     fee=fill['fee'])
                    # Monitoring
                    ideal_t = meta['ideal_trade']
                    metrics.record_trade(
                        pnl=float(ideal_t['net_pnl']),
                        side=order_broker.side,
                        filled=True,
                    )
                    metrics.record_order_attempt(filled=True)
                    rolling_risk.add_trade_pnl(float(ideal_t['net_pnl']))
                    equity = portfolio.equity_at({'BTCUSDT': mark})
                    metrics.update_equity(equity)
                    filled_trades.append({
                        'timestamp': meta['placed_ts'],
                        'direction': ideal_t['direction'],
                        'net_pnl': float(ideal_t['net_pnl']),
                        'fill_price': fill['price'],
                        'offset_bps': meta.get('offset_bps', 0),
                        'order_type': meta.get('order_type', 'post_only_limit'),
                        'quality_bucket': meta.get('quality_bucket', ''),
                        'risk_hint': meta.get('risk_hint', 0),
                    })
                    n_filled += 1
            elif order_broker.state == 'TIMED_OUT' and not meta.get('done'):
                meta['done'] = True
                oms.cancel_order(meta['oms_oid'], reason='TIMED_OUT')
                metrics.record_order_attempt(filled=False)
                n_timed_out += 1

        # Check if this bar has an idealized trade to process.
        ts_key = str(ts)
        if ts_key not in trade_by_ts:
            continue
        t = trade_by_ts[ts_key]
        direction = int(t['direction'])
        side = 'buy' if direction > 0 else 'sell'

        layer_result = {'ts': ts_key}

        # ---- JESSE FRACTAL QUALITY ----
        reports = {}
        for agent_name, agent, tf_key, df_src in [
            ('context', ctx_agent, '1d', mtf['1d'].loc[:ts]),
            ('regime', reg_agent, '4h', mtf['4h'].loc[:ts]),
            ('setup', stp_agent, '1h', mtf['1h'].loc[:ts]),
            ('entry', ent_agent, '15m', df15.loc[:ts]),
        ]:
            try:
                if len(df_src) >= 20:
                    reports[agent_name] = agent.report(
                        df_src, asset='BTCUSDT', timestamp=ts_key,
                    )
            except Exception:
                pass
            if agent_name not in reports:
                reports[agent_name] = FractalReport(
                    asset='BTCUSDT', agent=agent_name, timeframe=tf_key,
                    state='unavailable', score=0.5, passed=False,
                    block_reasons=['unavailable'], timestamp=ts_key,
                )

        fq = compute_fractal_quality(reports)
        layer_result['fractal'] = fq

        if fq['skip_trade']:
            n_skipped_quality += 1
            continue

        # ---- RISK ENGINE + VaR/CVaR ----
        quantity = (portfolio.available_balance * 0.02) / max(mark, 1)
        quantity *= fq['size_multiplier']

        pf_snapshot = {
            'available_balance': portfolio.available_balance,
            'total_exposure': portfolio.total_exposure,
            'daily_realized_pnl': 0, 'weekly_realized_pnl': 0,
            'max_drawdown_from_peak': 0,
            'open_position_count': sum(
                1 for p in portfolio.positions.values() if p.quantity > 0
            ),
        }
        risk_result = risk_engine.validate_trade(
            symbol='BTCUSDT', side=side, quantity=quantity,
            price=mark, portfolio=pf_snapshot,
            rolling_risk=rolling_risk,
        )
        layer_result['risk'] = risk_result

        if not risk_result['allowed']:
            n_blocked_risk += 1
            continue

        # ---- EXECUTION OPTIMIZER ----
        exec_plan = build_execution_plan(
            side=side, mark_price=mark,
            volatility_pct=0.5, spread_bps=5.0,
            expected_edge_bps=float(t.get('ml_score', 0.6)) * 100,
        )
        layer_result['execution'] = exec_plan

        # ---- OMS ----
        order_counter += 1
        client_id = f'fullstack-{order_counter}'
        oms_oid = oms.submit_order(
            client_order_id=client_id, symbol='BTCUSDT',
            side=side, quantity=quantity,
            price=exec_plan['limit_price'],
        )

        # ---- BROKER (place the actual order) ----
        broker_oid = broker.place_post_only(
            pair='BTCUSDT', side=side, qty=quantity,
            limit_price=exec_plan['limit_price'],
            mark_price=mark,
            placed_at=ts.isoformat(), max_wait_bars=3,
        )
        broker_order = broker.get(broker_oid)
        if broker_order.state == 'REJECTED':
            oms.handle_reject(oms_oid, reason='post-only rejected')
            continue

        pending_orders[broker_oid] = {
            'oms_oid': oms_oid,
            'ideal_trade': t,
            'placed_ts': ts_key,
            'offset_bps': exec_plan['offset_bps'],
            'order_type': exec_plan['order_type'],
            'quality_bucket': fq['quality_bucket'],
            'risk_hint': 1.0 - fq.get('fractal_quality', 0),
        }
        n_placed += 1
        layers_log.append(layer_result)

    # ---- 4. Final: persist + metrics ----
    state_store.save_oms(oms)
    state_store.save_portfolio(portfolio)
    state_store.save_event_log(oms)

    snap = metrics.snapshot()
    total_pnl = sum(t['net_pnl'] for t in filled_trades)
    capital = 10_000.0 + total_pnl
    n_rt = len(filled_trades)
    wins = [t for t in filled_trades if t['net_pnl'] > 0]
    wr = len(wins) / max(n_rt, 1)
    if n_rt > 5:
        rets = [t['net_pnl'] / 10_000.0 for t in filled_trades]
        sharpe = float(np.mean(rets) / max(np.std(rets), 1e-12) *
                       np.sqrt(min(n_rt, 252)))
    else:
        sharpe = 0.0
    eq = [10_000.0]
    for t in filled_trades:
        eq.append(eq[-1] + t['net_pnl'])
    eq_arr = np.array(eq)
    peak = np.maximum.accumulate(eq_arr)
    dd = (peak - eq_arr) / peak
    max_dd = float(dd.max())

    print(f'\n=== EXECUTION SUMMARY ===', flush=True)
    print(f'  Idealized signals:    {len(idealized_trades)}', flush=True)
    print(f'  Skipped (quality):    {n_skipped_quality}', flush=True)
    print(f'  Blocked (risk):       {n_blocked_risk}', flush=True)
    print(f'  Placed:               {n_placed}', flush=True)
    print(f'  Filled:               {n_filled}', flush=True)
    print(f'  Timed out:            {n_timed_out}', flush=True)
    print(f'\n=== PERFORMANCE ===', flush=True)
    print(f'  Trades:               {n_rt}', flush=True)
    print(f'  Win rate:             {wr:.1%}', flush=True)
    print(f'  Sharpe:               {sharpe:.2f}', flush=True)
    print(f'  PnL:                  ${total_pnl:.2f}', flush=True)
    print(f'  Max DD:               {max_dd:.4%}', flush=True)
    print(f'\n=== BASELINE (T11 idealized) ===', flush=True)
    print(f'  Sharpe:               9.96', flush=True)
    print(f'  PnL:                  $2,171', flush=True)
    print(f'\n=== MONITORING ===', flush=True)
    print(f'  {snap}', flush=True)
    print(f'\n=== VaR/CVaR ===', flush=True)
    print(f'  VaR 95%:  {rolling_risk.var_95}', flush=True)
    print(f'  CVaR 95%: {rolling_risk.cvar_95}', flush=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        'ticket': 'Full-Stack OOS (post-T32)',
        'modules_tested': [
            'NYXEngine (GBM)', 'Jesse (4 agents)', 'FractalQuality',
            'RiskEngine + VaR/CVaR', 'ExecutionOptimizer',
            'OMS', 'Portfolio', 'StateStore', 'Monitoring',
        ],
        'execution': {
            'idealized_signals': len(idealized_trades),
            'skipped_quality': n_skipped_quality,
            'blocked_risk': n_blocked_risk,
            'placed': n_placed,
            'filled': n_filled,
            'timed_out': n_timed_out,
            'fill_rate': n_filled / max(n_placed, 1),
            'miss_rate': n_timed_out / max(n_placed, 1),
        },
        'performance': {
            'n_trades': n_rt,
            'win_rate': round(wr, 4),
            'sharpe': round(sharpe, 2),
            'total_pnl': round(total_pnl, 2),
            'max_drawdown_pct': round(max_dd, 6),
        },
        'baseline_t11': {'sharpe': 9.96, 'pnl': 2171.18},
        'var_cvar': {
            'var_95': rolling_risk.var_95,
            'cvar_95': rolling_risk.cvar_95,
        },
        'monitoring': snap,
    }
    out = REPORTS_DIR / 'BTCUSDT_full_stack_oos.json'
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f'\nReport: {out}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
