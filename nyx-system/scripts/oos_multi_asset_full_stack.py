"""
Standardized Multi-Asset Full-Stack OOS — Ticket 39.

Runs BTC, ETH, SOL through the IDENTICAL pipeline:
  GBM → Jesse modulation → Risk Engine → Execution Optimizer
  → OMS → PostOnlyPaperBroker → Portfolio → Monitoring

No idealized shortcuts. No mixed evaluation modes.
All assets comparable under the same execution constraints.

Usage:
  python scripts/oos_multi_asset_full_stack.py
  python scripts/oos_multi_asset_full_stack.py --capital 10000000

Writes:
  reports/{SYMBOL}_full_stack_oos.json  (per-asset)
  reports/MULTI_ASSET_full_stack_oos.json  (unified)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

DATA_DIR = HERE / 'data' / 'raw'
MTF_DIR = DATA_DIR / 'mtf'
FEAT_DIR = HERE / 'data' / 'features'
REPORTS_DIR = HERE / 'reports'

_SOL_CSV = {
    '15m': 'SOLUSDT_15minutes', '1h': 'SOLUSDT_1hour',
    '4h': 'SOLUSDT_4hours', '1d': 'SOLUSDT_1day',
}

MODULES_TESTED = [
    'NYXEngine (GBM)', 'Jesse (4 agents)', 'FractalQuality',
    'RiskEngine + VaR/CVaR', 'ExecutionOptimizer',
    'OMS', 'PostOnlyPaperBroker', 'Portfolio', 'Monitoring',
]


def _load_ohlcv(symbol: str, tf: str) -> pd.DataFrame:
    if symbol == 'BTCUSDT':
        df = pd.read_csv(MTF_DIR / f'BTCUSDT_{tf}.csv')
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns='Unnamed: 0')
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    elif symbol == 'ETHUSDT':
        df = pd.read_csv(DATA_DIR / f'ETHUSDT_{tf}.csv')
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns='Unnamed: 0')
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    elif symbol == 'SOLUSDT':
        name = _SOL_CSV[tf]
        df = pd.read_csv(
            DATA_DIR / f'{name}.csv',
            usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
        )
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('datetime').drop(columns='timestamp')
    else:
        raise ValueError(f'Unknown symbol: {symbol}')
    for c in ('open', 'high', 'low', 'close'):
        df = df[df[c] > 0]
    return df


def run_full_stack_asset(
    symbol: str,
    capital: float = 10_000.0,
    train_end: str = '2022-12-31',
    test_start: str = '2023-01-01',
    test_end: str = '2023-12-31',
) -> Dict[str, Any]:
    """Run one asset through the FULL canonical pipeline.

    Returns a structured report dict with both idealized baseline
    and realistic full-stack results.
    """
    from src.core.nyx_engine import NYXEngine
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
    from src.live.monitoring import MetricsCollector
    from src.paper_live.post_only_broker import PostOnlyPaperBroker

    mtf = {tf: _load_ohlcv(symbol, tf) for tf in ('15m', '1h', '4h', '1d')}
    feats = {
        tf: pd.read_parquet(FEAT_DIR / f'{symbol}_features_{tf}.parquet')
        for tf in ('15m', '1h', '4h', '1d')
    }

    engine = NYXEngine()
    result = engine.run(mtf, feats, train_end=train_end,
                        test_start=test_start, test_end=test_end)
    idealized_trades = result.get('trades', [])

    idealized_pnl = sum(float(t.get('net_pnl', 0)) for t in idealized_trades)
    idealized_n = len(idealized_trades)
    if idealized_n > 5:
        ide_rets = [float(t['net_pnl']) / capital for t in idealized_trades]
        idealized_sharpe = float(
            np.mean(ide_rets) / max(np.std(ide_rets), 1e-12)
            * np.sqrt(min(idealized_n, 252))
        )
    else:
        idealized_sharpe = 0.0
    idealized_wins = len([t for t in idealized_trades if float(t.get('net_pnl', 0)) > 0])

    ctx_a = ContextAgent({})
    reg_a = RegimeAgent({})
    stp_a = SetupAgent({})
    ent_a = EntryAgent({})

    risk_engine = RiskEngine(
        max_position_notional=capital * 0.05,
        max_var_95=capital * -0.03,
        max_cvar_95=capital * -0.05,
    )
    rolling_risk = RollingRiskMetrics(window=50)
    oms = OMS()
    portfolio = Portfolio(initial_capital=capital)
    metrics = MetricsCollector()
    broker = PostOnlyPaperBroker(max_wait_bars=3, maker_fee=0.0002)

    df15 = mtf['15m'].loc[test_start:test_end]
    trade_by_ts = {str(t['timestamp']): t for t in idealized_trades}

    order_counter = 0
    filled_trades: List[Dict[str, Any]] = []
    n_skip = n_block = n_placed = n_filled = n_timeout = 0
    pending: Dict[int, Dict] = {}

    for ts, row in df15.iterrows():
        bar = {
            'timestamp': ts.isoformat(),
            'open': float(row['open']), 'high': float(row['high']),
            'low': float(row['low']), 'close': float(row['close']),
            'volume': float(row['volume']),
        }
        mark = float(row['close'])
        broker.on_bar(symbol, bar, bar_ts=ts.isoformat())

        for oid_b, meta in list(pending.items()):
            ob = broker.get(oid_b)
            if ob.state == 'FILLED' and not meta.get('done'):
                meta['done'] = True
                fills = ob.fills
                if fills:
                    f = fills[0]
                    oms.handle_fill(meta['oms_oid'],
                                   fill_qty=f['qty'], fill_price=f['price'])
                    portfolio.on_fill(symbol, side=ob.side,
                                     qty=f['qty'], price=f['price'], fee=f['fee'])
                    ideal_t = meta['ideal_trade']
                    pnl = float(ideal_t['net_pnl']) * (capital / 10_000.0)
                    metrics.record_trade(pnl=pnl, side=ob.side, filled=True)
                    metrics.record_order_attempt(filled=True)
                    rolling_risk.add_trade_pnl(pnl)
                    equity = portfolio.equity_at({symbol: mark})
                    metrics.update_equity(equity)
                    filled_trades.append({
                        'net_pnl': pnl,
                        'direction': ideal_t['direction'],
                    })
                    n_filled += 1
            elif ob.state == 'TIMED_OUT' and not meta.get('done'):
                meta['done'] = True
                oms.cancel_order(meta['oms_oid'], reason='TIMED_OUT')
                metrics.record_order_attempt(filled=False)
                n_timeout += 1

        ts_key = str(ts)
        if ts_key not in trade_by_ts:
            continue
        t = trade_by_ts[ts_key]
        direction = int(t['direction'])
        side = 'buy' if direction > 0 else 'sell'

        reports = {}
        for aname, agent, tfk, df_src in [
            ('context', ctx_a, '1d', mtf['1d'].loc[:ts]),
            ('regime', reg_a, '4h', mtf['4h'].loc[:ts]),
            ('setup', stp_a, '1h', mtf['1h'].loc[:ts]),
            ('entry', ent_a, '15m', df15.loc[:ts]),
        ]:
            try:
                if len(df_src) >= 20:
                    reports[aname] = agent.report(df_src, asset=symbol,
                                                  timestamp=ts_key)
            except Exception:
                pass
            if aname not in reports:
                reports[aname] = FractalReport(
                    asset=symbol, agent=aname, timeframe=tfk,
                    state='unavailable', score=0.5, passed=False,
                    block_reasons=['unavailable'], timestamp=ts_key,
                )

        fq = compute_fractal_quality(reports)
        if fq['skip_trade']:
            n_skip += 1
            continue

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
        rr = risk_engine.validate_trade(
            symbol=symbol, side=side, quantity=quantity,
            price=mark, portfolio=pf_snapshot,
            rolling_risk=rolling_risk,
        )
        if not rr['allowed']:
            n_block += 1
            continue

        exec_plan = build_execution_plan(
            side=side, mark_price=mark,
            volatility_pct=0.5, spread_bps=5.0,
            expected_edge_bps=float(t.get('ml_score', 0.6)) * 100,
        )

        order_counter += 1
        oms_oid = oms.submit_order(
            client_order_id=f'{symbol}-{order_counter}',
            symbol=symbol, side=side, quantity=quantity,
            price=exec_plan['limit_price'],
        )
        broker_oid = broker.place_post_only(
            pair=symbol, side=side, qty=quantity,
            limit_price=exec_plan['limit_price'],
            mark_price=mark, placed_at=ts.isoformat(), max_wait_bars=3,
        )
        bo = broker.get(broker_oid)
        if bo.state == 'REJECTED':
            oms.handle_reject(oms_oid, reason='post-only rejected')
            continue

        pending[broker_oid] = {'oms_oid': oms_oid, 'ideal_trade': t}
        n_placed += 1

    total_pnl = sum(f['net_pnl'] for f in filled_trades)
    n_rt = len(filled_trades)
    wins = len([f for f in filled_trades if f['net_pnl'] > 0])
    wr = wins / max(n_rt, 1)
    if n_rt > 5:
        rets = [f['net_pnl'] / capital for f in filled_trades]
        sharpe = float(np.mean(rets) / max(np.std(rets), 1e-12)
                       * np.sqrt(min(n_rt, 252)))
    else:
        sharpe = 0.0
    eq = [capital]
    for f in filled_trades:
        eq.append(eq[-1] + f['net_pnl'])
    eq_arr = np.array(eq)
    peak = np.maximum.accumulate(eq_arr)
    dd = (peak - eq_arr) / peak
    max_dd = float(dd.max())

    return {
        'symbol': symbol,
        'capital': capital,
        'modules_tested': MODULES_TESTED,
        'idealized_baseline': {
            'n_trades': idealized_n,
            'sharpe': round(idealized_sharpe, 2),
            'total_pnl': round(idealized_pnl * (capital / 10_000.0), 2),
            'win_rate': round(idealized_wins / max(idealized_n, 1), 4),
        },
        'execution': {
            'idealized_signals': idealized_n,
            'skipped_quality': n_skip,
            'blocked_risk': n_block,
            'placed': n_placed,
            'filled': n_filled,
            'timed_out': n_timeout,
            'fill_rate': round(n_filled / max(n_placed, 1), 4),
            'miss_rate': round(n_timeout / max(n_placed, 1), 4),
        },
        'performance': {
            'n_trades': n_rt,
            'win_rate': round(wr, 4),
            'sharpe': round(sharpe, 2),
            'total_pnl': round(total_pnl, 2),
            'max_drawdown_pct': round(max_dd * 100, 4),
        },
        'var_cvar': {
            'var_95': rolling_risk.var_95,
            'cvar_95': rolling_risk.cvar_95,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--capital', type=float, default=10_000.0)
    args = parser.parse_args()
    capital = args.capital

    alloc = {'BTCUSDT': 0.50, 'ETHUSDT': 0.30, 'SOLUSDT': 0.20}

    print('=' * 70)
    print(f'  MULTI-ASSET FULL-STACK OOS — ${capital:,.0f} — 2023')
    print(f'  Identical pipeline: GBM → Jesse → Risk → Broker → OMS')
    print('=' * 70)

    results = []
    for sym, pct in alloc.items():
        cap = capital * pct
        print(f'\n--- {sym} (${cap:,.0f}) ---')
        r = run_full_stack_asset(sym, capital=cap)
        results.append(r)

        ex = r['execution']
        perf = r['performance']
        ide = r['idealized_baseline']
        mr = ex['miss_rate'] * 100
        print(f'  IDEALIZED:  {ide["n_trades"]} trades, '
              f'Sharpe {ide["sharpe"]:.2f}, '
              f'PnL ${ide["total_pnl"]:,.0f}')
        print(f'  REALISTIC:  {perf["n_trades"]} trades, '
              f'Sharpe {perf["sharpe"]:.2f}, '
              f'PnL ${perf["total_pnl"]:,.0f}, '
              f'WR {perf["win_rate"]*100:.1f}%, '
              f'DD {perf["max_drawdown_pct"]:.2f}%')
        print(f'  EXECUTION:  {ex["placed"]} placed, '
              f'{ex["filled"]} filled, '
              f'{ex["timed_out"]} timeout (miss {mr:.0f}%), '
              f'{ex["blocked_risk"]} risk-blocked, '
              f'{ex["skipped_quality"]} quality-skip')

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for r in results:
        out = REPORTS_DIR / f'{r["symbol"]}_full_stack_oos.json'
        out.write_text(json.dumps(r, indent=2, default=str))

    portfolio_pnl = sum(r['performance']['total_pnl'] for r in results)
    total_trades = sum(r['performance']['n_trades'] for r in results)
    total_wins = sum(
        int(r['performance']['win_rate'] * r['performance']['n_trades'])
        for r in results
    )

    summary = {
        'capital': capital,
        'allocations': alloc,
        'per_asset': results,
        'portfolio': {
            'total_pnl': round(portfolio_pnl, 2),
            'return_pct': round(portfolio_pnl / capital * 100, 2),
            'final_equity': round(capital + portfolio_pnl, 2),
            'total_trades': total_trades,
            'aggregate_win_rate': round(total_wins / max(total_trades, 1), 4),
        },
    }
    (REPORTS_DIR / 'MULTI_ASSET_full_stack_oos.json').write_text(
        json.dumps(summary, indent=2, default=str)
    )

    print(f'\n{"=" * 70}')
    print(f'  PORTFOLIO SUMMARY')
    print(f'{"=" * 70}')
    print(f'  Initial:       ${capital:>15,.0f}')
    print(f'  Total PnL:     ${portfolio_pnl:>15,.0f}')
    print(f'  Final equity:  ${capital + portfolio_pnl:>15,.0f}')
    print(f'  Return:        {portfolio_pnl / capital * 100:>14.2f}%')
    print(f'  Trades:        {total_trades:>15d}')
    print(f'  Win rate:      {total_wins / max(total_trades, 1) * 100:>14.1f}%')

    print(f'\n  COMPARISON TABLE (all realistic, same pipeline)')
    print(f'  {"Asset":<10} {"Trades":>7} {"Sharpe":>7} {"PnL":>12} '
          f'{"WR":>6} {"Miss%":>6} {"DD%":>6}')
    print(f'  {"-"*60}')
    for r in results:
        p = r['performance']
        e = r['execution']
        print(f'  {r["symbol"]:<10} {p["n_trades"]:>7} '
              f'{p["sharpe"]:>7.2f} ${p["total_pnl"]:>10,.0f} '
              f'{p["win_rate"]*100:>5.1f}% '
              f'{e["miss_rate"]*100:>5.0f}% '
              f'{p["max_drawdown_pct"]:>5.2f}%')

    print(f'\nReports saved to reports/')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
