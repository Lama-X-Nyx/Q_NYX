"""
Run the 6 reality checks on the trio walk-forward and persist a
single JSON snapshot to reports/reality_check_numbers.json.

Each check matches the corresponding section in docs/REALITY_CHECK.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

DATA_DIR = HERE / 'data' / 'raw'
FEAT_DIR = HERE / 'data' / 'features'
BTC_MTF = DATA_DIR / 'mtf'
REPORT_PATH = HERE / 'reports' / 'reality_check_numbers.json'

INITIAL = 10_000.0
WALK_YEARS = (2020, 2021, 2022, 2023)


_SOL = {'15m': 'SOLUSDT_15minutes', '1h': 'SOLUSDT_1hour',
        '4h': 'SOLUSDT_4hours', '1d': 'SOLUSDT_1day'}


def _load_btc(tf):
    d = pd.read_csv(BTC_MTF / f'BTCUSDT_{tf}.csv')
    d['datetime'] = pd.to_datetime(d['datetime'])
    return d.set_index('datetime')


def _load_eth(tf):
    d = pd.read_csv(DATA_DIR / f'ETHUSDT_{tf}.csv')
    if 'Unnamed: 0' in d.columns:
        d = d.drop(columns='Unnamed: 0')
    d['datetime'] = pd.to_datetime(d['datetime'])
    d = d.set_index('datetime')
    for c in ('open', 'high', 'low', 'close'):
        d = d[d[c] > 0]
    return d


def _load_sol(tf):
    d = pd.read_csv(DATA_DIR / f'{_SOL[tf]}.csv',
                    usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    d['datetime'] = pd.to_datetime(d['timestamp'], unit='ms')
    d = d.set_index('datetime').drop(columns='timestamp')
    for c in ('open', 'high', 'low', 'close'):
        d = d[d[c] > 0]
    return d


def _walk_forward_trio():
    """Reproduce the headline trio walk-forward 2020-2023."""
    from src.ml.nyx_pipeline import NYXPipeline
    from src.assets.combined_portfolio import combine_trades
    btc = {tf: _load_btc(tf) for tf in ('15m', '1h', '4h', '1d')}
    eth = {tf: _load_eth(tf) for tf in ('15m', '1h', '4h', '1d')}
    sol = {tf: _load_sol(tf) for tf in ('15m', '1h', '4h', '1d')}
    btc_f = {tf: pd.read_parquet(FEAT_DIR / f'BTCUSDT_features_{tf}.parquet')
             for tf in ('15m', '1h', '4h', '1d')}
    eth_f = {tf: pd.read_parquet(FEAT_DIR / f'ETHUSDT_features_{tf}.parquet')
             for tf in ('15m', '1h', '4h', '1d')}
    sol_f = {tf: pd.read_parquet(FEAT_DIR / f'SOLUSDT_features_{tf}.parquet')
             for tf in ('15m', '1h', '4h', '1d')}

    pipe = NYXPipeline()
    all_trades = []
    for year in WALK_YEARS:
        per_asset = {}
        for nm, m, f in (('BTCUSDT', btc, btc_f),
                          ('ETHUSDT', eth, eth_f),
                          ('SOLUSDT', sol, sol_f)):
            if nm == 'SOLUSDT' and year < 2021:
                continue
            r = pipe.run(m, f,
                         train_end=f'{year-1}-12-31',
                         test_start=f'{year}-01-01',
                         test_end=f'{year}-12-31')
            for t in r['trades']:
                t.setdefault('regime', 'bear' if year == 2022 else 'bull')
                t.setdefault('asset', nm)
            per_asset[nm] = r['trades']
        all_trades.extend(combine_trades(per_asset))
    return {
        'trades': all_trades,
        'btc_15m': btc['15m'],
        'eth_15m': eth['15m'],
        'sol_15m': sol['15m'],
        'btc_mtf': btc, 'btc_f': btc_f,
        'eth_mtf': eth, 'eth_f': eth_f,
        'sol_mtf': sol, 'sol_f': sol_f,
    }


def main() -> int:
    from src.ml.reality_check import (
        pure_oos_sol, post_only_filter, taker_fees_run,
        daily_equity_sharpe, block_bootstrap_long, buy_and_hold_benchmark,
    )
    from src.ml.bootstrap import standard_bootstrap

    print("=== REALITY CHECK — 6 corrections ===\n")
    wf = _walk_forward_trio()
    bar_by_asset = {
        'BTCUSDT': wf['btc_15m'],
        'ETHUSDT': wf['eth_15m'],
        'SOLUSDT': wf['sol_15m'],
    }
    trades = wf['trades']
    print(f"Headline walk-forward: {len(trades)} pooled trades 2020-2023\n")

    out = {'n_trades_walk_forward': len(trades)}

    # ----- 1. SOL pure OOS 2024-2026 -----
    print("[1] SOL pure OOS 2024 → 2026 …")
    sol_oos = pure_oos_sol(wf['sol_mtf'], wf['sol_f'])
    out['sol_pure_oos_2024_2026'] = {
        k: sol_oos.get(k) for k in (
            'n_candidates', 'n_trades', 'sharpe', 'total_pnl_dollars',
            'max_drawdown_pct', 'win_rate', 'execution_reject_rate',
            'bear_dial_activation_rate',
        )
    }
    print(f"    n_trades={sol_oos['n_trades']}  "
          f"PnL=${sol_oos['total_pnl_dollars']:+.0f}  "
          f"Sharpe={sol_oos['sharpe']:+.2f}  "
          f"DD={sol_oos['max_drawdown_pct']:.1%}\n")

    # ----- 2. Post-only filter -----
    print("[2] Post-only filter …")
    po = post_only_filter(trades, bar_by_asset, max_wait_bars=3, sub_market_bps=5)
    out['post_only_filter'] = {k: v for k, v in po.items()
                                if k != 'filled_trades'}
    print(f"    miss_rate={po['miss_rate']:.1%}  "
          f"missed={po['missed_count']}/{po['n_trades_before']}  "
          f"PnL before=${po['total_pnl_before']:+.0f} → "
          f"after=${po['total_pnl_after']:+.0f}\n")

    # ----- 3. Taker fees walk-forward -----
    print("[3] Taker fees walk-forward (2023 OOS each asset) …")
    taker_per_asset = {}
    for nm, mt, ft in (('BTCUSDT', wf['btc_mtf'], wf['btc_f']),
                        ('ETHUSDT', wf['eth_mtf'], wf['eth_f']),
                        ('SOLUSDT', wf['sol_mtf'], wf['sol_f'])):
        r = taker_fees_run(mt, ft,
                           train_end='2022-12-31',
                           test_start='2023-01-01',
                           test_end='2023-12-31')
        taker_per_asset[nm] = {
            'n_trades':          r.get('n_trades'),
            'total_pnl_dollars': r.get('total_pnl_dollars'),
            'sharpe':            r.get('sharpe'),
            'max_drawdown_pct':  r.get('max_drawdown_pct'),
        }
        print(f"    {nm} taker 2023:  n={r['n_trades']:>3}  "
              f"PnL=${r['total_pnl_dollars']:+.0f}  Sh={r['sharpe']:+.2f}")
    out['taker_fees_2023_per_asset'] = taker_per_asset
    print()

    # ----- 4. Daily-equity Sharpe -----
    print("[4] Daily-equity Sharpe …")
    d = daily_equity_sharpe(trades, initial_capital=INITIAL,
                             annualization_days=365)
    pnls = np.array([t['net_pnl'] for t in trades])
    per_trade = float(pnls.mean() / pnls.std(ddof=0)
                      * np.sqrt(min(len(pnls), 252))) if pnls.std(ddof=0) > 0 else 0.0
    out['sharpe_comparison'] = {
        'per_trade_sqrt_n':  per_trade,
        'daily_equity':      d['sharpe_daily'],
        'mean_daily_return': d['mean_daily_return'],
        'std_daily_return':  d['std_daily_return'],
        'n_days':            d['n_days'],
    }
    print(f"    Sharpe per-trade (sqrt N): {per_trade:+.2f}")
    print(f"    Sharpe daily-equity:       {d['sharpe_daily']:+.2f}")
    print(f"    n_days: {d['n_days']}\n")

    # ----- 5. Block bootstrap n=30 -----
    print("[5] Block bootstrap n=30 vs standard IID …")
    std = standard_bootstrap(trades, n_sims=1500)
    blk = block_bootstrap_long(trades, block_size=30, n_sims=1500)
    out['bootstrap_compare'] = {
        'standard_iid': {k: std[k] for k in (
            'prob_loss', 'median_return', 'sharpe_median',
            'return_p5', 'sharpe_p5', 'dd_p95',
        ) if k in std},
        'block_n30': {k: blk[k] for k in (
            'prob_loss', 'median_return', 'sharpe_median',
            'return_p5', 'sharpe_p5', 'dd_p95',
        ) if k in blk},
    }
    print(f"    standard:   probL={std['prob_loss']:.1%}  "
          f"p5_ret={std['return_p5']:+.2%}  p5_sh={std['sharpe_p5']:+.2f}")
    print(f"    block n=30: probL={blk['prob_loss']:.1%}  "
          f"p5_ret={blk['return_p5']:+.2%}  p5_sh={blk['sharpe_p5']:+.2f}\n")

    # ----- 6. Buy & hold benchmark -----
    print("[6] Buy & hold benchmark (equal-weight trio 2020-2023) …")
    bh = buy_and_hold_benchmark(
        bar_by_asset, test_start='2020-01-01', test_end='2023-12-31',
    )
    out['buy_and_hold'] = bh
    strategy_ret = float(pnls.sum() / INITIAL)
    out['strategy_vs_benchmark'] = {
        'strategy_cum_return': strategy_ret,
        'b_and_h_cum_return':  bh['portfolio_return'],
        'alpha':               strategy_ret - bh['portfolio_return'],
    }
    print(f"    BTC B&H : {bh['per_asset_return']['BTCUSDT']:+.1%}")
    print(f"    ETH B&H : {bh['per_asset_return']['ETHUSDT']:+.1%}")
    print(f"    SOL B&H : {bh['per_asset_return']['SOLUSDT']:+.1%}")
    print(f"    Equal-weight B&H portfolio: {bh['portfolio_return']:+.1%}")
    print(f"    Strategy cumulative      : {strategy_ret:+.1%}")
    print(f"    Alpha (strategy - B&H)   : {strategy_ret - bh['portfolio_return']:+.1%}\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"saved: {REPORT_PATH.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
