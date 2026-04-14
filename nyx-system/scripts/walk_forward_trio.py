"""
Walk-forward annualized — BTC + ETH + SOL trio.

Per-year OOS: train on all available data BEFORE Y, test on year Y.
Composition evolves with data availability:

  Y=2020  → BTC, ETH (SOL not yet — starts Aug 2020)
  Y=2021  → BTC, ETH, SOL (SOL training = ~5 months)
  Y=2022  → BTC, ETH, SOL  (full trio, all assets ≥ 1y training)
  Y=2023  → BTC, ETH, SOL  (full trio, all assets ≥ 2y training)

BTC + ETH raw data ends 2024-01-01 → 2024 OOS not possible without
refreshing those CSVs. Out-of-scope for this walk-forward.

Outputs:
  reports/walk_forward_trio.json        — structured per-year + aggregate
  reports/walk_forward_trio_summary.txt — human-readable console dump
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

DATA_DIR = HERE / 'data' / 'raw'
FEAT_DIR = HERE / 'data' / 'features'
BTC_MTF = DATA_DIR / 'mtf'
REPORT_PATH = HERE / 'reports' / 'walk_forward_trio.json'
SUMMARY_PATH = HERE / 'reports' / 'walk_forward_trio_summary.txt'

INITIAL_CAPITAL = 10_000.0
YEARS = (2020, 2021, 2022, 2023)

_SOL_CSV_BY_TF = {
    '15m': 'SOLUSDT_15minutes',
    '1h':  'SOLUSDT_1hour',
    '4h':  'SOLUSDT_4hours',
    '1d':  'SOLUSDT_1day',
}

# Asset first-available-training-year. Training must cover ≥ 6 months
# before the OOS year for a result to be considered valid.
_FIRST_TRAIN_YEAR = {
    'BTCUSDT': 2020,   # BTC 2019-09+ → 2020 OOS usable
    'ETHUSDT': 2020,   # ETH 2019-12+ → 2020 OOS usable (1 month training; tight)
    'SOLUSDT': 2021,   # SOL 2020-08+ → 2021 OOS first viable
}


# ---------- loaders ----------

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
    n = _SOL_CSV_BY_TF[tf]
    d = pd.read_csv(DATA_DIR / f'{n}.csv',
                    usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    d['datetime'] = pd.to_datetime(d['timestamp'], unit='ms')
    d = d.set_index('datetime').drop(columns='timestamp')
    for c in ('open', 'high', 'low', 'close'):
        d = d[d[c] > 0]
    return d


# ---------- walk-forward ----------

def _run_year(asset, year, mtf, feats):
    """OOS a single asset on the year. Returns result dict (or None)."""
    from src.ml.nyx_pipeline import NYXPipeline
    if year < _FIRST_TRAIN_YEAR[asset]:
        return None
    try:
        return NYXPipeline().run(
            mtf, feats,
            train_end=f'{year-1}-12-31',
            test_start=f'{year}-01-01',
            test_end=f'{year}-12-31',
        )
    except Exception as e:
        return {'error': repr(e)}


def _year_metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute annualized-style metrics for one year's pooled trades."""
    if not trades:
        return {
            'n_trades': 0, 'total_pnl_dollars': 0.0, 'return_pct': 0.0,
            'sharpe': 0.0, 'max_drawdown_pct': 0.0, 'win_rate': 0.0,
        }
    pnls = np.array([t['net_pnl'] for t in trades], dtype=float)
    wins = (pnls > 0).sum()
    eq = INITIAL_CAPITAL + np.cumsum(pnls)
    eq = np.insert(eq, 0, INITIAL_CAPITAL)
    ret = (eq[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
    # Per-trade Sharpe annualized to trades/year (same convention the
    # pipeline uses in NYXPipeline.run).
    if pnls.std(ddof=0) > 0:
        sharpe = float(pnls.mean() / pnls.std(ddof=0) * np.sqrt(252))
    else:
        sharpe = 0.0
    peak = eq[0]; mdd = 0.0
    for x in eq:
        peak = max(peak, x)
        dd = (peak - x) / peak if peak > 0 else 0.0
        mdd = max(mdd, dd)
    return {
        'n_trades': int(len(pnls)),
        'total_pnl_dollars': float(pnls.sum()),
        'return_pct': float(ret),
        'sharpe': sharpe,
        'max_drawdown_pct': float(mdd),
        'win_rate': float(wins / len(pnls)),
    }


def main() -> int:
    print("=== Walk-forward annualized — TRIO ===")
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

    per_year: Dict[int, Dict[str, Any]] = {}
    all_trades: List[Dict[str, Any]] = []

    for year in YEARS:
        print(f"\n→ {year}")
        regime = 'bear' if year == 2022 else 'bull'
        assets_this_year = {}
        year_trades_by_asset: Dict[str, List[Dict]] = {}

        for asset, mtf, feats in (('BTCUSDT', btc, btc_f),
                                    ('ETHUSDT', eth, eth_f),
                                    ('SOLUSDT', sol, sol_f)):
            r = _run_year(asset, year, mtf, feats)
            if r is None:
                print(f"  {asset}: skipped (no training data yet)")
                continue
            if 'error' in r:
                print(f"  {asset}: error — {r['error']}")
                continue
            tr = list(r.get('trades', []))
            for t in tr:
                t.setdefault('regime', regime)
                t.setdefault('asset', asset)
            year_trades_by_asset[asset] = tr
            assets_this_year[asset] = {
                'n_candidates':      r.get('n_candidates'),
                'n_trades':          r.get('n_trades'),
                'total_pnl_dollars': r.get('total_pnl_dollars'),
                'sharpe':            r.get('sharpe'),
                'max_drawdown_pct':  r.get('max_drawdown_pct'),
                'win_rate':          r.get('win_rate'),
            }
            print(f"  {asset}: n_trades={r.get('n_trades')}  "
                  f"pnl=${r.get('total_pnl_dollars'):+.0f}  "
                  f"Sharpe={r.get('sharpe'):+.2f}  "
                  f"DD={r.get('max_drawdown_pct'):.1%}")

        combined_year = combine_trades(year_trades_by_asset)
        metrics_year = _year_metrics(combined_year)
        per_year[year] = {
            'composition': sorted(assets_this_year.keys()),
            'per_asset':   assets_this_year,
            'combined':    metrics_year,
        }
        all_trades.extend(combined_year)
        print(f"  TRIO {year}: n_trades={metrics_year['n_trades']}  "
              f"pnl=${metrics_year['total_pnl_dollars']:+.0f}  "
              f"Sharpe={metrics_year['sharpe']:+.2f}  "
              f"DD={metrics_year['max_drawdown_pct']:.1%}  "
              f"WR={metrics_year['win_rate']:.1%}")

    # -----------------------------------------------------------------
    # Annualized aggregates
    # -----------------------------------------------------------------
    equity = INITIAL_CAPITAL
    year_returns: List[float] = []
    for y in YEARS:
        ret = per_year[y]['combined']['return_pct']
        year_returns.append(ret)
        equity *= (1 + ret)
    n_years = len(YEARS)
    cagr = (equity / INITIAL_CAPITAL) ** (1.0 / n_years) - 1.0
    mean_yearly_return = float(np.mean(year_returns))
    mean_yearly_sharpe = float(np.mean([per_year[y]['combined']['sharpe']
                                         for y in YEARS]))
    max_yearly_dd = max(per_year[y]['combined']['max_drawdown_pct']
                         for y in YEARS)

    # Pooled Monte Carlo + Bootstrap across every year.
    from src.ml.monte_carlo import trade_shuffle_mc
    from src.ml.bootstrap import full_bootstrap_report
    mc = {k: v for k, v in trade_shuffle_mc(all_trades, n_sims=2000).items()
          if k not in ('returns', 'max_dds')}
    bs = full_bootstrap_report(all_trades, n_sims=1500)['institutional_summary']

    report = {
        'years':              list(YEARS),
        'initial_capital':    INITIAL_CAPITAL,
        'per_year':           per_year,
        'annualized': {
            'years_counted':         n_years,
            'final_equity':          float(equity),
            'cagr':                  float(cagr),
            'mean_yearly_return':    mean_yearly_return,
            'mean_yearly_sharpe':    mean_yearly_sharpe,
            'max_yearly_drawdown':   float(max_yearly_dd),
            'years_with_positive_return': int(sum(r > 0 for r in year_returns)),
        },
        'pooled_monte_carlo':  mc,
        'pooled_bootstrap':    bs,
        'n_total_trades':      len(all_trades),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))

    lines = []
    lines.append("=" * 70)
    lines.append("WALK-FORWARD TRIO — annualized summary")
    lines.append("=" * 70)
    lines.append(f"Years covered: {list(YEARS)}  (data limit: BTC+ETH stop 2024-01-01)")
    lines.append(f"Initial capital: ${INITIAL_CAPITAL:,.0f}")
    lines.append("")
    for y in YEARS:
        m = per_year[y]['combined']
        comp = ", ".join(per_year[y]['composition']) or "(none)"
        lines.append(
            f"  {y}  {comp:<30}  n={m['n_trades']:>3}  "
            f"pnl=${m['total_pnl_dollars']:+7.0f}  "
            f"ret={m['return_pct']:+6.2%}  Sh={m['sharpe']:+5.2f}  "
            f"DD={m['max_drawdown_pct']:5.1%}  WR={m['win_rate']:.1%}"
        )
    lines.append("")
    lines.append("Annualized:")
    lines.append(f"  final equity            ${equity:,.0f}  (start ${INITIAL_CAPITAL:,.0f})")
    lines.append(f"  CAGR                    {cagr:+.2%}")
    lines.append(f"  mean yearly return      {mean_yearly_return:+.2%}")
    lines.append(f"  mean yearly Sharpe      {mean_yearly_sharpe:+.2f}")
    lines.append(f"  max single-year DD      {max_yearly_dd:.1%}")
    lines.append(f"  years positive          {sum(r > 0 for r in year_returns)}/{n_years}")
    lines.append("")
    lines.append("Pooled trades stress (all years):")
    lines.append(f"  total trades            {len(all_trades)}")
    lines.append(f"  MC median return        {mc['median_return']:+.2%}")
    lines.append(f"  MC % profitable sims    {mc['pct_profitable']:.1%}")
    lines.append(f"  BS median Sharpe        {bs['median_sharpe']:.2f}")
    lines.append(f"  BS p5 return            {bs['p5_return']:+.2%}")
    lines.append(f"  BS p5 Sharpe            {bs['p5_sharpe']:.2f}")
    lines.append(f"  BS prob_loss            {bs['prob_loss']:.1%}")
    SUMMARY_PATH.write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"\nsaved: {REPORT_PATH.relative_to(HERE)}")
    print(f"saved: {SUMMARY_PATH.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
