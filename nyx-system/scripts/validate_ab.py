"""
A/B validation — BTC alone vs BTC + ETH combined portfolio.

Runs OOS walk-forward (2022, 2023), Monte Carlo (trade shuffle), and
Bootstrap (standard + block + regime) for:

  portfolio A: BTC only
  portfolio B: BTC ∪ ETH

Writes a single JSON snapshot to reports/AB_BTC_vs_BTC_ETH.json that the
A/B doc reads from.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

DATA_DIR = HERE / 'data' / 'raw'
FEAT_DIR = HERE / 'data' / 'features'
BTC_MTF = DATA_DIR / 'mtf'
REPORT_PATH = HERE / 'reports' / 'AB_BTC_vs_BTC_ETH.json'

YEARS = (2022, 2023)


def _load_eth(tf: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f'ETHUSDT_{tf}.csv')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    for col in ('open', 'high', 'low', 'close'):
        df = df[df[col] > 0]
    return df


def _load_btc(tf: str) -> pd.DataFrame:
    df = pd.read_csv(BTC_MTF / f'BTCUSDT_{tf}.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.set_index('datetime')


def _run_year(mtf_data, mtf_features, year: int):
    from src.ml.nyx_pipeline import NYXPipeline
    return NYXPipeline().run(
        mtf_data, mtf_features,
        train_end=f'{year-1}-12-31',
        test_start=f'{year}-01-01',
        test_end=f'{year}-12-31',
    )


def _strip_results(r: dict) -> dict:
    return {k: v for k, v in r.items() if k not in ('trades', 'equity_curve')}


def main() -> int:
    print("=== A/B validation: BTC vs BTC+ETH ===")

    btc_mtf = {tf: _load_btc(tf) for tf in ('15m', '1h', '1d')}
    eth_mtf = {tf: _load_eth(tf) for tf in ('15m', '1h', '1d')}
    btc_feats = {tf: pd.read_parquet(FEAT_DIR / f'BTCUSDT_features_{tf}.parquet')
                 for tf in ('15m', '1h', '1d')}
    eth_feats = {tf: pd.read_parquet(FEAT_DIR / f'ETHUSDT_features_{tf}.parquet')
                 for tf in ('15m', '1h', '1d')}

    # --- OOS ---
    print("→ OOS walk-forward")
    btc_by_year, eth_by_year = {}, {}
    btc_trades, eth_trades = [], []
    for y in YEARS:
        tag = 'bear' if y == 2022 else 'bull'
        rb = _run_year(btc_mtf, btc_feats, y)
        re = _run_year(eth_mtf, eth_feats, y)
        for t in rb['trades']:
            t.setdefault('regime', tag)
        for t in re['trades']:
            t.setdefault('regime', tag)
        btc_by_year[str(y)] = _strip_results(rb)
        eth_by_year[str(y)] = _strip_results(re)
        btc_trades.extend(rb['trades'])
        eth_trades.extend(re['trades'])

    # Combined trades
    from src.assets.combined_portfolio import combine_trades
    combined = combine_trades({'BTCUSDT': btc_trades, 'ETHUSDT': eth_trades})

    # --- Monte Carlo ---
    print("→ Monte Carlo (trade shuffle)")
    from src.ml.monte_carlo import trade_shuffle_mc
    mc_btc = {k: v for k, v in trade_shuffle_mc(btc_trades, n_sims=2000).items()
              if k not in ('returns', 'max_dds')}
    mc_eth = {k: v for k, v in trade_shuffle_mc(eth_trades, n_sims=2000).items()
              if k not in ('returns', 'max_dds')}
    mc_combined = {k: v for k, v in trade_shuffle_mc(combined, n_sims=2000).items()
                   if k not in ('returns', 'max_dds')}

    # --- Bootstrap ---
    print("→ Bootstrap (standard + block + regime)")
    from src.ml.bootstrap import full_bootstrap_report
    bs_btc = full_bootstrap_report(btc_trades, n_sims=1500)
    bs_eth = full_bootstrap_report(eth_trades, n_sims=1500)
    bs_combined = full_bootstrap_report(combined, n_sims=1500)

    # --- Assemble ---
    report = {
        'years': list(YEARS),
        'portfolio_A_BTC': {
            'oos_by_year':         btc_by_year,
            'n_trades_all_years':  len(btc_trades),
            'monte_carlo':         mc_btc,
            'bootstrap':           bs_btc['institutional_summary'],
        },
        'portfolio_B_BTC_ETH': {
            'oos_by_year': {
                'BTCUSDT': btc_by_year,
                'ETHUSDT': eth_by_year,
            },
            'n_trades_all_years':  len(combined),
            'monte_carlo':         mc_combined,
            'bootstrap':           bs_combined['institutional_summary'],
        },
        'portfolio_B_single_ETH': {
            'oos_by_year':         eth_by_year,
            'n_trades_all_years':  len(eth_trades),
            'monte_carlo':         mc_eth,
            'bootstrap':           bs_eth['institutional_summary'],
        },
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))

    # Summary line
    print()
    print("Portfolio A (BTC only):")
    print(f"  trades 2022+2023:   {len(btc_trades):>5}")
    print(f"  MC pct_profitable:  {mc_btc['pct_profitable']:.1%}")
    print(f"  MC median_return:   {mc_btc['median_return']:+.2%}")
    print(f"  BS prob_loss:       {bs_btc['institutional_summary']['prob_loss']:.1%}")
    print(f"  BS p5 return:       {bs_btc['institutional_summary']['p5_return']:+.2%}")
    print(f"  BS p5 sharpe:       {bs_btc['institutional_summary']['p5_sharpe']:+.2f}")
    print()
    print("Portfolio B (BTC + ETH):")
    print(f"  trades 2022+2023:   {len(combined):>5}")
    print(f"  MC pct_profitable:  {mc_combined['pct_profitable']:.1%}")
    print(f"  MC median_return:   {mc_combined['median_return']:+.2%}")
    print(f"  BS prob_loss:       {bs_combined['institutional_summary']['prob_loss']:.1%}")
    print(f"  BS p5 return:       {bs_combined['institutional_summary']['p5_return']:+.2%}")
    print(f"  BS p5 sharpe:       {bs_combined['institutional_summary']['p5_sharpe']:+.2f}")

    print(f"\nreport saved: {REPORT_PATH.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
