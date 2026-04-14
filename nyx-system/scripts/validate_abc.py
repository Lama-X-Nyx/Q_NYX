"""
A/B/C validation — BTC, BTC+ETH, BTC+ETH+SOL.

Runs OOS walk-forward (2022, 2023) + Monte Carlo + Bootstrap for:
  A : BTC only
  B : BTC ∪ ETH
  C : BTC ∪ ETH ∪ SOL

Also reports the 3 pair combinations and SOL standalone for the
full A/B/C doc.

Writes reports/AB_C_trio.json.
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
REPORT_PATH = HERE / 'reports' / 'ABC_trio_validation.json'

YEARS = (2022, 2023)

_SOL_CSV_BY_TF = {
    '15m': 'SOLUSDT_15minutes',
    '1h':  'SOLUSDT_1hour',
    '4h':  'SOLUSDT_4hours',
    '1d':  'SOLUSDT_1day',
}


# ---- loaders ---------------------------------------------------------------

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


def _load_sol(tf: str) -> pd.DataFrame:
    name = _SOL_CSV_BY_TF[tf]
    df = pd.read_csv(DATA_DIR / f'{name}.csv',
                     usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('datetime').drop(columns='timestamp')
    for col in ('open', 'high', 'low', 'close'):
        df = df[df[col] > 0]
    return df


def _run_year(mtf_data, mtf_features, year: int):
    from src.ml.nyx_pipeline import NYXPipeline
    return NYXPipeline().run(
        mtf_data, mtf_features,
        train_end=f'{year-1}-12-31',
        test_start=f'{year}-01-01',
        test_end=f'{year}-12-31',
    )


def _strip(r: dict) -> dict:
    return {k: v for k, v in r.items() if k not in ('trades', 'equity_curve')}


def _mc(trades):
    from src.ml.monte_carlo import trade_shuffle_mc
    r = trade_shuffle_mc(trades, n_sims=2000)
    return {k: v for k, v in r.items() if k not in ('returns', 'max_dds')}


def _bs(trades):
    from src.ml.bootstrap import full_bootstrap_report
    r = full_bootstrap_report(trades, n_sims=1500)
    return r['institutional_summary']


# ---- main ------------------------------------------------------------------

def main() -> int:
    from src.assets.combined_portfolio import combine_trades

    print("=== A/B/C validation ===")

    # PERMANENT RULE: every asset passed through NYXPipeline must carry
    # the 4 timeframes (15m + 1h + 4h + 1d).
    btc_mtf = {tf: _load_btc(tf) for tf in ('15m', '1h', '4h', '1d')}
    eth_mtf = {tf: _load_eth(tf) for tf in ('15m', '1h', '4h', '1d')}
    sol_mtf = {tf: _load_sol(tf) for tf in ('15m', '1h', '4h', '1d')}

    btc_feats = {tf: pd.read_parquet(FEAT_DIR / f'BTCUSDT_features_{tf}.parquet')
                 for tf in ('15m', '1h', '4h', '1d')}
    eth_feats = {tf: pd.read_parquet(FEAT_DIR / f'ETHUSDT_features_{tf}.parquet')
                 for tf in ('15m', '1h', '4h', '1d')}
    sol_feats = {tf: pd.read_parquet(FEAT_DIR / f'SOLUSDT_features_{tf}.parquet')
                 for tf in ('15m', '1h', '4h', '1d')}

    # --- harvest trades per year & per asset ---
    print("→ OOS walk-forward per asset × year")
    by_year_btc, by_year_eth, by_year_sol = {}, {}, {}
    btc_trades, eth_trades, sol_trades = [], [], []

    for y in YEARS:
        tag = 'bear' if y == 2022 else 'bull'
        rb = _run_year(btc_mtf, btc_feats, y)
        re = _run_year(eth_mtf, eth_feats, y)
        rs = _run_year(sol_mtf, sol_feats, y)
        for t in rb['trades']:
            t.setdefault('regime', tag)
        for t in re['trades']:
            t.setdefault('regime', tag)
        for t in rs['trades']:
            t.setdefault('regime', tag)
        by_year_btc[str(y)] = _strip(rb)
        by_year_eth[str(y)] = _strip(re)
        by_year_sol[str(y)] = _strip(rs)
        btc_trades.extend(rb['trades'])
        eth_trades.extend(re['trades'])
        sol_trades.extend(rs['trades'])

    # --- combined pools ---
    combined_btc_eth = combine_trades({'BTCUSDT': btc_trades, 'ETHUSDT': eth_trades})
    combined_btc_sol = combine_trades({'BTCUSDT': btc_trades, 'SOLUSDT': sol_trades})
    combined_eth_sol = combine_trades({'ETHUSDT': eth_trades, 'SOLUSDT': sol_trades})
    trio = combine_trades({
        'BTCUSDT': btc_trades, 'ETHUSDT': eth_trades, 'SOLUSDT': sol_trades,
    })

    # --- MC + BS for each pool ---
    print("→ Monte Carlo + Bootstrap for each portfolio")
    pools = {
        'A_BTC':              btc_trades,
        'single_ETH':         eth_trades,
        'single_SOL':         sol_trades,
        'B_BTC_ETH':          combined_btc_eth,
        'pair_BTC_SOL':       combined_btc_sol,
        'pair_ETH_SOL':       combined_eth_sol,
        'C_BTC_ETH_SOL':      trio,
    }

    validation = {}
    for name, trades in pools.items():
        validation[name] = {
            'n_trades':     len(trades),
            'monte_carlo':  _mc(trades),
            'bootstrap':    _bs(trades),
        }

    report = {
        'years': list(YEARS),
        'oos_by_year': {
            'BTCUSDT': by_year_btc,
            'ETHUSDT': by_year_eth,
            'SOLUSDT': by_year_sol,
        },
        'portfolios': validation,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))

    # --- console summary ---
    print()
    def _line(label, v):
        mc = v['monte_carlo']
        bs = v['bootstrap']
        print(f"  {label:<20}  n={v['n_trades']:>4}  "
              f"MC med={mc['median_return']:+7.2%} pct_prof={mc['pct_profitable']:6.1%}  "
              f"BS p5ret={bs['p5_return']:+7.2%} p5shp={bs['p5_sharpe']:5.2f} "
              f"probL={bs['prob_loss']:5.1%}")
    print("Portfolio             n_trades  MC_median   pct_prof  BS_p5ret  BS_p5shp  probLoss")
    _line("A  — BTC only",           validation['A_BTC'])
    _line("   — ETH only",           validation['single_ETH'])
    _line("   — SOL only",           validation['single_SOL'])
    _line("B  — BTC+ETH",            validation['B_BTC_ETH'])
    _line("   — BTC+SOL",            validation['pair_BTC_SOL'])
    _line("   — ETH+SOL",            validation['pair_ETH_SOL'])
    _line("C  — BTC+ETH+SOL",        validation['C_BTC_ETH_SOL'])
    print(f"\nreport saved: {REPORT_PATH.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
