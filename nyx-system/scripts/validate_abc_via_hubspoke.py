"""
A/B/C validation — re-run through `HubSpokeRunner` (task b.A).

Replicates `scripts/validate_abc.py` but pipes each asset through
`NYXPipelinePod` → `HubSpokeRunner` → `PortfolioAllocator` →
`PostOnlyPaperBroker`. Compares the resulting trade stream to the
direct-`NYXPipeline.run()` baseline so we can SEE the impact of :

  1. Post-only miss-rate (some trades don't fill as maker)
  2. Per-asset and per-cluster risk caps
  3. Portfolio-level `max_open_positions` throttle

Writes `reports/ABC_trio_via_hubspoke.json`.

HISTORICAL REPLAY only. For live real-time multi-TF inference, see
task (b.B).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

DATA_DIR = HERE / 'data' / 'raw'
FEAT_DIR = HERE / 'data' / 'features'
BTC_MTF = DATA_DIR / 'mtf'
REPORT_PATH = HERE / 'reports' / 'ABC_trio_via_hubspoke.json'

YEARS = (2022, 2023)

_SOL_CSV_BY_TF = {
    '15m': 'SOLUSDT_15minutes',
    '1h':  'SOLUSDT_1hour',
    '4h':  'SOLUSDT_4hours',
    '1d':  'SOLUSDT_1day',
}


# ---- loaders (match validate_abc.py for reproducibility) ------------------

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
                     usecols=['timestamp', 'open', 'high', 'low',
                              'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('datetime').drop(columns='timestamp')
    for col in ('open', 'high', 'low', 'close'):
        df = df[df[col] > 0]
    return df


# ---- core ------------------------------------------------------------------

def _build_pods(year_window: Tuple[str, str, str]):
    """Build three NYXPipelinePods for BTC, ETH, SOL covering the window.

    Returns (pods_list, mtf_data_by_symbol).
    """
    from src.assets.nyx_pipeline_pod import NYXPipelinePod
    train_end, test_start, test_end = year_window

    btc_mtf = {tf: _load_btc(tf) for tf in ('15m', '1h', '4h', '1d')}
    eth_mtf = {tf: _load_eth(tf) for tf in ('15m', '1h', '4h', '1d')}
    sol_mtf = {tf: _load_sol(tf) for tf in ('15m', '1h', '4h', '1d')}
    btc_f = {tf: pd.read_parquet(FEAT_DIR / f'BTCUSDT_features_{tf}.parquet')
             for tf in ('15m', '1h', '4h', '1d')}
    eth_f = {tf: pd.read_parquet(FEAT_DIR / f'ETHUSDT_features_{tf}.parquet')
             for tf in ('15m', '1h', '4h', '1d')}
    sol_f = {tf: pd.read_parquet(FEAT_DIR / f'SOLUSDT_features_{tf}.parquet')
             for tf in ('15m', '1h', '4h', '1d')}

    pods = [
        NYXPipelinePod('BTCUSDT', btc_mtf, btc_f,
                       train_end, test_start, test_end, cluster_group='majors'),
        NYXPipelinePod('ETHUSDT', eth_mtf, eth_f,
                       train_end, test_start, test_end, cluster_group='alts'),
        NYXPipelinePod('SOLUSDT', sol_mtf, sol_f,
                       train_end, test_start, test_end, cluster_group='alts'),
    ]
    return pods, {'BTCUSDT': btc_mtf, 'ETHUSDT': eth_mtf, 'SOLUSDT': sol_mtf}


def _replay_via_hubspoke(pods, mtf_by_symbol, tmp_storage: Path):
    """Feed every 15m bar in the test window through HubSpokeRunner.

    Returns runner (to harvest trades after) + list of approved-trade
    dicts intercepted at allocator output.
    """
    from src.assets.hub_spoke_runner import HubSpokeRunner

    runner = HubSpokeRunner(
        pods=pods,
        storage_root=tmp_storage,
        max_total_risk=0.04,
        max_asset_risk=0.015,
        max_cluster_risk={'alts': 0.02},
        max_open_positions=2,
    )

    approved_trades: List[Dict] = []

    try:
        # SPARSE iteration: only visit timestamps where at least one pod
        # has a pre-computed trade. This is the critical speed-up — full
        # 15m iteration over 2 years × 3 assets is ~70k bars which is
        # prohibitively slow and adds nothing since pods not having a
        # trade emit direction=0 (which the allocator skips anyway).
        interesting_ts = set()
        for p in pods:
            interesting_ts.update(p._precomputed_trade_timestamps)
        interesting_ts_sorted = sorted(interesting_ts)
        print(f"    sparse iteration: {len(interesting_ts_sorted)} timestamps",
              flush=True)

        for ts_str in interesting_ts_sorted:
            ts = pd.Timestamp(ts_str)
            bars = {}
            for sym, mtf in mtf_by_symbol.items():
                df15 = mtf['15m']
                if ts in df15.index:
                    row = df15.loc[ts]
                    bars[sym] = {
                        'timestamp': ts_str,
                        'open':   float(row['open']),
                        'high':   float(row['high']),
                        'low':    float(row['low']),
                        'close':  float(row['close']),
                        'volume': float(row['volume']),
                    }
            if not bars:
                continue
            trades = runner.on_bars(bars)
            for t in trades:
                approved_trades.append({
                    'symbol':        t.symbol,
                    'direction':     t.direction,
                    'final_risk':    t.final_risk,
                    'cluster_group': t.cluster_group,
                    'timestamp':     t.source_signal.timestamp,
                    'conviction':    t.source_signal.conviction,
                    'edge_net':      t.source_signal.expected_edge_net,
                })
    finally:
        runner.shutdown()

    return runner, approved_trades


def main() -> int:
    print("=== A/B/C via HubSpokeRunner (task b.A) ===\n")

    import tempfile
    storage = Path(tempfile.mkdtemp(prefix='hubspoke_abc_'))

    per_year = {}
    for year in YEARS:
        win = (f'{year-1}-12-31', f'{year}-01-01', f'{year}-12-31')
        print(f"→ Year {year}  ({win})")
        pods, mtf_by_sym = _build_pods(win)
        for p in pods:
            print(f"    {p.symbol}: {p.n_precomputed_trades} precomputed trades")

        _, approved = _replay_via_hubspoke(pods, mtf_by_sym, storage)
        per_year[str(year)] = {
            'precomputed_per_asset': {p.symbol: p.n_precomputed_trades
                                       for p in pods},
            'approved_trades':         approved,
            'n_approved':              len(approved),
            'approved_per_asset':      {
                sym: sum(1 for t in approved if t['symbol'] == sym)
                for sym in ('BTCUSDT', 'ETHUSDT', 'SOLUSDT')
            },
        }
        print(f"    → via HubSpoke: {len(approved)} approved trades\n")

    # Aggregate totals.
    total_precomputed = sum(sum(yr['precomputed_per_asset'].values())
                            for yr in per_year.values())
    total_approved = sum(yr['n_approved'] for yr in per_year.values())
    allocator_reduction = (
        1.0 - (total_approved / total_precomputed) if total_precomputed else 0.0
    )

    report = {
        'years':              list(YEARS),
        'per_year':           per_year,
        'aggregate': {
            'total_precomputed_trades': total_precomputed,
            'total_approved_trades':    total_approved,
            'allocator_reduction_pct':  allocator_reduction,
        },
        'note':                (
            'Trades precomputed by NYXPipeline.run() direct, then filtered '
            'through HubSpokeRunner + PortfolioAllocator. '
            'approved = the subset that would actually trade '
            'under max_total_risk=0.04, max_asset_risk=0.015, '
            'max_cluster_risk={alts: 0.02}, max_open_positions=2.'
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))

    print("=== SUMMARY ===")
    print(f"  precomputed total : {total_precomputed}")
    print(f"  approved total    : {total_approved}")
    print(f"  allocator cut     : {allocator_reduction:.1%}")
    for y, yr in per_year.items():
        print(f"  {y}:")
        print(f"    precomputed per asset : {yr['precomputed_per_asset']}")
        print(f"    approved per asset    : {yr['approved_per_asset']}")
    print(f"\nsaved: {REPORT_PATH.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
