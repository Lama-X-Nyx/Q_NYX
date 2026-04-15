"""
OOS Live Replay — run NYXLiveDecider bar-by-bar on the 2023 test window.

Purpose: produce HONEST out-of-sample numbers for the LIVE inference
engine (NYXLiveDecider + MTFFeatureStack + bear-dial wire), using
**NYXPipeline itself** for the forward TP/SL/TIME exit outcomes
(no duplication — Rule #7).

Methodology:
  1. For each symbol with a trained model (ETH, SOL):
     a. Run `NYXPipeline._generate_candidates(...)` on full MTF data
        to pre-compute outcomes for every hard-gate-passing bar.
        Index by `timestamp`.
     b. Build `NYXLiveDecider(symbol, models/<SYMBOL>/)`
     c. Seed 2022-07-01..2023-01-01 (6 months) so buffers warm.
     d. Reset state counters.
  2. Replay 2023 bar-by-bar. Every `decider.on_15m_bar(bar)`.
     For each emitted signal (direction != 0), look up the
     timestamp in NYXPipeline's candidate dict → reuse its
     pre-computed `outcome_net` (maker fees + slippage already
     accounted for, same exit logic as batch).
  3. Aggregate metrics: n_trades, win_rate, total_return_pct,
     Sharpe (per-trade), max_drawdown_pct, bear-dial activation rate.

Output: `reports/OOS_live_replay_2023.json`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.ml.nyx_live_decider import NYXLiveDecider
from src.ml.nyx_pipeline import NYXPipeline


# -----------------------------------------------------------------------------
# Data loaders (mirror tests/conftest.py)
# -----------------------------------------------------------------------------
DATA_DIR = ROOT / 'data' / 'raw'
FEAT_DIR = ROOT / 'data' / 'features'
MODELS_DIR = ROOT / 'models'
REPORTS_DIR = ROOT / 'reports'


def _load_eth(tf: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f'ETHUSDT_{tf}.csv')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    for c in ('open', 'high', 'low', 'close'):
        df = df[df[c] > 0]
    return df


_SOL_CSV_BY_TF = {
    '15m': 'SOLUSDT_15minutes',
    '1h':  'SOLUSDT_1hour',
    '4h':  'SOLUSDT_4hours',
    '1d':  'SOLUSDT_1day',
}


def _load_sol(tf: str) -> pd.DataFrame:
    df = pd.read_csv(
        DATA_DIR / f'{_SOL_CSV_BY_TF[tf]}.csv',
        usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
    )
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('datetime').drop(columns='timestamp')
    for c in ('open', 'high', 'low', 'close'):
        df = df[df[c] > 0]
    return df


def _load_mtf(symbol: str) -> Dict[str, pd.DataFrame]:
    loader = _load_eth if symbol == 'ETHUSDT' else _load_sol
    return {tf: loader(tf) for tf in ('15m', '1h', '4h', '1d')}


def _load_features(symbol: str) -> Dict[str, pd.DataFrame]:
    return {
        tf: pd.read_parquet(FEAT_DIR / f'{symbol}_features_{tf}.parquet')
        for tf in ('15m', '1h', '4h', '1d')
    }


# -----------------------------------------------------------------------------
# Outcome lookup via NYXPipeline._generate_candidates (no duplication).
# -----------------------------------------------------------------------------
def build_outcome_index(
    mtf_data: Dict[str, pd.DataFrame],
    mtf_features: Dict[str, pd.DataFrame],
) -> Dict[pd.Timestamp, Dict[str, Any]]:
    """Call NYXPipeline._generate_candidates to get the per-bar
    (hard-gate-passing) candidates — each already carries the
    pre-computed outcome_net via the SAME TP/SL/TIME logic the
    batch run uses. Index by timestamp for O(1) lookup."""
    pipe = NYXPipeline()
    # Build 1d + 1h context columns via the pipeline's own helpers.
    ctx_1d = pipe._build_1d_context(
        mtf_data.get('1d', pd.DataFrame()),
        mtf_features.get('1d', pd.DataFrame()),
    )
    ctx_1h = pipe._build_1h_context(
        mtf_data.get('1h', pd.DataFrame()),
        mtf_features.get('1h', pd.DataFrame()),
    )
    candidates = pipe._generate_candidates(
        df_15m=mtf_data['15m'],
        feat_15m=mtf_features['15m'],
        ctx_1d=ctx_1d,
        ctx_1h=ctx_1h,
        feat_1h=mtf_features.get('1h'),
        feat_1d=mtf_features.get('1d'),
        feat_4h=mtf_features.get('4h'),
    )
    return {pd.Timestamp(c['timestamp']): c for c in candidates}


# -----------------------------------------------------------------------------
# Core replay
# -----------------------------------------------------------------------------
def replay_symbol(
    symbol: str,
    seed_start: str = '2022-07-01',
    test_start: str = '2023-01-01',
    test_end: str = '2023-12-31',
) -> Dict[str, Any]:
    print(f'[{symbol}] loading MTF OHLCV + features...', flush=True)
    mtf_data = _load_mtf(symbol)
    mtf_features = _load_features(symbol)

    print(f'[{symbol}] pre-computing batch outcomes via '
          'NYXPipeline._generate_candidates...', flush=True)
    outcome_by_ts = build_outcome_index(mtf_data, mtf_features)

    artefact_dir = MODELS_DIR / symbol
    if not artefact_dir.is_dir():
        return {'symbol': symbol, 'error': f'missing {artefact_dir}'}

    print(f'[{symbol}] building NYXLiveDecider...', flush=True)
    decider = NYXLiveDecider(
        symbol=symbol,
        artifact_dir=artefact_dir,
    )

    # Seed
    df15 = mtf_data['15m']
    seed_slice = df15.loc[seed_start:test_start]
    print(
        f'[{symbol}] seeding {len(seed_slice)} bars '
        f'({seed_start} -> {test_start})...', flush=True,
    )
    for ts, row in seed_slice.iterrows():
        decider.on_15m_bar({
            'timestamp': ts.isoformat(),
            'open':   float(row['open']),
            'high':   float(row['high']),
            'low':    float(row['low']),
            'close':  float(row['close']),
            'volume': float(row['volume']),
        })
    decider._bars_seen = decider._warmup_bars + 1
    decider._last_trade_bar_idx = -10 ** 9
    decider._daily_counts = {}

    # Replay
    test_slice = df15.loc[test_start:test_end]
    print(
        f'[{symbol}] replaying {len(test_slice)} bars '
        f'({test_start} -> {test_end})...', flush=True,
    )

    trades_out: List[Dict[str, Any]] = []
    n_bear_active = 0
    n_live_without_batch_match = 0

    for ts, row in test_slice.iterrows():
        sig = decider.on_15m_bar({
            'timestamp': ts.isoformat(),
            'open':   float(row['open']),
            'high':   float(row['high']),
            'low':    float(row['low']),
            'close':  float(row['close']),
            'volume': float(row['volume']),
        })
        if decider._last_bear_active:
            n_bear_active += 1
        if sig.direction == 0:
            continue

        # Look up forward outcome from NYXPipeline's candidate dict.
        cand = outcome_by_ts.get(pd.Timestamp(ts))
        if cand is None:
            n_live_without_batch_match += 1
            continue

        # Sanity: direction should match (same hard gate).
        if int(cand['direction']) != int(sig.direction):
            n_live_without_batch_match += 1
            continue

        net = float(cand['outcome_net'])
        entry = float(cand['entry_price'])
        trades_out.append({
            'timestamp':    ts.isoformat(),
            'direction':    int(sig.direction),
            'conviction':   float(sig.conviction),
            'bear_active':  bool(decider._last_bear_active),
            'eff_threshold': float(decider._last_effective_threshold),
            'entry_price': entry,
            'reason':      str(cand['reason']),
            'net':         net,
            'pct_return':  float(net / entry) if entry > 0 else 0.0,
        })

    # Metrics
    n_trades = len(trades_out)
    if n_trades == 0:
        return {
            'symbol':                    symbol,
            'n_trades':                  0,
            'n_live_without_batch_match': n_live_without_batch_match,
            'note':                      'no actionable signals emitted',
            'seed_range':                f'{seed_start}..{test_start}',
            'test_range':                f'{test_start}..{test_end}',
        }

    nets = np.array([t['net'] for t in trades_out])
    pcts = np.array([t['pct_return'] for t in trades_out])
    wins = int((nets > 0).sum())
    losses = int((nets < 0).sum())
    win_rate = wins / n_trades
    gw = float(nets[nets > 0].sum())
    gl = float(abs(nets[nets <= 0].sum()))
    pf = gw / gl if gl > 0 else float('inf')

    equity = np.concatenate([[1.0], np.cumprod(1.0 + pcts)])
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    max_dd = float(dd.max())

    sharpe_pt = (
        float(pcts.mean() / max(pcts.std(), 1e-12))
        if len(pcts) > 1 else 0.0
    )

    n_long = sum(1 for t in trades_out if t['direction'] > 0)
    n_short = n_trades - n_long
    exits = {'TP': 0, 'SL': 0, 'TIME': 0}
    for t in trades_out:
        exits[t['reason']] = exits.get(t['reason'], 0) + 1

    return {
        'symbol':                    symbol,
        'seed_range':                f'{seed_start}..{test_start}',
        'test_range':                f'{test_start}..{test_end}',
        'n_test_bars':               int(len(test_slice)),
        'n_trades':                  n_trades,
        'n_long':                    n_long,
        'n_short':                   n_short,
        'n_bear_active_bars':        int(n_bear_active),
        'bear_active_rate':          (
            float(n_bear_active / max(len(test_slice), 1))
        ),
        'n_bear_active_trades':      sum(
            1 for t in trades_out if t['bear_active']
        ),
        'n_live_without_batch_match': n_live_without_batch_match,
        'win_rate':                  float(win_rate),
        'wins':                      wins,
        'losses':                    losses,
        'profit_factor':             pf,
        'total_return_pct':          float(equity[-1] - 1.0),
        'max_drawdown_pct':          max_dd,
        'sharpe_per_trade':          sharpe_pt,
        'exit_breakdown':            exits,
        'avg_pct_return':            float(pcts.mean()),
        'median_pct_return':         float(np.median(pcts)),
        'trades':                    trades_out,
    }


# -----------------------------------------------------------------------------
def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    symbols = ['ETHUSDT', 'SOLUSDT']
    results: Dict[str, Any] = {}

    for symbol in symbols:
        try:
            res = replay_symbol(symbol)
        except (FileNotFoundError, Exception) as e:
            res = {'symbol': symbol, 'error': repr(e)}
        results[symbol] = res
        print(
            f'[{symbol}] DONE — n_trades={res.get("n_trades", "?")}, '
            f'total_return={res.get("total_return_pct", 0.0):.4f}, '
            f'win_rate={res.get("win_rate", 0.0):.2%}, '
            f'bear_activation_rate='
            f'{res.get("bear_active_rate", 0.0):.2%}',
            flush=True,
        )

    out = REPORTS_DIR / 'OOS_live_replay_2023.json'
    with out.open('w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nwrote {out}')

    valid = [r for r in results.values() if r.get('n_trades', 0) > 0]
    if valid:
        n_total = sum(r['n_trades'] for r in valid)
        ret_avg = float(np.mean([r['total_return_pct'] for r in valid]))
        print(f'\nTrio aggregate: n_total={n_total}, '
              f'mean_return_per_asset={ret_avg:.4f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
