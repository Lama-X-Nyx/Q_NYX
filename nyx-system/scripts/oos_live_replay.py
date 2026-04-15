"""
OOS Live Replay — run NYXLiveDecider bar-by-bar on the 2023 test window.

Purpose: produce HONEST out-of-sample numbers for the LIVE inference
engine (NYXLiveDecider + MTFFeatureStack + bear-dial wire), NOT the
batch NYXPipeline. This is the measurement that matches what a real
live deployment would produce.

Methodology:
  1. For each symbol with a trained model (ETH, SOL):
     a. Build NYXLiveDecider(symbol, models/<SYMBOL>/)
     b. Seed with 2022-07-01..2022-12-31 (6 months) so buffers warm
        (EMA-200 + full 1d block converge)
     c. Reset state counters (cooldown + daily + bars_seen past warmup)
  2. Replay 2023-01-01..2023-12-31 bar-by-bar. Every bar calls
     `decider.on_15m_bar(bar)`. Emitted signals (direction != 0) are
     recorded with timestamp.
  3. For each emitted signal, compute forward outcome via the same
     TP/SL/TIME exit logic NYXPipeline uses (tp_mult=1.5, sl_mult=1.0,
     max_bars=50, fee_rate=0.0002 maker, slippage_rate=0.0001).
  4. Aggregate metrics: n_trades, win_rate, total_return_pct, Sharpe
     (per-trade), max_drawdown_pct, bear-dial activation rate.

Output: `reports/OOS_live_replay_2023.json`.

Rule #7: this complements `validate_abc_via_hubspoke.py` (which
wraps NYXPipeline) — here we DO the live inference path.
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

from src.ml.jesse_features import _atr
from src.ml.nyx_live_decider import NYXLiveDecider


# -----------------------------------------------------------------------------
# Data loaders (mirror tests/conftest.py)
# -----------------------------------------------------------------------------
DATA_DIR = ROOT / 'data' / 'raw'
MODELS_DIR = ROOT / 'models'
REPORTS_DIR = ROOT / 'reports'


def _load_eth_15m() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / 'ETHUSDT_15m.csv')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    for c in ('open', 'high', 'low', 'close'):
        df = df[df[c] > 0]
    return df


def _load_sol_15m() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_DIR / 'SOLUSDT_15minutes.csv',
        usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
    )
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('datetime').drop(columns='timestamp')
    for c in ('open', 'high', 'low', 'close'):
        df = df[df[c] > 0]
    return df


_LOADERS = {
    'ETHUSDT': _load_eth_15m,
    'SOLUSDT': _load_sol_15m,
}


# -----------------------------------------------------------------------------
# Forward outcome — match NYXPipeline exit logic exactly
# -----------------------------------------------------------------------------
TP_MULT = 1.5
SL_MULT = 1.0
MAX_BARS = 50
FEE_RATE = 0.0002      # maker
SLIPPAGE_RATE = 0.0001


def _forward_outcome(
    df_15m: pd.DataFrame, entry_idx: int, direction: int,
) -> Dict[str, Any]:
    """Replicates NYXPipeline._generate_candidates exit logic.

    Returns dict with entry_price, exit_price, reason, gross, fee, net.
    Net PnL is expressed in PRICE units (dollars/coin) — per-unit.
    """
    close = df_15m['close'].values.astype(float)
    high = df_15m['high'].values.astype(float)
    low = df_15m['low'].values.astype(float)
    n = len(close)

    atr = np.nan_to_num(
        _atr(high, low, close, 14), nan=0.0,
    )
    if entry_idx >= n - 1 or atr[entry_idx] <= 0:
        return {'valid': False}

    entry_p = close[entry_idx] * (1 + direction * SLIPPAGE_RATE)
    tp = entry_p + direction * TP_MULT * atr[entry_idx]
    sl = entry_p - direction * SL_MULT * atr[entry_idx]
    end_j = min(entry_idx + MAX_BARS + 1, n)
    fut_h = high[entry_idx + 1: end_j]
    fut_l = low[entry_idx + 1: end_j]

    if direction == 1:
        tp_hits = np.where(fut_h >= tp)[0]
        sl_hits = np.where(fut_l <= sl)[0]
    else:
        tp_hits = np.where(fut_l <= tp)[0]
        sl_hits = np.where(fut_h >= sl)[0]

    tp_bar = tp_hits[0] if len(tp_hits) > 0 else MAX_BARS + 1
    sl_bar = sl_hits[0] if len(sl_hits) > 0 else MAX_BARS + 1

    if tp_bar <= sl_bar and tp_bar < MAX_BARS:
        exit_p = tp * (1 - direction * SLIPPAGE_RATE)
        gross = direction * (exit_p - entry_p)
        reason = 'TP'
    elif sl_bar < tp_bar and sl_bar < MAX_BARS:
        exit_p = sl * (1 - direction * SLIPPAGE_RATE)
        gross = direction * (exit_p - entry_p)
        reason = 'SL'
    else:
        exit_p = close[min(entry_idx + MAX_BARS, n - 1)] * (
            1 - direction * SLIPPAGE_RATE
        )
        gross = direction * (exit_p - entry_p)
        reason = 'TIME'

    fee = entry_p * FEE_RATE + abs(exit_p) * FEE_RATE
    net = gross - fee
    return {
        'valid':       True,
        'entry_price': float(entry_p),
        'exit_price':  float(exit_p),
        'reason':      reason,
        'gross':       float(gross),
        'fee':         float(fee),
        'net':         float(net),
        'pct_return':  float(net / entry_p),
    }


# -----------------------------------------------------------------------------
# Core replay
# -----------------------------------------------------------------------------
def replay_symbol(
    symbol: str,
    seed_start: str = '2022-07-01',
    test_start: str = '2023-01-01',
    test_end: str = '2023-12-31',
) -> Dict[str, Any]:
    print(f'[{symbol}] loading 15m OHLCV...', flush=True)
    df15 = _LOADERS[symbol]()

    artefact_dir = MODELS_DIR / symbol
    if not artefact_dir.is_dir():
        return {'symbol': symbol, 'error': f'missing {artefact_dir}'}

    print(f'[{symbol}] building NYXLiveDecider...', flush=True)
    decider = NYXLiveDecider(
        symbol=symbol,
        artifact_dir=artefact_dir,
    )

    # Seed
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

    # Reset state (counter past warmup, no cooldown/daily bleed)
    decider._bars_seen = decider._warmup_bars + 1
    decider._last_trade_bar_idx = -10 ** 9
    decider._daily_counts = {}

    # Replay
    test_slice = df15.loc[test_start:test_end]
    print(
        f'[{symbol}] replaying {len(test_slice)} bars '
        f'({test_start} -> {test_end})...', flush=True,
    )

    signals_emitted: List[Dict[str, Any]] = []
    n_bear_active = 0
    n_hard_gate_pass = 0

    # We need the test_slice index to be able to look up forward
    # outcome given a timestamp. Use positional index lookup.
    ts_index = test_slice.index
    ts_to_pos = {ts: i for i, ts in enumerate(ts_index)}

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
        # "hard gate passed" = any non-flat ML evaluation happened
        # Proxy: _last_effective_threshold is set when feature vector
        # was built, i.e. past hard gate. We don't expose a bool
        # directly — use the signal.direction as a proxy for final.
        if sig.direction != 0:
            n_hard_gate_pass += 1
            pos = ts_to_pos.get(ts)
            if pos is None:
                continue
            outcome = _forward_outcome(test_slice, pos, int(sig.direction))
            if not outcome.get('valid'):
                continue
            signals_emitted.append({
                'timestamp':    ts.isoformat(),
                'direction':    int(sig.direction),
                'conviction':   float(sig.conviction),
                'bear_active':  bool(decider._last_bear_active),
                'eff_threshold': float(decider._last_effective_threshold),
                **outcome,
            })

    # Metrics
    n_trades = len(signals_emitted)
    if n_trades == 0:
        return {
            'symbol':        symbol,
            'n_trades':      0,
            'note':          'no actionable signals emitted',
            'n_bear_active': n_bear_active,
            'seed_range':    f'{seed_start}..{test_start}',
            'test_range':    f'{test_start}..{test_end}',
        }

    nets = np.array([s['net'] for s in signals_emitted])
    pcts = np.array([s['pct_return'] for s in signals_emitted])
    wins = int((nets > 0).sum())
    losses = int((nets < 0).sum())
    win_rate = wins / n_trades
    gross_wins = float(nets[nets > 0].sum())
    gross_losses = float(abs(nets[nets <= 0].sum()))
    profit_factor = (
        gross_wins / gross_losses if gross_losses > 0 else float('inf')
    )

    # Equity curve (per-unit PnL, normalized to pct returns)
    equity = np.concatenate([[1.0], np.cumprod(1.0 + pcts)])
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    max_dd = float(dd.max())

    # Sharpe per-trade (not annualized here — we note the count)
    sharpe_per_trade = (
        float(pcts.mean() / max(pcts.std(), 1e-12))
        if len(pcts) > 1 else 0.0
    )

    # Direction distribution
    n_long = sum(1 for s in signals_emitted if s['direction'] > 0)
    n_short = n_trades - n_long

    # Exit reason breakdown
    exits = {'TP': 0, 'SL': 0, 'TIME': 0}
    for s in signals_emitted:
        exits[s['reason']] = exits.get(s['reason'], 0) + 1

    return {
        'symbol':                symbol,
        'seed_range':            f'{seed_start}..{test_start}',
        'test_range':            f'{test_start}..{test_end}',
        'n_test_bars':           int(len(test_slice)),
        'n_trades':              n_trades,
        'n_long':                n_long,
        'n_short':               n_short,
        'n_bear_active_bars':    int(n_bear_active),
        'bear_active_rate':      (
            float(n_bear_active / max(len(test_slice), 1))
        ),
        'n_bear_active_trades':  sum(
            1 for s in signals_emitted if s['bear_active']
        ),
        'win_rate':              float(win_rate),
        'wins':                  wins,
        'losses':                losses,
        'profit_factor':         profit_factor,
        'total_return_pct':      float(equity[-1] - 1.0),
        'max_drawdown_pct':      max_dd,
        'sharpe_per_trade':      sharpe_per_trade,
        'exit_breakdown':        exits,
        'avg_pct_return':        float(pcts.mean()),
        'median_pct_return':     float(np.median(pcts)),
        'trades':                signals_emitted,
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
            f'total_return={res.get("total_return_pct", "?"):.4f}, '
            f'win_rate={res.get("win_rate", 0.0):.2%}, '
            f'bear_activation_rate={res.get("bear_active_rate", 0.0):.2%}',
            flush=True,
        )

    out = REPORTS_DIR / 'OOS_live_replay_2023.json'
    with out.open('w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nwrote {out}')

    # Trio aggregate
    valid = [r for r in results.values() if r.get('n_trades', 0) > 0]
    if valid:
        n_total = sum(r['n_trades'] for r in valid)
        ret_avg = float(np.mean([r['total_return_pct'] for r in valid]))
        print(f'\nTrio aggregate: n_total={n_total}, '
              f'mean_return_per_asset={ret_avg:.4f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
