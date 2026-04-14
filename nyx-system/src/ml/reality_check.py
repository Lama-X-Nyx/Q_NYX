"""
Reality-check helpers for the trio walk-forward.

Each function corresponds to one of the 6 corrections listed in
docs/REALITY_CHECK.md (2026-04 update):

  1. pure_oos_sol     — train ≤ 2023, test 2024+ (SOL only data left)
  2. post_only_filter — drop trades that would NOT fill as maker
  3. taker_fees_run   — re-run pipeline with taker fee/slippage
  4. daily_equity_sharpe — Sharpe on the daily capital curve
  5. block_bootstrap_long — block bootstrap with block_size=30
  6. buy_and_hold_benchmark — equal-weight B&H of trio over the test window
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. SOL pure OOS — drives train through 2023 and tests on 2024-2026
# ---------------------------------------------------------------------------

def pure_oos_sol(
    sol_mtf: Dict[str, pd.DataFrame],
    sol_features: Dict[str, pd.DataFrame],
    train_end: str = "2023-12-31",
    test_start: str = "2024-01-01",
    test_end: str = "2026-04-13",
) -> Dict[str, Any]:
    """Train SOL pipeline through `train_end`, test on the post-cutoff window."""
    from src.ml.nyx_pipeline import NYXPipeline
    pipe = NYXPipeline()
    return pipe.run(
        sol_mtf, sol_features,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )


# ---------------------------------------------------------------------------
# 2. Post-only filter — drop trades that would NOT have filled as maker
# ---------------------------------------------------------------------------

def post_only_filter(
    trades: List[Dict[str, Any]],
    bar_data_by_asset: Dict[str, pd.DataFrame],
    max_wait_bars: int = 3,
    sub_market_bps: int = 5,
) -> Dict[str, Any]:
    """Simulate post-only fill on every trade.

    For each trade with `entry_time`, `direction`, `entry_price`:
      - Place a limit at entry_price ± sub_market_bps (sub-market for buy,
        super-market for sell).
      - Look at the next `max_wait_bars` bars; if price crosses the limit,
        FILLED as maker. Otherwise, TIMED_OUT (missed trade).

    Returns a dict with:
      filled_trades        — trades that would have filled (subset)
      missed_count         — number of TIMED_OUT trades
      miss_rate            — missed / total
      total_pnl_after      — sum(net_pnl) over filled trades only
      total_pnl_before     — sum(net_pnl) over all input trades
    """
    if not trades:
        return {'filled_trades': [], 'missed_count': 0, 'miss_rate': 0.0,
                'total_pnl_after': 0.0, 'total_pnl_before': 0.0,
                'n_trades_before': 0, 'n_trades_after': 0}

    offset = sub_market_bps / 10_000.0
    filled: List[Dict[str, Any]] = []
    missed = 0

    for tr in trades:
        ts = pd.Timestamp(tr.get('entry_time') or tr['timestamp'])
        asset = tr.get('asset', 'UNKNOWN')
        direction = int(tr.get('direction', 0))
        entry = float(tr['entry_price'])
        df = bar_data_by_asset.get(asset)
        if df is None or direction == 0:
            # Can't simulate without bar data → assume filled.
            filled.append(tr)
            continue

        # Limit price: sub-market for buy, super-market for sell.
        if direction > 0:
            limit = entry * (1 - offset)
        else:
            limit = entry * (1 + offset)

        # Find bars strictly AFTER the entry timestamp.
        try:
            idx = df.index.get_indexer([ts], method='nearest')[0]
        except Exception:
            filled.append(tr)
            continue
        if idx < 0:
            filled.append(tr)
            continue
        window = df.iloc[idx + 1: idx + 1 + max_wait_bars]
        if window.empty:
            filled.append(tr)
            continue

        if direction > 0:
            crossed = (window['low'] <= limit).any()
        else:
            crossed = (window['high'] >= limit).any()

        if crossed:
            filled.append(tr)
        else:
            missed += 1

    pnl_before = float(sum(t.get('net_pnl', 0) for t in trades))
    pnl_after = float(sum(t.get('net_pnl', 0) for t in filled))
    n = len(trades)
    return {
        'filled_trades':    filled,
        'missed_count':     int(missed),
        'miss_rate':        missed / n if n else 0.0,
        'total_pnl_before': pnl_before,
        'total_pnl_after':  pnl_after,
        'n_trades_before':  n,
        'n_trades_after':   len(filled),
    }


# ---------------------------------------------------------------------------
# 3. Taker fees run — same pipeline but with taker fee/slippage
# ---------------------------------------------------------------------------

def taker_fees_run(
    mtf: Dict[str, pd.DataFrame],
    features: Dict[str, pd.DataFrame],
    train_end: str,
    test_start: str,
    test_end: str,
) -> Dict[str, Any]:
    """Re-run pipeline with realistic taker fees + slippage."""
    from src.ml.nyx_pipeline import NYXPipeline
    pipe = NYXPipeline(
        fee_rate=0.0004,        # taker
        slippage_rate=0.0003,   # 3 bps
    )
    return pipe.run(mtf, features,
                     train_end=train_end,
                     test_start=test_start, test_end=test_end)


# ---------------------------------------------------------------------------
# 4. Daily-equity Sharpe — Sharpe computed on daily capital curve
# ---------------------------------------------------------------------------

def daily_equity_sharpe(
    trades: List[Dict[str, Any]],
    initial_capital: float = 10_000.0,
    annualization_days: int = 365,
) -> Dict[str, Any]:
    """Compute Sharpe on a daily-equity series, not per trade.

    Each trade's `net_pnl` is attributed to its `entry_time` day. Days
    without trades have zero PnL. The Sharpe ratio is then:

      mean(daily_return) / std(daily_return) × sqrt(annualization_days)

    (annualization_days = 365 for crypto-24/7, 252 for traditional.)
    """
    if not trades:
        return {'sharpe_daily': 0.0, 'mean_daily_return': 0.0,
                'std_daily_return': 0.0, 'n_days': 0}

    by_day: Dict[pd.Timestamp, float] = {}
    for t in trades:
        d = pd.Timestamp(t.get('entry_time') or t['timestamp']).normalize()
        by_day[d] = by_day.get(d, 0.0) + float(t.get('net_pnl', 0.0))

    # Build a daily series from earliest to latest day.
    all_days = pd.date_range(min(by_day), max(by_day), freq='D')
    pnl_series = np.array([by_day.get(d, 0.0) for d in all_days])

    # Equity curve and daily returns.
    equity = initial_capital + np.cumsum(pnl_series)
    equity = np.insert(equity, 0, initial_capital)
    daily_rets = np.diff(equity) / equity[:-1]
    if len(daily_rets) < 2 or daily_rets.std(ddof=0) <= 0:
        return {'sharpe_daily': 0.0, 'mean_daily_return': float(daily_rets.mean()) if len(daily_rets) else 0.0,
                'std_daily_return': 0.0, 'n_days': len(all_days)}
    mean_r = float(daily_rets.mean())
    std_r = float(daily_rets.std(ddof=0))
    sharpe = mean_r / std_r * np.sqrt(annualization_days)
    return {
        'sharpe_daily':       float(sharpe),
        'mean_daily_return':  mean_r,
        'std_daily_return':   std_r,
        'n_days':             int(len(all_days)),
    }


# ---------------------------------------------------------------------------
# 5. Block bootstrap n=30 — preserve regime persistence
# ---------------------------------------------------------------------------

def block_bootstrap_long(
    trades: List[Dict[str, Any]],
    block_size: int = 30,
    n_sims: int = 1500,
) -> Dict[str, Any]:
    """Block bootstrap with block_size=30 (vs 5 in the headline doc).

    Returns the conservative prob_loss / p5 / etc. expected to be larger
    than the standard IID bootstrap.
    """
    from src.ml.bootstrap import block_bootstrap as _bs
    r = _bs(trades, n_sims=n_sims, block_size=block_size)
    # Strip the heavy 'returns' arrays.
    return {k: v for k, v in r.items() if k != 'returns'}


# ---------------------------------------------------------------------------
# 6. Buy & hold benchmark — equal-weighted trio over the same window
# ---------------------------------------------------------------------------

def buy_and_hold_benchmark(
    bar_data_by_asset: Dict[str, pd.DataFrame],
    test_start: str,
    test_end: str,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Equal-weight buy & hold of the trio over [test_start, test_end].

    Returns the total return per asset and the equally-weighted portfolio
    return for the same window the strategy is tested on.
    """
    weights = weights or {a: 1.0 / len(bar_data_by_asset)
                          for a in bar_data_by_asset}

    per_asset: Dict[str, float] = {}
    for asset, df in bar_data_by_asset.items():
        slc = df.loc[test_start:test_end]
        if slc.empty:
            per_asset[asset] = 0.0
            continue
        first = float(slc['close'].iloc[0])
        last = float(slc['close'].iloc[-1])
        per_asset[asset] = (last - first) / first

    portfolio_return = sum(per_asset[a] * weights.get(a, 0.0) for a in per_asset)
    return {
        'per_asset_return':  {a: float(r) for a, r in per_asset.items()},
        'portfolio_return':  float(portfolio_return),
        'window':            f'{test_start}..{test_end}',
        'weights':           {a: float(w) for a, w in weights.items()},
    }
