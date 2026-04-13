"""
Walk-Forward Validation
========================
Out-of-sample validation framework for the NYX trading system.

Methodology:
  - Split data into train / test windows (expanding or rolling)
  - For each window:
      1. Pre-train HSMM via Baum-Welch EM on train data
      2. Run event-driven backtest on test data (out-of-sample)
      3. Record performance metrics
  - Aggregate metrics across all test windows

Usage:
    cd nyx-system
    python scripts/walk_forward_validation.py --data-dir data/btcusdt
    python scripts/walk_forward_validation.py --data-dir data/btcusdt --mode rolling --train-months 6 --test-months 3
"""

import sys
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.hsmm import SemiMarkovHMM


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_ohlcv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=['datetime'], index_col='datetime')
    df.columns = [c.lower() for c in df.columns]
    df.sort_index(inplace=True)
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    high, low, close = np.asarray(df['high'].values), np.asarray(df['low'].values), np.asarray(df['close'].values)
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - np.roll(close, 1)),
                               np.abs(low  - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    df['atr_14']  = pd.Series(tr, index=df.index).rolling(14).mean()
    df['atr_50']  = pd.Series(tr, index=df.index).rolling(50).mean()
    df['sma_20']  = df['close'].rolling(20).mean()
    df['sma_50']  = df['close'].rolling(50).mean()
    if 'volume' in df.columns:
        df['volume_ma20'] = df['volume'].rolling(20).mean()
    df.dropna(inplace=True)
    return df


# ---------------------------------------------------------------------------
# Simplified single-timeframe backtest (no MTF dependencies)
# ---------------------------------------------------------------------------

def run_simple_backtest(
    df: pd.DataFrame,
    hsmm: SemiMarkovHMM,
    window: int = 200,
    sl_atr: float = 2.0,
    tp_atr: float = 3.0,
) -> dict:
    """
    Simplified backtest using only HSMM regime signal.

    Entry:  Regime = Trend+ or Squeeze, no open position
    Exit:   SL = entry - 2×ATR(14),  TP = entry + 3×ATR(14)

    Returns dict with performance metrics.
    """
    MIN_BARS = 100
    trades  = []
    in_pos  = False
    entry   = sl = tp = 0.0

    for i in range(MIN_BARS, len(df)):
        bar = df.iloc[i]
        close_ = float(bar['close'])
        atr_   = float(bar['atr_14']) if not np.isnan(bar['atr_14']) else 0.0

        # Check for exit first
        if in_pos:
            if close_ <= sl:
                trades.append({'pnl': (close_ - entry) / entry, 'type': 'SL'})
                in_pos = False
            elif close_ >= tp:
                trades.append({'pnl': (close_ - entry) / entry, 'type': 'TP'})
                in_pos = False
            continue

        # Get regime from HSMM
        start = max(0, i - window)
        w = df.iloc[start:i]
        obs = [{'price': float(w['returns'].iloc[j]) if not np.isnan(w['returns'].iloc[j]) else 0.0,
                'atr':   float(w['atr_14'].iloc[j])  if not np.isnan(w['atr_14'].iloc[j])  else 0.0}
               for j in range(len(w))]
        if not obs:
            continue

        gamma = hsmm.forward_backward(obs)
        state_idx = int(np.argmax(gamma[-1]))
        state = hsmm.states[state_idx]

        if state in ('Trend+', 'Squeeze') and atr_ > 0:
            in_pos = True
            entry  = close_
            sl     = close_ - sl_atr * atr_
            tp     = close_ + tp_atr * atr_

    if not trades:
        return {
            'n_trades': 0, 'total_return': 0.0, 'win_rate': 0.0,
            'avg_trade': 0.0, 'max_drawdown': 0.0, 'sharpe': np.nan,
        }

    pnls = [t['pnl'] for t in trades]
    cumret = np.cumprod([1 + p for p in pnls])
    peak   = np.maximum.accumulate(cumret)
    dd     = (cumret - peak) / peak

    return {
        'n_trades':     len(pnls),
        'total_return': float(cumret[-1] - 1),
        'win_rate':     float(np.mean([p > 0 for p in pnls])),
        'avg_trade':    float(np.mean(pnls)),
        'max_drawdown': float(dd.min()),
        'sharpe':       float(np.mean(pnls) / np.std(pnls)) if np.std(pnls) > 0 else np.nan,
    }


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

def build_windows(
    index: pd.DatetimeIndex,
    train_months: int,
    test_months: int,
    mode: str,  # 'expanding' or 'rolling'
) -> list:
    """Build list of (train_start, train_end, test_start, test_end) tuples."""
    start = pd.Timestamp(index[0]).to_pydatetime()
    end   = pd.Timestamp(index[-1]).to_pydatetime()

    windows = []
    test_start = start + relativedelta(months=train_months)

    while test_start < end:
        test_end = min(test_start + relativedelta(months=test_months), end)
        if mode == 'expanding':
            train_start = start
        else:  # rolling
            train_start = test_start - relativedelta(months=train_months)
        windows.append((train_start, test_start, test_start, test_end))
        test_start += relativedelta(months=test_months)

    return windows


def run_walk_forward(
    df: pd.DataFrame,
    train_months: int,
    test_months: int,
    mode: str,
    em_iters: int,
) -> pd.DataFrame:
    """Run full walk-forward validation."""
    states = ['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution']
    windows = build_windows(pd.DatetimeIndex(df.index), train_months, test_months, mode)

    if not windows:
        print("ERROR: Not enough data to build train/test windows.")
        return pd.DataFrame()

    print(f"\nWalk-forward: {len(windows)} windows "
          f"({mode}, {train_months}M train, {test_months}M test)")
    print("-" * 70)

    results = []
    for i, (tr_start, tr_end, ts_start, ts_end) in enumerate(windows, 1):
        train_raw = df[(df.index >= tr_start) & (df.index <  tr_end)]
        test_raw  = df[(df.index >= ts_start) & (df.index <= ts_end)]

        if len(train_raw) < 100 or len(test_raw) < 20:
            print(f"  Window {i}: skipping (train={len(train_raw)}, test={len(test_raw)})")
            continue

        train_df = prepare_features(pd.DataFrame(train_raw))
        test_df  = prepare_features(pd.DataFrame(test_raw))

        # Pre-train HSMM
        hsmm = SemiMarkovHMM(states=states)
        if em_iters > 0 and len(train_df) >= 50:
            ll = hsmm.initialize_parameters_with_em(train_df, n_iter=em_iters, tol=1e-3)
            em_info = f"EM {len(ll)} iters, LL={ll[-1]:.0f}"
        else:
            hsmm.initialize_parameters(train_df)
            em_info = "heuristic init"

        # Out-of-sample backtest
        metrics = run_simple_backtest(test_df, hsmm)
        metrics.update({
            'window':     i,
            'train_start': tr_start.strftime('%Y-%m'),
            'test_period': f"{ts_start.strftime('%Y-%m')} → {ts_end.strftime('%Y-%m')}",
            'em_info':    em_info,
        })
        results.append(metrics)

        flag = '✓' if metrics['total_return'] > 0 else '✗'
        print(f"  W{i:02d} [{metrics['test_period']}] "
              f"ret={metrics['total_return']:+.1%}  "
              f"trades={metrics['n_trades']}  "
              f"win={metrics['win_rate']:.0%}  "
              f"dd={metrics['max_drawdown']:.1%}  {flag}")

    return pd.DataFrame(results)


def print_summary(results: pd.DataFrame):
    """Print aggregate statistics."""
    if results.empty:
        print("No results to summarise.")
        return

    print("\n" + "=" * 70)
    print("AGGREGATE OOS STATISTICS")
    print("=" * 70)

    wins     = (results['total_return'] > 0).sum()
    total    = len(results)
    avg_ret  = results['total_return'].mean()
    med_ret  = results['total_return'].median()
    best     = results['total_return'].max()
    worst    = results['total_return'].min()
    avg_dd   = results['max_drawdown'].mean()
    avg_trd  = results['n_trades'].mean()
    avg_win  = results['win_rate'].mean()
    sharpes  = results['sharpe'].dropna()

    print(f"  Windows:        {total}  ({wins} positive, {total-wins} negative)")
    print(f"  Avg OOS return: {avg_ret:+.2%}")
    print(f"  Median OOS ret: {med_ret:+.2%}")
    print(f"  Best window:    {best:+.2%}")
    print(f"  Worst window:   {worst:+.2%}")
    print(f"  Avg max DD:     {avg_dd:.2%}")
    print(f"  Avg trades/w:   {avg_trd:.1f}")
    print(f"  Avg win rate:   {avg_win:.0%}")
    if len(sharpes):
        print(f"  Avg trade SR:   {sharpes.mean():.2f}")

    verdict = 'VALID ✓' if avg_ret > 0 and wins / total >= 0.55 else 'NOT VALIDATED ✗'
    print(f"\n  OOS Verdict: {verdict}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Walk-forward OOS validation for NYX')
    parser.add_argument('--data-dir',     default='data/btcusdt',
                        help='Directory with CSV files')
    parser.add_argument('--timeframe',    default='4h',
                        help='Candle timeframe (default: 4h)')
    parser.add_argument('--mode',         default='expanding',
                        choices=['expanding', 'rolling'],
                        help='Walk-forward mode (default: expanding)')
    parser.add_argument('--train-months', type=int, default=6,
                        help='Training window in months (default: 6)')
    parser.add_argument('--test-months',  type=int, default=3,
                        help='Test window in months (default: 3)')
    parser.add_argument('--em-iters',     type=int, default=20,
                        help='EM iterations per window (0 = heuristic only)')
    parser.add_argument('--output',       default=None,
                        help='Save results CSV to this path')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tf = args.timeframe.replace('h', 'H').replace('m', 'M').replace('d', 'D')
    candidates = list(data_dir.glob(f'*{tf}*.csv')) + list(data_dir.glob(f'*{args.timeframe}*.csv'))
    if not candidates:
        print(f"ERROR: No CSV found for timeframe {args.timeframe} in {data_dir}")
        sys.exit(1)
    csv_path = candidates[0]
    print(f"Data:      {csv_path}")

    df_all = load_ohlcv(csv_path)
    df_all = prepare_features(df_all)
    print(f"Range:     {pd.DatetimeIndex(df_all.index)[0].date()} → {pd.DatetimeIndex(df_all.index)[-1].date()}  ({len(df_all)} bars)")

    results = run_walk_forward(
        df_all,
        train_months=args.train_months,
        test_months=args.test_months,
        mode=args.mode,
        em_iters=args.em_iters,
    )

    print_summary(results)

    if args.output and not results.empty:
        out = Path(args.output)
        results.to_csv(out, index=False)
        print(f"\nResults saved: {out}")


if __name__ == '__main__':
    main()
