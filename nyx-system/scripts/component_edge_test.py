"""
Component Edge Test
===================
Validate that each system component has a positive edge BEFORE live deployment.

Tests:
  1. HSMM regime edge  — does the detected regime predict forward returns?
  2. SMC pattern edge  — do Order-Block / FVG patterns have directional edge?
  3. Combined signal   — does the full NYX signal outperform components alone?

Usage:
    cd nyx-system
    python scripts/component_edge_test.py --data-dir data/btcusdt --year 2023
    python scripts/component_edge_test.py --data-dir data/btcusdt --year 2023 --pretrain-year 2022
"""

import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.hsmm import SemiMarkovHMM
from src.agents.regime_agent import RegimeAgent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_ohlcv(csv_path: Path) -> pd.DataFrame:
    """Load a standard OHLCV CSV (datetime index)."""
    df = pd.read_csv(csv_path, parse_dates=['datetime'], index_col='datetime')
    df.columns = [c.lower() for c in df.columns]
    df.sort_index(inplace=True)
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add returns, ATR, SMAs required by HSMM."""
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


def label_regime_series(df: pd.DataFrame, use_em: bool = True,
                         train_df: "pd.DataFrame | None" = None) -> pd.Series:
    """
    Run HSMM regime detection across a DataFrame bar by bar.

    Returns a Series with state labels aligned to df.index.
    Uses a rolling window of 200 bars to stay look-ahead-free.
    If train_df is provided, warm-starts via EM on train_df first.
    """
    WINDOW = 200
    MIN_BARS = 100
    states = ['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution']
    hsmm = SemiMarkovHMM(states=states)

    if train_df is not None and len(train_df) >= MIN_BARS:
        print("  [EM] Pre-training on training data...", end=' ', flush=True)
        ll = hsmm.initialize_parameters_with_em(train_df, n_iter=20)
        print(f"done ({len(ll)} iters, final LL={ll[-1]:.1f})")

    labels = pd.Series(index=df.index, dtype=str)

    for i in range(MIN_BARS, len(df)):
        start = max(0, i - WINDOW)
        window = df.iloc[start:i]

        if train_df is None:
            hsmm.initialize_parameters(window)

        obs = [{'price': window['returns'].iloc[j],
                'atr':   window['atr_14'].iloc[j]}
               for j in range(len(window))]
        gamma = hsmm.forward_backward(obs)
        idx   = int(np.argmax(gamma[-1]))
        labels.iloc[i] = states[idx]

    return labels


# ---------------------------------------------------------------------------
# Test 1 — HSMM regime predictive edge
# ---------------------------------------------------------------------------

def test_hsmm_edge(df: pd.DataFrame, train_df: "pd.DataFrame | None" = None,
                   horizons: list = [1, 3, 5]) -> pd.DataFrame:
    """
    For each horizon h (bars), compute mean forward return per regime state.

    A state has positive edge if mean_return(Trend+, h) > 0 and
    mean_return(Trend-, h) < 0 consistently.
    """
    print("\n[1] HSMM Regime Edge Test")
    print("-" * 50)

    df_feat = prepare_features(df.copy())
    labels  = label_regime_series(df_feat, train_df=train_df)

    rows = []
    for h in horizons:
        fwd_ret = df_feat['close'].pct_change(h).shift(-h)
        for state in ['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution']:
            mask = labels == state
            if mask.sum() < 10:
                continue
            rets: pd.Series = fwd_ret[mask].dropna()
            rows.append({
                'state':     state,
                'horizon':   h,
                'n_bars':    len(rets),
                'mean_ret':  float(rets.mean()),
                'median':    float(rets.median()),
                'std':       float(rets.std()),
                'sharpe':    float(rets.mean() / rets.std()) if float(rets.std()) > 0 else np.nan,
                'win_rate':  float((rets > 0).mean()),
                'pct_bars':  float(mask.sum() / len(df_feat) * 100),
            })

    result = pd.DataFrame(rows)
    if result.empty:
        print("  WARNING: no labels generated (check data path)")
        return result

    for h in horizons:
        sub = result[result['horizon'] == h].set_index('state')
        print(f"\n  Horizon {h} bars:")
        print(sub[['n_bars', 'mean_ret', 'sharpe', 'win_rate', 'pct_bars']].to_string())

    # Edge validation
    h1 = result[result['horizon'] == horizons[0]]
    tp_s: pd.Series = h1[h1['state'] == 'Trend+']['mean_ret']
    tm_s: pd.Series = h1[h1['state'] == 'Trend-']['mean_ret']
    tp = np.asarray(tp_s.values)
    tm = np.asarray(tm_s.values)
    if len(tp) and len(tm):
        print(f"\n  Edge check (h={horizons[0]}):")
        print(f"    Trend+  mean_ret = {tp[0]:.4%}  {'✓' if tp[0] > 0 else '✗'}")
        print(f"    Trend-  mean_ret = {tm[0]:.4%}  {'✓' if tm[0] < 0 else '✗'}")
    return result


# ---------------------------------------------------------------------------
# Test 2 — SMC pattern edge
# ---------------------------------------------------------------------------

def test_smc_edge(df: pd.DataFrame, horizons: list = [1, 3, 5]) -> pd.DataFrame:
    """
    Simple SMC proxy edge test using price structure patterns.

    A full SMC test requires the SMCDetector; here we use a heuristic:
      - Bullish OB proxy: bar with large body down followed by impulse up
      - Bearish OB proxy: bar with large body up followed by impulse down

    This gives an indicative edge check. Replace with full SMCDetector
    for production validation.
    """
    print("\n[2] SMC Pattern Edge Test (heuristic proxy)")
    print("-" * 50)

    df = df.copy()
    df['body']    = (df['close'] - df['open']).abs()
    df['range_']  = df['high'] - df['low']
    df['ret1']    = df['close'].pct_change()

    # Bullish OB proxy: large bearish body + next bar closes higher
    bull_ob = (
        (df['close'] < df['open']) &                        # bearish candle
        (df['body'] > df['range_'].rolling(10).mean()) &    # larger than avg
        (df['ret1'].shift(-1) > 0)                          # followed by up move
    )
    # Bearish OB proxy
    bear_ob = (
        (df['close'] > df['open']) &
        (df['body'] > df['range_'].rolling(10).mean()) &
        (df['ret1'].shift(-1) < 0)
    )

    rows = []
    for h in horizons:
        fwd = df['close'].pct_change(h).shift(-h)
        for name, mask in [('BullishOB', bull_ob), ('BearishOB', bear_ob)]:
            rets: pd.Series = fwd[mask].dropna()
            if len(rets) < 5:
                continue
            rows.append({
                'pattern':  name,
                'horizon':  h,
                'n_signals': len(rets),
                'mean_ret':  float(rets.mean()),
                'sharpe':    float(rets.mean() / rets.std()) if float(rets.std()) > 0 else np.nan,
                'win_rate':  float((rets > 0).mean()),
            })

    result = pd.DataFrame(rows)
    if result.empty:
        print("  WARNING: no patterns found")
        return result

    for h in horizons:
        sub = result[result['horizon'] == h].set_index('pattern')
        print(f"\n  Horizon {h} bars:")
        print(sub[['n_signals', 'mean_ret', 'sharpe', 'win_rate']].to_string())

    return result


# ---------------------------------------------------------------------------
# Test 3 — Combined signal edge
# ---------------------------------------------------------------------------

def test_combined_edge(df: pd.DataFrame, train_df: "pd.DataFrame | None" = None,
                       horizons: list = [1, 3, 5]) -> pd.DataFrame:
    """
    Combined signal: regime in (Trend+, Squeeze) AND bullish OB proxy.
    Tests whether the intersection outperforms either component alone.
    """
    print("\n[3] Combined Signal Edge Test")
    print("-" * 50)

    df_feat = prepare_features(df.copy())
    labels  = label_regime_series(df_feat, train_df=train_df)

    # Bullish OB proxy aligned to df_feat index
    body    = (df_feat['close'] - df_feat['open']).abs()
    range_  = df_feat['high'] - df_feat['low']
    ret1    = df_feat['close'].pct_change()
    bull_ob = (
        (df_feat['close'] < df_feat['open']) &
        (body > range_.rolling(10).mean()) &
        (ret1.shift(-1) > 0)
    )

    bullish_regime = labels.isin(['Trend+', 'Squeeze'])
    combined = bullish_regime & bull_ob

    rows = []
    for h in horizons:
        fwd = df_feat['close'].pct_change(h).shift(-h)
        for name, mask in [
            ('Regime only',   bullish_regime),
            ('Pattern only',  bull_ob),
            ('Combined',      combined),
        ]:
            rets: pd.Series = fwd[mask].dropna()
            if len(rets) < 3:
                continue
            rows.append({
                'signal':    name,
                'horizon':   h,
                'n_signals': len(rets),
                'mean_ret':  float(rets.mean()),
                'sharpe':    float(rets.mean() / rets.std()) if float(rets.std()) > 0 else np.nan,
                'win_rate':  float((rets > 0).mean()),
            })

    result = pd.DataFrame(rows)
    if result.empty:
        print("  WARNING: no signals")
        return result

    for h in horizons:
        sub = result[result['horizon'] == h].set_index('signal')
        print(f"\n  Horizon {h} bars:")
        print(sub[['n_signals', 'mean_ret', 'sharpe', 'win_rate']].to_string())

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Component edge test for NYX system')
    parser.add_argument('--data-dir',     default='data/btcusdt',
                        help='Directory with CSV files')
    parser.add_argument('--timeframe',    default='4h',
                        help='Candle timeframe CSV to use (default: 4h)')
    parser.add_argument('--year',         type=int, default=2023,
                        help='Test year')
    parser.add_argument('--pretrain-year', type=int, default=None,
                        help='Year to pre-train HSMM via EM (optional)')
    parser.add_argument('--horizons',     default='1,3,5',
                        help='Forward horizons in bars (comma-separated)')
    args = parser.parse_args()

    horizons = [int(h) for h in args.horizons.split(',')]
    data_dir = Path(args.data_dir)

    # Find CSV
    tf = args.timeframe.replace('h', 'H').replace('m', 'M').replace('d', 'D')
    candidates = list(data_dir.glob(f'*{tf}*.csv')) + list(data_dir.glob(f'*{args.timeframe}*.csv'))
    if not candidates:
        print(f"ERROR: No CSV found for timeframe {args.timeframe} in {data_dir}")
        print(f"  Looked for: *{tf}*.csv, *{args.timeframe}*.csv")
        sys.exit(1)
    csv_path = candidates[0]
    print(f"Data: {csv_path}")

    df_all = load_ohlcv(csv_path)

    # Split by year
    df_test = df_all[pd.DatetimeIndex(df_all.index).year == args.year]
    if df_test.empty:
        print(f"ERROR: No data for year {args.year}")
        sys.exit(1)

    train_df = None
    if args.pretrain_year:
        train_raw = df_all[pd.DatetimeIndex(df_all.index).year == args.pretrain_year]
        if not train_raw.empty:
            train_df = prepare_features(pd.DataFrame(train_raw.copy()))
            print(f"Pre-train period: {args.pretrain_year} ({len(train_df)} bars)")
        else:
            print(f"WARNING: No data for pretrain year {args.pretrain_year}")

    print(f"Test period:  {args.year} ({len(df_test)} bars)")
    print(f"Horizons:     {horizons} bars")
    print("=" * 60)

    hsmm_edge  = test_hsmm_edge(pd.DataFrame(df_test), train_df=train_df, horizons=horizons)
    smc_edge   = test_smc_edge(pd.DataFrame(df_test), horizons=horizons)
    combo_edge = test_combined_edge(pd.DataFrame(df_test), train_df=train_df, horizons=horizons)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if not hsmm_edge.empty:
        h1 = hsmm_edge[hsmm_edge['horizon'] == horizons[0]]
        tp_df: pd.DataFrame = h1[h1['state'] == 'Trend+']
        tm_df: pd.DataFrame = h1[h1['state'] == 'Trend-']
        tp_ok = len(tp_df) > 0 and float(np.asarray(tp_df['mean_ret'].values)[0]) > 0
        tm_ok = len(tm_df) > 0 and float(np.asarray(tm_df['mean_ret'].values)[0]) < 0
        hsmm_pass = tp_ok and tm_ok
        print(f"HSMM edge:    {'PASS ✓' if hsmm_pass else 'FAIL ✗'}")

    if not smc_edge.empty:
        h1_smc = smc_edge[smc_edge['horizon'] == horizons[0]]
        bull_ok: pd.Series = h1_smc[h1_smc['pattern'] == 'BullishOB']['mean_ret']
        bear_ok: pd.Series = h1_smc[h1_smc['pattern'] == 'BearishOB']['mean_ret']
        smc_pass = (
            (len(bull_ok) > 0 and float(np.asarray(bull_ok.values)[0]) > 0) and
            (len(bear_ok) > 0 and float(np.asarray(bear_ok.values)[0]) < 0)
        )
        print(f"SMC edge:     {'PASS ✓' if smc_pass else 'FAIL ✗'}")

    if not combo_edge.empty:
        h1_c = combo_edge[combo_edge['horizon'] == horizons[0]]
        combo_sharpe: pd.Series = h1_c[h1_c['signal'] == 'Combined']['sharpe']
        regime_sharpe: pd.Series = h1_c[h1_c['signal'] == 'Regime only']['sharpe']
        if len(combo_sharpe) > 0 and len(regime_sharpe) > 0:
            combo_wins = float(np.asarray(combo_sharpe.values)[0]) > float(np.asarray(regime_sharpe.values)[0])
            print(f"Combined > Regime alone: {'YES ✓' if combo_wins else 'NO ✗'}")


if __name__ == '__main__':
    main()
