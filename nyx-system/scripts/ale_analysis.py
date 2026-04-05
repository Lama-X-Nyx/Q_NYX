"""
ALE Feature Isolation Analysis
===============================
Computes Accumulated Local Effects (ALE) for every feature × state combination
in the 5-state HSMM and reports which features discriminate each regime.

Also runs the Girsanov regime-change score on a test window and plots the
sequential log-likelihood ratio to show where regime transitions are detected
before the smoothed posterior shifts.

Usage:
    cd nyx-system
    python scripts/ale_analysis.py                          # synthetic data
    python scripts/ale_analysis.py --csv data/btcusdt/4h.csv
    python scripts/ale_analysis.py --csv data/btcusdt/4h.csv --plot
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


STATES = ['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution']
FEATURES = ['price', 'atr']


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
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    df['atr_14']  = pd.Series(tr, index=df.index).rolling(14).mean()
    df['atr_50']  = pd.Series(tr, index=df.index).rolling(50).mean()
    df['sma_20']  = df['close'].rolling(20).mean()
    df['sma_50']  = df['close'].rolling(50).mean()
    if 'volume' in df.columns:
        df['volume_ma20'] = df['volume'].rolling(20).mean()
    df.dropna(inplace=True)
    return df


def df_to_observations(df: pd.DataFrame):
    obs = []
    for i in range(len(df)):
        row = df.iloc[i]
        obs.append({
            'price': float(row['returns']) if not np.isnan(row['returns']) else 0.0,
            'atr':   float(row['atr_14'])  if not np.isnan(row['atr_14'])  else 0.0,
        })
    return obs


def synthetic_data(n: int = 800) -> pd.DataFrame:
    """Generate synthetic OHLCV with regime changes for demo purposes."""
    np.random.seed(42)
    dates = pd.date_range('2021-01-01', periods=n, freq='4h')
    close = np.zeros(n)
    close[0] = 40000.0
    regime = np.zeros(n, dtype=int)  # 0=trend+, 1=range, 2=trend-, 3=squeeze, 4=dist

    seg_len = n // 5
    for i in range(1, n):
        seg = i // seg_len
        regime[i] = min(seg, 4)
        if regime[i] == 0:
            close[i] = close[i-1] * (1 + np.random.normal(0.003, 0.008))
        elif regime[i] == 1:
            close[i] = close[i-1] * (1 + np.random.normal(0.000, 0.006))
        elif regime[i] == 2:
            close[i] = close[i-1] * (1 + np.random.normal(-0.003, 0.008))
        elif regime[i] == 3:
            close[i] = close[i-1] * (1 + np.random.normal(0.001, 0.002))
        else:
            close[i] = close[i-1] * (1 + np.random.normal(-0.001, 0.015))

    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low  = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    vol  = np.abs(np.random.normal(500, 100, n))

    df = pd.DataFrame({'close': close, 'high': high, 'low': low, 'open': close, 'volume': vol},
                      index=dates)
    return prepare_features(df)


# ---------------------------------------------------------------------------
# ALE report
# ---------------------------------------------------------------------------

def run_ale_analysis(hsmm: SemiMarkovHMM, observations: list, n_bins: int = 20) -> dict:
    """
    Compute ALE for all features × states.

    Returns nested dict: results[state_name][feature] = (bin_centers, ale_values)
    """
    results = {}
    for si, state in enumerate(hsmm.states):
        results[state] = {}
        for feat in FEATURES:
            try:
                centers, ale = hsmm.compute_ale(
                    observations, feature=feat, state_idx=si, n_bins=n_bins
                )
                results[state][feat] = (centers, ale)
            except Exception as e:
                results[state][feat] = None
                print(f"    [WARN] ALE({state}, {feat}) failed: {e}")
    return results


def print_ale_report(results: dict) -> None:
    """Print a table of ALE ranges (max−min) — larger = more discriminating."""
    print("\n" + "=" * 70)
    print("ALE FEATURE DISCRIMINATION REPORT")
    print("  (ALE range = max−min; larger → feature matters more for that state)")
    print("=" * 70)
    print(f"  {'State':<18} {'price ALE-range':>18} {'atr ALE-range':>18}  verdict")
    print("-" * 70)

    for state, feat_data in results.items():
        ranges = {}
        for feat in FEATURES:
            entry = feat_data.get(feat)
            if entry is None:
                ranges[feat] = float('nan')
            else:
                _, ale = entry
                ranges[feat] = float(np.ptp(ale))  # peak-to-peak

        p_r = ranges.get('price', float('nan'))
        a_r = ranges.get('atr',   float('nan'))

        if np.isnan(p_r) or np.isnan(a_r):
            verdict = 'n/a'
        elif p_r > a_r * 1.5:
            verdict = 'price-driven'
        elif a_r > p_r * 1.5:
            verdict = 'ATR-driven'
        else:
            verdict = 'both'

        print(f"  {state:<18} {p_r:>18.6f} {a_r:>18.6f}  {verdict}")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Girsanov score report
# ---------------------------------------------------------------------------

def run_girsanov_analysis(
    hsmm:         SemiMarkovHMM,
    observations: list,
    window:       int = 50,
) -> np.ndarray:
    scores = hsmm.compute_girsanov_score(observations, window=window)

    gamma = hsmm.forward_backward(observations)
    dominant_p = gamma[:, np.argmax(gamma[-1])]   # P(dominant state) over time

    print("\n" + "=" * 70)
    print("GIRSANOV REGIME-CHANGE SCORE")
    print(f"  Window: {window} bars  |  Observations: {len(observations)}")
    print("-" * 70)
    print(f"  Score statistics:")
    print(f"    mean  = {scores.mean():.4f}")
    print(f"    max   = {scores.max():.4f}  (at bar {int(np.argmax(scores))})")
    print(f"    P75   = {np.percentile(scores, 75):.4f}")
    print(f"    P90   = {np.percentile(scores, 90):.4f}")

    # Find top-5 peaks (local maxima with spacing ≥ 10 bars)
    peaks = []
    for t in range(1, len(scores) - 1):
        if scores[t] > scores[t-1] and scores[t] > scores[t+1]:
            peaks.append((scores[t], t))
    peaks.sort(key=lambda x: -x[0])

    print(f"\n  Top-5 regime-change alerts (highest Girsanov peaks):")
    seen = set()
    count = 0
    for score, t in peaks:
        if any(abs(t - s) < 10 for s in seen):
            continue
        seen.add(t)
        dom_p = float(dominant_p[t])
        print(f"    bar {t:4d}  score={score:8.4f}  P(dominant)={dom_p:.3f}")
        count += 1
        if count >= 5:
            break

    print("=" * 70)
    return scores


# ---------------------------------------------------------------------------
# Optional plotting
# ---------------------------------------------------------------------------

def plot_results(
    observations: list,
    results:      dict,
    scores:       np.ndarray,
    gamma:        np.ndarray,
    states:       list,
    out_dir:      Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("\n[INFO] matplotlib not installed — skipping plots.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    T = len(observations)
    t_ax = np.arange(T)

    # ---- ALE plots ----
    fig, axes = plt.subplots(len(states), len(FEATURES),
                              figsize=(5 * len(FEATURES), 3 * len(states)),
                              squeeze=False)
    fig.suptitle('ALE: Feature → P(state)', fontsize=13, fontweight='bold')

    for si, state in enumerate(states):
        for fi, feat in enumerate(FEATURES):
            ax = axes[si][fi]
            entry = results[state].get(feat)
            if entry is None:
                ax.set_visible(False)
                continue
            centers, ale = entry
            ax.plot(centers, ale, marker='o', ms=3)
            ax.axhline(0, color='grey', lw=0.8, ls='--')
            ax.set_title(f'{state} / {feat}', fontsize=9)
            ax.set_xlabel(feat, fontsize=8)
            ax.set_ylabel('ALE', fontsize=8)
            ax.grid(alpha=0.3)

    plt.tight_layout()
    path = out_dir / 'ale_feature_effects.png'
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved: {path}")

    # ---- Girsanov + posterior ----
    fig = plt.figure(figsize=(12, 7))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.35)

    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t_ax, scores, color='crimson', lw=1)
    ax0.axhline(np.percentile(scores, 90), color='orange', ls='--', lw=0.8,
                label='P90 threshold')
    ax0.set_title('Girsanov Regime-Change Score', fontsize=11)
    ax0.set_ylabel('Score')
    ax0.legend(fontsize=8)
    ax0.grid(alpha=0.3)

    ax1 = fig.add_subplot(gs[1])
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    for si, state in enumerate(states):
        ax1.fill_between(t_ax, gamma[:, si], alpha=0.55,
                         color=colors[si % len(colors)], label=state)
    ax1.set_title('HSMM Smoothed State Posteriors γ(t)', fontsize=11)
    ax1.set_xlabel('Bar')
    ax1.set_ylabel('P(state)')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(alpha=0.3)

    path2 = out_dir / 'girsanov_and_posteriors.png'
    plt.savefig(path2, dpi=120)
    plt.close()
    print(f"  Saved: {path2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='ALE + Girsanov analysis for NYX HSMM')
    parser.add_argument('--csv',        default=None,  help='OHLCV CSV path')
    parser.add_argument('--em-iters',   type=int, default=30, help='EM iterations')
    parser.add_argument('--n-bins',     type=int, default=20, help='ALE bins')
    parser.add_argument('--girs-window',type=int, default=50, help='Girsanov accumulation window')
    parser.add_argument('--plot',       action='store_true',   help='Save PNG plots')
    parser.add_argument('--plot-dir',   default='reports/ale', help='Output directory for plots')
    args = parser.parse_args()

    # Load data
    if args.csv:
        csv_path = Path(args.csv)
        print(f"Loading: {csv_path}")
        df_raw = load_ohlcv(csv_path)
        df     = prepare_features(df_raw)
    else:
        print("No CSV provided — using synthetic data (5-regime sequence).")
        df = synthetic_data(n=800)

    print(f"Bars:    {len(df)}  ({df.index[0].date()} → {df.index[-1].date()})")

    # Train HSMM
    hsmm = SemiMarkovHMM(states=STATES)
    print(f"\nTraining HSMM ({len(STATES)} states, {args.em_iters} EM iters)…")
    hsmm.initialize_parameters(df)
    observations = df_to_observations(df)
    ll = hsmm.fit(observations, n_iter=args.em_iters, tol=1e-4)
    print(f"  EM converged: {len(ll)} iters, final LL={ll[-1]:.0f}" if ll else "  EM skipped")

    # ---- ALE ----
    print(f"\nComputing ALE ({args.n_bins} bins) for {len(STATES)} states × {len(FEATURES)} features…")
    obs_window = observations[-500:]   # use last 500 bars to keep runtime reasonable
    ale_results = run_ale_analysis(hsmm, obs_window, n_bins=args.n_bins)
    print_ale_report(ale_results)

    # ---- Girsanov ----
    print(f"\nComputing Girsanov score (window={args.girs_window})…")
    scores = run_girsanov_analysis(hsmm, obs_window, window=args.girs_window)

    # ---- Plots ----
    if args.plot:
        print(f"\nGenerating plots → {args.plot_dir}/")
        gamma = hsmm.forward_backward(obs_window)
        plot_results(
            obs_window, ale_results, scores, gamma, STATES, Path(args.plot_dir)
        )
    else:
        print("\n[Tip] Re-run with --plot to save PNG charts.")


if __name__ == '__main__':
    main()
