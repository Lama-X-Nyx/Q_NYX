"""
Ticket 1 — Precompute Feature Parquet Files

Precomputes all MLFeatureEngine features for each timeframe and saves them
as Parquet files for fast loading during training and backtesting.

Outputs (in data/features/):
  features_1d.parquet    — daily features
  features_1h.parquet    — 1H features + gamma_1h HSMM columns
  features_15m.parquet   — 15M features + gamma_15m HSMM columns

Usage:
  python scripts/precompute_features.py --pair BTCUSDT
  python scripts/precompute_features.py --pair BTCUSDT --force
"""

import sys
import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.mtf_loader import MTFLoader
from src.ml.feature_engine import MLFeatureEngine

FEATURES_DIR = Path('data/features')
CACHE_DIR    = Path('data/pretrain_cache')
HSMM_STATES  = ['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution', 'Liquidation']
HSMM_COLS_1H = [f'hsmm_1h_{s.lower().replace("+","p").replace("-","m").replace(" ","_")}' for s in HSMM_STATES]
HSMM_COLS_15M = [f'hsmm_15m_{s.lower().replace("+","p").replace("-","m").replace(" ","_")}' for s in HSMM_STATES]


def parse_args():
    p = argparse.ArgumentParser(description='Precompute feature Parquet files')
    p.add_argument('--pair',     default='BTCUSDT')
    p.add_argument('--data-dir', default='data/raw/mtf')
    p.add_argument('--force',    action='store_true', help='Recompute even if Parquet exists')
    p.add_argument('--no-hsmm',  action='store_true', help='Skip HSMM gamma columns')
    return p.parse_args()


def _load_gamma(name: str) -> np.ndarray | None:
    """Load pre-trained HSMM gamma array from cache."""
    path = CACHE_DIR / f'{name}.npy'
    if path.exists():
        arr = np.load(str(path))
        print(f'  Loaded {name}: {arr.shape}')
        return arr
    print(f'  [WARN] {path} not found — HSMM columns will be uniform (1/6)')
    return None


def _append_hsmm_columns(df: pd.DataFrame,
                          gamma: np.ndarray | None,
                          col_names: list[str]) -> pd.DataFrame:
    """Append HSMM probability columns to feature DataFrame."""
    n = len(df)
    if gamma is not None and len(gamma) == n:
        for j, col in enumerate(col_names):
            df[col] = gamma[:, j]
    else:
        for col in col_names:
            df[col] = 1.0 / 6.0
    return df


def compute_1d_features(df_1d: pd.DataFrame) -> pd.DataFrame:
    """Compute daily features using MLFeatureEngine."""
    engine = MLFeatureEngine()
    feats  = engine.compute(df_1d)
    return feats


def compute_1h_features(df_1h: pd.DataFrame,
                         gamma_1h: np.ndarray | None) -> pd.DataFrame:
    """Compute 1H features + HSMM gamma columns."""
    engine = MLFeatureEngine()
    feats  = engine.compute(df_1h)
    feats  = _append_hsmm_columns(feats, gamma_1h, HSMM_COLS_1H)
    return feats


def compute_15m_features(df_15m: pd.DataFrame,
                          gamma_15m: np.ndarray | None) -> pd.DataFrame:
    """Compute 15M features + HSMM gamma columns."""
    engine = MLFeatureEngine()
    feats  = engine.compute(df_15m)
    feats  = _append_hsmm_columns(feats, gamma_15m, HSMM_COLS_15M)
    return feats


def main():
    args = parse_args()
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    out_1d  = FEATURES_DIR / f'{args.pair}_features_1d.parquet'
    out_1h  = FEATURES_DIR / f'{args.pair}_features_1h.parquet'
    out_15m = FEATURES_DIR / f'{args.pair}_features_15m.parquet'

    print('=' * 60)
    print('NYX Feature Precompute')
    print(f'  pair     : {args.pair}')
    print(f'  data-dir : {args.data_dir}')
    print(f'  output   : {FEATURES_DIR}')
    print('=' * 60)

    # ------------------------------------------------------------------
    # Load raw data
    # ------------------------------------------------------------------
    print('\n[1/4] Loading MTF data...')
    loader  = MTFLoader(data_dir=args.data_dir)
    mtf_all = loader.load(args.pair, tfs=['1d', '4h', '1h', '15m'])
    df_1d   = mtf_all['1d']
    df_1h   = mtf_all['1h']
    df_15m  = mtf_all['15m']
    print(f'  1D:  {len(df_1d):,} bars')
    print(f'  1H:  {len(df_1h):,} bars')
    print(f'  15M: {len(df_15m):,} bars')

    # ------------------------------------------------------------------
    # Load HSMM gamma arrays (if available)
    # ------------------------------------------------------------------
    gamma_1h  = None
    gamma_15m = None
    if not args.no_hsmm:
        print('\n[2/4] Loading HSMM gamma arrays...')
        gamma_1h  = _load_gamma('gamma_1h_train')
        gamma_15m = _load_gamma('gamma_15m_train')

    # ------------------------------------------------------------------
    # Compute 1D features
    # ------------------------------------------------------------------
    if args.force or not out_1d.exists():
        print('\n[3/4] Computing 1D features...')
        t0    = time.time()
        f_1d  = compute_1d_features(df_1d)
        print(f'  Done in {time.time()-t0:.1f}s — {f_1d.shape[1]} features, {len(f_1d):,} rows')
        f_1d.to_parquet(str(out_1d))
        print(f'  Saved → {out_1d}')
    else:
        print(f'\n[3/4] {out_1d.name} already exists (use --force to recompute)')

    # ------------------------------------------------------------------
    # Compute 1H features
    # ------------------------------------------------------------------
    if args.force or not out_1h.exists():
        print('\n Computing 1H features...')
        t0    = time.time()
        f_1h  = compute_1h_features(df_1h, gamma_1h)
        print(f'  Done in {time.time()-t0:.1f}s — {f_1h.shape[1]} features, {len(f_1h):,} rows')
        f_1h.to_parquet(str(out_1h))
        print(f'  Saved → {out_1h}')
    else:
        print(f'  {out_1h.name} already exists (use --force to recompute)')

    # ------------------------------------------------------------------
    # Compute 15M features
    # ------------------------------------------------------------------
    if args.force or not out_15m.exists():
        print('\n[4/4] Computing 15M features...')
        t0    = time.time()
        f_15m = compute_15m_features(df_15m, gamma_15m)
        print(f'  Done in {time.time()-t0:.1f}s — {f_15m.shape[1]} features, {len(f_15m):,} rows')
        f_15m.to_parquet(str(out_15m))
        print(f'  Saved → {out_15m}')
    else:
        print(f'\n[4/4] {out_15m.name} already exists (use --force to recompute)')

    print('\n' + '=' * 60)
    print('Feature precompute complete.')
    print(f'  1D  → {out_1d}')
    print(f'  1H  → {out_1h}')
    print(f'  15M → {out_15m}')
    print('=' * 60)


if __name__ == '__main__':
    main()
