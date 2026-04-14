"""
Compute + cache 4h stationary features for BTC, ETH, SOL.

4h was the missing timeframe in the original MTF block. This script
produces `data/features/<SYMBOL>_features_4h.parquet` for each asset
so the retrain can pick them up.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

DATA_DIR = HERE / 'data' / 'raw'
BTC_MTF = DATA_DIR / 'mtf'
FEAT_DIR = HERE / 'data' / 'features'


def _load_btc_4h() -> pd.DataFrame:
    df = pd.read_csv(BTC_MTF / 'BTCUSDT_4h.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.set_index('datetime')


def _load_eth_4h() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / 'ETHUSDT_4h.csv')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    for c in ('open', 'high', 'low', 'close'):
        df = df[df[c] > 0]
    return df


def _load_sol_4h() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / 'SOLUSDT_4hours.csv',
                     usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('datetime').drop(columns='timestamp')
    for c in ('open', 'high', 'low', 'close'):
        df = df[df[c] > 0]
    return df


def main() -> int:
    from src.ml.jesse_features import compute_stationary_features

    FEAT_DIR.mkdir(parents=True, exist_ok=True)

    for symbol, loader in (
        ('BTCUSDT', _load_btc_4h),
        ('ETHUSDT', _load_eth_4h),
        ('SOLUSDT', _load_sol_4h),
    ):
        out = FEAT_DIR / f'{symbol}_features_4h.parquet'
        df = loader()
        print(f"{symbol} 4h OHLCV: {df.shape}  [{df.index.min()} → {df.index.max()}]")
        feats = compute_stationary_features(df, feature_set='full')
        feats.to_parquet(out)
        print(f"  → {out.relative_to(HERE)}  {feats.shape}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
