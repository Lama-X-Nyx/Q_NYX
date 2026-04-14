"""
Production train of the ETH model + quick OOS report.

Usage:
  python scripts/train_eth_model.py

Writes:
  data/features/ETHUSDT_features_{15m,1h,1d}.parquet  (cache)
  models/ETHUSDT/ml_filter_v1.pkl
  models/ETHUSDT/scaler.pkl
  models/ETHUSDT/feature_names.json
  models/ETHUSDT/training_metadata.json
  reports/ETHUSDT_oos_report.json
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
MODEL_DIR = HERE / 'models' / 'ETHUSDT'
REPORT_DIR = HERE / 'reports'


def _load(tf: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f'ETHUSDT_{tf}.csv')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    for col in ('open', 'high', 'low', 'close'):
        df = df[df[col] > 0]
    return df


def main() -> int:
    print("=== ETH training ===")
    mtf = {tf: _load(tf) for tf in ('15m', '1h', '4h', '1d')}
    for tf, d in mtf.items():
        print(f"  {tf}: {len(d):,} rows  [{d.index.min()} → {d.index.max()}]")

    # Compute + cache features — FOUR TIMEFRAMES (hard rule).
    from src.ml.jesse_features import compute_stationary_features
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    feats = {}
    for tf in ('15m', '1h', '4h', '1d'):
        cache = FEAT_DIR / f'ETHUSDT_features_{tf}.parquet'
        if cache.exists():
            feats[tf] = pd.read_parquet(cache)
        else:
            feats[tf] = compute_stationary_features(mtf[tf], feature_set='full')
            feats[tf].to_parquet(cache)
        print(f"  features {tf}: {feats[tf].shape}")

    # Train through 2022-12-31 and save.
    from src.ml.train_asset_model import train_and_save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    meta = train_and_save(
        symbol='ETHUSDT', mtf_data=mtf, mtf_features=feats,
        train_end='2022-12-31', out_dir=MODEL_DIR,
    )
    print("  training:", meta)

    # Quick OOS pass on 2023.
    from src.ml.nyx_pipeline import NYXPipeline
    pipe = NYXPipeline()
    r = pipe.run(mtf, feats,
                 train_end='2022-12-31',
                 test_start='2023-01-01',
                 test_end='2023-12-31')
    print("  2023 OOS:", {k: r.get(k) for k in (
        'n_candidates', 'n_trades', 'sharpe', 'total_pnl_dollars',
        'max_drawdown_pct', 'execution_reject_rate', 'bear_dial_activation_rate',
    )})

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        'symbol': 'ETHUSDT',
        'training': meta,
        'oos_2023': {k: r.get(k) for k in (
            'n_candidates', 'n_trades', 'sharpe', 'total_pnl_dollars',
            'max_drawdown_pct', 'execution_reject_rate',
            'bear_dial_activation_rate',
        )},
    }
    (REPORT_DIR / 'ETHUSDT_oos_report.json').write_text(
        json.dumps(report, indent=2, default=str)
    )
    print(f"  report saved: reports/ETHUSDT_oos_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
