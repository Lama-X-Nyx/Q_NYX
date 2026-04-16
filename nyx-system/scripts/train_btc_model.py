"""
Production train of the BTC Meta-GBM + quick OOS report (Ticket 11).

Ticket 11 — Train the Meta-GBM on the canonical BTC pipeline.

Canonical training inputs :
  - BTC MTF OHLCV (15m / 1h / 4h / 1d) from data/raw/mtf/
  - Stationary features computed via compute_stationary_features
    (cached under data/features/BTCUSDT_features_<tf>.parquet)
  - `train_asset_model.train_and_save` — canonical helper enforcing
    MIN_FEATURES ≥ 65 + 4-TF prefixes (Rule 2)
  - train_end = 2022-12-31 (matches ETH / SOL convention)

Canonical OOS :
  - test_start = 2023-01-01, test_end = 2023-12-31
  - Runs through `NYXEngine.run()` (the canonical runtime entrypoint)
  - NYXEngine delegates scoring to `MetaGBM` (Ticket 07) which
    encapsulates the freshly trained GBM + scaler.

Usage :
  python scripts/train_btc_model.py

Writes :
  models/BTCUSDT/ml_filter_v1.pkl
  models/BTCUSDT/scaler.pkl
  models/BTCUSDT/feature_names.json
  models/BTCUSDT/training_metadata.json
  reports/BTCUSDT_oos_report.json

Reproducibility :
  The GBM uses random_state=42 inside train_and_save. Two runs on
  the same input data produce identical model artefacts (bit-for-bit
  pickle identity not guaranteed due to pickle format versioning,
  but n_features / accuracy / OOS metrics are deterministic).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

DATA_DIR = HERE / 'data' / 'raw' / 'mtf'   # BTC lives in mtf/ subdir
FEAT_DIR = HERE / 'data' / 'features'
MODEL_DIR = HERE / 'models' / 'BTCUSDT'
REPORT_DIR = HERE / 'reports'


def _load(tf: str) -> pd.DataFrame:
    """Load BTC OHLCV for a given TF (mirrors tests/conftest.py)."""
    df = pd.read_csv(DATA_DIR / f'BTCUSDT_{tf}.csv')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    for col in ('open', 'high', 'low', 'close'):
        df = df[df[col] > 0]
    return df


def main() -> int:
    print('=== BTC training (Ticket 11) ===')
    mtf = {tf: _load(tf) for tf in ('15m', '1h', '4h', '1d')}
    for tf, d in mtf.items():
        print(f'  {tf}: {len(d):,} rows  [{d.index.min()} → {d.index.max()}]')

    # Compute + cache features — FOUR TIMEFRAMES (Rule 2).
    from src.ml.jesse_features import compute_stationary_features
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    feats = {}
    for tf in ('15m', '1h', '4h', '1d'):
        cache = FEAT_DIR / f'BTCUSDT_features_{tf}.parquet'
        if cache.exists():
            feats[tf] = pd.read_parquet(cache)
        else:
            feats[tf] = compute_stationary_features(mtf[tf], feature_set='full')
            feats[tf].to_parquet(cache)
        print(f'  features {tf}: {feats[tf].shape}')

    # Train through 2022-12-31 and save.
    # train_and_save enforces MIN_FEATURES ≥ 65 + 4-TF prefixes.
    from src.ml.train_asset_model import train_and_save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    meta = train_and_save(
        symbol='BTCUSDT', mtf_data=mtf, mtf_features=feats,
        train_end='2022-12-31', out_dir=MODEL_DIR,
    )
    print('  training:', meta)

    # Quick OOS pass on 2023 through the canonical runtime.
    from src.core.nyx_engine import NYXEngine
    engine = NYXEngine()
    r = engine.run(
        mtf, feats,
        train_end='2022-12-31',
        test_start='2023-01-01',
        test_end='2023-12-31',
    )
    oos_summary = {k: r.get(k) for k in (
        'n_candidates', 'n_trades', 'sharpe', 'total_pnl_dollars',
        'max_drawdown_pct', 'execution_reject_rate',
        'bear_dial_activation_rate',
    )}
    print('  2023 OOS:', oos_summary)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        'symbol': 'BTCUSDT',
        'ticket': 'Ticket 11 — Train Meta-GBM on canonical BTC pipeline',
        'training': meta,
        'oos_2023': oos_summary,
    }
    (REPORT_DIR / 'BTCUSDT_oos_report.json').write_text(
        json.dumps(report, indent=2, default=str)
    )
    print('  report saved: reports/BTCUSDT_oos_report.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
