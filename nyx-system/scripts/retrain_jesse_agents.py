"""
Retrain the 4 mono-file Jesse agents on the canonical BTC data.

Tickets 14 + 15.

Ticket 14 — role-based feature plans (FEATURE_PLAN per agent).
Ticket 15 — agent-specific dataset sampling policy (`src/ml/jesse_dataset.py`)
            so Entry training drops from 100 k bars O(N²) down to a
            candidate-proximity-filtered selection that fits the
            sandbox budget.

The 4 dataset builders return `(df, sample_mask)`:
  Context : full 1D history, no mask
  Regime  : contiguous 4H, capped at 3 years, no mask
  Setup   : contiguous 1H, mask = volume-spike proxy
  Entry   : rolling 12-month 15m, mask = candidate-proximity (±5 bars)

The extended `.backtest(..., sample_mask=...)` skips masked-out bars
in the O(N²) per-bar `.analyze()` loop, which is where the Setup /
Entry runtime explodes.

Usage :
  python scripts/retrain_jesse_agents.py

Writes :
  reports/jesse_agents_retrain_ticket14.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

DATA_DIR = HERE / 'data' / 'raw' / 'mtf'
REPORT_DIR = HERE / 'reports'


def _load(tf: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f'BTCUSDT_{tf}.csv')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    for col in ('open', 'high', 'low', 'close'):
        df = df[df[col] > 0]
    return df


_OLD_FEATURE_COUNTS = {
    'JesseContextAgent': 8,
    'JesseRegimeAgent':  15,
    'JesseSetupAgent':   16,
    'JesseEntryAgent':   13,
}


def _agent_backtest(agent, df: pd.DataFrame,
                    sample_mask=None) -> Dict[str, Any]:
    t0 = time.time()
    res = agent.backtest(df, train_ratio=0.75, sample_mask=sample_mask)
    elapsed = time.time() - t0
    return {
        'accuracy':    float(res.get('accuracy', 0.0)),
        'n_bars':      int(res.get('n_bars', 0)),
        'pct_passed':  float(res.get('pct_passed', 0.0)),
        'avg_score':   float(res.get('avg_score', 0.0)),
        'elapsed_s':   round(elapsed, 2),
    }


def main() -> int:
    print('=== Ticket 14 + 15 — Retrain 4 Jesse agents ===', flush=True)

    print('Loading BTC MTF data...', flush=True)
    mtf_full = {tf: _load(tf) for tf in ('15m', '1h', '4h', '1d')}
    for tf, d in mtf_full.items():
        print(f'  {tf}: {len(d):,} rows', flush=True)

    # Canonical train window (matches Ticket 11).
    mtf = {}
    for tf in ('15m', '1h', '4h', '1d'):
        mtf[tf] = mtf_full[tf].loc['2020-01-01':'2022-12-31']

    from src.ml.jesse_agents import (
        JesseContextAgent, JesseRegimeAgent,
        JesseSetupAgent, JesseEntryAgent,
    )
    from src.ml.jesse_dataset import (
        build_context_dataset, build_regime_dataset,
        build_setup_dataset, build_entry_dataset,
    )

    # Apply per-agent dataset policy.
    ctx_df, ctx_mask = build_context_dataset(mtf)
    reg_df, reg_mask = build_regime_dataset(mtf, max_years=3)
    stp_df, stp_mask = build_setup_dataset(mtf, vol_threshold=1.5)
    ent_df, ent_mask = build_entry_dataset(
        mtf, rolling_months=12, proximity_bars=5, max_size=20_000,
    )

    jobs = [
        ('JesseContextAgent', JesseContextAgent(), ctx_df, ctx_mask),
        ('JesseRegimeAgent',  JesseRegimeAgent(),  reg_df, reg_mask),
        ('JesseSetupAgent',   JesseSetupAgent(),   stp_df, stp_mask),
        ('JesseEntryAgent',   JesseEntryAgent(),   ent_df, ent_mask),
    ]

    results = {}
    for name, agent, df, mask in jobs:
        kept = int(mask.sum()) if mask is not None else len(df)
        print(f'\n-- {name} on {len(df):,} bars '
              f'(mask keeps {kept:,}) --', flush=True)
        new_n_features = len(agent.FEATURE_PLAN)
        try:
            stats = _agent_backtest(agent, df, sample_mask=mask)
        except Exception as e:
            stats = {'error': repr(e)}
            print(f'  ERROR: {e}', flush=True)
        else:
            print(f'  accuracy:   {stats["accuracy"]:.3f}', flush=True)
            print(f'  n_bars:     {stats["n_bars"]}', flush=True)
            print(f'  pct_passed: {stats["pct_passed"]:.3f}', flush=True)
            print(f'  avg_score:  {stats["avg_score"]:.3f}', flush=True)
            print(f'  elapsed:    {stats["elapsed_s"]:.1f}s', flush=True)
        results[name] = {
            'old_n_features': _OLD_FEATURE_COUNTS[name],
            'new_n_features': new_n_features,
            'feature_plan_sample': list(agent.FEATURE_PLAN)[:6],
            'dataset_bars':    int(len(df)),
            'dataset_kept':    kept,
            'dataset_ratio':   round(kept / max(len(df), 1), 3),
            'ticket14_backtest': stats,
        }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        'ticket': 'Ticket 14 + Ticket 15 — Jesse retrain with role-based '
                  'feature plan + agent-specific dataset policy',
        'data_source': 'BTC 2020-01-01 to 2022-12-31',
        'dataset_policy': {
            'context': 'full 1D history, no mask',
            'regime':  'contiguous 4H, max 3 years, no mask',
            'setup':   'contiguous 1H, mask = volume-spike (vol/MA20 >= 1.5)',
            'entry':   'rolling 12-month 15m, mask = candidate-proximity '
                       '±5 bars, max 20k rows, reproducible via '
                       'random_state=42',
        },
        'agents': results,
    }
    out = REPORT_DIR / 'jesse_agents_retrain_ticket14.json'
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f'\nReport: {out}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
