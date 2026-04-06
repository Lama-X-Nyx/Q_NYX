#!/usr/bin/env python3
"""
NYX ML Ecosystem Training Pipeline

Trains all 4 ML agents + the meta-orchestrator in order:
  1. Pretrain HSMM (or load cache)
  2. Build PrecomputedStates (HSMM forward pass + SMC on ALL data)
  3. Train MLContextAgent   (1D)
  4. Train MLRegimeAgent    (1H + HSMM gamma)
  5. Train MLSetupAgent     (15M + HSMM gamma + SMC)
  6. Generate meta-dataset  (all agent signals on every bar)
  7. Train MLOrchestrator   (meta-LGB)
  8. Print calibration report

Usage:
    python scripts/train_ml_ecosystem.py --train-end 2022-12-31
    python scripts/train_ml_ecosystem.py --train-end 2022-12-31 --em-iters 50
"""

import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.orchestrator import Orchestrator
from src.core.precomputed_runner import (
    PrecomputedStates, _prepare_features_full, _compute_log_B_from_arrays,
    forward_streaming, _precompute_smc, _precompute_context
)
from src.ml.ml_agents import MLContextAgent, MLRegimeAgent, MLSetupAgent
from src.ml.ml_orchestrator import MLOrchestrator


BASE_CONFIG = {
    'fractal': {'context_tf': '1d', 'structure_tf': '4h',
                'regime_tf': '1h', 'setup_tf': '15m',
                'use_entry_agent': False},
    'mtf': {'timeframes': {'context': '1d', 'regime': '1h', 'setup': '15m'}},
    'strategy': {'mtf_conditions': {
        'sdc_min': 4.5, 'stability_4h_min': 0.55, 'alignment_15m_min': 0.40
    }, 'min_entry_score': 0.78},
    'fractal_readiness': {'context_min_bars': 200, 'regime_min_bars': 100, 'setup_min_bars': 50},
    'risk': {'risk_per_trade_pct': 0.02, 'max_leverage': 10,
             'atr_period': 50, 'atr_sl_multiplier': 2.5,
             'atr_k_by_regime': {'Trend+': 2.5, 'Trend-': 2.5, 'Range': 2.5,
                                 'Squeeze': 2.5, 'Distribution': 2.5},
             'min_profit_atr_mult': 8.0, 'trail_tp_retracement': 0.25,
             'cooldown_bars': 96, 'max_drawdown_pct': 0.05,
             'fee_pct': 0.0004, 'slippage_pct': 0.0005},
    'macro': {'enabled': False},
}


def load_mtf(pair: str, data_dir: str = 'data/raw/mtf') -> dict:
    tfs = {'1d': '1d', '4h': '4h', '1h': '1h', '15m': '15m'}
    out = {}
    for key, suffix in tfs.items():
        path = Path(data_dir) / f'{pair}_{suffix}.csv'
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df['datetime'])
        df = df[['open', 'high', 'low', 'close', 'volume']].sort_index()
        out[key] = df
    return out


def main():
    parser = argparse.ArgumentParser(description='NYX ML Ecosystem Training')
    parser.add_argument('--pair',      default='BTCUSDT')
    parser.add_argument('--train-end', default='2022-12-31',
                        help='Exclusive end date for training (default: 2022-12-31)')
    parser.add_argument('--data-dir',  default='data/raw/mtf')
    parser.add_argument('--em-iters',  type=int, default=30)
    parser.add_argument('--n-splits',  type=int, default=5)
    args = parser.parse_args()

    t_total = time.time()
    print(f'\n{"═"*70}')
    print(f'  NYX ML Ecosystem Training Pipeline')
    print(f'  Pair : {args.pair}  |  Train end : {args.train_end}')
    print(f'{"═"*70}\n')

    # ---- Load data ----
    print('Loading MTF data…')
    mtf_all = load_mtf(args.pair, args.data_dir)
    train_end = pd.Timestamp(args.train_end)
    mtf_train = {tf: df[df.index < train_end] for tf, df in mtf_all.items()}
    for tf, df in mtf_train.items():
        print(f'  {tf}: {len(df)} bars  ({df.index[0].date()} → {df.index[-1].date()})')

    # ---- Step 1: HSMM pre-training ----
    print('\n[Step 1] HSMM pre-training…')
    from scripts.backtest_mtf import pretrain_agents, _pretrain_cache_key, \
        _load_pretrain_cache, _save_pretrain_cache, CACHE_DIR
    import hashlib as _hlib

    orch = Orchestrator(BASE_CONFIG)
    start_ts = mtf_train['1h'].index[0]
    months   = int((train_end - start_ts).days / 30)

    cache_blob = f'{args.pair}|{args.train_end}|{months}|{args.em_iters}|v6_calibrate'
    cache_key  = _hlib.md5(cache_blob.encode()).hexdigest()[:16]
    cache_path = CACHE_DIR / f'{cache_key}.pkl'

    if _load_pretrain_cache(orch, cache_path):
        print('  HSMM loaded from cache')
    else:
        pretrain_agents(orch, mtf_train, pretrain_end=args.train_end,
                        pretrain_months=months, em_iters=args.em_iters,
                        pair=args.pair, use_cache=True)

    # ---- Step 2: Precompute HSMM gammas + SMC on ALL training data ----
    print('\n[Step 2] Precomputing HSMM states + SMC on all training data…')
    t2 = time.time()

    df_1h   = mtf_train['1h']
    df_4h   = mtf_train.get('4h')
    df_15m  = mtf_train['15m']
    df_1d   = mtf_train['1d']

    # Regime HSMM (1H)
    df_1h_prep = _prepare_features_full(df_1h, df_htf=df_4h)
    orch.regime_agent.hsmm.initialize_parameters(df_1h_prep.dropna().tail(500))
    from src.core.precomputed_runner import _build_obs_arrays
    obs_1h   = _build_obs_arrays(df_1h_prep)
    log_B_1h = _compute_log_B_from_arrays(orch.regime_agent.hsmm, obs_1h)
    gamma_1h = forward_streaming(orch.regime_agent.hsmm, log_B_1h)
    print(f'  gamma_1h   : {gamma_1h.shape}')

    # Setup HSMM (15M)
    df_15m_prep = _prepare_features_full(df_15m, df_htf=df_1h)
    orch.setup_agent.hsmm.initialize_parameters(df_15m_prep.dropna().tail(500))
    obs_15m   = _build_obs_arrays(df_15m_prep)
    log_B_15m = _compute_log_B_from_arrays(orch.setup_agent.hsmm, obs_15m)
    gamma_15m = forward_streaming(orch.setup_agent.hsmm, log_B_15m)
    print(f'  gamma_15m  : {gamma_15m.shape}')

    # Context SMA200 (1D)
    context_arr = _precompute_context(df_1d, orch.context_agent.trend_threshold)
    print(f'  context    : {len(context_arr)} bars  '
          f'({(context_arr=="bullish").sum()} bull / {(context_arr=="bearish").sum()} bear)')

    # SMC on ALL training 15M data (needed for SetupML training)
    print(f'  SMC 15M    : {len(df_15m)} bars… ', end='', flush=True)
    t_smc = time.time()
    smc_list, smc_start = _precompute_smc(
        df_15m_prep, orch.setup_agent.smc, window=200,
        start_iloc=0, end_iloc=len(df_15m)
    )
    print(f'{time.time()-t_smc:.0f}s  ({len(smc_list)} dicts)')
    print(f'  Step 2 total: {time.time()-t2:.0f}s')

    # Build context_series aligned to 15M index
    idx_1d  = df_1d.index.astype(np.int64)
    idx_15m = df_15m.index.astype(np.int64)
    ctx_15m = []
    for ts_ns in idx_15m:
        i_1d = int(np.searchsorted(idx_1d, ts_ns, side='left')) - 1
        ctx_15m.append(str(context_arr[i_1d]) if 0 <= i_1d < len(context_arr) else 'insufficient')
    context_series_15m = pd.Series(ctx_15m, index=df_15m.index)

    # ---- Step 3: Train MLContextAgent (1D) ----
    print('\n[Step 3] Training MLContextAgent…')
    ctx_agent = MLContextAgent()
    ctx_agent.pretrain(df_1d, n_splits=args.n_splits)

    # ---- Step 4: Train MLRegimeAgent (1H + HSMM) ----
    print('\n[Step 4] Training MLRegimeAgent…')
    reg_agent = MLRegimeAgent()
    reg_agent.pretrain(df_1h, gamma_1h, n_splits=args.n_splits)

    # ---- Step 5: Train MLSetupAgent (15M + HSMM + SMC) ----
    print('\n[Step 5] Training MLSetupAgent…')
    stp_agent = MLSetupAgent()
    stp_agent.pretrain(
        df_15m, gamma_15m, smc_list, context_series_15m,
        smc_start_iloc=smc_start, n_splits=args.n_splits
    )

    # ---- Step 6 + 7: Generate meta-dataset & train MLOrchestrator ----
    print('\n[Step 6] Generating MLOrchestrator meta-dataset…')
    ml_orch = MLOrchestrator()
    meta_ds = ml_orch.generate_training_data(
        df_15m=df_15m,
        context_arr=context_arr,
        gamma_1h=gamma_1h,
        gamma_15m=gamma_15m,
        smc_list=smc_list,
        smc_start_iloc=smc_start,
        df_1d=df_1d,
        df_1h=df_1h,
    )

    print('\n[Step 7] Training MLOrchestrator…')
    ml_orch.pretrain(meta_ds, n_splits=args.n_splits)

    # ---- Report ----
    elapsed = time.time() - t_total
    print(f'\n{"═"*70}')
    print(f'  Training complete in {elapsed:.0f}s ({elapsed/60:.1f} min)')
    print(f'  Saved:')
    print(f'    data/pretrain_cache/context_ml.pkl')
    print(f'    data/pretrain_cache/regime_ml.pkl')
    print(f'    data/pretrain_cache/setup_ml.pkl')
    print(f'    data/pretrain_cache/orchestrator_ml.pkl')
    print(f'\n  To run backtest with ML ecosystem:')
    print(f'    python scripts/backtest_mtf.py --start 2023-01-01 --end 2023-12-31 \\')
    print(f'        --pretrain-all --use-cache --precompute --precompute-cache')
    print(f'{"═"*70}\n')


if __name__ == '__main__':
    main()
