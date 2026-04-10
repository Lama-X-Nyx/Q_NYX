"""
train_ml_ecosystem.py — Train the full NYX ML ecosystem in order.

Steps:
  1. Load all MTF data
  2. Pretrain HSMM (or load cache)
  3. Run PrecomputedStates.precompute() on full dataset (all data for SMC)
  4. Train MLContextAgent  on 1D data
  5. Train MLRegimeAgent   on 1H data + gamma_1h
  6. Train MLSetupAgent    on 15M data + gamma_15m + smc_list
  7. Generate meta-dataset for MLOrchestrator
  8. Train MLOrchestrator  on meta-dataset
  9. Print AUC + calibration report

Usage:
  python scripts/train_ml_ecosystem.py --pair BTCUSDT --train-end 2022-12-31
"""

import sys
import argparse
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.mtf_loader import MTFLoader
from src.ml.ml_agents import MLContextAgent, MLRegimeAgent, MLSetupAgent
from src.ml.ml_orchestrator import MLOrchestrator


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Train NYX ML Ecosystem')
    p.add_argument('--pair',      default='BTCUSDT', help='Trading pair (default: BTCUSDT)')
    p.add_argument('--train-end', default=None,      help='Exclusive training end date YYYY-MM-DD')
    p.add_argument('--data-dir',  default='data/raw/mtf', help='MTF data directory')
    p.add_argument('--em-iters',  type=int, default=30,   help='HSMM EM iterations (default: 30)')
    p.add_argument('--n-splits',  type=int, default=5,    help='Walk-forward CV splits (default: 5)')
    p.add_argument('--force-retrain', action='store_true', help='Ignore all caches and retrain')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Calibration report
# ---------------------------------------------------------------------------

def calibration_report(name: str, proba: np.ndarray, labels: np.ndarray,
                        n_bins: int = 5):
    """Print ECE and per-bin calibration."""
    from sklearn.calibration import calibration_curve
    try:
        frac_pos, mean_pred = calibration_curve(labels, proba, n_bins=n_bins, strategy='uniform')
        ece = float(np.mean(np.abs(frac_pos - mean_pred)))
        print(f'  [{name}] ECE={ece:.4f}')
        print(f'    {"pred":>8}  {"actual":>8}')
        for mp, fp in zip(mean_pred, frac_pos):
            print(f'    {mp:8.3f}  {fp:8.3f}')
    except Exception as e:
        print(f'  [{name}] Calibration skipped: {e}')


# ---------------------------------------------------------------------------
# HSMM helper (reuse precomputed_runner machinery)
# ---------------------------------------------------------------------------

def load_or_train_hsmm(mtf_all, cache_dir, em_iters, force):
    """
    Initialize HSMM models using the streaming forward approach.
    Returns gamma_1h (T_1h, 6) and gamma_15m (T_15m, 6).
    """
    from src.core.precomputed_runner import (
        _prepare_features_full,
        _build_obs_arrays,
        _compute_log_B_from_arrays,
        forward_streaming,
    )
    from src.core.hsmm import SemiMarkovHMM

    cache_1h  = Path(cache_dir) / 'gamma_1h_train.npy'
    cache_15m = Path(cache_dir) / 'gamma_15m_train.npy'

    if not force and cache_1h.exists() and cache_15m.exists():
        print('  [HSMM] Loading gamma arrays from cache...')
        gamma_1h  = np.load(str(cache_1h))
        gamma_15m = np.load(str(cache_15m))
        print(f'    gamma_1h:  {gamma_1h.shape}')
        print(f'    gamma_15m: {gamma_15m.shape}')
        return gamma_1h, gamma_15m

    print(f'  [HSMM] Training with {em_iters} EM iterations...')
    states = ['Trend+', 'Range', 'Trend-', 'Squeeze', 'Distribution', 'Liquidation']

    # ---- 1H HSMM ----
    df_1h  = mtf_all['1h']
    df_4h  = mtf_all.get('4h')
    prep_1h = _prepare_features_full(df_1h, df_htf=df_4h)

    hsmm_1h = SemiMarkovHMM(states=states)
    init_data = prep_1h.dropna().tail(min(2000, len(prep_1h) // 2))
    hsmm_1h.initialize_parameters(init_data)
    # EM training
    t0 = time.time()
    for it in range(em_iters):
        hsmm_1h.fit(prep_1h.dropna().tail(5000), max_iter=1)
        if (it + 1) % 10 == 0:
            print(f'    1H EM iter {it+1}/{em_iters}  ({time.time()-t0:.1f}s)')

    obs_1h   = _build_obs_arrays(prep_1h)
    log_B_1h = _compute_log_B_from_arrays(hsmm_1h, obs_1h)
    gamma_1h = forward_streaming(hsmm_1h, log_B_1h)
    print(f'  [HSMM 1H] Done. gamma shape: {gamma_1h.shape}')

    # ---- 15M HSMM ----
    df_15m  = mtf_all['15m']
    prep_15m = _prepare_features_full(df_15m, df_htf=df_1h)

    hsmm_15m = SemiMarkovHMM(states=states)
    init_15m = prep_15m.dropna().tail(min(4000, len(prep_15m) // 2))
    hsmm_15m.initialize_parameters(init_15m)
    t0 = time.time()
    for it in range(em_iters):
        hsmm_15m.fit(prep_15m.dropna().tail(10000), max_iter=1)
        if (it + 1) % 10 == 0:
            print(f'    15M EM iter {it+1}/{em_iters}  ({time.time()-t0:.1f}s)')

    obs_15m   = _build_obs_arrays(prep_15m)
    log_B_15m = _compute_log_B_from_arrays(hsmm_15m, obs_15m)
    gamma_15m = forward_streaming(hsmm_15m, log_B_15m)
    print(f'  [HSMM 15M] Done. gamma shape: {gamma_15m.shape}')

    np.save(str(cache_1h),  gamma_1h)
    np.save(str(cache_15m), gamma_15m)
    print(f'  [HSMM] Saved gamma arrays.')
    return gamma_1h, gamma_15m


# ---------------------------------------------------------------------------
# SMC precomputation helper
# ---------------------------------------------------------------------------

def load_or_compute_smc(df_15m, cache_dir, force, orchestrator=None):
    """Compute or load SMC patterns for all 15M bars."""
    from src.core.precomputed_runner import _prepare_features_full, _precompute_smc
    from src.core.smc import SMCDetector

    cache_path = Path(cache_dir) / 'smc_all_train.pkl'
    if not force and cache_path.exists():
        print('  [SMC] Loading from cache...')
        with open(cache_path, 'rb') as f:
            d = pickle.load(f)
        return d['smc_list'], d['smc_start_iloc']

    print('  [SMC] Precomputing all 15M bars (this may take a while)...')
    smc = SMCDetector()
    prep_15m = _prepare_features_full(df_15m)
    t0 = time.time()
    smc_list, smc_start = _precompute_smc(prep_15m, smc, window=200,
                                          start_iloc=0, end_iloc=len(prep_15m))
    print(f'  [SMC] Done in {time.time()-t0:.1f}s — {len(smc_list)} patterns computed')
    with open(cache_path, 'wb') as f:
        pickle.dump({'smc_list': smc_list, 'smc_start_iloc': smc_start}, f, protocol=4)
    return smc_list, smc_start


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    force = args.force_retrain

    cache_dir = Path('data/pretrain_cache')
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Force-retrain: remove existing caches
    if force:
        print('[TRAIN] --force-retrain: removing existing caches...')
        for f in cache_dir.glob('*.pkl'):
            f.unlink()
        for f in cache_dir.glob('*.npy'):
            f.unlink()

    print('=' * 60)
    print(f'NYX ML Ecosystem Training')
    print(f'  pair      : {args.pair}')
    print(f'  train-end : {args.train_end or "full dataset"}')
    print(f'  data-dir  : {args.data_dir}')
    print(f'  em-iters  : {args.em_iters}')
    print('=' * 60)

    # ------------------------------------------------------------------
    # Step 1: Load MTF data
    # ------------------------------------------------------------------
    print('\n[1/8] Loading MTF data...')
    loader  = MTFLoader(data_dir=args.data_dir)
    mtf_all = loader.load(args.pair, tfs=['1d', '4h', '1h', '15m'])

    # Slice to training window
    if args.train_end:
        end_ts = pd.Timestamp(args.train_end)
        for tf in list(mtf_all.keys()):
            mtf_all[tf] = mtf_all[tf][mtf_all[tf].index < end_ts]
        print(f'  Sliced to <= {args.train_end}')

    df_1d  = mtf_all['1d']
    df_1h  = mtf_all['1h']
    df_15m = mtf_all['15m']

    print(f'  1D  bars: {len(df_1d)}')
    print(f'  1H  bars: {len(df_1h)}')
    print(f'  15M bars: {len(df_15m)}')

    # ------------------------------------------------------------------
    # Step 2: Train / load HSMM and get gamma arrays
    # ------------------------------------------------------------------
    print('\n[2/8] Training HSMM / loading cache...')
    gamma_1h, gamma_15m = load_or_train_hsmm(mtf_all, cache_dir, args.em_iters, force)

    # ------------------------------------------------------------------
    # Step 3: SMC precomputation on ALL data (no start/end restriction)
    # ------------------------------------------------------------------
    print('\n[3/8] Precomputing SMC patterns on ALL 15M data...')
    smc_list, smc_start_iloc = load_or_compute_smc(df_15m, cache_dir, force)

    # ------------------------------------------------------------------
    # Step 4: Train MLContextAgent
    # ------------------------------------------------------------------
    print('\n[4/8] Training MLContextAgent (1D)...')
    ctx_agent = MLContextAgent()
    ctx_result = ctx_agent.pretrain(df_1d, n_splits=args.n_splits)
    print(f'  Result: {ctx_result}')

    # Generate context_arr for downstream use
    from src.core.precomputed_runner import _precompute_context
    context_arr = _precompute_context(df_1d, trend_threshold=0.02)
    context_series_1d = pd.Series(context_arr, index=df_1d.index)

    # ------------------------------------------------------------------
    # Step 5: Train MLRegimeAgent
    # ------------------------------------------------------------------
    print('\n[5/8] Training MLRegimeAgent (1H + HSMM)...')
    reg_agent = MLRegimeAgent()
    reg_result = reg_agent.pretrain(df_1h, gamma_1h, n_splits=args.n_splits)
    print(f'  Result: {reg_result}')

    # ------------------------------------------------------------------
    # Step 6: Train MLSetupAgent
    # ------------------------------------------------------------------
    print('\n[6/8] Training MLSetupAgent (15M + HSMM + SMC)...')

    # Build context series aligned to 15M
    # Reindex 1D context to 15M using forward-fill
    ctx_1d_df = pd.DataFrame({'ctx': context_arr}, index=df_1d.index)
    ctx_15m = ctx_1d_df.reindex(df_15m.index, method='ffill')['ctx'].fillna('neutral')

    setup_agent = MLSetupAgent()
    setup_result = setup_agent.pretrain(
        df_15m, gamma_15m, smc_list, ctx_15m,
        n_splits=args.n_splits
    )
    print(f'  Result: {setup_result}')

    # ------------------------------------------------------------------
    # Step 7: Generate meta-dataset
    # ------------------------------------------------------------------
    print('\n[7/8] Generating MLOrchestrator meta-dataset...')
    orch = MLOrchestrator()
    meta_dataset = orch.generate_training_data(
        df_15m, context_arr, gamma_1h, gamma_15m,
        smc_list, smc_start_iloc=smc_start_iloc,
        warmup=200
    )
    print(f'  Meta-dataset: {len(meta_dataset)} samples')

    # ------------------------------------------------------------------
    # Step 8: Train MLOrchestrator
    # ------------------------------------------------------------------
    print('\n[8/8] Training MLOrchestrator...')
    orch_result = orch.pretrain(meta_dataset, n_splits=args.n_splits)
    print(f'  Result: {orch_result}')

    # ------------------------------------------------------------------
    # Step 9: Calibration report
    # ------------------------------------------------------------------
    print('\n[9/8] Calibration report...')
    _print_calibration_summary(ctx_agent, reg_agent, setup_agent, orch,
                                df_1d, df_1h, df_15m,
                                gamma_1h, gamma_15m, smc_list, smc_start_iloc,
                                context_arr, ctx_15m, meta_dataset)

    print('\n' + '=' * 60)
    print('Training complete.')
    print(f'Models saved to: {cache_dir}')
    print('=' * 60)


def _print_calibration_summary(ctx_agent, reg_agent, setup_agent, orch,
                                df_1d, df_1h, df_15m,
                                gamma_1h, gamma_15m, smc_list, smc_start_iloc,
                                context_arr, ctx_15m, meta_dataset):
    """Run held-out evaluation and print calibration."""
    try:
        from sklearn.metrics import roc_auc_score

        # Context calibration
        if ctx_agent._trained:
            X_ctx = ctx_agent.compute_features(df_1d)
            y_ctx = ctx_agent.create_labels(df_1d)
            mask  = X_ctx.notna().all(axis=1) & y_ctx.notna()
            mask.iloc[-5:] = False
            X_ctx, y_ctx = X_ctx[mask], y_ctx[mask]
            # Use last 20% as pseudo-holdout
            n = len(X_ctx)
            split = int(n * 0.8)
            proba_bull = ctx_agent._lgb_bull.predict_proba(X_ctx.iloc[split:])[:, 1]
            y_bull = (y_ctx.iloc[split:] == 1).astype(int)
            if y_bull.sum() > 5:
                auc = roc_auc_score(y_bull, proba_bull)
                print(f'\n  [MLContextAgent] Holdout AUC (bull): {auc:.4f}')
                calibration_report('MLContextAgent-bull', proba_bull, y_bull.values)

        # Regime calibration
        if reg_agent._trained:
            X_reg = reg_agent.compute_features(df_1h, gamma_1h)
            y_reg = reg_agent.create_labels(df_1h)
            mask  = X_reg.notna().all(axis=1) & y_reg.notna()
            mask.iloc[-4:] = False
            X_reg, y_reg = X_reg[mask], y_reg[mask]
            n = len(X_reg)
            split = int(n * 0.8)
            proba_reg = reg_agent._lgb.predict_proba(X_reg.iloc[split:])[:, 1]
            y_reg_ho  = y_reg.iloc[split:].values
            if y_reg_ho.sum() > 5:
                auc = roc_auc_score(y_reg_ho, proba_reg)
                print(f'\n  [MLRegimeAgent] Holdout AUC: {auc:.4f}')
                calibration_report('MLRegimeAgent', proba_reg, y_reg_ho)

        # Orchestrator calibration
        if orch._trained and len(meta_dataset) > 100:
            records = [fd for fd, _, _ in meta_dataset]
            labels  = [lbl for _, lbl, _ in meta_dataset]
            X_orch  = pd.DataFrame(records)[orch._feat_names].dropna()
            y_orch  = np.array(labels[:len(X_orch)])
            n = len(X_orch)
            split = int(n * 0.8)
            proba_orch = orch._lgb.predict_proba(X_orch.iloc[split:])[:, 1]
            y_orch_ho  = y_orch[split:]
            if y_orch_ho.sum() > 5:
                auc = roc_auc_score(y_orch_ho, proba_orch)
                print(f'\n  [MLOrchestrator] Holdout AUC: {auc:.4f}')
                calibration_report('MLOrchestrator', proba_orch, y_orch_ho)

    except Exception as e:
        print(f'  Calibration report failed: {e}')


if __name__ == '__main__':
    main()
