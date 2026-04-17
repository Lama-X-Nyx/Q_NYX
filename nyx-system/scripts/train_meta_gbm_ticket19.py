"""
Ticket 19 — Build and validate Meta-GBM on top of ML fractal agents.

Train a DEDICATED Meta-GBM on ~25 features derived ONLY from the
4 Jesse ML-native agent reports (meta_* schema from Phase A1).
This is architecturally different from Ticket 17 which injected
rep_* INTO the existing 173-feature GBM (rejected : -12 % Sharpe).

Hypothesis : a SMALL focused model on fractal probabilities might
capture COMBINATIONS of multi-TF opinions that the big GBM misses.

Training :
  - Same BTC 2020-2022 candidates (from NYXEngine._generate_candidates
    via EdgeStrategy hard gate)
  - Same binary label (outcome_net > 0 → 1, else 0)
  - Features : ONLY the 25 meta_* (derived from rep_* already
    computed per candidate in NYXEngine.run)

OOS : 2023, compare vs Ticket 11 baseline (Sharpe 9.96).

Usage :
  python scripts/train_meta_gbm_ticket19.py

Writes :
  models/BTCUSDT/meta_gbm_model.pkl
  models/BTCUSDT/meta_feature_names.json
  models/BTCUSDT/meta_training_metadata.json
  reports/BTCUSDT_meta_gbm_ticket19.json
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

DATA_DIR = HERE / 'data' / 'raw' / 'mtf'
FEAT_DIR = HERE / 'data' / 'features'
MODEL_DIR = HERE / 'models' / 'BTCUSDT'
REPORT_DIR = HERE / 'reports'

# -----------------------------------------------------------------------
# Phase A — meta_* feature schema (derived from rep_*)
# -----------------------------------------------------------------------

META_FEATURE_NAMES = [
    # Context (1D)
    'meta_ctx_p_bull', 'meta_ctx_p_bear', 'meta_ctx_p_neutral',
    'meta_ctx_conf',
    # Regime (4H)
    'meta_reg_p_trend_plus', 'meta_reg_p_trend_minus',
    'meta_reg_p_range', 'meta_reg_p_squeeze', 'meta_reg_conf',
    # Setup (1H)
    'meta_setup_p_valid', 'meta_setup_quality', 'meta_setup_conf',
    # Entry (15M)
    'meta_entry_p_up', 'meta_entry_p_down', 'meta_entry_p_neutral',
    'meta_entry_conf',
    # Cross-agent
    'meta_agreement_mean', 'meta_agreement_std', 'meta_disagreement',
    'meta_ctx_reg_alignment', 'meta_setup_entry_alignment',
    'meta_regime_setup_alignment',
]


def build_meta_features(cand_features: Dict[str, float]) -> Dict[str, float]:
    """Extract the ~22 meta_* features from a candidate's rep_* block.

    The rep_* features are already computed per candidate by
    NYXEngine._build_fractal_report_features. This function MAPS
    them to the meta_* schema and adds 3 alignment cross-features.
    """
    f = cand_features
    ctx_bull = float(f.get('rep_ctx_p_bull', 0.5))
    ctx_bear = float(f.get('rep_ctx_p_bear', 0.5))
    ctx_conf = float(f.get('rep_ctx_score', 0.5))

    reg_trend_plus = float(f.get('rep_regime_trend_plus', 0.0))
    reg_trend_minus = float(f.get('rep_regime_trend_minus', 0.0))
    reg_range = float(f.get('rep_regime_range', 1.0))
    reg_conf = float(f.get('rep_regime_score', 0.5))

    setup_p = float(f.get('rep_setup_prob', 0.5))
    setup_q = float(f.get('rep_setup_score', 0.5))
    setup_conf = float(f.get('rep_setup_score', 0.5))

    entry_up = float(f.get('rep_entry_p_up', 0.5))
    entry_down = float(f.get('rep_entry_p_down', 0.5))
    entry_conf = float(f.get('rep_entry_score', 0.5))

    agree_mean = float(f.get('rep_agreement_mean', 0.5))
    agree_std = float(f.get('rep_agreement_std', 0.0))
    disagree = float(f.get('rep_disagreement', 1.0))

    return {
        'meta_ctx_p_bull': ctx_bull,
        'meta_ctx_p_bear': ctx_bear,
        'meta_ctx_p_neutral': max(0.0, 1.0 - ctx_bull - ctx_bear),
        'meta_ctx_conf': ctx_conf,
        'meta_reg_p_trend_plus': reg_trend_plus,
        'meta_reg_p_trend_minus': reg_trend_minus,
        'meta_reg_p_range': reg_range,
        'meta_reg_p_squeeze': max(0.0, 1.0 - reg_trend_plus - reg_trend_minus - reg_range),
        'meta_reg_conf': reg_conf,
        'meta_setup_p_valid': setup_p,
        'meta_setup_quality': setup_q,
        'meta_setup_conf': setup_conf,
        'meta_entry_p_up': entry_up,
        'meta_entry_p_down': entry_down,
        'meta_entry_p_neutral': max(0.0, 1.0 - entry_up - entry_down),
        'meta_entry_conf': entry_conf,
        'meta_agreement_mean': agree_mean,
        'meta_agreement_std': agree_std,
        'meta_disagreement': disagree,
        'meta_ctx_reg_alignment': ctx_conf * reg_conf,
        'meta_setup_entry_alignment': setup_conf * entry_conf,
        'meta_regime_setup_alignment': reg_conf * setup_conf,
    }


# -----------------------------------------------------------------------
# Data loading (same as train_btc_model.py)
# -----------------------------------------------------------------------

def _load(tf: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f'BTCUSDT_{tf}.csv')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    for col in ('open', 'high', 'low', 'close'):
        df = df[df[col] > 0]
    return df


# -----------------------------------------------------------------------
def main() -> int:
    print('=== Ticket 19 — Meta-GBM on ML fractal agents ===', flush=True)
    mtf = {tf: _load(tf) for tf in ('15m', '1h', '4h', '1d')}
    feats = {
        tf: pd.read_parquet(FEAT_DIR / f'BTCUSDT_features_{tf}.parquet')
        for tf in ('15m', '1h', '4h', '1d')
    }

    # Phase A3 — generate candidates with rep_* via NYXEngine.
    from src.core.nyx_engine import NYXEngine
    engine = NYXEngine()
    result_train = engine.run(
        mtf, feats, train_end='2022-12-31',
        test_start='2020-01-01', test_end='2022-12-31',
    )
    train_trades = result_train.get('trades', [])
    print(f'  Train window candidates (trades): {len(train_trades)}', flush=True)

    # Actually we need ALL candidates (not just filtered trades).
    # Re-generate candidates directly.
    ctx_1d = engine._build_1d_context(mtf['1d'], feats.get('1d', pd.DataFrame()))
    ctx_1h = engine._build_1h_context(mtf['1h'], feats.get('1h', pd.DataFrame()))
    train_cands = engine._generate_candidates(
        df_15m=mtf['15m'].loc[:'2022-12-31'],
        feat_15m=feats['15m'].loc[:'2022-12-31'],
        ctx_1d=ctx_1d, ctx_1h=ctx_1h,
        feat_1h=feats.get('1h'), feat_1d=feats.get('1d'),
        feat_4h=feats.get('4h'), mtf_data=mtf,
    )
    test_cands = engine._generate_candidates(
        df_15m=mtf['15m'].loc['2023-01-01':'2023-12-31'],
        feat_15m=feats['15m'].loc['2023-01-01':'2023-12-31'],
        ctx_1d=ctx_1d, ctx_1h=ctx_1h,
        feat_1h=feats.get('1h'), feat_1d=feats.get('1d'),
        feat_4h=feats.get('4h'), mtf_data=mtf,
    )
    print(f'  Train candidates: {len(train_cands)}', flush=True)
    print(f'  Test candidates:  {len(test_cands)}', flush=True)

    if len(train_cands) < 30:
        print('ERROR: not enough train candidates')
        return 1

    # Phase B — build meta features + train.
    def _extract_X_y(cands):
        X_rows = []
        y = []
        for c in cands:
            meta = build_meta_features(c['features'])
            row = [meta.get(n, 0.0) for n in META_FEATURE_NAMES]
            X_rows.append(row)
            y.append(1 if c['outcome_net'] > 0 else 0)
        return np.array(X_rows, dtype=float), np.array(y, dtype=int)

    X_train, y_train = _extract_X_y(train_cands)
    X_test, y_test = _extract_X_y(test_cands)
    print(f'  X_train shape: {X_train.shape}', flush=True)
    print(f'  y_train balance: {y_train.mean():.3f}', flush=True)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        min_samples_leaf=20, subsample=0.8, random_state=42,
    )
    t0 = time.time()
    model.fit(X_tr_s, y_train)
    train_elapsed = time.time() - t0
    print(f'  Training elapsed: {train_elapsed:.1f}s', flush=True)

    from sklearn.metrics import accuracy_score
    train_acc = accuracy_score(y_train, model.predict(X_tr_s))
    print(f'  In-sample accuracy: {train_acc:.3f}', flush=True)

    # Save artefacts.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / 'meta_gbm_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(MODEL_DIR / 'meta_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    (MODEL_DIR / 'meta_feature_names.json').write_text(
        json.dumps(META_FEATURE_NAMES, indent=2)
    )
    meta_meta = {
        'symbol': 'BTCUSDT',
        'train_end': '2022-12-31',
        'n_train_candidates': len(train_cands),
        'n_features': len(META_FEATURE_NAMES),
        'in_sample_accuracy': float(train_acc),
        'positive_class_rate': float(y_train.mean()),
    }
    (MODEL_DIR / 'meta_training_metadata.json').write_text(
        json.dumps(meta_meta, indent=2)
    )

    # Phase C — OOS.
    proba = model.predict_proba(X_te_s)
    classes = list(model.classes_)
    p1_idx = classes.index(1) if 1 in classes else 0
    scores = proba[:, p1_idx]

    # Apply same threshold + cooldown as baseline.
    threshold = 0.60
    cooldown_bars = 32
    capital = 10_000.0
    trades = []
    last_bar = -999
    for i, (c, score) in enumerate(zip(test_cands, scores)):
        if score < threshold:
            continue
        if c['bar_idx'] - last_bar < cooldown_bars:
            continue
        capital += c['outcome_net']
        trades.append({
            'timestamp': str(c['timestamp']),
            'direction': c['direction'],
            'net_pnl': c['outcome_net'],
            'ml_score': float(score),
        })
        last_bar = c['bar_idx']

    n_trades = len(trades)
    total_pnl = capital - 10_000.0
    wins = [t for t in trades if t['net_pnl'] > 0]
    losses = [t for t in trades if t['net_pnl'] <= 0]
    wr = len(wins) / n_trades if n_trades > 0 else 0
    gw = sum(t['net_pnl'] for t in wins)
    gl = abs(sum(t['net_pnl'] for t in losses))
    pf = gw / gl if gl > 0 else float('inf')

    eq = [10_000.0]
    for t in trades:
        eq.append(eq[-1] + t['net_pnl'])
    eq_arr = np.array(eq)
    peak = np.maximum.accumulate(eq_arr)
    dd = (peak - eq_arr) / peak
    max_dd = float(dd.max())

    if n_trades > 5:
        rets = [t['net_pnl'] / 10_000.0 for t in trades]
        sharpe = float(np.mean(rets) / max(np.std(rets), 1e-12) * np.sqrt(min(n_trades, 252)))
    else:
        sharpe = 0.0

    oos = {
        'n_trades': n_trades,
        'win_rate': wr,
        'sharpe': sharpe,
        'total_pnl_dollars': total_pnl,
        'return_pct': total_pnl / 10_000.0,
        'max_drawdown_pct': max_dd,
        'profit_factor': pf,
    }
    print(f'  OOS 2023: {oos}', flush=True)

    # Baseline comparison.
    baseline = {
        'n_trades': 54, 'sharpe': 9.96, 'total_pnl_dollars': 2171.18,
        'max_drawdown_pct': 0.0037,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        'ticket': 'Ticket 19 — Meta-GBM on ML fractal agents',
        'meta_features': META_FEATURE_NAMES,
        'training': meta_meta,
        'oos_2023': oos,
        'baseline_ticket11': baseline,
        'decision': None,  # filled below
    }

    if sharpe >= baseline['sharpe'] * 0.95 and max_dd <= baseline['max_drawdown_pct'] * 1.5:
        report['decision'] = 'KEEP'
    elif sharpe >= baseline['sharpe'] * 0.80:
        report['decision'] = 'HYBRID'
    else:
        report['decision'] = 'REJECT'

    print(f'\n  DECISION: {report["decision"]}', flush=True)
    print(f'  Baseline Sharpe: {baseline["sharpe"]}', flush=True)
    print(f'  Meta-GBM Sharpe: {sharpe:.2f}', flush=True)

    (REPORT_DIR / 'BTCUSDT_meta_gbm_ticket19.json').write_text(
        json.dumps(report, indent=2, default=str)
    )
    print(f'\nReport: reports/BTCUSDT_meta_gbm_ticket19.json', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
