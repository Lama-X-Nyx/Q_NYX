"""
Threshold Optimizer — Find ML threshold that maximizes Sharpe

Sweeps thresholds on validation data, picks the one with best Sharpe
while maintaining minimum trade count for statistical significance.
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from src.ml.ml_filter_v2 import (
    EnhancedMLFilter, generate_enhanced_candidates,
)
from src.ml.soft_gate import compute_size_factor


def _simulate_at_threshold(
    scored: List[tuple],
    threshold: float,
    initial_capital: float = 10_000.0,
    risk_pct: float = 0.02,
    cooldown_bars: int = 32,
) -> Dict[str, Any]:
    """Simulate backtest at a given threshold on pre-scored candidates."""
    accepted = [(c, p) for c, p in scored if p >= threshold]

    capital = initial_capital
    trades = []
    last_bar = -999

    for c, prob in accepted:
        if c['bar_idx'] - last_bar < cooldown_bars:
            continue
        atr_est = max(c['entry_price'] * c['features'].get('rule_regime', 0.005) / 200, 1)
        dis = c['features'].get('disagreement', 0)
        ravg = np.mean([c['features'].get(f'rule_{x}', 0.5) for x in ['context', 'regime', 'setup']])
        sf = compute_size_factor(prob, dis, ravg)
        qty = min(capital * risk_pct * sf / max(atr_est, 1), capital / max(c['entry_price'], 1))
        if qty <= 0 or capital <= 0:
            continue
        pnl = c['outcome_net'] * qty
        capital += pnl
        capital = max(capital, 0)
        last_bar = c['bar_idx']
        trades.append({'pnl': pnl, 'win': pnl > 0, 'prob': prob})

    nt = len(trades)
    total_pnl = capital - initial_capital
    wr = sum(1 for t in trades if t['win']) / nt if nt > 0 else 0

    # Sharpe
    if nt > 5:
        rets = [t['pnl'] / initial_capital for t in trades]
        sharpe = float(np.mean(rets) / max(np.std(rets), 1e-8) * np.sqrt(min(nt, 252)))
    else:
        sharpe = 0.0

    # Max DD
    eq = initial_capital
    peak = initial_capital
    max_dd = 0
    for t in trades:
        eq += t['pnl']
        peak = max(peak, eq)
        dd = (peak - eq) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    gw = sum(t['pnl'] for t in trades if t['win'])
    gl = abs(sum(t['pnl'] for t in trades if not t['win']))

    return {
        'threshold': threshold,
        'n_trades': nt,
        'win_rate': wr,
        'sharpe': sharpe,
        'total_pnl': total_pnl,
        'max_drawdown_pct': max_dd,
        'profit_factor': gw / gl if gl > 0 else float('inf'),
        'reject_rate': 1 - len(accepted) / max(len(scored), 1),
    }


def optimize_threshold(
    df_ohlcv: pd.DataFrame,
    df_features: pd.DataFrame,
    train_end: str,
    val_start: str,
    val_end: str,
    min_trades: int = 10,
    thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Find optimal threshold by sweeping on validation data.

    Returns: best_threshold, best_sharpe, full sweep results.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.48, 0.68, 0.02)]

    # Train ML filter
    ml = EnhancedMLFilter()
    ml.train(df_ohlcv.loc[:train_end], df_features.loc[:train_end])

    # Score validation candidates
    candidates = generate_enhanced_candidates(
        df_ohlcv.loc[val_start:val_end],
        df_features.loc[val_start:val_end],
    )
    scored = [(c, ml.predict_proba(c['features'])) for c in candidates]

    # Sweep
    sweep = []
    best_sharpe = -999
    best_threshold = 0.50

    for thresh in thresholds:
        result = _simulate_at_threshold(scored, thresh)
        sweep.append(result)
        if result['n_trades'] >= min_trades and result['sharpe'] > best_sharpe:
            best_sharpe = result['sharpe']
            best_threshold = thresh

    # If no threshold meets min_trades, pick the one with most trades
    if best_sharpe == -999:
        valid = [s for s in sweep if s['n_trades'] > 0]
        if valid:
            best = max(valid, key=lambda s: s['sharpe'])
            best_threshold = best['threshold']
            best_sharpe = best['sharpe']

    best_result = next((s for s in sweep if s['threshold'] == best_threshold), sweep[0])

    return {
        'best_threshold': best_threshold,
        'best_sharpe': best_sharpe,
        'best_n_trades': best_result['n_trades'],
        'best_win_rate': best_result['win_rate'],
        'best_pnl': best_result['total_pnl'],
        'best_max_dd': best_result['max_drawdown_pct'],
        'sweep': sweep,
    }


def run_optimal_backtest(
    df_ohlcv: pd.DataFrame,
    df_features: pd.DataFrame,
    train_end: str = '2022-12-31',
    test_start: str = '2023-01-01',
    test_end: Optional[str] = None,
    min_trades: int = 10,
) -> Dict[str, Any]:
    """
    Train ML, optimize threshold on train tail, test on OOS.

    Uses last 20% of train as validation for threshold calibration.
    """
    # Split train into train_core + val for threshold tuning
    train_df = df_ohlcv.loc[:train_end]
    n_train = len(train_df)
    val_split = train_df.index[int(n_train * 0.8)]

    train_core_end = str((val_split - pd.Timedelta(days=1)).date())
    val_start_str = str(val_split.date())

    # Optimize threshold on validation portion
    opt = optimize_threshold(
        df_ohlcv, df_features,
        train_end=train_core_end,
        val_start=val_start_str,
        val_end=train_end,
        min_trades=min_trades,
    )
    threshold = opt['best_threshold']

    # Now train final model on FULL train data
    ml = EnhancedMLFilter()
    ml.train(df_ohlcv.loc[:train_end], df_features.loc[:train_end])

    # Generate and score test candidates
    te = test_end or str(df_ohlcv.index[-1].date())
    candidates = generate_enhanced_candidates(
        df_ohlcv.loc[test_start:te],
        df_features.loc[test_start:te],
    )
    scored = [(c, ml.predict_proba(c['features'])) for c in candidates]

    # Simulate at optimal threshold
    result = _simulate_at_threshold(scored, threshold)

    return {
        'n_trades': result['n_trades'],
        'win_rate': result['win_rate'],
        'sharpe': result['sharpe'],
        'total_pnl_dollars': result['total_pnl'],
        'max_drawdown_pct': result['max_drawdown_pct'],
        'profit_factor': result['profit_factor'],
        'optimized_threshold': threshold,
        'threshold_source': f'validated on {val_start_str} to {train_end}',
        'feature_importance': ml.feature_importance(),
    }
