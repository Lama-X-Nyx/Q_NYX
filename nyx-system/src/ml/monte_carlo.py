"""
Monte Carlo Stress Test — Validate edge is real, not artefact

3 stress tests:
  1. Trade shuffle: permute trade order → equity distribution
  2. Candle noise: perturb OHLCV → does edge survive noise?
  3. Fee/slippage stress: increase costs → where does edge break?

If the edge survives all 3 → ready for paper trading.
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from src.ml.nyx_pipeline import NYXPipeline


# ===================================================================
# 1. TRADE SHUFFLE MONTE CARLO
# ===================================================================

def trade_shuffle_mc(
    trades: List[Dict],
    n_sims: int = 500,
    initial_capital: float = 10_000.0,
) -> Dict[str, Any]:
    """
    Shuffle trade order N times, compute equity curve each time.
    Tests if the edge depends on lucky sequencing.
    """
    if not trades:
        return {'pct_profitable': 0, 'median_return': 0, 'dd_95th': 1.0,
                'returns': [], 'max_dds': []}

    pnls = np.array([t['net_pnl'] for t in trades])
    n_trades = len(pnls)

    returns = []
    max_dds = []

    for _ in range(n_sims):
        shuffled = np.random.permutation(pnls)
        equity = initial_capital + np.cumsum(shuffled)
        equity = np.insert(equity, 0, initial_capital)

        final_return = (equity[-1] - initial_capital) / initial_capital
        returns.append(final_return)

        # Max drawdown
        peak = initial_capital
        max_dd = 0.0
        for e in equity:
            peak = max(peak, e)
            dd = (peak - e) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        max_dds.append(max_dd)

    returns = np.array(returns)
    max_dds = np.array(max_dds)

    return {
        'n_sims': n_sims,
        'n_trades': n_trades,
        'pct_profitable': float((returns > 0).mean()),
        'median_return': float(np.median(returns)),
        'mean_return': float(np.mean(returns)),
        'std_return': float(np.std(returns)),
        'ci_5': float(np.percentile(returns, 5)),
        'ci_95': float(np.percentile(returns, 95)),
        'dd_median': float(np.median(max_dds)),
        'dd_95th': float(np.percentile(max_dds, 95)),
        'dd_max': float(np.max(max_dds)),
        'returns': returns.tolist(),
        'max_dds': max_dds.tolist(),
    }


# ===================================================================
# 2. CANDLE NOISE MONTE CARLO
# ===================================================================

def candle_noise_mc(
    mtf_data: Dict[str, pd.DataFrame],
    mtf_features: Dict[str, pd.DataFrame],
    train_end: str,
    test_start: str,
    test_end: str,
    noise_pct: float = 0.001,
    n_sims: int = 20,
) -> Dict[str, Any]:
    """
    Add random noise to 15m candles, re-run pipeline each time.
    Tests if edge is robust to data perturbation.
    """
    returns = []
    sharpes = []

    for sim in range(n_sims):
        # Perturb 15m test data
        df_15m = mtf_data['15m'].copy()
        test_mask = (df_15m.index >= test_start) & (df_15m.index <= test_end)
        n_test = test_mask.sum()

        np.random.seed(sim)
        for col in ['open', 'high', 'low', 'close']:
            noise = 1.0 + np.random.randn(len(df_15m)) * noise_pct
            df_15m.loc[:, col] = df_15m[col] * noise

        # Fix OHLC consistency
        df_15m['high'] = df_15m[['open', 'high', 'close']].max(axis=1)
        df_15m['low'] = df_15m[['open', 'low', 'close']].min(axis=1)

        noisy_data = {**mtf_data, '15m': df_15m}

        pipe = NYXPipeline()
        r = pipe.run(noisy_data, mtf_features,
                     train_end=train_end, test_start=test_start, test_end=test_end)

        returns.append(r['total_return_pct'])
        sharpes.append(r['sharpe'])

    returns = np.array(returns)
    sharpes = np.array(sharpes)

    return {
        'n_sims': n_sims,
        'noise_pct': noise_pct,
        'pct_profitable': float((returns > 0).mean()),
        'median_return': float(np.median(returns)),
        'mean_return': float(np.mean(returns)),
        'median_sharpe': float(np.median(sharpes)),
        'returns': returns.tolist(),
        'sharpes': sharpes.tolist(),
    }


# ===================================================================
# 3. FULL MONTE CARLO REPORT
# ===================================================================

def full_monte_carlo_report(
    trades: List[Dict],
    mtf_data: Dict[str, pd.DataFrame],
    mtf_features: Dict[str, pd.DataFrame],
    train_end: str,
    test_start: str,
    test_end: str,
) -> Dict[str, Any]:
    """
    Complete Monte Carlo stress test report.
    """
    # 1. Trade shuffle
    shuffle_result = trade_shuffle_mc(trades, n_sims=500)

    # 2. Candle noise
    noise_result = candle_noise_mc(
        mtf_data, mtf_features, train_end, test_start, test_end,
        noise_pct=0.001, n_sims=15,
    )

    # 3. Fee stress
    fee_results = {}
    for fee_label, fee_rate, slip_rate in [
        ('maker_1x', 0.0002, 0.0001),
        ('maker_2x', 0.0004, 0.0002),
        ('taker', 0.0004, 0.0003),
    ]:
        pipe = NYXPipeline(fee_rate=fee_rate, slippage_rate=slip_rate)
        r = pipe.run(mtf_data, mtf_features, train_end=train_end,
                     test_start=test_start, test_end=test_end)
        fee_results[fee_label] = {
            'pnl': r['total_pnl_dollars'],
            'sharpe': r['sharpe'],
            'win_rate': r['win_rate'],
            'n_trades': r['n_trades'],
        }

    # 4. Confidence interval from shuffle
    returns = np.array(shuffle_result['returns'])

    return {
        'trade_shuffle': {
            'n_sims': shuffle_result['n_sims'],
            'pct_profitable': shuffle_result['pct_profitable'],
            'median_return': shuffle_result['median_return'],
            'dd_95th': shuffle_result['dd_95th'],
        },
        'candle_noise': {
            'noise_pct': noise_result['noise_pct'],
            'n_sims': noise_result['n_sims'],
            'pct_profitable': noise_result['pct_profitable'],
            'median_return': noise_result['median_return'],
            'median_sharpe': noise_result['median_sharpe'],
        },
        'fee_stress': fee_results,
        'confidence_interval': {
            'ci_95_low': float(np.percentile(returns, 2.5)),
            'ci_95_high': float(np.percentile(returns, 97.5)),
            'ci_50_low': float(np.percentile(returns, 25)),
            'ci_50_high': float(np.percentile(returns, 75)),
            'median': float(np.median(returns)),
        },
    }
