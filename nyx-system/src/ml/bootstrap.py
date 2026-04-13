"""
Bootstrap Stress Tests — With Replacement, Block, By Regime

Unlike shuffle (permutation), bootstrap samples WITH REPLACEMENT:
  - Some trades appear multiple times, others not at all
  - This creates real variance in the distribution
  - More conservative than shuffle (which always uses exact same trades)

3 methods:
  1. Standard: iid sample with replacement
  2. Block: contiguous blocks to preserve temporal autocorrelation
  3. Regime: stratified sampling per market regime (bull/bear/range)
"""
import numpy as np
from typing import Any, Dict, List


def _compute_stats(
    pnls_matrix: np.ndarray,
    initial_capital: float = 10_000.0,
) -> Dict[str, Any]:
    """Compute stats from (n_sims, n_trades) PnL matrix."""
    n_sims = pnls_matrix.shape[0]
    n_trades = pnls_matrix.shape[1]

    returns = []
    sharpes = []
    max_dds = []
    losing_streaks = []

    for i in range(n_sims):
        pnls = pnls_matrix[i]
        equity = initial_capital + np.cumsum(pnls)
        equity = np.insert(equity, 0, initial_capital)

        # Return
        ret = (equity[-1] - initial_capital) / initial_capital
        returns.append(ret)

        # Sharpe
        trade_rets = pnls / initial_capital
        std = np.std(trade_rets)
        sharpe = float(np.mean(trade_rets) / std * np.sqrt(min(n_trades, 252))) if std > 0 else 0
        sharpes.append(sharpe)

        # Max DD
        peak = initial_capital
        max_dd = 0.0
        for e in equity:
            peak = max(peak, e)
            dd = (peak - e) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        max_dds.append(max_dd)

        # Losing streak
        streak = 0; max_streak = 0
        for p in pnls:
            if p < 0:
                streak += 1; max_streak = max(max_streak, streak)
            else:
                streak = 0
        losing_streaks.append(max_streak)

    returns = np.array(returns)
    sharpes = np.array(sharpes)
    max_dds = np.array(max_dds)
    losing_streaks = np.array(losing_streaks)

    return {
        'n_sims': n_sims,
        'n_trades_per_sim': n_trades,
        'returns': returns.tolist(),
        'median_return': float(np.median(returns)),
        'mean_return': float(np.mean(returns)),
        'return_std': float(np.std(returns)),
        'return_p5': float(np.percentile(returns, 5)),
        'return_p10': float(np.percentile(returns, 10)),
        'return_p25': float(np.percentile(returns, 25)),
        'return_p75': float(np.percentile(returns, 75)),
        'return_p95': float(np.percentile(returns, 95)),
        'sharpe_median': float(np.median(sharpes)),
        'sharpe_mean': float(np.mean(sharpes)),
        'sharpe_std': float(np.std(sharpes)),
        'sharpe_p5': float(np.percentile(sharpes, 5)),
        'sharpe_p10': float(np.percentile(sharpes, 10)),
        'sharpe_p95': float(np.percentile(sharpes, 95)),
        'dd_median': float(np.median(max_dds)),
        'dd_p75': float(np.percentile(max_dds, 75)),
        'dd_p90': float(np.percentile(max_dds, 90)),
        'dd_p95': float(np.percentile(max_dds, 95)),
        'dd_p99': float(np.percentile(max_dds, 99)),
        'dd_worst': float(np.max(max_dds)),
        'prob_loss': float((returns < 0).mean()),
        'prob_dd_10': float((max_dds > 0.10).mean()),
        'prob_dd_20': float((max_dds > 0.20).mean()),
        'losing_streak_median': int(np.median(losing_streaks)),
        'losing_streak_p95': int(np.percentile(losing_streaks, 95)),
        'losing_streak_worst': int(np.max(losing_streaks)),
    }


# ===================================================================
# 1. STANDARD BOOTSTRAP (with replacement)
# ===================================================================

def standard_bootstrap(
    trades: List[Dict],
    n_sims: int = 2000,
    initial_capital: float = 10_000.0,
) -> Dict[str, Any]:
    """
    IID bootstrap with replacement.
    Each simulation samples N trades from the pool (with replacement).
    """
    pnls = np.array([t['net_pnl'] for t in trades])
    n = len(pnls)
    # Sample with replacement: each sim picks n trades
    indices = np.random.randint(0, n, size=(n_sims, n))
    pnls_matrix = pnls[indices]
    return _compute_stats(pnls_matrix, initial_capital)


# ===================================================================
# 2. BLOCK BOOTSTRAP
# ===================================================================

def block_bootstrap(
    trades: List[Dict],
    n_sims: int = 1000,
    block_size: int = 5,
    initial_capital: float = 10_000.0,
) -> Dict[str, Any]:
    """
    Block bootstrap: sample contiguous blocks of trades.
    Preserves temporal autocorrelation (losing streaks, momentum runs).
    """
    pnls = np.array([t['net_pnl'] for t in trades])
    n = len(pnls)
    n_blocks = max(1, n // block_size)

    pnls_matrix = np.zeros((n_sims, n_blocks * block_size))
    for sim in range(n_sims):
        blocks = []
        for _ in range(n_blocks):
            start = np.random.randint(0, max(1, n - block_size + 1))
            blocks.append(pnls[start:start + block_size])
        pnls_matrix[sim, :] = np.concatenate(blocks)[:n_blocks * block_size]

    return _compute_stats(pnls_matrix, initial_capital)


# ===================================================================
# 3. REGIME BOOTSTRAP
# ===================================================================

def regime_bootstrap(
    trades: List[Dict],
    n_sims: int = 1000,
    initial_capital: float = 10_000.0,
) -> Dict[str, Any]:
    """
    Stratified bootstrap: sample within each regime separately,
    then concatenate proportionally.
    """
    # Split by regime
    regimes: Dict[str, List[float]] = {}
    for t in trades:
        reg = t.get('regime', 'unknown')
        regimes.setdefault(reg, []).append(t['net_pnl'])

    regime_counts = {r: len(v) for r, v in regimes.items()}
    total = sum(regime_counts.values())

    # Proportional sampling per regime
    pnls_all = []
    for sim in range(n_sims):
        sim_pnls = []
        for reg, reg_pnls in regimes.items():
            reg_arr = np.array(reg_pnls)
            n_sample = max(1, int(len(reg_arr) * total / max(total, 1)))
            sampled = reg_arr[np.random.randint(0, len(reg_arr), size=n_sample)]
            sim_pnls.extend(sampled)
        # Shuffle within simulation to mix regimes
        np.random.shuffle(sim_pnls)
        pnls_all.append(sim_pnls)

    # Pad to same length
    max_len = max(len(p) for p in pnls_all)
    pnls_matrix = np.zeros((n_sims, max_len))
    for i, p in enumerate(pnls_all):
        pnls_matrix[i, :len(p)] = p

    result = _compute_stats(pnls_matrix, initial_capital)
    result['regime_counts'] = regime_counts

    # Also compute per-regime stats
    for regime_name, regime_pnls in regimes.items():
        if len(regime_pnls) < 3:
            continue
        arr = np.array(regime_pnls)
        regime_matrix = arr[np.random.randint(0, len(arr), size=(n_sims, len(arr)))]
        regime_result = _compute_stats(regime_matrix, initial_capital)
        key = f"{regime_name}_only"
        result[key] = {
            'median_return': regime_result['median_return'],
            'sharpe_median': regime_result['sharpe_median'],
            'dd_p95': regime_result['dd_p95'],
            'prob_loss': regime_result['prob_loss'],
            'n_trades': len(regime_pnls),
        }

    return result


# ===================================================================
# FULL REPORT
# ===================================================================

def full_bootstrap_report(
    trades: List[Dict],
    n_sims: int = 2000,
) -> Dict[str, Any]:
    """Complete bootstrap report with all 3 methods."""
    std = standard_bootstrap(trades, n_sims=n_sims)
    blk = block_bootstrap(trades, n_sims=n_sims, block_size=5)
    reg = regime_bootstrap(trades, n_sims=n_sims)

    # Pick the MOST CONSERVATIVE estimate across methods
    worst_p5_return = min(std['return_p5'], blk['return_p5'], reg['return_p5'])
    worst_p5_sharpe = min(std['sharpe_p5'], blk['sharpe_p5'], reg['sharpe_p5'])
    worst_dd_p95 = max(std['dd_p95'], blk['dd_p95'], reg['dd_p95'])
    worst_prob_loss = max(std['prob_loss'], blk['prob_loss'], reg['prob_loss'])

    return {
        'standard': {k: v for k, v in std.items() if k != 'returns'},
        'block': {k: v for k, v in blk.items() if k != 'returns'},
        'regime': {k: v for k, v in reg.items() if k != 'returns'},
        'institutional_summary': {
            'median_return': std['median_return'],
            'median_sharpe': std['sharpe_median'],
            'p5_return': worst_p5_return,
            'p5_sharpe': worst_p5_sharpe,
            'dd_p95': worst_dd_p95,
            'prob_loss': worst_prob_loss,
            'prob_dd_10': max(std['prob_dd_10'], blk['prob_dd_10'], reg['prob_dd_10']),
            'prob_dd_20': max(std['prob_dd_20'], blk['prob_dd_20'], reg['prob_dd_20']),
            'losing_streak_p95': max(std['losing_streak_p95'], blk['losing_streak_p95'], reg['losing_streak_p95']),
            'method': 'worst-case across standard/block/regime bootstrap',
        },
    }
