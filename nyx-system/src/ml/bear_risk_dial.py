"""
Bear Risk Dial — Adaptive risk parameters by market regime

When bootstrap shows bear is fragile (P(loss)=10.2%):
  - Reduce position size
  - Raise ML threshold (be more selective)
  - Increase cooldown (trade less)

Regime detected from 1D EMA50/EMA200 crossover.
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
from src.ml.jesse_features import _ema


# ===================================================================
# REGIME DETECTION
# ===================================================================

def detect_regime(df_1d: pd.DataFrame, current_date: str) -> str:
    """
    Detect market regime from daily data.
    Uses EMA50/EMA200 crossover + 60-day momentum.

    Returns: 'bull', 'bear', or 'range'
    """
    df = df_1d.loc[:current_date]
    if len(df) < 60:
        return 'range'

    close = df['close'].values.astype(float)
    ema50 = _ema(close, 50)
    ema200 = _ema(close, min(200, len(close) - 1))

    if np.isnan(ema50[-1]) or np.isnan(ema200[-1]):
        return 'range'

    # 60-day momentum
    if len(close) >= 60:
        mom60 = (close[-1] - close[-60]) / close[-60]
    else:
        mom60 = 0

    # EMA cross + momentum for regime
    ema_bull = ema50[-1] > ema200[-1]
    ema_bear = ema50[-1] < ema200[-1]

    if ema_bull and mom60 > 0.05:
        return 'bull'
    elif ema_bear and mom60 < -0.05:
        return 'bear'
    else:
        return 'range'


# ===================================================================
# RISK PARAMETERS BY REGIME
# ===================================================================

RISK_PARAMS = {
    'bull': {
        'risk_pct': 0.02,          # 2% risk per trade
        'ml_threshold': 0.60,      # standard threshold
        'cooldown_bars': 32,       # 8 hours cooldown
        'max_daily_trades': 1,
        'size_mult': 1.0,          # full size
    },
    'range': {
        'risk_pct': 0.015,         # 1.5% risk
        'ml_threshold': 0.63,      # slightly stricter
        'cooldown_bars': 48,       # 12 hours cooldown
        'max_daily_trades': 1,
        'size_mult': 0.75,         # 75% size
    },
    'bear': {
        'risk_pct': 0.01,          # 1% risk — half of bull
        'ml_threshold': 0.68,      # much stricter — only high conviction
        'cooldown_bars': 64,       # 16 hours — trade much less
        'max_daily_trades': 1,
        'size_mult': 0.5,          # 50% size
    },
}


def get_risk_params(regime: str) -> Dict[str, Any]:
    """Get risk parameters for a given regime."""
    return RISK_PARAMS.get(regime, RISK_PARAMS['range']).copy()


# ===================================================================
# ADAPTIVE PIPELINE
# ===================================================================

class AdaptiveNYXPipeline:
    """
    NYX Pipeline with bear risk dial.
    Detects regime from 1D data and adjusts risk parameters dynamically.
    """

    def __init__(self):
        self.fee_rate = 0.0002
        self.slippage_rate = 0.0001

    def run(
        self,
        mtf_data: Dict[str, pd.DataFrame],
        mtf_features: Dict[str, pd.DataFrame],
        train_end: str,
        test_start: str,
        test_end: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run pipeline with per-trade regime detection.
        Each candidate gets its own risk params based on the regime
        AT THAT MOMENT — not a single regime for the whole period.
        """
        df_1d = mtf_data.get('1d', pd.DataFrame())
        te = test_end or str(mtf_data['15m'].index[-1].date())

        # Step 1: run base pipeline to get candidates + ML scores
        # Use bull params (most permissive) to generate all candidates
        from src.ml.nyx_pipeline import NYXPipeline  # lazy import
        base_pipe = NYXPipeline(
            risk_pct=RISK_PARAMS['bull']['risk_pct'],
            ml_threshold=RISK_PARAMS['bear']['ml_threshold'],  # strictest to get all scored
            cooldown_bars=RISK_PARAMS['bull']['cooldown_bars'],
            max_daily_trades=RISK_PARAMS['bull']['max_daily_trades'],
            fee_rate=self.fee_rate,
            slippage_rate=self.slippage_rate,
        )
        base_result = base_pipe.run(mtf_data, mtf_features,
                                     train_end=train_end, test_start=test_start, test_end=te)

        # Step 2: re-filter trades with per-trade regime detection
        # Build a monthly regime cache
        regime_cache = {}
        test_months = pd.date_range(test_start, te, freq='MS')
        for month_start in test_months:
            reg = detect_regime(df_1d, str(month_start.date()))
            regime_cache[month_start.to_period('M')] = reg

        # Step 3: replay trades with regime-adapted sizing
        from src.ml.soft_gate import compute_size_factor
        initial_capital = 10_000.0
        capital = initial_capital
        trades_out = []
        regimes_used = []

        for t in base_result['trades']:
            ts = t['timestamp']
            period = ts.to_period('M')
            regime = regime_cache.get(period, 'range')
            params = get_risk_params(regime)
            regimes_used.append(regime)

            # Apply regime-specific threshold filter
            if t.get('ml_score', 0) < params['ml_threshold']:
                continue

            # Apply regime-specific sizing
            size_mult = params['size_mult']
            adjusted_pnl = t['net_pnl'] * size_mult
            capital += adjusted_pnl
            capital = max(capital, 0)

            trade_out = {**t, 'regime': regime, 'size_mult': size_mult,
                         'net_pnl': adjusted_pnl}
            trades_out.append(trade_out)

        # Metrics
        nt = len(trades_out)
        total_pnl = capital - initial_capital
        wins = [t for t in trades_out if t['net_pnl'] > 0]
        losses = [t for t in trades_out if t['net_pnl'] <= 0]
        wr = len(wins) / nt if nt > 0 else 0

        gw = sum(t['net_pnl'] for t in wins)
        gl = abs(sum(t['net_pnl'] for t in losses))

        eq = [initial_capital]
        for t in trades_out:
            eq.append(eq[-1] + t['net_pnl'])
        eq_arr = np.array(eq)
        peak = eq_arr[0]; max_dd = 0
        for e in eq_arr:
            peak = max(peak, e); dd = (peak - e) / peak if peak > 0 else 0; max_dd = max(max_dd, dd)

        if nt > 5:
            rets = [t['net_pnl'] / initial_capital for t in trades_out]
            sharpe = float(np.mean(rets) / max(np.std(rets), 1e-8) * np.sqrt(min(nt, 252)))
        else:
            sharpe = 0.0

        from collections import Counter
        regime_counts = Counter(regimes_used)

        return {
            'n_trades': nt,
            'n_candidates': base_result['n_candidates'],
            'filter_reject_rate': 1 - nt / max(base_result['n_trades'], 1),
            'win_rate': wr,
            'sharpe': sharpe,
            'total_pnl_dollars': total_pnl,
            'total_return_pct': total_pnl / initial_capital,
            'max_drawdown_pct': max_dd,
            'profit_factor': gw / gl if gl > 0 else float('inf'),
            'total_fees': base_result.get('total_fees', 0),
            'detected_regime': max(regime_counts, key=regime_counts.get) if regime_counts else 'range',
            'regimes_per_trade': dict(regime_counts),
            'risk_params': 'per-trade adaptive',
            'trades': trades_out,
        }
