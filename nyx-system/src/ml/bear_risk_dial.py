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
from src.ml.nyx_pipeline import NYXPipeline


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
        df_1d = mtf_data.get('1d', pd.DataFrame())
        df_15m = mtf_data['15m']
        te = test_end or str(df_15m.index[-1].date())

        # Detect regime MONTHLY (not just at start)
        # Use the most conservative regime detected in the test period
        test_months = pd.date_range(test_start, te, freq='MS')
        regimes_detected = []
        for month_start in test_months:
            reg = detect_regime(df_1d, str(month_start.date()))
            regimes_detected.append(reg)

        # Use worst regime to set risk (conservative)
        if 'bear' in regimes_detected:
            dominant_regime = 'bear'
        elif 'range' in regimes_detected and 'bull' in regimes_detected:
            # Mixed: count which is more frequent
            bull_count = regimes_detected.count('bull')
            range_count = regimes_detected.count('range')
            dominant_regime = 'bull' if bull_count > range_count else 'range'
        elif 'bull' in regimes_detected:
            dominant_regime = 'bull'
        else:
            dominant_regime = 'range'

        params = get_risk_params(dominant_regime)

        pipe = NYXPipeline(
            risk_pct=params['risk_pct'],
            ml_threshold=params['ml_threshold'],
            cooldown_bars=params['cooldown_bars'],
            max_daily_trades=params['max_daily_trades'],
            fee_rate=self.fee_rate,
            slippage_rate=self.slippage_rate,
        )

        result = pipe.run(mtf_data, mtf_features,
                          train_end=train_end, test_start=test_start, test_end=te)

        result['detected_regime'] = dominant_regime
        result['regimes_monthly'] = regimes_detected
        result['risk_params'] = params

        return result
