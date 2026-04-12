"""
Triple Barrier Labeler — Jesse ML Pipeline

Labels each bar with:
  +1  if Take Profit is hit first (bullish)
  -1  if Stop Loss is hit first (bearish)
   0  if neither is hit within max_bars (range / time expired)

Based on "Advances in Financial Machine Learning" by Marcos Lopez de Prado.
Uses ATR for dynamic barrier sizing.
"""
import numpy as np
import pandas as pd


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 period: int = 14) -> np.ndarray:
    """Compute Average True Range (ATR) without Jesse dependency."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def triple_barrier_labels(
    df: pd.DataFrame,
    tp_mult: float = 1.5,
    sl_mult: float = 1.0,
    max_bars: int = 50,
    atr_period: int = 14,
) -> np.ndarray:
    """
    Apply triple barrier method to label each bar.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close' columns.
        tp_mult: ATR multiplier for take-profit barrier.
        sl_mult: ATR multiplier for stop-loss barrier.
        max_bars: Maximum bars before time expiry (vertical barrier).
        atr_period: ATR lookback period.

    Returns:
        np.ndarray of labels: +1, -1, or 0 for each bar.
    """
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    n = len(close)

    atr = _compute_atr(high, low, close, period=atr_period)
    labels = np.zeros(n, dtype=int)

    for i in range(n):
        if np.isnan(atr[i]) or atr[i] <= 0:
            labels[i] = 0
            continue

        entry_price = close[i]
        tp_level = entry_price + tp_mult * atr[i]
        sl_level = entry_price - sl_mult * atr[i]

        label = 0  # default: time expired
        for j in range(i + 1, min(i + max_bars + 1, n)):
            if high[j] >= tp_level:
                label = 1   # TP hit first → bullish
                break
            if low[j] <= sl_level:
                label = -1  # SL hit first → bearish
                break

        labels[i] = label

    return labels
