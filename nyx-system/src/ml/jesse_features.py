"""
Stationary Feature Engine — Jesse ML Pipeline

Key principle from the video:
  NEVER use raw values (price, EMA value, etc.).
  ALWAYS use RATIOS: (EMA21 - EMA50) / EMA50
  This makes features stationary and price-level independent.

When Jesse is installed, uses jesse.indicators as ta.
Falls back to pure numpy/pandas if Jesse is not available.
"""
import numpy as np
import pandas as pd
from typing import Optional

# Try importing Jesse indicators — fallback to manual computation
try:
    import jesse.indicators as ta  # type: ignore[import-untyped]
    _JESSE_AVAILABLE = True
except ImportError:
    _JESSE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Fallback indicator implementations (when Jesse not installed)
# ---------------------------------------------------------------------------

def _ema(series: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average — Wilder's method."""
    result = np.full_like(series, np.nan, dtype=float)
    if len(series) < period:
        return result
    result[period - 1] = np.mean(series[:period])
    k = 2.0 / (period + 1)
    for i in range(period, len(series)):
        result[i] = series[i] * k + result[i - 1] * (1 - k)
    return result


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index."""
    n = len(close)
    result = np.full(n, np.nan)
    if n < period + 1:
        return result
    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - 100.0 / (1.0 + rs)
    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    return result


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         period: int = 14) -> np.ndarray:
    """Average True Range."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))
    atr_arr = np.full(n, np.nan)
    if n >= period:
        atr_arr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr_arr[i] = (atr_arr[i - 1] * (period - 1) + tr[i]) / period
    return atr_arr


# ---------------------------------------------------------------------------
# Main feature computation
# ---------------------------------------------------------------------------

def compute_stationary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute stationary (ratio-based) features from OHLCV data.

    All features are ratios or normalized values — NEVER raw prices.
    This ensures the model generalizes across different price levels.

    Features:
        - ema_ratio_9_21:   (EMA9 - EMA21) / EMA21
        - ema_ratio_21_50:  (EMA21 - EMA50) / EMA50
        - close_vs_ema50:   (close - EMA50) / EMA50
        - rsi_14:           RSI(14) normalized to [-1, 1] range
        - atr_ratio:        ATR(14) / close (volatility relative to price)
        - volume_ratio:     volume / SMA(volume, 20)
        - momentum_10:      (close - close[10]) / close[10]
        - momentum_20:      (close - close[20]) / close[20]
        - high_low_ratio:   (high - low) / close (bar range)
        - close_position:   (close - low) / (high - low) (where in bar)
        - returns_1:        1-bar return
        - returns_5:        5-bar return
        - vol_change:       volume change ratio
    """
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    volume = df['volume'].values.astype(float)
    n = len(close)

    # --- EMAs ---
    if _JESSE_AVAILABLE:
        candles = np.column_stack([
            np.zeros(n),  # timestamp placeholder
            df['open'].values, high, low, close, volume
        ])
        ema9 = ta.ema(candles, 9)
        ema21 = ta.ema(candles, 21)
        ema50 = ta.ema(candles, 50)
        rsi_vals = ta.rsi(candles, 14)
        atr_vals = ta.atr(candles, 14)
    else:
        ema9 = _ema(close, 9)
        ema21 = _ema(close, 21)
        ema50 = _ema(close, 50)
        rsi_vals = _rsi(close, 14)
        atr_vals = _atr(high, low, close, 14)

    # --- Volume MA ---
    vol_ma20 = _ema(volume, 20)

    # --- Stationary features (RATIOS, not raw) ---
    features = pd.DataFrame(index=df.index)

    # EMA ratios — key trend indicators
    with np.errstate(divide='ignore', invalid='ignore'):
        features['ema_ratio_9_21'] = np.where(
            ema21 != 0, (ema9 - ema21) / np.abs(ema21), 0.0
        )
        features['ema_ratio_21_50'] = np.where(
            ema50 != 0, (ema21 - ema50) / np.abs(ema50), 0.0
        )
        features['close_vs_ema50'] = np.where(
            ema50 != 0, (close - ema50) / np.abs(ema50), 0.0
        )

    # RSI normalized to [-1, 1]
    features['rsi_14'] = (rsi_vals - 50.0) / 50.0

    # ATR ratio (volatility / price)
    with np.errstate(divide='ignore', invalid='ignore'):
        features['atr_ratio'] = np.where(close != 0, atr_vals / close, 0.0)

    # Volume ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        features['volume_ratio'] = np.where(
            vol_ma20 != 0, volume / vol_ma20, 1.0
        )

    # Momentum (returns over N bars)
    for lookback in [10, 20]:
        shifted = np.roll(close, lookback)
        shifted[:lookback] = np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            features[f'momentum_{lookback}'] = np.where(
                shifted != 0, (close - shifted) / np.abs(shifted), 0.0
            )

    # Bar structure ratios
    bar_range = high - low
    with np.errstate(divide='ignore', invalid='ignore'):
        features['high_low_ratio'] = np.where(close != 0, bar_range / close, 0.0)
        features['close_position'] = np.where(
            bar_range != 0, (close - low) / bar_range, 0.5
        )

    # Returns
    features['returns_1'] = pd.Series(close, index=df.index).pct_change().values
    features['returns_5'] = pd.Series(close, index=df.index).pct_change(5).values

    # Volume change
    features['vol_change'] = pd.Series(volume, index=df.index).pct_change().values

    return features
